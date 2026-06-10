"""Dataset loading and snapshot preparation.

CHORD2 testcases live under `data/<problem>/` in one of three formats:

`scalar1d`   - one scalar field, single `stats.npz` with `u` of shape (M, N).
               Example: KS (bursting/chaotic), Lorenz 96.

`vector2d_uv` - 2D velocity components, single `stats.npz` with `u` and `v`
                of shape (M, Ny, Nx). The loader flattens and concatenates
                them so the canonical CHORD2 snapshot matrix is
                `[u_flat | v_flat]` of shape (M, 2 * Ny * Nx).
                Example: Kolmogorov flow (Re = 20, 42), Rayleigh-Benard.

`latent`     - pre-reduced data (already POD-projected upstream), stored
               as `time.npy` + `latent.h5` + a `pod/` directory. The loader
               returns the latent coordinates directly as the `u` matrix.
               Example: EKG.

The metadata schema across `stats.npz` files follows the convention
established by `data/KS/ks_solver.py`:
    nx, dx, L                  spatial grid
    dt_sim, dtStats, tInit     timing
    bc_type                    BC tag
plus problem-specific parameters (`nu`, `F`, `Re`, ...).

Large datasets (KOL42 is 31 GB) are handled via a memmap into the
uncompressed `.npy` regions of the `.npz` archive, so only the snapshots
selected by `stride` and `t_range` are materialised into RAM.
"""

from __future__ import annotations

import argparse
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
RESULTS_ROOT = REPO_ROOT / "results"


# Registry: dataset name -> (relative path under data/, kind).
# Kind drives the loader code path; see module docstring for the three kinds.
REGISTRY: dict[str, tuple[str, str]] = {
    "KS_bursting":  ("KS/results/ks_bursting/stats.npz",  "scalar1d"),
    "KS_chaotic":   ("KS/results/ks_chaotic/stats.npz",   "scalar1d"),
    "LOR63":        ("LOR63/results/lor63/stats.npz",     "scalar1d"),
    "LOR96":        ("LOR96/results/lor96/stats.npz",     "scalar1d"),
    "LOR96_high_int": ("LOR96/results/high_int/stats.npz", "scalar1d"),
    "KOL20":        ("KOL20/stats.npz",                   "vector2d_uv"),
    "KOL42":        ("KOL42/stats.npz",                   "vector2d_uv"),
    "RB":           ("RB/RB_snapshots.npz",               "vector2d_uv"),
    "EKG":          ("EKG",                               "latent"),
}


# Indicator registry: dataset name -> dotted import path to a module
# exposing `indicator(ds: Dataset) -> dict`. The contract is fixed by the
# FD-specialist (Stage 2 of the clustering build) and consumed by Stage 3
# diagnostics and Stage 5 selection. See `load_indicator` for the resolver
# and `docs/notes/clustering-success-criterion.md` (signal 1) for the
# per-testbed rationale.
#
# EKG is omitted: it loads pre-reduced latents, has no physical-field
# indicator, and is not part of the clustering-rubric sweep. The Stage 3
# diagnostics that consume indicators skip EKG silently.
INDICATOR_REGISTRY: dict[str, str] = {
    "KS_bursting": "data.KS.indicator_bursting",
    "KS_chaotic":  "data.KS.indicator_chaotic",
    "KOL20":       "data.KOL20.indicator",
    "KOL42":       "data.KOL42.indicator",
    "RB":          "data.RB.indicator",
    "LOR96":       "data.LOR96.indicator",
}


@dataclass
class Dataset:
    """A loaded snapshot set in CHORD2's canonical (M, N) form.

    `u` is the snapshot matrix; `t` are the sample times; `metadata` carries
    everything else from the source file. For vector fields the components
    are concatenated along axis 1; `metadata['components']` records the order
    and `metadata['component_size']` the per-component length so downstream
    code can split back when it needs a physical field.
    """
    name: str
    u: np.ndarray
    t: np.ndarray
    metadata: dict
    source: Path

    @property
    def M(self) -> int:
        return int(self.u.shape[0])

    @property
    def N(self) -> int:
        return int(self.u.shape[1])

    @property
    def dt(self) -> float:
        """Source simulation snapshot interval, taken from `metadata["dtStats"]`.

        This is the *source* sample step, not the post-`stride` step. After a
        strided load the actual step between consecutive entries in `t` is
        `stride * dt`; use `t[1] - t[0]` if you need the loaded sampling rate.
        """
        if "dtStats" in self.metadata:
            return float(self.metadata["dtStats"])
        return float(self.t[1] - self.t[0])

    def __repr__(self) -> str:
        return (f"Dataset(name={self.name!r}, M={self.M}, N={self.N}, "
                f"dt={self.dt:g}, T={self.t[-1] - self.t[0]:g})")


def list_datasets() -> list[str]:
    """Names of all registered datasets."""
    return sorted(REGISTRY.keys())


def load_indicator(name: str):
    """Resolve and return the `indicator` callable for dataset `name`.

    Each per-dataset indicator module exposes a top-level function

        indicator(ds: Dataset) -> dict

    that returns the fixed-schema dict documented in the clustering build
    Stage 2 spec (keys: `name`, `values`, `units`, `threshold`,
    `symmetry_group`). The modules live under `data/<name>/` and are
    imported lazily so that `chord2.data` does not pull in scipy at the
    top level for callers that only want the snapshot loader.
    """
    if name not in INDICATOR_REGISTRY:
        raise KeyError(
            f"no indicator registered for {name!r}; "
            f"available: {sorted(INDICATOR_REGISTRY.keys())}"
        )
    import importlib
    mod = importlib.import_module(INDICATOR_REGISTRY[name])
    return mod.indicator


def results_dir(name: str) -> Path:
    """Per-system output directory under `results/`. Created if missing."""
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; available: {list_datasets()}")
    d = RESULTS_ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve(name: str) -> tuple[Path, str]:
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; available: {list_datasets()}")
    rel, kind = REGISTRY[name]
    return DATA_ROOT / rel, kind


def _open_npy_in_npz_memmap(npz_path: Path, member: str) -> np.ndarray:
    """Return a `np.memmap` view into an uncompressed `.npy` stored inside `.npz`.

    Bypasses `np.load` for large arrays where eager loading would OOM
    (KOL42 carries 31 GB of snapshots). Requires the `.npz` to have been
    written with `np.savez` (ZIP_STORED), not `np.savez_compressed`.

    The trick is to compute the byte offset of the `.npy` member's data
    inside the outer `.npz` zip archive, then point `np.memmap` at that
    offset with the dtype and shape parsed from the `.npy` header.
    """
    with zipfile.ZipFile(npz_path) as zf:
        info = zf.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            raise RuntimeError(
                f"{npz_path}::{member} is compressed; memmap slicing not supported. "
                f"Re-save with np.savez (not np.savez_compressed)."
            )

    with open(npz_path, "rb") as fh:
        fh.seek(info.header_offset)
        local_header = fh.read(30)
        if local_header[:4] != b"PK\x03\x04":
            raise RuntimeError(f"malformed local file header at {info.header_offset}")
        name_len  = int.from_bytes(local_header[26:28], "little")
        extra_len = int.from_bytes(local_header[28:30], "little")
        data_start = info.header_offset + 30 + name_len + extra_len

        fh.seek(data_start)
        version = np.lib.format.read_magic(fh)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(fh)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(fh)
        else:
            raise RuntimeError(f"unsupported .npy version {version}")
        array_offset = fh.tell()

    return np.memmap(npz_path, mode="r", dtype=dtype,
                     shape=shape, offset=array_offset,
                     order="F" if fortran else "C")


def _metadata_from_npz(npz_path: Path, skip_keys: set[str]) -> dict:
    """Read scalar and small-array metadata from a `.npz`, skipping `skip_keys`."""
    md: dict = {}
    with np.load(npz_path, allow_pickle=False) as z:
        for k in z.files:
            if k in skip_keys:
                continue
            v = z[k]
            if v.ndim == 0:
                md[k] = v.item() if v.dtype.kind not in ("O", "U") else str(v)
            else:
                md[k] = np.asarray(v)
    return md


def _select(t: np.ndarray, stride: int,
            t_range: Optional[Tuple[float, float]]) -> np.ndarray:
    """Snapshot index selector for `stride` and `t_range`."""
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if t_range is None:
        return np.arange(0, t.shape[0], stride)
    lo, hi = t_range
    idx = np.flatnonzero((t >= lo) & (t <= hi))
    return idx[::stride]


def load(name: str, *,
         stride: int = 1,
         t_range: Optional[Tuple[float, float]] = None,
         ) -> Dataset:
    """Load a registered dataset as a CHORD2 `Dataset`.

    Parameters
    ----------
    name
        Registry key; see `list_datasets()`.
    stride
        Take every `stride`-th snapshot after filtering by `t_range`.
        Default 1. The large 2D datasets (KOL42 at 31 GB, RB at 5.7 GB)
        will typically need `stride >= 10` to stay within RAM.
    t_range
        Optional `(t_lo, t_hi)` window; only snapshots with
        `t_lo <= t <= t_hi` are kept.
    """
    src, kind = _resolve(name)

    if kind == "latent":
        return _load_latent(name, src, stride=stride, t_range=t_range)

    if kind == "scalar1d":
        with np.load(src) as z:
            t = np.asarray(z["t"])
            sel = _select(t, stride, t_range)
            u = np.asarray(z["u"])[sel]
            t = t[sel]
        md = _metadata_from_npz(src, skip_keys={"u", "t"})
        md["components"] = ("u",)
        md["component_size"] = u.shape[1]

    elif kind == "vector2d_uv":
        # Prefer the memmap path so KOL42 (31 GB) stays lazy. Fall back to
        # eager `np.load` if the .npz was written compressed (RB), in which
        # case slicing happens after decompression - still correct, just
        # heavier.
        try:
            u_mm = _open_npy_in_npz_memmap(src, "u.npy")
            v_mm = _open_npy_in_npz_memmap(src, "v.npy")
            eager_fallback = False
        except RuntimeError as e:
            import sys
            print(f"data.load[{name}]: memmap unavailable ({e}); "
                  f"falling back to eager load", file=sys.stderr)
            u_mm = v_mm = None
            eager_fallback = True

        with np.load(src) as z:
            t = np.asarray(z["t"])
            sel = _select(t, stride, t_range)
            if eager_fallback:
                u_sel = np.asarray(z["u"])[sel]
                v_sel = np.asarray(z["v"])[sel]
            else:
                u_sel = np.asarray(u_mm[sel])
                v_sel = np.asarray(v_mm[sel])
        t = t[sel]
        M = u_sel.shape[0]
        u = np.concatenate([u_sel.reshape(M, -1),
                            v_sel.reshape(M, -1)], axis=1)
        md = _metadata_from_npz(src, skip_keys={"u", "v", "t", "dissipation"})
        md["components"] = ("u", "v")
        md["component_size"] = int(u_sel.shape[1] * u_sel.shape[2])

    else:
        raise RuntimeError(f"unsupported kind {kind!r}")

    _warn_if_non_monotonic(name, t)
    return Dataset(name=name, u=u, t=t, metadata=md, source=src)


def _warn_if_non_monotonic(name: str, t: np.ndarray) -> None:
    """Print a stderr warning if `t` is not strictly increasing.

    Downstream code that assumes monotonic time - `temporal_split`,
    finite-difference derivative estimators - will misbehave on
    non-monotonic data. The RB dataset is currently known to have a
    single jump where two runs appear concatenated out of order.
    """
    if t.size < 2:
        return
    dt = np.diff(t)
    bad = np.flatnonzero(dt <= 0)
    if bad.size == 0:
        return
    import sys
    i = int(bad[0])
    print(f"data.load[{name}]: WARNING t is not strictly increasing - "
          f"{bad.size} non-positive step(s); first at index {i} "
          f"(t[{i}]={t[i]:g} -> t[{i+1}]={t[i+1]:g}). "
          f"temporal_split and FD derivatives assume monotonic t.",
          file=sys.stderr)


def _load_latent(name: str, src: Path, *, stride: int,
                 t_range: Optional[Tuple[float, float]]) -> Dataset:
    """Load EKG-style pre-reduced data (`time.npy` + `latent.h5`).

    Downstream code that wants the physical field must reapply the POD
    basis from `<src>/pod/` (PyTorch tensors); CHORD2 itself operates on
    the latent and never touches the physical field.
    """
    import h5py  # local import: EKG is an optional dataset

    t = np.load(src / "time.npy").astype(np.float64)
    sel = _select(t, stride, t_range)
    with h5py.File(src / "latent.h5", "r") as f:
        latent = np.asarray(f["latent"][sel])
    md = {
        "components": ("latent",),
        "component_size": int(latent.shape[1]),
        "bc_type": "latent",
        "pod_dir": str(src / "pod"),
    }
    return Dataset(name=name, u=latent, t=t[sel], metadata=md, source=src)


def temporal_split(d: Dataset, train_fraction: float = 0.9
                   ) -> Tuple[Dataset, Dataset]:
    """Contiguous trailing-window split for short-term prediction evaluation.

    Returns `(train, test)` where the first `train_fraction` of snapshots are
    training and the remainder is held out. The paper's Section 4 takes the
    test window from the end of the record so that ROM prediction starts
    from a state not seen as a training snapshot.

    Assumes `d.t` is monotonically increasing - which `load` warns about if
    violated. The split is by snapshot index, not by time; on a non-monotonic
    record the trailing window is not the latest-in-time window.
    """
    if not 0 < train_fraction < 1:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    m_split = int(d.M * train_fraction)
    train = Dataset(name=f"{d.name}_train",
                    u=d.u[:m_split], t=d.t[:m_split],
                    metadata=d.metadata, source=d.source)
    test = Dataset(name=f"{d.name}_test",
                   u=d.u[m_split:], t=d.t[m_split:],
                   metadata=d.metadata, source=d.source)
    return train, test


def _inspect(name: str, stride: int) -> None:
    d = load(name, stride=stride)
    print(d)
    print(f"  source      : {d.source}")
    print(f"  components  : {d.metadata.get('components')} "
          f"(size {d.metadata.get('component_size')})")
    print(f"  bc_type     : {d.metadata.get('bc_type')}")
    print(f"  u stats     : mean={d.u.mean():.3e}  std={d.u.std():.3e}  "
          f"min={d.u.min():.3e}  max={d.u.max():.3e}")
    print(f"  t range     : [{d.t[0]:g}, {d.t[-1]:g}]  dt={d.dt:g}")
    extras = {k: v for k, v in d.metadata.items()
              if k not in ("components", "component_size", "bc_type", "pod_dir")
              and not isinstance(v, np.ndarray)}
    if extras:
        print(f"  metadata    : {extras}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CHORD2 dataset registry / inspector.")
    parser.add_argument("--list", action="store_true",
                        help="list registered datasets and exit")
    parser.add_argument("--inspect", metavar="NAME",
                        help="load and print a summary of NAME")
    parser.add_argument("--stride", type=int, default=1,
                        help="snapshot stride for --inspect (default 1; "
                             "use >>1 for KOL42 and RB to limit RAM use)")
    args = parser.parse_args()

    if args.list:
        for n in list_datasets():
            src, kind = _resolve(n)
            exists = src.exists() if kind != "latent" else src.is_dir()
            tag = "OK     " if exists else "MISSING"
            size = ""
            if exists and kind != "latent":
                size = f"  {src.stat().st_size / 1e9:.2f} GB"
            print(f"  {n:14s}  {kind:14s}  {tag}{size}")
        return

    if args.inspect:
        _inspect(args.inspect, stride=args.stride)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
