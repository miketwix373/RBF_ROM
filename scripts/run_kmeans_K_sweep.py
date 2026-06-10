"""K-means K sweep for clustering-convergence diagnostics.

For each K in [K_min, K_max], fit K-means on the snapshot matrix and report:

  (1) BIC and bic_eff (paper Eq. 18; effective-sample correction from
      `docs/notes/clustering-success-criterion.md`).
  (2) Residence-time statistics (mean/median run-length) and the
      `r2 / (K * r1)` fragmentation ratio - 1.0 means K-means is slicing a
      smooth manifold into angular sectors, >> 1 means clusters carry real
      temporal coherence.
  (3) Per-cluster PCA subspace stability across K: when going from K-1 to
      K, identify each daughter's parent by majority-snapshot vote and
      report the max principal angle (degrees) between the parent and
      daughter top-`r_compare` subspaces. Small angle means the split adds
      no new geometric direction - stop.

Parallelism: one K per worker via multiprocessing.Pool (spawn). Snapshot
matrix is shared via the worker initializer, pickled once per worker
process rather than once per task.

Outputs under `out_dir`:
  K=XX.npz       per-K shard (labels, centroids, V_pca, n_k, inertia, n_iter)
  summary.npz    K-indexed arrays for every metric
  K_sweep_summary.png   4-panel plot: BIC, residence, fragmentation, angles

The script does *not* itself pick K; it produces the diagnostics that
feed the K-selection journal entry.
"""

from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from chord2.clustering import (  # noqa: E402
    bic,
    bic_eff,
    kmeans_fit,
    residence_time_stats,
    switch_count,
)


# ---------------------------------------------------------------------------
# Diagnostics primitives
# ---------------------------------------------------------------------------

def integral_timescale(u: np.ndarray, dt: float) -> tuple[float, int]:
    """Component-mean autocorrelation integrated to first zero crossing.

    `tau_int = sum_{n=0}^{n_zero - 1} C(n*dt)` in continuous-time units.
    Returns (tau_int_seconds, n_zero_in_samples). Standard turbulence
    definition; matches the rubric in
    `docs/notes/clustering-success-criterion.md`.
    """
    M, _ = u.shape
    u_c = u - u.mean(axis=0)
    n_pow = 1
    while n_pow < 2 * M:
        n_pow *= 2
    F = np.fft.rfft(u_c, n=n_pow, axis=0)
    ac = np.fft.irfft(F * np.conj(F), n=n_pow, axis=0)[:M]
    ac_mean = ac.mean(axis=1)
    ac_mean = ac_mean / ac_mean[0]
    zero = np.flatnonzero(ac_mean <= 0.0)
    if zero.size == 0:
        return float(M * dt), M
    n_zero = int(zero[0])
    tau = float(ac_mean[:n_zero].sum())
    return float(tau * dt), max(n_zero, 1)


def per_cluster_pca(U: np.ndarray, labels: np.ndarray, K: int, r: int) -> np.ndarray:
    """Return `(K, N, r)` array of right singular vectors per cluster.

    Degenerate clusters (< r+1 members) get the first r canonical basis
    vectors so subspace_angles still has a defined argument; flagged via
    a zero in the returned `valid` mask.
    """
    N = U.shape[1]
    V = np.zeros((K, N, r), dtype=np.float64)
    for k in range(K):
        mask = labels == k
        n = int(mask.sum())
        if n < r + 1:
            V[k, :r, :r] = np.eye(r)
            continue
        X = U[mask].astype(np.float64)
        X = X - X.mean(axis=0)
        # Full-matrices=False so Vh is (min(n,N), N); we want top-r rows.
        _, _, Vh = np.linalg.svd(X, full_matrices=False)
        V[k] = Vh[:r].T
    return V


def parent_daughter_angles(
    V_parent: np.ndarray,
    V_daughter: np.ndarray,
    labels_parent: np.ndarray,
    labels_daughter: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Max principal angle (deg) between each daughter's PCA subspace and
    that of its majority-overlap parent.

    Returns `(angles_deg, parent_ids)` of shape `(K_daughter,)`.
    """
    from scipy.linalg import subspace_angles

    K_parent = V_parent.shape[0]
    K_daughter = V_daughter.shape[0]
    angles = np.full(K_daughter, np.nan, dtype=np.float64)
    parents = np.zeros(K_daughter, dtype=np.int32)
    for k_d in range(K_daughter):
        mask = labels_daughter == k_d
        if not mask.any():
            continue
        counts = np.bincount(labels_parent[mask], minlength=K_parent)
        k_p = int(counts.argmax())
        parents[k_d] = k_p
        ang_rad = subspace_angles(V_daughter[k_d], V_parent[k_p])
        angles[k_d] = float(np.degrees(ang_rad).max())
    return angles, parents


# ---------------------------------------------------------------------------
# Worker plumbing
# ---------------------------------------------------------------------------

_U_SHARED: np.ndarray | None = None


def _init_worker(U_arg: np.ndarray) -> None:
    global _U_SHARED
    _U_SHARED = U_arg


def _fit_K(payload: tuple[int, int, int]) -> dict:
    K, seed, n_init = payload
    assert _U_SHARED is not None
    labels, centroids, inertia, n_iter = kmeans_fit(
        _U_SHARED, K, seed=seed, n_init=n_init
    )
    n_k = np.bincount(labels, minlength=K).astype(np.int64)
    return {
        "K": int(K),
        "labels": np.asarray(labels, dtype=np.int32),
        "centroids": np.asarray(centroids, dtype=np.float32),
        "inertia": float(inertia),
        "n_k": n_k,
        "n_iter": int(n_iter),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_summary(out_dir: Path, summary: dict, tau_int_s: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K = summary["K"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(K, summary["bic"], "o-", label=r"BIC ($M$)")
    ax.plot(K, summary["bic_eff"], "s--", label=r"BIC ($M_{\rm eff}$)")
    ax.set_xlabel(r"$K$")
    ax.set_ylabel("BIC (smaller is better)")
    ax.set_title("(1) BIC vs $K$")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(K, summary["mean_residence_s"], "o-", label="mean residence")
    ax.plot(K, summary["median_residence_s"], "s--", label="median residence")
    ax.axhline(tau_int_s, color="k", lw=1, ls=":",
               label=fr"$\tau_{{\rm int}} = {tau_int_s:.2g}\,\mathrm{{s}}$")
    ax.set_xlabel(r"$K$")
    ax.set_ylabel("residence time (s)")
    ax.set_yscale("log")
    ax.set_title("(2) Residence time vs $K$")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    ax = axes[1, 0]
    ax.plot(K, summary["r2_over_Kr1"], "o-")
    ax.axhline(1.0, color="r", lw=1, ls=":", label="pure-slicing floor")
    ax.set_xlabel(r"$K$")
    ax.set_ylabel(r"$r_2 / (K \cdot r_1)$")
    ax.set_title("(2') Fragmentation ratio")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    ax = axes[1, 1]
    K_d = K[1:]
    ax.plot(K_d, summary["mean_max_angle_deg"][1:], "o-", label="mean over daughters")
    ax.plot(K_d, summary["max_max_angle_deg"][1:], "s--", label="max over daughters")
    ax.axhline(15.0, color="r", lw=1, ls=":", label=r"$15^{\circ}$ threshold")
    ax.set_xlabel(r"$K$ (vs $K-1$)")
    ax.set_ylabel("principal angle (deg)")
    ax.set_title(r"(3) $V_k$ stability: daughter vs parent")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(f"K-means K sweep - tau_int = {tau_int_s:.3g} s")
    fig.savefig(out_dir / "K_sweep_summary.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stats-path", type=Path, required=True,
                        help="path to a stats.npz with u (M, N) and t (M,)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--K-min", type=int, default=1)
    parser.add_argument("--K-max", type=int, default=15)
    parser.add_argument("--stride", type=int, default=10,
                        help="snapshot stride into stats.npz")
    parser.add_argument("--r-compare", type=int, default=5,
                        help="rank for parent-daughter PCA subspace-angle "
                             "comparison (fixed across K)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.stats_path}", flush=True)
    with np.load(args.stats_path) as z:
        u_full = np.asarray(z["u"], dtype=np.float32)
        dt = float(z["dtStats"])
    U = np.ascontiguousarray(u_full[::args.stride])
    dt_eff = dt * args.stride
    M, N = U.shape
    print(f"  shape={U.shape}, dt_src={dt}, stride={args.stride}, "
          f"dt_eff={dt_eff}", flush=True)

    print("[tau_int] FFT autocorr, integral to first zero", flush=True)
    tau_int_s, tau_int_steps = integral_timescale(U, dt_eff)
    print(f"  tau_int = {tau_int_s:.4g} s = {tau_int_steps} post-stride samples",
          flush=True)

    K_values = list(range(args.K_min, args.K_max + 1))
    print(f"[sweep] K in {K_values}, workers={args.workers}", flush=True)

    ctx = get_context("spawn")
    work = [(K, args.seed, args.n_init) for K in K_values]
    shards: dict[int, dict] = {}
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(U,)) as pool:
        for shard in pool.imap_unordered(_fit_K, work):
            K = shard["K"]
            shards[K] = shard
            print(f"  K={K:2d} done: inertia={shard['inertia']:.4e} "
                  f"n_iter={shard['n_iter']}", flush=True)

    print("[postproc] BIC, residence, PCA, parent-daughter angles", flush=True)
    rows: dict[str, list] = {
        "K": [], "bic": [], "bic_eff": [], "inertia": [],
        "mean_residence_s": [], "median_residence_s": [],
        "r1_steps": [], "r2_over_Kr1": [], "switches": [],
        "mean_max_angle_deg": [], "median_max_angle_deg": [],
        "max_max_angle_deg": [],
    }
    V_cache: dict[int, np.ndarray] = {}

    for K in sorted(K_values):
        shard = shards[K]
        labels = shard["labels"]
        n_k = shard["n_k"]

        b = bic(n_k, M)
        b_eff = bic_eff(n_k, M, tau_int_steps)

        res = residence_time_stats(labels)
        r1 = res["mean"]
        r2 = float(M)
        frag = r2 / (K * r1) if (K * r1) > 0 else float("inf")
        sw = switch_count(labels)

        V_K = per_cluster_pca(U, labels, K, args.r_compare)
        V_cache[K] = V_K

        if K > args.K_min:
            ang, _ = parent_daughter_angles(
                V_cache[K - 1], V_K,
                shards[K - 1]["labels"], labels,
            )
            mean_ang = float(np.nanmean(ang))
            median_ang = float(np.nanmedian(ang))
            max_ang = float(np.nanmax(ang))
        else:
            mean_ang = median_ang = max_ang = float("nan")

        np.savez(
            args.out_dir / f"K={K:02d}.npz",
            labels=labels,
            centroids=shard["centroids"],
            inertia=np.float64(shard["inertia"]),
            n_k=n_k,
            n_iter=np.int32(shard["n_iter"]),
            V_pca=V_K,
        )

        rows["K"].append(K)
        rows["bic"].append(b)
        rows["bic_eff"].append(b_eff)
        rows["inertia"].append(shard["inertia"])
        rows["mean_residence_s"].append(r1 * dt_eff)
        rows["median_residence_s"].append(res["median"] * dt_eff)
        rows["r1_steps"].append(r1)
        rows["r2_over_Kr1"].append(frag)
        rows["switches"].append(sw)
        rows["mean_max_angle_deg"].append(mean_ang)
        rows["median_max_angle_deg"].append(median_ang)
        rows["max_max_angle_deg"].append(max_ang)

        print(f"  K={K:2d}: bic={b:.3e} bic_eff={b_eff:.3e} "
              f"r1={r1*dt_eff:.3g}s frag={frag:.2g} "
              f"angle(med/max)={median_ang:.1f}/{max_ang:.1f} deg",
              flush=True)

    summary = {k: np.asarray(v) for k, v in rows.items()}
    np.savez(
        args.out_dir / "summary.npz",
        tau_int_s=np.float64(tau_int_s),
        tau_int_steps=np.int64(tau_int_steps),
        dt_eff=np.float64(dt_eff),
        M=np.int64(M),
        N=np.int64(N),
        stride=np.int32(args.stride),
        seed=np.int64(args.seed),
        r_compare=np.int32(args.r_compare),
        **summary,
    )
    plot_summary(args.out_dir, summary, tau_int_s)
    print(f"[done] outputs in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
