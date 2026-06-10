"""Global SVD reduced observable used as the Vlachas et al. 2020 baseline.

Reproduces the dimensionality reduction of §4.1 of Vlachas, Pathak, Hunt,
Sapsis, Girvan, Ott, Koumoutsakos, *Backpropagation algorithms and Reservoir
Computing in Recurrent Neural Networks for the forecasting of complex
spatiotemporal dynamics*, Neural Networks 126 (2020) 191-217:

    "we construct observables of dimension d_o ∈ {35, 40} by performing
     Singular Value Decomposition (SVD) and keeping the most energetic d_o
     components. The 35 most energetic modes ... explain approximately 98 %
     of the total energy of the system in both F ∈ {8, 10}."

This is the *global* PCA used by the paper for RNN/RC training - it is
distinct from `chord2/local_basis.py` (per-cluster local PCA used by the
quantized ROM). One basis per dataset, saved alongside the reduced
coordinates so downstream Vlachas-style forecasters can train on `a` directly.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DEFAULT_DO_REFERENCE = 35   # Vlachas et al. 2020 §4.1
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "LOR96" / "results"
OUT_DIR  = Path(__file__).resolve().parents[2] / "results" / "LOR96"


def compute_svd_basis(u: np.ndarray) -> dict:
    """Un-centred POD-style thin SVD of the snapshot matrix `u`, shape (T, N).

    Matches the convention used by Vlachas et al. 2020 §4.1 (and the classical
    POD-Galerkin literature): SVD is applied directly to the raw snapshot
    matrix without subtracting the time-mean, so the first mode typically
    captures the mean structure. Returns the right singular vectors as columns
    of `V` (POD modes in R^N), the singular values `S`, the cumulative energy
    fraction, and the reduced coordinates `a = u @ V` such that `u ≈ a @ V.T`.
    """
    # Thin SVD: u = U_t @ diag(S) @ Vh, V is (N, N), S is (N,).
    _, S, Vh = np.linalg.svd(u, full_matrices=False)
    V = Vh.T
    a = u @ V

    energy = S ** 2
    energy_frac = np.cumsum(energy) / energy.sum()

    return {"V": V, "S": S, "energy_frac": energy_frac, "a": a}


def run(F: float, d_o: int = DEFAULT_DO_REFERENCE) -> Path:
    preset = f"vlachas_F{int(F) if float(F).is_integer() else F}"
    src = DATA_DIR / preset / "stats.npz"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found. Generate it first with "
            f"`python data/LOR96/lor96.py --preset {preset}`."
        )

    d = np.load(src, allow_pickle=True)
    u = d["u"].astype(np.float64)
    N = int(d["nx"])
    print(f"Loaded {src}  u.shape={u.shape}  N={N}  F={F}")

    res = compute_svd_basis(u)
    S, ef = res["S"], res["energy_frac"]
    print(f"Singular values:   S[0]={S[0]:.3f}  S[-1]={S[-1]:.3e}  "
          f"S[{d_o-1}]={S[d_o-1]:.3e}")
    print(f"Energy fraction at d_o={d_o:>2}:  {ef[d_o-1]*100:.3f}%   "
          f"(paper: ≈ 98%)")
    print(f"Energy fraction at d_o={N:>2}:  {ef[N-1]*100:.3f}%   (full state)")

    out_dir = OUT_DIR / f"F{int(F) if float(F).is_integer() else F}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "svd_basis.npz"
    np.savez(
        out_path,
        V=res["V"].astype(np.float64),
        S=res["S"].astype(np.float64),
        energy_frac=res["energy_frac"].astype(np.float64),
        a=res["a"].astype(np.float32),
        F=np.float64(F),
        N=np.int64(N),
        d_o_paper=np.int64(d_o),
        source_stats=np.array(str(src)),
    )
    print(f"Saved {out_path}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Global SVD reduced observable (Vlachas et al. 2020 §4.1)."
    )
    ap.add_argument("--F", type=float, choices=[8.0, 10.0], required=True,
                    help="Forcing value: 8 or 10.")
    ap.add_argument("--d_o", type=int, default=DEFAULT_DO_REFERENCE,
                    help="Reference reduced dimension to report energy at "
                         "(default 35, matching the paper).")
    args = ap.parse_args()
    run(args.F, d_o=args.d_o)
