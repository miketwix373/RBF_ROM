"""L63 alpha-dot prediction comparison: RBF-only vs poly-only vs truth.

Fits two single-block sparse-regression models on the L63 snapshots:

  * **Poly-only** -- quadratic polynomial dictionary (sigma, rho, beta
    recoverable to many decimal places). Uses `_fit_quad_only` from
    `scripts/run_phase0_l96.py`, i.e. the exact baseline that the joint
    robustness sweep's Layer 2 silence check is computed against.
  * **RBF-only**  -- flat isotropic Gaussian dictionary at the upper end
    of the n_rbf sweep (n_rbf=1600, seed=1 -- the lowest-residual cell
    in `results/LOR63/phase0_rbf_only/summary.json`).

Plots a 10-second window of (x_dot, y_dot, z_dot) showing the 5-point
stencil derivative (truth), the polynomial prediction, and the RBF
prediction overlaid. The point is to make the residual structure
visible -- the RBF-only model fits in an L^2 sense but misses the
quadratic ridges that the polynomial captures exactly.

Output: results/LOR63/phase0_rbf_only/alpha_dot_compare_10s.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chord2 import data, sindy
from scripts.run_phase0_l96 import _fit_quad_only


def _fit_rbf_only(A_mid: np.ndarray, dAdt: np.ndarray, *,
                  n_rbf: int, seed: int, lambda_rbf: float,
                  gamma: float, n_knn: int, n_init: int,
                  max_iter: int = 20):
    """Single-cell RBF-only fit, matching `_stlsq_rbf_only` in the sweep."""
    centers, widths, rbf_meta = sindy.rbf_centers_flat_isotropic(
        A_mid, n_rbf, seed=seed, gamma=gamma, n_knn=n_knn, n_init=n_init,
    )
    Phi_raw = sindy.rbf_features_iso(
        A_mid, centers, widths,
        mu_A=rbf_meta["mu_A"], sigma_A=rbf_meta["sigma_A"],
    )
    col_norms = np.linalg.norm(Phi_raw, axis=0)
    col_norms = np.where(col_norms == 0.0, 1.0, col_norms)
    Phi_n = Phi_raw / col_norms

    n_f = Phi_n.shape[1]
    active = np.ones(n_f, dtype=bool)
    prev_active = None
    xi = np.zeros((n_f, dAdt.shape[1]))
    for _ in range(max_iter):
        idx = np.where(active)[0]
        if idx.size == 0:
            break
        xi_act, *_ = np.linalg.lstsq(Phi_n[:, idx], dAdt, rcond=None)
        xi = np.zeros((n_f, dAdt.shape[1]))
        xi[idx, :] = xi_act
        new_active = active.copy()
        mags = np.max(np.abs(xi), axis=1)
        new_active[mags < lambda_rbf] = False
        if prev_active is not None and np.array_equal(new_active, active):
            break
        prev_active = active.copy()
        active = new_active
    return Phi_n @ xi  # alpha_dot prediction on the training mid-points


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n-rbf", type=int, default=1600)
    p.add_argument("--seed", type=int, default=1,
                   help="seed of the n_rbf=1600 cell with the lowest residual")
    p.add_argument("--lambda-rbf", type=float, default=1e-3)
    p.add_argument("--lambda-poly", type=float, default=1.0,
                   help="match the joint robustness sweep default")
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--n-knn", type=int, default=5)
    p.add_argument("--n-init", type=int, default=1)
    p.add_argument("--t-start", type=float, default=200.0,
                   help="left edge of the 10s window, in simulation time")
    p.add_argument("--window", type=float, default=10.0,
                   help="window length in simulation time (seconds)")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    ds = data.load("LOR63")
    A = ds.u.astype(np.float64)
    dt = float(ds.t[1] - ds.t[0])
    print(f"Loaded {ds}; dt={dt:g}, M={A.shape[0]}, T={ds.t[-1]:g}")

    out_dir = args.out_dir or (data.results_dir("LOR63") / "phase0_rbf_only")
    out_dir.mkdir(parents=True, exist_ok=True)

    A_mid, dAdt_true = sindy.deriv_5point(A, dt)
    # deriv_5point drops 2 samples on each side; the time axis for A_mid
    # is t[2:M-2].
    t_mid = ds.t[2:A.shape[0] - 2]
    print(f"5-point stencil: M_mid={A_mid.shape[0]}, t_mid in "
          f"[{t_mid[0]:.2f}, {t_mid[-1]:.2f}]")

    print("\n--- poly-only fit ---")
    poly_model = _fit_quad_only(A, dt, args.lambda_poly)
    Theta_poly = sindy.poly_features(A_mid, 2)
    dAdt_poly = Theta_poly @ poly_model.xi_poly
    rel_poly = np.linalg.norm(dAdt_true - dAdt_poly) / np.linalg.norm(dAdt_true)
    print(f"  rel_err = {rel_poly:.3e}")

    print(f"\n--- RBF-only fit (n_rbf={args.n_rbf}, seed={args.seed}) ---")
    dAdt_rbf = _fit_rbf_only(
        A_mid, dAdt_true,
        n_rbf=args.n_rbf, seed=args.seed,
        lambda_rbf=args.lambda_rbf, gamma=args.gamma,
        n_knn=args.n_knn, n_init=args.n_init,
    )
    rel_rbf = np.linalg.norm(dAdt_true - dAdt_rbf) / np.linalg.norm(dAdt_true)
    print(f"  rel_err = {rel_rbf:.3e}")

    t0 = args.t_start
    t1 = t0 + args.window
    mask = (t_mid >= t0) & (t_mid <= t1)
    if not mask.any():
        raise SystemExit(f"window [{t0}, {t1}] outside t_mid range "
                         f"[{t_mid[0]}, {t_mid[-1]}]")
    t_win = t_mid[mask]
    print(f"\nPlotting t in [{t_win[0]:.2f}, {t_win[-1]:.2f}] "
          f"({t_win.size} samples)")

    comp_labels = [r"$\dot{x}$", r"$\dot{y}$", r"$\dot{z}$"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t_win, dAdt_true[mask, i], "-", color="k", lw=1.4,
                label="truth (5-pt stencil)")
        ax.plot(t_win, dAdt_poly[mask, i], "--", color="C0", lw=1.2,
                label=f"poly-only (rel={rel_poly:.2e})")
        ax.plot(t_win, dAdt_rbf[mask, i], ":", color="C3", lw=1.2,
                label=f"RBF-only n={args.n_rbf} (rel={rel_rbf:.2e})")
        ax.set_ylabel(comp_labels[i])
        ax.grid(True, ls=":", alpha=0.4)
        if i == 0:
            ax.legend(loc="upper right", fontsize=9, ncol=3)
    axes[-1].set_xlabel("$t$ (sim time)")
    fig.suptitle(f"L63 $\\dot{{a}}$ prediction: poly-only vs RBF-only "
                 f"({t0:g}s window)", y=1.0)
    fig.tight_layout()
    out_path = out_dir / "alpha_dot_compare_10s.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")

    # Residual-only panel: the part RBF cannot represent.
    fig2, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t_win, dAdt_true[mask, i] - dAdt_poly[mask, i],
                "-", color="C0", lw=1.0,
                label=f"truth $-$ poly (rel={rel_poly:.1e})")
        ax.plot(t_win, dAdt_true[mask, i] - dAdt_rbf[mask, i],
                "-", color="C3", lw=1.0,
                label=f"truth $-$ RBF (rel={rel_rbf:.1e})")
        ax.axhline(0.0, color="k", lw=0.6, alpha=0.5)
        ax.set_ylabel(comp_labels[i] + " residual")
        ax.grid(True, ls=":", alpha=0.4)
        if i == 0:
            ax.legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("$t$ (sim time)")
    fig2.suptitle(f"L63 $\\dot{{a}}$ residuals: poly vs RBF "
                  f"({t0:g}s window)", y=1.0)
    fig2.tight_layout()
    out_path2 = out_dir / "alpha_dot_residual_10s.png"
    fig2.savefig(out_path2, dpi=180, bbox_inches="tight")
    plt.close(fig2)
    print(f"Wrote {out_path2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
