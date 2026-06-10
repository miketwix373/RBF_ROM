"""L63 RBF-only integration experiment (rom-specialist plan).

Steps 1-5 of the consultant's revised plan:

  1. Short-window pointwise [0, 5] s for IC 0 (3 panels: x, y, z),
     scored by Eq. 21 prediction horizon at tau = 0.5 * X_std_climate
     -- median over the 8-IC ensemble, reported in seconds and Lyapunov
     times (lambda_L63 = 0.906).
  2. Long-window attractor projection [10, 50] s as an ensemble of 8 ICs
     drawn from the FOM trajectory. Two 3D panels side by side (poly,
     RBF) with the FOM butterfly overlaid in black.
  3. Marginal PDFs of x, y, z over [10, 50] s pooled across ICs +
     Wasserstein-1 per component (scipy.stats.wasserstein_distance).
  4. Vector-field residual ||f_truth - f_*|| vs distance from nearest
     K-means centre (standardised), on 10k FOM-cloud samples.
  5. Symmetry diagnostic: L63 has sigma_op = diag(-1, -1, 1).
     Compute ||f_rbf(a) - sigma_op * f_rbf(sigma_op * a)|| on the cloud.

Also: a dt = 0.0025 confirmation pass on IC 0 for each model, asserting
pointwise agreement to <= 1e-6 over [0, 2] s (RK4 step-halving sanity).

Inputs: a single (n_rbf, seed) RBF-only fit + the canonical poly-only
fit, both on the LOR63 dataset. Defaults match the lowest-residual cell
in `results/LOR63/phase0_rbf_only/summary.json` (n_rbf=1600, seed=1).

Outputs (results/LOR63/integration_experiment/):
  short_window_ic0.png
  long_window_3d_ensemble.png
  marginal_pdfs.png
  vector_field_residual.png
  symmetry_diagnostic.png
  step_halving_sanity.png
  metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _l63_rbf_lib import (  # noqa: E402
    f_poly, f_rbf, f_truth_l63, f_truth_l63_single,
    fit_rbf_only_keep_state, rk4_integrate,
)

from chord2 import data, sindy  # noqa: E402
from scripts.run_phase0_l96 import _fit_quad_only  # noqa: E402


LYAPUNOV_L63 = 0.906       # Wolf, Swift, Swinney, Vastano (1985)
T_LYAPUNOV = 1.0 / LYAPUNOV_L63
HORIZON_FRAC = 0.5         # matches `_FORECAST_HORIZON_FRAC` in scripts/run_phase0_l96.py


def _make_poly_callable(xi_poly: np.ndarray):
    def f(a: np.ndarray) -> np.ndarray:
        return f_poly(a[None, :], xi_poly)[0]
    return f


def _make_rbf_callable(rbf: dict):
    def f(a: np.ndarray) -> np.ndarray:
        return f_rbf(a[None, :], rbf["centers"], rbf["widths"],
                     rbf["mu_A"], rbf["sigma_A"],
                     rbf["col_norms"], rbf["xi"])[0]
    return f


def _integrate_fom_truth(x0: np.ndarray, dt: float, n_steps: int) -> np.ndarray:
    """RK4 integration with the analytic L63 RHS, returns shape (n_steps+1, 3)."""
    X = np.empty((n_steps + 1, 3))
    X[0] = x0
    for i in range(n_steps):
        a = X[i]
        k1 = f_truth_l63_single(a)
        k2 = f_truth_l63_single(a + 0.5 * dt * k1)
        k3 = f_truth_l63_single(a + 0.5 * dt * k2)
        k4 = f_truth_l63_single(a + dt * k3)
        X[i + 1] = a + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return X


def _eq21_horizon(X_pred: np.ndarray, X_true: np.ndarray, dt: float,
                  X_std_climate: float) -> float:
    """Eq. 21 horizon: first t where rmse_t > HORIZON_FRAC * X_std_climate.

    Returns t in seconds, capped at the trajectory length (X_pred can be
    shorter than X_true if the ROM diverged to non-finite).
    """
    n_cmp = min(X_pred.shape[0], X_true.shape[0])
    err = X_pred[:n_cmp] - X_true[:n_cmp]
    rmse_t = np.sqrt((err * err).mean(axis=1))
    threshold = HORIZON_FRAC * X_std_climate
    crossed = np.flatnonzero(rmse_t > threshold)
    if crossed.size == 0:
        return float(dt * (n_cmp - 1))
    return float(crossed[0] * dt)


def _pool_tail(X_list: list, n_keep_per: int) -> np.ndarray:
    """Stack the last `n_keep_per` rows of each (T, 3) traj, dropping non-finite."""
    parts = []
    for X in X_list:
        if X.shape[0] >= n_keep_per:
            tail = X[-n_keep_per:]
        else:
            tail = X
        finite = np.all(np.isfinite(tail), axis=1)
        if finite.any():
            parts.append(tail[finite])
    if not parts:
        return np.zeros((0, 3))
    return np.concatenate(parts, axis=0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n-rbf", type=int, default=1600)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--lambda-rbf", type=float, default=1e-3)
    p.add_argument("--lambda-poly", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--n-knn", type=int, default=5)
    p.add_argument("--n-init", type=int, default=1)
    p.add_argument("--n-ic", type=int, default=8)
    p.add_argument("--ic-rng-seed", type=int, default=0)
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--T", type=float, default=50.0)
    p.add_argument("--short-window-T", type=float, default=5.0)
    p.add_argument("--long-window-t0", type=float, default=10.0)
    p.add_argument("--n-cloud-residual", type=int, default=10_000)
    p.add_argument("--n-cloud-symmetry", type=int, default=10_000)
    p.add_argument("--cloud-rng-seed", type=int, default=1)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    ds = data.load("LOR63")
    A = ds.u.astype(np.float64)
    dt_data = float(ds.t[1] - ds.t[0])
    print(f"Loaded {ds}; dt_data={dt_data:g}, M={A.shape[0]}, T_data={ds.t[-1]:g}")

    out_dir = args.out_dir or (data.results_dir("LOR63") / "integration_experiment")
    out_dir.mkdir(parents=True, exist_ok=True)

    A_mid, dAdt_true = sindy.deriv_5point(A, dt_data)

    print("\n--- poly-only fit ---")
    poly_model = _fit_quad_only(A, dt_data, args.lambda_poly)
    rel_poly = (np.linalg.norm(dAdt_true - sindy.poly_features(A_mid, 2) @ poly_model.xi_poly)
                / np.linalg.norm(dAdt_true))
    print(f"  rel_err = {rel_poly:.3e}")

    print(f"\n--- RBF-only fit (n_rbf={args.n_rbf}, seed={args.seed}) ---")
    rbf = fit_rbf_only_keep_state(
        A_mid, dAdt_true,
        n_rbf=args.n_rbf, seed=args.seed, lambda_rbf=args.lambda_rbf,
        gamma=args.gamma, n_knn=args.n_knn, n_init=args.n_init,
    )
    Phi_n_train = sindy.rbf_features_iso(
        A_mid, rbf["centers"], rbf["widths"],
        mu_A=rbf["mu_A"], sigma_A=rbf["sigma_A"],
    ) / rbf["col_norms"]
    rel_rbf = (np.linalg.norm(dAdt_true - Phi_n_train @ rbf["xi"])
               / np.linalg.norm(dAdt_true))
    print(f"  rel_err = {rel_rbf:.3e}, active = {rbf['nnz']}/{args.n_rbf}")

    f_poly_cb = _make_poly_callable(poly_model.xi_poly)
    f_rbf_cb = _make_rbf_callable(rbf)

    X_std_climate = float(A.std(axis=0).mean())
    print(f"\nX_std_climate (per-axis std, then mean) = {X_std_climate:.4f}")
    print(f"Lyapunov time = 1/{LYAPUNOV_L63} = {T_LYAPUNOV:.3f} s")

    # -- Initial conditions ---------------------------------------------------
    rng_ic = np.random.default_rng(args.ic_rng_seed)
    ic_idx = rng_ic.choice(A.shape[0], size=args.n_ic, replace=False)
    ic_idx.sort()
    ICs = A[ic_idx]
    n_steps = int(round(args.T / args.dt))
    print(f"\nIntegrating {args.n_ic} ICs at dt={args.dt}, T={args.T} "
          f"({n_steps} steps).")

    # -- Integrate ROMs + truth -----------------------------------------------
    Xs_poly = []
    Xs_rbf = []
    Xs_true = []
    horizons_poly = []
    horizons_rbf = []
    for i, x0 in enumerate(ICs):
        X_true = _integrate_fom_truth(x0, args.dt, n_steps)
        _, X_poly = rk4_integrate(f_poly_cb, x0, args.dt, n_steps)
        _, X_rbf = rk4_integrate(f_rbf_cb, x0, args.dt, n_steps)
        Xs_true.append(X_true)
        Xs_poly.append(X_poly)
        Xs_rbf.append(X_rbf)
        h_p = _eq21_horizon(X_poly, X_true, args.dt, X_std_climate)
        h_r = _eq21_horizon(X_rbf, X_true, args.dt, X_std_climate)
        horizons_poly.append(h_p)
        horizons_rbf.append(h_r)
        print(f"  IC {i}: poly_horizon={h_p:.2f}s ({h_p/T_LYAPUNOV:.2f} t_L), "
              f"rbf_horizon={h_r:.2f}s ({h_r/T_LYAPUNOV:.2f} t_L), "
              f"n_poly={X_poly.shape[0]}, n_rbf={X_rbf.shape[0]}")

    h_p_med = float(np.median(horizons_poly))
    h_r_med = float(np.median(horizons_rbf))
    print(f"\nEq.21 horizon median:  poly={h_p_med:.2f}s ({h_p_med/T_LYAPUNOV:.2f} t_L) "
          f"|  rbf={h_r_med:.2f}s ({h_r_med/T_LYAPUNOV:.2f} t_L)")

    # -- (A) Short-window IC 0 pointwise --------------------------------------
    n_short = int(round(args.short_window_T / args.dt))
    t_short = np.arange(n_short + 1) * args.dt
    X_true_s = Xs_true[0][: n_short + 1]
    X_poly_s = Xs_poly[0][: min(n_short + 1, Xs_poly[0].shape[0])]
    X_rbf_s = Xs_rbf[0][: min(n_short + 1, Xs_rbf[0].shape[0])]
    labels = [r"$x(t)$", r"$y(t)$", r"$z(t)$"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    for k, ax in enumerate(axes):
        ax.plot(t_short, X_true_s[:, k], "-", color="k", lw=1.4, label="FOM (truth)")
        ax.plot(t_short[: X_poly_s.shape[0]], X_poly_s[:, k], "--", color="C0", lw=1.2,
                label=f"poly-only ($T_{{ph}}^{{IC0}}$={horizons_poly[0]:.2f}s)")
        ax.plot(t_short[: X_rbf_s.shape[0]], X_rbf_s[:, k], ":", color="C3", lw=1.4,
                label=f"RBF-only ($T_{{ph}}^{{IC0}}$={horizons_rbf[0]:.2f}s)")
        ax.set_ylabel(labels[k])
        ax.grid(True, ls=":", alpha=0.4)
        if k == 0:
            ax.legend(loc="upper right", fontsize=9, ncol=3)
    axes[-1].set_xlabel(r"$t$ (s)")
    fig.suptitle(f"L63 short-window trajectory, IC 0 (median $T_{{ph}}$: "
                 f"poly={h_p_med:.2f}s, RBF={h_r_med:.2f}s)", y=1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "short_window_ic0.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'short_window_ic0.png'}")

    # -- (B) Long-window 3D ensemble -----------------------------------------
    n_lo = int(round(args.long_window_t0 / args.dt))
    fig = plt.figure(figsize=(15, 6.5))
    for col, (lbl, Xs) in enumerate([("poly-only", Xs_poly), ("RBF-only", Xs_rbf)]):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        # FOM background
        X_true_long = Xs_true[0][n_lo:]
        ax.plot(X_true_long[:, 0], X_true_long[:, 1], X_true_long[:, 2],
                lw=0.3, color="k", alpha=0.45, label="FOM")
        for i, X in enumerate(Xs):
            if X.shape[0] <= n_lo:
                continue
            Xl = X[n_lo:]
            finite = np.all(np.isfinite(Xl), axis=1)
            Xl = Xl[finite]
            if Xl.shape[0] < 5:
                continue
            ax.plot(Xl[:, 0], Xl[:, 1], Xl[:, 2], lw=0.5,
                    color=plt.cm.tab10(i % 10), alpha=0.65)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_zlabel("$z$")
        ax.set_title(f"{lbl}: 8-IC ensemble  t in [{args.long_window_t0:g}, {args.T:g}] s")
    fig.suptitle("L63 long-window attractor projection", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "long_window_3d_ensemble.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'long_window_3d_ensemble.png'}")

    # -- (C) Marginal PDFs + Wasserstein-1 -----------------------------------
    n_tail = int(round((args.T - args.long_window_t0) / args.dt))
    pool_true = _pool_tail(Xs_true, n_tail)
    pool_poly = _pool_tail(Xs_poly, n_tail)
    pool_rbf = _pool_tail(Xs_rbf, n_tail)
    print(f"\nPooled tail samples: true={pool_true.shape[0]}, "
          f"poly={pool_poly.shape[0]}, rbf={pool_rbf.shape[0]}")

    W1 = {"poly": [], "rbf": []}
    for k in range(3):
        if pool_poly.shape[0] > 1:
            W1["poly"].append(float(wasserstein_distance(pool_true[:, k], pool_poly[:, k])))
        else:
            W1["poly"].append(float("nan"))
        if pool_rbf.shape[0] > 1:
            W1["rbf"].append(float(wasserstein_distance(pool_true[:, k], pool_rbf[:, k])))
        else:
            W1["rbf"].append(float("nan"))
    print(f"Wasserstein-1 (x, y, z):  poly={W1['poly']}, rbf={W1['rbf']}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=False)
    comp = ["x", "y", "z"]
    for k, ax in enumerate(axes):
        all_vals = np.concatenate([pool_true[:, k],
                                   pool_poly[:, k] if pool_poly.size else pool_true[:, k],
                                   pool_rbf[:, k] if pool_rbf.size else pool_true[:, k]])
        bins = np.linspace(np.min(all_vals), np.max(all_vals), 80)
        ax.hist(pool_true[:, k], bins=bins, density=True, alpha=0.55,
                color="k", label="FOM")
        if pool_poly.size:
            ax.hist(pool_poly[:, k], bins=bins, density=True, alpha=0.45,
                    color="C0",
                    label=f"poly (W1={W1['poly'][k]:.2f})")
        if pool_rbf.size:
            ax.hist(pool_rbf[:, k], bins=bins, density=True, alpha=0.45,
                    color="C3",
                    label=f"RBF (W1={W1['rbf'][k]:.2f})")
        ax.set_xlabel(f"${comp[k]}$")
        ax.set_title(f"marginal PDF of ${comp[k]}$  "
                     f"(t in [{args.long_window_t0:g}, {args.T:g}] s)")
        ax.legend(fontsize=9)
        ax.grid(True, ls=":", alpha=0.4)
    axes[0].set_ylabel("density")
    fig.tight_layout()
    fig.savefig(out_dir / "marginal_pdfs.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'marginal_pdfs.png'}")

    # -- (D) Vector-field residual vs distance from nearest centre -----------
    rng_v = np.random.default_rng(args.cloud_rng_seed)
    n_cv = min(args.n_cloud_residual, A.shape[0])
    A_cloud = A[rng_v.choice(A.shape[0], size=n_cv, replace=False)]
    f_t = f_truth_l63(A_cloud)
    f_p = f_poly(A_cloud, poly_model.xi_poly)
    f_r = f_rbf(A_cloud, rbf["centers"], rbf["widths"],
                rbf["mu_A"], rbf["sigma_A"], rbf["col_norms"], rbf["xi"])
    res_p = np.linalg.norm(f_t - f_p, axis=1)
    res_r = np.linalg.norm(f_t - f_r, axis=1)

    # distance from nearest centre, standardised coords
    centers_std = (rbf["centers"] - rbf["mu_A"]) / rbf["sigma_A"]
    A_std = (A_cloud - rbf["mu_A"]) / rbf["sigma_A"]
    d2 = ((A_std[:, None, :] - centers_std[None, :, :]) ** 2).sum(-1)
    d_nn = np.sqrt(d2.min(axis=1))

    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    ax.scatter(d_nn, res_p, s=4, alpha=0.35, color="C0", label="$\\Vert f_{truth}-f_{poly} \\Vert$")
    ax.scatter(d_nn, res_r, s=4, alpha=0.35, color="C3", label="$\\Vert f_{truth}-f_{rbf} \\Vert$")
    ax.set_xlabel("distance to nearest K-means centre (standardised coords)")
    ax.set_ylabel("vector-field residual")
    ax.set_yscale("log")
    ax.grid(True, ls=":", alpha=0.4, which="both")
    ax.legend(fontsize=10)
    ax.set_title(f"Vector-field residual on FOM cloud  "
                 f"(median: poly={np.median(res_p):.3e}, rbf={np.median(res_r):.3e})")
    fig.tight_layout()
    fig.savefig(out_dir / "vector_field_residual.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'vector_field_residual.png'}")

    # -- (E) Symmetry diagnostic ---------------------------------------------
    n_cs = min(args.n_cloud_symmetry, A.shape[0])
    A_cs = A[rng_v.choice(A.shape[0], size=n_cs, replace=False)]
    sigma_op = np.array([-1.0, -1.0, 1.0])
    f_rbf_a = f_rbf(A_cs, rbf["centers"], rbf["widths"],
                    rbf["mu_A"], rbf["sigma_A"], rbf["col_norms"], rbf["xi"])
    f_rbf_sa = f_rbf(A_cs * sigma_op[None, :],
                     rbf["centers"], rbf["widths"],
                     rbf["mu_A"], rbf["sigma_A"],
                     rbf["col_norms"], rbf["xi"])
    sym_resid_rbf = np.linalg.norm(f_rbf_a - sigma_op[None, :] * f_rbf_sa, axis=1)
    norm_rbf_a = np.linalg.norm(f_rbf_a, axis=1)
    sym_ratio = sym_resid_rbf / np.maximum(norm_rbf_a, 1e-30)

    # truth control: should be 0 exactly
    f_t_a = f_truth_l63(A_cs)
    f_t_sa = f_truth_l63(A_cs * sigma_op[None, :])
    sym_resid_truth = np.linalg.norm(f_t_a - sigma_op[None, :] * f_t_sa, axis=1)
    sym_truth_ratio = (sym_resid_truth / np.maximum(np.linalg.norm(f_t_a, axis=1), 1e-30))

    print(f"\nSymmetry residual ||f_rbf(a) - sigma f_rbf(sigma a)|| / ||f_rbf(a)||:")
    print(f"  RBF:   median={np.median(sym_ratio):.3e}, p90={np.quantile(sym_ratio, 0.9):.3e}")
    print(f"  truth: median={np.median(sym_truth_ratio):.3e}  (expected ~ machine epsilon)")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    bins = np.logspace(-15, 1, 80)
    ax.hist(np.maximum(sym_truth_ratio, 1e-15), bins=bins, alpha=0.55, color="k",
            density=True, label="truth (analytic L63)")
    ax.hist(np.maximum(sym_ratio, 1e-15), bins=bins, alpha=0.55, color="C3",
            density=True, label="RBF-only")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\Vert f(a) - \sigma f(\sigma a) \Vert / \Vert f(a) \Vert$")
    ax.set_ylabel("density")
    ax.set_title(r"L63 $(x,y,z) \to (-x,-y,z)$ symmetry residual on FOM cloud")
    ax.legend(fontsize=10)
    ax.grid(True, ls=":", alpha=0.4, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "symmetry_diagnostic.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'symmetry_diagnostic.png'}")

    # -- (F) Step-halving sanity for IC 0 ------------------------------------
    n_check = int(round(2.0 / args.dt))  # check over [0, 2] s
    n_steps_half = 2 * n_check
    dt_half = args.dt / 2.0
    _, X_poly_half = rk4_integrate(f_poly_cb, ICs[0], dt_half, n_steps_half)
    _, X_rbf_half = rk4_integrate(f_rbf_cb, ICs[0], dt_half, n_steps_half)
    X_poly_check = Xs_poly[0][: n_check + 1]
    X_rbf_check = Xs_rbf[0][: n_check + 1]
    if X_poly_half.shape[0] >= 2 * n_check + 1 and X_poly_check.shape[0] >= n_check + 1:
        diff_poly = np.linalg.norm(X_poly_half[::2][: n_check + 1] - X_poly_check, axis=1)
    else:
        diff_poly = np.array([np.nan])
    if X_rbf_half.shape[0] >= 2 * n_check + 1 and X_rbf_check.shape[0] >= n_check + 1:
        diff_rbf = np.linalg.norm(X_rbf_half[::2][: n_check + 1] - X_rbf_check, axis=1)
    else:
        diff_rbf = np.array([np.nan])
    print(f"\nStep-halving max diff over [0, 2] s, IC 0:  "
          f"poly={np.nanmax(diff_poly):.3e}, rbf={np.nanmax(diff_rbf):.3e}")
    t_check = np.arange(diff_poly.size) * args.dt
    t_check_r = np.arange(diff_rbf.size) * args.dt
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    ax.semilogy(t_check, np.maximum(diff_poly, 1e-16), "-", color="C0",
                label=f"poly  max={np.nanmax(diff_poly):.2e}")
    ax.semilogy(t_check_r, np.maximum(diff_rbf, 1e-16), "-", color="C3",
                label=f"RBF   max={np.nanmax(diff_rbf):.2e}")
    ax.set_xlabel("$t$ (s)")
    ax.set_ylabel(r"$\Vert X_{dt} - X_{dt/2} \Vert$")
    ax.set_title("RK4 step-halving sanity, IC 0  ($\\Delta t$={:g} vs {:g})".format(
        args.dt, dt_half,
    ))
    ax.grid(True, ls=":", alpha=0.4, which="both")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "step_halving_sanity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'step_halving_sanity.png'}")

    # -- Save metrics ---------------------------------------------------------
    metrics = {
        "config": {
            "n_rbf": args.n_rbf, "seed": args.seed,
            "lambda_rbf": args.lambda_rbf, "lambda_poly": args.lambda_poly,
            "gamma": args.gamma, "n_knn": args.n_knn, "n_init": args.n_init,
            "n_ic": args.n_ic, "ic_rng_seed": args.ic_rng_seed,
            "dt": args.dt, "T": args.T,
            "rel_err_poly": float(rel_poly), "rel_err_rbf": float(rel_rbf),
            "X_std_climate": X_std_climate,
            "lyapunov_l63": LYAPUNOV_L63,
            "horizon_frac": HORIZON_FRAC,
        },
        "horizons_seconds": {
            "poly": list(map(float, horizons_poly)),
            "rbf": list(map(float, horizons_rbf)),
            "poly_median": h_p_med, "rbf_median": h_r_med,
            "poly_median_lyap": h_p_med / T_LYAPUNOV,
            "rbf_median_lyap": h_r_med / T_LYAPUNOV,
        },
        "wasserstein_1": W1,
        "vector_field_residual_cloud": {
            "poly_median": float(np.median(res_p)),
            "poly_p90": float(np.quantile(res_p, 0.9)),
            "rbf_median": float(np.median(res_r)),
            "rbf_p90": float(np.quantile(res_r, 0.9)),
        },
        "symmetry_residual_relative": {
            "rbf_median": float(np.median(sym_ratio)),
            "rbf_p90": float(np.quantile(sym_ratio, 0.9)),
            "truth_median": float(np.median(sym_truth_ratio)),
        },
        "step_halving_max_diff_2s": {
            "poly": float(np.nanmax(diff_poly)),
            "rbf": float(np.nanmax(diff_rbf)),
        },
    }
    out_json = out_dir / "metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2))
    print(f"\nWrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
