"""L63 RBF-only step 1: K=1 isotropic n_rbf sweep on forward integration.

For each n_rbf in a wide grid, fit a single RBF-only model (seed=1,
lambda_rbf=1e-3, lambda_poly N/A -- no poly block), integrate 8 ICs
RK4 to T=50 s, and record:

  * alpha_dot relative error on the training mid-points (the static-fit
    residual we already swept in `phase0_rbf_only/summary.json`),
  * Eq.21 prediction horizon at HORIZON_FRAC = 0.5 * X_std_climate
    (mirrors `_FORECAST_HORIZON_FRAC` in scripts/run_phase0_l96.py),
  * finite-at-T per IC (the stability floor diagnostic -- the smallest
    n_rbf at which 8/8 ICs reach T=50 s without divergence),
  * Wasserstein-1 marginal climate distance on (x, y, z) over the
    [10, 50] s tail pooled across surviving ICs.

The headline test is whether the static-fit residual `rel_err_alpha_dot`
predicts the dynamic horizon -- if so, the cheap alpha_dot sweep is a
useful proxy for forecast quality on later testbeds; if not, the
forward integration is the only honest metric and we learn it now on
L63 where a poly baseline pins ground truth.

Grid extends to n_rbf = 6400 to locate any over-fitting plateau (the
existing sweep stopped at 1600). All ICs are the same across n_rbf
values so the per-IC comparison is controlled.

Outputs (results/LOR63/step1_nrbf_sweep/):
  step1_nrbf_sweep.json
  horizon_vs_nrbf.png
  horizon_vs_alpha_dot_residual.png
  w1_vs_nrbf.png
  stability_floor.png
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _l63_rbf_lib import (  # noqa: E402
    f_rbf, f_truth_l63_single, fit_rbf_only_keep_state, rk4_integrate,
)

from chord2 import data, sindy  # noqa: E402
from scripts.run_phase0_l96 import _fit_quad_only  # noqa: E402


LYAPUNOV_L63 = 0.906
T_LYAPUNOV = 1.0 / LYAPUNOV_L63
HORIZON_FRAC = 0.5


def _integrate_fom_truth(x0: np.ndarray, dt: float, n_steps: int) -> np.ndarray:
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
    n_cmp = min(X_pred.shape[0], X_true.shape[0])
    err = X_pred[:n_cmp] - X_true[:n_cmp]
    rmse_t = np.sqrt((err * err).mean(axis=1))
    threshold = HORIZON_FRAC * X_std_climate
    crossed = np.flatnonzero(rmse_t > threshold)
    if crossed.size == 0:
        return float(dt * (n_cmp - 1))
    return float(crossed[0] * dt)


_WORKER_STATE: dict = {}


def _worker_init(A_mid: np.ndarray, dAdt_true: np.ndarray,
                 ICs: np.ndarray, X_true_all: np.ndarray,
                 X_std_climate: float, dt: float, n_steps: int,
                 lambda_rbf: float, gamma: float, n_knn: int,
                 n_init: int, seed: int, long_n_lo: int) -> None:
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _WORKER_STATE["A_mid"] = A_mid
    _WORKER_STATE["dAdt_true"] = dAdt_true
    _WORKER_STATE["ICs"] = ICs
    _WORKER_STATE["X_true_all"] = X_true_all
    _WORKER_STATE["X_std_climate"] = X_std_climate
    _WORKER_STATE["dt"] = dt
    _WORKER_STATE["n_steps"] = n_steps
    _WORKER_STATE["lambda_rbf"] = lambda_rbf
    _WORKER_STATE["gamma"] = gamma
    _WORKER_STATE["n_knn"] = n_knn
    _WORKER_STATE["n_init"] = n_init
    _WORKER_STATE["seed"] = seed
    _WORKER_STATE["long_n_lo"] = long_n_lo


def _fit_and_integrate_one(n_rbf: int) -> dict:
    t0 = time.time()
    A_mid = _WORKER_STATE["A_mid"]
    dAdt_true = _WORKER_STATE["dAdt_true"]
    ICs = _WORKER_STATE["ICs"]
    X_true_all = _WORKER_STATE["X_true_all"]
    X_std_climate = _WORKER_STATE["X_std_climate"]
    dt = _WORKER_STATE["dt"]
    n_steps = _WORKER_STATE["n_steps"]

    rbf = fit_rbf_only_keep_state(
        A_mid, dAdt_true,
        n_rbf=int(n_rbf), seed=_WORKER_STATE["seed"],
        lambda_rbf=_WORKER_STATE["lambda_rbf"],
        gamma=_WORKER_STATE["gamma"],
        n_knn=_WORKER_STATE["n_knn"],
        n_init=_WORKER_STATE["n_init"],
    )
    Phi_n = sindy.rbf_features_iso(
        A_mid, rbf["centers"], rbf["widths"],
        mu_A=rbf["mu_A"], sigma_A=rbf["sigma_A"],
    ) / rbf["col_norms"]
    dAdt_pred = Phi_n @ rbf["xi"]
    rel_err = float(np.linalg.norm(dAdt_true - dAdt_pred)
                    / np.linalg.norm(dAdt_true))
    t_fit = time.time() - t0

    def f_cb(a: np.ndarray) -> np.ndarray:
        return f_rbf(a[None, :], rbf["centers"], rbf["widths"],
                     rbf["mu_A"], rbf["sigma_A"],
                     rbf["col_norms"], rbf["xi"])[0]

    horizons = []
    finite_flags = []
    n_pred_list = []
    tails_per_ic = []
    n_ic = ICs.shape[0]
    long_n_lo = _WORKER_STATE["long_n_lo"]
    for j in range(n_ic):
        _, X_pred = rk4_integrate(f_cb, ICs[j], dt, n_steps)
        h = _eq21_horizon(X_pred, X_true_all[j], dt, X_std_climate)
        finite_at_T = bool(X_pred.shape[0] == n_steps + 1
                           and np.all(np.isfinite(X_pred[-1])))
        horizons.append(h)
        finite_flags.append(finite_at_T)
        n_pred_list.append(int(X_pred.shape[0]))
        if X_pred.shape[0] > long_n_lo:
            tail = X_pred[long_n_lo:]
            finite_mask = np.all(np.isfinite(tail), axis=1)
            tail = tail[finite_mask]
            tails_per_ic.append(tail)
        else:
            tails_per_ic.append(np.zeros((0, 3)))

    pool_pred = (np.concatenate(tails_per_ic, axis=0)
                 if any(t.shape[0] > 0 for t in tails_per_ic)
                 else np.zeros((0, 3)))
    pool_true = X_true_all[:, long_n_lo:, :].reshape(-1, 3)

    w1 = [float("nan"), float("nan"), float("nan")]
    if pool_pred.shape[0] > 1:
        for k in range(3):
            w1[k] = float(wasserstein_distance(pool_true[:, k],
                                               pool_pred[:, k]))

    t_total = time.time() - t0
    return {
        "n_rbf": int(n_rbf),
        "rel_err_alpha_dot": rel_err,
        "horizons": list(map(float, horizons)),
        "horizon_median": float(np.median(horizons)),
        "horizon_min": float(np.min(horizons)),
        "horizon_max": float(np.max(horizons)),
        "finite_at_T": list(map(bool, finite_flags)),
        "n_finite_at_T": int(sum(finite_flags)),
        "n_pred": n_pred_list,
        "w1": w1,
        "n_tail_pooled": int(pool_pred.shape[0]),
        "t_fit_seconds": t_fit,
        "t_total_seconds": t_total,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n-rbf", type=int, nargs="+",
                   default=[5, 10, 20, 50, 100, 200, 400, 800, 1600, 3200, 6400])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--lambda-rbf", type=float, default=1e-3)
    p.add_argument("--lambda-poly", type=float, default=1.0,
                   help="for the poly-only reference fit only")
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--n-knn", type=int, default=5)
    p.add_argument("--n-init", type=int, default=1)
    p.add_argument("--n-ic", type=int, default=8)
    p.add_argument("--ic-rng-seed", type=int, default=0)
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--T", type=float, default=50.0)
    p.add_argument("--long-window-t0", type=float, default=10.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    ds = data.load("LOR63")
    A = ds.u.astype(np.float64)
    dt_data = float(ds.t[1] - ds.t[0])
    print(f"Loaded {ds}; dt_data={dt_data:g}, M={A.shape[0]}, T_data={ds.t[-1]:g}")

    out_dir = args.out_dir or (data.results_dir("LOR63") / "step1_nrbf_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    A_mid, dAdt_true = sindy.deriv_5point(A, dt_data)

    # -- poly-only baseline reference for the plots --------------------------
    poly_model = _fit_quad_only(A, dt_data, args.lambda_poly)
    rel_poly = float(np.linalg.norm(
        dAdt_true - sindy.poly_features(A_mid, 2) @ poly_model.xi_poly
    ) / np.linalg.norm(dAdt_true))
    print(f"\nPoly-only baseline rel_err_alpha_dot = {rel_poly:.3e}")

    X_std_climate = float(A.std(axis=0).mean())
    print(f"X_std_climate = {X_std_climate:.4f}")
    print(f"Lyapunov time = {T_LYAPUNOV:.3f} s")

    # -- Shared ICs + FOM trajectories ---------------------------------------
    rng_ic = np.random.default_rng(args.ic_rng_seed)
    ic_idx = rng_ic.choice(A.shape[0], size=args.n_ic, replace=False)
    ic_idx.sort()
    ICs = A[ic_idx]
    n_steps = int(round(args.T / args.dt))
    long_n_lo = int(round(args.long_window_t0 / args.dt))

    print(f"\nPre-integrating {args.n_ic} FOM trajectories at dt={args.dt}, "
          f"T={args.T} ({n_steps} steps)...")
    X_true_all = np.empty((args.n_ic, n_steps + 1, 3))
    for j in range(args.n_ic):
        X_true_all[j] = _integrate_fom_truth(ICs[j], args.dt, n_steps)
    print(f"  done. FOM trajectories ready.")

    # -- poly-only horizon for reference (single cell, sequential) -----------
    def f_poly_cb(a: np.ndarray) -> np.ndarray:
        return (sindy.poly_features(a[None, :], 2) @ poly_model.xi_poly)[0]
    horizons_poly = []
    for j in range(args.n_ic):
        _, X_pp = rk4_integrate(f_poly_cb, ICs[j], args.dt, n_steps)
        horizons_poly.append(_eq21_horizon(X_pp, X_true_all[j], args.dt, X_std_climate))
    h_poly_med = float(np.median(horizons_poly))
    print(f"  poly-only horizon median = {h_poly_med:.2f}s ({h_poly_med/T_LYAPUNOV:.2f} t_L)")

    # -- Pool over n_rbf -----------------------------------------------------
    n_rbf_list = sorted(set(int(n) for n in args.n_rbf))
    print(f"\nSweeping n_rbf = {n_rbf_list} with {args.workers} workers "
          f"(seed={args.seed}, lambda_rbf={args.lambda_rbf})...")
    init_args = (A_mid, dAdt_true, ICs, X_true_all, X_std_climate,
                 args.dt, n_steps, args.lambda_rbf, args.gamma,
                 args.n_knn, args.n_init, args.seed, long_n_lo)
    t_pool_start = time.time()
    results = []
    with mp.Pool(processes=args.workers, initializer=_worker_init,
                 initargs=init_args) as pool:
        for r in pool.imap_unordered(_fit_and_integrate_one, n_rbf_list):
            results.append(r)
            print(f"  [{time.time() - t_pool_start:6.1f}s] "
                  f"n_rbf={r['n_rbf']:>5d}  rel_err={r['rel_err_alpha_dot']:.3e}  "
                  f"T_ph_med={r['horizon_median']:6.2f}s  "
                  f"finite={r['n_finite_at_T']}/{args.n_ic}  "
                  f"W1=({r['w1'][0]:.2f},{r['w1'][1]:.2f},{r['w1'][2]:.2f})  "
                  f"t_fit={r['t_fit_seconds']:.1f}s",
                  flush=True)
    print(f"\nPool finished in {time.time() - t_pool_start:.1f} s.")

    results.sort(key=lambda r: r["n_rbf"])

    summary = {
        "config": {
            "n_rbf_grid": n_rbf_list,
            "seed": args.seed, "lambda_rbf": args.lambda_rbf,
            "lambda_poly_ref": args.lambda_poly,
            "gamma": args.gamma, "n_knn": args.n_knn, "n_init": args.n_init,
            "n_ic": args.n_ic, "ic_rng_seed": args.ic_rng_seed,
            "dt": args.dt, "T": args.T,
            "X_std_climate": X_std_climate,
            "lyapunov_l63": LYAPUNOV_L63, "horizon_frac": HORIZON_FRAC,
        },
        "poly_only_reference": {
            "rel_err_alpha_dot": rel_poly,
            "horizons": list(map(float, horizons_poly)),
            "horizon_median": h_poly_med,
        },
        "per_n_rbf": results,
    }
    (out_dir / "step1_nrbf_sweep.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_dir / 'step1_nrbf_sweep.json'}")

    # -- Plots ---------------------------------------------------------------
    nrbf = np.array([r["n_rbf"] for r in results], dtype=float)
    rel_err = np.array([r["rel_err_alpha_dot"] for r in results])
    h_med = np.array([r["horizon_median"] for r in results])
    h_min = np.array([r["horizon_min"] for r in results])
    h_max = np.array([r["horizon_max"] for r in results])
    n_finite = np.array([r["n_finite_at_T"] for r in results])
    w1 = np.array([r["w1"] for r in results])  # (Ncells, 3)

    # 1) horizon vs n_rbf
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    ax.fill_between(nrbf, h_min, h_max, color="C3", alpha=0.2,
                    label="min/max over 8 ICs")
    ax.plot(nrbf, h_med, "D-", color="C3", lw=1.6, label="RBF-only median")
    ax.axhline(h_poly_med, color="C0", lw=1.2, ls="--",
               label=f"poly-only median ({h_poly_med:.2f} s)")
    ax.set_xscale("log")
    ax.set_xlabel(r"$n_{rbf}$")
    ax.set_ylabel(r"Eq.21 prediction horizon $T_{ph}$ (s)")
    ax.grid(True, ls=":", alpha=0.4, which="both")
    ax.legend(fontsize=10)
    ax.set_title(f"L63 RBF-only $T_{{ph}}$ vs $n_{{rbf}}$  "
                 f"($\\lambda_{{rbf}}$={args.lambda_rbf:g}, seed={args.seed})")
    ax2 = ax.twinx()
    ax2.set_ylabel(r"$T_{ph}$ (Lyapunov times)", color="grey")
    ax2.set_ylim(np.array(ax.get_ylim()) / T_LYAPUNOV)
    ax2.tick_params(axis="y", colors="grey")
    fig.tight_layout()
    fig.savefig(out_dir / "horizon_vs_nrbf.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'horizon_vs_nrbf.png'}")

    # 2) horizon vs static-fit residual (the hypothesis-test plot)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    sc = ax.scatter(rel_err, h_med, c=np.log10(nrbf), cmap="viridis",
                    s=80, edgecolor="k", linewidth=0.6, zorder=3)
    for x, y, n in zip(rel_err, h_med, nrbf):
        ax.annotate(f"n={int(n)}", (x, y),
                    textcoords="offset points", xytext=(7, 4), fontsize=8)
    ax.axhline(h_poly_med, color="C0", lw=1.2, ls="--",
               label=f"poly-only $T_{{ph}}$ ({h_poly_med:.2f} s)")
    ax.axvline(rel_poly, color="C0", lw=0.8, ls=":",
               label=f"poly-only static rel_err ({rel_poly:.2e})")
    ax.set_xscale("log")
    ax.set_xlabel(r"static-fit relative error  $\Vert d\hat a/dt - da/dt\Vert / \Vert da/dt \Vert$")
    ax.set_ylabel(r"$T_{ph}$ (s, median over 8 ICs)")
    ax.grid(True, ls=":", alpha=0.4, which="both")
    ax.legend(fontsize=9, loc="upper right")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"$\log_{10}(n_{rbf})$")
    ax.set_title("L63 RBF-only: does static-fit residual predict dynamic horizon?")
    fig.tight_layout()
    fig.savefig(out_dir / "horizon_vs_alpha_dot_residual.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'horizon_vs_alpha_dot_residual.png'}")

    # 3) W1 vs n_rbf
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    comp_colors = ("C0", "C2", "C3")
    for k, ck in enumerate(comp_colors):
        ax.plot(nrbf, w1[:, k], "o-", color=ck, lw=1.4,
                label=f"$W_1$({'xyz'[k]})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$n_{rbf}$")
    ax.set_ylabel(r"$W_1$ marginal climate distance")
    ax.grid(True, ls=":", alpha=0.4, which="both")
    ax.legend(fontsize=10)
    ax.set_title(f"L63 RBF-only marginal climate vs $n_{{rbf}}$  "
                 f"(tail $t \\in [{args.long_window_t0:g}, {args.T:g}]$ s)")
    fig.tight_layout()
    fig.savefig(out_dir / "w1_vs_nrbf.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'w1_vs_nrbf.png'}")

    # 4) stability floor: finite-at-T count vs n_rbf
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.0))
    ax.plot(nrbf, n_finite, "s-", color="C3", lw=1.6, markersize=8)
    ax.axhline(args.n_ic, color="grey", lw=0.7, ls=":",
               label=f"all-finite ({args.n_ic})")
    ax.set_xscale("log")
    ax.set_xlabel(r"$n_{rbf}$")
    ax.set_ylabel(f"# ICs (of {args.n_ic}) finite at T={args.T:g} s")
    ax.set_ylim(-0.5, args.n_ic + 0.5)
    ax.grid(True, ls=":", alpha=0.4, which="both")
    ax.legend(fontsize=10)
    ax.set_title(f"L63 RBF-only stability floor "
                 f"($\\lambda_{{rbf}}$={args.lambda_rbf:g}, "
                 f"$T = {args.T:g}$ s)")
    fig.tight_layout()
    fig.savefig(out_dir / "stability_floor.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'stability_floor.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
