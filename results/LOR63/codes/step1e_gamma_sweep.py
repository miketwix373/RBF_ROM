"""L63 RBF-only step 1e: 2D (n_rbf, gamma) sweep with attractor-scaled bandwidth.

Tests fix #1 from the off-cloud risk-surface discussion: decouple Gaussian
width from centre density. With `bandwidth_mode="attr_scaled"`, every
centre gets width sigma_k = gamma in standardised coords (sigma_attr_std = 1),
so widening centres does not shrink Gaussian support. Sweep n_rbf x alpha
on a Pareto-style grid; per cell, record both in-cloud horizon (Eq.21) and
outer-shell absorbing behaviour (R_abs at canonical thresholds), so the
trade-off is visible in one shot.

Per cell:
  * fit RBF-only with bandwidth_mode="attr_scaled", gamma=alpha
  * relative alpha_dot residual (static fit)
  * Eq.21 prediction horizon median over n_ic ICs
  * finite-at-T flags
  * Wasserstein-1 marginal climate distance on the tail
  * radial profile: R(x) = r_hat . f_rbf, ||f_rbf||, dead-fraction
  * R_abs at (thresh, floor_frac) = (0.6, 0.05)  (canonical canonical)

Outputs (results/LOR63/step1e_gamma_sweep/):
  step1e_sweep.json      - all per-cell records
  pareto.png             - horizon vs R_abs scatter, coloured by alpha,
                           sized by n_rbf
  heatmap_horizon.png    - horizon over (n_rbf, alpha)
  heatmap_rabs.png       - R_abs over (n_rbf, alpha)
  radial_grid.png        - radial profiles per cell, rows=n_rbf cols=alpha
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _l63_rbf_lib import (  # noqa: E402
    f_rbf, f_truth_l63, f_truth_l63_single,
    fit_rbf_only_keep_state, rk4_integrate,
)
from chord2 import data, sindy  # noqa: E402


LYAPUNOV_L63 = 0.906
T_LYAPUNOV = 1.0 / LYAPUNOV_L63
HORIZON_FRAC = 0.5


def fib_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ], axis=1)


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


def _r_abs_outer(shell_R, inward_frac, mag_mean, *, thresh, floor):
    sR = np.asarray(shell_R)
    inf_ok = np.asarray(inward_frac) >= thresh
    mag_ok = np.asarray(mag_mean) >= floor
    ok = inf_ok & mag_ok
    if not ok.any():
        return math.inf
    tail_start = len(ok)
    for i in range(len(ok) - 1, -1, -1):
        if not ok[i]:
            break
        tail_start = i
    if tail_start == len(ok):
        return math.inf
    return float(sR[tail_start])


_WORKER_STATE: dict = {}


def _worker_init(A_mid, dAdt_true, ICs, X_true_all, X_std_climate, dt,
                 n_steps, lambda_rbf, seed, long_n_lo,
                 x_centroid, sigma_attr, shell_r_grid, d_hat,
                 dead_zone_eps, f_truth_mean_mag,
                 thresh_canon, floor_frac_canon, lambda_tikh,
                 models_dir) -> None:
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _WORKER_STATE.update(dict(
        A_mid=A_mid, dAdt_true=dAdt_true, ICs=ICs, X_true_all=X_true_all,
        X_std_climate=X_std_climate, dt=dt, n_steps=n_steps,
        lambda_rbf=lambda_rbf, seed=seed, long_n_lo=long_n_lo,
        x_centroid=x_centroid, sigma_attr=sigma_attr,
        shell_r_grid=np.asarray(shell_r_grid, dtype=float),
        d_hat=d_hat, dead_zone_eps=dead_zone_eps,
        f_truth_mean_mag=f_truth_mean_mag,
        thresh_canon=thresh_canon, floor_frac_canon=floor_frac_canon,
        lambda_tikh=lambda_tikh,
        models_dir=models_dir,
    ))


def _radial_profile(model, x_centroid, sigma_attr, shell_R, d_hat, dz_eps):
    R_mean, R_inward_frac, mag_mean, dead_frac = [], [], [], []
    for R in shell_R:
        pts = x_centroid + R * sigma_attr * d_hat
        f_r = f_rbf(pts, model["centers"], model["widths"],
                    model["mu_A"], model["sigma_A"],
                    model["col_norms"], model["xi"])
        rvec = pts - x_centroid
        rhat = rvec / np.linalg.norm(rvec, axis=1, keepdims=True)
        R_proj = np.einsum("ij,ij->i", f_r, rhat)
        mag = np.linalg.norm(f_r, axis=1)
        R_mean.append(float(R_proj.mean()))
        R_inward_frac.append(float((R_proj < 0).mean()))
        mag_mean.append(float(mag.mean()))
        dead_frac.append(float((mag < dz_eps).mean()))
    return R_mean, R_inward_frac, mag_mean, dead_frac


def _fit_and_diagnose_one(cell: tuple) -> dict:
    n_rbf, alpha = cell
    t0 = time.time()
    A_mid = _WORKER_STATE["A_mid"]
    dAdt_true = _WORKER_STATE["dAdt_true"]
    ICs = _WORKER_STATE["ICs"]
    X_true_all = _WORKER_STATE["X_true_all"]
    dt = _WORKER_STATE["dt"]
    n_steps = _WORKER_STATE["n_steps"]
    X_std_climate = _WORKER_STATE["X_std_climate"]
    long_n_lo = _WORKER_STATE["long_n_lo"]

    rbf = fit_rbf_only_keep_state(
        A_mid, dAdt_true,
        n_rbf=int(n_rbf), seed=_WORKER_STATE["seed"],
        lambda_rbf=_WORKER_STATE["lambda_rbf"],
        gamma=float(alpha), n_knn=5, n_init=1,
        bandwidth_mode="attr_scaled",
        lambda_tikh=_WORKER_STATE["lambda_tikh"],
        log_diagnostics=True,
    )
    Phi_n = sindy.rbf_features_iso(
        A_mid, rbf["centers"], rbf["widths"],
        mu_A=rbf["mu_A"], sigma_A=rbf["sigma_A"],
    ) / rbf["col_norms"]
    dAdt_pred = Phi_n @ rbf["xi"]
    rel_err = float(np.linalg.norm(dAdt_true - dAdt_pred)
                    / np.linalg.norm(dAdt_true))
    t_fit = time.time() - t0

    def f_cb(a):
        return f_rbf(a[None, :], rbf["centers"], rbf["widths"],
                     rbf["mu_A"], rbf["sigma_A"],
                     rbf["col_norms"], rbf["xi"])[0]

    horizons, finite_flags, tails_per_ic = [], [], []
    n_ic = ICs.shape[0]
    for j in range(n_ic):
        _, X_pred = rk4_integrate(f_cb, ICs[j], dt, n_steps)
        h = _eq21_horizon(X_pred, X_true_all[j], dt, X_std_climate)
        finite_at_T = bool(X_pred.shape[0] == n_steps + 1
                           and np.all(np.isfinite(X_pred[-1])))
        horizons.append(h)
        finite_flags.append(finite_at_T)
        if X_pred.shape[0] > long_n_lo:
            tail = X_pred[long_n_lo:]
            tail = tail[np.all(np.isfinite(tail), axis=1)]
            tails_per_ic.append(tail)
        else:
            tails_per_ic.append(np.zeros((0, 3)))

    pool_pred = (np.concatenate(tails_per_ic, axis=0)
                 if any(t.shape[0] > 0 for t in tails_per_ic)
                 else np.zeros((0, 3)))
    pool_true = X_true_all[:, long_n_lo:, :].reshape(-1, 3)
    w1 = [float("nan")] * 3
    if pool_pred.shape[0] > 1:
        for k in range(3):
            w1[k] = float(wasserstein_distance(pool_true[:, k],
                                               pool_pred[:, k]))

    # Off-cloud radial profile + R_abs
    R_mean, R_inward_frac, mag_mean, dead_frac = _radial_profile(
        rbf, _WORKER_STATE["x_centroid"], _WORKER_STATE["sigma_attr"],
        _WORKER_STATE["shell_r_grid"], _WORKER_STATE["d_hat"],
        _WORKER_STATE["dead_zone_eps"],
    )
    floor = _WORKER_STATE["floor_frac_canon"] * _WORKER_STATE["f_truth_mean_mag"]
    R_abs_canon = _r_abs_outer(_WORKER_STATE["shell_r_grid"],
                               R_inward_frac, mag_mean,
                               thresh=_WORKER_STATE["thresh_canon"],
                               floor=floor)

    diag = rbf.get("diagnostics", [])
    kappa_final = (diag[-1]["kappa"] if diag else float("nan"))
    kappa_max = (max(d["kappa"] for d in diag) if diag else float("nan"))
    t_svd_total = (sum(d["t_svd_seconds"] for d in diag) if diag else 0.0)

    # Persist the model so we can compare against the k-NN-bandwidth models
    # in results/LOR63/step1_nrbf_sweep/models/ side-by-side.
    models_dir = _WORKER_STATE["models_dir"]
    active = (np.max(np.abs(rbf["xi"]), axis=1)
              >= _WORKER_STATE["lambda_rbf"])
    npz_path = models_dir / f"model_n{int(n_rbf):05d}_a{float(alpha):.2f}.npz"
    meta_save = {
        "n_rbf": int(n_rbf),
        "alpha": float(alpha),
        "bandwidth_mode": "attr_scaled",
        "seed": int(_WORKER_STATE["seed"]),
        "lambda_rbf": float(_WORKER_STATE["lambda_rbf"]),
        "lambda_tikh": float(_WORKER_STATE["lambda_tikh"]),
        "rel_err_alpha_dot": rel_err,
        "nnz": int(active.sum()),
        "kappa_final": kappa_final,
        "t_fit_seconds": t_fit,
    }
    np.savez(
        npz_path,
        centers=rbf["centers"], widths=rbf["widths"],
        mu_A=rbf["mu_A"], sigma_A=rbf["sigma_A"],
        col_norms=rbf["col_norms"], xi=rbf["xi"], active=active,
        meta=np.array(meta_save, dtype=object),
    )

    return {
        "model_file": npz_path.name,
        "n_rbf": int(n_rbf),
        "alpha": float(alpha),
        "rel_err_alpha_dot": rel_err,
        "horizon_median": float(np.median(horizons)),
        "horizon_min": float(np.min(horizons)),
        "horizon_max": float(np.max(horizons)),
        "n_finite_at_T": int(sum(finite_flags)),
        "w1": w1,
        "shell_R": list(map(float, _WORKER_STATE["shell_r_grid"])),
        "R_mean": R_mean,
        "R_inward_frac": R_inward_frac,
        "mag_mean": mag_mean,
        "dead_frac": dead_frac,
        "R_abs_canon": ("inf" if math.isinf(R_abs_canon) else R_abs_canon),
        "nnz": int(rbf["nnz"]),
        "stlsq_n_iter": len(diag),
        "kappa_final": kappa_final,
        "kappa_max": kappa_max,
        "t_svd_total_seconds": float(t_svd_total),
        "stlsq_diagnostics": diag,
        "t_fit_seconds": t_fit,
        "t_total_seconds": time.time() - t0,
    }


def _render_figs(out_dir, results, args, f_truth_mean_mag):
    n_rbf_vals = sorted(set(args.n_rbf_list))
    alpha_vals = sorted(set(args.alpha_list))
    by_cell = {(r["n_rbf"], r["alpha"]): r for r in results}

    horizon_grid = np.full((len(n_rbf_vals), len(alpha_vals)), np.nan)
    rabs_grid = np.full_like(horizon_grid, np.nan)
    cap_R = max(args.shell_r_grid)
    for i, n in enumerate(n_rbf_vals):
        for j, a in enumerate(alpha_vals):
            r = by_cell.get((n, a))
            if r is None:
                continue
            horizon_grid[i, j] = r["horizon_median"]
            v = r["R_abs_canon"]
            rabs_grid[i, j] = (cap_R + 0.5) if v == "inf" else v

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.5))
    im = ax.imshow(horizon_grid, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(alpha_vals)))
    ax.set_xticklabels([f"{a:g}" for a in alpha_vals])
    ax.set_yticks(range(len(n_rbf_vals)))
    ax.set_yticklabels([str(n) for n in n_rbf_vals])
    ax.set_xlabel(r"$\alpha$ (Gaussian width in units of $\sigma_{\rm attr}$)")
    ax.set_ylabel(r"$n_{\rm rbf}$")
    ax.set_title("In-cloud horizon median (s) — higher is better")
    horizon_mean = float(np.nanmean(horizon_grid)) if np.any(~np.isnan(horizon_grid)) else 0.0
    for i in range(len(n_rbf_vals)):
        for j in range(len(alpha_vals)):
            v = horizon_grid[i, j]
            if np.isnan(v):
                ax.text(j, i, "skip", ha="center", va="center",
                        color="grey", fontsize=9)
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < horizon_mean else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label=r"$T_{\rm ph}$ (s)")
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap_horizon.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'heatmap_horizon.png'}")

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.5))
    im = ax.imshow(rabs_grid, aspect="auto", origin="lower", cmap="magma_r")
    ax.set_xticks(range(len(alpha_vals)))
    ax.set_xticklabels([f"{a:g}" for a in alpha_vals])
    ax.set_yticks(range(len(n_rbf_vals)))
    ax.set_yticklabels([str(n) for n in n_rbf_vals])
    ax.set_xlabel(r"$\alpha$ (Gaussian width in units of $\sigma_{\rm attr}$)")
    ax.set_ylabel(r"$n_{\rm rbf}$")
    ax.set_title(rf"$R_{{\rm abs}}/\sigma_{{\rm attr}}$  "
                 rf"(thresh={args.thresh_canon:g}, "
                 rf"floor={args.floor_frac_canon:g}) — lower is better")
    for i in range(len(n_rbf_vals)):
        for j in range(len(alpha_vals)):
            r = by_cell.get((n_rbf_vals[i], alpha_vals[j]))
            if r is None:
                ax.text(j, i, "skip", ha="center", va="center",
                        color="grey", fontsize=9)
                continue
            v = r["R_abs_canon"]
            tag = "no abs" if v == "inf" else f"{v:.2f}"
            ax.text(j, i, tag, ha="center", va="center",
                    color="white" if rabs_grid[i, j] > cap_R * 0.6 else "black",
                    fontsize=9,
                    fontweight="bold" if v == "inf" else "normal")
    fig.colorbar(im, ax=ax, label=r"$R_{\rm abs}/\sigma_{\rm attr}$")
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap_rabs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'heatmap_rabs.png'}")

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.0))
    cmap = plt.get_cmap("viridis")
    n_alpha = len(alpha_vals)
    for j, a in enumerate(alpha_vals):
        col = cmap(j / max(1, n_alpha - 1))
        xs, ys, sizes, ns_present = [], [], [], []
        for n in n_rbf_vals:
            r = by_cell.get((n, a))
            if r is None:
                continue
            v = r["R_abs_canon"]
            xs.append(cap_R + 0.5 if v == "inf" else v)
            ys.append(r["horizon_median"])
            sizes.append(30 + 70 * np.log10(max(n, 10)))
            ns_present.append(n)
        if not xs:
            continue
        ax.scatter(xs, ys, s=sizes, color=col, edgecolor="k",
                   linewidth=0.6, label=rf"$\alpha={a:g}$", zorder=3)
        for x, y, n in zip(xs, ys, ns_present):
            ax.annotate(f"n={n}", (x, y),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.axvline(cap_R + 0.5, color="crimson", lw=0.8, ls=":",
               label="no absorbing tail")
    ax.set_xlabel(r"$R_{\rm abs}/\sigma_{\rm attr}$"
                  " (lower = absorbing kicks in sooner; better)")
    ax.set_ylabel(r"In-cloud horizon $T_{\rm ph}$ (s, higher better)")
    ax.set_title("L63 RBF-only Pareto: attractor-scaled bandwidth sweep")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "pareto.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'pareto.png'}")

    n_rows = len(n_rbf_vals)
    n_cols = len(alpha_vals)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.7 * n_cols, 2.2 * n_rows),
                             sharex=True, squeeze=False)
    for i, n in enumerate(n_rbf_vals):
        for j, a in enumerate(alpha_vals):
            ax = axes[i, j]
            r = by_cell.get((n, a))
            if r is None:
                ax.text(0.5, 0.5, "skipped", ha="center", va="center",
                        transform=ax.transAxes, color="grey", fontsize=9)
                ax.set_title(rf"$n={n}$, $\alpha={a:g}$ (skipped)", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            shell_R = r["shell_R"]
            ax.plot(shell_R, r["R_inward_frac"], "C0o-", lw=1.2, ms=3,
                    label=r"$\rho_{\rm in}$")
            ax.axhline(args.thresh_canon, color="C0", lw=0.6, ls=":")
            ax2 = ax.twinx()
            ax2.semilogy(shell_R, np.maximum(r["mag_mean"], 1e-3),
                         "C3s-", lw=1.2, ms=3, label=r"$\|f\|_{\rm mean}$")
            floor_abs = args.floor_frac_canon * f_truth_mean_mag
            ax2.axhline(floor_abs, color="C3", lw=0.6, ls=":")
            ax.set_ylim(0, 1)
            ax2.set_ylim(1e-3, 1e3)
            ax.set_title(rf"$n={n}$, $\alpha={a:g}$", fontsize=9)
            if i == n_rows - 1:
                ax.set_xlabel(r"$R/\sigma$")
            if j == 0:
                ax.set_ylabel(r"$\rho_{\rm in}$", color="C0")
            if j == n_cols - 1:
                ax2.set_ylabel(r"$\|f\|_{\rm mean}$", color="C3")
    fig.suptitle("Radial profiles per cell — blue: inward-fraction, "
                 "red: ‖f‖ mean (log)", y=1.005)
    fig.tight_layout()
    fig.savefig(out_dir / "radial_grid.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'radial_grid.png'}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n-rbf-list", type=int, nargs="+",
                   default=[100, 800, 3200])
    p.add_argument("--alpha-list", type=float, nargs="+",
                   default=[0.25, 0.5, 1.0, 2.0, 4.0])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--lambda-rbf", type=float, default=1e-3)
    p.add_argument("--lambda-tikh", type=float, default=1e-8,
                   help="Tikhonov regulariser in STLSQ inner lstsq")
    p.add_argument("--n-ic", type=int, default=8)
    p.add_argument("--ic-rng-seed", type=int, default=0)
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--T", type=float, default=50.0)
    p.add_argument("--long-window-t0", type=float, default=10.0)
    p.add_argument("--shell-r-grid", type=float, nargs="+",
                   default=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5,
                            2.0, 2.5, 3.0, 4.0, 5.0])
    p.add_argument("--n-sphere", type=int, default=300)
    p.add_argument("--thresh-canon", type=float, default=0.6)
    p.add_argument("--floor-frac-canon", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--skip-cells", type=str, nargs="*",
                   default=["3200,0.25", "3200,0.5", "3200,4.0"],
                   help="pairs 'n_rbf,alpha' to skip from the full grid")
    p.add_argument("--aggregate-only", action="store_true",
                   help="skip fits; regenerate figures from existing JSON")
    args = p.parse_args()

    if args.aggregate_only:
        out_dir = args.out_dir or (data.results_dir("LOR63") / "step1e_gamma_sweep")
        in_json = out_dir / "step1e_sweep.json"
        print(f"[aggregate-only] loading {in_json}")
        summary = json.loads(in_json.read_text())
        cfg = summary["config"]
        results = summary["cells"]
        X_std_climate = cfg["X_std_climate"]
        x_centroid = np.asarray(cfg["x_centroid"], dtype=float)
        sigma_attr = cfg["sigma_attr"]
        f_truth_mean_mag = cfg["f_truth_mean_mag"]
        dead_zone_eps = cfg["dead_zone_eps"]
        n_rbf_list_eff = cfg["n_rbf_list"]
        alpha_list_eff = cfg["alpha_list"]
        shell_r_grid = cfg["shell_r_grid"]

        class _A:
            pass
        args2 = _A()
        args2.n_rbf_list = n_rbf_list_eff
        args2.alpha_list = alpha_list_eff
        args2.shell_r_grid = shell_r_grid
        args2.thresh_canon = cfg["thresh_canon"]
        args2.floor_frac_canon = cfg["floor_frac_canon"]
        return _render_figs(out_dir, results, args2, f_truth_mean_mag)

    ds = data.load("LOR63")
    A = ds.u.astype(np.float64)
    dt_data = float(ds.t[1] - ds.t[0])
    print(f"Loaded {ds}; dt_data={dt_data:g}, M={A.shape[0]}")

    out_dir = args.out_dir or (data.results_dir("LOR63") / "step1e_gamma_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    A_mid, dAdt_true = sindy.deriv_5point(A, dt_data)

    X_std_climate = float(A.std(axis=0).mean())
    print(f"X_std_climate = {X_std_climate:.4f}")

    x_centroid = A.mean(axis=0)
    sigma_attr = float(A.std(axis=0).mean())
    f_truth_mean_mag = float(np.linalg.norm(f_truth_l63(A), axis=1).mean())
    dead_zone_eps = 1e-3 * f_truth_mean_mag
    print(f"x_centroid={x_centroid}, sigma_attr={sigma_attr:.4f}, "
          f"<||f_truth||>={f_truth_mean_mag:.4f}")

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
    print("  FOM trajectories ready.")

    d_hat = fib_sphere(args.n_sphere)

    skip_set = set()
    for s in args.skip_cells:
        n_str, a_str = s.split(",")
        skip_set.add((int(n_str), float(a_str)))

    full_cells = list(product(sorted(set(args.n_rbf_list)),
                              sorted(set(args.alpha_list))))
    cells = [c for c in full_cells if c not in skip_set]
    skipped = [c for c in full_cells if c in skip_set]
    print(f"\nSweep grid: {len(cells)} cells of "
          f"{len(full_cells)} (full n_rbf={sorted(set(args.n_rbf_list))}, "
          f"alpha={sorted(set(args.alpha_list))})")
    if skipped:
        print(f"  skipped: {skipped}")
    print(f"  workers={args.workers}, seed={args.seed}, "
          f"lambda_rbf={args.lambda_rbf}, lambda_tikh={args.lambda_tikh:g}")
    print(f"  models -> {models_dir}")

    init_args = (
        A_mid, dAdt_true, ICs, X_true_all, X_std_climate, args.dt, n_steps,
        args.lambda_rbf, args.seed, long_n_lo,
        x_centroid, sigma_attr, args.shell_r_grid, d_hat,
        dead_zone_eps, f_truth_mean_mag,
        args.thresh_canon, args.floor_frac_canon, args.lambda_tikh,
        models_dir,
    )
    t_pool_start = time.time()
    results = []
    with mp.Pool(processes=args.workers, initializer=_worker_init,
                 initargs=init_args) as pool:
        for r in pool.imap_unordered(_fit_and_diagnose_one, cells):
            results.append(r)
            R_abs_str = ("inf" if r["R_abs_canon"] == "inf"
                         else f"{r['R_abs_canon']:.2f}")
            kappa_str = ("nan" if not np.isfinite(r["kappa_final"])
                         else f"{r['kappa_final']:.2e}")
            print(f"  [{time.time() - t_pool_start:6.1f}s] "
                  f"n={r['n_rbf']:>5d}  a={r['alpha']:.2f}  "
                  f"rel_err={r['rel_err_alpha_dot']:.3e}  "
                  f"T_ph={r['horizon_median']:5.2f}s  "
                  f"fin={r['n_finite_at_T']}/{args.n_ic}  "
                  f"R_abs={R_abs_str:>5s}  "
                  f"k={kappa_str}  "
                  f"its={r['stlsq_n_iter']:>2d}  "
                  f"t_svd={r['t_svd_total_seconds']:.1f}s  "
                  f"t_fit={r['t_fit_seconds']:.1f}s",
                  flush=True)
    print(f"\nPool finished in {time.time() - t_pool_start:.1f} s.")

    results.sort(key=lambda r: (r["n_rbf"], r["alpha"]))

    summary = {
        "config": {
            "n_rbf_list": sorted(set(args.n_rbf_list)),
            "alpha_list": sorted(set(args.alpha_list)),
            "seed": args.seed, "lambda_rbf": args.lambda_rbf,
            "lambda_tikh": args.lambda_tikh,
            "n_ic": args.n_ic, "dt": args.dt, "T": args.T,
            "long_window_t0": args.long_window_t0,
            "X_std_climate": X_std_climate,
            "x_centroid": list(map(float, x_centroid)),
            "sigma_attr": sigma_attr,
            "f_truth_mean_mag": f_truth_mean_mag,
            "dead_zone_eps": dead_zone_eps,
            "shell_r_grid": list(map(float, args.shell_r_grid)),
            "thresh_canon": args.thresh_canon,
            "floor_frac_canon": args.floor_frac_canon,
            "horizon_frac": HORIZON_FRAC,
        },
        "cells": results,
    }
    out_json = out_dir / "step1e_sweep.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_json}")

    return _render_figs(out_dir, results, args, f_truth_mean_mag)


if __name__ == "__main__":
    raise SystemExit(main())
