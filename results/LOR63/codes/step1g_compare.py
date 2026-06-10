"""L63 step 1g comparison: 4-way head-to-head at n_rbf=3200.

Loads four models at the matched cell n_rbf=3200:
  knn_rbf      -- step 1  : RBF-only, knn-per-centre bandwidth (gamma=1, k=5)
  attr_rbf     -- step 1e : RBF-only, attr-scaled bandwidth   (alpha=1)
  knn_linrbf   -- step 1g : linear + RBF, same centres as knn_rbf
  attr_linrbf  -- step 1g : linear + RBF, same centres as attr_rbf

For each setup: T_ph (Eq.21 prediction horizon), R_abs (outer absorbing
threshold), W1 marginal climatology distance, Lyapunov certificate
interval (V = rho x^2 + sigma y^2 + sigma (z - 2 rho)^2), in-cloud and
off-cloud trajectory overlays, marginal-PDF tails.

Outputs (`results/LOR63/step1g_compare/`):
  step1g_compare.json
  matched_metrics_4way.png
  linear_block_recovery.png
  trajectory_overlay_4way.png
  dead_zone_offcloud.png
  lyapunov_certificate_4way.png
  pdf_grid_4way.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "results" / "LOR63" / "codes"))

from chord2 import data, sindy                                          # noqa: E402
from _l63_rbf_lib import (                                              # noqa: E402
    f_rbf, f_truth_l63, f_truth_l63_single, rk4_integrate,
    SIGMA_L63, RHO_L63, BETA_L63,
)
from step1f_compare import (                                            # noqa: E402
    _fib_sphere, _grad_V_L63, _lyapunov_sweep,
)


# ---------------------------------------------------------------------------
# Model loaders / evaluators
# ---------------------------------------------------------------------------

def _load_rbf(npz_path: Path) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    return {
        "kind":      "rbf",
        "centers":   d["centers"], "widths": d["widths"],
        "mu_A":      d["mu_A"], "sigma_A": d["sigma_A"],
        "col_norms": d["col_norms"], "xi": d["xi"],
        "active":    d["active"],
        "meta": d["meta"].item() if "meta" in d.files else {},
    }


def _load_linrbf(npz_path: Path) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    return {
        "kind":      "linrbf",
        "centers":   d["centers"], "widths": d["widths"],
        "mu_A":      d["mu_A"], "sigma_A": d["sigma_A"],
        "col_norms_poly": d["col_norms_poly"],
        "col_norms_rbf":  d["col_norms_rbf"],
        "xi_poly":   d["xi_poly"], "xi_rbf": d["xi_rbf"],
        "active_poly": d["active_poly"], "active_rbf": d["active_rbf"],
        "meta": d["meta"].item() if "meta" in d.files else {},
    }


def _make_f_batch(model):
    if model["kind"] == "rbf":
        c, w, mu, sg = model["centers"], model["widths"], model["mu_A"], model["sigma_A"]
        cn, xi = model["col_norms"], model["xi"]
        def f_b(X):
            return f_rbf(X, c, w, mu, sg, cn, xi)
        return f_b
    else:
        c, w, mu, sg = model["centers"], model["widths"], model["mu_A"], model["sigma_A"]
        cn_p, cn_r = model["col_norms_poly"], model["col_norms_rbf"]
        xi_p, xi_r = model["xi_poly"], model["xi_rbf"]
        def f_b(X):
            M = X.shape[0]
            Phi_p = np.empty((M, 4), dtype=X.dtype)
            Phi_p[:, 0] = 1.0
            Phi_p[:, 1:] = X
            Phi_r = sindy.rbf_features_iso(X, c, w, mu_A=mu, sigma_A=sg)
            return Phi_p @ xi_p + Phi_r @ xi_r
        return f_b


def _make_f_single(model):
    f_b = _make_f_batch(model)
    def f(a):
        return f_b(a.reshape(1, -1)).ravel()
    return f


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

HORIZON_FRAC = 0.5


def _eq21_horizon(X_pred: np.ndarray, X_true: np.ndarray, dt: float,
                  X_std_climate: float) -> float:
    n = min(X_pred.shape[0], X_true.shape[0])
    if n < 2:
        return 0.0
    err = np.linalg.norm(X_pred[:n] - X_true[:n], axis=1)
    threshold = HORIZON_FRAC * X_std_climate
    over = np.where(err > threshold)[0]
    return float(dt * over[0]) if over.size else float(dt * (n - 1))


def _radial_R_abs(model, x_c, sigma_attr, R_grid_sigma, d_hat,
                  thresh: float = 0.60, floor_frac: float = 0.05) -> float:
    """First R (in sigma_attr units) where inward-fraction stays >= thresh
    AND ||f|| stays >= floor_frac * <||f_truth||>. R_abs = inf if no shell
    qualifies in the swept range."""
    f_b = _make_f_batch(model)
    inward = np.empty(len(R_grid_sigma))
    fnorm  = np.empty(len(R_grid_sigma))
    for k, R in enumerate(R_grid_sigma):
        X = x_c[None, :] + R * sigma_attr * d_hat
        F = f_b(X)
        rvec = X - x_c[None, :]
        rhat = rvec / np.maximum(np.linalg.norm(rvec, axis=1, keepdims=True), 1e-30)
        inward[k] = float((-(F * rhat).sum(axis=1) > 0).mean())
        fnorm[k]  = float(np.linalg.norm(F, axis=1).mean())
    return inward, fnorm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n-rbf", type=int, default=3200)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--dt",    type=float, default=0.005)
    p.add_argument("--T-overlay", type=float, default=10.0)
    p.add_argument("--off-cloud-distance", type=float, default=3.0)
    p.add_argument("--ic-rng-seed",   type=int,   default=11)
    p.add_argument("--n-ic-horizon",  type=int,   default=8)
    p.add_argument("--T-horizon",     type=float, default=50.0)
    p.add_argument("--lyap-r-min", type=float, default=0.2)
    p.add_argument("--lyap-r-max", type=float, default=12.0)
    p.add_argument("--lyap-r-n",   type=int,   default=30)
    p.add_argument("--lyap-n-sphere", type=int, default=600)
    p.add_argument("--pdf-n-ic",   type=int,   default=8)
    p.add_argument("--pdf-ic-seed", type=int,  default=0)
    p.add_argument("--pdf-T",      type=float, default=50.0)
    p.add_argument("--pdf-tail-t0", type=float, default=10.0)
    p.add_argument("--pdf-bins",   type=int,   default=80)
    p.add_argument("--rabs-thresh", type=float, default=0.60)
    p.add_argument("--rabs-floor",  type=float, default=0.05)
    p.add_argument("--rabs-r-min",  type=float, default=0.2)
    p.add_argument("--rabs-r-max",  type=float, default=6.0)
    p.add_argument("--rabs-r-n",    type=int,   default=60)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    out_dir = args.out_dir or (data.results_dir("LOR63") / "step1g_compare")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Path("/mnt/scratch/users/sbrw610/CHORD2/results/LOR63")

    ds = data.load("LOR63")
    A = ds.u.astype(np.float64)
    sigma_attr = float(A.std(axis=0).mean())
    x_centroid = A.mean(axis=0)
    X_std_climate = sigma_attr
    print(f"sigma_attr={sigma_attr:.4f}, x_centroid={x_centroid}")

    models = {
        "knn_rbf":    _load_rbf(base / "step1_nrbf_sweep" / "models"
                                / f"model_n_rbf_{args.n_rbf:05d}.npz"),
        "attr_rbf":   _load_rbf(base / "step1e_gamma_sweep" / "models"
                                / f"model_n{args.n_rbf:05d}_a{args.alpha:.2f}.npz"),
        "knn_linrbf": _load_linrbf(base / "step1g_linrbf"
                                   / f"model_linrbf_knn_n{args.n_rbf:05d}.npz"),
        "attr_linrbf": _load_linrbf(base / "step1g_linrbf"
                                    / f"model_linrbf_attr_n{args.n_rbf:05d}"
                                      f"_a{args.alpha:.2f}.npz"),
    }
    setup_keys  = ["knn_rbf", "attr_rbf", "knn_linrbf", "attr_linrbf"]
    setup_label = {
        "knn_rbf":    "knn  RBF",
        "attr_rbf":   r"attr RBF ($\alpha=1$)",
        "knn_linrbf":  "knn  lin+RBF",
        "attr_linrbf": r"attr lin+RBF ($\alpha=1$)",
    }
    setup_color = {
        "knn_rbf":     "#1f77b4",
        "attr_rbf":    "#d62728",
        "knn_linrbf":  "#5fa8d3",
        "attr_linrbf": "#e57373",
    }
    setup_ls = {
        "knn_rbf":     "-",  "attr_rbf":     "-",
        "knn_linrbf":  "--", "attr_linrbf":  "--",
    }

    # ---- ICs for prediction horizon --------------------------------------
    rng_ic = np.random.default_rng(args.ic_rng_seed)
    ic_idx = rng_ic.choice(A.shape[0], size=args.n_ic_horizon, replace=False)
    ic_idx.sort()
    ICs = A[ic_idx]
    n_steps_h = int(round(args.T_horizon / args.dt))

    print(f"\nPre-integrating {args.n_ic_horizon} FOM trajectories for T_ph "
          f"at dt={args.dt}, T={args.T_horizon}...")
    X_true_h = np.empty((args.n_ic_horizon, n_steps_h + 1, 3))
    for j in range(args.n_ic_horizon):
        _, X = rk4_integrate(f_truth_l63_single, ICs[j], args.dt, n_steps_h)
        X_true_h[j, :X.shape[0]] = X
        if X.shape[0] < n_steps_h + 1:
            X_true_h[j, X.shape[0]:] = X[-1]

    # ---- Per-setup metrics ------------------------------------------------
    print(f"\nFitting metrics for 4 setups at n_rbf={args.n_rbf}...")
    metrics = {}
    for key in setup_keys:
        m = models[key]
        f_single = _make_f_single(m)
        f_batch  = _make_f_batch(m)
        horizons = []
        tails = []
        for j in range(args.n_ic_horizon):
            _, X_p = rk4_integrate(f_single, ICs[j], args.dt, n_steps_h)
            h = _eq21_horizon(X_p, X_true_h[j], args.dt, X_std_climate)
            horizons.append(h)
            n_tail_lo = int(round(10.0 / args.dt))
            if X_p.shape[0] > n_tail_lo:
                tail = X_p[n_tail_lo:]
                tails.append(tail[np.all(np.isfinite(tail), axis=1)])
        tail_pool = np.concatenate(tails, axis=0) if tails else np.zeros((0, 3))
        truth_tail = X_true_h[:, int(round(10.0 / args.dt)):, :].reshape(-1, 3)
        if tail_pool.shape[0] > 0:
            w1 = [float(wasserstein_distance(truth_tail[:, k], tail_pool[:, k]))
                  for k in range(3)]
        else:
            w1 = [float("nan")] * 3
        R_grid = np.linspace(args.rabs_r_min, args.rabs_r_max, args.rabs_r_n)
        d_hat = _fib_sphere(64)
        inward, fnorm = _radial_R_abs(m, x_centroid, sigma_attr, R_grid, d_hat,
                                      thresh=args.rabs_thresh,
                                      floor_frac=args.rabs_floor)
        f_truth_mean_mag = float(np.linalg.norm(f_truth_l63(A), axis=1).mean())
        ok = (inward >= args.rabs_thresh) & (fnorm >= args.rabs_floor * f_truth_mean_mag)
        if ok.any():
            R_abs = float(R_grid[np.argmax(ok)])
        else:
            R_abs = float("inf")
        metrics[key] = {
            "T_ph_median": float(np.median(horizons)),
            "T_ph_min":    float(np.min(horizons)),
            "T_ph_max":    float(np.max(horizons)),
            "fin_count":   int(sum(h > args.T_horizon - 5*args.dt for h in horizons)),
            "n_ic":        int(args.n_ic_horizon),
            "w1":          w1,
            "R_abs":       R_abs,
        }
        print(f"  {key:>14s}: T_ph_med={metrics[key]['T_ph_median']:.2f}s  "
              f"R_abs={'inf' if not np.isfinite(R_abs) else f'{R_abs:.2f}'}  "
              f"W1=({w1[0]:.2f},{w1[1]:.2f},{w1[2]:.2f})")

    # ---- Lyapunov certificate sweep --------------------------------------
    R_lyap = np.logspace(np.log10(args.lyap_r_min), np.log10(args.lyap_r_max),
                         args.lyap_r_n)
    d_hat_lyap = _fib_sphere(args.lyap_n_sphere)
    print(f"\nLyapunov sweep: {args.lyap_r_n} shells x {args.lyap_n_sphere} dirs")

    def _grad_truth_batch(X):
        return f_truth_l63(X)
    lyap = {}
    m_t, mn_t, fn_t = _lyapunov_sweep(_grad_truth_batch, x_centroid, sigma_attr,
                                       R_lyap, d_hat_lyap)
    lyap["truth"] = {"max": m_t, "mean": mn_t, "frac_neg": fn_t}
    for key in setup_keys:
        f_b = _make_f_batch(models[key])
        m_, mn_, fn_ = _lyapunov_sweep(f_b, x_centroid, sigma_attr, R_lyap, d_hat_lyap)
        lyap[key] = {"max": m_, "mean": mn_, "frac_neg": fn_}

    def _cert_iv(R, max_arr):
        is_neg = max_arr < 0.0
        if not is_neg.any():
            return None
        diffs = np.diff(is_neg.astype(int))
        starts = list(np.where(diffs == 1)[0] + 1)
        ends = list(np.where(diffs == -1)[0])
        if is_neg[0]:
            starts = [0] + starts
        if is_neg[-1]:
            ends.append(len(is_neg) - 1)
        if not starts or not ends:
            return None
        runs = list(zip(starts, ends))
        s, e = max(runs, key=lambda se: se[1] - se[0])
        return float(R[s]), float(R[e])

    cert = {"truth": _cert_iv(R_lyap, lyap["truth"]["max"])}
    for key in setup_keys:
        cert[key] = _cert_iv(R_lyap, lyap[key]["max"])

    print("\nCertified-decrease interval (max <grad V, f> < 0, sigma_attr units):")
    for key in ["truth"] + setup_keys:
        iv = cert[key]
        if iv:
            print(f"  {key:>14s}:  [{iv[0]:>6.2f}, {iv[1]:>6.2f}]  width={iv[1]-iv[0]:.2f}")
        else:
            print(f"  {key:>14s}:  (none)")

    # ---- Trajectory overlays ---------------------------------------------
    print(f"\nTrajectory overlay: T={args.T_overlay}s (in-cloud + off-cloud ICs)")
    n_steps_ovl = int(round(args.T_overlay / args.dt))
    rng_ovl = np.random.default_rng(args.ic_rng_seed + 1)
    ic_in = A[rng_ovl.choice(A.shape[0])]
    ic_off = x_centroid + args.off_cloud_distance * sigma_attr * np.array([1.0, 0.0, 0.0])
    _, X_t_in = rk4_integrate(f_truth_l63_single, ic_in, args.dt, n_steps_ovl)
    _, X_t_off = rk4_integrate(f_truth_l63_single, ic_off, args.dt, n_steps_ovl)
    trajectories_in = {key: rk4_integrate(_make_f_single(models[key]),
                                          ic_in, args.dt, n_steps_ovl)[1]
                       for key in setup_keys}
    trajectories_off = {key: rk4_integrate(_make_f_single(models[key]),
                                            ic_off, args.dt, n_steps_ovl)[1]
                        for key in setup_keys}

    # ---- Figure 1: matched_metrics_4way ----------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    xs = np.arange(len(setup_keys))
    ax = axes[0]
    ax.bar(xs, [metrics[k]["T_ph_median"] for k in setup_keys],
           color=[setup_color[k] for k in setup_keys], edgecolor="k")
    ax.set_xticks(xs); ax.set_xticklabels([setup_label[k] for k in setup_keys],
                                          rotation=15, ha="right")
    ax.set_ylabel(r"$T_{\rm ph}$ median (s)")
    ax.set_title(rf"Prediction horizon  $n_{{\rm rbf}}={args.n_rbf}$")
    ax.grid(True, axis="y", ls=":", alpha=0.4)

    ax = axes[1]
    cap = 6.0
    R_vals = [min(cap, metrics[k]["R_abs"]) if np.isfinite(metrics[k]["R_abs"])
              else cap for k in setup_keys]
    ax.bar(xs, R_vals, color=[setup_color[k] for k in setup_keys], edgecolor="k")
    for i, k in enumerate(setup_keys):
        if not np.isfinite(metrics[k]["R_abs"]):
            ax.text(xs[i], cap + 0.05, "∞", ha="center", va="bottom", fontsize=11,
                    color=setup_color[k])
    ax.axhline(0.4, color="green", ls="--", lw=1.0, label="truth (0.40)")
    ax.set_xticks(xs); ax.set_xticklabels([setup_label[k] for k in setup_keys],
                                          rotation=15, ha="right")
    ax.set_ylabel(r"$R_{\rm abs}/\sigma_{\rm attr}$")
    ax.set_ylim(0, cap + 0.6)
    ax.set_title("Outer absorbing radius (lower better)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, axis="y", ls=":", alpha=0.4)

    ax = axes[2]
    w1_avg = [float(np.mean(metrics[k]["w1"])) for k in setup_keys]
    ax.bar(xs, w1_avg, color=[setup_color[k] for k in setup_keys], edgecolor="k")
    ax.set_xticks(xs); ax.set_xticklabels([setup_label[k] for k in setup_keys],
                                          rotation=15, ha="right")
    ax.set_ylabel(r"$\overline{W_1}$ (mean over $x,y,z$)")
    ax.set_title("Climate distance (lower better)")
    ax.grid(True, axis="y", ls=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_dir / "matched_metrics_4way.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'matched_metrics_4way.png'}")

    # ---- Figure 2: linear_block_recovery ---------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    truth_vals = {"sigma": SIGMA_L63, "rho": RHO_L63, "beta": BETA_L63}
    keys_lin = ["knn_linrbf", "attr_linrbf"]
    bar_groups = ["sigma", "rho", "beta"]
    x_groups = np.arange(len(bar_groups))
    width = 0.35
    for i, k in enumerate(keys_lin):
        meta = models[k]["meta"]
        vals = [meta["sigma_hat"], meta["rho_hat"], meta["beta_hat"]]
        ax.bar(x_groups + (i - 0.5) * width, vals, width,
               color=setup_color[k], edgecolor="k", label=setup_label[k])
    for g, name in enumerate(bar_groups):
        ax.axhline(0, color="grey", lw=0.6)
        ax.hlines(truth_vals[name],
                  x_groups[g] - width, x_groups[g] + width,
                  colors="green", linestyles="--", lw=1.2,
                  label="truth" if g == 0 else None)
    ax.set_xticks(x_groups)
    ax.set_xticklabels([r"$\hat\sigma$", r"$\hat\rho$", r"$\hat\beta$"])
    ax.set_ylabel("recovered coefficient")
    ax.set_title("Linear block recovery: linear + RBF fits at $n_{\\rm rbf}=3200$")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "linear_block_recovery.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'linear_block_recovery.png'}")

    # ---- Figure 3: trajectory_overlay_4way -------------------------------
    fig, axes = plt.subplots(4, 2, figsize=(11, 9), sharex="col", squeeze=False)
    t_axis = np.arange(n_steps_ovl + 1) * args.dt
    for r_idx, key in enumerate(setup_keys):
        for c_idx, (X_t, X_p, tag) in enumerate([
            (X_t_in,  trajectories_in[key],  "in-cloud IC"),
            (X_t_off, trajectories_off[key], f"off-cloud IC (+{args.off_cloud_distance}σ x̂)"),
        ]):
            ax = axes[r_idx, c_idx]
            ax.plot(t_axis[:X_t.shape[0]], X_t[:, 0], "k-", lw=1.0, alpha=0.5)
            ax.plot(t_axis[:X_t.shape[0]], X_t[:, 2], "k-", lw=1.0, alpha=0.5)
            ax.plot(t_axis[:X_p.shape[0]], X_p[:, 0],
                    color=setup_color[key], lw=1.1, label="x̂")
            ax.plot(t_axis[:X_p.shape[0]], X_p[:, 2],
                    color=setup_color[key], lw=1.1, ls=":", label="ẑ")
            if r_idx == 0:
                ax.set_title(tag)
            if c_idx == 0:
                ax.set_ylabel(setup_label[key], fontsize=9)
            if r_idx == 3:
                ax.set_xlabel("t (s)")
            ax.grid(True, ls=":", alpha=0.3)
    fig.suptitle(rf"Trajectory overlay (T={args.T_overlay} s); grey=truth (x,z)",
                 y=1.005)
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory_overlay_4way.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'trajectory_overlay_4way.png'}")

    # ---- Figure 4: dead_zone_offcloud ------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.0))
    R_t_truth = np.linalg.norm(X_t_off - x_centroid, axis=1) / sigma_attr
    ax.plot(np.arange(X_t_off.shape[0]) * args.dt, R_t_truth, "k-",
            lw=1.5, label="truth")
    for key in setup_keys:
        X_p = trajectories_off[key]
        R_t = np.linalg.norm(X_p - x_centroid, axis=1) / sigma_attr
        ax.plot(np.arange(X_p.shape[0]) * args.dt, R_t,
                color=setup_color[key], lw=1.4, ls=setup_ls[key],
                label=setup_label[key])
    ax.axhline(args.off_cloud_distance, color="grey", lw=0.7, ls=":",
               label=f"IC at +{args.off_cloud_distance}σ")
    ax.set_xlabel("t (s)")
    ax.set_ylabel(r"$R(t)/\sigma_{\rm attr}$")
    ax.set_title(f"Off-cloud IC: dead-zone vs linear-pull behaviour")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "dead_zone_offcloud.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'dead_zone_offcloud.png'}")

    # ---- Figure 5: lyapunov_certificate_4way -----------------------------
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    ax = axes[0]
    ax.plot(R_lyap, lyap["truth"]["max"], "k-", lw=1.5, label="truth")
    for key in setup_keys:
        ax.plot(R_lyap, lyap[key]["max"], color=setup_color[key],
                lw=1.3, ls=setup_ls[key], label=setup_label[key])
    ax.axhline(0, color="grey", lw=0.7)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylabel(r"$\max_{\hat d}\,\langle\nabla V, f\rangle$")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, ls=":", alpha=0.4)
    for key, col in [("truth", "k")] + [(k, setup_color[k]) for k in setup_keys]:
        iv = cert[key]
        if iv is not None:
            ax.axvspan(iv[0], iv[1], color=col, alpha=0.06, zorder=0)

    ax = axes[1]
    ax.plot(R_lyap, lyap["truth"]["mean"], "k-", lw=1.5)
    for key in setup_keys:
        ax.plot(R_lyap, lyap[key]["mean"], color=setup_color[key],
                lw=1.3, ls=setup_ls[key])
    ax.axhline(0, color="grey", lw=0.7)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylabel(r"mean $\langle\nabla V, f\rangle$")
    ax.grid(True, ls=":", alpha=0.4)

    ax = axes[2]
    ax.plot(R_lyap, lyap["truth"]["frac_neg"], "k-", lw=1.5)
    for key in setup_keys:
        ax.plot(R_lyap, lyap[key]["frac_neg"], color=setup_color[key],
                lw=1.3, ls=setup_ls[key])
    ax.axhline(1.0, color="green", lw=0.7, ls=":")
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel(r"frac($\langle\nabla V, f\rangle < 0$)")
    ax.set_xlabel(r"$R/\sigma_{\rm attr}$")
    ax.grid(True, ls=":", alpha=0.4)
    fig.suptitle(r"Lyapunov certificate; curve must stay below zero to certify decrease",
                 y=1.005)
    fig.tight_layout()
    fig.savefig(out_dir / "lyapunov_certificate_4way.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'lyapunov_certificate_4way.png'}")

    # ---- Figure 6: pdf_grid_4way -----------------------------------------
    pdf_n_steps = int(round(args.pdf_T / args.dt))
    tail_n_lo = int(round(args.pdf_tail_t0 / args.dt))
    rng_pdf = np.random.default_rng(args.pdf_ic_seed)
    pdf_ic_idx = rng_pdf.choice(A.shape[0], size=args.pdf_n_ic, replace=False)
    pdf_ic_idx.sort()
    pdf_ICs = A[pdf_ic_idx]
    print(f"\nMarginal-PDF run: {args.pdf_n_ic} ICs x T={args.pdf_T}s, "
          f"tail t>={args.pdf_tail_t0}s, dt={args.dt}")

    def _pool_tail(f_single):
        tails = []
        for ic in pdf_ICs:
            _, X = rk4_integrate(f_single, ic, args.dt, pdf_n_steps)
            if X.shape[0] > tail_n_lo:
                t = X[tail_n_lo:]
                tails.append(t[np.all(np.isfinite(t), axis=1)])
        return (np.concatenate(tails, axis=0)
                if tails else np.zeros((0, 3)))

    pdf_truth = _pool_tail(f_truth_l63_single)
    pdf_pools = {key: _pool_tail(_make_f_single(models[key])) for key in setup_keys}
    for key in setup_keys:
        print(f"  pooled tail {key:>14s}: {pdf_pools[key].shape[0]} samples")

    bins_per_axis = []
    axis_names = ["x", "y", "z"]
    for k in range(3):
        lo = float(pdf_truth[:, k].min())
        hi = float(pdf_truth[:, k].max())
        margin = 0.08 * (hi - lo)
        bins_per_axis.append(np.linspace(lo - margin, hi + margin, args.pdf_bins + 1))

    fig, axes = plt.subplots(4, 3, figsize=(11, 1.85 * 4),
                             sharex="col", squeeze=False)
    for r_idx, key in enumerate(setup_keys):
        pool = pdf_pools[key]
        for k in range(3):
            ax = axes[r_idx, k]
            ax.hist(pdf_truth[:, k], bins=bins_per_axis[k], density=True,
                    color="lightgrey", edgecolor="none")
            if pool.shape[0] > 0:
                ax.hist(pool[:, k], bins=bins_per_axis[k], density=True,
                        histtype="step", color=setup_color[key], lw=1.4)
            if r_idx == 0:
                ax.set_title(axis_names[k], fontsize=11)
            if r_idx == 3:
                ax.set_xlabel(axis_names[k])
            if k == 0:
                ax.set_ylabel(setup_label[key], fontsize=9)
            ax.grid(True, ls=":", alpha=0.3)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="lightgrey"),
        plt.Line2D([0], [0], color=setup_color["knn_rbf"], lw=1.5),
        plt.Line2D([0], [0], color=setup_color["attr_rbf"], lw=1.5),
        plt.Line2D([0], [0], color=setup_color["knn_linrbf"], lw=1.5, ls="--"),
        plt.Line2D([0], [0], color=setup_color["attr_linrbf"], lw=1.5, ls="--"),
    ]
    fig.legend(handles, ["FOM truth"] + [setup_label[k] for k in setup_keys],
               loc="upper center", ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        rf"L63 4-way marginal PDFs at $n_{{\rm rbf}}={args.n_rbf}$ — "
        rf"tail $t\in[{args.pdf_tail_t0:g},{args.pdf_T:g}]$ s, "
        rf"{args.pdf_n_ic} ICs",
        y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_dir / "pdf_grid_4way.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'pdf_grid_4way.png'}")

    # ---- Linrbf K/sigma ablation ----------------------------------------
    # Tests: (i) drop K from 3200 -> 800 for both knn and attr; (ii) narrow
    # attr at K=800 from alpha=1.0 to alpha=0.5. Question: does *either* knob
    # restore identifiability (rho_hat -> 28) and improve T_ph?
    abl_keys = [
        "knn_linrbf_K3200",
        "knn_linrbf_K800",
        "attr_linrbf_K3200_a1",
        "attr_linrbf_K800_a1",
        "attr_linrbf_K800_a0p5",
    ]
    abl_label = {
        "knn_linrbf_K3200":    r"knn lin+RBF, $K{=}3200$",
        "knn_linrbf_K800":     r"knn lin+RBF, $K{=}800$",
        "attr_linrbf_K3200_a1":r"attr lin+RBF, $K{=}3200,\alpha{=}1$",
        "attr_linrbf_K800_a1": r"attr lin+RBF, $K{=}800,\alpha{=}1$",
        "attr_linrbf_K800_a0p5": r"attr lin+RBF, $K{=}800,\alpha{=}0.5$",
    }
    abl_color = {
        "knn_linrbf_K3200":      "#1f77b4",
        "knn_linrbf_K800":       "#5fa8d3",
        "attr_linrbf_K3200_a1":  "#d62728",
        "attr_linrbf_K800_a1":   "#e57373",
        "attr_linrbf_K800_a0p5": "#f4a261",
    }
    abl_models = {
        "knn_linrbf_K3200":      models["knn_linrbf"],
        "attr_linrbf_K3200_a1":  models["attr_linrbf"],
        "knn_linrbf_K800":       _load_linrbf(
            base / "step1g_linrbf" / "model_linrbf_knn_n00800.npz"),
        "attr_linrbf_K800_a1":   _load_linrbf(
            base / "step1g_linrbf" / "model_linrbf_attr_n00800_a1.00.npz"),
        "attr_linrbf_K800_a0p5": _load_linrbf(
            base / "step1g_linrbf" / "model_linrbf_attr_n00800_a0.50.npz"),
    }
    print("\nLinrbf K/sigma ablation: computing T_ph for the 3 new K=800 fits...")
    abl_metrics = {
        "knn_linrbf_K3200":     {"T_ph_median": metrics["knn_linrbf"]["T_ph_median"]},
        "attr_linrbf_K3200_a1": {"T_ph_median": metrics["attr_linrbf"]["T_ph_median"]},
    }
    for key in ("knn_linrbf_K800", "attr_linrbf_K800_a1", "attr_linrbf_K800_a0p5"):
        m = abl_models[key]
        f_single = _make_f_single(m)
        horizons = []
        for j in range(args.n_ic_horizon):
            _, X_p = rk4_integrate(f_single, ICs[j], args.dt, n_steps_h)
            horizons.append(_eq21_horizon(X_p, X_true_h[j], args.dt, X_std_climate))
        abl_metrics[key] = {
            "T_ph_median": float(np.median(horizons)),
            "T_ph_min":    float(np.min(horizons)),
            "T_ph_max":    float(np.max(horizons)),
        }
        print(f"  {key:>26s}: T_ph_med={abl_metrics[key]['T_ph_median']:.2f}s")

    # Figure 7: linrbf_K_sigma_ablation -----------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    xs = np.arange(len(abl_keys))
    colors = [abl_color[k] for k in abl_keys]
    short_labels = ["knn\nK=3200", "knn\nK=800",
                    r"attr $\alpha$=1" "\nK=3200", r"attr $\alpha$=1" "\nK=800",
                    r"attr $\alpha$=0.5" "\nK=800"]

    # (a) rho_hat
    ax = axes[0]
    rho_vals = [float(abl_models[k]["meta"]["rho_hat"]) for k in abl_keys]
    ax.bar(xs, rho_vals, color=colors, edgecolor="k")
    ax.axhline(RHO_L63, color="green", ls="--", lw=1.2, label=rf"truth $\rho={RHO_L63}$")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xticks(xs); ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_ylabel(r"$\hat\rho$")
    ax.set_title("Identifiability target")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="y", ls=":", alpha=0.4)

    # (b) sigma_hat and beta_hat (grouped)
    ax = axes[1]
    width = 0.38
    sig_vals = [float(abl_models[k]["meta"]["sigma_hat"]) for k in abl_keys]
    bet_vals = [float(abl_models[k]["meta"]["beta_hat"])  for k in abl_keys]
    ax.bar(xs - width/2, sig_vals, width, color=colors, edgecolor="k",
           label=r"$\hat\sigma$")
    ax.bar(xs + width/2, bet_vals, width, color=colors, edgecolor="k",
           hatch="//", label=r"$\hat\beta$")
    ax.axhline(SIGMA_L63, color="green", ls="--", lw=1.0, alpha=0.6,
               label=rf"$\sigma={SIGMA_L63}$")
    ax.axhline(BETA_L63, color="purple", ls=":", lw=1.0, alpha=0.6,
               label=rf"$\beta={BETA_L63:.2f}$")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xticks(xs); ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_ylabel("recovered coef")
    ax.set_title(r"$\hat\sigma,\,\hat\beta$ recovery")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, axis="y", ls=":", alpha=0.4)

    # (c) median collinearity of active RBFs with poly block
    ax = axes[2]
    coll_med = [float(abl_models[k]["meta"]["collin_stats"]["median"])
                for k in abl_keys]
    coll_frac = [float(abl_models[k]["meta"]["collin_stats"]["frac_gt_0.5"])
                 for k in abl_keys]
    ax.bar(xs - width/2, coll_med,  width, color=colors, edgecolor="k",
           label="median")
    ax.bar(xs + width/2, coll_frac, width, color=colors, edgecolor="k",
           hatch="//", label=r"frac($>$0.5)")
    ax.axhline(0.5, color="grey", lw=0.7, ls=":")
    ax.set_xticks(xs); ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_ylabel(r"$\|P_{\rm poly}\phi_k\|/\|\phi_k\|$ over active $k$")
    ax.set_title("RBF/linear collinearity")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, axis="y", ls=":", alpha=0.4)

    # (d) T_ph median
    ax = axes[3]
    tph = [abl_metrics[k]["T_ph_median"] for k in abl_keys]
    ax.bar(xs, tph, color=colors, edgecolor="k")
    ax.set_xticks(xs); ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_ylabel(r"$T_{\rm ph}$ median (s)")
    ax.set_title("Prediction horizon (dynamics quality)")
    ax.grid(True, axis="y", ls=":", alpha=0.4)

    fig.suptitle("Linrbf K and σ ablation — does either knob restore "
                 r"identifiability? (truth: $\rho=28,\sigma=10,\beta=2.67$)",
                 y=1.04, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "linrbf_K_sigma_ablation.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'linrbf_K_sigma_ablation.png'}")

    abl_json = {
        k: {
            "sigma_hat": float(abl_models[k]["meta"]["sigma_hat"]),
            "rho_hat":   float(abl_models[k]["meta"]["rho_hat"]),
            "beta_hat":  float(abl_models[k]["meta"]["beta_hat"]),
            "kappa_final": float(abl_models[k]["meta"].get("kappa_final",
                                                           float("nan"))),
            "nnz_poly": int(abl_models[k]["meta"]["nnz_poly"]),
            "nnz_rbf":  int(abl_models[k]["meta"]["nnz_rbf"]),
            "n_rbf":    int(abl_models[k]["meta"].get("n_rbf",
                                                       abl_models[k]["centers"].shape[0])),
            "collin_stats": abl_models[k]["meta"]["collin_stats"],
            "T_ph_median": float(abl_metrics[k]["T_ph_median"]),
        }
        for k in abl_keys
    }

    # ---- Save JSON --------------------------------------------------------
    out_json = out_dir / "step1g_compare.json"
    out_json.write_text(json.dumps({
        "config": {
            "n_rbf": args.n_rbf, "alpha": args.alpha,
            "dt": args.dt, "T_overlay": args.T_overlay,
            "off_cloud_distance": args.off_cloud_distance,
            "sigma_attr": sigma_attr,
            "x_centroid": list(map(float, x_centroid)),
        },
        "metrics": {k: {**v,
                        "R_abs": ("inf" if not np.isfinite(v["R_abs"])
                                  else v["R_abs"])}
                    for k, v in metrics.items()},
        "linear_block_recovery": {
            k: {
                "sigma_hat": float(models[k]["meta"]["sigma_hat"]),
                "rho_hat":   float(models[k]["meta"]["rho_hat"]),
                "beta_hat":  float(models[k]["meta"]["beta_hat"]),
                "kappa_final": float(models[k]["meta"].get("kappa_final",
                                                           float("nan"))),
                "nnz_poly": int(models[k]["meta"]["nnz_poly"]),
                "nnz_rbf":  int(models[k]["meta"]["nnz_rbf"]),
                "collin_stats": models[k]["meta"]["collin_stats"],
            } for k in ["knn_linrbf", "attr_linrbf"]
        },
        "lyapunov": {
            "R_sigma": R_lyap.tolist(),
            **{key: {k: v.tolist() for k, v in lyap[key].items()}
               for key in ["truth"] + setup_keys},
            "certified_interval_sigma": {key: cert[key]
                                         for key in ["truth"] + setup_keys},
        },
        "linrbf_ablation": abl_json,
    }, indent=2))
    print(f"\nWrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
