"""L63 step 1f: head-to-head comparison of bandwidth rules.

Compares (i) k-NN-per-centre Gaussians (`bandwidth_mode='knn_per_center'`,
from `results/LOR63/step1_nrbf_sweep/`) against (ii) attractor-scaled
fixed Gaussians (`bandwidth_mode='attr_scaled'`, from
`results/LOR63/step1e_gamma_sweep/`) at matched `n_rbf`.

Loads pre-computed metrics from the two sweeps' JSON outputs (no
re-fitting), loads the saved model .npz files, then re-integrates each
model from a small set of test ICs for the trajectory-overlay figure.

Outputs (under `results/LOR63/step1f_compare/`):
  - step1f_compare.json
  - matched_metrics.png  : grouped bars of T_ph, R_abs, W1 across n_rbf
  - radial_overlay.png   : R_inward_frac & ‖f‖_mean overplotted per n_rbf
  - trajectory_overlay.png : truth vs both ROMs from in-cloud + off-cloud IC
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "results" / "LOR63" / "codes"))

from chord2 import data  # noqa: E402
from _l63_rbf_lib import (  # noqa: E402
    f_rbf, f_truth_l63, f_truth_l63_single, rk4_integrate,
    SIGMA_L63, RHO_L63,
)


def _fib_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ], axis=1)


def _grad_V_L63(X: np.ndarray) -> np.ndarray:
    """Gradient of V(x,y,z) = rho*x^2 + sigma*y^2 + sigma*(z - 2*rho)^2."""
    return np.stack([
        2.0 * RHO_L63 * X[:, 0],
        2.0 * SIGMA_L63 * X[:, 1],
        2.0 * SIGMA_L63 * (X[:, 2] - 2.0 * RHO_L63),
    ], axis=1)


def _lyapunov_sweep(f_batch, x_c, sigma_attr, R_grid_sigma, d_hat):
    """For each R in R_grid_sigma, sample x = x_c + R*sigma_attr*d_hat,
    return (max, mean, frac_neg) of <grad V_L63, f(x)> over the sphere."""
    max_arr = np.empty(len(R_grid_sigma))
    mean_arr = np.empty(len(R_grid_sigma))
    frac_neg = np.empty(len(R_grid_sigma))
    for k, R in enumerate(R_grid_sigma):
        X = x_c[None, :] + R * sigma_attr * d_hat
        F = f_batch(X)
        gV = np.sum(_grad_V_L63(X) * F, axis=1)
        max_arr[k] = float(gV.max())
        mean_arr[k] = float(gV.mean())
        frac_neg[k] = float((gV < 0).mean())
    return max_arr, mean_arr, frac_neg


def _make_f_rbf_batch(model):
    c = model["centers"]; w = model["widths"]
    mu = model["mu_A"]; sg = model["sigma_A"]
    cn = model["col_norms"]; xi = model["xi"]
    return lambda X: f_rbf(X, c, w, mu, sg, cn, xi)


def _load_model(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    return {
        "centers": d["centers"], "widths": d["widths"],
        "mu_A": d["mu_A"], "sigma_A": d["sigma_A"],
        "col_norms": d["col_norms"], "xi": d["xi"],
        "active": d["active"],
        "meta": d["meta"].item() if "meta" in d.files else {},
    }


def _make_f_rbf(model):
    c = model["centers"]; w = model["widths"]
    mu = model["mu_A"]; sg = model["sigma_A"]
    cn = model["col_norms"]; xi = model["xi"]

    def f(a):
        return f_rbf(a.reshape(1, -1), c, w, mu, sg, cn, xi).ravel()
    return f


def _integrate(model, x0, dt, n_steps):
    f = _make_f_rbf(model)
    return rk4_integrate(f, x0, dt, n_steps)


def _integrate_truth(x0, dt, n_steps):
    return rk4_integrate(f_truth_l63_single, x0, dt, n_steps)


def _knn_R_abs(outer_json, n_rbf, thresh="0.60", floor="0.05"):
    for m in outer_json["models"]:
        if int(m["n_rbf"]) == int(n_rbf):
            v = m["R_abs"][thresh][floor]
            return float("inf") if v == "inf" else float(v)
    return None


def _attr_cell(step1e, n_rbf, alpha):
    for r in step1e["cells"]:
        if int(r["n_rbf"]) == int(n_rbf) and abs(float(r["alpha"]) - float(alpha)) < 1e-9:
            return r
    return None


def _knn_cell(step1, n_rbf):
    for r in step1["per_n_rbf"]:
        if int(r["n_rbf"]) == int(n_rbf):
            return r
    return None


def _knn_radial(risk, n_rbf):
    for m in risk["models"]:
        if int(m["n_rbf"]) == int(n_rbf):
            return m
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n-rbf-list", type=int, nargs="+", default=[100, 800, 3200])
    p.add_argument("--alpha-by-n", type=str, nargs="+",
                   default=["100:2.0", "800:1.0", "3200:1.0"],
                   help="pairs 'n:alpha' picking which attr_scaled cell to pair with each n")
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--T-overlay", type=float, default=8.0,
                   help="integration horizon for trajectory overlay (s)")
    p.add_argument("--off-cloud-distance", type=float, default=3.0,
                   help="off-cloud IC distance from centroid, units of sigma_attr")
    p.add_argument("--ic-rng-seed", type=int, default=11)
    p.add_argument("--lyap-r-min", type=float, default=0.2,
                   help="Lyapunov shell grid min, in sigma_attr units")
    p.add_argument("--lyap-r-max", type=float, default=12.0,
                   help="Lyapunov shell grid max, in sigma_attr units")
    p.add_argument("--lyap-r-n", type=int, default=30,
                   help="Lyapunov shell grid resolution (log-spaced)")
    p.add_argument("--lyap-n-sphere", type=int, default=600,
                   help="Fibonacci sphere directions for Lyapunov worst-case")
    p.add_argument("--pdf-n-ic", type=int, default=8,
                   help="ICs to pool for marginal-PDF estimate")
    p.add_argument("--pdf-ic-seed", type=int, default=0,
                   help="seed for the PDF IC selection (matches step1e default)")
    p.add_argument("--pdf-T", type=float, default=50.0,
                   help="integration horizon per IC for PDF estimate (s)")
    p.add_argument("--pdf-tail-t0", type=float, default=10.0,
                   help="strip transient: tail = t >= pdf_tail_t0")
    p.add_argument("--pdf-bins", type=int, default=80)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    alpha_by_n = {}
    for s in args.alpha_by_n:
        n_str, a_str = s.split(":")
        alpha_by_n[int(n_str)] = float(a_str)

    out_dir = args.out_dir or (data.results_dir("LOR63") / "step1f_compare")
    out_dir.mkdir(parents=True, exist_ok=True)

    base = Path("/mnt/scratch/users/sbrw610/CHORD2/results/LOR63")
    knn_dir = base / "step1_nrbf_sweep"
    attr_dir = base / "step1e_gamma_sweep"

    step1 = json.loads((knn_dir / "step1_nrbf_sweep.json").read_text())
    outer = json.loads((knn_dir / "risk_surface" / "outer_absorbing.json").read_text())
    risk = json.loads((knn_dir / "risk_surface" / "risk_surface.json").read_text())
    step1e = json.loads((attr_dir / "step1e_sweep.json").read_text())

    ds = data.load("LOR63")
    A = ds.u.astype(np.float64)
    sigma_attr = float(A.std(axis=0).mean())
    x_centroid = A.mean(axis=0)
    print(f"sigma_attr={sigma_attr:.4f}, x_centroid={x_centroid}")

    # ---- Tabulate metrics -------------------------------------------------
    rows = []
    for n in args.n_rbf_list:
        a = alpha_by_n[n]
        knn = _knn_cell(step1, n)
        attr = _attr_cell(step1e, n, a)
        if knn is None or attr is None:
            print(f"  [skip] n={n} alpha={a}: knn={knn is not None}, attr={attr is not None}")
            continue
        rows.append({
            "n_rbf": n,
            "alpha": a,
            "knn": {
                "rel_err": float(knn["rel_err_alpha_dot"]),
                "T_ph": float(knn["horizon_median"]),
                "R_abs": _knn_R_abs(outer, n),
                "w1": list(map(float, knn["w1"])),
            },
            "attr": {
                "rel_err": float(attr["rel_err_alpha_dot"]),
                "T_ph": float(attr["horizon_median"]),
                "R_abs": (float("inf") if attr["R_abs_canon"] == "inf"
                          else float(attr["R_abs_canon"])),
                "w1": list(map(float, attr["w1"])),
                "kappa_final": float(attr["kappa_final"]),
            },
        })

    print("\nMatched-n comparison:")
    print(f"{'n':>5s}  {'method':>10s}  {'T_ph':>6s}  {'R_abs':>6s}  "
          f"{'rel_err':>9s}  {'W1_x':>6s}  {'W1_y':>6s}  {'W1_z':>6s}")
    for r in rows:
        for label, key in [("knn", "knn"), (f"attr a={r['alpha']:g}", "attr")]:
            m = r[key]
            ra = "inf" if not np.isfinite(m["R_abs"]) else f"{m['R_abs']:.2f}"
            print(f"{r['n_rbf']:>5d}  {label:>10s}  {m['T_ph']:>6.2f}  "
                  f"{ra:>6s}  {m['rel_err']:>9.3e}  "
                  f"{m['w1'][0]:>6.3f}  {m['w1'][1]:>6.3f}  {m['w1'][2]:>6.3f}")

    # ---- Figure 1: matched_metrics ---------------------------------------
    n_pairs = len(rows)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    width = 0.35
    xs = np.arange(n_pairs)
    labels = [f"n={r['n_rbf']}\n(α={r['alpha']:g})" for r in rows]

    ax = axes[0]
    ax.bar(xs - width / 2, [r["knn"]["T_ph"] for r in rows], width,
           color="#1f77b4", edgecolor="k", label="knn-per-centre")
    ax.bar(xs + width / 2, [r["attr"]["T_ph"] for r in rows], width,
           color="#d62728", edgecolor="k", label="attr-scaled (best α)")
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_ylabel(r"$T_{\rm ph}$ (s)")
    ax.set_title("Prediction horizon (higher better)")
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    ax.legend(fontsize=8)

    ax = axes[1]
    cap = 6.0
    knn_R = [min(cap, r["knn"]["R_abs"]) if np.isfinite(r["knn"]["R_abs"]) else cap for r in rows]
    attr_R = [min(cap, r["attr"]["R_abs"]) if np.isfinite(r["attr"]["R_abs"]) else cap for r in rows]
    ax.bar(xs - width / 2, knn_R, width, color="#1f77b4", edgecolor="k")
    ax.bar(xs + width / 2, attr_R, width, color="#d62728", edgecolor="k")
    for k, (kr, ar) in enumerate(zip([r["knn"]["R_abs"] for r in rows],
                                     [r["attr"]["R_abs"] for r in rows])):
        if not np.isfinite(kr):
            ax.text(xs[k] - width / 2, cap + 0.05, "∞",
                    ha="center", va="bottom", fontsize=11, color="#1f77b4")
        if not np.isfinite(ar):
            ax.text(xs[k] + width / 2, cap + 0.05, "∞",
                    ha="center", va="bottom", fontsize=11, color="#d62728")
    ax.axhline(0.4, color="green", ls="--", lw=1.0, label="truth (0.40)")
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_ylabel(r"$R_{\rm abs}/\sigma_{\rm attr}$")
    ax.set_ylim(0, cap + 0.6)
    ax.set_title("Outer absorbing radius (lower better; ∞ = no tail)")
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[2]
    knn_w1 = np.array([r["knn"]["w1"] for r in rows])
    attr_w1 = np.array([r["attr"]["w1"] for r in rows])
    knn_w1_avg = knn_w1.mean(axis=1)
    attr_w1_avg = attr_w1.mean(axis=1)
    ax.bar(xs - width / 2, knn_w1_avg, width, color="#1f77b4", edgecolor="k")
    ax.bar(xs + width / 2, attr_w1_avg, width, color="#d62728", edgecolor="k")
    for k in range(n_pairs):
        ax.scatter([xs[k] - width / 2] * 3, knn_w1[k],
                   s=18, marker="o", color="black", zorder=3)
        ax.scatter([xs[k] + width / 2] * 3, attr_w1[k],
                   s=18, marker="o", color="black", zorder=3)
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_ylabel(r"$W_1$ (mean of x,y,z marginals)")
    ax.set_title("Climatology mismatch (lower better; dots = per-axis)")
    ax.grid(True, axis="y", ls=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_dir / "matched_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_dir / 'matched_metrics.png'}")

    # ---- Figure 2: radial_overlay ----------------------------------------
    fig, axes = plt.subplots(n_pairs, 1, figsize=(8.0, 2.4 * n_pairs),
                             sharex=True, squeeze=False)
    for i, r in enumerate(rows):
        ax = axes[i, 0]
        n = r["n_rbf"]; a = r["alpha"]
        knn_rad = _knn_radial(risk, n)
        attr_cell = _attr_cell(step1e, n, a)
        shell = knn_rad["shell_R"]
        ax.plot(shell, knn_rad["R_inward_frac"], "C0o-",
                label="knn $\\rho_{in}$", lw=1.4, ms=4)
        ax.plot(shell, attr_cell["R_inward_frac"], "C3s-",
                label=f"attr a={a:g} $\\rho_{{in}}$", lw=1.4, ms=4)
        ax.axhline(0.6, color="grey", lw=0.6, ls=":")
        ax.set_ylim(0, 1)
        ax.set_ylabel(r"$\rho_{\rm in}$", color="C0")
        ax2 = ax.twinx()
        knn_mag = knn_rad.get("rbf_mag_mean") or knn_rad.get("mag_mean")
        if knn_mag is not None:
            ax2.semilogy(shell, np.maximum(np.array(knn_mag, dtype=float), 1e-3),
                         "C0d--", lw=1.2, ms=3, alpha=0.7, label="knn $\\|f\\|$")
        ax2.semilogy(shell, np.maximum(attr_cell["mag_mean"], 1e-3),
                     "C3v--", lw=1.2, ms=3, alpha=0.7, label="attr $\\|f\\|$")
        floor_abs = 0.05 * 97.234  # f_truth_mean_mag from outer-abs config
        ax2.axhline(floor_abs, color="grey", lw=0.6, ls=":")
        ax2.set_ylim(1e-3, 1e3)
        ax2.set_ylabel(r"$\|f\|_{\rm mean}$", color="grey")
        ax.set_title(rf"$n_{{\rm rbf}}={n}$  (attr $\alpha={a:g}$)", fontsize=10)
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(True, ls=":", alpha=0.4)
    axes[-1, 0].set_xlabel(r"$R/\sigma_{\rm attr}$")
    fig.suptitle("Radial profile overlay: knn-per-centre vs attr-scaled", y=1.005)
    fig.tight_layout()
    fig.savefig(out_dir / "radial_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'radial_overlay.png'}")

    # ---- Figure 3: trajectory_overlay -----------------------------------
    rng = np.random.default_rng(args.ic_rng_seed)
    in_idx = int(rng.integers(0, A.shape[0]))
    ic_in = A[in_idx].copy()
    off_dir = np.array([0.0, 0.0, 1.0])  # along +z (above the upper lobe centroid)
    ic_off = x_centroid + args.off_cloud_distance * sigma_attr * off_dir
    print(f"\nin-cloud IC index={in_idx}, x={ic_in}")
    print(f"off-cloud IC: x={ic_off} (distance={args.off_cloud_distance}σ along +z)")
    n_steps = int(round(args.T_overlay / args.dt))

    t_truth_in, X_truth_in = _integrate_truth(ic_in, args.dt, n_steps)
    t_truth_off, X_truth_off = _integrate_truth(ic_off, args.dt, n_steps)

    knn_models = {n: _load_model(knn_dir / "models" / f"model_n_rbf_{n:05d}.npz")
                  for n in args.n_rbf_list}
    attr_models = {n: _load_model(attr_dir / "models"
                                  / f"model_n{n:05d}_a{alpha_by_n[n]:.2f}.npz")
                   for n in args.n_rbf_list}

    fig, axes = plt.subplots(2, n_pairs, figsize=(4.2 * n_pairs, 6.5),
                             squeeze=False)
    for j, n in enumerate(args.n_rbf_list):
        a = alpha_by_n[n]
        t_k_in, X_k_in = _integrate(knn_models[n], ic_in, args.dt, n_steps)
        t_a_in, X_a_in = _integrate(attr_models[n], ic_in, args.dt, n_steps)
        t_k_off, X_k_off = _integrate(knn_models[n], ic_off, args.dt, n_steps)
        t_a_off, X_a_off = _integrate(attr_models[n], ic_off, args.dt, n_steps)

        ax = axes[0, j]
        ax.plot(t_truth_in, X_truth_in[:, 2], "k-", lw=1.0, label="truth z")
        ax.plot(t_k_in, X_k_in[:, 2], "C0--", lw=1.2,
                label=f"knn z (n={X_k_in.shape[0] - 1})")
        ax.plot(t_a_in, X_a_in[:, 2], "C3-.", lw=1.2,
                label=f"attr z (n={X_a_in.shape[0] - 1})")
        ax.set_title(rf"in-cloud IC, $n_{{\rm rbf}}={n}$ (α={a:g})",
                     fontsize=10)
        ax.set_xlabel("t (s)"); ax.set_ylabel("z")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, ls=":", alpha=0.4)

        ax = axes[1, j]
        ax.plot(t_truth_off, np.linalg.norm(X_truth_off - x_centroid, axis=1) / sigma_attr,
                "k-", lw=1.0, label="truth")
        ax.plot(t_k_off, np.linalg.norm(X_k_off - x_centroid, axis=1) / sigma_attr,
                "C0--", lw=1.2, label="knn")
        ax.plot(t_a_off, np.linalg.norm(X_a_off - x_centroid, axis=1) / sigma_attr,
                "C3-.", lw=1.2, label="attr")
        ax.axhline(args.off_cloud_distance, color="grey", lw=0.6, ls=":",
                   label=f"start ({args.off_cloud_distance:g}σ)")
        ax.set_title(rf"off-cloud IC ({args.off_cloud_distance:g}σ along +z), "
                     rf"$n_{{\rm rbf}}={n}$", fontsize=10)
        ax.set_xlabel("t (s)"); ax.set_ylabel(r"$\|x - \bar x\|/\sigma_{\rm attr}$")
        ax.set_yscale("symlog", linthresh=0.5)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, ls=":", alpha=0.4)

    fig.suptitle("Trajectory overlay: truth vs knn-per-centre vs attr-scaled",
                 y=1.005)
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'trajectory_overlay.png'}")

    # ---- Figure 4: Lyapunov certificate ----------------------------------
    R_lyap = np.logspace(np.log10(args.lyap_r_min),
                         np.log10(args.lyap_r_max),
                         args.lyap_r_n)
    d_hat_lyap = _fib_sphere(args.lyap_n_sphere)
    print(f"\nLyapunov check: R/sigma in [{args.lyap_r_min:g}, {args.lyap_r_max:g}], "
          f"{args.lyap_r_n} shells, {args.lyap_n_sphere} directions")

    print("  truth ...")
    truth_max, truth_mean, truth_fn = _lyapunov_sweep(
        f_truth_l63, x_centroid, sigma_attr, R_lyap, d_hat_lyap)

    lyap_results = {"R_sigma": R_lyap.tolist(),
                    "truth": {"max": truth_max.tolist(),
                              "mean": truth_mean.tolist(),
                              "frac_neg": truth_fn.tolist()}}
    knn_results, attr_results = {}, {}
    for n in args.n_rbf_list:
        print(f"  knn n={n} ...")
        f_knn = _make_f_rbf_batch(knn_models[n])
        m, mn, fn = _lyapunov_sweep(f_knn, x_centroid, sigma_attr, R_lyap, d_hat_lyap)
        knn_results[n] = {"max": m, "mean": mn, "frac_neg": fn}
        print(f"  attr n={n} a={alpha_by_n[n]:g} ...")
        f_attr = _make_f_rbf_batch(attr_models[n])
        m, mn, fn = _lyapunov_sweep(f_attr, x_centroid, sigma_attr, R_lyap, d_hat_lyap)
        attr_results[n] = {"max": m, "mean": mn, "frac_neg": fn}

    def _certified_interval(R, max_arr):
        """Return (R_lo, R_hi) of the longest contiguous range where
        max_arr < 0 (worst-case Lyapunov certificate holds).
        Returns None if no such range exists.
        """
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

    fig, axes = plt.subplots(3, n_pairs, figsize=(4.4 * n_pairs, 8.5),
                             sharex=True, squeeze=False)
    for j, n in enumerate(args.n_rbf_list):
        a = alpha_by_n[n]
        max_t, mean_t, fn_t = truth_max, truth_mean, truth_fn
        max_k, mean_k, fn_k = (knn_results[n][k] for k in ("max", "mean", "frac_neg"))
        max_a, mean_a, fn_a = (attr_results[n][k] for k in ("max", "mean", "frac_neg"))

        ax = axes[0, j]
        ax.plot(R_lyap, max_t, "k-", lw=1.4, label="truth")
        ax.plot(R_lyap, max_k, "C0--", lw=1.4, label="knn")
        ax.plot(R_lyap, max_a, "C3-.", lw=1.4, label=f"attr a={a:g}")
        ax.axhline(0, color="grey", lw=0.7)
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_ylabel(r"$\max_{\hat d}\,\langle\nabla V, f\rangle$")
        ax.set_title(rf"$n_{{\rm rbf}}={n}$", fontsize=10)
        ax.grid(True, ls=":", alpha=0.4)
        ax.legend(fontsize=8, loc="best")
        for arr, col, lbl in [(max_t, "k", "truth"), (max_k, "C0", "knn"),
                              (max_a, "C3", "attr")]:
            iv = _certified_interval(R_lyap, arr)
            if iv is not None:
                ax.axvspan(iv[0], iv[1], color=col, alpha=0.08, zorder=0)

        ax = axes[1, j]
        ax.plot(R_lyap, mean_t, "k-", lw=1.4)
        ax.plot(R_lyap, mean_k, "C0--", lw=1.4)
        ax.plot(R_lyap, mean_a, "C3-.", lw=1.4)
        ax.axhline(0, color="grey", lw=0.7)
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_ylabel(r"mean $\langle\nabla V, f\rangle$")
        ax.grid(True, ls=":", alpha=0.4)

        ax = axes[2, j]
        ax.plot(R_lyap, fn_t, "k-", lw=1.4)
        ax.plot(R_lyap, fn_k, "C0--", lw=1.4)
        ax.plot(R_lyap, fn_a, "C3-.", lw=1.4)
        ax.axhline(1.0, color="green", lw=0.7, ls=":",
                   label="all-directions inward")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(r"frac($\langle\nabla V, f\rangle < 0$)")
        ax.set_xlabel(r"$R/\sigma_{\rm attr}$")
        ax.set_xscale("log")
        ax.grid(True, ls=":", alpha=0.4)
        if j == 0:
            ax.legend(fontsize=8, loc="lower left")

    fig.suptitle(r"Lyapunov certificate with $V = \rho x^2 + \sigma y^2 + \sigma (z - 2\rho)^2$"
                 "  — curve must stay below zero to certify decrease",
                 y=1.005)
    fig.tight_layout()
    fig.savefig(out_dir / "lyapunov_certificate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'lyapunov_certificate.png'}")

    # ---- Figure 5: marginal PDFs from 50-s runs --------------------------
    pdf_n_steps = int(round(args.pdf_T / args.dt))
    tail_n_lo = int(round(args.pdf_tail_t0 / args.dt))
    rng_pdf = np.random.default_rng(args.pdf_ic_seed)
    pdf_ic_idx = rng_pdf.choice(A.shape[0], size=args.pdf_n_ic, replace=False)
    pdf_ic_idx.sort()
    pdf_ICs = A[pdf_ic_idx]
    print(f"\nMarginal-PDF run: {args.pdf_n_ic} ICs x T={args.pdf_T}s, "
          f"tail t>={args.pdf_tail_t0}s, dt={args.dt}")

    def _pool_tail(f_batch_or_single, batch):
        tails = []
        for ic in pdf_ICs:
            _, X = rk4_integrate(f_batch_or_single, ic, args.dt, pdf_n_steps)
            if X.shape[0] > tail_n_lo:
                t = X[tail_n_lo:]
                tails.append(t[np.all(np.isfinite(t), axis=1)])
        return (np.concatenate(tails, axis=0)
                if tails else np.zeros((0, 3)))

    pdf_truth = _pool_tail(f_truth_l63_single, False)
    pdf_pools = {}
    for n in args.n_rbf_list:
        a = alpha_by_n[n]
        f_knn = _make_f_rbf_batch(knn_models[n])
        f_attr = _make_f_rbf_batch(attr_models[n])
        # rk4_integrate expects a (r,) -> (r,) callable: wrap.
        f_knn_single = lambda x, fb=f_knn: fb(x.reshape(1, -1)).ravel()
        f_attr_single = lambda x, fb=f_attr: fb(x.reshape(1, -1)).ravel()
        pdf_pools[("knn", n)] = _pool_tail(f_knn_single, False)
        pdf_pools[("attr", n)] = _pool_tail(f_attr_single, False)
        print(f"  pooled tail: knn n={n}: {pdf_pools[('knn', n)].shape[0]} samples; "
              f"attr n={n} a={a:g}: {pdf_pools[('attr', n)].shape[0]} samples")

    pdf_setups = []
    for n in args.n_rbf_list:
        pdf_setups.append(("knn", n, rf"knn  $n={n}$"))
        pdf_setups.append(("attr", n, rf"attr  $n={n}$, $\alpha={alpha_by_n[n]:g}$"))

    bins_per_axis = []
    axis_names = ["x", "y", "z"]
    for k in range(3):
        lo = float(pdf_truth[:, k].min())
        hi = float(pdf_truth[:, k].max())
        margin = 0.08 * (hi - lo)
        bins_per_axis.append(np.linspace(lo - margin, hi + margin, args.pdf_bins + 1))

    n_setups = len(pdf_setups)
    fig, axes = plt.subplots(n_setups, 3, figsize=(11, 1.7 * n_setups),
                             sharex="col", squeeze=False)
    for r, (method, n, label) in enumerate(pdf_setups):
        pool = pdf_pools[(method, n)]
        for k in range(3):
            ax = axes[r, k]
            ax.hist(pdf_truth[:, k], bins=bins_per_axis[k],
                    density=True, color="lightgrey", edgecolor="none",
                    label="FOM truth")
            if pool.shape[0] > 0:
                ax.hist(pool[:, k], bins=bins_per_axis[k],
                        density=True, histtype="step", color="C3", lw=1.4,
                        label=method)
            if r == 0:
                ax.set_title(axis_names[k], fontsize=11)
            if r == n_setups - 1:
                ax.set_xlabel(axis_names[k])
            if k == 0:
                ax.set_ylabel(label, fontsize=9)
            ax.grid(True, ls=":", alpha=0.3)
    # one legend at top
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="lightgrey"),
        plt.Line2D([0], [0], color="C3", lw=1.5),
    ]
    fig.legend(handles, ["FOM truth", "RBF-only"], loc="upper center",
               ncol=2, fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, 1.0 - 0.5 / (n_setups + 1.5)))
    fig.suptitle(
        rf"L63 RBF-only marginal PDFs — tail $t\in[{args.pdf_tail_t0:g},{args.pdf_T:g}]$ s,"
        rf" {args.pdf_n_ic} ICs pooled",
        y=1.0 - 0.15 / (n_setups + 1.5))
    fig.tight_layout(rect=[0, 0, 1, 1.0 - 0.6 / (n_setups + 1.5)])
    fig.savefig(out_dir / "pdf_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'pdf_grid.png'}")

    print("\nCertified-decrease interval (max-curve < 0, in sigma units):")
    print(f"{'n':>5s}  {'method':>10s}  {'R_lo':>6s}  {'R_hi':>6s}  {'width':>6s}")
    iv = _certified_interval(R_lyap, truth_max)
    iv_str = (f"  {iv[0]:>6.2f}  {iv[1]:>6.2f}  {iv[1] - iv[0]:>6.2f}"
              if iv else "  (none)")
    print(f"{'truth':>5s}  {'-':>10s}{iv_str}")
    for n in args.n_rbf_list:
        for label, arr in [("knn", knn_results[n]["max"]),
                           (f"attr a={alpha_by_n[n]:g}", attr_results[n]["max"])]:
            iv = _certified_interval(R_lyap, arr)
            iv_str = (f"  {iv[0]:>6.2f}  {iv[1]:>6.2f}  {iv[1] - iv[0]:>6.2f}"
                      if iv else "  (none)")
            print(f"{n:>5d}  {label:>10s}{iv_str}")

    # ---- Save JSON --------------------------------------------------------
    out_json = out_dir / "step1f_compare.json"
    out_json.write_text(json.dumps({
        "config": {
            "n_rbf_list": args.n_rbf_list,
            "alpha_by_n": {str(k): v for k, v in alpha_by_n.items()},
            "dt": args.dt, "T_overlay": args.T_overlay,
            "off_cloud_distance": args.off_cloud_distance,
            "sigma_attr": sigma_attr,
            "x_centroid": list(map(float, x_centroid)),
        },
        "rows": [
            {**r, "knn": {**r["knn"],
                          "R_abs": "inf" if not np.isfinite(r["knn"]["R_abs"]) else r["knn"]["R_abs"]},
             "attr": {**r["attr"],
                      "R_abs": "inf" if not np.isfinite(r["attr"]["R_abs"]) else r["attr"]["R_abs"]}}
            for r in rows
        ],
        "lyapunov": {
            "config": {
                "V": "rho*x**2 + sigma*y**2 + sigma*(z - 2*rho)**2",
                "r_min_sigma": args.lyap_r_min,
                "r_max_sigma": args.lyap_r_max,
                "r_n": args.lyap_r_n,
                "n_sphere": args.lyap_n_sphere,
            },
            "R_sigma": R_lyap.tolist(),
            "truth": lyap_results["truth"],
            "knn": {str(n): {k: v.tolist() for k, v in knn_results[n].items()}
                    for n in args.n_rbf_list},
            "attr": {str(n): {k: v.tolist() for k, v in attr_results[n].items()}
                     for n in args.n_rbf_list},
            "certified_interval_sigma": {
                "truth": _certified_interval(R_lyap, truth_max),
                **{f"knn_n{n}": _certified_interval(R_lyap, knn_results[n]["max"])
                   for n in args.n_rbf_list},
                **{f"attr_n{n}_a{alpha_by_n[n]:g}":
                       _certified_interval(R_lyap, attr_results[n]["max"])
                   for n in args.n_rbf_list},
            },
        },
    }, indent=2))
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
