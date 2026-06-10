"""Diagnostics on the aniso-vs-iso sweep saved fits.

Consumes the per-cell npz files from scripts/run_aniso_vs_iso_sweep.py and,
per (K, n_per_cluster) cell in parallel, runs:

  1. Dead-zone shell coverage (mandated in
     docs/journal/2026-06-10-anisotropic-rbf-per-cluster-design.md):
     for each cluster, sample N_shell uniformly on a sphere of radius
     d * s_k^(tangent_max) around the cluster centroid, and report the
     fraction with `max_j phi_j(x) < tau`. d-grid spans near to far in
     units of the tangent scale; tau grid {0.1, 0.3} matches the design
     entry. For K=1 isotropic the global PCA frame is used so the
     diagnostic is apples-to-apples across K.

  2. Forward RK4 integration from a shared IC set, T=50 s. Side products:
       - Eq.21 prediction horizon T_ph at HORIZON_FRAC * X_std_climate
         (mirrors results/LOR63/codes/step1_nrbf_sweep.py).
       - finite-at-T per IC (the stability floor).
       - Marginal PDFs on (x, y, z) and Wasserstein-1 vs truth climate
         over the integration tail.

  3. Condition number kappa(Phi) on the training mid-points - cheap
     diagnostic that explains the K=1 plateau (attr_scaled overlapping
     Gaussians on a 3-D state hit the float64 singularity floor; see
     docs/notes/l63-rbf-bandwidth-rules.md).

Outputs in results/<dataset>/aniso_vs_iso_diagnostics/:
  diagnostics.json
  pdf_marginals_{x,y,z}.png
  dead_zone_vs_d_tau{010,030}.png
  t_ph_vs_total.png
  w1_vs_total.png
  finite_at_T.png
  kappa_vs_total.png
  roll_K{K}_n{n_per_cluster}.npz   8 trajectories per cell, for replay.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance

from chord2 import data, sindy


SIGMA_L63 = 10.0
RHO_L63 = 28.0
BETA_L63 = 8.0 / 3.0
LYAPUNOV_L63 = 0.906
T_LYAPUNOV = 1.0 / LYAPUNOV_L63
HORIZON_FRAC = 0.5


_WORKER_STATE: dict = {}


def _f_truth_single(a):
    x, y, z = a[0], a[1], a[2]
    return np.array([
        SIGMA_L63 * (y - x),
        x * (RHO_L63 - z) - y,
        x * y - BETA_L63 * z,
    ])


def _rk4_integrate(f_callable, x0, dt, n_steps):
    """Fixed-step RK4 with non-finite early exit. Returns (t, X)."""
    r = x0.size
    X = np.empty((n_steps + 1, r))
    X[0] = x0
    n_done = 0
    for i in range(n_steps):
        a = X[i]
        if not np.all(np.isfinite(a)):
            break
        k1 = f_callable(a)
        if not np.all(np.isfinite(k1)):
            break
        k2 = f_callable(a + 0.5 * dt * k1)
        if not np.all(np.isfinite(k2)):
            break
        k3 = f_callable(a + 0.5 * dt * k2)
        if not np.all(np.isfinite(k3)):
            break
        k4 = f_callable(a + dt * k3)
        if not np.all(np.isfinite(k4)):
            break
        X[i + 1] = a + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        n_done = i + 1
    t = np.arange(n_done + 1) * dt
    return t, X[: n_done + 1]


def _eq21_horizon(X_pred, X_true, dt, X_std_climate):
    """Earliest time at which ||X_pred - X_true|| / X_std_climate exceeds
    HORIZON_FRAC. Returns the t in seconds (or the rollout duration if never)."""
    n = min(X_pred.shape[0], X_true.shape[0])
    if n < 2:
        return 0.0
    err = np.linalg.norm(X_pred[:n] - X_true[:n], axis=1) / X_std_climate
    over = np.where(err > HORIZON_FRAC)[0]
    if over.size == 0:
        return (n - 1) * dt
    return over[0] * dt


def _make_f_iso(fit):
    centers = fit["centers"]
    widths = fit["widths"]
    mu_A = fit["mu_A"]
    sigma_A = fit["sigma_A"]
    col_norms = fit["col_norms"]
    xi = fit["xi"]

    def f(a):
        A_row = a[None, :]
        Phi = sindy.rbf_features_iso(
            A_row, centers, widths, mu_A=mu_A, sigma_A=sigma_A,
        )
        return ((Phi / col_norms) @ xi).ravel()
    return f


def _make_f_aniso(fit):
    centers_std = fit["centers_std"]
    Sigma_invs_std = fit["Sigma_invs_std"]
    mu_A = fit["mu_A"]
    sigma_A = fit["sigma_A"]
    col_norms = fit["col_norms"]
    xi = fit["xi"]

    def f(a):
        A_row = a[None, :]
        Phi = sindy.rbf_features_mahal(
            A_row, centers_std, Sigma_invs_std,
            mu_A=mu_A, sigma_A=sigma_A,
        )
        return ((Phi / col_norms) @ xi).ravel()
    return f


def _phi_full(A, fit):
    """Evaluate the (un-normalised by col_norms) Phi at A for either mode."""
    if fit["mode"] == "iso":
        return sindy.rbf_features_iso(
            A, fit["centers"], fit["widths"],
            mu_A=fit["mu_A"], sigma_A=fit["sigma_A"],
        )
    return sindy.rbf_features_mahal(
        A, fit["centers_std"], fit["Sigma_invs_std"],
        mu_A=fit["mu_A"], sigma_A=fit["sigma_A"],
    )


def _cluster_frames(fit, A_train_std):
    """Per-cluster (centroid_std, V_k, lambda_k) for the shell diagnostic.

    For K=1 iso the saved meta has no V_k, so we compute one from the
    full training set's standardised covariance — the global PCA frame.
    """
    meta = fit["meta"]
    if fit["mode"] == "iso":
        Sigma_g = np.cov(A_train_std.T, bias=False)
        eigvals, eigvecs = np.linalg.eigh(Sigma_g)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        V = eigvecs[:, order]
        mu = A_train_std.mean(axis=0)
        return [(mu, V, eigvals)]
    out = []
    for k, V_k in enumerate(meta["V_per_k"]):
        if V_k is None:
            continue
        out.append((
            meta["centroid_per_k"][k],
            V_k,
            meta["lambda_per_k"][k],
        ))
    return out


def _dead_zone_fraction(fit, frames, d_grid, tau_grid, n_shell, rng):
    """For each (d, tau), fraction of shell points x where max_j phi_j(x) < tau.

    Shell radius is `d * sqrt(lambda^(1))` (the dominant tangent scale) in
    standardised coords, around each cluster centroid. Points are sampled
    isotropically on the unit sphere in R^r then scaled. Aggregated as the
    mean over clusters of the per-cluster dead-zone fraction.
    """
    r = fit["mu_A"].size
    dirs = rng.normal(size=(n_shell, r))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    out = {}
    for d in d_grid:
        per_cluster_dz = {tau: [] for tau in tau_grid}
        for (mu_std, V, eigvals) in frames:
            s_tan = float(np.sqrt(max(eigvals[0], 1e-30)))
            shell = mu_std[None, :] + d * s_tan * dirs
            # _phi_full expects un-standardised x with internal standardisation;
            # we already have std-coord points -> de-standardise for the call.
            shell_unstd = shell * fit["sigma_A"] + fit["mu_A"]
            Phi = _phi_full(shell_unstd, fit)
            max_per_pt = Phi.max(axis=1)
            for tau in tau_grid:
                frac = float((max_per_pt < tau).mean())
                per_cluster_dz[tau].append(frac)
        for tau in tau_grid:
            out[(float(d), float(tau))] = float(np.mean(per_cluster_dz[tau]))
    return out


def _kappa_phi(fit, A_mid):
    Phi = _phi_full(A_mid, fit)
    col_norms = np.linalg.norm(Phi, axis=0)
    col_norms = np.where(col_norms == 0.0, 1.0, col_norms)
    Phi_n = Phi / col_norms
    s = np.linalg.svd(Phi_n, compute_uv=False)
    if s.size == 0 or s[-1] <= 0.0:
        return float("inf")
    return float(s[0] / s[-1])


_NAME_RE = re.compile(r"^fit_K(\d+)_n(\d+)\.npz$")


def _list_cells(models_dir: Path) -> list[tuple[int, int, Path]]:
    cells = []
    for p in sorted(models_dir.glob("fit_K*_n*.npz")):
        m = _NAME_RE.match(p.name)
        if m:
            cells.append((int(m.group(1)), int(m.group(2)), p))
    return cells


def _load_fit(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    mode = str(z["mode"])
    fit = {
        "mode": mode,
        "K_shape": int(z["K_shape"]),
        "n_per_cluster": int(z["n_per_cluster"]),
        "mu_A": z["mu_A"], "sigma_A": z["sigma_A"],
        "col_norms": z["col_norms"], "xi": z["xi"],
        "nnz": int(z["nnz"]), "rel_err": float(z["rel_err"]),
        "meta": z["meta"].item(),
    }
    if mode == "iso":
        fit["centers"] = z["centers"]
        fit["widths"] = z["widths"]
    else:
        fit["centers_std"] = z["centers_std"]
        fit["Sigma_invs_std"] = z["Sigma_invs_std"]
        fit["parent_k"] = z["parent_k"]
    return fit


def _worker_init(A, A_mid, dt_data, ICs, X_true_all, X_std_climate,
                 out_dir_str, d_grid, tau_grid, n_shell, n_steps,
                 shell_seed):
    _WORKER_STATE["A"] = A
    _WORKER_STATE["A_mid"] = A_mid
    _WORKER_STATE["dt_data"] = dt_data
    _WORKER_STATE["ICs"] = ICs
    _WORKER_STATE["X_true_all"] = X_true_all
    _WORKER_STATE["X_std_climate"] = X_std_climate
    _WORKER_STATE["out_dir"] = Path(out_dir_str)
    _WORKER_STATE["d_grid"] = d_grid
    _WORKER_STATE["tau_grid"] = tau_grid
    _WORKER_STATE["n_shell"] = n_shell
    _WORKER_STATE["n_steps"] = n_steps
    _WORKER_STATE["shell_seed"] = shell_seed


def _worker_compute(cell):
    K, n_pc, path = cell
    t0 = time.time()
    fit = _load_fit(path)
    A = _WORKER_STATE["A"]
    A_mid = _WORKER_STATE["A_mid"]
    dt_data = _WORKER_STATE["dt_data"]
    ICs = _WORKER_STATE["ICs"]
    X_true_all = _WORKER_STATE["X_true_all"]
    X_std_climate = _WORKER_STATE["X_std_climate"]
    out_dir = _WORKER_STATE["out_dir"]
    d_grid = _WORKER_STATE["d_grid"]
    tau_grid = _WORKER_STATE["tau_grid"]
    n_shell = _WORKER_STATE["n_shell"]
    n_steps = _WORKER_STATE["n_steps"]
    shell_seed = _WORKER_STATE["shell_seed"]

    A_train_std = (A_mid - fit["mu_A"]) / fit["sigma_A"]
    frames = _cluster_frames(fit, A_train_std)

    rng = np.random.default_rng(shell_seed + K * 100 + n_pc)
    dz = _dead_zone_fraction(fit, frames, d_grid, tau_grid, n_shell, rng)

    kappa = _kappa_phi(fit, A_mid)

    f = _make_f_iso(fit) if fit["mode"] == "iso" else _make_f_aniso(fit)
    n_ic = ICs.shape[0]
    r = ICs.shape[1]
    rollouts = np.full((n_ic, n_steps + 1, r), np.nan)
    t_phs = np.zeros(n_ic)
    finite_at_T = np.zeros(n_ic, dtype=bool)
    for i in range(n_ic):
        _, X_pred = _rk4_integrate(f, ICs[i], dt_data, n_steps)
        rollouts[i, : X_pred.shape[0]] = X_pred
        t_phs[i] = _eq21_horizon(
            X_pred, X_true_all[i], dt_data, X_std_climate
        )
        finite_at_T[i] = (X_pred.shape[0] == n_steps + 1
                          and np.all(np.isfinite(X_pred[-1])))

    # Climate PDF: pool surviving rollouts past a burn-in.
    burn = max(1, int(0.2 * n_steps))
    samples_per_axis = {0: [], 1: [], 2: []}
    truth_per_axis = {0: [], 1: [], 2: []}
    for i in range(n_ic):
        X = rollouts[i, burn:]
        valid = np.all(np.isfinite(X), axis=1)
        if not valid.any():
            continue
        Xv = X[valid]
        for ax in (0, 1, 2):
            samples_per_axis[ax].append(Xv[:, ax])
        Xt = X_true_all[i, burn: burn + Xv.shape[0]]
        for ax in (0, 1, 2):
            truth_per_axis[ax].append(Xt[:, ax])

    w1 = {}
    pdf_samples = {}
    pdf_truth = {}
    for ax, label in enumerate(("x", "y", "z")):
        if samples_per_axis[ax]:
            s = np.concatenate(samples_per_axis[ax])
            t_ = np.concatenate(truth_per_axis[ax])
            w1[label] = float(wasserstein_distance(s, t_))
            pdf_samples[label] = s
            pdf_truth[label] = t_
        else:
            w1[label] = float("nan")
            pdf_samples[label] = np.array([])
            pdf_truth[label] = np.array([])

    roll_path = out_dir / f"roll_K{K}_n{n_pc}.npz"
    np.savez(
        roll_path,
        rollouts=rollouts, ICs=ICs, t_ph=t_phs, finite_at_T=finite_at_T,
        pdf_x=pdf_samples["x"], pdf_y=pdf_samples["y"], pdf_z=pdf_samples["z"],
    )

    row = {
        "K": K, "n_per_cluster": n_pc, "total_rbf": K * n_pc,
        "rel_err": fit["rel_err"], "nnz": fit["nnz"],
        "kappa_phi": kappa,
        "t_ph_mean": float(t_phs.mean()),
        "t_ph_min": float(t_phs.min()),
        "finite_at_T": int(finite_at_T.sum()),
        "n_ic": int(n_ic),
        "w1_x": w1["x"], "w1_y": w1["y"], "w1_z": w1["z"],
        "dead_zone": {f"d{d:g}_tau{tau:g}": v for (d, tau), v in dz.items()},
        "elapsed_s": time.time() - t0,
        "fit_path": str(path.name),
        "roll_path": roll_path.name,
    }
    return row


def _grid_panel_pdf(rows, out_dir, axis_label, pdf_key,
                    K_vals, n_vals, dataset, truth_samples):
    # Truth histogram (precomputed once) sets the x- and y-limits so every
    # panel can be read against the same reference. The figsize is scaled
    # so each panel has the aspect of the truth-PDF box (wider for x/y,
    # which have heavier tails than z).
    bins = 60
    t_lo = float(np.min(truth_samples))
    t_hi = float(np.max(truth_samples))
    pad = 0.02 * (t_hi - t_lo)
    xlim = (t_lo - pad, t_hi + pad)
    truth_density, truth_edges = np.histogram(
        truth_samples, bins=bins, range=xlim, density=True,
    )
    truth_centers = 0.5 * (truth_edges[:-1] + truth_edges[1:])
    y_max = float(truth_density.max()) * 1.25

    # Per-panel size matches the truth-PDF box aspect (x-support : y-peak),
    # clamped so panels stay legible across x/y/z.
    panel_w = 3.0
    box_aspect = y_max * (xlim[1] - xlim[0])
    panel_h = panel_w * float(np.clip(1.0 / max(box_aspect, 1e-6), 0.55, 1.4))
    fig, axes = plt.subplots(
        len(K_vals), len(n_vals),
        figsize=(panel_w * len(n_vals), panel_h * len(K_vals)),
        sharex=True, sharey=True,
    )
    if len(K_vals) == 1:
        axes = np.atleast_2d(axes)
    if len(n_vals) == 1:
        axes = axes.reshape(-1, 1)
    by_cell = {(r["K"], r["n_per_cluster"]): r["roll_path"] for r in rows}
    out_dir = Path(out_dir)
    for ik, K in enumerate(K_vals):
        for jn, n in enumerate(n_vals):
            ax = axes[ik, jn]
            roll = by_cell.get((K, n))
            if roll is None:
                ax.set_axis_off()
                continue
            ax.fill_between(truth_centers, truth_density, step="mid",
                            color="0.75", alpha=0.6, label="truth")
            ax.plot(truth_centers, truth_density, color="0.35",
                    linewidth=0.8, drawstyle="steps-mid")
            z = np.load(out_dir / roll)
            samp = z[pdf_key]
            if samp.size > 0:
                ax.hist(samp, bins=bins, range=xlim, density=True,
                        histtype="step", color="C0", linewidth=1.2,
                        label="pred")
            ax.set_xlim(xlim)
            ax.set_ylim(0.0, y_max)
            ax.set_title(f"K={K}, n={n}", fontsize=8)
            ax.tick_params(labelsize=7)
    # One legend for the whole figure (truth/pred are the same in every panel).
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=8,
                   bbox_to_anchor=(0.995, 0.985))
    fig.suptitle(f"{dataset} marginal PDF, axis {axis_label}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = out_dir / f"pdf_marginals_{axis_label}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _plot_dead_zone(rows, out_dir, d_grid, tau_grid):
    by_K_n = {(r["K"], r["n_per_cluster"]): r["dead_zone"] for r in rows}
    K_vals = sorted({r["K"] for r in rows})
    # One marker per n_per_cluster, colour from viridis so the n-ordering
    # reads at a glance. Reused across all (K, tau) panels.
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    out_dir = Path(out_dir)
    for K in K_vals:
        ns = sorted({r["n_per_cluster"] for r in rows if r["K"] == K})
        cmap = plt.get_cmap("viridis")
        colors = [cmap(0.15 + 0.7 * i / max(len(ns) - 1, 1))
                  for i in range(len(ns))]
        for tau in tau_grid:
            fig, ax = plt.subplots(figsize=(5.5, 4.0))
            for i, n in enumerate(ns):
                dz = by_K_n.get((K, n))
                if dz is None:
                    continue
                ys = [dz[f"d{d:g}_tau{tau:g}"] for d in d_grid]
                ax.plot(d_grid, ys,
                        marker=markers[i % len(markers)],
                        linestyle="-", color=colors[i],
                        markersize=7, linewidth=1.5,
                        label=f"n={n}")
            ax.set_xlabel(r"shell distance $d$ "
                          r"($\times \sqrt{\lambda_k^{(1)}}$)")
            ax.set_ylabel(rf"dead-zone fraction "
                          rf"($\max_j \phi_j < {tau:g}$)")
            mode_tag = "iso" if K == 1 else "aniso-PCA"
            ax.set_title(f"K={K} ({mode_tag}), "
                         rf"$\tau$={tau:g}")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, loc="best")
            fig.tight_layout()
            tag = f"{int(tau * 100):03d}"
            fig.savefig(out_dir / f"dead_zone_K{K}_tau{tag}.png", dpi=140)
            plt.close(fig)


def _plot_scalar_vs_total(rows, out_dir, key, ylabel, fname, log=True):
    by_K: dict[int, list[tuple[int, float]]] = {}
    for r in rows:
        by_K.setdefault(r["K"], []).append((r["total_rbf"], r[key]))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for K in sorted(by_K):
        pts = sorted(by_K[K])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        label = "K=1, iso" if K == 1 else f"K={K}, aniso-PCA"
        if log:
            ax.loglog(xs, ys, "o-", label=label)
        else:
            ax.semilogx(xs, ys, "o-", label=label)
    ax.set_xlabel("total RBFs")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(out_dir) / fname, dpi=140)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=str, default="LOR63")
    p.add_argument("--models-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--T-max", type=float, default=50.0)
    p.add_argument("--n-ic", type=int, default=8)
    p.add_argument("--ic-rng-seed", type=int, default=7)
    p.add_argument("--d-grid", type=float, nargs="+",
                   default=[0.5, 1.0, 2.0, 3.0])
    p.add_argument("--tau-grid", type=float, nargs="+",
                   default=[0.1, 0.3])
    p.add_argument("--n-shell", type=int, default=256)
    p.add_argument("--shell-seed", type=int, default=11)
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()

    ds = data.load(args.dataset)
    A = ds.u.astype(np.float64)
    dt_data = float(ds.t[1] - ds.t[0])
    A_mid, _ = sindy.deriv_5point(A, dt_data)
    X_std_climate = float(A.std(axis=0).mean())
    print(f"Loaded {args.dataset}: M={A.shape[0]}, r={A.shape[1]}, "
          f"dt={dt_data:g}, X_std_climate={X_std_climate:.4g}")

    models_dir = (args.models_dir
                  or (data.results_dir(args.dataset) / "aniso_vs_iso_sweep"))
    out_dir = (args.out_dir
               or (data.results_dir(args.dataset)
                   / "aniso_vs_iso_diagnostics"))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Models dir: {models_dir}")
    print(f"Output dir: {out_dir}")

    cells = _list_cells(models_dir)
    print(f"Discovered {len(cells)} cells.")
    if not cells:
        raise SystemExit("No fit_K*_n*.npz files in models-dir.")

    rng_ic = np.random.default_rng(args.ic_rng_seed)
    idx = rng_ic.choice(A.shape[0] - int(args.T_max / dt_data) - 1,
                        size=args.n_ic, replace=False)
    ICs = A[idx].copy()
    n_steps = int(args.T_max / dt_data)
    print(f"ICs from indices {idx.tolist()}; n_steps={n_steps}")

    X_true_all = np.empty((args.n_ic, n_steps + 1, A.shape[1]))
    for i in range(args.n_ic):
        _, Xt = _rk4_integrate(_f_truth_single, ICs[i], dt_data, n_steps)
        X_true_all[i, : Xt.shape[0]] = Xt
    print(f"Truth trajectories integrated.")

    init_args = (A, A_mid, dt_data, ICs, X_true_all, X_std_climate,
                 str(out_dir), list(args.d_grid), list(args.tau_grid),
                 int(args.n_shell), n_steps, int(args.shell_seed))

    rows: list[dict] = []
    if args.workers <= 1:
        _worker_init(*init_args)
        for cell in cells:
            row = _worker_compute(cell)
            rows.append(row)
            print(f"K={row['K']} n={row['n_per_cluster']:5d}  "
                  f"T_ph={row['t_ph_mean']:.3f}  finite={row['finite_at_T']}/{row['n_ic']}  "
                  f"kappa={row['kappa_phi']:.2e}  "
                  f"W1=({row['w1_x']:.2f},{row['w1_y']:.2f},{row['w1_z']:.2f})  "
                  f"elapsed={row['elapsed_s']:.1f}s", flush=True)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers, initializer=_worker_init,
                      initargs=init_args) as pool:
            for row in pool.imap_unordered(_worker_compute, cells):
                rows.append(row)
                print(f"K={row['K']} n={row['n_per_cluster']:5d}  "
                      f"T_ph={row['t_ph_mean']:.3f}  "
                      f"finite={row['finite_at_T']}/{row['n_ic']}  "
                      f"kappa={row['kappa_phi']:.2e}  "
                      f"W1=({row['w1_x']:.2f},{row['w1_y']:.2f},"
                      f"{row['w1_z']:.2f})  "
                      f"elapsed={row['elapsed_s']:.1f}s", flush=True)
    rows.sort(key=lambda r: (r["K"], r["n_per_cluster"]))

    summary_path = out_dir / "diagnostics.json"
    with summary_path.open("w") as f:
        json.dump({
            "dataset": args.dataset,
            "T_max": args.T_max, "n_ic": args.n_ic,
            "ic_rng_seed": args.ic_rng_seed,
            "ic_indices": idx.tolist(),
            "d_grid": list(args.d_grid), "tau_grid": list(args.tau_grid),
            "n_shell": args.n_shell, "shell_seed": args.shell_seed,
            "X_std_climate": X_std_climate,
            "cells": rows,
        }, f, indent=2)
    print(f"Summary -> {summary_path}")

    K_vals = sorted({r["K"] for r in rows})
    n_vals = sorted({r["n_per_cluster"] for r in rows})

    _plot_dead_zone(rows, out_dir, args.d_grid, args.tau_grid)
    _plot_scalar_vs_total(rows, out_dir, "t_ph_mean",
                          "mean T_ph (s)", "t_ph_vs_total.png", log=False)
    _plot_scalar_vs_total(rows, out_dir, "kappa_phi",
                          r"$\kappa(\Phi)$", "kappa_vs_total.png", log=True)
    # W1 per axis - one plot showing mean across axes for compactness.
    for r in rows:
        vals = [r["w1_x"], r["w1_y"], r["w1_z"]]
        r["w1_mean"] = float(np.mean([v for v in vals
                                       if np.isfinite(v)] or [np.nan]))
    _plot_scalar_vs_total(rows, out_dir, "w1_mean",
                          "mean W1 (x,y,z)", "w1_vs_total.png", log=True)
    _plot_scalar_vs_total(rows, out_dir, "finite_at_T",
                          f"finite-at-T (of {args.n_ic} ICs)",
                          "finite_at_T.png", log=False)
    # Truth marginals from the climate trajectory itself, past the same
    # 20% burn-in used inside the worker for the predicted samples.
    truth_burn = max(1, int(0.2 * A.shape[0]))
    truth_marginals = {
        "pdf_x": A[truth_burn:, 0],
        "pdf_y": A[truth_burn:, 1],
        "pdf_z": A[truth_burn:, 2],
    }
    for ax_label, key in zip("xyz", ("pdf_x", "pdf_y", "pdf_z")):
        _grid_panel_pdf(rows, out_dir, ax_label, key,
                        K_vals, n_vals, args.dataset,
                        truth_marginals[key])
    print(f"Plots written to {out_dir}")


if __name__ == "__main__":
    main()
