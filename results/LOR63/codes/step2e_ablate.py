"""L63 step 2e: lambda_rbf x sigma_A-pinning ablation for clustered RBF SINDy.

Stage A at N_tot=1280 fit one operating point per K. Stage B integrated those
fits and showed K>=2 collapses to a quasi-fixed point in a numerical dead zone.
Step 2d (Tests C, D) traced the failure to three concurrent mechanisms:
  S1 sigma_A shrinkage  -> Gaussian decay rate scales with per-cluster std.
  S2 Voronoi boundaries -> field discontinuities at cluster edges.
  S3 small-m overfitting -> ||xi_k|| / sqrt(n_rbf) explodes for small clusters.

The rom-specialist consult of 2026-06-09 picked two ablations:
  Test 1 (lambda sweep): retighten STLSQ pruning. lambda_rbf in
         {1e-3, 1.0, 1e3, 1e5, 1e7}. The Stage A point used 1e-3.
  Test 2 (sigma_A pinned): replace per-cluster sigma_A by the global
         training std before standardising RBF centres and feature evals;
         keep mu_A per-cluster (the consult kept the center shift but
         globalised the scale). This kills S1 by construction.

This driver runs both sweeps in one parallel pass via mp.Pool over the
4 K x 5 lambda x 2 sigma_modes = 40-task grid. Each task is a from-scratch
clustered fit + inline diagnostics; nothing is loaded from Stage A.

Per task:
  * Outer K-means on (x, y, z) snapshots; centroids & counts cached per K.
  * RBF allocation: count-proportional for K=1; variance-proportional for
    K>=2 (Stage A's winning rule).
  * Per cluster fit via _l63_rbf_lib.fit_rbf_only_keep_state with the
    appropriate (lambda_rbf, mu_A_override, sigma_A_override) settings.
  * Score: train alpha_dot RMSE, test alpha_dot RMSE (held-out trajectory),
    dead-zone fraction on a coarse z-slice grid, ||xi||/sqrt(n_rbf)
    statistics across modelled clusters, sigma_A_min across clusters.

Outputs (results/LOR63/step2e_ablate/):
  step2e.json
  alpha_dot_vs_lambda.png      -- test RMSE vs lambda, panels per sigma_mode
  dead_zone_vs_lambda.png      -- dead-zone fraction vs lambda
  xi_norm_vs_lambda.png        -- median ||xi_k||/sqrt(n_rbf) vs lambda
  combined.png                 -- 3x2 grid summary
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

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _l63_rbf_lib import (  # noqa: E402
    f_rbf, f_truth_l63_single, fit_rbf_only_keep_state,
)

from chord2 import clustering, data, sindy  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _allocate_proportional(weights: np.ndarray, counts: np.ndarray,
                           n_tot: int) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    if w.sum() <= 0.0 or n_tot <= 0:
        return np.zeros_like(w, dtype=int)
    raw = n_tot * w / w.sum()
    floor = np.floor(raw).astype(int)
    remainder = int(n_tot - floor.sum())
    if remainder > 0:
        frac = raw - floor
        idx = np.argsort(-frac)[:remainder]
        floor[idx] += 1
    return np.minimum(floor, counts.astype(int))


def f_truth_grid(P: np.ndarray) -> np.ndarray:
    """Vectorised Lorenz-63 RHS on a (M, 3) batch."""
    SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    return np.column_stack([
        SIGMA * (y - x),
        x * (RHO - z) - y,
        x * y - BETA * z,
    ])


def assign_grid(P: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Hard nearest-centroid affiliation on a (M, 3) batch."""
    d2 = ((P[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(d2, axis=1)


def f_rbf_batch(P: np.ndarray, fit: dict) -> np.ndarray:
    """Cluster RBF field on a (M, 3) batch."""
    diffs_std = (P[:, None, :] - fit["centers"][None, :, :]) \
                / fit["sigma_A"][None, None, :]
    d2 = (diffs_std * diffs_std).sum(axis=2)
    phi = np.exp(-0.5 * d2 / (fit["widths"][None, :] ** 2))
    phi_n = phi / fit["col_norms"][None, :]
    return phi_n @ fit["xi"]


def f_rom_grid(P: np.ndarray, centroids: np.ndarray, K: int,
               models: list) -> tuple:
    labels = assign_grid(P, centroids)
    F = np.zeros_like(P)
    for k in range(K):
        if models[k] is None:
            continue
        mask = labels == k
        if not np.any(mask):
            continue
        F[mask] = f_rbf_batch(P[mask], models[k])
    return F, labels


# ---------------------------------------------------------------------------
# Fit + score
# ---------------------------------------------------------------------------

def _fit_clustered(A_mid: np.ndarray, dAdt: np.ndarray,
                   labels: np.ndarray, K: int, n_per_k: np.ndarray,
                   *, seed: int, lambda_rbf: float, gamma: float,
                   n_init: int, sigma_mode: str,
                   sigma_A_global: np.ndarray) -> list:
    """Fit one RBF-only SINDy per cluster, with optional sigma_A pinning.

    sigma_mode='cluster': default; sigma_A computed per-cluster from the
        cluster's own training rows.
    sigma_mode='global':  override sigma_A with the global training std
        (computed once, passed in). mu_A stays per-cluster -- the
        rom-specialist consult kept the centring per-cluster so that
        clusters with very different centroids do not lose their local
        coordinate origin.
    """
    models = []
    for k in range(K):
        n_k = int(n_per_k[k])
        if n_k <= 0:
            models.append(None)
            continue
        mask = labels == k
        if int(mask.sum()) < max(5, n_k):
            models.append(None)
            continue
        sigma_override = sigma_A_global if sigma_mode == "global" else None
        fit = fit_rbf_only_keep_state(
            A_mid[mask], dAdt[mask],
            n_rbf=n_k, seed=seed,
            lambda_rbf=lambda_rbf, gamma=gamma,
            n_knn=5, n_init=n_init,
            bandwidth_mode="attr_scaled",
            lambda_tikh=0.0,
            log_diagnostics=False,
            sigma_A_override=sigma_override,
        )
        models.append(fit)
    return models


def _score_rmse(A: np.ndarray, dAdt_true: np.ndarray,
                labels: np.ndarray, K: int, models: list) -> float:
    err_sq = 0.0
    total_norm_sq = float((dAdt_true * dAdt_true).sum())
    for k in range(K):
        mask = labels == k
        if not np.any(mask):
            continue
        m = models[k]
        if m is None:
            err_sq += float((dAdt_true[mask] * dAdt_true[mask]).sum())
            continue
        pred = f_rbf(A[mask], m["centers"], m["widths"],
                     m["mu_A"], m["sigma_A"], m["col_norms"], m["xi"])
        diff = dAdt_true[mask] - pred
        err_sq += float((diff * diff).sum())
    return float(np.sqrt(err_sq / max(total_norm_sq, 1e-30)))


def _grid_diagnostics(centroids: np.ndarray, K: int, models: list,
                      z_slice: float, A_train: np.ndarray,
                      n_grid: int = 100) -> dict:
    """Coarse 2D field diagnostics at z=z_slice. Dead-zone + near-training err."""
    xs = np.linspace(-25.0, 25.0, n_grid)
    ys = np.linspace(-30.0, 30.0, n_grid)
    Xg, Yg = np.meshgrid(xs, ys, indexing="xy")
    P = np.column_stack([Xg.ravel(), Yg.ravel(),
                         z_slice * np.ones(Xg.size)])
    F_truth = f_truth_grid(P)
    F_rom, _ = f_rom_grid(P, centroids, K, models)
    err = np.linalg.norm(F_rom - F_truth, axis=1)
    rom_mag = np.linalg.norm(F_rom, axis=1)
    dead_frac = float(np.mean(rom_mag < 1e-3))

    # Near-training mask: 2-unit radius around any snapshot in the z-band.
    z_band = np.abs(A_train[:, 2] - z_slice) < 1.5
    samples_in_band = A_train[z_band]
    in_band_grid = np.zeros(Xg.size, dtype=bool)
    if samples_in_band.size > 0:
        diff_x = P[:, 0:1] - samples_in_band[:, 0]
        diff_y = P[:, 1:2] - samples_in_band[:, 1]
        d2 = diff_x * diff_x + diff_y * diff_y
        in_band_grid = (d2.min(axis=1) < 4.0)
    err_inside = (float(np.median(err[in_band_grid]))
                  if in_band_grid.any() else float("nan"))
    err_outside = (float(np.median(err[~in_band_grid]))
                   if (~in_band_grid).any() else float("nan"))
    return {
        "dead_zone_frac": dead_frac,
        "err_near_training": err_inside,
        "err_off_training": err_outside,
    }


def _xi_norm_stats(models: list) -> dict:
    """||xi_k||_F / sqrt(n_rbf_k) summary across modelled clusters."""
    vals = []
    sigma_A_min = []
    for m in models:
        if m is None:
            continue
        n = int(m["xi"].shape[0])
        if n <= 0:
            continue
        vals.append(float(np.linalg.norm(m["xi"]) / np.sqrt(n)))
        sigma_A_min.append(float(m["sigma_A"].min()))
    if not vals:
        return {"xi_per_rbf_median": float("nan"),
                "xi_per_rbf_max": float("nan"),
                "sigma_A_min_min": float("nan"),
                "n_modelled": 0}
    return {
        "xi_per_rbf_median": float(np.median(vals)),
        "xi_per_rbf_max":    float(np.max(vals)),
        "sigma_A_min_min":   float(np.min(sigma_A_min)),
        "n_modelled":        int(len(vals)),
    }


def _nnz_total(models: list) -> int:
    return int(sum(m["nnz"] for m in models if m is not None))


# ---------------------------------------------------------------------------
# Pool worker
# ---------------------------------------------------------------------------

_WORKER_STATE: dict = {}


def _worker_init(A_mid: np.ndarray, dAdt: np.ndarray,
                 A_test: np.ndarray, dAdt_test: np.ndarray,
                 partitions: dict, sigma_A_global: np.ndarray,
                 fit_kwargs: dict, z_slice: float, n_grid_diag: int) -> None:
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _WORKER_STATE["A_mid"] = A_mid
    _WORKER_STATE["dAdt"] = dAdt
    _WORKER_STATE["A_test"] = A_test
    _WORKER_STATE["dAdt_test"] = dAdt_test
    _WORKER_STATE["partitions"] = partitions
    _WORKER_STATE["sigma_A_global"] = sigma_A_global
    _WORKER_STATE["fit_kwargs"] = fit_kwargs
    _WORKER_STATE["z_slice"] = z_slice
    _WORKER_STATE["n_grid_diag"] = n_grid_diag


def _run_task(task: tuple) -> dict:
    K, lambda_rbf, sigma_mode = task
    A_mid = _WORKER_STATE["A_mid"]
    dAdt = _WORKER_STATE["dAdt"]
    A_test = _WORKER_STATE["A_test"]
    dAdt_test = _WORKER_STATE["dAdt_test"]
    part = _WORKER_STATE["partitions"][K]
    sigma_A_global = _WORKER_STATE["sigma_A_global"]
    fk = _WORKER_STATE["fit_kwargs"]
    z_slice = _WORKER_STATE["z_slice"]
    n_grid_diag = _WORKER_STATE["n_grid_diag"]

    labels_train = part["labels_train"]
    labels_test = part["labels_test"]
    counts = part["counts"]
    centroids = part["centroids"]
    weights = (counts.astype(np.float64) if K == 1
               else part["var_weights"])
    n_per_k = _allocate_proportional(weights, counts, fk["n_tot"])

    t0 = time.time()
    models = _fit_clustered(
        A_mid, dAdt, labels_train, K, n_per_k,
        seed=fk["seed"], lambda_rbf=lambda_rbf, gamma=fk["gamma"],
        n_init=fk["n_init"], sigma_mode=sigma_mode,
        sigma_A_global=sigma_A_global,
    )
    t_fit = time.time() - t0

    rmse_train = _score_rmse(A_mid, dAdt, labels_train, K, models)
    rmse_test = _score_rmse(A_test, dAdt_test, labels_test, K, models)
    grid_diag = _grid_diagnostics(centroids, K, models, z_slice, A_mid,
                                  n_grid=n_grid_diag)
    xi_stats = _xi_norm_stats(models)

    return {
        "K": int(K),
        "lambda_rbf": float(lambda_rbf),
        "sigma_mode": sigma_mode,
        "alloc": "count" if K == 1 else "variance",
        "n_per_cluster": n_per_k.tolist(),
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
        "dead_zone_frac": grid_diag["dead_zone_frac"],
        "err_near_training": grid_diag["err_near_training"],
        "err_off_training": grid_diag["err_off_training"],
        "xi_per_rbf_median": xi_stats["xi_per_rbf_median"],
        "xi_per_rbf_max":    xi_stats["xi_per_rbf_max"],
        "sigma_A_min_min":   xi_stats["sigma_A_min_min"],
        "n_modelled":        xi_stats["n_modelled"],
        "nnz_total":         _nnz_total(models),
        "t_fit_seconds":     float(t_fit),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--k-grid", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--lambda-grid", type=float, nargs="+",
                   default=[1e-3, 1.0, 1e3, 1e5, 1e7],
                   help="STLSQ threshold sweep")
    p.add_argument("--sigma-modes", type=str, nargs="+",
                   default=["cluster", "global"])
    p.add_argument("--n-tot", type=int, default=1280)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--kmeans-seed", type=int, default=0)
    p.add_argument("--test-ic-seed", type=int, default=7)
    p.add_argument("--T-test", type=float, default=20.0)
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--n-init", type=int, default=1)
    p.add_argument("--n-init-outer", type=int, default=10)
    p.add_argument("--n-grid-diag", type=int, default=100,
                   help="2D diagnostic grid resolution (cheap; per task)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    ds = data.load("LOR63")
    A = ds.u.astype(np.float64)
    dt_data = float(ds.t[1] - ds.t[0])
    print(f"Loaded {ds}; M={A.shape[0]}, dt_data={dt_data:g}")

    out_dir = args.out_dir or (data.results_dir("LOR63") / "step2e_ablate")
    out_dir.mkdir(parents=True, exist_ok=True)

    A_mid, dAdt = sindy.deriv_5point(A, dt_data)
    print(f"Train: M_mid={A_mid.shape[0]}")
    sigma_A_global = A_mid.std(axis=0)
    sigma_A_global = np.where(sigma_A_global == 0.0, 1.0, sigma_A_global)
    print(f"Global sigma_A = {sigma_A_global.tolist()}")

    # Held-out test trajectory (same construction as step2a).
    rng_test = np.random.default_rng(args.test_ic_seed)
    x0_test = A[int(rng_test.integers(A.shape[0]))]
    n_steps_test = int(round(args.T_test / args.dt))
    X_test = _integrate_fom_truth(x0_test, args.dt, n_steps_test)
    A_test, dAdt_test = sindy.deriv_5point(X_test, args.dt)
    print(f"Test:  T={args.T_test:g} s, M_mid={A_test.shape[0]}")

    z_slice = float(A_mid[:, 2].mean())
    print(f"Diagnostic z-slice at z={z_slice:.2f}")

    # Outer K-means partitions, cached per K.
    partitions = {}
    for K in args.k_grid:
        labels_train, centroids, inertia, n_iter = clustering.kmeans_fit(
            A_mid.astype(np.float32), K,
            seed=args.kmeans_seed, n_init=args.n_init_outer,
        )
        labels_train = np.asarray(labels_train)
        labels_test = np.asarray(clustering.kmeans_predict_label(
            A_test.astype(np.float32), centroids
        ))
        counts = np.bincount(labels_train, minlength=K)
        cov_traces = np.zeros(K, dtype=np.float64)
        for k in range(K):
            if counts[k] > 1:
                cov_traces[k] = float(np.trace(
                    np.cov(A_mid[labels_train == k].T)
                ))
        var_weights = counts.astype(np.float64) * np.sqrt(
            np.maximum(cov_traces, 0.0)
        )
        partitions[K] = {
            "labels_train": labels_train,
            "labels_test":  labels_test,
            "centroids":    np.asarray(centroids),
            "counts":       counts,
            "var_weights":  var_weights,
        }
        print(f"K={K}: counts={counts.tolist()}")

    # Task grid.
    tasks = [(K, lam, mode)
             for K in args.k_grid
             for lam in args.lambda_grid
             for mode in args.sigma_modes]
    fit_kwargs = {
        "seed": args.seed, "gamma": args.gamma,
        "n_init": args.n_init, "n_tot": args.n_tot,
    }
    init_args = (A_mid, dAdt, A_test, dAdt_test, partitions,
                 sigma_A_global, fit_kwargs, z_slice, args.n_grid_diag)

    print(f"\nSweeping {len(tasks)} tasks with {args.workers} workers...")
    rows = []
    t_sweep = time.time()
    with mp.Pool(processes=args.workers, initializer=_worker_init,
                 initargs=init_args) as pool:
        for r in pool.imap_unordered(_run_task, tasks):
            rows.append(r)
            print(f"  [{time.time() - t_sweep:6.1f}s] "
                  f"K={r['K']} lam={r['lambda_rbf']:.1e} "
                  f"sig={r['sigma_mode']:<7s} "
                  f"rmse_te={r['rmse_test']:.2e} "
                  f"dead={r['dead_zone_frac']:.3f} "
                  f"|xi|/sqrt(n)_med={r['xi_per_rbf_median']:.2e} "
                  f"sigA_min={r['sigma_A_min_min']:.2f} "
                  f"nnz={r['nnz_total']} "
                  f"t={r['t_fit_seconds']:.1f}s",
                  flush=True)
    rows.sort(key=lambda r: (r["K"], r["sigma_mode"], r["lambda_rbf"]))
    print(f"Sweep finished in {time.time() - t_sweep:.1f} s.")

    summary = {
        "config": {
            "k_grid": list(args.k_grid),
            "lambda_grid": list(args.lambda_grid),
            "sigma_modes": list(args.sigma_modes),
            "n_tot": args.n_tot,
            "seed": args.seed,
            "kmeans_seed": args.kmeans_seed,
            "test_ic_seed": args.test_ic_seed,
            "T_test": args.T_test,
            "dt": args.dt,
            "gamma": args.gamma,
            "n_init": args.n_init,
            "n_init_outer": args.n_init_outer,
            "n_grid_diag": args.n_grid_diag,
            "dataset": "LOR63",
            "M_train_mid": int(A_mid.shape[0]),
            "M_test_mid": int(A_test.shape[0]),
            "dt_data": float(dt_data),
            "sigma_A_global": sigma_A_global.tolist(),
            "z_slice": float(z_slice),
        },
        "partitions": {
            str(K): {
                "counts": partitions[K]["counts"].tolist(),
                "var_weights": partitions[K]["var_weights"].tolist(),
            }
            for K in args.k_grid
        },
        "tasks": rows,
    }
    (out_dir / "step2e.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir / 'step2e.json'}")

    _make_plots(rows, args.k_grid, args.lambda_grid, args.sigma_modes, out_dir)
    return 0


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _pivot(rows, key, sigma_mode):
    """{K: (lambdas sorted, values sorted by lambda)} for given sigma_mode."""
    out = {}
    for r in rows:
        if r["sigma_mode"] != sigma_mode:
            continue
        out.setdefault(r["K"], []).append((r["lambda_rbf"], r[key]))
    for K in out:
        out[K].sort(key=lambda t: t[0])
        out[K] = (np.array([t[0] for t in out[K]]),
                  np.array([t[1] for t in out[K]]))
    return out


def _plot_K_lines(ax, by_K, *, ls="-", marker="o"):
    colors = {1: "C0", 2: "C1", 3: "C2", 4: "C3"}
    for K in sorted(by_K):
        xs, ys = by_K[K]
        ax.plot(xs, ys, ls + marker, color=colors.get(K, "k"),
                lw=1.6, label=f"K={K}")


def _make_plots(rows, k_grid, lambda_grid, sigma_modes, out_dir):
    # 1) test RMSE vs lambda, panels per sigma_mode
    fig, axs = plt.subplots(1, len(sigma_modes),
                            figsize=(6.0 * len(sigma_modes), 4.8),
                            sharey=True, squeeze=False)
    for ax, mode in zip(axs[0], sigma_modes):
        _plot_K_lines(ax, _pivot(rows, "rmse_test", mode))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\lambda_{\rm RBF}$")
        ax.set_title(f"sigma_mode = {mode}")
        ax.grid(True, ls=":", alpha=0.4, which="both")
        ax.legend(fontsize=9)
    axs[0][0].set_ylabel(r"test $\dot a$ relative RMSE")
    fig.suptitle("Step 2e: clustered RBF SINDy, test RMSE vs lambda")
    fig.tight_layout()
    fig.savefig(out_dir / "alpha_dot_vs_lambda.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'alpha_dot_vs_lambda.png'}")

    # 2) dead-zone fraction
    fig, axs = plt.subplots(1, len(sigma_modes),
                            figsize=(6.0 * len(sigma_modes), 4.8),
                            sharey=True, squeeze=False)
    for ax, mode in zip(axs[0], sigma_modes):
        _plot_K_lines(ax, _pivot(rows, "dead_zone_frac", mode))
        ax.set_xscale("log")
        ax.set_xlabel(r"$\lambda_{\rm RBF}$")
        ax.set_title(f"sigma_mode = {mode}")
        ax.grid(True, ls=":", alpha=0.4, which="both")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1)
    axs[0][0].set_ylabel("fraction of z-slice with ||f_K|| < 1e-3")
    fig.suptitle("Step 2e: dead-zone fraction vs lambda")
    fig.tight_layout()
    fig.savefig(out_dir / "dead_zone_vs_lambda.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'dead_zone_vs_lambda.png'}")

    # 3) xi norm
    fig, axs = plt.subplots(1, len(sigma_modes),
                            figsize=(6.0 * len(sigma_modes), 4.8),
                            sharey=True, squeeze=False)
    for ax, mode in zip(axs[0], sigma_modes):
        _plot_K_lines(ax, _pivot(rows, "xi_per_rbf_median", mode))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\lambda_{\rm RBF}$")
        ax.set_title(f"sigma_mode = {mode}")
        ax.grid(True, ls=":", alpha=0.4, which="both")
        ax.legend(fontsize=9)
    axs[0][0].set_ylabel(r"median$_k$  $\|\xi_k\|_F / \sqrt{n_{\rm rbf}}$")
    fig.suptitle("Step 2e: per-cluster coefficient norm vs lambda")
    fig.tight_layout()
    fig.savefig(out_dir / "xi_norm_vs_lambda.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'xi_norm_vs_lambda.png'}")

    # 4) Combined 3x|sigma_modes| panel grid for the journal
    fig, axs = plt.subplots(3, len(sigma_modes),
                            figsize=(6.0 * len(sigma_modes), 13.0),
                            sharex=True, squeeze=False)
    for col, mode in enumerate(sigma_modes):
        ax = axs[0][col]
        _plot_K_lines(ax, _pivot(rows, "rmse_test", mode))
        ax.set_yscale("log")
        ax.set_title(f"sigma_mode = {mode}: test RMSE")
        ax.grid(True, ls=":", alpha=0.4, which="both")
        ax.legend(fontsize=9)

        ax = axs[1][col]
        _plot_K_lines(ax, _pivot(rows, "dead_zone_frac", mode))
        ax.set_title("dead-zone fraction")
        ax.grid(True, ls=":", alpha=0.4, which="both")
        ax.set_ylim(0, 1)

        ax = axs[2][col]
        _plot_K_lines(ax, _pivot(rows, "xi_per_rbf_median", mode))
        ax.set_yscale("log")
        ax.set_title(r"median$_k\,\|\xi_k\|_F / \sqrt{n_{\rm rbf}}$")
        ax.grid(True, ls=":", alpha=0.4, which="both")
        ax.set_xlabel(r"$\lambda_{\rm RBF}$")
        ax.set_xscale("log")
    fig.suptitle("Step 2e: lambda x sigma_A-pinning ablation summary")
    fig.tight_layout()
    fig.savefig(out_dir / "combined.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'combined.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
