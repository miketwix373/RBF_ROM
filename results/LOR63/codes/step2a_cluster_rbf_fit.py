"""L63 step 2a: does clustering help RBF-only SINDy seeding?

Stage A of the Stage A / Stage B plan recorded in the chat for 2026-06-09:
the question is whether splitting the L63 attractor by K-means before fitting
an RBF-only SINDy dictionary lowers `alpha_dot` test error at fixed total
RBF count. Stage A is regression-only -- no integration, no cluster
switching, no change-of-coordinates. Stage B (Lyapunov spectra + PDF
survival on off-attractor ICs) is a separate driver.

Setup mirrors the rom-specialist consult of 2026-06-09:

  * Outer K-means on (x, y, z) training snapshots; K in {1, 2, 3, 4}.
  * Per cluster, fit RBF-only SINDy via `fit_rbf_only_keep_state` with
    `bandwidth_mode='attr_scaled'` (the consult flagged that knn would
    silently shrink sigma_k as K grows, conflating the K- and sigma-knobs).
  * Two allocation rules across clusters for the total RBF budget N_tot:
      'count'    : n_k proportional to cluster population
      'variance' : n_k proportional to population * sqrt(trace(Cov_k))
    The headline is 'count'; 'variance' is the sensitivity overlay. The
    consult's prediction: if they disagree, the disagreement *is* the
    diagnostic (count-proportional starves transition clusters where the
    field is fastest).
  * Held-out test trajectory: integrate FOM truth from a fresh IC seed for
    T_test seconds. Test rows are assigned via `kmeans_predict_label`
    using the *training* centroids (re-clustering the test set would
    measure clustering stability, not regression skill).
  * alpha_dot RMSE reported as one global number: sum over all test rows,
    each evaluated with its training-Voronoi cluster's model.

Diagnostics:
  * train RMSE (overfitting witness vs test)
  * max per-cluster condition number kappa(Phi^T Phi) -- the consult's
    sanity check that the regression problem actually gets easier per
    cluster, which is the proposed U-shape mechanism
  * total nnz across the per-cluster dictionaries

Outputs (results/LOR63/step2a_cluster_rbf/):
  step2a.json
  alpha_dot_test.png
  alpha_dot_train_test_diag.png
  alpha_dot_alloc.png
  kappa.png
  nnz.png
  models/cell_K<K>_N<N>_<alloc>.npz  -- per-cell fit state for Stage B

This script is `results/LOR63/`-local; it does not touch `chord2/` library
code. Per-cluster fits use `_l63_rbf_lib.fit_rbf_only_keep_state` exactly
as the step 1c-1g track did.
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
    """Distribute `n_tot` integer RBF slots across clusters by `weights`.

    Largest-remainder method (a.k.a. Hare-Niemeyer): floor of the
    proportional share + top-up by largest fractional parts. Clipped at
    `counts` so we never request more RBFs than snapshots in a cluster
    (the inner K-means inside `rbf_centers_flat_isotropic` would fail).
    Zero-weight clusters get zero centres.
    """
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


def _fit_clustered(A_mid: np.ndarray, dAdt: np.ndarray,
                   labels: np.ndarray, K: int, n_per_k: np.ndarray,
                   *, seed: int, lambda_rbf: float, gamma: float,
                   n_knn: int, n_init: int, lambda_tikh: float) -> list:
    """Fit one RBF-only SINDy per cluster; return list of fit dicts or None."""
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
        fit = fit_rbf_only_keep_state(
            A_mid[mask], dAdt[mask],
            n_rbf=n_k, seed=seed,
            lambda_rbf=lambda_rbf, gamma=gamma,
            n_knn=n_knn, n_init=n_init,
            bandwidth_mode="attr_scaled",
            lambda_tikh=lambda_tikh,
            log_diagnostics=True,
        )
        models.append(fit)
    return models


def _score_clustered(A: np.ndarray, dAdt_true: np.ndarray,
                     labels: np.ndarray, K: int, models: list) -> dict:
    """Per-cluster RBF eval, pooled into one global RMSE on `||dAdt_true||`.

    Rows in clusters whose model is None contribute `||dAdt_true[mask]||^2`
    to the error (prediction is taken as zero) and are counted separately.
    """
    err_sq = 0.0
    unmodelled = 0
    total_norm_sq = float((dAdt_true * dAdt_true).sum())
    for k in range(K):
        mask = labels == k
        if not np.any(mask):
            continue
        if models[k] is None:
            err_sq += float((dAdt_true[mask] * dAdt_true[mask]).sum())
            unmodelled += int(mask.sum())
            continue
        f = models[k]
        pred = f_rbf(A[mask], f["centers"], f["widths"],
                     f["mu_A"], f["sigma_A"], f["col_norms"], f["xi"])
        diff = dAdt_true[mask] - pred
        err_sq += float((diff * diff).sum())
    rmse = float(np.sqrt(err_sq / max(total_norm_sq, 1e-30)))
    return {"rmse": rmse, "unmodelled": unmodelled}


def _kappa_max(models: list) -> float:
    """Max per-cluster condition number from the final STLSQ iteration."""
    kappas = []
    for m in models:
        if m is None or not m["diagnostics"]:
            continue
        k = m["diagnostics"][-1]["kappa"]
        if np.isfinite(k):
            kappas.append(float(k))
    return float(max(kappas)) if kappas else float("nan")


def _nnz_total(models: list) -> int:
    return int(sum(m["nnz"] for m in models if m is not None))


# ---------------------------------------------------------------------------
# Pool worker
# ---------------------------------------------------------------------------

_WORKER_STATE: dict = {}


def _worker_init(A_mid: np.ndarray, dAdt: np.ndarray,
                 A_test: np.ndarray, dAdt_test: np.ndarray,
                 partitions: dict, fit_kwargs: dict,
                 models_dir: str) -> None:
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _WORKER_STATE["A_mid"] = A_mid
    _WORKER_STATE["dAdt"] = dAdt
    _WORKER_STATE["A_test"] = A_test
    _WORKER_STATE["dAdt_test"] = dAdt_test
    _WORKER_STATE["partitions"] = partitions
    _WORKER_STATE["fit_kwargs"] = fit_kwargs
    _WORKER_STATE["models_dir"] = Path(models_dir)


def _save_cell_models(path: Path, K: int, n_tot: int, alloc: str,
                      centroids: np.ndarray, n_per_k: np.ndarray,
                      counts: np.ndarray, models: list,
                      fit_kwargs: dict) -> None:
    """Persist all per-cluster fit state for one (K, N_tot, alloc) cell.

    Schema: one .npz with prefix-keyed arrays.
      K, n_tot, alloc           -- cell identifiers
      centroids       (K, 3)    -- outer K-means centroids (training)
      n_per_cluster   (K,)      -- RBFs allocated per cluster
      counts          (K,)      -- training snapshot count per cluster
      cluster_modelled (K,) bool
      c<k>_centers    (n_k, 3)
      c<k>_widths     (n_k,)
      c<k>_mu_A       (3,)
      c<k>_sigma_A    (3,)
      c<k>_col_norms  (n_k,)
      c<k>_xi         (n_k, 3)
      meta            object -- {gamma, lambda_rbf, lambda_tikh, seed,
                                 bandwidth_mode, n_knn, n_init}

    Stage B reads this and rebuilds the piecewise field via `f_rbf` per
    cluster, with cluster assignment by nearest training centroid.
    """
    payload = {
        "K": np.int32(K),
        "n_tot": np.int32(n_tot),
        "alloc": np.array(alloc),
        "centroids": np.asarray(centroids, dtype=np.float64),
        "n_per_cluster": np.asarray(n_per_k, dtype=np.int32),
        "counts": np.asarray(counts, dtype=np.int64),
        "cluster_modelled": np.array(
            [m is not None for m in models], dtype=bool
        ),
        "meta": np.array({
            "gamma": fit_kwargs["gamma"],
            "lambda_rbf": fit_kwargs["lambda_rbf"],
            "lambda_tikh": fit_kwargs["lambda_tikh"],
            "seed": fit_kwargs["seed"],
            "n_knn": fit_kwargs["n_knn"],
            "n_init": fit_kwargs["n_init"],
            "bandwidth_mode": "attr_scaled",
        }, dtype=object),
    }
    for k, m in enumerate(models):
        if m is None:
            continue
        payload[f"c{k}_centers"] = np.asarray(m["centers"], dtype=np.float64)
        payload[f"c{k}_widths"] = np.asarray(m["widths"], dtype=np.float64)
        payload[f"c{k}_mu_A"] = np.asarray(m["mu_A"], dtype=np.float64)
        payload[f"c{k}_sigma_A"] = np.asarray(m["sigma_A"], dtype=np.float64)
        payload[f"c{k}_col_norms"] = np.asarray(m["col_norms"], dtype=np.float64)
        payload[f"c{k}_xi"] = np.asarray(m["xi"], dtype=np.float64)
    np.savez(path, **payload)


def _run_cell(task: tuple) -> dict:
    K, n_tot, alloc = task
    A_mid = _WORKER_STATE["A_mid"]
    dAdt = _WORKER_STATE["dAdt"]
    A_test = _WORKER_STATE["A_test"]
    dAdt_test = _WORKER_STATE["dAdt_test"]
    part = _WORKER_STATE["partitions"][K]
    fk = _WORKER_STATE["fit_kwargs"]
    models_dir = _WORKER_STATE["models_dir"]

    labels_train = part["labels_train"]
    labels_test = part["labels_test"]
    counts = part["counts"]
    centroids = part["centroids"]
    weights = (counts.astype(np.float64) if alloc == "count"
               else part["var_weights"])
    n_per_k = _allocate_proportional(weights, counts, n_tot)

    t_cell = time.time()
    models = _fit_clustered(
        A_mid, dAdt, labels_train, K, n_per_k,
        seed=fk["seed"], lambda_rbf=fk["lambda_rbf"],
        gamma=fk["gamma"], n_knn=fk["n_knn"],
        n_init=fk["n_init"], lambda_tikh=fk["lambda_tikh"],
    )
    train = _score_clustered(A_mid, dAdt, labels_train, K, models)
    test = _score_clustered(A_test, dAdt_test, labels_test, K, models)

    model_path = models_dir / f"cell_K{K}_N{n_tot:05d}_{alloc}.npz"
    _save_cell_models(model_path, K, n_tot, alloc,
                      centroids, n_per_k, counts, models, fk)

    return {
        "K": int(K), "n_tot": int(n_tot), "alloc": alloc,
        "n_per_cluster": n_per_k.tolist(),
        "counts": counts.tolist(),
        "rmse_train": train["rmse"],
        "rmse_test": test["rmse"],
        "unmodelled_train": int(train["unmodelled"]),
        "unmodelled_test": int(test["unmodelled"]),
        "kappa_max": _kappa_max(models),
        "nnz_total": _nnz_total(models),
        "t_cell_seconds": float(time.time() - t_cell),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--k-grid", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--n-tot-grid", type=int, nargs="+",
                   default=[20, 40, 80, 160, 320, 640, 1280])
    p.add_argument("--seed", type=int, default=1,
                   help="SINDy / RBF-placement seed (single global value)")
    p.add_argument("--kmeans-seed", type=int, default=0,
                   help="seed for the outer K-means partitioning step")
    p.add_argument("--test-ic-seed", type=int, default=7,
                   help="seed for selecting the held-out IC from training")
    p.add_argument("--T-test", type=float, default=20.0)
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--gamma", type=float, default=1.0,
                   help="attr_scaled bandwidth: sigma in standardised coords")
    p.add_argument("--lambda-rbf", type=float, default=1e-3)
    p.add_argument("--lambda-tikh", type=float, default=0.0)
    p.add_argument("--n-knn", type=int, default=5,
                   help="forwarded but unused under bandwidth_mode='attr_scaled'")
    p.add_argument("--n-init", type=int, default=1,
                   help="K-means restarts inside the RBF centre placer")
    p.add_argument("--n-init-outer", type=int, default=10,
                   help="K-means restarts for the outer cluster partition")
    p.add_argument("--workers", type=int, default=8,
                   help="mp.Pool worker count over the (K, N_tot, alloc) grid")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    ds = data.load("LOR63")
    A = ds.u.astype(np.float64)
    dt_data = float(ds.t[1] - ds.t[0])
    print(f"Loaded {ds}; dt_data={dt_data:g}, M={A.shape[0]}, T_data={ds.t[-1]:g}")

    out_dir = args.out_dir or (data.results_dir("LOR63") / "step2a_cluster_rbf")
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # -- Training derivatives ------------------------------------------------
    A_mid, dAdt = sindy.deriv_5point(A, dt_data)
    print(f"Train: M_mid={A_mid.shape[0]}")

    # -- Held-out test trajectory --------------------------------------------
    rng_test = np.random.default_rng(args.test_ic_seed)
    x0_test = A[int(rng_test.integers(A.shape[0]))]
    n_steps_test = int(round(args.T_test / args.dt))
    X_test = _integrate_fom_truth(x0_test, args.dt, n_steps_test)
    A_test, dAdt_test = sindy.deriv_5point(X_test, args.dt)
    print(f"Test:  T={args.T_test:g} s, M_mid={A_test.shape[0]}, "
          f"x0 norm={np.linalg.norm(x0_test):.2f}")

    # -- Outer K-means partitions, one per K --------------------------------
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
            "inertia":      inertia,
            "n_iter":       n_iter,
        }
        print(f"K={K}: counts={counts.tolist()}, "
              f"var_weights={[f'{v:.1f}' for v in var_weights]}, "
              f"inertia={inertia:.2e}")

    # -- Sweep (parallel over (K, N_tot, alloc) cells) -----------------------
    tasks = []
    for K in args.k_grid:
        alloc_rules = ("count",) if K == 1 else ("count", "variance")
        for n_tot in args.n_tot_grid:
            for alloc in alloc_rules:
                tasks.append((K, n_tot, alloc))
    fit_kwargs = {
        "seed": args.seed, "lambda_rbf": args.lambda_rbf,
        "gamma": args.gamma, "n_knn": args.n_knn,
        "n_init": args.n_init, "lambda_tikh": args.lambda_tikh,
    }
    init_args = (A_mid, dAdt, A_test, dAdt_test, partitions, fit_kwargs,
                 str(models_dir))
    print(f"\nSweeping {len(tasks)} cells with {args.workers} workers...")
    cells = []
    t_sweep_start = time.time()
    with mp.Pool(processes=args.workers, initializer=_worker_init,
                 initargs=init_args) as pool:
        for r in pool.imap_unordered(_run_cell, tasks):
            cells.append(r)
            print(f"  [{time.time() - t_sweep_start:6.1f}s] "
                  f"K={r['K']} N_tot={r['n_tot']:>5d} alloc={r['alloc']:<8s} "
                  f"n_per_k={r['n_per_cluster']} "
                  f"rmse_tr={r['rmse_train']:.3e} rmse_te={r['rmse_test']:.3e} "
                  f"kappa_max={r['kappa_max']:.2e} nnz={r['nnz_total']} "
                  f"t={r['t_cell_seconds']:.1f}s",
                  flush=True)
    cells.sort(key=lambda c: (c["K"], c["alloc"], c["n_tot"]))
    print(f"Sweep finished in {time.time() - t_sweep_start:.1f} s.")

    # -- Persist -------------------------------------------------------------
    summary = {
        "config": {
            "k_grid": list(args.k_grid),
            "n_tot_grid": list(args.n_tot_grid),
            "seed": args.seed,
            "kmeans_seed": args.kmeans_seed,
            "test_ic_seed": args.test_ic_seed,
            "T_test": args.T_test,
            "dt": args.dt,
            "gamma": args.gamma,
            "lambda_rbf": args.lambda_rbf,
            "lambda_tikh": args.lambda_tikh,
            "n_knn": args.n_knn,
            "n_init": args.n_init,
            "n_init_outer": args.n_init_outer,
            "bandwidth_mode": "attr_scaled",
            "dataset": "LOR63",
            "M_train_mid": int(A_mid.shape[0]),
            "M_test_mid": int(A_test.shape[0]),
            "dt_data": float(dt_data),
        },
        "partitions": {
            str(K): {
                "counts": partitions[K]["counts"].tolist(),
                "var_weights": partitions[K]["var_weights"].tolist(),
                "inertia": float(partitions[K]["inertia"]),
                "n_iter": int(partitions[K]["n_iter"]),
            }
            for K in args.k_grid
        },
        "cells": cells,
    }
    (out_dir / "step2a.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir / 'step2a.json'}")

    # -- Plots ---------------------------------------------------------------
    _make_plots(cells, args.k_grid, args.n_tot_grid, out_dir)
    return 0


def _by_K(cells, alloc, key):
    """Pivot cells: -> dict K -> (n_tot sorted, values sorted by n_tot)."""
    out = {}
    for c in cells:
        if c["alloc"] != alloc:
            continue
        K = c["K"]
        out.setdefault(K, []).append((c["n_tot"], c[key]))
    for K in out:
        out[K].sort(key=lambda x: x[0])
        out[K] = (np.array([x[0] for x in out[K]]),
                  np.array([x[1] for x in out[K]]))
    return out


def _plot_K_lines(ax, by_K, *, ls="-", marker="o", label_prefix="K="):
    colors = {1: "C0", 2: "C1", 3: "C2", 4: "C3"}
    for K in sorted(by_K):
        xs, ys = by_K[K]
        ax.plot(xs, ys, ls + marker, color=colors.get(K, "k"), lw=1.6,
                label=f"{label_prefix}{K}")


def _make_plots(cells, k_grid, n_tot_grid, out_dir):
    test_count = _by_K(cells, "count", "rmse_test")
    train_count = _by_K(cells, "count", "rmse_train")
    test_var = _by_K(cells, "variance", "rmse_test")
    kappa = _by_K(cells, "count", "kappa_max")
    nnz = _by_K(cells, "count", "nnz_total")

    # 1) Headline: test RMSE vs N_tot, four K-lines, count-proportional
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    _plot_K_lines(ax, test_count, ls="-", marker="o")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"total RBF count $N_{tot}$")
    ax.set_ylabel(r"test $\Vert \dot a_{true} - \dot a_{pred} \Vert / \Vert \dot a_{true} \Vert$")
    ax.set_title(r"L63 step 2a: clustered RBF-only $\dot a$ error "
                 r"(attr_scaled $\gamma=1$, count-proportional)")
    ax.grid(True, ls=":", alpha=0.4, which="both")
    ax.legend(fontsize=10, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "alpha_dot_test.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'alpha_dot_test.png'}")

    # 2) Train (dashed) + test (solid) diagnostic, count-proportional
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    _plot_K_lines(ax, train_count, ls="--", marker="s", label_prefix="K=")
    _plot_K_lines(ax, test_count, ls="-", marker="o", label_prefix="K=")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"total RBF count $N_{tot}$")
    ax.set_ylabel(r"$\dot a$ relative error")
    ax.set_title(r"L63 step 2a: train (dashed) vs test (solid) -- overfitting witness")
    ax.grid(True, ls=":", alpha=0.4, which="both")
    ax.legend(fontsize=10, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "alpha_dot_train_test_diag.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'alpha_dot_train_test_diag.png'}")

    # 3) Count vs variance allocation, test RMSE, K>=2 only
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    _plot_K_lines(ax, {K: test_count[K] for K in test_count if K > 1},
                  ls="-", marker="o", label_prefix="K=")
    _plot_K_lines(ax, test_var, ls="--", marker="^", label_prefix="K(var)=")
    if 1 in test_count:
        xs, ys = test_count[1]
        ax.plot(xs, ys, "k:", lw=1.2, label="K=1 (reference)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"total RBF count $N_{tot}$")
    ax.set_ylabel(r"test $\dot a$ relative error")
    ax.set_title(r"L63 step 2a: count- (solid) vs variance-proportional (dashed) allocation")
    ax.grid(True, ls=":", alpha=0.4, which="both")
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "alpha_dot_alloc.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'alpha_dot_alloc.png'}")

    # 4) Max per-cluster kappa vs N_tot
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    _plot_K_lines(ax, kappa, ls="-", marker="o")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"total RBF count $N_{tot}$")
    ax.set_ylabel(r"$\max_k \kappa(\Phi_k^\top \Phi_k)$")
    ax.set_title("L63 step 2a: per-cluster STLSQ conditioning")
    ax.grid(True, ls=":", alpha=0.4, which="both")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "kappa.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'kappa.png'}")

    # 5) Total nnz across clusters
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    _plot_K_lines(ax, nnz, ls="-", marker="o")
    ax.set_xscale("log")
    ax.set_xlabel(r"total RBF count $N_{tot}$")
    ax.set_ylabel(r"$\sum_k \mathrm{nnz}(\xi_k)$")
    ax.set_title("L63 step 2a: total sparsity across per-cluster dictionaries")
    ax.grid(True, ls=":", alpha=0.4, which="both")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "nnz.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'nnz.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
