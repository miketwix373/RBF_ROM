"""L63 step 2f (refit): clustered RBF cells with sigma_A pinned to global std.

Step 2e established that per-cluster sigma_A standardisation is the
dominant Stage B failure mechanism on L63: cluster-local std shrinks as K
grows, the RBF Gaussians decay faster in real coordinates, and large
regions outside training support fall into numerical zero -- the dead
zones that captured K>=2 trajectories in step 2b.

The fix is to fit the RBF features with a *global* sigma_A (= std of the
full training A_mid) for every cluster. mu_A stays per-cluster -- the
rom-specialist consult of 2026-06-09 kept the centring local so that
clusters with very different centroids retain their natural coordinate
origin; only the scale is globalised.

This driver mirrors `step2a_cluster_rbf_fit.py` but:
  * Hardwires `--n-tot 1280` (Stage A noise floor used by Stage B).
  * Hardwires `--lambda-rbf 1e-3` (the lambda that step 2e showed is
    indistinguishable from any value below ~1e7 once the global-sigma
    branch is selected).
  * Passes `sigma_A_override` to every per-cluster `fit_rbf_only_keep_state`.
  * Saves cells using step2a's `_save_cell_models` schema so
    `step2b_stability.py --models-dir <step2f models>` loads them
    transparently.

Outputs (results/LOR63/step2f_globalsigma/models/):
  cell_K1_N01280_count.npz
  cell_K2_N01280_variance.npz
  cell_K3_N01280_variance.npz
  cell_K4_N01280_variance.npz
  step2f_refit.json    -- summary (counts, partition info, sigma_A_global)
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _l63_rbf_lib import (  # noqa: E402
    f_rbf, fit_rbf_only_keep_state,
)
# Reuse step2a's allocator and persistence helpers verbatim.
from step2a_cluster_rbf_fit import (  # noqa: E402
    _allocate_proportional, _save_cell_models,
)

from chord2 import clustering, data, sindy  # noqa: E402


def _fit_clustered_globalsigma(A_mid: np.ndarray, dAdt: np.ndarray,
                               labels: np.ndarray, K: int,
                               n_per_k: np.ndarray, *,
                               seed: int, lambda_rbf: float, gamma: float,
                               n_init: int, lambda_tikh: float,
                               sigma_A_global: np.ndarray) -> list:
    """Per-cluster RBF-only STLSQ with sigma_A pinned to global std.

    Mirrors step2a._fit_clustered byte-for-byte except that it passes
    `sigma_A_override=sigma_A_global` to every `fit_rbf_only_keep_state`
    call. mu_A stays per-cluster (i.e. left to default = cluster mean).
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
        fit = fit_rbf_only_keep_state(
            A_mid[mask], dAdt[mask],
            n_rbf=n_k, seed=seed,
            lambda_rbf=lambda_rbf, gamma=gamma,
            n_knn=5, n_init=n_init,
            bandwidth_mode="attr_scaled",
            lambda_tikh=lambda_tikh,
            log_diagnostics=True,
            sigma_A_override=sigma_A_global,
        )
        models.append(fit)
    return models


def _score_clustered(A: np.ndarray, dAdt_true: np.ndarray,
                     labels: np.ndarray, K: int, models: list) -> dict:
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


# ---------------------------------------------------------------------------
# Pool worker
# ---------------------------------------------------------------------------

_WORKER_STATE: dict = {}


def _worker_init(A_mid: np.ndarray, dAdt: np.ndarray,
                 partitions: dict, sigma_A_global: np.ndarray,
                 fit_kwargs: dict, models_dir: str) -> None:
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _WORKER_STATE["A_mid"] = A_mid
    _WORKER_STATE["dAdt"] = dAdt
    _WORKER_STATE["partitions"] = partitions
    _WORKER_STATE["sigma_A_global"] = sigma_A_global
    _WORKER_STATE["fit_kwargs"] = fit_kwargs
    _WORKER_STATE["models_dir"] = Path(models_dir)


def _run_cell(task: tuple) -> dict:
    K, n_tot, alloc = task
    A_mid = _WORKER_STATE["A_mid"]
    dAdt = _WORKER_STATE["dAdt"]
    part = _WORKER_STATE["partitions"][K]
    sigma_A_global = _WORKER_STATE["sigma_A_global"]
    fk = _WORKER_STATE["fit_kwargs"]
    models_dir = _WORKER_STATE["models_dir"]

    labels_train = part["labels_train"]
    counts = part["counts"]
    centroids = part["centroids"]
    weights = (counts.astype(np.float64) if alloc == "count"
               else part["var_weights"])
    n_per_k = _allocate_proportional(weights, counts, n_tot)

    t0 = time.time()
    models = _fit_clustered_globalsigma(
        A_mid, dAdt, labels_train, K, n_per_k,
        seed=fk["seed"], lambda_rbf=fk["lambda_rbf"],
        gamma=fk["gamma"], n_init=fk["n_init"],
        lambda_tikh=fk["lambda_tikh"],
        sigma_A_global=sigma_A_global,
    )
    train = _score_clustered(A_mid, dAdt, labels_train, K, models)
    t_cell = time.time() - t0

    model_path = models_dir / f"cell_K{K}_N{n_tot:05d}_{alloc}.npz"
    # Reuse step2a's persistence schema verbatim so step2b's load_cell
    # picks it up transparently. step2a stores `bandwidth_mode` in meta;
    # we keep that as "attr_scaled" and add a sigma_A_mode marker that
    # step2b ignores -- the actual sigma_A values are per-cluster fields,
    # set from the global override at fit time.
    fk_for_save = dict(fk)
    _save_cell_models(model_path, K, n_tot, alloc,
                      centroids, n_per_k, counts, models, fk_for_save)

    return {
        "K": int(K), "n_tot": int(n_tot), "alloc": alloc,
        "n_per_cluster": n_per_k.tolist(),
        "counts": counts.tolist(),
        "rmse_train": train["rmse"],
        "unmodelled_train": int(train["unmodelled"]),
        "t_cell_seconds": float(t_cell),
        "model_path": str(model_path),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--k-grid", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--n-tot", type=int, default=1280)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--kmeans-seed", type=int, default=0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--lambda-rbf", type=float, default=1e-3)
    p.add_argument("--lambda-tikh", type=float, default=0.0)
    p.add_argument("--n-init", type=int, default=1)
    p.add_argument("--n-init-outer", type=int, default=10)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    ds = data.load("LOR63")
    A = ds.u.astype(np.float64)
    dt_data = float(ds.t[1] - ds.t[0])
    print(f"Loaded {ds}; M={A.shape[0]}, dt_data={dt_data:g}")

    out_dir = args.out_dir or (data.results_dir("LOR63") / "step2f_globalsigma")
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    A_mid, dAdt = sindy.deriv_5point(A, dt_data)
    print(f"Train: M_mid={A_mid.shape[0]}")
    sigma_A_global = A_mid.std(axis=0)
    sigma_A_global = np.where(sigma_A_global == 0.0, 1.0, sigma_A_global)
    print(f"Global sigma_A = {sigma_A_global.tolist()}")

    partitions = {}
    for K in args.k_grid:
        labels_train, centroids, inertia, n_iter = clustering.kmeans_fit(
            A_mid.astype(np.float32), K,
            seed=args.kmeans_seed, n_init=args.n_init_outer,
        )
        labels_train = np.asarray(labels_train)
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
            "centroids":    np.asarray(centroids),
            "counts":       counts,
            "var_weights":  var_weights,
            "inertia":      inertia,
            "n_iter":       n_iter,
        }
        print(f"K={K}: counts={counts.tolist()}")

    # Tasks: K=1 has only count-alloc; K>=2 takes variance-alloc (Stage A
    # winner) -- matches step2b's loader expectation.
    tasks = []
    for K in args.k_grid:
        alloc = "count" if K == 1 else "variance"
        tasks.append((K, args.n_tot, alloc))
    fit_kwargs = {
        "seed": args.seed, "lambda_rbf": args.lambda_rbf,
        "gamma": args.gamma, "n_knn": 5,
        "n_init": args.n_init, "lambda_tikh": args.lambda_tikh,
    }
    init_args = (A_mid, dAdt, partitions, sigma_A_global,
                 fit_kwargs, str(models_dir))

    print(f"\nRefitting {len(tasks)} cells with {args.workers} workers "
          f"(sigma_A pinned to global)...")
    rows = []
    t_sweep = time.time()
    with mp.Pool(processes=args.workers, initializer=_worker_init,
                 initargs=init_args) as pool:
        for r in pool.imap_unordered(_run_cell, tasks):
            rows.append(r)
            print(f"  [{time.time() - t_sweep:6.1f}s] "
                  f"K={r['K']} N_tot={r['n_tot']:>5d} alloc={r['alloc']:<8s} "
                  f"n_per_k={r['n_per_cluster']} "
                  f"rmse_tr={r['rmse_train']:.3e} "
                  f"-> {Path(r['model_path']).name} "
                  f"t={r['t_cell_seconds']:.1f}s",
                  flush=True)
    rows.sort(key=lambda r: r["K"])
    print(f"Refit finished in {time.time() - t_sweep:.1f} s.")

    summary = {
        "config": {
            "k_grid": list(args.k_grid),
            "n_tot": args.n_tot,
            "seed": args.seed,
            "kmeans_seed": args.kmeans_seed,
            "gamma": args.gamma,
            "lambda_rbf": args.lambda_rbf,
            "lambda_tikh": args.lambda_tikh,
            "n_init": args.n_init,
            "n_init_outer": args.n_init_outer,
            "sigma_A_global": sigma_A_global.tolist(),
            "dataset": "LOR63",
            "M_train_mid": int(A_mid.shape[0]),
            "dt_data": float(dt_data),
            "sigma_A_mode": "global",
            "mu_A_mode":    "per_cluster",
            "bandwidth_mode": "attr_scaled",
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
        "cells": rows,
    }
    (out_dir / "step2f_refit.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir / 'step2f_refit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
