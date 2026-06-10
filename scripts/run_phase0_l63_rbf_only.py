"""Phase 0 L63 RBF-only counterfactual: alpha-dot error vs n_rbf.

Companion to `run_phase0_l63_robustness.py`. The robustness sweep fits a
joint polynomial + RBF model and asks whether the RBF block can be made
silent. This script removes the polynomial block entirely and asks the
mirror question: how well can an RBF dictionary on its own reconstruct
the time-derivative `alpha_dot` of the L63 attractor, as a function of
the number of centres?

Setup (deliberately matches the joint robustness fit where it can,
diverges where the RBF-only ablation demands):

  * RBF shape: `flat_isotropic` -- K-means centres in standardised
    coordinates, per-centre isotropic Gaussian widths from `n_knn=5`
    median inter-centre distance, `gamma=1.0`. Identical to the joint
    sweep.
  * Clustering: K-means with `n_init=10`, k-means++ init. Identical.
  * Seeding: `seed` axis identical to the joint sweep.
  * Fit: column-normalised RBF block only. No polynomial library, no
    moment orthogonalisation (nothing to orthogonalise against), no
    energy-preservation constraint (the constraint acts on quadratic
    polynomial coefficients).
  * Sparsification: STLSQ with a single fixed low `lambda_rbf` so the
    near-OLS solution is recovered; the threshold exists only to
    silence numerically-tiny coefficients.

Reports, per `(n_rbf, seed)` cell:
  * Relative alpha-dot error `||dAdt - Phi @ xi||_F / ||dAdt||_F` on the
    training points (mid-points of the 5-point stencil).
  * Per-component errors on (x_dot, y_dot, z_dot).
  * Active-column count after STLSQ.
  * `cond_gram` of `Phi_col_normalised`, computed as
    `sqrt(lam_max/lam_min)` of `Phi^T Phi` via `eigvalsh` (cheaper than
    the full SVD; saturates near ~1e8 due to Gram squaring -- diagnostic
    only).

Outputs (under `results/LOR63/phase0_rbf_only/`):
  grid_results.json     : one record per cell.
  error_vs_nrbf.png     : mean +/- std band of relative alpha-dot error
                          vs n_rbf across seeds, with per-seed curves.
  summary.json          : aggregate (median/min/max error per n_rbf).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chord2 import data, sindy


_WORKER_CTX: dict = {}


def _pool_initializer(A_mid, dAdt, dt, lambda_rbf, gamma, n_knn,
                      stlsq_max_iter, n_init):
    _WORKER_CTX["A_mid"] = A_mid
    _WORKER_CTX["dAdt"] = dAdt
    _WORKER_CTX["dt"] = dt
    _WORKER_CTX["lambda_rbf"] = lambda_rbf
    _WORKER_CTX["gamma"] = gamma
    _WORKER_CTX["n_knn"] = n_knn
    _WORKER_CTX["stlsq_max_iter"] = stlsq_max_iter
    _WORKER_CTX["n_init"] = n_init
    _WORKER_CTX["dAdt_norm"] = float(np.linalg.norm(dAdt)) or 1e-300
    _WORKER_CTX["dAdt_per_comp_norm"] = np.array(
        [float(np.linalg.norm(dAdt[:, i])) or 1e-300
         for i in range(dAdt.shape[1])]
    )


def _stlsq_rbf_only(Phi_n: np.ndarray, Y: np.ndarray,
                    lambda_rbf: float, max_iter: int):
    """STLSQ on a single (column-normalised) RBF block.

    The joint robustness fit uses `stlsq_constrained` with poly + RBF
    blocks and an equality constraint. Here neither is present, so a
    plain block-thresholded least-squares loop is the cleanest path.
    """
    n_rbf = Phi_n.shape[1]
    r = Y.shape[1]
    active = np.ones(n_rbf, dtype=bool)
    xi = np.zeros((n_rbf, r))
    history = []
    prev_active = None
    for it in range(max_iter):
        idx = np.where(active)[0]
        if idx.size == 0:
            xi = np.zeros((n_rbf, r))
            break
        xi_act, *_ = np.linalg.lstsq(Phi_n[:, idx], Y, rcond=None)
        xi = np.zeros((n_rbf, r))
        xi[idx, :] = xi_act
        new_active = active.copy()
        mags = np.max(np.abs(xi), axis=1)
        new_active[mags < lambda_rbf] = False
        history.append({"iter": it, "n_active": int(new_active.sum())})
        if prev_active is not None and np.array_equal(new_active, active):
            break
        prev_active = active.copy()
        active = new_active
    return xi, {"n_iter": len(history), "active": active,
                "history": history}


def _fit_one_cell(task):
    i, k, n_rbf, seed = task
    A_mid = _WORKER_CTX["A_mid"]
    dAdt = _WORKER_CTX["dAdt"]
    lambda_rbf = _WORKER_CTX["lambda_rbf"]
    gamma = _WORKER_CTX["gamma"]
    n_knn = _WORKER_CTX["n_knn"]
    max_iter = _WORKER_CTX["stlsq_max_iter"]

    centers, widths, rbf_meta = sindy.rbf_centers_flat_isotropic(
        A_mid, int(n_rbf), seed=int(seed),
        gamma=gamma, n_knn=n_knn,
        n_init=_WORKER_CTX["n_init"],
    )
    Phi_raw = sindy.rbf_features_iso(
        A_mid, centers, widths,
        mu_A=rbf_meta["mu_A"], sigma_A=rbf_meta["sigma_A"],
    )
    col_norms = np.linalg.norm(Phi_raw, axis=0)
    col_norms = np.where(col_norms == 0.0, 1.0, col_norms)
    Phi_n = Phi_raw / col_norms

    # Diagnostic proxy: cond^2 of Phi_n equals cond of Gram(Phi_n).
    # eigvalsh on a (n_rbf, n_rbf) Gram is O(n_rbf^3), vs O(M n_rbf^2) for
    # a full SVD of Phi_n; saves the dominant cost at large n_rbf. The
    # estimate saturates near 1/sqrt(eps_mach) ~ 1e8 (Higham 2002 sec 10.1)
    # because Gram squares the conditioning; this is acceptable for a
    # diagnostic-only field. Returns inf on rank deficiency to match
    # np.linalg.cond's semantics.
    gram_eigs = np.linalg.eigvalsh(Phi_n.T @ Phi_n)
    lam_min = float(max(gram_eigs[0], 0.0))
    lam_max = float(gram_eigs[-1])
    cond_gram = (np.inf if lam_min <= 0.0
                 else float(np.sqrt(lam_max / lam_min)))

    xi_n, fit_info = _stlsq_rbf_only(Phi_n, dAdt, lambda_rbf, max_iter)
    # alpha_dot prediction on training mid-points. Phi_n @ xi_n is
    # algebraically equal to Phi_raw @ (xi_n / col_norms[:, None]); we
    # do not need to unscale because we only use the prediction.
    Y_hat = Phi_n @ xi_n
    res = dAdt - Y_hat
    rel_err = float(np.linalg.norm(res) / _WORKER_CTX["dAdt_norm"])
    rel_err_per_comp = (
        np.linalg.norm(res, axis=0) / _WORKER_CTX["dAdt_per_comp_norm"]
    ).tolist()
    nnz = int(np.sum(np.max(np.abs(xi_n), axis=1) > 0))
    return {
        "i": int(i), "k": int(k),
        "n_rbf": int(n_rbf),
        "seed": int(seed),
        "rel_err": rel_err,
        "rel_err_x": float(rel_err_per_comp[0]),
        "rel_err_y": float(rel_err_per_comp[1]),
        "rel_err_z": float(rel_err_per_comp[2]),
        "nnz_rbf": nnz,
        "n_stlsq_iter": int(fit_info["n_iter"]),
        "cond_gram": cond_gram,
        "kmeans_inertia": float(rbf_meta["inertia"]),
        "kmeans_n_iter": int(rbf_meta["n_iter"]),
    }


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _plot_error_vs_nrbf(out_path: Path, grid_err: np.ndarray,
                        n_rbf_axis, seeds, lambda_rbf: float):
    """Mean +/- std band across seeds, with thin per-seed lines underneath."""
    fig, ax = plt.subplots(figsize=(7, 5))
    n_rbf_x = np.asarray(n_rbf_axis, dtype=float)
    for k, seed in enumerate(seeds):
        ax.plot(n_rbf_x, grid_err[:, k], "-", color="0.7", lw=0.8,
                label=f"seed {seed}" if k < 5 else None)
    mean = grid_err.mean(axis=1)
    std = grid_err.std(axis=1)
    ax.plot(n_rbf_x, mean, "o-", color="C0", lw=2, label="mean")
    ax.fill_between(n_rbf_x, mean - std, mean + std,
                    color="C0", alpha=0.2, label="$\\pm 1\\sigma$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$n_{\\rm rbf}$")
    ax.set_ylabel("$\\|\\dot{a} - \\Phi \\xi\\|_F / \\|\\dot{a}\\|_F$")
    ax.set_title(f"L63 RBF-only: relative $\\dot{{a}}$ error vs $n_{{\\rm rbf}}$\n"
                 f"($\\lambda_{{\\rm rbf}}$={lambda_rbf:g}, {len(seeds)} seeds)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n-rbf", type=int, nargs="*",
                   default=[20, 50, 100, 200, 400, 800, 1600],
                   help="centre-count axis (default: log-spaced 20..1600)")
    p.add_argument("--seeds", type=int, nargs="*",
                   default=[0, 1, 2, 3, 4],
                   help="seed axis for K-means init (default: 0..4)")
    p.add_argument("--lambda-rbf", type=float, default=1e-3,
                   help="fixed STLSQ threshold on column-normalised RBF block")
    p.add_argument("--gamma", type=float, default=1.0,
                   help="isotropic-width scale (multiplier on 5-NN median)")
    p.add_argument("--n-knn", type=int, default=5,
                   help="neighbours used in the width heuristic")
    p.add_argument("--n-init", type=int, default=1,
                   help="K-means restarts in centre placement (default: 1). "
                        "The joint robustness sweep uses 10 to suppress "
                        "bad-local-minimum variance; here the --seeds axis "
                        "already exposes centre-placement variance, so 1 is "
                        "the honest default and ~10x cheaper at large n_rbf.")
    p.add_argument("--stlsq-max-iter", type=int, default=20)
    p.add_argument("--workers", type=int, default=1,
                   help="multiprocessing workers (default: 1, serial). On "
                        "SLURM use --workers == SLURM_CPUS_PER_TASK.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="override default results/LOR63/phase0_rbf_only/")
    args = p.parse_args()

    ds = data.load("LOR63")
    A = ds.u.astype(np.float64)
    dt = float(ds.t[1] - ds.t[0])
    print(f"Loaded {ds}; dt={dt:g}, M={A.shape[0]}, r={A.shape[1]}")

    out_dir = args.out_dir or (data.results_dir("LOR63") / "phase0_rbf_only")
    out_dir.mkdir(parents=True, exist_ok=True)

    A_mid, dAdt = sindy.deriv_5point(A, dt)
    print(f"5-point stencil derivative: M_mid={A_mid.shape[0]}, "
          f"||dAdt||_F={np.linalg.norm(dAdt):.3e}")

    n_n = len(args.n_rbf)
    n_s = len(args.seeds)
    grid_err = np.zeros((n_n, n_s))
    grid_err_x = np.zeros((n_n, n_s))
    grid_err_y = np.zeros((n_n, n_s))
    grid_err_z = np.zeros((n_n, n_s))
    grid_nnz = np.zeros((n_n, n_s), dtype=int)
    grid_cond = np.zeros((n_n, n_s))

    tasks = [
        (i, k, int(n_rbf), int(seed))
        for i, n_rbf in enumerate(args.n_rbf)
        for k, seed in enumerate(args.seeds)
    ]
    grid_records = [None] * len(tasks)

    workers = max(1, int(args.workers))
    print(f"\n--- RBF-only sweep: {n_n} x {n_s} = {len(tasks)} fits "
          f"(workers={workers}, lambda_rbf={args.lambda_rbf:g}, "
          f"n_init={args.n_init}) ---")

    init_args = (A_mid, dAdt, dt, args.lambda_rbf, args.gamma, args.n_knn,
                 args.stlsq_max_iter, args.n_init)

    if workers == 1:
        _pool_initializer(*init_args)
        rec_iter = (_fit_one_cell(t) for t in tasks)
    else:
        ctx = mp.get_context("fork")
        pool = ctx.Pool(processes=workers,
                        initializer=_pool_initializer,
                        initargs=init_args)
        rec_iter = pool.imap_unordered(_fit_one_cell, tasks, chunksize=1)

    n_done = 0
    for rec in rec_iter:
        i, k = rec["i"], rec["k"]
        grid_err[i, k] = rec["rel_err"]
        grid_err_x[i, k] = rec["rel_err_x"]
        grid_err_y[i, k] = rec["rel_err_y"]
        grid_err_z[i, k] = rec["rel_err_z"]
        grid_nnz[i, k] = rec["nnz_rbf"]
        grid_cond[i, k] = rec["cond_gram"]
        flat_idx = i * n_s + k
        grid_records[flat_idx] = {kk: vv for kk, vv in rec.items()
                                  if kk not in ("i", "k")}
        n_done += 1
        print(f"  [{n_done:3d}/{len(tasks)}]  "
              f"n_rbf={rec['n_rbf']:4d}  seed={rec['seed']}: "
              f"rel_err={rec['rel_err']:.3e}  "
              f"nnz={rec['nnz_rbf']:4d}  "
              f"cond_gram={rec['cond_gram']:.2e}  "
              f"per_comp=({rec['rel_err_x']:.2e},"
              f"{rec['rel_err_y']:.2e},{rec['rel_err_z']:.2e})",
              flush=True)

    if workers > 1:
        pool.close()
        pool.join()

    _plot_error_vs_nrbf(out_dir / "error_vs_nrbf.png", grid_err,
                        args.n_rbf, args.seeds, args.lambda_rbf)

    (out_dir / "grid_results.json").write_text(
        json.dumps(_jsonable(grid_records), indent=2)
    )

    summary = {
        "n_rbf_axis": [int(x) for x in args.n_rbf],
        "seeds": [int(x) for x in args.seeds],
        "lambda_rbf": float(args.lambda_rbf),
        "gamma": float(args.gamma),
        "n_knn": int(args.n_knn),
        "n_init": int(args.n_init),
        "n_total_fits": int(n_n * n_s),
        "rel_err_median_per_nrbf": [
            float(np.median(grid_err[i])) for i in range(n_n)],
        "rel_err_min_per_nrbf": [
            float(np.min(grid_err[i])) for i in range(n_n)],
        "rel_err_max_per_nrbf": [
            float(np.max(grid_err[i])) for i in range(n_n)],
        "rel_err_std_per_nrbf": [
            float(np.std(grid_err[i])) for i in range(n_n)],
        "nnz_median_per_nrbf": [
            int(np.median(grid_nnz[i])) for i in range(n_n)],
        "cond_gram_median_per_nrbf": [
            float(np.median(grid_cond[i])) for i in range(n_n)],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n--- Verdict ---")
    for i, n_rbf in enumerate(args.n_rbf):
        print(f"  n_rbf={n_rbf:4d}: "
              f"rel_err median={summary['rel_err_median_per_nrbf'][i]:.3e}, "
              f"min={summary['rel_err_min_per_nrbf'][i]:.3e}, "
              f"max={summary['rel_err_max_per_nrbf'][i]:.3e}, "
              f"nnz median={summary['nnz_median_per_nrbf'][i]:4d}, "
              f"cond_gram median={summary['cond_gram_median_per_nrbf'][i]:.2e}")
    print(f"\nOutputs under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
