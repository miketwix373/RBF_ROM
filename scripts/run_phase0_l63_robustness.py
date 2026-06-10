"""Phase 0 L63 negative-control robustness driver.

Layer 1 (robustness grid) and Layer 2 (quad-only sanity arm) of the
L63 negative-control experiment, per the design recorded in the
phase-0 journal entry.

Layer 1 (the robustness claim):
    Sweep `(n_rbf, lambda_rbf, seed)` at fixed `lambda_poly` and
    apply the four-condition L63 acceptance rule
    (`chord2.diagnostics.l63_acceptance`) to each fit. Report the
    fraction of seeds passing per `(n_rbf, lambda_rbf)` cell. The
    negative-control claim succeeds only if a *broad band* of cells
    passes for all seeds, not just one tuned corner.

Layer 2 (the silence claim):
    Also fit a polynomial-only model (same `lambda_poly`, no RBF
    block) and, for every cell that passes Layer 1, report
    `||xi_joint_poly - xi_quad_only_poly|| / ||xi_quad_only_poly||`.
    The negative-control claim further requires this to be small at
    every passing cell -- i.e., the RBF block is not just sparse,
    it is silent.

Outputs (under `results/LOR63/phase0_robustness/`):
    heatmap_pass.png        : pass fraction over `(n_rbf, lambda_rbf)`
    heatmap_per_condition.png : per-condition cell statistics
    grid_results.json       : one record per fit
    quad_vs_joint.json      : Layer 2 silence check
    summary.json            : aggregate verdict
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
from chord2.diagnostics import l63_acceptance, phase0_summary
from scripts.run_phase0_l96 import _fit_quad_only


# Module-level holder for the shared dataset, populated in each worker by
# `_pool_initializer`. Passing `A` via this side channel keeps the per-task
# pickled payload tiny (just the three sweep coordinates) and avoids
# re-pickling the 50000x3 array on every cell.
_WORKER_CTX: dict = {}


def _pool_initializer(A, dt, lambda_poly, gamma, n_knn,
                      sigma_phys, rho_phys, beta_phys, quad_xi_poly):
    _WORKER_CTX["A"] = A
    _WORKER_CTX["dt"] = dt
    _WORKER_CTX["lambda_poly"] = lambda_poly
    _WORKER_CTX["gamma"] = gamma
    _WORKER_CTX["n_knn"] = n_knn
    _WORKER_CTX["sigma"] = sigma_phys
    _WORKER_CTX["rho"] = rho_phys
    _WORKER_CTX["beta"] = beta_phys
    _WORKER_CTX["quad_xi_poly"] = quad_xi_poly
    _WORKER_CTX["quad_poly_norm"] = float(
        np.linalg.norm(quad_xi_poly)) or 1e-300


def _fit_one_cell(task):
    """Worker: fit one (n_rbf, lambda_rbf, seed) cell, return its record.

    Module-level so it pickles cleanly with `multiprocessing.Pool`.
    """
    i, j, k, n_rbf, lambda_rbf, seed = task
    A = _WORKER_CTX["A"]
    dt = _WORKER_CTX["dt"]

    model = sindy.fit_phase0(
        A, dt,
        rbf_kind="flat_isotropic",
        rbf_kwargs={"n_rbf": int(n_rbf), "seed": int(seed),
                    "gamma": _WORKER_CTX["gamma"],
                    "n_knn": _WORKER_CTX["n_knn"]},
        lambda_poly=_WORKER_CTX["lambda_poly"],
        lambda_rbf=float(lambda_rbf),
        constrain_energy=True,
        drop_for_cond=False,
    )
    acc = l63_acceptance(model,
                         sigma=_WORKER_CTX["sigma"],
                         rho=_WORKER_CTX["rho"],
                         beta=_WORKER_CTX["beta"])
    joint_vs_quad_rel = float(
        np.linalg.norm(model.xi_poly - _WORKER_CTX["quad_xi_poly"])
        / _WORKER_CTX["quad_poly_norm"]
    )
    per_coeff = acc["cond3_coeffs"]["per_coeff"]
    sigma_rec = 0.5 * (per_coeff["sigma_dx_a1"]["fit"]
                       - per_coeff["sigma_dx_a0"]["fit"])
    rho_rec = per_coeff["rho_dy_a0"]["fit"]
    beta_rec = -per_coeff["beta_dz_a2"]["fit"]
    return {
        "i": int(i), "j": int(j), "k": int(k),
        "n_rbf": int(n_rbf),
        "lambda_rbf": float(lambda_rbf),
        "seed": int(seed),
        "pass": bool(acc["pass"]),
        "cond1_ratio": float(acc["cond1_ratio"]["value"]),
        "cond2_nnz_rbf": int(acc["cond2_nnz_rbf"]["value"]),
        "cond3_max_rel_err": float(acc["cond3_coeffs"]["max_rel_err"]),
        "cond4_pass": bool(acc["cond4_jacobians"]["pass"]),
        "quad_vs_joint_rel": joint_vs_quad_rel,
        "sigma_rec": float(sigma_rec),
        "rho_rec": float(rho_rec),
        "beta_rec": float(beta_rec),
        "sigma_err": float(sigma_rec - _WORKER_CTX["sigma"]),
        "rho_err": float(rho_rec - _WORKER_CTX["rho"]),
        "beta_err": float(beta_rec - _WORKER_CTX["beta"]),
    }


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _plot_pass_heatmap(out_path: Path, pass_frac: np.ndarray,
                       n_rbf_axis, lambda_rbf_axis,
                       n_seeds: int, lambda_poly: float):
    """Heatmap of pass fraction over `(n_rbf, lambda_rbf)`."""
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pass_frac, origin="lower", cmap="RdYlGn",
                   vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(lambda_rbf_axis)))
    ax.set_xticklabels([f"{x:g}" for x in lambda_rbf_axis])
    ax.set_yticks(range(len(n_rbf_axis)))
    ax.set_yticklabels([str(x) for x in n_rbf_axis])
    ax.set_xlabel("$\\lambda_{\\rm rbf}$")
    ax.set_ylabel("$n_{\\rm rbf}$")
    ax.set_title(f"L63 negative control: fraction of seeds passing\n"
                 f"({n_seeds} seeds, $\\lambda_{{\\rm poly}}$={lambda_poly:g}, "
                 f"4-condition acceptance)")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("pass fraction")
    for i in range(pass_frac.shape[0]):
        for j in range(pass_frac.shape[1]):
            ax.text(j, i, f"{pass_frac[i, j]:.2f}",
                    ha="center", va="center", fontsize=9,
                    color="white" if pass_frac[i, j] < 0.5 else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_lorenz_coeffs(out_path: Path,
                        grid_sigma, grid_rho, grid_beta,
                        sigma_truth, rho_truth, beta_truth,
                        n_rbf_axis, lambda_rbf_axis):
    """Three-panel heatmap of recovered (sigma, rho, beta).

    Each cell shows the mean recovered value across seeds, with the
    truth annotated; the colourbar is log10 of mean absolute error.
    A uniform near-zero field is the negative-control's strong reading:
    even when (cond1, cond2) fail because the RBF block has support,
    the polynomial block still recovers Lorenz to many digits.
    """
    truths = {"sigma": sigma_truth, "rho": rho_truth, "beta": beta_truth}
    grids = {"sigma": grid_sigma, "rho": grid_rho, "beta": grid_beta}
    names = ["sigma", "rho", "beta"]
    labels = {"sigma": "$\\sigma$ (truth=%g)" % sigma_truth,
              "rho": "$\\rho$ (truth=%g)" % rho_truth,
              "beta": "$\\beta$ (truth=%.4f)" % beta_truth}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    for ax, name in zip(axes, names):
        mean_rec = grids[name].mean(axis=2)
        abs_err = np.abs(mean_rec - truths[name])
        disp = np.log10(np.maximum(abs_err, 1e-16))
        im = ax.imshow(disp, origin="lower", cmap="viridis_r", aspect="auto")
        ax.set_title(f"recovered {labels[name]}")
        ax.set_xticks(range(len(lambda_rbf_axis)))
        ax.set_xticklabels([f"{x:g}" for x in lambda_rbf_axis])
        ax.set_yticks(range(len(n_rbf_axis)))
        ax.set_yticklabels([str(x) for x in n_rbf_axis])
        ax.set_xlabel("$\\lambda_{\\rm rbf}$")
        ax.set_ylabel("$n_{\\rm rbf}$")
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("$\\log_{10}$ mean $|$recovered $-$ truth$|$")
        for i in range(mean_rec.shape[0]):
            for j in range(mean_rec.shape[1]):
                ax.text(j, i, f"{mean_rec[i, j]:.3f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if disp[i, j] > disp.min() + 0.5 * (
                            disp.max() - disp.min()) else "black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_per_condition(out_path: Path,
                        grid_ratio, grid_nnz, grid_max_rel, grid_jac_pass,
                        n_rbf_axis, lambda_rbf_axis):
    """Four sub-heatmaps, one per acceptance condition."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    # cond1: log10(mean ratio)
    field = np.log10(np.maximum(grid_ratio.mean(axis=2), 1e-12))
    im = axes[0].imshow(field, origin="lower", cmap="viridis_r", aspect="auto")
    axes[0].set_title("cond1: $\\log_{10}$ mean $||\\xi_{\\rm rbf}||/||\\xi_{\\rm poly}||$")
    fig.colorbar(im, ax=axes[0])
    axes[0].axhline(-0.5)  # placeholder for axis aesthetics

    # cond2: mean nnz_rbf
    field = grid_nnz.astype(np.float64).mean(axis=2)
    im = axes[1].imshow(field, origin="lower", cmap="viridis_r", aspect="auto")
    axes[1].set_title("cond2: mean $n_{\\rm nz}^{\\rm rbf}$")
    fig.colorbar(im, ax=axes[1])

    # cond3: log10(mean max_rel_err)
    field = np.log10(np.maximum(grid_max_rel.mean(axis=2), 1e-12))
    im = axes[2].imshow(field, origin="lower", cmap="viridis_r", aspect="auto")
    axes[2].set_title("cond3: $\\log_{10}$ mean Lorenz-coeff max rel err")
    fig.colorbar(im, ax=axes[2])

    # cond4: mean jac-correct fraction
    field = grid_jac_pass.astype(np.float64).mean(axis=2)
    im = axes[3].imshow(field, origin="lower", cmap="RdYlGn",
                        vmin=0, vmax=1, aspect="auto")
    axes[3].set_title("cond4: mean jac unstable-count pass")
    fig.colorbar(im, ax=axes[3])

    for ax in axes:
        ax.set_xticks(range(len(lambda_rbf_axis)))
        ax.set_xticklabels([f"{x:g}" for x in lambda_rbf_axis], rotation=45,
                           fontsize=8)
        ax.set_yticks(range(len(n_rbf_axis)))
        ax.set_yticklabels([str(x) for x in n_rbf_axis])
        ax.set_xlabel("$\\lambda_{\\rm rbf}$")
        ax.set_ylabel("$n_{\\rm rbf}$")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n-rbf", type=int, nargs="*",
                   default=[20, 50, 100, 200],
                   help="centre-count axis (default: 20 50 100 200)")
    p.add_argument("--lambda-rbf", type=float, nargs="*",
                   default=[0.1, 0.3, 1.0, 3.0, 10.0],
                   help="STLSQ RBF threshold axis (default: log-spaced 0.1..10)")
    p.add_argument("--seeds", type=int, nargs="*",
                   default=[0, 1, 2, 3, 4],
                   help="seed axis for FPS/centre variance (default: 0..4)")
    p.add_argument("--lambda-poly", type=float, default=1.0,
                   help="STLSQ threshold on normalised polynomial block")
    p.add_argument("--gamma", type=float, default=1.0,
                   help="isotropic-width scale (multiplier on 5-NN median)")
    p.add_argument("--n-knn", type=int, default=5,
                   help="neighbours used in the width heuristic")
    p.add_argument("--workers", type=int, default=1,
                   help="multiprocessing workers (default: 1, i.e. serial). "
                        "Each cell is an independent fit. Recommended on "
                        "SLURM: --workers == SLURM_CPUS_PER_TASK.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="override default results/LOR63/phase0_robustness/")
    args = p.parse_args()

    ds = data.load("LOR63")
    A = ds.u.astype(np.float64)
    dt = float(ds.t[1] - ds.t[0])
    sigma_phys = float(ds.metadata.get("sigma", 10.0))
    rho_phys = float(ds.metadata.get("rho", 28.0))
    beta_phys = float(ds.metadata.get("beta", 8.0 / 3.0))
    print(f"Loaded {ds}; sigma={sigma_phys}, rho={rho_phys}, beta={beta_phys:.4f}")

    out_dir = args.out_dir or (data.results_dir("LOR63") / "phase0_robustness")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Layer 2 baseline: quad-only fit ---")
    quad_model = _fit_quad_only(A, dt, args.lambda_poly)
    quad_acc = l63_acceptance(quad_model, sigma=sigma_phys, rho=rho_phys,
                              beta=beta_phys)
    quad_summary = phase0_summary(quad_model)
    print(f"  quad-only PASS = {quad_acc['pass']}")
    print(f"    cond3 max_rel_err: "
          f"{quad_acc['cond3_coeffs']['max_rel_err']:.3e}")
    print(f"    cond4 pass       : {quad_acc['cond4_jacobians']['pass']}")
    quad_poly_norm = float(np.linalg.norm(quad_model.xi_poly))

    n_n = len(args.n_rbf)
    n_l = len(args.lambda_rbf)
    n_s = len(args.seeds)
    grid_pass = np.zeros((n_n, n_l, n_s), dtype=bool)
    grid_ratio = np.zeros((n_n, n_l, n_s))
    grid_nnz = np.zeros((n_n, n_l, n_s), dtype=int)
    grid_max_rel = np.zeros((n_n, n_l, n_s))
    grid_jac_pass = np.zeros((n_n, n_l, n_s), dtype=bool)
    grid_sigma = np.zeros((n_n, n_l, n_s))
    grid_rho = np.zeros((n_n, n_l, n_s))
    grid_beta = np.zeros((n_n, n_l, n_s))

    tasks = [
        (i, j, k, int(n_rbf), float(lam_rbf), int(seed))
        for i, n_rbf in enumerate(args.n_rbf)
        for j, lam_rbf in enumerate(args.lambda_rbf)
        for k, seed in enumerate(args.seeds)
    ]
    grid_records = [None] * len(tasks)
    quad_vs_joint = []

    workers = max(1, int(args.workers))
    print(f"\n--- Layer 1 grid: "
          f"{n_n} x {n_l} x {n_s} = {len(tasks)} fits "
          f"(workers={workers}) ---")

    init_args = (A, dt, args.lambda_poly, args.gamma, args.n_knn,
                 sigma_phys, rho_phys, beta_phys, quad_model.xi_poly)

    if workers == 1:
        _pool_initializer(*init_args)
        rec_iter = (_fit_one_cell(t) for t in tasks)
        ctx = None
    else:
        ctx = mp.get_context("fork")
        pool = ctx.Pool(processes=workers,
                        initializer=_pool_initializer,
                        initargs=init_args)
        rec_iter = pool.imap_unordered(_fit_one_cell, tasks, chunksize=1)

    n_done = 0
    for rec in rec_iter:
        i, j, k = rec["i"], rec["j"], rec["k"]
        grid_pass[i, j, k] = rec["pass"]
        grid_ratio[i, j, k] = rec["cond1_ratio"]
        grid_nnz[i, j, k] = rec["cond2_nnz_rbf"]
        grid_max_rel[i, j, k] = rec["cond3_max_rel_err"]
        grid_jac_pass[i, j, k] = rec["cond4_pass"]
        grid_sigma[i, j, k] = rec["sigma_rec"]
        grid_rho[i, j, k] = rec["rho_rec"]
        grid_beta[i, j, k] = rec["beta_rec"]
        flat_idx = i * (n_l * n_s) + j * n_s + k
        grid_records[flat_idx] = {kk: vv for kk, vv in rec.items()
                                  if kk not in ("i", "j", "k")}
        if rec["pass"]:
            quad_vs_joint.append({
                "n_rbf": rec["n_rbf"],
                "lambda_rbf": rec["lambda_rbf"],
                "seed": rec["seed"],
                "rel_diff": rec["quad_vs_joint_rel"],
            })
        n_done += 1
        print(f"  [{n_done:3d}/{len(tasks)}]  "
              f"n_rbf={rec['n_rbf']:3d}  lam_rbf={rec['lambda_rbf']:5.2f}  "
              f"seed={rec['seed']}: pass={str(rec['pass']):5s}  "
              f"ratio={rec['cond1_ratio']:.2e}  "
              f"nnz={rec['cond2_nnz_rbf']:3d}  "
              f"qj={rec['quad_vs_joint_rel']:.2e}  "
              f"sig={rec['sigma_rec']:.4f}  "
              f"rho={rec['rho_rec']:.4f}  "
              f"bet={rec['beta_rec']:.4f}",
              flush=True)

    if workers > 1:
        pool.close()
        pool.join()

    pass_frac = grid_pass.mean(axis=2)

    _plot_pass_heatmap(out_dir / "heatmap_pass.png", pass_frac,
                       args.n_rbf, args.lambda_rbf, n_s, args.lambda_poly)
    _plot_per_condition(out_dir / "heatmap_per_condition.png",
                        grid_ratio, grid_nnz, grid_max_rel, grid_jac_pass,
                        args.n_rbf, args.lambda_rbf)
    _plot_lorenz_coeffs(out_dir / "heatmap_lorenz_coeffs.png",
                        grid_sigma, grid_rho, grid_beta,
                        sigma_phys, rho_phys, beta_phys,
                        args.n_rbf, args.lambda_rbf)

    (out_dir / "grid_results.json").write_text(json.dumps(grid_records, indent=2))

    quad_vs_joint_sorted = sorted(quad_vs_joint, key=lambda x: x["rel_diff"])
    qj_report = {
        "quad_only_acceptance": _jsonable(quad_acc),
        "quad_only_summary": _jsonable(quad_summary),
        "passing_cells_quad_vs_joint": quad_vs_joint_sorted,
        "median_rel_diff": (
            float(np.median([x["rel_diff"] for x in quad_vs_joint_sorted]))
            if quad_vs_joint else None),
        "max_rel_diff": (
            float(np.max([x["rel_diff"] for x in quad_vs_joint_sorted]))
            if quad_vs_joint else None),
        "n_passing_cells_total_fits": len(quad_vs_joint),
    }
    (out_dir / "quad_vs_joint.json").write_text(json.dumps(qj_report, indent=2))

    n_full_pass = int((pass_frac == 1.0).sum())
    n_partial = int(((pass_frac > 0) & (pass_frac < 1.0)).sum())
    n_total_cells = n_n * n_l
    sigma_err = grid_sigma - sigma_phys
    rho_err = grid_rho - rho_phys
    beta_err = grid_beta - beta_phys
    pass_mask = grid_pass
    fail_mask = ~grid_pass

    def _stats(arr, mask):
        sel = arr[mask] if mask.any() else np.array([])
        if sel.size == 0:
            return {"n": 0, "min": None, "max": None, "median": None}
        return {"n": int(sel.size),
                "min": float(np.min(np.abs(sel))),
                "max": float(np.max(np.abs(sel))),
                "median": float(np.median(np.abs(sel)))}

    summary = {
        "n_rbf_axis": [int(x) for x in args.n_rbf],
        "lambda_rbf_axis": [float(x) for x in args.lambda_rbf],
        "seeds": [int(x) for x in args.seeds],
        "lambda_poly": float(args.lambda_poly),
        "n_total_fits": int(n_n * n_l * n_s),
        "n_cells_full_pass": n_full_pass,
        "n_cells_partial": n_partial,
        "n_cells_total": n_total_cells,
        "pass_fraction_grid": pass_frac.tolist(),
        "quad_only_pass": bool(quad_acc["pass"]),
        "quad_only_max_rel_err": float(
            quad_acc["cond3_coeffs"]["max_rel_err"]),
        "qj_median_rel_diff": qj_report["median_rel_diff"],
        "qj_max_rel_diff": qj_report["max_rel_diff"],
        "lorenz_coeff_recovery": {
            "truth": {"sigma": float(sigma_phys),
                      "rho": float(rho_phys),
                      "beta": float(beta_phys)},
            "passing_cells": {
                "sigma_abs_err": _stats(sigma_err, pass_mask),
                "rho_abs_err": _stats(rho_err, pass_mask),
                "beta_abs_err": _stats(beta_err, pass_mask),
            },
            "failing_cells": {
                "sigma_abs_err": _stats(sigma_err, fail_mask),
                "rho_abs_err": _stats(rho_err, fail_mask),
                "beta_abs_err": _stats(beta_err, fail_mask),
            },
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n--- Verdict ---")
    print(f"Cells fully passing (all {n_s} seeds): "
          f"{n_full_pass}/{n_total_cells}")
    print(f"Cells partially passing             : "
          f"{n_partial}/{n_total_cells}")
    print(f"Quad-only fit                       : "
          f"{'PASS' if quad_acc['pass'] else 'FAIL'} "
          f"(cond3 max_rel_err={quad_acc['cond3_coeffs']['max_rel_err']:.2e})")
    if quad_vs_joint:
        print(f"Quad-vs-joint over passing cells    : "
              f"median rel diff = {qj_report['median_rel_diff']:.2e}, "
              f"max = {qj_report['max_rel_diff']:.2e}")
    lcr = summary["lorenz_coeff_recovery"]
    print(f"Lorenz coeff recovery (all {pass_mask.sum() + fail_mask.sum()} fits):")
    print(f"  passing cells: |sigma-err| median={lcr['passing_cells']['sigma_abs_err']['median']}, "
          f"|rho-err| median={lcr['passing_cells']['rho_abs_err']['median']}, "
          f"|beta-err| median={lcr['passing_cells']['beta_abs_err']['median']}")
    print(f"  failing cells: |sigma-err| median={lcr['failing_cells']['sigma_abs_err']['median']}, "
          f"|rho-err| median={lcr['failing_cells']['rho_abs_err']['median']}, "
          f"|beta-err| median={lcr['failing_cells']['beta_abs_err']['median']}")
    print(f"\nOutputs under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
