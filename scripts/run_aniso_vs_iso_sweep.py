"""Static alpha_dot relative error vs total RBFs: K=1 isotropic vs K>=2 anisotropic.

Sweeps K in {1, 2, 3, 4} and n_per_cluster in {25, 50, 100, 200, 400, 800}.
For each cell, fits an RBF-only STLSQ model on the resolved coordinate of
the chosen dataset (default LOR63), records the static alpha_dot relative
error, and stores the full model state so any later analysis (dead-zone
shell, off-cloud probes, forward integration) can replay without refitting.

K=1 cells use isotropic kernels with `attr_scaled` widths (gamma=1, the
L63 RBF-only winner; see docs/notes/l63-rbf-bandwidth-rules.md).

K>=2 cells use the general per-cluster anisotropic-PCA construction added
in chord2/sindy.py:rbf_centers_hier_anisotropic_pca (design entry
docs/journal/2026-06-10-anisotropic-rbf-per-cluster-design.md): one PCA
frame V_k per outer cluster, tangent widths sqrt(lambda_k^(p)), normal
widths from a robust statistic of pairwise centre projections onto the
cluster's normal eigenvectors.

Both branches share the rest of the recipe (column-normalised features,
STLSQ with the same lambda_rbf, Tikhonov 1e-8) so the comparison isolates
the centre/width construction.

Outputs in results/<dataset>/aniso_vs_iso_sweep/:
    fit_K{K}_n{n_per_cluster}.npz   one per cell, full model state.
    summary.json                    grid of (K, n_per_cluster, total_rbf,
                                    rel_err, n_features_kept, fit_seconds).
    rel_err_vs_total_rbfs.png       one curve per K.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chord2 import data, sindy


_WORKER_STATE: dict = {}


def _stlsq_keep_state(Phi_raw, dAdt, *, lambda_rbf, lambda_tikh, max_iter):
    """Column-normalised STLSQ; returns xi, col_norms, n_active, residual.

    Mirrors the inner loop of `_l63_rbf_lib.fit_rbf_only_keep_state` so the
    iso and aniso branches share the same regression, isolating the
    centre/width construction.
    """
    col_norms = np.linalg.norm(Phi_raw, axis=0)
    col_norms = np.where(col_norms == 0.0, 1.0, col_norms)
    Phi_n = Phi_raw / col_norms
    n_f = Phi_n.shape[1]
    active = np.ones(n_f, dtype=bool)
    prev_active = None
    xi = np.zeros((n_f, dAdt.shape[1]))
    for _ in range(max_iter):
        idx = np.where(active)[0]
        if idx.size == 0:
            break
        if lambda_tikh > 0.0:
            n_act = idx.size
            A_mat = np.vstack([
                Phi_n[:, idx],
                np.sqrt(lambda_tikh) * np.eye(n_act),
            ])
            b_mat = np.vstack([dAdt, np.zeros((n_act, dAdt.shape[1]))])
        else:
            A_mat = Phi_n[:, idx]
            b_mat = dAdt
        xi_act, *_ = np.linalg.lstsq(A_mat, b_mat, rcond=None)
        xi = np.zeros((n_f, dAdt.shape[1]))
        xi[idx, :] = xi_act
        new_active = active.copy()
        mags = np.max(np.abs(xi), axis=1)
        new_active[mags < lambda_rbf] = False
        if prev_active is not None and np.array_equal(new_active, active):
            break
        prev_active = active.copy()
        active = new_active
    dAdt_pred = Phi_n @ xi
    rel_err = float(np.linalg.norm(dAdt - dAdt_pred)
                    / max(np.linalg.norm(dAdt), 1e-30))
    return xi, col_norms, int(active.sum()), rel_err


def _fit_iso(A_mid, dAdt, *, n_rbf, seed, n_init, lambda_rbf, lambda_tikh,
             max_iter):
    centers, widths, meta = sindy.rbf_centers_flat_isotropic(
        A_mid, n_rbf, seed=seed, gamma=1.0, n_init=n_init,
        bandwidth_mode="attr_scaled",
    )
    Phi_raw = sindy.rbf_features_iso(
        A_mid, centers, widths,
        mu_A=meta["mu_A"], sigma_A=meta["sigma_A"],
    )
    xi, col_norms, nnz, rel_err = _stlsq_keep_state(
        Phi_raw, dAdt,
        lambda_rbf=lambda_rbf, lambda_tikh=lambda_tikh, max_iter=max_iter,
    )
    return {
        "mode": "iso",
        "centers": centers, "widths": widths,
        "Sigma_invs_std": None, "parent_k": None,
        "mu_A": meta["mu_A"], "sigma_A": meta["sigma_A"],
        "col_norms": col_norms, "xi": xi, "nnz": nnz, "rel_err": rel_err,
        "meta": meta,
    }


def _fit_aniso(A_mid, dAdt, *, K_shape, n_per_cluster, seed, n_init,
               lambda_rbf, lambda_tikh, max_iter, energy_threshold,
               alpha_tangent, robust_stat):
    centers_std, Sigma_invs_std, parent_k, meta = (
        sindy.rbf_centers_hier_anisotropic_pca(
            A_mid, K_shape, n_per_cluster,
            tangent_rule="energy", energy_threshold=energy_threshold,
            alpha_tangent=alpha_tangent, robust_stat=robust_stat,
            seed=seed, n_init=n_init,
        )
    )
    Phi_raw = sindy.rbf_features_mahal(
        A_mid, centers_std, Sigma_invs_std,
        mu_A=meta["mu_A"], sigma_A=meta["sigma_A"],
    )
    xi, col_norms, nnz, rel_err = _stlsq_keep_state(
        Phi_raw, dAdt,
        lambda_rbf=lambda_rbf, lambda_tikh=lambda_tikh, max_iter=max_iter,
    )
    return {
        "mode": "aniso",
        "centers": None, "widths": None,
        "centers_std": centers_std, "Sigma_invs_std": Sigma_invs_std,
        "parent_k": parent_k,
        "mu_A": meta["mu_A"], "sigma_A": meta["sigma_A"],
        "col_norms": col_norms, "xi": xi, "nnz": nnz, "rel_err": rel_err,
        "meta": meta,
    }


def _save_cell(out_dir: Path, K: int, n_per_cluster: int, fit: dict) -> Path:
    fname = out_dir / f"fit_K{K}_n{n_per_cluster}.npz"
    payload = {
        "mode": fit["mode"],
        "K_shape": K,
        "n_per_cluster": n_per_cluster,
        "mu_A": fit["mu_A"], "sigma_A": fit["sigma_A"],
        "col_norms": fit["col_norms"], "xi": fit["xi"],
        "nnz": fit["nnz"], "rel_err": fit["rel_err"],
    }
    if fit["mode"] == "iso":
        payload["centers"] = fit["centers"]
        payload["widths"] = fit["widths"]
    else:
        payload["centers_std"] = fit["centers_std"]
        payload["Sigma_invs_std"] = fit["Sigma_invs_std"]
        payload["parent_k"] = fit["parent_k"]
    # meta carries per-cluster diagnostics; np.savez handles ragged via object array.
    payload["meta"] = np.array(fit["meta"], dtype=object)
    np.savez(fname, **payload)
    return fname


def _worker_init(A_mid, dAdt, out_dir_str, hparams):
    _WORKER_STATE["A_mid"] = A_mid
    _WORKER_STATE["dAdt"] = dAdt
    _WORKER_STATE["out_dir"] = Path(out_dir_str)
    _WORKER_STATE["hparams"] = hparams


def _worker_compute(cell):
    K, n_pc = cell
    A_mid = _WORKER_STATE["A_mid"]
    dAdt = _WORKER_STATE["dAdt"]
    out_dir = _WORKER_STATE["out_dir"]
    h = _WORKER_STATE["hparams"]
    t0 = time.time()
    if K == 1:
        fit = _fit_iso(
            A_mid, dAdt, n_rbf=n_pc, seed=h["seed"], n_init=h["n_init"],
            lambda_rbf=h["lambda_rbf"], lambda_tikh=h["lambda_tikh"],
            max_iter=h["max_iter"],
        )
    else:
        fit = _fit_aniso(
            A_mid, dAdt, K_shape=K, n_per_cluster=n_pc, seed=h["seed"],
            n_init=h["n_init"], lambda_rbf=h["lambda_rbf"],
            lambda_tikh=h["lambda_tikh"], max_iter=h["max_iter"],
            energy_threshold=h["energy_threshold"],
            alpha_tangent=h["alpha_tangent"], robust_stat=h["robust_stat"],
        )
    dt_fit = time.time() - t0
    fname = _save_cell(out_dir, K, n_pc, fit)
    return {
        "K": K, "n_per_cluster": n_pc, "total_rbf": K * n_pc,
        "rel_err": fit["rel_err"], "nnz": fit["nnz"],
        "fit_seconds": dt_fit, "fname": fname.name,
    }


def _plot(summary: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    by_K: dict[int, list[tuple[int, float]]] = {}
    for row in summary:
        by_K.setdefault(row["K"], []).append((row["total_rbf"], row["rel_err"]))
    for K in sorted(by_K):
        pts = sorted(by_K[K])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        label = "K=1, iso (attr_scaled)" if K == 1 else f"K={K}, aniso-PCA"
        ax.loglog(xs, ys, "o-", label=label)
    ax.set_xlabel("total RBFs ($K \\cdot n_{\\mathrm{per\\,cluster}}$)")
    ax.set_ylabel(r"static $\dot{\alpha}$ relative error")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=str, default="LOR63")
    p.add_argument("--K-grid", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--n-grid", type=int, nargs="+",
                   default=[25, 50, 100, 200, 400, 800])
    p.add_argument("--stride", type=int, default=1,
                   help="snapshot stride into the dataset (default 1). "
                        "Over-resolved datasets like LOR96 vlachas_F8 "
                        "(dt=0.01, tau_int~0.3 s) want stride>=10 to keep "
                        "Phi tractable and decorrelate rows.")
    p.add_argument("--cells", type=str, nargs="+", default=None,
                   help="explicit cell list as K:n_per_cluster pairs "
                        "(e.g. 1:6000 10:600). Overrides --K-grid x --n-grid "
                        "when set. Use when matched-budget pairs are needed "
                        "across K with asymmetric n_per_cluster (LOR96).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-init", type=int, default=10)
    p.add_argument("--lambda-rbf", type=float, default=1e-3)
    p.add_argument("--lambda-tikh", type=float, default=1e-8)
    p.add_argument("--max-iter", type=int, default=20)
    p.add_argument("--energy-threshold", type=float, default=0.99)
    p.add_argument("--alpha-tangent", type=float, default=1.0)
    p.add_argument("--robust-stat", type=str, default="median",
                   choices=["median", "p75"])
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--workers", type=int, default=1,
                   help="parallel processes; 1 runs serial in-process")
    args = p.parse_args()

    ds = data.load(args.dataset, stride=args.stride)
    A = ds.u.astype(np.float64)
    dt_data = float(ds.t[1] - ds.t[0])
    print(f"Loaded {args.dataset} (stride={args.stride}): M={A.shape[0]}, "
          f"r={A.shape[1]}, dt={dt_data:g}")

    A_mid, dAdt = sindy.deriv_5point(A, dt_data)
    print(f"Derivatives via 5-point stencil: M_mid={A_mid.shape[0]}, "
          f"||dAdt||={np.linalg.norm(dAdt):.4g}")

    out_dir = (args.out_dir
               or (data.results_dir(args.dataset) / "aniso_vs_iso_sweep"))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    if args.cells is not None:
        cells = []
        for tok in args.cells:
            K_str, n_str = tok.split(":")
            cells.append((int(K_str), int(n_str)))
    else:
        cells = [(K, n_pc) for K in args.K_grid for n_pc in args.n_grid]
    print(f"Cells ({len(cells)}): {cells}")
    hparams = {
        "seed": args.seed, "n_init": args.n_init,
        "lambda_rbf": args.lambda_rbf, "lambda_tikh": args.lambda_tikh,
        "max_iter": args.max_iter,
        "energy_threshold": args.energy_threshold,
        "alpha_tangent": args.alpha_tangent,
        "robust_stat": args.robust_stat,
    }
    init_args = (A_mid, dAdt, str(out_dir), hparams)

    summary: list[dict] = []
    if args.workers <= 1:
        _worker_init(*init_args)
        for cell in cells:
            row = _worker_compute(cell)
            summary.append(row)
            print(f"K={row['K']:d} n_per_cluster={row['n_per_cluster']:5d} "
                  f"total={row['total_rbf']:6d}  rel_err={row['rel_err']:.3e}  "
                  f"nnz={row['nnz']:5d}  fit={row['fit_seconds']:.1f}s  "
                  f"-> {row['fname']}")
    else:
        # spawn so workers inherit a clean BLAS-pinned env (set in launcher).
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers, initializer=_worker_init,
                      initargs=init_args) as pool:
            for row in pool.imap_unordered(_worker_compute, cells):
                summary.append(row)
                print(f"K={row['K']:d} n_per_cluster={row['n_per_cluster']:5d} "
                      f"total={row['total_rbf']:6d}  "
                      f"rel_err={row['rel_err']:.3e}  "
                      f"nnz={row['nnz']:5d}  fit={row['fit_seconds']:.1f}s  "
                      f"-> {row['fname']}", flush=True)
    summary.sort(key=lambda r: (r["K"], r["n_per_cluster"]))

    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump({
            "dataset": args.dataset,
            "K_grid": list(args.K_grid),
            "n_grid": list(args.n_grid),
            "seed": args.seed,
            "lambda_rbf": args.lambda_rbf,
            "lambda_tikh": args.lambda_tikh,
            "energy_threshold": args.energy_threshold,
            "alpha_tangent": args.alpha_tangent,
            "robust_stat": args.robust_stat,
            "cells": summary,
        }, f, indent=2)
    print(f"Summary -> {summary_path}")

    plot_path = out_dir / "rel_err_vs_total_rbfs.png"
    _plot(summary, plot_path)
    print(f"Plot    -> {plot_path}")


if __name__ == "__main__":
    main()
