"""Step 1c: fit + save the curated-n_rbf models for later inspection.

For each n_rbf in --n-rbf, fit a K=1 isotropic RBF-only SINDy model
with the same hyperparameters as step 1 (seed=1, lambda_rbf=1e-3,
gamma=1.0, n_knn=5), and save the full model state to
`results/LOR63/step1_nrbf_sweep/models/model_n_rbf_{n:05d}.npz`.

This is a pure-fit script — no integration, no plotting. Each npz
contains:
  - centers   (n_rbf, 3)   RBF centres in standardised coords
  - widths    (n_rbf,)     Gaussian widths
  - mu_A      (3,)         standardisation mean
  - sigma_A   (3,)         standardisation std
  - col_norms (n_rbf,)     column-normalisation constants
  - xi        (n_rbf, 3)   coefficient matrix
  - active    (n_rbf,)     bool: which features survived STLSQ
  - meta      dict-as-0d   {n_rbf, seed, lambda_rbf, gamma, n_knn,
                            n_init, rel_err_alpha_dot, nnz, t_fit_s}

A `manifest.json` summarises config + per-cell rel_err / nnz / t_fit
across the sweep.

Dictionary is purely Gaussian RBFs — no constant, no linear, no
polynomial features.
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
    SIGMA_L63, RHO_L63, BETA_L63,
    fit_rbf_only_keep_state,
)
from chord2 import data, sindy  # noqa: E402


_WORKER_STATE: dict = {}


def _worker_init(A_mid, dAdt_true,
                 lambda_rbf, gamma, n_knn, n_init, seed):
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _WORKER_STATE["A_mid"] = A_mid
    _WORKER_STATE["dAdt_true"] = dAdt_true
    _WORKER_STATE["lambda_rbf"] = lambda_rbf
    _WORKER_STATE["gamma"] = gamma
    _WORKER_STATE["n_knn"] = n_knn
    _WORKER_STATE["n_init"] = n_init
    _WORKER_STATE["seed"] = seed


def _fit_one(n_rbf: int) -> dict:
    t0 = time.time()
    A_mid = _WORKER_STATE["A_mid"]
    dAdt_true = _WORKER_STATE["dAdt_true"]
    rbf = fit_rbf_only_keep_state(
        A_mid, dAdt_true,
        n_rbf=int(n_rbf), seed=_WORKER_STATE["seed"],
        lambda_rbf=_WORKER_STATE["lambda_rbf"],
        gamma=_WORKER_STATE["gamma"],
        n_knn=_WORKER_STATE["n_knn"],
        n_init=_WORKER_STATE["n_init"],
    )
    Phi_n = sindy.rbf_features_iso(
        A_mid, rbf["centers"], rbf["widths"],
        mu_A=rbf["mu_A"], sigma_A=rbf["sigma_A"],
    ) / rbf["col_norms"]
    dAdt_pred = Phi_n @ rbf["xi"]
    rel_err = float(np.linalg.norm(dAdt_true - dAdt_pred)
                    / np.linalg.norm(dAdt_true))
    active = (np.max(np.abs(rbf["xi"]), axis=1) >= _WORKER_STATE["lambda_rbf"])
    return {
        "n_rbf": int(n_rbf),
        "centers": rbf["centers"],
        "widths": rbf["widths"],
        "mu_A": rbf["mu_A"],
        "sigma_A": rbf["sigma_A"],
        "col_norms": rbf["col_norms"],
        "xi": rbf["xi"],
        "active": active,
        "rel_err_alpha_dot": rel_err,
        "nnz": int(active.sum()),
        "t_fit_seconds": time.time() - t0,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n-rbf", type=int, nargs="+",
                   default=[5, 20, 100, 800, 3200, 6400])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--lambda-rbf", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--n-knn", type=int, default=5)
    p.add_argument("--n-init", type=int, default=1)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    print("Dictionary: pure Gaussian RBFs — no constant, no linear, "
          "no polynomial features.")
    print(f"L63 parameters: sigma={SIGMA_L63}, rho={RHO_L63}, beta={BETA_L63:.4f}")

    ds = data.load("LOR63")
    A = ds.u.astype(np.float64)
    dt_data = float(ds.t[1] - ds.t[0])
    print(f"Loaded {ds}; dt_data={dt_data:g}, M={A.shape[0]}")

    out_root = args.out_dir or (data.results_dir("LOR63") / "step1_nrbf_sweep")
    models_dir = out_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving models under {models_dir}")

    A_mid, dAdt_true = sindy.deriv_5point(A, dt_data)

    n_rbf_list = sorted(set(int(n) for n in args.n_rbf))
    print(f"\nFitting n_rbf = {n_rbf_list} with {args.workers} workers "
          f"(seed={args.seed}, lambda_rbf={args.lambda_rbf})...")

    init_args = (A_mid, dAdt_true, args.lambda_rbf, args.gamma,
                 args.n_knn, args.n_init, args.seed)
    t_pool = time.time()
    results = []
    with mp.Pool(processes=args.workers, initializer=_worker_init,
                 initargs=init_args) as pool:
        for r in pool.imap_unordered(_fit_one, n_rbf_list):
            out_path = models_dir / f"model_n_rbf_{r['n_rbf']:05d}.npz"
            meta = {
                "n_rbf": r["n_rbf"],
                "seed": args.seed,
                "lambda_rbf": args.lambda_rbf,
                "gamma": args.gamma,
                "n_knn": args.n_knn,
                "n_init": args.n_init,
                "rel_err_alpha_dot": r["rel_err_alpha_dot"],
                "nnz": r["nnz"],
                "t_fit_seconds": r["t_fit_seconds"],
            }
            np.savez(
                out_path,
                centers=r["centers"], widths=r["widths"],
                mu_A=r["mu_A"], sigma_A=r["sigma_A"],
                col_norms=r["col_norms"], xi=r["xi"], active=r["active"],
                meta=np.array(meta, dtype=object),
            )
            results.append(r)
            print(f"  [{time.time() - t_pool:6.1f}s] "
                  f"n_rbf={r['n_rbf']:>5d}  "
                  f"rel_err={r['rel_err_alpha_dot']:.3e}  "
                  f"nnz={r['nnz']:>5d}/{r['n_rbf']}  "
                  f"t={r['t_fit_seconds']:.1f}s  -> {out_path.name}",
                  flush=True)
    print(f"\nPool finished in {time.time() - t_pool:.1f} s.")

    results.sort(key=lambda r: r["n_rbf"])
    manifest = {
        "config": {
            "n_rbf_grid": n_rbf_list,
            "seed": args.seed,
            "lambda_rbf": args.lambda_rbf,
            "gamma": args.gamma,
            "n_knn": args.n_knn,
            "n_init": args.n_init,
            "M": int(A_mid.shape[0]),
            "N": int(A_mid.shape[1]),
            "dt_data": dt_data,
            "sigma_l63": SIGMA_L63,
            "rho_l63": RHO_L63,
            "beta_l63": BETA_L63,
        },
        "models": [
            {
                "n_rbf": r["n_rbf"],
                "file": f"model_n_rbf_{r['n_rbf']:05d}.npz",
                "rel_err_alpha_dot": r["rel_err_alpha_dot"],
                "nnz": r["nnz"],
                "t_fit_seconds": r["t_fit_seconds"],
            }
            for r in results
        ],
    }
    (models_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {models_dir / 'manifest.json'}")

    # -- Amplitude distribution figure ---------------------------------------
    # Per-model histograms of ||xi_j||_inf (matches STLSQ pruning rule)
    # on a log x-axis, with the lambda_rbf threshold marked. A second
    # panel shows the rank-decay (sorted |xi| vs rank, log-log) so the
    # spread of the spectrum is visible across all models on one plot.
    n_models = len(results)
    n_cols = 3
    n_rows = int(np.ceil(n_models / n_cols))
    fig, axes = plt.subplots(n_rows + 1, n_cols,
                             figsize=(4 * n_cols, 2.6 * (n_rows + 1)),
                             squeeze=False)

    all_amps = []
    cmap = plt.get_cmap("viridis")
    colours = [cmap(t) for t in np.linspace(0.05, 0.95, n_models)]

    for k, r in enumerate(results):
        amp = np.max(np.abs(r["xi"]), axis=1)
        all_amps.append(amp)
        i, j = divmod(k, n_cols)
        ax = axes[i, j]
        amp_nonzero = amp[amp > 0]
        if amp_nonzero.size > 0:
            lo = max(args.lambda_rbf * 1e-3, amp_nonzero.min())
            hi = amp_nonzero.max() * 1.1
            bins = np.logspace(np.log10(lo), np.log10(hi), 50)
            ax.hist(amp_nonzero, bins=bins, color=colours[k], alpha=0.85,
                    edgecolor="black", linewidth=0.3)
            ax.set_xscale("log")
        ax.axvline(args.lambda_rbf, color="red", lw=1.2, ls="--",
                   label=rf"$\lambda_{{\rm rbf}}={args.lambda_rbf:g}$")
        ax.set_title(rf"$n_{{\rm rbf}}={r['n_rbf']}$  "
                     rf"(nnz={r['nnz']}/{r['n_rbf']}, "
                     rf"rel_err={r['rel_err_alpha_dot']:.2e})",
                     fontsize=9)
        if i == n_rows - 1 or k >= n_models - n_cols:
            ax.set_xlabel(r"$\|\xi_j\|_\infty$")
        if j == 0:
            ax.set_ylabel("count")
        ax.legend(fontsize=7, loc="upper left")
        ax.tick_params(labelsize=8)

    for k in range(n_models, n_rows * n_cols):
        i, j = divmod(k, n_cols)
        axes[i, j].axis("off")

    # Rank-decay panel spans the bottom row
    gs = axes[n_rows, 0].get_gridspec()
    for j in range(n_cols):
        axes[n_rows, j].remove()
    ax_rank = fig.add_subplot(gs[n_rows, :])
    for k, r in enumerate(results):
        amp = np.sort(np.max(np.abs(r["xi"]), axis=1))[::-1]
        amp_nonzero = amp[amp > 0]
        if amp_nonzero.size > 0:
            ax_rank.loglog(np.arange(1, amp_nonzero.size + 1), amp_nonzero,
                           color=colours[k], lw=1.4,
                           label=rf"$n_{{\rm rbf}}={r['n_rbf']}$")
    ax_rank.axhline(args.lambda_rbf, color="red", lw=1.0, ls="--")
    ax_rank.set_xlabel("rank")
    ax_rank.set_ylabel(r"$\|\xi_j\|_\infty$ (sorted)")
    ax_rank.set_title(r"Rank-decay of active-feature amplitudes  "
                      r"(red dashed = $\lambda_{\rm rbf}$ threshold)",
                      fontsize=10)
    ax_rank.legend(fontsize=8, ncol=min(n_models, 6),
                   loc="lower left", frameon=False)
    ax_rank.grid(True, which="both", alpha=0.3)
    ax_rank.tick_params(labelsize=8)

    fig.suptitle("L63 RBF-only K=1 isotropic: amplitude distributions "
                 r"($\|\xi_j\|_\infty$ across output dims)", y=1.00)
    fig.tight_layout()
    out_fig = out_root / "amplitude_distributions.png"
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
