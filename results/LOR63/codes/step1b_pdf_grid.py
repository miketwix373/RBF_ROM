"""Step 1b: marginal-PDF grid for a curated set of n_rbf values.

For each n_rbf in --n-rbf, fit a K=1 isotropic RBF-only SINDy model
(seed=1, lambda_rbf=1e-3 by default — same as step 1), integrate 8
ICs RK4 to T=50 s at dt=0.005, pool the tail [t0, T] s across ICs,
and plot marginal histograms of (x, y, z) with the FOM truth as the
filled reference.

Output: results/LOR63/step1_nrbf_sweep/pdf_grid_marginals.png
(rows = n_rbf, cols = x, y, z).

The dictionary is purely Gaussian RBFs — no constant, no linear,
no polynomial features. This script documents that explicitly in
the header print so anyone reading the .out file does not have to
chase it down.
"""

from __future__ import annotations

import argparse
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
    f_rbf, f_truth_l63_single, fit_rbf_only_keep_state, rk4_integrate,
)
from chord2 import data, sindy  # noqa: E402


def _integrate_fom_truth(x0: np.ndarray, dt: float, n_steps: int) -> np.ndarray:
    X = np.empty((n_steps + 1, 3))
    X[0] = x0
    a = x0.copy()
    for i in range(n_steps):
        k1 = f_truth_l63_single(a)
        k2 = f_truth_l63_single(a + 0.5 * dt * k1)
        k3 = f_truth_l63_single(a + 0.5 * dt * k2)
        k4 = f_truth_l63_single(a + dt * k3)
        a = a + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        X[i + 1] = a
    return X


_WORKER_STATE: dict = {}


def _worker_init(A_mid, dAdt_true, ICs, dt, n_steps,
                 lambda_rbf, gamma, n_knn, n_init, seed, long_n_lo):
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _WORKER_STATE["A_mid"] = A_mid
    _WORKER_STATE["dAdt_true"] = dAdt_true
    _WORKER_STATE["ICs"] = ICs
    _WORKER_STATE["dt"] = dt
    _WORKER_STATE["n_steps"] = n_steps
    _WORKER_STATE["lambda_rbf"] = lambda_rbf
    _WORKER_STATE["gamma"] = gamma
    _WORKER_STATE["n_knn"] = n_knn
    _WORKER_STATE["n_init"] = n_init
    _WORKER_STATE["seed"] = seed
    _WORKER_STATE["long_n_lo"] = long_n_lo


def _fit_and_collect_tail(n_rbf: int) -> dict:
    t0 = time.time()
    A_mid = _WORKER_STATE["A_mid"]
    dAdt_true = _WORKER_STATE["dAdt_true"]
    ICs = _WORKER_STATE["ICs"]
    dt = _WORKER_STATE["dt"]
    n_steps = _WORKER_STATE["n_steps"]
    long_n_lo = _WORKER_STATE["long_n_lo"]

    rbf = fit_rbf_only_keep_state(
        A_mid, dAdt_true,
        n_rbf=int(n_rbf), seed=_WORKER_STATE["seed"],
        lambda_rbf=_WORKER_STATE["lambda_rbf"],
        gamma=_WORKER_STATE["gamma"],
        n_knn=_WORKER_STATE["n_knn"],
        n_init=_WORKER_STATE["n_init"],
    )

    def f_cb(a: np.ndarray) -> np.ndarray:
        return f_rbf(a[None, :], rbf["centers"], rbf["widths"],
                     rbf["mu_A"], rbf["sigma_A"],
                     rbf["col_norms"], rbf["xi"])[0]

    tails = []
    for j in range(ICs.shape[0]):
        _, X_pred = rk4_integrate(f_cb, ICs[j], dt, n_steps)
        if X_pred.shape[0] > long_n_lo:
            tail = X_pred[long_n_lo:]
            tail = tail[np.all(np.isfinite(tail), axis=1)]
            tails.append(tail)
    pooled = (np.concatenate(tails, axis=0)
              if any(t.shape[0] > 0 for t in tails)
              else np.zeros((0, 3)))
    return {"n_rbf": int(n_rbf),
            "tail": pooled.astype(np.float32),
            "t_seconds": time.time() - t0}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n-rbf", type=int, nargs="+",
                   default=[5, 20, 100, 800, 3200, 6400])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--lambda-rbf", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--n-knn", type=int, default=5)
    p.add_argument("--n-init", type=int, default=1)
    p.add_argument("--n-ic", type=int, default=8)
    p.add_argument("--ic-rng-seed", type=int, default=0)
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--T", type=float, default=50.0)
    p.add_argument("--long-window-t0", type=float, default=10.0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--bins", type=int, default=80)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    print("Dictionary: pure Gaussian RBFs — no constant, no linear, "
          "no polynomial features. xi lives entirely in RBF-feature space.")
    print(f"L63 parameters: sigma={SIGMA_L63}, rho={RHO_L63}, beta={BETA_L63:.4f}")

    ds = data.load("LOR63")
    A = ds.u.astype(np.float64)
    dt_data = float(ds.t[1] - ds.t[0])
    print(f"Loaded {ds}; dt_data={dt_data:g}, M={A.shape[0]}")

    out_dir = args.out_dir or (data.results_dir("LOR63") / "step1_nrbf_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    A_mid, dAdt_true = sindy.deriv_5point(A, dt_data)

    rng_ic = np.random.default_rng(args.ic_rng_seed)
    ic_idx = rng_ic.choice(A.shape[0], size=args.n_ic, replace=False)
    ic_idx.sort()
    ICs = A[ic_idx]
    n_steps = int(round(args.T / args.dt))
    long_n_lo = int(round(args.long_window_t0 / args.dt))

    print(f"\nPre-integrating {args.n_ic} FOM trajectories at dt={args.dt}, "
          f"T={args.T} ({n_steps} steps)...")
    X_true_all = np.empty((args.n_ic, n_steps + 1, 3))
    for j in range(args.n_ic):
        X_true_all[j] = _integrate_fom_truth(ICs[j], args.dt, n_steps)
    truth_tail = X_true_all[:, long_n_lo:, :].reshape(-1, 3)
    print(f"  done. FOM truth tail pool: {truth_tail.shape[0]} samples")

    n_rbf_list = sorted(set(int(n) for n in args.n_rbf))
    print(f"\nSweeping n_rbf = {n_rbf_list} with {args.workers} workers "
          f"(seed={args.seed}, lambda_rbf={args.lambda_rbf})...")

    init_args = (A_mid, dAdt_true, ICs, args.dt, n_steps,
                 args.lambda_rbf, args.gamma, args.n_knn, args.n_init,
                 args.seed, long_n_lo)
    t_pool = time.time()
    results = []
    with mp.Pool(processes=args.workers, initializer=_worker_init,
                 initargs=init_args) as pool:
        for r in pool.imap_unordered(_fit_and_collect_tail, n_rbf_list):
            results.append(r)
            print(f"  [{time.time() - t_pool:6.1f}s] "
                  f"n_rbf={r['n_rbf']:>5d}  "
                  f"tail samples={r['tail'].shape[0]:>7d}  "
                  f"t={r['t_seconds']:.1f}s",
                  flush=True)
    print(f"\nPool finished in {time.time() - t_pool:.1f} s.")

    results.sort(key=lambda r: r["n_rbf"])

    var_names = ["x", "y", "z"]
    xlims = []
    for d in range(3):
        lo = np.percentile(truth_tail[:, d], 0.5)
        hi = np.percentile(truth_tail[:, d], 99.5)
        pad = 0.15 * (hi - lo)
        xlims.append((lo - pad, hi + pad))

    n_rows = len(results)
    fig, axes = plt.subplots(n_rows, 3,
                             figsize=(11, 1.9 * n_rows),
                             sharex="col", sharey=False)
    if n_rows == 1:
        axes = axes[None, :]
    truth_color = "0.55"
    rbf_color = "tab:red"

    for i, r in enumerate(results):
        rbf_tail = r["tail"]
        for d in range(3):
            ax = axes[i, d]
            lo, hi = xlims[d]
            bins = np.linspace(lo, hi, args.bins + 1)
            ax.hist(truth_tail[:, d], bins=bins, density=True,
                    color=truth_color, alpha=0.55,
                    label="FOM truth" if (i == 0 and d == 0) else None)
            if rbf_tail.shape[0] > 0:
                rbf_in = rbf_tail[(rbf_tail[:, d] >= lo) & (rbf_tail[:, d] <= hi), d]
                if rbf_in.size > 0:
                    counts, edges = np.histogram(rbf_in, bins=bins, density=True)
                    rbf_frac_in = rbf_in.size / rbf_tail.shape[0]
                    counts = counts * rbf_frac_in
                    centres = 0.5 * (edges[:-1] + edges[1:])
                    ax.step(centres, counts, where="mid", color=rbf_color, lw=1.4,
                            label="RBF-only" if (i == 0 and d == 0) else None)
            ax.set_xlim(lo, hi)
            if i == 0:
                ax.set_title(rf"${var_names[d]}$")
            if d == 0:
                ax.set_ylabel(rf"$n_{{\rm rbf}}={r['n_rbf']}$",
                              rotation=0, ha="right", va="center",
                              labelpad=22, fontsize=10)
            if i == n_rows - 1:
                ax.set_xlabel(rf"${var_names[d]}$")
            ax.tick_params(labelsize=8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2,
                   bbox_to_anchor=(0.5, 1.00), frameon=False)
    fig.suptitle("L63 RBF-only K=1 isotropic: marginal-PDF grid "
                 r"(tail $t \in [10, 50]$ s)", y=1.02)
    fig.tight_layout()
    out_path = out_dir / "pdf_grid_marginals.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
