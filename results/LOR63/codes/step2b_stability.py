"""L63 step 2b: Lyapunov spectrum + PDF survival for clustered RBF ROM.

Stage B of the 2026-06-09 thread. Stage A (step 2a) established that
clustering monotonically hurts on-attractor alpha_dot RMSE for RBF-only
SINDy on L63 -- the U-shape was falsified. Stage B asks the dual
question: does cluster-switching trade on-attractor accuracy for
off-attractor *robustness*?

Operating point per the rom-specialist consult: N_tot=1280, where Stage A
puts all K within ~one order of magnitude of the test noise floor. Any
stability difference at this N_tot isolates the *switching* effect, not
the seeding effect.

The cluster-switched flow follows paper Algorithm 1 (Section 2.2,
Eq. 16): hard nearest-centroid affiliation, vector field jumps at
cluster boundaries. The integrator bisects on the affiliation function
so an RK4 step never straddles a switch (consultant: an O(dt) bias on
lambda_2, lambda_3 otherwise). The tangent equation `dY/dt = J(a) Y`
uses the active cluster's analytic RBF Jacobian.

Diagnostics (per the consultant):

  * **Lyapunov spectrum** via Benettin with periodic QR. Truth
    (0.906, 0, -14.572). Pass: lambda_2 +/- 0.02, lambda_3 +/- 0.3 after
    ~10^4 Lyapunov times.
  * **PDF survival** on off-attractor ICs: 8 directions x 3 radial
    scales F in {2, 5, 10} of the FOM bounding radius, T = 200 LTs.
    Records: time-to-enter the attractor box, switch rate per trajectory
    (ringing detector), W1 to FOM marginals on (x, y, z), KS to FOM
    z-marginal (bimodality witness). Primary panels: F=2 and F=5; F=10
    secondary (the FOM itself is transient there).

Cells loaded from `results/LOR63/step2a_cluster_rbf/models/`. K=1, 2, 3,
4 at the chosen N_tot; count-allocation only for K=1 (it equals
variance-allocation), variance-allocation for K>=2 (Stage A winner).

Outputs (results/LOR63/step2b_stability/):
  step2b.json
  lyapunov_spectrum.png
  survival_F<F>.png   (F=2, 5, 10)
  w1_marginals.png
  ks_z_marginal.png
  switch_rate.png
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
from scipy.stats import gaussian_kde, ks_2samp, wasserstein_distance

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _l63_rbf_lib import (  # noqa: E402
    SIGMA_L63, RHO_L63, BETA_L63, f_truth_l63_single,
)

from chord2 import data  # noqa: E402


LAMBDA_L63 = (0.906, 0.0, -14.572)
T_LYAPUNOV = 1.0 / LAMBDA_L63[0]


# ---------------------------------------------------------------------------
# Cell loading + per-cluster field/Jacobian
# ---------------------------------------------------------------------------

def load_cell(path: Path) -> dict:
    """Load a step2a cell .npz; return (centroids, clusters list, meta)."""
    d = np.load(path, allow_pickle=True)
    K = int(d["K"])
    clusters = []
    modelled = d["cluster_modelled"]
    for k in range(K):
        if not modelled[k]:
            clusters.append(None)
            continue
        clusters.append({
            "centers":    np.asarray(d[f"c{k}_centers"], dtype=np.float64),
            "widths":     np.asarray(d[f"c{k}_widths"], dtype=np.float64),
            "mu_A":       np.asarray(d[f"c{k}_mu_A"], dtype=np.float64),
            "sigma_A":    np.asarray(d[f"c{k}_sigma_A"], dtype=np.float64),
            "col_norms":  np.asarray(d[f"c{k}_col_norms"], dtype=np.float64),
            "xi":         np.asarray(d[f"c{k}_xi"], dtype=np.float64),
        })
    return {
        "K": K,
        "centroids": np.asarray(d["centroids"], dtype=np.float64),
        "clusters": clusters,
        "n_tot": int(d["n_tot"]),
        "alloc": str(d["alloc"]),
        "meta": dict(d["meta"].item()),
    }


def assign_cluster(a: np.ndarray, centroids: np.ndarray) -> int:
    """Nearest training centroid; paper Eq. 16."""
    diff = centroids - a[None, :]
    return int(np.argmin((diff * diff).sum(axis=1)))


def f_rbf_one(a: np.ndarray, cl: dict) -> np.ndarray:
    """Single-state RBF field for one cluster's fit state. Returns (3,)."""
    diffs_std = (a - cl["centers"]) / cl["sigma_A"]
    d2 = (diffs_std * diffs_std).sum(axis=1)
    phi = np.exp(-0.5 * d2 / (cl["widths"] ** 2))
    phi_n = phi / cl["col_norms"]
    return phi_n @ cl["xi"]


def J_rbf_one(a: np.ndarray, cl: dict) -> np.ndarray:
    """Analytic 3x3 Jacobian of the RBF field at single state `a`.

    d/da_i exp(-||(a - c_j)/sigma_A||^2 / (2 sigma_j^2))
        = -phi_j * (a_i - c_{j,i}) / (sigma_A_i^2 * sigma_j^2)

    Caller is responsible for picking the correct cluster.
    """
    diffs_std = (a - cl["centers"]) / cl["sigma_A"]
    d2 = (diffs_std * diffs_std).sum(axis=1)
    phi = np.exp(-0.5 * d2 / (cl["widths"] ** 2))
    factor = -phi / (cl["widths"] ** 2)
    dphi_da = (factor[:, None] * diffs_std) / cl["sigma_A"][None, :]
    dphi_n_da = dphi_da / cl["col_norms"][:, None]
    return cl["xi"].T @ dphi_n_da


# ---------------------------------------------------------------------------
# Switch-bisecting RK4
# ---------------------------------------------------------------------------

def _rk4_substep_in_cluster(a: np.ndarray, Y: np.ndarray | None,
                            dt: float, cl: dict,
                            with_tangent: bool) -> tuple:
    """One RK4 step using a single cluster's (f, J). Y is (3,3) or None.

    The Jacobian is evaluated at the same a-points as the field (joint
    RK4 on (a, Y)).
    """
    k1_a = f_rbf_one(a, cl)
    a2 = a + 0.5 * dt * k1_a
    k2_a = f_rbf_one(a2, cl)
    a3 = a + 0.5 * dt * k2_a
    k3_a = f_rbf_one(a3, cl)
    a4 = a + dt * k3_a
    k4_a = f_rbf_one(a4, cl)
    a_new = a + (dt / 6.0) * (k1_a + 2.0 * k2_a + 2.0 * k3_a + k4_a)
    if not with_tangent:
        return a_new, None
    k1_Y = J_rbf_one(a, cl) @ Y
    k2_Y = J_rbf_one(a2, cl) @ (Y + 0.5 * dt * k1_Y)
    k3_Y = J_rbf_one(a3, cl) @ (Y + 0.5 * dt * k2_Y)
    k4_Y = J_rbf_one(a4, cl) @ (Y + dt * k3_Y)
    Y_new = Y + (dt / 6.0) * (k1_Y + 2.0 * k2_Y + 2.0 * k3_Y + k4_Y)
    return a_new, Y_new


def _macro_step(a: np.ndarray, Y: np.ndarray | None, dt: float,
                cell: dict, k_current: int, with_tangent: bool,
                switch_tol: float) -> tuple:
    """Advance (a, Y) by dt with switch bisection.

    Returns (a_new, Y_new, k_new, n_switches). Repeats sub-stepping
    inside one macro step if multiple switches occur.
    """
    centroids = cell["centroids"]
    remaining = dt
    n_switches = 0
    while remaining > 0.0:
        cl = cell["clusters"][k_current]
        if cl is None:
            return a, Y, k_current, n_switches  # dead cluster
        a_try, Y_try = _rk4_substep_in_cluster(a, Y, remaining, cl, with_tangent)
        if not np.all(np.isfinite(a_try)):
            return a_try, Y_try, k_current, n_switches
        k_try = assign_cluster(a_try, centroids)
        if k_try == k_current:
            return a_try, Y_try, k_current, n_switches
        # Bisect on the sub-step time.
        lo, hi = 0.0, remaining
        a_hi, Y_hi = a_try, Y_try
        for _ in range(40):
            if (hi - lo) <= switch_tol * dt:
                break
            mid = 0.5 * (lo + hi)
            a_mid, Y_mid = _rk4_substep_in_cluster(
                a, Y, mid, cl, with_tangent,
            )
            if (not np.all(np.isfinite(a_mid))
                    or assign_cluster(a_mid, centroids) != k_current):
                hi, a_hi, Y_hi = mid, a_mid, Y_mid
            else:
                lo = mid
        a, Y = a_hi, Y_hi
        remaining -= hi
        k_current = assign_cluster(a, centroids)
        n_switches += 1
        if cell["clusters"][k_current] is None:
            return a, Y, k_current, n_switches
    return a, Y, k_current, n_switches


# ---------------------------------------------------------------------------
# Lyapunov via Benettin
# ---------------------------------------------------------------------------

def lyapunov_benettin(cell: dict, x0: np.ndarray, *,
                      dt: float, T: float, tau_renorm: float,
                      switch_tol: float) -> dict:
    """Benettin's algorithm with periodic QR on the (3,3) tangent.

    Returns dict with `lambda` (3,), running `log_growth` per direction,
    total `switches`, and `n_renorm`.
    """
    a = x0.copy().astype(np.float64)
    Y = np.eye(3, dtype=np.float64)
    centroids = cell["centroids"]
    k = assign_cluster(a, centroids)
    n_steps_per_renorm = max(1, int(round(tau_renorm / dt)))
    n_renorm = int(T / tau_renorm)
    log_sum = np.zeros(3, dtype=np.float64)
    switches_total = 0
    for r in range(n_renorm):
        for _ in range(n_steps_per_renorm):
            a, Y, k, ns = _macro_step(a, Y, dt, cell, k,
                                      with_tangent=True,
                                      switch_tol=switch_tol)
            switches_total += ns
            if not np.all(np.isfinite(a)) or not np.all(np.isfinite(Y)):
                return {
                    "lambda": np.array([np.nan] * 3),
                    "log_growth": log_sum,
                    "switches": int(switches_total),
                    "n_renorm": int(r),
                    "blew_up": True,
                    "blew_up_t": float(r * tau_renorm),
                }
        Q, R = np.linalg.qr(Y)
        diag = np.diag(R)
        log_sum += np.log(np.abs(diag) + 1e-300)
        Y = Q * np.sign(diag)
    lam = log_sum / (n_renorm * tau_renorm)
    return {
        "lambda": lam,
        "log_growth": log_sum,
        "switches": int(switches_total),
        "n_renorm": int(n_renorm),
        "blew_up": False,
    }


# ---------------------------------------------------------------------------
# PDF survival
# ---------------------------------------------------------------------------

def integrate_clustered(cell: dict, x0: np.ndarray, *,
                        dt: float, T: float, switch_tol: float,
                        sample_stride: int = 1) -> dict:
    """Integrate the cluster-switched flow; record sampled trajectory."""
    a = x0.copy().astype(np.float64)
    centroids = cell["centroids"]
    k = assign_cluster(a, centroids)
    n_steps = int(round(T / dt))
    n_out = n_steps // sample_stride + 1
    traj = np.empty((n_out, 3), dtype=np.float64)
    traj[0] = a
    j_out = 1
    switches_total = 0
    blew_up_t = None
    for i in range(1, n_steps + 1):
        a, _, k, ns = _macro_step(a, None, dt, cell, k,
                                  with_tangent=False,
                                  switch_tol=switch_tol)
        switches_total += ns
        if not np.all(np.isfinite(a)):
            blew_up_t = float(i * dt)
            break
        if i % sample_stride == 0:
            if j_out < n_out:
                traj[j_out] = a
                j_out += 1
    return {
        "traj": traj[:j_out],
        "n_samples": j_out,
        "switches": int(switches_total),
        "blew_up_t": blew_up_t,
    }


def _integrate_fom(x0: np.ndarray, dt: float, n_steps: int,
                   sample_stride: int = 1) -> np.ndarray:
    a = x0.copy().astype(np.float64)
    n_out = n_steps // sample_stride + 1
    X = np.empty((n_out, 3), dtype=np.float64)
    X[0] = a
    j_out = 1
    for i in range(1, n_steps + 1):
        k1 = f_truth_l63_single(a)
        k2 = f_truth_l63_single(a + 0.5 * dt * k1)
        k3 = f_truth_l63_single(a + 0.5 * dt * k2)
        k4 = f_truth_l63_single(a + dt * k3)
        a = a + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if i % sample_stride == 0:
            if j_out < n_out:
                X[j_out] = a
                j_out += 1
    return X[:j_out]


def make_ic_ensemble(rng: np.random.Generator, n_dir: int,
                     F_list: list, c_attr: np.ndarray,
                     r_attr: float) -> tuple:
    """Off-attractor ICs: `n_dir` directions on the sphere x len(F_list) scales.

    Directions are Fibonacci-sphere points for deterministic coverage; the
    rng decides the phase of the spiral.
    """
    phi = (1 + np.sqrt(5)) / 2
    i = np.arange(n_dir) + rng.uniform(0, 1)
    z = 1 - 2 * i / n_dir
    r = np.sqrt(np.maximum(0.0, 1 - z * z))
    theta = 2 * np.pi * i / phi
    dirs = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
    out = []
    keys = []
    for F in F_list:
        for d in range(n_dir):
            out.append(c_attr + F * r_attr * dirs[d])
            keys.append((F, d))
    return np.asarray(out), keys


def _in_box(traj: np.ndarray, bbox_lo: np.ndarray, bbox_hi: np.ndarray) -> np.ndarray:
    return np.all((traj >= bbox_lo) & (traj <= bbox_hi), axis=1)


# ---------------------------------------------------------------------------
# Worker pool
# ---------------------------------------------------------------------------

_WORKER_STATE: dict = {}


def _worker_init(cells_by_K: dict, ic_array: np.ndarray, ic_keys: list,
                 bbox_lo: np.ndarray, bbox_hi: np.ndarray,
                 fom_marginals: dict, kwargs: dict) -> None:
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _WORKER_STATE["cells_by_K"] = cells_by_K
    _WORKER_STATE["ic_array"] = ic_array
    _WORKER_STATE["ic_keys"] = ic_keys
    _WORKER_STATE["bbox_lo"] = bbox_lo
    _WORKER_STATE["bbox_hi"] = bbox_hi
    _WORKER_STATE["fom_marginals"] = fom_marginals
    _WORKER_STATE["kwargs"] = kwargs


def _run_lyapunov_task(task: tuple) -> dict:
    K = task[1]
    cell = _WORKER_STATE["cells_by_K"][K]
    kw = _WORKER_STATE["kwargs"]
    # Seed the Lyapunov IC from the FOM attractor centre for all K.
    x0 = kw["lyap_x0"]
    t0 = time.time()
    res = lyapunov_benettin(
        cell, x0, dt=kw["dt"], T=kw["T_lyap"],
        tau_renorm=kw["tau_renorm"], switch_tol=kw["switch_tol"],
    )
    res["K"] = int(K)
    res["mode"] = "lyapunov"
    res["t_seconds"] = float(time.time() - t0)
    res["lambda"] = res["lambda"].tolist()
    res["log_growth"] = res["log_growth"].tolist()
    return res


def _run_pdf_task(task: tuple) -> dict:
    _, K, ic_idx = task
    cell = _WORKER_STATE["cells_by_K"][K]
    kw = _WORKER_STATE["kwargs"]
    x0 = _WORKER_STATE["ic_array"][ic_idx]
    F, d = _WORKER_STATE["ic_keys"][ic_idx]
    bbox_lo = _WORKER_STATE["bbox_lo"]
    bbox_hi = _WORKER_STATE["bbox_hi"]
    fom = _WORKER_STATE["fom_marginals"]
    t0 = time.time()
    out = integrate_clustered(
        cell, x0, dt=kw["dt"], T=kw["T_pdf"],
        switch_tol=kw["switch_tol"],
        sample_stride=kw["sample_stride"],
    )
    traj = out["traj"]
    inside = _in_box(traj, bbox_lo, bbox_hi)
    if np.any(inside):
        t_enter = float(np.argmax(inside) * kw["dt"] * kw["sample_stride"])
        # Tail-in-box -- second half of integration once inside.
        i_enter = int(np.argmax(inside))
        i_tail_lo = max(i_enter, traj.shape[0] // 2)
        tail_in = traj[i_tail_lo:]
        tail_in = tail_in[_in_box(tail_in, bbox_lo, bbox_hi)]
    else:
        t_enter = float("inf")
        tail_in = np.zeros((0, 3))
    # Unconditioned second-half tail for marginal PDFs (regardless of box).
    # Reject runaways (|.|>1e4) so they don't dominate KDE bandwidth.
    tail_uncond = traj[traj.shape[0] // 2:]
    tail_uncond = tail_uncond[np.all(np.isfinite(tail_uncond), axis=1)]
    if tail_uncond.shape[0] > 0 and np.max(np.abs(tail_uncond)) > 1e4:
        tail_uncond = np.zeros((0, 3))
    w1 = [float("nan"), float("nan"), float("nan")]
    ks_z = float("nan")
    if tail_in.shape[0] > 50:
        for c in range(3):
            w1[c] = float(wasserstein_distance(fom["traj"][:, c], tail_in[:, c]))
        ks_z = float(ks_2samp(fom["traj"][:, 2], tail_in[:, 2]).statistic)
    return {
        "K": int(K), "ic_idx": int(ic_idx), "F": float(F), "dir": int(d),
        "mode": "pdf",
        "t_enter": t_enter,
        "blew_up_t": out["blew_up_t"],
        "switches": int(out["switches"]),
        "n_samples": int(out["n_samples"]),
        "n_tail": int(tail_in.shape[0]),
        "n_tail_uncond": int(tail_uncond.shape[0]),
        "w1": w1, "ks_z": ks_z,
        "alive": bool(np.any(inside) and out["blew_up_t"] is None),
        "tail_uncond": tail_uncond,
        "t_seconds": float(time.time() - t0),
    }


def _run_task(task: tuple) -> dict:
    if task[0] == "lyapunov":
        return _run_lyapunov_task(task)
    return _run_pdf_task(task)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n-tot", type=int, default=1280,
                   help="N_tot operating point for Stage B; default = noise floor")
    p.add_argument("--k-grid", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--alloc-K1", type=str, default="count")
    p.add_argument("--alloc-Kge2", type=str, default="variance",
                   help="Stage A's winning allocation rule for K>=2")
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--T-lyap", type=float, default=10000.0 * T_LYAPUNOV,
                   help="Lyapunov integration window (s); 1e4 Lyapunov times")
    p.add_argument("--tau-renorm", type=float, default=0.5 * T_LYAPUNOV,
                   help="Benettin QR interval (s)")
    p.add_argument("--switch-tol", type=float, default=1e-3,
                   help="switch-bisection tolerance, fraction of dt")
    p.add_argument("--T-pdf", type=float, default=200.0 * T_LYAPUNOV)
    p.add_argument("--sample-stride", type=int, default=10,
                   help="store every Nth state in the PDF integration")
    p.add_argument("--n-dir", type=int, default=8)
    p.add_argument("--F-list", type=float, nargs="+", default=[2.0, 5.0, 10.0])
    p.add_argument("--ic-seed", type=int, default=42)
    p.add_argument("--lyap-x0-seed", type=int, default=0,
                   help="seed for the Lyapunov IC draw from FOM attractor")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--models-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    models_dir = args.models_dir or (
        data.results_dir("LOR63") / "step2a_cluster_rbf" / "models"
    )
    out_dir = args.out_dir or (data.results_dir("LOR63") / "step2b_stability")
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = data.load("LOR63")
    A = ds.u.astype(np.float64)
    c_attr = A.mean(axis=0)
    r_attr = float(np.max(np.linalg.norm(A - c_attr[None, :], axis=1)))
    bbox_lo = A.min(axis=0) - 0.5 * (A.max(axis=0) - A.min(axis=0))
    bbox_hi = A.max(axis=0) + 0.5 * (A.max(axis=0) - A.min(axis=0))
    print(f"Attractor centre {c_attr.round(2)}, radius {r_attr:.2f}")
    print(f"Bounding box lo {bbox_lo.round(1)}, hi {bbox_hi.round(1)}")

    # -- Cells --------------------------------------------------------------
    cells_by_K = {}
    for K in args.k_grid:
        alloc = args.alloc_K1 if K == 1 else args.alloc_Kge2
        path = models_dir / f"cell_K{K}_N{args.n_tot:05d}_{alloc}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Stage A cell missing: {path}")
        cells_by_K[K] = load_cell(path)
        print(f"K={K}: loaded {path.name}, n_per_cluster="
              f"{cells_by_K[K]['clusters'][0]['xi'].shape[0] if K == 1 else '...'}")

    # -- FOM reference: long trajectory for marginals + Lyapunov truth ----
    rng_lyap = np.random.default_rng(args.lyap_x0_seed)
    lyap_x0 = A[int(rng_lyap.integers(A.shape[0]))]
    print(f"Lyapunov x0 from snapshot, norm {np.linalg.norm(lyap_x0):.2f}")

    print(f"\nIntegrating FOM reference for {args.T_pdf:g} s ...")
    t0 = time.time()
    fom_traj = _integrate_fom(
        lyap_x0, args.dt,
        int(round(args.T_pdf / args.dt)),
        sample_stride=args.sample_stride,
    )
    print(f"  done in {time.time() - t0:.1f}s; M_samples={fom_traj.shape[0]}")
    fom_marginals = {"traj": fom_traj}
    np.save(out_dir / "fom_reference_traj.npy", fom_traj)

    # -- IC ensemble ------------------------------------------------------
    rng_ic = np.random.default_rng(args.ic_seed)
    ic_array, ic_keys = make_ic_ensemble(
        rng_ic, args.n_dir, args.F_list, c_attr, r_attr,
    )
    print(f"\n{ic_array.shape[0]} off-attractor ICs "
          f"({args.n_dir} dirs x {len(args.F_list)} scales).")

    # -- Tasks -----------------------------------------------------------
    tasks = [("lyapunov", K) for K in args.k_grid]
    for K in args.k_grid:
        for ic_idx in range(ic_array.shape[0]):
            tasks.append(("pdf", K, ic_idx))
    print(f"\n{len(tasks)} tasks ({len(args.k_grid)} Lyapunov + "
          f"{len(args.k_grid) * ic_array.shape[0]} PDF) "
          f"on {args.workers} workers.")

    kwargs = {
        "dt": args.dt,
        "T_lyap": args.T_lyap,
        "tau_renorm": args.tau_renorm,
        "switch_tol": args.switch_tol,
        "T_pdf": args.T_pdf,
        "sample_stride": args.sample_stride,
        "lyap_x0": lyap_x0,
    }
    init_args = (cells_by_K, ic_array, ic_keys, bbox_lo, bbox_hi,
                 fom_marginals, kwargs)
    t_sweep = time.time()
    results = []
    with mp.Pool(processes=args.workers, initializer=_worker_init,
                 initargs=init_args) as pool:
        for r in pool.imap_unordered(_run_task, tasks):
            results.append(r)
            tag = "Lyap" if r["mode"] == "lyapunov" else "PDF "
            if r["mode"] == "lyapunov":
                lam = r["lambda"]
                print(f"  [{time.time() - t_sweep:6.1f}s] {tag} K={r['K']} "
                      f"lambdas=({lam[0]:+.3f}, {lam[1]:+.3f}, {lam[2]:+.3f}) "
                      f"switches={r['switches']} t={r['t_seconds']:.1f}s",
                      flush=True)
            else:
                print(f"  [{time.time() - t_sweep:6.1f}s] {tag} K={r['K']} "
                      f"F={r['F']:>4.1f} dir={r['dir']} "
                      f"t_enter={r['t_enter']:>7.2f} switches={r['switches']:>5d} "
                      f"alive={int(r['alive'])} t={r['t_seconds']:.1f}s",
                      flush=True)
    print(f"\nSweep finished in {time.time() - t_sweep:.1f} s.")

    # -- Pool unconditioned tails per (K, F) for marginal PDFs ------------
    pools = {}
    pool_sizes = {}
    for K in args.k_grid:
        for F in args.F_list:
            tails = [r["tail_uncond"] for r in results
                     if r.get("mode") == "pdf"
                     and r["K"] == K and abs(r["F"] - F) < 1e-9
                     and r["tail_uncond"].shape[0] > 0]
            arr = np.concatenate(tails, axis=0) if tails else np.zeros((0, 3))
            pools[(K, F)] = arr
            pool_sizes[f"K{K}_F{F}"] = int(arr.shape[0])
    np.savez_compressed(
        out_dir / "pools.npz",
        fom=fom_traj,
        **{f"K{K}_F{F}": pools[(K, F)]
           for K in args.k_grid for F in args.F_list},
    )

    # -- Strip array fields before JSON-serialising ----------------------
    json_results = []
    for r in results:
        rj = {k: v for k, v in r.items() if k != "tail_uncond"}
        json_results.append(rj)

    summary = {
        "config": {k: (str(v) if isinstance(v, Path) else
                       (v.tolist() if isinstance(v, np.ndarray) else v))
                   for k, v in vars(args).items()},
        "attractor": {
            "centre": c_attr.tolist(), "radius": r_attr,
            "bbox_lo": bbox_lo.tolist(), "bbox_hi": bbox_hi.tolist(),
        },
        "lyap_x0": lyap_x0.tolist(),
        "lambda_truth": list(LAMBDA_L63),
        "pool_sizes": pool_sizes,
        "results": json_results,
    }
    (out_dir / "step2b.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir / 'step2b.json'}")
    _make_plots(results, args, fom_traj, out_dir)
    _plot_pdf_marginals(pools, fom_traj, args, out_dir)
    return 0


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _make_plots(results: list, args: argparse.Namespace,
                fom_traj: np.ndarray, out_dir: Path) -> None:
    K_grid = list(args.k_grid)
    lyaps = {r["K"]: r["lambda"] for r in results if r["mode"] == "lyapunov"}
    pdf_by_K_F = {}
    for r in results:
        if r["mode"] != "pdf":
            continue
        pdf_by_K_F.setdefault((r["K"], r["F"]), []).append(r)

    # 1) Lyapunov spectrum vs truth
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    width = 0.15
    x = np.arange(3)
    truth = LAMBDA_L63
    ax.bar(x - 2 * width, truth, width, color="grey", label="FOM truth")
    colors = {1: "C0", 2: "C1", 3: "C2", 4: "C3"}
    for i, K in enumerate(K_grid):
        lam = lyaps.get(K, [np.nan, np.nan, np.nan])
        ax.bar(x + (i - 1) * width, lam, width,
               color=colors.get(K, "k"), label=f"K={K}")
    ax.set_xticks(x)
    ax.set_xticklabels([r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$"])
    ax.set_ylabel(r"Lyapunov exponent")
    ax.set_title(f"L63 step 2b: Lyapunov spectrum, N_tot={args.n_tot}")
    ax.axhline(0, color="k", lw=0.5)
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(fontsize=10, ncol=5)
    fig.tight_layout()
    fig.savefig(out_dir / "lyapunov_spectrum.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'lyapunov_spectrum.png'}")

    # 2) Survival fraction + median t_enter per (K, F)
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.5))
    axs[0].set_title("Survival fraction (alive at T)")
    axs[1].set_title("Median time-to-enter attractor box (s)")
    for ax in axs:
        ax.set_xlabel("K")
        ax.grid(True, ls=":", alpha=0.4)
    F_list = sorted({r["F"] for r in results if r["mode"] == "pdf"})
    F_markers = {2.0: "o", 5.0: "s", 10.0: "D"}
    for F in F_list:
        survs = []
        t_enters = []
        for K in K_grid:
            rs = pdf_by_K_F.get((K, F), [])
            survs.append(np.mean([r["alive"] for r in rs]) if rs else 0.0)
            ts = [r["t_enter"] for r in rs if np.isfinite(r["t_enter"])]
            t_enters.append(np.median(ts) if ts else float("nan"))
        axs[0].plot(K_grid, survs, F_markers.get(F, "o") + "-",
                    label=f"F={F:g}", lw=1.6)
        axs[1].plot(K_grid, t_enters, F_markers.get(F, "o") + "-",
                    label=f"F={F:g}", lw=1.6)
    axs[0].set_ylim(-0.05, 1.05)
    axs[0].set_ylabel("fraction")
    axs[1].set_ylabel("t_enter (s)")
    for ax in axs:
        ax.legend(fontsize=10)
    fig.suptitle(f"L63 step 2b: PDF survival, N_tot={args.n_tot}")
    fig.tight_layout()
    fig.savefig(out_dir / "survival_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'survival_summary.png'}")

    # 3) W1 marginals: median over alive trajectories per K, per F, per axis
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    axes_names = ("x", "y", "z")
    for c, ax in enumerate(axs):
        for F in F_list:
            vals = []
            for K in K_grid:
                rs = [r for r in pdf_by_K_F.get((K, F), [])
                      if r["alive"] and np.isfinite(r["w1"][c])]
                vals.append(np.median([r["w1"][c] for r in rs])
                            if rs else float("nan"))
            ax.plot(K_grid, vals, F_markers.get(F, "o") + "-",
                    label=f"F={F:g}", lw=1.6)
        ax.set_title(f"$W_1$ marginal {axes_names[c]}")
        ax.set_xlabel("K")
        ax.set_yscale("log")
        ax.grid(True, ls=":", alpha=0.4, which="both")
        if c == 0:
            ax.set_ylabel(r"$W_1$ (alive medians)")
        ax.legend(fontsize=10)
    fig.suptitle(f"L63 step 2b: W1 to FOM marginals, N_tot={args.n_tot}")
    fig.tight_layout()
    fig.savefig(out_dir / "w1_marginals.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'w1_marginals.png'}")

    # 4) KS distance on z-marginal: bimodality witness
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for F in F_list:
        vals = []
        for K in K_grid:
            rs = [r for r in pdf_by_K_F.get((K, F), [])
                  if r["alive"] and np.isfinite(r["ks_z"])]
            vals.append(np.median([r["ks_z"] for r in rs])
                        if rs else float("nan"))
        ax.plot(K_grid, vals, F_markers.get(F, "o") + "-",
                label=f"F={F:g}", lw=1.6)
    ax.set_title(f"KS distance to FOM z-PDF, N_tot={args.n_tot}")
    ax.set_xlabel("K")
    ax.set_ylabel("KS distance (alive medians)")
    ax.grid(True, ls=":", alpha=0.4, which="both")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "ks_z_marginal.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'ks_z_marginal.png'}")

    # 5) Switch rate per trajectory (ringing detector)
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for F in F_list:
        vals = []
        for K in K_grid:
            rs = pdf_by_K_F.get((K, F), [])
            rates = [r["switches"] / max(args.T_pdf, 1.0)
                     for r in rs if r["alive"]]
            vals.append(np.median(rates) if rates else float("nan"))
        ax.plot(K_grid, vals, F_markers.get(F, "o") + "-",
                label=f"F={F:g}", lw=1.6)
    ax.set_title(f"Cluster switch rate per trajectory, N_tot={args.n_tot}")
    ax.set_xlabel("K")
    ax.set_ylabel("switches per second (alive medians)")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.grid(True, ls=":", alpha=0.4, which="both")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "switch_rate.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'switch_rate.png'}")


def _kde_curve(samples: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if samples.shape[0] < 5:
        return np.full_like(grid, np.nan)
    if np.var(samples) < 1e-12:
        out = np.zeros_like(grid)
        out[int(np.argmin(np.abs(grid - samples.mean())))] = 1.0 / (grid[1] - grid[0])
        return out
    return gaussian_kde(samples)(grid)


def _plot_pdf_marginals(pools: dict, fom_traj: np.ndarray,
                        args: argparse.Namespace, out_dir: Path) -> None:
    """One figure per F: 3 panels (x, y, z), 5 lines (FOM + K=1..4).

    The pool per (K, F) is the union of unconditioned second-half tails
    across all `n_dir` directions, so each PDF reflects the *long-time*
    behaviour of the ROM starting from a sphere of off-attractor ICs.
    """
    K_grid = list(args.k_grid)
    colors = {1: "C0", 2: "C1", 3: "C2", 4: "C3"}

    all_axes = [fom_traj[:, c].copy() for c in range(3)]
    for arr in pools.values():
        if arr.shape[0] == 0:
            continue
        for c in range(3):
            all_axes[c] = np.concatenate([all_axes[c], arr[:, c]])
    lo = np.array([np.percentile(v, 0.5) for v in all_axes])
    hi = np.array([np.percentile(v, 99.5) for v in all_axes])
    grids = [np.linspace(lo[c], hi[c], 400) for c in range(3)]
    axes_names = ("x", "y", "z")

    for F in args.F_list:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
        for c, ax in enumerate(axs):
            ax.plot(grids[c], _kde_curve(fom_traj[:, c], grids[c]),
                    color="black", lw=2.2, label="FOM")
            for K in K_grid:
                arr = pools.get((K, F), np.zeros((0, 3)))
                if arr.shape[0] < 5:
                    continue
                ax.plot(grids[c], _kde_curve(arr[:, c], grids[c]),
                        color=colors[K], lw=1.5,
                        label=f"K={K} (n={arr.shape[0]})")
            ax.set_xlabel(axes_names[c])
            ax.set_ylabel("density" if c == 0 else "")
            ax.set_title(f"$p({axes_names[c]})$")
            ax.grid(True, ls=":", alpha=0.4)
            if c == 0:
                ax.legend(fontsize=9, loc="upper right")
        fig.suptitle(
            f"L63 step 2b: marginal PDFs, F={F:g} "
            f"(IC at $c_a + F\\,r_a\\,\\hat n$, T={args.T_pdf:g}s tail)"
        )
        fig.tight_layout()
        path = out_dir / f"pdfs_F{F:g}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
