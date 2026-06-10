"""Shared fit + evaluator helpers for the L63 RBF-only follow-up scripts.

This module is `results/LOR63/`-local on purpose; it duplicates the
counterfactual `_stlsq_rbf_only` and `f_rbf` paths from
`scripts/run_phase0_l63_rbf_only.py` so the integration / shell scripts
can be self-contained without lifting research-only utilities into the
chord2/ library. Import via sys.path injection from the consuming
script.
"""

from __future__ import annotations

import time

import numpy as np

from chord2 import sindy


SIGMA_L63 = 10.0
RHO_L63 = 28.0
BETA_L63 = 8.0 / 3.0


def f_truth_l63(A: np.ndarray) -> np.ndarray:
    """Analytic L63 RHS, vectorised over rows of A. (sigma, rho, beta) = (10, 28, 8/3)."""
    x, y, z = A[:, 0], A[:, 1], A[:, 2]
    out = np.empty_like(A)
    out[:, 0] = SIGMA_L63 * (y - x)
    out[:, 1] = x * (RHO_L63 - z) - y
    out[:, 2] = x * y - BETA_L63 * z
    return out


def f_truth_l63_single(a: np.ndarray) -> np.ndarray:
    """Single-state L63 RHS; faster than the vectorised path for RK4 inner loop."""
    x, y, z = a[0], a[1], a[2]
    return np.array([
        SIGMA_L63 * (y - x),
        x * (RHO_L63 - z) - y,
        x * y - BETA_L63 * z,
    ])


def f_poly(A: np.ndarray, xi_poly: np.ndarray) -> np.ndarray:
    """Quadratic polynomial vector field at the rows of A."""
    return sindy.poly_features(A, 2) @ xi_poly


def f_rbf(A: np.ndarray, centers: np.ndarray, widths: np.ndarray,
          mu_A: np.ndarray, sigma_A: np.ndarray,
          col_norms: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """Column-normalised iso-RBF vector field at the rows of A."""
    Phi = sindy.rbf_features_iso(A, centers, widths, mu_A=mu_A, sigma_A=sigma_A)
    return (Phi / col_norms) @ xi


def fit_rbf_only_keep_state(A_mid: np.ndarray, dAdt: np.ndarray, *,
                            n_rbf: int, seed: int, lambda_rbf: float,
                            gamma: float, n_knn: int, n_init: int,
                            max_iter: int = 20,
                            bandwidth_mode: str = "knn_per_center",
                            lambda_tikh: float = 0.0,
                            log_diagnostics: bool = False,
                            mu_A_override: np.ndarray | None = None,
                            sigma_A_override: np.ndarray | None = None):
    """RBF-only STLSQ fit; returns the state needed to evaluate f_rbf later.

    Mirrors `_stlsq_rbf_only` in scripts/run_phase0_l63_rbf_only.py.

    `bandwidth_mode` is forwarded to `sindy.rbf_centers_flat_isotropic`.
    `lambda_tikh > 0` adds a Tikhonov term `lambda_tikh * I` to the normal
    equations of each STLSQ inner solve - cheap insurance against
    ill-conditioned wide-Gaussian designs.

    `log_diagnostics=True` switches the STLSQ inner solve from
    `np.linalg.lstsq` (on the row-augmented Tikhonov system) to an explicit
    thin SVD of `Phi_active`, with Tikhonov applied in singular-value space
    (`s/(s^2 + lambda_tikh)`). Mathematically equivalent for our settings;
    the difference is that we record per-iteration condition number
    `kappa = s_max/s_min` of the *original* (un-augmented) design and the
    wall-time spent in the SVD. Returned in `"diagnostics"` as a list of
    one dict per STLSQ iteration.
    """
    centers, widths, meta = sindy.rbf_centers_flat_isotropic(
        A_mid, n_rbf, seed=seed, gamma=gamma, n_knn=n_knn, n_init=n_init,
        bandwidth_mode=bandwidth_mode,
        mu_A_override=mu_A_override, sigma_A_override=sigma_A_override,
    )
    Phi_raw = sindy.rbf_features_iso(
        A_mid, centers, widths,
        mu_A=meta["mu_A"], sigma_A=meta["sigma_A"],
    )
    col_norms = np.linalg.norm(Phi_raw, axis=0)
    col_norms = np.where(col_norms == 0.0, 1.0, col_norms)
    Phi_n = Phi_raw / col_norms

    n_f = Phi_n.shape[1]
    active = np.ones(n_f, dtype=bool)
    prev_active = None
    xi = np.zeros((n_f, dAdt.shape[1]))
    diagnostics = []
    for it in range(max_iter):
        idx = np.where(active)[0]
        if idx.size == 0:
            break
        t_iter_start = time.time()
        if log_diagnostics:
            t_svd_start = time.time()
            U, s, Vt = np.linalg.svd(Phi_n[:, idx], full_matrices=False)
            t_svd = time.time() - t_svd_start
            kappa = (float(s[0] / s[-1])
                     if s.size > 0 and s[-1] > 0.0 else float("inf"))
            if lambda_tikh > 0.0:
                filt = s / (s * s + lambda_tikh)
            else:
                # Mimic lstsq's default rcond cut: drop singulars below
                # eps * s_max * max(M, N).
                M, N = Phi_n.shape[0], idx.size
                cutoff = np.finfo(s.dtype).eps * (s[0] if s.size else 1.0) * max(M, N)
                keep = s > cutoff
                inv_s = np.where(keep, 1.0 / np.where(keep, s, 1.0), 0.0)
                filt = inv_s
            xi_act = Vt.T @ (filt[:, None] * (U.T @ dAdt))
        else:
            t_svd = float("nan")
            kappa = float("nan")
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
        if log_diagnostics:
            dAdt_pred = Phi_n[:, idx] @ xi_act
            res_rel = float(np.linalg.norm(dAdt - dAdt_pred)
                            / max(np.linalg.norm(dAdt), 1e-30))
            diagnostics.append({
                "iter": int(it),
                "n_active": int(idx.size),
                "kappa": kappa,
                "t_svd_seconds": float(t_svd),
                "residual_rel": res_rel,
                "t_iter_seconds": float(time.time() - t_iter_start),
            })
        new_active = active.copy()
        mags = np.max(np.abs(xi), axis=1)
        new_active[mags < lambda_rbf] = False
        if prev_active is not None and np.array_equal(new_active, active):
            break
        prev_active = active.copy()
        active = new_active

    return {
        "centers": centers, "widths": widths,
        "mu_A": meta["mu_A"], "sigma_A": meta["sigma_A"],
        "col_norms": col_norms, "xi": xi,
        "nnz": int(active.sum()),
        "meta": meta,
        "diagnostics": diagnostics,
    }


def rk4_integrate(f_callable, x0: np.ndarray, dt: float, n_steps: int):
    """Fixed-step RK4 with non-finite early exit.

    f_callable: a -> dadt where a has shape (r,). Caller is responsible
    for closing over fit state. Returns (t, X) where X has shape (n_done+1, r)
    and t has shape (n_done+1,); n_done <= n_steps. On non-finite RHS or
    state, integration stops and the (already-finite) prefix is returned.
    """
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
