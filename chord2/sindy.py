"""Sparse identification of nonlinear dynamics for the CHORD2 Phase 0 gate.

CHORD2 replaces the paper's intrusive Galerkin projection (Colanera & Magri
2025, Eq. 13) with a non-intrusive sparse regression on the resolved
coordinates:

    da/dt = Theta(a) . xi

Phase 0 of the research plan (`docs/research_plan_quantized_sindy_rbf.pdf`,
spec at `docs/notes/phase-0-design.md`) is the *existence gate* that decides
whether an RBF closure block is motivated at all. It runs on two testbeds:

- Lorenz-63 - negative control. The quadratic library spans the RHS
  exactly, so the RBF coefficients must vanish under the joint fit.
- Two-scale Lorenz-96 with X-only fit - existence gate. The closure
  `-(hc/b) Σ_j Y_{j,k}` is the only structurally non-polynomial term.

The two testbeds use different RBF dictionaries (locked 2026-06-07 in
`docs/journal/2026-06-07-phase-0-rbf-design.md`):

- L63: flat K-means + 5-NN-median isotropic widths. Deliberately weak,
  so the negative-control test is sharp.
- L96 X-only: K-means with `K_shape` clusters, trace-relative covariance
  ridge, 99th-percentile Mahalanobis outlier clip, FPS within each cluster
  under the cluster's Mahalanobis metric, kernel using the parent
  cluster's covariance. A mandatory conditioning gate
  (`cond(Phi_rbf) < 1e6`, greedy drop) follows orthogonalisation.

A third, general path (`rbf_centers_hier_anisotropic_pca`) targets the
multi-regime SINDy pipeline (KS-bursting, RB): K-means with `K_shape`
clusters, per-cluster PCA frame `V_k`, tangent widths from
`sqrt(lambda_k^(p))`, normal widths from a robust statistic of pairwise
centre-projection magnitudes onto the cluster's normal eigenvectors.
Closes the off-cloud dead zone (`docs/notes/l63-rbf-bandwidth-rules.md`)
without coupling kernel width to `n_rbf` density. Design entry:
`docs/journal/2026-06-10-anisotropic-rbf-per-cluster-design.md`.

Both testbeds enforce the Schlegel-Noack energy-preservation constraint on
the quadratic tensor (Schlegel & Noack 2015, *J. Fluid Mech.* **765**,
325-352, Eq. 2.4 with symmetrisation Eq. 2.3; verified 2026-06-07):

    Q_ijk + Q_jik + Q_kij = 0,   q_ijk = q_ikj.

The constraint is enforced inside a KKT inner solve at every STLSQ
iteration (per-block thresholds `lambda_poly`, `lambda_rbf`); zero rows
of the constraint matrix produced by column pruning are stripped before
the KKT block is built.

Derivatives of the resolved coordinate are estimated with a 5-point
stencil (O(dt^4)) rather than the central FD of the prior SINDy design
state - the central-FD error on the high-acceleration wings of L63's
butterfly is large enough to leak into `xi_rbf` magnitudes and confuse
the negative-control verdict.

Implemented from scratch in numpy; no pysindy, no sklearn, no torch.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Polynomial library
# ---------------------------------------------------------------------------

def poly_feature_layout(r: int, p: int = 2) -> dict:
    """Layout of the polynomial feature library at total degree `p`.

    Returns a dict with stable keys describing the column layout:
        names        : list[str] of column names, length N_poly
        bias_idx     : int, column of the constant feature (always 0)
        linear_idx   : list[int] of length r, columns of a_0, ..., a_{r-1}
        quad_pairs   : list[(j, k)] with j <= k, length N_quad
        quad_idx     : list[int] of length N_quad, columns of the quadratic
                       features in the same order as `quad_pairs`
        N_poly       : int, 1 + r + N_quad

    Phase 0 uses `p = 2`; higher `p` is not currently emitted. The (j <= k)
    convention matches the Schlegel-Noack symmetry q_ijk = q_ikj.
    """
    if p != 2:
        raise NotImplementedError("Phase 0 uses p=2 only; higher p not emitted.")
    names = ["1"] + [f"a_{i}" for i in range(r)]
    bias_idx = 0
    linear_idx = list(range(1, 1 + r))
    quad_pairs = [(j, k) for j in range(r) for k in range(j, r)]
    quad_idx = list(range(1 + r, 1 + r + len(quad_pairs)))
    for (j, k) in quad_pairs:
        names.append(f"a_{j}*a_{k}")
    return {
        "names": names,
        "bias_idx": bias_idx,
        "linear_idx": linear_idx,
        "quad_pairs": quad_pairs,
        "quad_idx": quad_idx,
        "N_poly": 1 + r + len(quad_pairs),
    }


def poly_features(A: np.ndarray, p: int = 2) -> np.ndarray:
    """Evaluate the polynomial feature library at snapshots `A`.

    Parameters
    ----------
    A : (M, r) array of snapshots in the resolved coordinate.
    p : int, total polynomial degree. Phase 0 uses p = 2.

    Returns
    -------
    Theta : (M, N_poly) array with column layout as in `poly_feature_layout`.
    """
    M, r = A.shape
    layout = poly_feature_layout(r, p)
    N_poly = layout["N_poly"]
    Theta = np.empty((M, N_poly), dtype=A.dtype)
    Theta[:, 0] = 1.0
    Theta[:, 1:1 + r] = A
    col = 1 + r
    for (j, k) in layout["quad_pairs"]:
        Theta[:, col] = A[:, j] * A[:, k]
        col += 1
    return Theta


def poly_apply(a: np.ndarray, xi_poly: np.ndarray, r: int, p: int = 2) -> np.ndarray:
    """Apply the polynomial part of a SINDy model at a single point.

    `xi_poly` has shape (N_poly, r); returns shape (r,).
    """
    layout = poly_feature_layout(r, p)
    theta = np.empty(layout["N_poly"], dtype=a.dtype)
    theta[0] = 1.0
    theta[1:1 + r] = a
    col = 1 + r
    for (j, k) in layout["quad_pairs"]:
        theta[col] = a[j] * a[k]
        col += 1
    return theta @ xi_poly


# ---------------------------------------------------------------------------
# RBF centre placement - L63 path: flat isotropic
# ---------------------------------------------------------------------------

def rbf_centers_flat_isotropic(
    A: np.ndarray,
    n_rbf: int,
    *,
    n_knn: int = 5,
    gamma: float = 1.0,
    seed: int = 0,
    n_init: int = 10,
    bandwidth_mode: str = "knn_per_center",
    mu_A_override: np.ndarray | None = None,
    sigma_A_override: np.ndarray | None = None,
):
    """Flat K-means centres with isotropic widths.

    Used on L63 (negative control). The dictionary is deliberately weak so
    that the negative-control test is sharp; see
    `docs/journal/2026-06-07-phase-0-rbf-design.md` for the spine-alignment
    argument that motivates this choice.

    Steps:
    1. Standardise A column-wise (zero mean, unit variance).
    2. K-means with `n_rbf` clusters, `n_init` restarts, k-means++ init.
    3. Assign per-centre isotropic width sigma_k by `bandwidth_mode`.

    Bandwidth modes
    ---------------
    "knn_per_center" (default): sigma_k = gamma * median(||mu_k - mu_j||)
        over the `n_knn` nearest centres, computed in standardised
        coordinates. Adaptive; widths shrink as n_rbf grows (~ N^(-1/d)),
        which kills off-cloud Gaussian support.
    "attr_scaled":   sigma_k = gamma for all k, in standardised coordinates.
        Because we standardised by per-axis std, sigma_attr_std = 1, so
        gamma is in units of the attractor scale. Decouples width from
        centre density; pick gamma to control off-cloud Gaussian reach.

    Returns
    -------
    centers : (n_rbf, r) array, in *original* (un-standardised) coordinates.
    widths  : (n_rbf,) array of sigma_k, in *standardised* coordinates -
              feature evaluation must standardise A consistently.
    meta    : dict carrying `mu_A, sigma_A` for the standardisation, plus
              the K-means inertia, n_iter, seed, gamma, n_knn, bandwidth_mode.
    """
    from chord2.clustering import kmeans_fit

    M, r = A.shape
    mu_A = mu_A_override if mu_A_override is not None else A.mean(axis=0)
    sigma_A = sigma_A_override if sigma_A_override is not None else A.std(axis=0)
    sigma_A = np.where(sigma_A == 0.0, 1.0, sigma_A)
    A_std = (A - mu_A) / sigma_A

    labels, centroids_std, inertia, n_iter = kmeans_fit(
        A_std.astype(np.float32), n_rbf, seed=seed, n_init=n_init
    )
    centroids_std = np.asarray(centroids_std, dtype=np.float64)

    if bandwidth_mode == "knn_per_center":
        diff = centroids_std[:, None, :] - centroids_std[None, :, :]
        d = np.sqrt((diff * diff).sum(-1))
        np.fill_diagonal(d, np.inf)
        k = min(n_knn, n_rbf - 1)
        nearest = np.partition(d, k - 1, axis=1)[:, :k]
        widths = gamma * np.median(nearest, axis=1)
    elif bandwidth_mode == "attr_scaled":
        widths = np.full(n_rbf, float(gamma), dtype=np.float64)
    else:
        raise ValueError(
            f"bandwidth_mode must be 'knn_per_center' or 'attr_scaled', "
            f"got {bandwidth_mode!r}"
        )

    # Un-standardise centres for storage.
    centers = centroids_std * sigma_A + mu_A

    meta = {
        "mu_A": mu_A,
        "sigma_A": sigma_A,
        "inertia": float(inertia),
        "n_iter": int(n_iter),
        "seed": int(seed),
        "gamma": float(gamma),
        "n_knn": int(n_knn),
        "bandwidth_mode": bandwidth_mode,
        "kind": "flat_isotropic",
    }
    return centers, widths, meta


def rbf_features_iso(
    A: np.ndarray,
    centers: np.ndarray,
    widths: np.ndarray,
    *,
    mu_A: np.ndarray,
    sigma_A: np.ndarray,
) -> np.ndarray:
    """Evaluate isotropic RBF features `exp(-||a_std - mu_std||^2 / (2 sigma^2))`.

    All distances are computed in standardised coordinates; `centers` is in
    original coordinates and is standardised internally.
    """
    A_std = (A - mu_A) / sigma_A
    centers_std = (centers - mu_A) / sigma_A
    diff = A_std[:, None, :] - centers_std[None, :, :]
    d2 = (diff * diff).sum(-1)
    return np.exp(-0.5 * d2 / (widths ** 2)[None, :])


# ---------------------------------------------------------------------------
# RBF centre placement - L96 path: hierarchical Mahalanobis FPS
# ---------------------------------------------------------------------------

def _ridge_covariance(Sigma: np.ndarray, eps: float) -> np.ndarray:
    """Trace-relative ridge: Sigma <- Sigma + eps * (tr(Sigma) / r) * I."""
    r = Sigma.shape[0]
    return Sigma + eps * (np.trace(Sigma) / r) * np.eye(r)


def _mahal_sq(X: np.ndarray, mu: np.ndarray, Sigma_inv: np.ndarray) -> np.ndarray:
    """Squared Mahalanobis distance from each row of X to mu."""
    d = X - mu[None, :]
    return np.einsum("mi,ij,mj->m", d, Sigma_inv, d)


def _fps_mahalanobis(X: np.ndarray, n_pick: int, Sigma_inv: np.ndarray,
                     seed_idx: int) -> np.ndarray:
    """Farthest-point sampling under Mahalanobis metric.

    Greedy: at each step pick the point with the largest min-Mahalanobis
    distance to the already-selected set. The first index is `seed_idx`.
    Returns the picked indices, in pick order.
    """
    M = X.shape[0]
    n_pick = min(n_pick, M)
    picked = [int(seed_idx)]
    d = _mahal_sq(X, X[seed_idx], Sigma_inv)
    for _ in range(1, n_pick):
        nxt = int(np.argmax(d))
        picked.append(nxt)
        d_new = _mahal_sq(X, X[nxt], Sigma_inv)
        d = np.minimum(d, d_new)
    return np.array(picked, dtype=np.int64)


def rbf_centers_hier_mahalanobis(
    A: np.ndarray,
    K_shape: int,
    n_k: int,
    *,
    eps_ridge: float = 1e-3,
    clip_pct: float = 99.0,
    seed: int = 0,
    n_init: int = 10,
):
    """Hierarchical Mahalanobis-FPS centres for the L96 existence gate.

    Pipeline (per `docs/notes/phase-0-design.md` §"RBF block - L96 X-only"):

    1. K-means with `K_shape` clusters on standardised `A`.
    2. Per-cluster sample covariance `Sigma_k`, trace-relative ridge with
       `eps_ridge`.
    3. Per-cluster outlier clip at the `clip_pct` percentile of Mahalanobis
       distance from the cluster centroid.
    4. FPS with `n_k` picks within each (clipped) cluster, under the
       cluster's Mahalanobis metric. Seed at the point of maximum
       Mahalanobis distance from the centroid (the centroid itself is
       already covered by the bias column of Theta_poly).
    5. Kernel for each centre uses the parent cluster's `Sigma_k`.

    Returns
    -------
    centers       : (n_total, r) array in *standardised* coordinates,
                    where n_total <= K_shape * n_k (clusters with < n_k
                    samples after clipping contribute fewer centres).
    Sigma_invs    : (n_total, r, r) per-centre inverse covariance, in
                    standardised coordinates. Each block is a copy of the
                    parent cluster's Sigma_k^-1.
    parent_k      : (n_total,) int array of parent cluster indices.
    meta          : dict with mu_A, sigma_A, K_shape, n_k, eps_ridge,
                    clip_pct, seed.
    """
    from chord2.clustering import kmeans_fit

    M, r = A.shape
    mu_A = A.mean(axis=0)
    sigma_A = A.std(axis=0)
    sigma_A = np.where(sigma_A == 0.0, 1.0, sigma_A)
    A_std = (A - mu_A) / sigma_A

    labels, centroids_std, _inertia, _n_iter = kmeans_fit(
        A_std.astype(np.float32), K_shape, seed=seed, n_init=n_init
    )
    labels = np.asarray(labels)
    centroids_std = np.asarray(centroids_std, dtype=np.float64)

    centers_list, Sigma_inv_list, parent_list = [], [], []
    for k in range(K_shape):
        mask = labels == k
        n_in_k = int(mask.sum())
        if n_in_k < r + 2:
            # Degenerate cluster - sample covariance ill-defined. Skip.
            continue
        X_k = A_std[mask]
        mu_k = centroids_std[k]
        Sigma_k = np.cov(X_k.T, bias=False)
        Sigma_k = _ridge_covariance(Sigma_k, eps_ridge)
        Sigma_k_inv = np.linalg.inv(Sigma_k)

        # Clip outliers.
        d_to_mu = _mahal_sq(X_k, mu_k, Sigma_k_inv)
        thresh = np.percentile(d_to_mu, clip_pct)
        keep = d_to_mu <= thresh
        X_kc = X_k[keep]
        d_to_mu_c = d_to_mu[keep]
        if X_kc.shape[0] < 2:
            continue

        # Seed FPS at the point of max Mahalanobis distance from the centroid.
        seed_idx = int(np.argmax(d_to_mu_c))
        n_pick = min(n_k, X_kc.shape[0])
        picked = _fps_mahalanobis(X_kc, n_pick, Sigma_k_inv, seed_idx)
        centers_list.append(X_kc[picked])
        # One inverse-covariance per centre (all the same within a cluster);
        # storing per-centre keeps feature evaluation uniform.
        Sigma_inv_list.append(np.broadcast_to(
            Sigma_k_inv, (n_pick, r, r)).copy())
        parent_list.append(np.full(n_pick, k, dtype=np.int64))

    centers = np.concatenate(centers_list, axis=0)
    Sigma_invs = np.concatenate(Sigma_inv_list, axis=0)
    parent_k = np.concatenate(parent_list)

    meta = {
        "mu_A": mu_A,
        "sigma_A": sigma_A,
        "K_shape": int(K_shape),
        "n_k": int(n_k),
        "eps_ridge": float(eps_ridge),
        "clip_pct": float(clip_pct),
        "seed": int(seed),
        "kind": "hier_mahalanobis",
    }
    return centers, Sigma_invs, parent_k, meta


def rbf_features_mahal(
    A: np.ndarray,
    centers_std: np.ndarray,
    Sigma_invs_std: np.ndarray,
    *,
    mu_A: np.ndarray,
    sigma_A: np.ndarray,
) -> np.ndarray:
    """Anisotropic RBF features `exp(-1/2 (a_std - mu_std)^T Sigma_inv (a_std - mu_std))`."""
    A_std = (A - mu_A) / sigma_A
    M = A_std.shape[0]
    n_c = centers_std.shape[0]
    out = np.empty((M, n_c), dtype=A.dtype)
    for j in range(n_c):
        d2 = _mahal_sq(A_std, centers_std[j], Sigma_invs_std[j])
        out[:, j] = np.exp(-0.5 * d2)
    return out


# ---------------------------------------------------------------------------
# RBF centre placement - general path: hierarchical anisotropic PCA
# ---------------------------------------------------------------------------

def _select_tangent_rank(
    eigvals: np.ndarray,
    rule: str,
    energy_threshold: float,
    r_fixed: int | None,
) -> int:
    """Number of tangent PCs for one cluster.

    `eigvals` must be in descending order. `rule="energy"` picks the smallest
    `r_t` for which the cumulative variance fraction reaches
    `energy_threshold`; `rule="fixed"` returns `r_fixed`, clamped to [1, r].
    Echoes Colanera & Magri Eqs. 11-13 truncation of the local-PCA basis.
    """
    r = int(eigvals.shape[0])
    if rule == "energy":
        total = float(eigvals.sum())
        # Degenerate: all-zero eigenvalues -> fall back to one tangent PC.
        if total <= 0.0:
            return 1
        cum = np.cumsum(eigvals) / total
        r_t = int(np.searchsorted(cum, energy_threshold) + 1)
        return max(1, min(r_t, r))
    if rule == "fixed":
        if r_fixed is None:
            raise ValueError("tangent_rule='fixed' requires r_tangent.")
        return max(1, min(int(r_fixed), r))
    raise ValueError(
        f"tangent_rule must be 'energy' or 'fixed', got {rule!r}"
    )


def _normal_centre_spacing(
    centers_std: np.ndarray,
    V_k: np.ndarray,
    r_tangent_k: int,
    robust_stat: str,
) -> np.ndarray:
    """Robust pairwise centre-spacing along the normal eigenvectors.

    For each normal PC `p` in `(r_tangent_k, r]`, project every pairwise
    displacement `c_i - c_j` onto `V_k[:, p]`, take the absolute value, and
    return a robust statistic. Per the design entry
    `docs/journal/2026-06-10-anisotropic-rbf-per-cluster-design.md`,
    construction (ii): density-decoupled in the n_centres -> infty limit.

    Returns array of length `r - r_tangent_k`.
    """
    n_k = centers_std.shape[0]
    if n_k < 2:
        raise ValueError(
            f"need >= 2 centres per cluster to estimate normal spacing, got {n_k}"
        )
    diff = centers_std[:, None, :] - centers_std[None, :, :]
    iu, ju = np.triu_indices(n_k, k=1)
    disp = diff[iu, ju]                       # (P, r), P = n_k(n_k-1)/2
    V_normal = V_k[:, r_tangent_k:]           # (r, n_normal)
    projections = np.abs(disp @ V_normal)     # (P, n_normal)
    if robust_stat == "median":
        return np.median(projections, axis=0)
    if robust_stat == "p75":
        return np.percentile(projections, 75.0, axis=0)
    raise ValueError(
        f"robust_stat must be 'median' or 'p75', got {robust_stat!r}"
    )


def rbf_centers_hier_anisotropic_pca(
    A: np.ndarray,
    K_shape: int,
    n_per_cluster: int,
    *,
    tangent_rule: str = "energy",
    energy_threshold: float = 0.99,
    r_tangent: int | None = None,
    alpha_tangent: float = 1.0,
    robust_stat: str = "median",
    eigenvalue_floor: float = 1e-12,
    seed: int = 0,
    n_init: int = 10,
):
    """Hierarchical per-cluster anisotropic-PCA RBF dictionary.

    General-framework path for the multi-regime SINDy pipeline. Design entry:
    `docs/journal/2026-06-10-anisotropic-rbf-per-cluster-design.md`,
    construction (ii). rom-specialist consult 2026-06-10 selected the
    eigenvalue rule on the tangent (with explicit `alpha_tangent`) over the
    symmetric centre-spacing rule, because the two estimators answer
    different questions: intrinsic cloud spread on the tangent vs. coverage
    gap on the normal.

    Pipeline (per cluster k):

      1. Outer K-means with `K_shape` clusters on standardised A.
      2. Eigendecompose the cluster sample covariance -> `V_k` (columns
         ordered by descending eigenvalue), `lambda_k`.
      3. Tangent rank `r_tangent_k` from `tangent_rule`:
            "energy": smallest r such that cum(lambda)/sum(lambda) >=
                      `energy_threshold` (Colanera & Magri Eqs. 11-13).
            "fixed":  use `r_tangent`.
      4. Place `n_per_cluster` centres inside the cluster via FPS-Euclidean
         on standardised coords (seed: farthest snapshot from centroid).
         Placement is decoupled from the precision matrix so the near-rank-
         deficient cluster covariance does not contaminate FPS.
      5. Tangent widths: `s_k^(p) = alpha_tangent * sqrt(max(lambda_k^(p),
         eigenvalue_floor))` for p in [1, r_tangent_k]. `alpha_tangent` is
         the single explicit knob the consult recommended exposing.
      6. Normal widths: robust statistic (`median` or `p75`) of
         `|(c_i - c_j)^T V_k^(p)|` over centre pairs, for each PC
         p in (r_tangent_k, r]. Closes the off-cloud dead zone
         (`docs/notes/l63-rbf-bandwidth-rules.md`) without coupling to
         centre density.
      7. `Sigma_k^{-1} = V_k diag(1 / s_k^2) V_k^T`. Broadcast to one
         precision per centre.

    Clusters with fewer than `max(r + 2, n_per_cluster)` samples are skipped
    (PCA frame ill-defined, or no FPS budget) and contribute no centres.

    Parameters
    ----------
    A : (M, r) snapshots in the resolved coordinate.
    K_shape : int. Outer cluster count.
    n_per_cluster : int >= 2. Constant centre budget per cluster.
    tangent_rule : "energy" or "fixed".
    energy_threshold : cumulative-variance threshold for "energy" rule.
    r_tangent : int, required for "fixed".
    alpha_tangent : scalar multiplier on `sqrt(lambda_k^(p))`. Default 1.0.
    robust_stat : "median" or "p75".
    eigenvalue_floor : numerical floor; guards `sqrt(.)` and any normal
        width that collapses to ~0 on a degenerate cluster.
    seed, n_init : K-means controls.

    Returns
    -------
    centers_std    : (n_total, r) centres in *standardised* coords (matches
                     `rbf_features_mahal` evaluator convention).
    Sigma_invs_std : (n_total, r, r) per-centre inverse covariance in
                     standardised coords. Each block is the parent cluster's
                     Sigma_k^{-1}.
    parent_k       : (n_total,) parent cluster index per centre.
    meta           : dict with mu_A, sigma_A, K-means outputs, hyperparams,
                     and per-cluster diagnostics (V_k, lambda_k,
                     r_tangent_k, widths_k, centroid_k, labels). Per-cluster
                     lists have length K_shape; skipped clusters carry None.
    """
    from chord2.clustering import kmeans_fit

    if n_per_cluster < 2:
        raise ValueError(
            f"n_per_cluster must be >= 2 (need >= 2 centres to estimate "
            f"normal centre-spacing), got {n_per_cluster}"
        )

    M, r = A.shape
    mu_A = A.mean(axis=0)
    sigma_A = A.std(axis=0)
    sigma_A = np.where(sigma_A == 0.0, 1.0, sigma_A)
    A_std = (A - mu_A) / sigma_A

    labels, centroids_std, inertia, n_iter = kmeans_fit(
        A_std.astype(np.float32), K_shape, seed=seed, n_init=n_init
    )
    labels = np.asarray(labels)
    centroids_std = np.asarray(centroids_std, dtype=np.float64)

    eye_r = np.eye(r)
    centers_list, Sigma_inv_list, parent_list = [], [], []
    V_per_k, lambda_per_k, r_tangent_per_k, widths_per_k, centroid_per_k = (
        [], [], [], [], []
    )

    for k in range(K_shape):
        mask = labels == k
        n_in_k = int(mask.sum())
        if n_in_k < max(r + 2, n_per_cluster):
            V_per_k.append(None)
            lambda_per_k.append(None)
            r_tangent_per_k.append(None)
            widths_per_k.append(None)
            centroid_per_k.append(centroids_std[k])
            continue
        X_k = A_std[mask]
        mu_k = centroids_std[k]

        Sigma_k = np.cov(X_k.T, bias=False)
        eigvals, eigvecs = np.linalg.eigh(Sigma_k)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        V_k = eigvecs[:, order]

        r_tangent_k = _select_tangent_rank(
            eigvals, tangent_rule, energy_threshold, r_tangent
        )

        d0 = _mahal_sq(X_k, mu_k, eye_r)
        seed_idx = int(np.argmax(d0))
        picked = _fps_mahalanobis(X_k, n_per_cluster, eye_r, seed_idx)
        C_k = X_k[picked]

        s2 = np.empty(r, dtype=np.float64)
        floored = np.maximum(eigvals, eigenvalue_floor)
        s2[:r_tangent_k] = (alpha_tangent ** 2) * floored[:r_tangent_k]
        if r_tangent_k < r:
            s_normal = _normal_centre_spacing(
                C_k, V_k, r_tangent_k, robust_stat
            )
            s_normal = np.maximum(s_normal, np.sqrt(eigenvalue_floor))
            s2[r_tangent_k:] = s_normal ** 2

        Sigma_k_inv = (V_k * (1.0 / s2)[None, :]) @ V_k.T

        centers_list.append(C_k)
        Sigma_inv_list.append(
            np.broadcast_to(Sigma_k_inv, (n_per_cluster, r, r)).copy()
        )
        parent_list.append(np.full(n_per_cluster, k, dtype=np.int64))

        V_per_k.append(V_k)
        lambda_per_k.append(eigvals)
        r_tangent_per_k.append(int(r_tangent_k))
        widths_per_k.append(np.sqrt(s2))
        centroid_per_k.append(mu_k)

    if not centers_list:
        raise RuntimeError(
            "no cluster yielded valid centres; check K_shape, n_per_cluster, "
            "and snapshot count"
        )

    centers_std = np.concatenate(centers_list, axis=0)
    Sigma_invs_std = np.concatenate(Sigma_inv_list, axis=0)
    parent_k = np.concatenate(parent_list)

    meta = {
        "mu_A": mu_A,
        "sigma_A": sigma_A,
        "K_shape": int(K_shape),
        "n_per_cluster": int(n_per_cluster),
        "tangent_rule": tangent_rule,
        "energy_threshold": float(energy_threshold),
        "r_tangent_kwarg": (None if r_tangent is None else int(r_tangent)),
        "alpha_tangent": float(alpha_tangent),
        "robust_stat": robust_stat,
        "eigenvalue_floor": float(eigenvalue_floor),
        "seed": int(seed),
        "kmeans_inertia": float(inertia),
        "kmeans_n_iter": int(n_iter),
        "kind": "hier_anisotropic_pca",
        "V_per_k": V_per_k,
        "lambda_per_k": lambda_per_k,
        "r_tangent_per_k": r_tangent_per_k,
        "widths_per_k": widths_per_k,
        "centroid_per_k": centroid_per_k,
        "labels": labels,
    }
    return centers_std, Sigma_invs_std, parent_k, meta


def greedy_drop_for_cond(Phi: np.ndarray, cond_max: float = 1e6,
                         max_drops: int | None = None) -> np.ndarray:
    """Greedy-drop columns of Phi until cond(Phi) <= cond_max.

    Returns the indices (into the original columns) that are kept, in order.
    See `docs/notes/phase-0-design.md` §"Conditioning gate (mandatory)".
    The L96 RBF block is indefensible without this gate: an ill-conditioned
    Phi_rbf would make ||xi_rbf|| magnitudes meaningless.
    """
    M, n = Phi.shape
    kept = list(range(n))
    if max_drops is None:
        max_drops = max(0, n - 1)
    dropped = 0
    while dropped < max_drops:
        c = np.linalg.cond(Phi[:, kept])
        if c <= cond_max:
            return np.array(kept, dtype=np.int64)
        # Try removing each remaining column; drop the one minimising cond.
        best_j, best_c = None, float("inf")
        for j_local, _ in enumerate(kept):
            trial = [kept[i] for i in range(len(kept)) if i != j_local]
            if len(trial) < 1:
                continue
            ct = np.linalg.cond(Phi[:, trial])
            if ct < best_c:
                best_c, best_j = ct, j_local
        if best_j is None:
            break
        kept.pop(best_j)
        dropped += 1
    return np.array(kept, dtype=np.int64)


# ---------------------------------------------------------------------------
# Moment-orthogonalisation
# ---------------------------------------------------------------------------

def moment_orthogonalise(Theta_poly: np.ndarray, Phi_rbf: np.ndarray,
                         layout: dict, *, project: bool = True):
    """QR-based moment-orthogonalisation of the RBF block against the polynomial block.

    With `project=True` (default), Phi_rbf is projected onto the orthogonal
    complement of the *full* polynomial column span (bias included). With
    `project=False`, the projection step is skipped — polynomial and RBF
    columns can then share the same span. Used as a no-shield control to
    measure how much of the shield's guarantees came from the projection
    itself rather than the standardisation / column normalisation that
    surround it.

    The bias column must be in the projection target. If it is left out,
    Phi_perp retains its mean and stays collinear with the bias - the joint
    fit then splits constant offsets ambiguously between xi_bias and xi_rbf.
    This was the original Phase 0 design's instruction (verified 2026-06-07
    on the L63 testbed - excluding the bias gave 16% σ-error and a 40% RBF
    norm even with no closure to absorb).

    Steps:
    - Standardise non-bias columns of Theta_poly to zero mean and unit
      variance (numerical conditioning; the bias column is left as 1).
    - Economy QR on [bias | standardised non-bias] = Q R.
    - If `project`: project Phi_rbf onto orth complement: Phi_perp = Phi_rbf - Q (Q^T Phi_rbf).
      Single re-orthogonalisation pass (Giraud, Langou, Rozloznik 2005):
          Phi_perp <- Phi_perp - Q (Q^T Phi_perp).
    - Column-normalise both blocks to unit L2 norm.

    Returns
    -------
    Theta_norm : (M, N_poly) column-normalised polynomial block.
    Phi_norm   : (M, n_rbf) column-normalised RBF block (orthogonalised
                 against Theta only if `project=True`).
    scales     : dict carrying the constants needed to map fitted
                 coefficients back to raw features.
    """
    M = Theta_poly.shape[0]
    bias_idx = layout["bias_idx"]
    non_bias = [j for j in range(Theta_poly.shape[1]) if j != bias_idx]

    poly_means = Theta_poly[:, non_bias].mean(axis=0)
    poly_stds = Theta_poly[:, non_bias].std(axis=0)
    poly_stds = np.where(poly_stds == 0.0, 1.0, poly_stds)

    Theta_full = np.empty_like(Theta_poly)
    Theta_full[:, bias_idx] = Theta_poly[:, bias_idx]
    Theta_full[:, non_bias] = (Theta_poly[:, non_bias] - poly_means) / poly_stds

    Q, _R = np.linalg.qr(Theta_full, mode="reduced")
    if project:
        Phi_perp = Phi_rbf - Q @ (Q.T @ Phi_rbf)
        Phi_perp = Phi_perp - Q @ (Q.T @ Phi_perp)
    else:
        Phi_perp = Phi_rbf

    poly_col_norms = np.linalg.norm(Theta_full, axis=0)
    poly_col_norms = np.where(poly_col_norms == 0.0, 1.0, poly_col_norms)
    Theta_norm = Theta_full / poly_col_norms

    rbf_col_norms = np.linalg.norm(Phi_perp, axis=0)
    rbf_col_norms = np.where(rbf_col_norms == 0.0, 1.0, rbf_col_norms)
    Phi_norm = Phi_perp / rbf_col_norms

    scales = {
        "non_bias_idx": np.array(non_bias, dtype=np.int64),
        "poly_means": poly_means,
        "poly_stds": poly_stds,
        "poly_col_norms": poly_col_norms,
        "rbf_col_norms": rbf_col_norms,
        "Q_poly": Q,
        "shield_projected": bool(project),
    }
    return Theta_norm, Phi_norm, scales


def unscale_coefficients(xi_full: np.ndarray, layout: dict, scales: dict,
                         N_poly: int):
    """Recover raw-feature coefficients from those fit on normalised features.

    The fit minimises ||Theta_norm @ xi_n.T - Y||^2 with Theta_norm produced
    by `moment_orthogonalise`. Map `xi_n` back to coefficients of the raw
    polynomial features in `A` (un-standardised) and the *orthogonalised*
    RBF block (since the orthogonalisation is part of the model, not of the
    coordinates). The bias column also absorbs the standardisation-induced
    shifts so that calling `predict` reproduces the fit residual.
    """
    # poly columns: xi_n[j] applies to Theta_norm[:, j]. Theta_norm[:, j] =
    # (1/poly_col_norms[j]) * (poly[:, j] - mean) / std for non-bias cols,
    # and (1/poly_col_norms[bias]) for bias.
    bias_idx = layout["bias_idx"]
    non_bias = scales["non_bias_idx"]
    poly_means = scales["poly_means"]
    poly_stds = scales["poly_stds"]
    poly_col_norms = scales["poly_col_norms"]
    rbf_col_norms = scales["rbf_col_norms"]

    xi_poly_n = xi_full[:N_poly, :]
    xi_rbf_n = xi_full[N_poly:, :]

    xi_poly_raw = np.zeros_like(xi_poly_n)
    # For non-bias polynomial columns:
    #   Theta_norm[:, j] = (poly[:, j] - mean_j) / (std_j * col_norm_j)
    # So contribution to dadt = xi_n[j] * (poly[:, j] - mean_j) / (std_j * col_norm_j)
    #                          = (xi_n[j] / (std_j * col_norm_j)) * poly[:, j]
    #                            - xi_n[j] * mean_j / (std_j * col_norm_j)
    # Absorb the constant shift into the bias coefficient.
    for local_idx, j in enumerate(non_bias):
        scale = poly_stds[local_idx] * poly_col_norms[j]
        xi_poly_raw[j, :] = xi_poly_n[j, :] / scale
        xi_poly_raw[bias_idx, :] -= (
            xi_poly_n[j, :] * poly_means[local_idx] / scale
        )
    # Bias column itself.
    xi_poly_raw[bias_idx, :] += xi_poly_n[bias_idx, :] / poly_col_norms[bias_idx]

    xi_rbf_raw = xi_rbf_n / rbf_col_norms[:, None]

    return xi_poly_raw, xi_rbf_raw


# ---------------------------------------------------------------------------
# Energy-preservation constraint
# ---------------------------------------------------------------------------

def constraint_matrix(r: int) -> np.ndarray:
    """Build the energy-preservation constraint matrix on the quadratic block.

    The SINDy quadratic coefficient S has shape (r, N_quad) where
    N_quad = r(r+1)/2, ordered by `poly_feature_layout(r,2)["quad_pairs"]`
    with (j, k) and j <= k. The relation to the Schlegel-Noack tensor is

        Q_ijk = Q_ikj = factor * S[i, p_{jk}]

    with factor = 1 if j == k else 1/2. The constraint is the cyclic sum

        Q_ijk + Q_jik + Q_kij = 0    for all i, j, k in {0, ..., r-1}.

    Returns C of shape (n_constraints, r * N_quad) with rows forming an
    orthonormal basis for the row span of the raw constraint set; that is,
    the rows are independent and well-conditioned for KKT use.

    Verified against Schlegel & Noack 2015 *JFM* **765**, 325-352, Eq. 2.4
    (with symmetrisation Eq. 2.3). Phase 0 testbeds:
        r = 3: 10 constraints, 8 free Q parameters.
        r = 4: 20 constraints, 20 free Q parameters.
    """
    N_quad = r * (r + 1) // 2
    pair_to_p = {}
    p = 0
    for jj in range(r):
        for kk in range(jj, r):
            pair_to_p[(jj, kk)] = p
            p += 1

    def s_index_and_factor(out_i, in_j, in_k):
        a, b = (in_j, in_k) if in_j <= in_k else (in_k, in_j)
        return out_i * N_quad + pair_to_p[(a, b)], (1.0 if in_j == in_k else 0.5)

    n_dof = r * N_quad
    raw = []
    for i in range(r):
        for j in range(r):
            for k in range(r):
                row = np.zeros(n_dof)
                idx, f = s_index_and_factor(i, j, k); row[idx] += f
                idx, f = s_index_and_factor(j, i, k); row[idx] += f
                idx, f = s_index_and_factor(k, i, j); row[idx] += f
                raw.append(row)
    C_raw = np.asarray(raw, dtype=np.float64)
    _U, s, Vt = np.linalg.svd(C_raw, full_matrices=False)
    if len(s) == 0:
        return np.zeros((0, n_dof))
    tol = max(C_raw.shape) * np.finfo(C_raw.dtype).eps * s[0]
    rank = int(np.sum(s > tol))
    expected = {3: 10, 4: 20}.get(r)
    if expected is not None and rank != expected:
        raise AssertionError(
            f"constraint_matrix({r}): rank {rank} disagrees with expected "
            f"{expected} from Schlegel-Noack constraint counting"
        )
    return Vt[:rank]


def embed_constraint(C_quad: np.ndarray, layout: dict, n_rbf: int,
                     scales: dict | None = None) -> np.ndarray:
    """Embed the quadratic constraint into the full coefficient vector.

    The full xi has shape (N_feat, r) with N_feat = N_poly + n_rbf. The
    flattening convention is vec_F(xi) = vec(xi.T) of length r * N_feat
    so that vec_F(xi)[i * N_feat + j] = xi[j, i].

    The constraint `C_quad @ vec(S) = 0` is on the *raw* quadratic-block
    coefficients S. When the KKT solve operates on normalised coefficients
    `xi_n` (after `moment_orthogonalise`), the constraint must be expressed
    in xi_n space: `xi_raw[j, i] = xi_n[j, i] / (std_j * col_norm_j)` for
    non-bias polynomial columns, so each constraint coefficient at column j
    is divided by `std_j * col_norm_j`. If `scales` is None the raw-space
    embedding is returned (used when fitting on raw features directly).
    """
    n_c = C_quad.shape[0]
    quad_idx = layout["quad_idx"]
    N_quad = len(quad_idx)
    r = (layout["N_poly"] - 1 - N_quad)  # 1 + r + N_quad = N_poly
    N_poly = layout["N_poly"]
    N_feat = N_poly + n_rbf
    A_eq = np.zeros((n_c, r * N_feat))

    if scales is None:
        quad_scale = np.ones(N_quad)
    else:
        non_bias = list(scales["non_bias_idx"])
        poly_stds = scales["poly_stds"]
        poly_col_norms = scales["poly_col_norms"]
        quad_scale = np.empty(N_quad)
        for p, col in enumerate(quad_idx):
            local = non_bias.index(int(col))
            quad_scale[p] = poly_stds[local] * poly_col_norms[col]

    for c in range(n_c):
        for i in range(r):
            for p, col in enumerate(quad_idx):
                A_eq[c, i * N_feat + col] = C_quad[c, i * N_quad + p] / quad_scale[p]
    return A_eq


# ---------------------------------------------------------------------------
# KKT inner solve and constrained STLSQ
# ---------------------------------------------------------------------------

def _kkt_solve(Theta: np.ndarray, Y: np.ndarray, A_eq: np.ndarray,
               ridge_eps: float = 0.0,
               linear_block_mask: np.ndarray | None = None) -> np.ndarray:
    """Solve min ||Theta @ xi.T - Y||_F^2 s.t. A_eq @ vec_F(xi) = 0.

    `Theta` is (M, N_feat), `Y` is (M, r), output `xi` is (N_feat, r).
    `vec_F(xi)[i * N_feat + j] = xi[j, i]` (row-major over outputs).

    `ridge_eps` is added only on the columns flagged by
    `linear_block_mask` (length N_feat, True where ridging is allowed).
    Per `docs/notes/phase-0-design.md`, the ridge applies to the linear
    block only - never to the RBF block (would invalidate the negative
    control on L63).
    """
    M, N_feat = Theta.shape
    r = Y.shape[1]
    G_block = Theta.T @ Theta
    if ridge_eps > 0.0 and linear_block_mask is not None:
        ridge = np.zeros(N_feat)
        ridge[linear_block_mask] = ridge_eps
        G_block = G_block + np.diag(ridge)
    b_block = Theta.T @ Y  # (N_feat, r)

    n_c = A_eq.shape[0]
    if n_c == 0:
        try:
            xi = np.linalg.solve(G_block, b_block)
        except np.linalg.LinAlgError:
            xi, *_ = np.linalg.lstsq(G_block, b_block, rcond=None)
        return xi

    G_full = np.kron(np.eye(r), G_block)
    b_full = b_block.T.reshape(-1)  # length r * N_feat
    K = np.zeros((r * N_feat + n_c, r * N_feat + n_c))
    K[:r * N_feat, :r * N_feat] = G_full
    K[:r * N_feat, r * N_feat:] = A_eq.T
    K[r * N_feat:, :r * N_feat] = A_eq
    rhs = np.concatenate([b_full, np.zeros(n_c)])
    sol = np.linalg.solve(K, rhs)
    xi_vec = sol[:r * N_feat]
    return xi_vec.reshape(r, N_feat).T  # (N_feat, r)


def stlsq_constrained(
    Theta_full: np.ndarray,
    Y: np.ndarray,
    *,
    poly_col_idx: np.ndarray,
    rbf_col_idx: np.ndarray,
    linear_block_mask: np.ndarray,
    lambda_poly: float,
    lambda_rbf: float,
    A_eq_full: np.ndarray,
    max_iter: int = 20,
    ridge_eps: float = 0.0,
    tol_constraint: float = 1e-10,
):
    """Constrained STLSQ over a joint poly+RBF design matrix.

    `Theta_full` columns are laid out as poly (indices in `poly_col_idx`)
    followed by RBF (indices in `rbf_col_idx`). The energy-preservation
    constraint `A_eq_full` is enforced at *every* iteration on the active
    column set: rows of A_eq that become identically zero after column
    pruning are stripped before the KKT solve, otherwise the constraint
    Gramian becomes singular.

    Convergence requires both:
      (a) the active set stabilises across two consecutive iterations, AND
      (b) ||A_eq @ vec_F(xi_active)||_inf < tol_constraint.

    Per-block thresholds: a poly column is dropped when
    `max_i |xi[j, i]| < lambda_poly`; same for RBF with `lambda_rbf`.
    """
    M, N_feat = Theta_full.shape
    r = Y.shape[1]
    active = np.ones(N_feat, dtype=bool)
    poly_set = set(int(j) for j in poly_col_idx)
    rbf_set = set(int(j) for j in rbf_col_idx)

    history = []
    xi = np.zeros((N_feat, r))

    prev_active = None
    for it in range(max_iter):
        active_idx = np.where(active)[0]
        Theta_act = Theta_full[:, active_idx]
        lin_mask_act = linear_block_mask[active_idx]
        # Slice A_eq to the active columns of vec_F(xi).
        # vec_F(xi)[i * N_feat + j], active in (i, j) iff j is active.
        # A_eq has columns indexed by (i, j); keep cols where j in active.
        keep_cols_in_vec = np.concatenate([
            i * N_feat + active_idx for i in range(r)
        ])
        A_eq_act = A_eq_full[:, keep_cols_in_vec]
        # Strip zero rows.
        row_norms = np.linalg.norm(A_eq_act, axis=1)
        keep_rows = row_norms > 1e-12
        A_eq_act = A_eq_act[keep_rows]

        xi_act = _kkt_solve(
            Theta_act, Y, A_eq_act,
            ridge_eps=ridge_eps, linear_block_mask=lin_mask_act,
        )

        xi = np.zeros((N_feat, r))
        xi[active_idx, :] = xi_act

        # Per-block thresholding.
        new_active = active.copy()
        for j in np.where(active)[0]:
            mag = np.max(np.abs(xi[j, :]))
            if j in poly_set:
                lam = lambda_poly
            elif j in rbf_set:
                lam = lambda_rbf
            else:
                lam = 0.0
            if mag < lam:
                new_active[j] = False

        # Constraint residual on the active xi.
        if A_eq_act.size:
            xi_vec = xi[active_idx, :].T.reshape(-1)
            constraint_res = float(np.max(np.abs(A_eq_act @ xi_vec)))
        else:
            constraint_res = 0.0

        history.append({
            "iter": it,
            "n_active": int(active.sum()),
            "n_active_poly": int(sum(1 for j in np.where(active)[0]
                                     if j in poly_set)),
            "n_active_rbf": int(sum(1 for j in np.where(active)[0]
                                    if j in rbf_set)),
            "constraint_residual": constraint_res,
        })

        if (prev_active is not None and np.array_equal(new_active, active)
                and constraint_res < tol_constraint):
            break
        prev_active = active.copy()
        active = new_active

    info = {
        "n_iter": len(history),
        "active": active,
        "history": history,
        "constraint_residual": constraint_res,
        "converged": (constraint_res < tol_constraint),
    }
    return xi, info


# ---------------------------------------------------------------------------
# Derivative estimator
# ---------------------------------------------------------------------------

def deriv_5point(A: np.ndarray, dt: float):
    """5-point central stencil derivative, O(dt^4).

        a'(t_k) ~ [-a(t_{k+2}) + 8 a(t_{k+1}) - 8 a(t_{k-1}) + a(t_{k-2})] / (12 dt)

    Replaces the central FD of the prior SINDy design state. Central FD's
    leading error term on the high-acceleration wings of L63's butterfly
    is large enough to leak into `xi_rbf` magnitudes and confuse the
    negative-control verdict.

    Parameters
    ----------
    A  : (M, r) snapshot array.
    dt : float, sample step.

    Returns
    -------
    A_mid  : (M-4, r) snapshots at the interior points where the stencil
             is defined (indices 2..M-3).
    dAdt   : (M-4, r) derivative estimate at the same points.
    """
    if A.shape[0] < 5:
        raise ValueError("deriv_5point requires at least 5 snapshots")
    A_p2 = A[4:]
    A_p1 = A[3:-1]
    A_m1 = A[1:-3]
    A_m2 = A[:-4]
    dAdt = (-A_p2 + 8.0 * A_p1 - 8.0 * A_m1 + A_m2) / (12.0 * dt)
    A_mid = A[2:-2]
    return A_mid, dAdt


# ---------------------------------------------------------------------------
# Model dataclass and integration helper
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SindyModel:
    """Fitted SINDy model on raw (un-normalised) features.

    Phase 0 model: poly(deg 2) + RBF block. Two RBF kinds:
        - "flat_isotropic": `widths` populated, `Sigma_invs` is None.
        - "hier_mahalanobis": `Sigma_invs` populated, `widths` is None.
    """
    r: int
    p: int
    layout: dict
    xi_poly: np.ndarray   # (N_poly, r)
    xi_rbf: np.ndarray    # (n_rbf, r)
    rbf_kind: str
    centers: np.ndarray   # (n_rbf, r) - in standardised coords for hier_mahalanobis,
                          # in raw coords for flat_isotropic
    widths: np.ndarray | None
    Sigma_invs: np.ndarray | None  # (n_rbf, r, r) for hier_mahalanobis
    rbf_mu_A: np.ndarray
    rbf_sigma_A: np.ndarray
    scales: dict
    info: dict

    @property
    def n_rbf(self) -> int:
        return int(self.xi_rbf.shape[0])

    @property
    def xi_poly_norms(self) -> float:
        return float(np.linalg.norm(self.xi_poly))

    @property
    def xi_rbf_norm(self) -> float:
        return float(np.linalg.norm(self.xi_rbf))

    def predict_one(self, a: np.ndarray) -> np.ndarray:
        """Evaluate da/dt at a single point `a` of shape (r,)."""
        dadt = poly_apply(a, self.xi_poly, self.r, self.p)
        if self.n_rbf == 0:
            return dadt
        # RBF contribution.
        a_std = (a - self.rbf_mu_A) / self.rbf_sigma_A
        if self.rbf_kind == "flat_isotropic":
            centers_std = (self.centers - self.rbf_mu_A) / self.rbf_sigma_A
            d2 = ((a_std[None, :] - centers_std) ** 2).sum(-1)
            phi = np.exp(-0.5 * d2 / (self.widths ** 2))
        else:
            phi = np.empty(self.n_rbf)
            for j in range(self.n_rbf):
                d = a_std - self.centers[j]
                phi[j] = np.exp(-0.5 * d @ self.Sigma_invs[j] @ d)
        dadt = dadt + phi @ self.xi_rbf
        return dadt

    def predict(self, A: np.ndarray) -> np.ndarray:
        """Batch evaluate. `A` is (M, r); returns (M, r)."""
        return np.array([self.predict_one(A[m]) for m in range(A.shape[0])])


def _jsonable(obj):
    """Recursively coerce numpy scalars/arrays to JSON-serialisable Python."""
    import numpy as _np
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, _np.ndarray):
        return {"__ndarray__": obj.tolist(), "dtype": str(obj.dtype),
                "shape": list(obj.shape)}
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, (_np.bool_,)):
        return bool(obj)
    return obj


def save_model(model: SindyModel, path) -> None:
    """Persist a fitted SindyModel to a single .npz file.

    Array-valued fields go as npz arrays; layout/scales/info are JSON-encoded
    (with a small wrapper for numpy scalars / arrays) into a single string
    field `meta_json`. Reload with `load_model(path)`.
    """
    import json as _json
    from pathlib import Path as _Path
    path = _Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {
        "xi_poly": model.xi_poly,
        "xi_rbf": model.xi_rbf,
        "centers": model.centers,
        "rbf_mu_A": model.rbf_mu_A,
        "rbf_sigma_A": model.rbf_sigma_A,
    }
    if model.widths is not None:
        arrays["widths"] = model.widths
    if model.Sigma_invs is not None:
        arrays["Sigma_invs"] = model.Sigma_invs

    meta = {
        "r": int(model.r),
        "p": int(model.p),
        "rbf_kind": str(model.rbf_kind),
        "has_widths": model.widths is not None,
        "has_Sigma_invs": model.Sigma_invs is not None,
        "layout": _jsonable(model.layout),
        "scales": _jsonable(model.scales),
        "info": _jsonable(model.info),
    }
    np.savez(path, meta_json=_json.dumps(meta), **arrays)


def load_model(path) -> SindyModel:
    """Inverse of `save_model`. Reconstructs the SindyModel from npz."""
    import json as _json
    z = np.load(path, allow_pickle=False)
    meta = _json.loads(str(z["meta_json"]))

    def _decode(o):
        if isinstance(o, dict) and "__ndarray__" in o:
            return np.asarray(o["__ndarray__"], dtype=np.dtype(o["dtype"])).reshape(o["shape"])
        if isinstance(o, dict):
            return {k: _decode(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_decode(v) for v in o]
        return o

    return SindyModel(
        r=int(meta["r"]),
        p=int(meta["p"]),
        layout=_decode(meta["layout"]),
        xi_poly=z["xi_poly"],
        xi_rbf=z["xi_rbf"],
        rbf_kind=meta["rbf_kind"],
        centers=z["centers"],
        widths=z["widths"] if meta["has_widths"] else None,
        Sigma_invs=z["Sigma_invs"] if meta["has_Sigma_invs"] else None,
        rbf_mu_A=z["rbf_mu_A"],
        rbf_sigma_A=z["rbf_sigma_A"],
        scales=_decode(meta["scales"]),
        info=_decode(meta["info"]),
    )


def integrate_rk4(model: SindyModel, a0: np.ndarray, dt: float, T: float):
    """Classic RK4 integration of `da/dt = model.predict_one(a)`.

    Used by the Phase 0 long-run boundedness / energy PDF checks. The
    cluster-switching integrator of `chord2.integrator` is not yet wired
    and is not needed for Phase 0 (single-cluster fits per testbed).
    """
    n_steps = int(round(T / dt))
    t = np.linspace(0.0, T, n_steps + 1)
    A = np.empty((n_steps + 1, a0.shape[0]))
    A[0] = a0
    for i in range(n_steps):
        a = A[i]
        k1 = model.predict_one(a)
        k2 = model.predict_one(a + 0.5 * dt * k1)
        k3 = model.predict_one(a + 0.5 * dt * k2)
        k4 = model.predict_one(a + dt * k3)
        A[i + 1] = a + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.all(np.isfinite(A[i + 1])):
            return t[:i + 2], A[:i + 2]
    return t, A


# ---------------------------------------------------------------------------
# Top-level fit driver
# ---------------------------------------------------------------------------

def fit_phase0(
    A: np.ndarray,
    dt: float,
    *,
    rbf_kind: str,
    rbf_kwargs: dict,
    lambda_poly: float,
    lambda_rbf: float,
    p: int = 2,
    constrain_energy: bool = True,
    ridge_eps: float = 0.0,
    cond_max: float = 1e6,
    stlsq_max_iter: int = 20,
    drop_for_cond: bool = True,
    shield: bool = True,
):
    """End-to-end Phase 0 fit.

    Pipeline:
      1. 5-point stencil derivative -> (A_mid, dA/dt).
      2. Polynomial library Theta_poly on A_mid.
      3. RBF library Phi_rbf on A_mid, kind-dependent.
      4. Optional greedy column drop on Phi_rbf for `cond < cond_max`
         (L96 path only; not needed on L63's deliberately weak dictionary).
      5. Moment-orthogonalise Phi_rbf against Theta_poly.
      6. Build energy-preservation constraint A_eq (if `constrain_energy`).
      7. Constrained STLSQ on [Theta_poly_norm | Phi_rbf_perp_norm] with
         per-block thresholds.
      8. Unscale coefficients back to raw features.

    `rbf_kind` is "flat_isotropic" (L63) or "hier_mahalanobis" (L96).
    `rbf_kwargs` is forwarded verbatim to the centre-placement routine.
    """
    A_mid, dAdt = deriv_5point(A, dt)
    M, r = A_mid.shape

    layout = poly_feature_layout(r, p)
    Theta_poly = poly_features(A_mid, p)

    if rbf_kind == "flat_isotropic":
        centers, widths, rbf_meta = rbf_centers_flat_isotropic(
            A_mid, **rbf_kwargs
        )
        Phi_rbf_raw = rbf_features_iso(
            A_mid, centers, widths,
            mu_A=rbf_meta["mu_A"], sigma_A=rbf_meta["sigma_A"],
        )
        Sigma_invs = None
        kept_idx = np.arange(centers.shape[0])
    elif rbf_kind == "hier_mahalanobis":
        centers_std, Sigma_invs_std, parent_k, rbf_meta = (
            rbf_centers_hier_mahalanobis(A_mid, **rbf_kwargs)
        )
        Phi_rbf_raw = rbf_features_mahal(
            A_mid, centers_std, Sigma_invs_std,
            mu_A=rbf_meta["mu_A"], sigma_A=rbf_meta["sigma_A"],
        )
        if drop_for_cond:
            # Conditioning gate after orth+normalisation is the right place to
            # check; for the raw block we just record the pre-gate condition.
            pass
        centers, widths = centers_std, None  # stored in standardised coords
        Sigma_invs = Sigma_invs_std
        kept_idx = np.arange(centers.shape[0])
    else:
        raise ValueError(f"unknown rbf_kind: {rbf_kind!r}")

    Theta_norm, Phi_norm, scales = moment_orthogonalise(
        Theta_poly, Phi_rbf_raw, layout, project=shield
    )

    if drop_for_cond and Phi_norm.shape[1] > 0:
        keep = greedy_drop_for_cond(Phi_norm, cond_max=cond_max)
        if keep.shape[0] < Phi_norm.shape[1]:
            Phi_norm = Phi_norm[:, keep]
            Phi_rbf_raw = Phi_rbf_raw[:, keep]
            centers = centers[keep]
            if Sigma_invs is not None:
                Sigma_invs = Sigma_invs[keep]
            scales["rbf_col_norms"] = scales["rbf_col_norms"][keep]
            kept_idx = keep

    n_rbf = Phi_norm.shape[1]
    Theta_full = np.concatenate([Theta_norm, Phi_norm], axis=1)
    N_poly = layout["N_poly"]
    N_feat = N_poly + n_rbf
    poly_col_idx = np.arange(N_poly)
    rbf_col_idx = np.arange(N_poly, N_feat)
    # Linear block = the r linear columns (used for the optional ridge).
    linear_block_mask = np.zeros(N_feat, dtype=bool)
    linear_block_mask[layout["linear_idx"]] = True

    if constrain_energy:
        C_quad = constraint_matrix(r)
        A_eq_full = embed_constraint(C_quad, layout, n_rbf, scales=scales)
    else:
        A_eq_full = np.zeros((0, r * N_feat))

    xi_n, fit_info = stlsq_constrained(
        Theta_full, dAdt,
        poly_col_idx=poly_col_idx,
        rbf_col_idx=rbf_col_idx,
        linear_block_mask=linear_block_mask,
        lambda_poly=lambda_poly,
        lambda_rbf=lambda_rbf,
        A_eq_full=A_eq_full,
        max_iter=stlsq_max_iter,
        ridge_eps=ridge_eps,
    )

    xi_poly, xi_rbf = unscale_coefficients(xi_n, layout, scales, N_poly)

    model = SindyModel(
        r=r, p=p, layout=layout,
        xi_poly=xi_poly, xi_rbf=xi_rbf,
        rbf_kind=rbf_kind,
        centers=centers, widths=widths, Sigma_invs=Sigma_invs,
        rbf_mu_A=rbf_meta["mu_A"], rbf_sigma_A=rbf_meta["sigma_A"],
        scales=scales,
        info={
            "fit": fit_info,
            "rbf_meta": rbf_meta,
            "kept_idx": kept_idx,
            "M_used": int(M),
            "lambda_poly": float(lambda_poly),
            "lambda_rbf": float(lambda_rbf),
            "cond_max": float(cond_max),
            "ridge_eps": float(ridge_eps),
            "constrain_energy": bool(constrain_energy),
        },
    )
    return model


# ---------------------------------------------------------------------------
# Backfit architecture (two-stage). See `docs/notes/phase-0-design.md` and the
# rom-specialist consults of 2026-06-09 (R/C indicator + backfit sign-off).
#
# Stage 1: constrained quadratic STLSQ on Theta_poly alone.  Stage 2: plain
# STLSQ on the bare Phi_rbf against r1 = dAdt - Theta_poly @ xi_poly.  The
# moment-orthogonalisation of the joint construction is replaced by the
# *implicit* polynomial-orthogonality of r1 on the training measure: any
# RBF column's polynomial-aligned mass annihilates against r1, so the
# driving signal Phi_rbf^T r1 equals Phi_rbf_perp^T r1 exactly.  The
# coefficient ξ_rbf lives in the bare basis, so reads of its magnitude
# must use the *function-norm* ||Phi_rbf @ xi_rbf|| rather than ||xi_rbf||.
# ---------------------------------------------------------------------------

def fit_backfit_stage1(
    A: np.ndarray,
    dt: float,
    *,
    lambda_poly: float,
    p: int = 2,
    constrain_energy: bool = True,
    stlsq_max_iter: int = 20,
    ridge_eps: float = 0.0,
):
    """Stage 1: constrained polynomial-only STLSQ.

    Polynomial library at total degree `p` (Phase 0: p=2), Schlegel-Noack
    energy-preservation constraint on the quadratic tensor, hard L0
    thresholding at `lambda_poly` on column-normalised features. No RBF
    block. Returns a dict with the fitted coefficients (in raw-feature
    coordinates), the residual `r1 = dAdt - Theta_poly_raw @ xi_poly_raw`
    on the interior 5-point-stencil snapshots, and the snapshot arrays so
    stage 2 can reuse them without recomputing the derivative.
    """
    A_mid, dAdt = deriv_5point(A, dt)
    M, r = A_mid.shape

    layout = poly_feature_layout(r, p)
    Theta_poly_raw = poly_features(A_mid, p)

    empty_phi = np.zeros((M, 0), dtype=Theta_poly_raw.dtype)
    Theta_norm, _Phi_norm, scales = moment_orthogonalise(
        Theta_poly_raw, empty_phi, layout, project=True,
    )

    N_poly = layout["N_poly"]
    poly_col_idx = np.arange(N_poly)
    rbf_col_idx = np.zeros(0, dtype=np.int64)
    linear_block_mask = np.zeros(N_poly, dtype=bool)
    linear_block_mask[layout["linear_idx"]] = True

    if constrain_energy:
        C_quad = constraint_matrix(r)
        A_eq_full = embed_constraint(C_quad, layout, n_rbf=0, scales=scales)
    else:
        A_eq_full = np.zeros((0, r * N_poly))

    xi_n, fit_info = stlsq_constrained(
        Theta_norm, dAdt,
        poly_col_idx=poly_col_idx,
        rbf_col_idx=rbf_col_idx,
        linear_block_mask=linear_block_mask,
        lambda_poly=lambda_poly,
        lambda_rbf=0.0,
        A_eq_full=A_eq_full,
        max_iter=stlsq_max_iter,
        ridge_eps=ridge_eps,
    )

    xi_poly_raw, _xi_rbf_raw = unscale_coefficients(xi_n, layout, scales, N_poly)
    dAdt_pred = Theta_poly_raw @ xi_poly_raw
    r1 = dAdt - dAdt_pred

    return {
        "A_mid": A_mid,
        "dAdt": dAdt,
        "Theta_poly_raw": Theta_poly_raw,
        "xi_poly": xi_poly_raw,
        "dAdt_pred": dAdt_pred,
        "r1": r1,
        "layout": layout,
        "fit_info": fit_info,
        "p": p,
        "constrain_energy": bool(constrain_energy),
        "lambda_poly": float(lambda_poly),
    }


def _stlsq_plain(Phi: np.ndarray, Y: np.ndarray, lam: float,
                 max_iter: int = 20) -> tuple[np.ndarray, dict]:
    """Unconstrained STLSQ on a single feature block.

    `Phi` is (M, n), `Y` is (M, r), output `xi` is (n, r). Active set
    selected by `max_i |xi[j, i]| >= lam`; columns are assumed
    pre-normalised so `lam` has the same meaning across columns.
    """
    M, n = Phi.shape
    r = Y.shape[1]
    active = np.ones(n, dtype=bool)
    xi = np.zeros((n, r))
    history = []
    for it in range(max_iter):
        idx = np.where(active)[0]
        if idx.size == 0:
            history.append({"iter": it, "n_active": 0})
            break
        xi_act, *_ = np.linalg.lstsq(Phi[:, idx], Y, rcond=None)
        xi = np.zeros((n, r))
        xi[idx, :] = xi_act
        new_active = active.copy()
        for j in idx:
            if np.max(np.abs(xi[j, :])) < lam:
                new_active[j] = False
        history.append({"iter": it, "n_active": int(active.sum())})
        if np.array_equal(new_active, active):
            break
        active = new_active
    return xi, {"n_iter": len(history), "history": history,
                "active": active.copy()}


def fit_backfit_stage2(
    stage1: dict,
    *,
    rbf_kind: str,
    rbf_kwargs: dict,
    lambda_rbf: float,
    cond_max: float = 1e6,
    stlsq_max_iter: int = 20,
):
    """Stage 2: plain STLSQ of the bare RBF block against the stage-1 residual.

    No moment-orthogonalisation: r1 is already polynomial-orthogonal on the
    training measure by construction. No constraint: the Schlegel-Noack
    constraint is on the quadratic tensor only and stage 1 already enforced
    it. The conditioning gate on the *bare* Phi_rbf is still mandatory and
    typically bites harder than under the orthogonalised joint construction
    (polynomial-aligned mass amplifies the constant-column overlap).

    Inputs come from `fit_backfit_stage1`. Outputs are returned alongside
    the stage-1 dict's reusable arrays so the caller can assemble the full
    backfit model in one place.
    """
    A_mid = stage1["A_mid"]
    r1 = stage1["r1"]
    M, r = A_mid.shape

    if rbf_kind == "flat_isotropic":
        centers, widths, rbf_meta = rbf_centers_flat_isotropic(
            A_mid, **rbf_kwargs,
        )
        Phi_rbf_raw = rbf_features_iso(
            A_mid, centers, widths,
            mu_A=rbf_meta["mu_A"], sigma_A=rbf_meta["sigma_A"],
        )
        Sigma_invs = None
    elif rbf_kind == "hier_mahalanobis":
        centers_std, Sigma_invs_std, _parent_k, rbf_meta = (
            rbf_centers_hier_mahalanobis(A_mid, **rbf_kwargs)
        )
        Phi_rbf_raw = rbf_features_mahal(
            A_mid, centers_std, Sigma_invs_std,
            mu_A=rbf_meta["mu_A"], sigma_A=rbf_meta["sigma_A"],
        )
        centers = centers_std
        widths = None
        Sigma_invs = Sigma_invs_std
    else:
        raise ValueError(f"unknown rbf_kind: {rbf_kind!r}")

    if Phi_rbf_raw.shape[1] == 0:
        kept_idx = np.zeros(0, dtype=np.int64)
        return {
            "xi_rbf": np.zeros((0, r)),
            "Phi_rbf_raw": Phi_rbf_raw,
            "centers": centers, "widths": widths, "Sigma_invs": Sigma_invs,
            "rbf_kind": rbf_kind, "rbf_meta": rbf_meta,
            "kept_idx": kept_idx,
            "fit_info": {"n_iter": 0, "history": [], "active": np.zeros(0, bool)},
            "lambda_rbf": float(lambda_rbf), "cond_max": float(cond_max),
            "cond_pre_gate": float("nan"), "cond_post_gate": float("nan"),
        }

    col_norms = np.linalg.norm(Phi_rbf_raw, axis=0)
    col_norms = np.where(col_norms == 0.0, 1.0, col_norms)
    Phi_norm = Phi_rbf_raw / col_norms[None, :]

    cond_pre = float(np.linalg.cond(Phi_norm))
    keep = greedy_drop_for_cond(Phi_norm, cond_max=cond_max)
    if keep.shape[0] < Phi_norm.shape[1]:
        Phi_norm = Phi_norm[:, keep]
        Phi_rbf_raw = Phi_rbf_raw[:, keep]
        col_norms = col_norms[keep]
        centers = centers[keep]
        if Sigma_invs is not None:
            Sigma_invs = Sigma_invs[keep]
    cond_post = float(np.linalg.cond(Phi_norm)) if Phi_norm.shape[1] > 0 else float("nan")

    xi_n, fit_info = _stlsq_plain(Phi_norm, r1, lam=lambda_rbf,
                                  max_iter=stlsq_max_iter)
    xi_rbf_raw = xi_n / col_norms[:, None]

    return {
        "xi_rbf": xi_rbf_raw,
        "Phi_rbf_raw": Phi_rbf_raw,
        "centers": centers, "widths": widths, "Sigma_invs": Sigma_invs,
        "rbf_kind": rbf_kind, "rbf_meta": rbf_meta,
        "kept_idx": keep,
        "fit_info": fit_info,
        "lambda_rbf": float(lambda_rbf), "cond_max": float(cond_max),
        "cond_pre_gate": cond_pre, "cond_post_gate": cond_post,
    }


def fit_phase0_backfit(
    A: np.ndarray,
    dt: float,
    *,
    rbf_kind: str,
    rbf_kwargs: dict,
    lambda_poly: float,
    lambda_rbf: float,
    p: int = 2,
    constrain_energy: bool = True,
    cond_max: float = 1e6,
    stlsq_max_iter: int = 20,
    ridge_eps: float = 0.0,
) -> SindyModel:
    """Compose stage 1 + stage 2 into a single SindyModel.

    The returned SindyModel evaluates `da/dt = poly(a) + Phi_rbf(a) @ xi_rbf`
    via the standard `predict` / `predict_one` path; the only thing that
    changes relative to `fit_phase0` is *how* `xi_poly` and `xi_rbf` were
    obtained.
    """
    s1 = fit_backfit_stage1(
        A, dt, lambda_poly=lambda_poly, p=p,
        constrain_energy=constrain_energy,
        stlsq_max_iter=stlsq_max_iter,
        ridge_eps=ridge_eps,
    )
    s2 = fit_backfit_stage2(
        s1, rbf_kind=rbf_kind, rbf_kwargs=rbf_kwargs,
        lambda_rbf=lambda_rbf, cond_max=cond_max,
        stlsq_max_iter=stlsq_max_iter,
    )

    A_mid = s1["A_mid"]
    r = A_mid.shape[1]
    return SindyModel(
        r=r, p=p, layout=s1["layout"],
        xi_poly=s1["xi_poly"], xi_rbf=s2["xi_rbf"],
        rbf_kind=s2["rbf_kind"],
        centers=s2["centers"], widths=s2["widths"], Sigma_invs=s2["Sigma_invs"],
        rbf_mu_A=s2["rbf_meta"]["mu_A"], rbf_sigma_A=s2["rbf_meta"]["sigma_A"],
        scales={"backfit": True},
        info={
            "architecture": "backfit",
            "stage1_fit": s1["fit_info"],
            "stage2_fit": s2["fit_info"],
            "rbf_meta": s2["rbf_meta"],
            "kept_idx": s2["kept_idx"],
            "M_used": int(A_mid.shape[0]),
            "lambda_poly": float(lambda_poly),
            "lambda_rbf": float(lambda_rbf),
            "cond_max": float(cond_max),
            "cond_pre_gate": s2["cond_pre_gate"],
            "cond_post_gate": s2["cond_post_gate"],
            "constrain_energy": bool(constrain_energy),
            "stage1_residual_norm": float(np.linalg.norm(s1["r1"])),
            "dAdt_norm": float(np.linalg.norm(s1["dAdt"])),
        },
    )
