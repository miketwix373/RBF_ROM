"""Phase-space quantization via K-means (paper Section 2.1).

Implements the cartography step of Fig. 1 / Fig. 2 of the paper:
- K-means++ clustering of snapshots in the full state space (Eqs. 7-8).
- Cluster centroids `c_k` as cluster barycentres (Eq. 5).
- Affiliation function `beta_c(u) = argmin_i ||u - c_i||` (Eq. 2).
- Bayesian Information Criterion for selecting `K` (Eq. 18 with J dropped).

Distance metric is squared Euclidean (Eq. 4); no per-component normalisation,
no POD pre-projection. Lloyd's iteration is rolled in numpy/cupy to keep the
dependency surface small; the `xp` argument selects the array module.

Scope of this module: anything that depends *only* on the raw snapshot matrix
`U` of shape `(M, N)` and the integer label vector `labels` of shape `(M,)`.
Field-touching diagnostics (per-cluster energy, local-PCA incompressibility,
indicator overlap, ARI shift-and-recluster) live in `chord2.diagnostics`
because they cross module boundaries to either `chord2.local_basis` or to
the per-dataset indicator function.
"""

from __future__ import annotations

import math

import numpy as np


# Lloyd batches rows of U when computing the (M, K) squared-distance matrix
# so the temporary stays at most ~1 GB float64. 4096 rows times a few thousand
# clusters and a few hundred ambient dims is well under that. KOL42-scale
# state vectors (N ~ 1e5) may want this lower; private constant for now.
_LLOYD_ROW_CHUNK = 4096


def _xp_of(a, xp=None):
    """Resolve the array module of `a` if `xp` is None.

    Uses `cupy.get_array_module` when cupy is importable so that callers can
    pass a cupy array without explicitly threading `xp`. Falls back to numpy.
    """
    if xp is not None:
        return xp
    try:
        import cupy as _cp  # type: ignore
        return _cp.get_array_module(a)
    except ImportError:
        return np


def _pairwise_sq_dists_chunked(U, centroids, xp):
    """Return the (M, K) squared-Euclidean distance matrix in row chunks.

    Avoids materialising the full `(M, K, N)` broadcast tensor; instead loops
    over `_LLOYD_ROW_CHUNK`-row slices and writes into a preallocated buffer.
    The `||u - c||^2 = ||u||^2 + ||c||^2 - 2 u.c` identity would be one matmul
    cheaper but loses precision when the inertia is small relative to the row
    norms; the explicit broadcast keeps Eq. 4 verbatim.
    """
    M = U.shape[0]
    K = centroids.shape[0]
    out = xp.empty((M, K), dtype=U.dtype)
    for lo in range(0, M, _LLOYD_ROW_CHUNK):
        hi = min(lo + _LLOYD_ROW_CHUNK, M)
        diff = U[lo:hi, None, :] - centroids[None, :, :]
        out[lo:hi] = (diff * diff).sum(-1)
    return out


def _kmeans_pp_init(U, K, gen, xp):
    """K-means++ centroid seeding (Arthur & Vassilvitskii 2007).

    First centroid uniform at random from rows of `U`; each subsequent centroid
    drawn with probability proportional to its squared distance to the nearest
    already-picked centroid. `gen` is a numpy `Generator` and supplies the
    randomness even when `xp` is cupy (the index draws are cheap on host).
    """
    M, N = U.shape
    centroids = xp.empty((K, N), dtype=U.dtype)
    first = int(gen.integers(0, M))
    centroids[0] = U[first]
    if K == 1:
        return centroids

    # Track running minimum squared distance from each row to picked centroids.
    diff = U - centroids[0]
    closest_sq = (diff * diff).sum(-1)
    for k in range(1, K):
        total = float(closest_sq.sum())
        if total <= 0.0:
            # All snapshots already coincide with picked centroids; pick any.
            idx = int(gen.integers(0, M))
        else:
            # `closest_sq` may live on the GPU; copy to host for np.random sampling.
            probs = closest_sq / total
            probs_host = np.asarray(probs.get() if hasattr(probs, "get") else probs)
            idx = int(gen.choice(M, p=probs_host))
        centroids[k] = U[idx]
        diff = U - centroids[k]
        new_sq = (diff * diff).sum(-1)
        closest_sq = xp.minimum(closest_sq, new_sq)
    return centroids


def _lloyd(U, centroids, *, tol, max_iter, xp):
    """Run Lloyd iteration to convergence, returning labels, centroids, inertia, n_iter."""
    K = centroids.shape[0]
    n_iter = 0
    labels = xp.zeros(U.shape[0], dtype=xp.int32)
    inertia = float("inf")

    for n_iter in range(1, max_iter + 1):
        d2 = _pairwise_sq_dists_chunked(U, centroids, xp)
        labels = xp.argmin(d2, axis=1).astype(xp.int32)
        # Inertia is the sum of the min squared distances.
        inertia = float(xp.take_along_axis(d2, labels[:, None].astype(xp.int64), axis=1).sum())

        new_centroids = xp.empty_like(centroids)
        for k in range(K):
            mask = labels == k
            count = int(mask.sum())
            if count == 0:
                # Re-seed the empty cluster at the snapshot farthest from its
                # current affiliated centroid - standard Lloyd recovery.
                row_min = xp.take_along_axis(
                    d2, labels[:, None].astype(xp.int64), axis=1
                ).reshape(-1)
                far = int(xp.argmax(row_min))
                new_centroids[k] = U[far]
            else:
                new_centroids[k] = U[mask].mean(axis=0)

        shift = float(xp.linalg.norm(new_centroids - centroids))
        scale = float(xp.linalg.norm(centroids))
        centroids = new_centroids
        if shift < tol * max(scale, 1.0):
            break

    return labels, centroids, inertia, n_iter


def kmeans_fit(U, K, *, seed, n_init=10, tol=1e-4, max_iter=300, xp=None):
    """Fit K-means with K-means++ init and Lloyd iteration.

    Paper Section 2.1, Eqs. 7-8. Squared Euclidean, no normalisation.

    `K=1` is an allowed input and short-circuits: centroid is the column mean,
    labels is zeros, inertia is the total variance, n_iter is 0. This case is
    required by the rubric in `docs/notes/clustering-success-criterion.md`
    where RB and KOL20 are expected to score "no clustering needed".

    Parameters
    ----------
    U : (M, N) array on `xp`
        Snapshot matrix; rows are snapshots, columns are state-vector entries.
    K : int
        Number of clusters, `K >= 1`.
    seed : int
        Single global RNG seed for both the K-means++ init and the `n_init`
        restarts.
    n_init, tol, max_iter
        Lloyd hyperparameters; defaults match the user-locked rubric.
    xp : module
        Backend; defaults to numpy. CuPy works.

    Returns
    -------
    labels : (M,) int32 on `xp`
    centroids : (K, N) float32 on `xp`
    inertia : float
        Total within-cluster squared-Euclidean inertia of the best restart.
    n_iter : int
        Lloyd iterations taken by the best restart.
    """
    xp = _xp_of(U, xp)
    M = U.shape[0]

    if K == 1:
        centroid = U.mean(axis=0, keepdims=True).astype(xp.float32)
        labels = xp.zeros(M, dtype=xp.int32)
        diff = U.astype(centroid.dtype) - centroid
        inertia = float((diff * diff).sum())
        return labels, centroid, inertia, 0

    parent = np.random.default_rng(seed)
    sub_seeds = parent.integers(0, 2**31, size=n_init)

    best = None  # (inertia, n_iter, labels, centroids)
    for s in sub_seeds:
        gen = np.random.default_rng(int(s))
        init = _kmeans_pp_init(U, K, gen, xp)
        labels, centroids, inertia, n_iter = _lloyd(
            U, init, tol=tol, max_iter=max_iter, xp=xp
        )
        key = (inertia, n_iter)
        if best is None or key < (best[0], best[1]):
            best = (inertia, n_iter, labels, centroids)

    inertia, n_iter, labels, centroids = best
    return labels, centroids.astype(xp.float32), inertia, n_iter


def kmeans_predict_label(u, centroids):
    """Affiliation function `beta_c(u)` (paper Eq. 2).

    Works on `u` of shape `(N,)` (single snapshot, returns int) or `(M, N)`
    (batch, returns `(M,)` int32 array). The array module is inferred from
    `centroids`.
    """
    xp = _xp_of(centroids)
    if u.ndim == 1:
        diff = u[None, :] - centroids
        return int(xp.argmin((diff * diff).sum(-1)))
    diff = u[..., None, :] - centroids
    return xp.argmin((diff * diff).sum(-1), axis=-1).astype(xp.int32)


def bic(n_k, M):
    """Paper Eq. 18 with the `J` term dropped.

        BIC = M log M + K log M - 2 * Sum_k n_k log(n_k / M).

    The inner-cluster variance term `J / sigma^2` is omitted because the
    sweep does not fit local PCAs; the remaining BIC is a pure function of
    the cluster populations. Paper notes the entropy term is negligible for
    `N >> 1` but the rubric keeps it; the M log M and K log M terms then
    dominate the K-vs-K comparison.

    Parameters
    ----------
    n_k : (K,) int array
        Per-cluster populations; must sum to `M`.
    M : int
        Total snapshot count.

    Returns
    -------
    float
    """
    n_k = np.asarray(n_k)
    assert int(n_k.sum()) == int(M), \
        f"n_k must sum to M; got sum(n_k)={int(n_k.sum())}, M={M}"
    K = int(n_k.shape[0])
    entropy = float((n_k * np.log(n_k / M)).sum())
    return float(M) * np.log(M) + float(K) * np.log(M) - 2.0 * entropy


def bic_eff(n_k, M, tau_int_steps):
    """BIC with `M_eff = M / (2 * tau_int_steps)` substituted for `M`.

    Effective-sample correction proposed in
    `docs/notes/clustering-success-criterion.md` (Open question:
    BIC effective-sample correction); not in the paper. The substitution
    is applied to the `M log M` and `K log M` terms; the entropy term
    keeps `n_k` as-is because cluster populations are observables, not
    sample-count parameters. Both `bic(M)` and `bic_eff(M_eff)` are
    reported by the sweep; the user picks K in a journal entry.

    Parameters
    ----------
    n_k : (K,) int array
    M : int
    tau_int_steps : float
        Integral timescale of the trajectory expressed in sample steps.

    Returns
    -------
    float
    """
    n_k = np.asarray(n_k)
    assert int(n_k.sum()) == int(M), \
        f"n_k must sum to M; got sum(n_k)={int(n_k.sum())}, M={M}"
    K = int(n_k.shape[0])
    M_eff = float(M) / (2.0 * float(tau_int_steps))
    entropy = float((n_k * np.log(n_k / M)).sum())
    return M_eff * np.log(M_eff) + float(K) * np.log(M_eff) - 2.0 * entropy


def residence_time_stats(labels):
    """Run-length statistics of the cluster-label sequence.

    A run is a maximal constant-label substring; `np.diff` locates the
    transition indices in O(M). For K=1 the trajectory is a single run of
    length `M`, all four percentiles equal `M`, and `runs_per_cluster = [1]`.

    Returns
    -------
    dict with keys
        mean, median, p10, p90 : float
            Sample-step run-length statistics.
        runs_per_cluster : (K,) int array
            Number of runs ending in each label, with `K = labels.max() + 1`.
    """
    labels = np.asarray(labels)
    M = labels.shape[0]
    K = int(labels.max()) + 1

    change = np.flatnonzero(np.diff(labels) != 0) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [M]))
    run_lengths = ends - starts
    run_labels = labels[starts]

    runs_per_cluster = np.zeros(K, dtype=np.int64)
    for lbl in run_labels:
        runs_per_cluster[int(lbl)] += 1

    return {
        "mean": float(run_lengths.mean()),
        "median": float(np.median(run_lengths)),
        "p10": float(np.percentile(run_lengths, 10)),
        "p90": float(np.percentile(run_lengths, 90)),
        "runs_per_cluster": runs_per_cluster,
    }


def switch_count(labels):
    """Number of cluster transitions along the trajectory.

    Convention: `np.count_nonzero(np.diff(labels))`, so `K=1` returns 0.
    """
    xp = _xp_of(labels)
    return int(xp.count_nonzero(xp.diff(labels)))


def transition_matrix(labels, K):
    """Empirical (row-stochastic) `K x K` Markov transition matrix.

    `P[i, j] = count(label_m = i AND label_{m+1} = j) / count(label_m = i)`
    over `m < M-1`. Rows for never-visited (as a source) clusters are set to
    uniform `1/K`. For `K=1` the matrix is `[[1.0]]`.
    """
    labels = np.asarray(labels)
    P = np.zeros((K, K), dtype=np.float64)
    if K == 1:
        P[0, 0] = 1.0
        return P
    src = labels[:-1]
    dst = labels[1:]
    np.add.at(P, (src, dst), 1.0)
    row_sums = P.sum(axis=1)
    for i in range(K):
        if row_sums[i] == 0.0:
            P[i] = 1.0 / K
        else:
            P[i] /= row_sums[i]
    return P


def per_snapshot_distance(U, centroids, labels):
    """Per-snapshot Euclidean distance to the affiliated centroid.

    Returns `(M,)` float32 with `||u_m - c_{labels[m]}||_2`. Note the square
    root: callers wanting the K-means inertia should square and sum.

    NB. This is centroid-relative. The KS_bursting physical indicator
    `||u - u_bar||_2` (mean-relative) is a separate quantity owned by the
    dataset's `indicator.py` and lives in `chord2.diagnostics` for overlay.
    """
    xp = _xp_of(U)
    assigned = centroids[labels.astype(xp.int64)]
    diff = U - assigned
    return xp.sqrt((diff * diff).sum(-1)).astype(xp.float32)


def adjusted_rand_index(labels_a, labels_b):
    """Adjusted Rand Index of two label vectors of equal length.

    Standard Hubert & Arabie (1985) formulation. Returns 1.0 for identical
    partitions (up to label permutation) and expectation ~0 for independent
    random partitions.

    Lives here even though the broader ARI shift-and-recluster machinery is
    a `chord2.diagnostics` orchestration: the *label-pair primitive* is
    clustering-side because it operates only on the integer label vectors
    that this module produces. The *shift-and-recluster* loop, which calls
    `kmeans_fit` on a cyclically-shifted snapshot matrix, lives in
    `chord2.diagnostics` because it crosses module boundaries.

    Parameters
    ----------
    labels_a, labels_b : (M,) int arrays of equal length.

    Returns
    -------
    float in approximately [-1, 1].
    """
    a = np.asarray(labels_a).ravel()
    b = np.asarray(labels_b).ravel()
    assert a.shape == b.shape, f"label-vector shapes differ: {a.shape} vs {b.shape}"

    na = int(a.max()) + 1
    nb = int(b.max()) + 1
    cm = np.zeros((na, nb), dtype=np.int64)
    np.add.at(cm, (a, b), 1)

    def _comb2(x):
        return x * (x - 1) // 2

    sum_ij = int(_comb2(cm).sum())
    sum_a = int(_comb2(cm.sum(axis=1)).sum())
    sum_b = int(_comb2(cm.sum(axis=0)).sum())
    n_pairs = int(_comb2(np.int64(a.shape[0])))

    expected = (sum_a * sum_b) / n_pairs if n_pairs > 0 else 0.0
    max_index = 0.5 * (sum_a + sum_b)
    denom = max_index - expected
    if denom == 0.0:
        return 1.0
    return float((sum_ij - expected) / denom)


def adjusted_mutual_information(labels_a, labels_b):
    """Adjusted Mutual Information of two label vectors of equal length.

    Vinh, Epps & Bailey (2010), "Information Theoretic Measures for
    Clusterings Comparison: Variants, Properties, Normalization and
    Correction for Chance", JMLR 11. Returns

        AMI = (MI - E[MI]) / (max(H(a), H(b)) - E[MI])

    with `E[MI]` computed via the exact lgamma-based hypergeometric
    expectation in Eq. 24a of the paper. Returns 1.0 on identical
    partitions (up to label permutation) and expectation near 0 for
    independent random partitions.

    Lives here for the same reason as `adjusted_rand_index`: AMI is a
    label-vector primitive that touches only the integer labels produced
    by `kmeans_fit`. The shift-and-recluster orchestration that consumes
    it sits in `chord2.diagnostics` (signal 1 of the rubric, 2-regime
    ground truth with K-cluster prediction).

    Parameters
    ----------
    labels_a, labels_b : (M,) int arrays of equal length.

    Returns
    -------
    float in approximately [0, 1].
    """
    a = np.asarray(labels_a).ravel()
    b = np.asarray(labels_b).ravel()
    assert a.shape == b.shape, f"label-vector shapes differ: {a.shape} vs {b.shape}"

    M = a.shape[0]
    na = int(a.max()) + 1
    nb = int(b.max()) + 1
    cm = np.zeros((na, nb), dtype=np.int64)
    np.add.at(cm, (a, b), 1)
    ai = cm.sum(axis=1).astype(np.int64)  # row marginals
    bj = cm.sum(axis=0).astype(np.int64)  # column marginals

    def _entropy(counts):
        nz = counts[counts > 0].astype(np.float64)
        p = nz / float(M)
        return float(-(p * np.log(p)).sum())

    H_a = _entropy(ai)
    H_b = _entropy(bj)

    # MI = sum_{ij} (n_ij / M) * log( (n_ij * M) / (a_i * b_j) )
    MI = 0.0
    for i in range(na):
        for j in range(nb):
            nij = int(cm[i, j])
            if nij == 0:
                continue
            MI += (nij / M) * (math.log(nij) + math.log(M)
                               - math.log(int(ai[i])) - math.log(int(bj[j])))

    # E[MI] = sum_{i,j} sum_{n=max(1,a_i+b_j-M)}^{min(a_i,b_j)}
    #         (n/M) * log( (n*M) / (a_i b_j) )
    #         * exp( lgamma(a_i+1) + lgamma(b_j+1) + lgamma(M-a_i+1) + lgamma(M-b_j+1)
    #                - lgamma(M+1) - lgamma(n+1)
    #                - lgamma(a_i-n+1) - lgamma(b_j-n+1)
    #                - lgamma(M-a_i-b_j+n+1) )
    lg = math.lgamma
    log_M = math.log(M)
    EMI = 0.0
    for i in range(na):
        ai_i = int(ai[i])
        if ai_i == 0:
            continue
        for j in range(nb):
            bj_j = int(bj[j])
            if bj_j == 0:
                continue
            n_lo = max(1, ai_i + bj_j - M)
            n_hi = min(ai_i, bj_j)
            log_const = (lg(ai_i + 1) + lg(bj_j + 1)
                         + lg(M - ai_i + 1) + lg(M - bj_j + 1)
                         - lg(M + 1))
            log_ab = math.log(ai_i) + math.log(bj_j)
            for n in range(n_lo, n_hi + 1):
                log_pn = log_const - (lg(n + 1)
                                      + lg(ai_i - n + 1)
                                      + lg(bj_j - n + 1)
                                      + lg(M - ai_i - bj_j + n + 1))
                EMI += (n / M) * (math.log(n) + log_M - log_ab) * math.exp(log_pn)

    denom = max(H_a, H_b) - EMI
    if denom == 0.0:
        return 1.0
    return float((MI - EMI) / denom)


# ---------------------------------------------------------------------------
# Persistence schema for the sweep.
# ---------------------------------------------------------------------------
#
# `scripts/run_kmeans_sweep.py` writes a sharded sweep under
# `results/<NAME>/cluster_sweep/`:
#
#   summary.npz  - always cheap to load; one file for the whole sweep.
#     K_values        int32  (L,)        K values swept, sorted ascending
#     inertia         float64(L,)        best-restart inertia per K
#     bic             float64(L,)        paper Eq. 18 (J dropped), bic(n_k, M)
#     bic_eff         float64(L,)        bic_eff(n_k, M, tau_int_steps)
#     n_iter          int32  (L,)        Lloyd iterations taken
#     seed            int64               global RNG seed
#     tau_int_steps   float64             dataset integral timescale (sample steps)
#     M               int64               snapshot count
#     N               int64               ambient dimension
#     dataset_name    str
#     stride          int32
#     created_utc     str                 ISO-8601 timestamp
#     chord2_version  str
#
#   K=01.npz, K=02.npz, ..., K=KK.npz - one shard per K. Selection only
#     reads the shards for its candidate K values.
#     labels             int32  (M,)
#     centroids          float32(K, N)
#     inertia            float64
#     bic                float64
#     bic_eff            float64
#     n_k                int64  (K,)        cluster populations (input to bic/bic_eff)
#     residence_mean     float64            run-length statistics
#     residence_median   float64
#     residence_p10      float64
#     residence_p90      float64
#     runs_per_cluster   int64  (K,)
#     switch_count       int64
#     transition_matrix  float64(K, K)
#     dist_to_centroid   float32(M,)        per_snapshot_distance output
#     n_iter             int32
#
# A `sweep.log` plain-text file alongside records seed, per-K wall time,
# device used, and the memory-footprint decision (see chord2.backend).
# ---------------------------------------------------------------------------
