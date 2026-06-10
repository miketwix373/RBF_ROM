"""Per-cluster local PCA bases and change-of-coordinates (paper Section 2.2.1).

For each cluster `k`:
1. Subtract the centroid (Eq. 9): `u'_m = u_m - c_{beta_c(m)}`.
2. Build the cluster snapshot matrix `Q'_k = [u'_{m_1}, ..., u'_{m_{n_k}}]` (Eq. 10).
3. SVD: `Q'_k = U_k Sigma_k V_k^H` (Eq. 11). Retain the `r_k` leading modes.
4. Reduced coordinates: `a^k(t) = U_k^{r_k,H} (u(t) - c_k)`.

For the cluster-switching prediction loop (Algorithm 1):
- Change-of-coordinates matrix `U_j^H U_i` and centroid-shift vector
  `U_j^H (c_i - c_j)` are precomputed for every ordered pair `(i, j)` of
  clusters that the affiliation function actually visits, and cached.
- These are the only quantities needed to translate a reduced state from
  the basis of cluster `i` to the basis of cluster `j` (Eq. 14).

The number of retained modes `r_k` is a user-chosen hyperparameter. In the
paper the same `r_k = r` is used for all clusters; CHORD2 inherits that
default but does not hard-code it.
"""

raise NotImplementedError("chord2.local_basis: stubs only; implementation pending.")
