"""Cluster-switching ROM integration (paper Section 2.2, Algorithm 1).

Given an initial condition `u_{r0}` and a fitted quantized local ROM:

1. Initial cluster index: `k_0 = beta_c(u_{r0})`.
2. Project to local reduced coordinates: `a^{k_0}_0 = U_{k_0}^H (u_{r0} - c_{k_0})`.
3. Integrate the local ODE (in CHORD2: the SINDy model
   `da^k/dt = Theta(a^k) . xi^k`) until the affiliation function indicates a
   change of cluster.
4. On a switch `i -> j`:
   - Reconstruct the physical state: `u_r = c_i + U_i a^i`.
   - Apply the change of coordinates:
       `a^j = U_j^H U_i . a^i + U_j^H (c_i - c_j)`     (Eq. 14)
   - Continue integration with the cluster-`j` SINDy model.
5. Store the reduced trajectory `a^k(t)`, the cluster-affiliation sequence
   `beta_c(t)`, and the reconstructed physical state `u_r(t) = c_k + U_k a^k(t)`.

The dynamics function is intentionally abstract so the same integrator
drives the local-SINDy ROM and the POD-Galerkin baselines under
`chord2.baselines.galerkin`. Time stepping defaults to RK4 (matches the
Lorenz 96 driver in `data/LOR96/lor96.py`) with a user-overridable scheme.

Paper footnote 1 notes that the reconstructed solution may be non-
differentiable at cluster boundaries; this is left as a known caveat for
now, with future work to add a spline-based smoother.
"""

raise NotImplementedError("chord2.integrator: stubs only; implementation pending.")
