"""Global and quantized-local POD-Galerkin ROMs (paper Eqs. 12-13, Algorithm 1).

Faithful reproduction of the intrusive method that the paper presents:

    da^k/dt + B^k a^k + N^k(a^k, c_k) + f^k = 0,   k = 1, ..., K   (Eq. 13)

obtained by Galerkin projection of the full-order PDE onto the local POD
basis of cluster `k`. With `K = 1` this collapses to the classical global
POD-Galerkin ROM, which serves as the second baseline ("g-ROM" in the paper).

Required inputs:
- A problem-specific evaluator for the FOM nonlinear operator `N(u, t)` at a
  vector of test states. This is the *only* place in CHORD2 where the FOM
  operator appears - the rest of the library is non-intrusive by design.
  The evaluator is supplied by the dataset-specific driver in `data/<problem>/`.

This module exists purely to generate the comparison curves and statistics
that justify the local-SINDy substitution. It is one-shot reference code -
do not depend on it from the rest of `chord2/`.
"""

raise NotImplementedError("chord2.baselines.galerkin: stubs only; implementation pending.")
