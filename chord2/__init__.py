"""CHORD2 - quantized local reduced-order modelling with local SINDy.

Implements the ql-ROM framework of Colanera & Magri,
"Quantized local reduced-order modeling in time",
Comput. Methods Appl. Mech. Engrg. 447 (2025) 118393,
with the per-cluster intrusive POD-Galerkin dynamics (Eq. 13) replaced by
a non-intrusive sparse-regression identification of `da^k/dt = Theta(a^k) xi^k`
on the local reduced coordinates. The clustering (Section 2.1), per-cluster
PCA (Section 2.2.1, Eqs. 9-11), and cluster-switching change of coordinates
(Eq. 14, Algorithm 1) of the paper are retained unchanged.
"""

from chord2.backend import get_backend, asnumpy, to_device, describe_device

__all__ = ["get_backend", "asnumpy", "to_device", "describe_device"]
