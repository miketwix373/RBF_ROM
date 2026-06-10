"""Field-touching diagnostics for the clustering stage of CHORD2.

Operationalises two signals of the clustering rubric
(`docs/notes/clustering-success-criterion.md`):

1. Indicator overlap (`cluster_indicator_overlap`): does the K-means
   partition align with a per-snapshot physical regime indicator? Uses
   the Stage 2 indicator dict supplied by `chord2.data.load_indicator`.
2. Residence vs integral timescale (`cluster_residence_vs_timescale`):
   does the cluster-label run-length sit within a small factor of
   `tau_int`, the dataset's integral timescale?

Two supporting per-cluster diagnostics share the module:

- `cluster_energy`: per-cluster `<u^2>` (scalar1d) or `<(u^2 + v^2)/2>`
  (vector2d_uv) averages, including a fluctuation form for scalar1d
  because the KS PDE only conserves the `k=0` mode.
- `cluster_incompressibility`: per-cluster `<|div u|>` and
  `<|div u|^2 / |grad u|^2>` on the vector2d_uv testbeds. KOL is fully
  spectral; RB switches to a 2nd-order centred FD stencil in `z`
  with one-sided 2nd-order stencils at the walls so the wall does not
  manufacture a spurious O(1) divergence under a circular FFT.

None of these diagnostics are paper-mandated. Colanera & Magri (2025)
ships only reconstruction error and `a(t)` traces as ROM diagnostics;
the rubric above is a CHORD2 addition. The signatures take the field
metadata explicitly (`components`, `component_size`, `metadata`) rather
than a `Dataset` so each function is pure-numpy and testable on
synthetic arrays.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from chord2 import clustering
from chord2.backend import asnumpy


def _nan_dict(keys):
    """Helper: build a NaN-populated dict so the don't-apply paths keep stable keys."""
    return {k: float("nan") for k in keys}


# ---------------------------------------------------------------------------
# 1. Per-cluster energy
# ---------------------------------------------------------------------------

def cluster_energy(U, labels, K, *, components, component_size):
    """Per-cluster snapshot-mean energy.

    CHORD2 addition; no paper analogue. Supports both 1D scalar fields and
    2D velocity-component pairs:

    - `("u","v")` (KOL20/42, RB): `E_k = (1/2) <u^2 + v^2>` spatially
      then averaged over `m in cluster k`. Uniform spatial mean over
      the flattened component arrays; this is exact for the KOL grids
      and an acceptable approximation for RB's uniform z-grid.
    - `("u",)` (KS, LOR96): reports per-cluster `<u^2>` *and*
      `<(u - u_bar)^2>` where `u_bar` is the global mean. The KS PDE
      only conserves the `k = 0` mode, so `u_bar` need not be zero and
      the fluctuation form is the physically relevant energy.
    - `("latent",)` (EKG): NaN-filled dict; no physical-energy
      interpretation on pre-reduced latents.

    Parameters
    ----------
    U : (M, N) array
    labels : (M,) int array
    K : int
    components : tuple of str
    component_size : int
        Length of one physical component in `U[m]`; for `("u","v")`
        the two halves are split at this index.

    Returns
    -------
    dict with keys (NaN-filled for `("latent",)`):
        global_E         float
        cluster_E        (K,) float array
        global_E_fluct   float            (NaN on `("u","v")`)
        cluster_E_fluct  (K,) float array (NaN on `("u","v")`)
        units            str
    """
    U = asnumpy(U).astype(np.float64, copy=False)
    labels = asnumpy(labels)
    keys = ("global_E", "cluster_E", "global_E_fluct",
            "cluster_E_fluct", "units")

    if components == ("latent",):
        out = {k: float("nan") for k in keys if k != "units"}
        out["cluster_E"] = np.full(K, np.nan)
        out["cluster_E_fluct"] = np.full(K, np.nan)
        out["units"] = "latent"
        return out

    if components == ("u", "v"):
        csz = int(component_size)
        u = U[:, :csz]
        v = U[:, csz:]
        e_snap = 0.5 * (u * u + v * v).mean(axis=1)
        global_E = float(e_snap.mean())
        cluster_E = np.full(K, np.nan)
        for k in range(K):
            mask = labels == k
            if int(mask.sum()) > 0:
                cluster_E[k] = float(e_snap[mask].mean())
        return {
            "global_E": global_E,
            "cluster_E": cluster_E,
            "global_E_fluct": float("nan"),
            "cluster_E_fluct": np.full(K, np.nan),
            "units": "<0.5(u^2+v^2)>",
        }

    if components == ("u",):
        e_snap = (U * U).mean(axis=1)
        u_bar = U.mean(axis=0, keepdims=True)
        fluct = ((U - u_bar) ** 2).mean(axis=1)
        cluster_E = np.full(K, np.nan)
        cluster_E_fluct = np.full(K, np.nan)
        for k in range(K):
            mask = labels == k
            if int(mask.sum()) > 0:
                cluster_E[k] = float(e_snap[mask].mean())
                cluster_E_fluct[k] = float(fluct[mask].mean())
        return {
            "global_E": float(e_snap.mean()),
            "cluster_E": cluster_E,
            "global_E_fluct": float(fluct.mean()),
            "cluster_E_fluct": cluster_E_fluct,
            "units": "<u^2>",
        }

    raise ValueError(f"unrecognised components {components!r}")


# ---------------------------------------------------------------------------
# 2. Per-cluster incompressibility
# ---------------------------------------------------------------------------

_INCOMPRESSIBILITY_KEYS = (
    "global_abs_div", "cluster_abs_div",
    "global_rel_div_sq", "cluster_rel_div_sq",
)


def _kol_divergence(u_field, v_field, dx, dy):
    """Spectral divergence on a fully-periodic KOL-style box.

    Returns `(div, grad_sq)` arrays of shape `(B, ny, nx)` where
    `grad_sq = (du/dx)^2 + (du/dy)^2 + (dv/dx)^2 + (dv/dy)^2`.
    """
    B, ny, nx = u_field.shape
    kx_vec = 2.0 * np.pi * np.fft.fftfreq(nx, dx)
    ky_vec = 2.0 * np.pi * np.fft.fftfreq(ny, dy)
    KX, KY = np.meshgrid(kx_vec, ky_vec, indexing="xy")
    Uh = np.fft.fft2(u_field)
    Vh = np.fft.fft2(v_field)
    dudx = np.real(np.fft.ifft2(1j * KX * Uh))
    dudy = np.real(np.fft.ifft2(1j * KY * Uh))
    dvdx = np.real(np.fft.ifft2(1j * KX * Vh))
    dvdy = np.real(np.fft.ifft2(1j * KY * Vh))
    div = dudx + dvdy
    grad_sq = dudx * dudx + dudy * dudy + dvdx * dvdx + dvdy * dvdy
    return div, grad_sq


def _rb_divergence(u_field, w_field, dx, dz):
    """Hybrid FFT-spectral-x / FD-2nd-order-z divergence for RB.

    A circular FFT in `z` would manufacture an O(1) spurious divergence
    at the wall on a wall-bounded field. The 2nd-order centred stencil
    in the interior plus 2nd-order one-sided stencils at z=0 and z=Nz-1
    is the FD-specialist's recommended dispatch.
    """
    B, nx, nz = u_field.shape
    kx_vec = 2.0 * np.pi * np.fft.fftfreq(nx, dx)
    KX = kx_vec[None, :, None]
    dudx = np.real(np.fft.ifft(1j * KX * np.fft.fft(u_field, axis=1), axis=1))
    dvdx = np.real(np.fft.ifft(1j * KX * np.fft.fft(w_field, axis=1), axis=1))

    dudz = np.empty_like(u_field)
    dwdz = np.empty_like(w_field)
    # interior: centred 2nd-order
    dudz[..., 1:-1] = (u_field[..., 2:] - u_field[..., :-2]) / (2.0 * dz)
    dwdz[..., 1:-1] = (w_field[..., 2:] - w_field[..., :-2]) / (2.0 * dz)
    # walls: one-sided 2nd-order, (-3 f0 + 4 f1 - f2) / (2 dz)
    dudz[..., 0] = (-3.0 * u_field[..., 0] + 4.0 * u_field[..., 1]
                    - u_field[..., 2]) / (2.0 * dz)
    dwdz[..., 0] = (-3.0 * w_field[..., 0] + 4.0 * w_field[..., 1]
                    - w_field[..., 2]) / (2.0 * dz)
    # (3 f_{n-1} - 4 f_{n-2} + f_{n-3}) / (2 dz)
    dudz[..., -1] = (3.0 * u_field[..., -1] - 4.0 * u_field[..., -2]
                     + u_field[..., -3]) / (2.0 * dz)
    dwdz[..., -1] = (3.0 * w_field[..., -1] - 4.0 * w_field[..., -2]
                     + w_field[..., -3]) / (2.0 * dz)

    div = dudx + dwdz
    grad_sq = dudx * dudx + dudz * dudz + dvdx * dvdx + dwdz * dwdz
    return div, grad_sq


def cluster_incompressibility(U, labels, K, *, components, component_size,
                              metadata):
    """Per-cluster `<|div u|>` and `<|div u|^2 / |grad u|^2>`.

    CHORD2 addition; no paper analogue. KOL is fully periodic so the
    divergence is computed by a 2D FFT; RB is periodic in `x` but
    wall-bounded in `z`, so the dispatch is FFT in x and 2nd-order FD in
    z with one-sided 2nd-order stencils at the walls. The `bc_type` /
    metadata-key check at function entry picks the branch.

    Only the vector2d_uv testbeds get a non-trivial value; scalar1d and
    latent return NaN for all keys.

    Returns
    -------
    dict with keys
        global_abs_div         float
        cluster_abs_div        (K,) float array
        global_rel_div_sq      float
        cluster_rel_div_sq     (K,) float array
    """
    if components != ("u", "v"):
        return {
            "global_abs_div": float("nan"),
            "cluster_abs_div": np.full(K, np.nan),
            "global_rel_div_sq": float("nan"),
            "cluster_rel_div_sq": np.full(K, np.nan),
        }

    U = asnumpy(U).astype(np.float64, copy=False)
    labels = asnumpy(labels)
    csz = int(component_size)
    M = U.shape[0]

    rb_path = (metadata.get("bc_type") == "wall_z") or ("Lz" in metadata)

    if rb_path:
        Nx = int(metadata["Nx"])
        Nz = int(metadata["Nz"])
        Lx = float(metadata["Lx"])
        Lz = float(metadata["Lz"])
        dx = Lx / Nx
        dz = Lz / (Nz - 1)
        a_field = U[:, :csz].reshape(M, Nx, Nz)
        b_field = U[:, csz:].reshape(M, Nx, Nz)
    else:
        nx = int(metadata["nx"])
        ny = int(metadata["ny"])
        dx = float(metadata["dx"])
        dy = float(metadata["dy"])
        a_field = U[:, :csz].reshape(M, ny, nx)
        b_field = U[:, csz:].reshape(M, ny, nx)

    abs_div = np.empty(M)
    rel_div_sq = np.empty(M)
    block = 2048
    for b0 in range(0, M, block):
        b1 = min(M, b0 + block)
        if rb_path:
            div, grad_sq = _rb_divergence(a_field[b0:b1], b_field[b0:b1], dx, dz)
        else:
            div, grad_sq = _kol_divergence(a_field[b0:b1], b_field[b0:b1], dx, dy)
        abs_div[b0:b1] = np.abs(div).mean(axis=(1, 2))
        div_sq = (div * div).mean(axis=(1, 2))
        grad_sq_mean = grad_sq.mean(axis=(1, 2))
        rel_div_sq[b0:b1] = np.where(grad_sq_mean > 0.0,
                                     div_sq / np.maximum(grad_sq_mean, 1e-300),
                                     0.0)

    cluster_abs_div = np.full(K, np.nan)
    cluster_rel_div_sq = np.full(K, np.nan)
    for k in range(K):
        mask = labels == k
        if int(mask.sum()) > 0:
            cluster_abs_div[k] = float(abs_div[mask].mean())
            cluster_rel_div_sq[k] = float(rel_div_sq[mask].mean())

    return {
        "global_abs_div": float(abs_div.mean()),
        "cluster_abs_div": cluster_abs_div,
        "global_rel_div_sq": float(rel_div_sq.mean()),
        "cluster_rel_div_sq": cluster_rel_div_sq,
    }


# ---------------------------------------------------------------------------
# 3. Indicator overlap (rubric signal 1)
# ---------------------------------------------------------------------------

_OVERLAP_1D_KEYS = ("kind", "confusion", "ami", "regime_below_threshold")
_OVERLAP_2D_KEYS = ("kind", "mean_I_minus_D", "mean_D", "burst_fraction")
_OVERLAP_NONE_KEYS = ("kind",)


def cluster_indicator_overlap(indicator_dict, labels, K):
    """Operationalises signal 1 of the rubric.

    CHORD2 addition; no paper analogue. Dispatch on the shape of the
    indicator dict's `values` field:

    - 1D `values` + scalar `threshold` (KS_bursting):
      binary regime = `values < threshold` ("quiescent" = True,
      "burst" = False). Reports the `(K, 2)` confusion matrix and the
      AMI between the K-cluster prediction and the 2-regime ground
      truth. AMI (not precision/recall) because precision/recall
      presupposes a 1-to-1 cluster->regime mapping that breaks down at
      `K > 2`. The FD-specialist's explicit recommendation.
      Returned-dict keys: `kind, confusion, ami,
      regime_below_threshold`.

    - 1D integer `values` with `threshold = None` (KS_chaotic
      `pulse_count`): the indicator is already a categorical partition
      (configuration class). AMI between cluster labels and the integer
      classes (re-indexed to consecutive 0..n-1) measures partition
      agreement directly; the `(K, n_classes)` confusion matrix is
      reported for the journal.
      Returned-dict keys: `kind, confusion, ami, class_offset,
      n_classes`.

    - 2D `values` with `threshold = None` (KOL42 `(I, D)`):
      per-cluster `<I - D>` (signed; < 0 -> off-diagonal dissipation
      lobe), `<D>`, and a one-sided burst-fraction
      `Pr[D > <D>_global + sigma_D | cluster = k]`. No scalar collapse.
      Returned-dict keys: `kind, mean_I_minus_D, mean_D,
      burst_fraction`.

    - `values = None` (KOL20, KOL42 with no indicator, RB, LOR96):
      `kind = "none"`. The Stage 4 sweep treats this as "no signal 1
      diagnostic to write" and stores `kind` only.

    The function does not recompute `u_bar` or any indicator internals;
    the indicator dict is consumed as supplied. The threshold provenance
    (e.g. KS_bursting theta = 7.13 calibrated on the full intro dataset)
    is the indicator module's responsibility.
    """
    values = indicator_dict.get("values")
    threshold = indicator_dict.get("threshold")

    if values is None:
        return {"kind": "none"}

    labels = asnumpy(labels)
    values = np.asarray(values)

    if values.ndim == 1 and threshold is not None:
        below = values < float(threshold)
        # confusion[k, 0] = # snapshots in cluster k with below = True
        confusion = np.zeros((K, 2), dtype=np.int64)
        for k in range(K):
            mask = labels == k
            confusion[k, 0] = int(np.sum(mask & below))
            confusion[k, 1] = int(np.sum(mask & ~below))
        regime = (~below).astype(np.int32)  # 0 = quiescent, 1 = burst
        ami = clustering.adjusted_mutual_information(labels, regime)
        return {
            "kind": "1d_threshold",
            "confusion": confusion,
            "ami": float(ami),
            "regime_below_threshold": int(below.sum()),
        }

    if (values.ndim == 1 and threshold is None
            and np.issubdtype(values.dtype, np.integer)):
        classes = values.astype(np.int64)
        cls_min = int(classes.min())
        classes = classes - cls_min
        n_classes = int(classes.max()) + 1
        confusion = np.zeros((K, n_classes), dtype=np.int64)
        for k in range(K):
            mask = labels == k
            cls_k = classes[mask]
            if cls_k.size > 0:
                confusion[k, :] = np.bincount(cls_k, minlength=n_classes)
        ami = clustering.adjusted_mutual_information(labels, classes)
        return {
            "kind": "1d_categorical",
            "confusion": confusion,
            "ami": float(ami),
            "class_offset": cls_min,
            "n_classes": n_classes,
        }

    if values.ndim == 2 and values.shape[1] == 2 and threshold is None:
        I = values[:, 0].astype(np.float64)
        D = values[:, 1].astype(np.float64)
        diff = I - D
        D_mean = float(D.mean())
        D_std = float(D.std())
        burst_mask = D > (D_mean + D_std)
        mean_I_minus_D = np.full(K, np.nan)
        mean_D = np.full(K, np.nan)
        burst_fraction = np.full(K, np.nan)
        for k in range(K):
            mask = labels == k
            n_k = int(mask.sum())
            if n_k > 0:
                mean_I_minus_D[k] = float(diff[mask].mean())
                mean_D[k] = float(D[mask].mean())
                burst_fraction[k] = float(burst_mask[mask].mean())
        return {
            "kind": "2d_power_balance",
            "mean_I_minus_D": mean_I_minus_D,
            "mean_D": mean_D,
            "burst_fraction": burst_fraction,
        }

    raise ValueError(
        f"unrecognised indicator schema: values.shape="
        f"{None if values is None else values.shape}, threshold={threshold!r}"
    )


# ---------------------------------------------------------------------------
# 4. Cluster residence time vs integral timescale (rubric signal 2)
# ---------------------------------------------------------------------------

def cluster_residence_vs_timescale(residence_stats, K, dt_sample, tau_int):
    """Translate run-length statistics into rubric signal 2.

    CHORD2 addition; no paper analogue.

    The rubric's signal-2 ratio is
        r1 = residence_phys_mean / tau_int
    A regime that earns the name has `r1` of order one (loosely
    `[0.3, 3.0]`).

    The ROM-specialist's reframing adds a second ratio:
        r2 = K * residence_phys_mean / tau_int
    On `K` angular sectors of a smooth ring traversed at uniform angular
    rate, `K * mean_residence ~ tau_int`, so `r2 ~ 1` is geometric
    fragmentation (KOL20-like). `r1 ~ 1` is regime separation (KOL42-like).
    Reporting both separates the two cases.

    A heavy-tail signature
        mean_over_median = mean / max(median, 1)
    > 3 flags an exponential-or-heavier dwell-time distribution.

    `verdict`:
    - "degenerate" if the *median* residence is one sample and the
      dynamics is significantly over-resolved (`tau_int / dt_sample > 2`);
      otherwise the threshold check below.
    - "regime" if `median_phys / tau_int in [0.3, 3.0]`.
    - "fragmentation" otherwise.

    `tau_int` is in the same physical-time units as `dt_sample`. When
    multiple `tau_int` estimates are available (e.g. KOL42 31.3-63.6,
    RB 0.9-1.8), the caller should pass the *largest* per the
    FD-specialist's brief: that is the slowest physics and the one the
    clustering must be slower than to plausibly track regimes. The
    function does not pick.

    Parameters
    ----------
    residence_stats : dict
        `clustering.residence_time_stats(labels)` output; the run-length
        units there are *sample steps*.
    K : int
    dt_sample : float
        Physical time between consecutive snapshots in the loaded array,
        i.e. `t[1] - t[0]` post-stride. Not `metadata["dtStats"]` (which
        is pre-stride).
    tau_int : float
        Integral timescale of the dynamics, in physical-time units.

    Returns
    -------
    dict with keys
        r1, r2                          float
        mean_over_median                float
        residence_phys_mean             float
        residence_phys_median           float
        verdict                         str
    """
    mean_steps = float(residence_stats["mean"])
    median_steps = float(residence_stats["median"])

    residence_phys_mean = mean_steps * float(dt_sample)
    residence_phys_median = median_steps * float(dt_sample)

    tau = float(tau_int)
    if tau > 0.0:
        r1 = residence_phys_mean / tau
        r2 = float(K) * residence_phys_mean / tau
        ratio_median = residence_phys_median / tau
    else:
        r1 = float("nan")
        r2 = float("nan")
        ratio_median = float("nan")

    mean_over_median = mean_steps / max(median_steps, 1.0)

    if median_steps <= 1.0 and (tau / float(dt_sample) > 2.0):
        verdict = "degenerate"
    elif np.isnan(ratio_median):
        verdict = "fragmentation"
    elif 0.3 <= ratio_median <= 3.0:
        verdict = "regime"
    else:
        verdict = "fragmentation"

    return {
        "r1": float(r1),
        "r2": float(r2),
        "mean_over_median": float(mean_over_median),
        "residence_phys_mean": float(residence_phys_mean),
        "residence_phys_median": float(residence_phys_median),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# 5. Phase 0 SINDy-fit diagnostics
# ---------------------------------------------------------------------------
#
# Diagnostics specific to the Phase 0 existence gate, locked in
# `docs/journal/2026-06-07-phase-0-rbf-design.md`.
# - L63 (negative control): four-condition acceptance rule, including the
#   unstable-eigenvalue counts at the three L63 fixed points (Sparrow 1982).
# - L96 X-only (existence gate): long-run boundedness, energy-PDF KS
#   distance against the FOM, closure-magnitude ratio.
# ---------------------------------------------------------------------------

# Polynomial column layout in `chord2.sindy.poly_feature_layout(r=3, p=2)`:
#   col 0: 1 (bias)
#   col 1: a_0   col 2: a_1   col 3: a_2
#   col 4: a_0*a_0   col 5: a_0*a_1   col 6: a_0*a_2
#   col 7: a_1*a_1   col 8: a_1*a_2   col 9: a_2*a_2
_L63_TRUTH_INDICES = {
    "sigma_dx_a0": (1, 0),   # dx/dt coeff of a_0 = -sigma
    "sigma_dx_a1": (2, 0),   # dx/dt coeff of a_1 = +sigma
    "rho_dy_a0":   (1, 1),   # dy/dt coeff of a_0 = +rho
    "neg1_dy_a1":  (2, 1),   # dy/dt coeff of a_1 = -1
    "xz_dy":       (6, 1),   # dy/dt coeff of a_0*a_2 = -1
    "beta_dz_a2":  (3, 2),   # dz/dt coeff of a_2 = -beta
    "xy_dz":       (5, 2),   # dz/dt coeff of a_0*a_1 = +1
}


def l63_truth_coefficients(sigma: float, rho: float, beta: float) -> dict:
    """Truth values of the seven non-zero Lorenz-63 polynomial coefficients."""
    return {
        "sigma_dx_a0": -float(sigma),
        "sigma_dx_a1": +float(sigma),
        "rho_dy_a0":   +float(rho),
        "neg1_dy_a1":  -1.0,
        "xz_dy":       -1.0,
        "beta_dz_a2":  -float(beta),
        "xy_dz":       +1.0,
    }


def l63_fixed_points(sigma: float, rho: float, beta: float):
    """The three Lorenz-63 fixed points at supercritical rho.

    Returns a dict {origin, C_plus, C_minus} of (3,) arrays. For rho > 1
    the non-trivial fixed points sit at
        C_pm = (±sqrt(beta (rho - 1)), ±sqrt(beta (rho - 1)), rho - 1).
    """
    s = float(beta) * (float(rho) - 1.0)
    if s <= 0:
        raise ValueError(f"non-trivial fixed points need beta(rho-1) > 0; got {s}")
    a = float(np.sqrt(s))
    return {
        "origin": np.zeros(3),
        "C_plus": np.array([+a, +a, float(rho) - 1.0]),
        "C_minus": np.array([-a, -a, float(rho) - 1.0]),
    }


def numerical_jacobian(predict_one, a: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Central-difference Jacobian of `predict_one` at `a`."""
    r = a.shape[0]
    J = np.empty((r, r))
    for i in range(r):
        e = np.zeros(r); e[i] = eps
        J[:, i] = (predict_one(a + e) - predict_one(a - e)) / (2.0 * eps)
    return J


def l63_acceptance(model, *, sigma: float = 10.0, rho: float = 28.0,
                   beta: float = 8.0 / 3.0,
                   coeff_rel_tol: float = 5e-2,
                   ratio_tol: float = 1e-3,
                   nnz_rbf_max: int = 2) -> dict:
    """Apply the L63 four-condition acceptance rule.

    Conditions (locked 2026-06-07 in `docs/journal/2026-06-07-phase-0-rbf-design.md`):
    1. `||xi_rbf||_F / ||xi_poly||_F < ratio_tol`.
    2. `nnz(xi_rbf) <= nnz_rbf_max`.
    3. All seven non-zero Lorenz coefficients recovered to relative error
       `< coeff_rel_tol` (default 5e-2).
    4. Jacobian unstable-eigenvalue count matches Sparrow 1982 at every
       L63 fixed point: origin -> 1, C_pm -> 2.

    A failure of (1) but pass of (3) flags a broken moment-orthogonalisation
    or constraint scaling; this is reported in `notes`.
    """
    expected_unstable = {"origin": 1, "C_plus": 2, "C_minus": 2}

    xi_poly_norm = float(np.linalg.norm(model.xi_poly))
    xi_rbf_norm = float(np.linalg.norm(model.xi_rbf))
    ratio = xi_rbf_norm / max(xi_poly_norm, 1e-300)
    nnz_rbf = int(np.sum(np.max(np.abs(model.xi_rbf), axis=1) > 1e-12))

    truth = l63_truth_coefficients(sigma, rho, beta)
    coeffs = {}
    for name, (j, i) in _L63_TRUTH_INDICES.items():
        fit = float(model.xi_poly[j, i])
        true_val = float(truth[name])
        coeffs[name] = {
            "fit": fit,
            "truth": true_val,
            "rel_err": abs(fit - true_val) / max(abs(true_val), 1e-300),
        }
    max_coeff_rel_err = max(c["rel_err"] for c in coeffs.values())

    fixed_points = l63_fixed_points(sigma, rho, beta)
    jac_info = {}
    unstable_match = True
    for name, fp in fixed_points.items():
        J = numerical_jacobian(model.predict_one, fp)
        eigs = np.linalg.eigvals(J)
        n_unstable = int(np.sum(eigs.real > 0.0))
        jac_info[name] = {
            "fixed_point": fp.tolist(),
            "eigenvalues": [complex(e) for e in eigs],
            "n_unstable": n_unstable,
            "expected_unstable": expected_unstable[name],
            "match": n_unstable == expected_unstable[name],
        }
        if not jac_info[name]["match"]:
            unstable_match = False

    cond1 = ratio < ratio_tol
    cond2 = nnz_rbf <= nnz_rbf_max
    cond3 = max_coeff_rel_err < coeff_rel_tol
    cond4 = unstable_match

    notes = []
    if (not cond1) and cond3:
        notes.append(
            "ratio (cond 1) fails but coefficients (cond 3) pass: "
            "likely a broken moment-orthogonalisation or constraint scaling."
        )

    return {
        "pass": bool(cond1 and cond2 and cond3 and cond4),
        "cond1_ratio": {"value": ratio, "tol": ratio_tol, "pass": cond1},
        "cond2_nnz_rbf": {"value": nnz_rbf, "max": nnz_rbf_max, "pass": cond2},
        "cond3_coeffs": {
            "max_rel_err": max_coeff_rel_err,
            "tol": coeff_rel_tol,
            "pass": cond3,
            "per_coeff": coeffs,
        },
        "cond4_jacobians": {"pass": cond4, "per_point": jac_info},
        "xi_poly_norm": xi_poly_norm,
        "xi_rbf_norm": xi_rbf_norm,
        "notes": notes,
    }


def long_run_boundedness(model, a0: np.ndarray, dt: float, T: float,
                         hull_radius: float | None = None,
                         hull_centroid: np.ndarray | None = None) -> dict:
    """Integrate the model and check long-run boundedness.

    The L96 gate requires the quadratic-only fit, integrated over
    `>= 100 * T_Lyap`, to remain bounded inside the training hull. We
    operationalise "inside the training hull" by an L2 ball around the
    training centroid with radius `hull_radius`; the caller derives both
    from the training trajectory `A_train`:
        hull_centroid = A_train.mean(axis=0)
        hull_radius   = (multiplier) * max(||A_train[m] - centroid||).
    The multiplier (e.g. 1.5) is the slack on what "stays inside" means.

    Reports
    -------
    finite        : trajectory never blew up to NaN or Inf.
    bounded       : every snapshot stays within `hull_radius` of `hull_centroid`.
    exit_index    : first index leaving the hull (or n_steps if never).
    max_norm      : max(||A[k]||) over the run.
    n_steps_run   : actual number of integration steps that completed.
    """
    from chord2.sindy import integrate_rk4

    t, A = integrate_rk4(model, a0, dt, T)
    n_steps_run = A.shape[0]
    finite = bool(np.all(np.isfinite(A)))

    if not finite:
        return {
            "finite": False,
            "bounded": False,
            "exit_index": int(np.argmax(~np.isfinite(A).any(axis=1))),
            "max_norm": float("inf"),
            "n_steps_run": n_steps_run,
        }

    norms = np.linalg.norm(A, axis=1)
    if hull_radius is not None:
        c = hull_centroid if hull_centroid is not None else np.zeros(A.shape[1])
        dist = np.linalg.norm(A - c[None, :], axis=1)
        outside = dist > hull_radius
        if np.any(outside):
            exit_idx = int(np.argmax(outside))
            bounded = False
        else:
            exit_idx = n_steps_run
            bounded = True
    else:
        exit_idx = n_steps_run
        bounded = finite

    return {
        "finite": finite,
        "bounded": bool(bounded),
        "exit_index": int(exit_idx),
        "max_norm": float(norms.max()),
        "n_steps_run": int(n_steps_run),
    }


def energy_pdf_ks(A_truth: np.ndarray, A_pred: np.ndarray) -> dict:
    """KS distance between the per-snapshot energy PDFs of two trajectories.

    The L96 gate requires the ROM to reproduce the FOM energy PDF on X
    within tolerance. Energy is `<X^2>` per snapshot (spatial mean of the
    squared resolved coordinates). KS statistic is the sup-distance between
    the two empirical CDFs (Smirnov 1939; no scipy dependency).

    Returns
    -------
    ks               : float, sup |F_truth(e) - F_pred(e)|.
    n_truth, n_pred  : sample sizes.
    e_truth_mean, e_pred_mean : means of the two energy samples.
    """
    e_truth = np.sort((A_truth * A_truth).mean(axis=1))
    e_pred = np.sort((A_pred * A_pred).mean(axis=1))
    n_t, n_p = e_truth.size, e_pred.size

    all_e = np.concatenate([e_truth, e_pred])
    f_t = np.searchsorted(e_truth, all_e, side="right") / n_t
    f_p = np.searchsorted(e_pred, all_e, side="right") / n_p
    ks = float(np.max(np.abs(f_t - f_p)))

    return {
        "ks": ks,
        "n_truth": int(n_t),
        "n_pred": int(n_p),
        "e_truth_mean": float(e_truth.mean()),
        "e_pred_mean": float(e_pred.mean()),
    }


def closure_magnitude_l96(Xs: np.ndarray, Ys: np.ndarray,
                          h: float, c: float, b: float) -> dict:
    """L96 closure magnitude `||(hc/b) sum_j Y|| / ||dX/dt||`.

    The fundamental question Phase 0 asks on L96 is whether the closure
    term `-(hc/b) sum_j Y_{j,k}` is large enough that the X-only ROM is
    unable to reproduce the FOM without absorbing it into the RBF block.
    A small ratio means the closure is dynamically irrelevant and the
    existence-gate result is meaningless; a ratio of order one means the
    closure dominates and a positive RBF result is informative.

    `Xs` and `Ys` are the full two-scale FOM trajectories at matching
    snapshot times: `Xs` is `(M, K)`, `Ys` is `(M, K, J)`. dX/dt is the
    full RHS including the closure; the closure is computed explicitly
    and compared to the un-closed L96 RHS.

    Returns
    -------
    closure_mean_norm : mean over m of ||closure_m||
    full_rhs_mean_norm : mean over m of ||dX/dt_m||
    ratio             : closure_mean_norm / full_rhs_mean_norm
    per_k             : (K,) array of per-mode closure RMS / per-mode dX/dt RMS
    """
    from data.LOR96.lor96 import lorenz96_rhs_vec

    M, K = Xs.shape
    hcb = float(h) * float(c) / float(b)
    closure = -hcb * Ys.sum(axis=-1)  # (M, K)

    F_eff = 0.0
    rhs = np.empty_like(Xs)
    for m in range(M):
        rhs[m] = lorenz96_rhs_vec(Xs[m], F=F_eff) + closure[m]

    closure_norms = np.linalg.norm(closure, axis=1)
    rhs_norms = np.linalg.norm(rhs, axis=1)

    closure_per_k = np.sqrt((closure * closure).mean(axis=0))
    rhs_per_k = np.sqrt((rhs * rhs).mean(axis=0))
    per_k = closure_per_k / np.maximum(rhs_per_k, 1e-300)

    return {
        "closure_mean_norm": float(closure_norms.mean()),
        "full_rhs_mean_norm": float(rhs_norms.mean()),
        "ratio": float(closure_norms.mean() / max(rhs_norms.mean(), 1e-300)),
        "per_k": per_k,
    }


def phase0_summary(model) -> dict:
    """Compact diagnostic summary of a fitted Phase 0 model.

    Reports norms, sparsity, and fit metadata in one dict for journaling.
    """
    xi_poly = model.xi_poly
    xi_rbf = model.xi_rbf
    fit = model.info.get("fit", {})
    return {
        "rbf_kind": model.rbf_kind,
        "n_rbf": int(xi_rbf.shape[0]),
        "xi_poly_norm": float(np.linalg.norm(xi_poly)),
        "xi_rbf_norm": float(np.linalg.norm(xi_rbf)),
        "ratio": float(np.linalg.norm(xi_rbf)
                       / max(np.linalg.norm(xi_poly), 1e-300)),
        "nnz_poly": int(np.sum(np.max(np.abs(xi_poly), axis=1) > 1e-12)),
        "nnz_rbf": int(np.sum(np.max(np.abs(xi_rbf), axis=1) > 1e-12)),
        "stlsq_iters": int(fit.get("n_iter", 0)),
        "stlsq_converged": bool(fit.get("converged", False)),
        "constraint_residual": float(fit.get("constraint_residual", float("nan"))),
        "lambda_poly": float(model.info.get("lambda_poly", float("nan"))),
        "lambda_rbf": float(model.info.get("lambda_rbf", float("nan"))),
    }
