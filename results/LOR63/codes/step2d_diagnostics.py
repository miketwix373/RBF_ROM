"""L63 step 2d: diagnostic post-mortems for Stage B failures.

Stage B established three failure modes for the clustered RBF ROM:
  1. K=4 collapses to a quasi-fixed point at (31, 9, 31).
  2. On-attractor PDF fidelity (F=1) degrades monotonically with K.
  3. Off-attractor basin shrinks to zero for K>=2.

Three candidate mechanisms:
  S1 -- shrunken in-cluster sigma_A -> faster Gaussian decay.
  S2 -- Voronoi-boundary discontinuities -> field jumps + dead zones.
  S3 -- per-cluster overfitting (small m_k, big n_rbf).

This script implements Test C (field error map) and Test D (xi norms +
conditioning) from the discussion. Both are diagnostic; neither modifies
the integrator or refits.

Test C  -- 2D slice through z = z_attr_mean. Compute || f_rom_K(a) -
           f_truth(a) || on a 200x200 grid in (x, y). Plot one panel per
           K in {1, 2, 3, 4}. Overlay:
             * hard cluster boundaries (decision regions)
             * RBF centres (per cluster)
             * training-data density (FOM snapshot scatter)
           Also a second figure showing log||f_rom_K|| alone (to see
           where the ROM thinks the "real" dynamics live).

Test D  -- For each (K, k) report:
             * m_k        training-cluster count from Stage A
             * n_rbf_k    RBFs allocated to cluster k
             * ||xi_k||   Frobenius norm of per-cluster coefficient
             * sigma_A_k  per-feature in-cluster std
             * width      mean Gaussian width
             * col_norm_ratio = max(col_norms)/min(col_norms)
           Plot ||xi_k|| / sqrt(n_rbf_k) vs m_k; small m_k with large
           normalised xi norm = overfit signal.
"""

from __future__ import annotations

import argparse
import json
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

from step2b_stability import load_cell  # noqa: E402
from chord2 import data  # noqa: E402

SIGMA_L63, RHO_L63, BETA_L63 = 10.0, 28.0, 8.0 / 3.0


# ---------------------------------------------------------------------------
# Vectorised field evals (over a batch of states)
# ---------------------------------------------------------------------------

def f_truth_grid(P: np.ndarray) -> np.ndarray:
    """Vectorised Lorenz-63 RHS on a (M, 3) batch."""
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    fx = SIGMA_L63 * (y - x)
    fy = x * (RHO_L63 - z) - y
    fz = x * y - BETA_L63 * z
    return np.column_stack([fx, fy, fz])


def assign_grid(P: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Hard nearest-centroid affiliation on a batch."""
    d2 = ((P[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(d2, axis=1)


def f_rbf_batch(P: np.ndarray, cl: dict) -> np.ndarray:
    """Evaluate one cluster's RBF field on a (M, 3) batch."""
    # diffs_std: (M, n_rbf, 3)
    diffs_std = (P[:, None, :] - cl["centers"][None, :, :]) / cl["sigma_A"][None, None, :]
    d2 = (diffs_std * diffs_std).sum(axis=2)  # (M, n_rbf)
    phi = np.exp(-0.5 * d2 / (cl["widths"][None, :] ** 2))
    phi_n = phi / cl["col_norms"][None, :]
    return phi_n @ cl["xi"]


def f_rom_grid(P: np.ndarray, cell: dict) -> tuple:
    """Cluster-piecewise field. Returns (F, labels)."""
    labels = assign_grid(P, cell["centroids"])
    F = np.zeros_like(P)
    for k in range(cell["K"]):
        mask = labels == k
        if not np.any(mask):
            continue
        cl = cell["clusters"][k]
        if cl is None:
            continue
        F[mask] = f_rbf_batch(P[mask], cl)
    return F, labels


# ---------------------------------------------------------------------------
# Test C -- 2D field error map
# ---------------------------------------------------------------------------

def run_test_C(cells_by_K: dict, k_grid: list, ds_u: np.ndarray,
               out_dir: Path, n_grid: int = 200) -> None:
    """2D slice z = mean(snapshot z). Plot log||f_rom - f_truth|| per K."""
    z_slice = float(ds_u[:, 2].mean())
    xs = np.linspace(-25.0, 25.0, n_grid)
    ys = np.linspace(-30.0, 30.0, n_grid)
    Xg, Yg = np.meshgrid(xs, ys, indexing="xy")
    P = np.column_stack([Xg.ravel(), Yg.ravel(),
                         z_slice * np.ones(Xg.size)])
    F_truth = f_truth_grid(P)
    nrm_truth = np.linalg.norm(F_truth, axis=1).reshape(Xg.shape)

    # Pre-compute fields per K
    F_rom = {}
    labels = {}
    for K in k_grid:
        cell = cells_by_K[K]
        F_K, lab_K = f_rom_grid(P, cell)
        F_rom[K] = F_K
        labels[K] = lab_K

    # Errors, magnitudes
    err = {K: np.linalg.norm(F_rom[K] - F_truth, axis=1).reshape(Xg.shape)
           for K in k_grid}
    rom_mag = {K: np.linalg.norm(F_rom[K], axis=1).reshape(Xg.shape)
               for K in k_grid}

    # FOM snapshots whose z is near the slice -- visualises training density
    z_band = (np.abs(ds_u[:, 2] - z_slice) < 1.5)
    samples_in_band = ds_u[z_band]

    # --- Figure 1: error map ---
    fig, axs = plt.subplots(2, 2, figsize=(13, 11))
    vmax = max(np.nanpercentile(err[K], 99.0) for K in k_grid)
    for ax, K in zip(axs.ravel(), k_grid):
        cell = cells_by_K[K]
        # Underflow / catastrophic regions -> set to vmax for visibility
        E = np.minimum(err[K], vmax)
        im = ax.pcolormesh(Xg, Yg, np.log10(E + 1e-3),
                           shading="auto", cmap="viridis",
                           vmin=-1, vmax=np.log10(vmax))
        # Boundaries: K-means decision regions (categorical contour)
        if K > 1:
            ax.contour(Xg, Yg, labels[K].reshape(Xg.shape),
                       levels=np.arange(K) + 0.5, colors="white", lw=0.8)
        # Cluster centroids
        ax.scatter(cell["centroids"][:, 0], cell["centroids"][:, 1],
                   marker="*", c="white", edgecolor="black",
                   s=180, lw=1.0, zorder=5, label="centroid")
        # Sub-sample of training snapshots within band
        sub = samples_in_band[::max(1, samples_in_band.shape[0] // 800)]
        ax.scatter(sub[:, 0], sub[:, 1], c="red", s=2, alpha=0.4,
                   label="FOM snapshots near slice")
        ax.set_title(f"K={K}: log$_{{10}}$ $\\|f_K - f_{{\\rm truth}}\\|$ "
                     f"@ z={z_slice:.1f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, label="log$_{10}$ error")
        if K == 1:
            ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(
        f"Test C: 2D field error vs FOM truth at z={z_slice:.1f}  "
        f"(white contour = Voronoi boundary; stars = centroids)"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "testC_error_map.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'testC_error_map.png'}")

    # --- Figure 2: ROM field magnitude (where the ROM thinks dynamics live) ---
    fig, axs = plt.subplots(2, 2, figsize=(13, 11))
    vmin_mag, vmax_mag = -3, np.log10(np.nanpercentile(nrm_truth, 99.0))
    for ax, K in zip(axs.ravel(), k_grid):
        cell = cells_by_K[K]
        M = np.log10(rom_mag[K] + 1e-3)
        im = ax.pcolormesh(Xg, Yg, M, shading="auto",
                           cmap="magma", vmin=vmin_mag, vmax=vmax_mag)
        if K > 1:
            ax.contour(Xg, Yg, labels[K].reshape(Xg.shape),
                       levels=np.arange(K) + 0.5, colors="cyan", lw=0.8)
        ax.scatter(cell["centroids"][:, 0], cell["centroids"][:, 1],
                   marker="*", c="white", edgecolor="black",
                   s=180, lw=1.0, zorder=5)
        # Mark per-cluster RBF centres (subsampled)
        for k in range(cell["K"]):
            cl = cell["clusters"][k]
            if cl is None:
                continue
            C = cl["centers"]
            stride = max(1, C.shape[0] // 60)
            ax.scatter(C[::stride, 0], C[::stride, 1], c="lime", s=4,
                       alpha=0.6, edgecolor="none")
        ax.set_title(f"K={K}: log$_{{10}} \\|f_K\\|$ @ z={z_slice:.1f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, label="log$_{10}$ ||f_K||")
    fig.suptitle(
        f"Test C (companion): ROM field magnitude  "
        f"(green dots = RBF centres; cyan = Voronoi boundary)"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "testC_rom_magnitude.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'testC_rom_magnitude.png'}")

    # --- Figure 3: error stratified by inside/outside the training band ---
    in_band_grid = np.zeros(Xg.shape, dtype=bool)
    for s in samples_in_band:
        in_band_grid |= ((Xg - s[0]) ** 2 + (Yg - s[1]) ** 2) < 4.0  # 2-unit radius
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.0))
    err_inside, err_outside = [], []
    for K in k_grid:
        E = err[K]
        err_inside.append(np.median(E[in_band_grid]))
        err_outside.append(np.median(E[~in_band_grid]))
    axs[0].plot(k_grid, err_inside, "o-", c="C0", label="near training")
    axs[0].plot(k_grid, err_outside, "s-", c="C3", label="off training")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("K")
    axs[0].set_ylabel("median ||f_K - f_truth||")
    axs[0].set_title("Field error: near-training vs off-training (z slice)")
    axs[0].grid(True, ls=":", alpha=0.4, which="both")
    axs[0].legend()

    # Also: fraction of grid where ||f_K|| < 1e-3 (dead-zone area)
    dead_frac = []
    for K in k_grid:
        dead_frac.append(np.mean(rom_mag[K] < 1e-3))
    axs[1].plot(k_grid, dead_frac, "o-", c="C2")
    axs[1].set_xlabel("K")
    axs[1].set_ylabel("fraction of slice with ||f_K|| < 1e-3")
    axs[1].set_title("Dead-zone fraction (where RBFs have underflowed)")
    axs[1].grid(True, ls=":", alpha=0.4, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "testC_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'testC_summary.png'}")

    # JSON summary
    return {
        "z_slice": z_slice,
        "n_grid": n_grid,
        "median_err_near_training": dict(zip(map(int, k_grid),
                                             [float(v) for v in err_inside])),
        "median_err_off_training": dict(zip(map(int, k_grid),
                                            [float(v) for v in err_outside])),
        "dead_zone_fraction": dict(zip(map(int, k_grid),
                                       [float(v) for v in dead_frac])),
    }


# ---------------------------------------------------------------------------
# Test D -- per-cluster xi norms + conditioning
# ---------------------------------------------------------------------------

def run_test_D(cells_by_K: dict, k_grid: list, out_dir: Path) -> dict:
    """Tabulate per-cluster diagnostics; plot ||xi_k|| / sqrt(n_rbf) vs m_k."""
    rows = []
    for K in k_grid:
        cell = cells_by_K[K]
        # counts come from Stage A; we stored them
        path = cell.get("__path", None)
        for k in range(cell["K"]):
            cl = cell["clusters"][k]
            if cl is None:
                continue
            n_rbf = int(cl["xi"].shape[0])
            xi_F = float(np.linalg.norm(cl["xi"]))  # Frobenius
            xi_F_norm = xi_F / np.sqrt(n_rbf)
            width_mean = float(cl["widths"].mean())
            width_med = float(np.median(cl["widths"]))
            col_ratio = float(cl["col_norms"].max() / max(cl["col_norms"].min(), 1e-30))
            rows.append({
                "K": int(K), "k": int(k), "n_rbf": n_rbf,
                "sigma_A": cl["sigma_A"].tolist(),
                "sigma_A_min": float(cl["sigma_A"].min()),
                "xi_frobenius": xi_F,
                "xi_per_rbf": xi_F_norm,
                "width_mean": width_mean,
                "width_median": width_med,
                "col_norm_ratio": col_ratio,
            })

    # Cluster counts: need to come from Stage A counts stored in npz
    cell_path_root = data.results_dir("LOR63") / "step2a_cluster_rbf" / "models"
    counts_by_K = {}
    for K in k_grid:
        alloc = "count" if K == 1 else "variance"
        d = np.load(cell_path_root / f"cell_K{K}_N01280_{alloc}.npz",
                    allow_pickle=True)
        counts_by_K[K] = np.asarray(d["counts"])
        d.close()
    for r in rows:
        r["m_k"] = int(counts_by_K[r["K"]][r["k"]])

    print("Per-cluster Test D table (K, k, m_k, n_rbf, ||xi||, "
          "||xi||/sqrt(n), sigma_A_min, width_mean, col_ratio):")
    for r in rows:
        print(f"  K={r['K']} k={r['k']}: m={r['m_k']:>5d}  n_rbf={r['n_rbf']:>4d}  "
              f"||xi||={r['xi_frobenius']:8.3f}  ||xi||/sqrt(n)={r['xi_per_rbf']:6.3f}  "
              f"sigA_min={r['sigma_A_min']:5.2f}  width_mean={r['width_mean']:.2f}  "
              f"col_ratio={r['col_norm_ratio']:.2e}")

    # --- Plots ---
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = {1: "C0", 2: "C1", 3: "C2", 4: "C3"}
    for K in k_grid:
        rs = [r for r in rows if r["K"] == K]
        m_ks = [r["m_k"] for r in rs]
        xi_norms = [r["xi_per_rbf"] for r in rs]
        sig_mins = [r["sigma_A_min"] for r in rs]
        widths = [r["width_mean"] for r in rs]
        axs[0].scatter(m_ks, xi_norms, c=colors[K], s=70, label=f"K={K}",
                       edgecolor="black")
        axs[1].scatter(m_ks, sig_mins, c=colors[K], s=70, label=f"K={K}",
                       edgecolor="black")
        axs[2].scatter(m_ks, widths, c=colors[K], s=70, label=f"K={K}",
                       edgecolor="black")
    for ax, ylabel, title in zip(
        axs,
        [r"$\|\xi_k\|_F / \sqrt{n_{\rm rbf}}$",
         r"min component of $\sigma_A$",
         r"mean RBF width $\sigma_j$"],
        ["per-cluster fit magnitude (overfitting signal)",
         "in-cluster std (Gaussian decay rate)",
         "RBF width (Gaussian spread)"],
    ):
        ax.set_xlabel("cluster training count $m_k$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xscale("log")
        ax.grid(True, ls=":", alpha=0.4, which="both")
        ax.legend(fontsize=9)
    axs[0].set_yscale("log")
    axs[1].set_yscale("log")
    fig.suptitle("Test D: per-cluster Stage A diagnostics at N_tot=1280")
    fig.tight_layout()
    fig.savefig(out_dir / "testD_per_cluster.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'testD_per_cluster.png'}")

    return {"rows": rows,
            "counts_by_K": {int(k): v.tolist() for k, v in counts_by_K.items()}}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n-tot", type=int, default=1280)
    p.add_argument("--k-grid", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--n-grid", type=int, default=200)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    out_dir = args.out_dir or (data.results_dir("LOR63") / "step2d_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = data.load("LOR63")
    A = ds.u.astype(np.float64)
    print(f"Loaded LOR63: u shape {A.shape}")

    models_dir = data.results_dir("LOR63") / "step2a_cluster_rbf" / "models"
    cells_by_K = {}
    for K in args.k_grid:
        alloc = "count" if K == 1 else "variance"
        path = models_dir / f"cell_K{K}_N{args.n_tot:05d}_{alloc}.npz"
        cell = load_cell(path)
        cell["__path"] = str(path)
        cells_by_K[K] = cell
        print(f"  K={K}: {path.name}")

    t0 = time.time()
    print("\n== Test C: 2D field error map ==")
    sum_C = run_test_C(cells_by_K, args.k_grid, A, out_dir, n_grid=args.n_grid)
    print(f"  C done in {time.time() - t0:.1f}s")

    t1 = time.time()
    print("\n== Test D: per-cluster xi norms ==")
    sum_D = run_test_D(cells_by_K, args.k_grid, out_dir)
    print(f"  D done in {time.time() - t1:.1f}s")

    (out_dir / "step2d.json").write_text(json.dumps({
        "config": {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in vars(args).items()},
        "test_C": sum_C,
        "test_D": sum_D,
    }, indent=2))
    print(f"\nWrote {out_dir / 'step2d.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
