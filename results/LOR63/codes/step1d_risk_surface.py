"""Step 1d: off-cloud risk-surface diagnostics from saved models.

Loads each per-n_rbf model saved by step1c_save_models.py and computes
analytic diagnostics that flag regions of state space where the RBF
field is weak or pushes outward (i.e. the "off-cloud return" failure
mode that worries us).

Per-point diagnostics:
  ||f_rbf(x)||         (coverage / dead-zone detector)
  R(x) = r_hat . f_rbf (restoring projection; <0 = pull-back, >0 = push-out)
  cos theta vs truth   (direction agreement with the analytic L63 field)

Outputs:
  heatmaps_z*.png      2D slices (rows = n_rbf, cols = the metrics above)
  radial_profile.png   <R(x)>_solid_angle and dead-zone fraction vs shell
  summary.png          bar charts of R_esc, dead-zone vol, adverse fraction
  risk_surface.json    per-model scalars

R_esc per model = smallest shell radius (in units of X_std_climate) at
which <R(x)>_solid_angle >= 0, i.e. the field stops pulling inward.
Below R_esc the RBF is doing the right thing globally; above it,
off-cloud trajectories are no longer guaranteed to return.
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

from _l63_rbf_lib import (  # noqa: E402
    SIGMA_L63, RHO_L63, BETA_L63,
    f_rbf, f_truth_l63,
)
from chord2 import data  # noqa: E402


def fib_sphere(n: int) -> np.ndarray:
    """Fibonacci sphere: n approximately equidistributed unit vectors in R^3."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ], axis=1)


def load_model(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    meta = z["meta"].item()
    return {
        "centers": z["centers"], "widths": z["widths"],
        "mu_A": z["mu_A"], "sigma_A": z["sigma_A"],
        "col_norms": z["col_norms"], "xi": z["xi"],
        "active": z["active"], "meta": meta,
    }


def field_rbf(m: dict, X: np.ndarray) -> np.ndarray:
    return f_rbf(X, m["centers"], m["widths"],
                 m["mu_A"], m["sigma_A"],
                 m["col_norms"], m["xi"])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--models-dir", type=Path,
                   default=Path("results/LOR63/step1_nrbf_sweep/models"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("results/LOR63/step1_nrbf_sweep/risk_surface"))
    p.add_argument("--z-slice", type=float, nargs="+", default=[27.0],
                   help="z values at which to draw the 2D heatmap")
    p.add_argument("--grid-n", type=int, default=180,
                   help="grid resolution for the 2D slice")
    p.add_argument("--grid-half-width", type=float, default=30.0,
                   help="half-width of the (x, y) grid in absolute units")
    p.add_argument("--shell-r-grid", type=float, nargs="+",
                   default=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5,
                            2.0, 2.5, 3.0, 4.0, 5.0])
    p.add_argument("--n-sphere", type=int, default=300,
                   help="Fibonacci-sphere samples for solid-angle averages")
    p.add_argument("--dead-zone-thresh", type=float, default=1e-3,
                   help="||f_rbf|| < ||f_truth||_mean * this is 'dead'")
    args = p.parse_args()

    models_dir = args.models_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = models_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    n_rbf_list = manifest["config"]["n_rbf_grid"]
    print(f"Loading {len(n_rbf_list)} models from {models_dir}")
    print(f"n_rbf grid: {n_rbf_list}")

    models = []
    for entry in manifest["models"]:
        path = models_dir / entry["file"]
        m = load_model(path)
        m["n_rbf"] = entry["n_rbf"]
        m["rel_err_alpha_dot"] = entry["rel_err_alpha_dot"]
        models.append(m)
        print(f"  loaded n_rbf={entry['n_rbf']:>5d} from {path.name}  "
              f"(xi shape {m['xi'].shape})")

    ds = data.load("LOR63")
    A = ds.u.astype(np.float64)
    x_centroid = A.mean(axis=0)
    x_std = A.std(axis=0)
    sigma_attr = float(x_std.mean())
    print(f"\nAttractor centroid = {x_centroid}, std per axis = {x_std}, "
          f"sigma_attr = {sigma_attr:.4f}")

    # Truth reference: ||f_truth||_mean on the attractor
    f_truth_attr = f_truth_l63(A[::100])
    f_truth_mean_mag = float(np.linalg.norm(f_truth_attr, axis=1).mean())
    print(f"<||f_truth||> on attractor = {f_truth_mean_mag:.4f}")
    dz_eps = args.dead_zone_thresh * f_truth_mean_mag
    print(f"Dead-zone threshold ||f_rbf|| < {dz_eps:.4e}")

    # -- 2D slice heatmaps ----------------------------------------------------
    H = args.grid_half_width
    g = np.linspace(-H, H, args.grid_n)
    XX, YY = np.meshgrid(g, g, indexing="xy")
    grid_xy = np.stack([XX.ravel(), YY.ravel()], axis=1)

    for z_slice in args.z_slice:
        print(f"\nGenerating heatmap slice at z = {z_slice} ...")
        grid_xyz = np.concatenate(
            [grid_xy, np.full((grid_xy.shape[0], 1), z_slice)], axis=1)
        f_truth_grid = f_truth_l63(grid_xyz)
        truth_mag = np.linalg.norm(f_truth_grid, axis=1)

        rvec = grid_xyz - x_centroid
        rmag = np.linalg.norm(rvec, axis=1)
        rmag_safe = np.where(rmag > 0, rmag, 1.0)
        rhat = rvec / rmag_safe[:, None]

        n_rows = len(models)
        fig, axes = plt.subplots(n_rows, 4, figsize=(16, 3.0 * n_rows),
                                 squeeze=False)

        for i, m in enumerate(models):
            f_rbf_grid = field_rbf(m, grid_xyz)
            rbf_mag = np.linalg.norm(f_rbf_grid, axis=1)
            R_proj = np.einsum("ij,ij->i", f_rbf_grid, rhat)
            denom = rbf_mag * truth_mag
            denom = np.where(denom > 0, denom, 1.0)
            cos_theta = np.einsum("ij,ij->i", f_rbf_grid, f_truth_grid) / denom
            ratio = rbf_mag / np.where(truth_mag > 0, truth_mag, 1.0)

            mag2 = rbf_mag.reshape(args.grid_n, args.grid_n)
            R2 = R_proj.reshape(args.grid_n, args.grid_n)
            cos2 = cos_theta.reshape(args.grid_n, args.grid_n)
            ratio2 = ratio.reshape(args.grid_n, args.grid_n)

            ax = axes[i, 0]
            mag_floor = max(mag2.max() * 1e-6, 1e-12)
            im = ax.pcolormesh(XX, YY, np.maximum(mag2, mag_floor),
                                norm=matplotlib.colors.LogNorm(),
                                cmap="viridis", shading="auto")
            ax.set_title(rf"$\|f_{{\rm rbf}}\|$  (n_rbf={m['n_rbf']})",
                         fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.045)

            ax = axes[i, 1]
            R_lim = max(abs(R2.min()), abs(R2.max()), 1e-3)
            im = ax.pcolormesh(XX, YY, R2, cmap="RdBu_r",
                                vmin=-R_lim, vmax=R_lim, shading="auto")
            ax.set_title(r"$\hat r \cdot f_{\rm rbf}$  "
                         "(blue=pull-back, red=push-out)", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.045)

            ax = axes[i, 2]
            im = ax.pcolormesh(XX, YY, cos2, cmap="RdBu_r",
                                vmin=-1, vmax=1, shading="auto")
            ax.set_title(r"$\cos\theta$ vs truth", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.045)

            ax = axes[i, 3]
            ratio_floor = max(ratio2.max() * 1e-6, 1e-12)
            im = ax.pcolormesh(XX, YY, np.maximum(ratio2, ratio_floor),
                                norm=matplotlib.colors.LogNorm(),
                                cmap="magma", shading="auto")
            ax.set_title(r"$\|f_{\rm rbf}\|/\|f_{\rm truth}\|$", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.045)

            for k in range(4):
                axes[i, k].set_xlim(-H, H)
                axes[i, k].set_ylim(-H, H)
                axes[i, k].set_aspect("equal")
                # overlay near-slice attractor samples
                near = (np.abs(A[:, 2] - z_slice) < 1.0)
                axes[i, k].scatter(A[near, 0], A[near, 1],
                                   s=0.5, c="white", alpha=0.25)
                if k == 0:
                    axes[i, k].set_ylabel(rf"$n_{{\rm rbf}}={m['n_rbf']}$  "
                                          rf"$y$", fontsize=9)
                if i == n_rows - 1:
                    axes[i, k].set_xlabel(r"$x$")
                axes[i, k].tick_params(labelsize=7)

        fig.suptitle(f"L63 risk surface: 2D slice at z = {z_slice} "
                     "(attractor cloud overlaid in white)", y=1.005)
        fig.tight_layout()
        z_tag = f"{z_slice:.0f}".replace("-", "m").replace(".", "p")
        out_path = out_dir / f"heatmaps_z{z_tag}.png"
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out_path}")

    # -- Radial profile -------------------------------------------------------
    print("\nComputing radial profile ...")
    d_hat = fib_sphere(args.n_sphere)
    shell_R = np.asarray(args.shell_r_grid, dtype=float)

    radial_records = []
    truth_R_mean = []
    truth_R_inward_frac = []
    truth_mag_mean = []
    for R in shell_R:
        pts = x_centroid + R * sigma_attr * d_hat
        f_t = f_truth_l63(pts)
        rvec = pts - x_centroid
        rhat = rvec / np.linalg.norm(rvec, axis=1, keepdims=True)
        R_t = np.einsum("ij,ij->i", f_t, rhat)
        truth_R_mean.append(R_t.mean())
        truth_R_inward_frac.append(float((R_t < 0).mean()))
        truth_mag_mean.append(float(np.linalg.norm(f_t, axis=1).mean()))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
    cmap = plt.get_cmap("viridis")
    colours = [cmap(t) for t in np.linspace(0.05, 0.95, len(models))]

    axes[0].plot(shell_R, truth_R_mean, "k--", lw=1.6, label="truth")
    axes[1].plot(shell_R, [0.0] * len(shell_R), "k--", lw=0.8)
    axes[2].plot(shell_R, truth_mag_mean, "k--", lw=1.6, label="truth")

    for k, m in enumerate(models):
        R_mean = []
        R_inward_frac = []
        rbf_mag_mean = []
        dead_frac = []
        for R in shell_R:
            pts = x_centroid + R * sigma_attr * d_hat
            f_r = field_rbf(m, pts)
            rvec = pts - x_centroid
            rhat = rvec / np.linalg.norm(rvec, axis=1, keepdims=True)
            R_proj = np.einsum("ij,ij->i", f_r, rhat)
            R_mean.append(float(R_proj.mean()))
            R_inward_frac.append(float((R_proj < 0).mean()))
            mag = np.linalg.norm(f_r, axis=1)
            rbf_mag_mean.append(float(mag.mean()))
            dead_frac.append(float((mag < dz_eps).mean()))

        # R_esc: smallest shell at which <R(x)> >= 0
        R_mean_arr = np.array(R_mean)
        if (R_mean_arr < 0).all():
            R_esc = float("inf")
        elif (R_mean_arr >= 0).all():
            R_esc = float(shell_R[0])
        else:
            cross_idx = int(np.argmax(R_mean_arr >= 0))
            r0 = shell_R[cross_idx - 1]
            r1 = shell_R[cross_idx]
            y0 = R_mean_arr[cross_idx - 1]
            y1 = R_mean_arr[cross_idx]
            R_esc = float(r0 + (r1 - r0) * (-y0) / (y1 - y0))

        radial_records.append({
            "n_rbf": m["n_rbf"],
            "R_esc": R_esc,
            "R_mean": R_mean,
            "R_inward_frac": R_inward_frac,
            "rbf_mag_mean": rbf_mag_mean,
            "dead_frac": dead_frac,
            "shell_R": list(map(float, shell_R)),
        })

        label = rf"$n_{{\rm rbf}}={m['n_rbf']}$"
        axes[0].plot(shell_R, R_mean, color=colours[k], lw=1.5,
                     marker="o", ms=4, label=label)
        axes[1].plot(shell_R, dead_frac, color=colours[k], lw=1.5,
                     marker="s", ms=4, label=label)
        axes[2].semilogy(shell_R, np.maximum(rbf_mag_mean, 1e-12),
                         color=colours[k], lw=1.5, marker="^", ms=4,
                         label=label)

    for ax in axes:
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_xlabel(r"shell radius / $\sigma_{\rm attr}$")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    axes[0].set_ylabel(r"$\langle \hat r \cdot f \rangle$  "
                       "(inward when $<0$)")
    axes[0].set_title("Mean radial flow vs shell (truth = dashed)")
    axes[0].legend(fontsize=7, ncol=2)

    axes[1].set_ylabel("dead-zone fraction")
    axes[1].set_title(rf"$P(\|f_{{\rm rbf}}\| < $"
                      rf"$ {args.dead_zone_thresh:g}\cdot\|f_{{\rm truth}}\|)$")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend(fontsize=7, ncol=2)

    axes[2].set_ylabel(r"$\langle \|f\| \rangle$ on shell")
    axes[2].set_title("Field magnitude vs shell")
    axes[2].legend(fontsize=7, ncol=2)

    fig.suptitle("L63 risk surface: solid-angle averages on concentric shells",
                 y=1.02)
    fig.tight_layout()
    out_path = out_dir / "radial_profile.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")

    # -- Summary bar charts ---------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
    n_rbf_arr = np.array([r["n_rbf"] for r in radial_records])
    R_esc_arr = np.array([r["R_esc"] for r in radial_records])

    axes[0].bar(np.arange(len(n_rbf_arr)),
                np.where(np.isfinite(R_esc_arr), R_esc_arr, 0.0),
                color=colours)
    axes[0].set_xticks(np.arange(len(n_rbf_arr)))
    axes[0].set_xticklabels([str(n) for n in n_rbf_arr], rotation=0)
    axes[0].set_xlabel(r"$n_{\rm rbf}$")
    axes[0].set_ylabel(r"$R_{\rm esc} / \sigma_{\rm attr}$")
    axes[0].set_title(r"Escape radius (larger = safer)")
    axes[0].grid(True, axis="y", alpha=0.3)
    for k, R in enumerate(R_esc_arr):
        tag = "inf" if not np.isfinite(R) else f"{R:.2f}"
        axes[0].text(k, 0.05, tag, ha="center", va="bottom",
                     fontsize=8, color="white" if np.isfinite(R) else "black",
                     fontweight="bold")

    selected_R_idx = [int(np.argmin(np.abs(shell_R - r))) for r in (1.0, 2.0, 3.0)]
    width = 0.25
    for j, r_idx in enumerate(selected_R_idx):
        adverse = [1.0 - rec["R_inward_frac"][r_idx] for rec in radial_records]
        axes[1].bar(np.arange(len(n_rbf_arr)) + (j - 1) * width,
                    adverse, width=width,
                    label=rf"$R={shell_R[r_idx]:g}\sigma$")
    axes[1].set_xticks(np.arange(len(n_rbf_arr)))
    axes[1].set_xticklabels([str(n) for n in n_rbf_arr], rotation=0)
    axes[1].set_xlabel(r"$n_{\rm rbf}$")
    axes[1].set_ylabel("adverse-flow fraction")
    axes[1].set_title(r"Solid-angle fraction with $\hat r \cdot f_{\rm rbf} \geq 0$")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend(fontsize=8)

    for j, r_idx in enumerate(selected_R_idx):
        deadf = [rec["dead_frac"][r_idx] for rec in radial_records]
        axes[2].bar(np.arange(len(n_rbf_arr)) + (j - 1) * width,
                    deadf, width=width,
                    label=rf"$R={shell_R[r_idx]:g}\sigma$")
    axes[2].set_xticks(np.arange(len(n_rbf_arr)))
    axes[2].set_xticklabels([str(n) for n in n_rbf_arr], rotation=0)
    axes[2].set_xlabel(r"$n_{\rm rbf}$")
    axes[2].set_ylabel("dead-zone fraction")
    axes[2].set_title("Solid-angle fraction with $\\|f\\| < \\epsilon$")
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(True, axis="y", alpha=0.3)
    axes[2].legend(fontsize=8)

    fig.suptitle("Per-model risk scalars (off-cloud safety summary)", y=1.02)
    fig.tight_layout()
    out_path = out_dir / "summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")

    summary_json = {
        "config": {
            "models_dir": str(models_dir),
            "z_slices": list(args.z_slice),
            "grid_n": args.grid_n,
            "grid_half_width": args.grid_half_width,
            "shell_r_grid": list(map(float, shell_R)),
            "n_sphere": args.n_sphere,
            "dead_zone_thresh": args.dead_zone_thresh,
            "sigma_attr": sigma_attr,
            "x_centroid": list(map(float, x_centroid)),
            "f_truth_mean_mag": f_truth_mean_mag,
            "dead_zone_eps_abs": dz_eps,
        },
        "truth": {
            "shell_R": list(map(float, shell_R)),
            "R_mean": list(map(float, truth_R_mean)),
            "R_inward_frac": list(map(float, truth_R_inward_frac)),
            "mag_mean": list(map(float, truth_mag_mean)),
        },
        "models": radial_records,
    }
    (out_dir / "risk_surface.json").write_text(json.dumps(summary_json, indent=2))
    print(f"  wrote {out_dir / 'risk_surface.json'}")

    # -- Console table --------------------------------------------------------
    print("\nPer-model scalars:")
    print(f"  {'n_rbf':>6} {'R_esc/sig':>10} "
          f"{'adv@R=1':>10} {'adv@R=2':>10} {'adv@R=3':>10} "
          f"{'dead@R=1':>10} {'dead@R=2':>10} {'dead@R=3':>10}")
    for rec in radial_records:
        sel = [int(np.argmin(np.abs(np.array(rec['shell_R']) - r)))
               for r in (1.0, 2.0, 3.0)]
        adv = [1.0 - rec["R_inward_frac"][i] for i in sel]
        dead = [rec["dead_frac"][i] for i in sel]
        R_esc_str = "inf" if not np.isfinite(rec['R_esc']) else f"{rec['R_esc']:.3f}"
        print(f"  {rec['n_rbf']:>6d} {R_esc_str:>10} "
              f"{adv[0]:>10.3f} {adv[1]:>10.3f} {adv[2]:>10.3f} "
              f"{dead[0]:>10.3f} {dead[1]:>10.3f} {dead[2]:>10.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
