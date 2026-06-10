"""Step 0: pre-integration field-magnitude shell scan (rom-specialist).

Before committing to the full RBF-only L63 RK4 integration experiment,
evaluate ||f(a)|| on:

  * 10k samples from the FOM attractor cloud, and
  * 10k samples on a *family* of expanding shells (rescaled outward
    around mu_phys by factors {1.10, 1.20, 1.50, 2.00, 3.00, 5.00} by
    default),

for three models -- truth (analytic L63 RHS), poly-only, RBF-only --
and report shell/cloud ratios as a function of shell scale. The locus
where the RBF ratio collapses below ~0.1 is the dictionary's effective
radius of support; trajectories that wander past it during integration
see a vanishing field.

Reference: rom-specialist assessment, "What you did not ask but should",
item 1. Literature analogue: Boninsegna, Nuske & Clementi 2018
(J. Chem. Phys. 148, 241723).

Output:
  results/LOR63/phase0_rbf_only/step0_field_shell.json
  results/LOR63/phase0_rbf_only/step0_field_shell.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _l63_rbf_lib import (  # noqa: E402
    f_poly, f_rbf, f_truth_l63, fit_rbf_only_keep_state,
)

from chord2 import data, sindy  # noqa: E402
from scripts.run_phase0_l96 import _fit_quad_only  # noqa: E402


def _summarise(norms: np.ndarray) -> dict:
    return {
        "mean": float(norms.mean()),
        "median": float(np.median(norms)),
        "p10": float(np.quantile(norms, 0.10)),
        "p90": float(np.quantile(norms, 0.90)),
        "max": float(norms.max()),
        "min": float(norms.min()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n-rbf", type=int, default=1600)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--lambda-rbf", type=float, default=1e-3)
    p.add_argument("--lambda-poly", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--n-knn", type=int, default=5)
    p.add_argument("--n-init", type=int, default=1)
    p.add_argument("--n-cloud", type=int, default=10_000)
    p.add_argument("--shell-scales", type=float, nargs="+",
                   default=[1.10, 1.20, 1.50, 2.00, 3.00, 5.00],
                   help="radial expansion factors around mu_phys")
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    ds = data.load("LOR63")
    A = ds.u.astype(np.float64)
    dt = float(ds.t[1] - ds.t[0])
    print(f"Loaded {ds}; dt={dt:g}, M={A.shape[0]}, T={ds.t[-1]:g}")

    out_dir = args.out_dir or (data.results_dir("LOR63") / "phase0_rbf_only")
    out_dir.mkdir(parents=True, exist_ok=True)

    A_mid, dAdt_true = sindy.deriv_5point(A, dt)

    print("\n--- poly-only fit ---")
    poly_model = _fit_quad_only(A, dt, args.lambda_poly)
    rel_poly = (np.linalg.norm(dAdt_true - sindy.poly_features(A_mid, 2) @ poly_model.xi_poly)
                / np.linalg.norm(dAdt_true))
    print(f"  rel_err = {rel_poly:.3e}")

    print(f"\n--- RBF-only fit (n_rbf={args.n_rbf}, seed={args.seed}) ---")
    rbf = fit_rbf_only_keep_state(
        A_mid, dAdt_true,
        n_rbf=args.n_rbf, seed=args.seed, lambda_rbf=args.lambda_rbf,
        gamma=args.gamma, n_knn=args.n_knn, n_init=args.n_init,
    )
    Phi_n_train = sindy.rbf_features_iso(
        A_mid, rbf["centers"], rbf["widths"],
        mu_A=rbf["mu_A"], sigma_A=rbf["sigma_A"],
    ) / rbf["col_norms"]
    rel_rbf = (np.linalg.norm(dAdt_true - Phi_n_train @ rbf["xi"])
               / np.linalg.norm(dAdt_true))
    print(f"  rel_err = {rel_rbf:.3e}, active = {rbf['nnz']}/{args.n_rbf}")

    rng = np.random.default_rng(args.rng_seed)
    n_cloud = min(args.n_cloud, A.shape[0])
    idx = rng.choice(A.shape[0], size=n_cloud, replace=False)
    A_cloud = A[idx]
    mu_phys = A.mean(axis=0)
    print(f"\nCloud: {A_cloud.shape[0]} samples; mu_phys={mu_phys}")

    n_truth_c = np.linalg.norm(f_truth_l63(A_cloud), axis=1)
    n_poly_c = np.linalg.norm(f_poly(A_cloud, poly_model.xi_poly), axis=1)
    n_rbf_c = np.linalg.norm(
        f_rbf(A_cloud, rbf["centers"], rbf["widths"],
              rbf["mu_A"], rbf["sigma_A"], rbf["col_norms"], rbf["xi"]),
        axis=1,
    )
    cloud_summary = {
        "truth": _summarise(n_truth_c),
        "poly": _summarise(n_poly_c),
        "rbf": _summarise(n_rbf_c),
    }

    print("\n=========================================")
    print("Shell-scan: shell/cloud ratios of median ||f||")
    print("=========================================")
    print("{:>8s} | {:>10s} {:>10s} {:>10s} | {:>10s} {:>10s} {:>10s}".format(
        "scale", "truth_med", "poly_med", "rbf_med",
        "truth/c", "poly/c", "rbf/c",
    ))

    per_scale = []
    for s in args.shell_scales:
        A_shell = mu_phys[None, :] + s * (A_cloud - mu_phys[None, :])
        n_truth_s = np.linalg.norm(f_truth_l63(A_shell), axis=1)
        n_poly_s = np.linalg.norm(f_poly(A_shell, poly_model.xi_poly), axis=1)
        n_rbf_s = np.linalg.norm(
            f_rbf(A_shell, rbf["centers"], rbf["widths"],
                  rbf["mu_A"], rbf["sigma_A"], rbf["col_norms"], rbf["xi"]),
            axis=1,
        )
        rec = {
            "scale": float(s),
            "truth": _summarise(n_truth_s),
            "poly": _summarise(n_poly_s),
            "rbf": _summarise(n_rbf_s),
            "truth_over_cloud_median": float(np.median(n_truth_s) / np.median(n_truth_c)),
            "poly_over_cloud_median": float(np.median(n_poly_s) / np.median(n_poly_c)),
            "rbf_over_cloud_median": float(np.median(n_rbf_s) / np.median(n_rbf_c)),
        }
        per_scale.append(rec)
        print("{:>8.3f} | {:>10.3e} {:>10.3e} {:>10.3e} | "
              "{:>10.3f} {:>10.3f} {:>10.3f}".format(
                  s, rec["truth"]["median"], rec["poly"]["median"], rec["rbf"]["median"],
                  rec["truth_over_cloud_median"], rec["poly_over_cloud_median"],
                  rec["rbf_over_cloud_median"],
              ))

    crossing = None
    for rec in per_scale:
        if rec["rbf_over_cloud_median"] < 0.1:
            crossing = rec["scale"]
            break
    if crossing is None:
        verdict = (f"RBF shell/cloud stays >= 0.1 across all scales tested "
                   f"(max scale tested = {max(args.shell_scales):g}). "
                   f"Effective support extends well beyond the cloud; "
                   f"full integration is required to characterise failure.")
    else:
        verdict = (f"RBF shell/cloud crosses 0.1 at shell scale = {crossing:g}. "
                   f"Trajectories that wander past ~{crossing:g}x the attractor "
                   f"radius enter the dead zone.")
    print(f"\nVerdict: {verdict}")

    out_json = out_dir / "step0_field_shell.json"
    out_json.write_text(json.dumps({
        "config": {
            "n_rbf": args.n_rbf, "seed": args.seed,
            "lambda_rbf": args.lambda_rbf, "lambda_poly": args.lambda_poly,
            "gamma": args.gamma, "n_knn": args.n_knn, "n_init": args.n_init,
            "n_cloud": n_cloud, "shell_scales": list(args.shell_scales),
            "rng_seed": args.rng_seed,
            "rel_err_poly": float(rel_poly), "rel_err_rbf": float(rel_rbf),
            "mu_phys": mu_phys.tolist(),
        },
        "cloud": cloud_summary,
        "per_scale": per_scale,
        "verdict": verdict,
    }, indent=2))
    print(f"\nWrote {out_json}")

    # -- Plot: shell/cloud ratio vs scale, log-log --------------------------
    scales = np.array([r["scale"] for r in per_scale])
    ratio_truth = np.array([r["truth_over_cloud_median"] for r in per_scale])
    ratio_poly = np.array([r["poly_over_cloud_median"] for r in per_scale])
    ratio_rbf = np.array([r["rbf_over_cloud_median"] for r in per_scale])

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.4))
    ax.plot(scales, ratio_truth, "o-", color="k", lw=1.4, label="truth (L63 RHS)")
    ax.plot(scales, ratio_poly, "s--", color="C0", lw=1.4, label="poly-only")
    ax.plot(scales, ratio_rbf, "D:", color="C3", lw=1.6, label=f"RBF-only n={args.n_rbf}")
    ax.axhline(1.0, color="grey", lw=0.7, ls=":")
    ax.axhline(0.1, color="red", lw=0.7, ls=":",
               label="dead-zone threshold (0.1)")
    ax.set_xlabel("shell scale (radial multiplier around $\\mu_{phys}$)")
    ax.set_ylabel("median $\\Vert f \\Vert$  shell / cloud")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, ls=":", alpha=0.4, which="both")
    ax.legend(fontsize=10, loc="lower left")
    ax.set_title(f"L63 vector-field shell scan  "
                 f"(n_rbf={args.n_rbf}, seed={args.seed})")
    fig.tight_layout()
    out_png = out_dir / "step0_field_shell.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
