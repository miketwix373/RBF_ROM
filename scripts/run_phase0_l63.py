"""Phase 0 negative-control driver: Lorenz-63 SINDy fit and acceptance.

Locked spec: `docs/journal/2026-06-07-phase-0-rbf-design.md` and the
*RBF block - L63* section of `docs/notes/phase-0-design.md`.

The L63 testbed is the *negative control*: the constrained quadratic
backbone exactly captures the Lorenz dynamics, and the flat-isotropic
RBF dictionary is deliberately weak. Acceptance is the four-condition
rule (norm ratio, RBF sparsity, Lorenz coefficient recovery, Jacobian
unstable-eigenvalue counts at the three fixed points; Sparrow 1982).

Pipeline:
    1. `chord2.data.load("LOR63")` snapshots.
    2. `chord2.sindy.fit_phase0(..., rbf_kind="flat_isotropic")`.
    3. `chord2.diagnostics.l63_acceptance` + `phase0_summary`
       + `long_run_boundedness`.
    4. Persist `sindy_model.npz`, `acceptance.json`, `diagnostics.png`
       under `results/LOR63/phase0/`.

The locked n_rbf sweep `{20, 50, 100}` runs by default. A single n_rbf
can be forced with `--n-rbf`.

No clustering / local-basis stage here: Phase 0 is a single global fit
on the resolved coordinates of a 3-D system.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chord2 import data, sindy
from chord2.diagnostics import (
    l63_acceptance,
    long_run_boundedness,
    phase0_summary,
)


def _jsonable(obj):
    """Recursively make a structure JSON-serialisable (numpy + complex)."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _plot_diagnostics(out_path: Path, ds, model, acc, boundedness_T,
                      lambda_poly, lambda_rbf, n_rbf):
    """Three-panel diagnostics figure."""
    sigma = float(ds.metadata.get("sigma", 10.0))
    rho = float(ds.metadata.get("rho", 28.0))
    beta = float(ds.metadata.get("beta", 8.0 / 3.0))
    dt = float(ds.t[1] - ds.t[0])

    a0 = ds.u[0].astype(np.float64)
    t_pred, A_pred = sindy.integrate_rk4(model, a0, dt, boundedness_T)
    n = min(A_pred.shape[0], ds.u.shape[0])

    fig = plt.figure(figsize=(15, 5))

    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.plot(ds.u[:n, 0], ds.u[:n, 1], ds.u[:n, 2],
            lw=0.3, color="black", label="FOM")
    ax.plot(A_pred[:n, 0], A_pred[:n, 1], A_pred[:n, 2],
            lw=0.3, color="C3", alpha=0.7, label="SINDy")
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.set_zlabel("$z$")
    ax.set_title(f"L63 attractor  ($\\sigma$={sigma}, $\\rho$={rho}, "
                 f"$\\beta$=8/3)")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(1, 3, 2)
    mag = np.max(np.abs(model.xi_poly), axis=1)
    names = model.layout["names"]
    bars = ax.bar(range(len(names)), mag, color="C0")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, fontsize=8)
    ax.set_ylabel("max$_i |\\xi_{\\rm poly}[j, i]|$")
    ax.set_title(f"Polynomial coefficients  ($\\lambda_{{\\rm poly}}$="
                 f"{lambda_poly:g})")
    ax.set_yscale("log")
    ax.axhline(lambda_poly, color="grey", ls="--", lw=0.7,
               label=f"threshold {lambda_poly}")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(1, 3, 3)
    rbf_mag = np.max(np.abs(model.xi_rbf), axis=1)
    if rbf_mag.size > 0:
        ax.bar(range(len(rbf_mag)), rbf_mag, color="C3")
    ax.set_xlabel("RBF index")
    ax.set_ylabel("max$_i |\\xi_{\\rm rbf}[j, i]|$")
    verdict = "PASS" if acc["pass"] else "FAIL"
    ax.set_title(f"RBF coefficients  (n_rbf={n_rbf})\n"
                 f"$||\\xi_{{\\rm rbf}}||/||\\xi_{{\\rm poly}}||$="
                 f"{acc['cond1_ratio']['value']:.2e}  --  {verdict}")
    if rbf_mag.size > 0 and rbf_mag.max() > 0:
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_single(ds, n_rbf: int, lambda_poly: float, lambda_rbf: float,
               seed: int, gamma: float, n_knn: int,
               boundedness_T: float, out_dir: Path) -> dict:
    """One Phase 0 fit on L63 at fixed n_rbf, with all diagnostics."""
    A = ds.u.astype(np.float64)
    dt = float(ds.t[1] - ds.t[0])
    sigma = float(ds.metadata.get("sigma", 10.0))
    rho = float(ds.metadata.get("rho", 28.0))
    beta = float(ds.metadata.get("beta", 8.0 / 3.0))

    model = sindy.fit_phase0(
        A, dt,
        rbf_kind="flat_isotropic",
        rbf_kwargs={"n_rbf": n_rbf, "seed": seed,
                    "gamma": gamma, "n_knn": n_knn},
        lambda_poly=lambda_poly,
        lambda_rbf=lambda_rbf,
        constrain_energy=True,
        drop_for_cond=False,
    )

    summary = phase0_summary(model)
    acceptance = l63_acceptance(model, sigma=sigma, rho=rho, beta=beta)

    centroid = A.mean(axis=0)
    hull_radius = 1.5 * float(np.linalg.norm(A - centroid[None, :], axis=1).max())
    boundedness = long_run_boundedness(
        model, A[0], dt, T=boundedness_T,
        hull_radius=hull_radius, hull_centroid=centroid,
    )

    shard = out_dir / f"n_rbf_{n_rbf:03d}"
    shard.mkdir(parents=True, exist_ok=True)

    np.savez(
        shard / "sindy_model.npz",
        xi_poly=model.xi_poly,
        xi_rbf=model.xi_rbf,
        centers=model.centers,
        widths=model.widths,
        rbf_mu_A=model.rbf_mu_A,
        rbf_sigma_A=model.rbf_sigma_A,
        feature_names=np.array(model.layout["names"]),
    )

    record = {
        "n_rbf": int(n_rbf),
        "lambda_poly": float(lambda_poly),
        "lambda_rbf": float(lambda_rbf),
        "summary": summary,
        "acceptance": _jsonable(acceptance),
        "boundedness": _jsonable(boundedness),
    }
    (shard / "acceptance.json").write_text(json.dumps(record, indent=2))

    _plot_diagnostics(shard / "diagnostics.png", ds, model, acceptance,
                      boundedness_T, lambda_poly, lambda_rbf, n_rbf)

    return record


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n-rbf", type=int, nargs="*", default=[20, 50, 100],
                   help="centre counts to sweep (default: 20 50 100)")
    p.add_argument("--lambda-poly", type=float, default=1.0,
                   help="STLSQ threshold on normalised polynomial block")
    p.add_argument("--lambda-rbf", type=float, default=0.5,
                   help="STLSQ threshold on normalised RBF block")
    p.add_argument("--gamma", type=float, default=1.0,
                   help="isotropic-width scale (multiplier on 5-NN median)")
    p.add_argument("--n-knn", type=int, default=5,
                   help="neighbours used in the width heuristic")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--T-boundedness", type=float, default=50.0,
                   help="integration horizon for the boundedness check")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="override default results/LOR63/phase0/")
    args = p.parse_args()

    ds = data.load("LOR63")
    print(f"Loaded {ds}")

    out_dir = args.out_dir or (data.results_dir("LOR63") / "phase0")
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep = []
    for n_rbf in args.n_rbf:
        print(f"\n=== n_rbf = {n_rbf} ===")
        record = run_single(
            ds, n_rbf, args.lambda_poly, args.lambda_rbf,
            args.seed, args.gamma, args.n_knn, args.T_boundedness, out_dir,
        )
        sweep.append(record)
        acc = record["acceptance"]
        print(f"  PASS = {acc['pass']}")
        print(f"    cond1 ratio   : {acc['cond1_ratio']['value']:.3e}  "
              f"(< {acc['cond1_ratio']['tol']})")
        print(f"    cond2 nnz_rbf : {acc['cond2_nnz_rbf']['value']}  "
              f"(<= {acc['cond2_nnz_rbf']['max']})")
        print(f"    cond3 coeffs  : max_rel_err="
              f"{acc['cond3_coeffs']['max_rel_err']:.3e}")
        for nm, info in acc["cond4_jacobians"]["per_point"].items():
            print(f"    {nm:8s}  unstable={info['n_unstable']} "
                  f"(expected {info['expected_unstable']})")
        b = record["boundedness"]
        print(f"  bounded over T={args.T_boundedness}: "
              f"finite={b['finite']}, bounded={b['bounded']}, "
              f"max_norm={b['max_norm']:.3f}")

    (out_dir / "sweep_summary.json").write_text(json.dumps(sweep, indent=2))
    print(f"\nResults under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
