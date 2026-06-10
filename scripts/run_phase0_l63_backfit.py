"""Phase 0 / L63 negative-control under the backfit architecture.

L63 is the locked Phase 0 negative-control: the polynomial library exactly
spans the L63 RHS (sigma(y-x), x(rho-z)-y, xy-beta*z), so any well-formed
joint fit must drive the RBF block to vanish. Pass rule (locked
2026-06-07, `docs/journal/2026-06-07-phase-0-rbf-design.md`, encoded in
`chord2.diagnostics.l63_acceptance`):

  1. ||xi_rbf||_F / ||xi_poly||_F < 1e-3   (*coefficient-norm*; locked
     under the joint moment-orthogonal STLSQ architecture)
  2. nnz(xi_rbf) <= 2
  3. The seven non-zero Lorenz polynomial coefficients (sigma, rho, beta
     and the xy/xz cross terms) recovered to relative error < 5e-2
  4. Jacobian unstable-eigenvalue count matches Sparrow 1982 at every
     L63 fixed point: origin -> 1, C_pm -> 2.

The backfit architecture changes (1)'s identifiability story — stage-2's
bare-RBF ``xi_rbf`` lives in an un-orthogonalised basis. We compute and
report the original coefficient-norm ratio (cond1) AND the consultant-
recommended backfit-invariant restatement,

  ||Phi_rbf @ xi_rbf||_2 / ||Theta_poly @ xi_poly||_2   (functional norm),

so the verdict can be read in both bases.

Sweep `n_rbf` in `{20, 50, 100}` to match the locked recipe in
`scripts/run_phase0_l63.py`. RBF dictionary is flat-isotropic K-means
with `gamma = 1.0`, `n_knn = 5`. No conditioning gate drop on L63
(`cond_max = inf`) — the dictionary is deliberately weak so the
negative-control test is sharp.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from chord2 import data, diagnostics, sindy

REPO = Path(__file__).resolve().parents[1]


def _jsonable(obj):
    import numpy as _np
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, _np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, complex):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, (_np.bool_,)):
        return bool(obj)
    return obj


def run_one(n_rbf: int, *, lambda_poly: float, lambda_rbf: float,
            gamma: float, n_knn: int, seed: int, n_init: int,
            constrain_energy: bool, model_dir: Path | None = None) -> dict:
    ds = data.load("LOR63")
    A = np.asarray(ds.u, dtype=np.float64)
    dt = float(ds.dt)
    sigma_true = float(ds.metadata["sigma"])
    rho_true = float(ds.metadata["rho"])
    beta_true = float(ds.metadata["beta"])

    model = sindy.fit_phase0_backfit(
        A, dt,
        rbf_kind="flat_isotropic",
        rbf_kwargs={"n_rbf": n_rbf, "seed": seed, "n_init": n_init,
                    "gamma": gamma, "n_knn": n_knn},
        lambda_poly=lambda_poly,
        lambda_rbf=lambda_rbf,
        p=2,
        constrain_energy=constrain_energy,
        cond_max=float("inf"),  # match locked L63 'drop_for_cond=False'
    )

    acceptance = diagnostics.l63_acceptance(
        model, sigma=sigma_true, rho=rho_true, beta=beta_true,
    )
    summary = diagnostics.phase0_summary(model)

    # Backfit-invariant functional-norm RBF gate (restatement of cond1).
    A_mid, dAdt = sindy.deriv_5point(A, dt)
    Theta_poly = sindy.poly_features(A_mid, model.p)
    poly_pred = Theta_poly @ model.xi_poly
    poly_pred_norm = float(np.linalg.norm(poly_pred))
    if model.xi_rbf.shape[0] > 0:
        Phi_rbf = sindy.rbf_features_iso(
            A_mid, model.centers, model.widths,
            mu_A=model.rbf_mu_A, sigma_A=model.rbf_sigma_A,
        )
        rbf_pred = Phi_rbf @ model.xi_rbf
    else:
        rbf_pred = np.zeros_like(poly_pred)
    rbf_pred_norm = float(np.linalg.norm(rbf_pred))
    functional_ratio = (rbf_pred_norm / poly_pred_norm
                        if poly_pred_norm > 0 else float("nan"))

    info = model.info
    record = {
        "n_rbf": int(n_rbf),
        "lambda_poly": float(lambda_poly),
        "lambda_rbf": float(lambda_rbf),
        "gamma": float(gamma),
        "n_knn": int(n_knn),
        "constrain_energy": bool(constrain_energy),
        "stage1_residual_norm": float(info["stage1_residual_norm"]),
        "dAdt_norm": float(info["dAdt_norm"]),
        "stage1_R_over_dAdt": float(info["stage1_residual_norm"]
                                    / info["dAdt_norm"]),
        "cond_pre_gate": float(info["cond_pre_gate"]),
        "cond_post_gate": float(info["cond_post_gate"]),
        "stage1_n_iter": int(info["stage1_fit"]["n_iter"]),
        "stage2_n_iter": int(info["stage2_fit"]["n_iter"]),
        "xi_poly_norm": float(np.linalg.norm(model.xi_poly)),
        "xi_rbf_norm": float(np.linalg.norm(model.xi_rbf)),
        "poly_pred_norm": poly_pred_norm,
        "rbf_pred_norm": rbf_pred_norm,
        "cond1_coeff_ratio": float(np.linalg.norm(model.xi_rbf)
                                   / max(np.linalg.norm(model.xi_poly), 1e-300)),
        "functional_ratio_rbf_over_poly": functional_ratio,
        "acceptance": _jsonable(acceptance),
        "summary": _jsonable(summary),
    }

    if model_dir is not None:
        model_path = model_dir / f"model_LOR63_n_rbf_{n_rbf:03d}.npz"
        sindy.save_model(model, model_path)
        record["model_path"] = str(model_path)

    return record


def _print_record(r: dict) -> None:
    acc = r["acceptance"]
    cond1 = acc["cond1_ratio"]
    cond2 = acc["cond2_nnz_rbf"]
    cond3 = acc["cond3_coeffs"]
    cond4 = acc["cond4_jacobians"]
    fr = r["functional_ratio_rbf_over_poly"]
    print(
        f"  stage1 R/||dAdt||                 = {r['stage1_R_over_dAdt']:.3e}\n"
        f"  cond_pre/post gate                = {r['cond_pre_gate']:.3e}"
        f" / {r['cond_post_gate']:.3e}\n"
        f"  cond1  ||xi_rbf||/||xi_poly||     = {cond1['value']:.3e}"
        f"  [tol < {cond1['tol']:.0e}]   "
        f"{'PASS' if cond1['pass'] else 'FAIL'}\n"
        f"  cond1' ||Phi xi_rbf|| /            \n"
        f"         ||Theta xi_poly|| (funct.)  = {fr:.3e}\n"
        f"  cond2  nnz(xi_rbf)                = {cond2['value']}"
        f"   [max <= {cond2['max']}]          "
        f"{'PASS' if cond2['pass'] else 'FAIL'}\n"
        f"  cond3  max coeff rel-err          = {cond3['max_rel_err']:.3e}"
        f"  [tol < {cond3['tol']:.0e}]   "
        f"{'PASS' if cond3['pass'] else 'FAIL'}\n"
        f"  cond4  Jacobian unstable counts   "
        f"{'PASS' if cond4['pass'] else 'FAIL'}\n"
        f"  -------> overall {'PASS' if acc['pass'] else 'FAIL'}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rbf", nargs="+", type=int, default=[20, 50, 100])
    ap.add_argument("--lambda-poly", type=float, default=1.0)
    ap.add_argument("--lambda-rbf", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--n-knn", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-init", type=int, default=10)
    ap.add_argument("--no-constrain", action="store_true")
    args = ap.parse_args()

    out_dir = REPO / "results" / "LOR63" / "phase0_backfit"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for n_rbf in args.n_rbf:
        print(f"--- LOR63   n_rbf = {n_rbf} ---")
        r = run_one(
            n_rbf,
            lambda_poly=args.lambda_poly,
            lambda_rbf=args.lambda_rbf,
            gamma=args.gamma,
            n_knn=args.n_knn,
            seed=args.seed,
            n_init=args.n_init,
            constrain_energy=not args.no_constrain,
            model_dir=model_dir,
        )
        records.append(r)
        _print_record(r)
        print()

    out_path = out_dir / "phase0_l63_backfit.json"
    out_path.write_text(json.dumps(records, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
