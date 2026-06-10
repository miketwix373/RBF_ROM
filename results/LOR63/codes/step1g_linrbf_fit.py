"""L63 step 1g: refit linear + RBF using *same* centres/widths as step 1e/1 models.

For each of the two reference RBF-only models (attr-scaled n=3200 alpha=1, and
knn n=3200), re-fit STLSQ on the hybrid library

    Phi = [ 1, x, y, z | phi_1(x), ..., phi_K(x) ]

with the four polynomial columns prepended to the existing iso-RBF block.
Centres `mu_j`, widths `sigma_j`, the standardisation (`mu_A, sigma_A`), and
the hyperparameters (`lambda_rbf=1e-3`, `lambda_tikh=1e-8`) are reused exactly
from the saved RBF-only model so the hybrid fit differs from the pure-RBF fit
only by the four polynomial columns.

Columns are L2-normalised before STLSQ (poly columns get the same treatment as
the RBF block), so the single threshold `lambda_rbf` treats the two blocks on
equal footing. Coefficients are unscaled at the end and saved separately as
`xi_poly` (4 x 3) and `xi_rbf` (K x 3).

Outputs (results/LOR63/step1g_linrbf/):
  model_linrbf_attr_n03200_a1.00.npz
  model_linrbf_knn_n03200.npz
  step1g_linrbf.json    -- recovered (sigma_hat, rho_hat, beta_hat), nnz, residuals

Truth recovery target (from L63):
  axis x  ->  ( bias,   -sigma,   +sigma,   0    ) = (0, -10, +10, 0)
  axis y  ->  ( bias,   +rho,     -1,        0    ) = (0, +28, -1,  0)        + RBFs carry -xz
  axis z  ->  ( bias,    0,        0,       -beta) = (0,   0,  0, -8/3)       + RBFs carry +xy
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _l63_rbf_lib import f_truth_l63, SIGMA_L63, RHO_L63, BETA_L63  # noqa: E402
from chord2 import data, sindy  # noqa: E402


POLY_NAMES = ("1", "x", "y", "z")


def _load_ref_model(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    out = {
        "centers":   d["centers"],
        "widths":    d["widths"],
        "mu_A":      d["mu_A"],
        "sigma_A":   d["sigma_A"],
        "col_norms": d["col_norms"],
        "xi_rbf0":   d["xi"],
        "active0":   d["active"],
        "meta":      dict(d["meta"].item()),
    }
    return out


def _build_poly1(A: np.ndarray) -> np.ndarray:
    """Return [1, x, y, z] for snapshots A of shape (M, 3)."""
    M = A.shape[0]
    Phi = np.empty((M, 4), dtype=A.dtype)
    Phi[:, 0] = 1.0
    Phi[:, 1:] = A
    return Phi


def _stlsq(Phi_n: np.ndarray, dAdt: np.ndarray, *,
           lambda_rbf: float, lambda_tikh: float, max_iter: int = 50):
    """STLSQ on column-normalised library; returns (xi_n, active, history).

    `history` is a list of dicts (one per iteration) carrying `kappa` of the
    un-augmented active design, `n_active`, and residual_rel. SVD-with-Tikhonov
    in singular-value space matches the row-augmented Tikhonov solution; the
    SVD path is taken so we can log kappa without a second decomposition.
    """
    n_f = Phi_n.shape[1]
    active = np.ones(n_f, dtype=bool)
    prev_active = None
    xi_n = np.zeros((n_f, dAdt.shape[1]))
    history = []
    for it in range(max_iter):
        idx = np.where(active)[0]
        if idx.size == 0:
            break
        U, s, Vt = np.linalg.svd(Phi_n[:, idx], full_matrices=False)
        kappa = float(s[0] / s[-1]) if s.size > 0 and s[-1] > 0.0 else float("inf")
        if lambda_tikh > 0.0:
            filt = s / (s * s + lambda_tikh)
        else:
            M, N = Phi_n.shape[0], idx.size
            cutoff = np.finfo(s.dtype).eps * (s[0] if s.size else 1.0) * max(M, N)
            inv_s = np.where(s > cutoff, 1.0 / np.where(s > cutoff, s, 1.0), 0.0)
            filt = inv_s
        xi_act = Vt.T @ (filt[:, None] * (U.T @ dAdt))
        xi_n = np.zeros((n_f, dAdt.shape[1]))
        xi_n[idx, :] = xi_act
        dAdt_pred = Phi_n[:, idx] @ xi_act
        res_rel = float(np.linalg.norm(dAdt - dAdt_pred)
                        / max(np.linalg.norm(dAdt), 1e-30))
        history.append({
            "iter":     int(it),
            "n_active": int(idx.size),
            "kappa":    kappa,
            "residual_rel": res_rel,
        })
        new_active = active.copy()
        mags = np.max(np.abs(xi_n), axis=1)
        new_active[mags < lambda_rbf] = False
        if prev_active is not None and np.array_equal(new_active, active):
            break
        prev_active = active.copy()
        active = new_active
    return xi_n, active, history


def _fit_linrbf(ref: dict, A_mid: np.ndarray, dAdt: np.ndarray, *,
                lambda_rbf: float, lambda_tikh: float, tag: str):
    """Refit hybrid linear+RBF using the centres/widths from `ref`.

    Returns the saved-model dict (np.savez-compatible) plus diagnostics.
    """
    centers   = ref["centers"]
    widths    = ref["widths"]
    mu_A      = ref["mu_A"]
    sigma_A   = ref["sigma_A"]

    Phi_poly_raw = _build_poly1(A_mid)                                  # (M, 4)
    Phi_rbf_raw  = sindy.rbf_features_iso(A_mid, centers, widths,
                                          mu_A=mu_A, sigma_A=sigma_A)   # (M, K)
    K = Phi_rbf_raw.shape[1]

    cn_poly = np.linalg.norm(Phi_poly_raw, axis=0)
    cn_rbf  = np.linalg.norm(Phi_rbf_raw,  axis=0)
    cn_poly = np.where(cn_poly == 0.0, 1.0, cn_poly)
    cn_rbf  = np.where(cn_rbf  == 0.0, 1.0, cn_rbf)

    Phi_full_n = np.hstack([Phi_poly_raw / cn_poly,
                            Phi_rbf_raw  / cn_rbf])                     # (M, 4+K)

    t0 = time.time()
    max_iter_stlsq = 50
    xi_n, active, history = _stlsq(Phi_full_n, dAdt,
                                   lambda_rbf=lambda_rbf,
                                   lambda_tikh=lambda_tikh,
                                   max_iter=max_iter_stlsq)
    t_fit = time.time() - t0
    n_iter = len(history)
    hit_max_iter = n_iter >= max_iter_stlsq and history[-1].get("n_active", 0) > 0
    kappa_final = float(history[-1]["kappa"]) if history else float("nan")

    xi_poly_n = xi_n[:4, :]
    xi_rbf_n  = xi_n[4:, :]
    xi_poly   = xi_poly_n / cn_poly[:, None]
    xi_rbf    = xi_rbf_n  / cn_rbf [:, None]

    dAdt_pred = Phi_full_n @ xi_n
    rel_err = float(np.linalg.norm(dAdt - dAdt_pred)
                    / max(np.linalg.norm(dAdt), 1e-30))

    # Identifiability gap: refit linear block on the RBF-residual; the
    # difference vs the joint solution measures linear/RBF coefficient leakage.
    Phi_poly_n = Phi_poly_raw / cn_poly
    dAdt_minus_rbf = dAdt - (Phi_rbf_raw / cn_rbf) @ xi_rbf_n
    xi_poly_n_only, *_ = np.linalg.lstsq(Phi_poly_n, dAdt_minus_rbf, rcond=None)
    xi_poly_only = xi_poly_n_only / cn_poly[:, None]
    sigma_hat_only = float(xi_poly_only[2, 0])
    rho_hat_only   = float(xi_poly_only[1, 1])
    beta_hat_only  = float(-xi_poly_only[3, 2])

    # Collinearity of each active RBF column with the poly block.
    # Project Phi_rbf_active onto span(Phi_poly_n) using the QR of the poly
    # block; report ||P_poly phi_k|| / ||phi_k|| over active k.
    Q_poly, _ = np.linalg.qr(Phi_poly_n)
    active_rbf_mask = active[4:]
    active_rbf_idx = np.where(active_rbf_mask)[0]
    if active_rbf_idx.size > 0:
        Phi_rbf_active_n = (Phi_rbf_raw[:, active_rbf_idx]
                            / cn_rbf[active_rbf_idx])
        proj = Q_poly @ (Q_poly.T @ Phi_rbf_active_n)
        col_norm_active = np.linalg.norm(Phi_rbf_active_n, axis=0)
        col_norm_active = np.where(col_norm_active == 0.0, 1.0, col_norm_active)
        ratios = (np.linalg.norm(proj, axis=0) / col_norm_active)
        collin_stats = {
            "n_active":  int(active_rbf_idx.size),
            "min":       float(ratios.min()),
            "median":    float(np.median(ratios)),
            "max":       float(ratios.max()),
            "frac_gt_0.5": float((ratios > 0.5).mean()),
        }
    else:
        collin_stats = {"n_active": 0, "min": float("nan"),
                        "median": float("nan"), "max": float("nan"),
                        "frac_gt_0.5": float("nan")}

    sigma_hat = float(xi_poly[2, 0])
    rho_hat   = float(xi_poly[1, 1])
    beta_hat  = float(-xi_poly[3, 2])

    saved = {
        "centers":     centers,
        "widths":      widths,
        "mu_A":        mu_A,
        "sigma_A":     sigma_A,
        "col_norms_poly": cn_poly,
        "col_norms_rbf":  cn_rbf,
        "xi_poly":     xi_poly,
        "xi_rbf":      xi_rbf,
        "active_poly": active[:4],
        "active_rbf":  active[4:],
        "meta": {
            "kind":           "linrbf",
            "tag":            tag,
            "ref_meta":       ref["meta"],
            "lambda_rbf":     float(lambda_rbf),
            "lambda_tikh":    float(lambda_tikh),
            "n_rbf":          int(K),
            "n_poly":         4,
            "n_iter":         int(n_iter),
            "hit_max_iter":   bool(hit_max_iter),
            "kappa_final":    float(kappa_final),
            "rel_err_alpha_dot": float(rel_err),
            "nnz_poly":       int(active[:4].sum()),
            "nnz_rbf":        int(active[4:].sum()),
            "sigma_hat":      sigma_hat,
            "rho_hat":        rho_hat,
            "beta_hat":       beta_hat,
            "sigma_hat_only": sigma_hat_only,
            "rho_hat_only":   rho_hat_only,
            "beta_hat_only":  beta_hat_only,
            "collin_stats":   collin_stats,
            "t_fit_seconds":  float(t_fit),
        },
    }
    diag = {
        "tag":         tag,
        "n_rbf":       int(K),
        "n_iter":      int(n_iter),
        "hit_max_iter": bool(hit_max_iter),
        "kappa_final": float(kappa_final),
        "rel_err":     float(rel_err),
        "nnz_poly":    int(active[:4].sum()),
        "nnz_rbf":     int(active[4:].sum()),
        "rbf_nnz_frac": float(active[4:].sum() / K),
        "xi_poly":     xi_poly.tolist(),
        "sigma_hat":   sigma_hat,
        "rho_hat":     rho_hat,
        "beta_hat":    beta_hat,
        "sigma_hat_only": sigma_hat_only,
        "rho_hat_only":   rho_hat_only,
        "beta_hat_only":  beta_hat_only,
        "collin_stats":   collin_stats,
        "history":     history,
        "t_fit_seconds": float(t_fit),
    }
    return saved, diag


def _save_model(out_path: Path, saved: dict) -> None:
    np.savez(
        out_path,
        centers=saved["centers"],
        widths=saved["widths"],
        mu_A=saved["mu_A"],
        sigma_A=saved["sigma_A"],
        col_norms_poly=saved["col_norms_poly"],
        col_norms_rbf=saved["col_norms_rbf"],
        xi_poly=saved["xi_poly"],
        xi_rbf=saved["xi_rbf"],
        active_poly=saved["active_poly"],
        active_rbf=saved["active_rbf"],
        meta=np.array(saved["meta"], dtype=object),
    )


def _print_recovery(diag: dict) -> None:
    print(f"  [{diag['tag']:>22s}]  rel_err={diag['rel_err']:.3e}  "
          f"nnz_poly={diag['nnz_poly']}/4  "
          f"nnz_rbf={diag['nnz_rbf']}/{diag['n_rbf']} "
          f"({100*diag['rbf_nnz_frac']:.1f}%)  "
          f"kappa={diag['kappa_final']:.2e}  "
          f"iters={diag['n_iter']}{' (HIT MAX)' if diag['hit_max_iter'] else ''}  "
          f"t={diag['t_fit_seconds']:.1f}s")
    truth = {"sigma": SIGMA_L63, "rho": RHO_L63, "beta": BETA_L63}
    s, r, b = diag["sigma_hat"], diag["rho_hat"], diag["beta_hat"]
    so, ro, bo = (diag["sigma_hat_only"], diag["rho_hat_only"],
                  diag["beta_hat_only"])
    print(f"      sigma_hat = {s:+9.4f}   (joint, truth={truth['sigma']:+.4f}, "
          f"err={s-truth['sigma']:+.2e})")
    print(f"      rho_hat   = {r:+9.4f}   (joint, truth={truth['rho']:+.4f}, "
          f"err={r-truth['rho']:+.2e})")
    print(f"      beta_hat  = {b:+9.4f}   (joint, truth={truth['beta']:+.4f}, "
          f"err={b-truth['beta']:+.2e})")
    print(f"      identifiability gap (linear-only refit minus joint):")
    print(f"        Dsigma = {so-s:+.3e}   Drho = {ro-r:+.3e}   "
          f"Dbeta = {bo-b:+.3e}")
    cs = diag["collin_stats"]
    print(f"      active-RBF/poly collinearity: n={cs['n_active']}  "
          f"min={cs['min']:.3f}  med={cs['median']:.3f}  max={cs['max']:.3f}  "
          f"frac(>0.5)={cs['frac_gt_0.5']:.3f}")
    print(f"      xi_poly (rows = [1, x, y, z], cols = [dx/dt, dy/dt, dz/dt]):")
    xi = np.asarray(diag["xi_poly"])
    for i, nm in enumerate(POLY_NAMES):
        print(f"        {nm:>3s}: {xi[i, 0]:+10.5f}  {xi[i, 1]:+10.5f}  "
              f"{xi[i, 2]:+10.5f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir",
                    default="results/LOR63/step1g_linrbf")
    ap.add_argument("--lambda-rbf",  type=float, default=1e-3)
    ap.add_argument("--lambda-tikh", type=float, default=1e-8)
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip cells whose output .npz already exists")
    args = ap.parse_args()

    repo = REPO_ROOT
    out_dir = (repo / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = data.load("LOR63")
    A = ds.u.astype(np.float64)
    dt_data = float(ds.t[1] - ds.t[0])
    print(f"Loaded {ds}; dt_data={dt_data:g}, M={A.shape[0]}")

    A_mid, dAdt = sindy.deriv_5point(A, dt_data)
    print(f"A_mid shape = {A_mid.shape}, dAdt shape = {dAdt.shape}")

    f_truth_mean_mag = float(np.linalg.norm(f_truth_l63(A_mid), axis=1).mean())
    dAdt_norm = float(np.linalg.norm(dAdt))
    print(f"<||f_truth||> = {f_truth_mean_mag:.4f}  "
          f"||dAdt||_F   = {dAdt_norm:.4f}")
    print(f"Truth: sigma={SIGMA_L63}, rho={RHO_L63}, beta={BETA_L63:.4f}")

    sweep_dir = repo / "results/LOR63/step1_nrbf_sweep/models"
    gamma_dir = repo / "results/LOR63/step1e_gamma_sweep/models"
    refs = [
        # n_rbf = 3200 baseline (original two reference fits)
        ("attr_n03200_a1.00", gamma_dir / "model_n03200_a1.00.npz",
         out_dir / "model_linrbf_attr_n03200_a1.00.npz"),
        ("knn_n03200",        sweep_dir / "model_n_rbf_03200.npz",
         out_dir / "model_linrbf_knn_n03200.npz"),
        # n_rbf = 800 — does dropping K rescue identifiability?
        ("knn_n00800",        sweep_dir / "model_n_rbf_00800.npz",
         out_dir / "model_linrbf_knn_n00800.npz"),
        ("attr_n00800_a1.00", gamma_dir / "model_n00800_a1.00.npz",
         out_dir / "model_linrbf_attr_n00800_a1.00.npz"),
        # narrower attr (alpha=0.5) at K=800 — does sigma reduction break
        # the Taylor-affine collinearity for attr-scaled Gaussians?
        ("attr_n00800_a0.50", gamma_dir / "model_n00800_a0.50.npz",
         out_dir / "model_linrbf_attr_n00800_a0.50.npz"),
    ]
    diagnostics = []
    for tag, ref_path, out_path in refs:
        print(f"\n--- {tag} ---")
        print(f"  ref:   {ref_path}")
        print(f"  out:   {out_path}")
        if args.skip_existing and out_path.exists():
            print(f"  -> exists, skip-existing set; skipping fit.")
            continue
        ref = _load_ref_model(ref_path)
        print(f"  centers shape = {ref['centers'].shape}, "
              f"widths min/max = {ref['widths'].min():.4f}/"
              f"{ref['widths'].max():.4f}")
        saved, diag = _fit_linrbf(
            ref, A_mid, dAdt,
            lambda_rbf=args.lambda_rbf,
            lambda_tikh=args.lambda_tikh,
            tag=tag,
        )
        _print_recovery(diag)
        _save_model(out_path, saved)
        diagnostics.append(diag)

    summary = {
        "lambda_rbf":  float(args.lambda_rbf),
        "lambda_tikh": float(args.lambda_tikh),
        "truth": {"sigma": SIGMA_L63, "rho": RHO_L63, "beta": BETA_L63},
        "fits": diagnostics,
    }
    summary_path = out_dir / "step1g_linrbf.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    raise SystemExit(main())
