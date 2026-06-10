"""Step 1d post: outer-absorbing-radius metric from the saved risk_surface.json.

Refines the off-cloud-safety conclusion. The original step 1d implicitly
asked "is the field inward everywhere?" — too strict. The relevant
property for absorbing-ball stability is "is the field inward at large
enough R" — outward push at intermediate radii is fine if the outer
shell pulls back.

Defines, per model:
  R_abs(thresh, floor) =
    smallest R among sampled shells such that, for *all* R' >= R,
    inward-fraction(R') >= thresh  AND
    ||f||_mean(R')   >= floor

  R_abs = inf  if no such R exists (no absorbing tail).

Uses the radial data already in risk_surface.json (no re-sampling).
Writes outer_absorbing.json + outer_absorbing_summary.png next to it.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def r_abs(shell_R, inward_frac, mag_mean, *, thresh, floor):
    """Smallest R in shell_R such that all R' >= R satisfy both constraints.

    shell_R must be sorted ascending. Returns +inf when no R works.
    """
    sR = np.asarray(shell_R)
    inf_ok = np.asarray(inward_frac) >= thresh
    mag_ok = np.asarray(mag_mean) >= floor
    ok = inf_ok & mag_ok

    if not ok.any():
        return math.inf
    # Largest contiguous tail of True at the end.
    tail_start = len(ok)
    for i in range(len(ok) - 1, -1, -1):
        if not ok[i]:
            break
        tail_start = i
    if tail_start == len(ok):
        return math.inf
    return float(sR[tail_start])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--risk-json", type=Path,
                   default=Path("results/LOR63/step1_nrbf_sweep/"
                                "risk_surface/risk_surface.json"))
    p.add_argument("--thresh-list", type=float, nargs="+",
                   default=[0.5, 0.6, 0.7],
                   help="inward-fraction thresholds to test")
    p.add_argument("--floor-frac-list", type=float, nargs="+",
                   default=[0.01, 0.05, 0.10],
                   help="magnitude floors as fractions of "
                        "<||f_truth||>_attractor")
    args = p.parse_args()

    risk = json.loads(args.risk_json.read_text())
    out_dir = args.risk_json.parent
    f_truth_attr = risk["config"]["f_truth_mean_mag"]

    print(f"Loaded {args.risk_json}")
    print(f"<||f_truth||>_attractor = {f_truth_attr:.3f}")

    # Truth absorbing radius for context
    truth_R_abs = {}
    for thresh in args.thresh_list:
        truth_R_abs[thresh] = {}
        for ff in args.floor_frac_list:
            floor = ff * f_truth_attr
            truth_R_abs[thresh][ff] = r_abs(
                risk["truth"]["shell_R"],
                risk["truth"]["R_inward_frac"],
                risk["truth"]["mag_mean"],
                thresh=thresh, floor=floor,
            )

    # Per-model R_abs across the threshold/floor grid
    records = []
    for m in risk["models"]:
        entry = {"n_rbf": m["n_rbf"], "R_abs": {}}
        for thresh in args.thresh_list:
            entry["R_abs"][f"{thresh:.2f}"] = {}
            for ff in args.floor_frac_list:
                floor = ff * f_truth_attr
                R = r_abs(m["shell_R"], m["R_inward_frac"], m["rbf_mag_mean"],
                          thresh=thresh, floor=floor)
                entry["R_abs"][f"{thresh:.2f}"][f"{ff:.2f}"] = R
        records.append(entry)

    # -- Console table -------------------------------------------------------
    print("\nR_abs(thresh, floor as fraction of <||f_truth||>_attr):")
    print("Truth (analytic) R_abs:")
    print(f"  {'thresh':>8} {'ff=0.01':>10} {'ff=0.05':>10} {'ff=0.10':>10}")
    for thresh in args.thresh_list:
        row = f"  {thresh:>8.2f}"
        for ff in args.floor_frac_list:
            R = truth_R_abs[thresh][ff]
            row += f" {('inf' if math.isinf(R) else f'{R:.2f}'):>10}"
        print(row)
    print("\nPer-model R_abs:")
    print(f"  {'n_rbf':>6} {'thresh':>8} {'ff=0.01':>10} {'ff=0.05':>10} "
          f"{'ff=0.10':>10}")
    for rec in records:
        first = True
        for thresh in args.thresh_list:
            row = (f"  {rec['n_rbf'] if first else '':>6}"
                   f" {thresh:>8.2f}")
            for ff in args.floor_frac_list:
                R = rec["R_abs"][f"{thresh:.2f}"][f"{ff:.2f}"]
                row += f" {('inf' if math.isinf(R) else f'{R:.2f}'):>10}"
            print(row)
            first = False

    # -- Persist JSON --------------------------------------------------------
    out_json = out_dir / "outer_absorbing.json"
    payload = {
        "config": {
            "risk_json": str(args.risk_json),
            "thresh_list": list(args.thresh_list),
            "floor_frac_list": list(args.floor_frac_list),
            "f_truth_mean_mag": f_truth_attr,
        },
        "truth_R_abs": {f"{t:.2f}": {f"{ff:.2f}": ("inf" if math.isinf(
                            truth_R_abs[t][ff]) else truth_R_abs[t][ff])
                                     for ff in args.floor_frac_list}
                        for t in args.thresh_list},
        "models": [
            {
                "n_rbf": r["n_rbf"],
                "R_abs": {t: {ff: ("inf" if math.isinf(v) else v)
                              for ff, v in inner.items()}
                          for t, inner in r["R_abs"].items()},
            }
            for r in records
        ],
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_json}")

    # -- Figure --------------------------------------------------------------
    canonical_thresh = 0.6
    canonical_ff = 0.05
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

    cmap = plt.get_cmap("viridis")
    n_models = len(records)
    colours = [cmap(t) for t in np.linspace(0.05, 0.95, n_models)]

    # Left: bar of R_abs at canonical (0.6, 0.05) per model
    ax = axes[0]
    n_rbf_vals = [r["n_rbf"] for r in records]
    R_canon = [
        r["R_abs"][f"{canonical_thresh:.2f}"][f"{canonical_ff:.2f}"]
        for r in records
    ]
    cap = max(risk["truth"]["shell_R"])
    R_canon_clip = [
        cap + 0.5 if math.isinf(R) else R for R in R_canon
    ]
    bars = ax.bar(np.arange(n_models), R_canon_clip, color=colours)
    for i, (R, R_disp) in enumerate(zip(R_canon, R_canon_clip)):
        tag = "no absorbing tail" if math.isinf(R) else f"{R:.2f}"
        ax.text(i, R_disp + 0.1, tag, ha="center", va="bottom",
                fontsize=8, rotation=90 if math.isinf(R) else 0,
                fontweight="bold" if math.isinf(R) else "normal",
                color="crimson" if math.isinf(R) else "black")
    truth_R = risk["truth"]["R_inward_frac"]
    truth_mag = risk["truth"]["mag_mean"]
    truth_R_canon = r_abs(risk["truth"]["shell_R"], truth_R, truth_mag,
                          thresh=canonical_thresh,
                          floor=canonical_ff * f_truth_attr)
    if not math.isinf(truth_R_canon):
        ax.axhline(truth_R_canon, color="k", lw=1.3, ls="--",
                   label=rf"truth $R_{{\rm abs}} = {truth_R_canon:.2f}$")
    ax.set_xticks(np.arange(n_models))
    ax.set_xticklabels([str(n) for n in n_rbf_vals])
    ax.set_xlabel(r"$n_{\rm rbf}$")
    ax.set_ylabel(rf"$R_{{\rm abs}} / \sigma_{{\rm attr}}$")
    ax.set_title(rf"$R_{{\rm abs}}$ at "
                 rf"$\rho_{{\rm in}} \geq {canonical_thresh}$, "
                 rf"$\|f\|/\langle\|f_{{\rm truth}}\|\rangle "
                 rf"\geq {canonical_ff:g}$"
                 "\n(smaller = absorbing kicks in earlier; better)")
    ax.grid(True, axis="y", alpha=0.3)
    if not math.isinf(truth_R_canon):
        ax.legend(fontsize=8, loc="upper left")

    # Right: heatmap of R_abs over (thresh, floor) grid per model
    ax = axes[1]
    # Stack as: rows = (n_rbf, ff), cols = thresh; but a single
    # composite "no-absorbing-tail count" per model is more useful.
    n_thresh = len(args.thresh_list)
    n_ff = len(args.floor_frac_list)
    # Robust scalar: total fraction of (thresh, ff) cells that yield a
    # finite R_abs <= max sampled shell. 1.0 = absorbing for every
    # criterion; 0.0 = no absorbing under any criterion.
    robustness = []
    for r in records:
        n_ok = 0
        for t in args.thresh_list:
            for ff in args.floor_frac_list:
                R = r["R_abs"][f"{t:.2f}"][f"{ff:.2f}"]
                if not math.isinf(R):
                    n_ok += 1
        robustness.append(n_ok / (n_thresh * n_ff))
    ax.bar(np.arange(n_models), robustness, color=colours)
    for i, v in enumerate(robustness):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(np.arange(n_models))
    ax.set_xticklabels([str(n) for n in n_rbf_vals])
    ax.set_xlabel(r"$n_{\rm rbf}$")
    ax.set_ylabel("absorbing-tail robustness")
    ax.set_title("Fraction of (thresh, floor) criteria with finite "
                 r"$R_{\rm abs}$"
                 "\n(higher = absorbing under more criteria)")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("L63 RBF-only: outer-absorbing-radius diagnostic", y=1.02)
    fig.tight_layout()
    out_fig = out_dir / "outer_absorbing_summary.png"
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_fig}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
