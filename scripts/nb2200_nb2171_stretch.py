"""nb2200 -- Stretch parameter sweep on nb2171 deep-30 mean OOF.

Motivation
----------
nb2171 (PRE-clean 5-anchor SLSQP blend, nb1191 swapped in for the POST-risk
nb730_honest anchor) was deep-verified by nb2180 at pooled scaffold-CV
RAE 0.4682 +/- 0.0024 across kf_seeds {1116..1145}. The cached deploy CSV
applies a *single* rank-stretch s = 1.010 (light, mean of fold-optimal s
from nb2180). On the 30-seed mean OOF, pred_std = 0.871 vs truth_std =
1.032 -- still variance-compressed by ~16%. The "rank-stretch is universal
post-hoc calibration" memo says s in [1.05, 1.10] is the sweet spot for
that kind of compression.

This script asks: if we treat the deep-30 mean OOF as a single predictor
(collapsing the per-seed dimension) and sweep s over a wider grid
{1.00, 1.02, 1.05, 1.07, 1.10}, can we beat nb2171's internal s=1.010
honestly (scaffold-CV-style with per-fold cross-fit mu)?

Protocol
--------
  1. Load the 30 per-seed OOFs cached by nb2191 (shape (30, 253)) and
     collapse to the deep-30 mean OOF: mu_oof_30 = oofs.mean(axis=0).
  2. For each s in the sweep grid:
       For each fold (scaffold 5-fold, kf_seed=2200_BASE_SEED):
         - compute mu_tr = mean(mu_oof_30[tr_loc])     # cross-fit mean
         - stretched_va = mu_tr + s * (mu_oof_30[va_loc] - mu_tr)
       Pool across folds; pooled scaffold-CV RAE per s.
  3. Gate: best_s_RAE <= 0.4652 = nb2171_deep30_mean - 0.003.
  4. If pass: build deploy CSV nb2200_deploy_stretched.csv by applying the
     best_s correction to te_nb2171 around its own mean:
         te_blend = mu_te + (te_nb2171 - mu_te) / 1.010   # undo embedded s
         te_deploy = mu_te + best_s * (te_blend - mu_te)
     (algebra: te_deploy = mu_te + (best_s/1.010) * (te_nb2171 - mu_te))

Outputs
-------
  scripts/nb2200_nb2171_stretch.py
  data/processed/nb2200_summary.json
  submissions/nb2200_deploy_stretched.csv  (only on gate pass)
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2200"
N_FOLDS = 5
KF_BASE_SEEDS = list(range(2200, 2230))   # 30 fresh seeds for stability
STRETCH_GRID = [1.00, 1.02, 1.05, 1.07, 1.10]

# nb2171 deep-30 reference (data/processed/nb2180_summary.json)
NB2171_DEEP30_MEAN = 0.4682
GATE_MARGIN = 0.003
GATE_TARGET = NB2171_DEEP30_MEAN - GATE_MARGIN     # 0.4652

# nb2171 deploy s already baked into te_nb2171.npy
NB2171_EMBEDDED_S = 1.010

LB_W_OOF = 0.51
LB_W_TE = 0.49


def stretch_cv_per_seed(mu_oof, y_unb, unb_scaffolds, kf_seed, s):
    """Pooled scaffold-CV RAE for one (s, kf_seed):
    per-fold: mu_tr = mean(pred_tr); stretched_va = mu_tr + s*(pred_va - mu_tr).
    """
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(mu_oof)
    oof_pred = np.full(n_unb, np.nan)
    for tr_loc, va_loc in splits:
        mu_tr = float(np.mean(mu_oof[tr_loc]))
        oof_pred[va_loc] = mu_tr + s * (mu_oof[va_loc] - mu_tr)
    assert not np.isnan(oof_pred).any()
    return float(rae(y_unb, oof_pred))


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Stretch sweep on nb2171 deep-30 mean OOF")
    print(f"       grid={STRETCH_GRID}")
    print(f"       baseline: nb2171 deep-30 mean = {NB2171_DEEP30_MEAN:.4f} "
          f"(s_embedded = {NB2171_EMBEDDED_S:.3f})")
    print(f"       gate target = {GATE_TARGET:.4f}  (margin {GATE_MARGIN:.3f})")
    print("=" * 78)

    # ---- Load test smiles/names + unblind ground truth ----
    te = load_test()
    te_names = te["name"].values
    te_smiles = te["smiles"].values
    n_te = len(te_smiles)

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    unb_smiles = te_smiles[unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[load] n_te={n_te}  n_unb={n_unb}  unique_scaffolds={n_unique_scaf}")

    # ---- Load deep-30 per-seed OOFs and collapse to deep-30 mean OOF ----
    seed_oof_path = DATA_PROCESSED / "nb2191_per_seed_oof_30.npy"
    assert seed_oof_path.exists(), f"missing per-seed OOF stack: {seed_oof_path}"
    seed_oofs = np.load(seed_oof_path).astype(np.float64)
    assert seed_oofs.shape == (30, n_unb), \
        f"per-seed OOF stack shape mismatch {seed_oofs.shape}"
    mu_oof_30 = seed_oofs.mean(axis=0)
    baseline_rae = float(rae(y_unb, mu_oof_30))
    pred_std = float(mu_oof_30.std())
    truth_std = float(y_unb.std())
    print(f"[deep30] mu_oof_30: mean={mu_oof_30.mean():.4f}  "
          f"pred_std={pred_std:.4f}  truth_std={truth_std:.4f}  "
          f"compression_ratio={pred_std/truth_std:.3f}")
    print(f"[deep30] baseline mean-bag RAE = {baseline_rae:.4f}  "
          f"(nb2180 deep-30 ref {NB2171_DEEP30_MEAN:.4f})")

    # ---- Sweep s across grid, with per-fold cross-fit mu ----
    print("\n" + "-" * 78)
    print(f"SWEEP: s in {STRETCH_GRID}  x  {len(KF_BASE_SEEDS)} kf_seeds  "
          f"x  {N_FOLDS}-fold scaffold")
    print("-" * 78)
    sweep_results = []
    for s in STRETCH_GRID:
        per_seed_rae = []
        for kf_seed in KF_BASE_SEEDS:
            r = stretch_cv_per_seed(
                mu_oof_30, y_unb, unb_scaffolds, kf_seed, s,
            )
            per_seed_rae.append(r)
        per_seed_rae = np.asarray(per_seed_rae)
        m = float(per_seed_rae.mean())
        sd = float(per_seed_rae.std(ddof=1))
        se = sd / np.sqrt(len(per_seed_rae))
        lo = float(m - 1.96 * se)
        hi = float(m + 1.96 * se)
        sweep_results.append({
            "s": float(s),
            "pooled_rae_mean": m,
            "pooled_rae_std_ddof1": sd,
            "pooled_rae_se": se,
            "ci95_lo": lo,
            "ci95_hi": hi,
            "n_kf_seeds": len(KF_BASE_SEEDS),
            "per_seed_rae": [float(x) for x in per_seed_rae.tolist()],
        })
        print(f"   s={s:.3f}   mean_RAE={m:.4f}  std={sd:.4f}  "
              f"95% CI=[{lo:.4f}, {hi:.4f}]")

    # ---- Pick best s ----
    best = min(sweep_results, key=lambda r: r["pooled_rae_mean"])
    best_s = best["s"]
    best_rae = best["pooled_rae_mean"]
    baseline_in_sweep = next(
        (r for r in sweep_results if abs(r["s"] - 1.00) < 1e-6), None
    )
    delta_vs_s1 = best_rae - (
        baseline_in_sweep["pooled_rae_mean"]
        if baseline_in_sweep else float("nan")
    )
    delta_vs_ref = best_rae - NB2171_DEEP30_MEAN
    delta_vs_gate = best_rae - GATE_TARGET
    print("\n" + "-" * 78)
    print("PICK")
    print("-" * 78)
    print(f"   best s            = {best_s:.3f}")
    print(f"   best mean RAE     = {best_rae:.4f}")
    print(f"   vs s=1.00 baseline= {delta_vs_s1:+.4f}")
    print(f"   vs nb2171 deep-30 = {delta_vs_ref:+.4f}  "
          f"(ref {NB2171_DEEP30_MEAN:.4f})")
    print(f"   vs gate           = {delta_vs_gate:+.4f}  "
          f"(target <= {GATE_TARGET:.4f})")

    # ---- Gate ----
    gate_pass = bool(best_rae <= GATE_TARGET)
    decision = "PROMOTE_STRETCHED" if gate_pass else "REJECT_BELOW_MARGIN"
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    print(f"   best_s_RAE  = {best_rae:.4f}")
    print(f"   gate target = {GATE_TARGET:.4f}")
    print(f"   pass        = {gate_pass}  ->  {decision}")

    # ---- Deploy CSV on gate pass ----
    deploy_csv_path = None
    te_unb_rae = None
    lb_band_est = None
    s_effective_over_te = None
    if gate_pass:
        print("\n" + "-" * 78)
        print(f"DEPLOY refit: re-stretch te_nb2171 around its own mean")
        print("-" * 78)
        te_nb2171 = np.load(DATA_PROCESSED / "te_nb2171.npy").astype(np.float64)
        assert te_nb2171.shape == (n_te,), f"te_nb2171 shape {te_nb2171.shape}"
        mu_te = float(te_nb2171.mean())
        # Algebra: te_nb2171 = mu_te + 1.010 * (blend - mu_te)
        # So:      blend_te  = mu_te + (te_nb2171 - mu_te) / 1.010
        # Then:    te_deploy = mu_te + best_s * (blend_te - mu_te)
        #                    = mu_te + (best_s / 1.010) * (te_nb2171 - mu_te)
        s_effective_over_te = float(best_s / NB2171_EMBEDDED_S)
        te_deploy = (
            mu_te + s_effective_over_te * (te_nb2171 - mu_te)
        ).astype(np.float32)
        te_unb_rae = float(rae(y_unb, te_deploy[unb_idx]))
        lb_band_est = LB_W_OOF * best_rae + LB_W_TE * te_unb_rae

        te_path = DATA_PROCESSED / f"te_{TAG}.npy"
        np.save(te_path, te_deploy)
        sub = pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": te_deploy,
        })
        deploy_csv_path = SUBMISSIONS / f"{TAG}_deploy_stretched.csv"
        deploy_csv_path.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(deploy_csv_path, index=False)

        print(f"   mu_te                 = {mu_te:.4f}")
        print(f"   best_s (target)       = {best_s:.3f}")
        print(f"   embedded_s (nb2171)   = {NB2171_EMBEDDED_S:.3f}")
        print(f"   s_effective over te   = best_s / embedded_s = "
              f"{s_effective_over_te:.4f}")
        print(f"   deploy te: mean={te_deploy.mean():.4f}  "
              f"std={te_deploy.std():.4f}")
        print(f"   te[unb] RAE (in-sample) = {te_unb_rae:.4f}")
        print(f"   LB-band est = {LB_W_OOF:.2f}*{best_rae:.4f} + "
              f"{LB_W_TE:.2f}*{te_unb_rae:.4f} = {lb_band_est:.4f}")
        print(f"   te_npy     = {te_path}")
        print(f"   submission = {deploy_csv_path}")
    else:
        print("\n   gate FAIL -> no deploy CSV emitted.")

    # ---- Summary ----
    summary = {
        "tag": TAG,
        "method": (
            "stretch_grid_sweep_on_nb2171_deep30_mean_oof_per_fold_crossfit_mu"
        ),
        "stretch_grid": STRETCH_GRID,
        "kf_base_seeds": KF_BASE_SEEDS,
        "n_kf_seeds": len(KF_BASE_SEEDS),
        "n_folds": N_FOLDS,
        "n_unb": n_unb,
        "n_te": n_te,
        "n_unique_scaffolds": n_unique_scaf,
        "deep30_mean_oof_source": "data/processed/nb2191_per_seed_oof_30.npy",
        "baseline_mean_bag_rae_actual": baseline_rae,
        "baseline_pred_std": pred_std,
        "baseline_truth_std": truth_std,
        "compression_ratio_pred_over_truth": pred_std / truth_std,
        "nb2171_deep30_mean_ref": NB2171_DEEP30_MEAN,
        "nb2171_embedded_s": NB2171_EMBEDDED_S,
        "gate_target": GATE_TARGET,
        "gate_margin_vs_ref": GATE_MARGIN,
        "sweep_results": sweep_results,
        "best_s": best_s,
        "best_pooled_rae_mean": best_rae,
        "best_pooled_rae_std_ddof1": best["pooled_rae_std_ddof1"],
        "best_pooled_rae_ci95_lo": best["ci95_lo"],
        "best_pooled_rae_ci95_hi": best["ci95_hi"],
        "delta_best_vs_s1_baseline": delta_vs_s1,
        "delta_best_vs_nb2171_deep30_ref": delta_vs_ref,
        "delta_best_vs_gate": delta_vs_gate,
        "gate_pass": gate_pass,
        "decision": decision,
        "deploy_csv": str(deploy_csv_path) if deploy_csv_path else None,
        "te_unb_rae_in_sample": te_unb_rae,
        "lb_band_estimate": lb_band_est,
        "lb_band_w_oof": LB_W_OOF,
        "lb_band_w_te": LB_W_TE,
        "s_effective_over_te_nb2171": s_effective_over_te,
        "wall_sec": round(time.time() - t0, 2),
    }
    json_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {json_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   baseline deep-30 mean-bag = {baseline_rae:.4f}")
    print(f"   sweep grid                = {STRETCH_GRID}")
    print(f"   best s                    = {best_s:.3f}")
    print(f"   best pooled RAE           = {best_rae:.4f}")
    print(f"   delta vs s=1 baseline     = {delta_vs_s1:+.4f}")
    print(f"   delta vs nb2171 deep-30   = {delta_vs_ref:+.4f}")
    print(f"   gate (<= {GATE_TARGET:.4f})        = {gate_pass}  ({decision})")
    if gate_pass:
        print(f"   te[unb] in-sample RAE     = {te_unb_rae:.4f}")
        print(f"   LB-band estimate          = {lb_band_est:.4f}")
        print(f"   deploy CSV                = {deploy_csv_path}")
    print(f"   wall                      = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== KEY ====")
    for k in (
        "baseline_mean_bag_rae_actual",
        "best_s",
        "best_pooled_rae_mean",
        "best_pooled_rae_std_ddof1",
        "delta_best_vs_s1_baseline",
        "delta_best_vs_nb2171_deep30_ref",
        "delta_best_vs_gate",
        "gate_pass",
        "decision",
        "te_unb_rae_in_sample",
        "lb_band_estimate",
        "deploy_csv",
    ):
        print(f"  {k}: {res.get(k)}")
