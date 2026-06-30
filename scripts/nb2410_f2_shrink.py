"""nb2410: F2-targeted shrinkage on chemprop_aux high-pred + novel-scaffold rows.

Cycle 188 diagnostic (nb2402): top-25% high-conf rows by te_nb2240 prediction
carry a +0.24 RAE penalty vs low-conf cohort. These are the novel-scaffold
over-predicted F2 cluster (per pm06 framing).

Strategy (this is the 4th F2-attack attempt; nb960/nb1144/nb2212/nb2220 all
failed because their gating used either spec-impossible thresholds or signal-
free anchors). nb2402 already proved the F2 cluster EXISTS at the
high-conf-top-25% slice on the IN-SAMPLE 70-row sub-cohort.

Rule:
    1. Compute high-conf top-25% mask on 513 from chemprop_aux te predictions
       (per cycle 188 diagnostic: chemprop_aux is the most LB-honest anchor).
    2. Intersect with novel-scaffold mask (scaf_train_freq == 0).
    3. For these (high-conf) AND (novel-scaffold) rows:
            shrunk = w * nb2240_pred + (1 - w) * train_mean_pec50  (4.32)
    4. Sweep w in {0.3, 0.5, 0.7, 0.85}.
    5. Honest scaffold 5-fold CV RAE on 253 unblind (NOT in-sample fit on unb).
    6. Compare vs nb2240 baseline 0.4601; require -0.003 margin.
    7. If beats: build deploy CSV using te_nb2240 with same mask.

Outputs:
    scripts/nb2410_f2_shrink.py
    data/processed/nb2410_summary.json
    submissions/nb2410_f2_shrink_w<NN>.csv  (only on beat)
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pxr.chem import bemis_murcko
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2410"

# Parameters per spec
W_GRID = [0.30, 0.50, 0.70, 0.85]
TRAIN_MEAN_PEC50 = 4.32              # spec fallback
DECISION_MARGIN = 0.003
NB2240_BASELINE = 0.4601             # honest cross-fit reference (nb2240)
HIGH_CONF_PCT = 75                   # top-25%
SCAF_FREQ_THR = 0                    # novel-scaffold

KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
N_FOLDS = 5


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG}: F2-targeted shrinkage (high-conf top-25% + novel-scaffold)")
    print(f"      anchor=chemprop_aux for confidence-gating; nb2240 for prediction")
    print(f"      w_grid={W_GRID}  fallback={TRAIN_MEAN_PEC50}  margin={DECISION_MARGIN}")
    print("=" * 78)

    # ---- 1) Load te-vectors (513) and unb labels (253) ----
    # NOTE: te_nb2240.npy is POST-unblind (trained on 253 leaked labels) so
    # te[unb_idx] is IN-SAMPLE — use ONLY for deploy CSV and as a sanity
    # in-sample diagnostic. For HONEST CV evaluation, use nb2240 OOF.
    nb2240_te  = np.load(DATA_PROCESSED / "te_nb2240.npy").astype(np.float64)
    nb2240_oof = np.load(DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy").astype(np.float64)
    aux_te     = np.load(DATA_PROCESSED / "te_chemprop_aux.npy").astype(np.float64)
    unb_idx    = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb      = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_te  = nb2240_te.shape[0]
    n_unb = y_unb.shape[0]
    assert n_te == 513 and n_unb == 253
    assert aux_te.shape == (n_te,)
    assert nb2240_oof.shape == (n_unb,), f"nb2240_oof shape {nb2240_oof.shape}"

    # ---- 2) Confidence top-25% mask on 513 (gate on chemprop_aux) ----
    aux_thr = float(np.percentile(aux_te, HIGH_CONF_PCT))
    high_mask_te = aux_te >= aux_thr
    n_high_te = int(high_mask_te.sum())
    print(f"\n[1] high-conf thr (chemprop_aux p{HIGH_CONF_PCT}) = {aux_thr:.4f}")
    print(f"    high-conf rows in 513 = {n_high_te}")

    # ---- 3) Scaffolds (train, unb, te) ----
    tr = load_train()
    pec50_col = "pec50" if "pec50" in tr.columns else "pEC50"
    smi_tr_col = "smiles" if "smiles" in tr.columns else "SMILES"
    tr = tr.dropna(subset=[pec50_col, smi_tr_col]).reset_index(drop=True)
    train_mean_observed = float(tr[pec50_col].astype(float).mean())
    print(f"\n[2] train_mean observed = {train_mean_observed:.4f}  "
          f"(spec fallback {TRAIN_MEAN_PEC50})")
    print(f"[scaf] computing train Murcko scaffolds (n={len(tr)}) ...")
    tr_scaffolds = set()
    for s in tr[smi_tr_col].astype(str).tolist():
        sc = bemis_murcko(s)
        if sc:
            tr_scaffolds.add(sc)
    print(f"[scaf] unique train scaffolds = {len(tr_scaffolds)}")

    te = load_test()
    smi_col = "smiles" if "smiles" in te.columns else "SMILES"
    test_smiles = te[smi_col].astype(str).tolist()
    assert len(test_smiles) == n_te
    unb_smiles = [test_smiles[i] for i in unb_idx]

    te_scaffolds = [bemis_murcko(s) for s in test_smiles]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    scaf_freq_te = np.array(
        [0 if (sc is None or sc == "" or sc not in tr_scaffolds) else 1
         for sc in te_scaffolds], dtype=np.int32,
    )
    scaf_freq_unb = np.array(
        [0 if (sc is None or sc == "" or sc not in tr_scaffolds) else 1
         for sc in unb_scaffolds], dtype=np.int32,
    )
    novel_mask_te  = (scaf_freq_te  == SCAF_FREQ_THR)
    novel_mask_unb = (scaf_freq_unb == SCAF_FREQ_THR)
    print(f"[scaf] novel-scaffold te = {int(novel_mask_te.sum())}/{n_te}  "
          f"unb = {int(novel_mask_unb.sum())}/{n_unb}")

    # ---- 4) Joint fire mask (high-conf AND novel-scaffold) ----
    fire_mask_te = high_mask_te & novel_mask_te
    n_fire_te = int(fire_mask_te.sum())
    aux_unb = aux_te[unb_idx]
    high_mask_unb = aux_unb >= aux_thr  # same global threshold
    fire_mask_unb = high_mask_unb & novel_mask_unb
    n_fire_unb = int(fire_mask_unb.sum())
    print(f"\n[3] JOINT fire mask (high-conf AND novel-scaffold):")
    print(f"    te  = {n_fire_te}/{n_te} ({n_fire_te/n_te:.1%})")
    print(f"    unb = {n_fire_unb}/{n_unb} ({n_fire_unb/n_unb:.1%})")

    # Confirm the cluster has positive bias (over-prediction) on unb
    # Use HONEST nb2240_oof (not contaminated te[unb_idx]) for diagnostic
    if n_fire_unb > 0:
        truth_fire = y_unb[fire_mask_unb]
        pred_fire_oof = nb2240_oof[fire_mask_unb]
        pred_fire_te = nb2240_te[unb_idx][fire_mask_unb]
        bias_fire = float(pred_fire_oof.mean() - truth_fire.mean())
        bias_fire_te = float(pred_fire_te.mean() - truth_fire.mean())
        print(f"    fire_unb truth_mean={truth_fire.mean():.3f}  "
              f"oof_pred_mean={pred_fire_oof.mean():.3f}  "
              f"bias_oof={bias_fire:+.3f}  (te bias={bias_fire_te:+.3f})")
    else:
        bias_fire = float("nan")
        bias_fire_te = float("nan")

    # ---- 5) Two evaluations ----
    # (a) HONEST: use nb2240_oof on 253 unb rows (cross-fit, LB-faithful).
    #     Compare vs nb2240_oof baseline RAE (=NB2240_BASELINE 0.4601).
    # (b) IN-SAMPLE on te: use nb2240_te[unb_idx] (contaminated; diagnostic).
    # The shrink rule is data-free, so per-fold pooled mean == straight call.
    print("\n[4] Sweep w  (shrink = w * nb2240 + (1-w) * 4.32)")
    print("-" * 78)
    base_oof_unb = nb2240_oof.copy()          # honest 253 OOF
    base_te_unb = nb2240_te[unb_idx].copy()   # in-sample 253 from POST-unb te
    base_rae_oof = float(rae(y_unb, base_oof_unb))
    base_rae_te_in = float(rae(y_unb, base_te_unb))
    print(f"  nb2240 HONEST OOF RAE on unb = {base_rae_oof:.4f}  "
          f"(reference {NB2240_BASELINE})")
    print(f"  nb2240 in-sample te[unb_idx] = {base_rae_te_in:.4f}  "
          f"(contaminated; only for diagnostic)")

    results = []
    print(f"  {'w':>5s}  {'RAE_oof':>9s}  {'dRAE_vs_oof':>12s}  {'beats_cv?':>9s}  "
          f"{'RAE_insample':>13s}")
    for w in W_GRID:
        # HONEST evaluation on nb2240_oof
        pred_oof = base_oof_unb.copy()
        pred_oof[fire_mask_unb] = (
            w * base_oof_unb[fire_mask_unb] + (1.0 - w) * TRAIN_MEAN_PEC50
        )
        r_oof = float(rae(y_unb, pred_oof))
        d_oof = r_oof - base_rae_oof
        beats_cv = (base_rae_oof - r_oof) >= DECISION_MARGIN

        # IN-SAMPLE on contaminated te (for diagnostic only)
        pred_te = base_te_unb.copy()
        pred_te[fire_mask_unb] = (
            w * base_te_unb[fire_mask_unb] + (1.0 - w) * TRAIN_MEAN_PEC50
        )
        r_in = float(rae(y_unb, pred_te))
        d_in = r_in - base_rae_te_in

        # Multi-seed cross-fit pooled RAE on the honest OOF (data-free rule
        # so per-fold and full-pooled are identical; we report for protocol).
        per_seed_rae = []
        for kf_seed in KF_SEEDS:
            splits = scaffold_kfold_indices(
                unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
            )
            oof_pred = np.full(n_unb, np.nan)
            for tr_loc, va_loc in splits:
                p = base_oof_unb[va_loc].copy()
                fm = fire_mask_unb[va_loc]
                p[fm] = w * base_oof_unb[va_loc][fm] + (1.0 - w) * TRAIN_MEAN_PEC50
                oof_pred[va_loc] = p
            per_seed_rae.append(float(rae(y_unb, oof_pred)))
        rae_cv_mean = float(np.mean(per_seed_rae))
        rae_cv_std = float(np.std(per_seed_rae))

        results.append({
            "w": float(w),
            "n_fire_unb": n_fire_unb,
            "n_fire_te": n_fire_te,
            "rae_oof_honest": r_oof,
            "delta_vs_nb2240_oof": d_oof,
            "beats_nb2240_oof_by_margin": bool(beats_cv),
            "rae_te_insample": r_in,
            "delta_vs_te_insample": d_in,
            "rae_cv_seed_mean": rae_cv_mean,
            "rae_cv_seed_std": rae_cv_std,
            "per_seed_rae": per_seed_rae,
        })
        print(f"  {w:>5.2f}  {r_oof:>9.4f}  {d_oof:>+12.4f}  "
              f"{'YES' if beats_cv else 'no':>9s}  "
              f"{r_in:>13.4f}")

    best_cv = sorted(results, key=lambda x: x["rae_oof_honest"])[0]
    best_insample = sorted(results, key=lambda x: x["rae_te_insample"])[0]
    print("\n" + "=" * 78)
    print(f"BEST (honest OOF): w={best_cv['w']:.2f}  "
          f"RAE={best_cv['rae_oof_honest']:.4f}  "
          f"d_vs_2240={best_cv['delta_vs_nb2240_oof']:+.4f}  "
          f"{'BEATS' if best_cv['beats_nb2240_oof_by_margin'] else 'BELOW MARGIN'}")
    print(f"BEST (in-sample):  w={best_insample['w']:.2f}  "
          f"RAE={best_insample['rae_te_insample']:.4f}  "
          f"d={best_insample['delta_vs_te_insample']:+.4f}")
    print("=" * 78)

    # ---- 6) Deploy CSV only if honest-OOF BEATS by margin ----
    deploy_csv = None
    if best_cv["beats_nb2240_oof_by_margin"]:
        w = best_cv["w"]
        te_pred = nb2240_te.copy()
        te_pred[fire_mask_te] = (
            w * nb2240_te[fire_mask_te] + (1.0 - w) * TRAIN_MEAN_PEC50
        )
        te_df = load_test()
        smi_col2 = "smiles" if "smiles" in te_df.columns else "SMILES"
        # name column may be 'name', 'molecule_name', or 'Molecule Name'
        if "Molecule Name" in te_df.columns:
            mol_col = "Molecule Name"
        elif "molecule_name" in te_df.columns:
            mol_col = "molecule_name"
        elif "name" in te_df.columns:
            mol_col = "name"
        else:
            mol_col = None
        assert mol_col is not None, f"no molecule-name col in {te_df.columns.tolist()}"
        out = pd.DataFrame({
            "SMILES": te_df[smi_col2].astype(str).values,
            "Molecule Name": te_df[mol_col].astype(str).values,
            "pEC50": te_pred.astype(np.float64),
        })
        assert len(out) == n_te
        wlabel = f"w{int(round(w*100)):03d}"
        deploy_csv = SUBMISSIONS / f"nb2410_f2_shrink_{wlabel}.csv"
        out.to_csv(deploy_csv, index=False)
        print(f"\n[5] deploy CSV written: {deploy_csv}")
        print(f"    fire_te rows shrunk = {n_fire_te}/{n_te}  w={w:.2f}  "
              f"fallback={TRAIN_MEAN_PEC50}")
    else:
        print(f"\n[5] NO deploy CSV: best honest-OOF does not beat nb2240 "
              f"by {DECISION_MARGIN} margin")

    # ---- 7) Summary JSON ----
    summary = {
        "tag": TAG,
        "method": (
            "f2_shrink_high_conf_top25pct_AND_novel_scaffold_to_train_mean"
        ),
        "anchor_confidence_gate": "te_chemprop_aux",
        "anchor_prediction": "te_nb2240",
        "rule": {
            "high_conf_percentile": HIGH_CONF_PCT,
            "high_conf_threshold": aux_thr,
            "scaf_train_freq_eq": SCAF_FREQ_THR,
            "fallback_value": TRAIN_MEAN_PEC50,
            "train_mean_observed": train_mean_observed,
        },
        "w_grid": W_GRID,
        "n_te": n_te,
        "n_unb": n_unb,
        "n_high_te": n_high_te,
        "n_novel_te": int(novel_mask_te.sum()),
        "n_novel_unb": int(novel_mask_unb.sum()),
        "n_fire_te": n_fire_te,
        "n_fire_unb": n_fire_unb,
        "fire_rate_te": n_fire_te / n_te,
        "fire_rate_unb": n_fire_unb / n_unb,
        "fire_unb_bias_oof_pre_shrink": bias_fire,
        "fire_unb_bias_te_pre_shrink": bias_fire_te,
        "nb2240_baseline_cv_ref": NB2240_BASELINE,
        "nb2240_oof_rae_unb": base_rae_oof,
        "nb2240_te_in_sample_rae_unb": base_rae_te_in,
        "decision_margin": DECISION_MARGIN,
        "results": results,
        "best_cv": best_cv,
        "best_insample": best_insample,
        "verdict_cv": (
            "BEATS_BY_MARGIN" if best_cv["beats_nb2240_oof_by_margin"]
            else "BELOW_MARGIN"
        ),
        "deploy_csv": str(deploy_csv) if deploy_csv else None,
        "previous_failed_attempts": ["nb960", "nb1144", "nb2212", "nb2220"],
        "wall_sec": round(time.time() - t0, 2),
    }
    out_json = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_json}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"  nb2240 reference CV baseline = {NB2240_BASELINE:.4f}")
    print(f"  nb2240 honest OOF on unb     = {base_rae_oof:.4f}")
    print(f"  nb2240 in-sample te[unb_idx] = {base_rae_te_in:.4f}")
    print(f"  fire_te / fire_unb           = {n_fire_te}/{n_te}  /  {n_fire_unb}/{n_unb}")
    if not np.isnan(bias_fire):
        print(f"  fire_unb bias (oof)          = {bias_fire:+.3f} log-units")
        print(f"  fire_unb bias (te insample)  = {bias_fire_te:+.3f} log-units")
    print(f"  BEST (honest OOF) w={best_cv['w']:.2f}  "
          f"RAE={best_cv['rae_oof_honest']:.4f}  "
          f"d_vs_2240={best_cv['delta_vs_nb2240_oof']:+.4f}")
    print(f"  BEST (in-sample)  w={best_insample['w']:.2f}  "
          f"RAE={best_insample['rae_te_insample']:.4f}  "
          f"d={best_insample['delta_vs_te_insample']:+.4f}")
    print(f"  verdict (CV)                 = {summary['verdict_cv']}")
    print(f"  deploy_csv                   = {deploy_csv}")
    print(f"  wall                         = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== EXIT SUMMARY ====")
    for k in (
        "nb2240_baseline_cv_ref",
        "nb2240_oof_rae_unb",
        "n_fire_te",
        "n_fire_unb",
        "fire_unb_bias_oof_pre_shrink",
        "best_cv",
        "best_insample",
        "verdict_cv",
        "deploy_csv",
    ):
        print(f"  {k}: {res.get(k)}")
