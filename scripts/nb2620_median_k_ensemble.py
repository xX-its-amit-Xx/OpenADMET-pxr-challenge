"""nb2620 -- Per-row MEDIAN ensemble of K-RFE pyramids {K=18, K=20, K=24, K=28}.

NEW PARADIGM (vs nb2604 mean):
    nb2604 used the equal-weight MEAN of the 4 K-RFE residual-LGBM mean-bag
    predictions.  Mean is the L2-optimal aggregator under Gaussian
    cross-anchor noise; per-row MEDIAN is the L1-optimal aggregator and is
    robust to one K being an outlier per row.

    Hypothesis: on the 90% novel-scaffold tail (feedback_failure_mode), one
    K may produce a wild residual prediction on a row where the underlying
    SHAP feature subset does not generalise -- median rejects that single
    outlier per row whereas mean averages it in at weight 0.25.

    df = 0 (no learning), same paradigm-class as nb2604: equal-weight,
    no SLSQP, no rank-stretch.

PROTOCOL:
    1. Load 4 K-RFE OOF arrays from data/processed/:
         K=18 -> nb2604_mean_bag_oof_K18.npy  (rebuilt by nb2604 from nb2263 K_opt)
         K=20 -> nb2240_mean_bag_oof_K20.npy
         K=24 -> nb2310_mean_bag_oof_K24.npy
         K=28 -> nb2103_mean_bag_oof_K28.npy
       And 4 te arrays:
         K=18 -> te_nb2604_K18.npy
         K=20 -> te_nb2240_K20.npy
         K=24 -> te_nb2310_K24.npy
         K=28 -> te_nb2112.npy
    2. Per-row median across 4 -> pred_oof_unb (253-vec) + pred_te (513-vec).
    3. 5-fold scaffold CV on 253 across kf_seeds {1001..1005}.  No learning,
       so each kf_seed is deterministic on the input median vector.

GATE:
    mean_rae < 0.4570 -> PROMOTE
    mean_rae < 0.4580 -> BETTER_THAN_NB2604
    else              -> FAIL

Outputs:
    scripts/nb2620_median_k_ensemble.py
    data/processed/nb2620_summary.json
    data/processed/nb2620_pred_oof.npy   (253,) float32
    data/processed/te_nb2620.npy         (513,) float32
    submissions/nb2620_median_k_ensemble.csv  (on any non-FAIL)
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
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2620"

# ---- Cached K-RFE OOFs + tes ----
K18_OOF_PATH = DATA_PROCESSED / "nb2604_mean_bag_oof_K18.npy"
K18_TE_PATH = DATA_PROCESSED / "te_nb2604_K18.npy"
K20_OOF_PATH = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"
K20_TE_PATH = DATA_PROCESSED / "te_nb2240_K20.npy"
K24_OOF_PATH = DATA_PROCESSED / "nb2310_mean_bag_oof_K24.npy"
K24_TE_PATH = DATA_PROCESSED / "te_nb2310_K24.npy"
K28_OOF_PATH = DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy"
K28_TE_PATH = DATA_PROCESSED / "te_nb2112.npy"

# ---- Anchor for diagnostic ----
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# ---- CV eval ----
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# ---- Gate ----
GATE_PROMOTE = 0.4570
GATE_BETTER = 0.4580

# ---- Refs ----
CHEMPROP_AUX_REF = 0.6216
NB2604_REF = 0.4580   # nb2604 mean ensemble reference
NB2171_REF = 0.4682   # ceiling deep-30 PRIMARY-1


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- per-row MEDIAN ensemble of K-RFE {{18, 20, 24, 28}}")
    print(f"          dual of nb2604 (mean); df = 0; no SLSQP")
    print(f"          ref nb2604 (mean)  = {NB2604_REF:.4f}")
    print(f"          ref nb2171 ceiling = {NB2171_REF:.4f}")
    print("=" * 78)

    # ---- Load truth + anchor + scaffold splits ----
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values
                if "name" in te.columns else te["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] chemprop_aux te[unb_idx] in_RAE = {rae_anchor:.4f} "
          f"(ref {CHEMPROP_AUX_REF:.4f})")

    # ---- Step 1: Load cached K-RFE OOFs + tes ----
    print("\n" + "-" * 78)
    print("STEP 1: load cached K-RFE OOFs + tes for K in {18, 20, 24, 28}")
    print("-" * 78)
    for p in (K18_OOF_PATH, K18_TE_PATH, K20_OOF_PATH, K20_TE_PATH,
              K24_OOF_PATH, K24_TE_PATH, K28_OOF_PATH, K28_TE_PATH):
        if not p.exists():
            raise FileNotFoundError(f"missing cached array: {p}")

    K18_oof = np.load(K18_OOF_PATH).astype(np.float64)
    K18_te = np.load(K18_TE_PATH).astype(np.float64)
    K20_oof = np.load(K20_OOF_PATH).astype(np.float64)
    K20_te = np.load(K20_TE_PATH).astype(np.float64)
    K24_oof = np.load(K24_OOF_PATH).astype(np.float64)
    K24_te = np.load(K24_TE_PATH).astype(np.float64)
    K28_oof = np.load(K28_OOF_PATH).astype(np.float64)
    K28_te = np.load(K28_TE_PATH).astype(np.float64)

    for name, arr, expected in (
        ("K18_oof", K18_oof, n_unb), ("K18_te", K18_te, n_test),
        ("K20_oof", K20_oof, n_unb), ("K20_te", K20_te, n_test),
        ("K24_oof", K24_oof, n_unb), ("K24_te", K24_te, n_test),
        ("K28_oof", K28_oof, n_unb), ("K28_te", K28_te, n_test),
    ):
        if arr.shape != (expected,):
            raise ValueError(f"shape mismatch {name} = {arr.shape}  expected ({expected},)")

    rae_K18 = float(rae(y_unb, K18_oof))
    rae_K20 = float(rae(y_unb, K20_oof))
    rae_K24 = float(rae(y_unb, K24_oof))
    rae_K28 = float(rae(y_unb, K28_oof))
    print(f"   K=18  oof_RAE={rae_K18:.4f}  te_mean={K18_te.mean():.3f}  te_std={K18_te.std():.3f}")
    print(f"   K=20  oof_RAE={rae_K20:.4f}  te_mean={K20_te.mean():.3f}  te_std={K20_te.std():.3f}")
    print(f"   K=24  oof_RAE={rae_K24:.4f}  te_mean={K24_te.mean():.3f}  te_std={K24_te.std():.3f}")
    print(f"   K=28  oof_RAE={rae_K28:.4f}  te_mean={K28_te.mean():.3f}  te_std={K28_te.std():.3f}")

    # ---- Step 2: Per-row MEDIAN ensemble ----
    print("\n" + "-" * 78)
    print("STEP 2: per-row MEDIAN = np.median([K18, K20, K24, K28], axis=0)")
    print("-" * 78)
    P_unb = np.column_stack([K18_oof, K20_oof, K24_oof, K28_oof])  # (253, 4)
    P_te = np.column_stack([K18_te, K20_te, K24_te, K28_te])       # (513, 4)
    # For an even count (4 inputs), numpy.median averages the 2 middle values.
    pred_oof_unb = np.median(P_unb, axis=1)   # 253-vec
    pred_te_513 = np.median(P_te, axis=1)     # 513-vec
    print(f"   P_unb shape = {P_unb.shape}  P_te shape = {P_te.shape}")
    print(f"   pred_oof_unb mean={pred_oof_unb.mean():.3f}  "
          f"std={pred_oof_unb.std():.3f}  (truth_std {y_unb.std():.3f})")
    print(f"   pred_te_513  mean={pred_te_513.mean():.3f}  "
          f"std={pred_te_513.std():.3f}")

    # Diagnostic: compare to mean (nb2604 paradigm)
    pred_mean_unb = P_unb.mean(axis=1)
    pred_mean_te = P_te.mean(axis=1)
    rae_mean_pooled = float(rae(y_unb, pred_mean_unb))
    rae_median_pooled = float(rae(y_unb, pred_oof_unb))
    print(f"\n   [diag] MEAN single-shot pooled_RAE   = {rae_mean_pooled:.4f}  "
          f"(should match nb2604 ref {NB2604_REF:.4f})")
    print(f"   [diag] MEDIAN single-shot pooled_RAE = {rae_median_pooled:.4f}  "
          f"(delta vs mean = {rae_median_pooled - rae_mean_pooled:+.4f})")

    # Per-row disagreement diagnostic: how often does median differ meaningfully from mean?
    delta_per_row = np.abs(pred_oof_unb - pred_mean_unb)
    n_diff_05 = int((delta_per_row > 0.05).sum())
    n_diff_10 = int((delta_per_row > 0.10).sum())
    print(f"   [diag] |median-mean| per row:  max={delta_per_row.max():.3f}  "
          f"mean={delta_per_row.mean():.3f}  "
          f">0.05 in {n_diff_05}/{n_unb} rows  "
          f">0.10 in {n_diff_10}/{n_unb} rows")

    # Pair-wise OOF correlations (for the record)
    corr_mat = np.corrcoef(P_unb.T)
    K_labels = ["K18", "K20", "K24", "K28"]
    print(f"\n   K-pyramid OOF correlation matrix (4x4):")
    print(f"        {'  '.join([f'{k:>6s}' for k in K_labels])}")
    for i, ki in enumerate(K_labels):
        row = "  ".join([f"{corr_mat[i, j]:6.3f}" for j in range(4)])
        print(f"   {ki:>5s}  {row}")

    # ---- Step 3: 5-fold scaffold CV (deterministic, no learning) ----
    print("\n" + "-" * 78)
    print(f"STEP 3: 5-fold scaffold CV  kf_seeds={KF_SEEDS}  n_folds={N_FOLDS}")
    print("-" * 78)
    per_seed_pooled = []
    per_seed_fold_rae = []
    for kf_seed in KF_SEEDS:
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        oof_pooled = np.full(n_unb, np.nan, dtype=np.float64)
        per_fold_rae = []
        for tr_loc, va_loc in splits:
            oof_pooled[va_loc] = pred_oof_unb[va_loc]
            per_fold_rae.append(float(rae(y_unb[va_loc], pred_oof_unb[va_loc])))
        if np.isnan(oof_pooled).any():
            raise RuntimeError(
                "scaffold splits did not cover all rows; check protocol"
            )
        pooled = float(rae(y_unb, oof_pooled))
        per_seed_pooled.append(pooled)
        per_seed_fold_rae.append(per_fold_rae)
        print(f"   kf_seed={kf_seed:5d}  pooled_RAE={pooled:.4f}  "
              f"per_fold_mean={np.mean(per_fold_rae):.4f}  "
              f"per_fold=[" + ", ".join(f"{r:.4f}" for r in per_fold_rae) + "]")

    mean_rae = float(np.mean(per_seed_pooled))
    std_rae = float(np.std(per_seed_pooled))
    print(f"\n[eval] mean pooled RAE across {len(KF_SEEDS)} seeds = "
          f"{mean_rae:.4f} +/- {std_rae:.4f}")

    # ---- Step 4: Gate ----
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_BETTER:
        verdict = "BETTER_THAN_NB2604"
    else:
        verdict = "FAIL"
    print(f"[gate] mean_rae={mean_rae:.4f}  "
          f"thresholds(<{GATE_PROMOTE} PROMOTE / <{GATE_BETTER} BETTER_THAN_NB2604)"
          f"  -> {verdict}")

    # ---- Step 5: Save artifacts ----
    print("\n" + "-" * 78)
    print("STEP 5: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof_unb.astype(np.float32))
    np.save(te_path, pred_te_513.astype(np.float32))
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_median_k_ensemble.csv"
    if verdict != "FAIL":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": pred_te_513.astype(np.float32),
        }).to_csv(sub_csv, index=False)
        print(f"   [save] {sub_csv}")
    else:
        print(f"   [skip submission] verdict=FAIL")

    te_unb_in = float(rae(y_unb, pred_te_513[unb_idx]))
    print(f"\n   te[unb_idx] in-sample RAE = {te_unb_in:.4f}")

    delta_vs_nb2604 = mean_rae - NB2604_REF
    delta_vs_nb2171 = mean_rae - NB2171_REF
    print(f"   delta vs nb2604 (mean) ({NB2604_REF:.4f}) = {delta_vs_nb2604:+.4f}")
    print(f"   delta vs nb2171 ceiling ({NB2171_REF:.4f}) = {delta_vs_nb2171:+.4f}")

    # ---- summary ----
    summary = {
        "tag": TAG,
        "method": "per_row_MEDIAN_ensemble_K18_K20_K24_K28_LGBM_no_SLSQP",
        "paradigm": "plain_median_no_learning_df_zero",
        "dual_of": "nb2604_mean_ensemble",
        "anchor_base": "chemprop_aux",
        "anchor_in_rae": rae_anchor,
        "anchor_pre_unblind": True,
        "k18_oof_path": str(K18_OOF_PATH),
        "k18_te_path": str(K18_TE_PATH),
        "k20_oof_path": str(K20_OOF_PATH),
        "k20_te_path": str(K20_TE_PATH),
        "k24_oof_path": str(K24_OOF_PATH),
        "k24_te_path": str(K24_TE_PATH),
        "k28_oof_path": str(K28_OOF_PATH),
        "k28_te_path": str(K28_TE_PATH),
        "per_anchor_rae_in_sample": {
            "K18": rae_K18,
            "K20": rae_K20,
            "K24": rae_K24,
            "K28": rae_K28,
        },
        "oof_corr_matrix": corr_mat.tolist(),
        "oof_corr_labels": K_labels,
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "rae_mean_single_shot": rae_mean_pooled,
        "rae_median_single_shot": rae_median_pooled,
        "median_minus_mean_pooled": rae_median_pooled - rae_mean_pooled,
        "median_vs_mean_row_disagreement": {
            "max_abs_diff": float(delta_per_row.max()),
            "mean_abs_diff": float(delta_per_row.mean()),
            "n_rows_diff_gt_005": n_diff_05,
            "n_rows_diff_gt_010": n_diff_10,
            "n_unb": int(n_unb),
        },
        "per_seed_pooled_rae": per_seed_pooled,
        "per_seed_fold_rae": per_seed_fold_rae,
        "mean_rae": mean_rae,
        "std_rae": std_rae,
        "gate_promote": GATE_PROMOTE,
        "gate_better": GATE_BETTER,
        "verdict": verdict,
        "delta_vs_nb2604_mean": delta_vs_nb2604,
        "nb2604_ref": NB2604_REF,
        "delta_vs_nb2171": delta_vs_nb2171,
        "nb2171_ref": NB2171_REF,
        "te_mean": float(pred_te_513.mean()),
        "te_std": float(pred_te_513.std()),
        "te_unb_in_sample_rae": te_unb_in,
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict != "FAIL" else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   per-anchor OOF RAEs   = "
          f"K18={rae_K18:.4f}  K20={rae_K20:.4f}  "
          f"K24={rae_K24:.4f}  K28={rae_K28:.4f}")
    print(f"   single-shot pooled    = MEAN {rae_mean_pooled:.4f}  "
          f"MEDIAN {rae_median_pooled:.4f}  "
          f"(delta {rae_median_pooled - rae_mean_pooled:+.4f})")
    print(f"   per-seed pooled RAE   = "
          f"{[round(r, 4) for r in per_seed_pooled]}")
    print(f"   MEAN  pooled RAE      = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   gate                  = <{GATE_PROMOTE} PROMOTE / "
          f"<{GATE_BETTER} BETTER_THAN_NB2604  -> {verdict}")
    print(f"   delta vs nb2604       = {delta_vs_nb2604:+.4f}")
    print(f"   delta vs nb2171       = {delta_vs_nb2171:+.4f}")
    print(f"   te[unb_idx] RAE       = {te_unb_in:.4f}")
    print(f"   wall                  = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "rae_mean_single_shot",
        "rae_median_single_shot",
        "median_minus_mean_pooled",
        "mean_rae",
        "std_rae",
        "verdict",
        "delta_vs_nb2604_mean",
        "te_unb_in_sample_rae",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
