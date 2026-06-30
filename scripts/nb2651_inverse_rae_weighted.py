"""nb2651 -- Inverse-RAE weighted ensemble of K-RFE pyramids {K=18, K=20, K=24, K=28}.

NEW PARADIGM:
    nb2604 used PLAIN equal-weight average (w = 0.25 each).  HERE we test the
    natural generalization: weight each K-pyramid by 1 / RAE_K, normalized to
    sum 1.  Better K gets more weight, no SLSQP simplex learning required.

    The weights are computed PER FOLD using only training-fold RAE (no peeking
    at validation-fold truth).  At inference we apply the same weight schema
    across all 4 anchors.  Total degrees of freedom = 0 (no learnable params;
    1/RAE is a deterministic function of training-fold residual fit).

HYPOTHESIS:
    If the 4 K-pyramids have distinguishable per-fold RAEs (some K better in
    one fold than another), 1/RAE weighting will route mass toward the
    fold-specific winner without paying SLSQP simplex df cost.  If the K-band
    is genuinely flat, weights stay near 0.25 each and recover nb2604.

PROTOCOL:
    1. Load 4 cached K-RFE OOFs + tes:
         K=18 -> nb2604_mean_bag_oof_K18.npy + te_nb2604_K18.npy
         K=20 -> nb2240_mean_bag_oof_K20.npy + te_nb2240_K20.npy
         K=24 -> nb2310_mean_bag_oof_K24.npy + te_nb2310_K24.npy
         K=28 -> nb2103_mean_bag_oof_K28.npy + te_nb2112.npy
    2. 5-fold scaffold CV across 5 kf_seeds {1001..1005}.
    3. Per (kf_seed, fold):
         a. compute RAE_K on TRAINING-fold rows (tr_loc) for each K
         b. weights w_K = (1 / RAE_K_train) / sum_K(1 / RAE_K_train)
         c. apply weights to VALIDATION-fold predictions:
                pred_val = sum_K(w_K * pred_K_OOF[va_loc])
    4. Pool fold predictions back to full 253-vec; compute pooled RAE.
    5. Build te_nb2651 = sum_K(w_avg_K * te_K) using mean-of-folds weights.

GATE:
    mean_rae < 0.4570 -> PROMOTE
    mean_rae < 0.4580 -> BETTER_THAN_NB2604
    else                -> FAIL

Outputs:
    scripts/nb2651_inverse_rae_weighted.py
    data/processed/nb2651_summary.json
    data/processed/nb2651_pred_oof.npy            (253,) float32
    data/processed/te_nb2651.npy                  (513,) float32
    submissions/nb2651_inverse_rae_weighted.csv   (on any non-FAIL)
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
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2651"

# ---- Anchor + cached K-RFE artifacts ----
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

K18_OOF_PATH = DATA_PROCESSED / "nb2604_mean_bag_oof_K18.npy"
K18_TE_PATH = DATA_PROCESSED / "te_nb2604_K18.npy"
K20_OOF_PATH = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"
K20_TE_PATH = DATA_PROCESSED / "te_nb2240_K20.npy"
K24_OOF_PATH = DATA_PROCESSED / "nb2310_mean_bag_oof_K24.npy"
K24_TE_PATH = DATA_PROCESSED / "te_nb2310_K24.npy"
K28_OOF_PATH = DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy"
K28_TE_PATH = DATA_PROCESSED / "te_nb2112.npy"

K_LABELS = ["K18", "K20", "K24", "K28"]

# ---- CV eval ----
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# ---- Gate ----
GATE_PROMOTE = 0.4570
GATE_BETTER = 0.4580

# ---- Refs ----
CHEMPROP_AUX_REF = 0.6216
NB2604_REF = 0.4582      # nb2604 equal-weight reference (typical mean_rae)
NB2171_REF = 0.4682      # prior ceiling deep-30


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- inverse-RAE weighted ensemble of K-RFE {{18, 20, 24, 28}}")
    print(f"          w_K = (1/RAE_K_train) / sum_K(1/RAE_K_train)  per fold")
    print(f"          df = 0  (deterministic function of training-fold RAE)")
    print(f"          ref nb2604 equal-weight = {NB2604_REF:.4f}")
    print(f"          ref nb2171 deep-30      = {NB2171_REF:.4f}")
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

    # ---- Step 1: Load 4 cached K-RFE OOFs + tes ----
    print("\n" + "-" * 78)
    print("STEP 1: load cached K-RFE OOFs + tes for K in {18, 20, 24, 28}")
    print("-" * 78)
    K18_oof = np.load(K18_OOF_PATH).astype(np.float64)
    K18_te = np.load(K18_TE_PATH).astype(np.float64)
    K20_oof = np.load(K20_OOF_PATH).astype(np.float64)
    K20_te = np.load(K20_TE_PATH).astype(np.float64)
    K24_oof = np.load(K24_OOF_PATH).astype(np.float64)
    K24_te = np.load(K24_TE_PATH).astype(np.float64)
    K28_oof = np.load(K28_OOF_PATH).astype(np.float64)
    K28_te = np.load(K28_TE_PATH).astype(np.float64)

    for nm, arr, expected in [
        ("K18_oof", K18_oof, n_unb), ("K18_te", K18_te, n_test),
        ("K20_oof", K20_oof, n_unb), ("K20_te", K20_te, n_test),
        ("K24_oof", K24_oof, n_unb), ("K24_te", K24_te, n_test),
        ("K28_oof", K28_oof, n_unb), ("K28_te", K28_te, n_test),
    ]:
        if arr.shape != (expected,):
            raise ValueError(f"{nm} shape {arr.shape} != ({expected},)")

    rae_K18 = float(rae(y_unb, K18_oof))
    rae_K20 = float(rae(y_unb, K20_oof))
    rae_K24 = float(rae(y_unb, K24_oof))
    rae_K28 = float(rae(y_unb, K28_oof))
    print(f"   K=18  oof_RAE={rae_K18:.4f}  te_mean={K18_te.mean():.3f}  "
          f"te_std={K18_te.std():.3f}")
    print(f"   K=20  oof_RAE={rae_K20:.4f}  te_mean={K20_te.mean():.3f}  "
          f"te_std={K20_te.std():.3f}")
    print(f"   K=24  oof_RAE={rae_K24:.4f}  te_mean={K24_te.mean():.3f}  "
          f"te_std={K24_te.std():.3f}")
    print(f"   K=28  oof_RAE={rae_K28:.4f}  te_mean={K28_te.mean():.3f}  "
          f"te_std={K28_te.std():.3f}")

    P_unb = np.column_stack([K18_oof, K20_oof, K24_oof, K28_oof])  # (253, 4)
    P_te = np.column_stack([K18_te, K20_te, K24_te, K28_te])       # (513, 4)
    print(f"   P_unb shape = {P_unb.shape}  P_te shape = {P_te.shape}")

    corr_mat = np.corrcoef(P_unb.T)
    print(f"\n   K-pyramid OOF correlation matrix (4x4):")
    print(f"        {'  '.join([f'{k:>6s}' for k in K_LABELS])}")
    for i, ki in enumerate(K_LABELS):
        row = "  ".join([f"{corr_mat[i, j]:6.3f}" for j in range(4)])
        print(f"   {ki:>5s}  {row}")

    # ---- Step 2: per-fold inverse-RAE weighting ----
    print("\n" + "-" * 78)
    print(f"STEP 2: 5-fold scaffold CV with per-fold inverse-RAE weights")
    print(f"        kf_seeds={KF_SEEDS}  n_folds={N_FOLDS}")
    print("-" * 78)

    per_seed_pooled = []
    per_seed_fold_rae = []
    per_seed_fold_weights = []  # (n_seeds, n_folds, 4)
    per_seed_pred_oof = []

    for kf_seed in KF_SEEDS:
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        oof_pooled = np.full(n_unb, np.nan, dtype=np.float64)
        per_fold_rae = []
        per_fold_weights = []
        for fold_idx, (tr_loc, va_loc) in enumerate(splits):
            # Compute training-fold RAE per K (no peeking at validation truth)
            train_rae_per_K = np.array([
                rae(y_unb[tr_loc], P_unb[tr_loc, k]) for k in range(4)
            ], dtype=np.float64)
            inv = 1.0 / np.maximum(train_rae_per_K, 1e-9)
            w = inv / inv.sum()
            per_fold_weights.append(w.tolist())

            # Apply to validation-fold predictions
            pred_val = P_unb[va_loc] @ w
            oof_pooled[va_loc] = pred_val
            fold_rae = float(rae(y_unb[va_loc], pred_val))
            per_fold_rae.append(fold_rae)

        if np.isnan(oof_pooled).any():
            raise RuntimeError(
                "scaffold splits did not cover all rows; check protocol"
            )
        pooled = float(rae(y_unb, oof_pooled))
        per_seed_pooled.append(pooled)
        per_seed_fold_rae.append(per_fold_rae)
        per_seed_fold_weights.append(per_fold_weights)
        per_seed_pred_oof.append(oof_pooled.copy())

        avg_w = np.mean(per_fold_weights, axis=0)
        print(f"   kf_seed={kf_seed:5d}  pooled_RAE={pooled:.4f}  "
              f"per_fold_mean={np.mean(per_fold_rae):.4f}  "
              f"avg_w=[" + ", ".join(f"{x:.3f}" for x in avg_w) + "]")

    mean_rae = float(np.mean(per_seed_pooled))
    std_rae = float(np.std(per_seed_pooled))
    print(f"\n[eval] mean pooled RAE across {len(KF_SEEDS)} seeds = "
          f"{mean_rae:.4f} +/- {std_rae:.4f}")

    # Mean OOF predictions across seeds (for saved artifact)
    pred_oof_unb = np.mean(per_seed_pred_oof, axis=0)

    # Mean weights across all (seed, fold) -- used to build deploy te
    all_weights = np.array(
        [w for seed_w in per_seed_fold_weights for w in seed_w],
        dtype=np.float64,
    )  # (n_seeds * n_folds, 4)
    w_deploy = all_weights.mean(axis=0)
    w_deploy = w_deploy / w_deploy.sum()  # re-normalize after mean
    print(f"\n   deploy weights (mean across all folds, normalized):")
    for k_lab, w in zip(K_LABELS, w_deploy):
        print(f"     {k_lab}: {w:.4f}")
    print(f"   weight std across folds: " + ", ".join(
        f"{k_lab}={all_weights[:, i].std():.4f}"
        for i, k_lab in enumerate(K_LABELS)
    ))

    # ---- Step 3: build deploy te ----
    print("\n" + "-" * 78)
    print("STEP 3: build deploy te using mean-of-folds weights")
    print("-" * 78)
    pred_te_513 = P_te @ w_deploy
    print(f"   pred_te_513 mean={pred_te_513.mean():.3f}  "
          f"std={pred_te_513.std():.3f}")

    # ---- Step 4: gate ----
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_BETTER:
        verdict = "BETTER_THAN_NB2604"
    else:
        verdict = "FAIL"
    print(f"\n[gate] mean_rae={mean_rae:.4f}  "
          f"thresholds(<{GATE_PROMOTE} PROMOTE / <{GATE_BETTER} BETTER) "
          f"-> {verdict}")

    # ---- Step 5: save artifacts ----
    print("\n" + "-" * 78)
    print("STEP 5: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof_unb.astype(np.float32))
    np.save(te_path, pred_te_513.astype(np.float32))
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_inverse_rae_weighted.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": pred_te_513.astype(np.float32),
    }).to_csv(sub_csv, index=False)
    print(f"   [save] {sub_csv}")

    te_unb_in = float(rae(y_unb, pred_te_513[unb_idx]))
    print(f"\n   te[unb_idx] in-sample RAE = {te_unb_in:.4f}")

    delta_vs_nb2604 = mean_rae - NB2604_REF
    delta_vs_nb2171 = mean_rae - NB2171_REF
    print(f"   delta vs nb2604 ({NB2604_REF:.4f}) = {delta_vs_nb2604:+.4f}")
    print(f"   delta vs nb2171 ({NB2171_REF:.4f}) = {delta_vs_nb2171:+.4f}")

    # ---- summary ----
    summary = {
        "tag": TAG,
        "method": "inverse_rae_weighted_K18_K20_K24_K28",
        "paradigm": "per_fold_inv_rae_weight_train_only_df_zero",
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
        "oof_corr_labels": K_LABELS,
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "per_seed_pooled_rae": per_seed_pooled,
        "per_seed_fold_rae": per_seed_fold_rae,
        "per_seed_fold_weights": per_seed_fold_weights,
        "deploy_weights": {
            k_lab: float(w) for k_lab, w in zip(K_LABELS, w_deploy)
        },
        "weight_std_across_folds": {
            k_lab: float(all_weights[:, i].std())
            for i, k_lab in enumerate(K_LABELS)
        },
        "mean_rae": mean_rae,
        "std_rae": std_rae,
        "gate_promote": GATE_PROMOTE,
        "gate_better": GATE_BETTER,
        "verdict": verdict,
        "nb2604_ref": NB2604_REF,
        "delta_vs_nb2604": delta_vs_nb2604,
        "nb2171_ref": NB2171_REF,
        "delta_vs_nb2171": delta_vs_nb2171,
        "te_mean": float(pred_te_513.mean()),
        "te_std": float(pred_te_513.std()),
        "te_unb_in_sample_rae": te_unb_in,
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv),
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
    print(f"   deploy weights        = " + ", ".join(
        f"{k_lab}={w:.4f}" for k_lab, w in zip(K_LABELS, w_deploy)
    ))
    print(f"   per-seed pooled RAE   = "
          f"{[round(r, 4) for r in per_seed_pooled]}")
    print(f"   MEAN  pooled RAE      = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   gate                  = <{GATE_PROMOTE} PROMOTE / "
          f"<{GATE_BETTER} BETTER  -> {verdict}")
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
        "mean_rae",
        "std_rae",
        "verdict",
        "delta_vs_nb2604",
        "delta_vs_nb2171",
        "te_unb_in_sample_rae",
        "deploy_weights",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
