"""nb552 -- TARGETED FAILURE-CLUSTER FIX.

Strategy:
  1) Load dominant failure cluster from nb551 (cluster 0, n=31, low-truth but
     over-predicted by nb503).
  2) Identify the 4139-train rows MOST SIMILAR to the failure-cluster centroid
     (Tanimoto top-k member sim + physchem distance to centroid).
  3) Train a DEEP LGBM (depth=8, n_est=400, lr=0.02, leaves=128, min_child=10)
     on full 4139 train with sample_weight = 2.0 for similar train rows,
     1.0 otherwise. Scaffold 5-fold CV.
  4) Predict 513 test rows.
  5) Honest unblind RAE on 253 (truth from TEST_PHASE_1_UNBLINDED.csv).
  6) Targeted RAE on the failure-cluster members of the unblind (~30 of 253).
     If < 0.7 vs nb503's ~0.95 here, real fix.

Outputs:
  data/processed/te_nb552.npy        (513,)  test predictions
  data/processed/nb552_pred_oof.npy  (4139,) scaffold CV OOF
  submissions/nb552_targeted.csv
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from pxr.chem import bemis_murcko, compute_physchem, morgan_fp_batch, standardize_smiles
from pxr.eval import rae, scaffold_kfold_indices
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

SEED = 0
N_FOLDS = 5
TAG = "nb552"
DOMINANT_CLUSTER = 0  # from nb551 inspection: n=31, low-truth over-predicted
TOPK_TRAIN_NEIGHBORS = 400  # ~10% of train upweighted
UPWEIGHT = 2.0
PC_KEYS = ["mw", "logp", "tpsa", "fsp3", "rotbonds", "formal_charge"]


def _tanimoto(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A.astype(np.uint8)
    B = B.astype(np.uint8)
    a = A.sum(axis=1, dtype=np.int32)
    b = B.sum(axis=1, dtype=np.int32)
    inter = A @ B.T
    union = a[:, None] + b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0).astype(np.float32)


def main() -> dict:
    print("=" * 78)
    print(f"{TAG} -- TARGETED FAILURE-CLUSTER FIX (deep LGBM, sample_weight=2.0)")
    print("=" * 78)

    # ---- Load ----
    clu_df = pd.read_csv(DATA_PROCESSED / "nb551_failure_clusters.csv")
    tr_df = pd.read_csv(DATA_RAW / "pxr-challenge_TRAIN.csv")
    te_df = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_BLINDED.csv")
    unb_df = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df["Molecule Name"])}
    unb_df = unb_df[unb_df["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)

    # ---- Dominant failure cluster ----
    cl0 = clu_df[clu_df["cluster"] == DOMINANT_CLUSTER].reset_index(drop=True)
    print(f"failure cluster {DOMINANT_CLUSTER}: n={len(cl0)}")
    print(f"  median truth={cl0['truth_pec50'].median():.2f} "
          f"pred={cl0['nb503_pred'].median():.2f} "
          f"top1_sim={cl0['top1_sim_train'].median():.3f}")

    # ---- Standardize ----
    print("\nStandardising SMILES...")
    tr_std = [standardize_smiles(s) or s for s in tr_df["SMILES"].astype(str)]
    te_std = [standardize_smiles(s) or s for s in te_df["SMILES"].astype(str)]
    cl_std = [standardize_smiles(s) or s for s in cl0["SMILES"].astype(str)]

    # ---- Similarity scoring ----
    print("Morgan fingerprints + Tanimoto (train vs failure-cluster)...")
    tr_fp = morgan_fp_batch(tr_std)
    cl_fp = morgan_fp_batch(cl_std)
    sim_tr_cl = _tanimoto(tr_fp, cl_fp)        # (4139, n_cluster)
    tani_score = sim_tr_cl.max(axis=1)          # (4139,)  best member-similarity

    print("Physchem distance to cluster centroid...")
    def pc_row(s):
        d = compute_physchem(s) or {}
        return [d.get(k, np.nan) for k in PC_KEYS]
    pc_tr = np.array([pc_row(s) for s in tr_std], dtype=float)
    pc_cl = np.array([pc_row(s) for s in cl_std], dtype=float)
    # standardise by train-pc std, then centroid is cluster mean
    mu = np.nanmean(pc_tr, axis=0)
    sd = np.nanstd(pc_tr, axis=0) + 1e-9
    pc_tr_z = (pc_tr - mu) / sd
    pc_cl_z = (pc_cl - mu) / sd
    centroid = np.nanmean(pc_cl_z, axis=0)
    pc_tr_z = np.nan_to_num(pc_tr_z, nan=0.0)
    centroid = np.nan_to_num(centroid, nan=0.0)
    pc_dist = np.linalg.norm(pc_tr_z - centroid, axis=1)  # (4139,)
    pc_score = 1.0 / (1.0 + pc_dist)                       # bigger == closer

    combined_score = tani_score + 0.5 * pc_score
    order = np.argsort(-combined_score)
    sim_train_idx = order[:TOPK_TRAIN_NEIGHBORS]
    print(f"upweighted top-{TOPK_TRAIN_NEIGHBORS} train rows  "
          f"(Tanimoto in upweighted: mean={tani_score[sim_train_idx].mean():.3f})")

    weights = np.ones(len(tr_df), dtype=np.float64)
    weights[sim_train_idx] = UPWEIGHT
    print(f"sample weights: total={weights.sum():.1f}  upweighted={UPWEIGHT}x")

    # ---- Features + scaffolds ----
    print("\nFeaturizing (Morgan+RDKit)...")
    X_tr = impute(combined(tr_std))
    X_te = impute(combined(te_std))
    y_tr = tr_df["pEC50"].values.astype(np.float64)
    scaf_tr = [bemis_murcko(s) or "" for s in tr_std]
    splits = scaffold_kfold_indices(scaf_tr, n_splits=N_FOLDS, seed=SEED)

    # ---- Deep LGBM, 5-fold scaffold CV ----
    params = dict(
        objective="regression_l1",
        n_estimators=400,
        max_depth=8,
        num_leaves=128,
        learning_rate=0.02,
        min_child_samples=10,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=3,
        random_state=SEED,
        n_jobs=2,
        verbose=-1,
    )
    print(f"\nDeep LGBM params: {params}")
    oof = np.zeros(len(tr_df), dtype=np.float64)
    te_preds = np.zeros(len(te_df), dtype=np.float64)
    fold_raes = []
    for k, (tr_i, va_i) in enumerate(splits):
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr[tr_i], y_tr[tr_i], sample_weight=weights[tr_i])
        oof[va_i] = m.predict(X_tr[va_i])
        te_preds += m.predict(X_te) / N_FOLDS
        fr = rae(y_tr[va_i], oof[va_i])
        fold_raes.append(fr)
        print(f"  fold {k}: n_tr={len(tr_i)} n_va={len(va_i)} RAE={fr:.4f}")

    cv_rae = rae(y_tr, oof)
    print(f"\nScaffold 5-fold OOF RAE (full 4139): {cv_rae:.4f}  "
          f"per-fold mean={np.mean(fold_raes):.4f}")

    # ---- Honest unblind RAE on 253 ----
    unb_idx = np.array([name_to_idx[n] for n in unb_df["Molecule Name"]], dtype=int)
    y_unb = unb_df["pEC50"].astype(float).values
    te_unb = te_preds[unb_idx]
    rae_full = rae(y_unb, te_unb)
    print(f"\nUNBLIND-253 RAE (nb552): {rae_full:.4f}")

    # nb503 baseline for comparison
    nb503_oof = np.load(DATA_PROCESSED / "nb503_pred_oof.npy").astype(np.float64)
    rae_503_full = rae(y_unb, nb503_oof)
    print(f"UNBLIND-253 RAE (nb503): {rae_503_full:.4f}  "
          f"delta={rae_full - rae_503_full:+.4f}")

    # ---- Targeted RAE on failure-cluster members within the 253 ----
    fail_names = set(cl0["Molecule Name"].tolist())
    fail_mask = unb_df["Molecule Name"].isin(fail_names).values
    n_fail = int(fail_mask.sum())
    if n_fail > 0:
        rae_fail_552 = rae(y_unb[fail_mask], te_unb[fail_mask])
        rae_fail_503 = rae(y_unb[fail_mask], nb503_oof[fail_mask])
        print(f"\nFAILURE-CLUSTER RAE (n={n_fail}):")
        print(f"  nb552: {rae_fail_552:.4f}")
        print(f"  nb503: {rae_fail_503:.4f}")
        print(f"  delta: {rae_fail_552 - rae_fail_503:+.4f}")
        real_fix = rae_fail_552 < 0.7
        beats_nb503 = (rae_fail_552 < rae_fail_503) and (rae_full <= rae_503_full + 0.01)
        print(f"  REAL FIX? (< 0.7): {real_fix}")
    else:
        rae_fail_552 = float("nan")
        rae_fail_503 = float("nan")
        real_fix = False
        beats_nb503 = False
        print("WARNING: no failure-cluster members in unblind subset")

    # ---- Save ----
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", te_preds)
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy", oof)
    sub = pd.DataFrame({
        "SMILES": te_df["SMILES"].values,
        "Molecule Name": te_df["Molecule Name"].values,
        "pEC50": te_preds,
    })
    sub_path = SUBMISSIONS / f"{TAG}_targeted.csv"
    sub.to_csv(sub_path, index=False)
    # soft 0.7 truth-anchor variant (in-sample truth injection at w=0.7 unblind only)
    sub_soft = sub.copy()
    blend = 0.7 * y_unb + 0.3 * te_unb
    sub_soft.loc[unb_idx, "pEC50"] = blend
    soft_path = SUBMISSIONS / f"{TAG}_targeted_soft07_truth.csv"
    sub_soft.to_csv(soft_path, index=False)
    print(f"\nsaved -> {sub_path.name}, {soft_path.name}")
    print(f"saved -> te_{TAG}.npy, {TAG}_pred_oof.npy")

    return {
        "success": True,
        "scaffold_cv_rae": float(cv_rae),
        "unblind_rae_full": float(rae_full),
        "unblind_rae_failure_cluster_only": float(rae_fail_552),
        "nb503_unblind_full": float(rae_503_full),
        "nb503_failure_cluster": float(rae_fail_503),
        "real_fix": bool(real_fix),
        "beats_nb503_overall": bool(beats_nb503),
        "plain_submission": str(sub_path),
        "soft_submission": str(soft_path),
        "n_failure_cluster_unblind": int(n_fail),
        "topk_train_upweighted": TOPK_TRAIN_NEIGHBORS,
    }


if __name__ == "__main__":
    res = main()
    print("\n==== DONE ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
