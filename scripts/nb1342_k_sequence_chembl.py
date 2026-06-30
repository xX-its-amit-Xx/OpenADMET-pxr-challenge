"""nb1342 -- k-sequence ChEMBL kNN features (k=1,3,5,10,20 as parallel cols).

Hypothesis:
    nb1242 used only k=5 mean (sim-weighted pEC50, mean similarity) -> RAE
    0.5431 (beats nb1183 0.5513, narrowly misses nb1211 0.5451 by +0.002).
    Exposing multiple k values as parallel features gives the LGBM residual
    learner more granular shape information about the neighbor distance
    distribution -- small k captures the closest match, large k captures
    the regional density / regression-to-mean tendency.  The tree can pick
    the best k bin per row.

Protocol:
    1. Reuse 945-cpd ChEMBL PXR pool (same union as nb1242).
    2. Morgan-2048 Tanimoto kNN to 513 test, top-K_MAX (=20).
    3. For each k in {1, 3, 5, 10, 20}: derive
         mean_k_pec50  -- similarity-weighted mean of pool pEC50 over top-k
         mean_k_sim    -- arithmetic mean of top-k similarities
       => 10 ChEMBL features.
    4. Features for residual learner: MACCS-167 (unblind slice) + 10 ChEMBL
       features (unblind slice) = 177 cols.
    5. Identical 5-seed shallow LGBM Huber bag, 5-fold cross-fit (matches
       nb1242 hyperparams exactly so the only change is feature set).
    6. Verdict at 0.003 margin vs nb1242 (0.5431).

Outputs:
    scripts/nb1342_k_sequence_chembl.py       (this file)
    data/processed/nb1342_summary.json
    data/processed/nb1342_mean_bag_oof.npy    (253,) float32
    data/processed/nb1342_per_seed_corrected_oof.npy (5, 253) float32
    data/processed/nb1342_median_bag_oof.npy  (253,) float32
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
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1342"
ANCHOR = "nb1070"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

K_LIST = [1, 3, 5, 10, 20]
K_MAX = max(K_LIST)
SIM_FLOOR = 1e-6

NB1070_REF = 0.5771
NB1183_REF = 0.5513
NB1211_REF = 0.5451
NB1242_REF = 0.5431      # nb1242 mean_bag pooled RAE -- key reference here
DECISION_MARGIN = 0.003


def _safe_inchikey(mol):
    try:
        return Chem.MolToInchiKey(mol) if mol is not None else None
    except Exception:
        return None


def _safe_can_smiles(mol):
    try:
        return Chem.MolToSmiles(mol) if mol is not None else None
    except Exception:
        return None


def _load_chembl_pool() -> pd.DataFrame:
    """Identical union to nb1242 -- 945 ChEMBL PXR cpds (canonical)."""
    frames = []

    p1 = EXT_DIR / "chembl_pxr_CHEMBL3401.parquet"
    if p1.exists():
        d = pd.read_parquet(p1)
        mask = (
            d["standard_type"].isin(KEEP_TYPES)
            & d["canonical_smiles"].notna()
            & (d["standard_units"] == "nM")
            & d["standard_value"].notna()
            & d["standard_relation"].isin(KEEP_RELATIONS)
        )
        d = d[mask].copy()
        v = d["standard_value"].astype(float)
        d = d[(v > MIN_NM) & (v < MAX_NM)].copy()
        d["pec50_raw"] = 9.0 - np.log10(d["standard_value"].astype(float))
        d = d[["canonical_smiles", "pec50_raw"]].rename(
            columns={"canonical_smiles": "smiles", "pec50_raw": "pec50"}
        )
        d["src"] = "CHEMBL3401_raw"
        frames.append(d)
        print(f"   [src] CHEMBL3401_raw kept: {len(d)} rows")

    p2 = EXT_DIR / "chembl_nr_extended.parquet"
    if p2.exists():
        d = pd.read_parquet(p2)
        d = d[d["target_name"] == "PXR"].copy()
        d = d[d["pec50"].notna() & d["std_smiles"].notna()].copy()
        d["pec50"] = d["pec50"].astype(float)
        d = d[(d["pec50"] >= 3.0) & (d["pec50"] <= 11.0)].copy()
        d = d[["std_smiles", "pec50"]].rename(columns={"std_smiles": "smiles"})
        d["src"] = "nr_extended"
        frames.append(d)
        print(f"   [src] chembl_nr_extended PXR kept: {len(d)} rows")

    p3 = EXT_DIR / "chembl_pxr_all_types.parquet"
    if p3.exists():
        d = pd.read_parquet(p3)
        d = d[d["target"] == "PXR"].copy()
        d = d[d["pec50"].notna() & d["smiles"].notna()].copy()
        d["pec50"] = d["pec50"].astype(float)
        d = d[(d["pec50"] >= 3.0) & (d["pec50"] <= 11.0)].copy()
        d = d[["smiles", "pec50"]]
        d["src"] = "pxr_all_types"
        frames.append(d)
        print(f"   [src] chembl_pxr_all_types kept: {len(d)} rows")

    if not frames:
        raise FileNotFoundError("No local ChEMBL PXR parquets in data/external/")

    pool = pd.concat(frames, ignore_index=True)
    print(f"   [pool] pre-standardize union: {len(pool)} rows")

    mols = pool["smiles"].apply(standardize)
    pool["inchikey"] = mols.apply(_safe_inchikey)
    pool["std_smiles"] = mols.apply(_safe_can_smiles)
    pool = pool[pool["inchikey"].notna() & pool["std_smiles"].notna()].copy()
    print(f"   [pool] after RDKit standardize: {len(pool)} rows")

    agg = (
        pool.groupby("inchikey", as_index=False)
        .agg(pec50=("pec50", "median"),
             std_smiles=("std_smiles", "first"),
             src_first=("src", "first"),
             n_meas=("pec50", "count"))
    )
    agg = agg.rename(columns={"src_first": "src"})
    print(f"   [pool] after InChIKey dedup (median agg): {len(agg)} unique cpds")
    return agg


def _tanimoto_topk(fp_q: np.ndarray, fp_pool: np.ndarray, k: int):
    a = fp_q.astype(np.float32)
    b = fp_pool.astype(np.float32)
    a_sum = a.sum(axis=1)
    b_sum = b.sum(axis=1)
    n_q = a.shape[0]
    n_pool = b.shape[0]
    top_idx = np.zeros((n_q, k), dtype=np.int32)
    top_sim = np.zeros((n_q, k), dtype=np.float32)

    BLOCK = 64
    for s in range(0, n_q, BLOCK):
        e = min(n_q, s + BLOCK)
        inter = a[s:e] @ b.T
        denom = a_sum[s:e, None] + b_sum[None, :] - inter
        denom = np.maximum(denom, 1.0)
        sim = inter / denom
        if k >= n_pool:
            idx_part = np.argsort(-sim, axis=1)[:, :k]
        else:
            part = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
            row_idx = np.arange(e - s)[:, None]
            sim_part = sim[row_idx, part]
            order = np.argsort(-sim_part, axis=1)
            idx_part = part[row_idx, order]
        row_idx = np.arange(e - s)[:, None]
        top_idx[s:e] = idx_part
        top_sim[s:e] = sim[row_idx, idx_part]
    return top_idx, top_sim


def _knn_predict_at_k(top_idx_full: np.ndarray, top_sim_full: np.ndarray,
                       pool_labels: np.ndarray, k: int, fallback: float):
    """Re-use the K_MAX-sorted neighbor list; slice to top-k for any k <= K_MAX.

    Returns: (mean_k_pec50 (n_q,), mean_k_sim (n_q,)) both float32.
    """
    idx_k = top_idx_full[:, :k]
    sim_k = top_sim_full[:, :k]
    w = np.clip(sim_k, 0.0, 1.0)
    w_sum = w.sum(axis=1)
    n_q = idx_k.shape[0]
    pred = np.empty(n_q, dtype=np.float32)
    for i in range(n_q):
        if w_sum[i] < SIM_FLOOR:
            pred[i] = fallback
        else:
            pred[i] = np.sum(w[i] * pool_labels[idx_k[i]]) / w_sum[i]
    mean_sim = sim_k.mean(axis=1).astype(np.float32)
    return pred, mean_sim


def _lgbm_params(seed: int) -> dict:
    return dict(
        objective="huber",
        alpha=1.0,
        learning_rate=0.05,
        n_estimators=80,
        max_depth=3,
        num_leaves=7,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        verbosity=-1,
        random_state=seed,
        n_jobs=2,
    )


def _residual_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray,
                                  seed: int):
    """Returns (oof predictions, list of fold-trained models) for later
    feature-importance pooling."""
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_models = []
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
        fold_models.append(mdl)
    return oof, fold_models


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- k-sequence ChEMBL kNN features (k={K_LIST})")
    print(f"          seeds = {RESID_SEEDS}  folds = {RESID_FOLDS}")
    print(f"          features = MACCS-167 + 10 ChEMBL (mean_k_pec50, mean_k_sim x 5)")
    print("=" * 78)

    # ---- Load anchor + truth ----
    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    anchor = np.load(DATA_PROCESSED / f"{ANCHOR}_pred_oof.npy").astype(np.float64)
    if anchor.shape[0] != n_unb:
        raise ValueError(f"{ANCHOR} shape mismatch: {anchor.shape} vs {n_unb}")
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} pooled RAE = {rae_anchor:.4f}  (ref ~{NB1070_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- ChEMBL pool ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL (945 cpds, identical union to nb1242)")
    print("-" * 78)
    pool = _load_chembl_pool()

    # ---- Test InChIKey leak guard ----
    print("\n" + "-" * 78)
    print("TEST-SET LEAK GUARD")
    print("-" * 78)
    test_mols = [standardize(s) for s in test_smiles]
    test_inchikeys = set()
    for m in test_mols:
        ik = _safe_inchikey(m)
        if ik is not None:
            test_inchikeys.add(ik)
    n_before = len(pool)
    pool = pool[~pool["inchikey"].isin(test_inchikeys)].reset_index(drop=True)
    n_after = len(pool)
    print(f"   pool: {n_before} -> {n_after}  (dropped {n_before - n_after})")

    # ---- Morgan fingerprints ----
    print("\n" + "-" * 78)
    print("MORGAN-2048 FINGERPRINTS")
    print("-" * 78)
    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    print(f"   pool FP: {fp_pool.shape}  density={fp_pool.mean():.4f}")
    keep_pool = fp_pool.sum(axis=1) > 0
    if not keep_pool.all():
        n_drop = int((~keep_pool).sum())
        print(f"   dropped {n_drop} pool rows with zero FP")
        pool = pool[keep_pool].reset_index(drop=True)
        fp_pool = fp_pool[keep_pool]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))
    print(f"   final pool size: {len(pool)}  median pEC50 = {pool_median:.3f}")

    std_test_smiles = []
    for m in test_mols:
        if m is None:
            std_test_smiles.append("")
        else:
            std_test_smiles.append(Chem.MolToSmiles(m))
    fp_test = morgan_fp_batch(std_test_smiles)
    print(f"   test FP: {fp_test.shape}  density={fp_test.mean():.4f}")

    # ---- kNN at K_MAX, then slice per k ----
    print("\n" + "-" * 78)
    print(f"TANIMOTO kNN (K_MAX={K_MAX}) -- test (513) vs ChEMBL pool ({len(pool)})")
    print("-" * 78)
    top_idx_full, top_sim_full = _tanimoto_topk(fp_test, fp_pool, k=K_MAX)
    top1_sim = top_sim_full[:, 0]
    print(f"   top1 sim   p10={np.percentile(top1_sim, 10):.3f}  "
          f"p50={np.percentile(top1_sim, 50):.3f}  "
          f"p90={np.percentile(top1_sim, 90):.3f}  "
          f"max={top1_sim.max():.3f}")

    # ---- Build 10 ChEMBL features for all 513 test rows ----
    print("\n" + "-" * 78)
    print(f"BUILD k-SEQUENCE FEATURES  k in {K_LIST}")
    print("-" * 78)
    chembl_cols = []
    chembl_col_names = []
    per_k_stats = {}
    for k in K_LIST:
        pred_k, sim_k = _knn_predict_at_k(
            top_idx_full, top_sim_full, pool_labels, k=k, fallback=pool_median
        )
        chembl_cols.append(pred_k.reshape(-1, 1))
        chembl_cols.append(sim_k.reshape(-1, 1))
        chembl_col_names.append(f"mean_{k}_pec50")
        chembl_col_names.append(f"mean_{k}_sim")
        per_k_stats[f"k={k}"] = {
            "mean_pec50_mean": float(pred_k.mean()),
            "mean_pec50_std": float(pred_k.std()),
            "mean_sim_p10": float(np.percentile(sim_k, 10)),
            "mean_sim_p50": float(np.percentile(sim_k, 50)),
            "mean_sim_p90": float(np.percentile(sim_k, 90)),
        }
        print(f"   k={k:2d}: pec50 mean={pred_k.mean():.3f} std={pred_k.std():.3f} | "
              f"sim p50={np.percentile(sim_k, 50):.3f}")

    X_chembl_te = np.concatenate(chembl_cols, axis=1).astype(np.float32)
    print(f"   ChEMBL feature block: {X_chembl_te.shape}  cols={chembl_col_names}")

    # ---- MACCS-167 (unblind slice) ----
    X_maccs_te = np.load(MACCS_TE_PATH)
    if X_maccs_te.shape[0] != n_test:
        raise ValueError(f"MACCS cache shape mismatch: {X_maccs_te.shape}")
    X_maccs_unb = X_maccs_te[unb_idx].astype(np.float32)
    X_chembl_unb = X_chembl_te[unb_idx].astype(np.float32)
    X_unb = np.concatenate([X_maccs_unb, X_chembl_unb], axis=1).astype(np.float32)
    feat_dim = X_unb.shape[1]
    feat_names = [f"maccs_{i}" for i in range(X_maccs_unb.shape[1])] + chembl_col_names
    assert len(feat_names) == feat_dim, (len(feat_names), feat_dim)
    print(f"   residual feature matrix: {X_unb.shape}  (MACCS-167 + 10 ChEMBL)")

    # ---- Per-seed residual cross-fit ----
    print("\n" + "-" * 78)
    print(f"PER-SEED RESIDUAL CROSS-FIT  shallow LGBM Huber  dim={feat_dim}")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []
    all_importance = np.zeros(feat_dim, dtype=np.float64)
    n_models_total = 0
    for i, s in enumerate(RESID_SEEDS):
        resid_oof_s, fold_models = _residual_cross_fit_one_seed(X_unb, residual, s)
        pred_corr_s = anchor + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        delta_s = rae_s - rae_anchor
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_nb1070": delta_s,
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
        })
        for mdl in fold_models:
            all_importance += mdl.feature_importances_.astype(np.float64)
            n_models_total += 1
        print(f"   seed {s:3d}:  rae_corr = {rae_s:.4f}  "
              f"(d_vs_nb1070 = {delta_s:+.4f})  "
              f"|resid_oof|.std = {resid_oof_s.std():.3f}")

    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))

    per_seed_rae_arr = np.array(per_seed_rae)
    rae_per_seed_mean = float(per_seed_rae_arr.mean())
    rae_per_seed_median = float(np.median(per_seed_rae_arr))
    rae_per_seed_std = float(per_seed_rae_arr.std())

    # ---- Pooled feature importance (sum across 25 fold-models) ----
    avg_importance = all_importance / max(1, n_models_total)
    order = np.argsort(-avg_importance)
    top5 = [(feat_names[j], float(avg_importance[j])) for j in order[:5]]
    chembl_idx_in_full = [feat_names.index(c) for c in chembl_col_names]
    chembl_importance = [(c, float(avg_importance[feat_names.index(c)]))
                          for c in chembl_col_names]

    print("\n" + "-" * 78)
    print("BAG AGGREGATIONS")
    print("-" * 78)
    print(f"   per-seed RAE          = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   per-seed mean         = {rae_per_seed_mean:.4f}")
    print(f"   per-seed median       = {rae_per_seed_median:.4f}")
    print(f"   per-seed std          = {rae_per_seed_std:.4f}")
    print(f"   pooled RAE(mean_bag)  = {rae_mean_bag:.4f}  "
          f"(d_vs_nb1070 = {rae_mean_bag - rae_anchor:+.4f})  "
          f"(d_vs_nb1242 = {rae_mean_bag - NB1242_REF:+.4f})")
    print(f"   pooled RAE(median_bag)= {rae_median_bag:.4f}")
    print(f"   nb1242 ref            = {NB1242_REF:.4f}  (k=5 only)")
    print(f"   nb1211 ref            = {NB1211_REF:.4f}  (SLSQP variants)")
    print(f"   nb1183 ref            = {NB1183_REF:.4f}  (MACCS-only)")

    print("\n   top-5 feature importance (avg over 25 fold-models):")
    for nm, imp in top5:
        print(f"     {nm:24s}  imp = {imp:8.3f}")
    print("\n   ChEMBL feature importance:")
    for nm, imp in chembl_importance:
        print(f"     {nm:24s}  imp = {imp:8.3f}")
    chembl_total = float(sum(imp for _, imp in chembl_importance))
    total_imp = float(avg_importance.sum())
    chembl_share = chembl_total / max(1e-9, total_imp)
    print(f"   ChEMBL share of total importance = {chembl_share:.3f}  "
          f"({chembl_total:.2f} / {total_imp:.2f})")

    # ---- Verdict ----
    beats_nb1242 = rae_mean_bag < NB1242_REF - DECISION_MARGIN
    beats_nb1211 = rae_mean_bag < NB1211_REF - DECISION_MARGIN
    beats_nb1070 = rae_mean_bag < rae_anchor - DECISION_MARGIN
    if beats_nb1242:
        verdict = "K_SEQUENCE_BEATS_NB1242_NEW_PRIMARY_CANDIDATE"
    elif abs(rae_mean_bag - NB1242_REF) < DECISION_MARGIN:
        verdict = "K_SEQUENCE_TIES_NB1242_NO_NEW_SIGNAL"
    elif beats_nb1211:
        verdict = "K_SEQUENCE_BEATS_NB1211_BUT_NOT_NB1242"
    elif beats_nb1070:
        verdict = "K_SEQUENCE_HELPS_NB1070_BUT_WORSE_THAN_NB1242"
    else:
        verdict = "K_SEQUENCE_HURTS_NB1070"
    print(f"   verdict               = {verdict}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_per_seed_corrected_oof.npy",
            per_seed_corrected.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy",
            mean_bag_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_median_bag_oof.npy",
            median_bag_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_per_seed_corrected_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_mean_bag_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_median_bag_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "data_source": "local_chembl_caches_union",
        "chembl_caches": [
            "data/external/chembl_pxr_CHEMBL3401.parquet",
            "data/external/chembl_nr_extended.parquet (PXR rows)",
            "data/external/chembl_pxr_all_types.parquet",
        ],
        "n_chembl_pool": int(len(pool)),
        "test_inchikeys_in_pool_dropped": int(n_before - n_after),
        "k_list": K_LIST,
        "k_max": K_MAX,
        "n_chembl_features": len(chembl_col_names),
        "chembl_col_names": chembl_col_names,
        "per_k_stats": per_k_stats,
        "top1_sim_p10": float(np.percentile(top1_sim, 10)),
        "top1_sim_p50": float(np.percentile(top1_sim, 50)),
        "top1_sim_p90": float(np.percentile(top1_sim, 90)),
        "n_unb": n_unb,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "feature_dim": feat_dim,
        "lgbm_max_depth": 3,
        "lgbm_num_leaves": 7,
        "lgbm_n_estimators": 80,
        "lgbm_learning_rate": 0.05,
        "lgbm_min_child_samples": 20,
        "lgbm_huber_alpha": 1.0,
        "rae_anchor_nb1070": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_seed_rae": per_seed_rae,
        "per_seed_records": per_seed_records,
        "rae_per_seed_mean": rae_per_seed_mean,
        "rae_per_seed_median": rae_per_seed_median,
        "rae_per_seed_std": rae_per_seed_std,
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_nb1070": rae_mean_bag - rae_anchor,
        "delta_mean_bag_vs_nb1183": rae_mean_bag - NB1183_REF,
        "delta_mean_bag_vs_nb1211": rae_mean_bag - NB1211_REF,
        "delta_mean_bag_vs_nb1242": rae_mean_bag - NB1242_REF,
        "beats_nb1070": bool(beats_nb1070),
        "beats_nb1211": bool(beats_nb1211),
        "beats_nb1242": bool(beats_nb1242),
        "top5_feature_importance": [
            {"feature": nm, "importance": imp} for nm, imp in top5
        ],
        "chembl_feature_importance": [
            {"feature": nm, "importance": imp} for nm, imp in chembl_importance
        ],
        "chembl_importance_share": chembl_share,
        "verdict": verdict,
        "nb1070_ref_pooled": NB1070_REF,
        "nb1183_ref": NB1183_REF,
        "nb1211_ref": NB1211_REF,
        "nb1242_ref": NB1242_REF,
        "decision_margin": DECISION_MARGIN,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "n_chembl_pool", "k_list", "top1_sim_p50",
        "rae_anchor_nb1070", "per_seed_rae",
        "rae_per_seed_mean", "rae_per_seed_std",
        "rae_mean_bag", "rae_median_bag",
        "delta_mean_bag_vs_nb1242",
        "delta_mean_bag_vs_nb1211",
        "beats_nb1242", "beats_nb1211",
        "top5_feature_importance",
        "chembl_importance_share",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
