"""nb1253 -- Extended ChEMBL kNN features: k=1, k=3, k=5, k=10, plus
std/min/max neighbor pEC50 statistics.

Hypothesis:
    nb1242 used (k=5 sim-weighted mean of ChEMBL pEC50 + mean similarity) as
    two scalar features bolted onto the MACCS-167 residual learner.  The
    residual learner had to infer "uncertainty about this ChEMBL prediction"
    from a single sim scalar.  By exposing richer kNN distance-shape
    statistics -- k=1 nearest, k=3 spread, k=5 min/max, k=10 broader pool --
    the LGBM gets a more granular view of neighbor-cluster structure that
    may help on rare-scaffold test rows where the top-1 sim is meaningful
    but the k=5 mean is washed out by distant noise.

Protocol:
    1. Reuse 945 standardized ChEMBL PXR compounds (same _load_chembl_pool
       union + InChIKey dedup + 513-test leak guard as nb1242).
    2. Morgan-2048 fingerprints for all 945 and 513 test SMILES.
    3. For each test row: top-10 Tanimoto neighbors over pool.  Build 13
       features:
         k=1:  top1_sim, top1_pec50
         k=3:  mean_3_pec50 (sim-weighted), mean_3_sim, std_3_pec50
         k=5:  mean_5_pec50, mean_5_sim, std_5_pec50, max_5_pec50,
               min_5_pec50
         k=10: mean_10_pec50, mean_10_sim, std_10_pec50
       Fallbacks: any row with all-zero sim at the relevant k -> mean_*_pec50
       and min/max -> pool median; std_* -> 0.0; sim stats -> 0.0; top1_pec50
       -> pool median.
    4. Residual learner: anchor = nb1070_pred_oof on 253 unblind rows;
       features = MACCS-167(unb) + 13 ChEMBL = (253, 180).
       5-seed shallow LGBM Huber (depth=3, leaves=7, n_est=80, lr=0.05,
       min_child=20, alpha=1.0) -- identical capacity to nb1242.
       5-fold cross-fit per seed (same KFold seeds = [0, 1, 7, 42, 137]).
       mean_bag = average of 5 seed-OOFs.
    5. Verdict at 0.003 margin vs nb1242 (0.5431).
    6. Feature importance: aggregate LGBM gain across all (5 seeds x 5 folds)
       = 25 fitted models, normalize, report top-5 features by mean gain.

Outputs:
    scripts/nb1253_extended_chembl_features.py     (this file)
    data/processed/nb1253_summary.json
    data/processed/nb1253_mean_bag_oof.npy           (253,) float32
    data/processed/nb1253_per_seed_corrected_oof.npy (5, 253) float32
    data/processed/nb1253_median_bag_oof.npy         (253,) float32
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

TAG = "nb1253"
ANCHOR = "nb1070"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K_MAX = 10
SIM_FLOOR = 1e-6

NB1070_REF = 0.5771
NB1242_REF = 0.5431      # extended baseline: k=5 sim-weighted mean + mean sim
DECISION_MARGIN = 0.003

# Feature names (in column order)
FEATURE_NAMES_CHEMBL = [
    "top1_sim",
    "top1_pec50",
    "mean_3_pec50",
    "mean_3_sim",
    "std_3_pec50",
    "mean_5_pec50",
    "mean_5_sim",
    "std_5_pec50",
    "max_5_pec50",
    "min_5_pec50",
    "mean_10_pec50",
    "mean_10_sim",
    "std_10_pec50",
]


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
    """Union of three local ChEMBL PXR caches -> (smi, pec50) frame.

    Returns columns: ['inchikey', 'std_smiles', 'pec50', 'src', 'n_meas'].
    Standardization + InChIKey dedup (median pEC50 when duplicated).
    Mirrors nb1242._load_chembl_pool exactly to guarantee a 945-row pool.
    """
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


def _weighted_mean(sim: np.ndarray, vals: np.ndarray, fallback: float) -> float:
    """Similarity-weighted mean of vals; fallback if sum(sim) ~ 0."""
    w = np.clip(sim, 0.0, 1.0)
    s = w.sum()
    if s < SIM_FLOOR:
        return float(fallback)
    return float(np.sum(w * vals) / s)


def _build_chembl_features(top_idx_10: np.ndarray, top_sim_10: np.ndarray,
                           pool_labels: np.ndarray,
                           pool_median: float) -> np.ndarray:
    """Build (n_test, 13) feature matrix from top-10 neighbor results.

    Column order matches FEATURE_NAMES_CHEMBL.
    """
    n_q = top_idx_10.shape[0]
    F = np.zeros((n_q, 13), dtype=np.float32)

    for i in range(n_q):
        sims_10 = top_sim_10[i]                    # (10,)
        idx_10 = top_idx_10[i]                     # (10,)
        labs_10 = pool_labels[idx_10]              # (10,)

        sims_1 = sims_10[:1]
        labs_1 = labs_10[:1]
        sims_3 = sims_10[:3]
        labs_3 = labs_10[:3]
        sims_5 = sims_10[:5]
        labs_5 = labs_10[:5]

        # k=1
        top1_sim = float(sims_1[0])
        top1_pec50 = float(labs_1[0]) if top1_sim >= SIM_FLOOR else pool_median

        # k=3
        mean_3_pec50 = _weighted_mean(sims_3, labs_3, pool_median)
        mean_3_sim = float(np.mean(sims_3))
        std_3_pec50 = float(np.std(labs_3)) if np.sum(sims_3) >= SIM_FLOOR else 0.0

        # k=5
        mean_5_pec50 = _weighted_mean(sims_5, labs_5, pool_median)
        mean_5_sim = float(np.mean(sims_5))
        std_5_pec50 = float(np.std(labs_5)) if np.sum(sims_5) >= SIM_FLOOR else 0.0
        max_5_pec50 = float(np.max(labs_5)) if np.sum(sims_5) >= SIM_FLOOR else pool_median
        min_5_pec50 = float(np.min(labs_5)) if np.sum(sims_5) >= SIM_FLOOR else pool_median

        # k=10
        mean_10_pec50 = _weighted_mean(sims_10, labs_10, pool_median)
        mean_10_sim = float(np.mean(sims_10))
        std_10_pec50 = float(np.std(labs_10)) if np.sum(sims_10) >= SIM_FLOOR else 0.0

        F[i] = [
            top1_sim, top1_pec50,
            mean_3_pec50, mean_3_sim, std_3_pec50,
            mean_5_pec50, mean_5_sim, std_5_pec50, max_5_pec50, min_5_pec50,
            mean_10_pec50, mean_10_sim, std_10_pec50,
        ]
    return F


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
                                 seed: int,
                                 gain_accum: np.ndarray) -> np.ndarray:
    """Cross-fit; accumulate LGBM split-gain importances into `gain_accum`.

    `gain_accum` is shape (n_feat,) -- one entry per feature; we sum gain
    importance across the 5 folds for this seed.  Caller normalises across
    all (seeds x folds) jointly.
    """
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = LGBMRegressor(**_lgbm_params(seed), importance_type="gain")
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
        gain_accum += mdl.feature_importances_.astype(np.float64)
    return oof


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Extended ChEMBL kNN features (13 stats) + MACCS-167 "
          f"residual on nb1070")
    print(f"         seeds = {RESID_SEEDS}  folds = {RESID_FOLDS}")
    print(f"         residual feature dim = 167 + 13 = 180")
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
    print("CHEMBL PXR POOL (local cache union -- same as nb1242)")
    print("-" * 78)
    pool = _load_chembl_pool()

    # Leak guard
    test_mols = [standardize(s) for s in test_smiles]
    test_inchikeys = set()
    for m in test_mols:
        ik = _safe_inchikey(m)
        if ik is not None:
            test_inchikeys.add(ik)
    n_before = len(pool)
    pool = pool[~pool["inchikey"].isin(test_inchikeys)].reset_index(drop=True)
    n_after = len(pool)
    print(f"   leak guard: {n_before} -> {n_after}")

    # ---- Morgan fingerprints ----
    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep_pool = fp_pool.sum(axis=1) > 0
    if not keep_pool.all():
        n_drop = int((~keep_pool).sum())
        print(f"   dropped {n_drop} pool rows with zero FP")
        pool = pool[keep_pool].reset_index(drop=True)
        fp_pool = fp_pool[keep_pool]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))
    print(f"   pool shape: {fp_pool.shape}  median pEC50 = {pool_median:.3f}")

    std_test_smiles = []
    for m in test_mols:
        std_test_smiles.append(Chem.MolToSmiles(m) if m is not None else "")
    fp_test = morgan_fp_batch(std_test_smiles)
    print(f"   test FP shape: {fp_test.shape}")

    # ---- Top-10 Tanimoto kNN ----
    print("\n" + "-" * 78)
    print(f"TANIMOTO kNN k=10 (will derive k=1,3,5,10 slices)")
    print("-" * 78)
    top_idx_10, top_sim_10 = _tanimoto_topk(fp_test, fp_pool, k=KNN_K_MAX)
    top1_sim = top_sim_10[:, 0]
    print(f"   top1 sim  p10={np.percentile(top1_sim, 10):.3f}  "
          f"p50={np.percentile(top1_sim, 50):.3f}  "
          f"p90={np.percentile(top1_sim, 90):.3f}  "
          f"max={top1_sim.max():.3f}")
    n_zero_neighbor = int((top1_sim < SIM_FLOOR).sum())
    print(f"   {n_zero_neighbor}/513 rows had no neighbor at any k")

    # ---- Build 13-dim ChEMBL features ----
    print("\n" + "-" * 78)
    print("BUILD 13-DIM CHEMBL FEATURE MATRIX")
    print("-" * 78)
    F_chembl = _build_chembl_features(top_idx_10, top_sim_10,
                                      pool_labels, pool_median)
    print(f"   F_chembl shape: {F_chembl.shape}")
    for j, name in enumerate(FEATURE_NAMES_CHEMBL):
        v = F_chembl[:, j]
        print(f"   {name:18s}: mean={v.mean():.3f}  std={v.std():.3f}  "
              f"min={v.min():.3f}  max={v.max():.3f}")

    # ---- MACCS unblind ----
    X_maccs_te = np.load(MACCS_TE_PATH)
    if X_maccs_te.shape[0] != n_test:
        raise ValueError(f"MACCS cache shape mismatch: {X_maccs_te.shape}")
    X_maccs_unb = X_maccs_te[unb_idx].astype(np.float32)

    # ---- Residual feature matrix on 253 ----
    F_chembl_unb = F_chembl[unb_idx].astype(np.float32)
    X_unb = np.concatenate([X_maccs_unb, F_chembl_unb], axis=1).astype(np.float32)
    feat_dim = X_unb.shape[1]
    print(f"\n   final residual feature matrix: {X_unb.shape}  "
          f"(MACCS-167 + 13 ChEMBL)")
    assert feat_dim == 180, f"expected 180-dim features, got {feat_dim}"

    feature_names = [f"maccs_{i}" for i in range(167)] + list(FEATURE_NAMES_CHEMBL)

    # ---- Per-seed cross-fit + feature importance accumulation ----
    print("\n" + "-" * 78)
    print(f"PER-SEED RESIDUAL CROSS-FIT (depth=3 shallow LGBM Huber, dim={feat_dim})")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []
    gain_accum = np.zeros(feat_dim, dtype=np.float64)

    for i, s in enumerate(RESID_SEEDS):
        resid_oof_s = _residual_cross_fit_one_seed(X_unb, residual, s, gain_accum)
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
        print(f"   seed {s:3d}:  rae_corr = {rae_s:.4f}  "
              f"(d_vs_nb1070 = {delta_s:+.4f})")

    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))

    per_seed_rae_arr = np.array(per_seed_rae)
    rae_per_seed_mean = float(per_seed_rae_arr.mean())
    rae_per_seed_median = float(np.median(per_seed_rae_arr))
    rae_per_seed_std = float(per_seed_rae_arr.std())

    print("\n" + "-" * 78)
    print("BAG AGGREGATIONS")
    print("-" * 78)
    print(f"   per-seed RAE list      = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   per-seed mean          = {rae_per_seed_mean:.4f}")
    print(f"   per-seed std           = {rae_per_seed_std:.4f}")
    print(f"   pooled RAE(mean_bag)   = {rae_mean_bag:.4f}  "
          f"(d_vs_nb1070 = {rae_mean_bag - rae_anchor:+.4f})")
    print(f"   pooled RAE(median_bag) = {rae_median_bag:.4f}")
    print(f"   nb1242 ref             = {NB1242_REF:.4f}  (k=5 mean + sim only)")
    print(f"   d_vs_nb1242            = {rae_mean_bag - NB1242_REF:+.4f}")

    # ---- Feature importance ranking ----
    print("\n" + "-" * 78)
    print("FEATURE IMPORTANCE (LGBM gain, summed across 25 fits, normalised)")
    print("-" * 78)
    total_gain = gain_accum.sum()
    if total_gain <= 0:
        gain_norm = gain_accum.copy()
    else:
        gain_norm = gain_accum / total_gain
    fi_pairs = sorted(
        zip(feature_names, gain_norm, gain_accum),
        key=lambda x: x[1], reverse=True,
    )
    top_features = []
    for rank, (name, frac, raw) in enumerate(fi_pairs[:5], 1):
        print(f"   #{rank}  {name:18s}  gain_frac = {frac:.4f}  "
              f"raw_gain_sum = {raw:.1f}")
        top_features.append({
            "rank": rank,
            "feature": name,
            "gain_fraction": float(frac),
            "raw_gain_sum": float(raw),
        })
    # Also: subset ChEMBL-only ranking (drop maccs_*)
    chembl_pairs = [(n, f, g) for (n, f, g) in fi_pairs
                    if not n.startswith("maccs_")]
    print("   (ChEMBL-feature-only ranking)")
    for rank, (name, frac, raw) in enumerate(chembl_pairs[:5], 1):
        print(f"   *#{rank}  {name:18s}  gain_frac = {frac:.4f}")

    # ---- Verdict ----
    beats_nb1242 = rae_mean_bag < NB1242_REF - DECISION_MARGIN
    beats_nb1070 = rae_mean_bag < rae_anchor - DECISION_MARGIN
    if beats_nb1242:
        verdict = "EXT_CHEMBL_FEATS_BEAT_NB1242_NEW_PRIMARY_CANDIDATE"
    elif abs(rae_mean_bag - NB1242_REF) < DECISION_MARGIN:
        verdict = "EXT_CHEMBL_FEATS_FLAT_VS_NB1242_NO_NEW_SIGNAL"
    elif beats_nb1070:
        verdict = "EXT_CHEMBL_FEATS_HELP_NB1070_BUT_HURT_VS_NB1242"
    else:
        verdict = "EXT_CHEMBL_FEATS_HURT_NB1070"
    print(f"\n   beats_nb1242 (margin {DECISION_MARGIN:.3f}): {beats_nb1242}")
    print(f"   verdict                = {verdict}")

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
        "n_chembl_pool": int(len(pool)),
        "knn_k_max": KNN_K_MAX,
        "chembl_feature_names": FEATURE_NAMES_CHEMBL,
        "n_chembl_features": len(FEATURE_NAMES_CHEMBL),
        "feature_dim": feat_dim,
        "top1_sim_p10": float(np.percentile(top1_sim, 10)),
        "top1_sim_p50": float(np.percentile(top1_sim, 50)),
        "top1_sim_p90": float(np.percentile(top1_sim, 90)),
        "top1_sim_max": float(top1_sim.max()),
        "n_zero_neighbor_rows": n_zero_neighbor,
        "n_unb": n_unb,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
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
        "delta_mean_bag_vs_nb1242": rae_mean_bag - NB1242_REF,
        "beats_nb1070": bool(beats_nb1070),
        "beats_nb1242": bool(beats_nb1242),
        "verdict": verdict,
        "feature_importance_top5": top_features,
        "feature_importance_top5_chembl_only": [
            {"rank": r, "feature": n, "gain_fraction": float(f),
             "raw_gain_sum": float(g)}
            for r, (n, f, g) in enumerate(chembl_pairs[:5], 1)
        ],
        "nb1070_ref_pooled": NB1070_REF,
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
        "n_chembl_pool", "n_chembl_features", "feature_dim",
        "n_zero_neighbor_rows",
        "rae_anchor_nb1070", "per_seed_rae",
        "rae_per_seed_mean", "rae_per_seed_std",
        "rae_mean_bag", "rae_median_bag",
        "delta_mean_bag_vs_nb1070",
        "delta_mean_bag_vs_nb1242",
        "beats_nb1070", "beats_nb1242",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
    print("  feature_importance_top5:")
    for fi in res.get("feature_importance_top5", []):
        print(f"    #{fi['rank']}  {fi['feature']:18s}  "
              f"gain_frac = {fi['gain_fraction']:.4f}")
