"""nb1273 -- Train (4139) + ChEMBL (945) UNION as kNN reference pool.

Hypothesis:
    The 4139-row PXR training set itself is the highest-relevance neighbor
    pool (assay-matched, same lab, same readout).  External ChEMBL extends
    scaffold coverage.  A UNIFIED train+ChEMBL pool may strictly dominate
    either alone because:
      - train pool gives assay-faithful neighbors when sim is high
      - ChEMBL gives scaffold-diverse neighbors when train sim is low
    The frac_5_from_chembl feature exposes regime to the residual learner,
    which can up-weight the appropriate signal per row.

Pipeline:
    1. Load 4139 PXR train compounds + pec50 (src.pxr.data.load_train).
    2. Load 945 ChEMBL PXR compounds + pec50 (reuses nb1242 pool builder).
    3. Standardize + InChIKey dedup ChEMBL vs train (train wins on collision).
    4. Test-set leak guard: drop any union compound whose InChIKey appears in 513.
    5. Morgan-2048 over union; top-5 Tanimoto kNN per 513 test row.
    6. Features (5 total): mean_5_pec50 (sim-weighted), mean_5_sim,
       frac_5_from_chembl, top1_sim, top1_pec50.
    7. Residual learner: anchor nb1070; features = MACCS-167 + 5 union-kNN
       -> 172 cols.  5-seed bag shallow LGBM Huber, 5-fold cross-fit
       (identical capacity to nb1242 for honest A/B comparison).
    8. Verdict at 0.003 margin vs nb1242 (0.5431) and vs nb1251 (0.5394).

Outputs:
    scripts/nb1273_train_chembl_union.py
    data/processed/nb1273_summary.json
    data/processed/nb1273_mean_bag_oof.npy          (253,) float32
    data/processed/nb1273_per_seed_corrected_oof.npy (5, 253) float32
    data/processed/nb1273_median_bag_oof.npy        (253,) float32
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
from pxr.data import load_train, load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1273"
ANCHOR = "nb1070"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

NB1070_REF = 0.5771
NB1242_REF = 0.5431          # ChEMBL-only kNN residual bag (mean_bag)
NB1251_REF = 0.5394          # ChEMBL/nb1211 best-fixed-w blend
DECISION_MARGIN = 0.003


def _safe_inchikey(mol) -> str | None:
    try:
        return Chem.MolToInchiKey(mol) if mol is not None else None
    except Exception:
        return None


def _safe_can_smiles(mol) -> str | None:
    try:
        return Chem.MolToSmiles(mol) if mol is not None else None
    except Exception:
        return None


def _load_train_pool() -> pd.DataFrame:
    """4139 PXR train compounds -> (inchikey, std_smiles, pec50, src='train').

    Aggregates by InChIKey using median pec50 if duplicate molecule rows exist.
    """
    df = load_train()
    df = df[df["pec50"].notna() & df["smiles"].notna()].copy()
    mols = df["smiles"].apply(standardize)
    df["inchikey"] = mols.apply(_safe_inchikey)
    df["std_smiles"] = mols.apply(_safe_can_smiles)
    df = df[df["inchikey"].notna() & df["std_smiles"].notna()].copy()
    df["pec50"] = df["pec50"].astype(float)

    agg = (
        df.groupby("inchikey", as_index=False)
        .agg(pec50=("pec50", "median"),
             std_smiles=("std_smiles", "first"),
             n_meas=("pec50", "count"))
    )
    agg["src"] = "train"
    print(f"   [train] {len(df)} rows -> {len(agg)} unique InChIKeys")
    print(f"   [train] pec50: mean={agg['pec50'].mean():.3f}  "
          f"std={agg['pec50'].std():.3f}  "
          f"min={agg['pec50'].min():.3f}  max={agg['pec50'].max():.3f}")
    return agg


def _load_chembl_pool() -> pd.DataFrame:
    """Union three local ChEMBL PXR caches -> (inchikey, std_smiles, pec50, src).

    Identical filtering to nb1242 for honest A/B comparison.
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
        print(f"   [chembl src] CHEMBL3401_raw kept: {len(d)} rows")

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
        print(f"   [chembl src] chembl_nr_extended PXR kept: {len(d)} rows")

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
        print(f"   [chembl src] chembl_pxr_all_types kept: {len(d)} rows")

    if not frames:
        raise FileNotFoundError("No local ChEMBL PXR parquets found")

    pool = pd.concat(frames, ignore_index=True)
    print(f"   [chembl] pre-standardize union: {len(pool)} rows")

    mols = pool["smiles"].apply(standardize)
    pool["inchikey"] = mols.apply(_safe_inchikey)
    pool["std_smiles"] = mols.apply(_safe_can_smiles)
    pool = pool[pool["inchikey"].notna() & pool["std_smiles"].notna()].copy()

    agg = (
        pool.groupby("inchikey", as_index=False)
        .agg(pec50=("pec50", "median"),
             std_smiles=("std_smiles", "first"),
             n_meas=("pec50", "count"))
    )
    agg["src"] = "chembl"
    print(f"   [chembl] after InChIKey dedup: {len(agg)} unique cpds")
    print(f"   [chembl] pec50: mean={agg['pec50'].mean():.3f}  "
          f"std={agg['pec50'].std():.3f}")
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


def _knn_features(top_idx: np.ndarray, top_sim: np.ndarray,
                  pool_labels: np.ndarray, pool_is_chembl: np.ndarray,
                  fallback: float):
    """Build the 5 union-kNN features per test row.

    Returns dict with arrays of shape (n_q,):
      mean_5_pec50          sim-weighted mean pec50 over top-5
      mean_5_sim            arithmetic mean of top-5 sim
      frac_5_from_chembl    count of chembl rows in top-5 / 5
      top1_sim              best (largest) sim
      top1_pec50            pec50 of the top-1 neighbor (or fallback if no nbr)
    """
    n_q = top_idx.shape[0]
    w = np.clip(top_sim, 0.0, 1.0)
    w_sum = w.sum(axis=1)

    mean_5_pec50 = np.empty(n_q, dtype=np.float32)
    top1_pec50 = np.empty(n_q, dtype=np.float32)
    frac_5_from_chembl = np.empty(n_q, dtype=np.float32)
    for i in range(n_q):
        idx_i = top_idx[i]
        if w_sum[i] < SIM_FLOOR:
            mean_5_pec50[i] = fallback
            top1_pec50[i] = fallback
        else:
            mean_5_pec50[i] = np.sum(w[i] * pool_labels[idx_i]) / w_sum[i]
            top1_pec50[i] = float(pool_labels[idx_i[0]])
        frac_5_from_chembl[i] = float(pool_is_chembl[idx_i].sum()) / KNN_K

    mean_5_sim = top_sim.mean(axis=1).astype(np.float32)
    top1_sim = top_sim[:, 0].astype(np.float32)
    return {
        "mean_5_pec50": mean_5_pec50,
        "mean_5_sim": mean_5_sim,
        "frac_5_from_chembl": frac_5_from_chembl,
        "top1_sim": top1_sim,
        "top1_pec50": top1_pec50,
    }


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
                                 seed: int) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- TRAIN+ChEMBL UNION kNN reference pool; "
          f"shallow residual-LGBM bag on nb1070 anchor")
    print(f"          seeds = {RESID_SEEDS}  folds = {RESID_FOLDS}")
    print(f"          features = MACCS-167 + 5 union-kNN cols (172)")
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

    # ---- Train pool ----
    print("\n" + "-" * 78)
    print("TRAIN POOL (4139 assay-matched PXR compounds)")
    print("-" * 78)
    train_pool = _load_train_pool()
    n_train_pool_init = len(train_pool)

    # ---- ChEMBL pool ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL (local cache; same filter as nb1242)")
    print("-" * 78)
    chembl_pool = _load_chembl_pool()
    n_chembl_pool_init = len(chembl_pool)

    # ---- Union: train wins on collision ----
    print("\n" + "-" * 78)
    print("UNION (train PRIORITIZED over ChEMBL on InChIKey collision)")
    print("-" * 78)
    train_keys = set(train_pool["inchikey"].tolist())
    chembl_only = chembl_pool[~chembl_pool["inchikey"].isin(train_keys)].copy()
    n_chembl_collision = n_chembl_pool_init - len(chembl_only)
    print(f"   train pool   = {n_train_pool_init}")
    print(f"   chembl pool  = {n_chembl_pool_init}  (collisions with train: "
          f"{n_chembl_collision} -> kept {len(chembl_only)})")
    union = pd.concat(
        [train_pool[["inchikey", "std_smiles", "pec50", "src"]],
         chembl_only[["inchikey", "std_smiles", "pec50", "src"]]],
        ignore_index=True,
    )
    print(f"   UNION size   = {len(union)}")

    # ---- Test InChIKey leak guard ----
    print("\n" + "-" * 78)
    print("TEST-SET LEAK GUARD (drop union cpds whose InChIKey in 513)")
    print("-" * 78)
    test_mols = [standardize(s) for s in test_smiles]
    test_inchikeys = set()
    for m in test_mols:
        ik = _safe_inchikey(m)
        if ik is not None:
            test_inchikeys.add(ik)
    n_before = len(union)
    union = union[~union["inchikey"].isin(test_inchikeys)].reset_index(drop=True)
    n_after = len(union)
    print(f"   {n_before} -> {n_after}  (dropped {n_before - n_after} test-overlap)")

    # ---- Morgan fingerprints ----
    print("\n" + "-" * 78)
    print("MORGAN-2048 FINGERPRINTS")
    print("-" * 78)
    fp_pool = morgan_fp_batch(union["std_smiles"].tolist())
    print(f"   union FP: {fp_pool.shape}  density={fp_pool.mean():.4f}")
    keep_pool = fp_pool.sum(axis=1) > 0
    if not keep_pool.all():
        n_drop = int((~keep_pool).sum())
        print(f"   dropped {n_drop} union rows with zero FP")
        union = union[keep_pool].reset_index(drop=True)
        fp_pool = fp_pool[keep_pool]
    pool_labels = union["pec50"].to_numpy(dtype=np.float32)
    pool_is_chembl = (union["src"].to_numpy() == "chembl").astype(np.int8)
    pool_median = float(np.median(pool_labels))
    n_union_final = len(union)
    print(f"   final union size: {n_union_final}  "
          f"(train={int((pool_is_chembl==0).sum())} / "
          f"chembl={int((pool_is_chembl==1).sum())})  "
          f"median pEC50 = {pool_median:.3f}")

    std_test_smiles = []
    for m in test_mols:
        if m is None:
            std_test_smiles.append("")
        else:
            std_test_smiles.append(Chem.MolToSmiles(m))
    fp_test = morgan_fp_batch(std_test_smiles)
    print(f"   test FP: {fp_test.shape}  density={fp_test.mean():.4f}")

    # ---- kNN k=5 Tanimoto ----
    print("\n" + "-" * 78)
    print(f"TANIMOTO kNN (k={KNN_K}) -- test (513) vs UNION ({n_union_final})")
    print("-" * 78)
    top_idx, top_sim = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    feats = _knn_features(top_idx, top_sim, pool_labels, pool_is_chembl,
                          fallback=pool_median)
    mean_5_pec50 = feats["mean_5_pec50"]
    mean_5_sim = feats["mean_5_sim"]
    frac_5_from_chembl = feats["frac_5_from_chembl"]
    top1_sim = feats["top1_sim"]
    top1_pec50 = feats["top1_pec50"]

    # top-1 src breakdown
    top1_is_chembl = pool_is_chembl[top_idx[:, 0]]
    n_top1_chembl = int(top1_is_chembl.sum())
    n_top1_train = int((1 - top1_is_chembl).sum())
    frac_top1_train = float(n_top1_train) / n_test
    frac_top1_chembl = float(n_top1_chembl) / n_test

    print(f"   top1 source breakdown:")
    print(f"      from train  : {n_top1_train}/{n_test} = {frac_top1_train:.3f}")
    print(f"      from chembl : {n_top1_chembl}/{n_test} = {frac_top1_chembl:.3f}")
    print(f"   top1 sim   p10={np.percentile(top1_sim, 10):.3f}  "
          f"p50={np.percentile(top1_sim, 50):.3f}  "
          f"p90={np.percentile(top1_sim, 90):.3f}  "
          f"max={top1_sim.max():.3f}")
    print(f"   mean5 sim  p10={np.percentile(mean_5_sim, 10):.3f}  "
          f"p50={np.percentile(mean_5_sim, 50):.3f}  "
          f"p90={np.percentile(mean_5_sim, 90):.3f}")
    print(f"   mean5 pec50 mean={mean_5_pec50.mean():.3f}  "
          f"std={mean_5_pec50.std():.3f}  "
          f"min={mean_5_pec50.min():.3f}  max={mean_5_pec50.max():.3f}")
    print(f"   frac_5_from_chembl: mean={frac_5_from_chembl.mean():.3f}  "
          f"p50={np.percentile(frac_5_from_chembl, 50):.3f}")
    n_zero_neighbor = int((top1_sim < SIM_FLOOR).sum())
    print(f"   {n_zero_neighbor}/{n_test} test rows had no neighbor")

    # ---- MACCS-167 (unblind slice) ----
    X_maccs_te = np.load(MACCS_TE_PATH)
    if X_maccs_te.shape[0] != n_test:
        raise ValueError(f"MACCS cache shape mismatch: {X_maccs_te.shape}")
    X_maccs_unb = X_maccs_te[unb_idx].astype(np.float32)
    print(f"   MACCS unb shape = {X_maccs_unb.shape}")

    # ---- Build residual feature matrix on 253 ----
    knn_cols_unb = np.stack(
        [mean_5_pec50[unb_idx], mean_5_sim[unb_idx],
         frac_5_from_chembl[unb_idx], top1_sim[unb_idx], top1_pec50[unb_idx]],
        axis=1,
    ).astype(np.float32)
    X_unb = np.concatenate([X_maccs_unb, knn_cols_unb], axis=1).astype(np.float32)
    feat_dim = X_unb.shape[1]
    print(f"   residual feature matrix: {X_unb.shape}  "
          f"(MACCS-167 + 5 union-kNN)")

    # ---- Per-seed residual cross-fit ----
    print("\n" + "-" * 78)
    print(f"PER-SEED RESIDUAL CROSS-FIT (depth=3 shallow LGBM Huber, dim={feat_dim})")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []
    for i, s in enumerate(RESID_SEEDS):
        resid_oof_s = _residual_cross_fit_one_seed(X_unb, residual, s)
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

    print("\n" + "-" * 78)
    print("BAG AGGREGATIONS")
    print("-" * 78)
    print(f"   per-seed RAE list      = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   per-seed mean          = {rae_per_seed_mean:.4f}")
    print(f"   per-seed median        = {rae_per_seed_median:.4f}")
    print(f"   per-seed std           = {rae_per_seed_std:.4f}")
    print(f"   pooled RAE(mean_bag)   = {rae_mean_bag:.4f}  "
          f"(d_vs_nb1070 = {rae_mean_bag - rae_anchor:+.4f})")
    print(f"   pooled RAE(median_bag) = {rae_median_bag:.4f}  "
          f"(d_vs_nb1070 = {rae_median_bag - rae_anchor:+.4f})")
    print(f"   nb1242 ref             = {NB1242_REF:.4f}  (ChEMBL-only kNN bag)")
    print(f"   nb1251 ref             = {NB1251_REF:.4f}  (best fixed-w blend)")

    beats_nb1070 = rae_mean_bag < rae_anchor - DECISION_MARGIN
    beats_nb1242 = rae_mean_bag < NB1242_REF - DECISION_MARGIN
    beats_nb1251 = rae_mean_bag < NB1251_REF - DECISION_MARGIN

    if beats_nb1251:
        verdict = "UNION_KNN_FEAT_BEATS_NB1251_NEW_PRIMARY_CANDIDATE"
    elif beats_nb1242:
        verdict = "UNION_KNN_FEAT_BEATS_NB1242_BUT_NOT_NB1251"
    elif beats_nb1070:
        verdict = "UNION_KNN_FEAT_HELPS_NB1070_BUT_NOT_NB1242"
    elif abs(rae_mean_bag - NB1242_REF) < DECISION_MARGIN:
        verdict = "UNION_KNN_FEAT_FLAT_VS_NB1242"
    else:
        verdict = "UNION_KNN_FEAT_HURTS_VS_NB1242"
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
        "data_source": "train_4139_union_chembl_945",
        "train_pool_init": int(n_train_pool_init),
        "chembl_pool_init": int(n_chembl_pool_init),
        "chembl_dropped_train_collision": int(n_chembl_collision),
        "union_pre_leak_guard": int(n_before),
        "union_after_leak_guard": int(n_after),
        "test_inchikeys_in_union_dropped": int(n_before - n_after),
        "union_final_size": int(n_union_final),
        "union_n_from_train": int((pool_is_chembl == 0).sum()),
        "union_n_from_chembl": int((pool_is_chembl == 1).sum()),
        "pool_pec50_mean": float(pool_labels.mean()),
        "pool_pec50_std": float(pool_labels.std()),
        "pool_pec50_median": pool_median,
        "knn_k": KNN_K,
        "top1_from_train": int(n_top1_train),
        "top1_from_chembl": int(n_top1_chembl),
        "frac_top1_from_train": frac_top1_train,
        "frac_top1_from_chembl": frac_top1_chembl,
        "top1_sim_p10": float(np.percentile(top1_sim, 10)),
        "top1_sim_p50": float(np.percentile(top1_sim, 50)),
        "top1_sim_p90": float(np.percentile(top1_sim, 90)),
        "top1_sim_max": float(top1_sim.max()),
        "mean5_sim_p10": float(np.percentile(mean_5_sim, 10)),
        "mean5_sim_p50": float(np.percentile(mean_5_sim, 50)),
        "mean5_sim_p90": float(np.percentile(mean_5_sim, 90)),
        "mean5_pec50_mean": float(mean_5_pec50.mean()),
        "mean5_pec50_std": float(mean_5_pec50.std()),
        "frac_5_from_chembl_mean": float(frac_5_from_chembl.mean()),
        "frac_5_from_chembl_p50": float(np.percentile(frac_5_from_chembl, 50)),
        "n_zero_neighbor_rows": int(n_zero_neighbor),
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
        "delta_mean_bag_vs_nb1242": rae_mean_bag - NB1242_REF,
        "delta_mean_bag_vs_nb1251": rae_mean_bag - NB1251_REF,
        "beats_nb1070": bool(beats_nb1070),
        "beats_nb1242": bool(beats_nb1242),
        "beats_nb1251": bool(beats_nb1251),
        "verdict": verdict,
        "nb1070_ref_pooled": NB1070_REF,
        "nb1242_ref": NB1242_REF,
        "nb1251_ref": NB1251_REF,
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
        "union_final_size", "union_n_from_train", "union_n_from_chembl",
        "frac_top1_from_train", "frac_top1_from_chembl",
        "top1_sim_p10", "top1_sim_p50", "top1_sim_p90",
        "n_zero_neighbor_rows",
        "rae_anchor_nb1070", "per_seed_rae",
        "rae_per_seed_mean", "rae_per_seed_std",
        "rae_mean_bag", "rae_median_bag",
        "delta_mean_bag_vs_nb1070",
        "delta_mean_bag_vs_nb1242",
        "delta_mean_bag_vs_nb1251",
        "beats_nb1070", "beats_nb1242", "beats_nb1251",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
