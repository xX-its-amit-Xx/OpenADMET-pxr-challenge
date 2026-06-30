"""nb1270 -- Combined ChEMBL + PubChem kNN as residual features.

Hypothesis:
    ChEMBL (945 cpds, median Tanimoto sim ~0.26) and PubChem BioAssay
    (780 cpds, median sim ~0.23) draw from DIFFERENT chemistry distributions:
        - ChEMBL median pEC50 ~5.60 (potent literature ligands, EC50/IC50/Ki/Kd)
        - PubChem median pEC50 ~4.45 (Tox21 qHTS, broader chemical space)
    Unioning them as ONE pool of 1,725 external bioactive compounds may yield
    richer kNN coverage for the 90% novel-scaffold failure tail.  But pEC50
    scales differ -- z-scoring per source PRIOR to union normalizes potency
    into a common "how potent within its source" axis; raw pEC50 averaging is
    also tested as an alternative input.

Protocol:
    1. Load ChEMBL 945 (via nb1242 pipeline) and PubChem 780 (via cached
       nb1263 parquet).  Source tag preserved on each row.
    2. Per source, z-score pec50:   z = (pec50 - mean_src) / std_src.
       Use z-scored pEC50 as kNN target (primary).  Also retain raw pEC50
       as a parallel target for an alt feature.
    3. Union: 1,725 compounds.  InChIKey dedup -- when ChEMBL and PubChem
       overlap on the same InChIKey, keep MEDIAN raw pEC50 and MEDIAN z
       across both sources; src becomes 'union'.  Otherwise carries the
       single-source tag.
    4. Compute Morgan-2048 over standardized SMILES.  Drop test InChIKeys
       (leak guard).
    5. For each of 513 test cpds, top-k=5 Tanimoto NN over the union pool.
       Features per test row:
         (a) pred_z       -- similarity-weighted mean z-scored pEC50
         (b) pred_raw     -- similarity-weighted mean raw pEC50
         (c) mean_sim     -- mean top-5 Tanimoto sim
         (d) src_top1     -- 1 if top-1 NN is ChEMBL else 0 (PubChem)
         (e) std_pec50    -- std of top-5 raw pEC50 (disagreement)
       Total = 5 union-kNN features.
    6. Residual learner: anchor nb1070_pred_oof; residual = y_unb - nb1070;
       features = concat[MACCS-167(unb), 5 union-kNN feats] -> (253, 172).
    7. 5-seed shallow LGBM Huber (identical capacity to nb1242/nb1263),
       5-fold cross-fit per seed, mean-bag pooled RAE.
    8. Verdict at 0.003 margin vs nb1242 (0.5431) and nb1251 (0.5394).

Outputs:
    scripts/nb1270_chembl_pubchem_union.py
    data/processed/nb1270_summary.json
    data/processed/nb1270_mean_bag_oof.npy             (253,) float32
    data/processed/nb1270_per_seed_corrected_oof.npy   (5, 253) float32
    data/processed/nb1270_median_bag_oof.npy           (253,) float32
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

TAG = "nb1270"
ANCHOR = "nb1070"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

# ChEMBL pool input files (same as nb1242)
KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

# PubChem pool: use nb1263 cached parquet (already standardized + deduped).
PUBCHEM_POOL_CACHE = EXT_DIR / "pubchem_pxr_pool.parquet"

KNN_K = 5
SIM_FLOOR = 1e-6

NB1070_REF = 0.5771
NB1242_REF = 0.5431
NB1251_REF = 0.5394
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


def _load_chembl_pool() -> pd.DataFrame:
    """Replicate nb1242 ChEMBL pool build:
    union of raw + nr_extended PXR + pxr_all_types parquets, standardize +
    InChIKey median-aggregate.

    Returns: ['inchikey', 'std_smiles', 'pec50', 'src', 'n_meas'] with
    src='chembl'.
    """
    frames = []

    # ---- 1. Raw CHEMBL3401 dump ----
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
        frames.append(d)
        print(f"   [chembl] CHEMBL3401_raw kept: {len(d)} rows")

    # ---- 2. NR-extended PXR rows ----
    p2 = EXT_DIR / "chembl_nr_extended.parquet"
    if p2.exists():
        d = pd.read_parquet(p2)
        d = d[d["target_name"] == "PXR"].copy()
        d = d[d["pec50"].notna() & d["std_smiles"].notna()].copy()
        d["pec50"] = d["pec50"].astype(float)
        d = d[(d["pec50"] >= 3.0) & (d["pec50"] <= 11.0)].copy()
        d = d[["std_smiles", "pec50"]].rename(columns={"std_smiles": "smiles"})
        frames.append(d)
        print(f"   [chembl] chembl_nr_extended PXR kept: {len(d)} rows")

    # ---- 3. PXR all-types ----
    p3 = EXT_DIR / "chembl_pxr_all_types.parquet"
    if p3.exists():
        d = pd.read_parquet(p3)
        d = d[d["target"] == "PXR"].copy()
        d = d[d["pec50"].notna() & d["smiles"].notna()].copy()
        d["pec50"] = d["pec50"].astype(float)
        d = d[(d["pec50"] >= 3.0) & (d["pec50"] <= 11.0)].copy()
        d = d[["smiles", "pec50"]]
        frames.append(d)
        print(f"   [chembl] chembl_pxr_all_types kept: {len(d)} rows")

    if not frames:
        raise FileNotFoundError("No ChEMBL PXR parquets found in data/external/")

    pool = pd.concat(frames, ignore_index=True)
    print(f"   [chembl] pre-standardize union: {len(pool)} rows")
    mols = pool["smiles"].apply(standardize)
    pool["inchikey"] = mols.apply(_safe_inchikey)
    pool["std_smiles"] = mols.apply(_safe_can_smiles)
    pool = pool[pool["inchikey"].notna() & pool["std_smiles"].notna()].copy()

    agg = (
        pool.groupby("inchikey", as_index=False)
        .agg(
            pec50=("pec50", "median"),
            std_smiles=("std_smiles", "first"),
            n_meas=("pec50", "count"),
        )
    )
    agg["src"] = "chembl"
    print(f"   [chembl] after InChIKey dedup (median): {len(agg)} unique cpds")
    print(f"   [chembl] pec50: mean={agg['pec50'].mean():.3f}  "
          f"std={agg['pec50'].std():.3f}  "
          f"median={agg['pec50'].median():.3f}")
    return agg


def _load_pubchem_pool() -> pd.DataFrame:
    """Load PubChem pool from nb1263 cached parquet (already standardized +
    InChIKey-deduped).
    Returns: ['inchikey', 'std_smiles', 'pec50', 'src', 'n_meas'] with
    src='pubchem'.
    """
    if not PUBCHEM_POOL_CACHE.exists():
        raise FileNotFoundError(
            f"PubChem pool cache missing: {PUBCHEM_POOL_CACHE}. "
            f"Run nb1263 first to populate it."
        )
    pool = pd.read_parquet(PUBCHEM_POOL_CACHE)
    # Standardize columns -- nb1263 saved: inchikey, std_smiles, pec50, src, n_meas
    keep_cols = ["inchikey", "std_smiles", "pec50", "n_meas"]
    pool = pool[keep_cols].copy()
    pool["src"] = "pubchem"
    print(f"   [pubchem] loaded {len(pool)} unique cpds from cache")
    print(f"   [pubchem] pec50: mean={pool['pec50'].mean():.3f}  "
          f"std={pool['pec50'].std():.3f}  "
          f"median={pool['pec50'].median():.3f}")
    return pool


def _build_union_pool() -> pd.DataFrame:
    """Build the combined ChEMBL + PubChem pool with per-source z-scoring,
    then InChIKey union-dedup.

    Returns columns:
        inchikey, std_smiles, pec50_raw, pec50_z, n_meas, src
    where src in {'chembl', 'pubchem', 'both'}.
    """
    print("\n" + "-" * 78)
    print("CHEMBL POOL BUILD")
    print("-" * 78)
    chembl = _load_chembl_pool()
    chembl_mean = float(chembl["pec50"].mean())
    chembl_std = float(chembl["pec50"].std())
    print(f"   [chembl] z-score params: mean={chembl_mean:.4f}  "
          f"std={chembl_std:.4f}")

    print("\n" + "-" * 78)
    print("PUBCHEM POOL BUILD")
    print("-" * 78)
    pubchem = _load_pubchem_pool()
    pubchem_mean = float(pubchem["pec50"].mean())
    pubchem_std = float(pubchem["pec50"].std())
    print(f"   [pubchem] z-score params: mean={pubchem_mean:.4f}  "
          f"std={pubchem_std:.4f}")

    # Per-source z-score
    chembl = chembl.copy()
    chembl["pec50_z"] = (chembl["pec50"] - chembl_mean) / chembl_std
    chembl = chembl.rename(columns={"pec50": "pec50_raw"})

    pubchem = pubchem.copy()
    pubchem["pec50_z"] = (pubchem["pec50"] - pubchem_mean) / pubchem_std
    pubchem = pubchem.rename(columns={"pec50": "pec50_raw"})

    print("\n" + "-" * 78)
    print("UNION + INCHIKEY DEDUP")
    print("-" * 78)
    print(f"   pre-union counts: chembl={len(chembl)}  pubchem={len(pubchem)}  "
          f"sum={len(chembl) + len(pubchem)}")

    combined = pd.concat([chembl, pubchem], ignore_index=True)
    n_concat = len(combined)

    # InChIKey-group: median across measurements; if both sources contribute,
    # tag src='both', else keep single source.
    def _agg_group(g: pd.DataFrame) -> pd.Series:
        srcs = set(g["src"].tolist())
        if srcs == {"chembl", "pubchem"}:
            src_tag = "both"
        elif "chembl" in srcs:
            src_tag = "chembl"
        else:
            src_tag = "pubchem"
        return pd.Series({
            "std_smiles": g["std_smiles"].iloc[0],
            "pec50_raw": float(np.median(g["pec50_raw"])),
            "pec50_z": float(np.median(g["pec50_z"])),
            "n_meas": int(g["n_meas"].sum()),
            "src": src_tag,
        })

    agg = combined.groupby("inchikey", as_index=False).apply(
        _agg_group, include_groups=False
    )
    print(f"   post-union InChIKey-dedup: {len(agg)} cpds  "
          f"(removed {n_concat - len(agg)} cross-source dups)")
    src_counts = agg["src"].value_counts().to_dict()
    print(f"   src breakdown: {src_counts}")
    print(f"   pec50_raw: mean={agg['pec50_raw'].mean():.3f}  "
          f"std={agg['pec50_raw'].std():.3f}  "
          f"median={agg['pec50_raw'].median():.3f}")
    print(f"   pec50_z:   mean={agg['pec50_z'].mean():.3f}  "
          f"std={agg['pec50_z'].std():.3f}  "
          f"median={agg['pec50_z'].median():.3f}")

    # Attach z-score params on the return frame's metadata via a separate dict
    agg.attrs["chembl_mean"] = chembl_mean
    agg.attrs["chembl_std"] = chembl_std
    agg.attrs["pubchem_mean"] = pubchem_mean
    agg.attrs["pubchem_std"] = pubchem_std
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
                  pool_z: np.ndarray, pool_raw: np.ndarray,
                  pool_src: np.ndarray,
                  fb_z: float, fb_raw: float) -> dict:
    """Build the 5 union-kNN features per test row."""
    w = np.clip(top_sim.copy(), 0.0, 1.0)
    w_sum = w.sum(axis=1)
    n_q = top_idx.shape[0]
    pred_z = np.empty(n_q, dtype=np.float32)
    pred_raw = np.empty(n_q, dtype=np.float32)
    std_pec50 = np.empty(n_q, dtype=np.float32)
    src_top1 = np.empty(n_q, dtype=np.float32)  # 1.0 if ChEMBL else 0.0
    for i in range(n_q):
        if w_sum[i] < SIM_FLOOR:
            pred_z[i] = fb_z
            pred_raw[i] = fb_raw
            std_pec50[i] = 0.0
            src_top1[i] = 0.0
        else:
            pred_z[i] = float(np.sum(w[i] * pool_z[top_idx[i]]) / w_sum[i])
            pred_raw[i] = float(np.sum(w[i] * pool_raw[top_idx[i]]) / w_sum[i])
            std_pec50[i] = float(np.std(pool_raw[top_idx[i]]))
            # src_top1: 1 if top-1 NN came from ChEMBL (or 'both' source), 0 otherwise
            top1_src = pool_src[top_idx[i, 0]]
            src_top1[i] = 1.0 if top1_src in ("chembl", "both") else 0.0
    mean_sim = top_sim.mean(axis=1).astype(np.float32)
    return dict(
        pred_z=pred_z,
        pred_raw=pred_raw,
        mean_sim=mean_sim,
        src_top1=src_top1,
        std_pec50=std_pec50,
    )


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
    print(f"{TAG} -- Combined ChEMBL + PubChem kNN union residual feature; "
          f"shallow residual-LGBM bag on nb1070 anchor")
    print(f"          seeds = {RESID_SEEDS}  folds = {RESID_FOLDS}")
    print(f"          features = MACCS-167 + 5 union-kNN feats  (172)")
    print("=" * 78)

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

    # ---- Union pool ----
    pool = _build_union_pool()
    n_pool_pre_leak = len(pool)
    chembl_mean = pool.attrs["chembl_mean"]
    chembl_std = pool.attrs["chembl_std"]
    pubchem_mean = pool.attrs["pubchem_mean"]
    pubchem_std = pool.attrs["pubchem_std"]

    # ---- Test InChIKey leak guard ----
    print("\n" + "-" * 78)
    print("TEST-SET LEAK GUARD (drop union cpds whose InChIKey appears in 513)")
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
    print(f"   pool: {n_before} -> {n_after}  "
          f"(dropped {n_before - n_after} test-overlapping cpds)")

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
    pool_raw = pool["pec50_raw"].to_numpy(dtype=np.float32)
    pool_z = pool["pec50_z"].to_numpy(dtype=np.float32)
    pool_src = pool["src"].to_numpy()
    pool_raw_median = float(np.median(pool_raw))
    pool_z_median = float(np.median(pool_z))
    print(f"   final pool size: {len(pool)}")
    print(f"   pool median pEC50_raw = {pool_raw_median:.3f}")
    print(f"   pool median pEC50_z   = {pool_z_median:.3f}")

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
    print(f"TANIMOTO kNN (k={KNN_K}) -- test (513) vs union pool ({len(pool)})")
    print("-" * 78)
    top_idx, top_sim = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    feats = _knn_features(
        top_idx, top_sim, pool_z, pool_raw, pool_src,
        fb_z=pool_z_median, fb_raw=pool_raw_median,
    )
    pred_z = feats["pred_z"]
    pred_raw = feats["pred_raw"]
    mean_sim = feats["mean_sim"]
    src_top1 = feats["src_top1"]
    std_pec50 = feats["std_pec50"]
    top1_sim = top_sim[:, 0]

    print(f"   pred_z      mean={pred_z.mean():.3f}  std={pred_z.std():.3f}  "
          f"min={pred_z.min():.3f}  max={pred_z.max():.3f}")
    print(f"   pred_raw    mean={pred_raw.mean():.3f}  std={pred_raw.std():.3f}  "
          f"min={pred_raw.min():.3f}  max={pred_raw.max():.3f}")
    print(f"   top1 sim    p10={np.percentile(top1_sim, 10):.3f}  "
          f"p50={np.percentile(top1_sim, 50):.3f}  "
          f"p90={np.percentile(top1_sim, 90):.3f}  "
          f"max={top1_sim.max():.3f}")
    print(f"   mean5 sim   p10={np.percentile(mean_sim, 10):.3f}  "
          f"p50={np.percentile(mean_sim, 50):.3f}  "
          f"p90={np.percentile(mean_sim, 90):.3f}")
    print(f"   src_top1    chembl-or-both rate = {src_top1.mean():.3f}  "
          f"(1.0 = ChEMBL/both, 0.0 = PubChem)")
    print(f"   std_pec50   p10={np.percentile(std_pec50, 10):.3f}  "
          f"p50={np.percentile(std_pec50, 50):.3f}  "
          f"p90={np.percentile(std_pec50, 90):.3f}")

    n_zero_neighbor = int((top1_sim < SIM_FLOOR).sum())
    print(f"   {n_zero_neighbor}/513 test rows had no neighbor "
          f"(fell back to pool median)")

    # ---- MACCS-167 (unblind slice) ----
    X_maccs_te = np.load(MACCS_TE_PATH)
    if X_maccs_te.shape[0] != n_test:
        raise ValueError(f"MACCS cache shape mismatch: {X_maccs_te.shape}")
    X_maccs_unb = X_maccs_te[unb_idx].astype(np.float32)
    print(f"   MACCS unb shape = {X_maccs_unb.shape}")

    # ---- Build residual feature matrix on 253 ----
    feats_unb = np.column_stack([
        pred_z[unb_idx].astype(np.float32),
        pred_raw[unb_idx].astype(np.float32),
        mean_sim[unb_idx].astype(np.float32),
        src_top1[unb_idx].astype(np.float32),
        std_pec50[unb_idx].astype(np.float32),
    ])
    X_unb = np.concatenate([X_maccs_unb, feats_unb], axis=1).astype(np.float32)
    feat_dim = X_unb.shape[1]
    print(f"   residual feature matrix: {X_unb.shape}  "
          f"(MACCS-167 + 5 union-kNN feats)")

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
    print(f"   nb1242 ref             = {NB1242_REF:.4f}  (ChEMBL kNN residual)")
    print(f"   nb1251 ref             = {NB1251_REF:.4f}  (ChEMBL bob blend)")

    beats_nb1070 = rae_mean_bag < rae_anchor - DECISION_MARGIN
    beats_nb1242 = rae_mean_bag < NB1242_REF - DECISION_MARGIN
    beats_nb1251 = rae_mean_bag < NB1251_REF - DECISION_MARGIN

    if beats_nb1251:
        verdict = "UNION_KNN_BEATS_NB1251_NEW_PRIMARY_CANDIDATE"
    elif beats_nb1242:
        verdict = "UNION_KNN_BEATS_NB1242_BUT_NOT_NB1251"
    elif beats_nb1070:
        verdict = "UNION_KNN_HELPS_NB1070_BUT_NOT_NB1242"
    elif abs(rae_mean_bag - rae_anchor) < DECISION_MARGIN:
        verdict = "UNION_KNN_FLAT_NO_NEW_SIGNAL"
    else:
        verdict = "UNION_KNN_HURTS_NB1070"
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

    src_counts = pool["src"].value_counts().to_dict()

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "data_source": "chembl_local_caches + pubchem_pxr_pool.parquet",
        "n_chembl_input": None,  # filled below
        "n_pubchem_input": None,
        "n_union_pre_leakguard": int(n_pool_pre_leak),
        "n_union_post_leakguard": int(n_after),
        "n_union_final": int(len(pool)),
        "test_inchikeys_in_pool_dropped": int(n_before - n_after),
        "src_breakdown_final": {str(k): int(v) for k, v in src_counts.items()},
        "chembl_zscore_mean": chembl_mean,
        "chembl_zscore_std": chembl_std,
        "pubchem_zscore_mean": pubchem_mean,
        "pubchem_zscore_std": pubchem_std,
        "pool_raw_median": pool_raw_median,
        "pool_z_median": pool_z_median,
        "knn_k": KNN_K,
        "top1_sim_p10": float(np.percentile(top1_sim, 10)),
        "top1_sim_p50": float(np.percentile(top1_sim, 50)),
        "top1_sim_p90": float(np.percentile(top1_sim, 90)),
        "top1_sim_max": float(top1_sim.max()),
        "mean5_sim_p10": float(np.percentile(mean_sim, 10)),
        "mean5_sim_p50": float(np.percentile(mean_sim, 50)),
        "mean5_sim_p90": float(np.percentile(mean_sim, 90)),
        "src_top1_chembl_or_both_rate": float(src_top1.mean()),
        "n_zero_neighbor_rows": n_zero_neighbor,
        "n_unb": n_unb,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "feature_dim": feat_dim,
        "knn_feature_list": [
            "pred_z", "pred_raw", "mean_sim", "src_top1", "std_pec50"
        ],
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
        "n_union_pre_leakguard", "n_union_final", "src_breakdown_final",
        "chembl_zscore_mean", "chembl_zscore_std",
        "pubchem_zscore_mean", "pubchem_zscore_std",
        "top1_sim_p10", "top1_sim_p50", "top1_sim_p90",
        "src_top1_chembl_or_both_rate",
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
