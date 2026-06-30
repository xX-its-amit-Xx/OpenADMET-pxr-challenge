"""nb1304 -- Train-augmented kNN pool: PXR train + ChEMBL + PubChem + BindingDB
ALL unioned as ONE 7000+ row external/train pool, k=5 NN as residual features.

Hypothesis:
    nb1270 (ChEMBL+PubChem union, 1,725 cpds) and nb1272 (BindingDB) each carry
    independent external chemistry. nb1242 (ChEMBL alone) hit 0.5431; nb1251
    (blend with internal BoB) reached 0.5394.  What we have NOT yet tried is
    pulling the 4,139 PXR-train pEC50 labels into the same union pool, so the
    kNN can reach for a TRAIN neighbor (the strongest signal, since train and
    test share the assay) FIRST, and only fall back to external sources when
    no good train neighbor exists.  Mixing all sources in one pool lets the
    Tanimoto top-k pick the best neighbor regardless of source -- the train
    labels dominate when they're close, the externals fill in for novel
    scaffolds.

    Engineering bet: frac_from_train is a routing signal -- it's a binary
    "top-1 came from PXR-train" flag the residual learner can condition on
    (train neighbor -> trust the anchor; external neighbor -> trust the
    external pec50).

Protocol:
  1. Load PXR train (4,139 dose-response cpds, pEC50 column).
  2. Load ChEMBL pool (~945 unique cpds) via nb1242's pipeline.
  3. Load PubChem pool (~780 unique cpds) via nb1263 cached parquet.
  4. Load BindingDB PXR pool (~208 unique cpds) via nb1272's pipeline.
  5. Union all four sources, standardize SMILES, dedupe by InChIKey:
       priority for retained pEC50: PXR train (assay-matched) > median of
       any other present source.  Source-tag retained.
  6. Compute Morgan-2048 over the unioned pool and the 513 test cpds.
  7. Test InChIKey leak guard (drop any pool row whose InChIKey is in 513).
  8. For each 513 test row, top-k=5 Tanimoto NN over the union pool.
     Per-row features (3 union-kNN feats):
       (a) pred_pec50        -- similarity-weighted mean pool pEC50
       (b) mean_sim          -- mean top-5 Tanimoto sim
       (c) frac_from_train   -- 1 if top-1 NN is from PXR-train else 0
  9. Residual learner: anchor nb1070_pred_oof on 253 unblind; residual =
     y_unb - anchor; features = MACCS-167 + 3 union-kNN feats -> (253, 170).
 10. 5-seed shallow LGBM Huber bag (identical capacity to nb1242/nb1270),
     5-fold cross-fit per seed, mean-bag pooled RAE.
 11. Verdict at 0.003 margin vs nb1242 (0.5431) and nb1251 (0.5394).

Outputs:
  scripts/nb1304_train_all_external_union.py     (this file)
  data/processed/nb1304_summary.json
  data/processed/nb1304_mean_bag_oof.npy             (253,) float32
  data/processed/nb1304_per_seed_corrected_oof.npy   (5, 253) float32
  data/processed/nb1304_median_bag_oof.npy           (253,) float32
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

TAG = "nb1304"
ANCHOR = "nb1070"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

# ChEMBL filtering rules (same as nb1242/nb1270)
KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
PEC50_LOW = 3.0
PEC50_HIGH = 11.0

# PubChem pool cache (built by nb1263)
PUBCHEM_POOL_CACHE = EXT_DIR / "pubchem_pxr_pool.parquet"

# BindingDB NR cache (filtered to PXR by uniprot O75469)
BDB_NR_PARQUET = EXT_DIR / "bindingdb_nr_data.parquet"
PXR_UNIPROT = "O75469"

KNN_K = 5
SIM_FLOOR = 1e-6

# Reference numbers (pooled RAE on 253 unblind)
NB1070_REF = 0.5771
NB1242_REF = 0.5431      # ChEMBL kNN residual
NB1251_REF = 0.5394      # ChEMBL kNN residual + nb1211 blend, best fixed-w
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


# ---------------------------------------------------------------------------
# Per-source loaders -- each returns columns ['inchikey', 'std_smiles', 'pec50',
# 'n_meas', 'src']
# ---------------------------------------------------------------------------
def _load_train_pool() -> pd.DataFrame:
    """4,139 PXR train cpds -> per-InChIKey median pEC50."""
    df = load_train()
    df = df[df["pec50"].notna() & df["smiles"].notna()].copy()
    df["pec50"] = df["pec50"].astype(float)
    print(f"   [train] raw train rows w/ pec50: {len(df)}")
    mols = df["smiles"].apply(standardize)
    df["inchikey"] = mols.apply(_safe_inchikey)
    df["std_smiles"] = mols.apply(_safe_can_smiles)
    df = df[df["inchikey"].notna() & df["std_smiles"].notna()].copy()
    print(f"   [train] after RDKit standardize: {len(df)} rows")
    agg = (
        df.groupby("inchikey", as_index=False)
        .agg(
            pec50=("pec50", "median"),
            std_smiles=("std_smiles", "first"),
            n_meas=("pec50", "count"),
        )
    )
    agg["src"] = "train"
    print(f"   [train] after InChIKey dedup (median): {len(agg)} unique cpds")
    print(f"   [train] pec50: mean={agg['pec50'].mean():.3f}  "
          f"std={agg['pec50'].std():.3f}  median={agg['pec50'].median():.3f}")
    return agg


def _load_chembl_pool() -> pd.DataFrame:
    """Union of three ChEMBL PXR caches -> ~945 unique cpds (median pEC50)."""
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
        frames.append(d)
        print(f"   [chembl] CHEMBL3401_raw kept: {len(d)} rows")

    p2 = EXT_DIR / "chembl_nr_extended.parquet"
    if p2.exists():
        d = pd.read_parquet(p2)
        d = d[d["target_name"] == "PXR"].copy()
        d = d[d["pec50"].notna() & d["std_smiles"].notna()].copy()
        d["pec50"] = d["pec50"].astype(float)
        d = d[(d["pec50"] >= PEC50_LOW) & (d["pec50"] <= PEC50_HIGH)].copy()
        d = d[["std_smiles", "pec50"]].rename(columns={"std_smiles": "smiles"})
        frames.append(d)
        print(f"   [chembl] chembl_nr_extended PXR kept: {len(d)} rows")

    p3 = EXT_DIR / "chembl_pxr_all_types.parquet"
    if p3.exists():
        d = pd.read_parquet(p3)
        d = d[d["target"] == "PXR"].copy()
        d = d[d["pec50"].notna() & d["smiles"].notna()].copy()
        d["pec50"] = d["pec50"].astype(float)
        d = d[(d["pec50"] >= PEC50_LOW) & (d["pec50"] <= PEC50_HIGH)].copy()
        d = d[["smiles", "pec50"]]
        frames.append(d)
        print(f"   [chembl] chembl_pxr_all_types kept: {len(d)} rows")

    if not frames:
        raise FileNotFoundError("No ChEMBL PXR parquets found in data/external/")

    pool = pd.concat(frames, ignore_index=True)
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
          f"std={agg['pec50'].std():.3f}  median={agg['pec50'].median():.3f}")
    return agg


def _load_pubchem_pool() -> pd.DataFrame:
    """Load PubChem pool from nb1263 cached parquet."""
    if not PUBCHEM_POOL_CACHE.exists():
        raise FileNotFoundError(
            f"PubChem pool cache missing: {PUBCHEM_POOL_CACHE}"
        )
    pool = pd.read_parquet(PUBCHEM_POOL_CACHE)
    keep_cols = ["inchikey", "std_smiles", "pec50", "n_meas"]
    pool = pool[keep_cols].copy()
    pool["src"] = "pubchem"
    print(f"   [pubchem] loaded {len(pool)} unique cpds from cache")
    print(f"   [pubchem] pec50: mean={pool['pec50'].mean():.3f}  "
          f"std={pool['pec50'].std():.3f}  "
          f"median={pool['pec50'].median():.3f}")
    return pool


def _load_bindingdb_pool() -> pd.DataFrame:
    """BindingDB PXR pool (uniprot O75469) -> per-InChIKey median pEC50."""
    if not BDB_NR_PARQUET.exists():
        raise FileNotFoundError(
            f"BindingDB NR cache not found: {BDB_NR_PARQUET}"
        )
    d = pd.read_parquet(BDB_NR_PARQUET)
    print(f"   [bdb] raw rows in NR cache: {len(d)}")
    pxr = d[(d["target_name"] == "PXR") & (d["uniprot"] == PXR_UNIPROT)].copy()
    print(f"   [bdb] PXR rows (uniprot {PXR_UNIPROT}): {len(pxr)}")
    pxr = pxr[pxr["pec50"].notna() & pxr["std_smiles"].notna()].copy()
    pxr["pec50"] = pxr["pec50"].astype(float)
    pxr = pxr[(pxr["pec50"] >= PEC50_LOW) & (pxr["pec50"] <= PEC50_HIGH)].copy()
    mols = pxr["std_smiles"].apply(standardize)
    pxr["inchikey_local"] = mols.apply(_safe_inchikey)
    pxr["std_smiles_local"] = mols.apply(_safe_can_smiles)
    pxr = pxr[pxr["inchikey_local"].notna() & pxr["std_smiles_local"].notna()].copy()
    agg = (
        pxr.groupby("inchikey_local", as_index=False)
        .agg(
            pec50=("pec50", "median"),
            std_smiles=("std_smiles_local", "first"),
            n_meas=("pec50", "count"),
        )
        .rename(columns={"inchikey_local": "inchikey"})
    )
    agg["src"] = "bindingdb"
    print(f"   [bdb] after InChIKey dedup (median): {len(agg)} unique cpds")
    print(f"   [bdb] pec50: mean={agg['pec50'].mean():.3f}  "
          f"std={agg['pec50'].std():.3f}  median={agg['pec50'].median():.3f}")
    return agg


# ---------------------------------------------------------------------------
# Union build with PRIORITY: train pec50 wins, else median of any other source
# ---------------------------------------------------------------------------
def _build_union_pool() -> tuple[pd.DataFrame, dict]:
    print("\n" + "-" * 78)
    print("TRAIN POOL BUILD (PXR-challenge train, 4,139 cpds)")
    print("-" * 78)
    train_pool = _load_train_pool()
    print("\n" + "-" * 78)
    print("CHEMBL POOL BUILD")
    print("-" * 78)
    chembl_pool = _load_chembl_pool()
    print("\n" + "-" * 78)
    print("PUBCHEM POOL BUILD")
    print("-" * 78)
    pubchem_pool = _load_pubchem_pool()
    print("\n" + "-" * 78)
    print("BINDINGDB POOL BUILD")
    print("-" * 78)
    bdb_pool = _load_bindingdb_pool()

    input_counts = {
        "train": len(train_pool),
        "chembl": len(chembl_pool),
        "pubchem": len(pubchem_pool),
        "bindingdb": len(bdb_pool),
    }
    n_sum = sum(input_counts.values())
    print(f"\n   pre-union sum: {n_sum}  per-source: {input_counts}")

    print("\n" + "-" * 78)
    print("UNION + INCHIKEY DEDUP (priority: train > median(other sources))")
    print("-" * 78)

    combined = pd.concat(
        [train_pool, chembl_pool, pubchem_pool, bdb_pool], ignore_index=True
    )
    n_concat = len(combined)

    # Aggregator: if any train row exists -> use train pec50; else median across
    # other sources.  Source tag is the joined set.
    def _agg_group(g: pd.DataFrame) -> pd.Series:
        srcs = set(g["src"].tolist())
        has_train = "train" in srcs
        if has_train:
            train_rows = g[g["src"] == "train"]
            pec50 = float(np.median(train_rows["pec50"]))
        else:
            pec50 = float(np.median(g["pec50"]))
        # Source tag: comma-joined, sorted, e.g. "chembl,pubchem"
        src_tag = ",".join(sorted(srcs))
        return pd.Series({
            "std_smiles": g["std_smiles"].iloc[0],
            "pec50": pec50,
            "n_meas": int(g["n_meas"].sum()),
            "src": src_tag,
            "has_train": bool(has_train),
        })

    agg = combined.groupby("inchikey", as_index=False).apply(
        _agg_group, include_groups=False
    )
    print(f"   union pool after InChIKey dedup: {len(agg)} cpds  "
          f"(removed {n_concat - len(agg)} cross-source dups)")
    src_counts = agg["src"].value_counts().to_dict()
    print(f"   top-15 src breakdown: "
          f"{dict(list(sorted(src_counts.items(), key=lambda kv: -kv[1])[:15]))}")
    has_train_count = int(agg["has_train"].sum())
    print(f"   compounds with train pec50 (assay-matched anchor): {has_train_count}")
    print(f"   pool pec50: mean={agg['pec50'].mean():.3f}  "
          f"std={agg['pec50'].std():.3f}  "
          f"median={agg['pec50'].median():.3f}  "
          f"min={agg['pec50'].min():.3f}  max={agg['pec50'].max():.3f}")
    stats = {
        "input_counts": input_counts,
        "pre_union_sum": int(n_sum),
        "post_union_dedup": int(len(agg)),
        "n_with_train": has_train_count,
    }
    return agg, stats


# ---------------------------------------------------------------------------
# Tanimoto top-k + kNN-feature builder
# ---------------------------------------------------------------------------
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


def _knn_features(top_idx, top_sim, pool_labels, pool_has_train,
                  pool_src, fallback_pec50):
    """Build 3 union-kNN features + diagnostics (top-1 src share)."""
    w = np.clip(top_sim.copy(), 0.0, 1.0)
    w_sum = w.sum(axis=1)
    n_q = top_idx.shape[0]
    pred = np.empty(n_q, dtype=np.float32)
    frac_from_train = np.empty(n_q, dtype=np.float32)
    top1_src_arr: list[str] = []
    for i in range(n_q):
        if w_sum[i] < SIM_FLOOR:
            pred[i] = fallback_pec50
            frac_from_train[i] = 0.0
            top1_src_arr.append("none")
        else:
            pred[i] = float(np.sum(w[i] * pool_labels[top_idx[i]]) / w_sum[i])
            frac_from_train[i] = 1.0 if pool_has_train[top_idx[i, 0]] else 0.0
            top1_src_arr.append(str(pool_src[top_idx[i, 0]]))
    mean_sim = top_sim.mean(axis=1).astype(np.float32)
    return dict(
        pred=pred,
        mean_sim=mean_sim,
        frac_from_train=frac_from_train,
        top1_src=top1_src_arr,
    )


def _top1_src_share(top1_src_arr: list[str]) -> dict:
    """Fraction of test rows whose top-1 NN came from each canonical source.

    Sources tagged on union rows can be multi-tagged ('chembl,pubchem'); we
    take the first canonical match in the priority order train > chembl >
    pubchem > bindingdb > none.
    """
    canon = {"train": 0, "chembl": 0, "pubchem": 0, "bindingdb": 0, "none": 0}
    for s in top1_src_arr:
        if s == "none":
            canon["none"] += 1
            continue
        parts = set(s.split(","))
        if "train" in parts:
            canon["train"] += 1
        elif "chembl" in parts:
            canon["chembl"] += 1
        elif "pubchem" in parts:
            canon["pubchem"] += 1
        elif "bindingdb" in parts:
            canon["bindingdb"] += 1
        else:
            canon["none"] += 1
    total = len(top1_src_arr)
    return {k: round(v / max(total, 1), 4) for k, v in canon.items()}


# ---------------------------------------------------------------------------
# Residual learner (5-seed bag, 5-fold cross-fit each)
# ---------------------------------------------------------------------------
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


def _residual_cross_fit_one_seed(X, residual, seed):
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Train+ChEMBL+PubChem+BindingDB UNION kNN residual feature;")
    print(f"          shallow residual-LGBM bag on nb1070 anchor")
    print(f"          seeds = {RESID_SEEDS}  folds = {RESID_FOLDS}")
    print(f"          features = MACCS-167 + 3 union-kNN feats  (170)")
    print(f"          verdict refs: nb1242 ({NB1242_REF}) / nb1251 ({NB1251_REF})  "
          f"margin {DECISION_MARGIN}")
    print("=" * 78)

    # ---- Load anchor + truth ----
    te = load_test()
    n_test = len(te)
    test_smiles = (
        te["smiles"].astype(str).tolist()
        if "smiles" in te.columns
        else te["SMILES"].astype(str).tolist()
    )
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

    # ---- Build the 4-source union pool ----
    pool, build_stats = _build_union_pool()
    n_pool_pre_leak = len(pool)

    # ---- Test InChIKey leak guard ----
    print("\n" + "-" * 78)
    print("TEST-SET LEAK GUARD (drop union cpds whose InChIKey is in the 513)")
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
    n_test_leaks = n_before - n_after
    print(f"   pool: {n_before} -> {n_after}  "
          f"(dropped {n_test_leaks} test-overlapping cpds)")

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
    pool_has_train = pool["has_train"].to_numpy(dtype=bool)
    pool_src = pool["src"].to_numpy()
    pool_median = float(np.median(pool_labels))
    print(f"   final pool size: {len(pool)}  median pEC50 = {pool_median:.3f}")
    print(f"   pool has_train fraction = {pool_has_train.mean():.4f}  "
          f"({int(pool_has_train.sum())}/{len(pool)})")

    std_test_smiles = []
    for m in test_mols:
        if m is None:
            std_test_smiles.append("")
        else:
            std_test_smiles.append(Chem.MolToSmiles(m))
    fp_test = morgan_fp_batch(std_test_smiles)
    print(f"   test FP: {fp_test.shape}  density={fp_test.mean():.4f}")

    # ---- Tanimoto kNN ----
    print("\n" + "-" * 78)
    print(f"TANIMOTO kNN (k={KNN_K}) -- test (513) vs union pool ({len(pool)})")
    print("-" * 78)
    top_idx, top_sim = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    feats = _knn_features(
        top_idx, top_sim, pool_labels, pool_has_train, pool_src,
        fallback_pec50=pool_median,
    )
    pred = feats["pred"]
    mean_sim = feats["mean_sim"]
    frac_from_train = feats["frac_from_train"]
    top1_src_arr = feats["top1_src"]
    top1_sim = top_sim[:, 0]

    print(f"   pred_pec50      mean={pred.mean():.3f}  std={pred.std():.3f}  "
          f"min={pred.min():.3f}  max={pred.max():.3f}")
    print(f"   top1 sim        p10={np.percentile(top1_sim, 10):.3f}  "
          f"p50={np.percentile(top1_sim, 50):.3f}  "
          f"p90={np.percentile(top1_sim, 90):.3f}  "
          f"max={top1_sim.max():.3f}")
    print(f"   mean5 sim       p10={np.percentile(mean_sim, 10):.3f}  "
          f"p50={np.percentile(mean_sim, 50):.3f}  "
          f"p90={np.percentile(mean_sim, 90):.3f}")
    print(f"   frac_from_train mean (513) = {frac_from_train.mean():.4f}")
    n_zero_neighbor = int((top1_sim < SIM_FLOOR).sum())
    print(f"   {n_zero_neighbor}/513 test rows had no neighbor "
          f"(fell back to pool median {pool_median:.3f})")

    src_share_513 = _top1_src_share(top1_src_arr)
    print(f"   513 top-1 source share: {src_share_513}")

    # Also report share on unblind slice
    src_share_unb = _top1_src_share([top1_src_arr[i] for i in unb_idx])
    print(f"   253 (unb) top-1 source share: {src_share_unb}")

    # ---- MACCS-167 (unblind slice) ----
    X_maccs_te = np.load(MACCS_TE_PATH)
    if X_maccs_te.shape[0] != n_test:
        raise ValueError(f"MACCS cache shape mismatch: {X_maccs_te.shape}")
    X_maccs_unb = X_maccs_te[unb_idx].astype(np.float32)

    # ---- Build residual feature matrix on 253 ----
    feats_unb = np.column_stack([
        pred[unb_idx].astype(np.float32),
        mean_sim[unb_idx].astype(np.float32),
        frac_from_train[unb_idx].astype(np.float32),
    ])
    X_unb = np.concatenate([X_maccs_unb, feats_unb], axis=1).astype(np.float32)
    feat_dim = X_unb.shape[1]
    print(f"   residual feature matrix: {X_unb.shape}  "
          f"(MACCS-167 + 3 union-kNN feats)")

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
    print(f"   nb1251 ref             = {NB1251_REF:.4f}  (nb1242+nb1211 blend)")

    beats_nb1070 = rae_mean_bag < rae_anchor - DECISION_MARGIN
    beats_nb1242 = rae_mean_bag < NB1242_REF - DECISION_MARGIN
    beats_nb1251 = rae_mean_bag < NB1251_REF - DECISION_MARGIN
    flat_nb1242 = abs(rae_mean_bag - NB1242_REF) < DECISION_MARGIN
    flat_nb1251 = abs(rae_mean_bag - NB1251_REF) < DECISION_MARGIN

    if beats_nb1251:
        verdict = "TRAIN_ALL_EXTERNAL_UNION_BEATS_NB1251_NEW_PRIMARY_CANDIDATE"
    elif beats_nb1242:
        verdict = "TRAIN_ALL_EXTERNAL_UNION_BEATS_NB1242_BUT_NOT_NB1251"
    elif flat_nb1251 or flat_nb1242:
        verdict = "TRAIN_ALL_EXTERNAL_UNION_FLAT_VS_REFERENCES"
    elif beats_nb1070:
        verdict = "TRAIN_ALL_EXTERNAL_UNION_HELPS_NB1070_BUT_HURTS_REFS"
    else:
        verdict = "TRAIN_ALL_EXTERNAL_UNION_HURTS_NB1070"
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
        "data_sources": ["pxr_train", "chembl_local_caches",
                         "pubchem_pxr_pool", "bindingdb_nr_O75469"],
        "input_counts": build_stats["input_counts"],
        "pre_union_sum_rows": build_stats["pre_union_sum"],
        "n_union_pre_leakguard": int(n_pool_pre_leak),
        "n_union_post_leakguard": int(n_after),
        "n_union_final": int(len(pool)),
        "test_inchikeys_in_pool_dropped": int(n_test_leaks),
        "n_with_train_in_final_pool": int(pool_has_train.sum()),
        "pool_pec50_mean": float(pool_labels.mean()),
        "pool_pec50_std": float(pool_labels.std()),
        "pool_pec50_median": pool_median,
        "knn_k": KNN_K,
        "top1_sim_p10": float(np.percentile(top1_sim, 10)),
        "top1_sim_p50": float(np.percentile(top1_sim, 50)),
        "top1_sim_p90": float(np.percentile(top1_sim, 90)),
        "top1_sim_max": float(top1_sim.max()),
        "mean5_sim_p10": float(np.percentile(mean_sim, 10)),
        "mean5_sim_p50": float(np.percentile(mean_sim, 50)),
        "mean5_sim_p90": float(np.percentile(mean_sim, 90)),
        "frac_from_train_mean_513": float(frac_from_train.mean()),
        "frac_from_train_mean_unb253": float(frac_from_train[unb_idx].mean()),
        "top1_src_share_513": src_share_513,
        "top1_src_share_unb253": src_share_unb,
        "n_zero_neighbor_rows": n_zero_neighbor,
        "n_unb": n_unb,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "feature_dim": feat_dim,
        "knn_feature_list": ["pred_pec50", "mean_sim", "frac_from_train"],
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
        "flat_vs_nb1242": bool(flat_nb1242),
        "flat_vs_nb1251": bool(flat_nb1251),
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
        "input_counts", "pre_union_sum_rows",
        "n_union_pre_leakguard", "n_union_post_leakguard", "n_union_final",
        "test_inchikeys_in_pool_dropped",
        "n_with_train_in_final_pool",
        "top1_sim_p10", "top1_sim_p50", "top1_sim_p90",
        "frac_from_train_mean_513", "frac_from_train_mean_unb253",
        "top1_src_share_513", "top1_src_share_unb253",
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
