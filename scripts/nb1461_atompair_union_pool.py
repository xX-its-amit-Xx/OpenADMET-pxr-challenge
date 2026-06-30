"""nb1461 -- External-pool SHAP-pruned: top-30 AtomPair (nb1373) + 3-feature
ChEMBL+PubChem union kNN, residual learner on nb1070 anchor.

Hypothesis:
    nb1373 already showed that SHAP-pruning AtomPair to top-30 bits + 2 ChEMBL
    kNN feats (pred_chembl, mean_sim) hits residual-bag RAE 0.5095 on 253
    unblind -- the best score in the SHAP-pruned series.  nb1270 swapped the
    ChEMBL-only kNN for a ChEMBL+PubChem union kNN (1689 cpds) with MACCS-167,
    hitting 0.5528.

    nb1461 fuses both: REPLACE nb1373's 2 ChEMBL-only kNN features with 3
    union-pool kNN features (weighted-mean pEC50, mean sim, frac_from_chembl)
    against the same 1689-cpd ChEMBL+PubChem union as nb1270.  Total feature
    dim = 33 = top-30 AtomPair + 3 union-kNN.  No MACCS, no Mordred.

    Why this might win over nb1373 (0.5095):
        - Wider chemical coverage: 1689 union pool vs 945 ChEMBL-only.
        - frac_from_chembl carries source-provenance signal that pred + sim
          alone don't (which neighbors are from potent literature vs Tox21 HTS).
        - Top-30 AtomPair bits provide the pair-distance channel; 3 kNN feats
          provide the global-chemistry-similarity anchor.

    Why this might lose:
        - nb1373's SHAP importances were computed on ChEMBL-only kNN feats;
          the union-pool features have different distribution (lower mean_sim
          ~0.25 vs ~0.27, different pred_chembl distribution).
        - 33 features on 253 rows is still 7.7 rows/feat -- high overfit risk.
        - Per-source z-scoring is dropped here (raw pEC50 only) to match the
          nb1373 ChEMBL kNN feature definition.

Protocol:
    1. Load ChEMBL pool (3 parquets, KEEP_TYPES, KEEP_RELATIONS, nM filter,
       median-aggregate by InChIKey) and PubChem pool (cached parquet from
       nb1263).  Source-tag each row, union, InChIKey-dedup (median over
       both raw pEC50 and within-source).
    2. Test InChIKey leak guard.
    3. Compute Morgan-2048 over union pool.  kNN k=5 Tanimoto to 513 test
       cpds.  Build 3 union-kNN features (513,):
            (a) pred_union  -- similarity-weighted mean RAW pEC50 of top-5
            (b) mean_sim    -- mean top-5 Tanimoto sim
            (c) frac_chembl -- fraction of top-5 NN sourced from ChEMBL
                              (including 'both'); pubchem-only -> 0
    4. Load AtomPair-2048 cache.  Slice to top-30 bit indices from
       nb1373_summary.json (`top_atompair_bit_indices_ranked`).
    5. Slice both feature blocks to 253 unblind idx.  Concatenate to (253, 33):
            X = [top30_AtomPair | pred_union | mean_sim | frac_chembl]
    6. Anchor = nb1070_pred_oof; residual = y_unb - anchor.
    7. 5-seed bag (seeds [0, 1, 7, 42, 137]), KFold(n=5) shallow LGBM Huber
       (same hyperparams as nb1373).  Mean-bag pooled RAE.
    8. Verdict at 0.003 margin vs:
            nb1373 (0.5095) -- best ChEMBL-only SHAP-pruned AtomPair bag
            nb1270 (0.5528) -- MACCS + union-kNN bag
            nb1070 (~0.5771) -- anchor
       Compute (rae_anchor) at runtime for true delta vs nb1070.

Outputs:
    scripts/nb1461_atompair_union_pool.py        (this file)
    data/processed/nb1461_summary.json
    data/processed/nb1461_mean_bag_oof.npy        (253,) float32
    data/processed/nb1461_median_bag_oof.npy      (253,) float32
    data/processed/nb1461_per_seed_corrected_oof.npy (5, 253) float32
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

TAG = "nb1461"
ANCHOR = "nb1070"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
NB1373_SUMMARY_PATH = DATA_PROCESSED / "nb1373_summary.json"

EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"
PUBCHEM_POOL_CACHE = EXT_DIR / "pubchem_pxr_pool.parquet"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

NB1070_REF = 0.5771
NB1373_REF = 0.5095
NB1270_REF = 0.5528
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
    """Same union as nb1242/nb1270/nb1373."""
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
        d = d[(d["pec50"] >= 3.0) & (d["pec50"] <= 11.0)].copy()
        d = d[["std_smiles", "pec50"]].rename(columns={"std_smiles": "smiles"})
        frames.append(d)
        print(f"   [chembl] chembl_nr_extended PXR kept: {len(d)} rows")

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
    if not PUBCHEM_POOL_CACHE.exists():
        raise FileNotFoundError(
            f"PubChem pool cache missing: {PUBCHEM_POOL_CACHE}. "
            f"Run nb1263 first to populate it."
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


def _build_union_pool() -> pd.DataFrame:
    """Union ChEMBL + PubChem, InChIKey-dedup with src='both' when overlap.
    Returns columns: inchikey, std_smiles, pec50, n_meas, src.
    """
    print("\n" + "-" * 78)
    print("CHEMBL POOL")
    print("-" * 78)
    chembl = _load_chembl_pool()

    print("\n" + "-" * 78)
    print("PUBCHEM POOL")
    print("-" * 78)
    pubchem = _load_pubchem_pool()

    print("\n" + "-" * 78)
    print("UNION + INCHIKEY DEDUP")
    print("-" * 78)
    print(f"   pre-union counts: chembl={len(chembl)}  pubchem={len(pubchem)}  "
          f"sum={len(chembl) + len(pubchem)}")
    combined = pd.concat([chembl, pubchem], ignore_index=True)
    n_concat = len(combined)

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
            "pec50": float(np.median(g["pec50"])),
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
    print(f"   pec50: mean={agg['pec50'].mean():.3f}  "
          f"std={agg['pec50'].std():.3f}  "
          f"median={agg['pec50'].median():.3f}")
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


def _knn_union_features(top_idx: np.ndarray, top_sim: np.ndarray,
                        pool_labels: np.ndarray, pool_src: np.ndarray,
                        fallback_pec50: float) -> dict:
    """Build 3 union-kNN features per query row:
       pred_union, mean_sim, frac_chembl.
    """
    w = np.clip(top_sim.copy(), 0.0, 1.0)
    w_sum = w.sum(axis=1)
    n_q = top_idx.shape[0]
    pred_union = np.empty(n_q, dtype=np.float32)
    frac_chembl = np.empty(n_q, dtype=np.float32)
    for i in range(n_q):
        if w_sum[i] < SIM_FLOOR:
            pred_union[i] = fallback_pec50
            frac_chembl[i] = 0.0
        else:
            pred_union[i] = float(
                np.sum(w[i] * pool_labels[top_idx[i]]) / w_sum[i]
            )
            srcs = pool_src[top_idx[i]]
            n_chembl_neighbors = int(np.sum(
                np.isin(srcs, ["chembl", "both"])
            ))
            frac_chembl[i] = float(n_chembl_neighbors) / float(top_idx.shape[1])
    mean_sim = top_sim.mean(axis=1).astype(np.float32)
    return dict(
        pred_union=pred_union,
        mean_sim=mean_sim,
        frac_chembl=frac_chembl,
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
    print(f"{TAG} -- top-30 AtomPair (nb1373 SHAP) + 3-feat ChEMBL+PubChem union "
          f"kNN; residual learner on {ANCHOR}")
    print(f"          seeds = {RESID_SEEDS}  folds = {RESID_FOLDS}")
    print(f"          baselines: nb1373 ({NB1373_REF:.4f}), "
          f"nb1270 ({NB1270_REF:.4f})  margin = {DECISION_MARGIN}")
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

    # ---- Load top-30 AtomPair bit indices from nb1373 ----
    if not NB1373_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"nb1373 summary missing: {NB1373_SUMMARY_PATH}; "
            f"run nb1373 first to compute SHAP-pruned AtomPair bits."
        )
    with open(NB1373_SUMMARY_PATH) as f:
        nb1373_summary = json.load(f)
    top_bit_idx_ranked = np.array(
        nb1373_summary["top_atompair_bit_indices_ranked"], dtype=np.int32
    )
    n_top_bits = len(top_bit_idx_ranked)
    print(f"[load] nb1373 top-{n_top_bits} AtomPair bit indices loaded")
    print(f"       (first 10: {top_bit_idx_ranked[:10].tolist()})")

    # ---- Union pool ----
    pool = _build_union_pool()
    n_pool_pre_leak = len(pool)

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
    print(f"   pool: {n_before} -> {n_after}  "
          f"(dropped {n_before - n_after} test-overlapping cpds)")

    # ---- Morgan FPs for kNN ----
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
    pool_src = pool["src"].to_numpy()
    pool_median = float(np.median(pool_labels))
    print(f"   final pool size: {len(pool)}  median pEC50 = {pool_median:.3f}")
    src_counts_final = pool["src"].value_counts().to_dict()
    print(f"   src breakdown final: {src_counts_final}")

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
    feats = _knn_union_features(
        top_idx, top_sim, pool_labels, pool_src, fallback_pec50=pool_median
    )
    pred_union = feats["pred_union"]
    mean_sim = feats["mean_sim"]
    frac_chembl = feats["frac_chembl"]
    top1_sim = top_sim[:, 0]

    print(f"   pred_union    mean={pred_union.mean():.3f}  "
          f"std={pred_union.std():.3f}  "
          f"min={pred_union.min():.3f}  max={pred_union.max():.3f}")
    print(f"   top1 sim      p10={np.percentile(top1_sim, 10):.3f}  "
          f"p50={np.percentile(top1_sim, 50):.3f}  "
          f"p90={np.percentile(top1_sim, 90):.3f}  "
          f"max={top1_sim.max():.3f}")
    print(f"   mean5 sim     p10={np.percentile(mean_sim, 10):.3f}  "
          f"p50={np.percentile(mean_sim, 50):.3f}  "
          f"p90={np.percentile(mean_sim, 90):.3f}")
    print(f"   frac_chembl   mean={frac_chembl.mean():.3f}  "
          f"std={frac_chembl.std():.3f}  "
          f"frac@1.0={float((frac_chembl == 1.0).mean()):.3f}  "
          f"frac@0.0={float((frac_chembl == 0.0).mean()):.3f}")
    n_zero_neighbor = int((top1_sim < SIM_FLOOR).sum())
    print(f"   {n_zero_neighbor}/513 test rows had no neighbor (fell back to median)")

    # ---- AtomPair-2048 (top-30 bit slice on unblind) ----
    print("\n" + "-" * 78)
    print(f"ATOMPAIR TOP-{n_top_bits} SLICE (using nb1373 SHAP-pruned bits)")
    print("-" * 78)
    if not ATOMPAIR_TE_PATH.exists():
        raise FileNotFoundError(
            f"AtomPair test cache missing: {ATOMPAIR_TE_PATH}"
        )
    X_ap_te = np.load(ATOMPAIR_TE_PATH)
    if X_ap_te.shape[0] != n_test:
        raise ValueError(f"AtomPair cache shape mismatch: {X_ap_te.shape}")
    n_ap = int(X_ap_te.shape[1])
    print(f"   AtomPair cache shape = {X_ap_te.shape}  (n_bits={n_ap})")
    X_ap_unb_top = X_ap_te[unb_idx][:, top_bit_idx_ranked].astype(np.float32)
    print(f"   top-{n_top_bits} AtomPair slice (unb): {X_ap_unb_top.shape}  "
          f"density={X_ap_unb_top.mean():.4f}  "
          f"const cols = {int((X_ap_unb_top.var(axis=0) == 0).sum())}/{n_top_bits}")

    # ---- Build PRUNED 33-col feature matrix on unblind ----
    print("\n" + "-" * 78)
    print("BUILD FEATURE MATRIX (top-30 AtomPair + 3 union-kNN feats = 33)")
    print("-" * 78)
    pred_union_unb = pred_union[unb_idx].astype(np.float32)
    mean_sim_unb = mean_sim[unb_idx].astype(np.float32)
    frac_chembl_unb = frac_chembl[unb_idx].astype(np.float32)
    X_unb = np.concatenate(
        [
            X_ap_unb_top,
            pred_union_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
            frac_chembl_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim = X_unb.shape[1]
    print(f"   X_unb shape = {X_unb.shape}  (n_feat={feat_dim})")

    # ---- Per-seed residual cross-fit ----
    print("\n" + "-" * 78)
    print(f"PER-SEED RESIDUAL CROSS-FIT (shallow LGBM Huber, dim={feat_dim})")
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
    rae_per_seed_min = float(per_seed_rae_arr.min())
    rae_per_seed_max = float(per_seed_rae_arr.max())

    print("\n" + "-" * 78)
    print("BAG AGGREGATIONS")
    print("-" * 78)
    print(f"   per-seed RAE list      = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   per-seed mean          = {rae_per_seed_mean:.4f}")
    print(f"   per-seed median        = {rae_per_seed_median:.4f}")
    print(f"   per-seed std           = {rae_per_seed_std:.4f}")
    print(f"   per-seed min/max       = "
          f"{rae_per_seed_min:.4f} / {rae_per_seed_max:.4f}")
    print(f"   pooled RAE(mean_bag)   = {rae_mean_bag:.4f}  "
          f"(d_vs_nb1070 = {rae_mean_bag - rae_anchor:+.4f}"
          f"  d_vs_nb1373 = {rae_mean_bag - NB1373_REF:+.4f}"
          f"  d_vs_nb1270 = {rae_mean_bag - NB1270_REF:+.4f})")
    print(f"   pooled RAE(median_bag) = {rae_median_bag:.4f}  "
          f"(d_vs_nb1373 = {rae_median_bag - NB1373_REF:+.4f})")
    print(f"   nb1070 ref             = {NB1070_REF:.4f}")
    print(f"   nb1373 ref             = {NB1373_REF:.4f}  "
          f"(top-30 AtomPair + 2 ChEMBL-only kNN)")
    print(f"   nb1270 ref             = {NB1270_REF:.4f}  "
          f"(MACCS-167 + 5 union-kNN)")

    beats_nb1070 = rae_mean_bag < rae_anchor - DECISION_MARGIN
    beats_nb1373 = rae_mean_bag < NB1373_REF - DECISION_MARGIN
    beats_nb1270 = rae_mean_bag < NB1270_REF - DECISION_MARGIN

    if beats_nb1373:
        verdict = "ATOMPAIR_UNION_BEATS_NB1373_NEW_PRIMARY_CANDIDATE"
    elif abs(rae_mean_bag - NB1373_REF) < DECISION_MARGIN:
        verdict = "ATOMPAIR_UNION_FLAT_VS_NB1373"
    elif beats_nb1270:
        verdict = "ATOMPAIR_UNION_BEATS_NB1270_BUT_WORSE_THAN_NB1373"
    elif beats_nb1070:
        verdict = "ATOMPAIR_UNION_HELPS_NB1070_BUT_WORSE_THAN_NB1270_NB1373"
    else:
        verdict = "ATOMPAIR_UNION_HURTS_NB1070"
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
        "data_source": "chembl_local_caches + pubchem_pxr_pool.parquet",
        "n_union_pre_leakguard": int(n_pool_pre_leak),
        "n_union_post_leakguard": int(n_after),
        "n_union_final": int(len(pool)),
        "test_inchikeys_in_pool_dropped": int(n_before - n_after),
        "src_breakdown_final": {str(k): int(v) for k, v in src_counts_final.items()},
        "pool_median_pec50": pool_median,
        "knn_k": KNN_K,
        "top1_sim_p10": float(np.percentile(top1_sim, 10)),
        "top1_sim_p50": float(np.percentile(top1_sim, 50)),
        "top1_sim_p90": float(np.percentile(top1_sim, 90)),
        "top1_sim_max": float(top1_sim.max()),
        "mean5_sim_p10": float(np.percentile(mean_sim, 10)),
        "mean5_sim_p50": float(np.percentile(mean_sim, 50)),
        "mean5_sim_p90": float(np.percentile(mean_sim, 90)),
        "pred_union_mean": float(pred_union.mean()),
        "pred_union_std": float(pred_union.std()),
        "frac_chembl_mean": float(frac_chembl.mean()),
        "frac_chembl_pure_chembl_rate": float((frac_chembl == 1.0).mean()),
        "frac_chembl_pure_pubchem_rate": float((frac_chembl == 0.0).mean()),
        "n_zero_neighbor_rows": n_zero_neighbor,
        "n_atompair_bits_cache": n_ap,
        "n_top_atompair_bits_used": int(n_top_bits),
        "top_atompair_bit_indices_ranked": [int(b) for b in top_bit_idx_ranked.tolist()],
        "atompair_top_bit_density_unb": float(X_ap_unb_top.mean()),
        "atompair_top_const_cols": int((X_ap_unb_top.var(axis=0) == 0).sum()),
        "feature_dim": int(feat_dim),
        "feature_layout": "top30_atompair + pred_union + mean_sim + frac_chembl",
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
        "rae_per_seed_min": rae_per_seed_min,
        "rae_per_seed_max": rae_per_seed_max,
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_nb1070": rae_mean_bag - rae_anchor,
        "delta_mean_bag_vs_nb1373": rae_mean_bag - NB1373_REF,
        "delta_mean_bag_vs_nb1270": rae_mean_bag - NB1270_REF,
        "beats_nb1070": bool(beats_nb1070),
        "beats_nb1373": bool(beats_nb1373),
        "beats_nb1270": bool(beats_nb1270),
        "verdict": verdict,
        "nb1070_ref_pooled": NB1070_REF,
        "nb1373_ref": NB1373_REF,
        "nb1270_ref": NB1270_REF,
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
        "top1_sim_p10", "top1_sim_p50", "top1_sim_p90",
        "mean5_sim_p50",
        "pred_union_mean", "pred_union_std",
        "frac_chembl_mean", "frac_chembl_pure_chembl_rate",
        "frac_chembl_pure_pubchem_rate",
        "n_zero_neighbor_rows",
        "n_top_atompair_bits_used", "feature_dim",
        "rae_anchor_nb1070", "per_seed_rae",
        "rae_per_seed_mean", "rae_per_seed_std",
        "rae_mean_bag", "rae_median_bag",
        "delta_mean_bag_vs_nb1070",
        "delta_mean_bag_vs_nb1373",
        "delta_mean_bag_vs_nb1270",
        "beats_nb1070", "beats_nb1373", "beats_nb1270",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
