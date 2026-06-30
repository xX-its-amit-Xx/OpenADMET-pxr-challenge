"""nb1284 -- ChEMBL kNN with multiple similarity metrics.

Hypothesis:
    nb1242 uses Morgan-Tanimoto for ChEMBL kNN distance.  Different distance
    metrics may identify DIFFERENT neighbors:
        * Morgan-Tanimoto (substructure)
        * Mordred-Euclidean on top-100-variance descriptors (descriptor-based)
        * MACCS-Jaccard on MACCS-167 keys (substructure-key based)
    Three different kNN candidates per test compound -> feature ensemble may
    help on the OOD novel-scaffold tail.

Pipeline:
    1. Reuse 945 ChEMBL PXR compounds with pEC50 (same _load_chembl_pool as
       nb1242).
    2. Compute Morgan-2048 (Tanimoto), Mordred top-100-variance features
       (z-scored, Euclidean), MACCS-167 (Jaccard).
    3. For each metric -> k=5 NN per test compound -> 6 ChEMBL features:
         (mean_5_pec50_morgan,  sim_5_morgan)
         (mean_5_pec50_mordred, sim_5_mordred)
         (mean_5_pec50_maccs,   sim_5_maccs)
    4. Residual learner: features = MACCS-167 + 6 ChEMBL = 173 cols.
       5-seed bag shallow LGBM Huber on residual = (y_unb - nb1070_pred_oof),
       5-fold cross-fit per seed.
    5. Verdict at 0.003 margin vs nb1242 mean-bag (0.5431).

Outputs:
    scripts/nb1284_chembl_multi_metric.py        (this file)
    data/processed/nb1284_summary.json
    data/processed/nb1284_mean_bag_oof.npy       (253,) float32
    data/processed/nb1284_per_seed_corrected_oof.npy  (5, 253) float32
    data/processed/nb1284_median_bag_oof.npy     (253,) float32
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
from rdkit.Chem import MACCSkeys

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1284"
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
MORDRED_TOP_N = 100

NB1070_REF = 0.5771
NB1242_REF = 0.5431
DECISION_MARGIN = 0.003


# ----------------------------------------------------------------------
# Helpers shared with nb1242 (kept verbatim for reproducibility)
# ----------------------------------------------------------------------
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
    """Same union as nb1242 -> 945 PXR cpds dedup'd by InChIKey."""
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
        raise FileNotFoundError(
            "No local ChEMBL PXR parquets found in data/external/"
        )

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


# ----------------------------------------------------------------------
# Distance/feature builders for the three similarity metrics
# ----------------------------------------------------------------------
def _morgan_fp(smiles_list) -> np.ndarray:
    return morgan_fp_batch(list(smiles_list))


def _maccs_fp(smiles_list) -> np.ndarray:
    """MACCS-167 bit vectors as (N, 167) uint8.  Bit 0 of GenMACCSKeys is
    always zero -- we drop it to match the conventional 166-bit set as well,
    but we keep all 167 columns for simplicity; the distance is invariant."""
    out = np.zeros((len(smiles_list), 167), dtype=np.uint8)
    for i, s in enumerate(smiles_list):
        if not s:
            continue
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        bv = MACCSkeys.GenMACCSKeys(m)
        # ToBitString returns "0101..." length 167
        bs = bv.ToBitString()
        out[i] = np.frombuffer(bs.encode("ascii"), dtype=np.uint8) - ord("0")
    return out


def _mordred_top_variance(pool_smiles, test_smiles, top_n: int = MORDRED_TOP_N):
    """Compute Mordred 2D descriptors over (pool + test), drop non-numeric
    failures, z-score per column on the pool half, select top-N by variance
    on the pool half, return (Z_pool (n_pool, top_n), Z_test (n_test, top_n)).

    Memory: 1613 2D descriptors * (945 + 513) rows is ~18 MB, fine."""
    from mordred import Calculator, descriptors

    calc = Calculator(descriptors, ignore_3D=True)
    all_smiles = list(pool_smiles) + list(test_smiles)
    mols = [Chem.MolFromSmiles(s) if s else None for s in all_smiles]
    # mordred handles None gracefully but we filter explicitly
    df = calc.pandas([m if m is not None else Chem.MolFromSmiles("CC") for m in mols],
                     nproc=1, quiet=True)
    # Coerce to numeric, replace mordred error objects with NaN
    df = df.apply(pd.to_numeric, errors="coerce")
    arr = df.to_numpy(dtype=np.float32)
    n_pool = len(pool_smiles)
    pool_part = arr[:n_pool]
    test_part = arr[n_pool:]

    # Column-wise: replace NaN with pool-median, then z-score on pool.
    col_med = np.nanmedian(pool_part, axis=0)
    col_med = np.where(np.isnan(col_med), 0.0, col_med)
    pool_filled = np.where(np.isnan(pool_part), col_med[None, :], pool_part)
    test_filled = np.where(np.isnan(test_part), col_med[None, :], test_part)
    # Replace +/- inf with 0 after median fill (rare)
    pool_filled[~np.isfinite(pool_filled)] = 0.0
    test_filled[~np.isfinite(test_filled)] = 0.0

    mu = pool_filled.mean(axis=0)
    sigma = pool_filled.std(axis=0)
    nonconst = sigma > 1e-8
    pool_z = np.zeros_like(pool_filled)
    test_z = np.zeros_like(test_filled)
    pool_z[:, nonconst] = (pool_filled[:, nonconst] - mu[nonconst]) / sigma[nonconst]
    test_z[:, nonconst] = (test_filled[:, nonconst] - mu[nonconst]) / sigma[nonconst]

    # Variance ranking on pool half (now equal across nonconst since z-scored to var=1).
    # Instead use raw variance pre-z-score as the ranking criterion.
    raw_var = pool_filled.var(axis=0)
    raw_var = np.where(nonconst, raw_var, -1.0)
    top_cols = np.argsort(-raw_var)[:top_n]
    print(f"   [mordred] selected top-{top_n} of {arr.shape[1]} descriptors by raw variance")
    return pool_z[:, top_cols], test_z[:, top_cols]


def _tanimoto_topk(fp_q, fp_pool, k):
    """Identical to nb1242 -- block matmul Tanimoto."""
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


def _jaccard_topk(fp_q, fp_pool, k):
    """MACCS bit vectors -- Jaccard = Tanimoto over binary keys (same math)."""
    return _tanimoto_topk(fp_q, fp_pool, k)


def _euclidean_topk(zq, zp, k):
    """Top-k smallest Euclidean distance on z-scored vectors.  Returns
    (top_idx (n_q, k), top_sim (n_q, k)) where 'sim' is mapped from
    distance via sim = 1 / (1 + d) so it lives in (0, 1] and the
    downstream weighting code reused from nb1242 still applies."""
    n_q = zq.shape[0]
    n_pool = zp.shape[0]
    top_idx = np.zeros((n_q, k), dtype=np.int32)
    top_sim = np.zeros((n_q, k), dtype=np.float32)
    BLOCK = 64
    # ||q-p||^2 = ||q||^2 + ||p||^2 - 2 q.p
    p_sq = (zp * zp).sum(axis=1)
    for s in range(0, n_q, BLOCK):
        e = min(n_q, s + BLOCK)
        q_sq = (zq[s:e] * zq[s:e]).sum(axis=1)
        cross = zq[s:e] @ zp.T
        d2 = q_sq[:, None] + p_sq[None, :] - 2.0 * cross
        d2 = np.maximum(d2, 0.0)
        d = np.sqrt(d2)
        if k >= n_pool:
            idx_part = np.argsort(d, axis=1)[:, :k]
        else:
            part = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
            row_idx = np.arange(e - s)[:, None]
            d_part = d[row_idx, part]
            order = np.argsort(d_part, axis=1)
            idx_part = part[row_idx, order]
        row_idx = np.arange(e - s)[:, None]
        d_top = d[row_idx, idx_part]
        sim = 1.0 / (1.0 + d_top)
        top_idx[s:e] = idx_part
        top_sim[s:e] = sim
    return top_idx, top_sim


def _knn_predict(top_idx, top_sim, pool_labels, fallback):
    """Same as nb1242."""
    w = np.clip(top_sim, 0.0, None)
    w_sum = w.sum(axis=1)
    n_q = top_idx.shape[0]
    pred = np.empty(n_q, dtype=np.float32)
    for i in range(n_q):
        if w_sum[i] < SIM_FLOOR:
            pred[i] = fallback
        else:
            pred[i] = float(np.sum(w[i] * pool_labels[top_idx[i]]) / w_sum[i])
    mean_sim = top_sim.mean(axis=1).astype(np.float32)
    return pred, mean_sim


# ----------------------------------------------------------------------
# Cross-fit residual LGBM (capacity identical to nb1242)
# ----------------------------------------------------------------------
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


def _residual_cross_fit_one_seed(X, residual, seed, return_importance=False):
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    imps = []
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
        if return_importance:
            imps.append(mdl.feature_importances_)
    if return_importance:
        imps = np.stack(imps).mean(axis=0)
        return oof, imps
    return oof


def _overlap_rate(top_a, top_b, k):
    """Fraction of (test_row, neighbor_slot) pairs where the index matches."""
    return float((top_a[:, :k] == top_b[:, :k]).mean())


def _set_overlap(top_a, top_b, k):
    """Fraction of test rows whose top-k sets intersect (>=1 shared neighbor)."""
    n = top_a.shape[0]
    sa = [set(top_a[i, :k].tolist()) for i in range(n)]
    sb = [set(top_b[i, :k].tolist()) for i in range(n)]
    inter_size = np.array([len(sa[i] & sb[i]) for i in range(n)])
    return {
        "frac_any_share": float((inter_size > 0).mean()),
        "mean_jaccard": float(np.mean([
            len(sa[i] & sb[i]) / max(1, len(sa[i] | sb[i])) for i in range(n)
        ])),
        "mean_intersect_count": float(inter_size.mean()),
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- ChEMBL kNN multi-metric residual feature")
    print(f"          metrics = Morgan-Tanimoto, Mordred-top{MORDRED_TOP_N}-Euclidean, MACCS-Jaccard")
    print(f"          seeds = {RESID_SEEDS}  folds = {RESID_FOLDS}")
    print(f"          features = MACCS-167 + 6 ChEMBL = 173")
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

    # ---- ChEMBL pool (same as nb1242) ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL (reuse nb1242 union)")
    print("-" * 78)
    pool = _load_chembl_pool()
    test_mols = [standardize(s) for s in test_smiles]
    test_inchikeys = set()
    for m in test_mols:
        ik = _safe_inchikey(m)
        if ik is not None:
            test_inchikeys.add(ik)
    n_before = len(pool)
    pool = pool[~pool["inchikey"].isin(test_inchikeys)].reset_index(drop=True)
    n_after = len(pool)
    print(f"   pool: {n_before} -> {n_after}  (leak guard dropped {n_before - n_after})")

    pool_smiles = pool["std_smiles"].tolist()
    std_test_smiles = []
    for m in test_mols:
        std_test_smiles.append(Chem.MolToSmiles(m) if m is not None else "")
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))

    # ---- Three featurizations ----
    print("\n" + "-" * 78)
    print("FEATURIZE x 3")
    print("-" * 78)
    print("[1/3] Morgan-2048 ...")
    fp_morgan_pool = _morgan_fp(pool_smiles)
    fp_morgan_test = _morgan_fp(std_test_smiles)
    keep = fp_morgan_pool.sum(axis=1) > 0
    if not keep.all():
        n_drop = int((~keep).sum())
        print(f"   [morgan] dropped {n_drop} pool rows with zero Morgan FP")
        pool = pool[keep].reset_index(drop=True)
        pool_smiles = pool["std_smiles"].tolist()
        pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
        fp_morgan_pool = fp_morgan_pool[keep]
    print(f"   morgan pool {fp_morgan_pool.shape}  test {fp_morgan_test.shape}")

    print("[2/3] MACCS-167 ...")
    fp_maccs_pool = _maccs_fp(pool_smiles)
    fp_maccs_test = _maccs_fp(std_test_smiles)
    print(f"   maccs pool {fp_maccs_pool.shape}  test {fp_maccs_test.shape}")

    print(f"[3/3] Mordred 2D -> top-{MORDRED_TOP_N} by variance ...")
    zp_mordred, zt_mordred = _mordred_top_variance(
        pool_smiles, std_test_smiles, top_n=MORDRED_TOP_N
    )
    print(f"   mordred z pool {zp_mordred.shape}  test {zt_mordred.shape}")

    # ---- kNN per metric ----
    print("\n" + "-" * 78)
    print(f"k={KNN_K} NN PER METRIC")
    print("-" * 78)
    top_idx_morgan, top_sim_morgan = _tanimoto_topk(fp_morgan_test, fp_morgan_pool, KNN_K)
    pred_morgan, sim_morgan = _knn_predict(top_idx_morgan, top_sim_morgan,
                                           pool_labels, pool_median)
    print(f"   [morgan]  top1 sim p50={np.percentile(top_sim_morgan[:, 0], 50):.3f} "
          f"pred mean={pred_morgan.mean():.3f} std={pred_morgan.std():.3f}")

    top_idx_maccs, top_sim_maccs = _jaccard_topk(fp_maccs_test, fp_maccs_pool, KNN_K)
    pred_maccs, sim_maccs = _knn_predict(top_idx_maccs, top_sim_maccs,
                                         pool_labels, pool_median)
    print(f"   [maccs ]  top1 sim p50={np.percentile(top_sim_maccs[:, 0], 50):.3f} "
          f"pred mean={pred_maccs.mean():.3f} std={pred_maccs.std():.3f}")

    top_idx_mord, top_sim_mord = _euclidean_topk(zt_mordred, zp_mordred, KNN_K)
    pred_mord, sim_mord = _knn_predict(top_idx_mord, top_sim_mord,
                                       pool_labels, pool_median)
    print(f"   [mordred] top1 sim p50={np.percentile(top_sim_mord[:, 0], 50):.3f} "
          f"pred mean={pred_mord.mean():.3f} std={pred_mord.std():.3f}")

    # ---- NN-overlap rate between metrics ----
    print("\n" + "-" * 78)
    print("NN-OVERLAP RATE BETWEEN METRICS")
    print("-" * 78)
    ovl_morgan_maccs = _set_overlap(top_idx_morgan, top_idx_maccs, KNN_K)
    ovl_morgan_mord = _set_overlap(top_idx_morgan, top_idx_mord, KNN_K)
    ovl_maccs_mord = _set_overlap(top_idx_maccs, top_idx_mord, KNN_K)
    print(f"   morgan vs maccs :  any-share={ovl_morgan_maccs['frac_any_share']:.3f}  "
          f"jacc={ovl_morgan_maccs['mean_jaccard']:.3f}  "
          f"mean_inter={ovl_morgan_maccs['mean_intersect_count']:.2f}/{KNN_K}")
    print(f"   morgan vs mord  :  any-share={ovl_morgan_mord['frac_any_share']:.3f}  "
          f"jacc={ovl_morgan_mord['mean_jaccard']:.3f}  "
          f"mean_inter={ovl_morgan_mord['mean_intersect_count']:.2f}/{KNN_K}")
    print(f"   maccs  vs mord  :  any-share={ovl_maccs_mord['frac_any_share']:.3f}  "
          f"jacc={ovl_maccs_mord['mean_jaccard']:.3f}  "
          f"mean_inter={ovl_maccs_mord['mean_intersect_count']:.2f}/{KNN_K}")

    # ---- Pairwise correlations between ChEMBL pred features ----
    print("\n   Pairwise pred correlations (513 rows):")
    corr_morgan_maccs = float(np.corrcoef(pred_morgan, pred_maccs)[0, 1])
    corr_morgan_mord = float(np.corrcoef(pred_morgan, pred_mord)[0, 1])
    corr_maccs_mord = float(np.corrcoef(pred_maccs, pred_mord)[0, 1])
    print(f"     pred(morgan) ~ pred(maccs)  : r = {corr_morgan_maccs:+.3f}")
    print(f"     pred(morgan) ~ pred(mord)   : r = {corr_morgan_mord:+.3f}")
    print(f"     pred(maccs)  ~ pred(mord)   : r = {corr_maccs_mord:+.3f}")

    # ---- Residual feature matrix ----
    X_maccs_te = np.load(MACCS_TE_PATH)
    if X_maccs_te.shape[0] != n_test:
        raise ValueError(f"MACCS cache shape mismatch: {X_maccs_te.shape}")
    X_maccs_unb = X_maccs_te[unb_idx].astype(np.float32)

    feats_513 = np.stack([
        pred_morgan, sim_morgan,
        pred_mord, sim_mord,
        pred_maccs, sim_maccs,
    ], axis=1).astype(np.float32)
    feats_unb = feats_513[unb_idx]
    X_unb = np.concatenate([X_maccs_unb, feats_unb], axis=1).astype(np.float32)
    feat_dim = X_unb.shape[1]
    print(f"\n   residual feature matrix: {X_unb.shape} "
          f"(MACCS-167 + 6 ChEMBL)")
    feat_names = (
        [f"maccs_{i}" for i in range(X_maccs_unb.shape[1])]
        + ["pred_morgan", "sim_morgan",
           "pred_mord", "sim_mord",
           "pred_maccs", "sim_maccs"]
    )

    # ---- Per-seed residual cross-fit (collect importances from seed 0) ----
    print("\n" + "-" * 78)
    print(f"PER-SEED RESIDUAL CROSS-FIT (depth=3 LGBM Huber, dim={feat_dim})")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []
    importance_acc = np.zeros(feat_dim, dtype=np.float64)
    for i, s in enumerate(RESID_SEEDS):
        resid_oof_s, imp_s = _residual_cross_fit_one_seed(
            X_unb, residual, s, return_importance=True
        )
        importance_acc += imp_s
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

    importance_acc /= len(RESID_SEEDS)
    top5_idx = np.argsort(-importance_acc)[:5]
    top5_features = [
        {"rank": int(r + 1), "name": feat_names[int(top5_idx[r])],
         "importance": float(importance_acc[int(top5_idx[r])])}
        for r in range(len(top5_idx))
    ]

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
    print(f"   per-seed RAE = [{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   per-seed mean   = {rae_per_seed_mean:.4f}")
    print(f"   per-seed median = {rae_per_seed_median:.4f}")
    print(f"   per-seed std    = {rae_per_seed_std:.4f}")
    print(f"   mean_bag  RAE   = {rae_mean_bag:.4f}  "
          f"(d_vs_nb1070 = {rae_mean_bag - rae_anchor:+.4f}, "
          f"d_vs_nb1242 = {rae_mean_bag - NB1242_REF:+.4f})")
    print(f"   median_bag RAE  = {rae_median_bag:.4f}")
    print(f"   nb1242 ref      = {NB1242_REF:.4f}")

    beats_nb1070 = rae_mean_bag < rae_anchor - DECISION_MARGIN
    beats_nb1242 = rae_mean_bag < NB1242_REF - DECISION_MARGIN

    if beats_nb1242:
        verdict = "MULTI_METRIC_BEATS_NB1242"
    elif rae_mean_bag < NB1242_REF + DECISION_MARGIN:
        verdict = "MULTI_METRIC_TIES_NB1242"
    elif beats_nb1070:
        verdict = "MULTI_METRIC_HELPS_NB1070_BUT_WORSE_THAN_NB1242"
    else:
        verdict = "MULTI_METRIC_WORSE_THAN_NB1070"
    print(f"   verdict         = {verdict}")

    print("\n   Top-5 feature importance (bag-mean across seeds):")
    for rec in top5_features:
        print(f"     #{rec['rank']}  {rec['name']:<18s}  imp = {rec['importance']:.2f}")

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
        "n_chembl_pool": int(len(pool)),
        "knn_k": KNN_K,
        "mordred_top_n": MORDRED_TOP_N,
        "metrics": ["morgan_tanimoto", "mordred_top100_euclidean", "maccs_jaccard"],
        "overlap_morgan_vs_maccs": ovl_morgan_maccs,
        "overlap_morgan_vs_mord": ovl_morgan_mord,
        "overlap_maccs_vs_mord": ovl_maccs_mord,
        "pred_corr_morgan_maccs": corr_morgan_maccs,
        "pred_corr_morgan_mord": corr_morgan_mord,
        "pred_corr_maccs_mord": corr_maccs_mord,
        "top1_sim_p50_morgan": float(np.percentile(top_sim_morgan[:, 0], 50)),
        "top1_sim_p50_maccs": float(np.percentile(top_sim_maccs[:, 0], 50)),
        "top1_sim_p50_mord_pseudo": float(np.percentile(top_sim_mord[:, 0], 50)),
        "n_unb": int(n_unb),
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "feature_dim": int(feat_dim),
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
        "top5_feature_importance": top5_features,
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
        "n_chembl_pool",
        "overlap_morgan_vs_maccs", "overlap_morgan_vs_mord", "overlap_maccs_vs_mord",
        "pred_corr_morgan_maccs", "pred_corr_morgan_mord", "pred_corr_maccs_mord",
        "rae_anchor_nb1070", "per_seed_rae",
        "rae_per_seed_mean", "rae_per_seed_std",
        "rae_mean_bag", "rae_median_bag",
        "delta_mean_bag_vs_nb1070", "delta_mean_bag_vs_nb1242",
        "beats_nb1070", "beats_nb1242",
        "verdict",
        "top5_feature_importance",
    ):
        print(f"  {k}: {res.get(k)}")
