"""nb1590 -- Outer-bag VALIDATION of nb1571 blend (0.55*nb1561 + 0.45*nb1560).

HYPOTHESIS:
    nb1571 reported a 5-fold cross-fit SLSQP blend RAE of 0.5130 on the
    253 PRE-unblind set, blending the BoB-mean OOFs of nb1561 (CatBoost,
    0.5155) and nb1560 (LGBM-Huber, 0.5211) with grid-best mix
    w_nb1561 = 0.55, w_nb1560 = 0.45.

    This script outer-bag validates the blend by, for each of 5 outer seeds,
    rebuilding BOTH base learners on the SAME pinned 117-col 5-way K-tuned
    feature matrix (top-25 AtomPair + top-20 MACCS + top-20 Mordred +
    top-20 ChempropEmbed + top-30 Avalon + pred_chembl + mean_sim) and
    forming blend_o = 0.55 * nb1561_o + 0.45 * nb1560_o.  Both base
    learners use 5 inner seeds = [o*1000 + offset for offset in
    [0,1,7,42,137]] -- identical inner-seed schedule across the two models
    so the per-outer comparison is faithful.

PROTOCOL
    1. For each outer seed o in {0, 1, 7, 42, 137}:
         a. Build the 117-col 5-way K-tuned feature stack (pinned).
         b. inner_seeds = [o*1000 + s for s in [0,1,7,42,137]].
         c. For each inner seed s' (5 total):
              i. KFold(n=5, shuffle=True, random_state=s')
              ii. CatBoost(MAE, d4, n200, lr0.05, l2=5, random_seed=s')
                   residual cross-fit -> resid_cat_oof_{s'}.
              iii. LGBM(huber alpha=1.0, d3, leaves=7, n200, lr=0.05,
                   random_state=s') residual cross-fit -> resid_lgb_oof_{s'}.
              iv. pred_cat_{s'} = anchor + resid_cat_oof_{s'}
                  pred_lgb_{s'} = anchor + resid_lgb_oof_{s'}
         d. nb1561_o = mean of 5 pred_cat_{s'}.
         e. nb1560_o = mean of 5 pred_lgb_{s'}.
         f. blend_o = 0.55 * nb1561_o + 0.45 * nb1560_o.
         g. Per-outer pooled RAE = rae(y_unb, blend_o).
    2. Row-level BoB MEAN + MEDIAN across the 5 blend_o vectors.
    3. Verdict NB1571_REPRODUCES iff
           |per_outer_mean - nb1571_ref| < 0.003
       where nb1571_ref = 0.5130 (the cross-fit SLSQP RAE from
       nb1571_summary.json).

Outputs:
    scripts/nb1590_bag_nb1571.py                  (this file)
    data/processed/nb1590_summary.json
    data/processed/nb1590_bob_mean_oof.npy        (253,) float32
    data/processed/nb1590_bob_median_oof.npy      (253,) float32
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
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1590"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

OUTER_SEEDS = [0, 1, 7, 42, 137]
INNER_OFFSETS = [0, 1, 7, 42, 137]
RESID_FOLDS = 5

# nb1571 grid-best blend (CatBoost dominant, LGBM minority)
W_NB1561 = 0.55
W_NB1560 = 0.45

# Reference: nb1571 cross-fit SLSQP RAE on 253 PRE-unblind.
NB1571_REF = 0.5130
REPRODUCE_MARGIN = 0.003

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")

EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
NB1392_SUMMARY = DATA_PROCESSED / "nb1392_summary.json"
NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"
NB1523_SUMMARY = DATA_PROCESSED / "nb1523_summary.json"
NB1524_SUMMARY = DATA_PROCESSED / "nb1524_summary.json"
NB1541_SUMMARY = DATA_PROCESSED / "nb1541_summary.json"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6


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
    """Same union as nb1561/nb1554."""
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
        raise FileNotFoundError("No local ChEMBL PXR parquets found")

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


def _knn_predict(top_idx: np.ndarray, top_sim: np.ndarray,
                 pool_labels: np.ndarray, fallback: float):
    w = top_sim.copy()
    w = np.clip(w, 0.0, 1.0)
    w_sum = w.sum(axis=1)
    n_q = top_idx.shape[0]
    pred = np.empty(n_q, dtype=np.float32)
    for i in range(n_q):
        if w_sum[i] < SIM_FLOOR:
            pred[i] = fallback
        else:
            pred[i] = np.sum(w[i] * pool_labels[top_idx[i]]) / w_sum[i]
    mean_sim = top_sim.mean(axis=1).astype(np.float32)
    return pred, mean_sim


# ---- CatBoost (nb1561 / nb1554-style) -----------------------------------
def _cat_params(seed: int) -> dict:
    return dict(
        loss_function="MAE",
        depth=4,
        iterations=200,
        learning_rate=0.05,
        l2_leaf_reg=5.0,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=2,
    )


def _cat_residual_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray,
                                     seed: int) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = CatBoostRegressor(**_cat_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


# ---- LGBM Huber (nb1560 / nb1553-style) ---------------------------------
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


def _lgb_residual_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray,
                                     seed: int) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _load_mordred_test(n_test_expected: int) -> np.ndarray:
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(
            f"Mordred cache missing -- run nb1030 first ({mte_p})"
        )
    X_te_m = np.load(mte_p).astype(np.float32)
    if X_te_m.shape[0] != n_test_expected:
        raise ValueError(
            f"Mordred test shape mismatch: {X_te_m.shape} vs "
            f"n_test={n_test_expected}"
        )
    X_te_m = np.where(np.isfinite(X_te_m), X_te_m, np.nan).astype(np.float32)
    col_med = np.nanmedian(X_te_m, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X_te_m)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X_te_m[idx_r, idx_c] = col_med[idx_c]
    return X_te_m


def _load_npy_test(path: Path, n_test_expected: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"missing cache: {path}")
    X = np.load(path)
    if X.shape[0] != n_test_expected:
        raise ValueError(f"shape mismatch {path}: {X.shape}")
    X = X.astype(np.float32)
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X


def _extract_atompair_top_idx_from_nb1484(sum_1484: dict) -> np.ndarray:
    for f in sum_1484["families"]:
        if f["family"] == "AtomPair":
            return np.array(f["top_idx_ranked"], dtype=int)
    raise KeyError("AtomPair entry not found in nb1484_summary.json")


def _extract_best_K_record(sum_dict: dict, records_key: str,
                           best_K_key: str = "best_K") -> dict:
    best_K = int(sum_dict[best_K_key])
    for r in sum_dict[records_key]:
        if int(r["K"]) == best_K:
            return r
    raise KeyError(f"best_K {best_K} not found in {records_key}")


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- OUTER-BAG VALIDATION of nb1571 blend "
          f"({W_NB1561:.2f}*nb1561 + {W_NB1560:.2f}*nb1560)")
    print(f"         outer seeds      = {OUTER_SEEDS}")
    print(f"         inner offsets    = {INNER_OFFSETS}  "
          f"(inner_seed = o*1000 + offset)")
    print(f"         folds (KFold rs) = {RESID_FOLDS}  random_state = inner_seed")
    print(f"         nb1571_ref       = {NB1571_REF:.4f}  "
          f"margin = {REPRODUCE_MARGIN}")
    print("=" * 78)

    # ---- Load truth + anchor (PRE-unblind chemprop_aux) ----
    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"\n[load] n_test={n_test}  n_unb={n_unb}")

    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"chemprop_aux te file missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(f"anchor shape mismatch: {te_anchor_513.shape}")
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Load all K-grid winners + SHAP rankings (same as nb1561 / nb1554) ----
    for p in (NB1352_SUMMARY, NB1392_SUMMARY, NB1484_SUMMARY,
              NB1523_SUMMARY, NB1524_SUMMARY, NB1541_SUMMARY):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")
    with open(NB1352_SUMMARY) as f:
        sum_1352 = json.load(f)
    with open(NB1392_SUMMARY) as f:
        sum_1392 = json.load(f)
    with open(NB1484_SUMMARY) as f:
        sum_1484 = json.load(f)
    with open(NB1523_SUMMARY) as f:
        sum_1523 = json.load(f)
    with open(NB1524_SUMMARY) as f:
        sum_1524 = json.load(f)
    with open(NB1541_SUMMARY) as f:
        sum_1541 = json.load(f)

    top_maccs_bit_idx = np.array(
        sum_1352["top_maccs_bit_indices_ranked"], dtype=int
    )
    rec_mord = _extract_best_K_record(sum_1523, "per_K_records",
                                       best_K_key="best_K")
    K_Mord_best = int(rec_mord["K"])
    top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)
    assert K_Mord_best == int(sum_1523["best_K"])

    full_ap_ranked = _extract_atompair_top_idx_from_nb1484(sum_1484)
    K_AP_best = int(sum_1524["best_K"])
    top_ap_bit_idx = full_ap_ranked[:K_AP_best]

    K_Embed_best = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed_col_idx = top_embed_full[:K_Embed_best]

    top_avalon_bit_idx = np.array(
        sum_1392["top_avalon_bit_indices_ranked"], dtype=int
    )
    K_Avalon_used = int(len(top_avalon_bit_idx))

    n_top_maccs = int(len(top_maccs_bit_idx))
    n_top_mord = int(len(top_mord_col_idx))
    n_top_ap = int(len(top_ap_bit_idx))
    n_top_embed = int(len(top_embed_col_idx))
    n_top_avalon = int(K_Avalon_used)
    print(f"\n[reuse] top-{n_top_ap}     AtomPair bits  (nb1524 K={K_AP_best})")
    print(f"[reuse] top-{n_top_maccs}     MACCS bits     (nb1352)")
    print(f"[reuse] top-{n_top_mord}     Mordred cols   (nb1523 K={K_Mord_best})")
    print(f"[reuse] top-{n_top_embed}     ChempropEmbed  (nb1541 K={K_Embed_best})")
    print(f"[reuse] top-{n_top_avalon}     Avalon bits    (nb1392 SHAP K=30)")

    # ---- Per-family pruned slices on unb ----
    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test)
    X_ap_unb = X_ap_te[unb_idx].astype(np.float32)
    X_ap_unb_top = X_ap_unb[:, top_ap_bit_idx].astype(np.float32)
    print(f"\n[feat] X_ap_unb_top      = {X_ap_unb_top.shape}")

    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)
    X_maccs_unb = X_maccs_te[unb_idx].astype(np.float32)
    X_maccs_unb_top = X_maccs_unb[:, top_maccs_bit_idx].astype(np.float32)
    print(f"[feat] X_maccs_unb_top   = {X_maccs_unb_top.shape}")

    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    X_mord_unb = X_mord_te[unb_idx].astype(np.float32)
    X_mord_unb_top = X_mord_unb[:, top_mord_col_idx].astype(np.float32)
    print(f"[feat] X_mord_unb_top    = {X_mord_unb_top.shape}")

    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)
    X_emb_unb = X_emb_te[unb_idx].astype(np.float32)
    X_emb_unb_top = X_emb_unb[:, top_embed_col_idx].astype(np.float32)
    print(f"[feat] X_emb_unb_top     = {X_emb_unb_top.shape}")

    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)
    X_av_unb = X_av_te[unb_idx].astype(np.float32)
    X_av_unb_top = X_av_unb[:, top_avalon_bit_idx].astype(np.float32)
    print(f"[feat] X_av_unb_top      = {X_av_unb_top.shape}")

    # ---- ChEMBL pool + kNN feature ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL + kNN feature build")
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
    print(f"   pool: {n_before} -> {n_after}  (dropped {n_before - n_after})")

    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep_pool = fp_pool.sum(axis=1) > 0
    if not keep_pool.all():
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

    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )
    pred_chembl_unb = pred_chembl_pec50[unb_idx].astype(np.float32)
    mean_sim_unb = mean_sim[unb_idx].astype(np.float32)
    print(f"   pred_chembl_pec50 (unb) mean={pred_chembl_unb.mean():.3f}  "
          f"std={pred_chembl_unb.std():.3f}")
    print(f"   mean_sim (unb)         mean={mean_sim_unb.mean():.3f}")

    # ---- Build COMBINED 5-way K-tuned 117-col feature matrix (pinned) ----
    X_unb = np.concatenate(
        [
            X_ap_unb_top,
            X_maccs_unb_top,
            X_mord_unb_top,
            X_emb_unb_top,
            X_av_unb_top,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim = X_unb.shape[1]
    expected_dim = (n_top_ap + n_top_maccs + n_top_mord
                    + n_top_embed + n_top_avalon + 2)
    if feat_dim != expected_dim:
        raise ValueError(f"feat_dim {feat_dim} != expected {expected_dim}")
    print(f"\n   COMBINED 5-WAY K-TUNED matrix: {X_unb.shape}  "
          f"(top-{n_top_ap} AP + top-{n_top_maccs} MACCS + "
          f"top-{n_top_mord} Mord + top-{n_top_embed} Embed + "
          f"top-{n_top_avalon} Aval + 2)")

    # ---- Outer-bag rebuild of nb1561_o + nb1560_o + blend_o --------------
    print("\n" + "=" * 78)
    print("OUTER-BAG x [5 inner-seed CatBoost + 5 inner-seed LGBM-Huber bags]")
    print("         blend_o = 0.55 * nb1561_o + 0.45 * nb1560_o")
    print("=" * 78)
    n_outer = len(OUTER_SEEDS)

    outer_blend = np.zeros((n_outer, n_unb), dtype=np.float64)
    outer_nb1561 = np.zeros((n_outer, n_unb), dtype=np.float64)
    outer_nb1560 = np.zeros((n_outer, n_unb), dtype=np.float64)
    per_outer_records = []

    for oi, o in enumerate(OUTER_SEEDS):
        t_outer = time.time()
        inner_seeds = [int(o) * 1000 + int(s) for s in INNER_OFFSETS]
        print(f"\n   --- outer seed {o}  inner_seeds = {inner_seeds} ---")

        inner_cat = np.zeros((len(inner_seeds), n_unb), dtype=np.float64)
        inner_lgb = np.zeros((len(inner_seeds), n_unb), dtype=np.float64)
        per_inner_cat_rae: list[float] = []
        per_inner_lgb_rae: list[float] = []
        per_inner_blend_rae: list[float] = []
        per_inner_records = []
        for si, s_inner in enumerate(inner_seeds):
            ts = time.time()
            resid_cat = _cat_residual_cross_fit_one_seed(
                X_unb, residual, seed=s_inner
            )
            pred_cat_s = anchor + resid_cat
            inner_cat[si] = pred_cat_s
            r_cat_s = float(rae(y_unb, pred_cat_s))
            per_inner_cat_rae.append(r_cat_s)
            t_cat = time.time() - ts

            ts = time.time()
            resid_lgb = _lgb_residual_cross_fit_one_seed(
                X_unb, residual, seed=s_inner
            )
            pred_lgb_s = anchor + resid_lgb
            inner_lgb[si] = pred_lgb_s
            r_lgb_s = float(rae(y_unb, pred_lgb_s))
            per_inner_lgb_rae.append(r_lgb_s)
            t_lgb = time.time() - ts

            blend_s = W_NB1561 * pred_cat_s + W_NB1560 * pred_lgb_s
            r_blend_s = float(rae(y_unb, blend_s))
            per_inner_blend_rae.append(r_blend_s)

            per_inner_records.append({
                "inner_seed": int(s_inner),
                "rae_cat": r_cat_s,
                "rae_lgb": r_lgb_s,
                "rae_blend": r_blend_s,
                "resid_cat_std": float(resid_cat.std()),
                "resid_lgb_std": float(resid_lgb.std()),
                "wall_cat_sec": round(t_cat, 2),
                "wall_lgb_sec": round(t_lgb, 2),
            })
            print(f"     [outer {o:3d}] inner {s_inner:6d}: "
                  f"cat={r_cat_s:.4f}  lgb={r_lgb_s:.4f}  blend={r_blend_s:.4f}  "
                  f"wall = {t_cat + t_lgb:.1f}s")

        # Per-outer mean-bag = mean of inner pred vectors (matches nb1561/nb1560)
        nb1561_o = inner_cat.mean(axis=0)
        nb1560_o = inner_lgb.mean(axis=0)
        blend_o = W_NB1561 * nb1561_o + W_NB1560 * nb1560_o

        outer_nb1561[oi] = nb1561_o
        outer_nb1560[oi] = nb1560_o
        outer_blend[oi] = blend_o

        rae_nb1561_o = float(rae(y_unb, nb1561_o))
        rae_nb1560_o = float(rae(y_unb, nb1560_o))
        rae_blend_o = float(rae(y_unb, blend_o))

        per_outer_records.append({
            "outer_seed": int(o),
            "inner_seeds": inner_seeds,
            "per_inner_cat_rae": per_inner_cat_rae,
            "per_inner_lgb_rae": per_inner_lgb_rae,
            "per_inner_blend_rae": per_inner_blend_rae,
            "per_inner_records": per_inner_records,
            "rae_nb1561_o_mean_bag": rae_nb1561_o,
            "rae_nb1560_o_mean_bag": rae_nb1560_o,
            "rae_blend_o": rae_blend_o,
            "wall_sec": round(time.time() - t_outer, 2),
        })
        print(f"     [nb1561_o] RAE = {rae_nb1561_o:.4f}")
        print(f"     [nb1560_o] RAE = {rae_nb1560_o:.4f}")
        print(f"     [blend_o ] RAE = {rae_blend_o:.4f}  "
              f"(0.55*nb1561 + 0.45*nb1560)")
        print(f"   outer {o:3d}  wall = {time.time() - t_outer:.1f}s")

    # ---- Per-outer summary ----
    per_outer_blend_rae = [rec["rae_blend_o"] for rec in per_outer_records]
    per_outer_nb1561_rae = [rec["rae_nb1561_o_mean_bag"] for rec in per_outer_records]
    per_outer_nb1560_rae = [rec["rae_nb1560_o_mean_bag"] for rec in per_outer_records]

    blend_arr = np.array(per_outer_blend_rae)
    per_outer_mean = float(blend_arr.mean())
    per_outer_std = float(blend_arr.std())
    per_outer_min = float(blend_arr.min())
    per_outer_max = float(blend_arr.max())
    per_outer_median = float(np.median(blend_arr))

    # ---- BoB row-level aggregations across 5 blend_o vectors ----
    bob_mean_oof = outer_blend.mean(axis=0)
    bob_median_oof = np.median(outer_blend, axis=0)
    rae_bob_mean = float(rae(y_unb, bob_mean_oof))
    rae_bob_median = float(rae(y_unb, bob_median_oof))

    print("\n" + "=" * 78)
    print("OUTER-BAG SUMMARY")
    print("=" * 78)
    print(f"   per-outer blend_o RAE list = "
          f"[{', '.join(f'{r:.4f}' for r in per_outer_blend_rae)}]")
    print(f"   per-outer blend_o mean    = {per_outer_mean:.4f}")
    print(f"   per-outer blend_o std     = {per_outer_std:.4f}")
    print(f"   per-outer blend_o min     = {per_outer_min:.4f}")
    print(f"   per-outer blend_o max     = {per_outer_max:.4f}")
    print(f"   per-outer blend_o median  = {per_outer_median:.4f}")
    print(f"   per-outer nb1561_o RAE = "
          f"[{', '.join(f'{r:.4f}' for r in per_outer_nb1561_rae)}]  "
          f"mean = {np.mean(per_outer_nb1561_rae):.4f}")
    print(f"   per-outer nb1560_o RAE = "
          f"[{', '.join(f'{r:.4f}' for r in per_outer_nb1560_rae)}]  "
          f"mean = {np.mean(per_outer_nb1560_rae):.4f}")
    print(f"   BoB MEAN   pooled RAE = {rae_bob_mean:.4f}  "
          f"(d vs nb1571_ref = {rae_bob_mean - NB1571_REF:+.4f})")
    print(f"   BoB MEDIAN pooled RAE = {rae_bob_median:.4f}  "
          f"(d vs nb1571_ref = {rae_bob_median - NB1571_REF:+.4f})")

    # ---- Pearson sanity vs nb1571 best and anchor ----
    def _pearson_vs(path: Path, vec: np.ndarray):
        if not path.exists():
            return None
        oof = np.load(path).astype(np.float64)
        if oof.shape[0] != n_unb:
            return None
        return float(np.corrcoef(vec, oof)[0, 1])

    pearson_bobmean_vs_nb1571 = _pearson_vs(
        DATA_PROCESSED / "nb1571_best_oof.npy", bob_mean_oof
    )
    pearson_bobmean_vs_nb1561 = _pearson_vs(
        DATA_PROCESSED / "nb1561_bob_mean_oof.npy", bob_mean_oof
    )
    pearson_bobmean_vs_nb1560 = _pearson_vs(
        DATA_PROCESSED / "nb1560_bob_mean_oof.npy", bob_mean_oof
    )
    pearson_bobmean_vs_anchor = float(np.corrcoef(bob_mean_oof, anchor)[0, 1])
    if pearson_bobmean_vs_nb1571 is not None:
        print(f"   Pearson(bob_mean, nb1571_best)     = "
              f"{pearson_bobmean_vs_nb1571:.4f}")
    if pearson_bobmean_vs_nb1561 is not None:
        print(f"   Pearson(bob_mean, nb1561_bob_mean) = "
              f"{pearson_bobmean_vs_nb1561:.4f}")
    if pearson_bobmean_vs_nb1560 is not None:
        print(f"   Pearson(bob_mean, nb1560_bob_mean) = "
              f"{pearson_bobmean_vs_nb1560:.4f}")
    print(f"   Pearson(bob_mean, anchor)          = "
          f"{pearson_bobmean_vs_anchor:.4f}")

    # ---- Verdict ----
    delta_per_outer = per_outer_mean - NB1571_REF
    reproduces = abs(delta_per_outer) < REPRODUCE_MARGIN
    if reproduces:
        verdict = "NB1571_REPRODUCES"
    elif per_outer_mean < NB1571_REF - REPRODUCE_MARGIN:
        verdict = "NB1571_OPTIMISTIC_OUTER_BAG_BETTER"
    else:
        verdict = "NB1571_LUCKY_SEED_OUTER_BAG_WORSE"

    print("\n" + "-" * 78)
    print(f"VERDICT (per-outer blend_o mean within {REPRODUCE_MARGIN} of "
          f"nb1571_ref = {NB1571_REF:.4f}):")
    print(f"   per-outer-mean = {per_outer_mean:.4f}   "
          f"(d vs ref = {delta_per_outer:+.4f})")
    print(f"   verdict        = {verdict}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_bob_mean_oof.npy",
            bob_mean_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_bob_median_oof.npy",
            bob_median_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_bob_mean_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_bob_median_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "data_source": ("AtomPair-cache + MACCS-cache + "
                        "Mordred-cached_nb1030 + ChempropEmbed-cache + "
                        "Avalon-cache + local_chembl_caches_union"),
        "model_family": "CatBoost + LGBM-Huber blend",
        "blend_weights": {"nb1561_catboost": W_NB1561, "nb1560_lgbm": W_NB1560},
        "catboost_loss": "MAE",
        "catboost_depth": 4,
        "catboost_iterations": 200,
        "catboost_learning_rate": 0.05,
        "catboost_l2_leaf_reg": 5.0,
        "lgbm_objective": "huber",
        "lgbm_alpha": 1.0,
        "lgbm_depth": 3,
        "lgbm_num_leaves": 7,
        "lgbm_n_estimators": 80,
        "lgbm_learning_rate": 0.05,
        "K_AP_best": K_AP_best,
        "K_Mord_best": K_Mord_best,
        "K_Embed_best": K_Embed_best,
        "K_Avalon_used": K_Avalon_used,
        "K_MACCS_fixed": n_top_maccs,
        "feat_dim": int(feat_dim),
        "feat_breakdown": {
            "atompair": n_top_ap,
            "maccs": n_top_maccs,
            "mordred": n_top_mord,
            "chemprop_embed": n_top_embed,
            "avalon": n_top_avalon,
            "pred_chembl_pec50": 1,
            "mean_sim": 1,
            "total": int(feat_dim),
        },
        "outer_seeds": OUTER_SEEDS,
        "inner_offsets": INNER_OFFSETS,
        "inner_seed_formula": "o * 1000 + offset",
        "resid_folds": RESID_FOLDS,
        "n_unb": n_unb,
        "n_test": int(n_test),
        "n_chembl_pool": int(len(pool)),
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "nb1571_ref": NB1571_REF,
        "reproduce_margin": REPRODUCE_MARGIN,
        "per_outer_records": per_outer_records,
        "per_outer_blend_rae": per_outer_blend_rae,
        "per_outer_nb1561_rae": per_outer_nb1561_rae,
        "per_outer_nb1560_rae": per_outer_nb1560_rae,
        "per_outer_mean": per_outer_mean,
        "per_outer_std": per_outer_std,
        "per_outer_min": per_outer_min,
        "per_outer_max": per_outer_max,
        "per_outer_median": per_outer_median,
        "per_outer_nb1561_mean": float(np.mean(per_outer_nb1561_rae)),
        "per_outer_nb1560_mean": float(np.mean(per_outer_nb1560_rae)),
        "rae_bob_mean": rae_bob_mean,
        "rae_bob_median": rae_bob_median,
        "delta_per_outer_mean_vs_nb1571_ref": delta_per_outer,
        "delta_bob_mean_vs_nb1571_ref": rae_bob_mean - NB1571_REF,
        "delta_bob_median_vs_nb1571_ref": rae_bob_median - NB1571_REF,
        "reproduces": bool(reproduces),
        "pearson_bobmean_vs_nb1571_best": pearson_bobmean_vs_nb1571,
        "pearson_bobmean_vs_nb1561_bob_mean": pearson_bobmean_vs_nb1561,
        "pearson_bobmean_vs_nb1560_bob_mean": pearson_bobmean_vs_nb1560,
        "pearson_bobmean_vs_anchor": pearson_bobmean_vs_anchor,
        "verdict": verdict,
        "pre_unblind_clean": True,
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
        "n_unb", "n_test", "n_chembl_pool",
        "feat_dim", "feat_breakdown",
        "blend_weights",
        "K_AP_best", "K_Mord_best", "K_Embed_best",
        "K_Avalon_used", "K_MACCS_fixed",
        "outer_seeds", "inner_offsets",
        "rae_anchor_chemprop_aux",
        "nb1571_ref",
        "per_outer_blend_rae",
        "per_outer_nb1561_rae",
        "per_outer_nb1560_rae",
        "per_outer_mean", "per_outer_std",
        "per_outer_min", "per_outer_max", "per_outer_median",
        "per_outer_nb1561_mean", "per_outer_nb1560_mean",
        "rae_bob_mean", "rae_bob_median",
        "delta_per_outer_mean_vs_nb1571_ref",
        "delta_bob_mean_vs_nb1571_ref",
        "delta_bob_median_vs_nb1571_ref",
        "reproduces",
        "pearson_bobmean_vs_nb1571_best",
        "pearson_bobmean_vs_nb1561_bob_mean",
        "pearson_bobmean_vs_nb1560_bob_mean",
        "pearson_bobmean_vs_anchor",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
