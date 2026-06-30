"""nb1521 -- Outer-bag the nb1511 grid blend (w=0.50 of nb1501 + nb1484).

HYPOTHESIS:
    nb1511 reported grid_best at w=0.50 with pooled RAE 0.5191 on the 253
    PRE-unblind slice, beating nb1501 (0.5223, CatBoost mean-bag) and
    nb1484 (0.5231, LGBM-Huber 4-way). The blend is a 50/50 mean of two
    independently-trained residual-learner stacks anchored on chemprop_aux
    te[unb_idx].

    Outer-bag validation: rebuild BOTH nb1501_o (CatBoost mean-bag over
    5 inner seeds [o*1000+s for s in {0,1,7,42,137}]) and nb1484_o
    (LGBM Huber, 4 separate per-family residual learners, naive 1/4
    mean; outer seed drives KFold split + LGBM random_state) for each
    of 5 outer seeds {0,1,7,42,137}.  Per outer:
        blend_o = 0.50 * nb1501_o + 0.50 * nb1484_o
    Aggregate row-level MEAN and MEDIAN across the 5 blend_o vectors.

    Verdict NB1511_REPRODUCES iff
        |per_outer_mean - 0.5191| < 0.003

PROTOCOL
    1. For each outer seed o in {0,1,7,42,137}:
         a. Rebuild nb1501_o: 5-inner-seed CatBoost(MAE, d4, n200, lr0.05,
            l2=5) bag on the 112-col 4-way pruned matrix, inner seeds
            [o*1000+s for s in INNER_BASE]. Per inner seed, KFold(n=5,
            shuffle, random_state=inner) cross-fit residual.
            nb1501_o = mean over 5 inner seeds of (anchor + resid_oof).
         b. Rebuild nb1484_o: 4 separate per-family LGBM Huber residual
            learners (AtomPair-30, MACCS-20, Mordred-30, ChempropEmbed-30,
            each with pred_chembl + sim appended -> 32/22/32/32 cols).
            Per family, single LGBM Huber (depth=3, num_leaves=7, n_est=80,
            lr=0.05, huber alpha=1.0) with KFold(n=5, shuffle, random_state=o)
            cross-fit residual at random_state=o; pred_corr_family_o = anchor
            + resid_oof_family_o. nb1484_o = naive 1/4 mean over the 4
            family-corrected OOF vectors.
         c. blend_o = 0.50 * nb1501_o + 0.50 * nb1484_o.
         d. Per-outer pooled RAE = rae(y_unb, blend_o).
    2. Row-level BoB MEAN + MEDIAN across the 5 blend_o vectors.
    3. Verdict at 0.003 margin vs nb1511 grid_best (0.5191).

Outputs:
    scripts/nb1521_bag_nb1511.py             (this file)
    data/processed/nb1521_summary.json
    data/processed/nb1521_bob_mean_oof.npy   (253,) float32
    data/processed/nb1521_bob_median_oof.npy (253,) float32
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

TAG = "nb1521"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

NB1511_GRID_BEST_REF = 0.5191       # nb1511 grid w=0.50 pooled RAE
NB1501_REF_MEAN_BAG = 0.5223        # nb1501 CatBoost mean-bag
NB1484_REF_BEST = 0.5231            # nb1484 4-way best (naive 1/4)
REPRODUCE_MARGIN = 0.003
BLEND_W = 0.50                      # nb1511 grid_best weight on nb1501

OUTER_SEEDS = [0, 1, 7, 42, 137]
INNER_BASE_SEEDS = [0, 1, 7, 42, 137]    # nb1501-style inner bag base
RESID_FOLDS = 5

NB1501_SUMMARY = DATA_PROCESSED / "nb1501_summary.json"
NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"
NB1511_SUMMARY = DATA_PROCESSED / "nb1511_summary.json"
NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
NB1364_SUMMARY = DATA_PROCESSED / "nb1364_summary.json"
NB1373_SUMMARY = DATA_PROCESSED / "nb1373_summary.json"

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")

EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

FAMILIES = ["AtomPair", "MACCS", "Mordred", "ChempropEmbed"]


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
    """Same union as nb1501 / nb1484 / nb1512."""
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


# ---- CatBoost (nb1501-style) ----------------------------------------------
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


# ---- LGBM Huber (nb1484-style) --------------------------------------------
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


def _lgbm_residual_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray,
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
            f"Mordred test shape mismatch: {X_te_m.shape} vs n_test={n_test_expected}"
        )
    X_te_m = np.where(np.isfinite(X_te_m), X_te_m, np.nan).astype(np.float32)
    col_med = np.nanmedian(X_te_m, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X_te_m)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X_te_m[idx_r, idx_c] = col_med[idx_c]
    return X_te_m


def _extract_embed_top_idx_from_nb1484(sum_1484: dict) -> np.ndarray:
    for f in sum_1484["families"]:
        if f["family"] == "ChempropEmbed":
            return np.array(f["top_idx_ranked"], dtype=int)
    raise KeyError("ChempropEmbed entry not found in nb1484_summary.json")


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- OUTER-BAG VALIDATION of nb1511 "
          f"(blend w={BLEND_W} of nb1501 + nb1484)")
    print(f"         outer seeds        = {OUTER_SEEDS}")
    print(f"         inner base seeds   = {INNER_BASE_SEEDS}  "
          f"(nb1501 inner bag)")
    print(f"         nb1501 inner_seeds(o) = [o*1000 + s for s in base]")
    print(f"         nb1484 family seed(o) = o    "
          f"(KFold + LGBM random_state)")
    print(f"         target ref         = {NB1511_GRID_BEST_REF}  "
          f"(margin = {REPRODUCE_MARGIN})")
    print("=" * 78)

    # ---- Optional: load nb1511 / nb1501 / nb1484 refs from summaries ----
    nb1511_grid_best = NB1511_GRID_BEST_REF
    nb1501_rae_mean_bag = NB1501_REF_MEAN_BAG
    nb1484_rae_best = NB1484_REF_BEST
    if NB1511_SUMMARY.exists():
        with open(NB1511_SUMMARY) as f:
            sum_1511 = json.load(f)
        gbr = sum_1511.get("grid_best_rae")
        if gbr is not None:
            nb1511_grid_best = float(gbr)
        gbw = sum_1511.get("grid_best_w_nb1501")
        print(f"[load] nb1511 grid_best_rae       = {nb1511_grid_best:.6f}  "
              f"(w_nb1501 = {gbw})")
    if NB1501_SUMMARY.exists():
        with open(NB1501_SUMMARY) as f:
            sum_1501 = json.load(f)
        if sum_1501.get("rae_mean_bag") is not None:
            nb1501_rae_mean_bag = float(sum_1501["rae_mean_bag"])
        print(f"[load] nb1501 rae_mean_bag        = {nb1501_rae_mean_bag:.6f}")
    if NB1484_SUMMARY.exists():
        with open(NB1484_SUMMARY) as f:
            sum_1484_top = json.load(f)
        if sum_1484_top.get("rae_best") is not None:
            nb1484_rae_best = float(sum_1484_top["rae_best"])
        print(f"[load] nb1484 rae_best            = {nb1484_rae_best:.6f}  "
              f"(variant = {sum_1484_top.get('best_variant')})")

    # ---- Pull pinned SHAP top-idx per family (same as nb1501 / nb1484) ----
    for p in (NB1352_SUMMARY, NB1364_SUMMARY, NB1373_SUMMARY, NB1484_SUMMARY):
        if not p.exists():
            raise FileNotFoundError(f"missing {p} -- run prerequisite first")
    with open(NB1352_SUMMARY) as f:
        sum_1352 = json.load(f)
    with open(NB1364_SUMMARY) as f:
        sum_1364 = json.load(f)
    with open(NB1373_SUMMARY) as f:
        sum_1373 = json.load(f)
    with open(NB1484_SUMMARY) as f:
        sum_1484 = json.load(f)
    top_maccs_bit_idx = np.array(
        sum_1352["top_maccs_bit_indices_ranked"], dtype=int
    )
    top_mord_col_idx = np.array(
        sum_1364["top_mordred_col_indices_ranked"], dtype=int
    )
    top_ap_bit_idx = np.array(
        sum_1373["top_atompair_bit_indices_ranked"], dtype=int
    )
    top_embed_col_idx = _extract_embed_top_idx_from_nb1484(sum_1484)
    n_top_maccs = int(len(top_maccs_bit_idx))
    n_top_mord = int(len(top_mord_col_idx))
    n_top_ap = int(len(top_ap_bit_idx))
    n_top_embed = int(len(top_embed_col_idx))
    print(f"\n[pin]  top-{n_top_maccs} MACCS  bits  (from nb1352)")
    print(f"[pin]  top-{n_top_mord} Mordred cols (from nb1364)")
    print(f"[pin]  top-{n_top_ap} AtomPair bits (from nb1373)")
    print(f"[pin]  top-{n_top_embed} Embed   cols (from nb1484)")

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

    # ---- Per-family pruned slices on 253 ----
    X_ap_te = np.load(ATOMPAIR_TE_PATH)
    X_ap_unb = X_ap_te[unb_idx].astype(np.float32)
    X_ap_unb_top = X_ap_unb[:, top_ap_bit_idx].astype(np.float32)
    print(f"\n[feat] X_ap_unb_top    shape = {X_ap_unb_top.shape}")

    X_maccs_te = np.load(MACCS_TE_PATH)
    X_maccs_unb = X_maccs_te[unb_idx].astype(np.float32)
    X_maccs_unb_top = X_maccs_unb[:, top_maccs_bit_idx].astype(np.float32)
    print(f"[feat] X_maccs_unb_top shape = {X_maccs_unb_top.shape}")

    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    X_mord_unb = X_mord_te[unb_idx].astype(np.float32)
    X_mord_unb_top = X_mord_unb[:, top_mord_col_idx].astype(np.float32)
    print(f"[feat] X_mord_unb_top  shape = {X_mord_unb_top.shape}")

    if not CHEMPROP_EMBED_TE_PATH.exists():
        raise FileNotFoundError(
            f"Chemprop embed cache missing: {CHEMPROP_EMBED_TE_PATH}"
        )
    X_emb_te = np.load(CHEMPROP_EMBED_TE_PATH).astype(np.float32)
    X_emb_te = np.where(np.isfinite(X_emb_te), X_emb_te, 0.0).astype(np.float32)
    X_emb_unb = X_emb_te[unb_idx].astype(np.float32)
    X_emb_unb_top = X_emb_unb[:, top_embed_col_idx].astype(np.float32)
    print(f"[feat] X_emb_unb_top   shape = {X_emb_unb_top.shape}")

    # ---- ChEMBL pool + kNN feature build ----
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

    # ---- Build nb1501 QUAD-PRUNED 112-col matrix ----
    X_unb_quad = np.concatenate(
        [
            X_ap_unb_top,
            X_maccs_unb_top,
            X_mord_unb_top,
            X_emb_unb_top,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim_quad = X_unb_quad.shape[1]
    expected_quad = n_top_ap + n_top_maccs + n_top_mord + n_top_embed + 2
    if feat_dim_quad != expected_quad:
        raise ValueError(f"feat_dim_quad {feat_dim_quad} != {expected_quad}")
    print(f"\n   nb1501 QUAD-PRUNED matrix: {X_unb_quad.shape}")

    # ---- Build nb1484 4 per-family pruned matrices ----
    family_matrices = {
        "AtomPair": np.concatenate(
            [X_ap_unb_top,
             pred_chembl_unb.reshape(-1, 1),
             mean_sim_unb.reshape(-1, 1)],
            axis=1,
        ).astype(np.float32),
        "MACCS": np.concatenate(
            [X_maccs_unb_top,
             pred_chembl_unb.reshape(-1, 1),
             mean_sim_unb.reshape(-1, 1)],
            axis=1,
        ).astype(np.float32),
        "Mordred": np.concatenate(
            [X_mord_unb_top,
             pred_chembl_unb.reshape(-1, 1),
             mean_sim_unb.reshape(-1, 1)],
            axis=1,
        ).astype(np.float32),
        "ChempropEmbed": np.concatenate(
            [X_emb_unb_top,
             pred_chembl_unb.reshape(-1, 1),
             mean_sim_unb.reshape(-1, 1)],
            axis=1,
        ).astype(np.float32),
    }
    for fam in FAMILIES:
        print(f"   nb1484 {fam:<14s} PRUNED matrix: "
              f"{family_matrices[fam].shape}")

    # ---- Outer x (nb1501 inner-bag + nb1484 family-bag) ---------------------
    print("\n" + "=" * 78)
    print("OUTER x [nb1501 (inner-bag CatBoost) + nb1484 (4-family LGBM)]")
    print("=" * 78)
    n_outer = len(OUTER_SEEDS)
    n_inner = len(INNER_BASE_SEEDS)

    outer_blend = np.zeros((n_outer, n_unb), dtype=np.float64)
    outer_nb1501 = np.zeros((n_outer, n_unb), dtype=np.float64)
    outer_nb1484 = np.zeros((n_outer, n_unb), dtype=np.float64)
    per_outer_records = []
    per_outer_inner_seeds_lst: list[list[int]] = []

    for oi, o in enumerate(OUTER_SEEDS):
        t_outer = time.time()
        inner_seeds = [int(o * 1000 + s) for s in INNER_BASE_SEEDS]
        per_outer_inner_seeds_lst.append(inner_seeds)
        print(f"\n   --- outer seed {o}  "
              f"inner CatBoost seeds = {inner_seeds}  "
              f"nb1484 family seed = {o} ---")

        # ----- (a) nb1501_o : 5-inner CatBoost bag on QUAD-PRUNED -----
        inner_corrected = np.zeros((n_inner, n_unb), dtype=np.float64)
        inner_per_seed_rae: list[float] = []
        for ii, isd in enumerate(inner_seeds):
            ts = time.time()
            resid_oof_s = _cat_residual_cross_fit_one_seed(
                X_unb_quad, residual, isd
            )
            pred_corr_s = anchor + resid_oof_s
            inner_corrected[ii] = pred_corr_s
            r_s = float(rae(y_unb, pred_corr_s))
            inner_per_seed_rae.append(r_s)
            print(f"     [nb1501] outer {o:3d}  inner {isd:6d}: "
                  f"rae = {r_s:.4f}  |resid|.std = {resid_oof_s.std():.3f}  "
                  f"wall = {time.time() - ts:.1f}s")
        nb1501_o = inner_corrected.mean(axis=0)
        outer_nb1501[oi] = nb1501_o
        rae_nb1501_o = float(rae(y_unb, nb1501_o))
        print(f"     [nb1501_o = 5-inner mean_bag]  pooled RAE = "
              f"{rae_nb1501_o:.4f}")

        # ----- (b) nb1484_o : 4 family LGBM Huber, naive 1/4 mean -----
        family_corrected = np.zeros((len(FAMILIES), n_unb), dtype=np.float64)
        family_rae: dict[str, float] = {}
        for fi, fam in enumerate(FAMILIES):
            ts = time.time()
            X_fam_pruned = family_matrices[fam]
            resid_oof_fam = _lgbm_residual_cross_fit_one_seed(
                X_fam_pruned, residual, seed=int(o)
            )
            pred_corr_fam = anchor + resid_oof_fam
            family_corrected[fi] = pred_corr_fam
            r_f = float(rae(y_unb, pred_corr_fam))
            family_rae[fam] = r_f
            print(f"     [nb1484] outer {o:3d}  family {fam:<14s} "
                  f"seed = {o:3d}: rae = {r_f:.4f}  "
                  f"|resid|.std = {resid_oof_fam.std():.3f}  "
                  f"wall = {time.time() - ts:.1f}s")
        nb1484_o = family_corrected.mean(axis=0)
        outer_nb1484[oi] = nb1484_o
        rae_nb1484_o = float(rae(y_unb, nb1484_o))
        print(f"     [nb1484_o = naive 1/4 mean over families] pooled RAE = "
              f"{rae_nb1484_o:.4f}")

        # ----- (c) blend_o -----
        blend_o = BLEND_W * nb1501_o + (1.0 - BLEND_W) * nb1484_o
        outer_blend[oi] = blend_o
        rae_blend_o = float(rae(y_unb, blend_o))
        print(f"     [blend_o = {BLEND_W:.2f} * nb1501_o + "
              f"{1.0 - BLEND_W:.2f} * nb1484_o] pooled RAE = "
              f"{rae_blend_o:.4f}")

        per_outer_records.append({
            "outer_seed": int(o),
            "inner_seeds_nb1501": inner_seeds,
            "nb1484_family_seed": int(o),
            "nb1501_inner_per_seed_rae": inner_per_seed_rae,
            "rae_nb1501_o_inner_per_seed_mean": float(np.mean(inner_per_seed_rae)),
            "rae_nb1501_o": rae_nb1501_o,
            "rae_nb1484_per_family": family_rae,
            "rae_nb1484_o": rae_nb1484_o,
            "blend_w": float(BLEND_W),
            "rae_blend_o": rae_blend_o,
            "wall_sec": round(time.time() - t_outer, 2),
        })
        print(f"   outer {o:3d}  (outer wall = "
              f"{time.time() - t_outer:.1f}s)")

    # ---- Per-outer summary ----
    per_outer_rae_blend: list[float] = [
        rec["rae_blend_o"] for rec in per_outer_records
    ]
    per_outer_rae_nb1501: list[float] = [
        rec["rae_nb1501_o"] for rec in per_outer_records
    ]
    per_outer_rae_nb1484: list[float] = [
        rec["rae_nb1484_o"] for rec in per_outer_records
    ]
    per_outer_arr = np.array(per_outer_rae_blend)
    per_outer_mean = float(per_outer_arr.mean())
    per_outer_std = float(per_outer_arr.std())
    per_outer_min = float(per_outer_arr.min())
    per_outer_max = float(per_outer_arr.max())
    per_outer_median = float(np.median(per_outer_arr))

    # ---- BoB row-level aggregations across 5 blend_o vectors ----
    bob_mean_oof = outer_blend.mean(axis=0)
    bob_median_oof = np.median(outer_blend, axis=0)
    rae_bob_mean = float(rae(y_unb, bob_mean_oof))
    rae_bob_median = float(rae(y_unb, bob_median_oof))

    # auxiliary BoB on nb1501_o and nb1484_o
    bob_nb1501_mean = outer_nb1501.mean(axis=0)
    bob_nb1484_mean = outer_nb1484.mean(axis=0)
    rae_bob_nb1501_mean = float(rae(y_unb, bob_nb1501_mean))
    rae_bob_nb1484_mean = float(rae(y_unb, bob_nb1484_mean))

    print("\n" + "=" * 78)
    print("OUTER-BAG SUMMARY")
    print("=" * 78)
    print(f"   per-outer nb1501_o   RAE list = "
          f"[{', '.join(f'{r:.4f}' for r in per_outer_rae_nb1501)}]")
    print(f"   per-outer nb1484_o   RAE list = "
          f"[{', '.join(f'{r:.4f}' for r in per_outer_rae_nb1484)}]")
    print(f"   per-outer blend_o    RAE list = "
          f"[{', '.join(f'{r:.4f}' for r in per_outer_rae_blend)}]")
    print(f"   per-outer blend_o    mean    = {per_outer_mean:.4f}")
    print(f"   per-outer blend_o    std     = {per_outer_std:.4f}")
    print(f"   per-outer blend_o    min     = {per_outer_min:.4f}")
    print(f"   per-outer blend_o    max     = {per_outer_max:.4f}")
    print(f"   per-outer blend_o    median  = {per_outer_median:.4f}")
    print(f"   BoB MEAN   blend     pooled RAE = {rae_bob_mean:.4f}  "
          f"(d vs nb1511_grid = {rae_bob_mean - nb1511_grid_best:+.4f})")
    print(f"   BoB MEDIAN blend     pooled RAE = {rae_bob_median:.4f}  "
          f"(d vs nb1511_grid = {rae_bob_median - nb1511_grid_best:+.4f})")
    print(f"   BoB MEAN nb1501_o    pooled RAE = {rae_bob_nb1501_mean:.4f}")
    print(f"   BoB MEAN nb1484_o    pooled RAE = {rae_bob_nb1484_mean:.4f}")

    # ---- Pearson sanity vs nb1501 / nb1484 / nb1511 -----------------------
    def _pearson_vs(path: Path, vec: np.ndarray):
        if not path.exists():
            return None
        oof = np.load(path).astype(np.float64)
        if oof.shape[0] != n_unb:
            return None
        return float(np.corrcoef(vec, oof)[0, 1])

    pearson_bobmean_vs_nb1501 = _pearson_vs(
        DATA_PROCESSED / "nb1501_mean_bag_oof.npy", bob_mean_oof
    )
    pearson_bobmean_vs_nb1484 = _pearson_vs(
        DATA_PROCESSED / "nb1484_best_oof.npy", bob_mean_oof
    )
    pearson_bobmean_vs_nb1511 = _pearson_vs(
        DATA_PROCESSED / "nb1511_best_oof.npy", bob_mean_oof
    )
    pearson_bobmedian_vs_nb1511 = _pearson_vs(
        DATA_PROCESSED / "nb1511_best_oof.npy", bob_median_oof
    )
    pearson_bobmean_vs_anchor = float(np.corrcoef(bob_mean_oof, anchor)[0, 1])
    if pearson_bobmean_vs_nb1501 is not None:
        print(f"   Pearson(bob_mean, nb1501_mean_bag) = "
              f"{pearson_bobmean_vs_nb1501:.4f}")
    if pearson_bobmean_vs_nb1484 is not None:
        print(f"   Pearson(bob_mean, nb1484_best)     = "
              f"{pearson_bobmean_vs_nb1484:.4f}")
    if pearson_bobmean_vs_nb1511 is not None:
        print(f"   Pearson(bob_mean, nb1511_best)     = "
              f"{pearson_bobmean_vs_nb1511:.4f}")
    if pearson_bobmedian_vs_nb1511 is not None:
        print(f"   Pearson(bob_median, nb1511_best)   = "
              f"{pearson_bobmedian_vs_nb1511:.4f}")
    print(f"   Pearson(bob_mean, anchor)          = "
          f"{pearson_bobmean_vs_anchor:.4f}")

    # ---- Verdict (per-outer-mean vs 0.5191 within 0.003) ----
    delta_per_outer = per_outer_mean - nb1511_grid_best
    reproduces = abs(delta_per_outer) < REPRODUCE_MARGIN
    if reproduces:
        verdict = "NB1511_REPRODUCES"
    elif per_outer_mean < nb1511_grid_best - REPRODUCE_MARGIN:
        verdict = "NB1511_OPTIMISTIC_OUTER_BAG_BETTER"
    else:
        verdict = "NB1511_LUCKY_SEED_OUTER_BAG_WORSE"

    print("\n" + "-" * 78)
    print(f"VERDICT (per-outer blend_o mean within {REPRODUCE_MARGIN} of "
          f"nb1511 grid_best = {nb1511_grid_best:.4f}):")
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
        "data_source": "AtomPair-cache + MACCS-cache + Mordred-cached_nb1030 + "
                       "ChempropEmbed-cache + local_chembl_caches_union",
        "blend_w_nb1501": float(BLEND_W),
        "blend_w_nb1484": float(1.0 - BLEND_W),
        "nb1501_model": "CatBoost(MAE, d4, n200, lr0.05, l2=5)",
        "nb1501_inner_bag_size": int(n_inner),
        "nb1484_model": "LGBM(huber alpha=1.0, d3, leaves=7, n_est=80, lr=0.05)",
        "nb1484_family_aggregation": "naive_1_4_mean",
        "n_unb": n_unb,
        "n_test": int(n_test),
        "n_chembl_pool": int(len(pool)),
        "outer_seeds": OUTER_SEEDS,
        "inner_base_seeds": INNER_BASE_SEEDS,
        "per_outer_inner_seeds_nb1501": per_outer_inner_seeds_lst,
        "resid_folds": RESID_FOLDS,
        "n_top_atompair": n_top_ap,
        "n_top_maccs": n_top_maccs,
        "n_top_mordred": n_top_mord,
        "n_top_chemprop_embed": n_top_embed,
        "feat_dim_nb1501_quad": int(feat_dim_quad),
        "feat_dim_nb1484_per_family": {
            fam: int(family_matrices[fam].shape[1]) for fam in FAMILIES
        },
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "nb1511_grid_best_ref": nb1511_grid_best,
        "nb1501_rae_mean_bag_ref": nb1501_rae_mean_bag,
        "nb1484_rae_best_ref": nb1484_rae_best,
        "reproduce_margin": REPRODUCE_MARGIN,
        "per_outer_records": per_outer_records,
        "per_outer_rae_blend": per_outer_rae_blend,
        "per_outer_rae_nb1501": per_outer_rae_nb1501,
        "per_outer_rae_nb1484": per_outer_rae_nb1484,
        "per_outer_mean": per_outer_mean,
        "per_outer_std": per_outer_std,
        "per_outer_min": per_outer_min,
        "per_outer_max": per_outer_max,
        "per_outer_median": per_outer_median,
        "rae_bob_mean": rae_bob_mean,
        "rae_bob_median": rae_bob_median,
        "rae_bob_nb1501_mean": rae_bob_nb1501_mean,
        "rae_bob_nb1484_mean": rae_bob_nb1484_mean,
        "delta_per_outer_mean_vs_nb1511": delta_per_outer,
        "delta_bob_mean_vs_nb1511": rae_bob_mean - nb1511_grid_best,
        "delta_bob_median_vs_nb1511": rae_bob_median - nb1511_grid_best,
        "reproduces": bool(reproduces),
        "pearson_bobmean_vs_nb1501_mean_bag": pearson_bobmean_vs_nb1501,
        "pearson_bobmean_vs_nb1484_best": pearson_bobmean_vs_nb1484,
        "pearson_bobmean_vs_nb1511_best": pearson_bobmean_vs_nb1511,
        "pearson_bobmedian_vs_nb1511_best": pearson_bobmedian_vs_nb1511,
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
        "feat_dim_nb1501_quad", "feat_dim_nb1484_per_family",
        "outer_seeds",
        "rae_anchor_chemprop_aux",
        "nb1511_grid_best_ref",
        "nb1501_rae_mean_bag_ref",
        "nb1484_rae_best_ref",
        "per_outer_rae_blend",
        "per_outer_rae_nb1501",
        "per_outer_rae_nb1484",
        "per_outer_mean", "per_outer_std",
        "per_outer_min", "per_outer_max", "per_outer_median",
        "rae_bob_mean", "rae_bob_median",
        "rae_bob_nb1501_mean", "rae_bob_nb1484_mean",
        "delta_per_outer_mean_vs_nb1511",
        "delta_bob_mean_vs_nb1511",
        "delta_bob_median_vs_nb1511",
        "reproduces",
        "pearson_bobmean_vs_nb1501_mean_bag",
        "pearson_bobmean_vs_nb1484_best",
        "pearson_bobmean_vs_nb1511_best",
        "pearson_bobmean_vs_anchor",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
