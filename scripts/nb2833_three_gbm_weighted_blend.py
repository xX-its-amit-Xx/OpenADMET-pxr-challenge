"""nb2833 -- Three-GBM weighted blend: LGBM + CatBoost + XGBoost on K=20 chemprop_aux residual,
SLSQP simplex blend of three OOFs.

NEW PARADIGM:
    Cycle 169 closed the post-hoc-blend axes (substrate-change still open). This script
    tests whether the THREE-implementation diversity (LGBM leaf-wise + CatBoost ordered/Plain
    + XGBoost level-wise histogram) on the SAME K=20 chemprop_aux residual provides
    orthogonality the K=28-or-K=20 single-implementation runs cannot.

PROTOCOL:
    1. Load K=20 RFE surviving feature indices from nb2231 (same exact set used by nb2240
       and nb2270; family counts {Mordred 4, ChempropEmbed 8, AtomPair 4, MACCS 1,
       Avalon 2, ChEMBL_kNN 1}).
    2. Rebuild the 117-col 5-way K-tuned feature matrix on the 513 test compounds
       (identical construction to nb2240), slice to K=20, take unb_idx slice for 253.
    3. Anchor = chemprop_aux te[unb_idx]; residual = y_unb - anchor.
    4. Three implementations, all fit on residual, all mean-bagged across 5 seeds
       {0, 1, 7, 42, 137} with KFold(5, shuffle=True, random_state=seed):
         (a) LGBM: nb2240 config -- max_depth=4, num_leaves=15, n_estimators=300,
             learning_rate=0.03, min_child_samples=5, reg_lambda=2.
         (b) CatBoost: depth=4, iterations=300, learning_rate=0.05, loss_function='RMSE',
             allow_writing_files=False, verbose=False.
         (c) XGBoost: max_depth=4, n_estimators=300, learning_rate=0.05, reg_lambda=1,
             tree_method='hist', objective='reg:squarederror'.
       Each yields mean_bag_oof (253,) corrected (= anchor + resid_oof) + te (513,).
    5. SLSQP simplex blend of three (LGBM, CatBoost, XGBoost) corrected OOFs.
       5-fold scaffold CV on the 253 across 5 kf_seeds {1001..1005} with simplex SLSQP
       per fold (no rank-stretch -- this script tests the pure 3-impl SLSQP signal).
    6. Gate:
         mean_rae < 0.4570  -> "PROMOTE"
         mean_rae < 0.4598  -> "MARGINAL_BEAT"  (vs nb2240 pyramid pooled)
         else               -> "FAIL"
       If PROMOTE: deploy refit weights on full 253, write te_nb2833.npy + submission CSV.

Outputs:
    scripts/nb2833_three_gbm_weighted_blend.py
    data/processed/nb2833_summary.json
    data/processed/nb2833_lgbm_oof.npy             (253,) float32  corrected
    data/processed/nb2833_catboost_oof.npy         (253,) float32  corrected
    data/processed/nb2833_xgboost_oof.npy          (253,) float32  corrected
    data/processed/te_nb2833_lgbm.npy              (513,) float32
    data/processed/te_nb2833_catboost.npy          (513,) float32
    data/processed/te_nb2833_xgboost.npy           (513,) float32
    data/processed/te_nb2833.npy                   (513,) float32  (deploy blend)
    submissions/nb2833_three_gbm_weighted_blend.csv (only if PROMOTE)
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
import lightgbm as lgb
from rdkit import Chem
from rdkit import RDLogger
from scipy.optimize import minimize
from sklearn.model_selection import KFold

RDLogger.DisableLog("rdApp.*")

try:
    import xgboost as xgb
    from xgboost import XGBRegressor
except Exception as e:
    raise RuntimeError(f"xgboost import failed: {e}")

try:
    from catboost import CatBoostRegressor
except Exception as e:
    raise RuntimeError(f"catboost import failed: {e}")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2833"

# Anchor / residual config
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

# Cached feature matrices
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
NB2231_SUMMARY = DATA_PROCESSED / "nb2231_summary.json"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

# Blend / gate config
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
LB_W_OOF = 0.51
LB_W_TE = 0.49

GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598
NB2240_REF_PYRAMID = 0.4598    # nb2240 pyramid pooled_rae_mean_seeds reference
CHEMPROP_AUX_REF = 0.6216


# ============================================================================
# Helpers (copy of nb2240 feature loading)
# ============================================================================

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
    if not frames:
        raise FileNotFoundError("No local ChEMBL PXR parquets found")
    pool = pd.concat(frames, ignore_index=True)
    mols = pool["smiles"].apply(standardize)
    pool["inchikey"] = mols.apply(_safe_inchikey)
    pool["std_smiles"] = mols.apply(_safe_can_smiles)
    pool = pool[pool["inchikey"].notna() & pool["std_smiles"].notna()].copy()
    agg = (
        pool.groupby("inchikey", as_index=False)
        .agg(pec50=("pec50", "median"),
             std_smiles=("std_smiles", "first"),
             src_first=("src", "first"),
             n_meas=("pec50", "count"))
    )
    agg = agg.rename(columns={"src_first": "src"})
    return agg


def _tanimoto_topk(fp_q, fp_pool, k):
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


def _knn_predict(top_idx, top_sim, pool_labels, fallback):
    w = np.clip(top_sim, 0.0, 1.0)
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


def _load_mordred_test(n_test_expected):
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(f"Mordred cache missing -- run nb1030 first ({mte_p})")
    X_te_m = np.load(mte_p).astype(np.float32)
    if X_te_m.shape[0] != n_test_expected:
        raise ValueError(f"Mordred test shape mismatch: {X_te_m.shape}")
    X_te_m = np.where(np.isfinite(X_te_m), X_te_m, np.nan).astype(np.float32)
    col_med = np.nanmedian(X_te_m, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X_te_m)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X_te_m[idx_r, idx_c] = col_med[idx_c]
    return X_te_m


def _load_npy_test(path, n_test_expected):
    if not path.exists():
        raise FileNotFoundError(f"missing cache: {path}")
    X = np.load(path)
    if X.shape[0] != n_test_expected:
        raise ValueError(f"shape mismatch {path}: {X.shape}")
    X = X.astype(np.float32)
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X


def _extract_atompair_top_idx_from_nb1484(sum_1484):
    for f in sum_1484["families"]:
        if f["family"] == "AtomPair":
            return np.array(f["top_idx_ranked"], dtype=int)
    raise KeyError("AtomPair entry not found in nb1484_summary.json")


def _extract_best_K_record(sum_dict, records_key, best_K_key="best_K"):
    best_K = int(sum_dict[best_K_key])
    for r in sum_dict[records_key]:
        if int(r["K"]) == best_K:
            return r
    raise KeyError(f"best_K {best_K} not found in {records_key}")


# ============================================================================
# Per-implementation params
# ============================================================================

def _lgbm_params(seed):
    return dict(
        objective="regression",
        max_depth=4,
        num_leaves=15,
        n_estimators=300,
        learning_rate=0.03,
        min_child_samples=5,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _catboost_params(seed):
    return dict(
        depth=4,
        iterations=300,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=seed,
        thread_count=2,
        allow_writing_files=False,
        verbose=False,
    )


def _xgb_params(seed):
    return dict(
        max_depth=4,
        n_estimators=300,
        learning_rate=0.05,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=seed,
        n_jobs=2,
        verbosity=0,
    )


def _make_model(name, seed):
    if name == "lgbm":
        return lgb.LGBMRegressor(**_lgbm_params(seed))
    if name == "catboost":
        return CatBoostRegressor(**_catboost_params(seed))
    if name == "xgboost":
        return XGBRegressor(**_xgb_params(seed))
    raise ValueError(f"unknown impl: {name}")


# ============================================================================
# Residual cross-fit + deploy refit (per implementation)
# ============================================================================

def _residual_cross_fit_one_seed(name, X, residual, seed):
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = _make_model(name, seed)
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _train_full_then_predict_te(name, X_unb, residual, X_te, seed):
    mdl = _make_model(name, seed)
    mdl.fit(X_unb, residual)
    return mdl.predict(X_te).astype(np.float32)


def build_anchor_corrected(name, X_unb, residual, X_te, anchor_unb, te_anchor_513, y_unb,
                            n_test, n_unb, seeds):
    """Return (mean_bag_corrected_oof_253, te_corrected_513, per_seed_rae)."""
    per_seed_corrected = np.zeros((len(seeds), n_unb), dtype=np.float64)
    per_seed_te_resid = np.zeros((len(seeds), n_test), dtype=np.float64)
    per_seed_rae = []
    for i, s in enumerate(seeds):
        ts = time.time()
        resid_oof = _residual_cross_fit_one_seed(name, X_unb, residual, s)
        per_seed_corrected[i] = anchor_unb + resid_oof
        per_seed_rae.append(float(rae(y_unb, anchor_unb + resid_oof)))
        te_resid_s = _train_full_then_predict_te(name, X_unb, residual, X_te, s)
        per_seed_te_resid[i] = te_resid_s
        print(f"   [{name}] seed={s:3d}: rae_corr={per_seed_rae[-1]:.4f}  wall={time.time()-ts:.1f}s")
    mean_bag_oof = per_seed_corrected.mean(axis=0)
    mean_bag_te_resid = per_seed_te_resid.mean(axis=0)
    te_513 = te_anchor_513 + mean_bag_te_resid
    return mean_bag_oof, te_513, per_seed_rae


# ============================================================================
# SLSQP simplex blend + scaffold CV
# ============================================================================

def slsqp_simplex(P, y):
    K = P.shape[1]
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * K
    res = minimize(
        lambda w: float(np.sum((P @ w - y) ** 2)),
        np.full(K, 1.0 / K),
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
        options={"ftol": 1e-10, "maxiter": 1000},
    )
    w = np.clip(res.x, 0.0, 1.0)
    s = w.sum()
    return w / s if s > 0 else np.full(K, 1.0 / K)


def cv_run_for_seed(P_unb, y_unb, unb_scaffolds, kf_seed):
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = P_unb.shape[0]
    oof_blend = np.full(n_unb, np.nan)
    fold_w = []
    for tr_loc, va_loc in splits:
        w_f = slsqp_simplex(P_unb[tr_loc], y_unb[tr_loc])
        oof_blend[va_loc] = P_unb[va_loc] @ w_f
        fold_w.append(w_f)
    return float(rae(y_unb, oof_blend)), oof_blend, fold_w


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- LGBM + CatBoost + XGBoost weighted blend on K=20 residual")
    print("=" * 78)

    # ---- Load K=20 surviving feature indices ----
    with open(NB2231_SUMMARY) as f:
        nb2231 = json.load(f)
    surviving_K20 = list(nb2231["snapshots"]["20"]["surviving_idx_in_117"])
    surviving_K20_names = list(nb2231["snapshots"]["20"]["surviving_names"])
    assert len(surviving_K20) == 20, f"expected 20 features, got {len(surviving_K20)}"
    print(f"[load] K=20 surviving features = {len(surviving_K20)}")

    # ---- Load truth + anchor ----
    te = load_test()
    n_test = len(te)
    te_smiles = te["smiles"].astype(str).tolist() if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    te_names = te["name"].values if "name" in te.columns else te["Molecule Name"].values
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    print(f"[load] chemprop_aux in_RAE = {rae_anchor:.4f} (ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor_unb

    # ---- Rebuild 117-col matrix ----
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

    top_maccs_bit_idx = np.array(sum_1352["top_maccs_bit_indices_ranked"], dtype=int)
    rec_mord = _extract_best_K_record(sum_1523, "per_K_records", best_K_key="best_K")
    top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)
    full_ap_ranked = _extract_atompair_top_idx_from_nb1484(sum_1484)
    K_AP_best = int(sum_1524["best_K"])
    top_ap_bit_idx = full_ap_ranked[:K_AP_best]
    K_Embed_best = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed_col_idx = top_embed_full[:K_Embed_best]
    top_avalon_bit_idx = np.array(sum_1392["top_avalon_bit_indices_ranked"], dtype=int)

    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test)
    X_ap_te_top = X_ap_te[:, top_ap_bit_idx].astype(np.float32)
    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)
    X_maccs_te_top = X_maccs_te[:, top_maccs_bit_idx].astype(np.float32)
    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    X_mord_te_top = X_mord_te[:, top_mord_col_idx].astype(np.float32)
    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)
    X_emb_te_top = X_emb_te[:, top_embed_col_idx].astype(np.float32)
    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)
    X_av_te_top = X_av_te[:, top_avalon_bit_idx].astype(np.float32)

    pool = _load_chembl_pool()
    test_mols = [standardize(s) for s in te_smiles]
    test_inchikeys = set()
    for m in test_mols:
        ik = _safe_inchikey(m)
        if ik is not None:
            test_inchikeys.add(ik)
    pool = pool[~pool["inchikey"].isin(test_inchikeys)].reset_index(drop=True)
    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep_pool = fp_pool.sum(axis=1) > 0
    if not keep_pool.all():
        pool = pool[keep_pool].reset_index(drop=True)
        fp_pool = fp_pool[keep_pool]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))
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

    X_te_full = np.concatenate(
        [
            X_ap_te_top,
            X_maccs_te_top,
            X_mord_te_top,
            X_emb_te_top,
            X_av_te_top,
            pred_chembl_pec50.reshape(-1, 1).astype(np.float32),
            mean_sim.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim = X_te_full.shape[1]
    assert feat_dim == 117, f"feat_dim {feat_dim} != 117"

    X_te_K20 = X_te_full[:, surviving_K20].astype(np.float32)
    X_unb_K20 = X_te_K20[unb_idx]
    print(f"[feat] X_unb_K20 = {X_unb_K20.shape}  X_te_K20 = {X_te_K20.shape}")

    # ---- Build three corrected predictions ----
    print("\n" + "-" * 78)
    print(f"BUILD 3 IMPL CORRECTED OOFs  seeds={RESID_SEEDS}  folds={RESID_FOLDS}")
    print("-" * 78)
    impl_results = {}
    for name in ("lgbm", "catboost", "xgboost"):
        ts = time.time()
        print(f"\n[{name}]")
        mb_oof, mb_te, per_rae = build_anchor_corrected(
            name, X_unb_K20, residual, X_te_K20, anchor_unb, te_anchor_513,
            y_unb, n_test, n_unb, RESID_SEEDS,
        )
        mb_rae = float(rae(y_unb, mb_oof))
        ps_mean = float(np.mean(per_rae))
        print(f"   [{name}] per-seed mean RAE = {ps_mean:.4f}")
        print(f"   [{name}] mean-bag RAE      = {mb_rae:.4f}")
        print(f"   [{name}] wall              = {time.time()-ts:.1f}s")
        impl_results[name] = dict(
            mean_bag_oof=mb_oof,
            te_513=mb_te,
            per_seed_rae=per_rae,
            per_seed_mean_rae=ps_mean,
            mean_bag_rae=mb_rae,
        )
        np.save(DATA_PROCESSED / f"{TAG}_{name}_oof.npy", mb_oof.astype(np.float32))
        np.save(DATA_PROCESSED / f"te_{TAG}_{name}.npy", mb_te.astype(np.float32))

    # ---- SLSQP simplex 3-impl blend  (scaffold-CV) ----
    print("\n" + "=" * 78)
    print("STAGE 2: SLSQP SIMPLEX BLEND of (LGBM, CatBoost, XGBoost)")
    print("=" * 78)
    impl_order = ["lgbm", "catboost", "xgboost"]
    oof_cols = [impl_results[n]["mean_bag_oof"] for n in impl_order]
    te_cols = [impl_results[n]["te_513"] for n in impl_order]
    P_unb = np.column_stack(oof_cols).astype(np.float64)
    P_te = np.column_stack(te_cols).astype(np.float64)
    print(f"[stack] P_unb {P_unb.shape}  P_te {P_te.shape}")
    indiv_rae = {n: impl_results[n]["mean_bag_rae"] for n in impl_order}
    for n in impl_order:
        print(f"   {n:9s} oof_RAE={indiv_rae[n]:.4f}")

    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV  kf_seeds={KF_SEEDS}")
    print("-" * 78)
    per_seed = []
    all_oofs = []
    for kf_seed in KF_SEEDS:
        pooled, oof_blend, fw = cv_run_for_seed(P_unb, y_unb, unb_scaffolds, kf_seed)
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": float(pooled),
            "fold_w_mean": [float(x) for x in np.mean(fw, axis=0)],
        })
        all_oofs.append(oof_blend)
        print(f"   seed={kf_seed}  pooled_RAE={pooled:.4f}  "
              f"w_mean={np.round(np.mean(fw, axis=0), 3).tolist()}")
    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    final_oof_rae = float(rae(y_unb, mean_oof))
    pooled_rae_mean_seeds = float(np.mean([r["pooled_rae"] for r in per_seed]))
    pooled_rae_std_seeds = float(np.std([r["pooled_rae"] for r in per_seed]))
    print(f"\n[cv] mean across 5 seeds  pooled_RAE = {pooled_rae_mean_seeds:.4f} "
          f"(+/- {pooled_rae_std_seeds:.4f})")
    print(f"[cv] RAE of mean-of-seed OOFs        = {final_oof_rae:.4f}")

    # ---- Deploy ----
    print("\n" + "-" * 78)
    print("DEPLOY (refit SLSQP weights on full 253)")
    print("-" * 78)
    w_deploy = slsqp_simplex(P_unb, y_unb)
    deploy_te = (P_te @ w_deploy).astype(np.float32)
    in_rae_final = float(rae(y_unb, P_unb @ w_deploy))
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))
    w_str = ", ".join(f"{n}={w:.4f}" for n, w in zip(impl_order, w_deploy))
    print(f"   deploy weights      = {w_str}")
    print(f"   in-sample RAE (253) = {in_rae_final:.4f}")
    print(f"   te[unb_idx] RAE     = {te_unb_rae:.4f}")
    print(f"   te(513) mean/std    = {deploy_te.mean():.3f}/{deploy_te.std():.3f}")

    lb_band_est = LB_W_OOF * pooled_rae_mean_seeds + LB_W_TE * te_unb_rae
    lb_low = lb_band_est - 0.05
    lb_high = lb_band_est + 0.05
    print(f"\n[LB-band] {LB_W_OOF:.2f}*OOF + {LB_W_TE:.2f}*te_unb = {lb_band_est:.4f}  "
          f"[{lb_low:.4f}, {lb_high:.4f}]")

    # ---- Gate ----
    print("\n" + "-" * 78)
    print(f"GATE  PROMOTE<{GATE_PROMOTE}  MARGINAL_BEAT<{GATE_MARGINAL}  else FAIL")
    print("-" * 78)
    if pooled_rae_mean_seeds < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif pooled_rae_mean_seeds < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    delta_vs_nb2240 = pooled_rae_mean_seeds - NB2240_REF_PYRAMID
    print(f"   mean_rae           = {pooled_rae_mean_seeds:.4f}")
    print(f"   delta_vs_nb2240    = {delta_vs_nb2240:+.4f}")
    print(f"   verdict            = {verdict}")

    # ---- Save te artefact ----
    te_npy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_npy_path, deploy_te)
    print(f"\n[save] {te_npy_path}")

    sub_csv_path = SUBMISSIONS / f"{TAG}_three_gbm_weighted_blend.csv"
    if verdict == "PROMOTE":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": deploy_te,
        }).to_csv(sub_csv_path, index=False)
        print(f"[save] {sub_csv_path}  (PROMOTE)")
    else:
        print(f"[skip] {verdict} -- no submission CSV written")

    summary = {
        "tag": TAG,
        "method": "LGBM+CatBoost+XGBoost weighted SLSQP simplex blend on K=20 residual",
        "anchor": "chemprop_aux",
        "anchor_pre_unblind": True,
        "rae_anchor_chemprop_aux": rae_anchor,
        "k20_surviving_idx_in_117": [int(j) for j in surviving_K20],
        "k20_surviving_names": surviving_K20_names,
        "k20_family_counts": dict(nb2231["snapshots"]["20"]["family_counts"]),
        "resid_folds": RESID_FOLDS,
        "resid_seeds": RESID_SEEDS,
        "impl_order": impl_order,
        "impl_per_seed_rae": {n: impl_results[n]["per_seed_rae"] for n in impl_order},
        "impl_per_seed_mean_rae": {n: impl_results[n]["per_seed_mean_rae"] for n in impl_order},
        "impl_mean_bag_rae": indiv_rae,
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "n_unb": n_unb,
        "n_te": n_test,
        "n_unique_scaffolds": n_unique_scaf,
        "per_seed_results": per_seed,
        "pooled_rae_mean_seeds": pooled_rae_mean_seeds,
        "pooled_rae_std_seeds": pooled_rae_std_seeds,
        "mean_rae": pooled_rae_mean_seeds,
        "rae_of_mean_of_seed_oofs": final_oof_rae,
        "deploy_weights": [
            {"name": n, "w": float(w)} for n, w in zip(impl_order, w_deploy)
        ],
        "in_sample_rae_overfit_bound": in_rae_final,
        "te_unb_rae_in_sample": te_unb_rae,
        "lb_band_estimate": lb_band_est,
        "lb_band_low": lb_low,
        "lb_band_high": lb_high,
        "lb_band_w_oof": LB_W_OOF,
        "lb_band_w_te": LB_W_TE,
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "compare_nb2240_pyramid": NB2240_REF_PYRAMID,
        "delta_vs_nb2240": delta_vs_nb2240,
        "verdict": verdict,
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "te_npy_path": str(te_npy_path),
        "submission_csv": str(sub_csv_path) if verdict == "PROMOTE" else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    for n in impl_order:
        print(f"   {n:9s} mean_bag_RAE         = {indiv_rae[n]:.4f}")
    print(f"   SLSQP 3-impl pooled (5 seeds) = {pooled_rae_mean_seeds:.4f}")
    print(f"   delta vs nb2240 pyramid       = {delta_vs_nb2240:+.4f}")
    print(f"   verdict                       = {verdict}")
    print(f"   LB band estimate              = {lb_band_est:.4f}")
    print(f"   wall                          = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "impl_mean_bag_rae",
        "mean_rae",
        "delta_vs_nb2240",
        "verdict",
        "deploy_weights",
        "lb_band_estimate",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
