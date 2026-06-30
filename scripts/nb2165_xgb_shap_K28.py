"""nb2165 -- XGBoost as SHAP source for K=28 feature selection.

HYPOTHESIS:
    nb2159 found that LGBM SHAP rankings on the 117-col 5-way K-tuned matrix
    are seed-invariant (3-seed top-50 intersection >= 28 -> seed-0 ranking is
    robust signal, not noise). If we swap the SHAP SOURCE to a structurally
    different tree learner (XGBoost) we should get a DIFFERENT top-28 set
    because XGBoost uses level-wise growth with different reg shrinkage than
    LGBM's leaf-wise growth. Even if the resulting feat set is no better than
    nb2103 (mean_bag 0.4737 / median_bag 0.4698), it provides a diversity
    axis for downstream blends. Two evaluation arms:
      A) eval XGBoost on XGB-SHAP top-28 (consistency check)
      B) eval LGBM (same nb2103 config) on XGB-SHAP top-28
         -> does LGBM benefit from XGBoost's feat picks?

PROTOCOL:
    1. Anchor   = chemprop_aux te[unb_idx] (PRE-unblind, in_RAE 0.6216).
       residual = y_unb - anchor.
    2. Build the same 117-col 5-way K-tuned feature matrix used by
       nb2063/nb2103/nb2159 (AtomPair/MACCS/Mordred/ChempropEmbed/Avalon
       + ChEMBL kNN).
    3. SHAP SOURCE: Fit ONE XGBRegressor(objective='reg:squarederror',
       max_depth=4, n_estimators=300, learning_rate=0.03, reg_lambda=2,
       subsample=1.0, colsample_bytree=1.0, random_state=0) on the FULL 253
       residual. TreeExplainer -> mean |SHAP| per feature -> top-28 by
       descending importance. Report overlap with nb2063 single-seed LGBM
       top-28.
    4. EVAL ARM A: XGBoost(same hyper-params, seed swept) on the XGB-SHAP
       top-28 cols, 5-seed bag (seeds 0,1,7,42,137), KFold(n=5, shuffle=True)
       cross-fit per seed; mean-bag and median-bag RAE.
    5. EVAL ARM B: LGBM(MSE) (same nb2103 config: L=15, lr=0.03, mc=5,
       lambda=2, n_est=300) on the XGB-SHAP top-28 cols, 5-seed bag
       (seeds 0,1,7,42,137), KFold(n=5, shuffle=True) cross-fit per seed;
       mean-bag and median-bag RAE.
    6. Compare vs nb2103.K=28 (mean_bag 0.4737, median_bag 0.4698) at
       decision_margin=0.003.

Outputs:
    scripts/nb2165_xgb_shap_K28.py
    data/processed/nb2165_summary.json
    data/processed/nb2165_xgb_shap_importance_full117.npy   (117,) float32
    data/processed/nb2165_top28_idx_xgb.npy                 (28,)  int32
    data/processed/nb2165_xgb_mean_bag_oof.npy              (253,) float32
    data/processed/nb2165_lgbm_mean_bag_oof.npy             (253,) float32
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
import lightgbm as lgb
import xgboost as xgb
import shap
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb2165"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]
TARGET_K = 28
XGB_SHAP_SEED = 0

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
NB2063_SUMMARY = DATA_PROCESSED / "nb2063_summary.json"
NB2063_SHAP_IMP = DATA_PROCESSED / "nb2063_shap_importance_full117.npy"
NB2103_SUMMARY = DATA_PROCESSED / "nb2103_summary.json"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

CHEMPROP_AUX_REF = 0.6216
NB2103_K28_MEAN_BAG_REF = 0.4737
NB2103_K28_MEDIAN_BAG_REF = 0.4698
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
    """Same union as nb1852/nb1861/nb2063/nb2081/nb2091/nb2103/nb2159."""
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


def _lgbm_params(seed: int) -> dict:
    """LGBM(MSE) -- identical to nb2103/nb2159."""
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


def _xgb_params(seed: int) -> dict:
    """XGBoost params from user spec."""
    return dict(
        objective="reg:squarederror",
        max_depth=4,
        n_estimators=300,
        learning_rate=0.03,
        reg_lambda=2.0,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=seed,
        n_jobs=2,
        verbosity=0,
        tree_method="hist",
    )


def _residual_cross_fit_lgbm_one_seed(X: np.ndarray, residual: np.ndarray,
                                      seed: int) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _residual_cross_fit_xgb_one_seed(X: np.ndarray, residual: np.ndarray,
                                     seed: int) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = xgb.XGBRegressor(**_xgb_params(seed))
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


def _bag_eval(per_seed_corrected: np.ndarray, y_unb: np.ndarray,
              per_seed_rae: list[float], rae_anchor: float, label: str):
    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))
    arr = np.array(per_seed_rae)
    info = {
        "label": label,
        "per_seed_rae": per_seed_rae,
        "rae_per_seed_mean": float(arr.mean()),
        "rae_per_seed_median": float(np.median(arr)),
        "rae_per_seed_std": float(arr.std()),
        "rae_per_seed_min": float(arr.min()),
        "rae_per_seed_max": float(arr.max()),
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_nb2103_K28": rae_mean_bag - NB2103_K28_MEAN_BAG_REF,
        "delta_median_bag_vs_nb2103_K28": rae_median_bag - NB2103_K28_MEDIAN_BAG_REF,
        "delta_mean_bag_vs_anchor": rae_mean_bag - rae_anchor,
    }
    print(f"   [{label}] per-seed RAE = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   [{label}] per-seed mean/median/std = "
          f"{info['rae_per_seed_mean']:.4f} / "
          f"{info['rae_per_seed_median']:.4f} / "
          f"{info['rae_per_seed_std']:.4f}")
    print(f"   [{label}] POOLED mean_bag   = {rae_mean_bag:.4f}  "
          f"(vs nb2103.K28 mean   = {NB2103_K28_MEAN_BAG_REF:.4f}, "
          f"delta = {info['delta_mean_bag_vs_nb2103_K28']:+.4f})")
    print(f"   [{label}] POOLED median_bag = {rae_median_bag:.4f}  "
          f"(vs nb2103.K28 median = {NB2103_K28_MEDIAN_BAG_REF:.4f}, "
          f"delta = {info['delta_median_bag_vs_nb2103_K28']:+.4f})")
    return mean_bag_oof.astype(np.float32), median_bag_oof.astype(np.float32), info


def _verdict_for(label: str, info: dict, rae_anchor: float) -> str:
    beats_mean = info["rae_mean_bag"] < NB2103_K28_MEAN_BAG_REF - DECISION_MARGIN
    flat_mean = abs(info["delta_mean_bag_vs_nb2103_K28"]) < DECISION_MARGIN
    beats_anchor = info["rae_mean_bag"] < rae_anchor - DECISION_MARGIN
    if beats_mean:
        return f"{label}_BEATS_NB2103_K28_NEW_CANDIDATE"
    if flat_mean:
        return f"{label}_FLAT_VS_NB2103_K28"
    if beats_anchor:
        return f"{label}_BEATS_ANCHOR_BUT_WORSE_THAN_NB2103_K28"
    if abs(info["rae_mean_bag"] - rae_anchor) < DECISION_MARGIN:
        return f"{label}_FLAT_VS_ANCHOR"
    return f"{label}_HURTS_ANCHOR"


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- XGBoost as SHAP source, K={TARGET_K} on 117-col 5-way "
          f"K-tuned matrix")
    print(f"          anchor={ANCHOR}  eval-seeds={RESID_SEEDS}  "
          f"folds={RESID_FOLDS}")
    print(f"          ref: nb2103.K=28 mean_bag={NB2103_K28_MEAN_BAG_REF:.4f} "
          f"median_bag={NB2103_K28_MEDIAN_BAG_REF:.4f}  "
          f"margin={DECISION_MARGIN}")
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

    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"chemprop_aux te file missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(
            f"chemprop_aux te shape mismatch: {te_anchor_513.shape} vs {n_test}"
        )
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Load all K-grid winners ----
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
    print(f"[reuse] top-{n_top_ap}     AtomPair bits")
    print(f"[reuse] top-{n_top_maccs}     MACCS bits")
    print(f"[reuse] top-{n_top_mord}     Mordred cols")
    print(f"[reuse] top-{n_top_embed}     ChempropEmbed dims")
    print(f"[reuse] top-{n_top_avalon}     Avalon bits")

    # ---- Feature matrices ----
    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test)
    X_ap_unb = X_ap_te[unb_idx].astype(np.float32)
    X_ap_unb_top = X_ap_unb[:, top_ap_bit_idx].astype(np.float32)

    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)
    X_maccs_unb = X_maccs_te[unb_idx].astype(np.float32)
    X_maccs_unb_top = X_maccs_unb[:, top_maccs_bit_idx].astype(np.float32)

    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    X_mord_unb = X_mord_te[unb_idx].astype(np.float32)
    X_mord_unb_top = X_mord_unb[:, top_mord_col_idx].astype(np.float32)

    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)
    X_emb_unb = X_emb_te[unb_idx].astype(np.float32)
    X_emb_unb_top = X_emb_unb[:, top_embed_col_idx].astype(np.float32)

    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)
    X_av_unb = X_av_te[unb_idx].astype(np.float32)
    X_av_unb_top = X_av_unb[:, top_avalon_bit_idx].astype(np.float32)

    # ---- ChEMBL kNN feature ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL (union)")
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

    # ---- Build COMBINED 5-way K-tuned 117-col feature matrix ----
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
    print(f"\n   COMBINED 5-WAY K-TUNED matrix: {X_unb.shape}")

    # ---- Feature names ----
    feat_names: list[str] = []
    feat_family: list[str] = []
    for j, b in enumerate(top_ap_bit_idx):
        feat_names.append(f"AtomPair_bit_{int(b)}")
        feat_family.append("AtomPair")
    for j, b in enumerate(top_maccs_bit_idx):
        feat_names.append(f"MACCS_bit_{int(b)}")
        feat_family.append("MACCS")
    for j, c in enumerate(top_mord_col_idx):
        feat_names.append(f"Mordred_col_{int(c)}")
        feat_family.append("Mordred")
    for j, d in enumerate(top_embed_col_idx):
        feat_names.append(f"ChempropEmbed_dim_{int(d)}")
        feat_family.append("ChempropEmbed")
    for j, b in enumerate(top_avalon_bit_idx):
        feat_names.append(f"Avalon_bit_{int(b)}")
        feat_family.append("Avalon")
    feat_names.append("pred_chembl_pec50")
    feat_family.append("ChEMBL_kNN")
    feat_names.append("mean_sim")
    feat_family.append("ChEMBL_kNN")
    assert len(feat_names) == feat_dim

    # ---- STEP 1: XGBoost SHAP on FULL 253 residual ----
    print("\n" + "-" * 78)
    print(f"STEP 1: XGBoost SHAP source on FULL 253 residual (seed="
          f"{XGB_SHAP_SEED})")
    print("-" * 78)
    ts = time.time()
    xgb_shap_model = xgb.XGBRegressor(**_xgb_params(seed=XGB_SHAP_SEED))
    xgb_shap_model.fit(X_unb, residual)
    explainer = shap.TreeExplainer(xgb_shap_model)
    sv = explainer.shap_values(X_unb)
    xgb_shap_imp = np.abs(sv).mean(axis=0).astype(np.float32)
    if xgb_shap_imp.shape[0] != feat_dim:
        raise ValueError(
            f"XGB SHAP imp shape {xgb_shap_imp.shape} != feat_dim {feat_dim}"
        )
    print(f"   XGB SHAP fit + explain wall = {time.time() - ts:.1f}s")

    xgb_top28_idx = np.argsort(-xgb_shap_imp)[:TARGET_K].astype(np.int32)
    xgb_top28_set = set(int(x) for x in xgb_top28_idx)

    # ---- Compare to nb2063 single-seed (seed=0) LGBM SHAP top-28 ----
    if not NB2063_SHAP_IMP.exists():
        raise FileNotFoundError(f"missing {NB2063_SHAP_IMP} -- run nb2063 first")
    nb2063_shap_imp = np.load(NB2063_SHAP_IMP).astype(np.float32)
    nb2063_top28_idx = np.argsort(-nb2063_shap_imp)[:TARGET_K].astype(np.int32)
    nb2063_top28_set = set(int(x) for x in nb2063_top28_idx)

    overlap_set = xgb_top28_set & nb2063_top28_set
    overlap_count = int(len(overlap_set))
    new_in_xgb = sorted(int(x) for x in xgb_top28_set - nb2063_top28_set)
    dropped_from_lgbm = sorted(
        int(x) for x in nb2063_top28_set - xgb_top28_set
    )

    xgb_top28_names = [feat_names[i] for i in xgb_top28_idx.tolist()]
    xgb_top28_family = [feat_family[i] for i in xgb_top28_idx.tolist()]
    xgb_top28_imp = [float(xgb_shap_imp[i]) for i in xgb_top28_idx.tolist()]
    fam_counts: dict[str, int] = {}
    for fam in xgb_top28_family:
        fam_counts[fam] = fam_counts.get(fam, 0) + 1

    print(f"\n   XGB top-{TARGET_K} family breakdown: {fam_counts}")
    print(f"   XGB top-10 by mean |SHAP|:")
    for rank, (name, imp, fam) in enumerate(
            zip(xgb_top28_names[:10], xgb_top28_imp[:10], xgb_top28_family[:10]), 1):
        print(f"     {rank:2d}. [{fam:14s}] {name:30s}  mean|SHAP|={imp:.5f}")

    print(f"\n   vs nb2063 LGBM single-seed top-28:")
    print(f"     overlap          = {overlap_count} / {TARGET_K}")
    print(f"     new (XGB only)   = {len(new_in_xgb)}")
    for idx in new_in_xgb:
        print(f"        + col {idx:3d} [{feat_family[idx]:14s}] {feat_names[idx]}")
    print(f"     dropped (LGBM only) = {len(dropped_from_lgbm)}")
    for idx in dropped_from_lgbm:
        print(f"        - col {idx:3d} [{feat_family[idx]:14s}] {feat_names[idx]}")

    # ---- STEP 2 (ARM A): XGBoost cross-fit on XGB-SHAP top-28 ----
    X_top28 = X_unb[:, xgb_top28_idx].astype(np.float32)
    print(f"\n   X_top28 shape = {X_top28.shape}")
    print("\n" + "-" * 78)
    print(f"STEP 2A: XGBoost 5-seed bag 5-fold cross-fit on XGB-SHAP K={TARGET_K}")
    print("-" * 78)
    xgb_per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb),
                                       dtype=np.float64)
    xgb_per_seed_rae: list[float] = []
    xgb_per_seed_records = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        resid_oof_s = _residual_cross_fit_xgb_one_seed(X_top28, residual, s)
        pred_corr_s = anchor + resid_oof_s
        xgb_per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        xgb_per_seed_rae.append(rae_s)
        delta_s = rae_s - rae_anchor
        xgb_per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_chemprop_aux": delta_s,
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   XGB eval-seed {s:3d}:  rae = {rae_s:.4f}  "
              f"(d_vs_anchor = {delta_s:+.4f})  wall = {time.time() - ts:.1f}s")

    xgb_mean_bag_oof, xgb_median_bag_oof, xgb_info = _bag_eval(
        xgb_per_seed_corrected, y_unb, xgb_per_seed_rae, rae_anchor,
        label="XGB_on_XGBSHAP"
    )

    # ---- STEP 2 (ARM B): LGBM cross-fit on XGB-SHAP top-28 ----
    print("\n" + "-" * 78)
    print(f"STEP 2B: LGBM(MSE) 5-seed bag 5-fold cross-fit on XGB-SHAP "
          f"K={TARGET_K}")
    print("-" * 78)
    lgbm_per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb),
                                        dtype=np.float64)
    lgbm_per_seed_rae: list[float] = []
    lgbm_per_seed_records = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        resid_oof_s = _residual_cross_fit_lgbm_one_seed(X_top28, residual, s)
        pred_corr_s = anchor + resid_oof_s
        lgbm_per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        lgbm_per_seed_rae.append(rae_s)
        delta_s = rae_s - rae_anchor
        lgbm_per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_chemprop_aux": delta_s,
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   LGBM eval-seed {s:3d}: rae = {rae_s:.4f}  "
              f"(d_vs_anchor = {delta_s:+.4f})  wall = {time.time() - ts:.1f}s")

    lgbm_mean_bag_oof, lgbm_median_bag_oof, lgbm_info = _bag_eval(
        lgbm_per_seed_corrected, y_unb, lgbm_per_seed_rae, rae_anchor,
        label="LGBM_on_XGBSHAP"
    )

    # ---- Verdicts ----
    verdict_xgb = _verdict_for("XGB_on_XGBSHAP", xgb_info, rae_anchor)
    verdict_lgbm = _verdict_for("LGBM_on_XGBSHAP", lgbm_info, rae_anchor)
    if (lgbm_info["rae_mean_bag"] <
            NB2103_K28_MEAN_BAG_REF - DECISION_MARGIN):
        global_verdict = (
            "LGBM_BENEFITS_FROM_XGB_SHAP_PICKS_NEW_PRIMARY_CANDIDATE"
        )
    elif abs(lgbm_info["delta_mean_bag_vs_nb2103_K28"]) < DECISION_MARGIN:
        global_verdict = "LGBM_ON_XGB_SHAP_FLAT_VS_NB2103_K28"
    elif (xgb_info["rae_mean_bag"] <
          NB2103_K28_MEAN_BAG_REF - DECISION_MARGIN):
        global_verdict = "XGB_ON_XGB_SHAP_BEATS_NB2103_K28_NEW_CANDIDATE"
    elif overlap_count == TARGET_K:
        global_verdict = (
            "XGB_SHAP_IDENTICAL_TO_LGBM_SHAP_NO_DIVERSITY_AXIS"
        )
    else:
        global_verdict = "XGB_SHAP_PROVIDES_DIVERSITY_BUT_NO_LB_GAIN"

    print(f"\n   verdict_xgb_arm   = {verdict_xgb}")
    print(f"   verdict_lgbm_arm  = {verdict_lgbm}")
    print(f"   global_verdict    = {global_verdict}")
    print(f"   PRE-unblind clean = True")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_xgb_shap_importance_full117.npy",
            xgb_shap_imp)
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_xgb_shap_importance_full117.npy'}")
    np.save(DATA_PROCESSED / f"{TAG}_top28_idx_xgb.npy", xgb_top28_idx)
    print(f"[save] {DATA_PROCESSED / f'{TAG}_top28_idx_xgb.npy'}")
    np.save(DATA_PROCESSED / f"{TAG}_xgb_mean_bag_oof.npy", xgb_mean_bag_oof)
    print(f"[save] {DATA_PROCESSED / f'{TAG}_xgb_mean_bag_oof.npy'}")
    np.save(DATA_PROCESSED / f"{TAG}_lgbm_mean_bag_oof.npy", lgbm_mean_bag_oof)
    print(f"[save] {DATA_PROCESSED / f'{TAG}_lgbm_mean_bag_oof.npy'}")

    xgb_top28_records = [
        {
            "rank": int(rank),
            "feat_idx_in_117": int(idx),
            "feat_name": str(xgb_top28_names[rank - 1]),
            "feat_family": str(xgb_top28_family[rank - 1]),
            "mean_abs_shap_xgb_seed0": float(xgb_top28_imp[rank - 1]),
        }
        for rank, idx in enumerate(xgb_top28_idx.tolist(), 1)
    ]

    summary = {
        "tag": TAG,
        "method": ("xgboost_shap_top28_on_117col_plus_dual_eval_xgb_and_lgbm"),
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "data_source": ("nb2063/nb2103/nb2159 117-col 5-way K-tuned matrix: "
                        "AtomPair/MACCS/Mordred/ChempropEmbed/Avalon + ChEMBL kNN"),
        "shap_source_model": "XGBRegressor",
        "xgb_shap_seed": XGB_SHAP_SEED,
        "xgb_params": _xgb_params(XGB_SHAP_SEED),
        "lgbm_params": _lgbm_params(0),
        "target_K": TARGET_K,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "feat_dim_full": int(feat_dim),
        "feat_breakdown_full": {
            "atompair": n_top_ap,
            "maccs": n_top_maccs,
            "mordred": n_top_mord,
            "chemprop_embed": n_top_embed,
            "avalon": n_top_avalon,
            "pred_chembl_pec50": 1,
            "mean_sim": 1,
            "total": int(feat_dim),
        },
        "xgb_top28_idx_in_117": [int(x) for x in xgb_top28_idx.tolist()],
        "xgb_top28_records": xgb_top28_records,
        "family_counts_xgb_top28": fam_counts,
        "nb2063_lgbm_top28_idx_seed0": [int(x) for x in nb2063_top28_idx.tolist()],
        "vs_nb2063_lgbm_top28": {
            "overlap_count": overlap_count,
            "new_in_xgb": [
                {"col": int(i), "name": feat_names[i],
                 "family": feat_family[i]}
                for i in new_in_xgb
            ],
            "dropped_from_lgbm": [
                {"col": int(i), "name": feat_names[i],
                 "family": feat_family[i]}
                for i in dropped_from_lgbm
            ],
        },
        "n_chembl_pool": int(len(pool)),
        "n_unb": n_unb,
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        # ARM A: XGB on XGB-SHAP
        "xgb_arm": {
            "per_seed_rae": xgb_per_seed_rae,
            "per_seed_records": xgb_per_seed_records,
            "rae_per_seed_mean": xgb_info["rae_per_seed_mean"],
            "rae_per_seed_median": xgb_info["rae_per_seed_median"],
            "rae_per_seed_std": xgb_info["rae_per_seed_std"],
            "rae_per_seed_min": xgb_info["rae_per_seed_min"],
            "rae_per_seed_max": xgb_info["rae_per_seed_max"],
            "rae_mean_bag": xgb_info["rae_mean_bag"],
            "rae_median_bag": xgb_info["rae_median_bag"],
            "delta_mean_bag_vs_nb2103_K28": xgb_info["delta_mean_bag_vs_nb2103_K28"],
            "delta_median_bag_vs_nb2103_K28": xgb_info["delta_median_bag_vs_nb2103_K28"],
            "delta_mean_bag_vs_anchor": xgb_info["delta_mean_bag_vs_anchor"],
            "verdict": verdict_xgb,
        },
        # ARM B: LGBM on XGB-SHAP
        "lgbm_arm": {
            "per_seed_rae": lgbm_per_seed_rae,
            "per_seed_records": lgbm_per_seed_records,
            "rae_per_seed_mean": lgbm_info["rae_per_seed_mean"],
            "rae_per_seed_median": lgbm_info["rae_per_seed_median"],
            "rae_per_seed_std": lgbm_info["rae_per_seed_std"],
            "rae_per_seed_min": lgbm_info["rae_per_seed_min"],
            "rae_per_seed_max": lgbm_info["rae_per_seed_max"],
            "rae_mean_bag": lgbm_info["rae_mean_bag"],
            "rae_median_bag": lgbm_info["rae_median_bag"],
            "delta_mean_bag_vs_nb2103_K28": lgbm_info["delta_mean_bag_vs_nb2103_K28"],
            "delta_median_bag_vs_nb2103_K28": lgbm_info["delta_median_bag_vs_nb2103_K28"],
            "delta_mean_bag_vs_anchor": lgbm_info["delta_mean_bag_vs_anchor"],
            "verdict": verdict_lgbm,
        },
        "nb2103_K28_mean_bag_ref": NB2103_K28_MEAN_BAG_REF,
        "nb2103_K28_median_bag_ref": NB2103_K28_MEDIAN_BAG_REF,
        "decision_margin": DECISION_MARGIN,
        "global_verdict": global_verdict,
        "pre_unblind_clean": True,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "target_K",
        "rae_anchor_chemprop_aux",
        "nb2103_K28_mean_bag_ref", "nb2103_K28_median_bag_ref",
        "global_verdict", "pre_unblind_clean",
    ):
        print(f"  {k}: {res.get(k)}")
    print("\n==== vs nb2063 LGBM single-seed top-28 ====")
    vs = res["vs_nb2063_lgbm_top28"]
    print(f"  overlap_count: {vs['overlap_count']} / {res['target_K']}")
    print(f"  new in XGB:")
    for d in vs["new_in_xgb"]:
        print(f"    + col {d['col']:3d} [{d['family']:14s}] {d['name']}")
    print(f"  dropped from LGBM seed-0:")
    for d in vs["dropped_from_lgbm"]:
        print(f"    - col {d['col']:3d} [{d['family']:14s}] {d['name']}")
    print("\n==== ARM A: XGB on XGB-SHAP K=28 ====")
    a = res["xgb_arm"]
    print(f"  per_seed_rae:   {a['per_seed_rae']}")
    print(f"  rae_mean_bag:   {a['rae_mean_bag']:.4f}  "
          f"(d_vs_nb2103.K28 = {a['delta_mean_bag_vs_nb2103_K28']:+.4f})")
    print(f"  rae_median_bag: {a['rae_median_bag']:.4f}  "
          f"(d_vs_nb2103.K28 = {a['delta_median_bag_vs_nb2103_K28']:+.4f})")
    print(f"  verdict:        {a['verdict']}")
    print("\n==== ARM B: LGBM on XGB-SHAP K=28 ====")
    b = res["lgbm_arm"]
    print(f"  per_seed_rae:   {b['per_seed_rae']}")
    print(f"  rae_mean_bag:   {b['rae_mean_bag']:.4f}  "
          f"(d_vs_nb2103.K28 = {b['delta_mean_bag_vs_nb2103_K28']:+.4f})")
    print(f"  rae_median_bag: {b['rae_median_bag']:.4f}  "
          f"(d_vs_nb2103.K28 = {b['delta_median_bag_vs_nb2103_K28']:+.4f})")
    print(f"  verdict:        {b['verdict']}")
