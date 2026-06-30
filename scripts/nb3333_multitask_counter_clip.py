"""nb3333 -- K=18 MULTITASK (pEC50 + counter pEC50 heads) + learned clip.

NEW PARADIGM (two LGBM heads sharing K=18 SHAP features):
    Head A -- predicts the pEC50 residual    r_A = y_unb - chemprop_aux.
    Head B -- predicts the counter pEC50 residual
              r_B = counter_clean - chemprop_aux  (selectivity axis).

    Both heads consume the SAME K=18 SHAP-pruned feature block (top-18 of the
    nb2063 117-col 5-way K-tuned matrix). Head B is trained jointly as a
    REGULARIZER / feature-enricher: its out-of-fold prediction is STACKED as a
    19th input feature into head A. Because the counter axis is biologically
    orthogonal to PXR pEC50 (corr(chemprop_aux, counter)=0.07 on the 253),
    head B injects a selectivity-aware signal that head A could not otherwise
    see from the K=18 block alone -- a multitask soft-share through the
    stacked OOF rather than a hard shared trunk.

    We deploy HEAD A (corrected pEC50). Head B exists only to enrich A.
    Finally a per-fold LEARNED clip (q_low, q_high grid, picked on fold-train
    only) tames the variance-compressed tails (nb3201 primitive).

LEAK-SAFE NESTED STACKING (per outer fold of head A):
    To prevent head B's OOF feature from leaking the val rows into head A:
      a) Inner 5-fold (same scaffold splitter, derived seed) on OUTER-TRAIN
         generates head-B OOF features for every outer-train row.
      b) Head B is refit on the FULL outer-train, then predicts head-B
         features for the OUTER-VAL rows (these val rows were never in any
         head-B training fold for this outer fold).
      c) Head A is fit on [X_K18(outer_train) | bB_oof(outer_train)] vs
         r_A(outer_train); predicts [X_K18(outer_val) | bB(outer_val)].
      d) Learned clip (q_low*, q_high*) picked on fold-train r_A-corrected,
         applied to fold-val corrected preds.

PROTOCOL:
    anchor      = te_chemprop_aux.npy   (PRE-clean, RAE 0.6216 on 253)
    counter     = te_nb2490_counter.npy (PRE-clean counter pEC50 on 513)
    r_A         = y_unb - anchor[unb]
    r_B(target) = counter[unb] - anchor[unb]
    15 FRESH kf_seeds {1216..1230}; per outer fold report val RAE.
    GATE metric = per-fold-mean (mean of 5 per-fold val RAEs, averaged over
                  15 seeds).

GATE:
    per_fold_mean < 0.4423 -> "BETTER"
    else                   -> "FAIL"

References:
    chemprop_aux te[unb] RAE      = 0.6216   <- anchor (PRE-clean)
    counter_clean te[unb] RAE     = 2.1382   <- counter axis (poor alone)
    nb2960 K18 deep-30 OOF        = 0.4536
    nb3030 K18 wide-seed ceiling  = 0.4509
    nb2171 prior post-hoc top     = 0.4682
    nb3201 learned-clip K18 prim  = clip grid {0.01,0.05,0.10}x{0.90,0.95,0.98}

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/te_chemprop_aux.npy
    data/processed/te_nb2490_counter.npy
    data/processed/nb2063_shap_importance_full117.npy
    + 117-col feature pieces (te_atompair/te_maccs/te_avalon512/
      te_chemprop_embed_300 + Mordred cache + ChEMBL kNN), same as nb3111.

Outputs:
    scripts/nb3333_multitask_counter_clip.py
    data/processed/nb3333_summary.json
    data/processed/nb3333_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3333.npy         (513,) float32 -- deploy te
    submissions/nb3333_multitask_counter_clip.csv  (only on BETTER verdict)
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3333"

# -- Anchors -------------------------------------------------------------------
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"       # PRE-clean pEC50
COUNTER_TE_PATH = DATA_PROCESSED / "te_nb2490_counter.npy"    # PRE-clean counter

# -- SHAP K=18 selection (top-18 of nb2063 117-col matrix) ---------------------
NB2063_SHAP_IMP = DATA_PROCESSED / "nb2063_shap_importance_full117.npy"
K = 18

# -- 117-col feature pieces (same as nb3111 / nb2103) --------------------------
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

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
INNER_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))  # 15 FRESH seeds {1216..1230}

# -- Per-fold learned-clip grid (nb3201 primitive) -----------------------------
Q_LOW_GRID = [0.01, 0.05, 0.10]
Q_HIGH_GRID = [0.90, 0.95, 0.98]

# -- Gate / references ---------------------------------------------------------
GATE_BETTER = 0.4423
REF_CHEMPROP_AUX = 0.6216
REF_COUNTER = 2.1382
REF_K18_DEEP30 = 0.4536
REF_NB3030 = 0.4509
REF_NB2171 = 0.4682


# ============================================================================
# 117-col feature builder (verbatim port from nb3111 _build_X_te_117)
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


def _build_X_te_117(te_smiles, n_test):
    """Build the 117-col 5-way K-tuned feature matrix on the 513 test."""
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

    # ChEMBL kNN
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
    return X_te_full


# ============================================================================
# LGBM heads
# ============================================================================

def _lgbm_params(seed):
    """LGBM(MSE) -- identical hyperparams to nb2103 / nb2960 / nb3111 K-residual."""
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


def _fit_headB_oof_on_train(
    X_tr: np.ndarray,
    rB_tr: np.ndarray,
    scaf_tr: list,
    seed: int,
) -> np.ndarray:
    """Inner CV: head-B OOF predictions for every OUTER-TRAIN row (no leak)."""
    n_tr = len(rB_tr)
    inner_splits = scaffold_kfold_indices(
        scaf_tr, n_splits=INNER_FOLDS, shuffle=True, seed=seed + 7919,
    )
    bB_oof = np.full(n_tr, np.nan, dtype=np.float64)
    for itr, iva in inner_splits:
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X_tr[itr], rB_tr[itr])
        bB_oof[iva] = mdl.predict(X_tr[iva])
    # Inner scaffold split may leave a few singleton-scaffold rows uncovered if
    # a scaffold lands entirely in one inner fold's val with no train sibling;
    # fill any residual NaN with a full-train-fit head-B prediction.
    if np.isnan(bB_oof).any():
        nan_mask = np.isnan(bB_oof)
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X_tr, rB_tr)
        bB_oof[nan_mask] = mdl.predict(X_tr[nan_mask])
    return bB_oof


def _pick_best_clip(y_tr: np.ndarray, pred_tr: np.ndarray):
    """Inner grid: pick (q_low*, q_high*) minimizing fold-train RAE (nb3201)."""
    best_rae = np.inf
    best_ql, best_qh = Q_LOW_GRID[0], Q_HIGH_GRID[-1]
    best_lo = float(np.quantile(y_tr, best_ql))
    best_hi = float(np.quantile(y_tr, best_qh))
    for ql in Q_LOW_GRID:
        lo = float(np.quantile(y_tr, ql))
        for qh in Q_HIGH_GRID:
            hi = float(np.quantile(y_tr, qh))
            if hi <= lo:
                continue
            clipped = np.clip(pred_tr, lo, hi)
            r = float(rae(y_tr, clipped))
            if r < best_rae:
                best_rae, best_ql, best_qh = r, ql, qh
                best_lo, best_hi = lo, hi
    return best_ql, best_qh, best_lo, best_hi


def _run_one_seed(
    X_unb: np.ndarray,
    rA: np.ndarray,
    rB: np.ndarray,
    anchor_unb: np.ndarray,
    y_unb: np.ndarray,
    scaffolds: list,
    seed: int,
) -> dict:
    """One kf_seed: nested multitask (head B -> head A) + learned clip."""
    splits = scaffold_kfold_indices(
        scaffolds, n_splits=N_FOLDS, shuffle=True, seed=seed,
    )
    n_unb = len(y_unb)
    oof_corr = np.full(n_unb, np.nan, dtype=np.float64)   # head-A corrected (pre-clip)
    oof_clip = np.full(n_unb, np.nan, dtype=np.float64)   # head-A corrected + clip
    fold_val_raes_corr = []
    fold_val_raes_clip = []
    fold_ql, fold_qh = [], []
    fold_bB_gain = []  # head-B feature gain (importance share) in head A
    for tr_loc, va_loc in splits:
        scaf_tr = [scaffolds[i] for i in tr_loc]
        # -- head B: OOF features for outer-train (inner CV), refit -> outer-val
        bB_tr = _fit_headB_oof_on_train(X_unb[tr_loc], rB[tr_loc], scaf_tr, seed)
        mdlB_full = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdlB_full.fit(X_unb[tr_loc], rB[tr_loc])
        bB_va = mdlB_full.predict(X_unb[va_loc])
        # -- head A: stack head-B OOF as 19th feature
        XA_tr = np.column_stack([X_unb[tr_loc], bB_tr]).astype(np.float32)
        XA_va = np.column_stack([X_unb[va_loc], bB_va]).astype(np.float32)
        mdlA = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdlA.fit(XA_tr, rA[tr_loc])
        # head-B feature importance share inside head A (gain)
        imp = mdlA.booster_.feature_importance(importance_type="gain")
        bB_gain = float(imp[-1] / max(imp.sum(), 1e-9))
        fold_bB_gain.append(bB_gain)
        # -- corrected pEC50 on val (and on train, for clip fitting)
        resid_va = mdlA.predict(XA_va)
        resid_tr = mdlA.predict(XA_tr)
        corr_va = anchor_unb[va_loc] + resid_va
        corr_tr = anchor_unb[tr_loc] + resid_tr
        oof_corr[va_loc] = corr_va
        fold_val_raes_corr.append(float(rae(y_unb[va_loc], corr_va)))
        # -- learned clip: pick band on fold-train corrected, apply to val
        ql, qh, lo, hi = _pick_best_clip(y_unb[tr_loc], corr_tr)
        fold_ql.append(ql)
        fold_qh.append(qh)
        clip_va = np.clip(corr_va, lo, hi)
        oof_clip[va_loc] = clip_va
        fold_val_raes_clip.append(float(rae(y_unb[va_loc], clip_va)))

    if np.isnan(oof_clip).any():
        raise RuntimeError(f"seed={seed}: scaffold splits did not cover all rows")
    pooled_corr = float(rae(y_unb, oof_corr))
    pooled_clip = float(rae(y_unb, oof_clip))
    return {
        "kf_seed": int(seed),
        "per_fold_mean_corr": float(np.mean(fold_val_raes_corr)),
        "per_fold_mean_clip": float(np.mean(fold_val_raes_clip)),
        "per_fold_std_clip": float(np.std(fold_val_raes_clip, ddof=1)),
        "pooled_corr": pooled_corr,
        "pooled_clip": pooled_clip,
        "fold_ql": fold_ql,
        "fold_qh": fold_qh,
        "bB_gain_mean": float(np.mean(fold_bB_gain)),
        "oof_clip": oof_clip,
        "oof_corr": oof_corr,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- K=18 MULTITASK (pEC50 head A + counter head B) + learned clip")
    print(f"         head B counter residual STACKED as 19th feature into head A")
    print(f"         kf_seeds = {len(KF_SEEDS)} fresh "
          f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}  folds={N_FOLDS} (inner={INNER_FOLDS})")
    print(f"         gate metric = per-fold-mean; BETTER if < {GATE_BETTER:.4f}")
    print("=" * 78)

    # -- Load test, truth ----------------------------------------------------
    te = load_test()
    n_test = len(te)
    te_smiles = (
        te["smiles"].astype(str).tolist()
        if "smiles" in te.columns
        else te["SMILES"].astype(str).tolist()
    )
    te_names = (
        te["name"].values if "name" in te.columns else te["Molecule Name"].values
    )
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    assert n_unb == 253, f"expected 253 unblind, got {n_unb}"
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    # -- Anchors (PRE-clean) -------------------------------------------------
    anchor_te = np.load(ANCHOR_TE_PATH).astype(np.float64)
    counter_te = np.load(COUNTER_TE_PATH).astype(np.float64)
    if anchor_te.shape[0] != n_test or counter_te.shape[0] != n_test:
        raise ValueError("anchor/counter te shape mismatch with n_test")
    anchor_unb = anchor_te[unb_idx]
    counter_unb = counter_te[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    rae_counter = float(rae(y_unb, counter_unb))
    print(f"[anchor] chemprop_aux te[unb] RAE = {rae_anchor:.4f}  (ref {REF_CHEMPROP_AUX:.4f})")
    print(f"[anchor] counter_clean te[unb] RAE = {rae_counter:.4f}  (ref {REF_COUNTER:.4f})")

    # head A target = pEC50 residual; head B target = counter residual (selectivity)
    rA = y_unb - anchor_unb
    rB = counter_unb - anchor_unb
    print(f"[resid A] y - anchor:        mean={rA.mean():+.4f}  std={rA.std():.4f}")
    print(f"[resid B] counter - anchor:  mean={rB.mean():+.4f}  std={rB.std():.4f}")
    print(f"[resid] corr(rA, rB) = {np.corrcoef(rA, rB)[0, 1]:+.4f}")

    # leak sanity
    for lbl, p in (("anchor", anchor_unb), ("counter", counter_unb)):
        frac = float(np.mean(np.isclose(p, y_unb, atol=1e-6)))
        if frac > 0.05:
            print(f"   WARN {lbl}: {frac:.1%} rows == truth -- possible leak")

    # -- Build 117-col matrix, slice to SHAP top-18 --------------------------
    print("\n" + "-" * 78)
    print(f"STEP 1: build 117-col matrix, slice to SHAP top-{K}")
    print("-" * 78)
    shap_imp = np.load(NB2063_SHAP_IMP).astype(np.float32)
    if shap_imp.shape[0] != 117:
        raise ValueError(f"SHAP importance length {shap_imp.shape[0]} != 117")
    topK_idx = np.argsort(-shap_imp)[:K].astype(np.int32)
    print(f"   SHAP top-{K} idx (in 117) = {topK_idx.tolist()}")

    X_te_117 = _build_X_te_117(te_smiles, n_test)
    X_te_K = X_te_117[:, topK_idx].astype(np.float32)
    X_unb_K = X_te_K[unb_idx].astype(np.float32)
    print(f"   X_unb_K = {X_unb_K.shape}   X_te_K = {X_te_K.shape}")

    # -- Unblind scaffolds ---------------------------------------------------
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"   n_unique_scaffolds (unb) = {n_unique_scaf}")

    # -- Multi-seed sweep ----------------------------------------------------
    print("\n" + "-" * 78)
    print(f"SWEEP: {len(KF_SEEDS)} FRESH kf_seeds  "
          f"(nested head-B -> head-A + learned clip)")
    print("-" * 78)
    seed_records = []
    per_fold_means_clip = []
    pooled_clips = []
    oof_clip_stack = []
    all_ql, all_qh = [], []
    bB_gains = []
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(
            X_unb_K, rA, rB, anchor_unb, y_unb, unb_scaffolds, s,
        )
        per_fold_means_clip.append(res["per_fold_mean_clip"])
        pooled_clips.append(res["pooled_clip"])
        oof_clip_stack.append(res["oof_clip"])
        all_ql.extend(res["fold_ql"])
        all_qh.extend(res["fold_qh"])
        bB_gains.append(res["bB_gain_mean"])
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "per_fold_mean_clip": round(res["per_fold_mean_clip"], 4),
            "per_fold_std_clip": round(res["per_fold_std_clip"], 4),
            "per_fold_mean_corr": round(res["per_fold_mean_corr"], 4),
            "pooled_clip": round(res["pooled_clip"], 4),
            "pooled_corr": round(res["pooled_corr"], 4),
            "fold_ql": [round(v, 3) for v in res["fold_ql"]],
            "fold_qh": [round(v, 3) for v in res["fold_qh"]],
            "bB_gain_mean": round(res["bB_gain_mean"], 4),
        })
        print(f"   kf={s}: per_fold_mean(clip)={res['per_fold_mean_clip']:.4f}  "
              f"pooled(clip)={res['pooled_clip']:.4f}  "
              f"per_fold_mean(corr)={res['per_fold_mean_corr']:.4f}  "
              f"bB_gain={res['bB_gain_mean']:.3f}  wall={time.time()-ts:.1f}s")

    arr = np.asarray(per_fold_means_clip, dtype=np.float64)  # GATE metric
    pooled_arr = np.asarray(pooled_clips, dtype=np.float64)
    n_s = len(arr)
    per_fold_mean = float(arr.mean())     # <- gated quantity
    per_fold_std = float(arr.std(ddof=1)) if n_s > 1 else 0.0
    sem = per_fold_std / np.sqrt(n_s) if n_s > 1 else 0.0
    t_mult = 2.145  # df=14 two-sided 95%
    ci_low = per_fold_mean - t_mult * sem
    ci_high = per_fold_mean + t_mult * sem
    pooled_mean = float(pooled_arr.mean())
    pooled_std = float(pooled_arr.std(ddof=1)) if n_s > 1 else 0.0

    ql_counter = Counter(all_ql)
    qh_counter = Counter(all_qh)
    ql_mode = ql_counter.most_common(1)[0][0]
    qh_mode = qh_counter.most_common(1)[0][0]

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds)")
    print("-" * 78)
    print(f"   per_fold_mean (GATE)  = {per_fold_mean:.4f}")
    print(f"   per_fold_std          = {per_fold_std:.4f}")
    print(f"   95% CI                = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   pooled mean (clip)    = {pooled_mean:.4f} +/- {pooled_std:.4f}")
    print(f"   min/max per_fold_mean = [{arr.min():.4f}, {arr.max():.4f}]")
    print(f"   head-B gain in head A = {np.mean(bB_gains):.3f} "
          f"(mean feature-importance share)")
    print(f"   ql_dist (75 folds) = {dict(ql_counter)}  mode={ql_mode}")
    print(f"   qh_dist (75 folds) = {dict(qh_counter)}  mode={qh_mode}")
    print(f"\n   ref chemprop_aux anchor = {REF_CHEMPROP_AUX:.4f}  "
          f"(delta {per_fold_mean - REF_CHEMPROP_AUX:+.4f})")
    print(f"   ref K18 deep-30 OOF     = {REF_K18_DEEP30:.4f}  "
          f"(delta {per_fold_mean - REF_K18_DEEP30:+.4f})")
    print(f"   ref nb3030 K18 ceiling  = {REF_NB3030:.4f}  "
          f"(delta {per_fold_mean - REF_NB3030:+.4f})")

    # -- Median-seed OOF for storage -----------------------------------------
    med_seed_idx = int(np.argsort(arr)[n_s // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_clip_stack[med_seed_idx].astype(np.float32)
    oof_med_rae = float(rae(y_unb, oof_for_save))
    print(f"\n   median seed = {median_seed}  "
          f"(per_fold_mean={arr[med_seed_idx]:.4f}, pooled={oof_med_rae:.4f})")

    # -- Deploy on 513 -------------------------------------------------------
    # Head B: inner-CV OOF features on full 253 -> head A fit on full 253 ->
    # predict te. Head B + head A refit on FULL 253; learned clip from full 253.
    print("\n" + "-" * 78)
    print("DEPLOY: refit heads on full 253, predict 513, learned clip")
    print("-" * 78)
    # head-B OOF features for full 253 (inner CV, no leak into head A train)
    bB_oof_full = _fit_headB_oof_on_train(
        X_unb_K, rB, unb_scaffolds, seed=KF_SEEDS[0],
    )
    # head B refit on full 253 -> te head-B feature
    mdlB_dep = lgb.LGBMRegressor(**_lgbm_params(KF_SEEDS[0]))
    mdlB_dep.fit(X_unb_K, rB)
    bB_te = mdlB_dep.predict(X_te_K)
    # head A on full 253 with head-B OOF column
    XA_unb = np.column_stack([X_unb_K, bB_oof_full]).astype(np.float32)
    XA_te = np.column_stack([X_te_K, bB_te]).astype(np.float32)
    mdlA_dep = lgb.LGBMRegressor(**_lgbm_params(KF_SEEDS[0]))
    mdlA_dep.fit(XA_unb, rA)
    resid_te = mdlA_dep.predict(XA_te)
    corr_te = anchor_te + resid_te
    corr_unb_insample = anchor_unb + mdlA_dep.predict(XA_unb)
    # learned clip from full-253 corrected
    dep_ql, dep_qh, dep_lo, dep_hi = _pick_best_clip(y_unb, corr_unb_insample)
    te_pred = np.clip(corr_te, dep_lo, dep_hi).astype(np.float32)
    n_te_lo = int(np.sum(corr_te < dep_lo))
    n_te_hi = int(np.sum(corr_te > dep_hi))
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"   deploy clip = (q{dep_ql:.2f}, q{dep_qh:.2f}) -> "
          f"({dep_lo:.3f}, {dep_hi:.3f})")
    print(f"   te clipped: lo={n_te_lo}/513  hi={n_te_hi}/513")
    print(f"   te(513) mean={te_pred.mean():.3f}  std={te_pred.std():.3f}  "
          f"min={te_pred.min():.3f}  max={te_pred.max():.3f}")
    print(f"   te[unb] in-sample RAE = {te_unb_in_rae:.4f}  "
          f"(in-sample optimism vs per_fold_mean expected)")

    # -- Gate ----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    if per_fold_mean < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE-CANDIDATE. nb3333 K=18 multitask (counter head B stacked "
            f"into pEC50 head A) + learned clip per-fold-mean {per_fold_mean:.4f} "
            f"clears BETTER gate {GATE_BETTER:.4f}. Head-B counter-residual "
            f"feature carries {np.mean(bB_gains):.1%} of head-A split gain -- "
            f"the selectivity axis is a load-bearing regularizer, not noise. "
            f"Re-verify with deep-30 seeds before any PRIMARY swap; confirm no "
            f"leak (te[unb] {te_unb_in_rae:.4f} is in-sample, expected < CV)."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3333 per-fold-mean {per_fold_mean:.4f} fails BETTER gate "
            f"{GATE_BETTER:.4f} ({per_fold_mean - GATE_BETTER:+.4f}). Multitask "
            f"counter-head soft-share through stacked OOF does not beat the "
            f"single-head K=18 ceiling; head-B gain {np.mean(bB_gains):.1%}. "
            f"Counter axis adds no orthogonal pEC50 signal at n=253 beyond what "
            f"the K=18 SHAP block already encodes. Keep current PRIMARY."
        )
    print(f"   verdict       = {verdict}")
    print(f"   per_fold_mean = {per_fold_mean:.4f}  gate = {GATE_BETTER:.4f}")
    print(f"   ladder_action = {ladder_action}")

    # -- Save ----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("SAVE")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_for_save)
    np.save(te_path, te_pred)
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_multitask_counter_clip.csv"
    if verdict == "BETTER":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": te_pred,
        }).to_csv(sub_csv, index=False)
        print(f"   [save] {sub_csv}")
    else:
        print(f"   [skip] verdict={verdict}; no submission CSV written")

    summary = {
        "tag": TAG,
        "method": (
            "K18_multitask_pEC50_headA_plus_counter_headB_stacked_oof_feature_"
            "plus_per_fold_learned_clip"
        ),
        "paradigm": (
            "two LGBM heads share K=18 SHAP features; head B (counter residual) "
            "OOF stacked as 19th feature into head A (pEC50 residual); head B "
            "as regularizer/feature-enricher; deploy head A + learned clip"
        ),
        "anchor": "chemprop_aux",
        "anchor_pre_unblind": True,
        "anchor_te_path": str(ANCHOR_TE_PATH),
        "counter_anchor": "nb2490_counter_clean",
        "counter_te_path": str(COUNTER_TE_PATH),
        "counter_pre_unblind": True,
        "K": K,
        "shap_topK_idx_in_117": topK_idx.tolist(),
        "rae_anchor_chemprop_aux": round(rae_anchor, 4),
        "rae_counter_clean": round(rae_counter, 4),
        "residA_mean": float(rA.mean()),
        "residA_std": float(rA.std()),
        "residB_mean": float(rB.mean()),
        "residB_std": float(rB.std()),
        "corr_rA_rB": float(np.corrcoef(rA, rB)[0, 1]),
        "lgbm_params": {
            "objective": "regression", "max_depth": 4, "num_leaves": 15,
            "n_estimators": 300, "learning_rate": 0.03,
            "min_child_samples": 5, "reg_lambda": 2.0,
        },
        "n_folds": N_FOLDS,
        "inner_folds": INNER_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": n_s,
        "q_low_grid": Q_LOW_GRID,
        "q_high_grid": Q_HIGH_GRID,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "seed_records": seed_records,
        "per_fold_mean_array": [round(float(v), 4) for v in per_fold_means_clip],
        "pooled_clip_array": [round(float(v), 4) for v in pooled_clips],
        # GATE quantity
        "per_fold_mean": round(per_fold_mean, 4),
        "per_fold_std": round(per_fold_std, 4),
        "sem": round(sem, 4),
        "ci95_low": round(ci_low, 4),
        "ci95_high": round(ci_high, 4),
        "per_fold_min": round(float(arr.min()), 4),
        "per_fold_max": round(float(arr.max()), 4),
        # pooled diagnostic
        "pooled_mean_clip": round(pooled_mean, 4),
        "pooled_std_clip": round(pooled_std, 4),
        "headB_gain_share_mean": round(float(np.mean(bB_gains)), 4),
        "headB_gain_share_per_seed": [round(float(v), 4) for v in bB_gains],
        "ql_distribution": {str(k): int(v) for k, v in ql_counter.items()},
        "qh_distribution": {str(k): int(v) for k, v in qh_counter.items()},
        "ql_mode": float(ql_mode),
        "qh_mode": float(qh_mode),
        "ref_chemprop_aux": REF_CHEMPROP_AUX,
        "ref_counter": REF_COUNTER,
        "ref_K18_deep30": REF_K18_DEEP30,
        "ref_nb3030": REF_NB3030,
        "ref_nb2171": REF_NB2171,
        "delta_vs_chemprop_aux": round(per_fold_mean - REF_CHEMPROP_AUX, 4),
        "delta_vs_K18_deep30": round(per_fold_mean - REF_K18_DEEP30, 4),
        "delta_vs_nb3030": round(per_fold_mean - REF_NB3030, 4),
        # deploy
        "deploy_ql": float(dep_ql),
        "deploy_qh": float(dep_qh),
        "deploy_lo": round(dep_lo, 4),
        "deploy_hi": round(dep_hi, 4),
        "n_te_clipped_lo": n_te_lo,
        "n_te_clipped_hi": n_te_hi,
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "te_min": float(te_pred.min()),
        "te_max": float(te_pred.max()),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "median_seed": int(median_seed),
        "median_seed_oof_rae": round(oof_med_rae, 4),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict == "BETTER" else None,
        "gate_better": GATE_BETTER,
        "verdict": verdict,
        "ladder_action": ladder_action,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   per_fold_mean (GATE)  = {per_fold_mean:.4f} +/- {per_fold_std:.4f}")
    print(f"   95% CI                = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   pooled mean (clip)    = {pooled_mean:.4f}")
    print(f"   head-B gain share     = {np.mean(bB_gains):.3f}")
    print(f"   delta vs gate         = {per_fold_mean - GATE_BETTER:+.4f}")
    print(f"   verdict               = {verdict}")
    print(f"   wall                  = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "per_fold_mean", "per_fold_std", "ci95_low", "ci95_high",
        "pooled_mean_clip", "headB_gain_share_mean",
        "delta_vs_chemprop_aux", "delta_vs_K18_deep30", "delta_vs_nb3030",
        "ql_mode", "qh_mode", "deploy_ql", "deploy_qh", "deploy_lo", "deploy_hi",
        "n_te_clipped_lo", "n_te_clipped_hi", "te_unb_in_sample_rae",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
