"""nb2862 -- Nested OOF cross-validation for meta-stack on the 253 unblind.

NEW PARADIGM:
    Outer 5-fold scaffold CV; within each outer fold, inner 3-fold OOF for
    K=20 base learners (different LGBM seeds on chemprop_aux residual on the
    RFE K=20 feature slice) + chemprop_aux residual learner = K=20 base
    columns; meta-learner = LGBM stack on inner OOFs evaluated on the
    outer-val fold.  5 kf_seeds for the outer split.

WHY THIS IS DIFFERENT vs nb2103/nb2240/nb2270 flat OOF:
    Flat 5-fold OOF on K=20 chemprop_aux residual gives biased meta-learner
    training data because the base predictions on outer-val rows used the
    same training labels as the meta-learner fit on outer-train rows.
    Nested OOF cuts this dependency: the meta-learner sees inner-OOF
    predictions on outer-train that were never trained on outer-val, AND
    base predictions on outer-val that were trained ONLY on outer-train.
    This matches the deploy regime (meta refit on 253, base refit on 253,
    predict on 513) without leaking outer-val labels into meta training.

PROTOCOL:
    Outer loop:
        For each outer kf_seed in {1001..1005}:
            5-fold scaffold split on the 253 unblind.
            For each outer fold (tr_outer, va_outer):
                Inner loop:
                    For each base learner k in {0..K-1} (= 20 LGBM seeds):
                        3-fold KFold(seed=base_seed) split on tr_outer
                        Build inner_oof_tr[k] on tr_outer (OOF over inner)
                        Refit base on full tr_outer -> predict on va_outer
                Meta-train:
                    LGBM meta on inner_oof_tr (n_tr_outer, K) -> y[tr_outer]
                Meta-eval:
                    Predict on base_va_pred (n_va_outer, K) -> meta_pred_va
                Store meta_pred_va into outer_oof[va_outer]
        Pooled outer_oof RAE per kf_seed.
        Aggregate: mean +/- std across 5 kf_seeds.

BASE LEARNERS (K=20):
    All chemprop_aux residual LGBMs on the same K=20 feature slice (nb2231
    RFE survivors); only the LGBM random_state varies across the 20 base
    learners -- this is the "K=20 base learners" interpretation that
    matches existing nb2240 mean-bag scaffolding.
    Hyperparams identical to nb2240: max_depth=4, num_leaves=15,
    n_estimators=300, learning_rate=0.03, reg_lambda=2.0.
    The 20 base seeds are 0..19 (one per base).
    Inner cross-fit uses KFold(n=3, shuffle=True, random_state=base_seed)
    on tr_outer.

META-LEARNER:
    LGBM stack on K=20 base columns.
    max_depth=3, num_leaves=7, n_estimators=200, learning_rate=0.05,
    reg_lambda=2.0, min_child_samples=5.
    Random state 42.

DEPLOY:
    Base refit on full 253 -> 513 predictions = (513, K=20).
    Meta refit on (253, K=20) inner-OOF on full 253 (KFold 3-fold per base
    seed, same as outer-train inner) -> 513 meta predictions.

GATE:
    mean_rae across 5 outer kf_seeds:
      < 0.4570 -> "PROMOTE"
      < 0.4598 -> "MARGINAL_BEAT"
      else      -> "FAIL"

Outputs:
    scripts/nb2862_nested_oof_cv.py
    data/processed/nb2862_summary.json
    data/processed/nb2862_pred_oof.npy            (253,) float32
    data/processed/te_nb2862.npy                  (513,) float32
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
import lightgbm as lgb
from rdkit import Chem
from rdkit import RDLogger
from sklearn.model_selection import KFold

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2862"

# ---------------------------- nested CV config -------------------------------
N_OUTER_FOLDS = 5
OUTER_KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
N_INNER_FOLDS = 3
N_BASE = 20
BASE_SEEDS = list(range(N_BASE))   # 0..19

# ---------------------------- base LGBM hyperparams --------------------------
def base_lgbm_params(seed):
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


# ---------------------------- meta LGBM hyperparams --------------------------
META_SEED = 42

def meta_lgbm_params():
    return dict(
        objective="regression",
        max_depth=3,
        num_leaves=7,
        n_estimators=200,
        learning_rate=0.05,
        min_child_samples=5,
        reg_lambda=2.0,
        random_state=META_SEED,
        n_jobs=2,
        verbosity=-1,
    )


# ---------------------------- gates ------------------------------------------
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

LB_W_OOF = 0.51
LB_W_TE = 0.49

# ---------------------------- paths ------------------------------------------
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
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

CHEMPROP_AUX_REF = 0.6216
NB2240_PYRAMID_REF = 0.4598


# ============================================================================
# helpers -- copied / specialised from nb2240
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


def _load_chembl_pool():
    import pandas as pd
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


def _build_K20_matrix_on_test(n_test, te_smiles, surviving_K20):
    """Rebuild the 117-col 5-way feature matrix on the 513 test compounds,
    slice to the 20 RFE survivors and return (n_test, 20) float32.
    Mirrors nb2240 exactly so the base learners see the same K=20 view.
    """
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
    rec_mord = _extract_best_K_record(
        sum_1523, "per_K_records", best_K_key="best_K"
    )
    top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)
    full_ap_ranked = _extract_atompair_top_idx_from_nb1484(sum_1484)
    K_AP_best = int(sum_1524["best_K"])
    top_ap_bit_idx = full_ap_ranked[:K_AP_best]
    K_Embed_best = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed_col_idx = top_embed_full[:K_Embed_best]
    top_avalon_bit_idx = np.array(
        sum_1392["top_avalon_bit_indices_ranked"], dtype=int
    )

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
    std_test_smiles = [
        Chem.MolToSmiles(m) if m is not None else "" for m in test_mols
    ]
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
    feat_dim_full = X_te_full.shape[1]
    assert feat_dim_full == 117, f"feat_dim {feat_dim_full} != 117"
    X_te_K20 = X_te_full[:, surviving_K20].astype(np.float32)
    return X_te_K20


# ============================================================================
# nested CV core
# ============================================================================

def _build_inner_oof_one_base(
    X_tr_outer: np.ndarray,
    residual_tr_outer: np.ndarray,
    anchor_tr_outer: np.ndarray,
    X_va_outer: np.ndarray,
    anchor_va_outer: np.ndarray,
    base_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """For ONE base learner:
      - 3-fold inner KFold cross-fit on tr_outer -> inner OOF residual
        on tr_outer, then add anchor -> inner_pred_tr_outer (n_tr_outer,).
      - Refit on full tr_outer, predict residual on va_outer -> add anchor
        -> base_pred_va_outer (n_va_outer,).

    Returns (inner_pred_tr_outer, base_pred_va_outer).
    """
    n_tr = X_tr_outer.shape[0]
    kf = KFold(
        n_splits=N_INNER_FOLDS, shuffle=True, random_state=base_seed,
    )
    inner_oof_resid_tr = np.full(n_tr, np.nan, dtype=np.float64)
    for tr_in, va_in in kf.split(np.arange(n_tr)):
        mdl_in = lgb.LGBMRegressor(**base_lgbm_params(base_seed))
        mdl_in.fit(X_tr_outer[tr_in], residual_tr_outer[tr_in])
        inner_oof_resid_tr[va_in] = mdl_in.predict(X_tr_outer[va_in])
    inner_pred_tr = anchor_tr_outer + inner_oof_resid_tr

    mdl_full = lgb.LGBMRegressor(**base_lgbm_params(base_seed))
    mdl_full.fit(X_tr_outer, residual_tr_outer)
    base_pred_va = anchor_va_outer + mdl_full.predict(X_va_outer)

    return inner_pred_tr, base_pred_va


def _run_one_outer_seed(
    X_unb_K20: np.ndarray,
    anchor_unb: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list,
    kf_seed: int,
) -> tuple[float, np.ndarray, list]:
    """Run one outer scaffold 5-fold CV with nested inner OOF + meta-stack.

    Returns (pooled_rae, outer_oof, per_fold_records).
    """
    n_unb = X_unb_K20.shape[0]
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_OUTER_FOLDS, shuffle=True, seed=kf_seed,
    )
    outer_oof = np.full(n_unb, np.nan, dtype=np.float64)
    per_fold = []
    residual_unb = y_unb - anchor_unb

    for fold_id, (tr_outer, va_outer) in enumerate(splits):
        n_tr_outer = len(tr_outer)
        n_va_outer = len(va_outer)
        X_tr = X_unb_K20[tr_outer]
        X_va = X_unb_K20[va_outer]
        anchor_tr = anchor_unb[tr_outer]
        anchor_va = anchor_unb[va_outer]
        residual_tr = residual_unb[tr_outer]
        y_tr = y_unb[tr_outer]

        # Build (n_tr_outer, K) inner OOF preds + (n_va_outer, K) base preds
        Z_tr = np.zeros((n_tr_outer, N_BASE), dtype=np.float64)
        Z_va = np.zeros((n_va_outer, N_BASE), dtype=np.float64)
        for k, base_seed in enumerate(BASE_SEEDS):
            inner_pred_tr, base_pred_va = _build_inner_oof_one_base(
                X_tr, residual_tr, anchor_tr, X_va, anchor_va, base_seed,
            )
            Z_tr[:, k] = inner_pred_tr
            Z_va[:, k] = base_pred_va

        # Meta-LGBM stack on (Z_tr, y_tr) -> predict Z_va
        meta = lgb.LGBMRegressor(**meta_lgbm_params())
        meta.fit(Z_tr, y_tr)
        meta_pred_va = meta.predict(Z_va).astype(np.float64)
        outer_oof[va_outer] = meta_pred_va

        fold_rae = float(rae(y_unb[va_outer], meta_pred_va))
        per_fold.append({
            "fold_id": int(fold_id),
            "n_tr_outer": int(n_tr_outer),
            "n_va_outer": int(n_va_outer),
            "fold_rae": fold_rae,
            "Z_tr_mean": float(Z_tr.mean()),
            "Z_tr_std":  float(Z_tr.std()),
        })

    pooled_rae = float(rae(y_unb, outer_oof))
    return pooled_rae, outer_oof, per_fold


def _build_deploy(
    X_unb_K20: np.ndarray,
    X_te_K20: np.ndarray,
    anchor_unb: np.ndarray,
    anchor_te: np.ndarray,
    y_unb: np.ndarray,
) -> np.ndarray:
    """Deploy refit on the full 253:
      Inner OOF (3-fold) on the full 253 per base seed -> (253, K=20).
      Each base refit on full 253 -> 513 preds = (513, K=20).
      Meta fit on (253, K=20) inner OOF -> y_unb.
      Meta predict on (513, K=20) -> deploy te (513,).
    """
    n_unb = X_unb_K20.shape[0]
    n_te = X_te_K20.shape[0]
    residual_unb = y_unb - anchor_unb

    Z_unb_full = np.zeros((n_unb, N_BASE), dtype=np.float64)
    Z_te_full = np.zeros((n_te, N_BASE), dtype=np.float64)
    for k, base_seed in enumerate(BASE_SEEDS):
        kf = KFold(
            n_splits=N_INNER_FOLDS, shuffle=True, random_state=base_seed,
        )
        inner_oof_resid = np.full(n_unb, np.nan, dtype=np.float64)
        for tr_in, va_in in kf.split(np.arange(n_unb)):
            mdl_in = lgb.LGBMRegressor(**base_lgbm_params(base_seed))
            mdl_in.fit(X_unb_K20[tr_in], residual_unb[tr_in])
            inner_oof_resid[va_in] = mdl_in.predict(X_unb_K20[va_in])
        Z_unb_full[:, k] = anchor_unb + inner_oof_resid

        mdl_full = lgb.LGBMRegressor(**base_lgbm_params(base_seed))
        mdl_full.fit(X_unb_K20, residual_unb)
        Z_te_full[:, k] = anchor_te + mdl_full.predict(X_te_K20)

    meta = lgb.LGBMRegressor(**meta_lgbm_params())
    meta.fit(Z_unb_full, y_unb)
    deploy_te = meta.predict(Z_te_full).astype(np.float32)
    deploy_unb_in = meta.predict(Z_unb_full).astype(np.float32)
    return deploy_te, deploy_unb_in, Z_unb_full, Z_te_full


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Nested OOF CV (outer {N_OUTER_FOLDS}-fold scaffold x "
          f"inner {N_INNER_FOLDS}-fold) for meta-stack")
    print(f"          K={N_BASE} LGBM base learners on chemprop_aux residual "
          f"(K=20 RFE features)")
    print(f"          meta = LGBM stack on inner OOFs")
    print(f"          outer kf_seeds = {OUTER_KF_SEEDS}")
    print("=" * 78)

    # ---- Load K=20 surviving indices ----
    if not NB2231_SUMMARY.exists():
        raise FileNotFoundError(f"missing {NB2231_SUMMARY}")
    with open(NB2231_SUMMARY) as f:
        nb2231 = json.load(f)
    surviving_K20 = list(nb2231["snapshots"]["20"]["surviving_idx_in_117"])
    surviving_K20_names = list(nb2231["snapshots"]["20"]["surviving_names"])
    assert len(surviving_K20) == 20, (
        f"expected 20 features, got {len(surviving_K20)}"
    )
    print(f"[load] K=20 RFE survivors loaded ({len(surviving_K20)} cols)")

    # ---- Load truth + anchor ----
    te = load_test()
    n_test = len(te)
    te_smiles = (
        te["smiles"].astype(str).tolist()
        if "smiles" in te.columns
        else te["SMILES"].astype(str).tolist()
    )
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
    print(f"[load] chemprop_aux in_RAE = {rae_anchor:.4f} "
          f"(ref {CHEMPROP_AUX_REF:.4f})")

    # ---- Build K=20 feature matrix on 513 ----
    print("\n[feat] rebuilding 117-col 5-way feature matrix and slicing to K=20")
    X_te_K20 = _build_K20_matrix_on_test(n_test, te_smiles, surviving_K20)
    X_unb_K20 = X_te_K20[unb_idx]
    print(f"[feat] X_unb_K20 = {X_unb_K20.shape}  X_te_K20 = {X_te_K20.shape}")

    # ---- Nested OOF CV across 5 outer kf_seeds ----
    print("\n" + "=" * 78)
    print(f"NESTED OOF CV  outer kf_seeds={OUTER_KF_SEEDS}  "
          f"inner {N_INNER_FOLDS}-fold per base  K={N_BASE} base seeds")
    print("=" * 78)
    per_seed_records = []
    all_outer_oofs = []
    for kf_seed in OUTER_KF_SEEDS:
        ts = time.time()
        pooled_rae, outer_oof, per_fold = _run_one_outer_seed(
            X_unb_K20, anchor_unb, y_unb, unb_scaffolds, kf_seed,
        )
        wall = time.time() - ts
        per_seed_records.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": float(pooled_rae),
            "per_fold": per_fold,
            "wall_sec": round(wall, 2),
        })
        all_outer_oofs.append(outer_oof)
        fold_raes = [r["fold_rae"] for r in per_fold]
        print(f"   seed={kf_seed}  pooled_RAE={pooled_rae:.4f}  "
              f"fold_RAEs={[f'{r:.4f}' for r in fold_raes]}  "
              f"wall={wall:.1f}s")

    rae_arr = np.asarray([r["pooled_rae"] for r in per_seed_records])
    mean_rae = float(rae_arr.mean())
    std_rae = float(rae_arr.std())
    min_rae = float(rae_arr.min())
    max_rae = float(rae_arr.max())
    print(f"\n[nested-cv] mean_rae = {mean_rae:.4f}  std = {std_rae:.4f}  "
          f"range = [{min_rae:.4f}, {max_rae:.4f}]")

    # Mean across seeds of outer OOFs = the seed-averaged OOF prediction
    pred_oof = np.mean(np.column_stack(all_outer_oofs), axis=1).astype(
        np.float32
    )
    rae_pred_oof = float(rae(y_unb, pred_oof))
    print(f"[nested-cv] RAE of mean-of-seed OOFs = {rae_pred_oof:.4f}")

    # ---- Deploy refit on the 253 ----
    print("\n" + "-" * 78)
    print("DEPLOY (base refit on 253 -> Z_unb full inner-OOF + Z_te full "
          "refit; meta refit on Z_unb -> 513)")
    print("-" * 78)
    deploy_te, deploy_unb_in, Z_unb_full, Z_te_full = _build_deploy(
        X_unb_K20, X_te_K20, anchor_unb, te_anchor_513, y_unb,
    )
    in_rae_final = float(rae(y_unb, deploy_unb_in))
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))
    print(f"   in-sample RAE (253)   = {in_rae_final:.4f}  (overfit bound)")
    print(f"   te[unb_idx] RAE       = {te_unb_rae:.4f}  (in-sample on 253)")
    print(f"   te(513) mean/std      = {deploy_te.mean():.3f}/"
          f"{deploy_te.std():.3f}")

    lb_band_est = LB_W_OOF * mean_rae + LB_W_TE * te_unb_rae
    lb_low = lb_band_est - 0.05
    lb_high = lb_band_est + 0.05
    print(f"\n[LB-band] {LB_W_OOF:.2f}*OOF({mean_rae:.4f}) + "
          f"{LB_W_TE:.2f}*te_unb({te_unb_rae:.4f}) = {lb_band_est:.4f}  "
          f"[{lb_low:.4f}, {lb_high:.4f}]")

    # ---- Gate ----
    delta_vs_promote = mean_rae - GATE_PROMOTE
    delta_vs_marginal = mean_rae - GATE_MARGINAL
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "-" * 78)
    print(f"GATE   mean_rae={mean_rae:.4f}  "
          f"promote={GATE_PROMOTE:.4f}  marginal={GATE_MARGINAL:.4f}")
    print(f"   delta vs PROMOTE  = {delta_vs_promote:+.4f}")
    print(f"   delta vs MARGINAL = {delta_vs_marginal:+.4f}")
    print(f"   verdict           = {verdict}")
    print("-" * 78)

    # ---- Save artefacts ----
    pred_oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(pred_oof_path, pred_oof)
    np.save(te_path, deploy_te)
    print(f"\n[save] {pred_oof_path}")
    print(f"[save] {te_path}")

    summary = {
        "tag": TAG,
        "method": (
            "nested_oof_cv_outer5fold_scaffold_inner3fold_K20_base_LGBM_"
            "meta_LGBM_stack_chemprop_aux_residual"
        ),
        "n_outer_folds": N_OUTER_FOLDS,
        "outer_kf_seeds": OUTER_KF_SEEDS,
        "n_inner_folds": N_INNER_FOLDS,
        "n_base": N_BASE,
        "base_seeds": BASE_SEEDS,
        "base_lgbm_params": base_lgbm_params(0),
        "meta_lgbm_params": meta_lgbm_params(),
        "anchor": "chemprop_aux",
        "anchor_te_path": str(ANCHOR_TE_PATH),
        "feat_set": "RFE_K20_from_nb2231",
        "K_surviving_idx_in_117": [int(j) for j in surviving_K20],
        "K_surviving_names": surviving_K20_names,
        "n_test": n_test,
        "n_unb": n_unb,
        "n_unique_scaffolds": n_unique_scaf,
        "rae_anchor_chemprop_aux": rae_anchor,
        "rae_chemprop_aux_ref": CHEMPROP_AUX_REF,
        "per_seed_records": per_seed_records,
        "mean_rae": mean_rae,
        "std_rae": std_rae,
        "min_rae": min_rae,
        "max_rae": max_rae,
        "rae_pred_oof_mean_of_seeds": rae_pred_oof,
        "in_sample_rae_overfit_bound": in_rae_final,
        "te_unb_rae_in_sample": te_unb_rae,
        "lb_band_estimate": lb_band_est,
        "lb_band_low": lb_low,
        "lb_band_high": lb_high,
        "lb_band_w_oof": LB_W_OOF,
        "lb_band_w_te": LB_W_TE,
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "delta_vs_promote": delta_vs_promote,
        "delta_vs_marginal": delta_vs_marginal,
        "verdict": verdict,
        "compare_nb2240_pyramid_ref": NB2240_PYRAMID_REF,
        "delta_vs_nb2240_pyramid": mean_rae - NB2240_PYRAMID_REF,
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "pred_oof_path": str(pred_oof_path),
        "te_npy_path": str(te_path),
        "pre_unblind_clean": False,
        "wall_sec": round(time.time() - t0, 2),
    }
    summary_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {summary_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   nested OOF mean_rae (5 outer seeds) = {mean_rae:.4f} "
          f"+/- {std_rae:.4f}")
    print(f"   range                               = "
          f"[{min_rae:.4f}, {max_rae:.4f}]")
    print(f"   RAE of mean-of-seed OOFs            = {rae_pred_oof:.4f}")
    print(f"   delta vs PROMOTE ({GATE_PROMOTE:.4f})         = "
          f"{delta_vs_promote:+.4f}")
    print(f"   delta vs MARGINAL ({GATE_MARGINAL:.4f})        = "
          f"{delta_vs_marginal:+.4f}")
    print(f"   verdict                             = {verdict}")
    print(f"   LB-band estimate                    = {lb_band_est:.4f}")
    print(f"   in-sample RAE (overfit bound)       = {in_rae_final:.4f}")
    print(f"   wall                                = "
          f"{time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "rae_anchor_chemprop_aux",
        "mean_rae",
        "std_rae",
        "min_rae",
        "max_rae",
        "rae_pred_oof_mean_of_seeds",
        "delta_vs_promote",
        "delta_vs_marginal",
        "verdict",
        "lb_band_estimate",
        "in_sample_rae_overfit_bound",
        "te_unb_rae_in_sample",
        "pred_oof_path",
        "te_npy_path",
        "wall_sec",
    ):
        print(f"  {k}: {res.get(k)}")
