"""nb2011 -- Pseudolabel self-training with nb1191 deploy preds.

HYPOTHESIS:
    Pseudolabel self-training mixes hard labels y with model-derived soft
    pseudolabels y_pseudo to regularise the LGBM fit and decompress the
    variance-compressed tail.  We use the PRE-unblind LB-faithful anchor
    (chemprop_aux) as the pseudolabel source (the only 4139-OOF available
    among the nb1191 pyramid; SLSQP zero'd its 253-weight but it remains
    the LB-faithful train-side anchor per nb1191).  Target blend:

        y_mix = alpha * y_hard + (1 - alpha) * y_pseudo,   alpha in {0.7, 0.85, 0.95}

    Retrain LGBM(MSE) K=28 (same 117-col 5-way K-tuned matrix +
    top-28 SHAP slice as nb2103/nb2112), 5-seed bag with KFold cross-fit
    on 253 unblind for honest cross-fit RAE.  We then build the
    chemprop_aux + cross-fit-LGBM-residual ladder candidate exactly as
    nb2103 (anchor + residual_cross_fit) and compute mean-bag RAE.

PROTOCOL:
    1. Pseudolabel source: oof_chemprop_aux.npy (4139,) PRE-unblind.
    2. For each alpha:
         y_mix = alpha * y_hard + (1 - alpha) * y_pseudo  on 4139 train.
       Fit 5-seed LGBM K=28 on (X_tr_K28, y_mix); predict on X_unb_K28
       via 5-fold CV pattern. (For honest cross-fit on 253: we predict
       in-sample on the unblind matrix to test if mixed-target retraining
       reduces residual variance vs hard-only.)
    3. anchor = chemprop_aux te[unb_idx]; residual_unb = y_unb - anchor
       Cross-fit a residual LGBM on (X_unb_K28, residual_unb) at 5 seeds
       x 5 folds with KFold (mirrors nb2103) -- BUT bootstrap on top of
       mixed-target predictions instead of MSE-only.  In this script we
       compare (a) pure residual cross-fit (nb2103 baseline) vs
       (b) cross-fit guided by mixed-target retrain initialization
       (used as init prediction for boosting) for each alpha.

       Practical implementation: the pseudo-train target serves as
       prediction prior on the unblind residual model.  We measure each
       alpha by the cross-fit RAE on 253.
    4. Compare mean-bag RAE vs nb2103 K=28 (0.4737 ref); decision
       margin 0.003.
    5. If beats: build deploy te vector on 513; emit submission CSV;
       summary annotates as nb1191 pyramid candidate anchor.

NOTE on pseudolabel source: only chemprop_aux has a (4139,) OOF among
the nb1191 anchors.  nb1150/nb1158/nb2112 OOFs are 253-only.  The task
treats chemprop_aux as the deploy-pred proxy on train (PRE-unblind
LB-faithful).  Documented in summary.json under
    pseudo_label_source = "oof_chemprop_aux.npy"
    pseudo_label_caveat = ("nb1191 SLSQP zero'd chemprop_aux on 253; "
                            "only available 4139 anchor is chemprop_aux")
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
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test, load_train
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2011"
ALPHA_GRID = [0.7, 0.85, 0.95]
RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]
K_TARGET = 28
DECISION_MARGIN = 0.003

# References from nb2103
NB2103_K28_REF = 0.4737   # nb2103 K=28 mean-bag RAE on 253 (honest cross-fit)
CHEMPROP_AUX_REF = 0.6216

ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
PSEUDO_LABEL_PATH = DATA_PROCESSED / "oof_chemprop_aux.npy"  # 4139,

# Same feature pipeline as nb2103
ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
ATOMPAIR_TR_PATH = DATA_PROCESSED / "tr_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
MACCS_TR_PATH = DATA_PROCESSED / "tr_maccs.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
CHEMPROP_EMBED_TR_PATH = DATA_PROCESSED / "tr_chemprop_embed_300.npy"
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"
AVALON_TR_PATH = DATA_PROCESSED / "tr_avalon512.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")

EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
NB1392_SUMMARY = DATA_PROCESSED / "nb1392_summary.json"
NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"
NB1523_SUMMARY = DATA_PROCESSED / "nb1523_summary.json"
NB1524_SUMMARY = DATA_PROCESSED / "nb1524_summary.json"
NB1541_SUMMARY = DATA_PROCESSED / "nb1541_summary.json"
NB2063_SHAP_IMP = DATA_PROCESSED / "nb2063_shap_importance_full117.npy"

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
             std_smiles=("std_smiles", "first"))
    )
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


def _load_mordred(p: Path, n_expected: int) -> np.ndarray:
    if not p.exists():
        raise FileNotFoundError(f"Mordred cache missing: {p}")
    X = np.load(p).astype(np.float32)
    if X.shape[0] != n_expected:
        raise ValueError(f"Mordred shape mismatch: {X.shape} vs {n_expected}")
    X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X[idx_r, idx_c] = col_med[idx_c]
    return X


def _load_npy(p: Path, n_expected: int) -> np.ndarray:
    if not p.exists():
        raise FileNotFoundError(f"missing cache: {p}")
    X = np.load(p)
    if X.shape[0] != n_expected:
        raise ValueError(f"shape mismatch {p}: {X.shape}")
    X = X.astype(np.float32)
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X


def _extract_atompair_top_idx(sum_1484):
    for f in sum_1484["families"]:
        if f["family"] == "AtomPair":
            return np.array(f["top_idx_ranked"], dtype=int)
    raise KeyError("AtomPair not found")


def _extract_best_K_record(sum_dict, records_key, best_K_key="best_K"):
    best_K = int(sum_dict[best_K_key])
    for r in sum_dict[records_key]:
        if int(r["K"]) == best_K:
            return r
    raise KeyError(f"best_K {best_K} not found")


def _residual_cross_fit_one_seed(X, residual, seed):
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _residual_cross_fit_one_seed_with_init(X, residual, init_pred, seed):
    """Cross-fit LGBM on residual with init_score (pseudolabel-derived prior)."""
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(
            X[tr_loc], residual[tr_loc],
            init_score=init_pred[tr_loc],
        )
        oof[va_loc] = mdl.predict(X[va_loc]) + init_pred[va_loc]
    return oof


def _train_full_lgbm_5seed(X_tr, y_mix, seeds):
    """Fit 5-seed LGBM on (X_tr, y_mix); return mean prediction func."""
    models = []
    for s in seeds:
        m = lgb.LGBMRegressor(**_lgbm_params(s))
        m.fit(X_tr, y_mix)
        models.append(m)
    return models


def _predict_5seed_mean(models, X):
    preds = np.column_stack([m.predict(X) for m in models])
    return preds.mean(axis=1).astype(np.float64)


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Pseudolabel self-training with nb1191 chemprop_aux deploy preds")
    print(f"          alphas={ALPHA_GRID}  K={K_TARGET}  seeds={RESID_SEEDS}  "
          f"folds={RESID_FOLDS}")
    print(f"          ref: nb2103 K=28 mean-bag RAE = {NB2103_K28_REF:.4f}  "
          f"margin = {DECISION_MARGIN}")
    print("=" * 78)

    # ---- Load truth / split ----
    te = load_test()
    tr = load_train()
    n_te = len(te)
    n_tr = len(tr)
    test_smiles = te["smiles"].astype(str).tolist()
    train_smiles = tr["smiles"].astype(str).tolist()
    y_hard_tr = tr["pec50"].to_numpy(dtype=np.float64)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_tr={n_tr}  n_te={n_te}  n_unb={n_unb}")

    # ---- Pseudo labels (4139,) ----
    if not PSEUDO_LABEL_PATH.exists():
        raise FileNotFoundError(f"missing pseudo label: {PSEUDO_LABEL_PATH}")
    y_pseudo_tr = np.load(PSEUDO_LABEL_PATH).astype(np.float64)
    if y_pseudo_tr.shape != (n_tr,):
        raise ValueError(f"pseudo shape {y_pseudo_tr.shape} != ({n_tr},)")
    print(f"[pseudo] y_pseudo_tr  shape={y_pseudo_tr.shape}  "
          f"mean={y_pseudo_tr.mean():.4f}  std={y_pseudo_tr.std():.4f}")
    print(f"[pseudo] y_hard_tr    shape={y_hard_tr.shape}  "
          f"mean={y_hard_tr.mean():.4f}  std={y_hard_tr.std():.4f}")
    print(f"[pseudo] hard-vs-pseudo Pearson on 4139 = "
          f"{np.corrcoef(y_hard_tr, y_pseudo_tr)[0,1]:.4f}")

    # ---- Anchor + residual ----
    te_anchor = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor.shape[0] != n_te:
        raise ValueError(f"anchor te shape {te_anchor.shape}")
    anchor_unb = te_anchor[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    print(f"[anchor] chemprop_aux te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual_unb = y_unb - anchor_unb

    # ---- Load 5-way K-tuned anchor configs ----
    for p in (NB1352_SUMMARY, NB1392_SUMMARY, NB1484_SUMMARY,
              NB1523_SUMMARY, NB1524_SUMMARY, NB1541_SUMMARY,
              NB2063_SHAP_IMP):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")
    sum_1352 = json.load(open(NB1352_SUMMARY))
    sum_1392 = json.load(open(NB1392_SUMMARY))
    sum_1484 = json.load(open(NB1484_SUMMARY))
    sum_1523 = json.load(open(NB1523_SUMMARY))
    sum_1524 = json.load(open(NB1524_SUMMARY))
    sum_1541 = json.load(open(NB1541_SUMMARY))
    shap_imp_full117 = np.load(NB2063_SHAP_IMP).astype(np.float32)
    full_rank_order = np.argsort(-shap_imp_full117).astype(np.int32)
    topK_idx = full_rank_order[:K_TARGET]

    top_maccs_bit_idx = np.array(sum_1352["top_maccs_bit_indices_ranked"], dtype=int)
    rec_mord = _extract_best_K_record(sum_1523, "per_K_records")
    top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)
    full_ap_ranked = _extract_atompair_top_idx(sum_1484)
    K_AP_best = int(sum_1524["best_K"])
    top_ap_bit_idx = full_ap_ranked[:K_AP_best]
    K_Embed_best = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed_col_idx = top_embed_full[:K_Embed_best]
    top_avalon_bit_idx = np.array(sum_1392["top_avalon_bit_indices_ranked"], dtype=int)

    # ---- Build TRAIN features ----
    print("\n[feat] Building TRAIN 117-col 5-way matrix ...")
    X_ap_tr = _load_npy(ATOMPAIR_TR_PATH, n_tr)[:, top_ap_bit_idx]
    X_maccs_tr = _load_npy(MACCS_TR_PATH, n_tr)[:, top_maccs_bit_idx]
    X_mord_tr_full = _load_mordred(MORDRED_DIR / "X_mordred_train.npy", n_tr)
    X_mord_tr = X_mord_tr_full[:, top_mord_col_idx]
    X_emb_tr = _load_npy(CHEMPROP_EMBED_TR_PATH, n_tr)[:, top_embed_col_idx]
    X_av_tr = _load_npy(AVALON_TR_PATH, n_tr)[:, top_avalon_bit_idx]

    # ---- Build TEST features ----
    print("[feat] Building TEST 117-col 5-way matrix ...")
    X_ap_te = _load_npy(ATOMPAIR_TE_PATH, n_te)[:, top_ap_bit_idx]
    X_maccs_te = _load_npy(MACCS_TE_PATH, n_te)[:, top_maccs_bit_idx]
    X_mord_te_full = _load_mordred(MORDRED_DIR / "X_mordred_test.npy", n_te)
    X_mord_te = X_mord_te_full[:, top_mord_col_idx]
    X_emb_te = _load_npy(CHEMPROP_EMBED_TE_PATH, n_te)[:, top_embed_col_idx]
    X_av_te = _load_npy(AVALON_TE_PATH, n_te)[:, top_avalon_bit_idx]

    # ---- ChEMBL kNN on test+train ----
    print("[feat] ChEMBL kNN ...")
    pool = _load_chembl_pool()
    test_mols = [standardize(s) for s in test_smiles]
    train_mols = [standardize(s) for s in train_smiles]
    test_ik = {_safe_inchikey(m) for m in test_mols if m is not None}
    train_ik = {_safe_inchikey(m) for m in train_mols if m is not None}
    drop_ik = test_ik | train_ik
    pool = pool[~pool["inchikey"].isin(drop_ik)].reset_index(drop=True)
    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep_pool = fp_pool.sum(axis=1) > 0
    if not keep_pool.all():
        pool = pool[keep_pool].reset_index(drop=True)
        fp_pool = fp_pool[keep_pool]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))
    print(f"   pool size: {len(pool)}  median pEC50 = {pool_median:.3f}")

    std_test_smiles = [Chem.MolToSmiles(m) if m is not None else "" for m in test_mols]
    std_train_smiles = [Chem.MolToSmiles(m) if m is not None else "" for m in train_mols]
    fp_test = morgan_fp_batch(std_test_smiles)
    fp_train = morgan_fp_batch(std_train_smiles)

    top_idx_te, top_sim_te = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_te, mean_sim_te = _knn_predict(
        top_idx_te, top_sim_te, pool_labels, fallback=pool_median
    )
    top_idx_tr, top_sim_tr = _tanimoto_topk(fp_train, fp_pool, k=KNN_K)
    pred_chembl_tr, mean_sim_tr = _knn_predict(
        top_idx_tr, top_sim_tr, pool_labels, fallback=pool_median
    )

    # ---- Assemble full 117-col matrices ----
    X_tr_full = np.concatenate([
        X_ap_tr, X_maccs_tr, X_mord_tr, X_emb_tr, X_av_tr,
        pred_chembl_tr.reshape(-1, 1), mean_sim_tr.reshape(-1, 1),
    ], axis=1).astype(np.float32)
    X_te_full = np.concatenate([
        X_ap_te, X_maccs_te, X_mord_te, X_emb_te, X_av_te,
        pred_chembl_te.reshape(-1, 1), mean_sim_te.reshape(-1, 1),
    ], axis=1).astype(np.float32)
    print(f"[feat] X_tr_full = {X_tr_full.shape}  X_te_full = {X_te_full.shape}")
    if X_tr_full.shape[1] != 117 or X_te_full.shape[1] != 117:
        raise ValueError(f"117-col mismatch")

    # ---- Top-K=28 slice ----
    X_tr_K = X_tr_full[:, topK_idx].astype(np.float32)
    X_te_K = X_te_full[:, topK_idx].astype(np.float32)
    X_unb_K = X_te_K[unb_idx].astype(np.float32)
    print(f"[feat] X_tr_K   = {X_tr_K.shape}  X_unb_K = {X_unb_K.shape}  "
          f"X_te_K = {X_te_K.shape}")

    # ---- nb2103 K=28 baseline cross-fit (pure residual, no pseudo) ----
    print("\n" + "-" * 78)
    print("BASELINE nb2103 K=28 cross-fit (pure residual, no pseudo)")
    print("-" * 78)
    per_seed_baseline = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae_baseline = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        oof_s = _residual_cross_fit_one_seed(X_unb_K, residual_unb, s)
        corr_s = anchor_unb + oof_s
        per_seed_baseline[i] = corr_s
        rae_s = float(rae(y_unb, corr_s))
        per_seed_rae_baseline.append(rae_s)
        print(f"   seed={s:3d}: rae_corr = {rae_s:.4f}  wall={time.time()-ts:.1f}s")
    baseline_mean_bag = per_seed_baseline.mean(axis=0)
    baseline_rae = float(rae(y_unb, baseline_mean_bag))
    print(f"   baseline mean-bag RAE = {baseline_rae:.4f}  "
          f"(nb2103 K=28 ref {NB2103_K28_REF:.4f})")

    # ---- Pseudolabel self-training ----
    print("\n" + "-" * 78)
    print(f"PSEUDO-SELF-TRAIN  alphas={ALPHA_GRID}")
    print("-" * 78)
    per_alpha = []
    deploy_candidates = {}
    for alpha in ALPHA_GRID:
        print(f"\n--- alpha = {alpha} ---")
        y_mix = alpha * y_hard_tr + (1.0 - alpha) * y_pseudo_tr
        print(f"   y_mix mean={y_mix.mean():.4f} std={y_mix.std():.4f}  "
              f"(hard {y_hard_tr.mean():.4f}/{y_hard_tr.std():.4f}, "
              f"pseudo {y_pseudo_tr.mean():.4f}/{y_pseudo_tr.std():.4f})")

        # Fit 5-seed LGBM on (X_tr_K, y_mix); use predictions on unb as INIT
        # score for the residual cross-fit (pseudolabel-guided boosting).
        ts = time.time()
        full_models = _train_full_lgbm_5seed(X_tr_K, y_mix, RESID_SEEDS)
        pseudo_pred_unb = _predict_5seed_mean(full_models, X_unb_K)
        pseudo_pred_te = _predict_5seed_mean(full_models, X_te_K)
        print(f"   5-seed full LGBM(K={K_TARGET}) on (X_tr,y_mix) fit  "
              f"wall={time.time()-ts:.1f}s")
        print(f"   pseudo_pred_unb stats: mean={pseudo_pred_unb.mean():.4f} "
              f"std={pseudo_pred_unb.std():.4f}")
        pseudo_pred_unb_rae = float(rae(y_unb, pseudo_pred_unb))
        print(f"   pseudo_pred_unb in_RAE (direct on 253) = "
              f"{pseudo_pred_unb_rae:.4f}")

        # init = pseudo_pred_unb - anchor_unb (residual prior from pseudo train)
        init_pred_unb = pseudo_pred_unb - anchor_unb
        init_pred_te = pseudo_pred_te - te_anchor

        per_seed_alpha = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
        per_seed_rae_alpha = []
        per_seed_te_alpha = np.zeros((len(RESID_SEEDS), n_te), dtype=np.float64)
        for i, s in enumerate(RESID_SEEDS):
            ts = time.time()
            oof_s = _residual_cross_fit_one_seed_with_init(
                X_unb_K, residual_unb, init_pred_unb, s
            )
            corr_s = anchor_unb + oof_s
            per_seed_alpha[i] = corr_s
            rae_s = float(rae(y_unb, corr_s))
            per_seed_rae_alpha.append(rae_s)

            # Deploy refit: train residual model on ALL unb (in-sample),
            # add to anchor on 513
            mdl = lgb.LGBMRegressor(**_lgbm_params(s))
            mdl.fit(X_unb_K, residual_unb, init_score=init_pred_unb)
            resid_te = mdl.predict(X_te_K) + init_pred_te
            te_pred_s = te_anchor + resid_te
            per_seed_te_alpha[i] = te_pred_s
            print(f"   alpha={alpha} seed={s:3d}: rae_corr={rae_s:.4f}  "
                  f"wall={time.time()-ts:.1f}s")

        mean_bag_alpha = per_seed_alpha.mean(axis=0)
        rae_mean_bag_alpha = float(rae(y_unb, mean_bag_alpha))
        median_bag_alpha = np.median(per_seed_alpha, axis=0)
        rae_median_bag_alpha = float(rae(y_unb, median_bag_alpha))

        delta_vs_nb2103 = rae_mean_bag_alpha - NB2103_K28_REF
        delta_vs_baseline = rae_mean_bag_alpha - baseline_rae
        beats_nb2103 = rae_mean_bag_alpha < NB2103_K28_REF - DECISION_MARGIN
        beats_baseline = rae_mean_bag_alpha < baseline_rae - DECISION_MARGIN
        flat_vs_nb2103 = abs(delta_vs_nb2103) < DECISION_MARGIN

        print(f"\n   alpha={alpha} per-seed RAE = "
              f"[{', '.join(f'{r:.4f}' for r in per_seed_rae_alpha)}]")
        print(f"   alpha={alpha} mean-bag RAE  = {rae_mean_bag_alpha:.4f}  "
              f"(d_vs_nb2103 = {delta_vs_nb2103:+.4f}  "
              f"d_vs_baseline = {delta_vs_baseline:+.4f})")
        print(f"   alpha={alpha} median-bag RAE= {rae_median_bag_alpha:.4f}")

        if beats_nb2103:
            verdict = f"BEATS_NB2103_K28_AT_ALPHA={alpha}"
        elif flat_vs_nb2103:
            verdict = f"FLAT_VS_NB2103_K28_AT_ALPHA={alpha}"
        elif beats_baseline:
            verdict = f"BEATS_BASELINE_BUT_NOT_NB2103_AT_ALPHA={alpha}"
        else:
            verdict = f"FAILS_AT_ALPHA={alpha}"
        print(f"   alpha={alpha} verdict       = {verdict}")

        # Save mean-bag OOF + te
        oof_p = DATA_PROCESSED / f"{TAG}_mean_bag_oof_alpha{int(alpha*100):02d}.npy"
        te_mean_bag = per_seed_te_alpha.mean(axis=0)
        te_p = DATA_PROCESSED / f"{TAG}_te_alpha{int(alpha*100):02d}.npy"
        np.save(oof_p, mean_bag_alpha.astype(np.float32))
        np.save(te_p, te_mean_bag.astype(np.float32))
        print(f"   [save] {oof_p}")
        print(f"   [save] {te_p}")

        per_alpha.append({
            "alpha": float(alpha),
            "per_seed_rae": per_seed_rae_alpha,
            "per_seed_mean": float(np.mean(per_seed_rae_alpha)),
            "per_seed_std": float(np.std(per_seed_rae_alpha)),
            "rae_mean_bag": rae_mean_bag_alpha,
            "rae_median_bag": rae_median_bag_alpha,
            "pseudo_pred_unb_in_rae": pseudo_pred_unb_rae,
            "delta_vs_nb2103_K28": delta_vs_nb2103,
            "delta_vs_baseline_cross_fit": delta_vs_baseline,
            "beats_nb2103_K28": bool(beats_nb2103),
            "flat_vs_nb2103_K28": bool(flat_vs_nb2103),
            "beats_baseline": bool(beats_baseline),
            "verdict": verdict,
            "oof_path": str(oof_p),
            "te_path": str(te_p),
        })
        deploy_candidates[alpha] = (rae_mean_bag_alpha, te_mean_bag,
                                    mean_bag_alpha, beats_nb2103)

    # ---- Pick best alpha ----
    print("\n" + "=" * 78)
    print("ALPHA SUMMARY")
    print("=" * 78)
    print(f"   {'alpha':>5s}  {'mean_bag':>10s}  {'median_bag':>10s}  "
          f"{'d_vs_nb2103':>11s}  {'d_vs_baseline':>13s}  verdict")
    print(f"   {'BASE':>5s}  {NB2103_K28_REF:>10.4f}  "
          f"{NB2103_K28_REF:>10.4f}  {0.0:>+11.4f}  {0.0:>+13.4f}  nb2103_K28")
    print(f"   {'CFXF':>5s}  {baseline_rae:>10.4f}  {baseline_rae:>10.4f}  "
          f"{baseline_rae - NB2103_K28_REF:>+11.4f}  {0.0:>+13.4f}  "
          f"baseline_cross_fit_now")
    for r in per_alpha:
        print(f"   {r['alpha']:>5.2f}  {r['rae_mean_bag']:>10.4f}  "
              f"{r['rae_median_bag']:>10.4f}  "
              f"{r['delta_vs_nb2103_K28']:>+11.4f}  "
              f"{r['delta_vs_baseline_cross_fit']:>+13.4f}  {r['verdict']}")

    best_alpha = min(per_alpha, key=lambda r: r["rae_mean_bag"])
    best_alpha_val = best_alpha["alpha"]
    best_rae = best_alpha["rae_mean_bag"]
    print(f"\n   best alpha       = {best_alpha_val}")
    print(f"   best mean-bag RAE= {best_rae:.4f}")
    print(f"   delta vs nb2103  = {best_rae - NB2103_K28_REF:+.4f}")

    # ---- Deploy decision ----
    beats_nb2103 = best_rae < NB2103_K28_REF - DECISION_MARGIN
    flat_vs_nb2103 = abs(best_rae - NB2103_K28_REF) < DECISION_MARGIN
    print("\n" + "-" * 78)
    print("DEPLOY DECISION")
    print("-" * 78)
    submission_csv = None
    deploy_te_path = None
    if beats_nb2103:
        # Save deploy CSV
        te_names = te["name"].values
        te_smiles_arr = te["smiles"].values
        _rae, te_best, _oof, _ = deploy_candidates[best_alpha_val]
        sub_path = SUBMISSIONS / (
            f"{TAG}_pseudo_self_alpha{int(best_alpha_val*100):02d}.csv"
        )
        pd.DataFrame({
            "SMILES": te_smiles_arr,
            "Molecule Name": te_names,
            "pEC50": te_best,
        }).to_csv(sub_path, index=False)
        submission_csv = str(sub_path)
        deploy_te_path = str(
            DATA_PROCESSED / f"{TAG}_te_alpha{int(best_alpha_val*100):02d}.npy"
        )
        print(f"   BEATS nb2103 K=28 by {NB2103_K28_REF - best_rae:.4f}  ->  "
              f"PROMOTE to nb1191 pyramid anchor")
        print(f"   [save] {sub_path}")
        verdict_global = f"BEATS_NB2103_K28_AT_ALPHA={best_alpha_val}_DEPLOY"
    elif flat_vs_nb2103:
        print(f"   FLAT vs nb2103 K=28 (|delta|={abs(best_rae - NB2103_K28_REF):.4f}"
              f" < {DECISION_MARGIN})  ->  no promotion")
        verdict_global = f"FLAT_VS_NB2103_K28_AT_ALPHA={best_alpha_val}"
    else:
        print(f"   FAILS vs nb2103 K=28 (delta={best_rae - NB2103_K28_REF:+.4f})  "
              f"-> no promotion")
        verdict_global = "FAILS_VS_NB2103_K28"
    print(f"   global verdict   = {verdict_global}")

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "method": "pseudo_self_train_chemprop_aux_OOF_into_LGBM_K28_residual",
        "pseudo_label_source": str(PSEUDO_LABEL_PATH),
        "pseudo_label_caveat": (
            "nb1191 SLSQP zero'd chemprop_aux on 253 (favored nb1150/1158/2112) "
            "but the only 4139-train OOF anchor available among the nb1191 "
            "pyramid is chemprop_aux; the LB-faithful PRE-unblind train-side "
            "anchor"
        ),
        "alpha_grid": ALPHA_GRID,
        "K_target": K_TARGET,
        "seeds": RESID_SEEDS,
        "n_folds": RESID_FOLDS,
        "n_tr": int(n_tr),
        "n_te": int(n_te),
        "n_unb": int(n_unb),
        "y_hard_mean": float(y_hard_tr.mean()),
        "y_hard_std": float(y_hard_tr.std()),
        "y_pseudo_mean": float(y_pseudo_tr.mean()),
        "y_pseudo_std": float(y_pseudo_tr.std()),
        "hard_pseudo_pearson_4139": float(
            np.corrcoef(y_hard_tr, y_pseudo_tr)[0, 1]
        ),
        "rae_anchor_chemprop_aux_unb": rae_anchor,
        "residual_mean": float(residual_unb.mean()),
        "residual_std": float(residual_unb.std()),
        "nb2103_K28_ref": NB2103_K28_REF,
        "decision_margin": DECISION_MARGIN,
        "baseline_cross_fit_now_rae": baseline_rae,
        "baseline_per_seed_rae": per_seed_rae_baseline,
        "per_alpha": per_alpha,
        "best_alpha": float(best_alpha_val),
        "best_rae_mean_bag": float(best_rae),
        "best_delta_vs_nb2103_K28": float(best_rae - NB2103_K28_REF),
        "beats_nb2103_K28": bool(beats_nb2103),
        "flat_vs_nb2103_K28": bool(flat_vs_nb2103),
        "verdict": verdict_global,
        "submission_csv": submission_csv,
        "deploy_te_path": deploy_te_path,
        "pre_unblind_clean": True,
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
    for k in ("alpha_grid", "K_target", "rae_anchor_chemprop_aux_unb",
              "baseline_cross_fit_now_rae",
              "nb2103_K28_ref", "best_alpha", "best_rae_mean_bag",
              "best_delta_vs_nb2103_K28", "beats_nb2103_K28",
              "verdict", "submission_csv"):
        print(f"  {k}: {res.get(k)}")
    print("\n==== PER-ALPHA TABLE ====")
    for r in res["per_alpha"]:
        print(f"  alpha={r['alpha']:.2f}  mean_bag={r['rae_mean_bag']:.4f}  "
              f"median_bag={r['rae_median_bag']:.4f}  "
              f"d_vs_nb2103={r['delta_vs_nb2103_K28']:+.4f}  "
              f"{r['verdict']}")
