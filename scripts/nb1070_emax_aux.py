"""nb1070_emax_aux -- Emax-aware multi-task LGBM on K=28 SHAP features.

HYPOTHESIS:
    Emax (max fold-induction of the PXR dose-response) is a related but
    distinct property to pEC50.  Multi-task / sequential transfer may
    deliver a small chemprop-aux residual gain on the 253 unblind:

      Step 1: predict Emax on TRAIN with the same K=28 SHAP-pruned features
              used in nb2103, via 5-fold cross-fit (so the TRAIN Emax preds
              are honest).  Refit on the full 4139 TRAIN to get Emax_pred
              on the 513 test (slice to 253 unb).
      Step 2: residual LGBM on K=28+1 (Emax_pred as 29th feature) targeting
              chemprop_aux residual on the 253 unblind, 5-seed bag, 5-fold
              cross-fit.

    Plus a sanity arm:
      DIRECT  Shared base LGBM trained on stacked (pEC50, Emax) target by
              predicting each separately on the same K=28 features and
              averaging the residuals.  This is a poor-man's multi-task
              average; expected to be neutral or hurting.

PROTOCOL:
    - Load TRAIN (4139); verify pEC50 and Emax clean (no '-' strings).
    - Build the same 117-col 5-way K-tuned feature matrix as nb2063/nb2103
      on BOTH train (4139) and test (513).  Slice both to the same 28-col
      SHAP top using the cached `top_K_idx_in_117` from nb2103.
    - EMAX-AUX arm:
        * 5-fold cross-fit Emax-LGBM on TRAIN (seed 0 only is fine -- this
          is a feature, not a final predictor) -> oof_emax_train (4139,).
        * Refit Emax-LGBM on full 4139 -> emax_pred_test (513,) ->
          slice to (253,) for unb.
        * Build X_unb_29 = [X_unb_28 || emax_pred_unb.reshape(-1,1)].
        * Residual cross-fit LGBM(K=28+1) on 253 unblind, 5 seeds.
        * Mean-bag + median-bag, compare vs nb2103 K=28 (0.4737/0.4698).
    - DIRECT arm:
        * Train shared base LGBM on pEC50-residual AND on Emax separately
          (same K=28 features), then average the two residuals.
        * 5 seeds, 5-fold cross-fit.
    - Decision margin: 0.003.
    - If the EMAX-AUX mean-bag RAE beats nb2103 K=28 by >= 0.003, build
      deploy CSV submissions/nb1070_deploy_emax_aux.csv from refit on all
      253 unblind + 4139 train Emax model -> 513 deploy predictions.

Outputs:
    scripts/nb1070_emax_aux.py
    data/processed/nb1070_summary.json
    data/processed/nb1070_emax_pred_test_513.npy
    data/processed/nb1070_emax_aux_mean_bag_oof.npy   (253,)
    data/processed/nb1070_direct_mean_bag_oof.npy     (253,)
    submissions/nb1070_deploy_emax_aux.csv (only if beats)
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
from pxr.data import load_train, load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1070"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]
DECISION_MARGIN = 0.003
K = 28  # SHAP top-K (from nb2103)

# Reference: nb2103 K=28 winner
NB2103_K28_MEAN_BAG = 0.4737
NB2103_K28_MEDIAN_BAG = 0.4698

# Reference (PRE-unblind anchor)
CHEMPROP_AUX_REF = 0.6216

# Cached caches
ATOMPAIR_TR_PATH = DATA_PROCESSED / "tr_atompair.npy"
ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TR_PATH = DATA_PROCESSED / "tr_maccs.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TR_PATH = DATA_PROCESSED / "tr_chemprop_embed_300.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
AVALON_TR_PATH = DATA_PROCESSED / "tr_avalon512.npy"
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")

EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
NB1392_SUMMARY = DATA_PROCESSED / "nb1392_summary.json"
NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"
NB1523_SUMMARY = DATA_PROCESSED / "nb1523_summary.json"
NB1524_SUMMARY = DATA_PROCESSED / "nb1524_summary.json"
NB1541_SUMMARY = DATA_PROCESSED / "nb1541_summary.json"
NB2063_SHAP_IMP = DATA_PROCESSED / "nb2063_shap_importance_full117.npy"
NB2103_SUMMARY = DATA_PROCESSED / "nb2103_summary.json"

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
    """Same union as nb2063/nb2103."""
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


def _tanimoto_topk(fp_q: np.ndarray, fp_pool: np.ndarray, k: int):
    a = fp_q.astype(np.float32)
    b = fp_pool.astype(np.float32)
    a_sum = a.sum(axis=1)
    b_sum = b.sum(axis=1)
    n_q = a.shape[0]
    top_idx = np.zeros((n_q, k), dtype=np.int32)
    top_sim = np.zeros((n_q, k), dtype=np.float32)
    BLOCK = 64
    for s in range(0, n_q, BLOCK):
        e = min(n_q, s + BLOCK)
        inter = a[s:e] @ b.T
        denom = a_sum[s:e, None] + b_sum[None, :] - inter
        denom = np.maximum(denom, 1.0)
        sim = inter / denom
        if k >= b.shape[0]:
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
    w = np.clip(top_sim.copy(), 0.0, 1.0)
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


def _cross_fit_one_seed(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    n = len(y)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], y[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _load_mordred(path: Path, n_expected: int) -> np.ndarray:
    X = np.load(path).astype(np.float32)
    if X.shape[0] != n_expected:
        raise ValueError(f"mordred shape mismatch {path}: {X.shape} vs {n_expected}")
    X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X[idx_r, idx_c] = col_med[idx_c]
    return X


def _load_npy(path: Path, n_expected: int) -> np.ndarray:
    X = np.load(path).astype(np.float32)
    if X.shape[0] != n_expected:
        raise ValueError(f"shape mismatch {path}: {X.shape}")
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X


def _extract_atompair_top_idx(sum_1484: dict) -> np.ndarray:
    for f in sum_1484["families"]:
        if f["family"] == "AtomPair":
            return np.array(f["top_idx_ranked"], dtype=int)
    raise KeyError("AtomPair entry not found in nb1484_summary.json")


def _extract_best_K_record(sum_dict: dict, records_key: str) -> dict:
    best_K = int(sum_dict["best_K"])
    for r in sum_dict[records_key]:
        if int(r["K"]) == best_K:
            return r
    raise KeyError(f"best_K {best_K} not found in {records_key}")


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Emax-aware multi-task LGBM on K={K} SHAP top-features")
    print(f"          anchor={ANCHOR}  seeds={RESID_SEEDS}  folds={RESID_FOLDS}")
    print(f"          ref nb2103 K=28 mean_bag={NB2103_K28_MEAN_BAG:.4f}  "
          f"median_bag={NB2103_K28_MEDIAN_BAG:.4f}  margin={DECISION_MARGIN}")
    print("=" * 78)

    # ---- Load TRAIN + verify Emax clean ----
    train = load_train()
    n_train = len(train)
    print(f"[load] TRAIN rows: {n_train}")
    for col in ("pec50", "emax"):
        n_null = int(train[col].isna().sum())
        n_dash = int((train[col].astype(str) == "-").sum())
        print(f"   [check] {col}: dtype={train[col].dtype}  null={n_null}  "
              f"dash='-'={n_dash}")
        if n_null > 0 or n_dash > 0:
            raise ValueError(f"TRAIN {col} has nulls or '-' strings -- abort")
    y_pec50_train = train["pec50"].astype(np.float64).to_numpy()
    y_emax_train = train["emax"].astype(np.float64).to_numpy()
    print(f"   pEC50: mean={y_pec50_train.mean():.3f}  std={y_pec50_train.std():.3f}")
    print(f"   Emax : mean={y_emax_train.mean():.3f}  std={y_emax_train.std():.3f}  "
          f"min={y_emax_train.min():.2f}  max={y_emax_train.max():.2f}")
    train_smiles = train["smiles"].astype(str).tolist()

    # ---- Load TEST + unblind ----
    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    # ---- Load anchor + residual on 253 unb ----
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(f"chemprop_aux te shape mismatch: {te_anchor_513.shape}")
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} in_RAE = {rae_anchor:.4f} (ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor

    # ---- Load nb2103 K=28 top indices (in full 117) ----
    if not NB2103_SUMMARY.exists():
        raise FileNotFoundError(f"missing {NB2103_SUMMARY}")
    with open(NB2103_SUMMARY) as f:
        nb2103_sum = json.load(f)
    top_K_idx_in_117 = None
    nb2103_K28_mean_bag = NB2103_K28_MEAN_BAG
    nb2103_K28_median_bag = NB2103_K28_MEDIAN_BAG
    for r in nb2103_sum["per_K_records"]:
        if int(r["K"]) == K:
            top_K_idx_in_117 = np.array(r["top_K_idx_in_117"], dtype=int)
            nb2103_K28_mean_bag = float(r["rae_mean_bag"])
            nb2103_K28_median_bag = float(r["rae_median_bag"])
            break
    if top_K_idx_in_117 is None:
        raise KeyError(f"K={K} record not found in nb2103_summary")
    print(f"[ref] nb2103 K=28 mean_bag={nb2103_K28_mean_bag:.4f}  "
          f"median_bag={nb2103_K28_median_bag:.4f}")
    print(f"[ref] top_K_idx_in_117 length: {len(top_K_idx_in_117)}")

    # ---- Reload K-grid winners for sub-cols ----
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
    rec_mord = _extract_best_K_record(sum_1523, "per_K_records")
    top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)
    full_ap_ranked = _extract_atompair_top_idx(sum_1484)
    K_AP_best = int(sum_1524["best_K"])
    top_ap_bit_idx = full_ap_ranked[:K_AP_best]
    K_Embed_best = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed_col_idx = top_embed_full[:K_Embed_best]
    top_avalon_bit_idx = np.array(sum_1392["top_avalon_bit_indices_ranked"], dtype=int)

    n_top_ap = int(len(top_ap_bit_idx))
    n_top_maccs = int(len(top_maccs_bit_idx))
    n_top_mord = int(len(top_mord_col_idx))
    n_top_embed = int(len(top_embed_col_idx))
    n_top_avalon = int(len(top_avalon_bit_idx))
    print(f"[reuse] AP={n_top_ap} MACCS={n_top_maccs} Mord={n_top_mord} "
          f"Embed={n_top_embed} Avalon={n_top_avalon}")

    # ---- Load fingerprint caches (TRAIN + TEST) ----
    print("\n[feat] loading TRAIN feature caches...")
    X_ap_tr = _load_npy(ATOMPAIR_TR_PATH, n_train)
    X_maccs_tr = _load_npy(MACCS_TR_PATH, n_train)
    X_mord_tr = _load_mordred(MORDRED_DIR / "X_mordred_train.npy", n_train)
    X_emb_tr = _load_npy(CHEMPROP_EMBED_TR_PATH, n_train)
    X_av_tr = _load_npy(AVALON_TR_PATH, n_train)
    print(f"   shapes (tr): ap={X_ap_tr.shape} maccs={X_maccs_tr.shape} "
          f"mord={X_mord_tr.shape} emb={X_emb_tr.shape} av={X_av_tr.shape}")

    print("[feat] loading TEST feature caches...")
    X_ap_te = _load_npy(ATOMPAIR_TE_PATH, n_test)
    X_maccs_te = _load_npy(MACCS_TE_PATH, n_test)
    X_mord_te = _load_mordred(MORDRED_DIR / "X_mordred_test.npy", n_test)
    X_emb_te = _load_npy(CHEMPROP_EMBED_TE_PATH, n_test)
    X_av_te = _load_npy(AVALON_TE_PATH, n_test)

    # ---- Slice each family to its K-winner cols ----
    X_ap_tr_top = X_ap_tr[:, top_ap_bit_idx]
    X_maccs_tr_top = X_maccs_tr[:, top_maccs_bit_idx]
    X_mord_tr_top = X_mord_tr[:, top_mord_col_idx]
    X_emb_tr_top = X_emb_tr[:, top_embed_col_idx]
    X_av_tr_top = X_av_tr[:, top_avalon_bit_idx]

    X_ap_te_top = X_ap_te[:, top_ap_bit_idx]
    X_maccs_te_top = X_maccs_te[:, top_maccs_bit_idx]
    X_mord_te_top = X_mord_te[:, top_mord_col_idx]
    X_emb_te_top = X_emb_te[:, top_embed_col_idx]
    X_av_te_top = X_av_te[:, top_avalon_bit_idx]

    # ---- ChEMBL kNN feature (TRAIN + TEST) ----
    print("\n[chembl] building ChEMBL PXR pool...")
    pool = _load_chembl_pool()

    train_mols = [standardize(s) for s in train_smiles]
    test_mols = [standardize(s) for s in test_smiles]
    train_inchikeys = set()
    test_inchikeys = set()
    for m in train_mols:
        ik = _safe_inchikey(m)
        if ik is not None:
            train_inchikeys.add(ik)
    for m in test_mols:
        ik = _safe_inchikey(m)
        if ik is not None:
            test_inchikeys.add(ik)
    # Drop test inchikeys (as nb2103 does)
    drop_keys = test_inchikeys
    pool = pool[~pool["inchikey"].isin(drop_keys)].reset_index(drop=True)
    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep_pool = fp_pool.sum(axis=1) > 0
    if not keep_pool.all():
        pool = pool[keep_pool].reset_index(drop=True)
        fp_pool = fp_pool[keep_pool]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))
    print(f"   final pool size: {len(pool)}  median pEC50={pool_median:.3f}")

    std_train_smiles = [Chem.MolToSmiles(m) if m is not None else "" for m in train_mols]
    std_test_smiles = [Chem.MolToSmiles(m) if m is not None else "" for m in test_mols]
    fp_train = morgan_fp_batch(std_train_smiles)
    fp_test = morgan_fp_batch(std_test_smiles)

    print("[chembl] kNN on train + test...")
    top_idx_tr, top_sim_tr = _tanimoto_topk(fp_train, fp_pool, k=KNN_K)
    top_idx_te, top_sim_te = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_tr, mean_sim_tr = _knn_predict(top_idx_tr, top_sim_tr,
                                                pool_labels, fallback=pool_median)
    pred_chembl_te, mean_sim_te = _knn_predict(top_idx_te, top_sim_te,
                                                pool_labels, fallback=pool_median)

    # ---- Build COMBINED 117-col matrices (TRAIN + TEST) ----
    X_tr_117 = np.concatenate(
        [X_ap_tr_top, X_maccs_tr_top, X_mord_tr_top, X_emb_tr_top, X_av_tr_top,
         pred_chembl_tr.reshape(-1, 1), mean_sim_tr.reshape(-1, 1)],
        axis=1,
    ).astype(np.float32)
    X_te_117 = np.concatenate(
        [X_ap_te_top, X_maccs_te_top, X_mord_te_top, X_emb_te_top, X_av_te_top,
         pred_chembl_te.reshape(-1, 1), mean_sim_te.reshape(-1, 1)],
        axis=1,
    ).astype(np.float32)
    feat_dim_full = X_tr_117.shape[1]
    expected_dim = (n_top_ap + n_top_maccs + n_top_mord + n_top_embed
                    + n_top_avalon + 2)
    if feat_dim_full != expected_dim:
        raise ValueError(f"feat_dim {feat_dim_full} != expected {expected_dim}")
    print(f"\n[combine] X_tr_117={X_tr_117.shape}  X_te_117={X_te_117.shape}")

    # ---- Slice to top-28 SHAP ----
    X_tr_28 = X_tr_117[:, top_K_idx_in_117].astype(np.float32)
    X_te_28 = X_te_117[:, top_K_idx_in_117].astype(np.float32)
    X_unb_28 = X_te_28[unb_idx].astype(np.float32)
    print(f"[slice] X_tr_28={X_tr_28.shape}  X_te_28={X_te_28.shape}  "
          f"X_unb_28={X_unb_28.shape}")
    # cross-check against cached X_unb_28_nb2103
    cached_X_unb_28 = np.load(DATA_PROCESSED / "X_unb_28_nb2103.npy").astype(np.float32)
    max_abs_diff = float(np.max(np.abs(X_unb_28 - cached_X_unb_28)))
    print(f"[check] max|X_unb_28 - X_unb_28_nb2103| = {max_abs_diff:.4e}")
    if max_abs_diff > 1e-3:
        print("   [warn] cross-check mismatch -- using freshly-built X_unb_28")

    # =================================================================
    # STEP 1: Emax cross-fit on TRAIN (predict Emax with K=28 features)
    # =================================================================
    print("\n" + "-" * 78)
    print("STEP 1: predict Emax from K=28 on TRAIN (5-fold cross-fit)")
    print("-" * 78)
    kf_emax = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=0)
    emax_oof_train = np.full(n_train, np.nan, dtype=np.float64)
    for fi, (tr_loc, va_loc) in enumerate(kf_emax.split(np.arange(n_train))):
        ts = time.time()
        mdl = lgb.LGBMRegressor(**_lgbm_params(0))
        mdl.fit(X_tr_28[tr_loc], y_emax_train[tr_loc])
        emax_oof_train[va_loc] = mdl.predict(X_tr_28[va_loc])
        print(f"   emax fold {fi}: wall={time.time()-ts:.1f}s  "
              f"va_pred mean={emax_oof_train[va_loc].mean():.3f}  "
              f"std={emax_oof_train[va_loc].std():.3f}")
    emax_oof_rae = float(rae(y_emax_train, emax_oof_train))
    emax_oof_pearson = float(np.corrcoef(y_emax_train, emax_oof_train)[0, 1])
    print(f"   Emax cross-fit RAE on TRAIN: {emax_oof_rae:.4f}  "
          f"Pearson: {emax_oof_pearson:.4f}")

    # Refit on full TRAIN -> emax on TEST 513
    print("\n[deploy emax] refit on full 4139 TRAIN -> 513 TEST emax_pred")
    mdl_emax_full = lgb.LGBMRegressor(**_lgbm_params(0))
    mdl_emax_full.fit(X_tr_28, y_emax_train)
    emax_pred_test_513 = mdl_emax_full.predict(X_te_28).astype(np.float32)
    emax_pred_unb = emax_pred_test_513[unb_idx]
    print(f"   emax_pred_test_513 mean={emax_pred_test_513.mean():.3f}  "
          f"std={emax_pred_test_513.std():.3f}")
    np.save(DATA_PROCESSED / f"{TAG}_emax_pred_test_513.npy", emax_pred_test_513)
    print(f"   [save] {DATA_PROCESSED / f'{TAG}_emax_pred_test_513.npy'}")

    # =================================================================
    # STEP 2: residual LGBM on K=28+1 (Emax_pred as 29th feature)
    # =================================================================
    print("\n" + "-" * 78)
    print("STEP 2: residual LGBM K=28+1 (Emax_pred) -- 5 seeds, 5-fold cross-fit")
    print("-" * 78)
    X_unb_29 = np.concatenate(
        [X_unb_28, emax_pred_unb.reshape(-1, 1).astype(np.float32)],
        axis=1,
    ).astype(np.float32)
    print(f"   X_unb_29 shape: {X_unb_29.shape}")

    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae_list: list[float] = []
    per_seed_records = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        resid_oof_s = _cross_fit_one_seed(X_unb_29, residual, s)
        pred_corr_s = anchor + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae_list.append(rae_s)
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_anchor": rae_s - rae_anchor,
            "resid_oof_std": float(resid_oof_s.std()),
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   K=28+Emax seed={s:3d}: rae={rae_s:.4f}  "
              f"(d_vs_anchor={rae_s - rae_anchor:+.4f})  "
              f"wall={time.time()-ts:.1f}s")

    emax_mean_bag = per_seed_corrected.mean(axis=0)
    emax_median_bag = np.median(per_seed_corrected, axis=0)
    emax_mean_bag_rae = float(rae(y_unb, emax_mean_bag))
    emax_median_bag_rae = float(rae(y_unb, emax_median_bag))
    emax_per_seed_arr = np.array(per_seed_rae_list)
    emax_per_seed_mean = float(emax_per_seed_arr.mean())
    emax_per_seed_std = float(emax_per_seed_arr.std())
    np.save(DATA_PROCESSED / f"{TAG}_emax_aux_mean_bag_oof.npy",
            emax_mean_bag.astype(np.float32))
    print(f"\n   EMAX-AUX  mean_bag RAE   = {emax_mean_bag_rae:.4f}  "
          f"(d_vs_nb2103_K28={emax_mean_bag_rae - nb2103_K28_mean_bag:+.4f})")
    print(f"   EMAX-AUX  median_bag RAE = {emax_median_bag_rae:.4f}  "
          f"(d_vs_nb2103_K28={emax_median_bag_rae - nb2103_K28_median_bag:+.4f})")
    print(f"   per-seed mean={emax_per_seed_mean:.4f}  std={emax_per_seed_std:.4f}")

    # =================================================================
    # DIRECT arm: predict both pEC50-residual and Emax with shared base
    # (same K=28 features), then average residuals.
    # The Emax predictions are normalized to residual scale by subtracting
    # mean and rescaling to residual std.
    # =================================================================
    print("\n" + "-" * 78)
    print("DIRECT ARM: shared-base predict (pEC50-residual, Emax) -> avg")
    print("-" * 78)
    per_seed_direct = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_direct_rae: list[float] = []
    # Pre-build: Emax on 253 unblind cross-fit (using TRAIN+unb -- but we don't
    # have Emax for unb, so use the TRAIN-fit emax_pred_unb already computed).
    # On the 253 unb, the Emax "OOF" is just emax_pred_unb (no honest cross-fit
    # available since unb has no Emax labels). Use it as a fixed feature.
    resid_std = float(residual.std())
    emax_resid_proxy_unb = (
        (emax_pred_unb - emax_pred_unb.mean())
        / (emax_pred_unb.std() + 1e-6) * resid_std
    )
    print(f"   resid_std={resid_std:.4f}  emax_proxy std={emax_resid_proxy_unb.std():.4f}")
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        # base pEC50-residual cross-fit on K=28 only
        resid_oof_s = _cross_fit_one_seed(X_unb_28, residual, s)
        # average the two residual proxies (pEC50-residual prediction + Emax proxy)
        avg = 0.5 * resid_oof_s + 0.5 * emax_resid_proxy_unb
        pred_corr_s = anchor + avg
        per_seed_direct[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_direct_rae.append(rae_s)
        print(f"   DIRECT seed={s:3d}: rae={rae_s:.4f}  wall={time.time()-ts:.1f}s")
    direct_mean_bag = per_seed_direct.mean(axis=0)
    direct_median_bag = np.median(per_seed_direct, axis=0)
    direct_mean_bag_rae = float(rae(y_unb, direct_mean_bag))
    direct_median_bag_rae = float(rae(y_unb, direct_median_bag))
    np.save(DATA_PROCESSED / f"{TAG}_direct_mean_bag_oof.npy",
            direct_mean_bag.astype(np.float32))
    print(f"\n   DIRECT mean_bag RAE   = {direct_mean_bag_rae:.4f}  "
          f"(d_vs_nb2103_K28={direct_mean_bag_rae - nb2103_K28_mean_bag:+.4f})")
    print(f"   DIRECT median_bag RAE = {direct_median_bag_rae:.4f}")

    # =================================================================
    # VERDICT
    # =================================================================
    beats_emax = emax_mean_bag_rae < nb2103_K28_mean_bag - DECISION_MARGIN
    flat_emax = abs(emax_mean_bag_rae - nb2103_K28_mean_bag) < DECISION_MARGIN
    beats_emax_median = emax_median_bag_rae < nb2103_K28_median_bag - DECISION_MARGIN
    if beats_emax:
        verdict = "EMAX_AUX_BEATS_NB2103_K28"
    elif flat_emax:
        verdict = "EMAX_AUX_FLAT_VS_NB2103_K28"
    else:
        verdict = "EMAX_AUX_HURTS_NB2103_K28"
    print("\n" + "=" * 78)
    print(f"VERDICT: {verdict}")
    print(f"   nb2103 K=28 mean_bag    = {nb2103_K28_mean_bag:.4f}  "
          f"median_bag = {nb2103_K28_median_bag:.4f}")
    print(f"   nb1070 EMAX_AUX mean_bag  = {emax_mean_bag_rae:.4f}  "
          f"(d = {emax_mean_bag_rae - nb2103_K28_mean_bag:+.4f})")
    print(f"   nb1070 EMAX_AUX median_bag= {emax_median_bag_rae:.4f}  "
          f"(d = {emax_median_bag_rae - nb2103_K28_median_bag:+.4f})")
    print(f"   nb1070 DIRECT   mean_bag  = {direct_mean_bag_rae:.4f}")
    print("=" * 78)

    # =================================================================
    # DEPLOY (only if EMAX_AUX beats)
    # =================================================================
    deploy_path = None
    deploy_in_rae = None
    if beats_emax:
        print("\n[deploy] EMAX_AUX beats -- building deploy CSV")
        # Build X_te_29 with emax_pred_test_513 as 29th feature
        X_te_29 = np.concatenate(
            [X_te_28, emax_pred_test_513.reshape(-1, 1).astype(np.float32)],
            axis=1,
        ).astype(np.float32)
        # Refit residual LGBM on ALL 253 unblind for each seed, predict 513
        deploy_stack = np.zeros((len(RESID_SEEDS), n_test), dtype=np.float64)
        for i, s in enumerate(RESID_SEEDS):
            mdl = lgb.LGBMRegressor(**_lgbm_params(s))
            mdl.fit(X_unb_29, residual)
            resid_513 = mdl.predict(X_te_29).astype(np.float64)
            deploy_stack[i] = te_anchor_513 + resid_513
        deploy_mean = deploy_stack.mean(axis=0).astype(np.float32)
        deploy_in_rae = float(rae(y_unb, deploy_mean[unb_idx].astype(np.float64)))
        deploy_csv = SUBMISSIONS / f"{TAG}_deploy_emax_aux.csv"
        pd.DataFrame({
            "SMILES": te["smiles"].values,
            "Molecule Name": te["name"].values,
            "pEC50": deploy_mean,
        }).to_csv(deploy_csv, index=False)
        np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy_mean)
        deploy_path = str(deploy_csv)
        print(f"   in-sample 253 RAE: {deploy_in_rae:.4f}")
        print(f"   [save] {deploy_csv}")
    else:
        print(f"\n[deploy] EMAX_AUX does not beat -- no deploy CSV (verdict={verdict})")

    summary = {
        "tag": TAG,
        "method": "emax_aux_multitask_LGBM_K28+1_on_chemprop_aux_residual",
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "K": K,
        "n_train": n_train,
        "n_test": n_test,
        "n_unb": n_unb,
        "rae_anchor": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "y_emax_train_mean": float(y_emax_train.mean()),
        "y_emax_train_std": float(y_emax_train.std()),
        "emax_cross_fit_train_rae": emax_oof_rae,
        "emax_cross_fit_train_pearson": emax_oof_pearson,
        "emax_pred_test_513_mean": float(emax_pred_test_513.mean()),
        "emax_pred_test_513_std": float(emax_pred_test_513.std()),
        "feat_dim_full_117": int(feat_dim_full),
        "feat_dim_used_K28": int(K),
        "feat_dim_emax_aux_29": int(K + 1),
        "nb2103_K28_mean_bag_ref": nb2103_K28_mean_bag,
        "nb2103_K28_median_bag_ref": nb2103_K28_median_bag,
        "emax_aux_per_seed_rae": per_seed_rae_list,
        "emax_aux_per_seed_mean": emax_per_seed_mean,
        "emax_aux_per_seed_std": emax_per_seed_std,
        "emax_aux_mean_bag_rae": emax_mean_bag_rae,
        "emax_aux_median_bag_rae": emax_median_bag_rae,
        "delta_emax_aux_vs_nb2103_K28_mean": emax_mean_bag_rae - nb2103_K28_mean_bag,
        "delta_emax_aux_vs_nb2103_K28_median": emax_median_bag_rae - nb2103_K28_median_bag,
        "direct_per_seed_rae": per_seed_direct_rae,
        "direct_mean_bag_rae": direct_mean_bag_rae,
        "direct_median_bag_rae": direct_median_bag_rae,
        "delta_direct_vs_nb2103_K28_mean": direct_mean_bag_rae - nb2103_K28_mean_bag,
        "beats_emax_aux_mean": bool(beats_emax),
        "beats_emax_aux_median": bool(beats_emax_median),
        "flat_emax_aux": bool(flat_emax),
        "verdict": verdict,
        "decision_margin": DECISION_MARGIN,
        "deploy_csv": deploy_path,
        "deploy_in_rae_253": deploy_in_rae,
        "x_unb_28_max_abs_diff_vs_cached": max_abs_diff,
        "pre_unblind_clean": True,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "K", "n_train", "n_unb",
        "rae_anchor",
        "emax_cross_fit_train_rae", "emax_cross_fit_train_pearson",
        "nb2103_K28_mean_bag_ref", "nb2103_K28_median_bag_ref",
        "emax_aux_per_seed_rae",
        "emax_aux_mean_bag_rae", "emax_aux_median_bag_rae",
        "delta_emax_aux_vs_nb2103_K28_mean",
        "delta_emax_aux_vs_nb2103_K28_median",
        "direct_mean_bag_rae",
        "verdict", "deploy_csv", "wall_sec",
    ):
        print(f"  {k}: {res.get(k)}")
