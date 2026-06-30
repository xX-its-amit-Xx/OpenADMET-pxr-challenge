"""nb1604 -- Meta-stack over the 5 nb1554 per-seed CatBoost corrected OOFs.

HYPOTHESIS:
    nb1554 produced 5 per-seed corrected OOFs (seeds 0, 1, 7, 42, 137) by
    cross-fitting a CatBoost(MAE, d4, n200, lr0.05, l2=5) residual on the
    117-col 5-way K-tuned feature stack and adding back the chemprop_aux
    anchor.  The naive 1/K mean of those 5 vectors yielded pool RAE
    0.5163 (nb1554 mean_bag).  nb1561's outer-bag BoB MEAN over 25 inner
    seeds is 0.5155.  If the 5 seed-specific corrected OOFs disagree in a
    structured way (per-seed disagreement carries signal), a meta-learner
    fed those 5 vectors as columns may beat the naive mean.

PROTOCOL
    1. Rebuild the 117-col 5-way K-tuned feature stack (PRE-unblind clean,
       same recipe as nb1554/nb1561):
            top-25 AtomPair  + top-20 MACCS + top-20 Mordred
            + top-20 ChempropEmbed + top-30 Avalon + pred_chembl_pec50
            + mean_sim
       Anchor = chemprop_aux te[unb_idx] (in_RAE 0.6216).
    2. For seed s in [0, 1, 7, 42, 137]:
         a. KFold(5, shuffle=True, random_state=s) cross-fit
            CatBoost(MAE, d4, n200, lr0.05, l2=5, random_seed=s) on
            residual = y_unb - anchor.
         b. pred_corr_s = anchor + resid_oof_s   --> 1 column of P (253, 5).
    3. P = stack of the 5 corrected OOFs.  Save as
       data/processed/nb1604_per_seed_corrected_oof.npy   (5, 253) float32.
    4. Methods:
         A. naive 1/K mean of 5 columns  (==  nb1554 mean_bag).
         B. 5-fold SLSQP simplex cross-fit on (P, y_unb).
         C. 5-fold shallow LGBM-Huber cross-fit on (P, y_unb).
            depth=2, n_est=40, lr=0.03, min_child=30, alpha=1.0  (nb1563 shape).
         D. Closed-form SLSQP-on-all (no cross-fit, sanity in-sample only).
    5. Verdict at 0.003 margin vs nb1561 BoB MEAN ref (0.5155).
    6. Save best honest cross-fit OOF to nb1604_best_oof.npy.

Outputs:
    scripts/nb1604_meta_seed_features.py            (this file)
    data/processed/nb1604_per_seed_corrected_oof.npy   (5, 253) float32
    data/processed/nb1604_best_oof.npy                 (253,) float32
    data/processed/nb1604_summary.json
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
from scipy.optimize import minimize
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

TAG = "nb1604"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

RESID_SEEDS = [0, 1, 7, 42, 137]      # match nb1554
RESID_FOLDS = 5

META_FOLDS = 5
META_SEED = 42

# References
CHEMPROP_AUX_REF = 0.6216
NB1554_REF = 0.5163      # naive 1/K mean of the same 5 columns
NB1561_BOB_MEAN_REF = 0.5155
DECISION_MARGIN = 0.003

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
    """Identical to nb1554/nb1561 (PRE-unblind clean)."""
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


def _residual_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray,
                                 seed: int) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = CatBoostRegressor(**_cat_params(seed))
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


# ----------- meta-learners ----------------------------------------------------
def _lgbm_meta_params(seed: int) -> dict:
    return dict(
        objective="huber",
        alpha=1.0,
        learning_rate=0.03,
        n_estimators=40,
        max_depth=2,
        num_leaves=4,
        min_child_samples=30,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        verbosity=-1,
        random_state=seed,
        n_jobs=2,
    )


def _lgbm_cross_fit(X: np.ndarray, y: np.ndarray,
                    n_splits: int, seed: int) -> np.ndarray:
    n = len(y)
    oof = np.full(n, np.nan, dtype=np.float64)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = LGBMRegressor(**_lgbm_meta_params(seed))
        mdl.fit(X[tr_loc], y[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _slsqp_blend_weights(P_tr: np.ndarray, y_tr: np.ndarray) -> np.ndarray:
    K = P_tr.shape[1]
    w0 = np.full(K, 1.0 / K)

    def _loss(w: np.ndarray) -> float:
        r = y_tr - P_tr @ w
        return float(np.mean(r * r))

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bnds = [(0.0, 1.0)] * K
    res = minimize(
        _loss, w0, method="SLSQP",
        bounds=bnds, constraints=cons,
        options={"ftol": 1e-10, "maxiter": 500},
    )
    w = np.clip(np.asarray(res.x, dtype=np.float64), 0.0, 1.0)
    s = w.sum()
    if s <= 0:
        return np.full(K, 1.0 / K)
    return w / s


def _slsqp_cross_fit(P: np.ndarray, y: np.ndarray,
                     n_splits: int, seed: int):
    n = len(y)
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_records = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for f, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n))):
        w = _slsqp_blend_weights(P[tr_loc], y[tr_loc])
        oof[va_loc] = P[va_loc] @ w
        fold_records.append({
            "fold": int(f),
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "w": [float(x) for x in w.tolist()],
        })
    return oof, fold_records


# ----------- main -------------------------------------------------------------
def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- META-STACK over 5 nb1554 per-seed CatBoost corrected OOFs")
    print(f"         resid seeds = {RESID_SEEDS}   resid folds = {RESID_FOLDS}")
    print(f"         meta folds  = {META_FOLDS}    meta seed   = {META_SEED}")
    print(f"         refs: nb1554 ({NB1554_REF:.4f})  "
          f"nb1561 BoB MEAN ({NB1561_BOB_MEAN_REF:.4f})  "
          f"margin = {DECISION_MARGIN}")
    print("=" * 78)

    # ---- Load truth + anchor ----
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
        raise ValueError(f"anchor shape mismatch: {te_anchor_513.shape}")
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Load K-grid winners (same as nb1554/nb1561) ----
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
    print(f"[reuse] AP {n_top_ap} (K={K_AP_best}) | MACCS {n_top_maccs} | "
          f"Mord {n_top_mord} (K={K_Mord_best}) | "
          f"Embed {n_top_embed} (K={K_Embed_best}) | "
          f"Aval {n_top_avalon}")

    # ---- Build 117-col feature matrix on unb ----
    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test)
    X_ap_unb_top = X_ap_te[unb_idx][:, top_ap_bit_idx].astype(np.float32)
    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)
    X_maccs_unb_top = X_maccs_te[unb_idx][:, top_maccs_bit_idx].astype(np.float32)
    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    X_mord_unb_top = X_mord_te[unb_idx][:, top_mord_col_idx].astype(np.float32)
    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)
    X_emb_unb_top = X_emb_te[unb_idx][:, top_embed_col_idx].astype(np.float32)
    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)
    X_av_unb_top = X_av_te[unb_idx][:, top_avalon_bit_idx].astype(np.float32)

    # ChEMBL kNN feature
    print("-" * 78)
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
    std_test_smiles = [
        Chem.MolToSmiles(m) if m is not None else "" for m in test_mols
    ]
    fp_test = morgan_fp_batch(std_test_smiles)
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )
    pred_chembl_unb = pred_chembl_pec50[unb_idx].astype(np.float32)
    mean_sim_unb = mean_sim[unb_idx].astype(np.float32)

    X_unb = np.concatenate(
        [
            X_ap_unb_top, X_maccs_unb_top, X_mord_unb_top,
            X_emb_unb_top, X_av_unb_top,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim = X_unb.shape[1]
    print(f"   COMBINED 5-WAY K-TUNED matrix: {X_unb.shape}")

    # ---- Per-seed CatBoost residual cross-fit -> (5, 253) -------------------
    print("\n" + "-" * 78)
    print("PER-SEED CATBOOST RESIDUAL CROSS-FIT -> (5, 253) corrected OOFs")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        resid_oof_s = _residual_cross_fit_one_seed(X_unb, residual, s)
        pred_corr_s = anchor + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_anchor": rae_s - rae_anchor,
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   seed {s:3d}:  rae_corr = {rae_s:.4f}  "
              f"wall = {time.time() - ts:.1f}s")

    # Save the per-seed matrix
    per_seed_path = DATA_PROCESSED / f"{TAG}_per_seed_corrected_oof.npy"
    np.save(per_seed_path, per_seed_corrected.astype(np.float32))
    print(f"[save] {per_seed_path}  shape = (5, 253)")

    # ---- Feature matrix for meta-learner: P shape (253, 5) ------------------
    P = per_seed_corrected.T.astype(np.float64)   # (253, 5)
    K = P.shape[1]
    print(f"\n[meta] P feature matrix shape = {P.shape}")
    print(f"[meta] per-column RAE          = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"[meta] per-column mean         = "
          f"[{', '.join(f'{m:.3f}' for m in P.mean(axis=0).tolist())}]")
    print(f"[meta] per-column std          = "
          f"[{', '.join(f'{s:.3f}' for s in P.std(axis=0).tolist())}]")
    pairwise_corr = np.corrcoef(per_seed_corrected)
    off_diag = pairwise_corr[np.triu_indices_from(pairwise_corr, k=1)]
    print(f"[meta] mean off-diagonal pairwise Pearson(seed_i, seed_j) = "
          f"{off_diag.mean():.4f}  (min {off_diag.min():.4f}, "
          f"max {off_diag.max():.4f})")

    # ---- Method A: naive 1/K mean (== nb1554 mean_bag) ----------------------
    print("\n" + "-" * 78)
    print("METHOD A -- naive 1/K mean of 5 columns  (== nb1554 mean_bag)")
    print("-" * 78)
    oof_mean = P.mean(axis=1)
    rae_mean = float(rae(y_unb, oof_mean))
    print(f"   rae_naive_mean        = {rae_mean:.4f}")

    # ---- Method B: 5-fold SLSQP simplex cross-fit ---------------------------
    print("\n" + "-" * 78)
    print(f"METHOD B -- 5-fold SLSQP simplex cross-fit  meta_seed={META_SEED}")
    print("-" * 78)
    oof_slsqp, slsqp_folds = _slsqp_cross_fit(
        P, y_unb, n_splits=META_FOLDS, seed=META_SEED
    )
    rae_slsqp = float(rae(y_unb, oof_slsqp))
    W_slsqp = np.array([f["w"] for f in slsqp_folds])
    w_slsqp_mean = W_slsqp.mean(axis=0).tolist()
    w_slsqp_std = W_slsqp.std(axis=0).tolist()
    print(f"   mean w over folds     = "
          f"[{', '.join(f'{w:.3f}' for w in w_slsqp_mean)}]")
    print(f"   std  w over folds     = "
          f"[{', '.join(f'{w:.3f}' for w in w_slsqp_std)}]")
    print(f"   rae_slsqp_crossfit    = {rae_slsqp:.4f}")

    # ---- Method C: 5-fold shallow LGBM-Huber cross-fit ----------------------
    print("\n" + "-" * 78)
    print(f"METHOD C -- 5-fold shallow LGBM Huber cross-fit  meta_seed={META_SEED}")
    print(f"           (depth=2, n_est=40, lr=0.03, min_child=30, alpha=1.0)")
    print("-" * 78)
    oof_lgbm = _lgbm_cross_fit(P, y_unb, n_splits=META_FOLDS, seed=META_SEED)
    rae_lgbm = float(rae(y_unb, oof_lgbm))
    print(f"   rae_lgbm_meta_stack   = {rae_lgbm:.4f}")

    # ---- Method D: SLSQP-on-all sanity (in-sample, not honest) --------------
    print("\n" + "-" * 78)
    print("METHOD D -- SLSQP simplex on full P (IN-SAMPLE, not honest)")
    print("-" * 78)
    w_all = _slsqp_blend_weights(P, y_unb)
    oof_slsqp_all = P @ w_all
    rae_slsqp_all = float(rae(y_unb, oof_slsqp_all))
    print(f"   in-sample w           = "
          f"[{', '.join(f'{w:.3f}' for w in w_all.tolist())}]")
    print(f"   in-sample rae         = {rae_slsqp_all:.4f}  "
          f"(IN-SAMPLE OPTIMISTIC; not used for verdict)")

    # ---- Verdict ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    # Honest methods only (A, B, C)
    methods = [
        ("naive_mean", rae_mean, oof_mean),
        ("slsqp_crossfit", rae_slsqp, oof_slsqp),
        ("lgbm_meta_stack", rae_lgbm, oof_lgbm),
    ]
    methods_sorted = sorted(methods, key=lambda x: x[1])
    print("   ranking (best -> worst, honest cross-fit only):")
    for name, r, _ in methods_sorted:
        print(f"     {name:<22s}  RAE = {r:.4f}")

    best_name, rae_best, best_oof = methods_sorted[0]

    delta_mean_vs_nb1561 = rae_mean - NB1561_BOB_MEAN_REF
    delta_slsqp_vs_nb1561 = rae_slsqp - NB1561_BOB_MEAN_REF
    delta_lgbm_vs_nb1561 = rae_lgbm - NB1561_BOB_MEAN_REF
    delta_best_vs_nb1561 = rae_best - NB1561_BOB_MEAN_REF

    beats_nb1561_mean = rae_mean < NB1561_BOB_MEAN_REF - DECISION_MARGIN
    beats_nb1561_slsqp = rae_slsqp < NB1561_BOB_MEAN_REF - DECISION_MARGIN
    beats_nb1561_lgbm = rae_lgbm < NB1561_BOB_MEAN_REF - DECISION_MARGIN
    beats_nb1561_best = rae_best < NB1561_BOB_MEAN_REF - DECISION_MARGIN
    flat_vs_nb1561_best = abs(rae_best - NB1561_BOB_MEAN_REF) < DECISION_MARGIN

    if beats_nb1561_best:
        verdict = f"META_SEED_BEATS_NB1561_NEW_PRE_UNBLIND_PRIMARY_via_{best_name}"
    elif flat_vs_nb1561_best:
        verdict = f"META_SEED_FLAT_VS_NB1561_best_{best_name}"
    else:
        verdict = f"META_SEED_WORSE_THAN_NB1561_best_{best_name}"

    print(f"\n   d_mean_vs_nb1561      = {delta_mean_vs_nb1561:+.4f}")
    print(f"   d_slsqp_vs_nb1561     = {delta_slsqp_vs_nb1561:+.4f}")
    print(f"   d_lgbm_vs_nb1561      = {delta_lgbm_vs_nb1561:+.4f}")
    print(f"   d_best_vs_nb1561      = {delta_best_vs_nb1561:+.4f}  "
          f"(best = {best_name})")
    print(f"   beats_nb1561_best     = {beats_nb1561_best}")
    print(f"   verdict               = {verdict}")
    print("=" * 78)

    # ---- Save ---------------------------------------------------------------
    out_best = DATA_PROCESSED / f"{TAG}_best_oof.npy"
    np.save(out_best, best_oof.astype(np.float32))
    print(f"\n[save] {out_best}  (variant = {best_name})")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "data_source": ("rebuild nb1554 117-col 5-way K-tuned feature stack "
                        "(AP25 + MACCS20 + Mord20 + ChempropEmbed20 + Aval30 "
                        "+ pred_chembl_pec50 + mean_sim) + ChEMBL kNN feature"),
        "model_family": "CatBoost-per-seed + meta-learner",
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "meta_folds": META_FOLDS,
        "meta_seed": META_SEED,
        "lgbm_meta_params": _lgbm_meta_params(META_SEED),
        "n_unb": int(n_unb),
        "n_test": int(n_test),
        "n_chembl_pool": int(len(pool)),
        "feat_dim_inner_catboost": int(feat_dim),
        "rae_anchor_chemprop_aux": rae_anchor,
        "per_seed_rae": per_seed_rae,
        "per_seed_records": per_seed_records,
        "P_shape": list(P.shape),
        "P_column_means": P.mean(axis=0).tolist(),
        "P_column_stds": P.std(axis=0).tolist(),
        "pairwise_pearson_off_diag_mean": float(off_diag.mean()),
        "pairwise_pearson_off_diag_min": float(off_diag.min()),
        "pairwise_pearson_off_diag_max": float(off_diag.max()),
        "rae_naive_mean": rae_mean,
        "rae_slsqp_crossfit": rae_slsqp,
        "rae_lgbm_meta_stack": rae_lgbm,
        "rae_slsqp_in_sample": rae_slsqp_all,
        "slsqp_in_sample_weights": [float(x) for x in w_all.tolist()],
        "slsqp_fold_records": slsqp_folds,
        "slsqp_w_mean_over_folds": w_slsqp_mean,
        "slsqp_w_std_over_folds": w_slsqp_std,
        "rae_best": rae_best,
        "best_variant": best_name,
        "method_ranking": [{"name": n, "rae": r} for n, r, _ in methods_sorted],
        "delta_mean_vs_nb1561": delta_mean_vs_nb1561,
        "delta_slsqp_vs_nb1561": delta_slsqp_vs_nb1561,
        "delta_lgbm_vs_nb1561": delta_lgbm_vs_nb1561,
        "delta_best_vs_nb1561": delta_best_vs_nb1561,
        "beats_nb1561_mean": bool(beats_nb1561_mean),
        "beats_nb1561_slsqp": bool(beats_nb1561_slsqp),
        "beats_nb1561_lgbm": bool(beats_nb1561_lgbm),
        "beats_nb1561": bool(beats_nb1561_best),
        "flat_vs_nb1561_best": bool(flat_vs_nb1561_best),
        "verdict": verdict,
        "nb1554_ref": NB1554_REF,
        "nb1561_bob_mean_ref": NB1561_BOB_MEAN_REF,
        "decision_margin": DECISION_MARGIN,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
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
        "n_unb", "n_test", "n_chembl_pool", "feat_dim_inner_catboost",
        "resid_seeds", "resid_folds", "meta_folds", "meta_seed",
        "rae_anchor_chemprop_aux",
        "per_seed_rae",
        "pairwise_pearson_off_diag_mean",
        "rae_naive_mean",
        "rae_slsqp_crossfit",
        "rae_lgbm_meta_stack",
        "rae_slsqp_in_sample",
        "slsqp_in_sample_weights",
        "slsqp_w_mean_over_folds",
        "rae_best", "best_variant",
        "delta_best_vs_nb1561",
        "beats_nb1561", "flat_vs_nb1561_best",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
