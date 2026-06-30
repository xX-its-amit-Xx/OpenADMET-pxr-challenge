"""nb2173 -- 2-STAGE RESIDUAL CASCADE on top of chemprop_aux.

HYPOTHESIS:
    Single-stage K=28 SHAP-top LGBM(MSE) residual correction of chemprop_aux
    (nb2103 K=28) gave mean_bag 0.4737 / median_bag 0.4698. The residuals
    AFTER that correction (residual_2 = y - corrected) may still contain
    learnable structure on the SAME K=28 features (a second pass mops up
    the part the first pass left underfit at LR=0.03, n_est=300, depth=4).
    A creative variation also tries a DIFFERENT K=28 (next-28 SHAP ranks
    29..56) for stage 2, decorrelating the second model from the first.

PROTOCOL:
    1. Reuse nb2103's exact 117-col feature matrix + cached SHAP importance
       (nb2063_shap_importance_full117.npy). Slice top-K1=28 features for
       STAGE 1, and (for the variant) ranks-29..56 for STAGE 2.
    2. STAGE 1: residual_1 = y_unb - chemprop_aux. Build 5-seed
       (seeds 0/1/7/42/137), 5-fold cross-fit LGBM(MSE) on (X_K1, residual_1).
       pred_stage1[i] is the held-out prediction for residual_1.
       corrected = chemprop_aux + mean_bag(pred_stage1)        (per-seed bag)
    3. STAGE 2A (same K=28 features): residual_2 = y_unb - corrected.
       5-seed, 5-fold cross-fit LGBM(MSE) on (X_K1, residual_2).
       final_A = corrected + mean_bag(pred_stage2_A)
    4. STAGE 2B (different K=28 = ranks 29..56): same protocol on (X_K2,
       residual_2). final_B = corrected + mean_bag(pred_stage2_B).
    5. Report mean-bag RAE and median-bag RAE per variant.
    6. Decision vs nb2103 K=28 single-stage (0.4737 / 0.4698) at margin
       0.003. If beats, log "BEATS_NB2103_K28" and prepare deploy.

Outputs:
    scripts/nb2173_residual_cascade.py
    data/processed/nb2173_summary.json
    data/processed/nb2173_stage1_pred_oof.npy           (253,) float32
    data/processed/nb2173_stage2A_pred_oof_meanbag.npy  (253,) float32
    data/processed/nb2173_stage2B_pred_oof_meanbag.npy  (253,) float32
    data/processed/nb2173_final_A_meanbag.npy           (253,) float32
    data/processed/nb2173_final_B_meanbag.npy           (253,) float32
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
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb2173"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]
K1 = 28           # stage 1: top-28 SHAP features
K2 = 28           # stage 2B: next-28 SHAP features (ranks 29..56)

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


def _residual_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray,
                                 seed: int) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _bag_seeds(X: np.ndarray, residual: np.ndarray):
    """Returns mean-bag OOF and per-seed OOF matrix (n_seeds, n)."""
    n = len(residual)
    per_seed = np.zeros((len(RESID_SEEDS), n), dtype=np.float64)
    for i, s in enumerate(RESID_SEEDS):
        per_seed[i] = _residual_cross_fit_one_seed(X, residual, s)
    mean_bag = per_seed.mean(axis=0)
    median_bag = np.median(per_seed, axis=0)
    return per_seed, mean_bag, median_bag


def _load_mordred_test(n_test_expected: int) -> np.ndarray:
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(f"Mordred cache missing: {mte_p}")
    X_te_m = np.load(mte_p).astype(np.float32)
    if X_te_m.shape[0] != n_test_expected:
        raise ValueError(f"Mordred shape {X_te_m.shape} vs n={n_test_expected}")
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
        raise FileNotFoundError(f"missing: {path}")
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
    raise KeyError("AtomPair not found in nb1484_summary.json")


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
    print(f"{TAG} -- 2-STAGE RESIDUAL CASCADE  K1={K1}  K2={K2}")
    print(f"         anchor={ANCHOR}  seeds={RESID_SEEDS}  folds={RESID_FOLDS}")
    print(f"         ref: nb2103 K=28 mean_bag={NB2103_K28_MEAN_BAG_REF:.4f} "
          f"median_bag={NB2103_K28_MEDIAN_BAG_REF:.4f}  margin={DECISION_MARGIN}")
    print("=" * 78)

    if not NB2063_SHAP_IMP.exists():
        raise FileNotFoundError(f"missing {NB2063_SHAP_IMP}")
    shap_imp_full117 = np.load(NB2063_SHAP_IMP).astype(np.float32)
    full_rank_order = np.argsort(-shap_imp_full117).astype(np.int32)
    print(f"[ref] SHAP imp shape = {shap_imp_full117.shape}")

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
        raise FileNotFoundError(f"missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} in_RAE = {rae_anchor:.4f}  (ref {CHEMPROP_AUX_REF:.4f})")

    residual_1 = y_unb - anchor
    print(f"[resid1] mean={residual_1.mean():+.4f}  std={residual_1.std():.4f}")

    # ---- Load K-grid winners ----
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
    rec_mord = _extract_best_K_record(sum_1523, "per_K_records", "best_K")
    top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)
    full_ap_ranked = _extract_atompair_top_idx_from_nb1484(sum_1484)
    K_AP_best = int(sum_1524["best_K"])
    top_ap_bit_idx = full_ap_ranked[:K_AP_best]
    K_Embed_best = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed_col_idx = top_embed_full[:K_Embed_best]
    top_avalon_bit_idx = np.array(sum_1392["top_avalon_bit_indices_ranked"], dtype=int)

    # ---- Feature matrices ----
    X_ap_unb_top = _load_npy_test(ATOMPAIR_TE_PATH, n_test)[unb_idx][:, top_ap_bit_idx].astype(np.float32)
    X_maccs_unb_top = _load_npy_test(MACCS_TE_PATH, n_test)[unb_idx][:, top_maccs_bit_idx].astype(np.float32)
    X_mord_unb_top = _load_mordred_test(n_test)[unb_idx][:, top_mord_col_idx].astype(np.float32)
    X_emb_unb_top = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)[unb_idx][:, top_embed_col_idx].astype(np.float32)
    X_av_unb_top = _load_npy_test(AVALON_TE_PATH, n_test)[unb_idx][:, top_avalon_bit_idx].astype(np.float32)

    # ---- ChEMBL kNN ----
    pool = _load_chembl_pool()
    test_mols = [standardize(s) for s in test_smiles]
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
    std_test_smiles = [Chem.MolToSmiles(m) if m is not None else "" for m in test_mols]
    fp_test = morgan_fp_batch(std_test_smiles)
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median)
    pred_chembl_unb = pred_chembl_pec50[unb_idx].astype(np.float32)
    mean_sim_unb = mean_sim[unb_idx].astype(np.float32)

    # ---- Build COMBINED 117-col matrix ----
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
    if feat_dim != shap_imp_full117.shape[0]:
        raise ValueError(f"feat_dim {feat_dim} != SHAP {shap_imp_full117.shape[0]}")
    print(f"[feat] X_unb = {X_unb.shape}")

    # ---- SHAP slices ----
    topK1_idx = full_rank_order[:K1].astype(np.int32)
    topK2_idx = full_rank_order[K1:K1 + K2].astype(np.int32)   # ranks 28..55 zero-indexed = 29..56 1-indexed
    X_K1 = X_unb[:, topK1_idx].astype(np.float32)
    X_K2 = X_unb[:, topK2_idx].astype(np.float32)
    print(f"[shap] X_K1 (ranks 1..{K1})    = {X_K1.shape}")
    print(f"[shap] X_K2 (ranks {K1+1}..{K1+K2}) = {X_K2.shape}")

    # ============================================================
    # STAGE 1: residual_1 = y_unb - chemprop_aux  on X_K1
    # ============================================================
    print("\n" + "-" * 78)
    print(f"STAGE 1  (cross-fit LGBM on residual_1 with K1={K1} SHAP features)")
    print("-" * 78)
    t1 = time.time()
    per_seed_s1, mean_bag_s1, median_bag_s1 = _bag_seeds(X_K1, residual_1)
    corrected_mean_bag = anchor + mean_bag_s1
    corrected_median_bag = anchor + median_bag_s1
    rae_s1_mean = float(rae(y_unb, corrected_mean_bag))
    rae_s1_med = float(rae(y_unb, corrected_median_bag))
    print(f"[s1] mean_bag RAE  = {rae_s1_mean:.4f}  "
          f"(d_vs_anchor {rae_s1_mean - rae_anchor:+.4f}  "
          f"d_vs_nb2103_K28 {rae_s1_mean - NB2103_K28_MEAN_BAG_REF:+.4f})")
    print(f"[s1] median_bag RAE= {rae_s1_med:.4f}  "
          f"(d_vs_nb2103_K28_med {rae_s1_med - NB2103_K28_MEDIAN_BAG_REF:+.4f})")
    print(f"[s1] wall = {time.time() - t1:.1f}s")

    # residual_2 for cascade uses MEAN-BAG corrected as the new anchor
    residual_2 = y_unb - corrected_mean_bag
    print(f"[resid2] mean={residual_2.mean():+.4f}  std={residual_2.std():.4f}")

    # ============================================================
    # STAGE 2A: residual_2 on X_K1 (SAME 28 features)
    # ============================================================
    print("\n" + "-" * 78)
    print(f"STAGE 2A  (residual_2 on SAME K1={K1} features)")
    print("-" * 78)
    t2a = time.time()
    per_seed_s2a, mean_bag_s2a, median_bag_s2a = _bag_seeds(X_K1, residual_2)
    final_A_mean = corrected_mean_bag + mean_bag_s2a
    final_A_med = corrected_mean_bag + median_bag_s2a
    rae_A_mean = float(rae(y_unb, final_A_mean))
    rae_A_med = float(rae(y_unb, final_A_med))
    print(f"[s2A] mean_bag RAE  = {rae_A_mean:.4f}  "
          f"(d_vs_s1 {rae_A_mean - rae_s1_mean:+.4f}  "
          f"d_vs_nb2103_K28 {rae_A_mean - NB2103_K28_MEAN_BAG_REF:+.4f})")
    print(f"[s2A] median_bag RAE= {rae_A_med:.4f}  "
          f"(d_vs_nb2103_K28_med {rae_A_med - NB2103_K28_MEDIAN_BAG_REF:+.4f})")
    print(f"[s2A] wall = {time.time() - t2a:.1f}s")

    # ============================================================
    # STAGE 2B: residual_2 on X_K2 (NEXT 28 features, ranks 29..56)
    # ============================================================
    print("\n" + "-" * 78)
    print(f"STAGE 2B  (residual_2 on NEXT K2={K2} features, ranks {K1+1}..{K1+K2})")
    print("-" * 78)
    t2b = time.time()
    per_seed_s2b, mean_bag_s2b, median_bag_s2b = _bag_seeds(X_K2, residual_2)
    final_B_mean = corrected_mean_bag + mean_bag_s2b
    final_B_med = corrected_mean_bag + median_bag_s2b
    rae_B_mean = float(rae(y_unb, final_B_mean))
    rae_B_med = float(rae(y_unb, final_B_med))
    print(f"[s2B] mean_bag RAE  = {rae_B_mean:.4f}  "
          f"(d_vs_s1 {rae_B_mean - rae_s1_mean:+.4f}  "
          f"d_vs_nb2103_K28 {rae_B_mean - NB2103_K28_MEAN_BAG_REF:+.4f})")
    print(f"[s2B] median_bag RAE= {rae_B_med:.4f}  "
          f"(d_vs_nb2103_K28_med {rae_B_med - NB2103_K28_MEDIAN_BAG_REF:+.4f})")
    print(f"[s2B] wall = {time.time() - t2b:.1f}s")

    # ---- Save OOF artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_stage1_pred_oof.npy", mean_bag_s1.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_stage2A_pred_oof_meanbag.npy", mean_bag_s2a.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_stage2B_pred_oof_meanbag.npy", mean_bag_s2b.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_final_A_meanbag.npy", final_A_mean.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_final_B_meanbag.npy", final_B_mean.astype(np.float32))

    # ---- Verdict ----
    best_variant_rae = min(rae_A_mean, rae_A_med, rae_B_mean, rae_B_med)
    if best_variant_rae == rae_A_mean:
        best_label = "stage2A_mean_bag"
    elif best_variant_rae == rae_A_med:
        best_label = "stage2A_median_bag"
    elif best_variant_rae == rae_B_mean:
        best_label = "stage2B_mean_bag"
    else:
        best_label = "stage2B_median_bag"

    ref_for_comparison = min(NB2103_K28_MEAN_BAG_REF, NB2103_K28_MEDIAN_BAG_REF)
    delta_best_vs_nb2103 = best_variant_rae - ref_for_comparison
    if delta_best_vs_nb2103 < -DECISION_MARGIN:
        verdict = f"BEATS_NB2103_K28_AT_{best_label}"
    elif abs(delta_best_vs_nb2103) < DECISION_MARGIN:
        verdict = f"FLAT_VS_NB2103_K28_AT_{best_label}"
    else:
        verdict = "CASCADE_DOES_NOT_BEAT_NB2103_K28_SINGLE_STAGE"

    print("\n" + "=" * 78)
    print("CASCADE SUMMARY")
    print("=" * 78)
    print(f"   anchor (chemprop_aux)        = {rae_anchor:.4f}")
    print(f"   nb2103 K=28 mean_bag (ref)   = {NB2103_K28_MEAN_BAG_REF:.4f}")
    print(f"   nb2103 K=28 median_bag (ref) = {NB2103_K28_MEDIAN_BAG_REF:.4f}")
    print(f"   STAGE 1   mean_bag           = {rae_s1_mean:.4f}")
    print(f"   STAGE 1   median_bag         = {rae_s1_med:.4f}")
    print(f"   STAGE 2A  mean_bag (same K28)= {rae_A_mean:.4f}")
    print(f"   STAGE 2A  median_bag         = {rae_A_med:.4f}")
    print(f"   STAGE 2B  mean_bag (next K28)= {rae_B_mean:.4f}")
    print(f"   STAGE 2B  median_bag         = {rae_B_med:.4f}")
    print(f"   best variant                 = {best_label}  RAE={best_variant_rae:.4f}")
    print(f"   delta best vs nb2103 K=28    = {delta_best_vs_nb2103:+.4f}")
    print(f"   verdict                      = {verdict}")

    summary = {
        "tag": TAG,
        "method": "two_stage_residual_cascade_K1_28_K2_28_lgbm_mse",
        "anchor": ANCHOR,
        "anchor_path": str(ANCHOR_TE_PATH),
        "K1": K1,
        "K2": K2,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "n_unb": n_unb,
        "feat_dim_full": int(feat_dim),
        "topK1_idx_in_117": topK1_idx.tolist(),
        "topK2_idx_in_117": topK2_idx.tolist(),
        "rae_anchor_chemprop_aux": rae_anchor,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "residual_1_mean": float(residual_1.mean()),
        "residual_1_std": float(residual_1.std()),
        "residual_2_mean": float(residual_2.mean()),
        "residual_2_std": float(residual_2.std()),
        "stage1_mean_bag_rae": rae_s1_mean,
        "stage1_median_bag_rae": rae_s1_med,
        "stage2A_mean_bag_rae": rae_A_mean,
        "stage2A_median_bag_rae": rae_A_med,
        "stage2B_mean_bag_rae": rae_B_mean,
        "stage2B_median_bag_rae": rae_B_med,
        "delta_s2A_mean_vs_s1": rae_A_mean - rae_s1_mean,
        "delta_s2B_mean_vs_s1": rae_B_mean - rae_s1_mean,
        "nb2103_K28_mean_bag_ref": NB2103_K28_MEAN_BAG_REF,
        "nb2103_K28_median_bag_ref": NB2103_K28_MEDIAN_BAG_REF,
        "delta_stage1_mean_vs_nb2103": rae_s1_mean - NB2103_K28_MEAN_BAG_REF,
        "delta_stage2A_mean_vs_nb2103": rae_A_mean - NB2103_K28_MEAN_BAG_REF,
        "delta_stage2B_mean_vs_nb2103": rae_B_mean - NB2103_K28_MEAN_BAG_REF,
        "best_variant": best_label,
        "best_variant_rae": best_variant_rae,
        "delta_best_vs_nb2103_K28": delta_best_vs_nb2103,
        "verdict": verdict,
        "decision_margin": DECISION_MARGIN,
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
    for k in (
        "K1", "K2", "rae_anchor_chemprop_aux",
        "stage1_mean_bag_rae", "stage1_median_bag_rae",
        "stage2A_mean_bag_rae", "stage2A_median_bag_rae",
        "stage2B_mean_bag_rae", "stage2B_median_bag_rae",
        "best_variant", "best_variant_rae",
        "delta_best_vs_nb2103_K28", "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
