"""nb1103 -- Permutation importance feature ranking on 117-col 5-way matrix.

HYPOTHESIS:
    nb2103.K=28 (mean_bag RAE = 0.4737) selects features via TreeExplainer SHAP
    on a single LGBM(MSE) seed=0 fit.  nb2159 showed seed-bias in single-seed
    SHAP rankings.  sklearn permutation_importance is MODEL-AGNOSTIC and
    measures *predictive contribution* (drop in score when a column is
    shuffled on held-out folds), not internal split-attribution.  If perm-imp
    top-28 picks different / better features than SHAP top-28 on the same
    117-col 5-way K-tuned matrix, it should beat nb2103.K=28 at
    decision_margin = 0.005 on the residual cross-fit.

PROTOCOL:
    1. Anchor   = chemprop_aux te[unb_idx] (PRE-unblind, in_RAE 0.6216).
       residual = y_unb - anchor.
    2. Rebuild the same 117-col 5-way K-tuned feature matrix used by
       nb2063/nb2103/nb2159 (AtomPair / MACCS / Mordred / ChempropEmbed /
       Avalon + ChEMBL kNN).
    3. BASELINE: fit one LGBM(MSE) K=117 on the FULL 253 residual at seed=0
       (no CV -- importance seed).
    4. PERMUTATION IMPORTANCE:
         - 5-fold KFold cross-fit (shuffle=True, seed=0).
         - For each held-out fold, fit LGBM(MSE) on tr, run
           sklearn.inspection.permutation_importance on va with
           n_repeats=10, random_state=0, scoring="neg_mean_squared_error",
           n_jobs=2.
         - Aggregate mean-importance across folds (mean of per-fold mean).
    5. Rank features by aggregated mean perm-imp; take top-28.
    6. OVERLAP: compare perm-imp-28 vs nb2103 SHAP-28 (intersection size +
       Jaccard).
    7. EVAL: restrict X to perm-imp-28; 5-seed bag (seeds 0,1,7,42,137) of
       LGBM(MSE) with KFold(n=5, shuffle=True) cross-fit per seed -- same
       protocol as nb2103.
    8. DECISION_MARGIN = 0.005 vs nb2103.K=28 baseline (0.4737 mean_bag).
       If passes -> fresh-seed verification on seeds [11, 23, 91, 222, 555].

Outputs:
    scripts/nb1103_perm_importance.py
    data/processed/nb1103_summary.json
    data/processed/nb1103_perm_importance_full117.npy   (117,)  float32
    data/processed/nb1103_top28_idx_perm.npy             (28,)   int32
    data/processed/nb1103_mean_bag_oof_K28.npy           (253,)  float32
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
from sklearn.inspection import permutation_importance
import lightgbm as lgb
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1103"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]
PERM_FOLDS = 5
PERM_N_REPEATS = 10
PERM_SEED = 0
TARGET_K = 28
FRESH_SEEDS = [11, 23, 91, 222, 555]

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

# References (PRE-unblind path)
CHEMPROP_AUX_REF = 0.6216
NB2103_K28_MEAN_BAG_REF = 0.4737
NB2103_K28_MEDIAN_BAG_REF = 0.4698
DECISION_MARGIN = 0.005


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
    """Same union as nb2063/nb2103/nb2159."""
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
    """LGBM(MSE) -- identical to nb2063/nb2103/nb2159."""
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


def _build_feature_matrix(n_test, unb_idx, test_smiles):
    """Build the same 117-col 5-way K-tuned matrix as nb2063/nb2103/nb2159."""
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

    n_top_maccs = int(len(top_maccs_bit_idx))
    n_top_mord = int(len(top_mord_col_idx))
    n_top_ap = int(len(top_ap_bit_idx))
    n_top_embed = int(len(top_embed_col_idx))
    n_top_avalon = int(len(top_avalon_bit_idx))
    print(f"[reuse] top-{n_top_ap}     AtomPair bits (nb1524 K={K_AP_best})")
    print(f"[reuse] top-{n_top_maccs}     MACCS bits  (nb1352)")
    print(f"[reuse] top-{n_top_mord}     Mordred cols (nb1523 K={K_Mord_best})")
    print(f"[reuse] top-{n_top_embed}     ChempropEmbed dims (nb1541 K={K_Embed_best})")
    print(f"[reuse] top-{n_top_avalon}     Avalon bits (nb1392 SHAP K=30)")

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
    print(f"   pool: {n_before} -> {len(pool)}  (dropped {n_before - len(pool)})")
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
    print(f"\n   COMBINED 5-WAY K-TUNED matrix: {X_unb.shape}")

    feat_names: list[str] = []
    feat_family: list[str] = []
    for b in top_ap_bit_idx:
        feat_names.append(f"AtomPair_bit_{int(b)}")
        feat_family.append("AtomPair")
    for b in top_maccs_bit_idx:
        feat_names.append(f"MACCS_bit_{int(b)}")
        feat_family.append("MACCS")
    for c in top_mord_col_idx:
        feat_names.append(f"Mordred_col_{int(c)}")
        feat_family.append("Mordred")
    for d in top_embed_col_idx:
        feat_names.append(f"ChempropEmbed_dim_{int(d)}")
        feat_family.append("ChempropEmbed")
    for b in top_avalon_bit_idx:
        feat_names.append(f"Avalon_bit_{int(b)}")
        feat_family.append("Avalon")
    feat_names.append("pred_chembl_pec50")
    feat_family.append("ChEMBL_kNN")
    feat_names.append("mean_sim")
    feat_family.append("ChEMBL_kNN")
    assert len(feat_names) == feat_dim

    fb = {
        "atompair": n_top_ap,
        "maccs": n_top_maccs,
        "mordred": n_top_mord,
        "chemprop_embed": n_top_embed,
        "avalon": n_top_avalon,
        "chembl_knn": 2,
        "total": feat_dim,
    }
    return X_unb, feat_names, feat_family, fb, len(pool)


def _perm_importance_crossfit(X: np.ndarray, residual: np.ndarray) -> tuple:
    """5-fold cross-fit permutation importance.  Returns (mean_imp, per_fold)."""
    n, p = X.shape
    kf = KFold(n_splits=PERM_FOLDS, shuffle=True, random_state=PERM_SEED)
    per_fold = np.zeros((PERM_FOLDS, p), dtype=np.float64)
    for f_i, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n))):
        ts = time.time()
        mdl = lgb.LGBMRegressor(**_lgbm_params(PERM_SEED))
        mdl.fit(X[tr_loc], residual[tr_loc])
        r = permutation_importance(
            mdl, X[va_loc], residual[va_loc],
            n_repeats=PERM_N_REPEATS,
            random_state=PERM_SEED,
            scoring="neg_mean_squared_error",
            n_jobs=2,
        )
        per_fold[f_i] = r.importances_mean
        print(f"   [perm] fold {f_i+1}/{PERM_FOLDS}  "
              f"n_va={len(va_loc)}  "
              f"max_imp={r.importances_mean.max():+.4f}  "
              f"wall = {time.time() - ts:.1f}s")
    mean_imp = per_fold.mean(axis=0)
    return mean_imp.astype(np.float32), per_fold.astype(np.float32)


def _bag_eval(X_K: np.ndarray, residual: np.ndarray, anchor: np.ndarray,
              y_unb: np.ndarray, seeds: list[int], tag: str):
    """5-seed bag cross-fit -- returns (per_seed_rae, mean_bag_oof, mean_bag_rae, median_bag_rae)."""
    n_unb = len(y_unb)
    per_seed_corrected = np.zeros((len(seeds), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    rae_anchor = float(rae(y_unb, anchor))
    per_seed_records = []
    for i, s in enumerate(seeds):
        ts = time.time()
        resid_oof_s = _residual_cross_fit_one_seed(X_K, residual, s)
        pred_corr_s = anchor + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_chemprop_aux": rae_s - rae_anchor,
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   [{tag}] seed={s:3d}: rae={rae_s:.4f}  "
              f"(d_anchor = {rae_s - rae_anchor:+.4f})  "
              f"wall = {time.time() - ts:.1f}s")
    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))
    return (per_seed_rae, mean_bag_oof, median_bag_oof,
            rae_mean_bag, rae_median_bag, per_seed_records)


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- PERM-IMPORTANCE K={TARGET_K} on 117-col 5-way K-tuned matrix")
    print(f"          anchor={ANCHOR}  seeds={RESID_SEEDS}  folds={RESID_FOLDS}")
    print(f"          ref: nb2103.K=28 mean_bag RAE = {NB2103_K28_MEAN_BAG_REF:.4f}")
    print(f"          decision_margin = {DECISION_MARGIN}")
    print("=" * 78)

    # ---- Pin nb2103.K=28 baseline (and SHAP top-28 for overlap) ----
    if not NB2103_SUMMARY.exists():
        raise FileNotFoundError(f"missing {NB2103_SUMMARY} -- run nb2103 first")
    with open(NB2103_SUMMARY) as f:
        nb2103_sum = json.load(f)
    nb2103_k28_record = None
    for r in nb2103_sum.get("per_K_records", []):
        if int(r.get("K", -1)) == TARGET_K:
            nb2103_k28_record = r
            break
    if nb2103_k28_record is None:
        raise KeyError(f"nb2103 has no K={TARGET_K} record")
    nb2103_k28_mean_bag = float(nb2103_k28_record["rae_mean_bag"])
    nb2103_k28_median_bag = float(nb2103_k28_record["rae_median_bag"])
    shap_top28_idx = np.array(nb2103_k28_record["top_K_idx_in_117"], dtype=int)
    print(f"[ref] nb2103.K=28 mean_bag_rae   = {nb2103_k28_mean_bag:.4f}")
    print(f"[ref] nb2103.K=28 median_bag_rae = {nb2103_k28_median_bag:.4f}")
    print(f"[ref] nb2103.K=28 SHAP top-28 idx (sorted): "
          f"{sorted(shap_top28_idx.tolist())}")

    if NB2063_SHAP_IMP.exists():
        shap_imp_full117 = np.load(NB2063_SHAP_IMP).astype(np.float32)
    else:
        shap_imp_full117 = None

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

    # ---- Build 117-col matrix ----
    X_unb, feat_names, feat_family, fb, n_pool = _build_feature_matrix(
        n_test, unb_idx, test_smiles
    )
    feat_dim = X_unb.shape[1]
    if feat_dim != 117:
        print(f"[warn] feat_dim={feat_dim} != 117 -- proceeding anyway")

    # ---- Baseline LGBM K=117 (full feature set, single fit) ----
    print("\n" + "-" * 78)
    print(f"BASELINE LGBM K={feat_dim} (single seed=0 fit on full residual)")
    print("-" * 78)
    base_mdl = lgb.LGBMRegressor(**_lgbm_params(PERM_SEED))
    base_mdl.fit(X_unb, residual)
    base_train_pred = base_mdl.predict(X_unb)
    base_train_mse = float(np.mean((base_train_pred - residual) ** 2))
    print(f"   baseline train MSE (residual) = {base_train_mse:.4f}")

    # ---- Permutation importance via 5-fold cross-fit ----
    print("\n" + "-" * 78)
    print(f"PERMUTATION IMPORTANCE  ({PERM_FOLDS}-fold cross-fit, "
          f"n_repeats={PERM_N_REPEATS}, scoring=neg_MSE)")
    print("-" * 78)
    perm_imp_mean, perm_imp_per_fold = _perm_importance_crossfit(X_unb, residual)
    perm_rank_order = np.argsort(-perm_imp_mean).astype(np.int32)
    perm_top28_idx = perm_rank_order[:TARGET_K].astype(np.int32)
    print(f"\n   perm-imp top-{TARGET_K} (rank order): "
          f"{perm_top28_idx.tolist()}")
    print(f"   perm-imp top-{TARGET_K} (sorted):     "
          f"{sorted(perm_top28_idx.tolist())}")

    # save raw arrays
    np.save(DATA_PROCESSED / f"{TAG}_perm_importance_full117.npy",
            perm_imp_mean)
    np.save(DATA_PROCESSED / f"{TAG}_perm_importance_per_fold.npy",
            perm_imp_per_fold)
    np.save(DATA_PROCESSED / f"{TAG}_top28_idx_perm.npy", perm_top28_idx)

    # ---- Overlap with SHAP-28 ----
    perm_set = set(perm_top28_idx.tolist())
    shap_set = set(shap_top28_idx.tolist())
    overlap = sorted(perm_set & shap_set)
    union = sorted(perm_set | shap_set)
    jaccard = float(len(overlap) / len(union)) if union else 0.0
    perm_only = sorted(perm_set - shap_set)
    shap_only = sorted(shap_set - perm_set)
    print("\n" + "-" * 78)
    print(f"OVERLAP  perm-{TARGET_K} vs SHAP-{TARGET_K}")
    print("-" * 78)
    print(f"   intersection    = {len(overlap)}/{TARGET_K}  "
          f"(jaccard = {jaccard:.3f})")
    print(f"   shared idx      = {overlap}")
    print(f"   perm-only       = {perm_only}")
    print(f"   shap-only       = {shap_only}")
    perm_top28_families = {}
    for i in perm_top28_idx:
        fam = feat_family[i]
        perm_top28_families[fam] = perm_top28_families.get(fam, 0) + 1
    print(f"   perm-top28 family breakdown: {perm_top28_families}")

    # ---- Eval perm-imp top-28 ----
    print("\n" + "-" * 78)
    print(f"EVAL  perm-imp K={TARGET_K}  ({len(RESID_SEEDS)}-seed bag, "
          f"{RESID_FOLDS}-fold cross-fit)")
    print("-" * 78)
    X_perm28 = X_unb[:, perm_top28_idx].astype(np.float32)
    (per_seed_rae, mean_bag_oof, median_bag_oof,
     rae_mean_bag, rae_median_bag,
     per_seed_records) = _bag_eval(
        X_perm28, residual, anchor, y_unb, RESID_SEEDS, tag="perm28"
    )
    per_seed_arr = np.array(per_seed_rae)
    print(f"\n   per-seed RAE   = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   per-seed mean  = {per_seed_arr.mean():.4f}  "
          f"std = {per_seed_arr.std():.4f}")
    print(f"   mean-bag RAE   = {rae_mean_bag:.4f}  "
          f"(d_anchor = {rae_mean_bag - rae_anchor:+.4f}, "
          f"d_nb2103 = {rae_mean_bag - nb2103_k28_mean_bag:+.4f})")
    print(f"   median-bag RAE = {rae_median_bag:.4f}  "
          f"(d_nb2103 = {rae_median_bag - nb2103_k28_median_bag:+.4f})")
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof_K{TARGET_K}.npy",
            mean_bag_oof.astype(np.float32))

    delta_vs_nb2103 = rae_mean_bag - nb2103_k28_mean_bag
    beats_nb2103 = rae_mean_bag < nb2103_k28_mean_bag - DECISION_MARGIN
    flat_vs_nb2103 = abs(delta_vs_nb2103) < DECISION_MARGIN
    beats_anchor = rae_mean_bag < rae_anchor - DECISION_MARGIN
    if beats_nb2103:
        verdict = f"BEATS_NB2103_K28_AT_PERM_K{TARGET_K}"
    elif flat_vs_nb2103:
        verdict = f"FLAT_VS_NB2103_K28_AT_PERM_K{TARGET_K}"
    elif beats_anchor:
        verdict = f"BEATS_ANCHOR_BUT_WORSE_THAN_NB2103_K28"
    else:
        verdict = f"DOES_NOT_BEAT_NB2103_K28"
    print(f"   verdict        = {verdict}")

    # ---- Fresh-seed verification if passes ----
    fresh_records = None
    fresh_per_seed_rae = None
    fresh_mean_bag_rae = None
    fresh_median_bag_rae = None
    fresh_verdict = None
    if beats_nb2103:
        print("\n" + "-" * 78)
        print(f"FRESH-SEED VERIFICATION  seeds={FRESH_SEEDS}")
        print("-" * 78)
        (fresh_per_seed_rae, fresh_mean_bag_oof, fresh_median_bag_oof,
         fresh_mean_bag_rae, fresh_median_bag_rae,
         fresh_records) = _bag_eval(
            X_perm28, residual, anchor, y_unb, FRESH_SEEDS, tag="fresh"
        )
        fresh_arr = np.array(fresh_per_seed_rae)
        print(f"\n   fresh per-seed RAE  = "
              f"[{', '.join(f'{r:.4f}' for r in fresh_per_seed_rae)}]")
        print(f"   fresh per-seed mean = {fresh_arr.mean():.4f}  "
              f"std = {fresh_arr.std():.4f}")
        print(f"   fresh mean-bag RAE  = {fresh_mean_bag_rae:.4f}  "
              f"(d_nb2103 = {fresh_mean_bag_rae - nb2103_k28_mean_bag:+.4f})")
        print(f"   fresh median-bag    = {fresh_median_bag_rae:.4f}")
        if fresh_mean_bag_rae < nb2103_k28_mean_bag - DECISION_MARGIN:
            fresh_verdict = "FRESH_SEED_CONFIRMS_BEATS_NB2103"
        elif abs(fresh_mean_bag_rae - nb2103_k28_mean_bag) < DECISION_MARGIN:
            fresh_verdict = "FRESH_SEED_FLAT_VS_NB2103"
        else:
            fresh_verdict = "FRESH_SEED_DOES_NOT_BEAT_NB2103"
        print(f"   fresh verdict       = {fresh_verdict}")

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "method": ("perm_importance_K28_on_117col_5way_K_tuned"),
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "data_source": ("same 117-col 5-way K-tuned matrix as "
                        "nb2063/nb2103/nb2159"),
        "model_family": "LightGBM",
        "lgbm_params": _lgbm_params(0),
        "perm_n_repeats": PERM_N_REPEATS,
        "perm_folds": PERM_FOLDS,
        "perm_seed": PERM_SEED,
        "perm_scoring": "neg_mean_squared_error",
        "target_K": TARGET_K,
        "feat_dim_full": int(feat_dim),
        "feat_breakdown_full": fb,
        "n_chembl_pool": int(n_pool),
        "n_unb": int(n_unb),
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "baseline_train_mse_K117": base_train_mse,
        "perm_top28_idx": perm_top28_idx.tolist(),
        "perm_top28_families": perm_top28_families,
        "shap_top28_idx_ref_nb2103": shap_top28_idx.tolist(),
        "overlap_idx_perm_vs_shap": overlap,
        "perm_only_idx": perm_only,
        "shap_only_idx": shap_only,
        "overlap_size": int(len(overlap)),
        "jaccard_perm_vs_shap_K28": jaccard,
        "per_seed_rae_perm28": per_seed_rae,
        "per_seed_records_perm28": per_seed_records,
        "rae_per_seed_mean_perm28": float(per_seed_arr.mean()),
        "rae_per_seed_median_perm28": float(np.median(per_seed_arr)),
        "rae_per_seed_std_perm28": float(per_seed_arr.std()),
        "rae_mean_bag_perm28": rae_mean_bag,
        "rae_median_bag_perm28": rae_median_bag,
        "delta_mean_bag_vs_chemprop_aux": rae_mean_bag - rae_anchor,
        "delta_mean_bag_vs_nb2103_K28": delta_vs_nb2103,
        "beats_chemprop_aux": bool(beats_anchor),
        "beats_nb2103_K28": bool(beats_nb2103),
        "flat_vs_nb2103_K28": bool(flat_vs_nb2103),
        "verdict": verdict,
        "fresh_seeds": FRESH_SEEDS if beats_nb2103 else None,
        "fresh_per_seed_rae": fresh_per_seed_rae,
        "fresh_per_seed_records": fresh_records,
        "fresh_mean_bag_rae_perm28": fresh_mean_bag_rae,
        "fresh_median_bag_rae_perm28": fresh_median_bag_rae,
        "fresh_verdict": fresh_verdict,
        "pre_unblind_clean": True,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "nb2103_K28_mean_bag_ref": NB2103_K28_MEAN_BAG_REF,
        "nb2103_K28_median_bag_ref": NB2103_K28_MEDIAN_BAG_REF,
        "decision_margin": DECISION_MARGIN,
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
        "target_K", "feat_dim_full", "n_chembl_pool",
        "rae_anchor_chemprop_aux",
        "baseline_train_mse_K117",
        "overlap_size", "jaccard_perm_vs_shap_K28",
        "perm_top28_idx",
        "rae_per_seed_mean_perm28", "rae_per_seed_std_perm28",
        "rae_mean_bag_perm28", "rae_median_bag_perm28",
        "delta_mean_bag_vs_nb2103_K28",
        "beats_nb2103_K28", "verdict",
        "fresh_mean_bag_rae_perm28", "fresh_verdict",
        "pre_unblind_clean", "wall_sec",
    ):
        print(f"  {k}: {res.get(k)}")
