"""nb1612 -- PRE-unblind 6-way: nb1553 5-way + ChemBERTa SHAP-pruned slot.

Depends on nb1611 producing a valid ChemBERTa OOF.  Since nb1611 has not been
written/run, the K-grid for ChemBERTa is executed *inline* here on the same
chemprop_aux anchor / same residual / same shallow-LGBM Huber recipe as
nb1541 / nb1553.  Best K (by mean-bag pooled RAE) is selected and the
resulting mean-bag OOF becomes the 6th component.

PROTOCOL
    1. Anchor = chemprop_aux_te[unb_idx]  (PRE-unblind, in_RAE 0.6216).
    2. ChEMBL PXR kNN-5 (Tanimoto on Morgan-2048): same union as nb1553.
    3. For each FAMILY in [AtomPair-25, MACCS-20, Mordred-20,
       ChempropEmbed-20, Avalon-30, ChemBERTa-bestK]:
         a. Build FULL feature matrix on 253 = family + pred_chembl + sim.
         b. Train 1 seed-0 shallow LGBM Huber on residual; pull SHAP importance.
         c. Pick top-K family columns by importance.
         d. PRUNED matrix = top-K family + pred_chembl + sim.
         e. 5-seed bag (seeds [0,1,7,42,137]), KFold(n=5) cross-fit per seed
            on shallow LGBM Huber (depth=3, num_leaves=7, n_est=80, lr=0.05,
            huber_alpha=1.0, min_child_samples=20).
         f. Mean-bag pred_corr = anchor + mean_bag(residual_oof).
         g. Pool RAE.
       For ChemBERTa: K-grid {10, 15, 20, 30, 50}; pick K with best mean-bag RAE.
    4. Naive 1/6 mean blend across the 6 mean-bag corrected OOFs.
    5. 5-fold SLSQP cross-fit (simplex K=6) on the 6 corrected OOFs.
    6. Pool RAE for both.
    7. Verdict at 0.003 margin vs nb1554 ref = 0.5163.

Outputs:
    scripts/nb1612_preunblind_6way_chemberta.py      (this file)
    data/processed/nb1612_summary.json
    data/processed/nb1612_best_oof.npy               (253,) float32  best variant
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
from lightgbm import LGBMRegressor
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1612"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

SLSQP_FOLDS = 5
SLSQP_SEED = 42

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"
CHEMBERTA_TE_PATH = DATA_PROCESSED / "chemberta_test_emb.npy"  # (513, 768)
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")

EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

CHEMPROP_AUX_REF = 0.6216
NB1484_REF = 0.5231
NB1492_REF = 0.5260
NB1550_REF = NB1492_REF
NB1553_REF = 0.5225  # observed best blend from nb1553_summary.json
NB1554_REF = 0.5163  # nb1554 mean_bag pooled RAE (CatBoost 5-way K-tuned)
DECISION_MARGIN = 0.003

# nb1553 K-tuned values (per family)
TOP_K = {
    "AtomPair": 25,
    "MACCS": 20,
    "Mordred": 20,
    "ChempropEmbed": 20,
    "Avalon": 30,
}
# ChemBERTa K-grid swept inline (since nb1611 not available); pick best
CHEMBERTA_K_GRID = [10, 15, 20, 30, 50]

FAMILIES_FIXED_K = ["AtomPair", "MACCS", "Mordred", "ChempropEmbed", "Avalon"]


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
    """Same union as nb1472/nb1484/nb1492/nb1553."""
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


def _residual_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray,
                                 seed: int) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _compute_shap_importance(X: np.ndarray, residual: np.ndarray, seed: int = 0):
    mdl = LGBMRegressor(**_lgbm_params(seed))
    mdl.fit(X, residual)
    try:
        import shap
        explainer = shap.TreeExplainer(mdl)
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[0]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[..., 0]
        imp = np.abs(sv).mean(axis=0)
        return imp.astype(np.float64), "shap_tree_explainer"
    except Exception as e:
        print(f"   [shap] WARN: shap failed ({e}); falling back to LGBM gain")
        imp = mdl.booster_.feature_importance(importance_type="gain")
        return imp.astype(np.float64), "lgbm_gain_fallback"


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


def _load_chemberta_test(n_test: int) -> np.ndarray:
    if not CHEMBERTA_TE_PATH.exists():
        raise FileNotFoundError(f"ChemBERTa cache missing: {CHEMBERTA_TE_PATH}")
    X = np.load(CHEMBERTA_TE_PATH).astype(np.float32)
    if X.shape[0] != n_test:
        raise ValueError(f"ChemBERTa shape mismatch: {X.shape} vs n_test={n_test}")
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X


def _load_family_te(family: str, n_test: int) -> np.ndarray:
    if family == "AtomPair":
        p = ATOMPAIR_TE_PATH
        if not p.exists():
            raise FileNotFoundError(f"AtomPair cache missing: {p}")
        X = np.load(p)
        return X.astype(np.float32)
    if family == "MACCS":
        p = MACCS_TE_PATH
        if not p.exists():
            raise FileNotFoundError(f"MACCS cache missing: {p}")
        X = np.load(p)
        return X.astype(np.float32)
    if family == "Mordred":
        return _load_mordred_test(n_test)
    if family == "ChempropEmbed":
        p = CHEMPROP_EMBED_TE_PATH
        if not p.exists():
            raise FileNotFoundError(f"Chemprop embed cache missing: {p}")
        X = np.load(p).astype(np.float32)
        X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
        return X
    if family == "Avalon":
        p = AVALON_TE_PATH
        if not p.exists():
            raise FileNotFoundError(f"Avalon cache missing: {p}")
        X = np.load(p)
        return X.astype(np.float32)
    if family == "ChemBERTa":
        return _load_chemberta_test(n_test)
    raise ValueError(f"unknown family: {family}")


def _run_family_fixed_k(family: str, X_fam_unb: np.ndarray,
                        pred_chembl_unb: np.ndarray, mean_sim_unb: np.ndarray,
                        anchor: np.ndarray, y_unb: np.ndarray,
                        residual: np.ndarray, top_k: int) -> dict:
    print("\n" + "=" * 78)
    print(f"FAMILY = {family}   X_fam_unb shape = {X_fam_unb.shape}   "
          f"top_k = {top_k}")
    print("=" * 78)
    n_fam = int(X_fam_unb.shape[1])
    X_full = np.concatenate(
        [
            X_fam_unb,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    print(f"   FULL feature matrix: {X_full.shape}  "
          f"({family}-{n_fam} + pred_chembl + sim)")

    imp_full, imp_src = _compute_shap_importance(X_full, residual, seed=0)
    fam_imp = imp_full[:n_fam]
    top_k_eff = min(top_k, n_fam)
    top_order = np.argsort(-fam_imp)
    top_idx = top_order[:top_k_eff].astype(int)
    print(f"   importance source = {imp_src}")
    print(f"   pred_chembl importance = {imp_full[n_fam]:.4f}")
    print(f"   sim importance         = {imp_full[n_fam + 1]:.4f}")

    X_fam_pruned = X_fam_unb[:, top_idx]
    X_pruned = np.concatenate(
        [
            X_fam_pruned,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim_pruned = X_pruned.shape[1]
    print(f"   PRUNED feature matrix: {X_pruned.shape}")

    n_unb = len(y_unb)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae = []
    for s in RESID_SEEDS:
        resid_oof_s = _residual_cross_fit_one_seed(X_pruned, residual, s)
        pred_corr_s = anchor + resid_oof_s
        per_seed_corrected[RESID_SEEDS.index(s)] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        print(f"     seed {s:3d}: rae = {rae_s:.4f}")

    mean_bag_oof = per_seed_corrected.mean(axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    print(f"   mean_bag pooled RAE = {rae_mean_bag:.4f}")

    return {
        "family": family,
        "n_fam_bits": n_fam,
        "top_k": int(top_k_eff),
        "shap_source": imp_src,
        "feat_dim_pruned": int(feat_dim_pruned),
        "per_seed_rae": per_seed_rae,
        "rae_mean_bag": rae_mean_bag,
        "mean_bag_oof": mean_bag_oof.astype(np.float32),
    }


def _run_chemberta_kgrid(X_fam_unb: np.ndarray,
                         pred_chembl_unb: np.ndarray, mean_sim_unb: np.ndarray,
                         anchor: np.ndarray, y_unb: np.ndarray,
                         residual: np.ndarray) -> dict:
    """ChemBERTa K-grid sweep; pick K with best mean-bag pooled RAE.

    SHAP importance computed ONCE on the full (768 + 2) matrix, then top-K
    slicing for each K in CHEMBERTA_K_GRID (mirrors nb1541's recipe).
    """
    print("\n" + "=" * 78)
    print(f"FAMILY = ChemBERTa   X_fam_unb shape = {X_fam_unb.shape}   "
          f"K_grid = {CHEMBERTA_K_GRID}")
    print("=" * 78)
    n_fam = int(X_fam_unb.shape[1])
    X_full = np.concatenate(
        [
            X_fam_unb,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    print(f"   FULL feature matrix: {X_full.shape}  (ChemBERTa-{n_fam} + 2)")
    imp_full, imp_src = _compute_shap_importance(X_full, residual, seed=0)
    fam_imp = imp_full[:n_fam]
    n_nonzero = int((fam_imp > 0).sum())
    print(f"   importance source = {imp_src}   nonzero_imp = {n_nonzero}/{n_fam}")
    print(f"   pred_chembl importance = {imp_full[n_fam]:.4f}")
    print(f"   sim importance         = {imp_full[n_fam + 1]:.4f}")
    top_order = np.argsort(-fam_imp)

    n_unb = len(y_unb)
    grid_records = []
    best_K = None
    best_rae = np.inf
    best_mean_bag = None

    for K in CHEMBERTA_K_GRID:
        K_eff = min(K, n_fam)
        top_idx = top_order[:K_eff].astype(int)
        X_fam_pruned = X_fam_unb[:, top_idx]
        X_pruned = np.concatenate(
            [
                X_fam_pruned,
                pred_chembl_unb.reshape(-1, 1),
                mean_sim_unb.reshape(-1, 1),
            ],
            axis=1,
        ).astype(np.float32)
        per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
        per_seed_rae = []
        for s in RESID_SEEDS:
            resid_oof_s = _residual_cross_fit_one_seed(X_pruned, residual, s)
            pred_corr_s = anchor + resid_oof_s
            per_seed_corrected[RESID_SEEDS.index(s)] = pred_corr_s
            per_seed_rae.append(float(rae(y_unb, pred_corr_s)))
        mean_bag_oof = per_seed_corrected.mean(axis=0)
        rae_mean_bag = float(rae(y_unb, mean_bag_oof))
        print(f"   K={K_eff:3d}  pruned_dim={X_pruned.shape[1]:3d}  "
              f"mean_bag RAE = {rae_mean_bag:.4f}  "
              f"per_seed_mean = {float(np.mean(per_seed_rae)):.4f}")
        grid_records.append({
            "K": int(K_eff),
            "feat_dim_pruned": int(X_pruned.shape[1]),
            "per_seed_rae": per_seed_rae,
            "rae_mean_bag": rae_mean_bag,
        })
        if rae_mean_bag < best_rae:
            best_rae = rae_mean_bag
            best_K = int(K_eff)
            best_mean_bag = mean_bag_oof.astype(np.float32)

    print(f"   BEST K = {best_K}   mean_bag RAE = {best_rae:.4f}")
    return {
        "family": "ChemBERTa",
        "n_fam_bits": n_fam,
        "K_grid": [int(K) for K in CHEMBERTA_K_GRID],
        "grid_records": grid_records,
        "best_K": best_K,
        "top_k": best_K,
        "shap_source": imp_src,
        "rae_mean_bag": best_rae,
        "mean_bag_oof": best_mean_bag,
    }


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


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- PRE-unblind 6-way: nb1553 5-way + ChemBERTa SHAP-pruned")
    print(f"          seeds = {RESID_SEEDS}  folds = {RESID_FOLDS}")
    print(f"          refs: chemprop_aux={CHEMPROP_AUX_REF:.4f}  "
          f"nb1484={NB1484_REF:.4f}  nb1553={NB1553_REF:.4f}  "
          f"nb1554={NB1554_REF:.4f}")
    print(f"          decision margin = {DECISION_MARGIN}")
    print("=" * 78)

    # ---- Truth + indices ----
    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    # ---- Anchor (PRE-unblind chemprop_aux) ----
    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"chemprop_aux te file missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- ChEMBL pool + kNN ----
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

    # ---- Run 5 fixed-K families ----
    fam_results = []
    for family in FAMILIES_FIXED_K:
        X_fam_te = _load_family_te(family, n_test)
        X_fam_unb = X_fam_te[unb_idx].astype(np.float32)
        result = _run_family_fixed_k(
            family=family,
            X_fam_unb=X_fam_unb,
            pred_chembl_unb=pred_chembl_unb,
            mean_sim_unb=mean_sim_unb,
            anchor=anchor,
            y_unb=y_unb,
            residual=residual,
            top_k=TOP_K[family],
        )
        fam_results.append(result)

    # ---- Run ChemBERTa K-grid (inline nb1611) ----
    X_fam_te = _load_family_te("ChemBERTa", n_test)
    X_fam_unb = X_fam_te[unb_idx].astype(np.float32)
    chemberta_result = _run_chemberta_kgrid(
        X_fam_unb=X_fam_unb,
        pred_chembl_unb=pred_chembl_unb,
        mean_sim_unb=mean_sim_unb,
        anchor=anchor,
        y_unb=y_unb,
        residual=residual,
    )
    fam_results.append(chemberta_result)

    # Save ChemBERTa best_K OOF as the "nb1611_best_K_oof.npy" proxy
    np.save(DATA_PROCESSED / f"{TAG}_chemberta_best_K_oof.npy",
            chemberta_result["mean_bag_oof"])

    # ---- Naive 1/6 mean blend ----
    print("\n" + "=" * 78)
    print("6-WAY NAIVE 1/6 MEAN BLEND")
    print("=" * 78)
    per_family_mean_bag = np.stack(
        [r["mean_bag_oof"] for r in fam_results], axis=0
    )  # (6, 253)
    mean_oof = per_family_mean_bag.mean(axis=0)
    rae_mean = float(rae(y_unb, mean_oof))

    for r in fam_results:
        kname = r.get("top_k", "?")
        print(f"   {r['family']:<14s} (K={kname:>3}) "
              f"mean_bag RAE = {r['rae_mean_bag']:.4f}")
    print(f"   naive 1/6 mean blend RAE = {rae_mean:.4f}")

    # ---- 5-fold SLSQP cross-fit (K=6) ----
    print("\n" + "=" * 78)
    print(f"6-WAY 5-FOLD SLSQP CROSS-FIT (seed={SLSQP_SEED})")
    print("=" * 78)
    P_6 = per_family_mean_bag.T.astype(np.float64)  # (253, 6)
    slsqp_oof, slsqp_folds = _slsqp_cross_fit(
        P_6, y_unb, n_splits=SLSQP_FOLDS, seed=SLSQP_SEED
    )
    rae_slsqp = float(rae(y_unb, slsqp_oof))
    W = np.array([f["w"] for f in slsqp_folds])  # (5, 6)
    w_mean = W.mean(axis=0).tolist()
    w_std = W.std(axis=0).tolist()
    for f_rec in slsqp_folds:
        wlist = ", ".join(f"{w:.3f}" for w in f_rec["w"])
        print(f"   fold {f_rec['fold']}  w = [{wlist}]")
    print(f"   mean w over folds = "
          f"[{', '.join(f'{w:.3f}' for w in w_mean)}]")
    print(f"   std  w over folds = "
          f"[{', '.join(f'{w:.3f}' for w in w_std)}]")
    print(f"   5-fold SLSQP cross-fit RAE = {rae_slsqp:.4f}")

    # ---- Verdict ----
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"   anchor (chemprop_aux)        = {rae_anchor:.4f}")
    print(f"   nb1484 ref                   = {NB1484_REF:.4f}")
    print(f"   nb1553 ref                   = {NB1553_REF:.4f}")
    print(f"   nb1554 ref                   = {NB1554_REF:.4f}")
    print(f"   naive 1/6 mean blend RAE     = {rae_mean:.4f}")
    print(f"   5-fold SLSQP cross-fit RAE   = {rae_slsqp:.4f}")

    rae_best = min(rae_mean, rae_slsqp)
    which_best = "naive_1_6_mean" if rae_mean <= rae_slsqp else "slsqp_5fold"
    if which_best == "naive_1_6_mean":
        best_oof = mean_oof
    else:
        best_oof = slsqp_oof

    delta_naive_vs_nb1554 = rae_mean - NB1554_REF
    delta_slsqp_vs_nb1554 = rae_slsqp - NB1554_REF
    delta_best_vs_nb1554 = rae_best - NB1554_REF
    delta_naive_vs_nb1553 = rae_mean - NB1553_REF
    delta_slsqp_vs_nb1553 = rae_slsqp - NB1553_REF
    delta_best_vs_nb1553 = rae_best - NB1553_REF
    delta_best_vs_anchor = rae_best - rae_anchor

    beats_nb1554_naive = rae_mean < NB1554_REF - DECISION_MARGIN
    beats_nb1554_slsqp = rae_slsqp < NB1554_REF - DECISION_MARGIN
    beats_nb1554_best = rae_best < NB1554_REF - DECISION_MARGIN
    flat_vs_nb1554_best = abs(rae_best - NB1554_REF) < DECISION_MARGIN
    beats_nb1553_best = rae_best < NB1553_REF - DECISION_MARGIN

    if beats_nb1554_best:
        verdict = (
            f"PRE_UNBLIND_6WAY_CHEMBERTA_BEATS_NB1554_NEW_PRIMARY_via_{which_best}"
        )
    elif flat_vs_nb1554_best:
        verdict = (
            f"PRE_UNBLIND_6WAY_CHEMBERTA_FLAT_VS_NB1554_best_{which_best}"
        )
    elif beats_nb1553_best:
        verdict = (
            f"PRE_UNBLIND_6WAY_CHEMBERTA_BEATS_NB1553_BUT_WORSE_THAN_NB1554"
            f"_best_{which_best}"
        )
    else:
        verdict = "PRE_UNBLIND_6WAY_CHEMBERTA_FAILS_VS_NB1554"

    print(f"   d_naive_vs_nb1554   = {delta_naive_vs_nb1554:+.4f}")
    print(f"   d_slsqp_vs_nb1554   = {delta_slsqp_vs_nb1554:+.4f}")
    print(f"   d_best_vs_nb1554    = {delta_best_vs_nb1554:+.4f}   "
          f"(best = {which_best})")
    print(f"   beats_nb1554_naive  = {beats_nb1554_naive}")
    print(f"   beats_nb1554_slsqp  = {beats_nb1554_slsqp}")
    print(f"   beats_nb1554_best   = {beats_nb1554_best}")
    print(f"   verdict             = {verdict}")
    print("=" * 78)

    # ---- Save ----
    np.save(DATA_PROCESSED / f"{TAG}_best_oof.npy",
            best_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_best_oof.npy'}  "
          f"(variant = {which_best})")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "n_chembl_pool": int(len(pool)),
        "n_unb": int(n_unb),
        "n_test": int(n_test),
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "slsqp_folds": SLSQP_FOLDS,
        "slsqp_seed": SLSQP_SEED,
        "families_order": FAMILIES_FIXED_K + ["ChemBERTa"],
        "top_k_config_fixed": TOP_K,
        "chemberta_k_grid": CHEMBERTA_K_GRID,
        "chemberta_best_K": chemberta_result["best_K"],
        "chemberta_grid_records": chemberta_result["grid_records"],
        "families": [
            {
                "family": r["family"],
                "n_fam_bits": r["n_fam_bits"],
                "top_k": r.get("top_k"),
                "shap_source": r["shap_source"],
                "per_seed_rae": r.get("per_seed_rae"),
                "rae_mean_bag": r["rae_mean_bag"],
            }
            for r in fam_results
        ],
        "per_family_mean_bag_rae": {
            r["family"]: r["rae_mean_bag"] for r in fam_results
        },
        "rae_naive_mean_blend": rae_mean,
        "rae_slsqp_crossfit": rae_slsqp,
        "rae_best": rae_best,
        "best_variant": which_best,
        "slsqp_fold_weights": slsqp_folds,
        "slsqp_w_mean_over_folds": w_mean,
        "slsqp_w_std_over_folds": w_std,
        "delta_naive_vs_nb1553": delta_naive_vs_nb1553,
        "delta_slsqp_vs_nb1553": delta_slsqp_vs_nb1553,
        "delta_best_vs_nb1553": delta_best_vs_nb1553,
        "delta_naive_vs_nb1554": delta_naive_vs_nb1554,
        "delta_slsqp_vs_nb1554": delta_slsqp_vs_nb1554,
        "delta_best_vs_nb1554": delta_best_vs_nb1554,
        "delta_best_vs_anchor": delta_best_vs_anchor,
        "beats_nb1553_best": bool(beats_nb1553_best),
        "beats_nb1554_naive": bool(beats_nb1554_naive),
        "beats_nb1554_slsqp": bool(beats_nb1554_slsqp),
        "beats_nb1554_best": bool(beats_nb1554_best),
        "flat_vs_nb1554_best": bool(flat_vs_nb1554_best),
        "verdict": verdict,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "nb1484_ref": NB1484_REF,
        "nb1553_ref": NB1553_REF,
        "nb1554_ref": NB1554_REF,
        "decision_margin": DECISION_MARGIN,
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
        "n_unb", "n_chembl_pool", "rae_anchor_chemprop_aux",
        "chemberta_best_K",
        "per_family_mean_bag_rae",
        "rae_naive_mean_blend",
        "rae_slsqp_crossfit",
        "rae_best",
        "best_variant",
        "slsqp_w_mean_over_folds",
        "delta_naive_vs_nb1554",
        "delta_slsqp_vs_nb1554",
        "delta_best_vs_nb1554",
        "beats_nb1554_naive",
        "beats_nb1554_slsqp",
        "beats_nb1554_best",
        "flat_vs_nb1554_best",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
