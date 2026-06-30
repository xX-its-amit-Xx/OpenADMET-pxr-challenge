"""nb1523 -- Fresh SHAP K-grid for Mordred on PRE-unblind path (chemprop_aux anchor).

Hypothesis:
    nb1412 found Mordred K=20 best (RAE 0.5180) on the nb1070 anchor.
    The PRE-unblind path uses chemprop_aux as anchor (te[unb_idx], in_RAE
    0.6216).  The residual distribution is different (mean -0.13, std 0.70
    vs nb1070 residual which is centered near 0), so the SHAP-pruned Mordred
    optimum K may shift.  Sweep K in {10, 15, 20, 25, 30, 35, 40, 50} on the
    SAME pipeline as nb1472's Mordred slot, keep ChEMBL pred + sim cols
    identical, and pick the K that minimizes 5-seed mean-bag pooled RAE.

Protocol (PRE-unblind):
    1. Anchor = chemprop_aux_te[unb_idx]  (PRE-unblind, in_RAE ~0.6216).
    2. ChEMBL PXR kNN-5 (Tanimoto on Morgan-2048): same union as nb1472.
    3. Compute SHAP feature importance of LGBM Huber on FULL Mordred-1533
       + pred_chembl + sim (seed=0; fallback to LGBM gain).
    4. For each K in K_GRID, slice top-K Mordred col indices.  Build
       (K + 2)-col feature matrix on 253 unblind rows.
    5. 5-seed bag (seeds [0, 1, 7, 42, 137]), 5-fold cross-fit per K.
    6. Pool mean-bag RAE per K.
    7. Verdict at 0.003 margin vs nb1472 Mordred slot (0.5384).

Outputs:
    scripts/nb1523_mordred_K_grid_preunblind.py        (this file)
    data/processed/nb1523_summary.json
    data/processed/nb1523_best_K_oof.npy               (253,) float32
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
from lightgbm import LGBMRegressor
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1523"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

CHEMPROP_AUX_REF = 0.6216
NB1472_MORDRED_REF = 0.5384      # nb1472 Mordred-30 slot (chemprop_aux anchor)
NB1412_BEST_REF = 0.5180         # nb1412 K=20 best (nb1070 anchor)
DECISION_MARGIN = 0.003

K_GRID = [10, 15, 20, 25, 30, 35, 40, 50]


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
    """Same union as nb1472 (chemprop_aux PRE-unblind path)."""
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
    """Train one global LGBM on residual; return (importance vector, source_tag)."""
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
    """Load cached Mordred test matrix (513 x 1533).  Median-impute NaN/Inf."""
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


def _evaluate_K(K: int, X_mord_unb: np.ndarray, top_col_order: np.ndarray,
                pred_chembl_unb: np.ndarray, mean_sim_unb: np.ndarray,
                residual: np.ndarray, anchor: np.ndarray, y_unb: np.ndarray,
                rae_anchor: float) -> dict:
    """Build (K+2)-col feature matrix and run 5-seed bag x 5-fold cross-fit."""
    top_col_idx = top_col_order[:K].astype(int)
    X_mord_unb_pruned = X_mord_unb[:, top_col_idx]
    X_unb_pruned = np.concatenate(
        [
            X_mord_unb_pruned,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim = X_unb_pruned.shape[1]
    n_unb = len(residual)

    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    for i, s in enumerate(RESID_SEEDS):
        resid_oof_s = _residual_cross_fit_one_seed(X_unb_pruned, residual, s)
        pred_corr_s = anchor + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        per_seed_rae.append(float(rae(y_unb, pred_corr_s)))

    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))

    return {
        "K": int(K),
        "feat_dim": int(feat_dim),
        "top_col_idx": [int(c) for c in top_col_idx.tolist()],
        "per_seed_rae": per_seed_rae,
        "rae_per_seed_mean": float(np.mean(per_seed_rae)),
        "rae_per_seed_std": float(np.std(per_seed_rae)),
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_nb1472": rae_mean_bag - NB1472_MORDRED_REF,
        "delta_mean_bag_vs_anchor": rae_mean_bag - rae_anchor,
        "mean_bag_oof": mean_bag_oof.astype(np.float32),
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Fresh SHAP K-grid for Mordred  (PRE-unblind, anchor={ANCHOR})")
    print(f"          K_GRID = {K_GRID}")
    print(f"          nb1472 Mordred slot ref = {NB1472_MORDRED_REF:.4f}")
    print(f"          nb1412 K=20 (nb1070 anchor) ref = {NB1412_BEST_REF:.4f}")
    print(f"          seeds = {RESID_SEEDS}  folds = {RESID_FOLDS}  "
          f"margin = {DECISION_MARGIN}")
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
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(f"anchor shape mismatch: {te_anchor_513.shape}")
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Mordred-1533 (full test then unblind slice) ----
    print(f"\n[feat] loading cached Mordred test matrix from {MORDRED_DIR}")
    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    n_mord = int(X_mord_te.shape[1])
    print(f"[feat] X_mord_te shape = {X_mord_te.shape}  (n_mordred={n_mord})")
    X_mord_unb = X_mord_te[unb_idx].astype(np.float32)

    # ---- ChEMBL pool + kNN feature build (513-level) ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL + kNN feature build (same union as nb1472)")
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

    # ---- Build FULL feature matrix and compute SHAP importance ----
    X_unb_full = np.concatenate(
        [
            X_mord_unb,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim_full = X_unb_full.shape[1]
    print(f"\n   full feature matrix: {X_unb_full.shape}  "
          f"(Mordred-{n_mord} + pred_chembl + sim)")

    print("\n" + "-" * 78)
    print(f"SHAP IMPORTANCE FRAME (1 global LGBM, seed=0, full {feat_dim_full} cols)")
    print(f"  residual: y_unb - chemprop_aux_te[unb_idx]")
    print("-" * 78)
    imp_full, imp_src = _compute_shap_importance(X_unb_full, residual, seed=0)
    print(f"   importance source        = {imp_src}")
    print(f"   pred_chembl importance   = {imp_full[n_mord]:.4f}")
    print(f"   sim importance           = {imp_full[n_mord + 1]:.4f}")
    mord_imp = imp_full[:n_mord]
    top_col_order_full = np.argsort(-mord_imp).astype(int)
    n_nonzero = int((mord_imp > 0).sum())
    print(f"   Mordred cols with nonzero importance: {n_nonzero}/{n_mord}")

    # ---- Sanity-check ranking vs nb1412 (different anchor; expect different) ----
    nb1412_summary_p = DATA_PROCESSED / "nb1412_summary.json"
    rank_match_vs_nb1412 = None
    set_overlap_top20_nb1412 = None
    if nb1412_summary_p.exists():
        with open(nb1412_summary_p, "r") as f:
            nb1412_summary = json.load(f)
        # nb1412 K=20 record holds the top-20 col_idx
        nb1412_K20 = None
        for rec in nb1412_summary.get("per_K_records", []):
            if rec.get("K") == 20:
                nb1412_K20 = rec.get("top_col_idx", [])
                break
        if nb1412_K20:
            local_top20 = top_col_order_full[:20].tolist()
            rank_match_vs_nb1412 = (local_top20 == nb1412_K20)
            set_overlap_top20_nb1412 = len(set(local_top20) & set(nb1412_K20))
            print(f"   ranking match vs nb1412 (nb1070 anchor) top-20: "
                  f"{rank_match_vs_nb1412}")
            print(f"   set overlap (top-20): "
                  f"{set_overlap_top20_nb1412}/20  "
                  f"(expected != 20 because anchor differs)")

    # ---- Sweep K ----
    print("\n" + "-" * 78)
    print(f"K-GRID SWEEP  ({len(K_GRID)} values)")
    print("-" * 78)
    per_K_records: list[dict] = []
    per_K_oof: dict[int, np.ndarray] = {}
    for K in K_GRID:
        t_k0 = time.time()
        rec = _evaluate_K(
            K=K,
            X_mord_unb=X_mord_unb,
            top_col_order=top_col_order_full,
            pred_chembl_unb=pred_chembl_unb,
            mean_sim_unb=mean_sim_unb,
            residual=residual,
            anchor=anchor,
            y_unb=y_unb,
            rae_anchor=rae_anchor,
        )
        oof = rec.pop("mean_bag_oof")
        per_K_oof[K] = oof
        per_K_records.append(rec)
        dt = time.time() - t_k0
        print(f"   K={K:3d}  feat_dim={rec['feat_dim']:3d}  "
              f"per_seed_mean={rec['rae_per_seed_mean']:.4f}  "
              f"mean_bag={rec['rae_mean_bag']:.4f}  "
              f"median_bag={rec['rae_median_bag']:.4f}  "
              f"d_vs_nb1472={rec['delta_mean_bag_vs_nb1472']:+.4f}  "
              f"[{dt:.1f}s]")

    # ---- Pick best K ----
    rae_by_K = {r["K"]: r["rae_mean_bag"] for r in per_K_records}
    best_K = min(rae_by_K, key=lambda k: rae_by_K[k])
    best_rae = rae_by_K[best_K]
    best_oof = per_K_oof[best_K]

    beats_nb1472 = best_rae < NB1472_MORDRED_REF - DECISION_MARGIN
    ties_nb1472 = abs(best_rae - NB1472_MORDRED_REF) <= DECISION_MARGIN
    beats_anchor = best_rae < rae_anchor - DECISION_MARGIN
    if beats_nb1472:
        verdict = f"K_GRID_BEATS_NB1472_MORDRED_AT_K={best_K}_NEW_PREUNBLIND_CANDIDATE"
    elif ties_nb1472:
        verdict = f"K_GRID_FLAT_VS_NB1472_MORDRED_AT_K={best_K}"
    elif beats_anchor:
        verdict = (f"K_GRID_BEATS_ANCHOR_BUT_LOSES_TO_NB1472_MORDRED_"
                   f"BEST_K={best_K}")
    else:
        verdict = f"K_GRID_LOSES_TO_NB1472_MORDRED_BEST_K={best_K}"

    print("\n" + "-" * 78)
    print("K-GRID SUMMARY")
    print("-" * 78)
    print(f"   {'K':>4}  {'feat_dim':>8}  {'per_seed_mean':>13}  "
          f"{'mean_bag':>9}  {'median_bag':>10}  {'d_vs_nb1472':>11}")
    for r in sorted(per_K_records, key=lambda x: x["K"]):
        print(f"   {r['K']:>4}  {r['feat_dim']:>8}  "
              f"{r['rae_per_seed_mean']:>13.4f}  "
              f"{r['rae_mean_bag']:>9.4f}  {r['rae_median_bag']:>10.4f}  "
              f"{r['delta_mean_bag_vs_nb1472']:>+11.4f}")
    print(f"\n   anchor (chemprop_aux te[unb_idx]) RAE = {rae_anchor:.4f}")
    print(f"   nb1472 Mordred-30 ref RAE             = {NB1472_MORDRED_REF:.4f}")
    print(f"   nb1412 K=20 (nb1070 anchor) ref RAE   = {NB1412_BEST_REF:.4f}")
    print(f"   best K (sweep)                        = {best_K}  "
          f"RAE = {best_rae:.4f}")
    print(f"   delta(best - nb1472 Mordred)          = "
          f"{best_rae - NB1472_MORDRED_REF:+.4f}  (margin = {DECISION_MARGIN})")
    print(f"   delta(best - anchor)                  = "
          f"{best_rae - rae_anchor:+.4f}")
    print(f"   verdict                               = {verdict}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_best_K_oof.npy", best_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_best_K_oof.npy'}  "
          f"(best K={best_K})")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "rae_anchor_chemprop_aux": rae_anchor,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "data_source": "mordred_cached_nb1030 + local_chembl_caches_union",
        "n_chembl_pool": int(len(pool)),
        "n_unb": int(n_unb),
        "n_test": int(n_test),
        "n_mordred_cols": int(n_mord),
        "shap_importance_source": imp_src,
        "n_mordred_cols_nonzero_imp": n_nonzero,
        "pred_chembl_importance": float(imp_full[n_mord]),
        "sim_importance": float(imp_full[n_mord + 1]),
        "rank_match_vs_nb1412_top20": rank_match_vs_nb1412,
        "set_overlap_top20_vs_nb1412": set_overlap_top20_nb1412,
        "feat_dim_full": int(feat_dim_full),
        "K_grid": K_GRID,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "lgbm_max_depth": 3,
        "lgbm_num_leaves": 7,
        "lgbm_n_estimators": 80,
        "lgbm_learning_rate": 0.05,
        "lgbm_min_child_samples": 20,
        "lgbm_huber_alpha": 1.0,
        "nb1472_mordred_ref": NB1472_MORDRED_REF,
        "nb1412_best_ref": NB1412_BEST_REF,
        "decision_margin": DECISION_MARGIN,
        "per_K_records": per_K_records,
        "rae_by_K": {str(k): v for k, v in rae_by_K.items()},
        "best_K": int(best_K),
        "best_K_rae_mean_bag": float(best_rae),
        "delta_best_K_vs_nb1472_mordred": float(best_rae - NB1472_MORDRED_REF),
        "delta_best_K_vs_anchor": float(best_rae - rae_anchor),
        "beats_nb1472_mordred": bool(beats_nb1472),
        "ties_nb1472_mordred": bool(ties_nb1472),
        "beats_anchor": bool(beats_anchor),
        "verdict": verdict,
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
        "rae_anchor_chemprop_aux",
        "n_chembl_pool", "n_mordred_cols", "shap_importance_source",
        "n_mordred_cols_nonzero_imp",
        "rank_match_vs_nb1412_top20", "set_overlap_top20_vs_nb1412",
        "K_grid", "rae_by_K",
        "best_K", "best_K_rae_mean_bag",
        "delta_best_K_vs_nb1472_mordred", "delta_best_K_vs_anchor",
        "beats_nb1472_mordred", "ties_nb1472_mordred", "beats_anchor",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
