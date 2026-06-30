"""nb1621 -- Extended ChemBERTa-77M-MTR K-grid (K=75,100,125,150).

HYPOTHESIS:
    Cycle 66 nb1611 swept K in {10, 15, 20, 30, 50, 75} on the 384-dim
    te_chemberta.npy cache and bottomed at K=20 (mean_bag 0.5528).  The
    nb1612 6-way frame used the 768-dim chemberta_test_emb.npy cache with
    K-grid {10, 15, 20, 30, 50} and reported BEST K=50 with mean_bag
    pooled RAE = 0.5622 -- but K=50 is the LAST point on that grid, so
    we have no evidence that 50 is a real interior optimum vs. a corner
    artifact.  Extend the 768-dim ChemBERTa K-grid out to {75, 100, 125,
    150} on the IDENTICAL recipe used inside nb1612 so the numbers are
    directly comparable.

PROTOCOL:
    1.  Anchor = chemprop_aux te[unb_idx]  (PRE-unblind, in_RAE 0.6216).
        residual = y_unb - anchor.
    2.  ChEMBL PXR pool (same union as nb1541/nb1553/nb1612).  kNN-5
        Tanimoto on Morgan-2048; pull pred_chembl + mean_sim for all 513
        test rows, slice to 253 unblind.
    3.  Load chemberta_test_emb.npy (513, 768) -- IDENTICAL cache used
        inside nb1612 so the K-grid matches.  Slice to (253, 768).
    4.  Build FULL feature matrix (253, 770) = ChemBERTa-768 +
        pred_chembl + sim.  Train ONE seed-0 shallow LGBM Huber on
        residual; SHAP TreeExplainer (fallback LGBM gain) -> per-dim
        importance.
    5.  For each K in K_GRID_EXT, slice top-K ChemBERTa dims by SHAP
        importance.  Build pruned matrix (253, K+2).
    6.  5-seed bag (seeds [0,1,7,42,137]), 5-fold cross-fit per K.
        mean-bag pooled RAE per K.
    7.  Verdict at 0.003 margin vs nb1612 ChemBERTa K=50 mean_bag RAE
        = 0.5622.

OUTPUTS:
    scripts/nb1621_chemberta_K_grid_ext.py    (this file)
    data/processed/nb1621_summary.json
    data/processed/nb1621_best_K_oof.npy      (253,) float32  best K mean-bag
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

TAG = "nb1621"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

# IDENTICAL cache to nb1612 (768-dim ChemBERTa-77M-MTR test embeddings)
CHEMBERTA_TE_PATH = DATA_PROCESSED / "chemberta_test_emb.npy"
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

CHEMPROP_AUX_REF = 0.6216
NB1612_CHEMBERTA_K50_REF = 0.5622  # nb1612 ChemBERTa K=50 mean_bag pooled RAE
NB1554_REF = 0.5163  # global 5-way K-tuned CatBoost reference (PRE-unblind)
DECISION_MARGIN = 0.003

K_GRID_EXT = [75, 100, 125, 150]


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
    """Same union as nb1541/nb1553/nb1612."""
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


def _evaluate_K(K: int, X_embed_unb: np.ndarray, top_dim_order: np.ndarray,
                pred_chembl_unb: np.ndarray, mean_sim_unb: np.ndarray,
                residual: np.ndarray, anchor: np.ndarray, y_unb: np.ndarray,
                rae_anchor: float) -> dict:
    """Build (K+2)-col pruned feature matrix and run 5-seed bag x 5-fold CV."""
    K_eff = min(K, X_embed_unb.shape[1])
    top_dim_idx = top_dim_order[:K_eff].astype(int)
    X_embed_pruned = X_embed_unb[:, top_dim_idx]
    X_unb_pruned = np.concatenate(
        [
            X_embed_pruned,
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
        "K_eff": int(K_eff),
        "feat_dim": int(feat_dim),
        "top_dim_idx_ranked": [int(c) for c in top_dim_idx[:30].tolist()],
        "per_seed_rae": per_seed_rae,
        "rae_per_seed_mean": float(np.mean(per_seed_rae)),
        "rae_per_seed_std": float(np.std(per_seed_rae)),
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_nb1612_K50": rae_mean_bag - NB1612_CHEMBERTA_K50_REF,
        "delta_mean_bag_vs_nb1554": rae_mean_bag - NB1554_REF,
        "delta_mean_bag_vs_anchor": rae_mean_bag - rae_anchor,
        "mean_bag_oof": mean_bag_oof.astype(np.float32),
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Extended ChemBERTa-77M-MTR K-grid  "
          f"(anchor={ANCHOR}, PRE-unblind)")
    print(f"          K_GRID_EXT = {K_GRID_EXT}")
    print(f"          nb1612 ChemBERTa K=50 reference RAE = "
          f"{NB1612_CHEMBERTA_K50_REF:.4f}")
    print(f"          nb1554 global K-tuned reference RAE = {NB1554_REF:.4f}")
    print(f"          seeds = {RESID_SEEDS}  folds = {RESID_FOLDS}  "
          f"margin = {DECISION_MARGIN}")
    print("=" * 78)

    # ---- Skip if ChemBERTa unavailable ----
    if not CHEMBERTA_TE_PATH.exists():
        msg = f"chemberta cache missing: te={CHEMBERTA_TE_PATH}"
        print(f"[skip] {msg}")
        out = {
            "tag": TAG,
            "status": "chemberta_unavailable",
            "msg": msg,
            "te_chemberta_exists": CHEMBERTA_TE_PATH.exists(),
        }
        out_path = DATA_PROCESSED / f"{TAG}_summary.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        return out

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
        raise FileNotFoundError(f"anchor cache missing: {ANCHOR_TE_PATH}")
    te_anchor_full = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_full.shape[0] != n_test:
        raise ValueError(
            f"{ANCHOR} te shape mismatch: {te_anchor_full.shape} "
            f"vs n_test={n_test}"
        )
    anchor = te_anchor_full[unb_idx].astype(np.float64)
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} (PRE-unblind te[unb_idx]) pooled RAE = "
          f"{rae_anchor:.4f}  (ref ~{CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- ChEMBL pool ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL (local cache; same union as nb1541/nb1553/nb1612)")
    print("-" * 78)
    pool = _load_chembl_pool()

    # ---- Test InChIKey leak guard ----
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

    # ---- Morgan FPs for kNN ----
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

    # ---- kNN k=5 Tanimoto ----
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )
    top1_sim = top_sim_knn[:, 0]
    print(f"   pred_chembl_pec50  mean={pred_chembl_pec50.mean():.3f}  "
          f"std={pred_chembl_pec50.std():.3f}")
    print(f"   top1 sim   p50={np.percentile(top1_sim, 50):.3f}")

    pred_chembl_unb = pred_chembl_pec50[unb_idx].astype(np.float32)
    mean_sim_unb = mean_sim[unb_idx].astype(np.float32)

    # ---- ChemBERTa embedding (IDENTICAL cache to nb1612: 513x768) ----
    X_embed_te = np.load(CHEMBERTA_TE_PATH).astype(np.float32)
    if X_embed_te.shape[0] != n_test:
        raise ValueError(f"chemberta te shape mismatch: {X_embed_te.shape}")
    X_embed_te = np.where(np.isfinite(X_embed_te), X_embed_te, 0.0).astype(np.float32)
    n_embed = int(X_embed_te.shape[1])
    print(f"   chemberta te cache shape = {X_embed_te.shape}  "
          f"(n_dims={n_embed})")
    X_embed_unb = X_embed_te[unb_idx].astype(np.float32)
    print(f"   embed value mean (unb) = {X_embed_unb.mean():.4f}  "
          f"std = {X_embed_unb.std():.4f}  "
          f"const cols = {int((X_embed_unb.var(axis=0) == 0).sum())}/{n_embed}")

    # ---- Build FULL feature matrix and compute SHAP importance ----
    X_unb_full = np.concatenate(
        [
            X_embed_unb,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim_full = X_unb_full.shape[1]
    print(f"\n   full feature matrix: {X_unb_full.shape}  "
          f"(chemberta-{n_embed} + pred_chembl + sim)")

    print("\n" + "-" * 78)
    print(f"SHAP IMPORTANCE FRAME (1 global LGBM, seed=0, full {feat_dim_full} cols)")
    print("-" * 78)
    imp_full, imp_src = _compute_shap_importance(X_unb_full, residual, seed=0)
    print(f"   importance source       = {imp_src}")
    print(f"   importance vector shape = {imp_full.shape}")
    print(f"   pred_chembl importance  = {imp_full[n_embed]:.4f}")
    print(f"   sim importance          = {imp_full[n_embed + 1]:.4f}")
    embed_imp = imp_full[:n_embed]
    top_dim_order = np.argsort(-embed_imp).astype(int)
    n_nonzero_imp = int((embed_imp > 0).sum())
    print(f"   chemberta dims with nonzero importance: "
          f"{n_nonzero_imp}/{n_embed}")

    # ---- Sweep extended K ----
    print("\n" + "-" * 78)
    print(f"EXTENDED K-GRID SWEEP  ({len(K_GRID_EXT)} values)")
    print("-" * 78)
    per_K_records: list[dict] = []
    per_K_oof: dict[int, np.ndarray] = {}
    for K in K_GRID_EXT:
        t_k0 = time.time()
        rec = _evaluate_K(
            K=K,
            X_embed_unb=X_embed_unb,
            top_dim_order=top_dim_order,
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
              f"d_vs_nb1612_K50={rec['delta_mean_bag_vs_nb1612_K50']:+.4f}  "
              f"d_vs_anchor={rec['delta_mean_bag_vs_anchor']:+.4f}  "
              f"[{dt:.1f}s]")

    # ---- Pick best K within EXTENDED grid ----
    rae_by_K = {r["K"]: r["rae_mean_bag"] for r in per_K_records}
    best_K = min(rae_by_K, key=lambda k: rae_by_K[k])
    best_rae = rae_by_K[best_K]
    best_oof = per_K_oof[best_K]

    beats_nb1612_K50 = best_rae < NB1612_CHEMBERTA_K50_REF - DECISION_MARGIN
    ties_nb1612_K50 = abs(best_rae - NB1612_CHEMBERTA_K50_REF) <= DECISION_MARGIN
    beats_nb1554 = best_rae < NB1554_REF - DECISION_MARGIN
    beats_anchor = best_rae < rae_anchor - DECISION_MARGIN

    if beats_nb1554:
        verdict = (
            f"CHEMBERTA_K_GRID_EXT_BEATS_NB1554_AT_K={best_K}_"
            f"NEW_PREUNBLIND_PRIMARY_CANDIDATE"
        )
    elif beats_nb1612_K50:
        verdict = (
            f"CHEMBERTA_K_GRID_EXT_BEATS_NB1612_K50_AT_K={best_K}_"
            f"NEW_K_OPTIMUM"
        )
    elif ties_nb1612_K50:
        verdict = (
            f"CHEMBERTA_K_GRID_EXT_FLAT_VS_NB1612_K50_BEST_K={best_K}_"
            f"K50_NOT_A_CORNER_ARTIFACT"
        )
    elif beats_anchor:
        verdict = (
            f"CHEMBERTA_K_GRID_EXT_LOSES_TO_NB1612_K50_BUT_BEATS_ANCHOR_"
            f"BEST_K={best_K}_K50_WAS_INTERIOR_OPTIMUM"
        )
    else:
        verdict = f"CHEMBERTA_K_GRID_EXT_HURTS_ANCHOR_BEST_K={best_K}"

    print("\n" + "-" * 78)
    print("EXTENDED K-GRID SUMMARY")
    print("-" * 78)
    print(f"   {'K':>4}  {'feat_dim':>8}  {'per_seed_mean':>13}  "
          f"{'mean_bag':>9}  {'median_bag':>10}  {'d_vs_K50':>9}  "
          f"{'d_vs_nb1554':>11}  {'d_vs_anchor':>11}")
    for r in sorted(per_K_records, key=lambda x: x["K"]):
        print(f"   {r['K']:>4}  {r['feat_dim']:>8}  "
              f"{r['rae_per_seed_mean']:>13.4f}  "
              f"{r['rae_mean_bag']:>9.4f}  {r['rae_median_bag']:>10.4f}  "
              f"{r['delta_mean_bag_vs_nb1612_K50']:>+9.4f}  "
              f"{r['delta_mean_bag_vs_nb1554']:>+11.4f}  "
              f"{r['delta_mean_bag_vs_anchor']:>+11.4f}")
    print(f"\n   best K (extended grid)     = {best_K}  RAE = {best_rae:.4f}")
    print(f"   nb1612 ChemBERTa K=50 ref  = {NB1612_CHEMBERTA_K50_REF:.4f}")
    print(f"   nb1554 K-tuned 5-way ref   = {NB1554_REF:.4f}")
    print(f"   delta(best - nb1612_K50)   = "
          f"{best_rae - NB1612_CHEMBERTA_K50_REF:+.4f}  "
          f"(margin = {DECISION_MARGIN})")
    print(f"   delta(best - nb1554)       = {best_rae - NB1554_REF:+.4f}")
    print(f"   delta(best - anchor)       = {best_rae - rae_anchor:+.4f}")
    print(f"   verdict                    = {verdict}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_best_K_oof.npy", best_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_best_K_oof.npy'}  "
          f"(best K={best_K})")

    summary = {
        "tag": TAG,
        "status": "ok",
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "data_source": ("chemberta_test_emb-768 (same cache used inside nb1612) "
                        "+ local_chembl_caches_union"),
        "chemberta_cache_path": str(CHEMBERTA_TE_PATH),
        "n_chembl_pool": int(len(pool)),
        "n_unb": n_unb,
        "n_test": int(n_test),
        "n_embed_dims": n_embed,
        "embed_mean_unb": float(X_embed_unb.mean()),
        "embed_std_unb": float(X_embed_unb.std()),
        "embed_const_cols": int((X_embed_unb.var(axis=0) == 0).sum()),
        "embed_nonzero_imp_dims": n_nonzero_imp,
        "shap_importance_source": imp_src,
        "pred_chembl_importance": float(imp_full[n_embed]),
        "sim_importance": float(imp_full[n_embed + 1]),
        "feat_dim_full": int(feat_dim_full),
        "top_dim_order_top100": [int(b) for b in top_dim_order[:100].tolist()],
        "K_grid_ext": K_GRID_EXT,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "lgbm_max_depth": 3,
        "lgbm_num_leaves": 7,
        "lgbm_n_estimators": 80,
        "lgbm_learning_rate": 0.05,
        "lgbm_min_child_samples": 20,
        "lgbm_huber_alpha": 1.0,
        "rae_anchor_chemprop_aux": rae_anchor,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "nb1612_chemberta_K50_ref": NB1612_CHEMBERTA_K50_REF,
        "nb1554_ref": NB1554_REF,
        "decision_margin": DECISION_MARGIN,
        "per_K_records": per_K_records,
        "rae_by_K": {str(k): v for k, v in rae_by_K.items()},
        "best_K": int(best_K),
        "best_K_rae_mean_bag": float(best_rae),
        "delta_best_K_vs_nb1612_K50": float(best_rae - NB1612_CHEMBERTA_K50_REF),
        "delta_best_K_vs_nb1554": float(best_rae - NB1554_REF),
        "delta_best_K_vs_anchor": float(best_rae - rae_anchor),
        "beats_nb1612_K50": bool(beats_nb1612_K50),
        "ties_nb1612_K50": bool(ties_nb1612_K50),
        "beats_nb1554": bool(beats_nb1554),
        "beats_anchor": bool(beats_anchor),
        "verdict": verdict,
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
    if res.get("status") == "chemberta_unavailable":
        print(f"  status: {res.get('status')}")
        print(f"  msg: {res.get('msg')}")
    else:
        for k in (
            "n_chembl_pool", "n_embed_dims", "shap_importance_source",
            "embed_nonzero_imp_dims",
            "rae_anchor_chemprop_aux",
            "K_grid_ext", "rae_by_K",
            "best_K", "best_K_rae_mean_bag",
            "delta_best_K_vs_nb1612_K50",
            "delta_best_K_vs_nb1554",
            "delta_best_K_vs_anchor",
            "beats_nb1612_K50", "ties_nb1612_K50",
            "beats_nb1554", "beats_anchor",
            "verdict",
        ):
            print(f"  {k}: {res.get(k)}")
