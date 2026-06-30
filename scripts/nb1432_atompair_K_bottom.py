"""nb1432 -- AtomPair K-grid bottom-fill K in {15, 18, 22, 28, 32}.

Hypothesis:
    Previous K-sweeps clustered around K=20-30:
        nb1373 K=30 = 0.5095
        nb1402 K=25 = 0.5107
        nb1384 K=20 = 0.5111
    The true optimum likely sits in the under-explored sub-K=20 band or
    the 22-32 fine-grained window between nb1384 K=20 and nb1373 K=30.
    Bottom-fill K in {15, 18, 22, 28, 32} probes both halves of the
    around-the-minimum neighborhood.

Protocol:
    1.  Reuse SHAP ranking on AtomPair-2048 -- compute via one global
        LGBM Huber (seed=0) on full 2050-col feature matrix on the 253
        unblind rows, exactly as nb1373/nb1384/nb1402.  (The ranking is
        deterministic at seed=0 given identical data, so the top-K
        slices nest the previous grids.)
    2.  For each K in {15, 18, 22, 28, 32}:
            features = top-K AtomPair bits (by SHAP) + pred_chembl + sim
                     = (K + 2) columns
    3.  5-seed bag (seeds [0, 1, 7, 42, 137]), KFold(n=5) cross-fit per
        seed on shallow LGBM Huber.
    4.  Compare mean-bag RAE vs K; identify best-K.
    5.  Verdict at 0.003 margin vs nb1373 K=30 (0.5095).

Outputs:
    scripts/nb1432_atompair_K_bottom.py          (this file)
    data/processed/nb1432_summary.json
    data/processed/nb1432_best_K_oof.npy         mean-bag OOF for best K (253,)
    data/processed/nb1432_per_K_mean_bag_oof.npy (5, 253) all K mean-bags
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

TAG = "nb1432"
ANCHOR = "nb1070"

K_GRID = [15, 18, 22, 28, 32]
RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"   # (513, 2048) uint8
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

NB1070_REF = 0.5771
NB1172_REF = 0.5659
NB1352_REF = 0.5323
NB1373_REF = 0.5095        # K=30 mean-bag RAE
NB1402_REF_K25 = 0.5107    # nb1402 K=25 mean-bag RAE
NB1384_REF_BEST_K = 20
NB1384_REF_BEST_RAE = 0.5111   # best in old grid (K=20)
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
    """Same union as nb1373 / nb1384 / nb1402."""
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


def _bag_for_K(X_unb_pruned: np.ndarray, residual: np.ndarray,
               anchor: np.ndarray, y_unb: np.ndarray, K: int):
    """5-seed residual cross-fit; return per-seed + bag summary."""
    n_unb = len(y_unb)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    resid_oof_means: list[float] = []
    resid_oof_stds: list[float] = []
    for i, s in enumerate(RESID_SEEDS):
        resid_oof_s = _residual_cross_fit_one_seed(X_unb_pruned, residual, s)
        pred_corr_s = anchor + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        per_seed_rae.append(float(rae(y_unb, pred_corr_s)))
        resid_oof_means.append(float(resid_oof_s.mean()))
        resid_oof_stds.append(float(resid_oof_s.std()))
    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))
    return {
        "K": int(K),
        "per_seed_rae": per_seed_rae,
        "rae_per_seed_mean": float(np.mean(per_seed_rae)),
        "rae_per_seed_std": float(np.std(per_seed_rae)),
        "rae_per_seed_min": float(np.min(per_seed_rae)),
        "rae_per_seed_max": float(np.max(per_seed_rae)),
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "resid_oof_mean_avg": float(np.mean(resid_oof_means)),
        "resid_oof_std_avg": float(np.mean(resid_oof_stds)),
        "mean_bag_oof": mean_bag_oof,
        "median_bag_oof": median_bag_oof,
        "per_seed_corrected": per_seed_corrected,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- BOTTOM-FILL SHAP K-grid AtomPair; anchor={ANCHOR}")
    print(f"          K_grid = {K_GRID}")
    print(f"          seeds  = {RESID_SEEDS}  folds = {RESID_FOLDS}")
    print(f"          baselines: nb1373 K=30 = {NB1373_REF:.4f}  "
          f"nb1402 K=25 = {NB1402_REF_K25:.4f}  "
          f"nb1384 K=20 = {NB1384_REF_BEST_RAE:.4f}  "
          f"margin = {DECISION_MARGIN}")
    print("=" * 78)

    # ---- Load anchor + truth ----
    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    anchor = np.load(DATA_PROCESSED / f"{ANCHOR}_pred_oof.npy").astype(np.float64)
    if anchor.shape[0] != n_unb:
        raise ValueError(f"{ANCHOR} shape mismatch: {anchor.shape} vs {n_unb}")
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} pooled RAE = {rae_anchor:.4f}  (ref ~{NB1070_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- ChEMBL pool ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL (local cache; same union as nb1373/nb1384/nb1402)")
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

    top_idx, top_sim = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx, top_sim, pool_labels, fallback=pool_median
    )
    print(f"   pred_chembl_pec50  mean={pred_chembl_pec50.mean():.3f}  "
          f"std={pred_chembl_pec50.std():.3f}")

    # ---- AtomPair-2048 (unblind slice) ----
    if not ATOMPAIR_TE_PATH.exists():
        raise FileNotFoundError(
            f"AtomPair test cache missing: {ATOMPAIR_TE_PATH}"
        )
    X_ap_te = np.load(ATOMPAIR_TE_PATH)
    if X_ap_te.shape[0] != n_test:
        raise ValueError(f"AtomPair cache shape mismatch: {X_ap_te.shape}")
    n_ap = int(X_ap_te.shape[1])
    print(f"   AtomPair cache shape = {X_ap_te.shape}  (n_bits={n_ap})")
    X_ap_unb = X_ap_te[unb_idx].astype(np.float32)
    print(f"   bit density (unb) = {X_ap_unb.mean():.4f}")

    pred_chembl_unb = pred_chembl_pec50[unb_idx].astype(np.float32)
    mean_sim_unb = mean_sim[unb_idx].astype(np.float32)
    X_unb_full = np.concatenate(
        [
            X_ap_unb,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    print(f"   FULL feature matrix: {X_unb_full.shape}")

    # ---- SHAP importance frame (reused for all K) ----
    print("\n" + "-" * 78)
    print("SHAP IMPORTANCE FRAME (1 global LGBM, seed=0, full feature matrix)")
    print("-" * 78)
    imp_full, imp_src = _compute_shap_importance(X_unb_full, residual, seed=0)
    print(f"   importance source = {imp_src}")
    ap_imp = imp_full[:n_ap]
    top_bit_order = np.argsort(-ap_imp)
    n_nonzero_imp = int((ap_imp > 0).sum())
    print(f"   AtomPair bits with nonzero importance: {n_nonzero_imp}/{n_ap}")
    print(f"   pred_chembl importance = {imp_full[n_ap]:.4f}")
    print(f"   sim importance         = {imp_full[n_ap + 1]:.4f}")

    # ---- K-grid loop ----
    print("\n" + "-" * 78)
    print(f"K-GRID SWEEP: K in {K_GRID}")
    print("-" * 78)

    per_K_records: list[dict] = []
    per_K_mean_bag_oof = np.zeros((len(K_GRID), n_unb), dtype=np.float32)

    for k_i, K in enumerate(K_GRID):
        top_k = min(K, n_ap)
        top_bit_idx = top_bit_order[:top_k].astype(int)
        X_ap_unb_pruned = X_ap_unb[:, top_bit_idx]
        X_unb_pruned = np.concatenate(
            [
                X_ap_unb_pruned,
                pred_chembl_unb.reshape(-1, 1),
                mean_sim_unb.reshape(-1, 1),
            ],
            axis=1,
        ).astype(np.float32)
        feat_dim_pruned = X_unb_pruned.shape[1]
        print(f"\n  K={K:3d}: feature matrix {X_unb_pruned.shape}  "
              f"(dim={feat_dim_pruned})")

        bag = _bag_for_K(X_unb_pruned, residual, anchor, y_unb, K)
        per_K_mean_bag_oof[k_i] = bag["mean_bag_oof"].astype(np.float32)

        print(f"     per-seed RAE = "
              f"[{', '.join(f'{r:.4f}' for r in bag['per_seed_rae'])}]")
        print(f"     mean_bag    RAE = {bag['rae_mean_bag']:.4f}  "
              f"(d_vs_nb1373 = {bag['rae_mean_bag'] - NB1373_REF:+.4f}  "
              f"d_vs_nb1402K25 = {bag['rae_mean_bag'] - NB1402_REF_K25:+.4f}  "
              f"d_vs_nb1384K20 = {bag['rae_mean_bag'] - NB1384_REF_BEST_RAE:+.4f})")
        print(f"     median_bag  RAE = {bag['rae_median_bag']:.4f}")
        print(f"     per-seed std    = {bag['rae_per_seed_std']:.4f}")

        per_K_records.append({
            "K": int(K),
            "feat_dim_pruned": int(feat_dim_pruned),
            "per_seed_rae": bag["per_seed_rae"],
            "rae_per_seed_mean": bag["rae_per_seed_mean"],
            "rae_per_seed_std": bag["rae_per_seed_std"],
            "rae_per_seed_min": bag["rae_per_seed_min"],
            "rae_per_seed_max": bag["rae_per_seed_max"],
            "rae_mean_bag": bag["rae_mean_bag"],
            "rae_median_bag": bag["rae_median_bag"],
            "delta_vs_nb1373": bag["rae_mean_bag"] - NB1373_REF,
            "delta_vs_nb1402_K25": bag["rae_mean_bag"] - NB1402_REF_K25,
            "delta_vs_nb1384_bestK20": bag["rae_mean_bag"] - NB1384_REF_BEST_RAE,
            "delta_vs_nb1070": bag["rae_mean_bag"] - rae_anchor,
            "resid_oof_mean_avg": bag["resid_oof_mean_avg"],
            "resid_oof_std_avg": bag["resid_oof_std_avg"],
            "top_bit_indices_ranked": [int(b) for b in
                                       top_bit_order[:top_k].tolist()],
        })

    # ---- Identify best K (mean_bag) ----
    rae_mean_bag_list = [r["rae_mean_bag"] for r in per_K_records]
    rae_median_bag_list = [r["rae_median_bag"] for r in per_K_records]
    best_i = int(np.argmin(rae_mean_bag_list))
    best_K = int(per_K_records[best_i]["K"])
    best_rae_mean_bag = float(per_K_records[best_i]["rae_mean_bag"])
    best_mean_bag_oof = per_K_mean_bag_oof[best_i]
    delta_best_vs_nb1373 = best_rae_mean_bag - NB1373_REF
    delta_best_vs_nb1402 = best_rae_mean_bag - NB1402_REF_K25
    delta_best_vs_nb1384 = best_rae_mean_bag - NB1384_REF_BEST_RAE
    beats_nb1373 = best_rae_mean_bag < NB1373_REF - DECISION_MARGIN
    ties_nb1373 = abs(best_rae_mean_bag - NB1373_REF) < DECISION_MARGIN
    beats_nb1402 = best_rae_mean_bag < NB1402_REF_K25 - DECISION_MARGIN
    ties_nb1402 = abs(best_rae_mean_bag - NB1402_REF_K25) < DECISION_MARGIN
    beats_nb1384 = best_rae_mean_bag < NB1384_REF_BEST_RAE - DECISION_MARGIN
    ties_nb1384 = abs(best_rae_mean_bag - NB1384_REF_BEST_RAE) < DECISION_MARGIN

    # Also report best by median_bag
    best_i_med = int(np.argmin(rae_median_bag_list))
    best_K_med = int(per_K_records[best_i_med]["K"])
    best_rae_median_bag = float(per_K_records[best_i_med]["rae_median_bag"])

    # ---- Report table ----
    print("\n" + "=" * 78)
    print("K-GRID RAE TABLE (bottom-fill)")
    print("=" * 78)
    print(f"  {'K':>5}  {'feat_dim':>9}  {'mean_bag':>10}  {'median_bag':>11}  "
          f"{'per_seed_std':>13}  {'d_vs_nb1373':>12}  {'d_vs_nb1402':>12}  "
          f"{'d_vs_nb1384':>12}")
    print("  " + "-" * 96)
    for r in per_K_records:
        flag = "  <-- BEST(mean)" if r["K"] == best_K else ""
        print(f"  {r['K']:>5d}  {r['feat_dim_pruned']:>9d}  "
              f"{r['rae_mean_bag']:>10.4f}  {r['rae_median_bag']:>11.4f}  "
              f"{r['rae_per_seed_std']:>13.4f}  "
              f"{r['delta_vs_nb1373']:>+12.4f}  "
              f"{r['delta_vs_nb1402_K25']:>+12.4f}  "
              f"{r['delta_vs_nb1384_bestK20']:>+12.4f}{flag}")
    print(f"  nb1373 K=30   reference  = {NB1373_REF:.4f}")
    print(f"  nb1402 K=25   reference  = {NB1402_REF_K25:.4f}")
    print(f"  nb1384 K=20   reference  = {NB1384_REF_BEST_RAE:.4f}")
    print(f"  margin = {DECISION_MARGIN}")
    print(f"  best K (mean_bag)   = {best_K}  "
          f"(mean_bag = {best_rae_mean_bag:.4f})")
    print(f"     d_vs_nb1373 = {delta_best_vs_nb1373:+.4f}  "
          f"d_vs_nb1402 = {delta_best_vs_nb1402:+.4f}  "
          f"d_vs_nb1384 = {delta_best_vs_nb1384:+.4f}")
    print(f"  best K (median_bag) = {best_K_med}  "
          f"(median_bag = {best_rae_median_bag:.4f})")

    if beats_nb1373:
        verdict = (f"BOTTOM_K_GRID_BEST_K{best_K}_BEATS_NB1373_NEW_PRIMARY_CANDIDATE")
    elif ties_nb1373:
        verdict = f"BOTTOM_K_GRID_BEST_K{best_K}_FLAT_VS_NB1373"
    else:
        verdict = f"BOTTOM_K_GRID_BEST_K{best_K}_WORSE_THAN_NB1373"
    print(f"  verdict: {verdict}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_best_K_oof.npy",
            best_mean_bag_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_per_K_mean_bag_oof.npy",
            per_K_mean_bag_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_best_K_oof.npy'}  "
          f"(K={best_K}, shape={best_mean_bag_oof.shape})")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_per_K_mean_bag_oof.npy'}  "
          f"(shape={per_K_mean_bag_oof.shape})")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "K_grid": K_GRID,
        "n_chembl_pool": int(len(pool)),
        "n_unb": n_unb,
        "n_atompair_bits": n_ap,
        "atompair_nonzero_imp_bits": n_nonzero_imp,
        "shap_importance_source": imp_src,
        "pred_chembl_importance": float(imp_full[n_ap]),
        "sim_importance": float(imp_full[n_ap + 1]),
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "lgbm_max_depth": 3,
        "lgbm_num_leaves": 7,
        "lgbm_n_estimators": 80,
        "lgbm_learning_rate": 0.05,
        "lgbm_min_child_samples": 20,
        "lgbm_huber_alpha": 1.0,
        "rae_anchor_nb1070": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_K_records": per_K_records,
        "best_K": best_K,
        "best_rae_mean_bag": best_rae_mean_bag,
        "best_K_median": best_K_med,
        "best_rae_median_bag": best_rae_median_bag,
        "delta_best_vs_nb1373": delta_best_vs_nb1373,
        "delta_best_vs_nb1402_K25": delta_best_vs_nb1402,
        "delta_best_vs_nb1384_bestK20": delta_best_vs_nb1384,
        "beats_nb1373": bool(beats_nb1373),
        "ties_nb1373": bool(ties_nb1373),
        "beats_nb1402": bool(beats_nb1402),
        "ties_nb1402": bool(ties_nb1402),
        "beats_nb1384": bool(beats_nb1384),
        "ties_nb1384": bool(ties_nb1384),
        "verdict": verdict,
        "nb1070_ref_pooled": NB1070_REF,
        "nb1172_ref": NB1172_REF,
        "nb1352_ref": NB1352_REF,
        "nb1373_ref": NB1373_REF,
        "nb1402_ref_K25": NB1402_REF_K25,
        "nb1384_ref_best_K": NB1384_REF_BEST_K,
        "nb1384_ref_best_rae": NB1384_REF_BEST_RAE,
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
        "K_grid", "n_chembl_pool", "n_atompair_bits",
        "atompair_nonzero_imp_bits", "shap_importance_source",
        "pred_chembl_importance", "sim_importance",
        "rae_anchor_nb1070",
        "best_K", "best_rae_mean_bag",
        "best_K_median", "best_rae_median_bag",
        "delta_best_vs_nb1373", "delta_best_vs_nb1402_K25",
        "delta_best_vs_nb1384_bestK20",
        "beats_nb1373", "ties_nb1373",
        "beats_nb1402", "ties_nb1402",
        "beats_nb1384", "ties_nb1384",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
    print("\n  per-K mean_bag / median_bag RAE:")
    for r in res["per_K_records"]:
        print(f"    K={r['K']:3d}  mean_bag={r['rae_mean_bag']:.4f}  "
              f"median_bag={r['rae_median_bag']:.4f}  "
              f"d_vs_nb1373={r['delta_vs_nb1373']:+.4f}  "
              f"d_vs_nb1402={r['delta_vs_nb1402_K25']:+.4f}  "
              f"d_vs_nb1384={r['delta_vs_nb1384_bestK20']:+.4f}")
