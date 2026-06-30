"""nb1414 -- AtomPair-25 variant (best K from nb1402 ext grid) + outer-bag.

Hypothesis:
    nb1402 found K=25 ties K=30 at RAE 0.5107 (vs nb1373 K=30 at 0.5095).
    Try outer-bag of K=25 variant -- may stabilize better than K=30 single-pass.

Protocol:
    1. Reuse top-25 AtomPair indices from nb1402 (first 25 of nb1373's
       top-30 SHAP ranking computed at seed=0 on full feature matrix).
    2. For 5 outer seeds {0, 1, 7, 42, 137}: a 5-inner-seed bag of shallow
       LGBM Huber over residual y_unb - nb1070_pred_oof on top-25 AtomPair
       + 25 ChEMBL columns (= 27 cols total).  Inner seeds = [o*1000 + s].
    3. Per-outer pooled RAE (RAE of the inner-5 mean-bag for that outer).
    4. Row-level Bag-of-Bags (BoB) MEAN and MEDIAN across the 5 outer bags
       (each outer bag is itself an inner-5 mean of cross-fit corrected
       predictions).
    5. Compare to nb1373 K=30 (0.5095) and nb1402 K=25 single-seed (0.5107).

Outputs:
    scripts/nb1414_atompair_K25_bob.py        (this file)
    data/processed/nb1414_summary.json
    data/processed/nb1414_bob_mean_oof.npy    (253,)
    data/processed/nb1414_bob_median_oof.npy  (253,)
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

TAG = "nb1414"
ANCHOR = "nb1070"

K_BEST = 25                           # locked from nb1402
OUTER_SEEDS = [0, 1, 7, 42, 137]      # 5 outer seeds
INNER_SEEDS_PER_OUTER = 5             # 5 inner seeds each
RESID_FOLDS = 5

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"   # (513, 2048) uint8
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

NB1070_REF = 0.5771
NB1373_REF = 0.5095        # K=30 mean-bag RAE
NB1402_K25_REF = 0.5107    # K=25 single 5-seed bag
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
    """Same union as nb1373 / nb1402."""
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
    """Single global LGBM (seed=0) -> SHAP TreeExplainer importance.

    This nests with nb1373/nb1402: the top-K bit ordering is deterministic
    at seed=0 and slices into top-25 identical to nb1402's K=25 row.
    """
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


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- AtomPair K={K_BEST} outer-bag-of-inner-bags; anchor={ANCHOR}")
    print(f"        outer seeds  = {OUTER_SEEDS}")
    print(f"        inner seeds  = [o*1000 + s for s in range({INNER_SEEDS_PER_OUTER})]")
    print(f"        folds        = {RESID_FOLDS}")
    print(f"        refs: nb1373 K=30 = {NB1373_REF:.4f}  "
          f"nb1402 K=25 single-bag = {NB1402_K25_REF:.4f}  "
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
    print("CHEMBL PXR POOL (local cache; same union as nb1373/nb1402)")
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

    # ---- SHAP importance frame (seed=0 -- nests w/ nb1373 / nb1402) ----
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

    # ---- Slice top-25 (the locked best K) ----
    top_k = min(K_BEST, n_ap)
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
    print(f"\n[slice] K={K_BEST}: feature matrix = {X_unb_pruned.shape}  "
          f"(dim={feat_dim_pruned})  -- top-{K_BEST} AtomPair + 2 ChEMBL cols")

    # ---- Outer-bag-of-inner-bags loop ----
    print("\n" + "-" * 78)
    print(f"OUTER-BAG: {len(OUTER_SEEDS)} outer x {INNER_SEEDS_PER_OUTER} inner "
          f"= {len(OUTER_SEEDS) * INNER_SEEDS_PER_OUTER} total cross-fit passes")
    print("-" * 78)

    n_outer = len(OUTER_SEEDS)
    # per_outer_corrected[o] = mean-of-inner cross-fit corrected pred for outer o
    per_outer_corrected = np.zeros((n_outer, n_unb), dtype=np.float64)
    per_outer_rae: list[float] = []
    per_outer_records: list[dict] = []

    # also track all 25 individual cross-fit predictions for diag
    all_individual_corrected = np.zeros(
        (n_outer * INNER_SEEDS_PER_OUTER, n_unb), dtype=np.float64
    )

    for o_i, o_seed in enumerate(OUTER_SEEDS):
        inner_seeds = [o_seed * 1000 + s for s in range(INNER_SEEDS_PER_OUTER)]
        inner_corrected = np.zeros(
            (INNER_SEEDS_PER_OUTER, n_unb), dtype=np.float64
        )
        inner_rae: list[float] = []
        inner_resid_means: list[float] = []
        inner_resid_stds: list[float] = []

        for s_i, in_seed in enumerate(inner_seeds):
            resid_oof_s = _residual_cross_fit_one_seed(
                X_unb_pruned, residual, in_seed
            )
            pred_corr_s = anchor + resid_oof_s
            inner_corrected[s_i] = pred_corr_s
            inner_rae.append(float(rae(y_unb, pred_corr_s)))
            inner_resid_means.append(float(resid_oof_s.mean()))
            inner_resid_stds.append(float(resid_oof_s.std()))
            all_individual_corrected[o_i * INNER_SEEDS_PER_OUTER + s_i] = pred_corr_s

        outer_mean_bag = inner_corrected.mean(axis=0)
        outer_rae = float(rae(y_unb, outer_mean_bag))
        per_outer_corrected[o_i] = outer_mean_bag
        per_outer_rae.append(outer_rae)
        per_outer_records.append({
            "outer_seed": int(o_seed),
            "inner_seeds": [int(x) for x in inner_seeds],
            "inner_rae": inner_rae,
            "inner_rae_mean": float(np.mean(inner_rae)),
            "inner_rae_std": float(np.std(inner_rae)),
            "outer_mean_bag_rae": outer_rae,
            "delta_vs_nb1373": outer_rae - NB1373_REF,
            "delta_vs_nb1402_k25": outer_rae - NB1402_K25_REF,
            "inner_resid_mean_avg": float(np.mean(inner_resid_means)),
            "inner_resid_std_avg": float(np.mean(inner_resid_stds)),
        })

        print(f"  outer={o_seed:>3d}  inner_seeds={inner_seeds}")
        print(f"     inner per-seed RAE = "
              f"[{', '.join(f'{r:.4f}' for r in inner_rae)}]")
        print(f"     outer mean-bag RAE = {outer_rae:.4f}  "
              f"(d_vs_nb1373 = {outer_rae - NB1373_REF:+.4f}  "
              f"d_vs_nb1402K25 = {outer_rae - NB1402_K25_REF:+.4f})")

    # ---- Bag-of-Bags (BoB) ----
    bob_mean_oof = per_outer_corrected.mean(axis=0)
    bob_median_oof = np.median(per_outer_corrected, axis=0)
    rae_bob_mean = float(rae(y_unb, bob_mean_oof))
    rae_bob_median = float(rae(y_unb, bob_median_oof))

    # Also: pool ALL 25 individual cross-fit preds row-wise (deepest pool)
    deep_mean_oof = all_individual_corrected.mean(axis=0)
    deep_median_oof = np.median(all_individual_corrected, axis=0)
    rae_deep_mean = float(rae(y_unb, deep_mean_oof))
    rae_deep_median = float(rae(y_unb, deep_median_oof))

    per_outer_mean = float(np.mean(per_outer_rae))
    per_outer_std = float(np.std(per_outer_rae))
    per_outer_min = float(np.min(per_outer_rae))
    per_outer_max = float(np.max(per_outer_rae))

    delta_bob_mean_vs_nb1373 = rae_bob_mean - NB1373_REF
    delta_bob_median_vs_nb1373 = rae_bob_median - NB1373_REF
    delta_bob_mean_vs_nb1402 = rae_bob_mean - NB1402_K25_REF
    delta_bob_median_vs_nb1402 = rae_bob_median - NB1402_K25_REF

    # Best BoB statistic
    if rae_bob_median < rae_bob_mean:
        best_bob_name = "median"
        best_bob_rae = rae_bob_median
        best_bob_oof = bob_median_oof
    else:
        best_bob_name = "mean"
        best_bob_rae = rae_bob_mean
        best_bob_oof = bob_mean_oof

    beats_nb1373 = best_bob_rae < NB1373_REF - DECISION_MARGIN
    ties_nb1373 = abs(best_bob_rae - NB1373_REF) < DECISION_MARGIN
    beats_nb1402 = best_bob_rae < NB1402_K25_REF - DECISION_MARGIN
    ties_nb1402 = abs(best_bob_rae - NB1402_K25_REF) < DECISION_MARGIN

    # ---- Report ----
    print("\n" + "=" * 78)
    print("OUTER-BAG (BoB) RESULTS")
    print("=" * 78)
    print(f"  per-outer RAE   = "
          f"[{', '.join(f'{r:.4f}' for r in per_outer_rae)}]")
    print(f"     mean={per_outer_mean:.4f}  std={per_outer_std:.4f}  "
          f"min={per_outer_min:.4f}  max={per_outer_max:.4f}")
    print(f"  BoB MEAN   RAE  = {rae_bob_mean:.4f}  "
          f"(d_vs_nb1373 = {delta_bob_mean_vs_nb1373:+.4f}  "
          f"d_vs_nb1402K25 = {delta_bob_mean_vs_nb1402:+.4f})")
    print(f"  BoB MEDIAN RAE  = {rae_bob_median:.4f}  "
          f"(d_vs_nb1373 = {delta_bob_median_vs_nb1373:+.4f}  "
          f"d_vs_nb1402K25 = {delta_bob_median_vs_nb1402:+.4f})")
    print(f"  deep MEAN (pool 25) RAE   = {rae_deep_mean:.4f}")
    print(f"  deep MEDIAN (pool 25) RAE = {rae_deep_median:.4f}")
    print(f"  best BoB stat = {best_bob_name} (RAE {best_bob_rae:.4f})")
    print(f"  nb1373 K=30  reference  = {NB1373_REF:.4f}")
    print(f"  nb1402 K=25  reference  = {NB1402_K25_REF:.4f}")
    print(f"  margin = {DECISION_MARGIN}")

    if beats_nb1373 and beats_nb1402:
        verdict = (f"BOB_K{K_BEST}_BEATS_BOTH_NB1373_AND_NB1402_"
                   f"NEW_PRIMARY_CANDIDATE_{best_bob_name.upper()}")
    elif beats_nb1402:
        verdict = (f"BOB_K{K_BEST}_BEATS_NB1402K25_FLAT_OR_WORSE_vs_NB1373_"
                   f"{best_bob_name.upper()}")
    elif beats_nb1373:
        verdict = (f"BOB_K{K_BEST}_BEATS_NB1373_NEW_PRIMARY_CANDIDATE_"
                   f"{best_bob_name.upper()}")
    elif ties_nb1373 and ties_nb1402:
        verdict = f"BOB_K{K_BEST}_FLAT_VS_BOTH_NO_GAIN"
    elif ties_nb1373:
        verdict = f"BOB_K{K_BEST}_FLAT_VS_NB1373_NO_GAIN_OVER_K30"
    elif ties_nb1402:
        verdict = f"BOB_K{K_BEST}_FLAT_VS_NB1402K25_NO_GAIN_OVER_SINGLE_BAG"
    else:
        verdict = f"BOB_K{K_BEST}_WORSE_THAN_BOTH_NB1373_AND_NB1402"
    print(f"  verdict: {verdict}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_bob_mean_oof.npy",
            bob_mean_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_bob_median_oof.npy",
            bob_median_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_bob_mean_oof.npy'}  "
          f"(shape={bob_mean_oof.shape})")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_bob_median_oof.npy'}  "
          f"(shape={bob_median_oof.shape})")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "K_best": K_BEST,
        "outer_seeds": OUTER_SEEDS,
        "inner_seeds_per_outer": INNER_SEEDS_PER_OUTER,
        "resid_folds": RESID_FOLDS,
        "n_chembl_pool": int(len(pool)),
        "n_unb": n_unb,
        "n_atompair_bits": n_ap,
        "atompair_nonzero_imp_bits": n_nonzero_imp,
        "shap_importance_source": imp_src,
        "pred_chembl_importance": float(imp_full[n_ap]),
        "sim_importance": float(imp_full[n_ap + 1]),
        "feat_dim_pruned": int(feat_dim_pruned),
        "top_bit_indices_ranked": [int(b) for b in
                                   top_bit_order[:top_k].tolist()],
        "lgbm_max_depth": 3,
        "lgbm_num_leaves": 7,
        "lgbm_n_estimators": 80,
        "lgbm_learning_rate": 0.05,
        "lgbm_min_child_samples": 20,
        "lgbm_huber_alpha": 1.0,
        "rae_anchor_nb1070": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_outer_records": per_outer_records,
        "per_outer_rae": per_outer_rae,
        "per_outer_rae_mean": per_outer_mean,
        "per_outer_rae_std": per_outer_std,
        "per_outer_rae_min": per_outer_min,
        "per_outer_rae_max": per_outer_max,
        "rae_bob_mean": rae_bob_mean,
        "rae_bob_median": rae_bob_median,
        "rae_deep_mean": rae_deep_mean,
        "rae_deep_median": rae_deep_median,
        "best_bob_stat": best_bob_name,
        "best_bob_rae": best_bob_rae,
        "delta_bob_mean_vs_nb1373": delta_bob_mean_vs_nb1373,
        "delta_bob_median_vs_nb1373": delta_bob_median_vs_nb1373,
        "delta_bob_mean_vs_nb1402_k25": delta_bob_mean_vs_nb1402,
        "delta_bob_median_vs_nb1402_k25": delta_bob_median_vs_nb1402,
        "beats_nb1373": bool(beats_nb1373),
        "ties_nb1373": bool(ties_nb1373),
        "beats_nb1402_k25": bool(beats_nb1402),
        "ties_nb1402_k25": bool(ties_nb1402),
        "verdict": verdict,
        "nb1070_ref_pooled": NB1070_REF,
        "nb1373_ref": NB1373_REF,
        "nb1402_k25_ref": NB1402_K25_REF,
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
        "K_best", "outer_seeds", "inner_seeds_per_outer",
        "n_chembl_pool", "n_atompair_bits", "atompair_nonzero_imp_bits",
        "shap_importance_source",
        "rae_anchor_nb1070",
        "per_outer_rae", "per_outer_rae_mean", "per_outer_rae_std",
        "rae_bob_mean", "rae_bob_median",
        "rae_deep_mean", "rae_deep_median",
        "best_bob_stat", "best_bob_rae",
        "delta_bob_mean_vs_nb1373", "delta_bob_median_vs_nb1373",
        "delta_bob_mean_vs_nb1402_k25", "delta_bob_median_vs_nb1402_k25",
        "beats_nb1373", "ties_nb1373",
        "beats_nb1402_k25", "ties_nb1402_k25",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
