"""nb1443 -- 75-model per-feature-bag-of-bags (BoB-of-Bags).

Build a 75-model stack of corrected OOF vectors:
    5 outer seeds {0, 1, 7, 42, 137}
      x 5 inner seeds (inner_seeds(o) = [o*1000 + s for s in {0,1,7,42,137}])
      x 3 components (nb1373 AtomPair-top30,
                      nb1352 MACCS-top20,
                      nb1364 Mordred-top30)
    = 75 per-component-per-outer-per-inner CORRECTED OOFs

Each corrected OOF is:
    pred_corr_{c, o, i}(unb) = nb1070_anchor + 5-fold-cross-fit shallow
                                 LGBM-Huber residual learner on
                                 PRUNED feature matrix (component c)
                                 with inner seed i = (outer o)*1000 + base.

Aggregations:
    stack = (75, 253) float64
    mean_oof   = row-mean(stack)
    median_oof = row-median(stack)
    pool RAE for each.

Verdict:
    BEATS_NB1422_MEDIAN  iff median_oof_rae < 0.5016 - 0.003
    BEATS_NB1422_MEAN    iff mean_oof_rae   < 0.5022 - 0.003

Outputs:
    scripts/nb1443_75_model_bag.py             (this file)
    data/processed/nb1443_summary.json
    data/processed/nb1443_mean_oof.npy         (253,) float32
    data/processed/nb1443_median_oof.npy       (253,) float32
    data/processed/nb1443_stack_75.npy         (75, 253) float32  (debug)
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

TAG = "nb1443"
ANCHOR = "nb1070"

OUTER_SEEDS = [0, 1, 7, 42, 137]
INNER_BASE_SEEDS = [0, 1, 7, 42, 137]
RESID_FOLDS = 5

# Component IDs and references.
COMPONENTS = ["nb1373", "nb1352", "nb1364"]

# nb1422 references (3-way naive 1/3 mean blend BoB).
NB1422_MEAN_REF = 0.5022
NB1422_MEDIAN_REF = 0.5016
DECISION_MARGIN = 0.003

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

TOP_K_MORDRED = 30  # rebuilt SHAP-pruned


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


def _build_pruned_feature_matrix(component: str, y_unb: np.ndarray,
                                 anchor: np.ndarray) -> tuple[np.ndarray, dict]:
    """Build the pruned (top-K + pred_chembl + mean_sim) unblind feature
    matrix for one component, matching nb1373 / nb1352 / nb1364."""
    print("\n" + "=" * 78)
    print(f"BUILD pruned feature matrix for component {component}")
    print("=" * 78)

    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    n_unb = len(y_unb)
    residual = y_unb - anchor

    # ---- ChEMBL pool + kNN-5 (shared across all 3 components) ----
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
    pred_chembl_unb = pred_chembl_pec50[unb_idx].astype(np.float32)
    mean_sim_unb = mean_sim[unb_idx].astype(np.float32)

    # ---- Component-specific base feature matrix ----
    if component == "nb1373":
        # AtomPair top-30, indices from nb1373 summary
        nb1373_path = DATA_PROCESSED / "nb1373_summary.json"
        if not nb1373_path.exists():
            raise FileNotFoundError(f"Missing {nb1373_path}")
        with open(nb1373_path) as f:
            nb1373_summary = json.load(f)
        top_bit_idx = np.array(
            nb1373_summary["top_atompair_bit_indices_ranked"], dtype=int
        )
        top_k = int(nb1373_summary["top_k_atompair"])
        if not ATOMPAIR_TE_PATH.exists():
            raise FileNotFoundError(f"Missing {ATOMPAIR_TE_PATH}")
        X_te = np.load(ATOMPAIR_TE_PATH)
        X_unb = X_te[unb_idx].astype(np.float32)
        X_unb_pruned_feat = X_unb[:, top_bit_idx]
        src = "nb1373_summary_top_atompair"
        n_feat = top_k
    elif component == "nb1352":
        # MACCS top-20, indices from nb1352 summary
        nb1352_path = DATA_PROCESSED / "nb1352_summary.json"
        if not nb1352_path.exists():
            raise FileNotFoundError(f"Missing {nb1352_path}")
        with open(nb1352_path) as f:
            nb1352_summary = json.load(f)
        top_bit_idx = np.array(
            nb1352_summary["top_maccs_bit_indices_ranked"], dtype=int
        )
        top_k = int(nb1352_summary["top_k_maccs"])
        if not MACCS_TE_PATH.exists():
            raise FileNotFoundError(f"Missing {MACCS_TE_PATH}")
        X_te = np.load(MACCS_TE_PATH)
        X_unb = X_te[unb_idx].astype(np.float32)
        X_unb_pruned_feat = X_unb[:, top_bit_idx]
        src = "nb1352_summary_top_maccs"
        n_feat = top_k
    elif component == "nb1364":
        # Mordred top-30 -- rebuild SHAP frame (seed=0) for fidelity to nb1364.
        X_mord_te = _load_mordred_test(n_test_expected=n_test)
        n_mord = int(X_mord_te.shape[1])
        X_mord_unb = X_mord_te[unb_idx].astype(np.float32)
        X_unb_full = np.concatenate(
            [
                X_mord_unb,
                pred_chembl_unb.reshape(-1, 1),
                mean_sim_unb.reshape(-1, 1),
            ],
            axis=1,
        ).astype(np.float32)
        imp_full, imp_src = _compute_shap_importance(
            X_unb_full, residual, seed=0
        )
        mord_imp = imp_full[:n_mord]
        top_k = min(TOP_K_MORDRED, n_mord)
        top_col_order = np.argsort(-mord_imp)
        top_bit_idx = top_col_order[:top_k].astype(int)
        X_unb_pruned_feat = X_mord_unb[:, top_bit_idx]
        src = f"shap_rebuilt_seed0 ({imp_src})"
        n_feat = top_k
    else:
        raise ValueError(f"Unknown component {component}")

    X_unb_pruned = np.concatenate(
        [
            X_unb_pruned_feat,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    print(f"   PRUNED feature matrix shape = {X_unb_pruned.shape}  src = {src}")

    meta = {
        "component": component,
        "top_k": int(n_feat),
        "top_idx_first10": [int(b) for b in top_bit_idx[:10].tolist()],
        "feat_src": src,
        "n_chembl_pool": int(len(pool)),
        "feat_dim_pruned": int(X_unb_pruned.shape[1]),
    }
    return X_unb_pruned, meta


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- 75-model per-feature-bag-of-bags")
    print(f"         5 outer x 5 inner x 3 components = 75 corrected OOFs")
    print(f"         outer seeds      = {OUTER_SEEDS}")
    print(f"         inner base seeds = {INNER_BASE_SEEDS}")
    print(f"         components       = {COMPONENTS}")
    print(f"         nb1422 ref       = mean {NB1422_MEAN_REF:.4f}  "
          f"median {NB1422_MEDIAN_REF:.4f}  margin {DECISION_MARGIN}")
    print("=" * 78)

    # ---- Load truth + anchor ----
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    anchor = np.load(DATA_PROCESSED / f"{ANCHOR}_pred_oof.npy").astype(np.float64)
    if anchor.shape[0] != n_unb:
        raise ValueError(f"{ANCHOR} shape mismatch: {anchor.shape} vs {n_unb}")
    residual = y_unb - anchor
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] y_unb shape = ({n_unb},)")
    print(f"[load] {ANCHOR} pooled RAE = {rae_anchor:.4f}")
    print(f"[resid] mean = {residual.mean():+.4f}  std = {residual.std():.4f}")

    # ---- Build all 75 corrected OOFs ----
    n_outer = len(OUTER_SEEDS)
    n_inner = len(INNER_BASE_SEEDS)
    n_comp = len(COMPONENTS)
    n_total = n_outer * n_inner * n_comp
    assert n_total == 75, f"expected 75, got {n_total}"

    stack = np.zeros((n_total, n_unb), dtype=np.float64)
    per_model_records = []
    per_component_meta = {}

    model_idx = 0
    for comp in COMPONENTS:
        X_pruned, meta = _build_pruned_feature_matrix(comp, y_unb, anchor)
        per_component_meta[comp] = meta

        print("\n" + "-" * 78)
        print(f"COMPONENT {comp}  --  fit 5 outer x 5 inner = 25 cross-fits")
        print("-" * 78)
        for oi, o in enumerate(OUTER_SEEDS):
            inner_seeds = [o * 1000 + s for s in INNER_BASE_SEEDS]
            print(f"   outer {o:3d}  inner seeds = {inner_seeds}")
            for ii, isd in enumerate(inner_seeds):
                resid_oof_s = _residual_cross_fit_one_seed(
                    X_pruned, residual, isd
                )
                pred_corr_s = anchor + resid_oof_s
                stack[model_idx] = pred_corr_s
                rae_m = float(rae(y_unb, pred_corr_s))
                per_model_records.append({
                    "model_idx": int(model_idx),
                    "component": comp,
                    "outer_seed": int(o),
                    "inner_seed": int(isd),
                    "rae": rae_m,
                })
                print(f"      [{model_idx:2d}] {comp}  outer={o:3d}  "
                      f"inner={isd:6d}  rae = {rae_m:.4f}")
                model_idx += 1

    assert model_idx == n_total

    per_model_rae_arr = np.array([r["rae"] for r in per_model_records])

    # ---- Row-level aggregation across all 75 vectors ----
    mean_oof = stack.mean(axis=0)
    median_oof = np.median(stack, axis=0)
    rae_mean = float(rae(y_unb, mean_oof))
    rae_median = float(rae(y_unb, median_oof))

    # ---- Per-component sub-bag for diagnostic ----
    per_comp_25_rae_mean: dict[str, float] = {}
    per_comp_25_rae_median: dict[str, float] = {}
    for ci, comp in enumerate(COMPONENTS):
        sl = stack[ci * 25:(ci + 1) * 25]
        rae_c_mean = float(rae(y_unb, sl.mean(axis=0)))
        rae_c_median = float(rae(y_unb, np.median(sl, axis=0)))
        per_comp_25_rae_mean[comp] = rae_c_mean
        per_comp_25_rae_median[comp] = rae_c_median

    print("\n" + "=" * 78)
    print("75-MODEL BAG SUMMARY")
    print("=" * 78)
    print(f"   per-model individual RAE   mean = {per_model_rae_arr.mean():.4f}")
    print(f"                              std  = {per_model_rae_arr.std():.4f}")
    print(f"                              min  = {per_model_rae_arr.min():.4f}")
    print(f"                              max  = {per_model_rae_arr.max():.4f}")
    print(f"                              p25  = "
          f"{np.percentile(per_model_rae_arr, 25):.4f}")
    print(f"                              p75  = "
          f"{np.percentile(per_model_rae_arr, 75):.4f}")
    print()
    print("   per-component 25-model sub-bag pooled RAE")
    for comp in COMPONENTS:
        print(f"      {comp}:  mean-bag = {per_comp_25_rae_mean[comp]:.4f}   "
              f"median-bag = {per_comp_25_rae_median[comp]:.4f}")
    print()
    print(f"   75-model MEAN   pooled RAE = {rae_mean:.4f}")
    print(f"   75-model MEDIAN pooled RAE = {rae_median:.4f}")

    # ---- Verdict ----
    delta_mean = rae_mean - NB1422_MEAN_REF
    delta_median = rae_median - NB1422_MEDIAN_REF
    beats_nb1422_mean = bool(rae_mean < NB1422_MEAN_REF - DECISION_MARGIN)
    beats_nb1422_median = bool(rae_median < NB1422_MEDIAN_REF - DECISION_MARGIN)
    beats_nb1422 = bool(beats_nb1422_mean or beats_nb1422_median)

    if beats_nb1422_median and beats_nb1422_mean:
        verdict = "NB1443_BEATS_NB1422_BOTH"
    elif beats_nb1422_median:
        verdict = "NB1443_BEATS_NB1422_MEDIAN_ONLY"
    elif beats_nb1422_mean:
        verdict = "NB1443_BEATS_NB1422_MEAN_ONLY"
    elif rae_median > NB1422_MEDIAN_REF + DECISION_MARGIN \
            and rae_mean > NB1422_MEAN_REF + DECISION_MARGIN:
        verdict = "NB1443_WORSE_THAN_NB1422"
    else:
        verdict = "NB1443_TIES_NB1422_WITHIN_MARGIN"

    print("\n" + "-" * 78)
    print(f"VERDICT  (margin {DECISION_MARGIN} vs nb1422 ref)")
    print(f"   nb1422 ref:   mean {NB1422_MEAN_REF:.4f}   "
          f"median {NB1422_MEDIAN_REF:.4f}")
    print(f"   nb1443:       mean {rae_mean:.4f} (d {delta_mean:+.4f})   "
          f"median {rae_median:.4f} (d {delta_median:+.4f})")
    print(f"   beats_mean   = {beats_nb1422_mean}")
    print(f"   beats_median = {beats_nb1422_median}")
    print(f"   beats_nb1422 = {beats_nb1422}")
    print(f"   verdict      = {verdict}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_mean_oof.npy",
            mean_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_median_oof.npy",
            median_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_stack_75.npy",
            stack.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_mean_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_median_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_stack_75.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "n_unb": n_unb,
        "outer_seeds": OUTER_SEEDS,
        "inner_base_seeds": INNER_BASE_SEEDS,
        "components": COMPONENTS,
        "n_models": int(n_total),
        "per_component_meta": per_component_meta,
        "per_model_records": per_model_records,
        "per_model_rae_mean": float(per_model_rae_arr.mean()),
        "per_model_rae_std": float(per_model_rae_arr.std()),
        "per_model_rae_min": float(per_model_rae_arr.min()),
        "per_model_rae_max": float(per_model_rae_arr.max()),
        "per_component_25_mean_rae": per_comp_25_rae_mean,
        "per_component_25_median_rae": per_comp_25_rae_median,
        "rae_75_mean": rae_mean,
        "rae_75_median": rae_median,
        "rae_anchor_nb1070": rae_anchor,
        "nb1422_mean_ref": NB1422_MEAN_REF,
        "nb1422_median_ref": NB1422_MEDIAN_REF,
        "decision_margin": DECISION_MARGIN,
        "delta_mean_vs_nb1422": delta_mean,
        "delta_median_vs_nb1422": delta_median,
        "beats_nb1422_mean": beats_nb1422_mean,
        "beats_nb1422_median": beats_nb1422_median,
        "beats_nb1422": beats_nb1422,
        "verdict": verdict,
        "resid_folds": RESID_FOLDS,
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
        "n_unb", "n_models",
        "per_component_25_mean_rae", "per_component_25_median_rae",
        "rae_75_mean", "rae_75_median",
        "delta_mean_vs_nb1422", "delta_median_vs_nb1422",
        "beats_nb1422_mean", "beats_nb1422_median", "beats_nb1422",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
