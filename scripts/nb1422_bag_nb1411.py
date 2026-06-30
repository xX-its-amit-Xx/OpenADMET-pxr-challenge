"""nb1422 -- Outer-bag VALIDATION of nb1411 (3-way naive 1/3 mean blend).

Validates the 3-way naive-mean blend
    nb1411 = (nb1373 + nb1352 + nb1364) / 3
by repeating the underlying inner 5-seed bag of each component across five
OUTER seeds {0, 1, 7, 42, 137}, with inner seeds reparameterized as
    inner_seeds(o) = [o*1000 + s for s in {0, 1, 7, 42, 137}].

For each outer seed o:
    nb1373_o  = top-30 AtomPair SHAP-pruned + ChEMBL residual learner,
                5-seed inner bag MEAN  -> (253,)
    nb1352_o  = top-20 MACCS SHAP-pruned + ChEMBL residual learner,
                5-seed inner bag MEAN  -> (253,)
    nb1364_o  = top-30 Mordred SHAP-pruned + ChEMBL residual learner,
                5-seed inner bag MEAN  -> (253,)
    nb1411_o  = (nb1373_o + nb1352_o + nb1364_o) / 3       <- naive 1/3 mean

Aggregates:
    per_outer_rae         = rae(y_unb, nb1411_o)  for o in OUTER_SEEDS
    bob_mean_oof   = row-mean   of 5 nb1411_o vectors  -> pooled RAE
    bob_median_oof = row-median of 5 nb1411_o vectors  -> pooled RAE

Verdict NB1411_REPRODUCES iff |per_outer_mean - 0.5037| < 0.003.

IMPLEMENTATION NOTE
    Two of the three component outer-bag families are already cached on disk,
    each built with the EXACT outer/inner seed mapping required:
        nb1373 outer-bag-means -> data/processed/nb1381_per_outer_mean_oof.npy
        nb1352 outer-bag-means -> data/processed/nb1361_outer_mean_bag.npy
    The Mordred-pruned component (nb1364) has no cached outer-bag; this script
    rebuilds it from scratch using the same pipeline as nb1364 (top-30 Mordred
    SHAP-pruned + ChEMBL kNN(k=5) + 5-fold cross-fit shallow LGBM Huber on
    residual y_unb - nb1070_pred_oof) with the outer-seed inner remap.

Outputs:
    scripts/nb1422_bag_nb1411.py                          (this file)
    data/processed/nb1422_summary.json
    data/processed/nb1422_bob_mean_oof.npy                (253,) float32
    data/processed/nb1422_bob_median_oof.npy              (253,) float32
    data/processed/nb1422_per_outer_nb1411_oof.npy        (5, 253) float32
    data/processed/nb1422_per_outer_nb1364_oof.npy        (5, 253) float32
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

TAG = "nb1422"
ANCHOR = "nb1070"

OUTER_SEEDS = [0, 1, 7, 42, 137]
INNER_BASE_SEEDS = [0, 1, 7, 42, 137]
RESID_FOLDS = 5

# nb1411 reference (naive 1/3 mean blend pooled RAE at outer=0).
NB1411_REF = 0.5037
REPRODUCE_MARGIN = 0.003

# Component A (nb1373) outer-bag cache  (validated by nb1381).
NB1373_OUTER_PATH = DATA_PROCESSED / "nb1381_per_outer_mean_oof.npy"
# Component B (nb1352) outer-bag cache  (validated by nb1361).
NB1352_OUTER_PATH = DATA_PROCESSED / "nb1361_outer_mean_bag.npy"

# Component C (nb1364) -- rebuild from scratch.
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

TOP_K_MORDRED = 30


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


def _build_nb1364_outer_bag(y_unb: np.ndarray, anchor: np.ndarray,
                            outer_seeds: list[int],
                            inner_base: list[int]) -> tuple[np.ndarray, dict]:
    """Rebuild nb1364 outer-bag means (n_outer, n_unb) using inner-seed remap."""
    print("\n" + "=" * 78)
    print("REBUILD nb1364 outer-bag (Mordred top-30 + ChEMBL residual)")
    print("=" * 78)

    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    n_unb = len(y_unb)

    residual = y_unb - anchor

    # ---- Mordred-1533 (full test then unblind slice) ----
    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    n_mord = int(X_mord_te.shape[1])
    print(f"[feat] X_mord_te shape = {X_mord_te.shape}  (n_mordred={n_mord})")
    X_mord_unb = X_mord_te[unb_idx].astype(np.float32)

    # ---- ChEMBL pool + kNN feature build (513-level) ----
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

    # ---- SHAP importance (seed=0, identical to nb1364) ----
    X_unb_full = np.concatenate(
        [
            X_mord_unb,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    print(f"   full feature matrix: {X_unb_full.shape}")

    imp_full, imp_src = _compute_shap_importance(X_unb_full, residual, seed=0)
    print(f"   importance source = {imp_src}")
    mord_imp = imp_full[:n_mord]
    top_k = min(TOP_K_MORDRED, n_mord)
    top_col_order = np.argsort(-mord_imp)
    top_col_idx = top_col_order[:top_k].astype(int)
    print(f"   top-{top_k} Mordred cols (ranked first 10): "
          f"{top_col_idx[:10].tolist()}")

    X_mord_unb_pruned = X_mord_unb[:, top_col_idx]
    X_unb_pruned = np.concatenate(
        [
            X_mord_unb_pruned,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    print(f"   PRUNED feature matrix: {X_unb_pruned.shape}")

    # ---- Outer x Inner cross-fit ----
    n_outer = len(outer_seeds)
    outer_mean_bag = np.zeros((n_outer, n_unb), dtype=np.float64)
    per_outer_records = []
    for oi, o in enumerate(outer_seeds):
        inner_seeds = [o * 1000 + s for s in inner_base]
        n_inner = len(inner_seeds)
        inner_corrected = np.zeros((n_inner, n_unb), dtype=np.float64)
        inner_per_seed_rae = []
        for ii, isd in enumerate(inner_seeds):
            resid_oof_s = _residual_cross_fit_one_seed(
                X_unb_pruned, residual, isd
            )
            pred_corr_s = anchor + resid_oof_s
            inner_corrected[ii] = pred_corr_s
            r_s = float(rae(y_unb, pred_corr_s))
            inner_per_seed_rae.append(r_s)
        mean_bag_o = inner_corrected.mean(axis=0)
        outer_mean_bag[oi] = mean_bag_o
        rae_mean_o = float(rae(y_unb, mean_bag_o))
        per_outer_records.append({
            "outer_seed": int(o),
            "inner_seeds": [int(s) for s in inner_seeds],
            "inner_per_seed_rae": inner_per_seed_rae,
            "rae_mean_bag": rae_mean_o,
        })
        print(f"   outer {o:3d}  inner={inner_seeds}")
        print(f"             inner per-seed RAE = "
              f"[{', '.join(f'{r:.4f}' for r in inner_per_seed_rae)}]")
        print(f"             pooled mean_bag RAE   = {rae_mean_o:.4f}")

    meta = {
        "n_mordred_cols": n_mord,
        "top_k_mordred": int(top_k),
        "top_mordred_col_indices_ranked": [int(c) for c in top_col_idx.tolist()],
        "shap_importance_source": imp_src,
        "n_chembl_pool": int(len(pool)),
        "per_outer_records": per_outer_records,
    }
    return outer_mean_bag, meta


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- OUTER-BAG VALIDATION of nb1411 "
          f"((nb1373 + nb1352 + nb1364) / 3)")
    print(f"         outer seeds      = {OUTER_SEEDS}")
    print(f"         inner base seeds = {INNER_BASE_SEEDS}")
    print(f"         inner_seeds(o)   = [o*1000 + s for s in base]")
    print(f"         nb1411 ref pooled RAE = {NB1411_REF:.4f}  "
          f"margin = {REPRODUCE_MARGIN}")
    print("=" * 78)

    # ---- Load truth + anchor ----
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    anchor = np.load(DATA_PROCESSED / f"{ANCHOR}_pred_oof.npy").astype(np.float64)
    if anchor.shape[0] != n_unb:
        raise ValueError(f"{ANCHOR} shape mismatch: {anchor.shape} vs {n_unb}")
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] y_unb shape = ({n_unb},)")
    print(f"[load] {ANCHOR} pooled RAE = {rae_anchor:.4f}")

    # ---- Load cached nb1373 and nb1352 outer-bag matrices ----
    if not NB1373_OUTER_PATH.exists():
        raise FileNotFoundError(
            f"{NB1373_OUTER_PATH} missing -- run nb1381 first."
        )
    if not NB1352_OUTER_PATH.exists():
        raise FileNotFoundError(
            f"{NB1352_OUTER_PATH} missing -- run nb1361 first."
        )

    nb1373_outer = np.load(NB1373_OUTER_PATH).astype(np.float64)
    nb1352_outer = np.load(NB1352_OUTER_PATH).astype(np.float64)
    print(f"[load] nb1373 outer-bag means  shape = {nb1373_outer.shape}  "
          f"(from {NB1373_OUTER_PATH.name})")
    print(f"[load] nb1352 outer-bag means  shape = {nb1352_outer.shape}  "
          f"(from {NB1352_OUTER_PATH.name})")

    n_outer = len(OUTER_SEEDS)
    if nb1373_outer.shape != (n_outer, n_unb):
        raise ValueError(
            f"nb1373 outer-bag shape mismatch: {nb1373_outer.shape} != "
            f"({n_outer}, {n_unb})"
        )
    if nb1352_outer.shape != (n_outer, n_unb):
        raise ValueError(
            f"nb1352 outer-bag shape mismatch: {nb1352_outer.shape} != "
            f"({n_outer}, {n_unb})"
        )

    # ---- Rebuild nb1364 outer-bag ----
    nb1364_outer, nb1364_meta = _build_nb1364_outer_bag(
        y_unb, anchor, OUTER_SEEDS, INNER_BASE_SEEDS
    )
    nb1364_outer = nb1364_outer.astype(np.float64)
    np.save(DATA_PROCESSED / f"{TAG}_per_outer_nb1364_oof.npy",
            nb1364_outer.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_per_outer_nb1364_oof.npy'}")

    # ---- Per-outer component sanity ----
    print("\n" + "-" * 78)
    print("PER-OUTER COMPONENT RAE")
    print("-" * 78)
    per_outer_rae_nb1373: list[float] = []
    per_outer_rae_nb1352: list[float] = []
    per_outer_rae_nb1364: list[float] = []
    for oi, o in enumerate(OUTER_SEEDS):
        r1 = float(rae(y_unb, nb1373_outer[oi]))
        r2 = float(rae(y_unb, nb1352_outer[oi]))
        r3 = float(rae(y_unb, nb1364_outer[oi]))
        per_outer_rae_nb1373.append(r1)
        per_outer_rae_nb1352.append(r2)
        per_outer_rae_nb1364.append(r3)
        print(f"   outer {o:3d}:  nb1373={r1:.4f}   "
              f"nb1352={r2:.4f}   nb1364={r3:.4f}")

    # ---- Per-outer 3-way naive 1/3 mean blend ----
    print("\n" + "-" * 78)
    print("PER-OUTER nb1411 = (nb1373_o + nb1352_o + nb1364_o) / 3")
    print("-" * 78)
    per_outer_blend = (
        nb1373_outer + nb1352_outer + nb1364_outer
    ) / 3.0  # (5, 253)
    per_outer_rae_blend: list[float] = []
    per_outer_inner_seeds: list[list[int]] = []
    for oi, o in enumerate(OUTER_SEEDS):
        r = float(rae(y_unb, per_outer_blend[oi]))
        per_outer_rae_blend.append(r)
        inner_seeds = [int(o * 1000 + s) for s in INNER_BASE_SEEDS]
        per_outer_inner_seeds.append(inner_seeds)
        print(f"   outer {o:3d}:  nb1411_o RAE = {r:.4f}   "
              f"inner seeds = {inner_seeds}")

    per_outer_arr = np.array(per_outer_rae_blend)
    per_outer_mean = float(per_outer_arr.mean())
    per_outer_std = float(per_outer_arr.std())
    per_outer_min = float(per_outer_arr.min())
    per_outer_max = float(per_outer_arr.max())
    per_outer_median = float(np.median(per_outer_arr))

    # ---- BoB row-level aggregations across 5 nb1411_o vectors ----
    bob_mean_oof = per_outer_blend.mean(axis=0)
    bob_median_oof = np.median(per_outer_blend, axis=0)
    rae_bob_mean = float(rae(y_unb, bob_mean_oof))
    rae_bob_median = float(rae(y_unb, bob_median_oof))

    print("\n" + "=" * 78)
    print("OUTER-BAG SUMMARY")
    print("=" * 78)
    print(f"   per-outer nb1411 RAE list = "
          f"[{', '.join(f'{r:.4f}' for r in per_outer_rae_blend)}]")
    print(f"   per-outer mean   = {per_outer_mean:.4f}")
    print(f"   per-outer std    = {per_outer_std:.4f}")
    print(f"   per-outer min    = {per_outer_min:.4f}")
    print(f"   per-outer max    = {per_outer_max:.4f}")
    print(f"   per-outer median = {per_outer_median:.4f}")
    print(f"   BoB MEAN   pooled RAE = {rae_bob_mean:.4f}")
    print(f"   BoB MEDIAN pooled RAE = {rae_bob_median:.4f}")

    # ---- Verdict ----
    delta_per_outer = per_outer_mean - NB1411_REF
    reproduces = abs(delta_per_outer) < REPRODUCE_MARGIN
    if reproduces:
        verdict = "NB1411_REPRODUCES"
    elif per_outer_mean < NB1411_REF - REPRODUCE_MARGIN:
        verdict = "NB1411_OPTIMISTIC_OUTER_BAG_BETTER"
    else:
        verdict = "NB1411_LUCKY_SEED_OUTER_BAG_WORSE"

    delta_outer0 = per_outer_rae_blend[0] - NB1411_REF
    outer0_reproduces = abs(delta_outer0) < REPRODUCE_MARGIN

    print("\n" + "-" * 78)
    print(f"VERDICT (per-outer-mean within {REPRODUCE_MARGIN} of nb1411 ref "
          f"{NB1411_REF:.4f}):")
    print(f"   per-outer-mean = {per_outer_mean:.4f}   "
          f"(d vs ref = {delta_per_outer:+.4f})")
    print(f"   outer=0 blend  = {per_outer_rae_blend[0]:.4f}   "
          f"(d vs ref = {delta_outer0:+.4f})   "
          f"outer0_reproduces = {outer0_reproduces}")
    print(f"   verdict = {verdict}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_bob_mean_oof.npy",
            bob_mean_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_bob_median_oof.npy",
            bob_median_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_per_outer_nb1411_oof.npy",
            per_outer_blend.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_bob_mean_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_bob_median_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_per_outer_nb1411_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "n_unb": n_unb,
        "outer_seeds": OUTER_SEEDS,
        "inner_base_seeds": INNER_BASE_SEEDS,
        "per_outer_inner_seeds": per_outer_inner_seeds,
        "resid_folds": RESID_FOLDS,
        "blend_form": "(nb1373 + nb1352 + nb1364) / 3",
        "nb1373_outer_source": str(NB1373_OUTER_PATH.name),
        "nb1352_outer_source": str(NB1352_OUTER_PATH.name),
        "nb1364_outer_source": "rebuilt_in_nb1422",
        "nb1364_rebuild_meta": nb1364_meta,
        "rae_anchor_nb1070": rae_anchor,
        "per_outer_rae_nb1373": per_outer_rae_nb1373,
        "per_outer_rae_nb1352": per_outer_rae_nb1352,
        "per_outer_rae_nb1364": per_outer_rae_nb1364,
        "per_outer_rae_nb1411": per_outer_rae_blend,
        "per_outer_mean": per_outer_mean,
        "per_outer_std": per_outer_std,
        "per_outer_min": per_outer_min,
        "per_outer_max": per_outer_max,
        "per_outer_median": per_outer_median,
        "rae_bob_mean": rae_bob_mean,
        "rae_bob_median": rae_bob_median,
        "nb1411_ref": NB1411_REF,
        "reproduce_margin": REPRODUCE_MARGIN,
        "delta_per_outer_mean_vs_nb1411": delta_per_outer,
        "delta_outer0_vs_nb1411": delta_outer0,
        "outer0_reproduces": bool(outer0_reproduces),
        "reproduces": bool(reproduces),
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
        "n_unb", "outer_seeds",
        "blend_form",
        "per_outer_rae_nb1373",
        "per_outer_rae_nb1352",
        "per_outer_rae_nb1364",
        "per_outer_rae_nb1411",
        "per_outer_mean", "per_outer_std",
        "per_outer_min", "per_outer_max", "per_outer_median",
        "rae_bob_mean", "rae_bob_median",
        "delta_per_outer_mean_vs_nb1411",
        "delta_outer0_vs_nb1411",
        "outer0_reproduces", "reproduces",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
