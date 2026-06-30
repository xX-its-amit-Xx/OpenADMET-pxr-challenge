"""nb1473 -- Outer-bag VALIDATION of nb1462 (SHAP-pruned chemprop embed top-30
+ ChEMBL residual, shallow LGBM Huber).

Same 32-col feature recipe as nb1462 (top-30 chemprop embed dims + pred_chembl
+ mean_sim) but repeats the inner 5-seed shallow LGBM Huber bag across five
OUTER seeds {0, 1, 7, 42, 137} with inner seeds reparameterized as
    inner_seeds(o) = [o*1000 + s for s in {0, 1, 7, 42, 137}].

For each outer seed o:
    nb1462_o = inner-mean bag of 5 LGBM(huber, depth=3, num_leaves=7,
               n_est=80, lr=0.05, alpha=1.0, min_child_samples=20,
               random_state=isd) residual learners, each evaluated 5-fold
               cross-fit -> (253,)
Aggregates:
    per_outer_rae   = rae(y_unb, nb1462_o) for o in OUTER_SEEDS
    bob_mean_oof    = row-mean   of 5 nb1462_o vectors -> pooled RAE
    bob_median_oof  = row-median of 5 nb1462_o vectors -> pooled RAE

Verdict NB1462_REPRODUCES iff |per_outer_mean - 0.5264| < 0.003
(nb1462 ref = inner-5-seed mean of per-seed RAEs at outer=0).

Outputs:
    scripts/nb1473_bag_nb1462.py                  (this file)
    data/processed/nb1473_summary.json
    data/processed/nb1473_bob_mean_oof.npy        (253,) float32
    data/processed/nb1473_bob_median_oof.npy      (253,) float32
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

TAG = "nb1473"
ANCHOR = "nb1070"

OUTER_SEEDS = [0, 1, 7, 42, 137]
INNER_BASE_SEEDS = [0, 1, 7, 42, 137]
RESID_FOLDS = 5

# nb1462 reference (inner-5-seed pooled per-seed mean RAE at outer=0).
NB1462_REF = 0.5264
NB1462_MEAN_BAG_REF = 0.5144
NB1462_MEDIAN_BAG_REF = 0.5155
REPRODUCE_MARGIN = 0.003

NB1462_SUMMARY = DATA_PROCESSED / "nb1462_summary.json"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

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


def _build_32col_unblind_features(n_unb: int):
    """Build the 32-col chemprop-embed-top-30 + ChEMBL feature matrix on 253."""
    print("=" * 78)
    print("BUILD 32-col features (top-30 chemprop embed + pred_chembl + sim)")
    print("=" * 78)

    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")

    # ---- Reuse top-30 chemprop embed indices from nb1462 ----
    if not NB1462_SUMMARY.exists():
        raise FileNotFoundError(
            f"missing {NB1462_SUMMARY} -- run nb1462 first"
        )
    with open(NB1462_SUMMARY) as f:
        sum_1462 = json.load(f)
    top_embed_dim_idx = np.array(
        sum_1462["top_embed_dim_indices_ranked"], dtype=int
    )
    n_top_embed = int(len(top_embed_dim_idx))
    print(f"[reuse] top-{n_top_embed} chemprop embed dims (from nb1462)")
    print(f"        ranked indices = {top_embed_dim_idx.tolist()}")

    # ---- chemprop embedding (unblind slice -> top-30 dims) ----
    if not CHEMPROP_EMBED_TE_PATH.exists():
        raise FileNotFoundError(
            f"chemprop embed test cache missing: {CHEMPROP_EMBED_TE_PATH}"
        )
    X_embed_te = np.load(CHEMPROP_EMBED_TE_PATH)
    if X_embed_te.shape[0] != n_test:
        raise ValueError(f"chemprop embed cache shape mismatch: "
                         f"{X_embed_te.shape}")
    n_embed_dims = int(X_embed_te.shape[1])
    print(f"[feat] chemprop embed cache shape = {X_embed_te.shape}")
    X_embed_unb = X_embed_te[unb_idx].astype(np.float32)
    X_embed_unb_top = X_embed_unb[:, top_embed_dim_idx].astype(np.float32)
    print(f"[feat] X_embed_unb_top shape = {X_embed_unb_top.shape}")

    # ---- ChEMBL pool ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL (same union as nb1462)")
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
    pred_chembl_unb = pred_chembl_pec50[unb_idx].astype(np.float32)
    mean_sim_unb = mean_sim[unb_idx].astype(np.float32)

    # ---- Build 32-col feature matrix on 253 ----
    X_unb = np.concatenate(
        [
            X_embed_unb_top,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim = X_unb.shape[1]
    expected_dim = n_top_embed + 2
    if feat_dim != expected_dim:
        raise ValueError(f"feat_dim {feat_dim} != expected {expected_dim}")
    print(f"\n   PRUNED feature matrix: {X_unb.shape}  "
          f"(top-{n_top_embed} chemprop embed + pred_chembl + sim)")

    meta = {
        "n_top_embed": n_top_embed,
        "n_embed_dims_total": n_embed_dims,
        "feat_dim": int(feat_dim),
        "n_chembl_pool": int(len(pool)),
        "feat_breakdown": {
            "chemprop_embed_top": n_top_embed,
            "pred_chembl_pec50": 1,
            "mean_sim": 1,
            "total": int(feat_dim),
        },
        "top_embed_dim_indices_ranked": [int(b) for b in top_embed_dim_idx.tolist()],
    }
    return X_unb, meta


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- OUTER-BAG VALIDATION of nb1462 "
          f"(LGBM Huber on 32-col chemprop-embed-top-30 + ChEMBL features)")
    print(f"         outer seeds      = {OUTER_SEEDS}")
    print(f"         inner base seeds = {INNER_BASE_SEEDS}")
    print(f"         inner_seeds(o)   = [o*1000 + s for s in base]")
    print(f"         nb1462 per-seed mean ref = {NB1462_REF:.4f}  "
          f"margin = {REPRODUCE_MARGIN}")
    print(f"         nb1462 mean_bag ref     = {NB1462_MEAN_BAG_REF:.4f}")
    print(f"         nb1462 median_bag ref   = {NB1462_MEDIAN_BAG_REF:.4f}")
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
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Build 32-col features once ----
    X_unb, feat_meta = _build_32col_unblind_features(n_unb)
    feat_dim = int(X_unb.shape[1])

    # ---- Outer x Inner cross-fit ----
    print("\n" + "=" * 78)
    print(f"OUTER x INNER LGBM HUBER RESIDUAL CROSS-FIT (dim={feat_dim})")
    print("=" * 78)
    n_outer = len(OUTER_SEEDS)
    n_inner = len(INNER_BASE_SEEDS)
    outer_mean_bag = np.zeros((n_outer, n_unb), dtype=np.float64)
    per_outer_records = []
    per_outer_inner_seeds: list[list[int]] = []

    for oi, o in enumerate(OUTER_SEEDS):
        inner_seeds = [int(o * 1000 + s) for s in INNER_BASE_SEEDS]
        per_outer_inner_seeds.append(inner_seeds)
        inner_corrected = np.zeros((n_inner, n_unb), dtype=np.float64)
        inner_per_seed_rae: list[float] = []
        t_outer = time.time()
        for ii, isd in enumerate(inner_seeds):
            ts = time.time()
            resid_oof_s = _residual_cross_fit_one_seed(X_unb, residual, isd)
            pred_corr_s = anchor + resid_oof_s
            inner_corrected[ii] = pred_corr_s
            r_s = float(rae(y_unb, pred_corr_s))
            inner_per_seed_rae.append(r_s)
            print(f"   outer {o:3d}  inner seed {isd:6d}:  "
                  f"rae_corr = {r_s:.4f}  wall = {time.time() - ts:.1f}s")
        mean_bag_o = inner_corrected.mean(axis=0)
        outer_mean_bag[oi] = mean_bag_o
        rae_mean_o = float(rae(y_unb, mean_bag_o))
        per_outer_records.append({
            "outer_seed": int(o),
            "inner_seeds": inner_seeds,
            "inner_per_seed_rae": inner_per_seed_rae,
            "rae_mean_bag": rae_mean_o,
            "wall_sec": round(time.time() - t_outer, 2),
        })
        print(f"   outer {o:3d}  pooled mean_bag RAE = {rae_mean_o:.4f}  "
              f"(outer wall = {time.time() - t_outer:.1f}s)")

    per_outer_rae_blend: list[float] = [
        rec["rae_mean_bag"] for rec in per_outer_records
    ]
    per_outer_arr = np.array(per_outer_rae_blend)
    per_outer_mean = float(per_outer_arr.mean())
    per_outer_std = float(per_outer_arr.std())
    per_outer_min = float(per_outer_arr.min())
    per_outer_max = float(per_outer_arr.max())
    per_outer_median = float(np.median(per_outer_arr))

    # ---- BoB row-level aggregations across 5 nb1462_o vectors ----
    bob_mean_oof = outer_mean_bag.mean(axis=0)
    bob_median_oof = np.median(outer_mean_bag, axis=0)
    rae_bob_mean = float(rae(y_unb, bob_mean_oof))
    rae_bob_median = float(rae(y_unb, bob_median_oof))

    print("\n" + "=" * 78)
    print("OUTER-BAG SUMMARY")
    print("=" * 78)
    print(f"   per-outer nb1462 RAE list = "
          f"[{', '.join(f'{r:.4f}' for r in per_outer_rae_blend)}]")
    print(f"   per-outer mean   = {per_outer_mean:.4f}")
    print(f"   per-outer std    = {per_outer_std:.4f}")
    print(f"   per-outer min    = {per_outer_min:.4f}")
    print(f"   per-outer max    = {per_outer_max:.4f}")
    print(f"   per-outer median = {per_outer_median:.4f}")
    print(f"   BoB MEAN   pooled RAE = {rae_bob_mean:.4f}  "
          f"(d vs nb1462_mean_bag = {rae_bob_mean - NB1462_MEAN_BAG_REF:+.4f})")
    print(f"   BoB MEDIAN pooled RAE = {rae_bob_median:.4f}  "
          f"(d vs nb1462_median_bag = {rae_bob_median - NB1462_MEDIAN_BAG_REF:+.4f})")

    # ---- Pearson vs nb1462 mean_bag (sanity) ----
    def _pearson_vs(path: Path, vec: np.ndarray):
        if not path.exists():
            return None
        oof = np.load(path).astype(np.float64)
        if oof.shape[0] != n_unb:
            return None
        return float(np.corrcoef(vec, oof)[0, 1])

    pearson_bobmean_vs_nb1462_mean = _pearson_vs(
        DATA_PROCESSED / "nb1462_mean_bag_oof.npy", bob_mean_oof
    )
    pearson_bobmedian_vs_nb1462_mean = _pearson_vs(
        DATA_PROCESSED / "nb1462_mean_bag_oof.npy", bob_median_oof
    )

    if pearson_bobmean_vs_nb1462_mean is not None:
        print(f"   Pearson(bob_mean,   nb1462_mean_bag) = "
              f"{pearson_bobmean_vs_nb1462_mean:.4f}")
    if pearson_bobmedian_vs_nb1462_mean is not None:
        print(f"   Pearson(bob_median, nb1462_mean_bag) = "
              f"{pearson_bobmedian_vs_nb1462_mean:.4f}")

    # ---- Verdict ----
    delta_per_outer = per_outer_mean - NB1462_REF
    reproduces = abs(delta_per_outer) < REPRODUCE_MARGIN
    if reproduces:
        verdict = "NB1462_REPRODUCES"
    elif per_outer_mean < NB1462_REF - REPRODUCE_MARGIN:
        verdict = "NB1462_OPTIMISTIC_OUTER_BAG_BETTER"
    else:
        verdict = "NB1462_LUCKY_SEED_OUTER_BAG_WORSE"

    delta_outer0 = per_outer_rae_blend[0] - NB1462_REF
    outer0_reproduces = abs(delta_outer0) < REPRODUCE_MARGIN

    print("\n" + "-" * 78)
    print(f"VERDICT (per-outer-mean within {REPRODUCE_MARGIN} of nb1462 ref "
          f"{NB1462_REF:.4f}):")
    print(f"   per-outer-mean = {per_outer_mean:.4f}   "
          f"(d vs ref = {delta_per_outer:+.4f})")
    print(f"   outer=0 bag    = {per_outer_rae_blend[0]:.4f}   "
          f"(d vs ref = {delta_outer0:+.4f})   "
          f"outer0_reproduces = {outer0_reproduces}")
    print(f"   verdict = {verdict}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_bob_mean_oof.npy",
            bob_mean_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_bob_median_oof.npy",
            bob_median_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_bob_mean_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_bob_median_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "data_source": "chemprop_embed_cache_nb1312 + local_chembl_caches_union",
        "model_family": "LightGBM",
        "lgbm_objective": "huber",
        "lgbm_huber_alpha": 1.0,
        "lgbm_max_depth": 3,
        "lgbm_num_leaves": 7,
        "lgbm_n_estimators": 80,
        "lgbm_learning_rate": 0.05,
        "lgbm_min_child_samples": 20,
        "n_unb": n_unb,
        "outer_seeds": OUTER_SEEDS,
        "inner_base_seeds": INNER_BASE_SEEDS,
        "per_outer_inner_seeds": per_outer_inner_seeds,
        "resid_folds": RESID_FOLDS,
        "feat_dim": feat_dim,
        **{f"feat_meta_{k}": v for k, v in feat_meta.items()},
        "rae_anchor_nb1070": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_outer_records": per_outer_records,
        "per_outer_rae_nb1462": per_outer_rae_blend,
        "per_outer_mean": per_outer_mean,
        "per_outer_std": per_outer_std,
        "per_outer_min": per_outer_min,
        "per_outer_max": per_outer_max,
        "per_outer_median": per_outer_median,
        "rae_bob_mean": rae_bob_mean,
        "rae_bob_median": rae_bob_median,
        "nb1462_ref": NB1462_REF,
        "nb1462_mean_bag_ref": NB1462_MEAN_BAG_REF,
        "nb1462_median_bag_ref": NB1462_MEDIAN_BAG_REF,
        "reproduce_margin": REPRODUCE_MARGIN,
        "delta_per_outer_mean_vs_nb1462": delta_per_outer,
        "delta_outer0_vs_nb1462": delta_outer0,
        "delta_bob_mean_vs_nb1462_mean_bag":
            rae_bob_mean - NB1462_MEAN_BAG_REF,
        "delta_bob_median_vs_nb1462_median_bag":
            rae_bob_median - NB1462_MEDIAN_BAG_REF,
        "outer0_reproduces": bool(outer0_reproduces),
        "reproduces": bool(reproduces),
        "pearson_bobmean_vs_nb1462_mean_bag": pearson_bobmean_vs_nb1462_mean,
        "pearson_bobmedian_vs_nb1462_mean_bag":
            pearson_bobmedian_vs_nb1462_mean,
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
        "n_unb", "outer_seeds", "feat_dim",
        "rae_anchor_nb1070",
        "per_outer_rae_nb1462",
        "per_outer_mean", "per_outer_std",
        "per_outer_min", "per_outer_max", "per_outer_median",
        "rae_bob_mean", "rae_bob_median",
        "delta_per_outer_mean_vs_nb1462",
        "delta_outer0_vs_nb1462",
        "delta_bob_mean_vs_nb1462_mean_bag",
        "delta_bob_median_vs_nb1462_median_bag",
        "outer0_reproduces", "reproduces",
        "pearson_bobmean_vs_nb1462_mean_bag",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
