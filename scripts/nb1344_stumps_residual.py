"""nb1344 -- LGBM depth-1 stumps residual on MACCS + ChEMBL features.

Hypothesis:
    nb1242 (depth=3 shallow LGBM) reaches 0.5431 mean-bag pooled RAE on the
    same 169-feature substrate (MACCS-167 + pred_chembl_pec50 + sim).  At
    n=253 the residual learner has a tight overfitting margin -- each
    additional bit of model capacity directly trades into variance.  Depth-1
    stumps (each tree is a single split, num_leaves=2) is the MAXIMUM
    regularization a tree learner offers without going linear.  We compensate
    the shallow per-tree capacity by raising n_estimators to 200 and dropping
    lr to 0.03 so cumulative model capacity ~ matches nb1242 while keeping
    each tree's variance contribution tiny.  If the depth=3->depth=1 swap
    EXCEEDS the 0.003 decision margin, the implication is that nb1242 was
    leaving a small amount of generalization on the table to noise.

Pipeline:
    1. Anchor nb1070, residual y_unb - nb1070_pred_oof.
    2. Features: MACCS-167 + pred_chembl_pec50 + sim_chembl = 169 cols.
       Identical 169-d substrate as nb1242 (replays the SAME ChEMBL kNN
       computation rather than touching nb1242 caches, to keep this script
       self-contained).
    3. 5-seed bag LGBM Huber with max_depth=1, num_leaves=2, n_estimators=200,
       learning_rate=0.03, min_child_samples=20, huber alpha=1.0.
    4. 5-fold cross-fit per seed, mean-bag RAE pooled.
    5. Verdict at 0.003 margin vs nb1242 (0.5431) -- depth=3 same substrate.

Outputs:
    scripts/nb1344_stumps_residual.py            (this file)
    data/processed/nb1344_summary.json
    data/processed/nb1344_mean_bag_oof.npy       (253,) float32
    data/processed/nb1344_per_seed_corrected_oof.npy (5, 253) float32
    data/processed/nb1344_median_bag_oof.npy     (253,) float32
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

TAG = "nb1344"
ANCHOR = "nb1070"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

NB1070_REF = 0.5771
NB1242_REF = 0.5431      # depth=3 LGBM on identical substrate
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
    """Union three local ChEMBL PXR caches -> (smi, pec50) frame.
    Identical loader to nb1242 -- self-contained copy so this script
    doesn't depend on nb1242 outputs."""
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


def _knn_predict(top_idx, top_sim, pool_labels, fallback):
    w = np.clip(top_sim, 0.0, 1.0)
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
    """Depth-1 stumps -- MAXIMUM regularization in the tree-learner family.
    Each tree is a single split (num_leaves=2).  More trees (200) and lower
    lr (0.03) compensate the shallow per-tree capacity so cumulative model
    capacity ~ matches nb1242 (200 * 0.03 ~ 80 * 0.05 within boosting-scale)
    while per-tree variance contribution is tiny."""
    return dict(
        objective="huber",
        alpha=1.0,
        learning_rate=0.03,
        n_estimators=200,
        max_depth=1,
        num_leaves=2,
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


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- depth-1 stump residual on MACCS+ChEMBL (169-d)")
    print(f"          seeds = {RESID_SEEDS}  folds = {RESID_FOLDS}")
    print(f"          stumps: depth=1, num_leaves=2, n_est=200, lr=0.03")
    print("=" * 78)

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

    # ---- ChEMBL pool (identical to nb1242) ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL (local cache; identical loader to nb1242)")
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
    print(f"   pool: {n_before} -> {n_after}  (dropped {n_before - n_after} "
          f"test-overlapping cpds)")

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
        std_test_smiles.append("" if m is None else Chem.MolToSmiles(m))
    fp_test = morgan_fp_batch(std_test_smiles)
    print(f"   test FP: {fp_test.shape}")

    print("\n" + "-" * 78)
    print(f"TANIMOTO kNN (k={KNN_K})")
    print("-" * 78)
    top_idx, top_sim = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx, top_sim, pool_labels, fallback=pool_median
    )
    top1_sim = top_sim[:, 0]
    print(f"   pred_chembl_pec50  mean={pred_chembl_pec50.mean():.3f}  "
          f"std={pred_chembl_pec50.std():.3f}")
    print(f"   top1 sim   p50={np.percentile(top1_sim, 50):.3f}  "
          f"p90={np.percentile(top1_sim, 90):.3f}")
    n_zero_neighbor = int((top1_sim < SIM_FLOOR).sum())

    # ---- Build 169-d residual feature matrix on 253 ----
    X_maccs_te = np.load(MACCS_TE_PATH)
    if X_maccs_te.shape[0] != n_test:
        raise ValueError(f"MACCS cache shape mismatch: {X_maccs_te.shape}")
    X_maccs_unb = X_maccs_te[unb_idx].astype(np.float32)

    pred_chembl_unb = pred_chembl_pec50[unb_idx].astype(np.float32)
    mean_sim_unb = mean_sim[unb_idx].astype(np.float32)
    X_unb = np.concatenate(
        [
            X_maccs_unb,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim = X_unb.shape[1]
    print(f"   residual feature matrix: {X_unb.shape}")

    # ---- Per-seed residual cross-fit ----
    print("\n" + "-" * 78)
    print(f"PER-SEED CROSS-FIT (depth=1 stumps, n_est=200, lr=0.03)")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []
    for i, s in enumerate(RESID_SEEDS):
        resid_oof_s = _residual_cross_fit_one_seed(X_unb, residual, s)
        pred_corr_s = anchor + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        delta_s = rae_s - rae_anchor
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_nb1070": delta_s,
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
        })
        print(f"   seed {s:3d}:  rae_corr = {rae_s:.4f}  "
              f"(d_vs_nb1070 = {delta_s:+.4f})  "
              f"|resid_oof|.std = {resid_oof_s.std():.3f}")

    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))

    per_seed_rae_arr = np.array(per_seed_rae)
    rae_per_seed_mean = float(per_seed_rae_arr.mean())
    rae_per_seed_std = float(per_seed_rae_arr.std())

    # ---- Pearson vs nb1242 ----
    nb1242_path = DATA_PROCESSED / "nb1242_mean_bag_oof.npy"
    pearson_vs_nb1242 = None
    if nb1242_path.exists():
        nb1242_oof = np.load(nb1242_path).astype(np.float64)
        if nb1242_oof.shape == mean_bag_oof.shape:
            x = mean_bag_oof - mean_bag_oof.mean()
            y = nb1242_oof - nb1242_oof.mean()
            denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
            pearson_vs_nb1242 = float((x * y).sum() / denom) if denom > 0 else None

    print("\n" + "-" * 78)
    print("BAG AGGREGATIONS")
    print("-" * 78)
    print(f"   per-seed RAE list      = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   per-seed mean          = {rae_per_seed_mean:.4f}")
    print(f"   per-seed std           = {rae_per_seed_std:.4f}")
    print(f"   pooled RAE(mean_bag)   = {rae_mean_bag:.4f}  "
          f"(d_vs_nb1070 = {rae_mean_bag - rae_anchor:+.4f})")
    print(f"   pooled RAE(median_bag) = {rae_median_bag:.4f}")
    print(f"   nb1242 ref (depth=3)   = {NB1242_REF:.4f}  "
          f"(delta = {rae_mean_bag - NB1242_REF:+.4f})")
    print(f"   pearson vs nb1242      = {pearson_vs_nb1242}")

    beats_nb1070 = rae_mean_bag < rae_anchor - DECISION_MARGIN
    beats_nb1242 = rae_mean_bag < NB1242_REF - DECISION_MARGIN

    if beats_nb1242:
        verdict = "STUMPS_BEAT_DEPTH3_NEW_PRIMARY_CANDIDATE"
    elif abs(rae_mean_bag - NB1242_REF) < DECISION_MARGIN:
        verdict = "STUMPS_TIE_DEPTH3_NO_NEW_SIGNAL"
    elif beats_nb1070:
        verdict = "STUMPS_HELP_NB1070_BUT_WORSE_THAN_DEPTH3"
    else:
        verdict = "STUMPS_HURT_NB1070"
    print(f"   verdict                = {verdict}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_per_seed_corrected_oof.npy",
            per_seed_corrected.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy",
            mean_bag_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_median_bag_oof.npy",
            median_bag_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_mean_bag_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "data_source": "local_chembl_caches_union (same as nb1242)",
        "n_chembl_pool": int(len(pool)),
        "test_inchikeys_in_pool_dropped": int(n_before - n_after),
        "pool_pec50_mean": float(pool_labels.mean()),
        "pool_pec50_std": float(pool_labels.std()),
        "pool_pec50_median": pool_median,
        "knn_k": KNN_K,
        "n_zero_neighbor_rows": n_zero_neighbor,
        "top1_sim_p50": float(np.percentile(top1_sim, 50)),
        "top1_sim_p90": float(np.percentile(top1_sim, 90)),
        "n_unb": n_unb,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "feature_dim": feat_dim,
        "lgbm_max_depth": 1,
        "lgbm_num_leaves": 2,
        "lgbm_n_estimators": 200,
        "lgbm_learning_rate": 0.03,
        "lgbm_min_child_samples": 20,
        "lgbm_huber_alpha": 1.0,
        "rae_anchor_nb1070": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_seed_rae": per_seed_rae,
        "per_seed_records": per_seed_records,
        "rae_per_seed_mean": rae_per_seed_mean,
        "rae_per_seed_std": rae_per_seed_std,
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_nb1070": rae_mean_bag - rae_anchor,
        "delta_mean_bag_vs_nb1242": rae_mean_bag - NB1242_REF,
        "pearson_vs_nb1242": pearson_vs_nb1242,
        "beats_nb1070": bool(beats_nb1070),
        "beats_nb1242": bool(beats_nb1242),
        "verdict": verdict,
        "nb1070_ref_pooled": NB1070_REF,
        "nb1242_ref": NB1242_REF,
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
        "n_chembl_pool", "rae_anchor_nb1070", "per_seed_rae",
        "rae_per_seed_mean", "rae_per_seed_std",
        "rae_mean_bag", "rae_median_bag",
        "delta_mean_bag_vs_nb1070", "delta_mean_bag_vs_nb1242",
        "pearson_vs_nb1242",
        "beats_nb1070", "beats_nb1242", "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
