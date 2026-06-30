"""nb1381 -- Outer-bag VALIDATION of nb1373 (SHAP-pruned AtomPair top-30 + ChEMBL).

Hypothesis:
    nb1373 reported mean-bag pooled RAE 0.5095 from a single 5-seed inner bag
    (seeds = [0, 1, 7, 42, 137]).  We want to verify that the 0.5095 number is
    not a single-seed-choice artifact by repeating the inner 5-seed bag across
    five OUTER seeds {0, 1, 7, 42, 137}, with inner seeds reparameterized as
        inner_seeds(o) = [o*1000 + s for s in {0, 1, 7, 42, 137}].
    For outer=0 this becomes [0, 1, 7, 42, 137] -> must reproduce nb1373's
    mean-bag pooled RAE within numerical noise.

Protocol:
    1.  Anchor = nb1070_pred_oof on 253 unblind rows.  residual = y_unb - anchor.
    2.  Build ChEMBL PXR pool + kNN-5 Tanimoto on cached Morgan-2048 (same union
        as nb1373/nb1352/nb1242).
    3.  Reuse nb1373's SHAP-selected top-30 AtomPair bit indices (read from
        data/processed/nb1373_summary.json) -> 32-col PRUNED feature matrix.
    4.  For each outer seed o in [0, 1, 7, 42, 137]:
            inner_seeds = [o*1000 + s for s in [0, 1, 7, 42, 137]]
            for each inner seed i: 5-fold cross-fit shallow LGBM Huber on
                X_pruned -> resid_oof; pred_corr = anchor + resid_oof.
            inner_bag_mean[o]   = mean of 5 inner pred_corr vectors
            inner_bag_median[o] = median of 5 inner pred_corr vectors
            per_outer_pooled_rae[o] = rae(y_unb, inner_bag_mean[o])
    5.  BoB MEAN  = mean of 5 outer-mean vectors  (row-level avg of bags)
        BoB MED  = median of 5 outer-mean vectors
        Also report BoB on outer-median bags as a sanity tile.
    6.  Verdict NB1373_REPRODUCES iff |per_outer_rae[0] - 0.5095| < 0.003.

Outputs:
    scripts/nb1381_bag_nb1373.py                          (this file)
    data/processed/nb1381_summary.json
    data/processed/nb1381_bob_mean_oof.npy                (253,) float32
    data/processed/nb1381_bob_median_oof.npy              (253,) float32
    data/processed/nb1381_per_outer_mean_oof.npy          (5, 253) float32
    data/processed/nb1381_per_outer_median_oof.npy        (5, 253) float32
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

TAG = "nb1381"
ANCHOR = "nb1070"
NB1373_TAG = "nb1373"

RESID_FOLDS = 5
INNER_BASE_SEEDS = [0, 1, 7, 42, 137]
OUTER_SEEDS = [0, 1, 7, 42, 137]

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"   # (513, 2048) uint8
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

NB1373_REF_MEAN_BAG = 0.5095     # mean-bag pooled RAE reported in nb1373
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
    """Same union as nb1373 / nb1352 / nb1242."""
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


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- OUTER-BAG VALIDATION of nb1373 (SHAP-pruned AtomPair top-30)")
    print(f"          outer seeds = {OUTER_SEEDS}")
    print(f"          inner base seeds = {INNER_BASE_SEEDS}")
    print(f"          inner_seeds(o) = [o*1000 + s for s in base]")
    print(f"          5-fold cross-fit per inner; nb1373 mean-bag ref = "
          f"{NB1373_REF_MEAN_BAG:.4f}; margin = {DECISION_MARGIN}")
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
    print(f"[load] {ANCHOR} pooled RAE = {rae_anchor:.4f}")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Reuse nb1373 SHAP-selected top-30 AtomPair bit indices ----
    nb1373_summary_path = DATA_PROCESSED / f"{NB1373_TAG}_summary.json"
    if not nb1373_summary_path.exists():
        raise FileNotFoundError(f"Missing {nb1373_summary_path}; run nb1373 first.")
    with open(nb1373_summary_path) as f:
        nb1373_summary = json.load(f)
    top_bit_idx = np.array(nb1373_summary["top_atompair_bit_indices_ranked"],
                           dtype=int)
    top_k = int(nb1373_summary["top_k_atompair"])
    assert top_bit_idx.shape[0] == top_k, "top-k mismatch vs summary"
    print(f"[reuse] nb1373 top-{top_k} AtomPair bit indices loaded from summary")
    print(f"        bits (ranked) = {top_bit_idx.tolist()}")

    # ---- ChEMBL pool ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL (local cache; same union as nb1373)")
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
    top_idx, top_sim = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx, top_sim, pool_labels, fallback=pool_median
    )
    print(f"   pred_chembl_pec50  mean={pred_chembl_pec50.mean():.3f}  "
          f"std={pred_chembl_pec50.std():.3f}")

    # ---- AtomPair-2048 (unblind slice) -> PRUNED 32-col matrix ----
    if not ATOMPAIR_TE_PATH.exists():
        raise FileNotFoundError(
            f"AtomPair test cache missing: {ATOMPAIR_TE_PATH}"
        )
    X_ap_te = np.load(ATOMPAIR_TE_PATH)
    if X_ap_te.shape[0] != n_test:
        raise ValueError(f"AtomPair cache shape mismatch: {X_ap_te.shape}")
    X_ap_unb = X_ap_te[unb_idx].astype(np.float32)
    n_ap = int(X_ap_te.shape[1])

    pred_chembl_unb = pred_chembl_pec50[unb_idx].astype(np.float32)
    mean_sim_unb = mean_sim[unb_idx].astype(np.float32)
    X_ap_unb_pruned = X_ap_unb[:, top_bit_idx]
    X_unb_pruned = np.concatenate(
        [
            X_ap_unb_pruned,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    print(f"   PRUNED feature matrix: {X_unb_pruned.shape}  "
          f"(top-{top_k} AtomPair + pred_chembl + sim)")

    # ---- Outer-bag loop ----
    print("\n" + "-" * 78)
    print(f"OUTER-BAG LOOP (5 outer seeds x 5 inner seeds x 5 folds = 125 fits)")
    print("-" * 78)

    n_outer = len(OUTER_SEEDS)
    per_outer_mean = np.zeros((n_outer, n_unb), dtype=np.float64)
    per_outer_median = np.zeros((n_outer, n_unb), dtype=np.float64)
    per_outer_rae_mean: list[float] = []
    per_outer_rae_median: list[float] = []
    per_outer_inner_seeds: list[list[int]] = []
    per_outer_inner_rae: list[list[float]] = []

    for oi, o in enumerate(OUTER_SEEDS):
        inner_seeds = [o * 1000 + s for s in INNER_BASE_SEEDS]
        per_outer_inner_seeds.append([int(s) for s in inner_seeds])
        inner_corr = np.zeros((len(inner_seeds), n_unb), dtype=np.float64)
        inner_rae_list: list[float] = []
        print(f"\n   outer seed {o}:  inner seeds = {inner_seeds}")
        for ii, s in enumerate(inner_seeds):
            resid_oof_s = _residual_cross_fit_one_seed(X_unb_pruned, residual, s)
            pred_corr_s = anchor + resid_oof_s
            inner_corr[ii] = pred_corr_s
            rae_s = float(rae(y_unb, pred_corr_s))
            inner_rae_list.append(rae_s)
            print(f"      inner seed {s:6d}  rae = {rae_s:.4f}")
        per_outer_inner_rae.append([float(x) for x in inner_rae_list])
        inner_bag_mean = inner_corr.mean(axis=0)
        inner_bag_median = np.median(inner_corr, axis=0)
        per_outer_mean[oi] = inner_bag_mean
        per_outer_median[oi] = inner_bag_median
        rae_mean_o = float(rae(y_unb, inner_bag_mean))
        rae_median_o = float(rae(y_unb, inner_bag_median))
        per_outer_rae_mean.append(rae_mean_o)
        per_outer_rae_median.append(rae_median_o)
        if o == 0:
            d_repro = rae_mean_o - NB1373_REF_MEAN_BAG
            print(f"      [REPRO check] outer=0 mean-bag rae = {rae_mean_o:.4f}  "
                  f"(d vs nb1373 ref {NB1373_REF_MEAN_BAG:.4f} = {d_repro:+.4f})")
        else:
            print(f"      outer={o:3d} mean-bag rae = {rae_mean_o:.4f}   "
                  f"median-bag rae = {rae_median_o:.4f}")

    per_outer_rae_mean_arr = np.array(per_outer_rae_mean)
    per_outer_rae_median_arr = np.array(per_outer_rae_median)
    outer_mean_mean = float(per_outer_rae_mean_arr.mean())
    outer_mean_std = float(per_outer_rae_mean_arr.std())
    outer_mean_min = float(per_outer_rae_mean_arr.min())
    outer_mean_max = float(per_outer_rae_mean_arr.max())
    outer_mean_median = float(np.median(per_outer_rae_mean_arr))

    # ---- Bag-of-bags (BoB) row-level aggregation ----
    bob_mean_oof = per_outer_mean.mean(axis=0)
    bob_median_oof = np.median(per_outer_mean, axis=0)
    rae_bob_mean = float(rae(y_unb, bob_mean_oof))
    rae_bob_median = float(rae(y_unb, bob_median_oof))

    # Also: BoB built on outer-median bags (sanity)
    bob_mean_of_medians = per_outer_median.mean(axis=0)
    bob_median_of_medians = np.median(per_outer_median, axis=0)
    rae_bob_mean_of_medians = float(rae(y_unb, bob_mean_of_medians))
    rae_bob_median_of_medians = float(rae(y_unb, bob_median_of_medians))

    print("\n" + "=" * 78)
    print("OUTER-BAG SUMMARY")
    print("=" * 78)
    for oi, o in enumerate(OUTER_SEEDS):
        print(f"   outer {o:3d}:  mean-bag = {per_outer_rae_mean[oi]:.4f}   "
              f"median-bag = {per_outer_rae_median[oi]:.4f}   "
              f"inner seeds = {per_outer_inner_seeds[oi]}")
    print(f"\n   per-outer mean-bag RAE list = "
          f"[{', '.join(f'{r:.4f}' for r in per_outer_rae_mean)}]")
    print(f"   per-outer mean   = {outer_mean_mean:.4f}")
    print(f"   per-outer std    = {outer_mean_std:.4f}")
    print(f"   per-outer min    = {outer_mean_min:.4f}")
    print(f"   per-outer max    = {outer_mean_max:.4f}")
    print(f"   per-outer median = {outer_mean_median:.4f}")
    print(f"   BoB MEAN   pooled RAE (mean of 5 outer-mean vectors)  = "
          f"{rae_bob_mean:.4f}")
    print(f"   BoB MEDIAN pooled RAE (median of 5 outer-mean vectors) = "
          f"{rae_bob_median:.4f}")
    print(f"   BoB(mean-of-medians)   pooled RAE = {rae_bob_mean_of_medians:.4f}")
    print(f"   BoB(median-of-medians) pooled RAE = {rae_bob_median_of_medians:.4f}")

    # ---- Verdict ----
    repro_delta = per_outer_rae_mean[0] - NB1373_REF_MEAN_BAG
    repro_ok = abs(repro_delta) < DECISION_MARGIN

    # Per-task spec: verdict NB1373_REPRODUCES if PER-OUTER MEAN within 0.003 of 0.5095
    per_outer_mean_within_margin = abs(outer_mean_mean - NB1373_REF_MEAN_BAG) \
        < DECISION_MARGIN

    if per_outer_mean_within_margin:
        verdict = "NB1373_REPRODUCES"
    else:
        if outer_mean_mean < NB1373_REF_MEAN_BAG - DECISION_MARGIN:
            verdict = "NB1373_OPTIMISTIC_BoB_BETTER_THAN_REF"
        else:
            verdict = "NB1373_LUCKY_SEED_PER_OUTER_MEAN_WORSE_THAN_REF"

    print("\n" + "-" * 78)
    print(f"VERDICT (per-outer-mean within 0.003 of nb1373 ref {NB1373_REF_MEAN_BAG:.4f}):")
    print(f"   per-outer-mean = {outer_mean_mean:.4f}  "
          f"(d vs ref = {outer_mean_mean - NB1373_REF_MEAN_BAG:+.4f})")
    print(f"   outer=0 mean-bag = {per_outer_rae_mean[0]:.4f}  "
          f"(d vs ref = {repro_delta:+.4f})  repro_ok = {repro_ok}")
    print(f"   verdict = {verdict}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_bob_mean_oof.npy",
            bob_mean_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_bob_median_oof.npy",
            bob_median_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_per_outer_mean_oof.npy",
            per_outer_mean.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_per_outer_median_oof.npy",
            per_outer_median.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_bob_mean_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_bob_median_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_per_outer_mean_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_per_outer_median_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "nb1373_ref_mean_bag": NB1373_REF_MEAN_BAG,
        "decision_margin": DECISION_MARGIN,
        "n_unb": n_unb,
        "n_atompair_bits": n_ap,
        "top_k_atompair": int(top_k),
        "top_atompair_bit_indices_ranked": [int(b) for b in top_bit_idx.tolist()],
        "outer_seeds": OUTER_SEEDS,
        "inner_base_seeds": INNER_BASE_SEEDS,
        "per_outer_inner_seeds": per_outer_inner_seeds,
        "per_outer_inner_rae": per_outer_inner_rae,
        "per_outer_rae_mean_bag": [float(x) for x in per_outer_rae_mean],
        "per_outer_rae_median_bag": [float(x) for x in per_outer_rae_median],
        "outer_mean_mean": outer_mean_mean,
        "outer_mean_std": outer_mean_std,
        "outer_mean_min": outer_mean_min,
        "outer_mean_max": outer_mean_max,
        "outer_mean_median": outer_mean_median,
        "rae_bob_mean": rae_bob_mean,
        "rae_bob_median": rae_bob_median,
        "rae_bob_mean_of_medians": rae_bob_mean_of_medians,
        "rae_bob_median_of_medians": rae_bob_median_of_medians,
        "rae_anchor_nb1070": rae_anchor,
        "outer0_mean_bag_rae": float(per_outer_rae_mean[0]),
        "outer0_repro_delta_vs_nb1373": float(repro_delta),
        "outer0_repro_ok": bool(repro_ok),
        "per_outer_mean_within_margin": bool(per_outer_mean_within_margin),
        "verdict": verdict,
        "resid_folds": RESID_FOLDS,
        "lgbm_max_depth": 3,
        "lgbm_num_leaves": 7,
        "lgbm_n_estimators": 80,
        "lgbm_learning_rate": 0.05,
        "lgbm_min_child_samples": 20,
        "lgbm_huber_alpha": 1.0,
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
        "n_unb", "outer_seeds", "per_outer_rae_mean_bag",
        "outer_mean_mean", "outer_mean_std",
        "outer_mean_min", "outer_mean_max",
        "rae_bob_mean", "rae_bob_median",
        "outer0_mean_bag_rae", "outer0_repro_delta_vs_nb1373",
        "outer0_repro_ok",
        "per_outer_mean_within_margin",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
