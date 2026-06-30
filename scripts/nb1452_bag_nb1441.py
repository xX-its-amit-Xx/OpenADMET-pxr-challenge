"""nb1452 -- Outer-bag VALIDATION of nb1441 (CatBoost MAE on 82-col 3-way
pruned features).

Same 82-col feature recipe as nb1441 (top-30 AtomPair + top-20 MACCS +
top-30 Mordred + pred_chembl + sim) but repeats the underlying inner
5-seed CatBoost bag across five OUTER seeds {0, 1, 7, 42, 137} with
inner seeds reparameterized as
    inner_seeds(o) = [o*1000 + s for s in {0, 1, 7, 42, 137}].

For each outer seed o:
    nb1441_o = inner-mean bag of 5 CatBoost(MAE, depth=4, n_est=200,
               lr=0.05, l2=5, random_seed=isd) residual learners,
               each evaluated 5-fold cross-fit -> (253,)
Aggregates:
    per_outer_rae   = rae(y_unb, nb1441_o) for o in OUTER_SEEDS
    bob_mean_oof    = row-mean   of 5 nb1441_o vectors -> pooled RAE
    bob_median_oof  = row-median of 5 nb1441_o vectors -> pooled RAE

Verdict NB1441_REPRODUCES iff |per_outer_mean - 0.5051| < 0.003.

Outputs:
    scripts/nb1452_bag_nb1441.py                          (this file)
    data/processed/nb1452_summary.json
    data/processed/nb1452_bob_mean_oof.npy                (253,) float32
    data/processed/nb1452_bob_median_oof.npy              (253,) float32
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
from catboost import CatBoostRegressor
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1452"
ANCHOR = "nb1070"

OUTER_SEEDS = [0, 1, 7, 42, 137]
INNER_BASE_SEEDS = [0, 1, 7, 42, 137]
RESID_FOLDS = 5

# nb1441 reference (per-seed mean pooled RAE at outer=0).
NB1441_REF = 0.5051
REPRODUCE_MARGIN = 0.003

# nb1422 BoB references for orthogonality comparison.
NB1422_BOB_MEAN_REF = 0.5022
NB1422_BOB_MEDIAN_REF = 0.5016

MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
NB1364_SUMMARY = DATA_PROCESSED / "nb1364_summary.json"
NB1373_SUMMARY = DATA_PROCESSED / "nb1373_summary.json"

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


def _cat_params(seed: int) -> dict:
    return dict(
        loss_function="MAE",
        depth=4,
        iterations=200,
        learning_rate=0.05,
        l2_leaf_reg=5.0,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=2,
    )


def _residual_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray,
                                 seed: int) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = CatBoostRegressor(**_cat_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


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


def _build_82col_unblind_features(y_unb: np.ndarray, anchor: np.ndarray):
    """Build the 82-col 3-way pruned feature matrix on the 253 unblind rows."""
    print("=" * 78)
    print("BUILD 82-col 3-way pruned features (same recipe as nb1441)")
    print("=" * 78)

    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    n_unb = len(y_unb)

    # ---- Reuse SHAP picks from nb1352 + nb1364 + nb1373 ----
    for p in (NB1352_SUMMARY, NB1364_SUMMARY, NB1373_SUMMARY):
        if not p.exists():
            raise FileNotFoundError(f"missing {p} -- run prerequisite first")
    with open(NB1352_SUMMARY) as f:
        sum_1352 = json.load(f)
    with open(NB1364_SUMMARY) as f:
        sum_1364 = json.load(f)
    with open(NB1373_SUMMARY) as f:
        sum_1373 = json.load(f)
    top_maccs_bit_idx = np.array(
        sum_1352["top_maccs_bit_indices_ranked"], dtype=int
    )
    top_mord_col_idx = np.array(
        sum_1364["top_mordred_col_indices_ranked"], dtype=int
    )
    top_ap_bit_idx = np.array(
        sum_1373["top_atompair_bit_indices_ranked"], dtype=int
    )
    n_top_maccs = int(len(top_maccs_bit_idx))
    n_top_mord = int(len(top_mord_col_idx))
    n_top_ap = int(len(top_ap_bit_idx))
    print(f"[reuse] top-{n_top_maccs} MACCS bits    (from nb1352)")
    print(f"[reuse] top-{n_top_mord} Mordred cols  (from nb1364)")
    print(f"[reuse] top-{n_top_ap} AtomPair bits (from nb1373)")

    # ---- MACCS-167 (unblind slice -> top-K cols) ----
    X_maccs_te = np.load(MACCS_TE_PATH)
    X_maccs_unb = X_maccs_te[unb_idx].astype(np.float32)
    X_maccs_unb_top = X_maccs_unb[:, top_maccs_bit_idx].astype(np.float32)
    print(f"[feat] X_maccs_unb_top shape  = {X_maccs_unb_top.shape}")

    # ---- Mordred (unblind slice -> top-K cols) ----
    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    X_mord_unb = X_mord_te[unb_idx].astype(np.float32)
    X_mord_unb_top = X_mord_unb[:, top_mord_col_idx].astype(np.float32)
    print(f"[feat] X_mord_unb_top shape    = {X_mord_unb_top.shape}")

    # ---- AtomPair (unblind slice -> top-K bits) ----
    if not ATOMPAIR_TE_PATH.exists():
        raise FileNotFoundError(
            f"AtomPair test cache missing: {ATOMPAIR_TE_PATH}"
        )
    X_ap_te = np.load(ATOMPAIR_TE_PATH)
    X_ap_unb = X_ap_te[unb_idx].astype(np.float32)
    X_ap_unb_top = X_ap_unb[:, top_ap_bit_idx].astype(np.float32)
    print(f"[feat] X_ap_unb_top shape      = {X_ap_unb_top.shape}")

    # ---- ChEMBL pool + kNN feature build (513-level) ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL (same union as nb1441)")
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

    # ---- Build TRIPLE-PRUNED 82-col feature matrix on 253 ----
    X_unb = np.concatenate(
        [
            X_maccs_unb_top,
            X_mord_unb_top,
            X_ap_unb_top,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim = X_unb.shape[1]
    expected_dim = n_top_maccs + n_top_mord + n_top_ap + 2
    if feat_dim != expected_dim:
        raise ValueError(f"feat_dim {feat_dim} != expected {expected_dim}")
    print(f"\n   3-WAY PRUNED feature matrix: {X_unb.shape}  "
          f"(top-{n_top_maccs} MACCS + top-{n_top_mord} Mordred + "
          f"top-{n_top_ap} AtomPair + pred_chembl + sim)")

    meta = {
        "n_top_maccs": n_top_maccs,
        "n_top_mordred": n_top_mord,
        "n_top_atompair": n_top_ap,
        "feat_dim": int(feat_dim),
        "n_chembl_pool": int(len(pool)),
        "feat_breakdown": {
            "maccs": n_top_maccs,
            "mordred": n_top_mord,
            "atompair": n_top_ap,
            "pred_chembl_pec50": 1,
            "mean_sim": 1,
            "total": int(feat_dim),
        },
    }
    return X_unb, meta


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- OUTER-BAG VALIDATION of nb1441 "
          f"(CatBoost MAE on 82-col 3-way pruned features)")
    print(f"         outer seeds      = {OUTER_SEEDS}")
    print(f"         inner base seeds = {INNER_BASE_SEEDS}")
    print(f"         inner_seeds(o)   = [o*1000 + s for s in base]")
    print(f"         nb1441 ref per-outer mean = {NB1441_REF:.4f}  "
          f"margin = {REPRODUCE_MARGIN}")
    print(f"         nb1422 BoB mean ref = {NB1422_BOB_MEAN_REF:.4f}  "
          f"median ref = {NB1422_BOB_MEDIAN_REF:.4f}")
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

    # ---- Build 82-col features once ----
    X_unb, feat_meta = _build_82col_unblind_features(y_unb, anchor)
    feat_dim = int(X_unb.shape[1])

    # ---- Outer x Inner cross-fit ----
    print("\n" + "=" * 78)
    print(f"OUTER x INNER CATBOOST RESIDUAL CROSS-FIT (dim={feat_dim})")
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

    # ---- BoB row-level aggregations across 5 nb1441_o vectors ----
    bob_mean_oof = outer_mean_bag.mean(axis=0)
    bob_median_oof = np.median(outer_mean_bag, axis=0)
    rae_bob_mean = float(rae(y_unb, bob_mean_oof))
    rae_bob_median = float(rae(y_unb, bob_median_oof))

    print("\n" + "=" * 78)
    print("OUTER-BAG SUMMARY")
    print("=" * 78)
    print(f"   per-outer nb1441 RAE list = "
          f"[{', '.join(f'{r:.4f}' for r in per_outer_rae_blend)}]")
    print(f"   per-outer mean   = {per_outer_mean:.4f}")
    print(f"   per-outer std    = {per_outer_std:.4f}")
    print(f"   per-outer min    = {per_outer_min:.4f}")
    print(f"   per-outer max    = {per_outer_max:.4f}")
    print(f"   per-outer median = {per_outer_median:.4f}")
    print(f"   BoB MEAN   pooled RAE = {rae_bob_mean:.4f}  "
          f"(d vs nb1422m = {rae_bob_mean - NB1422_BOB_MEAN_REF:+.4f})")
    print(f"   BoB MEDIAN pooled RAE = {rae_bob_median:.4f}  "
          f"(d vs nb1422md = {rae_bob_median - NB1422_BOB_MEDIAN_REF:+.4f})")

    # ---- Pearson vs prior BoB OOFs (orthogonality) ----
    def _pearson_vs(path: Path, vec: np.ndarray):
        if not path.exists():
            return None
        oof = np.load(path).astype(np.float64)
        if oof.shape[0] != n_unb:
            return None
        return float(np.corrcoef(vec, oof)[0, 1])

    pearson_bobmean_vs_nb1422_mean = _pearson_vs(
        DATA_PROCESSED / "nb1422_bob_mean_oof.npy", bob_mean_oof
    )
    pearson_bobmean_vs_nb1422_median = _pearson_vs(
        DATA_PROCESSED / "nb1422_bob_median_oof.npy", bob_mean_oof
    )
    pearson_bobmedian_vs_nb1422_mean = _pearson_vs(
        DATA_PROCESSED / "nb1422_bob_mean_oof.npy", bob_median_oof
    )
    pearson_bobmedian_vs_nb1422_median = _pearson_vs(
        DATA_PROCESSED / "nb1422_bob_median_oof.npy", bob_median_oof
    )
    pearson_bobmean_vs_nb1441_mean = _pearson_vs(
        DATA_PROCESSED / "nb1441_mean_bag_oof.npy", bob_mean_oof
    )

    if pearson_bobmean_vs_nb1422_mean is not None:
        print(f"   Pearson(bob_mean,   nb1422_bob_mean)  "
              f"= {pearson_bobmean_vs_nb1422_mean:.4f}")
    if pearson_bobmean_vs_nb1422_median is not None:
        print(f"   Pearson(bob_mean,   nb1422_bob_med )  "
              f"= {pearson_bobmean_vs_nb1422_median:.4f}")
    if pearson_bobmedian_vs_nb1422_mean is not None:
        print(f"   Pearson(bob_median, nb1422_bob_mean)  "
              f"= {pearson_bobmedian_vs_nb1422_mean:.4f}")
    if pearson_bobmedian_vs_nb1422_median is not None:
        print(f"   Pearson(bob_median, nb1422_bob_med )  "
              f"= {pearson_bobmedian_vs_nb1422_median:.4f}")
    if pearson_bobmean_vs_nb1441_mean is not None:
        print(f"   Pearson(bob_mean,   nb1441_mean_bag)  "
              f"= {pearson_bobmean_vs_nb1441_mean:.4f}")

    # ---- Verdict ----
    delta_per_outer = per_outer_mean - NB1441_REF
    reproduces = abs(delta_per_outer) < REPRODUCE_MARGIN
    if reproduces:
        verdict = "NB1441_REPRODUCES"
    elif per_outer_mean < NB1441_REF - REPRODUCE_MARGIN:
        verdict = "NB1441_OPTIMISTIC_OUTER_BAG_BETTER"
    else:
        verdict = "NB1441_LUCKY_SEED_OUTER_BAG_WORSE"

    delta_outer0 = per_outer_rae_blend[0] - NB1441_REF
    outer0_reproduces = abs(delta_outer0) < REPRODUCE_MARGIN

    beats_nb1422_bob_mean = rae_bob_mean < NB1422_BOB_MEAN_REF - REPRODUCE_MARGIN
    beats_nb1422_bob_median = (
        rae_bob_median < NB1422_BOB_MEDIAN_REF - REPRODUCE_MARGIN
    )

    print("\n" + "-" * 78)
    print(f"VERDICT (per-outer-mean within {REPRODUCE_MARGIN} of nb1441 ref "
          f"{NB1441_REF:.4f}):")
    print(f"   per-outer-mean = {per_outer_mean:.4f}   "
          f"(d vs ref = {delta_per_outer:+.4f})")
    print(f"   outer=0 bag    = {per_outer_rae_blend[0]:.4f}   "
          f"(d vs ref = {delta_outer0:+.4f})   "
          f"outer0_reproduces = {outer0_reproduces}")
    print(f"   verdict = {verdict}")
    print(f"   BoB beats nb1422 bob_mean   : {beats_nb1422_bob_mean}")
    print(f"   BoB beats nb1422 bob_median : {beats_nb1422_bob_median}")

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
        "data_source": "MACCS-cache + Mordred-cached_nb1030 + AtomPair-cache "
                       "+ local_chembl_caches_union",
        "model_family": "CatBoost",
        "catboost_loss": "MAE",
        "catboost_depth": 4,
        "catboost_iterations": 200,
        "catboost_learning_rate": 0.05,
        "catboost_l2_leaf_reg": 5.0,
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
        "per_outer_rae_nb1441": per_outer_rae_blend,
        "per_outer_mean": per_outer_mean,
        "per_outer_std": per_outer_std,
        "per_outer_min": per_outer_min,
        "per_outer_max": per_outer_max,
        "per_outer_median": per_outer_median,
        "rae_bob_mean": rae_bob_mean,
        "rae_bob_median": rae_bob_median,
        "nb1441_ref": NB1441_REF,
        "nb1422_bob_mean_ref": NB1422_BOB_MEAN_REF,
        "nb1422_bob_median_ref": NB1422_BOB_MEDIAN_REF,
        "reproduce_margin": REPRODUCE_MARGIN,
        "delta_per_outer_mean_vs_nb1441": delta_per_outer,
        "delta_outer0_vs_nb1441": delta_outer0,
        "delta_bob_mean_vs_nb1422_bob_mean":
            rae_bob_mean - NB1422_BOB_MEAN_REF,
        "delta_bob_median_vs_nb1422_bob_median":
            rae_bob_median - NB1422_BOB_MEDIAN_REF,
        "outer0_reproduces": bool(outer0_reproduces),
        "reproduces": bool(reproduces),
        "beats_nb1422_bob_mean": bool(beats_nb1422_bob_mean),
        "beats_nb1422_bob_median": bool(beats_nb1422_bob_median),
        "pearson_bobmean_vs_nb1422_bob_mean": pearson_bobmean_vs_nb1422_mean,
        "pearson_bobmean_vs_nb1422_bob_median":
            pearson_bobmean_vs_nb1422_median,
        "pearson_bobmedian_vs_nb1422_bob_mean":
            pearson_bobmedian_vs_nb1422_mean,
        "pearson_bobmedian_vs_nb1422_bob_median":
            pearson_bobmedian_vs_nb1422_median,
        "pearson_bobmean_vs_nb1441_mean_bag":
            pearson_bobmean_vs_nb1441_mean,
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
        "per_outer_rae_nb1441",
        "per_outer_mean", "per_outer_std",
        "per_outer_min", "per_outer_max", "per_outer_median",
        "rae_bob_mean", "rae_bob_median",
        "delta_per_outer_mean_vs_nb1441",
        "delta_outer0_vs_nb1441",
        "delta_bob_mean_vs_nb1422_bob_mean",
        "delta_bob_median_vs_nb1422_bob_median",
        "outer0_reproduces", "reproduces",
        "beats_nb1422_bob_mean", "beats_nb1422_bob_median",
        "pearson_bobmean_vs_nb1422_bob_mean",
        "pearson_bobmean_vs_nb1441_mean_bag",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
