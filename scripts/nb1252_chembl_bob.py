"""nb1252 -- OUTER-BAG validation of nb1242 (ChEMBL PXR kNN residual feature on
nb1070 anchor; mean-bag pooled RAE 0.5431).

PRECEDENT
---------
nb1242 added two ChEMBL-derived columns -- pred_chembl_pec50 (Tanimoto-k5 mean
of nearest ChEMBL pEC50) and the mean5 similarity -- to the MACCS-167 residual
feature matrix on the 253 unblind rows.  Inner seeds {0, 1, 7, 42, 137} on the
SAME 5-fold KFold cross-fit family produced per-seed RAE in [0.5471, 0.5552]
and mean-bag pooled RAE 0.5431 -- a -0.020 improvement over the MACCS-only
nb1183 0.5513 baseline but still 0.002 above nb1211 (0.5451).

The question this script answers
--------------------------------
Is 0.5431 reproducible under outer-seed perturbation, or did it land on a
lucky inner-seed family?  We rebuild the entire ChEMBL kNN + MACCS residual
LGBM bag five times, each time with a fresh inner-seed quintet driven by an
outer seed.

PROTOCOL
--------
OUTER SEEDS  : {0, 1, 7, 42, 137}
INNER FAMILY : inner_seeds(o) = [o * 1000 + s for s in {0, 1, 7, 42, 137}]
    => outer 0    -> inner [    0,     1,     7,    42,   137]
       outer 1    -> inner [ 1000,  1001,  1007,  1042,  1137]
       outer 7    -> inner [ 7000,  7001,  7007,  7042,  7137]
       outer 42   -> inner [42000, 42001, 42007, 42042, 42137]
       outer 137  -> inner [137000,137001,137007,137042,137137]

For each outer seed o:
  inner_seeds = [o * 1000 + s for s in {0, 1, 7, 42, 137}]
  build 5-seed mean bag of shallow LGBM Huber on the SAME residual feature
  matrix (MACCS-167 + pred_chembl_pec50 + sim) over residual
  = y_unb - nb1070_pred_oof.  Cross-fit 5-fold KFold per inner seed,
  with the inner seed driving BOTH the KFold split AND the LGBM
  random_state -- identical to nb1242's primitive.
  Add anchor -> corrected OOF; mean across 5 inner seeds -> nb1252_o (253,)
  pooled RAE on 253 unblind -> per-outer RAE.

BAG-OF-BAGS
-----------
Row-level mean and median across the 5 per-outer corrected OOFs.
pooled RAE(bob_mean) and pooled RAE(bob_median) reported.

BLEND TEST
----------
Combine nb1252 BoB mean (this run) with nb1211 mean OOF via naive 0.5/0.5
mean.  Pooled RAE compared to nb1252 BoB mean alone and to nb1211 mean
alone.  This tests whether the outer-bagged ChEMBL variant is a stronger
blend partner than the standalone nb1242 mean bag (which the SLSQP in
nb1211 used to combine with).

VERDICT
-------
NB1242_REPRODUCES if abs(per_outer_mean - 0.5431) <= 0.003.
Otherwise: NB1242_PESSIMISTIC (outer bag better) or NB1242_LUCKY (pulls up).

Outputs:
  data/processed/nb1252_per_seed_corrected_oof.npy  (5, 253) float32  per
                                                                    outer
  data/processed/nb1252_bob_mean_oof.npy            (253,)   float32  row
                                                                    mean
  data/processed/nb1252_bob_median_oof.npy          (253,)   float32  row
                                                                    median
  data/processed/nb1252_summary.json
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

TAG = "nb1252"
ANCHOR = "nb1070"

OUTER_SEEDS = [0, 1, 7, 42, 137]
INNER_BASE = [0, 1, 7, 42, 137]  # inner = outer * 1000 + base (matches nb1242)
RESID_FOLDS = 5

# Cached paths.
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"
NB1211_MEAN_OOF_PATH = DATA_PROCESSED / "nb1211_mean_oof.npy"

# ChEMBL pool filters (identical to nb1242).
KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

# Reference: nb1242 mean-bag pooled RAE on 253 unblind.
NB1242_MEAN_BAG_REF = 0.5431
NB1211_MEAN_REF = 0.5451
REPRO_MARGIN = 0.003


# -----------------------------------------------------------------------------
# ChEMBL pool builder (replicates nb1242 logic exactly).
# -----------------------------------------------------------------------------
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
        raise FileNotFoundError(
            "No local ChEMBL PXR parquets found in data/external/"
        )

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
        .rename(columns={"src_first": "src"})
    )
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
    w = np.clip(top_sim.copy(), 0.0, 1.0)
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


# -----------------------------------------------------------------------------
# Residual cross-fit (identical capacity to nb1242).
# -----------------------------------------------------------------------------
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


def _residual_cross_fit_one_seed(X, residual, seed):
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


# -----------------------------------------------------------------------------
# Main.
# -----------------------------------------------------------------------------
def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- OUTER-BAG validation of nb1242 (ChEMBL kNN feature + MACCS-167")
    print(f"          residual LGBM bag on {ANCHOR} anchor)  ref = {NB1242_MEAN_BAG_REF:.4f}")
    print(f"          OUTER seeds  = {OUTER_SEEDS}")
    print(f"          inner family = inner_seeds(o) = [o*1000 + s for s in "
          f"{INNER_BASE}]")
    print(f"          residual     = y_unb - {ANCHOR}_pred_oof")
    print(f"          features     = MACCS-167 + pred_chembl_pec50 + sim  (169)")
    print(f"          LGBM         = depth=3, leaves=7, n_est=80, lr=0.05,")
    print(f"                          min_child_samples=20, obj=huber(alpha=1.0)")
    print(f"          repro margin = {REPRO_MARGIN:.3f}")
    print("=" * 78)

    # ---- Load truth + anchor ----
    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    anchor_oof = np.load(DATA_PROCESSED / f"{ANCHOR}_pred_oof.npy").astype(np.float64)
    if anchor_oof.shape[0] != n_unb:
        raise ValueError(f"{ANCHOR} shape mismatch: {anchor_oof.shape} vs {n_unb}")
    rae_anchor = float(rae(y_unb, anchor_oof))
    residual = y_unb - anchor_oof
    print(f"[load] {ANCHOR}_pred_oof pooled RAE = {rae_anchor:.4f}")
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- ChEMBL pool ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL (local cache; same union as nb1242)")
    print("-" * 78)
    pool = _load_chembl_pool()

    # Test InChIKey leak guard.
    test_mols = [standardize(s) for s in test_smiles]
    test_inchikeys = set()
    for m in test_mols:
        ik = _safe_inchikey(m)
        if ik is not None:
            test_inchikeys.add(ik)
    n_before = len(pool)
    pool = pool[~pool["inchikey"].isin(test_inchikeys)].reset_index(drop=True)
    n_after = len(pool)
    print(f"   leak-guard: pool {n_before} -> {n_after}")

    # Morgan FPs.
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
        std_test_smiles.append(Chem.MolToSmiles(m) if m is not None else "")
    fp_test = morgan_fp_batch(std_test_smiles)

    # kNN k=5 Tanimoto.
    top_idx, top_sim = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx, top_sim, pool_labels, fallback=pool_median
    )
    print(f"   pred_chembl_pec50  mean={pred_chembl_pec50.mean():.3f}  "
          f"std={pred_chembl_pec50.std():.3f}")
    n_zero_neighbor = int((top_sim[:, 0] < SIM_FLOOR).sum())
    print(f"   {n_zero_neighbor}/{n_test} test rows had no neighbor (fallback)")

    # ---- MACCS unb slice ----
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
    print(f"   residual feature matrix: {X_unb.shape}  "
          f"(MACCS-167 + pred_chembl + sim)")

    # ---- Per-outer rebuild ----
    print("\n" + "-" * 78)
    print(f"PER-OUTER REBUILD  ({len(OUTER_SEEDS)} outer x {len(INNER_BASE)} inner x "
          f"{RESID_FOLDS} folds = "
          f"{len(OUTER_SEEDS) * len(INNER_BASE) * RESID_FOLDS} LGBM fits)")
    print("-" * 78)

    per_outer_corrected = np.zeros((len(OUTER_SEEDS), n_unb), dtype=np.float64)
    per_outer_rae: list[float] = []
    per_outer_records: list[dict] = []

    for oi, o in enumerate(OUTER_SEEDS):
        t_outer = time.time()
        inner_seeds = [int(o) * 1000 + int(s) for s in INNER_BASE]
        bag = np.zeros((len(inner_seeds), n_unb), dtype=np.float64)
        inner_rae_list: list[float] = []
        for j, s in enumerate(inner_seeds):
            resid_oof = _residual_cross_fit_one_seed(X_unb, residual, s)
            bag[j] = anchor_oof + resid_oof
            inner_rae_list.append(float(rae(y_unb, bag[j])))
        outer_mean_pred = bag.mean(axis=0)
        rae_outer = float(rae(y_unb, outer_mean_pred))
        per_outer_corrected[oi] = outer_mean_pred
        per_outer_rae.append(rae_outer)

        per_outer_records.append({
            "outer_seed": int(o),
            "inner_seeds": inner_seeds,
            "inner_per_seed_rae": inner_rae_list,
            "rae_mean_bag": rae_outer,
            "delta_vs_nb1242_ref": rae_outer - NB1242_MEAN_BAG_REF,
            "elapsed_sec": round(time.time() - t_outer, 1),
        })
        print(f"   outer {o:5d}  inner={inner_seeds}")
        print(f"     per-inner RAE = "
              f"[{', '.join(f'{r:.4f}' for r in inner_rae_list)}]")
        print(f"     mean-bag pooled RAE = {rae_outer:.4f}   "
              f"(d vs nb1242 = {rae_outer - NB1242_MEAN_BAG_REF:+.4f})   "
              f"elapsed {time.time() - t_outer:.1f}s")

    # ---- Aggregate ----
    per_outer_arr = np.array(per_outer_rae)
    outer_mean = float(per_outer_arr.mean())
    outer_std = float(per_outer_arr.std())
    outer_min = float(per_outer_arr.min())
    outer_max = float(per_outer_arr.max())

    bob_mean_oof = per_outer_corrected.mean(axis=0)
    bob_median_oof = np.median(per_outer_corrected, axis=0)
    rae_bob_mean = float(rae(y_unb, bob_mean_oof))
    rae_bob_median = float(rae(y_unb, bob_median_oof))

    delta = outer_mean - NB1242_MEAN_BAG_REF
    reproduces = abs(delta) <= REPRO_MARGIN
    if reproduces:
        verdict = "NB1242_REPRODUCES"
    elif delta < -REPRO_MARGIN:
        verdict = "NB1242_PESSIMISTIC_OUTER_BAG_BETTER"
    else:
        verdict = "NB1242_LUCKY_OUTER_BAG_PULLS_UP"

    print("\n" + "=" * 78)
    print("OUTER-BAG AGGREGATIONS")
    print("=" * 78)
    print(f"   per-outer mean-bag RAE = "
          f"[{', '.join(f'{r:.4f}' for r in per_outer_rae)}]")
    print(f"   per-outer mean   = {outer_mean:.4f}")
    print(f"   per-outer std    = {outer_std:.4f}")
    print(f"   per-outer min    = {outer_min:.4f}")
    print(f"   per-outer max    = {outer_max:.4f}")
    print(f"   bag-of-bags MEAN   pooled RAE = {rae_bob_mean:.4f}")
    print(f"   bag-of-bags MEDIAN pooled RAE = {rae_bob_median:.4f}")
    print(f"   nb1242 mean-bag reference     = {NB1242_MEAN_BAG_REF:.4f}")
    print(f"   delta(per-outer mean vs nb1242) = {delta:+.4f}  "
          f"(margin {REPRO_MARGIN:.3f})")
    print(f"   VERDICT = {verdict}")

    # ---- Blend test: nb1252 BoB mean + nb1211 mean ----
    print("\n" + "-" * 78)
    print("BLEND TEST: nb1252 BoB mean + nb1211 mean OOF (naive 0.5/0.5 mean)")
    print("-" * 78)
    if not NB1211_MEAN_OOF_PATH.exists():
        raise FileNotFoundError(f"{NB1211_MEAN_OOF_PATH} missing")
    nb1211_mean = np.load(NB1211_MEAN_OOF_PATH).astype(np.float64)
    if nb1211_mean.shape[0] != n_unb:
        raise ValueError(
            f"nb1211_mean_oof shape mismatch: {nb1211_mean.shape} vs n_unb={n_unb}"
        )
    rae_nb1211_alone = float(rae(y_unb, nb1211_mean))
    blend_half = 0.5 * bob_mean_oof + 0.5 * nb1211_mean
    rae_blend_half = float(rae(y_unb, blend_half))
    pred_corr_blend = float(np.corrcoef(bob_mean_oof, nb1211_mean)[0, 1])
    resid_corr_blend = float(np.corrcoef(
        bob_mean_oof - y_unb, nb1211_mean - y_unb
    )[0, 1])

    print(f"   nb1252 BoB mean alone        = {rae_bob_mean:.4f}")
    print(f"   nb1211 mean alone            = {rae_nb1211_alone:.4f}  "
          f"(ref {NB1211_MEAN_REF:.4f})")
    print(f"   pred corr (nb1252, nb1211)   = {pred_corr_blend:.4f}")
    print(f"   resid corr (nb1252, nb1211)  = {resid_corr_blend:.4f}")
    print(f"   0.5/0.5 mean blend           = {rae_blend_half:.4f}")
    print(f"   delta vs nb1252 alone        = "
          f"{rae_blend_half - rae_bob_mean:+.4f}")
    print(f"   delta vs nb1211 alone        = "
          f"{rae_blend_half - rae_nb1211_alone:+.4f}")

    if rae_blend_half < min(rae_bob_mean, rae_nb1211_alone) - REPRO_MARGIN:
        blend_verdict = "OUTER_BAGGED_CHEMBL_STRONGER_BLEND_PARTNER"
    elif rae_blend_half < min(rae_bob_mean, rae_nb1211_alone):
        blend_verdict = "BLEND_MARGINAL_GAIN_BELOW_MARGIN"
    else:
        blend_verdict = "BLEND_DOES_NOT_BEAT_BETTER_COMPONENT"
    print(f"   blend verdict                = {blend_verdict}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_per_seed_corrected_oof.npy",
            per_outer_corrected.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_bob_mean_oof.npy",
            bob_mean_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_bob_median_oof.npy",
            bob_median_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_per_seed_corrected_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_bob_mean_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_bob_median_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "n_unb": n_unb,
        "outer_seeds": OUTER_SEEDS,
        "inner_base": INNER_BASE,
        "inner_family_rule": "inner = outer * 1000 + base  (matches nb1242)",
        "resid_folds": RESID_FOLDS,
        "feature_layout": "MACCS-167 + pred_chembl_pec50 + sim  (169)",
        "n_chembl_pool": int(len(pool)),
        "n_zero_neighbor_rows": n_zero_neighbor,
        "rae_anchor_nb1070": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_outer_records": per_outer_records,
        "per_outer_rae": [float(x) for x in per_outer_rae],
        "outer_mean": outer_mean,
        "outer_std": outer_std,
        "outer_min": outer_min,
        "outer_max": outer_max,
        "rae_bob_mean": rae_bob_mean,
        "rae_bob_median": rae_bob_median,
        "nb1242_mean_bag_ref": NB1242_MEAN_BAG_REF,
        "delta_outer_mean_vs_nb1242": delta,
        "repro_margin": REPRO_MARGIN,
        "reproduces": bool(reproduces),
        "verdict": verdict,
        # ---- Blend test ----
        "blend_partner": "nb1211_mean_oof",
        "rae_nb1211_alone": rae_nb1211_alone,
        "nb1211_mean_ref": NB1211_MEAN_REF,
        "pred_corr_nb1252_nb1211": pred_corr_blend,
        "resid_corr_nb1252_nb1211": resid_corr_blend,
        "rae_blend_nb1252_nb1211_half_half": rae_blend_half,
        "delta_blend_vs_nb1252": rae_blend_half - rae_bob_mean,
        "delta_blend_vs_nb1211": rae_blend_half - rae_nb1211_alone,
        "blend_verdict": blend_verdict,
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
        "n_chembl_pool", "rae_anchor_nb1070",
        "per_outer_rae",
        "outer_mean", "outer_std", "outer_min", "outer_max",
        "rae_bob_mean", "rae_bob_median",
        "delta_outer_mean_vs_nb1242",
        "reproduces", "verdict",
        "rae_nb1211_alone",
        "pred_corr_nb1252_nb1211", "resid_corr_nb1252_nb1211",
        "rae_blend_nb1252_nb1211_half_half",
        "delta_blend_vs_nb1252", "delta_blend_vs_nb1211",
        "blend_verdict",
    ):
        print(f"  {k}: {res.get(k)}")
