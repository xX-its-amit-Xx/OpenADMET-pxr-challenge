"""nb1331 -- Outer-bag of nb1290 best-w blend (BoB-of-BoBs at the blend level).

PROTOCOL
--------
For each OUTER seed o in {0, 1, 7, 42, 137}:
    nb1190_o : triple-FP outer-seed BoB on residual-over-nb1070
               (cached row from nb1190_per_outer_oof.npy[oi]; outer_seeds order
                [0, 1, 7, 42, 137]).
    nb1242_o : ChEMBL+MACCS shallow LGBM Huber bag with INNER seeds
               = [o*1000 + s for s in {0, 1, 7, 42, 137}]
               -- for outer=0 this becomes [0, 1, 7, 42, 137] which reproduces
               the cached nb1242_mean_bag_oof.npy verbatim (used for sanity).
    nb1290_o = 0.35 * nb1190_o + 0.65 * nb1242_o     (nb1290 best-w blend)

Aggregate:
    per_outer_rae  -- pooled RAE(nb1290_o vs y_unb) for each o
    BoB MEAN       -- row-level mean across the 5 outer-seed nb1290_o vectors
    BoB MEDIAN     -- row-level median across the 5 outer-seed nb1290_o vectors

Reference: nb1290 (best fixed-w, w=0.35 on nb1190) pooled RAE = 0.5390
Verdict NB1290_REPRODUCES if abs(per_outer_mean - 0.5390) <= 0.003.
Variance-reduction win if BoB RAE < per_outer_mean.

Outputs:
    scripts/nb1331_bag_nb1290.py                  (this file)
    data/processed/nb1331_summary.json
    data/processed/nb1331_per_outer_oof.npy       (5, 253) float32
    data/processed/nb1331_bob_mean_oof.npy        (253,)   float32
    data/processed/nb1331_bob_median_oof.npy      (253,)   float32
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

TAG = "nb1331"
ANCHOR = "nb1070"

OUTER_SEEDS = [0, 1, 7, 42, 137]
INNER_BASES = [0, 1, 7, 42, 137]   # inner_seed = outer * 1000 + base
RESID_FOLDS = 5

# nb1290 best-w blend coefficients on (nb1190, nb1242).
W_NB1190 = 0.35
W_NB1242 = 0.65

# References.
NB1290_REF = 0.5390
REPRO_MARGIN = 0.003

# ChEMBL pool config (copied from nb1242 -- must match capacity).
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"
KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6


# ---- ChEMBL helpers (verbatim from nb1242) -----------------------------------
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
        raise FileNotFoundError("No ChEMBL caches found")
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
        .rename(columns={"src_first": "src"})
    )
    print(f"   [pool] InChIKey-dedup median agg: {len(agg)} unique cpds")
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
    print(f"{TAG} -- OUTER-BAG of nb1290 best-w blend (BoB-of-BoBs at blend level)")
    print(f"          outer seeds = {OUTER_SEEDS}")
    print(f"          inner bases = {INNER_BASES}   inner = outer*1000 + base")
    print(f"          per outer: nb1290_o = "
          f"{W_NB1190:.2f}*nb1190_o + {W_NB1242:.2f}*nb1242_o")
    print(f"          nb1190_o source: cached nb1190_per_outer_oof.npy[oi]")
    print(f"          nb1242_o source: rebuilt with inner_seeds = o*1000+base")
    print(f"          reference nb1290 (best-w 0.35/0.65) = {NB1290_REF:.4f}")
    print(f"          margin = {REPRO_MARGIN:.3f}")
    print("=" * 78)

    # ---- Truth + anchor ----
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
        raise ValueError(f"anchor shape mismatch")
    rae_anchor = float(rae(y_unb, anchor))
    residual = y_unb - anchor
    print(f"[load] nb1070 anchor RAE = {rae_anchor:.4f}")
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- nb1190 per-outer (cached) ----
    nb1190_p = DATA_PROCESSED / "nb1190_per_outer_oof.npy"
    if not nb1190_p.exists():
        raise FileNotFoundError(f"{nb1190_p} missing (need nb1190 outer rebuild)")
    nb1190_per_outer = np.load(nb1190_p).astype(np.float64)  # (5, 253)
    if nb1190_per_outer.shape != (len(OUTER_SEEDS), n_unb):
        raise ValueError(
            f"nb1190 per-outer shape mismatch: {nb1190_per_outer.shape}"
        )
    print(f"\n[load] nb1190 per-outer cache shape = {nb1190_per_outer.shape}")
    for oi, o in enumerate(OUTER_SEEDS):
        r = float(rae(y_unb, nb1190_per_outer[oi]))
        print(f"   nb1190 outer {o:5d}: RAE = {r:.4f}")

    # Sanity vs cached mean.
    cached_mean = np.load(DATA_PROCESSED / "nb1190_bob_mean_oof.npy").astype(np.float64)
    rec_mean = nb1190_per_outer.mean(axis=0)
    if not np.allclose(cached_mean, rec_mean, atol=1e-5):
        raise ValueError("nb1190 per-outer mean disagrees with cached bob_mean")
    print(f"   nb1190 per-outer mean reconstructs nb1190_bob_mean_oof.npy: OK")

    # ---- Rebuild ChEMBL pool + Morgan FPs ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL (local caches, union)")
    print("-" * 78)
    pool = _load_chembl_pool()

    # Test-set leak guard.
    test_mols = [standardize(s) for s in test_smiles]
    test_inchikeys = set()
    for m in test_mols:
        ik = _safe_inchikey(m)
        if ik is not None:
            test_inchikeys.add(ik)
    n_before = len(pool)
    pool = pool[~pool["inchikey"].isin(test_inchikeys)].reset_index(drop=True)
    n_after = len(pool)
    print(f"   leak guard: pool {n_before} -> {n_after} "
          f"(dropped {n_before - n_after})")

    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep_pool = fp_pool.sum(axis=1) > 0
    if not keep_pool.all():
        pool = pool[keep_pool].reset_index(drop=True)
        fp_pool = fp_pool[keep_pool]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))
    print(f"   final pool size: {len(pool)}  median pEC50 = {pool_median:.3f}")

    std_test_smiles = [Chem.MolToSmiles(m) if m is not None else "" for m in test_mols]
    fp_test = morgan_fp_batch(std_test_smiles)
    print(f"   test FP shape: {fp_test.shape}")

    print("\n" + "-" * 78)
    print(f"TANIMOTO kNN k={KNN_K} (test 513 vs pool {len(pool)})")
    print("-" * 78)
    top_idx, top_sim = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx, top_sim, pool_labels, fallback=pool_median
    )
    print(f"   pred_chembl mean={pred_chembl_pec50.mean():.3f} "
          f"std={pred_chembl_pec50.std():.3f}")
    print(f"   top1 sim p50={np.percentile(top_sim[:, 0], 50):.3f}")

    # MACCS unblind slice.
    X_maccs_te = np.load(MACCS_TE_PATH)
    if X_maccs_te.shape[0] != n_test:
        raise ValueError(f"MACCS shape mismatch: {X_maccs_te.shape}")
    X_maccs_unb = X_maccs_te[unb_idx].astype(np.float32)

    pred_chembl_unb = pred_chembl_pec50[unb_idx].astype(np.float32)
    mean_sim_unb = mean_sim[unb_idx].astype(np.float32)
    X_unb = np.concatenate(
        [X_maccs_unb,
         pred_chembl_unb.reshape(-1, 1),
         mean_sim_unb.reshape(-1, 1)],
        axis=1,
    ).astype(np.float32)
    feat_dim = X_unb.shape[1]
    print(f"   residual feat matrix: {X_unb.shape}")

    # ---- Per-outer rebuild of nb1242 + blend with cached nb1190 ----
    print("\n" + "-" * 78)
    print(f"PER-OUTER REBUILD: nb1242_o (5 inner LGBM Huber bags) + nb1290_o blend")
    print(f"   per outer: {len(INNER_BASES)} inner seeds x {RESID_FOLDS} folds = "
          f"{len(INNER_BASES) * RESID_FOLDS} fits")
    print("-" * 78)

    nb1242_per_outer = np.zeros((len(OUTER_SEEDS), n_unb), dtype=np.float64)
    nb1290_per_outer = np.zeros((len(OUTER_SEEDS), n_unb), dtype=np.float64)
    per_outer_rae_nb1290: list[float] = []
    per_outer_rae_nb1242: list[float] = []
    per_outer_rae_nb1190: list[float] = []
    per_outer_records: list[dict] = []

    for oi, o in enumerate(OUTER_SEEDS):
        t_o = time.time()
        inner_seeds = [int(o) * 1000 + int(b) for b in INNER_BASES]

        # nb1242_o: 5-seed mean-bag of residual LGBM on (MACCS + ChEMBL feats).
        bag = np.zeros((len(inner_seeds), n_unb), dtype=np.float64)
        per_inner_rae = []
        for j, s in enumerate(inner_seeds):
            resid_oof = _residual_cross_fit_one_seed(X_unb, residual, s)
            bag[j] = anchor + resid_oof
            per_inner_rae.append(float(rae(y_unb, bag[j])))
        nb1242_o = bag.mean(axis=0)
        nb1242_per_outer[oi] = nb1242_o
        rae_1242 = float(rae(y_unb, nb1242_o))
        per_outer_rae_nb1242.append(rae_1242)

        # nb1190_o from cache.
        nb1190_o = nb1190_per_outer[oi]
        rae_1190 = float(rae(y_unb, nb1190_o))
        per_outer_rae_nb1190.append(rae_1190)

        # nb1290_o blend (best-w from nb1290 grid).
        nb1290_o = W_NB1190 * nb1190_o + W_NB1242 * nb1242_o
        nb1290_per_outer[oi] = nb1290_o
        rae_1290 = float(rae(y_unb, nb1290_o))
        per_outer_rae_nb1290.append(rae_1290)

        repro_note = ""
        if o == 0:
            d = rae_1290 - NB1290_REF
            repro_note = (f"  [REPRO outer=0 vs nb1290 {NB1290_REF:.4f}: "
                          f"d={d:+.4f}, |d|<{REPRO_MARGIN}? "
                          f"{abs(d) < REPRO_MARGIN}]")

        per_outer_records.append({
            "outer_seed": int(o),
            "inner_seeds": inner_seeds,
            "per_inner_nb1242_rae": per_inner_rae,
            "rae_nb1190_o": rae_1190,
            "rae_nb1242_o": rae_1242,
            "rae_nb1290_o": rae_1290,
            "elapsed_sec": round(time.time() - t_o, 1),
        })
        print(f"   outer {o:5d}  inner={inner_seeds}")
        print(f"     nb1190_o={rae_1190:.4f}  nb1242_o={rae_1242:.4f}  "
              f"nb1290_o={rae_1290:.4f}  "
              f"(elapsed {time.time()-t_o:.1f}s){repro_note}")

    # ---- Sanity: outer=0 nb1242_o should reproduce cached nb1242_mean_bag_oof
    cached_1242_mean = np.load(
        DATA_PROCESSED / "nb1242_mean_bag_oof.npy"
    ).astype(np.float64)
    cached_match = np.allclose(nb1242_per_outer[0], cached_1242_mean, atol=1e-5)
    print(f"\n[sanity] outer=0 nb1242_o == cached nb1242_mean_bag_oof? "
          f"{cached_match}")

    # ---- Aggregate per-outer ----
    arr = np.array(per_outer_rae_nb1290)
    outer_mean = float(arr.mean())
    outer_std = float(arr.std())
    outer_min = float(arr.min())
    outer_max = float(arr.max())

    # ---- BoB mean / median across outer-seed nb1290_o vectors ----
    bob_mean_oof = nb1290_per_outer.mean(axis=0)
    bob_median_oof = np.median(nb1290_per_outer, axis=0)
    rae_bob_mean = float(rae(y_unb, bob_mean_oof))
    rae_bob_median = float(rae(y_unb, bob_median_oof))

    # ---- Verdict ----
    delta_outer_mean = outer_mean - NB1290_REF
    delta_outer0 = per_outer_rae_nb1290[0] - NB1290_REF
    reproduces = abs(delta_outer_mean) <= REPRO_MARGIN
    outer0_reproduces = abs(delta_outer0) <= REPRO_MARGIN

    if reproduces:
        verdict_repro = "NB1290_REPRODUCES"
    elif outer_mean < NB1290_REF - REPRO_MARGIN:
        verdict_repro = "NB1290_PESSIMISTIC_OUTER_BAG_BETTER"
    else:
        verdict_repro = "NB1290_OPTIMISTIC_OUTER_BAG_PULLS_UP"

    bob_mean_wins = rae_bob_mean < outer_mean - 1e-6
    bob_median_wins = rae_bob_median < outer_mean - 1e-6
    best_bob_rae = min(rae_bob_mean, rae_bob_median)
    best_bob_tag = "bob_mean" if rae_bob_mean <= rae_bob_median else "bob_median"

    if best_bob_rae < outer_mean - REPRO_MARGIN:
        verdict_var = "BOB_VARIANCE_REDUCTION_WIN"
    elif abs(best_bob_rae - outer_mean) < REPRO_MARGIN:
        verdict_var = "BOB_FLAT_VS_PER_OUTER_MEAN"
    else:
        verdict_var = "BOB_HURTS_VS_PER_OUTER_MEAN"

    if best_bob_rae < NB1290_REF - REPRO_MARGIN:
        verdict_vs_ref = "BOB_BEATS_NB1290_REF"
    elif abs(best_bob_rae - NB1290_REF) < REPRO_MARGIN:
        verdict_vs_ref = "BOB_FLAT_VS_NB1290_REF"
    else:
        verdict_vs_ref = "BOB_HURTS_VS_NB1290_REF"

    print("\n" + "=" * 78)
    print("AGGREGATION + VERDICT")
    print("=" * 78)
    print(f"   per-outer nb1190_o RAE list   = "
          f"[{', '.join(f'{r:.4f}' for r in per_outer_rae_nb1190)}]")
    print(f"   per-outer nb1242_o RAE list   = "
          f"[{', '.join(f'{r:.4f}' for r in per_outer_rae_nb1242)}]")
    print(f"   per-outer nb1290_o RAE list   = "
          f"[{', '.join(f'{r:.4f}' for r in per_outer_rae_nb1290)}]")
    print(f"   per-outer mean nb1290         = {outer_mean:.4f}")
    print(f"   per-outer std  nb1290         = {outer_std:.4f}")
    print(f"   per-outer min  nb1290         = {outer_min:.4f}")
    print(f"   per-outer max  nb1290         = {outer_max:.4f}")
    print(f"   BoB MEAN   row-level RAE      = {rae_bob_mean:.4f}")
    print(f"   BoB MEDIAN row-level RAE      = {rae_bob_median:.4f}")
    print(f"   nb1290 reference              = {NB1290_REF:.4f}")
    print(f"   delta(per-outer mean - ref)   = {delta_outer_mean:+.4f}  "
          f"(margin {REPRO_MARGIN:.3f})")
    print(f"   delta(outer0 - ref)           = {delta_outer0:+.4f}")
    print(f"   outer0 reproduces?            = {outer0_reproduces}")
    print(f"   per-outer mean reproduces?    = {reproduces}")
    print(f"   verdict (repro)               = {verdict_repro}")
    print(f"   verdict (variance reduction)  = {verdict_var}")
    print(f"   verdict (vs ref)              = {verdict_vs_ref}")
    print(f"   bob_mean_wins_vs_outer_mean   = {bob_mean_wins}")
    print(f"   bob_median_wins_vs_outer_mean = {bob_median_wins}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_per_outer_oof.npy",
            nb1290_per_outer.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_bob_mean_oof.npy",
            bob_mean_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_bob_median_oof.npy",
            bob_median_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_per_outer_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_bob_mean_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_bob_median_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "n_unb": n_unb,
        "outer_seeds": OUTER_SEEDS,
        "inner_bases": INNER_BASES,
        "resid_folds": RESID_FOLDS,
        "w_nb1190": W_NB1190,
        "w_nb1242": W_NB1242,
        "nb1290_ref": NB1290_REF,
        "repro_margin": REPRO_MARGIN,
        "n_chembl_pool": int(len(pool)),
        "chembl_pool_median_pec50": pool_median,
        "feat_dim_nb1242_lgbm": feat_dim,
        "outer0_nb1242_matches_cached_mean_bag": bool(cached_match),
        "rae_anchor_nb1070": rae_anchor,
        "per_outer_records": per_outer_records,
        "per_outer_rae_nb1190": [float(x) for x in per_outer_rae_nb1190],
        "per_outer_rae_nb1242": [float(x) for x in per_outer_rae_nb1242],
        "per_outer_rae_nb1290": [float(x) for x in per_outer_rae_nb1290],
        "outer_mean_nb1290": outer_mean,
        "outer_std_nb1290":  outer_std,
        "outer_min_nb1290":  outer_min,
        "outer_max_nb1290":  outer_max,
        "rae_bob_mean":      rae_bob_mean,
        "rae_bob_median":    rae_bob_median,
        "delta_outer_mean_vs_nb1290_ref": delta_outer_mean,
        "delta_outer0_vs_nb1290_ref":     delta_outer0,
        "outer0_reproduces": bool(outer0_reproduces),
        "reproduces": bool(reproduces),
        "bob_mean_wins_vs_outer_mean": bool(bob_mean_wins),
        "bob_median_wins_vs_outer_mean": bool(bob_median_wins),
        "best_bob_tag": best_bob_tag,
        "best_bob_rae": best_bob_rae,
        "delta_best_bob_vs_outer_mean": best_bob_rae - outer_mean,
        "delta_best_bob_vs_nb1290_ref": best_bob_rae - NB1290_REF,
        "verdict_repro":  verdict_repro,
        "verdict_var":    verdict_var,
        "verdict_vs_ref": verdict_vs_ref,
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
        "per_outer_rae_nb1190",
        "per_outer_rae_nb1242",
        "per_outer_rae_nb1290",
        "outer_mean_nb1290", "outer_std_nb1290",
        "outer_min_nb1290", "outer_max_nb1290",
        "rae_bob_mean", "rae_bob_median",
        "delta_outer_mean_vs_nb1290_ref", "delta_outer0_vs_nb1290_ref",
        "outer0_reproduces", "reproduces",
        "best_bob_tag", "best_bob_rae",
        "delta_best_bob_vs_outer_mean",
        "delta_best_bob_vs_nb1290_ref",
        "verdict_repro", "verdict_var", "verdict_vs_ref",
    ):
        print(f"  {k}: {res.get(k)}")
