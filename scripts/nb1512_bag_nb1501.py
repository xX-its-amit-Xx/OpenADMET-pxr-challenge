"""nb1512 -- Outer-bag VALIDATION of nb1501 (CatBoost PRE-unblind 4-way).

Independent confirmation pass of nb1501. The CatBoost(MAE, d4, n200, lr0.05,
l2=5) residual learner of nb1501 on the 4-way pruned 112-col feature matrix
(top-30 AtomPair + top-20 MACCS + top-30 Mordred + top-30 Chemprop-embed +
pred_chembl + sim, chemprop_aux PRE-unblind anchor) is rebuilt fresh per
outer seed, with inner-bag aggregation per outer and row-level MEAN / MEDIAN
across the 5 outer-seed nb1501_o vectors.

PROTOCOL
    1. For each outer seed o in {0, 1, 7, 42, 137}, REBUILD a 5-inner-seed
       CatBoost bag on the 112-col 4-way pruned feature matrix anchored to
       chemprop_aux te[unb_idx].  Inner seeds reparameterised per outer:
           inner_seeds(o) = [o*1000 + s for s in {0, 1, 7, 42, 137}].
       Family top-K (30/20/30/30 for AtomPair/MACCS/Mordred/ChempropEmbed)
       is pinned from nb1484's published top-idx (deterministic SHAP at
       seed=0), exactly matching nb1501's input pruning.
    2. Per outer o, inner-bag-corrected OOF:
           pred_o(seed s_inner) = anchor + resid_cross_fit_one_seed(X, s_inner)
           nb1501_o = mean over 5 inner seeds of pred_o(s_inner)
       Per-outer pooled RAE = rae(y_unb, nb1501_o).
    3. Aggregate across the 5 outer seeds:
           bob_mean_oof   = row-mean   of {nb1501_o}_5
           bob_median_oof = row-median of {nb1501_o}_5
       Pool RAE for each.
    4. Verdict NB1501_REPRODUCES iff
           |per_outer_mean - 0.5223| < 0.003
       (where 0.5223 = nb1501 CatBoost bag mean RAE,
       see data/processed/nb1501_summary.json::rae_mean_bag).

Outputs:
    scripts/nb1512_bag_nb1501.py             (this file)
    data/processed/nb1512_summary.json
    data/processed/nb1512_bob_mean_oof.npy   (253,) float32
    data/processed/nb1512_bob_median_oof.npy (253,) float32
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

TAG = "nb1512"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

NB1501_SUMMARY = DATA_PROCESSED / "nb1501_summary.json"
NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"
NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
NB1364_SUMMARY = DATA_PROCESSED / "nb1364_summary.json"
NB1373_SUMMARY = DATA_PROCESSED / "nb1373_summary.json"

NB1501_REF_MEAN_BAG = 0.5223          # nb1501 rae_mean_bag (target)
REPRODUCE_MARGIN = 0.003

OUTER_SEEDS = [0, 1, 7, 42, 137]
INNER_BASE_SEEDS = [0, 1, 7, 42, 137]
RESID_FOLDS = 5

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")

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
    """Same union as nb1501 / nb1484."""
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
    """Same CatBoost params as nb1501."""
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


def _extract_embed_top_idx_from_nb1484(sum_1484: dict) -> np.ndarray:
    for f in sum_1484["families"]:
        if f["family"] == "ChempropEmbed":
            return np.array(f["top_idx_ranked"], dtype=int)
    raise KeyError("ChempropEmbed entry not found in nb1484_summary.json")


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- OUTER-BAG VALIDATION of nb1501 "
          f"(PRE-unblind CatBoost 4-way 112-col on chemprop_aux anchor)")
    print(f"         outer seeds      = {OUTER_SEEDS}")
    print(f"         inner base seeds = {INNER_BASE_SEEDS}")
    print(f"         inner_seeds(o)   = [o*1000 + s for s in base]")
    print(f"         target ref       = {NB1501_REF_MEAN_BAG} "
          f"(margin = {REPRODUCE_MARGIN})")
    print(f"         model            = CatBoost(MAE, d4, n200, lr0.05, l2=5)")
    print("=" * 78)

    # ---- Load nb1501 reference (sanity check / record) ----
    nb1501_rae_mean_bag = NB1501_REF_MEAN_BAG
    nb1501_rae_median_bag = None
    nb1501_per_seed_rae = None
    if NB1501_SUMMARY.exists():
        with open(NB1501_SUMMARY) as f:
            sum_1501 = json.load(f)
        nb1501_rae_mean_bag = float(sum_1501.get("rae_mean_bag",
                                                 NB1501_REF_MEAN_BAG))
        nb1501_rae_median_bag = sum_1501.get("rae_median_bag")
        nb1501_per_seed_rae = sum_1501.get("per_seed_rae")
        print(f"\n[load] nb1501 rae_mean_bag   = {nb1501_rae_mean_bag:.6f}")
        if nb1501_rae_median_bag is not None:
            print(f"[load] nb1501 rae_median_bag = {float(nb1501_rae_median_bag):.6f}")
        if nb1501_per_seed_rae is not None:
            print(f"[load] nb1501 per_seed_rae   = "
                  f"[{', '.join(f'{r:.4f}' for r in nb1501_per_seed_rae)}]")
    else:
        print(f"\n[warn] nb1501_summary.json not found; using hardcoded ref "
              f"{NB1501_REF_MEAN_BAG}")

    # ---- Pull pinned SHAP top-idx per family (same as nb1501) ----
    for p in (NB1352_SUMMARY, NB1364_SUMMARY, NB1373_SUMMARY, NB1484_SUMMARY):
        if not p.exists():
            raise FileNotFoundError(f"missing {p} -- run prerequisite first")
    with open(NB1352_SUMMARY) as f:
        sum_1352 = json.load(f)
    with open(NB1364_SUMMARY) as f:
        sum_1364 = json.load(f)
    with open(NB1373_SUMMARY) as f:
        sum_1373 = json.load(f)
    with open(NB1484_SUMMARY) as f:
        sum_1484 = json.load(f)
    top_maccs_bit_idx = np.array(
        sum_1352["top_maccs_bit_indices_ranked"], dtype=int
    )
    top_mord_col_idx = np.array(
        sum_1364["top_mordred_col_indices_ranked"], dtype=int
    )
    top_ap_bit_idx = np.array(
        sum_1373["top_atompair_bit_indices_ranked"], dtype=int
    )
    top_embed_col_idx = _extract_embed_top_idx_from_nb1484(sum_1484)
    n_top_maccs = int(len(top_maccs_bit_idx))
    n_top_mord = int(len(top_mord_col_idx))
    n_top_ap = int(len(top_ap_bit_idx))
    n_top_embed = int(len(top_embed_col_idx))
    print(f"[pin]  top-{n_top_maccs} MACCS  bits  (from nb1352)")
    print(f"[pin]  top-{n_top_mord} Mordred cols (from nb1364)")
    print(f"[pin]  top-{n_top_ap} AtomPair bits (from nb1373)")
    print(f"[pin]  top-{n_top_embed} Embed   cols (from nb1484)")

    # ---- Load truth + anchor (PRE-unblind chemprop_aux) ----
    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"\n[load] n_test={n_test}  n_unb={n_unb}")

    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"chemprop_aux te file missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(f"anchor shape mismatch: {te_anchor_513.shape}")
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Per-family pruned slices on 253 ----
    X_ap_te = np.load(ATOMPAIR_TE_PATH)
    X_ap_unb = X_ap_te[unb_idx].astype(np.float32)
    X_ap_unb_top = X_ap_unb[:, top_ap_bit_idx].astype(np.float32)
    print(f"\n[feat] X_ap_unb_top    shape = {X_ap_unb_top.shape}")

    X_maccs_te = np.load(MACCS_TE_PATH)
    X_maccs_unb = X_maccs_te[unb_idx].astype(np.float32)
    X_maccs_unb_top = X_maccs_unb[:, top_maccs_bit_idx].astype(np.float32)
    print(f"[feat] X_maccs_unb_top shape = {X_maccs_unb_top.shape}")

    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    X_mord_unb = X_mord_te[unb_idx].astype(np.float32)
    X_mord_unb_top = X_mord_unb[:, top_mord_col_idx].astype(np.float32)
    print(f"[feat] X_mord_unb_top  shape = {X_mord_unb_top.shape}")

    if not CHEMPROP_EMBED_TE_PATH.exists():
        raise FileNotFoundError(
            f"Chemprop embed cache missing: {CHEMPROP_EMBED_TE_PATH}"
        )
    X_emb_te = np.load(CHEMPROP_EMBED_TE_PATH).astype(np.float32)
    X_emb_te = np.where(np.isfinite(X_emb_te), X_emb_te, 0.0).astype(np.float32)
    X_emb_unb = X_emb_te[unb_idx].astype(np.float32)
    X_emb_unb_top = X_emb_unb[:, top_embed_col_idx].astype(np.float32)
    print(f"[feat] X_emb_unb_top   shape = {X_emb_unb_top.shape}")

    # ---- ChEMBL pool + kNN feature build (same as nb1501) ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL + kNN feature build")
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

    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )
    pred_chembl_unb = pred_chembl_pec50[unb_idx].astype(np.float32)
    mean_sim_unb = mean_sim[unb_idx].astype(np.float32)
    print(f"   pred_chembl_pec50 (unb) mean={pred_chembl_unb.mean():.3f}  "
          f"std={pred_chembl_unb.std():.3f}")
    print(f"   mean_sim (unb)         mean={mean_sim_unb.mean():.3f}")

    # ---- Build QUAD-PRUNED 112-col feature matrix on 253 (same as nb1501) ----
    X_unb = np.concatenate(
        [
            X_ap_unb_top,
            X_maccs_unb_top,
            X_mord_unb_top,
            X_emb_unb_top,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim = X_unb.shape[1]
    expected_dim = n_top_ap + n_top_maccs + n_top_mord + n_top_embed + 2
    if feat_dim != expected_dim:
        raise ValueError(f"feat_dim {feat_dim} != expected {expected_dim}")
    print(f"\n   4-WAY PRUNED feature matrix: {X_unb.shape}  "
          f"(top-{n_top_ap} AP + top-{n_top_maccs} MACCS + "
          f"top-{n_top_mord} Mordred + top-{n_top_embed} Embed "
          f"+ pred_chembl + sim)")

    # ---- Outer x inner CatBoost cross-fit ----
    print("\n" + "=" * 78)
    print("OUTER x INNER CATBOOST RESIDUAL CROSS-FIT")
    print("=" * 78)
    n_outer = len(OUTER_SEEDS)
    n_inner = len(INNER_BASE_SEEDS)

    outer_blend = np.zeros((n_outer, n_unb), dtype=np.float64)
    per_outer_records = []
    per_outer_inner_seeds: list[list[int]] = []

    for oi, o in enumerate(OUTER_SEEDS):
        inner_seeds = [int(o * 1000 + s) for s in INNER_BASE_SEEDS]
        per_outer_inner_seeds.append(inner_seeds)
        t_outer = time.time()
        print(f"\n   --- outer seed {o}  inner seeds = {inner_seeds} ---")
        inner_corrected = np.zeros((n_inner, n_unb), dtype=np.float64)
        inner_per_seed_rae: list[float] = []
        for ii, isd in enumerate(inner_seeds):
            ts = time.time()
            resid_oof_s = _residual_cross_fit_one_seed(X_unb, residual, isd)
            pred_corr_s = anchor + resid_oof_s
            inner_corrected[ii] = pred_corr_s
            r_s = float(rae(y_unb, pred_corr_s))
            inner_per_seed_rae.append(r_s)
            print(f"     outer {o:3d}  inner {isd:6d}: rae = {r_s:.4f}  "
                  f"|resid|.std = {resid_oof_s.std():.3f}  "
                  f"wall = {time.time() - ts:.1f}s")
        # nb1501_o = mean over 5 inner seeds (== nb1501 pooling rule)
        blend_o = inner_corrected.mean(axis=0)
        outer_blend[oi] = blend_o
        rae_blend_o = float(rae(y_unb, blend_o))
        per_outer_records.append({
            "outer_seed": int(o),
            "inner_seeds": inner_seeds,
            "inner_per_seed_rae": inner_per_seed_rae,
            "rae_per_seed_mean": float(np.mean(inner_per_seed_rae)),
            "rae_nb1501_o_mean_bag": rae_blend_o,
            "wall_sec": round(time.time() - t_outer, 2),
        })
        print(f"   outer {o:3d}  nb1501_o (5-inner mean_bag) pooled RAE = "
              f"{rae_blend_o:.4f}  (outer wall = {time.time() - t_outer:.1f}s)")

    per_outer_rae_blend: list[float] = [
        rec["rae_nb1501_o_mean_bag"] for rec in per_outer_records
    ]
    per_outer_arr = np.array(per_outer_rae_blend)
    per_outer_mean = float(per_outer_arr.mean())
    per_outer_std = float(per_outer_arr.std())
    per_outer_min = float(per_outer_arr.min())
    per_outer_max = float(per_outer_arr.max())
    per_outer_median = float(np.median(per_outer_arr))

    # ---- BoB row-level aggregations across 5 nb1501_o vectors ----
    bob_mean_oof = outer_blend.mean(axis=0)
    bob_median_oof = np.median(outer_blend, axis=0)
    rae_bob_mean = float(rae(y_unb, bob_mean_oof))
    rae_bob_median = float(rae(y_unb, bob_median_oof))

    print("\n" + "=" * 78)
    print("OUTER-BAG SUMMARY")
    print("=" * 78)
    print(f"   per-outer nb1501_o RAE list = "
          f"[{', '.join(f'{r:.4f}' for r in per_outer_rae_blend)}]")
    print(f"   per-outer mean   = {per_outer_mean:.4f}")
    print(f"   per-outer std    = {per_outer_std:.4f}")
    print(f"   per-outer min    = {per_outer_min:.4f}")
    print(f"   per-outer max    = {per_outer_max:.4f}")
    print(f"   per-outer median = {per_outer_median:.4f}")
    print(f"   BoB MEAN   pooled RAE = {rae_bob_mean:.4f}  "
          f"(d vs nb1501_mean_bag = {rae_bob_mean - nb1501_rae_mean_bag:+.4f})")
    print(f"   BoB MEDIAN pooled RAE = {rae_bob_median:.4f}  "
          f"(d vs nb1501_mean_bag = {rae_bob_median - nb1501_rae_mean_bag:+.4f})")

    # ---- Pearson sanity vs nb1501 ----
    def _pearson_vs(path: Path, vec: np.ndarray):
        if not path.exists():
            return None
        oof = np.load(path).astype(np.float64)
        if oof.shape[0] != n_unb:
            return None
        return float(np.corrcoef(vec, oof)[0, 1])

    pearson_bobmean_vs_nb1501 = _pearson_vs(
        DATA_PROCESSED / "nb1501_mean_bag_oof.npy", bob_mean_oof
    )
    pearson_bobmedian_vs_nb1501 = _pearson_vs(
        DATA_PROCESSED / "nb1501_mean_bag_oof.npy", bob_median_oof
    )
    pearson_bobmean_vs_anchor = float(np.corrcoef(bob_mean_oof, anchor)[0, 1])
    if pearson_bobmean_vs_nb1501 is not None:
        print(f"   Pearson(bob_mean,   nb1501_mean_bag) = "
              f"{pearson_bobmean_vs_nb1501:.4f}")
    if pearson_bobmedian_vs_nb1501 is not None:
        print(f"   Pearson(bob_median, nb1501_mean_bag) = "
              f"{pearson_bobmedian_vs_nb1501:.4f}")
    print(f"   Pearson(bob_mean,   anchor)          = "
          f"{pearson_bobmean_vs_anchor:.4f}")

    # ---- Verdict (per-outer-mean vs 0.5223 within 0.003) ----
    delta_per_outer = per_outer_mean - nb1501_rae_mean_bag
    reproduces = abs(delta_per_outer) < REPRODUCE_MARGIN
    if reproduces:
        verdict = "NB1501_REPRODUCES"
    elif per_outer_mean < nb1501_rae_mean_bag - REPRODUCE_MARGIN:
        verdict = "NB1501_OPTIMISTIC_OUTER_BAG_BETTER"
    else:
        verdict = "NB1501_LUCKY_SEED_OUTER_BAG_WORSE"

    print("\n" + "-" * 78)
    print(f"VERDICT (per-outer-mean within {REPRODUCE_MARGIN} of nb1501 "
          f"rae_mean_bag = {nb1501_rae_mean_bag:.4f}):")
    print(f"   per-outer-mean = {per_outer_mean:.4f}   "
          f"(d vs ref = {delta_per_outer:+.4f})")
    print(f"   verdict        = {verdict}")

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
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "data_source": "AtomPair-cache + MACCS-cache + Mordred-cached_nb1030 + "
                       "ChempropEmbed-cache + local_chembl_caches_union",
        "model_family": "CatBoost",
        "catboost_loss": "MAE",
        "catboost_depth": 4,
        "catboost_iterations": 200,
        "catboost_learning_rate": 0.05,
        "catboost_l2_leaf_reg": 5.0,
        "n_unb": n_unb,
        "n_test": int(n_test),
        "n_chembl_pool": int(len(pool)),
        "outer_seeds": OUTER_SEEDS,
        "inner_base_seeds": INNER_BASE_SEEDS,
        "per_outer_inner_seeds": per_outer_inner_seeds,
        "resid_folds": RESID_FOLDS,
        "n_top_atompair": n_top_ap,
        "n_top_maccs": n_top_maccs,
        "n_top_mordred": n_top_mord,
        "n_top_chemprop_embed": n_top_embed,
        "feat_dim": int(feat_dim),
        "feat_breakdown": {
            "atompair": n_top_ap,
            "maccs": n_top_maccs,
            "mordred": n_top_mord,
            "chemprop_embed": n_top_embed,
            "pred_chembl_pec50": 1,
            "mean_sim": 1,
            "total": int(feat_dim),
        },
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "nb1501_rae_mean_bag": nb1501_rae_mean_bag,
        "nb1501_rae_median_bag": (None if nb1501_rae_median_bag is None
                                  else float(nb1501_rae_median_bag)),
        "nb1501_per_seed_rae": nb1501_per_seed_rae,
        "nb1501_ref_for_verdict": nb1501_rae_mean_bag,
        "nb1501_ref_kind": "rae_mean_bag",
        "reproduce_margin": REPRODUCE_MARGIN,
        "per_outer_records": per_outer_records,
        "per_outer_rae_nb1501": per_outer_rae_blend,
        "per_outer_mean": per_outer_mean,
        "per_outer_std": per_outer_std,
        "per_outer_min": per_outer_min,
        "per_outer_max": per_outer_max,
        "per_outer_median": per_outer_median,
        "rae_bob_mean": rae_bob_mean,
        "rae_bob_median": rae_bob_median,
        "delta_per_outer_mean_vs_nb1501": delta_per_outer,
        "delta_bob_mean_vs_nb1501": rae_bob_mean - nb1501_rae_mean_bag,
        "delta_bob_median_vs_nb1501": rae_bob_median - nb1501_rae_mean_bag,
        "reproduces": bool(reproduces),
        "pearson_bobmean_vs_nb1501_mean_bag": pearson_bobmean_vs_nb1501,
        "pearson_bobmedian_vs_nb1501_mean_bag": pearson_bobmedian_vs_nb1501,
        "pearson_bobmean_vs_anchor": pearson_bobmean_vs_anchor,
        "verdict": verdict,
        "pre_unblind_clean": True,
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
        "n_unb", "n_test", "n_chembl_pool", "feat_dim",
        "outer_seeds",
        "rae_anchor_chemprop_aux",
        "nb1501_rae_mean_bag",
        "nb1501_ref_for_verdict",
        "per_outer_rae_nb1501",
        "per_outer_mean", "per_outer_std",
        "per_outer_min", "per_outer_max", "per_outer_median",
        "rae_bob_mean", "rae_bob_median",
        "delta_per_outer_mean_vs_nb1501",
        "delta_bob_mean_vs_nb1501",
        "delta_bob_median_vs_nb1501",
        "reproduces",
        "pearson_bobmean_vs_nb1501_mean_bag",
        "pearson_bobmean_vs_anchor",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
