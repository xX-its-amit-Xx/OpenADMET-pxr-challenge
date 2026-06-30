"""nb1794 -- LightGBM small leaves on 6-way K-tuned 167-col features.

HYPOTHESIS:
    nb1771 (LGBM huber, max_depth=4, num_leaves=15, n_est=300, lr=0.03,
    min_child=5, reg_lambda=2) on the 5-way K-tuned 117-col matrix hit
    rae_mean_bag 0.5100 on the 253 PRE-unblind set anchored on
    chemprop_aux (in_RAE 0.6216).  nb1780's outer-bag BoB MEAN validation
    over 5 outer x 5 inner cross-fits reported pooled bag mean 0.5032 --
    confirming nb1771 is on the optimistic side of its honest band.

    nb1620 took the same 5-way 117-col stack used by nb1771's CatBoost
    sibling (nb1554) and spliced in top-50 ChemBERTa-77M-MTR dims (nb1611
    SHAP ranking) for a 6-way 167-col matrix.  This script tests whether
    that 50-col ChemBERTa add transfers to the LGBM small-leaves regime
    (which beat the CatBoost shallow-tree sibling on the 5-way matrix).

PROTOCOL:
    1. Features = 167-col 6-way K-tuned matrix:
         top-25 AtomPair      (nb1524 best_K)
         top-20 MACCS         (nb1352 standard)
         top-20 Mordred       (nb1523 best_K)
         top-20 ChempropEmbed (nb1541 best_K)
         top-30 Avalon        (nb1392 SHAP K=30)
         top-50 ChemBERTa     (nb1611 top_dim_order_top100[:50])
         pred_chembl_pec50    (ChEMBL PXR kNN-5)
         mean_sim
    2. Anchor = chemprop_aux te[unb_idx]  (PRE-unblind, in_RAE 0.6216).
       residual = y_unb - anchor.
    3. 5-seed bag LightGBM(huber, max_depth=4, num_leaves=15, n_est=300,
       lr=0.03, min_child_samples=5, reg_lambda=2).
       KFold(n=5, shuffle=True, random_state=seed) cross-fit per seed.
    4. Pool mean_bag and median_bag corrected anchors.
    5. Verdict at 0.003 margin vs nb1771 (0.5100) and nb1780 BoB MEAN (0.5032).

Outputs:
    scripts/nb1794_lgbm_small_leaves_6way.py
    data/processed/nb1794_summary.json
    data/processed/nb1794_mean_bag_oof.npy        (253,) float32
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
import lightgbm as lgb
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1794"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"
CHEMBERTA_TE_PATH = DATA_PROCESSED / "te_chemberta.npy"  # (513, 384) -- nb1611 cache
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")

EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
NB1392_SUMMARY = DATA_PROCESSED / "nb1392_summary.json"
NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"
NB1523_SUMMARY = DATA_PROCESSED / "nb1523_summary.json"
NB1524_SUMMARY = DATA_PROCESSED / "nb1524_summary.json"
NB1541_SUMMARY = DATA_PROCESSED / "nb1541_summary.json"
NB1611_SUMMARY = DATA_PROCESSED / "nb1611_summary.json"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

# ChemBERTa K (fixed per protocol -- top 50 by nb1611 SHAP ranking)
K_CHEMBERTA = 50

# References (PRE-unblind path)
CHEMPROP_AUX_REF = 0.6216
NB1554_REF = 0.5163  # CatBoost(MAE, d4, ...) on 5-way 117-col
NB1620_REF = None    # CatBoost on 6-way 167-col (filled if summary present)
NB1771_REF = 0.5100  # LGBM small leaves on 5-way 117-col
NB1780_BOB_MEAN_REF = 0.5032  # outer-bag validation of nb1771
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
    """Identical union to nb1771 / nb1620."""
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
        max_depth=4,
        num_leaves=15,
        n_estimators=300,
        learning_rate=0.03,
        min_child_samples=5,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _residual_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray,
                                 seed: int) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
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


def _load_npy_test(path: Path, n_test_expected: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"missing cache: {path}")
    X = np.load(path)
    if X.shape[0] != n_test_expected:
        raise ValueError(f"shape mismatch {path}: {X.shape}")
    X = X.astype(np.float32)
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X


def _extract_embed_top_idx_from_nb1484(sum_1484: dict) -> np.ndarray:
    for f in sum_1484["families"]:
        if f["family"] == "ChempropEmbed":
            return np.array(f["top_idx_ranked"], dtype=int)
    raise KeyError("ChempropEmbed entry not found in nb1484_summary.json")


def _extract_atompair_top_idx_from_nb1484(sum_1484: dict) -> np.ndarray:
    for f in sum_1484["families"]:
        if f["family"] == "AtomPair":
            return np.array(f["top_idx_ranked"], dtype=int)
    raise KeyError("AtomPair entry not found in nb1484_summary.json")


def _extract_best_K_record(sum_dict: dict, records_key: str,
                           best_K_key: str = "best_K") -> dict:
    best_K = int(sum_dict[best_K_key])
    for r in sum_dict[records_key]:
        if int(r["K"]) == best_K:
            return r
    raise KeyError(f"best_K {best_K} not found in {records_key}")


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- LightGBM(huber, alpha=1.0, max_depth=4, num_leaves=15, "
          f"n_est=300, lr=0.03, min_child=5, reg_lambda=2) on 6-way K-tuned "
          f"167-col matrix; PRE-unblind anchor={ANCHOR}")
    print(f"          seeds = {RESID_SEEDS}  folds = {RESID_FOLDS}")
    print(f"          refs: chemprop_aux ({CHEMPROP_AUX_REF:.4f}), "
          f"nb1771 ({NB1771_REF:.4f}), "
          f"nb1780 BoB MEAN ({NB1780_BOB_MEAN_REF:.4f})  "
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

    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"chemprop_aux te file missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(
            f"chemprop_aux te shape mismatch: {te_anchor_513.shape} vs {n_test}"
        )
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Load all K-grid winners + SHAP rankings ----
    for p in (NB1352_SUMMARY, NB1392_SUMMARY, NB1484_SUMMARY,
              NB1523_SUMMARY, NB1524_SUMMARY, NB1541_SUMMARY,
              NB1611_SUMMARY):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")
    with open(NB1352_SUMMARY) as f:
        sum_1352 = json.load(f)
    with open(NB1392_SUMMARY) as f:
        sum_1392 = json.load(f)
    with open(NB1484_SUMMARY) as f:
        sum_1484 = json.load(f)
    with open(NB1523_SUMMARY) as f:
        sum_1523 = json.load(f)
    with open(NB1524_SUMMARY) as f:
        sum_1524 = json.load(f)
    with open(NB1541_SUMMARY) as f:
        sum_1541 = json.load(f)
    with open(NB1611_SUMMARY) as f:
        sum_1611 = json.load(f)

    # MACCS = standard top-20 from nb1352
    top_maccs_bit_idx = np.array(
        sum_1352["top_maccs_bit_indices_ranked"], dtype=int
    )

    # Mordred = nb1523 best_K (=20)
    rec_mord = _extract_best_K_record(sum_1523, "per_K_records",
                                       best_K_key="best_K")
    K_Mord_best = int(rec_mord["K"])
    top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)
    assert K_Mord_best == int(sum_1523["best_K"])

    # AtomPair = nb1524 best_K (=25), sliced from nb1484 SHAP ranking
    full_ap_ranked = _extract_atompair_top_idx_from_nb1484(sum_1484)
    K_AP_best = int(sum_1524["best_K"])
    top_ap_bit_idx = full_ap_ranked[:K_AP_best]

    # ChempropEmbed = nb1541 best_K (=20), sliced from nb1541 top_dim_order
    K_Embed_best = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed_col_idx = top_embed_full[:K_Embed_best]

    # Avalon = nb1392 top-30 SHAP ranking
    top_avalon_bit_idx = np.array(
        sum_1392["top_avalon_bit_indices_ranked"], dtype=int
    )
    K_Avalon_used = int(len(top_avalon_bit_idx))

    # ChemBERTa = nb1611 top-50 dims by SHAP ranking
    top_chemberta_full = np.array(sum_1611["top_dim_order_top100"], dtype=int)
    if K_CHEMBERTA > len(top_chemberta_full):
        raise ValueError(
            f"K_CHEMBERTA={K_CHEMBERTA} > len(top_dim_order_top100)="
            f"{len(top_chemberta_full)}"
        )
    top_chemberta_col_idx = top_chemberta_full[:K_CHEMBERTA]

    n_top_maccs = int(len(top_maccs_bit_idx))
    n_top_mord = int(len(top_mord_col_idx))
    n_top_ap = int(len(top_ap_bit_idx))
    n_top_embed = int(len(top_embed_col_idx))
    n_top_avalon = int(K_Avalon_used)
    n_top_chemberta = int(len(top_chemberta_col_idx))
    print(f"[reuse] top-{n_top_ap}     AtomPair  bits (nb1524 K={K_AP_best})")
    print(f"[reuse] top-{n_top_maccs}     MACCS     bits (nb1352)")
    print(f"[reuse] top-{n_top_mord}     Mordred   cols (nb1523 K={K_Mord_best})")
    print(f"[reuse] top-{n_top_embed}     ChempropEmbed (nb1541 K={K_Embed_best})")
    print(f"[reuse] top-{n_top_avalon}     Avalon    bits (nb1392 SHAP K=30)")
    print(f"[reuse] top-{n_top_chemberta}     ChemBERTa dims (nb1611 SHAP K={K_CHEMBERTA})")

    # ---- AtomPair ----
    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test)
    X_ap_unb = X_ap_te[unb_idx].astype(np.float32)
    X_ap_unb_top = X_ap_unb[:, top_ap_bit_idx].astype(np.float32)
    print(f"[feat] X_ap_unb_top        = {X_ap_unb_top.shape}")

    # ---- MACCS ----
    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)
    X_maccs_unb = X_maccs_te[unb_idx].astype(np.float32)
    X_maccs_unb_top = X_maccs_unb[:, top_maccs_bit_idx].astype(np.float32)
    print(f"[feat] X_maccs_unb_top     = {X_maccs_unb_top.shape}")

    # ---- Mordred ----
    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    X_mord_unb = X_mord_te[unb_idx].astype(np.float32)
    X_mord_unb_top = X_mord_unb[:, top_mord_col_idx].astype(np.float32)
    print(f"[feat] X_mord_unb_top      = {X_mord_unb_top.shape}")

    # ---- Chemprop embed ----
    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)
    X_emb_unb = X_emb_te[unb_idx].astype(np.float32)
    X_emb_unb_top = X_emb_unb[:, top_embed_col_idx].astype(np.float32)
    print(f"[feat] X_emb_unb_top       = {X_emb_unb_top.shape}")

    # ---- Avalon ----
    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)
    X_av_unb = X_av_te[unb_idx].astype(np.float32)
    X_av_unb_top = X_av_unb[:, top_avalon_bit_idx].astype(np.float32)
    print(f"[feat] X_av_unb_top        = {X_av_unb_top.shape}")

    # ---- ChemBERTa ----
    if not CHEMBERTA_TE_PATH.exists():
        raise FileNotFoundError(f"missing ChemBERTa cache: {CHEMBERTA_TE_PATH}")
    X_cb_te = np.load(CHEMBERTA_TE_PATH).astype(np.float32)
    if X_cb_te.shape[0] != n_test:
        raise ValueError(f"chemberta te shape mismatch: {X_cb_te.shape}")
    X_cb_te = np.where(np.isfinite(X_cb_te), X_cb_te, 0.0).astype(np.float32)
    X_cb_unb = X_cb_te[unb_idx].astype(np.float32)
    X_cb_unb_top = X_cb_unb[:, top_chemberta_col_idx].astype(np.float32)
    print(f"[feat] X_cb_unb_top        = {X_cb_unb_top.shape}")

    # ---- ChEMBL kNN feature ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL (union)")
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

    # ---- Build COMBINED 6-way K-tuned feature matrix ----
    X_unb = np.concatenate(
        [
            X_ap_unb_top,
            X_maccs_unb_top,
            X_mord_unb_top,
            X_emb_unb_top,
            X_av_unb_top,
            X_cb_unb_top,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim = X_unb.shape[1]
    expected_dim = (n_top_ap + n_top_maccs + n_top_mord
                    + n_top_embed + n_top_avalon + n_top_chemberta + 2)
    if feat_dim != expected_dim:
        raise ValueError(f"feat_dim {feat_dim} != expected {expected_dim}")
    print(f"\n   COMBINED 6-WAY K-TUNED matrix: {X_unb.shape}  "
          f"(top-{n_top_ap} AP + top-{n_top_maccs} MACCS + "
          f"top-{n_top_mord} Mord + top-{n_top_embed} Embed + "
          f"top-{n_top_avalon} Aval + top-{n_top_chemberta} CB + 2)")

    # ---- Per-seed LGBM small-leaf residual cross-fit ----
    print("\n" + "-" * 78)
    print(f"PER-SEED LGBM (small leaves, min_child=5) RESIDUAL CROSS-FIT "
          f"(dim={feat_dim})")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        resid_oof_s = _residual_cross_fit_one_seed(X_unb, residual, s)
        pred_corr_s = anchor + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        delta_s = rae_s - rae_anchor
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_chemprop_aux": delta_s,
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   seed {s:3d}:  rae_corr = {rae_s:.4f}  "
              f"(d_vs_anchor = {delta_s:+.4f})  "
              f"|resid_oof|.std = {resid_oof_s.std():.3f}  "
              f"wall = {time.time() - ts:.1f}s")

    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))

    per_seed_rae_arr = np.array(per_seed_rae)
    rae_per_seed_mean = float(per_seed_rae_arr.mean())
    rae_per_seed_median = float(np.median(per_seed_rae_arr))
    rae_per_seed_std = float(per_seed_rae_arr.std())
    rae_per_seed_min = float(per_seed_rae_arr.min())
    rae_per_seed_max = float(per_seed_rae_arr.max())

    # ---- Pearson vs prior PRE-unblind candidates ----
    def _pearson_vs(path: Path):
        if not path.exists():
            return None
        oof = np.load(path).astype(np.float64)
        if oof.shape[0] != n_unb:
            return None
        return float(np.corrcoef(mean_bag_oof, oof)[0, 1])

    pearson_vs_anchor = float(np.corrcoef(mean_bag_oof, anchor)[0, 1])
    pearson_vs_nb1771 = _pearson_vs(DATA_PROCESSED / "nb1771_mean_bag_oof.npy")
    pearson_vs_nb1780_bob_mean = _pearson_vs(
        DATA_PROCESSED / "nb1780_bob_mean_oof.npy"
    )
    pearson_vs_nb1554 = _pearson_vs(DATA_PROCESSED / "nb1554_mean_bag_oof.npy")
    pearson_vs_nb1620 = _pearson_vs(DATA_PROCESSED / "nb1620_mean_bag_oof.npy")
    pearson_vs_nb1611_best = _pearson_vs(
        DATA_PROCESSED / "nb1611_best_K_oof.npy"
    )

    print("\n" + "-" * 78)
    print("BAG AGGREGATIONS")
    print("-" * 78)
    print(f"   per-seed RAE list           = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   per-seed mean               = {rae_per_seed_mean:.4f}")
    print(f"   per-seed median             = {rae_per_seed_median:.4f}")
    print(f"   per-seed std                = {rae_per_seed_std:.4f}")
    print(f"   per-seed min/max            = "
          f"{rae_per_seed_min:.4f} / {rae_per_seed_max:.4f}")
    print(f"   pooled RAE(mean_bag)        = {rae_mean_bag:.4f}  "
          f"(d_vs_anchor = {rae_mean_bag - rae_anchor:+.4f}"
          f"  d_vs_nb1771 = {rae_mean_bag - NB1771_REF:+.4f}"
          f"  d_vs_nb1780_BoB = {rae_mean_bag - NB1780_BOB_MEAN_REF:+.4f})")
    print(f"   pooled RAE(median_bag)      = {rae_median_bag:.4f}")
    print(f"   Pearson(mean_bag, anchor)            = {pearson_vs_anchor:.4f}")
    if pearson_vs_nb1771 is not None:
        print(f"   Pearson(mean_bag, nb1771_mean_bag)   = "
              f"{pearson_vs_nb1771:.4f}")
    if pearson_vs_nb1780_bob_mean is not None:
        print(f"   Pearson(mean_bag, nb1780_bob_mean)   = "
              f"{pearson_vs_nb1780_bob_mean:.4f}")
    if pearson_vs_nb1554 is not None:
        print(f"   Pearson(mean_bag, nb1554_mean_bag)   = "
              f"{pearson_vs_nb1554:.4f}")
    if pearson_vs_nb1620 is not None:
        print(f"   Pearson(mean_bag, nb1620_mean_bag)   = "
              f"{pearson_vs_nb1620:.4f}")
    if pearson_vs_nb1611_best is not None:
        print(f"   Pearson(mean_bag, nb1611_best_K)     = "
              f"{pearson_vs_nb1611_best:.4f}")

    # ---- Verdict (per spec): 0.003 margin vs nb1771 (0.5100) and nb1780 BoB MEAN (0.5032) ----
    beats_anchor = rae_mean_bag < rae_anchor - DECISION_MARGIN
    beats_nb1771 = rae_mean_bag < NB1771_REF - DECISION_MARGIN
    beats_nb1780_bob = rae_mean_bag < NB1780_BOB_MEAN_REF - DECISION_MARGIN
    flat_vs_nb1771 = abs(rae_mean_bag - NB1771_REF) < DECISION_MARGIN
    flat_vs_nb1780_bob = abs(rae_mean_bag - NB1780_BOB_MEAN_REF) < DECISION_MARGIN

    if beats_nb1780_bob:
        verdict = ("LGBM_SMALL_LEAVES_6WAY_BEATS_NB1780_BOB_"
                   "NEW_PRE_UNBLIND_PRIMARY")
    elif flat_vs_nb1780_bob:
        verdict = "LGBM_SMALL_LEAVES_6WAY_FLAT_VS_NB1780_BOB"
    elif beats_nb1771:
        verdict = ("LGBM_SMALL_LEAVES_6WAY_BEATS_NB1771_BUT_"
                   "WORSE_THAN_NB1780_BOB")
    elif flat_vs_nb1771:
        verdict = "LGBM_SMALL_LEAVES_6WAY_FLAT_VS_NB1771"
    elif beats_anchor:
        verdict = ("LGBM_SMALL_LEAVES_6WAY_BEATS_ANCHOR_BUT_"
                   "WORSE_THAN_NB1771")
    elif abs(rae_mean_bag - rae_anchor) < DECISION_MARGIN:
        verdict = "LGBM_SMALL_LEAVES_6WAY_FLAT_VS_ANCHOR"
    else:
        verdict = "LGBM_SMALL_LEAVES_6WAY_HURTS_ANCHOR"

    pre_unblind_clean = True
    print(f"   verdict                     = {verdict}")
    print(f"   PRE-unblind clean           = {pre_unblind_clean}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy",
            mean_bag_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_mean_bag_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "data_source": ("AtomPair-cache + MACCS-cache + "
                        "Mordred-cached_nb1030 + ChempropEmbed-cache + "
                        "Avalon-cache + ChemBERTa-cache + "
                        "local_chembl_caches_union"),
        "model_family": "LightGBM",
        "lgbm_objective": "huber",
        "lgbm_alpha": 1.0,
        "lgbm_max_depth": 4,
        "lgbm_num_leaves": 15,
        "lgbm_n_estimators": 300,
        "lgbm_learning_rate": 0.03,
        "lgbm_min_child_samples": 5,
        "lgbm_reg_lambda": 2.0,
        "K_AP_best": K_AP_best,
        "K_Mord_best": K_Mord_best,
        "K_Embed_best": K_Embed_best,
        "K_Avalon_used": K_Avalon_used,
        "K_MACCS_fixed": n_top_maccs,
        "K_ChemBERTa_fixed": K_CHEMBERTA,
        "K_source": {
            "AtomPair": "nb1524 best_K",
            "MACCS": "nb1352 standard top-20",
            "Mordred": "nb1523 best_K",
            "ChempropEmbed": "nb1541 best_K",
            "Avalon": "nb1392 SHAP top-30 (K=30 default)",
            "ChemBERTa": ("nb1611 SHAP top-50 (broader than nb1611 best_K=20 "
                          "to let LGBM small-leaves prune)"),
        },
        "n_chembl_pool": int(len(pool)),
        "n_unb": n_unb,
        "n_top_atompair": n_top_ap,
        "n_top_maccs": n_top_maccs,
        "n_top_mordred": n_top_mord,
        "n_top_chemprop_embed": n_top_embed,
        "n_top_avalon": n_top_avalon,
        "n_top_chemberta": n_top_chemberta,
        "feat_dim": int(feat_dim),
        "feat_breakdown": {
            "atompair": n_top_ap,
            "maccs": n_top_maccs,
            "mordred": n_top_mord,
            "chemprop_embed": n_top_embed,
            "avalon": n_top_avalon,
            "chemberta": n_top_chemberta,
            "pred_chembl_pec50": 1,
            "mean_sim": 1,
            "total": int(feat_dim),
        },
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_seed_rae": per_seed_rae,
        "per_seed_records": per_seed_records,
        "rae_per_seed_mean": rae_per_seed_mean,
        "rae_per_seed_median": rae_per_seed_median,
        "rae_per_seed_std": rae_per_seed_std,
        "rae_per_seed_min": rae_per_seed_min,
        "rae_per_seed_max": rae_per_seed_max,
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_chemprop_aux": rae_mean_bag - rae_anchor,
        "delta_mean_bag_vs_nb1554": rae_mean_bag - NB1554_REF,
        "delta_mean_bag_vs_nb1771": rae_mean_bag - NB1771_REF,
        "delta_mean_bag_vs_nb1780_bob_mean": (
            rae_mean_bag - NB1780_BOB_MEAN_REF
        ),
        "beats_chemprop_aux": bool(beats_anchor),
        "beats_nb1771": bool(beats_nb1771),
        "beats_nb1780_bob_mean": bool(beats_nb1780_bob),
        "flat_vs_nb1771": bool(flat_vs_nb1771),
        "flat_vs_nb1780_bob_mean": bool(flat_vs_nb1780_bob),
        "pearson_vs_anchor": pearson_vs_anchor,
        "pearson_vs_nb1771_mean_bag": pearson_vs_nb1771,
        "pearson_vs_nb1780_bob_mean": pearson_vs_nb1780_bob_mean,
        "pearson_vs_nb1554_mean_bag": pearson_vs_nb1554,
        "pearson_vs_nb1620_mean_bag": pearson_vs_nb1620,
        "pearson_vs_nb1611_best_K": pearson_vs_nb1611_best,
        "verdict": verdict,
        "pre_unblind_clean": pre_unblind_clean,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "nb1554_ref": NB1554_REF,
        "nb1771_ref": NB1771_REF,
        "nb1780_bob_mean_ref": NB1780_BOB_MEAN_REF,
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
        "K_AP_best", "K_Mord_best", "K_Embed_best", "K_Avalon_used",
        "K_MACCS_fixed", "K_ChemBERTa_fixed",
        "n_chembl_pool", "feat_dim", "feat_breakdown",
        "rae_anchor_chemprop_aux", "per_seed_rae",
        "rae_per_seed_mean", "rae_per_seed_std",
        "rae_mean_bag", "rae_median_bag",
        "delta_mean_bag_vs_chemprop_aux",
        "delta_mean_bag_vs_nb1771",
        "delta_mean_bag_vs_nb1780_bob_mean",
        "beats_chemprop_aux",
        "beats_nb1771", "flat_vs_nb1771",
        "beats_nb1780_bob_mean", "flat_vs_nb1780_bob_mean",
        "pearson_vs_anchor",
        "pearson_vs_nb1771_mean_bag",
        "pearson_vs_nb1780_bob_mean",
        "pearson_vs_nb1620_mean_bag",
        "verdict", "pre_unblind_clean",
    ):
        print(f"  {k}: {res.get(k)}")
