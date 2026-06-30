"""nb2131 -- Pooled-25 at K=28 L=14 (5 outer x 5 inner).

HYPOTHESIS:
    nb2123 leaves-grid sweep at K=28 identified num_leaves=14 as a
    promising operating point (median validate 0.4681) vs the nb2103/
    nb2111 default L=15 (mean_bag 0.4737 / median_bag 0.4698).

    This notebook re-runs the nb2111 pooled-25 design (5 outer x 5 inner)
    with num_leaves=14 (all other LGBM hyperparams unchanged) to test
    whether the pooled-25 stabilisation effect carries over to L=14 and
    whether L=14 beats the L=15 pooled-25 reference.

PROTOCOL:
    1. Reuse top-28 SHAP indices from nb2063 / nb2103.
    2. Rebuild the same 117-col 5-way K-tuned feature matrix as
       nb2063/nb2081/nb2091/nb2103/nb2111, slice to top-28 SHAP cols.
    3. 5 outer x 5 inner = 25 fits of LGBM(MSE, max_depth=4,
       num_leaves=14, n_estimators=300, lr=0.03, min_child_samples=5,
       reg_lambda=2.0) with KFold(n=5, shuffle=True, random_state=inner_seed)
       on residuals y_unb - chemprop_aux.
       Inner seeds = outer*1000 + offset, offsets in {0, 1, 7, 42, 137}.
    4. Pool the 25 corrected OOF vectors via MEAN and MEDIAN; also
       compute BoB (outer-then-pool) aggregates.
    5. Report pooled-25 mean/median RAE; verdict vs nb2111 L=15 pooled-25
       references at decision_margin = 0.003.

Outputs:
    scripts/nb2131_pooled_25bag_K28L14.py
    data/processed/nb2131_summary.json
    data/processed/nb2131_pooled25_mean_oof.npy   (253,) float32
    data/processed/nb2131_pooled25_median_oof.npy (253,) float32
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

TAG = "nb2131"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

RESID_FOLDS = 5
OUTER_SEEDS = [0, 1, 7, 42, 137]
INNER_OFFSETS = [0, 1, 7, 42, 137]
K_SHAP = 28
NUM_LEAVES = 14  # nb2131: L=14 (vs nb2111 L=15)

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")

EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
NB1392_SUMMARY = DATA_PROCESSED / "nb1392_summary.json"
NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"
NB1523_SUMMARY = DATA_PROCESSED / "nb1523_summary.json"
NB1524_SUMMARY = DATA_PROCESSED / "nb1524_summary.json"
NB1541_SUMMARY = DATA_PROCESSED / "nb1541_summary.json"
NB2063_SHAP_IMP = DATA_PROCESSED / "nb2063_shap_importance_full117.npy"
NB2103_SUMMARY = DATA_PROCESSED / "nb2103_summary.json"
NB2111_SUMMARY = DATA_PROCESSED / "nb2111_summary.json"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

CHEMPROP_AUX_REF = 0.6216
NB2103_K28_MEAN_BAG_REF = 0.4737
NB2103_K28_MEDIAN_BAG_REF = 0.4698
NB2123_L14_MEDIAN_REF = 0.4681
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
    """Same union as nb1852/nb1861/nb2063/nb2081/nb2091/nb2103/nb2111."""
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
    """LGBM(MSE) -- L=14 variant of the nb2063/nb2103/nb2111 baseline."""
    return dict(
        objective="regression",
        max_depth=4,
        num_leaves=NUM_LEAVES,
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
    print(f"{TAG} -- Pooled-25 SHAP top-{K_SHAP} K=28 L={NUM_LEAVES}")
    print(f"          anchor={ANCHOR}  outer_seeds={OUTER_SEEDS}")
    print(f"          inner offsets per outer: {INNER_OFFSETS}  folds={RESID_FOLDS}")
    print(f"          ref: nb2111 K=28 L=15 mean_bag {NB2103_K28_MEAN_BAG_REF:.4f} / "
          f"median_bag {NB2103_K28_MEDIAN_BAG_REF:.4f}  "
          f"nb2123 L=14 median validate {NB2123_L14_MEDIAN_REF:.4f}  "
          f"margin = {DECISION_MARGIN}")
    print("=" * 78)

    # ---- nb2063 SHAP importance + nb2103 reference ----
    if not NB2063_SHAP_IMP.exists():
        raise FileNotFoundError(f"missing {NB2063_SHAP_IMP} -- run nb2063 first")
    shap_imp_full117 = np.load(NB2063_SHAP_IMP).astype(np.float32)
    print(f"[ref] nb2063 SHAP importance shape = {shap_imp_full117.shape}")
    full_rank_order = np.argsort(-shap_imp_full117).astype(np.int32)
    top28_idx = full_rank_order[:K_SHAP].astype(np.int32)

    nb2103_k28_mean_bag = NB2103_K28_MEAN_BAG_REF
    nb2103_k28_median_bag = NB2103_K28_MEDIAN_BAG_REF
    if NB2103_SUMMARY.exists():
        with open(NB2103_SUMMARY) as f:
            nb2103_sum = json.load(f)
        for r in nb2103_sum.get("per_K_records", []):
            if int(r.get("K", -1)) == K_SHAP:
                nb2103_k28_mean_bag = float(r["rae_mean_bag"])
                nb2103_k28_median_bag = float(r["rae_median_bag"])
                ref_top = np.array(r.get("top_K_idx_in_117", []), dtype=int)
                if ref_top.size == K_SHAP:
                    if not np.array_equal(ref_top, top28_idx):
                        print("   [warn] top-28 indices differ from nb2103 "
                              "cached -- will use full re-rank from "
                              "nb2063 SHAP importance")
                    else:
                        print("   [check] top-28 indices match nb2103 (OK)")
                break
    print(f"[ref] nb2103.K=28 L=15 mean_bag_rae   = {nb2103_k28_mean_bag:.4f}")
    print(f"[ref] nb2103.K=28 L=15 median_bag_rae = {nb2103_k28_median_bag:.4f}")

    # Pull nb2111 L=15 pooled-25 references if available
    nb2111_pooled25_mean_ref = None
    nb2111_pooled25_median_ref = None
    if NB2111_SUMMARY.exists():
        with open(NB2111_SUMMARY) as f:
            nb2111_sum = json.load(f)
        nb2111_pooled25_mean_ref = nb2111_sum.get("rae_pooled25_mean")
        nb2111_pooled25_median_ref = nb2111_sum.get("rae_pooled25_median")
        if nb2111_pooled25_mean_ref is not None:
            print(f"[ref] nb2111 (L=15) pooled25 mean   = "
                  f"{nb2111_pooled25_mean_ref:.4f}")
        if nb2111_pooled25_median_ref is not None:
            print(f"[ref] nb2111 (L=15) pooled25 median = "
                  f"{nb2111_pooled25_median_ref:.4f}")

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

    # ---- Load all K-grid winners (same as nb2063/nb2081/nb2091/nb2103/nb2111) ----
    for p in (NB1352_SUMMARY, NB1392_SUMMARY, NB1484_SUMMARY,
              NB1523_SUMMARY, NB1524_SUMMARY, NB1541_SUMMARY):
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

    top_maccs_bit_idx = np.array(
        sum_1352["top_maccs_bit_indices_ranked"], dtype=int
    )

    rec_mord = _extract_best_K_record(sum_1523, "per_K_records",
                                       best_K_key="best_K")
    K_Mord_best = int(rec_mord["K"])
    top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)
    assert K_Mord_best == int(sum_1523["best_K"])

    full_ap_ranked = _extract_atompair_top_idx_from_nb1484(sum_1484)
    K_AP_best = int(sum_1524["best_K"])
    top_ap_bit_idx = full_ap_ranked[:K_AP_best]

    K_Embed_best = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed_col_idx = top_embed_full[:K_Embed_best]

    top_avalon_bit_idx = np.array(
        sum_1392["top_avalon_bit_indices_ranked"], dtype=int
    )
    K_Avalon_used = int(len(top_avalon_bit_idx))

    n_top_maccs = int(len(top_maccs_bit_idx))
    n_top_mord = int(len(top_mord_col_idx))
    n_top_ap = int(len(top_ap_bit_idx))
    n_top_embed = int(len(top_embed_col_idx))
    n_top_avalon = int(K_Avalon_used)
    print(f"[reuse] top-{n_top_ap}     AtomPair bits (nb1524 K={K_AP_best})")
    print(f"[reuse] top-{n_top_maccs}     MACCS bits  (nb1352)")
    print(f"[reuse] top-{n_top_mord}     Mordred cols (nb1523 K={K_Mord_best})")
    print(f"[reuse] top-{n_top_embed}     ChempropEmbed dims (nb1541 K={K_Embed_best})")
    print(f"[reuse] top-{n_top_avalon}     Avalon bits (nb1392 SHAP K=30)")

    # ---- Feature matrices ----
    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test)
    X_ap_unb = X_ap_te[unb_idx].astype(np.float32)
    X_ap_unb_top = X_ap_unb[:, top_ap_bit_idx].astype(np.float32)

    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)
    X_maccs_unb = X_maccs_te[unb_idx].astype(np.float32)
    X_maccs_unb_top = X_maccs_unb[:, top_maccs_bit_idx].astype(np.float32)

    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    X_mord_unb = X_mord_te[unb_idx].astype(np.float32)
    X_mord_unb_top = X_mord_unb[:, top_mord_col_idx].astype(np.float32)

    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)
    X_emb_unb = X_emb_te[unb_idx].astype(np.float32)
    X_emb_unb_top = X_emb_unb[:, top_embed_col_idx].astype(np.float32)

    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)
    X_av_unb = X_av_te[unb_idx].astype(np.float32)
    X_av_unb_top = X_av_unb[:, top_avalon_bit_idx].astype(np.float32)

    # ---- ChEMBL kNN feature (same as nb2103/nb2111) ----
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

    # ---- Build COMBINED 5-way K-tuned 117-col feature matrix ----
    X_unb = np.concatenate(
        [
            X_ap_unb_top,
            X_maccs_unb_top,
            X_mord_unb_top,
            X_emb_unb_top,
            X_av_unb_top,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim = X_unb.shape[1]
    expected_dim = (n_top_ap + n_top_maccs + n_top_mord
                    + n_top_embed + n_top_avalon + 2)
    if feat_dim != expected_dim:
        raise ValueError(f"feat_dim {feat_dim} != expected {expected_dim}")
    print(f"\n   COMBINED 5-WAY K-TUNED matrix: {X_unb.shape}")

    if feat_dim != shap_imp_full117.shape[0]:
        raise ValueError(
            f"feat_dim {feat_dim} != nb2063 SHAP importance length "
            f"{shap_imp_full117.shape[0]}"
        )

    # ---- Feature names (same as nb2103/nb2111) ----
    feat_names: list[str] = []
    feat_family: list[str] = []
    for j, b in enumerate(top_ap_bit_idx):
        feat_names.append(f"AtomPair_bit_{int(b)}")
        feat_family.append("AtomPair")
    for j, b in enumerate(top_maccs_bit_idx):
        feat_names.append(f"MACCS_bit_{int(b)}")
        feat_family.append("MACCS")
    for j, c in enumerate(top_mord_col_idx):
        feat_names.append(f"Mordred_col_{int(c)}")
        feat_family.append("Mordred")
    for j, d in enumerate(top_embed_col_idx):
        feat_names.append(f"ChempropEmbed_dim_{int(d)}")
        feat_family.append("ChempropEmbed")
    for j, b in enumerate(top_avalon_bit_idx):
        feat_names.append(f"Avalon_bit_{int(b)}")
        feat_family.append("Avalon")
    feat_names.append("pred_chembl_pec50")
    feat_family.append("ChEMBL_kNN")
    feat_names.append("mean_sim")
    feat_family.append("ChEMBL_kNN")
    assert len(feat_names) == feat_dim

    # ---- Slice to top-28 ----
    topK_names = [feat_names[i] for i in top28_idx]
    topK_family = [feat_family[i] for i in top28_idx]
    fam_counts: dict[str, int] = {}
    for fam in topK_family:
        fam_counts[fam] = fam_counts.get(fam, 0) + 1
    print(f"\n   top-{K_SHAP} family breakdown: {fam_counts}")

    X_topK = X_unb[:, top28_idx].astype(np.float32)
    print(f"   X_topK shape: {X_topK.shape}")

    # ---- Pooled-25 cross-fit: 5 outer x 5 inner @ L=14 ----
    print("\n" + "-" * 78)
    print(f"POOLED-25 @ L={NUM_LEAVES}: outer={OUTER_SEEDS}  "
          f"inner_offsets={INNER_OFFSETS}")
    print("-" * 78)

    n_outer = len(OUTER_SEEDS)
    n_inner = len(INNER_OFFSETS)
    n_total = n_outer * n_inner
    per_inner_corrected = np.zeros((n_total, n_unb), dtype=np.float64)
    per_inner_rae: list[float] = []
    per_inner_records: list[dict] = []

    outer_corrected_mean = np.zeros((n_outer, n_unb), dtype=np.float64)
    outer_corrected_median = np.zeros((n_outer, n_unb), dtype=np.float64)
    outer_rae_mean: list[float] = []
    outer_rae_median: list[float] = []
    outer_per_inner_seeds: list[list[int]] = []
    outer_per_inner_rae: list[list[float]] = []

    fit_i = 0
    for oi, o in enumerate(OUTER_SEEDS):
        inner_seeds = [int(o) * 1000 + int(s) for s in INNER_OFFSETS]
        outer_per_inner_seeds.append(inner_seeds)
        inner_corrected_o = np.zeros((n_inner, n_unb), dtype=np.float64)
        inner_rae_o: list[float] = []
        for ii, inner_s in enumerate(inner_seeds):
            ts = time.time()
            resid_oof_s = _residual_cross_fit_one_seed(
                X_topK, residual, inner_s
            )
            pred_corr_s = anchor + resid_oof_s
            inner_corrected_o[ii] = pred_corr_s
            per_inner_corrected[fit_i] = pred_corr_s
            rae_s = float(rae(y_unb, pred_corr_s))
            inner_rae_o.append(rae_s)
            per_inner_rae.append(rae_s)
            delta_s = rae_s - rae_anchor
            per_inner_records.append({
                "outer_seed": int(o),
                "inner_seed": int(inner_s),
                "rae_corrected": rae_s,
                "delta_vs_chemprop_aux": delta_s,
                "resid_oof_std": float(resid_oof_s.std()),
                "resid_oof_mean": float(resid_oof_s.mean()),
                "wall_sec": round(time.time() - ts, 2),
            })
            print(f"   outer={o:3d} inner={inner_s:6d}:  "
                  f"rae_corr = {rae_s:.4f}  (d_vs_anchor = {delta_s:+.4f})  "
                  f"wall = {time.time() - ts:.1f}s")
            fit_i += 1

        outer_per_inner_rae.append(inner_rae_o)
        o_mean_oof = inner_corrected_o.mean(axis=0)
        o_median_oof = np.median(inner_corrected_o, axis=0)
        outer_corrected_mean[oi] = o_mean_oof
        outer_corrected_median[oi] = o_median_oof
        rae_o_mean = float(rae(y_unb, o_mean_oof))
        rae_o_median = float(rae(y_unb, o_median_oof))
        outer_rae_mean.append(rae_o_mean)
        outer_rae_median.append(rae_o_median)
        print(f"   outer={o:3d} inner-mean RAE   = {rae_o_mean:.4f}")
        print(f"   outer={o:3d} inner-median RAE = {rae_o_median:.4f}")

    # ---- Pooled-25 RAE ----
    pooled25_mean_oof = per_inner_corrected.mean(axis=0)
    pooled25_median_oof = np.median(per_inner_corrected, axis=0)
    rae_pooled25_mean = float(rae(y_unb, pooled25_mean_oof))
    rae_pooled25_median = float(rae(y_unb, pooled25_median_oof))

    # ---- BoB aggregates ----
    bob_mean_of_mean_oof = outer_corrected_mean.mean(axis=0)
    bob_median_of_mean_oof = np.median(outer_corrected_mean, axis=0)
    bob_mean_of_median_oof = outer_corrected_median.mean(axis=0)
    bob_median_of_median_oof = np.median(outer_corrected_median, axis=0)
    rae_bob_mean_of_mean = float(rae(y_unb, bob_mean_of_mean_oof))
    rae_bob_median_of_mean = float(rae(y_unb, bob_median_of_mean_oof))
    rae_bob_mean_of_median = float(rae(y_unb, bob_mean_of_median_oof))
    rae_bob_median_of_median = float(rae(y_unb, bob_median_of_median_oof))

    per_inner_rae_arr = np.array(per_inner_rae)
    rae_per_inner_mean = float(per_inner_rae_arr.mean())
    rae_per_inner_median = float(np.median(per_inner_rae_arr))
    rae_per_inner_std = float(per_inner_rae_arr.std())
    rae_per_inner_min = float(per_inner_rae_arr.min())
    rae_per_inner_max = float(per_inner_rae_arr.max())

    outer_rae_mean_arr = np.array(outer_rae_mean)
    outer_rae_median_arr = np.array(outer_rae_median)

    # ---- Verdict ----
    delta_pooled25_mean_vs_nb2103 = (
        rae_pooled25_mean - nb2103_k28_mean_bag
    )
    delta_pooled25_median_vs_nb2103 = (
        rae_pooled25_median - nb2103_k28_median_bag
    )
    delta_bob_mean_of_mean_vs_nb2103 = (
        rae_bob_mean_of_mean - nb2103_k28_mean_bag
    )
    delta_pooled25_mean_vs_nb2111 = (
        (rae_pooled25_mean - nb2111_pooled25_mean_ref)
        if nb2111_pooled25_mean_ref is not None else None
    )
    delta_pooled25_median_vs_nb2111 = (
        (rae_pooled25_median - nb2111_pooled25_median_ref)
        if nb2111_pooled25_median_ref is not None else None
    )

    def _verdict(rae_x: float, ref: float) -> str:
        if rae_x < ref - DECISION_MARGIN:
            return "BEATS"
        if abs(rae_x - ref) < DECISION_MARGIN:
            return "FLAT"
        return "WORSE"

    v_pooled25_mean = _verdict(rae_pooled25_mean, nb2103_k28_mean_bag)
    v_pooled25_median = _verdict(rae_pooled25_median, nb2103_k28_median_bag)
    v_bob_mean_of_mean = _verdict(
        rae_bob_mean_of_mean, nb2103_k28_mean_bag
    )
    v_bob_median_of_mean = _verdict(
        rae_bob_median_of_mean, nb2103_k28_mean_bag
    )

    print("\n" + "=" * 78)
    print(f"POOLED-25 @ L={NUM_LEAVES} SUMMARY")
    print("=" * 78)
    print(f"   per-inner (25 fits): "
          f"mean={rae_per_inner_mean:.4f}  "
          f"median={rae_per_inner_median:.4f}  "
          f"std={rae_per_inner_std:.4f}  "
          f"min={rae_per_inner_min:.4f}  max={rae_per_inner_max:.4f}")
    print(f"\n   POOLED-25 mean   RAE = {rae_pooled25_mean:.4f}  "
          f"(d_vs_nb2103_K28_mean   = {delta_pooled25_mean_vs_nb2103:+.4f})  "
          f"{v_pooled25_mean}")
    print(f"   POOLED-25 median RAE = {rae_pooled25_median:.4f}  "
          f"(d_vs_nb2103_K28_median = {delta_pooled25_median_vs_nb2103:+.4f})  "
          f"{v_pooled25_median}")
    if delta_pooled25_mean_vs_nb2111 is not None:
        print(f"   POOLED-25 mean   vs nb2111 L=15 mean   = "
              f"{delta_pooled25_mean_vs_nb2111:+.4f}")
    if delta_pooled25_median_vs_nb2111 is not None:
        print(f"   POOLED-25 median vs nb2111 L=15 median = "
              f"{delta_pooled25_median_vs_nb2111:+.4f}")

    print("\n   per-outer inner-mean RAE:")
    for oi, o in enumerate(OUTER_SEEDS):
        print(f"     outer={o:3d}: {outer_rae_mean[oi]:.4f}  "
              f"(inner-median {outer_rae_median[oi]:.4f})")

    print(f"\n   BoB MEAN-of-MEAN     = {rae_bob_mean_of_mean:.4f}  "
          f"(d_vs_nb2103_K28_mean = {delta_bob_mean_of_mean_vs_nb2103:+.4f})  "
          f"{v_bob_mean_of_mean}")
    print(f"   BoB MEDIAN-of-MEAN   = {rae_bob_median_of_mean:.4f}  "
          f"{v_bob_median_of_mean}")
    print(f"   BoB MEAN-of-MEDIAN   = {rae_bob_mean_of_median:.4f}")
    print(f"   BoB MEDIAN-of-MEDIAN = {rae_bob_median_of_median:.4f}")
    print(f"\n   anchor (chemprop_aux te[unb_idx]) RAE = {rae_anchor:.4f}")
    print(f"   nb2103 K=28 L=15 mean_bag ref         = "
          f"{nb2103_k28_mean_bag:.4f}")
    print(f"   nb2103 K=28 L=15 median_bag ref       = "
          f"{nb2103_k28_median_bag:.4f}")
    print(f"   nb2123 K=28 L=14 median validate ref  = "
          f"{NB2123_L14_MEDIAN_REF:.4f}")

    # ---- Save pooled OOFs ----
    out_mean = DATA_PROCESSED / f"{TAG}_pooled25_mean_oof.npy"
    out_median = DATA_PROCESSED / f"{TAG}_pooled25_median_oof.npy"
    np.save(out_mean, pooled25_mean_oof.astype(np.float32))
    np.save(out_median, pooled25_median_oof.astype(np.float32))
    print(f"\n[save] {out_mean}")
    print(f"[save] {out_median}")

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "method": ("lgbm_mse_pooled_25bag_shap_top28_on_117col"
                   f"_outer5_inner5_L{NUM_LEAVES}"),
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "data_source": ("nb2063 cached SHAP importance + same 117-col "
                        "5-way K-tuned matrix as nb2063/nb2081/nb2091/nb2103/"
                        "nb2111 (AtomPair / MACCS / Mordred / ChempropEmbed / "
                        "Avalon + ChEMBL kNN); sliced to nb2103 K=28 "
                        "top-SHAP indices; L=14 variant"),
        "model_family": "LightGBM",
        "lgbm_objective": "regression",
        "lgbm_max_depth": 4,
        "lgbm_num_leaves": NUM_LEAVES,
        "lgbm_n_estimators": 300,
        "lgbm_learning_rate": 0.03,
        "lgbm_min_child_samples": 5,
        "lgbm_reg_lambda": 2.0,
        "K_shap": int(K_SHAP),
        "outer_seeds": OUTER_SEEDS,
        "inner_offsets": INNER_OFFSETS,
        "inner_seed_rule": "inner = outer*1000 + offset",
        "n_total_fits": int(n_total),
        "resid_folds": RESID_FOLDS,
        "feat_dim_full": int(feat_dim),
        "feat_breakdown_full": {
            "atompair": n_top_ap,
            "maccs": n_top_maccs,
            "mordred": n_top_mord,
            "chemprop_embed": n_top_embed,
            "avalon": n_top_avalon,
            "pred_chembl_pec50": 1,
            "mean_sim": 1,
            "total": int(feat_dim),
        },
        "top_K_idx_in_117": top28_idx.tolist(),
        "top_K_feat_names": topK_names,
        "top_K_family_counts": fam_counts,
        "n_chembl_pool": int(len(pool)),
        "n_unb": n_unb,
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "nb2103_K28_mean_bag_ref": nb2103_k28_mean_bag,
        "nb2103_K28_median_bag_ref": nb2103_k28_median_bag,
        "nb2111_pooled25_mean_ref": nb2111_pooled25_mean_ref,
        "nb2111_pooled25_median_ref": nb2111_pooled25_median_ref,
        "nb2123_L14_median_ref": NB2123_L14_MEDIAN_REF,
        "per_inner_records": per_inner_records,
        "per_inner_rae": per_inner_rae,
        "rae_per_inner_mean": rae_per_inner_mean,
        "rae_per_inner_median": rae_per_inner_median,
        "rae_per_inner_std": rae_per_inner_std,
        "rae_per_inner_min": rae_per_inner_min,
        "rae_per_inner_max": rae_per_inner_max,
        "outer_inner_seeds": outer_per_inner_seeds,
        "outer_per_inner_rae": outer_per_inner_rae,
        "outer_rae_inner_mean": outer_rae_mean,
        "outer_rae_inner_median": outer_rae_median,
        "outer_rae_inner_mean_mean": float(outer_rae_mean_arr.mean()),
        "outer_rae_inner_mean_std": float(outer_rae_mean_arr.std()),
        "outer_rae_inner_median_mean": float(outer_rae_median_arr.mean()),
        "outer_rae_inner_median_std": float(outer_rae_median_arr.std()),
        "rae_pooled25_mean": rae_pooled25_mean,
        "rae_pooled25_median": rae_pooled25_median,
        "rae_bob_mean_of_mean": rae_bob_mean_of_mean,
        "rae_bob_median_of_mean": rae_bob_median_of_mean,
        "rae_bob_mean_of_median": rae_bob_mean_of_median,
        "rae_bob_median_of_median": rae_bob_median_of_median,
        "delta_pooled25_mean_vs_nb2103_K28_mean": (
            delta_pooled25_mean_vs_nb2103
        ),
        "delta_pooled25_median_vs_nb2103_K28_median": (
            delta_pooled25_median_vs_nb2103
        ),
        "delta_bob_mean_of_mean_vs_nb2103_K28_mean": (
            delta_bob_mean_of_mean_vs_nb2103
        ),
        "delta_pooled25_mean_vs_nb2111": delta_pooled25_mean_vs_nb2111,
        "delta_pooled25_median_vs_nb2111": delta_pooled25_median_vs_nb2111,
        "verdict_pooled25_mean": v_pooled25_mean,
        "verdict_pooled25_median": v_pooled25_median,
        "verdict_bob_mean_of_mean": v_bob_mean_of_mean,
        "verdict_bob_median_of_mean": v_bob_median_of_mean,
        "pre_unblind_clean": True,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "nb2103_K28_mean_bag_REF_const": NB2103_K28_MEAN_BAG_REF,
        "nb2103_K28_median_bag_REF_const": NB2103_K28_MEDIAN_BAG_REF,
        "nb2123_L14_median_REF_const": NB2123_L14_MEDIAN_REF,
        "decision_margin": DECISION_MARGIN,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "K_shap",
        "lgbm_num_leaves",
        "n_total_fits",
        "feat_dim_full",
        "n_chembl_pool",
        "rae_anchor_chemprop_aux",
        "nb2103_K28_mean_bag_ref",
        "nb2103_K28_median_bag_ref",
        "nb2111_pooled25_mean_ref",
        "nb2111_pooled25_median_ref",
        "rae_per_inner_mean",
        "rae_per_inner_median",
        "rae_per_inner_std",
        "rae_pooled25_mean",
        "rae_pooled25_median",
        "rae_bob_mean_of_mean",
        "rae_bob_median_of_mean",
        "rae_bob_mean_of_median",
        "rae_bob_median_of_median",
        "delta_pooled25_mean_vs_nb2103_K28_mean",
        "delta_pooled25_median_vs_nb2103_K28_median",
        "delta_pooled25_mean_vs_nb2111",
        "delta_pooled25_median_vs_nb2111",
        "verdict_pooled25_mean",
        "verdict_pooled25_median",
        "verdict_bob_mean_of_mean",
        "pre_unblind_clean",
    ):
        print(f"  {k}: {res.get(k)}")
