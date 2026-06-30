"""nb2082 -- 25-bag POOLED SHAP top-50 (vs 5-bag nb2063 / 5x5 BoB nb2071).

PROTOCOL:
    1. Reuse top-50 SHAP feature indices from nb2063_summary.json
       (field 'top50_idx_in_117').
    2. 5 OUTER x 5 INNER = 25 fits of LGBM(MSE) with chemprop_aux anchor.
       Inner seeds = [o*1000 + s for s in {0, 1, 7, 42, 137}] per outer o.
    3. Pool all 25 corrected OOFs (mean), compute single POOLED-25 RAE.
       (Contrast with nb2063 5-bag mean and nb2071 BoB MEAN of per-outer
       pooled means.)

Outputs:
    scripts/nb2082_pooled_25bag_shap50.py
    data/processed/nb2082_summary.json
    data/processed/nb2082_pooled25_oof.npy  (253,) float32
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

TAG = "nb2082"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

RESID_FOLDS = 5
OUTER_SEEDS = [0, 1, 7, 42, 137]
INNER_OFFSETS = [0, 1, 7, 42, 137]
N_INNER = len(INNER_OFFSETS)
TOP_K_SHAP = 50

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
NB2063_SUMMARY = DATA_PROCESSED / "nb2063_summary.json"
NB2071_SUMMARY = DATA_PROCESSED / "nb2071_summary.json"

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
        objective="regression",
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
    print(f"{TAG} -- 25-bag POOLED SHAP top-{TOP_K_SHAP} "
          f"(5 outer x 5 inner)")
    print(f"          outer seeds   = {OUTER_SEEDS}")
    print(f"          inner offsets = {INNER_OFFSETS}  (s_inner = o*1000 + s)")
    print(f"          folds         = {RESID_FOLDS}")
    print("=" * 78)

    # ---- Load nb2063 top-50 SHAP indices ----
    if not NB2063_SUMMARY.exists():
        raise FileNotFoundError(f"missing {NB2063_SUMMARY}")
    with open(NB2063_SUMMARY) as f:
        nb2063_sum = json.load(f)
    top50_idx = np.array(nb2063_sum["top50_idx_in_117"], dtype=np.int32)
    if top50_idx.shape[0] != TOP_K_SHAP:
        raise ValueError(
            f"nb2063 top50_idx_in_117 has {top50_idx.shape[0]} entries, "
            f"expected {TOP_K_SHAP}"
        )
    nb2063_mean_bag_actual = float(nb2063_sum.get("rae_mean_bag", float("nan")))
    print(f"[reuse] nb2063 top-{TOP_K_SHAP} indices: "
          f"{top50_idx[:10].tolist()}...")
    print(f"[reuse] nb2063 rae_mean_bag (5-bag) = "
          f"{nb2063_mean_bag_actual:.6f}")

    nb2071_bob_mean = None
    if NB2071_SUMMARY.exists():
        with open(NB2071_SUMMARY) as f:
            nb2071_sum = json.load(f)
        nb2071_bob_mean = float(nb2071_sum.get("bob_mean_rae", float("nan")))
        print(f"[reuse] nb2071 BoB MEAN (5x5) = {nb2071_bob_mean:.6f}")

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
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Load all K-grid winners ----
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

    # ---- Feature matrices (unb slice) ----
    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test)
    X_ap_unb_top = X_ap_te[unb_idx][:, top_ap_bit_idx].astype(np.float32)
    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)
    X_maccs_unb_top = X_maccs_te[unb_idx][:, top_maccs_bit_idx].astype(
        np.float32)
    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    X_mord_unb_top = X_mord_te[unb_idx][:, top_mord_col_idx].astype(np.float32)
    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)
    X_emb_unb_top = X_emb_te[unb_idx][:, top_embed_col_idx].astype(np.float32)
    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)
    X_av_unb_top = X_av_te[unb_idx][:, top_avalon_bit_idx].astype(np.float32)

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
    print(f"   pool: {n_before} -> {n_after}")

    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep_pool = fp_pool.sum(axis=1) > 0
    if not keep_pool.all():
        pool = pool[keep_pool].reset_index(drop=True)
        fp_pool = fp_pool[keep_pool]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))

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

    # ---- Build COMBINED 117-col matrix on unb ----
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
    print(f"\n   COMBINED 117-col matrix: {X_unb.shape}")

    # ---- Restrict to top-50 (reused from nb2063) ----
    if int(top50_idx.max()) >= feat_dim:
        raise ValueError(
            f"nb2063 top50_idx.max()={int(top50_idx.max())} >= feat_dim "
            f"{feat_dim}")
    X_top50 = X_unb[:, top50_idx].astype(np.float32)
    print(f"   X_top50 shape = {X_top50.shape}")

    # ---- 5 OUTER x 5 INNER = 25 corrected OOFs, then POOL ----
    print("\n" + "-" * 78)
    print(f"25-BAG POOLED:  {len(OUTER_SEEDS)} outer x {N_INNER} inner "
          f"x {RESID_FOLDS}-fold cross-fit")
    print("-" * 78)
    n_total = len(OUTER_SEEDS) * N_INNER
    all_corrected = np.zeros((n_total, n_unb), dtype=np.float64)
    per_fit_records = []
    per_outer_records = []
    per_outer_rae: list[float] = []
    k_global = 0
    for o_i, o in enumerate(OUTER_SEEDS):
        t_o = time.time()
        inner_seeds = [o * 1000 + s for s in INNER_OFFSETS]
        inner_rae_list: list[float] = []
        inner_corrected = np.zeros((N_INNER, n_unb), dtype=np.float64)
        for i_s, s in enumerate(inner_seeds):
            resid_oof_s = _residual_cross_fit_one_seed(X_top50, residual, s)
            pred_corr_s = anchor + resid_oof_s
            all_corrected[k_global] = pred_corr_s
            inner_corrected[i_s] = pred_corr_s
            rae_s = float(rae(y_unb, pred_corr_s))
            inner_rae_list.append(rae_s)
            per_fit_records.append({
                "k": int(k_global),
                "outer_seed": int(o),
                "inner_seed": int(s),
                "rae": rae_s,
            })
            k_global += 1
        mean_bag_o = inner_corrected.mean(axis=0)
        rae_o = float(rae(y_unb, mean_bag_o))
        per_outer_rae.append(rae_o)
        per_outer_records.append({
            "outer_seed": int(o),
            "inner_seeds": [int(s) for s in inner_seeds],
            "inner_rae": inner_rae_list,
            "inner_mean_rae": float(np.mean(inner_rae_list)),
            "inner_median_rae": float(np.median(inner_rae_list)),
            "pooled_mean_bag_rae": rae_o,
            "wall_sec": round(time.time() - t_o, 2),
        })
        print(f"   outer {o:3d}:  inner_seeds={inner_seeds}  "
              f"inner_RAE=[{', '.join(f'{r:.4f}' for r in inner_rae_list)}]  "
              f"per-outer pooled={rae_o:.4f}  "
              f"wall={time.time() - t_o:.1f}s")

    # ---- POOL all 25 ----
    pooled25_pred = all_corrected.mean(axis=0)
    pooled25_rae = float(rae(y_unb, pooled25_pred))
    pooled25_median_pred = np.median(all_corrected, axis=0)
    pooled25_median_rae = float(rae(y_unb, pooled25_median_pred))

    per_fit_rae_arr = np.array([r["rae"] for r in per_fit_records],
                                dtype=np.float64)
    per_fit_mean = float(per_fit_rae_arr.mean())
    per_fit_median = float(np.median(per_fit_rae_arr))
    per_fit_std = float(per_fit_rae_arr.std())
    per_fit_min = float(per_fit_rae_arr.min())
    per_fit_max = float(per_fit_rae_arr.max())

    per_outer_arr = np.array(per_outer_rae, dtype=np.float64)
    bob_mean_per_outer = float(per_outer_arr.mean())
    bob_median_per_outer = float(np.median(per_outer_arr))

    print("\n" + "-" * 78)
    print("AGGREGATIONS")
    print("-" * 78)
    print(f"   per-fit (25)   RAE mean   = {per_fit_mean:.4f}")
    print(f"   per-fit (25)   RAE median = {per_fit_median:.4f}")
    print(f"   per-fit (25)   RAE std    = {per_fit_std:.4f}")
    print(f"   per-fit (25)   RAE min/max= {per_fit_min:.4f} / "
          f"{per_fit_max:.4f}")
    print(f"   per-outer pooled RAE list = "
          f"[{', '.join(f'{r:.4f}' for r in per_outer_rae)}]")
    print(f"   BoB MEAN  per-outer       = {bob_mean_per_outer:.4f}")
    print(f"   BoB MED   per-outer       = {bob_median_per_outer:.4f}")
    print(f"   POOLED-25 RAE  (mean over 25) = {pooled25_rae:.4f}")
    print(f"   POOLED-25 RAE  (median over 25) = {pooled25_median_rae:.4f}")

    delta_vs_nb2063 = (pooled25_rae - nb2063_mean_bag_actual) \
        if not np.isnan(nb2063_mean_bag_actual) else None
    delta_vs_nb2071 = (pooled25_rae - nb2071_bob_mean) \
        if (nb2071_bob_mean is not None
            and not np.isnan(nb2071_bob_mean)) else None
    print(f"\n   POOLED-25 vs nb2063 mean_bag (5-bag): "
          f"d = {delta_vs_nb2063:+.4f}"
          if delta_vs_nb2063 is not None else
          "   POOLED-25 vs nb2063: nan ref")
    if delta_vs_nb2071 is not None:
        print(f"   POOLED-25 vs nb2071 BoB MEAN  (5x5):  "
              f"d = {delta_vs_nb2071:+.4f}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_pooled25_oof.npy",
            pooled25_pred.astype(np.float32))
    print(f"\n[save] "
          f"{DATA_PROCESSED / f'{TAG}_pooled25_oof.npy'}")

    summary = {
        "tag": TAG,
        "method": "pooled_25_bag_shap_top50_lgbm_mse",
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "model_family": "LightGBM",
        "lgbm_objective": "regression",
        "lgbm_max_depth": 4,
        "lgbm_num_leaves": 15,
        "lgbm_n_estimators": 300,
        "lgbm_learning_rate": 0.03,
        "lgbm_min_child_samples": 5,
        "lgbm_reg_lambda": 2.0,
        "top_K_shap": TOP_K_SHAP,
        "top50_idx_in_117_from_nb2063": top50_idx.tolist(),
        "feat_dim_full": int(feat_dim),
        "feat_dim_top50": int(X_top50.shape[1]),
        "n_chembl_pool": int(len(pool)),
        "n_unb": n_unb,
        "outer_seeds": OUTER_SEEDS,
        "inner_offsets": INNER_OFFSETS,
        "n_inner_per_outer": N_INNER,
        "n_total_fits": int(n_total),
        "resid_folds": RESID_FOLDS,
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_fit_records": per_fit_records,
        "per_outer_records": per_outer_records,
        "per_outer_rae": per_outer_rae,
        "per_fit_mean_rae": per_fit_mean,
        "per_fit_median_rae": per_fit_median,
        "per_fit_std_rae": per_fit_std,
        "per_fit_min_rae": per_fit_min,
        "per_fit_max_rae": per_fit_max,
        "bob_mean_per_outer": bob_mean_per_outer,
        "bob_median_per_outer": bob_median_per_outer,
        "pooled25_rae": pooled25_rae,
        "pooled25_median_rae": pooled25_median_rae,
        "nb2063_mean_bag_actual": nb2063_mean_bag_actual,
        "nb2071_bob_mean_rae": nb2071_bob_mean,
        "delta_pooled25_vs_nb2063_5bag": delta_vs_nb2063,
        "delta_pooled25_vs_nb2071_bob_mean": delta_vs_nb2071,
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
        "feat_dim_full", "feat_dim_top50",
        "n_chembl_pool",
        "rae_anchor_chemprop_aux",
        "per_outer_rae",
        "per_fit_mean_rae", "per_fit_median_rae",
        "per_fit_std_rae", "per_fit_min_rae", "per_fit_max_rae",
        "bob_mean_per_outer", "bob_median_per_outer",
        "pooled25_rae", "pooled25_median_rae",
        "nb2063_mean_bag_actual", "nb2071_bob_mean_rae",
        "delta_pooled25_vs_nb2063_5bag",
        "delta_pooled25_vs_nb2071_bob_mean",
        "pre_unblind_clean",
    ):
        print(f"  {k}: {res.get(k)}")
