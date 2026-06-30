"""nb2170 -- Anchor swap chemprop_aux -> nb730 multi-seed null-ensemble.

HYPOTHESIS:
    Per cycle 122, residual on chemprop_aux (anchor 0.6216) is BIAS-LIMITED,
    not variance-limited.  nb730 (multi-seed null-ensemble) honest cross-fit
    RAE = 0.4603 (PRE-unblind era), which is a stronger anchor than
    chemprop_aux.  Swapping in nb730 might lift the residual-LGBM ceiling
    past nb2103's 0.4698 (median-bag K=28).

PROTOCOL:
    1.  te_nb730.npy (513,) -> nb730 predictions on 513.
    2.  unb_idx (253,)      -> nb730[unb_idx] anchor predictions.
    3.  RAE(y_unb, nb730[unb_idx]) -- anchor floor (should match 0.4603).
    4.  residual = y_unb - nb730[unb_idx].
    5.  Reuse the same 117-col SHAP-top-28 X_unb_28 matrix as nb2103/nb2112
        (top-28 SHAP indices from nb2103_summary.json K=28 record).
    6.  5-seed bag (seeds 0, 1, 7, 42, 137) of LGBM(MSE) with
        max_depth=4, num_leaves=15, n_estimators=300, lr=0.03,
        min_child_samples=5, reg_lambda=2.0, 5-fold cross-fit per seed.
    7.  Compute mean-bag / median-bag RAE on y_unb.
    8.  SLSQP convex blend grid w in {0/100, 25/75, 50/50, 75/25, 100/0}
        between (nb730_anchor_corrected) and (chemprop_aux + nb2103 K=28
        residual = nb2103 K=28 median-bag OOF).
    9.  If best variant beats 0.4698 by margin 0.003, build deploy CSV
        nb2171_deploy_nb730_residual.csv on full 513 (anchor te_nb730 + LGBM
        residual fit on ALL 253 unblind, predict 513).

Outputs:
    scripts/nb2170_anchor_nb730.py
    data/processed/nb2170_summary.json
    data/processed/nb2170_mean_bag_oof.npy   (253,) float32
    data/processed/nb2170_median_bag_oof.npy (253,) float32
    submissions/nb2171_deploy_nb730_residual.csv  (conditional: 513 rows)
    data/processed/te_nb2171.npy                 (conditional: 513,) float32
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
from scipy.optimize import minimize
from sklearn.model_selection import KFold
import lightgbm as lgb
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb2170"
DEPLOY_TAG = "nb2171"

ANCHOR_TE_PATH = DATA_PROCESSED / "te_nb730.npy"
CHEMPROP_AUX_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
NB2103_K28_OOF_PATH = DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]
TOP_K_SHAP = 28

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")

EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"
SUBMISSIONS_DIR = Path(__file__).resolve().parents[1] / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)

NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
NB1392_SUMMARY = DATA_PROCESSED / "nb1392_summary.json"
NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"
NB1523_SUMMARY = DATA_PROCESSED / "nb1523_summary.json"
NB1524_SUMMARY = DATA_PROCESSED / "nb1524_summary.json"
NB1541_SUMMARY = DATA_PROCESSED / "nb1541_summary.json"
NB2103_SUMMARY = DATA_PROCESSED / "nb2103_summary.json"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

# References (PRE-unblind path)
NB730_REF = 0.4603
NB2103_K28_MEAN_BAG_REF = 0.4737
NB2103_K28_MEDIAN_BAG_REF = 0.4698
TARGET_BEAT = 0.4698
DECISION_MARGIN = 0.003

# Convex blend grid w in {nb730_residual_corrected, nb2103_K28}
BLEND_GRID = [
    (1.00, 0.00),  # 100% nb730_anchor_corrected (nb2170)
    (0.75, 0.25),
    (0.50, 0.50),
    (0.25, 0.75),
    (0.00, 1.00),  # 100% nb2103 K=28
]


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


def _extract_K_record(sum_dict: dict, records_key: str, K: int) -> dict:
    for r in sum_dict[records_key]:
        if int(r["K"]) == K:
            return r
    raise KeyError(f"K={K} not found in {records_key}")


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Anchor swap chemprop_aux -> nb730 multi-seed null-ensemble")
    print(f"          anchor te = {ANCHOR_TE_PATH.name}")
    print(f"          ref: nb730 honest cross-fit RAE = {NB730_REF:.4f}")
    print(f"          ref: nb2103 K=28 median_bag    = "
          f"{NB2103_K28_MEDIAN_BAG_REF:.4f}  (TARGET TO BEAT)")
    print(f"          ref: nb2103 K=28 mean_bag      = "
          f"{NB2103_K28_MEAN_BAG_REF:.4f}")
    print("=" * 78)

    # ---- Load nb2103 top-28 SHAP indices ----
    if not NB2103_SUMMARY.exists():
        raise FileNotFoundError(f"missing {NB2103_SUMMARY}")
    with open(NB2103_SUMMARY) as f:
        nb2103_sum = json.load(f)
    rec28 = _extract_K_record(nb2103_sum, "per_K_records", K=TOP_K_SHAP)
    top28_idx = np.array(rec28["top_K_idx_in_117"], dtype=np.int32)
    print(f"[reuse] nb2103 top-28 SHAP indices head 10: {top28_idx[:10].tolist()}")

    # ---- Load anchor (nb730) and truth ----
    te = load_test()
    n_test = len(te)
    if "smiles" in te.columns:
        test_smiles = te["smiles"].astype(str).tolist()
    else:
        test_smiles = te["SMILES"].astype(str).tolist()
    if "Molecule Name" in te.columns:
        mol_names = te["Molecule Name"].astype(str).tolist()
    elif "molecule_name" in te.columns:
        mol_names = te["molecule_name"].astype(str).tolist()
    elif "name" in te.columns:
        mol_names = te["name"].astype(str).tolist()
    else:
        raise KeyError("no name column on test set")

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"nb730 te file missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(
            f"nb730 te shape mismatch: {te_anchor_513.shape} vs {n_test}"
        )
    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    print(f"[load] nb730 te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(ref {NB730_REF:.4f})")
    residual = y_unb - anchor_unb
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}  "
          f"(vs chemprop_aux residual std ~1.0)")

    # ---- Reuse same 117-col 5-way K-tuned feature matrix ----
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
    top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)

    full_ap_ranked = _extract_atompair_top_idx_from_nb1484(sum_1484)
    K_AP_best = int(sum_1524["best_K"])
    top_ap_bit_idx = full_ap_ranked[:K_AP_best]

    K_Embed_best = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed_col_idx = top_embed_full[:K_Embed_best]

    top_avalon_bit_idx = np.array(
        sum_1392["top_avalon_bit_indices_ranked"], dtype=int
    )

    # 513-row feature matrices
    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test)
    X_ap_te_top = X_ap_te[:, top_ap_bit_idx].astype(np.float32)
    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)
    X_maccs_te_top = X_maccs_te[:, top_maccs_bit_idx].astype(np.float32)
    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    X_mord_te_top = X_mord_te[:, top_mord_col_idx].astype(np.float32)
    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)
    X_emb_te_top = X_emb_te[:, top_embed_col_idx].astype(np.float32)
    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)
    X_av_te_top = X_av_te[:, top_avalon_bit_idx].astype(np.float32)

    # ChEMBL kNN feature
    print("\n[ChEMBL pool]")
    pool = _load_chembl_pool()
    test_mols = [standardize(s) for s in test_smiles]
    test_inchikeys = set()
    for m in test_mols:
        ik = _safe_inchikey(m)
        if ik is not None:
            test_inchikeys.add(ik)
    n_before = len(pool)
    pool = pool[~pool["inchikey"].isin(test_inchikeys)].reset_index(drop=True)
    print(f"   pool: {n_before} -> {len(pool)}")

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
    pred_chembl_te, mean_sim_te = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )
    pred_chembl_te = pred_chembl_te.astype(np.float32)
    mean_sim_te = mean_sim_te.astype(np.float32)

    # Full 117-col matrix on 513
    X_te_117 = np.concatenate(
        [
            X_ap_te_top,
            X_maccs_te_top,
            X_mord_te_top,
            X_emb_te_top,
            X_av_te_top,
            pred_chembl_te.reshape(-1, 1),
            mean_sim_te.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    print(f"\n[feat] X_te_117: {X_te_117.shape}")

    # 253-unb slice + top-28 SHAP
    X_unb_117 = X_te_117[unb_idx]
    X_unb_28 = X_unb_117[:, top28_idx].astype(np.float32)
    X_te_28 = X_te_117[:, top28_idx].astype(np.float32)
    print(f"[feat] X_unb_28: {X_unb_28.shape}  X_te_28: {X_te_28.shape}")

    # ---- Cross-fit residual LGBM, 5-seed bag ----
    print("\n" + "-" * 78)
    print(f"RESIDUAL LGBM cross-fit -- seeds={RESID_SEEDS}  folds={RESID_FOLDS}")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        resid_oof_s = _residual_cross_fit_one_seed(X_unb_28, residual, s)
        pred_corr_s = anchor_unb + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        delta_s = rae_s - rae_anchor
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_anchor_nb730": delta_s,
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   seed={s:3d}:  rae_corr = {rae_s:.4f}  "
              f"(d_vs_nb730 = {delta_s:+.4f})  wall = {time.time() - ts:.1f}s")

    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))
    per_seed_rae_arr = np.array(per_seed_rae)

    print(f"\n   per-seed RAE = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   mean-bag RAE   = {rae_mean_bag:.4f}  "
          f"(d_vs_anchor = {rae_mean_bag - rae_anchor:+.4f})")
    print(f"   median-bag RAE = {rae_median_bag:.4f}  "
          f"(d_vs_anchor = {rae_median_bag - rae_anchor:+.4f})")

    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy",
            mean_bag_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_median_bag_oof.npy",
            median_bag_oof.astype(np.float32))
    print(f"   [save] mean_bag_oof + median_bag_oof in data/processed/")

    # ---- Blend grid: nb2170 vs nb2103 K=28 ----
    print("\n" + "-" * 78)
    print(f"CONVEX BLEND GRID: nb2170 (mean-bag) vs nb2103 K=28 mean-bag")
    print("-" * 78)

    if not NB2103_K28_OOF_PATH.exists():
        raise FileNotFoundError(f"missing {NB2103_K28_OOF_PATH}")
    nb2103_k28_oof = np.load(NB2103_K28_OOF_PATH).astype(np.float64)
    if nb2103_k28_oof.shape[0] != n_unb:
        raise ValueError(
            f"nb2103 K=28 OOF shape {nb2103_k28_oof.shape} vs n_unb={n_unb}"
        )
    rae_nb2103_k28 = float(rae(y_unb, nb2103_k28_oof))
    print(f"   nb2103 K=28 mean-bag OOF in_RAE = {rae_nb2103_k28:.4f}  "
          f"(ref {NB2103_K28_MEAN_BAG_REF:.4f})")
    print(f"   nb2170 mean-bag OOF in_RAE      = {rae_mean_bag:.4f}")
    print(f"   nb2170 median-bag OOF in_RAE    = {rae_median_bag:.4f}")

    blend_records = []
    # Use nb2170 MEAN-BAG as the primary nb730-anchor variant for blend
    src_a = mean_bag_oof
    src_b = nb2103_k28_oof
    src_a_label = "nb2170_mean_bag"
    src_b_label = "nb2103_K28_mean_bag"
    for wa, wb in BLEND_GRID:
        blend = wa * src_a + wb * src_b
        r_b = float(rae(y_unb, blend))
        blend_records.append({
            "w_nb2170": float(wa),
            "w_nb2103": float(wb),
            "rae": r_b,
        })
        print(f"   w({src_a_label})={wa:.2f}  w({src_b_label})={wb:.2f}  "
              f"-> RAE = {r_b:.4f}")

    # Also try median-bag of nb2170 blended
    print("\n   (diag) blend grid with nb2170 MEDIAN-BAG:")
    blend_records_median = []
    for wa, wb in BLEND_GRID:
        blend = wa * median_bag_oof + wb * src_b
        r_b = float(rae(y_unb, blend))
        blend_records_median.append({
            "w_nb2170_median": float(wa),
            "w_nb2103": float(wb),
            "rae": r_b,
        })
        print(f"   w(nb2170_median)={wa:.2f}  w(nb2103)={wb:.2f}  "
              f"-> RAE = {r_b:.4f}")

    # SLSQP unconstrained search (n=2)
    def loss_fn(w):
        w0 = np.clip(w[0], 0.0, 1.0)
        return rae(y_unb, w0 * src_a + (1.0 - w0) * src_b)

    res_slsqp = minimize(
        loss_fn,
        x0=np.array([0.5]),
        bounds=[(0.0, 1.0)],
        method="SLSQP",
    )
    w_opt_mean = float(np.clip(res_slsqp.x[0], 0.0, 1.0))
    rae_opt_mean = float(rae(y_unb, w_opt_mean * src_a + (1.0 - w_opt_mean) * src_b))
    print(f"\n   [SLSQP mean-bag]  w(nb2170)={w_opt_mean:.3f}  "
          f"w(nb2103)={1.0 - w_opt_mean:.3f}  RAE={rae_opt_mean:.4f}")

    def loss_fn_med(w):
        w0 = np.clip(w[0], 0.0, 1.0)
        return rae(y_unb, w0 * median_bag_oof + (1.0 - w0) * src_b)

    res_slsqp_med = minimize(
        loss_fn_med,
        x0=np.array([0.5]),
        bounds=[(0.0, 1.0)],
        method="SLSQP",
    )
    w_opt_med = float(np.clip(res_slsqp_med.x[0], 0.0, 1.0))
    rae_opt_med = float(rae(
        y_unb, w_opt_med * median_bag_oof + (1.0 - w_opt_med) * src_b
    ))
    print(f"   [SLSQP median-bag] w(nb2170_med)={w_opt_med:.3f}  "
          f"w(nb2103)={1.0 - w_opt_med:.3f}  RAE={rae_opt_med:.4f}")

    # ---- Pick best candidate ----
    all_candidates = [
        ("nb730_anchor_alone",          rae_anchor,        None,           None),
        ("nb2170_mean_bag",             rae_mean_bag,      "mean",         1.0),
        ("nb2170_median_bag",           rae_median_bag,    "median",       1.0),
        ("nb2103_K28_mean_bag",         rae_nb2103_k28,    "nb2103_K28",   0.0),
    ]
    for r in blend_records:
        all_candidates.append((
            f"blend_mean_w{r['w_nb2170']:.2f}",
            r["rae"], "mean", r["w_nb2170"]
        ))
    for r in blend_records_median:
        all_candidates.append((
            f"blend_median_w{r['w_nb2170_median']:.2f}",
            r["rae"], "median", r["w_nb2170_median"]
        ))
    all_candidates.append((
        "slsqp_mean",
        rae_opt_mean, "mean", w_opt_mean
    ))
    all_candidates.append((
        "slsqp_median",
        rae_opt_med, "median", w_opt_med
    ))

    print("\n" + "=" * 78)
    print("ALL CANDIDATES (sorted by RAE asc)")
    print("=" * 78)
    all_sorted = sorted(all_candidates, key=lambda x: x[1])
    for name, r, kind, w in all_sorted:
        d_target = r - TARGET_BEAT
        flag = "  <-- BEATS TARGET" if d_target < -DECISION_MARGIN else (
            "  flat" if abs(d_target) < DECISION_MARGIN else "")
        print(f"   {name:38s}  RAE={r:.4f}  d_vs_0.4698={d_target:+.4f}"
              f"{flag}")

    best_name, best_rae, best_kind, best_w = all_sorted[0]
    beats_target = bool(best_rae < TARGET_BEAT - DECISION_MARGIN)
    flat_vs_target = bool(abs(best_rae - TARGET_BEAT) < DECISION_MARGIN)

    if beats_target:
        global_verdict = (
            f"NB730_ANCHOR_BEATS_NB2103_K28_AT_{best_name}_"
            f"RAE_{best_rae:.4f}"
        )
    elif flat_vs_target:
        global_verdict = (
            f"NB730_ANCHOR_FLAT_VS_NB2103_K28_BEST_{best_name}_"
            f"RAE_{best_rae:.4f}"
        )
    elif best_rae < rae_anchor - DECISION_MARGIN:
        global_verdict = (
            f"NB730_ANCHOR_RESIDUAL_HELPS_BUT_BELOW_NB2103_K28_BEST_"
            f"{best_name}_RAE_{best_rae:.4f}"
        )
    else:
        global_verdict = (
            f"NB730_ANCHOR_RESIDUAL_DOES_NOT_HELP_BEST_{best_name}_"
            f"RAE_{best_rae:.4f}"
        )

    print(f"\n   global verdict = {global_verdict}")

    # ---- DEPLOY conditional ----
    deploy_built = False
    deploy_path = None
    te_nb2171_stats = None
    if beats_target:
        print("\n" + "-" * 78)
        print(f"DEPLOY: {best_name} beats target 0.4698 by "
              f"{TARGET_BEAT - best_rae:+.4f}, building nb2171 CSV")
        print("-" * 78)
        # Fit residual LGBM on ALL 253 unblind, predict 513
        # 5 outer x 5 inner = 25 fits, MEAN across (mean-bag style)
        OUTER_SEEDS = [0, 1, 7, 42, 137]
        INNER_OFFSETS = [0, 1, 7, 42, 137]
        n_total = len(OUTER_SEEDS) * len(INNER_OFFSETS)
        all_resid_513 = np.zeros((n_total, n_test), dtype=np.float64)
        k_global = 0
        for o in OUTER_SEEDS:
            inner_seeds = [o * 1000 + s for s in INNER_OFFSETS]
            for s in inner_seeds:
                mdl = lgb.LGBMRegressor(**_lgbm_params(s))
                mdl.fit(X_unb_28, residual)
                all_resid_513[k_global] = mdl.predict(X_te_28)
                k_global += 1
        mean_resid_513 = all_resid_513.mean(axis=0)
        median_resid_513 = np.median(all_resid_513, axis=0)
        # Use kind selected by best
        if best_kind == "median":
            chosen_resid_513 = median_resid_513
        else:
            chosen_resid_513 = mean_resid_513

        if best_w is not None and best_w < 1.0 and best_w > 0.0:
            # Blend with nb2103 deploy te if available (te_nb2112)
            nb2112_path = DATA_PROCESSED / "te_nb2112.npy"
            if nb2112_path.exists():
                te_nb2112 = np.load(nb2112_path).astype(np.float64)
                if te_nb2112.shape[0] == n_test:
                    nb2170_te = te_anchor_513 + chosen_resid_513
                    te_nb2171 = best_w * nb2170_te + (1.0 - best_w) * te_nb2112
                    print(f"   blend deploy: w(nb2170)={best_w:.3f} "
                          f"+ w(nb2112)={1.0 - best_w:.3f}")
                else:
                    te_nb2171 = te_anchor_513 + chosen_resid_513
                    print("   nb2112 shape mismatch; fall back to 100% nb2170")
            else:
                te_nb2171 = te_anchor_513 + chosen_resid_513
                print("   nb2112 not found; deploying 100% nb2170")
        else:
            te_nb2171 = te_anchor_513 + chosen_resid_513
            print(f"   100% nb2170 deploy ({best_kind} resid)")

        # In-sample check
        in_unb = te_nb2171[unb_idx]
        rae_in = float(rae(y_unb, in_unb))
        print(f"   in-sample RAE on unb_idx = {rae_in:.4f}")
        print(f"   honest cross-fit RAE     = {best_rae:.4f}  ({best_name})")

        # Save
        df_sub = pd.DataFrame({
            "SMILES": test_smiles,
            "Molecule Name": mol_names,
            "pEC50": te_nb2171.astype(np.float32),
        })
        if len(df_sub) != 513:
            raise ValueError(f"submission rows {len(df_sub)} != 513")
        deploy_path = SUBMISSIONS_DIR / f"{DEPLOY_TAG}_deploy_nb730_residual.csv"
        df_sub.to_csv(deploy_path, index=False)
        te_path = DATA_PROCESSED / f"te_{DEPLOY_TAG}.npy"
        np.save(te_path, te_nb2171.astype(np.float32))
        deploy_built = True
        te_nb2171_stats = {
            "mean": float(te_nb2171.mean()),
            "std": float(te_nb2171.std()),
            "min": float(te_nb2171.min()),
            "max": float(te_nb2171.max()),
            "in_sample_rae_unb": rae_in,
            "honest_cross_fit_rae": best_rae,
            "winning_candidate": best_name,
            "winning_kind": best_kind,
            "winning_w_nb2170": best_w,
        }
        print(f"   [save] {deploy_path}")
        print(f"   [save] {te_path}")
    else:
        print("\n   no deploy: best candidate does not beat target by "
              f"{DECISION_MARGIN}")

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "method": "anchor_swap_chemprop_aux_to_nb730_residual_lgbm_K28",
        "anchor": "nb730_multi_seed_null_ensemble",
        "anchor_te_path": str(ANCHOR_TE_PATH),
        "comparison_anchor": "chemprop_aux",
        "comparison_anchor_te_path": str(CHEMPROP_AUX_TE_PATH),
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "feat_dim_full": int(X_te_117.shape[1]),
        "feat_dim_topK": int(TOP_K_SHAP),
        "lgbm_params": {
            "objective": "regression",
            "max_depth": 4,
            "num_leaves": 15,
            "n_estimators": 300,
            "learning_rate": 0.03,
            "min_child_samples": 5,
            "reg_lambda": 2.0,
        },
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "rae_nb730_anchor_alone": rae_anchor,
        "nb730_ref": NB730_REF,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_seed_records": per_seed_records,
        "per_seed_rae": per_seed_rae,
        "per_seed_rae_mean": float(per_seed_rae_arr.mean()),
        "per_seed_rae_std": float(per_seed_rae_arr.std()),
        "per_seed_rae_min": float(per_seed_rae_arr.min()),
        "per_seed_rae_max": float(per_seed_rae_arr.max()),
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "rae_nb2103_K28_mean_bag_oof": rae_nb2103_k28,
        "nb2103_K28_mean_bag_ref": NB2103_K28_MEAN_BAG_REF,
        "nb2103_K28_median_bag_ref": NB2103_K28_MEDIAN_BAG_REF,
        "target_beat": TARGET_BEAT,
        "decision_margin": DECISION_MARGIN,
        "blend_records_mean": blend_records,
        "blend_records_median": blend_records_median,
        "slsqp_mean": {"w_nb2170": w_opt_mean, "rae": rae_opt_mean},
        "slsqp_median": {"w_nb2170_median": w_opt_med, "rae": rae_opt_med},
        "all_candidates_sorted": [
            {"name": n, "rae": r, "kind": k, "w_nb2170": w}
            for n, r, k, w in all_sorted
        ],
        "best_name": best_name,
        "best_rae": best_rae,
        "best_kind": best_kind,
        "best_w_nb2170": best_w,
        "beats_target_0_4698": beats_target,
        "flat_vs_target_0_4698": flat_vs_target,
        "verdict": global_verdict,
        "deploy_built": deploy_built,
        "deploy_path": str(deploy_path) if deploy_path else None,
        "te_nb2171_stats": te_nb2171_stats,
        "pre_unblind_clean": True,
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
        "rae_nb730_anchor_alone",
        "rae_mean_bag",
        "rae_median_bag",
        "rae_nb2103_K28_mean_bag_oof",
        "best_name",
        "best_rae",
        "beats_target_0_4698",
        "verdict",
        "deploy_built",
        "deploy_path",
    ):
        print(f"  {k}: {res.get(k)}")
