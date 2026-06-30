"""nb1086 -- Recursive single-feature add beyond the K=28 SHAP set.

HYPOTHESIS:
    nb2103 found K=28 mean_bag RAE = 0.4737, median_bag RAE = 0.4698 on the
    117-col 5-way K-tuned feature matrix using the top-28 SHAP-ranked features
    (chemprop_aux residual, LGBM(MSE), 5-seed bag, 5-fold cross-fit).

    Hypothesis: ONE additional feature outside the top-28 SHAP set may carry
    orthogonal signal that beats this baseline WITHOUT forcing a global
    SHAP-reselection.  Test each of the 89 (= 117 - 28) candidates one at a
    time, K=29 (= top-28 + 1 add).

PROTOCOL:
    1. Rebuild the 117-col 5-way K-tuned feature matrix (same as
       nb2063 / nb2081 / nb2091 / nb2103: AtomPair-25 + MACCS-20 + Mordred-20
       + ChempropEmbed-20 + Avalon-30 + ChEMBL kNN-2).
    2. Identify top-28 SHAP indices = argsort(-nb2063_shap_importance)[:28].
    3. Identify 89 remaining feature indices (the complement in {0..116}).
    4. For each of the 89 candidate adds:
       a. Build X_unb_K29 = top-28 cols + 1 added col (29 features).
       b. 5-seed bag (0, 1, 7, 42, 137) of LGBM(MSE) with KFold(5,
          shuffle=True, random_state=seed) cross-fit on
          residual = y_unb - chemprop_aux te[unb_idx].
       c. Compute mean-bag RAE and median-bag RAE.
    5. Rank by gain vs nb2103 K=28 baseline (mean_bag 0.4737, median_bag
       0.4698).
    6. Apply Bonferroni-style multiple-test correction: a successful add
       requires gain >= 0.005 RAE (~ 3x the usual 0.003 decision margin,
       since 89 trials).
    7. If the best add gain >= 0.005 (on mean-bag), build deploy CSV
       submissions/nb1086_deploy_K29.csv using 5 outer x 5 inner = 25-bag
       fit-on-all-253 / predict-513 / row-median pattern (matches nb2112).

Outputs:
    scripts/nb1086_recursive_add.py
    data/processed/nb1086_summary.json
    submissions/nb1086_deploy_K29.csv   (only if best gain >= 0.005)
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

TAG = "nb1086"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]
TOP_K_SHAP = 28        # K=28 SHAP set (from nb2103)
K_ADDED = 29           # K=28 + 1 new feature

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
NB2063_SHAP_IMP = DATA_PROCESSED / "nb2063_shap_importance_full117.npy"
NB2103_SUMMARY = DATA_PROCESSED / "nb2103_summary.json"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

# Anchors (PRE-unblind cross-fit)
CHEMPROP_AUX_REF = 0.6216
NB2103_K28_MEAN_BAG_REF = 0.4737      # nb2103 K=28 mean-bag RAE
NB2103_K28_MEDIAN_BAG_REF = 0.4698    # nb2103 K=28 median-bag RAE
N_TRIALS_EST = 89                      # 117 - 28
# Bonferroni-style margin: 3x the usual 0.003 single-test decision margin
DECISION_MARGIN_BONF = 0.005


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
    """Same union as nb2063/nb2081/nb2091/nb2103/nb2111/nb2112."""
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
    """LGBM(MSE) -- identical to nb2063/nb2081/nb2091/nb2103."""
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


def _build_full_117_matrices(test_smiles, test_mols, n_test, unb_idx):
    """Build full 117-col matrices on BOTH 513-test AND 253-unb."""
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

    full_ap_ranked = _extract_atompair_top_idx_from_nb1484(sum_1484)
    K_AP_best = int(sum_1524["best_K"])
    top_ap_bit_idx = full_ap_ranked[:K_AP_best]

    K_Embed_best = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed_col_idx = top_embed_full[:K_Embed_best]

    top_avalon_bit_idx = np.array(
        sum_1392["top_avalon_bit_indices_ranked"], dtype=int
    )

    n_top_maccs = int(len(top_maccs_bit_idx))
    n_top_mord = int(len(top_mord_col_idx))
    n_top_ap = int(len(top_ap_bit_idx))
    n_top_embed = int(len(top_embed_col_idx))
    n_top_avalon = int(len(top_avalon_bit_idx))
    print(f"[reuse] top-{n_top_ap}  AtomPair bits (nb1524 K={K_AP_best})")
    print(f"[reuse] top-{n_top_maccs}  MACCS bits  (nb1352)")
    print(f"[reuse] top-{n_top_mord}  Mordred cols (nb1523 K={K_Mord_best})")
    print(f"[reuse] top-{n_top_embed}  ChempropEmbed dims (nb1541 K={K_Embed_best})")
    print(f"[reuse] top-{n_top_avalon} Avalon bits (nb1392 SHAP K=30)")

    # ---- Slice feature caches on 513, then take unb_idx ----
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

    # ---- ChEMBL kNN ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL (union)")
    print("-" * 78)
    pool = _load_chembl_pool()
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
    pred_chembl_te = pred_chembl_pec50.astype(np.float32)
    mean_sim_te = mean_sim.astype(np.float32)

    # ---- Stack 513 117-col + slice 253 ----
    X_te_117 = np.concatenate(
        [
            X_ap_te_top, X_maccs_te_top, X_mord_te_top,
            X_emb_te_top, X_av_te_top,
            pred_chembl_te.reshape(-1, 1),
            mean_sim_te.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    expected_dim = (n_top_ap + n_top_maccs + n_top_mord
                    + n_top_embed + n_top_avalon + 2)
    if X_te_117.shape[1] != expected_dim:
        raise ValueError(
            f"feat_dim_full {X_te_117.shape[1]} != expected {expected_dim}"
        )
    X_unb_117 = X_te_117[unb_idx].astype(np.float32)

    # ---- Feature names ----
    feat_names: list[str] = []
    feat_family: list[str] = []
    for b in top_ap_bit_idx:
        feat_names.append(f"AtomPair_bit_{int(b)}")
        feat_family.append("AtomPair")
    for b in top_maccs_bit_idx:
        feat_names.append(f"MACCS_bit_{int(b)}")
        feat_family.append("MACCS")
    for c in top_mord_col_idx:
        feat_names.append(f"Mordred_col_{int(c)}")
        feat_family.append("Mordred")
    for d in top_embed_col_idx:
        feat_names.append(f"ChempropEmbed_dim_{int(d)}")
        feat_family.append("ChempropEmbed")
    for b in top_avalon_bit_idx:
        feat_names.append(f"Avalon_bit_{int(b)}")
        feat_family.append("Avalon")
    feat_names.append("pred_chembl_pec50")
    feat_family.append("ChEMBL_kNN")
    feat_names.append("mean_sim")
    feat_family.append("ChEMBL_kNN")
    assert len(feat_names) == X_te_117.shape[1]
    return X_te_117, X_unb_117, feat_names, feat_family, len(pool)


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- RECURSIVE single-feature add beyond K=28 SHAP set")
    print(f"          anchor={ANCHOR}  K_added={K_ADDED}  "
          f"seeds={RESID_SEEDS}  folds={RESID_FOLDS}")
    print(f"          ref: nb2103 K=28 mean_bag {NB2103_K28_MEAN_BAG_REF:.4f}"
          f" / median_bag {NB2103_K28_MEDIAN_BAG_REF:.4f}")
    print(f"          Bonferroni margin = {DECISION_MARGIN_BONF:.3f} "
          f"(~3x of 0.003, n_trials={N_TRIALS_EST})")
    print("=" * 78)

    # ---- nb2063 SHAP importance ----
    if not NB2063_SHAP_IMP.exists():
        raise FileNotFoundError(f"missing {NB2063_SHAP_IMP}")
    shap_imp_full117 = np.load(NB2063_SHAP_IMP).astype(np.float32)
    if shap_imp_full117.shape[0] != 117:
        raise ValueError(
            f"SHAP importance length {shap_imp_full117.shape[0]} != 117"
        )
    full_rank_order = np.argsort(-shap_imp_full117).astype(np.int32)
    top28_idx = full_rank_order[:TOP_K_SHAP].astype(np.int32)
    remaining_idx = np.array(
        [i for i in range(117) if i not in set(top28_idx.tolist())],
        dtype=np.int32,
    )
    if remaining_idx.shape[0] != 117 - TOP_K_SHAP:
        raise ValueError(
            f"remaining_idx has {remaining_idx.shape[0]} entries, "
            f"expected {117 - TOP_K_SHAP}"
        )
    print(f"[shap] top-{TOP_K_SHAP} indices (head 10): {top28_idx[:10].tolist()}")
    print(f"[shap] {len(remaining_idx)} candidate adds (heads): "
          f"{remaining_idx[:10].tolist()} ...")

    # ---- Load anchor + truth ----
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
    test_mols = [standardize(s) for s in test_smiles]

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"chemprop_aux te file missing")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(
            f"chemprop_aux te shape mismatch: {te_anchor_513.shape} vs {n_test}"
        )
    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor_unb
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Build full 117-col matrices ----
    print("\n" + "-" * 78)
    print("BUILD 117-COL 5-WAY K-TUNED MATRIX")
    print("-" * 78)
    X_te_117, X_unb_117, feat_names, feat_family, n_pool = \
        _build_full_117_matrices(test_smiles, test_mols, n_test, unb_idx)
    print(f"   X_te_117  = {X_te_117.shape}")
    print(f"   X_unb_117 = {X_unb_117.shape}")

    # ---- Sanity check: K=28 baseline should reproduce nb2103 ----
    print("\n" + "-" * 78)
    print(f"SANITY: re-run K=28 baseline on rebuilt X_unb_117")
    print("-" * 78)
    X_unb_28 = X_unb_117[:, top28_idx].astype(np.float32)
    base_per_seed_corr = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    base_per_seed_rae = []
    for i, s in enumerate(RESID_SEEDS):
        oof_s = _residual_cross_fit_one_seed(X_unb_28, residual, s)
        pred_s = anchor_unb + oof_s
        base_per_seed_corr[i] = pred_s
        rae_s = float(rae(y_unb, pred_s))
        base_per_seed_rae.append(rae_s)
        print(f"   K=28 seed={s:3d}: rae = {rae_s:.4f}")
    base_mean_bag_oof = base_per_seed_corr.mean(axis=0)
    base_median_bag_oof = np.median(base_per_seed_corr, axis=0)
    base_mean_bag = float(rae(y_unb, base_mean_bag_oof))
    base_median_bag = float(rae(y_unb, base_median_bag_oof))
    print(f"\n   K=28 baseline (rebuilt) mean_bag   = {base_mean_bag:.4f}  "
          f"(ref {NB2103_K28_MEAN_BAG_REF:.4f}, "
          f"delta {base_mean_bag - NB2103_K28_MEAN_BAG_REF:+.4f})")
    print(f"   K=28 baseline (rebuilt) median_bag = {base_median_bag:.4f}  "
          f"(ref {NB2103_K28_MEDIAN_BAG_REF:.4f}, "
          f"delta {base_median_bag - NB2103_K28_MEDIAN_BAG_REF:+.4f})")
    if abs(base_mean_bag - NB2103_K28_MEAN_BAG_REF) > 0.01:
        print(f"   [warn] K=28 baseline mean_bag drifted "
              f"{abs(base_mean_bag - NB2103_K28_MEAN_BAG_REF):.4f} from nb2103 ref")

    # ---- Sweep: 89 single-feature adds at K=29 ----
    print("\n" + "-" * 78)
    print(f"SWEEP: {len(remaining_idx)} single-feature adds at K={K_ADDED} "
          f"(top-{TOP_K_SHAP} + 1)")
    print("-" * 78)
    per_add_records: list[dict] = []
    for i_add, j in enumerate(remaining_idx):
        ts = time.time()
        # Build K=29 = top-28 + add j
        cols = np.concatenate(
            [top28_idx, np.array([j], dtype=np.int32)]
        ).astype(np.int32)
        X_unb_K29 = X_unb_117[:, cols].astype(np.float32)
        per_seed_corr_j = np.zeros((len(RESID_SEEDS), n_unb),
                                   dtype=np.float64)
        per_seed_rae_j = []
        for i_s, s in enumerate(RESID_SEEDS):
            oof_s = _residual_cross_fit_one_seed(X_unb_K29, residual, s)
            pred_s = anchor_unb + oof_s
            per_seed_corr_j[i_s] = pred_s
            per_seed_rae_j.append(float(rae(y_unb, pred_s)))
        mean_bag_oof_j = per_seed_corr_j.mean(axis=0)
        median_bag_oof_j = np.median(per_seed_corr_j, axis=0)
        rae_mean_bag_j = float(rae(y_unb, mean_bag_oof_j))
        rae_median_bag_j = float(rae(y_unb, median_bag_oof_j))
        gain_mean_bag = NB2103_K28_MEAN_BAG_REF - rae_mean_bag_j
        gain_median_bag = NB2103_K28_MEDIAN_BAG_REF - rae_median_bag_j
        rec = {
            "i_add": int(i_add),
            "add_feat_idx_in_117": int(j),
            "add_feat_name": feat_names[int(j)],
            "add_feat_family": feat_family[int(j)],
            "shap_importance": float(shap_imp_full117[int(j)]),
            "shap_rank_in_117": int(np.where(full_rank_order == j)[0][0]),
            "per_seed_rae": per_seed_rae_j,
            "rae_per_seed_mean": float(np.mean(per_seed_rae_j)),
            "rae_per_seed_std": float(np.std(per_seed_rae_j)),
            "rae_mean_bag": rae_mean_bag_j,
            "rae_median_bag": rae_median_bag_j,
            "gain_mean_bag_vs_K28": gain_mean_bag,
            "gain_median_bag_vs_K28": gain_median_bag,
            "wall_sec": round(time.time() - ts, 2),
        }
        per_add_records.append(rec)
        if i_add < 5 or i_add % 10 == 0 or gain_mean_bag > 0.003:
            print(f"   add#{i_add:3d}  feat_idx={int(j):3d}  "
                  f"name={feat_names[int(j)]:<25s}  fam={feat_family[int(j)]:<13s}"
                  f"  mean_bag={rae_mean_bag_j:.4f}  "
                  f"median_bag={rae_median_bag_j:.4f}  "
                  f"gain_mean={gain_mean_bag:+.4f}  "
                  f"gain_med={gain_median_bag:+.4f}  "
                  f"({time.time() - ts:.1f}s)")

    # ---- Rank by gain (mean-bag) ----
    print("\n" + "=" * 78)
    print(f"RANKED BY gain_mean_bag_vs_K28 (top 15)")
    print("=" * 78)
    sorted_by_gain_mean = sorted(
        per_add_records, key=lambda r: -r["gain_mean_bag_vs_K28"]
    )
    print(f"   {'rank':>4s}  {'feat_idx':>8s}  {'name':<28s}  {'fam':<13s}  "
          f"{'mean_bag':>10s}  {'gain_mean':>11s}  {'median_bag':>11s}  "
          f"{'gain_med':>10s}  {'shap_rank':>9s}")
    for ri, r in enumerate(sorted_by_gain_mean[:15]):
        print(f"   {ri + 1:>4d}  {r['add_feat_idx_in_117']:>8d}  "
              f"{r['add_feat_name']:<28s}  {r['add_feat_family']:<13s}  "
              f"{r['rae_mean_bag']:>10.4f}  "
              f"{r['gain_mean_bag_vs_K28']:>+11.4f}  "
              f"{r['rae_median_bag']:>11.4f}  "
              f"{r['gain_median_bag_vs_K28']:>+10.4f}  "
              f"{r['shap_rank_in_117']:>9d}")

    # ---- Best by median-bag (alternative) ----
    print(f"\n   Top 5 by gain_median_bag_vs_K28:")
    sorted_by_gain_med = sorted(
        per_add_records, key=lambda r: -r["gain_median_bag_vs_K28"]
    )
    for ri, r in enumerate(sorted_by_gain_med[:5]):
        print(f"     median#{ri + 1}  feat_idx={r['add_feat_idx_in_117']:3d}  "
              f"name={r['add_feat_name']:<28s}  "
              f"median_bag={r['rae_median_bag']:.4f}  "
              f"gain_med={r['gain_median_bag_vs_K28']:+.4f}")

    # ---- Bonferroni-style verdict on best-by-mean-bag ----
    best_rec = sorted_by_gain_mean[0]
    best_gain_mean = float(best_rec["gain_mean_bag_vs_K28"])
    best_gain_median = float(best_rec["gain_median_bag_vs_K28"])
    best_passes_bonf_mean = best_gain_mean >= DECISION_MARGIN_BONF
    best_rec_med = sorted_by_gain_med[0]
    best_gain_med_only = float(best_rec_med["gain_median_bag_vs_K28"])
    best_passes_bonf_median = best_gain_med_only >= DECISION_MARGIN_BONF
    print(f"\n   best-by-mean-bag      = idx {best_rec['add_feat_idx_in_117']}  "
          f"name={best_rec['add_feat_name']}  "
          f"gain_mean = {best_gain_mean:+.4f}  "
          f"(threshold {DECISION_MARGIN_BONF:.3f})  "
          f"{'PASS' if best_passes_bonf_mean else 'FAIL'} Bonferroni")
    print(f"   best-by-median-bag    = idx {best_rec_med['add_feat_idx_in_117']}  "
          f"name={best_rec_med['add_feat_name']}  "
          f"gain_med = {best_gain_med_only:+.4f}  "
          f"(threshold {DECISION_MARGIN_BONF:.3f})  "
          f"{'PASS' if best_passes_bonf_median else 'FAIL'} Bonferroni")

    # ---- Deploy verdict ----
    do_deploy = best_passes_bonf_mean
    deploy_csv_path = None
    deploy_stats = None
    deploy_in_rae_unb = None
    if do_deploy:
        print("\n" + "-" * 78)
        print(f"DEPLOY: best add (mean-bag) idx={best_rec['add_feat_idx_in_117']} "
              f"({best_rec['add_feat_name']}) gain {best_gain_mean:+.4f} "
              f">= {DECISION_MARGIN_BONF:.3f}, building 5x5=25-bag deploy")
        print("-" * 78)
        best_j = int(best_rec["add_feat_idx_in_117"])
        cols = np.concatenate(
            [top28_idx, np.array([best_j], dtype=np.int32)]
        ).astype(np.int32)
        X_te_K29 = X_te_117[:, cols].astype(np.float32)
        X_unb_K29 = X_unb_117[:, cols].astype(np.float32)
        outer = [0, 1, 7, 42, 137]
        inner = [0, 1, 7, 42, 137]
        n_total = len(outer) * len(inner)
        all_resid_513 = np.zeros((n_total, n_test), dtype=np.float64)
        k_global = 0
        for o in outer:
            for s in inner:
                seed = o * 1000 + s
                mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
                mdl.fit(X_unb_K29, residual)
                all_resid_513[k_global] = mdl.predict(X_te_K29)
                k_global += 1
        median_resid_513 = np.median(all_resid_513, axis=0)
        te_deploy = te_anchor_513 + median_resid_513
        deploy_in_rae_unb = float(rae(y_unb, te_deploy[unb_idx]))
        df_sub = pd.DataFrame({
            "SMILES": test_smiles,
            "Molecule Name": mol_names,
            "pEC50": te_deploy.astype(np.float32),
        })
        if len(df_sub) != n_test:
            raise ValueError(f"submission rows {len(df_sub)} != {n_test}")
        deploy_csv_path = SUBMISSIONS_DIR / "nb1086_deploy_K29.csv"
        df_sub.to_csv(deploy_csv_path, index=False)
        deploy_stats = {
            "te_mean": float(te_deploy.mean()),
            "te_std": float(te_deploy.std()),
            "te_min": float(te_deploy.min()),
            "te_max": float(te_deploy.max()),
            "median_resid_mean": float(median_resid_513.mean()),
            "median_resid_std": float(median_resid_513.std()),
        }
        print(f"   [save] {deploy_csv_path}  ({len(df_sub)} rows)")
        print(f"   in-sample RAE on unb_idx (median bag) = "
              f"{deploy_in_rae_unb:.4f}")
        print(f"   te stats: mean={deploy_stats['te_mean']:.4f}  "
              f"std={deploy_stats['te_std']:.4f}  "
              f"min={deploy_stats['te_min']:.4f}  "
              f"max={deploy_stats['te_max']:.4f}")
        verdict = (
            f"BEST_ADD_BONFERRONI_PASS_DEPLOYED_idx="
            f"{best_rec['add_feat_idx_in_117']}_"
            f"{best_rec['add_feat_name']}_gain={best_gain_mean:+.4f}"
        )
    else:
        verdict = (
            f"NO_ADD_BEATS_BONFERRONI_THRESHOLD_"
            f"best_gain_mean={best_gain_mean:+.4f}_"
            f"threshold={DECISION_MARGIN_BONF:.3f}_NO_DEPLOY"
        )
        print(f"\n   No deploy (best gain {best_gain_mean:+.4f} < threshold "
              f"{DECISION_MARGIN_BONF:.3f}).")

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "method": (
            "recursive_single_feature_add_beyond_top28_SHAP_117col_5way"
        ),
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "data_source": (
            "117-col 5-way K-tuned matrix (AtomPair-25/MACCS-20/Mordred-20/"
            "ChempropEmbed-20/Avalon-30 + ChEMBL kNN-2), top-28 SHAP set "
            "from nb2063 + 1 candidate add at K=29"
        ),
        "model_family": "LightGBM",
        "lgbm_objective": "regression",
        "lgbm_max_depth": 4,
        "lgbm_num_leaves": 15,
        "lgbm_n_estimators": 300,
        "lgbm_learning_rate": 0.03,
        "lgbm_min_child_samples": 5,
        "lgbm_reg_lambda": 2.0,
        "feat_dim_full": 117,
        "K_anchor": TOP_K_SHAP,
        "K_added": K_ADDED,
        "n_candidates": int(len(remaining_idx)),
        "decision_margin_bonf": DECISION_MARGIN_BONF,
        "decision_margin_single": 0.003,
        "n_chembl_pool": int(n_pool),
        "n_unb": int(n_unb),
        "n_test": int(n_test),
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "top28_idx_in_117": top28_idx.tolist(),
        "remaining_idx_in_117": remaining_idx.tolist(),
        "k28_baseline_mean_bag_rebuilt": base_mean_bag,
        "k28_baseline_median_bag_rebuilt": base_median_bag,
        "k28_baseline_mean_bag_ref_nb2103": NB2103_K28_MEAN_BAG_REF,
        "k28_baseline_median_bag_ref_nb2103": NB2103_K28_MEDIAN_BAG_REF,
        "per_add_records": per_add_records,
        "top_15_by_gain_mean_bag": [
            {
                "rank": ri + 1,
                "add_feat_idx_in_117": r["add_feat_idx_in_117"],
                "add_feat_name": r["add_feat_name"],
                "add_feat_family": r["add_feat_family"],
                "shap_rank_in_117": r["shap_rank_in_117"],
                "rae_mean_bag": r["rae_mean_bag"],
                "rae_median_bag": r["rae_median_bag"],
                "gain_mean_bag_vs_K28": r["gain_mean_bag_vs_K28"],
                "gain_median_bag_vs_K28": r["gain_median_bag_vs_K28"],
            }
            for ri, r in enumerate(sorted_by_gain_mean[:15])
        ],
        "best_add_by_mean_bag": {
            "add_feat_idx_in_117": best_rec["add_feat_idx_in_117"],
            "add_feat_name": best_rec["add_feat_name"],
            "add_feat_family": best_rec["add_feat_family"],
            "shap_rank_in_117": best_rec["shap_rank_in_117"],
            "rae_mean_bag": best_rec["rae_mean_bag"],
            "rae_median_bag": best_rec["rae_median_bag"],
            "gain_mean_bag_vs_K28": best_gain_mean,
            "gain_median_bag_vs_K28": best_gain_median,
            "passes_bonferroni": bool(best_passes_bonf_mean),
        },
        "best_add_by_median_bag": {
            "add_feat_idx_in_117": best_rec_med["add_feat_idx_in_117"],
            "add_feat_name": best_rec_med["add_feat_name"],
            "add_feat_family": best_rec_med["add_feat_family"],
            "shap_rank_in_117": best_rec_med["shap_rank_in_117"],
            "rae_mean_bag": best_rec_med["rae_mean_bag"],
            "rae_median_bag": best_rec_med["rae_median_bag"],
            "gain_mean_bag_vs_K28": float(best_rec_med["gain_mean_bag_vs_K28"]),
            "gain_median_bag_vs_K28": best_gain_med_only,
            "passes_bonferroni": bool(best_passes_bonf_median),
        },
        "deploy": do_deploy,
        "deploy_csv_path": str(deploy_csv_path) if deploy_csv_path else None,
        "deploy_in_rae_unb": deploy_in_rae_unb,
        "deploy_stats": deploy_stats,
        "verdict": verdict,
        "pre_unblind_clean": True,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
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
        "K_anchor", "K_added", "n_candidates",
        "decision_margin_bonf",
        "rae_anchor_chemprop_aux",
        "k28_baseline_mean_bag_rebuilt",
        "k28_baseline_median_bag_rebuilt",
        "best_add_by_mean_bag", "best_add_by_median_bag",
        "deploy", "deploy_csv_path", "deploy_in_rae_unb",
        "verdict",
    ):
        v = res.get(k)
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"      {kk}: {vv}")
        else:
            print(f"  {k}: {v}")
