"""nb3121 -- Mutual-Info top-18 features (vs SHAP-RFE) + LGBM deep-30 residual,
blended with nb2960 K=18 (RFE-selected) deep-30.

NEW PARADIGM:
    Alternative feature SELECTION CRITERION on the 117-col panel.  The
    cycle-139 finding (`feedback_cycle139_alt_feature_ranking_paradigm_matched`)
    showed LASSO/Boruta/MI/PermImp full-K28 panels under-perform SHAP-K28
    by +0.015 to +0.043 RAE because non-SHAP rankers favor features that
    LGBM splits POORLY (sparse binary FP bits / interaction-only signals).

    But at smaller K (K=18 is currently the sweet spot per nb2960 deep-30 =
    0.4536, lower variance than K28 = 0.4737), the picture may differ:
        - K=18 RFE/SHAP gives a tight 18-dim panel selected to MINIMIZE
          LGBM split error.
        - K=18 MI gives an 18-dim panel selected to MAXIMIZE marginal
          information about residual, independent of model class.
        - The OVERLAP between these two K=18 panels measures how much of
          SHAP's choice is information-content vs LGBM-split-friendliness.
        - The BLEND of MI-K18 + RFE-K18 tests whether MI surfaces a
          model-orthogonal axis a 50/50 blend can exploit.

    GATE: blend mean < 0.4475 -> BETTER (significantly under nb2960 K18 alone
    0.4536, AND under nb3112 K18+iso+Tan-conf calibration ceiling band).

PROTOCOL:
    1. Load nb1102 MI importance over the 117-col panel
       (`nb1102_mi_importance_full117.npy`).  This was computed via
       `mutual_info_regression(X_unb, y_chemprop_residual)` with
       n_neighbors=3, random_state=0 in the existing pipeline.
    2. Pick TOP-18 by MI -> MI_K18_idx (size 18).
    3. Rebuild the 117-col matrix (5-way K-tuned + ChEMBL kNN, identical
       to nb2960) so we can slice X[:, MI_K18_idx] on both unb and te.
    4. Run LGBM residual deep-30 (seeds 3001..3030, same recipe as nb2960)
       on MI_K18 features.  Save MI-K18 OOF (253,) and MI-K18 te (513,).
    5. Compare per-seed bag-mean RAE: MI-K18 vs RFE-K18 (nb2960).
    6. Compute SHAP-RFE-K18 vs MI-K18 overlap (set intersection of
       18-element subsets).
    7. Build 50/50 blend on the OOF: pred_oof = 0.5*MI_K18 + 0.5*RFE_K18.
    8. 5-fold scaffold-CV across 5 kf_seeds {1001..1005}; report grand mean.
    9. GATE on grand mean RAE.

Anchor: chemprop_aux PRE-unblind (`te_chemprop_aux.npy`), verified clean.

References:
    nb2960 K18 (RFE) deep-30 OOF RAE   = 0.4536
    nb2960 K20 (RFE) deep-30 OOF RAE   = 0.4625
    nb2960 K28 (RFE) deep-30 OOF RAE   = 0.4737
    nb1102 MI top-28 5-seed bag        = 0.5041 (loses to SHAP-28 0.4737)
    nb2171 5-anchor pyramid ceiling    = 0.4682
    nb3112 K18+iso+Tan-conf shrink     = (calibration ceiling search)
    chemprop_aux                       = 0.6216

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/te_chemprop_aux.npy
    data/processed/nb1102_mi_importance_full117.npy
    data/processed/nb2960_K18_30seed_oof.npy   (RFE-K18 deep-30 OOF)
    data/processed/nb2960_K18_30seed_te.npy    (RFE-K18 deep-30 te)
    data/processed/nb2960_summary.json         (for RFE-K18 idx + 117-col build)
    + 117-col matrix dependencies (see build_117col_feature_matrix)

Outputs:
    data/processed/nb3121_summary.json
    data/processed/nb3121_pred_oof.npy          (253,) float32 -- 50/50 blend OOF
    data/processed/te_nb3121.npy                (513,) float32 -- 50/50 blend te
    data/processed/nb3121_MI_K18_30seed_oof.npy (253,) float32 -- MI-K18 alone OOF
    data/processed/nb3121_MI_K18_30seed_te.npy  (513,) float32 -- MI-K18 alone te
    submissions/nb3121_MI_K18_blend.csv         (only if BETTER)
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
import lightgbm as lgb
from rdkit import Chem
from rdkit import RDLogger
from sklearn.model_selection import KFold

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3121"
PARENT_TAG = "nb2960_K18"

# -- Anchor + residual params (identical to nb2960) --------------------------
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
RESID_FOLDS = 5
RESID_SEEDS_DEEP = list(range(3001, 3031))    # 30 fresh seeds (match nb2960)

# -- Feature cache paths (identical to nb2960) -------------------------------
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
NB2960_SUMMARY = DATA_PROCESSED / "nb2960_summary.json"

# -- MI importance (precomputed in nb1102) -----------------------------------
MI_IMP_PATH = DATA_PROCESSED / "nb1102_mi_importance_full117.npy"

# -- RFE-K18 cached deep-30 artifacts (nb2960) -------------------------------
RFE_K18_OOF_PATH = DATA_PROCESSED / "nb2960_K18_30seed_oof.npy"
RFE_K18_TE_PATH = DATA_PROCESSED / "nb2960_K18_30seed_te.npy"

# -- Selection ----------------------------------------------------------------
K_PICK = 18

# -- ChEMBL kNN params (identical to nb2960) ---------------------------------
KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

# -- CV eval ------------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# -- Blend recipe -------------------------------------------------------------
W_MI = 0.5
W_RFE = 0.5

# -- Gates --------------------------------------------------------------------
GATE_BETTER = 0.4475

# -- References ---------------------------------------------------------------
NB2960_K18_REF = 0.4536
NB2960_K20_REF = 0.4625
NB2960_K28_REF = 0.4737
NB1102_MI28_REF = 0.5041
NB2171_REF = 0.4682
CHEMPROP_AUX_REF = 0.6216


# ============================================================================
# helpers (lifted verbatim from nb2960)
# ============================================================================

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


def _load_chembl_pool():
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
    if not frames:
        raise FileNotFoundError("No local ChEMBL PXR parquets found")
    pool = pd.concat(frames, ignore_index=True)
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
    )
    agg = agg.rename(columns={"src_first": "src"})
    return agg


def _tanimoto_topk(fp_q, fp_pool, k):
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
    w = np.clip(top_sim, 0.0, 1.0)
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


def _lgbm_params(seed):
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


def _load_mordred_test(n_test_expected):
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(f"Mordred cache missing (run nb1030): {mte_p}")
    X_te_m = np.load(mte_p).astype(np.float32)
    if X_te_m.shape[0] != n_test_expected:
        raise ValueError(f"Mordred test shape mismatch: {X_te_m.shape}")
    X_te_m = np.where(np.isfinite(X_te_m), X_te_m, np.nan).astype(np.float32)
    col_med = np.nanmedian(X_te_m, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X_te_m)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X_te_m[idx_r, idx_c] = col_med[idx_c]
    return X_te_m


def _load_npy_test(path, n_test_expected):
    if not path.exists():
        raise FileNotFoundError(f"missing cache: {path}")
    X = np.load(path)
    if X.shape[0] != n_test_expected:
        raise ValueError(f"shape mismatch {path}: {X.shape}")
    X = X.astype(np.float32)
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X


def _extract_atompair_top_idx_from_nb1484(sum_1484):
    for f in sum_1484["families"]:
        if f["family"] == "AtomPair":
            return np.array(f["top_idx_ranked"], dtype=int)
    raise KeyError("AtomPair entry not found in nb1484_summary.json")


def _extract_best_K_record(sum_dict, records_key, best_K_key="best_K"):
    best_K = int(sum_dict[best_K_key])
    for r in sum_dict[records_key]:
        if int(r["K"]) == best_K:
            return r
    raise KeyError(f"best_K {best_K} not found in {records_key}")


def _residual_cross_fit_one_seed(X, residual, seed):
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _train_full_then_predict_te(X_unb, residual, X_te, seed):
    mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
    mdl.fit(X_unb, residual)
    return mdl.predict(X_te).astype(np.float32)


def build_117col_feature_matrix(te_smiles, n_test):
    """117-col matrix identical to nb2960."""
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

    top_maccs_bit_idx = np.array(sum_1352["top_maccs_bit_indices_ranked"], dtype=int)
    rec_mord = _extract_best_K_record(sum_1523, "per_K_records", best_K_key="best_K")
    top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)
    full_ap_ranked = _extract_atompair_top_idx_from_nb1484(sum_1484)
    K_AP_best = int(sum_1524["best_K"])
    top_ap_bit_idx = full_ap_ranked[:K_AP_best]
    K_Embed_best = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed_col_idx = top_embed_full[:K_Embed_best]
    top_avalon_bit_idx = np.array(sum_1392["top_avalon_bit_indices_ranked"], dtype=int)

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

    pool = _load_chembl_pool()
    test_mols = [standardize(s) for s in te_smiles]
    test_inchikeys = set()
    for m in test_mols:
        ik = _safe_inchikey(m)
        if ik is not None:
            test_inchikeys.add(ik)
    pool = pool[~pool["inchikey"].isin(test_inchikeys)].reset_index(drop=True)
    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep_pool = fp_pool.sum(axis=1) > 0
    if not keep_pool.all():
        pool = pool[keep_pool].reset_index(drop=True)
        fp_pool = fp_pool[keep_pool]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))
    std_test_smiles = []
    for m in test_mols:
        std_test_smiles.append(Chem.MolToSmiles(m) if m is not None else "")
    fp_test = morgan_fp_batch(std_test_smiles)
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )

    X_te_full = np.concatenate(
        [
            X_ap_te_top,
            X_maccs_te_top,
            X_mord_te_top,
            X_emb_te_top,
            X_av_te_top,
            pred_chembl_pec50.reshape(-1, 1).astype(np.float32),
            mean_sim.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    if X_te_full.shape[1] != 117:
        raise ValueError(f"feat_dim {X_te_full.shape[1]} != 117")
    return X_te_full, int(len(pool))


def build_K_30seed_bag(K_label, K_idx_or_X, X_te_full, unb_idx, anchor,
                       residual, te_anchor_513, n_test, n_unb, seeds):
    """Build deep-30 bag-mean OOF (n_unb,) + te (n_test,) for one feature subset.

    Returns:
        bag_oof_unb : (n_unb,) float64   -- mean over seeds of (anchor + resid_oof)
        bag_te_513  : (n_test,) float64  -- mean over seeds of (anchor_te + te_resid)
        per_seed_rae : list[float]        -- per-seed RAE vs y_unb (anchor+resid_oof)
    """
    if isinstance(K_idx_or_X, np.ndarray) and K_idx_or_X.ndim == 2:
        X_te_K = K_idx_or_X.astype(np.float32)
    else:
        idx = np.asarray(K_idx_or_X, dtype=int)
        X_te_K = X_te_full[:, idx].astype(np.float32)
    X_unb_K = X_te_K[unb_idx]
    print(f"   [{K_label}] X_unb_K = {X_unb_K.shape}  X_te_K = {X_te_K.shape}")
    sum_unb = np.zeros(n_unb, dtype=np.float64)
    sum_te = np.zeros(n_test, dtype=np.float64)
    per_seed_rae = []
    for i, s in enumerate(seeds):
        ts = time.time()
        resid_oof = _residual_cross_fit_one_seed(X_unb_K, residual, s)
        pred_unb_s = anchor + resid_oof
        sum_unb += pred_unb_s
        per_seed_rae.append(float(rae(anchor + residual, pred_unb_s)))
        te_resid_s = _train_full_then_predict_te(X_unb_K, residual, X_te_K, s)
        pred_te_s = te_anchor_513 + te_resid_s
        sum_te += pred_te_s
        if (i % 10) == 0 or i == len(seeds) - 1:
            print(f"      [{K_label}] seed={s:4d}  "
                  f"rae={per_seed_rae[-1]:.4f}  "
                  f"wall={time.time()-ts:.2f}s  ({i+1}/{len(seeds)})")
    bag_oof_unb = sum_unb / len(seeds)
    bag_te_513 = sum_te / len(seeds)
    return bag_oof_unb, bag_te_513, per_seed_rae


# ============================================================================
# main
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- MI top-{K_PICK} features (vs SHAP-RFE) + LGBM deep-30 residual")
    print(f"          parent: {PARENT_TAG} (RFE-K18 deep-30 ref = {NB2960_K18_REF:.4f})")
    print(f"          fresh seeds = {RESID_SEEDS_DEEP[0]}..{RESID_SEEDS_DEEP[-1]} "
          f"(n={len(RESID_SEEDS_DEEP)})")
    print(f"          blend: pred = {W_MI}*MI_K{K_PICK} + {W_RFE}*RFE_K{K_PICK}")
    print(f"          gate: blend mean < {GATE_BETTER} -> BETTER, else FAIL")
    print("=" * 78)

    # -- Load truth, anchor, scaffolds ---------------------------------------
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values
                if "name" in te.columns else te["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] chemprop_aux te[unb_idx] RAE = {rae_anchor:.4f} "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor

    # -- Load MI importance + pick top-K_PICK --------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 1: load MI importance + pick top-{K_PICK}")
    print("-" * 78)
    if not MI_IMP_PATH.exists():
        raise FileNotFoundError(
            f"missing MI importance: {MI_IMP_PATH}; run nb1102 first"
        )
    mi_imp = np.load(MI_IMP_PATH).astype(np.float64)
    if mi_imp.shape != (117,):
        raise ValueError(f"MI importance shape {mi_imp.shape} != (117,)")
    MI_K18_idx = np.argsort(-mi_imp)[:K_PICK].astype(int)
    MI_K18_idx_sorted = np.sort(MI_K18_idx)
    print(f"   MI top-{K_PICK} idx (rank order): {MI_K18_idx.tolist()}")
    print(f"   MI top-{K_PICK} idx (sorted)    : {MI_K18_idx_sorted.tolist()}")
    print(f"   MI top-{K_PICK} importance vals : "
          f"min={mi_imp[MI_K18_idx].min():.4f}  "
          f"max={mi_imp[MI_K18_idx].max():.4f}  "
          f"mean={mi_imp[MI_K18_idx].mean():.4f}")

    # -- Load RFE-K18 idx (from nb2960 summary) -----------------------------
    with open(NB2960_SUMMARY) as f:
        nb2960 = json.load(f)
    RFE_K18_idx = np.array(nb2960["K18_idx_in_117col"], dtype=int)
    assert len(RFE_K18_idx) == K_PICK, f"RFE K18 len {len(RFE_K18_idx)} != {K_PICK}"
    RFE_K18_idx_sorted = np.sort(RFE_K18_idx)
    print(f"   RFE K{K_PICK} idx (sorted)      : {RFE_K18_idx_sorted.tolist()}")

    overlap = sorted(set(MI_K18_idx.tolist()) & set(RFE_K18_idx.tolist()))
    mi_only = sorted(set(MI_K18_idx.tolist()) - set(RFE_K18_idx.tolist()))
    rfe_only = sorted(set(RFE_K18_idx.tolist()) - set(MI_K18_idx.tolist()))
    print(f"   MI vs RFE overlap            : {len(overlap)}/{K_PICK} = "
          f"{len(overlap)/K_PICK:.2%}")
    print(f"   overlap idx                  : {overlap}")
    print(f"   MI-only idx                  : {mi_only}")
    print(f"   RFE-only idx                 : {rfe_only}")

    # -- Build 117-col matrix ------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 2: rebuild 117-col 5-way feature matrix")
    print("-" * 78)
    X_te_full, chembl_pool_size = build_117col_feature_matrix(te_smiles, n_test)
    print(f"   X_te_full = {X_te_full.shape}  chembl_pool={chembl_pool_size}")

    # -- MI-K18 deep-30 bag --------------------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 3: MI-K{K_PICK} residual-LGBM deep-30 "
          f"(n_seeds={len(RESID_SEEDS_DEEP)})")
    print("-" * 78)
    MI_oof, MI_te, MI_per_seed_rae = build_K_30seed_bag(
        f"MI_K{K_PICK}", MI_K18_idx, X_te_full, unb_idx, anchor, residual,
        te_anchor_513, n_test, n_unb, RESID_SEEDS_DEEP,
    )
    MI_bag_rae = float(rae(y_unb, MI_oof))
    print(f"   [MI_K{K_PICK}] per-seed RAE  mean = {np.mean(MI_per_seed_rae):.4f}  "
          f"std = {np.std(MI_per_seed_rae, ddof=1):.4f}  "
          f"min={min(MI_per_seed_rae):.4f}  max={max(MI_per_seed_rae):.4f}")
    print(f"   [MI_K{K_PICK}] 30-seed BAG-MEAN RAE = {MI_bag_rae:.4f}")

    # Save MI-K18 alone artifacts
    np.save(DATA_PROCESSED / f"{TAG}_MI_K{K_PICK}_30seed_oof.npy",
            MI_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_MI_K{K_PICK}_30seed_te.npy",
            MI_te.astype(np.float32))

    # -- Load RFE-K18 cached deep-30 -----------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 4: load RFE-K{K_PICK} cached deep-30 (nb2960)")
    print("-" * 78)
    if not RFE_K18_OOF_PATH.exists():
        raise FileNotFoundError(f"missing RFE-K18 oof: {RFE_K18_OOF_PATH}")
    if not RFE_K18_TE_PATH.exists():
        raise FileNotFoundError(f"missing RFE-K18 te:  {RFE_K18_TE_PATH}")
    RFE_oof = np.load(RFE_K18_OOF_PATH).astype(np.float64)
    RFE_te = np.load(RFE_K18_TE_PATH).astype(np.float64)
    if RFE_oof.shape != (n_unb,):
        raise ValueError(f"RFE oof shape {RFE_oof.shape} != ({n_unb},)")
    if RFE_te.shape != (n_test,):
        raise ValueError(f"RFE te shape {RFE_te.shape} != ({n_test},)")
    RFE_bag_rae = float(rae(y_unb, RFE_oof))
    print(f"   [RFE_K{K_PICK}] cached deep-30 bag-mean RAE = {RFE_bag_rae:.4f}  "
          f"(ref {NB2960_K18_REF:.4f})")

    # -- 50/50 blend ----------------------------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 5: 50/50 blend: pred = {W_MI}*MI_K{K_PICK} + {W_RFE}*RFE_K{K_PICK}")
    print("-" * 78)
    pred_oof = W_MI * MI_oof + W_RFE * RFE_oof
    pred_te = W_MI * MI_te + W_RFE * RFE_te
    blend_inplace_rae = float(rae(y_unb, pred_oof))
    print(f"   blend in-sample RAE (no CV) = {blend_inplace_rae:.4f}")
    print(f"   delta vs MI-K{K_PICK}-alone     = {blend_inplace_rae - MI_bag_rae:+.4f}")
    print(f"   delta vs RFE-K{K_PICK}-alone    = {blend_inplace_rae - RFE_bag_rae:+.4f}")

    # -- Diagnostics: residual correlation MI vs RFE -------------------------
    resid_corr = float(np.corrcoef(MI_oof - anchor, RFE_oof - anchor)[0, 1])
    print(f"   MI vs RFE OOF residual Pearson r = {resid_corr:.4f}")
    print(f"   (low r -> orthogonal axes; high r -> redundant)")

    # -- 5-fold scaffold CV over 5 kf_seeds ----------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 6: 5-fold scaffold CV over {len(KF_SEEDS)} kf_seeds {KF_SEEDS}")
    print("-" * 78)
    per_kf_pooled = []
    per_kf_fold_rae = []
    for kf_seed in KF_SEEDS:
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        oof_pooled = np.full(n_unb, np.nan, dtype=np.float64)
        fold_rae = []
        for tr_loc, va_loc in splits:
            oof_pooled[va_loc] = pred_oof[va_loc]
            fold_rae.append(float(rae(y_unb[va_loc], pred_oof[va_loc])))
        if np.isnan(oof_pooled).any():
            raise RuntimeError(f"scaffold splits did not cover all rows (seed={kf_seed})")
        pooled = float(rae(y_unb, oof_pooled))
        per_kf_pooled.append(pooled)
        per_kf_fold_rae.append(fold_rae)
        print(f"   kf_seed={kf_seed}  pooled_RAE={pooled:.4f}  "
              f"per_fold_mean={np.mean(fold_rae):.4f}")

    pooled_arr = np.array(per_kf_pooled, dtype=np.float64)
    pooled_mean = float(pooled_arr.mean())
    pooled_std = float(pooled_arr.std(ddof=1)) if len(pooled_arr) > 1 else 0.0
    pooled_min = float(pooled_arr.min())
    pooled_max = float(pooled_arr.max())
    print(f"\n   grand mean over {len(KF_SEEDS)} kf_seeds: {pooled_mean:.4f} +/- "
          f"{pooled_std:.5f}  (min={pooled_min:.4f}  max={pooled_max:.4f})")

    # -- Gate ----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 7: GATE")
    print("-" * 78)
    if pooled_mean < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE nb3121 (MI-K{K_PICK} + RFE-K{K_PICK} 50/50 blend). Grand "
            f"mean {pooled_mean:.4f} beats gate {GATE_BETTER:.4f}. MI surfaces "
            f"a model-orthogonal axis on top of SHAP-RFE."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT nb3121. Grand mean {pooled_mean:.4f} above gate "
            f"{GATE_BETTER:.4f}. MI ranking adds no orthogonal signal beyond "
            f"RFE-K{K_PICK} alone; confirms cycle-139 finding that non-SHAP "
            f"rankers are paradigm-matched on a fixed LGBM downstream."
        )
    delta_vs_RFE = pooled_mean - RFE_bag_rae
    delta_vs_MI = pooled_mean - MI_bag_rae
    delta_vs_nb2960_ref = pooled_mean - NB2960_K18_REF
    delta_vs_nb2171 = pooled_mean - NB2171_REF
    print(f"   grand mean                   = {pooled_mean:.4f}")
    print(f"   delta vs RFE-K{K_PICK} alone     = {delta_vs_RFE:+.4f}")
    print(f"   delta vs MI-K{K_PICK} alone      = {delta_vs_MI:+.4f}")
    print(f"   delta vs nb2960 K18 ref      = {delta_vs_nb2960_ref:+.4f}")
    print(f"   delta vs nb2171 (0.4682)     = {delta_vs_nb2171:+.4f}")
    print(f"   verdict                      = {verdict}")

    # -- Save ----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 8: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof.astype(np.float32))
    np.save(te_path, pred_te.astype(np.float32))
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    te_unb_in_rae = float(rae(y_unb, pred_te[unb_idx]))
    sub_csv = SUBMISSIONS / f"{TAG}_MI_K{K_PICK}_blend.csv"
    if verdict == "BETTER":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": pred_te.astype(np.float32),
        }).to_csv(sub_csv, index=False)
        print(f"   [save] {sub_csv}")
    else:
        print(f"   [skip] verdict={verdict}; no submission CSV written")

    summary = {
        "tag": TAG,
        "parent_tag": PARENT_TAG,
        "method": f"MI_top{K_PICK}_LGBM_deep30_residual_blended_50_50_with_RFE_K{K_PICK}",
        "anchor_base": "chemprop_aux",
        "anchor_in_rae": rae_anchor,
        "anchor_pre_unblind": True,
        "K_pick": K_PICK,
        "MI_K18_idx_in_117col_rank": MI_K18_idx.tolist(),
        "MI_K18_idx_in_117col_sorted": MI_K18_idx_sorted.tolist(),
        "RFE_K18_idx_in_117col_sorted": RFE_K18_idx_sorted.tolist(),
        "mi_vs_rfe_overlap_n": len(overlap),
        "mi_vs_rfe_overlap_frac": len(overlap) / K_PICK,
        "mi_vs_rfe_overlap_idx": overlap,
        "mi_only_idx": mi_only,
        "rfe_only_idx": rfe_only,
        "mi_importance_path": str(MI_IMP_PATH),
        "rfe_k18_oof_path": str(RFE_K18_OOF_PATH),
        "rfe_k18_te_path": str(RFE_K18_TE_PATH),
        "resid_seeds_deep": RESID_SEEDS_DEEP,
        "resid_folds": RESID_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "n_seeds_deep": len(RESID_SEEDS_DEEP),
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "blend_w_MI": W_MI,
        "blend_w_RFE": W_RFE,
        "MI_per_seed_rae": MI_per_seed_rae,
        "MI_per_seed_mean": float(np.mean(MI_per_seed_rae)),
        "MI_per_seed_std": float(np.std(MI_per_seed_rae, ddof=1)),
        "MI_K18_bagmean_rae": MI_bag_rae,
        "RFE_K18_bagmean_rae": RFE_bag_rae,
        "blend_in_sample_rae": blend_inplace_rae,
        "mi_vs_rfe_oof_resid_pearson_r": resid_corr,
        "per_kf_pooled_rae": per_kf_pooled,
        "per_kf_fold_rae": per_kf_fold_rae,
        "pooled_rae_grand_mean": pooled_mean,
        "pooled_rae_grand_std": pooled_std,
        "pooled_rae_grand_min": pooled_min,
        "pooled_rae_grand_max": pooled_max,
        "mean_rae": pooled_mean,
        "te_unb_in_sample_rae": te_unb_in_rae,
        "te_mean": float(pred_te.mean()),
        "te_std": float(pred_te.std()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict == "BETTER" else None,
        "nb2960_K18_ref": NB2960_K18_REF,
        "nb2960_K20_ref": NB2960_K20_REF,
        "nb2960_K28_ref": NB2960_K28_REF,
        "nb1102_MI28_ref": NB1102_MI28_REF,
        "nb2171_ref": NB2171_REF,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "delta_vs_RFE_K18_alone": delta_vs_RFE,
        "delta_vs_MI_K18_alone": delta_vs_MI,
        "delta_vs_nb2960_K18_ref": delta_vs_nb2960_ref,
        "delta_vs_nb2171": delta_vs_nb2171,
        "gate_better": GATE_BETTER,
        "verdict": verdict,
        "ladder_action": ladder_action,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   MI top-{K_PICK} (rank)          = {MI_K18_idx.tolist()}")
    print(f"   MI vs RFE overlap            = {len(overlap)}/{K_PICK}")
    print(f"   MI-K{K_PICK} deep-30 RAE         = {MI_bag_rae:.4f}")
    print(f"   RFE-K{K_PICK} cached RAE         = {RFE_bag_rae:.4f}  "
          f"(ref {NB2960_K18_REF:.4f})")
    print(f"   blend in-sample RAE          = {blend_inplace_rae:.4f}")
    print(f"   blend grand mean (5 kf)      = {pooled_mean:.4f} +/- {pooled_std:.5f}")
    print(f"   delta vs RFE-K{K_PICK} alone     = {delta_vs_RFE:+.4f}")
    print(f"   delta vs MI-K{K_PICK} alone      = {delta_vs_MI:+.4f}")
    print(f"   te[unb_idx] RAE              = {te_unb_in_rae:.4f}")
    print(f"   verdict                      = {verdict}")
    print(f"   wall                         = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "MI_K18_bagmean_rae",
        "RFE_K18_bagmean_rae",
        "blend_in_sample_rae",
        "pooled_rae_grand_mean",
        "pooled_rae_grand_std",
        "mi_vs_rfe_overlap_n",
        "mi_vs_rfe_oof_resid_pearson_r",
        "delta_vs_RFE_K18_alone",
        "te_unb_in_sample_rae",
        "verdict",
        "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
