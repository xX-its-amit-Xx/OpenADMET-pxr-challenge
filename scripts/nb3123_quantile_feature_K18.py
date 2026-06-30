"""nb3123 -- Quantile-membership AS A FEATURE in K=18.

NEW PARADIGM (vs nb3073/nb3080):
    nb3073/nb3080 used quantile-conditional HARD-SPLIT BLEND on top of two
    independently fit anchors (K18, K19).  Here instead we treat the
    quantile membership as a binary INPUT FEATURE for a single LGBM fit:

        feat_19 = is_low_quantile = 1 if K18_pred <= q40 else 0
        X_K19 = [K18_orig_18_features, q_low]   -- 19 cols
        LGBM_deep30(X_K19) on chemprop_aux residual

    Hypothesis: giving the model the q-low/high regime indicator inside
    the tree splits is a STRONGER inductive bias than post-hoc blending
    two independently-trained predictors with that same indicator.

PROTOCOL:
    1. Load K=18 indices from nb2604_summary (18 cols in 117-col matrix).
    2. Build 117-col 5-way feature matrix (verbatim from nb2960).
    3. Load nb2960 K18 deep-30 OOF (253,) + te (513,) AS GIVEN.
       Use it to compute q40 threshold on its UNBLIND OOF distribution.
       feat_19_unb = (K18_pred_oof <= q40).astype(float32)
       feat_19_te  = (K18_pred_te  <= q40).astype(float32)   -- same q40
    4. X_unb_K19 = stack[X_unb_K18, feat_19_unb]  -- (253, 19)
       X_te_K19  = stack[X_te_K18,  feat_19_te ]  -- (513, 19)
    5. Residual-LGBM deep-30 (seeds {3001..3030}, 5-fold KFold) on
       (anchor=chemprop_aux, residual=y_unb-anchor) with X_unb_K19.
    6. 5-fold scaffold CV pooled RAE over 5 kf_seeds {1001..1005}.

GATE:
    mean_rae <  0.4475 -> "BETTER"
    else                -> "FAIL"

References:
    nb2960 K18 deep-30 bag-mean RAE          = 0.4536   (this is the
                                                          baseline to beat)
    nb3073 5-seed quantile-conditional blend = 0.4470   (different paradigm)
    nb3080 15-seed verify of nb3073          = pending  (different paradigm)
    GATE                                      = 0.4475
    chemprop_aux                              = 0.6216

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/te_chemprop_aux.npy
    data/processed/nb2604_summary.json (K=18 idx in 117-col)
    data/processed/nb2960_K18_30seed_oof.npy (gives q40 ref)
    data/processed/nb2960_K18_30seed_te.npy
    + nb1352/nb1392/nb1484/nb1523/nb1524/nb1541 summaries
    + te_atompair.npy, te_maccs.npy, te_chemprop_embed_300.npy, te_avalon512.npy
    + C:/pxr_artifacts/nb1030/X_mordred_test.npy
    + data/external/chembl_pxr_CHEMBL3401.parquet (+ siblings)

Outputs:
    data/processed/nb3123_summary.json
    data/processed/nb3123_pred_oof.npy     (253,) float32
    data/processed/te_nb3123.npy           (513,) float32
    submissions/nb3123_quantile_feature_K18.csv (BETTER only)
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

TAG = "nb3123"
PARENT_TAG = "nb2960_K18"

# -- Anchor + residual params (IDENTICAL to nb2960) --------------------------
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
RESID_FOLDS = 5
RESID_SEEDS_DEEP = list(range(3001, 3031))    # 30 fresh seeds {3001..3030}

# -- Feature cache paths -----------------------------------------------------
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

NB2604_SUMMARY = DATA_PROCESSED / "nb2604_summary.json"

# -- K18 deep-30 OOF / te (for q40 threshold + 19th feature) -----------------
K18_OOF_PATH = DATA_PROCESSED / "nb2960_K18_30seed_oof.npy"
K18_TE_PATH = DATA_PROCESSED / "nb2960_K18_30seed_te.npy"
Q_CUT = 0.40   # q40 threshold matches nb3073 best combo

# -- ChEMBL kNN params (identical to nb2604/nb2960) --------------------------
KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

# -- CV eval -----------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# -- Gates -------------------------------------------------------------------
GATE_BETTER = 0.4475

# -- References --------------------------------------------------------------
NB2960_K18_REF = 0.4536
NB3073_BLEND_REF = 0.4470
NB2171_REF = 0.4682
CHEMPROP_AUX_REF = 0.6216


# ============================================================================
# helpers (lifted from nb2960)
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
    """117-col matrix identical to nb2604/nb2960."""
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


def build_deep30_bag(X_unb, X_te, anchor, residual, te_anchor_513,
                     n_test, n_unb, seeds):
    """Build deep-30 bag-mean OOF (n_unb,) + te (n_test,) for one feature set.

    Returns:
        bag_oof_unb : (n_unb,) float64
        bag_te_513  : (n_test,) float64
        per_seed_rae : list[float]
    """
    print(f"   X_unb = {X_unb.shape}  X_te = {X_te.shape}")
    sum_unb = np.zeros(n_unb, dtype=np.float64)
    sum_te = np.zeros(n_test, dtype=np.float64)
    per_seed_rae = []
    for i, s in enumerate(seeds):
        ts = time.time()
        resid_oof = _residual_cross_fit_one_seed(X_unb, residual, s)
        pred_unb_s = anchor + resid_oof
        sum_unb += pred_unb_s
        per_seed_rae.append(float(rae(anchor + residual, pred_unb_s)))
        te_resid_s = _train_full_then_predict_te(X_unb, residual, X_te, s)
        pred_te_s = te_anchor_513 + te_resid_s
        sum_te += pred_te_s
        if (i % 10) == 0 or i == len(seeds) - 1:
            print(f"      seed={s:4d}  rae={per_seed_rae[-1]:.4f}  "
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
    print(f"{TAG} -- quantile-membership AS 19th FEATURE in K=18 LGBM")
    print(f"          parent: {PARENT_TAG} (deep-30 K18 RAE {NB2960_K18_REF:.4f})")
    print(f"          q_cut  = {Q_CUT} (q40 threshold from K18 OOF)")
    print(f"          seeds  = {RESID_SEEDS_DEEP[0]}..{RESID_SEEDS_DEEP[-1]} "
          f"(n={len(RESID_SEEDS_DEEP)})")
    print(f"          gate   = mean < {GATE_BETTER:.4f} -> BETTER else FAIL")
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
    assert n_unb == 253, f"expected 253 unblind, got {n_unb}"
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

    # -- Load K18 idx + K18 deep-30 OOF/te (q40 source) -----------------------
    print("\n" + "-" * 78)
    print("STEP 1: load K=18 idx + K18 deep-30 OOF/te (for q40 threshold)")
    print("-" * 78)
    with open(NB2604_SUMMARY) as f:
        nb2604 = json.load(f)
    K18_idx = np.array(nb2604["k18_idx_in_117col"], dtype=int)
    assert len(K18_idx) == 18, f"K18 len {len(K18_idx)} != 18"
    print(f"   K=18 idx (n={len(K18_idx)}): {K18_idx.tolist()}")

    k18_oof = np.load(K18_OOF_PATH).astype(np.float64)
    k18_te = np.load(K18_TE_PATH).astype(np.float64)
    if k18_oof.shape != (n_unb,):
        raise ValueError(f"K18 oof shape {k18_oof.shape} != ({n_unb},)")
    if k18_te.shape != (n_test,):
        raise ValueError(f"K18 te shape {k18_te.shape} != ({n_test},)")
    k18_full_rae = float(rae(y_unb, k18_oof))
    print(f"   K18 deep-30 OOF RAE = {k18_full_rae:.4f} (ref {NB2960_K18_REF:.4f})")

    # q40 threshold from K18 UNBLIND OOF (proxy for distribution observed at
    # both train and deploy: K18 OOF on 253 vs K18 te on 513)
    q40_unb = float(np.quantile(k18_oof, Q_CUT))
    q40_te = float(np.quantile(k18_te, Q_CUT))
    feat19_unb = (k18_oof <= q40_unb).astype(np.float32)
    feat19_te = (k18_te <= q40_te).astype(np.float32)
    print(f"   q40 threshold (K18 OOF on 253)   = {q40_unb:.4f}")
    print(f"   q40 threshold (K18 te  on 513)   = {q40_te:.4f}")
    print(f"   q_low share (unb 253) = {feat19_unb.mean():.3f}  "
          f"(target ~{Q_CUT:.2f})")
    print(f"   q_low share (te  513) = {feat19_te.mean():.3f}  "
          f"(target ~{Q_CUT:.2f})")

    # -- Build 117-col matrix ------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 2: rebuild 117-col 5-way feature matrix")
    print("-" * 78)
    X_te_full, chembl_pool_size = build_117col_feature_matrix(te_smiles, n_test)
    print(f"   X_te_full = {X_te_full.shape}  chembl_pool={chembl_pool_size}")
    X_te_K18 = X_te_full[:, K18_idx].astype(np.float32)
    assert X_te_K18.shape[1] == 18, f"K18 dim {X_te_K18.shape[1]} != 18"

    # Stack the q_low feature as the 19th column
    X_te_K19 = np.concatenate(
        [X_te_K18, feat19_te.reshape(-1, 1)], axis=1
    ).astype(np.float32)
    X_unb_K19 = X_te_K19[unb_idx]
    assert X_te_K19.shape[1] == 19, f"K19 dim {X_te_K19.shape[1]} != 19"
    print(f"   X_te_K19 = {X_te_K19.shape}  X_unb_K19 = {X_unb_K19.shape}")

    # -- Deep-30 bag --------------------------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 3: residual-LGBM deep-30 with {len(RESID_SEEDS_DEEP)} fresh seeds")
    print("-" * 78)
    bag_oof, bag_te, per_rae = build_deep30_bag(
        X_unb_K19, X_te_K19, anchor, residual, te_anchor_513,
        n_test, n_unb, RESID_SEEDS_DEEP,
    )
    bag_mean_rae = float(rae(y_unb, bag_oof))
    per_rae_mean = float(np.mean(per_rae))
    per_rae_std = float(np.std(per_rae, ddof=1))
    print(f"   per-seed RAE mean = {per_rae_mean:.4f}  "
          f"std = {per_rae_std:.4f}  "
          f"min={min(per_rae):.4f}  max={max(per_rae):.4f}")
    print(f"   30-seed BAG-MEAN OOF RAE = {bag_mean_rae:.4f}")

    # -- 5-fold scaffold CV pooled over 5 kf_seeds ----------------------------
    print("\n" + "-" * 78)
    print(f"STEP 4: 5-fold scaffold CV over {len(KF_SEEDS)} kf_seeds {KF_SEEDS}")
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
            oof_pooled[va_loc] = bag_oof[va_loc]
            fold_rae.append(float(rae(y_unb[va_loc], bag_oof[va_loc])))
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
    print("STEP 5: GATE")
    print("-" * 78)
    if pooled_mean < GATE_BETTER:
        verdict = "BETTER"
    else:
        verdict = "FAIL"
    delta_vs_k18 = pooled_mean - NB2960_K18_REF
    delta_vs_nb3073 = pooled_mean - NB3073_BLEND_REF
    print(f"   grand mean             = {pooled_mean:.4f}")
    print(f"   delta vs nb2960 K18    = {delta_vs_k18:+.4f}  "
          f"(ref {NB2960_K18_REF:.4f})")
    print(f"   delta vs nb3073 blend  = {delta_vs_nb3073:+.4f}  "
          f"(ref {NB3073_BLEND_REF:.4f})")
    print(f"   verdict                = {verdict}")

    # -- Save artifacts ------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 6: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, bag_oof.astype(np.float32))
    np.save(te_path, bag_te.astype(np.float32))
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    te_clipped = np.clip(bag_te, 3.0, 9.0).astype(np.float32)
    te_unb_in_rae = float(rae(y_unb, te_clipped[unb_idx]))
    sub_csv = SUBMISSIONS / f"{TAG}_quantile_feature_K18.csv"
    if verdict == "BETTER":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": te_clipped,
        }).to_csv(sub_csv, index=False)
        print(f"   [save] {sub_csv}")
    else:
        print(f"   [skip] verdict={verdict}; no submission CSV written")

    summary = {
        "tag": TAG,
        "parent_tag": PARENT_TAG,
        "method": "quantile_membership_as_19th_feature_in_K18_deep30",
        "anchor_base": "chemprop_aux",
        "anchor_in_rae": rae_anchor,
        "anchor_pre_unblind": True,
        "K_orig": 18,
        "K_with_qlow": 19,
        "K18_idx_in_117col": K18_idx.tolist(),
        "q_cut": Q_CUT,
        "q40_thr_unb": q40_unb,
        "q40_thr_te": q40_te,
        "q_low_share_unb": float(feat19_unb.mean()),
        "q_low_share_te": float(feat19_te.mean()),
        "k18_full_oof_rae": k18_full_rae,
        "k18_deep30_ref": NB2960_K18_REF,
        "resid_seeds_deep": RESID_SEEDS_DEEP,
        "resid_folds": RESID_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "n_seeds_deep": len(RESID_SEEDS_DEEP),
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "per_seed_rae": [round(float(v), 4) for v in per_rae],
        "per_seed_rae_mean": per_rae_mean,
        "per_seed_rae_std": per_rae_std,
        "bag_oof_rae": bag_mean_rae,
        "per_kf_pooled_rae": per_kf_pooled,
        "per_kf_fold_rae": per_kf_fold_rae,
        "pooled_rae_grand_mean": pooled_mean,
        "pooled_rae_grand_std": pooled_std,
        "pooled_rae_grand_min": pooled_min,
        "pooled_rae_grand_max": pooled_max,
        "mean_rae": pooled_mean,
        "te_unb_in_sample_rae": te_unb_in_rae,
        "te_mean": float(bag_te.mean()),
        "te_std": float(bag_te.std()),
        "te_clipped_mean": float(te_clipped.mean()),
        "te_clipped_std": float(te_clipped.std()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict == "BETTER" else None,
        "nb2960_K18_ref": NB2960_K18_REF,
        "nb3073_blend_ref": NB3073_BLEND_REF,
        "nb2171_ref": NB2171_REF,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "delta_vs_nb2960_K18": delta_vs_k18,
        "delta_vs_nb3073_blend": delta_vs_nb3073,
        "gate_better": GATE_BETTER,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   per-seed RAE mean      = {per_rae_mean:.4f} +/- {per_rae_std:.4f}")
    print(f"   30-seed BAG-MEAN OOF   = {bag_mean_rae:.4f}")
    print(f"   pooled grand mean (CV) = {pooled_mean:.4f} +/- {pooled_std:.5f}")
    print(f"   nb2960 K18 ref         = {NB2960_K18_REF:.4f}")
    print(f"   delta vs nb2960 K18    = {delta_vs_k18:+.4f}")
    print(f"   delta vs nb3073 blend  = {delta_vs_nb3073:+.4f}")
    print(f"   te[unb_idx] RAE        = {te_unb_in_rae:.4f}")
    print(f"   verdict                = {verdict}")
    print(f"   wall                   = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "pooled_rae_grand_mean",
        "pooled_rae_grand_std",
        "bag_oof_rae",
        "delta_vs_nb2960_K18",
        "delta_vs_nb3073_blend",
        "te_unb_in_sample_rae",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
    print(f"  q40_thr_unb: {res.get('q40_thr_unb')}")
    print(f"  q_low_share_unb: {res.get('q_low_share_unb')}")
    print(f"  q_low_share_te: {res.get('q_low_share_te')}")
