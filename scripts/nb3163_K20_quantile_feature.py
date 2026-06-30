"""nb3163 -- K=18 + K=20 quantile membership as 19th feature (cross-K signal).

NEW PARADIGM (vs nb3123 / nb3141 / nb3143):

    All prior quantile-membership scripts derived the q40 cut from the SAME
    K=18 predictor that supplied the 18 continuous features (within-K
    signal).  This script makes the binary indicator CROSS-K:

        18 continuous features  : K=18 slice of the 117-col matrix
        19th binary feature     : (K20_oof  <= q40_K20)  [per-fold threshold]

    Why cross-K is orthogonal:
      - K=18 and K=20 share 16 cols (per nb2604 / nb2280) but differ in
        which 2-4 features extend the slice; K=20 OOF RAE 0.4625 vs K=18
        0.4536 -- the K=20 quantile bit therefore carries a slightly
        different "low-tail-membership" signal because the underlying
        ordering of borderline rows is K-dependent.
      - The downstream LGBM gets a binary regime indicator computed from
        a DIFFERENT predictor than its continuous features, so the split
        the bit triggers is independent of the marginal-value features
        used to grow the rest of the tree (cross-K orthogonality, same
        spirit as nb730 multi-source orthogonality).

PROTOCOL:
    1. 117-col 5-way feature matrix identical to nb3123 / nb3141
       (nb1352/1392/1484/1523/1524/1541 summaries + ChEMBL PXR kNN).
    2. K=18 idx from nb2604_summary, K=20 idx from nb2280_summary.
    3. q40 source: K20 deep-30 OOF (nb2960_K20_30seed_oof.npy) on unblind
       253 and K20 deep-30 te (nb2960_K20_30seed_te.npy) on 513.
    4. HONEST 5-fold scaffold CV (per-fold q40 threshold from K20 OOF):
         for each kf_seed in {1141..1145}:
             splits = scaffold_kfold_indices(unb_scaffolds, n_splits=5,
                      shuffle=True, seed=kf_seed)
             oof = full(n_unb, nan)
             for (tr_loc, va_loc) in splits:
                 q_thr = quantile(K20_oof[tr_loc], 0.40)        # fold-honest
                 feat_19 = (K20_oof <= q_thr).astype(float32)   # cross-K bit
                 X_K19 = [X_K18, feat_19]
                 for s in {3001..3030}:                          # deep-30
                     mdl = LGBM(seed=s).fit(X_K19[tr_loc], resid[tr_loc])
                     bag_va += mdl.predict(X_K19[va_loc]) / 30
                 oof[va_loc] = anchor[va_loc] + bag_va
             pooled_rae[kf_seed] = rae(y_unb, oof)
         mean +/- std across 5 kf_seeds is the HONEST number.
    5. DEPLOY: feat_19_te uses GLOBAL q40 on 253 K20 OOF for the 513 te
       (boundary identical to nb3141 deploy convention).

GATE:
    honest mean RAE < 0.4475 -> "BETTER"
    else                     -> "FAIL"

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/te_chemprop_aux.npy
    data/processed/nb2604_summary.json    (K=18 idx in 117-col)
    data/processed/nb2280_summary.json    (K=20 idx in 117-col)
    data/processed/nb2960_K20_30seed_oof.npy   (K20 OOF -- q40 source)
    data/processed/nb2960_K20_30seed_te.npy    (K20 te  -- deploy q40 source)
    + nb1352/1392/1484/1523/1524/1541 summaries
    + te_atompair.npy, te_maccs.npy, te_chemprop_embed_300.npy, te_avalon512.npy
    + C:/pxr_artifacts/nb1030/X_mordred_test.npy
    + data/external/chembl_pxr_CHEMBL3401.parquet (+ siblings)

Outputs:
    data/processed/nb3163_summary.json
    data/processed/nb3163_pred_oof.npy    (253,) float32
    data/processed/te_nb3163.npy          (513,) float32
    submissions/nb3163_K20_quantile_feature.csv (BETTER only)
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

TAG = "nb3163"
PARENT_TAG = "nb3141"   # closest honest-mode parent (above-q40 binary on K18)

# -- Anchor / residual params (IDENTICAL to nb3123 / nb3141 / nb2960) --------
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
RESID_FOLDS = 5
RESID_SEEDS_DEEP = list(range(3001, 3031))     # deep-30 residual seeds

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

NB2604_SUMMARY = DATA_PROCESSED / "nb2604_summary.json"   # K=18 idx source
NB2280_SUMMARY = DATA_PROCESSED / "nb2280_summary.json"   # K=20 idx source

# -- K20 deep-30 OOF / te (q40 feature source -- CROSS-K vs K18 features) ----
K20_OOF_PATH = DATA_PROCESSED / "nb2960_K20_30seed_oof.npy"
K20_TE_PATH = DATA_PROCESSED / "nb2960_K20_30seed_te.npy"
Q_CUT = 0.40

# -- ChEMBL kNN params (identical to nb3123 / nb3141 / nb2604 / nb2960) ------
KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

# -- HONEST cross-fit eval ---------------------------------------------------
N_FOLDS = 5
KF_SEEDS = [1141, 1142, 1143, 1144, 1145]      # 5 fresh kf_seeds

# -- Gates -------------------------------------------------------------------
GATE_BETTER = 0.4475

# -- References --------------------------------------------------------------
NB3123_BAG_MEAN = 0.4389       # nb3123 below-q40 bag-mean (DEPLOY OPTIMISM)
NB3123_PER_SEED_MEAN = 0.4677  # nb3123 below-q40 honest-leaning per-seed mean
NB2960_K18_REF = 0.4536
NB2960_K20_REF = 0.4625        # K20 deep-30 OOF RAE (this script's q40 source)
NB2171_REF = 0.4682
CHEMPROP_AUX_REF = 0.6216


# ============================================================================
# helpers (lifted verbatim from nb3141)
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


def build_117col_feature_matrix(te_smiles, n_test):
    """117-col matrix identical to nb3123 / nb3141 / nb2604 / nb2960."""
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


# ============================================================================
# core honest CV with CROSS-K (K20) q40 binary feature
# ============================================================================

def fold_honest_one_kf_seed(
    X_unb_K18, k20_oof_unb, anchor, residual, y_unb, unb_scaffolds,
    kf_seed, n_folds, seeds, q_cut,
):
    """One kf_seed honest 5-fold scaffold CV pass with CROSS-K q40 binary
    feature derived from K20 OOF (vs K18 features).

    For each fold:
        q_thr  = quantile(K20_oof[tr_loc], q_cut)        # fold-honest, K20-source
        feat19 = (K20_oof <= q_thr).astype(float32)      # CROSS-K binary bit
        X_K19  = [X_K18, feat19]
        30-seed LGBM bag fit on tr_loc, predict va_loc
        oof[va_loc] = anchor[va_loc] + bag_va
    """
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=n_folds, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof = np.full(n_unb, np.nan, dtype=np.float64)
    per_fold_rae = []
    feat19_share_tr_per_fold = []
    q_thr_per_fold = []
    for fi, (tr_loc, va_loc) in enumerate(splits):
        # fold-honest q40 threshold from K20 OOF training rows only
        q_thr = float(np.quantile(k20_oof_unb[tr_loc], q_cut))
        feat19 = (k20_oof_unb <= q_thr).astype(np.float32)
        feat19_share_tr = float(feat19[tr_loc].mean())
        feat19_share_tr_per_fold.append(feat19_share_tr)
        q_thr_per_fold.append(q_thr)

        # X_K19 stack: K18 features + K20 q40 binary indicator (CROSS-K)
        X_unb_K19 = np.concatenate(
            [X_unb_K18, feat19.reshape(-1, 1)], axis=1
        ).astype(np.float32)

        X_tr = X_unb_K19[tr_loc]
        X_va = X_unb_K19[va_loc]
        resid_tr = residual[tr_loc]
        bag_va = np.zeros(len(va_loc), dtype=np.float64)
        for s in seeds:
            mdl = lgb.LGBMRegressor(**_lgbm_params(s))
            mdl.fit(X_tr, resid_tr)
            bag_va += mdl.predict(X_va)
        bag_va /= len(seeds)
        oof[va_loc] = anchor[va_loc] + bag_va
        per_fold_rae.append(float(rae(y_unb[va_loc], oof[va_loc])))
    if np.isnan(oof).any():
        raise RuntimeError(f"scaffold splits did not cover all rows (kf_seed={kf_seed})")
    pooled = float(rae(y_unb, oof))
    return pooled, oof, per_fold_rae, feat19_share_tr_per_fold, q_thr_per_fold


def build_deploy_te(X_unb_K19, X_te_K19, anchor, residual, te_anchor_513,
                    n_test, n_unb, seeds):
    """Deploy: refit on ALL 253 unblind + 30-seed bag te (residual).
    NOT used in honest gate -- diagnostic artifact only for stacking.
    """
    sum_unb = np.zeros(n_unb, dtype=np.float64)
    sum_te = np.zeros(n_test, dtype=np.float64)
    for s in seeds:
        # internal random KFold OOF for bag_oof (diagnostic, not gate)
        kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=s)
        oof_s = np.full(n_unb, np.nan, dtype=np.float64)
        for tr_loc, va_loc in kf.split(np.arange(n_unb)):
            mdl = lgb.LGBMRegressor(**_lgbm_params(s))
            mdl.fit(X_unb_K19[tr_loc], residual[tr_loc])
            oof_s[va_loc] = mdl.predict(X_unb_K19[va_loc])
        pred_unb_s = anchor + oof_s
        sum_unb += pred_unb_s
        # full refit -> te
        mdl_full = lgb.LGBMRegressor(**_lgbm_params(s))
        mdl_full.fit(X_unb_K19, residual)
        te_resid_s = mdl_full.predict(X_te_K19).astype(np.float64)
        sum_te += te_anchor_513 + te_resid_s
    bag_oof_unb = sum_unb / len(seeds)
    bag_te_513 = sum_te / len(seeds)
    return bag_oof_unb, bag_te_513


# ============================================================================
# main
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- K=18 + K=20 q40 BINARY MEMBERSHIP as 19th feature (CROSS-K)")
    print(f"          parent: {PARENT_TAG}  (above-q40 on K18)")
    print(f"          features: 18 cols from K=18 + 1 binary bit (K20_oof <= q40_K20)")
    print(f"          q_cut    = {Q_CUT}  (per-fold from K20 OOF training rows)")
    print(f"          kf_seeds (5)        : {KF_SEEDS}")
    print(f"          residual seeds (30) : {RESID_SEEDS_DEEP[0]}..{RESID_SEEDS_DEEP[-1]}")
    print(f"          gate    = honest mean < {GATE_BETTER:.4f} -> BETTER else FAIL")
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

    # -- Load K18 idx (continuous features), K20 idx (diagnostic only) -------
    with open(NB2604_SUMMARY) as f:
        nb2604 = json.load(f)
    K18_idx = np.array(nb2604["k18_idx_in_117col"], dtype=int)
    assert len(K18_idx) == 18, f"K18 len {len(K18_idx)} != 18"
    print(f"[load] K=18 idx (n={len(K18_idx)}): {K18_idx.tolist()}")

    with open(NB2280_SUMMARY) as f:
        nb2280 = json.load(f)
    K20_idx = np.array(nb2280["K20_rfe_surviving_idx_in_117"], dtype=int)
    assert len(K20_idx) == 20, f"K20 len {len(K20_idx)} != 20"
    overlap_cols = sorted(set(K18_idx.tolist()) & set(K20_idx.tolist()))
    print(f"[load] K=20 idx (n={len(K20_idx)}): {K20_idx.tolist()}")
    print(f"[diag] K18 K20 overlap = {len(overlap_cols)} cols")

    # -- Load K20 deep-30 OOF / te (q40 source -- the CROSS-K signal) --------
    k20_oof = np.load(K20_OOF_PATH).astype(np.float64)
    k20_te = np.load(K20_TE_PATH).astype(np.float64)
    if k20_oof.shape != (n_unb,):
        raise ValueError(f"K20 oof shape {k20_oof.shape} != ({n_unb},)")
    if k20_te.shape != (n_test,):
        raise ValueError(f"K20 te shape {k20_te.shape} != ({n_test},)")
    k20_full_rae = float(rae(y_unb, k20_oof))
    print(f"[load] K20 deep-30 OOF RAE = {k20_full_rae:.4f} (ref {NB2960_K20_REF:.4f})")

    # Global q40 (diagnostic + deploy boundary)
    q40_global_unb = float(np.quantile(k20_oof, Q_CUT))
    q40_global_te = float(np.quantile(k20_te, Q_CUT))
    feat19_share_unb_global = float((k20_oof <= q40_global_unb).mean())
    feat19_share_te_global = float((k20_te <= q40_global_te).mean())
    print(f"[diag] global q40 (K20)  unb={q40_global_unb:.4f}  te={q40_global_te:.4f}")
    print(f"[diag] (K20 <= q40) share unb={feat19_share_unb_global:.3f}  "
          f"te={feat19_share_te_global:.3f}  (target ~{Q_CUT:.2f})")

    # -- Build 117-col matrix, slice K=18 ------------------------------------
    print("\n" + "-" * 78)
    print("STEP 1: build 117-col 5-way feature matrix and slice K=18")
    print("-" * 78)
    X_te_full, chembl_pool_size = build_117col_feature_matrix(te_smiles, n_test)
    print(f"   X_te_full = {X_te_full.shape}  chembl_pool={chembl_pool_size}")
    X_te_K18 = X_te_full[:, K18_idx].astype(np.float32)
    X_unb_K18 = X_te_K18[unb_idx]
    assert X_unb_K18.shape == (n_unb, 18)
    print(f"   X_unb_K18 = {X_unb_K18.shape}  X_te_K18 = {X_te_K18.shape}")

    # -- HONEST cross-fit over 5 kf_seeds ------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 2: HONEST 5-fold scaffold CV over {len(KF_SEEDS)} kf_seeds "
          f"(per-fold q40 threshold from K20 OOF training rows; CROSS-K signal)")
    print("-" * 78)
    per_kf_pooled = []
    per_kf_fold_rae = []
    per_kf_feat19_share = []
    per_kf_q_thr = []
    per_kf_oof = []
    for kf_seed in KF_SEEDS:
        ts = time.time()
        pooled, oof_s, fold_rae, feat19_share_tr, q_thr_pf = fold_honest_one_kf_seed(
            X_unb_K18, k20_oof, anchor, residual, y_unb, unb_scaffolds,
            kf_seed=kf_seed, n_folds=N_FOLDS,
            seeds=RESID_SEEDS_DEEP, q_cut=Q_CUT,
        )
        per_kf_pooled.append(pooled)
        per_kf_fold_rae.append(fold_rae)
        per_kf_feat19_share.append(feat19_share_tr)
        per_kf_q_thr.append(q_thr_pf)
        per_kf_oof.append(oof_s)
        wall = time.time() - ts
        print(f"   kf_seed={kf_seed}  pooled_RAE={pooled:.4f}  "
              f"folds_mean={np.mean(fold_rae):.4f}  "
              f"feat19_share_tr_mean={np.mean(feat19_share_tr):.3f}  "
              f"q_thr_mean={np.mean(q_thr_pf):.4f}  "
              f"wall={wall:.1f}s")
    pooled_arr = np.array(per_kf_pooled, dtype=np.float64)
    honest_mean = float(pooled_arr.mean())
    honest_std = float(pooled_arr.std(ddof=1)) if len(pooled_arr) > 1 else 0.0
    honest_min = float(pooled_arr.min())
    honest_max = float(pooled_arr.max())
    print(f"\n   HONEST mean +/- std over {len(KF_SEEDS)} kf_seeds = "
          f"{honest_mean:.4f} +/- {honest_std:.5f}  "
          f"(min={honest_min:.4f}  max={honest_max:.4f})")

    # OOF artifact = simple average over kf_seeds
    bag_oof = np.mean(np.stack(per_kf_oof, axis=0), axis=0).astype(np.float64)

    # -- Deploy-style refit for te artifact (NOT used in gate) ----------------
    print("\n" + "-" * 78)
    print(f"STEP 3: deploy-style refit for te_{TAG}.npy artifact "
          f"(global q40 on K20; NOT used in gate)")
    print("-" * 78)
    feat19_unb_global = (k20_oof <= q40_global_unb).astype(np.float32)
    feat19_te_global = (k20_te <= q40_global_te).astype(np.float32)
    X_unb_K19 = np.concatenate(
        [X_unb_K18, feat19_unb_global.reshape(-1, 1)], axis=1
    ).astype(np.float32)
    X_te_K19 = np.concatenate(
        [X_te_K18, feat19_te_global.reshape(-1, 1)], axis=1
    ).astype(np.float32)
    deploy_oof, deploy_te = build_deploy_te(
        X_unb_K19, X_te_K19, anchor, residual, te_anchor_513,
        n_test, n_unb, RESID_SEEDS_DEEP,
    )
    deploy_oof_rae = float(rae(y_unb, deploy_oof))
    print(f"   deploy 30-seed bag OOF RAE = {deploy_oof_rae:.4f} "
          f"(DEPLOY OPTIMISM, NOT gate)")

    # -- Gate ----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 4: GATE")
    print("-" * 78)
    if honest_mean < GATE_BETTER:
        verdict = "BETTER"
    else:
        verdict = "FAIL"
    delta_vs_k18 = honest_mean - NB2960_K18_REF
    delta_vs_k20 = honest_mean - NB2960_K20_REF
    delta_vs_nb3123_perseed = honest_mean - NB3123_PER_SEED_MEAN
    delta_vs_nb3123_bagmean = honest_mean - NB3123_BAG_MEAN
    delta_vs_nb2171 = honest_mean - NB2171_REF
    print(f"   honest mean (5 kf_seeds)   = {honest_mean:.4f}")
    print(f"   delta vs nb2960 K18 ref    = {delta_vs_k18:+.4f}  "
          f"({NB2960_K18_REF:.4f})")
    print(f"   delta vs nb2960 K20 ref    = {delta_vs_k20:+.4f}  "
          f"({NB2960_K20_REF:.4f})")
    print(f"   delta vs nb3123 per-seed   = {delta_vs_nb3123_perseed:+.4f}  "
          f"({NB3123_PER_SEED_MEAN:.4f})")
    print(f"   delta vs nb3123 bag-mean   = {delta_vs_nb3123_bagmean:+.4f}  "
          f"({NB3123_BAG_MEAN:.4f})")
    print(f"   delta vs nb2171 ceiling    = {delta_vs_nb2171:+.4f}  "
          f"({NB2171_REF:.4f})")
    print(f"   verdict                    = {verdict}")

    # -- Save artifacts ------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 5: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, bag_oof.astype(np.float32))
    np.save(te_path, deploy_te.astype(np.float32))
    print(f"   [save] {oof_path}  (honest cross-fit OOF, 253,)")
    print(f"   [save] {te_path}   (deploy refit, 513,)")

    te_clipped = np.clip(deploy_te, 3.0, 9.0).astype(np.float32)
    te_unb_in_rae = float(rae(y_unb, te_clipped[unb_idx]))
    sub_csv = SUBMISSIONS / f"{TAG}_K20_quantile_feature.csv"
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
        "method": "K18_feats_plus_K20_q40_binary_19th_feat_CROSS_K_honest_5kfseeds",
        "feature_set": "K18_cols + (K20_oof <= q40_K20) binary bit",
        "q_source": "K20 deep-30 OOF (cross-K vs K18 continuous features)",
        "anchor_base": "chemprop_aux",
        "anchor_in_rae": rae_anchor,
        "anchor_pre_unblind": True,
        "K_orig_continuous": 18,
        "K_with_q20_bit": 19,
        "K18_idx_in_117col": K18_idx.tolist(),
        "K20_idx_in_117col": K20_idx.tolist(),
        "K18_K20_overlap_cols": overlap_cols,
        "n_K18_K20_overlap": len(overlap_cols),
        "q_cut": Q_CUT,
        "q40_global_unb_K20": q40_global_unb,
        "q40_global_te_K20": q40_global_te,
        "feat19_share_unb_global": feat19_share_unb_global,
        "feat19_share_te_global": feat19_share_te_global,
        "per_kf_feat19_share_tr": per_kf_feat19_share,
        "per_kf_q_thr": per_kf_q_thr,
        "k20_full_oof_rae": k20_full_rae,
        "k20_deep30_ref": NB2960_K20_REF,
        "k18_deep30_ref": NB2960_K18_REF,
        "kf_seeds": KF_SEEDS,
        "n_kf_seeds": len(KF_SEEDS),
        "n_folds": N_FOLDS,
        "resid_seeds_deep": RESID_SEEDS_DEEP,
        "n_seeds_deep": len(RESID_SEEDS_DEEP),
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "per_kf_pooled_rae": per_kf_pooled,
        "per_kf_fold_rae": per_kf_fold_rae,
        "honest_mean": honest_mean,
        "honest_std": honest_std,
        "honest_min": honest_min,
        "honest_max": honest_max,
        "mean_rae": honest_mean,
        "deploy_oof_rae_NOT_GATE": deploy_oof_rae,
        "te_unb_in_sample_rae": te_unb_in_rae,
        "te_mean": float(deploy_te.mean()),
        "te_std": float(deploy_te.std()),
        "te_clipped_mean": float(te_clipped.mean()),
        "te_clipped_std": float(te_clipped.std()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict == "BETTER" else None,
        "nb2960_K18_ref": NB2960_K18_REF,
        "nb2960_K20_ref": NB2960_K20_REF,
        "nb3123_bagmean_ref": NB3123_BAG_MEAN,
        "nb3123_perseed_ref": NB3123_PER_SEED_MEAN,
        "nb2171_ref": NB2171_REF,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "delta_vs_nb2960_K18": delta_vs_k18,
        "delta_vs_nb2960_K20": delta_vs_k20,
        "delta_vs_nb3123_perseed": delta_vs_nb3123_perseed,
        "delta_vs_nb3123_bagmean": delta_vs_nb3123_bagmean,
        "delta_vs_nb2171": delta_vs_nb2171,
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
    print(f"   honest mean (5 kf_seeds)   = {honest_mean:.4f} +/- {honest_std:.5f}")
    print(f"   honest min / max           = {honest_min:.4f} / {honest_max:.4f}")
    print(f"   deploy OOF (NOT GATE)      = {deploy_oof_rae:.4f}")
    print(f"   nb2960 K18 ref             = {NB2960_K18_REF:.4f}")
    print(f"   nb2960 K20 ref (q source)  = {NB2960_K20_REF:.4f}")
    print(f"   delta vs K18               = {delta_vs_k18:+.4f}")
    print(f"   delta vs K20               = {delta_vs_k20:+.4f}")
    print(f"   te[unb_idx] in-sample RAE  = {te_unb_in_rae:.4f}")
    print(f"   GATE (< {GATE_BETTER:.4f})       = {verdict}")
    print(f"   wall                       = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "honest_mean", "honest_std", "honest_min", "honest_max",
        "deploy_oof_rae_NOT_GATE",
        "delta_vs_nb2960_K18", "delta_vs_nb2960_K20",
        "delta_vs_nb3123_perseed",
        "te_unb_in_sample_rae",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
    print(f"  q40_global_unb_K20: {res.get('q40_global_unb_K20')}")
    print(f"  feat19_share_unb_global: {res.get('feat19_share_unb_global')}")
    print(f"  n_K18_K20_overlap: {res.get('n_K18_K20_overlap')}")
