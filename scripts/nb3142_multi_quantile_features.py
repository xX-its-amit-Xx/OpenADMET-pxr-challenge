"""nb3142 -- K=18 + 4 quantile binary features (q20, q40, q60, q80) -> K=22.

NEW PARADIGM (vs nb3123 single q40 binary):
    nb3123 added ONE binary indicator (q_low = K18_pred <= q40) as the 19th
    feature, lifting deploy bag-mean OOF RAE from 0.4536 -> 0.4389 (later
    nb3140 honest cross-fit revealed bag-mean optimism).  Here we generalize
    that single binary to FOUR mutually-exclusive bin indicators that
    encode QUANTILE MEMBERSHIP of K18_pred:

        q20_bin   = 1 if K18_pred in [q20, q40)   else 0
        q40_bin   = 1 if K18_pred in [q40, q60)   else 0
        q60_bin   = 1 if K18_pred in [q60, q80)   else 0
        q80_bin   = 1 if K18_pred >= q80          else 0
        (rows below q20 carry all-zeros across the four indicators)

    Hypothesis: encoding the WHOLE quantile-bin landscape (not just one
    low/high split) gives LGBM more axis-aligned split capacity at the
    boundaries where variance compression is largest (per the rare-scaffold
    quantile-compression failure mode).

PROTOCOL (HONEST cross-fit, following nb3140 cycle-167 standard):
    1. Reproduce nb3123 inputs verbatim (117-col 5-way feature matrix +
       K=18 idx from nb2604, chemprop_aux anchor on residual).
    2. K18_pred source = nb2960_K18_30seed_oof.npy (deep-30 OOF).
    3. For each kf_seed in {1141..1145} (5 fresh kf_seeds):
         splits = scaffold_kfold_indices(unb_scaffolds, n_splits=5,
                  shuffle=True, seed=kf_seed)
         oof = full(n_unb, nan)
         for (tr_loc, va_loc) in splits:
             # compute q20/q40/q60/q80 thresholds on TRAINING-FOLD k18_oof
             q20_thr, q40_thr, q60_thr, q80_thr = quantile(
                 k18_oof[tr_loc], [0.2, 0.4, 0.6, 0.8])
             # build the 4 bin indicators on the full 253-row k18_oof
             q20_bin = (q20_thr <= k18_oof < q40_thr)
             q40_bin = (q40_thr <= k18_oof < q60_thr)
             q60_bin = (q60_thr <= k18_oof < q80_thr)
             q80_bin = (k18_oof >= q80_thr)
             X_K22 = stack[X_K18, q20_bin, q40_bin, q60_bin, q80_bin]
             # train 30-seed LGBM bag on TRAINING rows only
             for s in {3001..3030}:
                 mdl = LGBM(seed=s).fit(X_K22[tr_loc], residual[tr_loc])
                 bag_va += mdl.predict(X_K22[va_loc]) / 30
             oof[va_loc] = anchor[va_loc] + bag_va
         pooled_rae[kf_seed] = rae(y_unb, oof)
    4. mean +/- std across 5 kf_seeds = HONEST cross-fit.
    5. te prediction (513,): for deploy refit, compute global quantile
       thresholds from ALL 253 k18_oof, build q20/q40/q60/q80 bins on
       k18_te (513,), stack into X_te_K22, train 30-seed bag on full
       X_unb_K22 and predict X_te_K22; bag-mean over 30 seeds.

GATE:
    honest mean RAE < 0.4475 -> "BETTER"
    else                     -> "FAIL"

References:
    nb2960 K18 deep-30 bag-mean RAE          = 0.4536
    nb3123 K18 + q_low bag-mean OOF          = 0.4389 (deploy optimism)
    nb3123 K18 + q_low per-seed mean         = 0.4677 (honest-leaning)
    nb2171 5-anchor pyramid                  = 0.4682 (post-hoc ceiling)
    GATE                                      = 0.4475
    chemprop_aux                              = 0.6216

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/te_chemprop_aux.npy
    data/processed/nb2604_summary.json (K=18 idx in 117-col)
    data/processed/nb2960_K18_30seed_oof.npy (K18 OOF -> q bin source)
    data/processed/nb2960_K18_30seed_te.npy   (K18 te -> q bin source for 513)
    + nb1352/1392/1484/1523/1524/1541 summaries
    + te_atompair.npy, te_maccs.npy, te_chemprop_embed_300.npy, te_avalon512.npy
    + C:/pxr_artifacts/nb1030/X_mordred_test.npy
    + data/external/chembl_pxr_CHEMBL3401.parquet + siblings

Outputs:
    data/processed/nb3142_summary.json
    data/processed/nb3142_pred_oof.npy     (253,) float32  -- last kf_seed oof
    data/processed/te_nb3142.npy           (513,) float32  -- deploy 30-bag
    submissions/nb3142_multi_quantile_features.csv (BETTER only)
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

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3142"
PARENT_TAG = "nb3123"

# -- Anchor + residual params (IDENTICAL to nb3123/nb2960) -------------------
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
RESID_FOLDS = 5
RESID_SEEDS_DEEP = list(range(3001, 3031))   # 30 fresh seeds {3001..3030}

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

# -- K18 deep-30 OOF / te (for quantile bin features) ------------------------
K18_OOF_PATH = DATA_PROCESSED / "nb2960_K18_30seed_oof.npy"
K18_TE_PATH = DATA_PROCESSED / "nb2960_K18_30seed_te.npy"
Q_CUTS = (0.20, 0.40, 0.60, 0.80)            # four quantile boundaries

# -- ChEMBL kNN params (identical to nb3123/nb2604/nb2960) ------------------
KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

# -- HONEST cross-fit eval ---------------------------------------------------
N_FOLDS = 5
KF_SEEDS = [1141, 1142, 1143, 1144, 1145]    # 5 fresh kf_seeds per protocol

# -- Gates -------------------------------------------------------------------
GATE_BETTER = 0.4475

# -- References --------------------------------------------------------------
NB2960_K18_REF = 0.4536
NB3123_BAG_MEAN = 0.4389
NB3123_PER_SEED_MEAN = 0.4677
NB2171_REF = 0.4682
CHEMPROP_AUX_REF = 0.6216


# ============================================================================
# helpers (lifted verbatim from nb3123/nb3140)
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
    """117-col matrix identical to nb3123/nb2604/nb2960."""
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


def _build_4_quantile_bins(k18_pred, q_thrs):
    """Build the 4 binary quantile-bin features.

    q_thrs = (q20, q40, q60, q80) thresholds (from k18_oof[tr_loc] in CV,
    or from full k18_oof / k18_te at deploy).

        q20_bin = 1 if k18_pred in [q20, q40)
        q40_bin = 1 if k18_pred in [q40, q60)
        q60_bin = 1 if k18_pred in [q60, q80)
        q80_bin = 1 if k18_pred >= q80
    rows below q20 have all-zeros across the four indicators.

    Returns (n, 4) float32 in column order [q20_bin, q40_bin, q60_bin, q80_bin].
    """
    q20, q40, q60, q80 = q_thrs
    q20_bin = ((k18_pred >= q20) & (k18_pred < q40)).astype(np.float32)
    q40_bin = ((k18_pred >= q40) & (k18_pred < q60)).astype(np.float32)
    q60_bin = ((k18_pred >= q60) & (k18_pred < q80)).astype(np.float32)
    q80_bin = (k18_pred >= q80).astype(np.float32)
    return np.stack([q20_bin, q40_bin, q60_bin, q80_bin], axis=1)


def fold_honest_one_kf_seed(
    X_unb_K18, k18_oof_unb, anchor, residual, y_unb, unb_scaffolds,
    kf_seed, n_folds, seeds, q_cuts,
):
    """Run one (kf_seed) honest 5-fold scaffold CV pass with K=22 features.

    For each fold:
        - Compute (q20, q40, q60, q80) thresholds on TRAINING-FOLD k18_oof.
        - Build 4 binary bin features on the full 253-row k18_oof using
          those training-fold thresholds.
        - Stack X_unb_K22 = [X_unb_K18, q20_bin, q40_bin, q60_bin, q80_bin].
        - Inside fold: train 30 LGBM seeds on TRAINING rows only,
          predict held-out validation rows, average -> bag_va.
        - oof[va_loc] = anchor[va_loc] + bag_va.

    Returns:
        pooled_rae : float
        oof        : (n_unb,) float64
        per_fold_rae : list[float]
    """
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=n_folds, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof = np.full(n_unb, np.nan, dtype=np.float64)
    per_fold_rae = []
    for fi, (tr_loc, va_loc) in enumerate(splits):
        # --- compute quantile thresholds on TRAINING-FOLD k18_oof ---------
        q_thrs = tuple(float(q) for q in np.quantile(k18_oof_unb[tr_loc], q_cuts))
        # build 4 bin features for the full 253 rows using these thrs
        bins = _build_4_quantile_bins(k18_oof_unb, q_thrs)
        # X_K22 stack
        X_unb_K22 = np.concatenate([X_unb_K18, bins], axis=1).astype(np.float32)

        # --- 30-seed bag of LGBM predictions on validation ----------------
        X_tr = X_unb_K22[tr_loc]
        X_va = X_unb_K22[va_loc]
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
    return pooled, oof, per_fold_rae


def deploy_30seed_te(X_unb_K22, X_te_K22, residual, te_anchor_513, seeds):
    """Deploy refit: train 30-seed LGBM bag on full unblind X_unb_K22,
    predict X_te_K22, average -> bag_te (513,).
    """
    n_te = X_te_K22.shape[0]
    sum_te = np.zeros(n_te, dtype=np.float64)
    for s in seeds:
        mdl = lgb.LGBMRegressor(**_lgbm_params(s))
        mdl.fit(X_unb_K22, residual)
        sum_te += mdl.predict(X_te_K22)
    bag_te_resid = sum_te / len(seeds)
    return (te_anchor_513 + bag_te_resid).astype(np.float64)


# ============================================================================
# main
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- K=18 + 4 quantile binary features (q20,q40,q60,q80) -> K=22")
    print(f"          parent: {PARENT_TAG} (K=18 + 1 q_low binary)")
    print(f"          q_cuts = {Q_CUTS}")
    print(f"          kf_seeds = {KF_SEEDS} ({len(KF_SEEDS)} fresh seeds)")
    print(f"          resid_seeds = {RESID_SEEDS_DEEP[0]}..{RESID_SEEDS_DEEP[-1]} "
          f"(n={len(RESID_SEEDS_DEEP)})")
    print(f"          GATE: honest mean < {GATE_BETTER:.4f} -> BETTER else FAIL")
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

    # -- Load K18 idx + K18 OOF/te (for q-bin sources) -----------------------
    with open(NB2604_SUMMARY) as f:
        nb2604 = json.load(f)
    K18_idx = np.array(nb2604["k18_idx_in_117col"], dtype=int)
    assert len(K18_idx) == 18, f"K18 len {len(K18_idx)} != 18"
    print(f"[load] K=18 idx: {K18_idx.tolist()}")

    k18_oof = np.load(K18_OOF_PATH).astype(np.float64)
    k18_te = np.load(K18_TE_PATH).astype(np.float64)
    if k18_oof.shape != (n_unb,):
        raise ValueError(f"K18 oof shape {k18_oof.shape} != ({n_unb},)")
    if k18_te.shape != (n_test,):
        raise ValueError(f"K18 te shape {k18_te.shape} != ({n_test},)")
    k18_full_rae = float(rae(y_unb, k18_oof))
    print(f"[load] K18 deep-30 OOF RAE = {k18_full_rae:.4f} "
          f"(ref {NB2960_K18_REF:.4f})")

    # -- Build 117-col matrix, slice K=18 ------------------------------------
    print("\n" + "-" * 78)
    print("STEP 1: build 117-col matrix and slice K=18 on unblind 253 and te 513")
    print("-" * 78)
    X_te_full, chembl_pool_size = build_117col_feature_matrix(te_smiles, n_test)
    print(f"   X_te_full = {X_te_full.shape}  chembl_pool={chembl_pool_size}")
    X_te_K18 = X_te_full[:, K18_idx].astype(np.float32)
    X_unb_K18 = X_te_K18[unb_idx]
    assert X_unb_K18.shape == (n_unb, 18)
    print(f"   X_unb_K18 = {X_unb_K18.shape}  X_te_K18 = {X_te_K18.shape}")

    # -- Diagnostic: print global quantile bins on unblind k18_oof ------------
    q_thrs_global_unb = tuple(
        float(q) for q in np.quantile(k18_oof, Q_CUTS)
    )
    q_thrs_global_te = tuple(
        float(q) for q in np.quantile(k18_te, Q_CUTS)
    )
    print(f"   global q20/q40/q60/q80 on unb k18_oof = "
          f"{[f'{q:.4f}' for q in q_thrs_global_unb]}")
    print(f"   global q20/q40/q60/q80 on te k18_te    = "
          f"{[f'{q:.4f}' for q in q_thrs_global_te]}")

    # -- Honest cross-fit over 5 kf_seeds (training-fold thresholds) ---------
    print("\n" + "-" * 78)
    print(f"STEP 2: honest 5-fold SCAFFOLD CV over {len(KF_SEEDS)} kf_seeds "
          f"(q thresholds from TRAINING-fold k18_oof inside each fold)")
    print("-" * 78)
    per_kf_pooled = []
    per_kf_fold_rae = []
    last_oof = None
    for kf_seed in KF_SEEDS:
        ts = time.time()
        pooled, oof_this, fold_rae = fold_honest_one_kf_seed(
            X_unb_K18, k18_oof, anchor, residual, y_unb, unb_scaffolds,
            kf_seed=kf_seed, n_folds=N_FOLDS,
            seeds=RESID_SEEDS_DEEP, q_cuts=Q_CUTS,
        )
        per_kf_pooled.append(pooled)
        per_kf_fold_rae.append(fold_rae)
        last_oof = oof_this
        wall = time.time() - ts
        print(f"   kf_seed={kf_seed}  pooled_RAE={pooled:.4f}  "
              f"folds_mean={np.mean(fold_rae):.4f}  wall={wall:.1f}s")

    pooled_arr = np.array(per_kf_pooled, dtype=np.float64)
    pooled_mean = float(pooled_arr.mean())
    pooled_std = float(pooled_arr.std(ddof=1)) if len(pooled_arr) > 1 else 0.0
    pooled_median = float(np.median(pooled_arr))
    pooled_min = float(pooled_arr.min())
    pooled_max = float(pooled_arr.max())
    print(f"\n   HONEST mean +/- std over {len(KF_SEEDS)} kf_seeds: "
          f"{pooled_mean:.4f} +/- {pooled_std:.5f}  "
          f"(median={pooled_median:.4f}  min={pooled_min:.4f}  max={pooled_max:.4f})")

    # -- Gate ----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 3: GATE")
    print("-" * 78)
    if pooled_mean < GATE_BETTER:
        verdict = "BETTER"
    else:
        verdict = "FAIL"
    delta_vs_k18 = pooled_mean - NB2960_K18_REF
    delta_vs_nb3123_perseed = pooled_mean - NB3123_PER_SEED_MEAN
    delta_vs_nb3123_bagmean = pooled_mean - NB3123_BAG_MEAN
    print(f"   honest mean (5 kf_seeds) = {pooled_mean:.4f}")
    print(f"   delta vs nb2960 K18      = {delta_vs_k18:+.4f}  "
          f"(ref {NB2960_K18_REF:.4f})")
    print(f"   delta vs nb3123 per-seed = {delta_vs_nb3123_perseed:+.4f}  "
          f"(ref {NB3123_PER_SEED_MEAN:.4f})")
    print(f"   delta vs nb3123 bag-mean = {delta_vs_nb3123_bagmean:+.4f}  "
          f"(ref {NB3123_BAG_MEAN:.4f}, deploy optimism)")
    print(f"   verdict                  = {verdict}")

    # -- Deploy te (513,) with global q thresholds ---------------------------
    print("\n" + "-" * 78)
    print("STEP 4: deploy refit -- 30-seed bag te (513,) with global q thresholds")
    print("-" * 78)
    # build K22 unblind with global thresholds (training is full 253 here)
    bins_unb_global = _build_4_quantile_bins(k18_oof, q_thrs_global_unb)
    X_unb_K22 = np.concatenate([X_unb_K18, bins_unb_global], axis=1).astype(np.float32)
    bins_te_global = _build_4_quantile_bins(k18_te, q_thrs_global_te)
    X_te_K22 = np.concatenate([X_te_K18, bins_te_global], axis=1).astype(np.float32)
    assert X_unb_K22.shape == (n_unb, 22), f"K22 unb dim {X_unb_K22.shape}"
    assert X_te_K22.shape == (n_test, 22), f"K22 te  dim {X_te_K22.shape}"
    print(f"   X_unb_K22 = {X_unb_K22.shape}  X_te_K22 = {X_te_K22.shape}")
    te_pred = deploy_30seed_te(
        X_unb_K22, X_te_K22, residual, te_anchor_513, RESID_SEEDS_DEEP,
    )
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"   te[unb_idx] in-sample RAE = {te_unb_in_rae:.4f}")

    # -- Save artifacts ------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 5: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, last_oof.astype(np.float32))
    np.save(te_path, te_pred.astype(np.float32))
    print(f"   [save] {oof_path}  (last kf_seed={KF_SEEDS[-1]} honest oof)")
    print(f"   [save] {te_path}   (30-seed deploy bag)")

    te_clipped = np.clip(te_pred, 3.0, 9.0).astype(np.float32)
    sub_csv = SUBMISSIONS / f"{TAG}_multi_quantile_features.csv"
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
        "method": ("K18_plus_4_quantile_bin_features_deep30_residual_LGBM_"
                   "honest_scaffold_CV"),
        "anchor_base": "chemprop_aux",
        "anchor_pre_unblind": True,
        "anchor_in_rae": rae_anchor,
        "K_orig": 18,
        "K_with_4qbins": 22,
        "K18_idx_in_117col": K18_idx.tolist(),
        "q_cuts": list(Q_CUTS),
        "q_thrs_global_unb": list(q_thrs_global_unb),
        "q_thrs_global_te": list(q_thrs_global_te),
        "k18_full_oof_rae": k18_full_rae,
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
        "pooled_rae_grand_mean": pooled_mean,
        "pooled_rae_grand_std": pooled_std,
        "pooled_rae_grand_median": pooled_median,
        "pooled_rae_grand_min": pooled_min,
        "pooled_rae_grand_max": pooled_max,
        "honest_mean_rae": pooled_mean,
        "honest_std_rae": pooled_std,
        "mean_rae": pooled_mean,
        "te_unb_in_sample_rae": te_unb_in_rae,
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "te_clipped_mean": float(te_clipped.mean()),
        "te_clipped_std": float(te_clipped.std()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict == "BETTER" else None,
        "nb2960_K18_ref": NB2960_K18_REF,
        "nb3123_bagmean_ref": NB3123_BAG_MEAN,
        "nb3123_perseed_ref": NB3123_PER_SEED_MEAN,
        "nb2171_ref": NB2171_REF,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "delta_vs_nb2960_K18": delta_vs_k18,
        "delta_vs_nb3123_perseed": delta_vs_nb3123_perseed,
        "delta_vs_nb3123_bagmean": delta_vs_nb3123_bagmean,
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
    print(f"   K=18 + 4 quantile-bin features (K=22)")
    print(f"   honest mean (5 kf_seeds) = {pooled_mean:.4f} +/- {pooled_std:.5f}")
    print(f"   nb2960 K18 ref           = {NB2960_K18_REF:.4f}")
    print(f"   nb3123 per-seed ref      = {NB3123_PER_SEED_MEAN:.4f}")
    print(f"   delta vs nb2960 K18      = {delta_vs_k18:+.4f}")
    print(f"   delta vs nb3123 per-seed = {delta_vs_nb3123_perseed:+.4f}")
    print(f"   te[unb_idx] in-sample    = {te_unb_in_rae:.4f}")
    print(f"   GATE (< {GATE_BETTER:.4f})        = {verdict}")
    print(f"   wall                     = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "pooled_rae_grand_mean",
        "pooled_rae_grand_std",
        "delta_vs_nb2960_K18",
        "delta_vs_nb3123_perseed",
        "te_unb_in_sample_rae",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
    print(f"  q_thrs_global_unb: {res.get('q_thrs_global_unb')}")
    print(f"  q_thrs_global_te: {res.get('q_thrs_global_te')}")
