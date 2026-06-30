"""nb3140 -- HONEST cross-fit verification of nb3123 (K=18 + q_low 19th feat).

CONTEXT (cycle 213 nb2640 lesson, cycle 167 nb2171 protocol):
    nb3123 reported bag-mean OOF RAE 0.4389 BUT per-seed RAE mean 0.4677.
    The bag-mean number is DEPLOY-STYLE OPTIMISM, not honest cross-fit:
    inside nb3123 the LGBM was trained with random KFold splits (per resid
    seed), the predictions were averaged across 30 seeds to form `bag_oof`,
    and then nb3123 "pooled" `bag_oof` over scaffold kf_seeds without
    actually re-fitting under scaffold-aware splits.  Because the bag_oof
    values are FIXED before scaffold-pooling, every kf_seed returns the same
    pooled RAE -- it's a no-op pool, not a cross-fit.

    The cycle-213 / cycle-167 standard for an HONEST cross-fit number is:
        for each kf_seed:
            5-fold SCAFFOLD CV (split on Bemis-Murcko scaffolds, NOT random)
            inside each fold: train 30-seed bag on TRAINING SCAFFOLDS ONLY
            predict the held-out scaffold validation fold
            gather predictions across folds (every row predicted once)
            pooled RAE per kf_seed
        mean +/- std across 15 kf_seeds = HONEST cross-fit

PROTOCOL (this script):
    1. Reproduce nb3123 inputs verbatim:
         - 117-col 5-way feature matrix (nb1352/1392/1484/1523/1524/1541
           summaries + ChEMBL PXR kNN)
         - K=18 idx from nb2604_summary
         - anchor=chemprop_aux on residual
         - K=19 = K18 + q_low binary indicator
    2. For each kf_seed in {1141..1155} (15 fresh kf_seeds):
         splits = scaffold_kfold_indices(unb_scaffolds, n_splits=5,
                  shuffle=True, seed=kf_seed)
         oof = full(n_unb, nan)
         for (tr_loc, va_loc) in splits:
             X_tr = X_unb_K19[tr_loc]; X_va = X_unb_K19[va_loc]
             resid_tr = residual[tr_loc]
             # build 30-seed bag of LGBM predictions on the VALIDATION fold
             pred_va_bag = zeros(n_va)
             for s in {3001..3030}:
                 mdl = LGBM(seed=s).fit(X_tr, resid_tr)
                 pred_va_bag += mdl.predict(X_va) / 30
             # anchor + residual prediction
             oof[va_loc] = anchor[va_loc] + pred_va_bag
         pooled_rae[kf_seed] = rae(y_unb, oof)
    3. mean +/- std across 15 kf_seeds is the HONEST number.

NOTE on q_low feature honesty:
    The 19th feature is `q_low = (K18_pred <= q40)`.  nb3123 sourced
    K18_pred from `nb2960_K18_30seed_oof.npy` -- itself an OOF prediction
    (each i predicted without i in training).  Using the GLOBAL quantile
    threshold q40 over all 253 K18 OOF preds inside a CV fold leaks
    distribution-of-other-rows info into the training set; the strictly
    honest alternative is to recompute q40 from training-fold K18 OOF
    only.  Both modes are run and reported:
        feat_qlow_global : threshold = q40 over all 253 K18 OOF
        feat_qlow_fold   : threshold = q40 over training-fold K18 OOF only
    Gate uses the FOLD-HONEST mean for decisioning.

GATE:
    fold-honest mean RAE < 0.4475 -> "VERIFIED"
    else                          -> "BAG_MEAN_OPTIMISM_TRAP"

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/te_chemprop_aux.npy
    data/processed/nb2604_summary.json (K=18 idx)
    data/processed/nb2960_K18_30seed_oof.npy (K18 OOF -- q_low source)
    data/processed/nb1352/1392/1484/1523/1524/1541 summaries
    data/processed/te_atompair.npy, te_maccs.npy, te_chemprop_embed_300.npy,
                       te_avalon512.npy
    C:/pxr_artifacts/nb1030/X_mordred_test.npy
    data/external/chembl_pxr_CHEMBL3401.parquet + siblings

Outputs:
    data/processed/nb3140_summary.json
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
from pxr.paths import DATA_PROCESSED

TAG = "nb3140"
PARENT_TAG = "nb3123"

# -- Identical anchor / residual setup to nb3123 -----------------------------
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
K18_OOF_PATH = DATA_PROCESSED / "nb2960_K18_30seed_oof.npy"
Q_CUT = 0.40

# -- ChEMBL kNN (identical to nb3123/nb2604/nb2960) --------------------------
KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

# -- HONEST cross-fit eval ---------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1141, 1156))           # 15 fresh kf_seeds {1141..1155}

# -- Gates --------------------------------------------------------------------
GATE_VERIFIED = 0.4475

# -- References --------------------------------------------------------------
NB3123_BAG_MEAN = 0.4389       # nb3123 reported (DEPLOY OPTIMISM)
NB3123_PER_SEED_MEAN = 0.4677  # nb3123 honest-leaning per-seed mean
NB2960_K18_REF = 0.4536
CHEMPROP_AUX_REF = 0.6216


# ============================================================================
# helpers (lifted from nb3123 -- identical 117-col matrix build)
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


def fold_honest_one_kf_seed(
    X_unb_K18, k18_oof_unb, anchor, residual, y_unb, unb_scaffolds,
    kf_seed, n_folds, seeds, q_cut, qlow_mode,
):
    """Run one (kf_seed) honest 5-fold scaffold CV pass.

    For each fold:
        - Build q_low feature:
            'global' mode -> threshold is q_cut quantile over ALL 253 k18_oof
                             (matches nb3123 wiring; uses non-fold info)
            'fold'   mode -> threshold is q_cut quantile over TRAINING-FOLD
                             k18_oof only (strict honest)
        - Stack X_unb_K19 = [X_unb_K18, q_low]
        - Inside fold: train 30 LGBM seeds on TRAINING rows only,
          predict held-out validation rows, average -> bag_va
        - oof[va_loc] = anchor[va_loc] + bag_va

    Returns:
        pooled_rae : float
        oof        : (n_unb,) float64
        per_fold_rae : list[float]  -- one per fold
    """
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=n_folds, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof = np.full(n_unb, np.nan, dtype=np.float64)
    per_fold_rae = []
    for fi, (tr_loc, va_loc) in enumerate(splits):
        # --- build q_low feature honestly per fold -------------------------
        if qlow_mode == "global":
            q_thr = float(np.quantile(k18_oof_unb, q_cut))
        elif qlow_mode == "fold":
            q_thr = float(np.quantile(k18_oof_unb[tr_loc], q_cut))
        else:
            raise ValueError(f"unknown qlow_mode {qlow_mode}")
        feat_qlow = (k18_oof_unb <= q_thr).astype(np.float32)

        # X_K19 stack
        X_unb_K19 = np.concatenate(
            [X_unb_K18, feat_qlow.reshape(-1, 1)], axis=1
        ).astype(np.float32)

        # --- 30-seed bag of LGBM predictions on validation ----------------
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
    return pooled, oof, per_fold_rae


# ============================================================================
# main
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- HONEST cross-fit verification of {PARENT_TAG}")
    print(f"          nb3123 reported bag-mean OOF RAE = {NB3123_BAG_MEAN:.4f} "
          f"(DEPLOY OPTIMISM)")
    print(f"          nb3123 per-seed RAE mean        = {NB3123_PER_SEED_MEAN:.4f}")
    print(f"          15 fresh kf_seeds: {KF_SEEDS[0]}..{KF_SEEDS[-1]}")
    print(f"          30-seed bag inside each fold     = "
          f"{RESID_SEEDS_DEEP[0]}..{RESID_SEEDS_DEEP[-1]}")
    print(f"          GATE: fold-honest mean < {GATE_VERIFIED:.4f} -> VERIFIED")
    print(f"                else                       -> BAG_MEAN_OPTIMISM_TRAP")
    print("=" * 78)

    # -- Load truth, anchor, scaffolds ---------------------------------------
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
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

    # -- Load K18 idx + K18 OOF (for q_low source) ----------------------------
    with open(NB2604_SUMMARY) as f:
        nb2604 = json.load(f)
    K18_idx = np.array(nb2604["k18_idx_in_117col"], dtype=int)
    assert len(K18_idx) == 18, f"K18 len {len(K18_idx)} != 18"
    print(f"[load] K=18 idx: {K18_idx.tolist()}")

    k18_oof = np.load(K18_OOF_PATH).astype(np.float64)
    if k18_oof.shape != (n_unb,):
        raise ValueError(f"K18 oof shape {k18_oof.shape} != ({n_unb},)")
    k18_full_rae = float(rae(y_unb, k18_oof))
    print(f"[load] K18 deep-30 OOF RAE = {k18_full_rae:.4f} "
          f"(ref {NB2960_K18_REF:.4f})")

    # -- Build 117-col matrix, slice K=18 ------------------------------------
    print("\n" + "-" * 78)
    print("STEP 1: build 117-col matrix and slice K=18 on unblind 253")
    print("-" * 78)
    X_te_full, chembl_pool_size = build_117col_feature_matrix(te_smiles, n_test)
    print(f"   X_te_full = {X_te_full.shape}  chembl_pool={chembl_pool_size}")
    X_te_K18 = X_te_full[:, K18_idx].astype(np.float32)
    X_unb_K18 = X_te_K18[unb_idx]
    assert X_unb_K18.shape == (n_unb, 18)
    print(f"   X_unb_K18 = {X_unb_K18.shape}")

    # -- Honest cross-fit over 15 kf_seeds, both q_low modes -----------------
    print("\n" + "-" * 78)
    print(f"STEP 2: honest 5-fold SCAFFOLD CV over {len(KF_SEEDS)} kf_seeds "
          f"(qlow_mode = 'fold' AND 'global')")
    print("-" * 78)

    results = {}
    for qmode in ("fold", "global"):
        print(f"\n--- qlow_mode = '{qmode}' ---")
        per_kf_pooled = []
        per_kf_fold_rae = []
        for kf_seed in KF_SEEDS:
            ts = time.time()
            pooled, _oof, fold_rae = fold_honest_one_kf_seed(
                X_unb_K18, k18_oof, anchor, residual, y_unb, unb_scaffolds,
                kf_seed=kf_seed, n_folds=N_FOLDS,
                seeds=RESID_SEEDS_DEEP, q_cut=Q_CUT, qlow_mode=qmode,
            )
            per_kf_pooled.append(pooled)
            per_kf_fold_rae.append(fold_rae)
            wall = time.time() - ts
            print(f"   [{qmode}] kf_seed={kf_seed}  pooled_RAE={pooled:.4f}  "
                  f"folds_mean={np.mean(fold_rae):.4f}  wall={wall:.1f}s")
        arr = np.array(per_kf_pooled, dtype=np.float64)
        mean_ = float(arr.mean())
        std_ = float(arr.std(ddof=1))
        median_ = float(np.median(arr))
        min_ = float(arr.min())
        max_ = float(arr.max())
        results[qmode] = {
            "per_kf_pooled_rae": per_kf_pooled,
            "per_kf_fold_rae": per_kf_fold_rae,
            "mean": mean_,
            "std": std_,
            "median": median_,
            "min": min_,
            "max": max_,
        }
        print(f"   [{qmode}] HONEST mean +/- std = {mean_:.4f} +/- {std_:.5f}  "
              f"min={min_:.4f}  max={max_:.4f}")

    # -- Gate (on fold-honest mode) ------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 3: GATE (fold-honest mode)")
    print("-" * 78)
    fold_honest_mean = results["fold"]["mean"]
    global_honest_mean = results["global"]["mean"]
    verdict = "VERIFIED" if fold_honest_mean < GATE_VERIFIED else "BAG_MEAN_OPTIMISM_TRAP"
    delta_vs_bagmean = fold_honest_mean - NB3123_BAG_MEAN
    delta_vs_perseed = fold_honest_mean - NB3123_PER_SEED_MEAN
    print(f"   fold-honest mean   = {fold_honest_mean:.4f}")
    print(f"   global-honest mean = {global_honest_mean:.4f}")
    print(f"   delta vs nb3123 bag-mean   {NB3123_BAG_MEAN:.4f} = "
          f"{delta_vs_bagmean:+.4f}")
    print(f"   delta vs nb3123 per-seed   {NB3123_PER_SEED_MEAN:.4f} = "
          f"{delta_vs_perseed:+.4f}")
    print(f"   verdict = {verdict}")

    # -- Save summary --------------------------------------------------------
    summary = {
        "tag": TAG,
        "parent_tag": PARENT_TAG,
        "method": "honest_5fold_scaffold_CV_per_kf_seed_with_30seed_bag_inside_fold",
        "anchor_base": "chemprop_aux",
        "anchor_pre_unblind": True,
        "anchor_in_rae": rae_anchor,
        "K_orig": 18,
        "K_with_qlow": 19,
        "K18_idx_in_117col": K18_idx.tolist(),
        "q_cut": Q_CUT,
        "k18_oof_full_rae": k18_full_rae,
        "kf_seeds": KF_SEEDS,
        "n_kf_seeds": len(KF_SEEDS),
        "n_folds": N_FOLDS,
        "resid_seeds_deep": RESID_SEEDS_DEEP,
        "n_seeds_deep": len(RESID_SEEDS_DEEP),
        "n_unb": int(n_unb),
        "n_unique_scaffolds": int(n_unique_scaf),
        "qlow_mode_fold": results["fold"],
        "qlow_mode_global": results["global"],
        "honest_mean_fold": fold_honest_mean,
        "honest_std_fold": results["fold"]["std"],
        "honest_mean_global": global_honest_mean,
        "honest_std_global": results["global"]["std"],
        "mean_rae": fold_honest_mean,  # canonical: fold-honest
        "nb3123_bagmean_ref": NB3123_BAG_MEAN,
        "nb3123_perseed_ref": NB3123_PER_SEED_MEAN,
        "nb2960_K18_ref": NB2960_K18_REF,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "delta_vs_nb3123_bagmean": delta_vs_bagmean,
        "delta_vs_nb3123_perseed": delta_vs_perseed,
        "gate_verified": GATE_VERIFIED,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   nb3123 bag-mean (DEPLOY)  = {NB3123_BAG_MEAN:.4f}")
    print(f"   nb3123 per-seed mean      = {NB3123_PER_SEED_MEAN:.4f}")
    print(f"   nb3140 HONEST fold mean   = {fold_honest_mean:.4f} +/- "
          f"{results['fold']['std']:.5f}")
    print(f"   nb3140 honest global mean = {global_honest_mean:.4f} +/- "
          f"{results['global']['std']:.5f}")
    print(f"   delta vs bag-mean         = {delta_vs_bagmean:+.4f}")
    print(f"   delta vs per-seed mean    = {delta_vs_perseed:+.4f}")
    print(f"   GATE (< {GATE_VERIFIED:.4f})         = {verdict}")
    print(f"   wall                      = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "honest_mean_fold", "honest_std_fold",
        "honest_mean_global", "honest_std_global",
        "delta_vs_nb3123_bagmean", "delta_vs_nb3123_perseed",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
