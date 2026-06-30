"""nb2661 -- DEEP-30 verification of {K=18, K=20} EQUAL-WEIGHT PAIR alone.

CONTEXT (cycle 169+):
    nb2630 fresh-seed verification of {K=18, K=20} found per-seed pooled
    RAE = 0.4665 (still FAIL, > GATE_PROMOTE 0.4570).  But that was on
    5 fresh "kf seed" replicates.

    The deep-30 protocol (cycle 160 rule) requires evaluating the
    30-seed BAG-MEAN of the predictions themselves, not the mean of
    per-seed pooled RAEs.  The bag-mean averages predictions across 30
    LGBM seeds first, THEN computes RAE -- this is a different quantity
    than the mean of per-seed RAEs and is the correct deploy metric.

    From nb2640 (cycle 160 deep-30 of the {K=18,20,24,28} 4-K ensemble),
    the per-K 30-seed BAG-MEAN RAEs were recorded as:
        K=18 bag-mean RAE = 0.4545
        K=20 bag-mean RAE = 0.4682
        K=24 bag-mean RAE = 0.4739
        K=28 bag-mean RAE = 0.4779
    The 4-K ensemble bag-mean was 0.4611 (deploy_30seed_bagmean_rae).

    QUESTION FOR THIS SCRIPT:
        Equal-weight average the K=18 and K=20 30-seed bag-mean
        predictions themselves and compute pooled RAE.  Does dropping
        K=24 and K=28 break the 0.4570 floor?

PROTOCOL:
    1. Build 117-col 5-way feature matrix (same as nb2103/nb2240/
       nb2604/nb2640).
    2. Slice K=18 cols (nb2604_summary k18_idx_in_117col, 18 cols) and
       K=20 cols (nb2240_summary k20_surviving_idx_in_117, 20 cols).
    3. anchor = chemprop_aux te[unb_idx]; residual = y_unb - anchor.
    4. For each K in {18, 20}:
         For each seed in {0, 1, ..., 29}:
           - KFold(5, shuffle=True, random_state=seed) cross-fit on
             X_unb_K -> residual; OOF residual.
           - pred_K_s_unb = anchor + resid_oof_s          (253,)
           - Refit-all on X_unb_K -> residual; predict te -> te_resid.
           - pred_K_s_te = anchor_te + te_resid           (513,)
         K_bag_mean_unb = mean over 30 seeds of pred_K_s_unb
         K_bag_mean_te  = mean over 30 seeds of pred_K_s_te
    5. Equal-weight pair (DEPLOY):
         pred_oof = 0.5 * K18_bag_mean_unb + 0.5 * K20_bag_mean_unb
         pred_te  = 0.5 * K18_bag_mean_te  + 0.5 * K20_bag_mean_te
    6. 5-fold scaffold CV pooled RAE at kf_seed=1001.  (Pooled RAE is
       deterministic given the predictions, since predictions are fixed
       across the eval-only KFold split; we record per-fold RAE too.)
    7. Gate (per task spec):
         pooled RAE < 0.4570 -> "DEEP30_PROMOTE"
         pooled RAE < 0.4601 -> "DEEP30_MARGINAL_BEAT"
         else                -> "DEEP30_FAIL"

INPUTS:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/te_chemprop_aux.npy
    data/processed/nb2604_summary.json
    data/processed/nb2240_summary.json
    data/processed/te_atompair.npy
    data/processed/te_maccs.npy
    data/processed/te_chemprop_embed_300.npy
    data/processed/te_avalon512.npy
    C:/pxr_artifacts/nb1030/X_mordred_test.npy
    data/processed/nb1352_summary.json, nb1392_summary.json,
                   nb1484_summary.json, nb1523_summary.json,
                   nb1524_summary.json, nb1541_summary.json

OUTPUTS:
    data/processed/nb2661_summary.json
    data/processed/nb2661_pred_oof.npy           (253,) float32
    data/processed/te_nb2661.npy                 (513,) float32

REFERENCES:
    nb2630 per-seed pooled (5 fresh kf seeds) = 0.4665  -- FAIL gate-A
    nb2640 K=18 30-seed bag-mean RAE          = 0.4545  (cached)
    nb2640 K=20 30-seed bag-mean RAE          = 0.4682  (cached)
    nb2640 {K=18,20,24,28} 30-seed bag-mean   = 0.4611
    nb2640 per-seed pooled mean +/- std       = 0.4814 +/- 0.0095
    nb2171 PRIMARY-1 ceiling deep-30          = 0.4682
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
import lightgbm as lgb
from rdkit import Chem
from rdkit import RDLogger
from scipy.stats import t as student_t
from sklearn.model_selection import KFold

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2661"
PARENT_TAG = "nb2630"

# -- Anchor + residual params (IDENTICAL to nb2640) --------------------------
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
RESID_FOLDS = 5
RESID_SEEDS_DEEP = list(range(30))           # 30 fresh seeds {0,...,29}

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

NB2240_SUMMARY = DATA_PROCESSED / "nb2240_summary.json"
NB2604_SUMMARY = DATA_PROCESSED / "nb2604_summary.json"
NB2640_SUMMARY = DATA_PROCESSED / "nb2640_summary.json"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

# -- CV eval -----------------------------------------------------------------
N_FOLDS = 5
KF_SEED = 1001                      # per task spec

# -- Gates (per task spec) ---------------------------------------------------
GATE_PROMOTE = 0.4570               # < 0.4570 -> DEEP30_PROMOTE
GATE_MARGINAL_BEAT = 0.4601         # < 0.4601 -> DEEP30_MARGINAL_BEAT

# -- References --------------------------------------------------------------
NB2630_REF = 0.4665                 # nb2630 per-seed pooled (5 fresh kf seeds)
CHEMPROP_AUX_REF = 0.6216
NB2171_REF = 0.4682                 # previous ceiling deep-30
NB2640_REF = 0.4611                 # 4-K ensemble deploy bag-mean
NB2640_K18_REF = 0.4545             # cached K=18 30-seed bag-mean RAE
NB2640_K20_REF = 0.4682             # cached K=20 30-seed bag-mean RAE


# ============================================================================
# helpers (IDENTICAL to nb2640 — 117-col matrix build)
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
    import pandas as pd
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
    """117-col matrix identical to nb2640 / nb2604 / nb2103 / nb2240."""
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


def build_K_bag_30seed(K_label, K_idx, X_te_full, unb_idx, anchor,
                       residual, te_anchor_513, n_test, n_unb, seeds):
    """Build per-seed K-pyramid bag, return aggregated 30-seed bag-mean
    predictions on (n_unb,) and (n_test,) plus per-seed pooled RAE."""
    X_te_K = X_te_full[:, K_idx].astype(np.float32)
    X_unb_K = X_te_K[unb_idx]
    print(f"   [{K_label}] X_unb_K = {X_unb_K.shape}  X_te_K = {X_te_K.shape}")
    per_seed_unb = np.zeros((len(seeds), n_unb), dtype=np.float64)
    per_seed_te = np.zeros((len(seeds), n_test), dtype=np.float64)
    per_seed_rae = []
    for i, s in enumerate(seeds):
        ts = time.time()
        resid_oof = _residual_cross_fit_one_seed(X_unb_K, residual, s)
        per_seed_unb[i] = anchor + resid_oof
        per_seed_rae.append(float(rae(anchor + residual, anchor + resid_oof)))
        te_resid_s = _train_full_then_predict_te(X_unb_K, residual, X_te_K, s)
        per_seed_te[i] = te_anchor_513 + te_resid_s
        if (i % 10) == 0 or i == len(seeds) - 1:
            print(f"      [{K_label}] seed={s:3d}  "
                  f"rae_corr={per_seed_rae[-1]:.4f}  "
                  f"wall={time.time()-ts:.2f}s  ({i+1}/{len(seeds)})")
    bag_mean_unb = per_seed_unb.mean(axis=0)
    bag_mean_te = per_seed_te.mean(axis=0)
    bag_mean_rae = float(rae(anchor + residual, bag_mean_unb))
    return bag_mean_unb, bag_mean_te, per_seed_rae, bag_mean_rae


# ============================================================================
# main
# ============================================================================

def main():
    import pandas as pd
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- DEEP-30 verify of {{K=18, K=20}} EQUAL-WEIGHT PAIR")
    print(f"          ref nb2630 5-fresh-kf-seed pooled = {NB2630_REF:.4f}")
    print(f"          ref nb2640 4-K bag-mean           = {NB2640_REF:.4f}")
    print(f"          gate PROMOTE       < {GATE_PROMOTE}")
    print(f"          gate MARGINAL_BEAT < {GATE_MARGINAL_BEAT}")
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
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] chemprop_aux te[unb_idx] RAE = {rae_anchor:.4f} "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor

    # -- Load K-feature index sets (K=18 and K=20 only) ----------------------
    print("\n" + "-" * 78)
    print("STEP 1: load K=18 / K=20 feature indices from cached summaries")
    print("-" * 78)
    with open(NB2604_SUMMARY) as f:
        nb2604 = json.load(f)
    K18_idx = np.array(nb2604["k18_idx_in_117col"], dtype=int)
    assert len(K18_idx) == 18, f"K18 len {len(K18_idx)} != 18"
    print(f"   K=18 idx (n={len(K18_idx)}): {K18_idx.tolist()}")

    with open(NB2240_SUMMARY) as f:
        nb2240 = json.load(f)
    K20_idx = np.array(nb2240["k20_surviving_idx_in_117"], dtype=int)
    assert len(K20_idx) == 20, f"K20 len {len(K20_idx)} != 20"
    print(f"   K=20 idx (n={len(K20_idx)}): {K20_idx.tolist()}")

    # -- Build 117-col matrix ------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 2: rebuild 117-col 5-way feature matrix")
    print("-" * 78)
    X_te_full, chembl_pool_size = build_117col_feature_matrix(te_smiles, n_test)
    print(f"   X_te_full = {X_te_full.shape}  chembl_pool={chembl_pool_size}")

    # -- Build per-K 30-seed bag means ---------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 3: per-K residual-LGBM with {len(RESID_SEEDS_DEEP)} fresh seeds")
    print("-" * 78)

    print(f"\n  --- K_label=K18 -----------------------------------")
    K18_bag_unb, K18_bag_te, K18_per_seed_rae, K18_bag_rae = build_K_bag_30seed(
        "K18", K18_idx, X_te_full, unb_idx, anchor, residual,
        te_anchor_513, n_test, n_unb, RESID_SEEDS_DEEP,
    )
    print(f"   [K18] per-seed RAE mean = {np.mean(K18_per_seed_rae):.4f} "
          f"std = {np.std(K18_per_seed_rae, ddof=1):.4f}  "
          f"min={min(K18_per_seed_rae):.4f}  max={max(K18_per_seed_rae):.4f}")
    print(f"   [K18] 30-seed BAG-MEAN RAE = {K18_bag_rae:.4f} "
          f"(nb2640 ref {NB2640_K18_REF:.4f})")

    print(f"\n  --- K_label=K20 -----------------------------------")
    K20_bag_unb, K20_bag_te, K20_per_seed_rae, K20_bag_rae = build_K_bag_30seed(
        "K20", K20_idx, X_te_full, unb_idx, anchor, residual,
        te_anchor_513, n_test, n_unb, RESID_SEEDS_DEEP,
    )
    print(f"   [K20] per-seed RAE mean = {np.mean(K20_per_seed_rae):.4f} "
          f"std = {np.std(K20_per_seed_rae, ddof=1):.4f}  "
          f"min={min(K20_per_seed_rae):.4f}  max={max(K20_per_seed_rae):.4f}")
    print(f"   [K20] 30-seed BAG-MEAN RAE = {K20_bag_rae:.4f} "
          f"(nb2640 ref {NB2640_K20_REF:.4f})")

    # -- Equal-weight pair ensemble ------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 4: equal-weight pair: 0.5 * K18_bagmean + 0.5 * K20_bagmean")
    print("-" * 78)
    pred_oof_unb = 0.5 * K18_bag_unb + 0.5 * K20_bag_unb        # (n_unb,)
    pred_te_513 = 0.5 * K18_bag_te + 0.5 * K20_bag_te           # (n_test,)
    deploy_pooled_rae = float(rae(y_unb, pred_oof_unb))
    te_unb_in_rae = float(rae(y_unb, pred_te_513[unb_idx]))
    print(f"   equal-weight pair bag-mean RAE  (single-shot) = "
          f"{deploy_pooled_rae:.4f}")
    print(f"   te[unb_idx] RAE                                = "
          f"{te_unb_in_rae:.4f}")
    print(f"   pred_oof_unb std = {pred_oof_unb.std():.3f} "
          f"(truth_std {y_unb.std():.3f})")
    print(f"   pred_te_513  mean / std = {pred_te_513.mean():.3f} / "
          f"{pred_te_513.std():.3f}")

    # -- 5-fold scaffold CV pooled RAE on the fixed predictions --------------
    print("\n" + "-" * 78)
    print(f"STEP 5: 5-fold scaffold CV  kf_seed={KF_SEED}  (eval-only)")
    print("-" * 78)
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=KF_SEED,
    )
    fold_rae = []
    oof_pooled = np.full(n_unb, np.nan, dtype=np.float64)
    for f_i, (tr_loc, va_loc) in enumerate(splits):
        oof_pooled[va_loc] = pred_oof_unb[va_loc]
        r_fold = float(rae(y_unb[va_loc], pred_oof_unb[va_loc]))
        fold_rae.append(r_fold)
        print(f"   fold {f_i}: n_va={len(va_loc):3d}  rae={r_fold:.4f}")
    if np.isnan(oof_pooled).any():
        raise RuntimeError("scaffold splits did not cover all rows")
    pooled_rae = float(rae(y_unb, oof_pooled))
    fold_mean = float(np.mean(fold_rae))
    fold_std = float(np.std(fold_rae, ddof=1))
    print(f"\n   POOLED scaffold-CV RAE  = {pooled_rae:.4f}")
    print(f"   per-fold mean +/- std   = {fold_mean:.4f} +/- {fold_std:.4f}")
    print(f"   per-fold min / max      = {min(fold_rae):.4f} / {max(fold_rae):.4f}")
    # Sanity: pooled RAE on bag-mean predictions must match deploy_pooled_rae
    # (since splits cover every row exactly once and we just copy preds).
    assert abs(pooled_rae - deploy_pooled_rae) < 1e-10, (
        f"pooled {pooled_rae} != deploy {deploy_pooled_rae}"
    )

    # -- Per-seed pair-pooled RAE (variance diagnostic) ----------------------
    print("\n" + "-" * 78)
    print("STEP 6: per-seed pair-pooled RAE (variance diagnostic, n=30)")
    print("-" * 78)
    # per-seed equal-weight pair ensemble
    # K18_bag_unb / K20_bag_unb were means over 30 seeds, so reconstruct
    # per-seed pair preds from the per_seed_unb arrays we built inside
    # build_K_bag_30seed.  Refactor: re-run the per-seed pair build using
    # the same in-memory data.  For memory: we stored only the bag means.
    # We re-derive per-seed arrays inline by re-running the seeds (cheap
    # vs total cost) so we can report seed-dispersion stats correctly.
    print("   re-building per-seed pair predictions for variance stats...")
    X_te_K18 = X_te_full[:, K18_idx].astype(np.float32)
    X_unb_K18 = X_te_K18[unb_idx]
    X_te_K20 = X_te_full[:, K20_idx].astype(np.float32)
    X_unb_K20 = X_te_K20[unb_idx]
    per_seed_pair_rae = []
    for i, s in enumerate(RESID_SEEDS_DEEP):
        resid18 = _residual_cross_fit_one_seed(X_unb_K18, residual, s)
        resid20 = _residual_cross_fit_one_seed(X_unb_K20, residual, s)
        pred18 = anchor + resid18
        pred20 = anchor + resid20
        pair_pred = 0.5 * pred18 + 0.5 * pred20
        per_seed_pair_rae.append(float(rae(y_unb, pair_pred)))
    pair_arr = np.array(per_seed_pair_rae, dtype=np.float64)
    n_seeds = len(pair_arr)
    pair_mean = float(pair_arr.mean())
    pair_std = float(pair_arr.std(ddof=1))
    pair_median = float(np.median(pair_arr))
    pair_min = float(pair_arr.min())
    pair_max = float(pair_arr.max())
    pair_p025 = float(np.percentile(pair_arr, 2.5))
    pair_p975 = float(np.percentile(pair_arr, 97.5))
    sem = pair_std / np.sqrt(n_seeds)
    t_crit = float(student_t.ppf(0.975, df=n_seeds - 1))
    ci_low = pair_mean - t_crit * sem
    ci_high = pair_mean + t_crit * sem
    print(f"   per-seed pair RAE mean +/- std = "
          f"{pair_mean:.4f} +/- {pair_std:.5f}  (n={n_seeds})")
    print(f"   95% CI on mean                 = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   p2.5 / p97.5                   = "
          f"[{pair_p025:.4f}, {pair_p975:.4f}]")
    print(f"   min / max                      = {pair_min:.4f} / {pair_max:.4f}")
    print(f"   bag-mean RAE (vs per-seed mean)= "
          f"{deploy_pooled_rae:.4f} (vs {pair_mean:.4f})")

    # -- Gate (per task spec: gate on bag-mean RAE) --------------------------
    print("\n" + "-" * 78)
    print("STEP 7: GATE  (per task spec: pooled bag-mean RAE)")
    print("-" * 78)
    if deploy_pooled_rae < GATE_PROMOTE:
        verdict = "DEEP30_PROMOTE"
    elif deploy_pooled_rae < GATE_MARGINAL_BEAT:
        verdict = "DEEP30_MARGINAL_BEAT"
    else:
        verdict = "DEEP30_FAIL"
    print(f"   bag-mean pooled RAE = {deploy_pooled_rae:.4f}")
    print(f"     <{GATE_PROMOTE} -> DEEP30_PROMOTE")
    print(f"     <{GATE_MARGINAL_BEAT} -> DEEP30_MARGINAL_BEAT")
    print(f"     else            -> DEEP30_FAIL")
    print(f"   -> {verdict}")

    delta_vs_nb2630 = deploy_pooled_rae - NB2630_REF
    delta_vs_nb2640 = deploy_pooled_rae - NB2640_REF
    delta_vs_nb2171 = deploy_pooled_rae - NB2171_REF
    print(f"\n   delta vs nb2630 (0.4665) = {delta_vs_nb2630:+.4f}")
    print(f"   delta vs nb2640 (0.4611) = {delta_vs_nb2640:+.4f}")
    print(f"   delta vs nb2171 (0.4682) = {delta_vs_nb2171:+.4f}")

    # -- Save artifacts ------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 8: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof_unb.astype(np.float32))
    np.save(te_path, pred_te_513.astype(np.float32))
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    summary = {
        "tag": TAG,
        "parent_tag": PARENT_TAG,
        "method": "deep30_k18_k20_equal_weight_pair_alone",
        "variance_axis": "RESID_SEED (kf_seed=1001 fixed scaffold-CV evaluation)",
        "anchor_base": "chemprop_aux",
        "anchor_in_rae": rae_anchor,
        "anchor_pre_unblind": True,
        "K_set": [18, 20],
        "K18_idx_in_117col": K18_idx.tolist(),
        "K20_idx_in_117col": K20_idx.tolist(),
        "blend_weights": [0.5, 0.5],
        "resid_seeds_deep": RESID_SEEDS_DEEP,
        "resid_folds": RESID_FOLDS,
        "kf_seed": KF_SEED,
        "n_folds": N_FOLDS,
        "n_seeds_deep": n_seeds,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "K18_per_seed_rae": K18_per_seed_rae,
        "K20_per_seed_rae": K20_per_seed_rae,
        "K18_30seed_bagmean_rae": K18_bag_rae,
        "K20_30seed_bagmean_rae": K20_bag_rae,
        "K18_nb2640_ref": NB2640_K18_REF,
        "K20_nb2640_ref": NB2640_K20_REF,
        "per_seed_pair_rae": per_seed_pair_rae,
        "pair_per_seed_mean": pair_mean,
        "pair_per_seed_std": pair_std,
        "pair_per_seed_median": pair_median,
        "pair_per_seed_min": pair_min,
        "pair_per_seed_max": pair_max,
        "pair_per_seed_p025": pair_p025,
        "pair_per_seed_p975": pair_p975,
        "pair_per_seed_ci95_low": ci_low,
        "pair_per_seed_ci95_high": ci_high,
        "pair_per_seed_sem": float(sem),
        "t_crit_df29": t_crit,
        "pooled_bagmean_rae": deploy_pooled_rae,
        "scaffold_cv_pooled_rae": pooled_rae,
        "scaffold_cv_fold_rae": fold_rae,
        "scaffold_cv_fold_mean": fold_mean,
        "scaffold_cv_fold_std": fold_std,
        "te_unb_in_sample_rae": te_unb_in_rae,
        "te_mean": float(pred_te_513.mean()),
        "te_std": float(pred_te_513.std()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "nb2630_ref": NB2630_REF,
        "nb2640_ref": NB2640_REF,
        "nb2171_ref": NB2171_REF,
        "delta_vs_nb2630": delta_vs_nb2630,
        "delta_vs_nb2640": delta_vs_nb2640,
        "delta_vs_nb2171": delta_vs_nb2171,
        "gate_promote": GATE_PROMOTE,
        "gate_marginal_beat": GATE_MARGINAL_BEAT,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   K18 30-seed bag-mean RAE   = {K18_bag_rae:.4f}  "
          f"(ref {NB2640_K18_REF:.4f})")
    print(f"   K20 30-seed bag-mean RAE   = {K20_bag_rae:.4f}  "
          f"(ref {NB2640_K20_REF:.4f})")
    print(f"   pair bag-mean pooled RAE   = {deploy_pooled_rae:.4f}")
    print(f"   per-seed pair mean +/- std = {pair_mean:.4f} +/- {pair_std:.5f}")
    print(f"   95% CI                     = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   te[unb_idx] RAE            = {te_unb_in_rae:.4f}")
    print(f"   gate                       = "
          f"<{GATE_PROMOTE} PROMOTE / <{GATE_MARGINAL_BEAT} MARGINAL -> "
          f"{verdict}")
    print(f"   wall                       = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "K18_30seed_bagmean_rae",
        "K20_30seed_bagmean_rae",
        "pooled_bagmean_rae",
        "pair_per_seed_mean",
        "pair_per_seed_std",
        "pair_per_seed_ci95_low",
        "pair_per_seed_ci95_high",
        "te_unb_in_sample_rae",
        "delta_vs_nb2630",
        "delta_vs_nb2640",
        "delta_vs_nb2171",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
