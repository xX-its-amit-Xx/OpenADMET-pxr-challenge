"""nb3151 -- Leave-one-out q40 19th feature in K=18 (vs nb3143 per-fold, nb3123 global).

NEW PARADIGM (vs nb3143 per-fold q40 / nb3123 global q40):

    nb3123 used a SINGLE GLOBAL q40 over ALL 253 K18 OOF preds (leakiest).
    nb3143 used a PER-FOLD q40 over the training fold K18 OOF (less leaky).

    This script makes the threshold PER-ROW HONEST via leave-one-out:

        for each row i in 0..252:
            q40_loo[i] = quantile(k18_oof_unb[~i], 0.40)   # ALL 252 others
            feat_19[i] = (k18_oof_unb[i] <= q40_loo[i])    # row-specific bit

        for each fold:
            X_K19 = [X_K18, feat_19]                       # already row-specific
            mdl = LGBM(seed=kf_seed).fit(X_K19[tr_loc], residual[tr_loc])
            oof[va_loc] = anchor[va_loc] + mdl.predict(X_K19[va_loc])

    Why LOO is strictly the cleanest:
      - GLOBAL q40 (nb3123): threshold for row i depends on row i itself -- leak.
      - PER-FOLD q40 (nb3143): threshold for va row i computed from tr fold only
        -> honest WRT row i, but threshold differs across folds; tree splits
        learn a fold-dependent boundary at deploy time we cannot reproduce.
      - PER-ROW LOO q40 (this script): threshold for row i computed from the
        OTHER 252 rows, independent of fold membership; the binary bit is a
        deterministic function of K18 OOF that has NEVER seen row i's prediction.
        At deploy time the LOO pattern is reproducible row-by-row.

    Hypothesis: removing both the within-row leak AND the fold-dependent
    boundary gives the cleanest binary regime indicator; if the q40-cut
    bit is genuinely a useful regime split, LOO should not hurt (and may
    help by stabilizing the boundary).  If LOO degrades, it confirms the
    nb3123 'lift' was at least partly the within-row leak.

PROTOCOL:
    1. Reproduce 117-col 5-way feature matrix verbatim (nb1352/1392/1484/
       1523/1524/1541 summaries + ChEMBL PXR kNN).
    2. K=18 idx from nb2604_summary.
    3. Precompute LOO q40 vector (one per row) from k18_oof_unb.
       feat_19_unb[i] = (k18_oof_unb[i] <= q40_loo[i])
    4. For each kf_seed in {1141..1145} (5 kf_seeds):
         splits = scaffold_kfold_indices(unb_scaffolds, 5, kf_seed)
         oof = full(n_unb, nan)
         for (tr_loc, va_loc) in splits:
             X_K19 = stack[X_unb_K18, feat_19_unb]
             mdl = LGBM(seed=kf_seed).fit(X_K19[tr_loc], residual[tr_loc])
             oof[va_loc] = anchor[va_loc] + mdl.predict(X_K19[va_loc])
         pooled_rae[kf_seed] = rae(y_unb, oof)
       mean over 5 kf_seeds = honest cross-fit RAE
    5. DEPLOY: feat_19_te uses GLOBAL q40 on 253 k18 OOF (the LOO pattern
       collapses to global at infinity), per the standard deploy convention;
       refit on all 253 with feat_19 and emit te_nb3151.npy.

GATE:
    honest mean RAE < 0.4475 -> "BETTER"
    else                     -> "FAIL"

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/te_chemprop_aux.npy
    data/processed/nb2604_summary.json (K=18 idx)
    data/processed/nb2960_K18_30seed_oof.npy (K18 OOF -- q_low source)
    data/processed/nb2960_K18_30seed_te.npy  (K18 te -- deploy q_low source)
    + nb1352/1392/1484/1523/1524/1541 summaries
    + te_atompair.npy, te_maccs.npy, te_chemprop_embed_300.npy, te_avalon512.npy
    + C:/pxr_artifacts/nb1030/X_mordred_test.npy
    + data/external/chembl_pxr_CHEMBL3401.parquet (+ siblings)

Outputs:
    data/processed/nb3151_summary.json
    data/processed/nb3151_pred_oof.npy   (253,) float32
    data/processed/te_nb3151.npy         (513,) float32
    submissions/nb3151_loo_q_feature.csv (BETTER only)
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

TAG = "nb3151"
PARENT_TAG = "nb3143"  # per-fold parent; nb3123 = global grandparent

# -- Anchor / residual params (IDENTICAL to nb3123 / nb3143 / nb2960) --------
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

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

# -- K18 deep-30 OOF / te (for q_low feature source) -------------------------
K18_OOF_PATH = DATA_PROCESSED / "nb2960_K18_30seed_oof.npy"
K18_TE_PATH = DATA_PROCESSED / "nb2960_K18_30seed_te.npy"
Q_CUT = 0.40

# -- ChEMBL kNN params (identical to nb3123 / nb3143 / nb2604 / nb2960) ------
KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

# -- CV eval -----------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = [1141, 1142, 1143, 1144, 1145]      # 5 fresh kf_seeds

# -- Gates -------------------------------------------------------------------
GATE_BETTER = 0.4475

# -- References --------------------------------------------------------------
NB3143_REF = None              # filled in at runtime if summary exists
NB3123_BAG_MEAN = 0.4389       # nb3123 reported (DEPLOY OPTIMISM)
NB3123_PER_SEED_MEAN = 0.4677  # nb3123 honest-leaning per-seed mean
NB2960_K18_REF = 0.4536
NB2171_REF = 0.4682
CHEMPROP_AUX_REF = 0.6216


# ============================================================================
# helpers (lifted verbatim from nb3143)
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
    """117-col matrix identical to nb3123 / nb3143 / nb2604 / nb2960."""
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
# core LOO logic
# ============================================================================

def compute_loo_q_feature(k18_oof_unb, q_cut):
    """Per-row leave-one-out q40: for each row i, q40 from the OTHER 252 rows.

    Returns:
        q40_loo : (n_unb,) float64 -- per-row threshold
        feat_19 : (n_unb,) float32 -- binary (k18_oof[i] <= q40_loo[i])
    """
    n = len(k18_oof_unb)
    q40_loo = np.empty(n, dtype=np.float64)
    # vectorize: sort once, then for each i drop i and take q-th index
    # simple loop is fine at n=253 (cheap)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        q40_loo[i] = float(np.quantile(k18_oof_unb[mask], q_cut))
    feat_19 = (k18_oof_unb <= q40_loo).astype(np.float32)
    return q40_loo, feat_19


def loo_q_cross_fit_one_kf_seed(
    X_unb_K19, anchor, residual, y_unb, unb_scaffolds,
    kf_seed, n_folds,
):
    """One kf_seed pass with LOO q40 feature already baked into X_unb_K19.

    For each fold:
        mdl = LGBM(seed=kf_seed).fit(X_unb_K19[tr_loc], residual[tr_loc])
        oof[va_loc] = anchor[va_loc] + mdl.predict(X_unb_K19[va_loc])
    """
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=n_folds, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof = np.full(n_unb, np.nan, dtype=np.float64)
    per_fold_rae = []
    for fi, (tr_loc, va_loc) in enumerate(splits):
        mdl = lgb.LGBMRegressor(**_lgbm_params(kf_seed))
        mdl.fit(X_unb_K19[tr_loc], residual[tr_loc])
        pred_va_resid = mdl.predict(X_unb_K19[va_loc])
        oof[va_loc] = anchor[va_loc] + pred_va_resid
        per_fold_rae.append(float(rae(y_unb[va_loc], oof[va_loc])))
    if np.isnan(oof).any():
        raise RuntimeError(f"scaffold splits did not cover all rows (kf_seed={kf_seed})")
    pooled = float(rae(y_unb, oof))
    return pooled, oof, per_fold_rae


def deploy_te_predict(X_unb_K19, X_te_K19, residual, kf_seeds):
    """Deploy: refit on all 253 + 5-kf-seed bag te (residual)."""
    n_te = X_te_K19.shape[0]
    sum_te_resid = np.zeros(n_te, dtype=np.float64)
    for s in kf_seeds:
        mdl = lgb.LGBMRegressor(**_lgbm_params(s))
        mdl.fit(X_unb_K19, residual)
        sum_te_resid += mdl.predict(X_te_K19)
    bag_te_resid = sum_te_resid / len(kf_seeds)
    return bag_te_resid


# ============================================================================
# main
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- LEAVE-ONE-OUT q40 19th feature (vs nb3143 per-fold, nb3123 global)")
    print(f"          parent: {PARENT_TAG} (per-fold q40)")
    print(f"          q_cut  = {Q_CUT}  (LOO per row)")
    print(f"          kf_seeds = {KF_SEEDS}  (n={len(KF_SEEDS)})")
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

    # -- Load K18 idx + K18 OOF/te (q_low feature source) --------------------
    print("\n" + "-" * 78)
    print("STEP 1: load K=18 idx + K18 deep-30 OOF/te (q_low source)")
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

    q40_global_ref = float(np.quantile(k18_oof, Q_CUT))
    print(f"   q40 (global ref on 253 K18 OOF) = {q40_global_ref:.4f}")

    # -- LOO q40 feature -----------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 2: compute LEAVE-ONE-OUT q40 per row (n=253)")
    print("-" * 78)
    q40_loo, feat_19_unb = compute_loo_q_feature(k18_oof, Q_CUT)
    feat_19_unb_share = float(feat_19_unb.mean())
    q40_loo_mean = float(q40_loo.mean())
    q40_loo_std = float(q40_loo.std(ddof=1))
    q40_loo_min = float(q40_loo.min())
    q40_loo_max = float(q40_loo.max())
    # global feat for comparison
    feat_19_unb_global = (k18_oof <= q40_global_ref).astype(np.float32)
    feat_19_global_share = float(feat_19_unb_global.mean())
    n_flipped = int((feat_19_unb != feat_19_unb_global).sum())
    print(f"   q40_loo  mean = {q40_loo_mean:.4f}  std = {q40_loo_std:.5f}  "
          f"[{q40_loo_min:.4f} .. {q40_loo_max:.4f}]")
    print(f"   q40 global ref = {q40_global_ref:.4f}")
    print(f"   feat_19_unb (LOO)    share = {feat_19_unb_share:.3f}  "
          f"(target ~{Q_CUT:.2f})")
    print(f"   feat_19_unb (global) share = {feat_19_global_share:.3f}")
    print(f"   rows flipped by LOO vs global = {n_flipped} / {n_unb}")

    # -- Build 117-col matrix, slice K=18 ------------------------------------
    print("\n" + "-" * 78)
    print("STEP 3: build 117-col matrix and slice K=18")
    print("-" * 78)
    X_te_full, chembl_pool_size = build_117col_feature_matrix(te_smiles, n_test)
    print(f"   X_te_full = {X_te_full.shape}  chembl_pool={chembl_pool_size}")
    X_te_K18 = X_te_full[:, K18_idx].astype(np.float32)
    X_unb_K18 = X_te_K18[unb_idx]
    assert X_unb_K18.shape == (n_unb, 18)
    assert X_te_K18.shape == (n_test, 18)
    print(f"   X_unb_K18 = {X_unb_K18.shape}  X_te_K18 = {X_te_K18.shape}")

    # Stack LOO feature into K=19 matrix
    X_unb_K19 = np.concatenate(
        [X_unb_K18, feat_19_unb.reshape(-1, 1)], axis=1
    ).astype(np.float32)
    print(f"   X_unb_K19 (LOO feat baked in) = {X_unb_K19.shape}")

    # -- Honest LOO q40 5-kf-seed pass ---------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 4: LOO q40 5-fold SCAFFOLD CV over {len(KF_SEEDS)} kf_seeds")
    print("-" * 78)
    per_kf_pooled = []
    per_kf_fold_rae = []
    oof_bag = np.zeros(n_unb, dtype=np.float64)
    for kf_seed in KF_SEEDS:
        ts = time.time()
        pooled, oof_seed, fold_rae = loo_q_cross_fit_one_kf_seed(
            X_unb_K19, anchor, residual, y_unb, unb_scaffolds,
            kf_seed=kf_seed, n_folds=N_FOLDS,
        )
        per_kf_pooled.append(pooled)
        per_kf_fold_rae.append(fold_rae)
        oof_bag += oof_seed
        wall = time.time() - ts
        print(f"   kf_seed={kf_seed}  pooled_RAE={pooled:.4f}  "
              f"folds_mean={np.mean(fold_rae):.4f}  wall={wall:.1f}s")
    oof_bag /= len(KF_SEEDS)
    oof_bag_rae = float(rae(y_unb, oof_bag))
    pooled_arr = np.array(per_kf_pooled, dtype=np.float64)
    pooled_mean = float(pooled_arr.mean())
    pooled_std = float(pooled_arr.std(ddof=1)) if len(pooled_arr) > 1 else 0.0
    pooled_min = float(pooled_arr.min())
    pooled_max = float(pooled_arr.max())
    print(f"\n   grand mean over {len(KF_SEEDS)} kf_seeds: {pooled_mean:.4f} +/- "
          f"{pooled_std:.5f}  (min={pooled_min:.4f}  max={pooled_max:.4f})")
    print(f"   bag-mean OOF RAE (avg over {len(KF_SEEDS)} kf_seeds) = "
          f"{oof_bag_rae:.4f}")

    # -- Deploy te (refit on all 253 with global q40 threshold for 513) ------
    print("\n" + "-" * 78)
    print(f"STEP 5: deploy refit on all 253 + 5-kf-seed bag te (global q40 for 513)")
    print("-" * 78)
    # Deploy convention: feat for 513 test uses the GLOBAL q40 from 253 K18 OOF
    # (the LOO pattern collapses to global at the deploy boundary -- we cannot
    # exclude an unknown test row from a 253 threshold)
    q40_dep_te = float(np.quantile(k18_te, Q_CUT))
    feat_19_te_dep = (k18_te <= q40_dep_te).astype(np.float32)
    feat_19_te_share = float(feat_19_te_dep.mean())
    print(f"   q40 deploy (te 513) = {q40_dep_te:.4f}  "
          f"feat_19_te share = {feat_19_te_share:.3f}")
    X_te_K19 = np.concatenate(
        [X_te_K18, feat_19_te_dep.reshape(-1, 1)], axis=1
    ).astype(np.float32)

    bag_te_resid = deploy_te_predict(X_unb_K19, X_te_K19, residual, KF_SEEDS)
    bag_te_513 = te_anchor_513 + bag_te_resid
    print(f"   bag_te_513 mean      = {bag_te_513.mean():.4f}  "
          f"std = {bag_te_513.std():.4f}")

    # -- Gate ----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 6: GATE")
    print("-" * 78)
    if pooled_mean < GATE_BETTER:
        verdict = "BETTER"
    else:
        verdict = "FAIL"
    delta_vs_bagmean = pooled_mean - NB3123_BAG_MEAN
    delta_vs_perseed = pooled_mean - NB3123_PER_SEED_MEAN
    delta_vs_k18 = pooled_mean - NB2960_K18_REF
    print(f"   grand mean             = {pooled_mean:.4f}")
    print(f"   delta vs nb3123 bag    = {delta_vs_bagmean:+.4f}  "
          f"(ref {NB3123_BAG_MEAN:.4f})")
    print(f"   delta vs nb3123 perseed= {delta_vs_perseed:+.4f}  "
          f"(ref {NB3123_PER_SEED_MEAN:.4f})")
    print(f"   delta vs nb2960 K18    = {delta_vs_k18:+.4f}  "
          f"(ref {NB2960_K18_REF:.4f})")
    print(f"   verdict                = {verdict}")

    # -- Save artifacts ------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 7: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_bag.astype(np.float32))
    np.save(te_path, bag_te_513.astype(np.float32))
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    te_clipped = np.clip(bag_te_513, 3.0, 9.0).astype(np.float32)
    te_unb_in_rae = float(rae(y_unb, te_clipped[unb_idx]))
    sub_csv = SUBMISSIONS / f"{TAG}_loo_q_feature.csv"
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
        "method": "loo_q40_19th_feat_K18_5kfseed_scaffold_cv",
        "anchor_base": "chemprop_aux",
        "anchor_in_rae": rae_anchor,
        "anchor_pre_unblind": True,
        "K_orig": 18,
        "K_with_qlow": 19,
        "K18_idx_in_117col": K18_idx.tolist(),
        "q_cut": Q_CUT,
        "q40_global_ref_unb": q40_global_ref,
        "q40_loo_mean": q40_loo_mean,
        "q40_loo_std": q40_loo_std,
        "q40_loo_min": q40_loo_min,
        "q40_loo_max": q40_loo_max,
        "q40_deploy_te": q40_dep_te,
        "feat_19_unb_loo_share": feat_19_unb_share,
        "feat_19_unb_global_share": feat_19_global_share,
        "feat_19_te_dep_share": feat_19_te_share,
        "n_rows_flipped_loo_vs_global": n_flipped,
        "k18_full_oof_rae": k18_full_rae,
        "kf_seeds": KF_SEEDS,
        "n_kf_seeds": len(KF_SEEDS),
        "n_folds": N_FOLDS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "per_kf_pooled_rae": per_kf_pooled,
        "per_kf_fold_rae": per_kf_fold_rae,
        "pooled_rae_grand_mean": pooled_mean,
        "pooled_rae_grand_std": pooled_std,
        "pooled_rae_grand_min": pooled_min,
        "pooled_rae_grand_max": pooled_max,
        "oof_bag_rae": oof_bag_rae,
        "mean_rae": pooled_mean,
        "te_unb_in_sample_rae": te_unb_in_rae,
        "te_mean": float(bag_te_513.mean()),
        "te_std": float(bag_te_513.std()),
        "te_clipped_mean": float(te_clipped.mean()),
        "te_clipped_std": float(te_clipped.std()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict == "BETTER" else None,
        "nb3123_bagmean_ref": NB3123_BAG_MEAN,
        "nb3123_perseed_ref": NB3123_PER_SEED_MEAN,
        "nb2960_K18_ref": NB2960_K18_REF,
        "nb2171_ref": NB2171_REF,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "delta_vs_nb3123_bagmean": delta_vs_bagmean,
        "delta_vs_nb3123_perseed": delta_vs_perseed,
        "delta_vs_nb2960_K18": delta_vs_k18,
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
    print(f"   pooled grand mean (CV) = {pooled_mean:.4f} +/- {pooled_std:.5f}")
    print(f"   bag-mean OOF RAE       = {oof_bag_rae:.4f}")
    print(f"   nb3123 bag-mean ref    = {NB3123_BAG_MEAN:.4f} (DEPLOY OPTIMISM)")
    print(f"   nb3123 per-seed ref    = {NB3123_PER_SEED_MEAN:.4f}")
    print(f"   nb2960 K18 ref         = {NB2960_K18_REF:.4f}")
    print(f"   delta vs nb3123 bag    = {delta_vs_bagmean:+.4f}")
    print(f"   delta vs nb3123 perseed= {delta_vs_perseed:+.4f}")
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
        "oof_bag_rae",
        "delta_vs_nb3123_bagmean",
        "delta_vs_nb3123_perseed",
        "te_unb_in_sample_rae",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
    print(f"  q40_global_ref_unb: {res.get('q40_global_ref_unb')}")
    print(f"  q40_loo_mean: {res.get('q40_loo_mean')}")
    print(f"  q40_loo_std: {res.get('q40_loo_std')}")
    print(f"  feat_19_unb_loo_share: {res.get('feat_19_unb_loo_share')}")
    print(f"  n_rows_flipped_loo_vs_global: {res.get('n_rows_flipped_loo_vs_global')}")
