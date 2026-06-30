"""nb2960 -- Fresh-seed DEEP-30 rebuild of K=18,20,24,28 RFE pyramids
for nb2943's best blend.

CONTEXT (cycle 213+):
    nb2943's best 0.4576 (w_nb2240=0.5, w_nb1191=0.0, w_K=0.5) uses CACHED
    K-pyramid OOFs that were built with 5-seed bags:
      K=18  cached: nb2604_mean_bag_oof_K18.npy
      K=20  cached: nb2240_mean_bag_oof_K20.npy
      K=24  cached: nb2310_mean_bag_oof_K24.npy
      K=28  cached: nb2103_mean_bag_oof_K28.npy

    Per cycle 160 deep-30 rule (nb2060 / nb2095 4x under-dispersion) AND
    cycle 213 nb2640 finding (fresh deep-30 K=20 is 0.4682 NOT 0.4630), the
    5-seed bag means are optimistically biased.  Does nb2943's 0.4576 hold
    when the K-pyramids are deep-30 rebuilt with fresh seeds?

PROTOCOL:
    1. Rebuild each K in {18, 20, 24, 28} with 30 fresh seeds {3001..3030}
       using the SAME residual-LGBM recipe as nb2604 / nb2240 / nb2310 / nb2103
       (chemprop_aux anchor + residual LGBM on K-feature slice of 117-col).
    2. Save per-K deep-30 bag-mean OOF (253,) and te (513,).
    3. Build equal_K_30 = mean(K18_30, K24_30, K28_30); nb2240_30 == K20_30.
    4. Apply nb2943-style blend with BEST grid combo (w_2240=0.5, w_1191=0.0,
       w_K=0.5):  pred = 0.5 * K20_30  +  0.5 * equal_K_30
    5. 5-fold scaffold CV with 5 kf_seeds {1001..1005}; report per-seed pooled
       RAE and grand mean.
    6. Compare K-by-K deep-30 RAE vs cached 5-seed RAE; compare final blend
       vs nb2943's reported 0.4576.

GATE:
    deep-30 blend < 0.4570  -> "PROMOTE_TRUE_BREAKTHROUGH"
    deep-30 blend < 0.4598  -> "MARGINAL_HOLDS"
    else                    -> "FAILS_FRESH_SEED"

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/te_chemprop_aux.npy
    data/processed/nb2604_summary.json (K=18 idx)
    data/processed/nb2240_summary.json (K=20 idx)
    data/processed/nb2103_summary.json (K=28 idx)
    data/processed/nb2310_dist_train_feats_te.npy (K=24 = K20 + 4 dist cols)
    data/processed/te_atompair.npy
    data/processed/te_maccs.npy
    data/processed/te_chemprop_embed_300.npy
    data/processed/te_avalon512.npy
    C:/pxr_artifacts/nb1030/X_mordred_test.npy

Outputs:
    data/processed/nb2960_summary.json
    data/processed/nb2960_pred_oof.npy           (253,) float32
    data/processed/te_nb2960.npy                 (513,) float32
    data/processed/nb2960_K18_30seed_oof.npy     (253,) float32
    data/processed/nb2960_K20_30seed_oof.npy     (253,) float32
    data/processed/nb2960_K24_30seed_oof.npy     (253,) float32
    data/processed/nb2960_K28_30seed_oof.npy     (253,) float32
    data/processed/nb2960_K18_30seed_te.npy      (513,) float32
    data/processed/nb2960_K20_30seed_te.npy      (513,) float32
    data/processed/nb2960_K24_30seed_te.npy      (513,) float32
    data/processed/nb2960_K28_30seed_te.npy      (513,) float32
    submissions/nb2960_deep30_K_rebuild_nb2943.csv  (non-FAIL only)
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

TAG = "nb2960"
PARENT_TAG = "nb2943"

# -- Anchor + residual params (IDENTICAL recipe to nb2640) --------------------
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

NB2103_SUMMARY = DATA_PROCESSED / "nb2103_summary.json"
NB2240_SUMMARY = DATA_PROCESSED / "nb2240_summary.json"
NB2310_DIST_TE_PATH = DATA_PROCESSED / "nb2310_dist_train_feats_te.npy"
NB2604_SUMMARY = DATA_PROCESSED / "nb2604_summary.json"

# -- Cached 5-seed K-pyramid OOF (for comparison) ----------------------------
CACHED_K18_OOF = DATA_PROCESSED / "nb2604_mean_bag_oof_K18.npy"
CACHED_K20_OOF = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"
CACHED_K24_OOF = DATA_PROCESSED / "nb2310_mean_bag_oof_K24.npy"
CACHED_K28_OOF = DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy"

# -- ChEMBL kNN params (identical to nb2604/nb2640) --------------------------
KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

# -- CV eval -----------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]   # 5 kf_seeds per task spec

# -- nb2943 best-blend recipe -------------------------------------------------
W_NB2240 = 0.5      # weight on K20 (== nb2240_30)
W_NB1191 = 0.0      # nb2943's best had w_1191=0
W_K_EQUAL = 0.5     # weight on equal_K = mean(K18_30, K24_30, K28_30)

# -- Gates -------------------------------------------------------------------
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

# -- References --------------------------------------------------------------
NB2943_REF = 0.4576           # nb2943 best blend (5-seed cached K-pyramids)
NB2604_5SEED_REF = 0.4580
NB1191_REF = 0.4718
NB2171_REF = 0.4682
CHEMPROP_AUX_REF = 0.6216


# ============================================================================
# helpers (lifted verbatim from nb2640)
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
    """117-col matrix identical to nb2604/nb2640."""
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
    """Build deep-30 bag-mean OOF (253,) + te (513,) for one K-pyramid.

    Returns:
        bag_oof_unb : (n_unb,) float64       -- mean over seeds of (anchor + resid_oof)
        bag_te_513  : (n_test,) float64      -- mean over seeds of (anchor_te + te_resid)
        per_seed_rae : list[float]            -- per-seed RAE vs y_unb
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
    print(f"{TAG} -- DEEP-30 fresh-seed rebuild of K=18,20,24,28 RFE pyramids")
    print(f"          parent: {PARENT_TAG} (best 0.4576 with cached 5-seed K-pyramids)")
    print(f"          fresh seeds = {RESID_SEEDS_DEEP[0]}..{RESID_SEEDS_DEEP[-1]} "
          f"(n={len(RESID_SEEDS_DEEP)})")
    print(f"          blend: pred = {W_NB2240}*K20_30 + {W_K_EQUAL}*equal_K_30  "
          f"(equal_K = mean(K18, K24, K28))")
    print(f"          gate: <{GATE_PROMOTE} PROMOTE / <{GATE_MARGINAL} MARGINAL")
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

    # -- Load K-feature index sets -------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 1: load K-feature indices from cached summaries")
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

    with open(NB2103_SUMMARY) as f:
        nb2103 = json.load(f)
    K28_idx = None
    for rec in nb2103["per_K_records"]:
        if int(rec["K"]) == 28:
            K28_idx = np.array(rec["top_K_idx_in_117"], dtype=int)
            break
    if K28_idx is None or len(K28_idx) != 28:
        raise RuntimeError(f"K=28 idx not found or wrong length")
    print(f"   K=28 idx (n={len(K28_idx)}): {K28_idx.tolist()}")

    if not NB2310_DIST_TE_PATH.exists():
        raise FileNotFoundError(f"missing {NB2310_DIST_TE_PATH}")
    dist_feats_te = np.load(NB2310_DIST_TE_PATH).astype(np.float32)
    if dist_feats_te.shape != (n_test, 4):
        raise ValueError(
            f"dist_feats_te shape {dist_feats_te.shape} != ({n_test}, 4)"
        )
    print(f"   K=24 = K20 + dist_feats_te {dist_feats_te.shape} -> 24 cols")

    # -- Build 117-col matrix ------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 2: rebuild 117-col 5-way feature matrix")
    print("-" * 78)
    X_te_full, chembl_pool_size = build_117col_feature_matrix(te_smiles, n_test)
    print(f"   X_te_full = {X_te_full.shape}  chembl_pool={chembl_pool_size}")

    # K=24 special: K20 cols + 4 dist feats
    X_te_K20 = X_te_full[:, K20_idx].astype(np.float32)
    X_te_K24 = np.concatenate([X_te_K20, dist_feats_te], axis=1).astype(np.float32)
    assert X_te_K24.shape[1] == 24, f"K24 dim {X_te_K24.shape[1]} != 24"

    # -- Build per-K, 30-seed bag means ---------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 3: per-K residual-LGBM with {len(RESID_SEEDS_DEEP)} fresh seeds")
    print("-" * 78)
    K_specs = [
        ("K18", K18_idx),
        ("K20", K20_idx),
        ("K24", X_te_K24),
        ("K28", K28_idx),
    ]
    per_K_30seed_oof = {}
    per_K_30seed_te = {}
    per_K_seed_rae = {}
    per_K_bag_mean_rae = {}
    for K_label, K_data in K_specs:
        print(f"\n  --- K_label={K_label} -----------------------------------")
        bag_oof, bag_te, per_rae = build_K_30seed_bag(
            K_label, K_data, X_te_full, unb_idx, anchor, residual,
            te_anchor_513, n_test, n_unb, RESID_SEEDS_DEEP,
        )
        per_K_30seed_oof[K_label] = bag_oof
        per_K_30seed_te[K_label] = bag_te
        per_K_seed_rae[K_label] = per_rae
        bag_mean_rae = float(rae(y_unb, bag_oof))
        per_K_bag_mean_rae[K_label] = bag_mean_rae
        # Save per-K artifacts
        np.save(DATA_PROCESSED / f"{TAG}_{K_label}_30seed_oof.npy", bag_oof.astype(np.float32))
        np.save(DATA_PROCESSED / f"{TAG}_{K_label}_30seed_te.npy", bag_te.astype(np.float32))
        print(f"   [{K_label}] per-seed RAE  mean = {np.mean(per_rae):.4f} "
              f"std = {np.std(per_rae, ddof=1):.4f}  "
              f"min={min(per_rae):.4f}  max={max(per_rae):.4f}")
        print(f"   [{K_label}] 30-seed BAG-MEAN RAE = {bag_mean_rae:.4f}")

    # -- Compare K-by-K: deep-30 vs cached 5-seed -----------------------------
    print("\n" + "-" * 78)
    print("STEP 4: K-by-K comparison: deep-30 vs cached 5-seed")
    print("-" * 78)
    cached_5seed_rae = {}
    cached_paths = {
        "K18": CACHED_K18_OOF,
        "K20": CACHED_K20_OOF,
        "K24": CACHED_K24_OOF,
        "K28": CACHED_K28_OOF,
    }
    print(f"   {'K':>4s}  {'5seed_RAE':>10s}  {'30seed_RAE':>11s}  {'delta':>8s}")
    for K_label, p in cached_paths.items():
        cached_oof = np.load(p).astype(np.float64)
        r5 = float(rae(y_unb, cached_oof))
        r30 = per_K_bag_mean_rae[K_label]
        cached_5seed_rae[K_label] = r5
        print(f"   {K_label:>4s}  {r5:>10.4f}  {r30:>11.4f}  {r30 - r5:>+8.4f}")

    # -- Build equal_K = mean(K18_30, K24_30, K28_30) -------------------------
    print("\n" + "-" * 78)
    print("STEP 5: build equal_K_30 = mean(K18_30, K24_30, K28_30)")
    print("-" * 78)
    equal_K_oof = (per_K_30seed_oof["K18"]
                   + per_K_30seed_oof["K24"]
                   + per_K_30seed_oof["K28"]) / 3.0
    equal_K_te = (per_K_30seed_te["K18"]
                  + per_K_30seed_te["K24"]
                  + per_K_30seed_te["K28"]) / 3.0
    equal_K_rae = float(rae(y_unb, equal_K_oof))
    print(f"   equal_K_30 oof_RAE = {equal_K_rae:.4f}  "
          f"te_mean={equal_K_te.mean():.3f}  te_std={equal_K_te.std():.3f}")

    # -- Build nb2943-style blend ---------------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 6: nb2943-style blend: pred = {W_NB2240}*K20_30 + "
          f"{W_K_EQUAL}*equal_K_30")
    print("-" * 78)
    K20_oof_30 = per_K_30seed_oof["K20"]
    K20_te_30 = per_K_30seed_te["K20"]
    pred_oof = W_NB2240 * K20_oof_30 + W_K_EQUAL * equal_K_oof
    pred_te = W_NB2240 * K20_te_30 + W_K_EQUAL * equal_K_te
    blend_inplace_rae = float(rae(y_unb, pred_oof))
    print(f"   blend in-sample RAE (no CV) = {blend_inplace_rae:.4f}")

    # -- 5-fold scaffold CV over 5 kf_seeds -----------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 7: 5-fold scaffold CV over {len(KF_SEEDS)} kf_seeds {KF_SEEDS}")
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
    print("STEP 8: DEEP-30 FRESH-SEED GATE")
    print("-" * 78)
    if pooled_mean < GATE_PROMOTE:
        verdict = "PROMOTE_TRUE_BREAKTHROUGH"
    elif pooled_mean < GATE_MARGINAL:
        verdict = "MARGINAL_HOLDS"
    else:
        verdict = "FAILS_FRESH_SEED"
    delta_vs_nb2943 = pooled_mean - NB2943_REF
    delta_vs_nb2171 = pooled_mean - NB2171_REF
    print(f"   grand mean = {pooled_mean:.4f}")
    print(f"   delta vs nb2943 (0.4576)   = {delta_vs_nb2943:+.4f}")
    print(f"   delta vs nb2171 (0.4682)   = {delta_vs_nb2171:+.4f}")
    print(f"   verdict                    = {verdict}")

    # -- Save artifacts ------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 9: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof.astype(np.float32))
    np.save(te_path, pred_te.astype(np.float32))
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    te_unb_in_rae = float(rae(y_unb, pred_te[unb_idx]))
    sub_csv = SUBMISSIONS / f"{TAG}_deep30_K_rebuild_nb2943.csv"
    if verdict != "FAILS_FRESH_SEED":
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
        "method": "deep30_freshseed_K_rebuild_for_nb2943_best_blend",
        "anchor_base": "chemprop_aux",
        "anchor_in_rae": rae_anchor,
        "anchor_pre_unblind": True,
        "K_set": [18, 20, 24, 28],
        "K18_idx_in_117col": K18_idx.tolist(),
        "K20_idx_in_117col": K20_idx.tolist(),
        "K28_idx_in_117col": K28_idx.tolist(),
        "K24_construction": "K20_idx + nb2310_dist_train_feats_te (4 dist cols)",
        "resid_seeds_deep": RESID_SEEDS_DEEP,
        "resid_folds": RESID_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "n_seeds_deep": len(RESID_SEEDS_DEEP),
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "blend_w_nb2240_K20": W_NB2240,
        "blend_w_nb1191": W_NB1191,
        "blend_w_K_equal": W_K_EQUAL,
        "blend_equal_K_components": ["K18", "K24", "K28"],
        "per_K_per_seed_rae": per_K_seed_rae,
        "per_K_30seed_bagmean_rae": per_K_bag_mean_rae,
        "per_K_cached_5seed_rae": cached_5seed_rae,
        "per_K_delta_30_minus_5": {
            k: per_K_bag_mean_rae[k] - cached_5seed_rae[k]
            for k in cached_5seed_rae
        },
        "equal_K_30_oof_rae": equal_K_rae,
        "blend_in_sample_rae": blend_inplace_rae,
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
        "submission_csv": str(sub_csv) if verdict != "FAILS_FRESH_SEED" else None,
        "nb2943_ref": NB2943_REF,
        "nb2604_5seed_ref": NB2604_5SEED_REF,
        "nb2171_ref": NB2171_REF,
        "nb1191_ref": NB1191_REF,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "delta_vs_nb2943": delta_vs_nb2943,
        "delta_vs_nb2171": delta_vs_nb2171,
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   per-K 30-seed bag-mean RAE = "
          + ", ".join([f"{k}={v:.4f}" for k, v in per_K_bag_mean_rae.items()]))
    print(f"   per-K cached 5-seed RAE    = "
          + ", ".join([f"{k}={v:.4f}" for k, v in cached_5seed_rae.items()]))
    print(f"   per-K delta (30 - 5)       = "
          + ", ".join([f"{k}={per_K_bag_mean_rae[k] - cached_5seed_rae[k]:+.4f}"
                       for k in cached_5seed_rae]))
    print(f"   equal_K_30 RAE             = {equal_K_rae:.4f}")
    print(f"   blend in-sample RAE        = {blend_inplace_rae:.4f}")
    print(f"   blend grand mean (5 kf)    = {pooled_mean:.4f} +/- {pooled_std:.5f}")
    print(f"   nb2943 ref (5-seed cached) = {NB2943_REF:.4f}")
    print(f"   delta vs nb2943            = {delta_vs_nb2943:+.4f}")
    print(f"   delta vs nb2171            = {delta_vs_nb2171:+.4f}")
    print(f"   te[unb_idx] RAE            = {te_unb_in_rae:.4f}")
    print(f"   verdict                    = {verdict}")
    print(f"   wall                       = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "pooled_rae_grand_mean",
        "pooled_rae_grand_std",
        "blend_in_sample_rae",
        "delta_vs_nb2943",
        "delta_vs_nb2171",
        "te_unb_in_sample_rae",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
    print(f"  per_K_30seed_bagmean_rae: {res.get('per_K_30seed_bagmean_rae')}")
    print(f"  per_K_cached_5seed_rae: {res.get('per_K_cached_5seed_rae')}")
