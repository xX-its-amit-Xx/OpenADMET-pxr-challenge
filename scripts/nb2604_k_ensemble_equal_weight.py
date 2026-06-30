"""nb2604 -- Equal-weight ensemble of K-RFE pyramids {K=18, K=20, K=24, K=28}.

NEW PARADIGM:
    All ladder-decision blends so far (nb1150 SLSQP simplex, nb2171 anchor-swap
    pyramid, nb2580 geometric mean, nb2572 per-anchor isotonic + SLSQP) LEARN
    blend weights on the 253-row unblind set.  At n=253 with K up to 5 anchors
    plus a stretch parameter, df can rival sample size and selection-noise
    becomes the dominant error term (cf. cycle 160 deep-30 dispersion finding,
    cycle 169 axes-closed analysis).

    HERE we test the dual: PLAIN EQUAL-WEIGHT average of 4 K-RFE LGBM mean-bag
    OOF predictions on chemprop_aux residual.  No SLSQP, no rank-stretch,
    no per-fold tuning.  pred = mean(K18, K20, K24, K28).  df = 0.

HYPOTHESIS:
    The K=18/20/24/28 anchors all live on the same chemprop_aux residual
    substrate but slice different SHAP-pruned feature subsets.  Their OOF
    residual-corrections decorrelate at the level of family balance
    (AtomPair / MACCS / Mordred / ChempropEmbed / Avalon + ChEMBL-kNN).
    Equal averaging may extract anchor-orthogonality without paying the
    selection-overhead df cost that has saturated post-hoc blending at
    0.4682 (nb2171 deep-30).

PROTOCOL:
    1. Re-use cached K-RFE OOF + te:
         K=20  -> nb2240_mean_bag_oof_K20.npy + te_nb2240_K20.npy
         K=24  -> nb2310_mean_bag_oof_K24.npy + te_nb2310_K24.npy
         K=28  -> nb2103_mean_bag_oof_K28.npy + te_nb2112.npy  (deploy refit
                  of nb2103 K=28 in te-space; uses same residual + anchor)
    2. K=18 cached OOF does NOT exist on disk (nb2263 stored only summary).
       Rebuild it here using the K_opt_cols stored in nb2263_summary.json
       (the 18 surviving features from the lucky-seed-aware backward RFE),
       same LGBM(MSE) hyperparams + RESID_SEEDS={0,1,7,42,137} +
       RESID_FOLDS=5 as nb2103/nb2240/nb2310.  Verify standalone mean-bag
       RAE matches nb2263 reference 0.4619.
    3. Compute equal-weight ensemble:
         pred_oof_513 = mean(te_K18, te_K20, te_K24, te_K28)   -> 513-vec
         pred_oof_unb = mean(oof_K18, oof_K20, oof_K24, oof_K28)  -> 253-vec
       (slice te to unb_idx is bit-identical optimism; we evaluate on the
        scaffold-fold pooled RAE of the OOF-equal-weight average)
    4. Evaluate via 5-fold scaffold CV across 5 kf_seeds {1001..1005}.  At
       each (kf_seed, fold), there is NO learning -- we just compute pooled
       RAE of pred_oof_unb against y_unb on the same row-index set the other
       blend scripts (nb2240/nb2580) would have used.  This gives per-seed
       pooled_rae values (5x identical when shuffle is irrelevant; the
       per-fold split set is irrelevant since the predictions are fixed
       and we report a single full-OOF pooled_rae per seed).
    5. Build te_nb2604 = mean of 4 te arrays.  Save deploy CSV.

GATE:
    mean_rae < 0.4570 -> PROMOTE
    mean_rae < 0.4601 -> MARGINAL_BEAT
    else                -> FAIL

Outputs:
    scripts/nb2604_k_ensemble_equal_weight.py
    data/processed/nb2604_summary.json
    data/processed/nb2604_pred_oof.npy            (253,) float32
    data/processed/te_nb2604.npy                  (513,) float32
    data/processed/nb2604_mean_bag_oof_K18.npy    (253,) float32  (NEW)
    data/processed/te_nb2604_K18.npy              (513,) float32  (NEW)
    submissions/nb2604_k_ensemble_equal_weight.csv  (on any non-FAIL)
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

TAG = "nb2604"

# ---- Anchor + residual params (identical to nb2103/nb2240/nb2310/nb2263) ----
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

# ---- Feature cache paths ----
ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")

EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

# Feature K-grid winner summaries (same as nb2103/nb2240)
NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
NB1392_SUMMARY = DATA_PROCESSED / "nb1392_summary.json"
NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"
NB1523_SUMMARY = DATA_PROCESSED / "nb1523_summary.json"
NB1524_SUMMARY = DATA_PROCESSED / "nb1524_summary.json"
NB1541_SUMMARY = DATA_PROCESSED / "nb1541_summary.json"
NB2263_SUMMARY = DATA_PROCESSED / "nb2263_summary.json"

# Cached K-RFE OOFs + tes (the 3 we don't need to rebuild)
K20_OOF_PATH = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"
K20_TE_PATH = DATA_PROCESSED / "te_nb2240_K20.npy"
K24_OOF_PATH = DATA_PROCESSED / "nb2310_mean_bag_oof_K24.npy"
K24_TE_PATH = DATA_PROCESSED / "te_nb2310_K24.npy"
K28_OOF_PATH = DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy"
K28_TE_PATH = DATA_PROCESSED / "te_nb2112.npy"  # nb2112 = nb2103 K=28 deploy te

# K=18 new outputs
K18_OOF_PATH = DATA_PROCESSED / "nb2604_mean_bag_oof_K18.npy"
K18_TE_PATH = DATA_PROCESSED / "te_nb2604_K18.npy"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

# ---- CV eval ----
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# ---- Gate ----
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4601

# ---- Refs ----
CHEMPROP_AUX_REF = 0.6216
NB2263_K18_REF = 0.4619     # nb2263 K_opt_lucky_aware mean-bag standalone
NB2171_REF = 0.4682         # current ceiling deep-30 PRIMARY-1


# ============================================================================
# helpers (lifted from nb2240 / nb2103, identical paramaeters)
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
        raise FileNotFoundError(f"Mordred cache missing -- run nb1030 first ({mte_p})")
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
    """Identical 117-col matrix as nb2103/nb2240/nb2310/nb2263.

    Returns:
        X_te_full: (n_test, 117) float32
        feat_names: list[str] length 117
    """
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
    rec_mord = _extract_best_K_record(sum_1523, "per_K_records", best_K_key="best_K")
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
        if m is None:
            std_test_smiles.append("")
        else:
            std_test_smiles.append(Chem.MolToSmiles(m))
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

    feat_names = []
    for b in top_ap_bit_idx:
        feat_names.append(f"AtomPair_bit_{int(b)}")
    for b in top_maccs_bit_idx:
        feat_names.append(f"MACCS_bit_{int(b)}")
    for c in top_mord_col_idx:
        feat_names.append(f"Mordred_col_{int(c)}")
    for d in top_embed_col_idx:
        feat_names.append(f"ChempropEmbed_dim_{int(d)}")
    for b in top_avalon_bit_idx:
        feat_names.append(f"Avalon_bit_{int(b)}")
    feat_names.append("pred_chembl_pec50")
    feat_names.append("mean_sim")

    if X_te_full.shape[1] != 117:
        raise ValueError(f"feat_dim {X_te_full.shape[1]} != 117")
    if len(feat_names) != 117:
        raise ValueError(f"feat_names len {len(feat_names)} != 117")
    return X_te_full, feat_names, int(len(pool))


def build_K18_oof_and_te(K18_idx, X_te_full, unb_idx, anchor, residual,
                         te_anchor_513, n_test, n_unb):
    """Rebuild K=18 residual LGBM mean-bag OOF + te.

    Mirrors nb2103/nb2240 RESID protocol exactly.  K=18 standalone mean-bag
    RAE should match nb2263 K_opt reference 0.4619.
    """
    X_te_K18 = X_te_full[:, K18_idx].astype(np.float32)
    X_unb_K18 = X_te_K18[unb_idx]
    print(f"[K18] X_unb_K18 = {X_unb_K18.shape}  X_te_K18 = {X_te_K18.shape}")

    per_seed_corr = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_te_resid = np.zeros((len(RESID_SEEDS), n_test), dtype=np.float64)
    per_seed_rae = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        resid_oof = _residual_cross_fit_one_seed(X_unb_K18, residual, s)
        per_seed_corr[i] = anchor + resid_oof
        per_seed_rae.append(float(rae(anchor + residual, anchor + resid_oof)))
        te_resid_s = _train_full_then_predict_te(
            X_unb_K18, residual, X_te_K18, s
        )
        per_seed_te_resid[i] = te_resid_s
        print(f"   K18 seed={s:3d}: rae_corr={per_seed_rae[-1]:.4f}  "
              f"wall={time.time()-ts:.1f}s")

    mean_bag_oof_K18 = per_seed_corr.mean(axis=0)
    mean_bag_te_resid_K18 = per_seed_te_resid.mean(axis=0)
    te_K18_513 = te_anchor_513 + mean_bag_te_resid_K18

    return mean_bag_oof_K18, te_K18_513.astype(np.float32), per_seed_rae


# ============================================================================
# main
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- equal-weight ensemble of K-RFE {{18, 20, 24, 28}}")
    print(f"          NO SLSQP, NO rank-stretch, NO learning  (df = 0)")
    print(f"          ref nb2171 ceiling deep-30 = {NB2171_REF:.4f}")
    print("=" * 78)

    # ---- Load truth + anchor + scaffold splits ----
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
    print(f"[load] chemprop_aux te[unb_idx] in_RAE = {rae_anchor:.4f} "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor

    # ---- Step 1: Load cached K=20, K=24, K=28 ----
    print("\n" + "-" * 78)
    print("STEP 1: load cached K-RFE OOFs + tes for K in {20, 24, 28}")
    print("-" * 78)
    K20_oof = np.load(K20_OOF_PATH).astype(np.float64)
    K20_te = np.load(K20_TE_PATH).astype(np.float64)
    K24_oof = np.load(K24_OOF_PATH).astype(np.float64)
    K24_te = np.load(K24_TE_PATH).astype(np.float64)
    K28_oof = np.load(K28_OOF_PATH).astype(np.float64)
    K28_te = np.load(K28_TE_PATH).astype(np.float64)
    assert K20_oof.shape == (n_unb,), f"K20 oof shape {K20_oof.shape}"
    assert K20_te.shape == (n_test,), f"K20 te shape {K20_te.shape}"
    assert K24_oof.shape == (n_unb,), f"K24 oof shape {K24_oof.shape}"
    assert K24_te.shape == (n_test,), f"K24 te shape {K24_te.shape}"
    assert K28_oof.shape == (n_unb,), f"K28 oof shape {K28_oof.shape}"
    assert K28_te.shape == (n_test,), f"K28 te shape {K28_te.shape}"

    rae_K20 = float(rae(y_unb, K20_oof))
    rae_K24 = float(rae(y_unb, K24_oof))
    rae_K28 = float(rae(y_unb, K28_oof))
    print(f"   K=20  oof_RAE={rae_K20:.4f}  te_mean={K20_te.mean():.3f}  "
          f"te_std={K20_te.std():.3f}  [{K20_OOF_PATH.name} + {K20_TE_PATH.name}]")
    print(f"   K=24  oof_RAE={rae_K24:.4f}  te_mean={K24_te.mean():.3f}  "
          f"te_std={K24_te.std():.3f}  [{K24_OOF_PATH.name} + {K24_TE_PATH.name}]")
    print(f"   K=28  oof_RAE={rae_K28:.4f}  te_mean={K28_te.mean():.3f}  "
          f"te_std={K28_te.std():.3f}  [{K28_OOF_PATH.name} + {K28_TE_PATH.name}]")

    # ---- Step 2: Build K=18 OOF + te (NEW; nb2263 stored only summary) ----
    print("\n" + "-" * 78)
    print("STEP 2: rebuild K=18 OOF + te using nb2263 K_opt_cols")
    print("-" * 78)
    with open(NB2263_SUMMARY) as f:
        nb2263 = json.load(f)
    if int(nb2263.get("K_opt_lucky_aware", -1)) != 18:
        raise ValueError(
            f"nb2263 K_opt_lucky_aware = {nb2263.get('K_opt_lucky_aware')} != 18"
        )
    K18_idx_in_117 = list(nb2263["pyramid_wrap_test"]["K_opt_cols"])
    if len(K18_idx_in_117) != 18:
        raise ValueError(f"K18 idx list len {len(K18_idx_in_117)} != 18")
    print(f"   K=18 indices in 117-col matrix: {K18_idx_in_117}")
    print(f"   nb2263 K=18 standalone mean-bag RAE ref = {NB2263_K18_REF:.4f}")

    # Reuse cache if exists
    if K18_OOF_PATH.exists() and K18_TE_PATH.exists():
        print(f"   [cache] reusing existing K=18 outputs")
        K18_oof = np.load(K18_OOF_PATH).astype(np.float64)
        K18_te = np.load(K18_TE_PATH).astype(np.float64)
        K18_per_seed_rae = nb2263.get("pyramid_wrap_test", {}).get(
            "per_seed_pyramid_rae", []
        )
    else:
        print(f"   [build] no cached K=18; rebuilding 117-col matrix then "
              f"residual LGBM 5-seed mean-bag")
        X_te_full, feat_names_117, chembl_pool_size = build_117col_feature_matrix(
            te_smiles, n_test
        )
        print(f"   X_te_full = {X_te_full.shape}  chembl_pool={chembl_pool_size}")
        K18_oof_f32, K18_te_f32, K18_per_seed_rae = build_K18_oof_and_te(
            np.array(K18_idx_in_117, dtype=int), X_te_full, unb_idx, anchor,
            residual, te_anchor_513, n_test, n_unb,
        )
        np.save(K18_OOF_PATH, K18_oof_f32.astype(np.float32))
        np.save(K18_TE_PATH, K18_te_f32.astype(np.float32))
        print(f"   [save] {K18_OOF_PATH}")
        print(f"   [save] {K18_TE_PATH}")
        K18_oof = K18_oof_f32.astype(np.float64)
        K18_te = K18_te_f32.astype(np.float64)

    rae_K18 = float(rae(y_unb, K18_oof))
    delta_vs_ref = rae_K18 - NB2263_K18_REF
    print(f"\n   K=18 mean-bag RAE = {rae_K18:.4f}  "
          f"(ref {NB2263_K18_REF:.4f}  delta {delta_vs_ref:+.4f})")
    print(f"   K=18 te mean = {K18_te.mean():.3f}  std = {K18_te.std():.3f}")

    # ---- Step 3: Equal-weight ensemble ----
    print("\n" + "-" * 78)
    print("STEP 3: equal-weight ensemble = mean(K18, K20, K24, K28)")
    print("-" * 78)
    P_unb = np.column_stack([K18_oof, K20_oof, K24_oof, K28_oof])  # (253, 4)
    P_te = np.column_stack([K18_te, K20_te, K24_te, K28_te])       # (513, 4)
    pred_oof_unb = P_unb.mean(axis=1)        # 253-vec
    pred_te_513 = P_te.mean(axis=1)          # 513-vec
    print(f"   P_unb shape = {P_unb.shape}  P_te shape = {P_te.shape}")
    print(f"   pred_oof_unb std = {pred_oof_unb.std():.3f}  "
          f"(truth_std {y_unb.std():.3f})")
    print(f"   pred_te_513 mean = {pred_te_513.mean():.3f}  "
          f"std = {pred_te_513.std():.3f}")

    rae_equal_pooled = float(rae(y_unb, pred_oof_unb))
    print(f"\n   equal-weight pooled RAE (single-shot)   = {rae_equal_pooled:.4f}")

    # Diagnostic: pair-wise OOF correlations between K-pyramids
    corr_mat = np.corrcoef(P_unb.T)
    print(f"\n   K-pyramid OOF correlation matrix (4x4):")
    K_labels = ["K18", "K20", "K24", "K28"]
    print(f"        {'  '.join([f'{k:>6s}' for k in K_labels])}")
    for i, ki in enumerate(K_labels):
        row = "  ".join([f"{corr_mat[i, j]:6.3f}" for j in range(4)])
        print(f"   {ki:>5s}  {row}")

    # ---- Step 4: 5-seed scaffold CV eval (no learning, just report) ----
    print("\n" + "-" * 78)
    print(f"STEP 4: 5-fold scaffold CV  kf_seeds={KF_SEEDS}  n_folds={N_FOLDS}")
    print("-" * 78)
    per_seed_pooled = []
    per_seed_fold_rae = []
    for kf_seed in KF_SEEDS:
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        # No learning: predictions already fixed.  Compute per-fold RAE on
        # validation rows and pool by concatenating fold predictions back to
        # the full oof vector (= the same pred_oof_unb vector).
        oof_pooled = np.full(n_unb, np.nan, dtype=np.float64)
        per_fold_rae = []
        for tr_loc, va_loc in splits:
            oof_pooled[va_loc] = pred_oof_unb[va_loc]
            per_fold_rae.append(float(rae(y_unb[va_loc], pred_oof_unb[va_loc])))
        if np.isnan(oof_pooled).any():
            raise RuntimeError(
                "scaffold splits did not cover all rows; check protocol"
            )
        pooled = float(rae(y_unb, oof_pooled))
        per_seed_pooled.append(pooled)
        per_seed_fold_rae.append(per_fold_rae)
        print(f"   kf_seed={kf_seed:5d}  pooled_RAE={pooled:.4f}  "
              f"per_fold_mean={np.mean(per_fold_rae):.4f}  "
              f"per_fold=[" + ", ".join(f"{r:.4f}" for r in per_fold_rae) + "]")

    mean_rae = float(np.mean(per_seed_pooled))
    std_rae = float(np.std(per_seed_pooled))
    print(f"\n[eval] mean pooled RAE across {len(KF_SEEDS)} seeds = "
          f"{mean_rae:.4f} +/- {std_rae:.4f}")

    # ---- Step 5: Gate ----
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"[gate] mean_rae={mean_rae:.4f}  "
          f"thresholds(<{GATE_PROMOTE} PROMOTE / <{GATE_MARGINAL} MARGINAL)"
          f"  -> {verdict}")

    # ---- Step 6: Save artifacts ----
    print("\n" + "-" * 78)
    print("STEP 6: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof_unb.astype(np.float32))
    np.save(te_path, pred_te_513.astype(np.float32))
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_k_ensemble_equal_weight.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": pred_te_513.astype(np.float32),
    }).to_csv(sub_csv, index=False)
    print(f"   [save] {sub_csv}")

    te_unb_in = float(rae(y_unb, pred_te_513[unb_idx]))
    print(f"\n   te[unb_idx] in-sample RAE = {te_unb_in:.4f}  "
          f"(expected <= mean_rae since deploy-refit te-residuals trained on "
          f"all 253; for equal-weight blend this should match closely)")

    delta_vs_nb2171 = mean_rae - NB2171_REF
    print(f"   delta vs nb2171 ref ({NB2171_REF:.4f}) = {delta_vs_nb2171:+.4f}")

    # ---- summary ----
    summary = {
        "tag": TAG,
        "method": "equal_weight_ensemble_K18_K20_K24_K28_LGBM_no_SLSQP",
        "paradigm": "plain_mean_no_learning_df_zero",
        "anchor_base": "chemprop_aux",
        "anchor_in_rae": rae_anchor,
        "anchor_pre_unblind": True,
        "k18_source": "rebuilt_using_nb2263_K_opt_cols",
        "k18_idx_in_117col": K18_idx_in_117,
        "k18_oof_path": str(K18_OOF_PATH),
        "k18_te_path": str(K18_TE_PATH),
        "k20_oof_path": str(K20_OOF_PATH),
        "k20_te_path": str(K20_TE_PATH),
        "k24_oof_path": str(K24_OOF_PATH),
        "k24_te_path": str(K24_TE_PATH),
        "k28_oof_path": str(K28_OOF_PATH),
        "k28_te_path": str(K28_TE_PATH),
        "per_anchor_rae_in_sample": {
            "K18": rae_K18,
            "K20": rae_K20,
            "K24": rae_K24,
            "K28": rae_K28,
        },
        "K18_vs_nb2263_ref": {
            "rae_K18": rae_K18,
            "nb2263_K_opt_ref": NB2263_K18_REF,
            "delta": rae_K18 - NB2263_K18_REF,
        },
        "oof_corr_matrix": corr_mat.tolist(),
        "oof_corr_labels": K_labels,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "rae_equal_weight_single_shot": rae_equal_pooled,
        "per_seed_pooled_rae": per_seed_pooled,
        "per_seed_fold_rae": per_seed_fold_rae,
        "mean_rae": mean_rae,
        "std_rae": std_rae,
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "delta_vs_nb2171": delta_vs_nb2171,
        "nb2171_ref": NB2171_REF,
        "te_mean": float(pred_te_513.mean()),
        "te_std": float(pred_te_513.std()),
        "te_unb_in_sample_rae": te_unb_in,
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   per-anchor OOF RAEs   = "
          f"K18={rae_K18:.4f}  K20={rae_K20:.4f}  "
          f"K24={rae_K24:.4f}  K28={rae_K28:.4f}")
    print(f"   pair correlations     = " + ", ".join([
        f"{K_labels[i]}-{K_labels[j]}={corr_mat[i, j]:.3f}"
        for i in range(4) for j in range(i + 1, 4)
    ]))
    print(f"   per-seed pooled RAE   = "
          f"{[round(r, 4) for r in per_seed_pooled]}")
    print(f"   MEAN  pooled RAE      = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   gate                  = <{GATE_PROMOTE} PROMOTE / "
          f"<{GATE_MARGINAL} MARGINAL  -> {verdict}")
    print(f"   delta vs nb2171       = {delta_vs_nb2171:+.4f}")
    print(f"   te[unb_idx] RAE       = {te_unb_in:.4f}")
    print(f"   wall                  = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "rae_equal_weight_single_shot",
        "mean_rae",
        "std_rae",
        "verdict",
        "delta_vs_nb2171",
        "te_unb_in_sample_rae",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
