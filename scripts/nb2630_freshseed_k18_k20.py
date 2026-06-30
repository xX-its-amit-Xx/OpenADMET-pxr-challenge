"""nb2630 -- Fresh-seed verification of {K=18, K=20} equal-weight ensemble.

CONTEXT:
    nb2621 ran an exhaustive 372-combo grid search across {K=14,16,18,20,
    22,24,28,32,36} equal-weight subsets (sizes 2..5) and ranked {K=18,
    K=20} as the BEST combo with pooled mean_rae = 0.4552 (claim).

    But the underlying K=18 OOF (nb2604_mean_bag_oof_K18.npy) and K=20 OOF
    (nb2240_mean_bag_oof_K20.npy) were each built with the SAME residual
    seed pool {0, 1, 7, 42, 137}.  The 372-combo selection across cached
    anchors picked the lowest pooled RAE -- this carries combinatorial
    selection-bias risk: if the {0,1,7,42,137} pool was lucky on K=18
    AND/OR K=20 specifically, the {K18, K20} winner is amplified.

HYPOTHESIS:
    Regenerate K=18 and K=20 mean-bag residual OOF + te using a FRESH,
    NEVER-USED-BEFORE seed pool {3001, 3002, 3003, 3004, 3005}.  Equal-
    weight average the two fresh-seed pyramids.  Evaluate 5-fold scaffold
    CV pooled RAE at kf_seed=1001 (same as nb2621 baseline).

    Compare to nb2621 0.4552:
      |delta| <= 0.005   -> DEEP_VERIFY_PASS  (not lucky-seed)
      |delta|  > 0.005   -> LUCKY_SEED_TRAP   (selection-bias amplified)

GATE:
    fresh result < 0.4570 -> VERIFIED_PROMOTE
    fresh result < 0.4580 -> VERIFIED_MARGINAL
    else                  -> FAILED_VERIFY

PROTOCOL:
    1. Load K=18 cols (nb2604_summary.json -> k18_idx_in_117col, len 18)
       and K=20 cols (nb2240_summary.json -> k20_surviving_idx_in_117,
       len 20).
    2. Build the same 117-col 5-way feature matrix (AtomPair + MACCS +
       Mordred + ChempropEmbed + Avalon + ChEMBL kNN) used by nb2103/
       nb2240/nb2604/nb2263 -- this is the canonical substrate.
    3. anchor = chemprop_aux te[unb_idx]; residual = y_unb - anchor.
    4. For each K in {18, 20}:
         X_unb_K = X_unb[:, K_cols].astype(float32)
         X_te_K  = X_te [:, K_cols].astype(float32)
         For each fresh_seed in {3001, 3002, 3003, 3004, 3005}:
           - KFold(5, shuffle=True, random_state=fresh_seed) cross-fit on
             X_unb_K -> residual; OOF residual -> corrected = anchor + r.
           - Refit-all then predict te_K -> te-side residual; te_K_513 =
             te_anchor_513 + te_resid.
         mean_bag_oof_K = mean over 5 seeds
         mean_bag_te_K  = mean over 5 seeds
    5. Equal-weight ensemble:
         pred_oof_253 = 0.5 * mean_bag_oof_K18 + 0.5 * mean_bag_oof_K20
         pred_te_513  = 0.5 * mean_bag_te_K18  + 0.5 * mean_bag_te_K20
    6. 5-fold scaffold CV pooled RAE at kf_seed=1001 (pure evaluation, no
       learning since predictions are fixed).
    7. Compare result against nb2621 0.4552 -> verdict.

OUTPUTS:
    scripts/nb2630_freshseed_k18_k20.py
    data/processed/nb2630_summary.json
    data/processed/nb2630_pred_oof.npy           (253,) float32
    data/processed/te_nb2630.npy                 (513,) float32
    data/processed/nb2630_mean_bag_oof_K18_fresh.npy   (253,) float32
    data/processed/nb2630_mean_bag_oof_K20_fresh.npy   (253,) float32
    data/processed/te_nb2630_K18_fresh.npy             (513,) float32
    data/processed/te_nb2630_K20_fresh.npy             (513,) float32

References:
    nb2621 372-combo grid search    -> {K18, K20} 0.4552 claim
    nb2604 K-equal-weight {18,20,24,28} mean_rae 0.4580 (5 KF seeds)
    nb2240 K=20 deep-30             -> 0.4601 +/- 0.0017 (pyramid wrap)
    nb2263 lucky-seed-aware RFE     -> K=18 standalone 0.4619 mean-bag
    nb2171 PRIMARY-1 ceiling        -> 0.4682 deep-30
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

TAG = "nb2630"

# ---- FRESH seed pool (NEVER used in nb2103/nb2240/nb2310/nb2604/nb2263) ----
FRESH_RESID_SEEDS = [3001, 3002, 3003, 3004, 3005]
RESID_FOLDS = 5

# ---- Anchor + cached eval ----
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# ---- Feature cache paths (identical to nb2604) ----
ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")

EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

# Feature K-grid winner summaries
NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
NB1392_SUMMARY = DATA_PROCESSED / "nb1392_summary.json"
NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"
NB1523_SUMMARY = DATA_PROCESSED / "nb1523_summary.json"
NB1524_SUMMARY = DATA_PROCESSED / "nb1524_summary.json"
NB1541_SUMMARY = DATA_PROCESSED / "nb1541_summary.json"
NB2604_SUMMARY = DATA_PROCESSED / "nb2604_summary.json"
NB2240_SUMMARY = DATA_PROCESSED / "nb2240_summary.json"

# Output paths
K18_FRESH_OOF_PATH = DATA_PROCESSED / "nb2630_mean_bag_oof_K18_fresh.npy"
K18_FRESH_TE_PATH = DATA_PROCESSED / "te_nb2630_K18_fresh.npy"
K20_FRESH_OOF_PATH = DATA_PROCESSED / "nb2630_mean_bag_oof_K20_fresh.npy"
K20_FRESH_TE_PATH = DATA_PROCESSED / "te_nb2630_K20_fresh.npy"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

# ---- CV eval ----
N_FOLDS = 5
KF_SEED = 1001

# ---- Gate ----
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4580

# ---- Refs ----
NB2621_CLAIM = 0.4552
CHEMPROP_AUX_REF = 0.6216
NB2604_REF = 0.4580
NB2171_REF = 0.4682
DEEP_VERIFY_TOL = 0.005


# ============================================================================
# Helpers (lifted from nb2604 -- identical substrate)
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
    """Identical to nb2103/nb2240/nb2604/nb2263."""
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


def build_117col_feature_matrix(te_smiles, n_test):
    """Identical 117-col matrix as nb2103/nb2240/nb2604/nb2263."""
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

    if X_te_full.shape[1] != 117:
        raise ValueError(f"feat_dim {X_te_full.shape[1]} != 117")
    return X_te_full, int(len(pool))


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


def build_K_anchor_freshseed(K_cols, X_te_full, unb_idx, anchor,
                             residual, te_anchor_513, n_test, n_unb,
                             fresh_seeds, label):
    """Rebuild K-anchor mean-bag OOF + te using FRESH residual seeds."""
    X_te_K = X_te_full[:, K_cols].astype(np.float32)
    X_unb_K = X_te_K[unb_idx]
    print(f"   [{label}] X_unb_K={X_unb_K.shape}  X_te_K={X_te_K.shape}")

    per_seed_corr = np.zeros((len(fresh_seeds), n_unb), dtype=np.float64)
    per_seed_te_resid = np.zeros((len(fresh_seeds), n_test), dtype=np.float64)
    per_seed_rae = []
    for i, s in enumerate(fresh_seeds):
        ts = time.time()
        resid_oof = _residual_cross_fit_one_seed(X_unb_K, residual, s)
        per_seed_corr[i] = anchor + resid_oof
        per_seed_rae.append(float(rae(anchor + residual, anchor + resid_oof)))
        te_resid_s = _train_full_then_predict_te(
            X_unb_K, residual, X_te_K, s
        )
        per_seed_te_resid[i] = te_resid_s
        print(f"      {label} fresh_seed={s:4d}: corr_RAE={per_seed_rae[-1]:.4f}  "
              f"wall={time.time()-ts:.1f}s")

    mean_bag_oof_K = per_seed_corr.mean(axis=0)
    mean_bag_te_resid_K = per_seed_te_resid.mean(axis=0)
    te_K_513 = te_anchor_513 + mean_bag_te_resid_K
    return mean_bag_oof_K, te_K_513.astype(np.float32), per_seed_rae


# ============================================================================
# main
# ============================================================================
def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- FRESH-SEED verify of {{K=18, K=20}} equal-weight ensemble")
    print(f"          fresh resid seeds: {FRESH_RESID_SEEDS}")
    print(f"          (replaces canonical pool {{0, 1, 7, 42, 137}})")
    print(f"          nb2621 claim: 0.4552  (from 372-combo grid search)")
    print(f"          tolerance for DEEP_VERIFY_PASS: |delta| <= {DEEP_VERIFY_TOL}")
    print(f"          gates: <{GATE_PROMOTE} PROMOTE / <{GATE_MARGINAL} MARGINAL")
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

    # ---- Step 1: load K=18 + K=20 col indices ----
    print("\n" + "-" * 78)
    print("STEP 1: load K=18 + K=20 col indices from canonical summaries")
    print("-" * 78)
    if not NB2604_SUMMARY.exists():
        raise FileNotFoundError(f"missing {NB2604_SUMMARY}")
    if not NB2240_SUMMARY.exists():
        raise FileNotFoundError(f"missing {NB2240_SUMMARY}")
    with open(NB2604_SUMMARY) as f:
        nb2604 = json.load(f)
    with open(NB2240_SUMMARY) as f:
        nb2240 = json.load(f)
    K18_cols = list(nb2604["k18_idx_in_117col"])
    K20_cols = list(nb2240["k20_surviving_idx_in_117"])
    if len(K18_cols) != 18:
        raise ValueError(f"K18 cols len {len(K18_cols)} != 18")
    if len(K20_cols) != 20:
        raise ValueError(f"K20 cols len {len(K20_cols)} != 20")
    print(f"   K=18 cols (n={len(K18_cols)}): {K18_cols}")
    print(f"   K=20 cols (n={len(K20_cols)}): {K20_cols}")
    overlap_K18_K20 = sorted(set(K18_cols) & set(K20_cols))
    print(f"   overlap K18 & K20: {len(overlap_K18_K20)} cols  {overlap_K18_K20}")

    # ---- Step 2: rebuild 117-col matrix ----
    print("\n" + "-" * 78)
    print("STEP 2: rebuild canonical 117-col feature matrix")
    print("-" * 78)
    X_te_full, chembl_pool_size = build_117col_feature_matrix(te_smiles, n_test)
    print(f"   X_te_full={X_te_full.shape}  chembl_pool={chembl_pool_size}")

    # ---- Step 3: fresh-seed K=18 anchor ----
    print("\n" + "-" * 78)
    print(f"STEP 3: fresh-seed K=18 mean-bag  (seeds {FRESH_RESID_SEEDS})")
    print("-" * 78)
    K18_oof_fresh, K18_te_fresh, K18_per_seed_rae = build_K_anchor_freshseed(
        K18_cols, X_te_full, unb_idx, anchor, residual,
        te_anchor_513, n_test, n_unb, FRESH_RESID_SEEDS, "K18",
    )
    np.save(K18_FRESH_OOF_PATH, K18_oof_fresh.astype(np.float32))
    np.save(K18_FRESH_TE_PATH, K18_te_fresh.astype(np.float32))
    print(f"   [save] {K18_FRESH_OOF_PATH}")
    print(f"   [save] {K18_FRESH_TE_PATH}")
    rae_K18_fresh = float(rae(y_unb, K18_oof_fresh))
    print(f"   K=18 fresh-seed mean-bag RAE = {rae_K18_fresh:.4f}")

    # ---- Step 4: fresh-seed K=20 anchor ----
    print("\n" + "-" * 78)
    print(f"STEP 4: fresh-seed K=20 mean-bag  (seeds {FRESH_RESID_SEEDS})")
    print("-" * 78)
    K20_oof_fresh, K20_te_fresh, K20_per_seed_rae = build_K_anchor_freshseed(
        K20_cols, X_te_full, unb_idx, anchor, residual,
        te_anchor_513, n_test, n_unb, FRESH_RESID_SEEDS, "K20",
    )
    np.save(K20_FRESH_OOF_PATH, K20_oof_fresh.astype(np.float32))
    np.save(K20_FRESH_TE_PATH, K20_te_fresh.astype(np.float32))
    print(f"   [save] {K20_FRESH_OOF_PATH}")
    print(f"   [save] {K20_FRESH_TE_PATH}")
    rae_K20_fresh = float(rae(y_unb, K20_oof_fresh))
    print(f"   K=20 fresh-seed mean-bag RAE = {rae_K20_fresh:.4f}")

    # ---- Step 5: equal-weight {K=18, K=20} ensemble ----
    print("\n" + "-" * 78)
    print("STEP 5: equal-weight mean(K=18_fresh, K=20_fresh)")
    print("-" * 78)
    pred_oof_unb = 0.5 * K18_oof_fresh + 0.5 * K20_oof_fresh.astype(np.float64)
    pred_te_513 = (0.5 * K18_te_fresh.astype(np.float64)
                   + 0.5 * K20_te_fresh.astype(np.float64)).astype(np.float32)
    print(f"   pred_oof_unb std = {pred_oof_unb.std():.4f}  "
          f"(truth_std {y_unb.std():.4f})")
    print(f"   pred_te mean = {pred_te_513.mean():.3f}  std = {pred_te_513.std():.3f}")
    rae_ensemble_singleshot = float(rae(y_unb, pred_oof_unb))
    print(f"   single-shot pooled RAE = {rae_ensemble_singleshot:.4f}")

    # corr of OOFs
    corr_pair = float(np.corrcoef(K18_oof_fresh, K20_oof_fresh)[0, 1])
    print(f"   K18_fresh vs K20_fresh OOF Pearson = {corr_pair:.4f}")

    # ---- Step 6: scaffold CV pooled RAE at kf_seed=1001 ----
    print("\n" + "-" * 78)
    print(f"STEP 6: scaffold {N_FOLDS}-fold CV pooled RAE  kf_seed={KF_SEED}")
    print("-" * 78)
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=KF_SEED,
    )
    oof_pooled = np.full(n_unb, np.nan, dtype=np.float64)
    per_fold_rae = []
    for fi, (tr_loc, va_loc) in enumerate(splits):
        oof_pooled[va_loc] = pred_oof_unb[va_loc]
        fold_r = float(rae(y_unb[va_loc], pred_oof_unb[va_loc]))
        per_fold_rae.append(fold_r)
        print(f"   fold={fi}  n_val={len(va_loc):3d}  RAE={fold_r:.4f}")
    if np.isnan(oof_pooled).any():
        raise RuntimeError("scaffold splits did not cover all rows")
    pooled_rae = float(rae(y_unb, oof_pooled))
    print(f"\n   pooled RAE = {pooled_rae:.4f}")
    print(f"   per-fold mean = {float(np.mean(per_fold_rae)):.4f}  "
          f"std = {float(np.std(per_fold_rae)):.4f}")

    # ---- Step 7: verdict + gate ----
    delta_vs_claim = pooled_rae - NB2621_CLAIM
    print("\n" + "-" * 78)
    print("STEP 7: verdict vs nb2621 claim 0.4552")
    print("-" * 78)
    if abs(delta_vs_claim) <= DEEP_VERIFY_TOL:
        deep_verdict = "DEEP_VERIFY_PASS"
    else:
        deep_verdict = "LUCKY_SEED_TRAP"
    print(f"   |delta| = {abs(delta_vs_claim):.4f}  "
          f"tol = {DEEP_VERIFY_TOL}  -> {deep_verdict}")
    print(f"   delta vs nb2621 claim = {delta_vs_claim:+.4f}")

    if pooled_rae < GATE_PROMOTE:
        gate_verdict = "VERIFIED_PROMOTE"
    elif pooled_rae < GATE_MARGINAL:
        gate_verdict = "VERIFIED_MARGINAL"
    else:
        gate_verdict = "FAILED_VERIFY"
    print(f"   gate verdict: {gate_verdict}  "
          f"(<{GATE_PROMOTE} PROMOTE / <{GATE_MARGINAL} MARGINAL)")

    # ---- Step 8: save artifacts ----
    print("\n" + "-" * 78)
    print("STEP 8: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof_unb.astype(np.float32))
    np.save(te_path, pred_te_513.astype(np.float32))
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_freshseed_k18_k20.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": pred_te_513.astype(np.float32),
    }).to_csv(sub_csv, index=False)
    print(f"   [save] {sub_csv}")

    te_unb_in = float(rae(y_unb, pred_te_513[unb_idx]))
    print(f"\n   te[unb_idx] in-sample RAE = {te_unb_in:.4f}")
    print(f"   delta vs nb2604 ref ({NB2604_REF}) = "
          f"{pooled_rae - NB2604_REF:+.4f}")
    print(f"   delta vs nb2171 ref ({NB2171_REF}) = "
          f"{pooled_rae - NB2171_REF:+.4f}")

    summary = {
        "tag": TAG,
        "method": "freshseed_verify_K18_K20_equal_weight",
        "paradigm": "freshseed_resid_LGBM_no_blend_learning",
        "anchor_base": "chemprop_aux",
        "anchor_in_rae": rae_anchor,
        "anchor_pre_unblind": True,
        "fresh_resid_seeds": FRESH_RESID_SEEDS,
        "canonical_resid_seeds_replaced": [0, 1, 7, 42, 137],
        "resid_folds": RESID_FOLDS,
        "k18_cols_in_117": K18_cols,
        "k20_cols_in_117": K20_cols,
        "k18_k20_overlap_cols": overlap_K18_K20,
        "n_k18_k20_overlap": int(len(overlap_K18_K20)),
        "k18_fresh_oof_path": str(K18_FRESH_OOF_PATH),
        "k18_fresh_te_path": str(K18_FRESH_TE_PATH),
        "k20_fresh_oof_path": str(K20_FRESH_OOF_PATH),
        "k20_fresh_te_path": str(K20_FRESH_TE_PATH),
        "rae_K18_fresh_mean_bag": rae_K18_fresh,
        "rae_K20_fresh_mean_bag": rae_K20_fresh,
        "K18_per_seed_corr_rae": K18_per_seed_rae,
        "K20_per_seed_corr_rae": K20_per_seed_rae,
        "rae_ensemble_singleshot": rae_ensemble_singleshot,
        "K18_K20_oof_pearson": corr_pair,
        "n_folds": N_FOLDS,
        "kf_seed": KF_SEED,
        "scaffold_cv_pooled_rae": pooled_rae,
        "per_fold_rae": per_fold_rae,
        "per_fold_mean": float(np.mean(per_fold_rae)),
        "per_fold_std": float(np.std(per_fold_rae)),
        "nb2621_claim": NB2621_CLAIM,
        "delta_vs_nb2621_claim": delta_vs_claim,
        "deep_verify_tol": DEEP_VERIFY_TOL,
        "deep_verdict": deep_verdict,
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "gate_verdict": gate_verdict,
        "delta_vs_nb2604": pooled_rae - NB2604_REF,
        "delta_vs_nb2171": pooled_rae - NB2171_REF,
        "te_mean": float(pred_te_513.mean()),
        "te_std": float(pred_te_513.std()),
        "te_unb_in_sample_rae": te_unb_in,
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv),
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "chembl_pool_size": chembl_pool_size,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   K=18 fresh mean-bag RAE   = {rae_K18_fresh:.4f}")
    print(f"   K=20 fresh mean-bag RAE   = {rae_K20_fresh:.4f}")
    print(f"   scaffold-CV pooled RAE    = {pooled_rae:.4f}")
    print(f"   nb2621 claim              = {NB2621_CLAIM:.4f}")
    print(f"   delta vs claim            = {delta_vs_claim:+.4f}")
    print(f"   deep verdict              = {deep_verdict}")
    print(f"   gate verdict              = {gate_verdict}")
    print(f"   wall                      = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "fresh_resid_seeds",
        "rae_K18_fresh_mean_bag",
        "rae_K20_fresh_mean_bag",
        "rae_ensemble_singleshot",
        "K18_K20_oof_pearson",
        "scaffold_cv_pooled_rae",
        "nb2621_claim",
        "delta_vs_nb2621_claim",
        "deep_verdict",
        "gate_verdict",
        "delta_vs_nb2604",
        "delta_vs_nb2171",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
