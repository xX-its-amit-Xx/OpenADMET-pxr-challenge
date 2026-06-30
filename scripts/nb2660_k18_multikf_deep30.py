"""nb2660 -- multi-kf deep-30 verification of nb2640 K=18 standalone.

CONTEXT (cycle 213 nb2640 K=18 single-kf result):
    nb2640 reported per-K 30-seed bag-mean RAE for K=18 standalone of
    0.4545 at scaffold kf_seed=1001.  This beats the nb2171 deep-30
    PRE-clean ceiling 0.4682 by -0.0137 RAE and is a potential new
    PRIMARY-1.  But nb2640 used only ONE scaffold-CV seed (1001), so the
    0.4545 number is exposed to the same single-kf lucky-seed risk that
    bit nb1086 (5 seeds insufficient to gate Bonferroni).

    The cycle-160 deep-30 rule prevents lucky-seed promotion on the
    bag_seed axis (5-seed under-dispersion 4-5x vs 30-seed) but does not
    address the kf_seed axis.  Different scaffold splits change WHICH
    rows are in which fold of the residual cross-fit, which changes the
    pred_oof vector.  Bag-mean RAE evaluated per-kf is a proper variance
    estimate of the K=18 deploy candidate.

PROTOCOL:
    1.  Rebuild 117-col 5-way feature matrix (identical to nb2640).
    2.  Slice K=18 cols using nb2640_summary['K18_idx_in_117col'].
    3.  For each kf_seed in {1001, 1002, 1003, 1004, 1005}:
          For each bag_seed in {0..29}:
            resid_oof[kf, s] = LGBM_5fold_cv(X_unb_K18, residual,
                                kf_seed=kf, lgbm_seed=s)
            (5-fold scaffold split, NOT random KFold; uses
             `scaffold_kfold_indices` from pxr.eval with seed=kf)
            te_resid[kf, s] = LGBM_full_fit(X_unb_K18, residual,
                              seed=s).predict(X_te_K18)
          bag_mean_pred[kf]    = mean over 30 bag_seeds of (anchor + resid_oof)
          per_kf_pooled[kf]    = rae(y_unb, bag_mean_pred[kf])
          bag_mean_te[kf]      = mean over 30 bag_seeds of (anchor_te + te_resid)
    4.  multikf_mean = mean of per_kf_pooled over 5 kf_seeds
        multikf_std  = std (ddof=1) over 5 kf_seeds
    5.  Deploy:
          pred_oof_unb_deploy = mean over (5 kf_seeds, 30 bag_seeds)
                                 of (anchor + resid_oof)
          pred_te_513_deploy  = mean over (5 kf_seeds, 30 bag_seeds)
                                 of (anchor_te + te_resid)
        Note te_resid only depends on bag_seed (kf_seed does not affect
        full-fit), so averaging over kf is equivalent to averaging 30
        bag_seeds; we keep the explicit double-mean form to make the
        deploy artifact symmetric with the OOF artifact.
    6.  Gate:
          multikf_mean < 0.4570 -> "VERIFIED_PROMOTE_PRIMARY1"
          multikf_mean < 0.4601 -> "VERIFIED_MARGINAL"
          else                  -> "SINGLE_KF_LUCKY"

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/te_chemprop_aux.npy
    data/processed/nb2640_summary.json   (K18_idx_in_117col)
    data/processed/nb1352_summary.json, nb1392_summary.json,
                  nb1484_summary.json, nb1523_summary.json,
                  nb1524_summary.json, nb1541_summary.json
    data/processed/te_atompair.npy, te_maccs.npy,
                  te_chemprop_embed_300.npy, te_avalon512.npy
    C:/pxr_artifacts/nb1030/X_mordred_test.npy

Outputs:
    data/processed/nb2660_summary.json
    data/processed/nb2660_pred_oof.npy   (253,) float32
    data/processed/te_nb2660.npy         (513,) float32
    submissions/nb2660_k18_multikf_deep30.csv
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

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2660"
PARENT_TAG = "nb2640"

# -- Protocol params --------------------------------------------------------
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
BAG_SEEDS = list(range(30))                      # {0,...,29}
RESID_FOLDS = 5                                  # 5-fold scaffold CV

# -- Reference numbers ------------------------------------------------------
NB2640_K18_SINGLEKF = 0.4545                     # nb2640 single-kf=1001 K=18 bag-mean
NB2171_REF = 0.4682                              # prior PRIMARY-1 deep-30 ceiling
CHEMPROP_AUX_REF = 0.6216

# -- Gates -----------------------------------------------------------------
GATE_PROMOTE_PRIMARY1 = 0.4570
GATE_MARGINAL = 0.4601

# -- Feature cache paths --------------------------------------------------
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
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
NB2640_SUMMARY = DATA_PROCESSED / "nb2640_summary.json"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6


# ============================================================================
# helpers (identical to nb2640 / nb2604)
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


def _residual_cross_fit_scaffold(X, residual, scaffolds, kf_seed, lgbm_seed):
    """5-fold scaffold-CV residual prediction.

    kf_seed governs scaffold split (`scaffold_kfold_indices`).
    lgbm_seed governs LGBM RNG.
    """
    n = len(residual)
    splits = scaffold_kfold_indices(
        scaffolds, n_splits=RESID_FOLDS, shuffle=True, seed=kf_seed,
    )
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in splits:
        mdl = lgb.LGBMRegressor(**_lgbm_params(lgbm_seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    if np.isnan(oof).any():
        raise RuntimeError("scaffold split did not cover all rows")
    return oof


def _train_full_then_predict_te(X_unb, residual, X_te, seed):
    mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
    mdl.fit(X_unb, residual)
    return mdl.predict(X_te).astype(np.float32)


def build_117col_feature_matrix(te_smiles, n_test):
    """117-col matrix identical to nb2640 / nb2604 / nb2103."""
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
# main
# ============================================================================

def main():
    import pandas as pd
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- K=18 standalone MULTI-KF deep-30 verify of {PARENT_TAG}")
    print(f"          5 kf_seeds x 30 bag_seeds = 150 LGBM cross-fits")
    print(f"          ref nb2640 single-kf K=18 = {NB2640_K18_SINGLEKF:.4f}")
    print(f"          gate PROMOTE_PRIMARY1 < {GATE_PROMOTE_PRIMARY1} / "
          f"MARGINAL < {GATE_MARGINAL}")
    print("=" * 78)

    # ---- Load truth, anchor, scaffolds ----
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

    # ---- K=18 feature indices ----
    print("\n" + "-" * 78)
    print("STEP 1: load K=18 feature indices from nb2640_summary.json")
    print("-" * 78)
    with open(NB2640_SUMMARY) as f:
        nb2640 = json.load(f)
    K18_idx = np.array(nb2640["K18_idx_in_117col"], dtype=int)
    if len(K18_idx) != 18:
        raise RuntimeError(f"K18 idx len {len(K18_idx)} != 18")
    print(f"   K=18 idx (n={len(K18_idx)}): {K18_idx.tolist()}")

    # ---- 117-col matrix ----
    print("\n" + "-" * 78)
    print("STEP 2: rebuild 117-col 5-way feature matrix")
    print("-" * 78)
    X_te_full, chembl_pool_size = build_117col_feature_matrix(te_smiles, n_test)
    print(f"   X_te_full = {X_te_full.shape}  chembl_pool={chembl_pool_size}")
    X_te_K = X_te_full[:, K18_idx].astype(np.float32)
    X_unb_K = X_te_K[unb_idx]
    print(f"   X_unb_K = {X_unb_K.shape}  X_te_K = {X_te_K.shape}")

    # ---- Multi-kf x deep-30 residual cross-fit ----
    print("\n" + "-" * 78)
    print(f"STEP 3: residual LGBM  5 kf_seeds x 30 bag_seeds = "
          f"{len(KF_SEEDS) * len(BAG_SEEDS)} cross-fits")
    print("-" * 78)

    # (n_kf, n_bag, n_unb)  pred_unb  = anchor + resid_oof
    pred_unb_all = np.zeros(
        (len(KF_SEEDS), len(BAG_SEEDS), n_unb), dtype=np.float64
    )
    # (n_bag, n_te)  pred_te (kf_seed does NOT affect te full-fit)
    pred_te_per_bag = np.zeros((len(BAG_SEEDS), n_test), dtype=np.float64)
    per_seed_te_done = np.zeros(len(BAG_SEEDS), dtype=bool)

    per_kf_bagmean_rae = []
    per_kf_per_seed_corr_rae = {}
    total = len(KF_SEEDS) * len(BAG_SEEDS)
    done = 0
    for k_i, kf_seed in enumerate(KF_SEEDS):
        print(f"\n  --- kf_seed={kf_seed} -----------------------------------")
        per_seed_corr_rae = []
        for b_i, b_seed in enumerate(BAG_SEEDS):
            ts = time.time()
            resid_oof = _residual_cross_fit_scaffold(
                X_unb_K, residual, unb_scaffolds, kf_seed=kf_seed, lgbm_seed=b_seed,
            )
            pred_unb_all[k_i, b_i] = anchor + resid_oof
            per_seed_corr_rae.append(
                float(rae(anchor + residual, anchor + resid_oof))
            )
            # te resid only needs to be computed once per bag_seed
            if not per_seed_te_done[b_i]:
                te_resid = _train_full_then_predict_te(
                    X_unb_K, residual, X_te_K, seed=b_seed,
                )
                pred_te_per_bag[b_i] = te_anchor_513 + te_resid
                per_seed_te_done[b_i] = True
            done += 1
            if (b_i % 10) == 0 or b_i == len(BAG_SEEDS) - 1:
                print(f"      kf={kf_seed} bag={b_seed:3d}  "
                      f"rae_corr={per_seed_corr_rae[-1]:.4f}  "
                      f"wall={time.time()-ts:.2f}s  "
                      f"({done}/{total})")
        bagmean_pred = pred_unb_all[k_i].mean(axis=0)
        bagmean_rae = float(rae(y_unb, bagmean_pred))
        per_kf_bagmean_rae.append(bagmean_rae)
        per_kf_per_seed_corr_rae[str(kf_seed)] = per_seed_corr_rae
        print(f"   kf_seed={kf_seed}  per-bag-seed corr RAE  "
              f"mean={np.mean(per_seed_corr_rae):.4f}  "
              f"std={np.std(per_seed_corr_rae, ddof=1):.4f}")
        print(f"   kf_seed={kf_seed}  30-bag-mean POOLED RAE = "
              f"{bagmean_rae:.4f}")

    per_kf_arr = np.array(per_kf_bagmean_rae, dtype=np.float64)
    multikf_mean = float(per_kf_arr.mean())
    multikf_std = float(per_kf_arr.std(ddof=1))
    multikf_min = float(per_kf_arr.min())
    multikf_max = float(per_kf_arr.max())
    multikf_median = float(np.median(per_kf_arr))
    sem = multikf_std / np.sqrt(len(KF_SEEDS))
    t_crit = float(student_t.ppf(0.975, df=len(KF_SEEDS) - 1))
    ci_low = multikf_mean - t_crit * sem
    ci_high = multikf_mean + t_crit * sem

    print("\n" + "-" * 78)
    print("STEP 4: multi-kf statistics")
    print("-" * 78)
    print(f"   per-kf 30-bag-mean RAE:")
    for kf, r in zip(KF_SEEDS, per_kf_bagmean_rae):
        print(f"     kf={kf}  RAE={r:.4f}")
    print(f"\n   multikf mean +/- std  = {multikf_mean:.4f} +/- {multikf_std:.5f}")
    print(f"   multikf median        = {multikf_median:.4f}")
    print(f"   multikf min / max     = {multikf_min:.4f} / {multikf_max:.4f}")
    print(f"   95% CI on mean        = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   nb2640 single-kf K=18 = {NB2640_K18_SINGLEKF:.4f}")
    print(f"   shift (multikf - nb2640) = "
          f"{multikf_mean - NB2640_K18_SINGLEKF:+.4f}")
    print(f"   delta vs nb2171 (0.4682) = "
          f"{multikf_mean - NB2171_REF:+.4f}")

    # ---- Deploy: bag-mean over all (kf, bag) on OOF; bag-mean over 30 on te ----
    print("\n" + "-" * 78)
    print("STEP 5: deploy artifacts")
    print("-" * 78)
    # OOF: average over (5 kf, 30 bag) = 150 entries
    pred_oof_unb = pred_unb_all.reshape(-1, n_unb).mean(axis=0)
    # te only varies with bag (not kf), so just average 30 bag-seed te-arrays
    pred_te_513 = pred_te_per_bag.mean(axis=0)
    deploy_pooled_rae = float(rae(y_unb, pred_oof_unb))
    te_unb_in_rae = float(rae(y_unb, pred_te_513[unb_idx]))
    print(f"   deploy bag-mean (5kf x 30bag) OOF pooled RAE = "
          f"{deploy_pooled_rae:.4f}")
    print(f"   te[unb_idx] in-sample RAE                    = "
          f"{te_unb_in_rae:.4f}")
    print(f"   pred_oof_unb std = {pred_oof_unb.std():.3f} "
          f"(truth_std {y_unb.std():.3f})")
    print(f"   pred_te_513  mean / std = {pred_te_513.mean():.3f} / "
          f"{pred_te_513.std():.3f}")

    # ---- Gate ----
    print("\n" + "-" * 78)
    print("STEP 6: GATE")
    print("-" * 78)
    if multikf_mean < GATE_PROMOTE_PRIMARY1:
        verdict = "VERIFIED_PROMOTE_PRIMARY1"
    elif multikf_mean < GATE_MARGINAL:
        verdict = "VERIFIED_MARGINAL"
    else:
        verdict = "SINGLE_KF_LUCKY"
    print(f"   multikf mean = {multikf_mean:.4f}")
    print(f"     <{GATE_PROMOTE_PRIMARY1} -> VERIFIED_PROMOTE_PRIMARY1")
    print(f"     <{GATE_MARGINAL} -> VERIFIED_MARGINAL")
    print(f"     else            -> SINGLE_KF_LUCKY")
    print(f"   -> {verdict}")

    # ---- Save artifacts ----
    print("\n" + "-" * 78)
    print("STEP 7: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof_unb.astype(np.float32))
    np.save(te_path, pred_te_513.astype(np.float32))
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_k18_multikf_deep30.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": pred_te_513.astype(np.float32),
    }).to_csv(sub_csv, index=False)
    print(f"   [save] {sub_csv}")

    summary = {
        "tag": TAG,
        "parent_tag": PARENT_TAG,
        "method": "k18_standalone_multikf_x_deep30",
        "rationale": "verify nb2640 single-kf K=18 0.4545 against 5 kf_seeds",
        "anchor_base": "chemprop_aux",
        "anchor_in_rae": rae_anchor,
        "anchor_pre_unblind": True,
        "K": 18,
        "K18_idx_in_117col": K18_idx.tolist(),
        "kf_seeds": KF_SEEDS,
        "bag_seeds": BAG_SEEDS,
        "n_kf": len(KF_SEEDS),
        "n_bag": len(BAG_SEEDS),
        "n_total_fits": len(KF_SEEDS) * len(BAG_SEEDS),
        "resid_folds": RESID_FOLDS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "per_kf_30bag_mean_rae": per_kf_bagmean_rae,
        "per_kf_per_seed_corr_rae": per_kf_per_seed_corr_rae,
        "multikf_mean_rae": multikf_mean,
        "multikf_std_rae": multikf_std,
        "multikf_median_rae": multikf_median,
        "multikf_min_rae": multikf_min,
        "multikf_max_rae": multikf_max,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "sem": float(sem),
        "t_crit_df4": t_crit,
        "deploy_pooled_rae_5kf_30bag_mean": deploy_pooled_rae,
        "te_unb_in_sample_rae": te_unb_in_rae,
        "te_mean": float(pred_te_513.mean()),
        "te_std": float(pred_te_513.std()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv),
        "nb2640_singlekf_K18_ref": NB2640_K18_SINGLEKF,
        "shift_vs_nb2640_singlekf": multikf_mean - NB2640_K18_SINGLEKF,
        "nb2171_ref": NB2171_REF,
        "delta_vs_nb2171": multikf_mean - NB2171_REF,
        "gate_promote_primary1": GATE_PROMOTE_PRIMARY1,
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
    print(f"   nb2640 single-kf K=18 ref  = {NB2640_K18_SINGLEKF:.4f}")
    print(f"   multikf mean +/- std       = "
          f"{multikf_mean:.4f} +/- {multikf_std:.5f}")
    print(f"   per-kf bag-means           = "
          + ", ".join(f"{r:.4f}" for r in per_kf_bagmean_rae))
    print(f"   95% CI                     = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   shift (multikf - singlekf) = "
          f"{multikf_mean - NB2640_K18_SINGLEKF:+.4f}")
    print(f"   delta vs nb2171 (0.4682)   = "
          f"{multikf_mean - NB2171_REF:+.4f}")
    print(f"   deploy bag-mean RAE        = {deploy_pooled_rae:.4f}")
    print(f"   te[unb_idx] RAE            = {te_unb_in_rae:.4f}")
    print(f"   verdict                    = {verdict}")
    print(f"   wall                       = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "multikf_mean_rae",
        "multikf_std_rae",
        "ci95_low",
        "ci95_high",
        "shift_vs_nb2640_singlekf",
        "deploy_pooled_rae_5kf_30bag_mean",
        "te_unb_in_sample_rae",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
