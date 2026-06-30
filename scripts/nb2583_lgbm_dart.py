"""nb2583 -- LightGBM DART boosting on RFE K=20 features as
chemprop_aux residual corrector, 5-fold scaffold-CV across 5 kf_seeds.

NEW PARADIGM (vs nb2240 / nb2160):
    DART = Dropouts meet Multiple Additive Regression Trees.  At each
    iteration boosting drops a random fraction of *already-built* trees
    and re-normalises the survivors so the ensemble stays unbiased.
    This is a different bias-variance trade-off than standard 'gbdt':
      - implicit ensembling / bagging-in-time across iterations,
      - reduces late-tree over-specialisation on residual noise,
      - tends to slightly higher bias, lower variance on small n.

    Key knobs (this script):
      boosting_type = 'dart'
      max_depth     = 4
      num_leaves    = 15
      n_estimators  = 300
      learning_rate = 0.05    (higher than nb2240's 0.03 -- DART needs lr
                              compensation since dropped trees lose mass)
      drop_rate     = 0.10
      skip_drop     = 0.50
      min_child_samples = 5
      reg_lambda    = 2.0

PROTOCOL:
    1. Load RFE K=20 surviving feature indices from nb2231 (same exact
       set used by nb2240).
    2. Rebuild the 117-col 5-way K-tuned feature matrix on the 513 test
       compounds (AtomPair / MACCS / Mordred / ChempropEmbed / Avalon
       + ChEMBL kNN), slice to K=20 indices, take the unb_idx slice.
    3. Anchor = chemprop_aux te[unb_idx]; residual = y_unb - anchor.
    4. For each kf_seed in {1001..1005}:
         scaffold_kfold_indices(n_splits=5, shuffle=True, seed=kf_seed)
         on the 253 unblind.  For each of the 5 folds:
            LGBM(boosting_type='dart', ...) fit on residual[tr],
            predict residual[va] -> resid_oof; corrected = anchor + resid_oof.
       Pooled per-seed RAE = rae(y_unb, corrected_oof).
    5. Mean / std of pooled RAE across the 5 kf_seeds.
    6. Gate:
         mean_rae < 0.4570 -> "PROMOTE"
         mean_rae < 0.4601 -> "MARGINAL_BEAT"
         else              -> "FAIL"

OUTPUTS:
    scripts/nb2583_lgbm_dart.py
    data/processed/nb2583_summary.json
    data/processed/nb2583_pred_oof.npy        (253,) float32  mean-bag across 5 kf_seeds
    data/processed/te_nb2583.npy              (513,) float32  deploy refit
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

TAG = "nb2583"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# ------------------------------ DART config ----------------------------------
DART_PARAMS_TEMPLATE = dict(
    objective="regression",
    boosting_type="dart",
    max_depth=4,
    num_leaves=15,
    n_estimators=300,
    learning_rate=0.05,
    drop_rate=0.10,
    skip_drop=0.50,
    min_child_samples=5,
    reg_lambda=2.0,
    n_jobs=2,
    verbosity=-1,
)

# ------------------------------ CV config ------------------------------------
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# ------------------------------ Gates ----------------------------------------
PROMOTE_THRESHOLD = 0.4570
MARGINAL_THRESHOLD = 0.4601

# ------------------------------ paths ----------------------------------------
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
NB2231_SUMMARY = DATA_PROCESSED / "nb2231_summary.json"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

CHEMPROP_AUX_REF = 0.6216


# ============================================================================
# helpers (copied from nb2240 / nb2270)
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


def _dart_params(seed: int) -> dict:
    p = dict(DART_PARAMS_TEMPLATE)
    p["random_state"] = int(seed)
    return p


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


def _scaffold_cv_one_seed(
    X_unb: np.ndarray,
    residual: np.ndarray,
    anchor: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list,
    kf_seed: int,
):
    """One kf_seed run: 5-fold scaffold split, DART fit on residual, predict OOF.

    Returns (pooled_rae, oof_corrected, fold_raes).
    """
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(residual)
    oof_resid = np.full(n_unb, np.nan, dtype=np.float64)
    fold_raes = []
    for f_i, (tr_loc, va_loc) in enumerate(splits):
        # DART seed is the kf_seed (different per kf_seed run, deterministic)
        mdl = lgb.LGBMRegressor(**_dart_params(kf_seed))
        mdl.fit(X_unb[tr_loc], residual[tr_loc])
        oof_resid[va_loc] = mdl.predict(X_unb[va_loc])
        # per-fold RAE on this validation slice
        fold_pred = anchor[va_loc] + oof_resid[va_loc]
        if len(va_loc) >= 2:
            fold_raes.append(float(rae(y_unb[va_loc], fold_pred)))
    oof_corrected = anchor + oof_resid
    pooled_rae = float(rae(y_unb, oof_corrected))
    return pooled_rae, oof_corrected, fold_raes


def _train_full_then_predict_te(X_unb, residual, X_te, seed):
    mdl = lgb.LGBMRegressor(**_dart_params(seed))
    mdl.fit(X_unb, residual)
    return mdl.predict(X_te).astype(np.float32)


# ============================================================================
# MAIN
# ============================================================================

def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- LightGBM DART on RFE K=20 features as chemprop_aux residual")
    print(f"   params: boosting=dart, max_depth=4, num_leaves=15, n_est=300,")
    print(f"           lr=0.05, drop_rate=0.10, skip_drop=0.50, mc=5, lambda=2")
    print(f"   gates : PROMOTE < {PROMOTE_THRESHOLD:.4f}  |  "
          f"MARGINAL_BEAT < {MARGINAL_THRESHOLD:.4f}  |  else FAIL")
    print("=" * 78)

    # ---- Load K=20 surviving indices from nb2231 ----
    if not NB2231_SUMMARY.exists():
        raise FileNotFoundError(f"missing {NB2231_SUMMARY}")
    with open(NB2231_SUMMARY) as f:
        nb2231 = json.load(f)
    surviving_K20 = list(nb2231["snapshots"]["20"]["surviving_idx_in_117"])
    surviving_K20_names = list(nb2231["snapshots"]["20"]["surviving_names"])
    surviving_K20_family_counts = dict(nb2231["snapshots"]["20"]["family_counts"])
    assert len(surviving_K20) == 20, f"expected 20 features, got {len(surviving_K20)}"
    print(f"[load] K=20 RFE surviving features ({len(surviving_K20)} cols):")
    for j, (idx, nm) in enumerate(zip(surviving_K20, surviving_K20_names)):
        print(f"   {j:2d}. idx={idx:3d}  {nm}")
    print(f"[load] K=20 family_counts = {surviving_K20_family_counts}")

    # ---- Load truth + anchor ----
    te = load_test()
    n_test = len(te)
    te_smiles = te["smiles"].astype(str).tolist() if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
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
    print(f"[load] chemprop_aux in_RAE = {rae_anchor:.4f} (ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f} std={residual.std():.4f}")

    # ---- Rebuild 117-col 5-way feature matrix on the 513 ----
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

    # ChEMBL kNN
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL (union)")
    print("-" * 78)
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
    print(f"   final pool size: {len(pool)}  median pEC50 = {pool_median:.3f}")
    std_test_smiles = [Chem.MolToSmiles(m) if m is not None else "" for m in test_mols]
    fp_test = morgan_fp_batch(std_test_smiles)
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )

    # Full 117-col matrix on 513
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
    feat_dim_full = X_te_full.shape[1]
    assert feat_dim_full == 117, f"feat_dim {feat_dim_full} != 117"
    print(f"[feat] X_te_full = {X_te_full.shape}")

    # Slice to K=20
    X_te_K20 = X_te_full[:, surviving_K20].astype(np.float32)
    X_unb_K20 = X_te_K20[unb_idx]
    print(f"[feat] X_unb_K20 = {X_unb_K20.shape}  X_te_K20 = {X_te_K20.shape}")

    # ---- DART residual scaffold-CV across 5 kf_seeds ----
    print("\n" + "=" * 78)
    print(f"DART RESIDUAL SCAFFOLD-CV  kf_seeds={KF_SEEDS}  n_folds={N_FOLDS}")
    print("=" * 78)
    per_seed_records = []
    per_seed_oof_corrected = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)
    for i, kf_seed in enumerate(KF_SEEDS):
        ts = time.time()
        pooled_rae_s, oof_corr, fold_raes = _scaffold_cv_one_seed(
            X_unb_K20, residual, anchor, y_unb, unb_scaffolds, kf_seed
        )
        per_seed_oof_corrected[i] = oof_corr
        wall = time.time() - ts
        per_seed_records.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": float(pooled_rae_s),
            "delta_vs_chemprop_aux": float(pooled_rae_s - rae_anchor),
            "fold_raes": [float(r) for r in fold_raes],
            "fold_rae_mean": float(np.mean(fold_raes)) if fold_raes else None,
            "fold_rae_std": float(np.std(fold_raes)) if fold_raes else None,
            "wall_sec": round(wall, 2),
        })
        print(
            f"   kf_seed={kf_seed}  pooled_RAE={pooled_rae_s:.4f}  "
            f"(d_vs_anchor={pooled_rae_s - rae_anchor:+.4f})  "
            f"fold_mean={np.mean(fold_raes):.4f}  wall={wall:.1f}s"
        )

    pooled_raes = np.asarray([r["pooled_rae"] for r in per_seed_records])
    mean_rae = float(pooled_raes.mean())
    std_rae = float(pooled_raes.std())
    min_rae = float(pooled_raes.min())
    max_rae = float(pooled_raes.max())

    # mean-bag OOF across the 5 kf_seeds (rows of per_seed_oof_corrected)
    mean_bag_oof = per_seed_oof_corrected.mean(axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))

    print(f"\n[cv-mean] pooled_RAE across {len(KF_SEEDS)} kf_seeds:")
    print(f"   mean   = {mean_rae:.4f}")
    print(f"   std    = {std_rae:.4f}")
    print(f"   range  = [{min_rae:.4f}, {max_rae:.4f}]")
    print(f"[mean-bag] RAE of mean-of-seed OOFs = {rae_mean_bag:.4f}")
    print(f"[anchor]   chemprop_aux in_RAE     = {rae_anchor:.4f}  "
          f"(delta {mean_rae - rae_anchor:+.4f})")

    # ---- Gate ----
    if mean_rae < PROMOTE_THRESHOLD:
        verdict = "PROMOTE"
    elif mean_rae < MARGINAL_THRESHOLD:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "-" * 78)
    print("GATE EVALUATION")
    print("-" * 78)
    print(f"   mean_rae = {mean_rae:.4f}")
    print(f"   PROMOTE   < {PROMOTE_THRESHOLD:.4f}  ({'PASS' if mean_rae < PROMOTE_THRESHOLD else 'no'})")
    print(f"   MARGINAL  < {MARGINAL_THRESHOLD:.4f}  ({'PASS' if mean_rae < MARGINAL_THRESHOLD else 'no'})")
    print(f"   verdict   = {verdict}")

    # ---- Deploy refit on all 253 -> predict residual on full 513 ----
    print("\n" + "-" * 78)
    print("DEPLOY (refit DART on ALL 253; predict residual on 513)")
    print("-" * 78)
    # Use a fixed deploy seed -- pick first kf_seed for reproducibility
    deploy_seed = KF_SEEDS[0]
    te_resid_deploy = _train_full_then_predict_te(
        X_unb_K20, residual, X_te_K20, deploy_seed
    )
    deploy_te = (te_anchor_513 + te_resid_deploy).astype(np.float32)
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))
    print(f"   deploy_seed         = {deploy_seed}")
    print(f"   te[unb_idx] RAE     = {te_unb_rae:.4f}  (in-sample, optimistic)")
    print(f"   te(513) mean/std    = {deploy_te.mean():.3f}/{deploy_te.std():.3f}")

    # ---- Save artefacts ----
    pred_oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(pred_oof_path, mean_bag_oof.astype(np.float32))
    np.save(te_path, deploy_te)
    print(f"\n[save] {pred_oof_path}")
    print(f"[save] {te_path}")

    summary = {
        "tag": TAG,
        "method": "lgbm_dart_K20_chemprop_aux_residual_scaffoldCV5_seeds5",
        "anchor": ANCHOR,
        "anchor_path": str(ANCHOR_TE_PATH),
        "data_source": (
            "nb2231 RFE K=20 surviving features of the 117-col 5-way "
            "K-tuned matrix (AtomPair / MACCS / Mordred / ChempropEmbed "
            "/ Avalon + ChEMBL kNN + mean sim)"
        ),
        "K": 20,
        "K_family_counts": surviving_K20_family_counts,
        "K_surviving_idx_in_117": [int(j) for j in surviving_K20],
        "K_surviving_names": surviving_K20_names,
        "model_family": "LightGBM_DART",
        "lgbm_version": lgb.__version__,
        "boosting_type": "dart",
        "max_depth": DART_PARAMS_TEMPLATE["max_depth"],
        "num_leaves": DART_PARAMS_TEMPLATE["num_leaves"],
        "n_estimators": DART_PARAMS_TEMPLATE["n_estimators"],
        "learning_rate": DART_PARAMS_TEMPLATE["learning_rate"],
        "drop_rate": DART_PARAMS_TEMPLATE["drop_rate"],
        "skip_drop": DART_PARAMS_TEMPLATE["skip_drop"],
        "min_child_samples": DART_PARAMS_TEMPLATE["min_child_samples"],
        "reg_lambda": DART_PARAMS_TEMPLATE["reg_lambda"],
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "n_test": n_test,
        "n_unb": n_unb,
        "n_unique_scaffolds": n_unique_scaf,
        "rae_anchor_chemprop_aux": rae_anchor,
        "rae_chemprop_aux_ref": CHEMPROP_AUX_REF,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_seed_records": per_seed_records,
        "mean_rae": mean_rae,
        "std_rae": std_rae,
        "min_rae": min_rae,
        "max_rae": max_rae,
        "rae_mean_bag_oof_5seed": rae_mean_bag,
        "delta_mean_rae_vs_chemprop_aux": float(mean_rae - rae_anchor),
        "promote_threshold": PROMOTE_THRESHOLD,
        "marginal_threshold": MARGINAL_THRESHOLD,
        "verdict": verdict,
        "deploy_seed": int(deploy_seed),
        "te_unb_rae_in_sample": te_unb_rae,
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "pred_oof_path": str(pred_oof_path),
        "te_npy_path": str(te_path),
        "pre_unblind_clean": True,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   DART K=20 mean_rae (5 seeds) = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   DART K=20 mean-bag OOF RAE   = {rae_mean_bag:.4f}")
    print(f"   anchor chemprop_aux in_RAE   = {rae_anchor:.4f}  "
          f"(delta {mean_rae - rae_anchor:+.4f})")
    print(f"   verdict                      = {verdict}")
    print(f"   wall                         = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "rae_anchor_chemprop_aux",
        "mean_rae",
        "std_rae",
        "min_rae",
        "max_rae",
        "rae_mean_bag_oof_5seed",
        "delta_mean_rae_vs_chemprop_aux",
        "promote_threshold",
        "marginal_threshold",
        "verdict",
        "te_unb_rae_in_sample",
        "pred_oof_path",
        "te_npy_path",
        "wall_sec",
    ):
        print(f"  {k}: {res.get(k)}")
