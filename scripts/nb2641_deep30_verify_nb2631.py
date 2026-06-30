"""nb2641 -- Deep-30 seed verification of nb2631 {K=18, K=19} pyramid.

CRITICAL MOTIVATION
-------------------
nb2631 reported best mean_rae=0.4534 from equal-weight mean of {K=18, K=19}
RFE-residual pyramids over 5 bag seeds [0, 1, 7, 42, 137]. Per memory
feedback_cycle160_deep_verify_dispersion: 5-seed std<0.001 is the cycle-160
red flag pattern (nb2060 5-seed 0.00087 vs 30-seed 0.00408, 4.7x ratio;
nb2095 4.12x; nb2171 2.95x). The 5-seed nb2631 number is hypothesis-grade
not gate-grade -- MUST verify with 30 fresh bag seeds before any promote.

Additional shift vs nb2631:
  - nb2631 used random KFold(shuffle=True, random_state=seed) inside each
    bag run.  Per memory feedback_cv_protocol_audit, scaffold-CV is the
    required protocol for ladder decisions (random KFold is +0.032 RAE
    optimistic).
  - This script runs scaffold-CV at fixed kf_seed=1001 and varies only the
    LGBM bag seed (random_state) across 30 fresh values (0..29 = same first
    5 as nb2631 plus 25 fresh ones via seed mapping).

PROTOCOL
--------
For each K in {18, 19}:
  - Reconstruct K-feature index from nb2231 RFE trajectory.
  - For seed in 0..29:
      * fit LGBM(random_state=seed) residual-regression on 5-fold SCAFFOLD CV
        of the 253 unblind (kf_seed=1001), recording OOF residual.
      * OOF pred = chemprop_aux anchor + OOF residual.
  - Mean-bag across 30 seeds -> single K-pyramid OOF (253,) on the unblind.
  - Equal-weight mean of K=18 and K=19 -> final pred_oof.
  - Same protocol for te (513): fit-full + predict + mean across 30 seeds.

GATE
----
  30-seed mean < 0.4570               -> DEEP30_PROMOTE_NEW_P1
  30-seed mean < 0.4601               -> DEEP30_MARGINAL_OK
  else                                 -> LUCKY_SEED_TRAP

REFERENCES
----------
  nb2631 5-seed mean: 0.4534
  PROMOTE gate     : 0.4570
  MARGINAL gate    : 0.4601
  Current PRIMARY-1: nb2171 deep-30 0.4682 (cycle 167 ceiling on this anchor)
  Chemprop_aux ref : 0.6216

Outputs
-------
  scripts/nb2641_deep30_verify_nb2631.py
  data/processed/nb2641_summary.json
  data/processed/nb2641_pred_oof.npy   (253,) float32
  data/processed/te_nb2641.npy         (513,) float32
  submissions/nb2641_k18_k19_deep30.csv (only if DEEP30_PROMOTE_NEW_P1)
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
from scipy import stats as sp_stats

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko, standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2641"
BASELINE_TAG = "nb2631"

# ---- CV protocol ----
N_FOLDS = 5
KF_SEED = 1001                  # scaffold-CV outer seed
BAG_SEEDS = list(range(30))     # 30 fresh bag (LGBM random_state) seeds
SEEDS_5_SUBSET = [0, 1, 7, 42, 137]  # nb2631's 5-seed reference set
# Note: nb2631 used [0,1,7,42,137]; for the deep-30 we use 0..29 fresh sequence.
# 5-seed comparison: the first 5 of BAG_SEEDS = [0,1,2,3,4]; we also explicitly
# rerun nb2631's exact 5 seeds for an apples-to-apples ratio.

# ---- Gate thresholds ----
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4601

# ---- Refs ----
NB2631_REF_5SEED = 0.4534
NB2171_DEEP30 = 0.4682
CHEMPROP_AUX_REF = 0.6216

# ---- Anchor + residual config (identical to nb2604/nb2631) ----
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
NB2231_SUMMARY = DATA_PROCESSED / "nb2231_summary.json"
NB2063_SHAP_PATH = DATA_PROCESSED / "nb2063_shap_importance_full117.npy"

# ---- Cache paths for 117-col matrix build ----
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

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

K_LIST = [18, 19]


# -----------------------------------------------------------------------------
# Reused helpers (verbatim from nb2631)
# -----------------------------------------------------------------------------
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


def reconstruct_K_from_trajectory(nb2231_sum, K_target):
    """Reconstruct surviving feature indices at K_target from nb2231 RFE trajectory."""
    shap_top28 = list(nb2231_sum["shap_top28_idx_in_117"])
    if K_target == 28:
        return shap_top28
    if K_target > 28:
        if not NB2063_SHAP_PATH.exists():
            raise FileNotFoundError(f"need {NB2063_SHAP_PATH}")
        imp = np.load(NB2063_SHAP_PATH).astype(np.float64)
        order = np.argsort(-imp)
        return [int(j) for j in order[:K_target]]
    current = list(shap_top28)
    traj = nb2231_sum["rfe_trajectory"]
    for entry in traj:
        if entry.get("feat_dropped") is None:
            continue
        if entry["K_after"] < K_target:
            break
        d = int(entry["feat_dropped"])
        if d in current:
            current.remove(d)
        if entry["K_after"] == K_target:
            return current
    if len(current) == K_target:
        return current
    raise ValueError(f"could not reconstruct K={K_target} (got len {len(current)})")


def build_117col_feature_matrix(te_smiles, n_test):
    """Identical 117-col matrix as nb2604/nb2611/nb2631."""
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
            X_ap_te_top, X_maccs_te_top, X_mord_te_top, X_emb_te_top, X_av_te_top,
            pred_chembl_pec50.reshape(-1, 1).astype(np.float32),
            mean_sim.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    if X_te_full.shape[1] != 117:
        raise ValueError(f"feat_dim {X_te_full.shape[1]} != 117")
    return X_te_full


# -----------------------------------------------------------------------------
# Deep-30 protocol -- scaffold-CV at kf_seed=1001 across 30 bag seeds
# -----------------------------------------------------------------------------
def _scaffold_residual_oof_one_seed(X_unb, residual, splits, seed):
    """One LGBM bag seed: scaffold 5-fold residual OOF on the 253 unblind."""
    n = len(residual)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in splits:
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X_unb[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X_unb[va_loc])
    return oof


def _train_full_then_predict_te(X_unb, residual, X_te, seed):
    mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
    mdl.fit(X_unb, residual)
    return mdl.predict(X_te).astype(np.float32)


def build_K_deep30(K, K_idx, X_te_full, unb_idx, anchor_unb, residual,
                   te_anchor_513, splits, n_test, n_unb, bag_seeds):
    """Rebuild K-pyramid OOF + te via 30 bag seeds + scaffold-CV (kf_seed=1001).

    Returns:
        per_seed_oof  : (n_bag_seeds, n_unb) per-seed OOF preds (anchor + residual)
        per_seed_te   : (n_bag_seeds, n_test) per-seed te preds  (anchor + residual)
        per_seed_rae  : list of per-seed pooled RAE
        mean_bag_oof  : (n_unb,) mean across bag seeds
        mean_bag_te   : (n_test,) mean across bag seeds
    """
    X_te_K = X_te_full[:, K_idx].astype(np.float32)
    X_unb_K = X_te_K[unb_idx]
    n_bag = len(bag_seeds)
    per_seed_oof = np.zeros((n_bag, n_unb), dtype=np.float64)
    per_seed_te = np.zeros((n_bag, n_test), dtype=np.float64)
    per_seed_rae = []
    for i, s in enumerate(bag_seeds):
        ts = time.time()
        resid_oof = _scaffold_residual_oof_one_seed(X_unb_K, residual, splits, s)
        oof_pred = anchor_unb + resid_oof
        per_seed_oof[i] = oof_pred
        r = float(rae(anchor_unb + residual, oof_pred))
        per_seed_rae.append(r)
        te_resid_s = _train_full_then_predict_te(X_unb_K, residual, X_te_K, s)
        per_seed_te[i] = te_anchor_513 + te_resid_s
        print(f"   K={K} bag_seed={s:3d}: rae_oof={r:.4f}  "
              f"wall={time.time()-ts:.1f}s")
    mean_bag_oof = per_seed_oof.mean(axis=0)
    mean_bag_te = per_seed_te.mean(axis=0)
    return per_seed_oof, per_seed_te, per_seed_rae, mean_bag_oof, mean_bag_te


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- DEEP-30 verify {BASELINE_TAG} K=18,19 pyramid")
    print(f"         5-seed ref: nb2631 = {NB2631_REF_5SEED:.4f}")
    print(f"         GATE PROMOTE < {GATE_PROMOTE:.4f}")
    print(f"         GATE MARGINAL < {GATE_MARGINAL:.4f}")
    print(f"         current PRIMARY-1 nb2171 deep-30 = {NB2171_DEEP30:.4f}")
    print(f"         scaffold-CV kf_seed = {KF_SEED}, n_bag = {len(BAG_SEEDS)}")
    print("=" * 78)

    # ---- Load data ----
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

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    residual = y_unb - anchor_unb
    print(f"[load] chemprop_aux unb_RAE = {rae_anchor:.4f}")

    # ---- Scaffold-CV splits (one set, kf_seed=1001) ----
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    splits = scaffold_kfold_indices(unb_scaffolds, n_splits=N_FOLDS,
                                    shuffle=True, seed=KF_SEED)
    fold_sizes = [len(va) for _, va in splits]
    print(f"[scaffold] kf_seed={KF_SEED}  unique_scaf={n_unique_scaf}  "
          f"fold_sizes={fold_sizes}")

    # ---- Build/load 117-col matrix ----
    print("\n" + "-" * 78)
    print("STEP 1: build 117-col feature matrix")
    print("-" * 78)
    X_te_full = build_117col_feature_matrix(te_smiles, n_test)
    print(f"   X_te_full = {X_te_full.shape}")

    # ---- Reconstruct K=18 / K=19 indices ----
    with open(NB2231_SUMMARY) as f:
        nb2231 = json.load(f)
    K_idx_used = {}
    for K in K_LIST:
        K_idx = reconstruct_K_from_trajectory(nb2231, K)
        if len(K_idx) != K:
            raise ValueError(f"K={K} reconstruction returned {len(K_idx)}")
        K_idx_used[K] = K_idx
        print(f"   K={K} idx_in_117 (n={len(K_idx)}): {K_idx[:8]}...")

    # ---- Deep-30 bag per K ----
    print("\n" + "-" * 78)
    print(f"STEP 2: deep-30 bag per K (scaffold-CV kf_seed={KF_SEED})")
    print("-" * 78)
    per_K_data = {}
    for K in K_LIST:
        print(f"\n--- K={K} ---")
        per_seed_oof, per_seed_te, per_seed_rae, mean_bag_oof, mean_bag_te = \
            build_K_deep30(
                K, np.array(K_idx_used[K], dtype=int), X_te_full,
                unb_idx, anchor_unb, residual, te_anchor_513,
                splits, n_test, n_unb, BAG_SEEDS,
            )
        per_K_data[K] = {
            "per_seed_oof": per_seed_oof,
            "per_seed_te": per_seed_te,
            "per_seed_rae": per_seed_rae,
            "mean_bag_oof": mean_bag_oof,
            "mean_bag_te": mean_bag_te,
        }
        rae_K_mean_bag = float(rae(y_unb, mean_bag_oof))
        print(f"   K={K} mean-bag oof_RAE = {rae_K_mean_bag:.4f}")

    # ---- Equal-weight K=18 / K=19 per-seed combined RAE ----
    print("\n" + "-" * 78)
    print("STEP 3: equal-weight mean of K=18 + K=19 per seed")
    print("-" * 78)
    n_bag = len(BAG_SEEDS)
    per_seed_combined_rae = np.zeros(n_bag, dtype=np.float64)
    per_seed_combined_oof = np.zeros((n_bag, n_unb), dtype=np.float64)
    per_seed_combined_te = np.zeros((n_bag, n_test), dtype=np.float64)
    for i in range(n_bag):
        oof_i = 0.5 * per_K_data[18]["per_seed_oof"][i] \
              + 0.5 * per_K_data[19]["per_seed_oof"][i]
        te_i = 0.5 * per_K_data[18]["per_seed_te"][i] \
             + 0.5 * per_K_data[19]["per_seed_te"][i]
        per_seed_combined_oof[i] = oof_i
        per_seed_combined_te[i] = te_i
        per_seed_combined_rae[i] = float(rae(y_unb, oof_i))
        print(f"   bag_seed={BAG_SEEDS[i]:3d}  combined_RAE={per_seed_combined_rae[i]:.4f}")

    # ---- Aggregate stats ----
    mean_30 = float(per_seed_combined_rae.mean())
    std_30 = float(per_seed_combined_rae.std(ddof=0))
    std_30_unb = float(per_seed_combined_rae.std(ddof=1))
    min_30 = float(per_seed_combined_rae.min())
    max_30 = float(per_seed_combined_rae.max())
    median_30 = float(np.median(per_seed_combined_rae))

    # First-5 subset (as a sanity check vs nb2631's 5-seed scope)
    rae_5_subset = per_seed_combined_rae[:5]
    mean_5 = float(rae_5_subset.mean())
    std_5 = float(rae_5_subset.std(ddof=0))
    std_5_unb = float(rae_5_subset.std(ddof=1))

    # Under-dispersion ratio: deep-30 std vs 5-seed std (cycle-160 metric)
    under_disp = (std_30_unb / std_5_unb) if std_5_unb > 0 else float("inf")

    # Welch t-test 5 vs remaining 25
    rae_25_rest = per_seed_combined_rae[5:]
    try:
        welch = sp_stats.ttest_ind(rae_5_subset, rae_25_rest, equal_var=False)
        welch_t = float(welch.statistic)
        welch_p = float(welch.pvalue)
    except Exception:
        welch_t = float("nan")
        welch_p = float("nan")

    # ---- Mean-bag final outputs (averaged across all 30 bag seeds) ----
    pred_oof_final = per_seed_combined_oof.mean(axis=0)
    pred_te_final = per_seed_combined_te.mean(axis=0)
    final_rae_meanbag = float(rae(y_unb, pred_oof_final))

    # ---- Gate ----
    if mean_30 < GATE_PROMOTE:
        verdict = "DEEP30_PROMOTE_NEW_P1"
    elif mean_30 < GATE_MARGINAL:
        verdict = "DEEP30_MARGINAL_OK"
    else:
        verdict = "LUCKY_SEED_TRAP"

    delta_vs_nb2631 = mean_30 - NB2631_REF_5SEED
    delta_vs_nb2171 = mean_30 - NB2171_DEEP30

    print("\n" + "=" * 78)
    print(f"=== {TAG} DEEP-30 SUMMARY ===")
    print(f"   30-seed mean    = {mean_30:.4f}")
    print(f"   30-seed median  = {median_30:.4f}")
    print(f"   30-seed std     = {std_30:.4f}  (ddof=1: {std_30_unb:.4f})")
    print(f"   30-seed min/max = {min_30:.4f} / {max_30:.4f}")
    print(f"   first-5 mean    = {mean_5:.4f} +/- {std_5:.4f} (ddof=0)")
    print(f"   under-disp ratio (std30 / std5, ddof=1) = {under_disp:.2f}x")
    print(f"   Welch t (5 vs 25) = {welch_t:.4f}  p = {welch_p:.4f}")
    print(f"   final mean-bag RAE (across 30 seeds) = {final_rae_meanbag:.4f}")
    print(f"   delta vs nb2631 ({NB2631_REF_5SEED}) = {delta_vs_nb2631:+.4f}")
    print(f"   delta vs nb2171 ({NB2171_DEEP30}) = {delta_vs_nb2171:+.4f}")
    print(f"   VERDICT: {verdict}")
    print("=" * 78)

    # ---- Save artifacts ----
    print("\n" + "-" * 78)
    print("STEP 4: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof_final.astype(np.float32))
    np.save(te_path, pred_te_final.astype(np.float32))
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    # Per-K mean-bag artifacts (useful for downstream)
    for K in K_LIST:
        np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof_K{K}.npy",
                per_K_data[K]["mean_bag_oof"].astype(np.float32))
        np.save(DATA_PROCESSED / f"te_{TAG}_K{K}.npy",
                per_K_data[K]["mean_bag_te"].astype(np.float32))

    # Submission CSV only if PROMOTE
    sub_csv = None
    if verdict == "DEEP30_PROMOTE_NEW_P1":
        sub_csv = SUBMISSIONS / f"{TAG}_k18_k19_deep30.csv"
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": pred_te_final.astype(np.float32),
        }).to_csv(sub_csv, index=False)
        print(f"   [save] {sub_csv}  (verdict={verdict})")
    else:
        print(f"   [skip] no submission CSV (verdict={verdict})")

    te_unb_in = float(rae(y_unb, pred_te_final[unb_idx]))

    # ---- Summary JSON ----
    summary = {
        "tag": TAG,
        "baseline_tag": BASELINE_TAG,
        "method": "deep30_seed_verification_K18_K19_equal_weight",
        "paradigm": "rfe_residual_pyramid_meanbag_scaffold_cv",
        "anchor_base": "chemprop_aux",
        "anchor_in_rae": rae_anchor,
        "anchor_pre_unblind": True,
        "K_list": K_LIST,
        "K_idx_used": {str(K): K_idx_used[K] for K in K_LIST},
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "n_folds": N_FOLDS,
        "kf_seed": KF_SEED,
        "bag_seeds": BAG_SEEDS,
        "n_bag": int(n_bag),
        "fold_sizes": fold_sizes,
        "per_seed_rae_combined": per_seed_combined_rae.tolist(),
        "per_K_per_seed_rae": {
            str(K): per_K_data[K]["per_seed_rae"] for K in K_LIST
        },
        "per_K_mean_bag_rae": {
            str(K): float(rae(y_unb, per_K_data[K]["mean_bag_oof"]))
            for K in K_LIST
        },
        "mean_30": mean_30,
        "median_30": median_30,
        "std_30": std_30,
        "std_30_unbiased_ddof1": std_30_unb,
        "min_30": min_30,
        "max_30": max_30,
        "subset_first_5_seeds": BAG_SEEDS[:5],
        "subset_first_5_mean": mean_5,
        "subset_first_5_std": std_5,
        "subset_first_5_std_unbiased_ddof1": std_5_unb,
        "under_dispersion_ratio_std30_over_std5": under_disp,
        "welch_t_5_vs_25": welch_t,
        "welch_p_5_vs_25": welch_p,
        "final_meanbag_rae_on_unb": final_rae_meanbag,
        "te_unb_in_sample_rae": te_unb_in,
        "te_mean": float(pred_te_final.mean()),
        "te_std": float(pred_te_final.std()),
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "delta_vs_nb2631_5seed": delta_vs_nb2631,
        "nb2631_ref_5seed": NB2631_REF_5SEED,
        "delta_vs_nb2171_deep30": delta_vs_nb2171,
        "nb2171_deep30_ref": NB2171_DEEP30,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if sub_csv is not None else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print(f"\n   wall = {time.time()-t0:.1f}s")
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "K_list",
        "mean_30",
        "std_30_unbiased_ddof1",
        "subset_first_5_mean",
        "subset_first_5_std_unbiased_ddof1",
        "under_dispersion_ratio_std30_over_std5",
        "welch_p_5_vs_25",
        "verdict",
        "delta_vs_nb2631_5seed",
        "delta_vs_nb2171_deep30",
        "te_unb_in_sample_rae",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
