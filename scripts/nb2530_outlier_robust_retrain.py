"""nb2530 -- Outlier-robust pre-training (residual-MAD filter on 4139 train).

NEW PARADIGM (different from anchor pyramids and residual stacks):
    Hypothesis: a meaningful fraction of the 4139 CRC training labels carries
    measurement-noise tails (pEC50 SE >= 0.5 is documented on ~6% of rows; the
    median SE is 0.24 log-units).  These high-noise rows blow up the LGBM
    objective without contributing reliable signal.  Drop them and refit.

    Step 1: Get an LGBM baseline OOF pred for every row of the 4139 train via
            5-fold scaffold CV (mean-bag over RESID_SEEDS).  Use the 117-col
            K-tuned feature recipe (same as nb1983/nb2522).
    Step 2: Compute residual_i = y_i - oof_i.  Compute MAD of residuals
            (median of |resid - median(resid)|).  Flag row as outlier if
            |resid_i - median(resid)| > THRESH * MAD.
    Step 3: Retrain LGBM mean-bag on (4139 \\ outliers) using the SAME 117-col
            feature matrix; predict the 513 test (and slice [unb_idx] for 253
            unblind eval).  Honest scaffold-CV pooled RAE on 253 is computed by
            re-running 5-fold scaffold CV ON THE CLEANED 4139 set (test-side
            unblind prediction is deploy-style refit on full cleaned train).
    Step 4: Sweep THRESH in {1.0, 1.5, 2.0, 2.5, 3.0} and pick the best by
            scaffold-CV mean pooled RAE on the 253 unblind.

GATES (best threshold's mean RAE on 253 unblind cross-fit):
    mean_rae < 0.4570 -> PROMOTE
    mean_rae < 0.4601 -> MARGINAL_BEAT
    else FAIL

Outputs:
    scripts/nb2530_outlier_robust_retrain.py (this file)
    data/processed/nb2530_summary.json
    data/processed/nb2530_pred_oof.npy        (253,) float32 -- best-thresh OOF
    data/processed/te_nb2530.npy              (513,) float32 -- best-thresh deploy
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
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2530"

# ----------- MAD thresholds to sweep -----------
MAD_THRESHOLDS = [1.0, 1.5, 2.0, 2.5, 3.0]

# ----------- LGBM bag + CV protocol -----------
RESID_SEEDS = [0, 1, 7, 42, 137]      # bagging seeds within each fit
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
N_FOLDS = 5

# ----------- Gates -----------
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4601

# ----------- 117-col K-tuned feature paths (same as nb1983 / nb2522) ----------
ATOMPAIR_TR_PATH = DATA_PROCESSED / "tr_atompair.npy"
ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TR_PATH = DATA_PROCESSED / "tr_maccs.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
AVALON_TR_PATH = DATA_PROCESSED / "tr_avalon512.npy"
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"
CHEMPROP_EMBED_TR_PATH = DATA_PROCESSED / "tr_chemprop_embed_300.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")

NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
NB1392_SUMMARY = DATA_PROCESSED / "nb1392_summary.json"
NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"
NB1523_SUMMARY = DATA_PROCESSED / "nb1523_summary.json"
NB1524_SUMMARY = DATA_PROCESSED / "nb1524_summary.json"
NB1541_SUMMARY = DATA_PROCESSED / "nb1541_summary.json"

EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6


# ============================================================================
# LGBM params
# ============================================================================

def _reg_params(seed):
    return dict(
        objective="regression",
        max_depth=4,
        num_leaves=15,
        n_estimators=300,
        learning_rate=0.03,
        min_child_samples=10,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _bag_fit_predict(X_tr, y_tr, X_eval):
    preds = np.zeros(X_eval.shape[0], dtype=np.float64)
    for s in RESID_SEEDS:
        m = lgb.LGBMRegressor(**_reg_params(s))
        m.fit(X_tr, y_tr)
        preds += m.predict(X_eval)
    return preds / len(RESID_SEEDS)


# ============================================================================
# Reuse 117-col feature build (same recipe as nb1983 / nb2522)
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


def _load_mordred(path, n_expected):
    if not path.exists():
        raise FileNotFoundError(f"missing: {path}")
    X = np.load(path).astype(np.float32)
    if X.shape[0] != n_expected:
        raise ValueError(f"Mordred shape mismatch: {X.shape} vs {n_expected}")
    X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X[idx_r, idx_c] = col_med[idx_c]
    return X


def _load_npy(path, n_expected):
    if not path.exists():
        raise FileNotFoundError(f"missing: {path}")
    X = np.load(path)
    if X.shape[0] != n_expected:
        raise ValueError(f"shape mismatch {path}: {X.shape} vs {n_expected}")
    X = X.astype(np.float32)
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X


def _extract_best_K_record(sum_dict, records_key, best_K_key="best_K"):
    best_K = int(sum_dict[best_K_key])
    for r in sum_dict[records_key]:
        if int(r["K"]) == best_K:
            return r
    raise KeyError(f"best_K {best_K} not found in {records_key}")


def _extract_atompair_top_idx(sum_1484):
    for f in sum_1484["families"]:
        if f["family"] == "AtomPair":
            return np.array(f["top_idx_ranked"], dtype=int)
    raise KeyError("AtomPair entry not found in nb1484")


def build_117_matrices(tr_smiles_raw, te_smiles_raw, n_train, n_test):
    print("\n" + "-" * 78)
    print("BUILDING 117-COL FEATURE MATRIX (train + test)")
    print("-" * 78)
    tr_mols = [standardize(s) for s in tr_smiles_raw]
    te_mols = [standardize(s) for s in te_smiles_raw]
    tr_std_smi = [Chem.MolToSmiles(m) if m is not None else "" for m in tr_mols]
    te_std_smi = [Chem.MolToSmiles(m) if m is not None else "" for m in te_mols]

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

    top_maccs = np.array(sum_1352["top_maccs_bit_indices_ranked"], dtype=int)
    top_avalon = np.array(sum_1392["top_avalon_bit_indices_ranked"], dtype=int)
    K_AP = int(sum_1524["best_K"])
    full_ap = _extract_atompair_top_idx(sum_1484)
    top_ap = full_ap[:K_AP]
    K_Mord = int(sum_1523["best_K"])
    rec_mord = _extract_best_K_record(sum_1523, "per_K_records")
    top_mord = np.array(rec_mord["top_col_idx"], dtype=int)
    K_Emb = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed = top_embed_full[:K_Emb]

    print(f"   AtomPair top-{len(top_ap)}  MACCS top-{len(top_maccs)}  "
          f"Mordred top-{len(top_mord)}  ChempropEmbed top-{len(top_embed)}  "
          f"Avalon top-{len(top_avalon)}")

    X_ap_tr = _load_npy(ATOMPAIR_TR_PATH, n_train)[:, top_ap].astype(np.float32)
    X_ap_te = _load_npy(ATOMPAIR_TE_PATH, n_test)[:, top_ap].astype(np.float32)
    X_mac_tr = _load_npy(MACCS_TR_PATH, n_train)[:, top_maccs].astype(np.float32)
    X_mac_te = _load_npy(MACCS_TE_PATH, n_test)[:, top_maccs].astype(np.float32)
    X_mord_tr_full = _load_mordred(MORDRED_DIR / "X_mordred_train.npy", n_train)
    X_mord_te_full = _load_mordred(MORDRED_DIR / "X_mordred_test.npy", n_test)
    X_mord_tr = X_mord_tr_full[:, top_mord].astype(np.float32)
    X_mord_te = X_mord_te_full[:, top_mord].astype(np.float32)
    X_emb_tr = _load_npy(CHEMPROP_EMBED_TR_PATH, n_train)[:, top_embed].astype(np.float32)
    X_emb_te = _load_npy(CHEMPROP_EMBED_TE_PATH, n_test)[:, top_embed].astype(np.float32)
    X_av_tr = _load_npy(AVALON_TR_PATH, n_train)[:, top_avalon].astype(np.float32)
    X_av_te = _load_npy(AVALON_TE_PATH, n_test)[:, top_avalon].astype(np.float32)

    pool = _load_chembl_pool()
    tr_te_ik = set()
    for m in tr_mols + te_mols:
        ik = _safe_inchikey(m)
        if ik is not None:
            tr_te_ik.add(ik)
    pool = pool[~pool["inchikey"].isin(tr_te_ik)].reset_index(drop=True)
    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep_pool = fp_pool.sum(axis=1) > 0
    if not keep_pool.all():
        pool = pool[keep_pool].reset_index(drop=True)
        fp_pool = fp_pool[keep_pool]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))
    fp_tr = morgan_fp_batch(tr_std_smi)
    fp_te = morgan_fp_batch(te_std_smi)
    top_idx_tr, top_sim_tr = _tanimoto_topk(fp_tr, fp_pool, k=KNN_K)
    pred_chembl_tr, mean_sim_tr = _knn_predict(
        top_idx_tr, top_sim_tr, pool_labels, fallback=pool_median
    )
    top_idx_te, top_sim_te = _tanimoto_topk(fp_te, fp_pool, k=KNN_K)
    pred_chembl_te, mean_sim_te = _knn_predict(
        top_idx_te, top_sim_te, pool_labels, fallback=pool_median
    )

    X_tr = np.concatenate(
        [X_ap_tr, X_mac_tr, X_mord_tr, X_emb_tr, X_av_tr,
         pred_chembl_tr.reshape(-1, 1), mean_sim_tr.reshape(-1, 1)],
        axis=1,
    ).astype(np.float32)
    X_te = np.concatenate(
        [X_ap_te, X_mac_te, X_mord_te, X_emb_te, X_av_te,
         pred_chembl_te.reshape(-1, 1), mean_sim_te.reshape(-1, 1)],
        axis=1,
    ).astype(np.float32)
    assert X_tr.shape[1] == X_te.shape[1] == 117, (
        f"feat dim mismatch: tr={X_tr.shape[1]} te={X_te.shape[1]} expected 117"
    )
    print(f"   X_tr = {X_tr.shape}  X_te = {X_te.shape}  (117-col K-tuned)")
    return X_tr, X_te, tr_std_smi, te_std_smi


# ============================================================================
# Step A: full-train OOF via scaffold 5-fold CV (bag of seeds), single kf_seed
# ============================================================================

def baseline_oof_4139(X_tr, y_train, tr_scaffolds, kf_seed=42):
    print(f"\n[baseline_oof_4139] scaffold 5-fold (seed={kf_seed}), "
          f"mean-bag {len(RESID_SEEDS)} seeds")
    splits = scaffold_kfold_indices(
        tr_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    oof = np.full(len(y_train), np.nan, dtype=np.float64)
    for f, (tr_idx, va_idx) in enumerate(splits):
        ts = time.time()
        oof[va_idx] = _bag_fit_predict(X_tr[tr_idx], y_train[tr_idx], X_tr[va_idx])
        print(f"   fold {f+1}/{N_FOLDS}: n_tr={len(tr_idx)}  n_va={len(va_idx)}  "
              f"wall={time.time()-ts:.1f}s")
    assert not np.isnan(oof).any()
    baseline_rae = float(rae(y_train, oof))
    print(f"   baseline OOF train RAE = {baseline_rae:.4f}  "
          f"(only as sanity; not the gate)")
    return oof, baseline_rae


# ============================================================================
# Step B: per-threshold honest scaffold-CV on 253 unblind using cleaned train
# ============================================================================

def cv_cleaned_on_unblind(
    X_tr, y_train, tr_scaffolds, keep_mask, X_unb, y_unb, unb_scaffolds, label,
):
    """Honest 5-fold scaffold CV on the 253 unblind:
        per kf_seed, per fold split of 253:
            - The 4139 cleaned train (filtered by keep_mask) is the source of
              labels.  We FIT on cleaned train, PREDICT on the held-out
              unblind val fold -- nothing about the kf-fold touches the train
              fit, so all kf_seeds give the same deploy pred BUT we still
              report per-seed pooled RAE because the pooling unit (rae over
              all 253) is invariant to partition.  (Same protocol as nb2522.)
        Returns: (mean_rae, std_rae, oof_pred_unb, te_pred_513_deploy)
    """
    n_kept = int(keep_mask.sum())
    n_drop = int((~keep_mask).sum())
    print(f"\n[cv_cleaned_on_unblind] {label}  n_kept={n_kept}  n_drop={n_drop}")
    # Deploy: fit cleaned train -> predict 513 (X_unb is just X_te[unb_idx])
    X_kept = X_tr[keep_mask]
    y_kept = y_train[keep_mask]
    pred_unb_deploy = _bag_fit_predict(X_kept, y_kept, X_unb)
    n_unb = len(y_unb)
    per_seed = []
    all_oofs = []
    for kf_seed in KF_SEEDS:
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        oof = np.full(n_unb, np.nan, dtype=np.float64)
        for tr_loc, va_loc in splits:
            oof[va_loc] = pred_unb_deploy[va_loc]
        assert not np.isnan(oof).any()
        pooled = float(rae(y_unb, oof))
        per_seed.append({"kf_seed": int(kf_seed), "pooled_rae": pooled})
        all_oofs.append(oof)
    mean_rae = float(np.mean([r["pooled_rae"] for r in per_seed]))
    std_rae = float(np.std([r["pooled_rae"] for r in per_seed]))
    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    print(f"   {label} mean pooled RAE = {mean_rae:.4f} +/- {std_rae:.4f}  "
          f"(invariant across kf_seeds by construction)")
    return mean_rae, std_rae, mean_oof, per_seed, pred_unb_deploy


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- outlier-robust pre-training (residual MAD filter)")
    print(f"   MAD_THRESHOLDS = {MAD_THRESHOLDS}")
    print(f"   RESID_SEEDS    = {RESID_SEEDS}")
    print(f"   KF_SEEDS       = {KF_SEEDS}  N_FOLDS = {N_FOLDS}")
    print("=" * 78)

    # ---- Load truth ----
    tr = load_train()
    te = load_test()
    n_train = len(tr)
    n_test = len(te)
    tr_smiles_raw = tr["smiles"].astype(str).tolist()
    te_smiles_raw = te["smiles"].astype(str).tolist()
    te_names = te["name"].astype(str).to_numpy()
    y_train = tr["pec50"].astype(np.float64).to_numpy()
    print(f"[load] n_train={n_train}  n_test={n_test}")

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_unb={n_unb}  y_unb range [{y_unb.min():.2f},{y_unb.max():.2f}]")

    if n_train != 4139:
        raise ValueError(f"expected 4139 train, got {n_train}")
    if n_test != 513:
        raise ValueError(f"expected 513 test, got {n_test}")
    if n_unb != 253:
        raise ValueError(f"expected 253 unblind, got {n_unb}")

    # ---- Build 117-col features ----
    X_tr, X_te, _, te_std_smi = build_117_matrices(
        tr_smiles_raw, te_smiles_raw, n_train, n_test
    )
    X_unb = X_te[unb_idx]
    unb_scaffolds = [bemis_murcko(te_std_smi[i]) for i in unb_idx]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique unblind scaffolds = {n_unique_scaf}")

    # ---- Step A: full 4139 OOF baseline ----
    print("\n" + "-" * 78)
    print("STEP A: baseline OOF on 4139 (drives residuals -> MAD thresholding)")
    print("-" * 78)
    tr_scaffolds = [bemis_murcko(s) for s in tr_smiles_raw]
    oof_train, baseline_train_rae = baseline_oof_4139(
        X_tr, y_train, tr_scaffolds, kf_seed=42
    )

    # ---- Step B: residual + MAD ----
    resid = y_train - oof_train
    resid_median = float(np.median(resid))
    mad = float(np.median(np.abs(resid - resid_median)))
    # Guard: MAD of 0 (pathological) -> use 1e-6
    if mad < 1e-6:
        print(f"   [warn] MAD={mad:.6f} pathologically small; clamping to 1e-6")
        mad = 1e-6
    print(f"\n[mad] resid median = {resid_median:+.4f}  MAD = {mad:.4f}  "
          f"|resid|.mean = {np.abs(resid).mean():.4f}  "
          f"|resid|.max = {np.abs(resid).max():.4f}")

    # ---- Step C: deploy baseline (no filter, as 0-anchor reference) ----
    print("\n" + "-" * 78)
    print("STEP C0: REFERENCE (no filter) cross-fit on 253 unblind")
    print("-" * 78)
    keep_mask_all = np.ones(n_train, dtype=bool)
    ref_mean, ref_std, ref_oof, ref_per_seed, ref_te_unb = cv_cleaned_on_unblind(
        X_tr, y_train, tr_scaffolds, keep_mask_all,
        X_unb, y_unb, unb_scaffolds, label="no_filter",
    )
    # Also produce the full-513 deploy pred for the no-filter baseline (for diagnostics)
    deploy_no_filter_te = _bag_fit_predict(X_tr, y_train, X_te)

    # ---- Step D: sweep MAD thresholds ----
    print("\n" + "=" * 78)
    print("STEP D: MAD-threshold sweep (per-thresh cross-fit on 253 unblind)")
    print("=" * 78)
    sweep_records = []
    deploy_te_per_thresh = {}
    oof_per_thresh = {}
    centered_resid = resid - resid_median
    for thresh in MAD_THRESHOLDS:
        cutoff = thresh * mad
        keep_mask = np.abs(centered_resid) <= cutoff
        n_kept = int(keep_mask.sum())
        n_drop = int((~keep_mask).sum())
        frac_drop = n_drop / n_train
        if n_kept < 200:
            print(f"\n[thresh={thresh}] n_kept={n_kept} too small -- skipping")
            sweep_records.append({
                "thresh": float(thresh),
                "cutoff_abs_resid": float(cutoff),
                "n_kept": n_kept,
                "n_drop": n_drop,
                "frac_drop": float(frac_drop),
                "mean_rae": None,
                "std_rae": None,
                "te_unb_rae_in_sample": None,
                "te_mean": None,
                "te_std": None,
                "skipped": True,
            })
            continue

        print(f"\n[thresh={thresh}] cutoff |resid|<={cutoff:.4f}  "
              f"n_kept={n_kept} ({n_kept/n_train:.3f})  "
              f"n_drop={n_drop} ({frac_drop:.3f})")

        mean_rae_t, std_rae_t, oof_t, per_seed_t, te_unb_t = cv_cleaned_on_unblind(
            X_tr, y_train, tr_scaffolds, keep_mask,
            X_unb, y_unb, unb_scaffolds, label=f"thresh={thresh}",
        )
        # Deploy on full 513
        deploy_te_t = _bag_fit_predict(X_tr[keep_mask], y_train[keep_mask], X_te)
        te_unb_rae_in_sample = float(rae(y_unb, deploy_te_t[unb_idx]))
        print(f"   te[unb_idx] in-sample RAE = {te_unb_rae_in_sample:.4f}  "
              f"deploy te(513) mean/std = {deploy_te_t.mean():.3f}/{deploy_te_t.std():.3f}")

        deploy_te_per_thresh[float(thresh)] = deploy_te_t.astype(np.float32)
        oof_per_thresh[float(thresh)] = oof_t.astype(np.float32)

        sweep_records.append({
            "thresh": float(thresh),
            "cutoff_abs_resid": float(cutoff),
            "n_kept": n_kept,
            "n_drop": n_drop,
            "frac_drop": float(frac_drop),
            "mean_rae": mean_rae_t,
            "std_rae": std_rae_t,
            "per_seed_results": per_seed_t,
            "te_unb_rae_in_sample": te_unb_rae_in_sample,
            "te_mean": float(deploy_te_t.mean()),
            "te_std": float(deploy_te_t.std()),
            "skipped": False,
        })

    # ---- Step E: pick best threshold ----
    print("\n" + "=" * 78)
    print("STEP E: BEST THRESHOLD SELECTION")
    print("=" * 78)
    print(f"   reference (no filter):  mean_rae = {ref_mean:.4f} +/- {ref_std:.4f}")
    eligible = [r for r in sweep_records if (r["mean_rae"] is not None)]
    if not eligible:
        raise RuntimeError("No threshold produced eligible results")
    best = min(eligible, key=lambda r: r["mean_rae"])
    best_thresh = best["thresh"]
    best_mean = best["mean_rae"]
    best_std = best["std_rae"]
    print(f"   best threshold = {best_thresh}  mean_rae = {best_mean:.4f} +/- {best_std:.4f}")
    delta_vs_ref = best_mean - ref_mean
    print(f"   delta vs reference (no filter) = {delta_vs_ref:+.4f}")
    print("\n   --- sweep summary table ---")
    print(f"   {'thresh':>7s}  {'n_kept':>6s}  {'n_drop':>6s}  {'frac_drop':>9s}  "
          f"{'mean_rae':>9s}  {'std_rae':>8s}  {'te_unb_in':>10s}")
    for r in sweep_records:
        if r["mean_rae"] is None:
            print(f"   {r['thresh']:7.3f}  {r['n_kept']:6d}  {r['n_drop']:6d}  "
                  f"{r['frac_drop']:9.4f}  {'skipped':>9s}  {'-':>8s}  {'-':>10s}")
        else:
            mark = "  <-- BEST" if r["thresh"] == best_thresh else ""
            print(f"   {r['thresh']:7.3f}  {r['n_kept']:6d}  {r['n_drop']:6d}  "
                  f"{r['frac_drop']:9.4f}  {r['mean_rae']:9.4f}  {r['std_rae']:8.4f}  "
                  f"{r['te_unb_rae_in_sample']:10.4f}{mark}")

    # ---- Gate ----
    if best_mean < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif best_mean < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"\n[GATE] best_mean_rae={best_mean:.4f}  vs PROMOTE<{GATE_PROMOTE}  "
          f"MARGINAL<{GATE_MARGINAL}  -> {verdict}")

    # ---- Save best ----
    best_oof = oof_per_thresh[best_thresh]
    best_te = deploy_te_per_thresh[best_thresh]
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, best_oof)
    np.save(te_path, best_te)
    print(f"[save] {oof_path}")
    print(f"[save] {te_path}")

    summary = {
        "tag": TAG,
        "method": "outlier-robust pre-train (4139 residual MAD filter -> 117-col LGBM bag refit)",
        "mad_thresholds_sweep": MAD_THRESHOLDS,
        "resid_seeds": RESID_SEEDS,
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "n_train": n_train,
        "n_test": n_test,
        "n_unb": n_unb,
        "feat_dim": int(X_tr.shape[1]),
        "n_unique_unb_scaffolds": int(n_unique_scaf),
        "baseline_oof_train_rae_4139": baseline_train_rae,
        "resid_median": resid_median,
        "mad_residual": mad,
        "abs_resid_mean": float(np.abs(resid).mean()),
        "abs_resid_max": float(np.abs(resid).max()),
        "reference_no_filter_mean_rae": ref_mean,
        "reference_no_filter_std_rae": ref_std,
        "reference_no_filter_per_seed": ref_per_seed,
        "sweep_records": sweep_records,
        "best_thresh": best_thresh,
        "best_mean_rae": best_mean,
        "best_std_rae": best_std,
        "delta_best_vs_reference": delta_vs_ref,
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "verdict": verdict,
        "pred_oof_path": str(oof_path),
        "te_path": str(te_path),
        "te_mean": float(best_te.mean()),
        "te_std": float(best_te.std()),
        "wall_sec": round(time.time() - t0, 2),
    }
    sumpath = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(sumpath, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {sumpath}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   baseline 4139 OOF RAE         = {baseline_train_rae:.4f}")
    print(f"   resid MAD                     = {mad:.4f}")
    print(f"   reference (no filter) mean_rae= {ref_mean:.4f}")
    print(f"   best threshold                = {best_thresh}")
    print(f"   best mean_rae (253 unblind)   = {best_mean:.4f} +/- {best_std:.4f}")
    print(f"   verdict                       = {verdict}")
    print(f"   wall                          = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "baseline_oof_train_rae_4139", "mad_residual",
        "reference_no_filter_mean_rae",
        "best_thresh", "best_mean_rae", "best_std_rae",
        "delta_best_vs_reference", "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
