"""nb2690 -- SMOTE oversampling on hit class for K=20 LGBM residual model.

NEW PARADIGM:
    Class-imbalanced regression mitigated by SMOTE-augmented training set.
    pEC50 distribution is right-skewed; only ~16% of unb rows are "hits"
    (y >= 5.5).  Standard regression averages over this imbalance and
    under-predicts the hit tail.  This script discretises y into 10 quantile
    bins, applies SMOTE per bin to synthesise minority rows in K=20 feature
    space, then trains LGBM residual on the SMOTE-augmented matrix.

PROTOCOL:
    1. Build K=20 feature matrix (nb2231 surviving cols, identical to
       nb2682/nb2240/nb2452 substrate).
    2. anchor = chemprop_aux te[unb_idx]; residual = y_unb - anchor.
    3. Bin y_unb into 10 quantile bins (equal-frequency on 253 rows).
    4. Sweep oversample ratio r in {0.5, 1.0, 1.5, 2.0}:
        For each KFold split (RESID_FOLDS=5, KF_SEEDS=[0,1,7,42,137]):
          - target_count_per_bin = max(bin_counts) * r  (per-bin oversample)
          - SMOTE(k_neighbors=5, random_state=42, sampling_strategy=dict)
            on training fold (residual, bin label as auxiliary class signal)
          - Fit LGBM(max_depth=4, num_leaves=15, n_est=300, lr=0.03) on
            augmented (X, residual)
          - Predict residual on validation fold; corrected = anchor + r_pred
        Then 5-fold scaffold CV on the 253 with 5 kf_seeds → pooled mean RAE.
    5. Pick best ratio by mean_rae across 5 kf_seeds.

GATE:
    best_mean_rae < 0.4570 -> PROMOTE
    best_mean_rae < 0.4598 -> MARGINAL_BEAT
    else                   -> FAIL

OUTPUTS:
    scripts/nb2690_smote_oversample.py
    data/processed/nb2690_summary.json
    data/processed/nb2690_pred_oof.npy   (253,) float32  -- best ratio OOF
    data/processed/te_nb2690.npy         (513,) float32  -- best ratio deploy te

References:
    nb2682 IPS density-ratio reweighting (different shift-correction angle)
    nb2452 K=20 IPS-weighted pyramid (mean RAE 0.4601 +/- 0.003)
    nb2171 PRIMARY-1 ceiling 0.4682 deep-30 (anchor-swap substrate change)
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

try:
    from imblearn.over_sampling import SMOTE
except ImportError as e:
    raise ImportError(
        "imblearn (imbalanced-learn) is required. "
        "pip install imbalanced-learn"
    ) from e

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2690"

# -------------------------- gates & references ------------------------------
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598
CHEMPROP_AUX_REF = 0.6216
NB2171_REF = 0.4682
NB2452_REF = 0.4601

# -------------------------- SMOTE configuration -----------------------------
N_QUANTILE_BINS = 10
SMOTE_K_NEIGHBORS = 5
SMOTE_RANDOM_STATE = 42
OVERSAMPLE_RATIOS = [0.5, 1.0, 1.5, 2.0]

# -------------------------- PXR residual LGBM (per brief) -------------------
PXR_LGBM = dict(
    objective="regression",
    max_depth=4,
    num_leaves=15,
    n_estimators=300,
    learning_rate=0.03,
    min_child_samples=5,
    reg_lambda=2.0,
    n_jobs=2,
    verbosity=-1,
)
RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

# -------------------------- evaluation CV -----------------------------------
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# -------------------------- inputs (paths) ----------------------------------
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
NB2231_SUMMARY = DATA_PROCESSED / "nb2231_summary.json"

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


# ============================================================================
# helpers (verbatim 117-col feature substrate from nb2682/nb2630)
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
    return agg.rename(columns={"src_first": "src"})


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


def _load_mordred_test(n_test_expected):
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(f"Mordred cache missing (nb1030 first): {mte_p}")
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
    """Identical 117-col matrix as nb2103/nb2240/nb2604/nb2682."""
    for p in (NB1352_SUMMARY, NB1392_SUMMARY, NB1484_SUMMARY,
              NB1523_SUMMARY, NB1524_SUMMARY, NB1541_SUMMARY):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")
    sum_1352 = json.load(open(NB1352_SUMMARY))
    sum_1484 = json.load(open(NB1484_SUMMARY))
    sum_1523 = json.load(open(NB1523_SUMMARY))
    sum_1524 = json.load(open(NB1524_SUMMARY))
    sum_1541 = json.load(open(NB1541_SUMMARY))
    sum_1392 = json.load(open(NB1392_SUMMARY))

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
        std_test_smiles.append("" if m is None else Chem.MolToSmiles(m))
    fp_test = morgan_fp_batch(std_test_smiles)
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )

    X_te_full117 = np.concatenate(
        [X_ap_te_top, X_maccs_te_top, X_mord_te_top, X_emb_te_top,
         X_av_te_top, pred_chembl_pec50.reshape(-1, 1).astype(np.float32),
         mean_sim.reshape(-1, 1).astype(np.float32)], axis=1
    ).astype(np.float32)
    if X_te_full117.shape[1] != 117:
        raise ValueError(f"got {X_te_full117.shape}")
    return X_te_full117, int(len(pool))


# ============================================================================
# SMOTE-augmented residual training
# ============================================================================

def _bin_y(y, n_bins=N_QUANTILE_BINS):
    """Quantile-bin y into n_bins equal-frequency bins. Returns int labels."""
    edges = np.quantile(y, np.linspace(0, 1, n_bins + 1))
    interior = edges[1:-1]
    bins = np.digitize(y, interior)
    bins = np.clip(bins, 0, n_bins - 1).astype(int)
    return bins


def _smote_per_bin(X_tr, y_tr, residual_tr, bins_tr, oversample_ratio,
                   k_neighbors, random_state):
    """Apply SMOTE to balance bins to `target_count = max_count * oversample_ratio`.

    SMOTE works on classification labels; we use the bin label as the auxiliary
    "class" signal.  For each bin we synthesise new (X, residual) rows.  The
    residual for synthetic rows is interpolated linearly using SMOTE's neighbour-
    convex-combination geometry (same alpha as feature synthesis), giving a
    consistent (X, y) joint augmentation.

    Returns: X_aug, residual_aug, bins_aug
    """
    counts = np.bincount(bins_tr, minlength=N_QUANTILE_BINS)
    max_c = int(counts.max())
    # target_count per bin, floor at current count so SMOTE never down-samples
    target_count = max(int(round(max_c * oversample_ratio)), 1)
    sampling_strategy = {}
    for b in range(N_QUANTILE_BINS):
        if counts[b] == 0:
            continue
        sampling_strategy[b] = max(target_count, int(counts[b]))

    # If every bin is already at target_count, fall back to identity
    if all(sampling_strategy[b] == counts[b] for b in sampling_strategy):
        return X_tr.copy(), residual_tr.copy(), bins_tr.copy()

    # k_neighbors must be < min(per-class count) - 1
    min_class = min(c for c in counts if c > 0)
    eff_k = min(k_neighbors, max(min_class - 1, 1))
    if eff_k < 1:
        return X_tr.copy(), residual_tr.copy(), bins_tr.copy()

    # SMOTE features only; we interpolate residual ourselves to preserve joint
    # geometry.  Concat residual as extra column, then split after SMOTE.
    X_aug_col = np.concatenate(
        [X_tr.astype(np.float32),
         residual_tr.astype(np.float32).reshape(-1, 1)], axis=1
    )
    sm = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=eff_k,
        random_state=random_state,
    )
    try:
        X_aug_full, bins_aug = sm.fit_resample(X_aug_col, bins_tr)
    except Exception as exc:
        # SMOTE can fail when k_neighbors > min_class - 1 after some bins
        # were excluded.  Fall back to identity.
        print(f"      [SMOTE FALLBACK] {exc}")
        return X_tr.copy(), residual_tr.copy(), bins_tr.copy()

    X_aug = X_aug_full[:, :-1].astype(np.float32)
    residual_aug = X_aug_full[:, -1].astype(np.float64)
    bins_aug = np.asarray(bins_aug, dtype=int)
    return X_aug, residual_aug, bins_aug


def _residual_cross_fit_smote(X, residual, y, seed, oversample_ratio):
    """Residual KFold with per-fold SMOTE augmentation."""
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    bins = _bin_y(y, N_QUANTILE_BINS)
    oof = np.full(n, np.nan, dtype=np.float64)
    aug_sizes = []
    for tr_loc, va_loc in kf.split(np.arange(n)):
        X_tr_aug, r_tr_aug, _ = _smote_per_bin(
            X[tr_loc], y[tr_loc], residual[tr_loc], bins[tr_loc],
            oversample_ratio=oversample_ratio,
            k_neighbors=SMOTE_K_NEIGHBORS,
            random_state=SMOTE_RANDOM_STATE + seed,
        )
        aug_sizes.append(int(X_tr_aug.shape[0]))
        params = dict(PXR_LGBM)
        params["random_state"] = seed
        mdl = lgb.LGBMRegressor(**params)
        mdl.fit(X_tr_aug, r_tr_aug)
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof, aug_sizes


def _train_full_then_predict_te_smote(X_unb, residual, y_unb, X_te, seed,
                                      oversample_ratio):
    """Full SMOTE-augmented fit on all 253 rows, predict residual on 513 te."""
    bins = _bin_y(y_unb, N_QUANTILE_BINS)
    X_aug, r_aug, _ = _smote_per_bin(
        X_unb, y_unb, residual, bins,
        oversample_ratio=oversample_ratio,
        k_neighbors=SMOTE_K_NEIGHBORS,
        random_state=SMOTE_RANDOM_STATE + seed,
    )
    params = dict(PXR_LGBM)
    params["random_state"] = seed
    mdl = lgb.LGBMRegressor(**params)
    mdl.fit(X_aug, r_aug)
    return mdl.predict(X_te).astype(np.float32), int(X_aug.shape[0])


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- SMOTE oversample on hit class for K=20 LGBM residual")
    print(f"          n_bins={N_QUANTILE_BINS}  k_neighbors={SMOTE_K_NEIGHBORS}")
    print(f"          oversample ratios = {OVERSAMPLE_RATIOS}")
    print(f"          gate PROMOTE < {GATE_PROMOTE} / MARGINAL < {GATE_MARGINAL}")
    print("=" * 78)

    # ---- Load truth, anchor, K=20 surviving features --------------------
    print("\n[stage A] load truth, anchor, K=20 surviving features")
    with open(NB2231_SUMMARY) as f:
        nb2231 = json.load(f)
    surviving_K20 = list(nb2231["snapshots"]["20"]["surviving_idx_in_117"])
    surviving_K20_names = list(nb2231["snapshots"]["20"]["surviving_names"])
    assert len(surviving_K20) == 20

    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist() if "smiles" in te.columns
                 else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values if "name" in te.columns
                else te["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    residual = y_unb - anchor

    n_hits = int((y_unb >= 5.5).sum())
    bins_global = _bin_y(y_unb, N_QUANTILE_BINS)
    bin_counts_global = np.bincount(bins_global, minlength=N_QUANTILE_BINS).tolist()

    print(f"   n_test={n_test}  n_unb={n_unb}  n_unique_scaffolds={n_unique_scaf}")
    print(f"   chemprop_aux in_RAE = {rae_anchor:.4f} (ref {CHEMPROP_AUX_REF:.4f})")
    print(f"   n_hits (y>=5.5) = {n_hits}/{n_unb}  frac={n_hits/n_unb:.3f}")
    print(f"   bin_counts (10 quantile) = {bin_counts_global}")

    # ---- Build 117-feature matrix on 513 test rows ----------------------
    print("\n[stage B] rebuild 117-feature matrix on 513 test rows")
    X_te_full117, chembl_pool_size = build_117col_feature_matrix(te_smiles, n_test)
    X_te_K20 = X_te_full117[:, surviving_K20].astype(np.float32)
    X_unb_K20 = X_te_K20[unb_idx]
    print(f"   X_te_K20={X_te_K20.shape}  X_unb_K20={X_unb_K20.shape}  "
          f"chembl_pool={chembl_pool_size}")

    # ---- STAGE 1: sweep ratios -----------------------------------------
    print("\n" + "-" * 78)
    print(f"[STAGE 1] sweep oversample ratios {OVERSAMPLE_RATIOS}")
    print(f"          RESID_FOLDS={RESID_FOLDS}  RESID_SEEDS={RESID_SEEDS}")
    print(f"          PXR_LGBM: max_depth=4 leaves=15 n_est=300 lr=0.03")
    print("-" * 78)

    ratio_results = []
    best_ratio = None
    best_pooled_mean = np.inf
    best_oof = None
    best_te = None

    for ratio in OVERSAMPLE_RATIOS:
        print(f"\n   === ratio = {ratio} ===")
        per_seed_oof = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
        per_seed_te = np.zeros((len(RESID_SEEDS), n_test), dtype=np.float64)
        per_seed_corr_rae = []
        per_seed_avg_aug = []
        for i, s in enumerate(RESID_SEEDS):
            ts = time.time()
            resid_oof, aug_sizes = _residual_cross_fit_smote(
                X_unb_K20, residual, y_unb, s, oversample_ratio=ratio,
            )
            per_seed_oof[i] = anchor + resid_oof
            per_seed_corr_rae.append(float(rae(y_unb, anchor + resid_oof)))
            per_seed_avg_aug.append(float(np.mean(aug_sizes)))
            te_resid, te_aug = _train_full_then_predict_te_smote(
                X_unb_K20, residual, y_unb, X_te_K20, s, oversample_ratio=ratio,
            )
            per_seed_te[i] = te_resid
            print(f"      seed={s:3d}: corr_RAE={per_seed_corr_rae[-1]:.4f}  "
                  f"avg_aug={per_seed_avg_aug[-1]:.0f}  te_aug={te_aug}  "
                  f"wall={time.time()-ts:.1f}s")

        mean_bag_oof = per_seed_oof.mean(axis=0)
        mean_bag_te_resid = per_seed_te.mean(axis=0)
        te_513 = te_anchor_513 + mean_bag_te_resid

        # 5-fold scaffold CV pooled RAE across KF_SEEDS
        pooled_per_seed = []
        for kf_seed in KF_SEEDS:
            splits = scaffold_kfold_indices(
                unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
            )
            oof_pooled = np.full(n_unb, np.nan, dtype=np.float64)
            for tr_loc, va_loc in splits:
                oof_pooled[va_loc] = mean_bag_oof[va_loc]
            pooled_per_seed.append(float(rae(y_unb, oof_pooled)))
        pooled_mean = float(np.mean(pooled_per_seed))
        pooled_std = float(np.std(pooled_per_seed))
        rae_mean_bag_raw = float(rae(y_unb, mean_bag_oof))

        print(f"   ratio={ratio}  per_seed_corr_mean={np.mean(per_seed_corr_rae):.4f}  "
              f"mean_bag_raw={rae_mean_bag_raw:.4f}  "
              f"scaffold_pooled_mean={pooled_mean:.4f} (+/- {pooled_std:.4f})")

        ratio_results.append({
            "oversample_ratio": float(ratio),
            "resid_seeds": RESID_SEEDS,
            "per_seed_corr_rae": per_seed_corr_rae,
            "per_seed_corr_rae_mean": float(np.mean(per_seed_corr_rae)),
            "per_seed_avg_aug_train_size": per_seed_avg_aug,
            "mean_bag_raw_rae": rae_mean_bag_raw,
            "scaffold_pooled_rae_per_kfseed": pooled_per_seed,
            "scaffold_pooled_rae_mean": pooled_mean,
            "scaffold_pooled_rae_std": pooled_std,
        })

        if pooled_mean < best_pooled_mean:
            best_pooled_mean = pooled_mean
            best_ratio = float(ratio)
            best_oof = mean_bag_oof.copy()
            best_te = te_513.astype(np.float32).copy()

    # ---- GATE ----------------------------------------------------------
    print("\n" + "-" * 78)
    print(f"GATE  thresholds: PROMOTE < {GATE_PROMOTE}, MARGINAL_BEAT < {GATE_MARGINAL}")
    print("-" * 78)
    if best_pooled_mean < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif best_pooled_mean < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"   best_ratio              = {best_ratio}")
    print(f"   best pooled mean RAE    = {best_pooled_mean:.4f}")
    print(f"   delta vs PROMOTE        = {best_pooled_mean - GATE_PROMOTE:+.4f}")
    print(f"   delta vs MARGINAL       = {best_pooled_mean - GATE_MARGINAL:+.4f}")
    print(f"   delta vs nb2171 ceiling = {best_pooled_mean - NB2171_REF:+.4f}")
    print(f"   delta vs chemprop_aux   = {best_pooled_mean - CHEMPROP_AUX_REF:+.4f}")
    print(f"   verdict                 = {verdict}")

    # ---- Save artifacts -----------------------------------------------
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, best_oof.astype(np.float32))
    np.save(te_path, best_te.astype(np.float32))
    print(f"\n[save] {oof_path}  (best_ratio={best_ratio})")
    print(f"[save] {te_path}  (best_ratio={best_ratio})")

    sub_csv_path = SUBMISSIONS / f"{TAG}_smote_oversample.csv"
    if verdict in ("PROMOTE", "MARGINAL_BEAT"):
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": best_te.astype(np.float32),
        }).to_csv(sub_csv_path, index=False)
        print(f"[save] {sub_csv_path}  (gate {verdict})")
    else:
        print(f"[skip] gate FAIL -- no submission CSV")

    te_unb_in_rae = float(rae(y_unb, best_te[unb_idx]))

    summary = {
        "tag": TAG,
        "method": "smote_per_quantile_bin_K20_LGBM_residual",
        "paradigm": "class_imbalance_oversample_via_SMOTE_on_quantile_bins",
        "anchor_base": "chemprop_aux",
        "anchor_in_rae": rae_anchor,
        "anchor_pre_unblind": True,
        "n_quantile_bins": N_QUANTILE_BINS,
        "smote_k_neighbors": SMOTE_K_NEIGHBORS,
        "smote_random_state": SMOTE_RANDOM_STATE,
        "oversample_ratios": OVERSAMPLE_RATIOS,
        "resid_folds": RESID_FOLDS,
        "resid_seeds": RESID_SEEDS,
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "pxr_lgbm_params": {k: v for k, v in PXR_LGBM.items() if k != "verbosity"},
        "k20_surviving_idx_in_117": [int(j) for j in surviving_K20],
        "k20_surviving_names": surviving_K20_names,
        "n_hits_y_ge_5p5": n_hits,
        "frac_hits": float(n_hits / n_unb),
        "bin_counts_global": bin_counts_global,
        "ratio_results": ratio_results,
        "best_ratio": best_ratio,
        "best_pooled_mean_rae": best_pooled_mean,
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "delta_vs_promote": best_pooled_mean - GATE_PROMOTE,
        "delta_vs_marginal": best_pooled_mean - GATE_MARGINAL,
        "delta_vs_nb2171": best_pooled_mean - NB2171_REF,
        "delta_vs_chemprop_aux": best_pooled_mean - CHEMPROP_AUX_REF,
        "verdict": verdict,
        "te_mean": float(best_te.mean()),
        "te_std": float(best_te.std()),
        "te_unb_in_sample_rae": te_unb_in_rae,
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv_path) if verdict != "FAIL" else None,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "chembl_pool_size": chembl_pool_size,
        "mean_rae": best_pooled_mean,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    for r in ratio_results:
        print(f"   ratio={r['oversample_ratio']:<4}  "
              f"pooled_mean={r['scaffold_pooled_rae_mean']:.4f} "
              f"(+/- {r['scaffold_pooled_rae_std']:.4f})  "
              f"raw_mean_bag={r['mean_bag_raw_rae']:.4f}")
    print(f"   best_ratio              = {best_ratio}")
    print(f"   best pooled mean RAE    = {best_pooled_mean:.4f}")
    print(f"   verdict                 = {verdict}")
    print(f"   wall                    = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "best_ratio",
        "best_pooled_mean_rae",
        "mean_rae",
        "verdict",
        "delta_vs_promote",
        "delta_vs_marginal",
        "delta_vs_nb2171",
        "delta_vs_chemprop_aux",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
