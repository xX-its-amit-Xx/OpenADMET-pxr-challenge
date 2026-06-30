"""nb1283 -- Distance-weighted KFold cross-fit for nb1242 protocol.

Hypothesis:
    Standard random KFold gives moderate-Tanimoto holdout per fold (median sim
    0.5+ between train fold and val fold within the 253 unblind), which is
    OPTIMISTIC relative to the actual 513 test (median sim ~0.52 to TRAIN,
    rare-scaffold tail much lower).  A DISTANCE-WEIGHTED KFold -- clustering
    folds by Tanimoto similarity so within-fold cpds are similar and
    fold-to-fold cpds are dissimilar -- forces the held-out fold to be
    SCAFFOLD-DISTANT from the training fold.  The resulting cross-fit RAE is
    a more LB-faithful (pessimistic) estimate.

Purpose:
    Not to BEAT nb1242 (0.5431).  Get a more reliable LB estimate.  If
    DW-KFold RAE is meaningfully higher (e.g. 0.57+), the predicted LB for
    nb1242 should be revised upward accordingly.

Protocol:
    1. Compute Morgan-2048 fingerprints for the 253 unblind compounds.
    2. Pairwise Tanimoto -> 1-T distance matrix (253, 253).
    3. AgglomerativeClustering(n_clusters=5, linkage='average',
       metric='precomputed') -> 5 cluster labels.
    4. Use cluster labels as the 5 KFold splits (cluster i as val, rest as
       train).
    5. Apply nb1242 protocol VERBATIM:
       - Anchor: nb1070_pred_oof
       - Residual = y_unb - anchor
       - Features: MACCS-167(unb) + pred_chembl_pec50[unb_idx] + sim[unb_idx]
       - 5-seed shallow LGBM Huber bag (depth=3, leaves=7, n_est=80,
         lr=0.05, huber_alpha=1.0)
    6. Pool RAE across the 5 clusters; report per-cluster RAE, mean-bag pooled
       RAE, cluster sizes, mean within-cluster Tanimoto, mean cluster-to-cluster
       Tanimoto.
    7. Compare DW-KFold RAE vs nb1242 random-KFold RAE (0.5431); compute LB
       estimate adjustment.

Outputs:
    scripts/nb1283_distance_weighted_kfold.py     (this file)
    data/processed/nb1283_summary.json
    data/processed/nb1283_mean_bag_oof.npy        (253,) float32
    data/processed/nb1283_per_seed_corrected_oof.npy (5, 253) float32
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
from sklearn.cluster import AgglomerativeClustering
from lightgbm import LGBMRegressor
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1283"
ANCHOR = "nb1070"

N_CLUSTERS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

# nb1242 ChEMBL pool filter constants
KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

NB1070_REF = 0.5771
NB1242_RANDOM_KFOLD_RAE = 0.5431   # the comparator we want to recalibrate
NB1242_LB_PRED_RANDOM = 0.542       # current LB estimate from random KFold


# ----------------------------------------------------------------------
# nb1242 helpers (verbatim from nb1242_chembl_knn_feat.py)
# ----------------------------------------------------------------------
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
    """Verbatim from nb1242."""
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
        print(f"   [src] CHEMBL3401_raw kept: {len(d)} rows")

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
        print(f"   [src] chembl_nr_extended PXR kept: {len(d)} rows")

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
        print(f"   [src] chembl_pxr_all_types kept: {len(d)} rows")

    if not frames:
        raise FileNotFoundError("No local ChEMBL PXR parquets found")

    pool = pd.concat(frames, ignore_index=True)
    print(f"   [pool] pre-standardize union: {len(pool)} rows")

    mols = pool["smiles"].apply(standardize)
    pool["inchikey"] = mols.apply(_safe_inchikey)
    pool["std_smiles"] = mols.apply(_safe_can_smiles)
    pool = pool[pool["inchikey"].notna() & pool["std_smiles"].notna()].copy()
    print(f"   [pool] after RDKit standardize: {len(pool)} rows")

    agg = (
        pool.groupby("inchikey", as_index=False)
        .agg(pec50=("pec50", "median"),
             std_smiles=("std_smiles", "first"),
             src_first=("src", "first"),
             n_meas=("pec50", "count"))
    )
    agg = agg.rename(columns={"src_first": "src"})
    print(f"   [pool] after InChIKey dedup: {len(agg)} unique cpds")
    return agg


def _tanimoto_topk(fp_q: np.ndarray, fp_pool: np.ndarray, k: int):
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
    w = top_sim.copy()
    w = np.clip(w, 0.0, 1.0)
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


def _lgbm_params(seed: int) -> dict:
    return dict(
        objective="huber",
        alpha=1.0,
        learning_rate=0.05,
        n_estimators=80,
        max_depth=3,
        num_leaves=7,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        verbosity=-1,
        random_state=seed,
        n_jobs=2,
    )


# ----------------------------------------------------------------------
# Distance-weighted KFold
# ----------------------------------------------------------------------
def _pairwise_tanimoto_dist(fp: np.ndarray) -> np.ndarray:
    """Pairwise Tanimoto distance (1 - sim) over uint8 0/1 fingerprints.
    Returns (n, n) float32 with zero diagonal."""
    a = fp.astype(np.float32)
    s = a.sum(axis=1)            # (n,)
    inter = a @ a.T              # (n, n)
    denom = s[:, None] + s[None, :] - inter
    denom = np.maximum(denom, 1.0)
    sim = inter / denom
    np.fill_diagonal(sim, 1.0)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    return dist.astype(np.float32), sim.astype(np.float32)


def _residual_dw_kfold_one_seed(X: np.ndarray, residual: np.ndarray,
                                cluster_labels: np.ndarray,
                                seed: int) -> tuple[np.ndarray, list[float]]:
    """Cross-fit using cluster labels as folds.  Returns OOF residual
    predictions and per-fold RAE (computed against anchor-corrected truth
    later)."""
    n = len(residual)
    oof = np.full(n, np.nan, dtype=np.float64)
    cluster_ids = sorted(np.unique(cluster_labels).tolist())
    for c in cluster_ids:
        va_loc = np.where(cluster_labels == c)[0]
        tr_loc = np.where(cluster_labels != c)[0]
        mdl = LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Distance-weighted KFold cross-fit for nb1242 protocol")
    print(f"          n_clusters = {N_CLUSTERS}  seeds = {RESID_SEEDS}")
    print(f"          features = MACCS-167 + pred_chembl_pec50 + sim  (169)")
    print(f"          purpose = LB-faithful RAE estimate (random-KFold optimism check)")
    print("=" * 78)

    # ---- Load anchor + truth ----
    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    anchor = np.load(DATA_PROCESSED / f"{ANCHOR}_pred_oof.npy").astype(np.float64)
    if anchor.shape[0] != n_unb:
        raise ValueError(f"{ANCHOR} shape mismatch: {anchor.shape} vs {n_unb}")
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} pooled RAE = {rae_anchor:.4f}  (ref ~{NB1070_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- ChEMBL pool (verbatim from nb1242) ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL (verbatim nb1242 protocol)")
    print("-" * 78)
    pool = _load_chembl_pool()

    test_mols = [standardize(s) for s in test_smiles]
    test_inchikeys = set()
    for m in test_mols:
        ik = _safe_inchikey(m)
        if ik is not None:
            test_inchikeys.add(ik)
    n_before = len(pool)
    pool = pool[~pool["inchikey"].isin(test_inchikeys)].reset_index(drop=True)
    n_after = len(pool)
    print(f"   leak guard: {n_before} -> {n_after}")

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

    top_idx, top_sim = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx, top_sim, pool_labels, fallback=pool_median
    )

    # ---- MACCS unblind slice ----
    X_maccs_te = np.load(MACCS_TE_PATH)
    if X_maccs_te.shape[0] != n_test:
        raise ValueError(f"MACCS cache shape mismatch: {X_maccs_te.shape}")
    X_maccs_unb = X_maccs_te[unb_idx].astype(np.float32)

    # ---- Build feature matrix (253, 169) -- same as nb1242 ----
    pred_chembl_unb = pred_chembl_pec50[unb_idx].astype(np.float32)
    mean_sim_unb = mean_sim[unb_idx].astype(np.float32)
    X_unb = np.concatenate(
        [
            X_maccs_unb,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim = X_unb.shape[1]
    print(f"   residual feature matrix: {X_unb.shape}")

    # ---- DISTANCE-WEIGHTED KFOLD via AgglomerativeClustering ----
    print("\n" + "-" * 78)
    print(f"DISTANCE-WEIGHTED KFOLD (AgglomerativeClustering, avg-linkage, "
          f"n_clusters={N_CLUSTERS})")
    print("-" * 78)
    print("   computing Tanimoto distance matrix over 253 unblind Morgan-2048 FPs...")
    fp_unb_test = fp_test[unb_idx]            # (253, 2048)
    dist_unb, sim_unb = _pairwise_tanimoto_dist(fp_unb_test)
    # Distance distribution (off-diagonal)
    mask_off = ~np.eye(n_unb, dtype=bool)
    pair_sim = sim_unb[mask_off]
    print(f"   intra-253 Tanimoto sim   "
          f"p10={np.percentile(pair_sim, 10):.3f}  "
          f"p50={np.percentile(pair_sim, 50):.3f}  "
          f"p90={np.percentile(pair_sim, 90):.3f}  "
          f"mean={pair_sim.mean():.3f}")

    agg = AgglomerativeClustering(
        n_clusters=N_CLUSTERS,
        linkage="average",
        metric="precomputed",
    )
    cluster_labels = agg.fit_predict(dist_unb).astype(np.int32)

    # ---- Cluster diagnostics ----
    cluster_ids = sorted(np.unique(cluster_labels).tolist())
    cluster_sizes = {int(c): int((cluster_labels == c).sum())
                     for c in cluster_ids}
    print(f"   cluster sizes = {cluster_sizes}")

    # mean within-cluster Tanimoto sim & mean between-cluster Tanimoto sim
    within_sims = []
    between_sims = []
    cluster_centroid_avg_sim = {}
    for c in cluster_ids:
        idx_c = np.where(cluster_labels == c)[0]
        if len(idx_c) >= 2:
            # within: pairs (i, j) with i<j both in c
            sub = sim_unb[np.ix_(idx_c, idx_c)]
            # take upper triangle off-diag
            iu = np.triu_indices(len(idx_c), k=1)
            within_sims.append(sub[iu])
        # between: pairs (i in c, j not in c)
        idx_oth = np.where(cluster_labels != c)[0]
        if len(idx_oth) > 0 and len(idx_c) > 0:
            sub = sim_unb[np.ix_(idx_c, idx_oth)].flatten()
            between_sims.append(sub)
            cluster_centroid_avg_sim[int(c)] = float(sub.mean())

    within_arr = np.concatenate(within_sims) if within_sims else np.array([0.0])
    between_arr = np.concatenate(between_sims) if between_sims else np.array([0.0])
    mean_within_sim = float(within_arr.mean())
    mean_between_sim = float(between_arr.mean())
    print(f"   mean WITHIN-cluster Tanimoto sim  = {mean_within_sim:.4f}")
    print(f"   mean BETWEEN-cluster Tanimoto sim = {mean_between_sim:.4f}")
    print(f"   separation ratio (within - between) = "
          f"{mean_within_sim - mean_between_sim:+.4f}")

    # For each cluster, "sim of holdout to training fold" = mean of
    # (cluster c rows) x (rows not in c) Tanimoto
    print(f"   per-cluster mean sim of HOLDOUT to TRAINING fold:")
    for c in cluster_ids:
        print(f"     cluster {c}: size={cluster_sizes[c]:3d}  "
              f"mean_sim_to_train = {cluster_centroid_avg_sim[c]:.4f}")

    # ---- Random-KFold mean holdout-to-train sim (for comparison) ----
    rng = np.random.default_rng(0)
    perm = rng.permutation(n_unb)
    random_fold_labels = np.zeros(n_unb, dtype=np.int32)
    splits = np.array_split(perm, N_CLUSTERS)
    for f_id, idx in enumerate(splits):
        random_fold_labels[idx] = f_id
    random_holdout_to_train_sims = []
    for c in range(N_CLUSTERS):
        idx_c = np.where(random_fold_labels == c)[0]
        idx_oth = np.where(random_fold_labels != c)[0]
        if len(idx_c) > 0 and len(idx_oth) > 0:
            sub = sim_unb[np.ix_(idx_c, idx_oth)].flatten()
            random_holdout_to_train_sims.append(sub.mean())
    random_mean_holdout_to_train_sim = float(np.mean(random_holdout_to_train_sims))
    dw_mean_holdout_to_train_sim = float(np.mean(list(cluster_centroid_avg_sim.values())))
    print(f"\n   RANDOM-KFold mean holdout-to-train sim = "
          f"{random_mean_holdout_to_train_sim:.4f}")
    print(f"   DW-KFold     mean holdout-to-train sim = "
          f"{dw_mean_holdout_to_train_sim:.4f}")
    print(f"   delta (DW - random) = "
          f"{dw_mean_holdout_to_train_sim - random_mean_holdout_to_train_sim:+.4f}  "
          f"(NEGATIVE = DW folds more dissimilar, as intended)")

    # ---- Per-seed DW-KFold residual cross-fit ----
    print("\n" + "-" * 78)
    print(f"PER-SEED DW-KFOLD RESIDUAL CROSS-FIT (depth=3 shallow LGBM Huber, "
          f"dim={feat_dim})")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []

    # Per-cluster RAE pooled across seeds (cluster c -> mean over seeds)
    per_cluster_seed_rae = {int(c): [] for c in cluster_ids}

    for i, s in enumerate(RESID_SEEDS):
        resid_oof_s = _residual_dw_kfold_one_seed(
            X_unb, residual, cluster_labels, s
        )
        pred_corr_s = anchor + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        delta_s = rae_s - rae_anchor
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_nb1070": delta_s,
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
        })
        # per-cluster RAE for this seed
        for c in cluster_ids:
            idx_c = np.where(cluster_labels == c)[0]
            if len(idx_c) >= 2:
                rae_cs = float(rae(y_unb[idx_c], pred_corr_s[idx_c]))
                per_cluster_seed_rae[int(c)].append(rae_cs)
        print(f"   seed {s:3d}:  rae_corr = {rae_s:.4f}  "
              f"(d_vs_nb1070 = {delta_s:+.4f})  "
              f"|resid_oof|.std = {resid_oof_s.std():.3f}")

    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))

    per_seed_rae_arr = np.array(per_seed_rae)
    rae_per_seed_mean = float(per_seed_rae_arr.mean())
    rae_per_seed_median = float(np.median(per_seed_rae_arr))
    rae_per_seed_std = float(per_seed_rae_arr.std())

    # ---- Per-cluster pooled RAE on mean_bag ----
    per_cluster_rae_pooled = {}
    for c in cluster_ids:
        idx_c = np.where(cluster_labels == c)[0]
        if len(idx_c) >= 2:
            per_cluster_rae_pooled[int(c)] = float(rae(y_unb[idx_c], mean_bag_oof[idx_c]))

    per_cluster_anchor_rae = {}
    for c in cluster_ids:
        idx_c = np.where(cluster_labels == c)[0]
        if len(idx_c) >= 2:
            per_cluster_anchor_rae[int(c)] = float(rae(y_unb[idx_c], anchor[idx_c]))

    # ---- LB estimate adjustment ----
    # Methodology: random-KFold gave RAE 0.5431 -> predicted LB 0.542.
    # DW-KFold gives a "more pessimistic" RAE.  We use the SAME pre/post-unblind
    # calibration as feedback_lb_two_regime_calibration: pre-unblind te (which
    # nb1070_pred_oof IS, since nb1070 was trained pre-unblind) follows
    # in_RAE ~= LB + 0.003.  The cross-fit estimator on 253 unblind labels is
    # closer to LB than in-sample te.  Random-KFold typically underestimates by
    # ~0.02-0.05 on novel-scaffold tail (feedback_train_oof_blend_transfer).
    #
    # Adjustment: predicted LB from DW-KFold = rae_mean_bag + 0.003
    # (small constant because DW-KFold already pessimistic).
    # Adjustment delta = (DW_RAE - random_KFold_RAE).
    optimism_gap_random_to_dw = rae_mean_bag - NB1242_RANDOM_KFOLD_RAE
    predicted_lb_from_dw = rae_mean_bag + 0.003
    predicted_lb_adjustment = predicted_lb_from_dw - NB1242_LB_PRED_RANDOM

    print("\n" + "-" * 78)
    print("DW-KFOLD AGGREGATIONS")
    print("-" * 78)
    print(f"   per-seed RAE list      = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   per-seed mean          = {rae_per_seed_mean:.4f}")
    print(f"   per-seed median        = {rae_per_seed_median:.4f}")
    print(f"   per-seed std           = {rae_per_seed_std:.4f}")
    print(f"   pooled RAE(mean_bag)   = {rae_mean_bag:.4f}")
    print(f"   pooled RAE(median_bag) = {rae_median_bag:.4f}")
    print(f"   nb1242 RANDOM-KFold    = {NB1242_RANDOM_KFOLD_RAE:.4f}")
    print(f"   optimism gap (DW-rand) = {optimism_gap_random_to_dw:+.4f}  "
          f"(POSITIVE = random was optimistic)")
    print(f"\n   per-cluster pooled RAE (mean_bag, anchor for ref):")
    for c in cluster_ids:
        if c in per_cluster_rae_pooled:
            print(f"     cluster {c}: size={cluster_sizes[c]:3d}  "
                  f"sim_to_train={cluster_centroid_avg_sim[c]:.3f}  "
                  f"rae_meanbag={per_cluster_rae_pooled[c]:.4f}  "
                  f"rae_anchor={per_cluster_anchor_rae[c]:.4f}  "
                  f"delta={per_cluster_rae_pooled[c] - per_cluster_anchor_rae[c]:+.4f}")
    print(f"\n   LB ESTIMATE ADJUSTMENT:")
    print(f"     random-KFold predicted LB    = {NB1242_LB_PRED_RANDOM:.4f}")
    print(f"     DW-KFold predicted LB        = {predicted_lb_from_dw:.4f}  "
          f"(= {rae_mean_bag:.4f} + 0.003 pre-unblind cal)")
    print(f"     LB adjustment (DW - random)  = {predicted_lb_adjustment:+.4f}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_per_seed_corrected_oof.npy",
            per_seed_corrected.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy",
            mean_bag_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_median_bag_oof.npy",
            median_bag_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_cluster_labels.npy",
            cluster_labels.astype(np.int32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_per_seed_corrected_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_mean_bag_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_median_bag_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_cluster_labels.npy'}")

    summary = {
        "tag": TAG,
        "purpose": "LB-faithful RAE estimate via distance-weighted KFold on nb1242 protocol",
        "anchor": ANCHOR,
        "n_clusters": N_CLUSTERS,
        "n_unb": n_unb,
        "feature_dim": feat_dim,
        "resid_seeds": RESID_SEEDS,
        # Cluster diagnostics
        "cluster_sizes": cluster_sizes,
        "intra_253_sim_p10": float(np.percentile(pair_sim, 10)),
        "intra_253_sim_p50": float(np.percentile(pair_sim, 50)),
        "intra_253_sim_p90": float(np.percentile(pair_sim, 90)),
        "intra_253_sim_mean": float(pair_sim.mean()),
        "mean_within_cluster_sim": mean_within_sim,
        "mean_between_cluster_sim": mean_between_sim,
        "separation_ratio_within_minus_between": float(mean_within_sim - mean_between_sim),
        "per_cluster_mean_sim_to_training_fold": cluster_centroid_avg_sim,
        # Comparator with random KFold
        "random_kfold_mean_holdout_to_train_sim": random_mean_holdout_to_train_sim,
        "dw_kfold_mean_holdout_to_train_sim": dw_mean_holdout_to_train_sim,
        "dw_minus_random_sim_delta": float(
            dw_mean_holdout_to_train_sim - random_mean_holdout_to_train_sim
        ),
        # Anchor + truth
        "rae_anchor_nb1070": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        # DW-KFold RAE
        "per_seed_rae": per_seed_rae,
        "per_seed_records": per_seed_records,
        "rae_per_seed_mean": rae_per_seed_mean,
        "rae_per_seed_median": rae_per_seed_median,
        "rae_per_seed_std": rae_per_seed_std,
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "per_cluster_rae_pooled_meanbag": per_cluster_rae_pooled,
        "per_cluster_rae_pooled_anchor": per_cluster_anchor_rae,
        # Comparison vs nb1242 random KFold
        "nb1242_random_kfold_rae": NB1242_RANDOM_KFOLD_RAE,
        "optimism_gap_dw_minus_random": optimism_gap_random_to_dw,
        # LB estimate
        "nb1242_lb_predicted_random_kfold": NB1242_LB_PRED_RANDOM,
        "predicted_lb_from_dw_kfold": predicted_lb_from_dw,
        "predicted_lb_adjustment_dw_minus_random": predicted_lb_adjustment,
        "lb_calibration_pre_unblind_offset": 0.003,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "n_clusters", "cluster_sizes",
        "intra_253_sim_p50", "intra_253_sim_mean",
        "mean_within_cluster_sim", "mean_between_cluster_sim",
        "separation_ratio_within_minus_between",
        "random_kfold_mean_holdout_to_train_sim",
        "dw_kfold_mean_holdout_to_train_sim",
        "dw_minus_random_sim_delta",
        "rae_anchor_nb1070",
        "per_seed_rae",
        "rae_per_seed_mean", "rae_per_seed_std",
        "rae_mean_bag", "rae_median_bag",
        "nb1242_random_kfold_rae",
        "optimism_gap_dw_minus_random",
        "per_cluster_rae_pooled_meanbag",
        "per_cluster_rae_pooled_anchor",
        "predicted_lb_from_dw_kfold",
        "predicted_lb_adjustment_dw_minus_random",
    ):
        print(f"  {k}: {res.get(k)}")
