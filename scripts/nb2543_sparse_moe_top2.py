"""nb2543 -- Sparse mixture-of-experts with top-2 Tanimoto routing.

NEW PARADIGM (vs single-model K=20 LGBM): partition training rows into 5
disjoint scaffold clusters via Bemis-Murcko + Tanimoto-K-means, fit one
K=20 LGBM expert per partition, then route each inference row to the
top-2 nearest cluster centroids in Tanimoto space and blend their
predictions 0.7 / 0.3.  This trades a single all-data model for a
sparse-routed ensemble whose experts are scaffold-specialised; the
gate selects experts on a feature axis (Tanimoto) orthogonal to the
LGBM split features.

PROTOCOL:
    1. 5-fold scaffold CV on 253 unblind, 5 kf_seeds in {1001..1005}.
    2. Per (kf_seed, fold):
        a. Within train slice (~ 202 rows), Bemis-Murcko cluster centroids
           via Morgan-FP Tanimoto-K-means (K=5, n_init=3).
        b. Assign each train row to its nearest cluster -> 5 disjoint
           partitions.
        c. Fit one LGBM(K=20 features, deep) per partition.
        d. For each val row: compute Tanimoto to each of 5 cluster
           centroids, pick top-2 by similarity.  Predict
           y_hat = 0.7 * top1_expert(x) + 0.3 * top2_expert(x).
    3. Pool OOF across folds, compute RAE; average over 5 kf_seeds.
    4. Gate: mean_rae < 0.4570 -> PROMOTE
             mean_rae < 0.4601 -> MARGINAL_BEAT
             else              -> FAIL

    Deploy: refit on all 253 unblind, cluster once, predict on 513 test
    via the same top-2 routing.

OUTPUTS:
    scripts/nb2543_sparse_moe_top2.py
    data/processed/nb2543_summary.json
    data/processed/nb2543_pred_oof.npy        (253,) float32
    data/processed/te_nb2543.npy              (513,) float32
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

TAG = "nb2543"

# -------- MoE routing constants --------
N_EXPERTS = 5
TOP_K_ROUTE = 2
TOP1_WEIGHT = 0.7
TOP2_WEIGHT = 0.3
KMEANS_N_INIT = 3
KMEANS_MAX_ITER = 25
ROUTE_SEED = 7

# -------- evaluation constants --------
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4601

# -------- nb2240 reference --------
NB2240_REF_OOF_K20 = 0.4682  # nb2171 5-anchor pyramid pooled RAE (K=20 sub-ensemble proxy)

# -------- feature build paths (copied from nb2502/nb2532) --------
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
NB2231_SUMMARY = DATA_PROCESSED / "nb2231_summary.json"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

# -------- LGBM hyperparams (mirror nb2502 / nb2240) --------
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


# =============================================================================
# Helpers (copied from nb2502 / nb2532 -- feature build + ChEMBL pool)
# =============================================================================

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


# =============================================================================
# Tanimoto K-means (continuous-centroid via mean of binary FPs)
# =============================================================================

def _tanimoto_sim_matrix(fp_q, fp_centroids):
    """Tanimoto between binary fp_q rows and (possibly continuous) centroid rows.

    Continuous-centroid Tanimoto:
        T(a, b) = sum(min(a, b)) / sum(max(a, b))
    For binary a and continuous b (centroid = mean of binary), this is
    the standard generalization preserving range [0, 1].
    """
    a = fp_q.astype(np.float32)
    b = fp_centroids.astype(np.float32)
    n_q = a.shape[0]
    n_c = b.shape[0]
    sim = np.zeros((n_q, n_c), dtype=np.float32)
    BLOCK = 128
    for s in range(0, n_q, BLOCK):
        e = min(n_q, s + BLOCK)
        block = a[s:e][:, None, :]              # (eb, 1, d)
        centroid = b[None, :, :]                # (1, n_c, d)
        inter = np.minimum(block, centroid).sum(axis=2)  # (eb, n_c)
        union = np.maximum(block, centroid).sum(axis=2)  # (eb, n_c)
        union = np.maximum(union, 1e-6)
        sim[s:e] = inter / union
    return sim


def _tanimoto_kmeans(fp, n_clusters, max_iter, seed):
    """Tanimoto-K-means using continuous centroids = mean of cluster FPs.

    Assignment uses Tanimoto similarity (higher = closer).
    Returns (labels, centroids) where centroids has shape (n_clusters, d).
    """
    rng = np.random.default_rng(seed)
    n = fp.shape[0]
    d = fp.shape[1]
    fpf = fp.astype(np.float32)
    init_idx = rng.choice(n, size=n_clusters, replace=False)
    centroids = fpf[init_idx].copy().astype(np.float32)
    labels = np.zeros(n, dtype=np.int32)
    for it in range(max_iter):
        sim = _tanimoto_sim_matrix(fpf, centroids)         # (n, n_clusters)
        new_labels = np.argmax(sim, axis=1).astype(np.int32)
        if it > 0 and np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        # update centroids (mean of binary FPs in cluster); reseed empty clusters
        for k in range(n_clusters):
            members = np.where(labels == k)[0]
            if len(members) == 0:
                # reseed empty cluster to the row farthest from any existing centroid
                worst = np.argmin(sim.max(axis=1))
                centroids[k] = fpf[worst]
            else:
                centroids[k] = fpf[members].mean(axis=0)
    return labels, centroids


def _kmeans_n_init(fp, n_clusters, n_init, max_iter, seed):
    """Run Tanimoto-K-means n_init times, pick partition with highest avg in-cluster Tanimoto."""
    best_labels, best_centroids, best_score = None, None, -np.inf
    for j in range(n_init):
        sub_seed = int(seed + j * 1009)
        labels, centroids = _tanimoto_kmeans(fp, n_clusters, max_iter, sub_seed)
        # compactness: average Tanimoto of each point to its centroid
        sim = _tanimoto_sim_matrix(fp, centroids)
        score = float(sim[np.arange(len(labels)), labels].mean())
        if score > best_score:
            best_score = score
            best_labels = labels
            best_centroids = centroids
    return best_labels, best_centroids, best_score


# =============================================================================
# MoE cross-fit
# =============================================================================

def _route_top_k(centroid_fp, query_fp, k):
    """Return (top_idx, top_sim) of shape (n_q, k) by Tanimoto similarity.

    centroid_fp may be continuous (cluster mean); query_fp is binary.
    """
    sim = _tanimoto_sim_matrix(query_fp, centroid_fp)
    n_q = query_fp.shape[0]
    n_c = centroid_fp.shape[0]
    if k >= n_c:
        top_idx = np.argsort(-sim, axis=1)[:, :k]
    else:
        part = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
        row = np.arange(n_q)[:, None]
        sim_part = sim[row, part]
        order = np.argsort(-sim_part, axis=1)
        top_idx = part[row, order]
    row = np.arange(n_q)[:, None]
    top_sim = sim[row, top_idx]
    return top_idx.astype(np.int32), top_sim.astype(np.float32)


def cv_run_moe(X_unb, y_unb, fp_unb, unb_scaffolds):
    """5-seed scaffold CV with per-fold MoE training + top-2 routing.

    Per (kf_seed, fold):
      - Cluster TRAIN slice into 5 partitions via Tanimoto-K-means on Morgan FP.
      - Empty clusters get the un-clustered fold-train median predictor as fallback.
      - Fit one LGBM(K=20 features) per non-empty partition (seed=ROUTE_SEED).
      - Route each VAL row to top-2 nearest cluster centroids; predict
        y_hat = 0.7 * top1 + 0.3 * top2.
    """
    n_unb = X_unb.shape[0]
    per_seed = []
    all_oofs = []
    diagnostics = []
    for kf_seed in KF_SEEDS:
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        oof = np.full(n_unb, np.nan, dtype=np.float64)
        seed_diag = []
        for fold_i, (tr_loc, va_loc) in enumerate(splits):
            X_tr, y_tr, fp_tr = X_unb[tr_loc], y_unb[tr_loc], fp_unb[tr_loc]
            X_va, fp_va = X_unb[va_loc], fp_unb[va_loc]

            labels, centroids, compact = _kmeans_n_init(
                fp_tr, n_clusters=N_EXPERTS,
                n_init=KMEANS_N_INIT, max_iter=KMEANS_MAX_ITER, seed=ROUTE_SEED,
            )
            # fit one expert per partition that has >=2 rows; fallback for tiny clusters
            expert_models = []
            expert_fallbacks = []
            cluster_sizes = []
            global_fallback = float(np.mean(y_tr))
            for k in range(N_EXPERTS):
                members = np.where(labels == k)[0]
                cluster_sizes.append(int(len(members)))
                if len(members) >= 5:
                    mdl = lgb.LGBMRegressor(**_lgbm_params(ROUTE_SEED + k))
                    mdl.fit(X_tr[members], y_tr[members])
                    expert_models.append(mdl)
                    expert_fallbacks.append(None)
                else:
                    # tiny partition -> constant predictor = cluster mean (or global fallback)
                    if len(members) > 0:
                        const = float(np.mean(y_tr[members]))
                    else:
                        const = global_fallback
                    expert_models.append(None)
                    expert_fallbacks.append(const)
            # route val rows -> top-2 experts -> blend
            top_idx, top_sim = _route_top_k(centroids, fp_va, k=TOP_K_ROUTE)
            preds_top = np.zeros((len(va_loc), TOP_K_ROUTE), dtype=np.float64)
            for col in range(TOP_K_ROUTE):
                expert_ids = top_idx[:, col]
                # vectorize per unique expert id to avoid per-row predict overhead
                unique_eids = np.unique(expert_ids)
                col_pred = np.zeros(len(va_loc), dtype=np.float64)
                for eid in unique_eids:
                    rows = np.where(expert_ids == eid)[0]
                    mdl = expert_models[int(eid)]
                    if mdl is not None:
                        col_pred[rows] = mdl.predict(X_va[rows])
                    else:
                        col_pred[rows] = expert_fallbacks[int(eid)]
                preds_top[:, col] = col_pred
            blend = TOP1_WEIGHT * preds_top[:, 0] + TOP2_WEIGHT * preds_top[:, 1]
            oof[va_loc] = blend
            seed_diag.append({
                "fold": int(fold_i),
                "n_train": int(len(tr_loc)),
                "n_val": int(len(va_loc)),
                "cluster_sizes": cluster_sizes,
                "kmeans_compactness": round(float(compact), 4),
                "n_tiny_experts": int(sum(1 for m in expert_models if m is None)),
            })
        assert not np.isnan(oof).any(), "oof has NaN -- fold cover incomplete"
        pooled = float(rae(y_unb, oof))
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": float(pooled),
            "oof_min": float(oof.min()),
            "oof_max": float(oof.max()),
            "oof_mean": float(oof.mean()),
            "oof_std": float(oof.std()),
        })
        all_oofs.append(oof)
        diagnostics.append({"kf_seed": int(kf_seed), "folds": seed_diag})
        print(f"   seed={kf_seed}  pooled_RAE={pooled:.4f}  "
              f"oof_range=[{oof.min():.3f}, {oof.max():.3f}]")
    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    return {
        "per_seed": per_seed,
        "diagnostics": diagnostics,
        "mean_oof": mean_oof,
        "mean_rae": float(np.mean([r["pooled_rae"] for r in per_seed])),
        "std_rae": float(np.std([r["pooled_rae"] for r in per_seed])),
        "rae_of_mean_oof": float(rae(y_unb, mean_oof)),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Sparse MoE (5 experts, top-2 Tanimoto routing) on K=20 features")
    print("=" * 78)

    # --- load 253 unblind labels, anchor, scaffolds, raw SMILES ---
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] n_test={n_test}  n_unb={n_unb}  chemprop_aux in_RAE={rae_anchor:.4f}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    # --- precompute Morgan FPs for routing (binary, n_bits=2048) ---
    test_mols = [standardize(s) for s in te_smiles]
    std_test_smiles = [Chem.MolToSmiles(m) if m is not None else "" for m in test_mols]
    fp_test = morgan_fp_batch(std_test_smiles).astype(np.float32)
    fp_unb = fp_test[unb_idx]
    print(f"[fp] morgan(2048): fp_test={fp_test.shape}  fp_unb={fp_unb.shape}")

    # --- load K=20 RFE-surviving feature indices ---
    with open(NB2231_SUMMARY) as f:
        nb2231 = json.load(f)
    surviving_K20 = list(nb2231["snapshots"]["20"]["surviving_idx_in_117"])
    surviving_K20_names = list(nb2231["snapshots"]["20"]["surviving_names"])
    assert len(surviving_K20) == 20, f"expected 20 features, got {len(surviving_K20)}"
    print(f"[feat] K=20 surviving features loaded from nb2231")

    # --- rebuild 117-col 5-way feature matrix exactly as nb2502/nb2532 ---
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

    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test)[:, top_ap_bit_idx].astype(np.float32)
    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)[:, top_maccs_bit_idx].astype(np.float32)
    X_mord_te = _load_mordred_test(n_test)[:, top_mord_col_idx].astype(np.float32)
    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)[:, top_embed_col_idx].astype(np.float32)
    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)[:, top_avalon_bit_idx].astype(np.float32)

    pool = _load_chembl_pool()
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
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )

    X_te_full = np.concatenate(
        [
            X_ap_te, X_maccs_te, X_mord_te, X_emb_te, X_av_te,
            pred_chembl_pec50.reshape(-1, 1).astype(np.float32),
            mean_sim.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    assert X_te_full.shape[1] == 117, f"feat_dim {X_te_full.shape[1]} != 117"
    X_te_K20 = X_te_full[:, surviving_K20].astype(np.float32)
    X_unb_K20 = X_te_K20[unb_idx]
    print(f"[feat] X_unb_K20 = {X_unb_K20.shape}  X_te_K20 = {X_te_K20.shape}")

    # =========================================================================
    # 5-seed x 5-fold scaffold CV with per-fold MoE training + top-2 routing
    # =========================================================================
    print("\n" + "-" * 78)
    print(f"MoE  experts={N_EXPERTS}  top_k_route={TOP_K_ROUTE}  "
          f"blend=({TOP1_WEIGHT},{TOP2_WEIGHT})  kf_seeds={KF_SEEDS}  folds={N_FOLDS}")
    print("-" * 78)
    tsw = time.time()
    res = cv_run_moe(X_unb_K20, y_unb, fp_unb, unb_scaffolds)
    cv_wall = time.time() - tsw
    print(f"\n[cv] mean across 5 seeds  pooled_RAE = {res['mean_rae']:.4f} "
          f"(+/- {res['std_rae']:.4f})")
    print(f"[cv] RAE of mean-of-seed OOFs        = {res['rae_of_mean_oof']:.4f}")
    print(f"[cv] wall = {cv_wall:.1f}s")

    # --- gate ---
    mean_rae = res["mean_rae"]
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    delta_nb2240 = mean_rae - NB2240_REF_OOF_K20
    print(f"\n[gate] mean_rae={mean_rae:.4f}  "
          f"PROMOTE<{GATE_PROMOTE}  MARGINAL<{GATE_MARGINAL}  -> {verdict}")
    print(f"[ref ] vs nb2240 K=20 ({NB2240_REF_OOF_K20:.4f})  delta={delta_nb2240:+.4f}")

    # =========================================================================
    # Deploy: refit MoE on all 253 unblind, predict on 513 test via top-2 route
    # =========================================================================
    print("\n" + "-" * 78)
    print("DEPLOY  (refit 5 experts on all 253; route 513 to top-2 experts)")
    print("-" * 78)
    deploy_labels, deploy_centroids, deploy_compact = _kmeans_n_init(
        fp_unb, n_clusters=N_EXPERTS,
        n_init=KMEANS_N_INIT, max_iter=KMEANS_MAX_ITER, seed=ROUTE_SEED,
    )
    deploy_experts = []
    deploy_fallbacks = []
    deploy_cluster_sizes = []
    global_fallback = float(np.mean(y_unb))
    for k in range(N_EXPERTS):
        members = np.where(deploy_labels == k)[0]
        deploy_cluster_sizes.append(int(len(members)))
        if len(members) >= 5:
            mdl = lgb.LGBMRegressor(**_lgbm_params(ROUTE_SEED + k))
            mdl.fit(X_unb_K20[members], y_unb[members])
            deploy_experts.append(mdl)
            deploy_fallbacks.append(None)
        else:
            if len(members) > 0:
                const = float(np.mean(y_unb[members]))
            else:
                const = global_fallback
            deploy_experts.append(None)
            deploy_fallbacks.append(const)
    print(f"   deploy cluster sizes  = {deploy_cluster_sizes}  "
          f"(compactness={deploy_compact:.4f})")
    print(f"   tiny experts          = {sum(1 for m in deploy_experts if m is None)}")
    deploy_top_idx, deploy_top_sim = _route_top_k(deploy_centroids, fp_test, k=TOP_K_ROUTE)
    deploy_preds_top = np.zeros((n_test, TOP_K_ROUTE), dtype=np.float64)
    for col in range(TOP_K_ROUTE):
        expert_ids = deploy_top_idx[:, col]
        unique_eids = np.unique(expert_ids)
        col_pred = np.zeros(n_test, dtype=np.float64)
        for eid in unique_eids:
            rows = np.where(expert_ids == eid)[0]
            mdl = deploy_experts[int(eid)]
            if mdl is not None:
                col_pred[rows] = mdl.predict(X_te_K20[rows])
            else:
                col_pred[rows] = deploy_fallbacks[int(eid)]
        deploy_preds_top[:, col] = col_pred
    deploy_te = (
        TOP1_WEIGHT * deploy_preds_top[:, 0] + TOP2_WEIGHT * deploy_preds_top[:, 1]
    ).astype(np.float32)
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))
    print(f"   deploy te range = [{deploy_te.min():.3f}, {deploy_te.max():.3f}]  "
          f"mean={deploy_te.mean():.3f} std={deploy_te.std():.3f}")
    print(f"   te[unb_idx] RAE = {te_unb_rae:.4f}")

    # --- routing diagnostics on the 513 ---
    top1_counts = np.bincount(deploy_top_idx[:, 0], minlength=N_EXPERTS).tolist()
    top2_counts = np.bincount(deploy_top_idx[:, 1], minlength=N_EXPERTS).tolist()
    avg_top1_sim = float(deploy_top_sim[:, 0].mean())
    avg_top2_sim = float(deploy_top_sim[:, 1].mean())
    print(f"   top1 expert dist (513) = {top1_counts}  avg top1 sim = {avg_top1_sim:.3f}")
    print(f"   top2 expert dist (513) = {top2_counts}  avg top2 sim = {avg_top2_sim:.3f}")

    # --- save artifacts ---
    pred_oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_npy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(pred_oof_path, res["mean_oof"].astype(np.float32))
    np.save(te_npy_path, deploy_te)
    print(f"\n[save] {pred_oof_path}")
    print(f"[save] {te_npy_path}")

    summary = {
        "tag": TAG,
        "method": "sparse_MoE_top2_tanimoto_routing_K20_LGBM_experts",
        "paradigm": "scaffold_partition_specialised_experts_with_sparse_routing",
        "moe_config": {
            "n_experts": N_EXPERTS,
            "top_k_route": TOP_K_ROUTE,
            "top1_weight": TOP1_WEIGHT,
            "top2_weight": TOP2_WEIGHT,
            "kmeans_n_init": KMEANS_N_INIT,
            "kmeans_max_iter": KMEANS_MAX_ITER,
            "route_seed": ROUTE_SEED,
            "morgan_n_bits": int(fp_unb.shape[1]),
        },
        "k20_surviving_idx_in_117": [int(j) for j in surviving_K20],
        "k20_surviving_names": surviving_K20_names,
        "k20_family_counts": dict(nb2231["snapshots"]["20"]["family_counts"]),
        "lgbm_params_template": _lgbm_params(ROUTE_SEED),
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_unb": n_unb,
        "n_te": n_test,
        "n_unique_scaffolds": n_unique_scaf,
        "rae_anchor_chemprop_aux": rae_anchor,
        "per_seed_results": res["per_seed"],
        "cv_diagnostics": res["diagnostics"],
        "pooled_rae_mean_seeds": res["mean_rae"],
        "pooled_rae_std_seeds": res["std_rae"],
        "rae_of_mean_of_seed_oofs": res["rae_of_mean_oof"],
        "cv_wall_sec": round(cv_wall, 2),
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "verdict": verdict,
        "compare_nb2240_K20_ref": NB2240_REF_OOF_K20,
        "delta_vs_nb2240_K20": delta_nb2240,
        "deploy_cluster_sizes": deploy_cluster_sizes,
        "deploy_kmeans_compactness": float(deploy_compact),
        "deploy_top1_expert_counts_513": top1_counts,
        "deploy_top2_expert_counts_513": top2_counts,
        "deploy_avg_top1_sim_513": avg_top1_sim,
        "deploy_avg_top2_sim_513": avg_top2_sim,
        "deploy_te_min": float(deploy_te.min()),
        "deploy_te_max": float(deploy_te.max()),
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "te_unb_rae_in_sample": te_unb_rae,
        "pred_oof_path": str(pred_oof_path),
        "te_npy_path": str(te_npy_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   mean_rae (5 seeds)             = {res['mean_rae']:.4f} "
          f"+/- {res['std_rae']:.4f}")
    print(f"   verdict                        = {verdict}")
    print(f"   delta vs nb2240 K=20 (0.4682)  = {delta_nb2240:+.4f}")
    print(f"   deploy te range                = "
          f"[{deploy_te.min():.3f}, {deploy_te.max():.3f}]")
    print(f"   deploy cluster sizes           = {deploy_cluster_sizes}")
    print(f"   wall                           = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "pooled_rae_mean_seeds",
        "pooled_rae_std_seeds",
        "verdict",
        "delta_vs_nb2240_K20",
        "deploy_te_min",
        "deploy_te_max",
        "te_unb_rae_in_sample",
        "deploy_cluster_sizes",
    ):
        print(f"  {k}: {res.get(k)}")
