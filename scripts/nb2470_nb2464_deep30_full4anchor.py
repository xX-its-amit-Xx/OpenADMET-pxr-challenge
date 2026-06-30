"""nb2470 -- Deep-30 seed verification of nb2464 TDA Mapper (FULL 4-anchor pool).

CONTEXT:
    nb2464 reported pooled_rae_mean = 0.4428 +/- 0.0014 over 5 seeds (1001-1005)
    on the 4-anchor pool (nb2240, nb730, nb562, chemprop_aux). Memory rule:
    5-seed std < 0.001 is severely under-dispersed (see feedback_cycle160_deep
    _verify_dispersion). The cycle-160 nb2060 audit found 5-seed std 0.00087
    while 30-seed std was 0.00408 (4.7x higher), and cycle-163 nb2095 confirmed
    a 4.12x ratio. nb2464's 0.0014 is in the same suspect band.

    This script re-runs the EXACT nb2464 pipeline with kf_seeds 1001..1030
    (30 seeds), produces deep-30 mean+std, Welch t-test vs the 5-seed subset,
    and a binary verdict.

PROTOCOL:
    Identical to nb2464 (lens = MACCS+Morgan+9physchem -> StdScale -> PCA-32,
    KeplerMapper l2norm lens, resolution=10, gain=0.3,
    AgglomerativeClustering(n_clusters=2) per cube, per-cluster SLSQP simplex
    blend on TRAIN fold, soft-membership avg on VALID).

GATE:
    - 30-seed mean < 0.4570 AND 30-seed std < 0.005  -> "DEEP30_HOLDS"
    - else                                            -> "DEEP30_FAILS"

Outputs:
    scripts/nb2470_nb2464_deep30_full4anchor.py
    data/processed/nb2470_summary.json
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
from rdkit import RDLogger
from scipy import stats as sp_stats
from scipy.optimize import minimize
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")

from pxr.chem import morgan_fp_batch, bemis_murcko, compute_physchem
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

try:
    import kmapper as km
    HAVE_KMAPPER = True
except Exception as e:
    HAVE_KMAPPER = False
    KMAPPER_IMPORT_ERR = str(e)

TAG = "nb2470"
BASELINE_TAG = "nb2464"

# ---- mapper hyperparams (identical to nb2464) ----
PCA_DIM = 32
RESOLUTION = 10
GAIN = 0.3
MIN_CLUSTER_N = 10

# ---- CV setup ----
N_FOLDS = 5
KF_SEEDS = list(range(1001, 1031))  # 30 seeds
SEEDS_5_SUBSET = [1001, 1002, 1003, 1004, 1005]

# ---- Gate ----
GATE_MEAN = 0.4570
GATE_STD = 0.005

NB2240_OOF_PATH = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"
NB730_OOF_PATH = DATA_PROCESSED / "nb730_pred_oof.npy"
NB562_OOF_PATH = DATA_PROCESSED / "nb562_pred_oof.npy"
CHEMPROP_AUX_OOF_PATH = DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy"


def _slsqp_convex(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    k = P.shape[1]
    w0 = np.full(k, 1.0 / k)

    def obj(w):
        r = P @ w - y
        return float(r @ r)

    def grad(w):
        r = P @ w - y
        return 2.0 * (P.T @ r)

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0),
             "jac": lambda w: np.ones_like(w)}]
    bnds = [(0.0, 1.0)] * k
    res = minimize(obj, w0, jac=grad, method="SLSQP", bounds=bnds,
                   constraints=cons,
                   options={"maxiter": 200, "ftol": 1e-9, "disp": False})
    w = np.clip(res.x, 0.0, 1.0)
    s = w.sum()
    if s > 0:
        w = w / s
    else:
        w = np.full(k, 1.0 / k)
    return w


def _build_lens(te_smiles):
    n_test = len(te_smiles)
    X_morgan = morgan_fp_batch(te_smiles, radius=2, n_bits=2048).astype(np.float32)
    maccs_path = DATA_PROCESSED / "te_maccs.npy"
    if maccs_path.exists():
        X_maccs = np.load(maccs_path).astype(np.float32)
        if X_maccs.shape[0] != n_test:
            X_maccs = None
    else:
        X_maccs = None
    phys_rows = []
    for smi in te_smiles:
        d = compute_physchem(smi)
        if d is None:
            phys_rows.append([np.nan] * 9)
        else:
            phys_rows.append([
                d.get("mw", np.nan), d.get("logp", np.nan), d.get("tpsa", np.nan),
                d.get("hbd", np.nan), d.get("hba", np.nan),
                d.get("rotbonds", np.nan), d.get("fsp3", np.nan),
                d.get("rings", np.nan), d.get("charge", np.nan),
            ])
    X_phys = np.asarray(phys_rows, dtype=np.float32)
    col_med = np.nanmedian(X_phys, axis=0)
    for j in range(X_phys.shape[1]):
        m = np.isnan(X_phys[:, j])
        if m.any():
            X_phys[m, j] = col_med[j]
    parts = [X_morgan, X_phys]
    if X_maccs is not None:
        parts.insert(0, X_maccs)
    X = np.concatenate(parts, axis=1).astype(np.float32)
    X = np.where(np.isfinite(X), X, 0.0)
    col_std = X.std(axis=0)
    keep_cols = col_std > 1e-12
    if keep_cols.sum() < X.shape[1]:
        X = X[:, keep_cols]
    scaler = StandardScaler(with_mean=False)
    X_std = scaler.fit_transform(X).astype(np.float64)
    X_std = np.where(np.isfinite(X_std), X_std, 0.0)
    pca = PCA(n_components=min(PCA_DIM, X_std.shape[1], X_std.shape[0] - 1),
              random_state=0)
    X_pca = pca.fit_transform(X_std).astype(np.float64)
    return X_pca, int(X.shape[1])


def _mapper_clusters(X_lens: np.ndarray, n_total: int):
    mapper = km.KeplerMapper(verbose=0)
    proj = mapper.fit_transform(X_lens, projection="l2norm")
    graph = mapper.map(
        proj,
        X_lens,
        cover=km.Cover(n_cubes=RESOLUTION, perc_overlap=GAIN),
        clusterer=AgglomerativeClustering(n_clusters=2),
    )
    clusters = []
    for node_id, member_ids in graph["nodes"].items():
        ids = np.asarray(member_ids, dtype=int)
        ids = ids[(ids >= 0) & (ids < n_total)]
        if len(ids) >= MIN_CLUSTER_N:
            clusters.append(ids)
    return clusters


def _per_cluster_blend_predict(
    P_train, y_train, clusters_train,
    P_query, M_query_full, anchors_n, fallback,
):
    C = len(clusters_train)
    K = P_train.shape[1]
    if C == 0:
        return fallback.copy(), np.zeros((0, K))
    W_per_cluster = np.zeros((C, K), dtype=np.float64)
    valid_mask = np.zeros(C, dtype=bool)
    for c_idx, tr_loc in enumerate(clusters_train):
        if len(tr_loc) < MIN_CLUSTER_N:
            continue
        w = _slsqp_convex(P_train[tr_loc], y_train[tr_loc])
        W_per_cluster[c_idx] = w
        valid_mask[c_idx] = True
    n_q = P_query.shape[0]
    pred_clusters = np.zeros((n_q, C), dtype=np.float64)
    for c_idx in range(C):
        if not valid_mask[c_idx]:
            continue
        pred_clusters[:, c_idx] = P_query @ W_per_cluster[c_idx]
    M_valid = M_query_full[:, valid_mask]
    pred_valid = pred_clusters[:, valid_mask]
    mem_sum = M_valid.sum(axis=1)
    weighted = (M_valid * pred_valid).sum(axis=1)
    out = np.where(mem_sum > 0, weighted / np.where(mem_sum > 0, mem_sum, 1.0),
                   fallback)
    return out, W_per_cluster


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- deep-30 verification of {BASELINE_TAG} TDA Mapper (4-anchor)")
    print("=" * 78)
    if not HAVE_KMAPPER:
        out = {"tag": TAG, "status": "INSTALL_FAILED", "error": KMAPPER_IMPORT_ERR}
        with open(DATA_PROCESSED / f"{TAG}_summary.json", "w") as f:
            json.dump(out, f, indent=2)
        print("INSTALL_FAILED")
        return out

    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test} n_unb={n_unb}")

    anchor_oof = {
        "nb2240": np.load(NB2240_OOF_PATH).astype(np.float64),
        "nb730": np.load(NB730_OOF_PATH).astype(np.float64),
        "nb562": np.load(NB562_OOF_PATH).astype(np.float64),
        "chemprop_aux": np.load(CHEMPROP_AUX_OOF_PATH).astype(np.float64),
    }
    anchor_names = ["nb2240", "nb730", "nb562", "chemprop_aux"]
    K = len(anchor_names)
    P_unb = np.column_stack([anchor_oof[k] for k in anchor_names])
    rae_anchors = {k: float(rae(y_unb, anchor_oof[k])) for k in anchor_names}
    for k in anchor_names:
        print(f"   anchor {k:14s}  unb-RAE = {rae_anchors[k]:.4f}")

    print("\n" + "-" * 78)
    print(f"LENS: MACCS+Morgan+physchem -> StdScale -> PCA-{PCA_DIM}")
    print("-" * 78)
    X_pca_te, raw_dim = _build_lens(te_smiles)
    X_pca_unb = X_pca_te[unb_idx]
    print(f"   raw_dim={raw_dim}  PCA dim={X_pca_te.shape[1]}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    print("\n" + "-" * 78)
    print(f"SCAFFOLD CV  folds={N_FOLDS}  n_seeds={len(KF_SEEDS)}  "
          f"(seeds {KF_SEEDS[0]}..{KF_SEEDS[-1]})")
    print("-" * 78)

    per_seed_summary = []
    for kf_seed in KF_SEEDS:
        splits = scaffold_kfold_indices(unb_scaffolds, n_splits=N_FOLDS,
                                        shuffle=True, seed=kf_seed)
        oof = np.full(n_unb, np.nan, dtype=np.float64)
        for f_idx, (tr_loc, va_loc) in enumerate(splits):
            X_tr = X_pca_unb[tr_loc]
            X_va = X_pca_unb[va_loc]
            X_comb = np.vstack([X_tr, X_va])
            clusters_comb = _mapper_clusters(X_comb, len(X_comb))
            n_tr = len(tr_loc)
            tr_clusters_local = []
            va_membership_per_cluster = []
            for ids in clusters_comb:
                tr_ids = ids[ids < n_tr]
                va_ids = ids[ids >= n_tr] - n_tr
                if len(tr_ids) >= MIN_CLUSTER_N:
                    tr_clusters_local.append(tr_ids)
                    va_membership_per_cluster.append(va_ids)
            n_va = len(va_loc)
            C_valid = len(tr_clusters_local)
            M_va = np.zeros((n_va, C_valid), dtype=np.float64)
            for c_idx, va_ids in enumerate(va_membership_per_cluster):
                M_va[va_ids, c_idx] = 1.0
            P_tr = P_unb[tr_loc]
            y_tr = y_unb[tr_loc]
            P_va = P_unb[va_loc]
            va_fallback = anchor_oof["nb2240"][va_loc]
            pred_va, _ = _per_cluster_blend_predict(
                P_tr, y_tr, tr_clusters_local,
                P_va, M_va, K, va_fallback,
            )
            oof[va_loc] = pred_va
        if np.isnan(oof).any():
            oof[np.isnan(oof)] = anchor_oof["nb2240"][np.isnan(oof)]
        rae_seed = float(rae(y_unb, oof))
        per_seed_summary.append({"kf_seed": int(kf_seed), "pooled_rae": rae_seed})
        print(f"   kf_seed={kf_seed}  pooled_RAE={rae_seed:.4f}")

    all_rae = np.array([r["pooled_rae"] for r in per_seed_summary])
    mean_30 = float(all_rae.mean())
    std_30 = float(all_rae.std(ddof=0))
    std_30_unbiased = float(all_rae.std(ddof=1))
    min_30 = float(all_rae.min())
    max_30 = float(all_rae.max())

    # 5-seed subset (1001-1005)
    five_mask = np.array([s["kf_seed"] in SEEDS_5_SUBSET for s in per_seed_summary])
    rae_5 = all_rae[five_mask]
    mean_5 = float(rae_5.mean())
    std_5 = float(rae_5.std(ddof=0))
    std_5_unbiased = float(rae_5.std(ddof=1))

    # Welch t-test 5-subset vs remaining 25
    rae_25 = all_rae[~five_mask]
    try:
        welch = sp_stats.ttest_ind(rae_5, rae_25, equal_var=False)
        welch_t = float(welch.statistic)
        welch_p = float(welch.pvalue)
    except Exception:
        welch_t = float("nan")
        welch_p = float("nan")

    # Under-dispersion ratio (deep-30 ddof=1 / 5-seed ddof=1)
    if std_5_unbiased > 0:
        under_dispersion_ratio = std_30_unbiased / std_5_unbiased
    else:
        under_dispersion_ratio = float("inf")

    # Gate
    gate_mean_pass = mean_30 < GATE_MEAN
    gate_std_pass = std_30 < GATE_STD
    gate_pass = gate_mean_pass and gate_std_pass
    verdict = "DEEP30_HOLDS" if gate_pass else "DEEP30_FAILS"

    print("\n" + "=" * 78)
    print(f"=== {TAG} DEEP-30 SUMMARY ===")
    print(f"   30-seed mean = {mean_30:.4f}")
    print(f"   30-seed std  = {std_30:.4f}  (ddof=1: {std_30_unbiased:.4f})")
    print(f"   30-seed min  = {min_30:.4f}    max  = {max_30:.4f}")
    print(f"   5-seed (1001-1005) mean = {mean_5:.4f} +/- {std_5:.4f}")
    print(f"   Welch t-test 5-subset vs other 25:  t = {welch_t:.4f}  p = {welch_p:.4f}")
    print(f"   under-dispersion ratio (std_30 / std_5) = {under_dispersion_ratio:.2f}x")
    print(f"   GATE mean < {GATE_MEAN:.4f} -> {'PASS' if gate_mean_pass else 'FAIL'}")
    print(f"   GATE std  < {GATE_STD:.4f}  -> {'PASS' if gate_std_pass else 'FAIL'}")
    print(f"   VERDICT: {verdict}")
    print("=" * 78)

    summary = {
        "tag": TAG,
        "baseline_tag": BASELINE_TAG,
        "method": "tda_mapper_cluster_conditional_4anchor_slsqp_blend_DEEP30",
        "anchor_pool": anchor_names,
        "anchor_oof_files": {
            "nb2240": "nb2240_mean_bag_oof_K20.npy",
            "nb730": "nb730_pred_oof.npy",
            "nb562": "nb562_pred_oof.npy",
            "chemprop_aux": "nb1133_chemprop_aux_pred_oof.npy",
        },
        "anchor_in_rae": rae_anchors,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "n_folds": int(N_FOLDS),
        "kf_seeds": [int(s) for s in KF_SEEDS],
        "per_seed_table": per_seed_summary,
        "mean_30": mean_30,
        "std_30": std_30,
        "std_30_unbiased_ddof1": std_30_unbiased,
        "min_30": min_30,
        "max_30": max_30,
        "subset_5_seeds": [int(s) for s in SEEDS_5_SUBSET],
        "subset_5_mean": mean_5,
        "subset_5_std": std_5,
        "subset_5_std_unbiased_ddof1": std_5_unbiased,
        "welch_t_5_vs_25": welch_t,
        "welch_p_5_vs_25": welch_p,
        "under_dispersion_ratio_std30_over_std5": under_dispersion_ratio,
        "gate_mean_threshold": GATE_MEAN,
        "gate_std_threshold": GATE_STD,
        "gate_mean_pass": bool(gate_mean_pass),
        "gate_std_pass": bool(gate_std_pass),
        "gate_pass": bool(gate_pass),
        "verdict": verdict,
        "lens": {
            "raw_dim": int(raw_dim),
            "pca_dim": int(X_pca_te.shape[1]),
            "projection": "l2norm",
            "resolution": RESOLUTION,
            "gain": GAIN,
            "min_cluster_n": MIN_CLUSTER_N,
        },
        "baseline_reference": {
            "nb2464_pooled_mean_5seeds": 0.4428,
            "nb2464_pooled_std_5seeds": 0.0014,
        },
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_30",
        "std_30",
        "min_30",
        "max_30",
        "subset_5_mean",
        "subset_5_std",
        "welch_t_5_vs_25",
        "welch_p_5_vs_25",
        "under_dispersion_ratio_std30_over_std5",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
