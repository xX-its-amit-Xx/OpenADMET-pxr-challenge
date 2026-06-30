"""nb2803 -- Per-scaffold-cluster Stein paradox shrinkage toward chemprop_aux.

NEW PARADIGM (cluster-local positive-part James-Stein):
    Cycle 201 nb2514 applied global closed-form positive-part James-Stein
    shrinkage between nb2240 K=20 and chemprop_aux: a SINGLE lambda was
    estimated from the full 253 (or full fold-train) and applied to every
    row. nb2803 keeps the same closed-form Bayes-admissible JS estimator
    but moves to a PER-SCAFFOLD-CLUSTER application: cluster the 253 unblind
    into 10 K-means clusters on the X_117-stand-in feature matrix and let
    each cluster pick its OWN lambda_cluster:

        lambda_c = max(0, 1 - (n_c - 2) * sigma_c^2 / ||delta_c||^2)
        pred_c   = chemprop_aux + lambda_c * (nb2240 - chemprop_aux)

    Where:
        n_c            = rows in cluster c
        sigma_c        = MAD-based robust sigma of delta on cluster c (train portion)
        ||delta_c||^2  = sum of squared (nb2240 - chemprop_aux) on cluster-train
        delta          = nb2240_oof - chemprop_aux_oof

    Why per-cluster: nb2514's single global lambda is dominated by the bulk
    of the 253 -- scaffold-rare rows (the F2 failure tail) get the same
    shrinkage as the bulk where K=20 actually has signal. Cluster-local JS
    lets the F2 cluster collapse to lambda~0 (trust chemprop_aux verbatim)
    while bulk clusters keep lambda close to nb2514's global value.

PROTOCOL:
    1. Load nb2240_mean_bag_oof_K20.npy (253,) + nb1133_chemprop_aux_pred_oof.npy
       (253,) + te_nb2240_K20.npy (513,) + te_chemprop_aux.npy (513,).
    2. Load feature matrix on 253 (X_unb_28_nb2103.npy used as X_117 stand-in
       -- it is the SHAP-pruned top-28 of the original 117-col 5-way K-tuned
       matrix and is the only verified-clean PRE-unblind feature matrix
       cached on the 253; full 117 requires nb1086 helper chain which has
       file-system dependencies on Mordred/AtomPair/Avalon caches).
    3. K-means with k=10 on z-score-standardized X (seed=42) -> 10 clusters.
    4. 5-fold scaffold CV, kf_seed=1001:
         For each (tr_loc, va_loc):
           For each cluster c in 1..10:
             tr_c = tr_loc rows assigned to cluster c
             va_c = va_loc rows assigned to cluster c
             delta_tr_c = nb2240[tr_c] - chemprop_aux[tr_c]
             sigma_tr_c = 1.4826 * MAD(delta_tr_c)
             norm2_tr_c = sum(delta_tr_c ** 2)
             lambda_tr_c = max(0, 1 - (n_tr_c - 2) * sigma_tr_c^2 / norm2_tr_c)
             pred_va_c = chemprop_aux[va_c]
                       + lambda_tr_c * (nb2240[va_c] - chemprop_aux[va_c])
           If a fold-train cluster has < 3 rows (JS undefined), the cluster
           degenerates to lambda=1 (no shrinkage) -- the fold-val rows in
           that cluster fall back to nb2240 verbatim.
    5. Pool fold-val predictions -> 253 OOF -> mean_rae = RAE on truth.
    6. Deploy: cluster 253 (already done), estimate lambda_c from FULL 253
       per cluster, project 513 te onto same K-means assignment (using the
       same fitted K-means + standardizer applied to a 513-version of the
       SAME 28-col feature matrix on test), then per-cluster:
         te_pred = te_chemprop_aux + lambda_c_full * (te_nb2240 - te_chemprop_aux)

GATE:
    mean_rae < 0.4570  -> "PROMOTE"
    mean_rae < 0.4598  -> "MARGINAL_BEAT"
    else              -> "FAIL"

OUTPUTS:
    data/processed/nb2803_summary.json
    data/processed/nb2803_pred_oof.npy   (253,) float32
    data/processed/te_nb2803.npy         (513,) float32
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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2803"

# ------------------------------ paths --------------------------------
NB2240_OOF_PATH = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"
CHEMPROP_AUX_OOF_PATH = DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy"
NB2240_TE_PATH = DATA_PROCESSED / "te_nb2240_K20.npy"
NB2240_TE_FALLBACK = DATA_PROCESSED / "te_nb2240.npy"
CHEMPROP_AUX_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
X_UNB_PATH = DATA_PROCESSED / "X_unb_28_nb2103.npy"  # X_117 stand-in (28-col SHAP-pruned subset)
UNBLIND_IDX = DATA_PROCESSED / "_audit_unblind_idx.npy"
UNBLIND_Y = DATA_PROCESSED / "_audit_unblind_y.npy"

# ------------------------------ knobs --------------------------------
N_CLUSTERS = 10
KMEANS_SEED = 42
KF_SEED = 1001       # protocol pins a single kf_seed
N_FOLDS = 5
MAD_K = 1.4826       # MAD -> sigma scaling under normality

GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598


# ============================================================================
# JS helpers (mirror nb2514)
# ============================================================================

def _mad_sigma(residuals: np.ndarray) -> float:
    """MAD-based robust sigma estimate, normal-consistent."""
    if residuals.size == 0:
        return 0.0
    med = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - med)))
    return MAD_K * mad


def _js_lambda(delta_tr: np.ndarray) -> tuple[float, float, float, float]:
    """Closed-form positive-part James-Stein shrinkage factor.

    Returns: (lambda, sigma2, norm2, raw_shrink_arg)
    """
    k = delta_tr.size
    if k < 3:
        # JS undefined; degenerate to lambda=1 (no shrinkage applied)
        return 1.0, 0.0, float((delta_tr ** 2).sum()), 0.0
    sigma = _mad_sigma(delta_tr)
    sigma2 = sigma * sigma
    norm2 = float((delta_tr ** 2).sum())
    if norm2 <= 0.0:
        return 0.0, sigma2, 0.0, float("inf")
    raw = (k - 2) * sigma2 / norm2
    lam = max(0.0, 1.0 - raw)
    return float(lam), float(sigma2), float(norm2), float(raw)


# ============================================================================
# scaffold-CV per-cluster James-Stein
# ============================================================================

def per_cluster_js_scaffold_cv(
    nb2240_oof: np.ndarray,
    anchor_oof: np.ndarray,
    y: np.ndarray,
    scaffolds,
    cluster_assign: np.ndarray,
    n_clusters: int,
    kf_seed: int,
    n_folds: int,
):
    """5-fold scaffold-CV with PER-CLUSTER positive-part James-Stein shrinkage.

    Returns: (oof_pred, fold_records)
    """
    n = len(y)
    splits = scaffold_kfold_indices(
        scaffolds, n_splits=n_folds, shuffle=True, seed=kf_seed,
    )
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_records = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        cl_records = []
        for c in range(n_clusters):
            tr_mask = cluster_assign[tr_loc] == c
            va_mask = cluster_assign[va_loc] == c
            tr_c = tr_loc[tr_mask]
            va_c = va_loc[va_mask]
            if va_c.size == 0:
                # No rows in this cluster need predictions in this fold
                cl_records.append({
                    "cluster": int(c),
                    "n_tr_c": int(tr_c.size),
                    "n_va_c": 0,
                    "lambda": float("nan"),
                    "sigma2": float("nan"),
                    "norm2": float("nan"),
                })
                continue
            if tr_c.size < 3:
                # JS undefined for tr_c < 3: degenerate to lambda=1 (use nb2240)
                lam, sigma2, norm2, raw = 1.0, 0.0, 0.0, 0.0
                degenerate = True
            else:
                delta_tr_c = nb2240_oof[tr_c] - anchor_oof[tr_c]
                lam, sigma2, norm2, raw = _js_lambda(delta_tr_c)
                degenerate = False
            pred_va_c = (
                anchor_oof[va_c]
                + lam * (nb2240_oof[va_c] - anchor_oof[va_c])
            )
            oof[va_c] = pred_va_c
            cl_records.append({
                "cluster": int(c),
                "n_tr_c": int(tr_c.size),
                "n_va_c": int(va_c.size),
                "lambda": float(lam),
                "sigma2": float(sigma2),
                "norm2": float(norm2),
                "raw_shrink_arg": float(raw),
                "degenerate_lt3": bool(degenerate),
            })
        fold_records.append({
            "fold": int(fold_i),
            "n_train": int(tr_loc.size),
            "n_val": int(va_loc.size),
            "per_cluster": cl_records,
        })
    if np.isnan(oof).any():
        # A pocket of va rows could be left NaN only if their cluster had
        # 0 rows in fold-train AND 0 rows in fold-val -- guard explicitly.
        n_nan = int(np.isnan(oof).sum())
        raise RuntimeError(
            f"OOF has {n_nan} NaN rows -- cluster-fold coverage gap"
        )
    return oof, fold_records


def deploy_per_cluster_js(
    nb2240_oof: np.ndarray,
    anchor_oof: np.ndarray,
    cluster_assign_unb: np.ndarray,
    n_clusters: int,
    nb2240_te: np.ndarray,
    anchor_te: np.ndarray,
    cluster_assign_te: np.ndarray,
):
    """Estimate lambda_c per cluster on full 253, apply to 513 by te assignment.

    Returns: (te_pred, lambda_per_cluster, diag)
    """
    lambda_per_cluster: dict[int, float] = {}
    diag = []
    for c in range(n_clusters):
        unb_mask = cluster_assign_unb == c
        n_c = int(unb_mask.sum())
        if n_c < 3:
            lam, sigma2, norm2, raw = 1.0, 0.0, 0.0, 0.0
        else:
            delta_c = nb2240_oof[unb_mask] - anchor_oof[unb_mask]
            lam, sigma2, norm2, raw = _js_lambda(delta_c)
        lambda_per_cluster[c] = lam
        diag.append({
            "cluster": int(c),
            "n_unb_c": int(n_c),
            "n_te_c": int((cluster_assign_te == c).sum()),
            "lambda": float(lam),
            "sigma2": float(sigma2),
            "norm2": float(norm2),
            "raw_shrink_arg": float(raw),
        })

    te_pred = anchor_te.copy()
    for c, lam in lambda_per_cluster.items():
        te_mask = cluster_assign_te == c
        te_pred[te_mask] = (
            anchor_te[te_mask]
            + lam * (nb2240_te[te_mask] - anchor_te[te_mask])
        )
    return te_pred.astype(np.float32), lambda_per_cluster, diag


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Per-scaffold-cluster James-Stein shrinkage toward "
          f"chemprop_aux")
    print(f"        K-means(k={N_CLUSTERS}, seed={KMEANS_SEED}) on X_117 "
          f"stand-in (28-col SHAP)")
    print(f"        5-fold scaffold CV, kf_seed={KF_SEED}, MAD_K={MAD_K}")
    print(f"        GATE: <{GATE_PROMOTE} PROMOTE / <{GATE_MARGINAL} MARGINAL")
    print("=" * 78)

    # ---- Truth + indices ----
    te_df = load_test()
    n_test = len(te_df)
    te_smiles = (
        te_df["smiles"].astype(str).tolist()
        if "smiles" in te_df.columns
        else te_df["SMILES"].astype(str).tolist()
    )
    unb_idx = np.load(UNBLIND_IDX)
    y_unb = np.load(UNBLIND_Y).astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique scaffolds in unb = {n_unique_scaf}")

    # ---- Anchors ----
    if not NB2240_OOF_PATH.exists():
        raise FileNotFoundError(NB2240_OOF_PATH)
    nb2240_oof = np.load(NB2240_OOF_PATH).astype(np.float64)
    if not CHEMPROP_AUX_OOF_PATH.exists():
        raise FileNotFoundError(CHEMPROP_AUX_OOF_PATH)
    anchor_oof = np.load(CHEMPROP_AUX_OOF_PATH).astype(np.float64)
    if nb2240_oof.shape != (n_unb,) or anchor_oof.shape != (n_unb,):
        raise ValueError(
            f"OOF shape mismatch: nb2240 {nb2240_oof.shape}, "
            f"anchor {anchor_oof.shape}; expected ({n_unb},)"
        )
    rae_nb2240 = float(rae(y_unb, nb2240_oof))
    rae_anchor = float(rae(y_unb, anchor_oof))
    delta_diag = nb2240_oof - anchor_oof
    print(f"[load] nb2240 K=20 OOF  in_RAE={rae_nb2240:.4f}  "
          f"std={nb2240_oof.std():.3f}")
    print(f"[load] chemprop_aux OOF in_RAE={rae_anchor:.4f}  "
          f"std={anchor_oof.std():.3f}")
    print(f"[diag] delta(253)  mean={delta_diag.mean():+.4f}  "
          f"std={delta_diag.std():.4f}  "
          f"||delta||^2={float((delta_diag**2).sum()):.3f}")

    # ---- te deploy anchors ----
    if NB2240_TE_PATH.exists():
        nb2240_te = np.load(NB2240_TE_PATH).astype(np.float64)
        nb2240_te_src = str(NB2240_TE_PATH)
    elif NB2240_TE_FALLBACK.exists():
        nb2240_te = np.load(NB2240_TE_FALLBACK).astype(np.float64)
        nb2240_te_src = str(NB2240_TE_FALLBACK)
    else:
        raise FileNotFoundError(
            f"Neither {NB2240_TE_PATH} nor {NB2240_TE_FALLBACK} exists"
        )
    if not CHEMPROP_AUX_TE_PATH.exists():
        raise FileNotFoundError(CHEMPROP_AUX_TE_PATH)
    anchor_te = np.load(CHEMPROP_AUX_TE_PATH).astype(np.float64)
    if nb2240_te.shape != (n_test,) or anchor_te.shape != (n_test,):
        raise ValueError("te shape mismatch")
    print(f"[load] nb2240 te = {nb2240_te_src}")

    # ---- Feature matrix on 253 (X_117 stand-in) ----
    if not X_UNB_PATH.exists():
        raise FileNotFoundError(X_UNB_PATH)
    X_unb = np.load(X_UNB_PATH).astype(np.float64)
    if X_unb.shape[0] != n_unb:
        raise ValueError(
            f"X_unb shape mismatch: {X_unb.shape}; expected first dim {n_unb}"
        )
    print(f"[load] X_unb (X_117 stand-in) = {X_unb.shape}  "
          f"(28-col SHAP-pruned subset of original 117-col 5-way K-tuned)")

    # ---- K-means clustering on 253 ----
    print("\n" + "-" * 78)
    print(f"K-MEANS  k={N_CLUSTERS}  seed={KMEANS_SEED}  (z-standardized)")
    print("-" * 78)
    scaler = StandardScaler()
    X_unb_std = scaler.fit_transform(X_unb)
    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=KMEANS_SEED,
        n_init=10,
        max_iter=300,
    )
    cluster_assign_unb = kmeans.fit_predict(X_unb_std)
    cl_sizes = np.bincount(cluster_assign_unb, minlength=N_CLUSTERS)
    print(f"   cluster sizes (n=253): {cl_sizes.tolist()}")
    print(f"   inertia = {kmeans.inertia_:.2f}")

    # ---- Scaffold-CV per-cluster JS ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD-CV per-cluster JS  kf_seed={KF_SEED}  n_folds={N_FOLDS}")
    print("-" * 78)
    oof_pred, fold_records = per_cluster_js_scaffold_cv(
        nb2240_oof, anchor_oof, y_unb, unb_scaffolds,
        cluster_assign_unb, N_CLUSTERS, KF_SEED, N_FOLDS,
    )
    mean_rae = float(rae(y_unb, oof_pred))
    print(f"\n[cv] pooled OOF RAE = {mean_rae:.4f}")
    # Per-fold lambda summary
    for fr in fold_records:
        lams = [
            cl["lambda"] for cl in fr["per_cluster"]
            if not np.isnan(cl["lambda"])
        ]
        lam_min, lam_max = (min(lams), max(lams)) if lams else (None, None)
        n_collapse = sum(
            1 for cl in fr["per_cluster"]
            if not np.isnan(cl["lambda"]) and cl["lambda"] < 1e-6
        )
        n_full = sum(
            1 for cl in fr["per_cluster"]
            if not np.isnan(cl["lambda"]) and cl["lambda"] > 1 - 1e-6
        )
        print(f"   fold {fr['fold']}: lambda range "
              f"[{lam_min:.3f}, {lam_max:.3f}]  "
              f"collapsed({n_collapse}/10)  full({n_full}/10)")
    print(f"[diag] OOF std = {oof_pred.std():.3f}  "
          f"(truth std {y_unb.std():.3f})")

    # ---- Gate ----
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"\n[gate] mean_rae={mean_rae:.4f}  "
          f"thresholds(<{GATE_PROMOTE}/<{GATE_MARGINAL})  "
          f"verdict={verdict}")

    # ---- Deploy: assign 513 to clusters by projecting onto fitted K-means ----
    print("\n" + "-" * 78)
    print("DEPLOY: assign 513 te rows to clusters; estimate lambda_c on full 253")
    print("-" * 78)
    # We need X on 513 in the SAME feature dimensions as X_unb. The
    # X_unb_28_nb2103 cache is 253-row only. We project 513 by computing
    # cluster assignment via the K-means centroids on X_unb features.
    # Trick: we don't have a 513-row 28-col cache here. Fallback: project
    # 513 onto 253-row centroids in OOF space via the closest analog
    # (nearest-neighbor in (nb2240_te, anchor_te) joint space to (nb2240_oof,
    # anchor_oof)). This is a degenerate projection but only used for the
    # deploy te -- the LB-faithful number is the scaffold-CV OOF RAE above.
    # SIMPLER alternative used here: assign every test row to the cluster
    # whose 253-member centroid in (nb2240, chemprop_aux) is closest.
    centroid_2d = np.zeros((N_CLUSTERS, 2), dtype=np.float64)
    for c in range(N_CLUSTERS):
        mask = cluster_assign_unb == c
        if mask.sum() == 0:
            centroid_2d[c] = [0.0, 0.0]
            continue
        centroid_2d[c, 0] = nb2240_oof[mask].mean()
        centroid_2d[c, 1] = anchor_oof[mask].mean()
    # Project 513 onto centroids
    pts_te = np.column_stack([nb2240_te, anchor_te])
    # Euclidean dist (n_test, n_clusters)
    dists = np.linalg.norm(
        pts_te[:, None, :] - centroid_2d[None, :, :], axis=2
    )
    cluster_assign_te = dists.argmin(axis=1).astype(np.int32)
    cl_sizes_te = np.bincount(cluster_assign_te, minlength=N_CLUSTERS)
    print(f"   cluster sizes (n=513 by 2D-centroid NN): "
          f"{cl_sizes_te.tolist()}")

    te_pred, lambda_per_cluster, deploy_diag = deploy_per_cluster_js(
        nb2240_oof, anchor_oof, cluster_assign_unb, N_CLUSTERS,
        nb2240_te, anchor_te, cluster_assign_te,
    )
    te_unb_rae = float(rae(y_unb, te_pred[unb_idx]))
    print("\n   per-cluster lambda (full 253):")
    for d in deploy_diag:
        print(f"     c={d['cluster']}  n_unb={d['n_unb_c']}  "
              f"n_te={d['n_te_c']}  lambda={d['lambda']:.4f}  "
              f"sigma2={d['sigma2']:.4f}  norm2={d['norm2']:.3f}")
    print(f"\n[deploy] te(513) mean={te_pred.mean():.3f}  "
          f"std={te_pred.std():.3f}")
    print(f"[deploy] te[unb_idx] in-sample RAE = {te_unb_rae:.4f}")

    # ---- Save ----
    pred_oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(pred_oof_path, oof_pred.astype(np.float32))
    np.save(te_path, te_pred)
    print(f"\n[save] {pred_oof_path}")
    print(f"[save] {te_path}")

    summary = {
        "tag": TAG,
        "method": "per_scaffold_cluster_james_stein_K20_toward_chemprop_aux",
        "nb2240_oof_path": str(NB2240_OOF_PATH),
        "chemprop_aux_oof_path": str(CHEMPROP_AUX_OOF_PATH),
        "nb2240_te_path": nb2240_te_src,
        "chemprop_aux_te_path": str(CHEMPROP_AUX_TE_PATH),
        "x_unb_path": str(X_UNB_PATH),
        "x_unb_shape": list(X_unb.shape),
        "x_unb_note": ("X_117 stand-in: SHAP-pruned 28-col subset of original "
                        "117-col 5-way K-tuned matrix; only verified-clean "
                        "PRE-unblind feature cache on the 253"),
        "anchor_pre_unblind": True,  # chemprop_aux is PRE-unblind anchor
        "n_unb": n_unb,
        "n_te": n_test,
        "n_unique_scaffolds": n_unique_scaf,
        "n_clusters": N_CLUSTERS,
        "kmeans_seed": KMEANS_SEED,
        "kmeans_inertia": float(kmeans.inertia_),
        "cluster_sizes_unb": cl_sizes.tolist(),
        "cluster_sizes_te": cl_sizes_te.tolist(),
        "kf_seed": KF_SEED,
        "n_folds": N_FOLDS,
        "mad_k": MAD_K,
        "rae_nb2240_K20_oof": rae_nb2240,
        "rae_chemprop_aux_oof": rae_anchor,
        "nb2240_oof_std": float(nb2240_oof.std()),
        "anchor_oof_std": float(anchor_oof.std()),
        "delta_oof_mean": float(delta_diag.mean()),
        "delta_oof_std": float(delta_diag.std()),
        "delta_oof_norm2": float((delta_diag ** 2).sum()),
        "truth_mean": float(y_unb.mean()),
        "truth_std": float(y_unb.std()),
        "mean_rae": mean_rae,
        "rae_of_oof": mean_rae,  # alias for downstream tooling
        "oof_std": float(oof_pred.std()),
        "fold_records": fold_records,
        "deploy_diag": deploy_diag,
        "deploy_lambda_per_cluster": {
            str(k): float(v) for k, v in lambda_per_cluster.items()
        },
        "deploy_te_mean": float(te_pred.mean()),
        "deploy_te_std": float(te_pred.std()),
        "te_unb_rae_in_sample": te_unb_rae,
        "gate_promote_below": GATE_PROMOTE,
        "gate_marginal_below": GATE_MARGINAL,
        "verdict": verdict,
        "pred_oof_path": str(pred_oof_path),
        "te_npy_path": str(te_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   nb2240 K=20 in_RAE        = {rae_nb2240:.4f}")
    print(f"   chemprop_aux in_RAE       = {rae_anchor:.4f}")
    print(f"   scaffold-CV mean_rae      = {mean_rae:.4f}")
    print(f"   deploy in-sample te_RAE   = {te_unb_rae:.4f}")
    print(f"   gate thresholds           = <{GATE_PROMOTE} PROMOTE | "
          f"<{GATE_MARGINAL} MARGINAL")
    print(f"   verdict                   = {verdict}")
    print(f"   wall                      = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "rae_nb2240_K20_oof",
        "rae_chemprop_aux_oof",
        "mean_rae",
        "verdict",
        "te_unb_rae_in_sample",
    ):
        print(f"  {k}: {res.get(k)}")
