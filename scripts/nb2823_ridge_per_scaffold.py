"""nb2823 -- Ridge per scaffold cluster fallback on chemprop_aux residual.

NEW PARADIGM (vs nb2793 DT per cluster):
    Same K=10 K-means cluster substrate, same K=20 features, same routing
    via Tanimoto-to-thresholded-centroid, same chemprop_aux residual target.
    SWAP the per-cluster model class from shallow DecisionTreeRegressor to
    Ridge(alpha=10.0).

    Hypothesis: at per-cluster n ~ 25, axis-aligned trees overfit on a few
    samples even at depth=3 (8 leaves max, ~3 rows per leaf). Ridge with
    a moderate L2 penalty (alpha=10.0) handles small-n regression more
    gracefully -- it's a single linear hyperplane regularized away from
    zero, so it cannot carve out noise pockets the way a tree can.

    This is a SUBSTRATE-change-within-substrate move: we already accepted
    nb2793's per-cluster routing substrate (K=10 K-means on Bemis-Murcko
    scaffold FPs); we now flip the per-cluster predictor from non-linear
    axis-aligned (DT) to linear regularized (Ridge). This tests whether
    the per-cluster routing is the load-bearing piece, or whether the
    inductive bias of the per-cluster predictor matters.

PROTOCOL:
    1. Compute Bemis-Murcko scaffold for each unb / test SMILES; fall back
       to whole-molecule Morgan FP when the scaffold is empty (acyclic).
    2. Morgan FP (radius=2, 1024 bits) for each scaffold -> stack to
       (253, 1024) for unb and (513, 1024) for test.
    3. K-means(K=10) on the unb scaffold-FP matrix with kf_seed=1001.
    4. Per scaffold-CV fold (5-fold, kf_seeds 1001..1005):
       a. Recompute K-means(K=10, seed=kf_seed) on train-fold scaffold-FPs.
       b. For each cluster c: fit Ridge(alpha=10.0) on chemprop_aux residual
          using K=20 features for compounds in c.
       c. For each val row: route via highest-Tanimoto centroid, predict
          residual using that cluster's Ridge.
    5. Mean-bag across 5 kf_seeds at OOF (residual) level.
    6. Deploy: refit K-means(K=10) + per-cluster Ridges on all 253; predict
       513 test rows by routing.

GATE (mean-bag corrected RAE):
    mean_rae < 0.4570 -> "PROMOTE"
    mean_rae < 0.4598 -> "MARGINAL_BEAT"
    else              -> "FAIL"

OUTPUTS:
    scripts/nb2823_ridge_per_scaffold.py
    data/processed/nb2823_summary.json
    data/processed/nb2823_pred_oof.npy   (253,) float32 mean-bag CORRECTED
    data/processed/te_nb2823.npy         (513,) float32 deploy refit
    submissions/nb2823_ridge_per_scaffold.csv
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
from sklearn.linear_model import Ridge

from pxr.chem import bemis_murcko, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2823"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"
NB2240_SUMMARY = DATA_PROCESSED / "nb2240_summary.json"

N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# Per-task spec
N_CLUSTERS = 10
SCAF_FP_RADIUS = 2
SCAF_FP_BITS = 1024
RIDGE_ALPHA = 10.0
# Minimum rows to fit Ridge in a cluster; below this fall back to mean
RIDGE_MIN_SAMPLES = 5

# Gate thresholds
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

CHEMPROP_AUX_REF = 0.6216
NB2240_K20_REF = 0.4630  # K=20 LGBM baseline on same anchor + substrate
NB2793_DT_REF = None  # filled in below if nb2793 summary present


def _scaffold_fp_for_smiles(smiles_list: list[str]) -> np.ndarray:
    """Bemis-Murcko scaffold Morgan FP (radius=2, n_bits=1024). When the
    Bemis-Murcko scaffold is empty (acyclic compounds), fall back to the
    whole-molecule Morgan FP so every row gets a routable FP."""
    scaffolds = [bemis_murcko(s) for s in smiles_list]
    scaf_fp = morgan_fp_batch(scaffolds, radius=SCAF_FP_RADIUS, n_bits=SCAF_FP_BITS)
    row_sums = scaf_fp.sum(axis=1)
    empty_rows = np.where(row_sums == 0)[0]
    if len(empty_rows) > 0:
        fallback_smis = [smiles_list[i] for i in empty_rows]
        fallback_fp = morgan_fp_batch(
            fallback_smis, radius=SCAF_FP_RADIUS, n_bits=SCAF_FP_BITS,
        )
        scaf_fp[empty_rows] = fallback_fp
    return scaf_fp.astype(np.uint8)


def _tanimoto_to_centroids(
    fp_query: np.ndarray, centroids_bool: np.ndarray
) -> np.ndarray:
    """Tanimoto similarity of one (n_bits,) query FP to (K, n_bits) centroids."""
    q = fp_query.astype(bool)
    sims = np.zeros(centroids_bool.shape[0], dtype=np.float64)
    for k in range(centroids_bool.shape[0]):
        c = centroids_bool[k]
        inter = np.bitwise_and(q, c).sum()
        union = np.bitwise_or(q, c).sum()
        sims[k] = inter / union if union else 0.0
    return sims


def _route_rows(
    fp_matrix: np.ndarray, centroids_bool: np.ndarray
) -> np.ndarray:
    """Route each row to its nearest cluster by Tanimoto similarity."""
    n = fp_matrix.shape[0]
    routes = np.zeros(n, dtype=np.int32)
    for i in range(n):
        sims = _tanimoto_to_centroids(fp_matrix[i], centroids_bool)
        routes[i] = int(np.argmax(sims))
    return routes


def _kmeans_and_centroids_bool(
    fp_matrix: np.ndarray, n_clusters: int, seed: int
) -> tuple[KMeans, np.ndarray, np.ndarray]:
    """Fit K-means; return (model, train_labels, boolean centroids)."""
    km = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=10,
        max_iter=300,
    )
    train_labels = km.fit_predict(fp_matrix.astype(np.float32))
    centroids_bool = (km.cluster_centers_ >= 0.5).astype(np.uint8)
    return km, train_labels.astype(np.int32), centroids_bool


def _fit_per_cluster_ridge(
    X_train: np.ndarray,
    resid_train: np.ndarray,
    train_labels: np.ndarray,
    n_clusters: int,
    seed: int,
) -> dict[int, object]:
    """Fit Ridge(alpha=RIDGE_ALPHA) per cluster on chemprop_aux residual.

    For clusters with too few samples (< RIDGE_MIN_SAMPLES), fall back to
    the cluster mean residual (or 0 if truly empty)."""
    cluster_models: dict[int, object] = {}
    for c in range(n_clusters):
        mask = train_labels == c
        n_c = int(mask.sum())
        if n_c < RIDGE_MIN_SAMPLES:
            const = float(resid_train[mask].mean()) if n_c > 0 else 0.0
            cluster_models[c] = ("const", const)
            continue
        ridge = Ridge(
            alpha=RIDGE_ALPHA,
            fit_intercept=True,
            random_state=seed,
        )
        ridge.fit(X_train[mask], resid_train[mask])
        cluster_models[c] = ("ridge", ridge)
    return cluster_models


def _predict_routed(
    X_query: np.ndarray, routes: np.ndarray, cluster_models: dict
) -> np.ndarray:
    """Predict residual for each query row using its routed cluster model."""
    n = X_query.shape[0]
    pred = np.zeros(n, dtype=np.float64)
    for i in range(n):
        c = int(routes[i])
        kind, mdl = cluster_models[c]
        if kind == "ridge":
            pred[i] = float(mdl.predict(X_query[i:i + 1])[0])
        else:  # const fallback
            pred[i] = float(mdl)
    return pred


def _scaffold_cv_one_seed(
    X: np.ndarray,
    residual: np.ndarray,
    fp_scaf: np.ndarray,
    unb_scaffolds: list,
    kf_seed: int,
) -> tuple[np.ndarray, list[dict]]:
    """One scaffold-CV pass: per-fold K-means + per-cluster Ridge on residual."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n = len(residual)
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_diags = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        km, tr_labels, centroids_bool = _kmeans_and_centroids_bool(
            fp_scaf[tr_loc], N_CLUSTERS, kf_seed,
        )
        cluster_models = _fit_per_cluster_ridge(
            X[tr_loc], residual[tr_loc], tr_labels, N_CLUSTERS, kf_seed,
        )
        va_routes = _route_rows(fp_scaf[va_loc], centroids_bool)
        oof[va_loc] = _predict_routed(X[va_loc], va_routes, cluster_models)

        per_cluster_sizes = [int((tr_labels == c).sum()) for c in range(N_CLUSTERS)]
        per_cluster_fit_kinds = [cluster_models[c][0] for c in range(N_CLUSTERS)]
        fold_diags.append({
            "fold": fold_i,
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "cluster_sizes_tr": per_cluster_sizes,
            "cluster_fit_kinds": per_cluster_fit_kinds,
            "n_const_fallback": int(sum(1 for k in per_cluster_fit_kinds if k == "const")),
        })
    if np.isnan(oof).any():
        raise RuntimeError("scaffold split did not cover all rows")
    return oof, fold_diags


def _deploy_te_one_seed(
    X_unb: np.ndarray,
    residual: np.ndarray,
    fp_scaf_unb: np.ndarray,
    X_te: np.ndarray,
    fp_scaf_te: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Fit K-means + per-cluster Ridges on all 253; predict 513 routed residuals."""
    km, tr_labels, centroids_bool = _kmeans_and_centroids_bool(
        fp_scaf_unb, N_CLUSTERS, seed,
    )
    cluster_models = _fit_per_cluster_ridge(
        X_unb, residual, tr_labels, N_CLUSTERS, seed,
    )
    te_routes = _route_rows(fp_scaf_te, centroids_bool)
    pred = _predict_routed(X_te, te_routes, cluster_models).astype(np.float32)
    diag = {
        "seed": int(seed),
        "cluster_sizes_unb": [int((tr_labels == c).sum()) for c in range(N_CLUSTERS)],
        "te_route_hist": [int((te_routes == c).sum()) for c in range(N_CLUSTERS)],
        "n_const_fallback_clusters": int(sum(
            1 for c in range(N_CLUSTERS) if cluster_models[c][0] == "const"
        )),
    }
    return pred, diag


def main() -> dict:
    import pandas as pd
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Ridge per scaffold-cluster (K={N_CLUSTERS}) on "
          f"{ANCHOR} residual")
    print(f"        K-means on Bemis-Murcko scaffold Morgan FP "
          f"(radius={SCAF_FP_RADIUS}, n_bits={SCAF_FP_BITS})")
    print(f"        Ridge(alpha={RIDGE_ALPHA}, "
          f"min_samples={RIDGE_MIN_SAMPLES}) per cluster")
    print(f"        anchor = {ANCHOR}  scaffold-CV {N_FOLDS}-fold  "
          f"kf_seeds={KF_SEEDS}")
    print(f"        ref nb2240 K=20 LGBM = {NB2240_K20_REF:.4f}")
    print(f"        GATE: <{GATE_PROMOTE} PROMOTE / "
          f"<{GATE_MARGINAL} MARGINAL_BEAT / else FAIL")
    print("=" * 78)

    # ---- Load truth + anchor ----
    te = load_test()
    n_test = len(te)
    test_smiles = (
        te["smiles"].astype(str).tolist()
        if "smiles" in te.columns
        else te["SMILES"].astype(str).tolist()
    )
    te_names = (
        te["name"].values
        if "name" in te.columns
        else te["Molecule Name"].values
    )

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"anchor te missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(f"anchor te shape {te_anchor_513.shape}")
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Load X_117 substrate + slice K=20 ----
    if not X117_UNB_PATH.exists() or not X117_TE_PATH.exists():
        raise FileNotFoundError(
            f"X_117 cache missing: {X117_UNB_PATH} or {X117_TE_PATH}"
        )
    X117_unb = np.load(X117_UNB_PATH).astype(np.float32)
    X117_te = np.load(X117_TE_PATH).astype(np.float32)
    X117_unb = np.where(np.isfinite(X117_unb), X117_unb, 0.0).astype(np.float32)
    X117_te = np.where(np.isfinite(X117_te), X117_te, 0.0).astype(np.float32)

    if not NB2240_SUMMARY.exists():
        raise FileNotFoundError(f"nb2240 summary missing: {NB2240_SUMMARY}")
    with open(NB2240_SUMMARY) as f:
        nb2240 = json.load(f)
    k20_idx = list(nb2240["k20_surviving_idx_in_117"])
    k20_names = list(nb2240["k20_surviving_names"])
    assert len(k20_idx) == 20, f"expected 20 K=20 indices, got {len(k20_idx)}"
    X_unb = X117_unb[:, k20_idx].astype(np.float32)
    X_te = X117_te[:, k20_idx].astype(np.float32)
    feat_dim = X_unb.shape[1]
    print(f"[feat] X_unb (K=20) = {X_unb.shape}  X_te = {X_te.shape}")

    # Optional cross-ref to nb2793 DT result
    nb2793_ref = None
    nb2793_summary_path = DATA_PROCESSED / "nb2793_summary.json"
    if nb2793_summary_path.exists():
        try:
            with open(nb2793_summary_path) as f:
                nb2793_ref = float(json.load(f).get("mean_bag_rae"))
            print(f"[ref] nb2793 DT per-cluster mean_bag = {nb2793_ref:.4f}")
        except Exception:
            nb2793_ref = None

    # ---- Scaffold FPs (clustering substrate) ----
    unb_smiles = [test_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique unb scaffolds = {n_unique_scaf}")
    fp_scaf_unb = _scaffold_fp_for_smiles(unb_smiles)
    fp_scaf_te = _scaffold_fp_for_smiles(test_smiles)
    print(f"[scaf_fp] unb = {fp_scaf_unb.shape}  te = {fp_scaf_te.shape}  "
          f"(avg bits unb={fp_scaf_unb.sum(axis=1).mean():.1f})")

    # ---- 5-seed scaffold CV ----
    print("\n" + "-" * 78)
    print(f"5-SEED SCAFFOLD-CV  seeds={KF_SEEDS}  folds={N_FOLDS}  "
          f"K_cluster={N_CLUSTERS}  dim={feat_dim}")
    print("-" * 78)
    per_seed_oof_resid = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)
    per_seed_te_resid = np.zeros((len(KF_SEEDS), n_test), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_fold_diags: list[list[dict]] = []
    per_seed_deploy_diag: list[dict] = []

    for i, seed in enumerate(KF_SEEDS):
        ts = time.time()
        resid_oof, fold_diags = _scaffold_cv_one_seed(
            X_unb, residual, fp_scaf_unb, unb_scaffolds, seed,
        )
        per_seed_oof_resid[i] = resid_oof
        per_seed_fold_diags.append(fold_diags)
        te_resid, deploy_diag = _deploy_te_one_seed(
            X_unb, residual, fp_scaf_unb, X_te, fp_scaf_te, seed,
        )
        per_seed_te_resid[i] = te_resid
        per_seed_deploy_diag.append(deploy_diag)
        pred_corr = anchor + resid_oof
        rae_s = float(rae(y_unb, pred_corr))
        per_seed_rae.append(rae_s)
        n_const = sum(d["n_const_fallback"] for d in fold_diags)
        print(f"   seed={seed}  rae_corr={rae_s:.4f}  "
              f"d_vs_anchor={rae_s - rae_anchor:+.4f}  "
              f"n_const_folds={n_const}/{N_FOLDS * N_CLUSTERS}  "
              f"wall={time.time() - ts:.1f}s")

    per_seed_mean = float(np.mean(per_seed_rae))
    per_seed_std = float(np.std(per_seed_rae))
    mean_bag_resid = per_seed_oof_resid.mean(axis=0)
    median_bag_resid = np.median(per_seed_oof_resid, axis=0)
    rae_mean_bag = float(rae(y_unb, anchor + mean_bag_resid))
    rae_median_bag = float(rae(y_unb, anchor + median_bag_resid))

    print("\n[cv] per_seed_mean RAE = "
          f"{per_seed_mean:.4f}  std={per_seed_std:.4f}")
    print(f"[cv] mean_bag   RAE = {rae_mean_bag:.4f}")
    print(f"[cv] median_bag RAE = {rae_median_bag:.4f}")
    print(f"[cv] anchor     RAE = {rae_anchor:.4f}  "
          f"(d_mean_bag = {rae_mean_bag - rae_anchor:+.4f})")
    print(f"[cv] reference  nb2240 K=20 LGBM = {NB2240_K20_REF:.4f}  "
          f"(d = {rae_mean_bag - NB2240_K20_REF:+.4f})")
    if nb2793_ref is not None:
        print(f"[cv] reference  nb2793 DT per-cluster = {nb2793_ref:.4f}  "
              f"(d = {rae_mean_bag - nb2793_ref:+.4f})")

    # ---- Deploy te (mean-bag corrected) ----
    mean_bag_te_resid = per_seed_te_resid.mean(axis=0)
    te_deploy = (te_anchor_513 + mean_bag_te_resid).astype(np.float32)
    te_unb_in_sample_rae = float(rae(y_unb, te_deploy[unb_idx]))
    print(f"\n[deploy] te(513) mean/std = "
          f"{te_deploy.mean():.3f}/{te_deploy.std():.3f}")
    print(f"[deploy] te[unb_idx] in-sample RAE = {te_unb_in_sample_rae:.4f}  "
          f"(deploy refit, in-sample optimism expected)")

    # ---- Save artefacts ----
    pred_oof_corrected = (anchor + mean_bag_resid).astype(np.float32)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof_corrected)
    np.save(te_path, te_deploy)
    print(f"[save] {oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_ridge_per_scaffold.csv"
    pd.DataFrame({
        "SMILES": test_smiles,
        "Molecule Name": te_names,
        "pEC50": te_deploy.astype(np.float32),
    }).to_csv(sub_csv, index=False)
    print(f"[save] {sub_csv}")

    # ---- Gate ----
    if rae_mean_bag < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif rae_mean_bag < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "=" * 78)
    print("GATE EVALUATION")
    print("=" * 78)
    print(f"   mean_bag_rae        = {rae_mean_bag:.4f}")
    print(f"   < {GATE_PROMOTE:.4f}  (PROMOTE)       = "
          f"{rae_mean_bag < GATE_PROMOTE}")
    print(f"   < {GATE_MARGINAL:.4f}  (MARGINAL_BEAT) = "
          f"{rae_mean_bag < GATE_MARGINAL}")
    print(f"   VERDICT             = {verdict}")

    summary = {
        "tag": TAG,
        "method": "Ridge_per_scaffold_cluster_K10_chemprop_aux_residual",
        "rationale": (
            "K-means(K=10) on Bemis-Murcko-scaffold Morgan FP partitions "
            "the 253 unblind; Ridge(alpha=10.0) per cluster on K=20 "
            "features regresses chemprop_aux residual; test rows routed "
            "by Tanimoto-to-centroid. Swap vs nb2793 is model class only "
            "(DT depth=3 -> Ridge alpha=10), keeping the per-cluster "
            "routing substrate intact."
        ),
        "anchor": ANCHOR,
        "anchor_te_path": str(ANCHOR_TE_PATH),
        "anchor_pre_unblind": True,
        "rae_anchor_chemprop_aux": rae_anchor,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "x117_unb_path": str(X117_UNB_PATH),
        "x117_te_path": str(X117_TE_PATH),
        "k20_idx_source": str(NB2240_SUMMARY),
        "k20_surviving_idx_in_117": [int(j) for j in k20_idx],
        "k20_surviving_names": k20_names,
        "n_unb": n_unb,
        "n_test": n_test,
        "n_unique_scaffolds": n_unique_scaf,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "feat_dim": int(feat_dim),
        "n_clusters": N_CLUSTERS,
        "scaffold_fp_radius": SCAF_FP_RADIUS,
        "scaffold_fp_bits": SCAF_FP_BITS,
        "scaffold_fp_fallback": "whole_molecule_morgan_when_acyclic",
        "model_class": "sklearn.linear_model.Ridge",
        "ridge_alpha": RIDGE_ALPHA,
        "ridge_min_samples": RIDGE_MIN_SAMPLES,
        "ridge_fit_intercept": True,
        "kmeans_n_init": 10,
        "kmeans_max_iter": 300,
        "centroid_threshold_for_tanimoto": 0.5,
        "routing_metric": "tanimoto_to_thresholded_centroid",
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_seed_rae": [float(r) for r in per_seed_rae],
        "per_seed_mean_rae": per_seed_mean,
        "per_seed_std_rae": per_seed_std,
        "mean_bag_rae": rae_mean_bag,
        "median_bag_rae": rae_median_bag,
        "mean_rae": rae_mean_bag,  # alias for gate consumers
        "delta_vs_anchor": rae_mean_bag - rae_anchor,
        "delta_vs_nb2240_K20_lgbm": rae_mean_bag - NB2240_K20_REF,
        "nb2240_K20_lgbm_ref": NB2240_K20_REF,
        "nb2793_dt_per_cluster_ref": nb2793_ref,
        "delta_vs_nb2793_dt": (
            rae_mean_bag - nb2793_ref if nb2793_ref is not None else None
        ),
        "per_seed_fold_diagnostics": per_seed_fold_diags,
        "per_seed_deploy_diagnostics": per_seed_deploy_diag,
        "te_unb_in_sample_rae": te_unb_in_sample_rae,
        "te_deploy_mean": float(te_deploy.mean()),
        "te_deploy_std": float(te_deploy.std()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv),
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "verdict": verdict,
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
        "per_seed_rae",
        "per_seed_mean_rae",
        "per_seed_std_rae",
        "mean_bag_rae",
        "median_bag_rae",
        "delta_vs_anchor",
        "delta_vs_nb2240_K20_lgbm",
        "delta_vs_nb2793_dt",
        "te_unb_in_sample_rae",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
