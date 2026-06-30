"""nb2813 -- Scaffold-stratified bootstrap LGBM bagging on K=20 substrate.

NEW PARADIGM (bagging-axis):
    Standard row-bootstrap (nb2166 LGBM `subsample` with `bagging_freq=1`)
    samples rows uniformly within each fold's TRAIN partition.  At n=253,
    rare scaffolds (singletons / 1-2 member families) are over-represented
    in some bootstrap draws and entirely absent from others -- this drives
    high tree-to-tree variance on the long tail and lets common-scaffold
    rows dominate the bag mean.

    nb2813 reshapes the bootstrap distribution: cluster the 253 unblind
    rows into 10 scaffold strata via K-means on the Bemis-Murcko scaffold
    Morgan FPs, then draw bootstrap samples STRATIFIED by cluster (each
    bag preserves per-cluster row counts, sampling with replacement
    within each cluster).  Rare-scaffold strata are guaranteed
    representation in every bag; the bag-to-bag decorrelation budget is
    spent on intra-cluster row identity rather than on stratum presence.

    10 bag members; each bag = fresh stratified bootstrap + LGBM
    (max_depth=4, num_leaves=15, n_est=300, lr=0.03, mcs=5, reg_lambda=2.0)
    on chemprop_aux residual.  Bag aggregate = MEAN of the 10 LGBM
    predictions.  Scaffold 5-fold CV at kf_seed=1001 on the 253.

PROTOCOL:
    1. Load X_117 substrate (nb2240) -> slice K=20 surviving cols.
    2. anchor = chemprop_aux te[unb_idx]; residual = y_unb - anchor.
    3. Compute Bemis-Murcko scaffold per unblind row; build scaffold
       Morgan FP (radius=2, 2048 bits); K-means(n_clusters=10, seed=1001)
       on the unique scaffold FPs -> per-row cluster label.
    4. Scaffold 5-fold CV (kf_seed=1001).  Inside each train fold:
          - For each of 10 bag members:
              - Within each cluster ∩ train fold rows, sample with
                replacement the same number of rows present -> stratified
                bootstrap of size n_train.
              - Fit LGBM on (X_bs, residual_bs).
              - Predict held-out fold rows.
          - OOF prediction for held-out rows = MEAN across 10 bag preds.
    5. Aggregate OOF over the 5 folds -> single mean_rae on the 253.
    6. Deploy: refit 10-bag stratified-bootstrap LGBM on full 253 ->
       predict 513 te residual -> add to chemprop_aux te(513).
    7. Gate (mean RAE on the OOF):
          mean_rae < 0.4570 -> "PROMOTE"
          mean_rae < 0.4598 -> "MARGINAL_BEAT"
          else              -> "FAIL"

OUTPUTS:
    scripts/nb2813_stratified_scaffold_bootstrap.py
    data/processed/nb2813_summary.json
    data/processed/nb2813_pred_oof.npy   (253,) float32 CORRECTED pred
    data/processed/te_nb2813.npy         (513,) float32 deploy refit
    submissions/nb2813_stratified_scaffold_bootstrap.csv
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.cluster import KMeans

from pxr.chem import bemis_murcko, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2813"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"
NB2240_SUMMARY = DATA_PROCESSED / "nb2240_summary.json"

N_FOLDS = 5
KF_SEED = 1001
N_CLUSTERS = 10
N_BAGS = 10
FP_RADIUS = 2
FP_BITS = 2048

# Gate thresholds (mean RAE)
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

CHEMPROP_AUX_REF = 0.6216
NB2240_K20_REF = 0.4630  # scaffold_kfold_indices baseline at K=20


def _lgbm_params(seed: int) -> dict:
    """Same LGBM hp as nb2240 / nb2700 / nb2804."""
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


def _scaffold_keys_for_unb(unb_smiles: list[str]) -> list[str]:
    """Bemis-Murcko per row; empty scaffolds get unique per-row placeholders
    so they never group (mirrors scaffold_kfold_indices semantics)."""
    keys: list[str] = []
    for i, s in enumerate(unb_smiles):
        sc = bemis_murcko(s)
        if sc and isinstance(sc, str) and len(sc) > 0:
            keys.append(sc)
        else:
            keys.append(f"__singleton_{i}__")
    return keys


def _kmeans_cluster_scaffolds(
    scaffold_keys: list[str], n_clusters: int, seed: int,
) -> tuple[np.ndarray, dict]:
    """K-means(n_clusters) on Morgan FPs of *unique* scaffold SMILES.

    Empty / singleton placeholders ("__singleton_*") are pooled into a single
    "no-scaffold" bucket and assigned to one cluster (the least-populated
    cluster after the main K-means).  This guarantees every row gets a
    cluster id in [0, n_clusters).

    Returns
    -------
    per_row_cluster : (n_unb,) int32
        Cluster id for each unblind row.
    diag : dict
        Diagnostics: per-cluster counts, n_unique_scaffolds, etc.
    """
    n_unb = len(scaffold_keys)
    # Map each row to a "scaffold SMILES" or sentinel placeholder
    unique_scafs = sorted({s for s in scaffold_keys if not s.startswith("__singleton_")})
    has_singleton = any(s.startswith("__singleton_") for s in scaffold_keys)
    n_unique_scaf = len(unique_scafs)

    if n_unique_scaf < n_clusters:
        raise RuntimeError(
            f"Cannot form {n_clusters} scaffold clusters: only "
            f"{n_unique_scaf} unique non-singleton scaffolds in 253 unblind"
        )

    # Morgan FPs of unique scaffold SMILES (radius=2, 2048 bits)
    fps = morgan_fp_batch(unique_scafs, radius=FP_RADIUS, n_bits=FP_BITS)
    fps_f32 = fps.astype(np.float32)

    km = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=10,
        max_iter=300,
    )
    scaf_labels = km.fit_predict(fps_f32)
    scaf_to_cluster: dict[str, int] = {
        s: int(scaf_labels[i]) for i, s in enumerate(unique_scafs)
    }

    cluster_counts_nonsingleton = np.bincount(scaf_labels, minlength=n_clusters)

    # Singletons -> placed in the (current) least-populated cluster so that
    # the singleton bucket adds to a small cluster, not the largest one.
    if has_singleton:
        target_cluster_for_singletons = int(np.argmin(cluster_counts_nonsingleton))
    else:
        target_cluster_for_singletons = -1

    per_row_cluster = np.empty(n_unb, dtype=np.int32)
    for i, s in enumerate(scaffold_keys):
        if s.startswith("__singleton_"):
            per_row_cluster[i] = target_cluster_for_singletons
        else:
            per_row_cluster[i] = scaf_to_cluster[s]

    per_cluster_row_counts = np.bincount(per_row_cluster, minlength=n_clusters)
    diag = {
        "n_unique_scaffolds_total": int(n_unique_scaf
                                        + (1 if has_singleton else 0)),
        "n_unique_scaffolds_kmeans": int(n_unique_scaf),
        "n_singleton_rows": int(sum(
            1 for s in scaffold_keys if s.startswith("__singleton_")
        )),
        "singleton_target_cluster": int(target_cluster_for_singletons),
        "per_cluster_scaffold_counts": [int(c) for c in cluster_counts_nonsingleton],
        "per_cluster_row_counts": [int(c) for c in per_cluster_row_counts],
        "kmeans_inertia": float(km.inertia_),
    }
    return per_row_cluster, diag


def _stratified_bootstrap_indices(
    tr_loc: np.ndarray,
    tr_cluster: np.ndarray,
    n_clusters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a single stratified bootstrap sample of size len(tr_loc).

    Per cluster c in [0, n_clusters): the train rows whose cluster == c are
    sampled WITH REPLACEMENT to preserve their original count (so the bag's
    per-cluster proportion equals the train fold's per-cluster proportion).

    Returns an (n_train,) array of absolute indices into X_unb / residual
    (i.e. the same index space as tr_loc).  Empty clusters in the train
    partition contribute zero rows (which is correct -- they have no rows
    to bootstrap).
    """
    bs_idx_chunks = []
    for c in range(n_clusters):
        mask_c = tr_cluster == c
        rows_c = tr_loc[mask_c]
        n_c = len(rows_c)
        if n_c == 0:
            continue
        # WITH REPLACEMENT, same count -> preserves per-cluster proportion
        draws = rng.integers(low=0, high=n_c, size=n_c)
        bs_idx_chunks.append(rows_c[draws])
    return np.concatenate(bs_idx_chunks)


def _bagging_oof_one_fold(
    X_unb: np.ndarray,
    residual: np.ndarray,
    per_row_cluster: np.ndarray,
    tr_loc: np.ndarray,
    va_loc: np.ndarray,
    n_bags: int,
    n_clusters: int,
    fold_seed: int,
) -> np.ndarray:
    """Mean of n_bags LGBM predictions on va_loc; each bag fit on a
    stratified bootstrap of the train fold."""
    n_va = len(va_loc)
    bag_preds = np.zeros((n_bags, n_va), dtype=np.float64)
    tr_cluster = per_row_cluster[tr_loc]
    for b in range(n_bags):
        # Distinct RNG stream per (fold, bag) but deterministic from fold_seed
        bag_seed = int(fold_seed * 1000 + b)
        rng = np.random.default_rng(bag_seed)
        bs_idx = _stratified_bootstrap_indices(
            tr_loc, tr_cluster, n_clusters, rng,
        )
        mdl = lgb.LGBMRegressor(**_lgbm_params(bag_seed))
        mdl.fit(X_unb[bs_idx], residual[bs_idx])
        bag_preds[b] = mdl.predict(X_unb[va_loc])
    return bag_preds.mean(axis=0)


def _deploy_bagging_te(
    X_unb: np.ndarray,
    residual: np.ndarray,
    per_row_cluster: np.ndarray,
    X_te: np.ndarray,
    n_bags: int,
    n_clusters: int,
    deploy_seed: int,
) -> np.ndarray:
    """Refit 10-bag stratified-bootstrap LGBM on full 253, return (n_test,)
    mean residual prediction."""
    full_loc = np.arange(len(residual), dtype=np.int64)
    n_te = X_te.shape[0]
    bag_preds = np.zeros((n_bags, n_te), dtype=np.float64)
    for b in range(n_bags):
        bag_seed = int(deploy_seed * 1000 + b)
        rng = np.random.default_rng(bag_seed)
        bs_idx = _stratified_bootstrap_indices(
            full_loc, per_row_cluster, n_clusters, rng,
        )
        mdl = lgb.LGBMRegressor(**_lgbm_params(bag_seed))
        mdl.fit(X_unb[bs_idx], residual[bs_idx])
        bag_preds[b] = mdl.predict(X_te)
    return bag_preds.mean(axis=0).astype(np.float32)


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- scaffold-STRATIFIED bootstrap LGBM bagging on K=20")
    print(f"        anchor = {ANCHOR}  n_clusters={N_CLUSTERS}  "
          f"n_bags={N_BAGS}")
    print(f"        n_folds={N_FOLDS}  kf_seed={KF_SEED}  "
          f"FP=Morgan r{FP_RADIUS} {FP_BITS}b")
    print(f"        ref nb2240 K=20 scaffold_kfold = {NB2240_K20_REF:.4f}")
    print(f"        GATE: <{GATE_PROMOTE} PROMOTE / "
          f"<{GATE_MARGINAL} MARGINAL_BEAT / else FAIL")
    print("=" * 78)

    # ---- Truth + anchor ----
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

    # ---- Load X_117 + slice K=20 ----
    if not X117_UNB_PATH.exists() or not X117_TE_PATH.exists():
        raise FileNotFoundError(
            f"X_117 cache missing: {X117_UNB_PATH} or {X117_TE_PATH}"
        )
    X117_unb = np.load(X117_UNB_PATH).astype(np.float32)
    X117_te = np.load(X117_TE_PATH).astype(np.float32)
    if X117_unb.shape != (n_unb, 117):
        raise ValueError(f"X117_unb shape {X117_unb.shape}")
    if X117_te.shape != (n_test, 117):
        raise ValueError(f"X117_te shape {X117_te.shape}")
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
    print(f"[feat] X_unb (K=20) = {X_unb.shape}  X_te = {X_te.shape}")

    # ---- Scaffold keys + K-means clustering on unique scaffold FPs ----
    unb_smiles = [test_smiles[i] for i in unb_idx]
    raw_scaffold_keys = _scaffold_keys_for_unb(unb_smiles)
    n_unique_scaf = len(set(raw_scaffold_keys))
    n_singletons = sum(1 for k in raw_scaffold_keys if k.startswith("__singleton_"))
    print(f"[scaffold] unique unb scaffolds = {n_unique_scaf}  "
          f"singletons = {n_singletons}")

    per_row_cluster, km_diag = _kmeans_cluster_scaffolds(
        raw_scaffold_keys, n_clusters=N_CLUSTERS, seed=KF_SEED,
    )
    print(f"[kmeans] inertia = {km_diag['kmeans_inertia']:.3f}")
    print(f"[kmeans] singleton bucket -> cluster "
          f"{km_diag['singleton_target_cluster']}")
    print(f"[kmeans] per-cluster row counts = "
          f"{km_diag['per_cluster_row_counts']}")
    if any(c == 0 for c in km_diag['per_cluster_row_counts']):
        raise RuntimeError("empty cluster after K-means; lower N_CLUSTERS")

    # ---- Scaffold 5-fold CV with stratified-bootstrap bag inside each fold ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV  kf_seed={KF_SEED}  "
          f"n_bags={N_BAGS} stratified-bootstrap per fold")
    print("-" * 78)
    splits = scaffold_kfold_indices(
        raw_scaffold_keys, n_splits=N_FOLDS, shuffle=True, seed=KF_SEED,
    )
    resid_oof = np.full(n_unb, np.nan, dtype=np.float64)
    per_fold_diags = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        ts = time.time()
        # Determinism: bind each fold's bag stream to a unique fold_seed
        fold_seed = int(KF_SEED * 100 + fold_i)
        oof_va = _bagging_oof_one_fold(
            X_unb=X_unb,
            residual=residual,
            per_row_cluster=per_row_cluster,
            tr_loc=tr_loc,
            va_loc=va_loc,
            n_bags=N_BAGS,
            n_clusters=N_CLUSTERS,
            fold_seed=fold_seed,
        )
        resid_oof[va_loc] = oof_va
        rae_fold_corr = float(rae(y_unb[va_loc], anchor[va_loc] + oof_va))
        tr_cluster_counts = np.bincount(
            per_row_cluster[tr_loc], minlength=N_CLUSTERS,
        ).tolist()
        va_cluster_counts = np.bincount(
            per_row_cluster[va_loc], minlength=N_CLUSTERS,
        ).tolist()
        per_fold_diags.append({
            "fold": fold_i,
            "fold_seed": fold_seed,
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "rae_fold_corrected": rae_fold_corr,
            "train_cluster_counts": [int(c) for c in tr_cluster_counts],
            "val_cluster_counts":   [int(c) for c in va_cluster_counts],
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   fold {fold_i}: n_tr={len(tr_loc):3d}  n_va={len(va_loc):3d}  "
              f"rae_fold_corr={rae_fold_corr:.4f}  "
              f"wall={time.time() - ts:.1f}s")

    if np.isnan(resid_oof).any():
        raise RuntimeError("CV did not cover all rows (NaN OOF)")

    pred_corr_oof = anchor + resid_oof
    mean_rae = float(rae(y_unb, pred_corr_oof))
    per_fold_rae = [d["rae_fold_corrected"] for d in per_fold_diags]
    print(f"\n[cv] global mean_rae        = {mean_rae:.4f}")
    print(f"[cv] per_fold_rae           = "
          f"[{', '.join(f'{r:.4f}' for r in per_fold_rae)}]")
    print(f"[cv] per_fold mean/std      = "
          f"{np.mean(per_fold_rae):.4f} / {np.std(per_fold_rae):.4f}")
    print(f"[cv] vs anchor              = "
          f"{mean_rae - rae_anchor:+.4f}  (anchor {rae_anchor:.4f})")
    print(f"[cv] vs nb2240 K=20         = "
          f"{mean_rae - NB2240_K20_REF:+.4f}  (ref {NB2240_K20_REF:.4f})")

    # ---- Deploy ----
    print("\n" + "-" * 78)
    print("DEPLOY (refit 10-bag stratified-bootstrap on full 253 -> 513 te)")
    print("-" * 78)
    ts = time.time()
    te_resid_513 = _deploy_bagging_te(
        X_unb=X_unb,
        residual=residual,
        per_row_cluster=per_row_cluster,
        X_te=X_te,
        n_bags=N_BAGS,
        n_clusters=N_CLUSTERS,
        deploy_seed=KF_SEED,
    )
    te_deploy = (te_anchor_513 + te_resid_513.astype(np.float64)).astype(np.float32)
    te_unb_in_sample_rae = float(rae(y_unb, te_deploy[unb_idx]))
    print(f"[deploy] te(513) mean/std = "
          f"{te_deploy.mean():.3f}/{te_deploy.std():.3f}")
    print(f"[deploy] te[unb_idx] in-sample RAE = {te_unb_in_sample_rae:.4f}  "
          f"(in-sample optimism expected)")
    print(f"[deploy] wall = {time.time() - ts:.1f}s")

    # ---- Save artefacts ----
    pred_oof_corrected = pred_corr_oof.astype(np.float32)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof_corrected)
    np.save(te_path, te_deploy)
    print(f"\n[save] {oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_stratified_scaffold_bootstrap.csv"
    pd.DataFrame({
        "SMILES": test_smiles,
        "Molecule Name": te_names,
        "pEC50": te_deploy.astype(np.float32),
    }).to_csv(sub_csv, index=False)
    print(f"[save] {sub_csv}")

    # ---- Gate ----
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "=" * 78)
    print("GATE EVALUATION")
    print("=" * 78)
    print(f"   mean_rae            = {mean_rae:.4f}")
    print(f"   < {GATE_PROMOTE:.4f}  (PROMOTE)       = {mean_rae < GATE_PROMOTE}")
    print(f"   < {GATE_MARGINAL:.4f}  (MARGINAL_BEAT) = "
          f"{mean_rae < GATE_MARGINAL}")
    print(f"   VERDICT             = {verdict}")

    summary = {
        "tag": TAG,
        "method": "scaffold_stratified_bootstrap_LGBM_bagging_K20_chemprop_aux_residual",
        "rationale": (
            "K-means(n_clusters=10) on Bemis-Murcko scaffold Morgan FPs "
            "of the 253 unblind, then 10 LGBM bags per CV fold where each "
            "bag draws a bootstrap sample stratified by scaffold cluster "
            "(per-cluster proportion preserved, with-replacement WITHIN "
            "each cluster).  Mean of 10 predictions per held-out row; "
            "scaffold 5-fold CV at kf_seed=1001; chemprop_aux residual."
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
        "n_unique_scaffolds": int(n_unique_scaf),
        "n_singleton_scaffolds": int(n_singletons),
        "n_folds": N_FOLDS,
        "kf_seed": KF_SEED,
        "n_clusters": int(N_CLUSTERS),
        "n_bags": int(N_BAGS),
        "fp_radius": int(FP_RADIUS),
        "fp_bits": int(FP_BITS),
        "kmeans_diagnostics": km_diag,
        "cv_protocol": "scaffold_kfold_indices(n_splits=5, kf_seed=1001)",
        "bagging_protocol": (
            "stratified_bootstrap_with_replacement_per_kmeans_cluster_"
            "preserving_per_cluster_row_counts"
        ),
        "feat_dim": int(X_unb.shape[1]),
        "model_class": "lightgbm.LGBMRegressor",
        "lgbm_params_sample": _lgbm_params(KF_SEED),
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "mean_rae": mean_rae,
        "per_fold_rae": [float(r) for r in per_fold_rae],
        "per_fold_rae_mean": float(np.mean(per_fold_rae)),
        "per_fold_rae_std": float(np.std(per_fold_rae)),
        "per_fold_diagnostics": per_fold_diags,
        "delta_vs_anchor": mean_rae - rae_anchor,
        "delta_vs_nb2240_K20_scaffold_kfold": mean_rae - NB2240_K20_REF,
        "nb2240_K20_scaffold_kfold_ref": NB2240_K20_REF,
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
        "mean_rae",
        "per_fold_rae",
        "per_fold_rae_mean",
        "per_fold_rae_std",
        "delta_vs_anchor",
        "delta_vs_nb2240_K20_scaffold_kfold",
        "te_unb_in_sample_rae",
        "n_unique_scaffolds",
        "n_singleton_scaffolds",
        "n_clusters",
        "n_bags",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
    print("\n==== KMEANS CLUSTER ROW COUNTS ====")
    print(f"  {res['kmeans_diagnostics']['per_cluster_row_counts']}")
