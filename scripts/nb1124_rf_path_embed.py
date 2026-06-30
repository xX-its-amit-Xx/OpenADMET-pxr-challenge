"""nb1124 -- Random Forest path embeddings: leaf-id one-hot -> Ridge on chemprop_aux residual.

HYPOTHESIS:
    nb2103 K=28 (SHAP-pruned 5-way K-tuned matrix + LGBM(MSE), 5-seed bag,
    5-fold cross-fit on 253) achieves mean-bag RAE 0.4737 / median-bag RAE
    0.4698 -- the current PRE-unblind champion on the chemprop_aux residual.

    nb1124 asks: is there orthogonal structure in the same chemprop_aux
    residual recoverable by a fundamentally different model family -- a
    Random Forest "path embedding" (tree.apply -> one-hot leaf-id) fed
    into RidgeCV?  The RF path embedding is a non-additive piecewise-constant
    basis: each tree partitions feature-space into ~50 leaves, and the
    one-hot stack across 200 trees gives a (N, ~10k) sparse representation
    on which a LINEAR model (Ridge) can fit interactions that LGBM(MSE)
    handles as additive boosting.  If this beats nb2103 by >=0.003 it
    confirms a real model-family gap; if it ties (|delta|<0.003) the SHAP-
    pruned LGBM has already extracted all available signal at this honest
    cross-fit level.

PROTOCOL:
    1. Load 4139-train + 513-test 2265-dim Morgan+RDKit features
       (cache_combined_features.npz) and 4139 chemprop_aux OOF / 513
       chemprop_aux deploy te.
    2. Train RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=2)
       on 4139 train with target = train_pec50 - oof_chemprop_aux (signed
       residual).  Also fit RF directly on pec50 as a sanity variant.
    3. Extract leaf-id matrices: rf.apply(X_train) -> (4139, 200),
       rf.apply(X_test) -> (513, 200).  Slice X_unb = X_test[unb_idx]
       -> (253, 200) leaf-ids.
    4. OneHotEncoder(handle_unknown='ignore', sparse_output=True) fit on
       leaf-ids of the union (train + test) -> sparse (N, ~10-50k) leaf
       embedding.  Project X_unb_oh = (253, n_leaves).
    5. Honest 5-fold cross-fit RidgeCV(alphas=logspace(-3,3,7)) on the
       253 unblind: residual_oof[va] = ridge.fit(X_unb_oh[tr],
       residual_unb[tr]).predict(X_unb_oh[va]); repeat across 5 seeds
       {0, 1, 7, 42, 137}.
    6. pred_corrected_oof_s = chemprop_aux_unb + residual_oof_s; pooled
       RAE per seed; mean-bag and median-bag across seeds.
    7. SLSQP blend with nb2103 K=28 mean-bag OOF (4-way bag of 5 seeds);
       verify whether nb1124 contributes non-zero weight.
    8. Decision: beats nb2103 (0.4737 mean / 0.4698 median) at margin
       0.003 -> WINNER; flat -> NEUTRAL; worse -> RF path embedding adds
       no signal beyond SHAP-pruned LGBM.
    9. If beats: also build a deploy CSV refit on all 253 unblind.

Outputs:
    scripts/nb1124_rf_path_embed.py
    data/processed/nb1124_summary.json
    data/processed/nb1124_per_seed_corrected_oof.npy   (5, 253) float32
    data/processed/nb1124_mean_bag_oof.npy             (253,)   float32
    data/processed/nb1124_median_bag_oof.npy           (253,)   float32
    (deploy artefacts only if WINNER:
       data/processed/te_nb1124.npy
       submissions/nb1124_rf_path_embed.csv)
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
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from pxr.data import load_test, load_train
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1124"
ANCHOR = "chemprop_aux"

# References (PRE-unblind clean track)
CHEMPROP_AUX_REF = 0.6216
NB2103_K28_MEAN_BAG_REF = 0.4737
NB2103_K28_MEDIAN_BAG_REF = 0.4698
DECISION_MARGIN = 0.003

# RF hyperparams (per spec)
RF_N_EST = 200
RF_MAX_DEPTH = 12
RF_SEED_TRAIN = 42

# Ridge hyperparams
RIDGE_ALPHAS = np.logspace(-3.0, 3.0, 7)

# Cross-fit seeds (match nb2103 protocol)
SEEDS = [0, 1, 7, 42, 137]
N_FOLDS = 5

# Paths
CACHE_FEAT = DATA_PROCESSED / "cache_combined_features.npz"
OOF_ANCHOR = DATA_PROCESSED / "oof_chemprop_aux.npy"
TE_ANCHOR = DATA_PROCESSED / "te_chemprop_aux.npy"
UNB_IDX_PATH = DATA_PROCESSED / "_audit_unblind_idx.npy"
Y_UNB_PATH = DATA_PROCESSED / "_audit_unblind_y.npy"
NB2103_K28_OOF_PATH = DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy"

SUBMISSIONS_DIR = Path(__file__).resolve().parents[1] / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)


def _cross_fit_ridge_one_seed(
    X_oh, residual: np.ndarray, seed: int
) -> tuple[np.ndarray, list[float], list[float]]:
    """Honest 5-fold cross-fit RidgeCV on sparse leaf-embedding.

    Returns OOF predictions (n_unb,) and per-fold alpha selections.
    """
    n = len(residual)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_alphas: list[float] = []
    fold_train_rae: list[float] = []
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = RidgeCV(alphas=RIDGE_ALPHAS, cv=5)
        mdl.fit(X_oh[tr_loc], residual[tr_loc])
        pred_tr = mdl.predict(X_oh[tr_loc])
        fold_train_rae.append(
            float(((residual[tr_loc] - pred_tr) ** 2).mean()) ** 0.5
        )
        oof[va_loc] = mdl.predict(X_oh[va_loc])
        fold_alphas.append(float(mdl.alpha_))
    return oof, fold_alphas, fold_train_rae


def _slsqp_blend(P: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """SLSQP-fit non-negative weights on simplex; minimize RAE."""
    K = P.shape[1]
    w0 = np.full(K, 1.0 / K)

    def loss(w):
        return rae(y, P @ w)

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bnds = [(0.0, 1.0)] * K
    res = minimize(loss, w0, method="SLSQP", bounds=bnds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-9})
    w = np.clip(res.x, 0.0, 1.0)
    w = w / max(w.sum(), 1e-12)
    return w, float(rae(y, P @ w))


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- RF path embeddings (leaf-id one-hot -> RidgeCV)")
    print(f"          anchor={ANCHOR}  RF(n={RF_N_EST}, depth={RF_MAX_DEPTH})")
    print(f"          5-fold cross-fit, 5 seeds={SEEDS}")
    print(f"          ref: nb2103 K=28 mean-bag {NB2103_K28_MEAN_BAG_REF:.4f}  "
          f"median-bag {NB2103_K28_MEDIAN_BAG_REF:.4f}  "
          f"margin {DECISION_MARGIN}")
    print("=" * 78)

    # ---- 1. Load 4139 train + 513 test 2265-dim features ----
    if not CACHE_FEAT.exists():
        raise FileNotFoundError(f"missing combined feature cache: {CACHE_FEAT}")
    cache = np.load(CACHE_FEAT)
    X_tr = cache["X_tr"].astype(np.float32)
    X_te = cache["X_te"].astype(np.float32)
    print(f"[feat] X_tr {X_tr.shape}  X_te {X_te.shape}")

    # ---- 2. Load 4139 train labels + chemprop_aux OOF (4139) + te (513) ----
    train_df = load_train()
    y_tr = train_df["pec50"].to_numpy(dtype=np.float64)
    assert y_tr.shape[0] == X_tr.shape[0], (
        f"train y/X mismatch: {y_tr.shape} vs {X_tr.shape}"
    )
    oof_anchor_tr = np.load(OOF_ANCHOR).astype(np.float64)
    assert oof_anchor_tr.shape[0] == X_tr.shape[0], (
        f"oof anchor shape {oof_anchor_tr.shape} != {X_tr.shape[0]}"
    )
    te_anchor_513 = np.load(TE_ANCHOR).astype(np.float64)
    assert te_anchor_513.shape[0] == X_te.shape[0]

    # Unblind index + truth
    unb_idx = np.load(UNB_IDX_PATH)
    y_unb = np.load(Y_UNB_PATH).astype(np.float64)
    n_unb = len(y_unb)
    n_test = X_te.shape[0]
    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    print(f"[load] te_{ANCHOR} in_RAE(unb_idx) = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")

    # Train residual target (signed)
    resid_tr = y_tr - oof_anchor_tr
    resid_unb = y_unb - anchor_unb
    print(f"[resid] train: mean={resid_tr.mean():+.4f} std={resid_tr.std():.4f}")
    print(f"[resid] unblind: mean={resid_unb.mean():+.4f} "
          f"std={resid_unb.std():.4f}")

    # ---- 3. Train RF on 4139 train (target = residual) ----
    print("\n" + "-" * 78)
    print(f"RANDOM FOREST FIT: n={RF_N_EST}, max_depth={RF_MAX_DEPTH}, "
          f"target=chemprop_aux residual, seed={RF_SEED_TRAIN}")
    print("-" * 78)
    t_rf = time.time()
    rf = RandomForestRegressor(
        n_estimators=RF_N_EST,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=5,
        n_jobs=2,
        random_state=RF_SEED_TRAIN,
        bootstrap=True,
    )
    rf.fit(X_tr, resid_tr)
    rf_wall = time.time() - t_rf
    rf_train_pred = rf.predict(X_tr)
    rf_train_r2 = float(
        1.0 - ((resid_tr - rf_train_pred) ** 2).sum()
        / max(((resid_tr - resid_tr.mean()) ** 2).sum(), 1e-12)
    )
    print(f"   RF fit done in {rf_wall:.1f}s   train R^2 (residual) = "
          f"{rf_train_r2:+.3f}")

    # ---- 4. tree.apply on train + test -> leaf-id matrices ----
    print("\n" + "-" * 78)
    print("LEAF-ID EXTRACTION (rf.apply)")
    print("-" * 78)
    t_leaf = time.time()
    leaf_tr = rf.apply(X_tr).astype(np.int32)   # (4139, 200)
    leaf_te = rf.apply(X_te).astype(np.int32)   # (513, 200)
    print(f"   leaf_tr {leaf_tr.shape}  leaf_te {leaf_te.shape}  "
          f"(wall {time.time() - t_leaf:.1f}s)")

    # Slice unblind leaves
    leaf_unb = leaf_te[unb_idx]                  # (253, 200)
    print(f"   leaf_unb (253, 200) -- slice from leaf_te[unb_idx]")
    n_unique_leaves_per_tree = [
        int(np.unique(leaf_tr[:, j]).size) for j in range(RF_N_EST)
    ]
    print(f"   per-tree unique-leaves: min={min(n_unique_leaves_per_tree)} "
          f"median={int(np.median(n_unique_leaves_per_tree))} "
          f"max={max(n_unique_leaves_per_tree)}   "
          f"sum={sum(n_unique_leaves_per_tree)}")

    # ---- OneHotEncoder fit on UNION (train + test) to handle test-only leaves ----
    print("\n" + "-" * 78)
    print("ONE-HOT LEAF EMBEDDING (OneHotEncoder on union of train+test leaves)")
    print("-" * 78)
    t_oh = time.time()
    leaf_union = np.vstack([leaf_tr, leaf_te])
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
    enc.fit(leaf_union)
    X_unb_oh = enc.transform(leaf_unb)
    X_te_oh = enc.transform(leaf_te)
    X_tr_oh = enc.transform(leaf_tr)
    n_features_oh = X_unb_oh.shape[1]
    print(f"   OH dims: {n_features_oh}  (sum unique leaves)")
    print(f"   X_unb_oh nnz={X_unb_oh.nnz}  density="
          f"{X_unb_oh.nnz / (X_unb_oh.shape[0] * n_features_oh):.4e}")
    print(f"   OH wall = {time.time() - t_oh:.1f}s")

    # ---- 5. 5-fold cross-fit RidgeCV on unblind residual (5 seeds) ----
    print("\n" + "-" * 78)
    print(f"RIDGE CV CROSS-FIT ON 253  alphas=logspace(-3,3,7)  "
          f"folds={N_FOLDS} seeds={SEEDS}")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(SEEDS), n_unb), dtype=np.float64)
    per_seed_records: list[dict] = []
    for i, s in enumerate(SEEDS):
        t_seed = time.time()
        resid_oof_s, alphas_s, train_rmse_s = _cross_fit_ridge_one_seed(
            X_unb_oh, resid_unb, s
        )
        pred_corr_s = anchor_unb + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        delta_s = rae_s - rae_anchor
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_chemprop_aux": delta_s,
            "fold_alphas": alphas_s,
            "fold_train_rmse_resid": train_rmse_s,
            "resid_oof_mean": float(resid_oof_s.mean()),
            "resid_oof_std": float(resid_oof_s.std()),
            "wall_sec": round(time.time() - t_seed, 2),
        })
        print(f"   seed {s:3d}: corr_RAE={rae_s:.4f}  "
              f"(d_anchor={delta_s:+.4f})  alphas={alphas_s}  "
              f"wall={time.time() - t_seed:.1f}s")

    mean_bag = per_seed_corrected.mean(axis=0)
    median_bag = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag))
    rae_median_bag = float(rae(y_unb, median_bag))
    per_seed_rae = [r["rae_corrected"] for r in per_seed_records]
    rae_per_seed_mean = float(np.mean(per_seed_rae))
    rae_per_seed_std = float(np.std(per_seed_rae))
    print(f"\n   per-seed RAE = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   per-seed mean = {rae_per_seed_mean:.4f}  "
          f"std = {rae_per_seed_std:.4f}")
    print(f"   mean-bag RAE   = {rae_mean_bag:.4f}  "
          f"(d_anchor={rae_mean_bag - rae_anchor:+.4f}  "
          f"d_nb2103_mean={rae_mean_bag - NB2103_K28_MEAN_BAG_REF:+.4f})")
    print(f"   median-bag RAE = {rae_median_bag:.4f}  "
          f"(d_anchor={rae_median_bag - rae_anchor:+.4f}  "
          f"d_nb2103_median={rae_median_bag - NB2103_K28_MEDIAN_BAG_REF:+.4f})")

    # ---- 6. Save OOF artefacts ----
    np.save(DATA_PROCESSED / f"{TAG}_per_seed_corrected_oof.npy",
            per_seed_corrected.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy",
            mean_bag.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_median_bag_oof.npy",
            median_bag.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_per_seed_corrected_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_mean_bag_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_median_bag_oof.npy'}")

    # ---- 7. SLSQP blend with nb2103 K=28 ----
    print("\n" + "-" * 78)
    print("SLSQP BLEND vs nb2103 K=28 (mean-bag)")
    print("-" * 78)
    if not NB2103_K28_OOF_PATH.exists():
        print(f"   [warn] missing {NB2103_K28_OOF_PATH} -- skipping SLSQP blend")
        slsqp_record = {"skipped": True}
    else:
        nb2103_oof = np.load(NB2103_K28_OOF_PATH).astype(np.float64)
        assert nb2103_oof.shape[0] == n_unb, (
            f"nb2103 OOF shape {nb2103_oof.shape} != {n_unb}"
        )
        rae_nb2103 = float(rae(y_unb, nb2103_oof))
        print(f"   nb2103 K=28 mean-bag in_RAE(253) = {rae_nb2103:.4f}  "
              f"(ref {NB2103_K28_MEAN_BAG_REF:.4f})")
        # 2-way blend
        P2 = np.column_stack([nb2103_oof, mean_bag])
        w2, rae_blend2 = _slsqp_blend(P2, y_unb)
        print(f"   SLSQP 2-way (nb2103, nb1124_mean): w={w2}  "
              f"RAE={rae_blend2:.4f}")
        # 2-way median bag variant
        P2m = np.column_stack([nb2103_oof, median_bag])
        w2m, rae_blend2m = _slsqp_blend(P2m, y_unb)
        print(f"   SLSQP 2-way (nb2103, nb1124_median): w={w2m}  "
              f"RAE={rae_blend2m:.4f}")
        # 3-way blend (anchor + both)
        P3 = np.column_stack([anchor_unb, nb2103_oof, mean_bag])
        w3, rae_blend3 = _slsqp_blend(P3, y_unb)
        print(f"   SLSQP 3-way (anchor, nb2103, nb1124_mean): w={w3}  "
              f"RAE={rae_blend3:.4f}")
        slsqp_record = {
            "rae_nb2103_K28_in_sample_253": rae_nb2103,
            "blend2_mean_w_nb2103": float(w2[0]),
            "blend2_mean_w_nb1124": float(w2[1]),
            "blend2_mean_rae": rae_blend2,
            "blend2_median_w_nb2103": float(w2m[0]),
            "blend2_median_w_nb1124": float(w2m[1]),
            "blend2_median_rae": rae_blend2m,
            "blend3_w_anchor": float(w3[0]),
            "blend3_w_nb2103": float(w3[1]),
            "blend3_w_nb1124": float(w3[2]),
            "blend3_rae": rae_blend3,
            "delta_blend2_mean_vs_nb2103": rae_blend2 - rae_nb2103,
            "delta_blend2_median_vs_nb2103": rae_blend2m - rae_nb2103,
        }

    # ---- 8. Decision verdict ----
    print("\n" + "=" * 78)
    print("DECISION TABLE")
    print("=" * 78)
    print(f"   {'metric':>20s}  {'nb1124':>10s}  {'nb2103':>10s}  {'delta':>10s}  "
          f"verdict")
    print(f"   {'mean-bag RAE':>20s}  {rae_mean_bag:>10.4f}  "
          f"{NB2103_K28_MEAN_BAG_REF:>10.4f}  "
          f"{rae_mean_bag - NB2103_K28_MEAN_BAG_REF:>+10.4f}  ", end="")
    if rae_mean_bag < NB2103_K28_MEAN_BAG_REF - DECISION_MARGIN:
        print("BEATS_NB2103_MEAN")
    elif abs(rae_mean_bag - NB2103_K28_MEAN_BAG_REF) < DECISION_MARGIN:
        print("FLAT_VS_NB2103_MEAN")
    else:
        print("WORSE_THAN_NB2103_MEAN")
    print(f"   {'median-bag RAE':>20s}  {rae_median_bag:>10.4f}  "
          f"{NB2103_K28_MEDIAN_BAG_REF:>10.4f}  "
          f"{rae_median_bag - NB2103_K28_MEDIAN_BAG_REF:>+10.4f}  ", end="")
    if rae_median_bag < NB2103_K28_MEDIAN_BAG_REF - DECISION_MARGIN:
        print("BEATS_NB2103_MEDIAN")
    elif abs(rae_median_bag - NB2103_K28_MEDIAN_BAG_REF) < DECISION_MARGIN:
        print("FLAT_VS_NB2103_MEDIAN")
    else:
        print("WORSE_THAN_NB2103_MEDIAN")

    beats_nb2103_mean = rae_mean_bag < NB2103_K28_MEAN_BAG_REF - DECISION_MARGIN
    beats_nb2103_median = (
        rae_median_bag < NB2103_K28_MEDIAN_BAG_REF - DECISION_MARGIN
    )
    is_winner = beats_nb2103_mean or beats_nb2103_median

    if is_winner:
        global_verdict = (
            f"WINNER_NB1124_BEATS_NB2103_K28"
            f"{'_MEAN' if beats_nb2103_mean else ''}"
            f"{'_MEDIAN' if beats_nb2103_median else ''}"
        )
    elif abs(rae_mean_bag - NB2103_K28_MEAN_BAG_REF) < DECISION_MARGIN:
        global_verdict = "FLAT_VS_NB2103_K28_RF_EMBED_TIES_LGBM"
    else:
        global_verdict = "NB2103_K28_REMAINS_OPTIMAL_RF_EMBED_NO_SIGNAL"
    print(f"\n   GLOBAL VERDICT = {global_verdict}")

    # ---- 9. Deploy CSV (only if winner) ----
    deploy_record: dict = {"built": False}
    if is_winner:
        print("\n" + "-" * 78)
        print("DEPLOY ARTEFACTS (winner branch)")
        print("-" * 78)
        # Refit Ridge on ALL 253 unblind for each seed; bag residual_513
        per_seed_residual_513 = np.zeros(
            (len(SEEDS), n_test), dtype=np.float64
        )
        for i, s in enumerate(SEEDS):
            mdl = RidgeCV(alphas=RIDGE_ALPHAS, cv=5)
            mdl.fit(X_unb_oh, resid_unb)
            per_seed_residual_513[i] = mdl.predict(X_te_oh).astype(np.float64)
        # NOTE: RidgeCV is deterministic on a given X/y; per-seed loop is
        # a placeholder for any future seed-jitter; here all rows are equal.
        mean_bag_resid_513 = per_seed_residual_513.mean(axis=0)
        te_nb1124 = te_anchor_513 + mean_bag_resid_513
        in_rae_te = float(rae(y_unb, te_nb1124[unb_idx]))
        delta_vs_anchor_in = in_rae_te - rae_anchor
        print(f"   te_nb1124 shape={te_nb1124.shape}  "
              f"in_RAE(unb_idx)={in_rae_te:.4f}  "
              f"(d_anchor={delta_vs_anchor_in:+.4f})")
        np.save(DATA_PROCESSED / f"te_{TAG}.npy",
                te_nb1124.astype(np.float32))
        print(f"   [save] {DATA_PROCESSED / f'te_{TAG}.npy'}")
        # CSV
        te = load_test()
        te_smiles = te["smiles"].values
        te_names = te["name"].values
        sub_df = pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": te_nb1124.astype(np.float32),
        })
        sub_path = SUBMISSIONS_DIR / f"{TAG}_rf_path_embed.csv"
        sub_df.to_csv(sub_path, index=False)
        print(f"   [save] {sub_path}")
        deploy_record = {
            "built": True,
            "te_path": str(DATA_PROCESSED / f"te_{TAG}.npy"),
            "csv_path": str(sub_path),
            "in_rae_te_unb": in_rae_te,
            "delta_vs_anchor_in_sample": delta_vs_anchor_in,
        }
    else:
        print("\n   (no deploy artefacts -- nb1124 did not beat nb2103)")

    # ---- 10. Summary ----
    summary = {
        "tag": TAG,
        "method": ("RandomForest path embedding (n_est=200 max_depth=12) -> "
                   "one-hot leaf-ids -> RidgeCV(alphas=logspace(-3,3,7)) "
                   "honest 5-fold cross-fit on 253 unblind, 5-seed bag"),
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(TE_ANCHOR),
        "rf_n_estimators": RF_N_EST,
        "rf_max_depth": RF_MAX_DEPTH,
        "rf_min_samples_leaf": 5,
        "rf_seed_train": RF_SEED_TRAIN,
        "rf_train_residual_R2_on_train": rf_train_r2,
        "rf_target": "y_train - oof_chemprop_aux (signed residual)",
        "ridge_alphas": RIDGE_ALPHAS.tolist(),
        "n_features_2265_dim_combined": int(X_tr.shape[1]),
        "n_unique_leaves_per_tree_min": int(min(n_unique_leaves_per_tree)),
        "n_unique_leaves_per_tree_median": int(np.median(n_unique_leaves_per_tree)),
        "n_unique_leaves_per_tree_max": int(max(n_unique_leaves_per_tree)),
        "n_features_oh": int(n_features_oh),
        "ohencoded_X_unb_nnz": int(X_unb_oh.nnz),
        "n_unb": n_unb,
        "n_test": n_test,
        "n_train": int(X_tr.shape[0]),
        "n_folds": N_FOLDS,
        "seeds": SEEDS,
        "rae_anchor_chemprop_aux_unb": rae_anchor,
        "rae_per_seed": per_seed_rae,
        "rae_per_seed_mean": rae_per_seed_mean,
        "rae_per_seed_std": rae_per_seed_std,
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_anchor": rae_mean_bag - rae_anchor,
        "delta_median_bag_vs_anchor": rae_median_bag - rae_anchor,
        "delta_mean_bag_vs_nb2103_K28_mean": (
            rae_mean_bag - NB2103_K28_MEAN_BAG_REF
        ),
        "delta_median_bag_vs_nb2103_K28_median": (
            rae_median_bag - NB2103_K28_MEDIAN_BAG_REF
        ),
        "beats_nb2103_K28_mean": bool(beats_nb2103_mean),
        "beats_nb2103_K28_median": bool(beats_nb2103_median),
        "is_winner": bool(is_winner),
        "verdict": global_verdict,
        "decision_margin": DECISION_MARGIN,
        "per_seed_records": per_seed_records,
        "slsqp_blend": slsqp_record,
        "deploy": deploy_record,
        "nb2103_K28_mean_bag_ref": NB2103_K28_MEAN_BAG_REF,
        "nb2103_K28_median_bag_ref": NB2103_K28_MEDIAN_BAG_REF,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "pre_unblind_clean": True,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "rf_n_estimators", "rf_max_depth", "n_features_oh",
        "rae_anchor_chemprop_aux_unb",
        "rae_mean_bag", "rae_median_bag",
        "delta_mean_bag_vs_nb2103_K28_mean",
        "delta_median_bag_vs_nb2103_K28_median",
        "is_winner", "verdict", "pre_unblind_clean",
    ):
        print(f"  {k}: {res.get(k)}")
    sl = res.get("slsqp_blend", {})
    if not sl.get("skipped"):
        print("\n==== SLSQP BLEND ====")
        for k in ("blend2_mean_w_nb2103", "blend2_mean_w_nb1124",
                  "blend2_mean_rae",
                  "blend2_median_w_nb2103", "blend2_median_w_nb1124",
                  "blend2_median_rae",
                  "blend3_w_anchor", "blend3_w_nb2103", "blend3_w_nb1124",
                  "blend3_rae"):
            print(f"  {k}: {sl.get(k)}")
