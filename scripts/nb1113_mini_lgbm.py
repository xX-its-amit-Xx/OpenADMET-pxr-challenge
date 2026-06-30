"""nb1113 -- Per-compound mini-LGBM trained on top-50 nearest neighbors.

For each test compound (513 rows), find the 50 nearest train compounds by
ECFP4 Tanimoto similarity, train a tiny LGBM on combined features (Morgan
+ RDKit, 2265 dims) restricted to those 50 + 1 test row, and predict the
single test pec50. Fall back to chemprop_aux when the neighborhood is too
sparse.

PROTOCOL:
  1. Load 4139 train + 513 test combined features (cache_combined_features.npz)
     and 4139 train pec50 labels.
  2. Compute ECFP4 (Morgan, 2048-bit) for both 4139 train + 513 test.
  3. For each test row: top-50 nearest train by Tanimoto (block matmul).
     Train mini-LGBM(n_est=200, leaves=15, lr=0.05, lambda_l2=2) on the
     50 neighbor combined-feature rows (labels: train pec50). Predict the
     test row.
  4. Fallback rule: if <5 of the 50 neighbors have sim >= 0.4, replace
     the mini-LGBM prediction with chemprop_aux.
  5. Honest cross-fit on 253 unblind: 5-fold KFold over unb_idx; for each
     held-out 1/5, re-do the per-row mini-LGBM using ONLY the 4139 train
     (no unblind leakage). Then compute RAE on the 253-row pooled OOF.
  6. Compare vs chemprop_aux ref (0.6216) and nb2103 K=28 (0.4737).
  7. If RAE beats both, build deploy CSV using the 513 final predictions.

Notes:
  - Each mini-LGBM is independent; we use n_jobs=1 per fit since 513
    fits are already CPU-bound. Total wall is dominated by the 513 fits.
  - "Honest" cross-fit means the train pool itself never changes (the
    4139 are always available). The fold split is over the 253 unblind
    only for the OOF eval. This matches the LB-faithful protocol in
    feedback_te_vs_pred_oof_protocol.md.
  - We DO NOT augment with the 4/5 unblind labels in cross-fit: the
    purpose is to see if the per-row mini-LGBM alone (with chemprop_aux
    fallback) is competitive. That gives a PRE-unblind-clean number.

Outputs:
  scripts/nb1113_mini_lgbm.py
  data/processed/nb1113_summary.json
  data/processed/nb1113_te.npy              (513,) deploy predictions
  data/processed/nb1113_pred_oof.npy        (253,) honest cross-fit OOF
  submissions/nb1113_mini_lgbm.csv          (only if RAE < min(0.6216, 0.4737))
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
from sklearn.model_selection import KFold
import lightgbm as lgb
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import morgan_fp_batch
from pxr.data import load_train, load_test
from pxr.eval import rae
from pxr.featurize import impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1113"

# References (PRE-unblind path, honest cross-fit RAE on 253 unblind)
CHEMPROP_AUX_REF = 0.6216
NB2103_K28_REF = 0.4737   # nb2103 K=28 mean-bag, the leading sibling
NB2103_K28_MEDIAN_REF = 0.4698

# Mini-LGBM hyperparams (per request)
K_NEIGHBORS = 50
N_ESTIMATORS = 200
NUM_LEAVES = 15
LEARNING_RATE = 0.05
REG_LAMBDA = 2.0
MIN_CHILD_SAMPLES = 3   # tiny to allow splits on 50-row training sets

# Fallback rule (per request)
FALLBACK_MIN_NEIGHBORS = 5
FALLBACK_SIM_THRESHOLD = 0.4

# Cross-fit
N_FOLDS = 5
CROSSFIT_SEED = 42

# Anchor for fallback
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

CACHE_COMBINED = DATA_PROCESSED / "cache_combined_features.npz"


def _tanimoto_topk(fp_q: np.ndarray, fp_pool: np.ndarray, k: int):
    """Block matmul Tanimoto, returns (n_q, k) top indices + sims."""
    a = fp_q.astype(np.float32)
    b = fp_pool.astype(np.float32)
    a_sum = a.sum(axis=1)
    b_sum = b.sum(axis=1)
    n_q = a.shape[0]
    n_pool = b.shape[0]
    if k > n_pool:
        k = n_pool
    top_idx = np.zeros((n_q, k), dtype=np.int32)
    top_sim = np.zeros((n_q, k), dtype=np.float32)
    BLOCK = 64
    for s in range(0, n_q, BLOCK):
        e = min(n_q, s + BLOCK)
        inter = a[s:e] @ b.T
        denom = a_sum[s:e, None] + b_sum[None, :] - inter
        denom = np.maximum(denom, 1.0)
        sim = inter / denom
        part = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
        row_idx = np.arange(e - s)[:, None]
        sim_part = sim[row_idx, part]
        order = np.argsort(-sim_part, axis=1)
        idx_part = part[row_idx, order]
        top_idx[s:e] = idx_part
        top_sim[s:e] = sim[row_idx, idx_part]
    return top_idx, top_sim


def _mini_lgbm_params(seed: int) -> dict:
    return dict(
        objective="regression",
        n_estimators=N_ESTIMATORS,
        num_leaves=NUM_LEAVES,
        learning_rate=LEARNING_RATE,
        reg_lambda=REG_LAMBDA,
        min_child_samples=MIN_CHILD_SAMPLES,
        random_state=seed,
        n_jobs=1,
        verbosity=-1,
    )


def _predict_one(X_train_neigh: np.ndarray,
                  y_train_neigh: np.ndarray,
                  X_test_row: np.ndarray,
                  seed: int = 0) -> float:
    """Train mini-LGBM on neighbors only, predict the single test row."""
    mdl = lgb.LGBMRegressor(**_mini_lgbm_params(seed))
    mdl.fit(X_train_neigh, y_train_neigh)
    return float(mdl.predict(X_test_row.reshape(1, -1))[0])


def _run_per_row_predict(X_train_full: np.ndarray,
                          y_train_full: np.ndarray,
                          fp_train_full: np.ndarray,
                          X_query: np.ndarray,
                          fp_query: np.ndarray,
                          anchor_fallback: np.ndarray,
                          tag: str = "") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each row in X_query, train mini-LGBM on top-K train neighbors and predict.

    Returns:
        preds (n_query,)
        fallback_mask (n_query,) bool   True => fallback used
        topk_sim_floor4 (n_query,) int   # neighbors with sim >= 0.4
    """
    n_query = X_query.shape[0]
    top_idx, top_sim = _tanimoto_topk(fp_query, fp_train_full, k=K_NEIGHBORS)
    preds = np.empty(n_query, dtype=np.float64)
    fallback_mask = np.zeros(n_query, dtype=bool)
    topk_sim_floor4 = (top_sim >= FALLBACK_SIM_THRESHOLD).sum(axis=1)
    t_block = time.time()
    for i in range(n_query):
        n_good = int(topk_sim_floor4[i])
        if n_good < FALLBACK_MIN_NEIGHBORS:
            preds[i] = anchor_fallback[i]
            fallback_mask[i] = True
            continue
        neigh_idx = top_idx[i]
        Xn = X_train_full[neigh_idx]
        yn = y_train_full[neigh_idx]
        preds[i] = _predict_one(Xn, yn, X_query[i])
        if (i + 1) % 100 == 0:
            elap = time.time() - t_block
            rate = (i + 1) / max(elap, 1e-6)
            eta = (n_query - i - 1) / max(rate, 1e-6)
            print(f"   [{tag}] {i+1:>4d}/{n_query}  rate={rate:.1f}/s  eta={eta:.0f}s  "
                  f"fallback_count={int(fallback_mask[:i+1].sum())}")
    return preds, fallback_mask, topk_sim_floor4


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- per-compound mini-LGBM on top-{K_NEIGHBORS} Tanimoto neighbors")
    print(f"   LGBM: n_est={N_ESTIMATORS} leaves={NUM_LEAVES} lr={LEARNING_RATE} "
          f"l2={REG_LAMBDA} min_child={MIN_CHILD_SAMPLES}")
    print(f"   Fallback: <{FALLBACK_MIN_NEIGHBORS} neighbors with sim>={FALLBACK_SIM_THRESHOLD} "
          f"-> chemprop_aux")
    print(f"   Refs: chemprop_aux={CHEMPROP_AUX_REF:.4f}  "
          f"nb2103_K28_mean={NB2103_K28_REF:.4f}  nb2103_K28_median={NB2103_K28_MEDIAN_REF:.4f}")
    print("=" * 78)

    # ---- Load combined features (Morgan + RDKit) ----
    if not CACHE_COMBINED.exists():
        raise FileNotFoundError(f"missing {CACHE_COMBINED}")
    cache = np.load(CACHE_COMBINED)
    X_tr_combined = cache["X_tr"].astype(np.float32)  # (4139, 2265)
    X_te_combined = cache["X_te"].astype(np.float32)  # (513, 2265)
    print(f"[load] X_tr={X_tr_combined.shape}  X_te={X_te_combined.shape}")

    # Impute (just to be safe -- the cache should already be clean)
    X_tr_combined = impute(X_tr_combined).astype(np.float32)
    X_te_combined = impute(X_te_combined).astype(np.float32)

    # ---- Load truth labels ----
    tr = load_train()
    te = load_test()
    n_train = len(tr)
    n_test = len(te)
    if n_train != X_tr_combined.shape[0] or n_test != X_te_combined.shape[0]:
        raise ValueError(
            f"shape mismatch: train {n_train} vs {X_tr_combined.shape[0]}; "
            f"test {n_test} vs {X_te_combined.shape[0]}"
        )
    y_tr = tr["pec50"].to_numpy(dtype=np.float64)
    train_smiles = tr["smiles"].astype(str).tolist()
    test_smiles = te["smiles"].astype(str).tolist()
    print(f"[load] y_tr mean={y_tr.mean():.3f}  std={y_tr.std():.3f}")

    # ---- Unblind set ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_unb={n_unb}")

    # ---- Anchor (chemprop_aux) ----
    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"missing {ANCHOR_TE_PATH}")
    te_anchor = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor.shape[0] != n_test:
        raise ValueError(f"anchor shape mismatch: {te_anchor.shape}")
    rae_anchor_unb = float(rae(y_unb, te_anchor[unb_idx]))
    print(f"[load] chemprop_aux te[unb_idx] in_RAE = {rae_anchor_unb:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")

    # ---- ECFP4 fingerprints for Tanimoto neighbor lookup ----
    print("\n[fp] computing ECFP4 Morgan (2048-bit) for train + test ...")
    fp_train_full = morgan_fp_batch(train_smiles).astype(np.uint8)
    fp_test_full = morgan_fp_batch(test_smiles).astype(np.uint8)
    print(f"   fp_train={fp_train_full.shape}  fp_test={fp_test_full.shape}")

    # =========================================================================
    # PART A: Deploy predictions for all 513 test rows (full 4139 train pool)
    # =========================================================================
    print("\n" + "-" * 78)
    print("PART A: deploy predictions (4139-row pool, all 513 test queries)")
    print("-" * 78)
    t_a = time.time()
    pred_test_513, fallback_test, ngood_test = _run_per_row_predict(
        X_train_full=X_tr_combined,
        y_train_full=y_tr,
        fp_train_full=fp_train_full,
        X_query=X_te_combined,
        fp_query=fp_test_full,
        anchor_fallback=te_anchor,
        tag="deploy",
    )
    fallback_rate_test = float(fallback_test.mean())
    print(f"\n   PART A done in {time.time()-t_a:.1f}s")
    print(f"   fallback used on {int(fallback_test.sum())}/{n_test} test rows "
          f"({fallback_rate_test*100:.2f}%)")
    print(f"   ngood (sim>=0.4) median={int(np.median(ngood_test))}  "
          f"mean={ngood_test.mean():.1f}  max={int(ngood_test.max())}  "
          f"min={int(ngood_test.min())}")
    np.save(DATA_PROCESSED / f"{TAG}_te.npy", pred_test_513.astype(np.float32))
    print(f"   [save] {DATA_PROCESSED / (TAG + '_te.npy')}")

    # In-sample (te slice to unblind) -- diagnostic only
    in_rae = float(rae(y_unb, pred_test_513[unb_idx]))
    print(f"   in-sample te[unb_idx] RAE = {in_rae:.4f}  (vs anchor {rae_anchor_unb:.4f})")

    # =========================================================================
    # PART B: Honest 5-fold cross-fit on 253 unblind (PRE-unblind clean)
    #         For each fold, exclude the held-out 1/5 of unblind from the
    #         pool AND from the test queries. Pool is 4139 train MINUS any
    #         held-out unblind rows (but unb_idx points into the 513 test set,
    #         not the 4139 train set -- so there is no train overlap to
    #         remove). The pool stays at 4139 across all folds.
    # =========================================================================
    print("\n" + "-" * 78)
    print(f"PART B: honest {N_FOLDS}-fold cross-fit on {n_unb} unblind rows")
    print("-" * 78)
    pred_oof = np.full(n_unb, np.nan, dtype=np.float64)
    fallback_oof = np.zeros(n_unb, dtype=bool)
    ngood_oof = np.zeros(n_unb, dtype=np.int32)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CROSSFIT_SEED)

    for fold, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n_unb))):
        print(f"\n--- fold {fold+1}/{N_FOLDS} : {len(tr_loc)} train_unb / "
              f"{len(va_loc)} held-out_unb ---")
        # Held-out test rows (positions in 513 test set)
        held_513_idx = unb_idx[va_loc]
        X_query_fold = X_te_combined[held_513_idx]
        fp_query_fold = fp_test_full[held_513_idx]
        anchor_fold = te_anchor[held_513_idx]

        # Pool: all 4139 train (no unblind-train mixing -- unb_idx is test set)
        pred_fold, fallback_fold, ngood_fold = _run_per_row_predict(
            X_train_full=X_tr_combined,
            y_train_full=y_tr,
            fp_train_full=fp_train_full,
            X_query=X_query_fold,
            fp_query=fp_query_fold,
            anchor_fallback=anchor_fold,
            tag=f"fold{fold+1}",
        )
        pred_oof[va_loc] = pred_fold
        fallback_oof[va_loc] = fallback_fold
        ngood_oof[va_loc] = ngood_fold
        fold_rae = float(rae(y_unb[va_loc], pred_fold))
        anchor_fold_rae = float(rae(y_unb[va_loc], anchor_fold))
        print(f"   fold {fold+1} RAE = {fold_rae:.4f}  "
              f"(anchor on same rows = {anchor_fold_rae:.4f})  "
              f"fallback={int(fallback_fold.sum())}/{len(va_loc)}")

    assert not np.isnan(pred_oof).any(), "pred_oof has NaNs"
    pooled_rae = float(rae(y_unb, pred_oof))
    fallback_rate_oof = float(fallback_oof.mean())
    print(f"\n[POOLED OOF] cross-fit RAE on {n_unb} unblind = {pooled_rae:.4f}")
    print(f"   fallback rate = {fallback_rate_oof*100:.2f}%  "
          f"({int(fallback_oof.sum())}/{n_unb})")

    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy", pred_oof.astype(np.float32))
    print(f"   [save] {DATA_PROCESSED / (TAG + '_pred_oof.npy')}")

    # =========================================================================
    # Verdict + deploy CSV
    # =========================================================================
    delta_vs_anchor = pooled_rae - CHEMPROP_AUX_REF
    delta_vs_nb2103_mean = pooled_rae - NB2103_K28_REF
    delta_vs_nb2103_median = pooled_rae - NB2103_K28_MEDIAN_REF
    beats_anchor = pooled_rae < CHEMPROP_AUX_REF - 0.005
    beats_nb2103_mean = pooled_rae < NB2103_K28_REF - 0.005
    beats_nb2103_median = pooled_rae < NB2103_K28_MEDIAN_REF - 0.005

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"   pooled cross-fit RAE       = {pooled_rae:.4f}")
    print(f"   d_vs_chemprop_aux (0.6216) = {delta_vs_anchor:+.4f}  "
          f"{'(BEATS)' if beats_anchor else '(does not beat)'}")
    print(f"   d_vs_nb2103_K28_mean(0.4737) = {delta_vs_nb2103_mean:+.4f}  "
          f"{'(BEATS)' if beats_nb2103_mean else '(does not beat)'}")
    print(f"   d_vs_nb2103_K28_median(0.4698)= {delta_vs_nb2103_median:+.4f}  "
          f"{'(BEATS)' if beats_nb2103_median else '(does not beat)'}")

    deploy_csv_path = None
    if beats_anchor and beats_nb2103_mean:
        verdict = "BEATS_BOTH_REFS_DEPLOY"
    elif beats_anchor:
        verdict = "BEATS_CHEMPROP_AUX_ONLY"
    elif beats_nb2103_mean:
        verdict = "BEATS_NB2103_K28_ONLY"
    else:
        verdict = "DOES_NOT_BEAT_REFS"
    print(f"   verdict = {verdict}")

    if beats_anchor and beats_nb2103_mean:
        sub = pd.DataFrame({
            "SMILES": te["smiles"].astype(str).values,
            "Molecule Name": te["name"].astype(str).values,
            "pEC50": pred_test_513.astype(np.float64),
        })
        deploy_csv_path = SUBMISSIONS / f"{TAG}_mini_lgbm.csv"
        sub.to_csv(deploy_csv_path, index=False)
        print(f"   [DEPLOY] wrote {deploy_csv_path}  ({len(sub)} rows)")

    # ---- Summary JSON ----
    summary = {
        "tag": TAG,
        "method": (
            f"per_compound_mini_lgbm_top{K_NEIGHBORS}_tanimoto_combined_feats"
            f"_with_chemprop_aux_fallback"
        ),
        "k_neighbors": K_NEIGHBORS,
        "fallback_min_neighbors": FALLBACK_MIN_NEIGHBORS,
        "fallback_sim_threshold": FALLBACK_SIM_THRESHOLD,
        "lgbm_n_estimators": N_ESTIMATORS,
        "lgbm_num_leaves": NUM_LEAVES,
        "lgbm_learning_rate": LEARNING_RATE,
        "lgbm_reg_lambda": REG_LAMBDA,
        "lgbm_min_child_samples": MIN_CHILD_SAMPLES,
        "n_folds": N_FOLDS,
        "crossfit_seed": CROSSFIT_SEED,
        "feat_dim": int(X_tr_combined.shape[1]),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "anchor": "chemprop_aux",
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "nb2103_K28_mean_ref": NB2103_K28_REF,
        "nb2103_K28_median_ref": NB2103_K28_MEDIAN_REF,
        "rae_anchor_unb_insample": rae_anchor_unb,
        "rae_pooled_crossfit_oof": pooled_rae,
        "rae_te_unb_insample_deploy": in_rae,
        "delta_vs_chemprop_aux": delta_vs_anchor,
        "delta_vs_nb2103_K28_mean": delta_vs_nb2103_mean,
        "delta_vs_nb2103_K28_median": delta_vs_nb2103_median,
        "beats_chemprop_aux": bool(beats_anchor),
        "beats_nb2103_K28_mean": bool(beats_nb2103_mean),
        "beats_nb2103_K28_median": bool(beats_nb2103_median),
        "fallback_rate_test_513": fallback_rate_test,
        "fallback_count_test_513": int(fallback_test.sum()),
        "fallback_rate_oof_253": fallback_rate_oof,
        "fallback_count_oof_253": int(fallback_oof.sum()),
        "ngood_sim_floor4_test_513": {
            "mean": float(ngood_test.mean()),
            "median": int(np.median(ngood_test)),
            "min": int(ngood_test.min()),
            "max": int(ngood_test.max()),
        },
        "ngood_sim_floor4_oof_253": {
            "mean": float(ngood_oof.mean()),
            "median": int(np.median(ngood_oof)),
            "min": int(ngood_oof.min()),
            "max": int(ngood_oof.max()),
        },
        "verdict": verdict,
        "deploy_csv_path": str(deploy_csv_path) if deploy_csv_path else None,
        "te_npy_path": str(DATA_PROCESSED / f"{TAG}_te.npy"),
        "pred_oof_npy_path": str(DATA_PROCESSED / f"{TAG}_pred_oof.npy"),
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
        "k_neighbors", "fallback_min_neighbors", "fallback_sim_threshold",
        "feat_dim", "n_test", "n_unb",
        "rae_anchor_unb_insample",
        "rae_te_unb_insample_deploy",
        "rae_pooled_crossfit_oof",
        "delta_vs_chemprop_aux", "delta_vs_nb2103_K28_mean",
        "beats_chemprop_aux", "beats_nb2103_K28_mean",
        "fallback_rate_test_513", "fallback_rate_oof_253",
        "verdict", "deploy_csv_path",
        "pre_unblind_clean", "wall_sec",
    ):
        print(f"  {k}: {res.get(k)}")
