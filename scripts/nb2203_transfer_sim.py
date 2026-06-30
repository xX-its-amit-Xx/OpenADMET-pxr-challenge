"""nb2203 -- Train-holdout LB transfer simulation for nb2189-style POST-unblind cross-fit.

MOTIVATION:
    Memory feedback_lb_two_regime_calibration warns that POST-unblind in_RAE
    may transfer 0.7-0.9 to LB.  nb2189 anchors a residual model on a
    5-fold OOF anchor computed on the SAME 253-row substrate as the
    residual cross-fit.  Even if nominally honest, this can be subtly
    LB-optimistic because the anchor was selected (and its hyperparams
    tuned) on the same 253 rows that the residual learner sees.

PROTOCOL:
    We avoid touching real unblind labels.  Instead we simulate the
    unblind situation entirely inside the 4139-row CRC training set.

    Step 1.  Use scaffold_kfold_indices(n_splits=16) -> each fold ~258 cpds.
             Pick one fold as the SIMULATED-UNBLIND ('sim_unb').  Pick a
             DIFFERENT fold as the SIMULATED-LB ('sim_lb', UNUSED outer
             253-fold).  Remaining ~3625 cpds form the 'sim_train' base.
    Step 2.  Train a chemprop_aux proxy: LGBM(MSE) on combined features
             on sim_train and predict sim_unb AND sim_lb.  This is the
             tractable PRE-unblind anchor for both folds.
    Step 3a. nb2103-style ANCHOR + LGBM-residual K=28 cross-fit on sim_unb:
                residual = y_sim_unb - anchor_proxy_sim_unb
                5-fold cross-fit LGBM bagged over 5 seeds
                final_pred_sim_unb = anchor + bag_residual_oof
                in_RAE = RAE(y_sim_unb, final_pred_sim_unb)
             EVALUATE OUTER: train LGBM-residual on full sim_unb (no folds)
             predict on sim_lb features; final_pred_sim_lb = anchor + resid;
                outer_RAE = RAE(y_sim_lb, final_pred_sim_lb)
    Step 3b. nb2189-style 5-FOLD-OOF-ANCHOR + LGBM-residual K=20 cross-fit:
                anchor_oof_sim_unb = 5-fold OOF of a 2nd-anchor model
                                     trained INSIDE sim_unb (LGBM on
                                     combined features, 5-fold OOF).  This
                                     mimics nb562_pred_oof's role.
                residual = y_sim_unb - anchor_oof_sim_unb
                5-fold cross-fit LGBM bagged (K=20)
                final_pred_sim_unb = anchor_oof + bag_residual_oof
                in_RAE = RAE(y_sim_unb, final_pred_sim_unb)
             EVALUATE OUTER: refit 2nd-anchor on full sim_unb, predict
                sim_lb to get anchor_sim_lb; refit LGBM-residual on
                full sim_unb residual, predict sim_lb;
                outer_RAE = RAE(y_sim_lb, anchor_sim_lb + resid_sim_lb)
    Step 4.  TRANSFER GAP = outer_RAE - in_RAE for each style.
    Step 5.  If nb2189-style transfer gap is materially larger than
             nb2103-style, this confirms POST-unblind cross-fit overfit.

NOTE on robustness:
    Single (sim_unb, sim_lb) pair is noisy.  We repeat over MULTIPLE
    (i, j) pairs from the 16 scaffold folds and average; also report
    distribution.  Three pairs are enough to expose a robust transfer
    gap.  We use seed 2026 throughout.

Outputs:
    scripts/nb2203_transfer_sim.py
    data/processed/nb2203_summary.json
"""
from __future__ import annotations

import hashlib
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
from sklearn.model_selection import KFold

from pxr.chem import standardize, bemis_murcko
from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED

TAG = "nb2203"

N_SPLITS_SCAFFOLD = 16
PAIRS = [(0, 1), (2, 3), (4, 5)]  # (sim_unb_fold, sim_lb_fold)
KFOLD_SEED = 2026
N_FOLDS_RESID = 5
BAG_SEEDS = [0, 1, 7, 42, 137]

K_NB2103 = 28
K_NB2189 = 20


def _sha256(arr) -> str:
    if isinstance(arr, np.ndarray):
        return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]
    return hashlib.sha256(str(arr).encode()).hexdigest()[:16]


def _lgbm_anchor_params(seed: int = 0) -> dict:
    """chemprop_aux-equivalent proxy: stronger LGBM on combined features."""
    return dict(
        objective="regression",
        n_estimators=500,
        num_leaves=64,
        learning_rate=0.05,
        min_child_samples=10,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )


def _lgbm_resid_params(seed: int) -> dict:
    """nb2103/nb2189 residual LGBM hyperparams (max_depth=4, L=15, lr=0.03)."""
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


def _residual_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray, seed: int) -> np.ndarray:
    """5-fold cross-fit OOF residual prediction for one seed."""
    n = len(residual)
    kf = KFold(n_splits=N_FOLDS_RESID, shuffle=True, random_state=KFOLD_SEED + seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**_lgbm_resid_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _anchor_5fold_oof(X_unb: np.ndarray, y_unb: np.ndarray, seed: int = 0) -> np.ndarray:
    """5-fold OOF anchor INSIDE sim_unb (mimics nb562_pred_oof's role)."""
    n = len(y_unb)
    kf = KFold(n_splits=5, shuffle=True, random_state=KFOLD_SEED + seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        # Use a slightly different LGBM config so it doesn't equal proxy
        mdl = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=400,
            num_leaves=32,
            learning_rate=0.04,
            min_child_samples=8,
            reg_lambda=1.5,
            random_state=seed,
            n_jobs=4,
            verbosity=-1,
        )
        mdl.fit(X_unb[tr_loc], y_unb[tr_loc])
        oof[va_loc] = mdl.predict(X_unb[va_loc])
    return oof


def _rank_top_K_by_shap(X_unb: np.ndarray, residual: np.ndarray, K: int) -> np.ndarray:
    """Approx SHAP-importance proxy: feature_importances_ from quick LGBM fit on residual."""
    mdl = lgb.LGBMRegressor(
        objective="regression",
        max_depth=4,
        num_leaves=15,
        n_estimators=200,
        learning_rate=0.05,
        random_state=0,
        n_jobs=4,
        verbosity=-1,
    )
    mdl.fit(X_unb, residual)
    imp = mdl.feature_importances_.astype(np.float64)
    if imp.sum() <= 0:
        # Degenerate: pick first K
        return np.arange(min(K, X_unb.shape[1]), dtype=np.int32)
    order = np.argsort(-imp)[:K].astype(np.int32)
    return order


def _eval_nb2103_style(X_unb: np.ndarray, y_unb: np.ndarray, anchor_unb: np.ndarray,
                       X_lb: np.ndarray, y_lb: np.ndarray, anchor_lb: np.ndarray,
                       K: int) -> dict:
    """nb2103-style: anchor + LGBM K=K residual cross-fit on sim_unb;
    outer eval on sim_lb."""
    residual_unb = y_unb - anchor_unb
    # SHAP-proxy ranking computed on full sim_unb residual (single-fit)
    topK_idx = _rank_top_K_by_shap(X_unb, residual_unb, K)
    X_unb_K = X_unb[:, topK_idx]
    X_lb_K = X_lb[:, topK_idx]

    # In-sample: 5-fold cross-fit bag
    per_seed_oof = np.zeros((len(BAG_SEEDS), len(y_unb)), dtype=np.float64)
    for si, s in enumerate(BAG_SEEDS):
        resid_oof = _residual_cross_fit_one_seed(X_unb_K, residual_unb, s)
        per_seed_oof[si] = anchor_unb + resid_oof
    mean_bag_in = per_seed_oof.mean(axis=0)
    rae_in_mean = float(rae(y_unb, mean_bag_in))
    rae_in_median = float(rae(y_unb, np.median(per_seed_oof, axis=0)))

    # Outer: refit each seed on full sim_unb, predict sim_lb
    per_seed_lb = np.zeros((len(BAG_SEEDS), len(y_lb)), dtype=np.float64)
    for si, s in enumerate(BAG_SEEDS):
        mdl = lgb.LGBMRegressor(**_lgbm_resid_params(s))
        mdl.fit(X_unb_K, residual_unb)
        per_seed_lb[si] = anchor_lb + mdl.predict(X_lb_K)
    mean_bag_lb = per_seed_lb.mean(axis=0)
    rae_outer_mean = float(rae(y_lb, mean_bag_lb))
    rae_outer_median = float(rae(y_lb, np.median(per_seed_lb, axis=0)))

    return {
        "style": "nb2103",
        "K": int(K),
        "in_RAE_mean": rae_in_mean,
        "in_RAE_median": rae_in_median,
        "outer_RAE_mean": rae_outer_mean,
        "outer_RAE_median": rae_outer_median,
        "transfer_gap_mean": rae_outer_mean - rae_in_mean,
        "transfer_gap_median": rae_outer_median - rae_in_median,
        "anchor_in_RAE": float(rae(y_unb, anchor_unb)),
        "anchor_outer_RAE": float(rae(y_lb, anchor_lb)),
    }


def _eval_nb2189_style(X_unb: np.ndarray, y_unb: np.ndarray, anchor_unb: np.ndarray,
                       X_lb: np.ndarray, y_lb: np.ndarray, anchor_lb: np.ndarray,
                       K: int) -> dict:
    """nb2189-style: 5-fold-OOF-anchor INSIDE sim_unb +
    LGBM K=K residual cross-fit on sim_unb; outer eval on sim_lb."""
    # 5-fold OOF anchor inside sim_unb (mimics nb562_pred_oof)
    anchor_oof_unb = _anchor_5fold_oof(X_unb, y_unb, seed=0)
    residual_unb = y_unb - anchor_oof_unb

    topK_idx = _rank_top_K_by_shap(X_unb, residual_unb, K)
    X_unb_K = X_unb[:, topK_idx]
    X_lb_K = X_lb[:, topK_idx]

    # In-sample: 5-fold cross-fit bag (residual cross-fit on top of OOF anchor)
    per_seed_oof = np.zeros((len(BAG_SEEDS), len(y_unb)), dtype=np.float64)
    for si, s in enumerate(BAG_SEEDS):
        resid_oof = _residual_cross_fit_one_seed(X_unb_K, residual_unb, s)
        per_seed_oof[si] = anchor_oof_unb + resid_oof
    mean_bag_in = per_seed_oof.mean(axis=0)
    rae_in_mean = float(rae(y_unb, mean_bag_in))
    rae_in_median = float(rae(y_unb, np.median(per_seed_oof, axis=0)))

    # Outer: refit anchor on FULL sim_unb (no folds), predict sim_lb;
    # refit residual model on full sim_unb, predict sim_lb features;
    # add the two together.
    deploy_anchor_mdl = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=400,
        num_leaves=32,
        learning_rate=0.04,
        min_child_samples=8,
        reg_lambda=1.5,
        random_state=0,
        n_jobs=4,
        verbosity=-1,
    )
    deploy_anchor_mdl.fit(X_unb, y_unb)
    anchor_deploy_lb = deploy_anchor_mdl.predict(X_lb)

    per_seed_lb = np.zeros((len(BAG_SEEDS), len(y_lb)), dtype=np.float64)
    for si, s in enumerate(BAG_SEEDS):
        mdl = lgb.LGBMRegressor(**_lgbm_resid_params(s))
        mdl.fit(X_unb_K, residual_unb)
        per_seed_lb[si] = anchor_deploy_lb + mdl.predict(X_lb_K)
    mean_bag_lb = per_seed_lb.mean(axis=0)
    rae_outer_mean = float(rae(y_lb, mean_bag_lb))
    rae_outer_median = float(rae(y_lb, np.median(per_seed_lb, axis=0)))

    return {
        "style": "nb2189",
        "K": int(K),
        "in_RAE_mean": rae_in_mean,
        "in_RAE_median": rae_in_median,
        "outer_RAE_mean": rae_outer_mean,
        "outer_RAE_median": rae_outer_median,
        "transfer_gap_mean": rae_outer_mean - rae_in_mean,
        "transfer_gap_median": rae_outer_median - rae_in_median,
        "anchor_oof_in_RAE": float(rae(y_unb, anchor_oof_unb)),
        "anchor_deploy_outer_RAE": float(rae(y_lb, anchor_deploy_lb)),
        "external_anchor_in_RAE": float(rae(y_unb, anchor_unb)),
        "external_anchor_outer_RAE": float(rae(y_lb, anchor_lb)),
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Train-holdout LB transfer simulation")
    print(f"           n_splits={N_SPLITS_SCAFFOLD}  pairs={PAIRS}  "
          f"K2103={K_NB2103}  K2189={K_NB2189}")
    print("=" * 78)

    # ------------------------------------------------------------------
    # Load training set; standardize; build scaffolds.
    # ------------------------------------------------------------------
    train = load_train()
    print(f"[load] train shape = {train.shape}")
    # Keep only rows with valid pec50 and smiles
    train = train.dropna(subset=["pec50", "smiles"]).reset_index(drop=True)
    print(f"[load] after dropna pec50/smiles = {train.shape}")

    print("[scaffold] computing Bemis-Murcko scaffolds...")
    train["scaffold"] = train["smiles"].apply(bemis_murcko)

    # Featurize combined (Morgan + RDKit) on whole train then slice per pair
    print("[feat] computing combined features (Morgan + RDKit) on full train...")
    t_feat = time.time()
    X_all = combined(train["smiles"].tolist())
    X_all = impute(X_all).astype(np.float32)
    y_all = train["pec50"].to_numpy(dtype=np.float64)
    print(f"[feat] X_all shape = {X_all.shape}  wall = {time.time()-t_feat:.1f}s")

    # 16-fold scaffold splits
    print(f"[split] building {N_SPLITS_SCAFFOLD}-fold scaffold splits...")
    splits = scaffold_kfold_indices(
        train["scaffold"].tolist(),
        n_splits=N_SPLITS_SCAFFOLD,
        shuffle=True,
        seed=KFOLD_SEED,
    )
    fold_sizes = [len(va) for (_, va) in splits]
    print(f"[split] fold sizes = {fold_sizes}")
    print(f"[split] mean = {np.mean(fold_sizes):.1f}  min/max = "
          f"{min(fold_sizes)}/{max(fold_sizes)}")

    # ------------------------------------------------------------------
    # Run nb2103-style and nb2189-style across PAIRS.
    # ------------------------------------------------------------------
    pair_records = []
    for pair_id, (i_unb, j_lb) in enumerate(PAIRS):
        if i_unb == j_lb:
            continue
        tr_idx_unb, va_idx_unb = splits[i_unb]  # va_idx_unb is the held-out
        tr_idx_lb, va_idx_lb = splits[j_lb]    # va_idx_lb is the OUTER held-out

        sim_unb_idx = va_idx_unb
        sim_lb_idx = va_idx_lb
        # sim_train base: all indices NOT in sim_unb or sim_lb
        all_idx = np.arange(len(y_all))
        used = np.union1d(sim_unb_idx, sim_lb_idx)
        sim_train_idx = np.setdiff1d(all_idx, used)

        print("\n" + "-" * 78)
        print(f"PAIR {pair_id}: sim_unb_fold={i_unb} (n={len(sim_unb_idx)})  "
              f"sim_lb_fold={j_lb} (n={len(sim_lb_idx)})  "
              f"sim_train n={len(sim_train_idx)}")
        print("-" * 78)

        # --- Build chemprop_aux PROXY anchor on sim_train, predict both folds ---
        ts = time.time()
        proxy = lgb.LGBMRegressor(**_lgbm_anchor_params(seed=0))
        proxy.fit(X_all[sim_train_idx], y_all[sim_train_idx])
        anchor_unb = proxy.predict(X_all[sim_unb_idx])
        anchor_lb = proxy.predict(X_all[sim_lb_idx])
        rae_anchor_unb = float(rae(y_all[sim_unb_idx], anchor_unb))
        rae_anchor_lb = float(rae(y_all[sim_lb_idx], anchor_lb))
        print(f"[anchor] chemprop_aux-proxy in_RAE(unb)  = {rae_anchor_unb:.4f}")
        print(f"[anchor] chemprop_aux-proxy outer_RAE(lb)= {rae_anchor_lb:.4f}  "
              f"(wall = {time.time()-ts:.1f}s)")

        # --- Slice X for the two folds (anchor proxy already uses sim_train) ---
        X_unb = X_all[sim_unb_idx]
        X_lb = X_all[sim_lb_idx]
        y_unb = y_all[sim_unb_idx]
        y_lb = y_all[sim_lb_idx]

        # --- nb2103-style ---
        ts = time.time()
        rec_2103 = _eval_nb2103_style(
            X_unb, y_unb, anchor_unb, X_lb, y_lb, anchor_lb, K=K_NB2103
        )
        print(f"[2103]  K={K_NB2103}  in_mean={rec_2103['in_RAE_mean']:.4f}  "
              f"outer_mean={rec_2103['outer_RAE_mean']:.4f}  "
              f"gap_mean={rec_2103['transfer_gap_mean']:+.4f}  "
              f"(wall = {time.time()-ts:.1f}s)")
        print(f"        in_median={rec_2103['in_RAE_median']:.4f}  "
              f"outer_median={rec_2103['outer_RAE_median']:.4f}  "
              f"gap_median={rec_2103['transfer_gap_median']:+.4f}")

        # --- nb2189-style ---
        ts = time.time()
        rec_2189 = _eval_nb2189_style(
            X_unb, y_unb, anchor_unb, X_lb, y_lb, anchor_lb, K=K_NB2189
        )
        print(f"[2189]  K={K_NB2189}  in_mean={rec_2189['in_RAE_mean']:.4f}  "
              f"outer_mean={rec_2189['outer_RAE_mean']:.4f}  "
              f"gap_mean={rec_2189['transfer_gap_mean']:+.4f}  "
              f"(wall = {time.time()-ts:.1f}s)")
        print(f"        in_median={rec_2189['in_RAE_median']:.4f}  "
              f"outer_median={rec_2189['outer_RAE_median']:.4f}  "
              f"gap_median={rec_2189['transfer_gap_median']:+.4f}")

        pair_records.append({
            "pair_id": int(pair_id),
            "sim_unb_fold": int(i_unb),
            "sim_lb_fold": int(j_lb),
            "n_sim_unb": int(len(sim_unb_idx)),
            "n_sim_lb": int(len(sim_lb_idx)),
            "n_sim_train": int(len(sim_train_idx)),
            "anchor_in_RAE_unb": rae_anchor_unb,
            "anchor_outer_RAE_lb": rae_anchor_lb,
            "nb2103": rec_2103,
            "nb2189": rec_2189,
        })

    # ------------------------------------------------------------------
    # Aggregate across pairs.
    # ------------------------------------------------------------------
    gap_mean_2103 = [r["nb2103"]["transfer_gap_mean"] for r in pair_records]
    gap_mean_2189 = [r["nb2189"]["transfer_gap_mean"] for r in pair_records]
    gap_med_2103 = [r["nb2103"]["transfer_gap_median"] for r in pair_records]
    gap_med_2189 = [r["nb2189"]["transfer_gap_median"] for r in pair_records]

    in_mean_2103 = [r["nb2103"]["in_RAE_mean"] for r in pair_records]
    in_mean_2189 = [r["nb2189"]["in_RAE_mean"] for r in pair_records]
    out_mean_2103 = [r["nb2103"]["outer_RAE_mean"] for r in pair_records]
    out_mean_2189 = [r["nb2189"]["outer_RAE_mean"] for r in pair_records]

    avg_gap_2103_mean = float(np.mean(gap_mean_2103))
    avg_gap_2189_mean = float(np.mean(gap_mean_2189))
    avg_gap_diff = avg_gap_2189_mean - avg_gap_2103_mean

    print("\n" + "=" * 78)
    print("TRANSFER-GAP SUMMARY (across pairs)")
    print("=" * 78)
    print(f"  pairs: {[(r['sim_unb_fold'], r['sim_lb_fold']) for r in pair_records]}")
    print()
    print(f"  {'style':>10s}  {'in_mean(avg)':>14s}  {'outer_mean(avg)':>16s}  "
          f"{'gap_mean(avg)':>14s}  {'gap_median(avg)':>16s}")
    print(f"  {'nb2103':>10s}  {np.mean(in_mean_2103):>14.4f}  "
          f"{np.mean(out_mean_2103):>16.4f}  "
          f"{avg_gap_2103_mean:>+14.4f}  {np.mean(gap_med_2103):>+16.4f}")
    print(f"  {'nb2189':>10s}  {np.mean(in_mean_2189):>14.4f}  "
          f"{np.mean(out_mean_2189):>16.4f}  "
          f"{avg_gap_2189_mean:>+14.4f}  {np.mean(gap_med_2189):>+16.4f}")
    print()
    print(f"  AVG GAP DIFF (nb2189 - nb2103, mean-bag) = {avg_gap_diff:+.4f}")
    if avg_gap_diff > 0.02:
        verdict = "POST-unblind cross-fit overfit CONFIRMED (gap diff > +0.02)"
    elif avg_gap_diff > 0.005:
        verdict = "POST-unblind cross-fit overfit suggestive (gap diff > +0.005)"
    elif avg_gap_diff < -0.005:
        verdict = "No POST-unblind overfit; nb2189 transfers BETTER"
    else:
        verdict = "Transfer gaps comparable (diff in noise)"
    print(f"  VERDICT: {verdict}")

    # ------------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------------
    summary = {
        "tag": TAG,
        "method": "train_holdout_transfer_simulation_nb2103_vs_nb2189",
        "n_splits_scaffold": N_SPLITS_SCAFFOLD,
        "pairs": PAIRS,
        "n_folds_resid": N_FOLDS_RESID,
        "bag_seeds": BAG_SEEDS,
        "kfold_seed": KFOLD_SEED,
        "K_nb2103": K_NB2103,
        "K_nb2189": K_NB2189,
        "fold_sizes": fold_sizes,
        "pair_records": pair_records,
        "avg_in_mean_nb2103": float(np.mean(in_mean_2103)),
        "avg_in_mean_nb2189": float(np.mean(in_mean_2189)),
        "avg_outer_mean_nb2103": float(np.mean(out_mean_2103)),
        "avg_outer_mean_nb2189": float(np.mean(out_mean_2189)),
        "avg_gap_mean_nb2103": avg_gap_2103_mean,
        "avg_gap_mean_nb2189": avg_gap_2189_mean,
        "avg_gap_median_nb2103": float(np.mean(gap_med_2103)),
        "avg_gap_median_nb2189": float(np.mean(gap_med_2189)),
        "avg_gap_diff_2189_minus_2103_mean": avg_gap_diff,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] summary -> {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== FINAL ====")
    for k in ("avg_in_mean_nb2103", "avg_outer_mean_nb2103", "avg_gap_mean_nb2103",
              "avg_in_mean_nb2189", "avg_outer_mean_nb2189", "avg_gap_mean_nb2189",
              "avg_gap_diff_2189_minus_2103_mean", "verdict"):
        print(f"  {k}: {res.get(k)}")
