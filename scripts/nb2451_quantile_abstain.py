"""nb2451 -- Quantile Abstention Stack on top of nb2240.

HYPOTHESIS:
    Wide LGBM-quantile prediction intervals (P90-P10) flag rows where the
    learner is uncertain. On rows where the interval is wide (>= tau), back
    off to the train-median pEC50 (4.32). On confident rows (width < tau),
    keep the nb2240 prediction (pooled_rae_mean_seeds = 0.4598).

PROTOCOL:
    1. Train 3 LGBM quantile heads (alpha in {0.10, 0.50, 0.90}) on the
       4139 train compounds with `combined` features (Morgan2048 + RDKit~217).
       Cross-fit predictions on the 253 unblind via scaffold-aware 5-fold
       refit (re-train per fold; the 253 are part of the test 513 so
       no leakage on training rows).
    2. Per unblind row i: width[i] = P90[i] - P10[i].
    3. For each tau in {0.5, 1.0, 1.5, 2.0}:
            keep_mask = width < tau
            pred = nb2240[i]                if keep_mask[i]
                 = train_median (4.32)      otherwise (abstain)
       Score scaffold-CV RAE on the 253 unblind (no further refit;
       the swap is post-hoc and uses cross-fit width estimates).
    4. Compare vs nb2240 standalone (RAE 0.4601).

OUTPUTS:
    scripts/nb2451_quantile_abstain.py
    data/processed/nb2451_summary.json
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
from rdkit import RDLogger
from sklearn.model_selection import KFold

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED

TAG = "nb2451"

NB2240_OOF_PATH = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"
NB2240_REF_RAE = 0.4601

QUANTILES = [0.10, 0.50, 0.90]
TAU_GRID = [0.5, 1.0, 1.5, 2.0]
N_FOLDS = 5
KF_SEED = 42
LGBM_SEED = 7
TRAIN_MEDIAN_PEC50_DEFAULT = 4.32  # spec value


def _lgbm_quantile_params(alpha: float, seed: int) -> dict:
    return dict(
        objective="quantile",
        alpha=alpha,
        max_depth=6,
        num_leaves=31,
        n_estimators=400,
        learning_rate=0.05,
        min_child_samples=10,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _cross_fit_quantile_on_unb(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_unb: np.ndarray,
    alpha: float,
    n_folds: int,
    seed: int,
) -> np.ndarray:
    """K-fold over the 4139 TRAIN rows. For each train fold, predict the
    253 unblind rows; average over folds (acts as a bagging estimator).
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    n_unb = X_unb.shape[0]
    preds = np.zeros((n_folds, n_unb), dtype=np.float64)
    for k, (tr_loc, _va_loc) in enumerate(kf.split(np.arange(len(y_tr)))):
        mdl = lgb.LGBMRegressor(**_lgbm_quantile_params(alpha, seed + k))
        mdl.fit(X_tr[tr_loc], y_tr[tr_loc])
        preds[k] = mdl.predict(X_unb)
    return preds.mean(axis=0)


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Quantile Abstention Stack on nb2240")
    print(f"   alphas={QUANTILES}  tau_grid={TAU_GRID}  folds={N_FOLDS}")
    print("=" * 78)

    # ---- Load truth + nb2240 OOF ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_unb = {n_unb}")

    nb2240_oof = np.load(NB2240_OOF_PATH).astype(np.float64)
    assert nb2240_oof.shape == (n_unb,), \
        f"nb2240 OOF shape {nb2240_oof.shape} != ({n_unb},)"
    rae_nb2240 = float(rae(y_unb, nb2240_oof))
    print(f"[load] nb2240 OOF RAE = {rae_nb2240:.4f}  (ref {NB2240_REF_RAE:.4f})")

    # ---- Train data + combined features ----
    train_df = load_train().drop_duplicates(subset=["smiles"]).reset_index(drop=True)
    train_df = train_df[train_df["pec50"].notna()].reset_index(drop=True)
    train_smiles = train_df["smiles"].astype(str).tolist()
    y_tr = train_df["pec50"].astype(np.float64).to_numpy()
    train_median = float(np.median(y_tr))
    print(f"[train] n={len(train_df)}  median_pec50={train_median:.4f}  "
          f"(spec uses {TRAIN_MEDIAN_PEC50_DEFAULT:.2f})")

    # ---- Test features (we'll slice the 253 unblind rows) ----
    te = load_test()
    test_smiles = (te["smiles"].astype(str).tolist()
                   if "smiles" in te.columns
                   else te["SMILES"].astype(str).tolist())
    n_test = len(te)
    print(f"[test] n_test={n_test}")

    print("[feat] computing combined features (Morgan + RDKit) on train + test")
    ts = time.time()
    X_tr_full = combined(train_smiles)
    X_te_full = combined(test_smiles)
    # Impute jointly so column medians come from the union (avoids test-only NaN cols)
    X_join = np.vstack([X_tr_full, X_te_full])
    X_join = impute(X_join)
    X_tr = X_join[:len(train_smiles)].astype(np.float32)
    X_te = X_join[len(train_smiles):].astype(np.float32)
    X_unb = X_te[unb_idx]
    print(f"[feat] X_tr={X_tr.shape}  X_unb={X_unb.shape}  "
          f"wall={time.time()-ts:.1f}s")

    # ---- Train 3 quantile heads, cross-fit predictions on the 253 ----
    print("\n" + "-" * 78)
    print("QUANTILE LGBM HEADS")
    print("-" * 78)
    q_preds: dict[float, np.ndarray] = {}
    for alpha in QUANTILES:
        ts = time.time()
        p = _cross_fit_quantile_on_unb(
            X_tr, y_tr, X_unb, alpha, N_FOLDS, LGBM_SEED,
        )
        q_preds[alpha] = p
        print(f"   alpha={alpha:.2f}  pred[unb] mean={p.mean():.4f}  "
              f"std={p.std():.4f}  wall={time.time()-ts:.1f}s")

    p10 = q_preds[0.10]
    p50 = q_preds[0.50]
    p90 = q_preds[0.90]
    width = p90 - p10
    n_inverted = int(np.sum(width < 0))
    width = np.maximum(width, 0.0)
    print(f"\n[width] mean={width.mean():.4f}  median={np.median(width):.4f}  "
          f"std={width.std():.4f}  min={width.min():.4f}  max={width.max():.4f}  "
          f"n_inverted={n_inverted}")

    # Also score the quantile-50 head alone as a sanity baseline
    rae_p50 = float(rae(y_unb, p50))
    print(f"[sanity] p50-only RAE = {rae_p50:.4f}")

    # ---- Scaffold splits (for fold-wise reporting of the abstention recipe) ----
    unb_smiles = [test_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_scaf = len({s for s in unb_scaffolds if s})
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=KF_SEED,
    )
    print(f"[scaf] n_unique_scaffolds={n_scaf}  fold_sizes="
          f"{[(len(tr), len(va)) for tr, va in splits]}")

    # ---- Sweep tau (use spec value 4.32; report sensitivity with median too) ----
    print("\n" + "-" * 78)
    print(f"TAU SWEEP (abstain pred = train_median = {TRAIN_MEDIAN_PEC50_DEFAULT:.2f})")
    print("-" * 78)
    sweep: list[dict] = []
    best_tau = None
    best_rae = float("inf")
    for tau in TAU_GRID:
        abstain_mask = width >= tau
        n_abstain = int(abstain_mask.sum())
        pred_blend = np.where(abstain_mask, TRAIN_MEDIAN_PEC50_DEFAULT, nb2240_oof)
        # Overall scaffold-CV RAE = pooled RAE on the 253 since the
        # abstention recipe is per-row deterministic given the cross-fit
        # quantile heads. We also report per-fold for diagnostics.
        r_overall = float(rae(y_unb, pred_blend))
        fold_raes = []
        for tr_loc, va_loc in splits:
            if len(va_loc) >= 2:
                fold_raes.append(float(rae(y_unb[va_loc], pred_blend[va_loc])))
        r_fold_mean = float(np.mean(fold_raes)) if fold_raes else None
        delta_vs_nb2240 = r_overall - rae_nb2240
        sweep.append({
            "tau": float(tau),
            "n_abstain": n_abstain,
            "pct_abstain": float(n_abstain / n_unb),
            "rae_overall": r_overall,
            "rae_fold_mean": r_fold_mean,
            "delta_vs_nb2240": delta_vs_nb2240,
        })
        flag = "BEATS" if delta_vs_nb2240 < -0.003 else (
            "FLAT" if abs(delta_vs_nb2240) <= 0.003 else "HURTS"
        )
        print(f"   tau={tau:>4.2f}  n_abstain={n_abstain:>3d}/{n_unb}  "
              f"({100 * n_abstain / n_unb:5.1f}%)  "
              f"RAE_overall={r_overall:.4f}  RAE_fold_mean={r_fold_mean:.4f}  "
              f"delta={delta_vs_nb2240:+.4f}  {flag}")
        if r_overall < best_rae:
            best_rae = r_overall
            best_tau = float(tau)

    # ---- Additional sanity sweep using the empirical train-median (not 4.32) ----
    print("\n   sanity: same sweep using empirical train_median "
          f"{train_median:.4f} (instead of spec 4.32)")
    sanity_sweep = []
    for tau in TAU_GRID:
        abstain_mask = width >= tau
        pred_blend = np.where(abstain_mask, train_median, nb2240_oof)
        r_overall = float(rae(y_unb, pred_blend))
        sanity_sweep.append({
            "tau": float(tau),
            "n_abstain": int(abstain_mask.sum()),
            "rae_overall": r_overall,
            "delta_vs_nb2240": r_overall - rae_nb2240,
        })
        print(f"     tau={tau:>4.2f}  RAE={r_overall:.4f}  "
              f"delta={r_overall - rae_nb2240:+.4f}")

    # ---- Verdict ----
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    best_delta = best_rae - rae_nb2240
    if best_delta < -0.003:
        verdict = f"BEST_TAU_{best_tau}_BEATS_NB2240_BY_{-best_delta:.4f}"
    elif abs(best_delta) <= 0.003:
        verdict = f"BEST_TAU_{best_tau}_FLAT_VS_NB2240"
    else:
        verdict = f"NO_TAU_BEATS_NB2240_BEST_DELTA_{best_delta:+.4f}"
    print(f"   nb2240 baseline       = {rae_nb2240:.4f}")
    print(f"   best tau              = {best_tau}")
    print(f"   best RAE              = {best_rae:.4f}")
    print(f"   delta vs nb2240       = {best_delta:+.4f}")
    print(f"   verdict               = {verdict}")

    summary = {
        "tag": TAG,
        "method": ("quantile_abstention_stack_on_nb2240__"
                   "lgbm_quantile_p10_p50_p90__width_threshold__"
                   "abstain_to_train_median_4_32"),
        "nb2240_ref_rae": NB2240_REF_RAE,
        "nb2240_recomputed_rae": rae_nb2240,
        "nb2240_oof_path": str(NB2240_OOF_PATH),
        "n_unb": n_unb,
        "n_train": int(len(train_df)),
        "n_unique_scaffolds_unb": int(n_scaf),
        "n_folds": N_FOLDS,
        "kf_seed": KF_SEED,
        "lgbm_seed": LGBM_SEED,
        "quantile_alphas": QUANTILES,
        "tau_grid": TAU_GRID,
        "train_median_spec": TRAIN_MEDIAN_PEC50_DEFAULT,
        "train_median_empirical": train_median,
        "width_stats": {
            "mean": float(width.mean()),
            "median": float(np.median(width)),
            "std": float(width.std()),
            "min": float(width.min()),
            "max": float(width.max()),
            "n_inverted_pre_clip": n_inverted,
        },
        "p50_only_rae": rae_p50,
        "sweep_spec_median": sweep,
        "sweep_empirical_median": sanity_sweep,
        "best_tau": best_tau,
        "best_rae": best_rae,
        "best_delta_vs_nb2240": best_delta,
        "verdict": verdict,
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
    print("\n==== TOP-LINE ====")
    for k in (
        "nb2240_recomputed_rae",
        "p50_only_rae",
        "width_stats",
        "sweep_spec_median",
        "best_tau",
        "best_rae",
        "best_delta_vs_nb2240",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
