"""nb1210 -- DEPLOY artifact for nb1200 (outer-bag MACCS-167 residual bag-of-bags)
on the 513-row test set.

PRECEDENT
---------
nb1200 (diagnostic) outer-bag stress test on the 253 unblind:
  per-outer mean-bag RAE [outer 0,1,7,42,137]
  bag-of-bags MEAN   row-level RAE = 0.5495
  bag-of-bags MEDIAN row-level RAE = 0.5491
These are the honest cross-fit LB anchors for this method.

PROTOCOL
--------
For each OUTER seed o in {0, 1, 7, 42, 137}:
  inner_seeds(o) = [o * 1000 + s for s in {0, 1, 7, 42, 137}]
  DEPLOY step: each of those 5 inner shallow LGBM Huber learners is fit on ALL
  253 unblind rows (no CV) with residual = y_unb - nb1070_pred_oof on MACCS-167
  features, then predicts on all 513 test rows. Mean-bag inner across the 5
  inner seeds -> one per-outer 513-residual vector.

Stack 5 per-outer deploy residuals -> shape (5, 513).
Row-level bag-of-bags MEAN across outer seeds -> te_residual_513.
te_nb1210 = te_nb1070 + te_residual_513.

Also compute MEDIAN row-level bag-of-bags across outer seeds and save the
matching MEDIAN deploy.

Outputs:
  data/processed/te_nb1210.npy           (513,) float32  MEAN bag-of-bags deploy
  data/processed/te_nb1210_median.npy    (513,) float32  MEDIAN bag-of-bags deploy
  submissions/nb1210_deploy_nb1200_mean.csv     (513 rows)
  submissions/nb1210_deploy_nb1200_median.csv   (513 rows)
  data/processed/nb1210_summary.json

NOTE
----
Per feedback_lb_two_regime_calibration: this is a POST-unblind deploy, so the
in_RAE on te[unb_idx] is in-sample optimistic. LB-faithful anchors are the
nb1200 honest cross-fit RAEs above.
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
from lightgbm import LGBMRegressor

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1210"
ANCHOR = "nb1070"

# Outer/inner seed grid (mirrors nb1200 exactly).
OUTER_SEEDS = [0, 1, 7, 42, 137]
INNER_BASES = [0, 1, 7, 42, 137]   # inner = outer * 1000 + base

# MACCS cached test features.
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"  # (513, 167) uint8

# Honest cross-fit LB anchors (from nb1200 summary).
NB1200_BOB_MEAN_LB_ANCHOR = 0.5495
NB1200_BOB_MEDIAN_LB_ANCHOR = 0.5491

SUBMISSIONS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "submissions")
)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Shared LGBM Huber config (identical capacity to nb1183/nb1200).
# -----------------------------------------------------------------------------
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


def _load_maccs_test(n_test_expected: int) -> np.ndarray:
    if not MACCS_TE_PATH.exists():
        raise FileNotFoundError(f"MACCS test cache missing: {MACCS_TE_PATH}")
    X = np.load(MACCS_TE_PATH)
    if X.shape[0] != n_test_expected:
        raise ValueError(
            f"MACCS test cache shape mismatch: {X.shape} vs n_test={n_test_expected}"
        )
    if X.shape[1] not in (166, 167):
        raise ValueError(
            f"MACCS test cache unexpected width: {X.shape[1]}"
        )
    return X.astype(np.float32)


def _fit_one_predict_513(
    X_unb: np.ndarray, residual: np.ndarray,
    X_test: np.ndarray, seed: int
) -> np.ndarray:
    mdl = LGBMRegressor(**_lgbm_params(seed))
    mdl.fit(X_unb, residual)
    return mdl.predict(X_test).astype(np.float64)


def _mean_bag_residual_513(
    X_unb: np.ndarray, residual: np.ndarray,
    X_test: np.ndarray, seeds: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    n_test = X_test.shape[0]
    per_seed = np.zeros((len(seeds), n_test), dtype=np.float64)
    for j, s in enumerate(seeds):
        per_seed[j] = _fit_one_predict_513(X_unb, residual, X_test, s)
    return per_seed.mean(axis=0), per_seed


def _save_submission_csv(
    te_pred: np.ndarray, te_smiles, te_names, csv_path: str, label: str
) -> dict:
    assert te_pred.shape[0] == 513, (
        f"{label}: te_pred shape {te_pred.shape}, expected (513,)"
    )
    assert np.all(np.isfinite(te_pred)), f"{label}: te_pred has NaN/Inf"
    sub = pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": te_pred.astype(np.float64),
    })
    assert len(sub) == 513, f"{label}: row count {len(sub)} != 513"
    assert list(sub.columns) == ["SMILES", "Molecule Name", "pEC50"], (
        f"{label}: column order wrong: {list(sub.columns)}"
    )
    assert sub.isna().sum().sum() == 0, f"{label}: CSV has NaN"
    sub.to_csv(csv_path, index=False)
    return {
        "csv_path": csv_path,
        "n_rows": int(len(sub)),
        "columns": list(sub.columns),
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- DEPLOY nb1200 outer-bag MACCS-167 residual bag-of-bags "
          f"(MEAN + MEDIAN)")
    print(f"          anchor       = {ANCHOR} (te_{ANCHOR}.npy + {ANCHOR}_pred_oof.npy)")
    print(f"          outer seeds  = {OUTER_SEEDS}")
    print(f"          inner bases  = {INNER_BASES}  (inner = outer*1000 + base)")
    print(f"          feature      = MACCS-167 cached ({MACCS_TE_PATH})")
    print(f"          LGBM:  depth=3, num_leaves=7, n_est=80, lr=0.05, "
          f"min_child=20, obj=huber(alpha=1.0)")
    print(f"          honest cross-fit LB anchors: "
          f"BoB MEAN={NB1200_BOB_MEAN_LB_ANCHOR:.4f}  "
          f"BoB MEDIAN={NB1200_BOB_MEDIAN_LB_ANCHOR:.4f}")
    print("=" * 78)

    # ---- Load 513 test, unblind index + truth, anchors ----
    te = load_test()
    te_smiles = te["smiles"].values
    te_names = te["name"].values
    n_test = len(te_smiles)

    te_nb1070 = np.load(DATA_PROCESSED / f"te_{ANCHOR}.npy").astype(np.float64)
    nb1070_oof = np.load(DATA_PROCESSED / f"{ANCHOR}_pred_oof.npy").astype(np.float64)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    assert te_nb1070.shape[0] == n_test
    assert nb1070_oof.shape[0] == n_unb

    rae_anchor_oof = float(rae(y_unb, nb1070_oof))
    rae_anchor_te_in = float(rae(y_unb, te_nb1070[unb_idx]))
    print(f"[load] te_{ANCHOR}.npy shape={te_nb1070.shape}  "
          f"in_RAE(unb_idx) = {rae_anchor_te_in:.4f}")
    print(f"[load] {ANCHOR}_pred_oof.npy shape={nb1070_oof.shape}  "
          f"pooled RAE = {rae_anchor_oof:.4f}")

    # Residual target (constant across seeds).
    residual_target = y_unb - nb1070_oof
    print(f"[resid] target mean={residual_target.mean():+.4f}  "
          f"std={residual_target.std():.4f}  "
          f"min={residual_target.min():+.4f}  max={residual_target.max():+.4f}")

    # ---- Features ----
    X_maccs_test = _load_maccs_test(n_test)
    X_maccs_unb = X_maccs_test[unb_idx]
    print(f"[feat] X_maccs_test shape = {X_maccs_test.shape}")
    print(f"[feat] X_maccs_unb  shape = {X_maccs_unb.shape}  "
          f"bit_density={X_maccs_unb.mean():.4f}  "
          f"const_cols={int((X_maccs_unb.var(axis=0) == 0).sum())}/{X_maccs_unb.shape[1]}")

    # ---- Per-outer deploy residual rebuild ----
    print("\n" + "-" * 78)
    print(f"PER-OUTER DEPLOY REBUILD  (each: 5 inner seeds x 1 fit on all 253 -> "
          f"predict 513)")
    print("-" * 78)
    per_outer_residual_513 = np.zeros((len(OUTER_SEEDS), n_test), dtype=np.float64)
    per_outer_records: list[dict] = []

    for oi, o in enumerate(OUTER_SEEDS):
        t_outer = time.time()
        inner_seeds = [int(o) * 1000 + int(b) for b in INNER_BASES]

        mean_resid_513_o, per_seed_resid_513_o = _mean_bag_residual_513(
            X_maccs_unb, residual_target, X_maccs_test, inner_seeds
        )
        per_outer_residual_513[oi] = mean_resid_513_o

        per_seed_in_rae = []
        for j, s in enumerate(inner_seeds):
            te_seed = te_nb1070 + per_seed_resid_513_o[j]
            per_seed_in_rae.append(float(rae(y_unb, te_seed[unb_idx])))
        te_outer = te_nb1070 + mean_resid_513_o
        in_rae_outer = float(rae(y_unb, te_outer[unb_idx]))

        per_outer_records.append({
            "outer_seed": int(o),
            "inner_seeds": inner_seeds,
            "per_inner_in_rae_te": per_seed_in_rae,
            "in_rae_te_mean_bag": in_rae_outer,
            "resid_mean": float(mean_resid_513_o.mean()),
            "resid_std": float(mean_resid_513_o.std()),
            "resid_min": float(mean_resid_513_o.min()),
            "resid_max": float(mean_resid_513_o.max()),
            "elapsed_sec": round(time.time() - t_outer, 1),
        })
        print(f"   outer {o:5d}  inner={inner_seeds}")
        print(f"     per_inner_in_rae(te) = "
              f"[{', '.join(f'{r:.4f}' for r in per_seed_in_rae)}]")
        print(f"     mean_bag in_rae(te)  = {in_rae_outer:.4f}  "
              f"resid_513 mean={mean_resid_513_o.mean():+.4f} "
              f"std={mean_resid_513_o.std():.4f}  "
              f"(elapsed {time.time() - t_outer:.1f}s)")

    # ---- Bag-of-bags aggregation across outer seeds (row-level) ----
    te_residual_513_mean = per_outer_residual_513.mean(axis=0)
    te_residual_513_median = np.median(per_outer_residual_513, axis=0)

    te_nb1210 = te_nb1070 + te_residual_513_mean
    te_nb1210_median = te_nb1070 + te_residual_513_median

    in_rae_mean = float(rae(y_unb, te_nb1210[unb_idx]))
    in_rae_median = float(rae(y_unb, te_nb1210_median[unb_idx]))

    print("\n" + "=" * 78)
    print("BAG-OF-BAGS DEPLOY  (row-level across 5 outer seeds)")
    print("=" * 78)
    print(f"   te_residual_513 MEAN   mean={te_residual_513_mean.mean():+.4f}  "
          f"std={te_residual_513_mean.std():.4f}  "
          f"min={te_residual_513_mean.min():+.4f}  "
          f"max={te_residual_513_mean.max():+.4f}")
    print(f"   te_residual_513 MEDIAN mean={te_residual_513_median.mean():+.4f}  "
          f"std={te_residual_513_median.std():.4f}  "
          f"min={te_residual_513_median.min():+.4f}  "
          f"max={te_residual_513_median.max():+.4f}")
    print(f"   te_nb1210 MEAN   mean={te_nb1210.mean():.3f}  "
          f"std={te_nb1210.std():.3f}  min={te_nb1210.min():.3f}  "
          f"max={te_nb1210.max():.3f}  in_RAE(unb)={in_rae_mean:.4f}  "
          f"(honest LB anchor {NB1200_BOB_MEAN_LB_ANCHOR:.4f})")
    print(f"   te_nb1210 MEDIAN mean={te_nb1210_median.mean():.3f}  "
          f"std={te_nb1210_median.std():.3f}  min={te_nb1210_median.min():.3f}  "
          f"max={te_nb1210_median.max():.3f}  in_RAE(unb)={in_rae_median:.4f}  "
          f"(honest LB anchor {NB1200_BOB_MEDIAN_LB_ANCHOR:.4f})")

    # ---- Save artefacts ----
    te_mean_path = DATA_PROCESSED / "te_nb1210.npy"
    te_median_path = DATA_PROCESSED / "te_nb1210_median.npy"
    np.save(te_mean_path, te_nb1210.astype(np.float32))
    np.save(te_median_path, te_nb1210_median.astype(np.float32))
    print(f"[save] {te_mean_path}")
    print(f"[save] {te_median_path}")

    csv_mean_path = os.path.join(SUBMISSIONS_DIR, "nb1210_deploy_nb1200_mean.csv")
    csv_median_path = os.path.join(SUBMISSIONS_DIR, "nb1210_deploy_nb1200_median.csv")
    csv_mean_info = _save_submission_csv(
        te_nb1210, te_smiles, te_names, csv_mean_path, "nb1210_mean"
    )
    csv_median_info = _save_submission_csv(
        te_nb1210_median, te_smiles, te_names, csv_median_path, "nb1210_median"
    )
    print(f"[save] {csv_mean_path}  rows={csv_mean_info['n_rows']}")
    print(f"[save] {csv_median_path}  rows={csv_median_info['n_rows']}")

    # ---- Summary ----
    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "feature_source": "maccs_cached_167",
        "maccs_cache_test": str(MACCS_TE_PATH),
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "outer_seeds": OUTER_SEEDS,
        "inner_bases": INNER_BASES,
        "feature_dim": int(X_maccs_unb.shape[1]),
        "lgbm_params_template": _lgbm_params(0),
        "rae_anchor_oof_253": rae_anchor_oof,
        "rae_anchor_te_in_sample_253": rae_anchor_te_in,
        "residual_target_mean": float(residual_target.mean()),
        "residual_target_std": float(residual_target.std()),
        "per_outer_records": per_outer_records,
        "deploys": {
            "nb1210_mean": {
                "te_path": str(te_mean_path),
                "csv_path": csv_mean_path,
                "te_mean": float(te_nb1210.mean()),
                "te_std": float(te_nb1210.std()),
                "te_min": float(te_nb1210.min()),
                "te_max": float(te_nb1210.max()),
                "in_rae_253": in_rae_mean,
                "crossfit_lb_anchor": NB1200_BOB_MEAN_LB_ANCHOR,
                "aggregation": "bag_of_bags_row_level_mean_across_outer_seeds",
            },
            "nb1210_median": {
                "te_path": str(te_median_path),
                "csv_path": csv_median_path,
                "te_mean": float(te_nb1210_median.mean()),
                "te_std": float(te_nb1210_median.std()),
                "te_min": float(te_nb1210_median.min()),
                "te_max": float(te_nb1210_median.max()),
                "in_rae_253": in_rae_median,
                "crossfit_lb_anchor": NB1200_BOB_MEDIAN_LB_ANCHOR,
                "aggregation": "bag_of_bags_row_level_median_across_outer_seeds",
            },
        },
        "wall_sec": round(time.time() - t0, 2),
        "note": (
            "POST-unblind deploy artifacts: each inner LGBM is fit on ALL 253 "
            "unblind rows so in_RAE on te[unb_idx] is in-sample and optimistic. "
            "LB-faithful numbers are the nb1200 honest cross-fit BoB anchors."
        ),
    }
    summary_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {summary_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== STRUCTURED SUMMARY ====")
    for tag in ("nb1210_mean", "nb1210_median"):
        d = res["deploys"][tag]
        print(f"  {tag}:")
        for k in ("te_mean", "te_std", "te_min", "te_max",
                  "in_rae_253", "crossfit_lb_anchor",
                  "te_path", "csv_path"):
            print(f"    {k}: {d.get(k)}")
