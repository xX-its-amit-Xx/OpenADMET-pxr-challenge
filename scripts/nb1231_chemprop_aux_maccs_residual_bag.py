"""nb1231 -- chemprop_aux + MACCS-167 residual mean bag (5 inner seeds).

PRECEDENT
---------
nb1133 verified chemprop_aux is a NON-collapsed PRE-unblind anchor with
in_RAE 0.6216 on the 253 unblind (predicted LB ~0.6246).  When a shallow
LGBM Huber on combined Morgan+RDKit (2265) is run as a residual cross-fit
on top of chemprop_aux, the corrected pooled RAE drops to 0.5879
(delta -0.0337).  nb1183 (MACCS-167 residual on top of nb1070) and nb1200
(outer-bag validation) established MACCS-167 fingerprints as a robust
residual feature family for this dataset (mean-bag pooled RAE 0.5494).

The combination CHEMPROP_AUX anchor + MACCS-167 residual mean-bag is
biologically and statistically orthogonal to nb1211 (= 0.5/0.5 mean of
nb1190 triple-BoB on nb1070 + nb1200 MACCS-BoB on nb1070, 0.5451).
nb1211 uses nb1070 as its anchor everywhere; nb1231 swaps to a different
anchor family entirely (raw MPNN MTL, no post-hoc stretch).

PROTOCOL
--------
  anchor_oof   = te_chemprop_aux.npy[unb_idx]
  residual     = y_unb - anchor_oof
  features     = MACCS-167 cached at data/processed/te_maccs.npy [unb_idx]
  per inner seed s in INNER_SEEDS:
    shallow LGBM Huber (depth=3, leaves=7, n_est=80, lr=0.05,
                        min_child_samples=20, alpha=1.0)
    5-fold KFold cross-fit -> residual_oof_s
    bag_s = anchor_oof + residual_oof_s
  mean_bag = mean over INNER_SEEDS of bag_s.

OUTPUTS
-------
  data/processed/nb1231_per_inner_oof.npy   (5, 253) float32
  data/processed/nb1231_mean_bag_oof.npy    (253,)   float32
  data/processed/nb1231_median_bag_oof.npy  (253,)   float32
  data/processed/nb1231_summary.json
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
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1231"
ANCHOR_LABEL = "chemprop_aux"
ANCHOR_TE_FILE = "te_chemprop_aux.npy"

# Inner-seed family identical to nb1183/nb1200 outer-seed-0 family.
INNER_SEEDS = [0, 1, 7, 42, 137]
RESID_FOLDS = 5

MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"


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


def _residual_cross_fit_one_seed(
    X: np.ndarray, residual: np.ndarray, seed: int
) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- chemprop_aux anchor + MACCS-167 residual mean-bag")
    print(f"     inner seeds = {INNER_SEEDS}   resid folds = {RESID_FOLDS}")
    print(f"     LGBM = huber(alpha=1.0) depth=3 leaves=7 n_est=80 lr=0.05")
    print("=" * 78)

    # ---- Load truth + anchor + features ----
    te = load_test()
    n_test = len(te)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    anchor_te_path = DATA_PROCESSED / ANCHOR_TE_FILE
    if not anchor_te_path.exists():
        raise FileNotFoundError(f"{anchor_te_path} missing")
    te_anchor = np.load(anchor_te_path).astype(np.float64)
    if te_anchor.shape[0] != n_test:
        raise ValueError(
            f"{anchor_te_path} shape {te_anchor.shape} != n_test {n_test}"
        )
    anchor_oof = te_anchor[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_oof))
    residual = y_unb - anchor_oof
    print(f"[load] {ANCHOR_LABEL} anchor pooled RAE on 253 = {rae_anchor:.4f}")
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    if not MACCS_TE_PATH.exists():
        raise FileNotFoundError(f"MACCS test cache missing: {MACCS_TE_PATH}")
    X_maccs_te = np.load(MACCS_TE_PATH)
    if X_maccs_te.shape[0] != n_test:
        raise ValueError(
            f"MACCS shape {X_maccs_te.shape} vs n_test={n_test}"
        )
    X_maccs_unb = X_maccs_te[unb_idx].astype(np.float32)
    print(f"[feat] X_maccs_unb shape = {X_maccs_unb.shape}")

    # ---- Per-inner-seed mean bag ----
    print("\n" + "-" * 78)
    print(f"PER-INNER-SEED CROSS-FIT  (5 seeds x {RESID_FOLDS} folds = {5*RESID_FOLDS} LGBM fits)")
    print("-" * 78)
    per_inner = np.zeros((len(INNER_SEEDS), n_unb), dtype=np.float64)
    per_inner_rae: list[float] = []
    per_inner_records: list[dict] = []
    for j, s in enumerate(INNER_SEEDS):
        t_s = time.time()
        resid_oof = _residual_cross_fit_one_seed(X_maccs_unb, residual, s)
        pred = anchor_oof + resid_oof
        rae_s = float(rae(y_unb, pred))
        per_inner[j] = pred
        per_inner_rae.append(rae_s)
        per_inner_records.append({
            "inner_seed": int(s),
            "rae": rae_s,
            "resid_oof_mean": float(resid_oof.mean()),
            "resid_oof_std": float(resid_oof.std()),
            "elapsed_sec": round(time.time() - t_s, 2),
        })
        print(f"   inner seed {s:5d}  pooled RAE = {rae_s:.4f}   "
              f"(elapsed {time.time() - t_s:.1f}s)")

    mean_bag = per_inner.mean(axis=0)
    median_bag = np.median(per_inner, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag))
    rae_median_bag = float(rae(y_unb, median_bag))

    print("\n" + "=" * 78)
    print(f"   mean   bag pooled RAE = {rae_mean_bag:.4f}")
    print(f"   median bag pooled RAE = {rae_median_bag:.4f}")
    print(f"   anchor (chemprop_aux) = {rae_anchor:.4f}")
    print(f"   delta vs anchor       = {rae_mean_bag - rae_anchor:+.4f}")
    print("=" * 78)

    np.save(DATA_PROCESSED / f"{TAG}_per_inner_oof.npy",
            per_inner.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy",
            mean_bag.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_median_bag_oof.npy",
            median_bag.astype(np.float32))
    print(f"[save] {DATA_PROCESSED / f'{TAG}_per_inner_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_mean_bag_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_median_bag_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR_LABEL,
        "anchor_te_file": ANCHOR_TE_FILE,
        "n_unb": n_unb,
        "inner_seeds": INNER_SEEDS,
        "resid_folds": RESID_FOLDS,
        "feature": "MACCS-167",
        "rae_anchor": rae_anchor,
        "per_inner_records": per_inner_records,
        "per_inner_rae": [float(x) for x in per_inner_rae],
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_anchor": rae_mean_bag - rae_anchor,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("rae_anchor", "per_inner_rae",
              "rae_mean_bag", "rae_median_bag",
              "delta_mean_bag_vs_anchor"):
        print(f"  {k}: {res.get(k)}")
