"""nb1130 -- Bag of nb1123 residual-LGBM across 5 KFold seeds.

Hypothesis: nb1123's single-seed (seed=42) honest cross-fit RAE of 0.5704
on the corrected prediction (nb1070 + residual_oof) could be partially
lucky -- a single KFold partition assigns the 50 worst nb1070 errors to
folds where the shallow LGBM happens to recover them. Bagging across
5 KFold seeds {0, 1, 7, 42, 137} gives the TRUE honest cross-fit RAE
distribution and isolates the seed-noise from the structural signal.

Protocol per seed s in {0, 1, 7, 42, 137}:
  1. Use the (already saved) nb1070 median-bag cross-fit OOF as the
     anchor (nb1070_pred_oof.npy) -- this is constant across residual
     seeds (it's already a median bag of 5 nb1070 KFold seeds).
  2. residual = y_unb - nb1070_oof
  3. KFold(n=5, shuffle=True, random_state=s) split of the 253 unblind.
  4. Shallow LGBM Huber (max_depth=3, n_est=80, lr=0.05, min_child_samples=20)
     on combined Morgan+RDKit (2265 features); cross-fit residual OOF.
  5. pred_corrected_s = nb1070_oof + residual_oof_s
  6. Record pooled RAE(y_unb, pred_corrected_s).

Then:
  - mean_bag    = mean over seeds of pred_corrected_s; pooled RAE
  - median_bag  = median over seeds of pred_corrected_s; pooled RAE

A 0.5704 single-seed RAE that holds up under bagging (>=0.575 mean/median)
is real; a 0.5704 that bag-degrades to >0.58 is seed-luck. Compare to
nb1070 baseline (0.5790).

Outputs:
  data/processed/nb1130_per_seed_corrected_oof.npy  (5, 253) float32
  data/processed/nb1130_mean_bag_oof.npy            (253,) float32
  data/processed/nb1130_median_bag_oof.npy          (253,) float32
  data/processed/nb1130_summary.json
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
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED

TAG = "nb1130"
ANCHOR = "nb1070"
STRETCH_ANCHOR = "nb1014"

# Residual model: shallow LGBM Huber (verbatim nb1123 spec).
RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

NB1070_REF_POOLED = 0.5790
NB1123_SINGLE_SEED_REF = 0.5704  # single-seed (42) honest cross-fit RAE


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
    """Honest 5-fold cross-fit shallow LGBM Huber on residual; return OOF."""
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
    print(f"{TAG} -- BAG of nb1123 residual-LGBM across "
          f"{len(RESID_SEEDS)} KFold seeds")
    print(f"          seeds = {RESID_SEEDS}")
    print(f"          residual target = y_unb - nb1070_pred_oof")
    print(f"          features = combined Morgan+RDKit (2265)")
    print(f"          LGBM: max_depth=3, n_est=80, lr=0.05, "
          f"min_child_samples=20, obj=huber(alpha=1.0)")
    print("=" * 78)

    # ---- Load 513 test, unblind index + truth ----
    te = load_test()
    te_smiles = te["smiles"].values
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] y_unb shape = {y_unb.shape}")

    # ---- Load nb1070 cross-fit OOF (median bag of 5 seeds; same anchor) ----
    nb1070_oof_path = DATA_PROCESSED / f"{ANCHOR}_pred_oof.npy"
    if not nb1070_oof_path.exists():
        raise FileNotFoundError(
            f"{nb1070_oof_path} not found; run nb1123 first to regenerate it."
        )
    nb1070_oof = np.load(nb1070_oof_path).astype(np.float64)
    if nb1070_oof.shape[0] != n_unb:
        raise ValueError(
            f"{nb1070_oof_path} shape mismatch: "
            f"{nb1070_oof.shape} vs n_unb={n_unb}"
        )
    rae_anchor = float(rae(y_unb, nb1070_oof))
    print(f"[load] {ANCHOR}_pred_oof.npy shape={nb1070_oof.shape}  "
          f"pooled RAE = {rae_anchor:.4f}  (ref ~{NB1070_REF_POOLED:.4f})")

    # ---- Signed residual (constant across residual seeds) ----
    residual = y_unb - nb1070_oof
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}  "
          f"min={residual.min():+.4f}  max={residual.max():+.4f}")

    # ---- Features (compute once; reused across seeds) ----
    smi_unb = te_smiles[unb_idx].tolist()
    print(f"[feat] computing combined(Morgan+RDKit) on n={len(smi_unb)} "
          f"unblind SMILES")
    X_unb = impute(combined(smi_unb))
    print(f"[feat] X_unb shape = {X_unb.shape}")

    # ---- Cross-fit shallow LGBM Huber on residual, per seed ----
    print("\n" + "-" * 78)
    print(f"PER-SEED RESIDUAL CROSS-FIT")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []
    for i, s in enumerate(RESID_SEEDS):
        resid_oof_s = _residual_cross_fit_one_seed(X_unb, residual, s)
        pred_corr_s = nb1070_oof + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        delta_s = rae_s - rae_anchor
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_nb1070": delta_s,
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
        })
        print(f"   seed {s:3d}:  rae_corr = {rae_s:.4f}  "
              f"(d_vs_nb1070 = {delta_s:+.4f})  "
              f"|resid_oof|.std = {resid_oof_s.std():.3f}")

    # ---- Bag aggregations ----
    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))

    per_seed_rae_arr = np.array(per_seed_rae)
    rae_per_seed_mean = float(per_seed_rae_arr.mean())
    rae_per_seed_median = float(np.median(per_seed_rae_arr))
    rae_per_seed_std = float(per_seed_rae_arr.std())
    rae_per_seed_min = float(per_seed_rae_arr.min())
    rae_per_seed_max = float(per_seed_rae_arr.max())

    print("\n" + "-" * 78)
    print("BAG AGGREGATIONS")
    print("-" * 78)
    print(f"   per-seed RAE list   = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   per-seed mean       = {rae_per_seed_mean:.4f}")
    print(f"   per-seed median     = {rae_per_seed_median:.4f}")
    print(f"   per-seed std        = {rae_per_seed_std:.4f}")
    print(f"   per-seed min/max    = "
          f"{rae_per_seed_min:.4f} / {rae_per_seed_max:.4f}")
    print(f"   pooled RAE(mean_bag)   = {rae_mean_bag:.4f}  "
          f"(d_vs_nb1070 = {rae_mean_bag - rae_anchor:+.4f})")
    print(f"   pooled RAE(median_bag) = {rae_median_bag:.4f}  "
          f"(d_vs_nb1070 = {rae_median_bag - rae_anchor:+.4f})")
    print(f"   nb1123 single-seed ref = {NB1123_SINGLE_SEED_REF:.4f}")

    beats_nb1070 = (
        rae_mean_bag < rae_anchor - 0.003
        or rae_median_bag < rae_anchor - 0.003
    )
    # Did the single-seed 0.5704 hold up?
    holds_up = rae_per_seed_median <= NB1123_SINGLE_SEED_REF + 0.005

    if beats_nb1070 and holds_up:
        verdict = "BAG_CONFIRMS_NB1123_WIN"
    elif beats_nb1070 and not holds_up:
        verdict = "BAG_WIN_BUT_NB1123_SINGLE_SEED_LUCKY"
    elif not beats_nb1070 and holds_up:
        verdict = "BAG_TIES_NB1070_NB1123_NOT_REAL"
    else:
        verdict = "NB1123_SINGLE_SEED_LUCKY_BAG_NO_WIN"
    print(f"   verdict             = {verdict}")

    # ---- Save artefacts ----
    np.save(DATA_PROCESSED / f"{TAG}_per_seed_corrected_oof.npy",
            per_seed_corrected.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy",
            mean_bag_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_median_bag_oof.npy",
            median_bag_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_per_seed_corrected_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_mean_bag_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_median_bag_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "stretch_anchor": STRETCH_ANCHOR,
        "n_unb": n_unb,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "feature_dim": int(X_unb.shape[1]),
        "rae_anchor_nb1070": rae_anchor,
        "per_seed_rae": per_seed_rae,
        "per_seed_records": per_seed_records,
        "rae_per_seed_mean": rae_per_seed_mean,
        "rae_per_seed_median": rae_per_seed_median,
        "rae_per_seed_std": rae_per_seed_std,
        "rae_per_seed_min": rae_per_seed_min,
        "rae_per_seed_max": rae_per_seed_max,
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_nb1070": rae_mean_bag - rae_anchor,
        "delta_median_bag_vs_nb1070": rae_median_bag - rae_anchor,
        "beats_nb1070": bool(beats_nb1070),
        "nb1123_single_seed_holds_up": bool(holds_up),
        "verdict": verdict,
        "nb1070_ref_pooled": NB1070_REF_POOLED,
        "nb1123_single_seed_ref": NB1123_SINGLE_SEED_REF,
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
    for k in ("rae_anchor_nb1070", "per_seed_rae",
              "rae_per_seed_mean", "rae_per_seed_median",
              "rae_per_seed_std",
              "rae_mean_bag", "rae_median_bag",
              "delta_mean_bag_vs_nb1070", "delta_median_bag_vs_nb1070",
              "beats_nb1070", "nb1123_single_seed_holds_up", "verdict"):
        print(f"  {k}: {res.get(k)}")
