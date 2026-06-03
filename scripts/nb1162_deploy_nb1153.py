"""nb1162 -- DEPLOY artifact for nb1153 (mean-bag of 5 residual-LGBM seeds,
              MORDRED-only features on top of nb1070 anchor).

This is the 513-row deploy companion to nb1153.  Where nb1153 is the
honest cross-fit diagnostic on the 253 unblind, this script:

Pipeline:
  For EACH seed s in {0, 1, 7, 42, 137}:
    1. Fit shallow LGBM Huber (max_depth=3, num_leaves=7, n_est=80,
       lr=0.05, min_child_samples=20, alpha=1.0, random_state=s) on
       ALL 253 unblind rows with target = y_unb - nb1070_pred_oof,
       using CACHED MORDRED (1533) features sliced to unblind rows.
    2. residual_pred_513_s = mdl.predict(X_mordred_test_513).
  Then:
    3. mean_bagged_residual_513 = mean over the 5 seeds.
    4. te_nb1162 = te_nb1070 + mean_bagged_residual_513.
    5. Write submissions/nb1162_deploy_nb1153.csv (SMILES, name, pEC50)
       and data/processed/te_nb1162.npy.

Caveat (per feedback_lb_two_regime_calibration): this is a POST-unblind
deploy artifact -- in_RAE on te_nb1162[unb_idx] is in-sample and
optimistic. The LB-faithful number is the nb1153 honest cross-fit RAE.
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

TAG = "nb1162"
ANCHOR = "nb1070"
RESID_SEEDS = [0, 1, 7, 42, 137]

MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")


def _lgbm_params(seed: int) -> dict:
    """Shallow LGBM Huber -- identical capacity to nb1153 cross-fit bag."""
    return dict(
        objective="huber",
        alpha=1.0,
        learning_rate=0.05,
        n_estimators=80,
        max_depth=3,
        num_leaves=7,            # 2^3 - 1
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        verbosity=-1,
        random_state=seed,
        n_jobs=2,
    )


SUBMISSIONS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "submissions")
)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)


def _load_mordred_test(n_test_expected: int) -> np.ndarray:
    """Load cached Mordred test matrix (513 x 1533); NaN/Inf median-impute."""
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(
            f"Mordred cache missing -- run nb1030 first ({mte_p})"
        )
    X_te_m = np.load(mte_p).astype(np.float32)
    if X_te_m.shape[0] != n_test_expected:
        raise ValueError(
            f"Mordred test shape mismatch: {X_te_m.shape} vs "
            f"n_test={n_test_expected}"
        )
    X_te_m = np.where(np.isfinite(X_te_m), X_te_m, np.nan).astype(np.float32)
    col_med = np.nanmedian(X_te_m, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X_te_m)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X_te_m[idx_r, idx_c] = col_med[idx_c]
    return X_te_m


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- DEPLOY mean-bag residual: 5 shallow LGBM Huber seeds")
    print(f"          seeds = {RESID_SEEDS}")
    print(f"          anchor = {ANCHOR} (te_{ANCHOR}.npy + {ANCHOR}_pred_oof.npy)")
    print(f"          features = cached MORDRED (1533) from {MORDRED_DIR}")
    print(f"          LGBM: max_depth=3, num_leaves=7, n_est=80, lr=0.05, "
          f"min_child_samples=20, obj=huber(alpha=1.0)")
    print("=" * 78)

    # ---- Load 513 test, unblind index + truth, anchor deploy/OOF preds ----
    te = load_test()
    te_smiles = te["smiles"].values
    te_names = te["name"].values
    n_test = len(te_smiles)

    te_nb1070_path = DATA_PROCESSED / f"te_{ANCHOR}.npy"
    nb1070_oof_path = DATA_PROCESSED / f"{ANCHOR}_pred_oof.npy"
    unb_idx_path = DATA_PROCESSED / "_audit_unblind_idx.npy"
    y_unb_path = DATA_PROCESSED / "_audit_unblind_y.npy"

    te_nb1070 = np.load(te_nb1070_path).astype(np.float64)
    nb1070_oof = np.load(nb1070_oof_path).astype(np.float64)
    unb_idx = np.load(unb_idx_path)
    y_unb = np.load(y_unb_path).astype(np.float64)
    n_unb = len(y_unb)

    assert te_nb1070.shape[0] == n_test, (
        f"te_{ANCHOR} shape {te_nb1070.shape} mismatch n_test={n_test}"
    )
    assert nb1070_oof.shape[0] == n_unb, (
        f"{ANCHOR}_pred_oof shape {nb1070_oof.shape} mismatch n_unb={n_unb}"
    )

    rae_anchor_oof = float(rae(y_unb, nb1070_oof))
    rae_anchor_te_in = float(rae(y_unb, te_nb1070[unb_idx]))
    print(f"[load] te_{ANCHOR}.npy shape={te_nb1070.shape}  "
          f"in_RAE(unb_idx)={rae_anchor_te_in:.4f}")
    print(f"[load] {ANCHOR}_pred_oof.npy shape={nb1070_oof.shape}  "
          f"pooled RAE={rae_anchor_oof:.4f}")

    # ---- Signed residual target (constant across seeds) ----
    residual_target = y_unb - nb1070_oof
    print(f"[resid] mean={residual_target.mean():+.4f}  "
          f"std={residual_target.std():.4f}  "
          f"min={residual_target.min():+.4f}  "
          f"max={residual_target.max():+.4f}")

    # ---- Featurize: cached Mordred for 513 test, slice to 253 for fit ----
    print(f"[feat] loading cached Mordred test matrix (513 x 1533) from "
          f"{MORDRED_DIR}")
    X_test = _load_mordred_test(n_test_expected=n_test)
    print(f"[feat] X_test shape = {X_test.shape}")
    X_unb = X_test[unb_idx]
    print(f"[feat] X_unb  shape = {X_unb.shape}  (sliced to unblind rows)")

    # ---- Fit deploy LGBM per seed; collect residual_pred_513 ----
    print("\n" + "-" * 78)
    print(f"PER-SEED DEPLOY FIT  (all n={n_unb} unblind rows)")
    print("-" * 78)
    per_seed_residual_513 = np.zeros(
        (len(RESID_SEEDS), n_test), dtype=np.float64
    )
    per_seed_records = []
    for i, s in enumerate(RESID_SEEDS):
        mdl = LGBMRegressor(**_lgbm_params(s))
        mdl.fit(X_unb, residual_target)
        residual_in_s = mdl.predict(X_unb)
        residual_pred_513_s = mdl.predict(X_test).astype(np.float64)
        per_seed_residual_513[i] = residual_pred_513_s
        # in-sample corrected RAE on the OOF anchor (matches nb1140 convention)
        in_pred_corr_253_s = nb1070_oof + residual_in_s
        in_rae_corr_oof_s = float(rae(y_unb, in_pred_corr_253_s))
        # in-sample corrected RAE on the deploy anchor (matches deploy vector)
        te_seed_s = te_nb1070 + residual_pred_513_s
        in_rae_te_s = float(rae(y_unb, te_seed_s[unb_idx]))
        per_seed_records.append({
            "seed": int(s),
            "residual_pred_513_mean": float(residual_pred_513_s.mean()),
            "residual_pred_513_std": float(residual_pred_513_s.std()),
            "in_rae_253_corr_on_oof_anchor": in_rae_corr_oof_s,
            "in_rae_253_te_anchor": in_rae_te_s,
        })
        print(f"   seed {s:3d}:  resid_513 mean={residual_pred_513_s.mean():+.4f} "
              f" std={residual_pred_513_s.std():.4f}  "
              f"in_RAE_corr(OOF)={in_rae_corr_oof_s:.4f}  "
              f"in_RAE(te)={in_rae_te_s:.4f}")

    # ---- Mean-bag the 5 deploy residuals ----
    te_residual_513 = per_seed_residual_513.mean(axis=0)
    print(f"\n[bag] te_residual_513: shape={te_residual_513.shape}  "
          f"mean={te_residual_513.mean():+.4f}  "
          f"std={te_residual_513.std():.4f}  "
          f"min={te_residual_513.min():+.4f}  "
          f"max={te_residual_513.max():+.4f}")

    # ---- te_nb1162 = te_nb1070 + te_residual_513 ----
    te_nb1162 = te_nb1070 + te_residual_513
    print(f"[deploy] te_nb1162 shape={te_nb1162.shape}  "
          f"mean={te_nb1162.mean():.3f}  std={te_nb1162.std():.3f}  "
          f"min={te_nb1162.min():.3f}  max={te_nb1162.max():.3f}")

    # ---- In-sample RAE on the 253 unblind subset (deploy-side, optimistic) ----
    in_rae_253 = float(rae(y_unb, te_nb1162[unb_idx]))
    delta_vs_anchor_te = in_rae_253 - rae_anchor_te_in
    print("\n" + "-" * 78)
    print("IN-SAMPLE DIAGNOSTIC (optimistic)")
    print("-" * 78)
    print(f"   in_RAE(te_nb1162[unb_idx]) = {in_rae_253:.4f}")
    print(f"   in_RAE(te_{ANCHOR}[unb_idx]) = {rae_anchor_te_in:.4f}")
    print(f"   delta vs anchor-in-sample   = {delta_vs_anchor_te:+.4f}")
    print(f"   anchor honest cross-fit RAE = {rae_anchor_oof:.4f}  "
          f"(reference for LB-faithful number; see nb1153 summary)")

    # ---- Save te artefact ----
    te_out = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_out, te_nb1162.astype(np.float32))
    print(f"\n[save] {te_out}")

    # ---- 513 mean/std summary (the structured-output ask) ----
    te_513_mean = float(te_nb1162.mean())
    te_513_std = float(te_nb1162.std())

    # ---- Save submission CSV (3 cols: SMILES, Molecule Name, pEC50) ----
    sub = pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": te_nb1162.astype(np.float64),
    })
    sub_path = os.path.join(SUBMISSIONS_DIR, f"{TAG}_deploy_nb1153.csv")
    sub.to_csv(sub_path, index=False)
    print(f"[save] {sub_path}  rows={len(sub)}")

    # ---- Summary ----
    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "deploy_companion_of": "nb1153",
        "feature_source": "mordred_cached_nb1030",
        "resid_seeds": RESID_SEEDS,
        "n_seeds": len(RESID_SEEDS),
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "lgbm_params_template": _lgbm_params(0),
        "feature_dim": int(X_unb.shape[1]),
        "rae_anchor_oof_253": rae_anchor_oof,
        "rae_anchor_te_in_sample_253": rae_anchor_te_in,
        "in_rae_253": in_rae_253,
        "delta_in_sample_vs_anchor_te": delta_vs_anchor_te,
        "residual_target_mean": float(residual_target.mean()),
        "residual_target_std": float(residual_target.std()),
        "te_residual_513_mean": float(te_residual_513.mean()),
        "te_residual_513_std": float(te_residual_513.std()),
        "per_seed_records": per_seed_records,
        "te_nb1162_mean": te_513_mean,
        "te_nb1162_std": te_513_std,
        "te_path": str(te_out),
        "submission_path": sub_path,
        "wall_sec": round(time.time() - t0, 2),
        "note": (
            "DEPLOY artifact: 5 LGBM seeds each fit on ALL 253 unblind rows "
            "(Mordred-only features), deploy residuals mean-bagged on 513. "
            "in_RAE is in-sample and optimistic; LB-faithful number is "
            "nb1153 honest cross-fit RAE."
        ),
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
        "rae_anchor_oof_253",
        "rae_anchor_te_in_sample_253",
        "in_rae_253",
        "delta_in_sample_vs_anchor_te",
        "te_residual_513_mean",
        "te_residual_513_std",
        "te_nb1162_mean",
        "te_nb1162_std",
        "te_path",
        "submission_path",
    ):
        print(f"  {k}: {res.get(k)}")
