"""nb1131 -- DEPLOY artifact: shallow LGBM Huber residual on top of nb1070.

This is the deploy-side companion to nb1123 / nb1130 (honest cross-fit
residual probes). Where those scripts cross-fit to estimate honest
generalization, this script TRAINS THE SHALLOW LGBM ON ALL 253 UNBLIND
ROWS and predicts the residual on the full 513-compound test set so we
have a deploy-ready prediction vector.

Pipeline:
  1. Load nb1070 cross-fit OOF on 253 unblind   (nb1070_pred_oof.npy)
     and nb1070 deploy preds on 513             (te_nb1070.npy).
  2. residual_target = y_unb - nb1070_oof       (n=253, signed).
  3. Featurize: combined Morgan+RDKit (2265) on the 253 unblind SMILES
     for fitting; same featurizer on all 513 test SMILES for inference.
  4. Fit a SINGLE shallow LGBM Huber (max_depth=3, n_est=80, lr=0.05,
     min_child_samples=20, alpha=1.0) on ALL 253 unblind rows
     (no held-out -- this is deploy).
  5. residual_pred_513 = mdl.predict(X_513).
  6. te_nb1131 = te_nb1070 + residual_pred_513.
  7. in_RAE = rae(y_unb, te_nb1131[unb_idx])   (IN-SAMPLE, optimistic).
  8. Write submissions/nb1131_residual_deploy.csv   (SMILES, name, pEC50)
     and data/processed/te_nb1131.npy.

Caveat (per feedback_lb_two_regime_calibration): this is a POST-unblind
deploy artifact -- in_RAE is the in-sample fit, NOT a faithful LB
estimate. The honest cross-fit RAE from nb1123 / nb1130 is the
LB-faithful number; this file produces the deploy vector for ladder
submission once the cross-fit number is judged acceptable.
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
from lightgbm import LGBMRegressor

from pxr.data import load_test
from pxr.eval import rae
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED

TAG = "nb1131"
ANCHOR = "nb1070"
RESID_SEED = 42

LGBM_PARAMS = dict(
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
    random_state=RESID_SEED,
    n_jobs=2,
)

SUBMISSIONS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "submissions")
)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- DEPLOY residual: shallow LGBM Huber on ALL 253 unblind")
    print(f"          anchor = {ANCHOR} (te_{ANCHOR}.npy + {ANCHOR}_pred_oof.npy)")
    print(f"          features = combined Morgan+RDKit (2265)")
    print(f"          LGBM: max_depth=3, n_est=80, lr=0.05, "
          f"min_child_samples=20, obj=huber(alpha=1.0), seed={RESID_SEED}")
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

    # ---- Signed residual target ----
    residual_target = y_unb - nb1070_oof
    print(f"[resid] mean={residual_target.mean():+.4f}  "
          f"std={residual_target.std():.4f}  "
          f"min={residual_target.min():+.4f}  "
          f"max={residual_target.max():+.4f}")

    # ---- Featurize: 253 unblind for fit, 513 test for inference ----
    smi_unb = te_smiles[unb_idx].tolist()
    print(f"[feat] computing combined(Morgan+RDKit) on n={len(smi_unb)} "
          f"unblind SMILES (fit set)")
    X_unb = impute(combined(smi_unb))
    print(f"[feat] X_unb shape = {X_unb.shape}")

    print(f"[feat] computing combined(Morgan+RDKit) on n={n_test} "
          f"test SMILES (inference set)")
    X_test = impute(combined(te_smiles.tolist()))
    print(f"[feat] X_test shape = {X_test.shape}")

    # ---- Fit shallow LGBM Huber on ALL 253 unblind ----
    print("\n" + "-" * 78)
    print(f"DEPLOY FIT  (all n={n_unb} unblind rows, seed={RESID_SEED})")
    print("-" * 78)
    mdl = LGBMRegressor(**LGBM_PARAMS)
    mdl.fit(X_unb, residual_target)
    print(f"[fit] LGBM trained on {n_unb} rows x {X_unb.shape[1]} features")

    # In-sample residual fit on 253 (will be optimistic).
    residual_in = mdl.predict(X_unb)
    in_pred_corr_253 = nb1070_oof + residual_in
    in_rae_253_corr_on_oof = float(rae(y_unb, in_pred_corr_253))
    print(f"[fit] in-sample pooled RAE (anchor=OOF + residual_in) = "
          f"{in_rae_253_corr_on_oof:.4f}  "
          f"(d_vs_anchor_oof = {in_rae_253_corr_on_oof - rae_anchor_oof:+.4f})")

    # ---- Predict residual on 513 ----
    residual_pred_513 = mdl.predict(X_test).astype(np.float64)
    print(f"[pred] residual_pred_513: shape={residual_pred_513.shape}  "
          f"mean={residual_pred_513.mean():+.4f}  "
          f"std={residual_pred_513.std():.4f}  "
          f"min={residual_pred_513.min():+.4f}  "
          f"max={residual_pred_513.max():+.4f}")

    # ---- te_nb1131 = te_nb1070 + residual_pred_513 ----
    te_nb1131 = te_nb1070 + residual_pred_513
    print(f"[deploy] te_nb1131 shape={te_nb1131.shape}  "
          f"mean={te_nb1131.mean():.3f}  std={te_nb1131.std():.3f}  "
          f"min={te_nb1131.min():.3f}  max={te_nb1131.max():.3f}")

    # ---- In-sample RAE on the 253 unblind subset (deploy-side) ----
    in_rae_253 = float(rae(y_unb, te_nb1131[unb_idx]))
    delta_vs_anchor_te = in_rae_253 - rae_anchor_te_in
    print("\n" + "-" * 78)
    print("IN-SAMPLE DIAGNOSTIC (optimistic)")
    print("-" * 78)
    print(f"   in_RAE(te_nb1131[unb_idx]) = {in_rae_253:.4f}")
    print(f"   in_RAE(te_{ANCHOR}[unb_idx]) = {rae_anchor_te_in:.4f}")
    print(f"   delta vs anchor-in-sample   = {delta_vs_anchor_te:+.4f}")
    print(f"   anchor honest cross-fit RAE = {rae_anchor_oof:.4f}  "
          f"(reference for LB-faithful number)")

    # ---- Save te artefact ----
    te_out = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_out, te_nb1131.astype(np.float32))
    print(f"\n[save] {te_out}")

    # ---- Save submission CSV (3 cols: SMILES, Molecule Name, pEC50) ----
    sub = pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": te_nb1131.astype(np.float64),
    })
    sub_path = os.path.join(SUBMISSIONS_DIR, f"{TAG}_residual_deploy.csv")
    sub.to_csv(sub_path, index=False)
    print(f"[save] {sub_path}  rows={len(sub)}")

    # ---- Summary ----
    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "lgbm_params": LGBM_PARAMS,
        "feature_dim": int(X_unb.shape[1]),
        "rae_anchor_oof_253": rae_anchor_oof,
        "rae_anchor_te_in_sample_253": rae_anchor_te_in,
        "in_rae_253": in_rae_253,
        "delta_in_sample_vs_anchor_te": delta_vs_anchor_te,
        "in_rae_253_corr_on_oof_anchor": in_rae_253_corr_on_oof,
        "residual_target_mean": float(residual_target.mean()),
        "residual_target_std": float(residual_target.std()),
        "residual_pred_513_mean": float(residual_pred_513.mean()),
        "residual_pred_513_std": float(residual_pred_513.std()),
        "te_nb1131_mean": float(te_nb1131.mean()),
        "te_nb1131_std": float(te_nb1131.std()),
        "te_path": str(te_out),
        "submission_path": sub_path,
        "wall_sec": round(time.time() - t0, 2),
        "note": (
            "DEPLOY artifact: LGBM fit on ALL 253 unblind rows; in_RAE "
            "is in-sample and optimistic. For LB-faithful RAE see "
            "nb1123 / nb1130 honest cross-fit summaries."
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
        "residual_pred_513_mean",
        "residual_pred_513_std",
        "te_nb1131_mean",
        "te_nb1131_std",
        "te_path",
        "submission_path",
    ):
        print(f"  {k}: {res.get(k)}")
