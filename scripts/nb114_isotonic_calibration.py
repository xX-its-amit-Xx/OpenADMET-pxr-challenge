"""nb114 — Isotonic Regression Calibration for Extreme Predictions.

Problem identified from nb107 error analysis:
  - Compounds with true pEC50 < 2.5: predicted as ~4-5 (mean-shrinkage)
  - Compounds with true pEC50 > 6.0: predicted as ~3-4 (dramatic underestimation)
  - Both cases: model shrinks predictions toward training mean (~4.3)

Root cause: scaffold-CV extrapolation forces conservative predictions.
  - Test compound scaffolds are not in train folds → model is uncertain
  - LGBM defaults to near-mean predictions when uncertain

Fix: Isotonic regression (monotone calibration) learned from OOF residuals
  - Maps raw predictions to calibrated predictions using a non-decreasing function
  - Learns: "when model predicts 4.5, true value is often 1.8 or 7.0"
  - Monotone constraint ensures rankings are preserved
  - Applied to test predictions to expand the prediction range

Validation: Nested calibration using scaffold CV (calibration fit on fold-train OOF,
applied to fold-val predictions) to avoid OOF leakage.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from pathlib import Path

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
N_FOLDS = 5


def calibrate_with_isotonic(oof_tr, y_tr, te_preds, out_of_bounds="clip"):
    """Fit isotonic regression on OOF residuals and apply to test."""
    ir = IsotonicRegression(out_of_bounds=out_of_bounds)
    ir.fit(oof_tr, y_tr)
    return ir.predict(te_preds), ir


def nested_calibration_cv(oof_tr, y_tr, splits):
    """Nested isotonic calibration: fit on fold-train OOF, apply to fold-val."""
    calib_oof = np.full(len(y_tr), np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(oof_tr[tr_idx], y_tr[tr_idx])
        calib_oof[va_idx] = ir.predict(oof_tr[va_idx])
    return calib_oof


def full_metrics(y_true, y_pred, label=""):
    yt = np.asarray(y_true, float); yp = np.asarray(y_pred, float)
    msk = np.isfinite(yt) & np.isfinite(yp); yt, yp = yt[msk], yp[msk]
    mae_v = float(np.mean(np.abs(yt - yp)))
    rae_v = mae_v / float(np.mean(np.abs(yt - yt.mean()))) if yt.std() > 0 else np.nan
    pr, _ = stats.pearsonr(yt, yp)
    if label:
        print(f"  [{label:55s}] RAE={rae_v:.4f}  MAE={mae_v:.4f}  r={pr:.4f}")
    return dict(RAE=rae_v, MAE=mae_v, Pearson=pr)


def main():
    print("=== nb114: Isotonic Calibration for Extreme Predictions ===\n")

    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    print(f"Train pEC50: mean={y_tr.mean():.3f}  std={y_tr.std():.3f}")
    print(f"  < 2.5: {(y_tr < 2.5).sum()}  (very inactive)")
    print(f"  > 6.0: {(y_tr > 6.0).sum()}  (very active, seeds)")

    # ── Load all non-collapsed OOF models ────────────────────────────────────
    models_to_calibrate = [
        ("nb107_assay_decomp",    "AssayDecomp"),
        ("nb109_deep_meta_stack", "DeepMetaStack"),
        ("nb110_scaffold_prior",  "ScaffoldPrior"),
        ("nb111_selectivity_primary", "SelectivityPrimary"),
        ("nb101_delta_base",      "Delta-ML"),
        ("grand_v6b",             "Grand-v6b"),
    ]

    results = []
    for stem, label in models_to_calibrate:
        oof_f = DATA_PROCESSED / f"oof_{stem}.npy"
        te_f  = DATA_PROCESSED / f"te_{stem}.npy"
        if not oof_f.exists() or not te_f.exists():
            print(f"  SKIP (not found): {label}")
            continue

        oof_raw = np.load(oof_f)
        te_raw  = np.load(te_f)
        if oof_raw.ndim == 2: oof_raw = oof_raw[:, 0]
        if te_raw.ndim == 2:  te_raw  = te_raw[:, 0]
        if len(oof_raw) != n_tr:
            print(f"  SKIP (size mismatch): {label}")
            continue

        oof_raw = np.where(np.isfinite(oof_raw), oof_raw, np.nanmean(oof_raw))
        te_raw  = np.where(np.isfinite(te_raw), te_raw, np.nanmean(te_raw))

        # Test distribution check
        te_oof_ratio = te_raw.std() / oof_raw.std() if oof_raw.std() > 0 else 0
        if te_oof_ratio < 0.55:
            print(f"  SKIP (collapsed ratio={te_oof_ratio:.2f}): {label}")
            continue

        r_raw = rae(y_tr, oof_raw)
        print(f"\n[{label}]")
        full_metrics(y_tr, oof_raw, f"{label} (raw)")

        # ── Method 1: Nested isotonic calibration ────────────────────────────
        oof_calib_nested = nested_calibration_cv(oof_raw, y_tr, splits)
        r_nested = rae(y_tr, oof_calib_nested)
        full_metrics(y_tr, oof_calib_nested, f"{label} (nested calibrated)")

        # ── Method 2: Full OOF calibration (in-sample) ───────────────────────
        te_calib, ir_full = calibrate_with_isotonic(oof_raw, y_tr, te_raw)
        # In-sample OOF (same ir)
        oof_calib_insample = ir_full.predict(oof_raw)
        r_insample = rae(y_tr, oof_calib_insample)
        print(f"  [{label + ' (in-sample OOF)':<50s}] RAE={r_insample:.4f} (BIASED)")

        te_calib = np.clip(te_calib, y_tr.min() - 0.5, y_tr.max() + 0.5)

        print(f"  Raw test: std={te_raw.std():.3f}  max={te_raw.max():.2f}")
        print(f"  Calibrated test: std={te_calib.std():.3f}  max={te_calib.max():.2f}")

        # ── Method 3: Hybrid — calibrate then blend ───────────────────────────
        print("\n  Sweeping raw vs calibrated blend:")
        best_r, best_a = r_raw, 0.0
        for alpha in np.arange(0.0, 1.01, 0.1):
            blend = (1 - alpha) * oof_raw + alpha * oof_calib_nested
            r = rae(y_tr, blend)
            if r < best_r:
                best_r, best_a = r, alpha
            if alpha in [0.0, 0.2, 0.5, 0.8, 1.0]:
                print(f"    alpha(calib)={alpha:.1f}  RAE={r:.4f}")
        print(f"  Best blend: alpha={best_a:.1f}  RAE={best_r:.4f}")

        # Test predictions for best strategy
        te_best_blend = np.clip(
            (1 - best_a) * te_raw + best_a * te_calib,
            y_tr.min() - 0.5, y_tr.max() + 0.5
        )
        # Use nested OOF for the calibrated portion
        oof_best_blend = np.clip(
            (1 - best_a) * oof_raw + best_a * oof_calib_nested,
            y_tr.min() - 0.5, y_tr.max() + 0.5
        )

        results.append(dict(
            stem=stem, label=label,
            r_raw=r_raw, r_nested=r_nested, r_best=best_r,
            best_a=best_a,
            oof_raw=oof_raw, oof_best=oof_best_blend,
            te_raw=te_raw, te_best=te_best_blend,
        ))

    # ── Save calibrated submissions ───────────────────────────────────────────
    print("\n\n=== SUMMARY + SAVING ===")
    for res in sorted(results, key=lambda x: x["r_best"]):
        print(f"  {res['label']:30s}: raw={res['r_raw']:.4f}  "
              f"calib={res['r_nested']:.4f}  best_blend={res['r_best']:.4f} "
              f"(alpha={res['best_a']:.1f})")

        if res["r_best"] < res["r_raw"]:
            # Save calibrated submission
            sub = pd.DataFrame({"Molecule Name": te["name"].values,
                                 "pEC50": res["te_best"]})
            out_name = f"114_{res['stem'].replace('nb', '').replace('_', '-')}_calibrated.csv"
            sub.to_csv(SUBMISSIONS / out_name, index=False)
            print(f"    Saved: {out_name}")

            # Save best calibrated OOF for further ensembling
            np.save(DATA_PROCESSED / f"oof_{res['stem']}_calib.npy", res["oof_best"])
            np.save(DATA_PROCESSED / f"te_{res['stem']}_calib.npy", res["te_best"])

    # Best overall calibrated model
    if results:
        best = min(results, key=lambda x: x["r_best"])
        print(f"\nBEST CALIBRATED MODEL: {best['label']}")
        print(f"  OOF RAE: {best['r_raw']:.4f} (raw) -> {best['r_best']:.4f} (calibrated)")
        print(f"  Test std: {best['te_raw'].std():.3f} (raw) -> {best['te_best'].std():.3f} (calibrated)")
        print(f"  Test max: {best['te_raw'].max():.2f} (raw) -> {best['te_best'].max():.2f} (calibrated)")


if __name__ == "__main__":
    main()
