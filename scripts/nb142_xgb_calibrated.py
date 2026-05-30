"""nb142 — Calibrated XGBoost Meta-Stack.

Apply isotonic regression calibration (like nb114 did for nb109) to nb136's
XGBoost meta-stack. Motivation:
  - nb114 improved nb109 from 0.3741 to 0.3734 via calibration
  - nb136 may have systematic shrinkage at the extremes (test_std/oof_std=0.59)
  - Calibration may improve ensemble compatibility

Also: blend nb136_calibrated with nb139 adaptive blend at various alphas.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from scipy import stats

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
N_FOLDS = 5
CALIB_ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # alpha = weight on calibrated predictions


def nested_isotonic_calibrate(oof, y, te_preds, splits):
    """Nested isotonic calibration: calibrate oof in each fold, produce calibrated OOF + blend for test."""
    oof_calib = np.full(len(y), np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        # Fit isotonic on training fold
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(oof[tr_idx], y[tr_idx])
        oof_calib[va_idx] = ir.transform(oof[va_idx])
    # Full-data isotonic for test calibration
    ir_full = IsotonicRegression(out_of_bounds="clip")
    ir_full.fit(oof, y)
    te_calib = ir_full.transform(te_preds)
    return oof_calib, te_calib


def main():
    print("=== nb142: Calibrated XGBoost Meta-Stack ===\n")

    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # Load nb136 OOF and test predictions
    oof_136_p = DATA_PROCESSED / "oof_nb136_xgb_meta.npy"
    te_136_p  = DATA_PROCESSED / "te_nb136_xgb_meta.npy"
    if not oof_136_p.exists():
        print("ERROR: nb136 OOF not found. Run nb136 first.")
        return

    oof_136 = np.load(oof_136_p).astype(np.float64)
    te_136  = np.load(te_136_p).astype(np.float64)
    if oof_136.ndim == 2: oof_136 = oof_136[:, 0]
    if te_136.ndim == 2:  te_136  = te_136[:, 0]
    print(f"nb136 OOF RAE: {rae(y_tr, oof_136):.4f}")

    # Load nb139 adaptive blend
    oof_139_p = DATA_PROCESSED / "oof_nb139_adaptive_blend.npy"
    te_139_p  = DATA_PROCESSED / "te_nb139_adaptive_blend.npy"
    oof_139, te_139 = None, None
    if oof_139_p.exists():
        oof_139 = np.load(oof_139_p).astype(np.float64)
        te_139  = np.load(te_139_p).astype(np.float64)
        if oof_139.ndim == 2: oof_139 = oof_139[:, 0]
        if te_139.ndim == 2:  te_139  = te_139[:, 0]
        print(f"nb139 OOF RAE: {rae(y_tr, oof_139):.4f}")

    # Calibrate nb136
    print("\nApplying nested isotonic calibration to nb136...")
    oof_136_calib, te_136_calib = nested_isotonic_calibrate(oof_136, y_tr, te_136, splits)
    print(f"  nb136 calibrated OOF RAE: {rae(y_tr, oof_136_calib):.4f}")
    ratio_calib = te_136_calib.std() / oof_136_calib.std()
    print(f"  Test: std={te_136_calib.std():.3f}  ratio={ratio_calib:.2f}")

    # Alpha blend: calibrated vs. raw nb136
    print("\nAlpha blend (calibrated vs. raw nb136):")
    best_alpha = 0.0
    best_r = rae(y_tr, oof_136)
    for alpha in CALIB_ALPHAS:
        blend_oof = alpha * oof_136_calib + (1-alpha) * oof_136
        r = rae(y_tr, blend_oof)
        print(f"  alpha={alpha:.1f}  OOF RAE={r:.4f}")
        if r < best_r:
            best_r, best_alpha = r, alpha

    print(f"\nBest calibration alpha: {best_alpha:.1f}  OOF RAE={best_r:.4f}")
    best_oof_136 = best_alpha * oof_136_calib + (1-best_alpha) * oof_136
    best_te_136  = best_alpha * te_136_calib  + (1-best_alpha) * te_136

    # Blend nb136 (possibly calibrated) with nb139
    if oof_139 is not None:
        print("\nBlend nb136_best with nb139:")
        for beta in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            blend_oof = beta * best_oof_136 + (1-beta) * oof_139
            r = rae(y_tr, blend_oof)
            print(f"  beta(136)={beta:.1f}  OOF RAE={r:.4f}")

    # Save calibrated nb136
    te_out = np.clip(best_te_136, y_tr.min() - 0.5, y_tr.max() + 0.5)
    np.save(DATA_PROCESSED / "oof_nb142_xgb_calibrated.npy", best_oof_136)
    np.save(DATA_PROCESSED / "te_nb142_xgb_calibrated.npy",  te_out)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_out})
    sub.to_csv(SUBMISSIONS / "142_xgb_calibrated.csv", index=False)
    print(f"\nSaved: submissions/142_xgb_calibrated.csv")
    print(f"OOF RAE: {best_r:.4f}")


if __name__ == "__main__":
    main()
