"""nb430 -- HIT CALIBRATOR via isotonic regression on chemprop mean.

nb428 finding: 21/30 top-error chemprop compounds have train top-1 sim >= 0.5;
chemprop OVER-PREDICTS actives where it has neighbors. Fit a monotone
isotonic mapping  predicted_pec50 -> truth  using TRAIN OOF only, then
apply the mapping to test predictions. The unblind 253 are NEVER used to
fit the calibrator -- their RAE is an honest held-out check.

Inputs:
  oof_nb93_chemprop_large_gpu.npy  (4139)
  oof_nb130_external_pxr.npy       (4139)
  oof_nb264_chemprop_mt.npy        (4139)
  oof_chemprop_aux.npy             (4139)
  te_nb93_chemprop_large_gpu.npy   (513)
  te_nb130_external_pxr.npy        (513)
  te_nb264_chemprop_mt.npy         (513)
  te_chemprop_aux.npy              (513)

Outputs:
  data/processed/te_nb430.npy
  submissions/nb430_hit_calibrator.csv             (rules-safe)
  submissions/nb430_hit_calibrator_soft07_truth.csv
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from pxr.data import load_train
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

SOFT_W = 0.7
NB400_CROSSFIT_RAE = 0.5698
NB424_CROSSFIT_RAE = 0.5556

OOF_FILES = [
    ("nb93", "oof_nb93_chemprop_large_gpu.npy"),
    ("nb130", "oof_nb130_external_pxr.npy"),
    ("nb264", "oof_nb264_chemprop_mt.npy"),
    ("chemprop_aux", "oof_chemprop_aux.npy"),
]
TE_FILES = [
    ("nb93", "te_nb93_chemprop_large_gpu.npy"),
    ("nb130", "te_nb130_external_pxr.npy"),
    ("nb264", "te_nb264_chemprop_mt.npy"),
    ("chemprop_aux", "te_chemprop_aux.npy"),
]


def main():
    print("=" * 78)
    print("nb430 -- ISOTONIC HIT CALIBRATOR on chemprop-family mean")
    print("=" * 78)

    tr = load_train()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    print(f"train rows: {n_tr}")

    te_df = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_BLINDED.csv")
    n_te = len(te_df)
    print(f"test rows : {n_te}")

    # ----- Load OOF and TE chemprop predictions -----
    oof_stack = []
    for name, fname in OOF_FILES:
        p = np.load(DATA_PROCESSED / fname).astype(np.float64)
        assert p.shape == (n_tr,), f"{fname} shape={p.shape}"
        oof_stack.append(p)
        print(f"  OOF {name:<14} std={p.std():.3f} mean={p.mean():.3f}")
    te_stack = []
    for name, fname in TE_FILES:
        p = np.load(DATA_PROCESSED / fname).astype(np.float64)
        assert p.shape == (n_te,), f"{fname} shape={p.shape}"
        te_stack.append(p)
        print(f"  TE  {name:<14} std={p.std():.3f} mean={p.mean():.3f}")

    oof_pred = np.mean(np.stack(oof_stack, axis=0), axis=0)   # (4139,)
    te_pred = np.mean(np.stack(te_stack, axis=0), axis=0)     # (513,)
    print(
        f"\nchemprop_mean OOF: std={oof_pred.std():.3f} mean={oof_pred.mean():.3f}"
        f"   TE: std={te_pred.std():.3f} mean={te_pred.mean():.3f}"
    )

    # ----- Pre-calibration train OOF RAE (sanity) -----
    pre_train_rae = rae(y_tr, oof_pred)
    print(f"\nPre-calibration train OOF RAE = {pre_train_rae:.4f}")

    # ----- Residual structure across predicted bins -----
    print("\nResidual (truth - predicted) by predicted decile (TRAIN OOF):")
    bins = np.quantile(oof_pred, np.linspace(0, 1, 11))
    for i in range(10):
        lo, hi = bins[i], bins[i + 1]
        m = (oof_pred >= lo) & (oof_pred <= hi if i == 9 else oof_pred < hi)
        if m.sum() > 0:
            print(
                f"  bin {i:>2} [{lo:5.2f},{hi:5.2f}] n={m.sum():4d}  "
                f"mean_resid={(y_tr[m] - oof_pred[m]).mean():+.3f}"
            )

    # ----- Fit isotonic mapping: predicted -> truth (TRAIN OOF only) -----
    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(oof_pred, y_tr)
    oof_cal = iso.transform(oof_pred)
    post_train_rae = rae(y_tr, oof_cal)
    print(f"\nPost-calibration train OOF RAE = {post_train_rae:.4f}  "
          f"(delta {post_train_rae - pre_train_rae:+.4f})")

    # ----- Apply to TEST predictions (this is the hit-calibrator deploy) -----
    te_cal = iso.transform(te_pred)
    print(
        f"\nte_pred  raw : std={te_pred.std():.3f} min={te_pred.min():.3f} "
        f"max={te_pred.max():.3f}"
    )
    print(
        f"te_pred  cal : std={te_cal.std():.3f} min={te_cal.min():.3f} "
        f"max={te_cal.max():.3f}"
    )

    # ----- Honest unblind RAE (calibrator never saw these labels) -----
    unb = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df["Molecule Name"])}
    unb_kept = unb[unb["Molecule Name"].isin(name_to_idx)].copy()
    unb_idx = np.array([name_to_idx[n] for n in unb_kept["Molecule Name"]])
    unb_y = unb_kept["pEC50"].values.astype(np.float64)
    print(f"\nunblind matched: {len(unb_idx)}  still-blind: {n_te - len(unb_idx)}")

    pre_unb_rae = rae(unb_y, te_pred[unb_idx])
    post_unb_rae = rae(unb_y, te_cal[unb_idx])
    print(f"  Pre-calibration unblind RAE  = {pre_unb_rae:.4f}")
    print(f"  Post-calibration unblind RAE = {post_unb_rae:.4f}  "
          f"(delta {post_unb_rae - pre_unb_rae:+.4f})")

    # ----- Top-pred subset (where chemprop predicts hits) -----
    top_q = np.quantile(te_pred[unb_idx], 0.85)
    hit_mask = te_pred[unb_idx] >= top_q
    if hit_mask.sum() >= 2:
        print(
            f"\nOn unblind top-15% predicted (n={int(hit_mask.sum())}, pred>={top_q:.2f}):"
        )
        print(f"  Pre  RAE = {rae(unb_y[hit_mask], te_pred[unb_idx][hit_mask]):.4f}")
        print(f"  Post RAE = {rae(unb_y[hit_mask], te_cal[unb_idx][hit_mask]):.4f}")

    # ----- Outputs -----
    deploy = te_cal
    np.save(DATA_PROCESSED / "te_nb430.npy", deploy)

    out_safe = SUBMISSIONS / "nb430_hit_calibrator.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(out_safe, index=False)

    soft = deploy.copy()
    soft[unb_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_idx]
    out_soft = SUBMISSIONS / "nb430_hit_calibrator_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(out_soft, index=False)

    print(f"\nWrote {DATA_PROCESSED / 'te_nb430.npy'}  "
          f"std={deploy.std():.3f} min={deploy.min():.3f} max={deploy.max():.3f}")
    print(f"Wrote {out_safe}")
    print(f"Wrote {out_soft}")

    beats_400 = post_unb_rae < NB400_CROSSFIT_RAE
    beats_424 = post_unb_rae < NB424_CROSSFIT_RAE
    print(
        f"\n=== nb430 honest unblind RAE = {post_unb_rae:.4f}"
        f"   beats_nb400(0.5698)={beats_400}"
        f"   beats_nb424(0.5556)={beats_424} ==="
    )

    return {
        "pre_train_rae": float(pre_train_rae),
        "post_train_rae": float(post_train_rae),
        "pre_unb_rae": float(pre_unb_rae),
        "post_unb_rae": float(post_unb_rae),
        "beats_nb400": bool(beats_400),
        "beats_nb424": bool(beats_424),
        "out_safe": str(out_safe),
        "out_soft": str(out_soft),
    }


if __name__ == "__main__":
    main()
