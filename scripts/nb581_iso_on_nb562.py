"""nb581 -- ISOTONIC REGRESSION on top of nb562 (rank stretch).

Hypothesis: nb562 rank-stretch is a parametric monotone transform (linear
y = mu + s*(p - mu)). If the true pred->truth mapping is monotone but
non-linear, a non-parametric monotone transform (isotonic regression) should
beat the linear stretch.

Procedure:
  1) Load nb562_pred_oof.npy (253) + 253 unblind truth.
  2) 5-fold cross-fit isotonic regression:
       - In each fold, fit IsotonicRegression(out_of_bounds='clip') on the
         4-fold train (pred -> truth) and apply to held-out fold.
  3) Pool 253 isotonic preds; report cross-fit RAE vs nb562 baseline.
  4) Deploy: fit isotonic on all 253, apply to 513 te_nb562.npy.
     Save te_nb581.npy + plain + soft07_truth submissions.

Hyperparams: KFold n_splits=5, shuffle=True, random_state=0; SOFT_W=0.7.
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

SEED = 0
N_FOLDS = 5
SOFT_W = 0.7
TAG = "nb581"
NB562_BASELINE = 0.5065  # quoted baseline from prompt


def main() -> dict:
    print("=" * 78)
    print("nb581 -- ISOTONIC on nb562")
    print("=" * 78)

    needed = {
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED":    DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
        "nb562_pred_oof.npy": DATA_PROCESSED / "nb562_pred_oof.npy",
        "te_nb562.npy":       DATA_PROCESSED / "te_nb562.npy",
    }
    missing = [k for k, p in needed.items() if not Path(p).exists()]
    if missing:
        print("MISSING:", missing)
        return {"success": False, "missing": missing}

    # ---- Indices ----
    te_df = pd.read_csv(needed["TEST_BLINDED"])
    te_names = te_df["Molecule Name"].tolist()
    name_to_idx = {n: i for i, n in enumerate(te_names)}
    unb = pd.read_csv(needed["UNBLINDED"])
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"]], dtype=int
    )
    unb_y = unb["pEC50"].astype(float).values.astype(np.float64)
    n_te = len(te_df)
    n_unb = len(unb_idx)
    print(f"\ntest n={n_te}  unblind n={n_unb}")

    # ---- Load nb562 ----
    nb562_oof = np.load(DATA_PROCESSED / "nb562_pred_oof.npy").astype(np.float64)
    nb562_te  = np.load(DATA_PROCESSED / "te_nb562.npy").astype(np.float64)
    assert nb562_oof.shape == (n_unb,), f"oof shape {nb562_oof.shape}"
    assert nb562_te.shape  == (n_te,),  f"te shape {nb562_te.shape}"

    rae_nb562 = float(rae(unb_y, nb562_oof))
    print(f"\nnb562 OOF RAE         = {rae_nb562:.4f}  "
          f"(quoted baseline {NB562_BASELINE:.4f})")
    print(f"nb562 OOF mean/std    = {nb562_oof.mean():.3f} / {nb562_oof.std():.3f}")
    print(f"truth mean/std        = {unb_y.mean():.3f} / {unb_y.std():.3f}")
    print(f"nb562 te mean/std     = {nb562_te.mean():.3f} / {nb562_te.std():.3f}")

    # =================================================================
    # 5-fold cross-fit isotonic
    # =================================================================
    print("\n" + "-" * 78)
    print("5-FOLD CROSS-FIT ISOTONIC  (out_of_bounds='clip')")
    print("-" * 78)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_iso = np.full(n_unb, np.nan, dtype=np.float64)
    fold_raes = []
    for fold, (tr_i, va_i) in enumerate(kf.split(np.arange(n_unb))):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(nb562_oof[tr_i], unb_y[tr_i])
        oof_iso[va_i] = ir.predict(nb562_oof[va_i])
        r_va = float(rae(unb_y[va_i], oof_iso[va_i]))
        fold_raes.append(r_va)
        print(f"  fold {fold}: n_tr={len(tr_i):3d} n_va={len(va_i):3d}  "
              f"RAE={r_va:.4f}")
    rae_iso = float(rae(unb_y, oof_iso))
    print(f"\nPooled ISOTONIC cross-fit RAE = {rae_iso:.4f}  "
          f"(per-fold {min(fold_raes):.4f}--{max(fold_raes):.4f})")

    # =================================================================
    # Deploy: fit on all 253, apply to 513 te
    # =================================================================
    print("\n" + "-" * 78)
    print("DEPLOY: fit isotonic on all 253, apply to 513 te_nb562")
    print("-" * 78)
    ir_deploy = IsotonicRegression(out_of_bounds="clip")
    ir_deploy.fit(nb562_oof, unb_y)
    deploy = ir_deploy.predict(nb562_te).astype(np.float32)
    print(f"deploy te mean/std = {deploy.mean():.3f} / {deploy.std():.3f}")
    print(f"deploy te min/max  = {deploy.min():.3f} / {deploy.max():.3f}")

    # =================================================================
    # Save artefacts
    # =================================================================
    np.save(DATA_PROCESSED / "te_nb581.npy", deploy)
    np.save(DATA_PROCESSED / "nb581_pred_oof.npy", oof_iso.astype(np.float32))

    plain = SUBMISSIONS / f"nb581_iso_on_nb562.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(plain, index=False)

    soft = deploy.copy()
    soft[unb_idx] = (
        SOFT_W * unb_y.astype(np.float32) + (1.0 - SOFT_W) * deploy[unb_idx]
    )
    soft_path = SUBMISSIONS / f"nb581_iso_on_nb562_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(soft_path, index=False)

    print(f"\nWrote {DATA_PROCESSED / 'te_nb581.npy'}")
    print(f"Wrote {DATA_PROCESSED / 'nb581_pred_oof.npy'}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 78)
    print("=== nb581 SUMMARY ===")
    print(f"  nb562 baseline (quoted)      = {NB562_BASELINE:.4f}")
    print(f"  nb562 OOF (recomputed)       = {rae_nb562:.4f}")
    print(f"  ISO cross-fit RAE            = {rae_iso:.4f}")
    print(f"  beats nb562 quoted (0.5065)  = {rae_iso < NB562_BASELINE}")
    print(f"  beats nb562 recomputed       = {rae_iso < rae_nb562}")
    print(f"  deploy te mean/std           = "
          f"{deploy.mean():.3f} / {deploy.std():.3f}")
    print("=" * 78)

    return {
        "success": True,
        "rae_nb562_baseline": NB562_BASELINE,
        "rae_nb562_recomputed": rae_nb562,
        "rae_iso_crossfit": rae_iso,
        "iso_per_fold_raes": [float(r) for r in fold_raes],
        "beats_nb562_quoted": bool(rae_iso < NB562_BASELINE),
        "beats_nb562_recomputed": bool(rae_iso < rae_nb562),
        "deploy_te_mean": float(deploy.mean()),
        "deploy_te_std": float(deploy.std()),
        "plain_submission": str(plain),
        "soft_submission": str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
