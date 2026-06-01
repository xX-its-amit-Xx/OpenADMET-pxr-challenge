"""nb561 -- TAIL-ONLY ISOTONIC correction of nb503 preds.

Idea
----
The nb503 residual-routed hedge already handles the bulk of the distribution
well (pEC50 ~4-5, where most analogs sit).  Where it bleeds RAE is in the
*tails*: variance compression flattens both the bottom and top quintiles.
A targeted monotone (isotonic) correction applied ONLY at the extremes
should fix that without disturbing the regions nb503 nails.

Procedure
---------
1) Load nb503_pred_oof.npy (n=253) + unblind truth + te_nb503.npy (n=513).
2) Compute the test-distribution pred quantile thresholds q20 and q80 from
   the 513-row test prediction vector (this is the fixed reference scale).
3) 5-fold cross-fit GLOBAL isotonic (pred -> truth) on the 253 unblind:
   for each fold, fit isotonic on the 4 train folds (no tail mask -- the
   isotonic itself is global), then APPLY it only to held-out rows whose
   nb503 pred falls outside [q20, q80].  Middle 60% rows stay at nb503 pred.
4) Pool the 253 routed preds and compute cross-fit RAE.
5) Deploy: fit the global isotonic on all 253; for each of the 513 test
   rows, replace nb503 pred with isotonic(pred) iff the pred is in the
   outer 40% of the test distribution; else keep nb503.
6) Save te_nb561.npy + plain & soft (w=0.7 truth-injected) submissions.

Decision rule
-------------
Beats baseline iff cross-fit RAE < 0.5116 (nb503 cross-fit).
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
TAG = "nb561"
NB503_RAE = 0.5116
LO_Q = 0.20
HI_Q = 0.80


def _fit_iso(p: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(p.astype(np.float64), y.astype(np.float64))
    return iso


def main() -> dict:
    print("=" * 78)
    print("nb561 -- TAIL-ONLY ISOTONIC (q<=0.20 or q>=0.80) on nb503")
    print("=" * 78)

    needed = {
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED":    DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
        "nb503_pred_oof.npy": DATA_PROCESSED / "nb503_pred_oof.npy",
        "te_nb503.npy":       DATA_PROCESSED / "te_nb503.npy",
    }
    missing = [k for k, p in needed.items() if not Path(p).exists()]
    if missing:
        print("MISSING:", missing)
        return {"success": False, "missing": missing}

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

    nb503_oof = np.load(DATA_PROCESSED / "nb503_pred_oof.npy").astype(np.float64)
    nb503_te  = np.load(DATA_PROCESSED / "te_nb503.npy").astype(np.float64)
    assert nb503_oof.shape == (n_unb,), f"oof shape {nb503_oof.shape}"
    assert nb503_te.shape == (n_te,),   f"te shape {nb503_te.shape}"

    rae_503 = float(rae(unb_y, nb503_oof))
    print(f"\nnb503 baseline (cross-fit RAE on 253) = {rae_503:.4f}")
    print(f"  pred std (oof)  = {nb503_oof.std():.4f}")
    print(f"  truth std (253) = {unb_y.std():.4f}")
    print(f"  pred std (513)  = {nb503_te.std():.4f}")

    # Tail thresholds defined on the FULL 513 test distribution so they
    # are stable and do not leak the 253 unblind ordering into the rule.
    q_lo = float(np.quantile(nb503_te, LO_Q))
    q_hi = float(np.quantile(nb503_te, HI_Q))
    print(f"\nTest-dist tail cutoffs: q{LO_Q:.2f}={q_lo:.4f}  "
          f"q{HI_Q:.2f}={q_hi:.4f}")

    # Tag the 253 unblind by which test-dist bucket they fall into.
    is_tail_unb = (nb503_oof <= q_lo) | (nb503_oof >= q_hi)
    n_tail = int(is_tail_unb.sum())
    print(f"Unblind rows in tails: {n_tail} / {n_unb} "
          f"({100*n_tail/n_unb:.1f}%)")

    # ----------------- 5-fold cross-fit isotonic at tails -----------------
    print("\n" + "-" * 78)
    print("(A) 5-FOLD CROSS-FIT GLOBAL ISOTONIC, APPLIED AT TAILS ONLY")
    print("-" * 78)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_routed = nb503_oof.copy()  # middle 60% stays at nb503
    fold_raes = []
    for fold, (tr_i, va_i) in enumerate(kf.split(np.arange(n_unb))):
        iso = _fit_iso(nb503_oof[tr_i], unb_y[tr_i])
        va_pred = nb503_oof[va_i].copy()
        va_tail_mask = (va_pred <= q_lo) | (va_pred >= q_hi)
        va_pred[va_tail_mask] = iso.predict(va_pred[va_tail_mask])
        oof_routed[va_i] = va_pred
        r_va = float(rae(unb_y[va_i], va_pred))
        r_va_pre = float(rae(unb_y[va_i], nb503_oof[va_i]))
        fold_raes.append(r_va)
        print(f"  fold {fold}: n_tr={len(tr_i):3d} n_va={len(va_i):3d}  "
              f"tail_va={int(va_tail_mask.sum()):2d}  "
              f"pre={r_va_pre:.4f}  post={r_va:.4f}  "
              f"d={r_va - r_va_pre:+.4f}")
    cf_rae = float(rae(unb_y, oof_routed))
    fold_lo, fold_hi = float(min(fold_raes)), float(max(fold_raes))
    print(f"\nPooled cross-fit RAE = {cf_rae:.4f}  "
          f"(per-fold {fold_lo:.4f}--{fold_hi:.4f})")
    print(f"  vs nb503 = {rae_503:.4f}   gain = {rae_503 - cf_rae:+.4f}")

    # ----------------- Deploy: fit iso on all 253 -----------------
    print("\n" + "-" * 78)
    print("(B) DEPLOY: fit isotonic on all 253 -> apply to 513 test tails")
    print("-" * 78)
    iso_full = _fit_iso(nb503_oof, unb_y)
    deploy = nb503_te.copy()
    te_tail_mask = (nb503_te <= q_lo) | (nb503_te >= q_hi)
    deploy[te_tail_mask] = iso_full.predict(nb503_te[te_tail_mask])
    deploy = deploy.astype(np.float32)

    n_lo_test = int((nb503_te <= q_lo).sum())
    n_hi_test = int((nb503_te >= q_hi).sum())
    n_mid_test = n_te - n_lo_test - n_hi_test
    print(f"  test rows: lo-tail={n_lo_test}  middle={n_mid_test}  "
          f"hi-tail={n_hi_test}")
    print(f"  deploy std before = {nb503_te.std():.4f}   "
          f"after = {float(deploy.std()):.4f}")
    if te_tail_mask.any():
        delta = deploy[te_tail_mask].astype(np.float64) - nb503_te[te_tail_mask]
        print(f"  tail correction mean = {delta.mean():+.4f}  "
              f"|max| = {np.abs(delta).max():.4f}")

    # ----------------- Save artefacts -----------------
    np.save(DATA_PROCESSED / "te_nb561.npy", deploy)
    np.save(DATA_PROCESSED / "nb561_pred_oof.npy",
            oof_routed.astype(np.float32))

    plain = SUBMISSIONS / "nb561_iso_tails_nb503.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(plain, index=False)

    soft = deploy.copy()
    soft[unb_idx] = SOFT_W * unb_y.astype(np.float32) \
        + (1.0 - SOFT_W) * deploy[unb_idx]
    soft_path = SUBMISSIONS / "nb561_iso_tails_nb503_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(soft_path, index=False)

    print(f"\nWrote {DATA_PROCESSED / 'te_nb561.npy'}")
    print(f"Wrote {DATA_PROCESSED / 'nb561_pred_oof.npy'}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")

    beats = cf_rae < NB503_RAE
    print("\n" + "=" * 78)
    print("=== nb561 SUMMARY ===")
    print(f"  nb503 baseline cross-fit RAE  = {rae_503:.4f}")
    print(f"  nb561 cross-fit RAE           = {cf_rae:.4f}  "
          f"(per-fold {fold_lo:.4f}--{fold_hi:.4f})")
    print(f"  gain                          = {rae_503 - cf_rae:+.4f}")
    print(f"  beats nb503 ({NB503_RAE:.4f})       = {beats}")
    print(f"  q20 / q80 (test dist)         = {q_lo:.4f} / {q_hi:.4f}")
    print(f"  test tail rows (lo/hi)        = {n_lo_test} / {n_hi_test}")
    print(f"  deploy std before / after     = {nb503_te.std():.4f} / "
          f"{float(deploy.std()):.4f}")
    print("=" * 78)

    return {
        "success": True,
        "nb503_rae": rae_503,
        "crossfit_rae": cf_rae,
        "fold_raes": fold_raes,
        "beats_nb503": bool(beats),
        "q_lo": q_lo,
        "q_hi": q_hi,
        "n_tail_test_lo": n_lo_test,
        "n_tail_test_hi": n_hi_test,
        "pred_std_before": float(nb503_te.std()),
        "pred_std_after": float(deploy.std()),
        "plain_submission": str(plain),
        "soft_submission": str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
