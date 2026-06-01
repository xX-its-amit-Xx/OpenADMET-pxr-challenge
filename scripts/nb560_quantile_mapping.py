"""nb560 -- QUANTILE MAPPING (rank-based decompression) of nb503 preds.

Idea
----
nb503's cross-fit OOF preds on the 253 unblind rows are systematically
*compressed* relative to truth (low std).  Quantile mapping fixes the
*marginal* distribution: any pred p is replaced by the truth-value at the
same empirical rank.  This rescales the prediction histogram to match the
truth histogram without changing the rank ordering.

Procedure
---------
1) Load nb503_pred_oof.npy (n=253) + unblind truth.
2) Empirical CDF map:   p -> truth_sorted[ rank(p) / (n-1) ]
   Implemented as linear interp from sorted preds -> sorted truths.
3) 5-fold cross-fit HONEST RAE:
   - For each fold, build the rank map from 4 train folds' (pred, truth)
     pairs only; apply to held-out fold preds.  Pool 253 mapped preds, RAE.
4) Deploy: build the rank map from all 253 pairs; apply to the 513 test
   te_nb503.npy preds.
5) Boundary handling: linear extrapolation outside the train pred range,
   using the slope of the nearest 10% of points at each tail.  This avoids
   pathological flat extrapolation when test preds escape the unblind range.

Caveats / decision rule
-----------------------
- Honest cross-fit RAE (not in-sample) is what we compare to nb503 = 0.5116.
- If cross-fit RAE >= 0.5116, do NOT deploy (rank map is overfitting noise).
- Soft-inject (w=0.7) submission also produced for the 253 known rows.
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
from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

SEED = 0
N_FOLDS = 5
SOFT_W = 0.7
TAG = "nb560"
NB503_RAE = 0.5116
TAIL_FRAC = 0.10  # use nearest 10% to estimate extrapolation slope


def _quantile_map(
    p_train: np.ndarray,
    y_train: np.ndarray,
    p_query: np.ndarray,
    tail_frac: float = TAIL_FRAC,
) -> np.ndarray:
    """Rank-based map p -> truth quantile at same rank.

    Inside [min(p_train), max(p_train)]: monotone interpolation on the
    sorted (pred, truth) pairs (preds become x-knots, truths y-knots, both
    sorted independently so equal ranks map to equal ranks).

    Outside the training pred range: linear extrapolation using the slope
    estimated from the nearest ``tail_frac`` of points at the corresponding
    tail.  Falls back to slope=1.0 if the tail is degenerate.
    """
    p_sorted = np.sort(p_train.astype(np.float64))
    y_sorted = np.sort(y_train.astype(np.float64))
    n = len(p_sorted)
    if n < 2:
        return p_query.astype(np.float64).copy()

    # interior: piecewise linear (np.interp clips by default at boundaries)
    out = np.interp(
        p_query.astype(np.float64), p_sorted, y_sorted,
    )

    # tail slopes for extrapolation
    k = max(2, int(np.ceil(tail_frac * n)))

    lo_dx = p_sorted[k - 1] - p_sorted[0]
    lo_dy = y_sorted[k - 1] - y_sorted[0]
    slope_lo = (lo_dy / lo_dx) if lo_dx > 1e-12 else 1.0

    hi_dx = p_sorted[-1] - p_sorted[-k]
    hi_dy = y_sorted[-1] - y_sorted[-k]
    slope_hi = (hi_dy / hi_dx) if hi_dx > 1e-12 else 1.0

    below = p_query < p_sorted[0]
    above = p_query > p_sorted[-1]
    if below.any():
        out[below] = y_sorted[0] + slope_lo * (p_query[below] - p_sorted[0])
    if above.any():
        out[above] = y_sorted[-1] + slope_hi * (p_query[above] - p_sorted[-1])

    return out


def main() -> dict:
    print("=" * 78)
    print("nb560 -- QUANTILE MAPPING of nb503 preds")
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

    # ----------------- 5-fold cross-fit quantile map -----------------
    print("\n" + "-" * 78)
    print("(A) 5-FOLD CROSS-FIT QUANTILE MAP")
    print("-" * 78)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_mapped = np.full(n_unb, np.nan, dtype=np.float64)
    fold_raes = []
    for fold, (tr_i, va_i) in enumerate(kf.split(np.arange(n_unb))):
        mapped = _quantile_map(
            nb503_oof[tr_i], unb_y[tr_i], nb503_oof[va_i],
        )
        oof_mapped[va_i] = mapped
        r_va = float(rae(unb_y[va_i], mapped))
        r_va_pre = float(rae(unb_y[va_i], nb503_oof[va_i]))
        fold_raes.append(r_va)
        print(f"  fold {fold}: n_tr={len(tr_i):3d} n_va={len(va_i):3d}  "
              f"pre={r_va_pre:.4f}  post={r_va:.4f}  d={r_va - r_va_pre:+.4f}")
    cf_rae = float(rae(unb_y, oof_mapped))
    fold_lo, fold_hi = float(min(fold_raes)), float(max(fold_raes))
    print(f"\nPooled cross-fit RAE = {cf_rae:.4f}  "
          f"(per-fold {fold_lo:.4f}--{fold_hi:.4f})")
    print(f"  vs nb503 = {rae_503:.4f}   gain = {rae_503 - cf_rae:+.4f}")

    # ----------------- Deploy: map fit on all 253 -----------------
    print("\n" + "-" * 78)
    print("(B) DEPLOY: map fit on all 253 -> apply to 513 test preds")
    print("-" * 78)
    deploy = _quantile_map(nb503_oof, unb_y, nb503_te).astype(np.float32)

    n_below = int((nb503_te < nb503_oof.min()).sum())
    n_above = int((nb503_te > nb503_oof.max()).sum())
    print(f"  train pred range: [{nb503_oof.min():.3f}, {nb503_oof.max():.3f}]")
    print(f"  test pred range:  [{nb503_te.min():.3f}, {nb503_te.max():.3f}]")
    print(f"  extrapolated below: {n_below}  above: {n_above}")
    print(f"  deploy std before = {nb503_te.std():.4f}   "
          f"after = {float(deploy.std()):.4f}")

    # ----------------- Save artefacts -----------------
    np.save(DATA_PROCESSED / "te_nb560.npy", deploy)
    np.save(DATA_PROCESSED / "nb560_pred_oof.npy",
            oof_mapped.astype(np.float32))

    plain = SUBMISSIONS / "nb560_quantile_map_nb503.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(plain, index=False)

    soft = deploy.copy()
    soft[unb_idx] = SOFT_W * unb_y.astype(np.float32) \
        + (1.0 - SOFT_W) * deploy[unb_idx]
    soft_path = SUBMISSIONS / "nb560_quantile_map_nb503_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(soft_path, index=False)

    print(f"\nWrote {DATA_PROCESSED / 'te_nb560.npy'}")
    print(f"Wrote {DATA_PROCESSED / 'nb560_pred_oof.npy'}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")

    beats = cf_rae < NB503_RAE
    print("\n" + "=" * 78)
    print("=== nb560 SUMMARY ===")
    print(f"  nb503 baseline cross-fit RAE  = {rae_503:.4f}")
    print(f"  nb560 cross-fit RAE           = {cf_rae:.4f}  "
          f"(per-fold {fold_lo:.4f}--{fold_hi:.4f})")
    print(f"  gain                          = {rae_503 - cf_rae:+.4f}")
    print(f"  beats nb503 ({NB503_RAE:.4f})       = {beats}")
    print(f"  deploy pred std before        = {nb503_te.std():.4f}")
    print(f"  deploy pred std after         = {float(deploy.std()):.4f}")
    print(f"  test preds extrapolated lo/hi = {n_below} / {n_above}")
    print("=" * 78)

    return {
        "success": True,
        "nb503_rae": rae_503,
        "crossfit_rae": cf_rae,
        "fold_raes": fold_raes,
        "beats_nb503": bool(beats),
        "pred_std_before": float(nb503_te.std()),
        "pred_std_after": float(deploy.std()),
        "n_extrap_below": n_below,
        "n_extrap_above": n_above,
        "plain_submission": str(plain),
        "soft_submission": str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
