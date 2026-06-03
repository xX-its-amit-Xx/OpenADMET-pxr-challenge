"""nb1082 -- Beta calibration (Kull 2017) on te_nb1014.

Motivation
----------
nb1070's per-quantile median bag (5-bin per-bin mu/s grid) hits OOF RAE 0.5771
on the nb1014 anchor. That calibrator has 5 disjoint (mu_b, s_b) cells (5
parameters when mu_b is data-derived; 10 if you count both) -- discretized
binning is a fragile family on n=253 (each bin ~50 points).

Kull (2017) -- "Beta calibration: a well-founded and easily implemented
improvement on logistic calibration for binary classifiers" -- proposes a
smooth 2-parameter family on [0,1]:

        log(z / (1 - z))   =   a * log(p)  -  b * log(1 - p)  +  c

i.e. the calibrated score is proportional to  p^a * (1-p)^b  (after the
logit link). Two free shape parameters (a, b) over the unit interval, plus
an intercept c -- continuous, monotone-friendly, and infinitely fewer
degrees of freedom than per-bin stretching.

We adapt this to *regression* by:
   1. Normalize raw pred p (and truth y) to [0,1] via a SHARED affine map
      derived from train-fold pred/truth range.
   2. Logit-transform p; fit (a, b, c) so that the logit-cal map applied to
      raw p best matches the affine-mapped truth (MSE on z-space).
   3. Invert the affine map to return to pEC50.

The two parameters (a, b) control the curve shape:  a>1 stretches high
end, b>1 compresses low end, etc. This is strictly smoother than the
5-bin per-quantile alternative.

Protocol
--------
Honest 5-fold cross-fit on 253 unblind (KFold shuffle, seed=42, single seed
since this calibrator is deterministic and low-variance):
    For each fold k:
        - Train fold (~202 rows):
            * Compute affine map (p_min, p_max) from train preds.
            * Compute affine map (y_min, y_max) from train truth.
            * Fit (a, b, c) by minimizing MSE on z-space.
        - Apply to held-out fold (~51 rows); collect OOF.
    Pooled OOF RAE = headline.

Compare to:
    nb1070 per-quantile median bag: 0.5771
    raw te_nb1014 in_RAE on 253:    (computed below)

We ALSO run a 5-seed pooled cross-fit to mirror nb1070's seed-bagged headline
for an apples-to-apples comparison (median across seeds of per-fold OOFs).

Outputs
-------
    data/processed/te_nb1082.npy             (513 deploy)
    data/processed/nb1082_summary.json
    submissions/nb1082_beta_calibration.csv
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
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1082"
ANCHOR_STEM = "nb1014"
SEEDS = [0, 1, 7, 42, 137]
N_FOLDS = 5

NB1070_BAG_MEDIAN_REF = 0.5771

# Affine padding -- avoid p=0 or p=1 (logit blows up).
EPS = 1e-3


# =====================================================================
# Beta calibration -- regression variant
# =====================================================================
def _affine(p: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map p in [lo, hi] to [eps, 1-eps]; clip out-of-range gracefully."""
    rng = hi - lo
    if rng < 1e-9:
        rng = 1e-9
    out = (p - lo) / rng
    out = np.clip(out, EPS, 1.0 - EPS)
    return out


def _logit(z: np.ndarray) -> np.ndarray:
    return np.log(z / (1.0 - z))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fit_beta_cal(p_train: np.ndarray, y_train: np.ndarray
                 ) -> tuple[float, float, float, float, float, float, float]:
    """Fit Beta calibration on a train fold.

    Returns (a, b, c, p_lo, p_hi, y_lo, y_hi) -- the seven scalars
    needed to transform new preds.
    """
    p_lo, p_hi = float(p_train.min()), float(p_train.max())
    y_lo, y_hi = float(y_train.min()), float(y_train.max())

    p_norm = _affine(p_train, p_lo, p_hi)
    y_norm = _affine(y_train, y_lo, y_hi)

    log_p = np.log(p_norm)
    log_1mp = np.log(1.0 - p_norm)
    z_target = _logit(y_norm)  # MSE in logit space -> stable optim.

    def loss(params: np.ndarray) -> float:
        a, b, c = params
        z = a * log_p - b * log_1mp + c
        return float(np.mean((z - z_target) ** 2))

    # Start at the identity-ish point (a=1, b=1, c=0 -> proportional to logit p).
    res = minimize(
        loss, np.array([1.0, 1.0, 0.0]),
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 5000},
    )
    a, b, c = res.x
    return float(a), float(b), float(c), p_lo, p_hi, y_lo, y_hi


def apply_beta_cal(p: np.ndarray, a: float, b: float, c: float,
                   p_lo: float, p_hi: float, y_lo: float, y_hi: float
                   ) -> np.ndarray:
    p_norm = _affine(p, p_lo, p_hi)
    log_p = np.log(p_norm)
    log_1mp = np.log(1.0 - p_norm)
    z = a * log_p - b * log_1mp + c
    y_norm = _sigmoid(z)
    return y_lo + (y_hi - y_lo) * y_norm


# =====================================================================
# Cross-fit driver
# =====================================================================
def one_seed(p_unb: np.ndarray, y_unb: np.ndarray, seed: int
             ) -> tuple[float, np.ndarray, list[dict]]:
    n = len(y_unb)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_records: list[dict] = []
    for k, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n))):
        a, b, c, p_lo, p_hi, y_lo, y_hi = fit_beta_cal(
            p_unb[tr_loc], y_unb[tr_loc]
        )
        oof[va_loc] = apply_beta_cal(p_unb[va_loc], a, b, c,
                                     p_lo, p_hi, y_lo, y_hi)
        rae_va = float(rae(y_unb[va_loc], oof[va_loc]))
        rae_tr = float(rae(
            y_unb[tr_loc],
            apply_beta_cal(p_unb[tr_loc], a, b, c, p_lo, p_hi, y_lo, y_hi),
        ))
        fold_records.append({
            "fold": k, "a": a, "b": b, "c": c,
            "p_lo": p_lo, "p_hi": p_hi, "y_lo": y_lo, "y_hi": y_hi,
            "train_rae": rae_tr, "val_rae": rae_va,
            "n_tr": int(len(tr_loc)), "n_va": int(len(va_loc)),
        })
    pooled = float(rae(y_unb, oof))
    return pooled, oof, fold_records


# =====================================================================
# Main
# =====================================================================
def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Beta calibration (Kull 2017) on te_{ANCHOR_STEM}")
    print(f"      seeds = {SEEDS}  N_FOLDS = {N_FOLDS}")
    print("=" * 78)

    te = load_test()
    te_names = te["name"].values
    preds_513 = np.load(DATA_PROCESSED / f"te_{ANCHOR_STEM}.npy").astype(
        np.float64)
    print(f"[load] te_{ANCHOR_STEM}.npy shape = {preds_513.shape}  "
          f"mean={preds_513.mean():.3f} std={preds_513.std():.3f}")

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    p_unb = preds_513[unb_idx]
    print(f"[load] p_unb shape = {p_unb.shape}  y shape = {y_unb.shape}")

    raw_in_rae = float(rae(y_unb, p_unb))
    print(f"\n[anchor] raw te_{ANCHOR_STEM} in_RAE(253) = {raw_in_rae:.4f}  "
          "(pre-calibration baseline)")

    # -------- Cross-fit per seed --------
    print("\n" + "-" * 78)
    print(f"Honest {N_FOLDS}-fold cross-fit over {len(SEEDS)} seeds")
    print("-" * 78)
    oof_stack = np.zeros((len(SEEDS), len(y_unb)), dtype=np.float64)
    per_seed_rae: list[float] = []
    seed_records: list[dict] = []
    for i, seed in enumerate(SEEDS):
        pooled, oof, folds = one_seed(p_unb, y_unb, seed)
        oof_stack[i] = oof
        per_seed_rae.append(pooled)
        seed_records.append({"seed": seed, "pooled_rae": pooled,
                             "folds": folds})
        a_arr = np.array([f["a"] for f in folds])
        b_arr = np.array([f["b"] for f in folds])
        c_arr = np.array([f["c"] for f in folds])
        print(f"   seed {seed:>3d}: pooled_RAE = {pooled:.4f}   "
              f"a_mean={a_arr.mean():.3f} b_mean={b_arr.mean():.3f} "
              f"c_mean={c_arr.mean():.3f}")

    per_seed_arr = np.array(per_seed_rae)
    seed_mean = float(per_seed_arr.mean())
    seed_std = float(per_seed_arr.std())
    seed_min = float(per_seed_arr.min())
    seed_max = float(per_seed_arr.max())
    print(f"\n[CV] per-seed RAE  mean={seed_mean:.4f}  std={seed_std:.4f}  "
          f"min={seed_min:.4f}  max={seed_max:.4f}")

    bagged_median_oof = np.median(oof_stack, axis=0)
    bagged_mean_oof = oof_stack.mean(axis=0)
    bag_median_rae = float(rae(y_unb, bagged_median_oof))
    bag_mean_rae = float(rae(y_unb, bagged_mean_oof))
    print(f"[CV] MEDIAN-bag OOF RAE = {bag_median_rae:.4f}  (HEADLINE)")
    print(f"[CV] MEAN-bag OOF RAE   = {bag_mean_rae:.4f}")

    # -------- Deploy: fit on ALL 253, apply to 513 (one fit per seed -> median)
    print("\n" + "-" * 78)
    print("DEPLOY  (fit on all 253 per seed -- but Beta cal is deterministic;")
    print("         we fit ONCE on full 253 and report that for 513)")
    print("-" * 78)
    a, b, c, p_lo, p_hi, y_lo, y_hi = fit_beta_cal(p_unb, y_unb)
    deploy_253 = apply_beta_cal(p_unb, a, b, c, p_lo, p_hi, y_lo, y_hi)
    in_rae_deploy = float(rae(y_unb, deploy_253))
    deploy_513 = apply_beta_cal(preds_513, a, b, c, p_lo, p_hi, y_lo, y_hi
                                ).astype(np.float32)
    print(f"   deploy (a,b,c) = ({a:.4f}, {b:.4f}, {c:.4f})")
    print(f"   anchor norm:   p in [{p_lo:.3f}, {p_hi:.3f}]   "
          f"y in [{y_lo:.3f}, {y_hi:.3f}]")
    print(f"   in-sample RAE(253) = {in_rae_deploy:.4f}  (lower bound)")
    print(f"   deploy_513 mean = {deploy_513.mean():.3f}  "
          f"std = {deploy_513.std():.3f}  "
          f"min = {deploy_513.min():.3f}  max = {deploy_513.max():.3f}")

    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy_513)
    plain = SUBMISSIONS / f"{TAG}_beta_calibration.csv"
    pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te_names,
        "pEC50": deploy_513,
    }).to_csv(plain, index=False)
    print(f"\n[save] te_{TAG}.npy")
    print(f"[save] {plain}")

    # -------- Verdicts --------
    delta_vs_nb1070 = bag_median_rae - NB1070_BAG_MEDIAN_REF
    delta_vs_raw = bag_median_rae - raw_in_rae
    beats_nb1070 = bool(bag_median_rae < NB1070_BAG_MEDIAN_REF)
    if delta_vs_nb1070 < -0.003:
        verdict = "BEATS_NB1070"
    elif abs(delta_vs_nb1070) <= 0.003:
        verdict = "TIES_NB1070"
    else:
        verdict = "WORSE_THAN_NB1070"

    print("\n" + "-" * 78)
    print("VERDICT")
    print("-" * 78)
    print(f"   raw te_{ANCHOR_STEM} in_RAE(253)   = {raw_in_rae:.4f}")
    print(f"   per-seed mean cross-fit RAE   = {seed_mean:.4f}  "
          f"std {seed_std:.4f}")
    print(f"   MEDIAN-bagged OOF RAE         = {bag_median_rae:.4f}  "
          "(HEADLINE)")
    print(f"   nb1070 per-q median bag ref   = {NB1070_BAG_MEDIAN_REF:.4f}  "
          f"delta = {delta_vs_nb1070:+.4f}")
    print(f"   vs raw anchor                 = {delta_vs_raw:+.4f}")
    print(f"   beats_nb1070                  = {beats_nb1070}")
    print(f"   verdict                       = {verdict}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR_STEM,
        "seeds": SEEDS,
        "n_folds": N_FOLDS,
        "raw_anchor_in_rae_253": raw_in_rae,
        "per_seed_pooled_rae": per_seed_rae,
        "per_seed_rae_mean": seed_mean,
        "per_seed_rae_std": seed_std,
        "per_seed_rae_min": seed_min,
        "per_seed_rae_max": seed_max,
        "bag_median_rae": bag_median_rae,
        "bag_mean_rae": bag_mean_rae,
        "seed_records": seed_records,
        "deploy_a": a,
        "deploy_b": b,
        "deploy_c": c,
        "deploy_p_lo": p_lo,
        "deploy_p_hi": p_hi,
        "deploy_y_lo": y_lo,
        "deploy_y_hi": y_hi,
        "deploy_in_rae_253": in_rae_deploy,
        "deploy_te_mean": float(deploy_513.mean()),
        "deploy_te_std": float(deploy_513.std()),
        "deploy_te_min": float(deploy_513.min()),
        "deploy_te_max": float(deploy_513.max()),
        "nb1070_reference": NB1070_BAG_MEDIAN_REF,
        "delta_vs_nb1070": delta_vs_nb1070,
        "delta_vs_raw_anchor": delta_vs_raw,
        "beats_nb1070": beats_nb1070,
        "verdict": verdict,
        "plain_submission": str(plain),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("raw_anchor_in_rae_253",
              "per_seed_pooled_rae", "per_seed_rae_mean",
              "bag_median_rae", "bag_mean_rae",
              "deploy_a", "deploy_b", "deploy_c",
              "deploy_in_rae_253",
              "delta_vs_nb1070", "beats_nb1070", "verdict",
              "plain_submission"):
        print(f"  {k}: {res.get(k)}")
