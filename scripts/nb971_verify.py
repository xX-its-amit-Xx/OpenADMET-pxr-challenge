"""nb971 verification — independent OUTER 5-fold cross-fit reproducibility check.

GOAL
    Verify the claimed cross-fit RAE 0.4675 from nb971_slsqp_6anchor.py is a
    true OUTER cross-fit (predictions on held-out fold), not in-sample SLSQP.

Protocol:
  1. Reload all 6 anchors + y_unb (253) + unb_idx using exact nb971 pipeline.
  2. Sanity: recompute per-anchor honest RAE.
  3. Re-run OUTER 5-fold cross-fit SLSQP (mae + huber); confirm reproducibility.
  4. Robustness: run OUTER 5-fold with multiple seeds (0, 1, 7, 42, 123) and
     report mean + std cross-fit RAE.
  5. In-sample baseline (single SLSQP fit on full 253) for the gap.
  6. Write verification report data/processed/nb971_verify_summary.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
SUB = ROOT / "submissions"
RAW = ROOT / "data" / "raw"
warnings.filterwarnings("ignore")

NB2112_FLOOR = 0.4698
IMPROVEMENT_DELTA = 0.005
HUBER_DELTA = 0.5
N_FOLDS = 5
SEEDS = [0, 1, 7, 42, 123]

ANCHORS = [
    {"name": "chemprop_aux", "te_path": "data/processed/te_chemprop_aux.npy", "unb_source": "te_unb_idx"},
    {"name": "nb503",        "te_path": "data/processed/te_nb503.npy",        "unb_source": "data/processed/nb503_pred_oof.npy"},
    {"name": "nb464",        "te_path": "data/processed/te_nb464.npy",        "unb_source": "te_unb_idx"},
    {"name": "nb471",        "te_path": "data/processed/te_nb471.npy",        "unb_source": "te_unb_idx"},
    {"name": "nb432",        "te_path": "data/processed/te_nb432.npy",        "unb_source": "te_unb_idx"},
    {"name": "nb2112",       "te_path": "data/processed/te_nb2112.npy",       "unb_source": "data/processed/nb2103_mean_bag_oof_K28.npy"},
]


def rae(y, p):
    return float(np.abs(y - p).sum() / np.abs(y - y.mean()).sum())


def mae(y, p):
    return float(np.abs(y - p).mean())


def huber(y, p, delta=HUBER_DELTA):
    r = np.abs(y - p)
    quad = np.minimum(r, delta)
    lin = r - quad
    return float((0.5 * quad ** 2 + delta * lin).mean())


def slsqp_blend(X, y, loss_fn):
    n = X.shape[1]
    w0 = np.ones(n) / n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bnds = [(0.0, 1.0)] * n
    res = minimize(lambda w: loss_fn(y, X @ w), w0, method="SLSQP",
                   bounds=bnds, constraints=cons,
                   options={"ftol": 1e-9, "maxiter": 500})
    w = np.clip(res.x, 0.0, None)
    return w / w.sum() if w.sum() > 0 else w


def outer_cv(X, y, loss_fn, n_folds=N_FOLDS, seed=0):
    """Explicit OUTER cross-fit: fit weights on 4/5, predict on 1/5 held-out."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    pred_oof = np.zeros_like(y, dtype=float)
    fold_w = []
    for tr, va in kf.split(X):
        w = slsqp_blend(X[tr], y[tr], loss_fn)
        pred_oof[va] = X[va] @ w
        fold_w.append(w)
    return pred_oof, np.asarray(fold_w)


def sha8(arr):
    return hashlib.sha256(arr.tobytes()).hexdigest()[:8]


def main():
    y_unb = np.load(PROC / "_audit_unblind_y.npy")
    unb_idx = np.load(PROC / "_audit_unblind_idx.npy")
    assert y_unb.shape == (253,)
    print(f"y_unb: n={len(y_unb)}, mean={y_unb.mean():.4f}, std={y_unb.std():.4f}, sha8={sha8(y_unb)}")

    cols_unb, cols_513 = [], []
    per_anchor = {}
    for a in ANCHORS:
        te = np.load(ROOT / a["te_path"])
        assert te.shape == (513,)
        if a["unb_source"] == "te_unb_idx":
            pred_unb = te[unb_idx]
        else:
            pred_unb = np.load(ROOT / a["unb_source"])
            assert pred_unb.shape == (253,)
        r = rae(y_unb, pred_unb)
        pear = float(np.corrcoef(pred_unb, y_unb)[0, 1])
        per_anchor[a["name"]] = {
            "rae": round(r, 4),
            "pearson": round(pear, 4),
            "sha8": sha8(pred_unb),
            "mean": float(pred_unb.mean()),
            "std": float(pred_unb.std()),
        }
        cols_unb.append(pred_unb)
        cols_513.append(te)
        print(f"  {a['name']:<14}  RAE={r:.4f}  pearson={pear:.3f}  sha={sha8(pred_unb)}")

    X_unb = np.column_stack(cols_unb)
    X_513 = np.column_stack(cols_513)
    names = [a["name"] for a in ANCHORS]

    # In-sample baseline (one-shot)
    insample = {}
    for ln, lf in [("mae", mae), ("huber", huber)]:
        w = slsqp_blend(X_unb, y_unb, lf)
        rae_in = rae(y_unb, X_unb @ w)
        insample[ln] = {"weights": [round(v, 4) for v in w.tolist()], "rae": round(rae_in, 4)}
        print(f"\n[in-sample {ln}] RAE={rae_in:.4f} w={dict(zip(names, np.round(w, 3).tolist()))}")

    # OUTER 5-fold cross-fit, primary seed=0 (matches nb971)
    print("\n--- OUTER 5-fold cross-fit (seed=0, matches nb971 protocol) ---")
    primary = {}
    for ln, lf in [("mae", mae), ("huber", huber)]:
        pred_cf, fw = outer_cv(X_unb, y_unb, lf, seed=0)
        rae_cf = rae(y_unb, pred_cf)
        primary[ln] = {
            "rae_crossfit": round(rae_cf, 4),
            "mean_fold_w": [round(v, 4) for v in fw.mean(axis=0).tolist()],
        }
        print(f"[outer-CV seed=0 {ln}] RAE={rae_cf:.4f}  mean_w={dict(zip(names, np.round(fw.mean(axis=0), 3).tolist()))}")

    # Multi-seed robustness
    print("\n--- Multi-seed OUTER CV (robustness check) ---")
    multi_seed = {}
    for ln, lf in [("mae", mae), ("huber", huber)]:
        raes = []
        for s in SEEDS:
            pred_cf, _ = outer_cv(X_unb, y_unb, lf, seed=s)
            raes.append(rae(y_unb, pred_cf))
        multi_seed[ln] = {
            "seeds": SEEDS,
            "raes": [round(r, 4) for r in raes],
            "mean": round(float(np.mean(raes)), 4),
            "std": round(float(np.std(raes)), 4),
            "max": round(float(np.max(raes)), 4),
        }
        print(f"[multi-seed {ln}] raes={[round(r,4) for r in raes]} mean={np.mean(raes):.4f}+-{np.std(raes):.4f}  max={np.max(raes):.4f}")

    # Verdict
    claimed_cf = 0.4675
    true_cf_mae = primary["mae"]["rae_crossfit"]
    true_cf_huber = primary["huber"]["rae_crossfit"]
    best_cf = min(true_cf_mae, true_cf_huber)
    best_loss = "mae" if true_cf_mae <= true_cf_huber else "huber"
    target = NB2112_FLOOR - IMPROVEMENT_DELTA
    multi_mean = multi_seed[best_loss]["mean"]
    multi_max = multi_seed[best_loss]["max"]

    matches_claim = abs(true_cf_mae - claimed_cf) < 0.001
    beats_floor_strict = best_cf <= target            # 0.005 mRAE delta
    beats_floor_loose = best_cf <= NB2112_FLOOR       # bare floor
    robust_across_seeds = multi_max <= NB2112_FLOOR   # even worst seed beats floor

    print("\n=== VERDICT ===")
    print(f"claimed cross-fit RAE      : {claimed_cf:.4f}")
    print(f"reproduced (mae, seed=0)   : {true_cf_mae:.4f}  matches={matches_claim}")
    print(f"reproduced (huber, seed=0) : {true_cf_huber:.4f}")
    print(f"best (loss={best_loss})           : {best_cf:.4f}")
    print(f"floor (nb2112)             : {NB2112_FLOOR}")
    print(f"target (floor - 0.005)     : {target:.4f}")
    print(f"multi-seed mean (best loss): {multi_mean:.4f}")
    print(f"multi-seed max (worst case): {multi_max:.4f}")
    print(f"beats target by >= 0.005   : {beats_floor_strict}")
    print(f"beats bare floor           : {beats_floor_loose}")
    print(f"robust (worst seed beats floor): {robust_across_seeds}")

    if beats_floor_strict and robust_across_seeds:
        verdict = "DEPLOY -- beats floor by >= 5 mRAE across all seeds"
        action = "WRITE deploy CSV submissions/nb971_deploy_slsqp6.csv"
    elif beats_floor_loose:
        verdict = "MARGINAL -- in claimed range but inside policy delta; HOLD"
        action = "DO NOT DEPLOY (within margin); keep nb2112 as PRIMARY-1"
    else:
        verdict = "IN-SAMPLE-SLSQP-OVERFIT or REGRESSION -- fails honest floor"
        action = "REJECT nb971; keep nb2112 as PRIMARY-1"
    print(f"\nVerdict: {verdict}\nAction : {action}")

    summary = {
        "tag": "nb971_verify",
        "claimed_crossfit_rae": claimed_cf,
        "floor": NB2112_FLOOR,
        "target_rae": target,
        "improvement_delta_required": IMPROVEMENT_DELTA,
        "per_anchor_honest_rae": per_anchor,
        "in_sample_slsqp": insample,
        "outer_cv_seed0": primary,
        "multi_seed_outer_cv": multi_seed,
        "best_outer_cv_rae": best_cf,
        "best_loss": best_loss,
        "claim_matches_reproduction": matches_claim,
        "beats_floor_strict_005": beats_floor_strict,
        "beats_floor_loose": beats_floor_loose,
        "robust_across_seeds": robust_across_seeds,
        "verdict": verdict,
        "ladder_action": action,
    }
    out = PROC / "nb971_verify_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
