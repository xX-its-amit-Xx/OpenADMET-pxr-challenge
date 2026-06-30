"""nb1080 -- SLSQP convex blend over 5 PRE-unblind PRIMARY-tier deploys.

5 anchors:
  1. nb2112             (chemprop_aux + LGBM K=28, honest median 0.4698)
  2. nb1660_median      (cycle 119 deploy, 0.5107)
  3. nb1014             (multi-seed bag, 0.5871-0.5930)
  4. chemprop_aux       (raw anchor, in_RAE 0.6216)
  5. nb1001             (cycle 6 stretch, 0.5994)

Protocol:
  * Confirm all 5 te files have sha256 != y_unb sha256 (PRE-unblind).
  * Evaluate each te[unb_idx] in_RAE on 253 as a sanity check.
  * 5-fold OUTER cross-fit (KFold seed=42 for reproducibility):
      For each fold:
        - Solve SLSQP convex blend (w >= 0, sum=1) on 4 train folds
          for THREE loss objectives: MAE, MSE, Huber(d=0.5).
        - Apply weights to held-out fold; collect OOF predictions.
  * Report pooled cross-fit RAE for each loss vs nb2112 (0.4698).
  * Report blend weights (mean across folds) and check collapse.
  * If best variant beats nb2112 by >= 0.003:
      build deploy CSV submissions/nb1080_deploy_slsqp5.csv
      by refitting SLSQP weights on ALL 253 (deploy-refit weights)
      then applying to the 513 te files.

Outputs:
  scripts/nb1080_slsqp_5anchor.py
  data/processed/nb1080_summary.json
  submissions/nb1080_deploy_slsqp5.csv  (if beats nb2112)
"""
from __future__ import annotations

import hashlib
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
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1080"
NB2112_REF = 0.4698  # honest median bag reference from nb2112_summary.json
BEAT_THRESHOLD = 0.003
N_OUTER_FOLDS = 5
OUTER_SEED = 42
HUBER_DELTA = 0.5

# 5 anchors: (label, te_file_basename)
ANCHORS = [
    ("nb2112",         "te_nb2112.npy"),
    ("nb1660_median",  "te_nb1660_median.npy"),
    ("nb1014",         "te_nb1014.npy"),
    ("chemprop_aux",   "te_chemprop_aux.npy"),
    ("nb1001",         "te_nb1001.npy"),
]


def sha16(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def slsqp_blend(P: np.ndarray, y: np.ndarray, loss: str) -> np.ndarray:
    """Solve convex blend w >= 0, sum(w)=1 minimizing loss(P @ w, y).

    loss in {"mae", "mse", "huber"}
    """
    K = P.shape[1]
    w0 = np.full(K, 1.0 / K)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * K

    if loss == "mse":
        def f(w):
            r = P @ w - y
            return float(np.mean(r * r))
    elif loss == "mae":
        def f(w):
            return float(np.mean(np.abs(P @ w - y)))
    elif loss == "huber":
        def f(w):
            r = P @ w - y
            a = np.abs(r)
            quad = 0.5 * r * r
            lin = HUBER_DELTA * (a - 0.5 * HUBER_DELTA)
            return float(np.mean(np.where(a <= HUBER_DELTA, quad, lin)))
    else:
        raise ValueError(loss)

    res = minimize(
        f, w0, method="SLSQP", bounds=bnds, constraints=cons,
        options={"ftol": 1e-10, "maxiter": 1000},
    )
    w = np.asarray(res.x, dtype=np.float64)
    w = np.clip(w, 0.0, 1.0)
    w /= max(w.sum(), 1e-12)
    return w


def cross_fit_one_loss(P_unb: np.ndarray, y_unb: np.ndarray,
                       loss: str) -> dict:
    """Pooled 5-fold cross-fit for given loss objective."""
    n = len(y_unb)
    kf = KFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=OUTER_SEED)
    oof = np.full(n, np.nan)
    all_w = []
    fold_recs = []
    for k, (tr, va) in enumerate(kf.split(np.arange(n))):
        w = slsqp_blend(P_unb[tr], y_unb[tr], loss)
        oof[va] = P_unb[va] @ w
        all_w.append(w)
        fold_recs.append({
            "fold": int(k),
            "n_tr": int(len(tr)),
            "n_va": int(len(va)),
            "w": [float(x) for x in w],
            "val_rae": float(rae(y_unb[va], oof[va])),
        })
    pooled = float(rae(y_unb, oof))
    mean_w = np.mean(np.stack(all_w, axis=0), axis=0)
    std_w = np.std(np.stack(all_w, axis=0), axis=0)
    return {
        "loss": loss,
        "pooled_rae": pooled,
        "mean_w": [float(x) for x in mean_w],
        "std_w": [float(x) for x in std_w],
        "folds": fold_recs,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- SLSQP convex blend over 5 PRE-unblind PRIMARY-tier deploys")
    print("=" * 78)

    # ---- Load 253 unblind ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    y_sha = sha16(y_unb)
    print(f"[load] y_unb shape={y_unb.shape} sha[:16]={y_sha}")

    # ---- Load 513 te files for all 5 anchors ----
    te_arrs = {}
    pre_unblind_audit = {}
    for label, fname in ANCHORS:
        arr = np.load(DATA_PROCESSED / fname).astype(np.float64)
        assert arr.shape == (513,), f"{label}: te shape {arr.shape} != (513,)"
        te_arrs[label] = arr
        sub = arr[unb_idx]
        is_pre = sha16(sub) != y_sha
        pre_unblind_audit[label] = {
            "te_path": str(DATA_PROCESSED / fname),
            "sha513": sha16(arr),
            "sha253_at_unb_idx": sha16(sub),
            "is_PRE_unblind_distinct_from_y": bool(is_pre),
        }
        print(f"[audit] {label:18s} te513_sha={sha16(arr)}  "
              f"te[unb]_sha={sha16(sub)}  PRE_distinct_from_y={is_pre}")
        if not is_pre:
            raise RuntimeError(f"{label} sha matches y_unb -- not PRE-unblind!")

    # ---- Stack into (253, 5) ----
    P_unb = np.column_stack([te_arrs[label][unb_idx] for label, _ in ANCHORS])
    P_513 = np.column_stack([te_arrs[label] for label, _ in ANCHORS])

    # ---- Individual in_RAE sanity on 253 ----
    print("\n[indiv] in_RAE on 253 unblind:")
    indiv_rae = {}
    for j, (label, _) in enumerate(ANCHORS):
        r = float(rae(y_unb, P_unb[:, j]))
        indiv_rae[label] = r
        print(f"   {label:18s}: in_RAE={r:.4f}")

    # ---- Cross-fit for 3 loss variants ----
    print("\n" + "-" * 78)
    print(f"5-FOLD OUTER CROSS-FIT (seed={OUTER_SEED})")
    print("-" * 78)
    results = {}
    for loss in ("mae", "mse", "huber"):
        r = cross_fit_one_loss(P_unb, y_unb, loss)
        results[loss] = r
        w_str = ", ".join(f"{x:.3f}" for x in r["mean_w"])
        mw = max(r["mean_w"])
        collapsed = mw >= 0.95
        print(f"   loss={loss:6s}  pooled_RAE={r['pooled_rae']:.4f}  "
              f"mean_w=[{w_str}]  max_w={mw:.3f}  "
              f"{'COLLAPSED_TO_SINGLE' if collapsed else 'multi-anchor'}")

    # ---- Pick best loss; compare vs nb2112 0.4698 ----
    best_loss = min(results, key=lambda k: results[k]["pooled_rae"])
    best_rae = results[best_loss]["pooled_rae"]
    delta = best_rae - NB2112_REF
    if delta <= -BEAT_THRESHOLD:
        verdict = "BEATS_NB2112"
    elif delta < 0:
        verdict = "SMALL_GAIN_BELOW_THRESHOLD"
    elif abs(delta) <= BEAT_THRESHOLD:
        verdict = "TIES_NB2112"
    else:
        verdict = "WORSE_THAN_NB2112"
    print(f"\n[verdict] best loss = {best_loss}  pooled_RAE={best_rae:.4f}  "
          f"vs nb2112 ref {NB2112_REF:.4f}  delta={delta:+.4f}  -> {verdict}")

    # ---- Deploy-refit on ALL 253 if BEATS ----
    deploy_info = {"built": False, "reason": verdict}
    csv_path = None
    if verdict == "BEATS_NB2112":
        print("\n" + "-" * 78)
        print(f"DEPLOY-REFIT: SLSQP on ALL 253 with loss={best_loss}")
        print("-" * 78)
        w_deploy = slsqp_blend(P_unb, y_unb, best_loss)
        in_sample_rae = float(rae(y_unb, P_unb @ w_deploy))
        pred_513 = (P_513 @ w_deploy).astype(np.float32)
        print(f"   deploy w = {[round(float(x),4) for x in w_deploy]}")
        print(f"   in-sample RAE (253) = {in_sample_rae:.4f}  "
              "(overfit lower bound)")
        print(f"   te(513) mean/std    = "
              f"{pred_513.mean():.3f} / {pred_513.std():.3f}")

        te = load_test()
        te_names = te["name"].values
        smiles = te["smiles"].values
        csv_path = SUBMISSIONS / f"{TAG}_deploy_slsqp5.csv"
        pd.DataFrame({
            "SMILES": smiles,
            "Molecule Name": te_names,
            "pEC50": pred_513,
        }).to_csv(csv_path, index=False)
        np.save(DATA_PROCESSED / f"te_{TAG}.npy", pred_513)
        print(f"[save] {csv_path}")
        print(f"[save] te_{TAG}.npy")
        deploy_info = {
            "built": True,
            "csv_path": str(csv_path),
            "te_npy_path": str(DATA_PROCESSED / f"te_{TAG}.npy"),
            "loss_used": best_loss,
            "w_deploy": [float(x) for x in w_deploy],
            "in_sample_rae_overfit_bound": in_sample_rae,
            "te_mean": float(pred_513.mean()),
            "te_std": float(pred_513.std()),
        }
    else:
        print(f"\n[skip-deploy] verdict={verdict} -- not building CSV "
              f"(only build if BEATS_NB2112 by >= {BEAT_THRESHOLD})")

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "anchors": [label for label, _ in ANCHORS],
        "pre_unblind_audit": pre_unblind_audit,
        "indiv_in_rae_253": indiv_rae,
        "n_outer_folds": N_OUTER_FOLDS,
        "outer_seed": OUTER_SEED,
        "huber_delta": HUBER_DELTA,
        "loss_results": results,
        "best_loss": best_loss,
        "best_pooled_rae": best_rae,
        "nb2112_ref": NB2112_REF,
        "delta_vs_nb2112": delta,
        "beat_threshold": BEAT_THRESHOLD,
        "verdict": verdict,
        "deploy": deploy_info,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_json = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_json}")

    # ---- Final summary ----
    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   anchors                = {[label for label,_ in ANCHORS]}")
    print(f"   indiv in_RAE (253)     = "
          f"{ {k: round(v,4) for k,v in indiv_rae.items()} }")
    for loss in ("mae", "mse", "huber"):
        r = results[loss]
        w_str = "[" + ", ".join(f"{x:.3f}" for x in r["mean_w"]) + "]"
        print(f"   loss={loss:6s} pooled_RAE = {r['pooled_rae']:.4f}  "
              f"mean_w={w_str}")
    print(f"   best loss              = {best_loss}  "
          f"pooled_RAE={best_rae:.4f}")
    print(f"   vs nb2112 ref 0.4698   = delta {delta:+.4f}  -> {verdict}")
    print(f"   deploy CSV built       = {deploy_info['built']}")
    if deploy_info["built"]:
        print(f"   deploy CSV path        = {deploy_info['csv_path']}")
    print(f"   wall                   = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
