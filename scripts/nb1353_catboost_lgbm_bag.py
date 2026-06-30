"""nb1353 -- Combined CatBoost + LGBM 10-model bag on the unblind 253.

Hypothesis:
    nb1341 (CatBoost MAE residual) and nb1242 (LGBM Huber residual) sit on
    the same MACCS-167 + ChEMBL kNN feature substrate, the same nb1070
    anchor, the same 5 cross-fit seeds [0, 1, 7, 42, 137], and their
    mean-bag OOFs Pearson at 0.9900 (nb1341 summary).  Despite the high
    correlation, the two learners use different gradient schemes (Ordered
    Boosting symmetric trees vs leaf-wise) and different losses (MAE vs
    Huber alpha=1.0).  Their corrected-OOF residuals may carry a sliver
    of independent noise.  Pooling all 10 per-seed corrected OOFs into a
    single 10-model bag (mean and median) may extract a last variance-
    reduction increment relative to either 5-bag in isolation.

Inputs (all pre-computed; no LGBM refit needed):
    data/processed/nb1341_per_seed_corrected_oof.npy   (5, 253) float32
    data/processed/nb1242_per_seed_corrected_oof.npy   (5, 253) float32
    data/processed/_audit_unblind_y.npy                (253,)   float64
References:
    nb1242 mean-bag RAE = 0.5431  (anchor-corrected LGBM Huber bag)
    nb1290 best blend   = 0.5390  (1190 + 1242 fixed-w 0.35/0.65)
    nb1341 mean-bag RAE = 0.5420  (anchor-corrected CatBoost MAE bag)

Pipeline:
    1. Stack nb1341 (5, 253) and nb1242 (5, 253) -> (10, 253).
    2. combined_mean   = mean across axis 0  -> (253,)
    3. combined_median = median across axis 0 -> (253,)
    4. Pool RAE of both vs y_unb.
    5. Verdict at 0.003 margin vs nb1242 (0.5431) and nb1290 (0.5390).

Outputs:
    scripts/nb1353_catboost_lgbm_bag.py            (this file)
    data/processed/nb1353_summary.json
    data/processed/nb1353_combined_oof.npy         (253,) float32  (mean bag)
    data/processed/nb1353_combined_median_oof.npy  (253,) float32  (median bag)
"""
from __future__ import annotations

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
from scipy.stats import pearsonr

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1353"
NB1242_REF = 0.5431
NB1290_REF = 0.5390
NB1341_REF = 0.5420
DECISION_MARGIN = 0.003


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Combined CatBoost (5) + LGBM (5) = 10-model bag")
    print(f"          refs: nb1242 {NB1242_REF:.4f}  nb1290 {NB1290_REF:.4f}")
    print("=" * 78)

    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] y_unb shape={y_unb.shape} mean={y_unb.mean():.3f} "
          f"std={y_unb.std():.3f}")

    cb_path = DATA_PROCESSED / "nb1341_per_seed_corrected_oof.npy"
    lg_path = DATA_PROCESSED / "nb1242_per_seed_corrected_oof.npy"
    if not cb_path.exists():
        raise FileNotFoundError(f"missing CatBoost per-seed OOF: {cb_path}")
    if not lg_path.exists():
        raise FileNotFoundError(f"missing LGBM per-seed OOF: {lg_path}")

    cb = np.load(cb_path).astype(np.float64)  # (5, 253)
    lg = np.load(lg_path).astype(np.float64)  # (5, 253)
    if cb.shape != (5, n_unb):
        raise ValueError(f"nb1341 shape {cb.shape} != (5, {n_unb})")
    if lg.shape != (5, n_unb):
        raise ValueError(f"nb1242 shape {lg.shape} != (5, {n_unb})")
    print(f"[load] nb1341 CatBoost: {cb.shape}")
    print(f"[load] nb1242 LGBM   : {lg.shape}")

    # ---- Per-model RAE for the books ----
    per_model_rae_cb = [float(rae(y_unb, cb[i])) for i in range(5)]
    per_model_rae_lg = [float(rae(y_unb, lg[i])) for i in range(5)]
    print("\n" + "-" * 78)
    print("PER-MODEL RAE")
    print("-" * 78)
    for i, r in enumerate(per_model_rae_cb):
        print(f"   CatBoost seed {[0,1,7,42,137][i]:3d}: rae = {r:.4f}")
    for i, r in enumerate(per_model_rae_lg):
        print(f"   LGBM     seed {[0,1,7,42,137][i]:3d}: rae = {r:.4f}")

    # ---- Stack to 10 models ----
    combined = np.vstack([cb, lg])  # (10, 253)
    assert combined.shape == (10, n_unb)
    print(f"\n[stack] combined shape = {combined.shape}")

    combined_mean_oof = combined.mean(axis=0)
    combined_median_oof = np.median(combined, axis=0)
    rae_mean = float(rae(y_unb, combined_mean_oof))
    rae_median = float(rae(y_unb, combined_median_oof))

    # 5-bag references re-computed in-place for sanity
    rae_cb_5bag = float(rae(y_unb, cb.mean(axis=0)))
    rae_lg_5bag = float(rae(y_unb, lg.mean(axis=0)))

    # Pearson between the two 5-bag means (decorrelation probe)
    pearson_cb_lg_5bag, _ = pearsonr(cb.mean(axis=0), lg.mean(axis=0))
    pearson_cb_lg_5bag = float(pearson_cb_lg_5bag)

    print("\n" + "-" * 78)
    print("BAG AGGREGATIONS")
    print("-" * 78)
    print(f"   CatBoost 5-bag mean  RAE = {rae_cb_5bag:.4f}  "
          f"(summary ref {NB1341_REF:.4f})")
    print(f"   LGBM     5-bag mean  RAE = {rae_lg_5bag:.4f}  "
          f"(summary ref {NB1242_REF:.4f})")
    print(f"   Pearson(cb_5bag, lg_5bag) = {pearson_cb_lg_5bag:.4f}")
    print(f"   COMBINED 10-bag mean   RAE = {rae_mean:.4f}  "
          f"(d_vs_nb1242 {rae_mean - NB1242_REF:+.4f}  "
          f"d_vs_nb1290 {rae_mean - NB1290_REF:+.4f})")
    print(f"   COMBINED 10-bag median RAE = {rae_median:.4f}  "
          f"(d_vs_nb1242 {rae_median - NB1242_REF:+.4f}  "
          f"d_vs_nb1290 {rae_median - NB1290_REF:+.4f})")

    # ---- Verdict ----
    best_rae = min(rae_mean, rae_median)
    best_agg = "mean" if rae_mean <= rae_median else "median"
    beats_nb1242 = best_rae < NB1242_REF - DECISION_MARGIN
    beats_nb1290 = best_rae < NB1290_REF - DECISION_MARGIN

    if beats_nb1290:
        verdict = ("COMBINED_BAG_BEATS_NB1290_NEW_PRIMARY_CANDIDATE "
                   f"({best_agg} @ {best_rae:.4f})")
    elif beats_nb1242:
        verdict = (f"COMBINED_BAG_BEATS_NB1242_ONLY ({best_agg} @ "
                   f"{best_rae:.4f}; flat vs nb1290)")
    elif abs(best_rae - NB1242_REF) < DECISION_MARGIN:
        verdict = (f"COMBINED_BAG_TIES_NB1242_FLAT ({best_agg} @ "
                   f"{best_rae:.4f})")
    elif abs(best_rae - NB1290_REF) < DECISION_MARGIN:
        verdict = (f"COMBINED_BAG_TIES_NB1290_FLAT ({best_agg} @ "
                   f"{best_rae:.4f})")
    else:
        verdict = (f"COMBINED_BAG_NO_HELP ({best_agg} @ {best_rae:.4f}; "
                   "worse than nb1242 and nb1290)")
    print(f"\n   verdict = {verdict}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_combined_oof.npy",
            combined_mean_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_combined_median_oof.npy",
            combined_median_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_combined_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_combined_median_oof.npy'}")

    summary = {
        "tag": TAG,
        "status": "ok",
        "n_unb": n_unb,
        "components": ["nb1341_catboost_5seed", "nb1242_lgbm_5seed"],
        "seeds": [0, 1, 7, 42, 137],
        "n_models": 10,
        "per_model_rae_catboost": per_model_rae_cb,
        "per_model_rae_lgbm": per_model_rae_lg,
        "rae_catboost_5bag_mean": rae_cb_5bag,
        "rae_lgbm_5bag_mean": rae_lg_5bag,
        "pearson_cb_lg_5bag": pearson_cb_lg_5bag,
        "rae_combined_mean": rae_mean,
        "rae_combined_median": rae_median,
        "delta_combined_mean_vs_nb1242": rae_mean - NB1242_REF,
        "delta_combined_mean_vs_nb1290": rae_mean - NB1290_REF,
        "delta_combined_median_vs_nb1242": rae_median - NB1242_REF,
        "delta_combined_median_vs_nb1290": rae_median - NB1290_REF,
        "best_aggregation": best_agg,
        "best_rae": best_rae,
        "beats_nb1242": bool(beats_nb1242),
        "beats_nb1290": bool(beats_nb1290),
        "verdict": verdict,
        "nb1242_ref": NB1242_REF,
        "nb1290_ref": NB1290_REF,
        "nb1341_ref": NB1341_REF,
        "decision_margin": DECISION_MARGIN,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.2f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "status", "n_models",
        "rae_catboost_5bag_mean", "rae_lgbm_5bag_mean",
        "pearson_cb_lg_5bag",
        "rae_combined_mean", "rae_combined_median",
        "delta_combined_mean_vs_nb1242",
        "delta_combined_mean_vs_nb1290",
        "best_aggregation", "best_rae",
        "beats_nb1242", "beats_nb1290",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
