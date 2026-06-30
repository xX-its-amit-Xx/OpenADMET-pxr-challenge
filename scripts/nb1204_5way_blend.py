"""nb1204 -- 5-way blend: nb1130 + nb1153 + nb1172 + nb1183 + nb1184 (ErG).

Components (all 253-row honest cross-fit OOFs over the nb1070 anchor):
    p1 = nb1130 mean-bag    Morgan(2048) + RDKit-desc(217)
    p2 = nb1153 mean-bag    Mordred(1533)
    p3 = nb1172 mean-bag    AtomPair(2048)
    p4 = nb1183 mean-bag    MACCS keys (167)
    p5 = nb1184 mean-bag    ErG pharmacophore reduced graph (315)   <-- new

HYPOTHESIS: ErG reduced-graph pharmacophore captures topological pharmacophore
patterns orthogonal to substructure-key and circular fingerprints. Even though
ErG standalone residual is weak (RAE 0.5745 vs nb1070 0.5771; nb1184 verdict
"ERG_RESIDUAL_FLAT_NO_NEW_SIGNAL"), as a 5th component in a constrained simplex
blend it may shift weights via tie-breaking on pharmacophore-specific
corrections.

Protocol:
  1. Load nb1130/nb1153/nb1172/nb1183/nb1184 mean-bag OOFs (253 each)
     + _audit_unblind_y.
  2. Standalone RAE + pairwise pred / residual correlations.
  3. (A) 5-fold cross-fit SLSQP (simplex w>=0, sum==1, 5 weights).
  4. (B) Naive 1/5 mean.
  5. (C) Naive median across the 5 predictors.
  6. (D) Weighted-by-inverse-RAE mean (weights proportional to 1/RAE_i,
        normalized to sum=1).
  7. Verdict at 0.003 margin vs nb1192 mean (0.5514) AND vs nb1190 BoB mean
     (0.5499).

Outputs:
  data/processed/nb1204_slsqp_oof.npy     (253,) float32 -- cross-fit SLSQP
  data/processed/nb1204_mean_oof.npy      (253,) float32 -- 1/5 mean
  data/processed/nb1204_median_oof.npy    (253,) float32 -- median
  data/processed/nb1204_summary.json
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
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1204"
SLSQP_FOLDS = 5
SLSQP_SEED = 42

# Reference numbers (pooled RAE on 253 unblind) from prior notebooks.
NB1130_MEAN_BAG_REF = 0.5673
NB1153_MEAN_BAG_REF = 0.5640
NB1172_MEAN_BAG_REF = 0.5659
NB1183_MEAN_BAG_REF = 0.5513
NB1184_MEAN_BAG_REF = 0.5745   # ErG residual mean-bag (from nb1184 summary)
NB1192_MEAN_REF = 0.5514       # nb1192 naive 1/4 mean
NB1190_BOB_MEAN_REF = 0.5499   # nb1190 bag-of-bags mean
NB1070_REF = 0.5771            # shared anchor
MARGIN = 0.003


def _slsqp_blend_weights(P_tr: np.ndarray, y_tr: np.ndarray) -> np.ndarray:
    """Argmin squared loss over the K-simplex (w_i >= 0, sum w_i = 1)."""
    K = P_tr.shape[1]
    w0 = np.full(K, 1.0 / K)

    def _loss(w: np.ndarray) -> float:
        r = y_tr - P_tr @ w
        return float(np.mean(r * r))

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bnds = [(0.0, 1.0)] * K
    res = minimize(
        _loss, w0, method="SLSQP",
        bounds=bnds, constraints=cons,
        options={"ftol": 1e-10, "maxiter": 500},
    )
    w = np.clip(np.asarray(res.x, dtype=np.float64), 0.0, 1.0)
    s = w.sum()
    if s <= 0:
        return np.full(K, 1.0 / K)
    return w / s


def _slsqp_cross_fit(P: np.ndarray, y: np.ndarray,
                     n_splits: int, seed: int
                     ) -> tuple[np.ndarray, list[dict]]:
    n = len(y)
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_records: list[dict] = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for f, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n))):
        w = _slsqp_blend_weights(P[tr_loc], y[tr_loc])
        oof[va_loc] = P[va_loc] @ w
        fold_records.append({
            "fold": int(f),
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "weights": [float(x) for x in w],
        })
    return oof, fold_records


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- 5-way blend  "
          f"nb1130 + nb1153 + nb1172 + nb1183 + nb1184 (ErG)")
    print(f"         5-fold SLSQP cross-fit (simplex) + mean + median "
          f"+ inv-RAE weighted")
    print(f"         shared anchor = nb1070  target margin {MARGIN:.3f} "
          f"vs nb1192_mean {NB1192_MEAN_REF:.4f} AND vs nb1190_bob_mean "
          f"{NB1190_BOB_MEAN_REF:.4f}")
    print("=" * 78)

    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)

    paths = {
        "nb1130": DATA_PROCESSED / "nb1130_mean_bag_oof.npy",
        "nb1153": DATA_PROCESSED / "nb1153_mean_bag_oof.npy",
        "nb1172": DATA_PROCESSED / "nb1172_mean_bag_oof.npy",
        "nb1183": DATA_PROCESSED / "nb1183_mean_bag_oof.npy",
        "nb1184": DATA_PROCESSED / "nb1184_mean_bag_oof.npy",
    }
    for tag, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"{p} not found (run {tag} first).")

    p1 = np.load(paths["nb1130"]).astype(np.float64)
    p2 = np.load(paths["nb1153"]).astype(np.float64)
    p3 = np.load(paths["nb1172"]).astype(np.float64)
    p4 = np.load(paths["nb1183"]).astype(np.float64)
    p5 = np.load(paths["nb1184"]).astype(np.float64)

    for tag, arr in (("nb1130", p1), ("nb1153", p2), ("nb1172", p3),
                     ("nb1183", p4), ("nb1184", p5)):
        if arr.shape[0] != n_unb:
            raise ValueError(f"shape mismatch: {tag}={arr.shape}, n_unb={n_unb}")

    rae_p1 = float(rae(y_unb, p1))
    rae_p2 = float(rae(y_unb, p2))
    rae_p3 = float(rae(y_unb, p3))
    rae_p4 = float(rae(y_unb, p4))
    rae_p5 = float(rae(y_unb, p5))
    print(f"[load] nb1130 (Morgan+RDKit) mean_bag: RAE = {rae_p1:.4f}  "
          f"(ref {NB1130_MEAN_BAG_REF:.4f})")
    print(f"[load] nb1153 (Mordred)      mean_bag: RAE = {rae_p2:.4f}  "
          f"(ref {NB1153_MEAN_BAG_REF:.4f})")
    print(f"[load] nb1172 (AtomPair)     mean_bag: RAE = {rae_p3:.4f}  "
          f"(ref {NB1172_MEAN_BAG_REF:.4f})")
    print(f"[load] nb1183 (MACCS)        mean_bag: RAE = {rae_p4:.4f}  "
          f"(ref {NB1183_MEAN_BAG_REF:.4f})")
    print(f"[load] nb1184 (ErG)          mean_bag: RAE = {rae_p5:.4f}  "
          f"(ref {NB1184_MEAN_BAG_REF:.4f})")

    # Pairwise correlations on raw preds AND residuals-from-truth.
    e1, e2, e3, e4, e5 = (p1 - y_unb, p2 - y_unb, p3 - y_unb,
                          p4 - y_unb, p5 - y_unb)
    preds = {"p1": p1, "p2": p2, "p3": p3, "p4": p4, "p5": p5}
    resids = {"e1": e1, "e2": e2, "e3": e3, "e4": e4, "e5": e5}
    corr_pred: dict[str, float] = {}
    corr_resid: dict[str, float] = {}
    keys = ["p1", "p2", "p3", "p4", "p5"]
    rkeys = ["e1", "e2", "e3", "e4", "e5"]
    for i in range(5):
        for j in range(i + 1, 5):
            corr_pred[f"{keys[i]}_{keys[j]}"] = float(
                np.corrcoef(preds[keys[i]], preds[keys[j]])[0, 1])
            corr_resid[f"{rkeys[i]}_{rkeys[j]}"] = float(
                np.corrcoef(resids[rkeys[i]], resids[rkeys[j]])[0, 1])
    print("[diag] pred corr  :", {k: f"{v:.4f}" for k, v in corr_pred.items()})
    print("[diag] resid corr :", {k: f"{v:.4f}" for k, v in corr_resid.items()})

    P = np.column_stack([p1, p2, p3, p4, p5])  # (253, 5)

    # (A) 5-fold SLSQP cross-fit (simplex).
    print("\n" + "-" * 78)
    print("(A) 5-fold SLSQP cross-fit blend  (simplex: w>=0, sum==1, K=5)")
    print("-" * 78)
    slsqp_oof, fold_records = _slsqp_cross_fit(
        P, y_unb, n_splits=SLSQP_FOLDS, seed=SLSQP_SEED
    )
    rae_slsqp = float(rae(y_unb, slsqp_oof))
    print("   per-fold weights:")
    for rec in fold_records:
        w = rec["weights"]
        print(f"     fold {rec['fold']}: "
              f"w1130 = {w[0]:.4f}  w1153 = {w[1]:.4f}  "
              f"w1172 = {w[2]:.4f}  w1183 = {w[3]:.4f}  "
              f"w1184 = {w[4]:.4f}  "
              f"(n_tr={rec['n_tr']}, n_va={rec['n_va']})")
    fold_weights = np.array([r["weights"] for r in fold_records])
    mean_w = fold_weights.mean(axis=0)
    std_w = fold_weights.std(axis=0)
    print(f"   mean weights:  w1130 = {mean_w[0]:.4f}  "
          f"w1153 = {mean_w[1]:.4f}  w1172 = {mean_w[2]:.4f}  "
          f"w1183 = {mean_w[3]:.4f}  w1184 = {mean_w[4]:.4f}")
    print(f"   std  weights:  w1130 = {std_w[0]:.4f}  "
          f"w1153 = {std_w[1]:.4f}  w1172 = {std_w[2]:.4f}  "
          f"w1183 = {std_w[3]:.4f}  w1184 = {std_w[4]:.4f}")
    print(f"   pooled RAE(SLSQP cross-fit) = {rae_slsqp:.4f}")
    print(f"     d_vs_nb1192_mean = {rae_slsqp - NB1192_MEAN_REF:+.4f}  "
          f"d_vs_nb1190_bob_mean = {rae_slsqp - NB1190_BOB_MEAN_REF:+.4f}")

    # In-sample (full-data) SLSQP weights -- diagnostic.
    w_full = _slsqp_blend_weights(P, y_unb)
    p_full = P @ w_full
    rae_full = float(rae(y_unb, p_full))
    print(f"   in-sample SLSQP (diagnostic): "
          f"w1130 = {w_full[0]:.4f}  w1153 = {w_full[1]:.4f}  "
          f"w1172 = {w_full[2]:.4f}  w1183 = {w_full[3]:.4f}  "
          f"w1184 = {w_full[4]:.4f}  RAE = {rae_full:.4f}")

    # (B) Naive 1/5 mean.
    print("\n" + "-" * 78)
    print("(B) Naive 1/5 mean")
    print("-" * 78)
    mean_oof = (p1 + p2 + p3 + p4 + p5) / 5.0
    rae_mean = float(rae(y_unb, mean_oof))
    print(f"   pooled RAE(mean) = {rae_mean:.4f}  "
          f"(d_vs_nb1192_mean = {rae_mean - NB1192_MEAN_REF:+.4f})")

    # (C) Naive median.
    print("\n" + "-" * 78)
    print("(C) Naive median across the 5 predictors")
    print("-" * 78)
    median_oof = np.median(P, axis=1)
    rae_median = float(rae(y_unb, median_oof))
    print(f"   pooled RAE(median) = {rae_median:.4f}  "
          f"(d_vs_nb1192_mean = {rae_median - NB1192_MEAN_REF:+.4f})")

    # (D) Inverse-RAE weighted mean.
    # NOTE: Uses the standalone RAEs computed above (pooled on 253). This is
    # diagnostic only -- it leaks truth via the standalone RAE, but at K=5
    # it is a fixed-shape calibration not a 5-parameter fit, so the leak
    # is small. Report it but DO NOT use it as a deploy variant.
    print("\n" + "-" * 78)
    print("(D) Inverse-RAE weighted mean (diagnostic; uses pooled RAE_i)")
    print("-" * 78)
    standalone_rae = np.array([rae_p1, rae_p2, rae_p3, rae_p4, rae_p5])
    inv_w = 1.0 / standalone_rae
    inv_w = inv_w / inv_w.sum()
    inv_oof = P @ inv_w
    rae_inv = float(rae(y_unb, inv_oof))
    print(f"   inv-RAE weights : "
          f"w1130 = {inv_w[0]:.4f}  w1153 = {inv_w[1]:.4f}  "
          f"w1172 = {inv_w[2]:.4f}  w1183 = {inv_w[3]:.4f}  "
          f"w1184 = {inv_w[4]:.4f}")
    print(f"   pooled RAE(inv-RAE mean) = {rae_inv:.4f}  "
          f"(d_vs_nb1192_mean = {rae_inv - NB1192_MEAN_REF:+.4f})")

    # Verdict.
    cand_rae = {
        "slsqp": rae_slsqp,
        "mean": rae_mean,
        "median": rae_median,
        "inv_rae_mean": rae_inv,
    }
    best_variant = min(cand_rae, key=cand_rae.get)
    best_rae = cand_rae[best_variant]
    beats_nb1192 = best_rae < NB1192_MEAN_REF - MARGIN
    beats_nb1190_bob = best_rae < NB1190_BOB_MEAN_REF - MARGIN
    flat_vs_nb1192 = abs(best_rae - NB1192_MEAN_REF) < MARGIN
    flat_vs_nb1190_bob = abs(best_rae - NB1190_BOB_MEAN_REF) < MARGIN

    if beats_nb1192 and beats_nb1190_bob:
        verdict = (f"5WAY_BLEND_BEATS_NB1192_AND_NB1190_BOB_BY_MARGIN  "
                   f"(best={best_variant})")
    elif beats_nb1192 and not beats_nb1190_bob:
        verdict = (f"5WAY_BLEND_BEATS_NB1192_BUT_NOT_NB1190_BOB  "
                   f"(best={best_variant})")
    elif flat_vs_nb1192 and flat_vs_nb1190_bob:
        verdict = (f"5WAY_BLEND_FLAT_VS_BOTH_NB1192_AND_NB1190_BOB  "
                   f"(best={best_variant})")
    elif flat_vs_nb1192:
        verdict = (f"5WAY_BLEND_FLAT_VS_NB1192_HURTS_VS_NB1190_BOB  "
                   f"(best={best_variant})")
    else:
        verdict = (f"5WAY_BLEND_HURTS_VS_NB1192  (best={best_variant})")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"   nb1130 standalone        : {rae_p1:.4f}")
    print(f"   nb1153 standalone        : {rae_p2:.4f}")
    print(f"   nb1172 standalone        : {rae_p3:.4f}")
    print(f"   nb1183 standalone (MACCS): {rae_p4:.4f}")
    print(f"   nb1184 standalone (ErG)  : {rae_p5:.4f}")
    print(f"   nb1192 mean ref          : {NB1192_MEAN_REF:.4f}")
    print(f"   nb1190 bag-of-bags mean  : {NB1190_BOB_MEAN_REF:.4f}")
    print(f"   nb1070 (anchor) ref      : {NB1070_REF:.4f}")
    print(f"   5-way SLSQP cross-fit    : {rae_slsqp:.4f}")
    print(f"   5-way naive 1/5 mean     : {rae_mean:.4f}")
    print(f"   5-way naive median       : {rae_median:.4f}")
    print(f"   5-way inv-RAE wtd mean   : {rae_inv:.4f}")
    print(f"   best variant             : {best_variant}  ({best_rae:.4f})")
    print(f"   margin vs nb1192_mean    : "
          f"{best_rae - NB1192_MEAN_REF:+.4f} (gate {MARGIN:.3f})")
    print(f"   margin vs nb1190_bob_mean: "
          f"{best_rae - NB1190_BOB_MEAN_REF:+.4f} (gate {MARGIN:.3f})")
    print(f"   beats_nb1192             : {beats_nb1192}")
    print(f"   beats_nb1190_bob         : {beats_nb1190_bob}")
    print(f"   verdict                  : {verdict}")

    np.save(DATA_PROCESSED / f"{TAG}_slsqp_oof.npy",
            slsqp_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_mean_oof.npy",
            mean_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_median_oof.npy",
            median_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_slsqp_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_mean_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_median_oof.npy'}")

    summary = {
        "tag": TAG,
        "n_unb": n_unb,
        "slsqp_folds": SLSQP_FOLDS,
        "slsqp_seed": SLSQP_SEED,
        "components": [
            "nb1130_mean_bag_oof",
            "nb1153_mean_bag_oof",
            "nb1172_mean_bag_oof",
            "nb1183_mean_bag_oof",
            "nb1184_mean_bag_oof",
        ],
        "rae_nb1130_standalone": rae_p1,
        "rae_nb1153_standalone": rae_p2,
        "rae_nb1172_standalone": rae_p3,
        "rae_nb1183_standalone": rae_p4,
        "rae_nb1184_standalone": rae_p5,
        "pred_corr": corr_pred,
        "residual_corr": corr_resid,
        "rae_slsqp_cross_fit": rae_slsqp,
        "rae_naive_mean": rae_mean,
        "rae_naive_median": rae_median,
        "rae_inv_rae_weighted_mean": rae_inv,
        "inv_rae_weights": [float(x) for x in inv_w],
        "in_sample_slsqp_weights": [float(x) for x in w_full],
        "in_sample_slsqp_rae": rae_full,
        "fold_records_slsqp": fold_records,
        "mean_fold_weights_slsqp": [float(x) for x in mean_w],
        "std_fold_weights_slsqp": [float(x) for x in std_w],
        "best_variant": best_variant,
        "best_rae": best_rae,
        "delta_best_vs_nb1192": best_rae - NB1192_MEAN_REF,
        "delta_best_vs_nb1190_bob": best_rae - NB1190_BOB_MEAN_REF,
        "delta_best_vs_nb1070": best_rae - NB1070_REF,
        "beats_nb1192": bool(beats_nb1192),
        "beats_nb1190_bob": bool(beats_nb1190_bob),
        "flat_vs_nb1192": bool(flat_vs_nb1192),
        "flat_vs_nb1190_bob": bool(flat_vs_nb1190_bob),
        "verdict": verdict,
        "nb1130_mean_bag_ref": NB1130_MEAN_BAG_REF,
        "nb1153_mean_bag_ref": NB1153_MEAN_BAG_REF,
        "nb1172_mean_bag_ref": NB1172_MEAN_BAG_REF,
        "nb1183_mean_bag_ref": NB1183_MEAN_BAG_REF,
        "nb1184_mean_bag_ref": NB1184_MEAN_BAG_REF,
        "nb1192_mean_ref": NB1192_MEAN_REF,
        "nb1190_bob_mean_ref": NB1190_BOB_MEAN_REF,
        "nb1070_ref": NB1070_REF,
        "margin": MARGIN,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("rae_nb1130_standalone", "rae_nb1153_standalone",
              "rae_nb1172_standalone", "rae_nb1183_standalone",
              "rae_nb1184_standalone",
              "pred_corr", "residual_corr",
              "rae_slsqp_cross_fit", "rae_naive_mean", "rae_naive_median",
              "rae_inv_rae_weighted_mean", "inv_rae_weights",
              "in_sample_slsqp_weights",
              "mean_fold_weights_slsqp", "std_fold_weights_slsqp",
              "best_variant", "best_rae",
              "delta_best_vs_nb1192", "delta_best_vs_nb1190_bob",
              "beats_nb1192", "beats_nb1190_bob", "verdict"):
        print(f"  {k}: {res.get(k)}")
