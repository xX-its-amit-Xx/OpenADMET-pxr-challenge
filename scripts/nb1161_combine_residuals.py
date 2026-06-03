"""nb1161 -- Combine nb1130 (Morgan+RDKit residual) + nb1153 (Mordred residual).

Hypothesis:
    nb1130 mean-bag (RAE 0.5673) and nb1153 mean-bag (RAE 0.5640) are both
    shallow residual-LGBM bags over the SAME anchor (nb1070, RAE ~0.5790)
    but use DISJOINT feature spaces:
        nb1130 = Morgan(2048) + RDKit-desc(217) = 2265 dims
        nb1153 = Mordred(1533) only
    Mordred encodes topology / autocorrelation / Burden / information-content
    axes that are absent from RDKit-desc.  If the two residual signals capture
    different chunks of the orthogonal-to-nb1070 variance, a positive-weight
    blend pushes pooled RAE below either standalone.

Protocol:
  1. Load nb1130_mean_bag_oof.npy  (253,) -- Morgan+RDKit residual on nb1070
  2. Load nb1153_mean_bag_oof.npy  (253,) -- Mordred residual on nb1070
  3. 5-fold cross-fit SLSQP blend on 253 unblind:
        for each fold:
            fit w = argmin ||y_tr - w0*p1_tr - w1*p2_tr||^2,
                s.t. w >= 0, sum(w) == 1
            predict on val with frozen w
        emit blended OOF; pooled RAE.
  4. Naive mean of the two predictors; pooled RAE.
  5. Naive median (== mean for K=2 but reported for completeness).
  6. Compare to nb1130 and nb1153 standalone pooled RAEs.

NO deploy (513) refit -- 253-only honest cross-fit diagnostic.

Outputs:
  data/processed/nb1161_slsqp_oof.npy     (253,) float32  -- cross-fit SLSQP blend
  data/processed/nb1161_mean_oof.npy      (253,) float32  -- naive mean
  data/processed/nb1161_median_oof.npy    (253,) float32  -- naive median
  data/processed/nb1161_summary.json
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

TAG = "nb1161"
SLSQP_FOLDS = 5
SLSQP_SEED = 42

# Reference numbers (pooled RAE on 253 unblind).
NB1130_MEAN_BAG_REF = 0.5673
NB1153_MEAN_BAG_REF = 0.5640
NB1070_ANCHOR_REF = 0.5790


def _slsqp_blend_weights(P_tr: np.ndarray, y_tr: np.ndarray) -> np.ndarray:
    """Argmin MSE over the K-simplex (w_i >= 0, sum w_i = 1).

    Closed-form alternatives exist for K=2 (a 1-D grid), but SLSQP is the
    same primitive used elsewhere in the ladder so we keep it consistent.
    """
    K = P_tr.shape[1]
    w0 = np.full(K, 1.0 / K)

    def _loss(w: np.ndarray) -> float:
        pred = P_tr @ w
        diff = y_tr - pred
        return float(np.mean(diff * diff))

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
                     n_splits: int, seed: int) -> tuple[np.ndarray, list[dict]]:
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
    print(f"{TAG} -- combine nb1130 (Morgan+RDKit residual) + nb1153 "
          f"(Mordred residual) over nb1070 anchor")
    print(f"          5-fold SLSQP cross-fit blend + mean + median")
    print(f"          shared anchor = nb1070 (pooled RAE ~{NB1070_ANCHOR_REF:.4f})")
    print("=" * 78)

    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)

    p1_path = DATA_PROCESSED / "nb1130_mean_bag_oof.npy"
    p2_path = DATA_PROCESSED / "nb1153_mean_bag_oof.npy"
    if not p1_path.exists():
        raise FileNotFoundError(f"{p1_path} not found (run nb1130 first).")
    if not p2_path.exists():
        raise FileNotFoundError(f"{p2_path} not found (run nb1153 first).")

    p1 = np.load(p1_path).astype(np.float64)  # nb1130 Morgan+RDKit residual bag
    p2 = np.load(p2_path).astype(np.float64)  # nb1153 Mordred residual bag

    if p1.shape[0] != n_unb or p2.shape[0] != n_unb:
        raise ValueError(
            f"shape mismatch: p1={p1.shape}, p2={p2.shape}, n_unb={n_unb}"
        )

    rae_p1 = float(rae(y_unb, p1))
    rae_p2 = float(rae(y_unb, p2))
    print(f"[load] nb1130 mean_bag_oof: shape={p1.shape}  pooled RAE = "
          f"{rae_p1:.4f}  (ref {NB1130_MEAN_BAG_REF:.4f})")
    print(f"[load] nb1153 mean_bag_oof: shape={p2.shape}  pooled RAE = "
          f"{rae_p2:.4f}  (ref {NB1153_MEAN_BAG_REF:.4f})")

    corr_12 = float(np.corrcoef(p1, p2)[0, 1])
    resid_corr = float(np.corrcoef(p1 - y_unb, p2 - y_unb)[0, 1])
    print(f"[diag] pred corr(p1, p2)         = {corr_12:.4f}")
    print(f"[diag] residual corr(e1, e2)     = {resid_corr:.4f}")

    P = np.column_stack([p1, p2])  # (253, 2)

    print("\n" + "-" * 78)
    print("(A) 5-fold SLSQP cross-fit blend  (positive weights, sum == 1)")
    print("-" * 78)
    slsqp_oof, fold_records = _slsqp_cross_fit(
        P, y_unb, n_splits=SLSQP_FOLDS, seed=SLSQP_SEED
    )
    rae_slsqp = float(rae(y_unb, slsqp_oof))
    print(f"   per-fold weights:")
    for rec in fold_records:
        w = rec["weights"]
        print(f"     fold {rec['fold']}: w_nb1130 = {w[0]:.4f}  "
              f"w_nb1153 = {w[1]:.4f}  (n_tr={rec['n_tr']}, n_va={rec['n_va']})")
    fold_weights = np.array([r["weights"] for r in fold_records])
    mean_weights = fold_weights.mean(axis=0)
    print(f"   mean weights across folds:  "
          f"w_nb1130 = {mean_weights[0]:.4f}  "
          f"w_nb1153 = {mean_weights[1]:.4f}")
    print(f"   pooled RAE(SLSQP cross-fit) = {rae_slsqp:.4f}  "
          f"(d_vs_nb1130 = {rae_slsqp - rae_p1:+.4f}, "
          f"d_vs_nb1153 = {rae_slsqp - rae_p2:+.4f})")

    # Whole-dataset SLSQP fit (in-sample, diagnostic only).
    w_full = _slsqp_blend_weights(P, y_unb)
    p_full = P @ w_full
    rae_full = float(rae(y_unb, p_full))
    print(f"   in-sample SLSQP (diagnostic): "
          f"w_nb1130 = {w_full[0]:.4f}  w_nb1153 = {w_full[1]:.4f}  "
          f"RAE = {rae_full:.4f}")

    print("\n" + "-" * 78)
    print("(B) Naive mean of the two predictors  (w = [0.5, 0.5])")
    print("-" * 78)
    mean_oof = 0.5 * p1 + 0.5 * p2
    rae_mean = float(rae(y_unb, mean_oof))
    print(f"   pooled RAE(mean) = {rae_mean:.4f}  "
          f"(d_vs_nb1130 = {rae_mean - rae_p1:+.4f}, "
          f"d_vs_nb1153 = {rae_mean - rae_p2:+.4f})")

    print("\n" + "-" * 78)
    print("(C) Naive median of the two predictors")
    print("-" * 78)
    # For K=2 median == mean, but compute explicitly for record.
    median_oof = np.median(P, axis=1)
    rae_median = float(rae(y_unb, median_oof))
    print(f"   pooled RAE(median) = {rae_median:.4f}  "
          f"(d_vs_nb1130 = {rae_median - rae_p1:+.4f}, "
          f"d_vs_nb1153 = {rae_median - rae_p2:+.4f})")

    # Verdict on cross-fit SLSQP vs best standalone.
    best_standalone = min(rae_p1, rae_p2)
    best_standalone_tag = "nb1153" if rae_p2 <= rae_p1 else "nb1130"
    beats_nb1153 = rae_slsqp < rae_p2 - 0.003
    beats_nb1130 = rae_slsqp < rae_p1 - 0.003

    if beats_nb1153 and beats_nb1130:
        verdict = "SLSQP_BLEND_BEATS_BOTH_RESIDUAL_BAGS"
    elif beats_nb1153:
        verdict = "SLSQP_BLEND_BEATS_NB1153_ONLY"
    elif beats_nb1130:
        verdict = "SLSQP_BLEND_BEATS_NB1130_ONLY"
    elif abs(rae_slsqp - best_standalone) < 0.003:
        verdict = "SLSQP_BLEND_FLAT_VS_BEST_STANDALONE"
    else:
        verdict = "SLSQP_BLEND_HURTS_VS_BEST_STANDALONE"

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"   nb1130 (Morgan+RDKit) standalone : {rae_p1:.4f}")
    print(f"   nb1153 (Mordred)      standalone : {rae_p2:.4f}")
    print(f"   best standalone                  : {best_standalone:.4f}  "
          f"({best_standalone_tag})")
    print(f"   SLSQP cross-fit blend            : {rae_slsqp:.4f}")
    print(f"   naive mean                       : {rae_mean:.4f}")
    print(f"   naive median                     : {rae_median:.4f}")
    print(f"   verdict                          : {verdict}")

    np.save(DATA_PROCESSED / f"{TAG}_slsqp_oof.npy", slsqp_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_mean_oof.npy", mean_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_median_oof.npy", median_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_slsqp_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_mean_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_median_oof.npy'}")

    summary = {
        "tag": TAG,
        "n_unb": n_unb,
        "slsqp_folds": SLSQP_FOLDS,
        "slsqp_seed": SLSQP_SEED,
        "components": ["nb1130_mean_bag_oof", "nb1153_mean_bag_oof"],
        "rae_nb1130_standalone": rae_p1,
        "rae_nb1153_standalone": rae_p2,
        "pred_corr_p1_p2": corr_12,
        "residual_corr_e1_e2": resid_corr,
        "rae_slsqp_cross_fit": rae_slsqp,
        "rae_naive_mean": rae_mean,
        "rae_naive_median": rae_median,
        "in_sample_slsqp_weights": [float(x) for x in w_full],
        "in_sample_slsqp_rae": rae_full,
        "fold_records": fold_records,
        "mean_fold_weights": [float(x) for x in mean_weights],
        "delta_slsqp_vs_nb1130": rae_slsqp - rae_p1,
        "delta_slsqp_vs_nb1153": rae_slsqp - rae_p2,
        "delta_mean_vs_nb1130": rae_mean - rae_p1,
        "delta_mean_vs_nb1153": rae_mean - rae_p2,
        "beats_nb1130": bool(beats_nb1130),
        "beats_nb1153": bool(beats_nb1153),
        "verdict": verdict,
        "nb1130_mean_bag_ref": NB1130_MEAN_BAG_REF,
        "nb1153_mean_bag_ref": NB1153_MEAN_BAG_REF,
        "nb1070_anchor_ref": NB1070_ANCHOR_REF,
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
              "pred_corr_p1_p2", "residual_corr_e1_e2",
              "rae_slsqp_cross_fit", "rae_naive_mean", "rae_naive_median",
              "in_sample_slsqp_weights", "mean_fold_weights",
              "delta_slsqp_vs_nb1153", "delta_slsqp_vs_nb1130",
              "beats_nb1130", "beats_nb1153", "verdict"):
        print(f"  {k}: {res.get(k)}")
