"""nb1181 -- Triple SLSQP/mean/median blend of three orthogonal residual bags.

Components (all 253-row honest cross-fit OOFs over the nb1070 anchor):
    p1 = nb1130 mean-bag    Morgan(2048) + RDKit-desc(217)  -- circular + descriptor
    p2 = nb1153 mean-bag    Mordred(1533)                   -- topology / autocorr / Burden
    p3 = nb1172 mean-bag    AtomPair(2048)                  -- pairwise distance fp

HYPOTHESIS: even though pairwise correlations are very high (corr(p1,p2)~0.98 in
nb1161, residual corr ~0.975), the 3-way simplex can break ties that 2-way blends
(nb1161 nb1130+nb1153 == 0.5615 / 0.5600; nb1180 nb1153+nb1172) cannot, by
exploiting the third orthogonal-ish axis (atom-pair distance) at a fraction
weight that mean+median miss.

Protocol:
  1. Load nb1130/nb1153/nb1172 mean-bag OOFs (253 each), load _audit_unblind_y.
  2. Pairwise Pearson on raw predictions AND on residuals-from-truth.
  3. (A) 5-fold cross-fit SLSQP, simplex constraint (w>=0, sum==1), 3 weights.
     Report per-fold weights, mean+std across folds, in-sample (full-data) SLSQP.
  4. (B) Naive 1/3 mean.
  5. (C) Naive median across the three predictors.
  6. (D) Huber-target equal-weight SLSQP variant -- replace squared loss with
     Huber (delta = 1.0 log-unit) for robustness against the rare-scaffold tail.
  7. Verdict at 0.003 margin vs nb1161 mean (0.5600).

NO deploy (513) refit -- 253-only honest cross-fit diagnostic.

Outputs:
  data/processed/nb1181_slsqp_oof.npy     (253,) float32 -- cross-fit SLSQP blend
  data/processed/nb1181_mean_oof.npy      (253,) float32 -- 1/3 mean
  data/processed/nb1181_median_oof.npy    (253,) float32 -- median
  data/processed/nb1181_huber_oof.npy     (253,) float32 -- Huber-target SLSQP CV
  data/processed/nb1181_summary.json
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

TAG = "nb1181"
SLSQP_FOLDS = 5
SLSQP_SEED = 42
HUBER_DELTA = 1.0

# Reference numbers (pooled RAE on 253 unblind) from prior notebooks.
NB1130_MEAN_BAG_REF = 0.5673
NB1153_MEAN_BAG_REF = 0.5640
NB1172_MEAN_BAG_REF = None   # filled in at runtime
NB1161_MEAN_REF = 0.5600     # 2-way naive mean of nb1130+nb1153 (nb1161 summary)
NB1161_SLSQP_REF = 0.5615    # 2-way SLSQP cross-fit (nb1161 summary)
MARGIN = 0.003


def _slsqp_blend_weights(P_tr: np.ndarray, y_tr: np.ndarray,
                         use_huber: bool = False,
                         delta: float = HUBER_DELTA) -> np.ndarray:
    """Argmin loss over the K-simplex (w_i >= 0, sum w_i = 1).

    use_huber=True swaps squared loss for Huber-loss-of-residuals, which
    down-weights the rare-scaffold tail (|err| > delta) while staying
    quadratic in the core.
    """
    K = P_tr.shape[1]
    w0 = np.full(K, 1.0 / K)

    if use_huber:
        def _loss(w: np.ndarray) -> float:
            r = y_tr - P_tr @ w
            a = np.abs(r)
            q = np.minimum(a, delta)
            return float(np.mean(0.5 * q * q + delta * (a - q)))
    else:
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
                     n_splits: int, seed: int,
                     use_huber: bool = False) -> tuple[np.ndarray, list[dict]]:
    n = len(y)
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_records: list[dict] = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for f, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n))):
        w = _slsqp_blend_weights(P[tr_loc], y[tr_loc], use_huber=use_huber)
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
    print(f"{TAG} -- triple blend  "
          f"nb1130 (Morgan+RDKit) + nb1153 (Mordred) + nb1172 (AtomPair)")
    print(f"         5-fold SLSQP cross-fit (simplex) + mean + median + Huber-SLSQP")
    print(f"         shared anchor = nb1070, target margin {MARGIN:.3f} vs "
          f"nb1161 mean = {NB1161_MEAN_REF:.4f}")
    print("=" * 78)

    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)

    paths = {
        "nb1130": DATA_PROCESSED / "nb1130_mean_bag_oof.npy",
        "nb1153": DATA_PROCESSED / "nb1153_mean_bag_oof.npy",
        "nb1172": DATA_PROCESSED / "nb1172_mean_bag_oof.npy",
    }
    for tag, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"{p} not found (run {tag} first).")

    p1 = np.load(paths["nb1130"]).astype(np.float64)
    p2 = np.load(paths["nb1153"]).astype(np.float64)
    p3 = np.load(paths["nb1172"]).astype(np.float64)

    for tag, arr in (("nb1130", p1), ("nb1153", p2), ("nb1172", p3)):
        if arr.shape[0] != n_unb:
            raise ValueError(
                f"shape mismatch: {tag}={arr.shape}, n_unb={n_unb}"
            )

    rae_p1 = float(rae(y_unb, p1))
    rae_p2 = float(rae(y_unb, p2))
    rae_p3 = float(rae(y_unb, p3))
    print(f"[load] nb1130 (Morgan+RDKit) mean_bag: shape={p1.shape}  RAE = "
          f"{rae_p1:.4f}  (ref {NB1130_MEAN_BAG_REF:.4f})")
    print(f"[load] nb1153 (Mordred)      mean_bag: shape={p2.shape}  RAE = "
          f"{rae_p2:.4f}  (ref {NB1153_MEAN_BAG_REF:.4f})")
    print(f"[load] nb1172 (AtomPair)     mean_bag: shape={p3.shape}  RAE = "
          f"{rae_p3:.4f}")

    # Pairwise correlations on raw preds AND residuals-from-truth.
    e1, e2, e3 = p1 - y_unb, p2 - y_unb, p3 - y_unb
    corr_pred = {
        "p1_p2": float(np.corrcoef(p1, p2)[0, 1]),
        "p1_p3": float(np.corrcoef(p1, p3)[0, 1]),
        "p2_p3": float(np.corrcoef(p2, p3)[0, 1]),
    }
    corr_resid = {
        "e1_e2": float(np.corrcoef(e1, e2)[0, 1]),
        "e1_e3": float(np.corrcoef(e1, e3)[0, 1]),
        "e2_e3": float(np.corrcoef(e2, e3)[0, 1]),
    }
    print(f"[diag] pred corr   p1_p2 = {corr_pred['p1_p2']:.4f}  "
          f"p1_p3 = {corr_pred['p1_p3']:.4f}  "
          f"p2_p3 = {corr_pred['p2_p3']:.4f}")
    print(f"[diag] resid corr  e1_e2 = {corr_resid['e1_e2']:.4f}  "
          f"e1_e3 = {corr_resid['e1_e3']:.4f}  "
          f"e2_e3 = {corr_resid['e2_e3']:.4f}")

    P = np.column_stack([p1, p2, p3])  # (253, 3)

    # (A) 5-fold SLSQP cross-fit (simplex).
    print("\n" + "-" * 78)
    print("(A) 5-fold SLSQP cross-fit blend  (simplex: w>=0, sum==1)")
    print("-" * 78)
    slsqp_oof, fold_records = _slsqp_cross_fit(
        P, y_unb, n_splits=SLSQP_FOLDS, seed=SLSQP_SEED
    )
    rae_slsqp = float(rae(y_unb, slsqp_oof))
    print("   per-fold weights:")
    for rec in fold_records:
        w = rec["weights"]
        print(f"     fold {rec['fold']}: w1130 = {w[0]:.4f}  "
              f"w1153 = {w[1]:.4f}  w1172 = {w[2]:.4f}  "
              f"(n_tr={rec['n_tr']}, n_va={rec['n_va']})")
    fold_weights = np.array([r["weights"] for r in fold_records])
    mean_w = fold_weights.mean(axis=0)
    std_w = fold_weights.std(axis=0)
    print(f"   mean weights:  w1130 = {mean_w[0]:.4f}  "
          f"w1153 = {mean_w[1]:.4f}  w1172 = {mean_w[2]:.4f}")
    print(f"   std  weights:  w1130 = {std_w[0]:.4f}  "
          f"w1153 = {std_w[1]:.4f}  w1172 = {std_w[2]:.4f}")
    print(f"   pooled RAE(SLSQP cross-fit) = {rae_slsqp:.4f}  "
          f"(d_vs_nb1161_mean = {rae_slsqp - NB1161_MEAN_REF:+.4f})")

    # In-sample (full-data) SLSQP weights -- diagnostic.
    w_full = _slsqp_blend_weights(P, y_unb)
    p_full = P @ w_full
    rae_full = float(rae(y_unb, p_full))
    print(f"   in-sample SLSQP (diagnostic): "
          f"w1130 = {w_full[0]:.4f}  w1153 = {w_full[1]:.4f}  "
          f"w1172 = {w_full[2]:.4f}  RAE = {rae_full:.4f}")

    # (B) Naive 1/3 mean.
    print("\n" + "-" * 78)
    print("(B) Naive 1/3 mean")
    print("-" * 78)
    mean_oof = (p1 + p2 + p3) / 3.0
    rae_mean = float(rae(y_unb, mean_oof))
    print(f"   pooled RAE(mean) = {rae_mean:.4f}  "
          f"(d_vs_nb1161_mean = {rae_mean - NB1161_MEAN_REF:+.4f})")

    # (C) Naive median.
    print("\n" + "-" * 78)
    print("(C) Naive median across the 3 predictors")
    print("-" * 78)
    median_oof = np.median(P, axis=1)
    rae_median = float(rae(y_unb, median_oof))
    print(f"   pooled RAE(median) = {rae_median:.4f}  "
          f"(d_vs_nb1161_mean = {rae_median - NB1161_MEAN_REF:+.4f})")

    # (D) Huber-target SLSQP cross-fit.
    print("\n" + "-" * 78)
    print(f"(D) Huber-target SLSQP cross-fit  (delta = {HUBER_DELTA:.2f})")
    print("-" * 78)
    huber_oof, huber_records = _slsqp_cross_fit(
        P, y_unb, n_splits=SLSQP_FOLDS, seed=SLSQP_SEED, use_huber=True
    )
    rae_huber = float(rae(y_unb, huber_oof))
    h_fold_weights = np.array([r["weights"] for r in huber_records])
    h_mean_w = h_fold_weights.mean(axis=0)
    h_std_w = h_fold_weights.std(axis=0)
    print("   per-fold weights:")
    for rec in huber_records:
        w = rec["weights"]
        print(f"     fold {rec['fold']}: w1130 = {w[0]:.4f}  "
              f"w1153 = {w[1]:.4f}  w1172 = {w[2]:.4f}")
    print(f"   mean weights:  w1130 = {h_mean_w[0]:.4f}  "
          f"w1153 = {h_mean_w[1]:.4f}  w1172 = {h_mean_w[2]:.4f}")
    print(f"   std  weights:  w1130 = {h_std_w[0]:.4f}  "
          f"w1153 = {h_std_w[1]:.4f}  w1172 = {h_std_w[2]:.4f}")
    print(f"   pooled RAE(Huber-SLSQP cross-fit) = {rae_huber:.4f}  "
          f"(d_vs_nb1161_mean = {rae_huber - NB1161_MEAN_REF:+.4f})")

    # Verdict.
    cand_rae = {
        "slsqp": rae_slsqp,
        "mean": rae_mean,
        "median": rae_median,
        "huber": rae_huber,
    }
    best_variant = min(cand_rae, key=cand_rae.get)
    best_rae = cand_rae[best_variant]
    beats_nb1161 = best_rae < NB1161_MEAN_REF - MARGIN
    flat_vs_nb1161 = abs(best_rae - NB1161_MEAN_REF) < MARGIN

    if beats_nb1161:
        verdict = f"TRIPLE_BLEND_BEATS_NB1161_BY_MARGIN  (best={best_variant})"
    elif flat_vs_nb1161:
        verdict = f"TRIPLE_BLEND_FLAT_VS_NB1161  (best={best_variant})"
    else:
        verdict = f"TRIPLE_BLEND_HURTS_VS_NB1161  (best={best_variant})"

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"   nb1130 standalone        : {rae_p1:.4f}")
    print(f"   nb1153 standalone        : {rae_p2:.4f}")
    print(f"   nb1172 standalone        : {rae_p3:.4f}")
    print(f"   nb1161 (2-way mean) ref  : {NB1161_MEAN_REF:.4f}")
    print(f"   nb1161 (2-way SLSQP) ref : {NB1161_SLSQP_REF:.4f}")
    print(f"   triple SLSQP cross-fit   : {rae_slsqp:.4f}")
    print(f"   triple naive mean        : {rae_mean:.4f}")
    print(f"   triple naive median      : {rae_median:.4f}")
    print(f"   triple Huber SLSQP CV    : {rae_huber:.4f}")
    print(f"   best variant             : {best_variant}  ({best_rae:.4f})")
    print(f"   margin vs nb1161 mean    : "
          f"{best_rae - NB1161_MEAN_REF:+.4f} (margin gate {MARGIN:.3f})")
    print(f"   beats_nb1161             : {beats_nb1161}")
    print(f"   verdict                  : {verdict}")

    np.save(DATA_PROCESSED / f"{TAG}_slsqp_oof.npy", slsqp_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_mean_oof.npy", mean_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_median_oof.npy", median_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_huber_oof.npy", huber_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_slsqp_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_mean_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_median_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_huber_oof.npy'}")

    summary = {
        "tag": TAG,
        "n_unb": n_unb,
        "slsqp_folds": SLSQP_FOLDS,
        "slsqp_seed": SLSQP_SEED,
        "huber_delta": HUBER_DELTA,
        "components": ["nb1130_mean_bag_oof", "nb1153_mean_bag_oof",
                       "nb1172_mean_bag_oof"],
        "rae_nb1130_standalone": rae_p1,
        "rae_nb1153_standalone": rae_p2,
        "rae_nb1172_standalone": rae_p3,
        "pred_corr": corr_pred,
        "residual_corr": corr_resid,
        "rae_slsqp_cross_fit": rae_slsqp,
        "rae_naive_mean": rae_mean,
        "rae_naive_median": rae_median,
        "rae_huber_cross_fit": rae_huber,
        "in_sample_slsqp_weights": [float(x) for x in w_full],
        "in_sample_slsqp_rae": rae_full,
        "fold_records_slsqp": fold_records,
        "mean_fold_weights_slsqp": [float(x) for x in mean_w],
        "std_fold_weights_slsqp": [float(x) for x in std_w],
        "fold_records_huber": huber_records,
        "mean_fold_weights_huber": [float(x) for x in h_mean_w],
        "std_fold_weights_huber": [float(x) for x in h_std_w],
        "best_variant": best_variant,
        "best_rae": best_rae,
        "delta_best_vs_nb1161_mean": best_rae - NB1161_MEAN_REF,
        "beats_nb1161": bool(beats_nb1161),
        "flat_vs_nb1161": bool(flat_vs_nb1161),
        "verdict": verdict,
        "nb1130_mean_bag_ref": NB1130_MEAN_BAG_REF,
        "nb1153_mean_bag_ref": NB1153_MEAN_BAG_REF,
        "nb1161_mean_ref": NB1161_MEAN_REF,
        "nb1161_slsqp_ref": NB1161_SLSQP_REF,
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
              "rae_nb1172_standalone",
              "pred_corr", "residual_corr",
              "rae_slsqp_cross_fit", "rae_naive_mean", "rae_naive_median",
              "rae_huber_cross_fit",
              "in_sample_slsqp_weights",
              "mean_fold_weights_slsqp", "std_fold_weights_slsqp",
              "best_variant", "best_rae", "delta_best_vs_nb1161_mean",
              "beats_nb1161", "verdict"):
        print(f"  {k}: {res.get(k)}")
