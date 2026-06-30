"""nb1514 -- Final consolidation: rank top-5 PRE-unblind candidates and compute
robust ensemble (naive mean / median / 5-fold cross-fit SLSQP simplex).

CANDIDATES (PRE-unblind, 253 unblind OOF arrays):
    nb1500_bob_median_oof.npy  --  nb1484 BoB MEDIAN bag           (~0.5234)
    nb1501_mean_bag_oof.npy    --  nb1501 CatBoost(MAE) 4-way bag  (~0.5223)
    nb1484_best_oof.npy        --  nb1484 4-way LGBM-Huber best    (~0.5231)
    nb1472_mean_oof.npy        --  nb1472 3-way LGBM-Huber mean    (~0.5330)
    nb1482_bob_median_oof.npy  --  nb1472 BoB MEDIAN bag           (~0.5310)

PROTOCOL:
    1.  Load all 5 OOFs and y_unb.
    2.  Compute pairwise Pearson.
    3.  Naive mean over 5 candidates -> RAE.
    4.  Naive median over 5 candidates -> RAE.
    5.  5-fold cross-fit SLSQP on the simplex (w >= 0, sum w = 1).
        Held-out predictions blended for honest RAE.
    6.  Verdict at 0.003 margin vs nb1501 (0.5223 ref).

Outputs:
    scripts/nb1514_robust_ensemble.py
    data/processed/nb1514_summary.json
    data/processed/nb1514_best_oof.npy        (253,) float32
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
from sklearn.model_selection import KFold
from scipy.optimize import minimize

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED


TAG = "nb1514"

CANDIDATES = [
    ("nb1500_bob_median",  "nb1500_bob_median_oof.npy",  0.5234, "nb1484 BoB MEDIAN"),
    ("nb1501_mean_bag",    "nb1501_mean_bag_oof.npy",    0.5223, "nb1501 CatBoost 4-way"),
    ("nb1484_best",        "nb1484_best_oof.npy",        0.5231, "nb1484 4-way LGBM-Huber"),
    ("nb1472_mean",        "nb1472_mean_oof.npy",        0.5330, "nb1472 3-way LGBM-Huber"),
    ("nb1482_bob_median",  "nb1482_bob_median_oof.npy",  0.5310, "nb1472 BoB MEDIAN"),
]

REF_NAME = "nb1501_mean_bag"
REF_RAE = 0.5223
DECISION_MARGIN = 0.003

CV_FOLDS = 5
CV_SEED = 0


def _slsqp_simplex_weights(P_tr: np.ndarray, y_tr: np.ndarray) -> np.ndarray:
    """Fit SLSQP weights on simplex by minimizing mean-absolute error
    (RAE numerator is mean abs error; denominator is constant w.r.t. weights).
    """
    K = P_tr.shape[1]
    w0 = np.full(K, 1.0 / K, dtype=np.float64)

    def loss(w):
        pred = P_tr @ w
        return float(np.mean(np.abs(y_tr - pred)))

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bnds = [(0.0, 1.0)] * K
    res = minimize(
        loss, w0,
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
        options={"maxiter": 500, "ftol": 1e-9, "disp": False},
    )
    w = np.asarray(res.x, dtype=np.float64)
    w = np.clip(w, 0.0, None)
    s = w.sum()
    if s <= 0.0:
        w = np.full(K, 1.0 / K, dtype=np.float64)
    else:
        w = w / s
    return w


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- final consolidation of top-5 PRE-unblind candidates")
    print(f"          ref: {REF_NAME} ({REF_RAE:.4f})  margin = {DECISION_MARGIN}")
    print("=" * 78)

    # ---- Load truth ----
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_unb = {n_unb}")

    # ---- Load all 5 candidate OOFs ----
    names: list[str] = []
    descs: list[str] = []
    ref_rae_each: list[float] = []
    P_list = []
    indiv_rae: dict[str, float] = {}
    for name, fname, ref_r, desc in CANDIDATES:
        path = DATA_PROCESSED / fname
        if not path.exists():
            raise FileNotFoundError(f"missing OOF file: {path}")
        oof = np.load(path).astype(np.float64)
        if oof.shape[0] != n_unb:
            raise ValueError(f"{fname} shape {oof.shape} != ({n_unb},)")
        r = float(rae(y_unb, oof))
        names.append(name)
        descs.append(desc)
        ref_rae_each.append(ref_r)
        P_list.append(oof)
        indiv_rae[name] = r
        print(f"   {name:24s}  RAE = {r:.4f}  (ref {ref_r:.4f}, "
              f"delta {r - ref_r:+.4f})  -- {desc}")
    P = np.column_stack(P_list)                       # (n_unb, K)
    K = P.shape[1]
    print(f"[stack] P shape = {P.shape}")

    # ---- Rank candidates by realized RAE ----
    rank_order = np.argsort([indiv_rae[n] for n in names])
    print("\n" + "-" * 78)
    print("RANK BY REALIZED RAE")
    print("-" * 78)
    ranking = []
    for rnk, j in enumerate(rank_order, start=1):
        ranking.append({
            "rank": rnk,
            "name": names[j],
            "rae": indiv_rae[names[j]],
            "ref_rae": ref_rae_each[j],
            "desc": descs[j],
        })
        print(f"   {rnk}.  {names[j]:24s}  RAE = {indiv_rae[names[j]]:.4f}")

    # ---- Pairwise Pearson ----
    print("\n" + "-" * 78)
    print("PAIRWISE PEARSON")
    print("-" * 78)
    pearson_mat = np.corrcoef(P, rowvar=False)
    pairwise = []
    for i in range(K):
        for j in range(i + 1, K):
            r_ij = float(pearson_mat[i, j])
            pairwise.append({
                "a": names[i],
                "b": names[j],
                "pearson": r_ij,
            })
            print(f"   {names[i]:24s} <-> {names[j]:24s}  r = {r_ij:.4f}")
    pearson_mean_offdiag = float(
        (pearson_mat.sum() - K) / (K * (K - 1))
    )
    pearson_min_offdiag = float(np.min(
        pearson_mat + np.eye(K) * 2.0
    ))  # eye+2 forces diag out of running for min
    print(f"   mean off-diag      = {pearson_mean_offdiag:.4f}")
    print(f"   min  off-diag      = {pearson_min_offdiag:.4f}")

    # ---- Naive MEAN ----
    naive_mean_oof = P.mean(axis=1)
    rae_naive_mean = float(rae(y_unb, naive_mean_oof))
    delta_mean_vs_ref = rae_naive_mean - REF_RAE
    print("\n" + "-" * 78)
    print("NAIVE MEAN OVER 5")
    print("-" * 78)
    print(f"   RAE(naive mean)    = {rae_naive_mean:.4f}  "
          f"(delta vs {REF_NAME} = {delta_mean_vs_ref:+.4f})")

    # ---- Naive MEDIAN ----
    naive_median_oof = np.median(P, axis=1)
    rae_naive_median = float(rae(y_unb, naive_median_oof))
    delta_median_vs_ref = rae_naive_median - REF_RAE
    print("\n" + "-" * 78)
    print("NAIVE MEDIAN OVER 5")
    print("-" * 78)
    print(f"   RAE(naive median)  = {rae_naive_median:.4f}  "
          f"(delta vs {REF_NAME} = {delta_median_vs_ref:+.4f})")

    # ---- 5-fold cross-fit SLSQP simplex ----
    print("\n" + "-" * 78)
    print(f"5-FOLD CROSS-FIT SLSQP SIMPLEX (K={K})")
    print("-" * 78)
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_SEED)
    slsqp_oof = np.full(n_unb, np.nan, dtype=np.float64)
    fold_weights = np.zeros((CV_FOLDS, K), dtype=np.float64)
    fold_records = []
    for fi, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n_unb))):
        w = _slsqp_simplex_weights(P[tr_loc], y_unb[tr_loc])
        fold_weights[fi] = w
        pred_va = P[va_loc] @ w
        slsqp_oof[va_loc] = pred_va
        rec = {
            "fold": int(fi),
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "weights": [float(x) for x in w],
        }
        fold_records.append(rec)
        wstr = "[" + ", ".join(f"{x:.3f}" for x in w) + "]"
        print(f"   fold {fi}:  w = {wstr}")
    assert np.isfinite(slsqp_oof).all()
    mean_weights = fold_weights.mean(axis=0)
    rae_slsqp = float(rae(y_unb, slsqp_oof))
    delta_slsqp_vs_ref = rae_slsqp - REF_RAE
    wmean_str = "[" + ", ".join(f"{x:.3f}" for x in mean_weights) + "]"
    print(f"   mean-of-fold w    = {wmean_str}")
    print(f"   RAE(cross-fit SLSQP) = {rae_slsqp:.4f}  "
          f"(delta vs {REF_NAME} = {delta_slsqp_vs_ref:+.4f})")

    # ---- Choose best of {naive mean, naive median, SLSQP, individual best} ----
    options = {
        "naive_mean":    rae_naive_mean,
        "naive_median":  rae_naive_median,
        "slsqp_cv":      rae_slsqp,
    }
    # also include the single best candidate for transparency
    best_indiv_name = names[int(rank_order[0])]
    options[f"individual_{best_indiv_name}"] = indiv_rae[best_indiv_name]
    best_key = min(options, key=lambda k: options[k])
    best_rae = options[best_key]

    if best_key == "naive_mean":
        best_oof = naive_mean_oof
    elif best_key == "naive_median":
        best_oof = naive_median_oof
    elif best_key == "slsqp_cv":
        best_oof = slsqp_oof
    else:
        best_oof = P[:, int(rank_order[0])]
    print("\n" + "-" * 78)
    print("BEST AGGREGATION")
    print("-" * 78)
    for k, v in options.items():
        marker = "  <-- BEST" if k == best_key else ""
        print(f"   {k:30s}  RAE = {v:.4f}{marker}")

    # ---- Verdict ----
    beats_ref = best_rae < REF_RAE - DECISION_MARGIN
    flat_vs_ref = abs(best_rae - REF_RAE) < DECISION_MARGIN
    hurts_ref = best_rae > REF_RAE + DECISION_MARGIN

    if beats_ref:
        verdict = (
            f"ROBUST_ENSEMBLE_BEATS_{REF_NAME.upper()}_NEW_PRE_UNBLIND_PRIMARY"
        )
    elif flat_vs_ref:
        verdict = f"ROBUST_ENSEMBLE_FLAT_VS_{REF_NAME.upper()}"
    else:
        verdict = f"ROBUST_ENSEMBLE_HURTS_VS_{REF_NAME.upper()}_KEEP_REF"

    print("\n" + "-" * 78)
    print("VERDICT")
    print("-" * 78)
    print(f"   best aggregation       = {best_key}")
    print(f"   best RAE               = {best_rae:.4f}")
    print(f"   ref {REF_NAME} RAE     = {REF_RAE:.4f}")
    print(f"   delta vs ref           = {best_rae - REF_RAE:+.4f}")
    print(f"   beats_ref (m={DECISION_MARGIN}) = {beats_ref}")
    print(f"   verdict                = {verdict}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_best_oof.npy",
            best_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_best_oof.npy'}")

    summary = {
        "tag": TAG,
        "candidates": [
            {
                "name": name,
                "file": fname,
                "ref_rae": ref_r,
                "desc": desc,
                "realized_rae": indiv_rae[name],
            }
            for (name, fname, ref_r, desc) in CANDIDATES
        ],
        "ranking_by_realized_rae": ranking,
        "n_unb": int(n_unb),
        "K": int(K),
        "pairwise_pearson": pairwise,
        "pearson_matrix": pearson_mat.tolist(),
        "pearson_mean_offdiag": pearson_mean_offdiag,
        "pearson_min_offdiag": pearson_min_offdiag,
        "rae_naive_mean": rae_naive_mean,
        "rae_naive_median": rae_naive_median,
        "cv_folds": CV_FOLDS,
        "cv_seed": CV_SEED,
        "fold_records": fold_records,
        "mean_fold_weights": [float(x) for x in mean_weights],
        "rae_slsqp_crossfit": rae_slsqp,
        "delta_naive_mean_vs_ref": delta_mean_vs_ref,
        "delta_naive_median_vs_ref": delta_median_vs_ref,
        "delta_slsqp_vs_ref": delta_slsqp_vs_ref,
        "options_rae": options,
        "best_aggregation": best_key,
        "best_rae": best_rae,
        "ref_name": REF_NAME,
        "ref_rae": REF_RAE,
        "decision_margin": DECISION_MARGIN,
        "beats_nb1501": bool(beats_ref),
        "flat_vs_nb1501": bool(flat_vs_ref),
        "hurts_vs_nb1501": bool(hurts_ref),
        "verdict": verdict,
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
    for k in (
        "n_unb", "K",
        "ranking_by_realized_rae",
        "pearson_mean_offdiag", "pearson_min_offdiag",
        "rae_naive_mean", "rae_naive_median", "rae_slsqp_crossfit",
        "mean_fold_weights",
        "options_rae", "best_aggregation", "best_rae",
        "delta_slsqp_vs_ref",
        "beats_nb1501", "flat_vs_nb1501",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
