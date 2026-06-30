"""nb1251 -- Combine nb1242 (ChEMBL-feat residual) + nb1211 (BoB-of-BoBs blend).

Hypothesis:
    nb1242 (0.5431) and nb1211 (0.5451) draw on different signal sources.
        - nb1242 = ChEMBL-kNN feature-augmented residual (external chemistry info)
        - nb1211 = BoB-of-BoBs blend (internal nb1190+nb1200 variance reduction)
    May be orthogonal at the residual level.

Protocol:
  1. Load nb1242_mean_bag_oof.npy (0.5431) and nb1211_mean_oof.npy (0.5451).
  2. Compute pred-pred Pearson and residual Pearson.
  3. 5-fold cross-fit SLSQP.  Also: naive 0.5/0.5 mean RAE, median RAE.
  4. Try asymmetric weights grid {0.3..0.7 step 0.05} on nb1242, complement on nb1211.
  5. Verdict at 0.003 margin vs nb1242 (0.5431).

NO deploy (513) refit -- 253-only honest cross-fit diagnostic.

Outputs:
  data/processed/nb1251_slsqp_oof.npy    (253,) float32  -- SLSQP cross-fit blend
  data/processed/nb1251_mean_oof.npy     (253,) float32  -- naive 0.5/0.5 mean
  data/processed/nb1251_median_oof.npy   (253,) float32  -- naive median (== mean for K=2)
  data/processed/nb1251_bestw_oof.npy    (253,) float32  -- best fixed-w blend from grid
  data/processed/nb1251_summary.json
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

TAG = "nb1251"
SLSQP_FOLDS = 5
SLSQP_SEED = 42

# Reference numbers (pooled RAE on 253 unblind).
NB1242_REF = 0.5431
NB1211_REF = 0.5451
MARGIN = 0.003


def _slsqp_blend_weights(P_tr: np.ndarray, y_tr: np.ndarray) -> np.ndarray:
    """Argmin MSE over the K-simplex (w_i >= 0, sum w_i = 1)."""
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
    print(f"{TAG} -- combine nb1242 (ChEMBL-kNN feat residual, 0.5431)")
    print(f"          + nb1211 (BoB-of-BoBs blend, 0.5451)")
    print(f"          5-fold SLSQP cross-fit + naive mean + median + asymmetric grid")
    print("=" * 78)

    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)

    paths = {
        "nb1242": DATA_PROCESSED / "nb1242_mean_bag_oof.npy",
        "nb1211": DATA_PROCESSED / "nb1211_mean_oof.npy",
    }
    for k, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"{p} not found ({k})")

    preds = {k: np.load(p).astype(np.float64) for k, p in paths.items()}
    for k, v in preds.items():
        if v.shape[0] != n_unb:
            raise ValueError(f"shape mismatch: {k}={v.shape}, n_unb={n_unb}")

    p1 = preds["nb1242"]
    p2 = preds["nb1211"]

    standalone_rae = {
        "nb1242": float(rae(y_unb, p1)),
        "nb1211": float(rae(y_unb, p2)),
    }
    print("\n[load] standalone pooled RAE on 253 unblind:")
    print(f"   nb1242 ChEMBL-feat residual : {standalone_rae['nb1242']:.4f}  "
          f"(ref {NB1242_REF:.4f})")
    print(f"   nb1211 BoB-of-BoBs blend    : {standalone_rae['nb1211']:.4f}  "
          f"(ref {NB1211_REF:.4f})")

    # Pred-pred and residual Pearson.
    pred_corr = float(np.corrcoef(p1, p2)[0, 1])
    r1 = p1 - y_unb
    r2 = p2 - y_unb
    resid_corr = float(np.corrcoef(r1, r2)[0, 1])
    print(f"\n[diag] pred-pred  Pearson (nb1242, nb1211) = {pred_corr:.4f}")
    print(f"[diag] residual  Pearson (nb1242, nb1211) = {resid_corr:.4f}")
    print(f"[diag] residual std nb1242 = {r1.std():.4f}")
    print(f"[diag] residual std nb1211 = {r2.std():.4f}")

    # ---- SLSQP cross-fit ----
    print("\n" + "-" * 78)
    print("  BLOCK: 2-way SLSQP cross-fit")
    print("-" * 78)
    P = np.column_stack([p1, p2])
    slsqp_oof, fold_records = _slsqp_cross_fit(P, y_unb, SLSQP_FOLDS, SLSQP_SEED)
    rae_slsqp = float(rae(y_unb, slsqp_oof))
    fold_w = np.array([r["weights"] for r in fold_records])
    mean_w = fold_w.mean(axis=0)
    print(f"   per-fold weights (nb1242, nb1211):")
    for rec in fold_records:
        w = rec["weights"]
        print(f"     fold {rec['fold']}: w[nb1242]={w[0]:.4f}  w[nb1211]={w[1]:.4f}")
    print(f"   mean weights across folds:  w[nb1242]={mean_w[0]:.4f}  "
          f"w[nb1211]={mean_w[1]:.4f}")
    print(f"   pooled RAE(SLSQP cross-fit) = {rae_slsqp:.4f}")

    # In-sample SLSQP for diagnostic.
    w_full = _slsqp_blend_weights(P, y_unb)
    p_full = P @ w_full
    rae_full = float(rae(y_unb, p_full))
    print(f"   in-sample SLSQP weights: w[nb1242]={w_full[0]:.4f}  "
          f"w[nb1211]={w_full[1]:.4f}   RAE = {rae_full:.4f}")

    # ---- Naive 0.5/0.5 mean ----
    mean_oof = 0.5 * p1 + 0.5 * p2
    rae_mean = float(rae(y_unb, mean_oof))
    print(f"\n[block] naive 0.5/0.5 mean  RAE = {rae_mean:.4f}")

    # ---- Naive median (for K=2 same as mean, but compute via np.median for fidelity) ----
    median_oof = np.median(P, axis=1)  # for K=2 == mean
    rae_median = float(rae(y_unb, median_oof))
    print(f"[block] naive median        RAE = {rae_median:.4f}  "
          f"(== mean for K=2)")

    # ---- Asymmetric weights grid w on nb1242 in {0.3..0.7 step 0.05} ----
    print("\n" + "-" * 78)
    print("  BLOCK: asymmetric fixed-w grid (w_nb1242 in 0.30..0.70 step 0.05)")
    print("-" * 78)
    grid_w = np.arange(0.30, 0.70 + 1e-9, 0.05)
    grid_results = []
    best_w = None
    best_rae = float("inf")
    best_oof = None
    for w in grid_w:
        blend = w * p1 + (1.0 - w) * p2
        r = float(rae(y_unb, blend))
        grid_results.append({"w_nb1242": float(round(w, 4)),
                             "w_nb1211": float(round(1.0 - w, 4)),
                             "rae": r})
        marker = ""
        if r < best_rae:
            best_rae = r
            best_w = float(round(w, 4))
            best_oof = blend
            marker = "  <-- best so far"
        print(f"   w[nb1242]={w:.2f}  w[nb1211]={1.0 - w:.2f}  RAE={r:.4f}{marker}")
    print(f"   best fixed-w blend: w[nb1242]={best_w:.4f}  "
          f"w[nb1211]={1.0 - best_w:.4f}  RAE={best_rae:.4f}")

    # ---- Verdict ----
    candidates = {
        "slsqp_cross_fit": rae_slsqp,
        "naive_mean":      rae_mean,
        "naive_median":    rae_median,
        "best_fixed_w":    best_rae,
    }
    best_blend_tag = min(candidates, key=candidates.get)
    best_blend_rae = candidates[best_blend_tag]

    nb1242_rae = standalone_rae["nb1242"]
    nb1211_rae = standalone_rae["nb1211"]
    best_standalone_tag = min(standalone_rae, key=standalone_rae.get)
    best_standalone = standalone_rae[best_standalone_tag]

    beats_nb1242 = best_blend_rae < nb1242_rae - MARGIN
    flat_nb1242 = abs(best_blend_rae - nb1242_rae) < MARGIN

    if beats_nb1242:
        verdict = (f"CHEMBL_BOB_BLEND_BEATS_NB1242 "
                   f"({best_blend_tag} @ {best_blend_rae:.4f})")
    elif flat_nb1242:
        verdict = (f"CHEMBL_BOB_BLEND_FLAT_VS_NB1242 "
                   f"({best_blend_tag} @ {best_blend_rae:.4f})")
    else:
        verdict = (f"CHEMBL_BOB_BLEND_HURTS_VS_NB1242 "
                   f"({best_blend_tag} @ {best_blend_rae:.4f})")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"   nb1242 standalone : {nb1242_rae:.4f}  (ref {NB1242_REF:.4f})")
    print(f"   nb1211 standalone : {nb1211_rae:.4f}  (ref {NB1211_REF:.4f})")
    print(f"   best standalone   : {best_standalone:.4f}  ({best_standalone_tag})")
    print(f"")
    print(f"   candidate pooled RAE table:")
    for tag, val in sorted(candidates.items(), key=lambda kv: kv[1]):
        print(f"     {tag:18s} = {val:.4f}")
    print(f"")
    print(f"   best blend                 : {best_blend_rae:.4f}  ({best_blend_tag})")
    print(f"   delta vs nb1242 (0.5431)   : {best_blend_rae - nb1242_rae:+.4f}")
    print(f"   beats_nb1242 (>= {MARGIN})    : {beats_nb1242}")
    print(f"   verdict                    : {verdict}")

    # Persist canonical artifacts.
    np.save(DATA_PROCESSED / f"{TAG}_slsqp_oof.npy",
            slsqp_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_mean_oof.npy",
            mean_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_median_oof.npy",
            median_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_bestw_oof.npy",
            best_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_slsqp_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_mean_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_median_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_bestw_oof.npy'}")

    summary = {
        "tag": TAG,
        "n_unb": n_unb,
        "slsqp_folds": SLSQP_FOLDS,
        "slsqp_seed": SLSQP_SEED,
        "components": list(paths.keys()),
        "standalone_rae": standalone_rae,
        "pred_corr":  pred_corr,
        "residual_corr": resid_corr,
        "residual_std_nb1242": float(r1.std()),
        "residual_std_nb1211": float(r2.std()),
        "slsqp_fold_records": fold_records,
        "slsqp_mean_fold_weights": [float(x) for x in mean_w],
        "slsqp_in_sample_weights": [float(x) for x in w_full],
        "slsqp_in_sample_rae": rae_full,
        "rae_slsqp_cross_fit": rae_slsqp,
        "rae_naive_mean": rae_mean,
        "rae_naive_median": rae_median,
        "grid_w_results": grid_results,
        "best_fixed_w_nb1242": best_w,
        "best_fixed_w_nb1211": float(1.0 - best_w),
        "rae_best_fixed_w": best_rae,
        "candidate_rae_table": candidates,
        "best_blend_tag": best_blend_tag,
        "best_blend_rae": best_blend_rae,
        "best_standalone_tag": best_standalone_tag,
        "best_standalone_rae": best_standalone,
        "nb1242_ref": NB1242_REF,
        "nb1211_ref": NB1211_REF,
        "delta_best_vs_nb1242": best_blend_rae - nb1242_rae,
        "delta_best_vs_nb1211": best_blend_rae - nb1211_rae,
        "beats_nb1242": bool(beats_nb1242),
        "flat_vs_nb1242": bool(flat_nb1242),
        "margin": MARGIN,
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
    for k in ("standalone_rae",
              "pred_corr", "residual_corr",
              "slsqp_mean_fold_weights", "rae_slsqp_cross_fit",
              "rae_naive_mean", "rae_naive_median",
              "best_fixed_w_nb1242", "rae_best_fixed_w",
              "candidate_rae_table",
              "best_blend_tag", "best_blend_rae",
              "delta_best_vs_nb1242",
              "beats_nb1242", "verdict"):
        print(f"  {k}: {res.get(k)}")
