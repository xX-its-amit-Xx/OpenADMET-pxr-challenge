"""nb1232 -- Multi-anchor blend of nb1211 (0.5451) + nb1231 (chemprop_aux + MACCS).

HYPOTHESIS
----------
nb1211 anchors everything on nb1070 (a heavily post-processed stretch median
bag).  nb1231 swaps the anchor to chemprop_aux (raw MTL MPNN, no stretch) and
runs a MACCS-167 residual mean-bag on top, landing standalone at 0.5805 (worse
than nb1211 standalone 0.5451).  But the residual correlation across the two
anchors should be LOW because the two anchors live on different post-processing
manifolds -- so a convex blend of the two predictors may extract orthogonal
chemistry signal that neither captures alone.

PROTOCOL
--------
1. Load nb1211_mean_oof.npy (block_mean naive mean, 0.5451) and
   nb1231_mean_bag_oof.npy (mean over 5 inner seeds, 0.5805).
2. Compute pred-pred Pearson and residual Pearson.
3. 5-fold cross-fit SLSQP convex blend (simplex weights), seed 42.
4. Also report naive mean and naive median across the two.
5. Verdict at 0.003 margin vs nb1211 standalone (0.5451):
     BEATS  if rae < 0.5451 - 0.003 = 0.5421
     TIES   if 0.5421 <= rae <= 0.5481
     LOSES  if rae > 0.5481.

OUTPUTS
-------
  data/processed/nb1232_slsqp_oof.npy    (253,) float32  cross-fit SLSQP
  data/processed/nb1232_mean_oof.npy     (253,) float32  naive 50/50 mean
  data/processed/nb1232_median_oof.npy   (253,) float32  naive median (= mean for n=2)
  data/processed/nb1232_summary.json
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
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1232"

NB1211_PATH = DATA_PROCESSED / "nb1211_mean_oof.npy"
NB1231_PATH = DATA_PROCESSED / "nb1231_mean_bag_oof.npy"

NB1211_REF = 0.5451
MARGIN = 0.003

SLSQP_FOLDS = 5
SLSQP_SEED = 42


def _slsqp_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Convex SLSQP fit for simplex weights minimizing pooled RAE."""
    k = P.shape[1]
    w0 = np.full(k, 1.0 / k)

    def loss(w):
        return float(rae(y, P @ w))

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bounds = [(0.0, 1.0)] * k
    res = minimize(loss, w0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 200, "ftol": 1e-8})
    w = np.asarray(res.x, dtype=np.float64)
    w = np.clip(w, 0.0, 1.0)
    if w.sum() == 0:
        w = np.full(k, 1.0 / k)
    else:
        w = w / w.sum()
    return w


def _crossfit_slsqp(P: np.ndarray, y: np.ndarray,
                    n_splits: int, seed: int) -> tuple[np.ndarray, list]:
    n = len(y)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_records: list[dict] = []
    for k_idx, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n))):
        w = _slsqp_weights(P[tr_loc], y[tr_loc])
        oof[va_loc] = P[va_loc] @ w
        fold_records.append({
            "fold": int(k_idx),
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "weights": [float(x) for x in w],
        })
    return oof, fold_records


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Multi-anchor blend nb1211 (0.5451) + nb1231 (chemprop_aux+MACCS)")
    print(f"     SLSQP folds={SLSQP_FOLDS}  seed={SLSQP_SEED}")
    print(f"     verdict margin = {MARGIN:.3f} vs nb1211 ref {NB1211_REF:.4f}")
    print("=" * 78)

    if not NB1211_PATH.exists():
        raise FileNotFoundError(f"{NB1211_PATH} missing")
    if not NB1231_PATH.exists():
        raise FileNotFoundError(f"{NB1231_PATH} missing -- run nb1231 first")
    p_nb1211 = np.load(NB1211_PATH).astype(np.float64)
    p_nb1231 = np.load(NB1231_PATH).astype(np.float64)
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n = len(y_unb)
    if p_nb1211.shape != (n,) or p_nb1231.shape != (n,):
        raise ValueError(
            f"shape mismatch: nb1211 {p_nb1211.shape}, nb1231 {p_nb1231.shape}, n={n}"
        )

    rae_a = float(rae(y_unb, p_nb1211))
    rae_b = float(rae(y_unb, p_nb1231))
    print(f"[load] nb1211 standalone pooled RAE = {rae_a:.4f}")
    print(f"[load] nb1231 standalone pooled RAE = {rae_b:.4f}")

    # ---- Diagnostics: pred-pred and residual Pearson ----
    pred_corr = float(np.corrcoef(p_nb1211, p_nb1231)[0, 1])
    res_a = y_unb - p_nb1211
    res_b = y_unb - p_nb1231
    resid_corr = float(np.corrcoef(res_a, res_b)[0, 1])
    print(f"[diag] pred-pred Pearson     = {pred_corr:.4f}")
    print(f"[diag] residual  Pearson     = {resid_corr:.4f}")

    P = np.stack([p_nb1211, p_nb1231], axis=1)

    # ---- Naive mean / median (identical for k=2) ----
    naive_mean = P.mean(axis=1)
    naive_median = np.median(P, axis=1)
    rae_naive_mean = float(rae(y_unb, naive_mean))
    rae_naive_median = float(rae(y_unb, naive_median))
    print(f"[naive] mean   pooled RAE = {rae_naive_mean:.4f}")
    print(f"[naive] median pooled RAE = {rae_naive_median:.4f}")

    # ---- 5-fold cross-fit SLSQP ----
    slsqp_oof, fold_records = _crossfit_slsqp(P, y_unb, SLSQP_FOLDS, SLSQP_SEED)
    rae_slsqp_cf = float(rae(y_unb, slsqp_oof))
    in_sample_w = _slsqp_weights(P, y_unb)
    in_sample_blend = P @ in_sample_w
    rae_in_sample = float(rae(y_unb, in_sample_blend))
    mean_fold_w = np.mean(
        np.asarray([r["weights"] for r in fold_records]), axis=0
    ).tolist()
    print(f"[slsqp] cross-fit pooled RAE  = {rae_slsqp_cf:.4f}")
    print(f"[slsqp] in-sample weights     = {[round(x,4) for x in in_sample_w]}  "
          f"rae={rae_in_sample:.4f}")
    print(f"[slsqp] mean fold weights     = {[round(x,4) for x in mean_fold_w]}")

    # ---- Verdict at margin vs nb1211 ----
    candidate_table = {
        "slsqp_cross_fit": rae_slsqp_cf,
        "naive_mean": rae_naive_mean,
        "naive_median": rae_naive_median,
    }
    best_tag = min(candidate_table, key=candidate_table.get)
    best_rae = candidate_table[best_tag]
    delta = best_rae - NB1211_REF
    beats = best_rae < (NB1211_REF - MARGIN)
    ties = abs(delta) <= MARGIN
    if beats:
        verdict = f"NB1232_BEATS_NB1211 ({best_tag} @ {best_rae:.4f})"
    elif ties:
        verdict = f"NB1232_TIES_NB1211 ({best_tag} @ {best_rae:.4f})"
    else:
        verdict = f"NB1232_LOSES_TO_NB1211 ({best_tag} @ {best_rae:.4f})"

    print("\n" + "=" * 78)
    print(f"   best_tag = {best_tag}")
    print(f"   best_rae = {best_rae:.4f}   delta vs nb1211 = {delta:+.4f}")
    print(f"   VERDICT  = {verdict}")
    print("=" * 78)

    # ---- Save artefacts ----
    np.save(DATA_PROCESSED / f"{TAG}_slsqp_oof.npy",
            slsqp_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_mean_oof.npy",
            naive_mean.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_median_oof.npy",
            naive_median.astype(np.float32))
    print(f"[save] {DATA_PROCESSED / f'{TAG}_slsqp_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_mean_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_median_oof.npy'}")

    summary = {
        "tag": TAG,
        "n_unb": n,
        "components": ["nb1211_mean_oof", "nb1231_mean_bag_oof"],
        "nb1211_ref": NB1211_REF,
        "margin": MARGIN,
        "standalone_rae": {
            "nb1211_mean_oof": rae_a,
            "nb1231_mean_bag_oof": rae_b,
        },
        "pred_pred_pearson": pred_corr,
        "residual_pearson": resid_corr,
        "slsqp_folds": SLSQP_FOLDS,
        "slsqp_seed": SLSQP_SEED,
        "slsqp_fold_records": fold_records,
        "slsqp_in_sample_weights": [float(x) for x in in_sample_w],
        "slsqp_in_sample_rae": rae_in_sample,
        "slsqp_mean_fold_weights": mean_fold_w,
        "slsqp_cross_fit_rae": rae_slsqp_cf,
        "naive_mean_rae": rae_naive_mean,
        "naive_median_rae": rae_naive_median,
        "candidate_table": candidate_table,
        "best_tag": best_tag,
        "best_rae": best_rae,
        "delta_vs_nb1211": delta,
        "beats_nb1211": bool(beats),
        "ties_nb1211": bool(ties and not beats),
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("standalone_rae", "pred_pred_pearson", "residual_pearson",
              "slsqp_mean_fold_weights", "slsqp_in_sample_weights",
              "slsqp_cross_fit_rae", "naive_mean_rae", "naive_median_rae",
              "best_tag", "best_rae", "delta_vs_nb1211",
              "beats_nb1211", "verdict"):
        print(f"  {k}: {res.get(k)}")
