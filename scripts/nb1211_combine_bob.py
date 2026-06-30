"""nb1211 -- Combine nb1190 BoB OOF + nb1200 BoB OOF (two ensembles, disjoint feature families).

Hypothesis:
    nb1190 BoB is a triple-mean of nb1130 (Morgan+RDKit residual), nb1153
    (Mordred residual), and nb1172 (AtomPair residual) at pooled RAE ~0.5499.
    nb1200 BoB is a MACCS-only residual bag at pooled RAE 0.5491-0.5495.

    Both ride the same nb1070 anchor but the BoB ensembles draw on disjoint
    chemistry-feature families:
        nb1190 = Morgan + RDKit-desc + Mordred + AtomPair (mostly topology /
                 fragment counts / autocorrelation)
        nb1200 = MACCS (167 substructure keys)
    MACCS encodes a set of hand-curated substructure SMARTS that are not
    densely represented in Morgan/RDKit-desc/Mordred/AtomPair, so the two BoB
    signals may capture different chunks of the orthogonal-to-nb1070 variance.

Protocol:
  1. Load nb1190_bob_mean_oof.npy  (253,)  -- triple-mean BoB (Mor+RDKit, Mordred, AP)
  2. Load nb1200_bob_mean_oof.npy  (253,)  -- MACCS BoB mean
  3. Load nb1190_bob_median_oof.npy (253,) -- triple-mean BoB median
  4. Load nb1200_bob_median_oof.npy (253,) -- MACCS BoB median
  5. 5-fold cross-fit SLSQP blend on (nb1190_mean, nb1200_mean):
        for each fold:
            fit w = argmin ||y_tr - w0*p1_tr - w1*p2_tr||^2,
                s.t. w >= 0, sum(w) == 1
            predict on val with frozen w
        emit blended OOF; pooled RAE.
  6. Naive 0.5/0.5 mean RAE (mean variants).
  7. Naive median RAE (mean variants).
  8. Also: MEDIAN variants (median oofs) -- SLSQP, mean, median.
  9. Also: MIXED (mean+median of all 4 sources) -- SLSQP, mean, median.
 10. Verdict at 0.003 margin vs nb1200 BoB median (0.5491) and nb1200 BoB mean (0.5495).

NO deploy (513) refit -- 253-only honest cross-fit diagnostic.

Outputs:
  data/processed/nb1211_slsqp_oof.npy   (253,) float32  -- cross-fit SLSQP blend of mean variants
  data/processed/nb1211_mean_oof.npy    (253,) float32  -- naive mean of mean variants
  data/processed/nb1211_median_oof.npy  (253,) float32  -- naive median of mean variants
  data/processed/nb1211_summary.json
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

TAG = "nb1211"
SLSQP_FOLDS = 5
SLSQP_SEED = 42

# Reference numbers (pooled RAE on 253 unblind).
NB1190_BOB_MEAN_REF = 0.5499
NB1200_BOB_MEAN_REF = 0.5495
NB1200_BOB_MEDIAN_REF = 0.5491


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


def _eval_block(label: str, P: np.ndarray, y: np.ndarray,
                comp_tags: list[str]) -> dict:
    """Run SLSQP cross-fit + mean + median on a stack P (n, K)."""
    print("\n" + "-" * 78)
    print(f"  BLOCK: {label}   K={P.shape[1]}  components={comp_tags}")
    print("-" * 78)

    slsqp_oof, fold_records = _slsqp_cross_fit(P, y, SLSQP_FOLDS, SLSQP_SEED)
    rae_slsqp = float(rae(y, slsqp_oof))

    fold_weights = np.array([r["weights"] for r in fold_records])
    mean_weights = fold_weights.mean(axis=0)
    print(f"   per-fold weights:")
    for rec in fold_records:
        w = rec["weights"]
        wstr = "  ".join(f"w[{comp_tags[i]}]={w[i]:.4f}" for i in range(len(w)))
        print(f"     fold {rec['fold']}: {wstr}")
    mstr = "  ".join(f"w[{comp_tags[i]}]={mean_weights[i]:.4f}"
                     for i in range(len(mean_weights)))
    print(f"   mean weights across folds:  {mstr}")
    print(f"   pooled RAE(SLSQP cross-fit) = {rae_slsqp:.4f}")

    w_full = _slsqp_blend_weights(P, y)
    p_full = P @ w_full
    rae_full = float(rae(y, p_full))
    fstr = "  ".join(f"w[{comp_tags[i]}]={w_full[i]:.4f}"
                     for i in range(len(w_full)))
    print(f"   in-sample SLSQP weights:    {fstr}   RAE = {rae_full:.4f}")

    mean_oof = P.mean(axis=1)
    rae_mean = float(rae(y, mean_oof))
    print(f"   pooled RAE(naive mean)   = {rae_mean:.4f}")

    median_oof = np.median(P, axis=1)
    rae_median = float(rae(y, median_oof))
    print(f"   pooled RAE(naive median) = {rae_median:.4f}")

    return {
        "label": label,
        "components": comp_tags,
        "fold_records": fold_records,
        "mean_fold_weights": [float(x) for x in mean_weights],
        "in_sample_slsqp_weights": [float(x) for x in w_full],
        "in_sample_slsqp_rae": rae_full,
        "rae_slsqp_cross_fit": rae_slsqp,
        "rae_naive_mean": rae_mean,
        "rae_naive_median": rae_median,
        "slsqp_oof": slsqp_oof,
        "mean_oof": mean_oof,
        "median_oof": median_oof,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- combine nb1190 BoB (triple-mean: Mor+RDKit / Mordred / AP) +")
    print(f"          nb1200 BoB (MACCS) -- disjoint feature families over nb1070")
    print(f"          5-fold SLSQP cross-fit blend + mean + median")
    print(f"          MEAN and MEDIAN OOF variants, plus MIXED 4-way block")
    print("=" * 78)

    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)

    paths = {
        "nb1190_mean":   DATA_PROCESSED / "nb1190_bob_mean_oof.npy",
        "nb1200_mean":   DATA_PROCESSED / "nb1200_bob_mean_oof.npy",
        "nb1190_median": DATA_PROCESSED / "nb1190_bob_median_oof.npy",
        "nb1200_median": DATA_PROCESSED / "nb1200_bob_median_oof.npy",
    }
    for k, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"{p} not found ({k})")

    preds = {k: np.load(p).astype(np.float64) for k, p in paths.items()}
    for k, v in preds.items():
        if v.shape[0] != n_unb:
            raise ValueError(f"shape mismatch: {k}={v.shape}, n_unb={n_unb}")

    standalone_rae = {k: float(rae(y_unb, v)) for k, v in preds.items()}
    print("\n[load] standalone pooled RAE on 253 unblind:")
    print(f"   nb1190 BoB mean   : {standalone_rae['nb1190_mean']:.4f}  "
          f"(ref {NB1190_BOB_MEAN_REF:.4f})")
    print(f"   nb1200 BoB mean   : {standalone_rae['nb1200_mean']:.4f}  "
          f"(ref {NB1200_BOB_MEAN_REF:.4f})")
    print(f"   nb1190 BoB median : {standalone_rae['nb1190_median']:.4f}")
    print(f"   nb1200 BoB median : {standalone_rae['nb1200_median']:.4f}  "
          f"(ref {NB1200_BOB_MEDIAN_REF:.4f})")

    # Pred-pred and residual Pearson.
    p1m = preds["nb1190_mean"]
    p2m = preds["nb1200_mean"]
    p1md = preds["nb1190_median"]
    p2md = preds["nb1200_median"]

    pred_corr_mean = float(np.corrcoef(p1m, p2m)[0, 1])
    resid_corr_mean = float(np.corrcoef(p1m - y_unb, p2m - y_unb)[0, 1])
    pred_corr_median = float(np.corrcoef(p1md, p2md)[0, 1])
    resid_corr_median = float(np.corrcoef(p1md - y_unb, p2md - y_unb)[0, 1])
    print("\n[diag] pred corr   (nb1190_mean,   nb1200_mean)   = "
          f"{pred_corr_mean:.4f}")
    print(f"[diag] resid corr (nb1190_mean,   nb1200_mean)   = "
          f"{resid_corr_mean:.4f}")
    print(f"[diag] pred corr   (nb1190_median, nb1200_median) = "
          f"{pred_corr_median:.4f}")
    print(f"[diag] resid corr (nb1190_median, nb1200_median) = "
          f"{resid_corr_median:.4f}")

    # ---- Block A: MEAN variants (2-way) ----
    P_mean = np.column_stack([p1m, p2m])
    block_mean = _eval_block(
        "MEAN variants", P_mean, y_unb,
        comp_tags=["nb1190_mean", "nb1200_mean"],
    )

    # ---- Block B: MEDIAN variants (2-way) ----
    P_median = np.column_stack([p1md, p2md])
    block_median = _eval_block(
        "MEDIAN variants", P_median, y_unb,
        comp_tags=["nb1190_median", "nb1200_median"],
    )

    # ---- Block C: MIXED (mean + median of both) (4-way) ----
    P_mixed = np.column_stack([p1m, p2m, p1md, p2md])
    block_mixed = _eval_block(
        "MIXED (mean+median 4-way)", P_mixed, y_unb,
        comp_tags=["nb1190_mean", "nb1200_mean", "nb1190_median", "nb1200_median"],
    )

    # ---- Verdict ----
    # Primary baseline: nb1200 BoB median (best standalone) at 0.003 margin.
    best_standalone_tag = min(standalone_rae, key=standalone_rae.get)
    best_standalone = standalone_rae[best_standalone_tag]
    nb1200_median_rae = standalone_rae["nb1200_median"]
    nb1200_mean_rae = standalone_rae["nb1200_mean"]

    # Best blend across all 9 candidates (3 blocks x {slsqp, mean, median}).
    candidates = {
        "MEAN_slsqp":      block_mean["rae_slsqp_cross_fit"],
        "MEAN_mean":       block_mean["rae_naive_mean"],
        "MEAN_median":     block_mean["rae_naive_median"],
        "MEDIAN_slsqp":    block_median["rae_slsqp_cross_fit"],
        "MEDIAN_mean":     block_median["rae_naive_mean"],
        "MEDIAN_median":   block_median["rae_naive_median"],
        "MIXED_slsqp":     block_mixed["rae_slsqp_cross_fit"],
        "MIXED_mean":      block_mixed["rae_naive_mean"],
        "MIXED_median":    block_mixed["rae_naive_median"],
    }
    best_blend_tag = min(candidates, key=candidates.get)
    best_blend_rae = candidates[best_blend_tag]

    beats_nb1200_median = best_blend_rae < nb1200_median_rae - 0.003
    beats_nb1200_mean = best_blend_rae < nb1200_mean_rae - 0.003

    if beats_nb1200_median:
        verdict = f"BOB_BLEND_BEATS_NB1200_BOB_MEDIAN ({best_blend_tag} @ {best_blend_rae:.4f})"
    elif beats_nb1200_mean:
        verdict = f"BOB_BLEND_BEATS_NB1200_BOB_MEAN_ONLY ({best_blend_tag} @ {best_blend_rae:.4f})"
    elif abs(best_blend_rae - nb1200_median_rae) < 0.003:
        verdict = f"BOB_BLEND_FLAT_VS_NB1200_BOB_MEDIAN ({best_blend_tag} @ {best_blend_rae:.4f})"
    else:
        verdict = f"BOB_BLEND_HURTS_VS_NB1200_BOB_MEDIAN ({best_blend_tag} @ {best_blend_rae:.4f})"

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"   nb1190 BoB mean   standalone : {standalone_rae['nb1190_mean']:.4f}")
    print(f"   nb1200 BoB mean   standalone : {standalone_rae['nb1200_mean']:.4f}")
    print(f"   nb1190 BoB median standalone : {standalone_rae['nb1190_median']:.4f}")
    print(f"   nb1200 BoB median standalone : {standalone_rae['nb1200_median']:.4f}")
    print(f"   best standalone              : {best_standalone:.4f}  "
          f"({best_standalone_tag})")
    print(f"")
    print(f"   candidate pooled RAE table:")
    for tag, val in sorted(candidates.items(), key=lambda kv: kv[1]):
        print(f"     {tag:18s} = {val:.4f}")
    print(f"")
    print(f"   best blend                   : {best_blend_rae:.4f}  ({best_blend_tag})")
    print(f"   delta vs nb1200 BoB median   : "
          f"{best_blend_rae - nb1200_median_rae:+.4f}")
    print(f"   delta vs nb1200 BoB mean     : "
          f"{best_blend_rae - nb1200_mean_rae:+.4f}")
    print(f"   beats_nb1200_bob_median (>= 0.003) : {beats_nb1200_median}")
    print(f"   beats_nb1200_bob_mean   (>= 0.003) : {beats_nb1200_mean}")
    print(f"   verdict                      : {verdict}")

    # Persist canonical artifacts: use the MEAN block as canonical.
    np.save(DATA_PROCESSED / f"{TAG}_slsqp_oof.npy",
            block_mean["slsqp_oof"].astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_mean_oof.npy",
            block_mean["mean_oof"].astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_median_oof.npy",
            block_mean["median_oof"].astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_slsqp_oof.npy'}  "
          f"(MEAN-block SLSQP cross-fit)")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_mean_oof.npy'}   "
          f"(MEAN-block naive mean)")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_median_oof.npy'} "
          f"(MEAN-block naive median)")

    def _block_summary(block: dict) -> dict:
        return {k: v for k, v in block.items()
                if k not in ("slsqp_oof", "mean_oof", "median_oof")}

    summary = {
        "tag": TAG,
        "n_unb": n_unb,
        "slsqp_folds": SLSQP_FOLDS,
        "slsqp_seed": SLSQP_SEED,
        "components": list(paths.keys()),
        "standalone_rae": standalone_rae,
        "pred_corr_mean_variants":    pred_corr_mean,
        "residual_corr_mean_variants": resid_corr_mean,
        "pred_corr_median_variants":   pred_corr_median,
        "residual_corr_median_variants": resid_corr_median,
        "block_mean":   _block_summary(block_mean),
        "block_median": _block_summary(block_median),
        "block_mixed":  _block_summary(block_mixed),
        "candidate_rae_table": candidates,
        "best_blend_tag": best_blend_tag,
        "best_blend_rae": best_blend_rae,
        "best_standalone_tag": best_standalone_tag,
        "best_standalone_rae": best_standalone,
        "nb1200_bob_median_ref": NB1200_BOB_MEDIAN_REF,
        "nb1200_bob_mean_ref": NB1200_BOB_MEAN_REF,
        "nb1190_bob_mean_ref": NB1190_BOB_MEAN_REF,
        "delta_best_vs_nb1200_bob_median":
            best_blend_rae - standalone_rae["nb1200_median"],
        "delta_best_vs_nb1200_bob_mean":
            best_blend_rae - standalone_rae["nb1200_mean"],
        "beats_nb1200_bob_median": bool(beats_nb1200_median),
        "beats_nb1200_bob_mean": bool(beats_nb1200_mean),
        "beats_nb1200_bob": bool(beats_nb1200_median or beats_nb1200_mean),
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
              "pred_corr_mean_variants", "residual_corr_mean_variants",
              "pred_corr_median_variants", "residual_corr_median_variants",
              "candidate_rae_table",
              "best_blend_tag", "best_blend_rae",
              "delta_best_vs_nb1200_bob_median",
              "delta_best_vs_nb1200_bob_mean",
              "beats_nb1200_bob_median", "beats_nb1200_bob_mean",
              "beats_nb1200_bob", "verdict"):
        print(f"  {k}: {res.get(k)}")
