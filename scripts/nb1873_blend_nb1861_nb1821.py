"""nb1873 -- Blend nb1861 + nb1821 (both lowest 25-bag candidates).

HYPOTHESIS
    nb1861 (LightGBM regression objective, pooled 25-bag RAE 0.5013) and
    nb1821 (LightGBM huber alpha=1.0 + goss, pooled 25-bag RAE 0.5025) share
    the same 117-D feature stack (AtomPair-25 + MACCS-20 + Mordred-20 +
    ChempropEmbed-20 + Avalon-30 + pred_chembl_pec50 + mean_sim) but differ
    on the residual-fit loss. Loss-shape orthogonality (L2 vs Huber+goss)
    may produce residual decorrelation worth a small blend gain.

PROTOCOL
    1. Load nb1861_pooled_25bag_oof.npy (0.5013) and nb1821_pooled_25bag_oof.npy
       (0.5025).
    2. Pearson on raw preds AND residuals-from-truth.
    3. Grid sweep w in {0.0..1.0 step 0.05} over w*nb1861 + (1-w)*nb1821.
    4. 5-fold SLSQP cross-fit (simplex K=2).
    5. Verdict at 0.003 margin vs nb1861 (0.5013).

OUTPUTS
    data/processed/nb1873_summary.json
    data/processed/nb1873_best_oof.npy   (253,) float32  best variant predictor
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

TAG = "nb1873"
SLSQP_FOLDS = 5
SLSQP_SEED = 42

# Reference numbers (pooled 25-bag RAE on 253 unblind) from prior summaries.
NB1861_REF = 0.5013   # LightGBM regression objective
NB1821_REF = 0.5025   # LightGBM huber alpha=1.0 + goss

ANCHOR_REF = NB1861_REF   # gate vs lowest standalone
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


def _grid_2way(p_a: np.ndarray, p_b: np.ndarray, y: np.ndarray,
               step: float = 0.05
               ) -> tuple[list[dict], dict]:
    records: list[dict] = []
    ws = np.arange(0.0, 1.0 + step / 2, step)
    for w in ws:
        blend = w * p_a + (1.0 - w) * p_b
        r = float(rae(y, blend))
        records.append({"w_a": float(w), "rae": r})
    best = min(records, key=lambda r: r["rae"])
    return records, best


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- 2-way blend  nb1861 (L2) + nb1821 (Huber+goss)")
    print(f"         grid sweep (step 0.05) + 5-fold SLSQP cross-fit")
    print(f"         gate vs nb1861 {ANCHOR_REF:.4f}  margin {MARGIN:.3f}")
    print("=" * 78)

    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)

    p_a_path = DATA_PROCESSED / "nb1861_pooled_25bag_oof.npy"
    p_b_path = DATA_PROCESSED / "nb1821_pooled_25bag_oof.npy"
    for tag, p in (("nb1861", p_a_path), ("nb1821", p_b_path)):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found ({tag}).")

    p_a = np.load(p_a_path).astype(np.float64)  # nb1861
    p_b = np.load(p_b_path).astype(np.float64)  # nb1821

    for tag, arr in (("nb1861", p_a), ("nb1821", p_b)):
        if arr.shape[0] != n_unb:
            raise ValueError(
                f"shape mismatch: {tag}={arr.shape}, n_unb={n_unb}")

    rae_a = float(rae(y_unb, p_a))
    rae_b = float(rae(y_unb, p_b))
    print(f"[load] nb1861 (L2)         RAE = {rae_a:.4f}  "
          f"(ref {NB1861_REF:.4f})")
    print(f"[load] nb1821 (Huber+goss) RAE = {rae_b:.4f}  "
          f"(ref {NB1821_REF:.4f})")

    # Pearson on raw preds AND residuals-from-truth.
    corr_pred = float(np.corrcoef(p_a, p_b)[0, 1])
    resid_a = p_a - y_unb
    resid_b = p_b - y_unb
    corr_resid = float(np.corrcoef(resid_a, resid_b)[0, 1])
    print(f"[diag] Pearson pred(nb1861, nb1821)     = {corr_pred:+.4f}")
    print(f"[diag] Pearson residual(nb1861, nb1821) = {corr_resid:+.4f}")

    # (A) Grid sweep.
    print("\n" + "-" * 78)
    print("(A) Grid sweep  w*nb1861 + (1-w)*nb1821   (step 0.05)")
    print("-" * 78)
    grid_records, best_grid = _grid_2way(p_a, p_b, y_unb)
    top5 = sorted(grid_records, key=lambda r: r["rae"])[:5]
    print("   top-5 grid:")
    for r in top5:
        print(f"     w_a={r['w_a']:.2f}   RAE={r['rae']:.4f}")
    print(f"   best grid : w_a={best_grid['w_a']:.2f}  "
          f"RAE={best_grid['rae']:.4f}")

    # (B) SLSQP cross-fit (K=2).
    print("\n" + "-" * 78)
    print("(B) 5-fold SLSQP cross-fit (simplex w>=0, sum==1, K=2)")
    print("-" * 78)
    P = np.column_stack([p_a, p_b])  # (253, 2)
    slsqp_oof, fold_records = _slsqp_cross_fit(
        P, y_unb, n_splits=SLSQP_FOLDS, seed=SLSQP_SEED)
    rae_slsqp = float(rae(y_unb, slsqp_oof))
    print(f"   SLSQP cross-fit  RAE = {rae_slsqp:.4f}")

    fw = np.array([r["weights"] for r in fold_records])
    mean_w = fw.mean(axis=0).tolist()
    std_w = fw.std(axis=0).tolist()
    print(f"   mean fold weights   w_nb1861={mean_w[0]:.4f}  "
          f"w_nb1821={mean_w[1]:.4f}  (std {std_w[0]:.4f}/{std_w[1]:.4f})")

    # In-sample SLSQP (diagnostic).
    w_full = _slsqp_blend_weights(P, y_unb)
    p_full = P @ w_full
    rae_full = float(rae(y_unb, p_full))
    print(f"   in-sample SLSQP  w={w_full.tolist()}  RAE={rae_full:.4f}")

    # Choose best variant across (grid, slsqp_cross_fit).
    candidates: list[dict] = []
    candidates.append({
        "name": "grid",
        "rae": best_grid["rae"], "w_a": best_grid["w_a"],
        "pred": best_grid["w_a"] * p_a + (1.0 - best_grid["w_a"]) * p_b,
    })
    candidates.append({
        "name": "slsqp_cross_fit",
        "rae": rae_slsqp, "w_a": None,
        "pred": slsqp_oof,
    })
    best = min(candidates, key=lambda c: c["rae"])
    best_rae = float(best["rae"])
    best_name = best["name"]

    beats_nb1861 = best_rae < ANCHOR_REF - MARGIN
    flat_vs_nb1861 = abs(best_rae - ANCHOR_REF) < MARGIN

    if beats_nb1861:
        verdict = f"NB1873_BEATS_NB1861_BY_MARGIN  (best={best_name})"
    elif flat_vs_nb1861:
        verdict = f"NB1873_FLAT_VS_NB1861  (best={best_name})"
    else:
        verdict = f"NB1873_HURTS_VS_NB1861  (best={best_name})"

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"   nb1861 standalone : {rae_a:.4f}")
    print(f"   nb1821 standalone : {rae_b:.4f}")
    print(f"   grid best         : {best_grid['rae']:.4f}  "
          f"@ w_a={best_grid['w_a']:.2f}")
    print(f"   SLSQP cross-fit   : {rae_slsqp:.4f}")
    print(f"   anchor nb1861     : {ANCHOR_REF:.4f}")
    print(f"   best variant      : {best_name}  ({best_rae:.4f})")
    print(f"   margin vs nb1861  : "
          f"{best_rae - ANCHOR_REF:+.4f} (gate {MARGIN:.3f})")
    print(f"   beats_nb1861      : {beats_nb1861}")
    print(f"   verdict           : {verdict}")

    best_oof = np.asarray(best["pred"], dtype=np.float32)
    np.save(DATA_PROCESSED / f"{TAG}_best_oof.npy", best_oof)
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_best_oof.npy'}  "
          f"(variant={best_name})")

    summary = {
        "tag": TAG,
        "n_unb": n_unb,
        "slsqp_folds": SLSQP_FOLDS,
        "slsqp_seed": SLSQP_SEED,
        "components": [
            "nb1861_pooled_25bag_oof",
            "nb1821_pooled_25bag_oof",
        ],
        "rae_nb1861_standalone": rae_a,
        "rae_nb1821_standalone": rae_b,
        "pred_corr_pearson": corr_pred,
        "residual_corr_pearson": corr_resid,
        "grid_top5": top5,
        "grid_best": best_grid,
        "grid_records": grid_records,
        "rae_slsqp_cross_fit": rae_slsqp,
        "fold_records_slsqp": fold_records,
        "mean_fold_weights_slsqp": mean_w,
        "std_fold_weights_slsqp": std_w,
        "in_sample_slsqp_weights": [float(x) for x in w_full],
        "in_sample_slsqp_rae": rae_full,
        "best_variant": best_name,
        "best_rae": best_rae,
        "best_w_a_nb1861": best.get("w_a"),
        "delta_best_vs_nb1861": best_rae - ANCHOR_REF,
        "beats_nb1861": bool(beats_nb1861),
        "flat_vs_nb1861": bool(flat_vs_nb1861),
        "verdict": verdict,
        "nb1861_ref": NB1861_REF,
        "nb1821_ref": NB1821_REF,
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
    for k in (
        "rae_nb1861_standalone",
        "rae_nb1821_standalone",
        "pred_corr_pearson",
        "residual_corr_pearson",
        "grid_best",
        "rae_slsqp_cross_fit",
        "mean_fold_weights_slsqp",
        "in_sample_slsqp_weights",
        "in_sample_slsqp_rae",
        "best_variant", "best_rae", "best_w_a_nb1861",
        "delta_best_vs_nb1861",
        "beats_nb1861", "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
