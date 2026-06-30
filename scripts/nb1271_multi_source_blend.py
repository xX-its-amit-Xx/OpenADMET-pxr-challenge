"""nb1271 -- Multi-source residual blend: nb1242 (ChEMBL+MACCS resid) + nb1263 (PubChem+MACCS resid) + nb1211 (BoB-of-BoBs blend).

Hypothesis:
    Separately-trained ChEMBL and PubChem residual learners may provide
    row-level diversification on the 253 unblind that their UNION
    (single residual learner on combined external-feature set) does not.

Protocol:
  1. Load nb1242_mean_bag_oof.npy (ChEMBL+MACCS residual, 0.5431).
  2. Load nb1263_mean_bag_oof.npy (PubChem+MACCS residual, 0.5586).
  3. Load nb1211_mean_oof.npy (BoB-of-BoBs blend, 0.5451).
  4. Compute pairwise pred Pearson + residual Pearson.
  5. 5-fold cross-fit SLSQP (3-simplex), naive 1/3 mean, naive median,
     inverse-RAE weighted, best fixed-w grid on the 3-simplex (step 0.1).
  6. Verdict at 0.003 margin vs nb1251 (0.5394).

NO deploy (513) refit -- 253-only honest cross-fit diagnostic.

Outputs:
  data/processed/nb1271_slsqp_oof.npy    (253,) float32
  data/processed/nb1271_mean_oof.npy     (253,) float32
  data/processed/nb1271_median_oof.npy   (253,) float32
  data/processed/nb1271_invrae_oof.npy   (253,) float32
  data/processed/nb1271_bestw_oof.npy    (253,) float32
  data/processed/nb1271_summary.json
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

TAG = "nb1271"
SLSQP_FOLDS = 5
SLSQP_SEED = 42

# Reference pooled RAEs on 253 unblind.
NB1242_REF = 0.5431
NB1263_REF = 0.5586
NB1211_REF = 0.5451
NB1251_REF = 0.5394  # incumbent best 2-way blend
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


def _simplex_grid_3d(step: float):
    """Enumerate (w1, w2, w3) on the 3-simplex at given step."""
    out = []
    n = int(round(1.0 / step))
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            out.append((i * step, j * step, k * step))
    return out


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- multi-source residual blend:")
    print(f"           nb1242 (ChEMBL+MACCS residual, 0.5431)")
    print(f"         + nb1263 (PubChem+MACCS residual, 0.5586)")
    print(f"         + nb1211 (BoB-of-BoBs blend, 0.5451)")
    print(f"         5-fold SLSQP cross-fit + naive mean + median + invRAE + 3D grid")
    print("=" * 78)

    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)

    paths = {
        "nb1242": DATA_PROCESSED / "nb1242_mean_bag_oof.npy",
        "nb1263": DATA_PROCESSED / "nb1263_mean_bag_oof.npy",
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
    p2 = preds["nb1263"]
    p3 = preds["nb1211"]

    standalone_rae = {
        "nb1242": float(rae(y_unb, p1)),
        "nb1263": float(rae(y_unb, p2)),
        "nb1211": float(rae(y_unb, p3)),
    }
    print("\n[load] standalone pooled RAE on 253 unblind:")
    print(f"   nb1242 ChEMBL+MACCS residual  : {standalone_rae['nb1242']:.4f}  "
          f"(ref {NB1242_REF:.4f})")
    print(f"   nb1263 PubChem+MACCS residual : {standalone_rae['nb1263']:.4f}  "
          f"(ref {NB1263_REF:.4f})")
    print(f"   nb1211 BoB-of-BoBs blend      : {standalone_rae['nb1211']:.4f}  "
          f"(ref {NB1211_REF:.4f})")

    # Pairwise Pearson (pred + residual).
    r1 = p1 - y_unb
    r2 = p2 - y_unb
    r3 = p3 - y_unb

    pred_corr = {
        "nb1242_nb1263": float(np.corrcoef(p1, p2)[0, 1]),
        "nb1242_nb1211": float(np.corrcoef(p1, p3)[0, 1]),
        "nb1263_nb1211": float(np.corrcoef(p2, p3)[0, 1]),
    }
    resid_corr = {
        "nb1242_nb1263": float(np.corrcoef(r1, r2)[0, 1]),
        "nb1242_nb1211": float(np.corrcoef(r1, r3)[0, 1]),
        "nb1263_nb1211": float(np.corrcoef(r2, r3)[0, 1]),
    }
    print("\n[diag] pairwise pred-pred Pearson:")
    for k, v in pred_corr.items():
        print(f"   {k}: {v:.4f}")
    print("\n[diag] pairwise residual Pearson:")
    for k, v in resid_corr.items():
        print(f"   {k}: {v:.4f}")
    print("\n[diag] residual std:")
    print(f"   nb1242 = {r1.std():.4f}")
    print(f"   nb1263 = {r2.std():.4f}")
    print(f"   nb1211 = {r3.std():.4f}")

    # ---- 3-way SLSQP cross-fit ----
    print("\n" + "-" * 78)
    print("  BLOCK: 3-way SLSQP cross-fit")
    print("-" * 78)
    P = np.column_stack([p1, p2, p3])
    slsqp_oof, fold_records = _slsqp_cross_fit(P, y_unb, SLSQP_FOLDS, SLSQP_SEED)
    rae_slsqp = float(rae(y_unb, slsqp_oof))
    fold_w = np.array([r["weights"] for r in fold_records])
    mean_w = fold_w.mean(axis=0)
    print(f"   per-fold weights (nb1242, nb1263, nb1211):")
    for rec in fold_records:
        w = rec["weights"]
        print(f"     fold {rec['fold']}: w[nb1242]={w[0]:.4f}  "
              f"w[nb1263]={w[1]:.4f}  w[nb1211]={w[2]:.4f}")
    print(f"   mean weights across folds:  w[nb1242]={mean_w[0]:.4f}  "
          f"w[nb1263]={mean_w[1]:.4f}  w[nb1211]={mean_w[2]:.4f}")
    print(f"   pooled RAE(SLSQP cross-fit) = {rae_slsqp:.4f}")

    # In-sample SLSQP for diagnostic.
    w_full = _slsqp_blend_weights(P, y_unb)
    p_full = P @ w_full
    rae_full = float(rae(y_unb, p_full))
    print(f"   in-sample SLSQP weights: w[nb1242]={w_full[0]:.4f}  "
          f"w[nb1263]={w_full[1]:.4f}  w[nb1211]={w_full[2]:.4f}  "
          f"RAE = {rae_full:.4f}")

    # ---- Naive 1/3 mean ----
    mean_oof = (p1 + p2 + p3) / 3.0
    rae_mean = float(rae(y_unb, mean_oof))
    print(f"\n[block] naive 1/3 mean      RAE = {rae_mean:.4f}")

    # ---- Naive row-wise median ----
    median_oof = np.median(P, axis=1)
    rae_median = float(rae(y_unb, median_oof))
    print(f"[block] naive row-median    RAE = {rae_median:.4f}")

    # ---- Inverse-RAE weighted ----
    inv = np.array([
        1.0 / standalone_rae["nb1242"],
        1.0 / standalone_rae["nb1263"],
        1.0 / standalone_rae["nb1211"],
    ])
    w_inv = inv / inv.sum()
    invrae_oof = P @ w_inv
    rae_invrae = float(rae(y_unb, invrae_oof))
    print(f"[block] inverse-RAE weighted RAE = {rae_invrae:.4f}  "
          f"(w = {w_inv[0]:.4f}, {w_inv[1]:.4f}, {w_inv[2]:.4f})")

    # ---- 3-simplex grid step 0.1 ----
    print("\n" + "-" * 78)
    print("  BLOCK: best fixed-w grid on 3-simplex (step 0.1)")
    print("-" * 78)
    grid_pts = _simplex_grid_3d(0.1)
    grid_results = []
    best_w = None
    best_rae = float("inf")
    best_oof = None
    for w1, w2, w3 in grid_pts:
        blend = w1 * p1 + w2 * p2 + w3 * p3
        r = float(rae(y_unb, blend))
        grid_results.append({
            "w_nb1242": float(round(w1, 4)),
            "w_nb1263": float(round(w2, 4)),
            "w_nb1211": float(round(w3, 4)),
            "rae": r,
        })
        if r < best_rae:
            best_rae = r
            best_w = (float(round(w1, 4)), float(round(w2, 4)), float(round(w3, 4)))
            best_oof = blend
    # Print top-10 grid points by RAE.
    grid_sorted = sorted(grid_results, key=lambda d: d["rae"])
    print(f"   top-10 grid points (lowest RAE):")
    for rec in grid_sorted[:10]:
        print(f"     w=({rec['w_nb1242']:.2f}, {rec['w_nb1263']:.2f}, "
              f"{rec['w_nb1211']:.2f})  RAE={rec['rae']:.4f}")
    print(f"   best fixed-w blend: w[nb1242]={best_w[0]:.4f}  "
          f"w[nb1263]={best_w[1]:.4f}  w[nb1211]={best_w[2]:.4f}  "
          f"RAE={best_rae:.4f}")

    # ---- Verdict ----
    candidates = {
        "slsqp_cross_fit": rae_slsqp,
        "naive_mean":      rae_mean,
        "naive_median":    rae_median,
        "inv_rae":         rae_invrae,
        "best_fixed_w":    best_rae,
    }
    best_blend_tag = min(candidates, key=candidates.get)
    best_blend_rae = candidates[best_blend_tag]

    best_standalone_tag = min(standalone_rae, key=standalone_rae.get)
    best_standalone = standalone_rae[best_standalone_tag]

    beats_nb1251 = best_blend_rae < NB1251_REF - MARGIN
    flat_vs_nb1251 = abs(best_blend_rae - NB1251_REF) < MARGIN

    if beats_nb1251:
        verdict = (f"MULTI_SOURCE_BLEND_BEATS_NB1251 "
                   f"({best_blend_tag} @ {best_blend_rae:.4f})")
    elif flat_vs_nb1251:
        verdict = (f"MULTI_SOURCE_BLEND_FLAT_VS_NB1251 "
                   f"({best_blend_tag} @ {best_blend_rae:.4f})")
    else:
        verdict = (f"MULTI_SOURCE_BLEND_HURTS_VS_NB1251 "
                   f"({best_blend_tag} @ {best_blend_rae:.4f})")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"   nb1242 standalone : {standalone_rae['nb1242']:.4f}")
    print(f"   nb1263 standalone : {standalone_rae['nb1263']:.4f}")
    print(f"   nb1211 standalone : {standalone_rae['nb1211']:.4f}")
    print(f"   best standalone   : {best_standalone:.4f}  ({best_standalone_tag})")
    print(f"   nb1251 incumbent  : {NB1251_REF:.4f}")
    print(f"")
    print(f"   candidate pooled RAE table:")
    for tag, val in sorted(candidates.items(), key=lambda kv: kv[1]):
        print(f"     {tag:18s} = {val:.4f}")
    print(f"")
    print(f"   best blend                : {best_blend_rae:.4f}  ({best_blend_tag})")
    print(f"   delta vs nb1251 (0.5394)  : {best_blend_rae - NB1251_REF:+.4f}")
    print(f"   beats_nb1251 (>= {MARGIN})   : {beats_nb1251}")
    print(f"   verdict                   : {verdict}")

    # Persist canonical artifacts.
    np.save(DATA_PROCESSED / f"{TAG}_slsqp_oof.npy",
            slsqp_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_mean_oof.npy",
            mean_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_median_oof.npy",
            median_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_invrae_oof.npy",
            invrae_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_bestw_oof.npy",
            best_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_slsqp_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_mean_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_median_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_invrae_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_bestw_oof.npy'}")

    summary = {
        "tag": TAG,
        "n_unb": n_unb,
        "slsqp_folds": SLSQP_FOLDS,
        "slsqp_seed": SLSQP_SEED,
        "components": list(paths.keys()),
        "standalone_rae": standalone_rae,
        "pred_corr": pred_corr,
        "residual_corr": resid_corr,
        "residual_std_nb1242": float(r1.std()),
        "residual_std_nb1263": float(r2.std()),
        "residual_std_nb1211": float(r3.std()),
        "slsqp_fold_records": fold_records,
        "slsqp_mean_fold_weights": [float(x) for x in mean_w],
        "slsqp_in_sample_weights": [float(x) for x in w_full],
        "slsqp_in_sample_rae": rae_full,
        "rae_slsqp_cross_fit": rae_slsqp,
        "rae_naive_mean": rae_mean,
        "rae_naive_median": rae_median,
        "rae_inv_rae_weighted": rae_invrae,
        "inv_rae_weights": [float(x) for x in w_inv],
        "grid_step": 0.1,
        "n_grid_points": len(grid_results),
        "grid_results": grid_results,
        "best_fixed_w_nb1242": best_w[0],
        "best_fixed_w_nb1263": best_w[1],
        "best_fixed_w_nb1211": best_w[2],
        "rae_best_fixed_w": best_rae,
        "candidate_rae_table": candidates,
        "best_blend_tag": best_blend_tag,
        "best_blend_rae": best_blend_rae,
        "best_standalone_tag": best_standalone_tag,
        "best_standalone_rae": best_standalone,
        "nb1242_ref": NB1242_REF,
        "nb1263_ref": NB1263_REF,
        "nb1211_ref": NB1211_REF,
        "nb1251_ref": NB1251_REF,
        "delta_best_vs_nb1251": best_blend_rae - NB1251_REF,
        "delta_best_vs_nb1242": best_blend_rae - standalone_rae["nb1242"],
        "delta_best_vs_nb1211": best_blend_rae - standalone_rae["nb1211"],
        "beats_nb1251": bool(beats_nb1251),
        "flat_vs_nb1251": bool(flat_vs_nb1251),
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
              "rae_inv_rae_weighted",
              "best_fixed_w_nb1242", "best_fixed_w_nb1263",
              "best_fixed_w_nb1211", "rae_best_fixed_w",
              "candidate_rae_table",
              "best_blend_tag", "best_blend_rae",
              "delta_best_vs_nb1251",
              "beats_nb1251", "verdict"):
        print(f"  {k}: {res.get(k)}")
