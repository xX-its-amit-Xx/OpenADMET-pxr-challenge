"""nb1520 -- Blend nb1512 BoB (CatBoost) + nb1500 BoB (LGBM).

Both anchors are outer-bag-validated (BoB-mean OOF on 253 unblind):
  nb1512 (CatBoost, MAE loss, depth=4) -> 0.5246
  nb1500 (LightGBM, Huber, depth=3)    -> 0.5236

Protocol:
  1. Load nb1512_bob_mean_oof.npy and nb1500_bob_mean_oof.npy.
  2. Pearson on preds and residuals.
  3. Grid search w in {0.0..1.0 step 0.05} for p = w*nb1512 + (1-w)*nb1500.
  4. 5-fold cross-fit SLSQP (sum=1, w>=0).
  5. Verdict at 0.003 margin vs nb1500 BoB mean (0.5236).

Outputs:
  scripts/nb1520_blend_bob_bob.py       (this file)
  data/processed/nb1520_summary.json
  data/processed/nb1520_best_oof.npy    (253,) float32
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

TAG = "nb1520"

NB1512_OOF = DATA_PROCESSED / "nb1512_bob_mean_oof.npy"
NB1500_OOF = DATA_PROCESSED / "nb1500_bob_mean_oof.npy"
Y_UNB = DATA_PROCESSED / "postmortem" / "pm_unblind_y.npy"

NB1512_BOB_MEAN_REF = 0.5246
NB1500_BOB_MEAN_REF = 0.5236
DECISION_MARGIN = 0.003

CV_FOLDS = 5
CV_SEED = 42


def slsqp_blend(stack: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve sum(w)=1, w>=0, min RAE(stack @ w, y). Returns weights."""
    K = stack.shape[1]
    w0 = np.full(K, 1.0 / K, dtype=np.float64)

    def loss(w):
        return rae(y, stack @ w)

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bounds = [(0.0, 1.0)] * K
    res = minimize(loss, w0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-9})
    return np.asarray(res.x, dtype=np.float64)


def main() -> int:
    t0 = time.time()

    a = np.load(NB1512_OOF).astype(np.float64)   # nb1512 BoB mean (CatBoost)
    b = np.load(NB1500_OOF).astype(np.float64)   # nb1500 BoB mean (LGBM)
    y = np.load(Y_UNB).astype(np.float64)
    assert a.shape == b.shape == y.shape == (253,), (a.shape, b.shape, y.shape)

    # --- 1. pairwise Pearson on preds and residuals ---
    pearson_pred = float(np.corrcoef(a, b)[0, 1])
    ra = a - y
    rb = b - y
    pearson_resid = float(np.corrcoef(ra, rb)[0, 1])

    # --- 2. standalone RAE check ---
    rae_a = float(rae(y, a))
    rae_b = float(rae(y, b))

    # --- 3. grid search (in-sample) ---
    grid_w = np.arange(0.0, 1.0 + 1e-9, 0.05)
    grid_rae = []
    for w in grid_w:
        p = w * a + (1.0 - w) * b
        grid_rae.append(float(rae(y, p)))
    grid_rae = np.asarray(grid_rae, dtype=np.float64)
    best_idx = int(np.argmin(grid_rae))
    best_w_grid = float(grid_w[best_idx])
    best_rae_grid = float(grid_rae[best_idx])
    best_oof = best_w_grid * a + (1.0 - best_w_grid) * b

    grid_table = [
        {"w_nb1512": float(w), "w_nb1500": float(1.0 - w), "rae": float(r)}
        for w, r in zip(grid_w, grid_rae)
    ]

    # --- 4. SLSQP in-sample (sanity) ---
    stack = np.column_stack([a, b])
    w_full = slsqp_blend(stack, y)
    p_full = stack @ w_full
    rae_slsqp_insample = float(rae(y, p_full))

    # --- 5. 5-fold cross-fit SLSQP ---
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_SEED)
    p_cf = np.zeros_like(y)
    w_per_fold = []
    for fold_idx, (tr, te) in enumerate(kf.split(np.arange(len(y)))):
        w_tr = slsqp_blend(stack[tr], y[tr])
        p_cf[te] = stack[te] @ w_tr
        w_per_fold.append({
            "fold": fold_idx,
            "w_nb1512": float(w_tr[0]),
            "w_nb1500": float(w_tr[1]),
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
        })
    rae_slsqp_cf = float(rae(y, p_cf))

    # --- 6. cross-fit grid: best w on train, apply to held-out ---
    p_grid_cf = np.zeros_like(y)
    w_grid_per_fold = []
    for fold_idx, (tr, te) in enumerate(kf.split(np.arange(len(y)))):
        best_w_tr = 0.5
        best_r_tr = np.inf
        for w in grid_w:
            r = float(rae(y[tr], w * a[tr] + (1.0 - w) * b[tr]))
            if r < best_r_tr:
                best_r_tr = r
                best_w_tr = float(w)
        p_grid_cf[te] = best_w_tr * a[te] + (1.0 - best_w_tr) * b[te]
        w_grid_per_fold.append({
            "fold": fold_idx,
            "best_w_nb1512_on_train": best_w_tr,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
        })
    rae_grid_cf = float(rae(y, p_grid_cf))

    # 50/50 reproducibility check
    rae_5050 = float(rae(y, 0.5 * a + 0.5 * b))

    # --- 7. verdicts (vs nb1500 bob_mean 0.5236) ---
    beats_nb1500 = (rae_slsqp_cf + DECISION_MARGIN) < NB1500_BOB_MEAN_REF
    grid_beats_nb1500 = (best_rae_grid + DECISION_MARGIN) < NB1500_BOB_MEAN_REF
    beats_nb1512 = (rae_slsqp_cf + DECISION_MARGIN) < NB1512_BOB_MEAN_REF
    grid_beats_nb1512 = (best_rae_grid + DECISION_MARGIN) < NB1512_BOB_MEAN_REF

    if grid_beats_nb1500:
        verdict = "BLEND_BEATS_NB1500_BOB_MEAN"
    elif (rae_slsqp_cf - NB1500_BOB_MEAN_REF) < DECISION_MARGIN:
        verdict = "BLEND_FLAT_VS_NB1500_BOB_MEAN"
    else:
        verdict = "BLEND_LOSES_TO_NB1500_BOB_MEAN"

    # save best_oof (grid pick) for downstream
    np.save(DATA_PROCESSED / f"{TAG}_best_oof.npy", best_oof.astype(np.float32))

    summary = {
        "tag": TAG,
        "n_unb": int(len(y)),
        "nb1512_source": str(NB1512_OOF.name),
        "nb1500_source": str(NB1500_OOF.name),
        "pearson_pred_nb1512_nb1500": pearson_pred,
        "pearson_resid_nb1512_nb1500": pearson_resid,
        "rae_nb1512_standalone": rae_a,
        "rae_nb1500_standalone": rae_b,
        "rae_5050_blend_insample": rae_5050,
        "grid_step": 0.05,
        "grid_table": grid_table,
        "best_w_grid_nb1512": best_w_grid,
        "best_rae_grid_insample": best_rae_grid,
        "slsqp_insample_w_nb1512": float(w_full[0]),
        "slsqp_insample_w_nb1500": float(w_full[1]),
        "rae_slsqp_insample": rae_slsqp_insample,
        "cv_folds": CV_FOLDS,
        "cv_seed": CV_SEED,
        "slsqp_per_fold": w_per_fold,
        "rae_slsqp_crossfit": rae_slsqp_cf,
        "grid_per_fold": w_grid_per_fold,
        "rae_grid_crossfit": rae_grid_cf,
        "nb1512_bob_mean_ref": NB1512_BOB_MEAN_REF,
        "nb1500_bob_mean_ref": NB1500_BOB_MEAN_REF,
        "decision_margin": DECISION_MARGIN,
        "beats_nb1500": bool(beats_nb1500),
        "grid_beats_nb1500": bool(grid_beats_nb1500),
        "beats_nb1512": bool(beats_nb1512),
        "grid_beats_nb1512": bool(grid_beats_nb1512),
        "delta_slsqp_cf_vs_nb1500": rae_slsqp_cf - NB1500_BOB_MEAN_REF,
        "delta_grid_vs_nb1500": best_rae_grid - NB1500_BOB_MEAN_REF,
        "delta_slsqp_cf_vs_nb1512": rae_slsqp_cf - NB1512_BOB_MEAN_REF,
        "delta_grid_vs_nb1512": best_rae_grid - NB1512_BOB_MEAN_REF,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }

    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    # --- console report ---
    print(f"[{TAG}] n_unb={len(y)}")
    print(f"  Pearson(pred)  nb1512 vs nb1500 = {pearson_pred:.6f}")
    print(f"  Pearson(resid) nb1512 vs nb1500 = {pearson_resid:.6f}")
    print(f"  RAE nb1512 standalone (BoB mean) = {rae_a:.4f}")
    print(f"  RAE nb1500 standalone (BoB mean) = {rae_b:.4f}")
    print(f"  RAE 50/50 blend (in-sample)      = {rae_5050:.4f}")
    print()
    print("  Grid (in-sample) w_nb1512 -> RAE:")
    for row in grid_table:
        marker = "  *" if abs(row["w_nb1512"] - best_w_grid) < 1e-9 else "   "
        print(f"    {marker} w={row['w_nb1512']:.2f}  RAE={row['rae']:.4f}")
    print(f"  Best-w grid (in-sample): w_nb1512={best_w_grid:.2f}, RAE={best_rae_grid:.4f}")
    print()
    print(f"  SLSQP in-sample: w=[{w_full[0]:.3f}, {w_full[1]:.3f}], RAE={rae_slsqp_insample:.4f}")
    print(f"  SLSQP 5-fold cross-fit: RAE={rae_slsqp_cf:.4f}")
    print(f"  Grid  5-fold cross-fit: RAE={rae_grid_cf:.4f}")
    for r in w_per_fold:
        print(f"    fold {r['fold']}: w_nb1512={r['w_nb1512']:.3f}, w_nb1500={r['w_nb1500']:.3f}")
    print()
    print(f"  delta slsqp_cf vs nb1500_bob_mean = {rae_slsqp_cf - NB1500_BOB_MEAN_REF:+.4f}")
    print(f"  delta grid (best) vs nb1500_bob_mean = {best_rae_grid - NB1500_BOB_MEAN_REF:+.4f}")
    print(f"  delta slsqp_cf vs nb1512_bob_mean = {rae_slsqp_cf - NB1512_BOB_MEAN_REF:+.4f}")
    print(f"  delta grid (best) vs nb1512_bob_mean = {best_rae_grid - NB1512_BOB_MEAN_REF:+.4f}")
    print(f"  beats_nb1500_bob_mean (slsqp_cf + margin) = {beats_nb1500}")
    print(f"  grid_beats_nb1500_bob_mean              = {grid_beats_nb1500}")
    print(f"  verdict: {verdict}")
    print(f"  wall_sec={summary['wall_sec']}")
    print(f"  wrote {out_path}")
    print(f"  wrote {DATA_PROCESSED / (TAG + '_best_oof.npy')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
