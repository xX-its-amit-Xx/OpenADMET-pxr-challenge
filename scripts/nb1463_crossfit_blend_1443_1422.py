"""nb1463 -- Cross-fit weight blend nb1443 (75-bag median) + nb1422 (BoB median).

Protocol:
  1. Load nb1443_median_oof.npy (75-bag median, 0.4999 in-sample)
     and nb1422_bob_median_oof.npy (BoB median, 0.5016 in-sample).
  2. Pearson on preds and residuals.
  3. 5-fold cross-fit SLSQP (sum=1, w>=0) blend.
  4. Grid search w in {0.0..1.0 step 0.05} for p = w*nb1443 + (1-w)*nb1422.
  5. Verdict at 0.003 margin vs nb1422 bob_median (0.5016).

Outputs:
  scripts/nb1463_crossfit_blend_1443_1422.py    (this file)
  data/processed/nb1463_summary.json
  data/processed/nb1463_best_oof.npy             (253,) float32
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

TAG = "nb1463"

NB1443_OOF = DATA_PROCESSED / "nb1443_median_oof.npy"
NB1422_OOF = DATA_PROCESSED / "nb1422_bob_median_oof.npy"
Y_UNB = DATA_PROCESSED / "postmortem" / "pm_unblind_y.npy"

NB1443_MEDIAN_REF = 0.4999
NB1422_BOB_MEDIAN_REF = 0.5016
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

    a = np.load(NB1443_OOF).astype(np.float64)   # nb1443 75-bag median
    b = np.load(NB1422_OOF).astype(np.float64)   # nb1422 BoB median
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
        {"w_nb1443": float(w), "w_nb1422": float(1.0 - w), "rae": float(r)}
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
            "w_nb1443": float(w_tr[0]),
            "w_nb1422": float(w_tr[1]),
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
            "best_w_nb1443_on_train": best_w_tr,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
        })
    rae_grid_cf = float(rae(y, p_grid_cf))

    # 50/50 reproducibility check
    rae_5050 = float(rae(y, 0.5 * a + 0.5 * b))

    # --- 7. verdicts (vs nb1422 bob_median 0.5016) ---
    beats_nb1422 = (rae_slsqp_cf + DECISION_MARGIN) < NB1422_BOB_MEDIAN_REF
    grid_beats_nb1422 = (best_rae_grid + DECISION_MARGIN) < NB1422_BOB_MEDIAN_REF
    beats_nb1443 = (rae_slsqp_cf + DECISION_MARGIN) < NB1443_MEDIAN_REF
    grid_beats_nb1443 = (best_rae_grid + DECISION_MARGIN) < NB1443_MEDIAN_REF

    if grid_beats_nb1422:
        verdict = "BLEND_BEATS_NB1422_BOB_MEDIAN"
    elif (rae_slsqp_cf - NB1422_BOB_MEDIAN_REF) < DECISION_MARGIN:
        verdict = "BLEND_FLAT_VS_NB1422_BOB_MEDIAN"
    else:
        verdict = "BLEND_LOSES_TO_NB1422_BOB_MEDIAN"

    # save best_oof (grid pick) for downstream
    np.save(DATA_PROCESSED / f"{TAG}_best_oof.npy", best_oof.astype(np.float32))

    summary = {
        "tag": TAG,
        "n_unb": int(len(y)),
        "nb1443_source": str(NB1443_OOF.name),
        "nb1422_source": str(NB1422_OOF.name),
        "pearson_pred_nb1443_nb1422": pearson_pred,
        "pearson_resid_nb1443_nb1422": pearson_resid,
        "rae_nb1443_standalone": rae_a,
        "rae_nb1422_standalone": rae_b,
        "rae_5050_blend_insample": rae_5050,
        "grid_step": 0.05,
        "grid_table": grid_table,
        "best_w_grid_nb1443": best_w_grid,
        "best_rae_grid_insample": best_rae_grid,
        "slsqp_insample_w_nb1443": float(w_full[0]),
        "slsqp_insample_w_nb1422": float(w_full[1]),
        "rae_slsqp_insample": rae_slsqp_insample,
        "cv_folds": CV_FOLDS,
        "cv_seed": CV_SEED,
        "slsqp_per_fold": w_per_fold,
        "rae_slsqp_crossfit": rae_slsqp_cf,
        "grid_per_fold": w_grid_per_fold,
        "rae_grid_crossfit": rae_grid_cf,
        "nb1443_median_ref": NB1443_MEDIAN_REF,
        "nb1422_bob_median_ref": NB1422_BOB_MEDIAN_REF,
        "decision_margin": DECISION_MARGIN,
        "beats_nb1422": bool(beats_nb1422),
        "grid_beats_nb1422": bool(grid_beats_nb1422),
        "beats_nb1443": bool(beats_nb1443),
        "grid_beats_nb1443": bool(grid_beats_nb1443),
        "delta_slsqp_cf_vs_nb1422": rae_slsqp_cf - NB1422_BOB_MEDIAN_REF,
        "delta_grid_vs_nb1422": best_rae_grid - NB1422_BOB_MEDIAN_REF,
        "delta_slsqp_cf_vs_nb1443": rae_slsqp_cf - NB1443_MEDIAN_REF,
        "delta_grid_vs_nb1443": best_rae_grid - NB1443_MEDIAN_REF,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }

    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    # --- console report ---
    print(f"[{TAG}] n_unb={len(y)}")
    print(f"  Pearson(pred)  nb1443 vs nb1422 = {pearson_pred:.6f}")
    print(f"  Pearson(resid) nb1443 vs nb1422 = {pearson_resid:.6f}")
    print(f"  RAE nb1443 standalone = {rae_a:.4f}")
    print(f"  RAE nb1422 standalone = {rae_b:.4f}")
    print(f"  RAE 50/50 blend (in-sample) = {rae_5050:.4f}")
    print()
    print("  Grid (in-sample) w_nb1443 -> RAE:")
    for row in grid_table:
        marker = "  *" if abs(row["w_nb1443"] - best_w_grid) < 1e-9 else "   "
        print(f"    {marker} w={row['w_nb1443']:.2f}  RAE={row['rae']:.4f}")
    print(f"  Best-w grid (in-sample): w_nb1443={best_w_grid:.2f}, RAE={best_rae_grid:.4f}")
    print()
    print(f"  SLSQP in-sample: w=[{w_full[0]:.3f}, {w_full[1]:.3f}], RAE={rae_slsqp_insample:.4f}")
    print(f"  SLSQP 5-fold cross-fit: RAE={rae_slsqp_cf:.4f}")
    print(f"  Grid  5-fold cross-fit: RAE={rae_grid_cf:.4f}")
    for r in w_per_fold:
        print(f"    fold {r['fold']}: w_nb1443={r['w_nb1443']:.3f}, w_nb1422={r['w_nb1422']:.3f}")
    print()
    print(f"  delta slsqp_cf vs nb1422_bob_median = {rae_slsqp_cf - NB1422_BOB_MEDIAN_REF:+.4f}")
    print(f"  delta grid (best) vs nb1422_bob_median = {best_rae_grid - NB1422_BOB_MEDIAN_REF:+.4f}")
    print(f"  delta slsqp_cf vs nb1443_median = {rae_slsqp_cf - NB1443_MEDIAN_REF:+.4f}")
    print(f"  delta grid (best) vs nb1443_median = {best_rae_grid - NB1443_MEDIAN_REF:+.4f}")
    print(f"  beats_nb1422_bob_median (slsqp_cf + margin) = {beats_nb1422}")
    print(f"  grid_beats_nb1422_bob_median           = {grid_beats_nb1422}")
    print(f"  verdict: {verdict}")
    print(f"  wall_sec={summary['wall_sec']}")
    print(f"  wrote {out_path}")
    print(f"  wrote {DATA_PROCESSED / (TAG + '_best_oof.npy')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
