"""nb1631 - 3-architecture blend: nb1561 (CatBoost) + nb1612 (LGBM 6-way) + nb1571 (CatBoost+LGBM blend).

Protocol:
1. Load three OOF files
2. Pairwise Pearson
3. 3D simplex grid step 0.05
4. SLSQP cross-fit
5. Verdict at 0.003 margin vs nb1561 (0.5155)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr

ROOT = Path("d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge")
PROC = ROOT / "data" / "processed"

NB1561_BASELINE = 0.5155
MARGIN = 0.003


def rae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    num = np.mean(np.abs(y_true - y_pred))
    den = np.mean(np.abs(y_true - np.mean(y_true)))
    return float(num / den)


def load_oof(path: Path) -> np.ndarray:
    arr = np.load(path)
    return np.asarray(arr, dtype=np.float64).ravel()


def main() -> None:
    # Load truth
    truth_path = PROC / "_audit_unblind_y.npy"
    if not truth_path.exists():
        for alt in [PROC / "postmortem" / "pm_unblind_y.npy"]:
            if alt.exists():
                truth_path = alt
                break
    y_true = np.load(truth_path).astype(np.float64).ravel()
    print(f"Truth loaded: {truth_path.name}, n={len(y_true)}")

    # Load three predictor OOFs
    p_nb1561 = load_oof(PROC / "nb1561_bob_mean_oof.npy")
    p_nb1612 = load_oof(PROC / "nb1612_best_oof.npy")
    p_nb1571 = load_oof(PROC / "nb1571_best_oof.npy")

    print(
        f"OOF shapes: nb1561={p_nb1561.shape}, nb1612={p_nb1612.shape}, nb1571={p_nb1571.shape}"
    )
    assert len(p_nb1561) == len(p_nb1612) == len(p_nb1571) == len(y_true)

    # Per-predictor standalone RAEs
    r_nb1561 = rae(y_true, p_nb1561)
    r_nb1612 = rae(y_true, p_nb1612)
    r_nb1571 = rae(y_true, p_nb1571)
    print(f"\nStandalone RAEs:")
    print(f"  nb1561 (CatBoost):       {r_nb1561:.4f}")
    print(f"  nb1612 (LGBM 6-way):     {r_nb1612:.4f}")
    print(f"  nb1571 (CatBoost+LGBM):  {r_nb1571:.4f}")

    # Pairwise Pearson
    r12, _ = pearsonr(p_nb1561, p_nb1612)
    r13, _ = pearsonr(p_nb1561, p_nb1571)
    r23, _ = pearsonr(p_nb1612, p_nb1571)
    print(f"\nPairwise Pearson:")
    print(f"  nb1561 vs nb1612: {r12:.4f}")
    print(f"  nb1561 vs nb1571: {r13:.4f}")
    print(f"  nb1612 vs nb1571: {r23:.4f}")

    # 3D simplex grid step 0.05
    step = 0.05
    weights = []
    vals = np.arange(0.0, 1.0 + 1e-9, step)
    for w1 in vals:
        for w2 in vals:
            w3 = 1.0 - w1 - w2
            if w3 < -1e-9 or w3 > 1.0 + 1e-9:
                continue
            w3 = max(0.0, min(1.0, w3))
            blend = w1 * p_nb1561 + w2 * p_nb1612 + w3 * p_nb1571
            r = rae(y_true, blend)
            weights.append((w1, w2, w3, r))

    weights.sort(key=lambda x: x[3])
    top5 = weights[:5]
    print(f"\nTop-5 grid (step={step}):")
    for w1, w2, w3, r in top5:
        print(f"  w_nb1561={w1:.2f}  w_nb1612={w2:.2f}  w_nb1571={w3:.2f}  RAE={r:.4f}")

    best_grid = top5[0]
    grid_best_rae = best_grid[3]
    grid_best_w = (best_grid[0], best_grid[1], best_grid[2])

    # SLSQP cross-fit (5-fold)
    n = len(y_true)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    folds = np.array_split(perm, 5)

    cf_preds = np.full(n, np.nan)
    slsqp_w_per_fold = []
    P_full = np.column_stack([p_nb1561, p_nb1612, p_nb1571])

    for fi, test_idx in enumerate(folds):
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        Ptr = P_full[train_mask]
        ytr = y_true[train_mask]

        def obj(w, P=Ptr, y=ytr):
            pred = P @ w
            return rae(y, pred)

        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0)] * 3
        x0 = np.array([1 / 3, 1 / 3, 1 / 3])

        res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 200, "ftol": 1e-8})
        w_fold = np.clip(res.x, 0.0, 1.0)
        w_fold = w_fold / w_fold.sum()
        slsqp_w_per_fold.append(w_fold.tolist())
        cf_preds[test_idx] = P_full[test_idx] @ w_fold

    slsqp_cf_rae = rae(y_true, cf_preds)

    # Also full-data SLSQP fit weights (for reporting + deploy)
    def obj_full(w, P=P_full, y=y_true):
        pred = P @ w
        return rae(y, pred)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * 3
    res_full = minimize(obj_full, np.array([1 / 3, 1 / 3, 1 / 3]),
                         method="SLSQP", bounds=bounds, constraints=cons,
                         options={"maxiter": 500, "ftol": 1e-10})
    w_full = np.clip(res_full.x, 0.0, 1.0)
    w_full = w_full / w_full.sum()
    slsqp_full_rae = rae(y_true, P_full @ w_full)

    print(f"\nSLSQP full-data weights: nb1561={w_full[0]:.4f} nb1612={w_full[1]:.4f} nb1571={w_full[2]:.4f}")
    print(f"SLSQP in-sample RAE: {slsqp_full_rae:.4f}")
    print(f"SLSQP 5-fold cross-fit RAE: {slsqp_cf_rae:.4f}")

    # Choose best OOF to save: cross-fit (LB-honest)
    best_oof = cf_preds.copy()
    best_rae = slsqp_cf_rae
    best_method = "SLSQP_cross_fit"
    if grid_best_rae < best_rae:
        # Use grid winner (in-sample, but report)
        # Honest LB-faithful is cross-fit; keep cross-fit as primary
        pass

    # Verdict
    beats_nb1561 = slsqp_cf_rae < (NB1561_BASELINE - MARGIN)
    verdict = "ACCEPT" if beats_nb1561 else "REJECT"

    print(f"\nnb1561 baseline: {NB1561_BASELINE:.4f}")
    print(f"Margin: {MARGIN:.4f}")
    print(f"SLSQP cross-fit RAE: {slsqp_cf_rae:.4f}")
    print(f"beats_nb1561: {beats_nb1561}")
    print(f"VERDICT: {verdict}")

    # Save artifacts
    np.save(PROC / "nb1631_best_oof.npy", best_oof)

    summary = {
        "method": "nb1631_three_arch_blend",
        "predictors": {
            "nb1561": {"file": "nb1561_bob_mean_oof.npy", "rae": r_nb1561},
            "nb1612": {"file": "nb1612_best_oof.npy", "rae": r_nb1612},
            "nb1571": {"file": "nb1571_best_oof.npy", "rae": r_nb1571},
        },
        "pairwise_pearson": {
            "nb1561_nb1612": r12,
            "nb1561_nb1571": r13,
            "nb1612_nb1571": r23,
        },
        "grid_step": step,
        "top5_grid": [
            {"w_nb1561": w1, "w_nb1612": w2, "w_nb1571": w3, "rae": r}
            for (w1, w2, w3, r) in top5
        ],
        "grid_best_weights": {
            "w_nb1561": grid_best_w[0],
            "w_nb1612": grid_best_w[1],
            "w_nb1571": grid_best_w[2],
        },
        "grid_best_rae": grid_best_rae,
        "slsqp_full_weights": {
            "w_nb1561": float(w_full[0]),
            "w_nb1612": float(w_full[1]),
            "w_nb1571": float(w_full[2]),
        },
        "slsqp_full_rae": float(slsqp_full_rae),
        "slsqp_cross_fit_rae": float(slsqp_cf_rae),
        "slsqp_weights_per_fold": slsqp_w_per_fold,
        "best_method_saved": best_method,
        "best_oof_rae": float(best_rae),
        "nb1561_baseline": NB1561_BASELINE,
        "margin": MARGIN,
        "beats_nb1561": bool(beats_nb1561),
        "verdict": verdict,
    }

    with open(PROC / "nb1631_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {PROC / 'nb1631_best_oof.npy'}")
    print(f"Saved: {PROC / 'nb1631_summary.json'}")


if __name__ == "__main__":
    main()
