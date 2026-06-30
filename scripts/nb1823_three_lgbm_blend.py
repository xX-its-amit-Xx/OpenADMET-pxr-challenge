"""
nb1823 — Three-way blend of nb1780 BoB + nb1811 mean + nb1812 mean.

Protocol:
1. Load three OOF arrays.
2. Pairwise Pearson correlation.
3. 3D simplex grid step 0.05.
4. SLSQP cross-fit (5-fold).
5. Verdict at 0.003 margin vs nb1780 baseline (0.5032).
"""

import json
from itertools import product
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import KFold

ROOT = Path("d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge")
PROC = ROOT / "data" / "processed"

BASELINE_RAE = 0.5032
MARGIN = 0.003


def rae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    num = np.mean(np.abs(y_true - y_pred))
    den = np.mean(np.abs(y_true - np.mean(y_true)))
    return float(num / den)


def main():
    # 1) Load OOFs
    p1 = np.load(PROC / "nb1780_bob_mean_oof.npy")
    p2 = np.load(PROC / "nb1811_mean_bag_oof.npy")
    p3 = np.load(PROC / "nb1812_mean_bag_oof.npy")

    # Truth: nb1780 was evaluated on 253 unblind; load truth from the same source
    # Try canonical truth file
    truth_candidates = [
        PROC / "postmortem" / "pm_unblind_y.npy",
        PROC / "unblind_truth_253.npy",
        PROC / "y_unblind_253.npy",
        PROC / "unblind253_truth.npy",
        PROC / "postmortem" / "unblind_truth_253.npy",
        PROC / "postmortem" / "y_true_253.npy",
    ]
    y_true = None
    for cand in truth_candidates:
        if cand.exists():
            y_true = np.load(cand)
            print(f"Loaded truth: {cand} shape={y_true.shape}")
            break
    if y_true is None:
        # Fall back: search any file containing 253 truth
        for p in PROC.glob("*truth*253*.npy"):
            y_true = np.load(p)
            print(f"Loaded truth (fallback): {p} shape={y_true.shape}")
            break
    if y_true is None:
        raise FileNotFoundError("Could not locate 253-unblind truth array")

    n = len(y_true)
    assert len(p1) == n == len(p2) == len(p3), f"len mismatch: {len(p1)},{len(p2)},{len(p3)},{n}"

    print(f"\nShapes: all n={n}")
    print(f"Individual RAE:")
    print(f"  nb1780 BoB     = {rae(y_true, p1):.4f}")
    print(f"  nb1811 mean    = {rae(y_true, p2):.4f}")
    print(f"  nb1812 mean    = {rae(y_true, p3):.4f}")

    # 2) Pairwise Pearson
    print("\nPairwise Pearson correlation:")
    corr = np.corrcoef(np.vstack([p1, p2, p3]))
    names = ["nb1780", "nb1811", "nb1812"]
    for i in range(3):
        for j in range(i + 1, 3):
            print(f"  r({names[i]}, {names[j]}) = {corr[i, j]:.4f}")

    # 3) 3D simplex grid step 0.05
    print("\n3D simplex grid (step=0.05):")
    grid_results = []
    step = 0.05
    grid = np.arange(0, 1 + 1e-9, step)
    for w1 in grid:
        for w2 in grid:
            w3 = 1.0 - w1 - w2
            if w3 < -1e-9 or w3 > 1 + 1e-9:
                continue
            w3 = max(0.0, w3)
            pred = w1 * p1 + w2 * p2 + w3 * p3
            r = rae(y_true, pred)
            grid_results.append((r, float(w1), float(w2), float(w3)))
    grid_results.sort(key=lambda x: x[0])

    print("Top-5 grid blends:")
    top5 = grid_results[:5]
    for k, (r, w1, w2, w3) in enumerate(top5, 1):
        print(f"  {k}. RAE={r:.4f}  w_nb1780={w1:.2f} w_nb1811={w2:.2f} w_nb1812={w3:.2f}")
    best_grid = top5[0]

    # 4) SLSQP cross-fit (5-fold)
    print("\nSLSQP 5-fold cross-fit:")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cf_pred = np.zeros(n)
    fold_weights = []
    for fold, (tr, va) in enumerate(kf.split(np.arange(n))):
        def obj(w):
            w = np.clip(w, 0, 1)
            w = w / max(w.sum(), 1e-9)
            pp = w[0] * p1[tr] + w[1] * p2[tr] + w[2] * p3[tr]
            return rae(y_true[tr], pp)

        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        bnds = [(0.0, 1.0)] * 3
        x0 = np.array([1 / 3, 1 / 3, 1 / 3])
        res = minimize(obj, x0, method="SLSQP", bounds=bnds, constraints=cons,
                       options={"maxiter": 200, "ftol": 1e-8})
        w = np.clip(res.x, 0, 1)
        w = w / max(w.sum(), 1e-9)
        cf_pred[va] = w[0] * p1[va] + w[1] * p2[va] + w[2] * p3[va]
        fold_weights.append(w.tolist())
        print(f"  fold {fold}: w={w.round(3).tolist()} train_RAE={res.fun:.4f}")

    slsqp_rae = rae(y_true, cf_pred)
    mean_w = np.mean(fold_weights, axis=0)
    print(f"\nSLSQP cross-fit RAE = {slsqp_rae:.4f}")
    print(f"Mean fold weights: nb1780={mean_w[0]:.3f} nb1811={mean_w[1]:.3f} nb1812={mean_w[2]:.3f}")

    # 5) Verdict
    best_rae = min(best_grid[0], slsqp_rae)
    delta = BASELINE_RAE - best_rae
    print(f"\n--- VERDICT ---")
    print(f"Baseline (nb1780)     : {BASELINE_RAE:.4f}")
    print(f"Best grid             : {best_grid[0]:.4f}  (w={best_grid[1]:.2f},{best_grid[2]:.2f},{best_grid[3]:.2f})")
    print(f"SLSQP cross-fit       : {slsqp_rae:.4f}")
    print(f"Best blend            : {best_rae:.4f}")
    print(f"Delta vs baseline     : {delta:+.4f}  (margin required: -{MARGIN})")
    if delta >= MARGIN:
        verdict = "PROMOTE"
    else:
        verdict = "REJECT"
    print(f"Verdict               : {verdict}")

    # Save summary
    summary = {
        "individual_rae": {
            "nb1780": rae(y_true, p1),
            "nb1811": rae(y_true, p2),
            "nb1812": rae(y_true, p3),
        },
        "pearson": {
            "nb1780_nb1811": float(corr[0, 1]),
            "nb1780_nb1812": float(corr[0, 2]),
            "nb1811_nb1812": float(corr[1, 2]),
        },
        "top5_grid": [
            {"rank": k + 1, "rae": r, "w_nb1780": w1, "w_nb1811": w2, "w_nb1812": w3}
            for k, (r, w1, w2, w3) in enumerate(top5)
        ],
        "slsqp_cross_fit_rae": slsqp_rae,
        "slsqp_mean_weights": {
            "nb1780": float(mean_w[0]),
            "nb1811": float(mean_w[1]),
            "nb1812": float(mean_w[2]),
        },
        "baseline_rae": BASELINE_RAE,
        "delta_vs_baseline": delta,
        "margin": MARGIN,
        "verdict": verdict,
    }
    out = PROC / "nb1823_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
