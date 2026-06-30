"""nb1601 — Ensemble of multiple CatBoost variants.

Members:
- nb1561 (CatBoost depth=4, Avalon K=30, OOF 0.5155) -- anchor
- nb1573 (CatBoost depth=3, K-tuned 5-way, alt-depth)
- nb1582 (CatBoost depth=4, Avalon K=20 upgrade)

PROTOCOL:
1. Load nb1561_bob_mean_oof.npy, nb1573_mean_bag_oof.npy, nb1582_bob_mean_oof.npy.
2. Compute pairwise Pearson.
3. 3D simplex grid step 0.05.
4. 5-fold cross-fit SLSQP.
5. Naive 1/3 mean.
6. Verdict at 0.003 margin vs nb1561 (0.5155).

Outputs:
- data/processed/nb1601_summary.json
- data/processed/nb1601_best_oof.npy
"""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr

ROOT = Path("d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge")
PROC = ROOT / "data" / "processed"


def rae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mean_y = float(np.mean(y_true))
    num = float(np.sum(np.abs(y_true - y_pred)))
    den = float(np.sum(np.abs(y_true - mean_y)))
    return num / den if den > 0 else float("inf")


def load_truth_253() -> np.ndarray:
    """Find the 253 unblind truth vector used by the nb15xx ladder."""
    candidates = [
        PROC / "_audit_unblind_y.npy",
        PROC / "unblind_253_truth.npy",
        PROC / "unblind_truth_253.npy",
        PROC / "truth_253.npy",
        PROC / "y_unblind_253.npy",
    ]
    for p in candidates:
        if p.exists():
            arr = np.load(p)
            if arr.shape[0] == 253:
                return arr.astype(float)
    for csv_name in [
        "postmortem/unblind_253.csv",
        "unblind_253.csv",
        "postmortem/analog_set1_unblind.csv",
    ]:
        p = PROC / csv_name
        if p.exists():
            df = pd.read_csv(p)
            for col in ["pEC50", "pec50", "y", "truth", "true_pec50"]:
                if col in df.columns and len(df) == 253:
                    return df[col].to_numpy(dtype=float)
    raise FileNotFoundError("Could not locate 253-row unblind truth file")


def slsqp_blend3(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
                 y: np.ndarray) -> tuple[tuple[float, float, float], float]:
    """Solve min RAE over w1*p1 + w2*p2 + w3*p3, simplex (w_i>=0, sum=1)."""

    def obj(w):
        w = np.asarray(w, dtype=float)
        s = float(w.sum())
        if s <= 0:
            return float("inf")
        w = w / s  # safe re-normalization (constraint also enforced)
        return rae(y, w[0] * p1 + w[1] * p2 + w[2] * p3)

    cons = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},)
    bounds = [(0.0, 1.0)] * 3

    best = None
    seeds = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1 / 3, 1 / 3, 1 / 3],
        [0.5, 0.25, 0.25],
        [0.25, 0.5, 0.25],
        [0.25, 0.25, 0.5],
    ]
    for s0 in seeds:
        res = minimize(obj, x0=s0, method="SLSQP", bounds=bounds,
                       constraints=cons,
                       options={"ftol": 1e-9, "maxiter": 300})
        if best is None or res.fun < best.fun:
            best = res
    w = np.asarray(best.x, dtype=float)
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s > 0:
        w = w / s
    return (float(w[0]), float(w[1]), float(w[2])), float(best.fun)


def crossfit_slsqp3(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
                    y: np.ndarray, n_splits: int = 5,
                    seed: int = 17) -> tuple[float, np.ndarray, list[tuple[float, float, float]]]:
    """5-fold cross-fit SLSQP: weights fit on 4 folds, applied to held-out fold."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    folds = np.array_split(idx, n_splits)
    blended = np.zeros_like(y, dtype=float)
    weights: list[tuple[float, float, float]] = []
    for k in range(n_splits):
        val = folds[k]
        trn = np.concatenate([folds[j] for j in range(n_splits) if j != k])
        w, _ = slsqp_blend3(p1[trn], p2[trn], p3[trn], y[trn])
        blended[val] = w[0] * p1[val] + w[1] * p2[val] + w[2] * p3[val]
        weights.append(w)
    return rae(y, blended), blended, weights


def simplex_grid(step: float = 0.05) -> list[tuple[float, float, float]]:
    """All (w1, w2, w3) on simplex w1+w2+w3=1, w_i in {0, step, 2*step, ..., 1}."""
    n_steps = int(round(1.0 / step))
    pts: list[tuple[float, float, float]] = []
    for i, j in product(range(n_steps + 1), repeat=2):
        k = n_steps - i - j
        if k < 0:
            continue
        w1 = i / n_steps
        w2 = j / n_steps
        w3 = k / n_steps
        pts.append((round(w1, 6), round(w2, 6), round(w3, 6)))
    return pts


def main() -> None:
    p1 = np.load(PROC / "nb1561_bob_mean_oof.npy").astype(float)  # depth=4
    p2 = np.load(PROC / "nb1573_mean_bag_oof.npy").astype(float)  # depth=3
    p3 = np.load(PROC / "nb1582_bob_mean_oof.npy").astype(float)  # Avalon K=20
    y = load_truth_253()

    assert p1.shape == p2.shape == p3.shape == y.shape, (
        p1.shape, p2.shape, p3.shape, y.shape,
    )
    n = len(y)

    rae_1561 = rae(y, p1)
    rae_1573 = rae(y, p2)
    rae_1582 = rae(y, p3)

    pear_12 = float(pearsonr(p1, p2).statistic)
    pear_13 = float(pearsonr(p1, p3).statistic)
    pear_23 = float(pearsonr(p2, p3).statistic)

    # Simplex grid (step 0.05)
    grid_pts = simplex_grid(step=0.05)
    grid_rows = []
    for w1, w2, w3 in grid_pts:
        pred = w1 * p1 + w2 * p2 + w3 * p3
        grid_rows.append({"w_nb1561": w1, "w_nb1573": w2, "w_nb1582": w3,
                          "rae": rae(y, pred)})
    grid_rows.sort(key=lambda r: r["rae"])
    top5 = grid_rows[:5]
    best_grid = grid_rows[0]

    # In-sample SLSQP
    w_is, rae_is = slsqp_blend3(p1, p2, p3, y)

    # 5-fold cross-fit SLSQP
    rae_cf, blend_cf, w_folds = crossfit_slsqp3(p1, p2, p3, y)

    # Naive 1/3 mean
    p_naive = (p1 + p2 + p3) / 3.0
    rae_naive = rae(y, p_naive)

    # Verdict @ 0.003 margin vs nb1561 (0.5155)
    margin = 0.003
    anchor_rae = 0.5155
    beats_anchor_cf = bool(rae_cf < (anchor_rae - margin))
    beats_anchor_grid = bool(best_grid["rae"] < (anchor_rae - margin))
    beats_anchor_naive = bool(rae_naive < (anchor_rae - margin))

    if rae_cf < anchor_rae - margin:
        verdict = (f"ADOPT — cross-fit 3-way ensemble RAE {rae_cf:.4f} beats "
                   f"nb1561 anchor {anchor_rae:.4f} by >=0.003")
    elif rae_cf < anchor_rae:
        verdict = (f"MARGINAL — cross-fit improves (RAE {rae_cf:.4f}) but under "
                   f"0.003 threshold vs nb1561 {anchor_rae:.4f}; HOLD nb1561")
    else:
        verdict = (f"REJECT — cross-fit RAE {rae_cf:.4f} does not beat nb1561 "
                   f"anchor {anchor_rae:.4f}")

    np.save(PROC / "nb1601_best_oof.npy", blend_cf)

    summary = {
        "tag": "nb1601",
        "members": {
            "nb1561": "CatBoost BoB depth=4 Avalon K=30 (anchor 0.5155)",
            "nb1573": "CatBoost depth=3 K-tuned mean-bag",
            "nb1582": "CatBoost BoB depth=4 Avalon K=20 upgrade",
        },
        "n": int(n),
        "rae_nb1561": rae_1561,
        "rae_nb1573": rae_1573,
        "rae_nb1582": rae_1582,
        "pearson": {
            "p1561_p1573": pear_12,
            "p1561_p1582": pear_13,
            "p1573_p1582": pear_23,
        },
        "grid_step": 0.05,
        "grid_n_points": len(grid_rows),
        "best_grid": best_grid,
        "top5_grid": top5,
        "slsqp_in_sample_weights": {
            "w_nb1561": w_is[0], "w_nb1573": w_is[1], "w_nb1582": w_is[2],
        },
        "slsqp_in_sample_rae": rae_is,
        "slsqp_crossfit_rae": rae_cf,
        "slsqp_crossfit_fold_weights": [
            {"w_nb1561": w[0], "w_nb1573": w[1], "w_nb1582": w[2]}
            for w in w_folds
        ],
        "slsqp_crossfit_mean_weights": {
            "w_nb1561": float(np.mean([w[0] for w in w_folds])),
            "w_nb1573": float(np.mean([w[1] for w in w_folds])),
            "w_nb1582": float(np.mean([w[2] for w in w_folds])),
        },
        "naive_third_mean_rae": rae_naive,
        "anchor_rae_nb1561": anchor_rae,
        "margin": margin,
        "beats_nb1561_crossfit": beats_anchor_cf,
        "beats_nb1561_grid": beats_anchor_grid,
        "beats_nb1561_naive": beats_anchor_naive,
        "verdict": verdict,
    }
    out_path = PROC / "nb1601_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))

    # Human-readable report
    print(f"n = {n}")
    print(f"\nMember RAEs (standalone vs 253 unblind):")
    print(f"  nb1561 depth=4         RAE = {rae_1561:.4f}")
    print(f"  nb1573 depth=3         RAE = {rae_1573:.4f}")
    print(f"  nb1582 Avalon K=20     RAE = {rae_1582:.4f}")
    print(f"\nPairwise Pearson:")
    print(f"  nb1561 <-> nb1573      = {pear_12:.4f}")
    print(f"  nb1561 <-> nb1582      = {pear_13:.4f}")
    print(f"  nb1573 <-> nb1582      = {pear_23:.4f}")

    print(f"\nSimplex grid (step 0.05, {len(grid_rows)} points) -- TOP 5:")
    print(f"{'w_1561':>8}  {'w_1573':>8}  {'w_1582':>8}  {'RAE':>10}")
    for row in top5:
        print(f"{row['w_nb1561']:>8.2f}  {row['w_nb1573']:>8.2f}  "
              f"{row['w_nb1582']:>8.2f}  {row['rae']:>10.4f}")
    print(f"\nBest grid: w=({best_grid['w_nb1561']:.2f}, "
          f"{best_grid['w_nb1573']:.2f}, {best_grid['w_nb1582']:.2f}) "
          f"-> RAE {best_grid['rae']:.4f}")

    print(f"\nSLSQP in-sample: w=({w_is[0]:.4f}, {w_is[1]:.4f}, {w_is[2]:.4f}) "
          f"-> RAE {rae_is:.4f}")
    print(f"SLSQP 5-fold cross-fit RAE: {rae_cf:.4f}")
    print(f"  mean fold weights: w_nb1561={np.mean([w[0] for w in w_folds]):.4f}, "
          f"w_nb1573={np.mean([w[1] for w in w_folds]):.4f}, "
          f"w_nb1582={np.mean([w[2] for w in w_folds]):.4f}")
    print(f"  per-fold:")
    for k, w in enumerate(w_folds):
        print(f"    fold {k}: ({w[0]:.3f}, {w[1]:.3f}, {w[2]:.3f})")

    print(f"\nNaive 1/3 mean RAE = {rae_naive:.4f}")

    print(f"\nAnchor nb1561 RAE = {anchor_rae:.4f}, margin = {margin}")
    print(f"  beats anchor (cross-fit) : {beats_anchor_cf}")
    print(f"  beats anchor (best grid) : {beats_anchor_grid}")
    print(f"  beats anchor (naive 1/3) : {beats_anchor_naive}")
    print(f"\nVerdict: {verdict}")
    print(f"\nSaved: {out_path}")
    print(f"Saved: {PROC / 'nb1601_best_oof.npy'}")


if __name__ == "__main__":
    main()
