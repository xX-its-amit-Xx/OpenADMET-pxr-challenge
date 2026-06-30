"""nb1262 — Triple blend: nb1242 (ChEMBL-feat) + nb1252 (BoB ChEMBL) + nb1211 (BoB blend).

Hypothesis: 3-way may capture variance across (single-bag ChEMBL feature,
outer-bag ChEMBL feature, internal BoB-of-BoBs). nb1242 vs nb1252 differ only
in outer-bagging dimension — may add diversification.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path("d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge")
PROC = ROOT / "data" / "processed"


def rae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    num = np.sum(np.abs(y_true - y_pred))
    den = np.sum(np.abs(y_true - np.mean(y_true)))
    return float(num / den)


def load_truth_253() -> np.ndarray:
    pm = PROC / "postmortem"
    candidates = [
        pm / "pm_unblind_y.npy",
        pm / "unblind_253_truth.npy",
        pm / "y_unblind_253.npy",
        pm / "unblind_truth.npy",
    ]
    for c in candidates:
        if c.exists():
            arr = np.load(c)
            if arr.shape == (253,):
                return arr.astype(float)
    # fallback: parse from unblind CSV
    for csv in pm.glob("*unblind*.csv"):
        df = pd.read_csv(csv)
        for col in ["pEC50_truth", "pEC50", "truth", "y", "label"]:
            if col in df.columns and len(df) == 253:
                return df[col].to_numpy(dtype=float)
    raise FileNotFoundError("Could not locate 253 unblind truth array.")


def slsqp_blend(preds_list, y, n_splits=5, seed=0):
    """5-fold cross-fit SLSQP convex weights."""
    n = len(y)
    K = len(preds_list)
    P = np.column_stack(preds_list)  # (n,K)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, n_splits)
    oof = np.zeros(n)
    fold_weights = []
    for f in range(n_splits):
        te = folds[f]
        tr = np.concatenate([folds[j] for j in range(n_splits) if j != f])
        Ptr, ytr = P[tr], y[tr]

        def loss(w):
            return float(np.sum(np.abs(ytr - Ptr @ w)) / np.sum(np.abs(ytr - ytr.mean())))

        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        bounds = [(0.0, 1.0)] * K
        x0 = np.full(K, 1.0 / K)
        res = minimize(loss, x0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 200, "ftol": 1e-9})
        w = res.x
        w = np.clip(w, 0, 1)
        w = w / w.sum()
        oof[te] = P[te] @ w
        fold_weights.append(w)
    avg_w = np.mean(fold_weights, axis=0)
    avg_w = avg_w / avg_w.sum()
    return oof, avg_w, fold_weights


def best_grid_weights(preds_list, y, step=0.1):
    P = np.column_stack(preds_list)
    K = len(preds_list)
    assert K == 3
    grid = np.arange(0.0, 1.0 + 1e-9, step)
    best = (np.inf, None, None)
    for w1 in grid:
        for w2 in grid:
            w3 = 1.0 - w1 - w2
            if w3 < -1e-9 or w3 > 1.0 + 1e-9:
                continue
            w = np.array([w1, w2, max(0.0, w3)])
            w = w / w.sum()
            pred = P @ w
            r = rae(y, pred)
            if r < best[0]:
                best = (r, w, pred)
    return best  # (rae, w, pred)


def main():
    p1 = np.load(PROC / "nb1242_mean_bag_oof.npy").astype(float)   # 0.5431
    p2 = np.load(PROC / "nb1252_bob_mean_oof.npy").astype(float)   # 0.5446
    p3 = np.load(PROC / "nb1211_mean_oof.npy").astype(float)       # 0.5451
    assert p1.shape == p2.shape == p3.shape == (253,), f"shapes {p1.shape} {p2.shape} {p3.shape}"

    y = load_truth_253()
    assert y.shape == (253,)

    r1, r2, r3 = rae(y, p1), rae(y, p2), rae(y, p3)

    # pairwise pred & residual correlations
    res1, res2, res3 = y - p1, y - p2, y - p3
    pred_corr = {
        "12_pred": float(np.corrcoef(p1, p2)[0, 1]),
        "13_pred": float(np.corrcoef(p1, p3)[0, 1]),
        "23_pred": float(np.corrcoef(p2, p3)[0, 1]),
    }
    resid_corr = {
        "12_resid": float(np.corrcoef(res1, res2)[0, 1]),
        "13_resid": float(np.corrcoef(res1, res3)[0, 1]),
        "23_resid": float(np.corrcoef(res2, res3)[0, 1]),
    }

    # naive mean / median
    mean_oof = (p1 + p2 + p3) / 3.0
    median_oof = np.median(np.column_stack([p1, p2, p3]), axis=1)
    r_mean = rae(y, mean_oof)
    r_median = rae(y, median_oof)

    # inverse-RAE weighted mean (using standalone RAEs)
    invs = np.array([1.0 / r1, 1.0 / r2, 1.0 / r3])
    w_inv = invs / invs.sum()
    inv_oof = w_inv[0] * p1 + w_inv[1] * p2 + w_inv[2] * p3
    r_inv = rae(y, inv_oof)

    # SLSQP cross-fit
    slsqp_oof, slsqp_w, fold_ws = slsqp_blend([p1, p2, p3], y, n_splits=5, seed=0)
    r_slsqp = rae(y, slsqp_oof)

    # best fixed-weight grid (in-sample)
    r_bestw, w_bestw, bestw_oof = best_grid_weights([p1, p2, p3], y, step=0.1)

    # save oof arrays
    np.save(PROC / "nb1262_slsqp_oof.npy", slsqp_oof)
    np.save(PROC / "nb1262_mean_oof.npy", mean_oof)
    np.save(PROC / "nb1262_median_oof.npy", median_oof)
    np.save(PROC / "nb1262_bestw_oof.npy", bestw_oof)

    nb1251_rae = 0.5394
    margin = 0.003
    best_overall = min(r_slsqp, r_mean, r_median, r_inv, r_bestw)
    beats_nb1251 = bool(best_overall < (nb1251_rae - margin))

    summary = {
        "method": "nb1262_triple_chembl_bob",
        "components": {
            "nb1242_mean_bag": {"file": "nb1242_mean_bag_oof.npy", "rae_standalone": r1},
            "nb1252_bob_mean": {"file": "nb1252_bob_mean_oof.npy", "rae_standalone": r2},
            "nb1211_mean":     {"file": "nb1211_mean_oof.npy",     "rae_standalone": r3},
        },
        "pred_pearson": pred_corr,
        "resid_pearson": resid_corr,
        "naive_mean_rae":   r_mean,
        "naive_median_rae": r_median,
        "inv_rae_weighted": {"weights": w_inv.tolist(), "rae": r_inv},
        "slsqp_crossfit": {
            "weights_avg": slsqp_w.tolist(),
            "rae": r_slsqp,
            "fold_weights": [w.tolist() for w in fold_ws],
        },
        "best_grid_w": {"weights": w_bestw.tolist(), "rae": r_bestw, "step": 0.1, "note": "in-sample"},
        "verdict": {
            "nb1251_baseline": nb1251_rae,
            "margin": margin,
            "best_blend_rae": best_overall,
            "beats_nb1251": beats_nb1251,
        },
    }
    out = PROC / "nb1262_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
