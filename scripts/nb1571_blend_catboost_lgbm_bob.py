"""nb1571 — Blend nb1561 CatBoost BoB + nb1560 LGBM BoB.

Protocol:
1. Load nb1561 (CatBoost BoB, OOF 0.5155) + nb1560 (LGBM BoB, OOF 0.5211).
2. Pearson correlation.
3. Grid w in {0.0..1.0 step 0.05} where pred = w * nb1561 + (1 - w) * nb1560.
4. 5-fold cross-fit SLSQP for honest blend weight.
5. Verdict at 0.003 margin vs nb1561 (0.5155).

Outputs:
- data/processed/nb1571_summary.json
- data/processed/nb1571_best_oof.npy
"""
from __future__ import annotations

import json
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
    # Try the standard unblind truth file
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
    # Fall back to CSV from postmortem
    for csv_name in ["postmortem/unblind_253.csv", "unblind_253.csv", "postmortem/analog_set1_unblind.csv"]:
        p = PROC / csv_name
        if p.exists():
            df = pd.read_csv(p)
            for col in ["pEC50", "pec50", "y", "truth", "true_pec50"]:
                if col in df.columns and len(df) == 253:
                    return df[col].to_numpy(dtype=float)
    raise FileNotFoundError("Could not locate 253-row unblind truth file")


def slsqp_blend(p1: np.ndarray, p2: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Solve min RAE over w*p1 + (1-w)*p2, 0<=w<=1."""

    def obj(w):
        ww = float(w[0])
        return rae(y, ww * p1 + (1.0 - ww) * p2)

    best = None
    for w0 in np.linspace(0.0, 1.0, 11):
        res = minimize(obj, x0=[w0], method="SLSQP", bounds=[(0.0, 1.0)],
                       options={"ftol": 1e-9, "maxiter": 200})
        if best is None or res.fun < best.fun:
            best = res
    return float(best.x[0]), float(best.fun)


def crossfit_slsqp(p1: np.ndarray, p2: np.ndarray, y: np.ndarray, n_splits: int = 5,
                   seed: int = 17) -> tuple[float, np.ndarray, list[float]]:
    """5-fold cross-fit SLSQP: weight fit on 4 folds, applied to held-out fold."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    folds = np.array_split(idx, n_splits)
    blended = np.zeros_like(y, dtype=float)
    weights = []
    for k in range(n_splits):
        val = folds[k]
        trn = np.concatenate([folds[j] for j in range(n_splits) if j != k])
        w, _ = slsqp_blend(p1[trn], p2[trn], y[trn])
        blended[val] = w * p1[val] + (1.0 - w) * p2[val]
        weights.append(w)
    return rae(y, blended), blended, weights


def main() -> None:
    p_cat = np.load(PROC / "nb1561_bob_mean_oof.npy").astype(float)
    p_lgb = np.load(PROC / "nb1560_bob_mean_oof.npy").astype(float)
    y = load_truth_253()

    assert p_cat.shape == p_lgb.shape == y.shape, (p_cat.shape, p_lgb.shape, y.shape)
    n = len(y)

    rae_cat = rae(y, p_cat)
    rae_lgb = rae(y, p_lgb)
    pear = float(pearsonr(p_cat, p_lgb).statistic)

    # Grid sweep w = weight on nb1561 (CatBoost)
    grid_rows = []
    best_grid = None
    for w in np.round(np.arange(0.0, 1.0001, 0.05), 4):
        pred = w * p_cat + (1.0 - w) * p_lgb
        r = rae(y, pred)
        grid_rows.append({"w_nb1561": float(w), "rae": r})
        if best_grid is None or r < best_grid["rae"]:
            best_grid = {"w_nb1561": float(w), "rae": r}

    # In-sample SLSQP
    w_is, rae_is = slsqp_blend(p_cat, p_lgb, y)

    # Honest 5-fold cross-fit SLSQP
    rae_cf, blend_cf, w_folds = crossfit_slsqp(p_cat, p_lgb, y)

    # Best-of (cross-fit is the honest number for verdict)
    margin = 0.003
    beats_nb1561 = bool(rae_cf < (rae_cat - margin))
    if rae_cf < rae_cat - margin:
        verdict = "ADOPT — cross-fit blend beats nb1561 by >=0.003"
    elif rae_cf < rae_cat:
        verdict = "MARGINAL — improves but under 0.003 threshold; HOLD nb1561"
    else:
        verdict = "REJECT — blend does not beat nb1561 standalone"

    np.save(PROC / "nb1571_best_oof.npy", blend_cf)

    summary = {
        "n": int(n),
        "rae_nb1561_catboost": rae_cat,
        "rae_nb1560_lgbm": rae_lgb,
        "pearson_p1561_p1560": pear,
        "grid": grid_rows,
        "best_grid_w_nb1561": best_grid["w_nb1561"],
        "best_grid_rae": best_grid["rae"],
        "slsqp_in_sample_w_nb1561": w_is,
        "slsqp_in_sample_rae": rae_is,
        "slsqp_crossfit_rae": rae_cf,
        "slsqp_crossfit_fold_w_nb1561": [float(w) for w in w_folds],
        "slsqp_crossfit_mean_w_nb1561": float(np.mean(w_folds)),
        "margin_vs_nb1561": margin,
        "rae_nb1561_minus_crossfit": rae_cat - rae_cf,
        "beats_nb1561": beats_nb1561,
        "verdict": verdict,
    }
    out_path = PROC / "nb1571_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))

    # Human-readable report
    print(f"n = {n}")
    print(f"Pearson(nb1561_cat, nb1560_lgb) = {pear:.4f}")
    print(f"RAE nb1561 (CatBoost BoB) = {rae_cat:.4f}")
    print(f"RAE nb1560 (LGBM BoB)     = {rae_lgb:.4f}")
    print("\nGrid sweep (w = weight on nb1561 CatBoost):")
    print(f"{'w_nb1561':>10}  {'RAE':>10}")
    for row in grid_rows:
        marker = "  <- best" if row["w_nb1561"] == best_grid["w_nb1561"] else ""
        print(f"{row['w_nb1561']:>10.2f}  {row['rae']:>10.4f}{marker}")
    print(f"\nBest grid: w_nb1561={best_grid['w_nb1561']:.2f} -> RAE {best_grid['rae']:.4f}")
    print(f"SLSQP in-sample:   w_nb1561={w_is:.4f} -> RAE {rae_is:.4f}")
    print(f"SLSQP 5-fold xfit: RAE {rae_cf:.4f} (mean w_nb1561 across folds = {np.mean(w_folds):.4f})")
    print(f"Per-fold w_nb1561: {[round(w, 4) for w in w_folds]}")
    print(f"\nBeats nb1561 by >=0.003 margin: {beats_nb1561}")
    print(f"Verdict: {verdict}")
    print(f"\nSaved: {out_path}")
    print(f"Saved: {PROC / 'nb1571_best_oof.npy'}")


if __name__ == "__main__":
    main()
