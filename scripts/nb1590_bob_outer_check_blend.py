"""nb1590 — Bag-of-Bags outer check on the nb1571 cross-fit blend.

Question: is nb1571's cross-fit 0.5130 a lucky single fold-split, or stable
across many seed choices? Outer-bag analogue for the downstream SLSQP blend.

Protocol:
1. Load p_cat = nb1561 BoB mean OOF, p_lgb = nb1560 BoB mean OOF, y = unblind truth (n=253).
2. For each of OUTER_SEEDS (40 seeds), run a fresh 5-fold cross-fit SLSQP blend
   on (p_cat, p_lgb) -> y. This mirrors how nb1571 produced 0.5130 with seed 17.
3. Each outer seed yields one cross-fit RAE = one "outer bag".
4. Report mean / median / std / min / max across the 40 outer bags.
5. Recommendation vs nb1571 single-seed RAE 0.5130 and nb1561 standalone 0.5155.

Outputs:
- data/processed/nb1590_summary.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path("d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge")
PROC = ROOT / "data" / "processed"

OUTER_SEEDS = list(range(40))  # 40 outer bags for the blend cross-fit


def rae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mean_y = float(np.mean(y_true))
    num = float(np.sum(np.abs(y_true - y_pred)))
    den = float(np.sum(np.abs(y_true - mean_y)))
    return num / den if den > 0 else float("inf")


def slsqp_blend(p1: np.ndarray, p2: np.ndarray, y: np.ndarray) -> tuple[float, float]:
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


def crossfit_slsqp(p1: np.ndarray, p2: np.ndarray, y: np.ndarray, n_splits: int,
                   seed: int) -> tuple[float, list[float]]:
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
    return rae(y, blended), weights


def load_truth_253() -> np.ndarray:
    for name in ["_audit_unblind_y.npy", "unblind_253_truth.npy",
                 "unblind_truth_253.npy", "truth_253.npy", "y_unblind_253.npy"]:
        p = PROC / name
        if p.exists():
            arr = np.load(p)
            if arr.shape[0] == 253:
                return arr.astype(float)
    raise FileNotFoundError("truth_253 not found")


def main() -> None:
    t0 = time.time()
    p_cat = np.load(PROC / "nb1561_bob_mean_oof.npy").astype(float)
    p_lgb = np.load(PROC / "nb1560_bob_mean_oof.npy").astype(float)
    y = load_truth_253()
    n = len(y)

    rae_cat = rae(y, p_cat)
    rae_lgb = rae(y, p_lgb)

    per_outer = []
    for s in OUTER_SEEDS:
        r, w_folds = crossfit_slsqp(p_cat, p_lgb, y, n_splits=5, seed=s)
        per_outer.append({"outer_seed": s, "rae": r,
                          "mean_w_nb1561": float(np.mean(w_folds))})

    raes = np.array([r["rae"] for r in per_outer], dtype=float)
    ws = np.array([r["mean_w_nb1561"] for r in per_outer], dtype=float)

    bob_mean = float(np.mean(raes))
    bob_median = float(np.median(raes))
    bob_std = float(np.std(raes, ddof=1))
    bob_min = float(np.min(raes))
    bob_max = float(np.max(raes))
    bob_q25 = float(np.quantile(raes, 0.25))
    bob_q75 = float(np.quantile(raes, 0.75))

    # nb1571 reference (cross-fit on seed=17)
    NB1571_CROSSFIT = 0.5130201158002257
    NB1561_STANDALONE = rae_cat  # 0.5155
    MARGIN = 0.003

    # Robust recommendation
    # If BoB mean stays under nb1561 - margin, the gain is real.
    beats_nb1561_mean = bob_mean < (NB1561_STANDALONE - MARGIN)
    beats_nb1561_median = bob_median < (NB1561_STANDALONE - MARGIN)
    # Fraction of outer bags that beat nb1561 standalone
    frac_beat_nb1561 = float(np.mean(raes < NB1561_STANDALONE))
    frac_beat_with_margin = float(np.mean(raes < (NB1561_STANDALONE - MARGIN)))
    # How extreme was the nb1571 single-seed result?
    rank_of_nb1571 = float(np.mean(raes <= NB1571_CROSSFIT))

    if beats_nb1561_mean and frac_beat_with_margin >= 0.5:
        recommendation = ("ADOPT nb1571 blend as PRIMARY — BoB mean beats nb1561 by "
                          ">=0.003 and majority of outer bags clear margin")
    elif beats_nb1561_mean:
        recommendation = ("LEAN nb1571 — BoB mean beats nb1561 by >=0.003 but "
                          "minority of outer bags clear full margin")
    elif bob_mean < NB1561_STANDALONE:
        recommendation = ("MARGINAL — BoB mean below nb1561 but under 0.003 margin; "
                          "HOLD nb1561 as PRIMARY")
    else:
        recommendation = ("HOLD nb1561 — BoB mean does NOT beat nb1561; nb1571 0.5130 "
                          "is a lucky-seed artifact")

    summary = {
        "n_unblind": n,
        "outer_seeds": OUTER_SEEDS,
        "n_outer_bags": len(OUTER_SEEDS),
        "rae_nb1561_standalone": rae_cat,
        "rae_nb1560_standalone": rae_lgb,
        "nb1571_crossfit_seed17": NB1571_CROSSFIT,
        "per_outer": per_outer,
        "bob_mean": bob_mean,
        "bob_median": bob_median,
        "bob_std": bob_std,
        "bob_min": bob_min,
        "bob_max": bob_max,
        "bob_q25": bob_q25,
        "bob_q75": bob_q75,
        "w_nb1561_mean": float(np.mean(ws)),
        "w_nb1561_median": float(np.median(ws)),
        "w_nb1561_std": float(np.std(ws, ddof=1)),
        "delta_bob_mean_vs_nb1561": bob_mean - NB1561_STANDALONE,
        "delta_bob_median_vs_nb1561": bob_median - NB1561_STANDALONE,
        "delta_bob_mean_vs_nb1571_seed17": bob_mean - NB1571_CROSSFIT,
        "beats_nb1561_at_margin_mean": bool(beats_nb1561_mean),
        "beats_nb1561_at_margin_median": bool(beats_nb1561_median),
        "frac_outer_bags_beat_nb1561": frac_beat_nb1561,
        "frac_outer_bags_beat_nb1561_with_margin": frac_beat_with_margin,
        "quantile_of_nb1571_seed17_in_bob": rank_of_nb1571,
        "margin": MARGIN,
        "recommendation": recommendation,
        "wall_sec": round(time.time() - t0, 2),
    }

    out_path = PROC / "nb1590_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))

    # Console report
    print(f"n = {n}, n_outer_bags = {len(OUTER_SEEDS)}")
    print(f"nb1561 standalone RAE      = {rae_cat:.4f}")
    print(f"nb1560 standalone RAE      = {rae_lgb:.4f}")
    print(f"nb1571 cross-fit (seed=17) = {NB1571_CROSSFIT:.4f}")
    print()
    print(f"BoB MEAN   RAE = {bob_mean:.4f}  (delta vs nb1561 = {bob_mean - NB1561_STANDALONE:+.4f})")
    print(f"BoB MEDIAN RAE = {bob_median:.4f}  (delta vs nb1561 = {bob_median - NB1561_STANDALONE:+.4f})")
    print(f"BoB STD        = {bob_std:.4f}")
    print(f"BoB MIN / MAX  = {bob_min:.4f} / {bob_max:.4f}")
    print(f"BoB Q25 / Q75  = {bob_q25:.4f} / {bob_q75:.4f}")
    print()
    print(f"Mean w_nb1561 across outer bags = {np.mean(ws):.4f}  (std {np.std(ws, ddof=1):.4f})")
    print(f"Fraction of bags beating nb1561 (no margin)   = {frac_beat_nb1561:.2%}")
    print(f"Fraction of bags beating nb1561 (>=0.003)     = {frac_beat_with_margin:.2%}")
    print(f"Quantile of nb1571 seed=17 in BoB             = {rank_of_nb1571:.2%}")
    print()
    print(f"RECOMMENDATION: {recommendation}")
    print(f"\nSaved: {out_path}  (wall {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
