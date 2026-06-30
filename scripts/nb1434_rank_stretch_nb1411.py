"""
nb1434 — Rank-stretch grid on nb1411 (naive 1/3 mean of nb1373+nb1352+nb1364).

Test if a smaller-bias pruned blend (nb1411) responds to scalar rank-stretch
calibration `mu + s*(p-mu)` where mu = pool mean of predictions.

Compares against nb1411 naive baseline (0.5037 in-sample on 253 unblind).
Decision margin = 0.003 RAE.

Outputs:
  data/processed/nb1434_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold


REPO = Path(r"d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge")
PROC = REPO / "data" / "processed"
OUT_JSON = PROC / "nb1434_summary.json"

S_GRID = [0.95, 1.00, 1.02, 1.05, 1.08, 1.10, 1.12, 1.15, 1.20]
NB1411_NAIVE_REF = 0.5037
DECISION_MARGIN = 0.003
N_FOLDS = 5
SEED = 0


def rae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    num = float(np.abs(y_true - y_pred).sum())
    den = float(np.abs(y_true - y_true.mean()).sum())
    return num / den if den > 0 else float("nan")


def apply_stretch(pred: np.ndarray, s: float, mu: float | None = None) -> np.ndarray:
    if mu is None:
        mu = float(pred.mean())
    return mu + s * (pred - mu)


def main() -> None:
    # --- Step 1: build nb1411 OOF as naive 1/3 mean -----------------------
    p1373 = np.load(PROC / "nb1373_mean_bag_oof.npy").astype(float)
    p1352 = np.load(PROC / "nb1352_mean_bag_oof.npy").astype(float)
    p1364 = np.load(PROC / "nb1364_mean_bag_oof.npy").astype(float)
    y = np.load(PROC / "_audit_unblind_y.npy").astype(float)
    assert p1373.shape == p1352.shape == p1364.shape == y.shape == (253,), (
        f"shape mismatch: {p1373.shape} {p1352.shape} {p1364.shape} {y.shape}"
    )

    pred = (p1373 + p1352 + p1364) / 3.0
    base_rae = rae(y, pred)

    # --- Step 2: pred_std vs truth_std ------------------------------------
    pred_std = float(pred.std(ddof=0))
    truth_std = float(y.std(ddof=0))
    ratio = pred_std / truth_std

    # --- Step 3: in-sample s grid -----------------------------------------
    mu_pool = float(pred.mean())
    grid_results = []
    for s in S_GRID:
        ps = apply_stretch(pred, s, mu=mu_pool)
        r = rae(y, ps)
        grid_results.append({"s": s, "rae": r, "delta_vs_naive": r - base_rae})
    best_in = min(grid_results, key=lambda d: d["rae"])

    # --- Step 4: 5-fold cross-fit s grid ----------------------------------
    # For each held-out fold, pick the best s on the training folds (using
    # train-fold's own mu), apply to held-out fold, accumulate predictions.
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_best_s = []
    cf_pred = np.zeros_like(pred)
    for tr_idx, te_idx in kf.split(pred):
        # pick best s on training fold
        y_tr = y[tr_idx]
        p_tr = pred[tr_idx]
        mu_tr = float(p_tr.mean())
        best_s = None
        best_r = float("inf")
        for s in S_GRID:
            ps_tr = apply_stretch(p_tr, s, mu=mu_tr)
            r_tr = rae(y_tr, ps_tr)
            if r_tr < best_r:
                best_r = r_tr
                best_s = s
        fold_best_s.append(best_s)
        # apply to held-out fold using train-fold mu
        cf_pred[te_idx] = apply_stretch(pred[te_idx], best_s, mu=mu_tr)
    cf_rae = rae(y, cf_pred)

    # --- Step 5: verdict ---------------------------------------------------
    # honest comparison: cross-fit RAE vs baseline naive RAE
    delta_in = best_in["rae"] - base_rae
    delta_cf = cf_rae - base_rae
    beats_nb1411_in = delta_in <= -DECISION_MARGIN
    beats_nb1411_cf = delta_cf <= -DECISION_MARGIN
    # also report vs the canonical nb1411 ref number (0.5037)
    delta_in_vs_ref = best_in["rae"] - NB1411_NAIVE_REF
    delta_cf_vs_ref = cf_rae - NB1411_NAIVE_REF

    if beats_nb1411_cf:
        verdict = "STRETCH_HELPS_NB1411"
    elif abs(delta_cf) < DECISION_MARGIN:
        verdict = "STRETCH_FLAT_ON_NB1411"
    else:
        verdict = "STRETCH_HURTS_NB1411"

    summary = {
        "tag": "nb1434",
        "n_unb": int(y.shape[0]),
        "components": ["nb1373", "nb1352", "nb1364"],
        "anchor": "nb1411_naive_third_mean",
        "anchor_rae": base_rae,
        "anchor_ref_rae": NB1411_NAIVE_REF,
        "pred_std": pred_std,
        "truth_std": truth_std,
        "pred_over_truth_std_ratio": ratio,
        "mu_pool": mu_pool,
        "s_grid": S_GRID,
        "grid_results": grid_results,
        "best_s_insample": best_in["s"],
        "best_rae_insample": best_in["rae"],
        "delta_insample_vs_anchor": delta_in,
        "delta_insample_vs_ref": delta_in_vs_ref,
        "crossfit_folds": N_FOLDS,
        "crossfit_seed": SEED,
        "crossfit_fold_best_s": fold_best_s,
        "crossfit_rae": cf_rae,
        "delta_crossfit_vs_anchor": delta_cf,
        "delta_crossfit_vs_ref": delta_cf_vs_ref,
        "decision_margin": DECISION_MARGIN,
        "beats_nb1411_insample": beats_nb1411_in,
        "beats_nb1411_crossfit": beats_nb1411_cf,
        "verdict": verdict,
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
