"""
nb1851 -- 4-way LGBM blend across BoB-validated variants.

Components:
  A = nb1780  (LGBM gbdt,  BoB mean)         standalone RAE ~0.508
  B = nb1821  (LGBM goss,  25-bag pooled)    standalone RAE ~0.5025
  C = nb1822  (LGBM feat_frac, BoB median)   standalone RAE ~0.508
  D = nb1771  (LGBM single bag)              standalone RAE ~0.510

Protocol:
  1. Load 4 OOFs + 253-row unblind truth.
  2. Standalone RAE per component.
  3. Pairwise Pearson on raw preds + on residuals.
  4. 4D simplex grid step 0.05 -- exhaustive.
  5. SLSQP 5-fold cross-fit (LB-honest) + in-sample SLSQP.
  6. Verdict at 0.003 margin vs nb1821 pooled 25-bag (0.5025).
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
ROOT = Path("d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge")
PROC = ROOT / "data" / "processed"

OOF_PATHS = {
    "nb1780": PROC / "nb1780_bob_mean_oof.npy",       # gbdt
    "nb1821": PROC / "nb1821_pooled_25bag_oof.npy",   # goss 25-bag
    "nb1822": PROC / "nb1822_bob_median_oof.npy",     # feat_frac BoB median
    "nb1771": PROC / "nb1771_mean_bag_oof.npy",       # single bag
}
TRUTH_PATH = PROC / "_audit_unblind_y.npy"

ANCHOR_TAG = "nb1821"
ANCHOR_RAE = 0.5025
MARGIN = 0.003
GRID_STEP = 0.05


def rae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    num = np.abs(y_true - y_pred).sum()
    den = np.abs(y_true - y_true.mean()).sum()
    return float(num / den)


def simplex_grid_4d(step: float):
    """Yield (w0,w1,w2,w3) with w_i >= 0, sum == 1, on a step grid."""
    n_steps = int(round(1.0 / step))
    for i in range(n_steps + 1):
        for j in range(n_steps + 1 - i):
            for k in range(n_steps + 1 - i - j):
                lvl = n_steps - i - j - k
                yield (
                    i * step,
                    j * step,
                    k * step,
                    lvl * step,
                )


def main() -> None:
    # --- Load ---
    tags = list(OOF_PATHS.keys())
    preds = {t: np.load(p).astype(float).reshape(-1) for t, p in OOF_PATHS.items()}
    y = np.load(TRUTH_PATH).astype(float).reshape(-1)
    n = y.shape[0]
    for t, arr in preds.items():
        assert arr.shape == (n,), f"shape mismatch {t}: {arr.shape}"

    P = np.column_stack([preds[t] for t in tags])  # (n, 4)

    # --- Standalone RAE ---
    standalone = {t: rae(y, preds[t]) for t in tags}

    # --- Pairwise Pearson (raw + residual) ---
    pearson_raw = pd.DataFrame(
        np.corrcoef(P.T), index=tags, columns=tags
    ).round(4)
    R = P - y[:, None]
    pearson_resid = pd.DataFrame(
        np.corrcoef(R.T), index=tags, columns=tags
    ).round(4)

    # --- 4D simplex grid ---
    grid_records = []
    for w in simplex_grid_4d(GRID_STEP):
        w_arr = np.array(w)
        p = P @ w_arr
        grid_records.append({
            "w_nb1780": w[0],
            "w_nb1821": w[1],
            "w_nb1822": w[2],
            "w_nb1771": w[3],
            "rae": rae(y, p),
        })
    grid_df = pd.DataFrame(grid_records).sort_values("rae").reset_index(drop=True)
    top5 = grid_df.head(5)
    best_grid_w = top5.iloc[0][["w_nb1780", "w_nb1821", "w_nb1822", "w_nb1771"]].tolist()
    best_grid_rae = float(top5.iloc[0]["rae"])

    # --- SLSQP cross-fit (5-fold) ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    pred_cf = np.full(n, np.nan, dtype=float)
    fold_weights = []
    for tr, te in kf.split(np.arange(n)):
        Ptr = P[tr]
        ytr = y[tr]

        def loss(w: np.ndarray) -> float:
            return float(np.mean(np.abs(ytr - Ptr @ w)))

        cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
        bnds = [(0.0, 1.0)] * 4
        w0 = np.full(4, 0.25)
        res = minimize(loss, w0, method="SLSQP", bounds=bnds, constraints=cons)
        w_hat = res.x
        fold_weights.append([float(x) for x in w_hat])
        pred_cf[te] = P[te] @ w_hat
    rae_slsqp_cf = rae(y, pred_cf)

    # --- SLSQP in-sample ---
    def loss_all(w: np.ndarray) -> float:
        return float(np.mean(np.abs(y - P @ w)))

    res_all = minimize(
        loss_all,
        np.full(4, 0.25),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * 4,
        constraints=({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},),
    )
    w_in = [float(x) for x in res_all.x]
    rae_slsqp_in = rae(y, P @ np.array(w_in))

    # --- Verdict ---
    best_overall = min(best_grid_rae, rae_slsqp_cf)
    delta = ANCHOR_RAE - best_overall
    pass_gate = bool(delta >= MARGIN)
    verdict = "PASS" if pass_gate else "FAIL"

    # --- Report ---
    print("=" * 78)
    print("nb1851 -- 4-way LGBM blend (nb1780 + nb1821 + nb1822 + nb1771)")
    print("=" * 78)
    print(f"n unblind                : {n}")
    print("Standalone RAE:")
    for t in tags:
        print(f"  {t:8s}             : {standalone[t]:.4f}")
    print("-" * 78)
    print("Pairwise Pearson (raw preds):")
    print(pearson_raw.to_string())
    print()
    print("Pairwise Pearson (residuals = pred - y):")
    print(pearson_resid.to_string())
    print("-" * 78)
    print(f"4D simplex grid step={GRID_STEP}  total points={len(grid_df)}")
    print("Top-5 grid points:")
    print(top5.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("-" * 78)
    print(
        "SLSQP in-sample        : w=["
        + ", ".join(f"{w:.3f}" for w in w_in)
        + f"]  RAE={rae_slsqp_in:.4f}"
    )
    print(f"SLSQP cross-fit (5fld) : RAE={rae_slsqp_cf:.4f}")
    for i, fw in enumerate(fold_weights):
        print(
            f"  fold {i}: w=["
            + ", ".join(f"{w:.3f}" for w in fw)
            + "]"
        )
    print("-" * 78)
    print(f"Anchor ({ANCHOR_TAG}) RAE   : {ANCHOR_RAE:.4f}")
    print(f"Best blend RAE         : {best_overall:.4f}")
    print(f"Delta (anchor - blend) : {delta:+.4f}  (margin {MARGIN})")
    print(f"VERDICT                : {verdict}")
    print("=" * 78)

    # --- Save artifacts ---
    summary = {
        "tag": "nb1851",
        "n": n,
        "components": tags,
        "standalone_rae": standalone,
        "pearson_raw": pearson_raw.to_dict(),
        "pearson_residual": pearson_resid.to_dict(),
        "grid_step": GRID_STEP,
        "grid_n_points": int(len(grid_df)),
        "grid_top5": top5.to_dict(orient="records"),
        "best_grid_w": {
            "nb1780": float(best_grid_w[0]),
            "nb1821": float(best_grid_w[1]),
            "nb1822": float(best_grid_w[2]),
            "nb1771": float(best_grid_w[3]),
        },
        "best_grid_rae": best_grid_rae,
        "slsqp_insample_w": {
            "nb1780": w_in[0],
            "nb1821": w_in[1],
            "nb1822": w_in[2],
            "nb1771": w_in[3],
        },
        "slsqp_insample_rae": rae_slsqp_in,
        "slsqp_crossfit_rae": rae_slsqp_cf,
        "slsqp_fold_weights": fold_weights,
        "anchor_tag": ANCHOR_TAG,
        "anchor_rae": ANCHOR_RAE,
        "margin": MARGIN,
        "delta": delta,
        "verdict": verdict,
    }
    out_json = PROC / "nb1851_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    np.save(PROC / "nb1851_blend_oof.npy", P @ np.array(w_in))
    print(f"Saved: {out_json}")
    print(f"Saved: {PROC / 'nb1851_blend_oof.npy'}")


if __name__ == "__main__":
    main()
