"""nb1062 -- Continuous LOWESS-style smoothed stretch on the nb1014 anchor.

nb1053 replaces nb562's single scalar `s` with 5 discrete quantile-bin
stretches. Diagnostic per-bin scans showed an inverted-U over the prediction
range (s smaller at center, larger at extremes), but the 5-bin step function
introduces visible jumps at the bin edges that the cross-fit penalises.

Hypothesis: replace the discrete bin map with a continuous smoothed
`s(pred)` learned via LOWESS-style local linear regression. The smooth
curve captures the same inverted-U shape without the discontinuities,
and one global `mu` keeps the anchor honest.

Procedure (honest 5-fold cross-fit on the 253 unblind):
  1. Load te_nb1014.npy, subset to 253 unblind.
  2. For each fold:
       a. mu = mean of train-fold predictions.
       b. Fit LOWESS f(pred) = E[truth | pred] on train fold, frac=0.3.
       c. Convert to a continuous stretch: s(pred) = (f(pred) - mu) /
          (pred - mu), with a small jitter to guard against pred ~= mu.
       d. Apply pointwise on val: stretched[i] = mu + s(pred[i]) *
          (pred[i] - mu)  (algebraically just f(pred[i])).
  3. Pooled cross-fit RAE on the 253.
  4. Deploy: fit f on all 253, apply to all 513.

statsmodels is not installed in this env, so LOWESS is hand-rolled with
tricube-weighted local linear regression on the k-nearest neighbours in
prediction space (k = round(frac * n_train)). This is the same kernel
Cleveland 1979 uses; we evaluate it pointwise on the val (and deploy)
prediction grid.

Outputs:
  data/processed/te_nb1062.npy
  data/processed/nb1062_summary.json
  submissions/nb1062_lowess_stretch.csv
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1062"
ANCHOR = "nb1014"
FRAC = 0.3
N_FOLDS = 5
SEED = 42

NB1014_BAGGED_HONEST_RAE = 0.5930
NB1053_HONEST_RAE = None  # filled from summary if present


def lowess_fit_predict(x_tr: np.ndarray, y_tr: np.ndarray,
                       x_query: np.ndarray, frac: float = FRAC) -> np.ndarray:
    """Hand-rolled LOWESS: tricube-weighted local linear regression.

    For each query x*, take the k = round(frac * n_tr) nearest train x's,
    weight by tricube of normalised distance, fit a weighted linear
    regression, evaluate at x*.
    """
    n_tr = len(x_tr)
    k = max(int(round(frac * n_tr)), 3)
    out = np.empty(len(x_query), dtype=np.float64)
    for i, xq in enumerate(x_query):
        d = np.abs(x_tr - xq)
        # k-th smallest distance -> bandwidth (Cleveland's "h").
        if k >= n_tr:
            h = d.max() + 1e-12
        else:
            h = np.partition(d, k - 1)[k - 1] + 1e-12
        u = np.clip(d / h, 0.0, 1.0)
        w = (1.0 - u ** 3) ** 3  # tricube
        # Weighted least squares for y = a + b*x.
        W = w.sum()
        if W <= 0:
            out[i] = y_tr.mean()
            continue
        wx = (w * x_tr).sum()
        wy = (w * y_tr).sum()
        wxx = (w * x_tr * x_tr).sum()
        wxy = (w * x_tr * y_tr).sum()
        denom = W * wxx - wx * wx
        if abs(denom) < 1e-12:
            out[i] = wy / W
            continue
        b = (W * wxy - wx * wy) / denom
        a = (wy - b * wx) / W
        out[i] = a + b * xq
    return out


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- continuous LOWESS stretch on te_{ANCHOR} (frac={FRAC})")
    print("=" * 78)

    # ---- Load anchor predictions ----
    te = load_test()
    te_names = te["name"].values
    preds_513 = np.load(DATA_PROCESSED / f"te_{ANCHOR}.npy").astype(np.float64)
    print(f"[load] te_{ANCHOR} shape={preds_513.shape}  "
          f"mean={preds_513.mean():.3f}  std={preds_513.std():.3f}")

    # ---- 253 unblind subset ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    p_unb = preds_513[unb_idx]
    n_unb = len(y_unb)
    print(f"[load] p_unb={p_unb.shape}  truth_std={y_unb.std():.4f}  "
          f"pred_std={p_unb.std():.4f}  "
          f"(compression={p_unb.std() / y_unb.std():.3f})")

    in_rae_baseline = float(rae(y_unb, p_unb))
    print(f"\n[baseline] in_RAE(te_{ANCHOR} on 253) = {in_rae_baseline:.4f}")

    # ---- Try to fetch nb1053 pooled RAE for the verdict line ----
    nb1053_path = DATA_PROCESSED / "nb1053_summary.json"
    nb1053_rae = None
    if nb1053_path.exists():
        try:
            nb1053_rae = float(json.load(open(nb1053_path))[
                "pooled_cross_fit_rae"])
        except Exception:
            nb1053_rae = None

    # =================================================================
    # Honest 5-fold cross-fit
    # =================================================================
    print("\n" + "-" * 78)
    print(f"HONEST CROSS-FIT  KFold(seed={SEED})  LOWESS frac={FRAC}")
    print("-" * 78)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.full(n_unb, np.nan)
    folds = []
    for k, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n_unb))):
        p_tr, y_tr = p_unb[tr_loc], y_unb[tr_loc]
        p_va, y_va = p_unb[va_loc], y_unb[va_loc]
        mu = float(p_tr.mean())
        # f(pred) approximates E[truth | pred] smoothly.
        f_va = lowess_fit_predict(p_tr, y_tr, p_va, frac=FRAC)
        # Stretched form: equivalent to f(pred) but exposes s(pred).
        denom = p_va - mu
        s_va = np.where(np.abs(denom) > 1e-6,
                        (f_va - mu) / np.where(np.abs(denom) > 1e-6,
                                                denom, 1.0),
                        1.0)
        stretched_va = mu + s_va * (p_va - mu)
        oof[va_loc] = stretched_va
        rae_va = float(rae(y_va, stretched_va))
        # Train-fit RAE for the smoother on its own training data.
        f_tr = lowess_fit_predict(p_tr, y_tr, p_tr, frac=FRAC)
        rae_tr = float(rae(y_tr, f_tr))
        s_va_summary = (float(s_va.min()), float(np.median(s_va)),
                        float(s_va.max()))
        folds.append({
            "fold": k,
            "mu": mu,
            "train_rae": rae_tr,
            "val_rae": rae_va,
            "n_va": int(len(va_loc)),
            "s_va_min": s_va_summary[0],
            "s_va_med": s_va_summary[1],
            "s_va_max": s_va_summary[2],
        })
        print(f"   fold {k}: mu={mu:.3f}  train_RAE={rae_tr:.4f}  "
              f"val_RAE={rae_va:.4f}  s_va(min/med/max)="
              f"{s_va_summary[0]:.2f}/{s_va_summary[1]:.2f}/"
              f"{s_va_summary[2]:.2f}")

    pooled = float(rae(y_unb, oof))
    print(f"\n[honest] pooled cross-fit RAE on 253 unblind = {pooled:.4f}")

    # =================================================================
    # Diagnostic: shape of s(pred) on a grid (deploy fit on all 253).
    # =================================================================
    mu_all = float(p_unb.mean())
    grid = np.linspace(p_unb.min(), p_unb.max(), 11)
    f_grid = lowess_fit_predict(p_unb, y_unb, grid, frac=FRAC)
    denom_g = grid - mu_all
    s_grid = np.where(np.abs(denom_g) > 1e-6,
                      (f_grid - mu_all) / np.where(np.abs(denom_g) > 1e-6,
                                                    denom_g, 1.0),
                      1.0)
    print("\n[diag] s(pred) shape on 253 (deploy fit):")
    print("   pred     f(pred)   s(pred)")
    for g, fg, sg in zip(grid, f_grid, s_grid):
        print(f"   {g:>5.2f}   {fg:>6.3f}    {sg:>5.2f}")

    # =================================================================
    # Deploy: fit on all 253, apply pointwise to all 513.
    # =================================================================
    print("\n" + "-" * 78)
    print("DEPLOY  (fit on all 253, apply to 513)")
    print("-" * 78)
    f_513 = lowess_fit_predict(p_unb, y_unb, preds_513, frac=FRAC)
    deploy_513 = f_513.astype(np.float32)
    in_rae_deploy = float(rae(y_unb, deploy_513[unb_idx].astype(np.float64)))
    print(f"   deploy mu     = {mu_all:.4f}")
    print(f"   in-sample 253 = {in_rae_deploy:.4f}  (overfit lower bound)")
    print(f"   te(513) mean  = {deploy_513.mean():.3f}  "
          f"std={deploy_513.std():.3f}")
    print(f"   anchor (513)  = mean {preds_513.mean():.3f}  "
          f"std {preds_513.std():.3f}")

    # =================================================================
    # Save artifacts
    # =================================================================
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy_513)
    plain = SUBMISSIONS / f"{TAG}_lowess_stretch.csv"
    pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te_names,
        "pEC50": deploy_513,
    }).to_csv(plain, index=False)
    print(f"\n[save] te_{TAG}.npy")
    print(f"[save] {plain}")

    delta_vs_nb1014 = pooled - NB1014_BAGGED_HONEST_RAE
    if delta_vs_nb1014 <= -0.005:
        verdict = "BEATS_NB1014"
    elif abs(delta_vs_nb1014) < 0.005:
        verdict = "TIES_NB1014"
    else:
        verdict = "WORSE_THAN_NB1014"
    delta_vs_nb1053 = (pooled - nb1053_rae) if nb1053_rae is not None else None
    print(f"\n[verdict] vs nb1014 bagged ({NB1014_BAGGED_HONEST_RAE}): "
          f"delta={delta_vs_nb1014:+.4f}  -> {verdict}")
    if delta_vs_nb1053 is not None:
        print(f"[verdict] vs nb1053 pooled ({nb1053_rae:.4f}): "
              f"delta={delta_vs_nb1053:+.4f}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "frac": FRAC,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "in_rae_anchor_on_253": in_rae_baseline,
        "pooled_cross_fit_rae": pooled,
        "in_rae_deploy_on_253": in_rae_deploy,
        "nb1014_bagged_honest_rae": NB1014_BAGGED_HONEST_RAE,
        "nb1053_pooled_cross_fit_rae": nb1053_rae,
        "delta_vs_nb1014_bagged": delta_vs_nb1014,
        "delta_vs_nb1053": delta_vs_nb1053,
        "verdict": verdict,
        "deploy_mu": mu_all,
        "deploy_grid_pred": grid.tolist(),
        "deploy_grid_f": f_grid.tolist(),
        "deploy_grid_s": s_grid.tolist(),
        "folds": folds,
        "deploy_te_mean": float(deploy_513.mean()),
        "deploy_te_std": float(deploy_513.std()),
        "anchor_te_mean": float(preds_513.mean()),
        "anchor_te_std": float(preds_513.std()),
        "plain_submission": str(plain),
        "wall_sec": round(time.time() - t0, 2),
    }
    with open(DATA_PROCESSED / f"{TAG}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {DATA_PROCESSED / f'{TAG}_summary.json'}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   anchor                    = {ANCHOR}")
    print(f"   in_RAE anchor on 253      = {in_rae_baseline:.4f}")
    print(f"   pooled cross-fit RAE      = {pooled:.4f}")
    print(f"   in-sample (deploy)        = {in_rae_deploy:.4f}")
    print(f"   nb1014 bagged honest      = {NB1014_BAGGED_HONEST_RAE:.4f}")
    print(f"   delta vs nb1014           = {delta_vs_nb1014:+.4f}")
    if nb1053_rae is not None:
        print(f"   nb1053 pooled             = {nb1053_rae:.4f}")
        print(f"   delta vs nb1053           = {delta_vs_nb1053:+.4f}")
    print(f"   verdict                   = {verdict}")
    print(f"   wall                      = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("in_rae_anchor_on_253", "pooled_cross_fit_rae",
              "in_rae_deploy_on_253", "delta_vs_nb1014_bagged",
              "delta_vs_nb1053", "verdict", "plain_submission"):
        print(f"  {k}: {res.get(k)}")
