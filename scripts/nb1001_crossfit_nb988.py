"""nb1001 -- Honest 5-fold cross-fit of (chemprop_aux + nb972 + stretch).

The nb988 (rank-stretch on nb982 blend) won on the 253 unblind subset
in-sample at RAE ~0.5863. nb982 itself was a SLSQP blend fit on the
253, and nb988 stretched it -- both stages saw the 253 truth.

This script asks: is that 0.5863 a real win, or in-sample optimism?

Procedure:
  Pool = [chemprop_aux, nb972_long_train] on the 253 unblind.
  5-fold KFold on the 253:
    For each fold f:
      a. Fit on the 4 train folds:
            - SLSQP for w0 (chemprop weight, K=2 -> w1 = 1 - w0)
            - Scan s in {1.00, 1.05, ..., 2.00} to pick best stretch
              on the train folds (using the train-fold mean as mu).
      b. Apply (w0_f, s_f) to the held-out fold predictions
         (using the same mu_train as the centering anchor).
  Pooled honest cross-fit RAE on the 253.
  Compare to nb988 in-sample 0.5863.

Deploy: refit (w0, s) on all 253, apply to the 513 te files.

Outputs:
  data/processed/te_nb1001.npy
  data/processed/nb1001_summary.json
  submissions/nb1001_crossfit_chempropaux_nb972_stretch.csv
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

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1001"
CANDIDATES = ["chemprop_aux", "nb972_long_train"]
STRETCH_GRID = np.round(np.arange(1.00, 2.001, 0.05), 2).tolist()
N_FOLDS = 5
SEED = 42
NB988_IN_SAMPLE = 0.5863


def load_te(name: str, te_names: np.ndarray) -> np.ndarray:
    npy = DATA_PROCESSED / f"te_{name}.npy"
    if npy.exists():
        return np.load(npy).astype(np.float64)
    sub = pd.read_csv(SUBMISSIONS / f"{name}.csv")
    assert (sub["Molecule Name"].values == te_names).all(), (
        f"{name}: submission row order does not match test order")
    return sub["pEC50"].values.astype(np.float64)


def slsqp_w0(p0: np.ndarray, p1: np.ndarray, y: np.ndarray) -> float:
    """Fit w0 in [0,1] minimizing SSE of w0*p0 + (1-w0)*p1 vs y. Returns w0."""
    P = np.column_stack([p0, p1])
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0), (0.0, 1.0)]
    res = minimize(
        lambda w: float(np.sum((P @ w - y) ** 2)),
        np.array([0.5, 0.5]),
        method="SLSQP", bounds=bnds, constraints=cons,
        options={"ftol": 1e-10, "maxiter": 500},
    )
    return float(res.x[0])


def best_stretch_on(blend_train: np.ndarray, y_train: np.ndarray,
                    mu: float) -> tuple[float, float]:
    """Grid-scan s on the train-fold; return (best_s, best_train_rae)."""
    best_s, best_r = 1.0, float("inf")
    for s in STRETCH_GRID:
        stretched = mu + s * (blend_train - mu)
        r = float(rae(y_train, stretched))
        if r < best_r:
            best_r = r
            best_s = float(s)
    return best_s, best_r


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- honest 5-fold cross-fit of "
          f"(chemprop_aux + nb972 + stretch)")
    print("=" * 78)

    # ---- Load 513 test ----
    te = load_test()
    te_names = te["name"].values
    n_te = len(te_names)
    preds_513 = np.column_stack([load_te(c, te_names) for c in CANDIDATES])
    print(f"[load] preds_513 shape = {preds_513.shape}")

    # ---- Load 253 unblind ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    P_unb = preds_513[unb_idx]
    n_unb = len(y_unb)
    print(f"[load] P_unb shape = {P_unb.shape}, y shape = {y_unb.shape}")

    # ---- Individual in_RAE sanity ----
    print("\n[indiv] in_RAE on 253 unblind:")
    indiv_rae = {}
    for j, c in enumerate(CANDIDATES):
        r = float(rae(y_unb, P_unb[:, j]))
        indiv_rae[c] = r
        print(f"   {c:30s}: {r:.4f}")

    # =================================================================
    # 5-fold cross-fit (SLSQP w0 + stretch grid both fit on train folds)
    # =================================================================
    print("\n" + "-" * 78)
    print(f"5-FOLD CROSS-FIT  (SLSQP w0 + stretch grid {STRETCH_GRID[0]}"
          f"..{STRETCH_GRID[-1]})")
    print("-" * 78)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.full(n_unb, np.nan)
    fold_rows = []
    for k, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n_unb))):
        # a) SLSQP w0 on train fold
        w0_f = slsqp_w0(P_unb[tr_loc, 0], P_unb[tr_loc, 1], y_unb[tr_loc])
        blend_tr = w0_f * P_unb[tr_loc, 0] + (1.0 - w0_f) * P_unb[tr_loc, 1]
        # b) stretch grid on train fold (mu = train blend mean)
        mu_tr = float(blend_tr.mean())
        s_f, rae_tr = best_stretch_on(blend_tr, y_unb[tr_loc], mu_tr)
        # c) apply to held-out fold using SAME mu_tr (no peeking)
        blend_va = w0_f * P_unb[va_loc, 0] + (1.0 - w0_f) * P_unb[va_loc, 1]
        oof[va_loc] = mu_tr + s_f * (blend_va - mu_tr)
        rae_va = float(rae(y_unb[va_loc], oof[va_loc]))
        fold_rows.append({"fold": k, "w0": w0_f, "s": s_f,
                          "mu_tr": mu_tr,
                          "train_rae": rae_tr, "val_rae": rae_va,
                          "n_va": int(len(va_loc))})
        print(f"   fold {k}: w0={w0_f:.3f}  s={s_f:.2f}  mu_tr={mu_tr:.3f}  "
              f"train_RAE={rae_tr:.4f}  val_RAE={rae_va:.4f}")

    pooled_rae = float(rae(y_unb, oof))
    print(f"\n[cv] pooled honest cross-fit RAE on 253 = {pooled_rae:.4f}")

    # =================================================================
    # Comparison vs nb988 in-sample
    # =================================================================
    delta = pooled_rae - NB988_IN_SAMPLE
    print("\n" + "-" * 78)
    print("COMPARISON vs nb988 (in-sample 0.5863)")
    print("-" * 78)
    print(f"   nb988 in-sample (overfit)      = {NB988_IN_SAMPLE:.4f}")
    print(f"   nb1001 honest cross-fit (253)  = {pooled_rae:.4f}")
    print(f"   delta (cross-fit - in-sample)  = {delta:+.4f}")
    if delta < 0.02:
        verdict = "REAL_WIN"
        print("   verdict                        = REAL WIN  "
              "(cross-fit ~ in-sample)")
    elif delta < 0.05:
        verdict = "MILD_OPTIMISM"
        print("   verdict                        = MILD IN-SAMPLE OPTIMISM")
    else:
        verdict = "IN_SAMPLE_OPTIMISM"
        print("   verdict                        = IN-SAMPLE OPTIMISM  "
              "(cross-fit >> in-sample)")

    # =================================================================
    # Deploy: refit (w0, s) on all 253
    # =================================================================
    print("\n" + "-" * 78)
    print("DEPLOY  (refit on all 253)")
    print("-" * 78)
    w0_deploy = slsqp_w0(P_unb[:, 0], P_unb[:, 1], y_unb)
    blend_unb_all = w0_deploy * P_unb[:, 0] + (1.0 - w0_deploy) * P_unb[:, 1]
    mu_deploy = float(blend_unb_all.mean())
    s_deploy, in_rae_blend = best_stretch_on(blend_unb_all, y_unb, mu_deploy)
    in_rae_final = float(rae(y_unb,
                              mu_deploy + s_deploy
                              * (blend_unb_all - mu_deploy)))
    print(f"   deploy w0           = {w0_deploy:.4f}  "
          f"(chemprop_aux weight; w1={1.0 - w0_deploy:.4f} for nb972)")
    print(f"   deploy mu (blend)   = {mu_deploy:.4f}")
    print(f"   deploy s            = {s_deploy:.2f}")
    print(f"   in-sample RAE (253) = {in_rae_final:.4f}  "
          "(overfit lower bound)")

    # Apply to all 513
    blend_513 = (w0_deploy * preds_513[:, 0]
                 + (1.0 - w0_deploy) * preds_513[:, 1])
    # Use deploy mu (computed from 253 blend); stretching around that mean
    deploy_513 = (mu_deploy + s_deploy * (blend_513 - mu_deploy)).astype(np.float32)
    print(f"   te(513) mean/std    = "
          f"{deploy_513.mean():.3f} / {deploy_513.std():.3f}")

    # =================================================================
    # Save
    # =================================================================
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy_513)
    plain = SUBMISSIONS / f"{TAG}_crossfit_chempropaux_nb972_stretch.csv"
    pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te_names,
        "pEC50": deploy_513,
    }).to_csv(plain, index=False)
    print(f"\n[save] te_{TAG}.npy")
    print(f"[save] {plain}")

    summary = {
        "tag": TAG,
        "candidates": CANDIDATES,
        "indiv_in_rae": indiv_rae,
        "stretch_grid": STRETCH_GRID,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "fold_results": fold_rows,
        "pooled_cv_rae_253": pooled_rae,
        "nb988_in_sample_rae": NB988_IN_SAMPLE,
        "delta_crossfit_vs_in_sample": delta,
        "verdict": verdict,
        "deploy_w0_chemprop_aux": float(w0_deploy),
        "deploy_w1_nb972": float(1.0 - w0_deploy),
        "deploy_mu_blend": mu_deploy,
        "deploy_s": float(s_deploy),
        "in_sample_rae_overfit_bound": in_rae_final,
        "deploy_te_mean": float(deploy_513.mean()),
        "deploy_te_std": float(deploy_513.std()),
        "plain_submission": str(plain),
        "wall_sec": round(time.time() - t0, 2),
    }
    with open(DATA_PROCESSED / f"{TAG}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {DATA_PROCESSED / f'{TAG}_summary.json'}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   pool                       = {CANDIDATES}")
    print(f"   honest cross-fit RAE (253) = {pooled_rae:.4f}")
    print(f"   nb988 in-sample            = {NB988_IN_SAMPLE:.4f}")
    print(f"   delta                      = {delta:+.4f}")
    print(f"   verdict                    = {verdict}")
    print(f"   deploy (w0,s)              = "
          f"({w0_deploy:.3f}, {s_deploy:.2f})")
    print(f"   wall                       = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("pooled_cv_rae_253", "delta_crossfit_vs_in_sample",
              "verdict", "deploy_w0_chemprop_aux", "deploy_s",
              "in_sample_rae_overfit_bound", "plain_submission"):
        print(f"  {k}: {res.get(k)}")
