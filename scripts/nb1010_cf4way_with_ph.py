"""nb1010 -- Honest 5-fold cross-fit of 4-way SLSQP + stretch with
real gudhi-version persistence homology (nb914) added to the pool.

Pool (K=4):
    0. chemprop_aux            (anchor; PRIMARY-1 candidate)
    1. nb972_long_train        (long-train Chemprop)
    2. nb914_persistence_homology  (real gudhi v0.7062 PH features)
    3. nb960_pseudo_self_train (pseudo-label self-train)

Procedure:
    5-fold KFold on the 253 unblind.
    Per fold:
        a. SLSQP w0..w3 in [0,1], sum=1, on the 4 train folds (SSE).
        b. Scan s in {1.00, 1.05, ..., 2.00} on train-fold blend
           around train-fold blend mean mu.
    Apply (w_f, s_f, mu_f) to held-out fold predictions.
    Pooled honest cross-fit RAE on the 253.

Compare to nb1001 (2-way + stretch) honest 0.5994.
Target: < 0.5994 (i.e. PH + pseudo-self-train add orthogonal value).

Deploy: refit (w, s) on all 253, apply to the 513 te files.

Outputs:
    data/processed/te_nb1010.npy
    data/processed/nb1010_summary.json
    submissions/nb1010_cf4way_with_ph.csv
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

TAG = "nb1010"
# (display_name, te_npy_stem, submission_csv_stem)
CANDIDATES = [
    ("chemprop_aux",                "chemprop_aux",      "chemprop_aux"),
    ("nb972_long_train",            "nb972_long_train",  "nb972_long_train_optim"),
    ("nb914_persistence_homology",  "nb914",             "nb914_persistence_homology"),
    ("nb960_pseudo_self_train",     "nb960",             "nb960_pseudo_self_train"),
]
STRETCH_GRID = np.round(np.arange(1.00, 2.001, 0.05), 2).tolist()
N_FOLDS = 5
SEED = 42
NB1001_CROSSFIT = 0.5994


def load_te(stem: str, csv_stem: str, te_names: np.ndarray) -> np.ndarray:
    npy = DATA_PROCESSED / f"te_{stem}.npy"
    if npy.exists():
        arr = np.load(npy).astype(np.float64)
        if arr.shape[0] == len(te_names):
            return arr
        print(f"   [warn] te_{stem}.npy shape {arr.shape}, expected "
              f"{len(te_names)}; falling back to csv")
    sub = pd.read_csv(SUBMISSIONS / f"{csv_stem}.csv")
    assert (sub["Molecule Name"].values == te_names).all(), (
        f"{csv_stem}: submission row order does not match test order")
    return sub["pEC50"].values.astype(np.float64)


def slsqp_simplex(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit w on K-simplex (>=0, sum=1) minimizing SSE of P@w vs y."""
    K = P.shape[1]
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * K
    res = minimize(
        lambda w: float(np.sum((P @ w - y) ** 2)),
        np.full(K, 1.0 / K),
        method="SLSQP", bounds=bnds, constraints=cons,
        options={"ftol": 1e-10, "maxiter": 1000},
    )
    w = np.clip(res.x, 0.0, 1.0)
    s = w.sum()
    return w / s if s > 0 else np.full(K, 1.0 / K)


def best_stretch_on(blend_train: np.ndarray, y_train: np.ndarray,
                    mu: float) -> tuple[float, float]:
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
    print(f"{TAG} -- honest 5-fold cross-fit, 4-way SLSQP + stretch "
          "(chemprop_aux + nb972 + nb914_PH + nb960)")
    print("=" * 78)

    # ---- Load 513 test ----
    te = load_test()
    te_names = te["name"].values
    n_te = len(te_names)
    cols = []
    for disp, stem, csv_stem in CANDIDATES:
        cols.append(load_te(stem, csv_stem, te_names))
    preds_513 = np.column_stack(cols)
    K = preds_513.shape[1]
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
    for j, (disp, _, _) in enumerate(CANDIDATES):
        r = float(rae(y_unb, P_unb[:, j]))
        indiv_rae[disp] = r
        print(f"   {disp:32s}: {r:.4f}")

    # =================================================================
    # 5-fold cross-fit
    # =================================================================
    print("\n" + "-" * 78)
    print(f"5-FOLD CROSS-FIT  (SLSQP w0..w{K-1} simplex + stretch "
          f"{STRETCH_GRID[0]}..{STRETCH_GRID[-1]})")
    print("-" * 78)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.full(n_unb, np.nan)
    fold_rows = []
    for k, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n_unb))):
        # a) SLSQP weights on train fold
        w_f = slsqp_simplex(P_unb[tr_loc], y_unb[tr_loc])
        blend_tr = P_unb[tr_loc] @ w_f
        # b) stretch grid on train fold (mu = train blend mean)
        mu_tr = float(blend_tr.mean())
        s_f, rae_tr = best_stretch_on(blend_tr, y_unb[tr_loc], mu_tr)
        # c) apply to held-out fold using SAME (w_f, mu_tr, s_f)
        blend_va = P_unb[va_loc] @ w_f
        oof[va_loc] = mu_tr + s_f * (blend_va - mu_tr)
        rae_va = float(rae(y_unb[va_loc], oof[va_loc]))
        fold_rows.append({
            "fold": k,
            "w": [float(x) for x in w_f],
            "s": s_f,
            "mu_tr": mu_tr,
            "train_rae": rae_tr,
            "val_rae": rae_va,
            "n_va": int(len(va_loc)),
        })
        w_str = ",".join(f"{x:.3f}" for x in w_f)
        print(f"   fold {k}: w=[{w_str}]  s={s_f:.2f}  mu_tr={mu_tr:.3f}  "
              f"train_RAE={rae_tr:.4f}  val_RAE={rae_va:.4f}")

    pooled_rae = float(rae(y_unb, oof))
    print(f"\n[cv] pooled honest cross-fit RAE on 253 = {pooled_rae:.4f}")

    # =================================================================
    # Comparison vs nb1001 honest cross-fit
    # =================================================================
    delta = pooled_rae - NB1001_CROSSFIT
    print("\n" + "-" * 78)
    print("COMPARISON vs nb1001 (honest cross-fit 0.5994, 2-way + stretch)")
    print("-" * 78)
    print(f"   nb1001 honest cross-fit (253)   = {NB1001_CROSSFIT:.4f}")
    print(f"   nb1010 honest cross-fit (253)   = {pooled_rae:.4f}")
    print(f"   delta (nb1010 - nb1001)         = {delta:+.4f}")
    if delta < -0.005:
        verdict = "PH_PSEUDO_ADD_VALUE"
        print("   verdict                         = 4-WAY BEATS nb1001  "
              "(PH + pseudo orthogonal)")
    elif delta < 0.005:
        verdict = "TIE"
        print("   verdict                         = TIE  "
              "(4-way no measurable gain)")
    else:
        verdict = "PH_PSEUDO_HURT"
        print("   verdict                         = WORSE  "
              "(extra anchors hurt cross-fit)")

    # =================================================================
    # Deploy: refit (w, s) on all 253
    # =================================================================
    print("\n" + "-" * 78)
    print("DEPLOY  (refit on all 253)")
    print("-" * 78)
    w_deploy = slsqp_simplex(P_unb, y_unb)
    blend_unb_all = P_unb @ w_deploy
    mu_deploy = float(blend_unb_all.mean())
    s_deploy, _ = best_stretch_on(blend_unb_all, y_unb, mu_deploy)
    in_rae_final = float(rae(y_unb,
                              mu_deploy + s_deploy
                              * (blend_unb_all - mu_deploy)))
    w_str = ", ".join(f"{disp}={w:.4f}"
                       for (disp, _, _), w in zip(CANDIDATES, w_deploy))
    print(f"   deploy weights      = {w_str}")
    print(f"   deploy mu (blend)   = {mu_deploy:.4f}")
    print(f"   deploy s            = {s_deploy:.2f}")
    print(f"   in-sample RAE (253) = {in_rae_final:.4f}  "
          "(overfit lower bound)")

    # Apply to all 513
    blend_513 = preds_513 @ w_deploy
    deploy_513 = (mu_deploy + s_deploy
                  * (blend_513 - mu_deploy)).astype(np.float32)
    print(f"   te(513) mean/std    = "
          f"{deploy_513.mean():.3f} / {deploy_513.std():.3f}")

    # =================================================================
    # Save
    # =================================================================
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy_513)
    plain = SUBMISSIONS / f"{TAG}_cf4way_with_ph.csv"
    pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te_names,
        "pEC50": deploy_513,
    }).to_csv(plain, index=False)
    print(f"\n[save] te_{TAG}.npy")
    print(f"[save] {plain}")

    deploy_weights = [
        {"name": disp, "w": float(w)}
        for (disp, _, _), w in zip(CANDIDATES, w_deploy)
    ]

    summary = {
        "tag": TAG,
        "candidates": [c[0] for c in CANDIDATES],
        "indiv_in_rae": indiv_rae,
        "stretch_grid": STRETCH_GRID,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "fold_results": fold_rows,
        "pooled_cv_rae_253": pooled_rae,
        "nb1001_crossfit_rae": NB1001_CROSSFIT,
        "delta_vs_nb1001": delta,
        "verdict": verdict,
        "deploy_weights": deploy_weights,
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
    print(f"   pool                       = {[c[0] for c in CANDIDATES]}")
    print(f"   honest cross-fit RAE (253) = {pooled_rae:.4f}")
    print(f"   nb1001 cross-fit           = {NB1001_CROSSFIT:.4f}")
    print(f"   delta                      = {delta:+.4f}")
    print(f"   verdict                    = {verdict}")
    print(f"   deploy s                   = {s_deploy:.2f}")
    print(f"   wall                       = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("pooled_cv_rae_253", "delta_vs_nb1001", "verdict",
              "deploy_weights", "deploy_s",
              "in_sample_rae_overfit_bound", "plain_submission"):
        print(f"  {k}: {res.get(k)}")
