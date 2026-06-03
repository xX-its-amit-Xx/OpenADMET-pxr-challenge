"""nb982 -- Best-blend SLSQP over top-3 PRE-unblind winners.

Pools the three PRE-unblind candidates with in_RAE < 0.70:
    - te_chemprop_aux.npy             (in_RAE 0.6216)
    - te_nb901_nr_multitask  (CSV)    (in_RAE 0.6765)
    - te_nb972_long_train.npy         (in_RAE 0.6898)

5-fold cross-fit SLSQP (positive weights, sum=1) is fit on the 253 unblind
labels. The pooled out-of-fold prediction yields the honest cross-fit RAE.
A single SLSQP fit on all 253 unblind rows produces the DEPLOY weights, which
are applied to the 513 te files to produce te_nb982.npy and a submission CSV.

Hypothesis: nb901 (NR multi-task aux) and nb972 (long-train) approach the
OOD novel-scaffold tail differently than chemprop_aux. SLSQP may extract
orthogonal value if residual correlation < 1.

Artifacts: data/processed/te_nb982.npy, data/processed/nb982_summary.json
Submission: submissions/nb982_best_blend.csv
"""
from __future__ import annotations
import json, os, sys, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

CANDIDATES = ["chemprop_aux", "nb901_nr_multitask", "nb972_long_train"]


def load_te(name: str, te_names: np.ndarray) -> np.ndarray:
    """Load a 513-vector candidate; reconstruct from submission CSV if no .npy."""
    npy = DATA_PROCESSED / f"te_{name}.npy"
    if npy.exists():
        return np.load(npy).astype(np.float64)
    sub = pd.read_csv(SUBMISSIONS / f"{name}.csv")
    assert (sub["Molecule Name"].values == te_names).all(), \
        f"{name}: submission row order does not match test order"
    return sub["pEC50"].values.astype(np.float64)


def slsqp_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Positive weights summing to 1 that minimize SSE of (P @ w - y)."""
    K = P.shape[1]
    w0 = np.full(K, 1.0 / K)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * K
    res = minimize(
        lambda w: float(np.sum((P @ w - y) ** 2)),
        w0, method="SLSQP", bounds=bnds, constraints=cons,
        options={"ftol": 1e-10, "maxiter": 500},
    )
    return res.x


def main():
    t_start = time.time()
    print("=== nb982: SLSQP blend of top-3 PRE-unblind winners ===")
    te = load_test()
    te_names = te["name"].values

    # --- Load 513-vectors ---
    preds_513 = np.column_stack([load_te(c, te_names) for c in CANDIDATES])
    print(f"[load] preds_513 shape = {preds_513.shape}")

    idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    P_unb = preds_513[idx]
    print(f"[load] unblind preds shape = {P_unb.shape}, y shape = {y_unb.shape}")

    # --- Individual in_RAEs (sanity) ---
    print("\n[indiv] in_RAE on 253 unblind:")
    for j, c in enumerate(CANDIDATES):
        print(f"   {c:30s}: {rae(y_unb, P_unb[:, j]):.4f}")

    # --- Residual correlations ---
    print("\n[corr] residual correlation matrix:")
    R = P_unb - y_unb[:, None]
    cc = np.corrcoef(R.T)
    for i, ci in enumerate(CANDIDATES):
        row = "  ".join(f"{cc[i, j]:+.3f}" for j in range(len(CANDIDATES)))
        print(f"   {ci:30s} {row}")

    # --- 5-fold cross-fit SLSQP ---
    print("\n[cv] 5-fold cross-fit SLSQP on 253 unblind:")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros_like(y_unb)
    fold_weights = []
    for k, (tr_loc, va_loc) in enumerate(kf.split(np.arange(len(y_unb)))):
        w_k = slsqp_weights(P_unb[tr_loc], y_unb[tr_loc])
        oof[va_loc] = P_unb[va_loc] @ w_k
        rae_k = rae(y_unb[va_loc], oof[va_loc])
        fold_weights.append(w_k.tolist())
        wstr = ", ".join(f"{w:.3f}" for w in w_k)
        print(f"   fold {k}: w=[{wstr}]  val_RAE={rae_k:.4f}")

    pooled_rae = float(rae(y_unb, oof))
    print(f"\n[cv] pooled 5-fold cross-fit RAE = {pooled_rae:.4f}")

    # --- DEPLOY weights (single SLSQP on all 253) ---
    w_deploy = slsqp_weights(P_unb, y_unb)
    print("\n[deploy] weights from full 253:")
    for c, w in zip(CANDIDATES, w_deploy):
        print(f"   {c:30s}: {w:.4f}")
    in_rae_deploy = float(rae(y_unb, P_unb @ w_deploy))
    print(f"[deploy] in-sample RAE (overfit lower bound) = {in_rae_deploy:.4f}")

    # --- Apply to 513 test set ---
    te_blend = (preds_513 @ w_deploy).astype(np.float32)
    np.save(DATA_PROCESSED / "te_nb982.npy", te_blend)
    print(f"[save] te_nb982.npy shape = {te_blend.shape}, "
          f"mean={te_blend.mean():.3f} std={te_blend.std():.3f}")

    sub = pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te_names,
        "pEC50": te_blend,
    })
    out_csv = SUBMISSIONS / "nb982_best_blend.csv"
    sub.to_csv(out_csv, index=False)
    print(f"[sub] wrote {out_csv}  ({len(sub)} rows)")

    # --- Summary ---
    summary = {
        "candidates": CANDIDATES,
        "indiv_in_rae": {c: float(rae(y_unb, P_unb[:, j]))
                         for j, c in enumerate(CANDIDATES)},
        "residual_corr": cc.tolist(),
        "fold_weights": fold_weights,
        "pooled_cv_rae": pooled_rae,
        "deploy_weights": dict(zip(CANDIDATES, w_deploy.tolist())),
        "in_sample_rae_overfit_bound": in_rae_deploy,
        "wall_sec": round(time.time() - t_start, 2),
    }
    with open(DATA_PROCESSED / "nb982_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDONE  pooled_cv_RAE={pooled_rae:.4f}  "
          f"deploy_w={dict(zip(CANDIDATES, [round(w, 3) for w in w_deploy]))}  "
          f"wall={time.time()-t_start:.1f}s")
    return summary


if __name__ == "__main__":
    main()
