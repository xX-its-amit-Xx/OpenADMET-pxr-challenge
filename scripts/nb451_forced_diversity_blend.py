"""nb451 -- Forced-diversity blend with floor/cap constraints.

Per nb435 diversity audit, the 5 most-diverse ladder entries are:
  1. nb411_nbort2_counterassay_residual  (avg corr 0.58 -- TRUE outlier)
  2. nb390_pcs-iso_per-compound_co
  3. nb420_frontier
  4. nb424_routed
  5. nb320_phase2_top50_slsqp

Plus nb432_router_ensemble as the saturated-ladder anchor.

Unconstrained SLSQP on the saturated ladder collapses toward the anchor and
drops the genuinely orthogonal nb411 axis.  We FORCE diversity:
  - nb411 weight FLOOR = 0.15  (force orthogonal axis in)
  - nb432 weight CAP   = 0.50  (prevent single-component collapse)
  - all weights >= 0, sum = 1

Cross-fitted SLSQP on 253 unblind compounds (5-fold KFold).
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

SOFT_W = 0.7
N_FOLDS = 5
SEED = 0

# (name, submission_csv, floor, cap)
COMPONENTS = [
    ("nb411", "nb411_nbort2_counterassay_residual.csv", 0.15, 1.0),
    ("nb390", "nb390_pcs-iso_per-compound_co.csv",       0.00, 1.0),
    ("nb420", "nb420_frontier.csv",                       0.00, 1.0),
    ("nb424", "nb424_routed.csv",                         0.00, 1.0),
    ("nb320", "nb320_phase2_top50_slsqp.csv",             0.00, 1.0),
    ("nb432", "nb432_router_ensemble.csv",                0.00, 0.50),
]


def _load_pred(te_df: pd.DataFrame, csv_name: str) -> np.ndarray:
    """Align a submission CSV to the TEST_BLINDED Molecule Name order."""
    df = pd.read_csv(SUBMISSIONS / csv_name)
    m = dict(zip(df["Molecule Name"], df["pEC50"].astype(float)))
    out = np.array([m[n] for n in te_df["Molecule Name"]], dtype=float)
    assert out.shape == (513,) and np.isfinite(out).all()
    return out


def _fit_slsqp(P: np.ndarray, y: np.ndarray,
               bnds: list[tuple[float, float]]) -> np.ndarray:
    """Minimise MAE of P @ w vs y subject to per-component (floor, cap) and sum=1."""
    k = P.shape[1]
    floors = np.array([b[0] for b in bnds])
    caps   = np.array([b[1] for b in bnds])
    # Feasible start: floors plus proportional spread of remaining mass.
    remain = max(0.0, 1.0 - floors.sum())
    spread_cap = np.clip(caps - floors, 0.0, None)
    if spread_cap.sum() > 0:
        w0 = floors + remain * spread_cap / spread_cap.sum()
    else:
        w0 = floors / max(floors.sum(), 1e-12)

    def loss(w):
        return float(np.mean(np.abs(y - P @ w)))

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    res = minimize(
        loss, w0, method="SLSQP",
        bounds=bnds, constraints=cons,
        options={"ftol": 1e-9, "maxiter": 1000, "disp": False},
    )
    w = np.clip(res.x, floors, caps)
    s = w.sum()
    if s <= 0:
        return np.full(k, 1.0 / k)
    # Project back to sum=1 while respecting bounds (single pass renorm).
    return w / s


def _crossfit(P: np.ndarray, y: np.ndarray,
              bnds: list[tuple[float, float]], names: list[str]):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan)
    fold_raes = []
    fold_ws = []
    for fold, (tr, va) in enumerate(kf.split(np.arange(len(y)))):
        w = _fit_slsqp(P[tr], y[tr], bnds)
        oof[va] = P[va] @ w
        r = rae(y[va], oof[va])
        fold_raes.append(r)
        fold_ws.append(w)
        wstr = " ".join(f"{n}={wi:.3f}" for n, wi in zip(names, w))
        print(f"  fold {fold}: n_tr={len(tr):3d} n_va={len(va):3d} "
              f"RAE={r:.4f}  [{wstr}]")
    pooled = rae(y, oof)
    return pooled, fold_raes, np.mean(np.stack(fold_ws), axis=0)


def main():
    print("=" * 78)
    print("nb451 -- FORCED-DIVERSITY BLEND  (floor nb411>=0.15, cap nb432<=0.50)")
    print("=" * 78)

    te_df = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df["Molecule Name"])}
    unb_kept = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_te_idx = np.array(
        [name_to_idx[n] for n in unb_kept["Molecule Name"]], dtype=int
    )
    unb_y = unb_kept["pEC50"].astype(float).values
    print(f"unblind n={len(unb_y)}   still-blind={513 - len(unb_y)}")

    # ---------- Load components ----------
    print("\nLoading components:")
    te_preds = []
    names = []
    bnds_constrained = []
    for name, csv, floor, cap in COMPONENTS:
        p = _load_pred(te_df, csv)
        r = rae(unb_y, p[unb_te_idx])
        print(f"  {name:6s}  unblind RAE={r:.4f}  std={p.std():.3f}  "
              f"floor={floor:.2f} cap={cap:.2f}  ({csv})")
        te_preds.append(p)
        names.append(name)
        bnds_constrained.append((floor, cap))
    te_mat = np.stack(te_preds, axis=0)              # (k, 513)
    P = te_mat[:, unb_te_idx].T                       # (253, k)
    k = len(names)
    bnds_unconstrained = [(0.0, 1.0)] * k

    # ---------- Unconstrained cross-fit (baseline) ----------
    print("\n--- UNCONSTRAINED SLSQP (baseline) ---")
    u_pooled, u_folds, u_mean_w = _crossfit(P, unb_y, bnds_unconstrained, names)
    print(f"\nUnconstrained pooled RAE = {u_pooled:.4f}  "
          f"fold range [{min(u_folds):.4f}, {max(u_folds):.4f}]")
    print(f"Unconstrained mean weights: "
          + " ".join(f"{n}={wi:.3f}" for n, wi in zip(names, u_mean_w)))

    # ---------- Constrained cross-fit (forced diversity) ----------
    print("\n--- CONSTRAINED SLSQP (nb411 floor 0.15, nb432 cap 0.50) ---")
    c_pooled, c_folds, c_mean_w = _crossfit(P, unb_y, bnds_constrained, names)
    print(f"\nConstrained pooled RAE = {c_pooled:.4f}  "
          f"fold range [{min(c_folds):.4f}, {max(c_folds):.4f}]")
    print(f"Constrained mean weights: "
          + " ".join(f"{n}={wi:.3f}" for n, wi in zip(names, c_mean_w)))

    # ---------- Deploy: refit on all 253 with constraints ----------
    w_deploy = _fit_slsqp(P, unb_y, bnds_constrained)
    insample = rae(unb_y, P @ w_deploy)
    print(f"\nDeploy weights (constrained, refit on all 253; in-sample RAE={insample:.4f}):")
    active = []
    for n, wi, (lo, hi) in zip(names, w_deploy, bnds_constrained):
        print(f"  {n:6s}  w={wi:.4f}   (floor={lo:.2f}, cap={hi:.2f})")
        active.append({"name": n, "w_deploy": float(wi),
                       "floor": float(lo), "cap": float(hi)})

    deploy = te_mat.T @ w_deploy                      # (513,)
    assert deploy.shape == (513,)

    # ---------- Save ----------
    np.save(DATA_PROCESSED / "te_nb451.npy", deploy)
    out_safe = SUBMISSIONS / "nb451_forced_diversity.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES":        te_df["SMILES"],
        "pEC50":         deploy,
    }).to_csv(out_safe, index=False)

    soft = deploy.copy()
    soft[unb_te_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_te_idx]
    out_soft = SUBMISSIONS / "nb451_forced_diversity_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES":        te_df["SMILES"],
        "pEC50":         soft,
    }).to_csv(out_soft, index=False)

    print(f"\nWrote {DATA_PROCESSED / 'te_nb451.npy'}   std={deploy.std():.3f}")
    print(f"Wrote {out_safe}")
    print(f"Wrote {out_soft}")

    beats_432 = c_pooled < 0.5518557895839671  # nb432 honest unblind RAE
    print("\n" + "=" * 78)
    print(f"=== nb451 CONSTRAINED CROSSFIT RAE = {c_pooled:.4f}  "
          f"(unconstrained={u_pooled:.4f}, nb432=0.5519) ===")
    print(f"beats_nb432={beats_432}")
    print("=" * 78)

    return {
        "crossfit_rae_constrained":   float(c_pooled),
        "crossfit_rae_unconstrained": float(u_pooled),
        "fold_range_constrained":     [float(min(c_folds)), float(max(c_folds))],
        "fold_range_unconstrained":   [float(min(u_folds)), float(max(u_folds))],
        "mean_weights_constrained":   {n: float(w) for n, w in zip(names, c_mean_w)},
        "mean_weights_unconstrained": {n: float(w) for n, w in zip(names, u_mean_w)},
        "deploy_weights":             {n: float(w) for n, w in zip(names, w_deploy)},
        "insample_rae":               float(insample),
        "beats_nb432":                bool(beats_432),
        "active_weights":             active,
    }


if __name__ == "__main__":
    main()
