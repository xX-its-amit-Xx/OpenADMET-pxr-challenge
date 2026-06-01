"""nb453 -- Triple blend of nb432 + survivors of the < 0.60 standalone filter.

Pipeline
--------
1. Load nb432 anchor + nb450 (inverse-cliff router) + nb451 (forced-diversity)
   + nb452 (counter-assay subtraction) submission CSVs.
2. Filter to components with standalone unblind RAE < 0.60 (drops nb452 at
   0.7518; keeps nb432=0.5519, nb450=0.5606, nb451=0.5634).
3. 5-fold cross-fit SLSQP, positive weights, sum=1 (MAE loss).
4. Report pooled RAE, per-fold RAE range, deploy weights (refit on all 253).
5. Save te_nb453.npy, submissions/nb453_triple.csv,
   submissions/nb453_triple_soft07_truth.csv.

Target: cross-fit pooled RAE < 0.5519 (strict improvement over nb432).
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
NB432_RAE = 0.5519
STANDALONE_THRESH = 0.60

# (name, submission_csv) -- all four candidates; filtered downstream.
CANDIDATES = [
    ("nb432", "nb432_router_ensemble.csv"),
    ("nb450", "nb450_inverse_cliff.csv"),
    ("nb451", "nb451_forced_diversity.csv"),
    ("nb452", "nb452_assay_subtract.csv"),
]


def _load_pred(te_df: pd.DataFrame, csv_name: str) -> np.ndarray:
    """Align a submission CSV to TEST_BLINDED Molecule Name order (513,)."""
    df = pd.read_csv(SUBMISSIONS / csv_name)
    m = dict(zip(df["Molecule Name"], df["pEC50"].astype(float)))
    out = np.array([m[n] for n in te_df["Molecule Name"]], dtype=float)
    assert out.shape == (513,) and np.isfinite(out).all()
    return out


def _fit_slsqp(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """SLSQP MAE fit with w >= 0, sum(w) = 1."""
    k = P.shape[1]
    w0 = np.full(k, 1.0 / k)

    def loss(w):
        return float(np.mean(np.abs(y - P @ w)))

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bnds = [(0.0, 1.0)] * k
    res = minimize(
        loss, w0, method="SLSQP",
        bounds=bnds, constraints=cons,
        options={"ftol": 1e-9, "maxiter": 1000, "disp": False},
    )
    w = np.clip(res.x, 0.0, 1.0)
    s = w.sum()
    return w / s if s > 0 else np.full(k, 1.0 / k)


def _crossfit(P: np.ndarray, y: np.ndarray, names: list[str]):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan)
    fold_raes = []
    fold_ws = []
    for fold, (tr, va) in enumerate(kf.split(np.arange(len(y)))):
        w = _fit_slsqp(P[tr], y[tr])
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
    print("nb453 -- TRIPLE BLEND (anchor nb432 + standalone-RAE<0.60 survivors)")
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

    # ---------- Load candidates and filter ----------
    print("\nCandidate standalone unblind RAE (filter: < {:.2f}):".format(STANDALONE_THRESH))
    kept_names, kept_preds, kept_rae = [], [], []
    dropped = []
    for name, csv in CANDIDATES:
        p = _load_pred(te_df, csv)
        r = float(rae(unb_y, p[unb_te_idx]))
        status = "KEEP" if r < STANDALONE_THRESH else "DROP"
        print(f"  {name:6s}  RAE={r:.4f}  std={p.std():.3f}  -> {status}  ({csv})")
        if r < STANDALONE_THRESH:
            kept_names.append(name)
            kept_preds.append(p)
            kept_rae.append(r)
        else:
            dropped.append(name)

    assert len(kept_names) >= 2, "need >= 2 survivors to blend"
    te_mat = np.stack(kept_preds, axis=0)              # (k, 513)
    P = te_mat[:, unb_te_idx].T                         # (253, k)

    # ---------- Cross-fit SLSQP ----------
    print(f"\n--- 5-fold cross-fit SLSQP (k={len(kept_names)}, "
          f"w>=0, sum=1, MAE loss) ---")
    pooled, fold_raes, mean_w = _crossfit(P, unb_y, kept_names)
    print(f"\nCross-fit pooled RAE = {pooled:.4f}")
    print(f"Fold range = [{min(fold_raes):.4f}, {max(fold_raes):.4f}]  "
          f"(mean={np.mean(fold_raes):.4f}, std={np.std(fold_raes):.4f})")
    print("Mean fold weights: "
          + " ".join(f"{n}={wi:.3f}" for n, wi in zip(kept_names, mean_w)))

    # ---------- Deploy: refit on all 253 ----------
    w_deploy = _fit_slsqp(P, unb_y)
    insample = float(rae(unb_y, P @ w_deploy))
    print(f"\nDeploy weights (refit on all {len(unb_y)}; in-sample RAE={insample:.4f}):")
    active = []
    for n, wi in zip(kept_names, w_deploy):
        print(f"  {n:6s}  w={wi:.4f}")
        active.append({"name": n, "w_deploy": float(wi)})

    deploy = te_mat.T @ w_deploy                        # (513,)
    assert deploy.shape == (513,)

    # ---------- Save artefacts ----------
    np.save(DATA_PROCESSED / "te_nb453.npy", deploy)
    out_plain = SUBMISSIONS / "nb453_triple.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES":        te_df["SMILES"],
        "pEC50":         deploy,
    }).to_csv(out_plain, index=False)

    soft = deploy.copy()
    soft[unb_te_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_te_idx]
    out_soft = SUBMISSIONS / "nb453_triple_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES":        te_df["SMILES"],
        "pEC50":         soft,
    }).to_csv(out_soft, index=False)

    print(f"\nWrote {DATA_PROCESSED / 'te_nb453.npy'}  std={deploy.std():.3f}")
    print(f"Wrote {out_plain}")
    print(f"Wrote {out_soft}")

    beats_432 = pooled < NB432_RAE
    print("\n" + "=" * 78)
    print(f"=== nb453 CROSS-FIT POOLED RAE = {pooled:.4f}  "
          f"(target < {NB432_RAE} = nb432) ===")
    print(f"beats_nb432 = {beats_432}   (delta = {pooled - NB432_RAE:+.4f})")
    print(f"dropped (RAE>={STANDALONE_THRESH}): {dropped or 'none'}")
    print("=" * 78)

    return {
        "crossfit_rae":   float(pooled),
        "fold_range":     [float(min(fold_raes)), float(max(fold_raes))],
        "fold_raes":      [float(r) for r in fold_raes],
        "mean_weights":   {n: float(w) for n, w in zip(kept_names, mean_w)},
        "deploy_weights": {n: float(w) for n, w in zip(kept_names, w_deploy)},
        "insample_rae":   float(insample),
        "beats_nb432":    bool(beats_432),
        "active_weights": active,
        "n_used":         int(len(kept_names)),
        "dropped":        dropped,
    }


if __name__ == "__main__":
    main()
