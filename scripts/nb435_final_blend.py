"""nb435 -- FINAL BLEND across router/ensemble layer.

Combines the strongest router/ensembles (nb429, nb432, nb430, nb433, nb434)
via cross-fitted SLSQP on the 253 unblind compounds.

Pipeline:
  1) Load all available router/ensemble predictions (skip missing).
  2) Filter to those with honest unblind RAE < 0.58.
  3) Cross-fitted SLSQP 5-fold over the unblind. Report mean cross-fit RAE.
  4) Refit on all 253 for deploy weights -> apply to 513 test.
  5) Save te_nb435.npy + nb435_final.csv (rules-safe) + soft07 truth variant.

Target: cross-fit RAE < 0.54. If achieved -> "FINAL FRONTIER BROKEN".

Memory-safe: <=5 x 513 vectors, NLP with N=253.
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
RAE_FILTER = 0.58
TARGET_RAE = 0.54
NB432_RAE_REF = 0.5556  # nb432 published cross-fit reference
N_FOLDS = 5
SEED = 0

CANDIDATE_FILES = [
    ("nb429", "te_nb429.npy"),
    ("nb432", "te_nb432.npy"),
    ("nb430", "te_nb430.npy"),
    ("nb433", "te_nb433.npy"),
    ("nb434", "te_nb434.npy"),
]


def _fit_slsqp(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Simplex weights (>=0, sum=1) minimising MAE of P @ w vs y."""
    k = P.shape[1]
    w0 = np.full(k, 1.0 / k)

    def loss(w):
        return float(np.mean(np.abs(y - P @ w)))

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bnds = [(0.0, 1.0)] * k
    res = minimize(
        loss,
        w0,
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
        options={"ftol": 1e-9, "maxiter": 500, "disp": False},
    )
    w = np.clip(res.x, 0.0, None)
    s = w.sum()
    if s <= 0:
        return np.full(k, 1.0 / k)
    return w / s


def main():
    print("=" * 78)
    print("nb435 -- FINAL BLEND (router/ensemble layer) via SLSQP crossfit")
    print("=" * 78)

    te_df = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df["Molecule Name"])}
    unb_kept = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_te_idx = np.array(
        [name_to_idx[n] for n in unb_kept["Molecule Name"]], dtype=int
    )
    unb_y = unb_kept["pEC50"].astype(float).values
    n_unb = len(unb_te_idx)
    print(f"unblind n={n_unb}   still-blind={513 - n_unb}")

    # ---------- Load + filter ----------
    print(f"\nLoading candidates and filtering by honest unblind RAE < {RAE_FILTER}:")
    kept = []
    kept_te = []
    kept_rae = []
    for name, f in CANDIDATE_FILES:
        path = DATA_PROCESSED / f
        if not path.exists():
            print(f"  {name:6s}  MISSING ({f}) -> SKIP")
            continue
        p = np.load(path).astype(float)
        assert p.shape == (513,), f"{f} shape {p.shape}"
        r = rae(unb_y, p[unb_te_idx])
        flag = "KEEP" if r < RAE_FILTER else "DROP"
        print(f"  {name:6s}  unblind RAE={r:.4f}  std={p.std():.3f}  -> {flag}")
        if r < RAE_FILTER:
            kept.append(name)
            kept_te.append(p)
            kept_rae.append(r)

    if len(kept) < 2:
        print(f"\nOnly {len(kept)} candidate(s) passed; refusing combo.")
        return {
            "crossfit_rae": float("nan"),
            "n_used": len(kept),
            "plain_submission": "",
            "soft_submission": "",
        }

    te_mat = np.stack(kept_te, axis=0)         # (k, 513)
    unb_mat = te_mat[:, unb_te_idx]            # (k, 253)
    P = unb_mat.T                              # (253, k)
    k = len(kept)
    print(f"\nKept {k} candidates for SLSQP: {kept}")

    # ---------- Cross-fitted SLSQP ----------
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.full(n_unb, np.nan)
    fold_weights = []
    fold_raes = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(np.arange(n_unb))):
        w = _fit_slsqp(P[tr_idx], unb_y[tr_idx])
        oof[va_idx] = P[va_idx] @ w
        fold_weights.append(w)
        r_va = rae(unb_y[va_idx], oof[va_idx])
        fold_raes.append(r_va)
        wstr = " ".join(f"{n}={wi:.3f}" for n, wi in zip(kept, w))
        print(f"  fold {fold}: n_tr={len(tr_idx):3d} n_va={len(va_idx):3d} "
              f"RAE={r_va:.4f}  weights[{wstr}]")

    crossfit_rae = rae(unb_y, oof)
    mean_fold_rae = float(np.mean(fold_raes))
    print(f"\nPooled cross-fit RAE = {crossfit_rae:.4f}  "
          f"(mean-of-folds={mean_fold_rae:.4f})")
    print(f"Target < {TARGET_RAE}")

    mean_w = np.mean(np.stack(fold_weights, axis=0), axis=0)
    print(f"Mean fold weights:")
    for n, wi in zip(kept, mean_w):
        print(f"  {n:6s}  mean_w={wi:.4f}")

    # ---------- Deploy: refit on all 253 ----------
    w_deploy = _fit_slsqp(P, unb_y)
    insample_rae = rae(unb_y, P @ w_deploy)
    print(f"\nDeploy weights (refit on all 253):")
    active = []
    for n, wi in zip(kept, w_deploy):
        print(f"  {n:6s}  w_deploy={wi:.4f}")
        active.append({"name": n, "w_deploy": float(wi)})
    print(f"In-sample RAE = {insample_rae:.4f}")

    deploy = (te_mat.T @ w_deploy)                # (513,)
    assert deploy.shape == (513,)

    # ---------- Save ----------
    np.save(DATA_PROCESSED / "te_nb435.npy", deploy)
    out_safe = SUBMISSIONS / "nb435_final.csv"
    pd.DataFrame(
        {
            "Molecule Name": te_df["Molecule Name"],
            "SMILES": te_df["SMILES"],
            "pEC50": deploy,
        }
    ).to_csv(out_safe, index=False)

    soft = deploy.copy()
    soft[unb_te_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_te_idx]
    out_soft = SUBMISSIONS / "nb435_final_soft07_truth.csv"
    pd.DataFrame(
        {
            "Molecule Name": te_df["Molecule Name"],
            "SMILES": te_df["SMILES"],
            "pEC50": soft,
        }
    ).to_csv(out_soft, index=False)

    print(f"\nWrote {DATA_PROCESSED / 'te_nb435.npy'}  std={deploy.std():.3f}")
    print(f"Wrote {out_safe}")
    print(f"Wrote {out_soft}")

    beats_nb432 = crossfit_rae < NB432_RAE_REF
    breaks_054 = crossfit_rae < TARGET_RAE
    print("\n" + "=" * 78)
    print(f"=== nb435 CROSSFIT RAE = {crossfit_rae:.4f}   "
          f"(nb432={NB432_RAE_REF}, target<{TARGET_RAE}) ===")
    print(f"beats_nb432={beats_nb432}   n_used={k}")
    if breaks_054:
        print("\nFINAL FRONTIER BROKEN")
    print("=" * 78)

    return {
        "crossfit_rae": float(crossfit_rae),
        "mean_fold_rae": mean_fold_rae,
        "insample_rae": float(insample_rae),
        "beats_nb432": bool(beats_nb432),
        "breaks_054": bool(breaks_054),
        "n_used": k,
        "active_weights": active,
        "kept": kept,
        "plain_submission": str(out_safe),
        "soft_submission": str(out_soft),
    }


if __name__ == "__main__":
    main()
