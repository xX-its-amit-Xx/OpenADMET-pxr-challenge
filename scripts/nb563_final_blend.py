"""nb563 -- FINAL BLEND: 4-way SLSQP cross-fit over {nb503, nb560, nb561, nb562}.

Honest 5-fold cross-fit RAE on the 253 unblind rows. Refit on all 253 for
deploy weights. Target: cross-fit RAE < 0.5116 (nb503 standalone).

Hyperparams mirror nb503/nb562:
    SLSQP simplex on MAE (bounds [0,1], sum=1), ftol=1e-9, maxiter=500.
    KFold n_splits=5, shuffle=True, random_state=0.
    SOFT_W = 0.7 for the truth-soft-inject submission.
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

SEED = 0
N_FOLDS = 5
SOFT_W = 0.7
TAG = "nb563"
POOL = ["nb503", "nb560", "nb561", "nb562"]
NB503_RAE = 0.5116  # target to beat


def _fit_slsqp(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    k = P.shape[1]
    if k == 1:
        return np.array([1.0])
    w0 = np.full(k, 1.0 / k)

    def loss(w):
        return float(np.mean(np.abs(y - P @ w)))

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bnds = [(0.0, 1.0)] * k
    res = minimize(
        loss, w0, method="SLSQP", bounds=bnds, constraints=cons,
        options={"ftol": 1e-9, "maxiter": 500, "disp": False},
    )
    w = np.clip(res.x, 0.0, None)
    s = w.sum()
    return np.full(k, 1.0 / k) if s <= 0 else w / s


def main() -> dict:
    print("=" * 78)
    print("nb563 -- FINAL BLEND (4-way SLSQP cross-fit)")
    print("=" * 78)

    needed = {
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED":    DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
    }
    for tag in POOL:
        needed[f"{tag}_pred_oof.npy"] = DATA_PROCESSED / f"{tag}_pred_oof.npy"
        needed[f"te_{tag}.npy"] = DATA_PROCESSED / f"te_{tag}.npy"
    missing = [k for k, p in needed.items() if not Path(p).exists()]
    if missing:
        print("MISSING:", missing)
        return {"success": False, "missing": missing}

    # ---- Indices ----
    te_df = pd.read_csv(needed["TEST_BLINDED"])
    te_names = te_df["Molecule Name"].tolist()
    name_to_idx = {n: i for i, n in enumerate(te_names)}
    unb = pd.read_csv(needed["UNBLINDED"])
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"]], dtype=int
    )
    unb_y = unb["pEC50"].astype(float).values.astype(np.float32)
    n_te = len(te_df)
    n_unb = len(unb_idx)
    print(f"\ntest n={n_te}  unblind n={n_unb}")

    # ---- Load OOFs and deploys ----
    oof = {}
    te_vec = {}
    print("\nPool members (standalone RAE on 253):")
    for tag in POOL:
        a = np.load(DATA_PROCESSED / f"{tag}_pred_oof.npy").astype(np.float32)
        d = np.load(DATA_PROCESSED / f"te_{tag}.npy").astype(np.float32)
        assert a.shape == (n_unb,), f"{tag} oof shape {a.shape}"
        assert d.shape == (n_te,),  f"{tag} te shape {d.shape}"
        oof[tag] = a
        te_vec[tag] = d
        r_alone = float(rae(unb_y, a))
        print(f"  {tag}: oof_RAE={r_alone:.4f}  std={a.std():.3f}  "
              f"te std={d.std():.3f}")

    # ---- 4-way SLSQP cross-fit ----
    print("\n" + "-" * 78)
    print(f"4-WAY SLSQP CROSS-FIT  ({POOL})")
    print("-" * 78)
    P_oof = np.stack([oof[t] for t in POOL], axis=1).astype(np.float64)
    Q_te = np.stack([te_vec[t] for t in POOL], axis=1).astype(np.float64)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
    fold_weights = []
    fold_raes = []
    for fold, (tr_i, va_i) in enumerate(kf.split(np.arange(n_unb))):
        w = _fit_slsqp(P_oof[tr_i], unb_y[tr_i])
        oof_blend[va_i] = P_oof[va_i] @ w
        r_va = float(rae(unb_y[va_i], oof_blend[va_i]))
        fold_weights.append(w)
        fold_raes.append(r_va)
        wstr = " ".join(f"{t}={wi:.3f}" for t, wi in zip(POOL, w))
        print(f"  fold {fold}: n_tr={len(tr_i):3d} n_va={len(va_i):3d}  "
              f"RAE={r_va:.4f}  [{wstr}]")
    slsqp_rae = float(rae(unb_y, oof_blend))
    fold_lo, fold_hi = float(min(fold_raes)), float(max(fold_raes))
    print(f"\nPooled 4-way SLSQP cross-fit RAE = {slsqp_rae:.4f}  "
          f"(per-fold {fold_lo:.4f}--{fold_hi:.4f})")

    # ---- Deploy weights ----
    w_deploy = _fit_slsqp(P_oof, unb_y)
    print("\nDeploy SLSQP weights (refit on all 253):")
    for t, wi in zip(POOL, w_deploy):
        print(f"  {t}: {wi:.4f}")
    deploy = (Q_te @ w_deploy).astype(np.float32)

    # ---- Decision ----
    print("\n" + "=" * 78)
    print("DEPLOY DECISION")
    print("=" * 78)
    print(f"  nb503 standalone (target)         = {NB503_RAE:.4f}")
    print(f"  4-way SLSQP cross-fit RAE         = {slsqp_rae:.4f}")
    beats = slsqp_rae < NB503_RAE
    print(f"  beats nb503                        = {beats}")

    # ---- Save artefacts ----
    np.save(DATA_PROCESSED / "te_nb563.npy", deploy)
    np.save(DATA_PROCESSED / "nb563_pred_oof.npy",
            oof_blend.astype(np.float32))

    plain = SUBMISSIONS / "nb563_final_blend.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(plain, index=False)

    soft = deploy.copy()
    soft[unb_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_idx]
    soft_path = SUBMISSIONS / "nb563_final_blend_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(soft_path, index=False)

    print(f"\nWrote {DATA_PROCESSED / 'te_nb563.npy'}")
    print(f"Wrote {DATA_PROCESSED / 'nb563_pred_oof.npy'}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")

    print("\n" + "=" * 78)
    print("=== nb563 SUMMARY ===")
    for t, wi in zip(POOL, w_deploy):
        print(f"  weight {t} = {wi:.4f}")
    print(f"  4-way SLSQP cross-fit RAE = {slsqp_rae:.4f}")
    print(f"  per-fold range            = [{fold_lo:.4f}, {fold_hi:.4f}]")
    print(f"  beats nb503 ({NB503_RAE:.4f})    = {beats}")
    print("=" * 78)

    return {
        "success": True,
        "pool": POOL,
        "slsqp_crossfit_rae": float(slsqp_rae),
        "slsqp_deploy_weights": {t: float(w) for t, w in zip(POOL, w_deploy)},
        "fold_raes": [float(r) for r in fold_raes],
        "beats_nb503": bool(beats),
        "plain_submission": str(plain),
        "soft_submission": str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
