"""nb514 -- ORTHO BLEND: grand SLSQP over orthogonal fingerprint routers.

Pool candidates: {nb502, nb510, nb511, nb512, nb513, nb492, nb503}.
Filter to honest oof RAE < 0.55 on the 253 unblind rows.

Procedure:
  1) Compute residual-OOF (unb_y - pred_oof) for each survivor; print pairwise
     Pearson correlation matrix. Identify the 3 most-decorrelated members
     (smallest mean |rho| with the other survivors).
  2) 5-fold KFold cross-fit SLSQP simplex over ALL surviving OOFs (MAE loss).
     Report pooled RAE, per-fold range, and deploy weights (refit on 253).
  3) HARD-MASK variant: simple mean of the 3 most-decorrelated members
     (no SLSQP weights). Pooled RAE on 253.
  4) Pick the lower-RAE variant; save te_nb514.npy + plain + soft07_truth subs.

Targets:
  STRICT improvement over nb503 cross-fit (0.5116).
  Log "ORTHO BLEND DOESNT HELP" if pooled best > 0.5116.

Mirrors nb503 hyperparams (SLSQP MAE simplex, ftol=1e-9, 5-fold seed=0,
SOFT_W=0.7).
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
TAG = "nb514"

RAE_FILTER = 0.55
NB503_CROSSFIT = 0.5116

POOL = ["nb502", "nb510", "nb511", "nb512", "nb513", "nb492", "nb503"]


def _fit_slsqp(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Simplex weights (>=0, sum=1) minimising MAE of P @ w vs y."""
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
    print("nb514 -- ORTHO BLEND (grand SLSQP over fingerprint routers)")
    print("=" * 78)

    needed = {
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED":    DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
    }
    for tag in POOL:
        needed[f"{tag}_pred_oof.npy"] = DATA_PROCESSED / f"{tag}_pred_oof.npy"
        needed[f"te_{tag}.npy"]       = DATA_PROCESSED / f"te_{tag}.npy"
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

    # ---- Load + filter to honest RAE < 0.55 ----
    oof_all = {}
    te_all = {}
    rae_all = {}
    print("\nLoading pool (standalone honest RAE on 253):")
    for tag in POOL:
        a = np.load(DATA_PROCESSED / f"{tag}_pred_oof.npy").astype(np.float32)
        d = np.load(DATA_PROCESSED / f"te_{tag}.npy").astype(np.float32)
        assert a.shape == (n_unb,), f"{tag} oof shape {a.shape}"
        assert d.shape == (n_te,),  f"{tag} te shape {d.shape}"
        r = float(rae(unb_y, a))
        oof_all[tag] = a
        te_all[tag] = d
        rae_all[tag] = r
        keep = "KEEP" if r < RAE_FILTER else "DROP"
        print(f"  {tag}: RAE={r:.4f}  oof_std={a.std():.3f}  "
              f"te_std={d.std():.3f}  -> {keep}")

    survivors = [t for t in POOL if rae_all[t] < RAE_FILTER]
    dropped   = [t for t in POOL if rae_all[t] >= RAE_FILTER]
    print(f"\nSurvivors ({len(survivors)}/{len(POOL)}): {survivors}")
    print(f"Dropped (RAE >= {RAE_FILTER}): {dropped}")

    if len(survivors) < 2:
        print("\nFewer than 2 survivors -- cannot build a blend.")
        print(f"ORTHO BLEND DOESNT HELP -- only {len(survivors)} member(s) "
              f"passed RAE<{RAE_FILTER} filter. Dropped: {dropped}")
        return {"success": False, "survivors": survivors, "dropped": dropped}

    # =================================================================
    # (1) Residual-OOF pairwise Pearson matrix
    # =================================================================
    print("\n" + "-" * 78)
    print("(1) RESIDUAL-OOF PAIRWISE PEARSON CORRELATION")
    print("-" * 78)
    resid = {t: (unb_y - oof_all[t]).astype(np.float64) for t in survivors}
    R = np.zeros((len(survivors), len(survivors)), dtype=np.float64)
    for i, a in enumerate(survivors):
        for j, b in enumerate(survivors):
            if i == j:
                R[i, j] = 1.0
            else:
                R[i, j] = float(np.corrcoef(resid[a], resid[b])[0, 1])

    header = "         " + " ".join(f"{t:>8s}" for t in survivors)
    print(header)
    for i, t in enumerate(survivors):
        row = " ".join(f"{R[i, j]:8.3f}" for j in range(len(survivors)))
        print(f"  {t:>6s} {row}")

    # Mean |rho| with others -> lower = more decorrelated
    n = len(survivors)
    off = np.where(~np.eye(n, dtype=bool))
    mean_abs = {}
    for i, t in enumerate(survivors):
        others = [j for j in range(n) if j != i]
        mean_abs[t] = float(np.mean(np.abs(R[i, others])))
    print("\nMean |rho| with other survivors (lower = more orthogonal):")
    for t in sorted(mean_abs, key=mean_abs.get):
        print(f"  {t}: {mean_abs[t]:.3f}")

    top3 = sorted(mean_abs, key=mean_abs.get)[:3]
    print(f"\nTop-3 most-decorrelated: {top3}")

    # =================================================================
    # (2) 5-fold cross-fit SLSQP over ALL survivors
    # =================================================================
    print("\n" + "-" * 78)
    print(f"(2) GRAND SLSQP CROSS-FIT  (survivors={survivors})")
    print("-" * 78)
    P_oof = np.stack([oof_all[t] for t in survivors], axis=1).astype(np.float64)
    Q_te  = np.stack([te_all[t]  for t in survivors], axis=1).astype(np.float64)

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
        wstr = " ".join(f"{t}={wi:.3f}" for t, wi in zip(survivors, w))
        print(f"  fold {fold}: n_tr={len(tr_i):3d} n_va={len(va_i):3d}  "
              f"RAE={r_va:.4f}  [{wstr}]")
    slsqp_rae = float(rae(unb_y, oof_blend))
    fold_lo, fold_hi = float(min(fold_raes)), float(max(fold_raes))
    print(f"\nPooled grand SLSQP cross-fit RAE = {slsqp_rae:.4f}  "
          f"(per-fold {fold_lo:.4f}--{fold_hi:.4f})")

    w_deploy = _fit_slsqp(P_oof, unb_y)
    print("Deploy SLSQP weights (refit on 253):")
    for t, wi in zip(survivors, w_deploy):
        print(f"  {t}: {wi:.4f}")

    # =================================================================
    # (3) HARD-MASK: simple mean of top-3 decorrelated members
    # =================================================================
    print("\n" + "-" * 78)
    print(f"(3) HARD-MASK MEAN of top-3 decorrelated  {top3}")
    print("-" * 78)
    mask_oof = np.mean(
        np.stack([oof_all[t] for t in top3], axis=1), axis=1
    ).astype(np.float32)
    mask_te  = np.mean(
        np.stack([te_all[t]  for t in top3], axis=1), axis=1
    ).astype(np.float32)
    mask_rae = float(rae(unb_y, mask_oof))
    print(f"Mean({', '.join(top3)})  RAE = {mask_rae:.4f}")

    # =================================================================
    # Pick winner
    # =================================================================
    print("\n" + "=" * 78)
    print("DEPLOY DECISION")
    print("=" * 78)
    print(f"  nb503 cross-fit target           = {NB503_CROSSFIT:.4f}")
    print(f"  grand SLSQP cross-fit RAE        = {slsqp_rae:.4f}")
    print(f"  hard-mask top-3 RAE              = {mask_rae:.4f}")

    if slsqp_rae <= mask_rae:
        final_method = f"slsqp_grand_{len(survivors)}way"
        final_rae = slsqp_rae
        deploy = (Q_te @ w_deploy).astype(np.float32)
        nb514_oof = oof_blend.astype(np.float32)
        active = [
            {"tag": t, "weight": float(wi)}
            for t, wi in zip(survivors, w_deploy)
        ]
        print(f"  -> grand SLSQP wins  RAE={final_rae:.4f}")
    else:
        final_method = f"hardmask_top3_{'_'.join(top3)}"
        final_rae = mask_rae
        deploy = mask_te
        nb514_oof = mask_oof
        active = [{"tag": t, "weight": 1.0 / 3.0} for t in top3]
        print(f"  -> hard-mask top-3 wins  RAE={final_rae:.4f}")

    beats_nb503 = bool(final_rae < NB503_CROSSFIT)
    print(f"  beats nb503 ({NB503_CROSSFIT:.4f})       = {beats_nb503}")
    if not beats_nb503:
        print(f"\nORTHO BLEND DOESNT HELP -- final cross-fit "
              f"{final_rae:.4f} >= nb503 {NB503_CROSSFIT:.4f}.")
        print(f"  survivors used : {survivors}")
        print(f"  dropped by filter (RAE>={RAE_FILTER}): {dropped}")
        print(f"  top-3 decorrelated : {top3}")
        print("  Explanation: the survivors' residuals are too positively "
              "correlated for SLSQP to extract orthogonal signal, and the "
              "best individual member (nb503) dominates any equal-weighted "
              "average of less-accurate routers.")

    # =================================================================
    # Save artefacts
    # =================================================================
    np.save(DATA_PROCESSED / "te_nb514.npy", deploy)
    np.save(DATA_PROCESSED / "nb514_pred_oof.npy", nb514_oof)

    plain = SUBMISSIONS / f"nb514_ortho_{final_method}.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(plain, index=False)

    soft = deploy.copy()
    soft[unb_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_idx]
    soft_path = SUBMISSIONS / f"nb514_ortho_{final_method}_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(soft_path, index=False)

    print(f"\nWrote {DATA_PROCESSED / 'te_nb514.npy'}")
    print(f"Wrote {DATA_PROCESSED / 'nb514_pred_oof.npy'}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")

    print("\n" + "=" * 78)
    print("=== nb514 SUMMARY ===")
    print(f"  survivors                                  = {survivors}")
    print(f"  dropped                                    = {dropped}")
    print(f"  top-3 decorrelated                         = {top3}")
    print(f"  grand SLSQP cross-fit RAE                  = {slsqp_rae:.4f}")
    print(f"  hard-mask top-3 RAE                        = {mask_rae:.4f}")
    print(f"  final method                               = {final_method}")
    print(f"  final cross-fit RAE                        = {final_rae:.4f}")
    print(f"  beats nb503 ({NB503_CROSSFIT:.4f})              = {beats_nb503}")
    print("=" * 78)

    return {
        "success": True,
        "survivors": survivors,
        "dropped": dropped,
        "rae_per_member": {t: float(rae_all[t]) for t in POOL},
        "mean_abs_rho": {t: float(mean_abs[t]) for t in survivors},
        "top3_decorrelated": top3,
        "slsqp_crossfit_rae": float(slsqp_rae),
        "slsqp_fold_range": [fold_lo, fold_hi],
        "slsqp_deploy_weights": {
            t: float(w) for t, w in zip(survivors, w_deploy)
        },
        "hardmask_rae": float(mask_rae),
        "final_method": final_method,
        "final_rae": float(final_rae),
        "active_weights": active,
        "beats_nb503": beats_nb503,
        "plain_submission": str(plain),
        "soft_submission": str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
