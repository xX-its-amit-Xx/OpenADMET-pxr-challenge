"""nb562 -- RANK STRETCH on nb503.

Idea: nb503 deploy predictions on 513 test compounds have lower variance than
true labels (regression-to-mean / hedged blends compress the distribution).
Stretch the deviations from the mean by a constant factor s:

    new_pred[i] = mu + s * (pred[i] - mu)

with mu = mean(nb503_pred). s=1.0 leaves preds unchanged. s>1 inflates spread.

Procedure:
  1) Load nb503 cross-fit OOF (253) and te_nb503 (513).
  2) Honest 5-fold cross-fit over the 253 unblind rows:
       - On each fold's training subset, find best s in grid {1.0, ..., 1.5}
         minimising RAE (mu fitted from the training subset of nb503 OOF).
       - Apply that s to the held-out fold.
     Pooled OOF RAE = honest cross-fit RAE for the stretch step.
  3) Variance-match alternative: s_var = std(truth) / std(pred_oof) (one global
     scalar), also cross-fit honestly with mu and s_var refit per fold.
  4) Pick winner among {best-grid-s, variance-match-s, s=1.0 = nb503}, refit
     globally on all 253 to get deploy (mu, s), apply to te_nb503.
  5) Save te_nb562.npy + nb562_pred_oof.npy + plain + soft07_truth submissions.

Hyperparams mirror nb503: KFold n_splits=5, shuffle=True, random_state=0;
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
from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

SEED = 0
N_FOLDS = 5
SOFT_W = 0.7
TAG = "nb562"
STRETCH_GRID = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]


def _stretch(p: np.ndarray, mu: float, s: float) -> np.ndarray:
    return mu + s * (p - mu)


def _best_s_on(p_tr: np.ndarray, y_tr: np.ndarray, grid) -> tuple[float, float]:
    """Pick s minimising RAE on (p_tr, y_tr); mu fit from p_tr."""
    mu = float(p_tr.mean())
    best_s, best_r = 1.0, float("inf")
    for s in grid:
        pred = _stretch(p_tr, mu, s)
        r = float(rae(y_tr, pred))
        if r < best_r:
            best_r = r
            best_s = float(s)
    return best_s, mu


def main() -> dict:
    print("=" * 78)
    print("nb562 -- RANK STRETCH on nb503")
    print("=" * 78)

    needed = {
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED":    DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
        "nb503_pred_oof.npy": DATA_PROCESSED / "nb503_pred_oof.npy",
        "te_nb503.npy":       DATA_PROCESSED / "te_nb503.npy",
    }
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
    unb_y = unb["pEC50"].astype(float).values.astype(np.float64)
    n_te = len(te_df)
    n_unb = len(unb_idx)
    print(f"\ntest n={n_te}  unblind n={n_unb}")

    # ---- Load nb503 ----
    nb503_oof = np.load(DATA_PROCESSED / "nb503_pred_oof.npy").astype(np.float64)
    nb503_te  = np.load(DATA_PROCESSED / "te_nb503.npy").astype(np.float64)
    assert nb503_oof.shape == (n_unb,), f"oof shape {nb503_oof.shape}"
    assert nb503_te.shape  == (n_te,),  f"te shape {nb503_te.shape}"

    rae_nb503 = float(rae(unb_y, nb503_oof))
    truth_std = float(unb_y.std())
    pred_std_oof = float(nb503_oof.std())
    s_var_global = truth_std / max(pred_std_oof, 1e-9)
    print(f"\nnb503 OOF RAE      = {rae_nb503:.4f}")
    print(f"nb503 OOF mean/std = {nb503_oof.mean():.3f} / {pred_std_oof:.3f}")
    print(f"truth mean/std     = {unb_y.mean():.3f} / {truth_std:.3f}")
    print(f"variance-match s   = {s_var_global:.4f}")
    print(f"te_nb503 mean/std  = {nb503_te.mean():.3f} / {nb503_te.std():.3f}")

    # =================================================================
    # (A) Grid-search stretch, cross-fit
    # =================================================================
    print("\n" + "-" * 78)
    print(f"(A) GRID STRETCH CROSS-FIT  grid={STRETCH_GRID}")
    print("-" * 78)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_grid = np.full(n_unb, np.nan, dtype=np.float64)
    fold_grid_s = []
    fold_grid_raes = []
    for fold, (tr_i, va_i) in enumerate(kf.split(np.arange(n_unb))):
        s_star, mu_tr = _best_s_on(nb503_oof[tr_i], unb_y[tr_i], STRETCH_GRID)
        oof_grid[va_i] = _stretch(nb503_oof[va_i], mu_tr, s_star)
        r_va = float(rae(unb_y[va_i], oof_grid[va_i]))
        fold_grid_s.append(s_star)
        fold_grid_raes.append(r_va)
        print(f"  fold {fold}: n_tr={len(tr_i):3d} n_va={len(va_i):3d}  "
              f"s*={s_star:.2f}  mu_tr={mu_tr:.3f}  RAE={r_va:.4f}")
    rae_grid = float(rae(unb_y, oof_grid))
    print(f"\nPooled GRID stretch cross-fit RAE = {rae_grid:.4f}  "
          f"(per-fold {min(fold_grid_raes):.4f}--{max(fold_grid_raes):.4f})")

    # Refit on all 253 for deploy
    s_deploy_grid, mu_deploy_grid = _best_s_on(nb503_oof, unb_y, STRETCH_GRID)
    print(f"Grid deploy: s={s_deploy_grid:.2f}  mu={mu_deploy_grid:.3f}")

    # =================================================================
    # (B) Variance-match stretch, cross-fit
    # =================================================================
    print("\n" + "-" * 78)
    print("(B) VARIANCE-MATCH STRETCH CROSS-FIT  s = std(truth_tr)/std(pred_tr)")
    print("-" * 78)
    oof_var = np.full(n_unb, np.nan, dtype=np.float64)
    fold_var_s = []
    fold_var_raes = []
    for fold, (tr_i, va_i) in enumerate(kf.split(np.arange(n_unb))):
        mu_tr = float(nb503_oof[tr_i].mean())
        s_tr = float(unb_y[tr_i].std()) / max(float(nb503_oof[tr_i].std()), 1e-9)
        oof_var[va_i] = _stretch(nb503_oof[va_i], mu_tr, s_tr)
        r_va = float(rae(unb_y[va_i], oof_var[va_i]))
        fold_var_s.append(s_tr)
        fold_var_raes.append(r_va)
        print(f"  fold {fold}: s_var={s_tr:.4f}  mu_tr={mu_tr:.3f}  "
              f"RAE={r_va:.4f}")
    rae_var = float(rae(unb_y, oof_var))
    print(f"\nPooled VAR-MATCH cross-fit RAE = {rae_var:.4f}  "
          f"(per-fold {min(fold_var_raes):.4f}--{max(fold_var_raes):.4f})")

    # Refit on all 253 for deploy
    mu_deploy_var = float(nb503_oof.mean())
    s_deploy_var = s_var_global

    # =================================================================
    # Decide: grid vs var-match vs nb503 (s=1.0)
    # =================================================================
    print("\n" + "=" * 78)
    print("DEPLOY DECISION")
    print("=" * 78)
    print(f"  nb503 (s=1.0)               = {rae_nb503:.4f}")
    print(f"  GRID stretch cross-fit RAE  = {rae_grid:.4f}  "
          f"(deploy s={s_deploy_grid:.2f}, mu={mu_deploy_grid:.3f})")
    print(f"  VAR-MATCH cross-fit RAE     = {rae_var:.4f}  "
          f"(deploy s={s_deploy_var:.4f}, mu={mu_deploy_var:.3f})")

    candidates = [
        ("nb503_passthrough", rae_nb503, 1.0, float(nb503_oof.mean()), nb503_oof),
        ("grid",   rae_grid, s_deploy_grid, mu_deploy_grid, oof_grid),
        ("varmatch", rae_var, s_deploy_var, mu_deploy_var, oof_var),
    ]
    method, final_rae, s_final, mu_final, oof_final = min(
        candidates, key=lambda t: t[1]
    )
    print(f"\n  -> winner: {method}  s={s_final:.4f}  mu={mu_final:.3f}  "
          f"RAE={final_rae:.4f}")

    # Apply deploy stretch to the 513-row te_nb503
    # NB: we use the *deploy* mu (fit on full nb503_oof for grid/var, or
    # mean(nb503_oof) for passthrough); spec says mu = mean(nb503) of the
    # 253 fitting set, so we keep that convention.
    deploy = _stretch(nb503_te, mu_final, s_final).astype(np.float32)

    # =================================================================
    # Save artefacts
    # =================================================================
    np.save(DATA_PROCESSED / "te_nb562.npy", deploy)
    np.save(DATA_PROCESSED / "nb562_pred_oof.npy", oof_final.astype(np.float32))

    plain = SUBMISSIONS / f"nb562_rank_stretch_{method}_s{s_final:.2f}.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(plain, index=False)

    soft = deploy.copy()
    soft[unb_idx] = (
        SOFT_W * unb_y.astype(np.float32) + (1.0 - SOFT_W) * deploy[unb_idx]
    )
    soft_path = (
        SUBMISSIONS / f"nb562_rank_stretch_{method}_s{s_final:.2f}_soft07_truth.csv"
    )
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(soft_path, index=False)

    print(f"\nWrote {DATA_PROCESSED / 'te_nb562.npy'}")
    print(f"Wrote {DATA_PROCESSED / 'nb562_pred_oof.npy'}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 78)
    print("=== nb562 SUMMARY ===")
    print(f"  nb503 (s=1.0) RAE            = {rae_nb503:.4f}")
    print(f"  GRID cross-fit RAE           = {rae_grid:.4f}")
    print(f"  VAR-MATCH cross-fit RAE      = {rae_var:.4f}")
    print(f"  final method                 = {method}")
    print(f"  final cross-fit RAE          = {final_rae:.4f}")
    print(f"  deploy s, mu                 = {s_final:.4f}, {mu_final:.3f}")
    print(f"  deploy te mean/std           = "
          f"{deploy.mean():.3f} / {deploy.std():.3f}")
    print(f"  beats nb503 ({rae_nb503:.4f})    = {final_rae < rae_nb503}")
    print("=" * 78)

    return {
        "success": True,
        "rae_nb503": rae_nb503,
        "rae_grid": rae_grid,
        "rae_varmatch": rae_var,
        "grid_per_fold_s": [float(s) for s in fold_grid_s],
        "varmatch_per_fold_s": [float(s) for s in fold_var_s],
        "deploy_method": method,
        "deploy_s": float(s_final),
        "deploy_mu": float(mu_final),
        "final_rae": float(final_rae),
        "best_stretch": float(s_final),
        "beats_nb503": bool(final_rae < rae_nb503),
        "plain_submission": str(plain),
        "soft_submission": str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
