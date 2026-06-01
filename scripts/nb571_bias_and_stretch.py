"""nb571 -- BIAS + STRETCH (2-param affine) on nb503.

Generalises nb562 (stretch-only, s=1.10) by adding a global bias shift to the
recentred pivot:

    new_pred[i] = s * (orig[i] - mu) + (mu + shift)

mu is fit per fold from the training subset of nb503 OOF (same convention as
nb562). Honest 5-fold cross-fit picks (s, shift) on each fold's training rows;
the held-out fold is scored with that (s, shift).

Grid:
    s     in {1.00, 1.05, 1.10, 1.15, 1.20}
    shift in {-0.2, -0.1, 0.0, +0.1, +0.2}

Compares pooled RAE against nb562 (s=1.10, shift=0). Deploys winning
(s*, shift*) refit on all 253 to the 513-row test set.
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
TAG = "nb571"
S_GRID = [1.00, 1.05, 1.10, 1.15, 1.20]
SHIFT_GRID = [-0.2, -0.1, 0.0, 0.1, 0.2]
NB562_S = 1.10
NB562_SHIFT = 0.0


def _affine(p: np.ndarray, mu: float, s: float, shift: float) -> np.ndarray:
    return s * (p - mu) + (mu + shift)


def _best_on(
    p_tr: np.ndarray, y_tr: np.ndarray, s_grid, shift_grid
) -> tuple[float, float, float]:
    """Pick (s, shift) minimising RAE on (p_tr, y_tr); mu = mean(p_tr)."""
    mu = float(p_tr.mean())
    best_s, best_shift, best_r = 1.0, 0.0, float("inf")
    for s in s_grid:
        for sh in shift_grid:
            pred = _affine(p_tr, mu, s, sh)
            r = float(rae(y_tr, pred))
            if r < best_r:
                best_r = r
                best_s = float(s)
                best_shift = float(sh)
    return best_s, best_shift, mu


def main() -> dict:
    print("=" * 78)
    print("nb571 -- BIAS + STRETCH (2-param affine) on nb503")
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
    print(f"\nnb503 OOF RAE      = {rae_nb503:.4f}")
    print(f"nb503 OOF mean/std = {nb503_oof.mean():.3f} / {nb503_oof.std():.3f}")
    print(f"truth mean/std     = {unb_y.mean():.3f} / {unb_y.std():.3f}")
    print(f"te_nb503 mean/std  = {nb503_te.mean():.3f} / {nb503_te.std():.3f}")

    # =================================================================
    # nb562 baseline reconstruction (s=1.10, shift=0), cross-fit
    # mu refit per fold to mirror nb562
    # =================================================================
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_562 = np.full(n_unb, np.nan, dtype=np.float64)
    for fold, (tr_i, va_i) in enumerate(kf.split(np.arange(n_unb))):
        mu_tr = float(nb503_oof[tr_i].mean())
        oof_562[va_i] = _affine(nb503_oof[va_i], mu_tr, NB562_S, NB562_SHIFT)
    rae_562 = float(rae(unb_y, oof_562))
    print(f"\nnb562 reconstr. (s=1.10, shift=0) cross-fit RAE = {rae_562:.4f}")

    # =================================================================
    # (A) Grid-search 2D affine, cross-fit
    # =================================================================
    print("\n" + "-" * 78)
    print(f"(A) GRID 2D AFFINE CROSS-FIT")
    print(f"    s grid     = {S_GRID}")
    print(f"    shift grid = {SHIFT_GRID}")
    print("-" * 78)
    oof_grid = np.full(n_unb, np.nan, dtype=np.float64)
    fold_s, fold_shift, fold_raes = [], [], []
    for fold, (tr_i, va_i) in enumerate(kf.split(np.arange(n_unb))):
        s_star, sh_star, mu_tr = _best_on(
            nb503_oof[tr_i], unb_y[tr_i], S_GRID, SHIFT_GRID
        )
        oof_grid[va_i] = _affine(nb503_oof[va_i], mu_tr, s_star, sh_star)
        r_va = float(rae(unb_y[va_i], oof_grid[va_i]))
        fold_s.append(s_star)
        fold_shift.append(sh_star)
        fold_raes.append(r_va)
        print(
            f"  fold {fold}: n_tr={len(tr_i):3d} n_va={len(va_i):3d}  "
            f"s*={s_star:.2f}  shift*={sh_star:+.2f}  mu_tr={mu_tr:.3f}  "
            f"RAE={r_va:.4f}"
        )
    rae_grid = float(rae(unb_y, oof_grid))
    print(
        f"\nPooled GRID 2D cross-fit RAE = {rae_grid:.4f}  "
        f"(per-fold {min(fold_raes):.4f}--{max(fold_raes):.4f})"
    )

    # Refit on all 253 for deploy
    s_deploy, shift_deploy, mu_deploy = _best_on(
        nb503_oof, unb_y, S_GRID, SHIFT_GRID
    )
    print(
        f"\nDeploy (full 253): s*={s_deploy:.2f}  shift*={shift_deploy:+.2f}  "
        f"mu={mu_deploy:.3f}"
    )

    # =================================================================
    # Compare to nb562
    # =================================================================
    gain_vs_562 = rae_562 - rae_grid
    print("\n" + "=" * 78)
    print("COMPARISON")
    print("=" * 78)
    print(f"  nb503 (s=1.0, shift=0)          = {rae_nb503:.4f}")
    print(f"  nb562 (s=1.10, shift=0)         = {rae_562:.4f}")
    print(f"  nb571 GRID 2D cross-fit         = {rae_grid:.4f}")
    print(f"  gain vs nb562                   = {gain_vs_562:+.4f}")
    print(f"  beats nb562?                    = {rae_grid < rae_562}")

    # =================================================================
    # Deploy: apply (s*, shift*) refit on full 253 to te_nb503
    # =================================================================
    deploy = _affine(nb503_te, mu_deploy, s_deploy, shift_deploy).astype(
        np.float32
    )

    np.save(DATA_PROCESSED / "te_nb571.npy", deploy)
    np.save(DATA_PROCESSED / "nb571_pred_oof.npy", oof_grid.astype(np.float32))

    tag_str = f"s{s_deploy:.2f}_sh{shift_deploy:+.2f}".replace("+", "p").replace(
        "-", "m"
    )
    plain = SUBMISSIONS / f"nb571_bias_stretch_{tag_str}.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(plain, index=False)

    soft = deploy.copy()
    soft[unb_idx] = (
        SOFT_W * unb_y.astype(np.float32) + (1.0 - SOFT_W) * deploy[unb_idx]
    )
    soft_path = SUBMISSIONS / f"nb571_bias_stretch_{tag_str}_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(soft_path, index=False)

    print(f"\nWrote {DATA_PROCESSED / 'te_nb571.npy'}")
    print(f"Wrote {DATA_PROCESSED / 'nb571_pred_oof.npy'}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")
    print(f"\ndeploy te mean/std = {deploy.mean():.3f} / {deploy.std():.3f}")

    print("\n" + "=" * 78)
    print("=== nb571 SUMMARY ===")
    print(f"  nb503 RAE                    = {rae_nb503:.4f}")
    print(f"  nb562 RAE                    = {rae_562:.4f}")
    print(f"  nb571 GRID 2D cross-fit RAE  = {rae_grid:.4f}")
    print(f"  gain vs nb562                = {gain_vs_562:+.4f}")
    print(f"  deploy (s, shift, mu)        = "
          f"({s_deploy:.2f}, {shift_deploy:+.2f}, {mu_deploy:.3f})")
    print(f"  beats nb562                  = {rae_grid < rae_562}")
    print("=" * 78)

    return {
        "success": True,
        "rae_nb503": rae_nb503,
        "rae_nb562": rae_562,
        "rae_nb571_crossfit": rae_grid,
        "gain_vs_nb562": gain_vs_562,
        "fold_s": [float(x) for x in fold_s],
        "fold_shift": [float(x) for x in fold_shift],
        "fold_raes": [float(x) for x in fold_raes],
        "deploy_s": float(s_deploy),
        "deploy_shift": float(shift_deploy),
        "deploy_mu": float(mu_deploy),
        "beats_nb562": bool(rae_grid < rae_562),
        "plain_submission": str(plain),
        "soft_submission": str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
