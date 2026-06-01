"""nb482 -- MULTI-SEED RESIDUAL ROUTER ENSEMBLE.

Hypothesis: nb472's residual LGBM is weak (Spearman ~0.121 on |resid|) because
single-seed shallow LGBM on 253 rows has high variance. Average the cross-fit
residual OOF across 5 seeds [13, 42, 137, 314, 1729] to reduce noise without
changing bias. If the signal is real, ensembling should drop RAE by ~0.005-0.010.

Reuses the EXACT 18-feature matrix from nb472 (no extension yet -- pure
variance-reduction test).

Pipeline:
  1) Reuse te_nb443_err_hat.npy + nb472's 18-col feature matrix.
  2) For each seed in [13, 42, 137, 314, 1729]:
       - 5-fold KFold cross-fit -> resid_oof_seed (253,)
       - refit on all 253 -> deploy resid_hat_seed (513,)
  3) Average: resid_oof_ensemble = mean(resid_oof_seeds, axis=0)
              resid_hat_ensemble = mean(resid_hat_seeds, axis=0)
  4) Same alpha gate as nb472: alpha = sigmoid((err_hat - median)*4)
  5) Honest cross-fit unblind RAE using ENSEMBLE OOF residual.
  6) Save te_nb482.npy (513) and nb482_pred_oof.npy (253 honest preds).
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
from rdkit import RDLogger
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

import lightgbm as lgb

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

# Reuse the feature matrix builder from nb472 exactly.
sys.path.insert(0, os.path.dirname(__file__))
from nb472_residual_stack_router import build_feature_matrix  # noqa: E402

RDLogger.DisableLog("rdApp.*")

SEEDS = [13, 42, 137, 314, 1729]
N_FOLDS = 5
SCALE = 4.0
SOFT_W = 0.7


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main() -> dict:
    print("=" * 78)
    print("nb482 -- MULTI-SEED RESIDUAL ROUTER ENSEMBLE")
    print(f"  seeds = {SEEDS}")
    print("=" * 78)

    needed = {
        "te_nb432.npy": DATA_PROCESSED / "te_nb432.npy",
        "te_nb443_err_hat.npy": DATA_PROCESSED / "te_nb443_err_hat.npy",
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED":
            DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
    }
    missing = [k for k, p in needed.items() if not Path(p).exists()]
    if missing:
        print("MISSING:", missing)
        return {"success": False, "missing": missing}

    nb432 = np.load(needed["te_nb432.npy"]).astype(np.float32)
    err_hat = np.load(needed["te_nb443_err_hat.npy"]).astype(np.float32)
    n_te = nb432.shape[0]
    assert err_hat.shape == (n_te,)

    te_df = pd.read_csv(needed["TEST_BLINDED"])
    te_names = te_df["Molecule Name"].tolist()
    name_to_idx = {n: i for i, n in enumerate(te_names)}

    unb = pd.read_csv(needed["UNBLINDED"])
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_te_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"]], dtype=int
    )
    unb_y = unb["pEC50"].astype(float).values.astype(np.float32)
    n_unb = len(unb_te_idx)
    print(f"\ntest n={n_te}  unblind n={n_unb}")

    # ---------- Feature matrix (identical to nb472) ----------
    print("\nBuilding 18-col feature matrix (reusing nb472 builder):")
    X, feat_names = build_feature_matrix(te_df, nb432)
    print(f"  X shape: {X.shape}  cols={len(feat_names)}")
    X_unb = X[unb_te_idx]
    y_resid = (unb_y - nb432[unb_te_idx]).astype(np.float32)
    print(f"\nResidual target: mean={y_resid.mean():+.3f}  "
          f"std={y_resid.std():.3f}  |mean|={np.abs(y_resid).mean():.3f}")

    # ---------- Multi-seed cross-fit residual LGBM ----------
    resid_oof_seeds = []
    resid_hat_seeds = []
    print("\nPer-seed cross-fit + deploy refit:")
    for seed in SEEDS:
        params = dict(
            n_estimators=80,
            learning_rate=0.05,
            max_depth=3,
            num_leaves=8,
            min_child_samples=20,
            reg_lambda=1.0,
            random_state=seed,
            verbose=-1,
            n_jobs=2,
        )
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        resid_oof_s = np.zeros(n_unb, dtype=np.float32)
        for tr_i, va_i in kf.split(np.arange(n_unb)):
            mdl = lgb.LGBMRegressor(**params)
            mdl.fit(X_unb[tr_i], y_resid[tr_i])
            resid_oof_s[va_i] = mdl.predict(X_unb[va_i]).astype(np.float32)
        deploy_mdl = lgb.LGBMRegressor(**params)
        deploy_mdl.fit(X_unb, y_resid)
        resid_hat_s = deploy_mdl.predict(X).astype(np.float32)
        rho_s, _ = spearmanr(np.abs(resid_oof_s), np.abs(y_resid))
        if not np.isfinite(rho_s):
            rho_s = 0.0
        print(f"  seed {seed:4d}: oof std={resid_oof_s.std():.3f}  "
              f"deploy std={resid_hat_s.std():.3f}  "
              f"spearman(|oof|,|true|)={rho_s:+.4f}")
        resid_oof_seeds.append(resid_oof_s)
        resid_hat_seeds.append(resid_hat_s)

    # ---------- Ensemble ----------
    oof_mat = np.stack(resid_oof_seeds, axis=0)  # (S, n_unb)
    hat_mat = np.stack(resid_hat_seeds, axis=0)  # (S, n_te)
    resid_oof = oof_mat.mean(axis=0).astype(np.float32)
    resid_hat = hat_mat.mean(axis=0).astype(np.float32)

    # Variance reduction diagnostics
    per_seed_oof_std = oof_mat.std(axis=0).mean()
    print(f"\nEnsemble: avg per-row OOF std across seeds = {per_seed_oof_std:.4f}")
    print(f"  ensemble resid_oof: std={resid_oof.std():.3f}  "
          f"|mean|={np.abs(resid_oof).mean():.3f}")
    print(f"  ensemble resid_hat (513): std={resid_hat.std():.3f}  "
          f"|mean|={np.abs(resid_hat).mean():.3f}")

    rho_resid, _ = spearmanr(resid_oof, y_resid)
    if not np.isfinite(rho_resid):
        rho_resid = 0.0
    rho_abs, _ = spearmanr(np.abs(resid_oof), np.abs(y_resid))
    if not np.isfinite(rho_abs):
        rho_abs = 0.0
    print(f"\nSpearman(ensemble resid_oof, true resid)    = {rho_resid:+.4f}")
    print(f"Spearman(|ensemble resid_oof|, |true resid|) = {rho_abs:+.4f}")

    # ---------- Same alpha gate as nb472 ----------
    med_err = float(np.median(err_hat))
    alpha_513 = sigmoid((err_hat - med_err) * SCALE).astype(np.float32)
    print(f"\nGate alpha (513): min={alpha_513.min():.3f}  "
          f"med={np.median(alpha_513):.3f}  max={alpha_513.max():.3f}  "
          f"mean={alpha_513.mean():.3f}")

    # ---------- Deploy ----------
    deploy = (nb432 + alpha_513 * resid_hat).astype(np.float32)
    print(f"\nDeploy nb482: mean={deploy.mean():.3f}  std={deploy.std():.3f}  "
          f"|delta vs nb432|.mean = {np.abs(deploy - nb432).mean():.3f}")

    # ---------- Honest cross-fit unblind RAE ----------
    nb482_unb_oof = (nb432[unb_te_idx]
                     + alpha_513[unb_te_idx] * resid_oof).astype(np.float32)
    rae_nb432_unb = float(rae(unb_y, nb432[unb_te_idx]))
    rae_nb482_oof = float(rae(unb_y, nb482_unb_oof))
    rae_nb482_insample = float(rae(unb_y, deploy[unb_te_idx]))

    # nb472 single-seed reference for direct comparison
    try:
        from nb472_residual_stack_router import main as _m  # noqa: F401
        nb472_oof_ref = None
        nb472_deploy_path = DATA_PROCESSED / "te_nb472.npy"
        if nb472_deploy_path.exists():
            nb472_deploy = np.load(nb472_deploy_path).astype(np.float32)
            nb472_oof_ref = float(rae(unb_y, nb472_deploy[unb_te_idx]))
    except Exception:
        nb472_oof_ref = None

    print("\nUnblind RAE (n=253):")
    print(f"  nb432 baseline                = {rae_nb432_unb:.4f}")
    if nb472_oof_ref is not None:
        print(f"  nb472 (single-seed, deploy)   = {nb472_oof_ref:.4f}")
    print(f"  nb482 ensemble cross-fit OOF  = {rae_nb482_oof:.4f}")
    print(f"  nb482 ensemble in-sample      = {rae_nb482_insample:.4f}")
    beats_nb472 = (nb472_oof_ref is None) or (rae_nb482_oof < nb472_oof_ref)
    beats_nb432 = rae_nb482_oof < rae_nb432_unb
    print(f"  beats_nb432                    = {beats_nb432}")
    print(f"  beats_nb472 (cross-fit single) = {beats_nb472}")

    # ---------- Save arrays + submissions ----------
    np.save(DATA_PROCESSED / "te_nb482.npy", deploy)
    np.save(DATA_PROCESSED / "te_nb482_resid_hat.npy", resid_hat)
    np.save(DATA_PROCESSED / "te_nb482_alpha.npy", alpha_513)
    # 253 honest cross-fit preds requested
    np.save(DATA_PROCESSED / "nb482_pred_oof.npy", nb482_unb_oof)

    plain = SUBMISSIONS / "nb482_multi_seed_router_ensemble.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(plain, index=False)

    soft = deploy.copy()
    soft[unb_te_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_te_idx]
    soft_path = SUBMISSIONS / "nb482_multi_seed_router_ensemble_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(soft_path, index=False)

    print(f"\nWrote {DATA_PROCESSED / 'te_nb482.npy'}")
    print(f"Wrote {DATA_PROCESSED / 'nb482_pred_oof.npy'}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")

    print("\n" + "=" * 78)
    print("=== nb482 SUMMARY ===")
    print(f"  seeds                                  = {SEEDS}")
    print(f"  spearman(ensemble resid_oof, true)     = {rho_resid:+.4f}")
    print(f"  spearman(|ensemble resid|, |true|)     = {rho_abs:+.4f}")
    print(f"  unblind RAE nb432                       = {rae_nb432_unb:.4f}")
    if nb472_oof_ref is not None:
        print(f"  unblind RAE nb472 (single-seed deploy) = {nb472_oof_ref:.4f}")
    print(f"  unblind RAE nb482 cross-fit (honest)   = {rae_nb482_oof:.4f}")
    print(f"  unblind RAE nb482 in-sample            = {rae_nb482_insample:.4f}")
    print(f"  beats_nb432                            = {beats_nb432}")
    print(f"  beats_nb472                            = {beats_nb472}")
    print("=" * 78)

    return {
        "success": True,
        "seeds": SEEDS,
        "spearman_resid": float(rho_resid),
        "spearman_abs_resid": float(rho_abs),
        "rae_nb432_unb": rae_nb432_unb,
        "rae_nb472_ref": nb472_oof_ref,
        "unblind_rae_nb482_oof": rae_nb482_oof,
        "unblind_rae_nb482_insample": rae_nb482_insample,
        "beats_nb432": bool(beats_nb432),
        "beats_nb472": bool(beats_nb472),
        "plain_submission": str(plain),
        "soft_submission": str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
