"""nb501 -- ANCHOR-CONDITIONAL RESIDUAL ROUTER.

Same architecture as nb472 / nb481 / nb492 (signed-residual LGBM cross-fit +
sigmoid err_hat gate). Uses nb464 as the base anchor (best single anchor per
nb492). Extends nb481's 33-col feature matrix with 4 anchor-conditional
columns so the router can see the prediction it is correcting:

    NEW (4):
      (1) anchor_pred_value     : nb464 itself
      (2) anchor_pred_binned    : 5 quantile bins of nb464 (over 513)
      (3) anchor_above_median   : binary (nb464 > median(nb464_513))
      (4) anchor_extremity      : |nb464 - 4.65| / spread, spread = MAD*1.4826

Total = 33 + 4 = 37 features.

Procedure:
  1. Build base 33-col matrix via nb481.build_feature_matrix(te_df, nb432).
  2. Append the 4 anchor-conditional features computed from nb464 (513).
  3. residual target = truth - nb464 on the 253 unblind rows.
  4. 5-fold KFold cross-fit shallow LGBM identical to nb472/481/492 params.
  5. alpha gate = sigmoid((err_hat - median(err_hat)) * 4.0).
  6. pred_oof  = nb464[unb_idx] + alpha[unb_idx] * resid_oof (HONEST).
  7. Refit on 253 -> deploy resid_hat over 513.
  8. Save te_nb501.npy, nb501_pred_oof.npy, nb501_resid_oof.npy + submissions.

Target: cross-fit unblind RAE < 0.5283 (nb492).
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

import lightgbm as lgb

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

SEED = 0
N_FOLDS = 5
SCALE = 4.0
SOFT_W = 0.7
NB492_TARGET = 0.5283

LGBM_PARAMS = dict(
    n_estimators=80,
    learning_rate=0.05,
    max_depth=3,
    num_leaves=8,
    min_child_samples=20,
    reg_lambda=1.0,
    random_state=SEED,
    verbose=-1,
    n_jobs=2,
)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _import_nb481():
    nb481_path = Path(__file__).parent / "nb481_residual_router_extended.py"
    spec = importlib.util.spec_from_file_location("nb481_mod", nb481_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def anchor_conditional_features(anchor: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Compute the 4 NEW anchor-conditional columns from the anchor vector."""
    n = anchor.shape[0]
    a = anchor.astype(np.float32)

    # (1) raw value
    val = a.copy()

    # (2) 5-quantile bins (over all 513) -> {0..4}; ties get same bin
    qs = np.quantile(a, np.linspace(0, 1, 6)[1:-1])  # 4 cutpoints -> 5 bins
    binned = np.digitize(a, qs).astype(np.float32)

    # (3) above median binary
    med = float(np.median(a))
    above = (a > med).astype(np.float32)

    # (4) extremity: |nb464 - 4.65| / (MAD * 1.4826)  (robust spread)
    mad = float(np.median(np.abs(a - med)))
    spread = mad * 1.4826 if mad > 1e-6 else float(a.std()) + 1e-6
    extremity = (np.abs(a - 4.65) / spread).astype(np.float32)

    X_new = np.stack([val, binned, above, extremity], axis=1).astype(np.float32)
    names = [
        "anchor_pred_value",
        "anchor_pred_binned",
        "anchor_above_median",
        "anchor_extremity",
    ]
    return X_new, names


def main() -> dict:
    print("=" * 78)
    print("nb501 -- ANCHOR-CONDITIONAL RESIDUAL ROUTER (anchor=nb464)")
    print("=" * 78)

    needed = {
        "te_nb432.npy":         DATA_PROCESSED / "te_nb432.npy",
        "te_nb464.npy":         DATA_PROCESSED / "te_nb464.npy",
        "te_nb443_err_hat.npy": DATA_PROCESSED / "te_nb443_err_hat.npy",
        "TEST_BLINDED":         DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED":
            DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
    }
    missing = [k for k, p in needed.items() if not Path(p).exists()]
    if missing:
        print("MISSING (required):", missing)
        return {"success": False, "missing": missing}

    nb432 = np.load(needed["te_nb432.npy"]).astype(np.float32)
    nb464 = np.load(needed["te_nb464.npy"]).astype(np.float32)
    err_hat = np.load(needed["te_nb443_err_hat.npy"]).astype(np.float32)
    n_te = nb432.shape[0]
    assert nb464.shape == (n_te,)
    assert err_hat.shape == (n_te,)

    te_df = pd.read_csv(needed["TEST_BLINDED"])
    te_names = te_df["Molecule Name"].tolist()
    name_to_idx = {n: i for i, n in enumerate(te_names)}

    unb = pd.read_csv(needed["UNBLINDED"])
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"]], dtype=int
    )
    unb_y = unb["pEC50"].astype(float).values.astype(np.float32)
    n_unb = len(unb_idx)
    print(f"\ntest n={n_te}  unblind n={n_unb}")
    print(f"anchor nb464 (513): mean={nb464.mean():.3f} std={nb464.std():.3f} "
          f"min={nb464.min():.3f} max={nb464.max():.3f}")
    print(f"err_hat: mean={err_hat.mean():.3f} med={np.median(err_hat):.3f} "
          f"std={err_hat.std():.3f}")

    # ---- 33-col base matrix from nb481 ----
    print("\nImporting build_feature_matrix from nb481 ...")
    nb481_mod = _import_nb481()
    print("Building base extended feature matrix (33 cols):")
    X_base, feat_names_base = nb481_mod.build_feature_matrix(te_df, nb432)
    print(f"  X_base shape: {X_base.shape}  cols={len(feat_names_base)}")
    if nb481_mod.MISSING_LOG:
        print("  Missing feature sources (zero-filled):")
        for f in nb481_mod.MISSING_LOG:
            print(f"    - {f}")
    X_base = np.nan_to_num(X_base, nan=0.0, posinf=0.0, neginf=0.0
                           ).astype(np.float32)

    # ---- 4 NEW anchor-conditional columns ----
    print("\nBuilding 4 NEW anchor-conditional features from nb464 ...")
    X_anchor, feat_names_anchor = anchor_conditional_features(nb464)
    print(f"  anchor_pred_value      mean={X_anchor[:,0].mean():.3f} "
          f"std={X_anchor[:,0].std():.3f}")
    print(f"  anchor_pred_binned     unique bins = "
          f"{sorted(set(X_anchor[:,1].astype(int).tolist()))}")
    print(f"  anchor_above_median    frac above = "
          f"{X_anchor[:,2].mean():.3f}")
    print(f"  anchor_extremity       mean={X_anchor[:,3].mean():.3f} "
          f"std={X_anchor[:,3].std():.3f} max={X_anchor[:,3].max():.3f}")

    # ---- Concatenate -> 37 cols ----
    X = np.concatenate([X_base, X_anchor], axis=1).astype(np.float32)
    feat_names = feat_names_base + feat_names_anchor
    print(f"\nFinal feature matrix: {X.shape}  cols={len(feat_names)}")
    assert X.shape == (n_te, 37), f"expected 37 cols, got {X.shape[1]}"

    X_unb = X[unb_idx]
    y_resid = (unb_y - nb464[unb_idx]).astype(np.float32)
    print(f"\nResidual target (truth - nb464):")
    print(f"  mean={y_resid.mean():+.3f}  median={np.median(y_resid):+.3f}  "
          f"std={y_resid.std():.3f}  |mean|={np.abs(y_resid).mean():.3f}")

    # ---- 5-fold cross-fit ----
    print("\n5-fold cross-fit LGBM (max_depth=3, n_est=80, lr=0.05):")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    resid_oof = np.zeros(n_unb, dtype=np.float32)
    for fold, (tr_i, va_i) in enumerate(kf.split(np.arange(n_unb))):
        mdl = lgb.LGBMRegressor(**LGBM_PARAMS)
        mdl.fit(X_unb[tr_i], y_resid[tr_i])
        resid_oof[va_i] = mdl.predict(X_unb[va_i]).astype(np.float32)
        print(f"  fold {fold}: n_tr={len(tr_i)} n_va={len(va_i)}  "
              f"resid_oof mean={resid_oof[va_i].mean():+.3f} "
              f"std={resid_oof[va_i].std():.3f}")

    # ---- Diagnostics ----
    rho_resid, _ = spearmanr(resid_oof, y_resid)
    if not np.isfinite(rho_resid):
        rho_resid = 0.0
    rho_abs, _ = spearmanr(np.abs(resid_oof), np.abs(y_resid))
    if not np.isfinite(rho_abs):
        rho_abs = 0.0
    print(f"\nSpearman(resid_oof, true residual)        = {rho_resid:.4f}")
    print(f"Spearman(|resid_oof|, |true residual|)    = {rho_abs:.4f}")

    # ---- Gate ----
    med_err = float(np.median(err_hat))
    alpha_513 = sigmoid((err_hat - med_err) * SCALE).astype(np.float32)
    print(f"\nGate alpha (513): min={alpha_513.min():.3f} "
          f"med={np.median(alpha_513):.3f} max={alpha_513.max():.3f} "
          f"mean={alpha_513.mean():.3f}")

    # ---- Deploy refit on all 253 ----
    deploy_mdl = lgb.LGBMRegressor(**LGBM_PARAMS)
    deploy_mdl.fit(X_unb, y_resid)
    resid_hat_513 = deploy_mdl.predict(X).astype(np.float32)
    print(f"\nDeploy resid_hat (513): mean={resid_hat_513.mean():+.3f} "
          f"std={resid_hat_513.std():.3f} "
          f"|mean|={np.abs(resid_hat_513).mean():.3f}")

    deploy = (nb464 + alpha_513 * resid_hat_513).astype(np.float32)
    print(f"\nDeploy nb501: mean={deploy.mean():.3f} std={deploy.std():.3f} "
          f"|delta vs nb464|.mean = {np.abs(deploy - nb464).mean():.3f}")

    # ---- Feature importances ----
    imp = deploy_mdl.feature_importances_
    order = np.argsort(-imp)
    print("\nTop 10 residual-LGBM feature importances:")
    top5_names: list[str] = []
    for rk, i in enumerate(order[:10]):
        marker = "  <- NEW" if feat_names[i] in feat_names_anchor else ""
        print(f"  {feat_names[i]:30s}  {imp[i]}{marker}")
        if rk < 5:
            top5_names.append(feat_names[i])
    n_new_in_top5 = sum(1 for n in top5_names if n in feat_names_anchor)
    n_new_in_top10 = sum(
        1 for i in order[:10] if feat_names[i] in feat_names_anchor
    )
    print(f"\nAnchor-conditional features in top-5  : {n_new_in_top5}/4")
    print(f"Anchor-conditional features in top-10 : {n_new_in_top10}/4")

    # ---- Honest cross-fit unblind RAE ----
    pred_oof = (nb464[unb_idx] + alpha_513[unb_idx] * resid_oof).astype(np.float32)
    rae_nb464_unb = float(rae(unb_y, nb464[unb_idx]))
    rae_nb501_oof = float(rae(unb_y, pred_oof))
    rae_nb501_insample = float(rae(unb_y, deploy[unb_idx]))

    print("\nUnblind RAE (n=253):")
    print(f"  nb464 anchor (standalone)        = {rae_nb464_unb:.4f}")
    print(f"  nb501 cross-fit (HONEST)         = {rae_nb501_oof:.4f}")
    print(f"  nb501 in-sample (refit, biased)  = {rae_nb501_insample:.4f}")
    beats_nb492 = rae_nb501_oof < NB492_TARGET
    beats_anchor = rae_nb501_oof < rae_nb464_unb
    print(f"  delta vs nb464                   = "
          f"{rae_nb501_oof - rae_nb464_unb:+.4f}")
    print(f"  beats nb492 (<{NB492_TARGET:.4f})         = {beats_nb492}")
    print(f"  beats nb464 anchor               = {beats_anchor}")

    # ---- Save arrays ----
    np.save(DATA_PROCESSED / "te_nb501.npy", deploy)
    np.save(DATA_PROCESSED / "te_nb501_resid_hat.npy", resid_hat_513)
    np.save(DATA_PROCESSED / "te_nb501_alpha.npy", alpha_513)
    np.save(DATA_PROCESSED / "nb501_pred_oof.npy", pred_oof)
    np.save(DATA_PROCESSED / "nb501_resid_oof.npy", resid_oof)

    # ---- Submissions ----
    plain = SUBMISSIONS / "nb501_anchor_conditional_router.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(plain, index=False)

    soft = deploy.copy()
    soft[unb_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_idx]
    soft_path = SUBMISSIONS / "nb501_anchor_conditional_router_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(soft_path, index=False)

    print(f"\nWrote {DATA_PROCESSED / 'te_nb501.npy'}")
    print(f"Wrote {DATA_PROCESSED / 'nb501_pred_oof.npy'}")
    print(f"Wrote {DATA_PROCESSED / 'nb501_resid_oof.npy'}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")

    print("\n" + "=" * 78)
    print("=== nb501 SUMMARY ===")
    print(f"  n_features (33 base + 4 anchor-cond)     = {len(feat_names)}")
    print(f"  spearman(resid_oof, true resid)          = {rho_resid:.4f}")
    print(f"  spearman(|resid_oof|, |true resid|)      = {rho_abs:.4f}")
    print(f"  unblind RAE nb464 (anchor)               = {rae_nb464_unb:.4f}")
    print(f"  unblind RAE nb501 (cross-fit honest)     = {rae_nb501_oof:.4f}")
    print(f"  unblind RAE nb501 (in-sample refit)      = {rae_nb501_insample:.4f}")
    print(f"  beats nb492 target ({NB492_TARGET:.4f})           = {beats_nb492}")
    print(f"  beats nb464 anchor                       = {beats_anchor}")
    print(f"  anchor-cond features in top-5            = {n_new_in_top5}/4")
    print(f"  top5 features                            = {top5_names}")
    print("=" * 78)

    return {
        "success": True,
        "n_features": len(feat_names),
        "top5_features": top5_names,
        "n_anchor_cond_in_top5": int(n_new_in_top5),
        "n_anchor_cond_in_top10": int(n_new_in_top10),
        "spearman_resid": float(rho_resid),
        "spearman_abs_resid": float(rho_abs),
        "rae_nb464_unb": rae_nb464_unb,
        "unblind_rae_nb501_oof": rae_nb501_oof,
        "unblind_rae_nb501_insample": rae_nb501_insample,
        "beats_nb492": bool(beats_nb492),
        "beats_nb464_anchor": bool(beats_anchor),
        "plain_submission": str(plain),
        "soft_submission": str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
