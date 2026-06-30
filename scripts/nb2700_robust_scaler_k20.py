"""nb2700 -- RobustScaler (median/IQR) + LGBM on K=20 substrate.

NEW PARADIGM (vs StandardScaler baseline):
    StandardScaler centers on mean and scales by std-dev; both estimators
    are non-robust to outliers in feature distribution.  RobustScaler
    swaps them out for median (location) and IQR (scale), which are
    breakdown-point ~0.25 estimators.  Several of the K=20 RFE-survivor
    columns are physchem descriptors with heavy-tailed distributions
    (Mordred topological indices, PEOE_VSA bins, partial-charge sums) --
    mean/std on those can be pulled around by a handful of large-MW
    outliers in the 253 unb set, so per-fold StandardScaler can produce
    scaled features whose dynamic range is dominated by the outliers and
    whose mid-range (where most splits happen) gets compressed.

    RobustScaler with the default IQR=(25,75) range puts the middle 50%
    of the training distribution into a unit-width band, so LGBM tree
    splits should land on more informative cut-points.  On the same
    K=20 substrate, fixed chemprop_aux anchor, and identical LGBM hp,
    this is a SCALER swap only -- a substrate-shape change in the
    cycle-134 sense (model class fixed, preprocessing changed).

    Hypothesis: if a non-trivial fraction of K=20 columns are
    outlier-driven on the 253 unblind, RobustScaler scaling will yield
    OOF residual splits that translate to a lower scaffold-CV RAE than
    the StandardScaler-equivalent nb2240 K=20 LGBM baseline (0.4630).

PROTOCOL:
    1. Load X_117 substrate -> slice K=20 surviving columns from
       nb2240 summary (idx_in_117 + names mirror nb2231 RFE result).
    2. residual = y_unb - chemprop_aux_te[unb_idx]  (only PRE-clean anchor).
    3. Per scaffold-fold: RobustScaler fit on train -> transform val.
       (RobustScaler default quantile_range=(25.0, 75.0), with_centering=True,
        with_scaling=True.)
    4. LGBM(max_depth=4, num_leaves=15, n_est=300, lr=0.03) on robust-scaled
       K=20 features, residual target.
    5. 5-fold scaffold CV (`scaffold_kfold_indices`), 5 kf_seeds
       {1001..1005}.
    6. Deploy: refit RobustScaler + LGBM on full 253 per seed -> predict
       513 te residual; mean-bag aggregate.

GATE (mean-bag corrected RAE):
    mean_rae < 0.4570 -> "PROMOTE"
    mean_rae < 0.4598 -> "MARGINAL_BEAT"
    else              -> "FAIL"

OUTPUTS:
    scripts/nb2700_robust_scaler_k20.py
    data/processed/nb2700_summary.json
    data/processed/nb2700_pred_oof.npy   (253,) float32 mean-bag CORRECTED
    data/processed/te_nb2700.npy         (513,) float32 deploy refit
    submissions/nb2700_robust_scaler_k20.csv
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2700"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"
NB2240_SUMMARY = DATA_PROCESSED / "nb2240_summary.json"

N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# RobustScaler default: q1=25, q3=75 -> IQR
ROBUST_Q_LOW = 25.0
ROBUST_Q_HIGH = 75.0

# Gate thresholds (mean-bag RAE)
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

CHEMPROP_AUX_REF = 0.6216
NB2240_K20_REF = 0.4630  # StandardScaler+LGBM K=20 baseline


def _lgbm_params(seed: int) -> dict:
    """LGBM hyperparams as requested in task spec."""
    return dict(
        objective="regression",
        max_depth=4,
        num_leaves=15,
        n_estimators=300,
        learning_rate=0.03,
        min_child_samples=5,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _scaffold_cv_one_seed(
    X: np.ndarray,
    residual: np.ndarray,
    unb_scaffolds: list,
    kf_seed: int,
) -> tuple[np.ndarray, list[dict]]:
    """One scaffold-CV pass: fit per-fold RobustScaler + LGBM.

    Returns (oof_residual_pred, per_fold_diagnostics).
    """
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n = len(residual)
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_diags = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        scaler = RobustScaler(
            quantile_range=(ROBUST_Q_LOW, ROBUST_Q_HIGH),
            with_centering=True,
            with_scaling=True,
        )
        X_tr = scaler.fit_transform(X[tr_loc]).astype(np.float32)
        X_va = scaler.transform(X[va_loc]).astype(np.float32)
        mdl = lgb.LGBMRegressor(**_lgbm_params(kf_seed))
        mdl.fit(X_tr, residual[tr_loc])
        oof[va_loc] = mdl.predict(X_va)
        # Per-fold diagnostic: scaler statistics
        fold_diags.append({
            "fold": fold_i,
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "center_min": float(scaler.center_.min()),
            "center_max": float(scaler.center_.max()),
            "scale_min": float(scaler.scale_.min()),
            "scale_max": float(scaler.scale_.max()),
            "scale_median": float(np.median(scaler.scale_)),
        })
    if np.isnan(oof).any():
        raise RuntimeError("scaffold split did not cover all rows")
    return oof, fold_diags


def _deploy_te_one_seed(
    X_unb: np.ndarray,
    residual: np.ndarray,
    X_te: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Fit RobustScaler + LGBM on full 253; predict 513 residual."""
    scaler = RobustScaler(
        quantile_range=(ROBUST_Q_LOW, ROBUST_Q_HIGH),
        with_centering=True,
        with_scaling=True,
    )
    X_unb_s = scaler.fit_transform(X_unb).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)
    mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
    mdl.fit(X_unb_s, residual)
    pred = mdl.predict(X_te_s).astype(np.float32)
    diag = {
        "seed": int(seed),
        "center_min": float(scaler.center_.min()),
        "center_max": float(scaler.center_.max()),
        "scale_min": float(scaler.scale_.min()),
        "scale_max": float(scaler.scale_.max()),
        "scale_median": float(np.median(scaler.scale_)),
    }
    return pred, diag


def main() -> dict:
    import pandas as pd
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- RobustScaler + LGBM on K=20 substrate "
          f"(chemprop_aux residual)")
    print(f"        RobustScaler quantile_range=({ROBUST_Q_LOW},{ROBUST_Q_HIGH})  "
          f"with_centering=True with_scaling=True")
    print(f"        anchor = {ANCHOR}  scaffold-CV {N_FOLDS}-fold  "
          f"kf_seeds={KF_SEEDS}")
    print(f"        ref nb2240 K=20 StandardScaler+LGBM = {NB2240_K20_REF:.4f}")
    print(f"        GATE: <{GATE_PROMOTE} PROMOTE / "
          f"<{GATE_MARGINAL} MARGINAL_BEAT / else FAIL")
    print("=" * 78)

    # ---- Load truth + anchor + scaffolds ----
    te = load_test()
    n_test = len(te)
    test_smiles = (
        te["smiles"].astype(str).tolist()
        if "smiles" in te.columns
        else te["SMILES"].astype(str).tolist()
    )
    te_names = (
        te["name"].values
        if "name" in te.columns
        else te["Molecule Name"].values
    )

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"anchor te missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(f"anchor te shape {te_anchor_513.shape}")
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Load X_117 substrate ----
    if not X117_UNB_PATH.exists() or not X117_TE_PATH.exists():
        raise FileNotFoundError(
            f"X_117 cache missing: {X117_UNB_PATH} or {X117_TE_PATH}"
        )
    X117_unb = np.load(X117_UNB_PATH).astype(np.float32)
    X117_te = np.load(X117_TE_PATH).astype(np.float32)
    if X117_unb.shape != (n_unb, 117):
        raise ValueError(f"X117_unb shape {X117_unb.shape} expected ({n_unb},117)")
    if X117_te.shape != (n_test, 117):
        raise ValueError(f"X117_te shape {X117_te.shape} expected ({n_test},117)")
    # Sanitize NaN/Inf carried in cache
    X117_unb = np.where(np.isfinite(X117_unb), X117_unb, 0.0).astype(np.float32)
    X117_te = np.where(np.isfinite(X117_te), X117_te, 0.0).astype(np.float32)
    print(f"[feat] X117_unb = {X117_unb.shape}  X117_te = {X117_te.shape}")

    # ---- Slice K=20 columns from nb2240 RFE ----
    if not NB2240_SUMMARY.exists():
        raise FileNotFoundError(f"nb2240 summary missing: {NB2240_SUMMARY}")
    with open(NB2240_SUMMARY) as f:
        nb2240 = json.load(f)
    k20_idx = list(nb2240["k20_surviving_idx_in_117"])
    k20_names = list(nb2240["k20_surviving_names"])
    assert len(k20_idx) == 20, f"expected 20 K=20 indices, got {len(k20_idx)}"
    print(f"[K20] loaded {len(k20_idx)} surviving indices from nb2240")

    X_unb = X117_unb[:, k20_idx].astype(np.float32)
    X_te = X117_te[:, k20_idx].astype(np.float32)
    feat_dim = X_unb.shape[1]
    assert feat_dim == 20, f"feat_dim {feat_dim} != 20"
    print(f"[feat] X_unb (K=20) = {X_unb.shape}  X_te = {X_te.shape}")

    # quick diagnostic: how many K=20 columns are heavy-tailed?
    iqr_per_col = (
        np.quantile(X_unb, ROBUST_Q_HIGH / 100.0, axis=0)
        - np.quantile(X_unb, ROBUST_Q_LOW / 100.0, axis=0)
    )
    std_per_col = X_unb.std(axis=0)
    # Use safe division: where IQR=0, ratio is undefined; treat as 1.0
    iqr_safe = np.where(iqr_per_col > 1e-9, iqr_per_col, 1.0)
    std_to_iqr = std_per_col / iqr_safe
    print(f"[feat] std/IQR per col: "
          f"median={np.median(std_to_iqr):.3f}  "
          f"max={std_to_iqr.max():.3f}  "
          f"(ratio>>1.349 indicates heavy tails vs normal)")

    # ---- Scaffolds ----
    unb_smiles = [test_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique unb scaffolds = {n_unique_scaf}")

    # ---- 5-seed scaffold CV ----
    print("\n" + "-" * 78)
    print(f"5-SEED SCAFFOLD-CV  seeds={KF_SEEDS}  folds={N_FOLDS}  dim={feat_dim}")
    print("-" * 78)
    per_seed_oof_resid = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)
    per_seed_te_resid = np.zeros((len(KF_SEEDS), n_test), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_fold_diags: list[list[dict]] = []
    per_seed_deploy_diag: list[dict] = []

    for i, seed in enumerate(KF_SEEDS):
        ts = time.time()
        resid_oof, fold_diags = _scaffold_cv_one_seed(
            X_unb, residual, unb_scaffolds, seed,
        )
        per_seed_oof_resid[i] = resid_oof
        per_seed_fold_diags.append(fold_diags)
        te_resid, deploy_diag = _deploy_te_one_seed(
            X_unb, residual, X_te, seed,
        )
        per_seed_te_resid[i] = te_resid
        per_seed_deploy_diag.append(deploy_diag)
        pred_corr = anchor + resid_oof
        rae_s = float(rae(y_unb, pred_corr))
        per_seed_rae.append(rae_s)
        print(f"   seed={seed}  rae_corr={rae_s:.4f}  "
              f"d_vs_anchor={rae_s - rae_anchor:+.4f}  "
              f"scale_med~{deploy_diag['scale_median']:.3f}  "
              f"wall={time.time() - ts:.1f}s")

    per_seed_mean = float(np.mean(per_seed_rae))
    per_seed_std = float(np.std(per_seed_rae))
    mean_bag_resid = per_seed_oof_resid.mean(axis=0)
    median_bag_resid = np.median(per_seed_oof_resid, axis=0)
    rae_mean_bag = float(rae(y_unb, anchor + mean_bag_resid))
    rae_median_bag = float(rae(y_unb, anchor + median_bag_resid))

    print("\n[cv] per_seed_mean RAE = "
          f"{per_seed_mean:.4f}  std={per_seed_std:.4f}")
    print(f"[cv] mean_bag   RAE = {rae_mean_bag:.4f}")
    print(f"[cv] median_bag RAE = {rae_median_bag:.4f}")
    print(f"[cv] anchor     RAE = {rae_anchor:.4f}  "
          f"(d_mean_bag = {rae_mean_bag - rae_anchor:+.4f})")
    print(f"[cv] reference  nb2240 K=20 StandardScaler+LGBM = "
          f"{NB2240_K20_REF:.4f}  "
          f"(d = {rae_mean_bag - NB2240_K20_REF:+.4f})")

    # ---- Deploy te (mean-bag corrected) ----
    mean_bag_te_resid = per_seed_te_resid.mean(axis=0)
    te_deploy = (te_anchor_513 + mean_bag_te_resid).astype(np.float32)
    te_unb_in_sample_rae = float(rae(y_unb, te_deploy[unb_idx]))
    print(f"\n[deploy] te(513) mean/std = "
          f"{te_deploy.mean():.3f}/{te_deploy.std():.3f}")
    print(f"[deploy] te[unb_idx] in-sample RAE = {te_unb_in_sample_rae:.4f}  "
          f"(deploy refit, in-sample optimism expected)")

    # ---- Save artefacts ----
    pred_oof_corrected = (anchor + mean_bag_resid).astype(np.float32)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof_corrected)
    np.save(te_path, te_deploy)
    print(f"[save] {oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_robust_scaler_k20.csv"
    pd.DataFrame({
        "SMILES": test_smiles,
        "Molecule Name": te_names,
        "pEC50": te_deploy.astype(np.float32),
    }).to_csv(sub_csv, index=False)
    print(f"[save] {sub_csv}")

    # ---- Gate ----
    if rae_mean_bag < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif rae_mean_bag < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "=" * 78)
    print("GATE EVALUATION")
    print("=" * 78)
    print(f"   mean_bag_rae        = {rae_mean_bag:.4f}")
    print(f"   < {GATE_PROMOTE:.4f}  (PROMOTE)       = "
          f"{rae_mean_bag < GATE_PROMOTE}")
    print(f"   < {GATE_MARGINAL:.4f}  (MARGINAL_BEAT) = "
          f"{rae_mean_bag < GATE_MARGINAL}")
    print(f"   VERDICT             = {verdict}")

    summary = {
        "tag": TAG,
        "method": "robust_scaler_K20_LGBM_residual_on_chemprop_aux",
        "rationale": (
            "RobustScaler (median/IQR) vs StandardScaler (mean/std) on "
            "K=20 substrate; preprocessing-only swap on fixed anchor+model"
        ),
        "anchor": ANCHOR,
        "anchor_te_path": str(ANCHOR_TE_PATH),
        "anchor_pre_unblind": True,
        "rae_anchor_chemprop_aux": rae_anchor,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "x117_unb_path": str(X117_UNB_PATH),
        "x117_te_path": str(X117_TE_PATH),
        "k20_idx_source": str(NB2240_SUMMARY),
        "k20_surviving_idx_in_117": [int(j) for j in k20_idx],
        "k20_surviving_names": k20_names,
        "n_unb": n_unb,
        "n_test": n_test,
        "n_unique_scaffolds": n_unique_scaf,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "feat_dim": int(feat_dim),
        "model_class": "lightgbm.LGBMRegressor",
        "lgbm_params_sample": _lgbm_params(KF_SEEDS[0]),
        "scaler": {
            "class": "sklearn.preprocessing.RobustScaler",
            "quantile_range_low": ROBUST_Q_LOW,
            "quantile_range_high": ROBUST_Q_HIGH,
            "with_centering": True,
            "with_scaling": True,
            "fit_policy": "per-fold-on-train",
        },
        "k20_std_to_iqr_median": float(np.median(std_to_iqr)),
        "k20_std_to_iqr_max": float(std_to_iqr.max()),
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_seed_rae": [float(r) for r in per_seed_rae],
        "per_seed_mean_rae": per_seed_mean,
        "per_seed_std_rae": per_seed_std,
        "mean_bag_rae": rae_mean_bag,
        "median_bag_rae": rae_median_bag,
        "mean_rae": rae_mean_bag,  # alias for gate consumers
        "delta_vs_anchor": rae_mean_bag - rae_anchor,
        "delta_vs_nb2240_K20_lgbm": rae_mean_bag - NB2240_K20_REF,
        "nb2240_K20_lgbm_ref": NB2240_K20_REF,
        "per_seed_fold_diagnostics": per_seed_fold_diags,
        "per_seed_deploy_diagnostics": per_seed_deploy_diag,
        "te_unb_in_sample_rae": te_unb_in_sample_rae,
        "te_deploy_mean": float(te_deploy.mean()),
        "te_deploy_std": float(te_deploy.std()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv),
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "per_seed_rae",
        "per_seed_mean_rae",
        "per_seed_std_rae",
        "mean_bag_rae",
        "median_bag_rae",
        "delta_vs_anchor",
        "delta_vs_nb2240_K20_lgbm",
        "te_unb_in_sample_rae",
        "k20_std_to_iqr_median",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
