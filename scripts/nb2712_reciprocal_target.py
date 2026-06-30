"""nb2712 -- Reciprocal pEC50 transform target on K=20 substrate.

NEW PARADIGM:
    All prior K=20 substrate experiments train LGBM on the
    chemprop_aux residual `y_unb - anchor`, then add the anchor back to
    produce a corrected pEC50.  Here we change the TARGET TRANSFORM
    entirely: predict the reciprocal `y_inv = 1 / max(y, 0.1)` of the
    full pEC50 target (not residual), then invert at inference time.

    The reciprocal transform reshapes the loss landscape:
      - High pEC50 (potent compounds) maps to small y_inv, so squared
        error in y_inv space corresponds to RELATIVE error in y space
        for potent rows -- the same rows that drive RAE through their
        large absolute residuals on the original axis.
      - Low pEC50 (inactive compounds, the F2 greasy-novel-inactive
        failure mode) maps to large y_inv (up to 10.0 with the 0.1
        floor), so the model's loss surface is dominated by getting
        the *inactives* right, which is the opposite weighting of MSE
        on raw pEC50.

    Hypothesis: by inverting the loss-axis weighting, the model may
    learn a different objective shape than the long-running
    chemprop_aux-residual paradigm.  Independent of all post-hoc
    blends, this is an alternative-anchor candidate -- a substrate
    change in the cycle-134 sense.  Trains on FULL target (not
    residual), so does NOT inherit the chemprop_aux PRE-unblind
    anchor; this is a standalone non-anchor estimator.

PROTOCOL:
    1. Compute y_inv = 1 / max(y_unb, 0.1)  (n=253)
       and y_inv_te = 1 / max(y_te_train_proxy, 0.1) is not needed --
       deploy fits on (X_unb, y_inv), predicts y_inv_hat on X_te.
    2. Per scaffold-fold (5 folds, 5 kf_seeds {1001..1005}):
       - LGBM fit on K=20 features with target = y_inv  (full target,
         NOT residual; no anchor subtraction).
       - Predict y_inv_va_hat on fold-val.
       - Invert: y_va_hat = 1 / max(y_inv_va_hat, 0.1).
       - Pool to pred_oof; compute pooled RAE against y_unb.
    3. Average pooled_rae across 5 kf_seeds.
    4. Deploy te: refit LGBM on full 253 (y_inv), predict 513 y_inv_te;
       invert -> te_pred_513.  Save submission CSV.

GATE (mean pooled RAE across 5 kf_seeds):
    mean_rae < 0.4570 -> "PROMOTE"
    mean_rae < 0.4598 -> "MARGINAL_BEAT"
    else              -> "FAIL"

OUTPUTS:
    scripts/nb2712_reciprocal_target.py
    data/processed/nb2712_summary.json
    data/processed/nb2712_pred_oof.npy   (253,) float32 mean-bag pred
    data/processed/te_nb2712.npy         (513,) float32 deploy refit
    submissions/nb2712_reciprocal_target.csv
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
import pandas as pd
import lightgbm as lgb
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2712"

# ---- Anchor only used for diagnostic comparison (not as residual base) ----
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"
NB2240_SUMMARY = DATA_PROCESSED / "nb2240_summary.json"

N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# Reciprocal target floor: y_inv = 1/max(y, Y_FLOOR), and we invert
# y_inv_hat back via 1/max(y_inv_hat, Y_FLOOR).
Y_FLOOR = 0.1

# Gate thresholds (mean across kf_seeds)
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

CHEMPROP_AUX_REF = 0.6216
NB2240_K20_REF = 0.4630  # StandardScaler+LGBM K=20 residual reference
NB2171_REF = 0.4682


def _lgbm_params(seed: int) -> dict:
    """LGBM hyperparams matching nb2103/nb2240/nb2700 family on K=20."""
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


def reciprocal_forward(y: np.ndarray) -> np.ndarray:
    """y -> y_inv = 1 / max(y, Y_FLOOR)."""
    return (1.0 / np.maximum(y.astype(np.float64), Y_FLOOR)).astype(np.float64)


def reciprocal_inverse(y_inv: np.ndarray) -> np.ndarray:
    """y_inv_hat -> y_hat = 1 / max(y_inv_hat, Y_FLOOR)."""
    return (1.0 / np.maximum(y_inv.astype(np.float64), Y_FLOOR)).astype(np.float64)


def _scaffold_cv_one_seed(
    X: np.ndarray,
    y: np.ndarray,
    y_inv: np.ndarray,
    unb_scaffolds: list,
    kf_seed: int,
) -> tuple[np.ndarray, list[dict]]:
    """One scaffold-CV pass: fit per-fold LGBM on y_inv, invert prediction.

    Returns (oof_pred_y, per_fold_diagnostics).
    """
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n = len(y_inv)
    oof_y = np.full(n, np.nan, dtype=np.float64)
    fold_diags = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        mdl = lgb.LGBMRegressor(**_lgbm_params(kf_seed))
        mdl.fit(X[tr_loc], y_inv[tr_loc])
        y_inv_va_hat = mdl.predict(X[va_loc])
        y_va_hat = reciprocal_inverse(y_inv_va_hat)
        oof_y[va_loc] = y_va_hat
        fold_diags.append({
            "fold": fold_i,
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "y_inv_tr_mean": float(y_inv[tr_loc].mean()),
            "y_inv_tr_std": float(y_inv[tr_loc].std()),
            "y_inv_va_hat_mean": float(y_inv_va_hat.mean()),
            "y_inv_va_hat_min": float(y_inv_va_hat.min()),
            "y_inv_va_hat_max": float(y_inv_va_hat.max()),
            "y_va_hat_mean": float(y_va_hat.mean()),
            "y_va_hat_std": float(y_va_hat.std()),
            "fold_rae": float(rae(y[va_loc], y_va_hat)),
        })
    if np.isnan(oof_y).any():
        raise RuntimeError("scaffold split did not cover all rows")
    return oof_y, fold_diags


def _deploy_te_one_seed(
    X_unb: np.ndarray,
    y_inv: np.ndarray,
    X_te: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Fit LGBM on full 253 (y_inv target); predict 513 y_inv_te; invert."""
    mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
    mdl.fit(X_unb, y_inv)
    y_inv_te_hat = mdl.predict(X_te)
    y_te_hat = reciprocal_inverse(y_inv_te_hat).astype(np.float32)
    diag = {
        "seed": int(seed),
        "y_inv_te_hat_mean": float(y_inv_te_hat.mean()),
        "y_inv_te_hat_min": float(y_inv_te_hat.min()),
        "y_inv_te_hat_max": float(y_inv_te_hat.max()),
        "y_te_hat_mean": float(y_te_hat.mean()),
        "y_te_hat_std": float(y_te_hat.std()),
        "y_te_hat_min": float(y_te_hat.min()),
        "y_te_hat_max": float(y_te_hat.max()),
    }
    return y_te_hat, diag


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Reciprocal target transform y_inv=1/max(y,{Y_FLOOR}) "
          f"on K=20 substrate")
    print(f"        TRAINS ON FULL TARGET (NOT chemprop_aux residual) -- "
          f"standalone non-anchor estimator")
    print(f"        scaffold-CV {N_FOLDS}-fold  kf_seeds={KF_SEEDS}")
    print(f"        ref nb2240 K=20 (residual) = {NB2240_K20_REF:.4f}  "
          f"ref nb2171 (pyramid) = {NB2171_REF:.4f}")
    print(f"        GATE: <{GATE_PROMOTE} PROMOTE / "
          f"<{GATE_MARGINAL} MARGINAL_BEAT / else FAIL")
    print("=" * 78)

    # ---- Load truth + scaffolds ----
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

    # Diagnostic-only: chemprop_aux anchor for delta computation
    te_anchor_513 = (
        np.load(ANCHOR_TE_PATH).astype(np.float64)
        if ANCHOR_TE_PATH.exists() else None
    )
    if te_anchor_513 is not None:
        anchor = te_anchor_513[unb_idx]
        rae_anchor = float(rae(y_unb, anchor))
        print(f"[diag] chemprop_aux te[unb_idx] in_RAE = {rae_anchor:.4f}  "
              f"(diagnostic only, not used in model)")
    else:
        anchor = None
        rae_anchor = float("nan")

    # ---- Build reciprocal target ----
    y_inv = reciprocal_forward(y_unb)
    print(f"[target] y_unb         mean={y_unb.mean():.4f}  "
          f"min={y_unb.min():.4f}  max={y_unb.max():.4f}  "
          f"std={y_unb.std():.4f}")
    print(f"[target] y_inv (recip) mean={y_inv.mean():.4f}  "
          f"min={y_inv.min():.4f}  max={y_inv.max():.4f}  "
          f"std={y_inv.std():.4f}")
    n_floored = int((y_unb < Y_FLOOR).sum())
    print(f"[target] n rows hitting floor (y<{Y_FLOOR}) = {n_floored}")

    # ---- Load X_117 substrate ----
    if not X117_UNB_PATH.exists() or not X117_TE_PATH.exists():
        raise FileNotFoundError(
            f"X_117 cache missing: {X117_UNB_PATH} or {X117_TE_PATH}"
        )
    X117_unb = np.load(X117_UNB_PATH).astype(np.float32)
    X117_te = np.load(X117_TE_PATH).astype(np.float32)
    if X117_unb.shape != (n_unb, 117):
        raise ValueError(
            f"X117_unb shape {X117_unb.shape} expected ({n_unb},117)"
        )
    if X117_te.shape != (n_test, 117):
        raise ValueError(
            f"X117_te shape {X117_te.shape} expected ({n_test},117)"
        )
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

    # ---- Scaffolds ----
    unb_smiles = [test_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique unb scaffolds = {n_unique_scaf}")

    # ---- 5-seed scaffold CV ----
    print("\n" + "-" * 78)
    print(f"5-SEED SCAFFOLD-CV  seeds={KF_SEEDS}  folds={N_FOLDS}  dim={feat_dim}")
    print(f"target=reciprocal  invert= 1/max(y_inv_hat,{Y_FLOOR})")
    print("-" * 78)

    per_seed_oof_y = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)
    per_seed_te_y = np.zeros((len(KF_SEEDS), n_test), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_fold_diags: list[list[dict]] = []
    per_seed_deploy_diag: list[dict] = []

    for i, seed in enumerate(KF_SEEDS):
        ts = time.time()
        oof_y, fold_diags = _scaffold_cv_one_seed(
            X_unb, y_unb, y_inv, unb_scaffolds, seed,
        )
        per_seed_oof_y[i] = oof_y
        per_seed_fold_diags.append(fold_diags)
        te_y, deploy_diag = _deploy_te_one_seed(
            X_unb, y_inv, X_te, seed,
        )
        per_seed_te_y[i] = te_y
        per_seed_deploy_diag.append(deploy_diag)
        rae_s = float(rae(y_unb, oof_y))
        per_seed_rae.append(rae_s)
        print(f"   seed={seed}  pooled_RAE={rae_s:.4f}  "
              f"y_hat_mean={oof_y.mean():.3f}  "
              f"y_hat_std={oof_y.std():.3f}  "
              f"wall={time.time() - ts:.1f}s")

    per_seed_mean = float(np.mean(per_seed_rae))
    per_seed_std = float(np.std(per_seed_rae))
    mean_bag_oof = per_seed_oof_y.mean(axis=0)
    median_bag_oof = np.median(per_seed_oof_y, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))

    print("\n[cv] per_seed_mean RAE = "
          f"{per_seed_mean:.4f}  std={per_seed_std:.4f}")
    print(f"[cv] mean_bag   RAE = {rae_mean_bag:.4f}")
    print(f"[cv] median_bag RAE = {rae_median_bag:.4f}")
    if not np.isnan(rae_anchor):
        print(f"[diag] chemprop_aux RAE = {rae_anchor:.4f}  "
              f"(d_per_seed_mean = {per_seed_mean - rae_anchor:+.4f})")

    # ---- Deploy te (mean-bag aggregate of 5 seeds) ----
    mean_bag_te = per_seed_te_y.mean(axis=0).astype(np.float32)
    te_unb_in_sample_rae = float(rae(y_unb, mean_bag_te[unb_idx]))
    print(f"\n[deploy] te(513) mean/std = "
          f"{mean_bag_te.mean():.3f}/{mean_bag_te.std():.3f}  "
          f"min={mean_bag_te.min():.3f}  max={mean_bag_te.max():.3f}")
    print(f"[deploy] te[unb_idx] in-sample RAE = {te_unb_in_sample_rae:.4f}  "
          f"(deploy refit, in-sample optimism expected)")

    # ---- Save artefacts ----
    pred_oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(pred_oof_path, mean_bag_oof.astype(np.float32))
    np.save(te_path, mean_bag_te)
    print(f"[save] {pred_oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_reciprocal_target.csv"
    pd.DataFrame({
        "SMILES": test_smiles,
        "Molecule Name": te_names,
        "pEC50": mean_bag_te.astype(np.float32),
    }).to_csv(sub_csv, index=False)
    print(f"[save] {sub_csv}")

    # ---- Gate (use per_seed_mean -- "mean_rae" in spec) ----
    mean_rae = per_seed_mean
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "=" * 78)
    print("GATE EVALUATION")
    print("=" * 78)
    print(f"   mean_rae (per-seed mean) = {mean_rae:.4f}")
    print(f"   mean_bag_rae             = {rae_mean_bag:.4f}")
    print(f"   < {GATE_PROMOTE:.4f}  (PROMOTE)       = {mean_rae < GATE_PROMOTE}")
    print(f"   < {GATE_MARGINAL:.4f}  (MARGINAL_BEAT) = {mean_rae < GATE_MARGINAL}")
    print(f"   VERDICT                  = {verdict}")

    summary = {
        "tag": TAG,
        "method": "reciprocal_target_K20_LGBM_full_target",
        "rationale": (
            "Predict y_inv = 1/max(y,0.1) on full target (not "
            "chemprop_aux residual); invert at inference. Loss-axis "
            "weighting flipped vs MSE-on-pEC50: small y_inv = potent "
            "compounds, large y_inv = inactive compounds."
        ),
        "paradigm": "target_transform_reciprocal_full_target",
        "is_residual_model": False,
        "anchor": None,
        "anchor_pre_unblind": None,
        "anchor_te_path": str(ANCHOR_TE_PATH),
        "rae_anchor_chemprop_aux_diagnostic": rae_anchor,
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
        "y_floor": Y_FLOOR,
        "n_rows_hit_floor": n_floored,
        "y_unb_stats": {
            "mean": float(y_unb.mean()),
            "std": float(y_unb.std()),
            "min": float(y_unb.min()),
            "max": float(y_unb.max()),
        },
        "y_inv_stats": {
            "mean": float(y_inv.mean()),
            "std": float(y_inv.std()),
            "min": float(y_inv.min()),
            "max": float(y_inv.max()),
        },
        "per_seed_rae": [float(r) for r in per_seed_rae],
        "per_seed_mean_rae": per_seed_mean,
        "per_seed_std_rae": per_seed_std,
        "mean_rae": mean_rae,  # gate field
        "mean_bag_rae": rae_mean_bag,
        "median_bag_rae": rae_median_bag,
        "delta_vs_chemprop_aux": (
            mean_rae - rae_anchor if not np.isnan(rae_anchor) else None
        ),
        "delta_vs_nb2240_K20_residual": mean_rae - NB2240_K20_REF,
        "delta_vs_nb2171": mean_rae - NB2171_REF,
        "nb2240_K20_residual_ref": NB2240_K20_REF,
        "nb2171_ref": NB2171_REF,
        "per_seed_fold_diagnostics": per_seed_fold_diags,
        "per_seed_deploy_diagnostics": per_seed_deploy_diag,
        "te_unb_in_sample_rae": te_unb_in_sample_rae,
        "te_deploy_mean": float(mean_bag_te.mean()),
        "te_deploy_std": float(mean_bag_te.std()),
        "te_deploy_min": float(mean_bag_te.min()),
        "te_deploy_max": float(mean_bag_te.max()),
        "pred_oof_path": str(pred_oof_path),
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
        "mean_rae",
        "mean_bag_rae",
        "median_bag_rae",
        "delta_vs_nb2240_K20_residual",
        "delta_vs_nb2171",
        "te_unb_in_sample_rae",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
