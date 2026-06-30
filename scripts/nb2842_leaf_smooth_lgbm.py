"""nb2842 -- LGBM with path_smooth + boost_from_average=False.

NEW PARADIGM (vs the cycle-134 paradigm-exhaustion thesis on the
chemprop_aux + K=20 anchor):
    Two orthogonal LGBM regularization knobs that have never been
    combined in this codebase's ~2842 attempted methods:

      1. path_smooth=0.5 -- smooths each leaf prediction toward its
         parent node's prediction. Higher values -> more shrinkage of
         deep leaves toward shallow-ancestor estimates. Acts as a
         leaf-level Bayesian prior with the parent as the prior mean.
         Effect: deep-tree overfit on noisy n=253 residual is damped
         by hierarchical smoothing along the tree path.

      2. boost_from_average=False -- normally LGBM initialises every
         leaf prediction at iter 0 to the global y_mean, then boosts
         residuals from that constant. With this flag off, iter 0 is
         a 0-vector init, so the first tree carves splits from the
         raw target instead of from (y - mean). For a *residual*
         target (which already has mean ~0 by construction), this
         removes a redundant mean-shift and lets the first tree's
         split decisions key off the actual residual signal rather
         than the constant-bias signal.

    Combined, the two flags target different stages: boost_from_average
    fixes the init, path_smooth fixes the leaf updates. The combination
    is orthogonal to bagging (nb2202/nb2240/nb2270/nb2281), trajectory
    averaging (nb2834), and feature-subsampling (nb2832).

PROTOCOL:
    - Substrate identical to nb2240_K20: K=20 SHAP-pruned features from
      the X_117 pyramid cache (k20_surviving_idx_in_117 from nb2240).
    - Target: chemprop_aux residual on 253 unblind.
    - LGBM(path_smooth=0.5, boost_from_average=False, max_depth=4,
           num_leaves=15, n_estimators=300, learning_rate=0.03).
    - 5-fold scaffold CV on 253 unblind, kf_seeds {1001..1005}.
    - Deploy: refit on all 253, predict te(513), add anchor.

GATE:
    mean_rae < 0.4570  -> "PROMOTE"
    mean_rae < 0.4598  -> "MARGINAL_BEAT"
    else                -> "FAIL"

OUTPUTS:
    scripts/nb2842_leaf_smooth_lgbm.py
    data/processed/nb2842_summary.json
    data/processed/nb2842_pred_oof.npy   (253,) float32 mean-bag CORRECTED
    data/processed/te_nb2842.npy         (513,) float32 deploy refit
    submissions/nb2842_leaf_smooth_lgbm.csv
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

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

TAG = "nb2842"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"
NB2240_SUMMARY = DATA_PROCESSED / "nb2240_summary.json"

N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# LGBM hyperparams per protocol
LGBM_PARAMS = dict(
    objective="regression",
    metric="rmse",
    path_smooth=0.5,             # NEW: smooth toward parent node
    boost_from_average=False,    # NEW: zero-init instead of y_mean init
    max_depth=4,
    num_leaves=15,
    n_estimators=300,
    learning_rate=0.03,
    min_data_in_leaf=10,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=5,
    lambda_l2=0.1,
    verbose=-1,
)
LGBM_SEED = 1001

# Gate thresholds
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

# Cross-references
CHEMPROP_AUX_REF = 0.6216
NB2240_K20_REF = 0.4630


def _build_lgbm(X_tr: np.ndarray, y_tr: np.ndarray) -> lgb.Booster:
    """Train one LGBM with the protocol hyperparams."""
    params = dict(LGBM_PARAMS)
    params["seed"] = LGBM_SEED
    params["bagging_seed"] = LGBM_SEED
    params["feature_fraction_seed"] = LGBM_SEED
    n_est = int(params.pop("n_estimators"))
    dtr = lgb.Dataset(X_tr, label=y_tr)
    booster = lgb.train(
        params,
        dtr,
        num_boost_round=n_est,
    )
    return booster


def _scaffold_cv_one_seed(
    X_unb: np.ndarray,
    residual: np.ndarray,
    unb_scaffolds: list,
    kf_seed: int,
) -> tuple[np.ndarray, list[dict]]:
    """One scaffold-CV pass on residual."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n = len(residual)
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_diags = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        b = _build_lgbm(X_unb[tr_loc], residual[tr_loc])
        va_p = b.predict(X_unb[va_loc])
        oof[va_loc] = va_p
        fold_diags.append({
            "fold": int(fold_i),
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "va_mae_resid": float(np.mean(np.abs(va_p - residual[va_loc]))),
        })
    if np.isnan(oof).any():
        raise RuntimeError("scaffold split did not cover all rows")
    return oof, fold_diags


def _deploy_te_one_seed(
    X_unb: np.ndarray, residual: np.ndarray, X_te: np.ndarray,
) -> np.ndarray:
    """Refit on all 253, predict te(513)."""
    b = _build_lgbm(X_unb, residual)
    return b.predict(X_te).astype(np.float32)


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- LGBM (path_smooth + boost_from_average=False) on "
          f"{ANCHOR} residual")
    print(f"        path_smooth=0.5  boost_from_average=False")
    print(f"        max_depth=4  num_leaves=15  n_est=300  lr=0.03")
    print(f"        scaffold-CV {N_FOLDS}-fold  kf_seeds={KF_SEEDS}")
    print(f"        ref nb2240 K=20 LGBM = {NB2240_K20_REF:.4f}")
    print(f"        GATE: <{GATE_PROMOTE} PROMOTE / "
          f"<{GATE_MARGINAL} MARGINAL_BEAT / else FAIL")
    print("=" * 78)

    # ---- Load truth + anchor ----
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

    # ---- Load X_117 substrate + slice K=20 ----
    if not X117_UNB_PATH.exists() or not X117_TE_PATH.exists():
        raise FileNotFoundError(
            f"X_117 cache missing: {X117_UNB_PATH} or {X117_TE_PATH}"
        )
    X117_unb = np.load(X117_UNB_PATH).astype(np.float32)
    X117_te = np.load(X117_TE_PATH).astype(np.float32)
    X117_unb = np.where(np.isfinite(X117_unb), X117_unb, 0.0).astype(np.float32)
    X117_te = np.where(np.isfinite(X117_te), X117_te, 0.0).astype(np.float32)

    if not NB2240_SUMMARY.exists():
        raise FileNotFoundError(f"nb2240 summary missing: {NB2240_SUMMARY}")
    with open(NB2240_SUMMARY) as f:
        nb2240 = json.load(f)
    k20_idx = list(nb2240["k20_surviving_idx_in_117"])
    k20_names = list(nb2240["k20_surviving_names"])
    assert len(k20_idx) == 20, f"expected 20 K=20 indices, got {len(k20_idx)}"
    X_unb = X117_unb[:, k20_idx].astype(np.float32)
    X_te = X117_te[:, k20_idx].astype(np.float32)
    feat_dim = X_unb.shape[1]
    print(f"[feat] X_unb (K=20) = {X_unb.shape}  X_te = {X_te.shape}")

    # ---- Scaffolds for CV splitter ----
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

    for i, seed in enumerate(KF_SEEDS):
        ts = time.time()
        resid_oof, fold_diags = _scaffold_cv_one_seed(
            X_unb, residual, unb_scaffolds, seed,
        )
        per_seed_oof_resid[i] = resid_oof
        per_seed_fold_diags.append(fold_diags)
        te_resid = _deploy_te_one_seed(X_unb, residual, X_te)
        per_seed_te_resid[i] = te_resid
        pred_corr = anchor + resid_oof
        rae_s = float(rae(y_unb, pred_corr))
        per_seed_rae.append(rae_s)
        print(f"   seed={seed}  rae_corr={rae_s:.4f}  "
              f"d_vs_anchor={rae_s - rae_anchor:+.4f}  "
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
    print(f"[cv] reference  nb2240 K=20 LGBM = {NB2240_K20_REF:.4f}  "
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

    sub_csv = SUBMISSIONS / f"{TAG}_leaf_smooth_lgbm.csv"
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
        "method": "LGBM_path_smooth_05_boost_from_avg_off_chemprop_aux_residual",
        "rationale": (
            "Two orthogonal LGBM regularization knobs combined for the "
            "first time on this anchor: path_smooth=0.5 smooths each leaf "
            "toward its parent node (hierarchical Bayesian shrinkage on "
            "deep splits), and boost_from_average=False removes the y_mean "
            "constant init so iter 0 starts from 0 instead of mean (more "
            "honest for a residual target whose mean is ~0 by construction)."
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
        "lgbm_params": LGBM_PARAMS,
        "lgbm_boosting_seed": LGBM_SEED,
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
        "te_unb_in_sample_rae": te_unb_in_sample_rae,
        "te_deploy_mean": float(te_deploy.mean()),
        "te_deploy_std": float(te_deploy.std()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv),
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   mean_bag RAE        = {rae_mean_bag:.4f}  ({verdict})")
    print(f"   per_seed mean       = {per_seed_mean:.4f} +/- {per_seed_std:.4f}")
    print(f"   delta vs anchor     = {rae_mean_bag - rae_anchor:+.4f}")
    print(f"   delta vs nb2240_K20 = {rae_mean_bag - NB2240_K20_REF:+.4f}")
    print(f"   wall                = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_bag_rae",
        "per_seed_mean_rae",
        "per_seed_std_rae",
        "verdict",
        "delta_vs_anchor",
        "delta_vs_nb2240_K20_lgbm",
        "te_unb_in_sample_rae",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
