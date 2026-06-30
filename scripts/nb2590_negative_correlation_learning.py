"""nb2590 -- Negative Correlation Learning (NCL) ensemble on chemprop_aux residual.

NEW PARADIGM
------------
Train K=5 LGBM members sequentially where each new member is *anti*-correlated
with the previously trained members.  Standard ensembles minimise each member's
own loss; NCL adds a penalty that *rewards* diverse error patterns, so the bag
mean averages out more error variance than a homogeneous bag of MSE-LGBMs.

LOSS PER MEMBER i
-----------------
    L_i(p_i) = MSE(y, p_i) + lambda * sum_{j<i} corr(resid_i, resid_j)

Approximation (paper Liu & Yao 1999):
    gradient adjusted by  - 2 * lambda * (resid_i - mean(resid_others))

So at training time we use a custom LightGBM objective that, given the current
predictions p_i, returns:
    g = (p_i - y) - lambda * (resid_others_mean - (p_i - y))
      = (1 + lambda) * (p_i - y) - lambda * resid_others_mean
    h = (1 + lambda)                                         (constant Hessian)

where resid_others_mean = mean_{j<i}(resid_j on TRAIN rows) is pre-computed per
fold and fixed during the boosting of member i.

PROTOCOL
--------
* Anchor: chemprop_aux (PRE-unblind clean te file, sliced to 253 unblind rows).
* Features: cached 253x28 matrix `X_unb_28_nb2103.npy` (SHAP top-28 of the
  117-col 5-way K-tuned matrix used by nb2095 / nb2103 / nb2151).
* Member 1: plain MSE (no NCL penalty yet -- no previous residuals).
* Members 2..5: NCL custom objective with lambda=0.5.
* CV: scaffold KFold (5 folds, Murcko skeleton on 253 unblind SMILES, seed 1001).
* Per fold: cross-fit -> OOF residual prediction for each member; bag = simple
  mean of the 5 members' OOF predictions; final = anchor + bag_mean.

DEPLOY te
---------
For each fold-free deploy fit, we use ALL 253 rows: each member's deploy model
is refit on all 253 rows with the NCL gradient anchored to the *deploy-fit
residuals* of preceding members.  Predict on 513 test rows, average the 5
member predictions, add anchor te to get `te_nb2590`.

GATE
----
mean_rae < 0.4570               -> "PROMOTE"
mean_rae < 0.4601               -> "MARGINAL_BEAT"
else                            -> "FAIL"

(0.4601 = nb2103 K=28 random-KFold; 0.4570 = nb2171 deep-30 mean 0.4682 minus
budget headroom; matches the user-specified gates verbatim.)

Outputs
-------
    scripts/nb2590_negative_correlation_learning.py
    data/processed/nb2590_summary.json
    data/processed/nb2590_pred_oof.npy         (253,) float32  -- final OOF
    data/processed/te_nb2590.npy               (513,) float32  -- deploy te
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
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2590"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
X_UNB_28_PATH = DATA_PROCESSED / "X_unb_28_nb2103.npy"

N_MEMBERS = 5
LAMBDA_NCL = 0.5
KF_SEED = 1001
RESID_FOLDS = 5

GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4601

CHEMPROP_AUX_REF = 0.6216


def _murcko_scaffold(mol):
    try:
        if mol is None:
            return None
        sc = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(sc) if sc is not None else None
    except Exception:
        return None


def _lgbm_params(seed: int) -> dict:
    """Same LGBM(MSE) hyperparams as nb2103 K=28 winner."""
    return dict(
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


def _ncl_objective_factory(resid_others_mean: np.ndarray, lam: float):
    """Build an sklearn-API LightGBM custom objective implementing NCL.

    Signature (sklearn wrapper):  objective(y_true, y_pred) -> (grad, hess)

    For prediction p on this train fold:
        resid_i = p - y_train          (sign: model_pred - truth)
        Loss    = 0.5 * resid_i^2 + lam * (resid_i - mean(resid_others_train))
                                              * resid_i      (covariance proxy)
        grad    = (1 + lam) * resid_i - lam * resid_others_mean
        hess    = (1 + lam)                                   (constant)
    """
    one_plus_lam = float(1.0 + lam)
    lam_mu = lam * resid_others_mean.astype(np.float64)

    def _obj(y_true, y_pred):
        resid = y_pred - y_true.astype(np.float64)
        grad = one_plus_lam * resid - lam_mu
        hess = np.full_like(resid, one_plus_lam)
        return grad, hess

    return _obj


def _fit_member_ncl(X_tr: np.ndarray, y_tr: np.ndarray,
                    resid_others_mean_tr: np.ndarray,
                    seed: int, lam: float) -> lgb.LGBMRegressor:
    """Fit one LGBM member with the NCL custom objective."""
    params = _lgbm_params(seed)
    obj = _ncl_objective_factory(resid_others_mean_tr, lam)
    # IMPORTANT: when objective is a callable, do NOT pass objective="regression".
    mdl = lgb.LGBMRegressor(objective=obj, **params)
    mdl.fit(X_tr, y_tr)
    return mdl


def _fit_member_mse(X_tr: np.ndarray, y_tr: np.ndarray,
                    seed: int) -> lgb.LGBMRegressor:
    """Fit the first member with plain MSE."""
    params = _lgbm_params(seed)
    mdl = lgb.LGBMRegressor(objective="regression", **params)
    mdl.fit(X_tr, y_tr)
    return mdl


def _cross_fit_ncl_ensemble(X: np.ndarray, residual: np.ndarray,
                            scaffolds: list[str], seed: int,
                            n_members: int, lam: float
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Run scaffold-CV NCL ensemble. Returns:
        oof_members:  (n_members, n)  per-member OOF prediction of residual
        oof_bag_mean: (n,)            simple mean across members
    """
    n = len(residual)
    splits = scaffold_kfold_indices(scaffolds, n_splits=RESID_FOLDS,
                                    shuffle=True, seed=seed)
    oof_members = np.full((n_members, n), np.nan, dtype=np.float64)

    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        X_tr, X_va = X[tr_loc], X[va_loc]
        y_tr = residual[tr_loc]

        member_seeds = [seed + 100 * (m + 1) for m in range(n_members)]

        # Member 1: plain MSE
        mdl1 = _fit_member_mse(X_tr, y_tr, member_seeds[0])
        train_preds = [mdl1.predict(X_tr)]
        oof_members[0, va_loc] = mdl1.predict(X_va)

        # Members 2..K: NCL
        for m in range(1, n_members):
            # residuals of preceding members on the TRAIN rows of this fold
            resids_prev = np.stack(
                [tp - y_tr for tp in train_preds], axis=0
            )  # (m, n_tr)
            resid_others_mean_tr = resids_prev.mean(axis=0)  # (n_tr,)
            mdl_m = _fit_member_ncl(X_tr, y_tr, resid_others_mean_tr,
                                     member_seeds[m], lam)
            train_preds.append(mdl_m.predict(X_tr))
            oof_members[m, va_loc] = mdl_m.predict(X_va)

    if np.isnan(oof_members).any():
        raise ValueError("NaN remains in oof_members after CV fill")
    oof_bag_mean = oof_members.mean(axis=0)
    return oof_members, oof_bag_mean


def _deploy_fit_ncl_ensemble(X_train: np.ndarray, y_train: np.ndarray,
                             X_test: np.ndarray,
                             seed: int, n_members: int, lam: float
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Refit on ALL training rows; predict 513 test rows. Returns:
        te_members:  (n_members, n_test)
        te_bag_mean: (n_test,)
    """
    n_test = X_test.shape[0]
    te_members = np.zeros((n_members, n_test), dtype=np.float64)

    member_seeds = [seed + 100 * (m + 1) for m in range(n_members)]

    # Member 1
    mdl1 = _fit_member_mse(X_train, y_train, member_seeds[0])
    train_preds = [mdl1.predict(X_train)]
    te_members[0] = mdl1.predict(X_test)

    for m in range(1, n_members):
        resids_prev = np.stack(
            [tp - y_train for tp in train_preds], axis=0
        )
        resid_others_mean_tr = resids_prev.mean(axis=0)
        mdl_m = _fit_member_ncl(X_train, y_train, resid_others_mean_tr,
                                 member_seeds[m], lam)
        train_preds.append(mdl_m.predict(X_train))
        te_members[m] = mdl_m.predict(X_test)

    te_bag_mean = te_members.mean(axis=0)
    return te_members, te_bag_mean


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Negative Correlation Learning ensemble")
    print(f"          anchor={ANCHOR}  K={N_MEMBERS}  lambda={LAMBDA_NCL}")
    print(f"          scaffold KFold seed={KF_SEED}  folds={RESID_FOLDS}")
    print(f"          gates: PROMOTE<{GATE_PROMOTE}  MARGINAL<{GATE_MARGINAL}")
    print("=" * 78)

    # ---- Load anchor + truth ----
    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(
            f"chemprop_aux te shape mismatch: {te_anchor_513.shape} vs {n_test}"
        )
    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")

    residual_unb = y_unb - anchor_unb
    print(f"[resid] mean={residual_unb.mean():+.4f}  "
          f"std={residual_unb.std():.4f}")

    # ---- Load 253x28 cached feature matrix (UNBLIND slice) ----
    if not X_UNB_28_PATH.exists():
        raise FileNotFoundError(f"missing cached features: {X_UNB_28_PATH}")
    X_unb_28 = np.load(X_UNB_28_PATH).astype(np.float32)
    if X_unb_28.shape != (n_unb, 28):
        raise ValueError(
            f"X_unb_28 shape mismatch: {X_unb_28.shape} vs ({n_unb}, 28)"
        )
    print(f"[feat] X_unb_28           = {X_unb_28.shape}")

    # ---- Compute Murcko scaffolds on the 253 unblind SMILES ----
    unb_smiles = [test_smiles[i] for i in unb_idx]
    unb_mols = [standardize(s) for s in unb_smiles]
    scaffolds_unb = [_murcko_scaffold(m) for m in unb_mols]
    from collections import Counter
    sc_counts = Counter(s for s in scaffolds_unb if s)
    n_unique_sc = len(sc_counts)
    n_none_sc = sum(1 for s in scaffolds_unb if not s)
    n_singleton_sc = sum(1 for v in sc_counts.values() if v == 1)
    largest_sc = max(sc_counts.values()) if sc_counts else 0
    print(f"[scaf] n_unb_with_scaf = {n_unb - n_none_sc}  "
          f"n_unique_scaffolds = {n_unique_sc}  "
          f"n_singleton_scaffolds = {n_singleton_sc}  "
          f"largest_scaffold = {largest_sc}")

    splits = scaffold_kfold_indices(scaffolds_unb, n_splits=RESID_FOLDS,
                                    shuffle=True, seed=KF_SEED)
    for f, (tr, va) in enumerate(splits):
        print(f"   [scaf-fold] fold{f}: n_tr={len(tr)}  n_va={len(va)}")

    # ---- Scaffold-CV NCL ensemble (5 members) ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD-CV NCL ENSEMBLE  K={N_MEMBERS}  lambda={LAMBDA_NCL}")
    print("-" * 78)
    t_cv = time.time()
    oof_members, oof_bag_mean = _cross_fit_ncl_ensemble(
        X_unb_28, residual_unb, scaffolds_unb,
        seed=KF_SEED, n_members=N_MEMBERS, lam=LAMBDA_NCL,
    )
    print(f"[cv] NCL CV wall = {time.time() - t_cv:.1f}s")

    pred_corr_bag = anchor_unb + oof_bag_mean
    rae_bag = float(rae(y_unb, pred_corr_bag))

    # Per-member OOF RAE (anchor + that member's residual)
    per_member = []
    for m in range(N_MEMBERS):
        pred_m = anchor_unb + oof_members[m]
        rae_m = float(rae(y_unb, pred_m))
        per_member.append({
            "member": int(m + 1),
            "rae": rae_m,
            "delta_vs_anchor": rae_m - rae_anchor,
            "resid_oof_mean": float(oof_members[m].mean()),
            "resid_oof_std": float(oof_members[m].std()),
        })
        print(f"   member {m+1}:  rae = {rae_m:.4f}  "
              f"(d_vs_anchor = {rae_m - rae_anchor:+.4f})  "
              f"resid_std = {oof_members[m].std():.4f}")

    # Pairwise residual correlation between members (diversity check)
    pair_corr = []
    for i in range(N_MEMBERS):
        for j in range(i + 1, N_MEMBERS):
            r_i = oof_members[i] - residual_unb
            r_j = oof_members[j] - residual_unb
            c = float(np.corrcoef(r_i, r_j)[0, 1])
            pair_corr.append({
                "i": int(i + 1), "j": int(j + 1), "corr": c,
            })
    mean_pair_corr = float(np.mean([p["corr"] for p in pair_corr]))
    print(f"\n   mean pairwise residual corr = {mean_pair_corr:.4f}  "
          f"(lower = more diverse)")

    print(f"\n   bag mean: rae = {rae_bag:.4f}  "
          f"(d_vs_anchor = {rae_bag - rae_anchor:+.4f})")

    # ---- Gate verdict ----
    if rae_bag < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif rae_bag < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"\n   verdict = {verdict}  (gate PROMOTE<{GATE_PROMOTE}  "
          f"MARGINAL<{GATE_MARGINAL})")

    # ---- Save OOF residual + final pred ----
    pred_oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    np.save(pred_oof_path, pred_corr_bag.astype(np.float32))
    print(f"[save] {pred_oof_path}")

    # ---- Deploy fit: use ALL 253 rows, predict 513 test rows ----
    print("\n" + "-" * 78)
    print("DEPLOY FIT (refit on all 253 rows, predict 513)")
    print("-" * 78)
    # For deploy, we need the FULL 513 feature matrix sliced from the same
    # 117-col 5-way K-tuned columns, then top-28.  But the cached 253x28 is
    # UNBLIND-only.  For an honest deploy te we'd need X_te_28; absent that,
    # we still produce a 513-vector by:
    #   - using the cached unblind 253x28 as TRAIN
    #   - looking for X_te_28 on disk (matches nb2103 deploy pattern)
    # If not found, fall back to anchor te + bag of zeros (te = anchor te).
    te_28_path = DATA_PROCESSED / "X_te_28_nb2103.npy"
    if te_28_path.exists():
        X_te_28 = np.load(te_28_path).astype(np.float32)
        if X_te_28.shape != (n_test, 28):
            raise ValueError(
                f"X_te_28 shape mismatch: {X_te_28.shape} vs ({n_test}, 28)"
            )
        print(f"[feat] X_te_28           = {X_te_28.shape}")
        te_members, te_bag_mean = _deploy_fit_ncl_ensemble(
            X_unb_28, residual_unb, X_te_28,
            seed=KF_SEED, n_members=N_MEMBERS, lam=LAMBDA_NCL,
        )
        te_final = te_anchor_513 + te_bag_mean
        deploy_te_built = True
        rae_te_unb_slice = float(rae(y_unb, te_final[unb_idx]))
        print(f"   te[unb_idx] in-sample RAE = {rae_te_unb_slice:.4f}  "
              f"(expected better than OOF -- in-sample optimism)")
    else:
        print(f"   X_te_28 not found at {te_28_path}; "
              f"te falls back to anchor te (residual=0)")
        te_final = te_anchor_513.copy()
        deploy_te_built = False
        rae_te_unb_slice = float(rae(y_unb, te_final[unb_idx]))

    te_out_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_out_path, te_final.astype(np.float32))
    print(f"[save] {te_out_path}")

    summary = {
        "tag": TAG,
        "method": ("negative_correlation_learning_lgbm_K5_lambda0p5_"
                   "scaffoldCV_kfseed1001"),
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "feature_path": str(X_UNB_28_PATH),
        "feat_dim": 28,
        "n_members": N_MEMBERS,
        "lambda_ncl": LAMBDA_NCL,
        "kf_seed": KF_SEED,
        "resid_folds": RESID_FOLDS,
        "n_unb": int(n_unb),
        "n_test": int(n_test),
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual_unb.mean()),
        "residual_std": float(residual_unb.std()),
        "per_member": per_member,
        "pairwise_residual_corr": pair_corr,
        "mean_pairwise_residual_corr": mean_pair_corr,
        "rae_bag_mean_oof": rae_bag,
        "mean_rae": rae_bag,
        "delta_bag_vs_anchor": rae_bag - rae_anchor,
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "deploy_te_built": bool(deploy_te_built),
        "rae_te_unb_slice_insample": rae_te_unb_slice,
        "pred_oof_path": str(pred_oof_path),
        "te_path": str(te_out_path),
        "pre_unblind_clean_anchor": True,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "n_members", "lambda_ncl", "kf_seed",
        "rae_anchor_chemprop_aux",
        "rae_bag_mean_oof", "mean_rae", "delta_bag_vs_anchor",
        "mean_pairwise_residual_corr",
        "verdict", "deploy_te_built",
        "rae_te_unb_slice_insample",
        "pre_unblind_clean_anchor",
    ):
        print(f"  {k}: {res.get(k)}")
    print("\n==== PER-MEMBER ====")
    for r in res["per_member"]:
        print(f"  member {r['member']}:  rae={r['rae']:.4f}  "
              f"d_vs_anchor={r['delta_vs_anchor']:+.4f}  "
              f"resid_std={r['resid_oof_std']:.4f}")
