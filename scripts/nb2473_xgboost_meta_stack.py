"""nb2473 -- XGBoost meta-learner stack (different tree paradigm vs LGBM meta).

CONTEXT:
    Recent meta-stack work (nb1052 LGBM, nb1281, nb1563 etc.) has saturated on
    LGBM-as-meta. Cycle-134/139 closed the LGBM-on-K=28 SHAP axis. We test
    XGBoost as a *different* tree paradigm on a small (4-feature) anchor pool,
    asking whether different default split criteria + L2 regularization extract
    signal LGBM cannot. Per cycle-139 lesson (cross-paradigm rankers fail on
    LGBM downstream), an LGBM meta is paradigm-matched to LGBM bases;
    XGBoost-as-meta is a genuine model-class swap that justifies the test.

SUBSTRATE:
    Meta-features (4 anchors, all on n=253 unblind):
        0. nb2240_mean_bag_oof_K20   (PRE-clean K=20 residual stack)
        1. chemprop_aux (nb1133)      (PRE-clean, 4139-trained)
        2. nb730_pred_oof              (POST -- counter-assay axis)
        3. nb562_pred_oof              (POST -- post-hoc stretch)
    Truth: _audit_unblind_y.npy on 253.
    Deploy meta-features on 513: te_{anchor}.npy.

    nb1191 EXCLUDED -- only te_nb1191[unb_idx] (deploy-refit) is available;
    using it as a "meta-feature" on 253 would inject in-sample contamination
    (te[unb] RAE = 0.263 in-sample vs honest cross-fit unavailable).

PROTOCOL:
    - 5-fold scaffold CV (Bemis-Murcko on unblind smiles).
    - XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05,
                   reg_lambda=1, tree_method='hist', random_state=42).
    - Fit on stacked OOF (4 folds) -> predict held-out fold.
    - Compare to LGBM meta-stack and SLSQP convex on identical pool.
    - Deploy: refit on all 253 with same params -> predict te 513.

GATE:
    mean_rae < 0.4570  -> PROMOTE
    else                -> FAIL
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
import xgboost as xgb
import lightgbm as lgb
from rdkit import RDLogger
from scipy.optimize import minimize

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2473"
N_FOLDS = 5
KF_SEED = 42
GATE = 0.4570

# ---- Anchors ----
ANCHOR_OOF_PATHS = {
    "nb2240":       DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy",
    "chemprop_aux": DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy",
    "nb730":        DATA_PROCESSED / "nb730_pred_oof.npy",
    "nb562":        DATA_PROCESSED / "nb562_pred_oof.npy",
}
ANCHOR_TE_PATHS = {
    "nb2240":       DATA_PROCESSED / "te_nb2240_K20.npy",
    "chemprop_aux": DATA_PROCESSED / "te_chemprop_aux.npy",
    "nb730":        DATA_PROCESSED / "te_nb730.npy",
    "nb562":        DATA_PROCESSED / "te_nb562.npy",
}
ANCHOR_PROVENANCE = {
    "nb2240":       "PRE-clean (K=20 residual stack on chemprop_aux)",
    "chemprop_aux": "PRE-clean (4139 PRE-unblind only)",
    "nb730":        "POST (counter-assay null-ensemble; pred_oof honest cross-fit)",
    "nb562":        "POST (post-hoc rank-stretch; pred_oof honest cross-fit)",
}

# XGBoost hyperparams per spec
XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=2,
    learning_rate=0.05,
    reg_lambda=1.0,
    tree_method="hist",
    random_state=42,
    n_jobs=2,
    verbosity=0,
)

# LGBM comparator (parameter-matched)
LGBM_PARAMS = dict(
    objective="regression",
    n_estimators=200,
    max_depth=2,
    num_leaves=3,
    learning_rate=0.05,
    reg_lambda=1.0,
    min_child_samples=10,
    random_state=42,
    n_jobs=2,
    verbosity=-1,
)


def _slsqp_convex(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    k = P.shape[1]
    w0 = np.full(k, 1.0 / k)

    def obj(w):
        r = P @ w - y
        return float(r @ r)

    def grad(w):
        r = P @ w - y
        return 2.0 * (P.T @ r)

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0),
             "jac": lambda w: np.ones_like(w)}]
    bnds = [(0.0, 1.0)] * k
    res = minimize(obj, w0, jac=grad, method="SLSQP", bounds=bnds,
                   constraints=cons,
                   options={"maxiter": 200, "ftol": 1e-9, "disp": False})
    w = np.clip(res.x, 0.0, 1.0)
    s = w.sum()
    if s > 0:
        w = w / s
    else:
        w = np.full(k, 1.0 / k)
    return w


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- XGBoost meta-learner stack on 4-anchor PRE+POST pool")
    print("=" * 78)

    # ---- Load ----
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values
                if "name" in te.columns else te["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test} n_unb={n_unb}")

    # ---- Anchors ----
    anchor_names = list(ANCHOR_OOF_PATHS.keys())
    K = len(anchor_names)
    P_unb = np.column_stack([
        np.load(ANCHOR_OOF_PATHS[k]).astype(np.float64) for k in anchor_names
    ])
    P_te = np.column_stack([
        np.load(ANCHOR_TE_PATHS[k]).astype(np.float64) for k in anchor_names
    ])
    rae_anchors = {k: float(rae(y_unb, P_unb[:, i]))
                   for i, k in enumerate(anchor_names)}
    print(f"[anchors] K={K}")
    for k in anchor_names:
        print(f"   {k:14s}  unb_RAE={rae_anchors[k]:.4f}  [{ANCHOR_PROVENANCE[k]}]")

    # ---- Scaffold folds ----
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    splits = scaffold_kfold_indices(unb_scaffolds, n_splits=N_FOLDS,
                                    shuffle=True, seed=KF_SEED)
    print(f"[scaffold] unique={n_unique_scaf}  n_folds={N_FOLDS}  kf_seed={KF_SEED}")

    # ---- XGBoost meta CV ----
    print("\n" + "-" * 78)
    print("XGBoost meta-stack CV")
    print("-" * 78)
    oof_xgb = np.full(n_unb, np.nan, dtype=np.float64)
    fold_rae_xgb = []
    for f_idx, (tr_loc, va_loc) in enumerate(splits):
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(P_unb[tr_loc], y_unb[tr_loc])
        pred = model.predict(P_unb[va_loc])
        oof_xgb[va_loc] = pred
        r = float(rae(y_unb[va_loc], pred))
        fold_rae_xgb.append(r)
        print(f"   fold {f_idx}: n_tr={len(tr_loc):3d} n_va={len(va_loc):3d}  RAE={r:.4f}")
    pooled_xgb = float(rae(y_unb, oof_xgb))
    mean_fold_xgb = float(np.mean(fold_rae_xgb))
    print(f"   pooled XGB RAE = {pooled_xgb:.4f}   mean-of-folds = {mean_fold_xgb:.4f}")

    # ---- LGBM meta comparator ----
    print("\n" + "-" * 78)
    print("LGBM meta-stack CV (paradigm-matched comparator)")
    print("-" * 78)
    oof_lgbm = np.full(n_unb, np.nan, dtype=np.float64)
    fold_rae_lgbm = []
    for f_idx, (tr_loc, va_loc) in enumerate(splits):
        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(P_unb[tr_loc], y_unb[tr_loc])
        pred = model.predict(P_unb[va_loc])
        oof_lgbm[va_loc] = pred
        r = float(rae(y_unb[va_loc], pred))
        fold_rae_lgbm.append(r)
        print(f"   fold {f_idx}: RAE={r:.4f}")
    pooled_lgbm = float(rae(y_unb, oof_lgbm))
    print(f"   pooled LGBM RAE = {pooled_lgbm:.4f}")

    # ---- SLSQP comparator ----
    print("\n" + "-" * 78)
    print("SLSQP convex comparator")
    print("-" * 78)
    oof_slsqp = np.full(n_unb, np.nan, dtype=np.float64)
    fold_rae_slsqp = []
    fold_weights_slsqp = []
    for f_idx, (tr_loc, va_loc) in enumerate(splits):
        w = _slsqp_convex(P_unb[tr_loc], y_unb[tr_loc])
        fold_weights_slsqp.append(w.tolist())
        pred = P_unb[va_loc] @ w
        oof_slsqp[va_loc] = pred
        r = float(rae(y_unb[va_loc], pred))
        fold_rae_slsqp.append(r)
        print(f"   fold {f_idx}: w={['%.3f'%x for x in w]}  RAE={r:.4f}")
    pooled_slsqp = float(rae(y_unb, oof_slsqp))
    print(f"   pooled SLSQP RAE = {pooled_slsqp:.4f}")

    # ---- Gate ----
    verdict = "PROMOTE" if pooled_xgb < GATE else "FAIL"
    print(f"\n[gate] XGB pooled {pooled_xgb:.4f}  <  {GATE}  ->  {verdict}")

    # ---- Deploy XGBoost on all 253 ----
    print("\n" + "-" * 78)
    print("DEPLOY: refit XGBoost on all 253, predict 513")
    print("-" * 78)
    model_deploy = xgb.XGBRegressor(**XGB_PARAMS)
    model_deploy.fit(P_unb, y_unb)
    te_pred = model_deploy.predict(P_te).astype(np.float32)
    te_pred = np.clip(te_pred, 3.0, 9.0)
    te_unb_in = float(rae(y_unb, te_pred[unb_idx]))
    print(f"   te mean={te_pred.mean():.3f} std={te_pred.std():.3f}")
    print(f"   te[unb_idx] in-sample RAE = {te_unb_in:.4f}  (expected << pooled)")

    feat_imp = dict(zip(anchor_names, [float(v) for v in model_deploy.feature_importances_]))
    print(f"   feature importances: {feat_imp}")

    # ---- Save ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_xgb.astype(np.float32))
    np.save(te_path, te_pred)
    print(f"[save] {oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_xgboost_meta_stack.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": te_pred,
    }).to_csv(sub_csv, index=False)
    print(f"[save] {sub_csv}")

    summary = {
        "tag": TAG,
        "method": "xgboost_meta_stack_4anchor",
        "anchor_pool": anchor_names,
        "anchor_provenance": ANCHOR_PROVENANCE,
        "anchor_excluded": {
            "nb1191": "only te_nb1191 (deploy-refit) exists; no honest OOF on 253"
        },
        "anchor_in_rae": rae_anchors,
        "xgb_params": XGB_PARAMS,
        "lgbm_params": LGBM_PARAMS,
        "n_folds": N_FOLDS,
        "kf_seed": KF_SEED,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "xgb_fold_rae": fold_rae_xgb,
        "xgb_pooled_rae": pooled_xgb,
        "xgb_mean_fold_rae": mean_fold_xgb,
        "lgbm_fold_rae": fold_rae_lgbm,
        "lgbm_pooled_rae": pooled_lgbm,
        "slsqp_fold_rae": fold_rae_slsqp,
        "slsqp_fold_weights": fold_weights_slsqp,
        "slsqp_pooled_rae": pooled_slsqp,
        "delta_xgb_vs_lgbm": pooled_xgb - pooled_lgbm,
        "delta_xgb_vs_slsqp": pooled_xgb - pooled_slsqp,
        "gate_threshold": GATE,
        "mean_rae": pooled_xgb,
        "verdict": verdict,
        "te_unb_in_sample_rae": te_unb_in,
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "feature_importances_deploy": feat_imp,
        "oof_npy_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   XGB pooled RAE   = {pooled_xgb:.4f}   (gate < {GATE}: {verdict})")
    print(f"   LGBM pooled RAE  = {pooled_lgbm:.4f}   (delta XGB-LGBM = {pooled_xgb-pooled_lgbm:+.4f})")
    print(f"   SLSQP pooled RAE = {pooled_slsqp:.4f}   (delta XGB-SLSQP = {pooled_xgb-pooled_slsqp:+.4f})")
    print(f"   wall             = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("xgb_pooled_rae", "lgbm_pooled_rae", "slsqp_pooled_rae",
              "delta_xgb_vs_lgbm", "delta_xgb_vs_slsqp",
              "verdict", "te_unb_in_sample_rae", "submission_csv"):
        print(f"  {k}: {res.get(k)}")
