"""nb2591 -- Direct RAE-loss LightGBM via custom gradient.

NEW PARADIGM:
    Every prior LGBM in the chemprop_aux residual ladder optimizes either
    L2 (MSE) or Huber, then we evaluate under RAE.  The objective-metric
    mismatch is a known source of variance compression: a model that
    minimizes Pearson-2nd-moment risk is not the same as one that minimizes
    L1-relative-to-target-MAD.  We close that loop by optimizing RAE
    DIRECTLY via LightGBM's custom objective API.

    RAE definition (lower = better):
        RAE(y, y_hat) = mean|y_hat - y| / mean|y - mean(y)|

    The denominator depends ONLY on the labels (it's the truth MAD), so it
    is a constant per fold-train.  The gradient wrt y_hat is therefore:

        dL/dy_hat = sign(y_hat - y) / (n * MAD_y)

    The 2nd derivative of |y_hat - y| is zero (cusp at y_hat = y), so we
    cannot use the true Hessian.  LightGBM requires a strictly positive
    Hessian for stable leaf-value computation; we provide a small constant
    (HESS_CONST = 0.1).  This is the standard trick for L1-style objectives
    (cf. LightGBM `regression_l1` internally uses constant Hessian).

PROTOCOL:
    1. Load X_K20 = first 20 cols of X_117_unb / X_117_te.
    2. Anchor: chemprop_aux (PRE-unblind, verified-clean).
       Residual target = y_unb - anchor_unb.
    3. LightGBM(custom_obj=rae_objective, max_depth=4, num_leaves=15,
       n_est=300, lr=0.05) fit on residual; final pred = anchor + lgb_resid.
    4. 5-fold scaffold CV on 253 unblind, kf_seeds {1001..1005}.
    5. Deploy: refit on ALL 253 -> predict on 513.

GATE: mean_rae < 0.4570 -> "PROMOTE"
      mean_rae < 0.4601 -> "MARGINAL_BEAT"
      else            -> "FAIL"

Outputs:
    scripts/nb2591_direct_rae_loss.py
    data/processed/nb2591_summary.json
    data/processed/nb2591_pred_oof.npy   (253,) float32
    data/processed/te_nb2591.npy         (513,) float32
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

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2591"

# -----------------------------
# LGBM hyperparameters (spec)
# -----------------------------
LGBM_MAX_DEPTH = 4
LGBM_NUM_LEAVES = 15
LGBM_N_EST = 300
LGBM_LR = 0.05
HESS_CONST = 0.1   # constant Hessian for L1-style objectives

# CV protocol
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# Gates
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4601

# Paths
X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"
TE_CHEM_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"     # (513,)

# Number of K=20 cols sliced from the 117-col block (first-20 contract).
K_SLICE = 20


def _make_rae_objective(mad_y: float, n: int):
    """Return an LGBM custom objective callable closing over MAD_y, n.

    LightGBM custom-objective signature:
        fn(y_pred, train_data) -> (grad, hess)
    where y_pred is the raw model output (here equal to the residual
    prediction, since we have no link / inverse-link).

    Gradient of RAE = mean|y_pred - y| / mad_y wrt y_pred (per-row):
        grad_i = sign(y_pred_i - y_i) / (n * mad_y)

    Hessian: |.| has zero 2nd derivative away from the cusp; we use a
    small positive constant for numerical stability of the LGBM
    leaf-value step.
    """
    denom = float(max(n * mad_y, 1e-8))

    def _obj(y_pred: np.ndarray, train_data) -> tuple[np.ndarray, np.ndarray]:
        y_true = train_data.get_label()
        diff = y_pred - y_true
        grad = np.sign(diff).astype(np.float64) / denom
        hess = np.full_like(grad, HESS_CONST, dtype=np.float64)
        return grad, hess

    return _obj


def _make_rae_metric(mad_y: float):
    """Return an LGBM custom feval reporting RAE during boosting.

    Signature: fn(y_pred, train_data) -> (eval_name, eval_result, is_higher_better)
    """
    denom = float(max(mad_y, 1e-8))

    def _feval(y_pred: np.ndarray, train_data) -> tuple[str, float, bool]:
        y_true = train_data.get_label()
        rae_val = float(np.mean(np.abs(y_pred - y_true)) / denom)
        return ("rae_residual", rae_val, False)  # lower-is-better

    return _feval


def _lgbm_params(seed: int, fobj) -> dict:
    """Booster params for lgb.train() with custom objective.

    In LightGBM >=4.0 the custom objective is passed via the `objective`
    key of params (a callable), not as a separate `fobj` kwarg.
    """
    return dict(
        objective=fobj,              # custom objective callable
        metric="None",               # disable default metric -- we use feval
        max_depth=LGBM_MAX_DEPTH,
        num_leaves=LGBM_NUM_LEAVES,
        learning_rate=LGBM_LR,
        seed=int(seed),
        bagging_seed=int(seed),
        feature_fraction_seed=int(seed),
        n_jobs=2,
        verbosity=-1,
        min_data_in_leaf=5,
    )


def _fit_predict(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Train a custom-RAE booster on (X_tr, y_tr) and predict X_va."""
    mad_y = float(np.mean(np.abs(y_tr - np.mean(y_tr))))
    fobj = _make_rae_objective(mad_y=mad_y, n=len(y_tr))
    feval = _make_rae_metric(mad_y=mad_y)
    dtrain = lgb.Dataset(X_tr, label=y_tr.astype(np.float64), free_raw_data=False)
    booster = lgb.train(
        params=_lgbm_params(seed, fobj),
        train_set=dtrain,
        num_boost_round=LGBM_N_EST,
        feval=feval,
    )
    return booster.predict(X_va).astype(np.float64)


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- LightGBM with CUSTOM RAE OBJECTIVE on K=20 "
          f"chemprop_aux residual")
    print("=" * 78)

    # ---- Load test set + scaffolds + truth ----
    te = load_test()
    n_test = len(te)
    smi_col = "smiles" if "smiles" in te.columns else "SMILES"
    te_smiles = te[smi_col].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_uniq_scaf = len({s for s in unb_scaffolds if s})
    print(f"[load] n_test={n_test}  n_unb={n_unb}  unique_scaf={n_uniq_scaf}")

    # ---- Load X_117 then slice to first K=20 cols ----
    X_unb_117 = np.load(X117_UNB_PATH).astype(np.float32)
    X_te_117 = np.load(X117_TE_PATH).astype(np.float32)
    assert X_unb_117.shape == (n_unb, 117), f"X_unb shape {X_unb_117.shape}"
    assert X_te_117.shape == (n_test, 117), f"X_te shape {X_te_117.shape}"
    X_unb = X_unb_117[:, :K_SLICE].astype(np.float32)
    X_te = X_te_117[:, :K_SLICE].astype(np.float32)
    # Sanitize NaN/Inf carried in cache
    X_unb = np.where(np.isfinite(X_unb), X_unb, 0.0).astype(np.float32)
    X_te = np.where(np.isfinite(X_te), X_te, 0.0).astype(np.float32)
    print(f"[feat] X_unb_K20={X_unb.shape}  X_te_K20={X_te.shape}  "
          f"slice=first-{K_SLICE}-cols")

    # ---- Anchor (chemprop_aux, PRE-unblind verified-clean) ----
    if not TE_CHEM_PATH.exists():
        raise FileNotFoundError(f"missing test anchor: {TE_CHEM_PATH}")
    te_chem = np.load(TE_CHEM_PATH).astype(np.float64)
    assert te_chem.shape == (n_test,), f"te_chem shape {te_chem.shape}"
    anchor_unb = te_chem[unb_idx]
    anchor_te = te_chem.copy()
    rae_anchor_unb = float(rae(y_unb, anchor_unb))
    print(f"[anchor] chemprop_aux te[unb_idx] RAE = {rae_anchor_unb:.4f} "
          f"(PRE-clean PRIMARY-1 baseline)")

    # ---- Residual target ----
    resid_unb = y_unb - anchor_unb
    print(f"[resid] mean={resid_unb.mean():+.3f}  std={resid_unb.std():.3f}  "
          f"min={resid_unb.min():+.2f}  max={resid_unb.max():+.2f}")
    print(f"[rae-obj] HESS_CONST={HESS_CONST}  (custom L1-style hessian)")

    # ---- Scaffold 5-fold CV across 5 kf_seeds ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV  seeds={KF_SEEDS}\n"
          f"LGBM(custom_rae): max_depth={LGBM_MAX_DEPTH}  "
          f"num_leaves={LGBM_NUM_LEAVES}  n_est={LGBM_N_EST}  lr={LGBM_LR}")
    print("-" * 78)

    per_seed = []
    all_oofs = []
    for kf_seed in KF_SEEDS:
        ts = time.time()
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        oof_resid = np.full(n_unb, np.nan, dtype=np.float64)
        fold_info = []
        for fi, (tr_loc, va_loc) in enumerate(splits):
            pred_resid_va = _fit_predict(
                X_unb[tr_loc], resid_unb[tr_loc],
                X_unb[va_loc],
                seed=kf_seed + fi,
            )
            oof_resid[va_loc] = pred_resid_va
            fold_info.append({
                "fold": fi,
                "n_tr": int(len(tr_loc)),
                "n_va": int(len(va_loc)),
                "resid_pred_mean": float(pred_resid_va.mean()),
                "resid_pred_std": float(pred_resid_va.std()),
            })
        assert not np.isnan(oof_resid).any(), "oof_resid NaN -- fold cover incomplete"
        oof_pred = anchor_unb + oof_resid
        # gentle clip to a sane pEC50 range
        oof_pred = np.clip(oof_pred, 3.0, 8.0)
        pooled = float(rae(y_unb, oof_pred))
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": pooled,
            "folds": fold_info,
            "wall_sec": round(time.time() - ts, 2),
        })
        all_oofs.append(oof_pred)
        print(f"   seed={kf_seed}  pooled_RAE={pooled:.4f}  wall={time.time()-ts:.1f}s")

    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    final_oof_rae = float(rae(y_unb, mean_oof))
    pooled_rae_mean = float(np.mean([r["pooled_rae"] for r in per_seed]))
    pooled_rae_std = float(np.std([r["pooled_rae"] for r in per_seed]))
    print(f"\n[cv] pooled RAE mean across seeds = {pooled_rae_mean:.4f} "
          f"(+/- {pooled_rae_std:.4f})")
    print(f"[cv] RAE of mean-of-seed OOFs      = {final_oof_rae:.4f}")
    print(f"[cv] delta vs anchor (chemprop_aux)= "
          f"{pooled_rae_mean - rae_anchor_unb:+.4f}")

    # ---- Deploy: refit on ALL 253 -> predict on 513 ----
    print("\n" + "-" * 78)
    print("DEPLOY: refit on all 253 unblind -> apply to 513")
    print("-" * 78)
    deploy_resid_te = _fit_predict(
        X_unb, resid_unb, X_te, seed=KF_SEEDS[0],
    )
    deploy_te = anchor_te + deploy_resid_te
    deploy_te = np.clip(deploy_te, 3.0, 8.0).astype(np.float32)
    te_unb_in_rae = float(rae(y_unb, deploy_te[unb_idx].astype(np.float64)))
    print(f"[deploy] te(513) mean={deploy_te.mean():.3f}  std={deploy_te.std():.3f}")
    print(f"[deploy] te[unb_idx] in_RAE={te_unb_in_rae:.4f}  "
          f"(in-sample, deploy refit on all 253)")

    # ---- Gate ----
    if pooled_rae_mean < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif pooled_rae_mean < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "-" * 78)
    print("GATE EVALUATION")
    print("-" * 78)
    print(f"   mean_rae       = {pooled_rae_mean:.4f}")
    print(f"   gate PROMOTE   = < {GATE_PROMOTE}")
    print(f"   gate MARGINAL  = < {GATE_MARGINAL}")
    print(f"   verdict        = {verdict}")

    # ---- Save artefacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, mean_oof.astype(np.float32))
    np.save(te_path, deploy_te)
    print(f"\n[save] {oof_path}")
    print(f"[save] {te_path}")

    summary = {
        "tag": TAG,
        "method": (
            "LightGBM with CUSTOM RAE OBJECTIVE (grad=sign(y_pred-y)/(n*MAD_y), "
            "hess=0.1 const) on K=20 first-cols of X_117 fit to chemprop_aux residual"
        ),
        "paradigm": "direct_rae_loss_via_custom_lgb_objective",
        "anchor_name": "chemprop_aux (PRE-unblind, verified clean)",
        "rae_anchor_unb": rae_anchor_unb,
        "lgbm_max_depth": LGBM_MAX_DEPTH,
        "lgbm_num_leaves": LGBM_NUM_LEAVES,
        "lgbm_n_estimators": LGBM_N_EST,
        "lgbm_learning_rate": LGBM_LR,
        "hess_const": HESS_CONST,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_unb": n_unb,
        "n_te": n_test,
        "n_unique_scaffolds": n_uniq_scaf,
        "feat_dim": int(X_unb.shape[1]),
        "k_slice_first_n_of_117": K_SLICE,
        "per_seed": per_seed,
        "mean_rae": pooled_rae_mean,
        "pooled_rae_std_seeds": pooled_rae_std,
        "rae_of_mean_of_seed_oofs": final_oof_rae,
        "delta_vs_anchor": pooled_rae_mean - rae_anchor_unb,
        "te_unb_rae_in_sample": te_unb_in_rae,
        "te_deploy_mean": float(deploy_te.mean()),
        "te_deploy_std": float(deploy_te.std()),
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "verdict": verdict,
        "oof_path": str(oof_path),
        "te_path": str(te_path),
        "pre_unblind_clean_anchor": True,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   mean_rae (5 seeds)            = {pooled_rae_mean:.4f} "
          f"(+/- {pooled_rae_std:.4f})")
    print(f"   delta vs anchor (chemprop_aux)= "
          f"{pooled_rae_mean - rae_anchor_unb:+.4f}")
    print(f"   verdict                       = {verdict}")
    print(f"   wall                          = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae",
        "pooled_rae_std_seeds",
        "rae_of_mean_of_seed_oofs",
        "rae_anchor_unb",
        "delta_vs_anchor",
        "verdict",
        "te_unb_rae_in_sample",
    ):
        print(f"  {k}: {res.get(k)}")
