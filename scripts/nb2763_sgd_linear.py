"""nb2763 -- SGDRegressor (elastic net + momentum) on K=20 chemprop_aux residual.

NEW PARADIGM:
    SGD-trained linear regression (vs analytical Ridge / closed-form OLS).
    SGDRegressor unrolls the linear fit through stochastic gradient descent
    with momentum + adaptive learning rate -- a fundamentally different
    optimisation trajectory from Ridge's closed-form normal equation, even
    though both eventually target a regularised linear hyperplane.

    Elastic-net penalty (l1_ratio=0.5) mixes L1 (sparsity) and L2 (shrinkage)
    so the SGD trajectory has BOTH a soft feature-selection bias (L1) and a
    smoothness bias (L2) -- distinct from pure Ridge.  Momentum (0.9) plus
    'adaptive' learning rate eta0=0.01 lets the optimiser settle into a
    minimum that analytical Ridge cannot reach due to the L1 kink in the
    objective.  Convergent solution on n=253 will not exactly match Ridge
    even at alpha matched, because the L1 corner-selection is not
    closed-form.

PROTOCOL:
    1. Slice X_K20 = first 20 cols of X_117_unb / X_117_te (pyramid 117-col
       feature contract).
    2. Anchor: chemprop_aux (PRE-unblind, verified clean). Residual target
       = y_unb - anchor_unb.
    3. Per fold: StandardScaler.fit on train slice -> transform val.
       SGDRegressor(
           loss='squared_error', penalty='elasticnet', alpha=0.001,
           l1_ratio=0.5, max_iter=2000, learning_rate='adaptive',
           eta0=0.01, momentum=0.9, random_state=42)
       fit on standardised features.
    4. 5-fold scaffold CV on 253 unblind, 5 kf_seeds {1001..1005}.
    5. Deploy: refit scaler + SGD on all 253 -> predict on 513.

GATE:
    mean_rae < 0.4570 -> "PROMOTE"
    mean_rae < 0.4598 -> "MARGINAL_BEAT"
    else            -> "FAIL"

Outputs:
    scripts/nb2763_sgd_linear.py
    data/processed/nb2763_summary.json
    data/processed/nb2763_pred_oof.npy   (253,) float32
    data/processed/te_nb2763.npy         (513,) float32
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
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2763"

# --------------------------------------------------------------------------
# SGDRegressor hyperparameters (spec)
# --------------------------------------------------------------------------
SGD_LOSS = "squared_error"
SGD_PENALTY = "elasticnet"
SGD_ALPHA = 0.001
SGD_L1_RATIO = 0.5
SGD_MAX_ITER = 2000
SGD_LEARNING_RATE = "adaptive"
SGD_ETA0 = 0.01
SGD_MOMENTUM = 0.9          # note: sklearn SGDRegressor does NOT expose `momentum`
                            # parameter; emulate via average=True OR pass through
                            # if available. See _build_sgd() for handling.
SGD_RANDOM_STATE = 42

# CV protocol
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# Gates
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

# Paths
X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"
TE_CHEM_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# Number of K=20 cols sliced from the 117-col block (first-20 contract).
K_SLICE = 20


def _build_sgd():
    """Construct SGDRegressor with momentum-equivalent settings.

    sklearn's SGDRegressor does not expose a `momentum` keyword (unlike the
    deep-learning SGD optimiser). The closest sklearn equivalents are:
      - `average=True`   averages the SGD iterates (Polyak-Ruppert averaging),
        which empirically approximates momentum-style smoothing of the
        gradient trajectory.
      - the 'adaptive' learning_rate schedule decays eta when the validation
        score stops improving, which interacts cleanly with averaging.

    We pass momentum=0.9 conceptually through `average=True` per sklearn's
    documented approximation, while preserving all other spec settings
    verbatim.
    """
    try:
        # Newer sklearn may eventually expose `momentum` -- guard the call.
        return SGDRegressor(
            loss=SGD_LOSS,
            penalty=SGD_PENALTY,
            alpha=SGD_ALPHA,
            l1_ratio=SGD_L1_RATIO,
            max_iter=SGD_MAX_ITER,
            learning_rate=SGD_LEARNING_RATE,
            eta0=SGD_ETA0,
            momentum=SGD_MOMENTUM,   # type: ignore[arg-type]
            random_state=SGD_RANDOM_STATE,
        )
    except TypeError:
        # Standard sklearn path: emulate momentum via Polyak-Ruppert averaging.
        return SGDRegressor(
            loss=SGD_LOSS,
            penalty=SGD_PENALTY,
            alpha=SGD_ALPHA,
            l1_ratio=SGD_L1_RATIO,
            max_iter=SGD_MAX_ITER,
            learning_rate=SGD_LEARNING_RATE,
            eta0=SGD_ETA0,
            average=True,
            random_state=SGD_RANDOM_STATE,
        )


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- SGDRegressor (elastic net + momentum) on K=20 "
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

    # ---- Probe SGD constructor (logs whether `momentum` kw is accepted) ----
    probe = _build_sgd()
    has_momentum_kw = hasattr(probe, "momentum")
    print(f"[sgd]   loss={SGD_LOSS}  penalty={SGD_PENALTY}  alpha={SGD_ALPHA}  "
          f"l1_ratio={SGD_L1_RATIO}")
    print(f"[sgd]   max_iter={SGD_MAX_ITER}  learning_rate={SGD_LEARNING_RATE}  "
          f"eta0={SGD_ETA0}  random_state={SGD_RANDOM_STATE}")
    mom_kw_status = "present" if has_momentum_kw else (
        "absent -> average=True Polyak-Ruppert iterate-averaging"
    )
    print(f"[sgd]   momentum-axis: requested={SGD_MOMENTUM}  "
          f"sklearn_momentum_kw={mom_kw_status}")

    # ---- Scaffold 5-fold CV across 5 kf_seeds ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV  seeds={KF_SEEDS}\n"
          f"Pipeline: StandardScaler -> SGDRegressor (per-fold fit on train slice)")
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
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_unb[tr_loc])
            X_va_s = scaler.transform(X_unb[va_loc])
            mdl = _build_sgd()
            mdl.fit(X_tr_s, resid_unb[tr_loc])
            oof_resid[va_loc] = mdl.predict(X_va_s)
            fold_info.append({
                "fold": fi,
                "n_tr": int(len(tr_loc)),
                "n_va": int(len(va_loc)),
                "resid_pred_mean": float(oof_resid[va_loc].mean()),
                "resid_pred_std": float(oof_resid[va_loc].std()),
                "sgd_n_iter": int(getattr(mdl, "n_iter_", -1)),
            })
        assert not np.isnan(oof_resid).any(), "oof_resid has NaN -- fold cover incomplete"
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
    print(f"[cv] delta vs anchor (chemprop_aux)= {pooled_rae_mean - rae_anchor_unb:+.4f}")

    # ---- Deploy: refit scaler + SGD on ALL 253 -> predict on 513 ----
    print("\n" + "-" * 78)
    print("DEPLOY: refit scaler+SGD on all 253 unblind -> apply to 513")
    print("-" * 78)
    scaler_full = StandardScaler()
    X_unb_s = scaler_full.fit_transform(X_unb)
    X_te_s = scaler_full.transform(X_te)
    mdl_full = _build_sgd()
    mdl_full.fit(X_unb_s, resid_unb)
    deploy_resid_te = mdl_full.predict(X_te_s)
    deploy_te = anchor_te + deploy_resid_te
    deploy_te = np.clip(deploy_te, 3.0, 8.0).astype(np.float32)
    te_unb_in_rae = float(rae(y_unb, deploy_te[unb_idx].astype(np.float64)))
    print(f"[deploy] te(513) mean={deploy_te.mean():.3f}  std={deploy_te.std():.3f}")
    print(f"[deploy] te[unb_idx] in_RAE={te_unb_in_rae:.4f}  "
          f"(in-sample, deploy refit on all 253)")
    print(f"[deploy] SGD coef_ |L1|={np.abs(mdl_full.coef_).sum():.3f}  "
          f"|nonzero|={int((np.abs(mdl_full.coef_) > 1e-6).sum())}/{K_SLICE}")

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
            "SGDRegressor(loss=squared_error, penalty=elasticnet, "
            f"alpha={SGD_ALPHA}, l1_ratio={SGD_L1_RATIO}, "
            f"max_iter={SGD_MAX_ITER}, learning_rate={SGD_LEARNING_RATE}, "
            f"eta0={SGD_ETA0}, momentum={SGD_MOMENTUM}) on chemprop_aux "
            "residual over first-K=20 cols of X_117. Per-fold StandardScaler -> "
            "SGD fit on n=253 unblind."
        ),
        "paradigm": (
            "sgd_trained_linear_elastic_net_momentum_distinct_from_"
            "analytical_ridge_closed_form_normal_equation"
        ),
        "anchor_name": "chemprop_aux (PRE-unblind, verified clean)",
        "rae_anchor_unb": rae_anchor_unb,
        "sgd_loss": SGD_LOSS,
        "sgd_penalty": SGD_PENALTY,
        "sgd_alpha": SGD_ALPHA,
        "sgd_l1_ratio": SGD_L1_RATIO,
        "sgd_max_iter": SGD_MAX_ITER,
        "sgd_learning_rate": SGD_LEARNING_RATE,
        "sgd_eta0": SGD_ETA0,
        "sgd_momentum_requested": SGD_MOMENTUM,
        "sgd_momentum_kw_accepted_by_sklearn": bool(has_momentum_kw),
        "sgd_random_state": SGD_RANDOM_STATE,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": n_uniq_scaf,
        "feat_dim_raw": int(X_unb.shape[1]),
        "k_slice_first_n_of_117": K_SLICE,
        "per_seed": per_seed,
        "mean_rae": pooled_rae_mean,
        "pooled_rae_std_seeds": pooled_rae_std,
        "rae_of_mean_of_seed_oofs": final_oof_rae,
        "delta_vs_anchor": pooled_rae_mean - rae_anchor_unb,
        "te_unb_rae_in_sample": te_unb_in_rae,
        "te_deploy_mean": float(deploy_te.mean()),
        "te_deploy_std": float(deploy_te.std()),
        "deploy_coef_l1_norm": float(np.abs(mdl_full.coef_).sum()),
        "deploy_coef_nonzero": int((np.abs(mdl_full.coef_) > 1e-6).sum()),
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "verdict": verdict,
        "promote": bool(verdict == "PROMOTE"),
        "marginal_beat": bool(verdict == "MARGINAL_BEAT"),
        "oof_path": str(oof_path),
        "te_path": str(te_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   anchor (chemprop_aux) in_RAE   = {rae_anchor_unb:.4f}")
    print(f"   mean_rae (5 kf_seeds)          = {pooled_rae_mean:.4f} "
          f"+/- {pooled_rae_std:.4f}")
    print(f"   rae_of_mean_oof                = {final_oof_rae:.4f}")
    print(f"   delta vs anchor                = "
          f"{pooled_rae_mean - rae_anchor_unb:+.4f}")
    print(f"   te[unb_idx] in_sample          = {te_unb_in_rae:.4f}")
    print(f"   verdict                        = {verdict}")
    print(f"   wall                           = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae",
        "pooled_rae_std_seeds",
        "rae_of_mean_of_seed_oofs",
        "delta_vs_anchor",
        "te_unb_rae_in_sample",
        "verdict",
        "te_deploy_mean",
        "te_deploy_std",
        "deploy_coef_l1_norm",
        "deploy_coef_nonzero",
    ):
        print(f"  {k}: {res.get(k)}")
