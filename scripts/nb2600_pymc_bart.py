"""nb2600 -- PyMC-BART (modern Bayesian Additive Regression Trees) on K=20.

NEW PARADIGM:
    PyMC-BART = MODERN Bayesian Additive Regression Trees on the actively
    maintained PyMC backend (pytensor). Replaces the deprecated bartpy
    (which depends on EOL 'sklearn' PyPI package and modern-sklearn-API
    incompatible).
    - PyMC-BART places a regularization PRIOR over m small trees and runs
      a particle-Gibbs sampler over tree configurations within the PyMC
      MCMC. Posterior is a true distribution over tree-sum function space.
    - Different inductive bias from LGBM/CatBoost/XGB (point-MAP gradient
      boosters); different inductive bias from RF (deep bagged trees);
      different from BART-on-Stan (full HMC sampler).
    - On small n=253 with chemprop_aux residual the prior + MCMC
      averaging gives natural regularization. Two short chains x 200
      samples each keeps wall time tractable in CPU-only.

PROTOCOL:
    1. Install pymc-bart in .venv. On install failure, save INSTALL_FAILED.
    2. Load X_K20 = first 20 cols of X_117_unb / X_117_te.
    3. Anchor: chemprop_aux (PRE-unblind, verified-clean).
       Residual target = y_unb - anchor_unb.
    4. Model:
         with pm.Model():
             sigma = pm.HalfNormal("sigma", 1)
             mu = pmb.BART("mu", X, y, m=50)
             y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
             idata = pm.sample(200, tune=200, chains=2, cores=2,
                               random_seed=42)
       Test pred via pm.sample_posterior_predictive on X_te.
    5. 5-fold scaffold CV on 253, kf_seed=1001 (single seed - MCMC is slow).
    6. Deploy: refit on ALL 253 -> predict on 513.

GATE: mean_rae < 0.4570 -> "PROMOTE"
      mean_rae < 0.4601 -> "MARGINAL_BEAT"
      else            -> "FAIL"

Outputs:
    scripts/nb2600_pymc_bart.py
    data/processed/nb2600_summary.json
    data/processed/nb2600_pred_oof.npy   (253,) float32  (only if successful)
    data/processed/te_nb2600.npy         (513,) float32  (only if successful)
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

TAG = "nb2600"

# Gates
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4601

# CV protocol
N_FOLDS = 5
KF_SEED = 1001

# PyMC-BART hyperparameters (spec)
BART_M = 50           # number of trees
SAMPLE_DRAWS = 200    # posterior draws / chain
SAMPLE_TUNE = 200     # tuning iterations / chain
SAMPLE_CHAINS = 2
SAMPLE_CORES = 2
SAMPLE_SEED = 42

# Number of K=20 cols sliced from the 117-col block
K_SLICE = 20

# How many posterior draws to average for out-of-sample BART prediction.
# 2 chains x 200 draws = 400 trees available in `all_trees`; sample 200.
POSTERIOR_PRED_SIZE = 200


def _save_install_failed_summary(reason: str):
    """pymc-bart install failed -- write minimal summary and exit clean."""
    from pxr.paths import DATA_PROCESSED
    summary = {
        "tag": TAG,
        "method": "PyMC-BART on K=20 chemprop_aux residual",
        "paradigm": "modern_bayesian_tree_ensemble_mcmc_pytensor_backend",
        "verdict": "INSTALL_FAILED",
        "install_error": reason,
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "n_folds": N_FOLDS,
        "kf_seed": KF_SEED,
        "bart_m": BART_M,
        "sample_draws": SAMPLE_DRAWS,
        "sample_tune": SAMPLE_TUNE,
        "sample_chains": SAMPLE_CHAINS,
        "sample_cores": SAMPLE_CORES,
        "sample_seed": SAMPLE_SEED,
        "k_slice_first_n_of_117": K_SLICE,
        "anchor_name": "chemprop_aux (PRE-unblind, verified clean)",
        "mean_rae": None,
        "delta_vs_anchor": None,
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   verdict = INSTALL_FAILED")
    print(f"   reason  = {reason}")
    print("=" * 78)
    return summary


def main():
    # ---- Probe pymc / pymc-bart availability ----
    try:
        import pymc as pm
        import pymc_bart as pmb
        print(f"[probe] pymc={pm.__version__}  pymc_bart={pmb.__version__}")
        BART_OK = True
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"[probe] pymc-bart import failed: {err}")
        BART_OK = False

    if not BART_OK:
        return _save_install_failed_summary(
            "pymc-bart import failed in .venv"
        )

    # ---- Real fit path (only reached if pymc-bart importable) ----
    import pymc as pm
    import pymc_bart as pmb
    # Private utility used for out-of-sample BART prediction (BART trees are
    # baked at definition time; pm.set_data / posterior_predictive don't
    # re-route X. Standard idiom is _sample_posterior(all_trees, X_new, ...).)
    from pymc_bart.utils import _sample_posterior as bart_sample_posterior

    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    from pxr.chem import bemis_murcko
    from pxr.data import load_test
    from pxr.eval import rae, scaffold_kfold_indices
    from pxr.paths import DATA_PROCESSED

    X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
    X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"
    TE_CHEM_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- PyMC-BART on K=20 chemprop_aux residual "
          f"(modern Bayesian tree ensemble, pytensor MCMC)")
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
    X_unb = X_unb_117[:, :K_SLICE].astype(np.float64)
    X_te = X_te_117[:, :K_SLICE].astype(np.float64)
    print(f"[feat] X_unb_K20={X_unb.shape}  X_te_K20={X_te.shape}  "
          f"slice=first-{K_SLICE}-cols")

    # ---- Anchor ----
    if not TE_CHEM_PATH.exists():
        raise FileNotFoundError(f"missing test anchor: {TE_CHEM_PATH}")
    te_chem = np.load(TE_CHEM_PATH).astype(np.float64)
    assert te_chem.shape == (n_test,), f"te_chem shape {te_chem.shape}"
    anchor_unb = te_chem[unb_idx]
    anchor_te = te_chem.copy()
    rae_anchor_unb = float(rae(y_unb, anchor_unb))
    print(f"[anchor] chemprop_aux te[unb_idx] RAE = {rae_anchor_unb:.4f}")

    # ---- Residual target ----
    resid_unb = y_unb - anchor_unb
    print(f"[resid] mean={resid_unb.mean():+.3f}  std={resid_unb.std():.3f}  "
          f"min={resid_unb.min():+.2f}  max={resid_unb.max():+.2f}")

    # ---- Scaffold 5-fold CV, single seed (MCMC slow) ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV  kf_seed={KF_SEED}  (single seed -- MCMC slow)\n"
          f"PyMC-BART: m={BART_M}  draws={SAMPLE_DRAWS}  tune={SAMPLE_TUNE}  "
          f"chains={SAMPLE_CHAINS}  cores={SAMPLE_CORES}  seed={SAMPLE_SEED}")
    print("-" * 78)

    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=KF_SEED,
    )
    oof_resid = np.full(n_unb, np.nan, dtype=np.float64)
    fold_info = []
    fold_failures = []
    for fi, (tr_loc, va_loc) in enumerate(splits):
        ts_f = time.time()
        try:
            X_tr = X_unb[tr_loc]
            y_tr = resid_unb[tr_loc].astype(np.float64)
            X_va = X_unb[va_loc]
            with pm.Model() as fold_model:
                sigma = pm.HalfNormal("sigma", 1.0)
                mu = pmb.BART("mu", X_tr, y_tr, m=BART_M)
                y_obs = pm.Normal(
                    "y_obs", mu=mu, sigma=sigma, observed=y_tr,
                )
                idata = pm.sample(
                    draws=SAMPLE_DRAWS,
                    tune=SAMPLE_TUNE,
                    chains=SAMPLE_CHAINS,
                    cores=SAMPLE_CORES,
                    random_seed=SAMPLE_SEED,
                    progressbar=False,
                    compute_convergence_checks=False,
                )
            # Out-of-sample prediction on validation X.
            # `all_trees` holds the posterior tree ensembles (one per draw).
            # _sample_posterior(size=N) draws N posterior samples, returning
            # array of shape (N, n_X, shape). We average across the N draws.
            all_trees = mu.owner.op.all_trees
            rng = np.random.default_rng(SAMPLE_SEED + fi)
            mu_draws_va = bart_sample_posterior(
                all_trees, X=X_va, rng=rng,
                size=POSTERIOR_PRED_SIZE, shape=1,
            )
            # Expected shape (POSTERIOR_PRED_SIZE, n_va, 1). Average over
            # draws then squeeze the trailing shape=1 axis.
            mu_draws_va = np.asarray(mu_draws_va)
            fold_pred = mu_draws_va.mean(axis=0).reshape(-1).astype(np.float64)
            assert fold_pred.shape == (len(va_loc),), (
                f"fold {fi} pred shape {fold_pred.shape} vs n_va {len(va_loc)}"
            )
            oof_resid[va_loc] = fold_pred
            fold_info.append({
                "fold": fi,
                "n_tr": int(len(tr_loc)),
                "n_va": int(len(va_loc)),
                "resid_pred_mean": float(oof_resid[va_loc].mean()),
                "resid_pred_std": float(oof_resid[va_loc].std()),
                "wall_sec": round(time.time() - ts_f, 2),
            })
            print(f"   fold={fi}  n_tr={len(tr_loc)}  n_va={len(va_loc)}  "
                  f"wall={time.time()-ts_f:.1f}s")
        except Exception as fold_err:
            err_txt = f"{type(fold_err).__name__}: {fold_err}"
            print(f"   fold={fi}  FAILED: {err_txt}")
            fold_failures.append({"fold": fi, "error": err_txt})
            # Mark this fold's val rows so we can detect cover holes later.
            oof_resid[va_loc] = np.nan

    if np.isnan(oof_resid).any() or fold_failures:
        n_nan = int(np.isnan(oof_resid).sum())
        reason = (
            f"oof_resid has {n_nan} NaN across {len(fold_failures)} failed "
            f"folds: {fold_failures}"
        )
        print(f"\n[abort] {reason}")
        return _save_install_failed_summary(reason)

    oof_pred = anchor_unb + oof_resid
    oof_pred = np.clip(oof_pred, 3.0, 8.0)
    pooled_rae = float(rae(y_unb, oof_pred))
    print(f"\n[cv] pooled RAE (single seed kf={KF_SEED}) = {pooled_rae:.4f}")
    print(f"[cv] delta vs anchor (chemprop_aux)       = {pooled_rae - rae_anchor_unb:+.4f}")

    # ---- Deploy: refit on all 253 unblind -> predict 513 ----
    print("\n" + "-" * 78)
    print("DEPLOY: refit on all 253 unblind -> apply to 513")
    print("-" * 78)
    try:
        with pm.Model() as deploy_model:
            sigma = pm.HalfNormal("sigma", 1.0)
            mu_deploy = pmb.BART("mu", X_unb, resid_unb, m=BART_M)
            y_obs = pm.Normal(
                "y_obs", mu=mu_deploy, sigma=sigma, observed=resid_unb,
            )
            idata_deploy = pm.sample(
                draws=SAMPLE_DRAWS,
                tune=SAMPLE_TUNE,
                chains=SAMPLE_CHAINS,
                cores=SAMPLE_CORES,
                random_seed=SAMPLE_SEED,
                progressbar=False,
                compute_convergence_checks=False,
            )
        all_trees_dep = mu_deploy.owner.op.all_trees
        rng_dep = np.random.default_rng(SAMPLE_SEED)
        mu_draws_te = bart_sample_posterior(
            all_trees_dep, X=X_te, rng=rng_dep,
            size=POSTERIOR_PRED_SIZE, shape=1,
        )
        mu_draws_te = np.asarray(mu_draws_te)
        deploy_resid_te = (
            mu_draws_te.mean(axis=0).reshape(-1).astype(np.float64)
        )
        assert deploy_resid_te.shape == (n_test,), (
            f"deploy_resid_te shape {deploy_resid_te.shape}"
        )
    except Exception as dep_err:
        err_txt = f"{type(dep_err).__name__}: {dep_err}"
        print(f"[deploy] FAILED: {err_txt}")
        return _save_install_failed_summary(f"deploy refit failed: {err_txt}")

    deploy_te = anchor_te + deploy_resid_te.astype(np.float64)
    deploy_te = np.clip(deploy_te, 3.0, 8.0).astype(np.float32)
    te_unb_in_rae = float(rae(y_unb, deploy_te[unb_idx].astype(np.float64)))
    print(f"[deploy] te(513) mean={deploy_te.mean():.3f}  std={deploy_te.std():.3f}")
    print(f"[deploy] te[unb_idx] in_RAE={te_unb_in_rae:.4f}")

    # ---- Gate ----
    if pooled_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif pooled_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "-" * 78)
    print("GATE EVALUATION")
    print("-" * 78)
    print(f"   mean_rae       = {pooled_rae:.4f}")
    print(f"   gate PROMOTE   = < {GATE_PROMOTE}")
    print(f"   gate MARGINAL  = < {GATE_MARGINAL}")
    print(f"   verdict        = {verdict}")

    # ---- Save artefacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_pred.astype(np.float32))
    np.save(te_path, deploy_te)
    print(f"\n[save] {oof_path}")
    print(f"[save] {te_path}")

    summary = {
        "tag": TAG,
        "method": (
            "PyMC-BART on K=20 first-cols of X_117 fit to chemprop_aux "
            "residual; modern MCMC posterior over tree ensembles via "
            "pytensor backend"
        ),
        "paradigm": "modern_bayesian_tree_ensemble_mcmc_pytensor_backend",
        "anchor_name": "chemprop_aux (PRE-unblind, verified clean)",
        "rae_anchor_unb": rae_anchor_unb,
        "bart_m": BART_M,
        "sample_draws": SAMPLE_DRAWS,
        "sample_tune": SAMPLE_TUNE,
        "sample_chains": SAMPLE_CHAINS,
        "sample_cores": SAMPLE_CORES,
        "sample_seed": SAMPLE_SEED,
        "posterior_pred_size": POSTERIOR_PRED_SIZE,
        "n_folds": N_FOLDS,
        "kf_seed": KF_SEED,
        "n_unb": n_unb,
        "n_te": n_test,
        "n_unique_scaffolds": n_uniq_scaf,
        "feat_dim": int(X_unb.shape[1]),
        "k_slice_first_n_of_117": K_SLICE,
        "folds": fold_info,
        "fold_failures": fold_failures,
        "mean_rae": pooled_rae,
        "delta_vs_anchor": pooled_rae - rae_anchor_unb,
        "te_unb_rae_in_sample": te_unb_in_rae,
        "te_deploy_mean": float(deploy_te.mean()),
        "te_deploy_std": float(deploy_te.std()),
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "verdict": verdict,
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
    print(f"   mean_rae (kf={KF_SEED})       = {pooled_rae:.4f}")
    print(f"   delta vs anchor (chemprop_aux)= {pooled_rae - rae_anchor_unb:+.4f}")
    print(f"   verdict                       = {verdict}")
    print(f"   wall                          = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae",
        "rae_anchor_unb",
        "delta_vs_anchor",
        "verdict",
        "te_unb_rae_in_sample",
    ):
        print(f"  {k}: {res.get(k)}")
