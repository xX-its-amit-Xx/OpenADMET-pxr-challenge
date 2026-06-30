"""nb2772 -- Hierarchical Bayesian linear with horseshoe prior on K=20 residual.

NEW PARADIGM:
    Horseshoe shrinkage prior selects sparse important features automatically
    via a half-Cauchy global scale (tau) x half-Cauchy local scales (lam_j).
    Heavy-tailed local scales let truly important features escape shrinkage
    while pushing noise features hard to zero -- a Bayesian sparse-selection
    alternative to L1/ElasticNet that doesn't require a discrete L0 search.

    Cycle-160+ post-hoc-blend ceiling at 0.4718-0.4720 sits on chemprop_aux
    anchor with K=28 LGBM residual.  This script asks: does a different
    inductive prior (continuous shrinkage at the coefficient level) on the
    SAME residual extract any orthogonal signal vs LGBM tree splits or plain
    Ridge?  Negative result is informative (closes the Bayesian-sparsity axis
    on this anchor+feature tuple).

PROTOCOL:
    1. Slice X_K20 = first 20 cols of X_117_unb / X_117_te (pyramid contract).
    2. Anchor: chemprop_aux (PRE-unblind, verified clean).  Residual target
       = y_unb - anchor_unb.
    3. Per fold: StandardScaler.fit on train slice -> transform val ->
       PyMC Horseshoe regression (NUTS, 200 tune + 200 draws, chains=2).
    4. 5-fold scaffold CV on 253 unblind, kf_seed 1001 (single seed -- MCMC slow).
    5. Deploy: refit StandardScaler + Horseshoe on all 253 -> predict on 513
       (posterior mean of beta).

GATE:
    mean_rae < 0.4570 -> "PROMOTE"
            < 0.4598 -> "MARGINAL_BEAT"
            else     -> "FAIL"

Outputs:
    data/processed/nb2772_summary.json
    data/processed/nb2772_pred_oof.npy   (253,) float32 -- horseshoe OOF
    data/processed/te_nb2772.npy         (513,) float32 -- horseshoe deploy
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
from sklearn.preprocessing import StandardScaler

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2772"

# --------------------------------------------------------------------------
# Hyperparameters (spec)
# --------------------------------------------------------------------------
N_TUNE = 200
N_DRAWS = 200
N_CHAINS = 2
N_CORES = 2
PYMC_RANDOM_SEED = 42

# CV protocol -- single seed since MCMC is slow
N_FOLDS = 5
KF_SEED = 1001

# Gates
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

# Paths
X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"
TE_CHEM_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# Number of K=20 cols sliced from the 117-col block (first-20 contract).
K_SLICE = 20


def _fit_horseshoe(X_tr: np.ndarray, y_tr: np.ndarray, seed: int = PYMC_RANDOM_SEED):
    """Fit a hierarchical Bayesian linear regression with horseshoe prior.

    Returns posterior-mean beta vector (K,) and intercept scalar.
    """
    import pymc as pm

    K = X_tr.shape[1]
    with pm.Model():
        # Global shrinkage scale (half-Cauchy(beta=1))
        tau = pm.HalfCauchy("tau", beta=1.0)
        # Local shrinkage scales (one per feature)
        lam = pm.HalfCauchy("lam", beta=1.0, shape=K)
        # Coefficients with horseshoe prior
        beta = pm.Normal("beta", mu=0.0, sigma=lam * tau, shape=K)
        # Intercept (broad normal)
        intercept = pm.Normal("intercept", mu=0.0, sigma=1.0)
        # Observation noise
        sigma = pm.HalfNormal("sigma", 1.0)
        # Linear predictor
        mu = intercept + pm.math.dot(X_tr, beta)
        # Likelihood
        _ = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_tr)
        # NUTS sampler
        idata = pm.sample(
            draws=N_DRAWS,
            tune=N_TUNE,
            chains=N_CHAINS,
            cores=N_CORES,
            random_seed=seed,
            progressbar=False,
            return_inferencedata=True,
            target_accept=0.9,
        )

    post = idata.posterior
    # Posterior mean across (chain, draw) -> coefficient vector
    beta_mean = post["beta"].mean(dim=("chain", "draw")).values.astype(np.float64)
    intercept_mean = float(post["intercept"].mean(dim=("chain", "draw")).values)
    return beta_mean, intercept_mean


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Hierarchical Bayesian Ridge w/ HORSESHOE prior on K=20 residual")
    print("=" * 78)

    # ---- pymc import test ----
    try:
        import pymc as pm  # noqa: F401
    except ImportError as exc:
        print(f"[FATAL] pymc import failed: {exc}")
        print("INSTALL_FAILED")
        return {"verdict": "INSTALL_FAILED", "error": str(exc)}

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

    # ---- Anchor (chemprop_aux, PRE-unblind verified-clean) ----
    if not TE_CHEM_PATH.exists():
        raise FileNotFoundError(f"missing test anchor: {TE_CHEM_PATH}")
    te_chem = np.load(TE_CHEM_PATH).astype(np.float64)
    assert te_chem.shape == (n_test,), f"te_chem shape {te_chem.shape}"
    anchor_unb = te_chem[unb_idx]
    anchor_te = te_chem.copy()
    rae_anchor_unb = float(rae(y_unb, anchor_unb))
    print(f"[anchor] chemprop_aux te[unb_idx] RAE = {rae_anchor_unb:.4f} "
          f"(PRE-clean baseline)")

    # ---- Residual target ----
    resid_unb = y_unb - anchor_unb
    print(f"[resid] mean={resid_unb.mean():+.3f}  std={resid_unb.std():.3f}  "
          f"min={resid_unb.min():+.2f}  max={resid_unb.max():+.2f}")

    # ---- 5-fold scaffold CV with horseshoe (single seed) ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV  kf_seed={KF_SEED}  (single seed -- MCMC slow)")
    print(f"NUTS: draws={N_DRAWS} tune={N_TUNE} chains={N_CHAINS} cores={N_CORES}")
    print("-" * 78)
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=KF_SEED,
    )
    oof_resid = np.full(n_unb, np.nan, dtype=np.float64)
    fold_results = []
    for fi, (tr_loc, va_loc) in enumerate(splits):
        ts = time.time()
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_unb[tr_loc])
        X_va = scaler.transform(X_unb[va_loc])
        y_tr = resid_unb[tr_loc]
        beta_mean, intercept_mean = _fit_horseshoe(X_tr, y_tr, seed=PYMC_RANDOM_SEED + fi)
        oof_resid[va_loc] = intercept_mean + X_va @ beta_mean
        nz = int(np.sum(np.abs(beta_mean) > 1e-3))
        wall = time.time() - ts
        fold_results.append({
            "fold": fi,
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "n_effective_features": nz,
            "beta_l2": float(np.linalg.norm(beta_mean)),
            "intercept": float(intercept_mean),
            "wall_sec": round(wall, 2),
        })
        print(f"   fold={fi}  n_tr={len(tr_loc):3d}  n_va={len(va_loc):3d}  "
              f"|beta|>1e-3 = {nz:2d}/{K_SLICE}  ||beta||_2={np.linalg.norm(beta_mean):.3f}  "
              f"wall={wall:.1f}s")

    assert not np.isnan(oof_resid).any(), "oof_resid has NaN -- fold cover incomplete"
    oof_pred = anchor_unb + oof_resid
    oof_pred = np.clip(oof_pred, 3.0, 8.0)
    pooled_rae = float(rae(y_unb, oof_pred))
    delta_vs_anchor = pooled_rae - rae_anchor_unb
    print(f"\n[cv] pooled scaffold-CV RAE = {pooled_rae:.4f}")
    print(f"[cv] delta vs anchor (chemprop_aux) = {delta_vs_anchor:+.4f}")

    # ---- Deploy: refit StandardScaler + horseshoe on ALL 253 -> 513 ----
    print("\n" + "-" * 78)
    print("DEPLOY: refit StandardScaler+Horseshoe on all 253 -> apply to 513")
    print("-" * 78)
    ts = time.time()
    scaler_full = StandardScaler()
    X_unb_s = scaler_full.fit_transform(X_unb)
    X_te_s = scaler_full.transform(X_te)
    beta_full, intercept_full = _fit_horseshoe(X_unb_s, resid_unb, seed=PYMC_RANDOM_SEED)
    nz_full = int(np.sum(np.abs(beta_full) > 1e-3))
    deploy_resid_te = intercept_full + X_te_s @ beta_full
    deploy_te = anchor_te + deploy_resid_te
    deploy_te = np.clip(deploy_te, 3.0, 8.0).astype(np.float32)
    te_unb_in_rae = float(rae(y_unb, deploy_te[unb_idx].astype(np.float64)))
    print(f"[deploy] wall={time.time()-ts:.1f}s   "
          f"|beta|>1e-3 = {nz_full:2d}/{K_SLICE}   "
          f"intercept={intercept_full:+.4f}")
    print(f"[deploy] te(513) mean={deploy_te.mean():.3f}  std={deploy_te.std():.3f}")
    print(f"[deploy] te[unb_idx] in_RAE={te_unb_in_rae:.4f}  "
          f"(in-sample, deploy refit on all 253)")

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
    print(f"   pooled_rae     = {pooled_rae:.4f}")
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
            "Hierarchical Bayesian linear regression with horseshoe prior "
            "(tau~HalfCauchy(1), lam_j~HalfCauchy(1), beta_j~Normal(0, lam_j*tau)) "
            "on chemprop_aux residual over first-K=20 cols of X_117. "
            f"NUTS draws={N_DRAWS} tune={N_TUNE} chains={N_CHAINS}. n=253 unblind."
        ),
        "paradigm": "bayesian_sparse_shrinkage_horseshoe_prior_k20_residual",
        "anchor_name": "chemprop_aux (PRE-unblind, verified clean)",
        "rae_anchor_unb": rae_anchor_unb,
        "n_tune": N_TUNE,
        "n_draws": N_DRAWS,
        "n_chains": N_CHAINS,
        "n_cores": N_CORES,
        "pymc_random_seed": PYMC_RANDOM_SEED,
        "n_folds": N_FOLDS,
        "kf_seed": KF_SEED,
        "single_seed_reason": "MCMC slow -- single seed per spec",
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": n_uniq_scaf,
        "feat_dim_raw": int(X_unb.shape[1]),
        "k_slice_first_n_of_117": K_SLICE,
        "fold_results": fold_results,
        "mean_rae": pooled_rae,
        "pooled_scaffold_cv_rae": pooled_rae,
        "delta_vs_anchor": delta_vs_anchor,
        "deploy_intercept": float(intercept_full),
        "deploy_n_effective_features": nz_full,
        "deploy_beta_l2": float(np.linalg.norm(beta_full)),
        "deploy_beta_abs_top5": sorted(
            [float(x) for x in np.abs(beta_full)], reverse=True
        )[:5],
        "te_unb_rae_in_sample": te_unb_in_rae,
        "te_deploy_mean": float(deploy_te.mean()),
        "te_deploy_std": float(deploy_te.std()),
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
    print(f"   K=20 raw features              = horseshoe prior (sparse Bayesian)")
    print(f"   NUTS                           = draws={N_DRAWS} tune={N_TUNE} chains={N_CHAINS}")
    print(f"   mean_rae (kf_seed={KF_SEED})       = {pooled_rae:.4f}")
    print(f"   delta vs anchor                = {delta_vs_anchor:+.4f}")
    print(f"   deploy |beta|>1e-3             = {nz_full}/{K_SLICE}")
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
        "delta_vs_anchor",
        "deploy_n_effective_features",
        "deploy_beta_l2",
        "te_unb_rae_in_sample",
        "verdict",
        "te_deploy_mean",
        "te_deploy_std",
        "wall_sec",
    ):
        print(f"  {k}: {res.get(k)}")
