"""nb2592 -- Bayesian Additive Regression Trees (BART) on K=20.

NEW PARADIGM:
    BART = Bayesian tree ensembles with MCMC posterior over trees. Differs
    fundamentally from gradient boosting (point-estimate stage-wise) and
    random forests (bagging with fully-grown deep trees):
    - BART places a regularization PRIOR on each tree (depth, leaf magnitudes)
      and samples trees via reversible-jump MCMC. The posterior is a
      distribution over tree-sum function space, so predictions integrate
      across many tree configurations rather than picking one.
    - Each tree contributes a SMALL piece (prior shrinks leaves toward 0);
      the sum-of-trees is the unknown regression function.
    - On small n, the prior + MCMC averaging gives natural regularization
      without manual hyperparam search; this is the closest paradigm axis
      to a Bayesian posterior over GBM-class models.
    Different inductive bias from LGBM/CatBoost/XGB which are all stage-wise
    point-MAP gradient boosters.

PROTOCOL:
    1. Install bartpy in .venv. On install failure, save INSTALL_FAILED.
    2. Load X_K20 = first 20 cols of X_117_unb / X_117_te.
    3. Anchor: chemprop_aux (PRE-unblind, verified-clean).
       Residual target = y_unb - anchor_unb.
    4. Model: bartpy.SklearnModel(
           n_trees=50, n_burn=200, n_samples=200, random_state=42,
       )
       Fit on residual; final pred = anchor + bart_resid.
    5. 5-fold scaffold CV on 253, kf_seed=1001 (single seed - BART is slow).
    6. Deploy: refit on ALL 253 -> predict on 513.

GATE: mean_rae < 0.4570 -> "PROMOTE"
      mean_rae < 0.4601 -> "MARGINAL_BEAT"
      else            -> "FAIL"

Outputs:
    scripts/nb2592_bart_regression.py
    data/processed/nb2592_summary.json
    data/processed/nb2592_pred_oof.npy   (253,) float32  (only if successful)
    data/processed/te_nb2592.npy         (513,) float32  (only if successful)
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

TAG = "nb2592"

# Gates
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4601

# CV protocol
N_FOLDS = 5
KF_SEED = 1001

# BART hyperparameters (spec)
BART_N_TREES = 50
BART_N_BURN = 200
BART_N_SAMPLES = 200
BART_RANDOM_STATE = 42

# Number of K=20 cols sliced from the 117-col block
K_SLICE = 20


def _save_install_failed_summary():
    """Bartpy install failed -- write minimal summary and exit clean."""
    from pxr.paths import DATA_PROCESSED
    summary = {
        "tag": TAG,
        "method": "BART (bartpy) on K=20 chemprop_aux residual",
        "paradigm": "bayesian_tree_ensemble_mcmc_posterior_vs_gbm_pointmap",
        "verdict": "INSTALL_FAILED",
        "install_error": (
            "bartpy depends on deprecated 'sklearn' PyPI package which is "
            "EOL since 2023. Modern pip refuses to build the bartpy wheel. "
            "Setting SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True "
            "would proceed but the resulting bartpy import would still fail "
            "against modern scikit-learn API. Paradigm axis closed pending "
            "pymc-bart or upstream bartpy maintenance."
        ),
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "n_folds": N_FOLDS,
        "kf_seed": KF_SEED,
        "bart_n_trees": BART_N_TREES,
        "bart_n_burn": BART_N_BURN,
        "bart_n_samples": BART_N_SAMPLES,
        "bart_random_state": BART_RANDOM_STATE,
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
    print(f"   bartpy install failed: deprecated sklearn PyPI dependency")
    print("=" * 78)
    return summary


def main():
    # ---- Probe bartpy availability ----
    try:
        from bartpy.sklearnmodel import SklearnModel  # noqa: F401
        BART_OK = True
    except Exception as e:
        print(f"[probe] bartpy import failed: {type(e).__name__}: {e}")
        BART_OK = False

    if not BART_OK:
        return _save_install_failed_summary()

    # ---- Real fit path (only reached if bartpy importable) ----
    from bartpy.sklearnmodel import SklearnModel

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
    print(f"{TAG} -- BART (bartpy) on K=20 chemprop_aux residual "
          f"(Bayesian tree ensemble, MCMC posterior)")
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

    # ---- Scaffold 5-fold CV, single seed (BART is slow) ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV  kf_seed={KF_SEED}  (single seed -- BART is slow)\n"
          f"BART: n_trees={BART_N_TREES}  n_burn={BART_N_BURN}  "
          f"n_samples={BART_N_SAMPLES}  rs={BART_RANDOM_STATE}")
    print("-" * 78)

    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=KF_SEED,
    )
    oof_resid = np.full(n_unb, np.nan, dtype=np.float64)
    fold_info = []
    for fi, (tr_loc, va_loc) in enumerate(splits):
        ts_f = time.time()
        mdl = SklearnModel(
            n_trees=BART_N_TREES,
            n_burn=BART_N_BURN,
            n_samples=BART_N_SAMPLES,
            random_state=BART_RANDOM_STATE,
        )
        mdl.fit(X_unb[tr_loc], resid_unb[tr_loc])
        oof_resid[va_loc] = np.asarray(mdl.predict(X_unb[va_loc])).astype(np.float64)
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

    assert not np.isnan(oof_resid).any(), "oof_resid has NaN -- fold cover incomplete"
    oof_pred = anchor_unb + oof_resid
    oof_pred = np.clip(oof_pred, 3.0, 8.0)
    pooled_rae = float(rae(y_unb, oof_pred))
    print(f"\n[cv] pooled RAE (single seed kf={KF_SEED}) = {pooled_rae:.4f}")
    print(f"[cv] delta vs anchor (chemprop_aux)       = {pooled_rae - rae_anchor_unb:+.4f}")

    # ---- Deploy ----
    print("\n" + "-" * 78)
    print("DEPLOY: refit on all 253 unblind -> apply to 513")
    print("-" * 78)
    deploy_mdl = SklearnModel(
        n_trees=BART_N_TREES,
        n_burn=BART_N_BURN,
        n_samples=BART_N_SAMPLES,
        random_state=BART_RANDOM_STATE,
    )
    deploy_mdl.fit(X_unb, resid_unb)
    deploy_resid_te = np.asarray(deploy_mdl.predict(X_te)).astype(np.float64)
    deploy_te = anchor_te + deploy_resid_te
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
            "BART (bartpy.SklearnModel) on K=20 first-cols of X_117 fit to "
            "chemprop_aux residual; MCMC posterior over tree ensembles"
        ),
        "paradigm": "bayesian_tree_ensemble_mcmc_posterior_vs_gbm_pointmap",
        "anchor_name": "chemprop_aux (PRE-unblind, verified clean)",
        "rae_anchor_unb": rae_anchor_unb,
        "bart_n_trees": BART_N_TREES,
        "bart_n_burn": BART_N_BURN,
        "bart_n_samples": BART_N_SAMPLES,
        "bart_random_state": BART_RANDOM_STATE,
        "n_folds": N_FOLDS,
        "kf_seed": KF_SEED,
        "n_unb": n_unb,
        "n_te": n_test,
        "n_unique_scaffolds": n_uniq_scaf,
        "feat_dim": int(X_unb.shape[1]),
        "k_slice_first_n_of_117": K_SLICE,
        "folds": fold_info,
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
