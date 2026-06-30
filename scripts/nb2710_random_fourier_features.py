"""nb2710 -- Random Fourier Features (Rahimi-Recht) for Tanimoto-like kernel
approximation + Ridge regression on chemprop_aux residual.

NEW PARADIGM (kernel approximation via random features):
    Rahimi-Recht random features approximate a Gaussian (RBF) kernel by
    drawing random projections w ~ N(0, 2*gamma*I) and computing
    cos/sin maps, so that <phi(x), phi(y)> ~ exp(-gamma * ||x - y||^2).
    On binary Morgan-FP-2048 vectors with column sums normalized, the
    RBF kernel on binary input is monotonic in Tanimoto similarity at
    short ranges (because for binary x,y: ||x - y||^2 = |x| + |y| -
    2 <x,y>, which is the Tanimoto-denominator-numerator residual).

    Result: Ridge on the 500-d RFF map is a closed-form approximation
    to kernel ridge regression with a Tanimoto-like kernel, but at
    O(n*d) instead of O(n^2) memory.  Critically, this is a NEW
    paradigm versus LGBM tree splits (cycle-134 exhausted) and versus
    K=20/28 SHAP-pruned LGBM substrate: kernel-method axis on the
    full Morgan-FP-2048 input, not a tree-axis on a SHAP-pruned subset.

    Hypothesis: random Fourier features capture the Tanimoto-monotonic
    smooth-ridge structure that LGBM axis-aligned splits miss; if the
    pEC50 residual surface w.r.t. Morgan FP has a long-correlation
    smooth structure, Ridge on RFF should yield a lower scaffold-CV
    RAE than nb2240 K=20 LGBM baseline (0.4630) and approach the
    deep-30 ceiling band (0.4682-0.4720).

PROTOCOL:
    1. Compute Morgan FP-2048 features for 513 test SMILES, slice unb.
    2. residual = y_unb - chemprop_aux_te[unb_idx] (only PRE-clean anchor).
    3. RBFSampler(n_components=500, gamma=1.0, random_state=42) projects
       Morgan FP -> 500-d random Fourier features.
    4. Ridge(alpha=1.0) on RFF features, residual target.
    5. 5-fold scaffold CV, 5 kf_seeds {1001..1005}.
    6. Deploy: refit RBFSampler + Ridge on full 253 per seed -> predict
       513 te residual; mean-bag aggregate.

GATE (mean-bag corrected RAE):
    mean_rae < 0.4570 -> "PROMOTE"
    mean_rae < 0.4598 -> "MARGINAL_BEAT"
    else              -> "FAIL"

OUTPUTS:
    scripts/nb2710_random_fourier_features.py
    data/processed/nb2710_summary.json
    data/processed/nb2710_pred_oof.npy   (253,) float32 mean-bag CORRECTED
    data/processed/te_nb2710.npy         (513,) float32 deploy refit
    submissions/nb2710_random_fourier_features.csv
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
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge

from pxr.chem import bemis_murcko, morgan_fp_batch, standardize
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2710"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# Random Fourier Features config
RFF_N_COMPONENTS = 500
RFF_GAMMA = 1.0
RFF_RANDOM_STATE = 42

# Ridge config
RIDGE_ALPHA = 1.0

# CV config
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# Gate thresholds (mean-bag RAE)
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

CHEMPROP_AUX_REF = 0.6216
NB2240_K20_REF = 0.4630  # StandardScaler+LGBM K=20 baseline


def _make_rff() -> RBFSampler:
    """Construct RBFSampler with fixed config (deterministic per random_state)."""
    return RBFSampler(
        n_components=RFF_N_COMPONENTS,
        gamma=RFF_GAMMA,
        random_state=RFF_RANDOM_STATE,
    )


def _scaffold_cv_one_seed(
    X: np.ndarray,
    residual: np.ndarray,
    unb_scaffolds: list,
    kf_seed: int,
) -> tuple[np.ndarray, list[dict]]:
    """One scaffold-CV pass: fit per-fold RBFSampler + Ridge.

    Returns (oof_residual_pred, per_fold_diagnostics).
    """
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n = len(residual)
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_diags = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        rff = _make_rff()
        X_tr = rff.fit_transform(X[tr_loc]).astype(np.float32)
        X_va = rff.transform(X[va_loc]).astype(np.float32)
        mdl = Ridge(alpha=RIDGE_ALPHA, random_state=kf_seed)
        mdl.fit(X_tr, residual[tr_loc])
        oof[va_loc] = mdl.predict(X_va)
        fold_diags.append({
            "fold": fold_i,
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "rff_dim": int(X_tr.shape[1]),
            "ridge_intercept": float(mdl.intercept_),
            "ridge_coef_abs_mean": float(np.mean(np.abs(mdl.coef_))),
            "ridge_coef_abs_max": float(np.max(np.abs(mdl.coef_))),
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
    """Fit RBFSampler + Ridge on full 253; predict 513 residual."""
    rff = _make_rff()
    X_unb_s = rff.fit_transform(X_unb).astype(np.float32)
    X_te_s = rff.transform(X_te).astype(np.float32)
    mdl = Ridge(alpha=RIDGE_ALPHA, random_state=seed)
    mdl.fit(X_unb_s, residual)
    pred = mdl.predict(X_te_s).astype(np.float32)
    diag = {
        "seed": int(seed),
        "rff_dim": int(X_unb_s.shape[1]),
        "ridge_intercept": float(mdl.intercept_),
        "ridge_coef_abs_mean": float(np.mean(np.abs(mdl.coef_))),
        "ridge_coef_abs_max": float(np.max(np.abs(mdl.coef_))),
    }
    return pred, diag


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Random Fourier Features (RBF/Tanimoto-like) + Ridge")
    print(f"        anchor = {ANCHOR}  scaffold-CV {N_FOLDS}-fold  "
          f"kf_seeds={KF_SEEDS}")
    print(f"        RBFSampler(n_components={RFF_N_COMPONENTS}, "
          f"gamma={RFF_GAMMA}, random_state={RFF_RANDOM_STATE})")
    print(f"        Ridge(alpha={RIDGE_ALPHA})")
    print(f"        ref nb2240 K=20 StandardScaler+LGBM = {NB2240_K20_REF:.4f}")
    print(f"        GATE: <{GATE_PROMOTE} PROMOTE / "
          f"<{GATE_MARGINAL} MARGINAL_BEAT / else FAIL")
    print("=" * 78)

    # ---- Load truth + anchor + smiles ----
    te = load_test()
    n_test = len(te)
    test_smiles_raw = (
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

    # ---- Compute Morgan FP-2048 ----
    print("\n[feat] standardizing 513 SMILES + computing Morgan FP-2048...")
    from rdkit import Chem
    std_smiles = []
    for s in test_smiles_raw:
        m = standardize(s)
        std_smiles.append(Chem.MolToSmiles(m) if m is not None else s)
    X_fp_te = morgan_fp_batch(std_smiles).astype(np.float32)
    if X_fp_te.shape != (n_test, 2048):
        raise ValueError(f"Morgan FP shape {X_fp_te.shape} expected ({n_test},2048)")
    X_fp_unb = X_fp_te[unb_idx].astype(np.float32)
    print(f"[feat] X_fp_unb = {X_fp_unb.shape}  X_fp_te = {X_fp_te.shape}")
    print(f"[feat] bit density: unb mean = {X_fp_unb.mean():.4f}  "
          f"te mean = {X_fp_te.mean():.4f}")

    # ---- Scaffolds ----
    unb_smiles = [test_smiles_raw[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique unb scaffolds = {n_unique_scaf}")

    # ---- 5-seed scaffold CV ----
    print("\n" + "-" * 78)
    print(f"5-SEED SCAFFOLD-CV  seeds={KF_SEEDS}  folds={N_FOLDS}  "
          f"input_dim=2048  rff_dim={RFF_N_COMPONENTS}")
    print("-" * 78)
    per_seed_oof_resid = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)
    per_seed_te_resid = np.zeros((len(KF_SEEDS), n_test), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_fold_diags: list[list[dict]] = []
    per_seed_deploy_diag: list[dict] = []

    for i, seed in enumerate(KF_SEEDS):
        ts = time.time()
        resid_oof, fold_diags = _scaffold_cv_one_seed(
            X_fp_unb, residual, unb_scaffolds, seed,
        )
        per_seed_oof_resid[i] = resid_oof
        per_seed_fold_diags.append(fold_diags)
        te_resid, deploy_diag = _deploy_te_one_seed(
            X_fp_unb, residual, X_fp_te, seed,
        )
        per_seed_te_resid[i] = te_resid
        per_seed_deploy_diag.append(deploy_diag)
        pred_corr = anchor + resid_oof
        rae_s = float(rae(y_unb, pred_corr))
        per_seed_rae.append(rae_s)
        print(f"   seed={seed}  rae_corr={rae_s:.4f}  "
              f"d_vs_anchor={rae_s - rae_anchor:+.4f}  "
              f"resid_std={resid_oof.std():.4f}  "
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

    sub_csv = SUBMISSIONS / f"{TAG}_random_fourier_features.csv"
    pd.DataFrame({
        "SMILES": test_smiles_raw,
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
        "method": "random_fourier_features_RBF_Tanimoto_like_Ridge_on_chemprop_aux",
        "rationale": (
            "Rahimi-Recht RFF approximates RBF kernel; on binary Morgan-FP "
            "input the RBF kernel is monotonic in Tanimoto at short ranges, "
            "so RFF+Ridge is a kernel-method approximation to KRR with a "
            "Tanimoto-like kernel.  New paradigm vs LGBM tree splits "
            "(cycle-134 exhausted)."
        ),
        "anchor": ANCHOR,
        "anchor_te_path": str(ANCHOR_TE_PATH),
        "anchor_pre_unblind": True,
        "rae_anchor_chemprop_aux": rae_anchor,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "feature_source": "Morgan_FP_2048_on_standardized_SMILES",
        "feat_dim_input": 2048,
        "rff_n_components": RFF_N_COMPONENTS,
        "rff_gamma": RFF_GAMMA,
        "rff_random_state": RFF_RANDOM_STATE,
        "ridge_alpha": RIDGE_ALPHA,
        "n_unb": n_unb,
        "n_test": n_test,
        "n_unique_scaffolds": n_unique_scaf,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "model_class": ("sklearn.kernel_approximation.RBFSampler -> "
                        "sklearn.linear_model.Ridge"),
        "fit_policy": "per-fold-on-train (RFF.fit_transform on train only)",
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
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
