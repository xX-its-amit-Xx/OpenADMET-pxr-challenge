"""nb2573 -- Random Gaussian projection K=20 -> K=100 -> LGBM.

NEW PARADIGM: Johnson-Lindenstrauss random feature expansion. The K=20
substrate has limited axis-aligned split candidates (20 raw dims) for the
LGBM tree splitter; expanding to 100 random Gaussian projections gives a
richer linear-combination basis that LGBM can split on. The intuition is
that JL-projection approximately preserves pairwise distances while
producing many uncorrelated random axes -- each axis is a fresh
linear-combination "view" of the original 20 features. If the residual
signal lives on a non-axis-aligned hyperplane of the K=20 subspace, the
random projection should expose it as a single splittable random axis.

CONTRAST WITH WHITENING (nb2523):
    Whitening (PCA, 20 axes, orthogonal, signal-preserving) FAILED to
    beat raw K=20. Random projection (100 axes, random, distance-
    approximation) is a different bet: not orthogonal-decorrelation but
    DIMENSION EXPANSION via random linear combinations, giving LGBM more
    candidate split directions even if they're noisier individually.

PROTOCOL:
    1. Load X_K20 = first 20 columns of pyramid/X_117_unb.npy (253, 20)
       and pyramid/X_117_te.npy (513, 20).
    2. Anchor: chemprop_aux (PRE-unblind, verified-clean).
       Residual target = y_unb - anchor_unb.
    3. StandardScaler on K=20 features (fit on all unb 253; deploy mirror
       on all 253 for te transform).
    4. GaussianRandomProjection(n_components=100, random_state=42) fit on
       scaled unb features; project to 100 dims.
    5. LGBM(max_depth=4, num_leaves=15, n_estimators=300, lr=0.03) on
       projected 100-d features, fit chemprop_aux residual.
    6. 5-fold scaffold CV on 253, kf_seeds {1001..1005}, mean-bag OOF.
    7. Deploy: refit scaler+projection+LGBM on all 253 -> apply to 513.

GATE: mean_rae < 0.4570 -> "PROMOTE"
      mean_rae < 0.4601 -> "MARGINAL_BEAT"
      else            -> "FAIL"

Outputs:
    scripts/nb2573_random_projection_expand.py
    data/processed/nb2573_summary.json
    data/processed/nb2573_pred_oof.npy   (253,) float32
    data/processed/te_nb2573.npy         (513,) float32
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
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2573"
ANCHOR = "chemprop_aux"
CHEMPROP_AUX_REF = 0.6216

# ---- Random projection ----
N_COMPONENTS = 100
RP_RANDOM_STATE = 42

# ---- LGBM ----
LGBM_MAX_DEPTH = 4
LGBM_NUM_LEAVES = 15
LGBM_N_EST = 300
LGBM_LR = 0.03
LGBM_MIN_CHILD = 5
LGBM_REG_LAMBDA = 2.0

# ---- CV protocol ----
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# ---- Gates ----
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4601

# ---- Paths ----
X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"
TE_CHEM_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"


def _lgbm_params(seed: int) -> dict:
    return dict(
        objective="regression",
        max_depth=LGBM_MAX_DEPTH,
        num_leaves=LGBM_NUM_LEAVES,
        n_estimators=LGBM_N_EST,
        learning_rate=LGBM_LR,
        min_child_samples=LGBM_MIN_CHILD,
        reg_lambda=LGBM_REG_LAMBDA,
        random_state=int(seed),
        n_jobs=2,
        verbosity=-1,
    )


def _rp_cv_one_seed(X: np.ndarray, residual: np.ndarray,
                    unb_scaffolds: list, kf_seed: int) -> np.ndarray:
    """Scaffold 5-fold CV; per fold-train fit StandardScaler+GRP,
    transform fold-val, fit LGBM on projected residual; return OOF residual."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n = len(residual)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in splits:
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(X[tr_loc])
        # Fix RP random_state on RP_RANDOM_STATE per spec; LGBM seed varies.
        rp = GaussianRandomProjection(
            n_components=N_COMPONENTS, random_state=RP_RANDOM_STATE,
        )
        Xtr_p = rp.fit_transform(Xtr_s)
        Xva_p = rp.transform(scaler.transform(X[va_loc]))
        mdl = lgb.LGBMRegressor(**_lgbm_params(kf_seed))
        mdl.fit(Xtr_p, residual[tr_loc])
        oof[va_loc] = mdl.predict(Xva_p).astype(np.float64)
    return oof


def _rp_deploy_te(X_unb: np.ndarray, residual: np.ndarray,
                  X_te: np.ndarray, seed: int) -> np.ndarray:
    """Refit StandardScaler+GRP+LGBM on all 253 unb -> predict 513 te resid."""
    scaler = StandardScaler()
    Xunb_s = scaler.fit_transform(X_unb)
    rp = GaussianRandomProjection(
        n_components=N_COMPONENTS, random_state=RP_RANDOM_STATE,
    )
    Xunb_p = rp.fit_transform(Xunb_s)
    Xte_p = rp.transform(scaler.transform(X_te))
    mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
    mdl.fit(Xunb_p, residual)
    return mdl.predict(Xte_p).astype(np.float64)


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Random Gaussian projection K=20 -> K={N_COMPONENTS} -> LGBM")
    print(f"        anchor={ANCHOR}  scaffold-CV {N_FOLDS}-fold seeds={KF_SEEDS}")
    print(f"        GRP: n_components={N_COMPONENTS}  random_state={RP_RANDOM_STATE}")
    print(f"        LGBM: max_depth={LGBM_MAX_DEPTH} num_leaves={LGBM_NUM_LEAVES} "
          f"n_est={LGBM_N_EST} lr={LGBM_LR}")
    print(f"        GATE: mean_rae < {GATE_PROMOTE} PROMOTE; "
          f"< {GATE_MARGINAL} MARGINAL_BEAT; else FAIL")
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

    # ---- Load X_117 then slice to first 20 cols ----
    X_unb_117 = np.load(X117_UNB_PATH).astype(np.float32)
    X_te_117 = np.load(X117_TE_PATH).astype(np.float32)
    assert X_unb_117.shape == (n_unb, 117), f"X_unb shape {X_unb_117.shape}"
    assert X_te_117.shape == (n_test, 117), f"X_te shape {X_te_117.shape}"
    X_unb = X_unb_117[:, :20].astype(np.float32)
    X_te = X_te_117[:, :20].astype(np.float32)
    print(f"[feat] X_unb_K20={X_unb.shape}  X_te_K20={X_te.shape} "
          f"(first 20 cols of X_117)")

    # ---- Diagnostic: feature stats ----
    print(f"[diag] raw K=20 unb  mean(|mean|)={np.abs(X_unb.mean(axis=0)).mean():.3f}  "
          f"mean(std)={X_unb.std(axis=0).mean():.3f}")

    # ---- Anchor (chemprop_aux, PRE-unblind verified-clean) ----
    if not TE_CHEM_PATH.exists():
        raise FileNotFoundError(f"missing test anchor: {TE_CHEM_PATH}")
    te_chem = np.load(TE_CHEM_PATH).astype(np.float64)
    assert te_chem.shape == (n_test,), f"te_chem shape {te_chem.shape}"
    anchor_unb = te_chem[unb_idx]
    anchor_te = te_chem.copy()
    rae_anchor_unb = float(rae(y_unb, anchor_unb))
    print(f"[anchor] chemprop_aux te[unb_idx] RAE = {rae_anchor_unb:.4f} "
          f"(ref {CHEMPROP_AUX_REF:.4f})")

    # ---- Residual target ----
    resid_unb = y_unb - anchor_unb
    print(f"[resid] mean={resid_unb.mean():+.3f}  std={resid_unb.std():.3f}  "
          f"min={resid_unb.min():+.2f}  max={resid_unb.max():+.2f}")

    # ---- Diagnostic: one global projection to verify shape ----
    print("\n" + "-" * 78)
    print("DIAGNOSTIC: global StandardScaler + GRP (no fold leakage in diag)")
    print("-" * 78)
    diag_scaler = StandardScaler().fit(X_unb)
    diag_rp = GaussianRandomProjection(
        n_components=N_COMPONENTS, random_state=RP_RANDOM_STATE,
    ).fit(diag_scaler.transform(X_unb))
    diag_proj = diag_rp.transform(diag_scaler.transform(X_unb))
    print(f"[diag] projected shape = {diag_proj.shape}  "
          f"mean={diag_proj.mean():+.3f}  std={diag_proj.std():.3f}")
    # JL approximate-distance check (sub-sample pairwise)
    rng = np.random.default_rng(0)
    pi = rng.choice(n_unb, size=min(50, n_unb), replace=False)
    Xs = diag_scaler.transform(X_unb[pi])
    Xp = diag_proj[pi]
    d_in = np.linalg.norm(Xs[:, None, :] - Xs[None, :, :], axis=-1)
    d_proj = np.linalg.norm(Xp[:, None, :] - Xp[None, :, :], axis=-1)
    mask = np.triu(np.ones_like(d_in, dtype=bool), k=1)
    ratio = (d_proj[mask] / np.maximum(d_in[mask], 1e-6)).mean()
    print(f"[diag] JL distance ratio (mean ||proj||/||input|| for pairs) = "
          f"{ratio:.3f}  (expect ~1.0 for distance-preserving JL)")

    # ---- 5-seed scaffold CV under random projection ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV  seeds={KF_SEEDS}")
    print("-" * 78)
    per_seed_oof_resid = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)
    per_seed_te_resid = np.zeros((len(KF_SEEDS), n_test), dtype=np.float64)
    per_seed_rae = []
    for i, kf_seed in enumerate(KF_SEEDS):
        ts = time.time()
        oof_resid_s = _rp_cv_one_seed(X_unb, resid_unb, unb_scaffolds, kf_seed)
        per_seed_oof_resid[i] = oof_resid_s
        per_seed_te_resid[i] = _rp_deploy_te(X_unb, resid_unb, X_te, kf_seed)
        r_s = float(rae(y_unb, anchor_unb + oof_resid_s))
        per_seed_rae.append(r_s)
        print(f"   seed={kf_seed}  rae_corr={r_s:.4f}  wall={time.time()-ts:.1f}s")

    per_seed_mean = float(np.mean(per_seed_rae))
    per_seed_std = float(np.std(per_seed_rae))
    mean_bag_oof_resid = per_seed_oof_resid.mean(axis=0)
    median_bag_oof_resid = np.median(per_seed_oof_resid, axis=0)
    rae_mean_bag = float(rae(y_unb, anchor_unb + mean_bag_oof_resid))
    rae_median_bag = float(rae(y_unb, anchor_unb + median_bag_oof_resid))
    print(f"\n[cv] per_seed_mean RAE = {per_seed_mean:.4f}  std={per_seed_std:.4f}")
    print(f"[cv] mean_bag RAE      = {rae_mean_bag:.4f}")
    print(f"[cv] median_bag RAE    = {rae_median_bag:.4f}")
    print(f"[cv] delta vs anchor   = {rae_mean_bag - rae_anchor_unb:+.4f}")

    # ---- Deploy: mean-bag te ----
    mean_bag_te_resid = per_seed_te_resid.mean(axis=0)
    te_deploy = (anchor_te + mean_bag_te_resid).astype(np.float64)
    te_deploy = np.clip(te_deploy, 3.0, 8.0).astype(np.float32)
    te_unb_in_sample = float(rae(y_unb, te_deploy[unb_idx].astype(np.float64)))
    print(f"\n[deploy] te(513) mean={te_deploy.mean():.3f}  std={te_deploy.std():.3f}")
    print(f"[deploy] te[unb_idx] in_RAE={te_unb_in_sample:.4f}  "
          f"(deploy refit on all 253, in-sample optimism expected)")

    # ---- Save artefacts ----
    pred_oof_corrected = (anchor_unb + mean_bag_oof_resid).astype(np.float32)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof_corrected)
    np.save(te_path, te_deploy)
    print(f"[save] {oof_path}")
    print(f"[save] {te_path}")

    # ---- Gate ----
    mean_rae_for_gate = rae_mean_bag
    if mean_rae_for_gate < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae_for_gate < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "=" * 78)
    print("GATE EVALUATION")
    print("=" * 78)
    print(f"   mean_rae (mean_bag) = {mean_rae_for_gate:.4f}")
    print(f"   < {GATE_PROMOTE:.4f} (PROMOTE)        = "
          f"{mean_rae_for_gate < GATE_PROMOTE}")
    print(f"   < {GATE_MARGINAL:.4f} (MARGINAL_BEAT) = "
          f"{mean_rae_for_gate < GATE_MARGINAL}")
    print(f"   VERDICT             = {verdict}")

    summary = {
        "tag": TAG,
        "method": (
            "Gaussian random projection K=20 -> K=100 expansion, "
            "LGBM on chemprop_aux residual"
        ),
        "paradigm": "JL_random_feature_expansion",
        "anchor": ANCHOR,
        "rae_anchor_unb": rae_anchor_unb,
        "n_components": N_COMPONENTS,
        "rp_random_state": RP_RANDOM_STATE,
        "lgbm_params_sample": _lgbm_params(KF_SEEDS[0]),
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_unb": n_unb,
        "n_te": n_test,
        "n_unique_scaffolds": n_uniq_scaf,
        "feat_dim_input": int(X_unb.shape[1]),
        "feat_dim_projected": int(N_COMPONENTS),
        "jl_distance_ratio_mean": float(ratio),
        "per_seed_rae": [float(r) for r in per_seed_rae],
        "per_seed_mean_rae": per_seed_mean,
        "per_seed_std_rae": per_seed_std,
        "mean_bag_rae": rae_mean_bag,
        "median_bag_rae": rae_median_bag,
        "mean_rae": rae_mean_bag,
        "delta_vs_anchor": rae_mean_bag - rae_anchor_unb,
        "te_unb_rae_in_sample": te_unb_in_sample,
        "te_deploy_mean": float(te_deploy.mean()),
        "te_deploy_std": float(te_deploy.std()),
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
    print(f"   mean_rae (mean_bag)            = {rae_mean_bag:.4f}")
    print(f"   per_seed mean (std)            = {per_seed_mean:.4f} "
          f"(+/- {per_seed_std:.4f})")
    print(f"   delta vs anchor (chemprop_aux) = {rae_mean_bag - rae_anchor_unb:+.4f}")
    print(f"   verdict                        = {verdict}")
    print(f"   wall                           = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "per_seed_mean_rae",
        "per_seed_std_rae",
        "mean_bag_rae",
        "median_bag_rae",
        "mean_rae",
        "rae_anchor_unb",
        "delta_vs_anchor",
        "te_unb_rae_in_sample",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
