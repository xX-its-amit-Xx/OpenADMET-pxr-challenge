"""nb2550 -- Elastic Net (linear L1+L2 baseline) on K=20 substrate.

NEW PARADIGM:
    Simplest possible model class -- linear with L1+L2 regularisation.  This
    gives the FLOOR for what 20 linearly-informative features can do on the
    chemprop_aux residual.  If a linear model on K=20 already gets close to
    the LGBM ceiling (nb2240 K=20 mean-bag = 0.4630), it means the K=20
    signal is mostly linear and tree non-linearity is not buying much;
    conversely, if Elastic Net is far away, the K=20 -> residual map is
    non-linear and tree-splits are doing real work.

    Either way, this is the linear baseline reference point for the K=20
    substrate, complementing the cross-paradigm-exhaustion memo (cycles
    134-139) by adding ONE more model class on the same fixed substrate
    tuple (chemprop_aux anchor + K=20 features).

PROTOCOL:
    1. Load X_117 (test 513) -> slice to nb2240's K=20 indices (idx 45, 67,
       66, 68, 65, 92, 27, 77, 81, 56, 1, 7, 115, 93, 80, 11, 70, 54, 8, 57
       from nb2231 RFE).
    2. residual = y_unb - chemprop_aux[unb_idx]   (only PRE-clean anchor)
    3. Per scaffold-fold: StandardScaler fit on train fold -> transform val.
    4. ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9], cv=3,
                    random_state=42, max_iter=5000) on residual.
    5. 5-fold scaffold CV on 253 unblind, 5 kf_seeds {1001..1005}.
       (scaffold_kfold_indices per cv-protocol-audit memo; random KFold
        forbidden for ladder decisions.)
    6. Aggregate per-seed corrected RAE -> mean-bag.
    7. Deploy: refit ElasticNetCV on full 253 per seed, predict 513
       residual, mean-bag.

GATE (on mean-bag corrected RAE):
    mean_rae < 0.4570 -> "PROMOTE"
    mean_rae < 0.4601 -> "MARGINAL_BEAT"
    else              -> "FAIL"

OUTPUTS:
    scripts/nb2550_elasticnet_k20.py
    data/processed/nb2550_summary.json
    data/processed/nb2550_pred_oof.npy   (253,) float32 mean-bag CORRECTED
    data/processed/te_nb2550.npy         (513,) float32 deploy refit
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
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2550"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"
NB2240_SUMMARY = DATA_PROCESSED / "nb2240_summary.json"

N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4601

# ElasticNetCV hyperparameters (per task spec)
L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]
INNER_CV = 3
ENET_RANDOM_STATE = 42
ENET_MAX_ITER = 5000

CHEMPROP_AUX_REF = 0.6216
NB2240_K20_REF = 0.4630  # LGBM K=20 mean-bag reference for comparison


def _make_enet() -> ElasticNetCV:
    """Fresh ElasticNetCV with the spec'd l1_ratio grid + 3-fold inner CV."""
    return ElasticNetCV(
        l1_ratio=L1_RATIOS,
        cv=INNER_CV,
        random_state=ENET_RANDOM_STATE,
        max_iter=ENET_MAX_ITER,
        n_jobs=1,
        selection="cyclic",
    )


def _scaffold_cv_one_seed(
    X: np.ndarray,
    residual: np.ndarray,
    unb_scaffolds: list,
    kf_seed: int,
) -> tuple[np.ndarray, list[dict]]:
    """One scaffold-CV pass: fit per-fold StandardScaler + ElasticNetCV.

    Returns (oof_residual_pred, per_fold_diagnostics).
    """
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n = len(residual)
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_diags = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_loc]).astype(np.float64)
        X_va = scaler.transform(X[va_loc]).astype(np.float64)
        mdl = _make_enet()
        mdl.fit(X_tr, residual[tr_loc])
        oof[va_loc] = mdl.predict(X_va)
        n_nonzero = int(np.sum(np.abs(mdl.coef_) > 1e-10))
        fold_diags.append({
            "fold": fold_i,
            "alpha": float(mdl.alpha_),
            "l1_ratio": float(mdl.l1_ratio_),
            "n_nonzero_coef": n_nonzero,
            "intercept": float(mdl.intercept_),
        })
    return oof, fold_diags


def _deploy_te_one_seed(
    X_unb: np.ndarray,
    residual: np.ndarray,
    X_te: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Fit StandardScaler + ElasticNetCV on full 253, predict 513 residual."""
    scaler = StandardScaler()
    X_unb_s = scaler.fit_transform(X_unb).astype(np.float64)
    X_te_s = scaler.transform(X_te).astype(np.float64)
    mdl = _make_enet()
    mdl.fit(X_unb_s, residual)
    pred = mdl.predict(X_te_s).astype(np.float32)
    diag = {
        "alpha": float(mdl.alpha_),
        "l1_ratio": float(mdl.l1_ratio_),
        "n_nonzero_coef": int(np.sum(np.abs(mdl.coef_) > 1e-10)),
        "intercept": float(mdl.intercept_),
    }
    return pred, diag


# ============================================================================
# MAIN
# ============================================================================
def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- ElasticNetCV linear baseline on K=20 substrate")
    print(f"        anchor = {ANCHOR}  scaffold-CV {N_FOLDS}-fold")
    print(f"        kf_seeds = {KF_SEEDS}")
    print(f"        ElasticNetCV(l1_ratio={L1_RATIOS}, cv={INNER_CV}, "
          f"random_state={ENET_RANDOM_STATE})")
    print(f"        per-fold StandardScaler (fit-on-train-only)")
    print(f"        GATE: mean_rae < {GATE_PROMOTE} PROMOTE; "
          f"< {GATE_MARGINAL} MARGINAL_BEAT; else FAIL")
    print("=" * 78)

    # ---- Load truth + anchor ----
    te = load_test()
    n_test = len(te)
    test_smiles = (
        te["smiles"].astype(str).tolist()
        if "smiles" in te.columns
        else te["SMILES"].astype(str).tolist()
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

    # ---- Load X_117 substrate + slice to K=20 ----
    if not X117_UNB_PATH.exists() or not X117_TE_PATH.exists():
        raise FileNotFoundError(
            f"X_117 cache missing: {X117_UNB_PATH} or {X117_TE_PATH}"
        )
    X117_unb = np.load(X117_UNB_PATH).astype(np.float32)
    X117_te = np.load(X117_TE_PATH).astype(np.float32)
    if X117_unb.shape != (n_unb, 117):
        raise ValueError(f"X117_unb shape {X117_unb.shape}")
    if X117_te.shape != (n_test, 117):
        raise ValueError(f"X117_te shape {X117_te.shape}")
    # Sanitize NaN/Inf carried in cache
    X117_unb = np.where(np.isfinite(X117_unb), X117_unb, 0.0).astype(np.float32)
    X117_te = np.where(np.isfinite(X117_te), X117_te, 0.0).astype(np.float32)
    print(f"[feat] X117_unb = {X117_unb.shape}  X117_te = {X117_te.shape}")

    # Load K=20 surviving indices from nb2240 (which mirrors nb2231 RFE)
    if not NB2240_SUMMARY.exists():
        raise FileNotFoundError(f"nb2240 summary missing: {NB2240_SUMMARY}")
    with open(NB2240_SUMMARY) as f:
        nb2240 = json.load(f)
    k20_idx = list(nb2240["k20_surviving_idx_in_117"])
    k20_names = list(nb2240["k20_surviving_names"])
    assert len(k20_idx) == 20, f"expected 20 K=20 indices, got {len(k20_idx)}"
    print(f"[K20] loaded {len(k20_idx)} surviving indices from nb2240")
    print(f"[K20] family counts: {nb2240.get('k20_family_counts', {})}")

    X_unb = X117_unb[:, k20_idx].astype(np.float32)
    X_te = X117_te[:, k20_idx].astype(np.float32)
    feat_dim = X_unb.shape[1]
    assert feat_dim == 20, f"feat_dim {feat_dim} != 20"
    print(f"[feat] X_unb (K=20) = {X_unb.shape}  X_te = {X_te.shape}")

    # ---- Scaffolds ----
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
    per_seed_deploy_diag: list[dict] = []

    for i, seed in enumerate(KF_SEEDS):
        ts = time.time()
        resid_oof, fold_diags = _scaffold_cv_one_seed(
            X_unb, residual, unb_scaffolds, seed,
        )
        per_seed_oof_resid[i] = resid_oof
        per_seed_fold_diags.append(fold_diags)
        te_resid, deploy_diag = _deploy_te_one_seed(X_unb, residual, X_te)
        per_seed_te_resid[i] = te_resid
        per_seed_deploy_diag.append(deploy_diag)
        pred_corr = anchor + resid_oof
        rae_s = float(rae(y_unb, pred_corr))
        per_seed_rae.append(rae_s)
        # Per-seed summary: mean alpha/l1 across folds + deploy
        mean_alpha = float(np.mean([d["alpha"] for d in fold_diags]))
        mean_l1 = float(np.mean([d["l1_ratio"] for d in fold_diags]))
        mean_nnz = float(np.mean([d["n_nonzero_coef"] for d in fold_diags]))
        print(f"   seed={seed}  rae_corr={rae_s:.4f}  "
              f"d_vs_anchor={rae_s - rae_anchor:+.4f}  "
              f"alpha~{mean_alpha:.4f}  l1~{mean_l1:.2f}  "
              f"nnz~{mean_nnz:.1f}/{feat_dim}  "
              f"deploy_alpha={deploy_diag['alpha']:.4f}  "
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
    print(f"[cv] reference  nb2240 K=20 LGBM mean-bag = {NB2240_K20_REF:.4f}  "
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
        "method": ("elasticnet_cv_linear_baseline_on_X117_K20_residual_"
                   "on_chemprop_aux"),
        "anchor": ANCHOR,
        "anchor_te_path": str(ANCHOR_TE_PATH),
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
        "model_class": "sklearn.linear_model.ElasticNetCV",
        "elasticnet_params": {
            "l1_ratio_grid": L1_RATIOS,
            "inner_cv": INNER_CV,
            "random_state": ENET_RANDOM_STATE,
            "max_iter": ENET_MAX_ITER,
            "selection": "cyclic",
            "scaler": "StandardScaler (per-fold fit-on-train)",
        },
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
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "verdict": verdict,
        "pre_unblind_clean_anchor": True,
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
