"""nb2601 -- Sparse PCA (n_components=20) + LGBM on chemprop_aux residual.

NEW PARADIGM (vs cycle 202 nb2523 PCA-whitening):
    PCA whitening orthogonalizes 20 features into 20 dense linear
    combinations of all input axes, *destroying* the axis identity that
    LGBM's tree splits depend on (cycle-202 nb2523 result: whitening did
    NOT beat the raw K=20 baseline -- mixing destroyed informative
    splits). Sparse PCA solves a similar variance-rich decomposition but
    each component is a SPARSE linear combination of a few input axes,
    so the resulting components remain interpretable as small axis
    subsets. The hypothesis: variance-rich orthogonal sparse directions
    preserve enough axis identity for tree splits to exploit while still
    removing redundancy among nb2231-K20 features.

PROTOCOL:
    1. Load the pre-cached 117-col 5-way feature matrix on the unblind
       253 from `data/processed/pyramid/X_117_unb.npy` and on the 513
       test from `data/processed/pyramid/X_117_te.npy`.  (We use the
       full 117 columns -- task spec says "Load X_117 from
       data/processed/pyramid/X_117_unb.npy". This is the same
       substrate the K=20 subset was carved from; SparsePCA picks 20
       components from the 117-d space.)
    2. Build chemprop_aux residual on the 253 unblind compounds (only
       verified-clean PRE-unblind anchor).
    3. 5-fold scaffold CV on the 253, 5 kf_seeds {1001..1005}:
         per fold-train: StandardScaler then
            SparsePCA(n_components=20, alpha=1.0, random_state=42)
         transform fold-val with the same scaler+SparsePCA
         fit LGBM (max_depth=4, num_leaves=15, n_est=300, lr=0.03)
         predict fold-val residual.
       Aggregate per-seed OOF; report per_seed RAE and mean-bag RAE.
    4. Deploy 513-test: refit StandardScaler+SparsePCA on all 253 unb,
       refit LGBM on full residual per seed, predict 513 test residual.
    5. Gate (mean-bag RAE):
         mean_rae < 0.4570  -> PROMOTE
         mean_rae < 0.4601  -> MARGINAL_BEAT
         else               -> FAIL

OUTPUTS:
    scripts/nb2601_sparse_pca_k20.py
    data/processed/nb2601_summary.json
    data/processed/nb2601_pred_oof.npy   (253,) float32  mean-bag corrected
    data/processed/te_nb2601.npy         (513,) float32  deploy refit
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
from sklearn.decomposition import SparsePCA
from sklearn.preprocessing import StandardScaler

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2601"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

PYRAMID_DIR = DATA_PROCESSED / "pyramid"
X_UNB_PATH = PYRAMID_DIR / "X_117_unb.npy"
X_TE_PATH = PYRAMID_DIR / "X_117_te.npy"

N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
N_COMPONENTS = 20
SPCA_ALPHA = 1.0
SPCA_RANDOM_STATE = 42

# Gate thresholds (mean-bag RAE)
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4601

CHEMPROP_AUX_REF = 0.6216


def _lgbm_params(seed: int) -> dict:
    """LGBM hyperparams as requested in task spec."""
    return dict(
        objective="regression",
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


def _spca_cv_one_seed(X: np.ndarray, residual: np.ndarray,
                      unb_scaffolds: list, kf_seed: int) -> np.ndarray:
    """Scaffold 5-fold CV; per fold-train fit StandardScaler+SparsePCA,
    transform fold-val, fit LGBM on sparse-PC residual; return OOF
    residual prediction."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n = len(residual)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in splits:
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(X[tr_loc])
        spca = SparsePCA(
            n_components=N_COMPONENTS,
            alpha=SPCA_ALPHA,
            random_state=SPCA_RANDOM_STATE,
            max_iter=200,
            tol=1e-3,
            n_jobs=1,
        )
        Xtr_z = spca.fit_transform(Xtr_s)
        Xva_z = spca.transform(scaler.transform(X[va_loc]))
        mdl = lgb.LGBMRegressor(**_lgbm_params(kf_seed))
        mdl.fit(Xtr_z, residual[tr_loc])
        oof[va_loc] = mdl.predict(Xva_z)
    return oof


def _spca_deploy_te(X_unb: np.ndarray, residual: np.ndarray,
                    X_te: np.ndarray, seed: int) -> tuple:
    """Refit StandardScaler+SparsePCA on all 253 unb features, then
    LGBM on residual, predict 513 te residual.  Returns (te_resid_pred,
    n_nonzero_per_component)."""
    scaler = StandardScaler()
    Xunb_s = scaler.fit_transform(X_unb)
    spca = SparsePCA(
        n_components=N_COMPONENTS,
        alpha=SPCA_ALPHA,
        random_state=SPCA_RANDOM_STATE,
        max_iter=200,
        tol=1e-3,
        n_jobs=1,
    )
    Xunb_z = spca.fit_transform(Xunb_s)
    Xte_z = spca.transform(scaler.transform(X_te))
    mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
    mdl.fit(Xunb_z, residual)
    # Sparse PCA components_ has shape (n_components, n_features)
    n_nonzero = (np.abs(spca.components_) > 1e-9).sum(axis=1)
    return (
        mdl.predict(Xte_z).astype(np.float32),
        n_nonzero.astype(int),
    )


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- SparsePCA(n={N_COMPONENTS},alpha={SPCA_ALPHA}) + LGBM "
          f"on chemprop_aux residual")
    print(f"        anchor = {ANCHOR}  scaffold-CV {N_FOLDS}-fold")
    print(f"        kf_seeds = {KF_SEEDS}")
    print(f"        GATE: mean_rae < {GATE_PROMOTE} PROMOTE; "
          f"< {GATE_MARGINAL} MARGINAL_BEAT; else FAIL")
    print("=" * 78)

    # ---- load pre-cached 117 features ----
    if not X_UNB_PATH.exists():
        raise FileNotFoundError(f"missing {X_UNB_PATH}")
    if not X_TE_PATH.exists():
        raise FileNotFoundError(f"missing {X_TE_PATH}")
    X_unb_117 = np.load(X_UNB_PATH).astype(np.float32)
    X_te_117 = np.load(X_TE_PATH).astype(np.float32)
    print(f"[load] X_unb_117 = {X_unb_117.shape}   X_te_117 = {X_te_117.shape}")

    # sanitise non-finite (defensive; pyramid cache should be clean)
    X_unb_117 = np.where(np.isfinite(X_unb_117), X_unb_117, 0.0).astype(np.float32)
    X_te_117 = np.where(np.isfinite(X_te_117), X_te_117, 0.0).astype(np.float32)

    # ---- load truth + anchor + scaffolds ----
    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")
    if X_unb_117.shape[0] != n_unb:
        raise ValueError(f"X_unb rows {X_unb_117.shape[0]} != n_unb {n_unb}")
    if X_te_117.shape[0] != n_test:
        raise ValueError(f"X_te rows {X_te_117.shape[0]} != n_test {n_test}")

    unb_smiles = [test_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique scaffolds in unb 253 = {n_unique_scaf}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- 5-seed scaffold CV under SparsePCA(20) ----
    print("\n" + "-" * 78)
    print(f"SPARSE-PCA(K={N_COMPONENTS}, alpha={SPCA_ALPHA}) 5-FOLD SCAFFOLD CV  "
          f"seeds={KF_SEEDS}")
    print("-" * 78)
    per_seed_oof = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)
    per_seed_te_resid = np.zeros((len(KF_SEEDS), n_test), dtype=np.float64)
    per_seed_rae = []
    per_seed_n_nonzero = []
    for i, s in enumerate(KF_SEEDS):
        ts = time.time()
        oof_s = _spca_cv_one_seed(X_unb_117, residual, unb_scaffolds, s)
        per_seed_oof[i] = oof_s
        te_resid_s, n_nz_s = _spca_deploy_te(X_unb_117, residual, X_te_117, s)
        per_seed_te_resid[i] = te_resid_s
        per_seed_n_nonzero.append(n_nz_s)
        r_s = float(rae(y_unb, anchor + oof_s))
        per_seed_rae.append(r_s)
        print(f"   seed={s}  rae_corr={r_s:.4f}  "
              f"spca_n_nz(mean/min/max)={n_nz_s.mean():.1f}/"
              f"{n_nz_s.min()}/{n_nz_s.max()}  wall={time.time()-ts:.1f}s")

    per_seed_mean = float(np.mean(per_seed_rae))
    per_seed_std = float(np.std(per_seed_rae))
    mean_bag_oof = per_seed_oof.mean(axis=0)
    median_bag_oof = np.median(per_seed_oof, axis=0)
    rae_mean_bag = float(rae(y_unb, anchor + mean_bag_oof))
    rae_median_bag = float(rae(y_unb, anchor + median_bag_oof))
    print(f"\n[cv] per_seed_mean RAE = {per_seed_mean:.4f}  std={per_seed_std:.4f}")
    print(f"[cv] mean_bag RAE      = {rae_mean_bag:.4f}")
    print(f"[cv] median_bag RAE    = {rae_median_bag:.4f}")
    delta_vs_anchor = rae_mean_bag - rae_anchor
    print(f"[cv] delta vs chemprop_aux anchor = {delta_vs_anchor:+.4f}")

    # ---- deploy te ----
    mean_bag_te_resid = per_seed_te_resid.mean(axis=0)
    te_deploy = (te_anchor_513 + mean_bag_te_resid).astype(np.float32)
    te_unb_in_sample = float(rae(y_unb, te_deploy[unb_idx]))
    print(f"\n[deploy] te(513) mean/std = {te_deploy.mean():.3f}/{te_deploy.std():.3f}")
    print(f"[deploy] te[unb_idx] in-sample RAE = {te_unb_in_sample:.4f}  "
          f"(deploy refit, in-sample optimism expected)")

    # ---- save artefacts ----
    pred_oof_corrected = (anchor + mean_bag_oof).astype(np.float32)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof_corrected)
    np.save(te_path, te_deploy)
    print(f"[save] {oof_path}")
    print(f"[save] {te_path}")

    # ---- gate ----
    if rae_mean_bag < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif rae_mean_bag < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "=" * 78)
    print("GATE EVALUATION")
    print("=" * 78)
    print(f"   mean_rae (mean_bag) = {rae_mean_bag:.4f}")
    print(f"   < {GATE_PROMOTE:.4f} (PROMOTE)        = {rae_mean_bag < GATE_PROMOTE}")
    print(f"   < {GATE_MARGINAL:.4f} (MARGINAL_BEAT) = {rae_mean_bag < GATE_MARGINAL}")
    print(f"   VERDICT             = {verdict}")

    # n_nonzero summary (deploy SparsePCA component sparsity)
    n_nonzero_stack = np.stack(per_seed_n_nonzero, axis=0)  # (n_seeds, n_components)
    summary = {
        "tag": TAG,
        "method": "sparse_pca_K20_alpha1_LGBM_residual_on_chemprop_aux",
        "anchor": ANCHOR,
        "rae_anchor_chemprop_aux": rae_anchor,
        "n_unb": n_unb,
        "n_te": n_test,
        "n_unique_scaffolds": n_unique_scaf,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "input_shape_unb": list(X_unb_117.shape),
        "input_shape_te": list(X_te_117.shape),
        "lgbm_params_sample": _lgbm_params(KF_SEEDS[0]),
        "spca_params": {
            "n_components": N_COMPONENTS,
            "alpha": SPCA_ALPHA,
            "random_state": SPCA_RANDOM_STATE,
        },
        "deploy_spca_n_nonzero_per_component_mean": n_nonzero_stack.mean(axis=0).tolist(),
        "deploy_spca_n_nonzero_overall_mean": float(n_nonzero_stack.mean()),
        "deploy_spca_n_nonzero_overall_min": int(n_nonzero_stack.min()),
        "deploy_spca_n_nonzero_overall_max": int(n_nonzero_stack.max()),
        "per_seed_rae": [float(r) for r in per_seed_rae],
        "per_seed_mean_rae": per_seed_mean,
        "per_seed_std_rae": per_seed_std,
        "mean_bag_rae": rae_mean_bag,
        "median_bag_rae": rae_median_bag,
        "mean_rae": rae_mean_bag,
        "delta_vs_anchor": delta_vs_anchor,
        "te_unb_in_sample_rae": te_unb_in_sample,
        "te_deploy_mean": float(te_deploy.mean()),
        "te_deploy_std": float(te_deploy.std()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"\nwall = {time.time() - t0:.1f}s")
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "per_seed_mean_rae",
        "per_seed_std_rae",
        "mean_bag_rae",
        "median_bag_rae",
        "delta_vs_anchor",
        "te_unb_in_sample_rae",
        "deploy_spca_n_nonzero_overall_mean",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
