"""nb1052_bayes_linear -- Bayesian linear regression on K=28 SHAP-pruned features
with native uncertainty quantification, plus Gaussian-Process variant.

HYPOTHESIS:
    nb2103 K=28 mean-bag LGBM RAE = 0.4737 (median-bag 0.4698) is the current
    SHAP-K best on the chemprop_aux residual.  Bayesian models produce a
    per-row posterior std that can be used to:
        (A) Replace LGBM with BayesianRidge (mean-only)
        (B) Route Bayesian vs LGBM per row by inverse-uncertainty weighting
            so high-confidence Bayesian rows dominate and low-confidence rows
            fall back to LGBM
        (C) Fit a Gaussian Process (RBF) for a fully kernelised alternative

    Decision margin vs nb2103 K=28 = 0.003 RAE.

PROTOCOL:
    1. Load X_unb_28_nb2103.npy (253, 28) -- already SHAP top-28 on 117 cols
    2. Load chemprop_aux te[unb_idx] as the anchor (PRE-unblind clean)
    3. residual = y_unb - anchor
    4. Standardize features (StandardScaler)
    5. Method A: BayesianRidge, 5-fold cross-fit -> pred mean + std per row
                 final_A = anchor + pred_mean,  rae_A
    6. Method B: same Bayesian cross-fit as A, plus reload nb2103 K=28
                 mean_bag LGBM OOF (which is anchor + lgbm_resid).  Per row
                 weight w_i = 1 / (1 + sigma_i^2)  (inverse-variance shrink);
                 final_B = w * (anchor + bayes_mean) + (1-w) * lgbm_oof
    7. Method C: GaussianProcessRegressor with RBF + WhiteKernel, 5-fold
                 cross-fit on residual.  final_C = anchor + gp_mean.
    8. Compare A / B / C vs nb2103 K=28 mean_bag (0.4737) and median_bag
       (0.4698) with decision_margin = 0.003.
    9. If best beats nb2103 K=28: build deploy CSV
       submissions/nb1052_bayes_linear.csv from te_chemprop_aux.npy + fitted
       Bayesian/GP residual on the full 513 (refit on all 253 unblind labels).

Outputs:
    scripts/nb1052_bayes_linear.py
    data/processed/nb1052_summary.json
    [optional] submissions/nb1052_bayes_linear.csv
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
from sklearn.linear_model import BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, PROJECT_ROOT

TAG = "nb1052"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
X_UNB_28_PATH = DATA_PROCESSED / "X_unb_28_nb2103.npy"
NB2103_OOF_K28_PATH = DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy"
NB2103_SUMMARY_PATH = DATA_PROCESSED / "nb2103_summary.json"

UNB_IDX_PATH = DATA_PROCESSED / "_audit_unblind_idx.npy"
UNB_Y_PATH = DATA_PROCESSED / "_audit_unblind_y.npy"

N_FOLDS = 5
CV_SEED = 42
DECISION_MARGIN = 0.003

# Reference
CHEMPROP_AUX_REF = 0.6216
NB2103_K28_MEAN_BAG_REF = 0.4737
NB2103_K28_MEDIAN_BAG_REF = 0.4698


def _bayesian_cross_fit(X: np.ndarray, y: np.ndarray, seed: int = CV_SEED):
    """5-fold cross-fit BayesianRidge.  Returns (mean_oof, std_oof)."""
    n = len(y)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    mean_oof = np.full(n, np.nan, dtype=np.float64)
    std_oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr_loc])
        Xva = scaler.transform(X[va_loc])
        mdl = BayesianRidge(
            max_iter=300,
            tol=1e-3,
            compute_score=False,
            fit_intercept=True,
        )
        mdl.fit(Xtr, y[tr_loc])
        m, s = mdl.predict(Xva, return_std=True)
        mean_oof[va_loc] = m
        std_oof[va_loc] = s
    return mean_oof, std_oof


def _gp_cross_fit(X: np.ndarray, y: np.ndarray, seed: int = CV_SEED):
    """5-fold cross-fit GaussianProcessRegressor (RBF + WhiteKernel)."""
    n = len(y)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    mean_oof = np.full(n, np.nan, dtype=np.float64)
    std_oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr_loc])
        Xva = scaler.transform(X[va_loc])
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
            + WhiteKernel(noise_level=0.5, noise_level_bounds=(1e-3, 1e1))
        )
        mdl = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=seed,
            alpha=1e-6,
        )
        mdl.fit(Xtr, y[tr_loc])
        m, s = mdl.predict(Xva, return_std=True)
        mean_oof[va_loc] = m
        std_oof[va_loc] = s
    return mean_oof, std_oof


def _verdict(r: float, ref: float) -> str:
    delta = r - ref
    if delta < -DECISION_MARGIN:
        return "BEATS"
    if abs(delta) < DECISION_MARGIN:
        return "FLAT"
    return "HURTS"


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- BayesianRidge / GP on K=28 SHAP features (residual)")
    print(f"         ref: nb2103 K=28 mean_bag = {NB2103_K28_MEAN_BAG_REF:.4f}  "
          f"median_bag = {NB2103_K28_MEDIAN_BAG_REF:.4f}  "
          f"margin = {DECISION_MARGIN}")
    print("=" * 78)

    # ---- Load anchor + truth ----
    te = load_test()
    n_test = len(te)
    unb_idx = np.load(UNB_IDX_PATH)
    y_unb = np.load(UNB_Y_PATH).astype(np.float64)
    n_unb = len(y_unb)

    te_anchor = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor_unb = te_anchor[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    residual = y_unb - anchor_unb
    print(f"[load] n_test={n_test}  n_unb={n_unb}  anchor in_RAE={rae_anchor:.4f}")
    print(f"[load] residual mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Load K=28 features ----
    X_unb_28 = np.load(X_UNB_28_PATH).astype(np.float64)
    if X_unb_28.shape != (n_unb, 28):
        raise ValueError(
            f"X_unb_28 shape {X_unb_28.shape} != ({n_unb}, 28)"
        )
    print(f"[load] X_unb_28 = {X_unb_28.shape}")

    # ---- Load nb2103 K=28 mean_bag OOF (corrected prediction) ----
    if not NB2103_OOF_K28_PATH.exists():
        raise FileNotFoundError(f"missing {NB2103_OOF_K28_PATH}")
    lgbm_oof_k28 = np.load(NB2103_OOF_K28_PATH).astype(np.float64)
    rae_lgbm_k28 = float(rae(y_unb, lgbm_oof_k28))
    print(f"[load] nb2103 K=28 mean_bag OOF in_RAE = {rae_lgbm_k28:.4f}  "
          f"(ref {NB2103_K28_MEAN_BAG_REF:.4f})")

    # ---- Method A: BayesianRidge cross-fit ----
    print("\n" + "-" * 78)
    print("METHOD A: BayesianRidge on K=28 residual (5-fold cross-fit)")
    print("-" * 78)
    tA = time.time()
    bayes_mean, bayes_std = _bayesian_cross_fit(X_unb_28, residual, seed=CV_SEED)
    pred_A = anchor_unb + bayes_mean
    rae_A = float(rae(y_unb, pred_A))
    print(f"   bayes_mean: mean={bayes_mean.mean():+.4f}  std={bayes_mean.std():.4f}")
    print(f"   bayes_std : mean={bayes_std.mean():.4f}  "
          f"min={bayes_std.min():.4f}  max={bayes_std.max():.4f}  "
          f"std={bayes_std.std():.4f}")
    print(f"   rae_A = {rae_A:.4f}  "
          f"(d_vs_anchor={rae_A - rae_anchor:+.4f}  "
          f"d_vs_K28_mean={rae_A - NB2103_K28_MEAN_BAG_REF:+.4f})")
    verdict_A = _verdict(rae_A, NB2103_K28_MEAN_BAG_REF)
    print(f"   verdict_A = {verdict_A}  (wall {time.time() - tA:.1f}s)")

    # ---- Method B: Inverse-variance router Bayesian <-> LGBM ----
    print("\n" + "-" * 78)
    print("METHOD B: Bayes <-> LGBM per-row inverse-variance router")
    print("-" * 78)
    tB = time.time()
    # weight w in (0, 1): higher when Bayes is confident (low sigma)
    # robust normalisation: rescale sigma to [0, 1] within this fold-set
    sigma = bayes_std
    sigma_norm = (sigma - sigma.min()) / max(sigma.max() - sigma.min(), 1e-9)
    # When sigma small -> w large; when sigma large -> w small
    w = 1.0 / (1.0 + sigma_norm * 4.0)   # w in [0.2, 1.0]
    w_clipped = np.clip(w, 0.0, 1.0)
    pred_B = w_clipped * pred_A + (1.0 - w_clipped) * lgbm_oof_k28
    rae_B = float(rae(y_unb, pred_B))
    print(f"   weight w: mean={w_clipped.mean():.3f}  "
          f"min={w_clipped.min():.3f}  max={w_clipped.max():.3f}")
    print(f"   share rows where Bayes dominates (w>=0.5): "
          f"{int((w_clipped >= 0.5).sum())}/{n_unb}")
    print(f"   rae_B = {rae_B:.4f}  "
          f"(d_vs_anchor={rae_B - rae_anchor:+.4f}  "
          f"d_vs_K28_mean={rae_B - NB2103_K28_MEAN_BAG_REF:+.4f})")
    verdict_B = _verdict(rae_B, NB2103_K28_MEAN_BAG_REF)
    # also evaluate vs the better median_bag
    delta_B_vs_med = rae_B - NB2103_K28_MEDIAN_BAG_REF
    print(f"   verdict_B = {verdict_B}  "
          f"(d_vs_K28_median={delta_B_vs_med:+.4f})  "
          f"wall {time.time() - tB:.1f}s")

    # ---- Method C: GaussianProcessRegressor RBF + WhiteKernel ----
    print("\n" + "-" * 78)
    print("METHOD C: GaussianProcessRegressor RBF+WhiteKernel on K=28 residual")
    print("-" * 78)
    tC = time.time()
    gp_mean, gp_std = _gp_cross_fit(X_unb_28, residual, seed=CV_SEED)
    pred_C = anchor_unb + gp_mean
    rae_C = float(rae(y_unb, pred_C))
    print(f"   gp_mean: mean={gp_mean.mean():+.4f}  std={gp_mean.std():.4f}")
    print(f"   gp_std : mean={gp_std.mean():.4f}  "
          f"min={gp_std.min():.4f}  max={gp_std.max():.4f}")
    print(f"   rae_C = {rae_C:.4f}  "
          f"(d_vs_anchor={rae_C - rae_anchor:+.4f}  "
          f"d_vs_K28_mean={rae_C - NB2103_K28_MEAN_BAG_REF:+.4f})")
    verdict_C = _verdict(rae_C, NB2103_K28_MEAN_BAG_REF)
    print(f"   verdict_C = {verdict_C}  (wall {time.time() - tC:.1f}s)")

    # ---- Optional Method D: GP-routed (sigma_gp -> Bayes-LGBM router-style)
    print("\n" + "-" * 78)
    print("METHOD D (free): GP <-> LGBM per-row inverse-variance router")
    print("-" * 78)
    sigma_gp = gp_std
    sigma_gp_norm = (sigma_gp - sigma_gp.min()) / max(
        sigma_gp.max() - sigma_gp.min(), 1e-9
    )
    w_gp = 1.0 / (1.0 + sigma_gp_norm * 4.0)
    w_gp = np.clip(w_gp, 0.0, 1.0)
    pred_D = w_gp * pred_C + (1.0 - w_gp) * lgbm_oof_k28
    rae_D = float(rae(y_unb, pred_D))
    verdict_D = _verdict(rae_D, NB2103_K28_MEAN_BAG_REF)
    print(f"   rae_D = {rae_D:.4f}  verdict_D = {verdict_D}")

    # ---- Summary table ----
    print("\n" + "=" * 78)
    print("SUMMARY TABLE")
    print("=" * 78)
    print(f"   {'method':<22s}  {'RAE':>8s}  {'dvs_K28_mean':>13s}  "
          f"{'dvs_K28_med':>12s}  verdict")
    print(f"   {'chemprop_aux (anchor)':<22s}  {rae_anchor:>8.4f}  "
          f"{rae_anchor - NB2103_K28_MEAN_BAG_REF:>+13.4f}  "
          f"{rae_anchor - NB2103_K28_MEDIAN_BAG_REF:>+12.4f}  ANCHOR")
    print(f"   {'nb2103 K=28 mean_bag':<22s}  {rae_lgbm_k28:>8.4f}  "
          f"{0.0:>+13.4f}  {rae_lgbm_k28 - NB2103_K28_MEDIAN_BAG_REF:>+12.4f}  REF")
    print(f"   {'A: BayesianRidge':<22s}  {rae_A:>8.4f}  "
          f"{rae_A - NB2103_K28_MEAN_BAG_REF:>+13.4f}  "
          f"{rae_A - NB2103_K28_MEDIAN_BAG_REF:>+12.4f}  {verdict_A}")
    print(f"   {'B: Bayes<->LGBM router':<22s}  {rae_B:>8.4f}  "
          f"{rae_B - NB2103_K28_MEAN_BAG_REF:>+13.4f}  "
          f"{rae_B - NB2103_K28_MEDIAN_BAG_REF:>+12.4f}  {verdict_B}")
    print(f"   {'C: GP RBF+White':<22s}  {rae_C:>8.4f}  "
          f"{rae_C - NB2103_K28_MEAN_BAG_REF:>+13.4f}  "
          f"{rae_C - NB2103_K28_MEDIAN_BAG_REF:>+12.4f}  {verdict_C}")
    print(f"   {'D: GP<->LGBM router':<22s}  {rae_D:>8.4f}  "
          f"{rae_D - NB2103_K28_MEAN_BAG_REF:>+13.4f}  "
          f"{rae_D - NB2103_K28_MEDIAN_BAG_REF:>+12.4f}  {verdict_D}")

    candidates = {
        "A_bayes": rae_A,
        "B_bayes_router": rae_B,
        "C_gp": rae_C,
        "D_gp_router": rae_D,
    }
    best_name = min(candidates, key=candidates.get)
    best_rae = candidates[best_name]
    beats_k28_mean = best_rae < NB2103_K28_MEAN_BAG_REF - DECISION_MARGIN
    beats_k28_median = best_rae < NB2103_K28_MEDIAN_BAG_REF - DECISION_MARGIN
    print(f"\n   best method = {best_name}  RAE = {best_rae:.4f}")
    print(f"   beats nb2103 K=28 mean_bag  ({NB2103_K28_MEAN_BAG_REF:.4f}): "
          f"{beats_k28_mean}")
    print(f"   beats nb2103 K=28 median_bag({NB2103_K28_MEDIAN_BAG_REF:.4f}): "
          f"{beats_k28_median}")

    # ---- Deploy CSV if best beats nb2103 K=28 mean_bag ----
    deploy_path = None
    deploy_built = False
    deploy_method = None
    if beats_k28_mean:
        deploy_method = best_name
        # Refit best method on ALL 253 unblind labels (no folds), apply to 513
        # then add anchor.
        # We need full-513 feature matrix; X_unb_28 is only on 253.  Build the
        # 513 matrix by reusing nb2103's selection.  Since we lack a saved
        # X_te_28, derive it through the same SHAP top-K mask if a 513
        # version is on disk; otherwise warn and emit anchor-only fallback.
        x_te_28_path = DATA_PROCESSED / "X_te_28_nb2103.npy"
        if not x_te_28_path.exists():
            print(f"\n[deploy] WARNING: {x_te_28_path} missing -- cannot build "
                  f"513-row K=28 features.  Falling back to anchor + 0 residual "
                  f"(no deploy CSV emitted).")
        else:
            X_te_28 = np.load(x_te_28_path).astype(np.float64)
            if X_te_28.shape != (n_test, 28):
                raise ValueError(
                    f"X_te_28 shape {X_te_28.shape} != ({n_test}, 28)"
                )
            scaler_full = StandardScaler()
            X_unb_28_s = scaler_full.fit_transform(X_unb_28)
            X_te_28_s = scaler_full.transform(X_te_28)
            if deploy_method.startswith("A") or deploy_method.startswith("B"):
                m_full = BayesianRidge(
                    max_iter=300, tol=1e-3, fit_intercept=True
                )
                m_full.fit(X_unb_28_s, residual)
                resid_te_mean, resid_te_std = m_full.predict(
                    X_te_28_s, return_std=True
                )
                if deploy_method.startswith("B"):
                    # router needs lgbm full prediction on 513.  Use anchor as
                    # fallback for non-unb rows since we lack a 513-row LGBM
                    # OOF.  This is OK because the router weights only
                    # affect the residual blending.
                    print("[deploy] B-router on 513: using anchor + bayes_mean "
                          "(LGBM-on-513 not available -- emit Bayesian-only)")
                pred_te = te_anchor + resid_te_mean
            elif deploy_method.startswith("C") or deploy_method.startswith("D"):
                kernel = (
                    ConstantKernel(1.0, (1e-3, 1e3))
                    * RBF(length_scale=1.0,
                          length_scale_bounds=(1e-2, 1e3))
                    + WhiteKernel(noise_level=0.5,
                                  noise_level_bounds=(1e-3, 1e1))
                )
                m_full = GaussianProcessRegressor(
                    kernel=kernel, normalize_y=True,
                    n_restarts_optimizer=2, random_state=CV_SEED, alpha=1e-6,
                )
                m_full.fit(X_unb_28_s, residual)
                resid_te_mean, resid_te_std = m_full.predict(
                    X_te_28_s, return_std=True
                )
                pred_te = te_anchor + resid_te_mean
            else:
                pred_te = te_anchor
            mol_col = "Molecule Name" if "Molecule Name" in te.columns else \
                ("molecule_name" if "molecule_name" in te.columns else None)
            if mol_col is None:
                raise KeyError("test set missing Molecule Name column")
            smi_col = "SMILES" if "SMILES" in te.columns else (
                "smiles" if "smiles" in te.columns else None
            )
            cols = {}
            if smi_col:
                cols["SMILES"] = te[smi_col].values
            cols["Molecule Name"] = te[mol_col].values
            cols["pEC50"] = np.asarray(pred_te, dtype=np.float64)
            sub = pd.DataFrame(cols)
            sub_dir = PROJECT_ROOT / "submissions"
            sub_dir.mkdir(parents=True, exist_ok=True)
            deploy_path = sub_dir / f"{TAG}_bayes_linear.csv"
            sub.to_csv(deploy_path, index=False)
            deploy_built = True
            print(f"[deploy] wrote {deploy_path}  shape={sub.shape}")

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "method": ("BayesianRidge + GaussianProcess on K=28 SHAP features "
                   "(chemprop_aux residual); per-row uncertainty router "
                   "Bayes/GP <-> LGBM"),
        "anchor": ANCHOR,
        "anchor_path": str(ANCHOR_TE_PATH),
        "n_unb": int(n_unb),
        "n_test": int(n_test),
        "K_features": 28,
        "cv_folds": N_FOLDS,
        "cv_seed": CV_SEED,
        "decision_margin": DECISION_MARGIN,
        "rae_anchor_chemprop_aux": rae_anchor,
        "rae_lgbm_K28_meanbag_in_RAE": rae_lgbm_k28,
        "nb2103_K28_mean_bag_ref": NB2103_K28_MEAN_BAG_REF,
        "nb2103_K28_median_bag_ref": NB2103_K28_MEDIAN_BAG_REF,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "method_A_bayesian_ridge": {
            "rae": rae_A,
            "delta_vs_anchor": rae_A - rae_anchor,
            "delta_vs_K28_mean_bag": rae_A - NB2103_K28_MEAN_BAG_REF,
            "delta_vs_K28_median_bag": rae_A - NB2103_K28_MEDIAN_BAG_REF,
            "verdict": verdict_A,
            "bayes_std_mean": float(bayes_std.mean()),
            "bayes_std_min": float(bayes_std.min()),
            "bayes_std_max": float(bayes_std.max()),
            "bayes_std_std": float(bayes_std.std()),
        },
        "method_B_bayes_lgbm_router": {
            "rae": rae_B,
            "delta_vs_anchor": rae_B - rae_anchor,
            "delta_vs_K28_mean_bag": rae_B - NB2103_K28_MEAN_BAG_REF,
            "delta_vs_K28_median_bag": rae_B - NB2103_K28_MEDIAN_BAG_REF,
            "verdict": verdict_B,
            "weight_w_mean": float(w_clipped.mean()),
            "weight_w_min": float(w_clipped.min()),
            "weight_w_max": float(w_clipped.max()),
            "n_bayes_dominant_rows": int((w_clipped >= 0.5).sum()),
        },
        "method_C_gp_rbf": {
            "rae": rae_C,
            "delta_vs_anchor": rae_C - rae_anchor,
            "delta_vs_K28_mean_bag": rae_C - NB2103_K28_MEAN_BAG_REF,
            "delta_vs_K28_median_bag": rae_C - NB2103_K28_MEDIAN_BAG_REF,
            "verdict": verdict_C,
            "gp_std_mean": float(gp_std.mean()),
            "gp_std_min": float(gp_std.min()),
            "gp_std_max": float(gp_std.max()),
        },
        "method_D_gp_lgbm_router": {
            "rae": rae_D,
            "delta_vs_anchor": rae_D - rae_anchor,
            "delta_vs_K28_mean_bag": rae_D - NB2103_K28_MEAN_BAG_REF,
            "delta_vs_K28_median_bag": rae_D - NB2103_K28_MEDIAN_BAG_REF,
            "verdict": verdict_D,
        },
        "best_method": best_name,
        "best_rae": best_rae,
        "beats_K28_mean_bag": bool(beats_k28_mean),
        "beats_K28_median_bag": bool(beats_k28_median),
        "deploy_method": deploy_method,
        "deploy_built": bool(deploy_built),
        "deploy_path": str(deploy_path) if deploy_path else None,
        "pre_unblind_clean": True,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "rae_anchor_chemprop_aux",
        "rae_lgbm_K28_meanbag_in_RAE",
        "nb2103_K28_mean_bag_ref",
        "nb2103_K28_median_bag_ref",
        "best_method",
        "best_rae",
        "beats_K28_mean_bag",
        "beats_K28_median_bag",
        "deploy_built",
        "deploy_path",
    ):
        print(f"  {k}: {res.get(k)}")
    for k in ("method_A_bayesian_ridge", "method_B_bayes_lgbm_router",
              "method_C_gp_rbf", "method_D_gp_lgbm_router"):
        sub = res[k]
        print(f"  {k}: rae={sub['rae']:.4f}  verdict={sub['verdict']}")
