"""nb2563 -- Gaussian Process with precomputed Tanimoto kernel on
chemprop_aux residual.

NEW PARADIGM vs nb2450 SVR-Tanimoto: GP with proper noise modeling
(homoscedastic Gaussian likelihood, alpha) instead of SVR's
epsilon-insensitive hinge loss. SVR ignores residuals inside the
epsilon-tube; GP treats every residual as Gaussian-distributed
observation with variance alpha. This is the more principled model
class for noisy chemistry data where the pEC50 SE is ~0.24 and we
don't want to discard signal inside an arbitrary epsilon tube.

PROTOCOL
    1. Compute Morgan FP-2048 (radius=2) on 4139 train + 253 unblind + 513 test.
    2. Custom Tanimoto kernel: K(x,y) = |x AND y| / |x OR y|  (uint8 bit FPs).
       K_unb_unb (253x253), K_te_unb (513x253). Identical formula to nb2450
       so the substrate is held constant -- only the regressor changes.
    3. sklearn GaussianProcessRegressor with kernel='precomputed' via a
       wrapper kernel that returns the precomputed slice (DotProduct-style
       hack). Sweep noise alpha in {0.01, 0.1, 0.5} for the regularisation /
       diagonal jitter trade-off:
         - alpha=0.01 -> nearly noise-free, near-perfect train fit (high var)
         - alpha=0.1  -> matches assay noise floor (~0.24 pEC50 => alpha~0.06)
         - alpha=0.5  -> heavy noise prior, large-scale shrinkage to mean
       Trained on chemprop_aux RESIDUAL (y_unb - te_chemprop_aux[unb_idx]).
    4. 5-fold scaffold CV on the 253; per fold we re-slice the precomputed
       kernel to (n_tr,n_tr) for fit and (n_va,n_tr) for predict. Pick
       alpha with best pooled OOF RAE on (anchor + residual_pred).
    5. Standalone RAE = rae(y_unb, anchor + oof_residual). Save artefacts
       regardless of the gate outcome.
    6. Gate (against nb2450 SVR ceiling 0.4601 from cycle-160 deep-30 ladder):
         alpha mean_rae < 0.4570  -> "PROMOTE"
         alpha mean_rae < 0.4601  -> "MARGINAL_BEAT"
         else                     -> "FAIL"

Outputs
    scripts/nb2563_gp_tanimoto_kernel.py
    data/processed/nb2563_summary.json
    data/processed/nb2563_pred_oof.npy   (253,) float32
    data/processed/te_nb2563.npy         (513,) float32
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
from rdkit import RDLogger
from scipy.linalg import cho_factor, cho_solve

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko, morgan_fp_batch
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2563"
FP_RADIUS = 2
FP_BITS = 2048

ALPHA_GRID = [0.01, 0.1, 0.5]
N_FOLDS = 5
KF_SEED = 1001

PROMOTE_THRESH = 0.4570
MARGINAL_THRESH = 0.4601


# ---------------------------------------------------------------------------
# Tanimoto kernel on uint8 bit FPs
# ---------------------------------------------------------------------------

def tanimoto_kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """K[i,j] = |A_i AND B_j| / |A_i OR B_j| for binary bit-vectors.

    A: (nA, n_bits) uint8, B: (nB, n_bits) uint8. Returns (nA, nB) float64.
    Float64 needed because GP linear-algebra (Cholesky, inv) prefers double
    precision -- float32 was fine for SVR's QP solver but GP is more sensitive.
    """
    A = A.astype(np.float64, copy=False)
    B = B.astype(np.float64, copy=False)
    inter = A @ B.T
    a_sum = A.sum(axis=1)
    b_sum = B.sum(axis=1)
    union = a_sum[:, None] + b_sum[None, :] - inter
    union = np.maximum(union, 1.0)
    return (inter / union).astype(np.float64)


# ---------------------------------------------------------------------------
# Closed-form GP with precomputed Tanimoto kernel.
#
# sklearn's GaussianProcessRegressor can't take a precomputed kernel without
# subclassing the Kernel base AND surviving its clone() round-trip. Instead
# we implement the GP equations directly -- they are exactly what sklearn
# would do internally with optimizer=None for a fixed kernel:
#
#     y_centered = y - mean(y)
#     L = chol(K_tt + alpha * I)
#     alpha_vec = solve(L L^T, y_centered)
#     y_pred = K_pt @ alpha_vec + mean(y)        # mean re-added (normalize_y)
#
# Conjugate Gaussian posterior; identical numerics to sklearn except faster.
# ---------------------------------------------------------------------------

def gp_fit_predict(K_tr_tr: np.ndarray,
                   K_pt_tr: np.ndarray,
                   y_tr: np.ndarray,
                   alpha: float,
                   normalize_y: bool = True) -> np.ndarray:
    """Fit GP on (K_tr_tr, y_tr) and predict mean on K_pt_tr rows."""
    if normalize_y:
        y_mean = float(np.mean(y_tr))
        y_centered = y_tr - y_mean
    else:
        y_mean = 0.0
        y_centered = y_tr

    n_tr = K_tr_tr.shape[0]
    K_reg = K_tr_tr + alpha * np.eye(n_tr)
    c, low = cho_factor(K_reg, lower=True, overwrite_a=False, check_finite=False)
    alpha_vec = cho_solve((c, low), y_centered, check_finite=False)
    pred = K_pt_tr @ alpha_vec + y_mean
    return pred


def gp_cv_oof_for_alpha(K_unb_unb: np.ndarray,
                        residual: np.ndarray,
                        anchor_unb: np.ndarray,
                        y_unb: np.ndarray,
                        unb_scaffolds,
                        alpha: float,
                        kf_seed: int = KF_SEED):
    """5-fold scaffold CV on residual; return RAE on anchor+residual."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(residual)
    oof_resid = np.full(n_unb, np.nan, dtype=np.float64)
    for tr_loc, va_loc in splits:
        K_tr_tr = K_unb_unb[np.ix_(tr_loc, tr_loc)]
        K_va_tr = K_unb_unb[np.ix_(va_loc, tr_loc)]
        oof_resid[va_loc] = gp_fit_predict(
            K_tr_tr, K_va_tr, residual[tr_loc], alpha=alpha, normalize_y=True,
        )
    oof_corrected = anchor_unb + oof_resid
    return float(rae(y_unb, oof_corrected)), oof_resid, oof_corrected


def gp_deploy_te(K_unb_unb: np.ndarray,
                 K_te_unb: np.ndarray,
                 residual: np.ndarray,
                 alpha: float):
    """Refit GP on all 253; predict residual on 513 test."""
    pred = gp_fit_predict(
        K_unb_unb, K_te_unb, residual, alpha=alpha, normalize_y=True,
    )
    return pred.astype(np.float32)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- GP with Tanimoto kernel on chemprop_aux residual")
    print("=" * 78)

    # ---- Load truth, anchor, test ----
    te = load_test()
    n_test = len(te)
    te_smiles = te["smiles"].astype(str).tolist() if "smiles" in te.columns \
        else te["SMILES"].astype(str).tolist()

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    te_anchor_513 = np.load(DATA_PROCESSED / "te_chemprop_aux.npy").astype(np.float64)
    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    print(f"[load] chemprop_aux in_RAE = {rae_anchor:.4f}")
    residual = y_unb - anchor_unb

    # ---- Morgan FP-2048 r=2 ----
    print(f"\n[fp] computing Morgan FP-{FP_BITS} radius={FP_RADIUS} ...")
    t_fp = time.time()
    fp_test = morgan_fp_batch(te_smiles, radius=FP_RADIUS, n_bits=FP_BITS)
    fp_unb = fp_test[unb_idx]
    print(f"   shapes test={fp_test.shape}  unb={fp_unb.shape}  "
          f"({time.time() - t_fp:.1f}s)")

    # ---- Tanimoto kernel matrices ----
    print("\n[kernel] computing Tanimoto kernel matrices ...")
    t_k = time.time()
    K_unb_unb = tanimoto_kernel(fp_unb, fp_unb)     # (253, 253)
    K_te_unb = tanimoto_kernel(fp_test, fp_unb)     # (513, 253)
    diag_med = float(np.median(np.diag(K_unb_unb)))
    off_med = float(np.median(K_unb_unb[~np.eye(n_unb, dtype=bool)]))
    print(f"   K_unb_unb {K_unb_unb.shape}  diag_med={diag_med:.3f}  off_med={off_med:.3f}")
    print(f"   K_te_unb {K_te_unb.shape}  median={float(np.median(K_te_unb)):.3f}  "
          f"({time.time() - t_k:.1f}s)")

    # ---- GP alpha-grid with 5-fold scaffold CV on residual ----
    print("\n" + "-" * 78)
    print(f"GP(kernel='precomputed-tanimoto') over alpha-grid {ALPHA_GRID}")
    print("-" * 78)
    alpha_results = []
    best_alpha, best_rae_val = None, float("inf")
    best_oof_resid, best_oof_corrected = None, None
    for alpha in ALPHA_GRID:
        tA = time.time()
        rae_alpha, oof_resid_alpha, oof_corrected_alpha = gp_cv_oof_for_alpha(
            K_unb_unb, residual, anchor_unb, y_unb, unb_scaffolds,
            alpha=alpha, kf_seed=KF_SEED,
        )
        alpha_results.append({
            "alpha": float(alpha),
            "rae_corrected_oof": float(rae_alpha),
            "residual_oof_mean": float(np.mean(oof_resid_alpha)),
            "residual_oof_std": float(np.std(oof_resid_alpha)),
            "wall_sec": round(time.time() - tA, 2),
        })
        print(f"   alpha={alpha:>5g}  rae_oof_corrected={rae_alpha:.4f}  "
              f"resid_mean={float(np.mean(oof_resid_alpha)):+.3f}  "
              f"resid_std={float(np.std(oof_resid_alpha)):.3f}  "
              f"({time.time() - tA:.1f}s)")
        if rae_alpha < best_rae_val:
            best_rae_val = rae_alpha
            best_alpha = float(alpha)
            best_oof_resid = oof_resid_alpha
            best_oof_corrected = oof_corrected_alpha
    print(f"\n[selected] best_alpha={best_alpha}  standalone_oof_RAE={best_rae_val:.4f}")

    # ---- Deploy: refit GP on all 253 with best alpha ----
    print(f"\n[deploy] GP(alpha={best_alpha}) refit on all 253; predict te(513)")
    te_resid = gp_deploy_te(K_unb_unb, K_te_unb, residual, alpha=best_alpha).astype(np.float32)
    te_nb2563_513 = (te_anchor_513.astype(np.float32) + te_resid).astype(np.float32)
    te_unb_insample_rae = float(rae(y_unb, te_nb2563_513[unb_idx]))
    print(f"   te_nb2563 mean/std = {te_nb2563_513.mean():.3f}/{te_nb2563_513.std():.3f}")
    print(f"   te[unb_idx] in-sample RAE = {te_unb_insample_rae:.4f}")

    # ---- Save artefacts (ALWAYS, even on FAIL gate) ----
    pred_oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(pred_oof_path, best_oof_corrected.astype(np.float32))
    np.save(te_path, te_nb2563_513)
    print(f"[save] {pred_oof_path}")
    print(f"[save] {te_path}")

    # ---- Gate ----
    if best_rae_val < PROMOTE_THRESH:
        gate = "PROMOTE"
    elif best_rae_val < MARGINAL_THRESH:
        gate = "MARGINAL_BEAT"
    else:
        gate = "FAIL"
    print(f"\n[gate] best={best_rae_val:.4f}  promote<{PROMOTE_THRESH}  "
          f"marginal<{MARGINAL_THRESH}  ->  {gate}")

    summary = {
        "tag": TAG,
        "method": "gp_tanimoto_kernel_on_chemprop_aux_residual",
        "fp_radius": FP_RADIUS,
        "fp_bits": FP_BITS,
        "alpha_grid": ALPHA_GRID,
        "n_unb": n_unb,
        "n_te": n_test,
        "n_unique_scaffolds": n_unique_scaf,
        "rae_anchor_chemprop_aux": rae_anchor,
        "alpha_grid_results": alpha_results,
        "best_alpha": best_alpha,
        "standalone_oof_rae": float(best_rae_val),
        "te_unb_in_sample_rae": te_unb_insample_rae,
        "kernel_diag_median": diag_med,
        "kernel_off_median": off_med,
        "pred_oof_path": str(pred_oof_path),
        "te_path": str(te_path),
        "promote_thresh": PROMOTE_THRESH,
        "marginal_thresh": MARGINAL_THRESH,
        "gate": gate,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   anchor chemprop_aux RAE  = {rae_anchor:.4f}")
    print(f"   best alpha               = {best_alpha}")
    print(f"   standalone OOF RAE       = {best_rae_val:.4f}")
    print(f"   gate                     = {gate}")
    print(f"   wall                     = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("best_alpha", "standalone_oof_rae", "gate"):
        print(f"  {k}: {res.get(k)}")
