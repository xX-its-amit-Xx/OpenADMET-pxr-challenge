"""nb2534 -- Isotonic + Polynomial Calibration on K=20 residual (nb2240 anchor).

NEW PARADIGM:
    2-stage post-hoc calibration of the nb2240 mean-bag-K20 anchor:
      Stage 1: IsotonicRegression(y_min=3.0, y_max=8.0) fit on
               (anchor, y) using fold-train rows.
      Stage 2: PolynomialFeatures(degree=2, include_bias=False) applied
               to the Stage-1 isotonic output, then Ridge(alpha=0.1)
               fit on the residual (y - iso(anchor)) using fold-train.
      Final  : iso(anchor_val) + poly_ridge_residual(iso(anchor_val))
    Both stages are fit ONLY on the fold-train rows; fold-val rows
    receive a fully out-of-fold prediction.

PROTOCOL:
    - Anchor: nb2240_mean_bag_oof_K20.npy  (on the 253 unblind)
              + te_nb2240.npy             (on the 513 test)
    - 5-fold scaffold CV on 253 unblind, kf_seeds {1001..1005}
    - Deploy: re-fit iso + poly_ridge on ALL 253 -> apply to 513
    - Gate: mean_rae < 0.4570 -> PROMOTE
            < 0.4601 -> MARGINAL_BEAT
            else FAIL.

Outputs:
    scripts/nb2534_iso_pc_residual.py
    data/processed/nb2534_summary.json
    data/processed/nb2534_pred_oof.npy   (253,) float32
    data/processed/te_nb2534.npy         (513,) float32
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
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2534"

# ------------------------- calibration hyperparameters ----------------------
ISO_YMIN = 3.0
ISO_YMAX = 8.0
POLY_DEGREE = 2
POLY_INCLUDE_BIAS = False
RIDGE_ALPHA = 0.1

N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# Gates
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4601


def fit_iso_poly(
    anchor_tr: np.ndarray,
    y_tr: np.ndarray,
):
    """Fit Stage-1 isotonic + Stage-2 polynomial ridge on residual.

    Returns (iso, poly, ridge) tuple of fitted transformers.
    """
    iso = IsotonicRegression(
        y_min=ISO_YMIN,
        y_max=ISO_YMAX,
        increasing=True,
        out_of_bounds="clip",
    )
    iso.fit(anchor_tr, y_tr)
    iso_tr = iso.predict(anchor_tr)

    poly = PolynomialFeatures(
        degree=POLY_DEGREE, include_bias=POLY_INCLUDE_BIAS
    )
    X_poly_tr = poly.fit_transform(iso_tr.reshape(-1, 1))

    resid_tr = y_tr - iso_tr
    ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    ridge.fit(X_poly_tr, resid_tr)
    return iso, poly, ridge


def predict_iso_poly(
    iso: IsotonicRegression,
    poly: PolynomialFeatures,
    ridge: Ridge,
    anchor_va: np.ndarray,
) -> np.ndarray:
    """Apply 2-stage calibration to held-out anchor values."""
    iso_va = iso.predict(anchor_va)
    X_poly_va = poly.transform(iso_va.reshape(-1, 1))
    resid_pred_va = ridge.predict(X_poly_va)
    final = iso_va + resid_pred_va
    return final


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Isotonic + Polynomial Calibration on nb2240 K=20 anchor")
    print("=" * 78)

    # ---- Load anchor on unblind (253) ----
    anchor_oof_path = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"
    if not anchor_oof_path.exists():
        raise FileNotFoundError(f"missing anchor: {anchor_oof_path}")
    anchor_unb = np.load(anchor_oof_path).astype(np.float64)

    # ---- Load deploy anchor on full test (513) ----
    anchor_te_path = DATA_PROCESSED / "te_nb2240.npy"
    if not anchor_te_path.exists():
        raise FileNotFoundError(f"missing test anchor: {anchor_te_path}")
    anchor_te = np.load(anchor_te_path).astype(np.float64)

    # ---- Load truth + unblind index ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    n_test = anchor_te.shape[0]
    print(f"[load] n_unb={n_unb}  n_test={n_test}")
    print(f"[load] anchor_unb shape={anchor_unb.shape}  "
          f"mean={anchor_unb.mean():.3f}  std={anchor_unb.std():.3f}")
    print(f"[load] anchor_te  shape={anchor_te.shape}   "
          f"mean={anchor_te.mean():.3f}  std={anchor_te.std():.3f}")

    if anchor_unb.shape[0] != n_unb:
        raise ValueError(
            f"anchor_unb shape {anchor_unb.shape} != n_unb={n_unb}"
        )

    rae_anchor_unb = float(rae(y_unb, anchor_unb))
    print(f"[load] nb2240 OOF unblind RAE = {rae_anchor_unb:.4f}")

    anchor_te_unb = anchor_te[unb_idx]
    rae_anchor_te_unb = float(rae(y_unb, anchor_te_unb))
    print(f"[load] te_nb2240[unb_idx] in_RAE = {rae_anchor_te_unb:.4f}")

    # ---- Test-set SMILES for scaffold-CV on the 253 ----
    te = load_test()
    te_smi_col = "smiles" if "smiles" in te.columns else "SMILES"
    te_smi = te[te_smi_col].astype(str).tolist()
    unb_smiles = [te_smi[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaf] n_unique={n_unique_scaf}")

    # ---- 5-fold scaffold CV on 253, 5 seeds ----
    print("\n" + "-" * 78)
    print(
        f"SCAFFOLD 5-FOLD CV  kf_seeds={KF_SEEDS}  "
        f"iso=[{ISO_YMIN},{ISO_YMAX}]  poly_deg={POLY_DEGREE}  "
        f"ridge_alpha={RIDGE_ALPHA}"
    )
    print("-" * 78)

    per_seed = []
    all_oofs = []
    for kf_seed in KF_SEEDS:
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        oof = np.full(n_unb, np.nan, dtype=np.float64)
        fold_info = []
        for fi, (tr_loc, va_loc) in enumerate(splits):
            iso, poly, ridge = fit_iso_poly(
                anchor_unb[tr_loc], y_unb[tr_loc]
            )
            oof[va_loc] = predict_iso_poly(
                iso, poly, ridge, anchor_unb[va_loc]
            )
            fold_info.append({
                "fold": fi,
                "n_tr": int(len(tr_loc)),
                "n_va": int(len(va_loc)),
                "ridge_intercept": float(ridge.intercept_),
                "ridge_coef": [float(c) for c in ridge.coef_],
            })
        # clip to a sane range
        oof = np.clip(oof, ISO_YMIN, ISO_YMAX)
        pooled = float(rae(y_unb, oof))
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": pooled,
            "folds": fold_info,
        })
        all_oofs.append(oof)
        print(f"   seed={kf_seed}  pooled_RAE={pooled:.4f}")

    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    final_oof_rae = float(rae(y_unb, mean_oof))
    pooled_rae_mean_seeds = float(np.mean([r["pooled_rae"] for r in per_seed]))
    pooled_rae_std_seeds = float(np.std([r["pooled_rae"] for r in per_seed]))
    print(
        f"\n[cv] pooled_RAE mean = {pooled_rae_mean_seeds:.4f} "
        f"(+/- {pooled_rae_std_seeds:.4f})"
    )
    print(f"[cv] RAE of mean-of-seed OOFs = {final_oof_rae:.4f}")

    # ---- Deploy: fit on ALL 253 -> apply to 513 ----
    iso_full, poly_full, ridge_full = fit_iso_poly(anchor_unb, y_unb)
    deploy_te = predict_iso_poly(iso_full, poly_full, ridge_full, anchor_te)
    deploy_te = np.clip(deploy_te, ISO_YMIN, ISO_YMAX).astype(np.float32)
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx].astype(np.float64)))
    print(f"\n[deploy] te(513) mean={deploy_te.mean():.3f}  "
          f"std={deploy_te.std():.3f}")
    print(f"[deploy] te[unb_idx] in_RAE={te_unb_rae:.4f}  "
          f"(in-sample, deploy refit on all 253)")
    print(f"[deploy] ridge_intercept={float(ridge_full.intercept_):.4f}  "
          f"ridge_coef={[round(float(c),4) for c in ridge_full.coef_]}")

    # ---- Gate ----
    if pooled_rae_mean_seeds < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif pooled_rae_mean_seeds < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    print(f"   pooled_RAE_mean    = {pooled_rae_mean_seeds:.4f}")
    print(f"   GATE_PROMOTE       < {GATE_PROMOTE:.4f}")
    print(f"   GATE_MARGINAL_BEAT < {GATE_MARGINAL:.4f}")
    print(f"   verdict            = {verdict}")
    print(f"   delta vs nb2240 OOF= {pooled_rae_mean_seeds - rae_anchor_unb:+.4f}")

    # ---- Save artefacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, mean_oof.astype(np.float32))
    np.save(te_path, deploy_te)
    print(f"\n[save] {oof_path}")
    print(f"[save] {te_path}")

    summary = {
        "tag": TAG,
        "method": "isotonic_plus_polynomial_ridge_residual_on_nb2240_K20",
        "anchor": "nb2240_mean_bag_oof_K20",
        "anchor_te": "te_nb2240",
        "iso_y_min": ISO_YMIN,
        "iso_y_max": ISO_YMAX,
        "poly_degree": POLY_DEGREE,
        "poly_include_bias": POLY_INCLUDE_BIAS,
        "ridge_alpha": RIDGE_ALPHA,
        "n_unb": n_unb,
        "n_te": n_test,
        "n_unique_scaffolds": n_unique_scaf,
        "rae_anchor_unb_oof": rae_anchor_unb,
        "rae_anchor_te_unb_insample": rae_anchor_te_unb,
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "per_seed": per_seed,
        "pooled_rae_mean_seeds": pooled_rae_mean_seeds,
        "pooled_rae_std_seeds": pooled_rae_std_seeds,
        "rae_of_mean_of_seed_oofs": final_oof_rae,
        "deploy_ridge_intercept": float(ridge_full.intercept_),
        "deploy_ridge_coef": [float(c) for c in ridge_full.coef_],
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "te_unb_in_rae": te_unb_rae,
        "delta_vs_anchor_oof": pooled_rae_mean_seeds - rae_anchor_unb,
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   pooled_RAE_mean        = {pooled_rae_mean_seeds:.4f}")
    print(f"   delta vs nb2240 OOF    = {pooled_rae_mean_seeds - rae_anchor_unb:+.4f}")
    print(f"   verdict                = {verdict}")
    print(f"   wall                   = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "pooled_rae_mean_seeds",
        "pooled_rae_std_seeds",
        "rae_anchor_unb_oof",
        "delta_vs_anchor_oof",
        "verdict",
        "te_unb_in_rae",
        "deploy_ridge_intercept",
        "deploy_ridge_coef",
    ):
        print(f"  {k}: {res.get(k)}")
