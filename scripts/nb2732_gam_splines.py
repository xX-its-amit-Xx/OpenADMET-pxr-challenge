"""nb2732 -- Generalized Additive Model (GAM) with smoothing splines on K=20.

NEW PARADIGM (vs LGBM tree splits, vs linear regression, vs RFF kernels):
    Every model in the cycle-149+ post-hoc-blend ceiling (LGBM, sklearn GBR,
    XGB, RF, CatBoost) is an axis-aligned tree splitter -- the only thing
    that ever varied between them was the loss / split heuristic / tree
    builder.  Linear regression / Ridge / Lasso are the opposite extreme:
    one global linear coefficient per feature, no per-feature flexibility.

    A Generalized Additive Model (GAM) sits between these two extremes:
        y_hat = beta0 + f_1(x_1) + f_2(x_2) + ... + f_K(x_K)
    where each f_j is a smooth penalized cubic spline (smoothing parameter
    lam controls roughness penalty on second derivative).  This is the
    classic Hastie & Tibshirani additive structure: per-feature non-linear
    transforms, but NO interaction terms (unlike trees which split on
    multiple features in a single path).

    Hypothesis: on K=20 chemprop_aux residual at n=253, if the residual
    structure is dominated by per-feature non-linear shape (saturation,
    monotone-non-linear, threshold) rather than by feature-feature
    interactions, then a GAM should capture the per-feature shape with
    fewer effective degrees of freedom than a tree ensemble that has to
    discover the same shape via many short-depth interaction splits.  The
    penalized-spline shrinkage is also a different regularizer axis than
    tree leaf-count / min-child-weight.

    PyGAM uses backfitting + penalized iteratively re-weighted least
    squares (PIRLS) to fit, with a smoothing parameter lam per term.
    spline_order=3 (cubic), n_splines=20 (basis size per feature), lam=0.6
    (mild smoothing).  Total dof bounded by K * n_splines = 400 raw basis,
    but the penalty shrinks effective dof well below that on n=253.

PROTOCOL:
    1. Load X_117 substrate -> slice K=20 surviving columns from nb2240.
    2. residual = y_unb - chemprop_aux_te[unb_idx]  (PRE-clean anchor only).
    3. PyGAM LinearGAM(s(0) + s(1) + ... + s(19), n_splines=20,
       spline_order=3, lam=0.6).  Fit per fold on K=20 features against
       the chemprop_aux residual target.
    4. 5-fold scaffold CV (`scaffold_kfold_indices`), 5 kf_seeds
       {1001..1005}.
    5. Deploy: refit GAM on full 253 per seed -> predict 513 residual;
       mean-bag aggregate.

GATE (mean-bag corrected RAE):
    mean_rae < 0.4570 -> "PROMOTE"
    mean_rae < 0.4598 -> "MARGINAL_BEAT"
    else              -> "FAIL"

OUTPUTS:
    scripts/nb2732_gam_splines.py
    data/processed/nb2732_summary.json
    data/processed/nb2732_pred_oof.npy   (253,) float32 mean-bag CORRECTED
    data/processed/te_nb2732.npy         (513,) float32 deploy refit
    submissions/nb2732_gam_splines.csv
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

# ---- pygam import guard: spec says save "INSTALL_FAILED" and exit clean ----
try:
    from pygam import LinearGAM, s as _spline_term  # type: ignore
    _PYGAM_OK = True
    _PYGAM_ERR: str | None = None
except Exception as _e:  # pragma: no cover -- install guard branch
    _PYGAM_OK = False
    _PYGAM_ERR = f"{type(_e).__name__}: {_e}"
    LinearGAM = None  # type: ignore
    _spline_term = None  # type: ignore

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2732"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"
NB2240_SUMMARY = DATA_PROCESSED / "nb2240_summary.json"

N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# PyGAM LinearGAM hyperparams per task spec
GAM_N_SPLINES = 20
GAM_SPLINE_ORDER = 3
GAM_LAM = 0.6

# Gate thresholds (mean-bag RAE)
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

CHEMPROP_AUX_REF = 0.6216
NB2240_K20_REF = 0.4630  # LGBM K=20 baseline (L2 loss, tree splits)


def _build_gam(feat_dim: int) -> "LinearGAM":
    """Build LinearGAM with one smooth term per K=20 feature.

    terms = s(0) + s(1) + ... + s(K-1)  with n_splines=20, spline_order=3,
    lam=0.6 applied per smooth term.
    """
    assert _PYGAM_OK, "pygam not importable (guarded above)"
    # Build the term-list using s(i, ...).  pygam's TermList supports + .
    terms = _spline_term(0, n_splines=GAM_N_SPLINES,
                         spline_order=GAM_SPLINE_ORDER, lam=GAM_LAM)
    for i in range(1, feat_dim):
        terms = terms + _spline_term(
            i, n_splines=GAM_N_SPLINES,
            spline_order=GAM_SPLINE_ORDER, lam=GAM_LAM,
        )
    return LinearGAM(terms)


def _scaffold_cv_one_seed(
    X: np.ndarray,
    residual: np.ndarray,
    unb_scaffolds: list,
    kf_seed: int,
) -> np.ndarray:
    """One scaffold-CV pass: per-fold PyGAM LinearGAM on K=20 residual."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n = len(residual)
    feat_dim = X.shape[1]
    oof = np.full(n, np.nan, dtype=np.float64)
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        gam = _build_gam(feat_dim)
        gam.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = np.asarray(gam.predict(X[va_loc]), dtype=np.float64)
    if np.isnan(oof).any():
        raise RuntimeError("scaffold split did not cover all rows")
    return oof


def _deploy_te_one_seed(
    X_unb: np.ndarray,
    residual: np.ndarray,
    X_te: np.ndarray,
) -> np.ndarray:
    """Fit GAM on full 253; predict 513 residual."""
    feat_dim = X_unb.shape[1]
    gam = _build_gam(feat_dim)
    gam.fit(X_unb, residual)
    return np.asarray(gam.predict(X_te), dtype=np.float32)


def _write_install_failed_summary(reason: str) -> dict:
    """Save a minimal summary stating pygam install failed; exit clean."""
    summary = {
        "tag": TAG,
        "status": "INSTALL_FAILED",
        "reason": reason,
        "method": "pygam_LinearGAM_K20_residual_on_chemprop_aux",
        "anchor": ANCHOR,
        "anchor_pre_unblind": True,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "gam_n_splines": GAM_N_SPLINES,
        "gam_spline_order": GAM_SPLINE_ORDER,
        "gam_lam": GAM_LAM,
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "verdict": "INSTALL_FAILED",
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}  (INSTALL_FAILED)")
    return summary


def main() -> dict:
    import pandas as pd
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- PyGAM LinearGAM (smoothing splines) on K=20 substrate "
          f"(chemprop_aux residual)")
    print(f"        n_splines={GAM_N_SPLINES}  spline_order={GAM_SPLINE_ORDER}"
          f"  lam={GAM_LAM}")
    print(f"        anchor = {ANCHOR}  scaffold-CV {N_FOLDS}-fold  "
          f"kf_seeds={KF_SEEDS}")
    print(f"        ref nb2240 K=20 LGBM (L2/trees) = {NB2240_K20_REF:.4f}")
    print(f"        GATE: <{GATE_PROMOTE} PROMOTE / "
          f"<{GATE_MARGINAL} MARGINAL_BEAT / else FAIL")
    print("=" * 78)

    # ---- pygam install guard ----
    if not _PYGAM_OK:
        print(f"[ERROR] pygam import failed: {_PYGAM_ERR}")
        return _write_install_failed_summary(_PYGAM_ERR or "pygam not importable")

    # ---- Load truth + anchor + scaffolds ----
    te = load_test()
    n_test = len(te)
    test_smiles = (
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
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}  "
          f"abs_p90={np.quantile(np.abs(residual), 0.9):.4f}")

    # ---- Load X_117 substrate ----
    if not X117_UNB_PATH.exists() or not X117_TE_PATH.exists():
        raise FileNotFoundError(
            f"X_117 cache missing: {X117_UNB_PATH} or {X117_TE_PATH}"
        )
    X117_unb = np.load(X117_UNB_PATH).astype(np.float32)
    X117_te = np.load(X117_TE_PATH).astype(np.float32)
    if X117_unb.shape != (n_unb, 117):
        raise ValueError(f"X117_unb shape {X117_unb.shape} expected ({n_unb},117)")
    if X117_te.shape != (n_test, 117):
        raise ValueError(f"X117_te shape {X117_te.shape} expected ({n_test},117)")
    X117_unb = np.where(np.isfinite(X117_unb), X117_unb, 0.0).astype(np.float32)
    X117_te = np.where(np.isfinite(X117_te), X117_te, 0.0).astype(np.float32)
    print(f"[feat] X117_unb = {X117_unb.shape}  X117_te = {X117_te.shape}")

    # ---- Slice K=20 columns from nb2240 RFE ----
    if not NB2240_SUMMARY.exists():
        raise FileNotFoundError(f"nb2240 summary missing: {NB2240_SUMMARY}")
    with open(NB2240_SUMMARY) as f:
        nb2240 = json.load(f)
    k20_idx = list(nb2240["k20_surviving_idx_in_117"])
    k20_names = list(nb2240["k20_surviving_names"])
    assert len(k20_idx) == 20, f"expected 20 K=20 indices, got {len(k20_idx)}"
    print(f"[K20] loaded {len(k20_idx)} surviving indices from nb2240")

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
    print(f"  pygam.LinearGAM(s(0)+...+s({feat_dim - 1}))  "
          f"n_splines={GAM_N_SPLINES}  spline_order={GAM_SPLINE_ORDER}  "
          f"lam={GAM_LAM}")
    print("-" * 78)
    per_seed_oof_resid = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)
    per_seed_te_resid = np.zeros((len(KF_SEEDS), n_test), dtype=np.float64)
    per_seed_rae: list[float] = []

    for i, seed in enumerate(KF_SEEDS):
        ts = time.time()
        resid_oof = _scaffold_cv_one_seed(
            X_unb, residual, unb_scaffolds, seed,
        )
        per_seed_oof_resid[i] = resid_oof
        te_resid = _deploy_te_one_seed(X_unb, residual, X_te)
        per_seed_te_resid[i] = te_resid
        pred_corr = anchor + resid_oof
        rae_s = float(rae(y_unb, pred_corr))
        per_seed_rae.append(rae_s)
        print(f"   seed={seed}  rae_corr={rae_s:.4f}  "
              f"d_vs_anchor={rae_s - rae_anchor:+.4f}  "
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
    print(f"[cv] reference  nb2240 K=20 LGBM (L2/trees) = "
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

    sub_csv = SUBMISSIONS / f"{TAG}_gam_splines.csv"
    pd.DataFrame({
        "SMILES": test_smiles,
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
        "status": "RAN",
        "method": "pygam_LinearGAM_K20_residual_on_chemprop_aux",
        "rationale": (
            "Generalized Additive Model: per-feature smoothing cubic-splines "
            "(pygam.LinearGAM s(0)+...+s(19), n_splines=20, spline_order=3, "
            "lam=0.6) on K=20 substrate; additive non-linear per-feature "
            "transforms with NO interactions, contrasting axis-aligned tree "
            "splitters (LGBM/GBR/XGB/RF/CatBoost) which discover the same "
            "shape via interaction splits; penalized-spline shrinkage is a "
            "different regularizer than tree leaf-count/min-child-weight"
        ),
        "anchor": ANCHOR,
        "anchor_te_path": str(ANCHOR_TE_PATH),
        "anchor_pre_unblind": True,
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
        "model_class": "pygam.LinearGAM",
        "gam_n_splines": GAM_N_SPLINES,
        "gam_spline_order": GAM_SPLINE_ORDER,
        "gam_lam": GAM_LAM,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "residual_abs_p90": float(np.quantile(np.abs(residual), 0.9)),
        "per_seed_rae": [float(r) for r in per_seed_rae],
        "per_seed_mean_rae": per_seed_mean,
        "per_seed_std_rae": per_seed_std,
        "mean_bag_rae": rae_mean_bag,
        "median_bag_rae": rae_median_bag,
        "mean_rae": rae_mean_bag,  # alias for gate consumers
        "delta_vs_anchor": rae_mean_bag - rae_anchor,
        "delta_vs_nb2240_K20_lgbm": rae_mean_bag - NB2240_K20_REF,
        "nb2240_K20_lgbm_ref": NB2240_K20_REF,
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
        "status",
        "per_seed_rae",
        "per_seed_mean_rae",
        "per_seed_std_rae",
        "mean_bag_rae",
        "median_bag_rae",
        "delta_vs_anchor",
        "delta_vs_nb2240_K20_lgbm",
        "te_unb_in_sample_rae",
        "residual_abs_p90",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
