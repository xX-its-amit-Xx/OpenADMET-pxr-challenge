"""nb2890 -- LGBM K=20 native Tweedie loss with variance_power sweep
on shifted chemprop_aux residual.

NEW PARADIGM (vs nb2872 Poisson, nb2240 L2 baseline):
    LightGBM's Tweedie objective interpolates between Poisson (variance_power=1.0)
    and Gamma (variance_power=2.0). The variance-mean relationship is
        var(y) = mu**variance_power
    and the log-link inside LightGBM means leaf splits trade off mid-magnitude
    vs high-magnitude residuals along a continuum controlled by variance_power.

    nb2872 fixed variance_power=1.0 (pure Poisson). The F2 failure mode
    (greasy-novel-inactive over-prediction; -0.11 RAE prize per Phase-2 pm06)
    has a heavy LEFT tail, but the optimal Tweedie variance_power for this
    geometry is unknown. variance_power closer to 2.0 (Gamma) tolerates more
    extreme high-magnitude values; closer to 1.0 (Poisson) concentrates mass.

    Sweeping variance_power ∈ {1.1, 1.3, 1.5, 1.7, 1.9} reveals the F2-tail-
    optimal point. The shift y_shift = y_resid - min(y_resid) + 0.1 satisfies
    the Tweedie y >= 0 constraint and preserves rank order.

PROTOCOL:
    1. Load X_117 substrate -> slice K=20 cols from nb2240 summary.
    2. residual = y_unb - chemprop_aux_te[unb_idx]  (only PRE-clean anchor).
    3. shift = -min(residual) + 0.1;  y_shift = residual + shift  (>= 0.1).
    4. For each variance_power in {1.1, 1.3, 1.5, 1.7, 1.9}:
         LGBM(objective='tweedie', tweedie_variance_power=vp,
              max_depth=4, num_leaves=15, n_estimators=300, learning_rate=0.03).
         5-fold scaffold CV per seed; 5 kf_seeds {1001..1005} -> mean-bag.
    5. Pick best variance_power by mean_bag corrected RAE; deploy refit at
       best variance_power on full 253 per seed -> predict 513.

GATE (best mean_bag corrected RAE across the sweep):
    best_mean_rae < 0.4570 -> "PROMOTE"
    best_mean_rae < 0.4598 -> "MARGINAL_BEAT"
    else                   -> "FAIL"

OUTPUTS:
    scripts/nb2890_tweedie_variance_sweep.py
    data/processed/nb2890_summary.json
    data/processed/nb2890_pred_oof.npy       (253,) float32 mean-bag CORRECTED (best vp)
    data/processed/te_nb2890.npy             (513,) float32 deploy (best vp)
    submissions/nb2890_tweedie_variance_sweep.csv
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

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2890"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"
NB2240_SUMMARY = DATA_PROCESSED / "nb2240_summary.json"

N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# Variance power sweep: 1.0=Poisson, 2.0=Gamma; LGBM requires 1<vp<2
VARIANCE_POWER_GRID = [1.1, 1.3, 1.5, 1.7, 1.9]

# Shift epsilon for Tweedie positivity (small, well below typical |residual|)
SHIFT_EPS = 0.1

# Gate thresholds (mean-bag RAE)
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

CHEMPROP_AUX_REF = 0.6216
NB2240_K20_REF = 0.4630   # K=20 LGBM baseline (L2)
NB2872_POISSON_REF = None  # filled if available


def _lgbm_params(seed: int, variance_power: float) -> dict:
    """LGBM hyperparams -- Tweedie objective with sweepable variance_power."""
    return dict(
        objective="tweedie",
        tweedie_variance_power=variance_power,
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


def _scaffold_cv_one_seed(
    X: np.ndarray,
    y_shift: np.ndarray,
    unb_scaffolds: list,
    kf_seed: int,
    variance_power: float,
) -> np.ndarray:
    """One scaffold-CV pass. Returns OOF predictions in SHIFTED space."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n = len(y_shift)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in splits:
        mdl = lgb.LGBMRegressor(**_lgbm_params(kf_seed, variance_power))
        mdl.fit(X[tr_loc], y_shift[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    if np.isnan(oof).any():
        raise RuntimeError("scaffold split did not cover all rows")
    return oof


def _deploy_te_one_seed(
    X_unb: np.ndarray,
    y_shift: np.ndarray,
    X_te: np.ndarray,
    seed: int,
    variance_power: float,
) -> np.ndarray:
    """Fit LGBM on full 253; predict 513 in SHIFTED space."""
    mdl = lgb.LGBMRegressor(**_lgbm_params(seed, variance_power))
    mdl.fit(X_unb, y_shift)
    return mdl.predict(X_te).astype(np.float64)


def main() -> dict:
    import pandas as pd
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- LGBM K=20 native Tweedie variance_power sweep "
          f"on shifted chemprop_aux residual")
    print(f"        anchor = {ANCHOR}  scaffold-CV {N_FOLDS}-fold  "
          f"kf_seeds={KF_SEEDS}")
    print(f"        shift = -min(resid) + {SHIFT_EPS}")
    print(f"        variance_power grid = {VARIANCE_POWER_GRID}")
    print(f"        ref nb2240 K=20 LGBM (L2) = {NB2240_K20_REF:.4f}")
    print(f"        GATE: <{GATE_PROMOTE} PROMOTE / "
          f"<{GATE_MARGINAL} MARGINAL_BEAT / else FAIL")
    print("=" * 78)

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
    resid_min = float(residual.min())
    resid_max = float(residual.max())
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}  "
          f"min={resid_min:+.4f}  max={resid_max:+.4f}")

    # ---- Shift residual into positive support for Tweedie ----
    shift = -resid_min + SHIFT_EPS
    y_shift = residual + shift
    assert (y_shift >= SHIFT_EPS - 1e-9).all(), \
        f"y_shift min {y_shift.min()} below SHIFT_EPS={SHIFT_EPS}"
    print(f"[shift] shift = {shift:+.4f}  "
          f"y_shift in [{y_shift.min():.4f}, {y_shift.max():.4f}]  "
          f"mean={y_shift.mean():.4f}  std={y_shift.std():.4f}")

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

    # ---- Sweep variance_power; for each, run per-seed scaffold-CV + deploy ----
    sweep_results: list[dict] = []
    sweep_oof_shift: dict[float, np.ndarray] = {}
    sweep_te_shift: dict[float, np.ndarray] = {}

    print("\n" + "-" * 78)
    print(f"TWEEDIE VARIANCE SWEEP  vp_grid={VARIANCE_POWER_GRID}  "
          f"seeds={KF_SEEDS}  folds={N_FOLDS}  shift={shift:+.4f}")
    print("-" * 78)

    for vp in VARIANCE_POWER_GRID:
        ts_vp = time.time()
        per_seed_oof_shift = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)
        per_seed_te_shift = np.zeros((len(KF_SEEDS), n_test), dtype=np.float64)
        per_seed_rae: list[float] = []
        for i, seed in enumerate(KF_SEEDS):
            oof_shift = _scaffold_cv_one_seed(
                X_unb, y_shift, unb_scaffolds, seed, vp
            )
            per_seed_oof_shift[i] = oof_shift
            te_shift = _deploy_te_one_seed(
                X_unb, y_shift, X_te, seed, vp
            )
            per_seed_te_shift[i] = te_shift
            resid_oof = oof_shift - shift
            pred_corr = anchor + resid_oof
            rae_s = float(rae(y_unb, pred_corr))
            per_seed_rae.append(rae_s)
        per_seed_mean = float(np.mean(per_seed_rae))
        per_seed_std = float(np.std(per_seed_rae))

        mean_bag_oof_shift = per_seed_oof_shift.mean(axis=0)
        mean_bag_resid = mean_bag_oof_shift - shift
        rae_mean_bag = float(rae(y_unb, anchor + mean_bag_resid))

        median_bag_oof_shift = np.median(per_seed_oof_shift, axis=0)
        median_bag_resid = median_bag_oof_shift - shift
        rae_median_bag = float(rae(y_unb, anchor + median_bag_resid))

        sweep_oof_shift[vp] = mean_bag_oof_shift
        sweep_te_shift[vp] = per_seed_te_shift.mean(axis=0)

        rec = dict(
            variance_power=float(vp),
            per_seed_rae=[float(r) for r in per_seed_rae],
            per_seed_mean_rae=per_seed_mean,
            per_seed_std_rae=per_seed_std,
            mean_bag_rae=rae_mean_bag,
            median_bag_rae=rae_median_bag,
            delta_vs_anchor=rae_mean_bag - rae_anchor,
            delta_vs_nb2240_K20_L2=rae_mean_bag - NB2240_K20_REF,
            wall_sec=round(time.time() - ts_vp, 2),
        )
        sweep_results.append(rec)
        print(
            f"   vp={vp:.2f}  mean_bag={rae_mean_bag:.4f}  "
            f"median_bag={rae_median_bag:.4f}  "
            f"seed_mean={per_seed_mean:.4f}±{per_seed_std:.4f}  "
            f"d_vs_anchor={rec['delta_vs_anchor']:+.4f}  "
            f"wall={rec['wall_sec']}s"
        )

    # ---- Pick best variance_power by mean_bag_rae ----
    best_rec = min(sweep_results, key=lambda r: r["mean_bag_rae"])
    best_vp = best_rec["variance_power"]
    best_mean_rae = best_rec["mean_bag_rae"]
    print("\n" + "-" * 78)
    print(f"BEST variance_power = {best_vp:.2f}  "
          f"mean_bag_rae = {best_mean_rae:.4f}  "
          f"d_vs_anchor = {best_rec['delta_vs_anchor']:+.4f}  "
          f"d_vs_nb2240_L2 = {best_rec['delta_vs_nb2240_K20_L2']:+.4f}")
    print("-" * 78)

    # ---- Build deploy artefacts at best_vp ----
    best_mean_bag_oof_shift = sweep_oof_shift[best_vp]
    best_mean_bag_resid = best_mean_bag_oof_shift - shift
    pred_oof_corrected = (anchor + best_mean_bag_resid).astype(np.float32)

    best_mean_bag_te_shift = sweep_te_shift[best_vp]
    best_mean_bag_te_resid = best_mean_bag_te_shift - shift
    te_deploy = (te_anchor_513 + best_mean_bag_te_resid).astype(np.float32)
    te_unb_in_sample_rae = float(rae(y_unb, te_deploy[unb_idx]))
    print(f"   te[unb_idx] in-sample (best vp) = {te_unb_in_sample_rae:.4f}  "
          f"(deploy refit, optimism expected)")

    # ---- Save artefacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof_corrected)
    np.save(te_path, te_deploy)
    print(f"[save] {oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_tweedie_variance_sweep.csv"
    pd.DataFrame({
        "SMILES": test_smiles,
        "Molecule Name": te_names,
        "pEC50": te_deploy.astype(np.float32),
    }).to_csv(sub_csv, index=False)
    print(f"[save] {sub_csv}")

    # ---- Gate ----
    if best_mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif best_mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "=" * 78)
    print("GATE EVALUATION (best variance_power)")
    print("=" * 78)
    print(f"   best_variance_power = {best_vp:.2f}")
    print(f"   best_mean_bag_rae   = {best_mean_rae:.4f}")
    print(f"   < {GATE_PROMOTE:.4f}  (PROMOTE)       = {best_mean_rae < GATE_PROMOTE}")
    print(f"   < {GATE_MARGINAL:.4f}  (MARGINAL_BEAT) = {best_mean_rae < GATE_MARGINAL}")
    print(f"   VERDICT             = {verdict}")

    summary = {
        "tag": TAG,
        "method": "lgbm_K20_TWEEDIE_variance_power_sweep_on_shifted_chemprop_aux_residual",
        "rationale": (
            "Native Tweedie objective at the K=20 substrate, sweeping "
            "variance_power between Poisson (1.0) and Gamma (2.0). Tweedie's "
            "variance-mean coupling var=mu**vp and log-link inside LightGBM "
            "shifts leaf-split priority between mid-magnitude and "
            "high-magnitude residuals as vp moves from 1.1 (Poisson-like) "
            "toward 1.9 (Gamma-like). The shift y_shift = residual - "
            "min(residual) + 0.1 satisfies y >= 0 while preserving rank order. "
            "Sweep finds the F2-tail-optimal vp -- a single-parameter "
            "extension of nb2872 that addresses Poisson's fixed var=mean "
            "constraint."
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
        "model_class": "lightgbm.LGBMRegressor",
        "objective": "tweedie",
        "variance_power_grid": VARIANCE_POWER_GRID,
        "lgbm_params_sample": _lgbm_params(KF_SEEDS[0], best_vp),
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "residual_min": resid_min,
        "residual_max": resid_max,
        "shift_eps": SHIFT_EPS,
        "shift": float(shift),
        "y_shift_min": float(y_shift.min()),
        "y_shift_max": float(y_shift.max()),
        "y_shift_mean": float(y_shift.mean()),
        "y_shift_std": float(y_shift.std()),
        "sweep_results": sweep_results,
        "best_variance_power": float(best_vp),
        "best_per_seed_rae": best_rec["per_seed_rae"],
        "best_per_seed_mean_rae": best_rec["per_seed_mean_rae"],
        "best_per_seed_std_rae": best_rec["per_seed_std_rae"],
        "best_mean_bag_rae": best_mean_rae,
        "best_median_bag_rae": best_rec["median_bag_rae"],
        "mean_rae": best_mean_rae,  # alias for gate consumers
        "te_unb_in_sample_rae": te_unb_in_sample_rae,
        "te_deploy_mean": float(te_deploy.mean()),
        "te_deploy_std": float(te_deploy.std()),
        "delta_vs_anchor": best_mean_rae - rae_anchor,
        "delta_vs_nb2240_K20_L2": best_mean_rae - NB2240_K20_REF,
        "nb2240_K20_L2_ref": NB2240_K20_REF,
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
        "objective",
        "shift",
        "best_variance_power",
        "best_per_seed_mean_rae",
        "best_per_seed_std_rae",
        "best_mean_bag_rae",
        "best_median_bag_rae",
        "delta_vs_anchor",
        "delta_vs_nb2240_K20_L2",
        "te_unb_in_sample_rae",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
    print("\n  sweep_results:")
    for rec in res.get("sweep_results", []):
        print(
            f"    vp={rec['variance_power']:.2f}  "
            f"mean_bag={rec['mean_bag_rae']:.4f}  "
            f"median_bag={rec['median_bag_rae']:.4f}  "
            f"seed_mean={rec['per_seed_mean_rae']:.4f}±{rec['per_seed_std_rae']:.4f}"
        )
