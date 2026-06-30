"""nb2860 -- LGBM K=20 reg_lambda sweep {2, 5, 10, 20, 50} on chemprop_aux residual.

NEW PARADIGM (vs default reg_lambda=2 baseline):
    The K=20 LGBM substrate (nb2240) and all its post-hoc descendants have
    used reg_lambda=2.0 as the L2 leaf-weight regularizer. With max_depth=4 /
    num_leaves=15 / n_est=300 / lr=0.03 we have ~5x as many leaf-weight
    parameters as effective unb_n / fold_n_va training samples per fit,
    which is the exact regime where stronger L2 should bite first (cf. the
    cycle-167 anchor-swap evidence that the post-hoc-blend ceiling is set
    by leaf-overfit on the chemprop_aux residual, not by anchor capacity).

    Stronger lambda shrinks leaf weights uniformly toward 0, which on a
    residual target acts as a per-leaf prior toward "no correction" --
    exactly the right inductive bias when most of the OOD failure tail
    comes from rare-scaffold rows that should NOT receive a confident
    correction. lambda in {5, 10, 20, 50} is well above the LightGBM
    default (0) and the nb2240 family value (2), into a regime where the
    leaf-weight shrinkage magnitude can match the residual noise scale
    sigma_resid ~ 0.6-0.7.

    Hypothesis: at the K=20 substrate, the L2 axis is currently
    under-explored; a non-trivial fraction of the K=20 columns are
    sufficiently low-signal that a tighter lambda gives the model
    permission to be more conservative on rare-scaffold rows without
    blunting its corrections on the within-manifold rows.

PROTOCOL:
    1. Load X_117 substrate -> slice K=20 surviving columns from nb2240
       summary (identical substrate as nb2700 RobustScaler experiment).
    2. residual = y_unb - chemprop_aux_te[unb_idx]  (only PRE-clean anchor).
    3. For each lambda in {2.0, 5.0, 10.0, 20.0, 50.0}:
         - LGBM(max_depth=4, num_leaves=15, n_est=300, lr=0.03,
                min_child_samples=5, reg_lambda=lambda) on K=20 features,
           residual target.
         - 5-fold scaffold CV (`scaffold_kfold_indices`) per seed.
         - 5 kf_seeds {1001..1005} -> mean-bag aggregate.
         - Deploy refit on full 253 per seed -> predict 513.
    4. Pick best lambda by mean-bag corrected RAE; save its pred_oof + te
       artefacts for downstream ladder use.

GATE (best mean-bag corrected RAE across the sweep):
    best_mean_rae < 0.4570 -> "PROMOTE"
    best_mean_rae < 0.4598 -> "MARGINAL_BEAT"
    else                   -> "FAIL"

OUTPUTS:
    scripts/nb2860_reg_lambda_sweep.py
    data/processed/nb2860_summary.json
    data/processed/nb2860_pred_oof.npy       (253,) float32 best-lambda mean-bag CORRECTED
    data/processed/nb2860_pred_oof_lam{lam}.npy per-lambda mean-bag CORRECTED
    data/processed/te_nb2860.npy             (513,) float32 best-lambda deploy
    submissions/nb2860_reg_lambda_sweep.csv
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

TAG = "nb2860"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"
NB2240_SUMMARY = DATA_PROCESSED / "nb2240_summary.json"

N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
LAMBDA_GRID = [2.0, 5.0, 10.0, 20.0, 50.0]

# Gate thresholds (mean-bag RAE)
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

CHEMPROP_AUX_REF = 0.6216
NB2240_K20_REF = 0.4630  # K=20 LGBM baseline at reg_lambda=2.0


def _lgbm_params(seed: int, reg_lambda: float) -> dict:
    """LGBM hyperparams -- reg_lambda is parameterized; all else fixed to nb2240/nb2700."""
    return dict(
        objective="regression",
        max_depth=4,
        num_leaves=15,
        n_estimators=300,
        learning_rate=0.03,
        min_child_samples=5,
        reg_lambda=float(reg_lambda),
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _scaffold_cv_one_seed(
    X: np.ndarray,
    residual: np.ndarray,
    unb_scaffolds: list,
    kf_seed: int,
    reg_lambda: float,
) -> np.ndarray:
    """One scaffold-CV pass: fit per-fold LGBM at given reg_lambda. Returns OOF residual."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n = len(residual)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in splits:
        mdl = lgb.LGBMRegressor(**_lgbm_params(kf_seed, reg_lambda))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    if np.isnan(oof).any():
        raise RuntimeError("scaffold split did not cover all rows")
    return oof


def _deploy_te_one_seed(
    X_unb: np.ndarray,
    residual: np.ndarray,
    X_te: np.ndarray,
    seed: int,
    reg_lambda: float,
) -> np.ndarray:
    """Fit LGBM on full 253; predict 513 residual."""
    mdl = lgb.LGBMRegressor(**_lgbm_params(seed, reg_lambda))
    mdl.fit(X_unb, residual)
    return mdl.predict(X_te).astype(np.float32)


def main() -> dict:
    import pandas as pd
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- LGBM K=20 reg_lambda sweep {LAMBDA_GRID}")
    print(f"        anchor = {ANCHOR}  scaffold-CV {N_FOLDS}-fold  "
          f"kf_seeds={KF_SEEDS}")
    print(f"        ref nb2240 K=20 LGBM (reg_lambda=2.0) = {NB2240_K20_REF:.4f}")
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
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

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

    # ---- Per-lambda sweep ----
    print("\n" + "-" * 78)
    print(f"LAMBDA SWEEP  lams={LAMBDA_GRID}  seeds={KF_SEEDS}  folds={N_FOLDS}")
    print("-" * 78)
    per_lam_records: list[dict] = []
    per_lam_pred_oof: dict[float, np.ndarray] = {}
    per_lam_te_deploy: dict[float, np.ndarray] = {}

    for lam in LAMBDA_GRID:
        t_lam = time.time()
        print(f"\n--- reg_lambda = {lam} ---")
        per_seed_oof_resid = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)
        per_seed_te_resid = np.zeros((len(KF_SEEDS), n_test), dtype=np.float64)
        per_seed_rae: list[float] = []
        for i, seed in enumerate(KF_SEEDS):
            ts = time.time()
            resid_oof = _scaffold_cv_one_seed(
                X_unb, residual, unb_scaffolds, seed, lam,
            )
            per_seed_oof_resid[i] = resid_oof
            te_resid = _deploy_te_one_seed(
                X_unb, residual, X_te, seed, lam,
            )
            per_seed_te_resid[i] = te_resid
            pred_corr = anchor + resid_oof
            rae_s = float(rae(y_unb, pred_corr))
            per_seed_rae.append(rae_s)
            print(f"   lam={lam:>5g}  seed={seed}  rae_corr={rae_s:.4f}  "
                  f"d_vs_anchor={rae_s - rae_anchor:+.4f}  "
                  f"wall={time.time() - ts:.1f}s")

        per_seed_mean = float(np.mean(per_seed_rae))
        per_seed_std = float(np.std(per_seed_rae))
        mean_bag_resid = per_seed_oof_resid.mean(axis=0)
        median_bag_resid = np.median(per_seed_oof_resid, axis=0)
        rae_mean_bag = float(rae(y_unb, anchor + mean_bag_resid))
        rae_median_bag = float(rae(y_unb, anchor + median_bag_resid))

        mean_bag_te_resid = per_seed_te_resid.mean(axis=0)
        te_deploy = (te_anchor_513 + mean_bag_te_resid).astype(np.float32)

        pred_oof_corrected = (anchor + mean_bag_resid).astype(np.float32)
        per_lam_pred_oof[lam] = pred_oof_corrected
        per_lam_te_deploy[lam] = te_deploy

        # save per-lambda mean-bag OOF for downstream inspection
        oof_lam_path = DATA_PROCESSED / f"{TAG}_pred_oof_lam{lam}.npy"
        np.save(oof_lam_path, pred_oof_corrected)

        print(f"   lam={lam:>5g}  per_seed_mean={per_seed_mean:.4f}  "
              f"std={per_seed_std:.4f}")
        print(f"   lam={lam:>5g}  mean_bag={rae_mean_bag:.4f}  "
              f"median_bag={rae_median_bag:.4f}  "
              f"d_vs_anchor={rae_mean_bag - rae_anchor:+.4f}  "
              f"d_vs_nb2240={rae_mean_bag - NB2240_K20_REF:+.4f}  "
              f"wall={time.time() - t_lam:.1f}s  [save {oof_lam_path.name}]")

        per_lam_records.append({
            "reg_lambda": float(lam),
            "per_seed_rae": [float(r) for r in per_seed_rae],
            "per_seed_mean_rae": per_seed_mean,
            "per_seed_std_rae": per_seed_std,
            "mean_bag_rae": rae_mean_bag,
            "median_bag_rae": rae_median_bag,
            "delta_vs_anchor": rae_mean_bag - rae_anchor,
            "delta_vs_nb2240_K20_lam2": rae_mean_bag - NB2240_K20_REF,
            "te_deploy_mean": float(te_deploy.mean()),
            "te_deploy_std": float(te_deploy.std()),
            "wall_sec": round(time.time() - t_lam, 2),
        })

    # ---- Pick best lambda ----
    print("\n" + "=" * 78)
    print("LAMBDA-SWEEP SUMMARY TABLE")
    print("=" * 78)
    print(f"   {'lam':>6s}  {'mean_bag':>10s}  {'median_bag':>10s}  "
          f"{'per_seed_mean':>13s}  {'per_seed_std':>12s}  "
          f"{'d_vs_anchor':>11s}  {'d_vs_nb2240':>11s}")
    for r in per_lam_records:
        print(f"   {r['reg_lambda']:>6g}  {r['mean_bag_rae']:>10.4f}  "
              f"{r['median_bag_rae']:>10.4f}  "
              f"{r['per_seed_mean_rae']:>13.4f}  "
              f"{r['per_seed_std_rae']:>12.4f}  "
              f"{r['delta_vs_anchor']:>+11.4f}  "
              f"{r['delta_vs_nb2240_K20_lam2']:>+11.4f}")

    best_i = int(np.argmin([r["mean_bag_rae"] for r in per_lam_records]))
    best_record = per_lam_records[best_i]
    best_lam = float(best_record["reg_lambda"])
    best_mean_rae = float(best_record["mean_bag_rae"])
    best_pred_oof = per_lam_pred_oof[best_lam]
    best_te_deploy = per_lam_te_deploy[best_lam]
    best_te_unb_in_sample_rae = float(rae(y_unb, best_te_deploy[unb_idx]))

    print(f"\n   best lambda  = {best_lam:g}  mean_bag = {best_mean_rae:.4f}")
    print(f"   best te[unb_idx] in-sample = {best_te_unb_in_sample_rae:.4f}  "
          f"(deploy refit, optimism expected)")

    # ---- Save best-lambda artefacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, best_pred_oof)
    np.save(te_path, best_te_deploy)
    print(f"[save] {oof_path}  (best lam={best_lam:g})")
    print(f"[save] {te_path}   (best lam={best_lam:g})")

    sub_csv = SUBMISSIONS / f"{TAG}_reg_lambda_sweep.csv"
    pd.DataFrame({
        "SMILES": test_smiles,
        "Molecule Name": te_names,
        "pEC50": best_te_deploy.astype(np.float32),
    }).to_csv(sub_csv, index=False)
    print(f"[save] {sub_csv}")

    # ---- Gate (on the BEST lambda's mean_bag_rae) ----
    if best_mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif best_mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "=" * 78)
    print("GATE EVALUATION (best lambda)")
    print("=" * 78)
    print(f"   best_lam            = {best_lam:g}")
    print(f"   best_mean_bag_rae   = {best_mean_rae:.4f}")
    print(f"   < {GATE_PROMOTE:.4f}  (PROMOTE)       = "
          f"{best_mean_rae < GATE_PROMOTE}")
    print(f"   < {GATE_MARGINAL:.4f}  (MARGINAL_BEAT) = "
          f"{best_mean_rae < GATE_MARGINAL}")
    print(f"   VERDICT             = {verdict}")

    summary = {
        "tag": TAG,
        "method": "lgbm_K20_reg_lambda_sweep_2_5_10_20_50_on_chemprop_aux_residual",
        "rationale": (
            "Sweep reg_lambda well above the default-2 baseline on K=20 LGBM "
            "substrate; tighter L2 acts as per-leaf prior toward 'no correction', "
            "appropriate for rare-scaffold OOD rows where confident corrections "
            "are the dominant failure mode"
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
        "lambda_grid": LAMBDA_GRID,
        "feat_dim": int(feat_dim),
        "model_class": "lightgbm.LGBMRegressor",
        "lgbm_params_sample": _lgbm_params(KF_SEEDS[0], LAMBDA_GRID[0]),
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_lambda_records": per_lam_records,
        "best_lambda": best_lam,
        "best_mean_bag_rae": best_mean_rae,
        "best_te_unb_in_sample_rae": best_te_unb_in_sample_rae,
        "best_te_deploy_mean": float(best_te_deploy.mean()),
        "best_te_deploy_std": float(best_te_deploy.std()),
        "mean_rae": best_mean_rae,  # alias for gate consumers (best across sweep)
        "mean_bag_rae": best_mean_rae,  # alias
        "delta_best_vs_anchor": best_mean_rae - rae_anchor,
        "delta_best_vs_nb2240_K20_lam2": best_mean_rae - NB2240_K20_REF,
        "nb2240_K20_lam2_ref": NB2240_K20_REF,
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
        "lambda_grid",
        "best_lambda",
        "best_mean_bag_rae",
        "delta_best_vs_anchor",
        "delta_best_vs_nb2240_K20_lam2",
        "best_te_unb_in_sample_rae",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
    print("\n==== PER-LAMBDA TABLE ====")
    for r in res["per_lambda_records"]:
        print(f"  lam={r['reg_lambda']:>5g}  "
              f"mean_bag={r['mean_bag_rae']:.4f}  "
              f"median_bag={r['median_bag_rae']:.4f}  "
              f"per_seed_mean={r['per_seed_mean_rae']:.4f}  "
              f"std={r['per_seed_std_rae']:.4f}  "
              f"d_vs_nb2240={r['delta_vs_nb2240_K20_lam2']:+.4f}")
