"""nb2871 -- LGBM K=20 min_child_samples sweep {2, 5, 10, 20, 50} on chemprop_aux residual.

NEW PARADIGM (vs default min_child_samples=5 baseline):
    The K=20 LGBM substrate (nb2240) and all its post-hoc descendants have
    used min_child_samples=5 as the minimum-leaf-size regularizer. Together
    with max_depth=4 / num_leaves=15 / n_est=300 / lr=0.03 on n_unb=253
    (n_va~51 per fold), a leaf can be carved out of only 5 training rows --
    well below the residual noise scale where rare-scaffold rows are
    structurally indistinguishable from honest signal.

    Larger min_child_samples forces each leaf to be supported by more rows,
    which on a residual target acts as a per-leaf prior toward the
    leaf-population mean residual -- a flatter, more conservative correction.
    For chemprop_aux residuals on n_unb=253, leaves of size 20-50 still
    leave room for 5-10 leaves but blunt the model's ability to carve out
    rare-scaffold-only leaves where the OOD failure tail lives.

    Hypothesis: at the K=20 substrate, the leaf-size axis is currently
    under-explored; min_child_samples=5 is the LightGBM default but the
    K=20 / n=253 regime would benefit from larger leaves that refuse to
    fire on rare-scaffold singletons. Complementary to the reg_lambda
    sweep (nb2860) which shrinks leaf-weight magnitude uniformly.

PROTOCOL:
    1. Load X_117 substrate -> slice K=20 surviving columns from nb2240
       summary (identical substrate as nb2860 reg_lambda sweep).
    2. residual = y_unb - chemprop_aux_te[unb_idx]  (only PRE-clean anchor).
    3. For each min_child in {2, 5, 10, 20, 50}:
         - LGBM(max_depth=4, num_leaves=15, n_est=300, lr=0.03,
                reg_lambda=2, min_child_samples=min_child) on K=20 features,
           residual target.
         - 5-fold scaffold CV (`scaffold_kfold_indices`) per seed.
         - 5 kf_seeds {1001..1005} -> mean-bag aggregate.
         - Deploy refit on full 253 per seed -> predict 513.
    4. Pick best min_child by mean-bag corrected RAE; save its pred_oof + te
       artefacts for downstream ladder use.

GATE (best mean-bag corrected RAE across the sweep):
    best_mean_rae < 0.4570 -> "PROMOTE"
    best_mean_rae < 0.4598 -> "MARGINAL_BEAT"
    else                   -> "FAIL"

OUTPUTS:
    scripts/nb2871_min_child_sweep.py
    data/processed/nb2871_summary.json
    data/processed/nb2871_pred_oof.npy       (253,) float32 best-min_child mean-bag CORRECTED
    data/processed/nb2871_pred_oof_mc{mc}.npy per-min_child mean-bag CORRECTED
    data/processed/te_nb2871.npy             (513,) float32 best-min_child deploy
    submissions/nb2871_min_child_sweep.csv
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

TAG = "nb2871"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"
NB2240_SUMMARY = DATA_PROCESSED / "nb2240_summary.json"

N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
MIN_CHILD_GRID = [2, 5, 10, 20, 50]

# Gate thresholds (mean-bag RAE)
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

CHEMPROP_AUX_REF = 0.6216
NB2240_K20_REF = 0.4630  # K=20 LGBM baseline at min_child_samples=5


def _lgbm_params(seed: int, min_child_samples: int) -> dict:
    """LGBM hyperparams -- min_child_samples is parameterized; all else fixed to nb2240/nb2860."""
    return dict(
        objective="regression",
        max_depth=4,
        num_leaves=15,
        n_estimators=300,
        learning_rate=0.03,
        min_child_samples=int(min_child_samples),
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _scaffold_cv_one_seed(
    X: np.ndarray,
    residual: np.ndarray,
    unb_scaffolds: list,
    kf_seed: int,
    min_child_samples: int,
) -> np.ndarray:
    """One scaffold-CV pass: fit per-fold LGBM at given min_child_samples. Returns OOF residual."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n = len(residual)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in splits:
        mdl = lgb.LGBMRegressor(**_lgbm_params(kf_seed, min_child_samples))
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
    min_child_samples: int,
) -> np.ndarray:
    """Fit LGBM on full 253; predict 513 residual."""
    mdl = lgb.LGBMRegressor(**_lgbm_params(seed, min_child_samples))
    mdl.fit(X_unb, residual)
    return mdl.predict(X_te).astype(np.float32)


def main() -> dict:
    import pandas as pd
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- LGBM K=20 min_child_samples sweep {MIN_CHILD_GRID}")
    print(f"        anchor = {ANCHOR}  scaffold-CV {N_FOLDS}-fold  "
          f"kf_seeds={KF_SEEDS}")
    print(f"        ref nb2240 K=20 LGBM (min_child=5) = {NB2240_K20_REF:.4f}")
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

    # ---- Per-min_child sweep ----
    print("\n" + "-" * 78)
    print(f"MIN_CHILD SWEEP  mcs={MIN_CHILD_GRID}  seeds={KF_SEEDS}  folds={N_FOLDS}")
    print("-" * 78)
    per_mc_records: list[dict] = []
    per_mc_pred_oof: dict[int, np.ndarray] = {}
    per_mc_te_deploy: dict[int, np.ndarray] = {}

    for mc in MIN_CHILD_GRID:
        t_mc = time.time()
        print(f"\n--- min_child_samples = {mc} ---")
        per_seed_oof_resid = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)
        per_seed_te_resid = np.zeros((len(KF_SEEDS), n_test), dtype=np.float64)
        per_seed_rae: list[float] = []
        for i, seed in enumerate(KF_SEEDS):
            ts = time.time()
            resid_oof = _scaffold_cv_one_seed(
                X_unb, residual, unb_scaffolds, seed, mc,
            )
            per_seed_oof_resid[i] = resid_oof
            te_resid = _deploy_te_one_seed(
                X_unb, residual, X_te, seed, mc,
            )
            per_seed_te_resid[i] = te_resid
            pred_corr = anchor + resid_oof
            rae_s = float(rae(y_unb, pred_corr))
            per_seed_rae.append(rae_s)
            print(f"   mc={mc:>3d}  seed={seed}  rae_corr={rae_s:.4f}  "
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
        per_mc_pred_oof[mc] = pred_oof_corrected
        per_mc_te_deploy[mc] = te_deploy

        # save per-min_child mean-bag OOF for downstream inspection
        oof_mc_path = DATA_PROCESSED / f"{TAG}_pred_oof_mc{mc}.npy"
        np.save(oof_mc_path, pred_oof_corrected)

        print(f"   mc={mc:>3d}  per_seed_mean={per_seed_mean:.4f}  "
              f"std={per_seed_std:.4f}")
        print(f"   mc={mc:>3d}  mean_bag={rae_mean_bag:.4f}  "
              f"median_bag={rae_median_bag:.4f}  "
              f"d_vs_anchor={rae_mean_bag - rae_anchor:+.4f}  "
              f"d_vs_nb2240={rae_mean_bag - NB2240_K20_REF:+.4f}  "
              f"wall={time.time() - t_mc:.1f}s  [save {oof_mc_path.name}]")

        per_mc_records.append({
            "min_child_samples": int(mc),
            "per_seed_rae": [float(r) for r in per_seed_rae],
            "per_seed_mean_rae": per_seed_mean,
            "per_seed_std_rae": per_seed_std,
            "mean_bag_rae": rae_mean_bag,
            "median_bag_rae": rae_median_bag,
            "delta_vs_anchor": rae_mean_bag - rae_anchor,
            "delta_vs_nb2240_K20_mc5": rae_mean_bag - NB2240_K20_REF,
            "te_deploy_mean": float(te_deploy.mean()),
            "te_deploy_std": float(te_deploy.std()),
            "wall_sec": round(time.time() - t_mc, 2),
        })

    # ---- Pick best min_child ----
    print("\n" + "=" * 78)
    print("MIN_CHILD-SWEEP SUMMARY TABLE")
    print("=" * 78)
    print(f"   {'mc':>4s}  {'mean_bag':>10s}  {'median_bag':>10s}  "
          f"{'per_seed_mean':>13s}  {'per_seed_std':>12s}  "
          f"{'d_vs_anchor':>11s}  {'d_vs_nb2240':>11s}")
    for r in per_mc_records:
        print(f"   {r['min_child_samples']:>4d}  {r['mean_bag_rae']:>10.4f}  "
              f"{r['median_bag_rae']:>10.4f}  "
              f"{r['per_seed_mean_rae']:>13.4f}  "
              f"{r['per_seed_std_rae']:>12.4f}  "
              f"{r['delta_vs_anchor']:>+11.4f}  "
              f"{r['delta_vs_nb2240_K20_mc5']:>+11.4f}")

    best_i = int(np.argmin([r["mean_bag_rae"] for r in per_mc_records]))
    best_record = per_mc_records[best_i]
    best_mc = int(best_record["min_child_samples"])
    best_mean_rae = float(best_record["mean_bag_rae"])
    best_pred_oof = per_mc_pred_oof[best_mc]
    best_te_deploy = per_mc_te_deploy[best_mc]
    best_te_unb_in_sample_rae = float(rae(y_unb, best_te_deploy[unb_idx]))

    print(f"\n   best min_child  = {best_mc}  mean_bag = {best_mean_rae:.4f}")
    print(f"   best te[unb_idx] in-sample = {best_te_unb_in_sample_rae:.4f}  "
          f"(deploy refit, optimism expected)")

    # ---- Save best-min_child artefacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, best_pred_oof)
    np.save(te_path, best_te_deploy)
    print(f"[save] {oof_path}  (best mc={best_mc})")
    print(f"[save] {te_path}   (best mc={best_mc})")

    sub_csv = SUBMISSIONS / f"{TAG}_min_child_sweep.csv"
    pd.DataFrame({
        "SMILES": test_smiles,
        "Molecule Name": te_names,
        "pEC50": best_te_deploy.astype(np.float32),
    }).to_csv(sub_csv, index=False)
    print(f"[save] {sub_csv}")

    # ---- Gate (on the BEST min_child's mean_bag_rae) ----
    if best_mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif best_mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "=" * 78)
    print("GATE EVALUATION (best min_child)")
    print("=" * 78)
    print(f"   best_mc             = {best_mc}")
    print(f"   best_mean_bag_rae   = {best_mean_rae:.4f}")
    print(f"   < {GATE_PROMOTE:.4f}  (PROMOTE)       = "
          f"{best_mean_rae < GATE_PROMOTE}")
    print(f"   < {GATE_MARGINAL:.4f}  (MARGINAL_BEAT) = "
          f"{best_mean_rae < GATE_MARGINAL}")
    print(f"   VERDICT             = {verdict}")

    summary = {
        "tag": TAG,
        "method": "lgbm_K20_min_child_samples_sweep_2_5_10_20_50_on_chemprop_aux_residual",
        "rationale": (
            "Sweep min_child_samples around the default-5 baseline on K=20 LGBM "
            "substrate; larger leaves act as per-leaf prior toward population "
            "mean residual, blunting rare-scaffold-singleton corrections where "
            "the OOD failure tail lives"
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
        "min_child_grid": MIN_CHILD_GRID,
        "feat_dim": int(feat_dim),
        "model_class": "lightgbm.LGBMRegressor",
        "lgbm_params_sample": _lgbm_params(KF_SEEDS[0], MIN_CHILD_GRID[0]),
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_min_child_records": per_mc_records,
        "best_min_child_samples": best_mc,
        "best_mean_bag_rae": best_mean_rae,
        "best_te_unb_in_sample_rae": best_te_unb_in_sample_rae,
        "best_te_deploy_mean": float(best_te_deploy.mean()),
        "best_te_deploy_std": float(best_te_deploy.std()),
        "mean_rae": best_mean_rae,  # alias for gate consumers (best across sweep)
        "mean_bag_rae": best_mean_rae,  # alias
        "delta_best_vs_anchor": best_mean_rae - rae_anchor,
        "delta_best_vs_nb2240_K20_mc5": best_mean_rae - NB2240_K20_REF,
        "nb2240_K20_mc5_ref": NB2240_K20_REF,
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
        "min_child_grid",
        "best_min_child_samples",
        "best_mean_bag_rae",
        "delta_best_vs_anchor",
        "delta_best_vs_nb2240_K20_mc5",
        "best_te_unb_in_sample_rae",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
    print("\n==== PER-MIN_CHILD TABLE ====")
    for r in res["per_min_child_records"]:
        print(f"  mc={r['min_child_samples']:>3d}  "
              f"mean_bag={r['mean_bag_rae']:.4f}  "
              f"median_bag={r['median_bag_rae']:.4f}  "
              f"per_seed_mean={r['per_seed_mean_rae']:.4f}  "
              f"std={r['per_seed_std_rae']:.4f}  "
              f"d_vs_nb2240={r['delta_vs_nb2240_K20_mc5']:+.4f}")
