"""nb2903 -- LGBM warm-start with init_score = nb1191 prediction (per row).

NEW PARADIGM vs nb2831 (anchor-axis swap):
    nb2831 used nb2240 (K=20 mean-bag) as the warm-start init -- nb2240 is
    itself a residual-stack on chemprop_aux, so the warm-start LGBM was
    effectively continuing the chemprop_aux + residual chain, with limited
    fresh degrees of freedom to add.

    nb2903 swaps the init axis to **nb1191** (PRE-unblind pyramid wide-seed
    candidate, deep-30 mean **0.4718 ± 0.0024**, verified in cycle 149).
    nb1191 lives on a different residual-target axis than nb2240 (its
    pyramid stack composes differently over the anchor chain), so warm-
    starting LGBM on top of nb1191 should expose a different residual
    surface for the trees to fit.

    The cycle-167 nb2171 anchor-swap finding (nb730 -> nb1191 broke the
    co-converged 0.4720 ceiling down to 0.4682) showed that nb1191 carries
    independent information not captured by the standard chemprop_aux +
    residual chain.  This script applies the same axis-swap idea to the
    warm-start boosting-continuation paradigm.

    Concretely:
        init_score = nb1191_pred    (per row, fold-train only)
        target     = y_unb          (full pEC50, NOT residual)
        LGBM       = max_depth=4, num_leaves=15, n_est=100, lr=0.01
                     (same small budget as nb2831 -- init is strong)
        substrate  = K=20 surviving cols from nb2240 K=20 (same 117 -> K=20
                     index list as nb2831; substrate held fixed so the only
                     variable is the init anchor axis)

    LightGBM `predict()` returns the boosted raw score on top of
    init_score, so we add the per-row init back to recover original scale.

PROTOCOL:
    1. Load nb1191 OOF on 253 (init for unb) and te_nb1191 on 513
       (init for test).
    2. Load K=20 feature substrate (X_117 sliced to nb2240
       k20_surviving_idx_in_117 = 20 cols).
    3. For each kf_seed in {1001..1005}: 5-fold scaffold-CV.  For each fold:
         init_tr   = nb1191_pred_oof[tr_loc]
         init_va   = nb1191_pred_oof[va_loc]
         LGBM.fit(X[tr_loc], y[tr_loc], init_score=init_tr)
         pred_va   = LGBM.predict(X[va_loc]) + init_va
       Pool OOF across folds -> per-seed RAE.
    4. Mean over 5 kf_seeds -> mean_rae (decision metric).
    5. Deploy refit on full 253 with init_score = nb1191_pred_oof, then
       predict 513 te with init_score = te_nb1191.

GATE (mean RAE over 5 kf_seeds):
    mean_rae < 0.4570 -> "PROMOTE"
    mean_rae < 0.4598 -> "MARGINAL_BEAT"
    else              -> "FAIL"

OUTPUTS:
    scripts/nb2903_lgbm_init_from_nb1191.py
    data/processed/nb2903_summary.json
    data/processed/nb2903_pred_oof.npy   (253,) float32 (mean-bag corrected pred)
    data/processed/te_nb2903.npy         (513,) float32 deploy refit
    submissions/nb2903_lgbm_init_from_nb1191.csv
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
import lightgbm as lgb

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2903"
ANCHOR = "nb1191"
INIT_OOF_PATH = DATA_PROCESSED / "nb1191_pred_oof.npy"
INIT_TE_PATH = DATA_PROCESSED / "te_nb1191.npy"

X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"
NB2240_SUMMARY = DATA_PROCESSED / "nb2240_summary.json"

UNBLIND_IDX = DATA_PROCESSED / "_audit_unblind_idx.npy"
UNBLIND_Y = DATA_PROCESSED / "_audit_unblind_y.npy"

# CV protocol
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# LGBM hyperparams (small incremental budget -- init is strong)
LGBM_MAX_DEPTH = 4
LGBM_NUM_LEAVES = 15
LGBM_N_ESTIMATORS = 100
LGBM_LEARNING_RATE = 0.01

# Gates
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

# Refs
CHEMPROP_AUX_REF = 0.6216
NB2240_K20_REF = 0.4630
NB2171_REF = 0.4682
NB1191_DEEP30_REF = 0.4718


def _lgbm_params(seed: int) -> dict:
    """Small-budget warm-start LGBM (init is strong, so smaller trees needed)."""
    return dict(
        objective="regression",
        max_depth=LGBM_MAX_DEPTH,
        num_leaves=LGBM_NUM_LEAVES,
        n_estimators=LGBM_N_ESTIMATORS,
        learning_rate=LGBM_LEARNING_RATE,
        min_child_samples=5,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _scaffold_keys_for_unb(unb_smiles: list[str]) -> list[str]:
    """Bemis-Murcko per row; singletons get unique placeholders so they don't
    co-group (mirrors `scaffold_kfold_indices` semantics)."""
    keys: list[str] = []
    for i, s in enumerate(unb_smiles):
        sc = bemis_murcko(s)
        if sc and isinstance(sc, str) and len(sc) > 0:
            keys.append(sc)
        else:
            keys.append(f"__singleton_{i}__")
    return keys


def _warm_cv_one_seed(
    X_unb: np.ndarray,
    y_unb: np.ndarray,
    init_oof: np.ndarray,
    scaffolds: list[str],
    seed: int,
) -> tuple[np.ndarray, list[float]]:
    """5-fold scaffold-CV warm-start LGBM.

    For each fold:
        init_tr   = init_oof[tr_loc]      (per-row warm-start)
        init_va   = init_oof[va_loc]
        LGBM.fit(X[tr_loc], y[tr_loc], init_score=init_tr)
        pred_va   = LGBM.predict(X[va_loc]) + init_va   (add init back)

    Returns (oof_pred, per_fold_rae).
    """
    n = len(y_unb)
    splits = scaffold_kfold_indices(
        scaffolds, n_splits=N_FOLDS, shuffle=True, seed=seed,
    )
    oof = np.full(n, np.nan, dtype=np.float64)
    per_fold_rae: list[float] = []
    for fi, (tr_loc, va_loc) in enumerate(splits):
        init_tr = init_oof[tr_loc].astype(np.float64)
        init_va = init_oof[va_loc].astype(np.float64)
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(
            X_unb[tr_loc],
            y_unb[tr_loc].astype(np.float64),
            init_score=init_tr,
        )
        # LightGBM sklearn predict returns RAW boosted score (on top of
        # init_score), so we must add init_va back to recover the full
        # original-scale prediction.
        raw_va = mdl.predict(X_unb[va_loc])
        oof[va_loc] = raw_va + init_va
        per_fold_rae.append(float(rae(y_unb[va_loc], oof[va_loc])))
    if np.isnan(oof).any():
        raise RuntimeError("scaffold split did not cover all rows")
    return oof, per_fold_rae


def _deploy_refit(
    X_unb: np.ndarray,
    y_unb: np.ndarray,
    init_oof: np.ndarray,
    X_te: np.ndarray,
    init_te: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Refit warm-start LGBM on full 253 unblind, predict 513 te."""
    init_all = init_oof.astype(np.float64)
    mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
    mdl.fit(X_unb, y_unb.astype(np.float64), init_score=init_all)
    raw_te = mdl.predict(X_te)
    return (raw_te + init_te.astype(np.float64)).astype(np.float32)


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- LGBM warm-start with init_score = nb1191 pred (per row)")
    print(f"          target = full pEC50  (NOT residual)")
    print(f"          init   = nb1191_pred_oof (unb) / te_nb1191 (te)")
    print(f"          LGBM   = max_depth={LGBM_MAX_DEPTH}  "
          f"num_leaves={LGBM_NUM_LEAVES}  n_est={LGBM_N_ESTIMATORS}  "
          f"lr={LGBM_LEARNING_RATE}")
    print(f"          CV     = {N_FOLDS}-fold scaffold  kf_seeds={KF_SEEDS}")
    print(f"          GATE   = <{GATE_PROMOTE} PROMOTE / "
          f"<{GATE_MARGINAL} MARGINAL_BEAT / else FAIL")
    print("=" * 78)

    # ---- Load test + unblind ----
    te = load_test()
    n_test = len(te)
    te_smiles = (
        te["smiles"].astype(str).tolist()
        if "smiles" in te.columns
        else te["SMILES"].astype(str).tolist()
    )
    te_names = (
        te["name"].values
        if "name" in te.columns
        else te["Molecule Name"].values
    )

    unb_idx = np.load(UNBLIND_IDX)
    y_unb = np.load(UNBLIND_Y).astype(np.float64)
    n_unb = len(y_unb)
    print(f"\n[load] n_test={n_test}  n_unb={n_unb}")

    # ---- Load nb1191 init (OOF on unb, te on 513) ----
    if not INIT_OOF_PATH.exists():
        raise FileNotFoundError(f"nb1191 OOF init missing: {INIT_OOF_PATH}")
    if not INIT_TE_PATH.exists():
        raise FileNotFoundError(f"nb1191 te init missing: {INIT_TE_PATH}")
    init_oof = np.load(INIT_OOF_PATH).astype(np.float64)
    init_te_513 = np.load(INIT_TE_PATH).astype(np.float64)
    if init_oof.shape != (n_unb,):
        raise ValueError(
            f"nb1191 OOF init shape {init_oof.shape} expected ({n_unb},)"
        )
    if init_te_513.shape != (n_test,):
        raise ValueError(
            f"nb1191 te init shape {init_te_513.shape} expected ({n_test},)"
        )
    rae_init = float(rae(y_unb, init_oof))
    print(f"[init] nb1191 OOF  RAE on 253 = {rae_init:.4f}  "
          f"(ref nb1191 deep-30 mean = {NB1191_DEEP30_REF:.4f})")
    print(f"[init] nb1191 OOF  mean/std   = "
          f"{init_oof.mean():.4f} / {init_oof.std():.4f}")
    print(f"[init] nb1191 te   mean/std   = "
          f"{init_te_513.mean():.4f} / {init_te_513.std():.4f}")

    # ---- Load X_117 substrate + K=20 cols (held fixed vs nb2831) ----
    if not X117_UNB_PATH.exists() or not X117_TE_PATH.exists():
        raise FileNotFoundError(
            f"X_117 cache missing: {X117_UNB_PATH} / {X117_TE_PATH}"
        )
    X117_unb = np.load(X117_UNB_PATH).astype(np.float32)
    X117_te = np.load(X117_TE_PATH).astype(np.float32)
    if X117_unb.shape != (n_unb, 117):
        raise ValueError(f"X117_unb shape {X117_unb.shape}")
    if X117_te.shape != (n_test, 117):
        raise ValueError(f"X117_te shape {X117_te.shape}")
    X117_unb = np.where(np.isfinite(X117_unb), X117_unb, 0.0).astype(np.float32)
    X117_te = np.where(np.isfinite(X117_te), X117_te, 0.0).astype(np.float32)

    if not NB2240_SUMMARY.exists():
        raise FileNotFoundError(f"nb2240 summary missing: {NB2240_SUMMARY}")
    with open(NB2240_SUMMARY) as f:
        nb2240 = json.load(f)
    k20_idx = list(nb2240["k20_surviving_idx_in_117"])
    k20_names = list(nb2240["k20_surviving_names"])
    assert len(k20_idx) == 20, f"expected 20 K=20 indices, got {len(k20_idx)}"

    X_unb = X117_unb[:, k20_idx].astype(np.float32)
    X_te = X117_te[:, k20_idx].astype(np.float32)
    print(f"[feat] X_unb (K=20) = {X_unb.shape}  X_te = {X_te.shape}")

    # ---- Scaffold keys for CV ----
    unb_smiles = [te_smiles[i] for i in unb_idx]
    raw_scaffolds = _scaffold_keys_for_unb(unb_smiles)
    n_unique_scaf = len(set(raw_scaffolds))
    n_singletons = sum(1 for k in raw_scaffolds if k.startswith("__singleton_"))
    print(f"[scaffold] unique scaffolds = {n_unique_scaf}  "
          f"singletons = {n_singletons}")

    # ---- 5-seed warm-start CV bag ----
    print("\n" + "-" * 78)
    print(f"5-SEED WARM-START SCAFFOLD CV   kf_seeds={KF_SEEDS}")
    print("-" * 78)
    per_seed_oof = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)
    per_seed_records: list[dict] = []
    per_seed_rae: list[float] = []
    for i, kf_seed in enumerate(KF_SEEDS):
        ts = time.time()
        oof_s, fold_raes = _warm_cv_one_seed(
            X_unb, y_unb, init_oof, raw_scaffolds, kf_seed,
        )
        per_seed_oof[i] = oof_s
        pooled = float(rae(y_unb, oof_s))
        per_seed_rae.append(pooled)
        per_seed_records.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": pooled,
            "per_fold_rae": [float(r) for r in fold_raes],
            "per_fold_rae_mean": float(np.mean(fold_raes)),
            "per_fold_rae_std": float(np.std(fold_raes)),
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   seed={kf_seed}:  pooled_rae = {pooled:.4f}  "
              f"per_fold_mean = {np.mean(fold_raes):.4f} +/- "
              f"{np.std(fold_raes):.4f}  "
              f"wall = {time.time() - ts:.1f}s")

    # Mean-bag OOF across seeds
    mean_bag_oof = per_seed_oof.mean(axis=0)
    mean_rae = float(np.mean(per_seed_rae))     # decision metric: mean over seeds
    std_rae = float(np.std(per_seed_rae))
    pooled_mean_bag_rae = float(rae(y_unb, mean_bag_oof))
    print(f"\n[5-seed] mean_rae over seeds   = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"[5-seed] pooled mean-bag RAE   = {pooled_mean_bag_rae:.4f}")
    print(f"[5-seed] delta vs nb1191 init  = "
          f"{mean_rae - rae_init:+.4f}  (negative = warm-start helps)")
    print(f"[5-seed] delta vs nb2171 ref   = {mean_rae - NB2171_REF:+.4f}")
    print(f"[5-seed] delta vs nb1191 deep30= {mean_rae - NB1191_DEEP30_REF:+.4f}")

    # ---- Deploy refit on full 253 ----
    print("\n" + "-" * 78)
    print("DEPLOY REFIT  (warm-start on full 253; predict 513 te)")
    print("-" * 78)
    per_seed_te = np.zeros((len(KF_SEEDS), n_test), dtype=np.float64)
    for i, kf_seed in enumerate(KF_SEEDS):
        te_pred = _deploy_refit(
            X_unb, y_unb, init_oof, X_te, init_te_513, kf_seed,
        )
        per_seed_te[i] = te_pred.astype(np.float64)
    te_deploy = per_seed_te.mean(axis=0).astype(np.float32)
    te_unb_in_sample_rae = float(rae(y_unb, te_deploy[unb_idx]))
    print(f"[deploy] te(513) mean/std       = "
          f"{te_deploy.mean():.3f}/{te_deploy.std():.3f}")
    print(f"[deploy] te[unb_idx] in-sample  = {te_unb_in_sample_rae:.4f}  "
          f"(refit on full 253, in-sample optimism expected)")

    # ---- Save artefacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, mean_bag_oof.astype(np.float32))
    np.save(te_path, te_deploy)
    print(f"\n[save] {oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_lgbm_init_from_nb1191.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": te_deploy.astype(np.float32),
    }).to_csv(sub_csv, index=False)
    print(f"[save] {sub_csv}")

    # ---- Gate ----
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "=" * 78)
    print("GATE EVALUATION")
    print("=" * 78)
    print(f"   mean_rae over 5 kf_seeds = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   < {GATE_PROMOTE:.4f}  (PROMOTE)       = {mean_rae < GATE_PROMOTE}")
    print(f"   < {GATE_MARGINAL:.4f}  (MARGINAL_BEAT) = "
          f"{mean_rae < GATE_MARGINAL}")
    print(f"   VERDICT                   = {verdict}")

    summary = {
        "tag": TAG,
        "method": "lgbm_warm_start_init_score_nb1191_pred_full_pec50_target",
        "rationale": (
            "Anchor-axis swap of nb2831 paradigm: init_score swapped from "
            "nb2240 (residual-stack on chemprop_aux) to nb1191 (PRE-unblind "
            "pyramid, deep-30 0.4718).  Cycle-167 nb2171 swap finding "
            "(nb730 -> nb1191 broke 0.4720 ceiling to 0.4682) showed nb1191 "
            "carries independent information not in the chemprop_aux + "
            "residual chain.  Target = full pEC50 (not residual); LGBM "
            "trees learn corrections on top of nb1191 prior with small "
            "incremental budget (n_est=100, lr=0.01).  Substrate held "
            "fixed at K=20 (same as nb2831) so only variable is init axis."
        ),
        "init_anchor": ANCHOR,
        "init_oof_path": str(INIT_OOF_PATH),
        "init_te_path": str(INIT_TE_PATH),
        "init_anchor_pre_unblind": True,
        "rae_init_nb1191": rae_init,
        "nb1191_deep30_ref": NB1191_DEEP30_REF,
        "nb2240_K20_ref": NB2240_K20_REF,
        "nb2171_ref": NB2171_REF,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "x117_unb_path": str(X117_UNB_PATH),
        "x117_te_path": str(X117_TE_PATH),
        "k20_idx_source": str(NB2240_SUMMARY),
        "k20_surviving_idx_in_117": [int(j) for j in k20_idx],
        "k20_surviving_names": k20_names,
        "feat_dim": int(X_unb.shape[1]),
        "n_unb": n_unb,
        "n_test": n_test,
        "n_unique_scaffolds": int(n_unique_scaf),
        "n_singleton_scaffolds": int(n_singletons),
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "cv_protocol": (
            "5-fold scaffold_kfold_indices per kf_seed; warm-start "
            "init_score = nb1191_pred_oof per row (fold-train only); "
            "predict adds init_va back to raw boosted score"
        ),
        "model_class": "lightgbm.LGBMRegressor",
        "lgbm_params_sample": _lgbm_params(KF_SEEDS[0]),
        "per_seed_records": per_seed_records,
        "per_seed_rae": per_seed_rae,
        "mean_rae": mean_rae,
        "std_rae": std_rae,
        "pooled_mean_bag_rae": pooled_mean_bag_rae,
        "delta_vs_nb1191_init": mean_rae - rae_init,
        "delta_vs_nb1191_deep30": mean_rae - NB1191_DEEP30_REF,
        "delta_vs_nb2171": mean_rae - NB2171_REF,
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
        "mean_rae",
        "std_rae",
        "pooled_mean_bag_rae",
        "rae_init_nb1191",
        "delta_vs_nb1191_init",
        "delta_vs_nb1191_deep30",
        "delta_vs_nb2171",
        "te_unb_in_sample_rae",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
