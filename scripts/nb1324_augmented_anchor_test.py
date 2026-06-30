"""nb1324 -- Augmented-train anchor refit shift test.

QUESTION
--------
Memory feedback_unblind_augmentation logs that nb590-593 (augmenting 4139
train with the 253 unblind labels) failed to break the nb562 floor (0.5099)
because that 6.1% extra coverage hit already-covered scaffolds.  But the
ANCHOR there was a post-hoc stretch on chemprop_aux -- the augmentation
never touched the model that produced the pred itself.

This notebook tests whether augmenting the ANCHOR LGBM training data with
the 253 unblind labels breaks the current 0.5390 floor (nb1290 fixed-w
blend of nb1190 BoB + nb1242 ChEMBL).  Specifically we replace the
"chemprop_aux/nb1070-equivalent" anchor with a deep LGBM-Huber trained on
combined Morgan+RDKit features, cross-fit so each held-out 1/5 of unblind
is predicted by a model trained on (4139 + 4/5 of unblind).  We then run
the nb1290-style residual stack (MACCS-167 + ChEMBL kNN feature + sim,
shallow LGBM Huber, 5-seed bag) on top.

PROTOCOL
--------
  Stage A -- augmented anchor (5-fold cross-fit):
    Features X_combined = Morgan-2048 + RDKit-desc (~2265)
    For each fold f (KFold seed=42, n_splits=5) on the 253 unblind:
        train_idx = ALL 4139 train  CONCAT  (4/5 unblind not in fold f)
        held = 1/5 unblind in fold f
        For each inner seed s in [0, 1, 7, 42, 137]:
            Fit LGBM-Huber(alpha=1.0) deep params (nb120-style):
              n_estimators=1500, num_leaves=64, learning_rate=0.03,
              min_child_samples=8, subsample=0.8, colsample_bytree=0.8,
              reg_alpha=0.05, reg_lambda=0.1, random_state=s.
            Predict held -> oof_s[f]
        oof_aug[f] = mean over s of oof_s[f]
    aug_anchor_oof_253 = concat over folds
    rae(y_unb, aug_anchor_oof_253) -> aug_anchor_rae.
    Compare vs nb1070 reference (0.5771).

  Stage B -- residual learner (nb1290-style):
    residual = y_unb - aug_anchor_oof
    features = MACCS-167(unb) + pred_chembl_pec50(unb) + sim_chembl(unb)
               -> (253, 169)
    For each seed s in [0,1,7,42,137]:
        5-fold KFold(seed=s) on the 253 unblind:
            shallow LGBM Huber (nb1242 params: depth=3, leaves=7,
                                n_est=80, lr=0.05, alpha=1.0)
            fit residual[tr] -> predict resid_oof[va]
        bag_s = aug_anchor_oof + resid_oof_s
    mean_bag = mean over seeds.
    rae(y_unb, mean_bag) -> final_residual_rae.

  Verdict thresholds (0.003 margin):
    BREAKS_FLOOR  : final_residual_rae < nb1290 floor (0.5390) - 0.003
    FLAT_VS_FLOOR : |final - 0.5390| < 0.003
    WORSE_THAN_FLOOR else

OUTPUTS
-------
  scripts/nb1324_augmented_anchor_test.py        (this file)
  data/processed/nb1324_aug_anchor_oof.npy       (253,) float32
  data/processed/nb1324_final_oof.npy            (253,) float32
  data/processed/nb1324_summary.json
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor

from pxr.data import load_train, load_test
from pxr.featurize import combined, impute
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1324"

# ---- Stage A (anchor) hyperparams ---------------------------------------
ANCHOR_OUTER_FOLDS = 5
ANCHOR_OUTER_SEED = 42
ANCHOR_INNER_SEEDS = [0, 1, 7, 42, 137]

ANCHOR_BASE_PARAMS = dict(
    objective="huber",
    alpha=1.0,
    n_estimators=1500,
    num_leaves=64,
    learning_rate=0.03,
    min_child_samples=8,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.05,
    reg_lambda=0.1,
    verbosity=-1,
    n_jobs=4,
)

# ---- Stage B (residual) hyperparams (nb1242 verbatim) -------------------
RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]


def _resid_lgbm_params(seed: int) -> dict:
    return dict(
        objective="huber",
        alpha=1.0,
        learning_rate=0.05,
        n_estimators=80,
        max_depth=3,
        num_leaves=7,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        verbosity=-1,
        random_state=seed,
        n_jobs=2,
    )


# Reference numbers (pooled RAE on 253 unblind).
NB1070_REF = 0.5771    # canonical pre-residual anchor
NB1290_FLOOR = 0.5390  # current best blend floor (nb1290 best fixed-w)
NB1242_REF = 0.5431    # nb1242 standalone residual mean bag
DECISION_MARGIN = 0.003


def stage_a_augmented_anchor(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_unb: np.ndarray, y_unb: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    """Cross-fit anchor where each fold trains on (4139 + 4/5 unblind).

    Returns
    -------
    aug_anchor_oof : (n_unb,) float64
        Pooled out-of-fold prediction for the 253 unblind, with inner-seed
        bagging (mean across ANCHOR_INNER_SEEDS).
    fold_records : list[dict]
        Per-fold diagnostics.
    """
    n_unb = len(y_unb)
    n_tr = len(y_tr)
    aug_oof = np.full(n_unb, np.nan, dtype=np.float64)
    inner_oof_stack = np.full((len(ANCHOR_INNER_SEEDS), n_unb), np.nan,
                              dtype=np.float64)
    fold_records: list[dict] = []

    kf = KFold(n_splits=ANCHOR_OUTER_FOLDS, shuffle=True,
               random_state=ANCHOR_OUTER_SEED)
    for f, (tr_loc_unb, va_loc_unb) in enumerate(kf.split(np.arange(n_unb))):
        t_f = time.time()
        # Build augmented training data: 4139 train + 4/5 unblind held-in.
        X_aug = np.vstack([X_tr, X_unb[tr_loc_unb]])
        y_aug = np.concatenate([y_tr, y_unb[tr_loc_unb]])
        X_va = X_unb[va_loc_unb]
        y_va = y_unb[va_loc_unb]
        n_aug = len(y_aug)

        # Per-inner-seed LGBM fits.
        seed_preds = np.zeros((len(ANCHOR_INNER_SEEDS), len(va_loc_unb)),
                              dtype=np.float64)
        per_seed_va_rae: list[float] = []
        for j, s in enumerate(ANCHOR_INNER_SEEDS):
            params = dict(ANCHOR_BASE_PARAMS, random_state=s)
            mdl = LGBMRegressor(**params)
            mdl.fit(X_aug, y_aug)
            pred_va = mdl.predict(X_va)
            seed_preds[j] = pred_va
            inner_oof_stack[j, va_loc_unb] = pred_va
            per_seed_va_rae.append(float(rae(y_va, pred_va)))

        bagged_va = seed_preds.mean(axis=0)
        aug_oof[va_loc_unb] = bagged_va
        rae_fold = float(rae(y_va, bagged_va))
        fold_records.append({
            "fold": int(f),
            "n_aug_train": int(n_aug),
            "n_va": int(len(va_loc_unb)),
            "rae_fold_bagged": rae_fold,
            "rae_per_inner_seed": per_seed_va_rae,
            "elapsed_sec": round(time.time() - t_f, 2),
        })
        print(f"   fold {f}: n_aug_train={n_aug}  n_va={len(va_loc_unb)}  "
              f"bag_rae={rae_fold:.4f}  "
              f"per_seed=[{', '.join(f'{x:.4f}' for x in per_seed_va_rae)}]  "
              f"({time.time() - t_f:.1f}s)")

    return aug_oof, fold_records, inner_oof_stack


def stage_b_residual_bag(
    anchor_oof: np.ndarray, residual: np.ndarray,
    X_resid: np.ndarray, y_unb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """nb1290/nb1242-style residual learner on top of the augmented anchor."""
    n_unb = len(y_unb)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_records: list[dict] = []
    for j, s in enumerate(RESID_SEEDS):
        t_s = time.time()
        oof = np.full(n_unb, np.nan, dtype=np.float64)
        kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=s)
        for tr_loc, va_loc in kf.split(np.arange(n_unb)):
            mdl = LGBMRegressor(**_resid_lgbm_params(s))
            mdl.fit(X_resid[tr_loc], residual[tr_loc])
            oof[va_loc] = mdl.predict(X_resid[va_loc])
        pred_corr = anchor_oof + oof
        per_seed_corrected[j] = pred_corr
        rae_s = float(rae(y_unb, pred_corr))
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "resid_oof_mean": float(oof.mean()),
            "resid_oof_std": float(oof.std()),
            "elapsed_sec": round(time.time() - t_s, 2),
        })
        print(f"   resid seed {s:5d}: rae_corr = {rae_s:.4f}  "
              f"|resid|.std = {oof.std():.3f}  ({time.time() - t_s:.1f}s)")
    return per_seed_corrected, per_seed_corrected.mean(axis=0), per_seed_records


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- AUGMENTED-TRAIN anchor refit shift test")
    print(f"          Stage A: deep LGBM Huber on (4139 + 4/5 unb) cross-fit")
    print(f"                   inner-seed bag over {ANCHOR_INNER_SEEDS}")
    print(f"          Stage B: nb1290-style MACCS + ChEMBL residual learner")
    print(f"                   shallow LGBM Huber, {len(RESID_SEEDS)}-seed bag")
    print(f"          Floor : nb1290 best fixed-w = {NB1290_FLOOR:.4f}")
    print("=" * 78)

    # ---- Load core ---------------------------------------------------
    tr = load_train()
    te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    n_te = len(te)

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] train={n_tr}  test={n_te}  unblind={n_unb}")

    # ---- Stage A features: combined Morgan + RDKit -------------------
    print("\n[feat] computing combined Morgan+RDKit features ...")
    t_ft = time.time()
    X_tr_full = impute(combined(tr["smiles"].tolist())).astype(np.float32)
    X_te_full = impute(combined(te["smiles"].tolist())).astype(np.float32)
    X_unb_full = X_te_full[unb_idx]
    print(f"   X_tr  : {X_tr_full.shape}")
    print(f"   X_te  : {X_te_full.shape}")
    print(f"   X_unb : {X_unb_full.shape}   ({time.time() - t_ft:.1f}s)")

    # ---- Sanity: nb1070 reference rae on 253 -----------------------
    try:
        nb1070_oof = np.load(DATA_PROCESSED / "nb1070_pred_oof.npy"
                             ).astype(np.float64)
        rae_nb1070 = float(rae(y_unb, nb1070_oof))
        print(f"\n[ref] nb1070_pred_oof in_RAE on 253 = {rae_nb1070:.4f}  "
              f"(ref {NB1070_REF:.4f})")
    except FileNotFoundError:
        rae_nb1070 = None
        print(f"\n[ref] nb1070_pred_oof missing -- skipping anchor sanity")

    # ===================================================================
    # STAGE A -- Augmented anchor cross-fit
    # ===================================================================
    print("\n" + "-" * 78)
    print(f"STAGE A: AUGMENTED ANCHOR CROSS-FIT  "
          f"({ANCHOR_OUTER_FOLDS} outer folds x {len(ANCHOR_INNER_SEEDS)} "
          f"inner seeds = {ANCHOR_OUTER_FOLDS * len(ANCHOR_INNER_SEEDS)} "
          f"LGBM fits)")
    print(f"           LGBM Huber: n_est={ANCHOR_BASE_PARAMS['n_estimators']}  "
          f"leaves={ANCHOR_BASE_PARAMS['num_leaves']}  "
          f"lr={ANCHOR_BASE_PARAMS['learning_rate']}  "
          f"alpha={ANCHOR_BASE_PARAMS['alpha']}")
    print("-" * 78)
    aug_anchor_oof, fold_records, inner_oof_stack = stage_a_augmented_anchor(
        X_tr_full, y_tr, X_unb_full, y_unb,
    )
    aug_anchor_rae = float(rae(y_unb, aug_anchor_oof))
    delta_vs_nb1070 = aug_anchor_rae - NB1070_REF
    print(f"\n[stage A] augmented anchor cross-fit RAE = {aug_anchor_rae:.4f}")
    print(f"          nb1070 reference                 = {NB1070_REF:.4f}")
    print(f"          delta vs nb1070                  = {delta_vs_nb1070:+.4f}")

    # Per-inner-seed standalone (no bag) on the SAME outer cross-fit.
    per_inner_rae = []
    for j, s in enumerate(ANCHOR_INNER_SEEDS):
        r = float(rae(y_unb, inner_oof_stack[j]))
        per_inner_rae.append(r)
        print(f"          inner seed {s:5d} pooled RAE     = {r:.4f}")
    print(f"          inner-seed mean / std            = "
          f"{np.mean(per_inner_rae):.4f}  /  {np.std(per_inner_rae):.4f}")

    np.save(DATA_PROCESSED / f"{TAG}_aug_anchor_oof.npy",
            aug_anchor_oof.astype(np.float32))
    print(f"[save] {DATA_PROCESSED / f'{TAG}_aug_anchor_oof.npy'}")

    # ===================================================================
    # STAGE B -- Residual learner (MACCS + ChEMBL features)
    # ===================================================================
    print("\n" + "-" * 78)
    print(f"STAGE B: RESIDUAL LEARNER  ({len(RESID_SEEDS)} seeds x "
          f"{RESID_FOLDS} folds = {len(RESID_SEEDS) * RESID_FOLDS} LGBM fits)")
    print(f"           features = MACCS-167 + pred_chembl_pec50 + sim (169)")
    print(f"           shallow LGBM Huber: depth=3 leaves=7 n_est=80 lr=0.05")
    print("-" * 78)

    # Load residual features.
    X_maccs_te = np.load(DATA_PROCESSED / "te_maccs.npy")
    if X_maccs_te.shape[0] != n_te:
        raise ValueError(f"te_maccs shape mismatch: {X_maccs_te.shape}")
    X_maccs_unb = X_maccs_te[unb_idx].astype(np.float32)

    pred_chembl_513 = np.load(DATA_PROCESSED / "pred_chembl_pec50_513.npy"
                              ).astype(np.float32)
    sim_chembl_513 = np.load(DATA_PROCESSED / "sim_chembl_513.npy"
                             ).astype(np.float32)
    pred_chembl_unb = pred_chembl_513[unb_idx]
    sim_chembl_unb = sim_chembl_513[unb_idx]

    X_resid = np.concatenate(
        [
            X_maccs_unb,
            pred_chembl_unb.reshape(-1, 1),
            sim_chembl_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    print(f"   X_resid shape = {X_resid.shape}")

    residual = y_unb - aug_anchor_oof
    print(f"   residual: mean={residual.mean():+.4f}  std={residual.std():.4f}")

    per_seed_corrected, mean_bag_oof, per_seed_records = stage_b_residual_bag(
        aug_anchor_oof, residual, X_resid, y_unb,
    )
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    final_residual_rae = float(rae(y_unb, mean_bag_oof))
    final_residual_median_rae = float(rae(y_unb, median_bag_oof))

    np.save(DATA_PROCESSED / f"{TAG}_final_oof.npy",
            mean_bag_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_final_oof.npy'}")

    # ===================================================================
    # VERDICT
    # ===================================================================
    delta_vs_floor = final_residual_rae - NB1290_FLOOR
    beats_floor = final_residual_rae < (NB1290_FLOOR - DECISION_MARGIN)
    flat_vs_floor = abs(delta_vs_floor) < DECISION_MARGIN
    beats_nb1290 = beats_floor  # alias for the user's exact ask

    delta_residual_vs_anchor = final_residual_rae - aug_anchor_rae

    if beats_floor:
        verdict = (f"AUGMENTED_ANCHOR_BREAKS_NB1290_FLOOR "
                   f"(final {final_residual_rae:.4f} vs floor {NB1290_FLOOR:.4f})")
    elif flat_vs_floor:
        verdict = (f"AUGMENTED_ANCHOR_FLAT_VS_NB1290 "
                   f"(final {final_residual_rae:.4f} vs floor {NB1290_FLOOR:.4f})")
    else:
        verdict = (f"AUGMENTED_ANCHOR_HURTS_VS_NB1290 "
                   f"(final {final_residual_rae:.4f} vs floor {NB1290_FLOOR:.4f})")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"   stage A  aug_anchor cross-fit RAE     = {aug_anchor_rae:.4f}  "
          f"(delta vs nb1070 ref {NB1070_REF:.4f}: "
          f"{delta_vs_nb1070:+.4f})")
    print(f"   stage B  residual mean-bag pooled RAE = {final_residual_rae:.4f}")
    print(f"            residual median-bag pooled   = {final_residual_median_rae:.4f}")
    print(f"   nb1290 floor (best fixed-w blend)     = {NB1290_FLOOR:.4f}")
    print(f"   delta vs nb1290 floor                 = {delta_vs_floor:+.4f}")
    print(f"   delta residual vs aug_anchor          = "
          f"{delta_residual_vs_anchor:+.4f}")
    print(f"   beats_nb1290 (margin {DECISION_MARGIN})         = {beats_nb1290}")
    print(f"   verdict                              = {verdict}")

    summary = {
        "tag": TAG,
        "n_train": n_tr,
        "n_unb": n_unb,
        "n_test": n_te,
        "stage_a": {
            "outer_folds": ANCHOR_OUTER_FOLDS,
            "outer_seed": ANCHOR_OUTER_SEED,
            "inner_seeds": ANCHOR_INNER_SEEDS,
            "features": "combined_morgan_rdkit",
            "feature_dim": int(X_tr_full.shape[1]),
            "lgbm_params": ANCHOR_BASE_PARAMS,
            "fold_records": fold_records,
            "per_inner_pooled_rae": per_inner_rae,
            "per_inner_mean_rae": float(np.mean(per_inner_rae)),
            "per_inner_std_rae": float(np.std(per_inner_rae)),
            "aug_anchor_rae": aug_anchor_rae,
            "nb1070_ref": NB1070_REF,
            "delta_vs_nb1070": delta_vs_nb1070,
            "nb1070_oof_rae_sanity": rae_nb1070,
        },
        "stage_b": {
            "resid_folds": RESID_FOLDS,
            "resid_seeds": RESID_SEEDS,
            "feature_dim": int(X_resid.shape[1]),
            "lgbm_params_template": _resid_lgbm_params(0),
            "per_seed_records": per_seed_records,
            "per_seed_rae": [r["rae_corrected"] for r in per_seed_records],
            "residual_mean_bag_rae": final_residual_rae,
            "residual_median_bag_rae": final_residual_median_rae,
            "delta_residual_vs_anchor": delta_residual_vs_anchor,
        },
        "reference_floor_nb1290": NB1290_FLOOR,
        "nb1242_ref": NB1242_REF,
        "decision_margin": DECISION_MARGIN,
        "delta_vs_floor": delta_vs_floor,
        "beats_nb1290": bool(beats_nb1290),
        "flat_vs_nb1290": bool(flat_vs_floor),
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    print(f"  aug_anchor_rae               : {res['stage_a']['aug_anchor_rae']:.4f}")
    print(f"  per_inner_pooled_rae         : "
          f"{[round(r, 4) for r in res['stage_a']['per_inner_pooled_rae']]}")
    print(f"  delta_vs_nb1070              : {res['stage_a']['delta_vs_nb1070']:+.4f}")
    print(f"  residual_mean_bag_rae        : {res['stage_b']['residual_mean_bag_rae']:.4f}")
    print(f"  residual_median_bag_rae      : {res['stage_b']['residual_median_bag_rae']:.4f}")
    print(f"  delta_residual_vs_anchor     : {res['stage_b']['delta_residual_vs_anchor']:+.4f}")
    print(f"  reference_floor_nb1290       : {res['reference_floor_nb1290']:.4f}")
    print(f"  delta_vs_floor               : {res['delta_vs_floor']:+.4f}")
    print(f"  beats_nb1290                 : {res['beats_nb1290']}")
    print(f"  flat_vs_nb1290               : {res['flat_vs_nb1290']}")
    print(f"  verdict                      : {res['verdict']}")
