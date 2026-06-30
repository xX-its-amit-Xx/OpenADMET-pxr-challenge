"""nb1240 -- Predicted counter-assay (pEC50_null) as a residual feature
appended to MACCS-167 keys for the shallow residual-LGBM bag on the nb1070
anchor.

Hypothesis
----------
The 2,859-row PXR-null counter-assay carries an orthogonal biological signal:
PXR-null binding strength is an off-target/promiscuity proxy that the pEC50
anchor (nb1070, RAE 0.5771) does not fully exploit because the 253 unblind
and 513 test compounds carry NO measured pEC50_null.  If we train a chemistry
predictor of pEC50_null on the 2,647 compounds with BOTH pEC50 and pEC50_null
labels (inner-join on Molecule Name, drop NaN), we can produce a predicted
pEC50_null for every 513 test compound -- including the 253 unblind --
and feed it as a residual-learner feature alongside MACCS-167.

Outer protocol
--------------
1. Inner-join load_train + load_counter on Molecule Name (counter is a subset
   of train).  Drop rows where either pEC50 or pEC50_null is NaN.  Expect
   ~2,647 rows.
2. Featurize those 2,647 compounds with src/pxr/featurize.combined
   (Morgan-2048 + RDKit-217, imputed).
3. Featurize the 513 test compounds with the same `combined` pipeline.
4. Counter-assay predictor: KFold(5, shuffle, random_state=42) LGBM regressor
   on pEC50_null.  Report the cross-val RAE (pooled across folds).
5. Refit on all 2,647 rows; predict pEC50_null for all 513 test compounds.
   Save as `data/processed/pred_pec50_null_513.npy`.
6. Residual learner setup (mirrors nb1183):
     anchor      = nb1070_pred_oof (constant across seeds)
     residual    = y_unb - anchor
     features    = concat[MACCS-166/167 unblind, pred_pec50_null[unb_idx]]
                   shape (253, 168) when MACCS is 167
   For each seed in {0, 1, 7, 42, 137}:
     KFold(5, shuffle, random_state=seed) on 253 unblind rows.
     Shallow LGBM Huber (depth=3, num_leaves=7, n_est=80, lr=0.05,
       min_child_samples=20, alpha=1.0).
     pred_corrected_s = anchor + residual_oof_s.
7. Mean-bag pooled cross-fit RAE = RAE(y_unb, mean over seeds).
8. Compare to nb1183 (MACCS-only mean-bag 0.5513) and nb1211
   (nb1190 + nb1200 BoB blend, 0.5451) at the 0.003 decision margin.

Outputs
-------
  data/processed/pred_pec50_null_513.npy             (513,)   float32
  data/processed/nb1240_per_seed_corrected_oof.npy   (5, 253) float32
  data/processed/nb1240_mean_bag_oof.npy             (253,)   float32
  data/processed/nb1240_median_bag_oof.npy           (253,)   float32
  data/processed/nb1240_summary.json
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
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor

from pxr.data import load_train, load_counter, load_test
from pxr.eval import rae
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED

TAG = "nb1240"
ANCHOR = "nb1070"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

# Counter-assay predictor settings.
COUNTER_FOLDS = 5
COUNTER_SEED = 42
COUNTER_N_EST = 500
COUNTER_LR = 0.05
COUNTER_NUM_LEAVES = 64

# Reference numbers (pooled RAE on 253 unblind).
NB1070_REF_POOLED = 0.5771
NB1183_MEAN_BAG_REF = 0.5513   # MACCS-only residual bag on nb1070
NB1211_REF = 0.5451            # nb1190+nb1200 BoB blend
DECISION_MARGIN = 0.003

MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"   # (513, 167) uint8


# -----------------------------------------------------------------------------
# Counter-assay predictor.
# -----------------------------------------------------------------------------
def _counter_lgbm_params(seed: int = COUNTER_SEED) -> dict:
    """Capacity-tuned LGBM for pEC50_null on combined features (n~2647, d=2265)."""
    return dict(
        objective="regression",
        learning_rate=COUNTER_LR,
        n_estimators=COUNTER_N_EST,
        num_leaves=COUNTER_NUM_LEAVES,
        min_child_samples=10,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        verbosity=-1,
        random_state=seed,
        n_jobs=2,
    )


def _build_counter_dataset() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Inner-join train + counter on Molecule Name, drop NaN labels.

    Returns
    -------
    df       : DataFrame with columns ['name', 'smiles', 'pec50_null']
    smiles   : list[str]    (n_co,)
    y_null   : np.ndarray   (n_co,) pEC50_null targets
    """
    tr = load_train()[["name", "smiles", "pec50"]].rename(
        columns={"pec50": "pec50_pxr"}
    )
    co = load_counter()[["name", "pec50"]].rename(columns={"pec50": "pec50_null"})
    joined = tr.merge(co, on="name", how="inner")
    df = joined.dropna(subset=["pec50_pxr", "pec50_null"]).reset_index(drop=True)
    return df, df["smiles"].tolist(), df["pec50_null"].to_numpy(dtype=np.float64)


def _counter_cross_val_rae(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """5-fold KFold cross-val of LGBM regressor on pEC50_null.  Returns
    (pooled RAE, per-fold validation predictions stitched back to order).
    """
    n = len(y)
    kf = KFold(n_splits=COUNTER_FOLDS, shuffle=True, random_state=COUNTER_SEED)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = LGBMRegressor(**_counter_lgbm_params(COUNTER_SEED))
        mdl.fit(X[tr_loc], y[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return float(rae(y, oof)), oof


# -----------------------------------------------------------------------------
# Residual learner primitives (mirrors nb1183).
# -----------------------------------------------------------------------------
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


def _residual_cross_fit_one_seed(
    X: np.ndarray, residual: np.ndarray, seed: int
) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = LGBMRegressor(**_resid_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _load_maccs_unblind(n_test_expected: int, unb_idx: np.ndarray) -> np.ndarray:
    if not MACCS_TE_PATH.exists():
        raise FileNotFoundError(f"MACCS test cache missing: {MACCS_TE_PATH}")
    X_te = np.load(MACCS_TE_PATH)
    if X_te.shape[0] != n_test_expected:
        raise ValueError(
            f"MACCS test cache shape mismatch: {X_te.shape} vs n_test={n_test_expected}"
        )
    if X_te.shape[1] not in (166, 167):
        raise ValueError(
            f"MACCS test cache unexpected width: {X_te.shape[1]} (expected 166 or 167)"
        )
    return X_te[unb_idx].astype(np.float32)


# -----------------------------------------------------------------------------
# Main.
# -----------------------------------------------------------------------------
def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Predicted counter-assay (pEC50_null) as residual feature")
    print(f"          appended to MACCS-167 for shallow residual-LGBM bag on {ANCHOR}")
    print(f"          residual seeds = {RESID_SEEDS}")
    print(f"          counter-pred LGBM: n_est={COUNTER_N_EST}, "
          f"num_leaves={COUNTER_NUM_LEAVES}, lr={COUNTER_LR}, seed={COUNTER_SEED}")
    print(f"          residual LGBM: depth=3, leaves=7, n_est=80, lr=0.05, "
          f"huber(alpha=1.0)")
    print("=" * 78)

    # ---- 1/2: Build counter-assay dataset + features ----
    print("\n[1] Building counter-assay dataset (inner-join train + counter) ...")
    co_df, co_smiles, y_null = _build_counter_dataset()
    n_co = len(co_df)
    print(f"    counter-assay (joined, both labels non-NaN) rows = {n_co}")
    print(f"    pEC50_null:  mean={y_null.mean():.3f}  std={y_null.std():.3f}  "
          f"min={y_null.min():.3f}  max={y_null.max():.3f}")

    print("\n[2] Featurizing counter-assay compounds (combined: Morgan + RDKit) ...")
    t_feat = time.time()
    X_co = impute(combined(co_smiles))
    print(f"    X_co shape = {X_co.shape}  ({time.time() - t_feat:.1f}s)")

    # ---- 3: Counter-assay cross-val ----
    print("\n[3] Counter-assay LGBM 5-fold cross-val RAE on training set ...")
    t_cv = time.time()
    counter_cv_rae, counter_oof = _counter_cross_val_rae(X_co, y_null)
    print(f"    counter cross-val pooled RAE = {counter_cv_rae:.4f}  "
          f"({time.time() - t_cv:.1f}s)")
    counter_oof_resid_std = float((y_null - counter_oof).std())
    print(f"    counter cross-val resid std  = {counter_oof_resid_std:.4f}")

    # ---- 4: Refit on full 2,647 -> predict pEC50_null for 513 test ----
    print("\n[4] Refitting counter-assay predictor on full set; "
          "predicting pEC50_null for 513 test ...")
    te_df = load_test()
    n_test = len(te_df)
    t_te = time.time()
    X_te = impute(combined(te_df["smiles"].tolist()))
    print(f"    X_te shape = {X_te.shape}  ({time.time() - t_te:.1f}s)")

    final_counter_mdl = LGBMRegressor(**_counter_lgbm_params(COUNTER_SEED))
    final_counter_mdl.fit(X_co, y_null)
    pred_pec50_null_513 = final_counter_mdl.predict(X_te).astype(np.float64)
    pp_out_path = DATA_PROCESSED / "pred_pec50_null_513.npy"
    np.save(pp_out_path, pred_pec50_null_513.astype(np.float32))
    print(f"    pred_pec50_null_513: mean={pred_pec50_null_513.mean():.3f}  "
          f"std={pred_pec50_null_513.std():.3f}  "
          f"min={pred_pec50_null_513.min():.3f}  "
          f"max={pred_pec50_null_513.max():.3f}")
    print(f"    [save] {pp_out_path}")

    # ---- 5/6/7: Residual learner setup ----
    print("\n[5] Loading anchor + unblind + MACCS ...")
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"    n_test={n_test}  n_unb={n_unb}")

    anchor_path = DATA_PROCESSED / f"{ANCHOR}_pred_oof.npy"
    if not anchor_path.exists():
        raise FileNotFoundError(f"{anchor_path} missing; run nb1070 first.")
    anchor_oof = np.load(anchor_path).astype(np.float64)
    if anchor_oof.shape[0] != n_unb:
        raise ValueError(
            f"{anchor_path} shape mismatch: {anchor_oof.shape} vs n_unb={n_unb}"
        )
    rae_anchor = float(rae(y_unb, anchor_oof))
    print(f"    {ANCHOR}_pred_oof: pooled RAE = {rae_anchor:.4f}  "
          f"(ref ~{NB1070_REF_POOLED:.4f})")

    residual = y_unb - anchor_oof
    print(f"    residual: mean={residual.mean():+.4f}  std={residual.std():.4f}")

    X_maccs_unb = _load_maccs_unblind(n_test_expected=n_test, unb_idx=unb_idx)
    pred_null_unb = pred_pec50_null_513[unb_idx].astype(np.float32).reshape(-1, 1)
    X_unb = np.hstack([X_maccs_unb, pred_null_unb])
    print(f"    X_unb shape = {X_unb.shape}  "
          f"(MACCS {X_maccs_unb.shape[1]} + predicted pEC50_null 1)")
    print(f"    pred_null_unb: mean={pred_null_unb.mean():.3f}  "
          f"std={pred_null_unb.std():.3f}  "
          f"min={pred_null_unb.min():.3f}  max={pred_null_unb.max():.3f}")

    # ---- 6: per-seed residual cross-fit ----
    print("\n" + "-" * 78)
    print(f"PER-SEED RESIDUAL CROSS-FIT (depth=3 shallow, "
          f"{X_unb.shape[1]} features)")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []
    for i, s in enumerate(RESID_SEEDS):
        resid_oof_s = _residual_cross_fit_one_seed(X_unb, residual, s)
        pred_corr_s = anchor_oof + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        delta_s = rae_s - rae_anchor
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_nb1070": delta_s,
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
        })
        print(f"   seed {s:3d}:  rae_corr = {rae_s:.4f}  "
              f"(d_vs_nb1070 = {delta_s:+.4f})  "
              f"|resid_oof|.std = {resid_oof_s.std():.3f}")

    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))

    per_seed_rae_arr = np.array(per_seed_rae)
    rae_per_seed_mean = float(per_seed_rae_arr.mean())
    rae_per_seed_median = float(np.median(per_seed_rae_arr))
    rae_per_seed_std = float(per_seed_rae_arr.std())
    rae_per_seed_min = float(per_seed_rae_arr.min())
    rae_per_seed_max = float(per_seed_rae_arr.max())

    print("\n" + "-" * 78)
    print("BAG AGGREGATIONS")
    print("-" * 78)
    print(f"   per-seed RAE list      = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   per-seed mean          = {rae_per_seed_mean:.4f}")
    print(f"   per-seed median        = {rae_per_seed_median:.4f}")
    print(f"   per-seed std           = {rae_per_seed_std:.4f}")
    print(f"   per-seed min/max       = "
          f"{rae_per_seed_min:.4f} / {rae_per_seed_max:.4f}")
    print(f"   pooled RAE(mean_bag)   = {rae_mean_bag:.4f}  "
          f"(d_vs_nb1070 = {rae_mean_bag - rae_anchor:+.4f})")
    print(f"   pooled RAE(median_bag) = {rae_median_bag:.4f}  "
          f"(d_vs_nb1070 = {rae_median_bag - rae_anchor:+.4f})")
    print(f"   nb1183 mean_bag ref    = {NB1183_MEAN_BAG_REF:.4f}  "
          f"(MACCS-only residual on {ANCHOR})")
    print(f"   nb1211 BoB ref         = {NB1211_REF:.4f}")

    # ---- 8: Verdict ----
    beats_nb1070 = rae_mean_bag < rae_anchor - DECISION_MARGIN
    beats_nb1183 = rae_mean_bag < NB1183_MEAN_BAG_REF - DECISION_MARGIN
    beats_nb1211 = rae_mean_bag < NB1211_REF - DECISION_MARGIN

    if beats_nb1211:
        verdict = "PRED_COUNTER_RESIDUAL_BEATS_NB1211_BOB_NEW_PRIMARY"
    elif beats_nb1183:
        verdict = "PRED_COUNTER_RESIDUAL_BEATS_NB1183_BUT_NOT_NB1211"
    elif beats_nb1070:
        verdict = "PRED_COUNTER_RESIDUAL_HELPS_NB1070_BUT_NOT_NB1183"
    elif abs(rae_mean_bag - NB1183_MEAN_BAG_REF) < DECISION_MARGIN:
        verdict = "PRED_COUNTER_RESIDUAL_FLAT_VS_NB1183_NO_NEW_SIGNAL"
    else:
        verdict = "PRED_COUNTER_RESIDUAL_HURTS_VS_NB1183"
    print(f"   verdict                = {verdict}")

    # ---- Save artefacts ----
    np.save(DATA_PROCESSED / f"{TAG}_per_seed_corrected_oof.npy",
            per_seed_corrected.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy",
            mean_bag_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_median_bag_oof.npy",
            median_bag_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_per_seed_corrected_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_mean_bag_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_median_bag_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "feature_source": "maccs_167_plus_pred_pec50_null",
        "maccs_cache_test": str(MACCS_TE_PATH),
        "pred_pec50_null_path": str(pp_out_path),
        "n_counter_rows_joined": n_co,
        "counter_folds": COUNTER_FOLDS,
        "counter_seed": COUNTER_SEED,
        "counter_lgbm_n_estimators": COUNTER_N_EST,
        "counter_lgbm_num_leaves": COUNTER_NUM_LEAVES,
        "counter_lgbm_learning_rate": COUNTER_LR,
        "counter_cv_pooled_rae": counter_cv_rae,
        "counter_cv_resid_std": counter_oof_resid_std,
        "counter_train_mean_y": float(y_null.mean()),
        "counter_train_std_y": float(y_null.std()),
        "pred_pec50_null_513_mean": float(pred_pec50_null_513.mean()),
        "pred_pec50_null_513_std": float(pred_pec50_null_513.std()),
        "pred_pec50_null_513_min": float(pred_pec50_null_513.min()),
        "pred_pec50_null_513_max": float(pred_pec50_null_513.max()),
        "pred_pec50_null_unb_mean": float(pred_null_unb.mean()),
        "pred_pec50_null_unb_std": float(pred_null_unb.std()),
        "pred_pec50_null_unb_min": float(pred_null_unb.min()),
        "pred_pec50_null_unb_max": float(pred_null_unb.max()),
        "n_unb": n_unb,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "feature_dim": int(X_unb.shape[1]),
        "maccs_dim": int(X_maccs_unb.shape[1]),
        "resid_lgbm_max_depth": 3,
        "resid_lgbm_num_leaves": 7,
        "resid_lgbm_n_estimators": 80,
        "resid_lgbm_learning_rate": 0.05,
        "resid_lgbm_min_child_samples": 20,
        "resid_lgbm_huber_alpha": 1.0,
        "rae_anchor_nb1070": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_seed_rae": per_seed_rae,
        "per_seed_records": per_seed_records,
        "rae_per_seed_mean": rae_per_seed_mean,
        "rae_per_seed_median": rae_per_seed_median,
        "rae_per_seed_std": rae_per_seed_std,
        "rae_per_seed_min": rae_per_seed_min,
        "rae_per_seed_max": rae_per_seed_max,
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_nb1070": rae_mean_bag - rae_anchor,
        "delta_median_bag_vs_nb1070": rae_median_bag - rae_anchor,
        "delta_mean_bag_vs_nb1183": rae_mean_bag - NB1183_MEAN_BAG_REF,
        "delta_mean_bag_vs_nb1211": rae_mean_bag - NB1211_REF,
        "beats_nb1070": bool(beats_nb1070),
        "beats_nb1183": bool(beats_nb1183),
        "beats_nb1211": bool(beats_nb1211),
        "verdict": verdict,
        "nb1070_ref_pooled": NB1070_REF_POOLED,
        "nb1183_mean_bag_ref": NB1183_MEAN_BAG_REF,
        "nb1211_ref": NB1211_REF,
        "decision_margin": DECISION_MARGIN,
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
        "counter_cv_pooled_rae",
        "counter_cv_resid_std",
        "pred_pec50_null_513_mean",
        "pred_pec50_null_513_std",
        "pred_pec50_null_513_min",
        "pred_pec50_null_513_max",
        "rae_anchor_nb1070",
        "per_seed_rae",
        "rae_per_seed_mean",
        "rae_per_seed_median",
        "rae_per_seed_std",
        "rae_mean_bag",
        "rae_median_bag",
        "delta_mean_bag_vs_nb1070",
        "delta_mean_bag_vs_nb1183",
        "delta_mean_bag_vs_nb1211",
        "beats_nb1070",
        "beats_nb1183",
        "beats_nb1211",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
