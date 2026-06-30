"""nb2802 -- LightGBM with categorical encoding on top-50 Morgan bits.

NEW PARADIGM (cycle 170+):
    Every prior LGBM attempt on this anchor-residual substrate has treated
    Morgan FP bits as binary FLOAT features.  LightGBM's tree-builder then
    routes the bit through a numeric split (threshold 0.5), which is
    semantically the same as a "missing vs present" branch but the
    booster still wastes split-budget computing histogram-based thresholds
    and missing-handling for what is intrinsically a 2-state factor.

    This script instead declares the same top-50 Morgan bits as PANDAS
    CATEGORICAL (int8 dtype) columns and lets LightGBM apply its
    categorical-feature optimization:
      * dedicated categorical-split routine (Fisher-style partitioning
        of category sets) rather than numeric threshold search,
      * no missing-value branch wasted (binary -> 2 categories only),
      * cleaner per-bit gain accounting that does not commingle the
        "present" and "absent" branches with downstream continuous splits.

    On a 2-state factor the Fisher partition reduces to the same two-way
    split a numeric threshold would produce, BUT the booster's per-iter
    feature-importance and the way bits interact in subsequent splits
    differs from the numeric-encoded version, giving a distinct
    inductive bias for the boosting trajectory.

    Distinct from prior LGBM-on-Morgan attempts:
      - nb2762 Cauchy loss   : loss-shape change, numeric bits
      - nb2754 focal loss    : loss-shape change, numeric bits
      - nb2774 symmetric LGBM: hyperparameter change, numeric bits
      - nb2782 hessian init  : initialization change, numeric bits
      - nb2790 ExtraTrees LGBM: tree-builder change, numeric bits
      All of the above pass Morgan bits as float to LightGBM and let the
      booster treat them as ordered numeric features.

PROTOCOL (per spec):
    1. Compute Morgan FP-2048 on 513 test SMILES and on the 253 unblind
       slice (te_smiles[unb_idx]).
    2. Rank all 2048 bits by population frequency on the 513 test rows
       (descending bit-sum); take the top-50 bits.
    3. Build feature matrices X_unb_50 / X_te_50 with INT8 dtype and
       declare them pandas Categorical via categorical_feature='auto'
       (the dtype itself signals categorical to LightGBM).
    4. Anchor: chemprop_aux (PRE-unblind, verified clean).
       Residual target r = y_unb - anchor_unb.
    5. Model: LGBMRegressor(max_depth=4, num_leaves=15, n_estimators=300,
                            learning_rate=0.03,
                            categorical_feature='auto').
    6. 5-fold scaffold CV across 5 kf_seeds {1001..1005}.
       Per fold: fit on residual, predict held-out OOF residual.
       Per seed: mean-bag the deploy refit (fit-on-all-253 -> predict-513).
    7. Mean-bag corrected OOF RAE on 253.

GATE:
    mean_rae < 0.4570  -> "PROMOTE"
    mean_rae < 0.4598  -> "MARGINAL_BEAT"
    else               -> "FAIL"

Outputs:
    scripts/nb2802_lgbm_bit_categorical.py
    data/processed/nb2802_summary.json
    data/processed/nb2802_pred_oof.npy   (253,) float32  mean-bag corrected OOF
    data/processed/te_nb2802.npy         (513,) float32  mean-bag deploy
    submissions/nb2802_lgbm_bit_categorical.csv

References (PRE-clean ceiling band):
    nb2171 PRIMARY-1 deep-30 = 0.4682
    nb2095 deep-30 PRE       = 0.4720
    nb1191 deep-30 PRE       = 0.4718
    chemprop_aux unb         = 0.6216 (anchor RAE)
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
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2802"

# ---- Anchor + indices ----
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
UNBLIND_IDX = DATA_PROCESSED / "_audit_unblind_idx.npy"
UNBLIND_Y = DATA_PROCESSED / "_audit_unblind_y.npy"

# ---- Morgan FP config ----
MORGAN_BITS = 2048
MORGAN_RADIUS = 2
TOP_K_BITS = 50

# ---- CV protocol ----
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# ---- LGBM hyperparams (per task spec) ----
LGBM_MAX_DEPTH = 4
LGBM_NUM_LEAVES = 15
LGBM_N_ESTIMATORS = 300
LGBM_LEARNING_RATE = 0.03

# ---- Gates ----
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

# ---- Refs ----
CHEMPROP_AUX_REF = 0.6216
NB2171_REF = 0.4682


def _lgbm_params(seed: int) -> dict:
    """LGBM hyperparams; categorical_feature='auto' picks up pandas Categorical dtype."""
    return dict(
        objective="regression",
        max_depth=LGBM_MAX_DEPTH,
        num_leaves=LGBM_NUM_LEAVES,
        n_estimators=LGBM_N_ESTIMATORS,
        learning_rate=LGBM_LEARNING_RATE,
        random_state=int(seed),
        n_jobs=2,
        verbosity=-1,
    )


def _to_categorical_df(X_int8: np.ndarray, col_names: list[str]) -> pd.DataFrame:
    """Convert int8 binary matrix to DataFrame with pandas categorical dtype.

    LightGBM treats pandas Categorical columns as categorical features when
    `categorical_feature='auto'` is passed (default).  This activates the
    Fisher-style categorical-split routine instead of numeric threshold search.
    """
    df = pd.DataFrame(X_int8, columns=col_names)
    for c in col_names:
        df[c] = df[c].astype("category")
    return df


def _residual_cv_one_seed(
    X_unb_df: pd.DataFrame,
    residual: np.ndarray,
    scaffolds: list,
    seed: int,
):
    """5-fold scaffold-CV residual LGBM with categorical Morgan bits."""
    n = len(residual)
    splits = scaffold_kfold_indices(
        scaffolds, n_splits=N_FOLDS, shuffle=True, seed=seed,
    )
    oof = np.full(n, np.nan, dtype=np.float64)
    per_fold_rae = []
    for fi, (tr_loc, va_loc) in enumerate(splits):
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed + fi))
        mdl.fit(
            X_unb_df.iloc[tr_loc], residual[tr_loc],
            categorical_feature="auto",
        )
        oof[va_loc] = mdl.predict(X_unb_df.iloc[va_loc])
        per_fold_rae.append(float(rae(residual[va_loc], oof[va_loc])))
    if np.isnan(oof).any():
        raise RuntimeError("scaffold split did not cover all rows")
    return oof, per_fold_rae


def _deploy_refit(
    X_unb_df: pd.DataFrame,
    residual: np.ndarray,
    X_te_df: pd.DataFrame,
    seed: int,
) -> np.ndarray:
    mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
    mdl.fit(X_unb_df, residual, categorical_feature="auto")
    return mdl.predict(X_te_df).astype(np.float32)


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- LGBM w/ pandas-categorical Morgan bits (top-{TOP_K_BITS})")
    print(f"          LGBM: depth={LGBM_MAX_DEPTH}  leaves={LGBM_NUM_LEAVES}  "
          f"n_est={LGBM_N_ESTIMATORS}  lr={LGBM_LEARNING_RATE}")
    print(f"          categorical_feature='auto'  (Fisher partition split routine)")
    print(f"          Morgan: {MORGAN_BITS} bits, radius {MORGAN_RADIUS}, "
          f"top-{TOP_K_BITS} by 513-row population freq")
    print(f"          5-fold scaffold CV  kf_seeds={KF_SEEDS}")
    print(f"          GATE: <{GATE_PROMOTE} PROMOTE / "
          f"<{GATE_MARGINAL} MARGINAL_BEAT / else FAIL")
    print("=" * 78)

    # ---- Load test set ----
    te = load_test()
    n_test = len(te)
    smi_col = "smiles" if "smiles" in te.columns else "SMILES"
    name_col = "name" if "name" in te.columns else "Molecule Name"
    te_smiles = te[smi_col].astype(str).tolist()
    te_names = te[name_col].values

    unb_idx = np.load(UNBLIND_IDX)
    y_unb = np.load(UNBLIND_Y).astype(np.float64)
    n_unb = len(y_unb)
    print(f"\n[load] n_test={n_test}  n_unb={n_unb}")

    # ---- Step 1: Morgan FP on 513 test rows; pick top-50 by frequency ----
    print(f"\n[step1] Morgan FP-{MORGAN_BITS} radius {MORGAN_RADIUS} on 513 test ...")
    fp_te_full = morgan_fp_batch(te_smiles, radius=MORGAN_RADIUS, n_bits=MORGAN_BITS)
    print(f"   fp_te_full {fp_te_full.shape}  density={fp_te_full.mean():.4f}")
    # CAST TO int64 BEFORE NEGATION: fp is uint8 -> sum returns uint64.
    # `-uint64` underflows, so we cast first to keep the negation meaningful.
    bit_freq_te = fp_te_full.sum(axis=0).astype(np.int64)  # (2048,) bit counts
    # Sort descending by frequency; tie-break by ascending bit index for stability.
    order = np.argsort(-bit_freq_te, kind="stable")
    top_bit_idx = order[:TOP_K_BITS].astype(int)
    top_bit_freq = bit_freq_te[top_bit_idx]
    print(f"   top-{TOP_K_BITS} bits  freq_max={int(top_bit_freq.max())}  "
          f"freq_min={int(top_bit_freq.min())}  "
          f"mean_density={top_bit_freq.mean() / n_test:.3f}")

    # ---- Step 2: slice 50 cols, cast to int8, wrap in pandas Categorical ----
    X_te_50 = fp_te_full[:, top_bit_idx].astype(np.int8)
    X_unb_50 = X_te_50[unb_idx]
    col_names = [f"bit_{int(b)}" for b in top_bit_idx]
    X_te_df = _to_categorical_df(X_te_50, col_names)
    X_unb_df = _to_categorical_df(X_unb_50, col_names)
    print(f"   X_unb_df={X_unb_df.shape}  dtype={X_unb_df.dtypes.iloc[0]}  "
          f"X_te_df={X_te_df.shape}")
    cat_cols = [c for c in X_unb_df.columns
                if str(X_unb_df[c].dtype) == "category"]
    print(f"   n_categorical_cols={len(cat_cols)} / {TOP_K_BITS}")

    # ---- Step 3: anchor + residual ----
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"\n[anchor] chemprop_aux te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] r mean={residual.mean():+.3f}  std={residual.std():.3f}  "
          f"min={residual.min():+.2f}  max={residual.max():+.2f}")

    # ---- Step 4: scaffold groups for the 253 ----
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique 253-row scaffolds = {n_unique_scaf}")

    # ---- Step 5: 5-seed bag ----
    per_seed_oof_corr = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)
    per_seed_te_resid = np.zeros((len(KF_SEEDS), n_test), dtype=np.float64)
    per_seed_oof_rae = []
    per_seed_resid_rae = []
    per_seed_per_fold_rae = []
    print("\n" + "-" * 78)
    print("5-seed scaffold-CV bag (categorical_feature='auto')")
    print("-" * 78)
    for i, seed in enumerate(KF_SEEDS):
        ts = time.time()
        oof_resid, per_fold_resid_rae = _residual_cv_one_seed(
            X_unb_df, residual, unb_scaffolds, seed,
        )
        corr_oof = anchor + oof_resid
        per_seed_oof_corr[i] = corr_oof
        rae_corr = float(rae(y_unb, corr_oof))
        rae_resid = float(rae(residual, oof_resid))
        per_seed_oof_rae.append(rae_corr)
        per_seed_resid_rae.append(rae_resid)
        per_seed_per_fold_rae.append(per_fold_resid_rae)

        te_resid_s = _deploy_refit(X_unb_df, residual, X_te_df, seed)
        per_seed_te_resid[i] = te_resid_s
        print(f"   seed={seed}  corr_OOF_RAE={rae_corr:.4f}  "
              f"resid_OOF_RAE={rae_resid:.4f}  "
              f"wall={time.time() - ts:.1f}s")

    mean_bag_oof_corr = per_seed_oof_corr.mean(axis=0)
    mean_bag_te_resid = per_seed_te_resid.mean(axis=0)
    te_corrected_513 = (te_anchor_513 + mean_bag_te_resid).astype(np.float32)
    # gentle clip to a sane pEC50 range
    te_corrected_513 = np.clip(te_corrected_513, 3.0, 8.0).astype(np.float32)
    rae_meanbag_oof = float(rae(y_unb, mean_bag_oof_corr))
    rae_meanbag_te_at_unb = float(rae(y_unb, te_corrected_513[unb_idx]))
    print(f"\n   mean-bag OOF RAE = {rae_meanbag_oof:.4f}  "
          f"(per-seed mean {float(np.mean(per_seed_oof_rae)):.4f}  "
          f"std {float(np.std(per_seed_oof_rae)):.4f})  "
          f"vs anchor {rae_meanbag_oof - rae_anchor:+.4f}")

    # ---- Step 6: artefacts ----
    print("\n" + "=" * 78)
    print(f"MEAN-BAG OOF RAE = {rae_meanbag_oof:.4f}")
    print("=" * 78)
    print(f"[summary] vs anchor             = "
          f"{rae_meanbag_oof - rae_anchor:+.4f}")
    print(f"[summary] vs nb2171 ceiling     = "
          f"{rae_meanbag_oof - NB2171_REF:+.4f}")

    pred_oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(pred_oof_path, mean_bag_oof_corr.astype(np.float32))
    np.save(te_path, te_corrected_513)
    print(f"\n[save] {pred_oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_lgbm_bit_categorical.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": te_corrected_513.astype(np.float32),
    }).to_csv(sub_csv, index=False)
    print(f"[save] {sub_csv}")

    # ---- Gate ----
    mean_rae = rae_meanbag_oof
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "=" * 78)
    print("GATE EVALUATION")
    print("=" * 78)
    print(f"   mean_rae (mean-bag corr OOF)              = {mean_rae:.4f}")
    print(f"   < {GATE_PROMOTE:.4f}  (PROMOTE)       = {mean_rae < GATE_PROMOTE}")
    print(f"   < {GATE_MARGINAL:.4f}  (MARGINAL_BEAT) = "
          f"{mean_rae < GATE_MARGINAL}")
    print(f"   VERDICT                                   = {verdict}")

    summary = {
        "tag": TAG,
        "method": (
            "LightGBM on top-50 Morgan-FP bits (selected by 513-row "
            "population frequency) wrapped as pandas Categorical (int8) "
            "with categorical_feature='auto'; fit on chemprop_aux residual; "
            "5-fold scaffold CV mean-bag over 5 kf_seeds."
        ),
        "paradigm": (
            "lgbm_categorical_split_routine_on_binary_morgan_bits_"
            "fisher_partition_distinct_from_numeric_threshold_search"
        ),
        "novelty": (
            "Pandas Categorical dtype on Morgan bits triggers LightGBM's "
            "categorical-feature optimization (Fisher-partition split "
            "routine) rather than the default numeric threshold search "
            "used by every prior LGBM-on-Morgan attempt on this anchor."
        ),
        "anchor": "chemprop_aux",
        "anchor_pre_unblind": True,
        "anchor_in_rae": rae_anchor,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "nb2171_ref": NB2171_REF,
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "n_unique_scaffolds_unb": int(n_unique_scaf),
        "morgan_n_bits": MORGAN_BITS,
        "morgan_radius": MORGAN_RADIUS,
        "top_k_bits": TOP_K_BITS,
        "top_bit_indices": [int(b) for b in top_bit_idx],
        "top_bit_freq_te513": [int(c) for c in top_bit_freq],
        "feat_dim": int(X_unb_df.shape[1]),
        "n_categorical_cols": int(len(cat_cols)),
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "lgbm_max_depth": LGBM_MAX_DEPTH,
        "lgbm_num_leaves": LGBM_NUM_LEAVES,
        "lgbm_n_estimators": LGBM_N_ESTIMATORS,
        "lgbm_learning_rate": LGBM_LEARNING_RATE,
        "lgbm_categorical_feature": "auto",
        "per_seed_corr_oof_rae": [float(x) for x in per_seed_oof_rae],
        "per_seed_resid_oof_rae": [float(x) for x in per_seed_resid_rae],
        "per_seed_per_fold_resid_rae": per_seed_per_fold_rae,
        "per_seed_corr_oof_mean_rae": float(np.mean(per_seed_oof_rae)),
        "per_seed_corr_oof_std_rae": float(np.std(per_seed_oof_rae)),
        "rae_meanbag_oof_corr": rae_meanbag_oof,
        "rae_meanbag_te_at_unb_in_sample": rae_meanbag_te_at_unb,
        "mean_rae": rae_meanbag_oof,
        "delta_vs_anchor": rae_meanbag_oof - rae_anchor,
        "delta_vs_nb2171": rae_meanbag_oof - NB2171_REF,
        "te_deploy_mean": float(te_corrected_513.mean()),
        "te_deploy_std": float(te_corrected_513.std()),
        "pred_oof_path": str(pred_oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv),
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "verdict": verdict,
        "promote": bool(verdict == "PROMOTE"),
        "marginal_beat": bool(verdict == "MARGINAL_BEAT"),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae",
        "delta_vs_anchor",
        "delta_vs_nb2171",
        "verdict",
        "feat_dim",
        "n_categorical_cols",
    ):
        print(f"  {k}: {res.get(k)}")
    print("\n  per_seed_corr_oof_rae:")
    for i, x in enumerate(res.get("per_seed_corr_oof_rae", [])):
        print(f"    kf_seed={KF_SEEDS[i]}  rae={x:.4f}")
