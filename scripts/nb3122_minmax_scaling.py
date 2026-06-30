"""nb3122 -- MinMaxScaler [0,1] + K=18 LGBM residual on chemprop_aux anchor,
            deep-30 fresh seeds, blend with nb2960 K=18 deep-30 bag.

NEW PARADIGM (vs StandardScaler / RobustScaler baselines):
    StandardScaler (mean/std) and RobustScaler (median/IQR) are LOCATION+SCALE
    estimators -- they shift to a center and scale to unit-ish spread.
    MinMaxScaler is a RANGE scaler: it maps each column to exactly [0,1]
    using only the per-column min and max.  This is a different inductive
    bias on the feature distribution:

      * StandardScaler:    z = (x - mean) / std        center=mean,  spread=std
      * RobustScaler:      z = (x - median) / IQR      center=median, spread=IQR
      * MinMaxScaler:      z = (x - min) / (max-min)   center=min,    spread=range

    LGBM tree splits are invariant under monotone transforms of a single
    feature in isolation, but the *interaction* of bucketization (histogram
    binning at min_child_samples=5) with the scale CAN matter: MinMax puts
    all features on a common [0,1] interval, which makes the per-feature
    histogram bin widths (default 256 bins / scale) approximately uniform
    across features.  Under Standard, features with heavy-tailed
    distributions get most of their mass crammed into one or two bins and
    splits land on extreme percentiles; under MinMax, the bins span the
    full range and rare-tail outliers get their own bins.

    On the K=18 substrate (chemprop_aux residual, n=253), this is a
    preprocessing-only swap on a fixed anchor + model.  Same cycle-134-style
    substrate change: model class fixed, scaling changed.

PROTOCOL:
    1. Load X_117_unb (253, 117) + X_117_te (513, 117) from pyramid cache.
    2. Slice K=18 indices from nb2604 (k18_idx_in_117col).
    3. residual = y_unb - chemprop_aux_te[unb_idx]  (only PRE-clean anchor).
    4. For each of 30 fresh seeds {3001..3030}:
        - Per scaffold-fold: MinMaxScaler [0,1] fit on train -> transform val.
        - LGBM(max_depth=4, num_leaves=15, n_est=300, lr=0.03) on K=18
          MinMax-scaled features, residual target.
        - Deploy: refit MinMaxScaler + LGBM on full 253 -> predict 513
          residual.
    5. Build deep-30 bag means: oof_resid_bag (mean over seeds),
       te_resid_bag (mean over seeds).
    6. pred_oof = chemprop_aux_anchor + oof_resid_bag    (253,)
       pred_te  = chemprop_aux_te + te_resid_bag         (513,)
    7. Blend with nb2960 deep-30 K-pyramid bag (the parent ladder candidate)
       at equal weight:
         blend_oof = 0.5 * pred_oof_minmax + 0.5 * nb2960_pred_oof
         blend_te  = 0.5 * pred_te_minmax  + 0.5 * te_nb2960
    8. Report:
        - K=18 MinMax mean-bag RAE on 253 unblind
        - blend mean-bag RAE on 253 unblind
        - delta vs K=18 StandardScaler reference (nb2960 K18 = 0.4536)
        - delta vs nb2960 blend reference (0.4576)

GATE:
    mean_rae (final blend) < 0.4475 -> "BETTER"
    else                            -> "FAIL"

References (deep-30 fresh-seed, chemprop_aux anchor, K=18 residual LGBM):
    nb2960 K18 deep-30 bag        = 0.4536  (StandardScaler-equivalent)
    nb2960 blend (0.5*K20+0.5*eq) = 0.4576
    nb2171 5-anchor pyramid       = 0.4682
    chemprop_aux baseline         = 0.6216

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/te_chemprop_aux.npy
    data/processed/pyramid/X_117_unb.npy
    data/processed/pyramid/X_117_te.npy
    data/processed/nb2604_summary.json   (K=18 idx)
    data/processed/nb2960_pred_oof.npy   (253,) blend partner
    data/processed/te_nb2960.npy         (513,) blend partner

Outputs:
    data/processed/nb3122_summary.json
    data/processed/nb3122_pred_oof.npy   (253,) float32 -- blended OOF
    data/processed/te_nb3122.npy         (513,) float32 -- blended te
    submissions/nb3122_minmax_K18_blend_nb2960.csv  (only on BETTER)
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3122"
PARENT_TAG = "nb2960"

# -- Inputs --------------------------------------------------------------------
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"
NB2604_SUMMARY = DATA_PROCESSED / "nb2604_summary.json"

# blend partner
NB2960_OOF_PATH = DATA_PROCESSED / "nb2960_pred_oof.npy"
NB2960_TE_PATH = DATA_PROCESSED / "te_nb2960.npy"

# -- Deep-30 + CV protocol ----------------------------------------------------
RESID_FOLDS = 5
RESID_SEEDS_DEEP = list(range(3001, 3031))   # 30 fresh seeds {3001..3030}
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]    # for scaffold-CV pooled eval

# -- Blend with nb2960 --------------------------------------------------------
W_MINMAX = 0.5
W_NB2960 = 0.5

# -- Gate ---------------------------------------------------------------------
GATE_BETTER = 0.4475

# -- References ---------------------------------------------------------------
REF_K18_DEEP30 = 0.4536   # nb2960 K18 deep-30 (StandardScaler-equivalent)
REF_NB2960_BLEND = 0.4576
REF_NB2171 = 0.4682
CHEMPROP_AUX_REF = 0.6216


def _lgbm_params(seed: int) -> dict:
    return dict(
        objective="regression",
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


def _residual_cross_fit_one_seed(
    X: np.ndarray,
    residual: np.ndarray,
    seed: int,
) -> np.ndarray:
    """KFold residual cross-fit with per-fold MinMaxScaler + LGBM."""
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        scaler = MinMaxScaler(feature_range=(0.0, 1.0), clip=True)
        X_tr = scaler.fit_transform(X[tr_loc]).astype(np.float32)
        X_va = scaler.transform(X[va_loc]).astype(np.float32)
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X_tr, residual[tr_loc])
        oof[va_loc] = mdl.predict(X_va)
    if np.isnan(oof).any():
        raise RuntimeError(f"KFold did not cover all rows (seed={seed})")
    return oof


def _deploy_te_one_seed(
    X_unb: np.ndarray,
    residual: np.ndarray,
    X_te: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Fit MinMaxScaler + LGBM on full 253 -> predict 513 residual."""
    scaler = MinMaxScaler(feature_range=(0.0, 1.0), clip=True)
    X_unb_s = scaler.fit_transform(X_unb).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)
    mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
    mdl.fit(X_unb_s, residual)
    return mdl.predict(X_te_s).astype(np.float32)


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- MinMaxScaler [0,1] + LGBM on K=18 substrate "
          f"(chemprop_aux residual, deep-30 fresh seeds)")
    print(f"        anchor={ANCHOR}  KFold residual {RESID_FOLDS}-fold")
    print(f"        deep seeds = {RESID_SEEDS_DEEP[0]}..{RESID_SEEDS_DEEP[-1]} "
          f"(n={len(RESID_SEEDS_DEEP)})")
    print(f"        blend: {W_MINMAX}*MinMax_K18  +  {W_NB2960}*nb2960")
    print(f"        ref nb2960 K18 deep-30 (Standard-eq) = {REF_K18_DEEP30:.4f}")
    print(f"        ref nb2960 blend                      = {REF_NB2960_BLEND:.4f}")
    print(f"        GATE: < {GATE_BETTER} -> BETTER  else FAIL")
    print("=" * 78)

    # ---- Load truth, anchor, scaffolds ----
    te = load_test()
    n_test = len(te)
    te_smi_col = "smiles" if "smiles" in te.columns else "SMILES"
    te_name_col = "name" if "name" in te.columns else "Molecule Name"
    test_smiles = te[te_smi_col].astype(str).tolist()
    te_names = te[te_name_col].values

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(f"anchor te shape {te_anchor_513.shape}")
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} te[unb_idx] RAE = {rae_anchor:.4f} "
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

    # ---- Slice K=18 columns from nb2604 ----
    if not NB2604_SUMMARY.exists():
        raise FileNotFoundError(f"nb2604 summary missing: {NB2604_SUMMARY}")
    with open(NB2604_SUMMARY) as f:
        nb2604 = json.load(f)
    k18_idx = list(nb2604["k18_idx_in_117col"])
    assert len(k18_idx) == 18, f"expected 18 K=18 indices, got {len(k18_idx)}"
    print(f"[K18] loaded {len(k18_idx)} surviving indices from nb2604")
    print(f"[K18] idx = {k18_idx}")

    X_unb = X117_unb[:, k18_idx].astype(np.float32)
    X_te = X117_te[:, k18_idx].astype(np.float32)
    feat_dim = X_unb.shape[1]
    assert feat_dim == 18, f"feat_dim {feat_dim} != 18"
    print(f"[feat] X_unb (K=18) = {X_unb.shape}  X_te = {X_te.shape}")

    # Diagnostic: per-column min/max BEFORE MinMax scaling
    col_min = X_unb.min(axis=0)
    col_max = X_unb.max(axis=0)
    col_range = col_max - col_min
    print(f"[feat] pre-scale ranges: "
          f"min={col_range.min():.3e}  "
          f"median={np.median(col_range):.3e}  "
          f"max={col_range.max():.3e}")

    # ---- Scaffolds ----
    unb_smiles = [test_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaf] unique unb scaffolds = {n_unique_scaf}")

    # ---- Deep-30 fresh-seed bag ----
    print("\n" + "-" * 78)
    print(f"DEEP-30 FRESH-SEED BAG  n_seeds={len(RESID_SEEDS_DEEP)}  "
          f"dim={feat_dim}")
    print("-" * 78)
    sum_oof_resid = np.zeros(n_unb, dtype=np.float64)
    sum_te_resid = np.zeros(n_test, dtype=np.float64)
    per_seed_rae: list[float] = []
    for i, seed in enumerate(RESID_SEEDS_DEEP):
        ts = time.time()
        resid_oof = _residual_cross_fit_one_seed(X_unb, residual, seed)
        pred_corr = anchor + resid_oof
        rae_s = float(rae(y_unb, pred_corr))
        per_seed_rae.append(rae_s)
        sum_oof_resid += resid_oof
        te_resid = _deploy_te_one_seed(X_unb, residual, X_te, seed)
        sum_te_resid += te_resid
        if (i % 10) == 0 or i == len(RESID_SEEDS_DEEP) - 1:
            print(f"   seed={seed}  rae_corr={rae_s:.4f}  "
                  f"d_vs_anchor={rae_s - rae_anchor:+.4f}  "
                  f"wall={time.time() - ts:.2f}s  "
                  f"({i + 1}/{len(RESID_SEEDS_DEEP)})")

    bag_oof_resid = sum_oof_resid / len(RESID_SEEDS_DEEP)
    bag_te_resid = sum_te_resid / len(RESID_SEEDS_DEEP)
    pred_oof_minmax = anchor + bag_oof_resid
    pred_te_minmax = te_anchor_513 + bag_te_resid
    rae_minmax_bag = float(rae(y_unb, pred_oof_minmax))
    per_seed_arr = np.asarray(per_seed_rae, dtype=np.float64)
    per_seed_mean = float(per_seed_arr.mean())
    per_seed_std = (float(per_seed_arr.std(ddof=1))
                    if len(per_seed_arr) > 1 else 0.0)

    print(f"\n[bag] per_seed_mean RAE = {per_seed_mean:.4f}  "
          f"std={per_seed_std:.4f}  "
          f"min={per_seed_arr.min():.4f}  max={per_seed_arr.max():.4f}")
    print(f"[bag] MinMax K18 30-seed BAG-MEAN RAE = {rae_minmax_bag:.4f}")
    print(f"[bag] vs ref K18 deep-30 (Standard-eq) {REF_K18_DEEP30:.4f}: "
          f"{rae_minmax_bag - REF_K18_DEEP30:+.4f}")

    # ---- Load nb2960 blend partner ----
    if not NB2960_OOF_PATH.exists() or not NB2960_TE_PATH.exists():
        raise FileNotFoundError(f"nb2960 OOF/te missing")
    nb2960_oof = np.load(NB2960_OOF_PATH).astype(np.float64)
    nb2960_te = np.load(NB2960_TE_PATH).astype(np.float64)
    if nb2960_oof.shape != (n_unb,):
        raise ValueError(f"nb2960_oof shape {nb2960_oof.shape}")
    if nb2960_te.shape != (n_test,):
        raise ValueError(f"nb2960_te shape {nb2960_te.shape}")
    rae_nb2960 = float(rae(y_unb, nb2960_oof))
    print(f"\n[blend] nb2960 OOF RAE = {rae_nb2960:.4f}  "
          f"(ref blend {REF_NB2960_BLEND:.4f})")

    # ---- Blend ----
    blend_oof = (W_MINMAX * pred_oof_minmax + W_NB2960 * nb2960_oof)
    blend_te = (W_MINMAX * pred_te_minmax + W_NB2960 * nb2960_te)
    rae_blend = float(rae(y_unb, blend_oof))
    print(f"[blend] {W_MINMAX}*MinMax + {W_NB2960}*nb2960 OOF RAE = "
          f"{rae_blend:.4f}")
    print(f"[blend] vs nb2960 blend ref {REF_NB2960_BLEND:.4f}: "
          f"{rae_blend - REF_NB2960_BLEND:+.4f}")
    print(f"[blend] vs MinMax bag alone {rae_minmax_bag:.4f}: "
          f"{rae_blend - rae_minmax_bag:+.4f}")

    # ---- Scaffold-CV pooled grand mean (5 kf_seeds) ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD-CV POOLED GRAND MEAN  kf_seeds={KF_SEEDS}  "
          f"n_folds={RESID_FOLDS}")
    print("-" * 78)
    per_kf_pooled: list[float] = []
    for kf_seed in KF_SEEDS:
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=RESID_FOLDS, shuffle=True, seed=kf_seed,
        )
        oof_pooled = np.full(n_unb, np.nan, dtype=np.float64)
        for tr_loc, va_loc in splits:
            oof_pooled[va_loc] = blend_oof[va_loc]
        if np.isnan(oof_pooled).any():
            raise RuntimeError(f"scaffold splits did not cover (seed={kf_seed})")
        pooled = float(rae(y_unb, oof_pooled))
        per_kf_pooled.append(pooled)
        print(f"   kf_seed={kf_seed}  pooled_RAE={pooled:.4f}")

    pooled_arr = np.asarray(per_kf_pooled, dtype=np.float64)
    pooled_mean = float(pooled_arr.mean())
    pooled_std = (float(pooled_arr.std(ddof=1))
                  if len(pooled_arr) > 1 else 0.0)
    pooled_min = float(pooled_arr.min())
    pooled_max = float(pooled_arr.max())
    print(f"\n[cv-pool] grand mean over {len(KF_SEEDS)} kf_seeds: "
          f"{pooled_mean:.4f} +/- {pooled_std:.5f}  "
          f"(min={pooled_min:.4f}  max={pooled_max:.4f})")

    # ---- Gate ----
    print("\n" + "=" * 78)
    print("GATE EVALUATION")
    print("=" * 78)
    if pooled_mean < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE nb3122 (MinMax K=18 + 0.5 nb2960 blend). Grand-mean "
            f"{pooled_mean:.4f} < {GATE_BETTER:.4f}; MinMaxScaler swap on "
            f"K=18 LGBM substrate genuinely lifts the post-hoc blend ceiling."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT nb3122. Grand-mean {pooled_mean:.4f} >= {GATE_BETTER:.4f}; "
            f"MinMax scaling does not improve over StandardScaler-equivalent "
            f"K=18 baseline {REF_K18_DEEP30:.4f}. Closes the MinMax axis on "
            f"K=18 + chemprop_aux residual."
        )
    print(f"   mean_rae (blend grand mean) = {pooled_mean:.4f}")
    print(f"   gate                         = < {GATE_BETTER:.4f}")
    print(f"   verdict                      = {verdict}")
    print(f"   ladder_action                = {ladder_action}")

    # ---- Save artifacts ----
    print("\n" + "-" * 78)
    print("SAVE")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, blend_oof.astype(np.float32))
    np.save(te_path, blend_te.astype(np.float32))
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    te_unb_in_rae = float(rae(y_unb, blend_te[unb_idx]))
    sub_csv = SUBMISSIONS / f"{TAG}_minmax_K18_blend_nb2960.csv"
    if verdict == "BETTER":
        pd.DataFrame({
            "SMILES": test_smiles,
            "Molecule Name": te_names,
            "pEC50": blend_te.astype(np.float32),
        }).to_csv(sub_csv, index=False)
        print(f"   [save] {sub_csv}")
    else:
        print(f"   [skip] verdict={verdict}; no submission CSV written")

    summary = {
        "tag": TAG,
        "parent_tag": PARENT_TAG,
        "method": "minmax_scaler_K18_LGBM_deep30_blend_nb2960",
        "rationale": (
            "MinMaxScaler [0,1] vs StandardScaler / RobustScaler on K=18 "
            "substrate; preprocessing-only swap on fixed chemprop_aux anchor "
            "+ LGBM (max_depth=4, n_est=300, lr=0.03), deep-30 fresh seeds, "
            "blended at equal weight with nb2960 K-pyramid bag."
        ),
        "anchor": ANCHOR,
        "anchor_pre_unblind": True,
        "anchor_te_path": str(ANCHOR_TE_PATH),
        "rae_anchor_chemprop_aux": rae_anchor,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "x117_unb_path": str(X117_UNB_PATH),
        "x117_te_path": str(X117_TE_PATH),
        "k18_idx_source": str(NB2604_SUMMARY),
        "k18_idx_in_117col": [int(j) for j in k18_idx],
        "feat_dim": int(feat_dim),
        "scaler": {
            "class": "sklearn.preprocessing.MinMaxScaler",
            "feature_range": [0.0, 1.0],
            "clip": True,
        },
        "model_class": "lightgbm.LGBMRegressor",
        "lgbm_params_sample": _lgbm_params(RESID_SEEDS_DEEP[0]),
        "resid_folds": RESID_FOLDS,
        "resid_seeds_deep": RESID_SEEDS_DEEP,
        "n_seeds_deep": len(RESID_SEEDS_DEEP),
        "kf_seeds": KF_SEEDS,
        "n_folds": RESID_FOLDS,
        "n_unb": int(n_unb),
        "n_test": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "blend_w_minmax": W_MINMAX,
        "blend_w_nb2960": W_NB2960,
        "nb2960_oof_path": str(NB2960_OOF_PATH),
        "nb2960_te_path": str(NB2960_TE_PATH),
        "nb2960_oof_rae": rae_nb2960,
        "per_seed_rae": per_seed_rae,
        "per_seed_mean_rae": per_seed_mean,
        "per_seed_std_rae": per_seed_std,
        "minmax_bag_oof_rae": rae_minmax_bag,
        "blend_oof_rae_in_sample": rae_blend,
        "per_kf_pooled_rae": per_kf_pooled,
        "pooled_rae_grand_mean": pooled_mean,
        "pooled_rae_grand_std": pooled_std,
        "pooled_rae_grand_min": pooled_min,
        "pooled_rae_grand_max": pooled_max,
        "mean_rae": pooled_mean,
        "te_unb_in_sample_rae": te_unb_in_rae,
        "te_mean": float(blend_te.mean()),
        "te_std": float(blend_te.std()),
        "ref_K18_deep30_standard": REF_K18_DEEP30,
        "ref_nb2960_blend": REF_NB2960_BLEND,
        "ref_nb2171": REF_NB2171,
        "delta_vs_K18_deep30": rae_minmax_bag - REF_K18_DEEP30,
        "delta_vs_nb2960_blend": pooled_mean - REF_NB2960_BLEND,
        "gate_better": GATE_BETTER,
        "verdict": verdict,
        "ladder_action": ladder_action,
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict == "BETTER" else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   MinMax K18 30-seed bag RAE   = {rae_minmax_bag:.4f}")
    print(f"   nb2960 OOF RAE               = {rae_nb2960:.4f}")
    print(f"   blend OOF RAE (in-sample)    = {rae_blend:.4f}")
    print(f"   blend grand mean ({len(KF_SEEDS)} kf)    = "
          f"{pooled_mean:.4f} +/- {pooled_std:.5f}")
    print(f"   te[unb] in-sample RAE        = {te_unb_in_rae:.4f}")
    print(f"   delta vs K18 Standard-eq     = "
          f"{rae_minmax_bag - REF_K18_DEEP30:+.4f}")
    print(f"   delta vs nb2960 blend ref    = "
          f"{pooled_mean - REF_NB2960_BLEND:+.4f}")
    print(f"   verdict                      = {verdict}")
    print(f"   wall                         = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "minmax_bag_oof_rae",
        "blend_oof_rae_in_sample",
        "pooled_rae_grand_mean",
        "pooled_rae_grand_std",
        "delta_vs_K18_deep30",
        "delta_vs_nb2960_blend",
        "te_unb_in_sample_rae",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
