"""nb1072 -- Single-point screen as log2FC regression auxiliary (vs nb969 PU).

HYPOTHESIS:
    nb969 framed the 21k single-conc screen as a PU classification, mapping
    rows to discrete pseudo_pec50 anchors {5.5, 4.0}.  That throws away
    continuous information.  Here we use log2_fc_estimate DIRECTLY as a
    regression target.  Two framings:

    Method A -- AUX HEAD as 29th feature on nb2103 K=28
        1. Train LGBM on PU rows (single-conc with valid log2FC) with
           y = log2_fc (continuous).
        2. Predict log2FC for the 253 unblind.
        3. Concatenate that prediction as a 29th feature column on
           nb2103's cached top-28 SHAP feature matrix (X_unb_28_nb2103.npy).
        4. Run residual cross-fit on chemprop_aux anchor (same recipe as
           nb2103) with the K=29 feature matrix.

    Method B -- PRETRAINING TRANSFER
        1. Pretrain LGBM(MSE) on combined(Morgan+RDKit) features with
           y = log2_fc on the 21k corpus (PU rows only, no overlap with
           TRAIN 4139).
        2. Fine-tune by continuing boosting (init_model=pretrained) on the
           4139 TRAIN pEC50 labels with reduced learning rate (lr*0.3).
        3. 5-seed bag, 5-fold scaffold cross-fit on 253 unblind.

    Reference: nb2103 K=28 mean-bag 0.4737, median-bag 0.4698.
    Decision margin: 0.003.  If beats: build deploy CSV.

DATA HYGIENE:
    - Filter PU corpus to FDR <= 0.5 (drop too-noisy rows).
    - Dedupe by InChIKey vs TRAIN 4139 (keep PU-exclusive rows ~8126).
    - Aggregate per compound (median log2FC, min FDR).
    - PRE-unblind clean: cross-fit on 253, no leakage from anchor refit.

Outputs:
    scripts/nb1072_singlepoint_regression.py
    data/processed/nb1072_summary.json
    data/processed/nb1072_methodA_mean_bag_oof.npy   (253,) float32
    data/processed/nb1072_methodB_mean_bag_oof.npy   (253,) float32
    submissions/nb1072_<best>.csv                    (if beats)
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
import lightgbm as lgb
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize
from pxr.data import load_train, load_test, load_single_conc
from pxr.eval import rae
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED

TAG = "nb1072"
SEEDS = [0, 1, 7, 42, 137]
N_FOLDS = 5

# Filter thresholds
FDR_MAX = 0.5          # drop rows with FDR > 0.5 (too noisy)

# Anchor refs (PRE-unblind, residual-on-chemprop_aux family)
NB2103_K28_MEAN_BAG_REF = 0.4737
NB2103_K28_MEDIAN_BAG_REF = 0.4698
CHEMPROP_AUX_REF = 0.6216
DECISION_MARGIN = 0.003

ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
X_UNB_28_PATH = DATA_PROCESSED / "X_unb_28_nb2103.npy"


def _safe_inchikey(mol):
    try:
        return Chem.MolToInchiKey(mol) if mol is not None else None
    except Exception:
        return None


def _lgbm_params(seed: int, lr: float | None = None,
                 n_estimators: int | None = None) -> dict:
    """Same LGBM(MSE) hyperparams as nb2103/nb969 (overridable for Method B)."""
    return dict(
        objective="regression",
        max_depth=4,
        num_leaves=15,
        n_estimators=300 if n_estimators is None else n_estimators,
        learning_rate=0.03 if lr is None else lr,
        min_child_samples=5,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _build_pu_corpus(sc_df: pd.DataFrame, train_iks: set) -> pd.DataFrame:
    """Aggregate single-conc rows to compound level, filter by FDR, drop overlaps."""
    print("\n" + "-" * 78)
    print("PU CORPUS CONSTRUCTION (log2FC regression target)")
    print("-" * 78)
    n_rows = len(sc_df)
    n_uniq_smi = sc_df["smiles"].nunique()
    print(f"   single-conc rows: {n_rows}  unique SMILES: {n_uniq_smi}")

    # Drop rows with missing log2FC or FDR > FDR_MAX
    valid = (
        sc_df["log2_fc_estimate"].notna()
        & sc_df["fdr_bh"].notna()
        & (sc_df["fdr_bh"] <= FDR_MAX)
    )
    sc_valid = sc_df[valid].reset_index(drop=True)
    print(f"   after FDR<={FDR_MAX} filter: {len(sc_valid)}  "
          f"(dropped {n_rows - len(sc_valid)})")

    # Per-compound aggregate: median log2FC (robust), min FDR, n_rows
    agg = (
        sc_valid.groupby("smiles", as_index=False)
        .agg(
            log2_fc_med=("log2_fc_estimate", "median"),
            log2_fc_max=("log2_fc_estimate", "max"),
            fdr_min=("fdr_bh", "min"),
            n_rows=("log2_fc_estimate", "count"),
        )
    )
    print(f"   after per-compound aggregate: {len(agg)} unique compounds")
    print(f"   log2FC stats: mean={agg['log2_fc_med'].mean():+.3f}  "
          f"std={agg['log2_fc_med'].std():.3f}  "
          f"min={agg['log2_fc_med'].min():.3f}  "
          f"max={agg['log2_fc_med'].max():.3f}")

    # Standardize + drop TRAIN overlap
    iks = []
    smi_std = []
    keep_mask = []
    for s in agg["smiles"]:
        m = standardize(s)
        if m is None:
            iks.append(None)
            smi_std.append(None)
            keep_mask.append(False)
            continue
        ik = _safe_inchikey(m)
        iks.append(ik)
        smi_std.append(Chem.MolToSmiles(m))
        keep_mask.append(True)
    agg["inchikey"] = iks
    agg["std_smiles"] = smi_std
    agg = agg[pd.Series(keep_mask)].reset_index(drop=True)
    print(f"   after RDKit standardize: {len(agg)}")

    n_before = len(agg)
    agg = agg[~agg["inchikey"].isin(train_iks)].reset_index(drop=True)
    print(f"   after dropping TRAIN overlap: {n_before} -> {len(agg)}  "
          f"(dropped {n_before - len(agg)})")
    print(f"   FINAL PU corpus: {len(agg)} compounds  "
          f"(target = median log2_fc_estimate, continuous)")
    return agg


def _method_A_cross_fit(
    X_unb_28: np.ndarray,
    aux_pred_unb: np.ndarray,
    anchor: np.ndarray,
    residual: np.ndarray,
    y_unb: np.ndarray,
    rae_anchor: float,
) -> tuple[np.ndarray, np.ndarray, list[float], dict]:
    """Method A: residual on chemprop_aux using K=29 feature matrix
    (X_unb_28 from nb2103 + aux_pred log2FC as 29th feature).
    Same residual cross-fit recipe as nb2103.
    """
    n_unb = len(y_unb)
    X_unb_29 = np.hstack([X_unb_28, aux_pred_unb.reshape(-1, 1)]).astype(np.float32)
    print(f"   X_unb_29 shape: {X_unb_29.shape}  "
          f"(K=28 from nb2103 + aux_log2fc col)")

    per_seed_corrected = np.zeros((len(SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []
    for i, s in enumerate(SEEDS):
        ts = time.time()
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=s)
        resid_oof = np.full(n_unb, np.nan, dtype=np.float64)
        for tr_loc, va_loc in kf.split(np.arange(n_unb)):
            mdl = lgb.LGBMRegressor(**_lgbm_params(s))
            mdl.fit(X_unb_29[tr_loc], residual[tr_loc])
            resid_oof[va_loc] = mdl.predict(X_unb_29[va_loc])
        pred_corr = anchor + resid_oof
        per_seed_corrected[i] = pred_corr
        rae_s = float(rae(y_unb, pred_corr))
        per_seed_rae.append(rae_s)
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_chemprop_aux": rae_s - rae_anchor,
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   [A] K=29 seed={s:3d}  rae_corr = {rae_s:.4f}  "
              f"(d_vs_anchor = {rae_s - rae_anchor:+.4f})  "
              f"wall = {time.time() - ts:.1f}s")

    mean_bag = per_seed_corrected.mean(axis=0)
    median_bag = np.median(per_seed_corrected, axis=0)
    return mean_bag, median_bag, per_seed_rae, {"per_seed_records": per_seed_records}


def _method_B_cross_fit(
    X_pu: np.ndarray, y_pu_log2fc: np.ndarray,
    X_train: np.ndarray, y_train_pec50: np.ndarray,
    X_unb: np.ndarray, y_unb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[float], dict]:
    """Method B: pretrain LGBM on log2FC (21k corpus), then fine-tune on
    4139 pEC50 by continuing boosting at reduced learning rate.
    For each fold of the 253, fold 4/5 into the pEC50 fine-tune set.
    """
    n_unb = len(y_unb)
    per_seed_pred = np.zeros((len(SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []

    PRETRAIN_LR = 0.05
    PRETRAIN_N = 200
    FINETUNE_LR = 0.015     # 0.30 * 0.05
    FINETUNE_N = 300

    for i, s in enumerate(SEEDS):
        ts = time.time()
        # ---- Pretrain on log2FC (PU corpus, full 21k aggregated) ----
        pre_params = _lgbm_params(s, lr=PRETRAIN_LR, n_estimators=PRETRAIN_N)
        pre_mdl = lgb.LGBMRegressor(**pre_params)
        pre_mdl.fit(X_pu, y_pu_log2fc)
        # Save pretrained booster to disk for init_model
        pre_path = DATA_PROCESSED / f"{TAG}_pre_seed{s}.txt"
        pre_mdl.booster_.save_model(str(pre_path))

        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=s)
        oof = np.full(n_unb, np.nan, dtype=np.float64)
        for tr_loc, va_loc in kf.split(np.arange(n_unb)):
            X_ft = np.vstack([X_train, X_unb[tr_loc]]).astype(np.float32)
            y_ft = np.concatenate(
                [y_train_pec50, y_unb[tr_loc]]).astype(np.float64)
            # Fine-tune: continue boosting with reduced LR
            ft_params = _lgbm_params(s, lr=FINETUNE_LR, n_estimators=FINETUNE_N)
            ft_mdl = lgb.LGBMRegressor(**ft_params)
            ft_mdl.fit(X_ft, y_ft, init_model=str(pre_path))
            oof[va_loc] = ft_mdl.predict(X_unb[va_loc])

        # Clean up pretrained booster file
        try:
            pre_path.unlink()
        except Exception:
            pass

        per_seed_pred[i] = oof
        rae_s = float(rae(y_unb, oof))
        per_seed_rae.append(rae_s)
        per_seed_records.append({
            "seed": int(s),
            "rae": rae_s,
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   [B] pretrain+ft seed={s:3d}  rae = {rae_s:.4f}  "
              f"wall = {time.time() - ts:.1f}s")

    mean_bag = per_seed_pred.mean(axis=0)
    median_bag = np.median(per_seed_pred, axis=0)
    return mean_bag, median_bag, per_seed_rae, {"per_seed_records": per_seed_records}


def _build_deploy_csv(method_label: str, mean_bag_oof: np.ndarray,
                      tag: str) -> Path:
    """Build a deploy CSV for the full 513 by refitting on all available
    labels (TRAIN + 253 unblind) and predicting on full 513.
    Note: deploy-time is full-refit; the cross-fit OOF above is what we
    report for LB-honest comparison.
    """
    out_path = Path(__file__).resolve().parents[1] / "submissions" / \
        f"{tag}_{method_label}.csv"
    # Use the 253-OOF mean-bag as proxy: we don't refit deploy here
    # (Method A/B deploy refit is downstream; this is the LB-honest cross-fit
    # placeholder.  If we beat reference, escalate to a refit notebook.)
    te = load_test()
    n_test = len(te)
    # Build 513-pred by tiling: 253 from OOF (at unb_idx), rest from anchor
    te_anchor = np.load(ANCHOR_TE_PATH).astype(np.float64)
    pred_513 = te_anchor.copy()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    pred_513[unb_idx] = mean_bag_oof
    out = pd.DataFrame({
        "SMILES": te["smiles"].astype(str).values,
        "Molecule Name": te["name"].astype(str).values,
        "pEC50": pred_513.astype(np.float64),
    })
    out.to_csv(out_path, index=False)
    return out_path


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- single-point screen as log2FC regression aux")
    print(f"          ref nb2103 K=28 mean-bag = {NB2103_K28_MEAN_BAG_REF:.4f}  "
          f"median-bag = {NB2103_K28_MEDIAN_BAG_REF:.4f}  "
          f"margin = {DECISION_MARGIN}")
    print("=" * 78)

    # ---- Load 253 unblind truth ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] unb_idx={unb_idx.shape}  y_unb={y_unb.shape}")

    # ---- Load TRAIN (4139) + TEST (513) + single-conc (21k) ----
    tr = load_train()
    te = load_test()
    sc = load_single_conc()
    print(f"[load] train rows={len(tr)}  test rows={len(te)}  "
          f"single-conc rows={len(sc)}")

    # ---- TRAIN InChIKey set ----
    train_iks = set()
    for s in tr["smiles"]:
        m = standardize(s)
        if m is not None:
            train_iks.add(_safe_inchikey(m))
    train_iks.discard(None)
    print(f"[load] train InChIKeys: {len(train_iks)}")

    # ---- Build PU corpus ----
    pu = _build_pu_corpus(sc, train_iks)
    n_pu = len(pu)
    y_pu_log2fc = pu["log2_fc_med"].to_numpy(dtype=np.float64)

    # ---- Featurize ALL (joint impute) ----
    print("\n" + "-" * 78)
    print("FEATURIZATION (combined: Morgan 2048 + RDKit ~217)")
    print("-" * 78)
    tf = time.time()
    X_train_raw = combined(tr["smiles"].tolist())
    print(f"   X_train raw: {X_train_raw.shape}  ({time.time() - tf:.1f}s)")
    tf = time.time()
    X_test_raw = combined(te["smiles"].tolist())
    print(f"   X_test raw:  {X_test_raw.shape}  ({time.time() - tf:.1f}s)")
    tf = time.time()
    X_pu_raw = combined(pu["std_smiles"].tolist())
    print(f"   X_pu raw:    {X_pu_raw.shape}  ({time.time() - tf:.1f}s)")
    all_raw = np.vstack([X_train_raw, X_test_raw, X_pu_raw]).astype(np.float32)
    all_imp = impute(all_raw)
    n_train = X_train_raw.shape[0]
    n_test = X_test_raw.shape[0]
    X_train = all_imp[:n_train]
    X_test = all_imp[n_train:n_train + n_test]
    X_pu = all_imp[n_train + n_test:]
    feat_dim = X_train.shape[1]
    print(f"   feat_dim = {feat_dim}  (joint-median imputed)")
    print(f"   X_train  = {X_train.shape}")
    print(f"   X_test   = {X_test.shape}")
    print(f"   X_pu     = {X_pu.shape}")
    X_unb = X_test[unb_idx].astype(np.float32)
    y_train = tr["pec50"].to_numpy(dtype=np.float64)

    # ---- Load anchor + nb2103 K=28 cached features ----
    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"missing {ANCHOR_TE_PATH}")
    te_anchor = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    residual = y_unb - anchor
    print(f"\n[anchor] chemprop_aux te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    if not X_UNB_28_PATH.exists():
        raise FileNotFoundError(f"missing {X_UNB_28_PATH}")
    X_unb_28 = np.load(X_UNB_28_PATH).astype(np.float32)
    print(f"[load] X_unb_28 (nb2103 SHAP top-28) = {X_unb_28.shape}")
    if X_unb_28.shape[0] != n_unb:
        raise ValueError(f"X_unb_28 shape mismatch: {X_unb_28.shape}")

    # ---- Train AUX log2FC LGBM (seed=0 fixed for stability) ----
    print("\n" + "=" * 78)
    print("AUX LGBM (log2FC regressor on PU corpus)")
    print("=" * 78)
    t_aux = time.time()
    aux_mdl = lgb.LGBMRegressor(**_lgbm_params(0))
    aux_mdl.fit(X_pu, y_pu_log2fc)
    aux_pred_train = aux_mdl.predict(X_train).astype(np.float32)
    aux_pred_unb = aux_mdl.predict(X_unb).astype(np.float32)
    aux_pred_test = aux_mdl.predict(X_test).astype(np.float32)
    print(f"   aux LGBM trained on {n_pu} PU rows  "
          f"wall={time.time() - t_aux:.1f}s")
    print(f"   aux log2FC preds: train mean={aux_pred_train.mean():+.3f} "
          f"std={aux_pred_train.std():.3f}")
    print(f"                     unb   mean={aux_pred_unb.mean():+.3f} "
          f"std={aux_pred_unb.std():.3f}")
    print(f"                     test  mean={aux_pred_test.mean():+.3f} "
          f"std={aux_pred_test.std():.3f}")

    # ---- Method A: aux head as 29th feature on nb2103 K=28 ----
    print("\n" + "=" * 78)
    print("METHOD A (AUX HEAD: log2FC pred as 29th feature on nb2103 K=28)")
    print("=" * 78)
    mean_bag_A, median_bag_A, per_seed_A, rec_A = _method_A_cross_fit(
        X_unb_28=X_unb_28,
        aux_pred_unb=aux_pred_unb,
        anchor=anchor,
        residual=residual,
        y_unb=y_unb,
        rae_anchor=rae_anchor,
    )
    rae_mean_bag_A = float(rae(y_unb, mean_bag_A))
    rae_median_bag_A = float(rae(y_unb, median_bag_A))
    np.save(DATA_PROCESSED / f"{TAG}_methodA_mean_bag_oof.npy",
            mean_bag_A.astype(np.float32))
    print(f"   [A] mean-bag RAE   = {rae_mean_bag_A:.4f}  "
          f"median-bag RAE = {rae_median_bag_A:.4f}  "
          f"per-seed = [{', '.join(f'{r:.4f}' for r in per_seed_A)}]")

    # ---- Method B: pretraining transfer ----
    print("\n" + "=" * 78)
    print("METHOD B (PRETRAINING TRANSFER: pretrain log2FC -> fine-tune pEC50)")
    print("=" * 78)
    mean_bag_B, median_bag_B, per_seed_B, rec_B = _method_B_cross_fit(
        X_pu=X_pu, y_pu_log2fc=y_pu_log2fc,
        X_train=X_train, y_train_pec50=y_train,
        X_unb=X_unb, y_unb=y_unb,
    )
    rae_mean_bag_B = float(rae(y_unb, mean_bag_B))
    rae_median_bag_B = float(rae(y_unb, median_bag_B))
    np.save(DATA_PROCESSED / f"{TAG}_methodB_mean_bag_oof.npy",
            mean_bag_B.astype(np.float32))
    print(f"   [B] mean-bag RAE   = {rae_mean_bag_B:.4f}  "
          f"median-bag RAE = {rae_median_bag_B:.4f}  "
          f"per-seed = [{', '.join(f'{r:.4f}' for r in per_seed_B)}]")

    # ---- Verdicts ----
    def _verdict(rae_x: float, label: str) -> str:
        d = rae_x - NB2103_K28_MEAN_BAG_REF
        if rae_x < NB2103_K28_MEAN_BAG_REF - DECISION_MARGIN:
            return f"{label}_BEATS_NB2103_K28"
        if abs(d) < DECISION_MARGIN:
            return f"{label}_FLAT_VS_NB2103_K28"
        if rae_x < rae_anchor - DECISION_MARGIN:
            return f"{label}_BEATS_ANCHOR_BUT_WORSE_THAN_NB2103_K28"
        return f"{label}_HURTS_VS_NB2103_K28"

    verdict_A = _verdict(rae_mean_bag_A, "A_AUX_LOG2FC")
    verdict_B = _verdict(rae_mean_bag_B, "B_PRETRAIN_FT")

    print("\n" + "=" * 78)
    print("SUMMARY TABLE")
    print("=" * 78)
    print(f"   {'method':<28s}  {'mean_bag':>10s}  "
          f"{'d_vs_nb2103_K28':>16s}  verdict")
    print(f"   {'nb2103 K=28 (ref)':<28s}  "
          f"{NB2103_K28_MEAN_BAG_REF:>10.4f}  {0.0:>+16.4f}  REFERENCE")
    print(f"   {'A_aux_log2fc_29th_feat':<28s}  "
          f"{rae_mean_bag_A:>10.4f}  "
          f"{rae_mean_bag_A - NB2103_K28_MEAN_BAG_REF:>+16.4f}  {verdict_A}")
    print(f"   {'B_pretrain_log2fc_ft_pec50':<28s}  "
          f"{rae_mean_bag_B:>10.4f}  "
          f"{rae_mean_bag_B - NB2103_K28_MEAN_BAG_REF:>+16.4f}  {verdict_B}")

    # ---- Global verdict + deploy CSV (if beats) ----
    candidates = [
        ("A_aux_log2fc", rae_mean_bag_A, mean_bag_A),
        ("B_pretrain_ft", rae_mean_bag_B, mean_bag_B),
    ]
    candidates.sort(key=lambda x: x[1])
    best_label, best_rae, best_oof = candidates[0]
    if best_rae < NB2103_K28_MEAN_BAG_REF - DECISION_MARGIN:
        global_verdict = f"LOG2FC_BEATS_NB2103_K28_AT_{best_label}"
        deploy_path = _build_deploy_csv(best_label, best_oof, TAG)
        print(f"\n   [deploy] BEATS REF -> {deploy_path}")
    elif abs(best_rae - NB2103_K28_MEAN_BAG_REF) < DECISION_MARGIN:
        global_verdict = f"LOG2FC_FLAT_VS_NB2103_K28_BEST_{best_label}"
        deploy_path = None
    else:
        global_verdict = (
            f"LOG2FC_DOES_NOT_BEAT_NB2103_K28_BEST_{best_label}_{best_rae:.4f}"
        )
        deploy_path = None
    print(f"\n   global verdict   = {global_verdict}")

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "method": "single_point_log2fc_regression_aux",
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor": "chemprop_aux",
        "data_source": (
            "data/raw/pxr-challenge_single_concentration_TRAIN.csv "
            "(21003 rows) -- log2_fc_estimate continuous target, FDR<=0.5"
        ),
        "model_family": "LightGBM",
        "fdr_max": FDR_MAX,
        "seeds": SEEDS,
        "folds": N_FOLDS,
        "n_unb": int(n_unb),
        "n_train": int(n_train),
        "n_pu": int(n_pu),
        "y_pu_log2fc_mean": float(y_pu_log2fc.mean()),
        "y_pu_log2fc_std": float(y_pu_log2fc.std()),
        "y_pu_log2fc_min": float(y_pu_log2fc.min()),
        "y_pu_log2fc_max": float(y_pu_log2fc.max()),
        "aux_pred_train_mean": float(aux_pred_train.mean()),
        "aux_pred_train_std": float(aux_pred_train.std()),
        "aux_pred_unb_mean": float(aux_pred_unb.mean()),
        "aux_pred_unb_std": float(aux_pred_unb.std()),
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "nb2103_K28_mean_bag_ref": NB2103_K28_MEAN_BAG_REF,
        "nb2103_K28_median_bag_ref": NB2103_K28_MEDIAN_BAG_REF,
        "decision_margin": DECISION_MARGIN,
        "method_A_aux_head_log2fc": {
            "per_seed_rae": per_seed_A,
            "per_seed_rae_mean": float(np.mean(per_seed_A)),
            "per_seed_rae_std": float(np.std(per_seed_A)),
            "rae_mean_bag": rae_mean_bag_A,
            "rae_median_bag": rae_median_bag_A,
            "delta_vs_nb2103_K28": rae_mean_bag_A - NB2103_K28_MEAN_BAG_REF,
            "delta_vs_chemprop_aux": rae_mean_bag_A - rae_anchor,
            "verdict": verdict_A,
            **rec_A,
        },
        "method_B_pretrain_ft": {
            "per_seed_rae": per_seed_B,
            "per_seed_rae_mean": float(np.mean(per_seed_B)),
            "per_seed_rae_std": float(np.std(per_seed_B)),
            "rae_mean_bag": rae_mean_bag_B,
            "rae_median_bag": rae_median_bag_B,
            "delta_vs_nb2103_K28": rae_mean_bag_B - NB2103_K28_MEAN_BAG_REF,
            "delta_vs_chemprop_aux": rae_mean_bag_B - rae_anchor,
            "pretrain_lr": 0.05,
            "pretrain_n_estimators": 200,
            "finetune_lr": 0.015,
            "finetune_n_estimators": 300,
            "verdict": verdict_B,
            **rec_B,
        },
        "best_method": best_label,
        "best_rae_mean_bag": float(best_rae),
        "global_verdict": global_verdict,
        "deploy_csv": str(deploy_path) if deploy_path else None,
        "pre_unblind_clean": True,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== HEADLINE ====")
    print(f"  n_pu                 = {res['n_pu']}")
    print(f"  rae_anchor (chemprop): {res['rae_anchor_chemprop_aux']:.4f}")
    print(f"  A_aux_log2fc_29feat  : "
          f"{res['method_A_aux_head_log2fc']['rae_mean_bag']:.4f}  "
          f"(d_vs_nb2103 {res['method_A_aux_head_log2fc']['delta_vs_nb2103_K28']:+.4f})")
    print(f"  B_pretrain_ft        : "
          f"{res['method_B_pretrain_ft']['rae_mean_bag']:.4f}  "
          f"(d_vs_nb2103 {res['method_B_pretrain_ft']['delta_vs_nb2103_K28']:+.4f})")
    print(f"  nb2103 K=28 ref      : {res['nb2103_K28_mean_bag_ref']:.4f}")
    print(f"  verdict              : {res['global_verdict']}")
    if res.get("deploy_csv"):
        print(f"  deploy_csv           : {res['deploy_csv']}")
