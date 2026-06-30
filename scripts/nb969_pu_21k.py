"""nb969 -- PU framing on 21k single-conc screen.

HYPOTHESIS:
    The 21,003-row single-concentration primary screen has 8,126 unique
    compounds NOT in the 4,139 TRAIN dose-response set.  Even without
    dose-response curves we have a binary-ish PU signal per row:
        positive: log2_fc_estimate > 0.5 AND fdr_bh < 0.05  -> likely PXR-active
        negative: log2_fc_estimate < 0.1                     -> likely inactive
        ambiguous: in between -- DROP from the PU set
    Aggregating per SMILES (max log2_fc, min FDR) gives compound-level
    PU labels we can fold back into the pEC50 regression as pseudo-targets:
        pseudo_pec50 = 5.5  (mid-active) for positive
        pseudo_pec50 = 4.0  (inactive)   for negative
    Penalty: PU rows carry sample_weight = 0.3 (PU bias).
    Inject only PU compounds NOT in TRAIN (avoid double-labeling).

DESIGN:
    Anchor reference = nb2103 K=28 mean-bag/median-bag RAE (0.4737/0.4698).
    nb2103 is a residual model on chemprop_aux with 28-col SHAP-tuned
    features cross-fit on 253; we compare the LB-honest RAE number here.

    Method A (SINGLE-TASK PU AUG)
        Train LGBM(MSE) on combined (Morgan + RDKit, 2265 dims) features.
        TRAIN rows (4139)         -> y = pec50           sample_weight = 1.0
        PU positives (kept)        -> y = pseudo_pec50_p sample_weight = 0.3
        PU negatives (kept)        -> y = pseudo_pec50_n sample_weight = 0.3
        253 unblind: 5-fold cross-fit; for each fold, train on the augmented
        corpus + 4/5 of the 253, predict on the 1/5 held-out.  5-seed bag
        (seeds 0, 1, 7, 42, 137).  Report mean-bag and median-bag RAE.

    Method B (AUXILIARY HEAD)
        First train an auxiliary LGBM on PU rows only:
            X = combined(PU smiles), y = pseudo_pec50 (continuous proxy)
        Use the aux LGBM to predict for every row in TRAIN + 253 + PU,
        and add that prediction as one extra feature column to the main
        LGBM (which is then trained on TRAIN only, NO PU pseudo-labels).
        Same 5-seed bag, 5-fold cross-fit on 253.

    Method C (BASELINE, no PU)
        Same LGBM(MSE) on combined features, TRAIN only (no PU).  Pure
        ablation control.

    Decision margin = 0.003.

Outputs:
    scripts/nb969_pu_21k.py
    data/processed/nb969_summary.json
    data/processed/nb969_methodA_oof_mean_bag.npy   (253,) float32
    data/processed/nb969_methodB_oof_mean_bag.npy   (253,) float32
    data/processed/nb969_methodC_oof_mean_bag.npy   (253,) float32
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

TAG = "nb969"
SEEDS = [0, 1, 7, 42, 137]
N_FOLDS = 5

# PU pseudo-pEC50 anchors
PSEUDO_POS = 5.5    # mid-active
PSEUDO_NEG = 4.0    # inactive

# PU thresholds (per spec)
POS_LOG2FC = 0.5
POS_FDR    = 0.05
NEG_LOG2FC = 0.1

# Sample weight for PU rows
PU_WEIGHT  = 0.3

# Reference numbers
NB2103_K28_MEAN_BAG_REF   = 0.4737
NB2103_K28_MEDIAN_BAG_REF = 0.4698
DECISION_MARGIN = 0.003


def _safe_inchikey(mol):
    try:
        return Chem.MolToInchiKey(mol) if mol is not None else None
    except Exception:
        return None


def _lgbm_params(seed: int) -> dict:
    """Same LGBM(MSE) hyperparams as nb2103/nb2081/nb2063."""
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


def _build_pu_corpus(sc_df: pd.DataFrame, train_iks: set) -> pd.DataFrame:
    """Aggregate single-conc rows to compound level, label PU, drop overlaps."""
    print("\n" + "-" * 78)
    print("PU CORPUS CONSTRUCTION")
    print("-" * 78)
    n_rows = len(sc_df)
    n_uniq_smi = sc_df["smiles"].nunique()
    print(f"   single-conc rows: {n_rows}  unique SMILES: {n_uniq_smi}")

    # Row-level PU label counts (informational)
    pos_row = (sc_df["log2_fc_estimate"] > POS_LOG2FC) & (sc_df["fdr_bh"] < POS_FDR)
    neg_row = sc_df["log2_fc_estimate"] < NEG_LOG2FC
    amb_row = ~(pos_row | neg_row)
    print(f"   ROW-LEVEL pos: {pos_row.sum()}  neg: {neg_row.sum()}  "
          f"amb: {amb_row.sum()}")

    # Per-compound aggregate
    agg = (
        sc_df.groupby("smiles", as_index=False)
        .agg(
            log2_fc_max=("log2_fc_estimate", "max"),
            log2_fc_med=("log2_fc_estimate", "median"),
            fdr_min=("fdr_bh", "min"),
            n_rows=("log2_fc_estimate", "count"),
        )
    )
    pos_cpd = (agg["log2_fc_max"] > POS_LOG2FC) & (agg["fdr_min"] < POS_FDR)
    neg_cpd = agg["log2_fc_max"] < NEG_LOG2FC
    amb_cpd = ~(pos_cpd | neg_cpd)

    agg["pu_label"] = "ambiguous"
    agg.loc[pos_cpd, "pu_label"] = "positive"
    agg.loc[neg_cpd, "pu_label"] = "negative"
    print(f"   COMPOUND-LEVEL pos: {pos_cpd.sum()}  neg: {neg_cpd.sum()}  "
          f"amb: {amb_cpd.sum()}  (total {len(agg)})")

    # Drop ambiguous
    pu = agg[agg["pu_label"] != "ambiguous"].reset_index(drop=True)
    print(f"   after dropping ambiguous: {len(pu)} candidates")

    # Standardize -> InChIKey, drop any in TRAIN to avoid double-label leak
    iks = []
    smi_std = []
    keep_mask = []
    for s in pu["smiles"]:
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
    pu["inchikey"] = iks
    pu["std_smiles"] = smi_std
    pu = pu[pd.Series(keep_mask)].reset_index(drop=True)
    print(f"   after RDKit standardize: {len(pu)}")

    n_before = len(pu)
    pu = pu[~pu["inchikey"].isin(train_iks)].reset_index(drop=True)
    print(f"   after dropping TRAIN overlap: {n_before} -> {len(pu)}  "
          f"(dropped {n_before - len(pu)})")

    # Assign pseudo-pec50 anchors
    pu["pseudo_pec50"] = np.where(
        pu["pu_label"] == "positive", PSEUDO_POS, PSEUDO_NEG
    )
    pu["sample_weight"] = PU_WEIGHT

    pu_pos_final = int((pu["pu_label"] == "positive").sum())
    pu_neg_final = int((pu["pu_label"] == "negative").sum())
    print(f"   FINAL PU set: {len(pu)}  pos={pu_pos_final}  neg={pu_neg_final}")
    print(f"   pseudo_pec50 anchors: pos={PSEUDO_POS} neg={PSEUDO_NEG}  "
          f"sample_weight={PU_WEIGHT}")
    return pu


def _bag_cross_fit(
    X_tr_full: np.ndarray,
    y_tr_full: np.ndarray,
    w_tr_full: np.ndarray,
    X_unb: np.ndarray,
    y_unb: np.ndarray,
    n_pu: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray, list[float], dict]:
    """Cross-fit on 253 unblind, treating TRAIN+PU as fixed training data
    that's always included; the 253 are folded into 5 groups, with 4/5
    included in training and 1/5 held out for prediction.

    Returns:
        mean_bag_oof    : (n_unb,) float64 -- mean across seeds
        median_bag_oof  : (n_unb,) float64 -- median across seeds
        per_seed_rae    : list of per-seed RAE
        records         : dict with timings, per-seed records
    """
    n_unb = len(y_unb)
    n_aug = X_tr_full.shape[0]  # TRAIN + PU rows
    n_total_features = X_tr_full.shape[1]
    assert X_unb.shape == (n_unb, n_total_features)

    per_seed_corrected = np.zeros((len(SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []

    for i, s in enumerate(SEEDS):
        ts = time.time()
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=s)
        oof_s = np.full(n_unb, np.nan, dtype=np.float64)
        for tr_loc, va_loc in kf.split(np.arange(n_unb)):
            X_unb_tr = X_unb[tr_loc]
            y_unb_tr = y_unb[tr_loc]
            w_unb_tr = np.ones(len(tr_loc), dtype=np.float32)
            X_full = np.vstack([X_tr_full, X_unb_tr]).astype(np.float32)
            y_full = np.concatenate([y_tr_full, y_unb_tr]).astype(np.float64)
            w_full = np.concatenate([w_tr_full, w_unb_tr]).astype(np.float32)
            mdl = lgb.LGBMRegressor(**_lgbm_params(s))
            mdl.fit(X_full, y_full, sample_weight=w_full)
            oof_s[va_loc] = mdl.predict(X_unb[va_loc])
        per_seed_corrected[i] = oof_s
        rae_s = float(rae(y_unb, oof_s))
        per_seed_rae.append(rae_s)
        per_seed_records.append({
            "seed": int(s),
            "rae": rae_s,
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   [{label}] seed={s:3d}  rae={rae_s:.4f}  "
              f"wall={time.time() - ts:.1f}s  (n_aug={n_aug}, n_pu={n_pu})")

    mean_bag = per_seed_corrected.mean(axis=0)
    median_bag = np.median(per_seed_corrected, axis=0)
    return mean_bag, median_bag, per_seed_rae, {"per_seed_records": per_seed_records}


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- PU framing on 21k single-conc screen")
    print(f"          ref nb2103 K=28 mean-bag = {NB2103_K28_MEAN_BAG_REF:.4f}  "
          f"median-bag = {NB2103_K28_MEDIAN_BAG_REF:.4f}  "
          f"margin = {DECISION_MARGIN}")
    print("=" * 78)

    # ---- Load 253 unblind truth (use audit cache) ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] unb_idx={unb_idx.shape}  y_unb={y_unb.shape}")

    # ---- Load TRAIN (4139) + TEST (513) ----
    tr = load_train()
    te = load_test()
    sc = load_single_conc()
    print(f"[load] train rows={len(tr)}  test rows={len(te)}  "
          f"single-conc rows={len(sc)}")

    # Train InChIKey set (to filter PU overlap)
    train_iks = set()
    for s in tr["smiles"]:
        m = standardize(s)
        if m is not None:
            train_iks.add(_safe_inchikey(m))
    train_iks.discard(None)
    print(f"[load] train InChIKeys: {len(train_iks)}")

    # ---- Build PU corpus ----
    pu = _build_pu_corpus(sc, train_iks)
    n_pu_pos = int((pu["pu_label"] == "positive").sum())
    n_pu_neg = int((pu["pu_label"] == "negative").sum())
    n_pu = len(pu)

    # ---- Featurize ----
    print("\n" + "-" * 78)
    print("FEATURIZATION (combined: Morgan 2048 + RDKit ~217)")
    print("-" * 78)
    tf = time.time()
    X_train_raw = combined(tr["smiles"].tolist())
    print(f"   X_train raw: {X_train_raw.shape}  "
          f"({time.time() - tf:.1f}s)")
    tf = time.time()
    X_test_raw = combined(te["smiles"].tolist())
    print(f"   X_test raw:  {X_test_raw.shape}  ({time.time() - tf:.1f}s)")
    tf = time.time()
    X_pu_raw = combined(pu["std_smiles"].tolist())
    print(f"   X_pu raw:    {X_pu_raw.shape}  ({time.time() - tf:.1f}s)")

    # Concatenated impute (so all share the same column medians)
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

    # Slice the 253 unblind from the 513 test feature matrix
    X_unb = X_test[unb_idx].astype(np.float32)

    # ---- Labels & weights ----
    y_train = tr["pec50"].to_numpy(dtype=np.float64)
    w_train = np.ones(n_train, dtype=np.float32)
    y_pu = pu["pseudo_pec50"].to_numpy(dtype=np.float64)
    w_pu = pu["sample_weight"].to_numpy(dtype=np.float32)

    # ---- Method C: baseline (TRAIN only, no PU) ----
    print("\n" + "=" * 78)
    print("METHOD C (BASELINE: TRAIN only, no PU)")
    print("=" * 78)
    mean_bag_C, median_bag_C, per_seed_C, rec_C = _bag_cross_fit(
        X_tr_full=X_train,
        y_tr_full=y_train,
        w_tr_full=w_train,
        X_unb=X_unb,
        y_unb=y_unb,
        n_pu=0,
        label="C",
    )
    rae_mean_bag_C = float(rae(y_unb, mean_bag_C))
    rae_median_bag_C = float(rae(y_unb, median_bag_C))
    np.save(DATA_PROCESSED / f"{TAG}_methodC_oof_mean_bag.npy",
            mean_bag_C.astype(np.float32))
    print(f"   [C] mean-bag RAE   = {rae_mean_bag_C:.4f}  "
          f"median-bag RAE = {rae_median_bag_C:.4f}  "
          f"per-seed = [{', '.join(f'{r:.4f}' for r in per_seed_C)}]")

    # ---- Method A: PU-augmented single task ----
    print("\n" + "=" * 78)
    print("METHOD A (PU-AUGMENTED SINGLE-TASK)")
    print("=" * 78)
    X_aug_A = np.vstack([X_train, X_pu]).astype(np.float32)
    y_aug_A = np.concatenate([y_train, y_pu]).astype(np.float64)
    w_aug_A = np.concatenate([w_train, w_pu]).astype(np.float32)
    print(f"   augmented corpus size: {X_aug_A.shape[0]} "
          f"(TRAIN {n_train} + PU {n_pu})")
    mean_bag_A, median_bag_A, per_seed_A, rec_A = _bag_cross_fit(
        X_tr_full=X_aug_A,
        y_tr_full=y_aug_A,
        w_tr_full=w_aug_A,
        X_unb=X_unb,
        y_unb=y_unb,
        n_pu=n_pu,
        label="A",
    )
    rae_mean_bag_A = float(rae(y_unb, mean_bag_A))
    rae_median_bag_A = float(rae(y_unb, median_bag_A))
    np.save(DATA_PROCESSED / f"{TAG}_methodA_oof_mean_bag.npy",
            mean_bag_A.astype(np.float32))
    print(f"   [A] mean-bag RAE   = {rae_mean_bag_A:.4f}  "
          f"median-bag RAE = {rae_median_bag_A:.4f}  "
          f"per-seed = [{', '.join(f'{r:.4f}' for r in per_seed_A)}]")

    # ---- Method B: PU as AUXILIARY task -> extra feature ----
    print("\n" + "=" * 78)
    print("METHOD B (PU AUX HEAD: train aux LGBM on PU, use prediction as feature)")
    print("=" * 78)
    # Aux model: train on PU rows only, predict pseudo_pec50.
    # Use seed=0 (fixed) for the aux model so it's stable across the bag.
    t_aux = time.time()
    aux_params = _lgbm_params(0)
    aux_mdl = lgb.LGBMRegressor(**aux_params)
    aux_mdl.fit(X_pu, y_pu)
    aux_pred_train = aux_mdl.predict(X_train).astype(np.float32)
    aux_pred_unb = aux_mdl.predict(X_unb).astype(np.float32)
    print(f"   aux LGBM trained on {len(X_pu)} PU rows  "
          f"wall={time.time() - t_aux:.1f}s")
    print(f"   aux preds: train mean={aux_pred_train.mean():.3f} "
          f"std={aux_pred_train.std():.3f}  "
          f"unb mean={aux_pred_unb.mean():.3f} std={aux_pred_unb.std():.3f}")

    X_train_B = np.hstack([X_train, aux_pred_train.reshape(-1, 1)]).astype(np.float32)
    X_unb_B = np.hstack([X_unb, aux_pred_unb.reshape(-1, 1)]).astype(np.float32)
    print(f"   X_train_B = {X_train_B.shape}  X_unb_B = {X_unb_B.shape}")
    mean_bag_B, median_bag_B, per_seed_B, rec_B = _bag_cross_fit(
        X_tr_full=X_train_B,
        y_tr_full=y_train,
        w_tr_full=w_train,
        X_unb=X_unb_B,
        y_unb=y_unb,
        n_pu=n_pu,
        label="B",
    )
    rae_mean_bag_B = float(rae(y_unb, mean_bag_B))
    rae_median_bag_B = float(rae(y_unb, median_bag_B))
    np.save(DATA_PROCESSED / f"{TAG}_methodB_oof_mean_bag.npy",
            mean_bag_B.astype(np.float32))
    print(f"   [B] mean-bag RAE   = {rae_mean_bag_B:.4f}  "
          f"median-bag RAE = {rae_median_bag_B:.4f}  "
          f"per-seed = [{', '.join(f'{r:.4f}' for r in per_seed_B)}]")

    # ---- Verdicts ----
    def _verdict(rae_x: float, label: str) -> str:
        d_vs_nb2103 = rae_x - NB2103_K28_MEAN_BAG_REF
        d_vs_C = rae_x - rae_mean_bag_C
        if rae_x < NB2103_K28_MEAN_BAG_REF - DECISION_MARGIN:
            return f"{label}_BEATS_NB2103_K28"
        if abs(d_vs_nb2103) < DECISION_MARGIN:
            return f"{label}_FLAT_VS_NB2103_K28"
        if rae_x < rae_mean_bag_C - DECISION_MARGIN:
            return f"{label}_BEATS_BASELINE_C_BUT_WORSE_THAN_NB2103"
        if abs(d_vs_C) < DECISION_MARGIN:
            return f"{label}_FLAT_VS_BASELINE_C"
        return f"{label}_HURTS_VS_BASELINE_C"

    verdict_A = _verdict(rae_mean_bag_A, "A_PU_AUG")
    verdict_B = _verdict(rae_mean_bag_B, "B_PU_AUX")
    verdict_C = _verdict(rae_mean_bag_C, "C_BASELINE")

    print("\n" + "=" * 78)
    print("SUMMARY TABLE")
    print("=" * 78)
    print(f"   {'method':<25s}  {'mean_bag':>10s}  {'d_vs_C':>9s}  "
          f"{'d_vs_nb2103_K28':>16s}  verdict")
    print(f"   {'nb2103_K28 (ref)':<25s}  "
          f"{NB2103_K28_MEAN_BAG_REF:>10.4f}  "
          f"{NB2103_K28_MEAN_BAG_REF - rae_mean_bag_C:>+9.4f}  "
          f"{0.0:>+16.4f}  REFERENCE")
    print(f"   {'C_baseline (TRAIN only)':<25s}  "
          f"{rae_mean_bag_C:>10.4f}  {0.0:>+9.4f}  "
          f"{rae_mean_bag_C - NB2103_K28_MEAN_BAG_REF:>+16.4f}  {verdict_C}")
    print(f"   {'A_PU_aug':<25s}  "
          f"{rae_mean_bag_A:>10.4f}  "
          f"{rae_mean_bag_A - rae_mean_bag_C:>+9.4f}  "
          f"{rae_mean_bag_A - NB2103_K28_MEAN_BAG_REF:>+16.4f}  {verdict_A}")
    print(f"   {'B_PU_aux':<25s}  "
          f"{rae_mean_bag_B:>10.4f}  "
          f"{rae_mean_bag_B - rae_mean_bag_C:>+9.4f}  "
          f"{rae_mean_bag_B - NB2103_K28_MEAN_BAG_REF:>+16.4f}  {verdict_B}")

    # Global verdict: best of A/B vs nb2103
    best_method, best_rae = max(
        [("A_PU_AUG", -rae_mean_bag_A),
         ("B_PU_AUX", -rae_mean_bag_B),
         ("C_BASELINE", -rae_mean_bag_C)],
        key=lambda t: t[1]
    )
    best_rae = -best_rae
    if best_rae < NB2103_K28_MEAN_BAG_REF - DECISION_MARGIN:
        global_verdict = f"PU_BEATS_NB2103_K28_AT_{best_method}"
    elif abs(best_rae - NB2103_K28_MEAN_BAG_REF) < DECISION_MARGIN:
        global_verdict = f"PU_FLAT_VS_NB2103_K28_BEST_{best_method}"
    else:
        global_verdict = f"PU_DOES_NOT_BEAT_NB2103_K28_BEST_{best_method}_{best_rae:.4f}"
    print(f"\n   global verdict   = {global_verdict}")

    summary = {
        "tag": TAG,
        "method": "pu_framing_21k_single_conc",
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "data_source": (
            "data/raw/pxr-challenge_single_concentration_TRAIN.csv "
            "(21003 rows) -- PU labels on log2_fc_estimate + fdr_bh"
        ),
        "model_family": "LightGBM",
        "lgbm_objective": "regression",
        "lgbm_max_depth": 4,
        "lgbm_num_leaves": 15,
        "lgbm_n_estimators": 300,
        "lgbm_learning_rate": 0.03,
        "lgbm_min_child_samples": 5,
        "lgbm_reg_lambda": 2.0,
        "feat_dim": int(feat_dim),
        "feat_source": "combined (Morgan 2048 + RDKit ~217)",
        "seeds": SEEDS,
        "folds": N_FOLDS,
        "n_unb": int(n_unb),
        "n_train": int(n_train),
        "n_pu": int(n_pu),
        "n_pu_pos": int(n_pu_pos),
        "n_pu_neg": int(n_pu_neg),
        "n_aug_method_A": int(n_train + n_pu),
        "pseudo_pec50_positive": PSEUDO_POS,
        "pseudo_pec50_negative": PSEUDO_NEG,
        "pu_sample_weight": PU_WEIGHT,
        "pu_thresholds": {
            "positive_log2fc_min": POS_LOG2FC,
            "positive_fdr_max": POS_FDR,
            "negative_log2fc_max": NEG_LOG2FC,
        },
        "nb2103_K28_mean_bag_ref": NB2103_K28_MEAN_BAG_REF,
        "nb2103_K28_median_bag_ref": NB2103_K28_MEDIAN_BAG_REF,
        "decision_margin": DECISION_MARGIN,
        "method_A_PU_aug": {
            "per_seed_rae": per_seed_A,
            "per_seed_rae_mean": float(np.mean(per_seed_A)),
            "per_seed_rae_std": float(np.std(per_seed_A)),
            "rae_mean_bag": rae_mean_bag_A,
            "rae_median_bag": rae_median_bag_A,
            "delta_vs_baseline_C": rae_mean_bag_A - rae_mean_bag_C,
            "delta_vs_nb2103_K28": rae_mean_bag_A - NB2103_K28_MEAN_BAG_REF,
            "verdict": verdict_A,
            **rec_A,
        },
        "method_B_PU_aux": {
            "per_seed_rae": per_seed_B,
            "per_seed_rae_mean": float(np.mean(per_seed_B)),
            "per_seed_rae_std": float(np.std(per_seed_B)),
            "rae_mean_bag": rae_mean_bag_B,
            "rae_median_bag": rae_median_bag_B,
            "delta_vs_baseline_C": rae_mean_bag_B - rae_mean_bag_C,
            "delta_vs_nb2103_K28": rae_mean_bag_B - NB2103_K28_MEAN_BAG_REF,
            "verdict": verdict_B,
            "aux_pred_train_mean": float(aux_pred_train.mean()),
            "aux_pred_train_std": float(aux_pred_train.std()),
            "aux_pred_unb_mean": float(aux_pred_unb.mean()),
            "aux_pred_unb_std": float(aux_pred_unb.std()),
            **rec_B,
        },
        "method_C_baseline": {
            "per_seed_rae": per_seed_C,
            "per_seed_rae_mean": float(np.mean(per_seed_C)),
            "per_seed_rae_std": float(np.std(per_seed_C)),
            "rae_mean_bag": rae_mean_bag_C,
            "rae_median_bag": rae_median_bag_C,
            "delta_vs_nb2103_K28": rae_mean_bag_C - NB2103_K28_MEAN_BAG_REF,
            "verdict": verdict_C,
            **rec_C,
        },
        "best_method": best_method,
        "best_rae_mean_bag": best_rae,
        "global_verdict": global_verdict,
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
    print(f"  n_pu_pos = {res['n_pu_pos']}")
    print(f"  n_pu_neg = {res['n_pu_neg']}")
    print(f"  n_aug_method_A = {res['n_aug_method_A']}")
    print(f"  C_baseline    : {res['method_C_baseline']['rae_mean_bag']:.4f}")
    print(f"  A_PU_aug      : {res['method_A_PU_aug']['rae_mean_bag']:.4f}  "
          f"(d_vs_nb2103 {res['method_A_PU_aug']['delta_vs_nb2103_K28']:+.4f})")
    print(f"  B_PU_aux      : {res['method_B_PU_aux']['rae_mean_bag']:.4f}  "
          f"(d_vs_nb2103 {res['method_B_PU_aux']['delta_vs_nb2103_K28']:+.4f})")
    print(f"  nb2103 K=28 ref: {res['nb2103_K28_mean_bag_ref']:.4f}")
    print(f"  verdict        : {res['global_verdict']}")
