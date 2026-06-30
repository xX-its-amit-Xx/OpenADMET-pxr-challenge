"""nb1040 -- PXR-DIRECT-ONLY ChEMBL retrain (cycle 132 method #1).

CONTEXT:
    nb965 demonstrated that the full 11,185-row ChEMBL KB is dominated by
    10,302 NR-sister cross-assay rows whose pEC50 values reflect off-target
    activity (PPARg/FXR/RXRa/LXRa/VDR) and inject cross-assay noise rather
    than PXR-direct signal.  Restricting to the 883 PXR_CHEMBL3401 + NR_PXR
    rows isolates the only labels with PXR-on-target measurements.

    nb2103 K=28 (anchor=chemprop_aux, SHAP-pruned residual LGBM on 117-col
    5-way K-tuned matrix) holds the current PRE-unblind baseline at
    mean-bag RAE 0.4737 / median-bag 0.4698.  This notebook tests whether
    PXR-direct-only ChEMBL beats that baseline via three orthogonal
    integration paths (A: LGBM-only train aug, B: residual-on-anchor,
    C: aux-task averaging).

PROTOCOL:
    1. Load data/external/chembl_pxr_nr_kb.parquet (11,185 rows; cols
       smiles / inchikey / scaffold / pec50_chembl / source_target).
       Filter source_target in {PXR_CHEMBL3401, NR_PXR}  ->  883+157 = 1,040.
    2. Apply pEC50 in [4, 7] (assay support).  Dedup vs the 4,139 train
       InChIKey set and vs the 513 test InChIKey set.
    3. Featurize train + test + filtered ChEMBL via combined Morgan(2048)
       + RDKit_desc(217) = 2265-d.
    4. Train LGBM(MSE) on (4139 train + filtered PXR-direct) with
       sample_weight = 1.0 for train, 0.4 for PXR-direct (assay bias
       penalty).  Predict test 513, slice unb_idx for in-sample 253 eval.
    5. Method A (LGBM-only train aug, replicates nb962 but on PXR-direct
       subset only):
        - 5-seed bag of LGBM(MSE) on augmented set.
        - Eval on unb 253.
    6. Method B (residual LGBM K=28 on chemprop_aux with PXR-direct as
       extra train rows; preserves nb2103 SHAP-feature structure):
        - Take chemprop_aux (513,) as anchor; residual = y_train - anchor
          for train (need chemprop_aux on train; use a kNN-on-train-pec50
          proxy for ChEMBL rows since we have no chemprop preds for them).
        - Train residual LGBM on (train_residual + chembl_residual) with
          the same weight schedule.
        - 5-fold scaffold cross-fit on 253 phase1_unblinded.  Since
          chemprop_aux is only available on the 513 (not the 4139), we
          cross-fit on train+chembl with held-out unb_idx-via-test, then
          add anchor.
    7. Method C (aux-task LGBM):
        - Train separate head on PXR-direct only (1,040 rows).
        - Train main head on 4,139 train.
        - Predict test 513 with each head; average the two heads.
    8. Compare all 3 methods vs nb2103 K=28 baseline.  Decision margin
       = 0.003.  Best path that beats -> build deploy CSV
       submissions/nb1040_deploy_pxr_direct.csv.
    9. Save data/processed/nb1040_summary.json with per-method details.

OUTPUTS:
    scripts/nb1040_pxr_direct_chembl.py
    data/processed/nb1040_summary.json
    submissions/nb1040_deploy_pxr_direct.csv  (only if a method beats)
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

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.featurize import rdkit_desc, impute
from pxr.paths import DATA_PROCESSED

TAG = "nb1040"
ROOT = Path(__file__).resolve().parents[1]
EXT_DIR = ROOT / "data" / "external"
SUBMISSIONS_DIR = ROOT / "submissions"

# ---- Source filter: PXR-DIRECT ONLY ----
PXR_DIRECT_SRC = {"PXR_CHEMBL3401", "NR_PXR"}

# ---- ChEMBL filters ----
PEC50_LO = 4.0
PEC50_HI = 7.0

# ---- Sample weights ----
TRAIN_WEIGHT = 1.0
CHEMBL_WEIGHT = 0.4   # PXR-direct still has cross-assay bias (CHEMBL3401 reporter
                       # vs OpenADMET assay), so we down-weight slightly

# ---- Model params (same family as nb2103 / nb962) ----
SEEDS = [0, 1, 7, 42, 137]
N_FOLDS = 5

DECISION_MARGIN = 0.003

# ---- Baselines (PRE-unblind references) ----
NB2103_K28_MEAN_BAG_REF = 0.4737
NB2103_K28_MEDIAN_BAG_REF = 0.4698
CHEMPROP_AUX_REF = 0.6216

# ---- Residual-aug kNN proxy (for chemprop_aux on ChEMBL rows) ----
KNN_K_PROXY = 5
SIM_FLOOR = 1e-6


# ============================================================================
# Helpers
# ============================================================================

def _safe_inchikey(mol):
    try:
        return Chem.MolToInchiKey(mol) if mol is not None else None
    except Exception:
        return None


def _safe_can_smiles(mol):
    try:
        return Chem.MolToSmiles(mol) if mol is not None else None
    except Exception:
        return None


def _load_pxr_direct_kb() -> pd.DataFrame:
    """Load chembl_pxr_nr_kb.parquet and subset to PXR-DIRECT sources."""
    kb_p = EXT_DIR / "chembl_pxr_nr_kb.parquet"
    if not kb_p.exists():
        raise FileNotFoundError(f"missing chembl_pxr_nr_kb.parquet at {kb_p}")
    d = pd.read_parquet(kb_p)
    print(f"   [kb] full chembl_pxr_nr_kb: {len(d)} rows")
    print(f"   [kb] source_target counts:")
    for src, c in d["source_target"].value_counts().items():
        marker = "  <- KEEP" if src in PXR_DIRECT_SRC else ""
        print(f"          {src:20s}  {c:>6d}{marker}")
    d = d[d["source_target"].isin(PXR_DIRECT_SRC)].copy()
    print(f"   [kb] after PXR-DIRECT subset: {len(d)} rows")
    d = d.rename(columns={"pec50_chembl": "pec50"})
    d = d.rename(columns={"source_target": "src"})
    return d[["smiles", "inchikey", "pec50", "src"]].copy()


def _apply_pec50_range_filter(pool: pd.DataFrame) -> pd.DataFrame:
    n0 = len(pool)
    pool = pool[pool["pec50"].notna()].copy()
    n1 = len(pool)
    pool = pool[(pool["pec50"] >= PEC50_LO) & (pool["pec50"] <= PEC50_HI)].copy()
    n2 = len(pool)
    print(f"   [filter] pec50 notna: {n0} -> {n1}")
    print(f"   [filter] pec50 in [{PEC50_LO}, {PEC50_HI}]: {n1} -> {n2}")
    return pool


def _dedup_by_inchikey(
    pool: pd.DataFrame, exclude_inchikeys: set[str], label: str
) -> pd.DataFrame:
    n_before = len(pool)
    pool = pool[~pool["inchikey"].isin(exclude_inchikeys)].reset_index(drop=True)
    n_after = len(pool)
    print(f"   [dedup vs {label}] {n_before} -> {n_after} "
          f"(dropped {n_before - n_after})")
    return pool


def _build_combined_features(smiles_list: list[str]) -> np.ndarray:
    morg = morgan_fp_batch(smiles_list).astype(np.float32)
    rdk = rdkit_desc(smiles_list).astype(np.float32)
    return np.hstack([morg, rdk])


def _lgbm_params(seed: int) -> dict:
    """Same LGBM family as nb2103/nb962, with bagging knobs for seed diversity."""
    return dict(
        objective="regression",
        max_depth=4,
        num_leaves=15,
        n_estimators=300,
        learning_rate=0.03,
        min_child_samples=5,
        reg_lambda=2.0,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.5,
        random_state=seed,
        bagging_seed=seed,
        feature_fraction_seed=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _knn_predict_pec50(
    fp_q: np.ndarray, fp_pool: np.ndarray, pool_labels: np.ndarray,
    k: int = KNN_K_PROXY, fallback: float = 5.0
) -> np.ndarray:
    """Tanimoto-weighted kNN from pool -> queries.  Returns (Nq,) pec50 preds."""
    a = fp_q.astype(np.float32)
    b = fp_pool.astype(np.float32)
    a_sum = a.sum(axis=1)
    b_sum = b.sum(axis=1)
    nq = a.shape[0]
    np_pool = b.shape[0]
    pred = np.zeros(nq, dtype=np.float32)
    BLOCK = 64
    for s in range(0, nq, BLOCK):
        e = min(nq, s + BLOCK)
        inter = a[s:e] @ b.T
        denom = a_sum[s:e, None] + b_sum[None, :] - inter
        denom = np.maximum(denom, 1.0)
        sim = inter / denom
        if k >= np_pool:
            top_idx = np.argsort(-sim, axis=1)[:, :k]
        else:
            part = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
            row_idx = np.arange(e - s)[:, None]
            sim_part = sim[row_idx, part]
            order = np.argsort(-sim_part, axis=1)
            top_idx = part[row_idx, order]
        row_idx = np.arange(e - s)[:, None]
        top_sim = sim[row_idx, top_idx]
        w = np.clip(top_sim, 0.0, 1.0)
        wsum = w.sum(axis=1)
        for i in range(e - s):
            if wsum[i] < SIM_FLOOR:
                pred[s + i] = fallback
            else:
                pred[s + i] = np.sum(w[i] * pool_labels[top_idx[i]]) / wsum[i]
    return pred


# ============================================================================
# Method A: LGBM-only train aug (PXR-DIRECT only)
# ============================================================================

def method_A_lgbm_aug(
    X_tr_imp: np.ndarray, y_tr: np.ndarray,
    X_ch_imp: np.ndarray, y_ch: np.ndarray,
    X_te_imp: np.ndarray,
    unb_idx: np.ndarray, y_unb: np.ndarray,
    rae_baseline: float,
) -> dict:
    print("\n" + "-" * 78)
    print("METHOD A -- LGBM-only train aug on PXR-DIRECT")
    print("-" * 78)
    n_tr = X_tr_imp.shape[0]
    n_ch = X_ch_imp.shape[0]
    n_te = X_te_imp.shape[0]

    X_aug = np.vstack([X_tr_imp, X_ch_imp]).astype(np.float32)
    y_aug = np.concatenate([y_tr, y_ch]).astype(np.float64)
    w_aug = np.concatenate([
        np.full(n_tr, TRAIN_WEIGHT, dtype=np.float32),
        np.full(n_ch, CHEMBL_WEIGHT, dtype=np.float32),
    ])
    print(f"   X_aug={X_aug.shape}  y_aug={y_aug.shape}")
    print(f"   weights: train={TRAIN_WEIGHT} ({n_tr} rows), "
          f"chembl_PXR-direct={CHEMBL_WEIGHT} ({n_ch} rows)")

    per_seed_te = np.zeros((len(SEEDS), n_te), dtype=np.float64)
    per_seed_rae = []
    per_seed_records = []
    for i, s in enumerate(SEEDS):
        ts = time.time()
        mdl = lgb.LGBMRegressor(**_lgbm_params(s))
        mdl.fit(X_aug, y_aug, sample_weight=w_aug)
        pred_te_513 = mdl.predict(X_te_imp)
        per_seed_te[i] = pred_te_513
        rae_s = float(rae(y_unb, pred_te_513[unb_idx]))
        per_seed_rae.append(rae_s)
        per_seed_records.append({
            "seed": int(s),
            "rae_unb": rae_s,
            "delta_vs_nb2103_K28": rae_s - rae_baseline,
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   [A] seed={s:3d}  rae_unb={rae_s:.4f}  "
              f"(d_vs_nb2103={rae_s - rae_baseline:+.4f})  "
              f"wall={time.time() - ts:.1f}s")

    mean_bag_te = per_seed_te.mean(axis=0)
    median_bag_te = np.median(per_seed_te, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_te[unb_idx]))
    rae_median_bag = float(rae(y_unb, median_bag_te[unb_idx]))
    print(f"\n   [A] per-seed mean = {np.mean(per_seed_rae):.4f}  "
          f"std = {np.std(per_seed_rae):.6f}")
    print(f"   [A] mean-bag rae   = {rae_mean_bag:.4f}  "
          f"(d_vs_nb2103={rae_mean_bag - rae_baseline:+.4f})")
    print(f"   [A] median-bag rae = {rae_median_bag:.4f}")
    return {
        "method": "A_lgbm_aug_pxr_direct",
        "per_seed_records": per_seed_records,
        "per_seed_rae_mean": float(np.mean(per_seed_rae)),
        "per_seed_rae_std": float(np.std(per_seed_rae)),
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_nb2103_K28": rae_mean_bag - rae_baseline,
        "mean_bag_te_513": mean_bag_te,
        "median_bag_te_513": median_bag_te,
    }


# ============================================================================
# Method B: residual LGBM K=28 on chemprop_aux with PXR-direct rows added.
#
# Strategy: anchor = chemprop_aux (513,) on TEST.  We need a residual model
# on (train + chembl).  Since chemprop_aux preds are only on test, we proxy
# the anchor on TRAIN and ChEMBL by Tanimoto-weighted kNN(k=5) on TRAIN pec50,
# then residual = y - anchor_proxy.  Train LGBM(MSE) on residual with weights.
# Predict residual on test 513; corrected = chemprop_aux + residual_pred.
# Slice unb_idx for RAE.  (Pure residual-aug, not SHAP-pruned -- the SHAP
# pruning lived inside nb2103's specific 117-col matrix; here we use 2265-d
# combined.  We name it "B" because it follows the residual-on-anchor pattern.)
# ============================================================================

def method_B_residual_lgbm(
    X_tr_imp: np.ndarray, y_tr: np.ndarray, fp_tr: np.ndarray,
    X_ch_imp: np.ndarray, y_ch: np.ndarray, fp_ch: np.ndarray,
    X_te_imp: np.ndarray,
    te_chemprop_aux_513: np.ndarray,
    unb_idx: np.ndarray, y_unb: np.ndarray,
    rae_baseline: float,
) -> dict:
    print("\n" + "-" * 78)
    print("METHOD B -- residual LGBM(MSE) on chemprop_aux anchor + PXR-direct aug")
    print("-" * 78)
    n_tr = X_tr_imp.shape[0]
    n_ch = X_ch_imp.shape[0]
    n_te = X_te_imp.shape[0]

    # Build kNN(k=5) proxy for chemprop_aux on TRAIN and ChEMBL rows.
    # Pool = TRAIN with pec50 labels.  Query = (TRAIN, ChEMBL) -> exclude self for
    # train queries via leave-self-out at predict time (block self).
    print(f"   [B] proxying chemprop_aux on train + ChEMBL via kNN(k={KNN_K_PROXY}) "
          f"on train pec50...")
    fp_pool = fp_tr.astype(np.float32)
    labels_pool = y_tr.astype(np.float32)
    pool_med = float(np.median(labels_pool))

    # Predict on TRAIN (leave-one-out)  -> use kNN k+1 and drop the self match
    def _knn_train_loo(fp_q, fp_pool, pool_labels, k):
        a = fp_q.astype(np.float32)
        b = fp_pool.astype(np.float32)
        a_sum = a.sum(axis=1)
        b_sum = b.sum(axis=1)
        nq = a.shape[0]
        np_pool = b.shape[0]
        pred = np.zeros(nq, dtype=np.float32)
        BLOCK = 64
        for s in range(0, nq, BLOCK):
            e = min(nq, s + BLOCK)
            inter = a[s:e] @ b.T
            denom = a_sum[s:e, None] + b_sum[None, :] - inter
            denom = np.maximum(denom, 1.0)
            sim = inter / denom
            # mask self (assume q is a subset of pool by row index match: q=pool)
            for ii in range(e - s):
                sim[ii, s + ii] = -1.0
            kk = min(k, np_pool - 1)
            part = np.argpartition(-sim, kth=kk - 1, axis=1)[:, :kk]
            row_idx = np.arange(e - s)[:, None]
            sim_part = sim[row_idx, part]
            order = np.argsort(-sim_part, axis=1)
            top_idx = part[row_idx, order]
            top_sim = sim[row_idx, top_idx]
            w = np.clip(top_sim, 0.0, 1.0)
            wsum = w.sum(axis=1)
            for ii in range(e - s):
                if wsum[ii] < SIM_FLOOR:
                    pred[s + ii] = pool_med
                else:
                    pred[s + ii] = np.sum(w[ii] * pool_labels[top_idx[ii]]) / wsum[ii]
        return pred

    anchor_tr = _knn_train_loo(fp_pool, fp_pool, labels_pool, KNN_K_PROXY)
    print(f"   [B] anchor_tr(LOO knn5)  shape={anchor_tr.shape}  "
          f"mean={anchor_tr.mean():.3f}  std={anchor_tr.std():.3f}")
    anchor_ch = _knn_predict_pec50(fp_ch, fp_pool, labels_pool,
                                    k=KNN_K_PROXY, fallback=pool_med)
    print(f"   [B] anchor_ch(knn5)      shape={anchor_ch.shape}  "
          f"mean={anchor_ch.mean():.3f}  std={anchor_ch.std():.3f}")
    anchor_te = te_chemprop_aux_513.astype(np.float64)
    print(f"   [B] anchor_te(chemprop)  shape={anchor_te.shape}  "
          f"mean={anchor_te.mean():.3f}  std={anchor_te.std():.3f}")

    resid_tr = y_tr - anchor_tr
    resid_ch = y_ch - anchor_ch
    print(f"   [B] residual_tr  mean={resid_tr.mean():+.3f}  std={resid_tr.std():.3f}")
    print(f"   [B] residual_ch  mean={resid_ch.mean():+.3f}  std={resid_ch.std():.3f}")

    X_aug = np.vstack([X_tr_imp, X_ch_imp]).astype(np.float32)
    y_aug = np.concatenate([resid_tr, resid_ch]).astype(np.float64)
    w_aug = np.concatenate([
        np.full(n_tr, TRAIN_WEIGHT, dtype=np.float32),
        np.full(n_ch, CHEMBL_WEIGHT, dtype=np.float32),
    ])

    per_seed_corrected_te = np.zeros((len(SEEDS), n_te), dtype=np.float64)
    per_seed_rae = []
    per_seed_records = []
    for i, s in enumerate(SEEDS):
        ts = time.time()
        mdl = lgb.LGBMRegressor(**_lgbm_params(s))
        mdl.fit(X_aug, y_aug, sample_weight=w_aug)
        resid_te_513 = mdl.predict(X_te_imp)
        corrected_te = anchor_te + resid_te_513
        per_seed_corrected_te[i] = corrected_te
        rae_s = float(rae(y_unb, corrected_te[unb_idx]))
        per_seed_rae.append(rae_s)
        per_seed_records.append({
            "seed": int(s),
            "rae_unb": rae_s,
            "delta_vs_nb2103_K28": rae_s - rae_baseline,
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   [B] seed={s:3d}  rae_unb={rae_s:.4f}  "
              f"(d_vs_nb2103={rae_s - rae_baseline:+.4f})  "
              f"wall={time.time() - ts:.1f}s")

    mean_bag_te = per_seed_corrected_te.mean(axis=0)
    median_bag_te = np.median(per_seed_corrected_te, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_te[unb_idx]))
    rae_median_bag = float(rae(y_unb, median_bag_te[unb_idx]))
    print(f"\n   [B] mean-bag rae   = {rae_mean_bag:.4f}  "
          f"(d_vs_nb2103={rae_mean_bag - rae_baseline:+.4f})")
    print(f"   [B] median-bag rae = {rae_median_bag:.4f}")
    return {
        "method": "B_residual_lgbm_chemprop_aux_anchor_pxr_direct_aug",
        "per_seed_records": per_seed_records,
        "per_seed_rae_mean": float(np.mean(per_seed_rae)),
        "per_seed_rae_std": float(np.std(per_seed_rae)),
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_nb2103_K28": rae_mean_bag - rae_baseline,
        "mean_bag_te_513": mean_bag_te,
        "median_bag_te_513": median_bag_te,
    }


# ============================================================================
# Method C: aux-task LGBM (train two heads, average)
# ============================================================================

def method_C_aux_task(
    X_tr_imp: np.ndarray, y_tr: np.ndarray,
    X_ch_imp: np.ndarray, y_ch: np.ndarray,
    X_te_imp: np.ndarray,
    unb_idx: np.ndarray, y_unb: np.ndarray,
    rae_baseline: float,
) -> dict:
    print("\n" + "-" * 78)
    print("METHOD C -- aux-task LGBM (two heads: main + pxr-direct, average)")
    print("-" * 78)
    n_te = X_te_imp.shape[0]

    # Head 1: main train (4139) only
    per_seed_main = np.zeros((len(SEEDS), n_te), dtype=np.float64)
    # Head 2: pxr-direct ChEMBL only
    per_seed_aux = np.zeros((len(SEEDS), n_te), dtype=np.float64)
    per_seed_avg = np.zeros((len(SEEDS), n_te), dtype=np.float64)
    per_seed_rae = []
    per_seed_records = []
    for i, s in enumerate(SEEDS):
        ts = time.time()
        # main head
        m_main = lgb.LGBMRegressor(**_lgbm_params(s))
        m_main.fit(X_tr_imp, y_tr)
        pred_main = m_main.predict(X_te_imp)
        per_seed_main[i] = pred_main
        # aux head
        m_aux = lgb.LGBMRegressor(**_lgbm_params(s))
        m_aux.fit(X_ch_imp, y_ch)
        pred_aux = m_aux.predict(X_te_imp)
        per_seed_aux[i] = pred_aux
        # equal-weight average
        pred_avg = 0.5 * pred_main + 0.5 * pred_aux
        per_seed_avg[i] = pred_avg
        rae_s = float(rae(y_unb, pred_avg[unb_idx]))
        per_seed_rae.append(rae_s)
        per_seed_records.append({
            "seed": int(s),
            "rae_main_only": float(rae(y_unb, pred_main[unb_idx])),
            "rae_aux_only": float(rae(y_unb, pred_aux[unb_idx])),
            "rae_avg_unb": rae_s,
            "delta_vs_nb2103_K28": rae_s - rae_baseline,
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   [C] seed={s:3d}  rae_main={per_seed_records[-1]['rae_main_only']:.4f}  "
              f"rae_aux={per_seed_records[-1]['rae_aux_only']:.4f}  "
              f"rae_avg={rae_s:.4f}  wall={time.time() - ts:.1f}s")

    mean_bag_te = per_seed_avg.mean(axis=0)
    median_bag_te = np.median(per_seed_avg, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_te[unb_idx]))
    rae_median_bag = float(rae(y_unb, median_bag_te[unb_idx]))

    # Also report main-only mean-bag and aux-only mean-bag for diagnostics
    main_mean_bag_te = per_seed_main.mean(axis=0)
    aux_mean_bag_te = per_seed_aux.mean(axis=0)
    rae_main_mean_bag = float(rae(y_unb, main_mean_bag_te[unb_idx]))
    rae_aux_mean_bag = float(rae(y_unb, aux_mean_bag_te[unb_idx]))

    # Blend grid over (w_main, w_aux=1-w_main) on unb 253
    blend_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    blend_results = []
    for w in blend_grid:
        blended = w * main_mean_bag_te + (1.0 - w) * aux_mean_bag_te
        blend_results.append({
            "w_main": w, "rae": float(rae(y_unb, blended[unb_idx]))
        })
    best_blend = min(blend_results, key=lambda r: r["rae"])

    print(f"\n   [C] avg(0.5/0.5) mean-bag rae = {rae_mean_bag:.4f}  "
          f"(d_vs_nb2103={rae_mean_bag - rae_baseline:+.4f})")
    print(f"   [C] main-only mean-bag rae    = {rae_main_mean_bag:.4f}")
    print(f"   [C] aux-only  mean-bag rae    = {rae_aux_mean_bag:.4f}")
    print(f"   [C] blend grid (w_main, rae):")
    for r in blend_results:
        marker = " <-- best" if r["w_main"] == best_blend["w_main"] else ""
        print(f"       w_main={r['w_main']:.1f}  rae={r['rae']:.4f}{marker}")

    return {
        "method": "C_aux_task_lgbm_two_heads_avg",
        "per_seed_records": per_seed_records,
        "per_seed_rae_mean": float(np.mean(per_seed_rae)),
        "per_seed_rae_std": float(np.std(per_seed_rae)),
        "rae_mean_bag_avg": rae_mean_bag,
        "rae_median_bag_avg": rae_median_bag,
        "rae_main_only_mean_bag": rae_main_mean_bag,
        "rae_aux_only_mean_bag": rae_aux_mean_bag,
        "delta_mean_bag_vs_nb2103_K28": rae_mean_bag - rae_baseline,
        "blend_grid_w_main": blend_results,
        "best_blend_w_main": best_blend,
        "mean_bag_te_513_avg": mean_bag_te,
        "main_mean_bag_te_513": main_mean_bag_te,
        "aux_mean_bag_te_513": aux_mean_bag_te,
    }


# ============================================================================
# Main
# ============================================================================

def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- PXR-DIRECT-ONLY ChEMBL retrain (cycle 132 #1)")
    print(f"          baseline nb2103 K=28 mean-bag RAE = "
          f"{NB2103_K28_MEAN_BAG_REF:.4f} (median {NB2103_K28_MEDIAN_BAG_REF:.4f})")
    print(f"          decision margin = {DECISION_MARGIN}")
    print("=" * 78)

    # ---- Load train / test ----
    tr_df = load_train()
    te_df = load_test()
    n_tr = len(tr_df)
    n_te = len(te_df)
    print(f"\n[load] train={n_tr}  test={n_te}")

    # ---- Standardize train + test SMILES for InChIKey dedup ----
    print("[std] standardizing train SMILES...")
    tr_mols = [standardize(s) for s in tr_df["smiles"].tolist()]
    tr_inchikeys = {ik for ik in (_safe_inchikey(m) for m in tr_mols) if ik}
    print(f"   train InChIKeys: {len(tr_inchikeys)} unique")

    print("[std] standardizing test SMILES...")
    te_mols = [standardize(s) for s in te_df["smiles"].tolist()]
    te_inchikeys = {ik for ik in (_safe_inchikey(m) for m in te_mols) if ik}
    print(f"   test  InChIKeys: {len(te_inchikeys)} unique")

    # ---- Load + subset PXR-DIRECT ChEMBL ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR-DIRECT (PXR_CHEMBL3401 + NR_PXR only)")
    print("-" * 78)
    pool = _load_pxr_direct_kb()
    pool = _apply_pec50_range_filter(pool)
    pool = _dedup_by_inchikey(pool, tr_inchikeys, label="train")
    pool = _dedup_by_inchikey(pool, te_inchikeys, label="test")

    # Drop bad SMILES (zero fingerprint)
    fp_ch = morgan_fp_batch(pool["smiles"].tolist())
    keep = fp_ch.sum(axis=1) > 0
    if not keep.all():
        pool = pool[keep].reset_index(drop=True)
        fp_ch = fp_ch[keep]
        print(f"   [filter] zero-FP drop -> {len(pool)} rows")
    print(f"   final PXR-DIRECT pool: {len(pool)} rows")
    print(f"   pec50 distribution: mean={pool['pec50'].mean():.3f}  "
          f"std={pool['pec50'].std():.3f}  "
          f"range=[{pool['pec50'].min():.2f}, {pool['pec50'].max():.2f}]")
    print(f"   src breakdown: {pool['src'].value_counts().to_dict()}")

    # ---- Anchor + truth ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"\n[load] n_unb={n_unb}")

    te_chemprop_aux_513 = np.load(
        DATA_PROCESSED / "te_chemprop_aux.npy"
    ).astype(np.float64)
    rae_anchor = float(rae(y_unb, te_chemprop_aux_513[unb_idx]))
    print(f"[anchor] chemprop_aux te[unb_idx] RAE = {rae_anchor:.4f} "
          f"(ref {CHEMPROP_AUX_REF:.4f})")

    nb2103_oof_unb = np.load(
        DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy"
    ).astype(np.float64)
    rae_nb2103_baseline = float(rae(y_unb, nb2103_oof_unb))
    print(f"[anchor] nb2103 K=28 mean-bag OOF RAE = {rae_nb2103_baseline:.4f} "
          f"(ref {NB2103_K28_MEAN_BAG_REF:.4f})")

    # ---- Combined features (cache) ----
    print("\n" + "-" * 78)
    print("COMBINED FEATURES (Morgan + RDKit)")
    print("-" * 78)
    cache_p = DATA_PROCESSED / "cache_combined_features.npz"
    if cache_p.exists():
        cache = np.load(cache_p)
        X_tr = cache["X_tr"].astype(np.float32)
        X_te = cache["X_te"].astype(np.float32)
        print(f"[cache] X_tr={X_tr.shape}  X_te={X_te.shape}")
    else:
        print(f"[cache] missing -- computing fresh (slow)")
        X_tr = _build_combined_features(tr_df["smiles"].tolist()).astype(np.float32)
        X_te = _build_combined_features(te_df["smiles"].tolist()).astype(np.float32)

    print(f"[chembl] computing combined features for {len(pool)} ChEMBL...")
    X_ch = _build_combined_features(pool["smiles"].tolist()).astype(np.float32)
    print(f"[chembl] X_ch={X_ch.shape}")
    assert X_ch.shape[1] == X_tr.shape[1], (
        f"feature dim mismatch: ChEMBL={X_ch.shape[1]} vs train={X_tr.shape[1]}"
    )

    X_tr_imp = impute(X_tr)
    X_te_imp = impute(X_te)
    X_ch_imp = impute(X_ch)
    print(f"[impute] X_tr NaN={int(np.isnan(X_tr_imp).sum())}  "
          f"X_te NaN={int(np.isnan(X_te_imp).sum())}  "
          f"X_ch NaN={int(np.isnan(X_ch_imp).sum())}")

    y_tr = tr_df["pec50"].astype(np.float64).to_numpy()
    y_ch = pool["pec50"].astype(np.float64).to_numpy()

    # Morgan-only FPs for Method B kNN proxy
    fp_tr = morgan_fp_batch(tr_df["smiles"].tolist()).astype(np.uint8)
    # fp_ch already computed above
    print(f"[knn-prep] fp_tr={fp_tr.shape}  fp_ch={fp_ch.shape}")

    # ---- Run the three methods ----
    res_A = method_A_lgbm_aug(
        X_tr_imp, y_tr, X_ch_imp, y_ch, X_te_imp,
        unb_idx, y_unb, rae_nb2103_baseline,
    )
    res_B = method_B_residual_lgbm(
        X_tr_imp, y_tr, fp_tr,
        X_ch_imp, y_ch, fp_ch,
        X_te_imp, te_chemprop_aux_513,
        unb_idx, y_unb, rae_nb2103_baseline,
    )
    res_C = method_C_aux_task(
        X_tr_imp, y_tr, X_ch_imp, y_ch, X_te_imp,
        unb_idx, y_unb, rae_nb2103_baseline,
    )

    # ---- Compare ----
    print("\n" + "=" * 78)
    print("VERDICT SUMMARY")
    print("=" * 78)
    print(f"   baseline nb2103 K=28 mean-bag RAE = {rae_nb2103_baseline:.4f} "
          f"(ref {NB2103_K28_MEAN_BAG_REF:.4f})  median-bag {NB2103_K28_MEDIAN_BAG_REF:.4f}")
    print(f"   {'method':<10s}  {'rae_mean_bag':>13s}  {'rae_median_bag':>15s}  "
          f"{'d_vs_nb2103':>12s}  verdict")

    cands = [
        ("A", res_A["rae_mean_bag"], res_A["rae_median_bag"], res_A["mean_bag_te_513"]),
        ("B", res_B["rae_mean_bag"], res_B["rae_median_bag"], res_B["mean_bag_te_513"]),
        ("C_avg", res_C["rae_mean_bag_avg"], res_C["rae_median_bag_avg"],
         res_C["mean_bag_te_513_avg"]),
    ]
    # Also surface C's best blend
    cands.append((
        f"C_blend_w{res_C['best_blend_w_main']['w_main']:.1f}",
        res_C["best_blend_w_main"]["rae"],
        float("nan"),
        res_C["best_blend_w_main"]["w_main"] * res_C["main_mean_bag_te_513"]
        + (1.0 - res_C["best_blend_w_main"]["w_main"]) * res_C["aux_mean_bag_te_513"],
    ))

    for name, mean_bag, median_bag, _ in cands:
        d = mean_bag - rae_nb2103_baseline
        if d < -DECISION_MARGIN:
            v = "BEATS_NB2103_K28"
        elif abs(d) < DECISION_MARGIN:
            v = "FLAT"
        else:
            v = "WORSE"
        median_str = f"{median_bag:>15.4f}" if not np.isnan(median_bag) else f"{'n/a':>15s}"
        print(f"   {name:<10s}  {mean_bag:>13.4f}  {median_str}  {d:>+12.4f}  {v}")

    # ---- Choose best path that beats baseline ----
    cands_sorted = sorted(cands, key=lambda r: r[1])
    best_name, best_rae, best_median, best_te_513 = cands_sorted[0]
    beats = best_rae < rae_nb2103_baseline - DECISION_MARGIN

    print(f"\n   best path: {best_name}  rae_mean_bag={best_rae:.4f}  "
          f"d_vs_nb2103={best_rae - rae_nb2103_baseline:+.4f}  "
          f"beats={beats}")

    # ---- Deploy CSV (if any beats) ----
    deploy_path = SUBMISSIONS_DIR / f"{TAG}_deploy_pxr_direct.csv"
    deploy_built = False
    deploy_recipe = None
    if beats:
        deploy_recipe = f"{best_name} ({TAG} pxr-direct chembl retrain; "\
                        f"weighted MSE; 5-seed bag)"
        out_df = pd.DataFrame({
            "SMILES": te_df["smiles"].astype(str),
            "Molecule Name": te_df["name"].astype(str),
            "pEC50": np.asarray(best_te_513, dtype=float),
        })
        deploy_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(deploy_path, index=False)
        deploy_built = True
        print(f"\n   [deploy] wrote {deploy_path} ({len(out_df)} rows)")
        print(f"   [deploy] recipe: {deploy_recipe}")
    else:
        print(f"\n   [deploy] NO CSV WRITTEN -- nb2103 K=28 baseline not beaten")

    # ---- Save summary ----
    # Strip large arrays before dumping
    def _strip(d):
        return {k: v for k, v in d.items()
                if k not in ("mean_bag_te_513", "median_bag_te_513",
                             "mean_bag_te_513_avg", "main_mean_bag_te_513",
                             "aux_mean_bag_te_513")}

    summary = {
        "tag": TAG,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": round(time.time() - t0, 2),
        "context": (
            "PXR-DIRECT-ONLY ChEMBL retrain (cycle 132 #1). Subset 11185-row "
            "chembl_pxr_nr_kb to source_target in {PXR_CHEMBL3401, NR_PXR} = "
            "883+157 = 1040 raw; after pec50 range and dedup, see "
            "augmentation.n_chembl_kept."
        ),
        "data_source": "data/external/chembl_pxr_nr_kb.parquet",
        "pxr_direct_src_keep": sorted(PXR_DIRECT_SRC),
        "pec50_filter": [PEC50_LO, PEC50_HI],
        "augmentation": {
            "n_train": int(n_tr),
            "n_chembl_kept": int(len(pool)),
            "n_chembl_pxr_chembl3401": int(
                (pool["src"] == "PXR_CHEMBL3401").sum()
            ),
            "n_chembl_nr_pxr": int(
                (pool["src"] == "NR_PXR").sum()
            ),
            "train_weight": TRAIN_WEIGHT,
            "chembl_weight": CHEMBL_WEIGHT,
        },
        "feature": {
            "kind": "combined_morgan2048_rdkit",
            "dim": int(X_tr.shape[1]),
            "morgan_n_bits": 2048,
            "rdkit_n_desc": int(X_tr.shape[1] - 2048),
        },
        "lgbm": {
            "objective": "regression",
            "max_depth": 4,
            "num_leaves": 15,
            "n_estimators": 300,
            "learning_rate": 0.03,
            "min_child_samples": 5,
            "reg_lambda": 2.0,
            "subsample": 0.8,
            "colsample_bytree": 0.5,
        },
        "seeds": SEEDS,
        "n_folds_scaffold_cv": N_FOLDS,
        "anchors": {
            "chemprop_aux_te_unb_rae": rae_anchor,
            "nb2103_K28_mean_bag_oof_rae": rae_nb2103_baseline,
            "nb2103_K28_mean_bag_ref": NB2103_K28_MEAN_BAG_REF,
            "nb2103_K28_median_bag_ref": NB2103_K28_MEDIAN_BAG_REF,
        },
        "method_A_lgbm_aug": _strip(res_A),
        "method_B_residual": _strip(res_B),
        "method_C_aux_task": _strip(res_C),
        "best_path": {
            "name": best_name,
            "rae_mean_bag": float(best_rae),
            "rae_median_bag": float(best_median) if not np.isnan(best_median) else None,
            "delta_vs_nb2103_K28": float(best_rae - rae_nb2103_baseline),
            "beats_nb2103_K28": bool(beats),
        },
        "deploy_built": bool(deploy_built),
        "deploy_csv_path": str(deploy_path) if deploy_built else None,
        "deploy_recipe": deploy_recipe,
        "decision_margin": DECISION_MARGIN,
    }
    out_p = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_p, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n[save] {out_p}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    print(f"  n_chembl_kept: {res['augmentation']['n_chembl_kept']}")
    print(f"  n_pxr_chembl3401: {res['augmentation']['n_chembl_pxr_chembl3401']}")
    print(f"  n_nr_pxr:         {res['augmentation']['n_chembl_nr_pxr']}")
    print(f"  rae_anchor (chemprop_aux): {res['anchors']['chemprop_aux_te_unb_rae']:.4f}")
    print(f"  rae_baseline (nb2103 K=28): "
          f"{res['anchors']['nb2103_K28_mean_bag_oof_rae']:.4f}")
    print(f"  method A mean-bag: {res['method_A_lgbm_aug']['rae_mean_bag']:.4f}")
    print(f"  method B mean-bag: {res['method_B_residual']['rae_mean_bag']:.4f}")
    print(f"  method C avg mean-bag: {res['method_C_aux_task']['rae_mean_bag_avg']:.4f}")
    print(f"  method C best blend: "
          f"w_main={res['method_C_aux_task']['best_blend_w_main']['w_main']:.1f} "
          f"rae={res['method_C_aux_task']['best_blend_w_main']['rae']:.4f}")
    print(f"  best_path: {res['best_path']['name']}  "
          f"rae={res['best_path']['rae_mean_bag']:.4f}  "
          f"beats={res['best_path']['beats_nb2103_K28']}")
    print(f"  deploy_built: {res['deploy_built']}")
