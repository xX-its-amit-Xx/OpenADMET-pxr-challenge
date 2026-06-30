"""nb2213 -- Per-quantile LGBM residual learner (5 bins by chemprop_aux pred).

HYPOTHESIS:
    Different chemprop_aux quantile bins of the test population may have
    different residual SAR patterns: the bottom bin (very low predicted pEC50)
    is dominated by greasy-novel inactives, the top bin is dominated by
    high-confidence actives near the assay ceiling, etc.  If those bins really
    have distinct residual surfaces, a per-bin LGBM(K=28) router should
    outperform the single nb2103-K=28 mean-bag baseline (RAE 0.4737).

PROTOCOL:
    1. Build the same 117-col 5-way K-tuned feature matrix as
       nb2063/nb2081/nb2091/nb2103 (AtomPair + MACCS + Mordred +
       ChempropEmbed + Avalon + ChEMBL kNN).
    2. Slice to top-28 SHAP indices (same as nb2103 K=28 winner).
    3. Compute 5 quantile-bin edges of chemprop_aux on 4139 train
       (oof_chemprop_aux.npy) -> qcut bin assignment.
    4. Bin each test row by its chemprop_aux pred quantile vs the train edges.
    5. 5-fold scaffold CV on 253 unblind:
       For each fold, train one LGBM(K=28) per bin on the training fold rows
       belonging to that bin.  Predict each val row using its bin's model.
       Fallback: if a training-fold bin has < MIN_BIN_TRAIN rows, fall back to
       a global LGBM trained on the full training fold.
    6. 5-seed bag (seeds 0,1,7,42,137).
    7. Compare vs nb2103 K=28 mean-bag RAE (0.4737); gate 0.003.

NOTE:
    The task brief stated nb2103 K=28 RAE = 0.5057, but the actual saved
    mean-bag-bag RAE is 0.4737 (re-verified from
    data/processed/nb2103_mean_bag_oof_K28.npy on the 253 unblind labels).
    Decision margin is 0.003.

Outputs:
    scripts/nb2213_quantile_lgbm.py
    data/processed/nb2213_summary.json
    data/processed/nb2213_mean_bag_oof.npy   (253,) float32
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from collections import defaultdict
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

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2213"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
ANCHOR_OOF_PATH = DATA_PROCESSED / "oof_chemprop_aux.npy"

# Folds + seeds (match nb2103)
N_FOLDS = 5
SEEDS = [0, 1, 7, 42, 137]

# Per-quantile config
N_BINS = 5
MIN_BIN_TRAIN = 20         # if a fold-bin has fewer rows, fall back to global LGBM

# K=28 winner from nb2103
K_TOP = 28

# Caches
ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")

EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

# Per-family K-tuned indices
NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
NB1392_SUMMARY = DATA_PROCESSED / "nb1392_summary.json"
NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"
NB1523_SUMMARY = DATA_PROCESSED / "nb1523_summary.json"
NB1524_SUMMARY = DATA_PROCESSED / "nb1524_summary.json"
NB1541_SUMMARY = DATA_PROCESSED / "nb1541_summary.json"
NB2063_SUMMARY = DATA_PROCESSED / "nb2063_summary.json"
NB2063_SHAP_IMP = DATA_PROCESSED / "nb2063_shap_importance_full117.npy"
NB2103_K28_OOF = DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy"
NB2103_SUMMARY = DATA_PROCESSED / "nb2103_summary.json"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

# References
CHEMPROP_AUX_REF = 0.6216
NB2103_K28_REF_FROM_TASK = 0.5057   # stated in task brief (likely stale)
DECISION_MARGIN = 0.003


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


def _load_chembl_pool() -> pd.DataFrame:
    """Same union as nb1852/nb1861/nb2063/nb2081/nb2091/nb2103."""
    frames = []
    p1 = EXT_DIR / "chembl_pxr_CHEMBL3401.parquet"
    if p1.exists():
        d = pd.read_parquet(p1)
        mask = (
            d["standard_type"].isin(KEEP_TYPES)
            & d["canonical_smiles"].notna()
            & (d["standard_units"] == "nM")
            & d["standard_value"].notna()
            & d["standard_relation"].isin(KEEP_RELATIONS)
        )
        d = d[mask].copy()
        v = d["standard_value"].astype(float)
        d = d[(v > MIN_NM) & (v < MAX_NM)].copy()
        d["pec50_raw"] = 9.0 - np.log10(d["standard_value"].astype(float))
        d = d[["canonical_smiles", "pec50_raw"]].rename(
            columns={"canonical_smiles": "smiles", "pec50_raw": "pec50"}
        )
        d["src"] = "CHEMBL3401_raw"
        frames.append(d)
        print(f"   [src] CHEMBL3401_raw kept: {len(d)} rows")

    p2 = EXT_DIR / "chembl_nr_extended.parquet"
    if p2.exists():
        d = pd.read_parquet(p2)
        d = d[d["target_name"] == "PXR"].copy()
        d = d[d["pec50"].notna() & d["std_smiles"].notna()].copy()
        d["pec50"] = d["pec50"].astype(float)
        d = d[(d["pec50"] >= 3.0) & (d["pec50"] <= 11.0)].copy()
        d = d[["std_smiles", "pec50"]].rename(columns={"std_smiles": "smiles"})
        d["src"] = "nr_extended"
        frames.append(d)
        print(f"   [src] chembl_nr_extended PXR kept: {len(d)} rows")

    p3 = EXT_DIR / "chembl_pxr_all_types.parquet"
    if p3.exists():
        d = pd.read_parquet(p3)
        d = d[d["target"] == "PXR"].copy()
        d = d[d["pec50"].notna() & d["smiles"].notna()].copy()
        d["pec50"] = d["pec50"].astype(float)
        d = d[(d["pec50"] >= 3.0) & (d["pec50"] <= 11.0)].copy()
        d = d[["smiles", "pec50"]]
        d["src"] = "pxr_all_types"
        frames.append(d)
        print(f"   [src] chembl_pxr_all_types kept: {len(d)} rows")

    if not frames:
        raise FileNotFoundError("No local ChEMBL PXR parquets found")

    pool = pd.concat(frames, ignore_index=True)
    print(f"   [pool] pre-standardize union: {len(pool)} rows")
    mols = pool["smiles"].apply(standardize)
    pool["inchikey"] = mols.apply(_safe_inchikey)
    pool["std_smiles"] = mols.apply(_safe_can_smiles)
    pool = pool[pool["inchikey"].notna() & pool["std_smiles"].notna()].copy()
    print(f"   [pool] after RDKit standardize: {len(pool)} rows")
    agg = (
        pool.groupby("inchikey", as_index=False)
        .agg(pec50=("pec50", "median"),
             std_smiles=("std_smiles", "first"),
             src_first=("src", "first"),
             n_meas=("pec50", "count"))
    )
    agg = agg.rename(columns={"src_first": "src"})
    print(f"   [pool] after InChIKey dedup (median agg): {len(agg)} unique cpds")
    return agg


def _tanimoto_topk(fp_q: np.ndarray, fp_pool: np.ndarray, k: int):
    a = fp_q.astype(np.float32)
    b = fp_pool.astype(np.float32)
    a_sum = a.sum(axis=1)
    b_sum = b.sum(axis=1)
    n_q = a.shape[0]
    n_pool = b.shape[0]
    top_idx = np.zeros((n_q, k), dtype=np.int32)
    top_sim = np.zeros((n_q, k), dtype=np.float32)
    BLOCK = 64
    for s in range(0, n_q, BLOCK):
        e = min(n_q, s + BLOCK)
        inter = a[s:e] @ b.T
        denom = a_sum[s:e, None] + b_sum[None, :] - inter
        denom = np.maximum(denom, 1.0)
        sim = inter / denom
        if k >= n_pool:
            idx_part = np.argsort(-sim, axis=1)[:, :k]
        else:
            part = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
            row_idx = np.arange(e - s)[:, None]
            sim_part = sim[row_idx, part]
            order = np.argsort(-sim_part, axis=1)
            idx_part = part[row_idx, order]
        row_idx = np.arange(e - s)[:, None]
        top_idx[s:e] = idx_part
        top_sim[s:e] = sim[row_idx, idx_part]
    return top_idx, top_sim


def _knn_predict(top_idx: np.ndarray, top_sim: np.ndarray,
                 pool_labels: np.ndarray, fallback: float):
    w = top_sim.copy()
    w = np.clip(w, 0.0, 1.0)
    w_sum = w.sum(axis=1)
    n_q = top_idx.shape[0]
    pred = np.empty(n_q, dtype=np.float32)
    for i in range(n_q):
        if w_sum[i] < SIM_FLOOR:
            pred[i] = fallback
        else:
            pred[i] = np.sum(w[i] * pool_labels[top_idx[i]]) / w_sum[i]
    mean_sim = top_sim.mean(axis=1).astype(np.float32)
    return pred, mean_sim


def _lgbm_params(seed: int) -> dict:
    """Same LGBM(MSE) hyperparams as nb2063/nb2081/nb2091/nb2103."""
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


def _load_mordred_test(n_test_expected: int) -> np.ndarray:
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(
            f"Mordred cache missing -- run nb1030 first ({mte_p})"
        )
    X_te_m = np.load(mte_p).astype(np.float32)
    if X_te_m.shape[0] != n_test_expected:
        raise ValueError(
            f"Mordred test shape mismatch: {X_te_m.shape} vs n_test={n_test_expected}"
        )
    X_te_m = np.where(np.isfinite(X_te_m), X_te_m, np.nan).astype(np.float32)
    col_med = np.nanmedian(X_te_m, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X_te_m)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X_te_m[idx_r, idx_c] = col_med[idx_c]
    return X_te_m


def _load_npy_test(path: Path, n_test_expected: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"missing cache: {path}")
    X = np.load(path)
    if X.shape[0] != n_test_expected:
        raise ValueError(f"shape mismatch {path}: {X.shape}")
    X = X.astype(np.float32)
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X


def _extract_atompair_top_idx_from_nb1484(sum_1484: dict) -> np.ndarray:
    for f in sum_1484["families"]:
        if f["family"] == "AtomPair":
            return np.array(f["top_idx_ranked"], dtype=int)
    raise KeyError("AtomPair entry not found in nb1484_summary.json")


def _extract_best_K_record(sum_dict: dict, records_key: str,
                           best_K_key: str = "best_K") -> dict:
    best_K = int(sum_dict[best_K_key])
    for r in sum_dict[records_key]:
        if int(r["K"]) == best_K:
            return r
    raise KeyError(f"best_K {best_K} not found in {records_key}")


def _per_quantile_oof_one_seed(
    X: np.ndarray,
    residual: np.ndarray,
    bin_id: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    n_bins: int,
) -> tuple[np.ndarray, dict]:
    """5-fold scaffold CV with per-bin LGBM routing.

    For each fold:
        For each bin b in 0..n_bins-1:
            If at least MIN_BIN_TRAIN rows in (training-fold AND bin==b):
                Train LGBM_b on those rows; predict val rows with bin==b.
            Else:
                Fall back: train one global LGBM on full training fold;
                predict val rows with bin==b using global LGBM.
        Any val row never assigned (shouldn't happen) gets global LGBM pred.
    Returns (oof_pred (n,), diagnostics).
    """
    n = len(residual)
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_bin_counts: list[dict] = []
    fold_fallback_counts: list[dict] = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        bin_train_counts = {int(b): int(np.sum(bin_id[tr_loc] == b))
                            for b in range(n_bins)}
        bin_val_counts = {int(b): int(np.sum(bin_id[va_loc] == b))
                          for b in range(n_bins)}
        fold_bin_counts.append({
            "fold": fold_i,
            "train_bin_counts": bin_train_counts,
            "val_bin_counts": bin_val_counts,
        })
        # Train global LGBM once per fold (used for fallback)
        global_mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        global_mdl.fit(X[tr_loc], residual[tr_loc])
        fb = 0
        for b in range(n_bins):
            val_b_mask = bin_id[va_loc] == b
            val_b_loc = va_loc[val_b_mask]
            if len(val_b_loc) == 0:
                continue
            tr_b_loc = tr_loc[bin_id[tr_loc] == b]
            if len(tr_b_loc) >= MIN_BIN_TRAIN:
                mdl_b = lgb.LGBMRegressor(**_lgbm_params(seed))
                mdl_b.fit(X[tr_b_loc], residual[tr_b_loc])
                oof[val_b_loc] = mdl_b.predict(X[val_b_loc])
            else:
                oof[val_b_loc] = global_mdl.predict(X[val_b_loc])
                fb += int(len(val_b_loc))
        fold_fallback_counts.append({"fold": fold_i, "fallback_val_rows": fb})
    diag = {
        "fold_bin_counts": fold_bin_counts,
        "fold_fallback_counts": fold_fallback_counts,
    }
    return oof, diag


def _global_oof_one_seed(
    X: np.ndarray,
    residual: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> np.ndarray:
    """Reference: same 5-fold scaffold CV with ONE global LGBM (no routing)."""
    n = len(residual)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in splits:
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Per-quantile LGBM(K={K_TOP}) router, {N_BINS} bins by chemprop_aux")
    print(f"          anchor={ANCHOR}  seeds={SEEDS}  folds={N_FOLDS} (scaffold)")
    print(f"          ref: nb2103 K=28 mean-bag RAE (recomputed) "
          f"= will be loaded; margin = {DECISION_MARGIN}")
    print("=" * 78)

    # ---- Confirm nb2103 reference ----
    if not NB2103_K28_OOF.exists():
        raise FileNotFoundError(f"missing {NB2103_K28_OOF} -- run nb2103 first")
    nb2103_k28_oof = np.load(NB2103_K28_OOF).astype(np.float64)

    if not NB2063_SHAP_IMP.exists():
        raise FileNotFoundError(f"missing {NB2063_SHAP_IMP} -- run nb2063 first")
    shap_imp_full117 = np.load(NB2063_SHAP_IMP).astype(np.float32)
    full_rank_order = np.argsort(-shap_imp_full117).astype(np.int32)
    topK_idx = full_rank_order[:K_TOP].astype(np.int32)

    # ---- Load anchor + truth ----
    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"chemprop_aux te file missing: {ANCHOR_TE_PATH}")
    if not ANCHOR_OOF_PATH.exists():
        raise FileNotFoundError(f"chemprop_aux oof file missing: {ANCHOR_OOF_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    oof_anchor_4139 = np.load(ANCHOR_OOF_PATH).astype(np.float64)
    print(f"[load] te_chemprop_aux shape={te_anchor_513.shape}  "
          f"oof_chemprop_aux shape={oof_anchor_4139.shape}")
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    nb2103_k28_in_rae = float(rae(y_unb, nb2103_k28_oof))
    print(f"[ref]  chemprop_aux te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    print(f"[ref]  nb2103 K=28 mean_bag RAE (recomputed) = {nb2103_k28_in_rae:.4f}  "
          f"(task-stated 0.5057)")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Quantile bin edges from 4139 train (oof_chemprop_aux) ----
    print("\n" + "-" * 78)
    print(f"QUANTILE BIN EDGES (n_bins={N_BINS}) from 4139 train OOF chemprop_aux")
    print("-" * 78)
    quantiles = np.linspace(0.0, 1.0, N_BINS + 1)[1:-1]   # internal edges
    bin_edges = np.quantile(oof_anchor_4139, quantiles).astype(np.float64)
    print(f"   quantile probs (internal) = {quantiles.tolist()}")
    print(f"   bin edges (chemprop_aux pred) = "
          f"[{', '.join(f'{e:.4f}' for e in bin_edges)}]")
    train_bin_id = np.digitize(oof_anchor_4139, bin_edges, right=False)
    print("   train bin counts  = " +
          ", ".join(f"b{b}:{int((train_bin_id == b).sum())}"
                    for b in range(N_BINS)))
    test_anchor_513_bin_id = np.digitize(te_anchor_513, bin_edges, right=False)
    print("   test  bin counts  = " +
          ", ".join(f"b{b}:{int((test_anchor_513_bin_id == b).sum())}"
                    for b in range(N_BINS)))
    unb_bin_id = test_anchor_513_bin_id[unb_idx].astype(np.int32)
    unb_bin_counts = {int(b): int((unb_bin_id == b).sum()) for b in range(N_BINS)}
    print(f"   unblind bin counts (253) = {unb_bin_counts}")

    # ---- Per-bin truth/anchor stats ----
    per_bin_diag: list[dict] = []
    for b in range(N_BINS):
        m = unb_bin_id == b
        if m.sum() == 0:
            per_bin_diag.append({
                "bin": int(b), "n": 0,
                "anchor_mean": None, "truth_mean": None,
                "residual_mean": None, "residual_std": None,
                "anchor_rae_in_bin": None,
            })
            continue
        anc_b = anchor[m]
        y_b = y_unb[m]
        res_b = y_b - anc_b
        per_bin_diag.append({
            "bin": int(b),
            "n": int(m.sum()),
            "anchor_mean": float(anc_b.mean()),
            "anchor_min": float(anc_b.min()),
            "anchor_max": float(anc_b.max()),
            "truth_mean": float(y_b.mean()),
            "truth_std": float(y_b.std()),
            "residual_mean": float(res_b.mean()),
            "residual_std": float(res_b.std()),
            "anchor_rae_in_bin": float(rae(y_b, anc_b)) if len(y_b) >= 2 else None,
        })
    print("\n   per-bin (unblind) diagnostics:")
    for d in per_bin_diag:
        if d["n"] == 0:
            print(f"     b{d['bin']}  n=  0  (empty)")
            continue
        print(f"     b{d['bin']}  n={d['n']:>3d}  "
              f"anchor_mean={d['anchor_mean']:.4f}  "
              f"truth_mean={d['truth_mean']:.4f}  "
              f"resid_mean={d['residual_mean']:+.4f}  "
              f"resid_std={d['residual_std']:.4f}")

    # ---- Load all K-grid winners (same as nb2103) ----
    for p in (NB1352_SUMMARY, NB1392_SUMMARY, NB1484_SUMMARY,
              NB1523_SUMMARY, NB1524_SUMMARY, NB1541_SUMMARY):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")
    with open(NB1352_SUMMARY) as f:
        sum_1352 = json.load(f)
    with open(NB1392_SUMMARY) as f:
        sum_1392 = json.load(f)
    with open(NB1484_SUMMARY) as f:
        sum_1484 = json.load(f)
    with open(NB1523_SUMMARY) as f:
        sum_1523 = json.load(f)
    with open(NB1524_SUMMARY) as f:
        sum_1524 = json.load(f)
    with open(NB1541_SUMMARY) as f:
        sum_1541 = json.load(f)

    top_maccs_bit_idx = np.array(
        sum_1352["top_maccs_bit_indices_ranked"], dtype=int
    )
    rec_mord = _extract_best_K_record(sum_1523, "per_K_records",
                                       best_K_key="best_K")
    K_Mord_best = int(rec_mord["K"])
    top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)
    full_ap_ranked = _extract_atompair_top_idx_from_nb1484(sum_1484)
    K_AP_best = int(sum_1524["best_K"])
    top_ap_bit_idx = full_ap_ranked[:K_AP_best]
    K_Embed_best = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed_col_idx = top_embed_full[:K_Embed_best]
    top_avalon_bit_idx = np.array(
        sum_1392["top_avalon_bit_indices_ranked"], dtype=int
    )

    # ---- Feature matrices on UNBLIND ----
    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test)
    X_ap_unb_top = X_ap_te[unb_idx][:, top_ap_bit_idx].astype(np.float32)
    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)
    X_maccs_unb_top = X_maccs_te[unb_idx][:, top_maccs_bit_idx].astype(np.float32)
    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    X_mord_unb_top = X_mord_te[unb_idx][:, top_mord_col_idx].astype(np.float32)
    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)
    X_emb_unb_top = X_emb_te[unb_idx][:, top_embed_col_idx].astype(np.float32)
    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)
    X_av_unb_top = X_av_te[unb_idx][:, top_avalon_bit_idx].astype(np.float32)
    print(f"\n[feat] AP={X_ap_unb_top.shape}  MACCS={X_maccs_unb_top.shape}  "
          f"Mord={X_mord_unb_top.shape}  Embed={X_emb_unb_top.shape}  "
          f"Av={X_av_unb_top.shape}")

    # ---- ChEMBL kNN feature (same as nb2103) ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL (union)")
    print("-" * 78)
    pool = _load_chembl_pool()
    test_mols = [standardize(s) for s in test_smiles]
    test_inchikeys = set()
    for m in test_mols:
        ik = _safe_inchikey(m)
        if ik is not None:
            test_inchikeys.add(ik)
    n_before = len(pool)
    pool = pool[~pool["inchikey"].isin(test_inchikeys)].reset_index(drop=True)
    n_after = len(pool)
    print(f"   pool: {n_before} -> {n_after}  (dropped {n_before - n_after})")
    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep_pool = fp_pool.sum(axis=1) > 0
    if not keep_pool.all():
        pool = pool[keep_pool].reset_index(drop=True)
        fp_pool = fp_pool[keep_pool]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))
    print(f"   final pool size: {len(pool)}  median pEC50 = {pool_median:.3f}")
    std_test_smiles = []
    for m in test_mols:
        if m is None:
            std_test_smiles.append("")
        else:
            std_test_smiles.append(Chem.MolToSmiles(m))
    fp_test = morgan_fp_batch(std_test_smiles)
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )
    pred_chembl_unb = pred_chembl_pec50[unb_idx].astype(np.float32)
    mean_sim_unb = mean_sim[unb_idx].astype(np.float32)

    # ---- Build COMBINED 5-way K-tuned 117-col matrix, then slice top-K=28 ----
    X_unb_full = np.concatenate(
        [
            X_ap_unb_top,
            X_maccs_unb_top,
            X_mord_unb_top,
            X_emb_unb_top,
            X_av_unb_top,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim_full = X_unb_full.shape[1]
    if feat_dim_full != shap_imp_full117.shape[0]:
        raise ValueError(
            f"feat_dim_full {feat_dim_full} != nb2063 SHAP length "
            f"{shap_imp_full117.shape[0]}"
        )
    X_topK = X_unb_full[:, topK_idx].astype(np.float32)
    print(f"\n[feat] X_unb_full = {X_unb_full.shape}  X_topK = {X_topK.shape}")

    # ---- 5-fold SCAFFOLD CV on the 253 unblind ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD {N_FOLDS}-FOLD ON 253 UNBLIND")
    print("-" * 78)
    unb_smiles = [test_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_scaffolds = len(set(s for s in unb_scaffolds if s))
    print(f"   n_unique_scaffolds on 253 unblind = {n_scaffolds}")
    scaffold_splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=42
    )
    fold_sizes = [(len(tr), len(va)) for tr, va in scaffold_splits]
    print(f"   fold sizes (train, val) = {fold_sizes}")

    # ---- Per-seed loop: per-quantile OOF + global OOF reference ----
    print("\n" + "-" * 78)
    print(f"PER-QUANTILE LGBM(K={K_TOP}) RESIDUAL CROSS-FIT + GLOBAL REFERENCE")
    print("-" * 78)
    per_seed_pq_corr = np.zeros((len(SEEDS), n_unb), dtype=np.float64)
    per_seed_gl_corr = np.zeros((len(SEEDS), n_unb), dtype=np.float64)
    per_seed_pq_rae: list[float] = []
    per_seed_gl_rae: list[float] = []
    per_seed_records = []
    for i, s in enumerate(SEEDS):
        ts = time.time()
        # per-quantile
        resid_oof_pq, diag = _per_quantile_oof_one_seed(
            X_topK, residual, unb_bin_id, scaffold_splits, s, N_BINS
        )
        pred_corr_pq = anchor + resid_oof_pq
        per_seed_pq_corr[i] = pred_corr_pq
        rae_pq_s = float(rae(y_unb, pred_corr_pq))
        per_seed_pq_rae.append(rae_pq_s)
        # global (same splits, same data, single LGBM)
        resid_oof_gl = _global_oof_one_seed(
            X_topK, residual, scaffold_splits, s
        )
        pred_corr_gl = anchor + resid_oof_gl
        per_seed_gl_corr[i] = pred_corr_gl
        rae_gl_s = float(rae(y_unb, pred_corr_gl))
        per_seed_gl_rae.append(rae_gl_s)
        # diagnostics summary
        total_fallback = int(sum(d["fallback_val_rows"]
                                 for d in diag["fold_fallback_counts"]))
        per_seed_records.append({
            "seed": int(s),
            "rae_per_quantile": rae_pq_s,
            "rae_global": rae_gl_s,
            "delta_pq_vs_global": rae_pq_s - rae_gl_s,
            "total_fallback_val_rows": total_fallback,
            "fold_bin_counts": diag["fold_bin_counts"],
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   seed={s:3d}: rae_PQ = {rae_pq_s:.4f}   rae_GL = {rae_gl_s:.4f}   "
              f"d_PQ-GL = {rae_pq_s - rae_gl_s:+.4f}   "
              f"fallback_val = {total_fallback}   "
              f"wall = {time.time() - ts:.1f}s")

    # ---- Mean/median bagging ----
    mean_bag_pq = per_seed_pq_corr.mean(axis=0)
    median_bag_pq = np.median(per_seed_pq_corr, axis=0)
    mean_bag_gl = per_seed_gl_corr.mean(axis=0)
    median_bag_gl = np.median(per_seed_gl_corr, axis=0)
    rae_pq_mean_bag = float(rae(y_unb, mean_bag_pq))
    rae_pq_median_bag = float(rae(y_unb, median_bag_pq))
    rae_gl_mean_bag = float(rae(y_unb, mean_bag_gl))
    rae_gl_median_bag = float(rae(y_unb, median_bag_gl))

    per_seed_pq_arr = np.array(per_seed_pq_rae)
    per_seed_gl_arr = np.array(per_seed_gl_rae)

    # Per-bin RAE in mean-bag-PQ vs anchor vs nb2103-K28
    per_bin_pq_rae: list[dict] = []
    for b in range(N_BINS):
        m = unb_bin_id == b
        if m.sum() < 2:
            per_bin_pq_rae.append({
                "bin": int(b), "n": int(m.sum()),
                "rae_anchor": None,
                "rae_nb2103_K28": None,
                "rae_pq_mean_bag": None,
                "rae_gl_mean_bag": None,
            })
            continue
        per_bin_pq_rae.append({
            "bin": int(b),
            "n": int(m.sum()),
            "rae_anchor": float(rae(y_unb[m], anchor[m])),
            "rae_nb2103_K28": float(rae(y_unb[m], nb2103_k28_oof[m])),
            "rae_pq_mean_bag": float(rae(y_unb[m], mean_bag_pq[m])),
            "rae_gl_mean_bag": float(rae(y_unb[m], mean_bag_gl[m])),
        })

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"   chemprop_aux te[unb_idx] in_RAE   = {rae_anchor:.4f}")
    print(f"   nb2103 K=28 mean_bag      in_RAE  = {nb2103_k28_in_rae:.4f}  "
          f"(task-stated 0.5057)")
    print(f"   per-quantile mean_bag     RAE     = {rae_pq_mean_bag:.4f}")
    print(f"   per-quantile median_bag   RAE     = {rae_pq_median_bag:.4f}")
    print(f"   global       mean_bag     RAE     = {rae_gl_mean_bag:.4f}")
    print(f"   global       median_bag   RAE     = {rae_gl_median_bag:.4f}")
    print(f"   per-seed PQ mean={per_seed_pq_arr.mean():.4f}  "
          f"std={per_seed_pq_arr.std():.4f}  "
          f"min={per_seed_pq_arr.min():.4f}  max={per_seed_pq_arr.max():.4f}")
    print(f"   per-seed GL mean={per_seed_gl_arr.mean():.4f}  "
          f"std={per_seed_gl_arr.std():.4f}  "
          f"min={per_seed_gl_arr.min():.4f}  max={per_seed_gl_arr.max():.4f}")
    print(f"   delta PQ - GL (mean_bag)          = "
          f"{rae_pq_mean_bag - rae_gl_mean_bag:+.4f}")
    print(f"   delta PQ - nb2103_K28 (in_RAE)    = "
          f"{rae_pq_mean_bag - nb2103_k28_in_rae:+.4f}")
    print(f"   delta PQ - chemprop_aux           = "
          f"{rae_pq_mean_bag - rae_anchor:+.4f}")

    print("\n   per-bin RAE breakdown (mean_bag):")
    print(f"     {'bin':>3s}  {'n':>3s}  {'anchor':>7s}  {'nb2103_K28':>10s}  "
          f"{'PQ':>7s}  {'GL':>7s}")
    for d in per_bin_pq_rae:
        if d["rae_pq_mean_bag"] is None:
            print(f"     {d['bin']:>3d}  {d['n']:>3d}  (insufficient n)")
        else:
            print(f"     {d['bin']:>3d}  {d['n']:>3d}  "
                  f"{d['rae_anchor']:>7.4f}  {d['rae_nb2103_K28']:>10.4f}  "
                  f"{d['rae_pq_mean_bag']:>7.4f}  {d['rae_gl_mean_bag']:>7.4f}")

    # ---- Verdict ----
    beats_nb2103_K28 = rae_pq_mean_bag < nb2103_k28_in_rae - DECISION_MARGIN
    beats_nb2103_K28_task_stated = (
        rae_pq_mean_bag < NB2103_K28_REF_FROM_TASK - DECISION_MARGIN
    )
    flat_vs_nb2103_K28 = abs(rae_pq_mean_bag - nb2103_k28_in_rae) < DECISION_MARGIN
    beats_global = rae_pq_mean_bag < rae_gl_mean_bag - DECISION_MARGIN
    beats_anchor = rae_pq_mean_bag < rae_anchor - DECISION_MARGIN

    if beats_nb2103_K28:
        verdict = f"PQ_BEATS_NB2103_K28_BY_{nb2103_k28_in_rae - rae_pq_mean_bag:.4f}"
    elif flat_vs_nb2103_K28:
        verdict = "PQ_FLAT_VS_NB2103_K28_WITHIN_MARGIN"
    elif beats_global:
        verdict = "PQ_BEATS_GLOBAL_BUT_NOT_NB2103_K28"
    elif beats_anchor:
        verdict = "PQ_BEATS_ANCHOR_BUT_NOT_GLOBAL_NOR_NB2103_K28"
    else:
        verdict = "PQ_DOES_NOT_BEAT_ANCHOR"
    print(f"\n   verdict = {verdict}")
    print(f"   beats_nb2103_K28 (recomputed 0.4737) = {beats_nb2103_K28}  "
          f"flat_within_margin = {flat_vs_nb2103_K28}")
    print(f"   beats_nb2103_K28 (task-stated 0.5057) = {beats_nb2103_K28_task_stated}")
    print(f"   beats_global_K28 LGBM (same splits)   = {beats_global}")
    print(f"   beats_anchor (chemprop_aux)           = {beats_anchor}")

    # ---- Save OOF and summary ----
    out_oof = DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy"
    np.save(out_oof, mean_bag_pq.astype(np.float32))
    print(f"\n[save] {out_oof}")

    summary = {
        "tag": TAG,
        "method": (f"per_quantile_lgbm_K{K_TOP}_residual_router_{N_BINS}bins"
                   "_scaffoldCV5_5seedbag"),
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_te_path": str(ANCHOR_TE_PATH),
        "anchor_oof_path": str(ANCHOR_OOF_PATH),
        "feature_source": ("nb2063 cached SHAP importance + same 117-col 5-way "
                           "K-tuned matrix (AtomPair / MACCS / Mordred / "
                           "ChempropEmbed / Avalon + ChEMBL kNN), sliced to "
                           f"top-K={K_TOP}"),
        "K_top": int(K_TOP),
        "n_bins": int(N_BINS),
        "min_bin_train": int(MIN_BIN_TRAIN),
        "n_folds": int(N_FOLDS),
        "seeds": SEEDS,
        "n_unb": int(n_unb),
        "n_chembl_pool": int(len(pool)),
        "n_unique_scaffolds_unb": int(n_scaffolds),
        "fold_sizes": fold_sizes,
        "quantile_probs": quantiles.tolist(),
        "bin_edges_chemprop_aux": bin_edges.tolist(),
        "train_bin_counts": {int(b): int((train_bin_id == b).sum())
                              for b in range(N_BINS)},
        "test513_bin_counts": {int(b): int((test_anchor_513_bin_id == b).sum())
                                for b in range(N_BINS)},
        "unb253_bin_counts": unb_bin_counts,
        "per_bin_diag_unb": per_bin_diag,
        "rae_anchor_chemprop_aux": rae_anchor,
        "rae_nb2103_K28_recomputed_unb": nb2103_k28_in_rae,
        "rae_nb2103_K28_task_stated": NB2103_K28_REF_FROM_TASK,
        "rae_pq_mean_bag": rae_pq_mean_bag,
        "rae_pq_median_bag": rae_pq_median_bag,
        "rae_gl_mean_bag": rae_gl_mean_bag,
        "rae_gl_median_bag": rae_gl_median_bag,
        "per_seed_pq_rae": per_seed_pq_rae,
        "per_seed_gl_rae": per_seed_gl_rae,
        "per_seed_pq_mean": float(per_seed_pq_arr.mean()),
        "per_seed_pq_std": float(per_seed_pq_arr.std()),
        "per_seed_gl_mean": float(per_seed_gl_arr.mean()),
        "per_seed_gl_std": float(per_seed_gl_arr.std()),
        "per_seed_records": per_seed_records,
        "per_bin_pq_rae_meanbag": per_bin_pq_rae,
        "delta_pq_vs_nb2103_K28_recomputed":
            rae_pq_mean_bag - nb2103_k28_in_rae,
        "delta_pq_vs_nb2103_K28_task_stated":
            rae_pq_mean_bag - NB2103_K28_REF_FROM_TASK,
        "delta_pq_vs_global_K28": rae_pq_mean_bag - rae_gl_mean_bag,
        "delta_pq_vs_chemprop_aux": rae_pq_mean_bag - rae_anchor,
        "beats_nb2103_K28_recomputed": bool(beats_nb2103_K28),
        "flat_vs_nb2103_K28_recomputed": bool(flat_vs_nb2103_K28),
        "beats_nb2103_K28_task_stated": bool(beats_nb2103_K28_task_stated),
        "beats_global_K28_same_splits": bool(beats_global),
        "beats_chemprop_aux": bool(beats_anchor),
        "verdict": verdict,
        "pre_unblind_clean": True,
        "decision_margin": DECISION_MARGIN,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_summary = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_summary}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== TOP-LINE SUMMARY ====")
    for k in (
        "K_top", "n_bins", "min_bin_train", "n_folds",
        "n_chembl_pool", "fold_sizes",
        "bin_edges_chemprop_aux",
        "unb253_bin_counts",
        "rae_anchor_chemprop_aux",
        "rae_nb2103_K28_recomputed_unb",
        "rae_pq_mean_bag", "rae_pq_median_bag",
        "rae_gl_mean_bag",
        "delta_pq_vs_nb2103_K28_recomputed",
        "delta_pq_vs_global_K28",
        "verdict", "pre_unblind_clean",
    ):
        print(f"  {k}: {res.get(k)}")
