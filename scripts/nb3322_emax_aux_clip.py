"""nb3322 -- Emax-aux multitask K=18 residual-LGBM + learned per-fold clip.

NEW PARADIGM:
    Per the project memory, Emax-aux *alone* HURTS the chemprop_aux residual
    (nb1070: Emax-aux mean-bag is flat-to-hurting vs nb2103 K=28; nb2960 K=18
    deep-30 = 0.4536 without Emax).  The hypothesis tested here is that an
    Emax-aux feature, while individually noisy, may NET POSITIVE once a
    learned tail-taming clip is layered on top -- i.e. the Emax column shifts
    a handful of tail predictions in a direction the clip can then exploit.

    Concretely we (1) augment the K=18 SHAP/RFE feature slice with the
    cached Emax-prediction column (nb1070), train a residual-LGBM on the
    chemprop_aux anchor over the 253 unblind, deep-bag over model seeds in
    each outer fold; then (2) apply the per-fold learned-clip primitive
    (grid over (q_low, q_high), fit on fold-train, applied to fold-val) --
    identical mechanics to nb3201.

ANCHOR:
    chemprop_aux (te_chemprop_aux.npy) -- the only verified-clean PRE-unblind
    anchor.  K=18 residual feature indices are reconstructed from the nb2231
    RFE trajectory (same path as nb3000 K=19).  The Emax column is the cached
    nb1070_emax_pred_test_513.npy (Emax model trained on the full 4139 TRAIN,
    predicted on the 513 test) -- PRE-unblind clean (no 253-label leakage).

PROTOCOL (per kf_seed, 5-fold scaffold split over the 253 unblind):
    base feature matrix = [K18 cols (from 117) || emax_pred]  (19 cols)
    residual target      = y_unb - chemprop_aux[unb]
    Per outer fold:
        a) Deep-bag residual-LGBM: for each model seed in BAG_SEEDS, 5-fold
           inner cross-fit is NOT needed here -- instead we fit on fold-train
           and predict fold-val directly (the OOF honesty comes from the
           OUTER scaffold split).  Bag = mean over BAG_SEEDS model seeds.
           pred_val = chemprop_aux[val] + bag_residual_val
        b) Learned clip: inner grid (q_low in Q_LOW_GRID, q_high in
           Q_HIGH_GRID) on fold-TRAIN bag predictions vs y[fold_train];
           pick (lo*, hi*) minimizing fold-train RAE; apply to fold-val.
        c) Stitch clipped fold-val into oof; record per-fold val RAE.
    pooled RAE across the 5 folds; per-fold-mean RAE (the gate metric).
    Repeat for 15 FRESH kf_seeds {1216..1230}; report MEAN per-fold-mean.

GATE (on the 15-seed mean of per-fold-mean):
    mean < 0.4423 -> "BETTER"
    else          -> "FAIL"

References:
    nb2960 K=18 deep-30 OOF (no Emax, no clip)   = 0.4536
    nb3201 learned-clip on K=18 deep-30 OOF      = (clip primitive baseline)
    nb1070 Emax-aux K=28+1 (Emax alone)          = flat/hurts vs nb2103
    nb2171 prior post-hoc-blend ceiling          = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/te_chemprop_aux.npy
    data/processed/nb1070_emax_pred_test_513.npy
    data/processed/nb2231_summary.json            (K=18 idx reconstruction)
    data/processed/nb2063_shap_importance_full117.npy
    data/processed/nb1352_summary.json .. nb1541_summary.json
    data/processed/te_atompair.npy / te_maccs.npy / te_chemprop_embed_300.npy /
        te_avalon512.npy
    C:/pxr_artifacts/nb1030/X_mordred_test.npy
    data/external/chembl_*.parquet                (ChEMBL kNN feature)

Outputs:
    data/processed/nb3322_summary.json
    data/processed/nb3322_pred_oof.npy   (253,) float32 -- median-seed clipped OOF
    data/processed/te_nb3322.npy         (513,) float32 -- deploy te (clipped)
    submissions/nb3322_emax_aux_clip.csv (only on BETTER verdict)
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3322"
PARENT_TAG = "nb2960_K18+nb1070_emax"
K_TARGET = 18

# -- Anchor + residual recipe (IDENTICAL to nb2960 / nb3000) -------------------
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
EMAX_PRED_TE_PATH = DATA_PROCESSED / "nb1070_emax_pred_test_513.npy"
BAG_SEEDS = [11, 23, 37, 51, 67, 83, 97, 113]   # model-seed bag inside each fold

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))   # 15 FRESH seeds {1216..1230}

# -- Per-fold learned clip grid (same as nb3201) -------------------------------
Q_LOW_GRID = [0.01, 0.05, 0.10]
Q_HIGH_GRID = [0.90, 0.95, 0.98]

# -- Gate ----------------------------------------------------------------------
GATE_BETTER = 0.4423

# -- Feature cache paths (same as nb3000) --------------------------------------
ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
NB1392_SUMMARY = DATA_PROCESSED / "nb1392_summary.json"
NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"
NB1523_SUMMARY = DATA_PROCESSED / "nb1523_summary.json"
NB1524_SUMMARY = DATA_PROCESSED / "nb1524_summary.json"
NB1541_SUMMARY = DATA_PROCESSED / "nb1541_summary.json"
NB2231_SUMMARY = DATA_PROCESSED / "nb2231_summary.json"
NB2063_SHAP_PATH = DATA_PROCESSED / "nb2063_shap_importance_full117.npy"

# -- ChEMBL kNN params (identical to nb2960 / nb3000) --------------------------
KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

# -- References ----------------------------------------------------------------
REF_K18_DEEP30 = 0.4536
REF_NB2171 = 0.4682
CHEMPROP_AUX_REF = 0.6216


# ============================================================================
# helpers (lifted verbatim from nb3000 / nb1070)
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


def _load_chembl_pool():
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
    if not frames:
        raise FileNotFoundError("No local ChEMBL PXR parquets found")
    pool = pd.concat(frames, ignore_index=True)
    mols = pool["smiles"].apply(standardize)
    pool["inchikey"] = mols.apply(_safe_inchikey)
    pool["std_smiles"] = mols.apply(_safe_can_smiles)
    pool = pool[pool["inchikey"].notna() & pool["std_smiles"].notna()].copy()
    agg = (
        pool.groupby("inchikey", as_index=False)
        .agg(pec50=("pec50", "median"),
             std_smiles=("std_smiles", "first"),
             src_first=("src", "first"),
             n_meas=("pec50", "count"))
    )
    agg = agg.rename(columns={"src_first": "src"})
    return agg


def _tanimoto_topk(fp_q, fp_pool, k):
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


def _knn_predict(top_idx, top_sim, pool_labels, fallback):
    w = np.clip(top_sim, 0.0, 1.0)
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


def _lgbm_params(seed):
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


def _load_mordred_test(n_test_expected):
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(f"Mordred cache missing (run nb1030): {mte_p}")
    X_te_m = np.load(mte_p).astype(np.float32)
    if X_te_m.shape[0] != n_test_expected:
        raise ValueError(f"Mordred test shape mismatch: {X_te_m.shape}")
    X_te_m = np.where(np.isfinite(X_te_m), X_te_m, np.nan).astype(np.float32)
    col_med = np.nanmedian(X_te_m, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X_te_m)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X_te_m[idx_r, idx_c] = col_med[idx_c]
    return X_te_m


def _load_npy_test(path, n_test_expected):
    if not path.exists():
        raise FileNotFoundError(f"missing cache: {path}")
    X = np.load(path)
    if X.shape[0] != n_test_expected:
        raise ValueError(f"shape mismatch {path}: {X.shape}")
    X = X.astype(np.float32)
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X


def _extract_atompair_top_idx_from_nb1484(sum_1484):
    for f in sum_1484["families"]:
        if f["family"] == "AtomPair":
            return np.array(f["top_idx_ranked"], dtype=int)
    raise KeyError("AtomPair entry not found in nb1484_summary.json")


def _extract_best_K_record(sum_dict, records_key, best_K_key="best_K"):
    best_K = int(sum_dict[best_K_key])
    for r in sum_dict[records_key]:
        if int(r["K"]) == best_K:
            return r
    raise KeyError(f"best_K {best_K} not found in {records_key}")


def reconstruct_K_from_trajectory(nb2231_sum, K_target):
    """Reconstruct surviving feature indices at K_target from nb2231 RFE
    trajectory (verbatim from nb3000 / nb2631)."""
    shap_top28 = list(nb2231_sum["shap_top28_idx_in_117"])
    if K_target == 28:
        return shap_top28
    if K_target > 28:
        if not NB2063_SHAP_PATH.exists():
            raise FileNotFoundError(f"need {NB2063_SHAP_PATH}")
        imp = np.load(NB2063_SHAP_PATH).astype(np.float64)
        order = np.argsort(-imp)
        return [int(j) for j in order[:K_target]]
    current = list(shap_top28)
    traj = nb2231_sum["rfe_trajectory"]
    for entry in traj:
        if entry.get("feat_dropped") is None:
            continue
        if entry["K_after"] < K_target:
            break
        d = int(entry["feat_dropped"])
        if d in current:
            current.remove(d)
        if entry["K_after"] == K_target:
            return current
    if len(current) == K_target:
        return current
    raise ValueError(f"could not reconstruct K={K_target} (got len {len(current)})")


def build_117col_feature_matrix(te_smiles, n_test):
    """117-col matrix identical to nb2604 / nb2631 / nb2960 / nb3000."""
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

    top_maccs_bit_idx = np.array(sum_1352["top_maccs_bit_indices_ranked"], dtype=int)
    rec_mord = _extract_best_K_record(sum_1523, "per_K_records", best_K_key="best_K")
    top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)
    full_ap_ranked = _extract_atompair_top_idx_from_nb1484(sum_1484)
    K_AP_best = int(sum_1524["best_K"])
    top_ap_bit_idx = full_ap_ranked[:K_AP_best]
    K_Embed_best = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed_col_idx = top_embed_full[:K_Embed_best]
    top_avalon_bit_idx = np.array(sum_1392["top_avalon_bit_indices_ranked"], dtype=int)

    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test)
    X_ap_te_top = X_ap_te[:, top_ap_bit_idx].astype(np.float32)
    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)
    X_maccs_te_top = X_maccs_te[:, top_maccs_bit_idx].astype(np.float32)
    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    X_mord_te_top = X_mord_te[:, top_mord_col_idx].astype(np.float32)
    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)
    X_emb_te_top = X_emb_te[:, top_embed_col_idx].astype(np.float32)
    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)
    X_av_te_top = X_av_te[:, top_avalon_bit_idx].astype(np.float32)

    pool = _load_chembl_pool()
    test_mols = [standardize(s) for s in te_smiles]
    test_inchikeys = set()
    for m in test_mols:
        ik = _safe_inchikey(m)
        if ik is not None:
            test_inchikeys.add(ik)
    pool = pool[~pool["inchikey"].isin(test_inchikeys)].reset_index(drop=True)
    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep_pool = fp_pool.sum(axis=1) > 0
    if not keep_pool.all():
        pool = pool[keep_pool].reset_index(drop=True)
        fp_pool = fp_pool[keep_pool]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))
    std_test_smiles = []
    for m in test_mols:
        std_test_smiles.append(Chem.MolToSmiles(m) if m is not None else "")
    fp_test = morgan_fp_batch(std_test_smiles)
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )

    X_te_full = np.concatenate(
        [
            X_ap_te_top,
            X_maccs_te_top,
            X_mord_te_top,
            X_emb_te_top,
            X_av_te_top,
            pred_chembl_pec50.reshape(-1, 1).astype(np.float32),
            mean_sim.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    if X_te_full.shape[1] != 117:
        raise ValueError(f"feat_dim {X_te_full.shape[1]} != 117")
    return X_te_full, int(len(pool))


# ============================================================================
# clip + bag primitives
# ============================================================================

def _pick_best_clip(y_tr: np.ndarray, pred_tr: np.ndarray):
    """Inner grid: pick (q_low*, q_high*) minimizing fold-train RAE."""
    best_rae = np.inf
    best_ql = Q_LOW_GRID[0]
    best_qh = Q_HIGH_GRID[-1]
    best_lo = float(np.quantile(pred_tr, best_ql))
    best_hi = float(np.quantile(pred_tr, best_qh))
    for ql in Q_LOW_GRID:
        lo = float(np.quantile(pred_tr, ql))
        for qh in Q_HIGH_GRID:
            hi = float(np.quantile(pred_tr, qh))
            if hi <= lo:
                continue
            clipped = np.clip(pred_tr, lo, hi)
            r = float(rae(y_tr, clipped))
            if r < best_rae:
                best_rae = r
                best_ql, best_qh, best_lo, best_hi = ql, qh, lo, hi
    return best_ql, best_qh, best_lo, best_hi


def _bag_residual(X_tr, resid_tr, X_apply, seeds):
    """Deep-bag residual-LGBM: fit on (X_tr, resid_tr), predict X_apply,
    averaged over model seeds."""
    acc = np.zeros(X_apply.shape[0], dtype=np.float64)
    for s in seeds:
        mdl = lgb.LGBMRegressor(**_lgbm_params(s))
        mdl.fit(X_tr, resid_tr)
        acc += mdl.predict(X_apply)
    return acc / len(seeds)


def _run_one_seed(X_unb, anchor, residual, y_unb, unb_scaffolds, kf_seed):
    """Outer scaffold split -> per-fold bag residual-LGBM + learned clip."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_clip = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_ql, fold_qh = [], []
    n_clipped_lo = n_clipped_hi = 0
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        # (a) deep-bag residual-LGBM: train on fold-train, predict both
        resid_tr_pred = _bag_residual(
            X_unb[tr_loc], residual[tr_loc], X_unb[tr_loc], BAG_SEEDS)
        resid_va_pred = _bag_residual(
            X_unb[tr_loc], residual[tr_loc], X_unb[va_loc], BAG_SEEDS)
        pred_tr = anchor[tr_loc] + resid_tr_pred
        pred_va = anchor[va_loc] + resid_va_pred
        # (b) learned clip fit on fold-train predictions
        ql, qh, lo, hi = _pick_best_clip(y_unb[tr_loc], pred_tr)
        fold_ql.append(ql)
        fold_qh.append(qh)
        n_clipped_lo += int(np.sum(pred_va < lo))
        n_clipped_hi += int(np.sum(pred_va > hi))
        clipped_va = np.clip(pred_va, lo, hi)
        oof_clip[va_loc] = clipped_va
        fold_val_raes.append(float(rae(y_unb[va_loc], clipped_va)))
    if np.isnan(oof_clip).any():
        raise RuntimeError(f"kf_seed={kf_seed}: scaffold splits left gaps")
    pooled = float(rae(y_unb, oof_clip))
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(fold_val_raes)),
        "per_fold_val_rae_std": float(np.std(fold_val_raes, ddof=1)),
        "fold_ql": fold_ql,
        "fold_qh": fold_qh,
        "n_clipped_lo": n_clipped_lo,
        "n_clipped_hi": n_clipped_hi,
        "oof": oof_clip,
    }


# ============================================================================
# main
# ============================================================================

def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Emax-aux K={K_TARGET} residual-LGBM + learned per-fold clip")
    print(f"          anchor = chemprop_aux  (PRE-unblind, ref {CHEMPROP_AUX_REF:.4f})")
    print(f"          Emax feature = nb1070_emax_pred_test_513 (PRE-unblind)")
    print(f"          bag model seeds = {BAG_SEEDS}")
    print(f"          outer CV = {N_FOLDS}-fold scaffold, {len(KF_SEEDS)} fresh "
          f"kf_seeds {{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}")
    print(f"          Q_LOW={Q_LOW_GRID}  Q_HIGH={Q_HIGH_GRID}")
    print(f"          gate: per-fold-mean < {GATE_BETTER:.4f} -> BETTER")
    print("=" * 78)

    # -- Load test, truth, anchor, Emax --------------------------------------
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values
                if "name" in te.columns else te["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    assert n_unb == 253, f"expected 253 unblind, got {n_unb}"
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(f"chemprop_aux te shape {te_anchor_513.shape}")
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    residual = y_unb - anchor
    print(f"[load] chemprop_aux te[unb] RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    print(f"       residual mean={residual.mean():.3f}  std={residual.std():.3f}")

    emax_pred_513 = np.load(EMAX_PRED_TE_PATH).astype(np.float64)
    if emax_pred_513.shape[0] != n_test:
        raise ValueError(f"emax_pred shape {emax_pred_513.shape}")
    # leak sanity on the Emax feature vs truth (should be ~0)
    emax_truth_corr = float(np.corrcoef(emax_pred_513[unb_idx], y_unb)[0, 1])
    print(f"[load] emax_pred_513 mean={emax_pred_513.mean():.3f} "
          f"std={emax_pred_513.std():.3f}  corr(emax_unb, y_unb)={emax_truth_corr:+.3f}")

    # -- Reconstruct K=18 indices --------------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 1: reconstruct K={K_TARGET} idx from nb2231 RFE trajectory")
    print("-" * 78)
    if not NB2231_SUMMARY.exists():
        raise FileNotFoundError(f"missing {NB2231_SUMMARY}")
    with open(NB2231_SUMMARY) as f:
        nb2231 = json.load(f)
    K18_idx = np.array(reconstruct_K_from_trajectory(nb2231, K_TARGET), dtype=int)
    if len(K18_idx) != K_TARGET:
        raise ValueError(f"K={K_TARGET} reconstruction returned {len(K18_idx)} cols")
    print(f"   K={K_TARGET} idx_in_117: {K18_idx.tolist()}")

    # -- Build 117-col matrix, slice to K=18, append Emax --------------------
    print("\n" + "-" * 78)
    print("STEP 2: build 117-col matrix, slice K=18, append Emax col -> K=18+1")
    print("-" * 78)
    X_te_full, chembl_pool_size = build_117col_feature_matrix(te_smiles, n_test)
    print(f"   X_te_full={X_te_full.shape}  chembl_pool={chembl_pool_size}")
    X_te_K = X_te_full[:, K18_idx].astype(np.float32)
    X_te_aug = np.concatenate(
        [X_te_K, emax_pred_513.reshape(-1, 1).astype(np.float32)], axis=1
    ).astype(np.float32)
    X_unb = X_te_aug[unb_idx].astype(np.float32)
    print(f"   X_te_aug={X_te_aug.shape}  X_unb={X_unb.shape}  "
          f"(K={K_TARGET} + 1 Emax = {X_unb.shape[1]} cols)")

    # -- Scaffolds for outer CV ----------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 3: scaffolds for outer CV")
    print("-" * 78)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"   n_unique_scaffolds = {n_unique_scaf}")

    # -- Per-seed sweep -------------------------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 4: {len(KF_SEEDS)} fresh kf_seeds -- bag residual-LGBM + clip")
    print("-" * 78)
    seed_records = []
    perfoldmean_list = []
    pooled_list = []
    oof_stack = []
    all_fold_ql, all_fold_qh = [], []
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(X_unb, anchor, residual, y_unb, unb_scaffolds, s)
        perfoldmean_list.append(res["per_fold_val_rae_mean"])
        pooled_list.append(res["pooled_rae"])
        oof_stack.append(res["oof"])
        all_fold_ql.extend(res["fold_ql"])
        all_fold_qh.extend(res["fold_qh"])
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
            "per_fold_val_rae_std": round(res["per_fold_val_rae_std"], 4),
            "fold_ql": [round(v, 3) for v in res["fold_ql"]],
            "fold_qh": [round(v, 3) for v in res["fold_qh"]],
            "n_clipped_lo": res["n_clipped_lo"],
            "n_clipped_hi": res["n_clipped_hi"],
        })
        print(f"   kf={s}: per-fold-mean={res['per_fold_val_rae_mean']:.4f}  "
              f"pooled={res['pooled_rae']:.4f}  "
              f"clipped(lo,hi)=({res['n_clipped_lo']},{res['n_clipped_hi']})  "
              f"wall={time.time()-ts:.1f}s")

    arr_pfm = np.asarray(perfoldmean_list, dtype=np.float64)
    arr_pooled = np.asarray(pooled_list, dtype=np.float64)
    n_s = len(arr_pfm)
    mean_pfm = float(arr_pfm.mean())
    std_pfm = float(arr_pfm.std(ddof=1)) if n_s > 1 else 0.0
    sem_pfm = std_pfm / np.sqrt(n_s) if n_s > 1 else 0.0
    t_mult = 2.145  # df=14, 95%
    ci_low = mean_pfm - t_mult * sem_pfm
    ci_high = mean_pfm + t_mult * sem_pfm
    median_pfm = float(np.median(arr_pfm))
    mean_pooled = float(arr_pooled.mean())

    ql_counter = Counter(all_fold_ql)
    qh_counter = Counter(all_fold_qh)
    ql_mode = ql_counter.most_common(1)[0][0]
    qh_mode = qh_counter.most_common(1)[0][0]

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds)")
    print("-" * 78)
    print(f"   PER-FOLD-MEAN RAE (gate metric):")
    print(f"     mean   = {mean_pfm:.4f}")
    print(f"     std    = {std_pfm:.4f}")
    print(f"     sem    = {sem_pfm:.4f}")
    print(f"     95% CI = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"     median = {median_pfm:.4f}")
    print(f"     min/max= [{arr_pfm.min():.4f}, {arr_pfm.max():.4f}]")
    print(f"   pooled RAE mean = {mean_pooled:.4f}")
    print(f"   delta vs K=18 deep-30 (0.4536) = {mean_pfm - REF_K18_DEEP30:+.4f}")
    print(f"   delta vs nb2171 ceiling (0.4682) = {mean_pfm - REF_NB2171:+.4f}")
    print(f"   ql_dist (75 folds) = {dict(ql_counter)}  mode={ql_mode}")
    print(f"   qh_dist (75 folds) = {dict(qh_counter)}  mode={qh_mode}")

    # -- Deploy: refit on FULL 253 -> te (clipped) ---------------------------
    print("\n" + "-" * 78)
    print("STEP 5: deploy -- bag residual-LGBM on FULL 253, clip from full-253 fit")
    print("-" * 78)
    resid_te_pred = _bag_residual(X_unb, residual, X_te_aug, BAG_SEEDS)
    te_raw = (te_anchor_513 + resid_te_pred).astype(np.float64)
    resid_unb_full = _bag_residual(X_unb, residual, X_unb, BAG_SEEDS)
    pred_unb_full = anchor + resid_unb_full
    deploy_ql, deploy_qh, deploy_lo, deploy_hi = _pick_best_clip(y_unb, pred_unb_full)
    te_pred = np.clip(te_raw, deploy_lo, deploy_hi).astype(np.float32)
    n_te_lo = int(np.sum(te_raw < deploy_lo))
    n_te_hi = int(np.sum(te_raw > deploy_hi))
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx].astype(np.float64)))
    print(f"   deploy clip = (q{deploy_ql:.2f},q{deploy_qh:.2f}) -> "
          f"({deploy_lo:.3f},{deploy_hi:.3f})")
    print(f"   te clipped lo={n_te_lo}/513 hi={n_te_hi}/513")
    print(f"   te(513) mean={te_pred.mean():.3f} std={te_pred.std():.3f} "
          f"min={te_pred.min():.3f} max={te_pred.max():.3f}")
    print(f"   te[unb] in-sample RAE = {te_unb_in_rae:.4f}  "
          f"(in-sample optimism vs CV expected)")

    # median-seed OOF for storage
    med_seed_idx = int(np.argsort(arr_pfm)[n_s // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(f"   median seed = {median_seed} (per-fold-mean={arr_pfm[med_seed_idx]:.4f})")

    # -- Gate -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 6: GATE")
    print("-" * 78)
    if mean_pfm < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE-CANDIDATE. nb3322 15-seed per-fold-mean {mean_pfm:.4f} "
            f"clears BETTER gate {GATE_BETTER:.4f}. Emax-aux K=18 residual + "
            f"learned clip NETS POSITIVE vs K=18 deep-30 0.4536 "
            f"({mean_pfm - REF_K18_DEEP30:+.4f}) -- contradicts the Emax-alone-"
            f"hurts memory only WHEN combined with the clip. Re-verify with a "
            f"deep-30 model bag before any PRIMARY-1 swap (current bag={len(BAG_SEEDS)})."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3322 15-seed per-fold-mean {mean_pfm:.4f} fails BETTER "
            f"gate {GATE_BETTER:.4f} ({mean_pfm - GATE_BETTER:+.4f}). Emax-aux "
            f"feature + learned clip does not beat the gate; consistent with "
            f"the memory that Emax-aux alone hurts (corr(emax,y)={emax_truth_corr:+.3f}). "
            f"Keep current PRIMARY-1; no ladder change."
        )
    print(f"   verdict       = {verdict}")
    print(f"   ladder action = {ladder_action}")

    # -- Save artifacts -------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 7: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_for_save)
    np.save(te_path, te_pred)
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_emax_aux_clip.csv"
    if verdict == "BETTER":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": te_pred,
        }).to_csv(sub_csv, index=False)
        print(f"   [save] {sub_csv}")
    else:
        print(f"   [skip] verdict={verdict}; no submission CSV written")

    summary = {
        "tag": TAG,
        "parent_tag": PARENT_TAG,
        "method": ("emax_aux_K18_residual_LGBM_plus_learned_perfold_clip_15seed"),
        "paradigm": "emax_alone_hurts_but_nets_positive_with_clip",
        "anchor_base": "chemprop_aux",
        "anchor_in_rae": round(rae_anchor, 4),
        "anchor_pre_unblind": True,
        "emax_feature_path": str(EMAX_PRED_TE_PATH),
        "emax_pre_unblind": True,
        "emax_truth_corr_unb": round(emax_truth_corr, 4),
        "K_target": K_TARGET,
        "K18_idx_in_117": K18_idx.tolist(),
        "n_aug_cols": int(X_unb.shape[1]),
        "bag_seeds": BAG_SEEDS,
        "n_bag": len(BAG_SEEDS),
        "q_low_grid": Q_LOW_GRID,
        "q_high_grid": Q_HIGH_GRID,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": n_s,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "chembl_pool_size": int(chembl_pool_size),
        "seed_records": seed_records,
        "per_fold_mean_array": [round(float(v), 4) for v in perfoldmean_list],
        "pooled_rae_array": [round(float(v), 4) for v in pooled_list],
        "per_fold_mean_rae": round(mean_pfm, 5),
        "mean_rae": round(mean_pfm, 5),
        "std_rae": round(std_pfm, 5),
        "sem_rae": round(sem_pfm, 5),
        "ci95_low": round(ci_low, 5),
        "ci95_high": round(ci_high, 5),
        "median_rae": round(median_pfm, 5),
        "min_rae": round(float(arr_pfm.min()), 5),
        "max_rae": round(float(arr_pfm.max()), 5),
        "pooled_rae_mean": round(mean_pooled, 5),
        "ql_distribution": {str(k): int(v) for k, v in ql_counter.items()},
        "qh_distribution": {str(k): int(v) for k, v in qh_counter.items()},
        "ql_mode": float(ql_mode),
        "qh_mode": float(qh_mode),
        "ref_K18_deep30": REF_K18_DEEP30,
        "delta_vs_K18_deep30": round(mean_pfm - REF_K18_DEEP30, 4),
        "ref_nb2171": REF_NB2171,
        "delta_vs_nb2171": round(mean_pfm - REF_NB2171, 4),
        "deploy_ql": float(deploy_ql),
        "deploy_qh": float(deploy_qh),
        "deploy_lo": round(float(deploy_lo), 4),
        "deploy_hi": round(float(deploy_hi), 4),
        "n_te_clipped_lo": n_te_lo,
        "n_te_clipped_hi": n_te_hi,
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "te_min": float(te_pred.min()),
        "te_max": float(te_pred.max()),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "median_seed": int(median_seed),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict == "BETTER" else None,
        "gate_better": GATE_BETTER,
        "verdict": verdict,
        "ladder_action": ladder_action,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   per-fold-mean RAE ({n_s} seeds) = {mean_pfm:.4f} +/- {std_pfm:.4f}")
    print(f"   95% CI                        = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   pooled RAE mean               = {mean_pooled:.4f}")
    print(f"   delta vs K=18 deep-30         = {mean_pfm - REF_K18_DEEP30:+.4f}")
    print(f"   te[unb] in-sample RAE         = {te_unb_in_rae:.4f}")
    print(f"   verdict                       = {verdict}")
    print(f"   wall                          = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "per_fold_mean_rae", "mean_rae", "std_rae", "ci95_low", "ci95_high",
        "pooled_rae_mean", "delta_vs_K18_deep30", "delta_vs_nb2171",
        "ql_mode", "qh_mode", "deploy_lo", "deploy_hi",
        "n_te_clipped_lo", "n_te_clipped_hi",
        "emax_truth_corr_unb", "te_unb_in_sample_rae",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
