"""nb2692 -- Test-Time Training (TTT) with self-supervised auxiliary objective.

NEW PARADIGM (cycle 270):
    Standard residual LGBM trained on (X_unb, residual) is FROZEN at inference.
    TTT at test time: for each test row x*, mask one input feature j*, then
    briefly fine-tune the trained LGBM (warm-start +10 rounds, lr=0.001) on
    a per-row self-supervised auxiliary task -- predict the masked feature j*
    from the OTHER 19 features (reconstruction).  The intuition is that the
    auxiliary task pulls the booster's split structure slightly toward the
    test-row's local manifold, so the TARGET prediction on x* becomes more
    locally calibrated.

    Auxiliary task data: the train rows (i.e. the other 252 unblind rows in
    each held-out fold) -- those rows have full features, so masking column j*
    on them gives a per-row supervised reconstruction loss
    (target = X_train[:, j*], features = X_train[:, !j*]).  We pick j* per
    test row as the column whose value on x* is the FURTHEST from the train
    median of that column (i.e. the most "off-manifold" feature, since
    pulling the booster toward an explanation for THAT feature is what is
    most expected to help).

    Then we PREDICT the target on x* using the warm-adjusted model with the
    MASKED feature set to the train-fold median (so the model is asked to
    impute it from context, exactly what TTT trained for).

PROTOCOL:
    1. Reuse 117-col 5-way feature matrix from nb2103/nb2240 build path
       (cached via build_117col_feature_matrix in nb2660/nb2640).
    2. Slice K=20 cols using nb2240_summary['k20_surviving_idx_in_117'].
    3. anchor = te_chemprop_aux[unb_idx]; residual = y_unb - anchor.
    4. For each kf_seed in {1001, 1002, 1003, 1004, 1005}:
         5-fold scaffold-CV (`scaffold_kfold_indices`):
           For each fold:
             a. Fit LGBM(K=20, residual) on tr rows  -- BASE MODEL.
             b. Base prediction on va rows  -> baseline_oof[va].
             c. For each va row v:
                  - Compute per-col z-distance of X[v] to tr-fold median:
                      z[j] = |X[v, j] - median(X[tr, j])| / (mad(X[tr, j]) + eps)
                    Pick j* = argmax z (most off-manifold feature on this row).
                  - Build masked tr rows: X_tr_masked = X_tr with col j* set to
                    median(X_tr[:, j*]) on tr.  Aux target = X_tr[:, j*].
                    Warm-start booster with init_model=base_booster,
                    +10 rounds at lr=0.001 on (X_tr_masked, X_tr[:, j*]).
                  - Build x_v_masked = X[v] with col j* set to
                    median(X_tr[:, j*]).
                  - ttt_oof[v] = base_booster_warm.predict(x_v_masked).
                    (model now has 310 trees -- 300 base + 10 aux warm.)
             d. Per-kf RAE = rae(y_unb, anchor + ttt_oof) and baseline equivalent.
       Average pooled RAE across 5 kf_seeds.

GATES:
    mean_rae < 0.4570  -> "PROMOTE"
    mean_rae < 0.4598  -> "MARGINAL_BEAT"
    else               -> "FAIL"

NOTE: The warm-start adds 10 trees at lr=0.001 (so target perturbation per
    tree is tiny).  Even if the aux task is mostly irrelevant, the small lr
    bounds the damage.

Outputs:
    scripts/nb2692_test_time_training.py
    data/processed/nb2692_summary.json
    data/processed/nb2692_pred_oof.npy   (253,) float32  (mean over 5 kf_seeds of ttt-corrected pred)
    data/processed/te_nb2692.npy         (513,) float32  (deploy refit on all 253 with TTT-on-te)
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
import lightgbm as lgb
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2692"

# -- Protocol params --------------------------------------------------------
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
RESID_FOLDS = 5
BASE_SEED = 0                           # LGBM seed for base + warm fit

# Base LGBM (same family as nb2103/nb2240/nb2660)
BASE_N_ESTIMATORS = 300
BASE_LR = 0.03

# TTT warm fine-tune
TTT_N_EXTRA = 10                        # +10 boosting rounds at warm-start
TTT_LR = 0.001                          # tiny lr to bound perturbation

# -- Reference numbers -----------------------------------------------------
CHEMPROP_AUX_REF = 0.6216
NB2240_K20_REF = 0.4720                 # nb2240 K=20 pyramid (deep-30 band)
NB2660_K18_DEEP30 = 0.4682              # PRIMARY-1 deep-30 ceiling

# -- Gates ----------------------------------------------------------------
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

# -- Feature cache paths --------------------------------------------------
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
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
NB2240_SUMMARY = DATA_PROCESSED / "nb2240_summary.json"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6


# ============================================================================
# helpers (identical to nb2660 / nb2640 build path)
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
    import pandas as pd
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


def build_117col_feature_matrix(te_smiles, n_test):
    """117-col matrix identical to nb2660/nb2640/nb2240/nb2103."""
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
    rec_mord = _extract_best_K_record(
        sum_1523, "per_K_records", best_K_key="best_K"
    )
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
# TTT: pick masked feature + warm-start fine-tune
# ============================================================================

def _pick_masked_feature(x_row: np.ndarray,
                         X_tr: np.ndarray,
                         med_tr: np.ndarray,
                         mad_tr: np.ndarray) -> int:
    """Pick column index whose value on x_row is furthest from train median
    (in MAD units).  This is the most off-manifold feature -- exactly what
    TTT should pull the booster toward.
    """
    z = np.abs(x_row - med_tr) / (mad_tr + 1e-6)
    return int(np.argmax(z))


def _ttt_predict_one(base_booster: lgb.Booster,
                     x_row: np.ndarray,
                     X_tr: np.ndarray,
                     med_tr: np.ndarray,
                     mad_tr: np.ndarray) -> float:
    """Test-Time Training on ONE row.

    1. Pick j* = argmax |x_row - med_tr| / mad_tr  (most off-manifold col).
    2. Build masked aux training data:  X_tr_masked = X_tr with col j*
       overwritten by med_tr[j*];  aux target = X_tr[:, j*].
    3. Warm-start fine-tune base_booster +TTT_N_EXTRA rounds at TTT_LR.
    4. Predict target on x_v_masked = x_row with col j* -> med_tr[j*].
    """
    j_star = _pick_masked_feature(x_row, X_tr, med_tr, mad_tr)

    # Build aux training set (mask col j* on X_tr)
    X_tr_masked = X_tr.copy()
    X_tr_masked[:, j_star] = med_tr[j_star]
    aux_target = X_tr[:, j_star].astype(np.float64)

    # Warm-start fine-tune at tiny lr.
    # Note: passing init_model + n_estimators=TTT_N_EXTRA fits TTT_N_EXTRA
    # MORE trees on top of the base booster's existing trees.
    warm = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=TTT_N_EXTRA,
        learning_rate=TTT_LR,
        max_depth=4,
        num_leaves=15,
        min_child_samples=5,
        reg_lambda=2.0,
        random_state=BASE_SEED,
        n_jobs=1,
        verbosity=-1,
    )
    warm.fit(X_tr_masked, aux_target, init_model=base_booster)

    # Predict TARGET on masked x_row (model imputes col j* from context).
    x_v_masked = x_row.copy()
    x_v_masked[j_star] = med_tr[j_star]
    pred = warm.predict(x_v_masked.reshape(1, -1))[0]
    return float(pred)


def _fit_base_booster(X_tr: np.ndarray,
                      y_tr: np.ndarray,
                      seed: int) -> lgb.Booster:
    mdl = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=BASE_N_ESTIMATORS,
        learning_rate=BASE_LR,
        max_depth=4,
        num_leaves=15,
        min_child_samples=5,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )
    mdl.fit(X_tr, y_tr)
    return mdl.booster_


# ============================================================================
# main
# ============================================================================

def main():
    import pandas as pd
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Test-Time Training (TTT) on K=20 LGBM chemprop_aux residual")
    print(f"          base: 300 trees @ lr={BASE_LR}, TTT: +{TTT_N_EXTRA} trees "
          f"@ lr={TTT_LR}")
    print(f"          {len(KF_SEEDS)} kf_seeds x {RESID_FOLDS} folds")
    print(f"          gates: PROMOTE < {GATE_PROMOTE} / MARGINAL < "
          f"{GATE_MARGINAL}")
    print("=" * 78)

    # ---- Load truth, anchor, scaffolds, names ----
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values
                if "name" in te.columns else te["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    residual = y_unb - anchor
    print(f"[anchor] chemprop_aux RAE = {rae_anchor:.4f}  "
          f"resid std = {residual.std():.4f}")

    # ---- Load K=20 feature indices ----
    print("\n" + "-" * 78)
    print("STEP 1: load K=20 surviving indices from nb2240_summary.json")
    print("-" * 78)
    with open(NB2240_SUMMARY) as f:
        nb2240 = json.load(f)
    K20_idx = np.array(nb2240["k20_surviving_idx_in_117"], dtype=int)
    if len(K20_idx) != 20:
        raise RuntimeError(f"K20 idx len {len(K20_idx)} != 20")
    print(f"   K=20 idx (n={len(K20_idx)}): {K20_idx.tolist()}")

    # ---- Build 117-col matrix, slice to K=20 ----
    print("\n" + "-" * 78)
    print("STEP 2: rebuild 117-col 5-way feature matrix, slice to K=20")
    print("-" * 78)
    X_te_full, chembl_pool_size = build_117col_feature_matrix(te_smiles, n_test)
    print(f"   X_te_full = {X_te_full.shape}  chembl_pool={chembl_pool_size}")
    X_te_K = X_te_full[:, K20_idx].astype(np.float32)
    X_unb_K = X_te_K[unb_idx]
    print(f"   X_unb_K = {X_unb_K.shape}  X_te_K = {X_te_K.shape}")

    # ---- TTT scaffold-CV across kf_seeds ----
    print("\n" + "-" * 78)
    print(f"STEP 3: scaffold-CV with TTT across {len(KF_SEEDS)} kf_seeds")
    print("-" * 78)

    per_kf_baseline_rae = []
    per_kf_ttt_rae = []
    # Accumulator for mean over kf_seeds  ->  pred_oof on 253
    ttt_oof_accum = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)
    base_oof_accum = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)

    n_eps = 1e-6
    for ki, kf_seed in enumerate(KF_SEEDS):
        print(f"\n  kf_seed = {kf_seed}")
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=RESID_FOLDS,
            shuffle=True, seed=kf_seed,
        )

        ttt_oof = np.full(n_unb, np.nan, dtype=np.float64)
        base_oof = np.full(n_unb, np.nan, dtype=np.float64)
        for fi, (tr_loc, va_loc) in enumerate(splits):
            ts = time.time()
            X_tr = X_unb_K[tr_loc].astype(np.float32)
            X_va = X_unb_K[va_loc].astype(np.float32)
            y_tr_resid = residual[tr_loc]

            # Train BASE booster on residual
            base_booster = _fit_base_booster(X_tr, y_tr_resid, seed=BASE_SEED)

            # Train-fold col medians + MADs (for masking + off-manifold pick)
            med_tr = np.median(X_tr, axis=0)
            mad_tr = np.median(np.abs(X_tr - med_tr[None, :]), axis=0)

            # BASE prediction (no TTT)
            base_pred = base_booster.predict(X_va)
            base_oof[va_loc] = base_pred

            # TTT per-row prediction
            for local_i, v_idx in enumerate(va_loc):
                ttt_oof[v_idx] = _ttt_predict_one(
                    base_booster, X_va[local_i], X_tr, med_tr, mad_tr,
                )

            print(f"   fold {fi+1}/{RESID_FOLDS}  n_va={len(va_loc):3d}  "
                  f"base_RAE_va={float(rae(residual[va_loc], base_pred)):.4f}  "
                  f"ttt_RAE_va={float(rae(residual[va_loc], ttt_oof[va_loc])):.4f}"
                  f"  wall={time.time()-ts:.1f}s")

        if np.isnan(ttt_oof).any() or np.isnan(base_oof).any():
            raise RuntimeError("scaffold split did not cover all rows")

        base_pred_corr = anchor + base_oof
        ttt_pred_corr = anchor + ttt_oof
        base_rae_kf = float(rae(y_unb, base_pred_corr))
        ttt_rae_kf = float(rae(y_unb, ttt_pred_corr))
        per_kf_baseline_rae.append(base_rae_kf)
        per_kf_ttt_rae.append(ttt_rae_kf)
        base_oof_accum[ki] = base_pred_corr
        ttt_oof_accum[ki] = ttt_pred_corr
        print(f"   kf={kf_seed}  base_RAE={base_rae_kf:.4f}  "
              f"ttt_RAE={ttt_rae_kf:.4f}  "
              f"delta={ttt_rae_kf - base_rae_kf:+.4f}")

    # ---- Aggregate ----
    baseline_arr = np.asarray(per_kf_baseline_rae)
    ttt_arr = np.asarray(per_kf_ttt_rae)
    baseline_mean = float(baseline_arr.mean())
    baseline_std = float(baseline_arr.std(ddof=1))
    ttt_mean = float(ttt_arr.mean())
    ttt_std = float(ttt_arr.std(ddof=1))
    delta_mean = ttt_mean - baseline_mean
    print("\n" + "=" * 78)
    print("AGGREGATE")
    print("=" * 78)
    print(f"   per-kf BASELINE RAE = {baseline_arr.tolist()}")
    print(f"   per-kf TTT      RAE = {ttt_arr.tolist()}")
    print(f"   BASELINE mean = {baseline_mean:.4f}  std = {baseline_std:.4f}")
    print(f"   TTT      mean = {ttt_mean:.4f}  std = {ttt_std:.4f}")
    print(f"   delta TTT vs BASE = {delta_mean:+.4f}")

    # ---- Gate ----
    if ttt_mean < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif ttt_mean < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"\n   verdict = {verdict}  (PROMOTE<{GATE_PROMOTE}  "
          f"MARGINAL<{GATE_MARGINAL})")

    # ---- Deploy artifacts ----
    # pred_oof on 253 = mean over kf_seeds of (anchor + ttt_oof)
    pred_oof_unb = ttt_oof_accum.mean(axis=0).astype(np.float32)
    base_oof_unb = base_oof_accum.mean(axis=0).astype(np.float32)

    # te deploy: refit base on ALL 253 (full unblind), then run TTT per test row.
    # For 513 test rows, pick masked col per row vs the same med/mad on unblind X.
    print("\n" + "-" * 78)
    print("STEP 4: deploy refit -- base on all 253, TTT for each of 513 te rows")
    print("-" * 78)
    base_booster_full = _fit_base_booster(
        X_unb_K.astype(np.float32), residual, seed=BASE_SEED,
    )
    med_full = np.median(X_unb_K.astype(np.float32), axis=0)
    mad_full = np.median(
        np.abs(X_unb_K.astype(np.float32) - med_full[None, :]), axis=0,
    )
    te_resid_ttt = np.zeros(n_test, dtype=np.float64)
    base_te_resid = base_booster_full.predict(X_te_K).astype(np.float64)
    for i in range(n_test):
        te_resid_ttt[i] = _ttt_predict_one(
            base_booster_full, X_te_K[i].astype(np.float32),
            X_unb_K.astype(np.float32), med_full, mad_full,
        )
        if (i + 1) % 100 == 0:
            print(f"   te TTT row {i+1}/{n_test}  "
                  f"wall={time.time()-t0:.1f}s")
    te_pred_ttt = te_anchor_513 + te_resid_ttt
    te_pred_base = te_anchor_513 + base_te_resid

    # In-sample evaluation on unblind slice (deploy refit)
    te_pred_ttt_unb = te_pred_ttt[unb_idx]
    te_pred_base_unb = te_pred_base[unb_idx]
    insample_ttt_rae = float(rae(y_unb, te_pred_ttt_unb))
    insample_base_rae = float(rae(y_unb, te_pred_base_unb))
    print(f"   deploy refit in-sample BASE RAE = {insample_base_rae:.4f}")
    print(f"   deploy refit in-sample TTT  RAE = {insample_ttt_rae:.4f}")

    # ---- Save ----
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy", pred_oof_unb)
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", te_pred_ttt.astype(np.float32))
    print(f"\n   [save] {DATA_PROCESSED / (TAG + '_pred_oof.npy')}")
    print(f"   [save] {DATA_PROCESSED / ('te_' + TAG + '.npy')}")

    summary = {
        "tag": TAG,
        "method": ("test_time_training_warmstart_lgbm_chemprop_aux_residual_K20"),
        "anchor": "chemprop_aux",
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "feature_set": "K=20 nb2240 RFE survivors (5-way nb2103 feature matrix)",
        "base_model": "LightGBM(MSE) 300 trees @ lr=0.03",
        "ttt_extra_rounds": TTT_N_EXTRA,
        "ttt_learning_rate": TTT_LR,
        "ttt_aux_task": ("predict masked feature j* from other 19 features "
                         "(per-row argmax z-score off-manifold col)"),
        "kf_seeds": KF_SEEDS,
        "resid_folds": RESID_FOLDS,
        "n_unb": n_unb,
        "rae_anchor_chemprop_aux": rae_anchor,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "nb2240_K20_ref": NB2240_K20_REF,
        "nb2660_K18_deep30_ref": NB2660_K18_DEEP30,
        "per_kf_baseline_rae": [float(r) for r in per_kf_baseline_rae],
        "per_kf_ttt_rae": [float(r) for r in per_kf_ttt_rae],
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "ttt_mean": ttt_mean,
        "ttt_std": ttt_std,
        "mean_rae": ttt_mean,
        "delta_ttt_vs_base": delta_mean,
        "delta_ttt_vs_nb2240_K20": ttt_mean - NB2240_K20_REF,
        "delta_ttt_vs_nb2660_K18_deep30": ttt_mean - NB2660_K18_DEEP30,
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "deploy_refit_insample_base_rae": insample_base_rae,
        "deploy_refit_insample_ttt_rae": insample_ttt_rae,
        "pred_oof_path": str(DATA_PROCESSED / f"{TAG}_pred_oof.npy"),
        "te_path": str(DATA_PROCESSED / f"te_{TAG}.npy"),
        "pre_unblind_clean": True,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_p = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_p, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_p}")
    print(f"\n[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "rae_anchor_chemprop_aux",
        "baseline_mean", "baseline_std",
        "ttt_mean", "ttt_std",
        "delta_ttt_vs_base",
        "delta_ttt_vs_nb2240_K20",
        "delta_ttt_vs_nb2660_K18_deep30",
        "verdict",
        "deploy_refit_insample_base_rae",
        "deploy_refit_insample_ttt_rae",
        "wall_sec",
    ):
        print(f"  {k}: {res.get(k)}")
