"""nb2774 -- K=20 LGBM residual with symmetric (oblivious-style) tree growth.

NEW PARADIGM (tree-structure novelty distinct from prior nb27xx attempts):
    Oblivious (symmetric) trees apply the SAME (feature, threshold) split at
    every node of a given depth -- the canonical CatBoost regularizer.
    LightGBM does not expose native oblivious growth, but we EMULATE the
    inductive bias by combining:

        1. max_depth = 4 with num_leaves = 2**4 = 16  -- force fully-saturated
           depth-4 trees (LGBM defaults to leaf-wise growth and only stops at
           num_leaves OR min_data; equating num_leaves = 2**max_depth removes
           the leaf-wise asymmetry budget).
        2. path_smooth = 1.0           -- shrink each leaf's output toward its
                                          parent (recursively up to root).
                                          This is LGBM's documented analogue of
                                          oblivious-tree shrinkage: leaves in
                                          the same subtree become closer in
                                          value, mimicking the value-tying of
                                          symmetric trees.
        3. min_data_in_leaf = 5        -- prevent leaf-wise from short-circuiting
                                          the saturation; every depth-4 path
                                          stays alive.
        4. n_estimators = 300, lr = 0.03 -- mirror the nb27xx K=20 substrate.

    Net effect: each tree's depth-4 partition is uniform (no leaf-wise
    truncation), and leaf values are recursively pulled toward parent
    averages, approximating the oblivious-tree "same split everywhere at
    depth d" hypothesis without changing model class.

    Distinct from:
      - nb2762 Cauchy loss        : changes the LOSS shape; tree structure
                                     untouched.
      - nb2754 focal loss         : per-row residual amplification, no
                                     structural change.
      - nb2152 colsample sweep    : varies feature-fraction per tree
                                     (asymmetry across trees, not within).
      - nb2750 linear tree        : per-leaf linear model (more parameters per
                                     leaf); orthogonal axis.
      - nb2722 GBR Huber          : sklearn HGB w/ Huber loss; standard
                                     leaf-wise structure.

PROTOCOL:
    1. Build canonical 117-col 5-way feature matrix (AtomPair + MACCS +
       Mordred + ChempropEmbed + Avalon + ChEMBL kNN); restrict to K=20
       cols from nb2240_summary.json k20_surviving_idx_in_117.
    2. anchor = chemprop_aux te[unb_idx]; residual = y_unb - anchor.
    3. For each of 5 kf_seeds {1001..1005}:
         5-fold scaffold CV: per fold, LGBM with num_leaves=16, max_depth=4,
         min_data_in_leaf=5, path_smooth=1.0, n_est=300, lr=0.03.
         Refit-all on residual; predict 513-row test.
       Mean-bag corrected OOF RAE on 253.

GATE:
    < 0.4570 -> PROMOTE
    < 0.4598 -> MARGINAL_BEAT
    else     -> FAIL

OUTPUTS:
    scripts/nb2774_symmetric_lgbm.py
    data/processed/nb2774_summary.json
    data/processed/nb2774_pred_oof.npy   (253,) float32 mean-bag OOF
    data/processed/te_nb2774.npy         (513,) float32 mean-bag deploy
    submissions/nb2774_symmetric_lgbm.csv
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
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2774"

# ---- Anchor + caches ----
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
UNBLIND_IDX = DATA_PROCESSED / "_audit_unblind_idx.npy"
UNBLIND_Y = DATA_PROCESSED / "_audit_unblind_y.npy"

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

# ---- CV protocol ----
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# ---- LGBM hyperparams (oblivious-style symmetric tree emulation) ----
LGBM_MAX_DEPTH = 4
LGBM_NUM_LEAVES = 2 ** LGBM_MAX_DEPTH       # = 16; saturates depth-4 budget
LGBM_MIN_CHILD_SAMPLES = 5                  # min_data_in_leaf
LGBM_PATH_SMOOTH = 1.0                      # parent-shrinkage
LGBM_N_ESTIMATORS = 300
LGBM_LEARNING_RATE = 0.03

# ---- Gates ----
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

# ---- Refs ----
CHEMPROP_AUX_REF = 0.6216
NB2171_REF = 0.4682
NB2240_K20_REF = 0.4630

# ---- ChEMBL kNN config (mirrors nb2754 / nb2743 / nb2630 / nb2762 substrate) ----
KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6


# ============================================================================
# Helpers (identical 117-col substrate as nb2754/nb2743/nb2103/nb2240/nb2604)
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


def _load_chembl_pool() -> pd.DataFrame:
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
    return agg.rename(columns={"src_first": "src"})


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


def _load_npy_test(path, n_test_expected):
    if not path.exists():
        raise FileNotFoundError(f"missing cache: {path}")
    X = np.load(path).astype(np.float32)
    if X.shape[0] != n_test_expected:
        raise ValueError(f"shape mismatch {path}: {X.shape}")
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X


def _load_mordred_test(n_test_expected):
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(f"Mordred cache missing -- run nb1030 first ({mte_p})")
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
    """Identical 117-col matrix as nb2754/nb2743/nb2103/nb2240/nb2604/nb2630."""
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
        if m is None:
            std_test_smiles.append("")
        else:
            std_test_smiles.append(Chem.MolToSmiles(m))
    fp_test = morgan_fp_batch(std_test_smiles)
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim_knn = _knn_predict(
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
            mean_sim_knn.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    if X_te_full.shape[1] != 117:
        raise ValueError(f"feat_dim {X_te_full.shape[1]} != 117")
    return X_te_full, int(len(pool))


# ============================================================================
# Symmetric (oblivious-style) LGBM params
# ============================================================================
def _lgbm_params_symmetric(seed):
    """LGBM hyperparams emulating CatBoost-style oblivious tree growth.

    Combination:
      - max_depth=4 + num_leaves=2**4=16  -> saturates leaf-wise to depth-4
                                              symmetric perfect binary tree
                                              budget (no leaf-wise truncation).
      - min_data_in_leaf=5                -> keeps every depth-4 path alive.
      - path_smooth=1.0                   -> LGBM's leaf-output parent-shrinkage
                                              regularizer; recursively pulls
                                              each leaf value toward its parent
                                              (and ancestors), mimicking the
                                              value-tying of oblivious trees.
    """
    return dict(
        objective="regression",
        max_depth=LGBM_MAX_DEPTH,
        num_leaves=LGBM_NUM_LEAVES,
        n_estimators=LGBM_N_ESTIMATORS,
        learning_rate=LGBM_LEARNING_RATE,
        min_child_samples=LGBM_MIN_CHILD_SAMPLES,
        path_smooth=LGBM_PATH_SMOOTH,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _residual_cv_one_seed(
    X_unb_K: np.ndarray,
    residual: np.ndarray,
    scaffolds: list,
    seed: int,
):
    """5-fold scaffold-CV residual LGBM with oblivious-style symmetric trees."""
    n = len(residual)
    splits = scaffold_kfold_indices(
        scaffolds, n_splits=N_FOLDS, shuffle=True, seed=seed,
    )
    oof = np.full(n, np.nan, dtype=np.float64)
    per_fold_rae = []
    for fi, (tr_loc, va_loc) in enumerate(splits):
        mdl = lgb.LGBMRegressor(**_lgbm_params_symmetric(seed))
        mdl.fit(X_unb_K[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X_unb_K[va_loc])
        per_fold_rae.append(float(rae(residual[va_loc], oof[va_loc])))
    if np.isnan(oof).any():
        raise RuntimeError("scaffold split did not cover all rows")
    return oof, per_fold_rae


def _deploy_refit(
    X_unb_K: np.ndarray,
    residual: np.ndarray,
    X_te_K: np.ndarray,
    seed: int,
) -> np.ndarray:
    mdl = lgb.LGBMRegressor(**_lgbm_params_symmetric(seed))
    mdl.fit(X_unb_K, residual)
    return mdl.predict(X_te_K).astype(np.float32)


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- K=20 LGBM residual w/ symmetric (oblivious-style) tree growth")
    print(f"          max_depth={LGBM_MAX_DEPTH}  num_leaves={LGBM_NUM_LEAVES}"
          f" (=2**{LGBM_MAX_DEPTH}, saturates depth budget)")
    print(f"          min_data_in_leaf={LGBM_MIN_CHILD_SAMPLES}  "
          f"path_smooth={LGBM_PATH_SMOOTH}  n_est={LGBM_N_ESTIMATORS}  "
          f"lr={LGBM_LEARNING_RATE}")
    print(f"          5-fold scaffold CV  kf_seeds={KF_SEEDS}")
    print(f"          GATE: <{GATE_PROMOTE} PROMOTE / "
          f"<{GATE_MARGINAL} MARGINAL_BEAT / else FAIL")
    print("=" * 78)

    # ---- Load test set ----
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values
                if "name" in te.columns else te["Molecule Name"].values)

    unb_idx = np.load(UNBLIND_IDX)
    y_unb = np.load(UNBLIND_Y).astype(np.float64)
    n_unb = len(y_unb)
    print(f"\n[load] n_test={n_test}  n_unb={n_unb}")

    # ---- Step 1: build 117-col feature matrix + K=20 cols ----
    print("\n[step1] building canonical 117-col feature matrix ...")
    X_te_full, chembl_pool_size = build_117col_feature_matrix(te_smiles, n_test)
    print(f"   X_te_full={X_te_full.shape}  chembl_pool={chembl_pool_size}")
    with open(NB2240_SUMMARY) as f:
        nb2240 = json.load(f)
    K20_cols = list(nb2240["k20_surviving_idx_in_117"])
    if len(K20_cols) != 20:
        raise ValueError(f"K20 cols len {len(K20_cols)} != 20")
    X_te_K = X_te_full[:, K20_cols].astype(np.float32)
    X_unb_K = X_te_K[unb_idx]
    print(f"   X_unb_K={X_unb_K.shape}  X_te_K={X_te_K.shape}")

    # ---- Step 2: anchor + residual ----
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"\n[anchor] chemprop_aux te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor

    # ---- Step 3: scaffold groups for the 253 ----
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique 253-row scaffolds = {n_unique_scaf}")

    # ---- Step 4: 5-seed bag ----
    per_seed_oof_corr = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)
    per_seed_te_resid = np.zeros((len(KF_SEEDS), n_test), dtype=np.float64)
    per_seed_oof_rae = []
    per_seed_resid_rae = []
    per_seed_per_fold_rae = []
    print("\n" + "-" * 78)
    print("5-seed scaffold-CV bag (symmetric trees: num_leaves=16, depth=4, "
          "path_smooth=1.0)")
    print("-" * 78)
    for i, seed in enumerate(KF_SEEDS):
        ts = time.time()
        oof_resid, per_fold_resid_rae = _residual_cv_one_seed(
            X_unb_K, residual, unb_scaffolds, seed,
        )
        corr_oof = anchor + oof_resid
        per_seed_oof_corr[i] = corr_oof
        rae_corr = float(rae(y_unb, corr_oof))
        rae_resid = float(rae(residual, oof_resid))
        per_seed_oof_rae.append(rae_corr)
        per_seed_resid_rae.append(rae_resid)
        per_seed_per_fold_rae.append(per_fold_resid_rae)

        te_resid_s = _deploy_refit(X_unb_K, residual, X_te_K, seed)
        per_seed_te_resid[i] = te_resid_s
        print(f"   seed={seed}  corr_OOF_RAE={rae_corr:.4f}  "
              f"resid_OOF_RAE={rae_resid:.4f}  wall={time.time() - ts:.1f}s")

    mean_bag_oof_corr = per_seed_oof_corr.mean(axis=0)
    mean_bag_te_resid = per_seed_te_resid.mean(axis=0)
    te_corrected_513 = (te_anchor_513 + mean_bag_te_resid).astype(np.float32)
    rae_meanbag_oof = float(rae(y_unb, mean_bag_oof_corr))
    rae_meanbag_te_at_unb = float(rae(y_unb, te_corrected_513[unb_idx]))
    print(f"\n   mean-bag OOF RAE = {rae_meanbag_oof:.4f}  "
          f"(per-seed mean {float(np.mean(per_seed_oof_rae)):.4f}  "
          f"std {float(np.std(per_seed_oof_rae)):.4f})  "
          f"vs anchor {rae_meanbag_oof - rae_anchor:+.4f}")

    # ---- Step 5: artefacts ----
    print("\n" + "=" * 78)
    print(f"MEAN-BAG OOF RAE = {rae_meanbag_oof:.4f}")
    print("=" * 78)
    print(f"[summary] vs anchor             = "
          f"{rae_meanbag_oof - rae_anchor:+.4f}")
    print(f"[summary] vs nb2240 K=20 ref    = "
          f"{rae_meanbag_oof - NB2240_K20_REF:+.4f}")
    print(f"[summary] vs nb2171 ceiling     = "
          f"{rae_meanbag_oof - NB2171_REF:+.4f}")

    pred_oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(pred_oof_path, mean_bag_oof_corr.astype(np.float32))
    np.save(te_path, te_corrected_513)
    print(f"\n[save] {pred_oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_symmetric_lgbm.csv"
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
            "K=20 LGBM residual on chemprop_aux anchor with oblivious-style "
            "symmetric tree emulation: num_leaves=2**max_depth=16 (saturates "
            "depth-4 budget, removing leaf-wise asymmetry), "
            "min_data_in_leaf=5, path_smooth=1.0 (parent-shrinkage analogue "
            "of oblivious value-tying); 5-fold scaffold CV mean-bag over 5 "
            "kf_seeds"
        ),
        "paradigm": "symmetric_oblivious_tree_LGBM",
        "novelty": (
            "Tree-structure inductive bias change distinct from prior nb27xx "
            "loss-shape (nb2762 Cauchy, nb2754 focal, nb2722 Huber), "
            "feature-subsampling (nb2152 colsample, nb2166 row-bootstrap), "
            "and linear-leaf (nb2750) attempts. Emulates CatBoost-style "
            "oblivious trees in LGBM via (a) saturating num_leaves to "
            "2**max_depth (full perfect-binary depth-4 partition), (b) "
            "path_smooth=1.0 to recursively shrink leaf outputs toward "
            "parent (mimicking oblivious value-tying), and (c) min_data=5 "
            "to keep every depth-4 path alive. Net effect: trees become "
            "more uniform / less feature-greedy, trading raw fit for "
            "lower-variance leaf estimates -- the regularization story that "
            "made CatBoost competitive on small heterogeneous tabular data."
        ),
        "anchor": "chemprop_aux",
        "anchor_pre_unblind": True,
        "anchor_in_rae": rae_anchor,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "nb2240_k20_ref": NB2240_K20_REF,
        "nb2171_ref": NB2171_REF,
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "n_unique_scaffolds_unb": int(n_unique_scaf),
        "K20_cols": K20_cols,
        "feat_dim_K20": int(X_unb_K.shape[1]),
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "lgbm_num_leaves": LGBM_NUM_LEAVES,
        "lgbm_max_depth": LGBM_MAX_DEPTH,
        "lgbm_n_estimators": LGBM_N_ESTIMATORS,
        "lgbm_learning_rate": LGBM_LEARNING_RATE,
        "lgbm_min_child_samples": LGBM_MIN_CHILD_SAMPLES,
        "lgbm_path_smooth": LGBM_PATH_SMOOTH,
        "per_seed_corr_oof_rae": [float(x) for x in per_seed_oof_rae],
        "per_seed_resid_oof_rae": [float(x) for x in per_seed_resid_rae],
        "per_seed_per_fold_resid_rae": per_seed_per_fold_rae,
        "per_seed_corr_oof_mean_rae": float(np.mean(per_seed_oof_rae)),
        "per_seed_corr_oof_std_rae": float(np.std(per_seed_oof_rae)),
        "rae_meanbag_oof_corr": rae_meanbag_oof,
        "rae_meanbag_te_at_unb_in_sample": rae_meanbag_te_at_unb,
        "mean_rae": rae_meanbag_oof,
        "delta_vs_anchor": rae_meanbag_oof - rae_anchor,
        "delta_vs_nb2240_k20": rae_meanbag_oof - NB2240_K20_REF,
        "delta_vs_nb2171": rae_meanbag_oof - NB2171_REF,
        "te_deploy_mean": float(te_corrected_513.mean()),
        "te_deploy_std": float(te_corrected_513.std()),
        "pred_oof_path": str(pred_oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv),
        "chembl_pool_size": chembl_pool_size,
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "verdict": verdict,
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
        "delta_vs_nb2240_k20",
        "delta_vs_nb2171",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
    print("\n  per_seed_corr_oof_rae:")
    for i, x in enumerate(res.get("per_seed_corr_oof_rae", [])):
        print(f"    kf_seed={KF_SEEDS[i]}  rae={x:.4f}")
