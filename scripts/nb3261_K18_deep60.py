"""nb3261 -- K=18 LGBM with 60 seeds (deep-60 bag), then nb3173-style learned-clip.

NEW PARADIGM (cycle 261+):
    Tests whether MORE seeds (60 vs the standard 30) reduces residual bag
    variance further on the K=18 substrate. Cycle-160 deep-30 rule was
    derived from nb2060's 4.7x under-dispersion of 5-seed std vs 30-seed
    std. The natural follow-up: does a 60-seed bag continue the trend, or
    has the bag already saturated at 30 seeds?

    If 60-seed pre-clip OOF RAE materially improves vs nb2960 K=18 deep-30
    (0.4536), OR if learned-clip on the 60-seed bag clears the 0.4423 gate
    (the new aspirational ceiling beyond nb3200's 0.4424 30-seed deep
    learned-clip), then deeper bags carry residual signal beyond 30.

    All K=18 residual seeds are independent random_state's on the same
    chemprop_aux residual + same K=18 feature slice. Bag is a simple mean.

PROTOCOL:
    STEP A  -- Rebuild K=18 deep-60 with seeds 0..59 (n=60 fresh seeds,
               distinct from nb2960's {3001..3030} deep-30 cache). Uses
               IDENTICAL recipe to nb2960 / nb3050 / nb3060: chemprop_aux
               anchor + residual LGBM on K=18 feature slice of nb2231's
               117-col 5-way feature matrix (AtomPair + MACCS + Mordred +
               ChempropEmbed + Avalon + ChEMBL kNN + mean_sim).
               Save K=18 deep-60 bag-mean OOF (253,) and te (513,).

    STEP B  -- Apply nb3173-style learned per-fold (q_low, q_high) clip on
               the K=18 deep-60 OOF, with 15 fresh outer kf_seeds
               {1216..1230} for scaffold-CV. Inner grid:
                   q_low in {0.01, 0.05, 0.10}
                   q_high in {0.90, 0.95, 0.98, 0.99}
               Per outer fold, pick (q_l*, q_h*) that minimize fold-train
               RAE; apply to fold-val; stitch into oof_clip; pool 5 folds
               -> pooled_rae. Aggregate over 15 kf_seeds.

    STEP C  -- Deploy: pick (q_l*, q_h*) on FULL 253 by same inner search,
               apply to the K=18 deep-60 te (513,) -> te_nb3261.

GATE (on 15 wide-seed mean pooled RAE after learned-clip):
    mean < 0.4423 -> "BETTER"
        (60-seed bag + learned-clip clears the new aspirational ceiling
         below nb3200's 0.4424 30-seed deep learned-clip. Confirms that
         deeper bags carry additional residual signal.)
    else          -> "FAIL"
        (60-seed bag does NOT improve over 30-seed; bag saturated by 30.
         No ladder change.)

References:
    nb2960 K=18 deep-30 OOF RAE        = 0.4536  <- 30-seed bag baseline
    nb2991 K=18 wide-seed mean         = 0.4536  (variance check, identical)
    nb3201 K=18 deep-30 + learned-clip = 0.4459  (15-seed mean)
    nb3200 nb3090 deep-30 learned-clip = 0.4424  (30-seed mean)
    nb3173 wide-15-seed learned-clip   = 0.4437
    nb3170 fixed q05/q95               = 0.4437
    nb3030 wide-seed ceiling           = 0.4509
    nb2171 prior post-hoc top          = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/te_chemprop_aux.npy
    data/processed/nb2231_summary.json  (K=18 idx via RFE trajectory)
    data/processed/nb1352_summary.json
    data/processed/nb1392_summary.json
    data/processed/nb1484_summary.json
    data/processed/nb1523_summary.json
    data/processed/nb1524_summary.json
    data/processed/nb1541_summary.json
    data/processed/te_atompair.npy
    data/processed/te_maccs.npy
    data/processed/te_chemprop_embed_300.npy
    data/processed/te_avalon512.npy
    C:/pxr_artifacts/nb1030/X_mordred_test.npy

Outputs:
    data/processed/nb3261_summary.json
    data/processed/nb3261_K18_60seed_oof.npy   (253,) float32 -- K=18 deep-60 OOF
    data/processed/nb3261_K18_60seed_te.npy    (513,) float32 -- K=18 deep-60 te
    data/processed/nb3261_pred_oof.npy         (253,) float32 -- median-seed post-clip OOF
    data/processed/te_nb3261.npy               (513,) float32 -- deploy post-clip te
    submissions/nb3261_K18_deep60.csv          (only on BETTER verdict)
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
from sklearn.model_selection import KFold

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko, standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3261"
PARENT_TAG = "nb2960_K18_deep60_rebuild_then_learned_clip"
K_LABEL = "K18"
K_TARGET_REBUILD = 18

# -- Anchor + residual params (IDENTICAL recipe to nb2960 / nb3050 / nb3060) ---
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
RESID_FOLDS = 5
RESID_SEEDS_DEEP = list(range(0, 60))   # 60 seeds {0..59} per task spec

# -- Feature cache paths ------------------------------------------------------
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

# -- ChEMBL kNN params (identical to nb3050/nb3060/nb2960) --------------------
KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

# -- CV protocol (learned-clip step) ------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))  # 15 FRESH kf_seeds {1216..1230} per task spec

# -- Per-fold grid (nb3173-style; same as nb3200/nb3232) ----------------------
Q_LOW_GRID = [0.01, 0.05, 0.10]
Q_HIGH_GRID = [0.90, 0.95, 0.98, 0.99]

# -- Gate ---------------------------------------------------------------------
GATE_BETTER = 0.4423

# -- References ---------------------------------------------------------------
REF_K18_DEEP30 = 0.4536          # nb2960 K=18 deep-30 OOF
REF_NB2991_WIDE = 0.4536         # variance-check identical
REF_NB3201_K18_CLIP = 0.4459     # learned-clip on K=18 deep-30
REF_NB3200_DEEP30_CLIP = 0.4424  # learned-clip on nb3090 wide-bag deep-30
REF_NB3173_CLIP = 0.4437         # learned-clip on nb3080 wide-bag wide-15
REF_NB3170_FIXED = 0.4437        # fixed q05/q95 on nb3080
REF_NB3030 = 0.4509              # wide-seed pyramid ceiling
REF_NB2171 = 0.4682              # prior post-hoc-blend top
CHEMPROP_AUX_REF = 0.6216


# ============================================================================
# helpers (lifted verbatim from nb2960/nb3060)
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


def _residual_cross_fit_one_seed(X, residual, seed):
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _train_full_then_predict_te(X_unb, residual, X_te, seed):
    mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
    mdl.fit(X_unb, residual)
    return mdl.predict(X_te).astype(np.float32)


def reconstruct_K_from_trajectory(nb2231_sum, K_target):
    """Reconstruct surviving feature indices at K_target from nb2231 RFE
    trajectory (verbatim from nb3050 / nb3000 / nb2631 / nb3010 / nb3014 /
    nb3020 / nb3060)."""
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
    """117-col matrix identical to nb3050 / nb2604 / nb2631 / nb2960 / nb3000 /
    nb3020 / nb3060."""
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


def build_K_60seed_bag(K_label, K_idx, X_te_full, unb_idx, anchor,
                       residual, te_anchor_513, n_test, n_unb, seeds):
    """Build deep-60 bag-mean OOF (253,) + te (513,) for one K-pyramid."""
    X_te_K = X_te_full[:, K_idx].astype(np.float32)
    X_unb_K = X_te_K[unb_idx]
    print(f"   [{K_label}] X_unb_K = {X_unb_K.shape}  X_te_K = {X_te_K.shape}")
    sum_unb = np.zeros(n_unb, dtype=np.float64)
    sum_te = np.zeros(n_test, dtype=np.float64)
    per_seed_rae = []
    per_seed_oof = []  # keep all 60 OOFs so we can compute running bag stats
    for i, s in enumerate(seeds):
        ts = time.time()
        resid_oof = _residual_cross_fit_one_seed(X_unb_K, residual, s)
        pred_unb_s = anchor + resid_oof
        sum_unb += pred_unb_s
        per_seed_rae.append(float(rae(anchor + residual, pred_unb_s)))
        per_seed_oof.append(pred_unb_s.astype(np.float32))
        te_resid_s = _train_full_then_predict_te(X_unb_K, residual, X_te_K, s)
        pred_te_s = te_anchor_513 + te_resid_s
        sum_te += pred_te_s
        if (i % 5) == 0 or i == len(seeds) - 1:
            print(f"      [{K_label}] seed={s:4d}  "
                  f"rae={per_seed_rae[-1]:.4f}  "
                  f"wall={time.time()-ts:.2f}s  ({i+1}/{len(seeds)})")
    bag_oof_unb = sum_unb / len(seeds)
    bag_te_513 = sum_te / len(seeds)
    return bag_oof_unb, bag_te_513, per_seed_rae, per_seed_oof


# ============================================================================
# learned-clip pipeline (nb3173-style)
# ============================================================================

def _pick_best_clip(
    y_tr: np.ndarray,
    pred_tr: np.ndarray,
) -> tuple[float, float, float, float]:
    """Inner grid search: pick (q_low*, q_high*) minimizing fold-train RAE."""
    best_rae = np.inf
    best_ql = Q_LOW_GRID[0]
    best_qh = Q_HIGH_GRID[-1]
    best_lo = float(np.quantile(y_tr, best_ql))
    best_hi = float(np.quantile(y_tr, best_qh))
    for ql in Q_LOW_GRID:
        lo = float(np.quantile(y_tr, ql))
        for qh in Q_HIGH_GRID:
            hi = float(np.quantile(y_tr, qh))
            if hi <= lo:
                continue
            clipped = np.clip(pred_tr, lo, hi)
            r = float(rae(y_tr, clipped))
            if r < best_rae:
                best_rae = r
                best_ql = ql
                best_qh = qh
                best_lo = lo
                best_hi = hi
    return best_ql, best_qh, best_lo, best_hi


def _run_one_clip_seed(
    pred_base: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Run learned-clip pipeline at a single outer kf_seed."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_clip = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_ql = []
    fold_qh = []
    fold_lo = []
    fold_hi = []
    fold_clipped_lo = []
    fold_clipped_hi = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        ql, qh, lo, hi = _pick_best_clip(y_unb[tr_loc], pred_base[tr_loc])
        fold_ql.append(ql)
        fold_qh.append(qh)
        fold_lo.append(lo)
        fold_hi.append(hi)
        val_pred = pred_base[va_loc]
        n_lo = int(np.sum(val_pred < lo))
        n_hi = int(np.sum(val_pred > hi))
        fold_clipped_lo.append(n_lo)
        fold_clipped_hi.append(n_hi)
        clipped = np.clip(val_pred, lo, hi)
        oof_clip[va_loc] = clipped
        fold_val_raes.append(float(rae(y_unb[va_loc], clipped)))

    if np.isnan(oof_clip).any():
        raise RuntimeError(
            f"kf_seed={kf_seed}: scaffold splits did not cover all rows"
        )
    pooled = float(rae(y_unb, oof_clip))
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(fold_val_raes)),
        "per_fold_val_rae_std": float(np.std(fold_val_raes, ddof=1)),
        "fold_ql": fold_ql,
        "fold_qh": fold_qh,
        "fold_lo_mean": float(np.mean(fold_lo)),
        "fold_hi_mean": float(np.mean(fold_hi)),
        "n_clipped_lo": int(np.sum(fold_clipped_lo)),
        "n_clipped_hi": int(np.sum(fold_clipped_hi)),
        "oof": oof_clip,
    }


# ============================================================================
# main
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(
        f"{TAG} -- K={K_TARGET_REBUILD} deep-{len(RESID_SEEDS_DEEP)} bag rebuild "
        f"+ nb3173-style learned per-fold clip"
    )
    print(
        f"          residual seeds = {RESID_SEEDS_DEEP[0]}..{RESID_SEEDS_DEEP[-1]} "
        f"(n={len(RESID_SEEDS_DEEP)})  [NEW PARADIGM: 60-seed bag vs std 30-seed]"
    )
    print(f"          Q_LOW_GRID  = {Q_LOW_GRID}")
    print(f"          Q_HIGH_GRID = {Q_HIGH_GRID}")
    print(
        f"          outer kf_seeds = {KF_SEEDS[0]}..{KF_SEEDS[-1]} "
        f"(n={len(KF_SEEDS)})"
    )
    print(f"          gate: per-fold-mean < {GATE_BETTER:.4f} -> BETTER else FAIL")
    print("=" * 78)

    # -- Load truth, anchor ---------------------------------------------------
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
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] chemprop_aux te[unb_idx] RAE = {rae_anchor:.4f} "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor

    # -- Reconstruct K=18 indices --------------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 1: reconstruct K={K_TARGET_REBUILD} idx from nb2231 RFE trajectory")
    print("-" * 78)
    if not NB2231_SUMMARY.exists():
        raise FileNotFoundError(f"missing {NB2231_SUMMARY}")
    with open(NB2231_SUMMARY) as f:
        nb2231 = json.load(f)
    K18_idx = np.array(reconstruct_K_from_trajectory(nb2231, K_TARGET_REBUILD), dtype=int)
    if len(K18_idx) != K_TARGET_REBUILD:
        raise ValueError(f"K={K_TARGET_REBUILD} idx returned {len(K18_idx)} cols")
    print(f"   K={K_TARGET_REBUILD} idx (n={len(K18_idx)}): {K18_idx.tolist()}")

    # -- Build 117-col matrix -------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 2: rebuild 117-col 5-way feature matrix")
    print("-" * 78)
    X_te_full, chembl_pool_size = build_117col_feature_matrix(te_smiles, n_test)
    print(f"   X_te_full = {X_te_full.shape}  chembl_pool={chembl_pool_size}")

    # -- Build K=18 deep-60 ---------------------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 3: K={K_TARGET_REBUILD} residual-LGBM with "
          f"{len(RESID_SEEDS_DEEP)} fresh seeds (deep-60 bag)")
    print("-" * 78)
    bag_oof_K18, bag_te_K18, per_seed_rae_K18, per_seed_oof_K18 = build_K_60seed_bag(
        K_LABEL, K18_idx, X_te_full, unb_idx, anchor, residual,
        te_anchor_513, n_test, n_unb, RESID_SEEDS_DEEP,
    )
    bag_mean_rae_K18 = float(rae(y_unb, bag_oof_K18))
    per_seed_arr_K18 = np.array(per_seed_rae_K18, dtype=np.float64)
    print(f"\n   [{K_LABEL}] per-seed RAE mean = {per_seed_arr_K18.mean():.4f}  "
          f"std = {per_seed_arr_K18.std(ddof=1):.4f}")
    print(f"   [{K_LABEL}] 60-seed BAG-MEAN OOF RAE = {bag_mean_rae_K18:.4f}")
    print(f"   [{K_LABEL}] delta vs nb2960 K=18 deep-30 "
          f"({REF_K18_DEEP30:.4f}) = {bag_mean_rae_K18 - REF_K18_DEEP30:+.4f}")
    K18_oof_path = DATA_PROCESSED / f"{TAG}_K18_60seed_oof.npy"
    K18_te_path = DATA_PROCESSED / f"{TAG}_K18_60seed_te.npy"
    np.save(K18_oof_path, bag_oof_K18.astype(np.float32))
    np.save(K18_te_path, bag_te_K18.astype(np.float32))
    print(f"   [save] {K18_oof_path}")
    print(f"   [save] {K18_te_path}")

    # Running bag RAE at sizes 5, 10, 15, 30, 45, 60 (variance saturation check)
    bag_size_check = [5, 10, 15, 30, 45, 60]
    bag_size_check = [b for b in bag_size_check if b <= len(per_seed_oof_K18)]
    cum_sum_unb = np.zeros(n_unb, dtype=np.float64)
    running_bag_rae = {}
    for i, p in enumerate(per_seed_oof_K18, start=1):
        cum_sum_unb += p
        if i in bag_size_check:
            running_bag_rae[str(i)] = round(
                float(rae(y_unb, cum_sum_unb / i)), 4
            )
    print(f"\n   [{K_LABEL}] running bag-RAE (variance saturation check):")
    for k, v in running_bag_rae.items():
        print(f"        n_seeds={k:>3s}: bag_RAE = {v:.4f}")

    # Cross-check vs cached nb2960 K=18 deep-30 (sanity: independent recipe)
    cached_oof_path = DATA_PROCESSED / "nb2960_K18_30seed_oof.npy"
    cached_check = None
    if cached_oof_path.exists():
        cached_oof = np.load(cached_oof_path).astype(np.float64)
        cached_rae = float(rae(y_unb, cached_oof))
        pearson_corr = float(np.corrcoef(bag_oof_K18, cached_oof)[0, 1])
        cached_check = {
            "cached_nb2960_K18_30seed_rae": round(cached_rae, 4),
            "pearson_vs_60seed_bag": round(pearson_corr, 4),
        }
        print(f"\n   sanity: nb2960 K=18 30-seed cache RAE = {cached_rae:.4f}  "
              f"(ref {REF_K18_DEEP30:.4f})")
        print(f"   pearson (cached 30-seed vs new 60-seed) = {pearson_corr:.4f}")

    # -- Scaffolds (kf_seed independent) -------------------------------------
    print("\n" + "-" * 78)
    print("STEP 4: scaffolds for outer CV")
    print("-" * 78)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"   n_unique_scaffolds = {n_unique_scaf}")

    # -- Learned-clip 15-seed sweep on K=18 deep-60 ---------------------------
    print("\n" + "-" * 78)
    print(f"STEP 5: WIDE-SEED LEARNED-CLIP SWEEP -- {len(KF_SEEDS)} fresh kf_seeds "
          f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}")
    print("-" * 78)
    pred_base = bag_oof_K18.astype(np.float64)
    seed_records = []
    pooled_raes = []
    per_fold_means = []  # per task spec: gate is on per-fold-mean
    oof_stack = []
    all_fold_ql = []
    all_fold_qh = []
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_clip_seed(pred_base, y_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        per_fold_means.append(res["per_fold_val_rae_mean"])
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
            "fold_lo_mean": round(res["fold_lo_mean"], 4),
            "fold_hi_mean": round(res["fold_hi_mean"], 4),
            "n_clipped_lo": res["n_clipped_lo"],
            "n_clipped_hi": res["n_clipped_hi"],
        })
        print(
            f"   kf={s}: pooled_RAE={res['pooled_rae']:.4f}  "
            f"per_fold_mean={res['per_fold_val_rae_mean']:.4f}  "
            f"ql={res['fold_ql']}  qh={res['fold_qh']}  "
            f"clipped(lo,hi)=({res['n_clipped_lo']},{res['n_clipped_hi']})  "
            f"wall={time.time()-ts:.2f}s"
        )

    arr_pool = np.asarray(pooled_raes, dtype=np.float64)
    arr_pfm = np.asarray(per_fold_means, dtype=np.float64)
    n_s = len(arr_pool)
    pooled_mean = float(arr_pool.mean())
    pooled_std = float(arr_pool.std(ddof=1)) if n_s > 1 else 0.0
    per_fold_mean = float(arr_pfm.mean())   # GATE METRIC per task
    per_fold_std = float(arr_pfm.std(ddof=1)) if n_s > 1 else 0.0
    sem_pool = pooled_std / np.sqrt(n_s) if n_s > 1 else 0.0
    sem_pfm = per_fold_std / np.sqrt(n_s) if n_s > 1 else 0.0
    t_mult = 2.145  # df=14
    ci_low_pool = pooled_mean - t_mult * sem_pool
    ci_high_pool = pooled_mean + t_mult * sem_pool
    ci_low_pfm = per_fold_mean - t_mult * sem_pfm
    ci_high_pfm = per_fold_mean + t_mult * sem_pfm

    # Most-picked q values across all 5*15=75 folds
    ql_counter = Counter(all_fold_ql)
    qh_counter = Counter(all_fold_qh)
    ql_mode = ql_counter.most_common(1)[0][0]
    qh_mode = qh_counter.most_common(1)[0][0]

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds)")
    print("-" * 78)
    print(f"   pooled_RAE       mean = {pooled_mean:.4f}  std = {pooled_std:.4f}  "
          f"95% CI [{ci_low_pool:.4f}, {ci_high_pool:.4f}]")
    print(f"   per_fold_mean    mean = {per_fold_mean:.4f}  std = {per_fold_std:.4f}  "
          f"95% CI [{ci_low_pfm:.4f}, {ci_high_pfm:.4f}]   <- GATE METRIC")
    print(f"   pooled  min/max  [{arr_pool.min():.4f}, {arr_pool.max():.4f}]")
    print(f"   per-fm  min/max  [{arr_pfm.min():.4f}, {arr_pfm.max():.4f}]")
    print(f"\n   ql distribution (75 folds) = {dict(ql_counter)}")
    print(f"   qh distribution (75 folds) = {dict(qh_counter)}")
    print(f"   ql_mode = {ql_mode}   qh_mode = {qh_mode}")

    # -- Deploy: pick (q_low, q_high) on FULL 253 + apply to te ---------------
    print("\n" + "-" * 78)
    print("STEP 6: DEPLOY -- pick clip on full 253 -> apply to K=18 deep-60 te(513)")
    print("-" * 78)
    deploy_ql, deploy_qh, deploy_lo, deploy_hi = _pick_best_clip(y_unb, pred_base)
    te_base = bag_te_K18.astype(np.float64)
    te_pred = np.clip(te_base, deploy_lo, deploy_hi).astype(np.float32)
    n_te_lo = int(np.sum(te_base < deploy_lo))
    n_te_hi = int(np.sum(te_base > deploy_hi))
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"   deploy clip = (q{deploy_ql:.2f}, q{deploy_qh:.2f}) -> "
          f"({deploy_lo:.3f}, {deploy_hi:.3f}) from full 253 y")
    print(f"   te clipped: lo={n_te_lo}/513  hi={n_te_hi}/513  "
          f"total={n_te_lo + n_te_hi}/513")
    print(f"   te(513) mean={te_pred.mean():.3f}  std={te_pred.std():.3f}  "
          f"min={te_pred.min():.3f}  max={te_pred.max():.3f}")
    print(f"   te[unb] in-sample RAE = {te_unb_in_rae:.4f}")

    # Median-seed OOF for storage
    med_seed_idx = int(np.argsort(arr_pool)[n_s // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(f"   median seed = {median_seed} (pooled_rae={arr_pool[med_seed_idx]:.4f})")

    # -- Gate -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 7: GATE")
    print("-" * 78)
    if per_fold_mean < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE nb3261 candidate. 60-seed K={K_TARGET_REBUILD} bag + "
            f"learned-clip per-fold-mean {per_fold_mean:.4f} clears the "
            f"strict aspirational gate {GATE_BETTER:.4f}, beating nb3200 "
            f"({REF_NB3200_DEEP30_CLIP:.4f}, learned-clip on nb3090 deep-30) "
            f"by {REF_NB3200_DEEP30_CLIP - per_fold_mean:+.4f}. 60-seed bag "
            f"carries additional residual signal beyond 30 seeds. Run a "
            f"30-seed wide-verification (nb326x_deep30_verify) before LB fire "
            f"to confirm under cycle-160 deep-30 rule. If verified, this "
            f"becomes the new PRIMARY-1; predicted LB under +0.0045 PRE "
            f"delta calibration = {per_fold_mean + 0.0045:.4f}."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT nb3261. 60-seed K={K_TARGET_REBUILD} bag + learned-clip "
            f"per-fold-mean {per_fold_mean:.4f} fails strict gate "
            f"{GATE_BETTER:.4f} (delta {per_fold_mean - GATE_BETTER:+.4f}). "
            f"Bag has SATURATED at 30 seeds -- doubling to 60 does not "
            f"extract additional residual signal. Close the deep-60 axis on "
            f"K=18 substrate; cycle-160 deep-30 rule remains the gate standard. "
            f"nb3200 ({REF_NB3200_DEEP30_CLIP:.4f}) stays PRIMARY-1; no "
            f"ladder change. Compare: bag-mean OOF pre-clip RAE = "
            f"{bag_mean_rae_K18:.4f} vs nb2960 K=18 deep-30 "
            f"({REF_K18_DEEP30:.4f}) -- delta "
            f"{bag_mean_rae_K18 - REF_K18_DEEP30:+.4f}."
        )
    print(f"   per_fold_mean   = {per_fold_mean:.4f}  (gate: < {GATE_BETTER:.4f})")
    print(f"   pooled_mean     = {pooled_mean:.4f}  (informational)")
    print(f"   verdict         = {verdict}")
    print(f"   ladder action   = {ladder_action}")

    # -- Save artifacts -------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 8: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_for_save)
    np.save(te_path, te_pred)
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_K18_deep60.csv"
    if verdict == "BETTER":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": te_pred,
        }).to_csv(sub_csv, index=False)
        print(f"   [save] {sub_csv}")
    else:
        print(f"   [skip] verdict={verdict}; no submission CSV written")

    delta_vs_nb3200 = per_fold_mean - REF_NB3200_DEEP30_CLIP
    delta_vs_nb3201 = per_fold_mean - REF_NB3201_K18_CLIP
    delta_vs_nb3173 = per_fold_mean - REF_NB3173_CLIP
    delta_vs_nb3030 = per_fold_mean - REF_NB3030
    delta_vs_K18_deep30 = bag_mean_rae_K18 - REF_K18_DEEP30

    summary = {
        "tag": TAG,
        "parent_tag": PARENT_TAG,
        "method": "K18_deep60_residual_LGBM_then_nb3173_learned_clip",
        "paradigm": "60_seed_bag_vs_std_30_seed_does_deeper_bag_help",
        "anchor_pre_unblind": True,
        "anchor_te_path": str(ANCHOR_TE_PATH),
        "K_target_rebuilt": K_TARGET_REBUILD,
        "K_label": K_LABEL,
        "K18_idx_in_117": K18_idx.tolist(),
        # bag
        "resid_seeds_deep": RESID_SEEDS_DEEP,
        "n_seeds_deep": len(RESID_SEEDS_DEEP),
        "K18_per_seed_rae": per_seed_rae_K18,
        "K18_per_seed_rae_mean": round(float(per_seed_arr_K18.mean()), 4),
        "K18_per_seed_rae_std": round(float(per_seed_arr_K18.std(ddof=1)), 4),
        "K18_per_seed_rae_min": round(float(per_seed_arr_K18.min()), 4),
        "K18_per_seed_rae_max": round(float(per_seed_arr_K18.max()), 4),
        "K18_60seed_bag_mean_oof_rae": round(bag_mean_rae_K18, 4),
        "K18_60seed_oof_path": str(K18_oof_path),
        "K18_60seed_te_path": str(K18_te_path),
        "K18_60seed_te_mean": float(bag_te_K18.mean()),
        "K18_60seed_te_std": float(bag_te_K18.std()),
        "K18_60seed_te_min": float(bag_te_K18.min()),
        "K18_60seed_te_max": float(bag_te_K18.max()),
        "running_bag_rae": running_bag_rae,
        "delta_K18_60seed_vs_nb2960_K18_30seed": round(delta_vs_K18_deep30, 4),
        "cached_30seed_sanity": cached_check,
        # learned-clip
        "q_low_grid": Q_LOW_GRID,
        "q_high_grid": Q_HIGH_GRID,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": n_s,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "seed_records": seed_records,
        "pooled_rae_array": [round(float(v), 4) for v in pooled_raes],
        "per_fold_mean_array": [round(float(v), 4) for v in per_fold_means],
        "pooled_mean_rae": round(pooled_mean, 4),
        "pooled_std_rae": round(pooled_std, 4),
        "pooled_sem_rae": round(sem_pool, 4),
        "pooled_ci95_low": round(ci_low_pool, 4),
        "pooled_ci95_high": round(ci_high_pool, 4),
        "per_fold_mean_rae": round(per_fold_mean, 4),
        "per_fold_std_rae": round(per_fold_std, 4),
        "per_fold_sem_rae": round(sem_pfm, 4),
        "per_fold_ci95_low": round(ci_low_pfm, 4),
        "per_fold_ci95_high": round(ci_high_pfm, 4),
        "pooled_min_rae": round(float(arr_pool.min()), 4),
        "pooled_max_rae": round(float(arr_pool.max()), 4),
        "per_fold_min_rae": round(float(arr_pfm.min()), 4),
        "per_fold_max_rae": round(float(arr_pfm.max()), 4),
        "ql_distribution": {str(k): int(v) for k, v in ql_counter.items()},
        "qh_distribution": {str(k): int(v) for k, v in qh_counter.items()},
        "ql_mode": float(ql_mode),
        "qh_mode": float(qh_mode),
        # references
        "ref_K18_deep30": REF_K18_DEEP30,
        "ref_nb3201_K18_clip": REF_NB3201_K18_CLIP,
        "ref_nb3200_deep30_clip": REF_NB3200_DEEP30_CLIP,
        "ref_nb3173_clip": REF_NB3173_CLIP,
        "ref_nb3030_ceiling": REF_NB3030,
        "ref_nb2171": REF_NB2171,
        "delta_vs_nb3200": round(delta_vs_nb3200, 4),
        "delta_vs_nb3201": round(delta_vs_nb3201, 4),
        "delta_vs_nb3173": round(delta_vs_nb3173, 4),
        "delta_vs_nb3030": round(delta_vs_nb3030, 4),
        # deploy
        "deploy_ql": float(deploy_ql),
        "deploy_qh": float(deploy_qh),
        "deploy_lo": round(deploy_lo, 4),
        "deploy_hi": round(deploy_hi, 4),
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
    print(f"   K=18 60-seed bag-mean OOF RAE     = {bag_mean_rae_K18:.4f}  "
          f"(vs nb2960 K=18 deep-30 {REF_K18_DEEP30:.4f}, "
          f"delta {delta_vs_K18_deep30:+.4f})")
    print(f"   running bag-RAE                   = {running_bag_rae}")
    print(f"   learned-clip per-fold-mean (15s)  = {per_fold_mean:.4f} "
          f"+/- {per_fold_std:.4f}  <- GATE METRIC")
    print(f"   learned-clip pooled mean    (15s) = {pooled_mean:.4f} "
          f"+/- {pooled_std:.4f}")
    print(f"   delta vs nb3200 (0.4424)          = {delta_vs_nb3200:+.4f}")
    print(f"   delta vs nb3201 K=18 deep-30 clip = {delta_vs_nb3201:+.4f}")
    print(f"   ql_mode / qh_mode                 = {ql_mode} / {qh_mode}")
    print(f"   te[unb] in-sample                 = {te_unb_in_rae:.4f}")
    print(f"   verdict                           = {verdict}")
    print(f"   wall                              = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "K18_60seed_bag_mean_oof_rae",
        "K18_per_seed_rae_mean",
        "K18_per_seed_rae_std",
        "delta_K18_60seed_vs_nb2960_K18_30seed",
        "running_bag_rae",
        "per_fold_mean_rae",
        "per_fold_std_rae",
        "pooled_mean_rae",
        "pooled_std_rae",
        "pooled_ci95_low",
        "pooled_ci95_high",
        "delta_vs_nb3200",
        "delta_vs_nb3201",
        "delta_vs_nb3173",
        "ql_mode",
        "qh_mode",
        "deploy_ql",
        "deploy_qh",
        "deploy_lo",
        "deploy_hi",
        "n_te_clipped_lo",
        "n_te_clipped_hi",
        "te_unb_in_sample_rae",
        "verdict",
        "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
