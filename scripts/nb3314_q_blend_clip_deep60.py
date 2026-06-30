"""nb3314 -- q35 quantile-conditional blend + learned per-fold clip, on
            deep-60 anchors {K18-deep60, K19-deep60}.

NEW PARADIGM (does deeper bag at EVERY stage compound?):
    Prior q-blend / learned-clip ladders (nb3070, nb3073, nb3083, nb3220,
    nb3223, nb3263) were all built on deep-30 anchors. The cycle-261 deep-60
    experiments (nb3261 K=18 deep-60) showed the bag-mean OOF RAE on the
    K=18 substrate ALONE saturates at 30 seeds (60-seed bag-mean 0.4650 vs
    30-seed 0.4536 -- 60 seeds did NOT improve a single anchor's bag).

    This script asks a DIFFERENT question: does a deeper bag help when it
    feeds a COMPOSITE pipeline (quantile-conditional 2-anchor blend +
    learned clip)? A deeper bag reduces the residual-LGBM seed variance,
    which may stabilize the q35 hard-split boundary and the learned-clip
    quantile estimates in a way that a single-anchor pre-clip RAE does not
    reveal. The composite may compound the marginal deep-60 stability gains.

    Anchors:
        K18-deep60  -- cached from nb3261 (seeds 0..59 residual LGBM, K=18
                       feature slice of nb2231's 117-col 5-way matrix)
        K19-deep60  -- BUILT HERE (seeds 0..59, IDENTICAL recipe to nb3261
                       but K=19 feature slice; nb3000 only cached K19 deep-30)

    Composite (per outer fold, fold-TRAIN-derived q35 -- no val leakage):
        STEP 1: q35 = 35th percentile of fold-TRAIN K18-deep60 OOF preds.
        STEP 2: quantile-conditional blend (nb3070/nb3073 hard split):
                  row K18 <= q35 -> 0.8*K18 + 0.2*K19   (low / inactive tail)
                  row K18  > q35 -> 0.5*K18 + 0.5*K19   (high / active tail)
        STEP 3: learned clip -- inner grid (q_low, q_high) on fold-TRAIN
                blended output, pick (lo*, hi*) minimizing fold-train RAE.
        STEP 4: val_blend = blend(K18_val, K19_val, q35);
                val_pred  = clip(val_blend, lo*, hi*).
        Stitch 5 fold-vals -> oof_final; pooled_rae + per-fold-mean RAE.
        Repeat for 15 FRESH kf_seeds {1216..1230}.

    Deploy:
        q35_deploy = 35th percentile of FULL-253 K18-deep60 OOF.
        te_blend   = blend(te_K18, te_K19, q35_deploy).
        (lo*, hi*) = learned clip on FULL-253 blended output.
        te_pred    = clip(clip(te_blend, lo*, hi*), 3.0, 9.0).

GATE (on PER-FOLD-MEAN across 15 seeds):
    mean < 0.4423 -> "BETTER"
    else          -> "FAIL"

References (PRE-unblind anchor chain; chemprop_aux anchored):
    nb2960 K18 deep-30 OOF              = 0.4536
    nb3000 K19 deep-30 OOF              = 0.4607
    nb3261 K18 deep-60 OOF              = 0.4650  (single-anchor bag saturated)
    nb3261 K18 deep-60 + learned-clip  = 0.4723  (pf-mean, FAIL)
    nb3070 wide q-cond {K18,K19} deep30 = 0.4477 / 0.4509
    nb3173 best clip-winner            = 0.4422
    nb3223 SLSQP {K18,K19} + clip       = 0.4424
    nb3263 K18-deep30 q-rescale + clip  = (sibling)
    nb2171 prior post-hoc top           = 0.4682
    chemprop_aux anchor                 = 0.6216

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/te_chemprop_aux.npy
    data/processed/nb3261_K18_60seed_oof.npy   (cached K18-deep60 OOF)
    data/processed/nb3261_K18_60seed_te.npy    (cached K18-deep60 te)
    data/processed/nb2231_summary.json         (K=19 idx via RFE trajectory)
    data/processed/nb1352/1392/1484/1523/1524/1541_summary.json
    data/processed/te_atompair.npy, te_maccs.npy,
        te_chemprop_embed_300.npy, te_avalon512.npy
    C:/pxr_artifacts/nb1030/X_mordred_test.npy

Outputs:
    data/processed/nb3314_summary.json
    data/processed/nb3314_K19_60seed_oof.npy   (253,) float32 -- K19 deep-60 OOF
    data/processed/nb3314_K19_60seed_te.npy    (513,) float32 -- K19 deep-60 te
    data/processed/nb3314_pred_oof.npy         (253,) float32 -- median-seed OOF
    data/processed/te_nb3314.npy               (513,) float32 -- deploy te
    submissions/nb3314_q_blend_clip_deep60.csv (only on BETTER verdict)
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

TAG = "nb3314"
PARENT_TAG = "nb3263_nb3070_quantile_blend_then_learned_clip_on_deep60"

# -- Anchor: deep-60 -----------------------------------------------------------
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
K18_DEEP60_OOF_PATH = DATA_PROCESSED / "nb3261_K18_60seed_oof.npy"
K18_DEEP60_TE_PATH = DATA_PROCESSED / "nb3261_K18_60seed_te.npy"
K19_TARGET = 19
RESID_FOLDS = 5
RESID_SEEDS_DEEP = list(range(0, 60))   # 60 seeds {0..59}, matches nb3261

# -- Feature cache paths (identical to nb3261 / nb2960 / nb3000) --------------
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

# -- ChEMBL kNN params (identical to nb3261/nb3000/nb2960) --------------------
KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

# -- CV protocol (q-blend + clip step) ----------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))  # 15 FRESH kf_seeds {1216..1230}

# -- q35 quantile-conditional blend weights (nb3070/nb3073 schedule) ----------
QUANTILE_CUT = 0.35   # q35 split (NEW: q35 vs nb3070's q50)
W_LOW_K18, W_LOW_K19 = 0.8, 0.2     # low / inactive regime
W_HIGH_K18, W_HIGH_K19 = 0.5, 0.5   # high / active regime

# -- Learned-clip grid (nb3263 family) ----------------------------------------
Q_LOW_GRID = [0.01, 0.05, 0.10]
Q_HIGH_GRID = [0.90, 0.95, 0.98]

# -- Output range clip (matches nb3070/nb3263 deploy stage) -------------------
TE_RANGE_LO = 3.0
TE_RANGE_HI = 9.0

# -- Gate ---------------------------------------------------------------------
GATE_BETTER = 0.4423

# -- References ---------------------------------------------------------------
REF_K18_DEEP30 = 0.4536
REF_K19_DEEP30 = 0.4607
REF_K18_DEEP60 = 0.4650          # nb3261 single-anchor bag (saturated)
REF_NB3261_CLIP = 0.4723         # nb3261 deep-60 + learned-clip pf-mean (FAIL)
REF_NB3070 = 0.4477
REF_NB3173_CLIP = 0.4422
REF_NB3223 = 0.4424
REF_NB2171 = 0.4682
CHEMPROP_AUX_REF = 0.6216


# ============================================================================
# helpers (lifted verbatim from nb3261 / nb2960 / nb3000)
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
    trajectory (verbatim from nb3261 / nb3050 / nb3000)."""
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
    """117-col matrix identical to nb3261 / nb3050 / nb2960 / nb3000."""
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
    """Build deep-60 bag-mean OOF (253,) + te (513,) for one K-pyramid.
    Verbatim recipe from nb3261.build_K_60seed_bag."""
    X_te_K = X_te_full[:, K_idx].astype(np.float32)
    X_unb_K = X_te_K[unb_idx]
    print(f"   [{K_label}] X_unb_K = {X_unb_K.shape}  X_te_K = {X_te_K.shape}")
    sum_unb = np.zeros(n_unb, dtype=np.float64)
    sum_te = np.zeros(n_test, dtype=np.float64)
    per_seed_rae = []
    per_seed_oof = []
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
        if (i % 10) == 0 or i == len(seeds) - 1:
            print(f"      [{K_label}] seed={s:4d}  "
                  f"rae={per_seed_rae[-1]:.4f}  "
                  f"wall={time.time()-ts:.2f}s  ({i+1}/{len(seeds)})")
    bag_oof_unb = sum_unb / len(seeds)
    bag_te_513 = sum_te / len(seeds)
    return bag_oof_unb, bag_te_513, per_seed_rae, per_seed_oof


# ============================================================================
# q35 quantile-conditional blend + learned clip
# ============================================================================

def _blend_quantile_conditional(p_k18, p_k19, q_thr):
    """Per-row hard-split blend by q_thr (fold-train-derived).

    rows with p_k18 <= q_thr -> 0.8*K18 + 0.2*K19   (low / inactive)
    rows with p_k18 >  q_thr -> 0.5*K18 + 0.5*K19   (high / active)
    """
    low_mask = p_k18 <= q_thr
    out = np.empty_like(p_k18, dtype=np.float64)
    out[low_mask] = W_LOW_K18 * p_k18[low_mask] + W_LOW_K19 * p_k19[low_mask]
    out[~low_mask] = W_HIGH_K18 * p_k18[~low_mask] + W_HIGH_K19 * p_k19[~low_mask]
    return out


def _pick_best_clip(y_tr, pred_tr):
    """Inner grid: pick (q_low*, q_high*) minimizing fold-train RAE."""
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


def _run_one_seed(p_k18_unb, p_k19_unb, y_unb, unb_scaffolds, kf_seed):
    """q35 quantile-conditional blend + learned clip at one outer kf_seed.

    q35 + clip params derived from fold-TRAIN ONLY (clean cross-fit).
    """
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_final = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_blend_train_raes = []
    fold_q35s = []
    fold_ql = []
    fold_qh = []
    fold_lo = []
    fold_hi = []
    fold_clipped_lo = []
    fold_clipped_hi = []
    fold_high_share = []

    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        # --- STEP 1: q35 from fold-TRAIN K18-deep60 preds ONLY --------------
        q35 = float(np.quantile(p_k18_unb[tr_loc], QUANTILE_CUT))
        fold_q35s.append(q35)

        # --- STEP 2: quantile-conditional blend -----------------------------
        tr_blend = _blend_quantile_conditional(
            p_k18_unb[tr_loc], p_k19_unb[tr_loc], q35
        )
        va_blend = _blend_quantile_conditional(
            p_k18_unb[va_loc], p_k19_unb[va_loc], q35
        )
        fold_blend_train_raes.append(float(rae(y_unb[tr_loc], tr_blend)))
        fold_high_share.append(float(np.mean(p_k18_unb[va_loc] > q35)))

        # --- STEP 3: learned clip on fold-train blended ---------------------
        ql, qh, lo, hi = _pick_best_clip(y_unb[tr_loc], tr_blend)
        fold_ql.append(ql)
        fold_qh.append(qh)
        fold_lo.append(lo)
        fold_hi.append(hi)

        n_lo = int(np.sum(va_blend < lo))
        n_hi = int(np.sum(va_blend > hi))
        fold_clipped_lo.append(n_lo)
        fold_clipped_hi.append(n_hi)

        # --- STEP 4: apply clip to fold-val blended -------------------------
        val_pred = np.clip(va_blend, lo, hi)
        oof_final[va_loc] = val_pred
        fold_val_raes.append(float(rae(y_unb[va_loc], val_pred)))

    if np.isnan(oof_final).any():
        raise RuntimeError(
            f"kf_seed={kf_seed}: scaffold splits did not cover all rows"
        )
    pooled = float(rae(y_unb, oof_final))
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(fold_val_raes)),
        "per_fold_val_rae_std": float(np.std(fold_val_raes, ddof=1)),
        "per_fold_blend_train_rae_mean": float(np.mean(fold_blend_train_raes)),
        "fold_q35_mean": float(np.mean(fold_q35s)),
        "fold_q35_std": float(np.std(fold_q35s, ddof=1)),
        "fold_high_share_mean": float(np.mean(fold_high_share)),
        "fold_ql": fold_ql,
        "fold_qh": fold_qh,
        "fold_lo_mean": float(np.mean(fold_lo)),
        "fold_hi_mean": float(np.mean(fold_hi)),
        "n_clipped_lo": int(np.sum(fold_clipped_lo)),
        "n_clipped_hi": int(np.sum(fold_clipped_hi)),
        "oof": oof_final,
    }


# ============================================================================
# main
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(
        f"{TAG} -- q{int(QUANTILE_CUT*100)} quantile-conditional blend "
        f"{{K18-deep60, K19-deep60}} + learned per-fold clip"
    )
    print(
        f"          deep-60 anchors: K18 (cached nb3261), K19 (built here, "
        f"seeds {RESID_SEEDS_DEEP[0]}..{RESID_SEEDS_DEEP[-1]})"
    )
    print(
        f"          q-cond: cut=q{QUANTILE_CUT:.2f}  "
        f"low=(K18={W_LOW_K18},K19={W_LOW_K19})  "
        f"high=(K18={W_HIGH_K18},K19={W_HIGH_K19})"
    )
    print(f"          clip grid: ql={Q_LOW_GRID}  qh={Q_HIGH_GRID}")
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

    # -- Load cached K18-deep60 ----------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 1: load cached K18-deep60 (nb3261) OOF + te")
    print("-" * 78)
    if not K18_DEEP60_OOF_PATH.exists() or not K18_DEEP60_TE_PATH.exists():
        raise FileNotFoundError(
            f"K18-deep60 cache missing: {K18_DEEP60_OOF_PATH} / "
            f"{K18_DEEP60_TE_PATH} (run nb3261 first)"
        )
    p_k18_unb = np.load(K18_DEEP60_OOF_PATH).astype(np.float64)
    p_k18_te = np.load(K18_DEEP60_TE_PATH).astype(np.float64)
    if p_k18_unb.shape != (n_unb,):
        raise ValueError(f"K18 OOF shape {p_k18_unb.shape} != ({n_unb},)")
    if p_k18_te.shape != (n_test,):
        raise ValueError(f"K18 te shape {p_k18_te.shape} != ({n_test},)")
    rae_k18 = float(rae(y_unb, p_k18_unb))
    leak_k18 = float(np.mean(np.isclose(p_k18_unb, y_unb, atol=1e-6)))
    print(f"   K18-deep60 OOF RAE = {rae_k18:.4f}  (ref nb3261 {REF_K18_DEEP60:.4f})")
    print(f"   K18-deep60 OOF mean={p_k18_unb.mean():.3f} std={p_k18_unb.std():.3f} "
          f"leak_eq={leak_k18:.2%}")
    print(f"   K18-deep60 te  mean={p_k18_te.mean():.3f} std={p_k18_te.std():.3f}")

    # -- Reconstruct K19 indices ---------------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 2: reconstruct K={K19_TARGET} idx from nb2231 RFE trajectory")
    print("-" * 78)
    if not NB2231_SUMMARY.exists():
        raise FileNotFoundError(f"missing {NB2231_SUMMARY}")
    with open(NB2231_SUMMARY) as f:
        nb2231 = json.load(f)
    K19_idx = np.array(reconstruct_K_from_trajectory(nb2231, K19_TARGET), dtype=int)
    if len(K19_idx) != K19_TARGET:
        raise ValueError(f"K={K19_TARGET} idx returned {len(K19_idx)} cols")
    print(f"   K={K19_TARGET} idx (n={len(K19_idx)}): {K19_idx.tolist()}")
    # sanity vs nb3000 cached K19 idx
    nb3000_path = DATA_PROCESSED / "nb3000_summary.json"
    if nb3000_path.exists():
        with open(nb3000_path) as f:
            nb3000_sum = json.load(f)
        cached_k19_idx = nb3000_sum.get("K19_idx_in_117col")
        if cached_k19_idx is not None:
            match = (list(K19_idx) == list(cached_k19_idx))
            print(f"   K19 idx matches nb3000 cache: {match}")
            if not match:
                print(f"   WARN: nb3000 K19 idx = {cached_k19_idx}")

    # -- Build 117-col matrix -------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 3: rebuild 117-col 5-way feature matrix")
    print("-" * 78)
    X_te_full, chembl_pool_size = build_117col_feature_matrix(te_smiles, n_test)
    print(f"   X_te_full = {X_te_full.shape}  chembl_pool={chembl_pool_size}")

    # -- Build K19-deep60 -----------------------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 4: K={K19_TARGET} residual-LGBM deep-60 "
          f"({len(RESID_SEEDS_DEEP)} seeds) [NEW: K19-deep60]")
    print("-" * 78)
    bag_oof_K19, bag_te_K19, per_seed_rae_K19, per_seed_oof_K19 = build_K_60seed_bag(
        "K19", K19_idx, X_te_full, unb_idx, anchor, residual,
        te_anchor_513, n_test, n_unb, RESID_SEEDS_DEEP,
    )
    p_k19_unb = bag_oof_K19.astype(np.float64)
    p_k19_te = bag_te_K19.astype(np.float64)
    bag_mean_rae_K19 = float(rae(y_unb, p_k19_unb))
    per_seed_arr_K19 = np.array(per_seed_rae_K19, dtype=np.float64)
    print(f"\n   [K19] per-seed RAE mean = {per_seed_arr_K19.mean():.4f}  "
          f"std = {per_seed_arr_K19.std(ddof=1):.4f}")
    print(f"   [K19] 60-seed BAG-MEAN OOF RAE = {bag_mean_rae_K19:.4f}")
    print(f"   [K19] delta vs nb3000 K=19 deep-30 "
          f"({REF_K19_DEEP30:.4f}) = {bag_mean_rae_K19 - REF_K19_DEEP30:+.4f}")
    K19_oof_path = DATA_PROCESSED / f"{TAG}_K19_60seed_oof.npy"
    K19_te_path = DATA_PROCESSED / f"{TAG}_K19_60seed_te.npy"
    np.save(K19_oof_path, p_k19_unb.astype(np.float32))
    np.save(K19_te_path, p_k19_te.astype(np.float32))
    print(f"   [save] {K19_oof_path}")
    print(f"   [save] {K19_te_path}")

    # Running bag RAE saturation check for K19
    bag_size_check = [5, 10, 15, 30, 45, 60]
    bag_size_check = [b for b in bag_size_check if b <= len(per_seed_oof_K19)]
    cum_sum_unb = np.zeros(n_unb, dtype=np.float64)
    running_bag_rae_K19 = {}
    for i, p in enumerate(per_seed_oof_K19, start=1):
        cum_sum_unb += p
        if i in bag_size_check:
            running_bag_rae_K19[str(i)] = round(
                float(rae(y_unb, cum_sum_unb / i)), 4
            )
    print(f"   [K19] running bag-RAE: {running_bag_rae_K19}")

    # Cross-check vs cached nb3000 K19 deep-30
    cached_k19_30_path = DATA_PROCESSED / "nb3000_K19_30seed_oof.npy"
    cached_check_K19 = None
    if cached_k19_30_path.exists():
        cached_oof = np.load(cached_k19_30_path).astype(np.float64)
        cached_rae = float(rae(y_unb, cached_oof))
        pearson_corr = float(np.corrcoef(p_k19_unb, cached_oof)[0, 1])
        cached_check_K19 = {
            "cached_nb3000_K19_30seed_rae": round(cached_rae, 4),
            "pearson_vs_60seed_bag": round(pearson_corr, 4),
        }
        print(f"   sanity: nb3000 K19 30-seed cache RAE = {cached_rae:.4f}  "
              f"pearson(30 vs 60) = {pearson_corr:.4f}")

    # Anchor pair correlation
    pair_corr = float(np.corrcoef(p_k18_unb, p_k19_unb)[0, 1])
    print(f"   K18-deep60 vs K19-deep60 OOF pearson = {pair_corr:.4f}")

    # -- Scaffolds ------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 5: scaffolds for outer CV")
    print("-" * 78)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"   n_unique_scaffolds = {n_unique_scaf}")

    # -- q35 blend + learned-clip 15-seed sweep -------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 6: WIDE-SEED q{int(QUANTILE_CUT*100)}-BLEND + LEARNED-CLIP SWEEP "
          f"-- {len(KF_SEEDS)} fresh kf_seeds {{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}")
    print("-" * 78)
    seed_records = []
    pooled_raes = []
    per_fold_means = []
    per_fold_stds = []
    oof_stack = []
    all_fold_ql = []
    all_fold_qh = []
    all_fold_q35 = []
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(p_k18_unb, p_k19_unb, y_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        per_fold_means.append(res["per_fold_val_rae_mean"])
        per_fold_stds.append(res["per_fold_val_rae_std"])
        oof_stack.append(res["oof"])
        all_fold_ql.extend(res["fold_ql"])
        all_fold_qh.extend(res["fold_qh"])
        all_fold_q35.append(res["fold_q35_mean"])
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
            "per_fold_val_rae_std": round(res["per_fold_val_rae_std"], 4),
            "per_fold_blend_train_rae_mean": round(
                res["per_fold_blend_train_rae_mean"], 4),
            "fold_q35_mean": round(res["fold_q35_mean"], 4),
            "fold_q35_std": round(res["fold_q35_std"], 4),
            "fold_high_share_mean": round(res["fold_high_share_mean"], 4),
            "fold_ql": [round(v, 3) for v in res["fold_ql"]],
            "fold_qh": [round(v, 3) for v in res["fold_qh"]],
            "fold_lo_mean": round(res["fold_lo_mean"], 4),
            "fold_hi_mean": round(res["fold_hi_mean"], 4),
            "n_clipped_lo": res["n_clipped_lo"],
            "n_clipped_hi": res["n_clipped_hi"],
        })
        print(
            f"   kf={s}: pooled={res['pooled_rae']:.4f}  "
            f"pf_mean={res['per_fold_val_rae_mean']:.4f}  "
            f"q35={res['fold_q35_mean']:.3f}  "
            f"hi_share={res['fold_high_share_mean']:.2f}  "
            f"clip(lo,hi)=({res['n_clipped_lo']},{res['n_clipped_hi']})  "
            f"wall={time.time()-ts:.2f}s"
        )

    arr_pool = np.asarray(pooled_raes, dtype=np.float64)
    arr_pfm = np.asarray(per_fold_means, dtype=np.float64)
    n_s = len(arr_pfm)
    pooled_mean = float(arr_pool.mean())
    pooled_std = float(arr_pool.std(ddof=1)) if n_s > 1 else 0.0
    per_fold_mean = float(arr_pfm.mean())   # GATE METRIC
    per_fold_std = float(arr_pfm.std(ddof=1)) if n_s > 1 else 0.0
    sem_pfm = per_fold_std / np.sqrt(n_s) if n_s > 1 else 0.0
    t_mult = 2.145  # df=14
    ci_low_pfm = per_fold_mean - t_mult * sem_pfm
    ci_high_pfm = per_fold_mean + t_mult * sem_pfm
    pf_median = float(np.median(arr_pfm))

    ql_counter = Counter(all_fold_ql)
    qh_counter = Counter(all_fold_qh)
    ql_mode = ql_counter.most_common(1)[0][0]
    qh_mode = qh_counter.most_common(1)[0][0]
    q35_grand_mean = float(np.mean(all_fold_q35))

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds)")
    print("-" * 78)
    print(f"   pooled_RAE       mean = {pooled_mean:.4f}  std = {pooled_std:.4f}")
    print(f"   pooled  min/max  [{arr_pool.min():.4f}, {arr_pool.max():.4f}]")
    print(f"   per_fold_mean    mean = {per_fold_mean:.4f}  std = {per_fold_std:.4f}  "
          f"95% CI [{ci_low_pfm:.4f}, {ci_high_pfm:.4f}]   <- GATE METRIC")
    print(f"   per-fm  median   = {pf_median:.4f}")
    print(f"   per-fm  min/max  [{arr_pfm.min():.4f}, {arr_pfm.max():.4f}]")
    print(f"\n   ql distribution (75 folds) = {dict(ql_counter)}")
    print(f"   qh distribution (75 folds) = {dict(qh_counter)}")
    print(f"   ql_mode = {ql_mode}  qh_mode = {qh_mode}")
    print(f"   grand-mean q35 across seeds = {q35_grand_mean:.4f}")
    print(f"\n   ref nb3173 best clip-winner = {REF_NB3173_CLIP:.4f}")
    print(f"   ref nb3070 q-cond {{K18,K19}} = {REF_NB3070:.4f}")
    print(f"   ref nb3261 deep60+clip (FAIL) = {REF_NB3261_CLIP:.4f}")
    print(f"   delta vs nb3173             = {per_fold_mean - REF_NB3173_CLIP:+.4f}")
    print(f"   delta vs nb3261 deep60+clip = {per_fold_mean - REF_NB3261_CLIP:+.4f}")

    # -- Deploy: q35 from FULL 253 K18-deep60, blend te, learned clip ---------
    print("\n" + "-" * 78)
    print("STEP 7: DEPLOY -- q35 from full-253 K18-deep60 -> blend te(513) "
          "-> learned clip on full-253 blended")
    print("-" * 78)
    deploy_q35 = float(np.quantile(p_k18_unb, QUANTILE_CUT))
    full_blend = _blend_quantile_conditional(p_k18_unb, p_k19_unb, deploy_q35)
    te_blend = _blend_quantile_conditional(p_k18_te, p_k19_te, deploy_q35)
    full_blend_rae = float(rae(y_unb, full_blend))
    print(f"   deploy q35 (full-253 K18-deep60) = {deploy_q35:.4f}")
    print(f"   full-253 blended in-sample RAE   = {full_blend_rae:.4f}")

    deploy_ql, deploy_qh, deploy_lo, deploy_hi = _pick_best_clip(y_unb, full_blend)
    print(f"   deploy clip = (q{deploy_ql:.2f}, q{deploy_qh:.2f}) -> "
          f"({deploy_lo:.3f}, {deploy_hi:.3f}) from full-253 y")

    te_pred_pre_range = np.clip(te_blend, deploy_lo, deploy_hi)
    te_pred = np.clip(te_pred_pre_range, TE_RANGE_LO, TE_RANGE_HI).astype(np.float32)
    n_te_lo = int(np.sum(te_blend < deploy_lo))
    n_te_hi = int(np.sum(te_blend > deploy_hi))
    te_low_share = float(np.mean(p_k18_te <= deploy_q35))
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"   te low-half share (te_K18 <= q35) = {te_low_share:.3f}")
    print(f"   te clipped: lo={n_te_lo}/513  hi={n_te_hi}/513  "
          f"total={n_te_lo + n_te_hi}/513")
    print(f"   te(513) mean={te_pred.mean():.3f}  std={te_pred.std():.3f}  "
          f"min={te_pred.min():.3f}  max={te_pred.max():.3f}")
    print(f"   te[unb] in-sample RAE = {te_unb_in_rae:.4f}")

    # Median-seed OOF for storage
    med_seed_idx = int(np.argsort(arr_pfm)[n_s // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(f"   median (by pf_mean) seed = {median_seed} "
          f"(pf_mean={arr_pfm[med_seed_idx]:.4f}, "
          f"pooled={arr_pool[med_seed_idx]:.4f})")

    # -- Gate -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 8: GATE")
    print("-" * 78)
    if per_fold_mean < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE-CANDIDATE. nb3314 q{int(QUANTILE_CUT*100)} quantile-"
            f"conditional blend on deep-60 anchors {{K18,K19}} + learned clip "
            f"15-seed per-fold-mean {per_fold_mean:.4f} clears the BETTER gate "
            f"{GATE_BETTER:.4f} ({per_fold_mean - GATE_BETTER:+.4f}), beating "
            f"nb3173 clip-winner ({REF_NB3173_CLIP:.4f}) by "
            f"{per_fold_mean - REF_NB3173_CLIP:+.4f}. DEEPER BAG COMPOUNDS "
            f"THROUGH THE COMPOSITE: even though single-anchor deep-60 saturated "
            f"(nb3261 0.4650 vs deep-30 0.4536) and deep-60+clip alone FAILED "
            f"(nb3261 0.4723), the q35-blend of two deep-60 anchors + clip "
            f"extracts net gain. Re-verify with deep-30 anchors at matched q35 "
            f"to isolate the deep-60 contribution before any PRIMARY-1 swap; "
            f"cycle-160 deep-30 rule mandatory. Predicted LB under +0.0045 PRE "
            f"delta = {per_fold_mean + 0.0045:.4f}. Modal clip = "
            f"(q{ql_mode:.2f}, q{qh_mode:.2f})."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3314 q{int(QUANTILE_CUT*100)}-blend on deep-60 anchors "
            f"{{K18,K19}} + learned clip 15-seed per-fold-mean {per_fold_mean:.4f} "
            f"fails BETTER gate {GATE_BETTER:.4f} "
            f"({per_fold_mean - GATE_BETTER:+.4f}). DEEPER BAG DOES NOT COMPOUND "
            f"through the composite: matching the single-anchor saturation "
            f"finding (nb3261 deep-60 bag 0.4650 vs deep-30 0.4536, and deep-60"
            f"+clip 0.4723), a deeper bag at every stage does not break the "
            f"post-hoc-blend ceiling. The q35-blend + clip ceiling is set by the "
            f"(anchor=chemprop_aux, K=18/19 feature, n=253) substrate, not bag "
            f"depth. Keep nb3173 ({REF_NB3173_CLIP:.4f}) / nb3223 "
            f"({REF_NB3223:.4f}) on ladder; no ladder change. Compare blended "
            f"pre-clip in-sample RAE = {full_blend_rae:.4f}. Modal clip = "
            f"(q{ql_mode:.2f}, q{qh_mode:.2f})."
        )
    print(f"   per_fold_mean   = {per_fold_mean:.4f}  (gate: < {GATE_BETTER:.4f})")
    print(f"   pooled_mean     = {pooled_mean:.4f}  (informational)")
    print(f"   verdict         = {verdict}")
    print(f"   ladder action   = {ladder_action}")

    # -- Save artifacts -------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 9: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_for_save)
    np.save(te_path, te_pred)
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_q_blend_clip_deep60.csv"
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
        "method": ("q35_quantile_conditional_blend_K18deep60_K19deep60_"
                   "then_learned_per_fold_clip"),
        "paradigm": "does_deeper_bag_at_every_stage_compound_through_composite",
        "anchor_pre_unblind": True,
        "anchor_te_path": str(ANCHOR_TE_PATH),
        "chemprop_aux_anchor_rae": round(rae_anchor, 4),
        # deep-60 anchors
        "resid_seeds_deep": RESID_SEEDS_DEEP,
        "n_seeds_deep": len(RESID_SEEDS_DEEP),
        "K18_deep60_source": "nb3261_cache",
        "K18_deep60_oof_path": str(K18_DEEP60_OOF_PATH),
        "K18_deep60_te_path": str(K18_DEEP60_TE_PATH),
        "K18_deep60_oof_rae": round(rae_k18, 4),
        "K18_deep60_leak_eq_truth_frac": round(leak_k18, 4),
        "K19_target": K19_TARGET,
        "K19_idx_in_117": K19_idx.tolist(),
        "K19_per_seed_rae_mean": round(float(per_seed_arr_K19.mean()), 4),
        "K19_per_seed_rae_std": round(float(per_seed_arr_K19.std(ddof=1)), 4),
        "K19_per_seed_rae_min": round(float(per_seed_arr_K19.min()), 4),
        "K19_per_seed_rae_max": round(float(per_seed_arr_K19.max()), 4),
        "K19_60seed_bag_mean_oof_rae": round(bag_mean_rae_K19, 4),
        "K19_60seed_oof_path": str(K19_oof_path),
        "K19_60seed_te_path": str(K19_te_path),
        "K19_60seed_te_mean": float(p_k19_te.mean()),
        "K19_60seed_te_std": float(p_k19_te.std()),
        "K19_running_bag_rae": running_bag_rae_K19,
        "K19_delta_60seed_vs_nb3000_30seed": round(
            bag_mean_rae_K19 - REF_K19_DEEP30, 4),
        "K19_cached_30seed_sanity": cached_check_K19,
        "anchor_pair_oof_pearson_K18_K19": round(pair_corr, 4),
        # q35 blend
        "quantile_cut": QUANTILE_CUT,
        "q_source": "fold_train_only",
        "w_low": {"K18": W_LOW_K18, "K19": W_LOW_K19},
        "w_high": {"K18": W_HIGH_K18, "K19": W_HIGH_K19},
        # learned clip
        "q_low_grid": Q_LOW_GRID,
        "q_high_grid": Q_HIGH_GRID,
        "te_range_lo": TE_RANGE_LO,
        "te_range_hi": TE_RANGE_HI,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": n_s,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "seed_records": seed_records,
        "pooled_rae_array": [round(float(v), 4) for v in pooled_raes],
        "per_fold_mean_array": [round(float(v), 4) for v in per_fold_means],
        "per_fold_std_array": [round(float(v), 4) for v in per_fold_stds],
        # Gate metric: per-fold-mean
        "per_fold_mean_rae": round(per_fold_mean, 4),
        "per_fold_std_rae": round(per_fold_std, 4),
        "per_fold_sem_rae": round(sem_pfm, 4),
        "per_fold_ci95_low": round(ci_low_pfm, 4),
        "per_fold_ci95_high": round(ci_high_pfm, 4),
        "per_fold_median_rae": round(pf_median, 4),
        "per_fold_min_rae": round(float(arr_pfm.min()), 4),
        "per_fold_max_rae": round(float(arr_pfm.max()), 4),
        # mean_rae mirror for ladder-script compatibility
        "mean_rae": round(per_fold_mean, 4),
        "std_rae": round(per_fold_std, 4),
        # Pooled (reference)
        "pooled_mean_rae": round(pooled_mean, 4),
        "pooled_std_rae": round(pooled_std, 4),
        "pooled_min_rae": round(float(arr_pool.min()), 4),
        "pooled_max_rae": round(float(arr_pool.max()), 4),
        # clip stats
        "ql_distribution": {str(k): int(v) for k, v in ql_counter.items()},
        "qh_distribution": {str(k): int(v) for k, v in qh_counter.items()},
        "ql_mode": float(ql_mode),
        "qh_mode": float(qh_mode),
        "q35_grand_mean_across_seeds": round(q35_grand_mean, 4),
        # references
        "ref_K18_deep30": REF_K18_DEEP30,
        "ref_K19_deep30": REF_K19_DEEP30,
        "ref_K18_deep60": REF_K18_DEEP60,
        "ref_nb3261_deep60_clip": REF_NB3261_CLIP,
        "ref_nb3070": REF_NB3070,
        "ref_nb3173_clip": REF_NB3173_CLIP,
        "ref_nb3223": REF_NB3223,
        "ref_nb2171": REF_NB2171,
        "delta_vs_nb3173": round(per_fold_mean - REF_NB3173_CLIP, 4),
        "delta_vs_nb3070": round(per_fold_mean - REF_NB3070, 4),
        "delta_vs_nb3223": round(per_fold_mean - REF_NB3223, 4),
        "delta_vs_nb3261_deep60_clip": round(per_fold_mean - REF_NB3261_CLIP, 4),
        # deploy
        "deploy_q35": round(deploy_q35, 4),
        "deploy_blend_in_sample_rae": round(full_blend_rae, 4),
        "deploy_ql": float(deploy_ql),
        "deploy_qh": float(deploy_qh),
        "deploy_lo": round(deploy_lo, 4),
        "deploy_hi": round(deploy_hi, 4),
        "n_te_clipped_lo": n_te_lo,
        "n_te_clipped_hi": n_te_hi,
        "te_low_share": round(te_low_share, 4),
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
    print(f"   K19-deep60 bag-mean OOF RAE       = {bag_mean_rae_K19:.4f}  "
          f"(vs nb3000 K19 deep-30 {REF_K19_DEEP30:.4f}, "
          f"delta {bag_mean_rae_K19 - REF_K19_DEEP30:+.4f})")
    print(f"   K19 running bag-RAE               = {running_bag_rae_K19}")
    print(f"   K18-deep60 OOF RAE                = {rae_k18:.4f}")
    print(f"   anchor pair pearson (K18,K19)     = {pair_corr:.4f}")
    print(f"   q35 grand-mean                    = {q35_grand_mean:.4f}")
    print(f"   q35-blend + clip per-fold-mean    = {per_fold_mean:.4f} "
          f"+/- {per_fold_std:.4f}  <- GATE METRIC")
    print(f"   pooled mean                       = {pooled_mean:.4f} "
          f"+/- {pooled_std:.4f}")
    print(f"   delta vs nb3173 clip (0.4422)     = "
          f"{per_fold_mean - REF_NB3173_CLIP:+.4f}")
    print(f"   modal clip (ql, qh)               = ({ql_mode}, {qh_mode})")
    print(f"   te[unb] in-sample                 = {te_unb_in_rae:.4f}")
    print(f"   verdict                           = {verdict}")
    print(f"   wall                              = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "K19_60seed_bag_mean_oof_rae",
        "K19_per_seed_rae_mean",
        "K19_per_seed_rae_std",
        "K19_delta_60seed_vs_nb3000_30seed",
        "K19_running_bag_rae",
        "K18_deep60_oof_rae",
        "anchor_pair_oof_pearson_K18_K19",
        "q35_grand_mean_across_seeds",
        "per_fold_mean_rae",
        "per_fold_std_rae",
        "per_fold_ci95_low",
        "per_fold_ci95_high",
        "pooled_mean_rae",
        "pooled_std_rae",
        "delta_vs_nb3173",
        "delta_vs_nb3070",
        "delta_vs_nb3261_deep60_clip",
        "ql_mode",
        "qh_mode",
        "deploy_q35",
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
