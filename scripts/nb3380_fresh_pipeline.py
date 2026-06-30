"""nb3380 -- Full nb3200 pipeline rebuilt on the FRESH chemprop anchor, then
blended with the real nb3200.

CONTEXT (cycle 250+):
    The fresh chemprop anchor (nb3350) is input-decorrelated vs the frozen
    chemprop_aux (Pearson 0.944 on the 253) but RAW-anchor blending failed --
    the orthogonal component behaved as noise (nb337x). NEW TEST: run the ENTIRE
    end-to-end pipeline (chemprop -> K18/K19 residual -> q-blend -> learned clip)
    on the fresh anchor to obtain fresh_nb3200, then check whether the FINAL
    pipeline OUTPUTS decorrelate enough to help when blended with the real nb3200.

    The real pipeline is:
        chemprop_aux (frozen)
          -> nb2960 K18 deep-30 residual + nb3000 K19 deep-30 residual
          -> nb3090 q35 quantile-conditional blend (w_low=0.95, w_high=0.40)
          -> nb3200 per-fold learned clip  (15-seed mean 0.4424)
    Here every stage is re-run with the FRESH anchor swapped in for the frozen one.

PROTOCOL:
    STEP 1  fresh_K18_v3, fresh_K19_v3: LGBM K=18 / K=19 residuals on the FRESH
            anchor (residual = y - fresh_chemprop_v3), deep-30 bag, identical
            residual-LGBM recipe to nb2960 / nb3000 (only the anchor changes).
    STEP 2  fresh_nb3090: q35 quantile-conditional blend {fresh_K18, fresh_K19}
            (q_cut=0.35, w_low=0.95, w_high=0.40), per-fold thresholded.
    STEP 3  fresh_nb3200: per-fold learned clip on fresh_nb3090 (inner grid
            q_low in {0.01,0.05,0.10}, q_high in {0.90,0.95,0.98,0.99},
            pick min fold-train RAE; apply to fold-val), identical to nb3190.
    STEP 4  report fresh_nb3200 RAE on 253 (per-fold-mean over 15 seeds) and
            correlation of fresh_nb3200 vs the REAL nb3200_pred_oof.
    STEP 5  blend: per-fold SLSQP {nb3200, fresh_nb3200} + final clip; also
            fixed w in {0.1, 0.2, 0.3}.

    15 fresh kf_seeds {1216..1230}; per-fold-mean RAE everywhere (mean of the 5
    fold-val RAEs, averaged over the 15 seeds) -- distinct from nb3200's pooled
    metric, matching this task's "per-fold-mean" prescription.

GATES:
    fresh_nb3200 per-fold-mean RAE < 0.50        -> "FRESH_PIPELINE_USABLE"
    corr(fresh_nb3200, nb3200) < 0.95            -> "OUTPUT_DECORRELATED"
    best blend per-fold-mean < 0.4423            -> "BREAKTHROUGH"

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb3350_chemprop_v3_oof.npy   (253,) fresh anchor on unblind
    data/processed/te_nb3350_chemprop_v3.npy    (513,) fresh anchor on test
    data/processed/nb3200_pred_oof.npy          (253,) real nb3200 OOF
    data/processed/te_nb3200.npy                (513,) real nb3200 deploy te
    data/processed/nb2604_summary.json          (K18 idx)
    data/processed/nb2231_summary.json          (K19 idx reconstruction)
    data/processed/nb2063_shap_importance_full117.npy
    data/processed/nb1352/1392/1484/1523/1524/1541_summary.json  (117-col feats)
    data/processed/te_atompair / te_maccs / te_chemprop_embed_300 / te_avalon512.npy
    C:/pxr_artifacts/nb1030/X_mordred_test.npy
    data/external/chembl_*.parquet  (ChEMBL kNN feature)

Outputs:
    data/processed/nb3380_summary.json
    data/processed/nb3380_fresh_nb3200_oof.npy      (253,) float32
    data/processed/te_nb3380_fresh_nb3200.npy       (513,) float32
    data/processed/nb3380_pred_oof.npy              (253,) float32  (best blend OOF)
    data/processed/te_nb3380.npy                    (513,) float32  (best blend deploy)
    data/processed/nb3380_fresh_K18_v3_oof.npy / _te.npy
    data/processed/nb3380_fresh_K19_v3_oof.npy / _te.npy
    data/processed/nb3380_fresh_nb3090_oof.npy / _te.npy
    submissions/nb3380_fresh_pipeline.csv  (only on BREAKTHROUGH)
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
from rdkit import Chem
from rdkit import RDLogger
from sklearn.model_selection import KFold
from scipy.optimize import minimize

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3380"

# -- FRESH anchor (nb3350) ----------------------------------------------------
FRESH_ANCHOR_OOF = DATA_PROCESSED / "nb3350_chemprop_v3_oof.npy"   # (253,)
FRESH_ANCHOR_TE = DATA_PROCESSED / "te_nb3350_chemprop_v3.npy"     # (513,)

# -- Real nb3200 (blend partner) ----------------------------------------------
NB3200_OOF = DATA_PROCESSED / "nb3200_pred_oof.npy"
NB3200_TE = DATA_PROCESSED / "te_nb3200.npy"

# -- Residual-LGBM params (IDENTICAL recipe to nb2960 / nb3000) ---------------
RESID_FOLDS = 5
RESID_SEEDS_DEEP = list(range(3001, 3031))    # 30 fresh seeds {3001..3030}

# -- Feature cache paths (identical to nb2960 / nb3000) -----------------------
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

NB2604_SUMMARY = DATA_PROCESSED / "nb2604_summary.json"      # K18 idx
NB2231_SUMMARY = DATA_PROCESSED / "nb2231_summary.json"      # K19 reconstruction
NB2063_SHAP_PATH = DATA_PROCESSED / "nb2063_shap_importance_full117.npy"

# -- ChEMBL kNN params (identical to nb2960 / nb3000) -------------------------
KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

# -- Quantile-conditional blend (nb3090 best combo) ---------------------------
Q_CUT = 0.35
W_LOW = 0.95
W_HIGH = 0.40

# -- Learned-clip grid (nb3190 / nb3200) --------------------------------------
Q_LOW_GRID = [0.01, 0.05, 0.10]
Q_HIGH_GRID = [0.90, 0.95, 0.98, 0.99]

# -- CV protocol --------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))   # 15 fresh seeds {1216..1230}

# -- Blend fixed-w grid -------------------------------------------------------
FIXED_W = [0.1, 0.2, 0.3]            # weight on fresh_nb3200

# -- Gates --------------------------------------------------------------------
GATE_USABLE = 0.50
GATE_DECORR = 0.95
GATE_BREAKTHROUGH = 0.4423

# -- References ---------------------------------------------------------------
REF_NB3200 = 0.4424
REF_FRESH_ANCHOR_RAE = 0.6292
REF_FROZEN_ANCHOR_RAE = 0.6216
REF_PEARSON_ANCHORS = 0.944


# ============================================================================
# helpers (lifted verbatim from nb2960 / nb3000)
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
    trajectory (verbatim from nb3000)."""
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
    """117-col matrix identical to nb2604 / nb2960 / nb3000."""
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


def build_K_30seed_bag(K_label, K_idx, X_te_full, unb_idx, anchor,
                       residual, te_anchor_513, n_test, n_unb, seeds):
    """Build deep-30 bag-mean OOF (253,) + te (513,) for one K-pyramid on the
    given anchor. residual = y_unb - anchor; te base = te_anchor_513.
    Identical to nb2960 / nb3000, anchor passed in."""
    X_te_K = X_te_full[:, K_idx].astype(np.float32)
    X_unb_K = X_te_K[unb_idx]
    print(f"   [{K_label}] X_unb_K = {X_unb_K.shape}  X_te_K = {X_te_K.shape}")
    sum_unb = np.zeros(n_unb, dtype=np.float64)
    sum_te = np.zeros(n_test, dtype=np.float64)
    per_seed_rae = []
    y_unb_implied = anchor + residual
    for i, s in enumerate(seeds):
        ts = time.time()
        resid_oof = _residual_cross_fit_one_seed(X_unb_K, residual, s)
        pred_unb_s = anchor + resid_oof
        sum_unb += pred_unb_s
        per_seed_rae.append(float(rae(y_unb_implied, pred_unb_s)))
        te_resid_s = _train_full_then_predict_te(X_unb_K, residual, X_te_K, s)
        pred_te_s = te_anchor_513 + te_resid_s
        sum_te += pred_te_s
        if (i % 10) == 0 or i == len(seeds) - 1:
            print(f"      [{K_label}] seed={s:4d}  rae={per_seed_rae[-1]:.4f}  "
                  f"wall={time.time()-ts:.2f}s  ({i+1}/{len(seeds)})")
    bag_oof_unb = sum_unb / len(seeds)
    bag_te_513 = sum_te / len(seeds)
    return bag_oof_unb, bag_te_513, per_seed_rae


# ============================================================================
# pipeline operators (q-blend + learned clip)
# ============================================================================

def blend_quantile_conditional(p_k18, p_k19, q_thr, w_low, w_high):
    """nb3090 per-row hard-split blend on K18 threshold."""
    low_mask = p_k18 <= q_thr
    out = np.empty_like(p_k18, dtype=np.float64)
    out[low_mask] = w_low * p_k18[low_mask] + (1.0 - w_low) * p_k19[low_mask]
    out[~low_mask] = w_high * p_k18[~low_mask] + (1.0 - w_high) * p_k19[~low_mask]
    return out


def pick_best_clip(y_tr, pred_tr):
    """nb3190 inner grid: pick (q_low*, q_high*) minimizing fold-train RAE."""
    best_rae = np.inf
    best_ql, best_qh = Q_LOW_GRID[0], Q_HIGH_GRID[-1]
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
                best_rae, best_ql, best_qh, best_lo, best_hi = r, ql, qh, lo, hi
    return best_ql, best_qh, best_lo, best_hi


def fresh_nb3200_one_seed(p_k18, p_k19, y_unb, unb_scaffolds, kf_seed):
    """Run fresh_nb3090 (q-blend) -> fresh_nb3200 (learned clip) at one kf_seed.

    Returns oof (253,) and the list of per-fold-val RAEs (post-clip).
    The q-threshold is computed PER FOLD on fold-train K18 (nb3090 protocol);
    the clip (lo,hi) is learned PER FOLD on fold-train truth (nb3190 protocol).
    """
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    for tr_loc, va_loc in splits:
        # --- nb3090 q-blend on fold-train threshold ---
        q_thr = float(np.quantile(p_k18[tr_loc], Q_CUT))
        # build the blended base for BOTH train (for clip-fit) and val
        base_tr = blend_quantile_conditional(
            p_k18[tr_loc], p_k19[tr_loc], q_thr, W_LOW, W_HIGH,
        )
        base_va = blend_quantile_conditional(
            p_k18[va_loc], p_k19[va_loc], q_thr, W_LOW, W_HIGH,
        )
        # --- nb3190 learned clip on fold-train truth ---
        ql, qh, lo, hi = pick_best_clip(y_unb[tr_loc], base_tr)
        val_clipped = np.clip(base_va, lo, hi)
        oof[va_loc] = val_clipped
        fold_val_raes.append(float(rae(y_unb[va_loc], val_clipped)))
    if np.isnan(oof).any():
        raise RuntimeError(f"kf_seed={kf_seed}: scaffold splits did not cover all rows")
    return oof, fold_val_raes


def blend_slsqp_one_seed(p_main, p_fresh, y_unb, unb_scaffolds, kf_seed,
                         do_clip=True):
    """Per-fold SLSQP {p_main, p_fresh} simplex blend + optional learned clip.

    Per outer fold: fit weights w=(w_main, w_fresh) on fold-train (simplex,
    minimize fold-train RAE), apply to fold-val. If do_clip, learn a clip on the
    blended fold-train and apply to the blended fold-val (final clip step).
    Returns (oof_blend, list_per_fold_val_rae, list_per_fold_w_fresh).
    """
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_w_fresh = []
    P_all = np.column_stack([p_main, p_fresh])  # (253, 2)
    for tr_loc, va_loc in splits:
        Ptr = P_all[tr_loc]
        ytr = y_unb[tr_loc]

        def obj(w):
            return rae(ytr, Ptr @ w)

        cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
        bnds = [(0.0, 1.0), (0.0, 1.0)]
        w0 = np.array([0.8, 0.2])
        res = minimize(obj, w0, method="SLSQP", bounds=bnds, constraints=cons,
                       options={"maxiter": 200, "ftol": 1e-9})
        w = res.x if res.success else w0
        w = np.clip(w, 0.0, 1.0)
        w = w / w.sum() if w.sum() > 0 else np.array([1.0, 0.0])
        fold_w_fresh.append(float(w[1]))

        blend_tr = Ptr @ w
        blend_va = P_all[va_loc] @ w
        if do_clip:
            ql, qh, lo, hi = pick_best_clip(ytr, blend_tr)
            blend_va = np.clip(blend_va, lo, hi)
        oof[va_loc] = blend_va
        fold_val_raes.append(float(rae(y_unb[va_loc], blend_va)))
    if np.isnan(oof).any():
        raise RuntimeError(f"SLSQP kf_seed={kf_seed}: splits did not cover all rows")
    return oof, fold_val_raes, fold_w_fresh


def blend_fixed_w_one_seed(p_main, p_fresh, y_unb, unb_scaffolds, kf_seed,
                           w_fresh, do_clip=True):
    """Fixed-weight blend pred = (1-w)*p_main + w*p_fresh + per-fold learned clip."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    blend_full = (1.0 - w_fresh) * p_main + w_fresh * p_fresh
    for tr_loc, va_loc in splits:
        blend_tr = blend_full[tr_loc]
        blend_va = blend_full[va_loc]
        if do_clip:
            ql, qh, lo, hi = pick_best_clip(y_unb[tr_loc], blend_tr)
            blend_va = np.clip(blend_va, lo, hi)
        oof[va_loc] = blend_va
        fold_val_raes.append(float(rae(y_unb[va_loc], blend_va)))
    if np.isnan(oof).any():
        raise RuntimeError(f"fixed-w kf_seed={kf_seed}: splits did not cover all rows")
    return oof, fold_val_raes


def per_fold_mean_over_seeds(run_fn):
    """Run a per-seed function returning (oof, fold_val_raes[, extra]) over all
    KF_SEEDS; return dict with per-fold-mean RAE (mean over folds, then seeds)
    and the median-seed OOF for deploy storage."""
    seed_pfm = []           # per-seed per-fold-mean RAE
    seed_pooled = []        # per-seed pooled RAE (for cross-check)
    oof_stack = []
    extra_stack = []
    for s in KF_SEEDS:
        out = run_fn(s)
        oof = out[0]
        fold_raes = out[1]
        seed_pfm.append(float(np.mean(fold_raes)))
        seed_pooled.append(float(rae(_per_fold_mean_over_seeds_y, oof)))
        oof_stack.append(oof)
        if len(out) > 2:
            extra_stack.append(out[2])
    arr = np.asarray(seed_pfm, dtype=np.float64)
    pooled_arr = np.asarray(seed_pooled, dtype=np.float64)
    # median seed by per-fold-mean
    med_idx = int(np.argsort(arr)[len(arr) // 2])
    return {
        "per_fold_mean_rae": float(arr.mean()),
        "per_fold_mean_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "per_fold_mean_min": float(arr.min()),
        "per_fold_mean_max": float(arr.max()),
        "per_seed_pfm": [round(float(v), 4) for v in arr],
        "pooled_mean_rae": float(pooled_arr.mean()),
        "median_seed": int(KF_SEEDS[med_idx]),
        "median_oof": oof_stack[med_idx],
        "extra_stack": extra_stack,
    }


# module-level handle so per_fold_mean_over_seeds can compute pooled RAE
_per_fold_mean_over_seeds_y = None


# ============================================================================
# main
# ============================================================================

def main():
    global _per_fold_mean_over_seeds_y
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- FULL nb3200 pipeline on the FRESH chemprop anchor, then "
          f"blend with real nb3200")
    print(f"          fresh anchor = nb3350 (RAE {REF_FRESH_ANCHOR_RAE:.4f}, "
          f"corr to frozen {REF_PEARSON_ANCHORS:.3f})")
    print(f"          q-blend: q_cut={Q_CUT} w_low={W_LOW} w_high={W_HIGH}")
    print(f"          clip grid: q_low {Q_LOW_GRID} q_high {Q_HIGH_GRID}")
    print(f"          kf_seeds = {KF_SEEDS[0]}..{KF_SEEDS[-1]} (n={len(KF_SEEDS)})")
    print(f"          gates: fresh_nb3200<{GATE_USABLE} USABLE | "
          f"corr<{GATE_DECORR} DECORR | blend<{GATE_BREAKTHROUGH} BREAKTHROUGH")
    print("=" * 78)

    # -- Load truth, test, fresh anchor, nb3200 ------------------------------
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
    _per_fold_mean_over_seeds_y = y_unb
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    # fresh anchor
    fresh_te_513 = np.load(FRESH_ANCHOR_TE).astype(np.float64)
    fresh_oof = np.load(FRESH_ANCHOR_OOF).astype(np.float64)
    assert fresh_te_513.shape == (n_test,)
    assert fresh_oof.shape == (n_unb,)
    # fresh_te[unb] must equal fresh_oof (both load_test() order)
    if not np.allclose(fresh_te_513[unb_idx], fresh_oof, atol=1e-5):
        raise RuntimeError("fresh_te[unb_idx] != fresh_oof -- alignment broken")
    fresh_anchor_unb = fresh_te_513[unb_idx]
    rae_fresh = float(rae(y_unb, fresh_anchor_unb))
    residual_fresh = y_unb - fresh_anchor_unb
    print(f"[load] fresh anchor te[unb] RAE = {rae_fresh:.4f} "
          f"(ref {REF_FRESH_ANCHOR_RAE:.4f})")

    # nb3200 blend partner
    nb3200_oof = np.load(NB3200_OOF).astype(np.float64)
    nb3200_te = np.load(NB3200_TE).astype(np.float64)
    assert nb3200_oof.shape == (n_unb,)
    assert nb3200_te.shape == (n_test,)
    rae_nb3200 = float(rae(y_unb, nb3200_oof))
    print(f"[load] real nb3200 pred_oof RAE = {rae_nb3200:.4f} (ref {REF_NB3200:.4f})")

    # scaffolds
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    # -- Load K18 / K19 feature indices --------------------------------------
    print("\n" + "-" * 78)
    print("STEP 0: load K18 (nb2604) + K19 (nb2231 trajectory) feature indices")
    print("-" * 78)
    with open(NB2604_SUMMARY) as f:
        nb2604 = json.load(f)
    K18_idx = np.array(nb2604["k18_idx_in_117col"], dtype=int)
    assert len(K18_idx) == 18, f"K18 len {len(K18_idx)} != 18"
    print(f"   K=18 idx (n={len(K18_idx)}): {K18_idx.tolist()}")
    with open(NB2231_SUMMARY) as f:
        nb2231 = json.load(f)
    K19_idx = np.array(reconstruct_K_from_trajectory(nb2231, 19), dtype=int)
    assert len(K19_idx) == 19, f"K19 len {len(K19_idx)} != 19"
    print(f"   K=19 idx (n={len(K19_idx)}): {K19_idx.tolist()}")

    # -- Build 117-col matrix ------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 0b: rebuild 117-col 5-way feature matrix")
    print("-" * 78)
    X_te_full, chembl_pool_size = build_117col_feature_matrix(te_smiles, n_test)
    print(f"   X_te_full = {X_te_full.shape}  chembl_pool={chembl_pool_size}")

    # -- STEP 1: fresh_K18_v3 + fresh_K19_v3 deep-30 bags on FRESH anchor -----
    print("\n" + "-" * 78)
    print(f"STEP 1: fresh_K18_v3 / fresh_K19_v3 residual-LGBM deep-30 "
          f"(n={len(RESID_SEEDS_DEEP)}) on FRESH anchor")
    print("-" * 78)
    K18_oof, K18_te, K18_seed_rae = build_K_30seed_bag(
        "fresh_K18", K18_idx, X_te_full, unb_idx, fresh_anchor_unb,
        residual_fresh, fresh_te_513, n_test, n_unb, RESID_SEEDS_DEEP,
    )
    K18_bag_rae = float(rae(y_unb, K18_oof))
    print(f"   [fresh_K18] 30-seed BAG-MEAN RAE = {K18_bag_rae:.4f}  "
          f"(per-seed mean {np.mean(K18_seed_rae):.4f})")
    K19_oof, K19_te, K19_seed_rae = build_K_30seed_bag(
        "fresh_K19", K19_idx, X_te_full, unb_idx, fresh_anchor_unb,
        residual_fresh, fresh_te_513, n_test, n_unb, RESID_SEEDS_DEEP,
    )
    K19_bag_rae = float(rae(y_unb, K19_oof))
    print(f"   [fresh_K19] 30-seed BAG-MEAN RAE = {K19_bag_rae:.4f}  "
          f"(per-seed mean {np.mean(K19_seed_rae):.4f})")

    np.save(DATA_PROCESSED / f"{TAG}_fresh_K18_v3_oof.npy", K18_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_fresh_K18_v3_te.npy", K18_te.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_fresh_K19_v3_oof.npy", K19_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_fresh_K19_v3_te.npy", K19_te.astype(np.float32))

    # -- STEP 2+3: fresh_nb3200 (q-blend -> learned clip), per-fold-mean ------
    print("\n" + "-" * 78)
    print(f"STEP 2+3: fresh_nb3090 (q-blend) -> fresh_nb3200 (learned clip), "
          f"per-fold-mean over {len(KF_SEEDS)} seeds")
    print("-" * 78)

    def _run_fresh(seed):
        return fresh_nb3200_one_seed(K18_oof, K19_oof, y_unb, unb_scaffolds, seed)

    fresh_res = per_fold_mean_over_seeds(_run_fresh)
    fresh_nb3200_pfm = fresh_res["per_fold_mean_rae"]
    fresh_nb3200_oof = fresh_res["median_oof"]
    print(f"   fresh_nb3200 per-fold-mean RAE = {fresh_nb3200_pfm:.4f} "
          f"+/- {fresh_res['per_fold_mean_std']:.4f}  "
          f"(pooled {fresh_res['pooled_mean_rae']:.4f})  "
          f"median_seed={fresh_res['median_seed']}")

    # -- Deploy fresh_nb3200 te (q-thr + clip on FULL 253) -------------------
    print("\n" + "-" * 78)
    print("STEP 3b: deploy fresh_nb3090 + fresh_nb3200 te (full-253 fit)")
    print("-" * 78)
    deploy_q_thr = float(np.quantile(K18_oof, Q_CUT))
    # fresh_nb3090 OOF + te
    fresh_nb3090_oof = blend_quantile_conditional(
        K18_oof, K19_oof, deploy_q_thr, W_LOW, W_HIGH,
    )
    fresh_nb3090_te = blend_quantile_conditional(
        K18_te, K19_te, deploy_q_thr, W_LOW, W_HIGH,
    )
    fresh_nb3090_rae = float(rae(y_unb, fresh_nb3090_oof))
    # learned clip on full 253 (nb3190 deploy convention)
    dql, dqh, dlo, dhi = pick_best_clip(y_unb, fresh_nb3090_oof)
    fresh_nb3200_te = np.clip(fresh_nb3090_te, dlo, dhi).astype(np.float32)
    fresh_nb3200_te_oof = np.clip(fresh_nb3090_oof, dlo, dhi)  # for te-alignment store
    fresh_nb3200_te_unb_rae = float(rae(y_unb, fresh_nb3200_te[unb_idx]))
    print(f"   fresh_nb3090 oof RAE = {fresh_nb3090_rae:.4f}  q_thr={deploy_q_thr:.4f}")
    print(f"   deploy clip = (q{dql:.2f},q{dqh:.2f}) -> ({dlo:.3f},{dhi:.3f})")
    print(f"   fresh_nb3200 te[unb] in-sample RAE = {fresh_nb3200_te_unb_rae:.4f}")
    print(f"   fresh_nb3200 te(513) mean={fresh_nb3200_te.mean():.3f} "
          f"std={fresh_nb3200_te.std():.3f}")

    np.save(DATA_PROCESSED / f"{TAG}_fresh_nb3090_oof.npy",
            fresh_nb3090_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_fresh_nb3090_te.npy",
            fresh_nb3090_te.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_fresh_nb3200_oof.npy",
            fresh_nb3200_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"te_{TAG}_fresh_nb3200.npy", fresh_nb3200_te)

    # -- STEP 4: output correlation fresh_nb3200 vs real nb3200 --------------
    print("\n" + "-" * 78)
    print("STEP 4: output decorrelation fresh_nb3200 vs real nb3200")
    print("-" * 78)
    # use the deploy-consistent fresh_nb3200 OOF (full-253 clip) vs nb3200 OOF
    corr_oof = float(np.corrcoef(fresh_nb3200_te_oof, nb3200_oof)[0, 1])
    corr_medianseed = float(np.corrcoef(fresh_nb3200_oof, nb3200_oof)[0, 1])
    resid_corr = float(np.corrcoef(
        fresh_nb3200_te_oof - y_unb, nb3200_oof - y_unb)[0, 1])
    print(f"   corr(fresh_nb3200_oof_full253, nb3200_oof) = {corr_oof:.4f}")
    print(f"   corr(fresh_nb3200_oof_medianseed, nb3200_oof) = {corr_medianseed:.4f}")
    print(f"   residual corr (vs truth)                  = {resid_corr:.4f}")
    output_corr = corr_oof  # gate uses the deploy-consistent full-253 OOF

    # -- STEP 5: blend {nb3200, fresh_nb3200} -------------------------------
    print("\n" + "-" * 78)
    print("STEP 5: per-fold SLSQP {nb3200, fresh_nb3200} + clip, "
          "and fixed-w grid")
    print("-" * 78)
    # use full-253-clip fresh_nb3200 OOF as the blend input (deploy-consistent,
    # honest per-fold-mean because the SLSQP/clip are re-fit per fold inside).
    p_fresh_for_blend = fresh_nb3200_te_oof

    # SLSQP blend (per-fold weights + per-fold clip)
    def _run_slsqp(seed):
        return blend_slsqp_one_seed(
            nb3200_oof, p_fresh_for_blend, y_unb, unb_scaffolds, seed,
            do_clip=True,
        )

    slsqp_res = per_fold_mean_over_seeds(_run_slsqp)
    slsqp_pfm = slsqp_res["per_fold_mean_rae"]
    # mean fold w_fresh across all seeds/folds
    all_w_fresh = [w for seed_ws in slsqp_res["extra_stack"] for w in seed_ws]
    mean_w_fresh = float(np.mean(all_w_fresh)) if all_w_fresh else 0.0
    print(f"   SLSQP blend per-fold-mean RAE = {slsqp_pfm:.4f} "
          f"+/- {slsqp_res['per_fold_mean_std']:.4f}  "
          f"(mean w_fresh={mean_w_fresh:.3f})")

    # Fixed-w blends
    fixed_results = {}
    for w in FIXED_W:
        def _run_fixed(seed, _w=w):
            return blend_fixed_w_one_seed(
                nb3200_oof, p_fresh_for_blend, y_unb, unb_scaffolds, seed,
                w_fresh=_w, do_clip=True,
            )
        fr = per_fold_mean_over_seeds(_run_fixed)
        fixed_results[w] = fr
        print(f"   fixed w_fresh={w:.1f} per-fold-mean RAE = "
              f"{fr['per_fold_mean_rae']:.4f} +/- {fr['per_fold_mean_std']:.4f}")

    # Baseline: nb3200 alone under the SAME per-fold-mean protocol (sanity ref)
    def _run_nb3200_alone(seed):
        return blend_fixed_w_one_seed(
            nb3200_oof, p_fresh_for_blend, y_unb, unb_scaffolds, seed,
            w_fresh=0.0, do_clip=False,
        )
    nb3200_alone_res = per_fold_mean_over_seeds(_run_nb3200_alone)
    nb3200_alone_pfm = nb3200_alone_res["per_fold_mean_rae"]
    print(f"   [ref] nb3200 alone (w=0, no extra clip) per-fold-mean RAE = "
          f"{nb3200_alone_pfm:.4f}")

    # -- Pick best blend ----------------------------------------------------
    blend_candidates = {"slsqp": slsqp_pfm}
    for w in FIXED_W:
        blend_candidates[f"fixed_{w}"] = fixed_results[w]["per_fold_mean_rae"]
    best_name = min(blend_candidates, key=blend_candidates.get)
    best_pfm = blend_candidates[best_name]
    print(f"\n   best blend = {best_name}  per-fold-mean RAE = {best_pfm:.4f}")

    # -- Build best-blend deploy te + OOF -----------------------------------
    if best_name == "slsqp":
        # deploy: fit SLSQP weights on full 253, then clip on full 253
        P_all = np.column_stack([nb3200_oof, p_fresh_for_blend])

        def obj_full(w):
            return rae(y_unb, P_all @ w)

        cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
        res_full = minimize(obj_full, np.array([0.8, 0.2]), method="SLSQP",
                            bounds=[(0.0, 1.0), (0.0, 1.0)], constraints=cons,
                            options={"maxiter": 200, "ftol": 1e-9})
        w_deploy = res_full.x if res_full.success else np.array([0.8, 0.2])
        w_deploy = np.clip(w_deploy, 0.0, 1.0)
        w_deploy = w_deploy / w_deploy.sum()
        best_oof = P_all @ w_deploy
        best_te = w_deploy[0] * nb3200_te + w_deploy[1] * fresh_nb3200_te
        deploy_w_fresh = float(w_deploy[1])
    else:
        w = float(best_name.split("_")[1])
        best_oof = (1.0 - w) * nb3200_oof + w * p_fresh_for_blend
        best_te = (1.0 - w) * nb3200_te + w * fresh_nb3200_te
        deploy_w_fresh = w
    # final clip on full 253
    bql, bqh, blo, bhi = pick_best_clip(y_unb, best_oof)
    best_oof_clipped = np.clip(best_oof, blo, bhi)
    best_te_clipped = np.clip(best_te, blo, bhi).astype(np.float32)
    best_oof_rae = float(rae(y_unb, best_oof_clipped))
    best_te_unb_rae = float(rae(y_unb, best_te_clipped[unb_idx]))
    print(f"   deploy w_fresh={deploy_w_fresh:.3f}  final clip=({blo:.3f},{bhi:.3f})")
    print(f"   best blend OOF(full253) RAE = {best_oof_rae:.4f}  "
          f"te[unb] in-sample RAE = {best_te_unb_rae:.4f}")

    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy", best_oof_clipped.astype(np.float32))
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", best_te_clipped)

    # -- GATES --------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATES")
    print("-" * 78)
    g_usable = fresh_nb3200_pfm < GATE_USABLE
    g_decorr = output_corr < GATE_DECORR
    g_break = best_pfm < GATE_BREAKTHROUGH
    verdicts = []
    if g_usable:
        verdicts.append("FRESH_PIPELINE_USABLE")
    if g_decorr:
        verdicts.append("OUTPUT_DECORRELATED")
    if g_break:
        verdicts.append("BREAKTHROUGH")
    verdict = " + ".join(verdicts) if verdicts else "NO_GATE"
    print(f"   fresh_nb3200 per-fold-mean RAE = {fresh_nb3200_pfm:.4f}  "
          f"(<{GATE_USABLE} -> {g_usable})")
    print(f"   output corr(fresh_nb3200, nb3200) = {output_corr:.4f}  "
          f"(<{GATE_DECORR} -> {g_decorr})")
    print(f"   best blend per-fold-mean RAE   = {best_pfm:.4f}  "
          f"(<{GATE_BREAKTHROUGH} -> {g_break})")
    print(f"   delta best blend vs nb3200 alone = {best_pfm - nb3200_alone_pfm:+.4f}")
    print(f"   delta best blend vs REF_NB3200   = {best_pfm - REF_NB3200:+.4f}")
    print(f"   verdict = {verdict}")

    # -- Submission (only on BREAKTHROUGH) ----------------------------------
    sub_csv = SUBMISSIONS / f"{TAG}_fresh_pipeline.csv"
    if g_break:
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": best_te_clipped,
        }).to_csv(sub_csv, index=False)
        print(f"   [save] {sub_csv}")
    else:
        print(f"   [skip] no BREAKTHROUGH; submission CSV not written")

    # -- Summary ------------------------------------------------------------
    summary = {
        "tag": TAG,
        "method": "full_nb3200_pipeline_on_fresh_chemprop_anchor_then_blend_nb3200",
        "fresh_anchor": "nb3350_chemprop_v3",
        "fresh_anchor_unblind_rae": rae_fresh,
        "frozen_anchor_unblind_rae": REF_FROZEN_ANCHOR_RAE,
        "pearson_anchors_ref": REF_PEARSON_ANCHORS,
        "nb3200_pred_oof_rae": rae_nb3200,
        "anchor_pre_unblind": False,
        "_note_anchor_pre_unblind": (
            "fresh nb3350 chemprop trained on the original 4139 (PRE-clean by "
            "corpus) but is a NEW 5-fold ensemble; the BLEND PARTNER nb3200 is "
            "POST-unblind-pipeline (built on frozen chemprop_aux). Treat the "
            "blend as POST for LB-shift purposes."
        ),
        "K18_idx_in_117col": K18_idx.tolist(),
        "K19_idx_in_117col": K19_idx.tolist(),
        "resid_seeds_deep": RESID_SEEDS_DEEP,
        "n_seeds_deep": len(RESID_SEEDS_DEEP),
        "resid_folds": RESID_FOLDS,
        "q_cut": Q_CUT,
        "w_low": W_LOW,
        "w_high": W_HIGH,
        "q_low_grid": Q_LOW_GRID,
        "q_high_grid": Q_HIGH_GRID,
        "kf_seeds": KF_SEEDS,
        "n_seeds": len(KF_SEEDS),
        "n_folds": N_FOLDS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "metric": "per_fold_mean (mean of 5 fold-val RAEs, averaged over seeds)",
        # STEP 1 K-bag
        "fresh_K18_bag_mean_rae": K18_bag_rae,
        "fresh_K19_bag_mean_rae": K19_bag_rae,
        "fresh_K18_perseed_rae_mean": float(np.mean(K18_seed_rae)),
        "fresh_K19_perseed_rae_mean": float(np.mean(K19_seed_rae)),
        # STEP 2 q-blend
        "fresh_nb3090_oof_rae": fresh_nb3090_rae,
        "deploy_q_thr": round(deploy_q_thr, 4),
        # STEP 3 fresh_nb3200
        "fresh_nb3200_per_fold_mean_rae": round(fresh_nb3200_pfm, 4),
        "fresh_nb3200_per_fold_mean_std": round(fresh_res["per_fold_mean_std"], 4),
        "fresh_nb3200_pooled_mean_rae": round(fresh_res["pooled_mean_rae"], 4),
        "fresh_nb3200_per_seed_pfm": fresh_res["per_seed_pfm"],
        "fresh_nb3200_median_seed": fresh_res["median_seed"],
        "fresh_nb3200_deploy_clip": [round(dlo, 4), round(dhi, 4)],
        "fresh_nb3200_deploy_clip_q": [float(dql), float(dqh)],
        "fresh_nb3200_te_unb_in_sample_rae": round(fresh_nb3200_te_unb_rae, 4),
        "fresh_nb3200_te_mean": float(fresh_nb3200_te.mean()),
        "fresh_nb3200_te_std": float(fresh_nb3200_te.std()),
        # STEP 4 decorrelation
        "output_corr_fresh_vs_nb3200": round(corr_oof, 4),
        "output_corr_medianseed_vs_nb3200": round(corr_medianseed, 4),
        "residual_corr_fresh_vs_nb3200": round(resid_corr, 4),
        # STEP 5 blends
        "slsqp_blend_per_fold_mean_rae": round(slsqp_pfm, 4),
        "slsqp_blend_per_fold_mean_std": round(slsqp_res["per_fold_mean_std"], 4),
        "slsqp_mean_w_fresh": round(mean_w_fresh, 4),
        "fixed_w_results": {
            f"{w}": {
                "per_fold_mean_rae": round(fixed_results[w]["per_fold_mean_rae"], 4),
                "per_fold_mean_std": round(fixed_results[w]["per_fold_mean_std"], 4),
                "per_seed_pfm": fixed_results[w]["per_seed_pfm"],
            } for w in FIXED_W
        },
        "nb3200_alone_per_fold_mean_rae": round(nb3200_alone_pfm, 4),
        "best_blend_name": best_name,
        "best_blend_per_fold_mean_rae": round(best_pfm, 4),
        "best_blend_deploy_w_fresh": round(deploy_w_fresh, 4),
        "best_blend_deploy_clip": [round(blo, 4), round(bhi, 4)],
        "best_blend_oof_full253_rae": round(best_oof_rae, 4),
        "best_blend_te_unb_in_sample_rae": round(best_te_unb_rae, 4),
        "best_blend_te_mean": float(best_te_clipped.mean()),
        "best_blend_te_std": float(best_te_clipped.std()),
        "delta_best_vs_nb3200_alone": round(best_pfm - nb3200_alone_pfm, 4),
        "delta_best_vs_ref_nb3200": round(best_pfm - REF_NB3200, 4),
        # gates
        "gate_usable": GATE_USABLE,
        "gate_decorr": GATE_DECORR,
        "gate_breakthrough": GATE_BREAKTHROUGH,
        "GATE_fresh_pipeline_usable": bool(g_usable),
        "GATE_output_decorrelated": bool(g_decorr),
        "GATE_breakthrough": bool(g_break),
        "verdict": verdict,
        # paths
        "fresh_nb3200_oof_path": str(DATA_PROCESSED / f"{TAG}_fresh_nb3200_oof.npy"),
        "fresh_nb3200_te_path": str(DATA_PROCESSED / f"te_{TAG}_fresh_nb3200.npy"),
        "pred_oof_path": str(DATA_PROCESSED / f"{TAG}_pred_oof.npy"),
        "te_npy_path": str(DATA_PROCESSED / f"te_{TAG}.npy"),
        "submission_csv": str(sub_csv) if g_break else None,
        "ref_nb3200": REF_NB3200,
        "ref_fresh_anchor_rae": REF_FRESH_ANCHOR_RAE,
        "ref_frozen_anchor_rae": REF_FROZEN_ANCHOR_RAE,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   fresh anchor RAE              = {rae_fresh:.4f}")
    print(f"   fresh_K18 / fresh_K19 bag RAE = {K18_bag_rae:.4f} / {K19_bag_rae:.4f}")
    print(f"   fresh_nb3090 oof RAE          = {fresh_nb3090_rae:.4f}")
    print(f"   fresh_nb3200 per-fold-mean    = {fresh_nb3200_pfm:.4f}  "
          f"(<{GATE_USABLE} -> {g_usable})")
    print(f"   corr(fresh_nb3200, nb3200)    = {output_corr:.4f}  "
          f"(<{GATE_DECORR} -> {g_decorr})")
    print(f"   nb3200 alone per-fold-mean    = {nb3200_alone_pfm:.4f}")
    print(f"   SLSQP blend per-fold-mean     = {slsqp_pfm:.4f} "
          f"(w_fresh={mean_w_fresh:.3f})")
    for w in FIXED_W:
        print(f"   fixed w={w:.1f} per-fold-mean      = "
              f"{fixed_results[w]['per_fold_mean_rae']:.4f}")
    print(f"   best blend ({best_name})        = {best_pfm:.4f}  "
          f"(<{GATE_BREAKTHROUGH} -> {g_break})")
    print(f"   verdict                       = {verdict}")
    print(f"   wall                          = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "fresh_K18_bag_mean_rae", "fresh_K19_bag_mean_rae",
        "fresh_nb3090_oof_rae",
        "fresh_nb3200_per_fold_mean_rae", "fresh_nb3200_per_fold_mean_std",
        "output_corr_fresh_vs_nb3200", "residual_corr_fresh_vs_nb3200",
        "nb3200_alone_per_fold_mean_rae",
        "slsqp_blend_per_fold_mean_rae", "slsqp_mean_w_fresh",
        "best_blend_name", "best_blend_per_fold_mean_rae",
        "delta_best_vs_nb3200_alone", "delta_best_vs_ref_nb3200",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
    print(f"  fixed_w_results: {res.get('fixed_w_results')}")
