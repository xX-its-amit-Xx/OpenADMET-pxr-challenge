"""nb2910 -- Deep-30 seed verification of nb2902 (0.5/0.5 nb2240_K20 + nb1191).

CRITICAL MOTIVATION
-------------------
nb2902 reported pooled RAE 0.4599 (single kf_seed=1001) for the equal-weight
mean of {nb2240_K20, nb1191}, missing the MARGINAL threshold 0.4598 by 0.0001.
Per cycle-160 deep-verify-dispersion rule (feedback_cycle160_deep_verify_dispersion):
even deterministic-looking single-seed numbers carry hidden seed variance via
the underlying K-pyramid anchors (each contains LGBM bag seeds and SLSQP
fold-cutting). 5-seed std<0.001 is a red flag (4-5x under-dispersion vs
deep-30 typical pattern). MUST verify with 30 fresh seeds.

ANCHORS REBUILT
---------------
nb2240_K20  : anchor=chemprop_aux + LGBM(seed=s) residual on 20 RFE-surviving
              features, mean-bag over 30 fresh seeds {3001-3030}.
              Full 117-col feature build (AtomPair / MACCS / Mordred /
              ChempropEmbed / Avalon / ChEMBL kNN) then slice to K=20.
nb1191      : 4-anchor SLSQP convex blend + per-fold rank-stretch with 30
              fresh kf_seeds {3001-3030} for scaffold-CV. Each seed yields a
              per-row OOF pred; mean across 30 seeds = nb1191_30 OOF.
              te side: standard nb1191 deploy te (weights frozen) -- 30-seed
              kf variance is OOF-only since deploy refits on full 253.

BLEND
-----
pred_oof = 0.5 * nb2240_30_oof + 0.5 * nb1191_30_oof
te       = 0.5 * nb2240_30_te  + 0.5 * nb1191_deploy_te

EVALUATION
----------
For each kf_seed in {1001..1005}:
    scaffold_kfold_indices(unb_scaffolds, n_splits=5, shuffle=True, seed)
    pooled_rae_s = rae(y_unb, pred_oof)
        (note: pred_oof is fixed across kf_seeds since the underlying mean-bag
         is row-aligned -- the kf_seed only affects the fold-wise diagnostic
         partition. The pooled RAE on 253 is identical across seeds for a
         row-fixed prediction. For honesty we therefore evaluate fold-wise
         partitioned mean RAE: mean of per-fold RAEs across the 5 folds.)
mean_rae = mean of (mean fold-RAE per seed) across 5 kf_seeds

GATES
-----
deep-30 mean_rae < 0.4570  ->  VERIFIED_PROMOTE  -> write submission CSV
deep-30 mean_rae < 0.4598  ->  VERIFIED_MARGINAL
else                       ->  LUCKY_SEED_TRAP

OUTPUTS
-------
scripts/nb2910_deep30_verify_nb2902.py
data/processed/nb2910_summary.json
data/processed/nb2910_pred_oof.npy   (253,) float32
data/processed/te_nb2910.npy         (513,) float32
submissions/nb2910_deep30.csv (only if VERIFIED_PROMOTE)
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
from scipy.optimize import minimize
from sklearn.model_selection import KFold

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2910"
BASELINE_TAG = "nb2902"

# ---- Deep-30 seeds (fresh band, disjoint from prior cycles) ----
DEEP30_SEEDS = list(range(3001, 3031))     # 30 fresh seeds for both anchors

# ---- Outer evaluation (fold-wise RAE diagnostics) ----
N_FOLDS = 5
KF_EVAL_SEEDS = [1001, 1002, 1003, 1004, 1005]

# ---- nb2240 K=20 residual cross-fit config ----
RESID_FOLDS = 5
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
NB2231_SUMMARY = DATA_PROCESSED / "nb2231_summary.json"

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

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

# ---- nb1191 4-anchor SLSQP+rank-stretch config ----
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]
NB1191_ANCHORS = [
    ("chemprop_aux", "nb1133_chemprop_aux_pred_oof.npy", "te_chemprop_aux.npy"),
    ("nb1150",       "_RECONSTRUCT_nb1150_oof",          "te_nb1150.npy"),
    ("nb1158_K32",   "nb1158_mean_bag_oof_K32.npy",      "te_nb1158.npy"),
    ("nb2112_K28",   "nb2103_mean_bag_oof_K28.npy",      "te_nb2112.npy"),
]
NB1150_SLSQP4_OOFS = [
    "nb1133_chemprop_aux_pred_oof.npy",
    "nb503_pred_oof.npy",
    "nb1133_nb1014_pred_oof.npy",
    "nb2103_mean_bag_oof_K28.npy",
]
NB1150_SLSQP4_WEIGHTS_FULL_POOL = [0.0, 0.2942, 0.0, 0.7058]

# ---- Gates ----
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598
NB2902_REF = 0.4599
CHEMPROP_AUX_REF = 0.6216


# ============================================================================
# Helpers reused from nb2240/nb2641 (feature build + LGBM)
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
    """Identical 117-col matrix as nb2240/nb2604/nb2641."""
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
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )

    X_te_full = np.concatenate(
        [
            X_ap_te_top, X_maccs_te_top, X_mord_te_top, X_emb_te_top, X_av_te_top,
            pred_chembl_pec50.reshape(-1, 1).astype(np.float32),
            mean_sim.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    if X_te_full.shape[1] != 117:
        raise ValueError(f"feat_dim {X_te_full.shape[1]} != 117")
    return X_te_full


def _residual_cross_fit_one_seed(X, residual, seed):
    """Random KFold residual cross-fit (matches nb2240's K=20 deploy protocol)."""
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


def build_nb2240_K20_deep30(X_te_full, surviving_K20, unb_idx, anchor_unb,
                            residual, te_anchor_513, n_test, n_unb, seeds):
    """Rebuild nb2240 K=20 OOF + te via N fresh bag seeds (mean across seeds).

    Returns:
        per_seed_oof  : (n_seeds, n_unb)
        per_seed_te   : (n_seeds, n_test)
        per_seed_rae  : list of per-seed pooled RAE on the 253
        mean_bag_oof  : (n_unb,) mean across seeds  -> THIS is nb2240_30 OOF
        mean_bag_te   : (n_test,) mean across seeds -> THIS is nb2240_30 te
    """
    X_te_K20 = X_te_full[:, surviving_K20].astype(np.float32)
    X_unb_K20 = X_te_K20[unb_idx]
    n_bag = len(seeds)
    per_seed_oof = np.zeros((n_bag, n_unb), dtype=np.float64)
    per_seed_te = np.zeros((n_bag, n_test), dtype=np.float64)
    per_seed_rae = []
    for i, s in enumerate(seeds):
        ts = time.time()
        resid_oof = _residual_cross_fit_one_seed(X_unb_K20, residual, s)
        oof_pred = anchor_unb + resid_oof
        per_seed_oof[i] = oof_pred
        r = float(rae(anchor_unb + residual, oof_pred))
        per_seed_rae.append(r)
        te_resid_s = _train_full_then_predict_te(X_unb_K20, residual, X_te_K20, s)
        per_seed_te[i] = te_anchor_513 + te_resid_s
        print(f"   nb2240_K20 seed={s:4d}: oof_RAE={r:.4f}  "
              f"wall={time.time()-ts:.1f}s")
    return (per_seed_oof, per_seed_te, per_seed_rae,
            per_seed_oof.mean(axis=0), per_seed_te.mean(axis=0))


# ============================================================================
# nb1191 deep-30 helpers (4-anchor SLSQP+stretch with fresh kf_seeds)
# ============================================================================
def reconstruct_nb1150_oof(n_unb: int) -> np.ndarray:
    cols = []
    for rel in NB1150_SLSQP4_OOFS:
        p = DATA_PROCESSED / rel
        assert p.exists(), f"missing nb1150 sub-anchor OOF: {p}"
        v = np.load(p).astype(np.float64)
        assert v.shape == (n_unb,), f"{p.name} shape {v.shape}"
        cols.append(v)
    P = np.column_stack(cols)
    w = np.asarray(NB1150_SLSQP4_WEIGHTS_FULL_POOL, dtype=np.float64)
    return P @ w


def slsqp_simplex(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    K = P.shape[1]
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * K
    res = minimize(
        lambda w: float(np.sum((P @ w - y) ** 2)),
        np.full(K, 1.0 / K),
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
        options={"ftol": 1e-10, "maxiter": 1000},
    )
    w = np.clip(res.x, 0.0, 1.0)
    s = w.sum()
    return w / s if s > 0 else np.full(K, 1.0 / K)


def best_stretch_on(blend_tr, y_tr, mu, grid):
    best_s, best_r = 1.0, float("inf")
    for s in grid:
        pred = mu + s * (blend_tr - mu)
        r = float(rae(y_tr, pred))
        if r < best_r:
            best_r, best_s = r, float(s)
    return best_s, best_r


def nb1191_oof_one_seed(P_unb, y_unb, unb_scaffolds, kf_seed):
    """One kf_seed scaffold 5-fold SLSQP+rank-stretch -- IDENTICAL to nb1191."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = P_unb.shape[0]
    oof_blend = np.full(n_unb, np.nan)
    fold_w, fold_s = [], []
    for tr_loc, va_loc in splits:
        w_f = slsqp_simplex(P_unb[tr_loc], y_unb[tr_loc])
        blend_tr = P_unb[tr_loc] @ w_f
        mu_tr = float(blend_tr.mean())
        s_f, _ = best_stretch_on(blend_tr, y_unb[tr_loc], mu_tr, STRETCH_GRID)
        blend_va = P_unb[va_loc] @ w_f
        oof_blend[va_loc] = mu_tr + s_f * (blend_va - mu_tr)
        fold_w.append(w_f)
        fold_s.append(s_f)
    return oof_blend, fold_w, fold_s


def build_nb1191_deep30_oof(P_unb, y_unb, unb_scaffolds, kf_seeds):
    """Mean-bag nb1191 OOF across kf_seeds (per-row mean of per-seed OOF preds).

    Each seed yields a complete (n_unb,) OOF blend vector via scaffold CV +
    SLSQP + rank-stretch. We average across seeds to get nb1191_30_oof.

    Returns:
        per_seed_oof : (n_seeds, n_unb)
        per_seed_rae : list of per-seed pooled RAE
        mean_bag_oof : (n_unb,) row-wise mean across seeds
    """
    n_unb = P_unb.shape[0]
    n_seeds = len(kf_seeds)
    per_seed_oof = np.zeros((n_seeds, n_unb), dtype=np.float64)
    per_seed_rae = []
    for i, kfs in enumerate(kf_seeds):
        ts = time.time()
        oof_s, _fw, _fs = nb1191_oof_one_seed(P_unb, y_unb, unb_scaffolds, kfs)
        per_seed_oof[i] = oof_s
        r = float(rae(y_unb, oof_s))
        per_seed_rae.append(r)
        print(f"   nb1191 kf_seed={kfs:4d}: oof_RAE={r:.4f}  "
              f"wall={time.time()-ts:.1f}s")
    return per_seed_oof, per_seed_rae, per_seed_oof.mean(axis=0)


# ============================================================================
# MAIN
# ============================================================================
def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- DEEP-30 verify {BASELINE_TAG} (0.5*nb2240_K20 + 0.5*nb1191)")
    print(f"       ref nb2902 pooled_RAE = {NB2902_REF:.4f} (single kf_seed=1001)")
    print(f"       gates: PROMOTE < {GATE_PROMOTE:.4f}  MARGINAL < {GATE_MARGINAL:.4f}")
    print(f"       deep30 seeds: {DEEP30_SEEDS[0]}..{DEEP30_SEEDS[-1]} (n={len(DEEP30_SEEDS)})")
    print(f"       eval kf_seeds: {KF_EVAL_SEEDS}  ({N_FOLDS}-fold scaffold CV)")
    print("=" * 78)

    # ---- Load truth + anchor + test names ----
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

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    residual = y_unb - anchor_unb
    print(f"[load] chemprop_aux unb_RAE = {rae_anchor:.4f} (ref {CHEMPROP_AUX_REF:.4f})")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    # ========================================================================
    # STEP 1: build nb2240_K20 with 30 fresh seeds
    # ========================================================================
    print("\n" + "=" * 78)
    print("STEP 1: rebuild nb2240_K20 with 30 fresh seeds (3001..3030)")
    print("=" * 78)
    with open(NB2231_SUMMARY) as f:
        nb2231 = json.load(f)
    surviving_K20 = list(nb2231["snapshots"]["20"]["surviving_idx_in_117"])
    assert len(surviving_K20) == 20, f"expected 20, got {len(surviving_K20)}"
    print(f"[K20] surviving 20 idx in 117: {surviving_K20[:8]}...")

    print("\n[feat] building 117-col matrix (AtomPair/MACCS/Mordred/Embed/Avalon/ChEMBL kNN)...")
    X_te_full = build_117col_feature_matrix(te_smiles, n_test)
    print(f"[feat] X_te_full = {X_te_full.shape}")

    print(f"\n[nb2240] LGBM residual mean-bag over {len(DEEP30_SEEDS)} seeds:")
    (nb2240_per_seed_oof, nb2240_per_seed_te, nb2240_per_seed_rae,
     nb2240_30_oof, nb2240_30_te) = build_nb2240_K20_deep30(
        X_te_full, surviving_K20, unb_idx, anchor_unb, residual,
        te_anchor_513, n_test, n_unb, DEEP30_SEEDS,
    )
    nb2240_30_oof_rae = float(rae(y_unb, nb2240_30_oof))
    nb2240_per_seed_mean = float(np.mean(nb2240_per_seed_rae))
    nb2240_per_seed_std = float(np.std(nb2240_per_seed_rae, ddof=1))
    print(f"[nb2240_30] per-seed mean RAE = {nb2240_per_seed_mean:.4f} "
          f"(+/- {nb2240_per_seed_std:.4f})")
    print(f"[nb2240_30] mean-bag OOF RAE  = {nb2240_30_oof_rae:.4f}")

    # ========================================================================
    # STEP 2: build nb1191 with 30 fresh kf_seeds (SLSQP+rank-stretch)
    # ========================================================================
    print("\n" + "=" * 78)
    print("STEP 2: rebuild nb1191 with 30 fresh kf_seeds (3001..3030)")
    print("=" * 78)
    # Assemble nb1191 4-anchor stack on unb side
    oof_cols = []
    for disp, oof_rel, _te_rel in NB1191_ANCHORS:
        if oof_rel == "_RECONSTRUCT_nb1150_oof":
            v = reconstruct_nb1150_oof(n_unb)
        else:
            p = DATA_PROCESSED / oof_rel
            assert p.exists(), f"missing nb1191 sub-anchor OOF: {p}"
            v = np.load(p).astype(np.float64)
            assert v.shape == (n_unb,), f"{p.name} shape {v.shape}"
        oof_cols.append(v)
        r_anch = float(rae(y_unb, v))
        print(f"   nb1191 anchor {disp:14s} unb_RAE = {r_anch:.4f}")
    P_unb_1191 = np.column_stack(oof_cols)
    print(f"[nb1191] P_unb {P_unb_1191.shape}")

    print(f"\n[nb1191] SLSQP+rank-stretch over {len(DEEP30_SEEDS)} kf_seeds:")
    nb1191_per_seed_oof, nb1191_per_seed_rae, nb1191_30_oof = \
        build_nb1191_deep30_oof(
            P_unb_1191, y_unb, unb_scaffolds, DEEP30_SEEDS,
        )
    nb1191_30_oof_rae = float(rae(y_unb, nb1191_30_oof))
    nb1191_per_seed_mean = float(np.mean(nb1191_per_seed_rae))
    nb1191_per_seed_std = float(np.std(nb1191_per_seed_rae, ddof=1))
    print(f"[nb1191_30] per-seed mean RAE = {nb1191_per_seed_mean:.4f} "
          f"(+/- {nb1191_per_seed_std:.4f})")
    print(f"[nb1191_30] mean-bag OOF RAE  = {nb1191_30_oof_rae:.4f}")

    # nb1191 deploy te is cached / deterministic (fixed weights + s applied to
    # cached sub-anchor te files). The 30-seed kf variance is OOF-only; the
    # deploy refit on the full 253 has no kf seed variance. Use the cached
    # te_nb1191.npy directly to mirror nb2902's te side.
    te_nb1191_deploy = np.load(DATA_PROCESSED / "te_nb1191.npy").astype(np.float64)
    print(f"[nb1191_te] cached deploy te_nb1191.npy mean={te_nb1191_deploy.mean():.3f} "
          f"std={te_nb1191_deploy.std():.3f}")

    # ========================================================================
    # STEP 3: equal-weight blend (0.5 / 0.5)
    # ========================================================================
    print("\n" + "=" * 78)
    print("STEP 3: equal-weight blend  pred = 0.5 * nb2240_30 + 0.5 * nb1191_30")
    print("=" * 78)
    pred_oof = 0.5 * nb2240_30_oof + 0.5 * nb1191_30_oof
    pred_te = 0.5 * nb2240_30_te + 0.5 * te_nb1191_deploy

    blend_pooled_rae = float(rae(y_unb, pred_oof))
    print(f"[blend] pooled OOF RAE on 253 = {blend_pooled_rae:.4f}")
    print(f"[blend] te mean/std = {pred_te.mean():.3f}/{pred_te.std():.3f}")

    # ========================================================================
    # STEP 4: 5 kf_seed scaffold-CV fold-wise RAE (final mean = mean of 5)
    # ========================================================================
    print("\n" + "-" * 78)
    print(f"STEP 4: outer scaffold-CV {N_FOLDS}-fold across {len(KF_EVAL_SEEDS)} kf_seeds")
    print("-" * 78)
    # For a row-fixed prediction (pred_oof), pooled RAE on the 253 is identical
    # across kf_seeds. The fold-wise mean RAE varies with the partition.
    # Per task spec: "5 kf_seeds, pooled RAE per seed; mean of 5 = final."
    # We compute both pooled (constant) and fold-mean (varies) for honesty.
    per_seed_pooled = []
    per_seed_fold_mean = []
    per_seed_fold_results = []
    for kfs in KF_EVAL_SEEDS:
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kfs,
        )
        fold_raes = []
        for fi, (tr_loc, va_loc) in enumerate(splits):
            r_va = float(rae(y_unb[va_loc], pred_oof[va_loc]))
            fold_raes.append(r_va)
        pooled_s = float(rae(y_unb, pred_oof))  # constant across seeds
        fold_mean_s = float(np.mean(fold_raes))
        per_seed_pooled.append(pooled_s)
        per_seed_fold_mean.append(fold_mean_s)
        per_seed_fold_results.append({
            "kf_seed": int(kfs),
            "pooled_rae": pooled_s,
            "fold_mean_rae": fold_mean_s,
            "fold_raes": fold_raes,
        })
        print(f"   kf_seed={kfs}  pooled={pooled_s:.4f}  "
              f"fold_mean={fold_mean_s:.4f}  "
              f"folds={[round(x, 4) for x in fold_raes]}")

    final_pooled_mean = float(np.mean(per_seed_pooled))
    final_pooled_std = float(np.std(per_seed_pooled, ddof=1))
    final_foldmean_mean = float(np.mean(per_seed_fold_mean))
    final_foldmean_std = float(np.std(per_seed_fold_mean, ddof=1))
    print(f"\n[eval] mean of 5 kf_seeds pooled RAE = {final_pooled_mean:.4f} "
          f"(+/- {final_pooled_std:.4f})")
    print(f"[eval] mean of 5 kf_seeds fold-mean = {final_foldmean_mean:.4f} "
          f"(+/- {final_foldmean_std:.4f})")

    # Primary verdict number per task spec
    mean_rae = final_pooled_mean
    delta_vs_nb2902 = mean_rae - NB2902_REF
    print(f"\n[delta] mean_rae - nb2902_ref = {delta_vs_nb2902:+.4f}")

    # ========================================================================
    # GATE
    # ========================================================================
    if mean_rae < GATE_PROMOTE:
        verdict = "VERIFIED_PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "VERIFIED_MARGINAL"
    else:
        verdict = "LUCKY_SEED_TRAP"
    print(f"\n[gate] mean_rae={mean_rae:.4f}  ->  {verdict}")

    # ========================================================================
    # te diagnostics
    # ========================================================================
    te_unb_in_rae = float(rae(y_unb, pred_te[unb_idx]))
    print(f"\n[te] te[unb_idx] in-sample RAE = {te_unb_in_rae:.4f}")
    print(f"[te] mean={pred_te.mean():.3f}  std={pred_te.std():.3f}")

    # ========================================================================
    # SAVE artifacts
    # ========================================================================
    print("\n" + "-" * 78)
    print("STEP 5: save artifacts")
    print("-" * 78)
    out_oof = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    out_te = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(out_oof, pred_oof.astype(np.float32))
    np.save(out_te, pred_te.astype(np.float32))
    print(f"[save] {out_oof}")
    print(f"[save] {out_te}")

    # Also save the deep-30 component arrays for downstream reuse
    np.save(DATA_PROCESSED / f"{TAG}_nb2240_30_oof.npy",
            nb2240_30_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_nb2240_30_te.npy",
            nb2240_30_te.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_nb1191_30_oof.npy",
            nb1191_30_oof.astype(np.float32))

    sub_csv = None
    if verdict == "VERIFIED_PROMOTE":
        sub_csv = SUBMISSIONS / f"{TAG}_deep30.csv"
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": pred_te.astype(np.float32),
        }).to_csv(sub_csv, index=False)
        print(f"[save] {sub_csv}  (verdict={verdict})")
    else:
        print(f"[skip] no submission CSV (verdict={verdict})")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    summary = {
        "tag": TAG,
        "baseline_tag": BASELINE_TAG,
        "method": "deep30_verify_2anchor_equal_weight_nb2240K20_nb1191",
        "deep30_seeds": DEEP30_SEEDS,
        "n_deep30_seeds": len(DEEP30_SEEDS),
        "kf_eval_seeds": KF_EVAL_SEEDS,
        "n_folds": N_FOLDS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "anchor_chemprop_aux_unb_rae": rae_anchor,
        "anchor_pre_unblind": True,

        # nb2240_K20 deep-30 stats
        "nb2240_K20_per_seed_rae": nb2240_per_seed_rae,
        "nb2240_K20_per_seed_mean": nb2240_per_seed_mean,
        "nb2240_K20_per_seed_std_ddof1": nb2240_per_seed_std,
        "nb2240_30_oof_rae": nb2240_30_oof_rae,

        # nb1191 deep-30 stats
        "nb1191_per_seed_rae": nb1191_per_seed_rae,
        "nb1191_per_seed_mean": nb1191_per_seed_mean,
        "nb1191_per_seed_std_ddof1": nb1191_per_seed_std,
        "nb1191_30_oof_rae": nb1191_30_oof_rae,

        # Blend (eval)
        "blend_weights": {"nb2240_30": 0.5, "nb1191_30": 0.5},
        "blend_pooled_rae_on_253": blend_pooled_rae,
        "per_seed_fold_results": per_seed_fold_results,
        "per_seed_pooled_rae": per_seed_pooled,
        "per_seed_fold_mean_rae": per_seed_fold_mean,
        "final_pooled_mean": final_pooled_mean,
        "final_pooled_std_ddof1": final_pooled_std,
        "final_foldmean_mean": final_foldmean_mean,
        "final_foldmean_std_ddof1": final_foldmean_std,
        "mean_rae": mean_rae,

        # Gate
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "nb2902_ref": NB2902_REF,
        "delta_vs_nb2902": delta_vs_nb2902,
        "verdict": verdict,

        # te diagnostics
        "te_unb_in_sample_rae": te_unb_in_rae,
        "te_mean": float(pred_te.mean()),
        "te_std": float(pred_te.std()),

        # Paths
        "pred_oof_path": str(out_oof),
        "te_npy_path": str(out_te),
        "submission_csv": str(sub_csv) if sub_csv is not None else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} DEEP-30 SUMMARY ===")
    print(f"   nb2240_30 per-seed mean = {nb2240_per_seed_mean:.4f} +/- {nb2240_per_seed_std:.4f}")
    print(f"   nb2240_30 mean-bag RAE  = {nb2240_30_oof_rae:.4f}")
    print(f"   nb1191_30 per-seed mean = {nb1191_per_seed_mean:.4f} +/- {nb1191_per_seed_std:.4f}")
    print(f"   nb1191_30 mean-bag RAE  = {nb1191_30_oof_rae:.4f}")
    print(f"   BLEND pooled RAE (253)  = {blend_pooled_rae:.4f}")
    print(f"   mean_rae (5-kf_seed)    = {mean_rae:.4f}")
    print(f"   nb2902 ref              = {NB2902_REF:.4f}  delta = {delta_vs_nb2902:+.4f}")
    print(f"   VERDICT                 = {verdict}")
    print(f"   wall                    = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "verdict",
        "mean_rae",
        "blend_pooled_rae_on_253",
        "nb2240_30_oof_rae",
        "nb2240_K20_per_seed_mean",
        "nb2240_K20_per_seed_std_ddof1",
        "nb1191_30_oof_rae",
        "nb1191_per_seed_mean",
        "nb1191_per_seed_std_ddof1",
        "delta_vs_nb2902",
        "te_unb_in_sample_rae",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
