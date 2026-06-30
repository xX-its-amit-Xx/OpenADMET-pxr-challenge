"""nb2263 -- Lucky-seed-aware RFE: pick K by minimising 30-seed mean RAE.

CONTEXT:
    nb2241 verified that nb2231's K=20 RFE snapshot is LUCKY-SEED: the
    original 5-seed mean_bag of 0.4763 inflated to 0.4844 (+0.008) under
    a fresh 30-seed pool, and the original 5-seed per_seed_mean 0.5068
    inflated to 0.5173 (+0.010). The under-dispersion ratio of the
    nb2231 claim std (0.0131) to the fresh 30-seed std (0.0154) was 0.85
    (lucky-mean, not lucky-std). The mean drift of +0.008..+0.010 is the
    real lucky-seed exposure.

    The deeper problem: nb2231 PICKED K=20 using single-seed (kf_seed=1001)
    greedy drop selection. Each drop was chosen to minimise the
    single-seed RAE at that step, then the resulting subset was
    snapshot-evaluated with 5 full seeds. The drop-by-drop greedy path
    was lucky-seeded at EVERY step of the trajectory.

HYPOTHESIS:
    A lucky-seed-aware RFE that uses 30 fresh kf_seeds at the DROP-SELECTION
    step (not just at the snapshot step) will:
      a. yield a different drop order than nb2231's K=20 path
      b. potentially pick a different K optimum (could be K=18, K=22,
         or anywhere on the K-grid)
      c. produce a substrate that PASSES the nb2241-style fresh-seed gate
         (mean_bag drift <= 0.005)

PROTOCOL:
    1. Rebuild the 117-col 5-way K-tuned matrix exactly as nb2231 (same
       AtomPair / MACCS / Mordred / ChempropEmbed / Avalon + ChEMBL kNN
       sources). PRE-unblind clean: chemprop_aux te slice as anchor.
    2. Start from nb2231's SHAP top-28 indices (identical seed of the
       greedy walk; we only change drop-selection criterion).
    3. Greedy backward RFE K=28 -> K=10. At EACH step, for each candidate
       feature in the current subset:
         a. Evaluate scaffold-CV RAE with that feature dropped, for
            ALL 30 fresh kf_seeds {1116..1145}.
         b. Compute the 30-seed mean RAE = the lucky-aware score.
       Drop the feature whose removal MINIMISES the 30-seed mean RAE.
    4. Save trajectory K=28 -> K=10 (19 snapshots: K_after values
       28, 27, ..., 10), each with 30-seed mean/std/min/max/median RAE
       and the dropped feature.
    5. Identify K_optimum_multi_seed = argmin_K of the trajectory's
       30-seed mean RAE.
    6. Compare K_optimum_multi_seed vs nb2231's K=20 (single-seed pick).
    7. If K_opt != 20, the lucky-aware path identified a different K
       and we should pyramid-wrap-test it (the pyramid is what extracts
       LB-translatable signal; a standalone-better K may or may not
       translate to a better pyramid wrap).

PYRAMID WRAP TEST (only if K_opt != 20):
    For K_opt, rebuild K_opt residual LGBM mean-bag over 5 RESID_SEEDS
    (identical to nb2240 / nb2250 protocol), then evaluate the 5-anchor
    pyramid {nb2263_K{K_opt}, chemprop_aux, nb1191, nb503, nb562} with
    SLSQP + rank-stretch under DEEP_SEEDS pool. Verdict vs
    nb2240 K=20 deep-30 reference 0.4601.

Outputs:
    scripts/nb2263_lucky_aware_rfe.py
    data/processed/nb2263_summary.json

References:
    nb2231 single-seed RFE K=20 pick (per_seed_mean 0.5068 SUSPECT)
    nb2241 fresh-30-seed verify (mean_bag 0.4844; FAIL gate by +0.008)
    nb2240 K=20 pyramid wrap deep-30 (0.4601 +/- 0.0017)  -- reference
    nb2250 K=18/22 pyramid wrap (K=20 stays optimal; deltas +0.003, +0.007)
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
from pxr.paths import DATA_PROCESSED

TAG = "nb2263"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

K_START = 28
K_MIN = 10
# 30 fresh kf_seeds (identical pool to nb2241 verify) for DROP SELECTION
KF_SEEDS_FRESH = list(range(1116, 1146))   # 30 seeds
# Original nb2231 5-seed pool retained for sanity comparison only
KF_SEEDS_ORIGINAL = [1001, 1002, 1003, 1004, 1005]

N_FOLDS = 5

# Pyramid-wrap test parameters (used only if K_opt != 20)
RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]
LB_W_OOF = 0.51
LB_W_TE = 0.49

# Substrate cache paths (same as nb2231 / nb2241 / nb2250)
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
NB2063_SHAP_IMP = DATA_PROCESSED / "nb2063_shap_importance_full117.npy"
NB2103_SUMMARY = DATA_PROCESSED / "nb2103_summary.json"
NB2231_SUMMARY = DATA_PROCESSED / "nb2231_summary.json"
NB2241_SUMMARY = DATA_PROCESSED / "nb2241_summary.json"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

CHEMPROP_AUX_REF = 0.6216
NB2231_K20_SINGLE_SEED_MEAN_BAG = 0.4763
NB2241_K20_FRESH30_MEAN_BAG = 0.4844
NB2240_K20_DEEP30_PYRAMID = 0.4601
DECISION_MARGIN = 0.003

# nb1191 pyramid-reconstruction (identical to nb2240/nb2250)
NB1191_DEPLOY_WEIGHTS = {
    "chemprop_aux": 0.0,
    "nb1150":       0.641721304028517,
    "nb1158_K32":   0.23970131778546713,
    "nb2112_K28":   0.11857737818601592,
}
NB1191_DEPLOY_S = 1.031
NB1150_SLSQP4_OOFS = [
    "nb1133_chemprop_aux_pred_oof.npy",
    "nb503_pred_oof.npy",
    "nb1133_nb1014_pred_oof.npy",
    "nb2103_mean_bag_oof_K28.npy",
]
NB1150_SLSQP4_WEIGHTS = [0.0, 0.2942, 0.0, 0.7058]


# ============================================================================
# helpers (mirror nb2231 / nb2241)
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


def _lgbm_params(seed: int) -> dict:
    """Identical to nb2231 / nb2241 / nb2250."""
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


def _scaffold_cv_one_seed(X: np.ndarray, residual: np.ndarray,
                          unb_scaffolds: list[str], kf_seed: int) -> np.ndarray:
    """One scaffold 5-fold cross-fit of residual LGBM; returns OOF residual."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n = len(residual)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in splits:
        mdl = lgb.LGBMRegressor(**_lgbm_params(kf_seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _multi_seed_mean_rae(X_full: np.ndarray, col_idx: list[int],
                         residual: np.ndarray, anchor: np.ndarray,
                         y_unb: np.ndarray, unb_scaffolds: list[str],
                         kf_seeds: list[int]) -> tuple[float, float, float, float, list[float]]:
    """Multi-seed scaffold-CV: returns (mean_rae, std_rae, min_rae, max_rae,
    per_seed_rae). The mean_rae is the lucky-aware score used for drop
    selection and snapshot."""
    X_sub = X_full[:, col_idx].astype(np.float32)
    per = []
    for s in kf_seeds:
        oof = _scaffold_cv_one_seed(X_sub, residual, unb_scaffolds, s)
        per.append(float(rae(y_unb, anchor + oof)))
    arr = np.asarray(per)
    return (
        float(arr.mean()),
        float(arr.std()),
        float(arr.min()),
        float(arr.max()),
        [float(x) for x in per],
    )


# ============================================================================
# Pyramid-wrap test helpers (only run if K_opt != 20)
# ============================================================================
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


def slsqp_simplex(P, y):
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


def cv_run_for_seed(P_unb, y_unb, unb_scaffolds, kf_seed):
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
    return float(rae(y_unb, oof_blend)), oof_blend, fold_w, fold_s


def reconstruct_nb1150_oof(n_unb):
    cols = []
    for rel in NB1150_SLSQP4_OOFS:
        p = DATA_PROCESSED / rel
        assert p.exists(), f"missing nb1150 sub-anchor OOF: {p}"
        v = np.load(p).astype(np.float64)
        assert v.shape == (n_unb,), f"{p.name} shape {v.shape}"
        cols.append(v)
    P = np.column_stack(cols)
    w = np.asarray(NB1150_SLSQP4_WEIGHTS, dtype=np.float64)
    return P @ w


def reconstruct_nb1191_oof(n_unb):
    chemprop_oof = np.load(DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy").astype(np.float64)
    nb1150_oof = reconstruct_nb1150_oof(n_unb)
    nb1158_oof = np.load(DATA_PROCESSED / "nb1158_mean_bag_oof_K32.npy").astype(np.float64)
    nb2112_oof = np.load(DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy").astype(np.float64)
    blend = (
        NB1191_DEPLOY_WEIGHTS["chemprop_aux"] * chemprop_oof
        + NB1191_DEPLOY_WEIGHTS["nb1150"]       * nb1150_oof
        + NB1191_DEPLOY_WEIGHTS["nb1158_K32"]   * nb1158_oof
        + NB1191_DEPLOY_WEIGHTS["nb2112_K28"]   * nb2112_oof
    )
    mu = float(blend.mean())
    return mu + NB1191_DEPLOY_S * (blend - mu)


def pyramid_wrap_test(X_unb_full, X_te_full, K_opt_cols,
                      anchor_unb, anchor_te_513,
                      residual, y_unb, n_test, n_unb,
                      unb_scaffolds):
    """Build K_opt anchor with mean-bag over RESID_SEEDS, then run
    deep-30 pyramid evaluation. Returns dict of metrics."""
    X_unb_K = X_unb_full[:, K_opt_cols].astype(np.float32)
    X_te_K = X_te_full[:, K_opt_cols].astype(np.float32)
    print(f"   [wrap] X_unb_K={X_unb_K.shape}  X_te_K={X_te_K.shape}")
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_te_resid = np.zeros((len(RESID_SEEDS), n_test), dtype=np.float64)
    per_seed_rae_corr = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        resid_oof = _residual_cross_fit_one_seed(X_unb_K, residual, s)
        per_seed_corrected[i] = anchor_unb + resid_oof
        per_seed_rae_corr.append(float(rae(y_unb, anchor_unb + resid_oof)))
        te_resid_s = _train_full_then_predict_te(X_unb_K, residual, X_te_K, s)
        per_seed_te_resid[i] = te_resid_s
        print(f"     [wrap seed={s:3d}] resid_oof_rae={per_seed_rae_corr[-1]:.4f}  "
              f"wall={time.time()-ts:.1f}s")
    mean_bag_oof_K = per_seed_corrected.mean(axis=0)
    mean_bag_te_resid_K = per_seed_te_resid.mean(axis=0)
    te_K_513 = anchor_te_513 + mean_bag_te_resid_K
    rae_K_mean_bag = float(rae(y_unb, mean_bag_oof_K))
    rae_K_per_seed_mean = float(np.mean(per_seed_rae_corr))
    print(f"   [wrap] mean-bag standalone RAE = {rae_K_mean_bag:.4f}  "
          f"per_seed_mean = {rae_K_per_seed_mean:.4f}")

    chemprop_oof = np.load(DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy").astype(np.float64)
    nb503_oof = np.load(DATA_PROCESSED / "nb503_pred_oof.npy").astype(np.float64)
    nb562_oof = np.load(DATA_PROCESSED / "nb562_pred_oof.npy").astype(np.float64)
    nb1191_oof = reconstruct_nb1191_oof(n_unb)
    te_nb1191 = np.load(DATA_PROCESSED / "te_nb1191.npy").astype(np.float64)
    te_nb503 = np.load(DATA_PROCESSED / "te_nb503.npy").astype(np.float64)
    te_nb562 = np.load(DATA_PROCESSED / "te_nb562.npy").astype(np.float64)

    anchors_list = [
        (f"nb{TAG}_Kopt",  mean_bag_oof_K.astype(np.float64), te_K_513.astype(np.float64)),
        ("chemprop_aux",   chemprop_oof,                       anchor_te_513),
        ("nb1191",         nb1191_oof,                         te_nb1191),
        ("nb503",          nb503_oof,                          te_nb503),
        ("nb562",          nb562_oof,                          te_nb562),
    ]
    oof_cols, te_cols, indiv_rae = [], [], {}
    for disp, oof, te_arr in anchors_list:
        r = float(rae(y_unb, oof))
        indiv_rae[disp] = r
        oof_cols.append(oof)
        te_cols.append(te_arr)
        print(f"     [wrap anchor] {disp:18s} oof_RAE={r:.4f}")
    P_unb = np.column_stack(oof_cols)
    P_te = np.column_stack(te_cols)

    per_seed_pyramid_rae = []
    fold_s_all = []
    for kf_seed in KF_SEEDS_FRESH:
        pooled, _o, _w, fs = cv_run_for_seed(P_unb, y_unb, unb_scaffolds, kf_seed)
        per_seed_pyramid_rae.append(float(pooled))
        fold_s_all.extend([float(x) for x in fs])
    arr = np.asarray(per_seed_pyramid_rae)
    pyramid_mean = float(arr.mean())
    pyramid_std = float(arr.std())
    pyramid_min = float(arr.min())
    pyramid_max = float(arr.max())
    print(f"   [wrap] PYRAMID DEEP-30 RAE = {pyramid_mean:.4f} +/- {pyramid_std:.4f}  "
          f"(min={pyramid_min:.4f} max={pyramid_max:.4f})")

    # Deploy refit (for downstream submission build, not consumed here)
    w_deploy = slsqp_simplex(P_unb, y_unb)
    blend_unb = P_unb @ w_deploy
    mu_deploy = float(blend_unb.mean())
    s_deploy = float(np.mean(fold_s_all))
    in_rae_final = float(rae(y_unb, mu_deploy + s_deploy * (blend_unb - mu_deploy)))
    blend_te = P_te @ w_deploy
    deploy_te = (mu_deploy + s_deploy * (blend_te - mu_deploy)).astype(np.float32)

    delta_vs_K20 = pyramid_mean - NB2240_K20_DEEP30_PYRAMID
    if delta_vs_K20 < -DECISION_MARGIN:
        verdict = "BEATS_K20"
    elif abs(delta_vs_K20) <= DECISION_MARGIN:
        verdict = "FLAT_K20"
    else:
        verdict = "WORSE_K20"

    return {
        "K_opt_cols": [int(j) for j in K_opt_cols],
        "rae_K_mean_bag_standalone": rae_K_mean_bag,
        "rae_K_per_seed_mean_standalone": rae_K_per_seed_mean,
        "indiv_anchor_rae": indiv_rae,
        "pyramid_deep30_mean": pyramid_mean,
        "pyramid_deep30_std": pyramid_std,
        "pyramid_deep30_min": pyramid_min,
        "pyramid_deep30_max": pyramid_max,
        "per_seed_pyramid_rae": per_seed_pyramid_rae,
        "deploy_w": [float(w) for w in w_deploy],
        "deploy_mu": mu_deploy,
        "deploy_s": s_deploy,
        "in_sample_rae": in_rae_final,
        "delta_vs_nb2240_K20_pyramid": delta_vs_K20,
        "verdict_vs_nb2240_K20": verdict,
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
    }


# ============================================================================
# MAIN
# ============================================================================
def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- LUCKY-SEED-AWARE RFE  K={K_START} -> K={K_MIN}")
    print(f"          anchor = {ANCHOR}  scaffold-CV {N_FOLDS}-fold")
    print(f"          DROP-SELECTION criterion: 30-seed mean RAE "
          f"(kf_seeds {KF_SEEDS_FRESH[0]}..{KF_SEEDS_FRESH[-1]})")
    print(f"          nb2231 single-seed pick was K=20  "
          f"(claim mean_bag {NB2231_K20_SINGLE_SEED_MEAN_BAG:.4f}, "
          f"fresh-30 mean_bag {NB2241_K20_FRESH30_MEAN_BAG:.4f})")
    print(f"          if K_opt != 20: pyramid-wrap test vs "
          f"nb2240 K=20 deep-30 {NB2240_K20_DEEP30_PYRAMID:.4f}")
    print("=" * 78)

    # ---- load SHAP top-28 ----
    if not NB2063_SHAP_IMP.exists():
        raise FileNotFoundError(f"missing {NB2063_SHAP_IMP}")
    shap_imp_full117 = np.load(NB2063_SHAP_IMP).astype(np.float32)
    full_rank_order = np.argsort(-shap_imp_full117).astype(np.int32)
    shap_top28 = full_rank_order[:K_START].tolist()
    print(f"[load] SHAP top-{K_START} idx in 117-col matrix: "
          f"{shap_top28[:10]}...")

    # ---- load truth + anchor ----
    te = load_test()
    n_test = len(te)
    test_smiles = (
        te["smiles"].astype(str).tolist()
        if "smiles" in te.columns
        else te["SMILES"].astype(str).tolist()
    )
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [test_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique scaffolds in unb 253 = {n_unique_scaf}")

    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"chemprop_aux te missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} in_RAE = {rae_anchor:.4f}  (ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- rebuild 117-col matrix (same as nb2231 / nb2241) ----
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

    n_top_maccs = int(len(top_maccs_bit_idx))
    n_top_mord = int(len(top_mord_col_idx))
    n_top_ap = int(len(top_ap_bit_idx))
    n_top_embed = int(len(top_embed_col_idx))
    n_top_avalon = int(len(top_avalon_bit_idx))

    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test)
    X_ap_te_top = X_ap_te[:, top_ap_bit_idx].astype(np.float32)
    X_ap_unb_top = X_ap_te_top[unb_idx]
    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)
    X_maccs_te_top = X_maccs_te[:, top_maccs_bit_idx].astype(np.float32)
    X_maccs_unb_top = X_maccs_te_top[unb_idx]
    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    X_mord_te_top = X_mord_te[:, top_mord_col_idx].astype(np.float32)
    X_mord_unb_top = X_mord_te_top[unb_idx]
    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)
    X_emb_te_top = X_emb_te[:, top_embed_col_idx].astype(np.float32)
    X_emb_unb_top = X_emb_te_top[unb_idx]
    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)
    X_av_te_top = X_av_te[:, top_avalon_bit_idx].astype(np.float32)
    X_av_unb_top = X_av_te_top[unb_idx]

    # ChEMBL kNN
    pool = _load_chembl_pool()
    test_mols = [standardize(s) for s in test_smiles]
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
    pred_chembl_unb = pred_chembl_pec50[unb_idx].astype(np.float32)
    mean_sim_unb = mean_sim[unb_idx].astype(np.float32)

    X_unb = np.concatenate(
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
    feat_dim = X_unb.shape[1]
    expected_dim = (n_top_ap + n_top_maccs + n_top_mord
                    + n_top_embed + n_top_avalon + 2)
    assert feat_dim == expected_dim, f"feat_dim {feat_dim} != {expected_dim}"
    assert feat_dim == 117, f"feat_dim {feat_dim} != 117"
    print(f"\n   COMBINED 5-way matrix unb={X_unb.shape}  te={X_te_full.shape}")

    # feature names + family
    feat_names: list[str] = []
    feat_family: list[str] = []
    for b in top_ap_bit_idx:
        feat_names.append(f"AtomPair_bit_{int(b)}"); feat_family.append("AtomPair")
    for b in top_maccs_bit_idx:
        feat_names.append(f"MACCS_bit_{int(b)}"); feat_family.append("MACCS")
    for c in top_mord_col_idx:
        feat_names.append(f"Mordred_col_{int(c)}"); feat_family.append("Mordred")
    for d in top_embed_col_idx:
        feat_names.append(f"ChempropEmbed_dim_{int(d)}"); feat_family.append("ChempropEmbed")
    for b in top_avalon_bit_idx:
        feat_names.append(f"Avalon_bit_{int(b)}"); feat_family.append("Avalon")
    feat_names.append("pred_chembl_pec50"); feat_family.append("ChEMBL_kNN")
    feat_names.append("mean_sim"); feat_family.append("ChEMBL_kNN")

    # ---- K=28 baseline 30-seed evaluation ----
    print("\n" + "-" * 78)
    print(f"K={K_START} BASELINE  (30-seed mean / std / min / max)")
    print("-" * 78)
    t_b = time.time()
    base_mean, base_std, base_min, base_max, base_per_seed = _multi_seed_mean_rae(
        X_unb, shap_top28, residual, anchor, y_unb,
        unb_scaffolds, KF_SEEDS_FRESH,
    )
    print(f"   30-seed mean = {base_mean:.4f}  std = {base_std:.4f}  "
          f"min..max = [{base_min:.4f}, {base_max:.4f}]")
    print(f"   wall = {time.time()-t_b:.1f}s")

    # ---- Greedy backward RFE ----
    print("\n" + "-" * 78)
    print(f"LUCKY-AWARE RFE  K={K_START} -> K={K_MIN}  "
          f"(drop selection = 30-seed mean over {len(KF_SEEDS_FRESH)} seeds)")
    print("-" * 78)

    current = list(shap_top28)
    rfe_trajectory: list[dict] = [{
        "step": 0,
        "K_after": len(current),
        "feat_dropped": None,
        "feat_dropped_name": None,
        "feat_dropped_family": None,
        "rae_30seed_mean": float(base_mean),
        "rae_30seed_std": float(base_std),
        "rae_30seed_min": float(base_min),
        "rae_30seed_max": float(base_max),
    }]

    step = 0
    while len(current) > K_MIN:
        step += 1
        t_step = time.time()
        # for each candidate, evaluate 30-seed mean with that feature dropped
        cand_records = []
        for j in current:
            trial = [k for k in current if k != j]
            m, sd, mn, mx, _ = _multi_seed_mean_rae(
                X_unb, trial, residual, anchor, y_unb,
                unb_scaffolds, KF_SEEDS_FRESH,
            )
            cand_records.append((j, m, sd, mn, mx))
        cand_records.sort(key=lambda x: x[1])  # min 30-seed mean wins
        drop_j, drop_mean, drop_std, drop_min, drop_max = cand_records[0]
        current = [k for k in current if k != drop_j]
        rfe_trajectory.append({
            "step": step,
            "K_after": len(current),
            "feat_dropped": int(drop_j),
            "feat_dropped_name": feat_names[drop_j],
            "feat_dropped_family": feat_family[drop_j],
            "rae_30seed_mean": float(drop_mean),
            "rae_30seed_std": float(drop_std),
            "rae_30seed_min": float(drop_min),
            "rae_30seed_max": float(drop_max),
        })
        # also record top-3 also-rans for diagnostics
        top3_alsoran = [
            {
                "feat_idx": int(r[0]),
                "feat_name": feat_names[r[0]],
                "feat_family": feat_family[r[0]],
                "rae_30seed_mean": float(r[1]),
                "rae_30seed_std": float(r[2]),
            }
            for r in cand_records[1:4]
        ]
        rfe_trajectory[-1]["top3_alsoran"] = top3_alsoran
        print(f"   step {step:2d}  drop col={drop_j:3d} "
              f"({feat_names[drop_j]:28s} fam={feat_family[drop_j]:14s})  "
              f"K_after={len(current):2d}  30s_mean={drop_mean:.4f}  "
              f"std={drop_std:.4f}  wall_step={time.time()-t_step:.1f}s")

    # ---- find K_opt across trajectory (lucky-aware) ----
    K_to_traj = {e["K_after"]: e for e in rfe_trajectory}
    best_K_opt, best_rae_opt = K_START, base_mean
    for K, e in K_to_traj.items():
        if e["rae_30seed_mean"] < best_rae_opt:
            best_rae_opt = e["rae_30seed_mean"]
            best_K_opt = K

    # ---- print full trajectory table ----
    print("\n" + "=" * 78)
    print("LUCKY-AWARE RFE TRAJECTORY (30-seed mean RAE)")
    print("=" * 78)
    print(f"   {'K':>4s}  {'30s_mean':>10s}  {'30s_std':>8s}  "
          f"{'30s_min':>8s}  {'30s_max':>8s}  step")
    for e in rfe_trajectory:
        marker = "  <-- LUCKY-AWARE OPTIMUM" if e["K_after"] == best_K_opt else ""
        print(f"   {e['K_after']:>4d}  {e['rae_30seed_mean']:>10.4f}  "
              f"{e['rae_30seed_std']:>8.4f}  {e['rae_30seed_min']:>8.4f}  "
              f"{e['rae_30seed_max']:>8.4f}  step={e['step']:2d}"
              f"{marker}")

    # ---- compare against nb2231 single-seed K=20 ----
    nb2231_k20_obj = K_to_traj.get(20, None)
    nb2231_k20_rae_under_30seed = (
        nb2231_k20_obj["rae_30seed_mean"] if nb2231_k20_obj else None
    )
    print("\n" + "-" * 78)
    print("COMPARISON: nb2231 single-seed pick K=20  vs  lucky-aware optimum")
    print("-" * 78)
    print(f"   nb2231 single-seed K=20 reported per_seed_mean (5 seeds) = "
          f"0.5068")
    print(f"   nb2241 fresh-30  K=20 verified per_seed_mean (30 seeds) = "
          f"0.5173  (lucky-seed drift +0.0105)")
    if nb2231_k20_rae_under_30seed is not None:
        print(f"   nb2263 K=20 along lucky-aware path 30-seed mean        = "
              f"{nb2231_k20_rae_under_30seed:.4f}  "
              f"(may differ from nb2241 path since drop sequence differs)")
    print(f"   nb2263 lucky-aware optimum: K={best_K_opt}  "
          f"30-seed mean = {best_rae_opt:.4f}")
    if best_K_opt != 20:
        print(f"   --> lucky-aware path picks K={best_K_opt}, NOT K=20")
    else:
        print(f"   --> lucky-aware path STILL picks K=20 (K=20 robust)")

    # ---- pyramid-wrap test ----
    pyramid_result = None
    pyramid_decision = None
    if best_K_opt != 20:
        print("\n" + "=" * 78)
        print(f"PYRAMID-WRAP TEST  K_opt={best_K_opt}  "
              f"(K_opt != 20, must validate vs nb2240 K=20 deep-30 "
              f"{NB2240_K20_DEEP30_PYRAMID:.4f})")
        print("=" * 78)
        # walk the trajectory to recover K_opt surviving subset
        current2 = list(shap_top28)
        K_opt_cols = None
        for e in rfe_trajectory:
            if e["feat_dropped"] is None:
                if e["K_after"] == best_K_opt:
                    K_opt_cols = list(current2)
                    break
                continue
            current2.remove(int(e["feat_dropped"]))
            if e["K_after"] == best_K_opt:
                K_opt_cols = list(current2)
                break
        assert K_opt_cols is not None and len(K_opt_cols) == best_K_opt, (
            f"failed to recover K_opt={best_K_opt} subset from trajectory"
        )
        pyramid_result = pyramid_wrap_test(
            X_unb, X_te_full, K_opt_cols,
            anchor, te_anchor_513,
            residual, y_unb, n_test, n_unb,
            unb_scaffolds,
        )
        pyramid_decision = pyramid_result["verdict_vs_nb2240_K20"]
    else:
        pyramid_decision = "SKIPPED_K20_REMAINS_OPTIMAL"

    # ---- save summary ----
    summary = {
        "tag": TAG,
        "method": ("lucky_seed_aware_greedy_backward_RFE_drop_selection_by_"
                   "30_seed_mean_RAE_on_117col_matrix"),
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "K_start": K_START,
        "K_min": K_MIN,
        "n_folds": N_FOLDS,
        "kf_seeds_fresh": KF_SEEDS_FRESH,
        "n_kf_seeds_fresh": len(KF_SEEDS_FRESH),
        "kf_seeds_original_5seed_reference": KF_SEEDS_ORIGINAL,
        "feat_dim_full": int(feat_dim),
        "feat_breakdown_full": {
            "atompair": n_top_ap,
            "maccs": n_top_maccs,
            "mordred": n_top_mord,
            "chemprop_embed": n_top_embed,
            "avalon": n_top_avalon,
            "pred_chembl_pec50": 1,
            "mean_sim": 1,
            "total": int(feat_dim),
        },
        "n_chembl_pool": int(len(pool)),
        "n_unb": n_unb,
        "n_te": n_test,
        "unb_n_unique_scaffolds": n_unique_scaf,
        "rae_anchor_chemprop_aux": rae_anchor,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "lgbm_params_used": _lgbm_params(0),
        "shap_top28_idx_in_117": [int(j) for j in shap_top28],
        "K28_baseline": {
            "K": K_START,
            "rae_30seed_mean": base_mean,
            "rae_30seed_std": base_std,
            "rae_30seed_min": base_min,
            "rae_30seed_max": base_max,
            "per_seed_rae": base_per_seed,
        },
        "rfe_trajectory": rfe_trajectory,
        "K_opt_lucky_aware": int(best_K_opt),
        "rae_K_opt_lucky_aware": float(best_rae_opt),
        "K_opt_differs_from_nb2231_K20": bool(best_K_opt != 20),
        "delta_K_opt_vs_K28_baseline": float(best_rae_opt - base_mean),
        "nb2231_single_seed_K20_pick": 20,
        "nb2231_K20_claim_per_seed_mean_5seed": 0.5068,
        "nb2241_K20_verified_per_seed_mean_30seed": 0.5173,
        "nb2241_K20_verified_mean_bag_30seed": NB2241_K20_FRESH30_MEAN_BAG,
        "K20_along_lucky_aware_path_30seed_mean": (
            float(nb2231_k20_rae_under_30seed)
            if nb2231_k20_rae_under_30seed is not None
            else None
        ),
        "nb2240_K20_deep30_pyramid_ref": NB2240_K20_DEEP30_PYRAMID,
        "decision_margin": DECISION_MARGIN,
        "pyramid_wrap_test": pyramid_result,
        "pyramid_decision_vs_nb2240_K20": pyramid_decision,
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
    print("\n==== SUMMARY ====")
    for k in (
        "K_start",
        "K_min",
        "feat_dim_full",
        "rae_anchor_chemprop_aux",
        "K_opt_lucky_aware",
        "rae_K_opt_lucky_aware",
        "K_opt_differs_from_nb2231_K20",
        "K20_along_lucky_aware_path_30seed_mean",
        "delta_K_opt_vs_K28_baseline",
        "pyramid_decision_vs_nb2240_K20",
        "pre_unblind_clean",
    ):
        print(f"  {k}: {res.get(k)}")

    if res.get("pyramid_wrap_test"):
        p = res["pyramid_wrap_test"]
        print("\n==== PYRAMID WRAP TEST ====")
        for k in (
            "rae_K_mean_bag_standalone",
            "rae_K_per_seed_mean_standalone",
            "pyramid_deep30_mean",
            "pyramid_deep30_std",
            "delta_vs_nb2240_K20_pyramid",
            "verdict_vs_nb2240_K20",
        ):
            print(f"  {k}: {p.get(k)}")

    print("\n==== K-TRAJECTORY TABLE (30-seed mean) ====")
    for e in res["rfe_trajectory"]:
        marker = "  <-- OPTIMUM" if e["K_after"] == res["K_opt_lucky_aware"] else ""
        print(f"  K={e['K_after']:>3d}  "
              f"30s_mean={e['rae_30seed_mean']:.4f}  "
              f"std={e['rae_30seed_std']:.4f}  "
              f"drop={e.get('feat_dropped')}{marker}")
