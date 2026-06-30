"""nb2280 -- Morgan radius variants (ECFP6 r=3, ECFP10 r=5) SHAP-augmentation
of the K=20 RFE feature slice as chemprop_aux residual corrector, then SLSQP-
blend into the nb2240 pyramid.

HYPOTHESIS:
    The nb2240 K=20 RFE slice (Mordred 4 / ChempropEmbed 8 / AtomPair 4 /
    MACCS 1 / Avalon 2 / ChEMBL_kNN 1) gives 0.4598 pooled pyramid RAE but
    contains zero Morgan-circular features (AtomPair is path-based, not
    circular).  ECFP4 (r=2) was the workhorse fingerprint in nb02 / nb1382
    but SHAP-pruned ECFP4 mean-bag (nb1382) plateaus at 0.5527.  Deeper
    Morgan radii (r=3 = ECFP6, r=5 = ECFP10) capture wider substructural
    context closer to a 3D-pharmacophore (the user's framing) and may add
    orthogonal signal on novel scaffolds (the F2 failure tail, 90% rare).

PROTOCOL:
    1. Load K=20 RFE surviving feature indices from nb2231_summary.json
       (same as nb2240 / nb2270).  Rebuild the 117-col 5-way K-tuned feature
       matrix on the 513 test compounds, slice to K=20 indices.
    2. Build Morgan-2048 at radius=3 (ECFP6) on the 513 standardized test
       SMILES.  Build Morgan-2048 at radius=5 (ECFP10) too.  No persistent
       cache for these radii -- rebuild via morgan_fp_batch(radius=r,
       n_bits=2048).
    3. For each radius r in {3, 5}:
         a. Build 2050-col matrix [Morgan_r + pred_chembl + sim] on unb_idx
         b. Train ONE global LGBM-Huber shallow on residual (seed=0).
         c. SHAP TreeExplainer -> mean(|SHAP|) per feature -> take top-15
            Morgan bit indices (drop pred_chembl/sim since K=20 already has
            ChEMBL_kNN).
    4. Combine: K=50 = [K=20 RFE (20)] + [top-15 ECFP6 (15)] + [top-15
       ECFP10 (15)].  This is the AUGMENTED feature matrix.
    5. Fit LGBM K=50 on residual; 5-seed bag {0,1,7,42,137}, KFold(n=5)
       cross-fit per seed.  Compute mean-bag OOF (253) + mean-bag te (513).
    6. Compute SHAP on the K=50 residual model (seed=0 global fit) ->
       re-prune to top-20 features (let SHAP pick the best mix across the
       K=20 RFE + ECFP6-15 + ECFP10-15).  Build K=20-pruned matrix and
       re-fit (5-seed bag, KFold(n=5) cross-fit) for the RE-PRUNED anchor.
    7. SLSQP-blend each candidate (K=50 and K=20-RePruned) into the nb2240
       pyramid {nb2240_K20, chemprop_aux, nb1191, nb503, nb562} -- 5-anchor
       stack, drop-in replacement of the nb2240_K20 anchor.  5-fold scaffold
       CV across 5 seeds {1001..1005} with simplex SLSQP + rank-stretch grid
       1.000..1.150.
    8. Gate: must beat 0.4601 by >= 0.003 -> deploy threshold 0.4571.  Take
       strict gate = min(nb2240_pyramid 0.4598, task ref 0.4601) - 0.003 =
       0.4568 (report both).
    9. If gate beats: deep-30 verify (5 canonical + 25 extra seeds), write
       submissions/nb2280_morgan_radius.csv (deploy refit on all 253).

ANCHORS:
    chemprop_aux in_RAE = 0.6216
    nb2240 K=20 LGBM mean_bag RAE = 0.4630
    nb2240 pyramid pooled (5-seed mean) RAE = 0.4598
    nb1382 SHAP-pruned ECFP4 mean_bag RAE = 0.5527  (anchor was nb1070)

Outputs:
    scripts/nb2280_morgan_radius.py        (this file)
    data/processed/nb2280_summary.json
    data/processed/nb2280_mean_bag_oof_K50.npy        (253,) float32
    data/processed/nb2280_mean_bag_oof_K20pruned.npy  (253,) float32
    data/processed/te_nb2280_K50.npy                  (513,) float32
    data/processed/te_nb2280_K20pruned.npy            (513,) float32
    submissions/nb2280_morgan_radius.csv              (only if gate pass)
    data/processed/te_nb2280.npy                      (only if deploy)
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
from lightgbm import LGBMRegressor
from rdkit import Chem
from rdkit import RDLogger
from scipy.optimize import minimize
from sklearn.model_selection import KFold

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2280"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# ------------------------------ residual config ------------------------------
RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

# Morgan radii under test (existing r=2 is already covered by nb1382; here we
# add r=3 = ECFP6 and r=5 = ECFP10)
MORGAN_RADII = [3, 5]
MORGAN_NBITS = 2048
TOP_K_PER_RADIUS = 15        # SHAP top-15 per radius
K_RFE = 20                    # nb2231 K=20 RFE slice
K_COMBINED = K_RFE + TOP_K_PER_RADIUS * len(MORGAN_RADII)  # 20 + 30 = 50
K_REPRUNE = 20                # final re-prune

# LGBM residual hyperparams (lean shallow Huber -- same as nb2240/nb1382-like)
LGBM_RESID_PARAMS = dict(
    objective="huber",
    alpha=1.0,
    max_depth=4,
    num_leaves=15,
    n_estimators=300,
    learning_rate=0.03,
    min_child_samples=5,
    reg_lambda=2.0,
    verbosity=-1,
    n_jobs=2,
)

# SHAP global-fit (single seed=0)
SHAP_SEED = 0
LGBM_SHAP_PARAMS = dict(
    objective="huber",
    alpha=1.0,
    max_depth=3,
    num_leaves=7,
    n_estimators=80,
    learning_rate=0.05,
    min_child_samples=20,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    verbosity=-1,
    n_jobs=2,
    random_state=SHAP_SEED,
)

# ------------------------------ pyramid config -------------------------------
GATE_NB2240_PYRAMID = 0.4598   # nb2240 pooled_rae_mean_seeds (5 kf seeds)
GATE_TASK_REF = 0.4601         # task-wording "0.003 below 0.4601"
GATE_MARGIN = 0.003
GATE_THRESHOLD = min(GATE_NB2240_PYRAMID, GATE_TASK_REF) - GATE_MARGIN

N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]
LB_W_OOF = 0.51
LB_W_TE = 0.49

# nb1191 reconstruction parameters (copied from nb2240/nb2270)
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

CHEMPROP_AUX_REF = 0.6216
NB2240_K20_MEAN_BAG_REF = 0.4630
NB2240_PYRAMID_REF = 0.4598
NB1382_MORGAN_R2_REF = 0.5527

# ------------------------------ paths ----------------------------------------
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

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6


# ============================================================================
# helpers (copy-paste from nb2240/nb2270 -- keep identical so K=20 indices
# align byte-for-byte)
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


def _resid_lgbm_params(seed: int) -> dict:
    p = dict(LGBM_RESID_PARAMS)
    p["random_state"] = int(seed)
    return p


def _residual_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray,
                                 seed: int) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = LGBMRegressor(**_resid_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _train_full_then_predict_te(X_unb: np.ndarray, residual: np.ndarray,
                                X_te: np.ndarray, seed: int) -> np.ndarray:
    mdl = LGBMRegressor(**_resid_lgbm_params(seed))
    mdl.fit(X_unb, residual)
    return mdl.predict(X_te).astype(np.float32)


def _compute_shap_importance(X: np.ndarray, residual: np.ndarray):
    """Train one global LGBM-Huber on residual at seed=0; return mean(|SHAP|)
    per feature.  Falls back to LGBM gain on shap-import failure."""
    mdl = LGBMRegressor(**LGBM_SHAP_PARAMS)
    mdl.fit(X, residual)
    try:
        import shap
        explainer = shap.TreeExplainer(mdl)
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[0]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[..., 0]
        imp = np.abs(sv).mean(axis=0)
        return imp.astype(np.float64), "shap_tree_explainer"
    except Exception as e:
        print(f"   [shap] WARN: shap failed ({e}); falling back to LGBM gain")
        imp = mdl.booster_.feature_importance(importance_type="gain")
        return imp.astype(np.float64), "lgbm_gain_fallback"


# ============================================================================
# stage 2 utils (SLSQP + rank-stretch -- copy-paste from nb2240)
# ============================================================================

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


def deep_verify_seeds(P_unb, y_unb, unb_scaffolds, n_extra=25):
    extra_seeds = list(range(2001, 2001 + n_extra))
    seeds = KF_SEEDS + extra_seeds
    per = []
    for seed in seeds:
        pooled, _o, _w, fs = cv_run_for_seed(P_unb, y_unb, unb_scaffolds, seed)
        per.append({
            "kf_seed": int(seed),
            "pooled_rae": float(pooled),
            "mean_s": float(np.mean(fs)),
        })
    raes = np.asarray([r["pooled_rae"] for r in per])
    return {
        "n_seeds": int(len(seeds)),
        "per_seed": per,
        "mean_rae": float(raes.mean()),
        "std_rae": float(raes.std()),
        "min_rae": float(raes.min()),
        "max_rae": float(raes.max()),
    }


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
    chemprop_oof = np.load(
        DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy"
    ).astype(np.float64)
    nb1150_oof = reconstruct_nb1150_oof(n_unb)
    nb1158_oof = np.load(
        DATA_PROCESSED / "nb1158_mean_bag_oof_K32.npy"
    ).astype(np.float64)
    nb2112_oof = np.load(
        DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy"
    ).astype(np.float64)
    blend = (
        NB1191_DEPLOY_WEIGHTS["chemprop_aux"] * chemprop_oof
        + NB1191_DEPLOY_WEIGHTS["nb1150"]       * nb1150_oof
        + NB1191_DEPLOY_WEIGHTS["nb1158_K32"]   * nb1158_oof
        + NB1191_DEPLOY_WEIGHTS["nb2112_K28"]   * nb2112_oof
    )
    mu = float(blend.mean())
    return mu + NB1191_DEPLOY_S * (blend - mu)


# ============================================================================
# MAIN
# ============================================================================

def _bag_residual_train_predict(
    X_unb: np.ndarray, X_te: np.ndarray,
    residual: np.ndarray, anchor_unb: np.ndarray, anchor_te: np.ndarray,
    tag_for_print: str,
):
    """Run 5-seed bag: per seed KFold OOF on 253 + full-fit te(513) residual.
    Return mean-bag corrected OOF (253), mean-bag te (513), per-seed records."""
    n_unb = len(anchor_unb)
    n_te = len(anchor_te)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_te_resid = np.zeros((len(RESID_SEEDS), n_te), dtype=np.float64)
    records = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        resid_oof = _residual_cross_fit_one_seed(X_unb, residual, s)
        pred_corr = anchor_unb + resid_oof
        per_seed_corrected[i] = pred_corr
        rae_s = float(rae(np.asarray(anchor_unb) + residual, pred_corr))  # y = anchor + residual
        te_resid_s = _train_full_then_predict_te(X_unb, residual, X_te, s)
        per_seed_te_resid[i] = te_resid_s
        wall = time.time() - ts
        records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "resid_oof_std": float(resid_oof.std()),
            "resid_oof_mean": float(resid_oof.mean()),
            "wall_sec": round(wall, 2),
        })
        print(f"   [{tag_for_print}] seed={s:3d}: rae_corr={rae_s:.4f}  "
              f"resid_oof_std={resid_oof.std():.4f}  wall={wall:.1f}s")
    mean_bag_oof = per_seed_corrected.mean(axis=0)
    mean_bag_te_resid = per_seed_te_resid.mean(axis=0)
    mean_bag_te = anchor_te + mean_bag_te_resid
    rae_mean_bag = float(rae(np.asarray(anchor_unb) + residual, mean_bag_oof))
    rae_per_seed = float(np.mean([r["rae_corrected"] for r in records]))
    return mean_bag_oof, mean_bag_te, records, rae_mean_bag, rae_per_seed


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Morgan radius variants (ECFP6 r=3, ECFP10 r=5) augmenting "
          f"K=20 RFE; residual on chemprop_aux")
    print(f"          per_radius SHAP top-{TOP_K_PER_RADIUS} bits;  "
          f"K_combined = {K_RFE} + {TOP_K_PER_RADIUS}*{len(MORGAN_RADII)} = "
          f"{K_COMBINED}")
    print(f"          5-seed bag seeds={RESID_SEEDS}  folds={RESID_FOLDS}")
    print(f"          ref: nb2240 K=20 LGBM mean_bag {NB2240_K20_MEAN_BAG_REF:.4f}  "
          f"pyramid {NB2240_PYRAMID_REF:.4f}  gate_thresh {GATE_THRESHOLD:.4f}")
    print("=" * 78)

    # ---- Load K=20 surviving indices ----
    if not NB2231_SUMMARY.exists():
        raise FileNotFoundError(f"missing {NB2231_SUMMARY}")
    with open(NB2231_SUMMARY) as f:
        nb2231 = json.load(f)
    surviving_K20 = list(nb2231["snapshots"]["20"]["surviving_idx_in_117"])
    surviving_K20_names = list(nb2231["snapshots"]["20"]["surviving_names"])
    surviving_K20_family_counts = dict(nb2231["snapshots"]["20"]["family_counts"])
    assert len(surviving_K20) == K_RFE, (
        f"expected {K_RFE} features, got {len(surviving_K20)}"
    )
    print(f"[load] K=20 RFE surviving features family_counts = "
          f"{surviving_K20_family_counts}")

    # ---- Load truth + anchor ----
    te = load_test()
    n_test = len(te)
    te_smiles = te["smiles"].astype(str).tolist() if "smiles" in te.columns \
        else te["SMILES"].astype(str).tolist()
    if "Molecule Name" in te.columns:
        mol_names = te["Molecule Name"].astype(str).tolist()
    elif "name" in te.columns:
        mol_names = te["name"].astype(str).tolist()
    else:
        raise KeyError("no name column on test set")
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
    print(f"[load] chemprop_aux in_RAE = {rae_anchor:.4f} (ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f} std={residual.std():.4f}")

    # ---- Rebuild K=20 slice on 513 ----
    print("\n" + "-" * 78)
    print("REBUILD K=20 RFE FEATURE MATRIX  (same as nb2240/nb2270)")
    print("-" * 78)
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

    # ChEMBL kNN
    pool = _load_chembl_pool()
    test_mols = [standardize(s) for s in te_smiles]
    test_inchikeys = set()
    for m in test_mols:
        ik = _safe_inchikey(m)
        if ik is not None:
            test_inchikeys.add(ik)
    pool = pool[~pool["inchikey"].isin(test_inchikeys)].reset_index(drop=True)
    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())  # r=2 for kNN (matches nb2240)
    keep_pool = fp_pool.sum(axis=1) > 0
    if not keep_pool.all():
        pool = pool[keep_pool].reset_index(drop=True)
        fp_pool = fp_pool[keep_pool]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))
    std_test_smiles = [Chem.MolToSmiles(m) if m is not None else ""
                       for m in test_mols]
    fp_test_r2 = morgan_fp_batch(std_test_smiles)  # r=2 for kNN lookup
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test_r2, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )

    # Full 117-col matrix
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
    assert X_te_full.shape[1] == 117, f"feat_dim {X_te_full.shape[1]} != 117"

    # K=20 RFE slice
    X_te_K20 = X_te_full[:, surviving_K20].astype(np.float32)
    X_unb_K20 = X_te_K20[unb_idx]
    print(f"[feat] X_te_K20 = {X_te_K20.shape}  X_unb_K20 = {X_unb_K20.shape}")

    # ---- Build Morgan-2048 at radius=3 and radius=5 on test SMILES ----
    print("\n" + "-" * 78)
    print(f"BUILD MORGAN-{MORGAN_NBITS} AT RADII {MORGAN_RADII}  (no persistent cache)")
    print("-" * 78)
    morgan_te_by_r: dict[int, np.ndarray] = {}
    for r in MORGAN_RADII:
        ts = time.time()
        fp_r = morgan_fp_batch(std_test_smiles, radius=r, n_bits=MORGAN_NBITS)
        morgan_te_by_r[r] = fp_r.astype(np.float32)
        n_const = int((fp_r.var(axis=0) == 0).sum())
        print(f"   r={r}: shape={fp_r.shape}  density={fp_r.mean():.4f}  "
              f"const_cols={n_const}/{MORGAN_NBITS}  "
              f"wall={time.time()-ts:.1f}s")

    # ---- For each radius: SHAP top-15 bits on residual ----
    print("\n" + "-" * 78)
    print(f"SHAP TOP-{TOP_K_PER_RADIUS} BITS PER RADIUS  (1 global LGBM-Huber, "
          f"seed={SHAP_SEED})")
    print("-" * 78)
    top_bits_by_r: dict[int, np.ndarray] = {}
    shap_records = []
    for r in MORGAN_RADII:
        X_morgan_unb = morgan_te_by_r[r][unb_idx]
        pred_chembl_unb = pred_chembl_pec50[unb_idx].astype(np.float32)
        mean_sim_unb = mean_sim[unb_idx].astype(np.float32)
        X_full = np.concatenate(
            [
                X_morgan_unb,
                pred_chembl_unb.reshape(-1, 1),
                mean_sim_unb.reshape(-1, 1),
            ],
            axis=1,
        ).astype(np.float32)
        n_morgan_bits = int(X_morgan_unb.shape[1])
        imp, imp_src = _compute_shap_importance(X_full, residual)
        m_imp = imp[:n_morgan_bits]
        top_order = np.argsort(-m_imp)
        top_bits = top_order[:TOP_K_PER_RADIUS].astype(int)
        top_bits_imp = m_imp[top_bits]
        top_bits_by_r[r] = top_bits
        n_nonzero = int((m_imp > 0).sum())
        print(f"   r={r}  imp_src={imp_src}  morgan_nonzero_bits={n_nonzero}/"
              f"{n_morgan_bits}  pred_chembl_imp={imp[n_morgan_bits]:.5f}  "
              f"sim_imp={imp[n_morgan_bits+1]:.5f}")
        print(f"        top-{TOP_K_PER_RADIUS} bits = "
              f"{top_bits.tolist()}")
        print(f"        top-{TOP_K_PER_RADIUS} imp  = "
              f"{[round(float(v), 5) for v in top_bits_imp.tolist()]}")
        shap_records.append({
            "radius": int(r),
            "imp_source": imp_src,
            "morgan_n_bits": n_morgan_bits,
            "morgan_nonzero_imp_bits": n_nonzero,
            "top_bits": [int(b) for b in top_bits.tolist()],
            "top_bits_imp": [float(v) for v in top_bits_imp.tolist()],
            "pred_chembl_imp": float(imp[n_morgan_bits]),
            "sim_imp": float(imp[n_morgan_bits + 1]),
        })

    # ---- Build K=50 combined feature matrix ----
    print("\n" + "-" * 78)
    print(f"COMBINED K={K_COMBINED} FEATURE MATRIX = K={K_RFE} RFE + per-radius "
          f"top-{TOP_K_PER_RADIUS} bits")
    print("-" * 78)
    morgan_blocks_te = []
    morgan_blocks_unb = []
    feature_names_K50 = list(surviving_K20_names)  # start with K=20 RFE names
    feature_idx_K50_src = [("RFE_K20", j, surviving_K20[j])
                           for j in range(K_RFE)]
    for r in MORGAN_RADII:
        bits = top_bits_by_r[r]
        block_te = morgan_te_by_r[r][:, bits].astype(np.float32)
        block_unb = block_te[unb_idx]
        morgan_blocks_te.append(block_te)
        morgan_blocks_unb.append(block_unb)
        for bit_i, b in enumerate(bits.tolist()):
            feature_names_K50.append(f"ECFP{2*r}_r{r}_bit{int(b)}")
            feature_idx_K50_src.append((f"ECFP{2*r}_r{r}", bit_i, int(b)))
    X_te_K50 = np.concatenate([X_te_K20] + morgan_blocks_te, axis=1).astype(np.float32)
    X_unb_K50 = X_te_K50[unb_idx]
    assert X_te_K50.shape[1] == K_COMBINED, (
        f"K_combined {X_te_K50.shape[1]} != {K_COMBINED}"
    )
    print(f"[feat] X_te_K50 = {X_te_K50.shape}  X_unb_K50 = {X_unb_K50.shape}")

    # ---- Fit LGBM K=50 (5-seed bag, scaffold-CV on residual) ----
    print("\n" + "=" * 78)
    print(f"LGBM K={K_COMBINED}  5-seed bag {RESID_SEEDS}  KFold(n={RESID_FOLDS})")
    print("=" * 78)
    mean_bag_oof_K50, mean_bag_te_K50, records_K50, rae_K50_mean_bag, \
        rae_K50_per_seed_mean = _bag_residual_train_predict(
            X_unb_K50, X_te_K50, residual, anchor, te_anchor_513, f"K{K_COMBINED}"
        )
    delta_K50_vs_lgbm_K20 = rae_K50_mean_bag - NB2240_K20_MEAN_BAG_REF
    print(f"\n[K{K_COMBINED}] mean-bag RAE      = {rae_K50_mean_bag:.4f}  "
          f"d_vs_LGBM_K20 = {delta_K50_vs_lgbm_K20:+.4f}")
    print(f"[K{K_COMBINED}] per-seed mean RAE = {rae_K50_per_seed_mean:.4f}")

    oof_K50_path = DATA_PROCESSED / f"{TAG}_mean_bag_oof_K{K_COMBINED}.npy"
    te_K50_path = DATA_PROCESSED / f"te_{TAG}_K{K_COMBINED}.npy"
    np.save(oof_K50_path, mean_bag_oof_K50.astype(np.float32))
    np.save(te_K50_path, mean_bag_te_K50.astype(np.float32))
    print(f"[save] {oof_K50_path}")
    print(f"[save] {te_K50_path}")

    # ---- SHAP re-prune K=50 -> K=20 ----
    print("\n" + "-" * 78)
    print(f"SHAP RE-PRUNE K={K_COMBINED} -> K={K_REPRUNE}  (let SHAP pick best mix)")
    print("-" * 78)
    imp_K50, imp_src_K50 = _compute_shap_importance(X_unb_K50, residual)
    order_K50 = np.argsort(-imp_K50)
    repruned_idx = np.sort(order_K50[:K_REPRUNE]).astype(int)
    print(f"   imp_src={imp_src_K50}  top-K positions (unsorted) = "
          f"{order_K50[:K_REPRUNE].tolist()}")
    reprune_family_counts = {}
    repruned_names = []
    for j in repruned_idx.tolist():
        src = feature_idx_K50_src[j][0]
        reprune_family_counts[src] = reprune_family_counts.get(src, 0) + 1
        repruned_names.append(feature_names_K50[j])
    print(f"   family_counts after re-prune = {reprune_family_counts}")
    print(f"   first 10 re-pruned features  = {repruned_names[:10]}")

    X_te_K20pruned = X_te_K50[:, repruned_idx].astype(np.float32)
    X_unb_K20pruned = X_te_K20pruned[unb_idx]
    print(f"   X_te_K20pruned = {X_te_K20pruned.shape}  "
          f"X_unb_K20pruned = {X_unb_K20pruned.shape}")

    print(f"\n[refit] LGBM K={K_REPRUNE} on re-pruned slice  5-seed bag")
    mean_bag_oof_K20pruned, mean_bag_te_K20pruned, records_K20pruned, \
        rae_K20pruned_mean_bag, rae_K20pruned_per_seed_mean = \
        _bag_residual_train_predict(
            X_unb_K20pruned, X_te_K20pruned, residual, anchor, te_anchor_513,
            f"K{K_REPRUNE}pruned"
        )
    delta_K20pruned_vs_lgbm_K20 = rae_K20pruned_mean_bag - NB2240_K20_MEAN_BAG_REF
    print(f"\n[K{K_REPRUNE}pruned] mean-bag RAE      = {rae_K20pruned_mean_bag:.4f}  "
          f"d_vs_LGBM_K20 = {delta_K20pruned_vs_lgbm_K20:+.4f}")
    print(f"[K{K_REPRUNE}pruned] per-seed mean RAE = {rae_K20pruned_per_seed_mean:.4f}")

    oof_K20p_path = DATA_PROCESSED / f"{TAG}_mean_bag_oof_K{K_REPRUNE}pruned.npy"
    te_K20p_path = DATA_PROCESSED / f"te_{TAG}_K{K_REPRUNE}pruned.npy"
    np.save(oof_K20p_path, mean_bag_oof_K20pruned.astype(np.float32))
    np.save(te_K20p_path, mean_bag_te_K20pruned.astype(np.float32))
    print(f"[save] {oof_K20p_path}")
    print(f"[save] {te_K20p_path}")

    # ---- Pick better of (K=50, K=20-RePruned) for pyramid ----
    if rae_K20pruned_mean_bag < rae_K50_mean_bag:
        primary_tag = f"K{K_REPRUNE}pruned"
        primary_oof = mean_bag_oof_K20pruned
        primary_te = mean_bag_te_K20pruned
        primary_rae = rae_K20pruned_mean_bag
    else:
        primary_tag = f"K{K_COMBINED}"
        primary_oof = mean_bag_oof_K50
        primary_te = mean_bag_te_K50
        primary_rae = rae_K50_mean_bag
    print(f"\n[choose] primary anchor for pyramid = nb2280_{primary_tag}  "
          f"(RAE {primary_rae:.4f})")

    # ============================================================================
    # Stage 2: 5-anchor pyramid SLSQP (nb2280_<primary> swaps in for nb2240_K20)
    # ============================================================================
    print("\n" + "=" * 78)
    print(f"STAGE 2: 5-ANCHOR PYRAMID  (nb2280_{primary_tag} swaps in for nb2240_K20)")
    print("=" * 78)
    nb1191_oof = reconstruct_nb1191_oof(n_unb)
    chemprop_oof = np.load(DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy").astype(np.float64)
    nb503_oof = np.load(DATA_PROCESSED / "nb503_pred_oof.npy").astype(np.float64)
    nb562_oof = np.load(DATA_PROCESSED / "nb562_pred_oof.npy").astype(np.float64)

    te_nb1191 = np.load(DATA_PROCESSED / "te_nb1191.npy").astype(np.float64)
    te_nb503 = np.load(DATA_PROCESSED / "te_nb503.npy").astype(np.float64)
    te_nb562 = np.load(DATA_PROCESSED / "te_nb562.npy").astype(np.float64)
    te_chemprop_aux = te_anchor_513

    anchors_list = [
        (f"nb2280_{primary_tag}", primary_oof.astype(np.float64),
         primary_te.astype(np.float64)),
        ("chemprop_aux",   chemprop_oof,                         te_chemprop_aux),
        ("nb1191",         nb1191_oof,                           te_nb1191),
        ("nb503",          nb503_oof,                            te_nb503),
        ("nb562",          nb562_oof,                            te_nb562),
    ]
    indiv_rae = {}
    oof_cols, te_cols = [], []
    print("\n[anchors]")
    for disp, oof, te_arr in anchors_list:
        assert oof.shape == (n_unb,), f"{disp} oof {oof.shape}"
        assert te_arr.shape == (n_test,), f"{disp} te {te_arr.shape}"
        r_ = float(rae(y_unb, oof))
        indiv_rae[disp] = r_
        oof_cols.append(oof)
        te_cols.append(te_arr)
        print(f"   {disp:24s} oof_RAE={r_:.4f}  te_mean={te_arr.mean():.3f}  "
              f"te_std={te_arr.std():.3f}")
    P_unb = np.column_stack(oof_cols)
    P_te = np.column_stack(te_cols)
    print(f"[stack] P_unb {P_unb.shape}  P_te {P_te.shape}  K={P_unb.shape[1]}")

    # ---- Scaffold 5-fold CV across 5 seeds ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV  kf_seeds={KF_SEEDS}  stretch_grid={STRETCH_GRID}")
    print("-" * 78)
    per_seed = []
    all_oofs = []
    for kf_seed in KF_SEEDS:
        pooled, oof_blend, fw, fs = cv_run_for_seed(
            P_unb, y_unb, unb_scaffolds, kf_seed
        )
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": float(pooled),
            "fold_s": [float(x) for x in fs],
            "fold_w_mean": [float(x) for x in np.mean(fw, axis=0)],
        })
        all_oofs.append(oof_blend)
        print(f"   seed={kf_seed}  pooled_RAE={pooled:.4f}  "
              f"mean_s={np.mean(fs):.3f}  "
              f"w_mean={np.round(np.mean(fw, axis=0), 3).tolist()}")
    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    final_oof_rae = float(rae(y_unb, mean_oof))
    pooled_rae_mean_seeds = float(np.mean([r["pooled_rae"] for r in per_seed]))
    pooled_rae_std_seeds = float(np.std([r["pooled_rae"] for r in per_seed]))
    print(f"\n[cv] mean across 5 seeds  pooled_RAE = {pooled_rae_mean_seeds:.4f} "
          f"(+/- {pooled_rae_std_seeds:.4f})")
    print(f"[cv] RAE of mean-of-seed OOFs        = {final_oof_rae:.4f}")

    # ---- Deploy ----
    print("\n" + "-" * 78)
    print("DEPLOY (refit weights on 253; mean(fold_s) across all 5 seeds)")
    print("-" * 78)
    w_deploy = slsqp_simplex(P_unb, y_unb)
    blend_unb = P_unb @ w_deploy
    mu_deploy = float(blend_unb.mean())
    s_deploy = float(np.mean([s for r in per_seed for s in r["fold_s"]]))
    in_rae_final = float(rae(y_unb, mu_deploy + s_deploy * (blend_unb - mu_deploy)))
    blend_te = P_te @ w_deploy
    deploy_te = (mu_deploy + s_deploy * (blend_te - mu_deploy)).astype(np.float32)
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))
    w_str = ", ".join(f"{disp}={w:.4f}"
                      for (disp, _, _), w in zip(anchors_list, w_deploy))
    print(f"   deploy weights      = {w_str}")
    print(f"   deploy mu / s       = {mu_deploy:.4f} / {s_deploy:.4f}")
    print(f"   in-sample RAE (253) = {in_rae_final:.4f}")
    print(f"   te[unb_idx] RAE     = {te_unb_rae:.4f}")
    print(f"   te(513) mean/std    = {deploy_te.mean():.3f}/{deploy_te.std():.3f}")

    lb_band_est = LB_W_OOF * pooled_rae_mean_seeds + LB_W_TE * te_unb_rae
    lb_low = lb_band_est - 0.05
    lb_high = lb_band_est + 0.05
    print(f"\n[LB-band] {LB_W_OOF:.2f}*OOF + {LB_W_TE:.2f}*te_unb = {lb_band_est:.4f}  "
          f"[{lb_low:.4f}, {lb_high:.4f}]")

    # ---- Gate ----
    delta_vs_nb2240_pyr = pooled_rae_mean_seeds - NB2240_PYRAMID_REF
    delta_vs_task_ref = pooled_rae_mean_seeds - GATE_TASK_REF
    gate_beat = pooled_rae_mean_seeds < GATE_THRESHOLD
    gate_flat = abs(delta_vs_nb2240_pyr) <= GATE_MARGIN
    print("\n" + "-" * 78)
    print(f"GATE EVALUATION  (vs nb2240 pyramid {NB2240_PYRAMID_REF} + task ref "
          f"{GATE_TASK_REF}, margin {GATE_MARGIN})")
    print("-" * 78)
    print(f"   nb2280 OOF (5-seed mean)  = {pooled_rae_mean_seeds:.4f}")
    print(f"   nb2240 pyramid reference  = {NB2240_PYRAMID_REF:.4f}  "
          f"delta = {delta_vs_nb2240_pyr:+.4f}")
    print(f"   task ref                  = {GATE_TASK_REF:.4f}  "
          f"delta = {delta_vs_task_ref:+.4f}")
    print(f"   strict gate threshold     = {GATE_THRESHOLD:.4f}")
    if gate_beat:
        verdict = "BEATS_GATE_DEPLOY"
    elif gate_flat:
        verdict = "FLAT_VS_NB2240_PYRAMID"
    else:
        verdict = "HURTS_VS_NB2240_PYRAMID"
    print(f"   verdict                   = {verdict}")

    # ---- Deep-30 verify on gate-pass ----
    deep30 = None
    if gate_beat:
        print("\n" + "-" * 78)
        print("DEEP-30 VERIFY  (5 canonical + 25 extra seeds)")
        print("-" * 78)
        deep30 = deep_verify_seeds(P_unb, y_unb, unb_scaffolds, n_extra=25)
        print(f"   n_seeds={deep30['n_seeds']}  mean_RAE={deep30['mean_rae']:.4f}  "
              f"std={deep30['std_rae']:.4f}  "
              f"range=[{deep30['min_rae']:.4f}, {deep30['max_rae']:.4f}]")

    # ---- Save te artefact + (optional) submission CSV ----
    submission_csv = None
    te_npy_path = None
    if gate_beat:
        te_npy_path = DATA_PROCESSED / f"te_{TAG}.npy"
        np.save(te_npy_path, deploy_te)
        print(f"\n[save] {te_npy_path}")
        sub_csv_path = SUBMISSIONS / f"{TAG}_morgan_radius.csv"
        df_sub = pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": mol_names,
            "pEC50": deploy_te,
        })
        if len(df_sub) != 513:
            raise ValueError(f"submission rows {len(df_sub)} != 513")
        df_sub.to_csv(sub_csv_path, index=False)
        print(f"[save] {sub_csv_path}  (gate BEATS, {len(df_sub)} rows)")
        submission_csv = str(sub_csv_path)
    else:
        print(f"\n[skip] gate not beat -- no te_nb2280.npy / submission CSV "
              f"written ({verdict})")

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "method": ("morgan_radius_variants_r3_r5_SHAP_top15_combine_RFE_K20_"
                   "then_re-prune_to_K20_LGBM_5seed_bag_with_nb2240_pyramid_slsqp"),
        "anchor": ANCHOR,
        "anchor_path": str(ANCHOR_TE_PATH),
        "data_source": ("nb2231 K=20 RFE 117-col 5-way features + Morgan-2048 "
                        "at radii {3, 5} SHAP top-15 each"),
        "morgan_radii": MORGAN_RADII,
        "morgan_n_bits": MORGAN_NBITS,
        "top_k_per_radius": TOP_K_PER_RADIUS,
        "K_rfe": K_RFE,
        "K_combined": K_COMBINED,
        "K_reprune": K_REPRUNE,
        "K20_rfe_family_counts": surviving_K20_family_counts,
        "K20_rfe_surviving_idx_in_117": [int(j) for j in surviving_K20],
        "K20_rfe_surviving_names": surviving_K20_names,
        "shap_per_radius_records": shap_records,
        "reprune_family_counts": reprune_family_counts,
        "reprune_feature_names": repruned_names,
        "reprune_indices_in_K50": [int(j) for j in repruned_idx.tolist()],
        "shap_K50_imp_source": imp_src_K50,
        "model_family": "LightGBM_Huber",
        "lgbm_resid_params": LGBM_RESID_PARAMS,
        "lgbm_shap_params": LGBM_SHAP_PARAMS,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "n_test": n_test,
        "n_unb": n_unb,
        "n_unique_scaffolds": n_unique_scaf,
        "rae_anchor_chemprop_aux": rae_anchor,
        "rae_chemprop_aux_ref": CHEMPROP_AUX_REF,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        # K=50 standalone
        "K50_per_seed_records": records_K50,
        "rae_K50_mean_bag": rae_K50_mean_bag,
        "rae_K50_per_seed_mean": rae_K50_per_seed_mean,
        "delta_K50_vs_lgbm_K20_ref": delta_K50_vs_lgbm_K20,
        "oof_K50_path": str(oof_K50_path),
        "te_K50_path": str(te_K50_path),
        # K=20-RePruned standalone
        "K20pruned_per_seed_records": records_K20pruned,
        "rae_K20pruned_mean_bag": rae_K20pruned_mean_bag,
        "rae_K20pruned_per_seed_mean": rae_K20pruned_per_seed_mean,
        "delta_K20pruned_vs_lgbm_K20_ref": delta_K20pruned_vs_lgbm_K20,
        "oof_K20pruned_path": str(oof_K20p_path),
        "te_K20pruned_path": str(te_K20p_path),
        # Primary chosen
        "primary_tag": primary_tag,
        "primary_oof_rae_mean_bag": primary_rae,
        # Pyramid
        "pyramid_anchors": [a[0] for a in anchors_list],
        "pyramid_anchor_oof_rae": indiv_rae,
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "stretch_grid": STRETCH_GRID,
        "per_seed_results": per_seed,
        "pooled_rae_mean_seeds": pooled_rae_mean_seeds,
        "pooled_rae_std_seeds": pooled_rae_std_seeds,
        "rae_of_mean_of_seed_oofs": final_oof_rae,
        "deploy_weights": [
            {"name": disp, "w": float(w)}
            for (disp, _, _), w in zip(anchors_list, w_deploy)
        ],
        "deploy_mu_blend": mu_deploy,
        "deploy_s": s_deploy,
        "in_sample_rae_overfit_bound": in_rae_final,
        "te_unb_rae_in_sample": te_unb_rae,
        "lb_band_estimate": lb_band_est,
        "lb_band_low": lb_low,
        "lb_band_high": lb_high,
        "lb_band_w_oof": LB_W_OOF,
        "lb_band_w_te": LB_W_TE,
        "nb2240_K20_mean_bag_ref": NB2240_K20_MEAN_BAG_REF,
        "nb2240_pyramid_ref": NB2240_PYRAMID_REF,
        "nb1382_morgan_r2_ref": NB1382_MORGAN_R2_REF,
        "compare_nb2240_pyramid_ref": NB2240_PYRAMID_REF,
        "delta_vs_nb2240_pyramid": delta_vs_nb2240_pyr,
        "compare_task_ref": GATE_TASK_REF,
        "delta_vs_task_ref": delta_vs_task_ref,
        "gate_margin": GATE_MARGIN,
        "gate_threshold_strict": GATE_THRESHOLD,
        "gate_beat": bool(gate_beat),
        "gate_flat": bool(gate_flat),
        "verdict": verdict,
        "deep_30_verify": deep30,
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "te_npy_path": str(te_npy_path) if te_npy_path is not None else None,
        "submission_csv": submission_csv,
        "pre_unblind_clean": True,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   K={K_COMBINED} mean-bag RAE                = {rae_K50_mean_bag:.4f}  "
          f"(d_vs_LGBM_K20 {delta_K50_vs_lgbm_K20:+.4f})")
    print(f"   K={K_REPRUNE}-RePruned mean-bag RAE        = {rae_K20pruned_mean_bag:.4f}  "
          f"(d_vs_LGBM_K20 {delta_K20pruned_vs_lgbm_K20:+.4f})")
    print(f"   primary anchor chosen           = nb2280_{primary_tag}")
    print(f"   pyramid pooled RAE (5 seeds)   = {pooled_rae_mean_seeds:.4f}")
    print(f"   delta vs nb2240 pyramid {NB2240_PYRAMID_REF:.4f} = "
          f"{delta_vs_nb2240_pyr:+.4f}")
    print(f"   delta vs task ref {GATE_TASK_REF:.4f}      = {delta_vs_task_ref:+.4f}")
    print(f"   gate threshold strict          = {GATE_THRESHOLD:.4f}")
    print(f"   verdict                        = {verdict}")
    print(f"   LB band                        = {lb_band_est:.4f}")
    print(f"   wall                           = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "rae_anchor_chemprop_aux",
        "rae_K50_mean_bag",
        "delta_K50_vs_lgbm_K20_ref",
        "rae_K20pruned_mean_bag",
        "delta_K20pruned_vs_lgbm_K20_ref",
        "primary_tag",
        "pooled_rae_mean_seeds",
        "delta_vs_nb2240_pyramid",
        "delta_vs_task_ref",
        "gate_threshold_strict",
        "gate_beat",
        "verdict",
        "deploy_weights",
        "deploy_s",
        "lb_band_estimate",
        "submission_csv",
        "te_npy_path",
        "wall_sec",
    ):
        print(f"  {k}: {res.get(k)}")
