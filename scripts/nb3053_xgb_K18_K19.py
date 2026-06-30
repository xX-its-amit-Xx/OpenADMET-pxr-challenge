"""nb3053 -- XGBoost residual on K=18 and K=19 features, blended with LGBM versions.

NEW PARADIGM:
    Cross-paradigm orthogonality at the LEARNER level.
    Prior ceiling (nb2171/nb2095/nb1191 ~ 0.4720, nb3030 0.4509, nb3033 log-blend)
    has been built ENTIRELY on LightGBM axis-aligned trees on
    chemprop_aux + K-feature-slice residuals. Per cycle-134 paradigm-exhaustion
    note ("the bottleneck is the (anchor, feature_set, n=253) tuple capacity, NOT
    the model class") the within-LGBM space appears exhausted.

    nb3053 introduces XGBoost (level-wise depth=4 binary trees w/ L2 reg) on the
    SAME K=18 and K=19 slices, then blends {K18_LGBM, K19_LGBM, K18_XGB, K19_XGB}
    via per-fold SLSQP simplex. The two GBM families share inductive bias
    (axis-aligned, gradient-boosted) but differ in:
        - split direction (LGBM leaf-wise vs XGB level-wise)
        - regularization (LGBM min_child_samples vs XGB reg_lambda)
        - histogram binning
    Orthogonality is small but real on these residual slices.

PROTOCOL:
    Anchors (cached deep-30):
        K18_LGBM: nb2960_K18_30seed_oof.npy + nb2960_K18_30seed_te.npy   (0.4536)
        K19_LGBM: nb3000_K19_30seed_oof.npy + te_nb3000_K19.npy          (0.4607)

    New (built fresh in this script):
        K18_XGB:  XGBoost 30-seed bag-mean on K=18 slice
        K19_XGB:  XGBoost 30-seed bag-mean on K=19 slice

    XGBoost params:
        max_depth=4, n_estimators=300, learning_rate=0.03, reg_lambda=2.0
        objective='reg:squarederror', tree_method='hist', n_jobs=2.

    Per (kf_seed, fold):
        SLSQP simplex (w_K18L, w_K19L, w_K18X, w_K19X) >= 0 sum 1 on fold-train,
        applied to fold-val.
    Outer CV: 5-fold scaffold split, 15 fresh kf_seeds {1081..1095}.
    Per-seed: pooled outer-val RAE over all 253.
    Reported gate metric = MEAN pooled RAE across the 15 seeds.

    Deploy:
        SLSQP simplex on FULL 253 -> single global w*
        Apply to (513, 4) te column-stack -> te_nb3053

GATE:
    mean_pooled_rae < 0.4509 -> "BETTER_THAN_NB3030"
    else                     -> "FAIL"

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/te_chemprop_aux.npy
    data/processed/nb2960_summary.json       (K18 idx)
    data/processed/nb2231_summary.json       (K19 idx reconstruction)
    data/processed/nb2063_shap_importance_full117.npy
    data/processed/nb2960_K18_30seed_oof.npy
    data/processed/nb2960_K18_30seed_te.npy
    data/processed/nb3000_K19_30seed_oof.npy
    data/processed/te_nb3000_K19.npy
    data/processed/nb1352_summary.json + cached te_*.npy fingerprint matrices

Outputs:
    data/processed/nb3053_summary.json
    data/processed/nb3053_pred_oof.npy     (253,) float32 -- median-seed OOF blend
    data/processed/te_nb3053.npy           (513,) float32 -- deploy full-pool te
    data/processed/nb3053_K18_XGB_oof.npy  (253,) float32
    data/processed/nb3053_K18_XGB_te.npy   (513,) float32
    data/processed/nb3053_K19_XGB_oof.npy  (253,) float32
    data/processed/nb3053_K19_XGB_te.npy   (513,) float32
    submissions/nb3053_xgb_K18_K19.csv     (only on BETTER verdict)
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
import xgboost as xgb
from rdkit import Chem
from rdkit import RDLogger
from scipy.optimize import minimize
from sklearn.model_selection import KFold

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3053"
PARENT_TAG = "nb2960+nb3000"

# -- Anchors / cached deep-30 OOFs --------------------------------------------
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
K18_LGBM_OOF = DATA_PROCESSED / "nb2960_K18_30seed_oof.npy"
K18_LGBM_TE = DATA_PROCESSED / "nb2960_K18_30seed_te.npy"
K19_LGBM_OOF = DATA_PROCESSED / "nb3000_K19_30seed_oof.npy"
K19_LGBM_TE = DATA_PROCESSED / "te_nb3000_K19.npy"

# -- XGBoost build params -----------------------------------------------------
RESID_FOLDS = 5
RESID_SEEDS_DEEP = list(range(3001, 3031))   # 30 fresh seeds (mirror nb2960/nb3000)

# -- Feature cache paths (same as nb2960/nb3000) ------------------------------
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
NB2960_SUMMARY = DATA_PROCESSED / "nb2960_summary.json"

# -- ChEMBL kNN params (identical to nb2960/nb3000) ---------------------------
KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

# -- Outer CV for blend evaluation --------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1081, 1096))    # 15 fresh seeds {1081..1095}
N_STARTS_FOLD = 8
N_STARTS_FULL = 12
DEGEN_MAX_W = 0.85

# -- Gates --------------------------------------------------------------------
GATE_BETTER = 0.4509

# -- References ---------------------------------------------------------------
REF_NB3030 = 0.4509       # 3-anchor LGBM-only ceiling (15-seed wide)
REF_NB3001 = 0.4511
REF_NB2992 = 0.4479
REF_K18_LGBM = 0.4536
REF_K19_LGBM = 0.4607
REF_NB2171 = 0.4682
CHEMPROP_AUX_REF = 0.6216


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


def _xgb_params(seed):
    return dict(
        objective="reg:squarederror",
        max_depth=4,
        n_estimators=300,
        learning_rate=0.03,
        reg_lambda=2.0,
        reg_alpha=0.0,
        subsample=1.0,
        colsample_bytree=1.0,
        tree_method="hist",
        random_state=seed,
        n_jobs=2,
        verbosity=0,
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


def _residual_xgb_cross_fit_one_seed(X, residual, seed):
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = xgb.XGBRegressor(**_xgb_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _train_xgb_full_then_predict_te(X_unb, residual, X_te, seed):
    mdl = xgb.XGBRegressor(**_xgb_params(seed))
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


def build_K_xgb_30seed_bag(K_label, K_idx, X_te_full, unb_idx, anchor,
                           residual, te_anchor_513, n_test, n_unb, seeds):
    """XGBoost deep-30 bag-mean OOF (253,) + te (513,) for one K-pyramid."""
    X_te_K = X_te_full[:, K_idx].astype(np.float32)
    X_unb_K = X_te_K[unb_idx]
    print(f"   [{K_label}-XGB] X_unb_K = {X_unb_K.shape}  X_te_K = {X_te_K.shape}")
    sum_unb = np.zeros(n_unb, dtype=np.float64)
    sum_te = np.zeros(n_test, dtype=np.float64)
    per_seed_rae = []
    for i, s in enumerate(seeds):
        ts = time.time()
        resid_oof = _residual_xgb_cross_fit_one_seed(X_unb_K, residual, s)
        pred_unb_s = anchor + resid_oof
        sum_unb += pred_unb_s
        per_seed_rae.append(float(rae(anchor + residual, pred_unb_s)))
        te_resid_s = _train_xgb_full_then_predict_te(X_unb_K, residual, X_te_K, s)
        pred_te_s = te_anchor_513 + te_resid_s
        sum_te += pred_te_s
        if (i % 10) == 0 or i == len(seeds) - 1:
            print(f"      [{K_label}-XGB] seed={s:4d}  "
                  f"rae={per_seed_rae[-1]:.4f}  "
                  f"wall={time.time()-ts:.2f}s  ({i+1}/{len(seeds)})")
    bag_oof_unb = sum_unb / len(seeds)
    bag_te_513 = sum_te / len(seeds)
    return bag_oof_unb, bag_te_513, per_seed_rae


def _simplex_slsqp(P: np.ndarray, y: np.ndarray, n_starts: int = 8,
                   seed: int = 0):
    """Minimize RAE(y, P @ w) over simplex (w>=0, sum w=1) with multi-start."""
    K = P.shape[1]
    rng = np.random.default_rng(seed)

    def loss(w):
        return float(rae(y, P @ w))

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bnds = [(0.0, 1.0)] * K

    starts = [np.full(K, 1.0 / K)]
    for _ in range(max(0, n_starts - 1)):
        starts.append(rng.dirichlet(np.ones(K)))

    best_w, best_r = None, np.inf
    for x0 in starts:
        try:
            res = minimize(loss, x0, method="SLSQP", bounds=bnds, constraints=cons,
                           options={"maxiter": 300, "ftol": 1e-9})
            w = np.clip(res.x, 0.0, 1.0)
            s = float(w.sum())
            if s <= 0.0:
                continue
            w = w / s
            r = float(rae(y, P @ w))
            if r < best_r:
                best_r, best_w = r, w
        except Exception:
            continue
    if best_w is None:
        best_w = np.full(K, 1.0 / K)
        best_r = float(rae(y, P @ best_w))
    return best_w, best_r


def _run_one_seed(P_unb, y_unb, unb_scaffolds, kf_seed, labels):
    """Per-fold SLSQP simplex over the 4 anchors at one kf_seed."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    K = P_unb.shape[1]
    oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_w_stack = []
    any_degen = False
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        w, _r_train = _simplex_slsqp(
            P_unb[tr_loc], y_unb[tr_loc],
            n_starts=N_STARTS_FOLD,
            seed=kf_seed * 11 + fold_i,
        )
        val_pred = P_unb[va_loc] @ w
        oof_blend[va_loc] = val_pred
        fold_val_raes.append(float(rae(y_unb[va_loc], val_pred)))
        fold_w_stack.append(w)
        if w.max() > DEGEN_MAX_W:
            any_degen = True
    if np.isnan(oof_blend).any():
        raise RuntimeError(f"scaffold splits did not cover all rows (kf_seed={kf_seed})")
    pooled = float(rae(y_unb, oof_blend))
    w_mean = np.stack(fold_w_stack, axis=0).mean(axis=0)
    w_mean = w_mean / w_mean.sum()
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(fold_val_raes)),
        "per_fold_val_rae_std": float(np.std(fold_val_raes, ddof=1)),
        "mean_fold_weights": w_mean.tolist(),
        "any_fold_degenerate": any_degen,
        "oof": oof_blend,
    }


# ============================================================================
# main
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- XGBoost residual on K=18, K=19; per-fold simplex blend with"
          f" LGBM versions")
    print(f"          XGB params: max_depth=4, n_est=300, lr=0.03, reg_lambda=2.0,"
          f" tree=hist")
    print(f"          deep-30 XGB seeds = {RESID_SEEDS_DEEP[0]}..{RESID_SEEDS_DEEP[-1]}"
          f" (n={len(RESID_SEEDS_DEEP)})")
    print(f"          outer CV: {N_FOLDS}-fold scaffold, {len(KF_SEEDS)} kf_seeds"
          f" {KF_SEEDS[0]}..{KF_SEEDS[-1]}")
    print(f"          gate: mean_pooled < {GATE_BETTER} -> BETTER_THAN_NB3030")
    print("=" * 78)

    # -- Load truth + anchor --------------------------------------------------
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
    print(f"[load] chemprop_aux te[unb_idx] RAE = {rae_anchor:.4f}"
          f" (ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor

    # -- K=18 + K=19 idx ------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 1: load K-feature indices")
    print("-" * 78)
    with open(NB2960_SUMMARY) as f:
        nb2960 = json.load(f)
    K18_idx = np.array(nb2960["K18_idx_in_117col"], dtype=int)
    assert len(K18_idx) == 18, f"K18 len {len(K18_idx)} != 18"
    print(f"   K=18 idx (n={len(K18_idx)}): {K18_idx.tolist()}")

    with open(NB2231_SUMMARY) as f:
        nb2231 = json.load(f)
    K19_idx = np.array(reconstruct_K_from_trajectory(nb2231, 19), dtype=int)
    assert len(K19_idx) == 19, f"K19 len {len(K19_idx)} != 19"
    print(f"   K=19 idx (n={len(K19_idx)}): {K19_idx.tolist()}")

    # -- 117-col matrix -------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 2: rebuild 117-col 5-way feature matrix")
    print("-" * 78)
    X_te_full, chembl_pool_size = build_117col_feature_matrix(te_smiles, n_test)
    print(f"   X_te_full = {X_te_full.shape}  chembl_pool = {chembl_pool_size}")

    # -- K18-XGB + K19-XGB deep-30 bag ----------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 3: build XGB deep-30 bags for K=18 and K=19")
    print("-" * 78)
    t_k18 = time.time()
    K18_xgb_oof, K18_xgb_te, K18_xgb_per_seed = build_K_xgb_30seed_bag(
        "K18", K18_idx, X_te_full, unb_idx, anchor, residual,
        te_anchor_513, n_test, n_unb, RESID_SEEDS_DEEP,
    )
    K18_xgb_rae = float(rae(y_unb, K18_xgb_oof))
    print(f"   K18-XGB bag-mean RAE = {K18_xgb_rae:.4f}  "
          f"per-seed mean = {np.mean(K18_xgb_per_seed):.4f}  "
          f"wall = {time.time()-t_k18:.1f}s")

    t_k19 = time.time()
    K19_xgb_oof, K19_xgb_te, K19_xgb_per_seed = build_K_xgb_30seed_bag(
        "K19", K19_idx, X_te_full, unb_idx, anchor, residual,
        te_anchor_513, n_test, n_unb, RESID_SEEDS_DEEP,
    )
    K19_xgb_rae = float(rae(y_unb, K19_xgb_oof))
    print(f"   K19-XGB bag-mean RAE = {K19_xgb_rae:.4f}  "
          f"per-seed mean = {np.mean(K19_xgb_per_seed):.4f}  "
          f"wall = {time.time()-t_k19:.1f}s")

    # Save XGB artifacts ------------------------------------------------------
    np.save(DATA_PROCESSED / f"{TAG}_K18_XGB_oof.npy", K18_xgb_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_K18_XGB_te.npy", K18_xgb_te.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_K19_XGB_oof.npy", K19_xgb_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_K19_XGB_te.npy", K19_xgb_te.astype(np.float32))

    # -- Load cached LGBM K18 + K19 OOFs / te ----------------------------------
    print("\n" + "-" * 78)
    print("STEP 4: load cached LGBM K=18 and K=19 deep-30 anchors")
    print("-" * 78)
    K18_lgbm_oof = np.load(K18_LGBM_OOF).astype(np.float64)
    K18_lgbm_te = np.load(K18_LGBM_TE).astype(np.float64)
    K19_lgbm_oof = np.load(K19_LGBM_OOF).astype(np.float64)
    K19_lgbm_te = np.load(K19_LGBM_TE).astype(np.float64)
    K18_lgbm_rae = float(rae(y_unb, K18_lgbm_oof))
    K19_lgbm_rae = float(rae(y_unb, K19_lgbm_oof))
    print(f"   K18-LGBM cached oof RAE = {K18_lgbm_rae:.4f} (ref {REF_K18_LGBM:.4f})")
    print(f"   K19-LGBM cached oof RAE = {K19_lgbm_rae:.4f} (ref {REF_K19_LGBM:.4f})")

    # -- Stack 4 anchors -------------------------------------------------------
    labels = ["K18_LGBM", "K19_LGBM", "K18_XGB", "K19_XGB"]
    P_unb = np.column_stack([K18_lgbm_oof, K19_lgbm_oof, K18_xgb_oof, K19_xgb_oof])
    P_te = np.column_stack([K18_lgbm_te, K19_lgbm_te, K18_xgb_te, K19_xgb_te])
    K = len(labels)

    per_K_full_rae = {
        "K18_LGBM": round(K18_lgbm_rae, 4),
        "K19_LGBM": round(K19_lgbm_rae, 4),
        "K18_XGB": round(K18_xgb_rae, 4),
        "K19_XGB": round(K19_xgb_rae, 4),
    }

    # Leak sanity
    leak_flags = {}
    for i, k in enumerate(labels):
        frac = float(np.mean(np.isclose(P_unb[:, i], y_unb, atol=1e-6)))
        leak_flags[k] = round(frac, 4)
        if frac > 0.05:
            print(f"   WARN {k}: {frac:.1%} rows == truth -- possible leak")

    # Correlation
    corr_mat = np.corrcoef(P_unb.T)
    print(f"\n   4-anchor OOF correlation matrix:")
    print(f"            {'  '.join([f'{k:>9s}' for k in labels])}")
    for i, ki in enumerate(labels):
        row = "  ".join([f"{corr_mat[i, j]:9.4f}" for j in range(K)])
        print(f"   {ki:>9s}  {row}")

    # -- Scaffolds ------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 5: scaffolds for outer CV")
    print("-" * 78)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"   unique scaffolds = {n_unique_scaf}")

    # -- Outer CV sweep --------------------------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 6: per-fold simplex on 4 anchors, {len(KF_SEEDS)} kf_seeds")
    print("-" * 78)
    seed_records = []
    pooled_raes = []
    w_mean_stack = []
    oof_stack = []
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(P_unb, y_unb, unb_scaffolds, s, labels)
        pooled_raes.append(res["pooled_rae"])
        w_mean_stack.append(res["mean_fold_weights"])
        oof_stack.append(res["oof"])
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
            "per_fold_val_rae_std": round(res["per_fold_val_rae_std"], 4),
            "mean_fold_weights": {
                labels[k]: round(float(res["mean_fold_weights"][k]), 4)
                for k in range(K)
            },
            "any_fold_degenerate": res["any_fold_degenerate"],
        })
        wstr = ", ".join(f"{labels[k]}={res['mean_fold_weights'][k]:.3f}"
                         for k in range(K))
        print(f"   kf={s}: pooled_RAE={res['pooled_rae']:.4f}  "
              f"degen={res['any_fold_degenerate']}  "
              f"w=[{wstr}]  wall={time.time()-ts:.2f}s")

    arr = np.asarray(pooled_raes, dtype=np.float64)
    mean_rae = float(arr.mean())
    std_rae = float(arr.std(ddof=1))
    sem = std_rae / np.sqrt(len(arr))
    t_mult = 2.145   # 95% CI, df=14
    ci_low = mean_rae - t_mult * sem
    ci_high = mean_rae + t_mult * sem
    median_rae = float(np.median(arr))

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({len(KF_SEEDS)} fresh seeds)")
    print("-" * 78)
    print(f"   mean    = {mean_rae:.4f}")
    print(f"   std     = {std_rae:.4f}")
    print(f"   95% CI  = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   median  = {median_rae:.4f}")
    print(f"   min/max = [{arr.min():.4f}, {arr.max():.4f}]")

    # Mean-of-seed mean-of-fold weights
    w_seed_mean = np.asarray(w_mean_stack).mean(axis=0)
    w_seed_mean = w_seed_mean / w_seed_mean.sum()
    print(f"\n   mean-of-seed mean-of-fold weights = "
          + ", ".join(f"{labels[k]}={w_seed_mean[k]:.4f}" for k in range(K)))

    # -- Deploy: SLSQP on FULL 253 --------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 7: deploy SLSQP on FULL 253")
    print("-" * 78)
    w_full, r_full = _simplex_slsqp(P_unb, y_unb, n_starts=N_STARTS_FULL, seed=0)
    full_pool_weights = {labels[k]: round(float(w_full[k]), 4) for k in range(K)}
    full_pool_degen = bool(w_full.max() > DEGEN_MAX_W)
    te_pred = (P_te @ w_full).astype(np.float32)
    te_pred = np.clip(te_pred, 3.0, 9.0)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"   full-pool SLSQP weights = "
          + ", ".join(f"{labels[k]}={w_full[k]:.4f}" for k in range(K)))
    print(f"   full-pool in-sample RAE = {r_full:.4f}")
    print(f"   te[unb] in-sample RAE   = {te_unb_in_rae:.4f}")
    print(f"   te(513) mean={te_pred.mean():.3f}  std={te_pred.std():.3f}")

    # Median-seed OOF for storage
    med_seed_idx = int(np.argsort(arr)[len(arr) // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)

    # -- Gate -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    if mean_rae < GATE_BETTER:
        verdict = "BETTER_THAN_NB3030"
        ladder_action = (f"PROMOTE: mean_pooled {mean_rae:.4f} < {GATE_BETTER} "
                         f"(nb3030 ceiling). XGB axis adds orthogonal signal.")
    else:
        verdict = "FAIL"
        ladder_action = (f"REJECT: mean_pooled {mean_rae:.4f} >= {GATE_BETTER} "
                         f"(nb3030 ceiling). XGB cross-paradigm fails to break.")

    print(f"   verdict       = {verdict}")
    print(f"   ladder action = {ladder_action}")

    # -- Save artifacts -------------------------------------------------------
    print("\n" + "-" * 78)
    print("SAVE")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_for_save)
    np.save(te_path, te_pred)
    print(f"   [save] {oof_path}  (median-seed OOF, kf_seed={median_seed})")
    print(f"   [save] {te_path}   (deploy from FULL-253 SLSQP)")

    sub_csv = SUBMISSIONS / f"{TAG}_xgb_K18_K19.csv"
    if verdict == "BETTER_THAN_NB3030":
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
        "method": "xgb_K18_K19_residual_plus_per_fold_simplex_w_lgbm_versions",
        "paradigm": "cross_paradigm_orthogonality_xgb_vs_lgbm",
        "anchor_labels": labels,
        "anchor_pre_unblind": True,
        "per_K_full_oof_rae": per_K_full_rae,
        "anchor_leak_eq_truth_frac": leak_flags,
        "oof_corr_matrix": corr_mat.tolist(),
        "oof_corr_labels": labels,
        "xgb_params": {
            "max_depth": 4,
            "n_estimators": 300,
            "learning_rate": 0.03,
            "reg_lambda": 2.0,
            "objective": "reg:squarederror",
            "tree_method": "hist",
        },
        "xgb_seeds_deep30": RESID_SEEDS_DEEP,
        "K18_XGB_per_seed_rae_mean": round(float(np.mean(K18_xgb_per_seed)), 4),
        "K18_XGB_per_seed_rae_std": round(float(np.std(K18_xgb_per_seed, ddof=1)), 4),
        "K19_XGB_per_seed_rae_mean": round(float(np.mean(K19_xgb_per_seed)), 4),
        "K19_XGB_per_seed_rae_std": round(float(np.std(K19_xgb_per_seed, ddof=1)), 4),
        "K18_XGB_bag_mean_rae": round(K18_xgb_rae, 4),
        "K19_XGB_bag_mean_rae": round(K19_xgb_rae, 4),
        "K18_idx_in_117col": K18_idx.tolist(),
        "K19_idx_in_117col": K19_idx.tolist(),
        "anchor_rae_in_sample": round(rae_anchor, 4),
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": len(KF_SEEDS),
        "n_starts_fold": N_STARTS_FOLD,
        "n_starts_full": N_STARTS_FULL,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "K_anchors": K,
        "seed_records": seed_records,
        "pooled_rae_array": [round(float(v), 4) for v in pooled_raes],
        "mean_rae": round(mean_rae, 5),
        "std_rae": round(std_rae, 5),
        "sem_rae": round(sem, 5),
        "ci95_low": round(ci_low, 5),
        "ci95_high": round(ci_high, 5),
        "median_rae": round(median_rae, 5),
        "min_rae": round(float(arr.min()), 5),
        "max_rae": round(float(arr.max()), 5),
        "mean_of_seed_mean_fold_weights": {
            labels[k]: round(float(w_seed_mean[k]), 4) for k in range(K)
        },
        "full_pool_weights": full_pool_weights,
        "full_pool_rae_in_sample": round(float(r_full), 4),
        "full_pool_degenerate": full_pool_degen,
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "median_seed": int(median_seed),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict == "BETTER_THAN_NB3030" else None,
        "ref_nb3030": REF_NB3030,
        "ref_nb3001": REF_NB3001,
        "ref_nb2992": REF_NB2992,
        "ref_K18_LGBM": REF_K18_LGBM,
        "ref_K19_LGBM": REF_K19_LGBM,
        "ref_nb2171": REF_NB2171,
        "delta_vs_nb3030": round(mean_rae - REF_NB3030, 5),
        "delta_vs_nb2992": round(mean_rae - REF_NB2992, 5),
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
    print(f"   per-K full-OOF RAE       = "
          + ", ".join(f"{k}={v:.4f}" for k, v in per_K_full_rae.items()))
    print(f"   mean pooled RAE (15-seed)= {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   95% CI                   = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   delta vs nb3030 (0.4509) = {mean_rae - REF_NB3030:+.5f}")
    print(f"   full-pool weights        = {full_pool_weights}")
    print(f"   te[unb] in-sample RAE    = {te_unb_in_rae:.4f}")
    print(f"   verdict                  = {verdict}")
    print(f"   wall                     = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae", "std_rae", "ci95_low", "ci95_high",
        "delta_vs_nb3030", "delta_vs_nb2992",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
    print(f"  per_K_full_oof_rae: {res.get('per_K_full_oof_rae')}")
    print(f"  full_pool_weights: {res.get('full_pool_weights')}")
