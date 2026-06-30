"""nb3051 -- Rebuild K=25 at deep-30, then 3-anchor SLSQP simplex
            {K18, K19, K25} at 15 fresh kf_seeds.

NEW PARADIGM (cycle 250+):
    K=25 sits between K=24 (5-seed 0.4687 per nb2261) and K=28 (SHAP-top
    canonical). Tests if an intermediate K-grid point unlocks gains over the
    co-converged ceiling {K18 0.4536 / K19 0.4607 / K23 deep-30 ceiling-band}
    after the nb3030 wide-seed mean 0.4509 was promoted to PRIMARY-1.

PROTOCOL:
    STEP A  -- Rebuild K=25 deep-30 (seeds 3001..3030) with identical recipe
               to nb3020 / nb3000 / nb2960 / nb3014 (chemprop_aux anchor +
               residual LGBM on K=25 feature slice of nb2231 117-col matrix).
    STEP B  -- Per-fold SLSQP simplex on {K18, K19, K25} all deep-30, swept
               over 15 fresh kf_seeds {1081..1095} (NOT used in any prior
               nb29xx/nb30xx wide-seed verify or single-fit).
    STEP C  -- Deploy: SLSQP refit on FULL 253 -> single weight vector ->
               apply to (513, 3) stacked te arrays -> te_nb3051.

GATE (on wide-seed mean across 15 fresh kf_seeds):
    mean < 0.4509 -> "BETTER_THAN_NB3030"   (beats nb3030 wide-seed PRIMARY-1)
    else          -> "FAIL"

References:
    nb2960 K18 deep-30 OOF             = 0.4536
    nb3000 K19 deep-30 OOF             = 0.4607
    nb2960 K20 deep-30 OOF             = 0.4625
    nb3014 K21 deep-30 OOF             = 0.4595
    nb3014 K22 deep-30 OOF             = 0.4732
    nb3020 K23 deep-30 OOF             = (cached)
    nb2261 K24 5-seed OOF (ref)        = 0.4687
    nb3030 wide-15-seed 3K {K18,K19,K23} mean = 0.4509   <- GATE
    nb3001 wide-15-seed 3K mean        = 0.4511
    nb3003 5-anchor single-seed        = 0.4518
    nb2992 per-fold simplex 3K         = 0.4479
    nb2171 prior ceiling deep-30       = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/te_chemprop_aux.npy
    data/processed/nb2231_summary.json   (K=25 idx reconstruction via traj)
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
    data/processed/nb2960_K18_30seed_oof.npy   (K18 deep-30 cached)
    data/processed/nb2960_K18_30seed_te.npy
    data/processed/nb3000_K19_30seed_oof.npy   (K19 deep-30 cached)
    data/processed/te_nb3000_K19.npy

Outputs:
    data/processed/nb3051_summary.json
    data/processed/nb3051_K25_30seed_oof.npy   (253,) float32
    data/processed/te_nb3051_K25.npy           (513,) float32
    data/processed/nb3051_pred_oof.npy         (253,) float32 -- median-seed OOF
    data/processed/te_nb3051.npy               (513,) float32 -- deploy te
    submissions/nb3051_K25_deep30_simplex.csv  (only on BETTER_THAN_NB3030)
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

from pxr.chem import bemis_murcko, standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3051"
PARENT_TAG = "nb2960+nb3000+nb3051(K25 deep30)"

# -- Anchor + residual params (IDENTICAL recipe to nb3020 / nb3000 / nb2960) --
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
RESID_FOLDS = 5
RESID_SEEDS_DEEP = list(range(3001, 3031))    # 30 fresh seeds {3001..3030}

# -- Feature cache paths -------------------------------------------------------
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

# -- Cached deep-30 anchors (K18/K19) -----------------------------------------
K18_DEEP30_OOF = DATA_PROCESSED / "nb2960_K18_30seed_oof.npy"
K18_DEEP30_TE = DATA_PROCESSED / "nb2960_K18_30seed_te.npy"
K19_DEEP30_OOF = DATA_PROCESSED / "nb3000_K19_30seed_oof.npy"
K19_DEEP30_TE = DATA_PROCESSED / "te_nb3000_K19.npy"

# -- ChEMBL kNN params (identical to nb2604 / nb2631 / nb2960 / nb3000) -------
KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

# -- CV protocol (SLSQP step) -------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1081, 1096))   # 15 fresh seeds {1081..1095}
                                     #   NOT 1001 (single-fit seed)
                                     #   NOT 1006..1080 (prior wide-seed verifies)
N_STARTS_FOLD = 8
N_STARTS_FULL = 12
DEGEN_MAX_W = 0.85

# -- Gates --------------------------------------------------------------------
GATE_BETTER_THAN_NB3030 = 0.4509

# -- References ---------------------------------------------------------------
REF_K18_DEEP30 = 0.4536
REF_K19_DEEP30 = 0.4607
REF_K22_DEEP30 = 0.4732
REF_K24_5SEED = 0.4687
REF_NB2992 = 0.4479
REF_NB3001 = 0.4511
REF_NB3003 = 0.4518
REF_NB3030 = 0.4509
REF_NB2171 = 0.4682
CHEMPROP_AUX_REF = 0.6216

K_LABELS = ["K18", "K19", "K25"]
K_TARGET_REBUILD = 25


# ============================================================================
# helpers (lifted verbatim from nb3020 / nb3014 / nb3000 / nb2960 / nb2631)
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
    trajectory (verbatim from nb3000 / nb2631 / nb3010 / nb3014 / nb3020)."""
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
    """117-col matrix identical to nb2604 / nb2631 / nb2960 / nb3000 / nb3020."""
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
    """Build deep-30 bag-mean OOF (253,) + te (513,) for one K-pyramid."""
    X_te_K = X_te_full[:, K_idx].astype(np.float32)
    X_unb_K = X_te_K[unb_idx]
    print(f"   [{K_label}] X_unb_K = {X_unb_K.shape}  X_te_K = {X_te_K.shape}")
    sum_unb = np.zeros(n_unb, dtype=np.float64)
    sum_te = np.zeros(n_test, dtype=np.float64)
    per_seed_rae = []
    for i, s in enumerate(seeds):
        ts = time.time()
        resid_oof = _residual_cross_fit_one_seed(X_unb_K, residual, s)
        pred_unb_s = anchor + resid_oof
        sum_unb += pred_unb_s
        per_seed_rae.append(float(rae(anchor + residual, pred_unb_s)))
        te_resid_s = _train_full_then_predict_te(X_unb_K, residual, X_te_K, s)
        pred_te_s = te_anchor_513 + te_resid_s
        sum_te += pred_te_s
        if (i % 5) == 0 or i == len(seeds) - 1:
            print(f"      [{K_label}] seed={s:4d}  "
                  f"rae={per_seed_rae[-1]:.4f}  "
                  f"wall={time.time()-ts:.2f}s  ({i+1}/{len(seeds)})")
    bag_oof_unb = sum_unb / len(seeds)
    bag_te_513 = sum_te / len(seeds)
    return bag_oof_unb, bag_te_513, per_seed_rae


def _simplex_slsqp(P: np.ndarray, y: np.ndarray, n_starts: int = 8,
                   seed: int = 0) -> tuple[np.ndarray, float]:
    """Minimize RAE(y, P @ w) over the simplex (w>=0, sum w=1) with multi-start."""
    K = P.shape[1]
    rng = np.random.default_rng(seed)

    def loss(w: np.ndarray) -> float:
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


def _run_one_seed(P_unb: np.ndarray, y_unb: np.ndarray,
                  unb_scaffolds: list, kf_seed: int) -> dict:
    """Per-fold SLSQP simplex pipeline at a single kf_seed (mirrors nb3030)."""
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
        raise RuntimeError(f"kf_seed={kf_seed}: scaffold splits did not cover all rows")
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
    print(f"{TAG} -- rebuild K=25 deep-30, then 3-anchor SLSQP simplex over "
          f"15 fresh kf_seeds")
    print(f"          resid_seeds = {RESID_SEEDS_DEEP[0]}..{RESID_SEEDS_DEEP[-1]} "
          f"(n={len(RESID_SEEDS_DEEP)})")
    print(f"          kf_seeds = {KF_SEEDS[0]}..{KF_SEEDS[-1]} "
          f"(n={len(KF_SEEDS)}, FRESH)")
    print(f"          K=22 ref deep30 = {REF_K22_DEEP30:.4f}  "
          f"K=24 ref 5seed = {REF_K24_5SEED:.4f}")
    print(f"          gate <{GATE_BETTER_THAN_NB3030} BETTER_THAN_NB3030 "
          f"(nb3030 wide-seed PRIMARY-1)")
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

    # -- Reconstruct K=25 indices ---------------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 1: reconstruct K=25 idx from nb2231 RFE trajectory")
    print("-" * 78)
    if not NB2231_SUMMARY.exists():
        raise FileNotFoundError(f"missing {NB2231_SUMMARY}")
    with open(NB2231_SUMMARY) as f:
        nb2231 = json.load(f)
    K25_idx = np.array(reconstruct_K_from_trajectory(nb2231, K_TARGET_REBUILD), dtype=int)
    if len(K25_idx) != K_TARGET_REBUILD:
        raise ValueError(f"K=25 idx reconstruction returned {len(K25_idx)} cols")
    print(f"   K=25 idx (n={len(K25_idx)}): {K25_idx.tolist()}")

    # -- Build 117-col matrix -------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 2: rebuild 117-col 5-way feature matrix")
    print("-" * 78)
    X_te_full, chembl_pool_size = build_117col_feature_matrix(te_smiles, n_test)
    print(f"   X_te_full = {X_te_full.shape}  chembl_pool={chembl_pool_size}")

    # -- Build K=25 deep-30 ---------------------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 3: K=25 residual-LGBM with {len(RESID_SEEDS_DEEP)} fresh seeds")
    print("-" * 78)
    bag_oof_K25, bag_te_K25, per_seed_rae_K25 = build_K_30seed_bag(
        "K25", K25_idx, X_te_full, unb_idx, anchor, residual,
        te_anchor_513, n_test, n_unb, RESID_SEEDS_DEEP,
    )
    bag_mean_rae_K25 = float(rae(y_unb, bag_oof_K25))
    per_seed_arr_K25 = np.array(per_seed_rae_K25, dtype=np.float64)
    print(f"\n   [K25] per-seed RAE mean = {per_seed_arr_K25.mean():.4f}  "
          f"std = {per_seed_arr_K25.std(ddof=1):.4f}")
    print(f"   [K25] 30-seed BAG-MEAN RAE = {bag_mean_rae_K25:.4f}")
    K25_oof_path = DATA_PROCESSED / f"{TAG}_K25_30seed_oof.npy"
    K25_te_path = DATA_PROCESSED / f"te_{TAG}_K25.npy"
    np.save(K25_oof_path, bag_oof_K25.astype(np.float32))
    np.save(K25_te_path, bag_te_K25.astype(np.float32))
    print(f"   [save] {K25_oof_path}")
    print(f"   [save] {K25_te_path}")

    # -- Load all 3 deep-30 anchors -------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 4: load 3 K-anchors (all deep-30) for per-fold SLSQP simplex")
    print("-" * 78)
    oof_list = {
        "K18": np.load(K18_DEEP30_OOF).astype(np.float64),
        "K19": np.load(K19_DEEP30_OOF).astype(np.float64),
        "K25": bag_oof_K25.astype(np.float64),
    }
    te_list = {
        "K18": np.load(K18_DEEP30_TE).astype(np.float64),
        "K19": np.load(K19_DEEP30_TE).astype(np.float64),
        "K25": bag_te_K25.astype(np.float64),
    }

    per_K_full_rae = {}
    for k in K_LABELS:
        oof = oof_list[k]
        te_arr = te_list[k]
        if oof.shape != (n_unb,):
            raise ValueError(f"{k} OOF shape {oof.shape} != ({n_unb},)")
        if te_arr.shape != (n_test,):
            raise ValueError(f"{k} te shape {te_arr.shape} != ({n_test},)")
        per_K_full_rae[k] = round(float(rae(y_unb, oof)), 4)
        print(f"   {k} (deep30): full_OOF_RAE = {per_K_full_rae[k]:.4f}  "
              f"te_mean={te_arr.mean():.3f}  te_std={te_arr.std():.3f}")

    P_unb = np.column_stack([oof_list[k] for k in K_LABELS])    # (253, 3)
    P_te = np.column_stack([te_list[k] for k in K_LABELS])      # (513, 3)
    K = len(K_LABELS)

    leak_flags = {}
    for i, k in enumerate(K_LABELS):
        frac = float(np.mean(np.isclose(P_unb[:, i], y_unb, atol=1e-6)))
        leak_flags[k] = round(frac, 4)
        if frac > 0.05:
            print(f"   WARN {k}: {frac:.1%} rows == truth -- possible leak")
    corr_mat = np.corrcoef(P_unb.T)
    print(f"\n  OOF correlation matrix (all deep-30):")
    print(f"        {'  '.join([f'{k:>6s}' for k in K_LABELS])}")
    for i, ki in enumerate(K_LABELS):
        row = "  ".join([f"{corr_mat[i, j]:6.3f}" for j in range(K)])
        print(f"   {ki:>6s}  {row}")

    # -- Scaffolds (kf_seed independent) --------------------------------------
    print("\n" + "-" * 78)
    print("STEP 5: scaffolds for outer CV")
    print("-" * 78)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"   n_unique_scaffolds = {n_unique_scaf}")

    # -- Wide-seed sweep ------------------------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 6: wide-seed sweep ({len(KF_SEEDS)} fresh kf_seeds "
          f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}})")
    print("-" * 78)
    seed_records = []
    pooled_raes = []
    w_mean_stack = []
    oof_stack = []
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(P_unb, y_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        w_mean_stack.append(res["mean_fold_weights"])
        oof_stack.append(res["oof"])
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
            "per_fold_val_rae_std": round(res["per_fold_val_rae_std"], 4),
            "mean_fold_weights": {
                K_LABELS[k]: round(float(res["mean_fold_weights"][k]), 4)
                for k in range(K)
            },
            "any_fold_degenerate": res["any_fold_degenerate"],
        })
        wstr = ", ".join(f"{K_LABELS[k]}={res['mean_fold_weights'][k]:.3f}"
                         for k in range(K))
        print(f"   kf={s}: pooled_RAE={res['pooled_rae']:.4f}  "
              f"degen={res['any_fold_degenerate']}  "
              f"w=[{wstr}]  wall={time.time()-ts:.2f}s")

    arr = np.asarray(pooled_raes, dtype=np.float64)
    mean_rae = float(arr.mean())
    std_rae = float(arr.std(ddof=1))
    sem = std_rae / np.sqrt(len(arr))
    t_mult = 2.145  # t_{0.975, df=14}
    ci_low = mean_rae - t_mult * sem
    ci_high = mean_rae + t_mult * sem
    median_rae = float(np.median(arr))
    p5 = float(np.percentile(arr, 5))
    p95 = float(np.percentile(arr, 95))

    print("\n" + "-" * 78)
    print("AGGREGATE (15 fresh seeds)")
    print("-" * 78)
    print(f"   mean    = {mean_rae:.4f}")
    print(f"   std     = {std_rae:.4f}")
    print(f"   sem     = {sem:.4f}")
    print(f"   95% CI  = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   median  = {median_rae:.4f}")
    print(f"   5/95p   = [{p5:.4f}, {p95:.4f}]")
    print(f"   min/max = [{arr.min():.4f}, {arr.max():.4f}]")
    print(f"   nb3030 wide-seed ref      = {REF_NB3030:.4f}")
    print(f"   nb3001 wide-seed ref      = {REF_NB3001:.4f}")
    print(f"   delta vs nb3030 (0.4509)  = {mean_rae - REF_NB3030:+.4f}")

    # Mean-of-seed mean-of-fold weights (deploy proxy)
    w_seed_mean = np.asarray(w_mean_stack).mean(axis=0)
    w_seed_mean = w_seed_mean / w_seed_mean.sum()
    print(f"\n   mean-of-seed mean-of-fold weights = "
          + ", ".join(f"{K_LABELS[k]}={w_seed_mean[k]:.4f}" for k in range(K)))

    # -- Deploy: SLSQP on FULL 253 (kf-seed-independent) ----------------------
    print("\n" + "-" * 78)
    print("STEP 7: deploy SLSQP on FULL 253")
    print("-" * 78)
    w_full, r_full = _simplex_slsqp(P_unb, y_unb, n_starts=N_STARTS_FULL, seed=0)
    full_pool_weights = {K_LABELS[k]: round(float(w_full[k]), 4) for k in range(K)}
    full_pool_degen = bool(w_full.max() > DEGEN_MAX_W)
    te_pred = (P_te @ w_full).astype(np.float32)
    te_pred = np.clip(te_pred, 3.0, 9.0)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"   full-pool SLSQP weights   = "
          + ", ".join(f"{K_LABELS[k]}={w_full[k]:.4f}" for k in range(K)))
    print(f"   full-pool in-sample RAE   = {r_full:.4f}")
    print(f"   te[unb] in-sample RAE     = {te_unb_in_rae:.4f}")
    print(f"   te(513) mean={te_pred.mean():.3f}  std={te_pred.std():.3f}")

    # Median-seed OOF for storage
    med_seed_idx = int(np.argsort(arr)[len(arr) // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(f"   median seed = {median_seed} (rae={arr[med_seed_idx]:.4f})")

    # -- Gate -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 8: GATE")
    print("-" * 78)
    if mean_rae < GATE_BETTER_THAN_NB3030:
        verdict = "BETTER_THAN_NB3030"
        ladder_action = (
            f"PROMOTE nb3051 to PRIMARY-1 candidate (wide-seed mean "
            f"{mean_rae:.4f} beats nb3030 ceiling {REF_NB3030:.4f}). Demote "
            "nb3030 to PRIMARY-2."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT nb3051. Wide-seed mean {mean_rae:.4f} >= nb3030 "
            f"ceiling {REF_NB3030:.4f}. Keep nb3030 PRIMARY-1."
        )
    delta_vs_nb3030 = mean_rae - REF_NB3030
    delta_vs_nb3001 = mean_rae - REF_NB3001
    delta_vs_nb3003 = mean_rae - REF_NB3003
    delta_vs_nb2992 = mean_rae - REF_NB2992
    delta_vs_nb2171 = mean_rae - REF_NB2171
    print(f"   wide_seed_mean           = {mean_rae:.4f}")
    print(f"   delta vs nb3030 (0.4509) = {delta_vs_nb3030:+.4f}")
    print(f"   delta vs nb3001 (0.4511) = {delta_vs_nb3001:+.4f}")
    print(f"   delta vs nb3003 (0.4518) = {delta_vs_nb3003:+.4f}")
    print(f"   delta vs nb2992 (0.4479) = {delta_vs_nb2992:+.4f}")
    print(f"   delta vs nb2171 (0.4682) = {delta_vs_nb2171:+.4f}")
    print(f"   verdict                  = {verdict}")
    print(f"   ladder action            = {ladder_action}")

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

    sub_csv = SUBMISSIONS / f"{TAG}_K25_deep30_simplex.csv"
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
        "method": "K25_deep30_then_3anchor_perfold_slsqp_simplex_wide_kf15",
        "paradigm": "rebuild_K25_at_deep30_then_3K_per_fold_slsqp_wide_seed_15",
        "anchor_pool": K_LABELS,
        "anchor_depth": {k: "deep30" for k in K_LABELS},
        "anchor_pre_unblind": True,
        "resid_seeds_deep": RESID_SEEDS_DEEP,
        "n_seeds_deep": len(RESID_SEEDS_DEEP),
        "K_target_rebuilt": K_TARGET_REBUILD,
        "K25_idx_in_117": K25_idx.tolist(),
        # K25 rebuild
        "K25_per_seed_rae": per_seed_rae_K25,
        "K25_per_seed_rae_mean": float(per_seed_arr_K25.mean()),
        "K25_per_seed_rae_std": float(per_seed_arr_K25.std(ddof=1)),
        "K25_per_seed_rae_min": float(per_seed_arr_K25.min()),
        "K25_per_seed_rae_max": float(per_seed_arr_K25.max()),
        "K25_30seed_bag_mean_rae": bag_mean_rae_K25,
        "K25_oof_path": str(K25_oof_path),
        "K25_te_path": str(K25_te_path),
        "K25_te_mean": float(bag_te_K25.mean()),
        "K25_te_std": float(bag_te_K25.std()),
        # SLSQP simplex per-K
        "per_K_full_oof_rae": per_K_full_rae,
        "anchor_leak_eq_truth_frac": leak_flags,
        "oof_corr_matrix": corr_mat.tolist(),
        "oof_corr_labels": K_LABELS,
        # CV protocol
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_kf_seeds": len(KF_SEEDS),
        "n_starts_fold": N_STARTS_FOLD,
        "n_starts_full": N_STARTS_FULL,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "K_anchors": K,
        # Wide-seed sweep
        "seed_records": seed_records,
        "pooled_rae_array": [round(float(v), 4) for v in pooled_raes],
        "mean_rae": round(mean_rae, 4),
        "std_rae": round(std_rae, 4),
        "sem_rae": round(sem, 4),
        "ci95_low": round(ci_low, 4),
        "ci95_high": round(ci_high, 4),
        "median_rae": round(median_rae, 4),
        "p5_rae": round(p5, 4),
        "p95_rae": round(p95, 4),
        "min_rae": round(float(arr.min()), 4),
        "max_rae": round(float(arr.max()), 4),
        # Deploy
        "mean_of_seed_mean_fold_weights": {
            K_LABELS[k]: round(float(w_seed_mean[k]), 4) for k in range(K)
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
        # References + gate
        "ref_K18_deep30": REF_K18_DEEP30,
        "ref_K19_deep30": REF_K19_DEEP30,
        "ref_K22_deep30": REF_K22_DEEP30,
        "ref_K24_5seed": REF_K24_5SEED,
        "ref_nb2992": REF_NB2992,
        "ref_nb3001": REF_NB3001,
        "ref_nb3003": REF_NB3003,
        "ref_nb3030": REF_NB3030,
        "ref_nb2171": REF_NB2171,
        "delta_vs_nb3030": delta_vs_nb3030,
        "delta_vs_nb3001": delta_vs_nb3001,
        "delta_vs_nb3003": delta_vs_nb3003,
        "delta_vs_nb2992": delta_vs_nb2992,
        "delta_vs_nb2171": delta_vs_nb2171,
        "gate_better_than_nb3030": GATE_BETTER_THAN_NB3030,
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
    print(f"   K25 deep-30 RAE          = {bag_mean_rae_K25:.4f}  "
          f"(K22 deep30 ref {REF_K22_DEEP30:.4f} / K24 5seed ref {REF_K24_5SEED:.4f})")
    print(f"   per-K full-OOF RAE       = "
          + ", ".join([f"{k}={v:.4f}" for k, v in per_K_full_rae.items()]))
    print(f"   wide-15 seed mean RAE    = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   95% CI                   = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   delta vs nb3030 PRIMARY-1 = {delta_vs_nb3030:+.4f}")
    print(f"   full-pool weights        = {full_pool_weights}")
    print(f"   te[unb_idx] in-sample    = {te_unb_in_rae:.4f}")
    print(f"   verdict                  = {verdict}")
    print(f"   wall                     = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "K25_30seed_bag_mean_rae",
        "K25_per_seed_rae_mean",
        "K25_per_seed_rae_std",
        "per_K_full_oof_rae",
        "mean_rae",
        "std_rae",
        "ci95_low",
        "ci95_high",
        "delta_vs_nb3030",
        "mean_of_seed_mean_fold_weights",
        "full_pool_weights",
        "te_unb_in_sample_rae",
        "verdict",
        "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
