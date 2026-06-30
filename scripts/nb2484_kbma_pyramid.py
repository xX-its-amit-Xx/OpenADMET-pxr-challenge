"""nb2484 -- Bayesian Model Averaging over K={18,20,24,28,32} RFE pyramids.

CONTEXT:
    Distinct from prior nb2171/nb2240/nb2310/nb2330 SLSQP pyramids:
    those optimise weights over a simplex by minimising training SSE.
    nb2484 instead computes weights from the *marginal likelihood*
    approximation (BIC-style on OOF residuals) of each K-anchor and
    blends by w_k = softmax(logML_k). This is data-driven posterior
    averaging, NOT operator-space optimisation, so it sits on a
    fundamentally different axis from the cycle-167-169 closed
    post-hoc-blend cluster.

ANCHORS (all PRE-clean -- chemprop_aux residual; nb730 POST-unblind
contamination chain intentionally excluded):

    K=18 -- RFE K_opt from nb2263 (lucky-seed-aware backward greedy)
            Cols: [45,67,0,66,68,65,92,27,63,1,7,115,46,80,11,70,8,57]
            Rebuilt here (no cached OOF/te in repo).

    K=20 -- nb2240_mean_bag_oof_K20.npy + te_nb2240_K20.npy
            From nb2240 (5-seed mean-bag, K=20 RFE survivors).

    K=24 -- nb2310_mean_bag_oof_K24.npy + te_nb2310_K24.npy
            From nb2310 (K=24 along RFE trajectory).

    K=28 -- nb2103_mean_bag_oof_K28.npy + te REBUILT here.
            (nb2103 only saved OOF, never te; we re-train using
            identical recipe.)

    K=32 -- nb1158_mean_bag_oof_K32.npy + te_nb1158.npy

LOG-MARGINAL-LIKELIHOOD APPROXIMATION:
    Under a Gaussian noise model with shared sigma, log-ML reduces (up
    to an additive constant) to:
        logML_k  ~  -0.5 * N * log(SSE_OOF_k / N)
    All K_k share the same complexity (one LGBM tree-ensemble per
    K-anchor), so the BIC penalty cancels and we use the negative
    log-mean-SSE term only:
        logML_k  =  -0.5 * N * log(SSE_OOF_k)
    Weights:    w_k  =  softmax(logML_k)
    Predict:    p    =  sum_k w_k * pred_K_k

CV PROTOCOL:
    Outer 5-fold scaffold CV on the 253 unblind, kf_seeds 1001-1005.
    BMA weights computed PER OUTER FOLD on tr-residuals against
    truth, then applied to va.

GATE:
    mean_rae < 0.4570  -> "PROMOTE"
    mean_rae < 0.4601  -> "MARGINAL_BEAT"  (nb2240 reference)
    else               -> "FAIL"

Outputs:
    scripts/nb2484_kbma_pyramid.py
    data/processed/nb2484_summary.json
    data/processed/nb2484_pred_oof.npy   (253,) float32
    data/processed/te_nb2484.npy         (513,) float32
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
from sklearn.model_selection import KFold

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2484"

# ---------- K-pyramid catalogue (PRE-clean only) ----------
# K=18 cols from nb2263 RFE K_opt (lucky-seed-aware)
K18_COLS_IN_117 = [45, 67, 0, 66, 68, 65, 92, 27, 63, 1, 7, 115, 46, 80, 11, 70, 8, 57]
# K=28 cols from nb2263 SHAP-top-28 baseline
K28_COLS_IN_117 = [45, 67, 48, 0, 66, 68, 65, 92, 50, 27, 77, 47, 63, 81, 56,
                   1, 52, 7, 53, 115, 93, 46, 80, 11, 70, 54, 8, 57]

# ---------- residual recipe (matches nb2103/nb2240) ----------
RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# ---------- feature-build paths (same as nb2240) ----------
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

# ---------- outer CV ----------
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# ---------- gates ----------
GATE_PROMOTE = 0.4570
GATE_MARGINAL_REF = 0.4601    # nb2240 deep-30 ref


# ============================================================================
# helpers (subset of nb2240 -- just enough to rebuild K=18 / K=28 if missing)
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


def build_117col_matrix(te_smiles, n_test):
    """Identical 117-col 5-way SHAP-tuned feature matrix to nb2240."""
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

    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test)[:, top_ap_bit_idx].astype(np.float32)
    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)[:, top_maccs_bit_idx].astype(np.float32)
    X_mord_te = _load_mordred_test(n_test)[:, top_mord_col_idx].astype(np.float32)
    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)[:, top_embed_col_idx].astype(np.float32)
    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)[:, top_avalon_bit_idx].astype(np.float32)

    # ChEMBL kNN
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
            X_ap_te, X_maccs_te, X_mord_te, X_emb_te, X_av_te,
            pred_chembl_pec50.reshape(-1, 1).astype(np.float32),
            mean_sim.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    assert X_te_full.shape[1] == 117, f"feat_dim {X_te_full.shape[1]} != 117"
    return X_te_full


def build_K_anchor(K_cols, te_anchor_513, residual, unb_idx, X_te_full):
    """Rebuild a K-anchor OOF (on 253) + te (on 513) via residual LGBM mean-bag."""
    n_test = X_te_full.shape[0]
    n_unb = len(unb_idx)
    X_te_K = X_te_full[:, K_cols].astype(np.float32)
    X_unb_K = X_te_K[unb_idx]
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_te_resid = np.zeros((len(RESID_SEEDS), n_test), dtype=np.float64)
    anchor_unb = te_anchor_513[unb_idx]
    for i, s in enumerate(RESID_SEEDS):
        resid_oof = _residual_cross_fit_one_seed(X_unb_K, residual, s)
        per_seed_corrected[i] = anchor_unb + resid_oof
        te_resid_s = _train_full_then_predict_te(X_unb_K, residual, X_te_K, s)
        per_seed_te_resid[i] = te_resid_s
    mean_bag_oof = per_seed_corrected.mean(axis=0)  # 253
    mean_bag_te_resid = per_seed_te_resid.mean(axis=0)  # 513
    te_K_513 = te_anchor_513 + mean_bag_te_resid
    return mean_bag_oof.astype(np.float32), te_K_513.astype(np.float32)


# ============================================================================
# BMA core
# ============================================================================

def bma_weights(P, y, eps=1e-12):
    """Compute Bayesian Model Averaging weights via log-marginal-likelihood
    approximation on OOF residuals (uniform-complexity case).

    logML_k = -0.5 * N * log(SSE_k)   (BIC-style, complexity penalty cancels)
    w_k     = softmax(logML_k)
    """
    K = P.shape[1]
    sse = np.zeros(K, dtype=np.float64)
    for k in range(K):
        r = y - P[:, k]
        sse[k] = float(np.sum(r * r)) + eps
    N = len(y)
    log_ml = -0.5 * N * np.log(sse)
    # numerical-stable softmax
    z = log_ml - log_ml.max()
    w = np.exp(z)
    w = w / w.sum()
    return w, log_ml, sse


def cv_run_for_seed_bma(P_unb, y_unb, unb_scaffolds, kf_seed):
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = P_unb.shape[0]
    oof_blend = np.full(n_unb, np.nan)
    fold_w = []
    fold_logml = []
    for tr_loc, va_loc in splits:
        w_f, log_ml_f, _ = bma_weights(P_unb[tr_loc], y_unb[tr_loc])
        oof_blend[va_loc] = P_unb[va_loc] @ w_f
        fold_w.append(w_f)
        fold_logml.append(log_ml_f)
    return float(rae(y_unb, oof_blend)), oof_blend, fold_w, fold_logml


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- BMA over K={{18,20,24,28,32}} RFE pyramids  (PRE-clean)")
    print("=" * 78)

    # ---- truth + anchor ----
    te_df = load_test()
    n_test = len(te_df)
    te_smiles = (te_df["smiles"].astype(str).tolist()
                 if "smiles" in te_df.columns
                 else te_df["SMILES"].astype(str).tolist())
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    print(f"[anchor] chemprop_aux in_RAE = {rae_anchor:.4f}")
    residual = y_unb - anchor_unb

    # ---- assemble K-pyramids ----
    pyramids = {}    # K -> {oof: (253,), te: (513,)}
    need_117_matrix = False

    # K=20 cached
    p_K20_oof = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"
    p_K20_te  = DATA_PROCESSED / "te_nb2240_K20.npy"
    # K=24 cached
    p_K24_oof = DATA_PROCESSED / "nb2310_mean_bag_oof_K24.npy"
    p_K24_te  = DATA_PROCESSED / "te_nb2310_K24.npy"
    # K=28 oof cached, te not cached
    p_K28_oof = DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy"
    # K=32 cached
    p_K32_oof = DATA_PROCESSED / "nb1158_mean_bag_oof_K32.npy"
    p_K32_te  = DATA_PROCESSED / "te_nb1158.npy"

    # cached paths
    cached = {
        20: (p_K20_oof, p_K20_te),
        24: (p_K24_oof, p_K24_te),
        32: (p_K32_oof, p_K32_te),
    }
    for K, (oof_p, te_p) in cached.items():
        if oof_p.exists() and te_p.exists():
            oof = np.load(oof_p).astype(np.float64)
            te_ = np.load(te_p).astype(np.float64)
            assert oof.shape == (n_unb,), f"K={K} oof shape {oof.shape}"
            assert te_.shape == (n_test,), f"K={K} te shape {te_.shape}"
            pyramids[K] = {"oof": oof, "te": te_, "rebuilt": False,
                           "oof_path": str(oof_p), "te_path": str(te_p)}
            r = float(rae(y_unb, oof))
            print(f"[cached] K={K:2d}  oof_RAE={r:.4f}  te_mean={te_.mean():.3f}  te_std={te_.std():.3f}")
        else:
            need_117_matrix = True
            print(f"[miss]   K={K:2d}  oof_p={oof_p.exists()}  te_p={te_p.exists()}  --> will rebuild")

    # K=28: oof cached, te missing -> rebuild te
    if p_K28_oof.exists():
        oof_K28_cached = np.load(p_K28_oof).astype(np.float64)
        assert oof_K28_cached.shape == (n_unb,)
        # need to rebuild te using same K=28 cols
        need_117_matrix = True
    else:
        need_117_matrix = True
        oof_K28_cached = None

    # K=18: nothing cached
    need_117_matrix = True

    # Build 117-col matrix once if any K needs rebuild
    X_te_full = None
    if need_117_matrix:
        print("\n[feat] Building 117-col SHAP-tuned matrix (one-time)...")
        ts = time.time()
        X_te_full = build_117col_matrix(te_smiles, n_test)
        print(f"[feat] X_te_full = {X_te_full.shape}  wall={time.time()-ts:.1f}s")

    # K=28 rebuild te (oof either cached or rebuilt below)
    print("\n[build] K=28  (rebuild te, oof from cache if avail)")
    ts = time.time()
    oof_K28_new, te_K28 = build_K_anchor(K28_COLS_IN_117, te_anchor_513,
                                         residual, unb_idx, X_te_full)
    if oof_K28_cached is not None:
        # diagnostic: compare cached vs rebuilt
        delta = float(np.mean(np.abs(oof_K28_cached - oof_K28_new)))
        print(f"   K=28 oof cache_vs_rebuild MAE = {delta:.4f}  (using CACHED)")
        oof_K28 = oof_K28_cached
    else:
        oof_K28 = oof_K28_new.astype(np.float64)
    r28 = float(rae(y_unb, oof_K28))
    print(f"   K=28 oof_RAE={r28:.4f}  te_mean={te_K28.mean():.3f}  te_std={te_K28.std():.3f}  wall={time.time()-ts:.1f}s")
    pyramids[28] = {"oof": oof_K28, "te": te_K28.astype(np.float64), "rebuilt": True,
                    "oof_path": str(p_K28_oof) if oof_K28_cached is not None else None,
                    "te_path": None}

    # K=18 rebuild both
    print("\n[build] K=18  (rebuild both oof + te)")
    ts = time.time()
    oof_K18, te_K18 = build_K_anchor(K18_COLS_IN_117, te_anchor_513,
                                     residual, unb_idx, X_te_full)
    r18 = float(rae(y_unb, oof_K18))
    print(f"   K=18 oof_RAE={r18:.4f}  te_mean={te_K18.mean():.3f}  te_std={te_K18.std():.3f}  wall={time.time()-ts:.1f}s")
    pyramids[18] = {"oof": oof_K18.astype(np.float64), "te": te_K18.astype(np.float64),
                    "rebuilt": True, "oof_path": None, "te_path": None}

    # ---- ordered K-list ----
    K_ORDER = [18, 20, 24, 28, 32]
    P_unb = np.column_stack([pyramids[K]["oof"] for K in K_ORDER])
    P_te  = np.column_stack([pyramids[K]["te"]  for K in K_ORDER])
    indiv_rae = {K: float(rae(y_unb, pyramids[K]["oof"])) for K in K_ORDER}
    print("\n[pyramids individual OOF RAE]")
    for K in K_ORDER:
        print(f"   K={K:2d}  RAE={indiv_rae[K]:.4f}")

    # ---- ALL-DATA BMA weights (diagnostic, NOT used for OOF eval) ----
    w_all, logml_all, sse_all = bma_weights(P_unb, y_unb)
    print("\n[BMA all-data diagnostic weights]")
    for K, w_k, lm_k, s_k in zip(K_ORDER, w_all, logml_all, sse_all):
        print(f"   K={K:2d}  SSE={s_k:.2f}  logML={lm_k:.2f}  w={w_k:.4f}")

    # ---- Outer 5-fold scaffold CV across 5 seeds ----
    print("\n" + "-" * 78)
    print(f"OUTER SCAFFOLD 5-FOLD CV  kf_seeds={KF_SEEDS}  (BMA weights per fold)")
    print("-" * 78)
    per_seed = []
    all_oofs = []
    for kf_seed in KF_SEEDS:
        pooled, oof_blend, fw, _fl = cv_run_for_seed_bma(
            P_unb, y_unb, unb_scaffolds, kf_seed,
        )
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": float(pooled),
            "fold_w_mean": [float(x) for x in np.mean(fw, axis=0)],
        })
        all_oofs.append(oof_blend)
        print(f"   seed={kf_seed}  pooled_RAE={pooled:.4f}  "
              f"w_mean={np.round(np.mean(fw, axis=0), 4).tolist()}")
    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    final_oof_rae = float(rae(y_unb, mean_oof))
    pooled_rae_mean_seeds = float(np.mean([r["pooled_rae"] for r in per_seed]))
    pooled_rae_std_seeds = float(np.std([r["pooled_rae"] for r in per_seed]))
    print(f"\n[cv] pooled_RAE mean across {len(KF_SEEDS)} seeds = "
          f"{pooled_rae_mean_seeds:.4f} (+/- {pooled_rae_std_seeds:.4f})")
    print(f"[cv] RAE of mean-of-seed OOFs            = {final_oof_rae:.4f}")

    # ---- Deploy ----
    print("\n" + "-" * 78)
    print("DEPLOY (BMA weights computed on all 253)")
    print("-" * 78)
    blend_te = P_te @ w_all
    blend_unb = P_unb @ w_all
    in_rae = float(rae(y_unb, blend_unb))
    te_unb_rae = float(rae(y_unb, blend_te[unb_idx]))
    w_str = ", ".join(f"K{K}={w:.4f}" for K, w in zip(K_ORDER, w_all))
    print(f"   deploy weights  = {w_str}")
    print(f"   in-sample RAE   = {in_rae:.4f}")
    print(f"   te[unb_idx] RAE = {te_unb_rae:.4f}")
    print(f"   te(513) mean/std= {blend_te.mean():.3f}/{blend_te.std():.3f}")

    # ---- gate ----
    print("\n" + "-" * 78)
    print(f"GATE  PROMOTE<{GATE_PROMOTE:.4f}  MARGINAL_BEAT<{GATE_MARGINAL_REF:.4f}")
    print("-" * 78)
    if pooled_rae_mean_seeds < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif pooled_rae_mean_seeds < GATE_MARGINAL_REF:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"   pooled_RAE mean = {pooled_rae_mean_seeds:.4f}")
    print(f"   verdict         = {verdict}")

    # ---- save artefacts ----
    pred_oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(pred_oof_path, mean_oof.astype(np.float32))
    np.save(te_path, blend_te.astype(np.float32))
    print(f"\n[save] {pred_oof_path}")
    print(f"[save] {te_path}")

    summary = {
        "tag": TAG,
        "method": "BMA_softmax_logML_K_pyramid_18_20_24_28_32_PRE_clean",
        "anchor": "chemprop_aux",
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "K_order": K_ORDER,
        "K18_cols_in_117": [int(x) for x in K18_COLS_IN_117],
        "K28_cols_in_117": [int(x) for x in K28_COLS_IN_117],
        "pyramid_provenance": {
            str(K): {
                "rebuilt": bool(pyramids[K]["rebuilt"]),
                "oof_path": pyramids[K]["oof_path"],
                "te_path": pyramids[K]["te_path"],
                "indiv_oof_rae": indiv_rae[K],
            } for K in K_ORDER
        },
        "rae_anchor_chemprop_aux": rae_anchor,
        "all_data_bma_diagnostic": {
            "weights": [float(w) for w in w_all],
            "logML": [float(x) for x in logml_all],
            "sse_oof": [float(x) for x in sse_all],
        },
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "n_unb": n_unb,
        "n_te": n_test,
        "n_unique_scaffolds_unb": n_unique_scaf,
        "per_seed_results": per_seed,
        "pooled_rae_mean_seeds": pooled_rae_mean_seeds,
        "pooled_rae_std_seeds": pooled_rae_std_seeds,
        "rae_of_mean_of_seed_oofs": final_oof_rae,
        "deploy_weights": [
            {"K": int(K), "w": float(w)} for K, w in zip(K_ORDER, w_all)
        ],
        "in_sample_rae_overfit_bound": in_rae,
        "te_unb_rae_in_sample": te_unb_rae,
        "deploy_te_mean": float(blend_te.mean()),
        "deploy_te_std": float(blend_te.std()),
        "gate_promote": GATE_PROMOTE,
        "gate_marginal_ref": GATE_MARGINAL_REF,
        "verdict": verdict,
        "pre_unblind_clean": True,
        "pred_oof_path": str(pred_oof_path),
        "te_npy_path": str(te_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   pooled_RAE mean (5 seeds)  = {pooled_rae_mean_seeds:.4f}")
    print(f"   verdict                    = {verdict}")
    print(f"   wall                       = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "pooled_rae_mean_seeds",
        "pooled_rae_std_seeds",
        "verdict",
        "deploy_weights",
        "all_data_bma_diagnostic",
        "pred_oof_path",
        "te_npy_path",
    ):
        print(f"  {k}: {res.get(k)}")
