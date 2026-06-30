"""nb2310 -- Dataset-shift features: per-row distance from train manifold.

HYPOTHESIS:
    Explicit dist-from-train features may help LGBM identify high-uncertainty
    rows and re-route residual capacity toward novel-scaffold OOD samples
    (the failure-mode cluster identified in pm04/pm06).  At K=20 the residual
    LGBM has only Mordred/Avalon/AtomPair/MACCS/ChEMBL_kNN+ChempropEmbed
    geometry features -- nothing that *directly* encodes "this compound is
    far from any training example".

PROTOCOL:
    1. For each of the 4139 train and 513 test compounds compute 4 features:
         a. max_tanimoto_to_train         (ECFP4 max)
         b. mean_tanimoto_top5_train      (mean of top-5)
         c. num_train_with_sim_ge_0p5     (count of train neighbours >=0.5)
         d. is_scaffold_novel             (1 iff scaf_train_freq == 0)
       For the 4139 train themselves we compute leave-self-out (so the
       compound's own row in the train-train sim matrix is masked).
    2. Stack the 4 dist-from-train cols onto the K=20 RFE-surviving feature
       matrix used by nb2240 -> K=24.
    3. Fit chemprop_aux+LGBM(MSE) residual K=24, 5 seeds {0,1,7,42,137},
       5-fold KFold cross-fit per seed.  Save oof + te.
    4. Build 5-anchor SLSQP pyramid identical to nb2240 but with nb2310_K24
       replacing nb2240_K20 (chemprop_aux, nb1191, nb503, nb562 unchanged).
    5. Compare pooled_RAE (5 seeds, scaffold 5-fold) vs nb2240 ref 0.4598.
       Gate margin 0.003.  On beat: deep-30 verify (5 canonical + 25 extra).
    6. Save summary + deploy CSV iff beat.

OUTPUTS:
    scripts/nb2310_dataset_shift.py
    data/processed/nb2310_summary.json
    data/processed/nb2310_dist_train_feats_unb.npy   (253, 4) float32
    data/processed/nb2310_dist_train_feats_te.npy    (513, 4) float32
    data/processed/nb2310_dist_train_feats_tr.npy    (4139, 4) float32
    data/processed/nb2310_mean_bag_oof_K24.npy       (253,) float32
    data/processed/te_nb2310_K24.npy                 (513,) float32
    data/processed/te_nb2310.npy                     (513,) float32  (deploy)
    submissions/nb2310_dataset_shift.csv             (only on gate pass)
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
from scipy.optimize import minimize
from sklearn.model_selection import KFold

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2310"

# ------------------------------ stage 1: residual ----------------------------
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

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

# ------------------------------ stage 2: pyramid -----------------------------
GATE_MARGIN = 0.003
NB2240_REF_OOF = 0.4598   # pooled_rae_mean_seeds from nb2240_summary.json

N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]
LB_W_OOF = 0.51
LB_W_TE = 0.49

# nb1191 reconstruction parameters (copied from nb2240)
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

DIST_FEAT_NAMES = [
    "max_tanimoto_to_train",
    "mean_tanimoto_top5_train",
    "num_train_with_sim_ge_0p5",
    "is_scaffold_novel",
]
SIM_THRESHOLD = 0.5
N_TOP_DIST = 5

# ============================================================================
# helpers (copy/specialised from nb2240)
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


# ============================================================================
# DIST-FROM-TRAIN feature computation
# ============================================================================

def _compute_dist_from_train(fp_query, fp_train, scaf_query, scaf_train_counter,
                              mask_self=False, top_k=N_TOP_DIST,
                              sim_thr=SIM_THRESHOLD, block=128):
    """For each row in fp_query compute 4 dist-from-train features against fp_train.

    Args:
        fp_query: (Nq, 2048) uint8/float Morgan FP
        fp_train: (Ntr, 2048) Morgan FP for training compounds
        scaf_query: list of N_q scaffold strings (or None)
        scaf_train_counter: Counter[str] of training-set scaffold frequencies
        mask_self: if True, query is the train set itself, so mask diagonal
                   before computing top-k (leave-self-out).
    Returns:
        feats: (Nq, 4) float32
            col 0: max_tanimoto_to_train
            col 1: mean_tanimoto_top5_train
            col 2: num_train_with_sim>=0.5
            col 3: is_scaffold_novel (scaf_train_freq==0)
    """
    a = fp_query.astype(np.float32)
    b = fp_train.astype(np.float32)
    a_sum = a.sum(axis=1)
    b_sum = b.sum(axis=1)
    n_q = a.shape[0]
    feats = np.zeros((n_q, 4), dtype=np.float32)

    for s in range(0, n_q, block):
        e = min(n_q, s + block)
        inter = a[s:e] @ b.T
        denom = a_sum[s:e, None] + b_sum[None, :] - inter
        denom = np.maximum(denom, 1.0)
        sim = inter / denom
        if mask_self:
            # Diagonal: when query==train and current global idx == column idx, mask.
            # Here block rows are global rows s..e-1, columns are 0..N_tr-1.
            for r_local, r_global in enumerate(range(s, e)):
                if r_global < sim.shape[1]:
                    sim[r_local, r_global] = -1.0
        # Feature 0: max
        feats[s:e, 0] = sim.max(axis=1).clip(0.0, 1.0)
        # Feature 1: mean of top-k
        if sim.shape[1] >= top_k:
            part = np.partition(-sim, kth=top_k - 1, axis=1)[:, :top_k]
            topk = -part
        else:
            topk = sim
        feats[s:e, 1] = topk.mean(axis=1).clip(0.0, 1.0)
        # Feature 2: count of train with sim>=thr (after mask)
        feats[s:e, 2] = (sim >= sim_thr).sum(axis=1).astype(np.float32)

    # Feature 3: is_scaffold_novel
    for i, sc in enumerate(scaf_query):
        feats[i, 3] = 0.0 if (sc is not None and sc in scaf_train_counter and scaf_train_counter[sc] > 0) else 1.0

    return feats


# ============================================================================
# stage 2 utils (SLSQP + rank-stretch -- copied from nb2240)
# ============================================================================

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

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- dataset-shift features (dist-from-train) + K=20 -> K=24")
    print("=" * 78)

    # ---- Load nb2231 K=20 surviving indices ----
    with open(NB2231_SUMMARY) as f:
        nb2231 = json.load(f)
    surviving_K20 = list(nb2231["snapshots"]["20"]["surviving_idx_in_117"])
    surviving_K20_names = list(nb2231["snapshots"]["20"]["surviving_names"])
    assert len(surviving_K20) == 20, f"expected 20 features, got {len(surviving_K20)}"
    print(f"[load] K=20 surviving features from nb2231 (will append 4 dist feats)")

    # ---- Load truth + anchor + test ----
    te = load_test()
    n_test = len(te)
    te_smiles = te["smiles"].astype(str).tolist() if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    te_names = te["name"].values if "name" in te.columns else te["Molecule Name"].values
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique unb_scaffolds={n_unique_scaf}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] chemprop_aux in_RAE = {rae_anchor:.4f} (ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor

    # ---- Load train labels + scaffolds for dist-from-train features ----
    print("\n" + "-" * 78)
    print("STAGE 1a: dist-from-train features (4 cols, leave-self-out on 4139)")
    print("-" * 78)
    tr = load_train()
    tr_smiles_col = "smiles" if "smiles" in tr.columns else "SMILES"
    tr_smiles = tr[tr_smiles_col].astype(str).tolist()
    n_train = len(tr_smiles)
    print(f"[load] n_train = {n_train}")
    t_scaf = time.time()
    tr_scaffolds = [bemis_murcko(s) for s in tr_smiles]
    te_scaffolds = [bemis_murcko(s) for s in te_smiles]
    print(f"[scaf] computed train+test scaffolds wall={time.time()-t_scaf:.1f}s")
    scaf_train_counter = Counter([s for s in tr_scaffolds if s])
    n_unique_scaf_train = len(scaf_train_counter)
    print(f"[scaf] unique scaffolds in train = {n_unique_scaf_train}")

    # Standardised SMILES strings for fingerprinting (consistent with morgan_fp_batch)
    t_fp = time.time()
    fp_train = morgan_fp_batch(tr_smiles)
    fp_test = morgan_fp_batch(te_smiles)
    # Drop train rows with all-zero FP (parse failures) from the *reference* set
    keep_tr = fp_train.sum(axis=1) > 0
    if not keep_tr.all():
        dropped = int((~keep_tr).sum())
        print(f"[fp] dropping {dropped} train rows with zero-FP (parse failures)")
    fp_train_ref = fp_train[keep_tr]
    tr_scaffolds_ref = [tr_scaffolds[i] for i in np.where(keep_tr)[0]]
    print(f"[fp] fp_train={fp_train.shape}  fp_train_ref={fp_train_ref.shape}  fp_test={fp_test.shape}  wall={time.time()-t_fp:.1f}s")

    t_dist = time.time()
    # Compute dist-from-train on:
    #   (a) the 4139 train rows themselves -- mask_self=True (using the FULL train fp as both Q and ref,
    #       which is fine; rows with zero-fp have zero similarity everywhere -> values are honest 0s).
    # We use fp_train (with the bad rows still) for the QUERY so feats_tr has shape (4139,4),
    # but the REFERENCE is fp_train (full) -- we mask diagonal to get leave-self-out.
    feats_tr = _compute_dist_from_train(
        fp_query=fp_train, fp_train=fp_train,
        scaf_query=tr_scaffolds, scaf_train_counter=scaf_train_counter,
        mask_self=True,
    )
    feats_te = _compute_dist_from_train(
        fp_query=fp_test, fp_train=fp_train_ref,
        scaf_query=te_scaffolds, scaf_train_counter=scaf_train_counter,
        mask_self=False,
    )
    feats_unb = feats_te[unb_idx].astype(np.float32)
    print(f"[dist] feats_tr={feats_tr.shape}  feats_te={feats_te.shape}  feats_unb={feats_unb.shape}  wall={time.time()-t_dist:.1f}s")
    print("[dist] test summary (max,mean5,nge0.5,novel) per col:")
    for j, nm in enumerate(DIST_FEAT_NAMES):
        col = feats_te[:, j]
        print(f"   {nm:32s}  min={col.min():.3f}  mean={col.mean():.3f}  median={float(np.median(col)):.3f}  max={col.max():.3f}")
    novel_test = int(feats_te[:, 3].sum())
    novel_unb = int(feats_unb[:, 3].sum())
    print(f"[dist] novel_scaffold count: te={novel_test}/{n_test}  unb={novel_unb}/{n_unb}")

    np.save(DATA_PROCESSED / f"{TAG}_dist_train_feats_tr.npy", feats_tr.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_dist_train_feats_te.npy", feats_te.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_dist_train_feats_unb.npy", feats_unb.astype(np.float32))
    print(f"[save] dist_train_feats_{{tr,te,unb}}.npy")

    # ---- Rebuild 117-col 5-way feature matrix on test ----
    print("\n" + "-" * 78)
    print("STAGE 1b: rebuild 117-col K-tuned feature matrix (test only)")
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
    fp_test_std = morgan_fp_batch(std_test_smiles)
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test_std, fp_pool, k=KNN_K)
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
    feat_dim = X_te_full.shape[1]
    assert feat_dim == 117, f"feat_dim {feat_dim} != 117"
    print(f"[feat] X_te_full = {X_te_full.shape}")

    # Slice to K=20
    X_te_K20 = X_te_full[:, surviving_K20].astype(np.float32)
    # Append 4 dist-from-train features -> K=24
    X_te_K24 = np.concatenate([X_te_K20, feats_te.astype(np.float32)], axis=1)
    X_unb_K24 = X_te_K24[unb_idx]
    K_final = X_te_K24.shape[1]
    assert K_final == 24, f"K_final {K_final} != 24"
    print(f"[feat] X_te_K24 = {X_te_K24.shape}  X_unb_K24 = {X_unb_K24.shape}")
    print(f"[feat] final feature names: {surviving_K20_names + DIST_FEAT_NAMES}")

    # ---- K=24 anchor: chemprop_aux + LGBM residual mean-bag ----
    print("\n" + "-" * 78)
    print(f"K=24 RESIDUAL LGBM  seeds={RESID_SEEDS}  folds={RESID_FOLDS}")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_te_resid = np.zeros((len(RESID_SEEDS), n_test), dtype=np.float64)
    per_seed_rae = []
    feat_importance_sum = np.zeros(K_final, dtype=np.float64)
    n_imp_models = 0
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        resid_oof = _residual_cross_fit_one_seed(X_unb_K24, residual, s)
        per_seed_corrected[i] = anchor + resid_oof
        per_seed_rae.append(float(rae(y_unb, anchor + resid_oof)))
        # Deploy refit on all 253 -> predict residual on full 513
        mdl_full = lgb.LGBMRegressor(**_lgbm_params(s))
        mdl_full.fit(X_unb_K24, residual)
        te_resid_s = mdl_full.predict(X_te_K24).astype(np.float32)
        per_seed_te_resid[i] = te_resid_s
        if hasattr(mdl_full, "feature_importances_"):
            feat_importance_sum += mdl_full.feature_importances_.astype(np.float64)
            n_imp_models += 1
        print(f"   seed={s:3d}: rae_corr={per_seed_rae[-1]:.4f}  wall={time.time()-ts:.1f}s")

    mean_bag_oof_K24 = per_seed_corrected.mean(axis=0)
    mean_bag_te_resid_K24 = per_seed_te_resid.mean(axis=0)
    te_K24_513 = te_anchor_513 + mean_bag_te_resid_K24
    rae_K24_mean_bag = float(rae(y_unb, mean_bag_oof_K24))
    rae_K24_per_seed_mean = float(np.mean(per_seed_rae))
    print(f"\n[K24] per-seed mean RAE = {rae_K24_per_seed_mean:.4f}")
    print(f"[K24] mean-bag RAE      = {rae_K24_mean_bag:.4f}")
    print(f"[K24] anchor in_RAE     = {rae_anchor:.4f}  (delta {rae_K24_mean_bag - rae_anchor:+.4f})")

    all_feat_names = surviving_K20_names + DIST_FEAT_NAMES
    if n_imp_models > 0:
        feat_importance_avg = (feat_importance_sum / n_imp_models)
        order = np.argsort(-feat_importance_avg)
        print("\n[K24] feature importance (avg deploy-refit gain across 5 seeds), top 12:")
        for rk, j in enumerate(order[:12]):
            tag = " <-- DIST" if j >= 20 else ""
            print(f"   {rk:2d}. {all_feat_names[j]:32s}  imp={feat_importance_avg[j]:.1f}{tag}")
        print("[K24] dist-feature importance ranks among K=24:")
        ranks_of_dist = []
        for j in range(20, 24):
            rank = int(np.where(order == j)[0][0])
            ranks_of_dist.append(rank)
            print(f"   {all_feat_names[j]:32s}  rank={rank}  imp={feat_importance_avg[j]:.1f}")
    else:
        feat_importance_avg = None
        ranks_of_dist = None

    oof_K24_path = DATA_PROCESSED / f"{TAG}_mean_bag_oof_K24.npy"
    te_K24_path = DATA_PROCESSED / f"te_{TAG}_K24.npy"
    np.save(oof_K24_path, mean_bag_oof_K24.astype(np.float32))
    np.save(te_K24_path, te_K24_513.astype(np.float32))
    print(f"\n[save] {oof_K24_path}")
    print(f"[save] {te_K24_path}")

    # ============================================================================
    # Stage 2: 5-anchor pyramid SLSQP + rank-stretch  (K=24 replaces K=20)
    # ============================================================================
    print("\n" + "=" * 78)
    print("STAGE 2: 5-ANCHOR PYRAMID  (nb2310_K24 swaps in for nb2240_K20)")
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
        ("nb2310_K24",   mean_bag_oof_K24.astype(np.float64), te_K24_513.astype(np.float64)),
        ("chemprop_aux", chemprop_oof,                        te_chemprop_aux),
        ("nb1191",       nb1191_oof,                          te_nb1191),
        ("nb503",        nb503_oof,                           te_nb503),
        ("nb562",        nb562_oof,                           te_nb562),
    ]
    indiv_rae = {}
    oof_cols, te_cols = [], []
    print("\n[anchors]")
    for disp, oof, te_arr in anchors_list:
        assert oof.shape == (n_unb,), f"{disp} oof {oof.shape}"
        assert te_arr.shape == (n_test,), f"{disp} te {te_arr.shape}"
        r = float(rae(y_unb, oof))
        indiv_rae[disp] = r
        oof_cols.append(oof)
        te_cols.append(te_arr)
        print(f"   {disp:14s} oof_RAE={r:.4f}  te_mean={te_arr.mean():.3f}  te_std={te_arr.std():.3f}")
    P_unb = np.column_stack(oof_cols)
    P_te = np.column_stack(te_cols)
    K_anch = P_unb.shape[1]
    print(f"[stack] P_unb {P_unb.shape}  P_te {P_te.shape}  K={K_anch}")

    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV  kf_seeds={KF_SEEDS}  stretch_grid={STRETCH_GRID}")
    print("-" * 78)
    per_seed = []
    all_oofs = []
    for kf_seed in KF_SEEDS:
        pooled, oof_blend, fw, fs = cv_run_for_seed(P_unb, y_unb, unb_scaffolds, kf_seed)
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": float(pooled),
            "fold_s": [float(x) for x in fs],
            "fold_w_mean": [float(x) for x in np.mean(fw, axis=0)],
        })
        all_oofs.append(oof_blend)
        print(f"   seed={kf_seed}  pooled_RAE={pooled:.4f}  mean_s={np.mean(fs):.3f}  "
              f"w_mean={np.round(np.mean(fw, axis=0), 3).tolist()}")
    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    final_oof_rae = float(rae(y_unb, mean_oof))
    pooled_rae_mean_seeds = float(np.mean([r["pooled_rae"] for r in per_seed]))
    pooled_rae_std_seeds = float(np.std([r["pooled_rae"] for r in per_seed]))
    print(f"\n[cv] mean across 5 seeds  pooled_RAE = {pooled_rae_mean_seeds:.4f} "
          f"(+/- {pooled_rae_std_seeds:.4f})")
    print(f"[cv] RAE of mean-of-seed OOFs        = {final_oof_rae:.4f}")

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
    w_str = ", ".join(f"{disp}={w:.4f}" for (disp, _, _), w in zip(anchors_list, w_deploy))
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

    delta_vs_nb2240 = pooled_rae_mean_seeds - NB2240_REF_OOF
    gate_beat = delta_vs_nb2240 < -GATE_MARGIN
    gate_flat = abs(delta_vs_nb2240) <= GATE_MARGIN
    print("\n" + "-" * 78)
    print(f"GATE EVALUATION  (vs nb2240 ref pooled_rae {NB2240_REF_OOF:.4f}, margin {GATE_MARGIN})")
    print("-" * 78)
    print(f"   nb2310 OOF (5-seed mean) = {pooled_rae_mean_seeds:.4f}")
    print(f"   nb2240 reference         = {NB2240_REF_OOF:.4f}")
    print(f"   delta                    = {delta_vs_nb2240:+.4f}")
    if gate_beat:
        verdict = "BEATS_NB2240"
    elif gate_flat:
        verdict = "FLAT_VS_NB2240"
    else:
        verdict = "HURTS_NB2240"
    print(f"   verdict                  = {verdict}")

    deep30 = None
    if gate_beat:
        print("\n" + "-" * 78)
        print("DEEP-30 VERIFY  (5 canonical + 25 extra seeds)")
        print("-" * 78)
        deep30 = deep_verify_seeds(P_unb, y_unb, unb_scaffolds, n_extra=25)
        print(f"   n_seeds={deep30['n_seeds']}  mean_RAE={deep30['mean_rae']:.4f}  "
              f"std={deep30['std_rae']:.4f}  range=[{deep30['min_rae']:.4f}, "
              f"{deep30['max_rae']:.4f}]")

    te_npy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_npy_path, deploy_te)
    print(f"\n[save] {te_npy_path}")

    sub_csv_path = SUBMISSIONS / f"{TAG}_dataset_shift.csv"
    if gate_beat:
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": deploy_te,
        }).to_csv(sub_csv_path, index=False)
        print(f"[save] {sub_csv_path}  (gate BEATS_NB2240)")
    else:
        print(f"[skip] gate not beat -- no submission CSV written ({verdict})")

    summary = {
        "tag": TAG,
        "method": "nb2171_pyramid_with_K20_plus_4_dist_from_train_feats",
        "anchors": [a[0] for a in anchors_list],
        "anchor_oof_rae_unb": indiv_rae,
        "rae_anchor_chemprop_aux": rae_anchor,
        "k20_surviving_idx_in_117": [int(j) for j in surviving_K20],
        "dist_feat_names": DIST_FEAT_NAMES,
        "dist_feat_te_summary": [
            {"name": DIST_FEAT_NAMES[j],
             "min": float(feats_te[:, j].min()),
             "mean": float(feats_te[:, j].mean()),
             "median": float(np.median(feats_te[:, j])),
             "max": float(feats_te[:, j].max())}
            for j in range(4)
        ],
        "dist_feat_unb_summary": [
            {"name": DIST_FEAT_NAMES[j],
             "min": float(feats_unb[:, j].min()),
             "mean": float(feats_unb[:, j].mean()),
             "median": float(np.median(feats_unb[:, j])),
             "max": float(feats_unb[:, j].max())}
            for j in range(4)
        ],
        "novel_scaffold_count_te": int(feats_te[:, 3].sum()),
        "novel_scaffold_count_unb": int(feats_unb[:, 3].sum()),
        "n_unique_scaffolds_train": int(n_unique_scaf_train),
        "feat_importance_avg_top12_indices": [
            int(j) for j in (
                order[:12].tolist() if feat_importance_avg is not None else []
            )
        ],
        "feat_importance_avg_top12": [
            {"name": all_feat_names[int(j)],
             "imp": float(feat_importance_avg[int(j)]) if feat_importance_avg is not None else None,
             "is_dist": bool(j >= 20)}
            for j in (order[:12].tolist() if feat_importance_avg is not None else [])
        ],
        "feat_importance_dist_ranks": (
            [{"name": all_feat_names[20 + d], "rank": int(ranks_of_dist[d])}
             for d in range(4)] if ranks_of_dist is not None else None
        ),
        "resid_folds": RESID_FOLDS,
        "resid_seeds": RESID_SEEDS,
        "rae_K24_per_seed_mean": rae_K24_per_seed_mean,
        "rae_K24_mean_bag": rae_K24_mean_bag,
        "delta_K24_vs_anchor": rae_K24_mean_bag - rae_anchor,
        "rae_K20_baseline_from_nb2231_per_seed_mean": float(
            nb2231["snapshots"]["20"]["rae_per_seed_mean"]
        ),
        "delta_K24_vs_K20_per_seed_mean": (
            rae_K24_per_seed_mean - float(nb2231["snapshots"]["20"]["rae_per_seed_mean"])
        ),
        "nb2310_oof_K24_path": str(oof_K24_path),
        "te_K24_path": str(te_K24_path),
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "stretch_grid": STRETCH_GRID,
        "n_unb": n_unb,
        "n_te": n_test,
        "n_train": n_train,
        "n_unique_scaffolds_unb": n_unique_scaf,
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
        "compare_nb2240_oof": NB2240_REF_OOF,
        "delta_vs_nb2240": delta_vs_nb2240,
        "gate_margin": GATE_MARGIN,
        "gate_beat_nb2240": bool(gate_beat),
        "gate_flat_vs_nb2240": bool(gate_flat),
        "verdict_vs_nb2240": verdict,
        "deep_30_verify": deep30,
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "te_npy_path": str(te_npy_path),
        "submission_csv": str(sub_csv_path) if gate_beat else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   K=24 mean-bag RAE              = {rae_K24_mean_bag:.4f}")
    print(f"   K=24 per-seed mean RAE         = {rae_K24_per_seed_mean:.4f}")
    print(f"   K=20 baseline (from nb2231)    = {summary['rae_K20_baseline_from_nb2231_per_seed_mean']:.4f}")
    print(f"   delta K=24 vs K=20 per-seed    = {summary['delta_K24_vs_K20_per_seed_mean']:+.4f}")
    print(f"   pyramid pooled RAE (5 seeds)   = {pooled_rae_mean_seeds:.4f}")
    print(f"   delta vs nb2240 (0.4598)       = {delta_vs_nb2240:+.4f}")
    print(f"   verdict                        = {verdict}")
    print(f"   LB band                        = {lb_band_est:.4f}")
    print(f"   wall                           = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "rae_K24_per_seed_mean",
        "rae_K24_mean_bag",
        "delta_K24_vs_anchor",
        "delta_K24_vs_K20_per_seed_mean",
        "pooled_rae_mean_seeds",
        "delta_vs_nb2240",
        "verdict_vs_nb2240",
        "gate_beat_nb2240",
        "novel_scaffold_count_te",
        "novel_scaffold_count_unb",
        "deploy_weights",
        "deploy_s",
        "lb_band_estimate",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
