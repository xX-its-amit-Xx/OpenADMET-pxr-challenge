"""nb2272 -- Bag-of-bags expansion of nb2240 K=20 pyramid.

CONTEXT:
    nb2240 (5-anchor pyramid: nb2240_K20 + chemprop_aux + nb1191 + nb503 +
    nb562) shipped pooled RAE 0.4598 across 5 kf_seeds, with the K=20
    residual anchor itself coming from a 5-seed mean-bag over LGBM seeds
    {0, 1, 7, 42, 137}.

    Per cycle 113-118 BoB pattern: pooling more inner seeds can reduce
    bagging variance and tighten the residual estimate without changing
    the underlying feature/model class.  We expand from 5 to 25 inner seeds
    arranged as 5 outer x 5 inner: each outer base seed b in
    {0, 1, 7, 42, 137} spawns 5 daughter seeds offset by 1000 (b,
    1000+b, 2000+b, 3000+b, 4000+b), giving 25 LGBM fits.

PROTOCOL:
    1. Reproduce nb2240 stage-1 setup: load chemprop_aux anchor, build the
       117-col 5-way feature matrix, slice to nb2231 K=20 surviving indices.
    2. For each of 25 inner seeds:
         - 5-fold KFold residual cross-fit on 253 -> OOF residual
         - deploy refit on all 253 -> residual on 513
       Pool via MEDIAN of 25 corrected predictions (per-row median of
       anchor + residual_oof_s for OOF; anchor + residual_te_s for te).
    3. Build the 5-anchor pyramid: nb2272_K20_BoB swaps in for nb2240_K20.
       Run scaffold 5-fold CV on the 253 across the SAME 5 kf_seeds
       {1001..1005} as nb2240 for an apples-to-apples comparison.
    4. Compare vs nb2240 pooled RAE 0.4598; gate margin 0.003.
       BEATS if pooled_rae_mean_seeds < 0.4598 - 0.003 = 0.4568.

Outputs:
    scripts/nb2272_bag_of_bags.py
    data/processed/nb2272_summary.json
    data/processed/nb2272_median_bag_oof_K20_BoB.npy   (253,) float32
    data/processed/te_nb2272_K20_BoB.npy               (513,) float32
    data/processed/te_nb2272.npy                       (513,) float32  (deploy)
    submissions/nb2272_bag_of_bags.csv                 (only on gate pass)
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

TAG = "nb2272"

# ------------------------------ stage 1: residual ----------------------------
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
RESID_FOLDS = 5
# Bag-of-bags: 5 outer base seeds x 5 inner offsets = 25 LGBM seeds.
OUTER_SEEDS = [0, 1, 7, 42, 137]
INNER_OFFSETS = [0, 1000, 2000, 3000, 4000]
RESID_SEEDS = [b + off for off in INNER_OFFSETS for b in OUTER_SEEDS]  # 25 distinct ints
assert len(set(RESID_SEEDS)) == 25, "Inner seed collision"

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

# nb1191 reconstruction parameters (copied from nb2171/nb2240)
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


# ============================================================================
# helpers (copied verbatim from nb2240; isolated here so this script stands
# alone if nb2240 is later refactored).
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
# stage 2 utils (SLSQP + rank-stretch)
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
    print(f"{TAG} -- bag-of-bags (5 outer x 5 inner = 25 seeds) on nb2240 K=20 pyramid")
    print("=" * 78)
    print(f"[seeds] outer={OUTER_SEEDS}")
    print(f"[seeds] inner_offsets={INNER_OFFSETS}")
    print(f"[seeds] total={len(RESID_SEEDS)}  first={RESID_SEEDS[:5]}  last={RESID_SEEDS[-5:]}")

    # ---- Load nb2231 K=20 surviving indices ----
    with open(NB2231_SUMMARY) as f:
        nb2231 = json.load(f)
    surviving_K20 = list(nb2231["snapshots"]["20"]["surviving_idx_in_117"])
    surviving_K20_names = list(nb2231["snapshots"]["20"]["surviving_names"])
    assert len(surviving_K20) == 20, f"expected 20 features, got {len(surviving_K20)}"
    print(f"[load] K=20 surviving features ({len(surviving_K20)})")

    # ---- Load truth + anchor ----
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
    print(f"[scaffold] unique={n_unique_scaf}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] chemprop_aux in_RAE = {rae_anchor:.4f} (ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor

    # ---- Rebuild 117-col feature matrix on BOTH unb and te (verbatim nb2240) ----
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

    # Full 117-col te matrix
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
    X_te_K20 = X_te_full[:, surviving_K20].astype(np.float32)
    X_unb_K20 = X_te_K20[unb_idx]
    print(f"[feat] X_unb_K20 = {X_unb_K20.shape}  X_te_K20 = {X_te_K20.shape}")

    # ---- Build K=20 BoB anchor: chemprop_aux + LGBM residual median-pool over 25 seeds ----
    print("\n" + "-" * 78)
    print(f"STAGE 1: K=20 BAG-OF-BAGS RESIDUAL  n_seeds={len(RESID_SEEDS)}  folds={RESID_FOLDS}")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_te_corr = np.zeros((len(RESID_SEEDS), n_test), dtype=np.float64)
    per_seed_rae = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        resid_oof = _residual_cross_fit_one_seed(X_unb_K20, residual, s)
        per_seed_corrected[i] = anchor + resid_oof
        rae_i = float(rae(y_unb, anchor + resid_oof))
        per_seed_rae.append(rae_i)
        te_resid_s = _train_full_then_predict_te(X_unb_K20, residual, X_te_K20, s)
        per_seed_te_corr[i] = te_anchor_513 + te_resid_s
        print(f"   [{i+1:2d}/25] seed={s:5d}: rae_corr={rae_i:.4f}  wall={time.time()-ts:.1f}s")

    # ---- Pool via MEDIAN of 25 corrected predictions (vs mean in nb2240) ----
    median_bag_oof_K20_BoB = np.median(per_seed_corrected, axis=0)  # 253
    median_bag_te_K20_BoB = np.median(per_seed_te_corr, axis=0)     # 513
    mean_bag_oof_K20_ref = per_seed_corrected.mean(axis=0)
    rae_K20_BoB_median = float(rae(y_unb, median_bag_oof_K20_BoB))
    rae_K20_BoB_mean = float(rae(y_unb, mean_bag_oof_K20_ref))
    rae_K20_per_seed_mean = float(np.mean(per_seed_rae))
    rae_K20_per_seed_std = float(np.std(per_seed_rae))
    print(f"\n[BoB] per-seed mean RAE        = {rae_K20_per_seed_mean:.4f} "
          f"(+/- {rae_K20_per_seed_std:.4f}; n=25)")
    print(f"[BoB] 25-seed MEDIAN-pool RAE  = {rae_K20_BoB_median:.4f}")
    print(f"[BoB] 25-seed MEAN-pool RAE    = {rae_K20_BoB_mean:.4f}")
    print(f"[BoB] anchor in_RAE            = {rae_anchor:.4f}  "
          f"(delta_median {rae_K20_BoB_median - rae_anchor:+.4f})")

    oof_BoB_path = DATA_PROCESSED / f"{TAG}_median_bag_oof_K20_BoB.npy"
    te_BoB_path = DATA_PROCESSED / f"te_{TAG}_K20_BoB.npy"
    np.save(oof_BoB_path, median_bag_oof_K20_BoB.astype(np.float32))
    np.save(te_BoB_path, median_bag_te_K20_BoB.astype(np.float32))
    print(f"[save] {oof_BoB_path}")
    print(f"[save] {te_BoB_path}")

    # ============================================================================
    # Stage 2: 5-anchor pyramid SLSQP + rank-stretch  (BoB K=20 replaces nb2240_K20)
    # ============================================================================
    print("\n" + "=" * 78)
    print("STAGE 2: 5-ANCHOR PYRAMID  (nb2272_K20_BoB swaps in for nb2240_K20)")
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
        ("nb2272_K20_BoB", median_bag_oof_K20_BoB.astype(np.float64), median_bag_te_K20_BoB.astype(np.float64)),
        ("chemprop_aux",   chemprop_oof,                              te_chemprop_aux),
        ("nb1191",         nb1191_oof,                                te_nb1191),
        ("nb503",          nb503_oof,                                 te_nb503),
        ("nb562",          nb562_oof,                                 te_nb562),
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
        print(f"   {disp:16s} oof_RAE={r:.4f}  te_mean={te_arr.mean():.3f}  te_std={te_arr.std():.3f}")
    P_unb = np.column_stack(oof_cols)
    P_te = np.column_stack(te_cols)
    K_anch = P_unb.shape[1]
    print(f"[stack] P_unb {P_unb.shape}  P_te {P_te.shape}  K={K_anch}")

    # ---- Scaffold 5-fold CV across 5 seeds (same kf_seeds as nb2240) ----
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

    # ---- Gate vs nb2240 ref ----
    delta_vs_nb2240 = pooled_rae_mean_seeds - NB2240_REF_OOF
    gate_beat = delta_vs_nb2240 < -GATE_MARGIN
    gate_flat = abs(delta_vs_nb2240) <= GATE_MARGIN
    print("\n" + "-" * 78)
    print("GATE EVALUATION  (vs nb2240 ref pooled_rae 0.4598, margin 0.003)")
    print("-" * 78)
    print(f"   nb2272 OOF (5-seed mean) = {pooled_rae_mean_seeds:.4f}")
    print(f"   nb2240 reference         = {NB2240_REF_OOF:.4f}")
    print(f"   delta                    = {delta_vs_nb2240:+.4f}")
    if gate_beat:
        verdict = "BEATS_NB2240"
    elif gate_flat:
        verdict = "FLAT_VS_NB2240"
    else:
        verdict = "HURTS_NB2240"
    print(f"   verdict                  = {verdict}")

    # ---- Always save te artefact ----
    te_npy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_npy_path, deploy_te)
    print(f"\n[save] {te_npy_path}")

    sub_csv_path = SUBMISSIONS / f"{TAG}_bag_of_bags.csv"
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
        "method": "bag_of_bags_5_outer_x_5_inner_median_pool_over_nb2240_K20_pyramid",
        "anchors": [a[0] for a in anchors_list],
        "anchor_oof_rae_unb": indiv_rae,
        "rae_anchor_chemprop_aux": rae_anchor,
        "k20_surviving_idx_in_117": [int(j) for j in surviving_K20],
        "k20_surviving_names": surviving_K20_names,
        "outer_seeds": OUTER_SEEDS,
        "inner_offsets": INNER_OFFSETS,
        "resid_seeds": RESID_SEEDS,
        "n_resid_seeds": len(RESID_SEEDS),
        "resid_folds": RESID_FOLDS,
        "rae_K20_per_seed_mean": rae_K20_per_seed_mean,
        "rae_K20_per_seed_std": rae_K20_per_seed_std,
        "rae_K20_BoB_median_pool": rae_K20_BoB_median,
        "rae_K20_BoB_mean_pool": rae_K20_BoB_mean,
        "delta_K20_median_vs_anchor": rae_K20_BoB_median - rae_anchor,
        "median_bag_oof_path": str(oof_BoB_path),
        "te_BoB_path": str(te_BoB_path),
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "stretch_grid": STRETCH_GRID,
        "n_unb": n_unb,
        "n_te": n_test,
        "n_unique_scaffolds": n_unique_scaf,
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
    print(f"   25-seed median-pool RAE        = {rae_K20_BoB_median:.4f}")
    print(f"   25-seed mean-pool RAE          = {rae_K20_BoB_mean:.4f}")
    print(f"   per-seed mean RAE (+/- std)    = {rae_K20_per_seed_mean:.4f} +/- {rae_K20_per_seed_std:.4f}")
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
        "n_resid_seeds",
        "rae_K20_per_seed_mean",
        "rae_K20_BoB_median_pool",
        "rae_K20_BoB_mean_pool",
        "pooled_rae_mean_seeds",
        "delta_vs_nb2240",
        "verdict_vs_nb2240",
        "gate_beat_nb2240",
        "deploy_weights",
        "deploy_s",
        "lb_band_estimate",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
