"""nb2560 -- 100-seed nb2240 deep bag vs cycle-160 30-seed standard.

CONTEXT:
    Cycle-160 elevated deep-30 as the canonical gate-decision standard
    after nb2060 5-seed std 0.00087 was found severely under-dispersed
    vs 30-seed std 0.00408 (4.7x ratio).  Three independent deep-30
    runs (nb1191, nb2060, nb2095) converged to 0.4718-0.4720 (band
    < 0.0003), establishing the canonical post-hoc-blend ceiling on
    chemprop_aux anchor at n=253.

    nb2240 swapped the K=28 anchor for K=20 RFE-pruned and broke the
    ceiling: deep-30 mean 0.4601 +/- 0.0017 (cached summary), with
    K=20 mean-bag built from 5 residual-LGBM seeds {0, 1, 7, 42, 137}.

    THIS SCRIPT TESTS:  is 0.4601 the true ceiling, or does scaling
    residual-LGBM seeds 5 -> 100 push it lower?  If 100-seed mean
    drops below 0.4570 we have a fresh ceiling and PROMOTE candidate;
    if it stays at ~0.46 we have confirmed the K=20 anchor variance
    is already saturated.

PROTOCOL:
    1. For each residual seed s in 0..99:
         a. KFold(5, random_state=s) cross-fit LGBM(MSE) on the K=20
            feature slice -> resid_oof_s (253) and te_resid_s (513).
       NOTE: This produces 100 independent single-seed K=20 anchors,
       NOT a 100-seed mean-bag.  The "100-seed mean" is the average
       pyramid pooled RAE across the 100 independent anchors, which
       directly mirrors the deep-30 protocol (each kf_seed gives one
       independent pooled RAE, average them).
    2. For each seed s: build the 5-anchor pyramid (single-seed
       K=20 anchor + chemprop_aux + nb1191 + nb503 + nb562) and run
       a single 5-fold scaffold-CV at kf_seed=1001 with SLSQP +
       rank-stretch (single kf_seed to keep wall < a few minutes at
       100 seeds).
    3. Aggregate: 100 pyramid pooled RAE values -> mean, std, min, max.
       Compare to:
         - cycle-163 nb2095 deep-30 mean 0.4720 (PRE-pyramid ceiling)
         - nb2240 deep-30 mean 0.4601 (nb2095's K=20 swap variant)
       Compute under-dispersion ratio vs 30-seed and gate.

GATES:
    100-seed mean < 0.4570 -> PROMOTE
    < 0.4601 -> MARGINAL_BEAT
    else FAIL

OUTPUTS:
    scripts/nb2560_100seed_nb2240_bag.py
    data/processed/nb2560_summary.json
    data/processed/nb2560_mean_bag_oof_K20_100seed.npy   (253,) float32
    data/processed/te_nb2560_K20_100seed.npy             (513,) float32
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

TAG = "nb2560"

# ============================================================================
# CONFIG
# ============================================================================
N_RESID_SEEDS = 100
RESID_SEEDS = list(range(N_RESID_SEEDS))
RESID_FOLDS = 5
KF_SEED_FOR_PYRAMID = 1001
N_FOLDS = 5
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]

# Reference points (from cached summaries)
NB2095_DEEP30_MEAN = 0.4720        # PRE-pyramid co-converged ceiling (K=28)
NB2240_DEEP30_MEAN = 0.4601        # K=20 swap deep-30 mean
NB2240_DEEP30_STD = 0.0017
NB2095_5SEED_STD = 0.0009          # under-dispersed 5-seed std at K=28
NB2240_5SEED_STD = 0.0009          # under-dispersed 5-seed std at K=20 (similar pattern)

PROMOTE_THR = 0.4570
MARGINAL_THR = 0.4601

LB_W_OOF = 0.51
LB_W_TE = 0.49

# Paths to cached anchors / feature slices
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
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

# nb1191 reconstruction (copied from nb2240/nb2171)
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
# helpers
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
    print(f"{TAG} -- 100-seed nb2240 deep bag vs cycle-160 30-seed standard")
    print("=" * 78)

    # ---- Load nb2231 K=20 surviving indices ----
    with open(NB2231_SUMMARY) as f:
        nb2231 = json.load(f)
    surviving_K20 = list(nb2231["snapshots"]["20"]["surviving_idx_in_117"])
    assert len(surviving_K20) == 20

    # ---- Load truth + anchor ----
    te = load_test()
    n_test = len(te)
    te_smiles = te["smiles"].astype(str).tolist() if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    residual = y_unb - anchor

    print(f"[load] n_test={n_test}  n_unb={n_unb}  scaffolds={n_unique_scaf}")
    print(f"[load] chemprop_aux in_RAE = {rae_anchor:.4f}")

    # ---- Rebuild 117-col 5-way feature matrix ----
    for p in (NB1352_SUMMARY, NB1392_SUMMARY, NB1484_SUMMARY,
              NB1523_SUMMARY, NB1524_SUMMARY, NB1541_SUMMARY):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")
    sum_1352 = json.load(open(NB1352_SUMMARY))
    sum_1392 = json.load(open(NB1392_SUMMARY))
    sum_1484 = json.load(open(NB1484_SUMMARY))
    sum_1523 = json.load(open(NB1523_SUMMARY))
    sum_1524 = json.load(open(NB1524_SUMMARY))
    sum_1541 = json.load(open(NB1541_SUMMARY))

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
    X_mord_te = _load_mordred_test(n_test_expected=n_test)[:, top_mord_col_idx].astype(np.float32)
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
        std_test_smiles.append(Chem.MolToSmiles(m) if m is not None else "")
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
    assert X_te_full.shape[1] == 117

    X_te_K20 = X_te_full[:, surviving_K20].astype(np.float32)
    X_unb_K20 = X_te_K20[unb_idx]
    print(f"[feat] X_unb_K20={X_unb_K20.shape}  X_te_K20={X_te_K20.shape}")

    # ---- Load anchors for pyramid ----
    print(f"\n[anchors] loading nb1191/chemprop_aux/nb503/nb562 ...")
    nb1191_oof = reconstruct_nb1191_oof(n_unb)
    chemprop_oof = np.load(DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy").astype(np.float64)
    nb503_oof = np.load(DATA_PROCESSED / "nb503_pred_oof.npy").astype(np.float64)
    nb562_oof = np.load(DATA_PROCESSED / "nb562_pred_oof.npy").astype(np.float64)
    te_nb1191 = np.load(DATA_PROCESSED / "te_nb1191.npy").astype(np.float64)
    te_nb503 = np.load(DATA_PROCESSED / "te_nb503.npy").astype(np.float64)
    te_nb562 = np.load(DATA_PROCESSED / "te_nb562.npy").astype(np.float64)

    # ============================================================================
    # MAIN LOOP: 100 single-seed K=20 anchors -> pyramid pooled RAE each
    # ============================================================================
    print("\n" + "=" * 78)
    print(f"100-SEED LOOP  resid_seeds={N_RESID_SEEDS}  kf_seed={KF_SEED_FOR_PYRAMID}")
    print("=" * 78)

    per_seed_rae_pyramid = np.full(N_RESID_SEEDS, np.nan, dtype=np.float64)
    per_seed_rae_K20_anchor = np.full(N_RESID_SEEDS, np.nan, dtype=np.float64)
    per_seed_mean_s = np.full(N_RESID_SEEDS, np.nan, dtype=np.float64)
    per_seed_w_mean = np.zeros((N_RESID_SEEDS, 5), dtype=np.float64)

    # Also accumulate mean-bag K=20 OOF + te-residual across all 100 seeds
    accum_corrected_oof = np.zeros(n_unb, dtype=np.float64)
    accum_te_resid = np.zeros(n_test, dtype=np.float64)

    t_loop = time.time()
    for i, s in enumerate(RESID_SEEDS):
        # 1) Single-seed K=20 residual cross-fit
        resid_oof_s = _residual_cross_fit_one_seed(X_unb_K20, residual, s)
        K20_oof_s = anchor + resid_oof_s
        per_seed_rae_K20_anchor[i] = float(rae(y_unb, K20_oof_s))
        accum_corrected_oof += K20_oof_s

        # 2) Deploy refit on all 253 -> predict residual on full 513
        te_resid_s = _train_full_then_predict_te(X_unb_K20, residual, X_te_K20, s)
        K20_te_s = te_anchor_513 + te_resid_s
        accum_te_resid += te_resid_s

        # 3) Build 5-anchor pyramid for this seed
        P_unb_s = np.column_stack([
            K20_oof_s.astype(np.float64),
            chemprop_oof,
            nb1191_oof,
            nb503_oof,
            nb562_oof,
        ])

        # 4) Single-kf scaffold-CV pyramid SLSQP + rank-stretch
        pooled, _oof, fw, fs = cv_run_for_seed(
            P_unb_s, y_unb, unb_scaffolds, KF_SEED_FOR_PYRAMID,
        )
        per_seed_rae_pyramid[i] = pooled
        per_seed_mean_s[i] = float(np.mean(fs))
        per_seed_w_mean[i] = np.mean(fw, axis=0)

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t_loop
            eta = elapsed / (i + 1) * (N_RESID_SEEDS - i - 1)
            print(
                f"  seed {s:3d} ({i+1:3d}/{N_RESID_SEEDS}): "
                f"K20={per_seed_rae_K20_anchor[i]:.4f}  "
                f"pyramid={per_seed_rae_pyramid[i]:.4f}  "
                f"s={per_seed_mean_s[i]:.3f}  "
                f"elapsed={elapsed:.0f}s  eta={eta:.0f}s"
            )

    # ---- Aggregate ----
    mean_bag_oof_K20_100 = accum_corrected_oof / N_RESID_SEEDS
    mean_bag_te_resid_K20_100 = accum_te_resid / N_RESID_SEEDS
    te_K20_513_100 = te_anchor_513 + mean_bag_te_resid_K20_100
    rae_K20_meanbag_100 = float(rae(y_unb, mean_bag_oof_K20_100))

    # Save mean-bag artefacts
    oof_path = DATA_PROCESSED / f"{TAG}_mean_bag_oof_K20_100seed.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}_K20_100seed.npy"
    np.save(oof_path, mean_bag_oof_K20_100.astype(np.float32))
    np.save(te_path, te_K20_513_100.astype(np.float32))
    print(f"\n[save] {oof_path}")
    print(f"[save] {te_path}")

    mean100 = float(np.mean(per_seed_rae_pyramid))
    std100 = float(np.std(per_seed_rae_pyramid))
    min100 = float(np.min(per_seed_rae_pyramid))
    max100 = float(np.max(per_seed_rae_pyramid))

    # 30-seed sub-comparison: average over first 30 seeds of this run
    mean_first30 = float(np.mean(per_seed_rae_pyramid[:30]))
    std_first30 = float(np.std(per_seed_rae_pyramid[:30]))

    # Under-dispersion ratio (vs nb2240 deep-30, computed against vendor reference)
    under_disp_ratio_nb2240 = (
        NB2240_DEEP30_STD / std100 if std100 > 0 else float("inf")
    )
    # vs intra-run 5-seed under-dispersion (re-check whether the "lucky-bag"
    # under-dispersion pattern that motivated cycle-160 still holds at 100)
    std_first5 = float(np.std(per_seed_rae_pyramid[:5]))
    under_disp_ratio_internal = std100 / std_first5 if std_first5 > 0 else float("inf")

    # ---- Gate ----
    if mean100 < PROMOTE_THR:
        verdict = "PROMOTE"
    elif mean100 < MARGINAL_THR:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"

    delta_vs_nb2095 = mean100 - NB2095_DEEP30_MEAN
    delta_vs_nb2240 = mean100 - NB2240_DEEP30_MEAN

    print("\n" + "=" * 78)
    print("100-SEED AGGREGATE")
    print("=" * 78)
    print(f"  mean   = {mean100:.4f}")
    print(f"  std    = {std100:.4f}")
    print(f"  min    = {min100:.4f}")
    print(f"  max    = {max100:.4f}")
    print(f"  range  = {max100 - min100:.4f}")
    print(f"  meanbag K=20 RAE = {rae_K20_meanbag_100:.4f}")
    print()
    print(f"  first-30 mean = {mean_first30:.4f}   first-30 std = {std_first30:.4f}")
    print(f"  first-5 std   = {std_first5:.4f}")
    print()
    print(f"  vs nb2095 deep-30 (0.4720): delta = {delta_vs_nb2095:+.4f}")
    print(f"  vs nb2240 deep-30 (0.4601): delta = {delta_vs_nb2240:+.4f}")
    print(f"  under-disp 100-std / 5-std = {under_disp_ratio_internal:.2f}x")
    print(f"  nb2240-30std / 100-std     = {under_disp_ratio_nb2240:.2f}x")
    print()
    print(f"  GATE:  PROMOTE<{PROMOTE_THR}  MARGINAL<{MARGINAL_THR}")
    print(f"  VERDICT = {verdict}")

    # ---- LB band on the meanbag (informative) ----
    lb_band = LB_W_OOF * mean100 + LB_W_TE * rae_K20_meanbag_100
    print(f"\n[LB-band] {LB_W_OOF}*100seed_mean + {LB_W_TE}*K20_meanbag = {lb_band:.4f}")

    # ---- Build summary ----
    summary = {
        "tag": TAG,
        "method": "100seed_nb2240_K20_residual_bag_with_single_kf_scaffold_cv_pyramid",
        "n_resid_seeds": N_RESID_SEEDS,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "kf_seed_for_pyramid": KF_SEED_FOR_PYRAMID,
        "n_folds": N_FOLDS,
        "stretch_grid": STRETCH_GRID,
        "n_unb": n_unb,
        "n_te": n_test,
        "n_unique_scaffolds": n_unique_scaf,
        "rae_chemprop_aux_anchor": rae_anchor,
        "k20_n_features": 20,
        "anchors_used": ["nb2240_K20_singleseed", "chemprop_aux", "nb1191", "nb503", "nb562"],

        "per_seed_rae_pyramid": [float(x) for x in per_seed_rae_pyramid],
        "per_seed_rae_K20_anchor": [float(x) for x in per_seed_rae_K20_anchor],
        "per_seed_mean_s": [float(x) for x in per_seed_mean_s],
        "per_seed_w_mean": per_seed_w_mean.tolist(),

        "mean_100": mean100,
        "std_100":  std100,
        "min_100":  min100,
        "max_100":  max100,

        "mean_first30": mean_first30,
        "std_first30":  std_first30,
        "std_first5":   std_first5,

        "rae_K20_meanbag_100": rae_K20_meanbag_100,

        "reference_nb2095_deep30_mean": NB2095_DEEP30_MEAN,
        "reference_nb2240_deep30_mean": NB2240_DEEP30_MEAN,
        "reference_nb2240_deep30_std":  NB2240_DEEP30_STD,
        "delta_vs_nb2095_deep30": delta_vs_nb2095,
        "delta_vs_nb2240_deep30": delta_vs_nb2240,
        "under_disp_ratio_100std_over_5std": under_disp_ratio_internal,
        "under_disp_ratio_nb2240_30std_over_100std": under_disp_ratio_nb2240,

        "promote_threshold": PROMOTE_THR,
        "marginal_threshold": MARGINAL_THR,
        "verdict": verdict,

        "lb_band_estimate": lb_band,
        "lb_band_w_oof": LB_W_OOF,
        "lb_band_w_te":  LB_W_TE,

        "oof_path": str(oof_path),
        "te_path":  str(te_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"  100-seed pyramid mean   = {mean100:.4f}  +/- {std100:.4f}")
    print(f"  K=20 meanbag RAE        = {rae_K20_meanbag_100:.4f}")
    print(f"  delta vs nb2240 deep-30 = {delta_vs_nb2240:+.4f}")
    print(f"  delta vs nb2095 deep-30 = {delta_vs_nb2095:+.4f}")
    print(f"  under-disp 100/5        = {under_disp_ratio_internal:.2f}x")
    print(f"  VERDICT                 = {verdict}")
    print(f"  wall                    = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_100", "std_100", "min_100", "max_100",
        "mean_first30", "std_first30", "std_first5",
        "rae_K20_meanbag_100",
        "delta_vs_nb2240_deep30", "delta_vs_nb2095_deep30",
        "under_disp_ratio_100std_over_5std",
        "verdict",
        "lb_band_estimate",
    ):
        print(f"  {k}: {res.get(k)}")
