"""nb2584 -- Quantile median voting across 100 nb2240 K=20 seeds.

NEW PARADIGM vs nb2560 (100-seed mean bag):
    Instead of MEAN-ing the 100 single-seed K=20 OOF and te predictions
    (which nb2560 already does and which still leaves us at 0.4665 +/- 0.0027
    deep-30 ceiling), this script uses the MEDIAN across the 100 seeds
    on a per-row basis.

    Hypothesis: median is robust to seed-outlier predictions that pull
    the mean toward the tails. If seed-conditional rank order is more
    stable than seed-conditional value, the median should de-noise the
    per-row variance without losing rank-information.

PROTOCOL:
    1. Rebuild K=20 feature slice on (253 unb, 513 te).
    2. For each residual seed s in 0..99:
         a. KFold(5, random_state=s) cross-fit LGBM(MSE) on K=20 -> resid_oof_s (253)
         b. Deploy refit on 253 -> predict on 513 -> te_resid_s (513)
       Stack 100 -> oof_matrix (100, 253), te_matrix (100, 513).
    3. MEDIAN across seed axis (axis=0):
         median_resid_oof = np.median(oof_matrix, axis=0)        # (253,)
         median_te_resid  = np.median(te_matrix,  axis=0)        # (513,)
       Anchor on chemprop_aux:
         median_K20_oof = anchor[unb_idx] + median_resid_oof
         median_K20_te  = te_anchor_513   + median_te_resid
    4. Build 5-anchor pyramid {median_K20, chemprop_aux, nb1191, nb503, nb562}.
    5. Single 5-fold scaffold-CV (kf_seed=1001) with SLSQP + rank-stretch.

GATES:
    mean_rae < 0.4570 -> PROMOTE
    < 0.4601 -> MARGINAL_BEAT
    else FAIL

OUTPUTS:
    scripts/nb2584_quantile_median_vote.py
    data/processed/nb2584_summary.json
    data/processed/nb2584_median_oof_K20_100seed.npy   (253,) float32
    data/processed/te_nb2584_median_K20_100seed.npy    (513,) float32
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

TAG = "nb2584"

# ============================================================================
# CONFIG
# ============================================================================
N_RESID_SEEDS = 100
RESID_SEEDS = list(range(N_RESID_SEEDS))
RESID_FOLDS = 5
KF_SEED_FOR_PYRAMID = 1001
N_FOLDS = 5
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]

# Reference points
NB2095_DEEP30_MEAN = 0.4720
NB2240_DEEP30_MEAN = 0.4601
NB2560_100SEED_MEAN_REF = 0.4665  # nb2560 cycle 174 100-seed mean (pyramid pooled)

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

# nb1191 reconstruction
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
# helpers (copied from nb2560)
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
    return mdl.predict(X_te).astype(np.float64)


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
    print(f"{TAG} -- Quantile median voting across 100 nb2240 K=20 seeds")
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

    # ============================================================================
    # 100-SEED LOOP: collect full (100, 253) and (100, 513) residual matrices
    # ============================================================================
    print("\n" + "=" * 78)
    print(f"100-SEED LOOP  collect oof_matrix & te_matrix  resid_seeds={N_RESID_SEEDS}")
    print("=" * 78)

    oof_matrix = np.full((N_RESID_SEEDS, n_unb), np.nan, dtype=np.float64)
    te_matrix = np.full((N_RESID_SEEDS, n_test), np.nan, dtype=np.float64)
    per_seed_rae_K20 = np.full(N_RESID_SEEDS, np.nan, dtype=np.float64)

    t_loop = time.time()
    for i, s in enumerate(RESID_SEEDS):
        resid_oof_s = _residual_cross_fit_one_seed(X_unb_K20, residual, s)
        K20_oof_s = anchor + resid_oof_s
        oof_matrix[i] = K20_oof_s
        per_seed_rae_K20[i] = float(rae(y_unb, K20_oof_s))

        te_resid_s = _train_full_then_predict_te(X_unb_K20, residual, X_te_K20, s)
        K20_te_s = te_anchor_513 + te_resid_s
        te_matrix[i] = K20_te_s

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t_loop
            eta = elapsed / (i + 1) * (N_RESID_SEEDS - i - 1)
            print(
                f"  seed {s:3d} ({i+1:3d}/{N_RESID_SEEDS}): "
                f"K20={per_seed_rae_K20[i]:.4f}  "
                f"elapsed={elapsed:.0f}s  eta={eta:.0f}s"
            )

    # ============================================================================
    # MEDIAN ACROSS 100 SEEDS (per-row robust aggregation)
    # ============================================================================
    print("\n" + "=" * 78)
    print("MEDIAN AGGREGATION  (axis=0 across 100 seeds, per-row)")
    print("=" * 78)
    median_K20_oof = np.median(oof_matrix, axis=0).astype(np.float64)
    median_K20_te = np.median(te_matrix, axis=0).astype(np.float64)
    rae_median_K20 = float(rae(y_unb, median_K20_oof))

    # Comparison: mean across the same 100 seeds (sanity vs nb2560)
    mean_K20_oof = oof_matrix.mean(axis=0).astype(np.float64)
    mean_K20_te = te_matrix.mean(axis=0).astype(np.float64)
    rae_mean_K20 = float(rae(y_unb, mean_K20_oof))

    print(f"  K20 median-bag RAE  = {rae_median_K20:.4f}")
    print(f"  K20 mean-bag   RAE  = {rae_mean_K20:.4f}  (nb2560 reference)")
    print(f"  delta median - mean = {rae_median_K20 - rae_mean_K20:+.4f}")
    print(f"  median(K20_oof) mean/std = {median_K20_oof.mean():.3f}/{median_K20_oof.std():.3f}")
    print(f"  mean(K20_oof)   mean/std = {mean_K20_oof.mean():.3f}/{mean_K20_oof.std():.3f}")

    # Save median artefacts
    oof_path = DATA_PROCESSED / f"{TAG}_median_oof_K20_100seed.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}_median_K20_100seed.npy"
    np.save(oof_path, median_K20_oof.astype(np.float32))
    np.save(te_path, median_K20_te.astype(np.float32))
    print(f"\n[save] {oof_path}")
    print(f"[save] {te_path}")

    # ============================================================================
    # 5-ANCHOR PYRAMID using MEDIAN K=20 anchor
    # ============================================================================
    print("\n" + "=" * 78)
    print("5-ANCHOR PYRAMID  (median_K20 + chemprop_aux + nb1191 + nb503 + nb562)")
    print("=" * 78)

    nb1191_oof = reconstruct_nb1191_oof(n_unb)
    chemprop_oof = np.load(DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy").astype(np.float64)
    nb503_oof = np.load(DATA_PROCESSED / "nb503_pred_oof.npy").astype(np.float64)
    nb562_oof = np.load(DATA_PROCESSED / "nb562_pred_oof.npy").astype(np.float64)

    te_nb1191 = np.load(DATA_PROCESSED / "te_nb1191.npy").astype(np.float64)
    te_nb503 = np.load(DATA_PROCESSED / "te_nb503.npy").astype(np.float64)
    te_nb562 = np.load(DATA_PROCESSED / "te_nb562.npy").astype(np.float64)

    P_unb = np.column_stack([
        median_K20_oof,
        chemprop_oof,
        nb1191_oof,
        nb503_oof,
        nb562_oof,
    ])
    P_te = np.column_stack([
        median_K20_te,
        te_anchor_513,
        te_nb1191,
        te_nb503,
        te_nb562,
    ])
    K_anch = P_unb.shape[1]
    print(f"[stack] P_unb={P_unb.shape}  P_te={P_te.shape}")

    pooled_rae, oof_blend, fold_w, fold_s = cv_run_for_seed(
        P_unb, y_unb, unb_scaffolds, KF_SEED_FOR_PYRAMID,
    )
    mean_s = float(np.mean(fold_s))
    fold_w_mean = np.mean(fold_w, axis=0)

    print(f"\n[cv] kf_seed={KF_SEED_FOR_PYRAMID}  pooled_RAE = {pooled_rae:.4f}")
    print(f"[cv] mean_s={mean_s:.3f}")
    anchor_names = ["median_K20", "chemprop_aux", "nb1191", "nb503", "nb562"]
    print("[cv] mean fold weights:")
    for nm, w in zip(anchor_names, fold_w_mean):
        print(f"     {nm:14s} = {w:.4f}")

    # Deploy refit (single SLSQP on all 253 + stretch from fold-mean)
    w_deploy = slsqp_simplex(P_unb, y_unb)
    blend_unb = P_unb @ w_deploy
    mu_deploy = float(blend_unb.mean())
    s_deploy = mean_s
    in_rae_deploy = float(rae(y_unb, mu_deploy + s_deploy * (blend_unb - mu_deploy)))
    blend_te = P_te @ w_deploy
    deploy_te = (mu_deploy + s_deploy * (blend_te - mu_deploy)).astype(np.float32)
    te_unb_rae_in_sample = float(rae(y_unb, deploy_te[unb_idx]))
    print(f"\n[deploy] w = " + ", ".join(
        f"{nm}={w:.4f}" for nm, w in zip(anchor_names, w_deploy)
    ))
    print(f"[deploy] mu={mu_deploy:.4f}  s={s_deploy:.4f}")
    print(f"[deploy] in-sample RAE  = {in_rae_deploy:.4f}")
    print(f"[deploy] te[unb_idx]    = {te_unb_rae_in_sample:.4f}")

    # ---- Gate ----
    mean_rae = pooled_rae  # single kf_seed result is "mean_rae"
    if mean_rae < PROMOTE_THR:
        verdict = "PROMOTE"
    elif mean_rae < MARGINAL_THR:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"

    delta_vs_nb2095 = mean_rae - NB2095_DEEP30_MEAN
    delta_vs_nb2240 = mean_rae - NB2240_DEEP30_MEAN
    delta_vs_nb2560_ref = mean_rae - NB2560_100SEED_MEAN_REF

    print("\n" + "=" * 78)
    print("PYRAMID AGGREGATE  (single kf_seed=1001)")
    print("=" * 78)
    print(f"  median-vote pyramid RAE = {mean_rae:.4f}")
    print(f"  K20 median-bag standalone RAE = {rae_median_K20:.4f}")
    print(f"  vs nb2095 deep-30 (0.4720): delta = {delta_vs_nb2095:+.4f}")
    print(f"  vs nb2240 deep-30 (0.4601): delta = {delta_vs_nb2240:+.4f}")
    print(f"  vs nb2560 100-seed (0.4665): delta = {delta_vs_nb2560_ref:+.4f}")
    print()
    print(f"  GATE: PROMOTE<{PROMOTE_THR}  MARGINAL<{MARGINAL_THR}")
    print(f"  VERDICT = {verdict}")

    lb_band = LB_W_OOF * mean_rae + LB_W_TE * te_unb_rae_in_sample
    print(f"\n[LB-band] {LB_W_OOF}*pyramid + {LB_W_TE}*te_unb = {lb_band:.4f}")

    # ---- Summary ----
    summary = {
        "tag": TAG,
        "method": "quantile_median_voting_100seed_nb2240_K20_then_pyramid",
        "aggregation": "median_axis0",
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
        "anchors_used": ["median_K20", "chemprop_aux", "nb1191", "nb503", "nb562"],

        "per_seed_rae_K20": [float(x) for x in per_seed_rae_K20],
        "per_seed_rae_K20_mean": float(np.mean(per_seed_rae_K20)),
        "per_seed_rae_K20_std": float(np.std(per_seed_rae_K20)),

        "rae_median_K20_standalone": rae_median_K20,
        "rae_mean_K20_standalone":   rae_mean_K20,
        "delta_median_minus_mean_K20": rae_median_K20 - rae_mean_K20,

        "mean_rae": mean_rae,        # pyramid pooled RAE at kf_seed=1001
        "pooled_rae": mean_rae,
        "kf_seed": KF_SEED_FOR_PYRAMID,
        "fold_s": [float(x) for x in fold_s],
        "fold_w_mean": [float(x) for x in fold_w_mean],
        "mean_s": mean_s,

        "deploy_weights": [
            {"name": nm, "w": float(w)}
            for nm, w in zip(anchor_names, w_deploy)
        ],
        "deploy_mu_blend": mu_deploy,
        "deploy_s": s_deploy,
        "in_sample_rae_overfit_bound": in_rae_deploy,
        "te_unb_rae_in_sample": te_unb_rae_in_sample,

        "reference_nb2095_deep30_mean": NB2095_DEEP30_MEAN,
        "reference_nb2240_deep30_mean": NB2240_DEEP30_MEAN,
        "reference_nb2560_100seed_mean": NB2560_100SEED_MEAN_REF,
        "delta_vs_nb2095_deep30":  delta_vs_nb2095,
        "delta_vs_nb2240_deep30":  delta_vs_nb2240,
        "delta_vs_nb2560_100seed": delta_vs_nb2560_ref,

        "promote_threshold": PROMOTE_THR,
        "marginal_threshold": MARGINAL_THR,
        "verdict": verdict,

        "lb_band_estimate": lb_band,
        "lb_band_w_oof": LB_W_OOF,
        "lb_band_w_te":  LB_W_TE,

        "median_oof_path": str(oof_path),
        "median_te_path":  str(te_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"  median-vote pyramid RAE = {mean_rae:.4f}")
    print(f"  K20 median standalone   = {rae_median_K20:.4f}")
    print(f"  K20 mean   standalone   = {rae_mean_K20:.4f} (nb2560 reference)")
    print(f"  delta vs nb2560 ref     = {delta_vs_nb2560_ref:+.4f}")
    print(f"  delta vs nb2240 deep-30 = {delta_vs_nb2240:+.4f}")
    print(f"  VERDICT                 = {verdict}")
    print(f"  wall                    = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae", "rae_median_K20_standalone", "rae_mean_K20_standalone",
        "delta_median_minus_mean_K20", "delta_vs_nb2560_100seed",
        "delta_vs_nb2240_deep30", "delta_vs_nb2095_deep30",
        "verdict", "lb_band_estimate", "wall_sec",
    ):
        print(f"  {k}: {res.get(k)}")
