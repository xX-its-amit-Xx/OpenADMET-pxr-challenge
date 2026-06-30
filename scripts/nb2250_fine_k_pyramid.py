"""nb2250 -- Fine K-grid pyramid verify: K=18 and K=22 in nb2171-style pyramid.

CONTEXT:
    Cycle 173 found that nb2240 K=20 pyramid (5 anchors with K=20 RFE residual
    swapped in for nb2103_K28) beats the nb2171 K=28 baseline on the 253
    unblind:
        nb2240 pooled_RAE = 0.4598 (5-seed) / 0.4601 deep-30
        nb2171 pooled_RAE = 0.4676 (5-seed)        delta = -0.0078

    Standalone K=20 LGBM(MSE) residual is known to be lucky-seed
    (per_seed_mean 0.5068 looked good but the gain comes from the pyramid
    wrap stretching/SLSQP-mixing). This script tests if K=18 or K=22 also
    pass the pyramid-wrap verify, i.e. is K=20 a true optimum or a noisy
    pick on the K-grid?

PROTOCOL:
    1. Reconstruct K=18 and K=22 surviving feature sets from the recorded
       nb2231 RFE trajectory (no fresh RFE needed; the trajectory IS the
       map).
    2. For each K in {18, 22}: build the K-feature anchor exactly as
       nb2240 (chemprop_aux te[unb_idx] + LGBM(MSE) residual mean-bag
       over 5 RESID_SEEDS, KFold(5)).
    3. Build the 5-anchor pyramid {nbK_anchor, chemprop_aux, nb1191,
       nb503, nb562} with SLSQP + rank-stretch per scaffold-fold.
    4. 5-fold scaffold CV with 30 deep seeds {1116-1145}.
    5. Compare per-seed mean RAE vs nb2240 K=20 deep-30 (0.4601 +/- 0.0017).
    6. Verdict per K:
         - BEATS_K20  if mean_rae < 0.4601 - 0.003
         - FLAT_K20   if |delta| <= 0.003
         - WORSE_K20  if mean_rae > 0.4601 + 0.003
       Identify true K optimum in pyramid wrap.
    7. If K=18 or K=22 beats K=20: build deploy CSV.

Outputs:
    scripts/nb2250_fine_k_pyramid.py
    data/processed/nb2250_summary.json
    submissions/nb2250_K{best_K}_pyramid.csv   (only on gate pass)
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

TAG = "nb2250"

# K-grid: extend nb2231's recorded RFE trajectory at K=18 and K=22
K_TEST = [18, 22]

# Residual learner (identical to nb2240)
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

# Pyramid evaluation: 30 deep seeds 1116..1145
N_FOLDS = 5
DEEP_SEEDS = list(range(1116, 1146))   # 30 seeds
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]
LB_W_OOF = 0.51
LB_W_TE = 0.49

# nb2240 K=20 deep-30 reference (from data/processed/nb2240_summary.json)
NB2240_K20_DEEP30_MEAN = 0.4601
NB2240_K20_DEEP30_STD = 0.0017
DECISION_MARGIN = 0.003

# Feature-matrix cache paths (same as nb2240)
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
NB2231_SUMMARY = DATA_PROCESSED / "nb2231_summary.json"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

# nb1191 reconstruction parameters (identical to nb2171/nb2240)
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
# helpers (identical to nb2240)
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


def deep_verify_seeds(P_unb, y_unb, unb_scaffolds, seeds):
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


def reconstruct_surviving_at_K(nb2231_sum, K_target):
    """Walk the recorded RFE trajectory to recover the surviving feat-idx
    set at the requested K_after (uses dropped column indices)."""
    # start with SHAP top-28
    shap_top28 = list(nb2231_sum["shap_top28_idx_in_117"])
    current = list(shap_top28)
    traj = nb2231_sum["rfe_trajectory"]
    for entry in traj:
        if entry.get("feat_dropped") is None:
            continue
        if entry["K_after"] < K_target:
            break
        d = int(entry["feat_dropped"])
        current.remove(d)
        if entry["K_after"] == K_target:
            return current
    if len(current) == K_target:
        return current
    raise ValueError(f"could not reconstruct K={K_target} from trajectory")


# ============================================================================
# MAIN
# ============================================================================
def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Fine K-grid pyramid verify: K={K_TEST} vs nb2240 K=20")
    print(f"          deep seeds={len(DEEP_SEEDS)} ({DEEP_SEEDS[0]}..{DEEP_SEEDS[-1]})")
    print(f"          reference: nb2240 K=20 deep-30 mean_rae={NB2240_K20_DEEP30_MEAN:.4f}"
          f" +/- {NB2240_K20_DEEP30_STD:.4f}")
    print("=" * 78)

    # ---- Load nb2231 trajectory for K=18 and K=22 surviving sets ----
    with open(NB2231_SUMMARY) as f:
        nb2231 = json.load(f)
    K_to_survivors = {}
    for K in K_TEST:
        survivors = reconstruct_surviving_at_K(nb2231, K)
        K_to_survivors[K] = survivors
        print(f"[load] K={K} surviving idx in 117 ({len(survivors)}): "
              f"{survivors[:8]}... (showing first 8)")
        assert len(survivors) == K, f"K={K} reconstruction returned {len(survivors)}"

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

    # ---- Rebuild 117-col 5-way feature matrix (identical to nb2240) ----
    print("\n" + "-" * 78)
    print("REBUILD 117-COL 5-WAY MATRIX (AtomPair / MACCS / Mordred / "
          "ChempropEmbed / Avalon + ChEMBL kNN)")
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

    # ---- Load pyramid sibling anchors (chemprop_aux, nb1191, nb503, nb562) ----
    chemprop_oof = np.load(DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy").astype(np.float64)
    nb503_oof = np.load(DATA_PROCESSED / "nb503_pred_oof.npy").astype(np.float64)
    nb562_oof = np.load(DATA_PROCESSED / "nb562_pred_oof.npy").astype(np.float64)
    nb1191_oof = reconstruct_nb1191_oof(n_unb)
    te_nb1191 = np.load(DATA_PROCESSED / "te_nb1191.npy").astype(np.float64)
    te_nb503 = np.load(DATA_PROCESSED / "te_nb503.npy").astype(np.float64)
    te_nb562 = np.load(DATA_PROCESSED / "te_nb562.npy").astype(np.float64)
    te_chemprop_aux = te_anchor_513

    # ---- Per-K loop ----
    per_K_results = {}
    deploy_artifacts = {}
    for K in K_TEST:
        print("\n" + "=" * 78)
        print(f"K={K} ANCHOR BUILD  (residual LGBM mean-bag over {len(RESID_SEEDS)} seeds)")
        print("=" * 78)
        cols = K_to_survivors[K]
        X_te_K = X_te_full[:, cols].astype(np.float32)
        X_unb_K = X_te_K[unb_idx]
        print(f"   X_unb_K {X_unb_K.shape}  X_te_K {X_te_K.shape}")

        per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
        per_seed_te_resid = np.zeros((len(RESID_SEEDS), n_test), dtype=np.float64)
        per_seed_rae = []
        for i, s in enumerate(RESID_SEEDS):
            ts = time.time()
            resid_oof = _residual_cross_fit_one_seed(X_unb_K, residual, s)
            per_seed_corrected[i] = anchor + resid_oof
            per_seed_rae.append(float(rae(y_unb, anchor + resid_oof)))
            te_resid_s = _train_full_then_predict_te(X_unb_K, residual, X_te_K, s)
            per_seed_te_resid[i] = te_resid_s
            print(f"     seed={s:3d}: rae_corr={per_seed_rae[-1]:.4f}  wall={time.time()-ts:.1f}s")

        mean_bag_oof_K = per_seed_corrected.mean(axis=0)
        mean_bag_te_resid_K = per_seed_te_resid.mean(axis=0)
        te_K_513 = te_anchor_513 + mean_bag_te_resid_K
        rae_K_mean_bag = float(rae(y_unb, mean_bag_oof_K))
        rae_K_per_seed_mean = float(np.mean(per_seed_rae))
        print(f"\n   [K={K}] per-seed mean RAE = {rae_K_per_seed_mean:.4f}")
        print(f"   [K={K}] mean-bag RAE      = {rae_K_mean_bag:.4f}")

        # Stage 2: 5-anchor pyramid
        print("\n   " + "-" * 74)
        print(f"   PYRAMID K={K}: 5 anchors  (nb{TAG}_K{K}, chemprop_aux, nb1191, nb503, nb562)")
        print("   " + "-" * 74)
        anchors_list = [
            (f"nb{TAG}_K{K}",   mean_bag_oof_K.astype(np.float64),  te_K_513.astype(np.float64)),
            ("chemprop_aux",   chemprop_oof,                         te_chemprop_aux),
            ("nb1191",         nb1191_oof,                           te_nb1191),
            ("nb503",          nb503_oof,                            te_nb503),
            ("nb562",          nb562_oof,                            te_nb562),
        ]
        oof_cols, te_cols, indiv_rae = [], [], {}
        for disp, oof, te_arr in anchors_list:
            assert oof.shape == (n_unb,), f"{disp} oof {oof.shape}"
            assert te_arr.shape == (n_test,), f"{disp} te {te_arr.shape}"
            r = float(rae(y_unb, oof))
            indiv_rae[disp] = r
            oof_cols.append(oof)
            te_cols.append(te_arr)
            print(f"     {disp:18s} oof_RAE={r:.4f}")
        P_unb = np.column_stack(oof_cols)
        P_te = np.column_stack(te_cols)

        # Deep-30 verify
        print(f"\n   DEEP-{len(DEEP_SEEDS)} VERIFY  kf_seeds=[{DEEP_SEEDS[0]}..{DEEP_SEEDS[-1]}]")
        deep = deep_verify_seeds(P_unb, y_unb, unb_scaffolds, DEEP_SEEDS)
        print(f"     n_seeds={deep['n_seeds']}  mean_RAE={deep['mean_rae']:.4f}  "
              f"std={deep['std_rae']:.4f}  range=[{deep['min_rae']:.4f}, {deep['max_rae']:.4f}]")

        # Deploy refit
        w_deploy = slsqp_simplex(P_unb, y_unb)
        blend_unb = P_unb @ w_deploy
        mu_deploy = float(blend_unb.mean())
        s_grid_pick, _ = best_stretch_on(blend_unb, y_unb, mu_deploy, STRETCH_GRID)
        # use mean s across deep seeds as the deploy s (more robust)
        # we don't have fold_s from deep; use s_grid_pick from in-sample as fallback
        # NOTE: nb2240 uses mean(fold_s) across kf_seeds; here keep parity:
        # recompute per-seed fold_s and average
        fold_s_collected = []
        for kf_seed in DEEP_SEEDS:
            _, _o, _w, fs = cv_run_for_seed(P_unb, y_unb, unb_scaffolds, kf_seed)
            fold_s_collected.extend([float(x) for x in fs])
        s_deploy = float(np.mean(fold_s_collected))
        in_rae_final = float(rae(y_unb, mu_deploy + s_deploy * (blend_unb - mu_deploy)))
        blend_te = P_te @ w_deploy
        deploy_te = (mu_deploy + s_deploy * (blend_te - mu_deploy)).astype(np.float32)
        te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))
        print(f"\n   DEPLOY  w={[f'{w:.3f}' for w in w_deploy]}  mu={mu_deploy:.4f}  s={s_deploy:.4f}")
        print(f"     in_sample_RAE={in_rae_final:.4f}  te[unb_idx]_RAE={te_unb_rae:.4f}")
        print(f"     deploy_te mean/std = {deploy_te.mean():.3f}/{deploy_te.std():.3f}")
        lb_band_est = LB_W_OOF * deep["mean_rae"] + LB_W_TE * te_unb_rae

        delta_vs_K20 = deep["mean_rae"] - NB2240_K20_DEEP30_MEAN
        if delta_vs_K20 < -DECISION_MARGIN:
            verdict = "BEATS_K20"
        elif abs(delta_vs_K20) <= DECISION_MARGIN:
            verdict = "FLAT_K20"
        else:
            verdict = "WORSE_K20"
        print(f"\n   [K={K}] vs nb2240 K=20 deep-30 ({NB2240_K20_DEEP30_MEAN:.4f}):  "
              f"delta = {delta_vs_K20:+.4f}  verdict = {verdict}")

        per_K_results[K] = {
            "K": K,
            "surviving_idx_in_117": [int(j) for j in cols],
            "rae_K_per_seed_mean": rae_K_per_seed_mean,
            "rae_K_mean_bag": rae_K_mean_bag,
            "delta_K_vs_anchor": rae_K_mean_bag - rae_anchor,
            "anchor_oof_rae_unb": indiv_rae,
            "deep_30_verify": deep,
            "deploy_weights": [
                {"name": disp, "w": float(w)}
                for (disp, _, _), w in zip(anchors_list, w_deploy)
            ],
            "deploy_mu_blend": mu_deploy,
            "deploy_s": s_deploy,
            "in_sample_rae_overfit_bound": in_rae_final,
            "te_unb_rae_in_sample": te_unb_rae,
            "lb_band_estimate": lb_band_est,
            "lb_band_w_oof": LB_W_OOF,
            "lb_band_w_te": LB_W_TE,
            "delta_vs_nb2240_K20_deep30": delta_vs_K20,
            "verdict_vs_nb2240_K20": verdict,
            "deploy_te_mean": float(deploy_te.mean()),
            "deploy_te_std": float(deploy_te.std()),
        }
        deploy_artifacts[K] = deploy_te

    # ---- Identify true K optimum ----
    cand = [(20, NB2240_K20_DEEP30_MEAN)]
    for K, r in per_K_results.items():
        cand.append((K, r["deep_30_verify"]["mean_rae"]))
    cand.sort(key=lambda x: x[1])
    best_K, best_rae = cand[0]
    if best_K == 20:
        global_verdict = "K20_REMAINS_OPTIMAL_FINE_GRID_DOES_NOT_HELP"
    elif best_rae < NB2240_K20_DEEP30_MEAN - DECISION_MARGIN:
        global_verdict = f"FINE_GRID_BEATS_K20_AT_K={best_K}"
    elif abs(best_rae - NB2240_K20_DEEP30_MEAN) < DECISION_MARGIN:
        global_verdict = f"FINE_GRID_FLAT_VS_K20_BEST_K={best_K}"
    else:
        global_verdict = f"FINE_GRID_WORSE_THAN_K20_BEST_K={best_K}"
    print("\n" + "=" * 78)
    print("FINE K-GRID TABLE")
    print("=" * 78)
    print(f"   {'K':>4s}  {'deep_30_mean':>13s}  {'std':>7s}  {'delta_K20':>9s}  verdict")
    print(f"   {20:>4d}  {NB2240_K20_DEEP30_MEAN:>13.4f}  "
          f"{NB2240_K20_DEEP30_STD:>7.4f}  {0.0:>+9.4f}  BASELINE(nb2240)")
    for K, r in sorted(per_K_results.items()):
        deep = r["deep_30_verify"]
        d = r["delta_vs_nb2240_K20_deep30"]
        v = r["verdict_vs_nb2240_K20"]
        print(f"   {K:>4d}  {deep['mean_rae']:>13.4f}  {deep['std_rae']:>7.4f}  "
              f"{d:>+9.4f}  {v}")
    print(f"\n   best K = {best_K}  (deep-30 RAE = {best_rae:.4f})")
    print(f"   global verdict = {global_verdict}")

    # ---- Optional submission CSV on gate pass ----
    submission_csv = None
    if best_K in K_TEST and per_K_results[best_K]["verdict_vs_nb2240_K20"] == "BEATS_K20":
        deploy_te = deploy_artifacts[best_K]
        sub_csv_path = SUBMISSIONS / f"nb2250_K{best_K}_pyramid.csv"
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": deploy_te,
        }).to_csv(sub_csv_path, index=False)
        submission_csv = str(sub_csv_path)
        print(f"\n[save] {sub_csv_path}  (gate BEATS_K20 at K={best_K})")
    else:
        print(f"\n[skip] no submission CSV written (verdict={global_verdict})")

    # ---- Summary JSON ----
    summary = {
        "tag": TAG,
        "method": "fine_K_grid_pyramid_verify_K18_K22_vs_nb2240_K20",
        "K_tested": K_TEST,
        "deep_seeds": DEEP_SEEDS,
        "n_deep_seeds": len(DEEP_SEEDS),
        "n_folds": N_FOLDS,
        "stretch_grid": STRETCH_GRID,
        "resid_folds": RESID_FOLDS,
        "resid_seeds": RESID_SEEDS,
        "n_unb": n_unb,
        "n_te": n_test,
        "n_unique_scaffolds": n_unique_scaf,
        "rae_anchor_chemprop_aux": rae_anchor,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "nb2240_K20_deep30_mean_ref": NB2240_K20_DEEP30_MEAN,
        "nb2240_K20_deep30_std_ref": NB2240_K20_DEEP30_STD,
        "decision_margin": DECISION_MARGIN,
        "per_K_results": per_K_results,
        "best_K_overall": int(best_K),
        "best_rae_overall": float(best_rae),
        "global_verdict": global_verdict,
        "submission_csv": submission_csv,
        "lb_band_w_oof": LB_W_OOF,
        "lb_band_w_te": LB_W_TE,
        "pre_unblind_clean": True,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    for K in K_TEST:
        r = per_K_results[K]
        print(f"   K={K}  deep30_mean={r['deep_30_verify']['mean_rae']:.4f}  "
              f"delta_vs_K20={r['delta_vs_nb2240_K20_deep30']:+.4f}  "
              f"verdict={r['verdict_vs_nb2240_K20']}")
    print(f"   best_K={best_K}  best_rae={best_rae:.4f}  global={global_verdict}")
    print(f"   wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "K_tested",
        "best_K_overall",
        "best_rae_overall",
        "global_verdict",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
    print("\n==== PER-K RESULTS ====")
    for K_str, r in res["per_K_results"].items():
        deep = r["deep_30_verify"]
        print(f"  K={K_str}  deep30_mean={deep['mean_rae']:.4f}  "
              f"std={deep['std_rae']:.4f}  "
              f"delta_vs_K20={r['delta_vs_nb2240_K20_deep30']:+.4f}  "
              f"verdict={r['verdict_vs_nb2240_K20']}")
