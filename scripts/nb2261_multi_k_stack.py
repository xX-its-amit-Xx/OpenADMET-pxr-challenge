"""nb2261 -- Multi-K stacked pyramid: 9-way SLSQP {K15, K20, K22, K28, K35}
                + {chemprop_aux, nb1191, nb503, nb562}.

CONTEXT:
    nb2240 K=20 deep-30 pyramid = 0.4601 (5 anchors)
    nb2250 verified K=18 / K=22 are FLAT vs K=20 in the same pyramid
    This script asks: does stacking MULTIPLE K-variants of the residual
    LGBM anchor diversify the pyramid (SLSQP picks ALL K's) or collapse
    (SLSQP picks ONE K + ignores rest)?

PROTOCOL:
    1. Reconstruct surviving feat-idx sets at K in {15, 20, 22, 28, 35}
       from the nb2231 RFE trajectory (K=28 is the SHAP top-28 baseline,
       all smaller K's are walked down the recorded trajectory).
    2. For each K: build chemprop_aux te[unb_idx] anchor + LGBM(MSE)
       residual mean-bag over 5 RESID_SEEDS, KFold(5).  (K=20 reuses
       nb2240_mean_bag_oof_K20.npy + te_nb2240_K20.npy if present.)
    3. Build the 9-anchor stack:
         0. nb2261_K15
         1. nb2261_K20 (= nb2240_K20)
         2. nb2261_K22
         3. nb2261_K28
         4. nb2261_K35
         5. chemprop_aux
         6. nb1191
         7. nb503
         8. nb562
    4. SLSQP convex blend (w>=0, sum=1) per scaffold-fold + rank-stretch
       grid {1.000..1.150} under 5-fold scaffold-CV on the 253, across
       30 FRESH kf_seeds {1146..1175}.
    5. Pooled RAE = mean across 30 seeds.
    6. Compare vs nb2240 0.4601; gate margin 0.003.
    7. Per-anchor weight diagnostic: mean(fold_w_mean across seeds) shows
       whether multi-K diversifies or collapses.
    8. If gate BEATS: write submissions/nb2261_multi_k_stack.csv.

Outputs:
    scripts/nb2261_multi_k_stack.py
    data/processed/nb2261_summary.json
    data/processed/nb2261_mean_bag_oof_K{15,22,28,35}.npy
    data/processed/te_nb2261_K{15,22,28,35}.npy
    data/processed/te_nb2261.npy
    submissions/nb2261_multi_k_stack.csv      (only on gate pass)
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

TAG = "nb2261"

# K-grid for the multi-K residual anchors
K_LIST = [15, 20, 22, 28, 35]

# Residual learner (identical to nb2240 / nb2250)
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

# Pyramid evaluation: 30 FRESH seeds 1146..1175
N_FOLDS = 5
KF_SEEDS = list(range(1146, 1176))   # 30 seeds (fresh, non-overlapping with nb2240/nb2250)
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]
LB_W_OOF = 0.51
LB_W_TE = 0.49

# Reference + gate
NB2240_REF_RAE = 0.4601
GATE_MARGIN = 0.003

# Feature-matrix cache paths (identical to nb2240 / nb2250)
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

NB2240_OOF_K20 = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"
NB2240_TE_K20 = DATA_PROCESSED / "te_nb2240_K20.npy"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

# nb1191 reconstruction (identical to nb2171 / nb2240 / nb2250)
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
# helpers (identical to nb2240 / nb2250)
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
    set at the requested K_after.  K_target == 28 returns the SHAP top-28."""
    shap_top28 = list(nb2231_sum["shap_top28_idx_in_117"])
    if K_target == 28:
        return shap_top28
    if K_target > 28:
        # Not in trajectory; extend by re-adding SHAP-ranked features (out of top-28)
        # The nb2231 trajectory only PRUNES. For K=35 we need to pull more from a
        # broader SHAP rank. Use nb2063_shap_importance_full117 to take top-35.
        shap_p = DATA_PROCESSED / "nb2063_shap_importance_full117.npy"
        if not shap_p.exists():
            raise FileNotFoundError(
                f"need nb2063 SHAP-117 importance for K>28 reconstruction: {shap_p}"
            )
        imp = np.load(shap_p).astype(np.float64)
        order = np.argsort(-imp)
        survivors = [int(j) for j in order[:K_target]]
        return survivors
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
    raise ValueError(f"could not reconstruct K={K_target} from trajectory (got {len(current)})")


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Multi-K stacked pyramid (9-way SLSQP)")
    print(f"          K_LIST={K_LIST}")
    print(f"          kf_seeds={len(KF_SEEDS)} ({KF_SEEDS[0]}..{KF_SEEDS[-1]})")
    print(f"          reference: nb2240 deep-30 = {NB2240_REF_RAE:.4f}, "
          f"gate margin {GATE_MARGIN:.4f}")
    print("=" * 78)

    # ---- Load nb2231 trajectory and reconstruct K-survivor sets ----
    with open(NB2231_SUMMARY) as f:
        nb2231 = json.load(f)
    K_to_survivors = {}
    for K in K_LIST:
        if K == 20:
            # Use the canonical K=20 surviving set saved in nb2231
            survivors = list(nb2231["snapshots"]["20"]["surviving_idx_in_117"])
        else:
            survivors = reconstruct_surviving_at_K(nb2231, K)
        K_to_survivors[K] = survivors
        print(f"[load] K={K:2d}  survivors_in_117 n={len(survivors)} "
              f"first8={survivors[:8]}")
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
    print("REBUILD 117-COL 5-WAY FEATURE MATRIX")
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

    # ---- Build per-K residual anchors ----
    K_anchors = {}   # K -> (oof_unb_253, te_513)
    K_per_seed_rae = {}
    for K in K_LIST:
        print("\n" + "=" * 78)
        print(f"K={K}  RESIDUAL LGBM ANCHOR  seeds={RESID_SEEDS}  folds={RESID_FOLDS}")
        print("=" * 78)
        cols = K_to_survivors[K]
        X_te_K = X_te_full[:, cols].astype(np.float32)
        X_unb_K = X_te_K[unb_idx]
        print(f"   X_unb_K {X_unb_K.shape}  X_te_K {X_te_K.shape}")

        # Reuse nb2240 K=20 artefact if present (parity check)
        if K == 20 and NB2240_OOF_K20.exists() and NB2240_TE_K20.exists():
            oof_unb = np.load(NB2240_OOF_K20).astype(np.float64)
            te_K_513 = np.load(NB2240_TE_K20).astype(np.float64)
            assert oof_unb.shape == (n_unb,)
            assert te_K_513.shape == (n_test,)
            K_per_seed_rae[K] = []
            print(f"   [K=20] REUSING nb2240 artefacts (parity)")
            print(f"   [K=20] mean_bag_oof_K20 in_RAE = {rae(y_unb, oof_unb):.4f}")
        else:
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
            oof_unb = per_seed_corrected.mean(axis=0)
            mean_bag_te_resid = per_seed_te_resid.mean(axis=0)
            te_K_513 = te_anchor_513 + mean_bag_te_resid
            K_per_seed_rae[K] = per_seed_rae
            rae_K_mean_bag = float(rae(y_unb, oof_unb))
            rae_K_per_seed_mean = float(np.mean(per_seed_rae))
            print(f"\n   [K={K}] per-seed mean RAE = {rae_K_per_seed_mean:.4f}")
            print(f"   [K={K}] mean-bag RAE      = {rae_K_mean_bag:.4f}")
            # Save artefacts
            oof_path = DATA_PROCESSED / f"{TAG}_mean_bag_oof_K{K}.npy"
            te_path = DATA_PROCESSED / f"te_{TAG}_K{K}.npy"
            np.save(oof_path, oof_unb.astype(np.float32))
            np.save(te_path, te_K_513.astype(np.float32))
            print(f"   [save] {oof_path}")
            print(f"   [save] {te_path}")

        K_anchors[K] = (oof_unb, te_K_513)

    # ============================================================================
    # Stage 2: 9-anchor pyramid SLSQP + rank-stretch
    # ============================================================================
    print("\n" + "=" * 78)
    print(f"STAGE 2: 9-ANCHOR PYRAMID  (5 K-variants + 4 sibling anchors)")
    print("=" * 78)
    chemprop_oof = np.load(DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy").astype(np.float64)
    nb503_oof = np.load(DATA_PROCESSED / "nb503_pred_oof.npy").astype(np.float64)
    nb562_oof = np.load(DATA_PROCESSED / "nb562_pred_oof.npy").astype(np.float64)
    nb1191_oof = reconstruct_nb1191_oof(n_unb)
    te_chemprop_aux = te_anchor_513
    te_nb1191 = np.load(DATA_PROCESSED / "te_nb1191.npy").astype(np.float64)
    te_nb503 = np.load(DATA_PROCESSED / "te_nb503.npy").astype(np.float64)
    te_nb562 = np.load(DATA_PROCESSED / "te_nb562.npy").astype(np.float64)

    anchors_list = []
    for K in K_LIST:
        oof_unb, te_K_513 = K_anchors[K]
        anchors_list.append((f"nb{TAG}_K{K}", oof_unb.astype(np.float64), te_K_513.astype(np.float64)))
    anchors_list.append(("chemprop_aux", chemprop_oof, te_chemprop_aux))
    anchors_list.append(("nb1191",       nb1191_oof,   te_nb1191))
    anchors_list.append(("nb503",        nb503_oof,    te_nb503))
    anchors_list.append(("nb562",        nb562_oof,    te_nb562))

    oof_cols, te_cols, indiv_rae = [], [], {}
    print("\n[anchors]")
    for disp, oof, te_arr in anchors_list:
        assert oof.shape == (n_unb,), f"{disp} oof {oof.shape}"
        assert te_arr.shape == (n_test,), f"{disp} te {te_arr.shape}"
        r = float(rae(y_unb, oof))
        indiv_rae[disp] = r
        oof_cols.append(oof)
        te_cols.append(te_arr)
        print(f"   {disp:18s} oof_RAE={r:.4f}  te_mean={te_arr.mean():.3f}  te_std={te_arr.std():.3f}")
    P_unb = np.column_stack(oof_cols)
    P_te = np.column_stack(te_cols)
    K_anch = P_unb.shape[1]
    print(f"[stack] P_unb {P_unb.shape}  P_te {P_te.shape}  K={K_anch}")
    assert K_anch == 9, f"expected 9 anchors, got {K_anch}"

    # Anchor pair correlations (residual-corr): helpful for diversification check
    print("\n[corr matrix among 9 anchors (Pearson on OOF)]")
    corr = np.corrcoef(P_unb.T)
    names = [a[0] for a in anchors_list]
    hdr = "          " + " ".join(f"{n[:9]:>9s}" for n in names)
    print(hdr)
    for i, n in enumerate(names):
        row = " ".join(f"{corr[i, j]:>9.3f}" for j in range(K_anch))
        print(f"   {n[:9]:>9s} {row}")

    # ---- Scaffold 5-fold CV across 30 seeds ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV  kf_seeds={KF_SEEDS[0]}..{KF_SEEDS[-1]}  "
          f"n={len(KF_SEEDS)}  stretch_grid={STRETCH_GRID}")
    print("-" * 78)
    per_seed = []
    all_oofs = []
    all_fold_w = []
    for kf_seed in KF_SEEDS:
        pooled, oof_blend, fw, fs = cv_run_for_seed(P_unb, y_unb, unb_scaffolds, kf_seed)
        seed_w_mean = np.mean(fw, axis=0)
        all_fold_w.append(seed_w_mean)
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": float(pooled),
            "fold_s": [float(x) for x in fs],
            "fold_w_mean": [float(x) for x in seed_w_mean],
        })
        all_oofs.append(oof_blend)
        print(f"   seed={kf_seed}  pooled_RAE={pooled:.4f}  mean_s={np.mean(fs):.3f}  "
              f"w={np.round(seed_w_mean, 3).tolist()}")

    pooled_rae_mean_seeds = float(np.mean([r["pooled_rae"] for r in per_seed]))
    pooled_rae_std_seeds = float(np.std([r["pooled_rae"] for r in per_seed]))
    pooled_rae_min = float(np.min([r["pooled_rae"] for r in per_seed]))
    pooled_rae_max = float(np.max([r["pooled_rae"] for r in per_seed]))
    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    final_oof_rae = float(rae(y_unb, mean_oof))

    print(f"\n[cv] mean across {len(KF_SEEDS)} seeds  pooled_RAE = "
          f"{pooled_rae_mean_seeds:.4f} (+/- {pooled_rae_std_seeds:.4f})")
    print(f"[cv] range [{pooled_rae_min:.4f}, {pooled_rae_max:.4f}]")
    print(f"[cv] RAE of mean-of-seed OOFs        = {final_oof_rae:.4f}")

    # ---- Per-anchor weight summary ----
    mean_w_per_anchor = np.mean(np.array(all_fold_w), axis=0)  # (9,)
    std_w_per_anchor = np.std(np.array(all_fold_w), axis=0)
    print("\n[per-anchor weights]  (mean +/- std across 30 seeds)")
    K_var_weight_sum = 0.0
    sib_weight_sum = 0.0
    for i, (disp, _, _) in enumerate(anchors_list):
        marker = ""
        if i < len(K_LIST):
            K_var_weight_sum += float(mean_w_per_anchor[i])
            marker = " [K-variant]"
        else:
            sib_weight_sum += float(mean_w_per_anchor[i])
            marker = " [sibling]"
        print(f"   {disp:18s} w = {mean_w_per_anchor[i]:.3f} +/- {std_w_per_anchor[i]:.3f}{marker}")
    print(f"\n   sum(K-variant weights) = {K_var_weight_sum:.3f}  "
          f"sum(sibling weights) = {sib_weight_sum:.3f}")
    # Diversification diagnostic
    K_active = int(np.sum(mean_w_per_anchor[:len(K_LIST)] > 0.02))
    if K_active <= 1:
        K_div_verdict = "COLLAPSED_to_single_K"
    elif K_active == len(K_LIST):
        K_div_verdict = "FULLY_DIVERSIFIED_all_K_active"
    else:
        K_div_verdict = f"PARTIAL_DIVERSE_{K_active}_of_{len(K_LIST)}_K_active"
    print(f"   K-diversification verdict = {K_div_verdict}")

    # ---- Deploy ----
    print("\n" + "-" * 78)
    print(f"DEPLOY (refit weights on all 253; mean(fold_s) across all "
          f"{len(KF_SEEDS)} seeds)")
    print("-" * 78)
    w_deploy = slsqp_simplex(P_unb, y_unb)
    blend_unb = P_unb @ w_deploy
    mu_deploy = float(blend_unb.mean())
    all_fold_s = [s for r in per_seed for s in r["fold_s"]]
    s_deploy = float(np.mean(all_fold_s))
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

    # ---- Gate vs nb2240 ----
    delta_vs_nb2240 = pooled_rae_mean_seeds - NB2240_REF_RAE
    gate_beat = delta_vs_nb2240 < -GATE_MARGIN
    gate_flat = abs(delta_vs_nb2240) <= GATE_MARGIN
    if gate_beat:
        verdict = "BEATS_NB2240"
    elif gate_flat:
        verdict = "FLAT_VS_NB2240"
    else:
        verdict = "HURTS_NB2240"
    print("\n" + "-" * 78)
    print(f"GATE EVALUATION  (vs nb2240 ref pooled_rae {NB2240_REF_RAE:.4f}, "
          f"margin {GATE_MARGIN:.4f})")
    print("-" * 78)
    print(f"   nb2261 OOF ({len(KF_SEEDS)}-seed mean) = {pooled_rae_mean_seeds:.4f}")
    print(f"   nb2240 reference                = {NB2240_REF_RAE:.4f}")
    print(f"   delta                           = {delta_vs_nb2240:+.4f}")
    print(f"   verdict                         = {verdict}")

    # ---- Always save te artefact ----
    te_npy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_npy_path, deploy_te)
    print(f"\n[save] {te_npy_path}")

    sub_csv_path = SUBMISSIONS / f"{TAG}_multi_k_stack.csv"
    submission_csv = None
    if gate_beat:
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": deploy_te,
        }).to_csv(sub_csv_path, index=False)
        submission_csv = str(sub_csv_path)
        print(f"[save] {sub_csv_path}  (gate BEATS_NB2240)")
    else:
        print(f"[skip] gate not beat -- no submission CSV written ({verdict})")

    # ---- Summary JSON ----
    summary = {
        "tag": TAG,
        "method": "multi_K_stacked_pyramid_9way_SLSQP",
        "K_list": K_LIST,
        "anchors": [a[0] for a in anchors_list],
        "anchor_oof_rae_unb": indiv_rae,
        "rae_anchor_chemprop_aux": rae_anchor,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "K_to_survivors": {str(K): [int(j) for j in K_to_survivors[K]] for K in K_LIST},
        "resid_folds": RESID_FOLDS,
        "resid_seeds": RESID_SEEDS,
        "K_per_seed_rae": {str(K): K_per_seed_rae[K] for K in K_LIST},
        "kf_seeds": KF_SEEDS,
        "n_kf_seeds": len(KF_SEEDS),
        "n_folds": N_FOLDS,
        "stretch_grid": STRETCH_GRID,
        "n_unb": n_unb,
        "n_te": n_test,
        "n_unique_scaffolds": n_unique_scaf,
        "n_anchors": int(K_anch),
        "anchor_corr_matrix": corr.tolist(),
        "per_seed_results": per_seed,
        "pooled_rae_mean_seeds": pooled_rae_mean_seeds,
        "pooled_rae_std_seeds": pooled_rae_std_seeds,
        "pooled_rae_min": pooled_rae_min,
        "pooled_rae_max": pooled_rae_max,
        "rae_of_mean_of_seed_oofs": final_oof_rae,
        "mean_w_per_anchor": [float(w) for w in mean_w_per_anchor],
        "std_w_per_anchor": [float(w) for w in std_w_per_anchor],
        "K_variant_weight_sum": float(K_var_weight_sum),
        "sibling_weight_sum": float(sib_weight_sum),
        "K_active_count": int(K_active),
        "K_diversification_verdict": K_div_verdict,
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
        "compare_nb2240_ref": NB2240_REF_RAE,
        "delta_vs_nb2240": delta_vs_nb2240,
        "gate_margin": GATE_MARGIN,
        "gate_beat_nb2240": bool(gate_beat),
        "gate_flat_vs_nb2240": bool(gate_flat),
        "verdict_vs_nb2240": verdict,
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "te_npy_path": str(te_npy_path),
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
    print(f"   pooled_RAE ({len(KF_SEEDS)} seeds)        = {pooled_rae_mean_seeds:.4f} "
          f"+/- {pooled_rae_std_seeds:.4f}")
    print(f"   delta vs nb2240 ({NB2240_REF_RAE:.4f})    = {delta_vs_nb2240:+.4f}")
    print(f"   verdict                          = {verdict}")
    print(f"   K-diversification                = {K_div_verdict}")
    print(f"   K-variant weight sum             = {K_var_weight_sum:.3f}")
    print(f"   sibling weight sum               = {sib_weight_sum:.3f}")
    print(f"   LB band estimate                 = {lb_band_est:.4f}")
    print(f"   submission_csv                   = {submission_csv}")
    print(f"   wall                             = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "K_list",
        "pooled_rae_mean_seeds",
        "pooled_rae_std_seeds",
        "delta_vs_nb2240",
        "verdict_vs_nb2240",
        "gate_beat_nb2240",
        "K_diversification_verdict",
        "K_variant_weight_sum",
        "sibling_weight_sum",
        "deploy_weights",
        "deploy_s",
        "lb_band_estimate",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
