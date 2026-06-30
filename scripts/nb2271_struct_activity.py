"""nb2271 -- Structure-activity hybrid: append TOP-5 SHAP-selected Boltz
pose features to the RFE K=20 anchor feature set -> K=25, then plug into the
nb2240/nb2171 5-anchor SLSQP pyramid in place of nb2240_K20.

CONTEXT:
    Cycle 153 nb2020 appended ALL 44 Boltz pose-quality features (mean/max/std
    /best aggregates of confidence_score, ptm, iptm, ligand_iptm, plDDT, pDE,
    iptm_0_1, iptm_1_0, pair_iptm_A_B_mean) to the K=28 SHAP feature matrix and
    HURT (mean-bag RAE 0.5087 vs nb2103 K=28 ref 0.4737, delta +0.0350).  Cause:
    44 noisy pose columns swamped the residual LGBM at n=253.

    This script tests the hypothesis that a SELECTIVE subset survives.  We use
    SHAP on the chemprop_aux residual with all 44 pose features available as
    candidates to pick the TOP-5 by mean(|SHAP|), then append them to the K=20
    RFE features used in nb2240 -> K=25.  Same 5-seed bag / 5-fold cross-fit
    LGBM(MSE) residual model.  K=25 anchor then swaps in for nb2240_K20 in the
    nb2171 5-anchor SLSQP pyramid.

PROTOCOL:
    1. Load Boltz parquet (513 rows, 44 numeric pose features).  Median-impute.
    2. Anchor   = chemprop_aux te[unb_idx] (PRE-unblind, in_RAE 0.6216).
       residual = y_unb - anchor.
    3. SHAP rank pose features:
         fit LGBM(MSE) on (X_pose_unb, residual), 5-fold cross-fit per seed
         {0, 1, 7, 42, 137}, accumulate |SHAP| values via TreeExplainer on the
         held-out fold, average across folds and seeds, pick top-5.
    4. Rebuild K=20 RFE feature matrix on 513 (same path as nb2240; load via
       nb2231 surviving idx into the 117-col 5-way matrix); slice to unb_idx.
       Append top-5 pose cols -> X_K25_unb (253, 25), X_K25_te (513, 25).
    5. Fit residual LGBM(MSE) mean-bag of 5 seeds {0,1,7,42,137}, 5-fold KFold
       cross-fit per seed on the 253; deploy refit on full 253 to predict te.
       Save nb2271_mean_bag_oof_K25.npy (253) + te_nb2271_K25.npy (513).
    6. Plug nb2271_K25 anchor into the nb2171 5-anchor SLSQP pyramid in place
       of nb2240_K20.  Scaffold 5-fold CV across 5 kf_seeds {1001..1005},
       per-fold SLSQP simplex + rank-stretch grid 1.000..1.150.
    7. Gate vs nb2240 pooled_rae_mean_seeds 0.4598 at margin 0.003.  If beats:
       deep-30 verify (5 canonical + 25 extra seeds) and write submission CSV.

Outputs:
    scripts/nb2271_struct_activity.py
    data/processed/nb2271_summary.json
    data/processed/nb2271_mean_bag_oof_K25.npy   (253,) float32
    data/processed/te_nb2271_K25.npy             (513,) float32
    data/processed/te_nb2271.npy                 (513,) float32  (deploy)
    submissions/nb2271_struct_activity.csv       (only on gate pass)
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

TAG = "nb2271"

# ------------------------------ stage 1: residual + pose SHAP ---------------
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")

EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

BOLTZ_POSE_PATH = DATA_PROCESSED / "boltz_dargason_features_test.parquet"

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

N_TOP_POSE = 5

# ------------------------------ stage 2: pyramid -----------------------------
GATE_MARGIN = 0.003
NB2240_REF_OOF = 0.4598   # pooled_rae_mean_seeds from nb2240_summary.json

N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]
LB_W_OOF = 0.51
LB_W_TE = 0.49

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


def _load_pose_features_513(test_names: list[str]) -> tuple[np.ndarray, list[str]]:
    if not BOLTZ_POSE_PATH.exists():
        raise FileNotFoundError(f"missing Boltz parquet: {BOLTZ_POSE_PATH}")
    bz = pd.read_parquet(BOLTZ_POSE_PATH)
    if "name" not in bz.columns:
        raise KeyError("Boltz parquet missing 'name' column")
    bz_lookup = {n: i for i, n in enumerate(bz["name"].tolist())}
    n_test = len(test_names)
    feat_cols = [c for c in bz.columns if c != "name"]
    X_pose = np.full((n_test, len(feat_cols)), np.nan, dtype=np.float32)
    matched = 0
    for i, name in enumerate(test_names):
        if name in bz_lookup:
            row = bz.iloc[bz_lookup[name]]
            X_pose[i] = row[feat_cols].values.astype(np.float32)
            matched += 1
    print(f"[pose] matched {matched}/{n_test} test compounds")
    # median impute
    if np.isnan(X_pose).any():
        col_med = np.nanmedian(X_pose, axis=0)
        col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
        bad = ~np.isfinite(X_pose)
        idx_r, idx_c = np.where(bad)
        X_pose[idx_r, idx_c] = col_med[idx_c]
    return X_pose, feat_cols


def _shap_rank_pose_features(X_pose_unb: np.ndarray, residual: np.ndarray,
                             feat_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Rank pose features by mean(|SHAP|) using LGBM(MSE) trained on residual.
    Accumulates SHAP values on each held-out fold across all seeds, then takes
    the mean of absolute values across all (fold, seed) rows.
    """
    n_feat = X_pose_unb.shape[1]
    n_unb = X_pose_unb.shape[0]
    accum = np.zeros(n_feat, dtype=np.float64)
    rows_seen = 0
    for seed in RESID_SEEDS:
        kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
        for tr_loc, va_loc in kf.split(np.arange(n_unb)):
            mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
            mdl.fit(X_pose_unb[tr_loc], residual[tr_loc])
            # SHAP via pred_contrib (LGBM native; shape: (n, n_feat + 1))
            contribs = mdl.predict(X_pose_unb[va_loc], pred_contrib=True)
            # drop the last col (bias)
            sh = np.abs(contribs[:, :-1]).sum(axis=0)
            accum += sh
            rows_seen += contribs.shape[0]
    mean_abs_shap = accum / max(rows_seen, 1)
    rank = np.argsort(-mean_abs_shap)
    return rank, mean_abs_shap


# ============================================================================
# stage 2 utils (SLSQP + rank-stretch -- mirrors nb2240)
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
    print(f"{TAG} -- struct-activity: K=20 RFE + top-{N_TOP_POSE} SHAP pose -> K=25")
    print("=" * 78)

    # ---- Load nb2231 K=20 surviving indices ----
    with open(NB2231_SUMMARY) as f:
        nb2231 = json.load(f)
    surviving_K20 = list(nb2231["snapshots"]["20"]["surviving_idx_in_117"])
    surviving_K20_names = list(nb2231["snapshots"]["20"]["surviving_names"])
    assert len(surviving_K20) == 20, f"expected 20, got {len(surviving_K20)}"

    # ---- Truth + anchor ----
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    te_names = te["name"].values if "name" in te.columns else te["Molecule Name"].values
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] chemprop_aux in_RAE = {rae_anchor:.4f} (ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor

    # ---- Pose features 513 ----
    test_names_list = list(te_names)
    X_pose_513, pose_feat_names = _load_pose_features_513(test_names_list)
    X_pose_unb = X_pose_513[unb_idx].astype(np.float32)
    print(f"[pose] feat_dim={X_pose_unb.shape[1]}  unb={X_pose_unb.shape}")

    # ---- SHAP rank pose features against chemprop_aux residual ----
    print("\n" + "-" * 78)
    print(f"SHAP rank pose features  seeds={RESID_SEEDS}  folds={RESID_FOLDS}")
    print("-" * 78)
    rank, mean_abs_shap = _shap_rank_pose_features(X_pose_unb, residual, pose_feat_names)
    top5_idx = rank[:N_TOP_POSE].tolist()
    top5_names = [pose_feat_names[j] for j in top5_idx]
    print(f"[shap] top-{N_TOP_POSE} pose features by mean(|SHAP|):")
    for k, j in enumerate(top5_idx):
        print(f"   {k+1:2d}. {pose_feat_names[j]:40s}  |SHAP|={mean_abs_shap[j]:.4f}")

    # ---- Rebuild K=20 RFE feature matrix on 513 (mirrors nb2240) ----
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
        std_test_smiles.append(Chem.MolToSmiles(m) if m is not None else "")
    fp_test = morgan_fp_batch(std_test_smiles)
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )

    X_te_full = np.concatenate(
        [
            X_ap_te_top, X_maccs_te_top, X_mord_te_top, X_emb_te_top, X_av_te_top,
            pred_chembl_pec50.reshape(-1, 1).astype(np.float32),
            mean_sim.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    assert X_te_full.shape[1] == 117, f"feat_dim {X_te_full.shape[1]} != 117"
    X_te_K20 = X_te_full[:, surviving_K20].astype(np.float32)

    # ---- K=25: append top-5 pose ----
    X_pose_te_top5 = X_pose_513[:, top5_idx].astype(np.float32)
    X_pose_unb_top5 = X_pose_unb[:, top5_idx].astype(np.float32)
    X_te_K25 = np.hstack([X_te_K20, X_pose_te_top5]).astype(np.float32)
    X_unb_K25 = X_te_K25[unb_idx]
    print(f"\n[feat] X_unb_K25 = {X_unb_K25.shape}  X_te_K25 = {X_te_K25.shape}")

    # ---- Fit residual LGBM, 5-seed bag, 5-fold cross-fit ----
    print("\n" + "-" * 78)
    print(f"K=25 RESIDUAL LGBM  seeds={RESID_SEEDS}  folds={RESID_FOLDS}")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_te_resid = np.zeros((len(RESID_SEEDS), n_test), dtype=np.float64)
    per_seed_rae = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        resid_oof = _residual_cross_fit_one_seed(X_unb_K25, residual, s)
        per_seed_corrected[i] = anchor + resid_oof
        per_seed_rae.append(float(rae(y_unb, anchor + resid_oof)))
        te_resid_s = _train_full_then_predict_te(X_unb_K25, residual, X_te_K25, s)
        per_seed_te_resid[i] = te_resid_s
        print(f"   seed={s:3d}: rae_corr={per_seed_rae[-1]:.4f}  wall={time.time()-ts:.1f}s")

    mean_bag_oof_K25 = per_seed_corrected.mean(axis=0)
    mean_bag_te_resid_K25 = per_seed_te_resid.mean(axis=0)
    te_K25_513 = te_anchor_513 + mean_bag_te_resid_K25
    rae_K25_mean_bag = float(rae(y_unb, mean_bag_oof_K25))
    rae_K25_per_seed_mean = float(np.mean(per_seed_rae))
    print(f"\n[K25] per-seed mean RAE = {rae_K25_per_seed_mean:.4f}")
    print(f"[K25] mean-bag RAE      = {rae_K25_mean_bag:.4f}")
    print(f"[K25] delta vs anchor   = {rae_K25_mean_bag - rae_anchor:+.4f}")

    oof_K25_path = DATA_PROCESSED / f"{TAG}_mean_bag_oof_K25.npy"
    te_K25_path = DATA_PROCESSED / f"te_{TAG}_K25.npy"
    np.save(oof_K25_path, mean_bag_oof_K25.astype(np.float32))
    np.save(te_K25_path, te_K25_513.astype(np.float32))

    # ============================================================================
    # Stage 2: 5-anchor pyramid -- K=25 replaces nb2240_K20
    # ============================================================================
    print("\n" + "=" * 78)
    print("STAGE 2: 5-ANCHOR PYRAMID  (nb2271_K25 swaps in for nb2240_K20)")
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
        ("nb2271_K25",   mean_bag_oof_K25.astype(np.float64), te_K25_513.astype(np.float64)),
        ("chemprop_aux", chemprop_oof,                        te_chemprop_aux),
        ("nb1191",       nb1191_oof,                          te_nb1191),
        ("nb503",        nb503_oof,                           te_nb503),
        ("nb562",        nb562_oof,                           te_nb562),
    ]
    indiv_rae = {}
    oof_cols, te_cols = [], []
    print("\n[anchors]")
    for disp, oof, te_arr in anchors_list:
        r = float(rae(y_unb, oof))
        indiv_rae[disp] = r
        oof_cols.append(oof)
        te_cols.append(te_arr)
        print(f"   {disp:14s} oof_RAE={r:.4f}  te_mean={te_arr.mean():.3f}  te_std={te_arr.std():.3f}")
    P_unb = np.column_stack(oof_cols)
    P_te = np.column_stack(te_cols)

    # Scaffold 5-fold CV across 5 seeds
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

    # Deploy
    w_deploy = slsqp_simplex(P_unb, y_unb)
    blend_unb = P_unb @ w_deploy
    mu_deploy = float(blend_unb.mean())
    s_deploy = float(np.mean([s for r in per_seed for s in r["fold_s"]]))
    in_rae_final = float(rae(y_unb, mu_deploy + s_deploy * (blend_unb - mu_deploy)))
    blend_te = P_te @ w_deploy
    deploy_te = (mu_deploy + s_deploy * (blend_te - mu_deploy)).astype(np.float32)
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))
    w_str = ", ".join(f"{disp}={w:.4f}" for (disp, _, _), w in zip(anchors_list, w_deploy))
    print(f"\n[deploy] weights = {w_str}")
    print(f"[deploy] mu/s    = {mu_deploy:.4f} / {s_deploy:.4f}")
    print(f"[deploy] in-sample RAE = {in_rae_final:.4f}  te[unb_idx] = {te_unb_rae:.4f}")

    lb_band_est = LB_W_OOF * pooled_rae_mean_seeds + LB_W_TE * te_unb_rae

    # Gate vs nb2240
    delta_vs_nb2240 = pooled_rae_mean_seeds - NB2240_REF_OOF
    gate_beat = delta_vs_nb2240 < -GATE_MARGIN
    gate_flat = abs(delta_vs_nb2240) <= GATE_MARGIN
    if gate_beat:
        verdict = "BEATS_NB2240"
    elif gate_flat:
        verdict = "FLAT_VS_NB2240"
    else:
        verdict = "HURTS_NB2240"
    print(f"\n[gate] pooled={pooled_rae_mean_seeds:.4f}  ref={NB2240_REF_OOF:.4f}  "
          f"delta={delta_vs_nb2240:+.4f}  -> {verdict}")

    deep30 = None
    if gate_beat:
        print("\n[deep30] verifying on 5 canonical + 25 extra seeds...")
        deep30 = deep_verify_seeds(P_unb, y_unb, unb_scaffolds, n_extra=25)
        print(f"   n={deep30['n_seeds']} mean_RAE={deep30['mean_rae']:.4f} "
              f"std={deep30['std_rae']:.4f}")

    # Always save te
    te_npy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_npy_path, deploy_te)

    sub_csv_path = SUBMISSIONS / f"{TAG}_struct_activity.csv"
    if gate_beat:
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": deploy_te,
        }).to_csv(sub_csv_path, index=False)
        print(f"[save] {sub_csv_path}  (gate BEATS_NB2240)")
    else:
        print(f"[skip] gate not beat -- no submission CSV ({verdict})")

    summary = {
        "tag": TAG,
        "method": "struct_activity_K20_plus_top5_SHAP_pose_into_nb2171_pyramid",
        "n_top_pose": N_TOP_POSE,
        "boltz_parquet_path": str(BOLTZ_POSE_PATH),
        "pose_feat_names_all": pose_feat_names,
        "shap_top5_idx": [int(j) for j in top5_idx],
        "shap_top5_names": top5_names,
        "shap_top5_values": [float(mean_abs_shap[j]) for j in top5_idx],
        "k20_surviving_idx_in_117": [int(j) for j in surviving_K20],
        "k20_surviving_names": surviving_K20_names,
        "anchors": [a[0] for a in anchors_list],
        "anchor_oof_rae_unb": indiv_rae,
        "rae_anchor_chemprop_aux": rae_anchor,
        "resid_folds": RESID_FOLDS,
        "resid_seeds": RESID_SEEDS,
        "rae_K25_per_seed_mean": rae_K25_per_seed_mean,
        "rae_K25_mean_bag": rae_K25_mean_bag,
        "delta_K25_vs_anchor": rae_K25_mean_bag - rae_anchor,
        "nb2271_oof_K25_path": str(oof_K25_path),
        "te_K25_path": str(te_K25_path),
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
    print(f"\n[save] {out_path}")
    print("=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   K=25 mean-bag RAE             = {rae_K25_mean_bag:.4f}")
    print(f"   pyramid pooled RAE (5 seeds)  = {pooled_rae_mean_seeds:.4f}")
    print(f"   delta vs nb2240 (0.4598)      = {delta_vs_nb2240:+.4f}")
    print(f"   verdict                       = {verdict}")
    print(f"   wall                          = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "shap_top5_names",
        "rae_K25_mean_bag",
        "delta_K25_vs_anchor",
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
