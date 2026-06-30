"""nb2682 -- Adversarial validation for distribution-shift detection + IPS-style
reweighting (from-scratch full pipeline).

PARADIGM:
    Stage 1: Train a binary classifier to distinguish train rows (the 253
             unblind subset, with labels) from test rows (the 513 blind set
             held back from training).  Both expressed in the same 117-feature
             representation that drives every cycle-160+ ceiling number.  Use
             5-fold StratifiedKFold OOF to get an honest P(test|x) for every
             row.
    Stage 2: Compute per-train-row IPS density-ratio weight
                 w_i = P(test|x_train_i) / max(P(test|x_test_mean_neighbour), tiny)
             implemented as:
                 num = P(test|x_train_i)                       (253-vec)
                 den = max(P(test|x_test).mean(), tiny)        (scalar)
                 w_i = clip(num / den, 0.1, 10)
             This is the standard density-ratio IPS form, NOT the per-row
             normalisation of nb2452 -- the denominator anchors the scale to
             the test-domain density, so a train row that the classifier
             confidently calls "test-like" gets a literal density ratio > 1.
    Stage 3: Fit LGBMRegressor(max_depth=4, num_leaves=15, n_estimators=300,
             learning_rate=0.03) on the PXR pEC50 residual (y_unb - anchor)
             using the K=20 surviving features and the IPS sample weights.
             5-fold scaffold CV on 253 rows, 5 kf_seeds.
    Stage 4: Build the same 5-anchor pyramid as nb2452 (K20_aw, chemprop_aux,
             nb1191, nb503, nb562), SLSQP convex blend + rank-stretch.
    Stage 5: Compare pooled mean RAE across 5 kf_seeds to the gate:
                <0.4570  -> PROMOTE
                <0.4598  -> MARGINAL_BEAT
                else     -> FAIL

DIFFERENT FROM nb2452 (cycle 195):
    - nb2452 normalised P(test) to mean-1 (per-row weight = P/mean(P)) which
      is a relative scaling, not a true density ratio.
    - This script uses the explicit IPS form num/den with a test-mean
      denominator, so the weights have absolute density-ratio meaning.
    - Different gate (nb2452 was 0.4601 +/- 0.003; this is 0.4570 / 0.4598).
    - Reports adversarial AUC + per-row weight diagnostics (top-10, histogram
      summary stats).

OUTPUTS:
    scripts/nb2682_adversarial_validation_weights.py
    data/processed/nb2682_summary.json
    data/processed/nb2682_pred_oof.npy            (253,) float32 pyramid OOF
    data/processed/te_nb2682.npy                  (513,) float32 deploy
    data/processed/nb2682_ips_weights_unb.npy     (253,) float32 IPS weights
    data/processed/nb2682_ptest_unb.npy           (253,) float32 raw P(test) on unb
    data/processed/nb2682_ptest_te.npy            (513,) float32 raw P(test) on 513
    submissions/nb2682_adv_validation_weights.csv (PROMOTE / MARGINAL_BEAT only)
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
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2682"

# -------------------------- gates & references ------------------------------
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598
CHEMPROP_AUX_REF = 0.6216

# -------------------------- adversarial classifier --------------------------
ADV_FOLDS = 5
ADV_SEED = 4242
ADV_LGBM = dict(
    objective="binary",
    max_depth=5,
    num_leaves=31,
    n_estimators=400,
    learning_rate=0.03,
    min_child_samples=10,
    reg_lambda=2.0,
    n_jobs=2,
    verbosity=-1,
)

# -------------------------- IPS weight transform ----------------------------
WEIGHT_CLIP_LO = 0.1
WEIGHT_CLIP_HI = 10.0
DENOM_TINY = 1e-3   # protects against P(test|x_test).mean() collapsing

# -------------------------- PXR residual LGBM (per brief) -------------------
PXR_LGBM = dict(
    objective="regression",
    max_depth=4,
    num_leaves=15,
    n_estimators=300,
    learning_rate=0.03,
    min_child_samples=5,
    reg_lambda=2.0,
    n_jobs=2,
    verbosity=-1,
)
RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

# -------------------------- pyramid CV --------------------------------------
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]
LB_W_OOF = 0.51
LB_W_TE = 0.49

# -------------------------- nb1191 reconstruction (verbatim) ----------------
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

# -------------------------- inputs (paths) ----------------------------------
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
NB2231_SUMMARY = DATA_PROCESSED / "nb2231_summary.json"

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


# ============================================================================
# helpers (verbatim feature builder from nb2452)
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
    return agg.rename(columns={"src_first": "src"})


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


def _load_mordred_test(n_test_expected):
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(f"Mordred cache missing (nb1030 first): {mte_p}")
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


# ============================================================================
# adversarial classifier
# ============================================================================

def adversarial_oof_classifier(X_train_domain, X_test_domain, seed, n_folds):
    """5-fold OOF P(test|x) on the stack [X_train_domain ; X_test_domain].

    Returns: p_train_oof, p_test_oof, fold_auc (n_train,) (n_test,) list[float]
    """
    n_tr, n_te = X_train_domain.shape[0], X_test_domain.shape[0]
    X_all = np.concatenate([X_train_domain, X_test_domain], axis=0).astype(np.float32)
    y_all = np.concatenate([
        np.zeros(n_tr, dtype=np.int32),
        np.ones(n_te, dtype=np.int32),
    ])
    oof_p = np.full(n_tr + n_te, np.nan, dtype=np.float64)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_auc = []
    for fold, (tr_loc, va_loc) in enumerate(skf.split(X_all, y_all)):
        params = dict(ADV_LGBM)
        params["random_state"] = seed + fold
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_all[tr_loc], y_all[tr_loc])
        p_va = clf.predict_proba(X_all[va_loc])[:, 1]
        oof_p[va_loc] = p_va
        try:
            auc = roc_auc_score(y_all[va_loc], p_va)
        except Exception:
            auc = float("nan")
        fold_auc.append(float(auc))
    p_train = oof_p[:n_tr].astype(np.float32)
    p_test = oof_p[n_tr:].astype(np.float32)
    return p_train, p_test, fold_auc


# ============================================================================
# weighted residual cross-fit (per brief: max_depth=4, leaves=15, n=300, lr=0.03)
# ============================================================================

def _residual_cross_fit_weighted(X, residual, seed, sample_weight):
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        params = dict(PXR_LGBM)
        params["random_state"] = seed
        mdl = lgb.LGBMRegressor(**params)
        mdl.fit(X[tr_loc], residual[tr_loc],
                sample_weight=sample_weight[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _train_then_predict_te_weighted(X_unb, residual, X_te, seed, sample_weight):
    params = dict(PXR_LGBM)
    params["random_state"] = seed
    mdl = lgb.LGBMRegressor(**params)
    mdl.fit(X_unb, residual, sample_weight=sample_weight)
    return mdl.predict(X_te).astype(np.float32)


# ============================================================================
# pyramid SLSQP + rank-stretch (verbatim nb2452)
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
    print(f"{TAG} -- adversarial validation + IPS-style reweighting (from scratch)")
    print("=" * 78)

    # ---- Load truth, anchor, K=20 surviving features --------------------
    print("\n[stage A] load truth, anchor, K=20 surviving feature indices")
    with open(NB2231_SUMMARY) as f:
        nb2231 = json.load(f)
    surviving_K20 = list(nb2231["snapshots"]["20"]["surviving_idx_in_117"])
    surviving_K20_names = list(nb2231["snapshots"]["20"]["surviving_names"])
    assert len(surviving_K20) == 20

    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist() if "smiles" in te.columns
                 else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values if "name" in te.columns
                else te["Molecule Name"].values)
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
    print(f"   n_test={n_test}  n_unb={n_unb}  n_unique_scaffolds={n_unique_scaf}")
    print(f"   chemprop_aux in_RAE = {rae_anchor:.4f} (ref {CHEMPROP_AUX_REF:.4f})")

    # ---- Build 117-feature matrix on 513 test rows ----------------------
    print("\n[stage B] rebuild 117-feature matrix on 513 test rows")
    for p in (NB1352_SUMMARY, NB1392_SUMMARY, NB1484_SUMMARY,
              NB1523_SUMMARY, NB1524_SUMMARY, NB1541_SUMMARY):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")
    sum_1352 = json.load(open(NB1352_SUMMARY))
    sum_1484 = json.load(open(NB1484_SUMMARY))
    sum_1523 = json.load(open(NB1523_SUMMARY))
    sum_1524 = json.load(open(NB1524_SUMMARY))
    sum_1541 = json.load(open(NB1541_SUMMARY))
    sum_1392 = json.load(open(NB1392_SUMMARY))

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
        std_test_smiles.append("" if m is None else Chem.MolToSmiles(m))
    fp_test = morgan_fp_batch(std_test_smiles)
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )

    X_te_full117 = np.concatenate(
        [X_ap_te_top, X_maccs_te_top, X_mord_te_top, X_emb_te_top,
         X_av_te_top, pred_chembl_pec50.reshape(-1, 1).astype(np.float32),
         mean_sim.reshape(-1, 1).astype(np.float32)], axis=1
    ).astype(np.float32)
    assert X_te_full117.shape[1] == 117, f"got {X_te_full117.shape}"

    # Split into train-domain (253 unb rows) and test-domain (full 513)
    X_117_train = X_te_full117[unb_idx].astype(np.float32)      # (253, 117)
    X_117_test = X_te_full117.astype(np.float32)                # (513, 117)
    X_te_K20 = X_te_full117[:, surviving_K20].astype(np.float32)
    X_unb_K20 = X_te_K20[unb_idx]
    print(f"   X_117_train={X_117_train.shape}  X_117_test={X_117_test.shape}")
    print(f"   X_unb_K20={X_unb_K20.shape}  X_te_K20={X_te_K20.shape}")

    # ---- STAGE 1: adversarial classifier (train=0, test=1) --------------
    print("\n" + "-" * 78)
    print("[STAGE 1] adversarial LGBMClassifier  (train=253 label=0 vs test=513 label=1)")
    print("-" * 78)
    p_train_oof, p_test_oof, fold_auc = adversarial_oof_classifier(
        X_117_train, X_117_test, seed=ADV_SEED, n_folds=ADV_FOLDS,
    )
    auc_mean = float(np.mean(fold_auc))
    auc_std = float(np.std(fold_auc))
    print(f"   OOF AUC mean={auc_mean:.4f} +/- {auc_std:.4f}  "
          f"fold=[{', '.join(f'{a:.3f}' for a in fold_auc)}]")
    print(f"   p_train  mean={p_train_oof.mean():.4f}  std={p_train_oof.std():.4f}  "
          f"max={p_train_oof.max():.4f}  min={p_train_oof.min():.4f}")
    print(f"   p_test   mean={p_test_oof.mean():.4f}  std={p_test_oof.std():.4f}  "
          f"max={p_test_oof.max():.4f}  min={p_test_oof.min():.4f}")
    auc_interp = ("STRONG shift" if auc_mean > 0.75 else
                  "MODERATE shift" if auc_mean > 0.60 else
                  "MILD / no shift")
    print(f"   adv-AUC interpretation: {auc_interp}")

    np.save(DATA_PROCESSED / f"{TAG}_ptest_unb.npy", p_train_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_ptest_te.npy", p_test_oof.astype(np.float32))

    # ---- STAGE 2: IPS weights = P(test|x_tr) / max(P(test|x_te), tiny) ---
    print("\n" + "-" * 78)
    print("[STAGE 2] IPS density-ratio weights")
    print("-" * 78)
    num = p_train_oof.astype(np.float64)                     # (253,)
    den_scalar = max(float(p_test_oof.mean()), DENOM_TINY)   # scalar anchor
    w_raw = num / den_scalar                                 # density-ratio form
    w_unb = np.clip(w_raw, WEIGHT_CLIP_LO, WEIGHT_CLIP_HI).astype(np.float64)
    n_clip_lo = int((w_raw < WEIGHT_CLIP_LO).sum())
    n_clip_hi = int((w_raw > WEIGHT_CLIP_HI).sum())
    print(f"   denom (mean P(test|x_test)) = {den_scalar:.4f}")
    print(f"   w_raw  mean={w_raw.mean():.4f}  std={w_raw.std():.4f}  "
          f"min={w_raw.min():.4f}  max={w_raw.max():.4f}")
    print(f"   w_clip mean={w_unb.mean():.4f}  std={w_unb.std():.4f}  "
          f"min={w_unb.min():.4f}  max={w_unb.max():.4f}")
    print(f"   clipped: lo={n_clip_lo}/{n_unb}  hi={n_clip_hi}/{n_unb}")
    # weight histogram (deciles)
    deciles = np.percentile(w_unb, [10, 25, 50, 75, 90])
    print(f"   w deciles [10,25,50,75,90] = {np.round(deciles, 3).tolist()}")
    # top-10 train rows the classifier thinks are most test-like
    top10 = np.argsort(-w_unb)[:10]
    print("   top-10 train rows by IPS weight:")
    for i, j in enumerate(top10):
        print(f"     {i:2d}. row={j:3d}  p_train(test)={num[j]:.4f}  "
              f"w={w_unb[j]:.4f}  y={y_unb[j]:.3f}  anchor={anchor[j]:.3f}")
    np.save(DATA_PROCESSED / f"{TAG}_ips_weights_unb.npy", w_unb.astype(np.float32))

    # ---- STAGE 3: weighted PXR residual LGBM (max_depth=4 leaves=15 n=300 lr=0.03)
    print("\n" + "-" * 78)
    print(f"[STAGE 3] PXR residual LGBM with IPS sample_weight  "
          f"params=(max_depth=4, num_leaves=15, n_est=300, lr=0.03)")
    print(f"          KFold n={RESID_FOLDS}  seeds={RESID_SEEDS}")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_te_resid = np.zeros((len(RESID_SEEDS), n_test), dtype=np.float64)
    per_seed_rae = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        resid_oof = _residual_cross_fit_weighted(
            X_unb_K20, residual, s, sample_weight=w_unb,
        )
        per_seed_corrected[i] = anchor + resid_oof
        per_seed_rae.append(float(rae(y_unb, anchor + resid_oof)))
        per_seed_te_resid[i] = _train_then_predict_te_weighted(
            X_unb_K20, residual, X_te_K20, s, sample_weight=w_unb,
        )
        print(f"   seed={s:3d}: rae_corr={per_seed_rae[-1]:.4f}  "
              f"wall={time.time()-ts:.1f}s")

    mean_bag_oof_K20 = per_seed_corrected.mean(axis=0)
    mean_bag_te_resid_K20 = per_seed_te_resid.mean(axis=0)
    te_K20_513 = te_anchor_513 + mean_bag_te_resid_K20
    rae_K20_per_seed_mean = float(np.mean(per_seed_rae))
    rae_K20_mean_bag = float(rae(y_unb, mean_bag_oof_K20))
    print(f"\n   [K20-IPS] per-seed mean RAE = {rae_K20_per_seed_mean:.4f}")
    print(f"   [K20-IPS] mean-bag RAE      = {rae_K20_mean_bag:.4f}")
    print(f"   [K20-IPS] anchor in_RAE     = {rae_anchor:.4f}  "
          f"(delta {rae_K20_mean_bag - rae_anchor:+.4f})")

    # ---- STAGE 4: 5-anchor pyramid SLSQP + rank-stretch -----------------
    print("\n" + "=" * 78)
    print("[STAGE 4] 5-ANCHOR PYRAMID  (K20_IPS, chemprop_aux, nb1191, nb503, nb562)")
    print("=" * 78)
    nb1191_oof = reconstruct_nb1191_oof(n_unb)
    chemprop_oof = np.load(
        DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy").astype(np.float64)
    nb503_oof = np.load(DATA_PROCESSED / "nb503_pred_oof.npy").astype(np.float64)
    nb562_oof = np.load(DATA_PROCESSED / "nb562_pred_oof.npy").astype(np.float64)

    te_nb1191 = np.load(DATA_PROCESSED / "te_nb1191.npy").astype(np.float64)
    te_nb503 = np.load(DATA_PROCESSED / "te_nb503.npy").astype(np.float64)
    te_nb562 = np.load(DATA_PROCESSED / "te_nb562.npy").astype(np.float64)
    te_chemprop_aux = te_anchor_513

    anchors_list = [
        ("nb2682_K20",   mean_bag_oof_K20.astype(np.float64), te_K20_513.astype(np.float64)),
        ("chemprop_aux", chemprop_oof,                        te_chemprop_aux),
        ("nb1191",       nb1191_oof,                          te_nb1191),
        ("nb503",        nb503_oof,                           te_nb503),
        ("nb562",        nb562_oof,                           te_nb562),
    ]
    indiv_rae = {}
    oof_cols, te_cols = [], []
    for disp, oof, te_arr in anchors_list:
        assert oof.shape == (n_unb,), f"{disp} oof {oof.shape}"
        assert te_arr.shape == (n_test,), f"{disp} te {te_arr.shape}"
        r = float(rae(y_unb, oof))
        indiv_rae[disp] = r
        oof_cols.append(oof)
        te_cols.append(te_arr)
        print(f"   {disp:14s} oof_RAE={r:.4f}")
    P_unb = np.column_stack(oof_cols)
    P_te = np.column_stack(te_cols)

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

    # ---- DEPLOY (full-fit blend weights + stretch) ----------------------
    w_deploy = slsqp_simplex(P_unb, y_unb)
    blend_unb = P_unb @ w_deploy
    mu_deploy = float(blend_unb.mean())
    s_deploy = float(np.mean([s for r in per_seed for s in r["fold_s"]]))
    in_rae_final = float(rae(y_unb, mu_deploy + s_deploy * (blend_unb - mu_deploy)))
    blend_te = P_te @ w_deploy
    deploy_te = (mu_deploy + s_deploy * (blend_te - mu_deploy)).astype(np.float32)
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))
    w_str = ", ".join(f"{disp}={w:.4f}"
                       for (disp, _, _), w in zip(anchors_list, w_deploy))
    print(f"\n[deploy]  weights      = {w_str}")
    print(f"[deploy]  mu / s        = {mu_deploy:.4f} / {s_deploy:.4f}")
    print(f"[deploy]  in-sample RAE = {in_rae_final:.4f}")
    print(f"[deploy]  te[unb_idx] RAE = {te_unb_rae:.4f}")

    lb_band_est = LB_W_OOF * pooled_rae_mean_seeds + LB_W_TE * te_unb_rae

    # ---- STAGE 5: GATE --------------------------------------------------
    print("\n" + "-" * 78)
    print(f"GATE  thresholds: PROMOTE < {GATE_PROMOTE}, MARGINAL_BEAT < {GATE_MARGINAL}")
    print("-" * 78)
    if pooled_rae_mean_seeds < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif pooled_rae_mean_seeds < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    delta_promote = pooled_rae_mean_seeds - GATE_PROMOTE
    delta_marginal = pooled_rae_mean_seeds - GATE_MARGINAL
    print(f"   pooled_rae_mean_seeds = {pooled_rae_mean_seeds:.4f}")
    print(f"   delta_vs_PROMOTE     = {delta_promote:+.4f}")
    print(f"   delta_vs_MARGINAL    = {delta_marginal:+.4f}")
    print(f"   verdict              = {verdict}")
    print("-" * 78)

    # ---- save outputs ---------------------------------------------------
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy", mean_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy_te)
    sub_csv_path = SUBMISSIONS / f"{TAG}_adv_validation_weights.csv"
    if verdict in ("PROMOTE", "MARGINAL_BEAT"):
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": deploy_te,
        }).to_csv(sub_csv_path, index=False)
        print(f"[save] {sub_csv_path}  (gate {verdict})")
    else:
        print(f"[skip] gate FAIL -- no submission CSV")

    summary = {
        "tag": TAG,
        "method": "adversarial_validation_IPS_density_ratio_reweighting",
        "paradigm_diff_vs_nb2452": (
            "nb2452 used mean-1 normalisation of P(test); this uses explicit "
            "density-ratio num/den with denominator = mean P(test|x_test); "
            "explicit AUC report + per-row diagnostics."
        ),
        "stage_1_adv_classifier": {
            "n_folds": ADV_FOLDS,
            "seed": ADV_SEED,
            "feature_dim": 117,
            "n_train_rows": int(n_unb),
            "n_test_rows": int(n_test),
            "oof_auc_mean": auc_mean,
            "oof_auc_std": auc_std,
            "oof_auc_per_fold": fold_auc,
            "auc_interpretation": auc_interp,
            "p_train_mean": float(p_train_oof.mean()),
            "p_train_std": float(p_train_oof.std()),
            "p_train_max": float(p_train_oof.max()),
            "p_train_min": float(p_train_oof.min()),
            "p_test_mean": float(p_test_oof.mean()),
            "p_test_std": float(p_test_oof.std()),
            "p_test_max": float(p_test_oof.max()),
            "p_test_min": float(p_test_oof.min()),
        },
        "stage_2_ips_weights": {
            "form": "w = clip(P(test|x_tr) / max(mean(P(test|x_te)), tiny), 0.1, 10)",
            "denom_scalar": float(den_scalar),
            "clip_lo": WEIGHT_CLIP_LO,
            "clip_hi": WEIGHT_CLIP_HI,
            "n_clipped_lo": n_clip_lo,
            "n_clipped_hi": n_clip_hi,
            "w_raw_mean": float(w_raw.mean()),
            "w_raw_std": float(w_raw.std()),
            "w_raw_min": float(w_raw.min()),
            "w_raw_max": float(w_raw.max()),
            "w_clip_mean": float(w_unb.mean()),
            "w_clip_std": float(w_unb.std()),
            "w_clip_min": float(w_unb.min()),
            "w_clip_max": float(w_unb.max()),
            "w_deciles_10_25_50_75_90": [float(x) for x in deciles],
            "top10_train_rows_by_weight": [
                {"row": int(j), "p_train_test": float(num[j]),
                 "w": float(w_unb[j]), "y": float(y_unb[j]),
                 "anchor": float(anchor[j])}
                for j in top10
            ],
        },
        "stage_3_pxr_residual_lgbm": {
            "lgbm_params": {
                k: v for k, v in PXR_LGBM.items() if k != "verbosity"
            },
            "n_folds": RESID_FOLDS,
            "seeds": RESID_SEEDS,
            "rae_per_seed": per_seed_rae,
            "rae_per_seed_mean": rae_K20_per_seed_mean,
            "rae_mean_bag": rae_K20_mean_bag,
            "k20_surviving_idx_in_117": [int(j) for j in surviving_K20],
        },
        "stage_4_pyramid": {
            "anchors": [a[0] for a in anchors_list],
            "anchor_oof_rae_unb": indiv_rae,
            "rae_anchor_chemprop_aux": rae_anchor,
            "kf_seeds": KF_SEEDS,
            "n_folds": N_FOLDS,
            "stretch_grid": STRETCH_GRID,
            "n_unb": int(n_unb),
            "n_te": int(n_test),
            "n_unique_scaffolds": int(n_unique_scaf),
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
        },
        "gate": {
            "threshold_promote": GATE_PROMOTE,
            "threshold_marginal": GATE_MARGINAL,
            "pooled_rae_mean_seeds": pooled_rae_mean_seeds,
            "delta_vs_promote": delta_promote,
            "delta_vs_marginal": delta_marginal,
            "verdict": verdict,
        },
        "mean_rae": pooled_rae_mean_seeds,
        "verdict": verdict,
        "te_npy_path": str(DATA_PROCESSED / f"te_{TAG}.npy"),
        "pred_oof_npy_path": str(DATA_PROCESSED / f"{TAG}_pred_oof.npy"),
        "submission_csv": str(sub_csv_path) if verdict != "FAIL" else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   adv-classifier AUC      = {auc_mean:.4f} +/- {auc_std:.4f}  ({auc_interp})")
    print(f"   IPS w_clip range        = [{w_unb.min():.3f}, {w_unb.max():.3f}]")
    print(f"   IPS w_clip mean/std     = {w_unb.mean():.3f} / {w_unb.std():.3f}")
    print(f"   K=20 IPS per-seed RAE   = {rae_K20_per_seed_mean:.4f}")
    print(f"   pyramid pooled RAE      = {pooled_rae_mean_seeds:.4f} (+/- {pooled_rae_std_seeds:.4f})")
    print(f"   verdict                 = {verdict}")
    print(f"   LB band estimate        = {lb_band_est:.4f}")
    print(f"   wall                    = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "stage_1_adv_classifier",
        "stage_2_ips_weights",
        "mean_rae",
        "verdict",
        "submission_csv",
    ):
        v = res.get(k)
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                if isinstance(vv, list):
                    print(f"    {kk}: <list len {len(vv)}>")
                else:
                    print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")
