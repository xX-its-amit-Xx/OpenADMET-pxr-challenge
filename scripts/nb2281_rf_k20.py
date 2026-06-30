"""nb2281 -- RandomForest (bagged, axis-aligned, no boosting) on RFE K=20
features as chemprop_aux residual corrector.  Then SLSQP-blend into the
nb2240 pyramid {nb2240_K20, chemprop_aux, nb1191, nb503, nb562}.

HYPOTHESIS:
    Cycle 134 documented two K=20 attempts that FAILED:
      - nb1052 Bayesian Linear  (FAIL)
      - nb1053 GNN              (FAIL)
    Random Forest was NEVER tested at K=20.  RF differs from the boosting
    family (LGBM nb2240, XGB nb2270) along three orthogonal axes:
      1. PURE BAGGING -- trees fit independent bootstrap samples (no
         sequential residual chasing -> different bias-variance trade).
      2. AXIS-ALIGNED, NO L1/L2 -- splits chosen by variance reduction
         alone; no shrinkage / regularisation against a leading score.
      3. AVERAGED OVER MANY UNPRUNED TREES (n_estimators=500, depth=12)
         -- low-bias / high-variance components Pa-Lev-averaged.
    On the lean K=20 set (Mordred 4 / ChempropEmbed 8 / AtomPair 4 /
    MACCS 1 / Avalon 2 / ChEMBL_kNN 1) this is a different decomposition
    of the same signal than LGBM's leaf-wise gradient boosting.  If it
    extracts even slightly different residual structure, the SLSQP blend
    can pick up a non-zero weight.

PROTOCOL:
    1. Load K=20 RFE surviving feature indices from nb2231 (same exact
       set used by nb2240 / nb2270).
    2. Rebuild the 117-col 5-way K-tuned feature matrix on the 513 test
       compounds, slice to K=20 indices, take the unb_idx slice for the
       253 unblind.
    3. Anchor = chemprop_aux te[unb_idx]; residual = y_unb - anchor.
    4. RandomForestRegressor(n_estimators=500, max_depth=12,
       min_samples_leaf=5, random_state in {0, 1, 7, 42, 137}).
       For each seed:
         KFold(n=5, shuffle=True, random_state=seed) on the 253;
         OOF cross-fit -> per-seed residual_oof -> per-seed corrected
         pred = anchor + resid_oof.
       Aggregate: mean-bag OOF + median-bag OOF across 5 seeds.
       For each seed: refit on ALL 253, predict residual on full 513
       -> mean-bag te residual -> te_K20_513 = anchor + mean_bag.
    5. Compare standalone RF-K20 (mean_bag + median_bag) vs nb2240
       K=20 LGBM anchor mean_bag RAE 0.4630.
       Also compute RF+LGBM convex blend at alphas {0.0, 0.25, 0.5,
       0.75, 1.0} (alpha=1.0 -> pure RF, alpha=0.0 -> pure LGBM K=20).
    6. SLSQP-blend RF-K20 mean_bag into the nb2240 pyramid (drop-in
       replacement of the nb2240_K20 anchor) -- 5-anchor stack:
         0. nb2281_K20    NEW  (RF mean-bag K=20)
         1. chemprop_aux
         2. nb1191
         3. nb503
         4. nb562
       5-fold scaffold-CV across 5 seeds {1001..1005} with simplex
       SLSQP + rank-stretch grid 1.000..1.150.
       Gate: must beat 0.4601 by >= 0.003 -> deploy threshold 0.4571
       (strict = min(0.4598, 0.4601) - 0.003 = 0.4568).
    7. If gate beats: deep-30 verify (5 canonical + 25 extra seeds),
       write submissions/nb2281_rf_k20.csv (deploy refit on all 253).

ANCHORS:
    chemprop_aux in_RAE = 0.6216
    nb2240 K=20 LGBM mean_bag RAE = 0.4630
    nb2240 pyramid pooled (5-seed mean) RAE = 0.4598
    GATE = min(0.4598, 0.4601) - 0.003 = 0.4568

Outputs:
    scripts/nb2281_rf_k20.py
    data/processed/nb2281_summary.json
    data/processed/nb2281_mean_bag_oof_K20.npy        (253,) float32
    data/processed/nb2281_median_bag_oof_K20.npy      (253,) float32
    data/processed/te_nb2281_K20.npy                  (513,) float32
    submissions/nb2281_rf_k20.csv                     (only if gate pass)
    data/processed/te_nb2281.npy                      (only if deploy)
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
import sklearn
from rdkit import Chem
from rdkit import RDLogger
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2281"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# ------------------------------ residual config ------------------------------
RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

# RF hyperparams (per task spec)
RF_N_ESTIMATORS = 500
RF_MAX_DEPTH = 12
RF_MIN_SAMPLES_LEAF = 5

# ------------------------------ pyramid config -------------------------------
GATE_NB2240_PYRAMID = 0.4598   # nb2240 pooled_rae_mean_seeds (5 kf seeds)
GATE_TASK_REF = 0.4601         # task-wording "0.003 below 0.4601"
GATE_MARGIN = 0.003
# strict threshold = min(GATE_NB2240_PYRAMID, GATE_TASK_REF) - margin
GATE_THRESHOLD = min(GATE_NB2240_PYRAMID, GATE_TASK_REF) - GATE_MARGIN

N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]
LB_W_OOF = 0.51
LB_W_TE = 0.49

# nb1191 reconstruction parameters (copied from nb2240/nb2270)
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
NB2240_K20_MEAN_BAG_REF = 0.4630      # standalone anchor reference
NB2240_PYRAMID_REF = 0.4598           # 5-seed pooled pyramid

BLEND_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]

# ------------------------------ paths ----------------------------------------
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
NB2240_K20_OOF = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6


# ============================================================================
# helpers (copy/specialised from nb2270/nb2240/nb2103)
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


def _rf_params(seed: int) -> dict:
    return dict(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        random_state=seed,
        n_jobs=2,
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


def _residual_cross_fit_rf(X: np.ndarray, residual: np.ndarray, seed: int) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = RandomForestRegressor(**_rf_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _train_full_then_predict_te_rf(
    X_unb: np.ndarray, residual: np.ndarray, X_te: np.ndarray, seed: int
) -> np.ndarray:
    mdl = RandomForestRegressor(**_rf_params(seed))
    mdl.fit(X_unb, residual)
    return mdl.predict(X_te).astype(np.float32)


# ============================================================================
# stage 2 utils (SLSQP + rank-stretch -- copied from nb2270/nb2240)
# ============================================================================

def slsqp_simplex(P: np.ndarray, y: np.ndarray) -> np.ndarray:
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

def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- RandomForest on RFE K=20 features as chemprop_aux residual")
    print(f"          RF(n_est={RF_N_ESTIMATORS}, depth={RF_MAX_DEPTH}, "
          f"min_leaf={RF_MIN_SAMPLES_LEAF}) seeds={RESID_SEEDS} "
          f"folds={RESID_FOLDS}")
    print(f"          ref: nb2240 K=20 LGBM mean_bag {NB2240_K20_MEAN_BAG_REF}  "
          f"pyramid {NB2240_PYRAMID_REF}  gate_thresh {GATE_THRESHOLD:.4f}")
    print("=" * 78)

    # ---- Load K=20 surviving indices ----
    if not NB2231_SUMMARY.exists():
        raise FileNotFoundError(f"missing {NB2231_SUMMARY}")
    with open(NB2231_SUMMARY) as f:
        nb2231 = json.load(f)
    surviving_K20 = list(nb2231["snapshots"]["20"]["surviving_idx_in_117"])
    surviving_K20_names = list(nb2231["snapshots"]["20"]["surviving_names"])
    surviving_K20_family_counts = dict(nb2231["snapshots"]["20"]["family_counts"])
    assert len(surviving_K20) == 20, f"expected 20 features, got {len(surviving_K20)}"
    print(f"[load] K=20 RFE surviving features ({len(surviving_K20)} cols):")
    for j, (idx, nm) in enumerate(zip(surviving_K20, surviving_K20_names)):
        print(f"   {j:2d}. idx={idx:3d}  {nm}")
    print(f"[load] K=20 family_counts = {surviving_K20_family_counts}")

    # ---- Load truth + anchor ----
    te = load_test()
    n_test = len(te)
    te_smiles = te["smiles"].astype(str).tolist() if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    if "Molecule Name" in te.columns:
        mol_names = te["Molecule Name"].astype(str).tolist()
    elif "name" in te.columns:
        mol_names = te["name"].astype(str).tolist()
    else:
        raise KeyError("no name column on test set")
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
    print(f"[resid] mean={residual.mean():+.4f} std={residual.std():.4f}")

    # ---- Rebuild 117-col 5-way feature matrix on the 513 ----
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
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL (union)")
    print("-" * 78)
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
    print(f"   final pool size: {len(pool)}  median pEC50 = {pool_median:.3f}")
    std_test_smiles = [Chem.MolToSmiles(m) if m is not None else "" for m in test_mols]
    fp_test = morgan_fp_batch(std_test_smiles)
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )

    # Full 117-col matrix on 513
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
    feat_dim_full = X_te_full.shape[1]
    assert feat_dim_full == 117, f"feat_dim {feat_dim_full} != 117"
    print(f"[feat] X_te_full = {X_te_full.shape}")

    # Slice to K=20
    X_te_K20 = X_te_full[:, surviving_K20].astype(np.float32)
    X_unb_K20 = X_te_K20[unb_idx]
    print(f"[feat] X_unb_K20 = {X_unb_K20.shape}  X_te_K20 = {X_te_K20.shape}")

    # ---- RF residual cross-fit: 5-seed bag, 5-fold CV ----
    print("\n" + "=" * 78)
    print(f"RF RESIDUAL CROSS-FIT  seeds={RESID_SEEDS}  folds={RESID_FOLDS}")
    print("=" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_te_resid = np.zeros((len(RESID_SEEDS), n_test), dtype=np.float64)
    per_seed_records = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        resid_oof = _residual_cross_fit_rf(X_unb_K20, residual, s)
        pred_corr = anchor + resid_oof
        per_seed_corrected[i] = pred_corr
        rae_s = float(rae(y_unb, pred_corr))
        te_resid_s = _train_full_then_predict_te_rf(
            X_unb_K20, residual, X_te_K20, s
        )
        per_seed_te_resid[i] = te_resid_s
        wall = time.time() - ts
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_chemprop_aux": rae_s - rae_anchor,
            "resid_oof_std": float(resid_oof.std()),
            "resid_oof_mean": float(resid_oof.mean()),
            "wall_sec": round(wall, 2),
        })
        print(f"   seed={s:3d}: rae_corr={rae_s:.4f}  "
              f"(d_vs_anchor={rae_s - rae_anchor:+.4f})  "
              f"resid_oof_std={resid_oof.std():.4f}  wall={wall:.1f}s")

    mean_bag_oof_K20 = per_seed_corrected.mean(axis=0)         # 253
    median_bag_oof_K20 = np.median(per_seed_corrected, axis=0)  # 253
    mean_bag_te_resid_K20 = per_seed_te_resid.mean(axis=0)      # 513
    median_bag_te_resid_K20 = np.median(per_seed_te_resid, axis=0)
    te_K20_513_mean = te_anchor_513 + mean_bag_te_resid_K20
    te_K20_513_median = te_anchor_513 + median_bag_te_resid_K20

    rae_rf_mean_bag = float(rae(y_unb, mean_bag_oof_K20))
    rae_rf_median_bag = float(rae(y_unb, median_bag_oof_K20))
    rae_per_seed_mean = float(np.mean([r["rae_corrected"] for r in per_seed_records]))
    print(f"\n[RF-K20] mean-bag RAE      = {rae_rf_mean_bag:.4f}  "
          f"(vs LGBM K=20 {NB2240_K20_MEAN_BAG_REF:.4f}: "
          f"{rae_rf_mean_bag - NB2240_K20_MEAN_BAG_REF:+.4f})")
    print(f"[RF-K20] median-bag RAE    = {rae_rf_median_bag:.4f}  "
          f"(vs LGBM K=20 {NB2240_K20_MEAN_BAG_REF:.4f}: "
          f"{rae_rf_median_bag - NB2240_K20_MEAN_BAG_REF:+.4f})")
    print(f"[RF-K20] per-seed mean     = {rae_per_seed_mean:.4f}")
    print(f"[RF-K20] anchor in_RAE     = {rae_anchor:.4f}")

    oof_K20_path = DATA_PROCESSED / f"{TAG}_mean_bag_oof_K20.npy"
    median_oof_K20_path = DATA_PROCESSED / f"{TAG}_median_bag_oof_K20.npy"
    te_K20_path = DATA_PROCESSED / f"te_{TAG}_K20.npy"
    np.save(oof_K20_path, mean_bag_oof_K20.astype(np.float32))
    np.save(median_oof_K20_path, median_bag_oof_K20.astype(np.float32))
    np.save(te_K20_path, te_K20_513_mean.astype(np.float32))
    print(f"[save] {oof_K20_path}")
    print(f"[save] {median_oof_K20_path}")
    print(f"[save] {te_K20_path}")

    # ---- RF + LGBM K=20 convex blend ----
    print("\n" + "-" * 78)
    print(f"RF + LGBM-K20 CONVEX BLEND  alphas={BLEND_ALPHAS}")
    print("-" * 78)
    if not NB2240_K20_OOF.exists():
        raise FileNotFoundError(f"missing {NB2240_K20_OOF} -- run nb2240 first")
    lgbm_K20_oof = np.load(NB2240_K20_OOF).astype(np.float64)
    rae_lgbm_K20 = float(rae(y_unb, lgbm_K20_oof))
    print(f"   cached LGBM K=20 OOF RAE = {rae_lgbm_K20:.4f}  "
          f"(ref {NB2240_K20_MEAN_BAG_REF:.4f})")
    blend_results = []
    for alpha in BLEND_ALPHAS:
        blend_oof = alpha * mean_bag_oof_K20 + (1.0 - alpha) * lgbm_K20_oof
        rae_b = float(rae(y_unb, blend_oof))
        beats_lgbm = rae_b < NB2240_K20_MEAN_BAG_REF - GATE_MARGIN
        flag = "BEATS_LGBM_K20" if beats_lgbm else ""
        print(f"   blend  alpha={alpha:.2f}  rae={rae_b:.4f}  "
              f"d_vs_LGBM={rae_b - NB2240_K20_MEAN_BAG_REF:+.4f}  {flag}")
        blend_results.append({
            "alpha": float(alpha),
            "rae": rae_b,
            "delta_vs_lgbm_K20": rae_b - NB2240_K20_MEAN_BAG_REF,
            "beats_lgbm_K20": bool(beats_lgbm),
        })

    # ---- Standalone verdict (vs LGBM K=20 anchor) ----
    best_standalone_rae = min(
        rae_rf_mean_bag,
        rae_rf_median_bag,
        *[br["rae"] for br in blend_results],
    )
    delta_standalone_vs_lgbm = best_standalone_rae - NB2240_K20_MEAN_BAG_REF
    if delta_standalone_vs_lgbm < -GATE_MARGIN:
        standalone_verdict = "BEATS_LGBM_K20"
    elif abs(delta_standalone_vs_lgbm) <= GATE_MARGIN:
        standalone_verdict = "FLAT_VS_LGBM_K20"
    else:
        standalone_verdict = "HURTS_VS_LGBM_K20"
    print(f"\n[standalone] best_RAE={best_standalone_rae:.4f}  "
          f"d_vs_LGBM_K20={delta_standalone_vs_lgbm:+.4f}  "
          f"verdict={standalone_verdict}")

    # ============================================================================
    # Stage 2: 5-anchor pyramid SLSQP (RF K=20 replaces nb2240_K20)
    # ============================================================================
    print("\n" + "=" * 78)
    print("STAGE 2: 5-ANCHOR PYRAMID  (nb2281_RF_K20 swaps in for nb2240_K20)")
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
        ("nb2281_RF_K20",  mean_bag_oof_K20.astype(np.float64), te_K20_513_mean.astype(np.float64)),
        ("chemprop_aux",   chemprop_oof,                         te_chemprop_aux),
        ("nb1191",         nb1191_oof,                           te_nb1191),
        ("nb503",          nb503_oof,                            te_nb503),
        ("nb562",          nb562_oof,                            te_nb562),
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
        print(f"   {disp:16s} oof_RAE={r:.4f}  te_mean={te_arr.mean():.3f}  "
              f"te_std={te_arr.std():.3f}")
    P_unb = np.column_stack(oof_cols)
    P_te = np.column_stack(te_cols)
    K_anch = P_unb.shape[1]
    print(f"[stack] P_unb {P_unb.shape}  P_te {P_te.shape}  K={K_anch}")

    # ---- Scaffold 5-fold CV across 5 seeds ----
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
        print(f"   seed={kf_seed}  pooled_RAE={pooled:.4f}  "
              f"mean_s={np.mean(fs):.3f}  "
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

    # ---- Gate vs nb2240 pyramid ref ----
    delta_vs_nb2240_pyr = pooled_rae_mean_seeds - NB2240_PYRAMID_REF
    delta_vs_task_ref = pooled_rae_mean_seeds - GATE_TASK_REF
    gate_beat = pooled_rae_mean_seeds < GATE_THRESHOLD
    gate_flat = abs(delta_vs_nb2240_pyr) <= GATE_MARGIN
    print("\n" + "-" * 78)
    print("GATE EVALUATION  (vs nb2240 pyramid 0.4598 + task ref 0.4601, "
          f"margin {GATE_MARGIN})")
    print("-" * 78)
    print(f"   nb2281 OOF (5-seed mean)  = {pooled_rae_mean_seeds:.4f}")
    print(f"   nb2240 pyramid reference  = {NB2240_PYRAMID_REF:.4f}  "
          f"delta = {delta_vs_nb2240_pyr:+.4f}")
    print(f"   task ref                  = {GATE_TASK_REF:.4f}  "
          f"delta = {delta_vs_task_ref:+.4f}")
    print(f"   strict gate threshold     = {GATE_THRESHOLD:.4f}")
    if gate_beat:
        verdict = "BEATS_GATE_DEPLOY"
    elif gate_flat:
        verdict = "FLAT_VS_NB2240_PYRAMID"
    else:
        verdict = "HURTS_VS_NB2240_PYRAMID"
    print(f"   verdict                  = {verdict}")

    # ---- Deep-30 verify on gate-pass ----
    deep30 = None
    if gate_beat:
        print("\n" + "-" * 78)
        print("DEEP-30 VERIFY  (5 canonical + 25 extra seeds)")
        print("-" * 78)
        deep30 = deep_verify_seeds(P_unb, y_unb, unb_scaffolds, n_extra=25)
        print(f"   n_seeds={deep30['n_seeds']}  mean_RAE={deep30['mean_rae']:.4f}  "
              f"std={deep30['std_rae']:.4f}  "
              f"range=[{deep30['min_rae']:.4f}, {deep30['max_rae']:.4f}]")

    # ---- Save te artefact + (optional) submission CSV ----
    submission_csv = None
    te_npy_path = None
    if gate_beat:
        te_npy_path = DATA_PROCESSED / f"te_{TAG}.npy"
        np.save(te_npy_path, deploy_te)
        print(f"\n[save] {te_npy_path}")
        sub_csv_path = SUBMISSIONS / f"{TAG}_rf_k20.csv"
        df_sub = pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": mol_names,
            "pEC50": deploy_te,
        })
        if len(df_sub) != 513:
            raise ValueError(f"submission rows {len(df_sub)} != 513")
        df_sub.to_csv(sub_csv_path, index=False)
        print(f"[save] {sub_csv_path}  (gate BEATS, {len(df_sub)} rows)")
        submission_csv = str(sub_csv_path)
    else:
        print(f"\n[skip] gate not beat -- no te_nb2281.npy / submission CSV written "
              f"({verdict})")

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "method": ("random_forest_RFE_K20_residual_5seed_bag_with_nb2240_pyramid_slsqp"),
        "anchor": ANCHOR,
        "anchor_path": str(ANCHOR_TE_PATH),
        "data_source": ("nb2231 RFE K=20 surviving features of the 117-col 5-way "
                        "K-tuned matrix (AtomPair / MACCS / Mordred / ChempropEmbed "
                        "/ Avalon + ChEMBL kNN + mean sim)"),
        "K": 20,
        "K_family_counts": surviving_K20_family_counts,
        "K_surviving_idx_in_117": [int(j) for j in surviving_K20],
        "K_surviving_names": surviving_K20_names,
        "model_family": "RandomForest",
        "rf_n_estimators": RF_N_ESTIMATORS,
        "rf_max_depth": RF_MAX_DEPTH,
        "rf_min_samples_leaf": RF_MIN_SAMPLES_LEAF,
        "sklearn_version": sklearn.__version__,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "n_test": n_test,
        "n_unb": n_unb,
        "n_unique_scaffolds": n_unique_scaf,
        "rae_anchor_chemprop_aux": rae_anchor,
        "rae_chemprop_aux_ref": CHEMPROP_AUX_REF,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        # Standalone RF
        "rf_per_seed_records": per_seed_records,
        "rae_rf_mean_bag": rae_rf_mean_bag,
        "rae_rf_median_bag": rae_rf_median_bag,
        "rae_per_seed_mean": rae_per_seed_mean,
        "delta_rf_mean_bag_vs_lgbm_K20": rae_rf_mean_bag - NB2240_K20_MEAN_BAG_REF,
        "delta_rf_median_bag_vs_lgbm_K20": rae_rf_median_bag - NB2240_K20_MEAN_BAG_REF,
        # Blend
        "blend_alphas": BLEND_ALPHAS,
        "blend_results": blend_results,
        "best_standalone_rae": best_standalone_rae,
        "delta_best_standalone_vs_lgbm_K20": delta_standalone_vs_lgbm,
        "standalone_verdict": standalone_verdict,
        "nb2240_K20_mean_bag_ref": NB2240_K20_MEAN_BAG_REF,
        # Pyramid
        "pyramid_anchors": [a[0] for a in anchors_list],
        "pyramid_anchor_oof_rae": indiv_rae,
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "stretch_grid": STRETCH_GRID,
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
        "compare_nb2240_pyramid_ref": NB2240_PYRAMID_REF,
        "delta_vs_nb2240_pyramid": delta_vs_nb2240_pyr,
        "compare_task_ref": GATE_TASK_REF,
        "delta_vs_task_ref": delta_vs_task_ref,
        "gate_margin": GATE_MARGIN,
        "gate_threshold_strict": GATE_THRESHOLD,
        "gate_beat": bool(gate_beat),
        "gate_flat": bool(gate_flat),
        "verdict": verdict,
        "deep_30_verify": deep30,
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "te_npy_path": str(te_npy_path) if te_npy_path is not None else None,
        "submission_csv": submission_csv,
        "oof_K20_path": str(oof_K20_path),
        "median_oof_K20_path": str(median_oof_K20_path),
        "te_K20_path": str(te_K20_path),
        "pre_unblind_clean": True,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   RF K=20 mean_bag RAE           = {rae_rf_mean_bag:.4f}")
    print(f"   RF K=20 median_bag RAE         = {rae_rf_median_bag:.4f}")
    print(f"   RF K=20 per-seed mean          = {rae_per_seed_mean:.4f}")
    print(f"   best standalone (incl. blend)  = {best_standalone_rae:.4f}  "
          f"vs LGBM K=20 {NB2240_K20_MEAN_BAG_REF:.4f} "
          f"({delta_standalone_vs_lgbm:+.4f}) -> {standalone_verdict}")
    print(f"   pyramid pooled RAE (5 seeds)   = {pooled_rae_mean_seeds:.4f}")
    print(f"   delta vs nb2240 pyramid 0.4598 = {delta_vs_nb2240_pyr:+.4f}")
    print(f"   delta vs task ref 0.4601       = {delta_vs_task_ref:+.4f}")
    print(f"   gate threshold strict          = {GATE_THRESHOLD:.4f}")
    print(f"   verdict                        = {verdict}")
    print(f"   LB band                        = {lb_band_est:.4f}")
    print(f"   wall                           = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "rae_anchor_chemprop_aux",
        "rae_rf_mean_bag",
        "rae_rf_median_bag",
        "delta_rf_mean_bag_vs_lgbm_K20",
        "best_standalone_rae",
        "standalone_verdict",
        "pooled_rae_mean_seeds",
        "delta_vs_nb2240_pyramid",
        "delta_vs_task_ref",
        "gate_threshold_strict",
        "gate_beat",
        "verdict",
        "deploy_weights",
        "deploy_s",
        "lb_band_estimate",
        "submission_csv",
        "te_npy_path",
        "wall_sec",
    ):
        print(f"  {k}: {res.get(k)}")
    print("\n==== BLEND TABLE ====")
    for br in res["blend_results"]:
        flag = "BEATS_LGBM_K20" if br["beats_lgbm_K20"] else ""
        print(f"  alpha={br['alpha']:.2f}  rae={br['rae']:.4f}  "
              f"d_vs_LGBM_K20={br['delta_vs_lgbm_K20']:+.4f}  {flag}")
