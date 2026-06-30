"""nb1673 -- Outer-bag VALIDATION of nb1632 with 10 outer seeds (instead of 5).

PROTOCOL
    1. For each outer seed o in {0, 1, 2, 7, 42, 99, 137, 250, 500, 750}: rebuild
         - nb1561_o : 5-inner-seed CatBoost bag on 117-col 5-way K-tuned
           features (mirror of nb1561 per-outer "mean_bag").
         - nb1612_o : 6-way SLSQP blend (rebuild 6 residual learners on
           AtomPair / MACCS / Mordred / ChempropEmbed / Avalon / ChemBERTa,
           SHAP-prune per outer, 5-inner-seed LGBM bag, naive 1/6 mean +
           5-fold SLSQP cross-fit, pick best variant -- mirror of nb1623).
         - blend_o = 0.55 * nb1561_o + 0.45 * nb1612_o
    2. Per-outer pooled RAE on the 253 PRE-unblind set.
    3. Row-level Bag-of-Bags MEAN + MEDIAN across the 10 blend_o vectors.
    4. Verdict vs nb1632 5-outer reference (0.5107) at 0.003 margin.

OUTPUTS
    scripts/nb1673_bag_nb1632_10seed.py            (this file)
    data/processed/nb1673_summary.json
    data/processed/nb1673_bob_mean_oof.npy         (253,) float32
    data/processed/nb1673_bob_median_oof.npy       (253,) float32
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
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1673"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# nb1622 best grid weight (CatBoost-side w on nb1561 bag)
W_NB1561 = 0.55
W_NB1612 = 1.0 - W_NB1561

# nb1632 5-outer reference RAE (BoB MEAN)
NB1632_REF = 0.5107
REPRODUCE_MARGIN = 0.003

# Outer bag definition -- 10 seeds (tighter estimate)
OUTER_SEEDS = [0, 1, 2, 7, 42, 99, 137, 250, 500, 750]
INNER_OFFSETS = [0, 1, 7, 42, 137]
RESID_FOLDS = 5
SLSQP_FOLDS = 5
SLSQP_SEED = 42

# ---- Feature caches (same as nb1561 / nb1612 / nb1623 / nb1632) ----
ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"
CHEMBERTA_TE_PATH = DATA_PROCESSED / "chemberta_test_emb.npy"
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

# nb1612 / nb1623 K-tuning (ChemBERTa @ K=50 from nb1612_summary best_K)
TOP_K_NB1612 = {
    "AtomPair": 25,
    "MACCS": 20,
    "Mordred": 20,
    "ChempropEmbed": 20,
    "Avalon": 30,
    "ChemBERTa": 50,
}
FAMILIES_NB1612 = ["AtomPair", "MACCS", "Mordred", "ChempropEmbed",
                   "Avalon", "ChemBERTa"]


# ---- ChEMBL pool / kNN helpers (same recipe used by nb1561 + nb1612) ----
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


def _tanimoto_topk(fp_q: np.ndarray, fp_pool: np.ndarray, k: int):
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


def _knn_predict(top_idx: np.ndarray, top_sim: np.ndarray,
                 pool_labels: np.ndarray, fallback: float):
    w = top_sim.copy()
    w = np.clip(w, 0.0, 1.0)
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


def _load_mordred_test(n_test_expected: int) -> np.ndarray:
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(
            f"Mordred cache missing -- run nb1030 first ({mte_p})"
        )
    X_te_m = np.load(mte_p).astype(np.float32)
    if X_te_m.shape[0] != n_test_expected:
        raise ValueError(
            f"Mordred test shape mismatch: {X_te_m.shape} vs "
            f"n_test={n_test_expected}"
        )
    X_te_m = np.where(np.isfinite(X_te_m), X_te_m, np.nan).astype(np.float32)
    col_med = np.nanmedian(X_te_m, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X_te_m)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X_te_m[idx_r, idx_c] = col_med[idx_c]
    return X_te_m


def _load_npy_test(path: Path, n_test_expected: int,
                   zero_fill: bool = True) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"missing cache: {path}")
    X = np.load(path)
    if X.shape[0] != n_test_expected:
        raise ValueError(f"shape mismatch {path}: {X.shape}")
    X = X.astype(np.float32)
    if zero_fill:
        X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X


def _load_family_te_nb1612(family: str, n_test: int) -> np.ndarray:
    if family == "AtomPair":
        return _load_npy_test(ATOMPAIR_TE_PATH, n_test, zero_fill=False)
    if family == "MACCS":
        return _load_npy_test(MACCS_TE_PATH, n_test, zero_fill=False)
    if family == "Mordred":
        return _load_mordred_test(n_test)
    if family == "ChempropEmbed":
        return _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test, zero_fill=True)
    if family == "Avalon":
        return _load_npy_test(AVALON_TE_PATH, n_test, zero_fill=False)
    if family == "ChemBERTa":
        return _load_npy_test(CHEMBERTA_TE_PATH, n_test, zero_fill=True)
    raise ValueError(f"unknown family: {family}")


# ---- CatBoost (nb1561 / nb1554) recipe ----
def _cat_params(seed: int) -> dict:
    return dict(
        loss_function="MAE",
        depth=4,
        iterations=200,
        learning_rate=0.05,
        l2_leaf_reg=5.0,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=2,
    )


def _cat_residual_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray,
                                     seed: int) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = CatBoostRegressor(**_cat_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


# ---- LGBM (nb1612 / nb1623) recipe ----
def _lgbm_params(seed: int) -> dict:
    return dict(
        objective="huber",
        alpha=1.0,
        learning_rate=0.05,
        n_estimators=80,
        max_depth=3,
        num_leaves=7,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        verbosity=-1,
        random_state=seed,
        n_jobs=2,
    )


def _lgbm_residual_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray,
                                      seed: int) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _compute_shap_importance(X: np.ndarray, residual: np.ndarray, seed: int = 0):
    mdl = LGBMRegressor(**_lgbm_params(seed))
    mdl.fit(X, residual)
    try:
        import shap
        explainer = shap.TreeExplainer(mdl)
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[0]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[..., 0]
        imp = np.abs(sv).mean(axis=0)
        return imp.astype(np.float64), "shap_tree_explainer"
    except Exception as e:
        print(f"   [shap] WARN: shap failed ({e}); falling back to LGBM gain")
        imp = mdl.booster_.feature_importance(importance_type="gain")
        return imp.astype(np.float64), "lgbm_gain_fallback"


def _run_family_nb1612(family: str, X_fam_unb: np.ndarray,
                       pred_chembl_unb: np.ndarray, mean_sim_unb: np.ndarray,
                       anchor: np.ndarray, residual: np.ndarray,
                       top_k: int, inner_seeds: list, outer_seed: int) -> dict:
    """Mirror of nb1623._run_family: SHAP-prune per outer + 5-inner-seed LGBM
    bag on residual. Returns mean-bag corrected OOF (anchor + mean(resid_oof))."""
    n_fam = int(X_fam_unb.shape[1])
    X_full = np.concatenate(
        [X_fam_unb, pred_chembl_unb.reshape(-1, 1), mean_sim_unb.reshape(-1, 1)],
        axis=1,
    ).astype(np.float32)
    imp_full, imp_src = _compute_shap_importance(X_full, residual,
                                                 seed=outer_seed)
    fam_imp = imp_full[:n_fam]
    top_k_eff = min(top_k, n_fam)
    top_order = np.argsort(-fam_imp)
    top_idx = top_order[:top_k_eff].astype(int)
    X_fam_pruned = X_fam_unb[:, top_idx]
    X_pruned = np.concatenate(
        [X_fam_pruned, pred_chembl_unb.reshape(-1, 1),
         mean_sim_unb.reshape(-1, 1)],
        axis=1,
    ).astype(np.float32)

    n_unb = anchor.shape[0]
    per_seed_corrected = np.zeros((len(inner_seeds), n_unb), dtype=np.float64)
    for i, s in enumerate(inner_seeds):
        resid_oof_s = _lgbm_residual_cross_fit_one_seed(X_pruned, residual, s)
        per_seed_corrected[i] = anchor + resid_oof_s
    mean_bag_oof = per_seed_corrected.mean(axis=0)
    return {
        "family": family,
        "n_fam_bits": n_fam,
        "top_k": int(top_k_eff),
        "shap_source": imp_src,
        "mean_bag_oof": mean_bag_oof.astype(np.float64),
    }


def _slsqp_blend_weights(P_tr: np.ndarray, y_tr: np.ndarray) -> np.ndarray:
    K = P_tr.shape[1]
    w0 = np.full(K, 1.0 / K)

    def _loss(w: np.ndarray) -> float:
        r = y_tr - P_tr @ w
        return float(np.mean(r * r))

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bnds = [(0.0, 1.0)] * K
    res = minimize(
        _loss, w0, method="SLSQP",
        bounds=bnds, constraints=cons,
        options={"ftol": 1e-10, "maxiter": 500},
    )
    w = np.clip(np.asarray(res.x, dtype=np.float64), 0.0, 1.0)
    s = w.sum()
    if s <= 0:
        return np.full(K, 1.0 / K)
    return w / s


def _slsqp_cross_fit(P: np.ndarray, y: np.ndarray,
                     n_splits: int, seed: int):
    n = len(y)
    oof = np.full(n, np.nan, dtype=np.float64)
    folds = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for f, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n))):
        w = _slsqp_blend_weights(P[tr_loc], y[tr_loc])
        oof[va_loc] = P[va_loc] @ w
        folds.append({"fold": int(f), "w": [float(x) for x in w.tolist()]})
    return oof, folds


def _extract_atompair_top_idx_from_nb1484(sum_1484: dict) -> np.ndarray:
    for f in sum_1484["families"]:
        if f["family"] == "AtomPair":
            return np.array(f["top_idx_ranked"], dtype=int)
    raise KeyError("AtomPair entry not found in nb1484_summary.json")


def _extract_best_K_record(sum_dict: dict, records_key: str,
                           best_K_key: str = "best_K") -> dict:
    best_K = int(sum_dict[best_K_key])
    for r in sum_dict[records_key]:
        if int(r["K"]) == best_K:
            return r
    raise KeyError(f"best_K {best_K} not found in {records_key}")


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- OUTER-BAG VALIDATION of nb1632 (10 outer seeds)")
    print(f"         outer seeds      = {OUTER_SEEDS}")
    print(f"         inner offsets    = {INNER_OFFSETS}")
    print(f"         w_nb1561 (CatBoost) = {W_NB1561:.2f}")
    print(f"         w_nb1612 (ChemBERTa) = {W_NB1612:.2f}")
    print(f"         nb1632_ref       = {NB1632_REF:.4f}  "
          f"margin = {REPRODUCE_MARGIN}")
    print("=" * 78)

    # ---- Truth + indices + anchor ----
    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"\n[load] n_test={n_test}  n_unb={n_unb}")

    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"chemprop_aux te file missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Load nb1554/nb1561 K-tuning summaries ----
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

    top_maccs_bit_idx = np.array(
        sum_1352["top_maccs_bit_indices_ranked"], dtype=int
    )
    rec_mord = _extract_best_K_record(sum_1523, "per_K_records",
                                       best_K_key="best_K")
    K_Mord_best = int(rec_mord["K"])
    top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)
    full_ap_ranked = _extract_atompair_top_idx_from_nb1484(sum_1484)
    K_AP_best = int(sum_1524["best_K"])
    top_ap_bit_idx = full_ap_ranked[:K_AP_best]
    K_Embed_best = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed_col_idx = top_embed_full[:K_Embed_best]
    top_avalon_bit_idx = np.array(
        sum_1392["top_avalon_bit_indices_ranked"], dtype=int
    )
    K_Avalon_used = int(len(top_avalon_bit_idx))

    n_top_maccs = int(len(top_maccs_bit_idx))
    n_top_mord = int(len(top_mord_col_idx))
    n_top_ap = int(len(top_ap_bit_idx))
    n_top_embed = int(len(top_embed_col_idx))
    n_top_avalon = int(K_Avalon_used)
    print(f"\n[reuse] AP={n_top_ap}  MACCS={n_top_maccs}  Mord={n_top_mord}  "
          f"Embed={n_top_embed}  Avalon={n_top_avalon}")

    # ---- ChEMBL pool + kNN (built ONCE; reused by both 1561 and 1612 paths) ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL + kNN feature build (once, shared by both branches)")
    print("-" * 78)
    pool = _load_chembl_pool()
    test_mols = [standardize(s) for s in test_smiles]
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

    std_test_smiles = [Chem.MolToSmiles(m) if m is not None else ""
                       for m in test_mols]
    fp_test = morgan_fp_batch(std_test_smiles)
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )
    pred_chembl_unb = pred_chembl_pec50[unb_idx].astype(np.float32)
    mean_sim_unb = mean_sim[unb_idx].astype(np.float32)

    # ---- Build nb1561-side 117-col 5-way K-tuned matrix on unb (built ONCE) ----
    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test, zero_fill=False)
    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test, zero_fill=False)
    X_mord_te = _load_mordred_test(n_test)
    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test, zero_fill=True)
    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test, zero_fill=False)

    X_ap_unb_top = X_ap_te[unb_idx].astype(np.float32)[:, top_ap_bit_idx]
    X_maccs_unb_top = X_maccs_te[unb_idx].astype(np.float32)[:, top_maccs_bit_idx]
    X_mord_unb_top = X_mord_te[unb_idx].astype(np.float32)[:, top_mord_col_idx]
    X_emb_unb_top = X_emb_te[unb_idx].astype(np.float32)[:, top_embed_col_idx]
    X_av_unb_top = X_av_te[unb_idx].astype(np.float32)[:, top_avalon_bit_idx]

    X_unb_1561 = np.concatenate(
        [
            X_ap_unb_top,
            X_maccs_unb_top,
            X_mord_unb_top,
            X_emb_unb_top,
            X_av_unb_top,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim_1561 = X_unb_1561.shape[1]
    expected_dim = (n_top_ap + n_top_maccs + n_top_mord
                    + n_top_embed + n_top_avalon + 2)
    if feat_dim_1561 != expected_dim:
        raise ValueError(f"feat_dim {feat_dim_1561} != expected {expected_dim}")
    print(f"\n[nb1561] 5-way K-tuned matrix: {X_unb_1561.shape}")

    # ---- Preload all 6 nb1612 family X_fam_unb tensors ----
    X_fam_unb_dict = {}
    for family in FAMILIES_NB1612:
        X_fam_te = _load_family_te_nb1612(family, n_test)
        X_fam_unb_dict[family] = X_fam_te[unb_idx].astype(np.float32)
    print("[nb1612] family unb tensors loaded: "
          + ", ".join(f"{k}={v.shape[1]}" for k, v in X_fam_unb_dict.items()))

    # ---- Outer-bag loop ----
    print("\n" + "=" * 78)
    print(f"OUTER-BAG  x [nb1561_o   = 5-inner CatBoost bag]")
    print(f"           x [nb1612_o   = 6-way naive/SLSQP best-variant]")
    print(f"           blend_o = {W_NB1561:.2f}*nb1561_o + {W_NB1612:.2f}*nb1612_o")
    print("=" * 78)
    n_outer = len(OUTER_SEEDS)

    outer_nb1561 = np.zeros((n_outer, n_unb), dtype=np.float64)
    outer_nb1612 = np.zeros((n_outer, n_unb), dtype=np.float64)
    outer_blend = np.zeros((n_outer, n_unb), dtype=np.float64)
    per_outer_records = []

    for oi, o in enumerate(OUTER_SEEDS):
        t_outer = time.time()
        inner_seeds = [int(o) * 1000 + int(s) for s in INNER_OFFSETS]
        print("\n" + "-" * 78)
        print(f"OUTER SEED {o}  ({oi + 1}/{n_outer})  "
              f"inner_seeds = {inner_seeds}")
        print("-" * 78)

        # ---- nb1561_o : 5-inner CatBoost bag ----
        t_1561 = time.time()
        inner_corrected_cat = np.zeros((len(inner_seeds), n_unb),
                                       dtype=np.float64)
        per_inner_cat_rae = []
        for si, s_inner in enumerate(inner_seeds):
            resid_oof_s = _cat_residual_cross_fit_one_seed(
                X_unb_1561, residual, seed=s_inner
            )
            pred_corr_s = anchor + resid_oof_s
            inner_corrected_cat[si] = pred_corr_s
            r_s = float(rae(y_unb, pred_corr_s))
            per_inner_cat_rae.append(r_s)
        nb1561_o = inner_corrected_cat.mean(axis=0)
        rae_nb1561_o = float(rae(y_unb, nb1561_o))
        print(f"   [nb1561_o]  per-inner_RAE = "
              f"{[round(r,4) for r in per_inner_cat_rae]}")
        print(f"   [nb1561_o]  mean-bag RAE = {rae_nb1561_o:.4f}  "
              f"(wall {time.time()-t_1561:.1f}s)")

        # ---- nb1612_o : rebuild 6 families + naive/SLSQP best-variant pick ----
        t_1612 = time.time()
        fam_results = []
        per_fam_rae = {}
        for family in FAMILIES_NB1612:
            r = _run_family_nb1612(
                family=family,
                X_fam_unb=X_fam_unb_dict[family],
                pred_chembl_unb=pred_chembl_unb,
                mean_sim_unb=mean_sim_unb,
                anchor=anchor,
                residual=residual,
                top_k=TOP_K_NB1612[family],
                inner_seeds=inner_seeds,
                outer_seed=int(o),
            )
            rae_fam = float(rae(y_unb, r["mean_bag_oof"]))
            r["rae_mean_bag"] = rae_fam
            fam_results.append(r)
            per_fam_rae[family] = rae_fam
            print(f"   [nb1612 fam] {family:<14s} (K={r['top_k']:>3})  "
                  f"mean_bag RAE = {rae_fam:.4f}")

        P = np.stack([r["mean_bag_oof"] for r in fam_results], axis=0)  # (6, n_unb)
        naive_oof = P.mean(axis=0)
        rae_naive = float(rae(y_unb, naive_oof))
        slsqp_oof, slsqp_folds = _slsqp_cross_fit(
            P.T.astype(np.float64), y_unb,
            n_splits=SLSQP_FOLDS, seed=SLSQP_SEED,
        )
        rae_slsqp = float(rae(y_unb, slsqp_oof))
        if rae_naive <= rae_slsqp:
            best_variant = "naive_1_6_mean"
            nb1612_o = naive_oof
            rae_nb1612_o = rae_naive
        else:
            best_variant = "slsqp_5fold"
            nb1612_o = slsqp_oof
            rae_nb1612_o = rae_slsqp
        W_slsqp = np.array([f["w"] for f in slsqp_folds])
        w_mean_slsqp = W_slsqp.mean(axis=0).tolist()
        print(f"   [nb1612_o]  naive_RAE = {rae_naive:.4f}  "
              f"slsqp_RAE = {rae_slsqp:.4f}  "
              f"best = {best_variant} ({rae_nb1612_o:.4f})  "
              f"(wall {time.time()-t_1612:.1f}s)")

        # ---- blend_o ----
        blend_o = W_NB1561 * nb1561_o + W_NB1612 * nb1612_o
        rae_blend_o = float(rae(y_unb, blend_o))
        delta_blend_vs_ref = rae_blend_o - NB1632_REF
        print(f"   [blend_o]   {W_NB1561:.2f}*nb1561_o + {W_NB1612:.2f}*nb1612_o  "
              f"-> RAE = {rae_blend_o:.4f}  "
              f"(d vs ref = {delta_blend_vs_ref:+.4f})")

        outer_nb1561[oi] = nb1561_o
        outer_nb1612[oi] = nb1612_o
        outer_blend[oi] = blend_o

        per_outer_records.append({
            "outer_seed": int(o),
            "inner_seeds": [int(x) for x in inner_seeds],
            "per_inner_cat_rae": per_inner_cat_rae,
            "rae_nb1561_o": rae_nb1561_o,
            "per_family_nb1612_rae": per_fam_rae,
            "rae_nb1612_naive": rae_naive,
            "rae_nb1612_slsqp": rae_slsqp,
            "nb1612_best_variant": best_variant,
            "rae_nb1612_o": rae_nb1612_o,
            "slsqp_w_mean_over_folds": w_mean_slsqp,
            "rae_blend_o": rae_blend_o,
            "delta_blend_vs_nb1632_ref": delta_blend_vs_ref,
            "wall_sec": round(time.time() - t_outer, 2),
        })
        print(f"   outer {o:3d}  total wall = {time.time() - t_outer:.1f}s")

    # ---- Per-outer summary on blend_o ----
    per_outer_rae = [rec["rae_blend_o"] for rec in per_outer_records]
    arr = np.array(per_outer_rae)
    per_outer_mean = float(arr.mean())
    per_outer_std = float(arr.std())
    per_outer_min = float(arr.min())
    per_outer_max = float(arr.max())
    per_outer_median = float(np.median(arr))

    # ---- BoB MEAN + MEDIAN over the 10 blend_o vectors ----
    bob_mean_oof = outer_blend.mean(axis=0)
    bob_median_oof = np.median(outer_blend, axis=0)
    rae_bob_mean = float(rae(y_unb, bob_mean_oof))
    rae_bob_median = float(rae(y_unb, bob_median_oof))

    # Sanity tracks: per-component BoB (for diagnostic only)
    rae_bob_mean_nb1561 = float(rae(y_unb, outer_nb1561.mean(axis=0)))
    rae_bob_mean_nb1612 = float(rae(y_unb, outer_nb1612.mean(axis=0)))

    # ---- Verdict (compare BoB MEAN to nb1632 5-outer ref) ----
    delta_per_outer = per_outer_mean - NB1632_REF
    delta_bob_mean = rae_bob_mean - NB1632_REF
    delta_bob_median = rae_bob_median - NB1632_REF
    reproduces = abs(delta_bob_mean) < REPRODUCE_MARGIN
    if reproduces:
        verdict = "NB1632_REPRODUCES_10SEED_TIGHT"
    elif rae_bob_mean < NB1632_REF - REPRODUCE_MARGIN:
        verdict = "NB1632_5SEED_PESSIMISTIC_10SEED_BETTER"
    else:
        verdict = "NB1632_5SEED_LUCKY_10SEED_WORSE"

    print("\n" + "=" * 78)
    print("OUTER-BAG SUMMARY (10 outer seeds)")
    print("=" * 78)
    print(f"   per-outer blend_o RAE list = "
          f"[{', '.join(f'{r:.4f}' for r in per_outer_rae)}]")
    print(f"   per-outer blend_o MEAN     = {per_outer_mean:.4f}")
    print(f"   per-outer blend_o STD      = {per_outer_std:.4f}")
    print(f"   per-outer blend_o MIN/MAX  = {per_outer_min:.4f} / {per_outer_max:.4f}")
    print(f"   per-outer blend_o MEDIAN   = {per_outer_median:.4f}")
    print(f"   BoB MEAN   pooled RAE      = {rae_bob_mean:.4f}  "
          f"(d vs nb1632_ref = {delta_bob_mean:+.4f})")
    print(f"   BoB MEDIAN pooled RAE      = {rae_bob_median:.4f}  "
          f"(d vs nb1632_ref = {delta_bob_median:+.4f})")
    print(f"   (diag) BoB MEAN nb1561 only = {rae_bob_mean_nb1561:.4f}")
    print(f"   (diag) BoB MEAN nb1612 only = {rae_bob_mean_nb1612:.4f}")
    print(f"\n   nb1632_ref (5-outer BoB)   = {NB1632_REF:.4f}")
    print(f"   d(BoB MEAN)                = {delta_bob_mean:+.4f}  "
          f"(margin {REPRODUCE_MARGIN})")
    print(f"   verdict                    = {verdict}")
    print("=" * 78)

    # ---- Save ----
    np.save(DATA_PROCESSED / f"{TAG}_bob_mean_oof.npy",
            bob_mean_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_bob_median_oof.npy",
            bob_median_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_bob_mean_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_bob_median_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "rae_anchor_chemprop_aux": rae_anchor,
        "n_unb": int(n_unb),
        "n_test": int(n_test),
        "n_chembl_pool": int(len(pool)),
        "outer_seeds": [int(o) for o in OUTER_SEEDS],
        "n_outer_seeds": int(n_outer),
        "inner_offsets": INNER_OFFSETS,
        "inner_seed_recipe": "[o*1000 + s for s in inner_offsets]",
        "resid_folds": RESID_FOLDS,
        "slsqp_folds": SLSQP_FOLDS,
        "slsqp_seed": SLSQP_SEED,
        "w_nb1561": W_NB1561,
        "w_nb1612": W_NB1612,
        "nb1561_feat_dim": int(feat_dim_1561),
        "nb1561_feat_breakdown": {
            "atompair": n_top_ap,
            "maccs": n_top_maccs,
            "mordred": n_top_mord,
            "chemprop_embed": n_top_embed,
            "avalon": n_top_avalon,
            "pred_chembl_pec50": 1,
            "mean_sim": 1,
            "total": int(feat_dim_1561),
        },
        "nb1612_top_k_config_fixed": TOP_K_NB1612,
        "nb1612_families_order": FAMILIES_NB1612,
        "per_outer_records": per_outer_records,
        "per_outer_blend_rae": per_outer_rae,
        "per_outer_mean": per_outer_mean,
        "per_outer_std": per_outer_std,
        "per_outer_min": per_outer_min,
        "per_outer_max": per_outer_max,
        "per_outer_median": per_outer_median,
        "rae_bob_mean": rae_bob_mean,
        "rae_bob_median": rae_bob_median,
        "rae_bob_mean_nb1561_only_diag": rae_bob_mean_nb1561,
        "rae_bob_mean_nb1612_only_diag": rae_bob_mean_nb1612,
        "nb1632_ref": NB1632_REF,
        "reproduce_margin": REPRODUCE_MARGIN,
        "delta_per_outer_mean_vs_nb1632_ref": delta_per_outer,
        "delta_bob_mean_vs_nb1632_ref": delta_bob_mean,
        "delta_bob_median_vs_nb1632_ref": delta_bob_median,
        "reproduces": bool(reproduces),
        "verdict": verdict,
        "pre_unblind_clean": True,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "n_unb", "n_test", "n_chembl_pool",
        "rae_anchor_chemprop_aux",
        "outer_seeds", "n_outer_seeds", "inner_offsets",
        "w_nb1561", "w_nb1612",
        "nb1632_ref",
        "per_outer_blend_rae",
        "per_outer_mean", "per_outer_std",
        "per_outer_min", "per_outer_max", "per_outer_median",
        "rae_bob_mean", "rae_bob_median",
        "rae_bob_mean_nb1561_only_diag",
        "rae_bob_mean_nb1612_only_diag",
        "delta_per_outer_mean_vs_nb1632_ref",
        "delta_bob_mean_vs_nb1632_ref",
        "delta_bob_median_vs_nb1632_ref",
        "reproduces",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
