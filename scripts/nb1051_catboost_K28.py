"""nb1051 -- CatBoost regressor on SHAP top-28 features (different boosting
paradigm than LGBM) as chemprop_aux residual corrector.

HYPOTHESIS:
    nb2103 K=28 LGBM (max_depth=4, num_leaves=15, n_estimators=300, lr=0.03)
    on the 117-col 5-way K-tuned feature matrix achieved cross-fit mean-bag
    RAE 0.4737 / median-bag RAE 0.4698.  CatBoost is a different gradient-
    boosting paradigm (symmetric/oblivious trees + ordered-target-statistic
    encoders) that can be orthogonal to LGBM's leaf-wise + histogram
    approach.  Two specialty modes are tested:
      (a) Plain CatBoost (boosting_type='Plain') matching LGBM's standard
          gradient-boosting behavior.
      (b) Ordered CatBoost (boosting_type='Ordered') -- CatBoost's anti-leak
          mode that uses ordered target statistics; designed to reduce
          overfit on small datasets like our n=253 cross-fit.
    Also tests CatBoost + LGBM convex blend at weights {0.0, 0.25, 0.5, 0.75,
    1.0} using the cached nb2103 K=28 cross-fit OOF.

PROTOCOL:
    1. Reuse SHAP top-28 indices from nb2063 SHAP importance ranking applied
       to the 117-col 5-way K-tuned feature matrix (same matrix construction
       as nb2103: AtomPair / MACCS / Mordred / ChempropEmbed / Avalon top-K
       slices + ChEMBL kNN pred + mean sim).
    2. Anchor = chemprop_aux te[unb_idx]; residual = y_unb - anchor on 253.
    3. For each boosting_type in {Plain, Ordered}:
         For each seed in {0, 1, 7, 42, 137}:
           CatBoostRegressor(iterations=500, learning_rate=0.03, depth=6,
                             l2_leaf_reg=3, random_state=seed,
                             boosting_type=boosting_type,
                             verbose=0, allow_writing_files=False).
           KFold(n=5, shuffle=True, random_state=seed) cross-fit per seed.
         mean-bag OOF, median-bag OOF across seeds; pooled RAE.
    4. Compare best (mean_bag, median_bag) vs nb2103 K=28 LGBM at
       decision_margin = 0.003.
    5. Convex blend test: alpha in {0.0, 0.25, 0.5, 0.75, 1.0}
       blended = alpha * catboost_mean_bag + (1 - alpha) * lgbm_K28_mean_bag.
       Same on median-bag.  Report RAE per alpha.
    6. If best (across plain/ordered/blend, mean_bag or median_bag) beats
       nb2103 K=28 mean_bag 0.4737 by >= 0.003 OR matches median_bag 0.4698,
       deploy: refit chosen recipe on ALL 253 unblind, predict residual on
       full 513, write submissions/nb1051_catboost_K28.csv.

ANCHORS:
    chemprop_aux in_RAE = 0.6216
    nb2103 K=28 LGBM mean_bag RAE = 0.4737 (median_bag = 0.4698)
    decision_margin = 0.003

Outputs:
    scripts/nb1051_catboost_K28.py
    data/processed/nb1051_summary.json
    data/processed/nb1051_mean_bag_oof_plain.npy        (253,) float32
    data/processed/nb1051_mean_bag_oof_ordered.npy      (253,) float32
    data/processed/nb1051_median_bag_oof_plain.npy      (253,) float32
    data/processed/nb1051_median_bag_oof_ordered.npy    (253,) float32
    submissions/nb1051_catboost_K28.csv                 (only if best beats)
    data/processed/te_nb1051_catboost_K28.npy           (only if deploy)
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
from sklearn.model_selection import KFold
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

try:
    from catboost import CatBoostRegressor
except Exception as e:
    raise RuntimeError(f"catboost import failed: {e}")

TAG = "nb1051"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]
K = 28

# CatBoost hyperparams (per task spec)
CAT_ITERATIONS = 500
CAT_LR = 0.03
CAT_DEPTH = 6
CAT_L2_LEAF_REG = 3

BLEND_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")

EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"
SUBMISSIONS_DIR = Path(__file__).resolve().parents[1] / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)

NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
NB1392_SUMMARY = DATA_PROCESSED / "nb1392_summary.json"
NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"
NB1523_SUMMARY = DATA_PROCESSED / "nb1523_summary.json"
NB1524_SUMMARY = DATA_PROCESSED / "nb1524_summary.json"
NB1541_SUMMARY = DATA_PROCESSED / "nb1541_summary.json"
NB2063_SUMMARY = DATA_PROCESSED / "nb2063_summary.json"
NB2063_SHAP_IMP = DATA_PROCESSED / "nb2063_shap_importance_full117.npy"
NB2103_SUMMARY = DATA_PROCESSED / "nb2103_summary.json"
NB2103_K28_OOF = DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

# References
CHEMPROP_AUX_REF = 0.6216
NB2103_K28_MEAN_BAG_REF = 0.4737
NB2103_K28_MEDIAN_BAG_REF = 0.4698
DECISION_MARGIN = 0.003


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
    """Same union as nb2063/nb2081/nb2091/nb2103."""
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
        print(f"   [src] CHEMBL3401_raw kept: {len(d)} rows")

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
        print(f"   [src] chembl_nr_extended PXR kept: {len(d)} rows")

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
        print(f"   [src] chembl_pxr_all_types kept: {len(d)} rows")

    if not frames:
        raise FileNotFoundError("No local ChEMBL PXR parquets found")

    pool = pd.concat(frames, ignore_index=True)
    print(f"   [pool] pre-standardize union: {len(pool)} rows")
    mols = pool["smiles"].apply(standardize)
    pool["inchikey"] = mols.apply(_safe_inchikey)
    pool["std_smiles"] = mols.apply(_safe_can_smiles)
    pool = pool[pool["inchikey"].notna() & pool["std_smiles"].notna()].copy()
    print(f"   [pool] after RDKit standardize: {len(pool)} rows")
    agg = (
        pool.groupby("inchikey", as_index=False)
        .agg(pec50=("pec50", "median"),
             std_smiles=("std_smiles", "first"),
             src_first=("src", "first"),
             n_meas=("pec50", "count"))
    )
    agg = agg.rename(columns={"src_first": "src"})
    print(f"   [pool] after InChIKey dedup (median agg): {len(agg)} unique")
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


def _catboost_params(seed: int, boosting_type: str) -> dict:
    return dict(
        iterations=CAT_ITERATIONS,
        learning_rate=CAT_LR,
        depth=CAT_DEPTH,
        l2_leaf_reg=CAT_L2_LEAF_REG,
        random_seed=seed,
        boosting_type=boosting_type,    # 'Plain' or 'Ordered'
        loss_function="RMSE",
        verbose=0,
        allow_writing_files=False,
        thread_count=2,
    )


def _residual_cross_fit_catboost(X: np.ndarray, residual: np.ndarray,
                                 seed: int, boosting_type: str) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = CatBoostRegressor(**_catboost_params(seed, boosting_type))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _load_mordred_test(n_test_expected: int) -> np.ndarray:
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(f"Mordred cache missing -- run nb1030 first ({mte_p})")
    X_te_m = np.load(mte_p).astype(np.float32)
    if X_te_m.shape[0] != n_test_expected:
        raise ValueError(f"Mordred test shape mismatch: {X_te_m.shape} vs {n_test_expected}")
    X_te_m = np.where(np.isfinite(X_te_m), X_te_m, np.nan).astype(np.float32)
    col_med = np.nanmedian(X_te_m, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X_te_m)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X_te_m[idx_r, idx_c] = col_med[idx_c]
    return X_te_m


def _load_npy_test(path: Path, n_test_expected: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"missing cache: {path}")
    X = np.load(path)
    if X.shape[0] != n_test_expected:
        raise ValueError(f"shape mismatch {path}: {X.shape}")
    X = X.astype(np.float32)
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X


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


def _verdict(rae_mean_bag: float, rae_median_bag: float) -> tuple[str, bool]:
    """Returns (verdict_str, deploy_flag)."""
    beats_mean = rae_mean_bag < NB2103_K28_MEAN_BAG_REF - DECISION_MARGIN
    beats_median = rae_median_bag < NB2103_K28_MEDIAN_BAG_REF - DECISION_MARGIN
    flat_mean = abs(rae_mean_bag - NB2103_K28_MEAN_BAG_REF) < DECISION_MARGIN
    if beats_mean and beats_median:
        return "BEATS_NB2103_K28_BOTH", True
    if beats_mean:
        return "BEATS_NB2103_K28_MEAN_BAG", True
    if beats_median:
        return "BEATS_NB2103_K28_MEDIAN_BAG", True
    if flat_mean:
        return "FLAT_VS_NB2103_K28", False
    return "HURTS_VS_NB2103_K28", False


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- CatBoost (Plain + Ordered) on SHAP top-{K} feats "
          f"(different boosting paradigm vs nb2103 K=28 LGBM)")
    print(f"          anchor={ANCHOR}  seeds={RESID_SEEDS}  folds={RESID_FOLDS}")
    print(f"          ref: nb2103 K=28 mean_bag {NB2103_K28_MEAN_BAG_REF:.4f}  "
          f"median_bag {NB2103_K28_MEDIAN_BAG_REF:.4f}  margin {DECISION_MARGIN}")
    print(f"          CatBoost: iters={CAT_ITERATIONS} lr={CAT_LR} "
          f"depth={CAT_DEPTH} l2={CAT_L2_LEAF_REG}")
    print("=" * 78)

    # ---- nb2103 reference (the head-to-head opponent at K=28) ----
    if not NB2103_SUMMARY.exists():
        raise FileNotFoundError(f"missing {NB2103_SUMMARY} -- run nb2103 first")
    if not NB2103_K28_OOF.exists():
        raise FileNotFoundError(f"missing {NB2103_K28_OOF} -- run nb2103 first")
    with open(NB2103_SUMMARY) as f:
        nb2103_sum = json.load(f)
    nb2103_k28_mean_bag = NB2103_K28_MEAN_BAG_REF
    nb2103_k28_median_bag = NB2103_K28_MEDIAN_BAG_REF
    for r in nb2103_sum.get("per_K_records", []):
        if int(r.get("K", -1)) == K:
            nb2103_k28_mean_bag = float(r["rae_mean_bag"])
            nb2103_k28_median_bag = float(r["rae_median_bag"])
            break
    print(f"[ref] nb2103.K=28 mean_bag   = {nb2103_k28_mean_bag:.4f}")
    print(f"[ref] nb2103.K=28 median_bag = {nb2103_k28_median_bag:.4f}")
    lgbm_K28_oof = np.load(NB2103_K28_OOF).astype(np.float64)
    print(f"[ref] nb2103 K=28 mean-bag OOF shape = {lgbm_K28_oof.shape}")

    # ---- SHAP importance ----
    if not NB2063_SHAP_IMP.exists():
        raise FileNotFoundError(f"missing {NB2063_SHAP_IMP} -- run nb2063 first")
    shap_imp_full117 = np.load(NB2063_SHAP_IMP).astype(np.float32)
    print(f"[ref] nb2063 SHAP importance shape = {shap_imp_full117.shape}")
    full_rank_order = np.argsort(-shap_imp_full117).astype(np.int32)

    # ---- Load anchor + truth ----
    te = load_test()
    n_test = len(te)
    if "smiles" in te.columns:
        test_smiles = te["smiles"].astype(str).tolist()
    else:
        test_smiles = te["SMILES"].astype(str).tolist()
    if "Molecule Name" in te.columns:
        mol_names = te["Molecule Name"].astype(str).tolist()
    elif "molecule_name" in te.columns:
        mol_names = te["molecule_name"].astype(str).tolist()
    elif "name" in te.columns:
        mol_names = te["name"].astype(str).tolist()
    else:
        raise KeyError("no name column on test set")
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"chemprop_aux te file missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(f"chemprop_aux te shape mismatch: {te_anchor_513.shape}")
    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual_unb = y_unb - anchor_unb
    print(f"[resid] mean={residual_unb.mean():+.4f}  std={residual_unb.std():.4f}")

    # ---- Validate cached LGBM K=28 OOF gives ref RAE ----
    rae_lgbm_K28 = float(rae(y_unb, lgbm_K28_oof))
    print(f"[check] cached nb2103 K=28 OOF RAE = {rae_lgbm_K28:.4f}  "
          f"(ref {nb2103_k28_mean_bag:.4f})")

    # ---- Build the 117-col 5-way K-tuned matrix (same as nb2103) ----
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

    n_top_maccs = int(len(top_maccs_bit_idx))
    n_top_mord = int(len(top_mord_col_idx))
    n_top_ap = int(len(top_ap_bit_idx))
    n_top_embed = int(len(top_embed_col_idx))
    n_top_avalon = int(len(top_avalon_bit_idx))
    print(f"[reuse] AP={n_top_ap}  MACCS={n_top_maccs}  Mord={n_top_mord}  "
          f"Embed={n_top_embed}  Av={n_top_avalon}")

    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test)
    X_ap_te_top = X_ap_te[:, top_ap_bit_idx].astype(np.float32)
    X_ap_unb_top = X_ap_te_top[unb_idx]
    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)
    X_maccs_te_top = X_maccs_te[:, top_maccs_bit_idx].astype(np.float32)
    X_maccs_unb_top = X_maccs_te_top[unb_idx]
    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    X_mord_te_top = X_mord_te[:, top_mord_col_idx].astype(np.float32)
    X_mord_unb_top = X_mord_te_top[unb_idx]
    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)
    X_emb_te_top = X_emb_te[:, top_embed_col_idx].astype(np.float32)
    X_emb_unb_top = X_emb_te_top[unb_idx]
    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)
    X_av_te_top = X_av_te[:, top_avalon_bit_idx].astype(np.float32)
    X_av_unb_top = X_av_te_top[unb_idx]

    # ChEMBL kNN
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL (union)")
    print("-" * 78)
    pool = _load_chembl_pool()
    test_mols = [standardize(s) for s in test_smiles]
    test_inchikeys = set()
    for m in test_mols:
        ik = _safe_inchikey(m)
        if ik is not None:
            test_inchikeys.add(ik)
    n_before = len(pool)
    pool = pool[~pool["inchikey"].isin(test_inchikeys)].reset_index(drop=True)
    print(f"   pool: {n_before} -> {len(pool)} after test-overlap drop")
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
    pred_chembl_te = pred_chembl_pec50.astype(np.float32)
    mean_sim_te = mean_sim.astype(np.float32)
    pred_chembl_unb = pred_chembl_te[unb_idx]
    mean_sim_unb = mean_sim_te[unb_idx]

    # ---- Build 117-col matrices (unb + 513) ----
    X_te_full = np.concatenate([
        X_ap_te_top, X_maccs_te_top, X_mord_te_top, X_emb_te_top, X_av_te_top,
        pred_chembl_te.reshape(-1, 1), mean_sim_te.reshape(-1, 1),
    ], axis=1).astype(np.float32)
    X_unb_full = np.concatenate([
        X_ap_unb_top, X_maccs_unb_top, X_mord_unb_top, X_emb_unb_top,
        X_av_unb_top,
        pred_chembl_unb.reshape(-1, 1), mean_sim_unb.reshape(-1, 1),
    ], axis=1).astype(np.float32)
    feat_dim_full = X_unb_full.shape[1]
    if feat_dim_full != shap_imp_full117.shape[0]:
        raise ValueError(
            f"feat_dim {feat_dim_full} != SHAP imp len {shap_imp_full117.shape[0]}"
        )
    print(f"   COMBINED full matrix: X_unb_full={X_unb_full.shape}  "
          f"X_te_full={X_te_full.shape}")

    # ---- Slice SHAP top-28 ----
    topK_idx = full_rank_order[:K].astype(np.int32)
    X_unb_K = X_unb_full[:, topK_idx].astype(np.float32)
    X_te_K = X_te_full[:, topK_idx].astype(np.float32)
    print(f"\n   SHAP top-{K} matrices: X_unb_K={X_unb_K.shape}  "
          f"X_te_K={X_te_K.shape}")

    # ---- CatBoost cross-fit: Plain + Ordered ----
    print("\n" + "=" * 78)
    print(f"CATBOOST CROSS-FIT  (5 seeds x 5 folds x 2 modes)")
    print("=" * 78)
    mode_results: dict[str, dict] = {}
    per_seed_oof_by_mode: dict[str, np.ndarray] = {}
    for boosting_type in ("Plain", "Ordered"):
        print(f"\n--- mode = {boosting_type} ---")
        per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
        per_seed_rae: list[float] = []
        per_seed_records = []
        for i, s in enumerate(RESID_SEEDS):
            ts = time.time()
            resid_oof_s = _residual_cross_fit_catboost(
                X_unb_K, residual_unb, s, boosting_type
            )
            pred_corr_s = anchor_unb + resid_oof_s
            per_seed_corrected[i] = pred_corr_s
            rae_s = float(rae(y_unb, pred_corr_s))
            per_seed_rae.append(rae_s)
            wall = time.time() - ts
            per_seed_records.append({
                "seed": int(s),
                "rae_corrected": rae_s,
                "delta_vs_chemprop_aux": rae_s - rae_anchor,
                "resid_oof_std": float(resid_oof_s.std()),
                "resid_oof_mean": float(resid_oof_s.mean()),
                "wall_sec": round(wall, 2),
            })
            print(f"   {boosting_type:7s} seed={s:3d}: rae={rae_s:.4f}  "
                  f"(d_vs_anchor={rae_s - rae_anchor:+.4f})  wall={wall:.1f}s")

        mean_bag_oof = per_seed_corrected.mean(axis=0)
        median_bag_oof = np.median(per_seed_corrected, axis=0)
        rae_mean_bag = float(rae(y_unb, mean_bag_oof))
        rae_median_bag = float(rae(y_unb, median_bag_oof))
        per_seed_rae_arr = np.array(per_seed_rae)

        delta_mean_bag_vs_nb2103 = rae_mean_bag - nb2103_k28_mean_bag
        delta_median_bag_vs_nb2103 = rae_median_bag - nb2103_k28_median_bag
        verdict_mode, _ = _verdict(rae_mean_bag, rae_median_bag)
        print(f"   {boosting_type:7s} POOLED  mean_bag={rae_mean_bag:.4f}  "
              f"median_bag={rae_median_bag:.4f}  "
              f"d_mean_vs_LGBM={delta_mean_bag_vs_nb2103:+.4f}  "
              f"d_median_vs_LGBM={delta_median_bag_vs_nb2103:+.4f}  "
              f"verdict={verdict_mode}")

        # Save OOF artifacts per mode
        mode_lower = boosting_type.lower()
        np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof_{mode_lower}.npy",
                mean_bag_oof.astype(np.float32))
        np.save(DATA_PROCESSED / f"{TAG}_median_bag_oof_{mode_lower}.npy",
                median_bag_oof.astype(np.float32))

        mode_results[boosting_type] = {
            "boosting_type": boosting_type,
            "per_seed_rae": per_seed_rae,
            "per_seed_records": per_seed_records,
            "rae_per_seed_mean": float(per_seed_rae_arr.mean()),
            "rae_per_seed_std": float(per_seed_rae_arr.std()),
            "rae_per_seed_min": float(per_seed_rae_arr.min()),
            "rae_per_seed_max": float(per_seed_rae_arr.max()),
            "rae_mean_bag": rae_mean_bag,
            "rae_median_bag": rae_median_bag,
            "delta_mean_bag_vs_nb2103_K28": delta_mean_bag_vs_nb2103,
            "delta_median_bag_vs_nb2103_K28": delta_median_bag_vs_nb2103,
            "verdict": verdict_mode,
            "mean_bag_oof_path": str(
                DATA_PROCESSED / f"{TAG}_mean_bag_oof_{mode_lower}.npy"
            ),
            "median_bag_oof_path": str(
                DATA_PROCESSED / f"{TAG}_median_bag_oof_{mode_lower}.npy"
            ),
        }
        per_seed_oof_by_mode[boosting_type] = mean_bag_oof.copy()

    # ---- CatBoost + LGBM convex blend ----
    print("\n" + "=" * 78)
    print(f"CONVEX BLEND: alpha * CatBoost + (1 - alpha) * nb2103 K=28 LGBM")
    print(f"   alphas = {BLEND_ALPHAS}")
    print("=" * 78)
    blend_results: list[dict] = []
    for mode_name, cat_oof in per_seed_oof_by_mode.items():
        for alpha in BLEND_ALPHAS:
            blend_oof = alpha * cat_oof + (1.0 - alpha) * lgbm_K28_oof
            rae_blend = float(rae(y_unb, blend_oof))
            beats_mean = rae_blend < nb2103_k28_mean_bag - DECISION_MARGIN
            beats_median = rae_blend < nb2103_k28_median_bag - DECISION_MARGIN
            tag = ("BEATS_BOTH" if (beats_mean and beats_median) else
                   ("BEATS_MEAN" if beats_mean else
                    ("BEATS_MEDIAN" if beats_median else "NO")))
            print(f"   blend  mode={mode_name:7s}  alpha={alpha:.2f}  "
                  f"rae={rae_blend:.4f}  "
                  f"d_mean={rae_blend - nb2103_k28_mean_bag:+.4f}  "
                  f"d_median={rae_blend - nb2103_k28_median_bag:+.4f}  {tag}")
            blend_results.append({
                "mode": mode_name,
                "alpha": float(alpha),
                "rae": rae_blend,
                "delta_vs_nb2103_mean": rae_blend - nb2103_k28_mean_bag,
                "delta_vs_nb2103_median": rae_blend - nb2103_k28_median_bag,
                "beats_mean": bool(beats_mean),
                "beats_median": bool(beats_median),
            })

    # ---- Pick best across (catboost mean_bag, catboost median_bag, blend) ----
    print("\n" + "=" * 78)
    print("OVERALL SELECTION")
    print("=" * 78)
    candidates: list[dict] = []
    for mode_name, mr in mode_results.items():
        candidates.append({
            "kind": f"catboost_{mode_name.lower()}_mean_bag",
            "rae": mr["rae_mean_bag"],
            "mode": mode_name,
            "alpha": 1.0,
            "agg": "mean_bag",
        })
        candidates.append({
            "kind": f"catboost_{mode_name.lower()}_median_bag",
            "rae": mr["rae_median_bag"],
            "mode": mode_name,
            "alpha": 1.0,
            "agg": "median_bag",
        })
    for br in blend_results:
        candidates.append({
            "kind": f"blend_{br['mode'].lower()}_alpha{br['alpha']:.2f}_mean_bag",
            "rae": br["rae"],
            "mode": br["mode"],
            "alpha": br["alpha"],
            "agg": "mean_bag",
        })

    best = min(candidates, key=lambda c: c["rae"])
    best_rae = best["rae"]
    beats_nb2103_mean = best_rae < nb2103_k28_mean_bag - DECISION_MARGIN
    beats_nb2103_median = best_rae < nb2103_k28_median_bag - DECISION_MARGIN
    print(f"\n   best candidate    = {best['kind']}  RAE={best_rae:.4f}")
    print(f"   vs nb2103 K=28 mean_bag  ({nb2103_k28_mean_bag:.4f}): "
          f"{best_rae - nb2103_k28_mean_bag:+.4f}  "
          f"beats(margin={DECISION_MARGIN}) = {beats_nb2103_mean}")
    print(f"   vs nb2103 K=28 median_bag ({nb2103_k28_median_bag:.4f}): "
          f"{best_rae - nb2103_k28_median_bag:+.4f}  "
          f"beats(margin={DECISION_MARGIN}) = {beats_nb2103_median}")

    do_deploy = beats_nb2103_mean or beats_nb2103_median
    if best_rae >= nb2103_k28_mean_bag - DECISION_MARGIN \
            and best_rae >= nb2103_k28_median_bag - DECISION_MARGIN:
        if abs(best_rae - nb2103_k28_mean_bag) < DECISION_MARGIN \
                or abs(best_rae - nb2103_k28_median_bag) < DECISION_MARGIN:
            global_verdict = "FLAT_VS_NB2103_K28"
        else:
            global_verdict = "HURTS_VS_NB2103_K28"
        do_deploy = False
    else:
        global_verdict = "BEATS_NB2103_K28_DEPLOY"

    print(f"\n   global verdict   = {global_verdict}")
    print(f"   do_deploy        = {do_deploy}")

    # ---- DEPLOY (only if best beats nb2103 K=28) ----
    submission_csv = None
    te_artifact = None
    te_in_RAE_unb = None
    deploy_info = None
    if do_deploy:
        print("\n" + "=" * 78)
        print(f"DEPLOY: refit best recipe ({best['kind']}) on ALL 253 unblind, "
              f"predict 513")
        print("=" * 78)
        if best["kind"].startswith("catboost_"):
            # Pure CatBoost: refit 5-seed bag on all 253, predict 513,
            # aggregate using best['agg']
            mode_name = best["mode"]
            per_seed_te = np.zeros((len(RESID_SEEDS), n_test), dtype=np.float64)
            for i, s in enumerate(RESID_SEEDS):
                ts = time.time()
                mdl = CatBoostRegressor(**_catboost_params(s, mode_name))
                mdl.fit(X_unb_K, residual_unb)
                resid_513 = mdl.predict(X_te_K)
                per_seed_te[i] = resid_513
                print(f"   refit {mode_name:7s} seed={s:3d}  "
                      f"resid_513_mean={resid_513.mean():+.4f}  "
                      f"resid_513_std={resid_513.std():.4f}  "
                      f"wall={time.time() - ts:.1f}s")
            if best["agg"] == "median_bag":
                resid_513_agg = np.median(per_seed_te, axis=0)
            else:
                resid_513_agg = per_seed_te.mean(axis=0)
            te_final_513 = te_anchor_513 + resid_513_agg
            deploy_info = {
                "kind": "pure_catboost",
                "mode": mode_name,
                "agg": best["agg"],
                "n_seeds": len(RESID_SEEDS),
            }
        else:
            # Blend: need both CatBoost and LGBM-K28 te_513 predictions
            mode_name = best["mode"]
            alpha = best["alpha"]
            # CatBoost refit on 253
            per_seed_cat_te = np.zeros((len(RESID_SEEDS), n_test), dtype=np.float64)
            for i, s in enumerate(RESID_SEEDS):
                ts = time.time()
                mdl = CatBoostRegressor(**_catboost_params(s, mode_name))
                mdl.fit(X_unb_K, residual_unb)
                resid_513 = mdl.predict(X_te_K)
                per_seed_cat_te[i] = resid_513
                print(f"   refit CAT {mode_name:7s} seed={s:3d}  "
                      f"wall={time.time() - ts:.1f}s")
            cat_resid_513 = per_seed_cat_te.mean(axis=0)
            # LGBM K=28 refit on 253 (reuse hyperparams from nb2103)
            import lightgbm as lgb
            per_seed_lgbm_te = np.zeros((len(RESID_SEEDS), n_test),
                                         dtype=np.float64)
            lgbm_p = dict(objective="regression", max_depth=4, num_leaves=15,
                          n_estimators=300, learning_rate=0.03,
                          min_child_samples=5, reg_lambda=2.0,
                          n_jobs=2, verbosity=-1)
            for i, s in enumerate(RESID_SEEDS):
                ts = time.time()
                lp = dict(lgbm_p)
                lp["random_state"] = s
                mdl = lgb.LGBMRegressor(**lp)
                mdl.fit(X_unb_K, residual_unb)
                resid_513 = mdl.predict(X_te_K)
                per_seed_lgbm_te[i] = resid_513
                print(f"   refit LGBM seed={s:3d}  "
                      f"wall={time.time() - ts:.1f}s")
            lgbm_resid_513 = per_seed_lgbm_te.mean(axis=0)
            resid_513_agg = alpha * cat_resid_513 + (1.0 - alpha) * lgbm_resid_513
            te_final_513 = te_anchor_513 + resid_513_agg
            deploy_info = {
                "kind": "blend",
                "mode": mode_name,
                "alpha": alpha,
                "agg": "mean_bag",
                "n_seeds": len(RESID_SEEDS),
            }

        # in-sample sanity
        te_in_RAE_unb = float(rae(y_unb, te_final_513[unb_idx]))
        print(f"\n   in-sample RAE on unb_idx = {te_in_RAE_unb:.4f}  "
              f"(cross-fit ref {best_rae:.4f})")
        print(f"   te stats: mean={te_final_513.mean():.4f}  "
              f"std={te_final_513.std():.4f}  "
              f"min={te_final_513.min():.4f}  max={te_final_513.max():.4f}")

        # Save CSV
        df_sub = pd.DataFrame({
            "SMILES": test_smiles,
            "Molecule Name": mol_names,
            "pEC50": te_final_513.astype(np.float32),
        })
        if len(df_sub) != 513:
            raise ValueError(f"submission rows {len(df_sub)} != 513")
        sub_path = SUBMISSIONS_DIR / f"{TAG}_catboost_K28.csv"
        df_sub.to_csv(sub_path, index=False)
        print(f"\n[save] submission CSV: {sub_path}  ({len(df_sub)} rows)")
        submission_csv = str(sub_path)

        te_path = DATA_PROCESSED / f"te_{TAG}_catboost_K28.npy"
        np.save(te_path, te_final_513.astype(np.float32))
        print(f"[save] te artifact:  {te_path}")
        te_artifact = str(te_path)

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "method": ("catboost_K28_plain_and_ordered_on_117col_5way_K_tuned"
                   "_with_LGBM_blend"),
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "data_source": ("nb2063 SHAP top-28 indices of the 117-col 5-way "
                        "K-tuned matrix (AtomPair / MACCS / Mordred / "
                        "ChempropEmbed / Avalon + ChEMBL kNN)"),
        "K": K,
        "feat_dim": int(K),
        "model_family": "CatBoost",
        "cat_iterations": CAT_ITERATIONS,
        "cat_learning_rate": CAT_LR,
        "cat_depth": CAT_DEPTH,
        "cat_l2_leaf_reg": CAT_L2_LEAF_REG,
        "modes_tested": ["Plain", "Ordered"],
        "n_chembl_pool": int(len(pool)),
        "n_test": n_test,
        "n_unb": n_unb,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean_unb": float(residual_unb.mean()),
        "residual_std_unb": float(residual_unb.std()),
        "nb2103_K28_mean_bag_ref": nb2103_k28_mean_bag,
        "nb2103_K28_median_bag_ref": nb2103_k28_median_bag,
        "nb2103_K28_oof_path": str(NB2103_K28_OOF),
        "nb2103_K28_oof_self_RAE_check": rae_lgbm_K28,
        "decision_margin": DECISION_MARGIN,
        "mode_results": mode_results,
        "blend_alphas": BLEND_ALPHAS,
        "blend_results": blend_results,
        "candidates": candidates,
        "best_candidate": best,
        "best_rae": best_rae,
        "best_delta_vs_nb2103_K28_mean_bag": (
            best_rae - nb2103_k28_mean_bag
        ),
        "best_delta_vs_nb2103_K28_median_bag": (
            best_rae - nb2103_k28_median_bag
        ),
        "beats_nb2103_K28_mean_bag": bool(beats_nb2103_mean),
        "beats_nb2103_K28_median_bag": bool(beats_nb2103_median),
        "global_verdict": global_verdict,
        "do_deploy": bool(do_deploy),
        "deploy_info": deploy_info,
        "submission_csv": submission_csv,
        "te_artifact": te_artifact,
        "te_in_RAE_unb": te_in_RAE_unb,
        "pre_unblind_clean": True,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] summary: {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "K", "feat_dim", "n_chembl_pool", "n_unb",
        "rae_anchor_chemprop_aux",
        "nb2103_K28_mean_bag_ref", "nb2103_K28_median_bag_ref",
        "best_rae", "best_delta_vs_nb2103_K28_mean_bag",
        "best_delta_vs_nb2103_K28_median_bag",
        "beats_nb2103_K28_mean_bag", "beats_nb2103_K28_median_bag",
        "global_verdict", "do_deploy",
        "submission_csv", "te_in_RAE_unb",
        "pre_unblind_clean",
    ):
        print(f"  {k}: {res.get(k)}")
    print("\n==== MODE TABLE ====")
    for mode_name, mr in res["mode_results"].items():
        print(f"  {mode_name:7s}  mean_bag={mr['rae_mean_bag']:.4f}  "
              f"median_bag={mr['rae_median_bag']:.4f}  "
              f"per_seed_mean={mr['rae_per_seed_mean']:.4f}  "
              f"std={mr['rae_per_seed_std']:.4f}  "
              f"verdict={mr['verdict']}")
    print("\n==== BLEND TABLE ====")
    for br in res["blend_results"]:
        flag = ("BEATS_BOTH" if (br["beats_mean"] and br["beats_median"]) else
                ("BEATS_MEAN" if br["beats_mean"] else
                 ("BEATS_MEDIAN" if br["beats_median"] else "")))
        print(f"  mode={br['mode']:7s}  alpha={br['alpha']:.2f}  "
              f"rae={br['rae']:.4f}  "
              f"d_mean={br['delta_vs_nb2103_mean']:+.4f}  "
              f"d_median={br['delta_vs_nb2103_median']:+.4f}  {flag}")
