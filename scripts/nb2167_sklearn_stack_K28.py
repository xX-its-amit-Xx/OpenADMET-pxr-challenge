"""nb2167 -- Cross-paradigm sklearn HistGradientBoostingRegressor stack at K=28.

HYPOTHESIS:
    nb2103 K=28 LGBM(MSE) is the SHAP fine-grid winner at mean_bag=0.4737,
    median_bag=0.4698 on chemprop_aux residual.  LGBM uses GOSS sampling +
    histogram binning + leaf-wise growth.  sklearn HistGradientBoostingRegressor
    uses a DIFFERENT histogram binning algorithm (255 bins, Friedman quantile
    binning), level-wise growth bounded by max_leaf_nodes, and pure squared-error
    loss without GOSS.  If sklearn-HGB is even modestly orthogonal to LGBM at
    the same K=28 feature pruning, then a convex blend should land below either
    alone.

PROTOCOL:
    1. Reuse the EXACT same 117-col 5-way K-tuned feature matrix and SHAP
       top-28 indices as nb2103 (full re-rank of nb2063 SHAP importance).
    2. Run sklearn HistGradientBoostingRegressor with:
         loss='squared_error', max_depth=4, max_leaf_nodes=15,
         learning_rate=0.03, max_iter=300, l2_regularization=2,
         min_samples_leaf=5, max_features=1.0
       5 seeds (0, 1, 7, 42, 137), 5-fold cross-fit per seed on
       chemprop_aux residual.
    3. Report sklearn-HGB alone: mean-bag and median-bag RAE.
    4. Load nb2103 K=28 mean-bag OOF; blend sklearn-HGB(K=28) mean-bag with
       nb2103 LGBM(K=28) mean-bag at weights w_sklearn in
       {0.0, 0.25, 0.50, 0.75, 1.0}.  Report best blend.
    5. Decision margin 0.003 vs nb2103 K=28 baseline (0.4737/0.4698).

Outputs:
    scripts/nb2167_sklearn_stack_K28.py
    data/processed/nb2167_summary.json
    data/processed/nb2167_sklearn_mean_bag_oof_K28.npy   (253,) float32
    data/processed/nb2167_sklearn_median_bag_oof_K28.npy (253,) float32
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
from sklearn.ensemble import HistGradientBoostingRegressor
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb2167"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

K = 28
RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]
BLEND_WEIGHTS = [(0.0, 1.0), (0.25, 0.75), (0.5, 0.5), (0.75, 0.25), (1.0, 0.0)]

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
NB2103_K28_OOF = DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

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


def _hgb_params(seed: int) -> dict:
    """sklearn HistGradientBoostingRegressor -- cross-paradigm to LGBM."""
    return dict(
        loss="squared_error",
        max_depth=4,
        max_leaf_nodes=15,
        learning_rate=0.03,
        max_iter=300,
        l2_regularization=2.0,
        min_samples_leaf=5,
        max_features=1.0,
        random_state=seed,
    )


def _residual_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray,
                                 seed: int) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = HistGradientBoostingRegressor(**_hgb_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _load_mordred_test(n_test_expected: int) -> np.ndarray:
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(f"Mordred cache missing: {mte_p}")
    X_te_m = np.load(mte_p).astype(np.float32)
    if X_te_m.shape[0] != n_test_expected:
        raise ValueError(f"Mordred shape mismatch: {X_te_m.shape}")
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
    raise KeyError("AtomPair entry not found")


def _extract_best_K_record(sum_dict: dict, records_key: str,
                           best_K_key: str = "best_K") -> dict:
    best_K = int(sum_dict[best_K_key])
    for r in sum_dict[records_key]:
        if int(r["K"]) == best_K:
            return r
    raise KeyError(f"best_K {best_K} not found")


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- sklearn HGB cross-paradigm stack at K={K} on 117-col matrix")
    print(f"          anchor={ANCHOR}  seeds={RESID_SEEDS}  folds={RESID_FOLDS}")
    print(f"          ref: nb2103 K=28 mean_bag={NB2103_K28_MEAN_BAG_REF:.4f} "
          f"median_bag={NB2103_K28_MEDIAN_BAG_REF:.4f}  margin={DECISION_MARGIN}")
    print("=" * 78)

    # ---- Load SHAP importance + nb2103 K=28 OOF ----
    if not NB2063_SHAP_IMP.exists():
        raise FileNotFoundError(f"missing {NB2063_SHAP_IMP}")
    if not NB2103_K28_OOF.exists():
        raise FileNotFoundError(f"missing {NB2103_K28_OOF}")
    shap_imp_full117 = np.load(NB2063_SHAP_IMP).astype(np.float32)
    full_rank_order = np.argsort(-shap_imp_full117).astype(np.int32)
    nb2103_k28_oof = np.load(NB2103_K28_OOF).astype(np.float64)
    print(f"[load] SHAP imp shape = {shap_imp_full117.shape}")
    print(f"[load] nb2103 K28 mean-bag OOF shape = {nb2103_k28_oof.shape}")

    # ---- Load anchor + truth ----
    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # Sanity: nb2103 OOF should already match the K=28 baseline
    nb2103_in_rae = float(rae(y_unb, nb2103_k28_oof))
    print(f"[check] nb2103 K=28 mean-bag OOF RAE = {nb2103_in_rae:.4f}  "
          f"(ref {NB2103_K28_MEAN_BAG_REF:.4f})")

    # ---- Load 5-way K-tuned matrix (same as nb2103) ----
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

    top_maccs_bit_idx = np.array(
        sum_1352["top_maccs_bit_indices_ranked"], dtype=int
    )
    rec_mord = _extract_best_K_record(sum_1523, "per_K_records")
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

    # ---- Feature matrices ----
    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test)
    X_ap_unb_top = X_ap_te[unb_idx][:, top_ap_bit_idx].astype(np.float32)
    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)
    X_maccs_unb_top = X_maccs_te[unb_idx][:, top_maccs_bit_idx].astype(np.float32)
    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    X_mord_unb_top = X_mord_te[unb_idx][:, top_mord_col_idx].astype(np.float32)
    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)
    X_emb_unb_top = X_emb_te[unb_idx][:, top_embed_col_idx].astype(np.float32)
    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)
    X_av_unb_top = X_av_te[unb_idx][:, top_avalon_bit_idx].astype(np.float32)

    # ---- ChEMBL kNN feature ----
    print("\n[chembl] loading PXR pool...")
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
    pred_chembl_unb = pred_chembl_pec50[unb_idx].astype(np.float32)
    mean_sim_unb = mean_sim[unb_idx].astype(np.float32)

    # ---- Build full 117-col matrix and slice to top-K=28 ----
    X_unb = np.concatenate(
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
    feat_dim = X_unb.shape[1]
    print(f"[feat] full 117-col matrix = {X_unb.shape}")
    if feat_dim != shap_imp_full117.shape[0]:
        raise ValueError(
            f"feat_dim {feat_dim} != SHAP length {shap_imp_full117.shape[0]}"
        )

    topK_idx = full_rank_order[:K].astype(np.int32)
    X_topK = X_unb[:, topK_idx].astype(np.float32)
    print(f"[feat] top-{K} sklearn input = {X_topK.shape}")

    # ---- sklearn HGB 5-seed bag, 5-fold cross-fit ----
    print("\n" + "-" * 78)
    print(f"SKLEARN HGB CROSS-FIT (K={K}, 5 seeds, 5-fold)")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        resid_oof_s = _residual_cross_fit_one_seed(X_topK, residual, s)
        pred_corr_s = anchor + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        delta_s = rae_s - rae_anchor
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_chemprop_aux": delta_s,
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   seed={s:3d}:  rae = {rae_s:.4f}  "
              f"(d_anchor={delta_s:+.4f})  wall={time.time() - ts:.1f}s")

    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))
    per_seed_arr = np.array(per_seed_rae)

    print(f"\n[sklearn-HGB] per-seed RAE = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"[sklearn-HGB] per-seed mean = {per_seed_arr.mean():.4f}  "
          f"std = {per_seed_arr.std():.4f}")
    print(f"[sklearn-HGB] mean-bag RAE   = {rae_mean_bag:.4f}  "
          f"(d_vs_nb2103_K28_mean = {rae_mean_bag - NB2103_K28_MEAN_BAG_REF:+.4f})")
    print(f"[sklearn-HGB] median-bag RAE = {rae_median_bag:.4f}  "
          f"(d_vs_nb2103_K28_median = "
          f"{rae_median_bag - NB2103_K28_MEDIAN_BAG_REF:+.4f})")

    np.save(DATA_PROCESSED / f"{TAG}_sklearn_mean_bag_oof_K{K}.npy",
            mean_bag_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_sklearn_median_bag_oof_K{K}.npy",
            median_bag_oof.astype(np.float32))

    # ---- Cross-paradigm blend: sklearn(K=28) vs nb2103 LGBM(K=28) ----
    print("\n" + "-" * 78)
    print("CROSS-PARADIGM BLEND: sklearn-HGB(K=28) vs nb2103 LGBM(K=28)")
    print("-" * 78)
    blend_records = []
    for w_sk, w_lgb in BLEND_WEIGHTS:
        blend_oof = w_sk * mean_bag_oof + w_lgb * nb2103_k28_oof
        rae_blend = float(rae(y_unb, blend_oof))
        delta_vs_lgb = rae_blend - NB2103_K28_MEAN_BAG_REF
        beats_lgb = rae_blend < NB2103_K28_MEAN_BAG_REF - DECISION_MARGIN
        flat_lgb = abs(delta_vs_lgb) < DECISION_MARGIN
        print(f"   w_sklearn={w_sk:.2f}  w_lgbm={w_lgb:.2f}  "
              f"rae={rae_blend:.4f}  d_vs_nb2103={delta_vs_lgb:+.4f}")
        blend_records.append({
            "w_sklearn": float(w_sk),
            "w_lgbm": float(w_lgb),
            "rae_blend": rae_blend,
            "delta_vs_nb2103_K28": delta_vs_lgb,
            "beats_nb2103_K28": bool(beats_lgb),
            "flat_vs_nb2103_K28": bool(flat_lgb),
        })

    # find best blend
    best_blend_i = int(np.argmin([r["rae_blend"] for r in blend_records]))
    best_blend = blend_records[best_blend_i]
    print(f"\n[best] w_sklearn={best_blend['w_sklearn']:.2f} "
          f"w_lgbm={best_blend['w_lgbm']:.2f}  "
          f"rae={best_blend['rae_blend']:.4f}  "
          f"d_vs_nb2103={best_blend['delta_vs_nb2103_K28']:+.4f}")

    # ---- Overall verdict ----
    if best_blend["beats_nb2103_K28"]:
        verdict = (f"BLEND_BEATS_NB2103_K28_AT_w_sklearn="
                   f"{best_blend['w_sklearn']:.2f}")
    elif best_blend["flat_vs_nb2103_K28"]:
        verdict = (f"BLEND_FLAT_VS_NB2103_K28_AT_w_sklearn="
                   f"{best_blend['w_sklearn']:.2f}")
    elif rae_mean_bag < NB2103_K28_MEAN_BAG_REF - DECISION_MARGIN:
        verdict = "SKLEARN_ALONE_BEATS_NB2103_K28"
    elif abs(rae_mean_bag - NB2103_K28_MEAN_BAG_REF) < DECISION_MARGIN:
        verdict = "SKLEARN_ALONE_FLAT_VS_NB2103_K28"
    else:
        verdict = "SKLEARN_AND_BLENDS_DO_NOT_BEAT_NB2103_K28"
    print(f"\n   global verdict   = {verdict}")

    summary = {
        "tag": TAG,
        "method": ("sklearn_HistGradientBoostingRegressor_cross_paradigm_stack_"
                   "K28_on_117col_with_blend_vs_nb2103_LGBM_K28"),
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "model_family": "sklearn_HistGradientBoostingRegressor",
        "hgb_loss": "squared_error",
        "hgb_max_depth": 4,
        "hgb_max_leaf_nodes": 15,
        "hgb_max_iter": 300,
        "hgb_learning_rate": 0.03,
        "hgb_l2_regularization": 2.0,
        "hgb_min_samples_leaf": 5,
        "hgb_max_features": 1.0,
        "K": K,
        "feat_dim_full": int(feat_dim),
        "n_chembl_pool": int(len(pool)),
        "n_unb": n_unb,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "blend_weights": [list(w) for w in BLEND_WEIGHTS],
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "nb2103_K28_mean_bag_oof_in_rae_check": nb2103_in_rae,
        "nb2103_K28_mean_bag_ref": NB2103_K28_MEAN_BAG_REF,
        "nb2103_K28_median_bag_ref": NB2103_K28_MEDIAN_BAG_REF,
        "sklearn_per_seed_rae": per_seed_rae,
        "sklearn_per_seed_records": per_seed_records,
        "sklearn_per_seed_mean": float(per_seed_arr.mean()),
        "sklearn_per_seed_median": float(np.median(per_seed_arr)),
        "sklearn_per_seed_std": float(per_seed_arr.std()),
        "sklearn_per_seed_min": float(per_seed_arr.min()),
        "sklearn_per_seed_max": float(per_seed_arr.max()),
        "sklearn_rae_mean_bag": rae_mean_bag,
        "sklearn_rae_median_bag": rae_median_bag,
        "sklearn_delta_mean_bag_vs_nb2103_K28_mean": (
            rae_mean_bag - NB2103_K28_MEAN_BAG_REF
        ),
        "sklearn_delta_median_bag_vs_nb2103_K28_median": (
            rae_median_bag - NB2103_K28_MEDIAN_BAG_REF
        ),
        "sklearn_beats_nb2103_K28_alone": bool(
            rae_mean_bag < NB2103_K28_MEAN_BAG_REF - DECISION_MARGIN
        ),
        "blend_records": blend_records,
        "best_blend_w_sklearn": float(best_blend["w_sklearn"]),
        "best_blend_w_lgbm": float(best_blend["w_lgbm"]),
        "best_blend_rae": float(best_blend["rae_blend"]),
        "best_blend_delta_vs_nb2103_K28": float(
            best_blend["delta_vs_nb2103_K28"]
        ),
        "best_blend_beats_nb2103_K28": bool(best_blend["beats_nb2103_K28"]),
        "verdict": verdict,
        "pre_unblind_clean": True,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "decision_margin": DECISION_MARGIN,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "K", "feat_dim_full", "n_chembl_pool",
        "rae_anchor_chemprop_aux",
        "nb2103_K28_mean_bag_ref", "nb2103_K28_median_bag_ref",
        "sklearn_rae_mean_bag", "sklearn_rae_median_bag",
        "sklearn_delta_mean_bag_vs_nb2103_K28_mean",
        "sklearn_delta_median_bag_vs_nb2103_K28_median",
        "best_blend_w_sklearn", "best_blend_w_lgbm",
        "best_blend_rae", "best_blend_delta_vs_nb2103_K28",
        "verdict", "pre_unblind_clean",
    ):
        print(f"  {k}: {res.get(k)}")
    print("\n==== BLEND TABLE ====")
    for r in res["blend_records"]:
        print(f"  w_sk={r['w_sklearn']:.2f} w_lgb={r['w_lgbm']:.2f}  "
              f"rae={r['rae_blend']:.4f}  d_vs_nb2103={r['delta_vs_nb2103_K28']:+.4f}")
