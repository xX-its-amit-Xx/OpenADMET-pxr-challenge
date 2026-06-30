"""nb1155 -- GENUINE scaffold-CV refit of nb2167 sklearn HistGB candidate.

CONTEXT (per cycle 142 nb1140 re-audit):
    nb1140 found nb2167 sklearn HGB had scaffold-stratified RAE 0.4726 on its
    EXISTING random-KFold OOF, slightly below the nb2103 LGBM scaffold floor
    of 0.5057. But nb1140 only RE-EVALUATED the OOF preds in scaffold groups;
    it did NOT re-train the model under scaffold splits. The 0.4726 number is
    therefore "OOF-regrouping" and not a true scaffold-CV refit.

    nb1130 established the apples-to-apples scaffold-CV refit for nb2103
    LGBM(MSE) at K=28: 0.5057 (vs claimed random-KFold 0.4737, +0.032 gap).

    This script does the SAME for nb2167 sklearn HGB at K=28: train from
    scratch under scaffold_kfold_indices (seed=42, 5-fold) on the IDENTICAL
    SHAP top-28 feature matrix, then compare against the random-KFold claim
    AND against nb2103's scaffold-CV floor of 0.5057.

PROTOCOL:
    1. Rebuild the EXACT same 117-col 5-way K-tuned feature matrix as nb2167
       (AtomPair / MACCS / Mordred / ChempropEmbed / Avalon + ChEMBL kNN),
       using same nb1352/1392/1484/1523/1524/1541 K-grid winners and ChEMBL
       union pool. Slice to top-K=28 SHAP indices (nb2063_shap_importance_full117).
    2. Run sklearn.ensemble.HistGradientBoostingRegressor with the EXACT
       hyperparams from nb2167:
           loss='squared_error', max_depth=4, max_leaf_nodes=15,
           learning_rate=0.03, max_iter=300, l2_regularization=2,
           min_samples_leaf=5, max_features=1.0
       5 seeds (0, 1, 7, 42, 137).
    3. For EACH seed, run scaffold_kfold_indices(scaffolds, n_splits=5,
       seed=42) once (NOT seed=current_seed -- we use ONE deterministic
       scaffold split across all bag seeds for apples-to-apples comparison
       with nb1130/nb1140's seed=42 split).
    4. Cross-fit per seed under the scaffold split; bag across 5 seeds.
    5. final_OOF[i] = chemprop_aux[i] + bag_residual_OOF[i]
    6. Report scaffold-CV mean-bag RAE, scaffold-CV median-bag RAE.
    7. Compare against:
         (a) nb2167 random-KFold claim 0.4725 (the OOF-regrouping number)
         (b) nb2103 LGBM scaffold-CV floor 0.5057
       Decision margin 0.003.
    8. If GENUINE scaffold-CV refit < 0.5057 - 0.003 = 0.5027: REAL candidate
       (real cross-paradigm gain beyond LGBM at the same scaffold-CV floor)
    9. If essentially same as 0.5057 (within ±0.003): nb2167 was random-KFold
       -optimistic, no real signal across LGBM/HGB at same K=28 SHAP pruning.

Outputs:
    scripts/nb1155_nb2167_refit.py
    data/processed/nb1155_summary.json
    data/processed/nb1155_sklearn_mean_bag_oof_scaffoldCV_K28.npy
    data/processed/nb1155_sklearn_median_bag_oof_scaffoldCV_K28.npy
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
from sklearn.ensemble import HistGradientBoostingRegressor
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb1155"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

K = 28
RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]
SCAFFOLD_SPLIT_SEED = 42  # same as nb1130 / nb1140 for apples-to-apples

# Feature caches (identical to nb2167)
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

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

# References
CHEMPROP_AUX_REF = 0.6216
NB2167_CLAIMED_RANDOM_KFOLD = 0.4725   # nb2167 sklearn HGB mean-bag (random KFold)
NB2103_SCAFFOLD_CV_FLOOR = 0.5057      # nb1130 honest scaffold-CV for nb2103 LGBM K=28
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
    """Identical union to nb2167 (CHEMBL3401_raw + nr_extended + pxr_all_types)."""
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
    """Identical hyperparams to nb2167 sklearn HistGB."""
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


def _scaffold_cross_fit_one_seed(
    X: np.ndarray,
    residual: np.ndarray,
    seed: int,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Cross-fit sklearn HGB under GIVEN scaffold splits for one bag seed."""
    n = len(residual)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in splits:
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
    print(f"{TAG} -- GENUINE scaffold-CV refit of nb2167 sklearn HGB at K={K}")
    print(f"          anchor={ANCHOR}  bag_seeds={RESID_SEEDS}  "
          f"folds={RESID_FOLDS}  scaffold_split_seed={SCAFFOLD_SPLIT_SEED}")
    print(f"          ref(a): nb2167 random-KFold claim = "
          f"{NB2167_CLAIMED_RANDOM_KFOLD:.4f}  (OOF-regrouping)")
    print(f"          ref(b): nb2103 LGBM scaffold-CV floor = "
          f"{NB2103_SCAFFOLD_CV_FLOOR:.4f}  margin = {DECISION_MARGIN}")
    print("=" * 78)

    # ---- Load SHAP importance ----
    if not NB2063_SHAP_IMP.exists():
        raise FileNotFoundError(f"missing {NB2063_SHAP_IMP}")
    shap_imp_full117 = np.load(NB2063_SHAP_IMP).astype(np.float32)
    full_rank_order = np.argsort(-shap_imp_full117).astype(np.int32)
    print(f"[load] SHAP imp shape = {shap_imp_full117.shape}")

    # ---- Load anchor + truth + unblind indices ----
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

    # ---- Build scaffold splits on the 253 unblind (same as nb1130/nb1140) ----
    te_unb_smiles = [test_smiles[i] for i in unb_idx]
    scaffs = [bemis_murcko(s) for s in te_unb_smiles]
    n_unique_scaff = len(set(s for s in scaffs if s))
    n_none = sum(1 for s in scaffs if not s)
    splits = scaffold_kfold_indices(
        scaffs, n_splits=RESID_FOLDS, seed=SCAFFOLD_SPLIT_SEED
    )
    fold_sizes = [len(va) for _, va in splits]
    print(f"[scaff] unique={n_unique_scaff}  none={n_none}  "
          f"fold sizes seed={SCAFFOLD_SPLIT_SEED}: {fold_sizes}")

    # ---- Load 5-way K-tuned matrix winners (identical to nb2167) ----
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

    # ---- Feature matrices on unblind 253 ----
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
    print(f"[feat] top-{K} sklearn HGB input = {X_topK.shape}")

    # ---- GENUINE scaffold-CV refit: 5-seed bag, scaffold splits ----
    print("\n" + "-" * 78)
    print(f"SKLEARN HGB GENUINE SCAFFOLD-CV REFIT (K={K}, 5 seeds, "
          f"5-fold scaffold seed={SCAFFOLD_SPLIT_SEED})")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        resid_oof_s = _scaffold_cross_fit_one_seed(X_topK, residual, s, splits)
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

    print(f"\n[sklearn-HGB scaffold-CV] per-seed RAE = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"[sklearn-HGB scaffold-CV] per-seed mean = {per_seed_arr.mean():.4f}  "
          f"std = {per_seed_arr.std():.4f}")
    print(f"[sklearn-HGB scaffold-CV] mean-bag RAE   = {rae_mean_bag:.4f}")
    print(f"[sklearn-HGB scaffold-CV] median-bag RAE = {rae_median_bag:.4f}")

    np.save(DATA_PROCESSED / f"{TAG}_sklearn_mean_bag_oof_scaffoldCV_K{K}.npy",
            mean_bag_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_sklearn_median_bag_oof_scaffoldCV_K{K}.npy",
            median_bag_oof.astype(np.float32))

    # ---- Comparison vs both references ----
    print("\n" + "-" * 78)
    print("COMPARISONS")
    print("-" * 78)
    delta_vs_random = rae_mean_bag - NB2167_CLAIMED_RANDOM_KFOLD
    delta_vs_nb2103_scaff = rae_mean_bag - NB2103_SCAFFOLD_CV_FLOOR
    print(f"   genuine scaffold-CV mean-bag RAE = {rae_mean_bag:.4f}")
    print(f"   vs nb2167 random-KFold (0.4725)   = {delta_vs_random:+.4f}  "
          f"(positive = scaffold-CV is honest, random was optimistic)")
    print(f"   vs nb2103 LGBM scaffold floor (0.5057) = {delta_vs_nb2103_scaff:+.4f}  "
          f"(negative+|.|>=.003 = REAL cross-paradigm gain)")

    beats_nb2103_scaff = rae_mean_bag < NB2103_SCAFFOLD_CV_FLOOR - DECISION_MARGIN
    flat_nb2103_scaff = abs(delta_vs_nb2103_scaff) < DECISION_MARGIN
    is_random_optimism = (
        abs(delta_vs_random) > DECISION_MARGIN
        and rae_mean_bag > NB2167_CLAIMED_RANDOM_KFOLD
    )

    if beats_nb2103_scaff:
        verdict = (f"REAL_CANDIDATE_HGB_GENUINELY_BEATS_NB2103_LGBM_AT_SCAFFOLD_CV"
                   f"_BY_{abs(delta_vs_nb2103_scaff):.4f}")
    elif flat_nb2103_scaff:
        verdict = (f"NO_REAL_SIGNAL_HGB_SCAFFOLD_CV_FLAT_VS_NB2103_LGBM"
                   f"_NB2167_WAS_RANDOM_KFOLD_OPTIMISTIC_BY_{abs(delta_vs_random):.4f}")
    elif rae_mean_bag > NB2103_SCAFFOLD_CV_FLOOR + DECISION_MARGIN:
        verdict = (f"WORSE_THAN_NB2103_LGBM_SCAFFOLD_FLOOR"
                   f"_BY_{rae_mean_bag - NB2103_SCAFFOLD_CV_FLOOR:.4f}")
    else:
        verdict = "INDETERMINATE"

    print(f"\n   verdict = {verdict}")
    if is_random_optimism:
        print(f"   NB2167 RANDOM-KFOLD OPTIMISM CONFIRMED: "
              f"gap = {delta_vs_random:+.4f}")

    summary = {
        "tag": TAG,
        "method": ("GENUINE_scaffold_CV_refit_of_nb2167_sklearn_HGB_K28_"
                   "on_117col_with_chemprop_aux_anchor"),
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
        "hgb_hyperparams_identical_to_nb2167": True,
        "K": K,
        "feat_dim_full": int(feat_dim),
        "n_chembl_pool": int(len(pool)),
        "n_unb": n_unb,
        "n_unique_scaffolds_in_253": int(n_unique_scaff),
        "fold_sizes_scaffold_seed42": fold_sizes,
        "scaffold_split_seed": SCAFFOLD_SPLIT_SEED,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "cv_strategy": "scaffold_kfold_indices_GENUINE_REFIT",
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "nb2167_random_KFold_claim": NB2167_CLAIMED_RANDOM_KFOLD,
        "nb2103_LGBM_scaffold_CV_floor": NB2103_SCAFFOLD_CV_FLOOR,
        "decision_margin": DECISION_MARGIN,
        "sklearn_per_seed_rae_scaffoldCV": per_seed_rae,
        "sklearn_per_seed_records": per_seed_records,
        "sklearn_per_seed_mean_scaffoldCV": float(per_seed_arr.mean()),
        "sklearn_per_seed_median_scaffoldCV": float(np.median(per_seed_arr)),
        "sklearn_per_seed_std_scaffoldCV": float(per_seed_arr.std()),
        "sklearn_per_seed_min_scaffoldCV": float(per_seed_arr.min()),
        "sklearn_per_seed_max_scaffoldCV": float(per_seed_arr.max()),
        "sklearn_rae_mean_bag_scaffoldCV": rae_mean_bag,
        "sklearn_rae_median_bag_scaffoldCV": rae_median_bag,
        "delta_genuine_scaffoldCV_vs_random_KFold": delta_vs_random,
        "delta_genuine_scaffoldCV_vs_nb2103_LGBM_floor": delta_vs_nb2103_scaff,
        "beats_nb2103_LGBM_at_scaffold_CV": bool(beats_nb2103_scaff),
        "flat_vs_nb2103_LGBM_at_scaffold_CV": bool(flat_nb2103_scaff),
        "nb2167_was_random_KFold_optimistic": bool(is_random_optimism),
        "verdict": verdict,
        "pre_unblind_clean": True,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
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
        "nb2167_random_KFold_claim",
        "nb2103_LGBM_scaffold_CV_floor",
        "sklearn_rae_mean_bag_scaffoldCV",
        "sklearn_rae_median_bag_scaffoldCV",
        "delta_genuine_scaffoldCV_vs_random_KFold",
        "delta_genuine_scaffoldCV_vs_nb2103_LGBM_floor",
        "beats_nb2103_LGBM_at_scaffold_CV",
        "flat_vs_nb2103_LGBM_at_scaffold_CV",
        "nb2167_was_random_KFold_optimistic",
        "verdict", "pre_unblind_clean",
    ):
        print(f"  {k}: {res.get(k)}")
