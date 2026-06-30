"""nb2231 -- Recursive Feature Elimination K=28 -> K=20 / K=15 / K=10.

HYPOTHESIS:
    nb2171 effective stack is the K=28 SHAP top-K slice (nb2103_K28).
    The honest scaffold-CV RAE on the 253 unblind for the K=28 residual
    learner (chemprop_aux anchor + LGBM(MSE) on 28 SHAP-pruned features)
    is 0.5057 (nb2172 reference).  This script tests whether greedy
    backward RFE (drop the feature whose removal HURTS LEAST under
    scaffold-CV) can produce a leaner K=20 / K=15 / K=10 slice that
    matches or beats the K=28 baseline.

PROTOCOL:
    1. Rebuild the 117-col 5-way K-tuned matrix exactly as
       nb2063/nb2081/nb2091/nb2103/nb2111 (AtomPair / MACCS / Mordred /
       ChempropEmbed / Avalon + ChEMBL kNN), evaluate on the 253 unblind.
    2. Load nb2103 K=28 SHAP indices (= full_rank_order[:28] on
       data/processed/nb2063_shap_importance_full117.npy).  These are the
       starting feature subset.
    3. Compute K=28 baseline:
       chemprop_aux anchor + LGBM(MSE) on residual,
       5-fold scaffold-CV (bemis_murcko on unb smiles),
       5 kf_seeds {1001..1005}, mean across seeds.
    4. Greedy backward RFE: at each step
         a. For each feature in the current subset, evaluate scaffold-CV
            RAE with that feature DROPPED (using 1 seed for speed)
         b. Drop the feature whose removal MINIMISES the leave-out RAE
            (i.e. the least-useful feature; if all hurt, drop the
            smallest-hurt one)
         c. Re-evaluate the new subset with full 5-seed scaffold-CV
            (for the trajectory record).
    5. Trajectory stops:
         K=28 -> K=20 (8 drops) -> K=15 (5 more) -> K=10 (5 more)
       Total RFE drops = 18; trajectory snapshots at K=20/15/10.
    6. Compare K=20 / K=15 / K=10 RAE vs K=28 baseline (0.5057).
       decision_margin = 0.003.

Outputs:
    scripts/nb2231_rfe_K20.py
    data/processed/nb2231_summary.json

References:
    nb2103 K=28 mean_bag RAE on KFold = 0.4737   (in-sample optimism)
    nb2103 K=28 honest scaffold-CV RAE = 0.5057  (nb2172 baseline)
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

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2231"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

K_START = 28
K_TARGETS = [20, 15, 10]  # snapshot RAE at each
K_MIN = min(K_TARGETS)    # 10  -- final stop

N_FOLDS = 5
KF_SEEDS_FULL = [1001, 1002, 1003, 1004, 1005]   # full evaluation
KF_SEED_RFE = 1001                               # fast eval during search

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
NB2103_SUMMARY = DATA_PROCESSED / "nb2103_summary.json"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

CHEMPROP_AUX_REF = 0.6216
NB2103_K28_HONEST_SCAFFOLD_REF = 0.5057  # nb2172 trajectory baseline
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
    """Same union as nb1852/nb1861/nb2063/nb2081/nb2091/nb2103/nb2111."""
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


def _lgbm_params(seed: int) -> dict:
    """LGBM(MSE) -- identical to nb1852/nb1861/nb2063/nb2081/nb2091/nb2103."""
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


def _scaffold_cv_one_seed(X: np.ndarray, residual: np.ndarray,
                          unb_scaffolds: list[str], kf_seed: int) -> np.ndarray:
    """One scaffold 5-fold cross-fit of residual LGBM; returns OOF residual."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n = len(residual)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in splits:
        mdl = lgb.LGBMRegressor(**_lgbm_params(kf_seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _eval_subset(X_full: np.ndarray, col_idx: list[int],
                 residual: np.ndarray, anchor: np.ndarray,
                 y_unb: np.ndarray, unb_scaffolds: list[str],
                 kf_seeds: list[int]) -> dict:
    """Multi-seed scaffold-CV evaluation of a feature subset."""
    X_sub = X_full[:, col_idx].astype(np.float32)
    n_unb = len(y_unb)
    per_seed_oof = np.zeros((len(kf_seeds), n_unb), dtype=np.float64)
    per_seed_rae = []
    for i, s in enumerate(kf_seeds):
        oof = _scaffold_cv_one_seed(X_sub, residual, unb_scaffolds, s)
        per_seed_oof[i] = oof
        pred_corr = anchor + oof
        per_seed_rae.append(float(rae(y_unb, pred_corr)))
    mean_bag = per_seed_oof.mean(axis=0)
    median_bag = np.median(per_seed_oof, axis=0)
    return {
        "n_feat": len(col_idx),
        "per_seed_rae": per_seed_rae,
        "rae_per_seed_mean": float(np.mean(per_seed_rae)),
        "rae_per_seed_std": float(np.std(per_seed_rae)),
        "rae_mean_bag": float(rae(y_unb, anchor + mean_bag)),
        "rae_median_bag": float(rae(y_unb, anchor + median_bag)),
    }


def _eval_subset_fast(X_full: np.ndarray, col_idx: list[int],
                      residual: np.ndarray, anchor: np.ndarray,
                      y_unb: np.ndarray, unb_scaffolds: list[str],
                      kf_seed: int) -> float:
    """Single-seed scaffold-CV RAE for RFE drop selection."""
    X_sub = X_full[:, col_idx].astype(np.float32)
    oof = _scaffold_cv_one_seed(X_sub, residual, unb_scaffolds, kf_seed)
    return float(rae(y_unb, anchor + oof))


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- RFE K=28 -> K=20 / K=15 / K=10 on 117-col 5-way matrix")
    print(f"          anchor={ANCHOR}  scaffold-CV {N_FOLDS}-fold "
          f"x {len(KF_SEEDS_FULL)} seeds")
    print(f"          baseline: nb2103 K=28 honest scaffold-CV "
          f"= {NB2103_K28_HONEST_SCAFFOLD_REF:.4f}  "
          f"(margin {DECISION_MARGIN})")
    print("=" * 78)

    # ---- load nb2103 SHAP top-28 indices ----
    if not NB2063_SHAP_IMP.exists():
        raise FileNotFoundError(f"missing {NB2063_SHAP_IMP}")
    shap_imp_full117 = np.load(NB2063_SHAP_IMP).astype(np.float32)
    full_rank_order = np.argsort(-shap_imp_full117).astype(np.int32)
    shap_top28 = full_rank_order[:K_START].tolist()
    print(f"[load] SHAP top-{K_START} indices in 117-col matrix: "
          f"{shap_top28[:10]}... ({len(shap_top28)} total)")

    # cross-check vs nb2103 K=28 entry
    if NB2103_SUMMARY.exists():
        with open(NB2103_SUMMARY) as f:
            nb2103_sum = json.load(f)
        for r in nb2103_sum.get("per_K_records", []):
            if int(r.get("K", -1)) == 28:
                k28_cached = list(r["top_K_idx_in_117"])
                if k28_cached == shap_top28:
                    print("   [check] nb2103 K=28 top-28 idx matches (OK)")
                else:
                    print("   [warn] nb2103 K=28 top-28 idx differs from "
                          "fresh re-rank -- using fresh re-rank")
                break

    # ---- load truth + anchor ----
    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [test_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique scaffolds in unb 253 = {n_unique_scaf}")

    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"chemprop_aux te missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} in_RAE = {rae_anchor:.4f}  (ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- rebuild 117-col matrix (same as nb2103) ----
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

    n_top_maccs = int(len(top_maccs_bit_idx))
    n_top_mord = int(len(top_mord_col_idx))
    n_top_ap = int(len(top_ap_bit_idx))
    n_top_embed = int(len(top_embed_col_idx))
    n_top_avalon = int(len(top_avalon_bit_idx))

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
    expected_dim = (n_top_ap + n_top_maccs + n_top_mord
                    + n_top_embed + n_top_avalon + 2)
    if feat_dim != expected_dim:
        raise ValueError(f"feat_dim {feat_dim} != expected {expected_dim}")
    print(f"\n   COMBINED 5-way K-tuned matrix: {X_unb.shape}")

    # ---- feature names (full 117) ----
    feat_names: list[str] = []
    feat_family: list[str] = []
    for b in top_ap_bit_idx:
        feat_names.append(f"AtomPair_bit_{int(b)}")
        feat_family.append("AtomPair")
    for b in top_maccs_bit_idx:
        feat_names.append(f"MACCS_bit_{int(b)}")
        feat_family.append("MACCS")
    for c in top_mord_col_idx:
        feat_names.append(f"Mordred_col_{int(c)}")
        feat_family.append("Mordred")
    for d in top_embed_col_idx:
        feat_names.append(f"ChempropEmbed_dim_{int(d)}")
        feat_family.append("ChempropEmbed")
    for b in top_avalon_bit_idx:
        feat_names.append(f"Avalon_bit_{int(b)}")
        feat_family.append("Avalon")
    feat_names.append("pred_chembl_pec50")
    feat_family.append("ChEMBL_kNN")
    feat_names.append("mean_sim")
    feat_family.append("ChEMBL_kNN")
    assert len(feat_names) == feat_dim

    # ---- K=28 baseline (full 5-seed scaffold-CV) ----
    print("\n" + "-" * 78)
    print(f"K={K_START} BASELINE (full {len(KF_SEEDS_FULL)}-seed scaffold-CV)")
    print("-" * 78)
    t_b = time.time()
    base = _eval_subset(
        X_unb, shap_top28, residual, anchor, y_unb,
        unb_scaffolds, KF_SEEDS_FULL,
    )
    base_fam: dict[str, int] = {}
    for j in shap_top28:
        base_fam[feat_family[j]] = base_fam.get(feat_family[j], 0) + 1
    print(f"   per_seed_rae = "
          f"[{', '.join(f'{r:.4f}' for r in base['per_seed_rae'])}]")
    print(f"   per_seed_mean= {base['rae_per_seed_mean']:.4f}  "
          f"std={base['rae_per_seed_std']:.4f}")
    print(f"   mean_bag     = {base['rae_mean_bag']:.4f}")
    print(f"   median_bag   = {base['rae_median_bag']:.4f}")
    print(f"   family_counts= {base_fam}")
    print(f"   delta vs nb2172_ref ({NB2103_K28_HONEST_SCAFFOLD_REF:.4f}) "
          f"= {base['rae_per_seed_mean'] - NB2103_K28_HONEST_SCAFFOLD_REF:+.4f}")
    print(f"   wall = {time.time() - t_b:.1f}s")

    # ---- greedy backward RFE: K=28 -> K=10 ----
    print("\n" + "-" * 78)
    print(f"GREEDY RFE: K={K_START} -> K={K_MIN}  "
          f"(scaffold-CV with kf_seed={KF_SEED_RFE} for drop selection)")
    print("-" * 78)
    current = list(shap_top28)
    rfe_trajectory: list[dict] = []
    snapshot_records: dict[int, dict] = {}

    # record initial state
    rfe_trajectory.append({
        "step": 0,
        "K_after": len(current),
        "feat_dropped": None,
        "feat_dropped_name": None,
        "feat_dropped_family": None,
        "rae_fast_seed1001": float(_eval_subset_fast(
            X_unb, current, residual, anchor, y_unb,
            unb_scaffolds, KF_SEED_RFE,
        )),
    })

    step = 0
    while len(current) > K_MIN:
        step += 1
        # find feature whose removal minimizes RAE
        cand_results = []
        for j in current:
            trial = [k for k in current if k != j]
            r_trial = _eval_subset_fast(
                X_unb, trial, residual, anchor, y_unb,
                unb_scaffolds, KF_SEED_RFE,
            )
            cand_results.append((j, r_trial))
        cand_results.sort(key=lambda x: x[1])
        drop_j, drop_rae = cand_results[0]
        current = [k for k in current if k != drop_j]
        rfe_trajectory.append({
            "step": step,
            "K_after": len(current),
            "feat_dropped": int(drop_j),
            "feat_dropped_name": feat_names[drop_j],
            "feat_dropped_family": feat_family[drop_j],
            "rae_fast_seed1001": float(drop_rae),
        })
        print(f"   step {step:2d}  drop col={drop_j:3d} "
              f"({feat_names[drop_j]:30s} fam={feat_family[drop_j]:14s})  "
              f"K_after={len(current):2d}  fast_rae={drop_rae:.4f}")

        if len(current) in K_TARGETS:
            # full-seed snapshot
            snap = _eval_subset(
                X_unb, current, residual, anchor, y_unb,
                unb_scaffolds, KF_SEEDS_FULL,
            )
            fam_counts: dict[str, int] = {}
            for j in current:
                fam_counts[feat_family[j]] = fam_counts.get(
                    feat_family[j], 0
                ) + 1
            snap["family_counts"] = fam_counts
            snap["surviving_idx_in_117"] = [int(j) for j in current]
            snap["surviving_names"] = [feat_names[j] for j in current]
            snap["delta_per_seed_mean_vs_K28_baseline"] = (
                snap["rae_per_seed_mean"] - base["rae_per_seed_mean"]
            )
            snap["delta_per_seed_mean_vs_nb2172_ref"] = (
                snap["rae_per_seed_mean"] - NB2103_K28_HONEST_SCAFFOLD_REF
            )
            snap["delta_mean_bag_vs_K28_baseline"] = (
                snap["rae_mean_bag"] - base["rae_mean_bag"]
            )
            snap["beats_K28_baseline"] = bool(
                snap["rae_per_seed_mean"]
                < base["rae_per_seed_mean"] - DECISION_MARGIN
            )
            snap["flat_vs_K28_baseline"] = bool(
                abs(snap["rae_per_seed_mean"] - base["rae_per_seed_mean"])
                < DECISION_MARGIN
            )
            snapshot_records[len(current)] = snap
            print(f"   ---- SNAPSHOT K={len(current)} ----")
            print(f"        per_seed_rae = "
                  f"[{', '.join(f'{r:.4f}' for r in snap['per_seed_rae'])}]")
            print(f"        per_seed_mean= {snap['rae_per_seed_mean']:.4f}  "
                  f"std={snap['rae_per_seed_std']:.4f}")
            print(f"        mean_bag     = {snap['rae_mean_bag']:.4f}")
            print(f"        median_bag   = {snap['rae_median_bag']:.4f}")
            print(f"        d_per_seed_vs_K28 baseline = "
                  f"{snap['delta_per_seed_mean_vs_K28_baseline']:+.4f}")
            print(f"        d_per_seed_vs_nb2172_ref   = "
                  f"{snap['delta_per_seed_mean_vs_nb2172_ref']:+.4f}")
            print(f"        family_counts= {fam_counts}")

    # ---- summary table ----
    print("\n" + "=" * 78)
    print("RFE SNAPSHOT TABLE")
    print("=" * 78)
    print(f"   {'K':>4s}  {'per_seed_mean':>13s}  {'std':>6s}  "
          f"{'mean_bag':>9s}  {'median_bag':>10s}  "
          f"{'d_per_seed':>10s}  verdict")
    print(f"   {K_START:>4d}  {base['rae_per_seed_mean']:>13.4f}  "
          f"{base['rae_per_seed_std']:>6.4f}  "
          f"{base['rae_mean_bag']:>9.4f}  {base['rae_median_bag']:>10.4f}  "
          f"{0.0:>+10.4f}  BASELINE(K={K_START})")
    for K in K_TARGETS:
        snap = snapshot_records[K]
        if snap["beats_K28_baseline"]:
            v = "BEATS_K28"
        elif snap["flat_vs_K28_baseline"]:
            v = "FLAT_VS_K28"
        else:
            v = "HURTS_K28"
        print(f"   {K:>4d}  {snap['rae_per_seed_mean']:>13.4f}  "
              f"{snap['rae_per_seed_std']:>6.4f}  "
              f"{snap['rae_mean_bag']:>9.4f}  {snap['rae_median_bag']:>10.4f}  "
              f"{snap['delta_per_seed_mean_vs_K28_baseline']:>+10.4f}  {v}")

    # ---- overall verdict ----
    cand = [(K_START, base["rae_per_seed_mean"])]
    for K in K_TARGETS:
        cand.append((K, snapshot_records[K]["rae_per_seed_mean"]))
    cand.sort(key=lambda x: x[1])
    best_K, best_rae = cand[0]
    if best_K == K_START:
        global_verdict = "K28_BASELINE_REMAINS_OPTIMAL_RFE_DOES_NOT_HELP"
    elif best_rae < base["rae_per_seed_mean"] - DECISION_MARGIN:
        global_verdict = f"RFE_BEATS_K28_AT_K={best_K}"
    elif abs(best_rae - base["rae_per_seed_mean"]) < DECISION_MARGIN:
        global_verdict = f"RFE_FLAT_VS_K28_BEST_K={best_K}"
    else:
        global_verdict = f"RFE_HURTS_K28_BEST_RFE_K={best_K}"
    print(f"\n   best K = {best_K}  (per_seed_mean RAE {best_rae:.4f})")
    print(f"   global verdict = {global_verdict}")

    # ---- save summary ----
    summary = {
        "tag": TAG,
        "method": ("greedy_backward_RFE_on_K28_SHAP_top_117col_"
                   "scaffold_cv_residual_lgbm"),
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "K_start": K_START,
        "K_targets": K_TARGETS,
        "K_min": K_MIN,
        "n_folds": N_FOLDS,
        "kf_seeds_full": KF_SEEDS_FULL,
        "kf_seed_rfe": KF_SEED_RFE,
        "feat_dim_full": int(feat_dim),
        "feat_breakdown_full": {
            "atompair": n_top_ap,
            "maccs": n_top_maccs,
            "mordred": n_top_mord,
            "chemprop_embed": n_top_embed,
            "avalon": n_top_avalon,
            "pred_chembl_pec50": 1,
            "mean_sim": 1,
            "total": int(feat_dim),
        },
        "n_chembl_pool": int(len(pool)),
        "n_unb": n_unb,
        "unb_n_unique_scaffolds": n_unique_scaf,
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "lgbm_objective": "regression",
        "lgbm_max_depth": 4,
        "lgbm_num_leaves": 15,
        "lgbm_n_estimators": 300,
        "lgbm_learning_rate": 0.03,
        "lgbm_min_child_samples": 5,
        "lgbm_reg_lambda": 2.0,
        "shap_top28_idx_in_117": [int(j) for j in shap_top28],
        "shap_top28_names": [feat_names[j] for j in shap_top28],
        "shap_top28_families": [feat_family[j] for j in shap_top28],
        "K28_baseline_family_counts": base_fam,
        "K28_baseline": {
            "K": K_START,
            "per_seed_rae": base["per_seed_rae"],
            "rae_per_seed_mean": base["rae_per_seed_mean"],
            "rae_per_seed_std": base["rae_per_seed_std"],
            "rae_mean_bag": base["rae_mean_bag"],
            "rae_median_bag": base["rae_median_bag"],
            "delta_vs_nb2172_ref": (
                base["rae_per_seed_mean"] - NB2103_K28_HONEST_SCAFFOLD_REF
            ),
        },
        "rfe_trajectory": rfe_trajectory,
        "snapshots": {str(K): snapshot_records[K] for K in K_TARGETS},
        "best_K_overall": int(best_K),
        "best_rae_overall": float(best_rae),
        "delta_best_vs_K28_baseline": float(
            best_rae - base["rae_per_seed_mean"]
        ),
        "verdict": global_verdict,
        "pre_unblind_clean": True,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "nb2172_K28_ref": NB2103_K28_HONEST_SCAFFOLD_REF,
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
        "K_start",
        "K_targets",
        "feat_dim_full",
        "rae_anchor_chemprop_aux",
        "nb2172_K28_ref",
        "best_K_overall",
        "best_rae_overall",
        "delta_best_vs_K28_baseline",
        "verdict",
        "pre_unblind_clean",
    ):
        print(f"  {k}: {res.get(k)}")
    print("\n==== K SNAPSHOT TABLE ====")
    base = res["K28_baseline"]
    print(f"  K={base['K']:>3d}  per_seed_mean={base['rae_per_seed_mean']:.4f}  "
          f"mean_bag={base['rae_mean_bag']:.4f}  median_bag={base['rae_median_bag']:.4f}  "
          f"BASELINE")
    for K_str, snap in res["snapshots"].items():
        print(f"  K={int(K_str):>3d}  "
              f"per_seed_mean={snap['rae_per_seed_mean']:.4f}  "
              f"mean_bag={snap['rae_mean_bag']:.4f}  "
              f"median_bag={snap['rae_median_bag']:.4f}  "
              f"d_vs_K28={snap['delta_per_seed_mean_vs_K28_baseline']:+.4f}")
