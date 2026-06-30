"""nb1151 -- Honest SCAFFOLD-CV K-sweep on LGBM(MSE) residual stack.

HYPOTHESIS:
    nb2103 found mean-bag RAE = 0.4737 at K=28 using random KFold cross-fit on
    the SHAP top-K of the 117-col 5-way K-tuned matrix.  Random KFold is
    optimistic on the analog-expansion test set because compounds from the
    same scaffold cluster leak between folds.  A SCAFFOLD-aware 5-fold CV
    (Murcko-scaffold groups, each scaffold entirely in one fold) reports the
    honest out-of-scaffold generalisation, which is what the LB sees.

    This notebook sweeps K in {15, 20, 28, 35, 50} under SCAFFOLD-CV and asks:
    does the random-KFold K=28 optimum survive scaffold-CV, or does a different
    K become optimal under the harder splitting protocol?

PROTOCOL:
    1. Reuse the SHAP importance ranking from nb2063
       (data/processed/nb2063_shap_importance_full117.npy) -- same 117-col
       5-way K-tuned matrix as nb2063/nb2081/nb2091/nb2103.
    2. For each K in {15, 20, 28, 35, 50}: slice X_unb to top-K cols, run a
       5-seed bag of LGBM(MSE) with SCAFFOLD KFold (n=5) cross-fit per seed.
       Scaffolds computed from the 253 unblind SMILES via Murcko skeleton.
    3. Compute final = chemprop_aux + cross-fit residual; mean-bag RAE per K.
    4. Compare each K vs nb2103 K=28 reference (0.5057 scaffold-CV).  Gate
       at decision_margin = 0.003.
    5. Identify scaffold-CV optimum K (may differ from nb2103 random-KFold K=28).
    6. If best K beats 0.5027 (= 0.5057 - 0.003): build deploy CSV using
       deploy-refit residual on top-K features at the winning K.

Outputs:
    scripts/nb1151_scaffold_k_sweep.py
    data/processed/nb1151_summary.json
    data/processed/nb1151_mean_bag_oof_K{K}.npy   (253,) float32 per K
    submissions/nb1151_scaffold_K{best_K}.csv     ONLY if best_K beats 0.5027
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
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb1151"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]
K_GRID = [15, 20, 28, 35, 50]

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
NB2063_SUMMARY = DATA_PROCESSED / "nb2063_summary.json"
NB2063_SHAP_IMP = DATA_PROCESSED / "nb2063_shap_importance_full117.npy"
NB2103_SUMMARY = DATA_PROCESSED / "nb2103_summary.json"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

# References (PRE-unblind path)
CHEMPROP_AUX_REF = 0.6216
NB2103_K28_SCAFFOLD_REF = 0.5057   # nb2103 K=28 under scaffold-CV (task spec)
NB2103_K28_RANDOM_REF = 0.4737     # nb2103 K=28 under random KFold (sanity)
DECISION_MARGIN = 0.003
DEPLOY_GATE = NB2103_K28_SCAFFOLD_REF - DECISION_MARGIN   # 0.5027


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


def _murcko_scaffold(mol):
    try:
        if mol is None:
            return None
        sc = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(sc) if sc is not None else None
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
    print(f"   [pool] after InChIKey dedup (median agg): {len(agg)} unique cpds")
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
    """LGBM(MSE) -- identical to nb2063/nb2081/nb2091/nb2103."""
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


def _residual_scaffold_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray,
                                          scaffolds: list,
                                          seed: int) -> np.ndarray:
    """Scaffold KFold cross-fit. Each scaffold entirely in one fold."""
    n = len(residual)
    splits = scaffold_kfold_indices(scaffolds, n_splits=RESID_FOLDS,
                                    shuffle=True, seed=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in splits:
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


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


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- SCAFFOLD-CV K-sweep {K_GRID} on 117-col 5-way K-tuned matrix")
    print(f"          anchor={ANCHOR}  seeds={RESID_SEEDS}  folds={RESID_FOLDS}")
    print(f"          ref: nb2103 K=28 SCAFFOLD-CV = {NB2103_K28_SCAFFOLD_REF:.4f}"
          f"   gate (deploy) = {DEPLOY_GATE:.4f}")
    print(f"          ref: nb2103 K=28 random-KF   = "
          f"{NB2103_K28_RANDOM_REF:.4f}  (sanity)")
    print("=" * 78)

    # ---- nb2063 SHAP importance reference ----
    if not NB2063_SHAP_IMP.exists():
        raise FileNotFoundError(f"missing {NB2063_SHAP_IMP} -- run nb2063 first")
    shap_imp_full117 = np.load(NB2063_SHAP_IMP).astype(np.float32)
    print(f"[ref] nb2063 SHAP importance shape = {shap_imp_full117.shape}")
    full_rank_order = np.argsort(-shap_imp_full117).astype(np.int32)

    # nb2103 reference (verify gate value if available)
    if NB2103_SUMMARY.exists():
        with open(NB2103_SUMMARY) as f:
            nb2103_sum = json.load(f)
        for r in nb2103_sum.get("per_K_records", []):
            if int(r.get("K", -1)) == 28:
                print(f"[ref] nb2103.K=28 random-KF mean_bag_rae = "
                      f"{float(r['rae_mean_bag']):.4f}  (note: scaffold-CV "
                      f"gate uses task-spec 0.5057)")
                break

    # ---- Load anchor + truth ----
    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"chemprop_aux te file missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(
            f"chemprop_aux te shape mismatch: {te_anchor_513.shape} vs {n_test}"
        )
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Compute scaffolds for the 253 unblind compounds ----
    unb_smiles = [test_smiles[i] for i in unb_idx]
    unb_mols = [standardize(s) for s in unb_smiles]
    scaffolds_unb = [_murcko_scaffold(m) for m in unb_mols]
    n_unique_sc = len({s for s in scaffolds_unb if s})
    n_none_sc = sum(1 for s in scaffolds_unb if not s)
    from collections import Counter
    sc_counts = Counter(s for s in scaffolds_unb if s)
    n_singleton_sc = sum(1 for v in sc_counts.values() if v == 1)
    largest_sc = max(sc_counts.values()) if sc_counts else 0
    print(f"[scaf] n_unb_with_scaf = {n_unb - n_none_sc}  "
          f"n_unique_scaffolds = {n_unique_sc}  "
          f"n_singleton_scaffolds = {n_singleton_sc}  "
          f"largest_scaffold = {largest_sc}")

    # Sanity print: split sizes for seed 0
    splits0 = scaffold_kfold_indices(scaffolds_unb, n_splits=RESID_FOLDS,
                                     shuffle=True, seed=0)
    for f, (tr, va) in enumerate(splits0):
        print(f"   [scaf-fold-seed0] fold{f}: n_tr={len(tr)}  n_va={len(va)}")

    # ---- Load all K-grid winners (same as nb2063/nb2103) ----
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
    print(f"[reuse] top-{n_top_ap}     AtomPair bits (nb1524 K={K_AP_best})")
    print(f"[reuse] top-{n_top_maccs}     MACCS bits  (nb1352)")
    print(f"[reuse] top-{n_top_mord}     Mordred cols (nb1523 K={K_Mord_best})")
    print(f"[reuse] top-{n_top_embed}     ChempropEmbed dims (nb1541 K={K_Embed_best})")
    print(f"[reuse] top-{n_top_avalon}     Avalon bits (nb1392 SHAP K=30)")

    # ---- Feature matrices (unblind slice only -- same as nb2103) ----
    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test)
    X_ap_unb_top = X_ap_te[unb_idx][:, top_ap_bit_idx].astype(np.float32)
    print(f"[feat] X_ap_unb_top      = {X_ap_unb_top.shape}")

    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)
    X_maccs_unb_top = X_maccs_te[unb_idx][:, top_maccs_bit_idx].astype(np.float32)
    print(f"[feat] X_maccs_unb_top   = {X_maccs_unb_top.shape}")

    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    X_mord_unb_top = X_mord_te[unb_idx][:, top_mord_col_idx].astype(np.float32)
    print(f"[feat] X_mord_unb_top    = {X_mord_unb_top.shape}")

    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)
    X_emb_unb_top = X_emb_te[unb_idx][:, top_embed_col_idx].astype(np.float32)
    print(f"[feat] X_emb_unb_top     = {X_emb_unb_top.shape}")

    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)
    X_av_unb_top = X_av_te[unb_idx][:, top_avalon_bit_idx].astype(np.float32)
    print(f"[feat] X_av_unb_top      = {X_av_unb_top.shape}")

    # ---- ChEMBL kNN feature (same as nb2063/nb2103) ----
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
    n_after = len(pool)
    print(f"   pool: {n_before} -> {n_after}  (dropped {n_before - n_after})")

    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep_pool = fp_pool.sum(axis=1) > 0
    if not keep_pool.all():
        pool = pool[keep_pool].reset_index(drop=True)
        fp_pool = fp_pool[keep_pool]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))
    print(f"   final pool size: {len(pool)}  median pEC50 = {pool_median:.3f}")

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

    # ---- Build COMBINED 5-way K-tuned 117-col feature matrix (unblind) ----
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
    print(f"\n   COMBINED 5-WAY K-TUNED matrix: {X_unb.shape}")

    if feat_dim != shap_imp_full117.shape[0]:
        raise ValueError(
            f"feat_dim {feat_dim} != nb2063 SHAP importance length "
            f"{shap_imp_full117.shape[0]}"
        )

    # ---- Feature names (same as nb2063/nb2103) ----
    feat_names: list[str] = []
    feat_family: list[str] = []
    for j, b in enumerate(top_ap_bit_idx):
        feat_names.append(f"AtomPair_bit_{int(b)}")
        feat_family.append("AtomPair")
    for j, b in enumerate(top_maccs_bit_idx):
        feat_names.append(f"MACCS_bit_{int(b)}")
        feat_family.append("MACCS")
    for j, c in enumerate(top_mord_col_idx):
        feat_names.append(f"Mordred_col_{int(c)}")
        feat_family.append("Mordred")
    for j, d in enumerate(top_embed_col_idx):
        feat_names.append(f"ChempropEmbed_dim_{int(d)}")
        feat_family.append("ChempropEmbed")
    for j, b in enumerate(top_avalon_bit_idx):
        feat_names.append(f"Avalon_bit_{int(b)}")
        feat_family.append("Avalon")
    feat_names.append("pred_chembl_pec50")
    feat_family.append("ChEMBL_kNN")
    feat_names.append("mean_sim")
    feat_family.append("ChEMBL_kNN")
    assert len(feat_names) == feat_dim

    # ---- SCAFFOLD-CV K-grid sweep ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD-CV K-GRID SWEEP: {K_GRID}")
    print("-" * 78)
    per_K_results: list[dict] = []
    for K in K_GRID:
        print(f"\n--- K={K} (SCAFFOLD-CV) ---")
        topK_idx = full_rank_order[:K].astype(np.int32)
        topK_names = [feat_names[i] for i in topK_idx]
        topK_family = [feat_family[i] for i in topK_idx]
        fam_counts: dict[str, int] = {}
        for fam in topK_family:
            fam_counts[fam] = fam_counts.get(fam, 0) + 1
        print(f"   top-{K} family breakdown: {fam_counts}")

        X_topK = X_unb[:, topK_idx].astype(np.float32)
        per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
        per_seed_rae: list[float] = []
        per_seed_records = []
        for i, s in enumerate(RESID_SEEDS):
            ts = time.time()
            resid_oof_s = _residual_scaffold_cross_fit_one_seed(
                X_topK, residual, scaffolds_unb, s
            )
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
            print(f"   K={K} seed={s:3d}:  rae_corr = {rae_s:.4f}  "
                  f"(d_vs_anchor = {delta_s:+.4f})  "
                  f"wall = {time.time() - ts:.1f}s")

        mean_bag_oof = per_seed_corrected.mean(axis=0)
        median_bag_oof = np.median(per_seed_corrected, axis=0)
        rae_mean_bag = float(rae(y_unb, mean_bag_oof))
        rae_median_bag = float(rae(y_unb, median_bag_oof))

        per_seed_rae_arr = np.array(per_seed_rae)
        rae_per_seed_mean = float(per_seed_rae_arr.mean())
        rae_per_seed_median = float(np.median(per_seed_rae_arr))
        rae_per_seed_std = float(per_seed_rae_arr.std())
        rae_per_seed_min = float(per_seed_rae_arr.min())
        rae_per_seed_max = float(per_seed_rae_arr.max())

        delta_vs_ref = rae_mean_bag - NB2103_K28_SCAFFOLD_REF
        beats_ref = rae_mean_bag < NB2103_K28_SCAFFOLD_REF - DECISION_MARGIN
        flat_vs_ref = abs(delta_vs_ref) < DECISION_MARGIN
        beats_anchor = rae_mean_bag < rae_anchor - DECISION_MARGIN

        print(f"   K={K} per-seed RAE   = "
              f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
        print(f"   K={K} per-seed mean  = {rae_per_seed_mean:.4f}  "
              f"std = {rae_per_seed_std:.4f}")
        print(f"   K={K} pooled mean    = {rae_mean_bag:.4f}  "
              f"(d_vs_anchor = {rae_mean_bag - rae_anchor:+.4f}  "
              f"d_vs_nb2103_K28_scaf = {delta_vs_ref:+.4f})")
        print(f"   K={K} pooled median  = {rae_median_bag:.4f}")
        if beats_ref:
            verdict_K = "BEATS_NB2103_K28_SCAFFOLD_REF"
        elif flat_vs_ref:
            verdict_K = "FLAT_VS_NB2103_K28_SCAFFOLD_REF"
        elif beats_anchor:
            verdict_K = "BEATS_ANCHOR_BUT_WORSE_THAN_NB2103_K28_SCAF"
        elif abs(rae_mean_bag - rae_anchor) < DECISION_MARGIN:
            verdict_K = "FLAT_VS_ANCHOR"
        else:
            verdict_K = "HURTS_ANCHOR"
        print(f"   K={K} verdict        = {verdict_K}")

        # Save mean-bag OOF per K
        out_p = DATA_PROCESSED / f"{TAG}_mean_bag_oof_K{K}.npy"
        np.save(out_p, mean_bag_oof.astype(np.float32))
        print(f"   [save] {out_p}")

        per_K_results.append({
            "K": int(K),
            "feat_dim": int(K),
            "family_counts": fam_counts,
            "top_K_idx_in_117": topK_idx.tolist(),
            "per_seed_rae": per_seed_rae,
            "per_seed_records": per_seed_records,
            "rae_per_seed_mean": rae_per_seed_mean,
            "rae_per_seed_median": rae_per_seed_median,
            "rae_per_seed_std": rae_per_seed_std,
            "rae_per_seed_min": rae_per_seed_min,
            "rae_per_seed_max": rae_per_seed_max,
            "rae_mean_bag": rae_mean_bag,
            "rae_median_bag": rae_median_bag,
            "delta_mean_bag_vs_chemprop_aux": rae_mean_bag - rae_anchor,
            "delta_mean_bag_vs_nb2103_K28_scaffold": delta_vs_ref,
            "beats_chemprop_aux": bool(beats_anchor),
            "beats_nb2103_K28_scaffold": bool(beats_ref),
            "flat_vs_nb2103_K28_scaffold": bool(flat_vs_ref),
            "verdict": verdict_K,
        })

    # ---- Select best K under scaffold-CV ----
    print("\n" + "=" * 78)
    print("SCAFFOLD-CV K-SWEEP SUMMARY TABLE")
    print("=" * 78)
    print(f"   {'K':>4s}  {'mean_bag':>10s}  {'median_bag':>10s}  "
          f"{'per_seed_mean':>13s}  {'per_seed_std':>12s}  "
          f"{'d_vs_anchor':>11s}  {'d_vs_ref':>10s}  verdict")
    print(f"   {'ref':>4s}  {NB2103_K28_SCAFFOLD_REF:>10.4f}  "
          f"{NB2103_K28_SCAFFOLD_REF:>10.4f}  {'N/A':>13s}  {'N/A':>12s}  "
          f"{NB2103_K28_SCAFFOLD_REF - rae_anchor:>+11.4f}  {0.0:>+10.4f}  "
          f"NB2103_K28_SCAF_REF")
    for r in per_K_results:
        print(f"   {r['K']:>4d}  {r['rae_mean_bag']:>10.4f}  "
              f"{r['rae_median_bag']:>10.4f}  {r['rae_per_seed_mean']:>13.4f}  "
              f"{r['rae_per_seed_std']:>12.4f}  "
              f"{r['delta_mean_bag_vs_chemprop_aux']:>+11.4f}  "
              f"{r['delta_mean_bag_vs_nb2103_K28_scaffold']:>+10.4f}  "
              f"{r['verdict']}")

    sweep_rae = [r["rae_mean_bag"] for r in per_K_results]
    best_sweep_i = int(np.argmin(sweep_rae))
    best_K = int(per_K_results[best_sweep_i]["K"])
    best_K_rae = float(per_K_results[best_sweep_i]["rae_mean_bag"])
    print(f"\n   scaffold-CV best K = {best_K}  (mean_bag RAE {best_K_rae:.4f})")
    print(f"   delta vs nb2103 K=28 scaffold-CV = "
          f"{best_K_rae - NB2103_K28_SCAFFOLD_REF:+.4f}")
    print(f"   deploy gate (ref - margin) = {DEPLOY_GATE:.4f}  "
          f"-> {'PASS' if best_K_rae < DEPLOY_GATE else 'FAIL'}")

    # ---- Global verdict ----
    if best_K_rae < DEPLOY_GATE:
        global_verdict = (
            f"SCAFFOLD_CV_BEATS_REF_AT_K={best_K}_DEPLOY_GATE_PASSED"
        )
    elif abs(best_K_rae - NB2103_K28_SCAFFOLD_REF) < DECISION_MARGIN:
        global_verdict = f"SCAFFOLD_CV_FLAT_VS_REF_BEST_K={best_K}"
    else:
        global_verdict = (
            f"SCAFFOLD_CV_DOES_NOT_BEAT_REF_BEST_K={best_K}_RAE={best_K_rae:.4f}"
        )
    print(f"\n   global verdict   = {global_verdict}")

    # ---- Deploy CSV (only if best K beats 0.5027) ----
    deploy_path = None
    if best_K_rae < DEPLOY_GATE:
        print("\n" + "-" * 78)
        print(f"DEPLOY: best K={best_K} beats gate {DEPLOY_GATE:.4f} -> build CSV")
        print("-" * 78)
        # Build feature matrix on FULL 513 test, fit residual model on ALL
        # 253 unblind rows, predict on 513 - 253 = 260 blinded rows; anchor
        # + residual prediction gives the deploy pEC50.
        # NOTE: te_chemprop_aux already has anchor predictions on all 513.
        X_ap_513 = X_ap_te[:, top_ap_bit_idx].astype(np.float32)
        X_maccs_513 = X_maccs_te[:, top_maccs_bit_idx].astype(np.float32)
        X_mord_513 = X_mord_te[:, top_mord_col_idx].astype(np.float32)
        X_emb_513 = X_emb_te[:, top_embed_col_idx].astype(np.float32)
        X_av_513 = X_av_te[:, top_avalon_bit_idx].astype(np.float32)
        X_513_full = np.concatenate(
            [
                X_ap_513, X_maccs_513, X_mord_513, X_emb_513, X_av_513,
                pred_chembl_pec50.reshape(-1, 1),
                mean_sim.reshape(-1, 1),
            ],
            axis=1,
        ).astype(np.float32)

        topK_idx_best = full_rank_order[:best_K].astype(np.int32)
        X_513_topK = X_513_full[:, topK_idx_best]
        X_unb_topK = X_unb[:, topK_idx_best]

        # 5-seed deploy bag, refit on all 253 unblind residuals
        seed_preds_513 = np.zeros((len(RESID_SEEDS), n_test), dtype=np.float64)
        for i, s in enumerate(RESID_SEEDS):
            mdl = lgb.LGBMRegressor(**_lgbm_params(s))
            mdl.fit(X_unb_topK, residual)
            seed_preds_513[i] = mdl.predict(X_513_topK)
        resid_pred_513 = seed_preds_513.mean(axis=0)
        final_513 = te_anchor_513 + resid_pred_513
        print(f"   deploy residual on 513: mean={resid_pred_513.mean():+.4f}  "
              f"std={resid_pred_513.std():.4f}")
        print(f"   deploy final pred: mean={final_513.mean():.4f}  "
              f"std={final_513.std():.4f}")

        # Write CSV with Molecule Name, SMILES, pEC50
        name_col = "name" if "name" in te.columns else "Molecule Name"
        names = te[name_col].astype(str).tolist()
        smis_full = (te["smiles"].astype(str).tolist()
                     if "smiles" in te.columns
                     else te["SMILES"].astype(str).tolist())
        sub_df = pd.DataFrame({
            "Molecule Name": names,
            "SMILES": smis_full,
            "pEC50": final_513.astype(np.float64),
        })
        SUB_DIR = Path(__file__).resolve().parents[1] / "submissions"
        SUB_DIR.mkdir(parents=True, exist_ok=True)
        deploy_path = SUB_DIR / f"{TAG}_scaffold_K{best_K}.csv"
        sub_df.to_csv(deploy_path, index=False)
        print(f"   [deploy] wrote {deploy_path}  ({len(sub_df)} rows)")
    else:
        print(f"\n[no-deploy] best_K_rae {best_K_rae:.4f} >= gate "
              f"{DEPLOY_GATE:.4f} -- skip deploy CSV")

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "method": "lgbm_mse_scaffold_KFold_K_sweep_15_20_28_35_50_on_117col",
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "cv_protocol": ("scaffold_kfold_indices(Murcko, n=5, shuffle=True, "
                        "seed=RESID_SEED) per seed"),
        "scaffold_kind": "RDKit Murcko skeleton",
        "data_source": ("nb2063 cached SHAP importance + same 117-col "
                        "5-way K-tuned matrix as nb2063/nb2103"),
        "model_family": "LightGBM",
        "lgbm_objective": "regression",
        "lgbm_max_depth": 4,
        "lgbm_num_leaves": 15,
        "lgbm_n_estimators": 300,
        "lgbm_learning_rate": 0.03,
        "lgbm_min_child_samples": 5,
        "lgbm_reg_lambda": 2.0,
        "K_grid": K_GRID,
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
        "scaffold_stats": {
            "n_unb": int(n_unb),
            "n_unb_with_scaffold": int(n_unb - n_none_sc),
            "n_unique_scaffolds": int(n_unique_sc),
            "n_singleton_scaffolds": int(n_singleton_sc),
            "largest_scaffold_size": int(largest_sc),
        },
        "n_chembl_pool": int(len(pool)),
        "n_unb": n_unb,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "nb2103_K28_scaffold_ref": NB2103_K28_SCAFFOLD_REF,
        "nb2103_K28_random_ref": NB2103_K28_RANDOM_REF,
        "deploy_gate": DEPLOY_GATE,
        "per_K_records": per_K_results,
        "best_K_sweep": best_K,
        "best_K_sweep_rae_mean_bag": best_K_rae,
        "delta_best_vs_nb2103_K28_scaffold": best_K_rae - NB2103_K28_SCAFFOLD_REF,
        "deploy_pass": bool(best_K_rae < DEPLOY_GATE),
        "deploy_csv_path": str(deploy_path) if deploy_path else None,
        "verdict": global_verdict,
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
        "K_grid", "feat_dim_full",
        "rae_anchor_chemprop_aux",
        "nb2103_K28_scaffold_ref",
        "best_K_sweep", "best_K_sweep_rae_mean_bag",
        "delta_best_vs_nb2103_K28_scaffold",
        "deploy_pass", "deploy_csv_path",
        "verdict", "pre_unblind_clean",
    ):
        print(f"  {k}: {res.get(k)}")
    print("\n==== PER-K TABLE (SCAFFOLD-CV) ====")
    for r in res["per_K_records"]:
        print(f"  K={r['K']:>3d}  mean_bag={r['rae_mean_bag']:.4f}  "
              f"median_bag={r['rae_median_bag']:.4f}  "
              f"per_seed_mean={r['rae_per_seed_mean']:.4f}  "
              f"std={r['rae_per_seed_std']:.4f}  "
              f"d_vs_ref={r['delta_mean_bag_vs_nb2103_K28_scaffold']:+.4f}  "
              f"{r['verdict']}")
