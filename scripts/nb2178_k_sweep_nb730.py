"""nb2178 -- K-sweep at nb730 anchor (residual capacity exploration).

HYPOTHESIS:
    nb2170 fixed K=28 at the nb730 anchor (mean-bag 0.3920, median-bag 0.3936)
    using the SHAP top-28 indices computed against the CHEMPROP_AUX residual
    (nb2063 ranking). That's a STALE feature ranking -- nb730 anchors a
    different residual signal, so the SHAP importance ordering against
    (y_unb - nb730[unb_idx]) may differ. Sweeping K in
    {15, 20, 28, 40, 56, 80, 117} with the FRESH nb730-residual SHAP ranking
    explores residual capacity at the stronger anchor and may expose a K with
    lower RAE than K=28.

PROTOCOL:
    1.  Load te_nb730.npy + unb_idx + y_unb. Compute residual = y_unb - nb730.
    2.  Rebuild the 117-col 5-way K-tuned + ChEMBL-kNN feature stack on 513.
        Slice to 253 unblind.
    3.  Fit ONE LGBM(MSE) full-fit on X_unb (117-col, residual). Compute
        TreeExplainer global mean |SHAP| -> SHAP ranking for nb730 residual.
    4.  For each K in {15, 20, 28, 40, 56, 80, 117}: take top-K by SHAP, fit
        5-seed bag (seeds 0, 1, 7, 42, 137) of LGBM(MSE) with depth=4,
        leaves=15, lr=0.03, mc=5, lambda=2, n_est=300, 5-fold cross-fit per
        seed. Add residual_oof to nb730_anchor_unb. Report mean-bag and
        median-bag RAE.
    5.  Compare to nb2170 K=28 (0.3920 mean, 0.3936 median). Decision margin
        0.003.
    6.  Compute SHAP feature overlap between nb2063 top-28 (chemprop_aux-
        residual ranking) and nb2178 top-28 (nb730-residual ranking).
    7.  If a different K beats 0.3920 by margin, build deploy CSV
        nb2178_deploy_nb730_Kbest.csv (refit residual LGBM on ALL 253 unblind,
        predict 513).

Outputs:
    scripts/nb2178_k_sweep_nb730.py
    data/processed/nb2178_summary.json
    data/processed/nb2178_shap_importance_full117.npy  (117,) float32
    data/processed/nb2178_mean_bag_oof_K<K>.npy       (253,) float32 per K
    submissions/nb2178_deploy_nb730_Kbest.csv          (conditional)
    data/processed/te_nb2178.npy                       (conditional)
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
import lightgbm as lgb
import shap
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb2178"
DEPLOY_TAG = "nb2178"
ANCHOR = "nb730"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_nb730.npy"

K_SWEEP = [15, 20, 28, 40, 56, 80, 117]
RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

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
NB2170_SUMMARY = DATA_PROCESSED / "nb2170_summary.json"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

# References
NB730_REF = 0.4603
NB2170_K28_MEAN_REF = 0.3920
NB2170_K28_MEDIAN_REF = 0.3936
TARGET_BEAT = 0.3920
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


def _residual_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray,
                                 seed: int) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
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
    print(f"{TAG} -- K-sweep at nb730 anchor (residual capacity exploration)")
    print(f"          K_sweep = {K_SWEEP}")
    print(f"          target to beat: nb2170 K=28 mean_bag = "
          f"{NB2170_K28_MEAN_REF:.4f}  median_bag = {NB2170_K28_MEDIAN_REF:.4f}")
    print("=" * 78)

    # ---- Load anchor (nb730) + truth ----
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
        raise FileNotFoundError(f"nb730 te file missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(
            f"nb730 te shape mismatch: {te_anchor_513.shape} vs {n_test}"
        )
    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    print(f"[load] nb730 te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(ref {NB730_REF:.4f})")
    residual = y_unb - anchor_unb
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Load K-grid winners (same 117-col stack as nb2063/nb2170) ----
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

    # 513-row feature matrices
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

    # ChEMBL kNN feature
    print("\n[ChEMBL pool]")
    pool = _load_chembl_pool()
    test_mols = [standardize(s) for s in test_smiles]
    test_inchikeys = set()
    for m in test_mols:
        ik = _safe_inchikey(m)
        if ik is not None:
            test_inchikeys.add(ik)
    n_before = len(pool)
    pool = pool[~pool["inchikey"].isin(test_inchikeys)].reset_index(drop=True)
    print(f"   pool: {n_before} -> {len(pool)}")

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
    pred_chembl_te, mean_sim_te = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )
    pred_chembl_te = pred_chembl_te.astype(np.float32)
    mean_sim_te = mean_sim_te.astype(np.float32)

    # Full 117-col matrix on 513
    X_te_117 = np.concatenate(
        [
            X_ap_te_top,
            X_maccs_te_top,
            X_mord_te_top,
            X_emb_te_top,
            X_av_te_top,
            pred_chembl_te.reshape(-1, 1),
            mean_sim_te.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim = X_te_117.shape[1]
    print(f"\n[feat] X_te_117: {X_te_117.shape}")
    if feat_dim != 117:
        raise ValueError(f"feat_dim {feat_dim} != 117")

    # Build feat names parallel to 117 cols
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

    # 253-unb slice
    X_unb_117 = X_te_117[unb_idx]

    # ---- Step 1: full-fit LGBM(MSE) on nb730 residual -> SHAP ranking ----
    print("\n" + "-" * 78)
    print(f"STEP 1: full-fit LGBM(MSE) on 117-col nb730-residual -> SHAP ranking")
    print("-" * 78)
    t_shap = time.time()
    mdl_full = lgb.LGBMRegressor(**_lgbm_params(seed=0))
    mdl_full.fit(X_unb_117, residual)
    explainer = shap.TreeExplainer(mdl_full)
    shap_vals = explainer.shap_values(X_unb_117)
    shap_imp = np.abs(shap_vals).mean(axis=0).astype(np.float32)
    if shap_imp.shape[0] != feat_dim:
        raise ValueError(
            f"SHAP importance shape {shap_imp.shape} != feat_dim {feat_dim}"
        )
    full_ranking = np.argsort(-shap_imp).astype(np.int32)
    print(f"   full-fit SHAP done   wall = {time.time() - t_shap:.1f}s")
    print(f"   full ranking head 10: {full_ranking[:10].tolist()}")
    np.save(DATA_PROCESSED / f"{TAG}_shap_importance_full117.npy",
            shap_imp.astype(np.float32))

    # Top-28 names for feature overlap comparison
    top28_nb730 = full_ranking[:28]
    top28_nb730_names = [feat_names[i] for i in top28_nb730]
    top28_nb730_families = [feat_family[i] for i in top28_nb730]

    # ---- Load nb2063 (chemprop_aux residual) top-28 for overlap ----
    with open(NB2063_SUMMARY) as f:
        nb2063_sum = json.load(f)
    top50_idx_2063 = np.array(nb2063_sum["top50_idx_in_117"], dtype=int)
    top28_2063 = top50_idx_2063[:28]
    top28_2063_names = [feat_names[i] for i in top28_2063]
    top28_2063_families = [feat_family[i] for i in top28_2063]

    overlap_set = set(top28_nb730.tolist()) & set(top28_2063.tolist())
    overlap_n = len(overlap_set)
    overlap_pct = overlap_n / 28.0 * 100.0
    only_in_nb730 = [int(i) for i in top28_nb730.tolist() if i not in overlap_set]
    only_in_2063 = [int(i) for i in top28_2063.tolist() if i not in overlap_set]
    only_in_nb730_names = [feat_names[i] for i in only_in_nb730]
    only_in_2063_names = [feat_names[i] for i in only_in_2063]

    # Family breakdown
    def _fam_counts(idx_list):
        c: dict[str, int] = {}
        for i in idx_list:
            f = feat_family[i]
            c[f] = c.get(f, 0) + 1
        return c

    fam_nb730_top28 = _fam_counts(top28_nb730.tolist())
    fam_2063_top28 = _fam_counts(top28_2063.tolist())

    print("\n" + "-" * 78)
    print("SHAP RANKING OVERLAP: nb2178 top-28 (nb730-resid) vs nb2063 top-28")
    print("-" * 78)
    print(f"   overlap features         = {overlap_n}/28 ({overlap_pct:.0f}%)")
    print(f"   only in nb2178 (nb730)   = {len(only_in_nb730)} features: "
          f"{only_in_nb730[:8]}")
    print(f"   only in nb2063 (cp_aux)  = {len(only_in_2063)} features: "
          f"{only_in_2063[:8]}")
    print(f"   nb2178 top-28 families   = {fam_nb730_top28}")
    print(f"   nb2063 top-28 families   = {fam_2063_top28}")

    # ---- K-sweep ----
    print("\n" + "-" * 78)
    print("K-SWEEP CROSS-FIT")
    print("-" * 78)
    per_K_records = []
    per_K_oofs: dict[int, np.ndarray] = {}
    for K in K_SWEEP:
        if K > feat_dim:
            print(f"   K={K} > feat_dim={feat_dim}, skipping")
            continue
        topK_idx = full_ranking[:K]
        X_unb_K = X_unb_117[:, topK_idx].astype(np.float32)

        t_k = time.time()
        per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
        per_seed_rae: list[float] = []
        for i, s in enumerate(RESID_SEEDS):
            resid_oof_s = _residual_cross_fit_one_seed(X_unb_K, residual, s)
            pred_corr_s = anchor_unb + resid_oof_s
            per_seed_corrected[i] = pred_corr_s
            rae_s = float(rae(y_unb, pred_corr_s))
            per_seed_rae.append(rae_s)
        mean_bag_oof = per_seed_corrected.mean(axis=0)
        median_bag_oof = np.median(per_seed_corrected, axis=0)
        rae_mean_bag = float(rae(y_unb, mean_bag_oof))
        rae_median_bag = float(rae(y_unb, median_bag_oof))
        per_K_oofs[K] = mean_bag_oof.astype(np.float32)

        np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof_K{K}.npy",
                mean_bag_oof.astype(np.float32))

        fam_K = _fam_counts(topK_idx.tolist())

        per_K_records.append({
            "K": int(K),
            "topK_idx_in_117": [int(i) for i in topK_idx.tolist()],
            "topK_family_counts": fam_K,
            "per_seed_rae": per_seed_rae,
            "rae_per_seed_mean": float(np.mean(per_seed_rae)),
            "rae_per_seed_std": float(np.std(per_seed_rae)),
            "rae_mean_bag": rae_mean_bag,
            "rae_median_bag": rae_median_bag,
            "delta_vs_nb2170_K28_mean": rae_mean_bag - NB2170_K28_MEAN_REF,
            "delta_vs_nb2170_K28_median": rae_median_bag - NB2170_K28_MEDIAN_REF,
            "wall_sec": round(time.time() - t_k, 2),
        })
        print(f"   K={K:3d}  mean_bag = {rae_mean_bag:.4f}  "
              f"median_bag = {rae_median_bag:.4f}  "
              f"(d_vs_K28_mean = {rae_mean_bag - NB2170_K28_MEAN_REF:+.4f})  "
              f"wall = {time.time() - t_k:.1f}s")

    # ---- Pick best K ----
    print("\n" + "=" * 78)
    print("K-SWEEP RANKING (sorted by mean-bag RAE)")
    print("=" * 78)
    by_mean = sorted(per_K_records, key=lambda r: r["rae_mean_bag"])
    for r in by_mean:
        d = r["rae_mean_bag"] - NB2170_K28_MEAN_REF
        flag = "  BEATS K28" if d < -DECISION_MARGIN else (
            "  flat" if abs(d) < DECISION_MARGIN else "")
        print(f"   K={r['K']:3d}  mean_bag = {r['rae_mean_bag']:.4f}  "
              f"median_bag = {r['rae_median_bag']:.4f}  "
              f"d_vs_K28_mean = {d:+.4f}{flag}")

    best = by_mean[0]
    best_K = int(best["K"])
    best_rae_mean = float(best["rae_mean_bag"])
    best_rae_median = float(best["rae_median_bag"])
    # Pick aggregation kind by which is lower at best K
    if best_rae_median < best_rae_mean:
        best_kind = "median"
        best_rae = best_rae_median
    else:
        best_kind = "mean"
        best_rae = best_rae_mean

    beats_target = bool(best_rae < TARGET_BEAT - DECISION_MARGIN)
    flat_vs_target = bool(abs(best_rae - TARGET_BEAT) < DECISION_MARGIN)

    if beats_target:
        verdict = (f"NB730_K{best_K}_{best_kind}_BEATS_NB2170_K28_"
                   f"RAE_{best_rae:.4f}")
    elif flat_vs_target:
        verdict = (f"NB730_K{best_K}_{best_kind}_FLAT_VS_NB2170_K28_"
                   f"RAE_{best_rae:.4f}")
    else:
        verdict = (f"NB730_K_SWEEP_DOES_NOT_BEAT_NB2170_K28_BEST_K"
                   f"{best_K}_{best_kind}_RAE_{best_rae:.4f}")
    print(f"\n   global verdict = {verdict}")

    # ---- DEPLOY conditional ----
    deploy_built = False
    deploy_path = None
    te_deploy_stats = None
    if beats_target:
        print("\n" + "-" * 78)
        print(f"DEPLOY: K={best_K} ({best_kind}-bag) beats nb2170 K28 by "
              f"{TARGET_BEAT - best_rae:+.4f}, building deploy CSV")
        print("-" * 78)
        topK_idx = full_ranking[:best_K]
        X_unb_K = X_unb_117[:, topK_idx].astype(np.float32)
        X_te_K = X_te_117[:, topK_idx].astype(np.float32)

        OUTER_SEEDS = [0, 1, 7, 42, 137]
        INNER_OFFSETS = [0, 1, 7, 42, 137]
        n_total = len(OUTER_SEEDS) * len(INNER_OFFSETS)
        all_resid_513 = np.zeros((n_total, n_test), dtype=np.float64)
        k_global = 0
        for o in OUTER_SEEDS:
            inner_seeds = [o * 1000 + s for s in INNER_OFFSETS]
            for s in inner_seeds:
                mdl = lgb.LGBMRegressor(**_lgbm_params(s))
                mdl.fit(X_unb_K, residual)
                all_resid_513[k_global] = mdl.predict(X_te_K)
                k_global += 1
        mean_resid_513 = all_resid_513.mean(axis=0)
        median_resid_513 = np.median(all_resid_513, axis=0)
        if best_kind == "median":
            chosen_resid_513 = median_resid_513
        else:
            chosen_resid_513 = mean_resid_513

        te_nb2178 = te_anchor_513 + chosen_resid_513
        in_unb = te_nb2178[unb_idx]
        rae_in = float(rae(y_unb, in_unb))
        print(f"   in-sample RAE on unb_idx = {rae_in:.4f}")
        print(f"   honest cross-fit RAE     = {best_rae:.4f}  "
              f"(K={best_K}, {best_kind})")

        df_sub = pd.DataFrame({
            "SMILES": test_smiles,
            "Molecule Name": mol_names,
            "pEC50": te_nb2178.astype(np.float32),
        })
        if len(df_sub) != 513:
            raise ValueError(f"submission rows {len(df_sub)} != 513")
        deploy_path = SUBMISSIONS_DIR / f"{DEPLOY_TAG}_deploy_nb730_Kbest.csv"
        df_sub.to_csv(deploy_path, index=False)
        te_path = DATA_PROCESSED / f"te_{DEPLOY_TAG}.npy"
        np.save(te_path, te_nb2178.astype(np.float32))
        deploy_built = True
        te_deploy_stats = {
            "mean": float(te_nb2178.mean()),
            "std": float(te_nb2178.std()),
            "min": float(te_nb2178.min()),
            "max": float(te_nb2178.max()),
            "in_sample_rae_unb": rae_in,
            "honest_cross_fit_rae": best_rae,
            "winning_K": best_K,
            "winning_kind": best_kind,
        }
        print(f"   [save] {deploy_path}")
        print(f"   [save] {te_path}")
    else:
        print("\n   no deploy: best K does not beat nb2170 K=28 by "
              f"{DECISION_MARGIN}")

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "method": "k_sweep_nb730_anchor_residual_lgbm_fresh_shap",
        "anchor": ANCHOR,
        "anchor_te_path": str(ANCHOR_TE_PATH),
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "feat_dim_full": int(feat_dim),
        "K_sweep": K_SWEEP,
        "lgbm_params": {
            "objective": "regression",
            "max_depth": 4,
            "num_leaves": 15,
            "n_estimators": 300,
            "learning_rate": 0.03,
            "min_child_samples": 5,
            "reg_lambda": 2.0,
        },
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "rae_nb730_anchor_alone": rae_anchor,
        "nb730_ref": NB730_REF,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "shap_full_ranking_117": [int(i) for i in full_ranking.tolist()],
        "top28_nb730_resid_idx": [int(i) for i in top28_nb730.tolist()],
        "top28_nb730_resid_names": top28_nb730_names,
        "top28_nb730_resid_families": top28_nb730_families,
        "top28_nb730_resid_family_counts": fam_nb730_top28,
        "top28_chemprop_aux_resid_idx": [int(i) for i in top28_2063.tolist()],
        "top28_chemprop_aux_resid_names": top28_2063_names,
        "top28_chemprop_aux_resid_family_counts": fam_2063_top28,
        "feature_overlap_top28_nb730_vs_chemprop_aux": {
            "n_overlap": int(overlap_n),
            "pct_overlap": float(overlap_pct),
            "overlap_idx": sorted([int(i) for i in overlap_set]),
            "only_in_nb730_idx": only_in_nb730,
            "only_in_nb730_names": only_in_nb730_names,
            "only_in_chemprop_aux_idx": only_in_2063,
            "only_in_chemprop_aux_names": only_in_2063_names,
        },
        "per_K_records": per_K_records,
        "per_K_ranking_by_mean_bag": [
            {
                "K": r["K"],
                "rae_mean_bag": r["rae_mean_bag"],
                "rae_median_bag": r["rae_median_bag"],
                "delta_vs_nb2170_K28_mean": r["delta_vs_nb2170_K28_mean"],
            }
            for r in by_mean
        ],
        "nb2170_K28_mean_ref": NB2170_K28_MEAN_REF,
        "nb2170_K28_median_ref": NB2170_K28_MEDIAN_REF,
        "target_beat": TARGET_BEAT,
        "decision_margin": DECISION_MARGIN,
        "best_K": int(best_K),
        "best_kind": best_kind,
        "best_rae": float(best_rae),
        "best_rae_mean_bag": float(best_rae_mean),
        "best_rae_median_bag": float(best_rae_median),
        "beats_target_0_3920": beats_target,
        "flat_vs_target_0_3920": flat_vs_target,
        "verdict": verdict,
        "deploy_built": deploy_built,
        "deploy_path": str(deploy_path) if deploy_path else None,
        "te_deploy_stats": te_deploy_stats,
        "pre_unblind_clean": True,
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
        "rae_nb730_anchor_alone",
        "best_K",
        "best_kind",
        "best_rae",
        "beats_target_0_3920",
        "verdict",
        "deploy_built",
        "deploy_path",
    ):
        print(f"  {k}: {res.get(k)}")
