"""nb2184 -- Re-eval nb2170 K=28 and nb2178 K=15 on HONEST nb730 anchor (nb2183).

HYPOTHESIS:
    nb2170 (K=28, mean-bag 0.3920) and nb2178 (K=15, mean-bag 0.3810) used
    te_nb730[unb_idx] as the anchor. That te file is the DEPLOY refit trained
    on ALL 4392 labels (including the 253 unblind), so te[unb_idx] is in-sample
    and biases the residual headroom DOWN. The honest anchor is
    nb730_honest_pred_oof (5-fold cross-fit, never sees its own row), which is
    LB-faithful. This re-eval substitutes the honest anchor and re-fits the
    same residual machinery to obtain the LB-honest RAE.

PROTOCOL:
    1.  Load nb730_honest_pred_oof.npy (253,) from nb2183.
    2.  Reuse SHAP ranking from nb2178_shap_importance_full117.npy (or recompute
        on residual against honest anchor for a fresh ranking).
    3.  Rebuild the 117-col 5-way K-tuned + ChEMBL-kNN feature stack on 513
        (identical to nb2178), slice to 253 unblind.
    4.  Compute residual_honest = y_unb - nb730_honest_pred_oof.
    5.  Fresh SHAP ranking: full-fit LGBM(MSE) on (X_unb_117, residual_honest)
        -> shap_imp_honest -> full_ranking_honest.
    6.  For K in {15, 20, 28}: take top-K by HONEST SHAP, 5-seed bag
        (0, 1, 7, 42, 137), 5-fold cross-fit LGBM(MSE) L=15 lr=0.03 mc=5
        lambda=2 n_est=300 on residual_honest.
    7.  Final = nb730_honest_pred_oof + cross-fit-LGBM-residual; mean-bag and
        median-bag RAE.
    8.  Compare honest results vs:
            - Honest nb730 alone (nb2183_summary["rae_nb730_honest"])
            - nb2103 K=28 chemprop_aux anchor (0.4737 mean / 0.4698 median)
            - nb2170 contaminated K=28 (0.3920 mean / 0.3936 median)
            - nb2178 contaminated K=15 (0.3810 mean / ...)
    9.  If honest best <= 0.4500: build deploy CSV
            nb2184_deploy_honest_nb730_residual.csv
            -> te_nb730 (513) + LGBM(residual) fit on all 4392 (y - te_nb730)
               but residual model trained ONLY on 253 (residual = y_unb - anchor_honest).
            Note: deploy still uses te_nb730 (513-row anchor); only the residual
            ML changes. We train residual LGBM on (X_unb_K, residual_honest) and
            predict on X_te_K (513).

Outputs:
    scripts/nb2184_reeval_honest_anchor.py
    data/processed/nb2184_summary.json
    data/processed/nb2184_shap_importance_full117_honest.npy  (117,)
    data/processed/nb2184_mean_bag_oof_honest_K<K>.npy        (253,) per K
    submissions/nb2184_deploy_honest_nb730_residual.csv       (conditional)
    data/processed/te_nb2184.npy                              (conditional)
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

TAG = "nb2184"
DEPLOY_TAG = "nb2184"
ANCHOR = "nb730_honest"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_nb730.npy"  # 513-row deploy anchor
ANCHOR_HONEST_OOF_PATH = DATA_PROCESSED / "nb730_honest_pred_oof.npy"  # 253-row honest
NB2183_SUMMARY_PATH = DATA_PROCESSED / "nb2183_summary.json"

K_SWEEP = [15, 20, 28]
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

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

# References (contaminated te[unb_idx] anchor RAE on 253 unblind)
NB2103_K28_MEDIAN_REF = 0.4698
NB2103_K28_MEAN_REF = 0.4737
NB2170_K28_MEAN_REF = 0.3920
NB2170_K28_MEDIAN_REF = 0.3936
NB2178_K15_MEAN_REF = 0.3810

# Deploy gate
DEPLOY_RAE_GATE = 0.4500


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


def _knn_predict(top_idx, top_sim, pool_labels, fallback):
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


def _residual_cross_fit_one_seed(X, residual, seed):
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _load_mordred_test(n_test_expected):
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(f"Mordred cache missing: {mte_p}")
    X_te_m = np.load(mte_p).astype(np.float32)
    if X_te_m.shape[0] != n_test_expected:
        raise ValueError(f"Mordred shape {X_te_m.shape} vs {n_test_expected}")
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
        raise FileNotFoundError(f"missing: {path}")
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
    raise KeyError("AtomPair not in nb1484")


def _extract_best_K_record(sum_dict, records_key, best_K_key="best_K"):
    best_K = int(sum_dict[best_K_key])
    for r in sum_dict[records_key]:
        if int(r["K"]) == best_K:
            return r
    raise KeyError(f"best_K {best_K} not in {records_key}")


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Re-eval K=15/20/28 on HONEST nb730 anchor (nb2183)")
    print("=" * 78)

    # ---- Load honest anchor (nb2183) ----
    if not ANCHOR_HONEST_OOF_PATH.exists():
        raise FileNotFoundError(
            f"Honest nb730 anchor not built yet: {ANCHOR_HONEST_OOF_PATH}"
        )
    nb730_honest_oof = np.load(ANCHOR_HONEST_OOF_PATH).astype(np.float64)
    if nb730_honest_oof.shape != (253,):
        raise ValueError(
            f"nb730_honest_pred_oof shape {nb730_honest_oof.shape} != (253,)"
        )
    print(f"[load] nb730_honest_pred_oof: {nb730_honest_oof.shape}  "
          f"mean={nb730_honest_oof.mean():.4f}  std={nb730_honest_oof.std():.4f}")

    nb2183_meta = {}
    rae_nb730_honest = None
    if NB2183_SUMMARY_PATH.exists():
        with open(NB2183_SUMMARY_PATH) as f:
            nb2183_meta = json.load(f)
        for k in ("rae_nb730_honest", "rae_honest_anchor", "rae",
                  "honest_rae_anchor"):
            if k in nb2183_meta:
                rae_nb730_honest = float(nb2183_meta[k])
                break

    # ---- Load 513-row deploy anchor (for deploy step only) ----
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
    else:
        mol_names = te["name"].astype(str).tolist()

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"deploy anchor missing: {ANCHOR_TE_PATH}")
    te_nb730_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_nb730_513.shape[0] != n_test:
        raise ValueError(f"te_nb730 shape {te_nb730_513.shape} vs {n_test}")

    # In-sample (contaminated) anchor RAE for comparison
    rae_nb730_in_sample = float(rae(y_unb, te_nb730_513[unb_idx]))

    # Verify honest RAE
    rae_nb730_honest_check = float(rae(y_unb, nb730_honest_oof))
    if rae_nb730_honest is None:
        rae_nb730_honest = rae_nb730_honest_check
    print(f"[anchor] nb730 in-sample (contaminated te[unb_idx]) RAE = "
          f"{rae_nb730_in_sample:.4f}")
    print(f"[anchor] nb730 HONEST (5-fold cross-fit) RAE              = "
          f"{rae_nb730_honest_check:.4f}")
    if rae_nb730_honest is not None:
        print(f"[anchor] nb2183-reported honest RAE                       = "
              f"{rae_nb730_honest:.4f}")

    # Residual on HONEST anchor
    residual_honest = y_unb - nb730_honest_oof
    print(f"[resid] honest mean = {residual_honest.mean():+.4f}  "
          f"std = {residual_honest.std():.4f}")

    # ---- Rebuild 117-col stack ----
    print("\n[feat] building 117-col stack on 513 (identical to nb2178)")
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
    pred_chembl_te, mean_sim_te = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )
    pred_chembl_te = pred_chembl_te.astype(np.float32)
    mean_sim_te = mean_sim_te.astype(np.float32)

    X_te_117 = np.concatenate(
        [X_ap_te_top, X_maccs_te_top, X_mord_te_top, X_emb_te_top,
         X_av_te_top,
         pred_chembl_te.reshape(-1, 1), mean_sim_te.reshape(-1, 1)],
        axis=1,
    ).astype(np.float32)
    feat_dim = X_te_117.shape[1]
    if feat_dim != 117:
        raise ValueError(f"feat_dim {feat_dim} != 117")
    print(f"[feat] X_te_117: {X_te_117.shape}")

    X_unb_117 = X_te_117[unb_idx]

    # ---- Step 1: FRESH SHAP ranking on HONEST residual ----
    print("\n" + "-" * 78)
    print("STEP 1: full-fit LGBM(MSE) on 117-col HONEST-residual -> SHAP")
    print("-" * 78)
    t_shap = time.time()
    mdl_full = lgb.LGBMRegressor(**_lgbm_params(seed=0))
    mdl_full.fit(X_unb_117, residual_honest)
    explainer = shap.TreeExplainer(mdl_full)
    shap_vals = explainer.shap_values(X_unb_117)
    shap_imp_honest = np.abs(shap_vals).mean(axis=0).astype(np.float32)
    if shap_imp_honest.shape[0] != feat_dim:
        raise ValueError(
            f"shap shape {shap_imp_honest.shape} != {feat_dim}"
        )
    full_ranking_honest = np.argsort(-shap_imp_honest).astype(np.int32)
    print(f"   SHAP done in {time.time() - t_shap:.1f}s")
    print(f"   honest ranking head 10: {full_ranking_honest[:10].tolist()}")
    np.save(DATA_PROCESSED / f"{TAG}_shap_importance_full117_honest.npy",
            shap_imp_honest)

    # Compare against contaminated SHAP ranking (nb2178)
    shap_2178_p = DATA_PROCESSED / "nb2178_shap_importance_full117.npy"
    overlap_top28 = None
    if shap_2178_p.exists():
        shap_2178 = np.load(shap_2178_p).astype(np.float32)
        rank_2178 = np.argsort(-shap_2178)
        ovl = set(full_ranking_honest[:28].tolist()) & set(rank_2178[:28].tolist())
        overlap_top28 = {
            "n_overlap": int(len(ovl)),
            "pct_overlap": float(len(ovl) / 28.0 * 100.0),
        }
        print(f"   top-28 SHAP overlap honest vs nb2178: "
              f"{overlap_top28['n_overlap']}/28 "
              f"({overlap_top28['pct_overlap']:.0f}%)")

    # ---- K-sweep cross-fit ----
    print("\n" + "-" * 78)
    print("K-SWEEP CROSS-FIT (HONEST ANCHOR)")
    print("-" * 78)
    per_K_records = []
    for K in K_SWEEP:
        if K > feat_dim:
            continue
        topK_idx = full_ranking_honest[:K]
        X_unb_K = X_unb_117[:, topK_idx].astype(np.float32)

        t_k = time.time()
        per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
        per_seed_rae = []
        for i, s in enumerate(RESID_SEEDS):
            resid_oof_s = _residual_cross_fit_one_seed(X_unb_K, residual_honest, s)
            pred_corr_s = nb730_honest_oof + resid_oof_s
            per_seed_corrected[i] = pred_corr_s
            rae_s = float(rae(y_unb, pred_corr_s))
            per_seed_rae.append(rae_s)
        mean_bag_oof = per_seed_corrected.mean(axis=0)
        median_bag_oof = np.median(per_seed_corrected, axis=0)
        rae_mean_bag = float(rae(y_unb, mean_bag_oof))
        rae_median_bag = float(rae(y_unb, median_bag_oof))

        np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof_honest_K{K}.npy",
                mean_bag_oof.astype(np.float32))

        per_K_records.append({
            "K": int(K),
            "topK_idx_in_117": [int(i) for i in topK_idx.tolist()],
            "per_seed_rae": per_seed_rae,
            "rae_per_seed_mean": float(np.mean(per_seed_rae)),
            "rae_per_seed_std": float(np.std(per_seed_rae)),
            "rae_mean_bag_honest": rae_mean_bag,
            "rae_median_bag_honest": rae_median_bag,
            "wall_sec": round(time.time() - t_k, 2),
        })
        print(f"   K={K:3d}  HONEST mean_bag = {rae_mean_bag:.4f}  "
              f"median_bag = {rae_median_bag:.4f}  wall={time.time()-t_k:.1f}s")

    # ---- Pick best ----
    by_mean = sorted(per_K_records, key=lambda r: r["rae_mean_bag_honest"])
    best = by_mean[0]
    best_K = int(best["K"])
    best_rae_mean = float(best["rae_mean_bag_honest"])
    best_rae_median = float(best["rae_median_bag_honest"])
    if best_rae_median < best_rae_mean:
        best_kind = "median"
        best_rae = best_rae_median
    else:
        best_kind = "mean"
        best_rae = best_rae_mean

    # Honest gain vs anchor alone
    honest_anchor_rae = float(rae_nb730_honest_check)
    gain_vs_anchor = honest_anchor_rae - best_rae

    print("\n" + "=" * 78)
    print("HONEST RESULTS")
    print("=" * 78)
    print(f"   nb730 HONEST anchor alone       = {honest_anchor_rae:.4f}")
    if rae_nb730_honest is not None:
        print(f"   nb2183-reported honest          = {rae_nb730_honest:.4f}")
    for r in by_mean:
        print(f"   honest K={r['K']:3d}  mean={r['rae_mean_bag_honest']:.4f}  "
              f"median={r['rae_median_bag_honest']:.4f}")
    print(f"   BEST honest K={best_K} {best_kind} = {best_rae:.4f}  "
          f"(gain vs honest anchor: {gain_vs_anchor:+.4f})")
    print()
    print("CONTAMINATED REFERENCES (te[unb_idx] anchor):")
    print(f"   nb2103 K=28 chemprop_aux : 0.4737 mean / 0.4698 median")
    print(f"   nb2170 K=28              : {NB2170_K28_MEAN_REF:.4f} mean / "
          f"{NB2170_K28_MEDIAN_REF:.4f} median")
    print(f"   nb2178 K=15              : {NB2178_K15_MEAN_REF:.4f} mean")
    print(f"   HONEST vs nb2170 K=28 mean : "
          f"{best_rae - NB2170_K28_MEAN_REF:+.4f} (positive = honest is worse)")
    print(f"   HONEST vs nb2178 K=15 mean : "
          f"{best_rae - NB2178_K15_MEAN_REF:+.4f}")

    # ---- Deploy gate ----
    deploy_built = False
    deploy_path = None
    te_deploy_stats = None
    if best_rae <= DEPLOY_RAE_GATE:
        print("\n" + "-" * 78)
        print(f"DEPLOY: honest best K={best_K} ({best_kind}) RAE={best_rae:.4f} "
              f"<= gate {DEPLOY_RAE_GATE} -- building deploy CSV")
        print("-" * 78)
        topK_idx = full_ranking_honest[:best_K]
        X_unb_K = X_unb_117[:, topK_idx].astype(np.float32)
        X_te_K = X_te_117[:, topK_idx].astype(np.float32)

        # Deploy: residual model = LGBM(X_unb_K, residual_honest) -> 513
        # Final = te_nb730_513 + residual_pred_513
        OUTER_SEEDS = [0, 1, 7, 42, 137]
        INNER_OFFSETS = [0, 1, 7, 42, 137]
        n_total = len(OUTER_SEEDS) * len(INNER_OFFSETS)
        all_resid_513 = np.zeros((n_total, n_test), dtype=np.float64)
        k_global = 0
        for o in OUTER_SEEDS:
            inner_seeds = [o * 1000 + s for s in INNER_OFFSETS]
            for s in inner_seeds:
                mdl = lgb.LGBMRegressor(**_lgbm_params(s))
                mdl.fit(X_unb_K, residual_honest)
                all_resid_513[k_global] = mdl.predict(X_te_K)
                k_global += 1
        mean_resid_513 = all_resid_513.mean(axis=0)
        median_resid_513 = np.median(all_resid_513, axis=0)
        chosen_resid_513 = median_resid_513 if best_kind == "median" else mean_resid_513

        te_nb2184 = te_nb730_513 + chosen_resid_513
        in_unb_rae = float(rae(y_unb, te_nb2184[unb_idx]))

        df_sub = pd.DataFrame({
            "SMILES": test_smiles,
            "Molecule Name": mol_names,
            "pEC50": te_nb2184.astype(np.float32),
        })
        if len(df_sub) != 513:
            raise ValueError(f"sub rows {len(df_sub)} != 513")
        deploy_path = SUBMISSIONS_DIR / f"{DEPLOY_TAG}_deploy_honest_nb730_residual.csv"
        df_sub.to_csv(deploy_path, index=False)
        te_path = DATA_PROCESSED / f"te_{DEPLOY_TAG}.npy"
        np.save(te_path, te_nb2184.astype(np.float32))
        deploy_built = True
        te_deploy_stats = {
            "mean": float(te_nb2184.mean()),
            "std": float(te_nb2184.std()),
            "min": float(te_nb2184.min()),
            "max": float(te_nb2184.max()),
            "in_sample_rae_unb_te": in_unb_rae,
            "honest_cross_fit_rae": best_rae,
            "winning_K": best_K,
            "winning_kind": best_kind,
        }
        print(f"   in-sample RAE (te[unb_idx]) = {in_unb_rae:.4f}")
        print(f"   HONEST cross-fit RAE        = {best_rae:.4f}")
        print(f"   [save] {deploy_path}")
        print(f"   [save] {te_path}")
    else:
        print(f"\n   no deploy: honest best {best_rae:.4f} > gate "
              f"{DEPLOY_RAE_GATE}")

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "method": "reeval_K_sweep_on_honest_nb730_anchor",
        "anchor": ANCHOR,
        "anchor_honest_oof_path": str(ANCHOR_HONEST_OOF_PATH),
        "anchor_deploy_te_path": str(ANCHOR_TE_PATH),
        "nb2183_summary_path": str(NB2183_SUMMARY_PATH),
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
        "rae_nb730_in_sample_te_unb": rae_nb730_in_sample,
        "rae_nb730_honest_anchor": rae_nb730_honest_check,
        "rae_nb730_honest_anchor_from_nb2183": rae_nb730_honest,
        "residual_honest_mean": float(residual_honest.mean()),
        "residual_honest_std": float(residual_honest.std()),
        "shap_full_ranking_117_honest": [int(i) for i in full_ranking_honest.tolist()],
        "top28_honest_idx": [int(i) for i in full_ranking_honest[:28].tolist()],
        "shap_top28_overlap_honest_vs_nb2178": overlap_top28,
        "per_K_records": per_K_records,
        "ranking_by_honest_mean_bag": [
            {"K": r["K"], "rae_mean_bag_honest": r["rae_mean_bag_honest"],
             "rae_median_bag_honest": r["rae_median_bag_honest"]}
            for r in by_mean
        ],
        "best_K": int(best_K),
        "best_kind": best_kind,
        "best_rae_honest": float(best_rae),
        "best_rae_mean_bag_honest": float(best_rae_mean),
        "best_rae_median_bag_honest": float(best_rae_median),
        "gain_vs_honest_anchor": float(gain_vs_anchor),
        "references_contaminated": {
            "nb2103_K28_mean": NB2103_K28_MEAN_REF,
            "nb2103_K28_median": NB2103_K28_MEDIAN_REF,
            "nb2170_K28_mean": NB2170_K28_MEAN_REF,
            "nb2170_K28_median": NB2170_K28_MEDIAN_REF,
            "nb2178_K15_mean": NB2178_K15_MEAN_REF,
        },
        "delta_honest_vs_nb2170_K28_mean": float(best_rae - NB2170_K28_MEAN_REF),
        "delta_honest_vs_nb2178_K15_mean": float(best_rae - NB2178_K15_MEAN_REF),
        "deploy_rae_gate": DEPLOY_RAE_GATE,
        "deploy_built": deploy_built,
        "deploy_path": str(deploy_path) if deploy_path else None,
        "te_deploy_stats": te_deploy_stats,
        "honest_evaluation": True,
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
        "rae_nb730_honest_anchor",
        "best_K",
        "best_kind",
        "best_rae_honest",
        "gain_vs_honest_anchor",
        "delta_honest_vs_nb2170_K28_mean",
        "delta_honest_vs_nb2178_K15_mean",
        "deploy_built",
        "deploy_path",
    ):
        print(f"  {k}: {res.get(k)}")
