"""nb1111 -- Focal-loss gradient boosting on SHAP top-28 features.

HYPOTHESIS:
    nb2103 K=28 LGBM (max_depth=4, num_leaves=15, n_estimators=300, lr=0.03,
    min_child_samples=5, reg_lambda=2.0) on the 117-col 5-way K-tuned matrix
    delivered cross-fit mean-bag RAE 0.4737 / median-bag RAE 0.4698 on the 253
    unblind, using anchor=chemprop_aux residual. The standard MSE objective
    weights all 253 residuals equally; focal weighting upweights hard examples
    (large |r|) and downweights easy ones (small |r|), which may improve fit on
    the F2 over-prediction tail (cf. PXR phase-1 post-mortem: 50 worst nb503
    errors are 90% novel scaffolds with 2-sided variance compression).

    Focal sample weight (analogous to focal loss in classification):
        w_i = (1 - exp(-r_i^2 / (2 * sigma^2)))^gamma
    where r_i = residual_unb[i] = y_unb[i] - anchor_unb[i] and
          sigma = std(residual_train_fold) (from train portion only, per fold).
    gamma=0 reduces to uniform; gamma>0 progressively focuses on hard samples.

PROTOCOL:
    1. Reuse SHAP top-28 indices from nb2063 SHAP importance on 117-col matrix.
       Build X_unb_K (253, 28) and X_te_K (513, 28) by reproducing nb1051's
       117-col 5-way K-tuned matrix construction (AtomPair / MACCS / Mordred /
       ChempropEmbed / Avalon + ChEMBL kNN), then slice top-28.
    2. Anchor = chemprop_aux te[unb_idx]; residual = y_unb - anchor on 253.
    3. For each gamma in {0.5, 1.0, 2.0}:
         For each seed in {0, 1, 7, 42, 137}:
           scaffold 5-fold CV per seed.
           For each fold:
             sigma_fold = residual[tr].std()
             w_tr = (1 - exp(-r_tr^2 / (2 * sigma_fold^2)))^gamma
             fit LGBMRegressor(objective='regression', max_depth=4,
                               num_leaves=15, n_estimators=300, lr=0.03,
                               min_child_samples=5, reg_lambda=2.0, seed=s)
             with sample_weight=w_tr.
             oof[va] = mdl.predict(X[va]); pred_corr[va] = anchor[va] + oof[va].
           mean-bag, median-bag OOF across seeds; pooled RAE.
    4. Compare best (mean_bag, median_bag) vs nb2103 K=28 at margin=0.003.
    5. Gate: test-pred std must exceed 0.85 to avoid variance collapse
       (the failure mode from nb710 pinball / nb711 tail-weight LGBM).
    6. Fresh-seed verification: best gamma is re-run with NEW seeds
       {3, 8, 21, 55, 100} to confirm the improvement is not seed-fluke.
    7. If best passes both gates and verification, refit on ALL 253 (5 seeds,
       no folds), aggregate via best['agg'], predict 513, write
       submissions/nb1111_focal_K28_gamma{X}.csv.

ANCHORS:
    chemprop_aux in_RAE                       = 0.6216
    nb2103 K=28 LGBM mean_bag RAE             = 0.4737
    nb2103 K=28 LGBM median_bag RAE           = 0.4698
    decision_margin                           = 0.003
    variance_collapse_floor (te_pred std)     = 0.85

REFS:
    nb710 (pinball alpha=0.3) RAE 0.7559    -- HURTS (variance-collapse failure)
    nb711 (tail-weight LGBM)   RAE 0.7020    -- HURTS (variance-collapse failure)
    nb2103 K=28 mean_bag       RAE 0.4737    -- THE BASELINE TO BEAT
    nb2103 K=28 median_bag     RAE 0.4698    -- THE BASELINE TO BEAT

Outputs:
    scripts/nb1111_focal_loss.py
    data/processed/nb1111_summary.json
    data/processed/nb1111_mean_bag_oof_g{gamma}.npy
    data/processed/nb1111_median_bag_oof_g{gamma}.npy
    submissions/nb1111_focal_K28_g{gamma}.csv  (only if best beats + verifies)
    data/processed/te_nb1111_focal_K28_g{gamma}.npy  (only if deploy)
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

TAG = "nb1111"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

K = 28
RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]
FRESH_SEEDS = [3, 8, 21, 55, 100]
GAMMA_GRID = [0.5, 1.0, 2.0]

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
VARIANCE_COLLAPSE_FLOOR = 0.85   # te-pred std floor (cf. nb710/nb711 failure)


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
    """Same union as nb2063 / nb1051."""
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
    """LGBM(MSE) -- identical to nb2103 K=28."""
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


def _focal_weights(residual: np.ndarray, sigma: float, gamma: float) -> np.ndarray:
    """Focal sample weight:
        w_i = (1 - exp(-r_i^2 / (2*sigma^2)))^gamma
    Upweights large |r| (hard examples). gamma=0 collapses to uniform.
    Returns float64 array (LGBM-friendly).
    """
    if sigma <= 0:
        return np.ones_like(residual, dtype=np.float64)
    z = residual ** 2 / (2.0 * sigma ** 2)
    w = (1.0 - np.exp(-z)) ** gamma
    # Guard: if all weights are ~0 (gamma huge + small residuals), fall back uniform.
    w = np.asarray(w, dtype=np.float64)
    s = w.sum()
    if not np.isfinite(s) or s < 1e-9:
        return np.ones_like(residual, dtype=np.float64)
    return w


def _residual_scaffold_cross_fit_one_seed(
    X: np.ndarray,
    residual: np.ndarray,
    scaffolds: list,
    seed: int,
    gamma: float,
) -> tuple[np.ndarray, list[float]]:
    """5-fold scaffold cross-fit on `residual`.

    Per fold: sigma = residual[tr].std(); weights = focal(r_tr, sigma, gamma).
    Returns (oof_residual, per_fold_sigma_list).
    """
    n = len(residual)
    splits = scaffold_kfold_indices(scaffolds, n_splits=RESID_FOLDS,
                                    shuffle=True, seed=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    sigmas: list[float] = []
    for fold, (tr_idx, va_idx) in enumerate(splits):
        sigma_f = float(residual[tr_idx].std())
        sigmas.append(sigma_f)
        w_tr = _focal_weights(residual[tr_idx], sigma_f, gamma)
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_idx], residual[tr_idx], sample_weight=w_tr)
        oof[va_idx] = mdl.predict(X[va_idx])
    return oof, sigmas


def _load_mordred_test(n_test_expected: int) -> np.ndarray:
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(
            f"Mordred cache missing -- run nb1030 first ({mte_p})"
        )
    X_te_m = np.load(mte_p).astype(np.float32)
    if X_te_m.shape[0] != n_test_expected:
        raise ValueError(
            f"Mordred test shape mismatch: {X_te_m.shape} vs {n_test_expected}"
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


def _build_117col_matrix(test_smiles, mol_names, unb_idx, n_test):
    """Reproduce nb1051/nb2103 117-col 5-way K-tuned matrix construction."""
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
    std_test_smiles = [Chem.MolToSmiles(m) if m is not None else "" for m in test_mols]
    fp_test = morgan_fp_batch(std_test_smiles)
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )
    pred_chembl_te = pred_chembl_pec50.astype(np.float32)
    mean_sim_te = mean_sim.astype(np.float32)

    X_te_full = np.concatenate([
        X_ap_te_top, X_maccs_te_top, X_mord_te_top, X_emb_te_top, X_av_te_top,
        pred_chembl_te.reshape(-1, 1), mean_sim_te.reshape(-1, 1),
    ], axis=1).astype(np.float32)
    X_unb_full = X_te_full[unb_idx].astype(np.float32)
    return X_unb_full, X_te_full, len(pool), test_mols


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Focal-loss LGBM (gamma sweep) on SHAP top-{K} feats")
    print(f"          anchor={ANCHOR}  seeds={RESID_SEEDS}  folds={RESID_FOLDS}")
    print(f"          ref: nb2103 K=28 mean_bag {NB2103_K28_MEAN_BAG_REF:.4f}  "
          f"median_bag {NB2103_K28_MEDIAN_BAG_REF:.4f}  margin {DECISION_MARGIN}")
    print(f"          gamma grid = {GAMMA_GRID}  fresh seeds = {FRESH_SEEDS}")
    print(f"          variance-collapse floor (te-std) = "
          f"{VARIANCE_COLLAPSE_FLOOR}")
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
    full_rank_order = np.argsort(-shap_imp_full117).astype(np.int32)

    # ---- Load anchor + truth + names ----
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
    sigma_total = float(residual_unb.std())
    print(f"[resid] mean={residual_unb.mean():+.4f}  std={sigma_total:.4f}")

    # ---- Validate cached LGBM K=28 OOF gives ref RAE ----
    rae_lgbm_K28 = float(rae(y_unb, lgbm_K28_oof))
    print(f"[check] cached nb2103 K=28 OOF RAE = {rae_lgbm_K28:.4f}  "
          f"(ref {nb2103_k28_mean_bag:.4f})")

    # ---- Build the 117-col matrix + slice top-28 ----
    print("\n" + "-" * 78)
    print("BUILD 117-col 5-way K-tuned matrix (same as nb2103)")
    print("-" * 78)
    X_unb_full, X_te_full, n_chembl_pool, test_mols = _build_117col_matrix(
        test_smiles, mol_names, unb_idx, n_test
    )
    feat_dim_full = X_unb_full.shape[1]
    if feat_dim_full != shap_imp_full117.shape[0]:
        raise ValueError(
            f"feat_dim {feat_dim_full} != SHAP imp len {shap_imp_full117.shape[0]}"
        )
    print(f"   COMBINED full matrix: X_unb_full={X_unb_full.shape}  "
          f"X_te_full={X_te_full.shape}")

    topK_idx = full_rank_order[:K].astype(np.int32)
    X_unb_K = X_unb_full[:, topK_idx].astype(np.float32)
    X_te_K = X_te_full[:, topK_idx].astype(np.float32)
    print(f"   SHAP top-{K} matrices: X_unb_K={X_unb_K.shape}  "
          f"X_te_K={X_te_K.shape}")

    # ---- Build scaffold list for the 253 unblind ----
    unb_smiles = [test_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    unique_scaf = len(set(s for s in unb_scaffolds if s is not None))
    print(f"   scaffolds on 253 unblind: {unique_scaf} unique "
          f"(of {n_unb} compounds)")

    # ---- Gamma sweep (5-seed bag, scaffold 5-fold cross-fit) ----
    print("\n" + "=" * 78)
    print(f"FOCAL GAMMA SWEEP  ({len(GAMMA_GRID)} gammas x "
          f"{len(RESID_SEEDS)} seeds x {RESID_FOLDS} scaffold folds)")
    print("=" * 78)
    gamma_results: dict[float, dict] = {}
    per_gamma_mean_bag_oof: dict[float, np.ndarray] = {}
    per_gamma_median_bag_oof: dict[float, np.ndarray] = {}
    for gamma in GAMMA_GRID:
        print(f"\n--- gamma = {gamma:.1f} ---")
        per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
        per_seed_rae: list[float] = []
        per_seed_records = []
        per_seed_w_stats = []
        for i, s in enumerate(RESID_SEEDS):
            ts = time.time()
            resid_oof_s, sigmas_s = _residual_scaffold_cross_fit_one_seed(
                X_unb_K, residual_unb, unb_scaffolds, s, gamma
            )
            # Diagnostic: capture sample-weight stats from a fold sample
            w_demo = _focal_weights(residual_unb, float(np.mean(sigmas_s)), gamma)
            w_min = float(w_demo.min())
            w_max = float(w_demo.max())
            w_mean = float(w_demo.mean())

            pred_corr_s = anchor_unb + resid_oof_s
            per_seed_corrected[i] = pred_corr_s
            rae_s = float(rae(y_unb, pred_corr_s))
            per_seed_rae.append(rae_s)
            wall = time.time() - ts
            per_seed_records.append({
                "seed": int(s),
                "rae_corrected": rae_s,
                "delta_vs_chemprop_aux": rae_s - rae_anchor,
                "delta_vs_nb2103_mean": rae_s - nb2103_k28_mean_bag,
                "resid_oof_std": float(resid_oof_s.std()),
                "resid_oof_mean": float(resid_oof_s.mean()),
                "fold_sigmas": [round(x, 4) for x in sigmas_s],
                "w_min": w_min, "w_max": w_max, "w_mean": w_mean,
                "wall_sec": round(wall, 2),
            })
            per_seed_w_stats.append((w_min, w_max, w_mean))
            print(f"   g={gamma:.1f}  seed={s:3d}: rae={rae_s:.4f}  "
                  f"(d_vs_anchor={rae_s - rae_anchor:+.4f}  "
                  f"d_vs_K28={rae_s - nb2103_k28_mean_bag:+.4f})  "
                  f"w[min/mean/max]=[{w_min:.3f}/{w_mean:.3f}/{w_max:.3f}]  "
                  f"wall={wall:.1f}s")

        mean_bag_oof = per_seed_corrected.mean(axis=0)
        median_bag_oof = np.median(per_seed_corrected, axis=0)
        rae_mean_bag = float(rae(y_unb, mean_bag_oof))
        rae_median_bag = float(rae(y_unb, median_bag_oof))
        per_seed_rae_arr = np.array(per_seed_rae)
        delta_mean_vs_nb2103 = rae_mean_bag - nb2103_k28_mean_bag
        delta_median_vs_nb2103 = rae_median_bag - nb2103_k28_median_bag
        beats_mean = rae_mean_bag < nb2103_k28_mean_bag - DECISION_MARGIN
        beats_median = rae_median_bag < nb2103_k28_median_bag - DECISION_MARGIN
        flat_mean = abs(delta_mean_vs_nb2103) < DECISION_MARGIN
        if beats_mean and beats_median:
            verdict_g = "BEATS_NB2103_K28_BOTH"
        elif beats_mean:
            verdict_g = "BEATS_NB2103_K28_MEAN_BAG"
        elif beats_median:
            verdict_g = "BEATS_NB2103_K28_MEDIAN_BAG"
        elif flat_mean:
            verdict_g = "FLAT_VS_NB2103_K28"
        else:
            verdict_g = "HURTS_VS_NB2103_K28"
        print(f"   g={gamma:.1f}  POOLED  mean_bag={rae_mean_bag:.4f}  "
              f"median_bag={rae_median_bag:.4f}  "
              f"d_mean={delta_mean_vs_nb2103:+.4f}  "
              f"d_median={delta_median_vs_nb2103:+.4f}  "
              f"verdict={verdict_g}")

        # Save OOFs per gamma
        gtag = f"{gamma:.1f}".replace(".", "_")
        np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof_g{gtag}.npy",
                mean_bag_oof.astype(np.float32))
        np.save(DATA_PROCESSED / f"{TAG}_median_bag_oof_g{gtag}.npy",
                median_bag_oof.astype(np.float32))

        gamma_results[gamma] = {
            "gamma": float(gamma),
            "per_seed_rae": per_seed_rae,
            "per_seed_records": per_seed_records,
            "rae_per_seed_mean": float(per_seed_rae_arr.mean()),
            "rae_per_seed_std": float(per_seed_rae_arr.std()),
            "rae_per_seed_min": float(per_seed_rae_arr.min()),
            "rae_per_seed_max": float(per_seed_rae_arr.max()),
            "rae_mean_bag": rae_mean_bag,
            "rae_median_bag": rae_median_bag,
            "delta_mean_bag_vs_nb2103_K28": delta_mean_vs_nb2103,
            "delta_median_bag_vs_nb2103_K28": delta_median_vs_nb2103,
            "verdict": verdict_g,
            "beats_nb2103_mean": bool(beats_mean),
            "beats_nb2103_median": bool(beats_median),
            "mean_bag_oof_path": str(
                DATA_PROCESSED / f"{TAG}_mean_bag_oof_g{gtag}.npy"
            ),
            "median_bag_oof_path": str(
                DATA_PROCESSED / f"{TAG}_median_bag_oof_g{gtag}.npy"
            ),
        }
        per_gamma_mean_bag_oof[gamma] = mean_bag_oof.copy()
        per_gamma_median_bag_oof[gamma] = median_bag_oof.copy()

    # ---- Pick best candidate (across all gammas x {mean_bag, median_bag}) ----
    print("\n" + "=" * 78)
    print("OVERALL SELECTION (gamma sweep)")
    print("=" * 78)
    candidates: list[dict] = []
    for gamma, gr in gamma_results.items():
        candidates.append({
            "kind": f"focal_g{gamma:.1f}_mean_bag",
            "rae": gr["rae_mean_bag"],
            "gamma": float(gamma),
            "agg": "mean_bag",
        })
        candidates.append({
            "kind": f"focal_g{gamma:.1f}_median_bag",
            "rae": gr["rae_median_bag"],
            "gamma": float(gamma),
            "agg": "median_bag",
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

    # ---- Fresh-seed verification (if best passes initial threshold) ----
    passes_initial = beats_nb2103_mean or beats_nb2103_median
    fresh_seed_verification = None
    fresh_passes = False
    if passes_initial:
        print("\n" + "-" * 78)
        print(f"FRESH-SEED VERIFICATION  best={best['kind']}  "
              f"new_seeds={FRESH_SEEDS}")
        print("-" * 78)
        best_gamma = best["gamma"]
        best_agg = best["agg"]
        per_seed_corrected_fresh = np.zeros((len(FRESH_SEEDS), n_unb),
                                              dtype=np.float64)
        per_seed_rae_fresh: list[float] = []
        for i, s in enumerate(FRESH_SEEDS):
            ts = time.time()
            resid_oof_s, _ = _residual_scaffold_cross_fit_one_seed(
                X_unb_K, residual_unb, unb_scaffolds, s, best_gamma
            )
            pred_corr_s = anchor_unb + resid_oof_s
            per_seed_corrected_fresh[i] = pred_corr_s
            rae_s = float(rae(y_unb, pred_corr_s))
            per_seed_rae_fresh.append(rae_s)
            wall = time.time() - ts
            print(f"   fresh seed={s:3d}: rae={rae_s:.4f}  wall={wall:.1f}s")
        if best_agg == "median_bag":
            fresh_agg_oof = np.median(per_seed_corrected_fresh, axis=0)
        else:
            fresh_agg_oof = per_seed_corrected_fresh.mean(axis=0)
        fresh_rae = float(rae(y_unb, fresh_agg_oof))
        fresh_passes = (fresh_rae < nb2103_k28_mean_bag - DECISION_MARGIN
                        or fresh_rae < nb2103_k28_median_bag - DECISION_MARGIN)
        print(f"\n   fresh pooled  ({best_agg})  RAE = {fresh_rae:.4f}  "
              f"(orig {best_rae:.4f}, delta {fresh_rae - best_rae:+.4f})")
        print(f"   fresh-seed verification PASSES: {fresh_passes}")
        fresh_seed_verification = {
            "gamma": float(best_gamma),
            "agg": best_agg,
            "fresh_seeds": FRESH_SEEDS,
            "per_seed_rae": per_seed_rae_fresh,
            "fresh_pooled_rae": fresh_rae,
            "orig_best_rae": best_rae,
            "delta_vs_orig": fresh_rae - best_rae,
            "passes": bool(fresh_passes),
        }

    # ---- Deploy gate: must pass initial AND fresh-seed AND variance floor ----
    print("\n" + "=" * 78)
    print("DEPLOY GATE")
    print("=" * 78)
    if not passes_initial:
        global_verdict = (
            "FLAT_VS_NB2103_K28" if any(
                abs(c["rae"] - nb2103_k28_mean_bag) < DECISION_MARGIN
                or abs(c["rae"] - nb2103_k28_median_bag) < DECISION_MARGIN
                for c in candidates
            ) else "HURTS_VS_NB2103_K28"
        )
        do_deploy = False
        reason = "no candidate beats nb2103 K=28 by margin"
    elif not fresh_passes:
        global_verdict = "INITIAL_BEAT_BUT_FAILS_FRESH_SEED_VERIFICATION"
        do_deploy = False
        reason = "fresh-seed verification failed"
    else:
        global_verdict = "BEATS_NB2103_K28_PASSES_FRESH_SEED"
        do_deploy = True
        reason = "passes both gates -- check variance floor at deploy"
    print(f"   global verdict   = {global_verdict}")
    print(f"   do_deploy        = {do_deploy}  ({reason})")

    # ---- DEPLOY (only if both gates pass + variance floor pass) ----
    submission_csv = None
    te_artifact = None
    te_in_RAE_unb = None
    te_pred_std = None
    deploy_info = None
    variance_floor_pass = None
    if do_deploy:
        print("\n" + "=" * 78)
        print(f"DEPLOY: refit best recipe ({best['kind']}) on ALL 253 unblind, "
              f"predict 513")
        print("=" * 78)
        best_gamma = best["gamma"]
        best_agg = best["agg"]
        # Refit on full 253 with focal weight; sigma = std(all residual_unb).
        sigma_full = float(residual_unb.std())
        w_full = _focal_weights(residual_unb, sigma_full, best_gamma)
        print(f"   sigma_full = {sigma_full:.4f}  "
              f"w[min/mean/max] = "
              f"[{w_full.min():.3f}/{w_full.mean():.3f}/{w_full.max():.3f}]")
        per_seed_te = np.zeros((len(RESID_SEEDS), n_test), dtype=np.float64)
        for i, s in enumerate(RESID_SEEDS):
            ts = time.time()
            mdl = lgb.LGBMRegressor(**_lgbm_params(s))
            mdl.fit(X_unb_K, residual_unb, sample_weight=w_full)
            resid_513 = mdl.predict(X_te_K)
            per_seed_te[i] = resid_513
            print(f"   refit seed={s:3d}  "
                  f"resid_513_mean={resid_513.mean():+.4f}  "
                  f"resid_513_std={resid_513.std():.4f}  "
                  f"wall={time.time() - ts:.1f}s")
        if best_agg == "median_bag":
            resid_513_agg = np.median(per_seed_te, axis=0)
        else:
            resid_513_agg = per_seed_te.mean(axis=0)
        te_final_513 = te_anchor_513 + resid_513_agg
        te_pred_std = float(te_final_513.std())
        print(f"\n   te pred  mean={te_final_513.mean():.4f}  "
              f"std={te_pred_std:.4f}  "
              f"min={te_final_513.min():.4f}  max={te_final_513.max():.4f}")

        # ---- Variance-collapse gate ----
        variance_floor_pass = te_pred_std > VARIANCE_COLLAPSE_FLOOR
        print(f"   variance-collapse gate: te_std={te_pred_std:.4f}  "
              f"floor={VARIANCE_COLLAPSE_FLOOR}  "
              f"PASS={variance_floor_pass}")
        if not variance_floor_pass:
            print(f"\n   ABORT DEPLOY -- variance collapse detected.")
            do_deploy = False
            global_verdict = "BEATS_BUT_FAILS_VARIANCE_FLOOR"
            deploy_info = {
                "gamma": float(best_gamma),
                "agg": best_agg,
                "te_pred_std": te_pred_std,
                "floor": VARIANCE_COLLAPSE_FLOOR,
                "aborted_reason": "variance collapse below floor",
            }
        else:
            te_in_RAE_unb = float(rae(y_unb, te_final_513[unb_idx]))
            print(f"   in-sample RAE on unb_idx = {te_in_RAE_unb:.4f}  "
                  f"(cross-fit ref {best_rae:.4f})")

            gtag = f"{best_gamma:.1f}".replace(".", "_")
            df_sub = pd.DataFrame({
                "SMILES": test_smiles,
                "Molecule Name": mol_names,
                "pEC50": te_final_513.astype(np.float32),
            })
            if len(df_sub) != 513:
                raise ValueError(f"submission rows {len(df_sub)} != 513")
            sub_path = SUBMISSIONS_DIR / f"{TAG}_focal_K28_g{gtag}.csv"
            df_sub.to_csv(sub_path, index=False)
            print(f"\n[save] submission CSV: {sub_path}  ({len(df_sub)} rows)")
            submission_csv = str(sub_path)

            te_path = DATA_PROCESSED / f"te_{TAG}_focal_K28_g{gtag}.npy"
            np.save(te_path, te_final_513.astype(np.float32))
            print(f"[save] te artifact:  {te_path}")
            te_artifact = str(te_path)

            deploy_info = {
                "kind": "focal_lgbm",
                "gamma": float(best_gamma),
                "agg": best_agg,
                "n_seeds": len(RESID_SEEDS),
                "te_pred_std": te_pred_std,
                "variance_floor": VARIANCE_COLLAPSE_FLOOR,
                "variance_floor_pass": True,
            }

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "method": ("focal_loss_lgbm_K28_gamma_sweep_on_117col_5way_K_tuned"),
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "data_source": ("nb2063 SHAP top-28 indices of the 117-col 5-way "
                        "K-tuned matrix (AtomPair / MACCS / Mordred / "
                        "ChempropEmbed / Avalon + ChEMBL kNN)"),
        "K": K,
        "feat_dim": int(K),
        "model_family": "LightGBM",
        "lgbm_objective": "regression",
        "lgbm_max_depth": 4,
        "lgbm_num_leaves": 15,
        "lgbm_n_estimators": 300,
        "lgbm_learning_rate": 0.03,
        "lgbm_min_child_samples": 5,
        "lgbm_reg_lambda": 2.0,
        "focal_weight_formula": "w_i = (1 - exp(-r_i^2 / (2*sigma^2)))^gamma",
        "sigma_source": "per-fold residual std (train portion only)",
        "gamma_grid": GAMMA_GRID,
        "resid_seeds": RESID_SEEDS,
        "fresh_seeds": FRESH_SEEDS,
        "resid_folds": RESID_FOLDS,
        "split_type": "scaffold_5fold",
        "n_chembl_pool": int(n_chembl_pool),
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean_unb": float(residual_unb.mean()),
        "residual_std_unb": float(residual_unb.std()),
        "nb2103_K28_mean_bag_ref": nb2103_k28_mean_bag,
        "nb2103_K28_median_bag_ref": nb2103_k28_median_bag,
        "nb2103_K28_oof_self_RAE_check": rae_lgbm_K28,
        "decision_margin": DECISION_MARGIN,
        "variance_collapse_floor": VARIANCE_COLLAPSE_FLOOR,
        "gamma_results": {str(k): v for k, v in gamma_results.items()},
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
        "passes_initial_beat": bool(passes_initial),
        "fresh_seed_verification": fresh_seed_verification,
        "fresh_seed_passes": bool(fresh_passes),
        "te_pred_std": te_pred_std,
        "variance_floor_pass": variance_floor_pass,
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
        "passes_initial_beat", "fresh_seed_passes",
        "te_pred_std", "variance_floor_pass",
        "global_verdict", "do_deploy",
        "submission_csv", "te_in_RAE_unb",
        "pre_unblind_clean",
    ):
        print(f"  {k}: {res.get(k)}")
    print("\n==== GAMMA TABLE ====")
    for gk, gr in res["gamma_results"].items():
        print(f"  gamma={gk:>5s}  mean_bag={gr['rae_mean_bag']:.4f}  "
              f"median_bag={gr['rae_median_bag']:.4f}  "
              f"per_seed_mean={gr['rae_per_seed_mean']:.4f}  "
              f"std={gr['rae_per_seed_std']:.4f}  "
              f"verdict={gr['verdict']}")
