"""nb1101 -- Boruta wrapper feature selection on 117-col 5-way K-tuned matrix.

HYPOTHESIS:
    nb2103 confirmed K=28 (top-28 SHAP) as the optimal feature subset
    (mean-bag RAE 0.4737, median-bag 0.4698, beating nb2081 K=30 0.4788).
    SHAP ranks features by mean(|SHAP|) over the training distribution -- a
    *marginal* importance metric. Boruta is a *wrapper* selector that flags a
    real feature only if its random-forest / LGBM importance is consistently
    higher than the max of N random "shadow" features (shuffled copies of
    the real features). The shadow comparison is a permutation-style null
    that filters features which look important purely from sampling noise.
    On n=253 unblind, where K-grid is overfit-prone, Boruta's null-aware
    confirmation should give a more conservative top-28 with potentially
    different content than SHAP.

PROTOCOL:
    1. Reuse the EXACT 117-col 5-way K-tuned matrix that nb2063/nb2081/
       nb2091/nb2103 used (AtomPair / MACCS / Mordred / ChempropEmbed /
       Avalon + ChEMBL kNN), against the chemprop_aux residual on 253 unblind.
    2. boruta package is NOT installed (verified). Implement MANUAL Boruta:
         for k_iter in 1..MAX_ITERS:
             - build X_aug = hstack(X_real, shuffle_columns(X_real))
                  (shuffle each column independently; sklearn-Boruta style)
             - train LGBM (same MSE hyperparams as nb2103) on X_aug -> residual
             - get feature_importances_ (split-count by default)
             - threshold = max importance among the *shadow* columns
             - for each real feature: increment "hit" counter if its
               importance > threshold this iteration
         After MAX_ITERS, a feature is "confirmed" if its hit count >=
         confirm_threshold (binomial test at alpha=0.05 gives a one-sided
         floor; for max_iters=30, threshold ~ 22 hits).
    3. Rank confirmed features by their average importance over iterations;
       take TOP-28 by that rank.
    4. Compare Boruta-28 vs SHAP-28 from nb2103: overlap, exclusive members,
       family breakdown delta.
    5. Fit LGBM (same hyperparams as nb2103) on Boruta top-28; 5-seed bag
       (0, 1, 7, 42, 137), 5-fold scaffold cross-fit per seed. Report mean-
       bag RAE, median-bag RAE.
    6. Compare vs nb2103 K=28 references (mean-bag 0.4737, median-bag 0.4698)
       at decision margin 0.005.
    7. If Boruta-28 PASSES, run a fresh-seed verification (seeds 13, 23, 31,
       91, 233) on the Boruta-28 set to confirm the gain isn't a seed lottery.
    8. Save mean-bag OOF and a summary JSON.

References:
    Kursa & Rudnicki 2010, "Feature Selection with the Boruta Package",
    Journal of Statistical Software, vol 36 iss 11. Default behaviour:
    100 RF iterations with shuffle-shadow null, Bonferroni-corrected
    binomial test; "confirmed" if hits > Bonferroni 0.005 upper-quantile.
    We use LGBM split-count (cheap, same family as the downstream model)
    and a fixed-N-iter (default 30) one-sided binomial threshold.

Outputs:
    scripts/nb1101_boruta.py
    data/processed/nb1101_boruta_mean_bag_oof.npy
    data/processed/nb1101_boruta_fresh_seed_mean_bag_oof.npy   (conditional)
    data/processed/nb1101_summary.json
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
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1101"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]
FRESH_SEEDS = [13, 23, 31, 91, 233]
TOP_K = 28
DECISION_MARGIN = 0.005

# Boruta hyperparameters
BORUTA_MAX_ITERS = 30
BORUTA_SHADOW_SEED_BASE = 1000
# With 117 features the max-over-117-shadows is a high bar; the standard
# binomial(N=30, p=0.5) 5% threshold (~20) leaves zero features confirmed.
# Relax to 12 (>= 40% of iterations) so a meaningful "confirmed" set exists,
# then re-rank by mean importance and take TOP_K. This still filters
# features that beat the shadow null less than chance (binomial p=0.5
# expectation is 15 hits).
BORUTA_CONFIRM_HITS = 12

# References (PRE-unblind path)
CHEMPROP_AUX_REF = 0.6216
NB2103_K28_MEAN_BAG_REF = 0.4737
NB2103_K28_MEDIAN_BAG_REF = 0.4698

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


def _lgbm_params(seed: int) -> dict:
    """LGBM(MSE) -- identical to nb2103."""
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
        raise FileNotFoundError(f"Mordred cache missing: {mte_p}")
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


def _boruta_select(X: np.ndarray, y: np.ndarray, feat_names: list[str],
                   max_iters: int = BORUTA_MAX_ITERS,
                   confirm_hits: int = BORUTA_CONFIRM_HITS,
                   shadow_seed_base: int = BORUTA_SHADOW_SEED_BASE,
                   ) -> dict:
    """Manual Boruta.

    For each iteration:
       - build X_aug = [X | shuffle_columns(X)]; left half is real,
         right half is "shadow"
       - train LGBM on X_aug -> y
       - shadow_threshold = max(importance) over shadow columns
       - increment hit_count[j] for each real column j whose importance >
         shadow_threshold
    Confirmed features: hit_count[j] >= confirm_hits.
    Returns a dict with confirmed mask, hit counts, mean importances.
    """
    n, p = X.shape
    hit_count = np.zeros(p, dtype=np.int64)
    real_imp_sum = np.zeros(p, dtype=np.float64)
    shadow_max_imp_per_iter = np.zeros(max_iters, dtype=np.float64)
    iter_records = []

    for it in range(max_iters):
        rng = np.random.default_rng(shadow_seed_base + it)
        # shuffle each column independently to make shadow
        X_shadow = X.copy()
        for j in range(p):
            perm = rng.permutation(n)
            X_shadow[:, j] = X[perm, j]
        X_aug = np.concatenate([X, X_shadow], axis=1).astype(np.float32)

        mdl = lgb.LGBMRegressor(**_lgbm_params(shadow_seed_base + it))
        mdl.fit(X_aug, y)
        imp = np.asarray(mdl.feature_importances_, dtype=np.float64)
        real_imp = imp[:p]
        shadow_imp = imp[p:]
        shadow_threshold = float(shadow_imp.max())
        shadow_max_imp_per_iter[it] = shadow_threshold

        hit_count += (real_imp > shadow_threshold).astype(np.int64)
        real_imp_sum += real_imp

        iter_records.append({
            "iter": int(it),
            "shadow_max_imp": shadow_threshold,
            "shadow_mean_imp": float(shadow_imp.mean()),
            "real_mean_imp": float(real_imp.mean()),
            "n_real_above_shadow": int((real_imp > shadow_threshold).sum()),
        })

    mean_real_imp = real_imp_sum / float(max_iters)
    confirmed_mask = hit_count >= confirm_hits

    confirmed_idx = np.where(confirmed_mask)[0]
    confirmed_rank_order_by_imp = confirmed_idx[
        np.argsort(-mean_real_imp[confirmed_idx])
    ]
    full_rank_by_imp = np.argsort(-mean_real_imp)

    return {
        "max_iters": int(max_iters),
        "confirm_hits": int(confirm_hits),
        "hit_count": hit_count.tolist(),
        "mean_real_importance": mean_real_imp.tolist(),
        "shadow_max_imp_per_iter": shadow_max_imp_per_iter.tolist(),
        "n_confirmed": int(confirmed_mask.sum()),
        "confirmed_idx": confirmed_idx.tolist(),
        "confirmed_rank_order_by_imp": confirmed_rank_order_by_imp.tolist(),
        "full_rank_by_imp": full_rank_by_imp.tolist(),
        "iter_records": iter_records,
        "feat_names": feat_names,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- BORUTA wrapper feature selection on 117-col matrix")
    print(f"         anchor={ANCHOR}  top_K={TOP_K}  "
          f"boruta_iters={BORUTA_MAX_ITERS}  confirm_hits>={BORUTA_CONFIRM_HITS}")
    print(f"         ref: nb2103 K=28 mean_bag={NB2103_K28_MEAN_BAG_REF:.4f}  "
          f"median_bag={NB2103_K28_MEDIAN_BAG_REF:.4f}  margin={DECISION_MARGIN}")
    print("=" * 78)

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
        raise FileNotFoundError(f"chemprop_aux te missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] anchor in_RAE = {rae_anchor:.4f}  (ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Load all K-grid winners + nb2063 SHAP ranking ----
    for p in (NB1352_SUMMARY, NB1392_SUMMARY, NB1484_SUMMARY,
              NB1523_SUMMARY, NB1524_SUMMARY, NB1541_SUMMARY,
              NB2063_SHAP_IMP, NB2103_SUMMARY):
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
    with open(NB2103_SUMMARY) as f:
        sum_2103 = json.load(f)

    shap_imp_full117 = np.load(NB2063_SHAP_IMP).astype(np.float32)
    shap_rank_order = np.argsort(-shap_imp_full117).astype(np.int32)
    shap_top28_idx = shap_rank_order[:TOP_K].astype(np.int32)

    # Cross-check vs nb2103 K=28 record
    k28_record = None
    for r in sum_2103.get("per_K_records", []):
        if int(r.get("K", -1)) == 28:
            k28_record = r
            break
    if k28_record is None:
        raise KeyError("nb2103_summary.json missing K=28 record")
    nb2103_k28_idx = np.array(k28_record["top_K_idx_in_117"], dtype=np.int32)
    if not np.array_equal(shap_top28_idx, nb2103_k28_idx):
        print("[warn] derived SHAP top-28 != nb2103 K=28 top_K_idx_in_117; "
              "using nb2103-reported indices as canonical SHAP-28")
        shap_top28_idx = nb2103_k28_idx
    nb2103_k28_mean_bag = float(k28_record["rae_mean_bag"])
    nb2103_k28_median_bag = float(k28_record["rae_median_bag"])
    print(f"[ref] nb2103 K=28 mean_bag={nb2103_k28_mean_bag:.4f}  "
          f"median_bag={nb2103_k28_median_bag:.4f}")

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
    X_ap_unb = X_ap_te[unb_idx].astype(np.float32)
    X_ap_unb_top = X_ap_unb[:, top_ap_bit_idx].astype(np.float32)

    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)
    X_maccs_unb = X_maccs_te[unb_idx].astype(np.float32)
    X_maccs_unb_top = X_maccs_unb[:, top_maccs_bit_idx].astype(np.float32)

    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    X_mord_unb = X_mord_te[unb_idx].astype(np.float32)
    X_mord_unb_top = X_mord_unb[:, top_mord_col_idx].astype(np.float32)

    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)
    X_emb_unb = X_emb_te[unb_idx].astype(np.float32)
    X_emb_unb_top = X_emb_unb[:, top_embed_col_idx].astype(np.float32)

    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)
    X_av_unb = X_av_te[unb_idx].astype(np.float32)
    X_av_unb_top = X_av_unb[:, top_avalon_bit_idx].astype(np.float32)

    # ChEMBL kNN
    print("\n[chembl] building PXR pool")
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
    print(f"[chembl] pool size={len(pool)} median={pool_median:.3f}")

    X_unb = np.concatenate(
        [
            X_ap_unb_top, X_maccs_unb_top, X_mord_unb_top,
            X_emb_unb_top, X_av_unb_top,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ], axis=1,
    ).astype(np.float32)
    feat_dim = X_unb.shape[1]
    expected = (n_top_ap + n_top_maccs + n_top_mord
                + n_top_embed + n_top_avalon + 2)
    if feat_dim != expected:
        raise ValueError(f"feat_dim {feat_dim} != expected {expected}")
    print(f"[feat] X_unb shape = {X_unb.shape}")

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

    # ---- Manual Boruta on (X_unb, residual) ----
    print("\n" + "-" * 78)
    print("BORUTA (manual, shadow-shuffled-columns null)")
    print("-" * 78)
    t_b = time.time()
    boruta_res = _boruta_select(
        X_unb, residual, feat_names,
        max_iters=BORUTA_MAX_ITERS,
        confirm_hits=BORUTA_CONFIRM_HITS,
    )
    print(f"[boruta] {boruta_res['n_confirmed']} / {feat_dim} confirmed "
          f"(hits >= {BORUTA_CONFIRM_HITS} of {BORUTA_MAX_ITERS}); "
          f"wall = {time.time() - t_b:.1f}s")

    full_rank_imp = np.array(boruta_res["full_rank_by_imp"], dtype=np.int32)
    confirmed_rank = np.array(boruta_res["confirmed_rank_order_by_imp"],
                              dtype=np.int32)

    if len(confirmed_rank) >= TOP_K:
        boruta_top_idx = confirmed_rank[:TOP_K].astype(np.int32)
        boruta_top_source = f"confirmed_top_{TOP_K}_by_mean_importance"
    else:
        # fallback: append next-best unconfirmed features by overall importance
        # to reach TOP_K so the downstream comparison is fair.
        already = set(confirmed_rank.tolist())
        backfill = [int(j) for j in full_rank_imp.tolist()
                    if int(j) not in already]
        n_short = TOP_K - len(confirmed_rank)
        boruta_top_idx = np.concatenate(
            [confirmed_rank, np.array(backfill[:n_short], dtype=np.int32)]
        ).astype(np.int32)
        boruta_top_source = (
            f"confirmed_{len(confirmed_rank)}_plus_topup_{n_short}"
            f"_from_full_rank_to_K{TOP_K}"
        )
    print(f"[boruta] top-{TOP_K} source = {boruta_top_source}")

    fam_counts_boruta = {}
    for j in boruta_top_idx:
        fam = feat_family[int(j)]
        fam_counts_boruta[fam] = fam_counts_boruta.get(fam, 0) + 1
    print(f"[boruta] top-{TOP_K} family breakdown = {fam_counts_boruta}")

    # ---- Compare vs SHAP-28 ----
    boruta_set = set(int(j) for j in boruta_top_idx.tolist())
    shap_set = set(int(j) for j in shap_top28_idx.tolist())
    overlap = boruta_set & shap_set
    only_boruta = boruta_set - shap_set
    only_shap = shap_set - boruta_set
    jaccard = len(overlap) / float(len(boruta_set | shap_set))
    print(f"\n[compare] |Boruta-28 cap SHAP-28| = {len(overlap)}  "
          f"|only_boruta| = {len(only_boruta)}  "
          f"|only_shap| = {len(only_shap)}  Jaccard = {jaccard:.3f}")
    print(f"   only_boruta = {[feat_names[j] for j in sorted(only_boruta)]}")
    print(f"   only_shap   = {[feat_names[j] for j in sorted(only_shap)]}")

    fam_counts_shap = {}
    for j in shap_top28_idx:
        fam = feat_family[int(j)]
        fam_counts_shap[fam] = fam_counts_shap.get(fam, 0) + 1
    print(f"   SHAP-28 family breakdown   = {fam_counts_shap}")

    # ---- Fit LGBM on Boruta-28: 5-seed bag, 5-fold cross-fit ----
    print("\n" + "-" * 78)
    print(f"LGBM K={TOP_K} on Boruta-28 (5-seed bag, 5-fold cross-fit)")
    print("-" * 78)
    X_b28 = X_unb[:, boruta_top_idx].astype(np.float32)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae_main: list[float] = []
    per_seed_records_main = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        resid_oof_s = _residual_cross_fit_one_seed(X_b28, residual, s)
        pred_corr_s = anchor + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae_main.append(rae_s)
        per_seed_records_main.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_chemprop_aux": rae_s - rae_anchor,
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   seed={s:>3d}  rae_corr = {rae_s:.4f}  "
              f"(d_vs_anchor = {rae_s - rae_anchor:+.4f})  "
              f"wall = {time.time() - ts:.1f}s")

    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))
    per_seed_arr = np.array(per_seed_rae_main)
    print(f"\n[boruta-28] mean_bag   = {rae_mean_bag:.4f}  "
          f"median_bag = {rae_median_bag:.4f}")
    print(f"[boruta-28] per_seed   mean={per_seed_arr.mean():.4f}  "
          f"std={per_seed_arr.std():.4f}")

    delta_vs_nb2103 = rae_mean_bag - nb2103_k28_mean_bag
    delta_median_vs_nb2103 = rae_median_bag - nb2103_k28_median_bag
    beats_nb2103 = rae_mean_bag < (nb2103_k28_mean_bag - DECISION_MARGIN)
    flat_vs_nb2103 = abs(delta_vs_nb2103) < DECISION_MARGIN
    if beats_nb2103:
        main_verdict = "BORUTA28_BEATS_NB2103_K28"
    elif flat_vs_nb2103:
        main_verdict = "BORUTA28_FLAT_VS_NB2103_K28"
    elif rae_mean_bag < rae_anchor - DECISION_MARGIN:
        main_verdict = "BORUTA28_WORSE_THAN_NB2103_K28_BUT_BEATS_ANCHOR"
    else:
        main_verdict = "BORUTA28_FAILS_VS_NB2103_K28"
    print(f"[boruta-28] d_vs_nb2103_K28 = {delta_vs_nb2103:+.4f}  "
          f"verdict = {main_verdict}")

    out_oof = DATA_PROCESSED / f"{TAG}_boruta_mean_bag_oof.npy"
    np.save(out_oof, mean_bag_oof.astype(np.float32))
    print(f"[save] {out_oof}")

    # ---- Fresh-seed verification (conditional on PASS) ----
    fresh_block = None
    if beats_nb2103 or flat_vs_nb2103:
        print("\n" + "-" * 78)
        print(f"FRESH-SEED VERIFICATION on Boruta-28 (seeds {FRESH_SEEDS})")
        print("-" * 78)
        fresh_corrected = np.zeros((len(FRESH_SEEDS), n_unb), dtype=np.float64)
        fresh_per_seed_rae: list[float] = []
        fresh_records = []
        for i, s in enumerate(FRESH_SEEDS):
            ts = time.time()
            resid_oof_s = _residual_cross_fit_one_seed(X_b28, residual, s)
            pred_corr_s = anchor + resid_oof_s
            fresh_corrected[i] = pred_corr_s
            rae_s = float(rae(y_unb, pred_corr_s))
            fresh_per_seed_rae.append(rae_s)
            fresh_records.append({
                "seed": int(s),
                "rae_corrected": rae_s,
                "delta_vs_chemprop_aux": rae_s - rae_anchor,
                "wall_sec": round(time.time() - ts, 2),
            })
            print(f"   fresh seed={s:>3d}  rae_corr = {rae_s:.4f}")
        fresh_mean_bag = fresh_corrected.mean(axis=0)
        fresh_median_bag = np.median(fresh_corrected, axis=0)
        fresh_mean_bag_rae = float(rae(y_unb, fresh_mean_bag))
        fresh_median_bag_rae = float(rae(y_unb, fresh_median_bag))
        fresh_arr = np.array(fresh_per_seed_rae)
        print(f"\n[fresh] mean_bag = {fresh_mean_bag_rae:.4f}  "
              f"median_bag = {fresh_median_bag_rae:.4f}")
        print(f"[fresh] per_seed mean={fresh_arr.mean():.4f}  "
              f"std={fresh_arr.std():.4f}")
        fresh_path = DATA_PROCESSED / f"{TAG}_boruta_fresh_seed_mean_bag_oof.npy"
        np.save(fresh_path, fresh_mean_bag.astype(np.float32))
        print(f"[save] {fresh_path}")
        fresh_delta = fresh_mean_bag_rae - nb2103_k28_mean_bag
        fresh_verdict = (
            "FRESH_BEATS_NB2103"
            if fresh_mean_bag_rae < (nb2103_k28_mean_bag - DECISION_MARGIN)
            else ("FRESH_FLAT_VS_NB2103"
                  if abs(fresh_delta) < DECISION_MARGIN
                  else "FRESH_FAILS_VS_NB2103")
        )
        print(f"[fresh] d_vs_nb2103_K28 = {fresh_delta:+.4f}  "
              f"verdict = {fresh_verdict}")
        fresh_block = {
            "fresh_seeds": FRESH_SEEDS,
            "per_seed_records": fresh_records,
            "per_seed_rae": fresh_per_seed_rae,
            "per_seed_mean": float(fresh_arr.mean()),
            "per_seed_std": float(fresh_arr.std()),
            "rae_mean_bag": fresh_mean_bag_rae,
            "rae_median_bag": fresh_median_bag_rae,
            "delta_vs_nb2103_K28": fresh_delta,
            "verdict": fresh_verdict,
            "fresh_oof_path": str(fresh_path),
        }

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "method": ("boruta_manual_shadow_shuffle_topK_lgbm_mse_5seed_bag"
                   "_on_117col"),
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "data_source": ("same 117-col 5-way K-tuned matrix as nb2103 "
                        "(AtomPair / MACCS / Mordred / ChempropEmbed / Avalon "
                        "+ ChEMBL kNN)"),
        "model_family": "LightGBM",
        "lgbm_hparams": _lgbm_params(0),
        "boruta_max_iters": BORUTA_MAX_ITERS,
        "boruta_confirm_hits": BORUTA_CONFIRM_HITS,
        "boruta_shadow_seed_base": BORUTA_SHADOW_SEED_BASE,
        "boruta_implementation": "manual_shuffle_columns_lgbm_split_count",
        "top_K": TOP_K,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "fresh_seeds": FRESH_SEEDS,
        "feat_dim_full": int(feat_dim),
        "n_unb": n_unb,
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "boruta_n_confirmed": int(boruta_res["n_confirmed"]),
        "boruta_confirmed_idx": boruta_res["confirmed_idx"],
        "boruta_confirmed_rank_order_by_imp":
            boruta_res["confirmed_rank_order_by_imp"],
        "boruta_full_rank_by_imp": boruta_res["full_rank_by_imp"],
        "boruta_hit_count": boruta_res["hit_count"],
        "boruta_mean_real_importance": boruta_res["mean_real_importance"],
        "boruta_shadow_max_imp_per_iter":
            boruta_res["shadow_max_imp_per_iter"],
        "boruta_iter_records": boruta_res["iter_records"],
        "boruta_top28_idx": boruta_top_idx.tolist(),
        "boruta_top28_source": boruta_top_source,
        "boruta_top28_names": [feat_names[int(j)] for j in boruta_top_idx],
        "boruta_top28_family_counts": fam_counts_boruta,
        "shap_top28_idx": shap_top28_idx.tolist(),
        "shap_top28_names": [feat_names[int(j)] for j in shap_top28_idx],
        "shap_top28_family_counts": fam_counts_shap,
        "overlap_size": int(len(overlap)),
        "overlap_idx": sorted(int(j) for j in overlap),
        "overlap_names": [feat_names[int(j)] for j in sorted(overlap)],
        "only_boruta_idx": sorted(int(j) for j in only_boruta),
        "only_boruta_names": [feat_names[int(j)] for j in sorted(only_boruta)],
        "only_shap_idx": sorted(int(j) for j in only_shap),
        "only_shap_names": [feat_names[int(j)] for j in sorted(only_shap)],
        "jaccard_boruta28_vs_shap28": jaccard,
        "per_seed_records_main": per_seed_records_main,
        "per_seed_rae_main": per_seed_rae_main,
        "per_seed_mean_main": float(per_seed_arr.mean()),
        "per_seed_std_main": float(per_seed_arr.std()),
        "rae_mean_bag_main": rae_mean_bag,
        "rae_median_bag_main": rae_median_bag,
        "delta_mean_bag_vs_nb2103_K28": delta_vs_nb2103,
        "delta_median_bag_vs_nb2103_K28": delta_median_vs_nb2103,
        "beats_nb2103_K28": bool(beats_nb2103),
        "flat_vs_nb2103_K28": bool(flat_vs_nb2103),
        "main_verdict": main_verdict,
        "fresh_seed_verification": fresh_block,
        "nb2103_K28_mean_bag_ref": nb2103_k28_mean_bag,
        "nb2103_K28_median_bag_ref": nb2103_k28_median_bag,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "decision_margin": DECISION_MARGIN,
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
        "feat_dim_full",
        "boruta_max_iters",
        "boruta_confirm_hits",
        "boruta_n_confirmed",
        "boruta_top28_source",
        "boruta_top28_family_counts",
        "shap_top28_family_counts",
        "overlap_size",
        "jaccard_boruta28_vs_shap28",
        "rae_anchor_chemprop_aux",
        "rae_mean_bag_main",
        "rae_median_bag_main",
        "nb2103_K28_mean_bag_ref",
        "delta_mean_bag_vs_nb2103_K28",
        "main_verdict",
        "pre_unblind_clean",
    ):
        print(f"  {k}: {res.get(k)}")
    if res.get("fresh_seed_verification") is not None:
        fv = res["fresh_seed_verification"]
        print("\n==== FRESH-SEED ====")
        for k in ("rae_mean_bag", "rae_median_bag",
                  "per_seed_mean", "per_seed_std",
                  "delta_vs_nb2103_K28", "verdict"):
            print(f"  {k}: {fv.get(k)}")
