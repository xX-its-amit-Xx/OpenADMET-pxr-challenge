"""nb2521 -- Heteroscedastic LGBM with quantile spread as variance proxy.

NEW PARADIGM:
    Train 3 LGBM quantile heads (alpha = 0.16, 0.50, 0.84) on the K=20 SHAP
    feature slice.  Use the spread (P84 - P16) / 2 per row as a sigma_row
    variance estimate.  Compute confidence_weight = 1 / sigma_row.  Blend
    nb2240 (K=20 anchor) with the median head P50 row-wise where the weight
    is set by the confidence proxy:

        final = w_row * nb2240 + (1 - w_row) * P50

    w_row = confidence_weight / max(confidence_weight)  (normalised in [0,1])

    Hypothesis: rows where the quantile heads agree (small spread, high
    confidence) get more nb2240 weight because the anchor is trustworthy
    on those.  Rows where the heads disagree (wide spread, low confidence)
    get pushed toward the median head P50 which is less aggressive.

PROTOCOL:
    1. Rebuild the 117-col 5-way K-tuned feature matrix exactly as nb2240
       (AtomPair / MACCS / Mordred / ChempropEmbed / Avalon + ChEMBL kNN),
       slice to K=20 indices from nb2231_summary.json.
    2. For each kf_seed in {1001..1005}, 5-fold scaffold CV on 253 unblind:
         a. Train 3 LGBM heads (alpha 0.16, 0.50, 0.84) per fold on K=20.
            max_depth=4, n_estimators=300, learning_rate=0.03.
         b. Predict P16/P50/P84 on the val fold.
         c. sigma_row = (P84 - P16) / 2
         d. conf_row = 1 / max(sigma_row, eps)
         e. w_row = conf_row / max(conf_row over val fold)
         f. final = w_row * nb2240_oof[va] + (1 - w_row) * P50[va]
    3. Pool OOF across folds, compute pooled RAE per kf_seed.
    4. Deploy: refit 3 quantile heads on ALL 253, predict on 513, use
       te_nb2240_K20 as the anchor for the row-wise confidence-weighted
       blend.  Sigma normalisation uses 513-row max for deploy w_row.

GATE:
    mean_rae < 0.4570 -> PROMOTE
    mean_rae < 0.4601 -> MARGINAL_BEAT
    else             -> FAIL

Outputs:
    scripts/nb2521_heteroscedastic_lgbm.py
    data/processed/nb2521_summary.json
    data/processed/nb2521_pred_oof.npy      (253,) float32
    data/processed/te_nb2521.npy            (513,) float32
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

TAG = "nb2521"

# ---- Paths ----
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
NB2240_OOF_PATH = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"
NB2240_TE_PATH = DATA_PROCESSED / "te_nb2240_K20.npy"

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
NB2231_SUMMARY = DATA_PROCESSED / "nb2231_summary.json"

# ---- ChEMBL pool filters ----
KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

# ---- Protocol ----
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
QUANTILES = [0.16, 0.50, 0.84]
SIGMA_EPS = 1e-3

GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4601


# ============================================================================
# helpers (copied from nb2240)
# ============================================================================

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


def _tanimoto_topk(fp_q, fp_pool, k):
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
    w = np.clip(top_sim, 0.0, 1.0)
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


def _load_mordred_test(n_test_expected):
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(f"Mordred cache missing -- run nb1030 first ({mte_p})")
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


def _load_npy_test(path, n_test_expected):
    if not path.exists():
        raise FileNotFoundError(f"missing cache: {path}")
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
    raise KeyError("AtomPair entry not found in nb1484_summary.json")


def _extract_best_K_record(sum_dict, records_key, best_K_key="best_K"):
    best_K = int(sum_dict[best_K_key])
    for r in sum_dict[records_key]:
        if int(r["K"]) == best_K:
            return r
    raise KeyError(f"best_K {best_K} not found in {records_key}")


def _quantile_lgbm_params(alpha, seed):
    return dict(
        objective="quantile",
        alpha=alpha,
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


def _fit_and_predict_quantiles(X_tr, y_tr, X_va, seed):
    """Train 3 quantile heads, return (P16, P50, P84) on X_va."""
    preds = {}
    for q in QUANTILES:
        mdl = lgb.LGBMRegressor(**_quantile_lgbm_params(q, seed))
        mdl.fit(X_tr, y_tr)
        preds[q] = mdl.predict(X_va)
    return preds[0.16], preds[0.50], preds[0.84]


def _confidence_blend(anchor, p50, sigma, anchor_max_w=None):
    """Row-wise confidence blend.

    sigma: variance proxy (P84 - P16)/2 per row
    anchor: nb2240 prediction
    p50: median quantile head
    returns: blended pred, w_row
    """
    sigma_safe = np.maximum(sigma, SIGMA_EPS)
    conf = 1.0 / sigma_safe
    # Normalise to [0, 1] using max over the slice
    w_row = conf / float(conf.max())
    blended = w_row * anchor + (1.0 - w_row) * p50
    return blended, w_row


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- heteroscedastic LGBM, spread as confidence proxy, blend nb2240")
    print("=" * 78)

    # ---- Load K=20 surviving indices ----
    with open(NB2231_SUMMARY) as f:
        nb2231 = json.load(f)
    surviving_K20 = list(nb2231["snapshots"]["20"]["surviving_idx_in_117"])
    assert len(surviving_K20) == 20, f"expected 20 features, got {len(surviving_K20)}"
    print(f"[load] K=20 surviving features from nb2231")

    # ---- Load truth + anchor ----
    te = load_test()
    n_test = len(te)
    te_smiles = te["smiles"].astype(str).tolist() if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]

    # nb2240 anchor (mean-bag K=20 OOF + deploy te)
    nb2240_oof = np.load(NB2240_OOF_PATH).astype(np.float64)
    nb2240_te = np.load(NB2240_TE_PATH).astype(np.float64)
    assert nb2240_oof.shape == (n_unb,)
    assert nb2240_te.shape == (n_test,)
    print(f"[load] nb2240 OOF RAE = {rae(y_unb, nb2240_oof):.4f}  te shape = {nb2240_te.shape}")

    # ---- Rebuild 117-col feature matrix ----
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

    # ChEMBL kNN
    pool = _load_chembl_pool()
    test_mols = [standardize(s) for s in te_smiles]
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

    # Full 117-col te matrix
    X_te_full = np.concatenate(
        [
            X_ap_te_top,
            X_maccs_te_top,
            X_mord_te_top,
            X_emb_te_top,
            X_av_te_top,
            pred_chembl_pec50.reshape(-1, 1).astype(np.float32),
            mean_sim.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    assert X_te_full.shape[1] == 117

    # Slice to K=20
    X_te_K20 = X_te_full[:, surviving_K20].astype(np.float32)
    X_unb_K20 = X_te_K20[unb_idx]
    print(f"[feat] X_unb_K20 = {X_unb_K20.shape}  X_te_K20 = {X_te_K20.shape}")

    # ============================================================================
    # Scaffold 5-fold CV  --  per kf_seed
    # ============================================================================
    print("\n" + "-" * 78)
    print(f"5-FOLD SCAFFOLD CV   kf_seeds={KF_SEEDS}  quantiles={QUANTILES}")
    print("-" * 78)

    per_seed = []
    all_oofs = []
    sigma_oof_acc = np.zeros(n_unb, dtype=np.float64)
    w_oof_acc = np.zeros(n_unb, dtype=np.float64)

    for kf_seed in KF_SEEDS:
        ts = time.time()
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
        oof_p50 = np.full(n_unb, np.nan, dtype=np.float64)
        oof_sigma = np.full(n_unb, np.nan, dtype=np.float64)
        oof_w = np.full(n_unb, np.nan, dtype=np.float64)
        for tr_loc, va_loc in splits:
            p16, p50, p84 = _fit_and_predict_quantiles(
                X_unb_K20[tr_loc], y_unb[tr_loc], X_unb_K20[va_loc], seed=kf_seed,
            )
            sigma = (p84 - p16) / 2.0
            # confidence blend at row level (anchor = nb2240 on val fold)
            blended, w_row = _confidence_blend(
                anchor=nb2240_oof[va_loc],
                p50=p50,
                sigma=sigma,
            )
            oof_blend[va_loc] = blended
            oof_p50[va_loc] = p50
            oof_sigma[va_loc] = sigma
            oof_w[va_loc] = w_row

        pooled = float(rae(y_unb, oof_blend))
        rae_p50_only = float(rae(y_unb, oof_p50))
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": pooled,
            "rae_p50_only": rae_p50_only,
            "mean_sigma": float(np.nanmean(oof_sigma)),
            "mean_w_anchor": float(np.nanmean(oof_w)),
            "wall_sec": round(time.time() - ts, 2),
        })
        all_oofs.append(oof_blend)
        sigma_oof_acc += oof_sigma
        w_oof_acc += oof_w
        print(f"   seed={kf_seed}  pooled_RAE={pooled:.4f}  p50_only={rae_p50_only:.4f}  "
              f"sigma_mean={np.nanmean(oof_sigma):.3f}  w_anchor_mean={np.nanmean(oof_w):.3f}  "
              f"wall={time.time()-ts:.1f}s")

    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    mean_rae = float(np.mean([r["pooled_rae"] for r in per_seed]))
    std_rae = float(np.std([r["pooled_rae"] for r in per_seed]))
    rae_of_mean_oof = float(rae(y_unb, mean_oof))
    sigma_oof_acc /= len(KF_SEEDS)
    w_oof_acc /= len(KF_SEEDS)

    print(f"\n[cv] mean pooled_RAE across {len(KF_SEEDS)} seeds = {mean_rae:.4f}  "
          f"(+/- {std_rae:.4f})")
    print(f"[cv] RAE of mean-of-seed OOFs              = {rae_of_mean_oof:.4f}")

    # ============================================================================
    # Deploy: refit 3 quantile heads on ALL 253, predict on 513, blend with te_nb2240
    # ============================================================================
    print("\n" + "-" * 78)
    print("DEPLOY (refit quantile heads on 253; row-wise confidence blend on 513)")
    print("-" * 78)
    deploy_seed = KF_SEEDS[0]
    p16_te, p50_te, p84_te = _fit_and_predict_quantiles(
        X_unb_K20, y_unb, X_te_K20, seed=deploy_seed,
    )
    sigma_te = (p84_te - p16_te) / 2.0
    deploy_te, w_te = _confidence_blend(
        anchor=nb2240_te,
        p50=p50_te,
        sigma=sigma_te,
    )
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))
    deploy_te_f32 = deploy_te.astype(np.float32)
    print(f"   in-sample te[unb_idx] RAE = {te_unb_rae:.4f}")
    print(f"   te deploy mean/std        = {deploy_te.mean():.3f} / {deploy_te.std():.3f}")
    print(f"   te sigma mean             = {sigma_te.mean():.3f}")
    print(f"   te w_anchor mean          = {w_te.mean():.3f}")

    # ============================================================================
    # Gate
    # ============================================================================
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"   mean_rae={mean_rae:.4f}  PROMOTE<{GATE_PROMOTE}  MARGINAL<{GATE_MARGINAL}")
    print(f"   verdict = {verdict}")

    # ---- Save artifacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, mean_oof.astype(np.float32))
    np.save(te_path, deploy_te_f32)
    print(f"[save] {oof_path}")
    print(f"[save] {te_path}")

    summary = {
        "tag": TAG,
        "method": "heteroscedastic_quantile_lgbm_confidence_blend_nb2240",
        "anchor": "nb2240_K20",
        "n_unb": n_unb,
        "n_te": n_test,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "quantiles": QUANTILES,
        "sigma_eps": SIGMA_EPS,
        "k20_surviving_idx_in_117": [int(j) for j in surviving_K20],
        "rae_anchor_nb2240": float(rae(y_unb, nb2240_oof)),
        "per_seed_results": per_seed,
        "mean_rae": mean_rae,
        "std_rae": std_rae,
        "rae_of_mean_of_seed_oofs": rae_of_mean_oof,
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "verdict": verdict,
        "te_unb_rae_in_sample": te_unb_rae,
        "te_mean": float(deploy_te.mean()),
        "te_std": float(deploy_te.std()),
        "te_sigma_mean": float(sigma_te.mean()),
        "te_w_anchor_mean": float(w_te.mean()),
        "oof_sigma_mean": float(sigma_oof_acc.mean()),
        "oof_w_anchor_mean": float(w_oof_acc.mean()),
        "oof_path": str(oof_path),
        "te_path": str(te_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   mean_rae (5 seeds)       = {mean_rae:.4f} (+/- {std_rae:.4f})")
    print(f"   rae_anchor_nb2240        = {summary['rae_anchor_nb2240']:.4f}")
    print(f"   te_unb_rae_in_sample     = {te_unb_rae:.4f}")
    print(f"   verdict                  = {verdict}")
    print(f"   wall                     = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae",
        "std_rae",
        "rae_of_mean_of_seed_oofs",
        "verdict",
        "te_unb_rae_in_sample",
        "rae_anchor_nb2240",
        "te_sigma_mean",
        "te_w_anchor_mean",
    ):
        print(f"  {k}: {res.get(k)}")
