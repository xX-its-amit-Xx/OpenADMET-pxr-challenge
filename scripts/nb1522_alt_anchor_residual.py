"""nb1522 -- Alternative anchor: median(chemprop_aux, nb1070).

Hypothesis:
    nb1070 anchor (POST-unblind, inner-fit on 4139+253) gave the best 253
    cross-fit number (nb1373 mean-bag 0.5095) but cross-fit RAE is no
    longer LB-faithful for POST-unblind anchors (two-regime calibration:
    PRE-unblind in_RAE ~ LB + 0.003; POST-unblind unreliable, typically
    LB 0.7-0.9).

    chemprop_aux is PRE-unblind (in_RAE 0.6216 -> predicted LB ~0.6246)
    but nb1460 SHAP-pruned AtomPair+ChEMBL residual on top hit 0.5550
    cross-fit -- still much worse than nb1373's 0.5095.

    Compromise: per-row median(chemprop_aux_te[unb_idx], nb1070_pred_oof)
    as the anchor.  Hopes that combining a PRE-unblind structural signal
    with a POST-unblind in-sample signal trades cross-fit precision for
    LB transferability.  The residual learner then corrects on top of
    this hybrid anchor.

Protocol:
    1.  Anchor = elementwise median over (chemprop_aux[unb_idx],
        nb1070_pred_oof).  (With two arrays, median == mean per row, so
        equivalently  0.5*(chemprop_aux[unb_idx] + nb1070_pred_oof).)
        residual = y_unb - anchor.  Compute anchor RAE on 253.
    2.  ChEMBL PXR pool: identical union as nb1373 / nb1460.
    3.  Build FULL 2050-col matrix (AtomPair-2048 + pred_chembl + sim);
        train seed-0 LGBM Huber on residual; SHAP importance with LGBM
        gain fallback; pick top-30 AtomPair bits.
    4.  Build PRUNED 32-col feature matrix = top-30 AtomPair + pred_chembl
        + mean_sim.
    5.  5-seed bag (seeds [0,1,7,42,137]), KFold(5) cross-fit per seed on
        shallow LGBM Huber (depth=3, num_leaves=7, n_est=80, lr=0.05,
        alpha=1.0, min_child_samples=20).  mean-bag and median-bag.
    6.  Verdict at 0.003 margin vs:
            nb1460 (chemprop_aux anchor)   0.5550
            nb1373 (nb1070 anchor)         0.5095

Outputs:
    scripts/nb1522_alt_anchor_residual.py
    data/processed/nb1522_summary.json
    data/processed/nb1522_mean_bag_oof.npy        (253,) float32
    data/processed/nb1522_median_bag_oof.npy      (253,) float32
    data/processed/nb1522_per_seed_corrected_oof.npy (5, 253) float32
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
from lightgbm import LGBMRegressor
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1522"
ANCHOR_NAME = "median_chemprop_aux_plus_nb1070"
CHEMPROP_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"        # PRE-unblind (513,)
NB1070_OOF_PATH = DATA_PROCESSED / "nb1070_pred_oof.npy"         # POST-unblind (253,)

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"            # (513, 2048) uint8
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

NB1460_REF = 0.5550   # chemprop_aux anchor mean-bag (PRE-unblind path)
NB1373_REF = 0.5095   # nb1070 anchor mean-bag (POST-unblind path)
DECISION_MARGIN = 0.003

TOP_K_ATOMPAIR = 30


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
    """Same union as nb1242 / nb1352 / nb1373 / nb1460."""
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
        objective="huber",
        alpha=1.0,
        learning_rate=0.05,
        n_estimators=80,
        max_depth=3,
        num_leaves=7,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        verbosity=-1,
        random_state=seed,
        n_jobs=2,
    )


def _residual_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray,
                                 seed: int) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _compute_shap_importance(X: np.ndarray, residual: np.ndarray, seed: int = 0):
    mdl = LGBMRegressor(**_lgbm_params(seed))
    mdl.fit(X, residual)
    try:
        import shap
        explainer = shap.TreeExplainer(mdl)
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[0]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[..., 0]
        imp = np.abs(sv).mean(axis=0)
        return imp.astype(np.float64), "shap_tree_explainer"
    except Exception as e:
        print(f"   [shap] WARN: shap failed ({e}); falling back to LGBM gain")
        imp = mdl.booster_.feature_importance(importance_type="gain")
        return imp.astype(np.float64), "lgbm_gain_fallback"


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- ALT-ANCHOR: median(chemprop_aux, nb1070) + SHAP-pruned "
          f"AtomPair-{TOP_K_ATOMPAIR} + ChEMBL residual learner")
    print(f"          seeds = {RESID_SEEDS}  folds = {RESID_FOLDS}")
    print(f"          baselines: nb1460 chemprop_aux ({NB1460_REF:.4f}), "
          f"nb1373 nb1070 ({NB1373_REF:.4f})  margin = {DECISION_MARGIN}")
    print("=" * 78)

    # ---- Load truth + both anchor sources ----
    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    # PRE-unblind anchor: chemprop_aux te file on 513, slice to unb_idx
    if not CHEMPROP_TE_PATH.exists():
        raise FileNotFoundError(f"chemprop_aux te file missing: {CHEMPROP_TE_PATH}")
    te_chemprop_513 = np.load(CHEMPROP_TE_PATH).astype(np.float64)
    if te_chemprop_513.shape[0] != n_test:
        raise ValueError(
            f"chemprop_aux te shape mismatch: {te_chemprop_513.shape} vs {n_test}"
        )
    chemprop_unb = te_chemprop_513[unb_idx]
    rae_chemprop = float(rae(y_unb, chemprop_unb))
    print(f"[load] chemprop_aux[unb_idx]  in_RAE = {rae_chemprop:.4f}  "
          f"mean={chemprop_unb.mean():.3f}  std={chemprop_unb.std():.3f}")

    # POST-unblind anchor: nb1070 pred_oof (253,)
    if not NB1070_OOF_PATH.exists():
        raise FileNotFoundError(f"nb1070 pred_oof missing: {NB1070_OOF_PATH}")
    nb1070_unb = np.load(NB1070_OOF_PATH).astype(np.float64)
    if nb1070_unb.shape[0] != n_unb:
        raise ValueError(
            f"nb1070 pred_oof shape mismatch: {nb1070_unb.shape} vs {n_unb}"
        )
    rae_nb1070 = float(rae(y_unb, nb1070_unb))
    print(f"[load] nb1070 pred_oof         RAE = {rae_nb1070:.4f}  "
          f"mean={nb1070_unb.mean():.3f}  std={nb1070_unb.std():.3f}")

    # Per-row median (two arrays -> identical to mean per row)
    anchor_stack = np.stack([chemprop_unb, nb1070_unb], axis=0)   # (2, n_unb)
    anchor = np.median(anchor_stack, axis=0)
    rae_anchor = float(rae(y_unb, anchor))
    pearson_anchors = float(np.corrcoef(chemprop_unb, nb1070_unb)[0, 1])
    print(f"[anchor] median(chemprop_aux, nb1070)  RAE = {rae_anchor:.4f}  "
          f"mean={anchor.mean():.3f}  std={anchor.std():.3f}")
    print(f"[anchor] Pearson(chemprop_aux, nb1070) on 253 = {pearson_anchors:+.4f}")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- ChEMBL pool ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL (local cache; same union as nb1352 / nb1373 / nb1460)")
    print("-" * 78)
    pool = _load_chembl_pool()

    # ---- Test InChIKey leak guard ----
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

    # ---- Morgan FPs for kNN ----
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

    # ---- kNN k=5 Tanimoto ----
    top_idx, top_sim = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx, top_sim, pool_labels, fallback=pool_median
    )
    top1_sim = top_sim[:, 0]
    print(f"   pred_chembl_pec50  mean={pred_chembl_pec50.mean():.3f}  "
          f"std={pred_chembl_pec50.std():.3f}")
    print(f"   top1 sim   p50={np.percentile(top1_sim, 50):.3f}")

    # ---- AtomPair-2048 (unblind slice) ----
    if not ATOMPAIR_TE_PATH.exists():
        raise FileNotFoundError(f"AtomPair test cache missing: {ATOMPAIR_TE_PATH}")
    X_ap_te = np.load(ATOMPAIR_TE_PATH)
    if X_ap_te.shape[0] != n_test:
        raise ValueError(f"AtomPair cache shape mismatch: {X_ap_te.shape}")
    n_ap = int(X_ap_te.shape[1])
    print(f"   AtomPair cache shape = {X_ap_te.shape}  (n_bits={n_ap})")
    X_ap_unb = X_ap_te[unb_idx].astype(np.float32)
    print(f"   bit density (unb) = {X_ap_unb.mean():.4f}  "
          f"const cols = {int((X_ap_unb.var(axis=0) == 0).sum())}/{n_ap}")

    # ---- FULL feature matrix ----
    pred_chembl_unb = pred_chembl_pec50[unb_idx].astype(np.float32)
    mean_sim_unb = mean_sim[unb_idx].astype(np.float32)
    X_unb_full = np.concatenate(
        [
            X_ap_unb,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim_full = X_unb_full.shape[1]
    print(f"   FULL feature matrix: {X_unb_full.shape}  "
          f"(AtomPair-{n_ap} + pred_chembl + sim)")

    # ---- SHAP importance frame ----
    print("\n" + "-" * 78)
    print("SHAP IMPORTANCE FRAME (1 global LGBM, seed=0, full feature matrix)")
    print("-" * 78)
    imp_full, imp_src = _compute_shap_importance(X_unb_full, residual, seed=0)
    print(f"   importance source = {imp_src}")
    print(f"   importance vector shape = {imp_full.shape}")
    print(f"   pred_chembl importance = {imp_full[n_ap]:.4f}")
    print(f"   sim importance         = {imp_full[n_ap + 1]:.4f}")

    ap_imp = imp_full[:n_ap]
    top_k = min(TOP_K_ATOMPAIR, n_ap)
    top_bit_order = np.argsort(-ap_imp)
    top_bit_idx = top_bit_order[:top_k].astype(int)
    top_bit_idx_sorted = np.sort(top_bit_idx)
    top_bit_imp = ap_imp[top_bit_idx]
    print(f"   top-{top_k} AtomPair bit indices (ranked by importance):")
    for rank, (bit, val) in enumerate(zip(top_bit_idx.tolist(),
                                          top_bit_imp.tolist())):
        print(f"      rank {rank+1:2d}:  bit {bit:5d}   imp = {val:.5f}")
    print(f"   top-{top_k} bit indices (sorted asc): {top_bit_idx_sorted.tolist()}")

    n_nonzero_imp = int((ap_imp > 0).sum())
    print(f"   AtomPair bits with nonzero importance: {n_nonzero_imp}/{n_ap}")

    # ---- PRUNED feature matrix ----
    X_ap_unb_pruned = X_ap_unb[:, top_bit_idx]
    X_unb_pruned = np.concatenate(
        [
            X_ap_unb_pruned,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim_pruned = X_unb_pruned.shape[1]
    print(f"\n   PRUNED feature matrix: {X_unb_pruned.shape}  "
          f"(top-{top_k} AtomPair + pred_chembl + sim)")

    # ---- Per-seed residual cross-fit on PRUNED features ----
    print("\n" + "-" * 78)
    print(f"PER-SEED RESIDUAL CROSS-FIT (PRUNED, dim={feat_dim_pruned})")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []
    for i, s in enumerate(RESID_SEEDS):
        resid_oof_s = _residual_cross_fit_one_seed(X_unb_pruned, residual, s)
        pred_corr_s = anchor + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        delta_s = rae_s - rae_anchor
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_anchor": delta_s,
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
        })
        print(f"   seed {s:3d}:  rae_corr = {rae_s:.4f}  "
              f"(d_vs_anchor = {delta_s:+.4f})  "
              f"|resid_oof|.std = {resid_oof_s.std():.3f}")

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

    print("\n" + "-" * 78)
    print("BAG AGGREGATIONS")
    print("-" * 78)
    print(f"   per-seed RAE list      = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   per-seed mean          = {rae_per_seed_mean:.4f}")
    print(f"   per-seed median        = {rae_per_seed_median:.4f}")
    print(f"   per-seed std           = {rae_per_seed_std:.4f}")
    print(f"   per-seed min/max       = "
          f"{rae_per_seed_min:.4f} / {rae_per_seed_max:.4f}")
    print(f"   anchor RAE             = {rae_anchor:.4f}")
    print(f"   pooled RAE(mean_bag)   = {rae_mean_bag:.4f}  "
          f"(d_vs_anchor = {rae_mean_bag - rae_anchor:+.4f}"
          f"  d_vs_nb1460 = {rae_mean_bag - NB1460_REF:+.4f}"
          f"  d_vs_nb1373 = {rae_mean_bag - NB1373_REF:+.4f})")
    print(f"   pooled RAE(median_bag) = {rae_median_bag:.4f}  "
          f"(d_vs_anchor = {rae_median_bag - rae_anchor:+.4f}"
          f"  d_vs_nb1460 = {rae_median_bag - NB1460_REF:+.4f}"
          f"  d_vs_nb1373 = {rae_median_bag - NB1373_REF:+.4f})")

    beats_anchor = rae_mean_bag < rae_anchor - DECISION_MARGIN
    beats_nb1460 = rae_mean_bag < NB1460_REF - DECISION_MARGIN
    beats_nb1373 = rae_mean_bag < NB1373_REF - DECISION_MARGIN

    if beats_nb1373:
        verdict = "ALT_ANCHOR_BEATS_NB1373_NB1070_NEW_BEST"
    elif beats_nb1460 and abs(rae_mean_bag - NB1373_REF) < DECISION_MARGIN:
        verdict = "ALT_ANCHOR_FLAT_VS_NB1373_BUT_BEATS_NB1460"
    elif beats_nb1460:
        verdict = "ALT_ANCHOR_BEATS_NB1460_BUT_WORSE_THAN_NB1373"
    elif abs(rae_mean_bag - NB1460_REF) < DECISION_MARGIN:
        verdict = "ALT_ANCHOR_FLAT_VS_NB1460_NO_LIFT"
    elif beats_anchor:
        verdict = "ALT_ANCHOR_RESIDUAL_LIFTS_ANCHOR_BUT_BELOW_NB1460"
    elif abs(rae_mean_bag - rae_anchor) < DECISION_MARGIN:
        verdict = "ALT_ANCHOR_FLAT_VS_STANDALONE_NO_RESIDUAL_LIFT"
    else:
        verdict = "ALT_ANCHOR_RESIDUAL_HURTS_STANDALONE"
    print(f"   verdict                = {verdict}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_per_seed_corrected_oof.npy",
            per_seed_corrected.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy",
            mean_bag_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_median_bag_oof.npy",
            median_bag_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_per_seed_corrected_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_mean_bag_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_median_bag_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor_name": ANCHOR_NAME,
        "anchor_kind": "per_row_median_of_chemprop_aux_te_and_nb1070_pred_oof",
        "chemprop_te_path": str(CHEMPROP_TE_PATH),
        "nb1070_oof_path": str(NB1070_OOF_PATH),
        "data_source": "local_chembl_caches_union",
        "n_chembl_pool": int(len(pool)),
        "n_unb": n_unb,
        "n_atompair_bits": n_ap,
        "atompair_bit_density_unb": float(X_ap_unb.mean()),
        "atompair_const_cols": int((X_ap_unb.var(axis=0) == 0).sum()),
        "atompair_nonzero_imp_bits": n_nonzero_imp,
        "shap_importance_source": imp_src,
        "top_k_atompair": int(top_k),
        "top_atompair_bit_indices_ranked": [int(b) for b in top_bit_idx.tolist()],
        "top_atompair_bit_importance_ranked": [float(v) for v in top_bit_imp.tolist()],
        "top_atompair_bit_indices_sorted_asc": [int(b) for b in top_bit_idx_sorted.tolist()],
        "pred_chembl_importance": float(imp_full[n_ap]),
        "sim_importance": float(imp_full[n_ap + 1]),
        "feat_dim_full": int(feat_dim_full),
        "feat_dim_pruned": int(feat_dim_pruned),
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "lgbm_max_depth": 3,
        "lgbm_num_leaves": 7,
        "lgbm_n_estimators": 80,
        "lgbm_learning_rate": 0.05,
        "lgbm_min_child_samples": 20,
        "lgbm_huber_alpha": 1.0,
        "rae_chemprop_aux": rae_chemprop,
        "rae_nb1070": rae_nb1070,
        "rae_anchor_median": rae_anchor,
        "pearson_anchors": pearson_anchors,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_seed_rae": per_seed_rae,
        "per_seed_records": per_seed_records,
        "rae_per_seed_mean": rae_per_seed_mean,
        "rae_per_seed_median": rae_per_seed_median,
        "rae_per_seed_std": rae_per_seed_std,
        "rae_per_seed_min": rae_per_seed_min,
        "rae_per_seed_max": rae_per_seed_max,
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_anchor": rae_mean_bag - rae_anchor,
        "delta_mean_bag_vs_nb1460": rae_mean_bag - NB1460_REF,
        "delta_mean_bag_vs_nb1373": rae_mean_bag - NB1373_REF,
        "beats_anchor": bool(beats_anchor),
        "beats_nb1460": bool(beats_nb1460),
        "beats_nb1373": bool(beats_nb1373),
        "verdict": verdict,
        "nb1460_ref": NB1460_REF,
        "nb1373_ref": NB1373_REF,
        "decision_margin": DECISION_MARGIN,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "n_chembl_pool", "n_atompair_bits", "shap_importance_source",
        "top_k_atompair",
        "top_atompair_bit_indices_ranked",
        "top_atompair_bit_importance_ranked",
        "pred_chembl_importance", "sim_importance",
        "feat_dim_full", "feat_dim_pruned",
        "atompair_nonzero_imp_bits",
        "rae_chemprop_aux", "rae_nb1070",
        "rae_anchor_median", "pearson_anchors",
        "per_seed_rae",
        "rae_per_seed_mean", "rae_per_seed_std",
        "rae_mean_bag", "rae_median_bag",
        "delta_mean_bag_vs_anchor",
        "delta_mean_bag_vs_nb1460",
        "delta_mean_bag_vs_nb1373",
        "beats_anchor", "beats_nb1460", "beats_nb1373",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
