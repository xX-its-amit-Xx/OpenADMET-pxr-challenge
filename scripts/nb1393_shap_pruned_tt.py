"""nb1393 -- SHAP-pruned Topological Torsion (TT) top-30 + ChEMBL residual learner.

Hypothesis:
    nb1173 used the full TT-2048 fingerprint as residual features over the
    nb1070 anchor and stalled flat at mean-bag RAE 0.5780 -- the AtomPair
    cousin nb1172 was 0.5659 at full dim, but SHAP-pruning AtomPair from
    2048 -> 30 bits (nb1373) cut it to 0.5095.  Same recipe on TT may rescue
    the channel: 2048-d residual capacity at n=253 is so over-parameterized
    that SHAP pruning to ~30 bits + (pred_chembl, sim) may extract the 4-atom
    path signal that nb1173's full-dim version was drowning out.

Protocol (identical to nb1373 except TT instead of AtomPair):
    1.  Anchor = nb1070_pred_oof on 253 unblind rows.
        residual = y_unb - anchor.
    2.  Build ChEMBL PXR pool (same union as nb1242/nb1352/nb1373).
        Compute kNN-5 Tanimoto on cached Morgan-2048 over std test SMILES;
        pull pred_chembl_pec50 + mean_sim for all test rows, slice to
        unblind.
    3.  Build FULL 2050-col feature matrix on 253:
            TT-2048 (cached te_topotorsion.npy, sliced) + pred_chembl + sim
        Train ONE seed-0 shallow LGBM Huber on residual to get SHAP
        importance via shap.TreeExplainer; fall back to LGBM gain on
        failure.  Take mean(|SHAP|) per feature, slice TT-only, pick
        top-30 bit indices.
    4.  Build PRUNED 32-col feature matrix = top-30 TT + pred_chembl + sim.
    5.  5-seed bag (seeds [0, 1, 7, 42, 137]), KFold(n=5) cross-fit per
        seed on shallow LGBM Huber (depth=3, num_leaves=7, n_est=80,
        lr=0.05, huber_alpha=1.0, min_child_samples=20).  Mean-bag pooled
        RAE.
    6.  Verdict at 0.003 margin vs:
            nb1373  (SHAP-pruned AtomPair top-30, 0.5095 mean-bag) -- target
            nb1173  (full TT-2048 standalone, 0.5780 mean-bag) -- channel ref
            nb1070  (anchor, ~0.5771-0.5790 pooled)

Outputs:
    scripts/nb1393_shap_pruned_tt.py        (this file)
    data/processed/nb1393_summary.json
    data/processed/nb1393_mean_bag_oof.npy            (253,) float32
    data/processed/nb1393_median_bag_oof.npy          (253,) float32
    data/processed/nb1393_per_seed_corrected_oof.npy  (5, 253) float32
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

TAG = "nb1393"
ANCHOR = "nb1070"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

TT_TE_PATH = DATA_PROCESSED / "te_topotorsion.npy"   # (513, 2048) uint8
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

NB1070_REF = 0.5771
NB1173_REF = 0.5780      # full TT-2048 standalone residual bag
NB1373_REF = 0.5095      # SHAP-pruned AtomPair top-30 + ChEMBL residual bag
DECISION_MARGIN = 0.003

TOP_K_TT = 30


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
    """Same union as nb1242 / nb1352 / nb1373."""
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
    """Train one global LGBM on residual; return (importance vector, source_tag)."""
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
    print(f"{TAG} -- SHAP-pruned TT top-{TOP_K_TT} + ChEMBL residual; "
          f"anchor={ANCHOR}")
    print(f"          seeds = {RESID_SEEDS}  folds = {RESID_FOLDS}")
    print(f"          baselines: nb1373 ({NB1373_REF:.4f}), "
          f"nb1173 ({NB1173_REF:.4f})  margin = {DECISION_MARGIN}")
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

    anchor = np.load(DATA_PROCESSED / f"{ANCHOR}_pred_oof.npy").astype(np.float64)
    if anchor.shape[0] != n_unb:
        raise ValueError(f"{ANCHOR} shape mismatch: {anchor.shape} vs {n_unb}")
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} pooled RAE = {rae_anchor:.4f}  (ref ~{NB1070_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- ChEMBL pool ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL (local cache; same union as nb1373)")
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

    # ---- TT-2048 (unblind slice) ----
    if not TT_TE_PATH.exists():
        raise FileNotFoundError(
            f"TT test cache missing: {TT_TE_PATH}"
        )
    X_tt_te = np.load(TT_TE_PATH)
    if X_tt_te.shape[0] != n_test:
        raise ValueError(f"TT cache shape mismatch: {X_tt_te.shape}")
    n_tt = int(X_tt_te.shape[1])
    print(f"   TT cache shape = {X_tt_te.shape}  (n_bits={n_tt})")
    X_tt_unb = X_tt_te[unb_idx].astype(np.float32)
    print(f"   bit density (unb) = {X_tt_unb.mean():.4f}  "
          f"const cols = {int((X_tt_unb.var(axis=0) == 0).sum())}/{n_tt}")

    # ---- Build FULL feature matrix (TT + pred_chembl + sim) ----
    pred_chembl_unb = pred_chembl_pec50[unb_idx].astype(np.float32)
    mean_sim_unb = mean_sim[unb_idx].astype(np.float32)
    X_unb_full = np.concatenate(
        [
            X_tt_unb,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim_full = X_unb_full.shape[1]
    print(f"   FULL feature matrix: {X_unb_full.shape}  "
          f"(TT-{n_tt} + pred_chembl + sim)")

    # ---- SHAP importance frame ----
    print("\n" + "-" * 78)
    print("SHAP IMPORTANCE FRAME (1 global LGBM, seed=0, full feature matrix)")
    print("-" * 78)
    imp_full, imp_src = _compute_shap_importance(X_unb_full, residual, seed=0)
    print(f"   importance source = {imp_src}")
    print(f"   importance vector shape = {imp_full.shape}")
    print(f"   pred_chembl importance = {imp_full[n_tt]:.4f}")
    print(f"   sim importance         = {imp_full[n_tt + 1]:.4f}")

    # TT-only importance slice
    tt_imp = imp_full[:n_tt]
    top_k = min(TOP_K_TT, n_tt)
    top_bit_order = np.argsort(-tt_imp)
    top_bit_idx = top_bit_order[:top_k].astype(int)
    top_bit_idx_sorted = np.sort(top_bit_idx)
    top_bit_imp = tt_imp[top_bit_idx]
    print(f"   top-{top_k} TT bit indices (ranked by importance):")
    for rank, (bit, val) in enumerate(zip(top_bit_idx.tolist(),
                                          top_bit_imp.tolist())):
        print(f"      rank {rank+1:2d}:  bit {bit:5d}   imp = {val:.5f}")
    print(f"   top-{top_k} bit indices (sorted asc): {top_bit_idx_sorted.tolist()}")

    n_nonzero_imp = int((tt_imp > 0).sum())
    print(f"   TT bits with nonzero importance: {n_nonzero_imp}/{n_tt}")

    # ---- Build PRUNED 32-col feature matrix ----
    X_tt_unb_pruned = X_tt_unb[:, top_bit_idx]
    X_unb_pruned = np.concatenate(
        [
            X_tt_unb_pruned,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim_pruned = X_unb_pruned.shape[1]
    print(f"\n   PRUNED feature matrix: {X_unb_pruned.shape}  "
          f"(top-{top_k} TT + pred_chembl + sim)")

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
            "delta_vs_nb1070": delta_s,
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
        })
        print(f"   seed {s:3d}:  rae_corr = {rae_s:.4f}  "
              f"(d_vs_nb1070 = {delta_s:+.4f})  "
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
    print(f"   pooled RAE(mean_bag)   = {rae_mean_bag:.4f}  "
          f"(d_vs_nb1070 = {rae_mean_bag - rae_anchor:+.4f}"
          f"  d_vs_nb1173 = {rae_mean_bag - NB1173_REF:+.4f}"
          f"  d_vs_nb1373 = {rae_mean_bag - NB1373_REF:+.4f})")
    print(f"   pooled RAE(median_bag) = {rae_median_bag:.4f}  "
          f"(d_vs_nb1070 = {rae_median_bag - rae_anchor:+.4f}"
          f"  d_vs_nb1173 = {rae_median_bag - NB1173_REF:+.4f}"
          f"  d_vs_nb1373 = {rae_median_bag - NB1373_REF:+.4f})")
    print(f"   nb1070 ref            = {NB1070_REF:.4f}")
    print(f"   nb1173 ref            = {NB1173_REF:.4f}  (full TT-2048)")
    print(f"   nb1373 ref            = {NB1373_REF:.4f}  (SHAP-pruned AtomPair top-30)")

    beats_nb1070 = rae_mean_bag < rae_anchor - DECISION_MARGIN
    beats_nb1173 = rae_mean_bag < NB1173_REF - DECISION_MARGIN
    beats_nb1373 = rae_mean_bag < NB1373_REF - DECISION_MARGIN

    if beats_nb1373:
        verdict = "SHAP_PRUNED_TT_BEATS_NB1373_NEW_PRIMARY_CANDIDATE"
    elif abs(rae_mean_bag - NB1373_REF) < DECISION_MARGIN:
        verdict = "SHAP_PRUNED_TT_FLAT_VS_NB1373"
    elif beats_nb1173:
        verdict = "SHAP_PRUNED_TT_BEATS_NB1173_BUT_WORSE_THAN_NB1373"
    elif beats_nb1070:
        verdict = "SHAP_PRUNED_TT_HELPS_NB1070_BUT_WORSE_THAN_NB1173"
    else:
        verdict = "SHAP_PRUNED_TT_HURTS_NB1070"
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
        "anchor": ANCHOR,
        "data_source": "local_chembl_caches_union",
        "n_chembl_pool": int(len(pool)),
        "n_unb": n_unb,
        "n_tt_bits": n_tt,
        "tt_bit_density_unb": float(X_tt_unb.mean()),
        "tt_const_cols": int((X_tt_unb.var(axis=0) == 0).sum()),
        "tt_nonzero_imp_bits": n_nonzero_imp,
        "shap_importance_source": imp_src,
        "top_k_tt": int(top_k),
        "top_tt_bit_indices_ranked": [int(b) for b in top_bit_idx.tolist()],
        "top_tt_bit_importance_ranked": [float(v) for v in top_bit_imp.tolist()],
        "top_tt_bit_indices_sorted_asc": [int(b) for b in top_bit_idx_sorted.tolist()],
        "pred_chembl_importance": float(imp_full[n_tt]),
        "sim_importance": float(imp_full[n_tt + 1]),
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
        "rae_anchor_nb1070": rae_anchor,
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
        "delta_mean_bag_vs_nb1070": rae_mean_bag - rae_anchor,
        "delta_mean_bag_vs_nb1173": rae_mean_bag - NB1173_REF,
        "delta_mean_bag_vs_nb1373": rae_mean_bag - NB1373_REF,
        "beats_nb1070": bool(beats_nb1070),
        "beats_nb1173": bool(beats_nb1173),
        "beats_nb1373": bool(beats_nb1373),
        "verdict": verdict,
        "nb1070_ref_pooled": NB1070_REF,
        "nb1173_ref": NB1173_REF,
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
        "n_chembl_pool", "n_tt_bits", "shap_importance_source",
        "top_k_tt",
        "top_tt_bit_indices_ranked",
        "top_tt_bit_importance_ranked",
        "pred_chembl_importance", "sim_importance",
        "feat_dim_full", "feat_dim_pruned",
        "tt_nonzero_imp_bits",
        "rae_anchor_nb1070", "per_seed_rae",
        "rae_per_seed_mean", "rae_per_seed_std",
        "rae_mean_bag", "rae_median_bag",
        "delta_mean_bag_vs_nb1070",
        "delta_mean_bag_vs_nb1173",
        "delta_mean_bag_vs_nb1373",
        "beats_nb1070", "beats_nb1173", "beats_nb1373",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
