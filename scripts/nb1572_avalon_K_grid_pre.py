"""nb1572 -- Fresh SHAP K-grid on PRE-unblind path for Avalon-512.

Anchor: chemprop_aux (PRE-unblind, te[unb_idx] in_RAE ~ 0.6216).
Residual learner: shallow LGBM Huber on top-K SHAP-ranked Avalon bits
+ pred_chembl + mean_sim.  Sweep K in {10, 15, 20, 25, 40, 50}, with
5-seed bag (seeds [0, 1, 7, 42, 137]) and KFold(5) cross-fit per K, per seed.

Reference: nb1553 used K=30 for Avalon in the 5-way Family slot, scoring
mean_bag RAE 0.5626 (slot baseline).  This notebook finds the true Avalon K
on the PRE-unblind / chemprop_aux anchor path, and the verdict is at the
0.003 margin vs that 0.5626 reference (nb1392 / nb1553 Avalon slot).

Outputs:
    scripts/nb1572_avalon_K_grid_pre.py
    data/processed/nb1572_summary.json
    data/processed/nb1572_best_K_oof.npy        (253,) float32 mean-bag at best K
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

TAG = "nb1572"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

CHEMPROP_AUX_REF = 0.6216
NB1553_AVALON_K30_REF = 0.5626   # nb1553 Avalon slot (K=30) mean-bag RAE
NB1392_K30_REF = 0.5391          # nb1392 SHAP-pruned Avalon-512 K=30 (nb1070 anchor)
DECISION_MARGIN = 0.003

K_GRID = [10, 15, 20, 25, 40, 50]


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
    print(f"{TAG} -- PRE-unblind: SHAP K-grid Avalon-512  anchor={ANCHOR}")
    print(f"          K grid = {K_GRID}")
    print(f"          seeds = {RESID_SEEDS}  folds = {RESID_FOLDS}")
    print(f"          baselines: chemprop_aux ({CHEMPROP_AUX_REF:.4f}), "
          f"nb1392/nb1553 Avalon K=30 ({NB1553_AVALON_K30_REF:.4f})  "
          f"margin = {DECISION_MARGIN}")
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

    # ---- ChEMBL pool ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL (local cache; same union as nb1352 / nb1373 / nb1460)")
    print("-" * 78)
    pool = _load_chembl_pool()

    # Leak guard against test InChIKeys
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

    top_idx, top_sim = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx, top_sim, pool_labels, fallback=pool_median
    )
    print(f"   pred_chembl_pec50  mean={pred_chembl_pec50.mean():.3f}  "
          f"std={pred_chembl_pec50.std():.3f}")

    # ---- Avalon-512 (unblind slice) ----
    if not AVALON_TE_PATH.exists():
        raise FileNotFoundError(
            f"Avalon test cache missing: {AVALON_TE_PATH}"
        )
    X_av_te = np.load(AVALON_TE_PATH)
    if X_av_te.shape[0] != n_test:
        raise ValueError(f"Avalon cache shape mismatch: {X_av_te.shape}")
    n_av = int(X_av_te.shape[1])
    print(f"   Avalon cache shape = {X_av_te.shape}  (n_bits={n_av})")
    X_av_unb = X_av_te[unb_idx].astype(np.float32)
    print(f"   bit density (unb) = {X_av_unb.mean():.4f}  "
          f"const cols = {int((X_av_unb.var(axis=0) == 0).sum())}/{n_av}")

    # ---- Build FULL feature matrix ----
    pred_chembl_unb = pred_chembl_pec50[unb_idx].astype(np.float32)
    mean_sim_unb = mean_sim[unb_idx].astype(np.float32)
    X_unb_full = np.concatenate(
        [
            X_av_unb,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim_full = X_unb_full.shape[1]
    print(f"   FULL feature matrix: {X_unb_full.shape}  "
          f"(Avalon-{n_av} + pred_chembl + sim)")

    # ---- FRESH SHAP frame on residual ----
    print("\n" + "-" * 78)
    print("SHAP IMPORTANCE FRAME (1 global LGBM, seed=0, full feature matrix)")
    print("-" * 78)
    imp_full, imp_src = _compute_shap_importance(X_unb_full, residual, seed=0)
    print(f"   importance source = {imp_src}")
    print(f"   pred_chembl importance = {imp_full[n_av]:.4f}")
    print(f"   sim importance         = {imp_full[n_av + 1]:.4f}")

    av_imp = imp_full[:n_av]
    top_bit_order = np.argsort(-av_imp)
    n_nonzero_imp = int((av_imp > 0).sum())
    print(f"   Avalon bits with nonzero importance: {n_nonzero_imp}/{n_av}")

    # ---- K-grid sweep ----
    print("\n" + "-" * 78)
    print("K-GRID SWEEP (per K: 5-seed bag, 5-fold cross-fit on residual)")
    print("-" * 78)
    k_records = []
    k_oof_store = {}
    for K in K_GRID:
        top_k = min(K, n_av)
        top_bit_idx = top_bit_order[:top_k].astype(int)
        X_av_unb_pruned = X_av_unb[:, top_bit_idx]
        X_unb_pruned = np.concatenate(
            [
                X_av_unb_pruned,
                pred_chembl_unb.reshape(-1, 1),
                mean_sim_unb.reshape(-1, 1),
            ],
            axis=1,
        ).astype(np.float32)
        feat_dim_pruned = X_unb_pruned.shape[1]

        per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
        per_seed_rae: list[float] = []
        for i, s in enumerate(RESID_SEEDS):
            resid_oof_s = _residual_cross_fit_one_seed(X_unb_pruned, residual, s)
            pred_corr_s = anchor + resid_oof_s
            per_seed_corrected[i] = pred_corr_s
            per_seed_rae.append(float(rae(y_unb, pred_corr_s)))
        mean_bag_oof = per_seed_corrected.mean(axis=0)
        median_bag_oof = np.median(per_seed_corrected, axis=0)
        rae_mean_bag = float(rae(y_unb, mean_bag_oof))
        rae_median_bag = float(rae(y_unb, median_bag_oof))

        beats_anchor = rae_mean_bag < rae_anchor - DECISION_MARGIN
        beats_nb1553_K30 = rae_mean_bag < NB1553_AVALON_K30_REF - DECISION_MARGIN

        k_records.append({
            "K": int(K),
            "feat_dim_pruned": int(feat_dim_pruned),
            "per_seed_rae": per_seed_rae,
            "rae_per_seed_mean": float(np.mean(per_seed_rae)),
            "rae_per_seed_std": float(np.std(per_seed_rae)),
            "rae_per_seed_min": float(np.min(per_seed_rae)),
            "rae_per_seed_max": float(np.max(per_seed_rae)),
            "rae_mean_bag": rae_mean_bag,
            "rae_median_bag": rae_median_bag,
            "delta_mean_bag_vs_anchor": rae_mean_bag - rae_anchor,
            "delta_mean_bag_vs_nb1553_K30": rae_mean_bag - NB1553_AVALON_K30_REF,
            "beats_anchor": bool(beats_anchor),
            "beats_nb1553_K30": bool(beats_nb1553_K30),
        })
        k_oof_store[K] = mean_bag_oof.astype(np.float32)

        print(f"   K={K:3d}  feat={feat_dim_pruned:3d}  "
              f"per-seed=[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]  "
              f"mean_bag={rae_mean_bag:.4f}  median_bag={rae_median_bag:.4f}  "
              f"d_vs_anchor={rae_mean_bag - rae_anchor:+.4f}  "
              f"d_vs_nb1553_K30={rae_mean_bag - NB1553_AVALON_K30_REF:+.4f}")

    # ---- Pick best K ----
    best = min(k_records, key=lambda r: r["rae_mean_bag"])
    best_K = int(best["K"])
    best_rae = best["rae_mean_bag"]
    best_oof = k_oof_store[best_K]

    beats_anchor = best_rae < rae_anchor - DECISION_MARGIN
    beats_nb1553_K30 = best_rae < NB1553_AVALON_K30_REF - DECISION_MARGIN
    flat_vs_nb1553_K30 = abs(best_rae - NB1553_AVALON_K30_REF) <= DECISION_MARGIN

    if beats_nb1553_K30:
        verdict = f"K{best_K}_BEATS_NB1553_AVALON_K30_NEW_PRE_UNBLIND_CANDIDATE"
    elif flat_vs_nb1553_K30 and beats_anchor:
        verdict = f"K{best_K}_FLAT_VS_NB1553_AVALON_K30_TIE_AT_MARGIN"
    elif beats_anchor:
        verdict = f"K{best_K}_BEATS_ANCHOR_BUT_WORSE_THAN_NB1553_AVALON_K30"
    elif abs(best_rae - rae_anchor) < DECISION_MARGIN:
        verdict = f"K{best_K}_FLAT_VS_ANCHOR_NO_RESIDUAL_LIFT"
    else:
        verdict = f"K{best_K}_HURTS_ANCHOR"

    print("\n" + "-" * 78)
    print("BEST K")
    print("-" * 78)
    print(f"   best K                       = {best_K}")
    print(f"   best mean-bag RAE            = {best_rae:.4f}")
    print(f"   anchor (chemprop_aux)        = {rae_anchor:.4f}")
    print(f"   nb1553 Avalon K=30 ref       = {NB1553_AVALON_K30_REF:.4f}")
    print(f"   d_vs_anchor                  = {best_rae - rae_anchor:+.4f}")
    print(f"   d_vs_nb1553_K30              = {best_rae - NB1553_AVALON_K30_REF:+.4f}")
    print(f"   beats_anchor                 = {beats_anchor}")
    print(f"   beats_nb1553_K30             = {beats_nb1553_K30}")
    print(f"   flat_vs_nb1553_K30           = {flat_vs_nb1553_K30}")
    print(f"   verdict                      = {verdict}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_best_K_oof.npy", best_oof)
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_best_K_oof.npy'}  K={best_K}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "feature_family": "Avalon",
        "data_source": "local_chembl_caches_union",
        "n_chembl_pool": int(len(pool)),
        "n_unb": n_unb,
        "n_avalon_bits": n_av,
        "avalon_bit_density_unb": float(X_av_unb.mean()),
        "avalon_const_cols": int((X_av_unb.var(axis=0) == 0).sum()),
        "avalon_nonzero_imp_bits": n_nonzero_imp,
        "shap_importance_source": imp_src,
        "pred_chembl_importance": float(imp_full[n_av]),
        "sim_importance": float(imp_full[n_av + 1]),
        "feat_dim_full": int(feat_dim_full),
        "K_grid": K_GRID,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "lgbm_max_depth": 3,
        "lgbm_num_leaves": 7,
        "lgbm_n_estimators": 80,
        "lgbm_learning_rate": 0.05,
        "lgbm_min_child_samples": 20,
        "lgbm_huber_alpha": 1.0,
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "k_records": k_records,
        "best_K": best_K,
        "best_rae_mean_bag": best_rae,
        "best_delta_vs_anchor": best_rae - rae_anchor,
        "best_delta_vs_nb1553_K30": best_rae - NB1553_AVALON_K30_REF,
        "best_beats_anchor": bool(beats_anchor),
        "best_beats_nb1553_K30": bool(beats_nb1553_K30),
        "best_flat_vs_nb1553_K30": bool(flat_vs_nb1553_K30),
        "verdict": verdict,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "nb1553_avalon_K30_ref": NB1553_AVALON_K30_REF,
        "nb1392_K30_ref_nb1070_anchor": NB1392_K30_REF,
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
        "n_chembl_pool", "n_avalon_bits", "shap_importance_source",
        "K_grid",
        "rae_anchor_chemprop_aux",
        "best_K", "best_rae_mean_bag",
        "best_delta_vs_anchor", "best_delta_vs_nb1553_K30",
        "best_beats_anchor", "best_beats_nb1553_K30",
        "best_flat_vs_nb1553_K30",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
    print("\n  K-grid table (K, mean_bag, median_bag, d_vs_nb1553_K30):")
    for rec in res["k_records"]:
        print(f"    K={rec['K']:3d}  mean_bag={rec['rae_mean_bag']:.4f}  "
              f"median_bag={rec['rae_median_bag']:.4f}  "
              f"d_vs_nb1553_K30={rec['delta_mean_bag_vs_nb1553_K30']:+.4f}")
