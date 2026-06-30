"""nb1472 -- PRE-unblind 3-way SHAP-pruned residual blend.

Combines three SHAP-pruned residual learners, ALL anchored to chemprop_aux
(PRE-unblind te file on 513), then naive 1/3 mean blend.

Parallel construction to nb1411 (which used nb1070 POST-unblind anchor):
    A AtomPair-30 + ChEMBL  (32 cols)      <-> nb1373 / nb1460
    B MACCS-20  + ChEMBL    (22 cols)      <-> nb1352
    C Mordred-30 + ChEMBL   (32 cols)      <-> nb1364

But each residual learner is re-fit with anchor = chemprop_aux:
    residual = y_unb - chemprop_aux_te[unb_idx]

PROTOCOL:
    1. Anchor = chemprop_aux_te[unb_idx]  (PRE-unblind, in_RAE 0.6216).
    2. ChEMBL PXR kNN-5 (Tanimoto on Morgan-2048): same union as nb1352 /
       nb1364 / nb1373 / nb1460.
    3. For each FAMILY in [AtomPair-2048, MACCS-167, Mordred-1533]:
         a. Build FULL feature matrix on 253 = family + pred_chembl + sim.
         b. Train 1 seed-0 shallow LGBM Huber on residual; pull SHAP
            importance (TreeExplainer; fallback gain).
         c. Pick top-K (30 / 20 / 30) family columns by importance.
         d. PRUNED matrix = top-K family + pred_chembl + sim.
         e. 5-seed bag (seeds [0,1,7,42,137]), KFold(n=5) cross-fit per
            seed on shallow LGBM Huber (depth=3, num_leaves=7, n_est=80,
            lr=0.05, huber_alpha=1.0, min_child_samples=20).
         f. Mean-bag pred_corr = anchor + mean_bag(residual_oof).
         g. Pool RAE.
    4. Naive 1/3 mean blend across the 3 mean-bag corrected OOFs.
    5. Pool RAE.
    6. Verdict at 0.003 margin vs:
         - nb1460 (0.5550 AtomPair-only chemprop_aux anchor)
         - nb1411 (0.5037 nb1070-anchor 3-way PRIMARY)

Outputs:
    scripts/nb1472_preunblind_3way.py            (this file)
    data/processed/nb1472_summary.json
    data/processed/nb1472_mean_oof.npy            (253,) float32
    data/processed/nb1472_per_family_mean_bag.npy (3, 253) float32
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

TAG = "nb1472"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")

EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

CHEMPROP_AUX_REF = 0.6216
NB1460_REF = 0.5550        # AtomPair-only chemprop_aux anchor
NB1411_REF = 0.5037        # 3-way nb1070-anchor PRIMARY (POST-unblind)
DECISION_MARGIN = 0.003

TOP_K = {"AtomPair": 30, "MACCS": 20, "Mordred": 30}


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
    """Same union as nb1242 / nb1352 / nb1364 / nb1373 / nb1460."""
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


def _load_mordred_test(n_test_expected: int) -> np.ndarray:
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(
            f"Mordred cache missing -- run nb1030 first ({mte_p})"
        )
    X_te_m = np.load(mte_p).astype(np.float32)
    if X_te_m.shape[0] != n_test_expected:
        raise ValueError(
            f"Mordred test shape mismatch: {X_te_m.shape} vs n_test={n_test_expected}"
        )
    X_te_m = np.where(np.isfinite(X_te_m), X_te_m, np.nan).astype(np.float32)
    col_med = np.nanmedian(X_te_m, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X_te_m)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X_te_m[idx_r, idx_c] = col_med[idx_c]
    return X_te_m


def _load_family_te(family: str, n_test: int) -> np.ndarray:
    if family == "AtomPair":
        p = ATOMPAIR_TE_PATH
        if not p.exists():
            raise FileNotFoundError(f"AtomPair cache missing: {p}")
        X = np.load(p)
        if X.shape[0] != n_test:
            raise ValueError(f"AtomPair shape mismatch: {X.shape}")
        return X.astype(np.float32)
    if family == "MACCS":
        p = MACCS_TE_PATH
        if not p.exists():
            raise FileNotFoundError(f"MACCS cache missing: {p}")
        X = np.load(p)
        if X.shape[0] != n_test:
            raise ValueError(f"MACCS shape mismatch: {X.shape}")
        return X.astype(np.float32)
    if family == "Mordred":
        return _load_mordred_test(n_test)
    raise ValueError(f"unknown family: {family}")


def _run_family(family: str, X_fam_unb: np.ndarray,
                pred_chembl_unb: np.ndarray, mean_sim_unb: np.ndarray,
                anchor: np.ndarray, y_unb: np.ndarray, residual: np.ndarray,
                top_k: int) -> dict:
    print("\n" + "=" * 78)
    print(f"FAMILY = {family}   X_fam_unb shape = {X_fam_unb.shape}   "
          f"top_k = {top_k}")
    print("=" * 78)
    n_fam = int(X_fam_unb.shape[1])
    X_full = np.concatenate(
        [
            X_fam_unb,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim_full = X_full.shape[1]
    print(f"   FULL feature matrix: {X_full.shape}  "
          f"({family}-{n_fam} + pred_chembl + sim)")

    imp_full, imp_src = _compute_shap_importance(X_full, residual, seed=0)
    print(f"   importance source = {imp_src}")
    print(f"   pred_chembl importance = {imp_full[n_fam]:.4f}")
    print(f"   sim importance         = {imp_full[n_fam + 1]:.4f}")

    fam_imp = imp_full[:n_fam]
    top_k_eff = min(top_k, n_fam)
    top_order = np.argsort(-fam_imp)
    top_idx = top_order[:top_k_eff].astype(int)
    top_idx_sorted = np.sort(top_idx)
    top_imp = fam_imp[top_idx]
    n_nonzero_imp = int((fam_imp > 0).sum())
    print(f"   {family} cols with nonzero importance: {n_nonzero_imp}/{n_fam}")
    print(f"   top-{top_k_eff} {family} indices (sorted asc): "
          f"{top_idx_sorted.tolist()}")

    X_fam_pruned = X_fam_unb[:, top_idx]
    X_pruned = np.concatenate(
        [
            X_fam_pruned,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim_pruned = X_pruned.shape[1]
    print(f"   PRUNED feature matrix: {X_pruned.shape}")

    n_unb = len(y_unb)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae = []
    print(f"\n   PER-SEED RESIDUAL CROSS-FIT (PRUNED, dim={feat_dim_pruned})")
    for i, s in enumerate(RESID_SEEDS):
        resid_oof_s = _residual_cross_fit_one_seed(X_pruned, residual, s)
        pred_corr_s = anchor + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        print(f"     seed {s:3d}: rae = {rae_s:.4f}  "
              f"|resid_oof|.std = {resid_oof_s.std():.3f}")

    mean_bag_oof = per_seed_corrected.mean(axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_per_seed_mean = float(np.mean(per_seed_rae))
    print(f"   per-seed mean RAE = {rae_per_seed_mean:.4f}  "
          f"mean_bag pooled RAE = {rae_mean_bag:.4f}")

    return {
        "family": family,
        "n_fam_bits": n_fam,
        "top_k": int(top_k_eff),
        "top_idx_ranked": [int(b) for b in top_idx.tolist()],
        "top_imp_ranked": [float(v) for v in top_imp.tolist()],
        "top_idx_sorted_asc": [int(b) for b in top_idx_sorted.tolist()],
        "shap_source": imp_src,
        "pred_chembl_importance": float(imp_full[n_fam]),
        "sim_importance": float(imp_full[n_fam + 1]),
        "n_nonzero_imp": n_nonzero_imp,
        "feat_dim_full": int(feat_dim_full),
        "feat_dim_pruned": int(feat_dim_pruned),
        "per_seed_rae": per_seed_rae,
        "rae_per_seed_mean": rae_per_seed_mean,
        "rae_mean_bag": rae_mean_bag,
        "mean_bag_oof": mean_bag_oof.astype(np.float32),
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- PRE-unblind 3-way: chemprop_aux anchor + "
          f"AtomPair-30 + MACCS-20 + Mordred-30")
    print(f"          seeds = {RESID_SEEDS}  folds = {RESID_FOLDS}")
    print(f"          refs: chemprop_aux={CHEMPROP_AUX_REF:.4f}  "
          f"nb1460={NB1460_REF:.4f}  nb1411={NB1411_REF:.4f}")
    print(f"          decision margin = {DECISION_MARGIN}")
    print("=" * 78)

    # ---- Truth + indices ----
    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    # ---- Anchor (PRE-unblind chemprop_aux) ----
    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"chemprop_aux te file missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(f"anchor shape mismatch: {te_anchor_513.shape}")
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- ChEMBL pool + kNN ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL + kNN feature build")
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
    print(f"   pred_chembl_pec50 (unb) mean={pred_chembl_unb.mean():.3f}  "
          f"std={pred_chembl_unb.std():.3f}")
    print(f"   mean_sim (unb)         mean={mean_sim_unb.mean():.3f}")

    # ---- Run 3 families ----
    families = ["AtomPair", "MACCS", "Mordred"]
    fam_results: list[dict] = []
    for family in families:
        X_fam_te = _load_family_te(family, n_test)
        X_fam_unb = X_fam_te[unb_idx].astype(np.float32)
        result = _run_family(
            family=family,
            X_fam_unb=X_fam_unb,
            pred_chembl_unb=pred_chembl_unb,
            mean_sim_unb=mean_sim_unb,
            anchor=anchor,
            y_unb=y_unb,
            residual=residual,
            top_k=TOP_K[family],
        )
        fam_results.append(result)

    # ---- Naive 1/3 mean blend ----
    print("\n" + "=" * 78)
    print("3-WAY NAIVE 1/3 MEAN BLEND")
    print("=" * 78)
    per_family_mean_bag = np.stack(
        [r["mean_bag_oof"] for r in fam_results], axis=0
    )
    mean_oof = per_family_mean_bag.mean(axis=0)
    rae_mean = float(rae(y_unb, mean_oof))

    for r in fam_results:
        print(f"   {r['family']:<10s} mean_bag RAE = {r['rae_mean_bag']:.4f}")
    print(f"   naive 1/3 mean blend RAE = {rae_mean:.4f}")
    print(f"   anchor (chemprop_aux)    = {rae_anchor:.4f}")
    print(f"   nb1460 ref               = {NB1460_REF:.4f}")
    print(f"   nb1411 ref               = {NB1411_REF:.4f}")

    delta_mean_vs_anchor = rae_mean - rae_anchor
    delta_mean_vs_nb1460 = rae_mean - NB1460_REF
    delta_mean_vs_nb1411 = rae_mean - NB1411_REF

    beats_anchor = rae_mean < rae_anchor - DECISION_MARGIN
    beats_nb1460 = rae_mean < NB1460_REF - DECISION_MARGIN
    flat_vs_nb1460 = abs(rae_mean - NB1460_REF) < DECISION_MARGIN
    beats_nb1411 = rae_mean < NB1411_REF - DECISION_MARGIN
    flat_vs_nb1411 = abs(rae_mean - NB1411_REF) < DECISION_MARGIN

    if beats_nb1460 and beats_nb1411:
        verdict = "PRE_UNBLIND_3WAY_BEATS_NB1460_AND_NB1411_NEW_PRIMARY"
    elif beats_nb1460:
        verdict = "PRE_UNBLIND_3WAY_BEATS_NB1460_PRE_UNBLIND_CANDIDATE"
    elif flat_vs_nb1460:
        verdict = "PRE_UNBLIND_3WAY_FLAT_VS_NB1460"
    elif beats_anchor:
        verdict = "PRE_UNBLIND_3WAY_BEATS_ANCHOR_BUT_WORSE_THAN_NB1460"
    elif abs(rae_mean - rae_anchor) < DECISION_MARGIN:
        verdict = "PRE_UNBLIND_3WAY_FLAT_VS_ANCHOR"
    else:
        verdict = "PRE_UNBLIND_3WAY_HURTS_ANCHOR"

    print(f"\n   d_vs_anchor  = {delta_mean_vs_anchor:+.4f}")
    print(f"   d_vs_nb1460  = {delta_mean_vs_nb1460:+.4f}")
    print(f"   d_vs_nb1411  = {delta_mean_vs_nb1411:+.4f}")
    print(f"   beats_nb1460 = {beats_nb1460}   flat_vs_nb1460 = {flat_vs_nb1460}")
    print(f"   beats_nb1411 = {beats_nb1411}   flat_vs_nb1411 = {flat_vs_nb1411}")
    print(f"   verdict      = {verdict}")
    print("=" * 78)

    # ---- Save ----
    np.save(DATA_PROCESSED / f"{TAG}_mean_oof.npy",
            mean_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_per_family_mean_bag.npy",
            per_family_mean_bag.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_mean_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_per_family_mean_bag.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "data_source": "local_chembl_caches_union",
        "n_chembl_pool": int(len(pool)),
        "n_unb": int(n_unb),
        "n_test": int(n_test),
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "top_k_config": TOP_K,
        "families": [
            {
                "family": r["family"],
                "n_fam_bits": r["n_fam_bits"],
                "top_k": r["top_k"],
                "shap_source": r["shap_source"],
                "n_nonzero_imp": r["n_nonzero_imp"],
                "pred_chembl_importance": r["pred_chembl_importance"],
                "sim_importance": r["sim_importance"],
                "feat_dim_full": r["feat_dim_full"],
                "feat_dim_pruned": r["feat_dim_pruned"],
                "top_idx_ranked": r["top_idx_ranked"],
                "top_imp_ranked": r["top_imp_ranked"],
                "top_idx_sorted_asc": r["top_idx_sorted_asc"],
                "per_seed_rae": r["per_seed_rae"],
                "rae_per_seed_mean": r["rae_per_seed_mean"],
                "rae_mean_bag": r["rae_mean_bag"],
            }
            for r in fam_results
        ],
        "per_family_mean_bag_rae": {
            r["family"]: r["rae_mean_bag"] for r in fam_results
        },
        "rae_mean_blend": rae_mean,
        "delta_mean_vs_anchor": delta_mean_vs_anchor,
        "delta_mean_vs_nb1460": delta_mean_vs_nb1460,
        "delta_mean_vs_nb1411": delta_mean_vs_nb1411,
        "beats_anchor": bool(beats_anchor),
        "beats_nb1460": bool(beats_nb1460),
        "flat_vs_nb1460": bool(flat_vs_nb1460),
        "beats_nb1411": bool(beats_nb1411),
        "flat_vs_nb1411": bool(flat_vs_nb1411),
        "verdict": verdict,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "nb1460_ref": NB1460_REF,
        "nb1411_ref": NB1411_REF,
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
        "n_unb", "n_chembl_pool", "rae_anchor_chemprop_aux",
        "per_family_mean_bag_rae",
        "rae_mean_blend",
        "delta_mean_vs_anchor",
        "delta_mean_vs_nb1460",
        "delta_mean_vs_nb1411",
        "beats_anchor", "beats_nb1460", "flat_vs_nb1460",
        "beats_nb1411", "flat_vs_nb1411",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
