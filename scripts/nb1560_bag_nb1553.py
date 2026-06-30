"""nb1560 -- Outer-bag VALIDATION of nb1553 5-way K-tuned residual blend.

HYPOTHESIS:
    nb1553 is the 5-way variant of nb1550 (4-way) with Avalon added back and the
    five K-slots all set to their per-family K-grid winners:
        AtomPair      K=25   (nb1524 best)
        MACCS         K=20   (nb1424 user-pinned)
        Mordred       K=20   (nb1523 best)
        ChempropEmbed K=20   (nb1541 best)
        Avalon        K=30   (nb1433 user-pinned)
    Anchor = chemprop_aux te[unb_idx] (PRE-unblind, in_RAE 0.6216 on 253).
    Per-family LGBM Huber (depth=3, leaves=7, n_est=80, lr=0.05, alpha=1.0) on
    [top-K SHAP-pruned feats + pred_chembl + sim], 5-seed bag, KFold(n=5) cross-fit
    per inner seed, mean-bag corrected OOF.
    nb1553_o = naive 1/5 mean over the 5 family-corrected OOFs.

PROTOCOL
    1. For each outer seed o in {0, 1, 7, 42, 137}:
         a. Build 5 family pruned matrices using top-K SHAP indices rebuilt from
            (X_full, residual) at SHAP-seed = 0 (identical to nb1553).
         b. For each family: 5-inner-seed bag with inner seeds = [o*1000 + s]
            for s in [0,1,2,3,4]; KFold(n=5, shuffle=True, random_state=inner)
            cross-fit residual; mean-bag corrected OOF.
         c. nb1553_o = naive 1/5 mean over the 5 family-corrected mean-bag OOFs.
         d. Per-outer pooled RAE = rae(y_unb, nb1553_o).
    2. Row-level BoB MEAN + MEDIAN across the 5 nb1553_o vectors.
    3. Verdict NB1553_REPRODUCES iff
           |per_outer_mean - nb1553_ref| < 0.003.
       nb1553_ref = rae_naive_mean_blend from nb1553_summary.json (= 0.5225).

Outputs:
    scripts/nb1560_bag_nb1553.py                  (this file)
    data/processed/nb1560_summary.json
    data/processed/nb1560_bob_mean_oof.npy        (253,) float32
    data/processed/nb1560_bob_median_oof.npy      (253,) float32
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

TAG = "nb1560"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

NB1553_SUMMARY = DATA_PROCESSED / "nb1553_summary.json"
REPRODUCE_MARGIN = 0.003

OUTER_SEEDS = [0, 1, 7, 42, 137]
INNER_SEEDS_PER_OUTER = 5            # inner seeds = [o*1000 + s] for s in 0..4
RESID_FOLDS = 5

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")

EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

FAMILIES = ["AtomPair", "MACCS", "Mordred", "ChempropEmbed", "Avalon"]

# K config matches nb1553 exactly.
TOP_K_CONFIG = {
    "AtomPair": 25,
    "MACCS": 20,
    "Mordred": 20,
    "ChempropEmbed": 20,
    "Avalon": 30,
}


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
    """Same union as nb1553."""
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
    if family == "ChempropEmbed":
        p = CHEMPROP_EMBED_TE_PATH
        if not p.exists():
            raise FileNotFoundError(f"Chemprop embed cache missing: {p}")
        X = np.load(p).astype(np.float32)
        if X.shape[0] != n_test:
            raise ValueError(f"Chemprop embed shape mismatch: {X.shape}")
        X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
        return X
    if family == "Avalon":
        p = AVALON_TE_PATH
        if not p.exists():
            raise FileNotFoundError(f"Avalon cache missing: {p}")
        X = np.load(p)
        if X.shape[0] != n_test:
            raise ValueError(f"Avalon shape mismatch: {X.shape}")
        return X.astype(np.float32)
    raise ValueError(f"unknown family: {family}")


def _build_family_pruned_matrix(family: str, X_fam_unb: np.ndarray,
                                pred_chembl_unb: np.ndarray,
                                mean_sim_unb: np.ndarray,
                                residual: np.ndarray,
                                top_k: int) -> tuple[np.ndarray, dict]:
    """Replicates nb1553 _run_family up to PRUNED matrix construction."""
    n_fam = int(X_fam_unb.shape[1])
    X_full = np.concatenate(
        [
            X_fam_unb,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    imp_full, imp_src = _compute_shap_importance(X_full, residual, seed=0)
    fam_imp = imp_full[:n_fam]
    top_k_eff = min(top_k, n_fam)
    top_order = np.argsort(-fam_imp)
    top_idx = top_order[:top_k_eff].astype(int)
    X_fam_pruned = X_fam_unb[:, top_idx]
    X_pruned = np.concatenate(
        [
            X_fam_pruned,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    meta = {
        "family": family,
        "n_fam_bits": n_fam,
        "top_k": int(top_k_eff),
        "feat_dim_pruned": int(X_pruned.shape[1]),
        "shap_source": imp_src,
        "top_idx_sorted_asc": [int(b) for b in np.sort(top_idx).tolist()],
    }
    return X_pruned, meta


def main() -> dict:
    t0 = time.time()
    if not NB1553_SUMMARY.exists():
        raise FileNotFoundError(f"missing {NB1553_SUMMARY} -- run nb1553 first")
    with open(NB1553_SUMMARY) as f:
        sum_1553 = json.load(f)
    nb1553_ref_naive = float(sum_1553["rae_naive_mean_blend"])
    nb1553_ref_slsqp = float(sum_1553["rae_slsqp_crossfit"])
    nb1553_ref_best = float(sum_1553["rae_best"])
    nb1553_ref_variant = str(sum_1553["best_variant"])
    nb1553_per_family_rae = dict(sum_1553["per_family_mean_bag_rae"])

    print("=" * 78)
    print(f"{TAG} -- OUTER-BAG VALIDATION of nb1553 5-way K-tuned")
    print(f"         outer seeds        = {OUTER_SEEDS}")
    print(f"         inner seeds        = [o*1000 + s for s in 0..4]  per outer")
    print(f"         top-K config       = {TOP_K_CONFIG}")
    print(f"         nb1553_ref_naive   = {nb1553_ref_naive:.4f}  "
          f"(per-outer reference)")
    print(f"         nb1553_ref_slsqp   = {nb1553_ref_slsqp:.4f}")
    print(f"         nb1553_ref_best    = {nb1553_ref_best:.4f}  "
          f"(variant = {nb1553_ref_variant})")
    print(f"         reproduce margin   = {REPRODUCE_MARGIN}")
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
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- ChEMBL pool + kNN feature build ----
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

    # ---- Build 5 per-family PRUNED matrices (SHAP seed=0, top-K from nb1553) ----
    print("\n" + "-" * 78)
    print("BUILD 5 PER-FAMILY PRUNED MATRICES  (top-K via SHAP seed=0)")
    print("-" * 78)
    family_matrices: dict[str, np.ndarray] = {}
    family_meta: dict[str, dict] = {}
    for fam in FAMILIES:
        X_fam_te = _load_family_te(fam, n_test)
        X_fam_unb = X_fam_te[unb_idx].astype(np.float32)
        X_pruned, meta = _build_family_pruned_matrix(
            family=fam,
            X_fam_unb=X_fam_unb,
            pred_chembl_unb=pred_chembl_unb,
            mean_sim_unb=mean_sim_unb,
            residual=residual,
            top_k=TOP_K_CONFIG[fam],
        )
        family_matrices[fam] = X_pruned
        family_meta[fam] = meta
        print(f"   {fam:<14s} fam_bits={meta['n_fam_bits']:4d}  "
              f"top_k={meta['top_k']:3d}  PRUNED dim={meta['feat_dim_pruned']:3d}  "
              f"shap_src={meta['shap_source']}")

    # ---- Outer-bag rebuild of nb1553_o ------------------------------------
    print("\n" + "=" * 78)
    print("OUTER-BAG x [5 family LGBM Huber 5-seed mean bags, naive 1/5 mean]")
    print("=" * 78)
    n_outer = len(OUTER_SEEDS)

    outer_nb1553 = np.zeros((n_outer, n_unb), dtype=np.float64)
    per_outer_records = []

    for oi, o in enumerate(OUTER_SEEDS):
        t_outer = time.time()
        inner_seeds = [int(o) * 1000 + int(s) for s in range(INNER_SEEDS_PER_OUTER)]
        print(f"\n   --- outer seed {o}  inner seeds = {inner_seeds} ---")

        family_corrected = np.zeros((len(FAMILIES), n_unb), dtype=np.float64)
        family_rae: dict[str, float] = {}
        family_per_seed_rae: dict[str, list[float]] = {}

        for fi, fam in enumerate(FAMILIES):
            ts = time.time()
            X_fam_pruned = family_matrices[fam]
            per_seed_corrected = np.zeros(
                (INNER_SEEDS_PER_OUTER, n_unb), dtype=np.float64
            )
            per_seed_rae_fam: list[float] = []
            for si, s_inner in enumerate(inner_seeds):
                resid_oof_s = _residual_cross_fit_one_seed(
                    X_fam_pruned, residual, seed=int(s_inner)
                )
                pred_corr_s = anchor + resid_oof_s
                per_seed_corrected[si] = pred_corr_s
                rae_s = float(rae(y_unb, pred_corr_s))
                per_seed_rae_fam.append(rae_s)
            mean_bag_fam = per_seed_corrected.mean(axis=0)
            rae_mean_bag = float(rae(y_unb, mean_bag_fam))
            family_corrected[fi] = mean_bag_fam
            family_rae[fam] = rae_mean_bag
            family_per_seed_rae[fam] = per_seed_rae_fam
            print(f"     [nb1553] outer {o:3d}  family {fam:<14s} "
                  f"per-seed RAE = [{', '.join(f'{r:.4f}' for r in per_seed_rae_fam)}]  "
                  f"mean_bag = {rae_mean_bag:.4f}  "
                  f"wall = {time.time() - ts:.1f}s")

        nb1553_o = family_corrected.mean(axis=0)
        outer_nb1553[oi] = nb1553_o
        rae_nb1553_o = float(rae(y_unb, nb1553_o))
        print(f"     [nb1553_o = naive 1/5 mean over 5 families] pooled RAE = "
              f"{rae_nb1553_o:.4f}")

        per_outer_records.append({
            "outer_seed": int(o),
            "inner_seeds": inner_seeds,
            "rae_nb1553_per_family_mean_bag": family_rae,
            "rae_nb1553_per_family_per_seed": family_per_seed_rae,
            "rae_nb1553_o": rae_nb1553_o,
            "wall_sec": round(time.time() - t_outer, 2),
        })
        print(f"   outer {o:3d}  (outer wall = "
              f"{time.time() - t_outer:.1f}s)")

    # ---- Per-outer summary ----
    per_outer_rae_nb1553: list[float] = [
        rec["rae_nb1553_o"] for rec in per_outer_records
    ]
    per_outer_arr = np.array(per_outer_rae_nb1553)
    per_outer_mean = float(per_outer_arr.mean())
    per_outer_std = float(per_outer_arr.std())
    per_outer_min = float(per_outer_arr.min())
    per_outer_max = float(per_outer_arr.max())
    per_outer_median = float(np.median(per_outer_arr))

    # ---- BoB row-level aggregations across 5 nb1553_o vectors ----
    bob_mean_oof = outer_nb1553.mean(axis=0)
    bob_median_oof = np.median(outer_nb1553, axis=0)
    rae_bob_mean = float(rae(y_unb, bob_mean_oof))
    rae_bob_median = float(rae(y_unb, bob_median_oof))

    print("\n" + "=" * 78)
    print("OUTER-BAG SUMMARY")
    print("=" * 78)
    print(f"   per-outer nb1553_o RAE list = "
          f"[{', '.join(f'{r:.4f}' for r in per_outer_rae_nb1553)}]")
    print(f"   per-outer nb1553_o mean    = {per_outer_mean:.4f}")
    print(f"   per-outer nb1553_o std     = {per_outer_std:.4f}")
    print(f"   per-outer nb1553_o min     = {per_outer_min:.4f}")
    print(f"   per-outer nb1553_o max     = {per_outer_max:.4f}")
    print(f"   per-outer nb1553_o median  = {per_outer_median:.4f}")
    print(f"   BoB MEAN   pooled RAE = {rae_bob_mean:.4f}  "
          f"(d vs nb1553_ref_naive = {rae_bob_mean - nb1553_ref_naive:+.4f})")
    print(f"   BoB MEDIAN pooled RAE = {rae_bob_median:.4f}  "
          f"(d vs nb1553_ref_naive = {rae_bob_median - nb1553_ref_naive:+.4f})")

    # ---- Pearson sanity vs anchor and nb1553_best_oof ----
    def _pearson_vs(path: Path, vec: np.ndarray):
        if not path.exists():
            return None
        oof = np.load(path).astype(np.float64)
        if oof.shape[0] != n_unb:
            return None
        return float(np.corrcoef(vec, oof)[0, 1])

    pearson_bobmean_vs_nb1553_best = _pearson_vs(
        DATA_PROCESSED / "nb1553_best_oof.npy", bob_mean_oof
    )
    pearson_bobmean_vs_anchor = float(np.corrcoef(bob_mean_oof, anchor)[0, 1])
    if pearson_bobmean_vs_nb1553_best is not None:
        print(f"   Pearson(bob_mean, nb1553_best)     = "
              f"{pearson_bobmean_vs_nb1553_best:.4f}")
    print(f"   Pearson(bob_mean, anchor)          = "
          f"{pearson_bobmean_vs_anchor:.4f}")

    # ---- Verdict (per-outer-mean vs nb1553_ref_naive within margin) ----
    delta_per_outer = per_outer_mean - nb1553_ref_naive
    reproduces = abs(delta_per_outer) < REPRODUCE_MARGIN
    if reproduces:
        verdict = "NB1553_REPRODUCES"
    elif per_outer_mean < nb1553_ref_naive - REPRODUCE_MARGIN:
        verdict = "NB1553_OPTIMISTIC_OUTER_BAG_BETTER"
    else:
        verdict = "NB1553_LUCKY_SEED_OUTER_BAG_WORSE"

    print("\n" + "-" * 78)
    print(f"VERDICT (per-outer nb1553_o mean within {REPRODUCE_MARGIN} of "
          f"nb1553_ref_naive = {nb1553_ref_naive:.4f}):")
    print(f"   per-outer-mean = {per_outer_mean:.4f}   "
          f"(d vs ref = {delta_per_outer:+.4f})")
    print(f"   verdict        = {verdict}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_bob_mean_oof.npy",
            bob_mean_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_bob_median_oof.npy",
            bob_median_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_bob_mean_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_bob_median_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "data_source": "AtomPair-cache + MACCS-cache + Mordred-cached_nb1030 + "
                       "ChempropEmbed-cache + Avalon-cache + "
                       "local_chembl_caches_union",
        "top_k_config": TOP_K_CONFIG,
        "families_order": FAMILIES,
        "nb1553_model": "LGBM(huber alpha=1.0, d3, leaves=7, n_est=80, lr=0.05)",
        "nb1553_family_aggregation": "naive_1_5_mean",
        "inner_seed_recipe": "o*1000 + s for s in 0..4",
        "n_unb": n_unb,
        "n_test": int(n_test),
        "n_chembl_pool": int(len(pool)),
        "outer_seeds": OUTER_SEEDS,
        "inner_seeds_per_outer": INNER_SEEDS_PER_OUTER,
        "resid_folds": RESID_FOLDS,
        "family_meta": family_meta,
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "nb1553_ref_naive": nb1553_ref_naive,
        "nb1553_ref_slsqp": nb1553_ref_slsqp,
        "nb1553_ref_best": nb1553_ref_best,
        "nb1553_ref_best_variant": nb1553_ref_variant,
        "nb1553_per_family_rae_ref": nb1553_per_family_rae,
        "reproduce_margin": REPRODUCE_MARGIN,
        "per_outer_records": per_outer_records,
        "per_outer_rae_nb1553": per_outer_rae_nb1553,
        "per_outer_mean": per_outer_mean,
        "per_outer_std": per_outer_std,
        "per_outer_min": per_outer_min,
        "per_outer_max": per_outer_max,
        "per_outer_median": per_outer_median,
        "rae_bob_mean": rae_bob_mean,
        "rae_bob_median": rae_bob_median,
        "delta_per_outer_mean_vs_nb1553_ref_naive": delta_per_outer,
        "delta_bob_mean_vs_nb1553_ref_naive": rae_bob_mean - nb1553_ref_naive,
        "delta_bob_median_vs_nb1553_ref_naive": rae_bob_median - nb1553_ref_naive,
        "reproduces": bool(reproduces),
        "pearson_bobmean_vs_nb1553_best": pearson_bobmean_vs_nb1553_best,
        "pearson_bobmean_vs_anchor": pearson_bobmean_vs_anchor,
        "verdict": verdict,
        "pre_unblind_clean": True,
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
        "n_unb", "n_test", "n_chembl_pool",
        "top_k_config",
        "outer_seeds",
        "inner_seeds_per_outer",
        "rae_anchor_chemprop_aux",
        "nb1553_ref_naive",
        "nb1553_ref_slsqp",
        "nb1553_ref_best",
        "nb1553_ref_best_variant",
        "per_outer_rae_nb1553",
        "per_outer_mean", "per_outer_std",
        "per_outer_min", "per_outer_max", "per_outer_median",
        "rae_bob_mean", "rae_bob_median",
        "delta_per_outer_mean_vs_nb1553_ref_naive",
        "delta_bob_mean_vs_nb1553_ref_naive",
        "delta_bob_median_vs_nb1553_ref_naive",
        "reproduces",
        "pearson_bobmean_vs_nb1553_best",
        "pearson_bobmean_vs_anchor",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
