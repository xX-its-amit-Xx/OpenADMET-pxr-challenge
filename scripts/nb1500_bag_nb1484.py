"""nb1500 -- Outer-bag VALIDATION of nb1484 (PRE-unblind 4-way naive 1/4 mean).

Independent confirmation pass extending nb1494. The 4-family SHAP-pruned
recipe of nb1484 (AtomPair-30 + MACCS-20 + Mordred-30 + ChempropEmbed-30,
anchored to chemprop_aux PRE-unblind te[unb_idx]) is rebuilt fresh per
outer seed, and the row-level mean / median across the 5 outer-seed
nb1484_o vectors is pooled.

PROTOCOL
    1. For each outer seed o in {0, 1, 7, 42, 137}, REBUILD the four
       residual learners on the PRUNED feature matrix
           X_pruned = [top-K(family) + pred_chembl + sim]
       where the family top-K (30/20/30/30 for
       AtomPair/MACCS/Mordred/ChempropEmbed) is taken as nb1484's
       published top-idx (pinned -- the SHAP step itself is seed=0
       deterministic so re-running it would not change the columns
       under nb1484's seed=0 SHAP fit).
    2. Inner seeds reparameterized per outer:
           inner_seeds(o) = [o*1000 + s for s in {0, 1, 7, 42, 137}].
       For each family build a 5-inner-seed mean-bag of shallow LGBM
       Huber residual learners (same hyperparams as nb1484: depth=3,
       num_leaves=7, n_est=80, lr=0.05, huber_alpha=1.0,
       min_child_samples=20).
    3. Per outer o, family-corrected OOF:
           fam_corrected_o = anchor + mean_bag_inner(residual_oof)
       and the per-outer 4-family naive 1/4 mean:
           nb1484_o = (1/4) * sum(fam_corrected_o for fam in 4 fams)
    4. Aggregate across the 5 outer seeds:
           bob_mean_oof   = row-mean   of {nb1484_o}_5
           bob_median_oof = row-median of {nb1484_o}_5
       Pool RAE for each.
    5. Verdict NB1484_REPRODUCES iff
           |per_outer_mean - 0.5257| < 0.003
       (where 0.5257 = nb1484 naive 1/4 mean cross-fit RAE,
       see data/processed/nb1484_summary.json::rae_naive_mean_blend).

Outputs:
    scripts/nb1500_bag_nb1484.py             (this file)
    data/processed/nb1500_summary.json
    data/processed/nb1500_bob_mean_oof.npy   (253,) float32
    data/processed/nb1500_bob_median_oof.npy (253,) float32
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

TAG = "nb1500"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"
NB1484_REF_NAIVE = 0.5257            # nb1484 naive 1/4 mean cross-fit
REPRODUCE_MARGIN = 0.003

OUTER_SEEDS = [0, 1, 7, 42, 137]
INNER_BASE_SEEDS = [0, 1, 7, 42, 137]
RESID_FOLDS = 5

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")

EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

FAMILIES = ["AtomPair", "MACCS", "Mordred", "ChempropEmbed"]
TOP_K = {"AtomPair": 30, "MACCS": 20, "Mordred": 30, "ChempropEmbed": 30}


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
    """Same union as nb1484."""
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
    raise ValueError(f"unknown family: {family}")


def _build_family_pruned_unblind(
    family: str,
    n_test: int,
    unb_idx: np.ndarray,
    pred_chembl_unb: np.ndarray,
    mean_sim_unb: np.ndarray,
    top_idx_pinned: np.ndarray,
) -> np.ndarray:
    """Build the (n_unb, top_k + 2) PRUNED feature matrix for a family using
    the SHAP top-idx pinned from nb1484 (nb1484 SHAP at seed=0 is
    deterministic; pinning makes outer-seed signal pure resid-fit noise)."""
    X_fam_te = _load_family_te(family, n_test)
    X_fam_unb = X_fam_te[unb_idx].astype(np.float32)
    X_fam_pruned = X_fam_unb[:, top_idx_pinned]
    X_pruned = np.concatenate(
        [
            X_fam_pruned,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    return X_pruned


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- OUTER-BAG VALIDATION of nb1484 "
          f"(PRE-unblind 4-way naive 1/4 mean on chemprop_aux anchor)")
    print(f"         outer seeds      = {OUTER_SEEDS}")
    print(f"         inner base seeds = {INNER_BASE_SEEDS}")
    print(f"         inner_seeds(o)   = [o*1000 + s for s in base]")
    print(f"         families         = {FAMILIES}")
    print(f"         top_k            = {TOP_K}")
    print(f"         target ref       = {NB1484_REF_NAIVE} "
          f"(margin = {REPRODUCE_MARGIN})")
    print("=" * 78)

    # ---- Step 1: Pull pinned top-idx per family from nb1484 ----
    if not NB1484_SUMMARY.exists():
        raise FileNotFoundError(f"nb1484 summary missing: {NB1484_SUMMARY}")
    with open(NB1484_SUMMARY) as f:
        sum_1484 = json.load(f)
    nb1484_naive = float(sum_1484["rae_naive_mean_blend"])
    nb1484_best = float(sum_1484["rae_best"])
    nb1484_best_variant = sum_1484["best_variant"]
    nb1484_slsqp = float(sum_1484["rae_slsqp_crossfit"])

    print(f"\n[load] nb1484 RAE naive 1/4 mean = {nb1484_naive:.6f}")
    print(f"[load] nb1484 RAE best           = {nb1484_best:.6f}  "
          f"(variant = {nb1484_best_variant})")
    print(f"[load] nb1484 RAE slsqp          = {nb1484_slsqp:.6f}")

    fam_top_idx: dict[str, np.ndarray] = {}
    for fam_rec in sum_1484["families"]:
        fam_name = fam_rec["family"]
        top_idx = np.asarray(fam_rec["top_idx_ranked"], dtype=int)
        fam_top_idx[fam_name] = top_idx
        print(f"   [pin] {fam_name:<14s} top-{len(top_idx)} idx pinned")
    for fam in FAMILIES:
        if fam not in fam_top_idx:
            raise KeyError(f"nb1484 summary missing family {fam}")

    # ---- Load truth + anchor (PRE-unblind chemprop_aux) ----
    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"\n[load] n_test={n_test}  n_unb={n_unb}")

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

    # ---- ChEMBL pool + kNN (same as nb1484) ----
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

    # ---- Build PRUNED feature matrices per family (pinned cols) ----
    print("\n" + "-" * 78)
    print("BUILD per-family PRUNED feature matrices (pinned SHAP cols)")
    print("-" * 78)
    X_pruned_by_fam: dict[str, np.ndarray] = {}
    for fam in FAMILIES:
        X_pruned = _build_family_pruned_unblind(
            family=fam,
            n_test=n_test,
            unb_idx=unb_idx,
            pred_chembl_unb=pred_chembl_unb,
            mean_sim_unb=mean_sim_unb,
            top_idx_pinned=fam_top_idx[fam],
        )
        X_pruned_by_fam[fam] = X_pruned
        print(f"   {fam:<14s} X_pruned shape = {X_pruned.shape}")

    # ---- Outer x family x inner cross-fit ----
    print("\n" + "=" * 78)
    print("OUTER x FAMILY x INNER LGBM HUBER RESIDUAL CROSS-FIT")
    print("=" * 78)
    n_outer = len(OUTER_SEEDS)
    n_inner = len(INNER_BASE_SEEDS)

    outer_blend = np.zeros((n_outer, n_unb), dtype=np.float64)
    per_outer_records = []
    per_outer_inner_seeds: list[list[int]] = []

    for oi, o in enumerate(OUTER_SEEDS):
        inner_seeds = [int(o * 1000 + s) for s in INNER_BASE_SEEDS]
        per_outer_inner_seeds.append(inner_seeds)
        t_outer = time.time()
        print(f"\n   --- outer seed {o}  inner seeds = {inner_seeds} ---")
        fam_corrected: dict[str, np.ndarray] = {}
        fam_records = []
        for fam in FAMILIES:
            X_pruned = X_pruned_by_fam[fam]
            inner_corrected = np.zeros((n_inner, n_unb), dtype=np.float64)
            inner_per_seed_rae: list[float] = []
            for ii, isd in enumerate(inner_seeds):
                ts = time.time()
                resid_oof_s = _residual_cross_fit_one_seed(X_pruned, residual, isd)
                pred_corr_s = anchor + resid_oof_s
                inner_corrected[ii] = pred_corr_s
                r_s = float(rae(y_unb, pred_corr_s))
                inner_per_seed_rae.append(r_s)
                print(f"     outer {o:3d}  fam {fam:<14s}  inner {isd:6d}: "
                      f"rae = {r_s:.4f}  wall = {time.time() - ts:.1f}s")
            mean_bag_fam = inner_corrected.mean(axis=0)
            rae_mean_bag_fam = float(rae(y_unb, mean_bag_fam))
            fam_corrected[fam] = mean_bag_fam
            fam_records.append({
                "family": fam,
                "inner_per_seed_rae": inner_per_seed_rae,
                "rae_per_seed_mean": float(np.mean(inner_per_seed_rae)),
                "rae_mean_bag": rae_mean_bag_fam,
            })
            print(f"     outer {o:3d}  fam {fam:<14s}  mean_bag pooled RAE "
                  f"= {rae_mean_bag_fam:.4f}")

        # naive 1/4 mean blend across 4 families for this outer seed
        blend_o = np.mean(
            np.stack([fam_corrected[fam] for fam in FAMILIES], axis=0),
            axis=0,
        )
        outer_blend[oi] = blend_o
        rae_blend_o = float(rae(y_unb, blend_o))
        per_outer_records.append({
            "outer_seed": int(o),
            "inner_seeds": inner_seeds,
            "families": fam_records,
            "rae_blend_4way_mean": rae_blend_o,
            "wall_sec": round(time.time() - t_outer, 2),
        })
        print(f"   outer {o:3d}  4-way naive 1/4 mean RAE = {rae_blend_o:.4f}  "
              f"(outer wall = {time.time() - t_outer:.1f}s)")

    per_outer_rae_blend: list[float] = [
        rec["rae_blend_4way_mean"] for rec in per_outer_records
    ]
    per_outer_arr = np.array(per_outer_rae_blend)
    per_outer_mean = float(per_outer_arr.mean())
    per_outer_std = float(per_outer_arr.std())
    per_outer_min = float(per_outer_arr.min())
    per_outer_max = float(per_outer_arr.max())
    per_outer_median = float(np.median(per_outer_arr))

    # ---- BoB row-level aggregations across 5 nb1484_o vectors ----
    bob_mean_oof = outer_blend.mean(axis=0)
    bob_median_oof = np.median(outer_blend, axis=0)
    rae_bob_mean = float(rae(y_unb, bob_mean_oof))
    rae_bob_median = float(rae(y_unb, bob_median_oof))

    print("\n" + "=" * 78)
    print("OUTER-BAG SUMMARY")
    print("=" * 78)
    print(f"   per-outer nb1484_o RAE list = "
          f"[{', '.join(f'{r:.4f}' for r in per_outer_rae_blend)}]")
    print(f"   per-outer mean   = {per_outer_mean:.4f}")
    print(f"   per-outer std    = {per_outer_std:.4f}")
    print(f"   per-outer min    = {per_outer_min:.4f}")
    print(f"   per-outer max    = {per_outer_max:.4f}")
    print(f"   per-outer median = {per_outer_median:.4f}")
    print(f"   BoB MEAN   pooled RAE = {rae_bob_mean:.4f}  "
          f"(d vs nb1484_naive = {rae_bob_mean - nb1484_naive:+.4f})")
    print(f"   BoB MEDIAN pooled RAE = {rae_bob_median:.4f}  "
          f"(d vs nb1484_naive = {rae_bob_median - nb1484_naive:+.4f})")

    # ---- Pearson sanity vs nb1484 best oof ----
    def _pearson_vs(path: Path, vec: np.ndarray):
        if not path.exists():
            return None
        oof = np.load(path).astype(np.float64)
        if oof.shape[0] != n_unb:
            return None
        return float(np.corrcoef(vec, oof)[0, 1])

    pearson_bobmean_vs_nb1484_best = _pearson_vs(
        DATA_PROCESSED / "nb1484_best_oof.npy", bob_mean_oof
    )
    pearson_bobmedian_vs_nb1484_best = _pearson_vs(
        DATA_PROCESSED / "nb1484_best_oof.npy", bob_median_oof
    )
    if pearson_bobmean_vs_nb1484_best is not None:
        print(f"   Pearson(bob_mean,   nb1484_best_oof) = "
              f"{pearson_bobmean_vs_nb1484_best:.4f}")
    if pearson_bobmedian_vs_nb1484_best is not None:
        print(f"   Pearson(bob_median, nb1484_best_oof) = "
              f"{pearson_bobmedian_vs_nb1484_best:.4f}")

    # ---- Verdict (per-outer-mean vs 0.5257 within 0.003) ----
    delta_per_outer = per_outer_mean - NB1484_REF_NAIVE
    reproduces = abs(delta_per_outer) < REPRODUCE_MARGIN
    if reproduces:
        verdict = "NB1484_REPRODUCES"
    elif per_outer_mean < NB1484_REF_NAIVE - REPRODUCE_MARGIN:
        verdict = "NB1484_OPTIMISTIC_OUTER_BAG_BETTER"
    else:
        verdict = "NB1484_LUCKY_SEED_OUTER_BAG_WORSE"

    print("\n" + "-" * 78)
    print(f"VERDICT (per-outer-mean within {REPRODUCE_MARGIN} of nb1484 "
          f"naive 1/4 mean ref = {NB1484_REF_NAIVE:.4f}):")
    print(f"   per-outer-mean = {per_outer_mean:.4f}   "
          f"(d vs ref = {delta_per_outer:+.4f})")
    print(f"   verdict = {verdict}")

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
        "data_source": "local_chembl_caches_union + family caches",
        "model_family": "LightGBM",
        "lgbm_objective": "huber",
        "lgbm_huber_alpha": 1.0,
        "lgbm_max_depth": 3,
        "lgbm_num_leaves": 7,
        "lgbm_n_estimators": 80,
        "lgbm_learning_rate": 0.05,
        "lgbm_min_child_samples": 20,
        "n_unb": n_unb,
        "n_test": int(n_test),
        "n_chembl_pool": int(len(pool)),
        "outer_seeds": OUTER_SEEDS,
        "inner_base_seeds": INNER_BASE_SEEDS,
        "per_outer_inner_seeds": per_outer_inner_seeds,
        "resid_folds": RESID_FOLDS,
        "families_order": FAMILIES,
        "top_k_config": TOP_K,
        "families_top_idx_pinned_from_nb1484": {
            fam: [int(b) for b in fam_top_idx[fam].tolist()]
            for fam in FAMILIES
        },
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "nb1484_rae_naive_mean_blend": nb1484_naive,
        "nb1484_rae_best": nb1484_best,
        "nb1484_best_variant": nb1484_best_variant,
        "nb1484_rae_slsqp_crossfit": nb1484_slsqp,
        "nb1484_ref_for_verdict": NB1484_REF_NAIVE,
        "nb1484_ref_kind": "naive_1_4_mean_blend",
        "reproduce_margin": REPRODUCE_MARGIN,
        "per_outer_records": per_outer_records,
        "per_outer_rae_nb1484": per_outer_rae_blend,
        "per_outer_mean": per_outer_mean,
        "per_outer_std": per_outer_std,
        "per_outer_min": per_outer_min,
        "per_outer_max": per_outer_max,
        "per_outer_median": per_outer_median,
        "rae_bob_mean": rae_bob_mean,
        "rae_bob_median": rae_bob_median,
        "delta_per_outer_mean_vs_nb1484_naive": delta_per_outer,
        "delta_bob_mean_vs_nb1484_naive": rae_bob_mean - nb1484_naive,
        "delta_bob_median_vs_nb1484_naive": rae_bob_median - nb1484_naive,
        "reproduces": bool(reproduces),
        "pearson_bobmean_vs_nb1484_best": pearson_bobmean_vs_nb1484_best,
        "pearson_bobmedian_vs_nb1484_best": pearson_bobmedian_vs_nb1484_best,
        "verdict": verdict,
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
        "n_unb", "outer_seeds",
        "rae_anchor_chemprop_aux",
        "nb1484_rae_naive_mean_blend",
        "nb1484_ref_for_verdict",
        "per_outer_rae_nb1484",
        "per_outer_mean", "per_outer_std",
        "per_outer_min", "per_outer_max", "per_outer_median",
        "rae_bob_mean", "rae_bob_median",
        "delta_per_outer_mean_vs_nb1484_naive",
        "delta_bob_mean_vs_nb1484_naive",
        "delta_bob_median_vs_nb1484_naive",
        "reproduces",
        "pearson_bobmean_vs_nb1484_best",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
