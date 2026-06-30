"""nb1264 -- Greedy forward feature pruning of nb1253's 13 ChEMBL kNN features.

Hypothesis:
    nb1253 (all 13 features) cross-fit RAE 0.5570, nb1242 (k=5 mean + k=5 sim,
    2 features) cross-fit RAE 0.5431. Intermediate subsets may strike a better
    bias-variance tradeoff at n=253. Forward-greedy search on the top-8
    candidates (ranked by nb1253 gain importance, all ChEMBL-only) to find the
    optimal subset for residual LGBM on MACCS-167 anchor=nb1070.

Protocol:
    1. Compute the 13 ChEMBL kNN features for unblind 253 (mirror nb1253
       exactly -- same pool, same FP, same kNN, same residual learner).
    2. Restrict the candidate pool to top-8 features by nb1253 gain importance
       (chembl-only ranking, augmented with nb1242 baseline picks):
           mean_10_pec50, top1_sim, top1_pec50, std_10_pec50, std_3_pec50,
           mean_5_pec50, mean_5_sim, mean_3_pec50.
    3. Greedy forward selection up to k_max=8. At each step, try adding each
       remaining feature, score by 5-seed bag pooled RAE (5-fold cross-fit),
       pick the best. Early-stop if improvement < 0.001 from previous step.
    4. Report best-RAE subset at each k in {2, 3, 4, 5, 6, 8} (where reached
       before early stop), global best, beats_nb1242 verdict at 0.003 margin.

Outputs:
    scripts/nb1264_chembl_feature_pruning.py    (this file)
    data/processed/nb1264_summary.json
    data/processed/nb1264_best_subset_oof.npy   (253,) float32
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

TAG = "nb1264"
ANCHOR = "nb1070"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K_MAX = 10
SIM_FLOOR = 1e-6

NB1070_REF = 0.5771
NB1242_REF = 0.5431
NB1253_REF = 0.5570
DECISION_MARGIN = 0.003
GREEDY_EARLY_STOP_DELTA = 0.001
K_MAX = 8
K_REPORT = [2, 3, 4, 5, 6, 8]

# Feature names mirror nb1253 column order
FEATURE_NAMES_CHEMBL = [
    "top1_sim",
    "top1_pec50",
    "mean_3_pec50",
    "mean_3_sim",
    "std_3_pec50",
    "mean_5_pec50",
    "mean_5_sim",
    "std_5_pec50",
    "max_5_pec50",
    "min_5_pec50",
    "mean_10_pec50",
    "mean_10_sim",
    "std_10_pec50",
]

# Top-8 candidates: combine nb1242 baseline (mean_5_pec50, mean_5_sim) with the
# top-6 ChEMBL features by gain importance from nb1253 (chembl-only ranking)
CANDIDATE_FEATURES = [
    "mean_10_pec50",   # nb1253 top-ChEMBL #1
    "top1_sim",        # nb1253 top-ChEMBL #2
    "top1_pec50",      # nb1253 top-ChEMBL #3
    "std_10_pec50",    # nb1253 top-ChEMBL #4
    "std_3_pec50",     # nb1253 top-ChEMBL #5
    "mean_5_pec50",    # nb1242 baseline anchor
    "mean_5_sim",      # nb1242 baseline anchor
    "mean_3_pec50",    # next-best aggregator
]


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
        raise FileNotFoundError("No local ChEMBL PXR parquets in data/external/")

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


def _weighted_mean(sim: np.ndarray, vals: np.ndarray, fallback: float) -> float:
    w = np.clip(sim, 0.0, 1.0)
    s = w.sum()
    if s < SIM_FLOOR:
        return float(fallback)
    return float(np.sum(w * vals) / s)


def _build_chembl_features(top_idx_10, top_sim_10, pool_labels, pool_median):
    n_q = top_idx_10.shape[0]
    F = np.zeros((n_q, 13), dtype=np.float32)
    for i in range(n_q):
        sims_10 = top_sim_10[i]
        idx_10 = top_idx_10[i]
        labs_10 = pool_labels[idx_10]
        sims_1 = sims_10[:1]; labs_1 = labs_10[:1]
        sims_3 = sims_10[:3]; labs_3 = labs_10[:3]
        sims_5 = sims_10[:5]; labs_5 = labs_10[:5]
        top1_sim = float(sims_1[0])
        top1_pec50 = float(labs_1[0]) if top1_sim >= SIM_FLOOR else pool_median
        mean_3_pec50 = _weighted_mean(sims_3, labs_3, pool_median)
        mean_3_sim = float(np.mean(sims_3))
        std_3_pec50 = float(np.std(labs_3)) if np.sum(sims_3) >= SIM_FLOOR else 0.0
        mean_5_pec50 = _weighted_mean(sims_5, labs_5, pool_median)
        mean_5_sim = float(np.mean(sims_5))
        std_5_pec50 = float(np.std(labs_5)) if np.sum(sims_5) >= SIM_FLOOR else 0.0
        max_5_pec50 = float(np.max(labs_5)) if np.sum(sims_5) >= SIM_FLOOR else pool_median
        min_5_pec50 = float(np.min(labs_5)) if np.sum(sims_5) >= SIM_FLOOR else pool_median
        mean_10_pec50 = _weighted_mean(sims_10, labs_10, pool_median)
        mean_10_sim = float(np.mean(sims_10))
        std_10_pec50 = float(np.std(labs_10)) if np.sum(sims_10) >= SIM_FLOOR else 0.0
        F[i] = [
            top1_sim, top1_pec50,
            mean_3_pec50, mean_3_sim, std_3_pec50,
            mean_5_pec50, mean_5_sim, std_5_pec50, max_5_pec50, min_5_pec50,
            mean_10_pec50, mean_10_sim, std_10_pec50,
        ]
    return F


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


def _score_subset(X_maccs: np.ndarray, F_chembl_sub: np.ndarray,
                  residual: np.ndarray, anchor: np.ndarray,
                  y_unb: np.ndarray) -> tuple[float, np.ndarray]:
    """Return (pooled mean-bag RAE, mean_bag_oof) for a candidate ChEMBL subset.

    Mirrors nb1253 residual learner exactly: 5-seed bag, 5-fold cross-fit,
    LGBM Huber on MACCS-167 + chosen ChEMBL subset, mean over seeds.
    """
    if F_chembl_sub.shape[1] == 0:
        X = X_maccs
    else:
        X = np.concatenate([X_maccs, F_chembl_sub], axis=1).astype(np.float32)
    n_unb = len(residual)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    for i, s in enumerate(RESID_SEEDS):
        kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=s)
        oof = np.full(n_unb, np.nan, dtype=np.float64)
        for tr_loc, va_loc in kf.split(np.arange(n_unb)):
            mdl = LGBMRegressor(**_lgbm_params(s))
            mdl.fit(X[tr_loc], residual[tr_loc])
            oof[va_loc] = mdl.predict(X[va_loc])
        per_seed_corrected[i] = anchor + oof
    mean_bag = per_seed_corrected.mean(axis=0)
    return float(rae(y_unb, mean_bag)), mean_bag


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Greedy forward feature pruning of nb1253's 13 ChEMBL feats")
    print(f"         candidate pool ({len(CANDIDATE_FEATURES)}): {CANDIDATE_FEATURES}")
    print(f"         k_max = {K_MAX}  early-stop delta = {GREEDY_EARLY_STOP_DELTA}")
    print(f"         seeds = {RESID_SEEDS}  folds = {RESID_FOLDS}")
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
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} pooled RAE = {rae_anchor:.4f}")
    residual = y_unb - anchor

    # ---- ChEMBL pool + features ----
    print("\n" + "-" * 78)
    print("REBUILD nb1253 13-DIM CHEMBL FEATURES (same pool, FP, kNN)")
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
    print(f"   leak guard: {n_before} -> {len(pool)}")
    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep_pool = fp_pool.sum(axis=1) > 0
    if not keep_pool.all():
        pool = pool[keep_pool].reset_index(drop=True)
        fp_pool = fp_pool[keep_pool]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))
    print(f"   pool shape: {fp_pool.shape}  median pEC50 = {pool_median:.3f}")
    std_test_smiles = [Chem.MolToSmiles(m) if m is not None else "" for m in test_mols]
    fp_test = morgan_fp_batch(std_test_smiles)
    top_idx_10, top_sim_10 = _tanimoto_topk(fp_test, fp_pool, k=KNN_K_MAX)
    F_chembl = _build_chembl_features(top_idx_10, top_sim_10, pool_labels, pool_median)
    print(f"   F_chembl shape: {F_chembl.shape}")

    # ---- MACCS unblind ----
    X_maccs_te = np.load(MACCS_TE_PATH)
    X_maccs_unb = X_maccs_te[unb_idx].astype(np.float32)
    F_chembl_unb = F_chembl[unb_idx].astype(np.float32)
    print(f"   MACCS-unb shape: {X_maccs_unb.shape}  "
          f"F_chembl-unb shape: {F_chembl_unb.shape}")

    # Column indexer
    name_to_col = {n: i for i, n in enumerate(FEATURE_NAMES_CHEMBL)}

    # ---- Baseline: nb1242 (k=5 mean_pec50 + mean_5_sim, 2 features) ----
    base_subset = ["mean_5_pec50", "mean_5_sim"]
    base_cols = [name_to_col[n] for n in base_subset]
    rae_nb1242_recomp, _ = _score_subset(
        X_maccs_unb, F_chembl_unb[:, base_cols], residual, anchor, y_unb,
    )
    print(f"\n[recomp] nb1242-style 2-feat subset RAE = {rae_nb1242_recomp:.4f}  "
          f"(published ref = {NB1242_REF:.4f})")

    # ---- Baseline: nb1253 full 13 features ----
    rae_nb1253_recomp, _ = _score_subset(
        X_maccs_unb, F_chembl_unb, residual, anchor, y_unb,
    )
    print(f"[recomp] nb1253 full 13-feat subset RAE = {rae_nb1253_recomp:.4f}  "
          f"(published ref = {NB1253_REF:.4f})")

    # ---- Greedy forward selection ----
    print("\n" + "-" * 78)
    print(f"GREEDY FORWARD SEARCH (top-{len(CANDIDATE_FEATURES)} candidates, "
          f"k_max={K_MAX})")
    print("-" * 78)
    selected: list[str] = []
    remaining = list(CANDIDATE_FEATURES)
    trace: list[dict] = []
    best_so_far_rae = float("inf")
    best_so_far_subset: list[str] = []
    best_so_far_oof: np.ndarray | None = None
    per_k_best: dict[int, dict] = {}
    stopped_early = False
    early_stop_reason = None

    for step in range(1, K_MAX + 1):
        if not remaining:
            break
        best_feat = None
        best_rae = float("inf")
        best_oof = None
        step_evals = []
        for cand in remaining:
            cand_subset = selected + [cand]
            cols = [name_to_col[n] for n in cand_subset]
            rae_c, oof_c = _score_subset(
                X_maccs_unb, F_chembl_unb[:, cols],
                residual, anchor, y_unb,
            )
            step_evals.append({"feature": cand, "rae": rae_c})
            if rae_c < best_rae:
                best_rae = rae_c
                best_feat = cand
                best_oof = oof_c

        improvement = best_so_far_rae - best_rae
        prev_best = best_so_far_rae if np.isfinite(best_so_far_rae) else None
        step_record = {
            "step": step,
            "k": step,
            "added_feature": best_feat,
            "rae_after_add": best_rae,
            "previous_best_rae": prev_best,
            "improvement": (improvement if np.isfinite(best_so_far_rae)
                            else None),
            "candidates_tried": step_evals,
            "current_subset": selected + [best_feat],
        }
        trace.append(step_record)
        selected = selected + [best_feat]
        remaining = [c for c in remaining if c != best_feat]
        print(f"   step {step}:  add {best_feat:18s}  RAE = {best_rae:.4f}  "
              f"(d_vs_prev = {-improvement if np.isfinite(prev_best or np.inf) else 0:+.4f})  "
              f"subset_size = {len(selected)}")

        # Record per-k best
        if step in K_REPORT:
            per_k_best[step] = {
                "k": step,
                "subset": list(selected),
                "rae": best_rae,
            }

        # Track global best
        if best_rae < best_so_far_rae:
            best_so_far_rae = best_rae
            best_so_far_subset = list(selected)
            best_so_far_oof = best_oof.copy() if best_oof is not None else None

        # Early stop
        if step >= 2 and improvement < GREEDY_EARLY_STOP_DELTA:
            stopped_early = True
            early_stop_reason = (
                f"step {step}: improvement {improvement:+.4f} < "
                f"{GREEDY_EARLY_STOP_DELTA}"
            )
            print(f"   [early-stop] {early_stop_reason}")
            break

    # ---- Verdict ----
    beats_nb1242 = best_so_far_rae < NB1242_REF - DECISION_MARGIN
    beats_nb1253 = best_so_far_rae < NB1253_REF - DECISION_MARGIN
    beats_anchor = best_so_far_rae < rae_anchor - DECISION_MARGIN

    if beats_nb1242:
        verdict = "PRUNED_SUBSET_BEATS_NB1242_NEW_PRIMARY_CANDIDATE"
    elif abs(best_so_far_rae - NB1242_REF) < DECISION_MARGIN:
        verdict = "PRUNED_SUBSET_FLAT_VS_NB1242_NO_NEW_SIGNAL"
    elif best_so_far_rae < NB1253_REF:
        verdict = "PRUNED_SUBSET_BEATS_NB1253_BUT_NOT_NB1242"
    else:
        verdict = "PRUNED_SUBSET_FAILS_NO_BETTER_THAN_NB1253"

    print("\n" + "-" * 78)
    print("FINAL VERDICT")
    print("-" * 78)
    print(f"   anchor nb1070 RAE      = {rae_anchor:.4f}")
    print(f"   nb1242 published ref   = {NB1242_REF:.4f}")
    print(f"   nb1242 recomputed      = {rae_nb1242_recomp:.4f}")
    print(f"   nb1253 published ref   = {NB1253_REF:.4f}")
    print(f"   nb1253 recomputed      = {rae_nb1253_recomp:.4f}")
    print(f"   best pruned subset RAE = {best_so_far_rae:.4f}")
    print(f"   best pruned subset     = {best_so_far_subset}")
    print(f"   k_best                 = {len(best_so_far_subset)}")
    print(f"   d_vs_nb1242            = {best_so_far_rae - NB1242_REF:+.4f}")
    print(f"   d_vs_nb1253            = {best_so_far_rae - NB1253_REF:+.4f}")
    print(f"   d_vs_anchor            = {best_so_far_rae - rae_anchor:+.4f}")
    print(f"   beats_nb1242 (m={DECISION_MARGIN}): {beats_nb1242}")
    print(f"   verdict                = {verdict}")

    # ---- Save artifacts ----
    if best_so_far_oof is not None:
        np.save(DATA_PROCESSED / f"{TAG}_best_subset_oof.npy",
                best_so_far_oof.astype(np.float32))
        print(f"\n[save] {DATA_PROCESSED / f'{TAG}_best_subset_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "candidate_features": CANDIDATE_FEATURES,
        "k_max": K_MAX,
        "k_report": K_REPORT,
        "early_stop_delta": GREEDY_EARLY_STOP_DELTA,
        "stopped_early": stopped_early,
        "early_stop_reason": early_stop_reason,
        "n_chembl_pool": int(len(pool)),
        "n_unb": n_unb,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "rae_anchor_nb1070": rae_anchor,
        "rae_nb1242_published": NB1242_REF,
        "rae_nb1242_recomputed": rae_nb1242_recomp,
        "rae_nb1253_published": NB1253_REF,
        "rae_nb1253_recomputed": rae_nb1253_recomp,
        "greedy_trace": trace,
        "per_k_best": {str(k): v for k, v in per_k_best.items()},
        "best_subset": best_so_far_subset,
        "best_subset_size": len(best_so_far_subset),
        "best_subset_rae": best_so_far_rae,
        "delta_best_vs_nb1242": best_so_far_rae - NB1242_REF,
        "delta_best_vs_nb1253": best_so_far_rae - NB1253_REF,
        "delta_best_vs_anchor": best_so_far_rae - rae_anchor,
        "decision_margin": DECISION_MARGIN,
        "beats_nb1242": bool(beats_nb1242),
        "beats_nb1253": bool(beats_nb1253),
        "beats_anchor": bool(beats_anchor),
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
        "rae_anchor_nb1070", "rae_nb1242_recomputed", "rae_nb1253_recomputed",
        "best_subset", "best_subset_size", "best_subset_rae",
        "delta_best_vs_nb1242", "delta_best_vs_nb1253",
        "beats_nb1242", "beats_nb1253", "verdict",
        "stopped_early", "early_stop_reason",
    ):
        print(f"  {k}: {res.get(k)}")
    print("  greedy_trace:")
    for st in res.get("greedy_trace", []):
        prev = st.get("previous_best_rae")
        prev_str = f"{prev:.4f}" if prev is not None else "----"
        imp = st.get("improvement")
        imp_str = f"{imp:+.4f}" if imp is not None else "----"
        print(f"    step {st['step']}: +{st['added_feature']:18s}  "
              f"RAE = {st['rae_after_add']:.4f}  "
              f"(prev = {prev_str}, d = {imp_str})")
