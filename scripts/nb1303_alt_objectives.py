"""nb1303 -- Alternative LGBM objectives on MACCS+ChEMBL residual.

HYPOTHESIS:
    nb1242 uses Huber loss (alpha=1.0). Different loss shapes may extract
    slightly different bias on the residual signal.  A bag across loss
    shapes may diversify and squeeze out an extra fraction of an RAE point.

PROTOCOL:
    1. Same anchor nb1070, same features (MACCS-167 + pred_chembl + sim_chembl,
       169 cols), same 5-fold KFold splits at seed 42.
    2. Train 5 LGBM regressors with DIFFERENT objectives, all at seed 42:
        (a) huber       (alpha=1.0)              -- reference (~nb1242 single-seed)
        (b) regression_l1 (MAE)
        (c) fair        (c=1.0)
        (d) tweedie     (variance_power=1.5)     -- requires non-negative target
        (e) quantile    (alpha=0.5, median)
    3. Per-objective pooled RAE.
    4. 5-objective mean-bag pooled RAE.
    5. Pred-pred Pearson between objectives.
    6. Verdict at 0.003 margin vs nb1242 (0.5431).

NOTE on tweedie: tweedie requires y >= 0.  Residual = y_unb - anchor has
sign-balanced values.  We FIT tweedie on (residual + SHIFT) and SUBTRACT
SHIFT from the prediction.  SHIFT = max(0, -residual.min()) + 1e-3.

Outputs:
    scripts/nb1303_alt_objectives.py             (this file)
    data/processed/nb1303_summary.json
    data/processed/nb1303_per_obj_corrected_oof.npy   (5, 253) float32
    data/processed/nb1303_mean_bag_oof.npy            (253,) float32
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

TAG = "nb1303"
ANCHOR = "nb1070"

# Same 5-fold KFold splits at seed 42 as nb1242 single-seed slice
SEED = 42
RESID_FOLDS = 5

MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

NB1070_REF = 0.5771
NB1242_REF = 0.5431      # 5-seed mean-bag (Huber) reference from nb1242
DECISION_MARGIN = 0.003

OBJECTIVES = [
    ("huber",         {"objective": "huber",         "alpha": 1.0}),
    ("regression_l1", {"objective": "regression_l1"}),
    ("fair",          {"objective": "fair",          "fair_c": 1.0}),
    ("tweedie",       {"objective": "tweedie",       "tweedie_variance_power": 1.5}),
    ("quantile",      {"objective": "quantile",      "alpha": 0.5}),
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
    """Same loader as nb1242 -- union three local ChEMBL caches, dedup by InChIKey."""
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
        raise FileNotFoundError("No ChEMBL caches found in data/external/")

    pool = pd.concat(frames, ignore_index=True)
    mols = pool["smiles"].apply(standardize)
    pool["inchikey"] = mols.apply(_safe_inchikey)
    pool["std_smiles"] = mols.apply(_safe_can_smiles)
    pool = pool[pool["inchikey"].notna() & pool["std_smiles"].notna()].copy()
    agg = (
        pool.groupby("inchikey", as_index=False)
        .agg(pec50=("pec50", "median"),
             std_smiles=("std_smiles", "first"))
    )
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


def _base_lgbm_params() -> dict:
    """Identical capacity to nb1242 except objective is left to caller."""
    return dict(
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
        random_state=SEED,
        n_jobs=2,
    )


def _cross_fit_one_objective(name, obj_params, X, residual):
    """5-fold KFold cross-fit at seed=42 with given LGBM objective.

    For tweedie: target must be non-negative; shift, fit, unshift.
    Returns residual OOF (253,) float64.
    """
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=SEED)
    oof = np.full(n, np.nan, dtype=np.float64)

    is_tweedie = name == "tweedie"
    if is_tweedie:
        shift = max(0.0, -float(residual.min())) + 1e-3
        target_full = residual + shift
    else:
        shift = 0.0
        target_full = residual

    params = _base_lgbm_params()
    params.update(obj_params)

    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = LGBMRegressor(**params)
        mdl.fit(X[tr_loc], target_full[tr_loc])
        pred = mdl.predict(X[va_loc])
        if is_tweedie:
            pred = pred - shift
        oof[va_loc] = pred
    return oof, shift


def _pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    am = a - a.mean()
    bm = b - b.mean()
    den = float(np.sqrt((am * am).sum() * (bm * bm).sum()))
    if den < 1e-12:
        return float("nan")
    return float((am * bm).sum() / den)


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Alternative LGBM objectives on MACCS+ChEMBL residual")
    print(f"          objectives = {[n for n, _ in OBJECTIVES]}")
    print(f"          seed = {SEED}  folds = {RESID_FOLDS}")
    print(f"          features = MACCS-167 + pred_chembl_pec50 + sim_chembl (169)")
    print("=" * 78)

    # ---- Anchor + truth ----
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
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}  "
          f"min={residual.min():+.3f}  max={residual.max():+.3f}")

    # ---- ChEMBL pool ----
    print("\n[chembl] loading local ChEMBL caches ...")
    pool = _load_chembl_pool()
    print(f"[chembl] pool after standardize+dedup: {len(pool)} cpds")

    # ---- Test leak guard ----
    test_mols = [standardize(s) for s in test_smiles]
    test_inchikeys = {
        _safe_inchikey(m) for m in test_mols if _safe_inchikey(m) is not None
    }
    n_before = len(pool)
    pool = pool[~pool["inchikey"].isin(test_inchikeys)].reset_index(drop=True)
    print(f"[chembl] leak guard: {n_before} -> {len(pool)} "
          f"(dropped {n_before - len(pool)})")

    # ---- Fingerprints + kNN ----
    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep_pool = fp_pool.sum(axis=1) > 0
    pool = pool[keep_pool].reset_index(drop=True)
    fp_pool = fp_pool[keep_pool]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))
    print(f"[chembl] final pool size: {len(pool)}  median pEC50 = {pool_median:.3f}")

    std_test_smiles = [
        Chem.MolToSmiles(m) if m is not None else "" for m in test_mols
    ]
    fp_test = morgan_fp_batch(std_test_smiles)

    top_idx, top_sim = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx, top_sim, pool_labels, fallback=pool_median
    )

    # ---- Feature matrix (253, 169) ----
    X_maccs_te = np.load(MACCS_TE_PATH)
    X_maccs_unb = X_maccs_te[unb_idx].astype(np.float32)
    pred_chembl_unb = pred_chembl_pec50[unb_idx].astype(np.float32)
    mean_sim_unb = mean_sim[unb_idx].astype(np.float32)
    X_unb = np.concatenate(
        [
            X_maccs_unb,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim = X_unb.shape[1]
    print(f"[feat] X_unb shape = {X_unb.shape}  (MACCS-167 + pred_chembl + sim)")

    # ---- Per-objective cross-fit ----
    print("\n" + "-" * 78)
    print(f"PER-OBJECTIVE CROSS-FIT (KFold seed={SEED}, "
          f"shallow LGBM depth=3 n_estim=80)")
    print("-" * 78)

    per_obj_resid_oof = np.zeros((len(OBJECTIVES), n_unb), dtype=np.float64)
    per_obj_corrected = np.zeros((len(OBJECTIVES), n_unb), dtype=np.float64)
    per_obj_rae: list[float] = []
    per_obj_records = []

    for i, (name, obj_params) in enumerate(OBJECTIVES):
        try:
            resid_oof, shift = _cross_fit_one_objective(
                name, obj_params, X_unb, residual
            )
            corrected = anchor + resid_oof
            r = float(rae(y_unb, corrected))
            err = None
        except Exception as e:
            resid_oof = np.zeros(n_unb, dtype=np.float64)
            corrected = anchor.copy()
            r = float(rae(y_unb, corrected))
            shift = 0.0
            err = repr(e)

        per_obj_resid_oof[i] = resid_oof
        per_obj_corrected[i] = corrected
        per_obj_rae.append(r)
        per_obj_records.append({
            "objective": name,
            "params": obj_params,
            "tweedie_shift": float(shift),
            "rae": r,
            "delta_vs_nb1070": r - rae_anchor,
            "delta_vs_nb1242": r - NB1242_REF,
            "resid_oof_mean": float(resid_oof.mean()),
            "resid_oof_std": float(resid_oof.std()),
            "error": err,
        })
        print(f"   [{i}] {name:13s}  rae = {r:.4f}  "
              f"(d_vs_nb1070 = {r - rae_anchor:+.4f}, "
              f"d_vs_nb1242 = {r - NB1242_REF:+.4f})  "
              f"resid_std = {resid_oof.std():.3f}"
              + (f"  ERR: {err}" if err else ""))

    # ---- Mean bag across 5 objectives ----
    mean_bag_oof = per_obj_corrected.mean(axis=0)
    median_bag_oof = np.median(per_obj_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))

    print("\n" + "-" * 78)
    print("BAG AGGREGATIONS (5 objectives)")
    print("-" * 78)
    print(f"   per-obj RAE list      = "
          f"[{', '.join(f'{r:.4f}' for r in per_obj_rae)}]")
    print(f"   per-obj mean          = {float(np.mean(per_obj_rae)):.4f}")
    print(f"   per-obj std           = {float(np.std(per_obj_rae)):.4f}")
    print(f"   pooled RAE(mean_bag)  = {rae_mean_bag:.4f}  "
          f"(d_vs_nb1242 = {rae_mean_bag - NB1242_REF:+.4f})")
    print(f"   pooled RAE(median_bag)= {rae_median_bag:.4f}  "
          f"(d_vs_nb1242 = {rae_median_bag - NB1242_REF:+.4f})")
    print(f"   nb1242 ref            = {NB1242_REF:.4f}")

    # ---- Pred-pred Pearson ----
    print("\n" + "-" * 78)
    print("PRED-PRED PEARSON BETWEEN OBJECTIVES (on corrected OOF, 253 unb)")
    print("-" * 78)
    n_obj = len(OBJECTIVES)
    pearson_mat = np.eye(n_obj, dtype=np.float64)
    pearson_pairs = []
    for i in range(n_obj):
        for j in range(i + 1, n_obj):
            r = _pearson(per_obj_corrected[i], per_obj_corrected[j])
            pearson_mat[i, j] = r
            pearson_mat[j, i] = r
            pearson_pairs.append({
                "obj_a": OBJECTIVES[i][0],
                "obj_b": OBJECTIVES[j][0],
                "pearson": r,
            })
            print(f"   {OBJECTIVES[i][0]:13s} <-> {OBJECTIVES[j][0]:13s}  r = {r:.4f}")
    off_diag = pearson_mat[np.triu_indices(n_obj, k=1)]
    print(f"\n   mean off-diagonal Pearson = {float(off_diag.mean()):.4f}")
    print(f"   min  off-diagonal Pearson = {float(off_diag.min()):.4f}")

    # ---- Verdict ----
    beats_nb1242 = rae_mean_bag < NB1242_REF - DECISION_MARGIN
    ties_nb1242 = abs(rae_mean_bag - NB1242_REF) < DECISION_MARGIN
    beats_nb1070 = rae_mean_bag < rae_anchor - DECISION_MARGIN

    if beats_nb1242:
        verdict = "ALT_OBJ_BAG_BEATS_NB1242_NEW_PRIMARY_CANDIDATE"
    elif ties_nb1242:
        verdict = "ALT_OBJ_BAG_TIES_NB1242_NO_NEW_SIGNAL"
    elif beats_nb1070:
        verdict = "ALT_OBJ_BAG_HELPS_NB1070_BUT_NOT_NB1242"
    else:
        verdict = "ALT_OBJ_BAG_HURTS_NB1070"
    print(f"\n   verdict = {verdict}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_per_obj_corrected_oof.npy",
            per_obj_corrected.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy",
            mean_bag_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_per_obj_corrected_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_mean_bag_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "seed": SEED,
        "folds": RESID_FOLDS,
        "feature_dim": feat_dim,
        "objectives": [n for n, _ in OBJECTIVES],
        "per_obj_records": per_obj_records,
        "per_obj_rae": per_obj_rae,
        "per_obj_rae_mean": float(np.mean(per_obj_rae)),
        "per_obj_rae_std": float(np.std(per_obj_rae)),
        "per_obj_rae_min": float(np.min(per_obj_rae)),
        "per_obj_rae_max": float(np.max(per_obj_rae)),
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "rae_anchor_nb1070": rae_anchor,
        "delta_mean_bag_vs_nb1070": rae_mean_bag - rae_anchor,
        "delta_mean_bag_vs_nb1242": rae_mean_bag - NB1242_REF,
        "pearson_pairs": pearson_pairs,
        "pearson_off_diag_mean": float(off_diag.mean()),
        "pearson_off_diag_min": float(off_diag.min()),
        "pearson_off_diag_max": float(off_diag.max()),
        "n_chembl_pool": int(len(pool)),
        "n_unb": n_unb,
        "nb1070_ref_pooled": NB1070_REF,
        "nb1242_ref": NB1242_REF,
        "decision_margin": DECISION_MARGIN,
        "beats_nb1070": bool(beats_nb1070),
        "beats_nb1242": bool(beats_nb1242),
        "ties_nb1242": bool(ties_nb1242),
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
        "per_obj_rae", "per_obj_rae_mean", "per_obj_rae_std",
        "rae_mean_bag", "rae_median_bag",
        "delta_mean_bag_vs_nb1242",
        "pearson_off_diag_mean", "pearson_off_diag_min",
        "beats_nb1242", "ties_nb1242", "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
