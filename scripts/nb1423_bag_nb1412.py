"""nb1423 -- Outer-bag VALIDATION of nb1412 (Mordred K=20 standalone).

Protocol:
    1. Reuse the stored top-20 Mordred col indices from nb1412 (K=20).
    2. For each OUTER seed o in {0, 1, 7, 42, 137}:
         - Define INNER seeds = [o*1000 + s for s in {0, 1, 7, 42, 137}].
         - For each inner seed: 5-fold cross-fit (KFold random_state=inner)
           of shallow LGBM Huber on the (K=20 Mordred + ChEMBL pred + sim) 22-col
           matrix, residual = y_unb - nb1070_pred_oof, anchor + resid_oof.
         - Mean-bag across 5 inner seeds -> per-outer OOF (253,).
         - Pooled RAE per outer.
    3. Row-level BoB MEAN  : stack outer-mean OOFs (5, 253) -> mean across axis 0.
       Row-level BoB MEDIAN: stack outer-mean OOFs (5, 253) -> median across axis 0.
    4. Verdict NB1412_REPRODUCES if mean(per_outer_rae) is within 0.003 of 0.5180.

Outputs:
    scripts/nb1423_bag_nb1412.py             (this file)
    data/processed/nb1423_summary.json
    data/processed/nb1423_bob_mean_oof.npy   (253,) float32
    data/processed/nb1423_bob_median_oof.npy (253,) float32
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

TAG = "nb1423"
ANCHOR = "nb1070"

OUTER_SEEDS = [0, 1, 7, 42, 137]
INNER_BASE = [0, 1, 7, 42, 137]   # inner seeds = outer*1000 + base
RESID_FOLDS = 5

MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

NB1412_REF = 0.5180        # nb1412 K=20 mean-bag pooled RAE (target)
NB1070_REF = 0.5771
DECISION_MARGIN = 0.003

# Top-20 Mordred column indices (from nb1412_summary.json, K=20 record)
NB1412_TOP20_IDX = [
    292, 431, 1242, 454, 255, 1150, 1178, 1197, 297, 504,
    253, 464, 231, 914, 239, 1494, 1089, 770, 233, 831,
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
    """Same union as nb1412 / nb1364."""
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
    """5-fold cross-fit with KFold(random_state=seed) and LGBM(random_state=seed)."""
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
            f"Mordred test shape mismatch: {X_te_m.shape} vs "
            f"n_test={n_test_expected}"
        )
    X_te_m = np.where(np.isfinite(X_te_m), X_te_m, np.nan).astype(np.float32)
    col_med = np.nanmedian(X_te_m, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X_te_m)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X_te_m[idx_r, idx_c] = col_med[idx_c]
    return X_te_m


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Outer-bag VALIDATION of nb1412 (Mordred K=20, anchor={ANCHOR})")
    print(f"          OUTER_SEEDS = {OUTER_SEEDS}")
    print(f"          INNER_BASE  = {INNER_BASE}  (inner = outer*1000 + base)")
    print(f"          nb1412 ref  = {NB1412_REF:.4f}   margin = {DECISION_MARGIN}")
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

    # ---- Mordred top-20 slice ----
    print(f"[feat] loading cached Mordred test matrix from {MORDRED_DIR}")
    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    n_mord = int(X_mord_te.shape[1])
    print(f"[feat] X_mord_te shape = {X_mord_te.shape}  (n_mordred={n_mord})")
    top_idx_arr = np.array(NB1412_TOP20_IDX, dtype=int)
    X_mord_unb_top20 = X_mord_te[unb_idx][:, top_idx_arr].astype(np.float32)
    print(f"[feat] top-20 Mordred slice on unblind: {X_mord_unb_top20.shape}")

    # ---- ChEMBL pool + kNN features ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL (local cache; same union as nb1412/nb1364)")
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

    top_nn, top_sim = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_nn, top_sim, pool_labels, fallback=pool_median
    )
    pred_chembl_unb = pred_chembl_pec50[unb_idx].astype(np.float32)
    mean_sim_unb = mean_sim[unb_idx].astype(np.float32)

    # ---- Build 22-col matrix on unblind ----
    X_unb = np.concatenate(
        [
            X_mord_unb_top20,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim = X_unb.shape[1]
    print(f"\n[feat] X_unb shape = {X_unb.shape}  (Mordred-20 + pred_chembl + sim)")

    # ---- Outer / inner bag loop ----
    print("\n" + "-" * 78)
    print(f"OUTER-BAG LOOP  ({len(OUTER_SEEDS)} outer x {len(INNER_BASE)} inner)")
    print("-" * 78)
    per_outer_records: list[dict] = []
    per_outer_oofs = np.zeros((len(OUTER_SEEDS), n_unb), dtype=np.float64)
    for oi, o in enumerate(OUTER_SEEDS):
        t_o0 = time.time()
        inner_seeds = [o * 1000 + b for b in INNER_BASE]
        per_inner_corrected = np.zeros((len(inner_seeds), n_unb), dtype=np.float64)
        per_inner_rae: list[float] = []
        for ii, s in enumerate(inner_seeds):
            resid_oof_s = _residual_cross_fit_one_seed(X_unb, residual, s)
            pred_corr_s = anchor + resid_oof_s
            per_inner_corrected[ii] = pred_corr_s
            per_inner_rae.append(float(rae(y_unb, pred_corr_s)))
        outer_mean_oof = per_inner_corrected.mean(axis=0)
        outer_rae = float(rae(y_unb, outer_mean_oof))
        per_outer_oofs[oi] = outer_mean_oof
        dt = time.time() - t_o0
        rec = {
            "outer_seed": int(o),
            "inner_seeds": [int(x) for x in inner_seeds],
            "per_inner_rae": per_inner_rae,
            "inner_mean_rae": float(np.mean(per_inner_rae)),
            "inner_std_rae": float(np.std(per_inner_rae)),
            "outer_mean_bag_rae": outer_rae,
            "delta_vs_nb1412": float(outer_rae - NB1412_REF),
            "wall_sec": round(dt, 2),
        }
        per_outer_records.append(rec)
        print(f"   outer={o:3d}  inner_mean={rec['inner_mean_rae']:.4f}+/-"
              f"{rec['inner_std_rae']:.4f}  outer_mean_bag={outer_rae:.4f}  "
              f"d_vs_nb1412={rec['delta_vs_nb1412']:+.4f}  [{dt:.1f}s]")

    # ---- Row-level BoB across outer seeds ----
    bob_mean_oof = per_outer_oofs.mean(axis=0)
    bob_median_oof = np.median(per_outer_oofs, axis=0)
    rae_bob_mean = float(rae(y_unb, bob_mean_oof))
    rae_bob_median = float(rae(y_unb, bob_median_oof))

    per_outer_rae = [r["outer_mean_bag_rae"] for r in per_outer_records]
    per_outer_mean = float(np.mean(per_outer_rae))
    per_outer_std = float(np.std(per_outer_rae))
    per_outer_min = float(np.min(per_outer_rae))
    per_outer_max = float(np.max(per_outer_rae))

    reproduces = abs(per_outer_mean - NB1412_REF) <= DECISION_MARGIN
    if reproduces:
        verdict = "NB1412_REPRODUCES"
    elif per_outer_mean < NB1412_REF - DECISION_MARGIN:
        verdict = "NB1412_OVER-REPRODUCES_BETTER_THAN_REPORTED"
    else:
        verdict = "NB1412_DOES_NOT_REPRODUCE_WORSE_THAN_REPORTED"

    print("\n" + "-" * 78)
    print("OUTER-BAG SUMMARY")
    print("-" * 78)
    print(f"   per-outer RAE list   = {[round(r, 4) for r in per_outer_rae]}")
    print(f"   per-outer mean+/-std = {per_outer_mean:.4f}+/-{per_outer_std:.4f}  "
          f"[min={per_outer_min:.4f} max={per_outer_max:.4f}]")
    print(f"   nb1412 reference     = {NB1412_REF:.4f}")
    print(f"   delta(outer_mean - nb1412) = {per_outer_mean - NB1412_REF:+.4f}  "
          f"(margin = {DECISION_MARGIN})")
    print(f"   BoB MEAN   pooled RAE = {rae_bob_mean:.4f}")
    print(f"   BoB MEDIAN pooled RAE = {rae_bob_median:.4f}")
    print(f"   verdict              = {verdict}")

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
        "data_source": "mordred_cached_nb1030 + local_chembl_caches_union",
        "n_chembl_pool": int(len(pool)),
        "n_unb": n_unb,
        "n_mordred_cols": n_mord,
        "feat_dim": int(feat_dim),
        "nb1412_top20_col_idx": NB1412_TOP20_IDX,
        "outer_seeds": OUTER_SEEDS,
        "inner_base": INNER_BASE,
        "inner_seed_formula": "outer*1000 + base",
        "resid_folds": RESID_FOLDS,
        "lgbm_max_depth": 3,
        "lgbm_num_leaves": 7,
        "lgbm_n_estimators": 80,
        "lgbm_learning_rate": 0.05,
        "lgbm_min_child_samples": 20,
        "lgbm_huber_alpha": 1.0,
        "rae_anchor_nb1070": rae_anchor,
        "nb1070_ref_pooled": NB1070_REF,
        "nb1412_ref": NB1412_REF,
        "decision_margin": DECISION_MARGIN,
        "per_outer_records": per_outer_records,
        "per_outer_rae_list": per_outer_rae,
        "per_outer_mean_rae": per_outer_mean,
        "per_outer_std_rae": per_outer_std,
        "per_outer_min_rae": per_outer_min,
        "per_outer_max_rae": per_outer_max,
        "bob_mean_pooled_rae": rae_bob_mean,
        "bob_median_pooled_rae": rae_bob_median,
        "delta_per_outer_mean_vs_nb1412": per_outer_mean - NB1412_REF,
        "delta_bob_mean_vs_nb1412": rae_bob_mean - NB1412_REF,
        "delta_bob_median_vs_nb1412": rae_bob_median - NB1412_REF,
        "reproduces_nb1412": bool(reproduces),
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
        "n_chembl_pool", "n_mordred_cols", "feat_dim",
        "per_outer_rae_list",
        "per_outer_mean_rae", "per_outer_std_rae",
        "bob_mean_pooled_rae", "bob_median_pooled_rae",
        "delta_per_outer_mean_vs_nb1412",
        "delta_bob_mean_vs_nb1412", "delta_bob_median_vs_nb1412",
        "reproduces_nb1412", "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
