"""nb1250 -- DEPLOY artifact for nb1242 (ChEMBL kNN + MACCS residual bag) on
the 513-row test set.

PRECEDENT
---------
nb1242 (diagnostic) 5-seed mean-bag pooled RAE on 253 unblind = 0.5431
(honest cross-fit; LB-faithful anchor).

PROTOCOL
--------
1. Rebuild 513-row ChEMBL kNN features:
   - Union three local ChEMBL PXR caches -> standardize -> InChIKey dedup
     -> 945 unique pool cpds (matches nb1242).
   - Morgan-2048 for ChEMBL pool + 513 standardized test SMILES.
   - For each of 513 test rows: top-k=5 Tanimoto NN, similarity-weighted
     mean of ChEMBL pEC50 -> pred_chembl_pec50_513 (513,).  Mean of top-5
     similarities -> sim_chembl_513 (513,).
2. Build deploy residual:
   - Anchor = nb1070_pred_oof (253) on unblind, te_nb1070 (513) on test.
   - residual target = y_unb - nb1070_pred_oof.
   - Feature matrix (253, 169): MACCS-167[unb] + pred_chembl_pec50[unb_idx]
     + sim[unb_idx]; (513, 169) analogous for the test slice.
   - 5-seed shallow LGBM Huber bag (seeds 0,1,7,42,137), each fit on ALL
     253 unblind rows, predicting residual on 513 test rows.
3. Mean-bag deploy residuals across seeds (513,) -> te_residual_513.
4. te_nb1250 = te_nb1070 + te_residual_513.
5. Save submission CSV (513 rows, SMILES + Molecule Name + pEC50).

NOTE
----
Per feedback_lb_two_regime_calibration / feedback_te_vs_pred_oof_protocol:
each inner LGBM is fit on ALL 253 unblind rows, so in_RAE on te[unb_idx] is
in-sample optimistic. The LB-faithful anchor is the honest 0.5431 cross-fit
RAE from nb1242 (mean-bag).

Outputs:
  data/processed/pred_chembl_pec50_513.npy   (513,) float32
  data/processed/sim_chembl_513.npy          (513,) float32
  data/processed/te_nb1250.npy               (513,) float32
  submissions/nb1250_deploy_nb1242.csv       (513 rows)
  data/processed/nb1250_summary.json
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
from lightgbm import LGBMRegressor
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1250"
ANCHOR = "nb1070"

RESID_SEEDS = [0, 1, 7, 42, 137]

MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

NB1242_HONEST_LB_ANCHOR = 0.5431  # mean-bag pooled RAE on 253 unblind

SUBMISSIONS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "submissions")
)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Helpers (mirror nb1242).
# -----------------------------------------------------------------------------
def _safe_inchikey(mol) -> str | None:
    try:
        return Chem.MolToInchiKey(mol) if mol is not None else None
    except Exception:
        return None


def _safe_can_smiles(mol) -> str | None:
    try:
        return Chem.MolToSmiles(mol) if mol is not None else None
    except Exception:
        return None


def _load_chembl_pool() -> pd.DataFrame:
    """Union three local ChEMBL PXR caches -> (inchikey, std_smiles, pec50)."""
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


def _knn_predict(top_idx, top_sim, pool_labels, fallback: float):
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


def _save_submission_csv(te_pred, te_smiles, te_names, csv_path: str,
                         label: str) -> dict:
    assert te_pred.shape[0] == 513, (
        f"{label}: te_pred shape {te_pred.shape}, expected (513,)"
    )
    assert np.all(np.isfinite(te_pred)), f"{label}: te_pred has NaN/Inf"
    sub = pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": te_pred.astype(np.float64),
    })
    assert len(sub) == 513, f"{label}: row count {len(sub)} != 513"
    assert list(sub.columns) == ["SMILES", "Molecule Name", "pEC50"], (
        f"{label}: column order wrong: {list(sub.columns)}"
    )
    assert sub.isna().sum().sum() == 0, f"{label}: CSV has NaN"
    sub.to_csv(csv_path, index=False)
    return {
        "csv_path": csv_path,
        "n_rows": int(len(sub)),
        "columns": list(sub.columns),
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- DEPLOY nb1242 ChEMBL kNN + MACCS residual bag on 513 test")
    print(f"          anchor      = {ANCHOR} (te_{ANCHOR}.npy + {ANCHOR}_pred_oof.npy)")
    print(f"          resid seeds = {RESID_SEEDS}")
    print(f"          features    = MACCS-167 + pred_chembl_pec50 + sim  (169)")
    print(f"          LGBM:  depth=3, num_leaves=7, n_est=80, lr=0.05, "
          f"min_child=20, obj=huber(alpha=1.0)")
    print(f"          honest cross-fit LB anchor = {NB1242_HONEST_LB_ANCHOR:.4f}")
    print("=" * 78)

    # ---- Load 513 test, unblind index + truth, anchors ----
    te = load_test()
    te_smiles = te["smiles"].values
    te_names = te["name"].values
    n_test = len(te_smiles)

    te_nb1070 = np.load(DATA_PROCESSED / f"te_{ANCHOR}.npy").astype(np.float64)
    nb1070_oof = np.load(DATA_PROCESSED / f"{ANCHOR}_pred_oof.npy").astype(np.float64)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    assert te_nb1070.shape[0] == n_test
    assert nb1070_oof.shape[0] == n_unb

    rae_anchor_oof = float(rae(y_unb, nb1070_oof))
    rae_anchor_te_in = float(rae(y_unb, te_nb1070[unb_idx]))
    print(f"[load] te_{ANCHOR}.npy shape={te_nb1070.shape}  "
          f"in_RAE(unb_idx) = {rae_anchor_te_in:.4f}")
    print(f"[load] {ANCHOR}_pred_oof.npy shape={nb1070_oof.shape}  "
          f"pooled RAE = {rae_anchor_oof:.4f}")

    residual_target = y_unb - nb1070_oof
    print(f"[resid] target mean={residual_target.mean():+.4f}  "
          f"std={residual_target.std():.4f}")

    # ---- Try existing per-test ChEMBL feature caches first ----
    pred_chembl_path = DATA_PROCESSED / "pred_chembl_pec50_513.npy"
    sim_chembl_path = DATA_PROCESSED / "sim_chembl_513.npy"
    if pred_chembl_path.exists() and sim_chembl_path.exists():
        pred_chembl = np.load(pred_chembl_path).astype(np.float32)
        sim_chembl = np.load(sim_chembl_path).astype(np.float32)
        if pred_chembl.shape[0] != n_test or sim_chembl.shape[0] != n_test:
            raise ValueError(
                f"Cached ChEMBL feats shape mismatch: "
                f"pred={pred_chembl.shape}, sim={sim_chembl.shape}, expected ({n_test},)"
            )
        n_chembl_pool = -1
        pool_median = float("nan")
        n_test_overlap_dropped = -1
        print(f"[feat] CACHED  pred_chembl_pec50_513.npy + sim_chembl_513.npy loaded")
    else:
        # ---- Rebuild ChEMBL pool + kNN features (mirror nb1242) ----
        print("\n" + "-" * 78)
        print("CHEMBL PXR POOL (local cache union)")
        print("-" * 78)
        pool = _load_chembl_pool()

        # Test InChIKey leak guard
        test_mols = [standardize(s) for s in te_smiles]
        test_inchikeys = set()
        for m in test_mols:
            ik = _safe_inchikey(m)
            if ik is not None:
                test_inchikeys.add(ik)
        n_before = len(pool)
        pool = pool[~pool["inchikey"].isin(test_inchikeys)].reset_index(drop=True)
        n_after = len(pool)
        n_test_overlap_dropped = n_before - n_after
        print(f"   [leak] pool: {n_before} -> {n_after}  "
              f"(dropped {n_test_overlap_dropped} test-overlapping cpds)")

        # Morgan FPs
        fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
        keep_pool = fp_pool.sum(axis=1) > 0
        if not keep_pool.all():
            pool = pool[keep_pool].reset_index(drop=True)
            fp_pool = fp_pool[keep_pool]
        pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
        pool_median = float(np.median(pool_labels))
        n_chembl_pool = int(len(pool))
        print(f"   [pool] final size = {n_chembl_pool}  "
              f"median pEC50 = {pool_median:.3f}")

        std_test_smiles = []
        for m in test_mols:
            std_test_smiles.append("" if m is None else Chem.MolToSmiles(m))
        fp_test = morgan_fp_batch(std_test_smiles)
        print(f"   [fp] test FP shape = {fp_test.shape}  "
              f"density={fp_test.mean():.4f}")

        top_idx, top_sim = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
        pred_chembl, sim_chembl = _knn_predict(
            top_idx, top_sim, pool_labels, fallback=pool_median
        )
        np.save(pred_chembl_path, pred_chembl.astype(np.float32))
        np.save(sim_chembl_path, sim_chembl.astype(np.float32))
        print(f"[save] {pred_chembl_path}")
        print(f"[save] {sim_chembl_path}")

    print(f"   pred_chembl_pec50  mean={pred_chembl.mean():.3f}  "
          f"std={pred_chembl.std():.3f}  "
          f"min={pred_chembl.min():.3f}  max={pred_chembl.max():.3f}")
    print(f"   sim_chembl         mean={sim_chembl.mean():.3f}  "
          f"std={sim_chembl.std():.3f}  "
          f"min={sim_chembl.min():.3f}  max={sim_chembl.max():.3f}")

    # ---- Build (513, 169) deploy feature matrix + (253, 169) train slice ----
    X_maccs_te = np.load(MACCS_TE_PATH).astype(np.float32)
    if X_maccs_te.shape[0] != n_test:
        raise ValueError(f"MACCS cache shape mismatch: {X_maccs_te.shape}")

    X_test = np.concatenate(
        [X_maccs_te,
         pred_chembl.reshape(-1, 1).astype(np.float32),
         sim_chembl.reshape(-1, 1).astype(np.float32)],
        axis=1,
    ).astype(np.float32)
    X_unb = X_test[unb_idx]
    feat_dim = X_test.shape[1]
    print(f"[feat] X_test shape = {X_test.shape}  X_unb shape = {X_unb.shape}  "
          f"(MACCS-167 + pred_chembl + sim, dim={feat_dim})")

    # ---- 5-seed deploy bag (fit on all 253, predict 513 residual) ----
    print("\n" + "-" * 78)
    print(f"PER-SEED DEPLOY (fit on ALL {n_unb} unblind, predict 513) -- 5-seed bag")
    print("-" * 78)
    per_seed_resid_513 = np.zeros((len(RESID_SEEDS), n_test), dtype=np.float64)
    per_seed_records = []
    for j, s in enumerate(RESID_SEEDS):
        mdl = LGBMRegressor(**_lgbm_params(s))
        mdl.fit(X_unb, residual_target)
        resid_pred_513 = mdl.predict(X_test).astype(np.float64)
        per_seed_resid_513[j] = resid_pred_513
        te_seed = te_nb1070 + resid_pred_513
        in_rae_s = float(rae(y_unb, te_seed[unb_idx]))
        per_seed_records.append({
            "seed": int(s),
            "in_rae_te_seed": in_rae_s,
            "resid_513_mean": float(resid_pred_513.mean()),
            "resid_513_std": float(resid_pred_513.std()),
            "resid_513_min": float(resid_pred_513.min()),
            "resid_513_max": float(resid_pred_513.max()),
        })
        print(f"   seed {s:3d}:  in_RAE(te_seed[unb]) = {in_rae_s:.4f}  "
              f"resid_513 mean={resid_pred_513.mean():+.4f} "
              f"std={resid_pred_513.std():.4f}")

    # ---- Mean-bag residual + final deploy ----
    te_residual_513 = per_seed_resid_513.mean(axis=0)
    te_nb1250 = te_nb1070 + te_residual_513
    in_rae_mean = float(rae(y_unb, te_nb1250[unb_idx]))

    print("\n" + "=" * 78)
    print("MEAN-BAG DEPLOY")
    print("=" * 78)
    print(f"   te_residual_513   mean={te_residual_513.mean():+.4f}  "
          f"std={te_residual_513.std():.4f}  "
          f"min={te_residual_513.min():+.4f}  max={te_residual_513.max():+.4f}")
    print(f"   te_nb1250         mean={te_nb1250.mean():.3f}  "
          f"std={te_nb1250.std():.3f}  "
          f"min={te_nb1250.min():.3f}  max={te_nb1250.max():.3f}")
    print(f"   in_RAE(unb)       = {in_rae_mean:.4f}  "
          f"(honest cross-fit LB anchor = {NB1242_HONEST_LB_ANCHOR:.4f})")

    # ---- Save artifacts ----
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_path, te_nb1250.astype(np.float32))
    print(f"[save] {te_path}")

    csv_path = os.path.join(SUBMISSIONS_DIR, f"{TAG}_deploy_nb1242.csv")
    csv_info = _save_submission_csv(
        te_nb1250, te_smiles, te_names, csv_path, "nb1250"
    )
    print(f"[save] {csv_path}  rows={csv_info['n_rows']}  "
          f"cols={csv_info['columns']}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "feature_source": "maccs_cached_167+chembl_knn_2",
        "maccs_cache_test": str(MACCS_TE_PATH),
        "pred_chembl_path": str(pred_chembl_path),
        "sim_chembl_path": str(sim_chembl_path),
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "n_chembl_pool": int(n_chembl_pool) if n_chembl_pool >= 0 else None,
        "test_inchikeys_dropped_from_pool": int(n_test_overlap_dropped)
            if n_test_overlap_dropped >= 0 else None,
        "knn_k": KNN_K,
        "resid_seeds": RESID_SEEDS,
        "feature_dim": int(feat_dim),
        "lgbm_params_template": _lgbm_params(0),
        "rae_anchor_oof_253": rae_anchor_oof,
        "rae_anchor_te_in_sample_253": rae_anchor_te_in,
        "residual_target_mean": float(residual_target.mean()),
        "residual_target_std": float(residual_target.std()),
        "pred_chembl_stats": {
            "mean": float(pred_chembl.mean()),
            "std": float(pred_chembl.std()),
            "min": float(pred_chembl.min()),
            "max": float(pred_chembl.max()),
        },
        "sim_chembl_stats": {
            "mean": float(sim_chembl.mean()),
            "std": float(sim_chembl.std()),
            "min": float(sim_chembl.min()),
            "max": float(sim_chembl.max()),
        },
        "per_seed_records": per_seed_records,
        "te_stats": {
            "mean": float(te_nb1250.mean()),
            "std": float(te_nb1250.std()),
            "min": float(te_nb1250.min()),
            "max": float(te_nb1250.max()),
        },
        "in_rae_mean_bag_253": in_rae_mean,
        "crossfit_lb_anchor_nb1242": NB1242_HONEST_LB_ANCHOR,
        "te_path": str(te_path),
        "csv_path": csv_path,
        "wall_sec": round(time.time() - t0, 2),
        "note": (
            "POST-unblind deploy: each LGBM is fit on ALL 253 unblind rows, so "
            "in_RAE on te[unb_idx] is in-sample optimistic. The LB-faithful "
            "anchor is the honest 0.5431 cross-fit mean-bag RAE from nb1242."
        ),
    }
    summary_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {summary_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== STRUCTURED SUMMARY ====")
    print(f"  te_mean: {res['te_stats']['mean']:.4f}")
    print(f"  te_std:  {res['te_stats']['std']:.4f}")
    print(f"  te_min:  {res['te_stats']['min']:.4f}")
    print(f"  te_max:  {res['te_stats']['max']:.4f}")
    print(f"  in_rae_253: {res['in_rae_mean_bag_253']:.4f}")
    print(f"  crossfit_lb_anchor_nb1242: {res['crossfit_lb_anchor_nb1242']:.4f}")
    print(f"  te_path: {res['te_path']}")
    print(f"  csv_path: {res['csv_path']}")
