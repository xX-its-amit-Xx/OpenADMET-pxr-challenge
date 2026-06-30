"""nb1360 -- Deploy nb1352 (SHAP-pruned MACCS top-20 + ChEMBL kNN feats) to 513.

Protocol:
    1. Reuse top-20 MACCS bit indices from nb1352_summary.json.
    2. Recompute pred_chembl_pec50 + sim on ALL 513 test rows using same
       local ChEMBL union pool as nb1352 (InChIKey-deduped, test-leak guard).
    3. Build pruned 22-col feature matrix for unb (253) and for full test (513).
    4. Fit 5-seed bag of shallow LGBM Huber on ALL 253 unblind rows with
       residual target y_unb - nb1070_pred_oof  (NO cross-fit -- deploy fit).
       Same hyperparams as nb1352:
           depth=3, num_leaves=7, n_estimators=80, lr=0.05,
           min_child_samples=20, huber_alpha=1.0
       Seeds [0, 1, 7, 42, 137].
    5. Predict residual on 513-row pruned feature matrix; collect (5, 513).
    6. Mean-bag and median-bag across 5 seeds -> two 513-vectors.
    7. te_nb1360_mean   = te_nb1070 + mean_bag_residual_513
       te_nb1360_median = te_nb1070 + median_bag_residual_513
    8. Save NPYs and 513-row CSVs (SMILES + Molecule Name + pEC50).

Reports te mean/std/min/max for both variants and in_RAE on unblind slice.
Honest LB anchors from nb1352 cross-fit: 0.5323 mean / 0.5315 median.

Outputs:
    data/processed/te_nb1360_mean.npy            (513,) float32
    data/processed/te_nb1360_median.npy          (513,) float32
    data/processed/nb1360_summary.json
    submissions/nb1360_deploy_nb1352_mean.csv
    submissions/nb1360_deploy_nb1352_median.csv
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

TAG = "nb1360"
ANCHOR = "nb1070"
NB1352_TAG = "nb1352"

RESID_SEEDS = [0, 1, 7, 42, 137]

MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"
SUBMISSIONS = Path(__file__).resolve().parents[1] / "submissions"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

NB1352_MEAN_ANCHOR = 0.5323
NB1352_MEDIAN_ANCHOR = 0.5315


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


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- DEPLOY nb1352 SHAP-pruned MACCS top-20 + ChEMBL to 513")
    print(f"          anchor={ANCHOR}  seeds={RESID_SEEDS}")
    print(f"          honest LB anchors  mean={NB1352_MEAN_ANCHOR}  median={NB1352_MEDIAN_ANCHOR}")
    print("=" * 78)

    # ---- Load top-20 MACCS bit indices from nb1352 summary ----
    nb1352_summary_path = DATA_PROCESSED / f"{NB1352_TAG}_summary.json"
    with open(nb1352_summary_path) as f:
        nb1352 = json.load(f)
    top_bit_idx_ranked = np.array(nb1352["top_maccs_bit_indices_ranked"], dtype=int)
    print(f"[load] nb1352 top-20 MACCS bits (ranked) = {top_bit_idx_ranked.tolist()}")

    # ---- Load test ----
    te = load_test()
    n_test = len(te)
    if "smiles" in te.columns:
        test_smiles = te["smiles"].astype(str).tolist()
    else:
        test_smiles = te["SMILES"].astype(str).tolist()
    if "molecule_name" in te.columns:
        mol_names = te["molecule_name"].astype(str).tolist()
    elif "Molecule Name" in te.columns:
        mol_names = te["Molecule Name"].astype(str).tolist()
    else:
        # fallback: any column with name pattern
        cand = [c for c in te.columns if "name" in c.lower()]
        if not cand:
            raise KeyError(f"No Molecule Name column found in test ({te.columns.tolist()})")
        mol_names = te[cand[0]].astype(str).tolist()

    print(f"[load] n_test={n_test}")

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_unb={n_unb}")

    # ---- Anchor (513) and anchor-OOF (253) ----
    te_anchor_513 = np.load(DATA_PROCESSED / f"te_{ANCHOR}.npy").astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(f"te_{ANCHOR} shape mismatch: {te_anchor_513.shape}")
    anchor_oof_253 = np.load(DATA_PROCESSED / f"{ANCHOR}_pred_oof.npy").astype(np.float64)
    if anchor_oof_253.shape[0] != n_unb:
        raise ValueError(f"{ANCHOR} OOF shape mismatch: {anchor_oof_253.shape}")
    rae_anchor = float(rae(y_unb, anchor_oof_253))
    print(f"[anchor] {ANCHOR}_pred_oof RAE = {rae_anchor:.4f}")
    print(f"[anchor] te_{ANCHOR}  mean={te_anchor_513.mean():.4f}  "
          f"std={te_anchor_513.std():.4f}  "
          f"min={te_anchor_513.min():.4f}  max={te_anchor_513.max():.4f}")

    # ---- ChEMBL pool (same union as nb1352) ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL (local cache; identical to nb1352)")
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

    # ---- Morgan fingerprints ----
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

    # ---- kNN k=5 Tanimoto on 513 ----
    top_idx_te, top_sim_te = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50_513, mean_sim_513 = _knn_predict(
        top_idx_te, top_sim_te, pool_labels, fallback=pool_median
    )
    print(f"   pred_chembl_pec50_513  mean={pred_chembl_pec50_513.mean():.3f}  "
          f"std={pred_chembl_pec50_513.std():.3f}")
    print(f"   sim_chembl_513          mean={mean_sim_513.mean():.3f}  "
          f"std={mean_sim_513.std():.3f}")

    # ---- MACCS-167 (full test cache) ----
    X_maccs_te = np.load(MACCS_TE_PATH)
    if X_maccs_te.shape[0] != n_test:
        raise ValueError(f"MACCS cache shape mismatch: {X_maccs_te.shape}")
    n_maccs = int(X_maccs_te.shape[1])
    print(f"   MACCS cache shape = {X_maccs_te.shape}  (n_bits={n_maccs})")

    # ---- Build PRUNED 22-col matrices ----
    # 513 test:
    X_maccs_te_pruned = X_maccs_te[:, top_bit_idx_ranked].astype(np.float32)
    X_te_pruned = np.concatenate(
        [
            X_maccs_te_pruned,
            pred_chembl_pec50_513.reshape(-1, 1).astype(np.float32),
            mean_sim_513.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    print(f"   X_te_pruned (513)  shape = {X_te_pruned.shape}")

    # 253 unblind:
    X_maccs_unb = X_maccs_te[unb_idx].astype(np.float32)
    X_maccs_unb_pruned = X_maccs_unb[:, top_bit_idx_ranked]
    pred_chembl_unb = pred_chembl_pec50_513[unb_idx].astype(np.float32)
    mean_sim_unb = mean_sim_513[unb_idx].astype(np.float32)
    X_unb_pruned = np.concatenate(
        [
            X_maccs_unb_pruned,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    print(f"   X_unb_pruned (253) shape = {X_unb_pruned.shape}")

    # ---- Residual target on 253 ----
    residual_unb = y_unb - anchor_oof_253
    print(f"[resid] mean={residual_unb.mean():+.4f}  std={residual_unb.std():.4f}")

    # ---- 5-seed deploy bag: fit on ALL 253, predict on 513 ----
    print("\n" + "-" * 78)
    print(f"DEPLOY 5-SEED LGBM HUBER BAG (fit on all 253, predict 513)")
    print("-" * 78)
    per_seed_resid_513 = np.zeros((len(RESID_SEEDS), n_test), dtype=np.float64)
    per_seed_in_sample_rae: list[float] = []
    per_seed_records = []
    for i, s in enumerate(RESID_SEEDS):
        mdl = LGBMRegressor(**_lgbm_params(s))
        mdl.fit(X_unb_pruned, residual_unb)
        # In-sample residual on 253 -> in-sample corrected pred -> in-sample RAE
        resid_in = mdl.predict(X_unb_pruned)
        corr_in = anchor_oof_253 + resid_in
        in_rae = float(rae(y_unb, corr_in))
        per_seed_in_sample_rae.append(in_rae)
        # Predict on 513
        resid_513 = mdl.predict(X_te_pruned)
        per_seed_resid_513[i] = resid_513
        per_seed_records.append({
            "seed": int(s),
            "in_sample_rae_253": in_rae,
            "resid_513_mean": float(resid_513.mean()),
            "resid_513_std": float(resid_513.std()),
        })
        print(f"   seed {s:3d}:  in_sample_RAE_253 = {in_rae:.4f}  "
              f"resid_513.mean = {resid_513.mean():+.4f}  "
              f"resid_513.std = {resid_513.std():.4f}")

    mean_bag_resid_513 = per_seed_resid_513.mean(axis=0)
    median_bag_resid_513 = np.median(per_seed_resid_513, axis=0)

    te_nb1360_mean = te_anchor_513 + mean_bag_resid_513
    te_nb1360_median = te_anchor_513 + median_bag_resid_513

    # ---- In-sample RAE on unblind slice (deploy-fit, in-sample!) ----
    in_rae_mean = float(rae(y_unb, te_nb1360_mean[unb_idx]))
    in_rae_median = float(rae(y_unb, te_nb1360_median[unb_idx]))

    print("\n" + "-" * 78)
    print("513-ROW DEPLOY VECTOR DIAGNOSTICS")
    print("-" * 78)
    print(f"   te_nb1360_mean    mean={te_nb1360_mean.mean():.4f}  "
          f"std={te_nb1360_mean.std():.4f}  "
          f"min={te_nb1360_mean.min():.4f}  max={te_nb1360_mean.max():.4f}")
    print(f"   te_nb1360_median  mean={te_nb1360_median.mean():.4f}  "
          f"std={te_nb1360_median.std():.4f}  "
          f"min={te_nb1360_median.min():.4f}  max={te_nb1360_median.max():.4f}")
    print(f"   in_RAE(unb, mean_bag)   = {in_rae_mean:.4f}   "
          f"(honest LB anchor {NB1352_MEAN_ANCHOR})")
    print(f"   in_RAE(unb, median_bag) = {in_rae_median:.4f}   "
          f"(honest LB anchor {NB1352_MEDIAN_ANCHOR})")

    # ---- Save NPYs ----
    np.save(DATA_PROCESSED / f"te_{TAG}_mean.npy",
            te_nb1360_mean.astype(np.float32))
    np.save(DATA_PROCESSED / f"te_{TAG}_median.npy",
            te_nb1360_median.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'te_{TAG}_mean.npy'}")
    print(f"[save] {DATA_PROCESSED / f'te_{TAG}_median.npy'}")

    # ---- Save CSVs (SMILES + Molecule Name + pEC50) ----
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    csv_mean = SUBMISSIONS / f"{TAG}_deploy_nb1352_mean.csv"
    csv_median = SUBMISSIONS / f"{TAG}_deploy_nb1352_median.csv"
    df_mean = pd.DataFrame({
        "SMILES": test_smiles,
        "Molecule Name": mol_names,
        "pEC50": te_nb1360_mean.astype(np.float64),
    })
    df_median = pd.DataFrame({
        "SMILES": test_smiles,
        "Molecule Name": mol_names,
        "pEC50": te_nb1360_median.astype(np.float64),
    })
    df_mean.to_csv(csv_mean, index=False)
    df_median.to_csv(csv_median, index=False)
    print(f"[save] {csv_mean}")
    print(f"[save] {csv_median}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "parent_method": NB1352_TAG,
        "top_maccs_bit_indices_ranked": top_bit_idx_ranked.tolist(),
        "feat_dim_pruned": int(X_te_pruned.shape[1]),
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "n_chembl_pool": int(len(pool)),
        "n_maccs_bits": int(n_maccs),
        "resid_seeds": RESID_SEEDS,
        "lgbm_max_depth": 3,
        "lgbm_num_leaves": 7,
        "lgbm_n_estimators": 80,
        "lgbm_learning_rate": 0.05,
        "lgbm_min_child_samples": 20,
        "lgbm_huber_alpha": 1.0,
        "rae_anchor_nb1070_oof_253": rae_anchor,
        "residual_unb_mean": float(residual_unb.mean()),
        "residual_unb_std": float(residual_unb.std()),
        "per_seed_records": per_seed_records,
        "per_seed_in_sample_rae_253": per_seed_in_sample_rae,
        "te_nb1360_mean_stats": {
            "mean": float(te_nb1360_mean.mean()),
            "std": float(te_nb1360_mean.std()),
            "min": float(te_nb1360_mean.min()),
            "max": float(te_nb1360_mean.max()),
        },
        "te_nb1360_median_stats": {
            "mean": float(te_nb1360_median.mean()),
            "std": float(te_nb1360_median.std()),
            "min": float(te_nb1360_median.min()),
            "max": float(te_nb1360_median.max()),
        },
        "in_rae_unb_mean_bag": in_rae_mean,
        "in_rae_unb_median_bag": in_rae_median,
        "honest_lb_anchor_mean": NB1352_MEAN_ANCHOR,
        "honest_lb_anchor_median": NB1352_MEDIAN_ANCHOR,
        "csv_mean": str(csv_mean),
        "csv_median": str(csv_median),
        "te_npy_mean": str(DATA_PROCESSED / f"te_{TAG}_mean.npy"),
        "te_npy_median": str(DATA_PROCESSED / f"te_{TAG}_median.npy"),
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
        "n_test", "n_unb", "n_chembl_pool", "feat_dim_pruned",
        "rae_anchor_nb1070_oof_253",
        "per_seed_in_sample_rae_253",
        "te_nb1360_mean_stats",
        "te_nb1360_median_stats",
        "in_rae_unb_mean_bag",
        "in_rae_unb_median_bag",
        "honest_lb_anchor_mean",
        "honest_lb_anchor_median",
        "csv_mean",
        "csv_median",
    ):
        print(f"  {k}: {res.get(k)}")
