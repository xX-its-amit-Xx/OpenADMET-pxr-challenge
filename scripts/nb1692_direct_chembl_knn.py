"""nb1692 -- Direct ChEMBL kNN as standalone pEC50 predictor (no anchor).

Truly novel angle vs nb1242/nb1632 family: those used the kNN output as a
RESIDUAL FEATURE on top of an existing anchor (nb1070, chemprop_aux).
Here we use the similarity-weighted kNN mean DIRECTLY as the prediction
on the 253 unblind rows, no residual learning, no blending.

HYPOTHESIS:
    If the local ChEMBL PXR pool (~945 cpds after dedup) genuinely covers
    the scaffold space of the 253 unblind, then a sim-weighted top-5 kNN
    over Morgan-2048 Tanimoto should be competitive with chemprop_aux
    (0.6216). If RAE >> 0.62, the pool doesn't generalize at the unblind
    scaffold cluster and the residual-feature usage (nb1632, RAE 0.5107)
    is doing genuine work beyond what direct kNN can extract.

PROTOCOL:
    1. Load 513 test SMILES + 253 unblind indices + truth.
    2. Build ChEMBL PXR pool exactly like nb1242 (union of 3 local caches,
       standardize, InChIKey dedup, drop any test-overlap, Morgan-2048).
    3. For each unblind 253: top-5 Tanimoto neighbors in pool, sim-weighted
       mean pEC50. Rows with all-zero sim fall back to pool median.
    4. Pool RAE on 253 unblind.
    5. Verdict:
         RAE < 0.62  -> beats chemprop_aux standalone (huge if true)
         RAE < 0.51  -> beats nb1632 (impossible at this capacity, but check)
         RAE > 0.62  -> direct kNN insufficient; residual-feature framing
                        IS the right way to use ChEMBL pool
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
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1692"
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

CHEMPROP_AUX_REF = 0.6216
NB1632_REF = 0.5107


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
    """Same union recipe as nb1242 -- 3 local PXR caches, InChIKey dedup."""
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
        d["pec50"] = 9.0 - np.log10(d["standard_value"].astype(float))
        d = d[["canonical_smiles", "pec50"]].rename(
            columns={"canonical_smiles": "smiles"}
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
             n_meas=("pec50", "count"))
    )
    print(f"   [pool] after InChIKey dedup (median agg): {len(agg)} unique cpds")
    print(f"   [pool] pec50: mean={agg['pec50'].mean():.3f}  "
          f"std={agg['pec50'].std():.3f}  "
          f"min={agg['pec50'].min():.3f}  max={agg['pec50'].max():.3f}")
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


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- DIRECT ChEMBL kNN (k={KNN_K}, sim-weighted, NO anchor)")
    print(f"          pool = union(3 local ChEMBL PXR caches)")
    print(f"          metric = pooled RAE on 253 unblind")
    print("=" * 78)

    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")
    print(f"[load] y_unb: mean={y_unb.mean():.3f}  std={y_unb.std():.3f}  "
          f"min={y_unb.min():.3f}  max={y_unb.max():.3f}")

    # ---- ChEMBL pool ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL")
    print("-" * 78)
    pool = _load_chembl_pool()

    # ---- Test-overlap leak guard ----
    print("\n" + "-" * 78)
    print("TEST-OVERLAP LEAK GUARD")
    print("-" * 78)
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
    print("\n" + "-" * 78)
    print("MORGAN-2048 FINGERPRINTS")
    print("-" * 78)
    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    print(f"   pool FP: {fp_pool.shape}  density={fp_pool.mean():.4f}")
    keep_pool = fp_pool.sum(axis=1) > 0
    if not keep_pool.all():
        n_drop = int((~keep_pool).sum())
        print(f"   dropped {n_drop} pool rows with zero FP")
        pool = pool[keep_pool].reset_index(drop=True)
        fp_pool = fp_pool[keep_pool]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))
    pool_mean = float(np.mean(pool_labels))
    print(f"   final pool size: {len(pool)}  median pEC50 = {pool_median:.3f}  "
          f"mean = {pool_mean:.3f}")

    std_test_smiles = []
    for m in test_mols:
        std_test_smiles.append(Chem.MolToSmiles(m) if m is not None else "")
    fp_test = morgan_fp_batch(std_test_smiles)
    print(f"   test FP: {fp_test.shape}  density={fp_test.mean():.4f}")

    # ---- kNN ----
    print("\n" + "-" * 78)
    print(f"TANIMOTO kNN (k={KNN_K}) -- test (513) vs ChEMBL pool ({len(pool)})")
    print("-" * 78)
    top_idx, top_sim = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_513, mean_sim_513 = _knn_predict(
        top_idx, top_sim, pool_labels, fallback=pool_median
    )
    top1_sim = top_sim[:, 0]
    print(f"   pred_chembl    mean={pred_chembl_513.mean():.3f}  "
          f"std={pred_chembl_513.std():.3f}  "
          f"min={pred_chembl_513.min():.3f}  max={pred_chembl_513.max():.3f}")
    print(f"   top1 sim   p10={np.percentile(top1_sim, 10):.3f}  "
          f"p50={np.percentile(top1_sim, 50):.3f}  "
          f"p90={np.percentile(top1_sim, 90):.3f}  "
          f"max={top1_sim.max():.3f}")
    print(f"   mean5 sim  p10={np.percentile(mean_sim_513, 10):.3f}  "
          f"p50={np.percentile(mean_sim_513, 50):.3f}  "
          f"p90={np.percentile(mean_sim_513, 90):.3f}")
    n_zero = int((top1_sim < SIM_FLOOR).sum())
    print(f"   {n_zero}/513 test rows had no neighbor (fallback pool median)")

    # ---- Slice to 253 unblind and score ----
    pred_unb = pred_chembl_513[unb_idx]
    mean_sim_unb = mean_sim_513[unb_idx]
    top1_sim_unb = top1_sim[unb_idx]
    n_zero_unb = int((top1_sim_unb < SIM_FLOOR).sum())
    rae_direct = float(rae(y_unb, pred_unb))

    print("\n" + "-" * 78)
    print("EVALUATION ON 253 UNBLIND (direct kNN as final prediction)")
    print("-" * 78)
    print(f"   pred_unb    mean={pred_unb.mean():.3f}  std={pred_unb.std():.3f}")
    print(f"   y_unb       mean={y_unb.mean():.3f}  std={y_unb.std():.3f}")
    print(f"   top1 sim p50 (unb) = {np.percentile(top1_sim_unb, 50):.3f}")
    print(f"   {n_zero_unb}/253 unblind rows had no neighbor")
    print(f"   pooled RAE(direct kNN) = {rae_direct:.4f}")

    # ---- Comparisons ----
    delta_vs_chemprop_aux = rae_direct - CHEMPROP_AUX_REF
    delta_vs_nb1632 = rae_direct - NB1632_REF

    print("\n" + "-" * 78)
    print("COMPARISONS")
    print("-" * 78)
    print(f"   chemprop_aux ref     = {CHEMPROP_AUX_REF:.4f}")
    print(f"   nb1632 ref           = {NB1632_REF:.4f}")
    print(f"   direct kNN           = {rae_direct:.4f}")
    print(f"   delta vs chemprop_aux = {delta_vs_chemprop_aux:+.4f}")
    print(f"   delta vs nb1632       = {delta_vs_nb1632:+.4f}")

    beats_chemprop_aux = rae_direct < CHEMPROP_AUX_REF
    beats_nb1632 = rae_direct < NB1632_REF

    if beats_nb1632:
        verdict = "DIRECT_KNN_BEATS_NB1632_NEW_PRIMARY_CANDIDATE"
    elif beats_chemprop_aux:
        verdict = "DIRECT_KNN_BEATS_CHEMPROP_AUX_BUT_NOT_NB1632"
    elif rae_direct < CHEMPROP_AUX_REF + 0.05:
        verdict = "DIRECT_KNN_NEAR_CHEMPROP_AUX_USABLE_AS_BLEND_INPUT"
    else:
        verdict = "DIRECT_KNN_TOO_WEAK_RESIDUAL_FEATURE_FRAMING_IS_RIGHT"
    print(f"   verdict              = {verdict}")

    # ---- Save ----
    out_pred = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    np.save(out_pred, pred_unb.astype(np.float32))
    print(f"\n[save] {out_pred}")

    summary = {
        "tag": TAG,
        "method": "direct_chembl_knn_no_anchor",
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "n_chembl_pool_initial": int(n_before),
        "n_chembl_pool_after_leak_guard": int(n_after),
        "test_inchikeys_in_pool_dropped": int(n_before - n_after),
        "pool_pec50_mean": pool_mean,
        "pool_pec50_median": pool_median,
        "pool_pec50_std": float(np.std(pool_labels)),
        "knn_k": KNN_K,
        "top1_sim_p10_513": float(np.percentile(top1_sim, 10)),
        "top1_sim_p50_513": float(np.percentile(top1_sim, 50)),
        "top1_sim_p90_513": float(np.percentile(top1_sim, 90)),
        "top1_sim_max_513": float(top1_sim.max()),
        "n_zero_neighbor_513": int(n_zero),
        "n_zero_neighbor_unb": int(n_zero_unb),
        "top1_sim_p50_unb": float(np.percentile(top1_sim_unb, 50)),
        "mean5_sim_p50_unb": float(np.percentile(mean_sim_unb, 50)),
        "pred_unb_mean": float(pred_unb.mean()),
        "pred_unb_std": float(pred_unb.std()),
        "y_unb_mean": float(y_unb.mean()),
        "y_unb_std": float(y_unb.std()),
        "rae_direct_knn": rae_direct,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "nb1632_ref": NB1632_REF,
        "delta_vs_chemprop_aux": delta_vs_chemprop_aux,
        "delta_vs_nb1632": delta_vs_nb1632,
        "beats_chemprop_aux": bool(beats_chemprop_aux),
        "beats_nb1632": bool(beats_nb1632),
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
        "n_chembl_pool_after_leak_guard",
        "top1_sim_p50_unb", "mean5_sim_p50_unb", "n_zero_neighbor_unb",
        "pred_unb_mean", "pred_unb_std",
        "y_unb_mean", "y_unb_std",
        "rae_direct_knn",
        "delta_vs_chemprop_aux", "delta_vs_nb1632",
        "beats_chemprop_aux", "beats_nb1632",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
