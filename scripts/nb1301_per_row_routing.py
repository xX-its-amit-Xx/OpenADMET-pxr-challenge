"""nb1301 -- Per-row sim-weighted blend between nb1190 (BoB) and nb1242 (ChEMBL).

Hypothesis:
    nb1190 leverages TRAIN kNN signal; nb1242 leverages CHEMBL kNN signal.
    Test rows with high sim to train should trust nb1190; rows with high sim
    to ChEMBL should trust nb1242.  A per-row routing rule may beat the
    fixed-w optimum from nb1290 (w=0.35 on nb1190 -> RAE 0.5390).

Protocol:
    1. Load nb1190_bob_mean_oof.npy (253,), nb1242_mean_bag_oof.npy (253,).
    2. Compute per-row sim:
        sim_train  = top-1 Tanimoto vs train 4139 Morgan FPs.
        sim_chembl = top-1 Tanimoto vs ChEMBL pool Morgan FPs (replicate
                     nb1242 pool construction so the ranking is faithful).
        Slice both by `unb_idx` to get the 253-row vectors.
    3. Variants:
        a. soft_linear  : w_chembl = sim_chembl / (sim_train + sim_chembl + eps)
        b. soft_temp_T  : softmax([sim_train, sim_chembl] / T) -> w_chembl
                          for T in {0.05, 0.1, 0.2, 0.5}
        c. sigmoid_k    : w_chembl = sigmoid(k * (sim_chembl - sim_train))
                          for k in {2, 5, 10, 20}
        d. hard         : w_chembl = 1.0 if sim_chembl > sim_train else 0.0
        e. baseline_w035: fixed-w optimum from nb1290
    4. Per-variant pooled RAE.  Best variant.
    5. Verdict at 0.003 margin vs nb1290 best-w (0.5390).

Outputs:
    scripts/nb1301_per_row_routing.py
    data/processed/nb1301_summary.json
    data/processed/nb1301_best_oof.npy
    data/processed/nb1301_sim_train_unb.npy
    data/processed/nb1301_sim_chembl_unb.npy
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
from pxr.data import load_test, load_train
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1301"

EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

NB1190_REF = 0.5499
NB1242_REF = 0.5431
NB1290_BEST_W = 0.35  # weight on nb1190; weight on nb1242 = 0.65
NB1290_BEST_RAE = 0.5390
MARGIN = 0.003
EPS = 1e-6


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
    """Replicates nb1242 ChEMBL pool union+dedup logic (canonical caches)."""
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


def _tanimoto_top1(fp_q: np.ndarray, fp_pool: np.ndarray) -> np.ndarray:
    """Top-1 Tanimoto for each row of fp_q against fp_pool (uint8 0/1)."""
    a = fp_q.astype(np.float32)
    b = fp_pool.astype(np.float32)
    a_sum = a.sum(axis=1)
    b_sum = b.sum(axis=1)
    n_q = a.shape[0]
    top1 = np.zeros(n_q, dtype=np.float32)
    BLOCK = 64
    for s in range(0, n_q, BLOCK):
        e = min(n_q, s + BLOCK)
        inter = a[s:e] @ b.T
        denom = a_sum[s:e, None] + b_sum[None, :] - inter
        denom = np.maximum(denom, 1.0)
        sim = inter / denom
        top1[s:e] = sim.max(axis=1)
    return top1


def _rae_blend(p_nb1190: np.ndarray, p_nb1242: np.ndarray,
               w_chembl: np.ndarray, y: np.ndarray) -> float:
    blend = w_chembl * p_nb1242 + (1.0 - w_chembl) * p_nb1190
    return float(rae(y, blend))


def _blend(p_nb1190: np.ndarray, p_nb1242: np.ndarray,
           w_chembl: np.ndarray) -> np.ndarray:
    return w_chembl * p_nb1242 + (1.0 - w_chembl) * p_nb1190


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softmax_w_chembl(sim_train: np.ndarray, sim_chembl: np.ndarray,
                       T: float) -> np.ndarray:
    """softmax over [sim_train, sim_chembl] / T; return w_chembl in (0, 1)."""
    a = sim_train / T
    b = sim_chembl / T
    m = np.maximum(a, b)
    ea = np.exp(a - m)
    eb = np.exp(b - m)
    return eb / (ea + eb)


def _summarize(name: str, x: np.ndarray) -> dict:
    return {
        "name": name,
        "min": float(x.min()),
        "p10": float(np.percentile(x, 10)),
        "p25": float(np.percentile(x, 25)),
        "p50": float(np.percentile(x, 50)),
        "p75": float(np.percentile(x, 75)),
        "p90": float(np.percentile(x, 90)),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std()),
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- per-row sim-weighted blend (nb1190 vs nb1242)")
    print(f"          variants: soft_linear / softmax(T) / sigmoid(k) / hard")
    print(f"          verdict margin {MARGIN} vs nb1290 best-w ({NB1290_BEST_RAE:.4f})")
    print("=" * 78)

    # ---- Load ----
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    n_unb = len(y_unb)

    p_nb1190 = np.load(DATA_PROCESSED / "nb1190_bob_mean_oof.npy").astype(np.float64)
    p_nb1242 = np.load(DATA_PROCESSED / "nb1242_mean_bag_oof.npy").astype(np.float64)
    if p_nb1190.shape[0] != n_unb or p_nb1242.shape[0] != n_unb:
        raise ValueError(
            f"shape mismatch: nb1190={p_nb1190.shape}, nb1242={p_nb1242.shape}, "
            f"n_unb={n_unb}"
        )

    rae_nb1190 = float(rae(y_unb, p_nb1190))
    rae_nb1242 = float(rae(y_unb, p_nb1242))
    print(f"\n[load] standalone:")
    print(f"   nb1190 RAE = {rae_nb1190:.4f}  (ref {NB1190_REF:.4f})")
    print(f"   nb1242 RAE = {rae_nb1242:.4f}  (ref {NB1242_REF:.4f})")
    print(f"   nb1290 best-w RAE = {NB1290_BEST_RAE:.4f}  "
          f"(w_chembl={1.0 - NB1290_BEST_W:.2f})")

    # ---- Test SMILES, standardized + Morgan FPs ----
    print("\n" + "-" * 78)
    print("STANDARDIZE 513 TEST + 4139 TRAIN; MORGAN-2048")
    print("-" * 78)
    te = load_test()
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    n_test = len(test_smiles)
    test_mols = [standardize(s) for s in test_smiles]
    std_test_smiles = [_safe_can_smiles(m) or "" for m in test_mols]
    fp_test = morgan_fp_batch(std_test_smiles)
    print(f"   test FP: {fp_test.shape}  density={fp_test.mean():.4f}")

    tr = load_train()
    train_smiles_col = "smiles" if "smiles" in tr.columns else "SMILES"
    train_smiles = tr[train_smiles_col].astype(str).tolist()
    print(f"   loaded train n={len(train_smiles)} rows (may include CRC reps)")

    # Standardize + dedupe train by InChIKey (unique-compound pool)
    train_mols = [standardize(s) for s in train_smiles]
    train_inchikeys = [_safe_inchikey(m) for m in train_mols]
    train_std = [_safe_can_smiles(m) or "" for m in train_mols]
    seen = set()
    keep_train_idx = []
    for i, ik in enumerate(train_inchikeys):
        if ik is None:
            continue
        if ik in seen:
            continue
        seen.add(ik)
        keep_train_idx.append(i)
    train_unique_smiles = [train_std[i] for i in keep_train_idx]
    fp_train = morgan_fp_batch(train_unique_smiles)
    # Drop any zero-density FP (RDKit failure)
    keep = fp_train.sum(axis=1) > 0
    fp_train = fp_train[keep]
    print(f"   train unique-by-InChIKey FP: {fp_train.shape}  "
          f"density={fp_train.mean():.4f}")

    # ---- sim_train: 513 test vs train pool ----
    print("\n" + "-" * 78)
    print("TOP-1 TANIMOTO: 513 test vs TRAIN pool")
    print("-" * 78)
    sim_train_513 = _tanimoto_top1(fp_test, fp_train)
    sim_train_unb = sim_train_513[unb_idx].astype(np.float32)
    s_train_stats = _summarize("sim_train_unb", sim_train_unb)
    print(f"   sim_train_unb (253):  "
          f"p10={s_train_stats['p10']:.3f}  p50={s_train_stats['p50']:.3f}  "
          f"p90={s_train_stats['p90']:.3f}  mean={s_train_stats['mean']:.3f}")

    # ---- ChEMBL pool reconstruction (same as nb1242) ----
    print("\n" + "-" * 78)
    print("CHEMBL POOL (union of 3 caches; leak-guard against 513 InChIKeys)")
    print("-" * 78)
    pool = _load_chembl_pool()
    test_inchikeys = set(_safe_inchikey(m) for m in test_mols if m is not None)
    pool = pool[~pool["inchikey"].isin(test_inchikeys)].reset_index(drop=True)
    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep_pool = fp_pool.sum(axis=1) > 0
    pool = pool[keep_pool].reset_index(drop=True)
    fp_pool = fp_pool[keep_pool]
    print(f"   chembl pool (leak-guarded): {len(pool)} cpds")

    # ---- sim_chembl: 513 test vs ChEMBL pool ----
    print("\n" + "-" * 78)
    print("TOP-1 TANIMOTO: 513 test vs CHEMBL pool")
    print("-" * 78)
    sim_chembl_513 = _tanimoto_top1(fp_test, fp_pool)
    sim_chembl_unb = sim_chembl_513[unb_idx].astype(np.float32)
    s_chembl_stats = _summarize("sim_chembl_unb", sim_chembl_unb)
    print(f"   sim_chembl_unb (253): "
          f"p10={s_chembl_stats['p10']:.3f}  p50={s_chembl_stats['p50']:.3f}  "
          f"p90={s_chembl_stats['p90']:.3f}  mean={s_chembl_stats['mean']:.3f}")

    # ---- Sim divergence diagnostics ----
    diff = sim_chembl_unb - sim_train_unb
    s_diff_stats = _summarize("sim_chembl_minus_train", diff)
    print(f"\n[diag] (sim_chembl - sim_train): mean={s_diff_stats['mean']:+.3f}  "
          f"std={s_diff_stats['std']:.3f}  "
          f"p10={s_diff_stats['p10']:+.3f}  "
          f"p50={s_diff_stats['p50']:+.3f}  "
          f"p90={s_diff_stats['p90']:+.3f}")
    n_chembl_wins = int((sim_chembl_unb > sim_train_unb).sum())
    print(f"   rows where sim_chembl > sim_train: {n_chembl_wins}/{n_unb} "
          f"({100*n_chembl_wins/n_unb:.1f}%)")
    corr_sims = float(np.corrcoef(sim_train_unb, sim_chembl_unb)[0, 1])
    print(f"   Pearson(sim_train, sim_chembl) = {corr_sims:.3f}")

    # ---- Variants ----
    print("\n" + "-" * 78)
    print("PER-ROW VARIANTS")
    print("-" * 78)
    results = []
    best_variant = None
    best_rae = float("inf")
    best_w_chembl = None
    best_oof = None

    # a. soft linear
    w_a = sim_chembl_unb / (sim_train_unb + sim_chembl_unb + EPS)
    r_a = _rae_blend(p_nb1190, p_nb1242, w_a, y_unb)
    results.append({"variant": "soft_linear",
                    "params": {},
                    "rae": r_a,
                    "w_chembl_mean": float(w_a.mean()),
                    "w_chembl_std": float(w_a.std()),
                    "w_chembl_p10": float(np.percentile(w_a, 10)),
                    "w_chembl_p50": float(np.percentile(w_a, 50)),
                    "w_chembl_p90": float(np.percentile(w_a, 90))})
    print(f"   soft_linear         RAE={r_a:.4f}  "
          f"w_chembl mean={w_a.mean():.3f}  std={w_a.std():.3f}")
    if r_a < best_rae:
        best_rae, best_variant, best_w_chembl, best_oof = (
            r_a, "soft_linear", w_a, _blend(p_nb1190, p_nb1242, w_a))

    # b. softmax T variants
    for T in [0.05, 0.10, 0.20, 0.50]:
        w_b = _softmax_w_chembl(sim_train_unb, sim_chembl_unb, T)
        r_b = _rae_blend(p_nb1190, p_nb1242, w_b, y_unb)
        results.append({"variant": f"softmax_T{T}",
                        "params": {"T": T},
                        "rae": r_b,
                        "w_chembl_mean": float(w_b.mean()),
                        "w_chembl_std": float(w_b.std()),
                        "w_chembl_p10": float(np.percentile(w_b, 10)),
                        "w_chembl_p50": float(np.percentile(w_b, 50)),
                        "w_chembl_p90": float(np.percentile(w_b, 90))})
        print(f"   softmax T={T:.2f}     RAE={r_b:.4f}  "
              f"w_chembl mean={w_b.mean():.3f}  std={w_b.std():.3f}")
        if r_b < best_rae:
            best_rae, best_variant, best_w_chembl, best_oof = (
                r_b, f"softmax_T{T}", w_b, _blend(p_nb1190, p_nb1242, w_b))

    # c. sigmoid k variants
    for k in [2.0, 5.0, 10.0, 20.0]:
        w_c = _sigmoid(k * (sim_chembl_unb - sim_train_unb))
        r_c = _rae_blend(p_nb1190, p_nb1242, w_c, y_unb)
        results.append({"variant": f"sigmoid_k{k}",
                        "params": {"k": k},
                        "rae": r_c,
                        "w_chembl_mean": float(w_c.mean()),
                        "w_chembl_std": float(w_c.std()),
                        "w_chembl_p10": float(np.percentile(w_c, 10)),
                        "w_chembl_p50": float(np.percentile(w_c, 50)),
                        "w_chembl_p90": float(np.percentile(w_c, 90))})
        print(f"   sigmoid k={k:5.2f}    RAE={r_c:.4f}  "
              f"w_chembl mean={w_c.mean():.3f}  std={w_c.std():.3f}")
        if r_c < best_rae:
            best_rae, best_variant, best_w_chembl, best_oof = (
                r_c, f"sigmoid_k{k}", w_c, _blend(p_nb1190, p_nb1242, w_c))

    # d. hard threshold
    w_d = (sim_chembl_unb > sim_train_unb).astype(np.float64)
    r_d = _rae_blend(p_nb1190, p_nb1242, w_d, y_unb)
    results.append({"variant": "hard",
                    "params": {},
                    "rae": r_d,
                    "w_chembl_mean": float(w_d.mean()),
                    "w_chembl_std": float(w_d.std()),
                    "w_chembl_p10": float(np.percentile(w_d, 10)),
                    "w_chembl_p50": float(np.percentile(w_d, 50)),
                    "w_chembl_p90": float(np.percentile(w_d, 90))})
    print(f"   hard threshold      RAE={r_d:.4f}  "
          f"w_chembl mean={w_d.mean():.3f}  std={w_d.std():.3f}")
    if r_d < best_rae:
        best_rae, best_variant, best_w_chembl, best_oof = (
            r_d, "hard", w_d, _blend(p_nb1190, p_nb1242, w_d))

    # e. baseline fixed-w (nb1290 anchor)
    w_e = np.full(n_unb, 1.0 - NB1290_BEST_W, dtype=np.float64)  # w_chembl = 0.65
    r_e = _rae_blend(p_nb1190, p_nb1242, w_e, y_unb)
    results.append({"variant": "baseline_fixed_w0.35",
                    "params": {"w_nb1190": NB1290_BEST_W,
                               "w_nb1242": 1.0 - NB1290_BEST_W},
                    "rae": r_e,
                    "w_chembl_mean": float(w_e.mean()),
                    "w_chembl_std": float(w_e.std()),
                    "w_chembl_p10": float(np.percentile(w_e, 10)),
                    "w_chembl_p50": float(np.percentile(w_e, 50)),
                    "w_chembl_p90": float(np.percentile(w_e, 90))})
    print(f"   baseline w=0.65 chembl  RAE={r_e:.4f}  (nb1290 anchor)")
    if r_e < best_rae:
        best_rae, best_variant, best_w_chembl, best_oof = (
            r_e, "baseline_fixed_w0.35", w_e,
            _blend(p_nb1190, p_nb1242, w_e))

    # ---- Verdict ----
    delta_vs_nb1290 = best_rae - NB1290_BEST_RAE
    beats_nb1290 = best_rae < NB1290_BEST_RAE - MARGIN
    flat_nb1290 = abs(best_rae - NB1290_BEST_RAE) < MARGIN
    if beats_nb1290:
        verdict = (f"PER_ROW_ROUTING_BEATS_NB1290 "
                   f"({best_variant} @ {best_rae:.4f})")
    elif flat_nb1290:
        verdict = (f"PER_ROW_ROUTING_FLAT_VS_NB1290 "
                   f"({best_variant} @ {best_rae:.4f})")
    else:
        verdict = (f"PER_ROW_ROUTING_HURTS_VS_NB1290 "
                   f"({best_variant} @ {best_rae:.4f})")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"   nb1190 standalone   : {rae_nb1190:.4f}")
    print(f"   nb1242 standalone   : {rae_nb1242:.4f}")
    print(f"   nb1290 fixed-w best : {NB1290_BEST_RAE:.4f}  (anchor)")
    print(f"   best per-row variant: {best_variant} @ {best_rae:.4f}")
    print(f"   delta vs nb1290     : {delta_vs_nb1290:+.4f}")
    print(f"   beats_nb1290        : {beats_nb1290}")
    print(f"   verdict             : {verdict}")

    np.save(DATA_PROCESSED / f"{TAG}_best_oof.npy",
            best_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_sim_train_unb.npy",
            sim_train_unb.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_sim_chembl_unb.npy",
            sim_chembl_unb.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_best_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_sim_train_unb.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_sim_chembl_unb.npy'}")

    summary = {
        "tag": TAG,
        "n_unb": n_unb,
        "n_test": n_test,
        "n_train_pool_unique": int(fp_train.shape[0]),
        "n_chembl_pool": int(len(pool)),
        "standalone": {
            "nb1190": rae_nb1190,
            "nb1242": rae_nb1242,
            "nb1290_best_fixed_w": NB1290_BEST_RAE,
        },
        "sim_train_unb_stats": s_train_stats,
        "sim_chembl_unb_stats": s_chembl_stats,
        "sim_diff_stats": s_diff_stats,
        "sim_corr_pearson": corr_sims,
        "n_chembl_wins_sim": n_chembl_wins,
        "frac_chembl_wins_sim": float(n_chembl_wins / n_unb),
        "variants": results,
        "best_variant": best_variant,
        "best_rae": best_rae,
        "best_w_chembl_mean": float(np.asarray(best_w_chembl).mean()),
        "nb1290_best_rae": NB1290_BEST_RAE,
        "delta_vs_nb1290": delta_vs_nb1290,
        "beats_nb1290": bool(beats_nb1290),
        "flat_vs_nb1290": bool(flat_nb1290),
        "margin": MARGIN,
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
        "n_chembl_pool", "n_train_pool_unique",
        "sim_train_unb_stats", "sim_chembl_unb_stats",
        "sim_corr_pearson", "frac_chembl_wins_sim",
        "standalone",
        "best_variant", "best_rae",
        "delta_vs_nb1290", "beats_nb1290",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
