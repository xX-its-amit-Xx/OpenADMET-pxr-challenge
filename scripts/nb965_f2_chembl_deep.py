"""nb965 -- Deep F2-cohort analysis on the new ChEMBL KB (11,185 rows).

PER-CYCLE-129-SUPPORT FINDING:
    The tight F2 cohort defined as max_train_sim<0.40 (n=9) is too narrow
    to give a stable signal.  Memory `feedback_failure_mode_quantile_compression`
    documents the WIDE F2 cohort as `scaf_train_freq==0` (novel Murcko scaffold
    against the 4139 train) which yields ~50 rows on the 253 unblinded labels
    (90% of the 50 worst nb503 errors).

PROTOCOL:
    1. Load chembl_pxr_nr_kb.parquet (11185 rows; PXR_CHEMBL3401 + NR_PPARg/FXR/
       RXRa/LXRa/VDR/PXR/PPARa) and the 253 unblind labels.
    2. Define WIDE F2 = {r in unblind 253 : Murcko_scaf(r) not in train_scaffolds};
       report the cohort size.
    3. For every F2 row: compute
         sim_train = max Tanimoto vs 4139 train ECFP4
         sim_kb    = max Tanimoto vs 11185 ChEMBL KB ECFP4
         delta     = sim_kb - sim_train  (positive means KB neighbor is closer)
    4. Filter "KB-helpable" rows where delta > 0.05; report count.
    5. For KB-helpable rows: predict via Tanimoto-weighted kNN(k=5) FROM KB;
       compare to chemprop_aux (te[unb_idx]) and nb2103 K=28 mean-bag OOF.
    6. Apply per-target source weighting in the kNN: PXR_direct = 1.0,
       NR-related = 0.3 (cross-assay penalty).
    7. Build a CALIBRATED kNN with assay-bias correction: shift_factor =
       median(train_pec50 overlap with KB on matched inchikeys).
    8. Compute RAE on the F2-helpable subset using:
         (a) pure chemprop_aux
         (b) pure kNN (KB-only)
         (c) per-row blend by sim_kb
    9. If kNN beats anchor on this subset by >0.05 RAE: KB IS a real signal
       source -> recommend integration in cycle 131.

OUTPUTS:
    scripts/nb965_f2_chembl_deep.py
    data/processed/nb965_summary.json
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
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_train, load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb965"
ROOT = Path(__file__).resolve().parents[1]
EXT_DIR = ROOT / "data" / "external"

# F2 + KB-helpable thresholds
DELTA_HELPABLE_MIN = 0.05   # KB neighbor must be at least +0.05 Tanimoto closer
                            # than the train neighbor before we trust KB to help
KNN_K = 5                   # 5-NN as spec'd
DECISION_RAE_DELTA = 0.05   # KB recommended if it beats anchor by >= 0.05 RAE
                            # on the helpable subset

# Per-source weights for the kNN
SRC_WEIGHT_PXR_DIRECT = 1.0
SRC_WEIGHT_NR_RELATED = 0.3

# Refs from MEMORY
CHEMPROP_AUX_REF_FULL_253 = 0.6216  # full-253 honest RAE


def _safe_inchikey(mol):
    try:
        return Chem.MolToInchiKey(mol) if mol is not None else None
    except Exception:
        return None


def _safe_canon_smiles(mol):
    try:
        return Chem.MolToSmiles(mol) if mol is not None else None
    except Exception:
        return None


def _safe_murcko(mol):
    """Murcko scaffold SMILES, empty string if RDKit fails."""
    if mol is None:
        return ""
    try:
        return MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""


def _tanimoto_topk(fp_q: np.ndarray, fp_pool: np.ndarray, k: int):
    """For each query, return (top-k sim, top-k pool index) sorted desc.

    fp_q: (Nq, B) uint8; fp_pool: (Np, B) uint8
    returns: (Nq, k) float32 sim, (Nq, k) int32 idx
    """
    a = fp_q.astype(np.float32)
    b = fp_pool.astype(np.float32)
    a_sum = a.sum(axis=1)
    b_sum = b.sum(axis=1)
    nq = a.shape[0]
    np_ = b.shape[0]
    top_sim = np.zeros((nq, k), dtype=np.float32)
    top_idx = np.zeros((nq, k), dtype=np.int32)
    BLOCK = 256
    for s in range(0, nq, BLOCK):
        e = min(nq, s + BLOCK)
        inter = a[s:e] @ b.T
        denom = a_sum[s:e, None] + b_sum[None, :] - inter
        denom = np.maximum(denom, 1.0)
        sim = inter / denom
        # top-k per row (largest)
        kk = min(k, np_)
        if kk < np_:
            part = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
            rows = np.arange(part.shape[0])[:, None]
            part_sim = sim[rows, part]
            order = np.argsort(-part_sim, axis=1)
            top_idx[s:e, :kk] = part[rows, order]
            top_sim[s:e, :kk] = part_sim[rows, order]
        else:
            order = np.argsort(-sim, axis=1)
            top_idx[s:e, :np_] = order[:, :np_]
            top_sim[s:e, :np_] = np.take_along_axis(sim, order[:, :np_], axis=1)
    return top_sim, top_idx


def _tanimoto_max_only(fp_q: np.ndarray, fp_pool: np.ndarray) -> np.ndarray:
    """Per-query max Tanimoto to any pool entry (only)."""
    a = fp_q.astype(np.float32)
    b = fp_pool.astype(np.float32)
    a_sum = a.sum(axis=1)
    b_sum = b.sum(axis=1)
    nq = a.shape[0]
    out = np.zeros(nq, dtype=np.float32)
    BLOCK = 256
    for s in range(0, nq, BLOCK):
        e = min(nq, s + BLOCK)
        inter = a[s:e] @ b.T
        denom = a_sum[s:e, None] + b_sum[None, :] - inter
        denom = np.maximum(denom, 1.0)
        out[s:e] = (inter / denom).max(axis=1)
    return out


def _src_to_weight(src: str) -> float:
    s = str(src).upper()
    if "PXR" in s:
        return SRC_WEIGHT_PXR_DIRECT
    return SRC_WEIGHT_NR_RELATED


def _weighted_knn_predict(top_sim, top_idx, pool_pec50, pool_src_w):
    """Sim-weighted (sim * src_weight) k-NN regression.

    top_sim: (N, k) Tanimoto
    top_idx: (N, k) into pool
    pool_pec50: (Np,) pEC50
    pool_src_w: (Np,) per-row source weight
    """
    n, k = top_sim.shape
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        idx = top_idx[i]
        s = top_sim[i].astype(np.float64)
        w_src = pool_src_w[idx].astype(np.float64)
        w = s * w_src
        if w.sum() <= 0:
            out[i] = float(pool_pec50.mean())
        else:
            out[i] = float((w * pool_pec50[idx].astype(np.float64)).sum() / w.sum())
    return out


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- F2-cohort deep dive on ChEMBL PXR/NR KB (11185 rows)")
    print(f"          WIDE F2 = scaf_train_freq==0  (Murcko vs 4139 train)")
    print(f"          KB-helpable: delta = sim_kb - sim_train > {DELTA_HELPABLE_MIN}")
    print(f"          decision: KB recommended if RAE-improvement >= "
          f"{DECISION_RAE_DELTA} on helpable subset")
    print("=" * 78)

    # ---- Load train / test ----
    tr_df = load_train()
    te_df = load_test()
    n_tr, n_te = len(tr_df), len(te_df)
    print(f"\n[load] train={n_tr}  test={n_te}")

    print("[std] standardizing train SMILES + Murcko scaffolds...")
    tr_mols = [standardize(s) for s in tr_df["smiles"].tolist()]
    tr_std_smi = [_safe_canon_smiles(m) or "" for m in tr_mols]
    tr_inchikeys = [_safe_inchikey(m) for m in tr_mols]
    tr_scaf = [_safe_murcko(m) for m in tr_mols]
    train_scaf_set = set(s for s in tr_scaf if s)
    print(f"   train unique Murcko scaffolds: {len(train_scaf_set)} "
          f"(of {n_tr} rows)")

    print("[std] standardizing test SMILES + Murcko scaffolds...")
    te_mols = [standardize(s) for s in te_df["smiles"].tolist()]
    te_std_smi = [_safe_canon_smiles(m) or "" for m in te_mols]
    te_scaf = [_safe_murcko(m) for m in te_mols]

    # ---- Unblind 253 ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] unblind n={n_unb}")

    # Anchor: chemprop_aux on 513 -> slice unb_idx
    te_chemprop_aux = np.load(DATA_PROCESSED / "te_chemprop_aux.npy").astype(np.float64)
    anchor_unb = te_chemprop_aux[unb_idx]
    rae_anchor_full = float(rae(y_unb, anchor_unb))
    # nb2103 K=28 mean-bag OOF on 253
    nb2103_oof_unb = np.load(DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy").astype(np.float64)
    rae_nb2103_full = float(rae(y_unb, nb2103_oof_unb))
    print(f"[anchor full-253] chemprop_aux RAE = {rae_anchor_full:.4f} "
          f"(memory ref {CHEMPROP_AUX_REF_FULL_253:.4f})")
    print(f"[anchor full-253] nb2103 K=28 mean-bag OOF RAE = {rae_nb2103_full:.4f}")

    # ---- WIDE F2 cohort: scaf_train_freq == 0 among the 253 unblind ----
    print("\n" + "-" * 78)
    print("WIDE F2 COHORT (Murcko scaffold absent from train)")
    print("-" * 78)
    unb_scaf = [te_scaf[i] for i in unb_idx]
    is_f2 = np.array(
        [(s == "") or (s not in train_scaf_set) for s in unb_scaf], dtype=bool
    )
    # Treat empty-scaffold as F2 (very small / unusual molecule)
    n_f2 = int(is_f2.sum())
    print(f"   F2 cohort size (scaf_train_freq==0): {n_f2} / {n_unb}")

    # ---- Load + standardize KB ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR/NR KB (cycle-129-support)")
    print("-" * 78)
    kb_p = EXT_DIR / "chembl_pxr_nr_kb.parquet"
    kb = pd.read_parquet(kb_p)
    print(f"   raw KB rows: {len(kb)}  cols: {list(kb.columns)}")

    # Standardize KB SMILES -> kb_std_smi, kb_inchikey
    print("   standardizing KB SMILES (RDKit)...")
    kb_mols = [standardize(s) for s in kb["smiles"].tolist()]
    kb_std_smi = [_safe_canon_smiles(m) or "" for m in kb_mols]
    kb_inchikeys = [_safe_inchikey(m) for m in kb_mols]
    kb["std_smiles"] = kb_std_smi
    kb["inchikey_std"] = kb_inchikeys
    valid = (kb["std_smiles"] != "") & kb["inchikey_std"].notna() & kb["pec50_chembl"].notna()
    n_drop = int((~valid).sum())
    kb = kb[valid].reset_index(drop=True)
    print(f"   dropped invalid/std-fail/no-pec50: {n_drop}; kept {len(kb)}")

    # Dedup KB on inchikey (median pEC50, keep first source)
    print("   deduping KB by inchikey (median pEC50, first src)...")
    kb_dedup = (
        kb.groupby("inchikey_std", as_index=False)
        .agg(
            std_smiles=("std_smiles", "first"),
            pec50=("pec50_chembl", "median"),
            src=("source_target", "first"),
            n_meas=("pec50_chembl", "count"),
        )
    )
    print(f"   KB unique compounds: {len(kb_dedup)}")
    print(f"   KB src breakdown: {kb_dedup['src'].value_counts().to_dict()}")

    # Drop KB rows that overlap (inchikey) the 4139 train (assay-bias source)
    train_ik_set = set(ik for ik in tr_inchikeys if ik is not None)
    overlap_ik = kb_dedup["inchikey_std"].isin(train_ik_set)
    n_overlap = int(overlap_ik.sum())
    # Save the train-overlap subset to estimate assay-bias shift_factor
    overlap_df = kb_dedup[overlap_ik].copy()
    tr_pec50_by_ik = {ik: pec for ik, pec in zip(tr_inchikeys, tr_df["pec50"]) if ik is not None}
    overlap_df["pec50_train"] = overlap_df["inchikey_std"].map(tr_pec50_by_ik)
    overlap_df = overlap_df.dropna(subset=["pec50_train"])
    shift_factor = (
        float((overlap_df["pec50_train"] - overlap_df["pec50"]).median())
        if len(overlap_df) > 0 else 0.0
    )
    print(f"   KB-train inchikey overlap: {n_overlap} compounds; "
          f"shift_factor (median(train - kb)) = {shift_factor:+.3f}")
    print(f"   -> kNN-calibrated will ADD {shift_factor:+.3f} to KB pEC50 "
          f"(assay-bias correction)")

    # Drop overlap from KB to avoid trivial leakage
    kb_use = kb_dedup[~overlap_ik].reset_index(drop=True)
    print(f"   final KB used for kNN (excluding train overlaps): {len(kb_use)}")

    # ---- Tanimoto sims: train vs unb-F2, KB vs unb-F2 ----
    print("\n" + "-" * 78)
    print("TANIMOTO MAX-SIM: F2 unb cohort vs train AND vs KB")
    print("-" * 78)
    f2_unb_idx_local = np.where(is_f2)[0]            # indices into 0..252
    f2_unb_idx_513 = unb_idx[f2_unb_idx_local]       # indices into 0..512
    f2_smi = [te_std_smi[i] for i in f2_unb_idx_513]
    f2_y = y_unb[f2_unb_idx_local]
    f2_anchor = anchor_unb[f2_unb_idx_local]
    f2_nb2103 = nb2103_oof_unb[f2_unb_idx_local]
    print(f"   F2 unblind count = {len(f2_smi)}")
    print(f"   F2 truth pEC50: mean={f2_y.mean():.3f} std={f2_y.std():.3f} "
          f"range=[{f2_y.min():.2f}, {f2_y.max():.2f}]")

    print("   computing Morgan fingerprints for F2 + train + KB...")
    fp_f2 = morgan_fp_batch(f2_smi)
    fp_tr = morgan_fp_batch(tr_std_smi)
    fp_kb = morgan_fp_batch(kb_use["std_smiles"].tolist())
    # drop any zero-bit FPs in kb
    kb_keep = fp_kb.sum(axis=1) > 0
    if not kb_keep.all():
        kb_use = kb_use[kb_keep].reset_index(drop=True)
        fp_kb = fp_kb[kb_keep]
    print(f"   shapes: fp_f2={fp_f2.shape} fp_tr={fp_tr.shape} fp_kb={fp_kb.shape}")

    sim_train = _tanimoto_max_only(fp_f2, fp_tr)
    sim_kb_top, idx_kb_top = _tanimoto_topk(fp_f2, fp_kb, KNN_K)
    sim_kb = sim_kb_top[:, 0]
    delta = sim_kb - sim_train
    helpable_mask = delta > DELTA_HELPABLE_MIN
    n_help = int(helpable_mask.sum())
    print(f"   F2 helpable (delta > {DELTA_HELPABLE_MIN}): {n_help} / {len(f2_smi)}")
    if n_help == 0:
        print("   [skip] no helpable rows -> verdict: KB NOT a real F2 signal source")
    print(f"   distribution of sim_train (F2): "
          f"mean={sim_train.mean():.3f} med={np.median(sim_train):.3f}")
    print(f"   distribution of sim_kb    (F2): "
          f"mean={sim_kb.mean():.3f} med={np.median(sim_kb):.3f}")
    print(f"   distribution of delta     (F2): "
          f"mean={delta.mean():+.3f} med={np.median(delta):+.3f}")

    # ---- Per-source weights for KB ----
    kb_src_w = kb_use["src"].apply(_src_to_weight).to_numpy()
    n_pxr_direct = int((kb_src_w == SRC_WEIGHT_PXR_DIRECT).sum())
    n_nr_related = int((kb_src_w == SRC_WEIGHT_NR_RELATED).sum())
    print(f"   KB src weights: PXR_direct={n_pxr_direct} (w=1.0)  "
          f"NR_related={n_nr_related} (w=0.3)")

    # ---- kNN(k=5) predictions for F2 rows ----
    print("\n" + "-" * 78)
    print("kNN(k=5) PREDICTIONS ON F2 COHORT (uncalibrated + assay-bias-calibrated)")
    print("-" * 78)
    pool_pec50 = kb_use["pec50"].to_numpy().astype(np.float64)
    knn_raw = _weighted_knn_predict(sim_kb_top, idx_kb_top, pool_pec50, kb_src_w)
    knn_cal = knn_raw + shift_factor

    rae_f2_anchor = float(rae(f2_y, f2_anchor))
    rae_f2_nb2103 = float(rae(f2_y, f2_nb2103))
    rae_f2_knn_raw = float(rae(f2_y, knn_raw))
    rae_f2_knn_cal = float(rae(f2_y, knn_cal))
    print(f"   F2 full ({len(f2_smi)} rows):")
    print(f"      chemprop_aux   RAE = {rae_f2_anchor:.4f}")
    print(f"      nb2103 K=28    RAE = {rae_f2_nb2103:.4f}")
    print(f"      kNN raw        RAE = {rae_f2_knn_raw:.4f}")
    print(f"      kNN calibrated RAE = {rae_f2_knn_cal:.4f}")

    # ---- Restrict to helpable subset ----
    if n_help > 0:
        y_h = f2_y[helpable_mask]
        anchor_h = f2_anchor[helpable_mask]
        nb2103_h = f2_nb2103[helpable_mask]
        knn_raw_h = knn_raw[helpable_mask]
        knn_cal_h = knn_cal[helpable_mask]
        sim_kb_h = sim_kb[helpable_mask]

        rae_h_anchor = float(rae(y_h, anchor_h))
        rae_h_nb2103 = float(rae(y_h, nb2103_h))
        rae_h_knn_raw = float(rae(y_h, knn_raw_h))
        rae_h_knn_cal = float(rae(y_h, knn_cal_h))

        # per-row blend by sim_kb (closer KB = higher trust)
        # convex weight clipped to [0.0, 0.8] so the chemprop anchor always gets >= 0.2
        w_blend = np.clip(sim_kb_h, 0.0, 0.8).astype(np.float64)
        # CALIBRATED kNN inside the blend
        blended_h = w_blend * knn_cal_h + (1.0 - w_blend) * anchor_h
        rae_h_blend = float(rae(y_h, blended_h))

        # also a constant-weight grid (KB calibrated)
        const_blend_grid = []
        for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            r_b = float(rae(y_h, w * knn_cal_h + (1.0 - w) * anchor_h))
            const_blend_grid.append({"w_knn_cal": float(w), "rae": r_b})
        best_const = min(const_blend_grid, key=lambda r: r["rae"])

        print(f"\n   helpable subset (n={n_help}):")
        print(f"      chemprop_aux       RAE = {rae_h_anchor:.4f}")
        print(f"      nb2103 K=28        RAE = {rae_h_nb2103:.4f}")
        print(f"      kNN raw            RAE = {rae_h_knn_raw:.4f}")
        print(f"      kNN calibrated     RAE = {rae_h_knn_cal:.4f}")
        print(f"      per-row blend      RAE = {rae_h_blend:.4f}")
        print(f"      best const-blend   RAE = {best_const['rae']:.4f} "
              f"@ w_knn_cal={best_const['w_knn_cal']:.1f}")

        delta_knn_vs_anchor = rae_h_anchor - rae_h_knn_cal
        delta_blend_vs_anchor = rae_h_anchor - rae_h_blend
        delta_best_const = rae_h_anchor - best_const["rae"]
        kb_real_signal = (delta_knn_vs_anchor >= DECISION_RAE_DELTA) or \
                         (delta_blend_vs_anchor >= DECISION_RAE_DELTA) or \
                         (delta_best_const >= DECISION_RAE_DELTA)
        print(f"\n   DELTAS (helpable subset, RAE drop vs chemprop_aux):")
        print(f"      kNN calibrated     = {delta_knn_vs_anchor:+.4f}")
        print(f"      per-row blend      = {delta_blend_vs_anchor:+.4f}")
        print(f"      best const-blend   = {delta_best_const:+.4f}")
    else:
        rae_h_anchor = rae_h_nb2103 = rae_h_knn_raw = rae_h_knn_cal = None
        rae_h_blend = None
        delta_knn_vs_anchor = delta_blend_vs_anchor = delta_best_const = None
        kb_real_signal = False
        best_const = {"w_knn_cal": None, "rae": None}
        const_blend_grid = []

    # ---- Verdict ----
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if kb_real_signal:
        verdict = (
            f"KB_REAL_F2_SIGNAL  (helpable n={n_help}; best RAE drop "
            f"{max(d for d in [delta_knn_vs_anchor, delta_blend_vs_anchor, delta_best_const] if d is not None):+.4f} "
            f">= {DECISION_RAE_DELTA}) -> recommend cycle 131 integration"
        )
    else:
        if n_help == 0:
            verdict = (
                f"KB_NOT_USEFUL_NO_HELPABLE_ROWS  "
                f"(F2 n={n_f2}; 0 rows had delta>{DELTA_HELPABLE_MIN}) -> "
                f"KB is not closer than train on novel scaffolds"
            )
        else:
            best_drop = max(
                d for d in [delta_knn_vs_anchor, delta_blend_vs_anchor, delta_best_const]
                if d is not None
            )
            verdict = (
                f"KB_DOES_NOT_BEAT_ANCHOR  (helpable n={n_help}; best RAE drop "
                f"{best_drop:+.4f} < {DECISION_RAE_DELTA}) -> do NOT integrate; "
                f"KB neighbors are close in Tanimoto but assay bias and "
                f"residual noise dominate"
            )
    print(f"   {verdict}")

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": round(time.time() - t0, 2),
        "kb_path": str(kb_p),
        "kb_raw_rows": 11185,
        "kb_unique_compounds_after_dedup": int(len(kb_dedup)),
        "kb_unique_compounds_used_after_train_overlap_drop": int(len(kb_use)),
        "kb_train_inchikey_overlap": int(n_overlap),
        "kb_src_breakdown_used": kb_use["src"].value_counts().to_dict(),
        "kb_src_weights": {
            "PXR_direct": SRC_WEIGHT_PXR_DIRECT,
            "NR_related": SRC_WEIGHT_NR_RELATED,
            "n_pxr_direct": n_pxr_direct,
            "n_nr_related": n_nr_related,
        },
        "shift_factor_pec50_train_minus_kb_median": shift_factor,
        "n_unblind": int(n_unb),
        "n_f2_wide_scaf_freq_zero": int(n_f2),
        "delta_helpable_threshold": DELTA_HELPABLE_MIN,
        "n_f2_helpable": int(n_help),
        "knn_k": KNN_K,
        "sim_train_summary_on_f2": {
            "mean": float(sim_train.mean()),
            "median": float(np.median(sim_train)),
            "p25": float(np.percentile(sim_train, 25)),
            "p75": float(np.percentile(sim_train, 75)),
        },
        "sim_kb_summary_on_f2": {
            "mean": float(sim_kb.mean()),
            "median": float(np.median(sim_kb)),
            "p25": float(np.percentile(sim_kb, 25)),
            "p75": float(np.percentile(sim_kb, 75)),
        },
        "delta_summary_on_f2": {
            "mean": float(delta.mean()),
            "median": float(np.median(delta)),
            "p25": float(np.percentile(delta, 25)),
            "p75": float(np.percentile(delta, 75)),
        },
        "rae_full_253": {
            "chemprop_aux": rae_anchor_full,
            "nb2103_K28_mean_bag_oof": rae_nb2103_full,
        },
        "rae_full_f2": {
            "chemprop_aux": rae_f2_anchor,
            "nb2103_K28_mean_bag_oof": rae_f2_nb2103,
            "knn_raw": rae_f2_knn_raw,
            "knn_calibrated": rae_f2_knn_cal,
        },
        "rae_helpable_subset": {
            "chemprop_aux": rae_h_anchor,
            "nb2103_K28_mean_bag_oof": rae_h_nb2103,
            "knn_raw": rae_h_knn_raw,
            "knn_calibrated": rae_h_knn_cal,
            "per_row_blend_sim_kb_capped_0_8": rae_h_blend,
            "best_const_blend": best_const,
            "const_blend_grid": const_blend_grid,
        },
        "deltas_helpable_vs_chemprop_aux": {
            "knn_calibrated": delta_knn_vs_anchor,
            "per_row_blend": delta_blend_vs_anchor,
            "best_const_blend": delta_best_const,
        },
        "decision_rae_delta": DECISION_RAE_DELTA,
        "kb_real_signal": bool(kb_real_signal),
        "verdict": verdict,
        "cycle_131_recommendation": (
            "INTEGRATE_KB_KNN_AS_F2_ROUTER" if kb_real_signal else
            "DROP_KB_KNN_ROUTE; revisit only with PXR-direct subset or "
            "an assay-bias-learned regression head"
        ),
    }

    out_p = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_p, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n[save] {out_p}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    print(f"  n_f2_wide_scaf_freq_zero: {res['n_f2_wide_scaf_freq_zero']}")
    print(f"  n_f2_helpable           : {res['n_f2_helpable']}")
    print(f"  shift_factor_pec50      : {res['shift_factor_pec50_train_minus_kb_median']:+.3f}")
    if res['n_f2_helpable'] > 0:
        h = res['rae_helpable_subset']
        d = res['deltas_helpable_vs_chemprop_aux']
        print(f"  helpable RAE (anchor)   : {h['chemprop_aux']:.4f}")
        print(f"  helpable RAE (kNN cal)  : {h['knn_calibrated']:.4f}")
        print(f"  helpable RAE (blend)    : {h['per_row_blend_sim_kb_capped_0_8']:.4f}")
        print(f"  helpable RAE (best const): {h['best_const_blend']['rae']:.4f}")
        print(f"  best const w_knn        : {h['best_const_blend']['w_knn_cal']}")
        print(f"  delta kNN vs anchor     : {d['knn_calibrated']:+.4f}")
        print(f"  delta blend vs anchor   : {d['per_row_blend']:+.4f}")
        print(f"  delta best const        : {d['best_const_blend']:+.4f}")
    print(f"  kb_real_signal          : {res['kb_real_signal']}")
    print(f"  verdict                 : {res['verdict']}")
    print(f"  cycle_131_recommend     : {res['cycle_131_recommendation']}")
