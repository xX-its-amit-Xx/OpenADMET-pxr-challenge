"""nb2453 -- ChEMBL PXR-Direct kNN ROUTED ON NOVEL SCAFFOLDS ONLY (F2 attack #6).

F2 failure mode (phase-1 post-mortem): novel-scaffold inactives over-predicted
+1.23 log-units; dominant residual tail.  Prior PXR-direct kNN (nb2230) blended
EVERYWHERE -> diluted by easy seen-scaffold rows that already work.  This
notebook routes the ChEMBL signal ONLY where the anchor is weakest:
    * scaf_train_freq == 0  (novel-scaffold test rows; ~416/513)
    * AND ChEMBL has at least one near neighbor (Tanimoto >= 0.40)

PROTOCOL:
    1. Load data/external/chembl_pxr_nr_kb.parquet (11185 rows).
       Filter source_target in {PXR_CHEMBL3401, NR_PXR} -> 1040 rows
       (memory ref: ~883, but file has both fields under different IDs).
       Filter to rows with pec50_chembl present.
    2. Identify the 416/513 test rows with scaf_train_freq == 0 using
       Bemis-Murcko on the 4139 training set.
    3. For each novel-scaffold test row: Morgan-Tanimoto k=5 kNN in the
       ChEMBL pool (PXR-direct, train-inchikey overlap dropped).
    4. Routing rule for FINAL 513-row prediction `te_final`:
          a. seen-scaffold rows                    -> te_nb2240   (anchor)
          b. novel-scaf rows with max_sim < 0.40   -> te_nb2240
          c. novel-scaf rows with max_sim >= 0.40  -> sim-weighted kNN(k=5)
                                                       ChEMBL labels
    5. Scaffold 5-fold CV RAE on the 253 unblind (5 seeds); compare vs
       nb2240 0.4601 (cross-fit baseline).
    6. Gate: improvement >= 0.003 RAE on cross-fit.

OUTPUTS:
    scripts/nb2453_chembl_f2_only.py
    data/processed/nb2453_summary.json
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

from pxr.chem import bemis_murcko, morgan_fp_batch, standardize
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2453"
ROOT = Path(__file__).resolve().parents[1]
EXT_DIR = ROOT / "data" / "external"

PXR_DIRECT_SOURCES = {"PXR_CHEMBL3401", "NR_PXR"}
KNN_K = 5
SIM_GATE = 0.40
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
N_FOLDS = 5
GATE_DELTA = 0.003
NB2240_OOF_REF = 0.4601                # honest cross-fit RAE on 253


def _safe_canon_smiles(mol):
    try:
        return Chem.MolToSmiles(mol) if mol is not None else None
    except Exception:
        return None


def _safe_inchikey(mol):
    try:
        return Chem.MolToInchiKey(mol) if mol is not None else None
    except Exception:
        return None


def _tanimoto_topk(fp_q: np.ndarray, fp_pool: np.ndarray, k: int):
    """Per-query top-k Tanimoto sim and pool index."""
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


def _knn_predict(top_sim: np.ndarray, top_idx: np.ndarray,
                 pool_pec50: np.ndarray) -> np.ndarray:
    n, _k = top_sim.shape
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = top_sim[i].astype(np.float64)
        if s.sum() <= 0:
            out[i] = float(pool_pec50.mean())
        else:
            out[i] = float(
                (s * pool_pec50[top_idx[i]].astype(np.float64)).sum() / s.sum()
            )
    return out


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- ChEMBL PXR-Direct kNN ROUTED ON NOVEL SCAFFOLDS ONLY")
    print(f"          sources kept: {sorted(PXR_DIRECT_SOURCES)}")
    print(f"          k={KNN_K}, sim_gate>={SIM_GATE}")
    print(f"          baseline: nb2240 cross-fit RAE = {NB2240_OOF_REF:.4f}")
    print(f"          gate delta = {GATE_DELTA:.3f}")
    print("=" * 78)

    # ---- Load test + unblind ----
    te = load_test()
    te_smiles = te["smiles"].values
    n_te = len(te_smiles)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_te={n_te}  n_unb={n_unb}")

    # nb2240 anchor (te on 513)
    te_nb2240 = np.load(DATA_PROCESSED / "te_nb2240.npy").astype(np.float64)
    assert te_nb2240.shape == (n_te,)
    anchor_unb = te_nb2240[unb_idx]
    rae_anchor_unb = float(rae(y_unb, anchor_unb))
    print(f"[anchor] nb2240 te_unb_rae (in-sample on 253) = {rae_anchor_unb:.4f}")
    print(f"         (cross-fit ref               = {NB2240_OOF_REF:.4f})")

    # ---- Train -> scaffolds + inchikey set ----
    tr_df = load_train()
    print(f"[load] train rows = {len(tr_df)}")
    tr_mols = [standardize(s) for s in tr_df["smiles"].tolist()]
    tr_inchikeys = [_safe_inchikey(m) for m in tr_mols]
    train_ik_set = set(ik for ik in tr_inchikeys if ik is not None)
    tr_scaffolds = [bemis_murcko(s) for s in tr_df["smiles"].tolist()]
    scaf_tr_counts: dict = {}
    for s in tr_scaffolds:
        if s:
            scaf_tr_counts[s] = scaf_tr_counts.get(s, 0) + 1
    print(f"   train unique inchikeys = {len(train_ik_set)}")
    print(f"   train unique scaffolds = {len(scaf_tr_counts)}")

    # ---- Per-test scaf_train_freq ----
    te_std_mols = [standardize(s) for s in te_smiles]
    te_std_smi = [_safe_canon_smiles(m) or "" for m in te_std_mols]
    te_scaffolds = [bemis_murcko(s) for s in te_smiles]
    scaf_freq_te = np.array(
        [scaf_tr_counts.get(s, 0) for s in te_scaffolds], dtype=np.int64
    )
    novel_te_mask = scaf_freq_te == 0
    n_novel_te = int(novel_te_mask.sum())
    print(f"[scaf] novel-scaffold rows on 513 (scaf_train_freq==0) = "
          f"{n_novel_te} / 513")
    scaf_freq_unb = scaf_freq_te[unb_idx]
    novel_unb_mask = scaf_freq_unb == 0
    n_novel_unb = int(novel_unb_mask.sum())
    print(f"       novel-scaffold rows on 253 = {n_novel_unb} / 253")

    # ---- Load + filter ChEMBL KB ----
    kb_p = EXT_DIR / "chembl_pxr_nr_kb.parquet"
    kb = pd.read_parquet(kb_p)
    n_raw = len(kb)
    src_breakdown_raw = kb["source_target"].value_counts().to_dict()
    print(f"\n[kb] raw rows = {n_raw}")
    print(f"     src breakdown (top 8): "
          f"{dict(list(sorted(src_breakdown_raw.items(), key=lambda x:-x[1])[:8]))}")

    kb = kb[kb["source_target"].isin(PXR_DIRECT_SOURCES)].reset_index(drop=True)
    n_pxr = len(kb)
    print(f"[kb] after PXR-direct filter = {n_pxr} rows")

    kb = kb[kb["pec50_chembl"].notna()].reset_index(drop=True)
    n_with_pec50 = len(kb)
    print(f"[kb] after pec50_chembl notna = {n_with_pec50} rows")

    # Standardize / inchikey
    print("[kb] standardizing PXR-direct SMILES...")
    kb_mols = [standardize(s) for s in kb["smiles"].tolist()]
    kb["std_smiles"] = [_safe_canon_smiles(m) or "" for m in kb_mols]
    kb["inchikey_std"] = [_safe_inchikey(m) for m in kb_mols]
    valid = (kb["std_smiles"] != "") & kb["inchikey_std"].notna()
    kb = kb[valid].reset_index(drop=True)
    print(f"   after valid std = {len(kb)}")

    # Dedup by inchikey (median pEC50)
    kb_dedup = (
        kb.groupby("inchikey_std", as_index=False)
        .agg(
            std_smiles=("std_smiles", "first"),
            pec50=("pec50_chembl", "median"),
            src=("source_target", "first"),
            n_meas=("pec50_chembl", "count"),
        )
    )
    print(f"   KB unique compounds (inchikey-dedup): {len(kb_dedup)}")

    # Drop train-inchikey overlap
    overlap_mask = kb_dedup["inchikey_std"].isin(train_ik_set)
    n_overlap = int(overlap_mask.sum())
    kb_use = kb_dedup[~overlap_mask].reset_index(drop=True)
    print(f"   train-inchikey overlap dropped = {n_overlap}; "
          f"final KB used = {len(kb_use)}")

    # ---- Fingerprints (only for novel-scaffold test rows + KB) ----
    print(f"\n[fp] Morgan ECFP4 for {n_novel_te} novel-scaf test + "
          f"{len(kb_use)} ChEMBL...")
    te_std_smi_novel = [te_std_smi[i] for i in range(n_te) if novel_te_mask[i]]
    fp_te_novel = morgan_fp_batch(te_std_smi_novel)
    fp_kb = morgan_fp_batch(kb_use["std_smiles"].tolist())
    kb_keep = fp_kb.sum(axis=1) > 0
    if not kb_keep.all():
        kb_use = kb_use[kb_keep].reset_index(drop=True)
        fp_kb = fp_kb[kb_keep]
    print(f"   shapes: fp_te_novel={fp_te_novel.shape}  fp_kb={fp_kb.shape}")

    # ---- Top-k Tanimoto for novel-scaf test only ----
    print(f"\n[knn] computing top-{KNN_K} Tanimoto neighbors for {n_novel_te} "
          f"novel-scaf test rows...")
    top_sim_novel, top_idx_novel = _tanimoto_topk(fp_te_novel, fp_kb, KNN_K)
    max_sim_novel = top_sim_novel[:, 0]

    # Place per-513 arrays for the novel rows; defaults for seen rows
    max_sim_te = np.zeros(n_te, dtype=np.float32)
    novel_te_positions = np.where(novel_te_mask)[0]
    max_sim_te[novel_te_positions] = max_sim_novel

    n_novel_hit_te = int((max_sim_novel >= SIM_GATE).sum())
    print(f"   novel rows on 513 with max_sim >= {SIM_GATE}: "
          f"{n_novel_hit_te} / {n_novel_te}")
    print(f"   novel-scaf 513 max_sim: mean={max_sim_novel.mean():.3f}  "
          f"median={np.median(max_sim_novel):.3f}  "
          f"p25={np.percentile(max_sim_novel, 25):.3f}  "
          f"p75={np.percentile(max_sim_novel, 75):.3f}")

    # ---- Predict ----
    pool_pec50 = kb_use["pec50"].to_numpy().astype(np.float64)
    knn_pred_novel = _knn_predict(top_sim_novel, top_idx_novel, pool_pec50)

    # Route: build te_final on 513
    te_final = te_nb2240.copy()
    for j, te_pos in enumerate(novel_te_positions):
        if max_sim_novel[j] >= SIM_GATE:
            te_final[te_pos] = knn_pred_novel[j]

    n_replaced_te = int(
        (te_final != te_nb2240).sum()
    )
    print(f"\n[route] te_final overrides on 513 = {n_replaced_te}  "
          f"(novel + sim>={SIM_GATE})")

    # ---- Slice to 253 ----
    blended_unb = te_final[unb_idx]
    # Diagnostic on 253
    max_sim_unb = max_sim_te[unb_idx]
    novel_hit_unb_mask = novel_unb_mask & (max_sim_unb >= SIM_GATE)
    n_novel_hit_unb = int(novel_hit_unb_mask.sum())
    n_replaced_unb = int((blended_unb != anchor_unb).sum())
    print(f"   253 novel-scaf rows = {n_novel_unb}")
    print(f"   253 novel-scaf rows with sim>={SIM_GATE} = {n_novel_hit_unb}")
    print(f"   253 anchor overrides applied = {n_replaced_unb}")

    rae_blend_insample = float(rae(y_unb, blended_unb))
    print(f"\n[insample-on-253]")
    print(f"   anchor (nb2240 te_unb) RAE = {rae_anchor_unb:.4f}")
    print(f"   blend RAE                  = {rae_blend_insample:.4f}  "
          f"delta = {rae_anchor_unb - rae_blend_insample:+.4f}")

    # Per-row F2-only diagnostic: anchor vs blend on novel-scaf rows that hit
    if n_novel_hit_unb >= 3:
        y_sub = y_unb[novel_hit_unb_mask]
        a_sub = anchor_unb[novel_hit_unb_mask]
        b_sub = blended_unb[novel_hit_unb_mask]
        rae_anchor_sub = float(rae(y_sub, a_sub))
        rae_blend_sub = float(rae(y_sub, b_sub))
        print(f"\n[F2-only-subset] n={n_novel_hit_unb}")
        print(f"   anchor RAE on hit-novel = {rae_anchor_sub:.4f}")
        print(f"   blend  RAE on hit-novel = {rae_blend_sub:.4f}  "
              f"delta = {rae_anchor_sub - rae_blend_sub:+.4f}")
    else:
        rae_anchor_sub = None
        rae_blend_sub = None
        print(f"\n[F2-only-subset] too few hits (n={n_novel_hit_unb}); "
              f"skipping sub-RAE")

    # ---- Scaffold 5-fold CV on the 253 ----
    unb_smiles = te_smiles[unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"\n[cv] scaffold 5-fold across {len(KF_SEEDS)} seeds; "
          f"unique scaffolds on unb = {n_unique_scaf}")
    per_seed_rae = []
    per_seed_rae_anchor = []
    for seed in KF_SEEDS:
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=seed,
        )
        oof_blend = np.full(n_unb, np.nan)
        oof_anchor = np.full(n_unb, np.nan)
        for _tr_loc, va_loc in splits:
            oof_blend[va_loc] = blended_unb[va_loc]
            oof_anchor[va_loc] = anchor_unb[va_loc]
        r_blend = float(rae(y_unb, oof_blend))
        r_anchor = float(rae(y_unb, oof_anchor))
        per_seed_rae.append(r_blend)
        per_seed_rae_anchor.append(r_anchor)
        print(f"   seed={seed}  blend_RAE={r_blend:.4f}  "
              f"anchor_RAE={r_anchor:.4f}  delta={r_anchor - r_blend:+.4f}")

    mean_blend = float(np.mean(per_seed_rae))
    std_blend = float(np.std(per_seed_rae))
    mean_anchor = float(np.mean(per_seed_rae_anchor))
    print(f"\n[cv] mean across {len(KF_SEEDS)} seeds:")
    print(f"   blend  = {mean_blend:.4f} (+/- {std_blend:.4f})")
    print(f"   anchor = {mean_anchor:.4f}")
    print(f"   delta vs anchor (recomputed) = {mean_anchor - mean_blend:+.4f}")
    print(f"   delta vs nb2240 ref ({NB2240_OOF_REF:.4f})       = "
          f"{NB2240_OOF_REF - mean_blend:+.4f}")

    # ---- Gate ----
    delta_vs_ref = NB2240_OOF_REF - mean_blend
    gate_pass = bool(delta_vs_ref >= GATE_DELTA)
    print("\n" + "-" * 78)
    print("GATE EVALUATION")
    print("-" * 78)
    print(f"   improvement vs nb2240 cross-fit = {delta_vs_ref:+.4f}  "
          f"(gate >= {GATE_DELTA:.3f})  -> "
          f"{'PASS' if gate_pass else 'FAIL'}")

    summary = {
        "tag": TAG,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": round(time.time() - t0, 2),
        "method": "ChEMBL_PXR_direct_kNN_novel_scaffold_only_F2_attack6",
        "kb_path": str(kb_p),
        "kb_raw_rows": int(n_raw),
        "kb_src_breakdown_raw_top": {
            k: int(v) for k, v in
            sorted(src_breakdown_raw.items(), key=lambda x: -x[1])[:10]
        },
        "pxr_direct_sources": sorted(PXR_DIRECT_SOURCES),
        "n_pxr_direct_raw": int(n_pxr),
        "n_pxr_direct_with_pec50": int(n_with_pec50),
        "n_kb_unique_dedup": int(len(kb_dedup)),
        "n_train_inchikey_overlap_dropped": int(n_overlap),
        "n_kb_used": int(len(kb_use)),
        "knn_k": KNN_K,
        "sim_gate": SIM_GATE,
        "n_novel_te_513": int(n_novel_te),
        "n_novel_te_hit": int(n_novel_hit_te),
        "n_replaced_te_513": int(n_replaced_te),
        "n_novel_unb_253": int(n_novel_unb),
        "n_novel_unb_hit": int(n_novel_hit_unb),
        "n_replaced_unb_253": int(n_replaced_unb),
        "max_sim_novel_te_summary": {
            "mean": float(max_sim_novel.mean()),
            "median": float(np.median(max_sim_novel)),
            "p25": float(np.percentile(max_sim_novel, 25)),
            "p75": float(np.percentile(max_sim_novel, 75)),
        },
        "anchor_te_unb_rae_in_sample": rae_anchor_unb,
        "blend_te_unb_rae_in_sample": rae_blend_insample,
        "delta_insample": rae_anchor_unb - rae_blend_insample,
        "F2_subset_anchor_rae": rae_anchor_sub,
        "F2_subset_blend_rae": rae_blend_sub,
        "F2_subset_n": int(n_novel_hit_unb),
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "n_unique_unb_scaffolds": int(n_unique_scaf),
        "per_seed_blend_rae": [float(x) for x in per_seed_rae],
        "per_seed_anchor_rae": [float(x) for x in per_seed_rae_anchor],
        "cv_mean_blend_rae": mean_blend,
        "cv_std_blend_rae": std_blend,
        "cv_mean_anchor_rae": mean_anchor,
        "delta_cv_vs_anchor_recomputed": mean_anchor - mean_blend,
        "nb2240_oof_ref": NB2240_OOF_REF,
        "delta_cv_vs_nb2240_ref": delta_vs_ref,
        "gate_delta": GATE_DELTA,
        "gate_pass": gate_pass,
        "verdict": (
            f"CHEMBL_F2_ONLY_BEATS_NB2240  delta={delta_vs_ref:+.4f} "
            f">= {GATE_DELTA:.3f}  -> consider integration"
            if gate_pass else
            f"CHEMBL_F2_ONLY_NOT_HELPFUL  delta={delta_vs_ref:+.4f} "
            f"< {GATE_DELTA:.3f}  -> do NOT promote (novel-scaf rows have too "
            f"low ChEMBL sim, or signal is anchor-correlated)"
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
    print(f"  n_kb_raw                     : {res['kb_raw_rows']}")
    print(f"  n_pxr_direct_with_pec50      : {res['n_pxr_direct_with_pec50']}")
    print(f"  n_kb_used                    : {res['n_kb_used']}")
    print(f"  n_novel_te_513 / hit         : {res['n_novel_te_513']} / "
          f"{res['n_novel_te_hit']}")
    print(f"  n_novel_unb_253 / hit        : {res['n_novel_unb_253']} / "
          f"{res['n_novel_unb_hit']}")
    print(f"  cv_mean_blend_rae            : {res['cv_mean_blend_rae']:.4f}")
    print(f"  cv_mean_anchor_rae           : {res['cv_mean_anchor_rae']:.4f}")
    print(f"  delta_cv_vs_anchor_recomp    : "
          f"{res['delta_cv_vs_anchor_recomputed']:+.4f}")
    print(f"  delta_cv_vs_nb2240_ref       : "
          f"{res['delta_cv_vs_nb2240_ref']:+.4f}")
    print(f"  gate_pass                    : {res['gate_pass']}")
    print(f"  verdict                      : {res['verdict']}")
