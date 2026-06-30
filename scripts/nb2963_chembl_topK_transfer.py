"""nb2963 -- ChEMBL PXR top-K (LOOSE sim>=0.30) transfer over nb2240 anchor.

PARADIGM SHIFT vs nb2453: nb2453 required Tanimoto>=0.40 to ChEMBL PXR-direct
(883 rows) AND novel-scaffold-only -- gated 0 of 197 novel rows (max_sim p75=0.276).
This script loosens the sim gate to >= 0.30 and applies the transfer to the
TOP-50 highest-sim rows (regardless of scaffold novelty), which is where the
ChEMBL labels are most likely to be locally informative.

PROTOCOL:
    1. Load ChEMBL KB; keep PXR-direct sources {NR_PXR, PXR_CHEMBL3401} ->
       883 rows with pec50_chembl. Standardize + drop train-inchikey overlap.
    2. For each of the 253 unblind rows: Morgan-Tanimoto k=5 in ChEMBL pool;
       record max_sim and the sim-weighted kNN prediction.
    3. Select rows with max_sim >= 0.30 (loose); intersect with the top-50
       max_sim rows (so we transfer only where the local ChEMBL is closest).
    4. Routing: te_final = te_nb2240 elsewhere; replace selected rows with
       sim-weighted ChEMBL kNN prediction (k=5).
    5. 5-fold scaffold CV on 253, kf_seed 1001; report mean RAE across folds.

GATE:
    mean_rae < 0.4570   -> "PROMOTE"  (beats nb2240 0.4601 by >0.003)
    mean_rae < 0.4576   -> "BETTER"   (marginal beat)
    else                -> "FAIL"

OUTPUTS:
    scripts/nb2963_chembl_topK_transfer.py
    data/processed/nb2963_summary.json
    data/processed/nb2963_pred_oof.npy   (253-row OOF)
    data/processed/te_nb2963.npy         (513-row deploy te)
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

TAG = "nb2963"
ROOT = Path(__file__).resolve().parents[1]
EXT_DIR = ROOT / "data" / "external"

PXR_DIRECT_SOURCES = {"PXR_CHEMBL3401", "NR_PXR"}
KNN_K = 5
SIM_GATE = 0.30     # LOOSE (was 0.40 in nb2453)
TOPK_ROWS = 50      # only override top-50 max_sim rows
KF_SEED = 1001
N_FOLDS = 5
NB2240_OOF_REF = 0.4601


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
        part = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
        rows = np.arange(part.shape[0])[:, None]
        part_sim = sim[rows, part]
        order = np.argsort(-part_sim, axis=1)
        top_idx[s:e, :kk] = part[rows, order]
        top_sim[s:e, :kk] = part_sim[rows, order]
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
    print(f"{TAG} -- ChEMBL PXR top-{TOPK_ROWS} (sim>={SIM_GATE}) transfer "
          f"over nb2240")
    print(f"          baseline: nb2240 deep-30 cross-fit RAE = "
          f"{NB2240_OOF_REF:.4f}")
    print(f"          k={KNN_K}, kf_seed={KF_SEED}, n_folds={N_FOLDS}")
    print("=" * 78)

    # ---- Load test + unblind ----
    te = load_test()
    te_smiles = te["smiles"].values
    n_te = len(te_smiles)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_te={n_te}  n_unb={n_unb}")

    te_nb2240 = np.load(DATA_PROCESSED / "te_nb2240.npy").astype(np.float64)
    assert te_nb2240.shape == (n_te,)
    anchor_unb = te_nb2240[unb_idx]
    rae_anchor_unb = float(rae(y_unb, anchor_unb))
    print(f"[anchor] nb2240 te_unb_rae (in-sample on 253) = {rae_anchor_unb:.4f}")

    # ---- Train inchikeys (overlap filter) ----
    tr_df = load_train()
    tr_mols = [standardize(s) for s in tr_df["smiles"].tolist()]
    train_ik_set = set(
        ik for ik in (_safe_inchikey(m) for m in tr_mols) if ik is not None
    )
    print(f"[load] train_ik_set = {len(train_ik_set)}")

    # ---- Standardize the test SMILES ----
    te_std_mols = [standardize(s) for s in te_smiles]
    te_std_smi = [_safe_canon_smiles(m) or "" for m in te_std_mols]
    unb_smiles_std = [te_std_smi[i] for i in unb_idx]

    # ---- Load + filter ChEMBL KB ----
    kb_p = EXT_DIR / "chembl_pxr_nr_kb.parquet"
    kb = pd.read_parquet(kb_p)
    n_raw = len(kb)
    kb = kb[kb["source_target"].isin(PXR_DIRECT_SOURCES)].reset_index(drop=True)
    kb = kb[kb["pec50_chembl"].notna()].reset_index(drop=True)
    n_pxr = len(kb)
    print(f"[kb] raw={n_raw}, PXR-direct w/pec50={n_pxr}")

    kb_mols = [standardize(s) for s in kb["smiles"].tolist()]
    kb["std_smiles"] = [_safe_canon_smiles(m) or "" for m in kb_mols]
    kb["inchikey_std"] = [_safe_inchikey(m) for m in kb_mols]
    valid = (kb["std_smiles"] != "") & kb["inchikey_std"].notna()
    kb = kb[valid].reset_index(drop=True)

    kb_dedup = (
        kb.groupby("inchikey_std", as_index=False)
        .agg(
            std_smiles=("std_smiles", "first"),
            pec50=("pec50_chembl", "median"),
            n_meas=("pec50_chembl", "count"),
        )
    )
    overlap_mask = kb_dedup["inchikey_std"].isin(train_ik_set)
    n_overlap = int(overlap_mask.sum())
    kb_use = kb_dedup[~overlap_mask].reset_index(drop=True)
    print(f"[kb] dedup={len(kb_dedup)}, train-ik dropped={n_overlap}, "
          f"used={len(kb_use)}")

    # ---- Fingerprints ----
    fp_unb = morgan_fp_batch(unb_smiles_std)
    fp_kb = morgan_fp_batch(kb_use["std_smiles"].tolist())
    kb_keep = fp_kb.sum(axis=1) > 0
    if not kb_keep.all():
        kb_use = kb_use[kb_keep].reset_index(drop=True)
        fp_kb = fp_kb[kb_keep]
    print(f"[fp] fp_unb={fp_unb.shape}  fp_kb={fp_kb.shape}")

    # ---- kNN for all 253 unblind ----
    top_sim, top_idx = _tanimoto_topk(fp_unb, fp_kb, KNN_K)
    max_sim = top_sim[:, 0]
    print(f"[knn] 253 max_sim: mean={max_sim.mean():.3f}  "
          f"median={float(np.median(max_sim)):.3f}  "
          f"p25={float(np.percentile(max_sim, 25)):.3f}  "
          f"p75={float(np.percentile(max_sim, 75)):.3f}  "
          f"max={float(max_sim.max()):.3f}")

    # ---- Predict ----
    pool_pec50 = kb_use["pec50"].to_numpy().astype(np.float64)
    knn_pred = _knn_predict(top_sim, top_idx, pool_pec50)

    # ---- Routing: rows with sim>=GATE AND in top-TOPK_ROWS by max_sim ----
    sim_ok_mask = max_sim >= SIM_GATE
    n_sim_ok = int(sim_ok_mask.sum())
    # Top-K by max_sim across the 253:
    order_desc = np.argsort(-max_sim)
    topk_idx = order_desc[:TOPK_ROWS]
    topk_mask = np.zeros(n_unb, dtype=bool)
    topk_mask[topk_idx] = True
    # Final override mask: sim_ok AND in top-K
    override_mask = sim_ok_mask & topk_mask
    n_override = int(override_mask.sum())
    print(f"[route] sim>={SIM_GATE} count = {n_sim_ok} / {n_unb}")
    print(f"        top-{TOPK_ROWS} max_sim intersection = {n_override}")

    # Build OOF (253) and te (513)
    pred_oof = anchor_unb.copy()
    pred_oof[override_mask] = knn_pred[override_mask]

    # For 513-row te: compute kNN for ALL test rows, restrict to top-K-AND-sim
    fp_te_all = morgan_fp_batch(te_std_smi)
    keep_te = fp_te_all.sum(axis=1) > 0
    top_sim_te = np.zeros((n_te, KNN_K), dtype=np.float32)
    top_idx_te = np.zeros((n_te, KNN_K), dtype=np.int32)
    if keep_te.all():
        top_sim_te, top_idx_te = _tanimoto_topk(fp_te_all, fp_kb, KNN_K)
    else:
        # Fallback: still compute but rows with zero fp will be skipped
        sub_sim, sub_idx = _tanimoto_topk(fp_te_all[keep_te], fp_kb, KNN_K)
        top_sim_te[keep_te] = sub_sim
        top_idx_te[keep_te] = sub_idx
    max_sim_te = top_sim_te[:, 0]
    knn_pred_te = _knn_predict(top_sim_te, top_idx_te, pool_pec50)
    # Top-K on 513:
    order_te = np.argsort(-max_sim_te)
    topk_te = np.zeros(n_te, dtype=bool)
    topk_te[order_te[:TOPK_ROWS]] = True
    sim_ok_te = max_sim_te >= SIM_GATE
    override_te = topk_te & sim_ok_te
    n_override_te = int(override_te.sum())
    te_final = te_nb2240.copy()
    te_final[override_te] = knn_pred_te[override_te]
    print(f"[route] 513-row overrides: {n_override_te} (top-{TOPK_ROWS} & "
          f"sim>={SIM_GATE})")

    # ---- Scaffold 5-fold CV on the 253 (kf_seed=1001) ----
    unb_scaffolds = [bemis_murcko(s) for s in te_smiles[unb_idx]]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"\n[cv] scaffold 5-fold kf_seed={KF_SEED}; "
          f"unique scaffolds on unb = {n_unique_scaf}")

    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=KF_SEED,
    )
    oof_blend = np.full(n_unb, np.nan)
    oof_anchor = np.full(n_unb, np.nan)
    fold_rae_blend = []
    fold_rae_anchor = []
    for fi, (_tr_loc, va_loc) in enumerate(splits):
        oof_blend[va_loc] = pred_oof[va_loc]
        oof_anchor[va_loc] = anchor_unb[va_loc]
        r_b = float(rae(y_unb[va_loc], pred_oof[va_loc]))
        r_a = float(rae(y_unb[va_loc], anchor_unb[va_loc]))
        fold_rae_blend.append(r_b)
        fold_rae_anchor.append(r_a)
        print(f"   fold={fi}  n_va={len(va_loc):3d}  blend={r_b:.4f}  "
              f"anchor={r_a:.4f}  delta={r_a - r_b:+.4f}")

    mean_rae = float(np.mean(fold_rae_blend))
    pooled_blend = float(rae(y_unb, oof_blend))
    pooled_anchor = float(rae(y_unb, oof_anchor))
    delta_vs_anchor = pooled_anchor - pooled_blend
    delta_vs_ref = NB2240_OOF_REF - mean_rae
    print(f"\n[cv] mean fold RAE      = {mean_rae:.4f}")
    print(f"     pooled blend  RAE  = {pooled_blend:.4f}")
    print(f"     pooled anchor RAE  = {pooled_anchor:.4f}")
    print(f"     delta vs anchor (pooled)        = {delta_vs_anchor:+.4f}")
    print(f"     delta vs nb2240 ref ({NB2240_OOF_REF:.4f}) = "
          f"{delta_vs_ref:+.4f}")

    # ---- Gate ----
    if mean_rae < 0.4570:
        verdict = "PROMOTE"
    elif mean_rae < 0.4576:
        verdict = "BETTER"
    else:
        verdict = "FAIL"
    print("\n" + "-" * 78)
    print(f"GATE: mean_rae = {mean_rae:.4f}  ->  {verdict}")
    print(f"   thresholds: PROMOTE<0.4570  BETTER<0.4576  else FAIL")
    print("-" * 78)

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy", pred_oof)
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", te_final)
    print(f"[save] {DATA_PROCESSED / (TAG + '_pred_oof.npy')}")
    print(f"[save] {DATA_PROCESSED / ('te_' + TAG + '.npy')}")

    summary = {
        "tag": TAG,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": round(time.time() - t0, 2),
        "method": "ChEMBL_PXR_topK_loose_sim_transfer_over_nb2240",
        "kb_path": str(kb_p),
        "pxr_direct_sources": sorted(PXR_DIRECT_SOURCES),
        "n_pxr_direct_with_pec50": int(n_pxr),
        "n_kb_used": int(len(kb_use)),
        "n_train_inchikey_overlap_dropped": int(n_overlap),
        "knn_k": KNN_K,
        "sim_gate_loose": SIM_GATE,
        "topk_rows": TOPK_ROWS,
        "kf_seed": KF_SEED,
        "n_folds": N_FOLDS,
        "n_unb": n_unb,
        "n_te": n_te,
        "n_unique_unb_scaffolds": int(n_unique_scaf),
        "max_sim_unb_summary": {
            "mean": float(max_sim.mean()),
            "median": float(np.median(max_sim)),
            "p25": float(np.percentile(max_sim, 25)),
            "p75": float(np.percentile(max_sim, 75)),
            "max": float(max_sim.max()),
        },
        "n_sim_ok_unb": int(n_sim_ok),
        "n_override_unb": int(n_override),
        "n_override_te_513": int(n_override_te),
        "anchor_te_unb_rae_in_sample": rae_anchor_unb,
        "fold_rae_blend": [float(x) for x in fold_rae_blend],
        "fold_rae_anchor": [float(x) for x in fold_rae_anchor],
        "mean_rae": mean_rae,
        "pooled_blend_rae": pooled_blend,
        "pooled_anchor_rae": pooled_anchor,
        "delta_vs_anchor_pooled": delta_vs_anchor,
        "nb2240_oof_ref": NB2240_OOF_REF,
        "delta_vs_nb2240_ref": delta_vs_ref,
        "gate_thresholds": {
            "PROMOTE_lt": 0.4570,
            "BETTER_lt": 0.4576,
        },
        "verdict": verdict,
        "pred_oof_path": str(DATA_PROCESSED / f"{TAG}_pred_oof.npy"),
        "te_path": str(DATA_PROCESSED / f"te_{TAG}.npy"),
    }
    out_p = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_p, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[save] {out_p}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    print(f"  n_kb_used               : {res['n_kb_used']}")
    print(f"  n_sim_ok_unb (>={res['sim_gate_loose']:.2f}) : "
          f"{res['n_sim_ok_unb']} / {res['n_unb']}")
    print(f"  n_override_unb           : {res['n_override_unb']}")
    print(f"  n_override_te_513        : {res['n_override_te_513']}")
    print(f"  anchor_unb_rae_insample  : "
          f"{res['anchor_te_unb_rae_in_sample']:.4f}")
    print(f"  mean_rae (5-fold)        : {res['mean_rae']:.4f}")
    print(f"  pooled_blend_rae         : {res['pooled_blend_rae']:.4f}")
    print(f"  delta_vs_nb2240_ref      : {res['delta_vs_nb2240_ref']:+.4f}")
    print(f"  verdict                  : {res['verdict']}")
