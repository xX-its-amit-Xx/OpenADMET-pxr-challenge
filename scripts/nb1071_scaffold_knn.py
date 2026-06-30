"""nb1071 -- Per-scaffold Tanimoto kNN (k=3) + blend with nb2103 K=28 anchor.

HYPOTHESIS:
    Scaffold-aware kNN (k=3 with Tanimoto + Murcko scaffold match) may capture
    chemistry context that LGBM-on-features misses.  For each test-513 row:
    take k=3 nearest TRAIN compounds restricted to the SAME Murcko scaffold,
    fall back to global Tanimoto top-3 if fewer than 3 scaffold-matches exist.
    Predict via similarity-weighted kNN.  Compare standalone vs chemprop_aux
    (0.6216) and nb2103 K=28 (0.4737).  Then blend with nb2103 K=28 anchor.

PROTOCOL:
    1. Load 4139 TRAIN + 253 unblind + 513 test SMILES + pEC50 labels.
    2. Compute Morgan FP (ECFP4 2048-bit) + Murcko scaffold for all.
    3. For each test row: find k=3 nearest TRAIN by Tanimoto AMONG same
       Murcko scaffold; fall back to global top-3 if scaffold pool < 3.
    4. Predict via weighted kNN (weights = sim values).
    5. Cross-fit on 253: per fold, use 4/5 of unblind + ALL 4139 TRAIN as
       pool; predict held-out 1/5 of unblind.
    6. Standalone RAE vs chemprop_aux (0.6216) and nb2103 K=28 (0.4737).
    7. Blend 0.7*nb2103 + 0.3*scaf_knn; sweep w in {0.0, 0.1, 0.2, 0.3, 0.5}.

Outputs:
    scripts/nb1071_scaffold_knn.py
    data/processed/nb1071_summary.json
    data/processed/nb1071_scaf_knn_oof_253.npy   (253,) float32
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
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_train, load_test, load_phase1_unblinded
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1071"
K_NN = 3
FOLDS = 5
SEED = 42
SIM_FLOOR = 1e-6

CHEMPROP_AUX_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
NB2103_K28_OOF_PATH = DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy"

CHEMPROP_AUX_REF = 0.6216
NB2103_K28_REF = 0.4737
BLEND_W_GRID = [0.0, 0.1, 0.2, 0.3, 0.5]
PRIMARY_BLEND_W_SCAF = 0.3  # 0.7*nb2103 + 0.3*scaf_knn


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


def _tanimoto_pairwise(fp_q: np.ndarray, fp_pool: np.ndarray) -> np.ndarray:
    """Compute pairwise Tanimoto sim (n_q, n_pool)."""
    a = fp_q.astype(np.float32)
    b = fp_pool.astype(np.float32)
    a_sum = a.sum(axis=1)
    b_sum = b.sum(axis=1)
    inter = a @ b.T
    denom = a_sum[:, None] + b_sum[None, :] - inter
    denom = np.maximum(denom, 1.0)
    return (inter / denom).astype(np.float32)


def _scaf_knn_predict(
    fp_query: np.ndarray,
    scaf_query: list[str],
    fp_pool: np.ndarray,
    scaf_pool: np.ndarray,
    labels_pool: np.ndarray,
    k: int,
    fallback: float,
):
    """For each query row, restrict pool to same scaffold; fallback to global
    top-k if fewer than k scaffold matches exist.  Returns
    (pred, n_scaf_match, used_fallback)."""
    n_q = fp_query.shape[0]
    pred = np.empty(n_q, dtype=np.float32)
    n_scaf_match = np.zeros(n_q, dtype=np.int32)
    used_fallback = np.zeros(n_q, dtype=np.uint8)

    # Precompute global Tanimoto sim matrix once -- pool is ~4400 rows worst
    # case and queries are 1..513, so memory is fine.
    sim_all = _tanimoto_pairwise(fp_query, fp_pool)  # (n_q, n_pool)

    for i in range(n_q):
        scaf_q = scaf_query[i]
        sim_row = sim_all[i]
        # Restrict to same Murcko scaffold (only if scaf_q is not empty/None)
        if scaf_q is not None and scaf_q != "":
            scaf_mask = (scaf_pool == scaf_q)
            n_match = int(scaf_mask.sum())
        else:
            scaf_mask = np.zeros_like(sim_row, dtype=bool)
            n_match = 0
        n_scaf_match[i] = n_match

        if n_match >= k:
            # Use scaffold-restricted top-k
            idx_pool = np.where(scaf_mask)[0]
            sim_sub = sim_row[idx_pool]
            order = np.argsort(-sim_sub)[:k]
            chosen = idx_pool[order]
            chosen_sim = sim_row[chosen]
        else:
            # Fallback: global top-k by Tanimoto
            used_fallback[i] = 1
            order = np.argpartition(-sim_row, kth=min(k, len(sim_row) - 1))[:k]
            order_sorted = order[np.argsort(-sim_row[order])]
            chosen = order_sorted
            chosen_sim = sim_row[chosen]

        w = np.clip(chosen_sim, 0.0, 1.0)
        w_sum = w.sum()
        if w_sum < SIM_FLOOR:
            pred[i] = fallback
        else:
            pred[i] = float(np.sum(w * labels_pool[chosen]) / w_sum)
    return pred, n_scaf_match, used_fallback


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- per-scaffold Tanimoto kNN k={K_NN} + blend with nb2103 K=28")
    print(f"          chemprop_aux ref = {CHEMPROP_AUX_REF}, nb2103 K=28 ref = {NB2103_K28_REF}")
    print("=" * 78)

    # ---- 1. Load data ----
    tr = load_train()
    te = load_test()
    p1 = load_phase1_unblinded()
    print(f"[load] train={len(tr)}  test={len(te)}  phase1_unblind={len(p1)}")

    tr = tr[tr["pec50"].notna() & tr["smiles"].notna()].reset_index(drop=True)
    print(f"[load] train after dropna pec50/smiles: {len(tr)}")

    test_smiles_orig = te["smiles"].astype(str).tolist()
    test_names = te["name"].astype(str).tolist()
    n_test = len(te)

    # 253 unblind labels + indices
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy").astype(int)
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_unb={n_unb}  unb_idx shape={unb_idx.shape}")

    # ---- 2. Standardize + Morgan FP + Murcko scaffold ----
    print("[chem] standardizing + computing FP + scaffold ...")
    tr_mols = [standardize(s) for s in tr["smiles"].tolist()]
    tr_smi_std = [_safe_can_smiles(m) if m is not None else None for m in tr_mols]
    tr_scaf = [bemis_murcko(m) if m is not None else None for m in tr_mols]
    tr_keep = [(s is not None and sc is not None) for s, sc in zip(tr_smi_std, tr_scaf)]
    tr_smi_std = [s for s, k in zip(tr_smi_std, tr_keep) if k]
    tr_scaf = [s for s, k in zip(tr_scaf, tr_keep) if k]
    tr_y = tr.loc[tr_keep, "pec50"].to_numpy(dtype=np.float32)
    print(f"   train kept after FP+scaffold: {len(tr_smi_std)} / {len(tr)}")

    te_mols = [standardize(s) for s in test_smiles_orig]
    te_smi_std = [_safe_can_smiles(m) if m is not None else "" for m in te_mols]
    te_scaf = [bemis_murcko(m) if m is not None else "" for m in te_mols]

    fp_tr = morgan_fp_batch(tr_smi_std)
    fp_te = morgan_fp_batch(te_smi_std)
    tr_scaf_arr = np.array(tr_scaf, dtype=object)
    print(f"   fp_tr={fp_tr.shape}  fp_te={fp_te.shape}")

    # Stats: how many test rows have ANY scaffold match in train?
    test_scaf_set = set(te_scaf)
    train_scaf_set = set(tr_scaf)
    overlap_scaf = test_scaf_set & train_scaf_set
    n_test_with_scaf_match = sum(1 for s in te_scaf if s != "" and s in train_scaf_set)
    print(f"   unique scaffolds: train={len(train_scaf_set)}  "
          f"test={len(test_scaf_set)}  overlap={len(overlap_scaf)}")
    print(f"   test rows with >=1 same-scaffold train match: "
          f"{n_test_with_scaf_match}/{n_test}")

    pool_median = float(np.median(tr_y))
    print(f"   train pEC50 median = {pool_median:.3f}")

    # ---- 3+4. Predict on ALL 513 using TRAIN ONLY (no unblind in pool) ----
    # This is the "honest" prediction for blend with PRE-unblind anchors.
    print("\n[predict] test-513 from TRAIN ONLY pool (n=%d)" % len(tr_smi_std))
    pred_test_train_only, n_match_train_only, used_fb_train_only = (
        _scaf_knn_predict(
            fp_te,
            te_scaf,
            fp_tr,
            tr_scaf_arr,
            tr_y,
            K_NN,
            pool_median,
        )
    )
    pct_fb_train_only = float(used_fb_train_only.mean() * 100)
    print(f"   used global fallback: {int(used_fb_train_only.sum())}/{n_test} "
          f"({pct_fb_train_only:.1f}%)")
    print(f"   pred range: [{pred_test_train_only.min():.3f}, "
          f"{pred_test_train_only.max():.3f}]  "
          f"mean={pred_test_train_only.mean():.3f}")

    # Slice to unblind 253 for in-sample (train-only-pool) RAE
    pred_unb_train_only = pred_test_train_only[unb_idx].astype(np.float64)
    rae_train_only = float(rae(y_unb, pred_unb_train_only))
    print(f"   RAE (train-only pool, on 253 unblind) = {rae_train_only:.4f}")

    # ---- 5. Cross-fit on 253: per fold, use 4/5 of unblind + all 4139 train
    #         as pool; predict held-out 1/5 of unblind ----
    print(f"\n[cf] 5-fold cross-fit (k={K_NN}, augmenting pool with 4/5 unblind)")
    # Build the 253 unblind set (FP + scaffold + label).
    unb_smiles_orig = [test_smiles_orig[i] for i in unb_idx]
    unb_mols = [standardize(s) for s in unb_smiles_orig]
    unb_smi_std = [_safe_can_smiles(m) if m is not None else "" for m in unb_mols]
    unb_scaf = [bemis_murcko(m) if m is not None else "" for m in unb_mols]
    fp_unb = morgan_fp_batch(unb_smi_std)
    unb_scaf_arr = np.array(unb_scaf, dtype=object)

    pred_cf = np.full(n_unb, np.nan, dtype=np.float64)
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for fold_i, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n_unb))):
        # Pool = ALL 4139 train + 4/5 of unblind
        fp_pool = np.concatenate([fp_tr, fp_unb[tr_loc]], axis=0)
        scaf_pool = np.concatenate(
            [tr_scaf_arr, unb_scaf_arr[tr_loc]], axis=0
        )
        y_pool = np.concatenate([tr_y, y_unb[tr_loc].astype(np.float32)], axis=0)
        pred_va, _, _ = _scaf_knn_predict(
            fp_unb[va_loc],
            [unb_scaf[i] for i in va_loc],
            fp_pool,
            scaf_pool,
            y_pool,
            K_NN,
            pool_median,
        )
        pred_cf[va_loc] = pred_va
        rae_fold = float(rae(y_unb[va_loc], pred_va))
        print(f"   fold {fold_i}: n_va={len(va_loc)}  pool={len(y_pool)}  "
              f"RAE_fold={rae_fold:.4f}")

    rae_cf = float(rae(y_unb, pred_cf))
    print(f"\n[cf] scaf_knn cross-fit RAE on 253 = {rae_cf:.4f}")
    print(f"     vs chemprop_aux ref            = {CHEMPROP_AUX_REF:.4f}  "
          f"(delta {rae_cf - CHEMPROP_AUX_REF:+.4f})")
    print(f"     vs nb2103 K=28 ref             = {NB2103_K28_REF:.4f}  "
          f"(delta {rae_cf - NB2103_K28_REF:+.4f})")

    # Save cross-fit OOF
    out_oof_p = DATA_PROCESSED / f"{TAG}_scaf_knn_oof_253.npy"
    np.save(out_oof_p, pred_cf.astype(np.float32))
    print(f"[save] {out_oof_p}")

    # ---- 6+7. Blend with nb2103 K=28 anchor ----
    if not NB2103_K28_OOF_PATH.exists():
        raise FileNotFoundError(f"missing nb2103 K=28 OOF: {NB2103_K28_OOF_PATH}")
    pred_nb2103 = np.load(NB2103_K28_OOF_PATH).astype(np.float64)
    if pred_nb2103.shape[0] != n_unb:
        raise ValueError(f"nb2103 K=28 OOF shape {pred_nb2103.shape} != {n_unb}")
    rae_nb2103_check = float(rae(y_unb, pred_nb2103))
    print(f"\n[blend] nb2103 K=28 OOF RAE (re-verified) = {rae_nb2103_check:.4f}")

    # Primary blend: 0.7 * nb2103 + 0.3 * scaf_knn
    pred_primary = 0.7 * pred_nb2103 + PRIMARY_BLEND_W_SCAF * pred_cf
    rae_primary = float(rae(y_unb, pred_primary))
    print(f"[blend] primary 0.7*nb2103 + 0.3*scaf = {rae_primary:.4f}  "
          f"(delta vs nb2103 = {rae_primary - rae_nb2103_check:+.4f})")

    # Sweep
    print(f"\n[blend sweep] w_scaf in {BLEND_W_GRID}")
    sweep_records = []
    for w in BLEND_W_GRID:
        pred_b = (1.0 - w) * pred_nb2103 + w * pred_cf
        rae_b = float(rae(y_unb, pred_b))
        delta_b = rae_b - rae_nb2103_check
        sweep_records.append({
            "w_scaf": float(w),
            "rae": rae_b,
            "delta_vs_nb2103": delta_b,
        })
        marker = ""
        if rae_b < rae_nb2103_check - 0.003:
            marker = "  <- BEATS_NB2103"
        elif abs(rae_b - rae_nb2103_check) < 0.003:
            marker = "  <- FLAT_VS_NB2103"
        print(f"   w_scaf={w:.2f}  RAE={rae_b:.4f}  "
              f"delta_vs_nb2103={delta_b:+.4f}{marker}")

    best_sweep_i = int(np.argmin([r["rae"] for r in sweep_records]))
    best_w = float(sweep_records[best_sweep_i]["w_scaf"])
    best_rae = float(sweep_records[best_sweep_i]["rae"])
    delta_best = best_rae - rae_nb2103_check
    print(f"\n[sweep] best w_scaf = {best_w:.2f}  RAE = {best_rae:.4f}  "
          f"(delta vs nb2103 = {delta_best:+.4f})")

    # Verdict
    if delta_best < -0.003:
        verdict = f"SCAF_KNN_BLEND_BEATS_NB2103_AT_W={best_w}"
    elif abs(delta_best) < 0.003:
        verdict = f"SCAF_KNN_BLEND_FLAT_VS_NB2103_AT_W={best_w}"
    else:
        verdict = "SCAF_KNN_DOES_NOT_BEAT_NB2103"
    print(f"[verdict] {verdict}")

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "method": "scaffold_restricted_tanimoto_knn_k3_blend_nb2103_K28",
        "k": K_NN,
        "folds": FOLDS,
        "seed": SEED,
        "n_train": int(len(tr_smi_std)),
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "n_unique_scaf_train": int(len(train_scaf_set)),
        "n_unique_scaf_test": int(len(test_scaf_set)),
        "n_overlap_scaf": int(len(overlap_scaf)),
        "n_test_with_scaf_match": int(n_test_with_scaf_match),
        "train_pec50_median": pool_median,
        "scaf_knn_train_only_RAE_on_253": rae_train_only,
        "scaf_knn_crossfit_RAE_on_253": rae_cf,
        "fallback_pct_train_only_pool": pct_fb_train_only,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "nb2103_K28_ref": NB2103_K28_REF,
        "nb2103_K28_reverified_oof": rae_nb2103_check,
        "delta_scaf_vs_chemprop_aux": rae_cf - CHEMPROP_AUX_REF,
        "delta_scaf_vs_nb2103": rae_cf - NB2103_K28_REF,
        "primary_blend_w_scaf": PRIMARY_BLEND_W_SCAF,
        "primary_blend_rae": rae_primary,
        "primary_blend_delta_vs_nb2103": rae_primary - rae_nb2103_check,
        "blend_w_grid": BLEND_W_GRID,
        "blend_sweep_records": sweep_records,
        "best_blend_w_scaf": best_w,
        "best_blend_rae": best_rae,
        "best_blend_delta_vs_nb2103": delta_best,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "n_train", "n_test", "n_unb",
        "n_unique_scaf_train", "n_unique_scaf_test", "n_overlap_scaf",
        "n_test_with_scaf_match",
        "scaf_knn_crossfit_RAE_on_253",
        "delta_scaf_vs_chemprop_aux", "delta_scaf_vs_nb2103",
        "primary_blend_rae", "primary_blend_delta_vs_nb2103",
        "best_blend_w_scaf", "best_blend_rae",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
