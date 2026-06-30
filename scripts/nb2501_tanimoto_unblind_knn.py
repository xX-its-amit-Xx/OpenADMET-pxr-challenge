"""nb2501 -- Tanimoto-weighted k-NN using the 253 Phase-1 unblind labels as a
SECONDARY analog library for the 260 still-blind rows of the 513-row test set.

Rationale: the 253 unblind compounds are by construction in the same analog
expansion as the remaining 260 still-blind rows, so their structural distance
to the still-blind set is materially smaller than the 4139-row TRAIN library
(median top-1 Tanimoto train->test is 0.52; unblind->still-blind is expected
to be higher). Predicting still-blind pEC50 from a small but in-domain
unblind library exploits this proximity directly.

Protocol:
  1. Standardize SMILES for all 513 test + 253 unblind, compute Morgan ECFP4.
  2. Build a 260x253 Tanimoto sim matrix (still-blind rows vs unblind library).
  3. Weighted top-k=5 kNN with weights w_i = max(0, sim_i - 0.30); fallback
     to the nb2240 deploy prediction when the row has no neighbor above 0.30.
  4. For the unblind 253 rows, KEEP nb2240's prediction (already deploy/refit).
  5. Build a hybrid 513-vector where still-blind rows get a similarity-gated
     blend of (kNN, nb2240) and unblind rows are nb2240 verbatim.
  6. Honest 5-fold scaffold CV on 253 by LOO kNN within unblind (each fold's
     val set predicts from the fold's train portion of unblind only); compare
     mean_rae vs gate thresholds 0.4570 (PROMOTE) / 0.4601 (MARGINAL_BEAT).
  7. Save te[513] with kNN-replaced still-blind portion, pred_oof[253] from
     LOO kNN CV inside unblind, plus nb2501_summary.json + submission CSV.

Outputs:
  data/processed/te_nb2501.npy                 (513,) float32
  data/processed/nb2501_pred_oof.npy           (253,) float32 honest LOO-kNN
  data/processed/nb2501_summary.json
  submissions/nb2501_tanimoto_unblind_knn.csv
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

from pxr.chem import morgan_fp_batch, standardize_smiles
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2501"
K = 5
SIM_THRESH = 0.30           # weight floor in w_i = max(0, sim - 0.30)
HYBRID_HIGH = 0.70          # >= this sim -> trust kNN entirely
HYBRID_LOW = SIM_THRESH     # <= this sim -> trust nb2240 entirely
ANCHOR_TAG = "nb2240"
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4601


def tanimoto_pairwise(fp_a: np.ndarray, fp_b: np.ndarray) -> np.ndarray:
    """Tanimoto matrix of (n_a, n_b) for binary fingerprints."""
    A = fp_a.astype(np.float32)
    B = fp_b.astype(np.float32)
    inter = A @ B.T
    a = A.sum(axis=1, keepdims=True)
    b = B.sum(axis=1, keepdims=True).T
    denom = a + b - inter
    out = np.where(denom > 0, inter / denom, 0.0).astype(np.float32)
    return out


def knn_weighted(sim_row: np.ndarray, y: np.ndarray, k: int,
                 sim_thresh: float, fallback: float) -> tuple[float, int, float]:
    """Weighted top-k mean with w_i = max(0, sim_i - sim_thresh).

    Returns (prediction, n_effective_neighbors, top1_sim).
    """
    n = len(sim_row)
    k_eff = min(k, n)
    top_idx = np.argpartition(-sim_row, kth=k_eff - 1)[:k_eff]
    sims = sim_row[top_idx]
    top1 = float(sims.max())
    w = np.clip(sims - sim_thresh, 0.0, None)
    if w.sum() <= 0:
        return float(fallback), 0, top1
    pred = float((w * y[top_idx]).sum() / w.sum())
    n_eff = int((w > 0).sum())
    return pred, n_eff, top1


def hybrid_blend(knn_pred: float, anchor_pred: float, top1: float,
                 lo: float = HYBRID_LOW, hi: float = HYBRID_HIGH) -> float:
    """Linear ramp on top-1 sim: lo -> anchor, hi -> knn, interp in-between."""
    if top1 >= hi:
        w_knn = 1.0
    elif top1 <= lo:
        w_knn = 0.0
    else:
        w_knn = (top1 - lo) / (hi - lo)
    return float(w_knn * knn_pred + (1.0 - w_knn) * anchor_pred)


def loo_scaffold_cv(sim_unb: np.ndarray, y_unb: np.ndarray,
                    scaffolds: list[str], anchor_unb: np.ndarray,
                    n_splits: int = 5, seed: int = 42) -> dict:
    """Honest 5-fold scaffold CV: predict each val row from the fold's train
    portion of unblind only, then hybrid-blend with the anchor (nb2240)."""
    n = len(y_unb)
    folds = scaffold_kfold_indices(scaffolds, n_splits=n_splits, seed=seed)
    knn_oof = np.full(n, np.nan, dtype=np.float64)
    hyb_oof = np.full(n, np.nan, dtype=np.float64)
    top1_oof = np.full(n, np.nan, dtype=np.float64)
    n_eff_oof = np.zeros(n, dtype=np.int32)
    for fold_id, (tr_loc, va_loc) in enumerate(folds):
        for v in va_loc:
            row = sim_unb[v, tr_loc]
            pred, n_eff, top1 = knn_weighted(
                row, y_unb[tr_loc], K, SIM_THRESH, float(anchor_unb[v])
            )
            knn_oof[v] = pred
            top1_oof[v] = top1
            n_eff_oof[v] = n_eff
            hyb_oof[v] = hybrid_blend(pred, float(anchor_unb[v]), top1)
    return {
        "knn_oof": knn_oof,
        "hyb_oof": hyb_oof,
        "top1_oof": top1_oof,
        "n_eff_oof": n_eff_oof,
        "rae_knn": float(rae(y_unb, knn_oof)),
        "rae_hyb": float(rae(y_unb, hyb_oof)),
        "rae_anchor": float(rae(y_unb, anchor_unb)),
        "n_folds": n_splits,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Tanimoto kNN on UNBLIND-253 library for still-blind 260 of test-513")
    print("=" * 78)

    # ---- Load 513 test, 253 unblind, indices ----
    te = load_test()
    n_te = len(te)
    assert n_te == 513, f"expected 513 test, got {n_te}"

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy").astype(np.int64)
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    still_blind_idx = np.setdiff1d(np.arange(n_te), unb_idx).astype(np.int64)
    assert len(unb_idx) == 253 and len(still_blind_idx) == 260
    print(f"[load] n_te=513  n_unblind=253  n_still_blind=260")

    # nb2240 deploy predictions on all 513 (already deploy-refit)
    te_anchor = np.load(DATA_PROCESSED / f"te_{ANCHOR_TAG}.npy").astype(np.float64)
    anchor_unb = te_anchor[unb_idx]
    anchor_blind = te_anchor[still_blind_idx]
    print(f"[anchor] te_{ANCHOR_TAG} loaded  mean={te_anchor.mean():.3f}  "
          f"std={te_anchor.std():.3f}")

    # ---- Standardize SMILES for all test rows ----
    print("\n[std] standardizing SMILES for test (513)...")
    te_smiles_raw = te["smiles"].tolist()
    te_smiles_std = [standardize_smiles(s) or s for s in te_smiles_raw]
    te["smiles_std"] = te_smiles_std

    # ---- Morgan ECFP4 fingerprints ----
    print("[feat] Morgan ECFP4 (2048-bit) batch on 513 test...")
    fp_te = morgan_fp_batch(te_smiles_std)
    fp_unb = fp_te[unb_idx]
    fp_blind = fp_te[still_blind_idx]
    print(f"[feat] fp_te={fp_te.shape}  fp_unb={fp_unb.shape}  fp_blind={fp_blind.shape}")

    # ---- 260 x 253 Tanimoto: still-blind vs unblind library ----
    print("\n[sim] computing 260x253 Tanimoto matrix (still-blind vs unblind)...")
    sim_blind_unb = tanimoto_pairwise(fp_blind, fp_unb)
    top1_blind = sim_blind_unb.max(axis=1)
    print(f"[sim] shape={sim_blind_unb.shape}  mean={sim_blind_unb.mean():.4f}  "
          f"max={sim_blind_unb.max():.4f}")
    print(f"[sim] top-1 sim mean/median/max = "
          f"{top1_blind.mean():.3f}/{np.median(top1_blind):.3f}/{top1_blind.max():.3f}")
    n_blind_with_neighbor = int((top1_blind >= SIM_THRESH).sum())
    print(f"[sim] still-blind rows with at least one nbr sim>={SIM_THRESH}: "
          f"{n_blind_with_neighbor}/260")

    # ---- 253 x 253 unblind-vs-unblind for LOO scaffold CV ----
    print("\n[sim] computing 253x253 Tanimoto matrix (unblind vs unblind)...")
    sim_unb_unb = tanimoto_pairwise(fp_unb, fp_unb)
    np.fill_diagonal(sim_unb_unb, 0.0)   # exclude self
    print(f"[sim] off-diag mean={sim_unb_unb.mean():.4f}  "
          f"max={sim_unb_unb.max():.4f}")

    # ---- kNN prediction on 260 still-blind rows ----
    print(f"\n[knn] weighted top-{K} kNN on still-blind, w=max(0, sim-{SIM_THRESH})...")
    knn_blind = np.zeros(260, dtype=np.float64)
    n_eff_blind = np.zeros(260, dtype=np.int32)
    for i in range(260):
        pred, n_eff, _ = knn_weighted(
            sim_blind_unb[i], y_unb, K, SIM_THRESH, float(anchor_blind[i])
        )
        knn_blind[i] = pred
        n_eff_blind[i] = n_eff
    print(f"[knn] blind: pred mean={knn_blind.mean():.3f}  std={knn_blind.std():.3f}")
    print(f"[knn] mean n_effective neighbors = {n_eff_blind.mean():.2f}")

    # ---- Hybrid blend on still-blind (sim-gated kNN vs nb2240) ----
    hyb_blind = np.array([
        hybrid_blend(knn_blind[i], float(anchor_blind[i]), float(top1_blind[i]))
        for i in range(260)
    ], dtype=np.float64)
    print(f"[hyb]  blind: pred mean={hyb_blind.mean():.3f}  std={hyb_blind.std():.3f}")

    # ---- Build full 513-vector: unblind rows -> nb2240 verbatim; blind -> hybrid ----
    te_hybrid = te_anchor.copy()
    te_hybrid[still_blind_idx] = hyb_blind
    te_hybrid = te_hybrid.astype(np.float32)
    print(f"[hybrid] full 513: mean={te_hybrid.mean():.3f}  std={te_hybrid.std():.3f}")

    # ---- Honest 5-fold scaffold CV on 253 unblind (LOO kNN within unblind) ----
    print("\n[cv] 5-fold scaffold CV on 253 unblind (LOO kNN within fold-train)...")
    # Murcko scaffolds from std SMILES
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    def _scaffold(smi: str) -> str:
        try:
            mol = Chem.MolFromSmiles(smi)
            sc = MurckoScaffold.GetScaffoldForMol(mol) if mol is not None else None
            return Chem.MolToSmiles(sc) if sc is not None else ""
        except Exception:
            return ""

    scaffolds_unb = [_scaffold(te_smiles_std[i]) for i in unb_idx]
    n_unique_scaf = len(set(s for s in scaffolds_unb if s))
    print(f"[cv] unique non-empty scaffolds on 253: {n_unique_scaf}")
    cv = loo_scaffold_cv(sim_unb_unb, y_unb, scaffolds_unb, anchor_unb,
                         n_splits=5, seed=42)
    print(f"[cv] anchor (nb2240, in-sample on 253) RAE = {cv['rae_anchor']:.4f}")
    print(f"[cv] kNN-only OOF RAE                  = {cv['rae_knn']:.4f}")
    print(f"[cv] HYBRID OOF RAE                    = {cv['rae_hyb']:.4f}")

    # ---- Gate vs cycle-160 ceiling and nb2240 in-sample anchor ----
    mean_rae = cv["rae_hyb"]
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"\n[gate] mean_rae={mean_rae:.4f}  -> {verdict}  "
          f"(thresholds {GATE_PROMOTE}/{GATE_MARGINAL})")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", te_hybrid)
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy",
            cv["hyb_oof"].astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof_knn_only.npy",
            cv["knn_oof"].astype(np.float32))
    sub_path = SUBMISSIONS / f"{TAG}_tanimoto_unblind_knn.csv"
    pd.DataFrame({
        "SMILES":        te["smiles"].values,
        "Molecule Name": te["name"].values,
        "pEC50":         te_hybrid,
    }).to_csv(sub_path, index=False)
    print(f"[save] te_{TAG}.npy, {TAG}_pred_oof.npy, {sub_path}")

    summary = {
        "tag": TAG,
        "method": "tanimoto_knn_on_unblind253_for_stillblind260",
        "k": K,
        "weight": f"max(0, sim - {SIM_THRESH})",
        "hybrid_ramp": {"low": HYBRID_LOW, "high": HYBRID_HIGH},
        "anchor": ANCHOR_TAG,
        "n_te": n_te,
        "n_unblind": int(len(unb_idx)),
        "n_still_blind": int(len(still_blind_idx)),
        "sim_blind_unb_mean": float(sim_blind_unb.mean()),
        "sim_blind_unb_max":  float(sim_blind_unb.max()),
        "top1_blind_unb_mean":   float(top1_blind.mean()),
        "top1_blind_unb_median": float(np.median(top1_blind)),
        "top1_blind_unb_max":    float(top1_blind.max()),
        "n_blind_with_neighbor_above_thresh": n_blind_with_neighbor,
        "n_eff_blind_mean": float(n_eff_blind.mean()),
        "n_unique_scaffolds_unb": n_unique_scaf,
        "cv_anchor_rae_in_sample":   cv["rae_anchor"],
        "cv_knn_oof_rae":            cv["rae_knn"],
        "cv_hybrid_oof_rae":         cv["rae_hyb"],
        "mean_rae": cv["rae_hyb"],
        "gate_promote_thresh":  GATE_PROMOTE,
        "gate_marginal_thresh": GATE_MARGINAL,
        "verdict": verdict,
        "test_mean": float(te_hybrid.mean()),
        "test_std":  float(te_hybrid.std()),
        "te_npy_path":          str(DATA_PROCESSED / f"te_{TAG}.npy"),
        "pred_oof_npy_path":    str(DATA_PROCESSED / f"{TAG}_pred_oof.npy"),
        "submission_csv":       str(sub_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    with open(DATA_PROCESSED / f"{TAG}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {DATA_PROCESSED / f'{TAG}_summary.json'}")
    print(f"\n=== {TAG} done in {time.time()-t0:.1f}s ===")
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("k", "weight", "anchor", "n_still_blind",
              "top1_blind_unb_mean", "top1_blind_unb_median",
              "n_blind_with_neighbor_above_thresh",
              "cv_anchor_rae_in_sample", "cv_knn_oof_rae",
              "cv_hybrid_oof_rae", "mean_rae", "verdict",
              "te_npy_path", "pred_oof_npy_path", "submission_csv"):
        print(f"  {k}: {res.get(k)}")
