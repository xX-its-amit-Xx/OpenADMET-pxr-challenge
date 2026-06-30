"""nb2670 -- Hyperdimensional Computing (HD) on Morgan FP.

NEW PARADIGM: HD encoding via random bipolar projection of Morgan FP-2048
to 10000-d hypervectors, then similarity-based prediction via cosine k-NN.
Distinct from every tree/linear/SLSQP paradigm in the ladder.

CONTEXT:
    The post-hoc-blend ceiling on chemprop_aux anchor at n=253 has co-converged
    at 0.4718-0.4720 (nb2095/nb1191/nb2060 deep-30) with nb2171's anchor-swap
    breakthrough at 0.4682.  Per the cycle-134 paradigm-exhaustion thesis,
    further gains require SUBSTRATE change, not operator refinement.

    HD computing (Kanerva 2009, Rahimi 2017) is a *holographic* encoding
    scheme totally distinct from tree splits and L2 regressors:
      - Project sparse binary FP into a dense bipolar hypervector via a
        fixed random {-1,+1} projection matrix
      - Sign() the resulting vector to make it bipolar again
      - Similarity is cosine in the hypervector space
      - Prediction is a similarity-weighted mean of k nearest training labels
    This is a different inductive bias from LGBM (axis-aligned splits) or
    LASSO/Ridge (sparse linear) -- it captures *holographic* structural
    similarity in a high-dim distributed code.

PROTOCOL:
    1.  Compute Morgan FP-2048 for the 253 unblind compounds and the 513 test.
    2.  Sample R ~ {-1,+1}^(2048, 10000) with seed=42 (bipolar, fixed).
    3.  H[mol] = sign(R.T @ Morgan[mol])    (in {-1,0,+1})
    4.  5-fold scaffold CV on 253 unblind (kf_seed=1001):
          For each fold:
            train = unblind[tr], val = unblind[va]
            For each val row: cosine-similarity to all 253-train HD vectors
              Top-5 NN by cosine, sim-weighted mean of pEC50 labels -> pred
        pred_oof = stitched 253 predictions
    5.  Full-fit deploy:
          For each test row: cosine to all 253 unblind HD vectors -> top-5
          sim-weighted mean -> te[i]  (513 predictions)
    6.  Save artifacts and gate.

GATE:
    mean_rae < 0.4570 -> PROMOTE
    mean_rae < 0.4598 -> MARGINAL_BEAT
    else              -> FAIL

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    (test SMILES via pxr.data.load_test)

Outputs:
    data/processed/nb2670_summary.json
    data/processed/nb2670_pred_oof.npy   (253,) float32
    data/processed/te_nb2670.npy         (513,) float32
    submissions/nb2670_hyperdimensional_computing.csv
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
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2670"

# -- Protocol params --------------------------------------------------------
HD_DIM = 10000
PROJ_SEED = 42
KF_SEED = 1001
N_FOLDS = 5
KNN_K = 5
SIM_FLOOR = 1e-9
MORGAN_BITS = 2048

# -- Gates -----------------------------------------------------------------
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598


def _safe_can_smiles(smi):
    m = standardize(smi)
    return Chem.MolToSmiles(m) if m is not None else ""


def hd_encode(morgan_fp: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Project sparse binary Morgan FP (N, 2048) -> bipolar HD (N, 10000).

    HD[i] = sign(R.T @ Morgan[i])  in {-1, 0, +1}
    Ties (proj == 0) get mapped to 0 to avoid arbitrary sign assignment.
    """
    proj = morgan_fp.astype(np.float32) @ R.astype(np.float32)
    return np.sign(proj).astype(np.float32)


def cosine_topk(query_hd: np.ndarray, pool_hd: np.ndarray, k: int):
    """Cosine similarity top-k per row.

    query_hd: (Nq, D), pool_hd: (Np, D); returns (top_idx, top_sim) each (Nq, k).
    """
    q_norm = np.linalg.norm(query_hd, axis=1, keepdims=True)
    p_norm = np.linalg.norm(pool_hd, axis=1, keepdims=True)
    q_norm = np.maximum(q_norm, 1e-9)
    p_norm = np.maximum(p_norm, 1e-9)
    qn = query_hd / q_norm
    pn = pool_hd / p_norm
    sim = qn @ pn.T  # (Nq, Np)
    n_pool = pool_hd.shape[0]
    if k >= n_pool:
        idx_part = np.argsort(-sim, axis=1)[:, :k]
        row_idx = np.arange(query_hd.shape[0])[:, None]
        return idx_part, sim[row_idx, idx_part]
    part = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    row_idx = np.arange(query_hd.shape[0])[:, None]
    sim_part = sim[row_idx, part]
    order = np.argsort(-sim_part, axis=1)
    idx_top = part[row_idx, order]
    sim_top = sim[row_idx, idx_top]
    return idx_top, sim_top


def sim_weighted_predict(top_idx: np.ndarray, top_sim: np.ndarray,
                         pool_labels: np.ndarray, fallback: float) -> np.ndarray:
    """Sim-weighted mean of pool_labels over top-k neighbors per row."""
    w = np.clip(top_sim, 0.0, None)  # cosine can be negative; clip to >=0
    w_sum = w.sum(axis=1)
    n_q = top_idx.shape[0]
    pred = np.empty(n_q, dtype=np.float32)
    for i in range(n_q):
        if w_sum[i] < SIM_FLOOR:
            pred[i] = fallback
        else:
            pred[i] = float(np.sum(w[i] * pool_labels[top_idx[i]]) / w_sum[i])
    return pred


def main():
    import pandas as pd
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Hyperdimensional Computing on Morgan FP")
    print(f"   HD_DIM={HD_DIM}  PROJ_SEED={PROJ_SEED}  KNN_K={KNN_K}")
    print(f"   {N_FOLDS}-fold scaffold CV, kf_seed={KF_SEED}")
    print(f"   gate PROMOTE<{GATE_PROMOTE}  MARGINAL<{GATE_MARGINAL}")
    print("=" * 78)

    # ---- Load truth + test SMILES ----
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values
                if "name" in te.columns else te["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles_raw = [te_smiles[i] for i in unb_idx]
    unb_std_smiles = [_safe_can_smiles(s) for s in unb_smiles_raw]
    te_std_smiles = [_safe_can_smiles(s) for s in te_smiles]

    # Scaffold for CV
    unb_scaffolds = [bemis_murcko(s) for s in unb_std_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    # ---- Morgan FP-2048 ----
    print("\n" + "-" * 78)
    print(f"STEP 1: Morgan FP-{MORGAN_BITS}")
    print("-" * 78)
    fp_unb = morgan_fp_batch(unb_std_smiles)
    fp_te = morgan_fp_batch(te_std_smiles)
    print(f"   fp_unb={fp_unb.shape}  fp_te={fp_te.shape}")
    print(f"   fp_unb density mean = {fp_unb.mean():.4f}")

    # ---- Random bipolar projection ----
    print("\n" + "-" * 78)
    print(f"STEP 2: bipolar projection R in {{-1,+1}}^({MORGAN_BITS},{HD_DIM})")
    print("-" * 78)
    rng = np.random.default_rng(PROJ_SEED)
    R = rng.choice(np.array([-1, 1], dtype=np.int8), size=(MORGAN_BITS, HD_DIM))
    R = R.astype(np.float32)
    print(f"   R.shape={R.shape}  R.dtype={R.dtype}  "
          f"R.mean={R.mean():+.4f} (expected ~0)")

    # ---- HD encoding ----
    print("\n" + "-" * 78)
    print(f"STEP 3: HD encoding H = sign(R.T @ Morgan)")
    print("-" * 78)
    hd_unb = hd_encode(fp_unb, R)
    hd_te = hd_encode(fp_te, R)
    print(f"   hd_unb={hd_unb.shape}  hd_te={hd_te.shape}")
    print(f"   hd_unb fraction nonzero = {(hd_unb != 0).mean():.4f}")
    print(f"   hd_unb mean = {hd_unb.mean():+.4f}  std = {hd_unb.std():.4f}")
    print(f"   hd_te  fraction nonzero = {(hd_te != 0).mean():.4f}")

    fallback = float(np.mean(y_unb))
    print(f"   fallback (mean truth) = {fallback:.4f}")

    # ---- 5-fold scaffold CV on 253 unblind ----
    print("\n" + "-" * 78)
    print(f"STEP 4: {N_FOLDS}-fold scaffold CV, kf_seed={KF_SEED}")
    print("-" * 78)
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=KF_SEED,
    )
    pred_oof = np.full(n_unb, np.nan, dtype=np.float64)
    fold_rae = []
    for f_i, (tr_loc, va_loc) in enumerate(splits):
        hd_tr = hd_unb[tr_loc]
        hd_va = hd_unb[va_loc]
        y_tr = y_unb[tr_loc].astype(np.float32)
        k_eff = min(KNN_K, len(tr_loc))
        top_idx, top_sim = cosine_topk(hd_va, hd_tr, k=k_eff)
        fold_fallback = float(np.mean(y_tr))
        pred_va = sim_weighted_predict(top_idx, top_sim, y_tr, fold_fallback)
        pred_oof[va_loc] = pred_va
        fr = float(rae(y_unb[va_loc], pred_va))
        fold_rae.append(fr)
        sim_mean = float(top_sim.mean())
        print(f"   fold {f_i+1}/{N_FOLDS}  n_va={len(va_loc):3d}  "
              f"k={k_eff}  mean_top_cos={sim_mean:+.4f}  RAE={fr:.4f}")
    if np.isnan(pred_oof).any():
        raise RuntimeError("scaffold split did not cover all rows")

    pooled_rae = float(rae(y_unb, pred_oof))
    fold_arr = np.array(fold_rae, dtype=np.float64)
    fold_mean = float(fold_arr.mean())
    fold_std = float(fold_arr.std(ddof=1))
    print(f"\n   per-fold RAE: " + ", ".join(f"{r:.4f}" for r in fold_rae))
    print(f"   mean fold RAE  = {fold_mean:.4f} +/- {fold_std:.4f}")
    print(f"   POOLED RAE     = {pooled_rae:.4f}")
    print(f"   pred_oof std   = {pred_oof.std():.3f}  truth_std = {y_unb.std():.3f}")

    # ---- Deploy: full-fit -> 513 test predictions ----
    print("\n" + "-" * 78)
    print("STEP 5: deploy full-fit -> 513 test predictions")
    print("-" * 78)
    top_idx_te, top_sim_te = cosine_topk(hd_te, hd_unb, k=KNN_K)
    pred_te_513 = sim_weighted_predict(top_idx_te, top_sim_te, y_unb.astype(np.float32),
                                        fallback)
    te_unb_in_rae = float(rae(y_unb, pred_te_513[unb_idx]))
    print(f"   pred_te_513 mean/std = {pred_te_513.mean():.3f} / "
          f"{pred_te_513.std():.3f}")
    print(f"   mean top-5 cos sim (test->unb) = {top_sim_te.mean():+.4f}")
    print(f"   te[unb_idx] in-sample RAE      = {te_unb_in_rae:.4f}")
    print(f"   (in-sample: each unb row picks itself among top-5)")

    # ---- Gate ----
    print("\n" + "-" * 78)
    print("STEP 6: GATE")
    print("-" * 78)
    if pooled_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif pooled_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"   pooled_rae = {pooled_rae:.4f}")
    print(f"     <{GATE_PROMOTE} -> PROMOTE")
    print(f"     <{GATE_MARGINAL} -> MARGINAL_BEAT")
    print(f"     else            -> FAIL")
    print(f"   -> {verdict}")

    # ---- Save artifacts ----
    print("\n" + "-" * 78)
    print("STEP 7: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof.astype(np.float32))
    np.save(te_path, pred_te_513.astype(np.float32))
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_hyperdimensional_computing.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": pred_te_513.astype(np.float32),
    }).to_csv(sub_csv, index=False)
    print(f"   [save] {sub_csv}")

    summary = {
        "tag": TAG,
        "method": "hyperdimensional_computing_morgan_bipolar_cosine_knn",
        "paradigm": "HD_holographic_encoding",
        "morgan_bits": MORGAN_BITS,
        "hd_dim": HD_DIM,
        "proj_seed": PROJ_SEED,
        "knn_k": KNN_K,
        "kf_seed": KF_SEED,
        "n_folds": N_FOLDS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "anchor_pre_unblind": True,
        "uses_external_anchor": False,
        "fold_rae": fold_rae,
        "fold_mean_rae": fold_mean,
        "fold_std_rae": fold_std,
        "mean_rae": pooled_rae,
        "pred_oof_std": float(pred_oof.std()),
        "truth_std": float(y_unb.std()),
        "te_unb_in_sample_rae": te_unb_in_rae,
        "te_mean": float(pred_te_513.mean()),
        "te_std": float(pred_te_513.std()),
        "mean_top5_cos_sim_te_to_unb": float(top_sim_te.mean()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv),
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   paradigm      = HD computing (bipolar projection, cosine kNN)")
    print(f"   HD_DIM        = {HD_DIM}")
    print(f"   mean RAE      = {pooled_rae:.4f}  (gate {GATE_PROMOTE}/{GATE_MARGINAL})")
    print(f"   fold mean+/-std = {fold_mean:.4f} +/- {fold_std:.4f}")
    print(f"   verdict       = {verdict}")
    print(f"   wall          = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae",
        "fold_mean_rae",
        "fold_std_rae",
        "pred_oof_std",
        "te_unb_in_sample_rae",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
