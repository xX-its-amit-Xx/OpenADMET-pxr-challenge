"""nb930 -- Random Forest with Tanimoto-Proximity hybrid prediction.

Hypothesis: pure RF interpolates poorly on novel scaffolds. By using each tree's
leaf membership to identify "structurally close per the tree", but then weighting
those neighbors by Tanimoto similarity (not by leaf size), we get a hybrid kNN/RF
that is biased toward chemically similar training compounds.

Pipeline:
  1) Morgan FP (2048, r=2) + RDKit descriptors for train (4139) + test (513).
  2) RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_leaf=5).
  3) baseline_pred = RF.predict(X_te)                          # (513,)
  4) proximity_pred:
       for each tree:
         leaf_tr = tree.apply(X_tr); leaf_te = tree.apply(X_te)
         for each test sample i:
           same-leaf train idx -> S_i  (set)
           weight by Tanimoto(test_i, train_j)  for j in S_i
       sum/average across trees and normalize per row.
  5) final = 0.5 * baseline_pred + 0.5 * proximity_pred
  6) in_RAE on 253 unblind rows. Save te + CSV submission.

Artifacts: C:/pxr_artifacts/nb930/
Wall-time budget: < 10 min.
"""
from __future__ import annotations

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
from sklearn.ensemble import RandomForestRegressor

from pxr.chem import morgan_fp_batch
from pxr.data import load_train, load_test
from pxr.eval import rae
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

ART = Path("C:/pxr_artifacts/nb930")
ART.mkdir(parents=True, exist_ok=True)

SEED = 42
N_EST = 500
MAX_DEPTH = 15
MIN_LEAF = 5
WALL_BUDGET_S = 600  # 10 min


def tanimoto_block(fp_te_rows: np.ndarray, fp_tr: np.ndarray) -> np.ndarray:
    """Vectorized Tanimoto for binary fingerprints.

    fp_te_rows : (Bt, n_bits) uint8
    fp_tr      : (Ntr, n_bits) uint8
    returns    : (Bt, Ntr) float32 Tanimoto
    """
    a = fp_te_rows.astype(np.float32)
    b = fp_tr.astype(np.float32)
    inter = a @ b.T  # (Bt, Ntr)
    sa = a.sum(axis=1, keepdims=True)  # (Bt, 1)
    sb = b.sum(axis=1, keepdims=True).T  # (1, Ntr)
    union = sa + sb - inter
    return (inter / np.maximum(union, 1.0)).astype(np.float32)


def main():
    t0 = time.time()
    print("=== nb930: RF + Tanimoto-proximity hybrid ===")

    tr = load_train()
    te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    print(f"train={len(tr)}  test={len(te)}")

    print("Featurizing (combined Morgan + RDKit)...")
    X_tr = impute(combined(tr["smiles"].tolist()))
    X_te = impute(combined(te["smiles"].tolist()))
    print(f"  X_tr={X_tr.shape}  X_te={X_te.shape}  "
          f"mem~{X_tr.nbytes/1e6:.1f}MB")

    print("Morgan FPs (binary) for Tanimoto...")
    fp_tr = morgan_fp_batch(tr["smiles"].tolist(), radius=2, n_bits=2048).astype(np.uint8)
    fp_te = morgan_fp_batch(te["smiles"].tolist(), radius=2, n_bits=2048).astype(np.uint8)
    print(f"  fp_tr={fp_tr.shape}  fp_te={fp_te.shape}")

    print(f"Fitting RF (n_est={N_EST}, depth={MAX_DEPTH}, min_leaf={MIN_LEAF})...")
    t_fit = time.time()
    rf = RandomForestRegressor(
        n_estimators=N_EST,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_LEAF,
        n_jobs=-1,
        random_state=SEED,
    )
    rf.fit(X_tr, y_tr)
    print(f"  fit done in {time.time()-t_fit:.1f}s")

    if time.time() - t0 > WALL_BUDGET_S * 0.6:
        print("WARN: fit consumed >60% of wall budget")

    baseline_pred = rf.predict(X_te).astype(np.float64)
    print(f"  baseline_pred mean={baseline_pred.mean():.3f}  "
          f"std={baseline_pred.std():.3f}")

    # Proximity-weighted: leaf membership x Tanimoto similarity
    print("Computing leaf indices for all trees (apply)...")
    t_apply = time.time()
    leaf_tr = rf.apply(X_tr)  # (Ntr, N_EST) int32
    leaf_te = rf.apply(X_te)  # (Nte, N_EST) int32
    print(f"  apply done in {time.time()-t_apply:.1f}s  "
          f"leaf_tr={leaf_tr.shape}")

    # Tanimoto in chunks to keep memory under control: 513*4139 = 2.1M floats = 8MB.
    print("Tanimoto sim matrix (Nte x Ntr)...")
    t_tan = time.time()
    BS = 64
    T_te_tr = np.empty((len(te), len(tr)), dtype=np.float32)
    for s in range(0, len(te), BS):
        e = min(s + BS, len(te))
        T_te_tr[s:e] = tanimoto_block(fp_te[s:e], fp_tr)
    print(f"  Tanimoto done in {time.time()-t_tan:.1f}s  "
          f"shape={T_te_tr.shape}  median={np.median(T_te_tr):.3f}")

    # For each tree, for each test row i: find train rows in same leaf,
    # weight their y by Tanimoto sim to test_i. Sum across trees.
    print("Proximity-weighted prediction (per-tree leaf x Tanimoto)...")
    t_prox = time.time()
    n_te = len(te)
    sum_wy = np.zeros(n_te, dtype=np.float64)
    sum_w = np.zeros(n_te, dtype=np.float64)
    y_tr_f = y_tr.astype(np.float64)

    for t_idx in range(N_EST):
        if time.time() - t0 > WALL_BUDGET_S - 30:
            print(f"WARN: wall budget near limit at tree {t_idx}; stopping early")
            break

        lt = leaf_tr[:, t_idx]   # (Ntr,)
        le = leaf_te[:, t_idx]   # (Nte,)

        # Group train indices by leaf id for this tree
        # Sort train by leaf for fast group lookup
        order = np.argsort(lt, kind="stable")
        lt_sorted = lt[order]
        # For each unique leaf in this tree, find its boundaries
        uniq, first = np.unique(lt_sorted, return_index=True)
        # Map: leaf_id -> (start, end) slice in `order`
        # Build a dict only over leaves that occur in test set for speed
        te_leaves = np.unique(le)
        # For each test leaf, get matching train indices
        # Build positional dict via searchsorted on uniq
        pos = np.searchsorted(uniq, te_leaves)
        # Some te_leaves may not exist in lt (rare with depth=15); mask them
        valid = (pos < len(uniq)) & (uniq[np.minimum(pos, len(uniq)-1)] == te_leaves)

        # leaf -> train index array
        leaf2train = {}
        for k, leaf_id in enumerate(te_leaves):
            if not valid[k]:
                continue
            p = pos[k]
            start = first[p]
            end = first[p + 1] if p + 1 < len(first) else len(lt_sorted)
            leaf2train[leaf_id] = order[start:end]

        # Score each test row
        for i in range(n_te):
            tr_idx = leaf2train.get(le[i])
            if tr_idx is None or len(tr_idx) == 0:
                continue
            w = T_te_tr[i, tr_idx]  # (k,) Tanimoto sims
            sw = float(w.sum())
            if sw <= 0:
                continue
            sum_wy[i] += float((w * y_tr_f[tr_idx]).sum())
            sum_w[i] += sw

    # Normalize across trees
    prox_pred = np.where(sum_w > 0, sum_wy / np.maximum(sum_w, 1e-9), baseline_pred)
    # Fallback for test rows with zero proximity weight anywhere -> baseline
    zero_mask = sum_w <= 0
    if zero_mask.any():
        print(f"  {int(zero_mask.sum())} test rows had zero proximity weight; "
              f"using RF baseline for those")
        prox_pred[zero_mask] = baseline_pred[zero_mask]

    print(f"  proximity done in {time.time()-t_prox:.1f}s  "
          f"prox mean={prox_pred.mean():.3f}  std={prox_pred.std():.3f}")

    final = 0.5 * baseline_pred + 0.5 * prox_pred
    print(f"  final mean={final.mean():.3f}  std={final.std():.3f}")

    # Save artifacts
    np.save(ART / "baseline_rf.npy", baseline_pred)
    np.save(ART / "proximity_pred.npy", prox_pred)
    np.save(ART / "final.npy", final)
    np.save(DATA_PROCESSED / "te_nb930.npy", final)
    np.save(DATA_PROCESSED / "te_nb930_baseline_rf.npy", baseline_pred)
    np.save(DATA_PROCESSED / "te_nb930_proximity.npy", prox_pred)

    # In-RAE on 253 unblind
    unb = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te["name"])}
    unb_te_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"] if n in name_to_idx]
    )
    unb_y = unb.loc[unb["Molecule Name"].isin(name_to_idx), "pEC50"].values

    in_rae_base = rae(unb_y, baseline_pred[unb_te_idx])
    in_rae_prox = rae(unb_y, prox_pred[unb_te_idx])
    in_rae_final = rae(unb_y, final[unb_te_idx])
    print(f"\nUnblind n={len(unb_te_idx)}")
    print(f"  in_RAE baseline RF       = {in_rae_base:.4f}")
    print(f"  in_RAE proximity         = {in_rae_prox:.4f}")
    print(f"  in_RAE 0.5/0.5 hybrid    = {in_rae_final:.4f}")

    # Submission CSV (SMILES + Molecule Name + pEC50)
    sub = pd.DataFrame({
        "Molecule Name": te["name"].values,
        "SMILES": te["smiles"].values,
        "pEC50": final,
    })
    out_csv = SUBMISSIONS / "nb930_rf_tanimoto_hybrid.csv"
    sub.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    # Persist metrics
    metrics = {
        "in_rae_baseline_rf": float(in_rae_base),
        "in_rae_proximity": float(in_rae_prox),
        "in_rae_final": float(in_rae_final),
        "n_unblind": int(len(unb_te_idx)),
        "wall_time_s": float(time.time() - t0),
    }
    pd.Series(metrics).to_json(ART / "metrics.json")
    print(f"Done in {time.time()-t0:.1f}s")
    return metrics


if __name__ == "__main__":
    main()
