"""nb117 — KNN Residual Correction on nb109 Deep Meta-Stack.

Key insight: nb109 has systematic residuals correlated with structural
similarity — similar compounds have similar errors. KNN on ECFP4 can
predict the residual (y_true - nb109_oof) for validation compounds using
training-fold neighbors' residuals. At test time, we look up the k most
similar training compounds and predict what nb109's error would be, then
add the correction.

This is targeted delta-ML: we don't re-predict from scratch; we just correct
the best existing model (nb109, OOF RAE=0.3741) for its known error patterns.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko, morgan_fp_batch
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
N_FOLDS = 5
K_NEIGHBORS = [3, 5, 10, 20]
SIM_THRESH = 0.25  # min Tanimoto for a neighbor to count
MAX_CORRECTION = 0.5  # clip residual correction magnitude


def tanimoto_row(fp_q, fps_db):
    """Tanimoto similarity between one query fp and a database of fps."""
    fp_q = fp_q.astype(np.float32)
    fps_db = fps_db.astype(np.float32)
    dot = fps_db @ fp_q
    q_bits = fp_q @ fp_q
    db_bits = (fps_db * fps_db).sum(axis=1)
    union = q_bits + db_bits - dot
    sim = np.where(union > 0, dot / union, 0.0)
    return sim


def tanimoto_matrix(fps_q, fps_db, batch=256):
    """Return (N_q, N_db) Tanimoto matrix, computed in batches."""
    n_q = len(fps_q)
    n_db = len(fps_db)
    out = np.zeros((n_q, n_db), dtype=np.float32)
    db_f = fps_db.astype(np.float32)
    db_norm = (db_f * db_f).sum(axis=1)  # (N_db,)
    for start in range(0, n_q, batch):
        end = min(start + batch, n_q)
        q_f = fps_q[start:end].astype(np.float32)
        q_norm = (q_f * q_f).sum(axis=1, keepdims=True)  # (B, 1)
        dot = q_f @ db_f.T  # (B, N_db)
        union = q_norm + db_norm[None, :] - dot
        out[start:end] = np.where(union > 0, dot / union, 0.0)
    return out


def knn_residual_predict(sims, residuals, k, sim_thresh=SIM_THRESH):
    """
    Predict residuals for query compounds using Tanimoto-weighted KNN.
    sims: (N_q, N_db) similarity matrix
    residuals: (N_db,) residuals of training compounds
    Returns (N_q,) predicted residuals.
    """
    n_q = len(sims)
    preds = np.zeros(n_q, dtype=np.float64)
    for i in range(n_q):
        row = sims[i]
        # mask out low-similarity neighbors
        mask = row >= sim_thresh
        if mask.sum() == 0:
            preds[i] = 0.0
            continue
        top_k_idx = np.argsort(-row[mask])[:k]
        top_sims = row[mask][top_k_idx]
        top_res = residuals[mask][top_k_idx]
        # Tanimoto-weighted average
        w = top_sims / top_sims.sum()
        preds[i] = np.dot(w, top_res)
    return preds


def main():
    print("=== nb117: KNN Residual Correction on nb109 ===\n")

    tr = load_train()
    te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # Load nb109 OOF and test predictions
    oof_109_path = DATA_PROCESSED / "oof_nb109_deep_meta_stack.npy"
    te_109_path  = DATA_PROCESSED / "te_nb109_deep_meta_stack.npy"
    if not oof_109_path.exists():
        print("ERROR: nb109 OOF not found. Run fix_nb109_final.py first.")
        return
    oof_109 = np.load(oof_109_path).astype(np.float64)
    te_109  = np.load(te_109_path).astype(np.float64)
    print(f"nb109 OOF RAE (baseline): {rae(y_tr, oof_109):.4f}")

    # Compute ECFP4 fingerprints
    print("Computing ECFP4 fingerprints...")
    fps_tr = morgan_fp_batch(tr["smiles"].tolist())  # (N_tr, 2048) uint8
    fps_te = morgan_fp_batch(te["smiles"].tolist())  # (N_te, 2048) uint8
    print(f"  Train FPs: {fps_tr.shape}  Test FPs: {fps_te.shape}")

    # Scaffold CV: compute OOF residuals with KNN correction
    residuals_109 = y_tr - oof_109  # OOF residuals from nb109
    print(f"\nnb109 OOF residual stats: mean={residuals_109.mean():.4f}  "
          f"std={residuals_109.std():.4f}")

    # Sweep over k values
    best_k, best_rae, best_alpha = 5, rae(y_tr, oof_109), 0.0
    oof_corrections = {}

    print("\nScaffold CV — sweeping k values:")
    for k in K_NEIGHBORS:
        oof_corr = np.zeros(n_tr, dtype=np.float64)
        for fold, (tr_idx, va_idx) in enumerate(splits):
            # Compute Tanimoto between val and train-fold compounds
            sims_va = tanimoto_matrix(fps_tr[va_idx], fps_tr[tr_idx], batch=128)
            # Use nb109 OOF residuals on train-fold as the correction targets
            fold_residuals = residuals_109[tr_idx]
            oof_corr[va_idx] = knn_residual_predict(sims_va, fold_residuals, k)

        # Sweep alpha: corrected = oof_109 + alpha * correction
        best_alpha_k, best_rae_k = 0.0, rae(y_tr, oof_109)
        for alpha in np.arange(0.0, 1.01, 0.05):
            corrected = oof_109 + alpha * oof_corr
            r = rae(y_tr, corrected)
            if r < best_rae_k:
                best_rae_k, best_alpha_k = r, alpha

        print(f"  k={k:2d}  best_alpha={best_alpha_k:.2f}  RAE={best_rae_k:.4f}")
        oof_corrections[k] = (oof_corr, best_alpha_k, best_rae_k)
        if best_rae_k < best_rae:
            best_rae, best_k, best_alpha = best_rae_k, k, best_alpha_k

    print(f"\nBest: k={best_k}  alpha={best_alpha:.2f}  OOF RAE={best_rae:.4f}")

    # Apply best correction to test predictions
    best_corr, best_alph, _ = oof_corrections[best_k]
    oof_final = oof_109 + best_alph * best_corr

    # Compute test correction using ALL training compounds as the KNN pool
    print(f"\nComputing test residual correction (k={best_k})...")
    sims_te = tanimoto_matrix(fps_te, fps_tr, batch=64)
    te_corr = knn_residual_predict(sims_te, residuals_109, best_k)
    te_corr = np.clip(te_corr, -MAX_CORRECTION, MAX_CORRECTION)
    te_final = te_109 + best_alph * te_corr

    print(f"Test correction: mean={te_corr.mean():.4f}  std={te_corr.std():.4f}")
    print(f"Test final: min={te_final.min():.2f}  med={np.median(te_final):.2f}  "
          f"max={te_final.max():.2f}  std={te_final.std():.3f}")

    # Also try alpha sweep on full training residuals for test
    print("\nAlpha sweep analysis on training residuals applied to test:")
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.5]:
        te_a = te_109 + alpha * te_corr
        print(f"  alpha={alpha:.1f}  test_std={te_a.std():.3f}  test_max={te_a.max():.2f}")

    te_final = np.clip(te_final, y_tr.min() - 0.5, y_tr.max() + 0.5)

    # Save
    np.save(DATA_PROCESSED / "oof_nb117_knn_residual.npy", oof_final)
    np.save(DATA_PROCESSED / "te_nb117_knn_residual.npy", te_final)

    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_final})
    sub.to_csv(SUBMISSIONS / "117_knn_residual_correction.csv", index=False)
    print(f"\nSaved: submissions/117_knn_residual_correction.csv")
    print(f"OOF RAE (corrected): {rae(y_tr, oof_final):.4f}  "
          f"(baseline: {rae(y_tr, oof_109):.4f})")


if __name__ == "__main__":
    main()
