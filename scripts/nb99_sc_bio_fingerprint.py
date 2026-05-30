"""nb99 — Single-Concentration Biological Fingerprint.

Instead of using single-conc data as training labels (which failed in nb65
due to pEC50 calibration mismatch), use SC data as a *retrieval index*:
for each compound, find its SC neighbors (Tanimoto >= 0.35) and compute:
  - mean_log2fc     : mean log2FC of SC neighbors (biological signal)
  - frac_sig        : fraction of SC neighbors with FDR < 0.05
  - max_log2fc      : max log2FC (best nearby hit)
  - n_sc_neighbors  : how many SC neighbors exist (coverage score)

These 4 features are a direct assay readout without pEC50 calibration.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from pathlib import Path

from pxr.data import load_train, load_test
from pxr.featurize import combined, impute
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko, morgan_fp_batch
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
N_FOLDS = 5
SC_SIM_THRESH = 0.35   # minimum Tanimoto for SC neighbor
FDR_THRESH = 0.05
LGBM_PARAMS = dict(
    n_estimators=1000, num_leaves=64, learning_rate=0.05,
    min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, random_state=SEED, verbose=-1, n_jobs=4
)


def tanimoto_matrix(fps_query: np.ndarray, fps_ref: np.ndarray) -> np.ndarray:
    """Compute Tanimoto similarity matrix. fps are binary uint8 arrays."""
    fps_q = fps_query.astype(np.float32)
    fps_r = fps_ref.astype(np.float32)
    dot   = fps_q @ fps_r.T
    sq    = fps_q.sum(1)[:, None]
    sr    = fps_r.sum(1)[None, :]
    return dot / np.maximum(sq + sr - dot, 1e-6)


def compute_sc_features(query_fps: np.ndarray,
                        sc_fps: np.ndarray,
                        sc_log2fc: np.ndarray,
                        sc_fdr: np.ndarray,
                        batch_size: int = 256) -> np.ndarray:
    """Compute 4 SC neighborhood features for each query compound."""
    N = len(query_fps)
    feats = np.zeros((N, 4), dtype=np.float32)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        sim_block = tanimoto_matrix(query_fps[start:end], sc_fps)  # (batch, M_sc)
        for local_i, global_i in enumerate(range(start, end)):
            sim_row = sim_block[local_i]
            mask = sim_row >= SC_SIM_THRESH
            n_nbr = mask.sum()
            if n_nbr == 0:
                feats[global_i] = [0.0, 0.0, 0.0, 0.0]
            else:
                nbr_log2fc = sc_log2fc[mask]
                nbr_fdr    = sc_fdr[mask]
                feats[global_i, 0] = float(np.mean(nbr_log2fc))
                feats[global_i, 1] = float(np.mean(nbr_fdr < FDR_THRESH))
                feats[global_i, 2] = float(np.max(nbr_log2fc))
                feats[global_i, 3] = float(np.log1p(n_nbr))

    return feats


def full_metrics(y_true, y_pred, label=""):
    yt = np.asarray(y_true, float); yp = np.asarray(y_pred, float)
    msk = np.isfinite(yt) & np.isfinite(yp); yt, yp = yt[msk], yp[msk]
    mae_v = float(np.mean(np.abs(yt - yp)))
    rae_v = mae_v / float(np.mean(np.abs(yt - yt.mean()))) if yt.std() > 0 else np.nan
    r2    = 1 - np.sum((yt-yp)**2) / np.sum((yt-yt.mean())**2) if yt.std() > 0 else np.nan
    pr, _ = stats.pearsonr(yt, yp); sp, _ = stats.spearmanr(yt, yp)
    if label:
        print(f"  [{label}] RAE={rae_v:.4f}  MAE={mae_v:.4f}  R2={r2:.4f}  "
              f"r={pr:.4f}  rho={sp:.4f}")
    return dict(RAE=rae_v, MAE=mae_v, R2=r2, Pearson=pr, Spearman=sp)


def main():
    print("=== nb99: Single-Concentration Biological Fingerprint ===")
    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # Load single-concentration data
    sc_path = Path("data/raw/pxr-challenge_single_concentration_TRAIN.csv")
    if not sc_path.exists():
        sc_path = Path("../data/raw/pxr-challenge_single_concentration_TRAIN.csv")
    sc = pd.read_csv(sc_path)
    print(f"Single-conc data: {len(sc)} rows, columns: {sc.columns.tolist()}")

    # Identify columns
    smiles_col  = next(c for c in sc.columns if "smiles" in c.lower())
    log2fc_col  = next(c for c in sc.columns if "log2" in c.lower())
    fdr_col     = next(c for c in sc.columns if "fdr" in c.lower())

    sc_clean = sc[[smiles_col, log2fc_col, fdr_col]].dropna()
    sc_log2fc = sc_clean[log2fc_col].values.astype(np.float32)
    sc_fdr    = sc_clean[fdr_col].values.astype(np.float32)
    print(f"SC clean: {len(sc_clean)} rows")
    print(f"  log2FC: mean={sc_log2fc.mean():.3f}  std={sc_log2fc.std():.3f}")
    print(f"  FDR<0.05: {(sc_fdr < FDR_THRESH).mean()*100:.1f}%")

    print("Computing Morgan FPs for SC compounds...")
    sc_fps = morgan_fp_batch(sc_clean[smiles_col].tolist()).astype(np.float32)

    print("Computing Morgan FPs for train/test...")
    fps_tr = morgan_fp_batch(tr["smiles"].tolist()).astype(np.float32)
    fps_te = morgan_fp_batch(te["smiles"].tolist()).astype(np.float32)

    print("Computing SC neighborhood features for train set...")
    SC_tr = compute_sc_features(fps_tr, sc_fps, sc_log2fc, sc_fdr)
    print(f"  Coverage: {(SC_tr[:, 3] > 0).mean()*100:.1f}% of train have >=1 SC neighbor")

    print("Computing SC neighborhood features for test set...")
    SC_te = compute_sc_features(fps_te, sc_fps, sc_log2fc, sc_fdr)
    print(f"  Coverage: {(SC_te[:, 3] > 0).mean()*100:.1f}% of test have >=1 SC neighbor")

    print("Computing standard combined features...")
    X_tr_base = impute(combined(tr["smiles"].tolist()))
    X_te_base = impute(combined(te["smiles"].tolist()))

    X_tr = np.hstack([X_tr_base, SC_tr])
    X_te = np.hstack([X_te_base, SC_te])
    print(f"Feature shape: train={X_tr.shape}  test={X_te.shape}")

    print("\n=== Scaffold 5-fold CV ===")
    oof = np.full(len(y_tr), np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = lgb.train(
            LGBM_PARAMS,
            lgb.Dataset(X_tr[tr_idx], label=y_tr[tr_idx]),
            valid_sets=[lgb.Dataset(X_tr[va_idx], label=y_tr[va_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
        )
        oof[va_idx] = m.predict(X_tr[va_idx])
        print(f"  fold {fold+1}  RAE={rae(y_tr[va_idx], oof[va_idx]):.4f}", flush=True)

    full_metrics(y_tr, oof, "nb99_sc_bio_fp")

    print("\nTraining final model...")
    m_final = lgb.train(LGBM_PARAMS, lgb.Dataset(X_tr, label=y_tr),
                        callbacks=[lgb.log_evaluation(-1)])
    te_preds = np.clip(m_final.predict(X_te), y_tr.min() - 0.5, y_tr.max() + 0.5)

    np.save(DATA_PROCESSED / "oof_nb99_sc_bio_fp.npy", oof)
    np.save(DATA_PROCESSED / "te_nb99_sc_bio_fp.npy", te_preds)

    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_preds})
    out = SUBMISSIONS / "99_sc_bio_fingerprint.csv"
    sub.to_csv(out, index=False)
    print(f"Saved: {out}")
    print(f"Test  min={te_preds.min():.2f}  med={np.median(te_preds):.2f}  max={te_preds.max():.2f}")


if __name__ == "__main__":
    main()
