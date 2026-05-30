"""nb98 — ChEMBL Bayesian MACCS Fingerprint for PXR.

Mines the 812 ChEMBL PXR compounds to compute per-MACCS-key
Naive Bayes P(active | bit). Creates a 166-dim "PXR preference vector"
per compound that encodes external biological knowledge without
directly using external pEC50 labels (avoids calibration noise).

Active threshold: pChEMBL >= 5.0 (10 µM).
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
from rdkit import Chem
from rdkit.Chem import MACCSkeys

from pxr.data import load_train, load_test
from pxr.featurize import combined, impute
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, DATA_EXTERNAL, SUBMISSIONS

SEED = 42
N_FOLDS = 5
LGBM_PARAMS = dict(
    n_estimators=1000, num_leaves=64, learning_rate=0.05,
    min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, random_state=SEED, verbose=-1, n_jobs=4
)
ACTIVE_THRESH = 5.0   # pChEMBL >= 5 → active (~10 µM)
LAPLACE_K = 1         # Laplace smoothing


def smiles_to_maccs(smiles_list: list) -> np.ndarray:
    """(N, 167) binary MACCS keys; NaN row on failure."""
    rows = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is None:
                rows.append(np.full(167, np.nan))
            else:
                rows.append(np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32))
        except Exception:
            rows.append(np.full(167, np.nan))
    return np.array(rows, dtype=np.float32)


def build_bayesian_fp(chembl_smiles, chembl_labels, query_smiles):
    """
    Compute log-odds Naive Bayes scores per MACCS bit.
    Returns (N_query, 167) matrix of log-odds per bit.
    """
    print(f"  Building Bayesian model: {len(chembl_smiles)} ChEMBL compounds, "
          f"{chembl_labels.sum():.0f} active / {(~chembl_labels).sum():.0f} inactive")

    X_chembl = smiles_to_maccs(chembl_smiles)
    valid = ~np.isnan(X_chembl).any(axis=1)
    X_chembl = X_chembl[valid]
    labels = chembl_labels[valid]

    n_active   = labels.sum()
    n_inactive = (~labels).sum()

    # P(bit=1 | active) and P(bit=1 | inactive) with Laplace smoothing
    p_bit_active   = (X_chembl[labels].sum(0)  + LAPLACE_K) / (n_active   + 2*LAPLACE_K)
    p_bit_inactive = (X_chembl[~labels].sum(0) + LAPLACE_K) / (n_inactive + 2*LAPLACE_K)

    # Log-odds ratio per bit: log[ P(bit|active)/P(bit|inactive) ]
    log_odds = np.log(p_bit_active / p_bit_inactive)  # shape (167,)

    # For each query compound: weighted sum of log-odds over present bits
    X_query = smiles_to_maccs(query_smiles)
    # Replace NaN with 0 (absent bit)
    X_query = np.where(np.isnan(X_query), 0.0, X_query)

    # Full per-bit feature matrix (N_query, 167) — let LGBM learn which bits matter
    log_odds_matrix = X_query * log_odds[None, :]  # (N, 167)

    # Also compute aggregate score as extra feature
    nb_score = log_odds_matrix.sum(axis=1, keepdims=True)  # (N, 1)

    return np.hstack([log_odds_matrix, nb_score]).astype(np.float32)  # (N, 168)


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
    print("=== nb98: ChEMBL Bayesian MACCS Fingerprint ===")
    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # Load ChEMBL PXR data
    chembl_path = DATA_EXTERNAL / "chembl_pxr_all_types.parquet"
    if not chembl_path.exists():
        print(f"ERROR: {chembl_path} not found. Run nb37/nb63 first.")
        sys.exit(1)

    chembl = pd.read_parquet(chembl_path)
    print(f"ChEMBL PXR data: {len(chembl)} rows")
    print(f"Columns: {chembl.columns.tolist()}")

    # Find pChEMBL / pEC50 column
    pchembl_col = next((c for c in chembl.columns
                        if any(k in c.lower() for k in ("pchembl", "pec50", "pic50"))), None)
    smiles_col  = next((c for c in chembl.columns
                        if c.lower() in ("smiles", "canonical_smiles", "std_smiles")), None)
    if pchembl_col is None or smiles_col is None:
        print(f"Cannot find pChEMBL ({pchembl_col}) or SMILES ({smiles_col}) column")
        print(chembl.head(2).to_string())
        sys.exit(1)

    chembl_clean = chembl[[smiles_col, pchembl_col]].dropna()
    chembl_smiles = chembl_clean[smiles_col].tolist()
    chembl_active = (chembl_clean[pchembl_col].values >= ACTIVE_THRESH)
    print(f"After dropna: {len(chembl_smiles)} rows, {chembl_active.sum()} active")

    # Build Bayesian FP for all compounds (uses ALL ChEMBL data — no leakage since it's external)
    print("Building Bayesian fingerprints...")
    BFP_tr = build_bayesian_fp(chembl_smiles, chembl_active, tr["smiles"].tolist())
    BFP_te = build_bayesian_fp(chembl_smiles, chembl_active, te["smiles"].tolist())
    print(f"Bayesian FP shape: train={BFP_tr.shape}  test={BFP_te.shape}")

    print("Computing standard combined features...")
    X_tr_base = impute(combined(tr["smiles"].tolist()))
    X_te_base = impute(combined(te["smiles"].tolist()))

    # Impute Bayesian FP (fill NaN with 0)
    BFP_tr = np.where(np.isnan(BFP_tr), 0.0, BFP_tr).astype(np.float32)
    BFP_te = np.where(np.isnan(BFP_te), 0.0, BFP_te).astype(np.float32)

    X_tr = np.hstack([X_tr_base, BFP_tr])
    X_te = np.hstack([X_te_base, BFP_te])
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

    full_metrics(y_tr, oof, "nb98_chembl_bayesian_fp")

    print("\nTraining final model...")
    m_final = lgb.train(LGBM_PARAMS, lgb.Dataset(X_tr, label=y_tr),
                        callbacks=[lgb.log_evaluation(-1)])
    te_preds = np.clip(m_final.predict(X_te), y_tr.min() - 0.5, y_tr.max() + 0.5)

    np.save(DATA_PROCESSED / "oof_nb98_chembl_bayesian_fp.npy", oof)
    np.save(DATA_PROCESSED / "te_nb98_chembl_bayesian_fp.npy", te_preds)

    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_preds})
    out = SUBMISSIONS / "98_chembl_bayesian_fp.csv"
    sub.to_csv(out, index=False)
    print(f"Saved: {out}")
    print(f"Test  min={te_preds.min():.2f}  med={np.median(te_preds):.2f}  max={te_preds.max():.2f}")


if __name__ == "__main__":
    main()
