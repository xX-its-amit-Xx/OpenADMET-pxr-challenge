"""nb186 -- Check if test compounds appear in single-concentration (primary screen) data.

Competition structure:
  - 4139 training compounds: dose-response (CRC) with pEC50
  - 513 test compounds: dose-response BUT labels blinded
  - 21003 single-conc: primary screen (log2FC, FDR, concentration_M)

If the primary screen was run before dose-response confirmation, all CRC compounds
(training AND test) would have primary screen data. If test compounds are in the
single-conc file, we get direct biological signal (log2FC) for each test compound.

This script:
  1. Load test SMILES, standardize, compute InChIKey
  2. Load single-conc SMILES, standardize, compute InChIKey
  3. Check overlap and report statistics
  4. If overlap exists: train pEC50~log2FC+structure calibration model
  5. Use calibration to predict test compound pEC50
  6. Blend with structure-only predictions (nb183/nb184)
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge

from pxr.data import load_train, load_test, load_single_conc
from pxr.chem import standardize, morgan_fp_batch, compute_physchem
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS
from scipy.optimize import minimize

try:
    from rdkit.Chem.inchi import MolToInchiKey
    from rdkit.Chem import MolFromSmiles
    def smiles_to_inchikey(smi):
        try:
            mol = MolFromSmiles(smi)
            if mol is None: return None
            return MolToInchiKey(mol)
        except:
            return None
except:
    def smiles_to_inchikey(smi):
        return None

COLLAPSE_THRESH = 0.58
SEED = 42


def main():
    print("=== nb186: Single-conc test compound lookup ===\n")

    tr = load_train(); te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr); n_te = len(te_df)
    print(f"Train: {n_tr} compounds, Test: {n_te} compounds")

    # Load single-conc data
    try:
        sc = load_single_conc()
        print(f"Single-conc: {len(sc)} rows")
        print(f"Single-conc columns: {list(sc.columns)}")
    except Exception as e:
        print(f"Error loading single-conc: {e}")
        return

    # Standardize SMILES for matching
    print("\nStandardizing SMILES for overlap check...")

    # Test SMILES
    te_smiles_raw = te_df["smiles"].tolist()
    te_std = [standardize(s) for s in te_smiles_raw]
    te_std_set = set(s for s in te_std if s)
    print(f"  Test std SMILES: {len(te_std_set)} unique")

    # Training SMILES (for reference)
    tr_smiles_raw = tr["smiles"].tolist()
    tr_std = [standardize(s) for s in tr_smiles_raw]
    tr_std_set = set(s for s in tr_std if s)

    # Single-conc SMILES
    sc_smiles_col = None
    for col in ["smiles", "SMILES", "std_smiles", "canonical_smiles"]:
        if col in sc.columns:
            sc_smiles_col = col; break
    if sc_smiles_col is None:
        print("ERROR: No SMILES column found in single-conc data!")
        print(f"Available columns: {list(sc.columns)}")
        return

    print(f"  Single-conc SMILES column: {sc_smiles_col}")
    sc_smiles_raw = sc[sc_smiles_col].tolist()
    sc_std = [standardize(s) if isinstance(s, str) else None for s in sc_smiles_raw]
    sc_std_series = pd.Series(sc_std, index=sc.index)
    sc_std_set = set(s for s in sc_std if s)
    print(f"  Single-conc std SMILES: {len(sc_std_set)} unique")

    # Check overlaps
    te_in_sc = te_std_set & sc_std_set
    tr_in_sc = tr_std_set & sc_std_set
    te_in_tr = te_std_set & tr_std_set

    print(f"\n=== Overlap Statistics ===")
    print(f"  Test in single-conc:    {len(te_in_sc)}/{len(te_std_set)} ({100*len(te_in_sc)/len(te_std_set):.1f}%)")
    print(f"  Train in single-conc:   {len(tr_in_sc)}/{len(tr_std_set)} ({100*len(tr_in_sc)/len(tr_std_set):.1f}%)")
    print(f"  Test in train:          {len(te_in_tr)}/{len(te_std_set)} ({100*len(te_in_tr)/len(te_std_set):.1f}%)")

    if len(te_in_sc) == 0:
        print("\nNo test compounds found in single-conc data.")
        print("Test log2FC signal not available. Cannot use single-conc for test prediction.")

        # Still useful: can we use single-conc train overlap for better training features?
        print(f"\nChecking single-conc features for training compounds...")
        # Find single-conc log2FC for train compounds
        sc['std_smiles_'] = sc_std
        sc_matched = sc[sc['std_smiles_'].isin(tr_std_set)].copy()
        print(f"  Training compounds with single-conc data: {len(sc_matched)}")
        if len(sc_matched) > 0:
            # Show column info
            for col in sc.columns:
                if col not in ['std_smiles_', sc_smiles_col]:
                    n_valid = sc_matched[col].notna().sum()
                    print(f"    {col}: {n_valid} non-null values")
        return

    # If test compounds found in single-conc
    print(f"\n*** FOUND {len(te_in_sc)} TEST COMPOUNDS IN SINGLE-CONC DATA! ***")

    # Build lookup: std_smiles -> single-conc row(s)
    sc['std_smiles_'] = sc_std
    sc_lookup = sc[sc['std_smiles_'].notna()].set_index('std_smiles_')

    # Show single-conc column statistics for matched test compounds
    sc_test_matched = sc[sc['std_smiles_'].isin(te_in_sc)]
    print(f"\nSingle-conc columns for matched test compounds:")
    for col in sc.columns:
        if col in ['std_smiles_', sc_smiles_col]: continue
        n_valid = sc_test_matched[col].notna().sum()
        if n_valid > 0:
            vals = sc_test_matched[col].dropna()
            print(f"  {col}: {n_valid} valid, mean={vals.mean():.3f}, std={vals.std():.3f}")

    # Find log2FC column
    log2fc_col = None
    for col in ["log2fc", "log2FC", "Log2FC", "log2_fc", "fc"]:
        if col in sc.columns:
            log2fc_col = col; break

    if log2fc_col is None:
        print(f"\nERROR: No log2FC column found! Columns: {list(sc.columns)}")
        return

    print(f"\nUsing log2FC column: {log2fc_col}")

    # Build feature for training compounds: single-conc log2FC
    # Match train SMILES to single-conc
    sc_tr_matched = sc[sc['std_smiles_'].isin(tr_std_set)].copy()
    n_tr_with_sc = len(set(sc_tr_matched['std_smiles_']))
    print(f"\nTraining compounds with single-conc: {n_tr_with_sc}/{n_tr}")

    # For training: create SC feature vector
    tr_sc_feature = np.full(n_tr, np.nan)
    std_to_log2fc = {}
    for _, row in sc_tr_matched.iterrows():
        smi = row['std_smiles_']
        if smi and pd.notna(row[log2fc_col]):
            if smi not in std_to_log2fc:
                std_to_log2fc[smi] = []
            std_to_log2fc[smi].append(row[log2fc_col])

    for i, smi in enumerate(tr_std):
        if smi in std_to_log2fc:
            tr_sc_feature[i] = np.mean(std_to_log2fc[smi])

    n_tr_matched = np.sum(np.isfinite(tr_sc_feature))
    print(f"  Training compounds matched: {n_tr_matched}")

    # For test: create SC feature vector
    te_sc_feature = np.full(n_te, np.nan)
    sc_te_matched = sc[sc['std_smiles_'].isin(te_in_sc)].copy()
    std_to_log2fc_te = {}
    for _, row in sc_te_matched.iterrows():
        smi = row['std_smiles_']
        if smi and pd.notna(row[log2fc_col]):
            if smi not in std_to_log2fc_te:
                std_to_log2fc_te[smi] = []
            std_to_log2fc_te[smi].append(row[log2fc_col])

    for i, smi in enumerate(te_std):
        if smi in std_to_log2fc_te:
            te_sc_feature[i] = np.mean(std_to_log2fc_te[smi])

    n_te_matched = np.sum(np.isfinite(te_sc_feature))
    print(f"  Test compounds matched: {n_te_matched}/{n_te}")

    if n_te_matched < 10:
        print("Too few test matches for reliable calibration. Stopping.")
        return

    # Calibration: pEC50 ~ log2FC using training compounds with SC data
    tr_has_sc = np.isfinite(tr_sc_feature)
    X_cal = tr_sc_feature[tr_has_sc].reshape(-1,1)
    y_cal = y_tr[tr_has_sc]
    print(f"\nCalibration data: {len(y_cal)} training compounds with SC+CRC")

    # Simple linear calibration
    from scipy.stats import pearsonr, spearmanr
    r_p, _ = pearsonr(X_cal.flatten(), y_cal)
    r_s, _ = spearmanr(X_cal.flatten(), y_cal)
    print(f"  Pearson r(log2FC, pEC50): {r_p:.3f}")
    print(f"  Spearman r(log2FC, pEC50): {r_s:.3f}")

    # Fit calibration model
    # Use scaffold CV to get unbiased estimate
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, 5, SEED)

    # LGBM with log2FC as sole feature + Morgan FP
    # Build full feature matrix: log2FC + Morgan FP
    morgan_tr = morgan_fp_batch(tr["smiles"].tolist()).astype(np.float32)
    morgan_te = morgan_fp_batch(te_df["smiles"].tolist()).astype(np.float32)

    X_full_tr = np.hstack([tr_sc_feature.reshape(-1,1), morgan_tr])
    X_full_te = np.hstack([te_sc_feature.reshape(-1,1), morgan_te])

    # Impute missing log2FC with median
    sc_median = np.nanmedian(tr_sc_feature)
    X_full_tr[:,0] = np.where(np.isfinite(X_full_tr[:,0]), X_full_tr[:,0], sc_median)
    te_sc_for_impute = te_sc_feature.copy()
    te_sc_for_impute = np.where(np.isfinite(te_sc_for_impute), te_sc_for_impute, sc_median)
    X_full_te[:,0] = te_sc_for_impute

    print(f"\nFull feature matrix: train={X_full_tr.shape}, test={X_full_te.shape}")
    print("Training LGBM with log2FC + Morgan FP...")

    LGBM_PARAMS = dict(
        objective="regression_l1", metric="mae",
        n_estimators=800, num_leaves=64, learning_rate=0.03,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.3,
        reg_alpha=0.05, reg_lambda=1.0, verbose=-1, n_jobs=4,
    )

    oof_sc = np.full(n_tr, np.nan)
    te_sc_preds = []
    for seed in [42, 123, 456]:
        params_s = {**LGBM_PARAMS, "random_state": seed}
        oof_s = np.full(n_tr, np.nan)
        for fold, (tr_idx, va_idx) in enumerate(splits):
            m = lgb.train(params_s, lgb.Dataset(X_full_tr[tr_idx], label=y_tr[tr_idx]),
                          callbacks=[lgb.log_evaluation(-1)])
            oof_s[va_idx] = m.predict(X_full_tr[va_idx])
        m_f = lgb.train(params_s, lgb.Dataset(X_full_tr, label=y_tr),
                        callbacks=[lgb.log_evaluation(-1)])
        te_sc_preds.append(m_f.predict(X_full_te))
        r_s = rae(y_tr, oof_s)
        print(f"  seed {seed}: RAE={r_s:.4f}")
        oof_sc = oof_sc + oof_s / 3 if not np.any(np.isnan(oof_sc)) else oof_s

    te_sc_pred = np.mean(te_sc_preds, axis=0)
    r_sc = rae(y_tr, oof_sc); ratio_sc = te_sc_pred.std()/oof_sc.std()
    flag_sc = "PASS" if ratio_sc >= COLLAPSE_THRESH else "FAIL"
    print(f"\nSC-feature LGBM: RAE={r_sc:.4f}  ratio={ratio_sc:.3f}  [{flag_sc}]")

    # Load nb183 for blending
    oof183_p = DATA_PROCESSED / "oof_nb183_qreg_poly10.npy"
    te183_p  = DATA_PROCESSED / "te_nb183_qreg_poly10.npy"
    if not oof183_p.exists():
        print("nb183 OOF not found, using SC model alone.")
        return

    oof183 = np.load(oof183_p).flatten().astype(np.float64)
    te183  = np.load(te183_p).flatten().astype(np.float64)

    # Blend SC model with nb183
    print("\nBlending SC-LGBM with nb184 (7-model SLSQP, 0.299711):")
    # Load nb184 grand ensemble
    oof184_p = DATA_PROCESSED / "oof_nb184_grand_v16.npy"
    te184_p  = DATA_PROCESSED / "te_nb184_grand_v16.npy"
    if oof184_p.exists():
        oof_ref = np.load(oof184_p).flatten().astype(np.float64)
        te_ref  = np.load(te184_p).flatten().astype(np.float64)
        r_ref   = rae(y_tr, oof_ref)
        ref_label = f"nb184({r_ref:.4f})"
    else:
        oof_ref, te_ref = oof183, te183
        r_ref = rae(y_tr, oof_ref); ref_label = f"nb183({r_ref:.4f})"

    best_blend_r = 1e9; best_blend_oof = best_blend_te = None
    for alpha_sc in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        oof_b = (1-alpha_sc) * oof_ref + alpha_sc * oof_sc
        te_b  = (1-alpha_sc) * te_ref  + alpha_sc * te_sc_pred
        r_b = rae(y_tr, oof_b); ratio_b = te_b.std()/oof_b.std()
        flag = "PASS" if ratio_b >= COLLAPSE_THRESH else "FAIL"
        beat = " ***NEW BEST***" if (ratio_b >= COLLAPSE_THRESH and r_b < 0.299711) else ""
        print(f"  sc_alpha={alpha_sc:.2f}  RAE={r_b:.6f}  ratio={ratio_b:.4f}  [{flag}]{beat}")
        if ratio_b >= COLLAPSE_THRESH and r_b < best_blend_r:
            best_blend_r = r_b; best_blend_oof = oof_b; best_blend_te = te_b

    if best_blend_r < 0.299711:
        print(f"\n*** NEW BEST: {best_blend_r:.6f} ***")
        best_te_clip = np.clip(best_blend_te, y_tr.min()-0.5, y_tr.max()+0.5)
        np.save(DATA_PROCESSED / "oof_nb186_sc_blend.npy", best_blend_oof)
        np.save(DATA_PROCESSED / "te_nb186_sc_blend.npy",  best_te_clip)
        sub = pd.DataFrame({"Molecule Name": te_df["name"].values, "pEC50": best_te_clip})
        sub.to_csv(SUBMISSIONS / "186_sc_blend.csv", index=False)
        print(f"Saved: submissions/186_sc_blend.csv")
    else:
        print(f"\nnb184 (0.299711) still best — SC data does not improve predictions.")


if __name__ == "__main__":
    main()
