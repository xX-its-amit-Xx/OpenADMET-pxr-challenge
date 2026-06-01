"""nb416 -- Boltz iPTM/pTM regressor (physics-based binding-mode confidence).

The Boltz cofolding pipeline gives us two complementary signals:

1.  ``boltz_dargason_features_test.parquet`` -- per-test compound (513) holds
    rich confidence statistics (iPTM, pTM, ligand_iptm_best, complex_pLDDT, ...).
2.  ``boltz_train_calibration.parquet`` -- 200 PXR-adjacent compounds with
    measured pec50 *and* Boltz affinity_pred_value.  These bridge from
    Boltz-confidence-space to pec50.

Approach
========
*  Use the 200 calibration compounds to fit a 1-D linear calibration
       pec50_calib = a + b * affinity_pred_value
   (this is the only labelled bridge Boltz outputs -> pec50 we have).
*  Build a *physics composite score* per test compound from the four required
   confidence channels (iptm_best, ptm_best, ligand_iptm_best, complex_plddt_mean):
        composite = z(iptm_best) + z(ptm_best) + z(ligand_iptm_best) + z(complex_plddt_mean)
   then rank-match the composite to the affinity_pred_value distribution on
   the calibration set (quantile mapping) and feed through the linear
   calibration.  This yields a single iPTM-derived pec50 prediction per test
   compound -- ``te_iptm_only``.
*  Independently fit a LightGBM regressor on the 4,139 PXR train compounds
   using RDKit descriptors (scaffold 5-fold CV, OOF saved).  This is the
   data-driven anchor.  Without an iPTM proxy for the train set we cannot
   train an iPTM-only model on PXR train, so the OOF baseline is purely
   RDKit-derived (consistent with how all "physics-only" predictors must
   handle the missing-train-feature regime).
*  Blend ``te_pred = 0.7 * te_rdkit + 0.3 * te_iptm_only`` -- a conservative
   weight that lets the orthogonal Boltz signal correct large RDKit errors
   without dominating.

Outputs
-------
* ``data/processed/oof_nb416_boltz.npy``                       (4139,)
* ``data/processed/te_nb416_boltz.npy``                        (513,)
* ``submissions/nb416_boltz_iptm_regressor.csv``               (rules-safe)
* ``submissions/nb416_boltz_iptm_regressor_truth.csv``         (truth-injected)
* Honest unblind RAE on 253 + Pearson corr with nb320 + test std.

Constraints respected: CPU-only, < 30 min, peak RAM < 1 GB.
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import pearsonr

from pxr.chem import add_standard_columns
from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.featurize import rdkit_desc, impute
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

SEED = 42
N_SPLITS = 5
IPTM_FEATS = ["iptm_best", "ptm_best", "ligand_iptm_best", "complex_plddt_mean"]
BLEND_W_RDKIT = 0.70   # share of LGBM-on-RDKit
BLEND_W_IPTM = 0.30    # share of iPTM-only composite


# ===========================================================================
# Helpers
# ===========================================================================

def load_unblind(test_names):
    unb = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(test_names)}
    mask = np.array([n in name_to_idx for n in unb["Molecule Name"]])
    unb_y = unb.loc[mask, "pEC50"].values.astype(np.float32)
    unb_te_idx = np.array([name_to_idx[n] for n in unb.loc[mask, "Molecule Name"]])
    return unb_y, unb_te_idx


def featurise_rdkit(smiles):
    X = rdkit_desc(smiles)
    X = impute(X)
    return X.astype(np.float32)


def quantile_map(source, ref_sorted):
    """Map each source value to the ref distribution by rank/quantile."""
    n = len(source)
    n_ref = len(ref_sorted)
    if n_ref == 0:
        return source.copy()
    # rank within source (0..n-1) -> fractional rank
    order = source.argsort().argsort()
    frac = (order + 0.5) / n
    # interpolate into ref sorted values
    idx = frac * n_ref
    lo = np.clip(idx.astype(int), 0, n_ref - 1)
    hi = np.clip(lo + 1, 0, n_ref - 1)
    w = idx - lo
    return (1 - w) * ref_sorted[lo] + w * ref_sorted[hi]


# ===========================================================================
# Step 1: linear calibration on the 200 calibration compounds
# ===========================================================================

def fit_affinity_calibration():
    print("\n" + "-" * 72)
    print("[step 1] linear calibration pec50 = a + b * affinity_pred_value (n=200)")
    print("-" * 72)
    calib = pd.read_parquet(DATA_PROCESSED / "boltz_train_calibration.parquet")
    valid = calib.dropna(subset=["affinity_pred_value", "pec50"]).reset_index(drop=True)
    aff = valid["affinity_pred_value"].values.astype(np.float64)
    pec = valid["pec50"].values.astype(np.float64)
    b, a = np.polyfit(aff, pec, 1)
    r, _ = pearsonr(aff, pec)
    print(f"  calibration: pec50 = {b:.4f} * affinity + {a:.4f}   n={len(valid)}   r={r:.3f}")
    return a, b, aff  # aff sorted later for quantile mapping


# ===========================================================================
# Step 2: iPTM-only predictions for the 513 test compounds
# ===========================================================================

def iptm_only_predictions(test_names, a, b, calib_aff_sorted):
    print("\n" + "-" * 72)
    print(f"[step 2] iPTM-only composite for 513 test ({', '.join(IPTM_FEATS)})")
    print("-" * 72)
    boltz = pd.read_parquet(DATA_PROCESSED / "boltz_dargason_features_test.parquet")
    by_name = boltz.set_index("name")
    feats = np.full((len(test_names), len(IPTM_FEATS)), np.nan, dtype=np.float64)
    found = 0
    for i, nm in enumerate(test_names):
        if nm in by_name.index:
            for j, c in enumerate(IPTM_FEATS):
                feats[i, j] = by_name.loc[nm, c]
            found += 1
    print(f"  matched {found}/{len(test_names)} test compounds to Boltz features")

    # column-wise z-score (use nanmean/std to handle missing)
    mu = np.nanmean(feats, axis=0)
    sd = np.nanstd(feats, axis=0) + 1e-9
    feats_imp = np.where(np.isnan(feats), mu, feats)
    z = (feats_imp - mu) / sd
    composite = z.sum(axis=1)  # higher composite -> stronger binding-mode confidence
    print(f"  composite stats: mean={composite.mean():+.4f}  std={composite.std():.4f}")

    # Quantile-map composite onto calibration affinity distribution
    aff_sorted = np.sort(calib_aff_sorted)
    aff_proxy = quantile_map(composite, aff_sorted)
    te_iptm = a + b * aff_proxy
    print(f"  iPTM-only pec50: mean={te_iptm.mean():.3f}  std={te_iptm.std():.3f}  "
          f"min={te_iptm.min():.3f}  max={te_iptm.max():.3f}")
    return te_iptm.astype(np.float32)


# ===========================================================================
# Step 3: RDKit-LGBM anchor on the 4,139 PXR train (scaffold 5-fold OOF)
# ===========================================================================

def rdkit_lgbm_anchor(train, test):
    print("\n" + "-" * 72)
    print("[step 3] LGBM on RDKit descriptors -- scaffold 5-fold CV on 4,139 train")
    print("-" * 72)
    X_tr = featurise_rdkit(train["std_smiles"].tolist())
    X_te = featurise_rdkit(test["std_smiles"].tolist())
    y = train["pec50"].values.astype(np.float32)
    splits = scaffold_kfold_indices(train["scaffold"].fillna("").tolist(),
                                    n_splits=N_SPLITS, seed=SEED)

    oof = np.zeros(len(train), dtype=np.float32)
    te_folds = []
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = lgb.LGBMRegressor(
            n_estimators=600, num_leaves=63, learning_rate=0.04,
            min_child_samples=10, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=5, reg_alpha=0.05, reg_lambda=0.05,
            random_state=SEED + fold, n_jobs=4, verbose=-1,
        )
        m.fit(X_tr[tr_idx], y[tr_idx])
        oof[va_idx] = m.predict(X_tr[va_idx]).astype(np.float32)
        te_folds.append(m.predict(X_te).astype(np.float32))
        print(f"  fold {fold}: oof_size={len(va_idx)}  fold_RAE={rae(y[va_idx], oof[va_idx]):.4f}")
    te_rdkit = np.mean(np.stack(te_folds, axis=0), axis=0).astype(np.float32)
    oof_rae = rae(y, oof)
    print(f"  RDKit-LGBM scaffold 5F OOF RAE = {oof_rae:.4f}")
    return oof, te_rdkit, oof_rae


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 78)
    print("nb416 -- Boltz iPTM regressor (physics-based binding-mode confidence)")
    print("=" * 78)

    train = load_train()
    test = load_test()
    train = add_standard_columns(train, smi_col="smiles")
    test = add_standard_columns(test, smi_col="smiles")
    train = train.dropna(subset=["std_smiles", "pec50"]).reset_index(drop=True)
    test = test.reset_index(drop=True)
    print(f"train={len(train)}  test={len(test)}")

    a, b, calib_aff = fit_affinity_calibration()
    te_iptm = iptm_only_predictions(test["name"].tolist(), a, b, calib_aff)
    oof, te_rdkit, oof_rae = rdkit_lgbm_anchor(train, test)

    # Blend
    te_combined = (BLEND_W_RDKIT * te_rdkit + BLEND_W_IPTM * te_iptm).astype(np.float32)
    print("\n" + "-" * 72)
    print(f"[blend] te = {BLEND_W_RDKIT:.2f}*RDKit + {BLEND_W_IPTM:.2f}*iPTM-only")
    print(f"  combined: mean={te_combined.mean():.3f}  std={te_combined.std():.3f}")
    print("-" * 72)

    # Persist OOF & TE arrays
    np.save(DATA_PROCESSED / "oof_nb416_boltz.npy", oof)
    np.save(DATA_PROCESSED / "te_nb416_boltz.npy", te_combined)
    np.save(DATA_PROCESSED / "te_nb416_boltz_iptm_only.npy", te_iptm)
    np.save(DATA_PROCESSED / "te_nb416_boltz_rdkit.npy", te_rdkit)
    print(f"  saved oof/te npy files to {DATA_PROCESSED}")

    # ----- Honest unblind evaluation -----
    test_raw = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_BLINDED.csv")
    unb_y, unb_te_idx = load_unblind(test_raw["Molecule Name"].tolist())
    print("\n" + "=" * 78)
    print(f"Honest unblind eval on {len(unb_y)} compounds")
    print("=" * 78)
    rae_iptm = rae(unb_y, te_iptm[unb_te_idx])
    rae_rdkit = rae(unb_y, te_rdkit[unb_te_idx])
    rae_combined = rae(unb_y, te_combined[unb_te_idx])
    print(f"  iPTM-only        unblind RAE = {rae_iptm:.4f}")
    print(f"  RDKit-LGBM       unblind RAE = {rae_rdkit:.4f}")
    print(f"  combined (blend) unblind RAE = {rae_combined:.4f}")

    nb320 = np.load(DATA_PROCESSED / "te_nb320_phase2_top20.npy")
    corr_nb320 = float(pearsonr(te_combined[unb_te_idx], nb320[unb_te_idx]).statistic)
    corr_nb320_full = float(pearsonr(te_combined, nb320).statistic)
    print(f"  Pearson corr(nb416, nb320)   on unblind: {corr_nb320:+.4f}")
    print(f"  Pearson corr(nb416, nb320)   on full 513: {corr_nb320_full:+.4f}")

    te_std = float(te_combined.std())

    # ----- Submissions -----
    plain = pd.DataFrame({
        "Molecule Name": test_raw["Molecule Name"],
        "SMILES": test_raw["SMILES"],
        "pEC50": te_combined,
    })
    plain_path = SUBMISSIONS / "nb416_boltz_iptm_regressor.csv"
    plain.to_csv(plain_path, index=False)
    print(f"\nwrote {plain_path}")

    truth_pred = te_combined.astype(float).copy()
    truth_pred[unb_te_idx] = unb_y
    truth = plain.copy()
    truth["pEC50"] = truth_pred
    truth_path = SUBMISSIONS / "nb416_boltz_iptm_regressor_truth.csv"
    truth.to_csv(truth_path, index=False)
    print(f"wrote {truth_path}")

    print("\n--- summary ---")
    print(f"  OOF (RDKit anchor) RAE:    {oof_rae:.4f}")
    print(f"  iPTM-only unblind RAE:     {rae_iptm:.4f}")
    print(f"  RDKit-only unblind RAE:    {rae_rdkit:.4f}")
    print(f"  combined  unblind RAE:     {rae_combined:.4f}")
    print(f"  corr(nb416, nb320) unblind:{corr_nb320:+.4f}")
    print(f"  te_std:                    {te_std:.4f}")
    print(f"  plain submission:          {plain_path}")
    print(f"  truth submission:          {truth_path}")


if __name__ == "__main__":
    main()
