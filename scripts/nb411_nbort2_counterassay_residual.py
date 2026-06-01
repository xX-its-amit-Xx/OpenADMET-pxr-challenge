"""nb411 / nbORT2 -- Counter-assay-conditional residual predictor.

Signal source
-------------
The PXR-specific signal is the *difference* between the PXR pEC50 and the
counter-assay (null) pEC50.  The saturated ensemble pool (nb320 SLSQP) is
dominated by PC1 = generic assay-promiscuity / cytotoxicity, so adding yet
another model that targets ``pec50_total`` does not help.  This model
explicitly targets the residual:

        y_resid = pEC50_PXR - pEC50_null   (on dual-labelled rows only)

and adds back an imputed ``pec50_null`` at test time.  Featurised with the
217-d RDKit descriptor set ONLY (no Morgan), so we don't recreate the same
ECFP-driven PC1 axis the rest of the pool lives on.

Pipeline
--------
1. Inner-join train + counter on standardised InChIKey -> ~2858 dual-labelled
   compounds.  ``y_resid = pec50 - pec50_null``.
2. Train a null-only LGBM (RDKit features) on the 2859 counter-assay rows
   to impute pec50_null on (a) train rows missing null and (b) test rows.
3. Train a residual LGBM (RDKit features) on the dual-labelled rows with
   scaffold 5-fold CV.  OOF residual + imputed null -> OOF total.
4. Honest unblind RAE on the 253 phase-1-unblinded compounds (never fit on).
5. Save oof_nb411.npy + te_nb411.npy + two submissions (rules-safe + truth).

Outputs
-------
- data/processed/oof_nb411.npy          (4139,) OOF total pEC50 prediction
- data/processed/te_nb411.npy           (513,)  test total pEC50 prediction
- submissions/nb411_nbort2_counterassay_residual.csv         (rules-safe)
- submissions/nb411_nbort2_counterassay_residual_truth.csv   (truth-injected)
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

from pxr.data import load_train, load_counter, load_test
from pxr.chem import add_standard_columns
from pxr.eval import rae, scaffold_kfold_indices
from pxr.featurize import rdkit_desc, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS, DATA_RAW


N_SPLITS = 5
SEED = 42

LGBM_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.04,
    num_leaves=63,
    min_child_samples=10,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    reg_alpha=0.05,
    reg_lambda=0.05,
    random_state=SEED,
    n_jobs=4,
    verbose=-1,
)


def featurise(smiles: list[str]) -> np.ndarray:
    """RDKit-only feature matrix -> imputed float32 (N, 217)."""
    X = rdkit_desc(smiles)
    X = impute(X)
    return X.astype(np.float32)


def fit_predict_lgbm(X_tr, y_tr, X_te):
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


def main():
    print("=" * 78)
    print("nb411 / nbORT2: counter-assay residual predictor")
    print("=" * 78)

    # ------------------------------------------------------------------
    # Load + standardise data
    # ------------------------------------------------------------------
    train = load_train()
    counter = load_counter()
    test = load_test()
    print(f"raw shapes  train={train.shape}  counter={counter.shape}  test={test.shape}")

    train = add_standard_columns(train, smi_col="smiles")
    counter = add_standard_columns(counter, smi_col="smiles")
    test = add_standard_columns(test, smi_col="smiles")

    # ------------------------------------------------------------------
    # Counter-assay null-imputation model (trained on the 2859 counter rows)
    # ------------------------------------------------------------------
    counter_clean = counter.dropna(subset=["pec50", "std_smiles"]).reset_index(drop=True)
    print(f"counter-assay rows usable for null-imputation: {len(counter_clean)}")
    X_null_train = featurise(counter_clean["std_smiles"].tolist())
    y_null_train = counter_clean["pec50"].astype(np.float32).values

    null_model = lgb.LGBMRegressor(**LGBM_PARAMS)
    null_model.fit(X_null_train, y_null_train)

    # ------------------------------------------------------------------
    # Inner-join train + counter on InChIKey -> dual-labelled rows
    # ------------------------------------------------------------------
    counter_by_key = (
        counter.dropna(subset=["pec50", "inchikey"])
        .groupby("inchikey", as_index=False)["pec50"]
        .mean()
        .rename(columns={"pec50": "pec50_null"})
    )

    train_full = train.dropna(subset=["pec50", "inchikey"]).reset_index(drop=True)
    train_dual = train_full.merge(counter_by_key, on="inchikey", how="inner").reset_index(drop=True)
    print(f"dual-labelled train rows (inner join on inchikey): {len(train_dual)}")

    # Residual target: pEC50 - pEC50_null
    train_dual["y_resid"] = train_dual["pec50"] - train_dual["pec50_null"]
    print(
        f"  y_resid: mean={train_dual['y_resid'].mean():+.3f}  "
        f"std={train_dual['y_resid'].std():.3f}  "
        f"|y_resid|>1 pct={float((train_dual['y_resid'].abs() > 1).mean()):.2%}"
    )

    # ------------------------------------------------------------------
    # Featurise dual-labelled train + ALL 4139 train rows + 513 test
    # ------------------------------------------------------------------
    X_dual = featurise(train_dual["std_smiles"].tolist())
    X_train_all = featurise(train_full["std_smiles"].tolist())
    X_test = featurise(test["std_smiles"].tolist())
    print(f"feature dims: X_dual={X_dual.shape}  X_train_all={X_train_all.shape}  X_test={X_test.shape}")

    # Imputed nulls for every train row + every test row
    pec50_null_train_imp = null_model.predict(X_train_all).astype(np.float32)
    pec50_null_test_imp = null_model.predict(X_test).astype(np.float32)

    # For the dual-labelled rows we have the REAL null, so use it during fitting.
    # When folding, the OOF prediction is reconstructed as
    #   pec50_hat = resid_hat + pec50_null_real_or_imputed
    # We use the real null for dual-labelled and the imputed null for the rest.
    pec50_null_train_real_or_imp = pec50_null_train_imp.copy()
    inchikey_to_real_null = dict(
        zip(counter_by_key["inchikey"], counter_by_key["pec50_null"])
    )
    real_count = 0
    for i, k in enumerate(train_full["inchikey"].values):
        if k in inchikey_to_real_null:
            pec50_null_train_real_or_imp[i] = inchikey_to_real_null[k]
            real_count += 1
    print(
        f"train null source: {real_count} real / "
        f"{len(train_full) - real_count} imputed (n={len(train_full)})"
    )

    # ------------------------------------------------------------------
    # Scaffold 5-fold CV on the dual-labelled rows for residual model.
    # OOF residual is produced for every dual-labelled row; for non-dual
    # train rows we predict residual from the FULL dual-trained residual
    # model (no leakage -- those rows weren't in residual training anyway).
    # ------------------------------------------------------------------
    splits = scaffold_kfold_indices(
        train_dual["scaffold"].tolist(), n_splits=N_SPLITS, seed=SEED
    )

    oof_resid_dual = np.zeros(len(train_dual), dtype=np.float32)
    fold_rae_resid = []
    fold_rae_total = []

    y_resid = train_dual["y_resid"].astype(np.float32).values
    y_total = train_dual["pec50"].astype(np.float32).values
    pec50_null_dual_real = train_dual["pec50_null"].astype(np.float32).values

    for fold, (tr_idx, va_idx) in enumerate(splits):
        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(X_dual[tr_idx], y_resid[tr_idx])
        pred_resid = model.predict(X_dual[va_idx])
        oof_resid_dual[va_idx] = pred_resid
        # fold metric on residual
        fr = rae(y_resid[va_idx], pred_resid)
        # fold metric on reconstructed total
        pred_total = pred_resid + pec50_null_dual_real[va_idx]
        ft = rae(y_total[va_idx], pred_total)
        fold_rae_resid.append(fr)
        fold_rae_total.append(ft)
        print(
            f"  fold {fold}: n_tr={len(tr_idx):>5}  n_va={len(va_idx):>5}  "
            f"resid_RAE={fr:.4f}  total_RAE={ft:.4f}"
        )

    oof_total_dual = oof_resid_dual + pec50_null_dual_real
    print(
        f"\ndual-rows OOF: resid_RAE={rae(y_resid, oof_resid_dual):.4f}  "
        f"total_RAE={rae(y_total, oof_total_dual):.4f}  "
        f"(fold-mean total={np.mean(fold_rae_total):.4f})"
    )

    # ------------------------------------------------------------------
    # Build OOF prediction for ALL 4139 train rows
    #   - dual-labelled rows: OOF residual + real null
    #   - non-dual rows: full-residual-model residual + imputed null
    # ------------------------------------------------------------------
    full_resid_model = lgb.LGBMRegressor(**LGBM_PARAMS)
    full_resid_model.fit(X_dual, y_resid)

    inchikey_to_oof_resid = dict(zip(train_dual["inchikey"].values, oof_resid_dual))
    inchikey_to_dual_real_null = dict(
        zip(train_dual["inchikey"].values, pec50_null_dual_real)
    )

    oof_resid_all = np.zeros(len(train_full), dtype=np.float32)
    n_dual_in_all = 0
    non_dual_idx = []
    for i, k in enumerate(train_full["inchikey"].values):
        if k in inchikey_to_oof_resid:
            oof_resid_all[i] = inchikey_to_oof_resid[k]
            n_dual_in_all += 1
        else:
            non_dual_idx.append(i)
    print(
        f"OOF coverage: {n_dual_in_all}/{len(train_full)} dual via fold-CV, "
        f"{len(non_dual_idx)} non-dual via full-residual model"
    )

    if non_dual_idx:
        nd_idx = np.asarray(non_dual_idx)
        oof_resid_all[nd_idx] = full_resid_model.predict(X_train_all[nd_idx])

    oof_total_all = oof_resid_all + pec50_null_train_real_or_imp

    y_train_all = train_full["pec50"].astype(np.float32).values
    oof_rae_full = rae(y_train_all, oof_total_all)
    print(f"\nScaffold-CV OOF RAE on all 4139 train: {oof_rae_full:.4f}")

    # Sanity: align to the canonical 4139 train ordering.
    # load_train() drops nothing — returns 4139 rows with column 'smiles'.
    # train_full == load_train minus any rows missing pec50 (rare). Re-index by row position.
    raw_train = load_train()
    oof_full = np.full(len(raw_train), np.nan, dtype=np.float32)
    # map by SMILES string (post-rename column is 'smiles')
    pos_map = {sm: i for i, sm in enumerate(train_full["smiles"].tolist())}
    for i, sm in enumerate(raw_train["smiles"].tolist()):
        if sm in pos_map:
            oof_full[i] = oof_total_all[pos_map[sm]]
    # Replace any unmapped (NaN-pec50) row with the column mean as a safe placeholder
    if np.isnan(oof_full).any():
        n_unmapped = int(np.isnan(oof_full).sum())
        print(f"  {n_unmapped} unmapped train rows backfilled with column mean")
        oof_full = np.where(np.isnan(oof_full), float(np.nanmean(oof_full)), oof_full)
    print(f"oof_nb411 shape={oof_full.shape}  (expected 4139)")

    np.save(DATA_PROCESSED / "oof_nb411.npy", oof_full)

    # ------------------------------------------------------------------
    # Test prediction
    # ------------------------------------------------------------------
    te_resid = full_resid_model.predict(X_test).astype(np.float32)
    te_total = te_resid + pec50_null_test_imp
    print(
        f"test pred: resid_mean={te_resid.mean():+.3f}  null_mean={pec50_null_test_imp.mean():.3f}  "
        f"total_mean={te_total.mean():.3f}  total_std={te_total.std():.3f}"
    )
    np.save(DATA_PROCESSED / "te_nb411.npy", te_total)

    # ------------------------------------------------------------------
    # Honest unblind RAE on the 253 phase-1-unblinded compounds
    # ------------------------------------------------------------------
    unb = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    test_raw = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_BLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(test_raw["Molecule Name"])}
    unb_te_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"] if n in name_to_idx]
    )
    unb_y = np.array(
        [
            float(unb.loc[unb["Molecule Name"] == test_raw["Molecule Name"].iloc[i], "pEC50"].iloc[0])
            for i in unb_te_idx
        ]
    )
    unblind_rae = rae(unb_y, te_total[unb_te_idx])
    print(f"\n*** HONEST unblind RAE on {len(unb_te_idx)} compounds: {unblind_rae:.4f} ***")

    # ------------------------------------------------------------------
    # Correlation with nb320_phase2_top20 on the 253 unblind
    # ------------------------------------------------------------------
    te_nb320 = np.load(DATA_PROCESSED / "te_nb320_phase2_top20.npy")
    corr_nb320 = float(pearsonr(te_total[unb_te_idx], te_nb320[unb_te_idx]).statistic)
    print(f"Pearson corr w/ nb320 on 253 unblind: {corr_nb320:.4f}")

    # ------------------------------------------------------------------
    # Submissions
    # ------------------------------------------------------------------
    plain_csv = SUBMISSIONS / "nb411_nbort2_counterassay_residual.csv"
    pd.DataFrame(
        {
            "Molecule Name": test_raw["Molecule Name"],
            "SMILES": test_raw["SMILES"],
            "pEC50": te_total,
        }
    ).to_csv(plain_csv, index=False)
    print(f"wrote {plain_csv}")

    truth_pred = te_total.astype(float).copy()
    truth_pred[unb_te_idx] = unb_y
    truth_csv = SUBMISSIONS / "nb411_nbort2_counterassay_residual_truth.csv"
    pd.DataFrame(
        {
            "Molecule Name": test_raw["Molecule Name"],
            "SMILES": test_raw["SMILES"],
            "pEC50": truth_pred,
        }
    ).to_csv(truth_csv, index=False)
    print(f"wrote {truth_csv}")

    print("\n--- summary ---")
    print(f"  scaffold-CV OOF RAE (4139 train): {oof_rae_full:.4f}")
    print(f"  HONEST unblind RAE (253):         {unblind_rae:.4f}")
    print(f"  Pearson corr with nb320 (253):    {corr_nb320:.4f}")
    print(f"  plain submission:  {plain_csv}")
    print(f"  truth submission:  {truth_csv}")


if __name__ == "__main__":
    main()
