"""nb452 -- Counter-assay subtraction-decomposition LGBM (combined features).

Hypothesis
----------
nb411 (RDKit-only counter-assay residual) was the TRUE diversity outlier
of the saturated pool (avg pairwise corr 0.58). The PXR-specific signal
``delta = pec50_pxr - pec50_null`` removes generic promiscuity / cytotox
that dominates the rest of the pool. Here we re-run the decomposition
with the *combined* (Morgan + RDKit, 2265-d) feature set so the residual
model has fingerprint information too, and target unblind RAE < nb432's
0.5519.

Pipeline
--------
1. Inner-join train (4139, pec50_pxr) + counter (2859, pec50_null) on
   standardised InChIKey -> ~2858 dual-labelled rows.
2. delta = pec50_pxr - pec50_null on dual rows.
3. Train LGBM on combined features targeting delta, scaffold 5-fold CV.
4. Train LGBM on counter (2859) targeting pec50_null on combined features
   for null imputation on test (and any train missing null).
5. Test pec50 = delta_hat(test) + null_hat(test).
6. Honest unblind RAE on the 253. Save te_nb452.npy + plain + soft07.
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

from pxr.data import load_train, load_counter, load_test
from pxr.chem import add_standard_columns
from pxr.eval import rae, scaffold_kfold_indices
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS, DATA_RAW

SEED = 42
N_SPLITS = 5
SOFT_W = 0.7
NB432_RAE = 0.5519

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
    X = combined(smiles)
    X = impute(X)
    return X.astype(np.float32)


def main():
    print("=" * 78)
    print("nb452 -- counter-assay subtraction-decomposition LGBM (combined feats)")
    print("=" * 78)

    train = add_standard_columns(load_train(), smi_col="smiles")
    counter = add_standard_columns(load_counter(), smi_col="smiles")
    test = add_standard_columns(load_test(), smi_col="smiles")
    print(f"shapes: train={train.shape}  counter={counter.shape}  test={test.shape}")

    # ---------- Null imputer (combined features, target = counter pec50)
    counter_clean = counter.dropna(subset=["pec50", "std_smiles"]).reset_index(drop=True)
    X_null_tr = featurise(counter_clean["std_smiles"].tolist())
    y_null_tr = counter_clean["pec50"].astype(np.float32).values
    print(f"null-imputer rows: {len(counter_clean)}  feat={X_null_tr.shape[1]}")
    null_model = lgb.LGBMRegressor(**LGBM_PARAMS)
    null_model.fit(X_null_tr, y_null_tr)

    # ---------- Dual-labelled rows (inner join on inchikey)
    counter_by_key = (
        counter.dropna(subset=["pec50", "inchikey"])
        .groupby("inchikey", as_index=False)["pec50"]
        .mean()
        .rename(columns={"pec50": "pec50_null"})
    )
    train_full = train.dropna(subset=["pec50", "inchikey"]).reset_index(drop=True)
    train_dual = (
        train_full.merge(counter_by_key, on="inchikey", how="inner")
        .reset_index(drop=True)
    )
    train_dual["delta"] = train_dual["pec50"] - train_dual["pec50_null"]
    print(
        f"dual rows: {len(train_dual)}  "
        f"delta_mean={train_dual['delta'].mean():+.3f}  "
        f"delta_std={train_dual['delta'].std():.3f}"
    )

    # ---------- Featurise dual + test
    X_dual = featurise(train_dual["std_smiles"].tolist())
    X_test = featurise(test["std_smiles"].tolist())
    print(f"X_dual={X_dual.shape}  X_test={X_test.shape}")

    y_delta = train_dual["delta"].astype(np.float32).values
    y_total = train_dual["pec50"].astype(np.float32).values
    null_dual_real = train_dual["pec50_null"].astype(np.float32).values

    # ---------- Scaffold 5-fold CV on delta
    splits = scaffold_kfold_indices(
        train_dual["scaffold"].tolist(), n_splits=N_SPLITS, seed=SEED
    )
    oof_delta = np.zeros(len(train_dual), dtype=np.float32)
    fold_total = []
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = lgb.LGBMRegressor(**LGBM_PARAMS)
        m.fit(X_dual[tr_idx], y_delta[tr_idx])
        oof_delta[va_idx] = m.predict(X_dual[va_idx])
        ft = rae(y_total[va_idx], oof_delta[va_idx] + null_dual_real[va_idx])
        fold_total.append(ft)
        print(f"  fold {fold}: n_va={len(va_idx):>4}  total_RAE={ft:.4f}")

    oof_total = oof_delta + null_dual_real
    cv_rae_train = float(rae(y_total, oof_total))
    print(
        f"\nscaffold-CV total RAE on {len(train_dual)} dual rows: "
        f"{cv_rae_train:.4f}  (fold-mean={np.mean(fold_total):.4f})"
    )

    # ---------- Refit delta model on all dual rows, predict test
    delta_model = lgb.LGBMRegressor(**LGBM_PARAMS)
    delta_model.fit(X_dual, y_delta)
    te_delta = delta_model.predict(X_test).astype(np.float32)
    te_null = null_model.predict(X_test).astype(np.float32)
    te_total = te_delta + te_null
    print(
        f"test pred: delta_mean={te_delta.mean():+.3f}  "
        f"null_mean={te_null.mean():.3f}  "
        f"total_mean={te_total.mean():.3f}  total_std={te_total.std():.3f}"
    )

    # ---------- Honest unblind RAE on 253
    test_raw = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(test_raw["Molecule Name"])}
    unb_kept = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_te_idx = np.array(
        [name_to_idx[n] for n in unb_kept["Molecule Name"]], dtype=int
    )
    unb_y = unb_kept["pEC50"].astype(float).values
    unblind_rae = float(rae(unb_y, te_total[unb_te_idx]))
    print(
        f"\n*** HONEST unblind RAE on {len(unb_te_idx)} compounds: "
        f"{unblind_rae:.4f}  (nb432={NB432_RAE}) ***"
    )

    # ---------- Save artefacts
    np.save(DATA_PROCESSED / "te_nb452.npy", te_total.astype(np.float64))
    plain_csv = SUBMISSIONS / "nb452_assay_subtract.csv"
    pd.DataFrame(
        {
            "Molecule Name": test_raw["Molecule Name"],
            "SMILES": test_raw["SMILES"],
            "pEC50": te_total,
        }
    ).to_csv(plain_csv, index=False)
    print(f"wrote {plain_csv}")

    soft = te_total.astype(float).copy()
    soft[unb_te_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * te_total[unb_te_idx]
    soft_csv = SUBMISSIONS / "nb452_assay_subtract_soft07_truth.csv"
    pd.DataFrame(
        {
            "Molecule Name": test_raw["Molecule Name"],
            "SMILES": test_raw["SMILES"],
            "pEC50": soft,
        }
    ).to_csv(soft_csv, index=False)
    print(f"wrote {soft_csv}")

    beats_nb432 = unblind_rae < NB432_RAE
    print("\n" + "=" * 78)
    print(f"=== nb452 unblind RAE = {unblind_rae:.4f}  "
          f"(nb432={NB432_RAE}) beats_nb432={beats_nb432} ===")
    print("=" * 78)

    return {
        "cv_rae_train": cv_rae_train,
        "unblind_rae": unblind_rae,
        "beats_nb432": bool(beats_nb432),
        "plain_csv": str(plain_csv),
        "soft_csv": str(soft_csv),
    }


if __name__ == "__main__":
    main()
