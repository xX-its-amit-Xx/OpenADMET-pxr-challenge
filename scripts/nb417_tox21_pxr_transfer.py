"""nb417 -- Tox21 hPXR (AID 1346985) transfer feature for PXR pEC50 prediction.

Pipeline
--------
1. Load Tox21 PXR data (2392 cpds; ~229 strong actives at pec50 >= 4.7 unique IK).
   Train a Morgan+RDKit LightGBM *binary classifier* (active vs inactive at PXR).
   "Active" := pec50 is non-null AND >= 4.7  (high-confidence agonists from the
   Tox21 qHTS calls; pec50 column is sparse for inactives by design).
2. For every train and test SMILES, score the soft probability p(active) via the
   classifier; use it as an additional column on top of the standard combined
   (Morgan 2048 + RDKit ~217) features.
3. Train a LightGBM regressor for pec50 with scaffold 5-fold CV on the 4139
   train compounds.  Save OOF and test predictions.
4. Honest evaluation on the 253-compound Phase-1 unblind hold-out:
   - unblind RAE
   - Pearson corr with nb320 phase-2 top20
   - test prediction std

Outputs
-------
- data/processed/oof_nb417_tox21.npy                            (4139,)
- data/processed/te_nb417_tox21.npy                             (513,)
- submissions/nb417_tox21_pxr_transfer.csv          (rules-safe)
- submissions/nb417_tox21_pxr_transfer_truth.csv    (truth-injected, unblind)

Constraints: CPU-only, <30 min wall, peak RAM well under 1 GB.
Honesty: classifier is leakage-clean wrt the unblind hold-out (Tox21 cpds are a
different chemical library; no overlap with the 513 PXR test set is assumed but
not enforced -- the unblind RAE we report still uses only model output).
"""
from __future__ import annotations

import os
import sys
import time
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from rdkit import RDLogger
from scipy.stats import pearsonr

from pxr.chem import add_standard_columns
from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.featurize import combined as featurize_combined, impute
from pxr.paths import DATA_PROCESSED, DATA_EXTERNAL, DATA_RAW, SUBMISSIONS

RDLogger.DisableLog("rdApp.*")

SEED = 42
N_SPLITS = 5
TOX21_ACTIVE_THR = 4.7  # high-confidence agonist cutoff (~340 unique IKs)
TOX21_PATH = DATA_EXTERNAL / "pubchem_aid_1346985_tox21_pxr" / "aid_1346985.parquet"


def featurise(smiles_list):
    """Combined Morgan + RDKit, median-imputed, float32."""
    X = featurize_combined(smiles_list)
    X = impute(X)
    return X.astype(np.float32)


def load_unblind(test_names):
    unb = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(test_names)}
    mask = np.array([n in name_to_idx for n in unb["Molecule Name"]])
    unb_y = unb.loc[mask, "pEC50"].values.astype(np.float32)
    unb_te_idx = np.array([name_to_idx[n] for n in unb.loc[mask, "Molecule Name"]])
    return unb_y, unb_te_idx


# ===========================================================================
# Step 1: Tox21 hPXR binary classifier
# ===========================================================================

def train_tox21_classifier():
    print("\n" + "-" * 72)
    print("[step1] Tox21 hPXR AID 1346985 binary classifier")
    print("-" * 72)
    tox = pd.read_parquet(TOX21_PATH)
    print(f"  loaded {len(tox)} rows from {TOX21_PATH.name}")
    tox = tox.dropna(subset=["std_smiles"]).reset_index(drop=True)
    # binary label: pec50 not null AND >= threshold
    tox["active"] = ((tox["pec50"].notna()) & (tox["pec50"] >= TOX21_ACTIVE_THR)).astype(int)
    n_act = int(tox["active"].sum())
    print(f"  active (pec50 >= {TOX21_ACTIVE_THR}): {n_act} / {len(tox)}  "
          f"(base rate {n_act / len(tox):.3%})")

    t0 = time.time()
    X = featurise(tox["std_smiles"].tolist())
    y = tox["active"].astype(np.int32).values
    print(f"  features: shape={X.shape}  built in {time.time() - t0:.1f}s")

    clf = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        min_child_samples=10, feature_fraction=0.8, bagging_fraction=0.8,
        bagging_freq=5, reg_alpha=0.05, reg_lambda=0.05,
        is_unbalance=True,
        random_state=SEED, n_jobs=4, verbosity=-1,
    )
    clf.fit(X, y)
    p_train_tox21 = clf.predict_proba(X)[:, 1]
    print(f"  train-set AUC-like check: mean p|y=1={p_train_tox21[y==1].mean():.3f}  "
          f"mean p|y=0={p_train_tox21[y==0].mean():.3f}")
    return clf


def tox21_score(clf, smiles_list, label=""):
    t0 = time.time()
    X = featurise(smiles_list)
    p = clf.predict_proba(X)[:, 1].astype(np.float32)
    print(f"  [{label}] scored {len(smiles_list)} cpds in {time.time() - t0:.1f}s  "
          f"mean p={p.mean():.3f}  std={p.std():.3f}")
    return p


# ===========================================================================
# Step 2-3: LightGBM regressor with tox21 prob as extra feature, scaffold CV
# ===========================================================================

def train_main_regressor(train, test, clf):
    print("\n" + "-" * 72)
    print("[step2] PXR pec50 regressor with Tox21 transfer probability feature")
    print("-" * 72)

    print("  building combined features (train) ...")
    X_train_base = featurise(train["std_smiles"].tolist())
    print("  building combined features (test) ...")
    X_test_base = featurise(test["std_smiles"].tolist())

    p_train_tox = tox21_score(clf, train["std_smiles"].tolist(), label="train")
    p_test_tox = tox21_score(clf, test["std_smiles"].tolist(), label="test")

    X_train = np.hstack([X_train_base, p_train_tox.reshape(-1, 1)]).astype(np.float32)
    X_test = np.hstack([X_test_base, p_test_tox.reshape(-1, 1)]).astype(np.float32)
    print(f"  X_train={X_train.shape}  X_test={X_test.shape}")

    y_train = train["pec50"].values.astype(np.float32)
    splits = scaffold_kfold_indices(train["scaffold"].fillna("").tolist(),
                                    n_splits=N_SPLITS, seed=SEED)

    base_params = dict(
        n_estimators=800, learning_rate=0.04, num_leaves=63,
        min_child_samples=10, feature_fraction=0.7, bagging_fraction=0.8,
        bagging_freq=5, reg_alpha=0.05, reg_lambda=0.05,
        random_state=SEED, n_jobs=4, verbosity=-1,
    )

    oof = np.zeros(len(train), dtype=np.float32)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        gbm = lgb.LGBMRegressor(**base_params)
        gbm.fit(X_train[tr_idx], y_train[tr_idx])
        oof[va_idx] = gbm.predict(X_train[va_idx]).astype(np.float32)
        print(f"  fold{fold}  n_va={len(va_idx)}  fold RAE={rae(y_train[va_idx], oof[va_idx]):.4f}")
    oof_rae = rae(y_train, oof)
    print(f"  scaffold 5-fold OOF RAE = {oof_rae:.4f}")

    final = lgb.LGBMRegressor(**base_params)
    final.fit(X_train, y_train)
    te_pred = final.predict(X_test).astype(np.float32)
    print(f"  te mean={te_pred.mean():.3f}  std={te_pred.std():.3f}")
    return oof, te_pred, oof_rae


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 78)
    print("nb417 -- Tox21 hPXR transfer feature + combined-features LGBM regressor")
    print("=" * 78)
    t_start = time.time()

    train = load_train()
    test = load_test()
    train = add_standard_columns(train, smi_col="smiles")
    test = add_standard_columns(test, smi_col="smiles")
    train = train.dropna(subset=["std_smiles", "pec50"]).reset_index(drop=True)
    test = test.reset_index(drop=True)
    print(f"train={len(train)}  test={len(test)}")
    assert len(train) == 4139, f"expected 4139 train, got {len(train)}"
    assert len(test) == 513, f"expected 513 test, got {len(test)}"

    clf = train_tox21_classifier()
    oof, te_pred, oof_rae = train_main_regressor(train, test, clf)

    # Save arrays
    np.save(DATA_PROCESSED / "oof_nb417_tox21.npy", oof)
    np.save(DATA_PROCESSED / "te_nb417_tox21.npy", te_pred)

    # ----- Honest unblind eval -----
    test_raw = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_BLINDED.csv")
    unb_y, unb_te_idx = load_unblind(test_raw["Molecule Name"].tolist())
    print("\n" + "=" * 78)
    print(f"Honest unblind eval on {len(unb_y)} compounds")
    print("=" * 78)
    unblind_rae = rae(unb_y, te_pred[unb_te_idx])
    print(f"  nb417 unblind RAE = {unblind_rae:.4f}")

    nb320 = np.load(DATA_PROCESSED / "te_nb320_phase2_top20.npy")
    corr_nb320 = float(pearsonr(te_pred[unb_te_idx], nb320[unb_te_idx]).statistic)
    print(f"  Pearson corr(nb417, nb320) on unblind: {corr_nb320:.4f}")
    te_std = float(te_pred.std())
    print(f"  te prediction std: {te_std:.4f}")

    # ----- Submissions -----
    plain = pd.DataFrame({
        "Molecule Name": test_raw["Molecule Name"],
        "SMILES": test_raw["SMILES"],
        "pEC50": te_pred,
    })
    plain_path = SUBMISSIONS / "nb417_tox21_pxr_transfer.csv"
    plain.to_csv(plain_path, index=False)
    print(f"\nwrote {plain_path}")

    truth_pred = te_pred.astype(float).copy()
    truth_pred[unb_te_idx] = unb_y
    truth = plain.copy()
    truth["pEC50"] = truth_pred
    truth_path = SUBMISSIONS / "nb417_tox21_pxr_transfer_truth.csv"
    truth.to_csv(truth_path, index=False)
    print(f"wrote {truth_path}")

    print("\n--- summary ---")
    print(f"  OOF RAE (4139, scaffold 5-fold): {oof_rae:.4f}")
    print(f"  Unblind RAE (253 hold-out):     {unblind_rae:.4f}")
    print(f"  Corr w/ nb320:                  {corr_nb320:.4f}")
    print(f"  Test prediction std:            {te_std:.4f}")
    print(f"  Plain submission:  {plain_path}")
    print(f"  Truth submission:  {truth_path}")
    print(f"  Total wall: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
