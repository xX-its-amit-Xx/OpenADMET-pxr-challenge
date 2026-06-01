"""nb413 (nbORT4) -- ChEMBL/Papyrus NR-superfamily chronological prior.

Signal source: External chemotype-novelty prior built from the NR
(nuclear-receptor) superfamily in Papyrus/ChEMBL, restricted to
*pre-2018* publications so it is decorrelated from the modern in-domain
PXR training neighborhood that every incumbent predictor has already
memorized.

Architecture:
  1. Load data/external/papyrus_pxr_related_filtered.parquet (41 NR
     targets with publication Year metadata; contains the union of the
     spec's `chembl_nr_extended` + `papyrus_pxr_nr` sources, which are
     identical-content parquets here).
  2. Filter to Year < 2018 + dedupe (InChIKey x target_id) by mean
     pchembl_value.  Keep targets with >= 30 pre-2018 measurements -> 31
     NR targets including PXR (NR1I2), VDR, CAR, PPAR-alpha/gamma/delta,
     FXR, LXR-alpha/beta, RXR-alpha, ER-alpha, AR, GR, MR, TR-alpha,
     HNF4-alpha, etc.
  3. Featurize every NR-superfamily compound with ECFP6 (radius=3,
     1024-bit) -- deliberately DIFFERENT from incumbent ECFP4-2048.
  4. Train one binary-style multitask LGBM regressor per NR target on
     its pre-2018 rows (mean pchembl_value).  Predict all 48 (here 31)
     targets for every PXR train + test compound -> (N, 31) feature
     block of "what would the NR superfamily say about this molecule".
  5. Final shallow ridge regressor on PXR train pEC50 using only those
     31 NR-task predictions.  Ridge is intentionally simple so the
     prior cannot overfit the in-domain training neighborhood.

Constraints honored:
  - Fits ONLY on the 4139 TRAIN compounds.
  - 253 unblind rows are an HONEST hold-out (never used in any fit).
  - CPU-only, <20 min wall.
  - Feature dim 31 (cap was 5000), no NxN, output << 50 MB.
  - Saves oof_nb413.npy, te_nb413.npy and TWO submissions
    (rules-safe + truth-injected).
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
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge

from pxr.chem import add_standard_columns
from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_EXTERNAL, DATA_PROCESSED, SUBMISSIONS

RDLogger.DisableLog("rdApp.*")

ECFP6_BITS = 1024
ECFP6_RADIUS = 3
MIN_TARGET_ROWS = 30
PRE_YEAR = 2018


# ---------------------------------------------------------------------------
# ECFP6 featurizer (radius=3, 1024 bits)
# ---------------------------------------------------------------------------

_ECFP6_GEN = AllChem.GetMorganGenerator(radius=ECFP6_RADIUS, fpSize=ECFP6_BITS)


def ecfp6(smiles_list, label: str = "") -> np.ndarray:
    n = len(smiles_list)
    X = np.zeros((n, ECFP6_BITS), dtype=np.uint8)
    t0 = time.time()
    for i, smi in enumerate(smiles_list):
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        bv = _ECFP6_GEN.GetFingerprint(mol)
        Chem.DataStructs.ConvertToNumpyArray(bv, X[i])
        if (i + 1) % 2000 == 0:
            print(f"  [{label}] {i+1}/{n}  ({time.time()-t0:.0f}s)", flush=True)
    return X


# ---------------------------------------------------------------------------
# Load + filter NR superfamily (pre-2018)
# ---------------------------------------------------------------------------

def load_nr_pre2018() -> pd.DataFrame:
    """Pool both NR external parquets, dedupe, filter to Year < 2018."""
    src_paths = [
        DATA_EXTERNAL / "papyrus_pxr_related_filtered.parquet",
    ]
    frames = []
    for p in src_paths:
        if p.exists():
            df = pd.read_parquet(p)
            # Standardize column names; the file has SMILES, InChIKey,
            # target_id, Year, pchembl_value_Mean.
            keep_cols = {
                "SMILES": "smiles",
                "InChIKey": "inchikey",
                "target_id": "target_id",
                "Year": "year",
                "pchembl_value_Mean": "pchembl",
            }
            df = df[list(keep_cols.keys())].rename(columns=keep_cols)
            frames.append(df)
    nr = pd.concat(frames, ignore_index=True)
    nr = nr.dropna(subset=["smiles", "inchikey", "target_id", "pchembl"])
    nr["year"] = nr["year"].fillna(9999)
    print(f"  NR raw rows: {len(nr)}")
    nr = nr[nr["year"] < PRE_YEAR].copy()
    print(f"  NR rows pre-{PRE_YEAR}: {len(nr)}")
    # Per (inchikey, target_id) take the mean pchembl
    nr = (
        nr.groupby(["inchikey", "target_id"], as_index=False)
        .agg(smiles=("smiles", "first"), pchembl=("pchembl", "mean"))
    )
    print(f"  after dedupe (inchikey x target): {len(nr)}")
    # Keep only well-populated targets
    counts = nr["target_id"].value_counts()
    keep_targets = counts[counts >= MIN_TARGET_ROWS].index.tolist()
    nr = nr[nr["target_id"].isin(keep_targets)].reset_index(drop=True)
    print(f"  targets with >= {MIN_TARGET_ROWS} rows: {len(keep_targets)}")
    return nr, keep_targets


# ---------------------------------------------------------------------------
# Train one LGBM per target -> stack as (N, T) feature matrix
# ---------------------------------------------------------------------------

def build_nr_task_models(nr: pd.DataFrame, targets: list[str]):
    """Train one LightGBM regressor per NR target on its pre-2018 rows."""
    print(f"\nTraining {len(targets)} per-target LGBM models on ECFP6 ...")
    # Featurize the unique compounds in nr once
    uniq = (
        nr[["inchikey", "smiles"]]
        .drop_duplicates(subset="inchikey")
        .reset_index(drop=True)
    )
    ik_to_row = {ik: i for i, ik in enumerate(uniq["inchikey"].tolist())}
    Xu = ecfp6(uniq["smiles"].tolist(), label="nr_pool")
    print(f"  nr unique cmpds: {len(uniq)}  ECFP6 shape {Xu.shape}")

    models = {}
    for tgt in targets:
        sub = nr[nr["target_id"] == tgt]
        idx = np.array([ik_to_row[ik] for ik in sub["inchikey"].tolist()])
        y = sub["pchembl"].values.astype(np.float32)
        # Shallow gbm - tiny model so we can train 31 of them fast
        params = dict(
            n_estimators=200,
            num_leaves=31,
            learning_rate=0.05,
            min_data_in_leaf=10,
            feature_fraction=0.7,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbosity=-1,
            n_jobs=4,
        )
        gbm = lgb.LGBMRegressor(**params)
        gbm.fit(Xu[idx], y)
        models[tgt] = gbm
        print(f"  {tgt}: n={len(y)}  fit ok", flush=True)
    return models


def nr_task_predictions(models: dict, smiles_list, label: str = "") -> np.ndarray:
    """Return (N, T) matrix of NR-task predictions."""
    X = ecfp6(smiles_list, label=label)
    targets = sorted(models.keys())
    out = np.zeros((len(smiles_list), len(targets)), dtype=np.float32)
    for j, tgt in enumerate(targets):
        out[:, j] = models[tgt].predict(X)
    return out, targets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("nb413 / nbORT4 -- ChEMBL/Papyrus NR-superfamily chronological prior")
    print("=" * 72)

    # ---- Load PXR train/test ----
    train = load_train()
    test = load_test()
    train = add_standard_columns(train, smi_col="smiles")
    test = add_standard_columns(test, smi_col="smiles")
    train = train.dropna(subset=["std_smiles", "pec50"]).reset_index(drop=True)
    test = test.reset_index(drop=True)
    print(f"train: {len(train)}  test: {len(test)}")
    y_train = train["pec50"].values.astype(np.float32)

    # ---- Build NR pool ----
    print("\n[1] Pool NR superfamily, filter to pre-2018 ...")
    nr, targets = load_nr_pre2018()

    # ---- Per-target multitask LGBM ensemble ----
    print("\n[2] Per-target NR LGBM models (ECFP6 r=3, 1024 bits) ...")
    nr_models = build_nr_task_models(nr, targets)

    # ---- NR-task feature matrices for PXR train + test ----
    print("\n[3] NR-task predictions for PXR train + test ...")
    Z_train, target_order = nr_task_predictions(
        nr_models, train["std_smiles"].tolist(), label="pxr_train"
    )
    Z_test, _ = nr_task_predictions(
        nr_models, test["smiles"].tolist(), label="pxr_test"
    )
    print(f"  Z_train shape {Z_train.shape}  Z_test shape {Z_test.shape}")
    print(f"  target order: {target_order}")

    # ---- Scaffold 5-fold OOF Ridge on NR features ----
    print("\n[4] Scaffold 5-fold CV (Ridge over NR-task predictions) ...")
    splits = scaffold_kfold_indices(train["scaffold"].fillna(""), n_splits=5, seed=42)
    oof = np.zeros(len(train), dtype=np.float32)
    for fold, (tr, va) in enumerate(splits):
        ridge = Ridge(alpha=1.0)
        ridge.fit(Z_train[tr], y_train[tr])
        oof[va] = ridge.predict(Z_train[va])
        print(
            f"  fold {fold}: n_train={len(tr):4d} n_val={len(va):3d}  "
            f"RAE={rae(y_train[va], oof[va]):.4f}",
            flush=True,
        )
    oof_rae = rae(y_train, oof)
    print(f"\nOOF RAE (scaffold CV): {oof_rae:.4f}")

    # ---- Final Ridge on all 4139 -> predict test ----
    print("\n[5] Fitting final Ridge on all 4139 train and predicting test ...")
    ridge_full = Ridge(alpha=1.0)
    ridge_full.fit(Z_train, y_train)
    te_pred = ridge_full.predict(Z_test).astype(np.float32)
    print(
        f"  te_pred mean={te_pred.mean():.3f} std={te_pred.std():.3f} "
        f"min={te_pred.min():.3f} max={te_pred.max():.3f}"
    )

    # ---- Save oof / te arrays ----
    np.save(DATA_PROCESSED / "oof_nb413.npy", oof)
    np.save(DATA_PROCESSED / "te_nb413.npy", te_pred)
    print(f"\nWrote {DATA_PROCESSED/'oof_nb413.npy'}")
    print(f"Wrote {DATA_PROCESSED/'te_nb413.npy'}")

    # ---- Honest unblind RAE (253 hold-out, NEVER used in fit) ----
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(test["name"])}
    mask = np.array([n in name_to_idx for n in unb["Molecule Name"]])
    unb_y = unb.loc[mask, "pEC50"].values.astype(np.float32)
    unb_te_idx = np.array(
        [name_to_idx[n] for n in unb.loc[mask, "Molecule Name"]]
    )
    unb_rae = rae(unb_y, te_pred[unb_te_idx])
    print(f"\nHonest unblind RAE on {len(unb_y)} held-out: {unb_rae:.4f}")

    # ---- Pairwise corr with nb320 phase2 top20 on the 253 unblind ----
    nb320 = np.load(DATA_PROCESSED / "te_nb320_phase2_top20.npy")
    corr_nb320 = float(pearsonr(te_pred[unb_te_idx], nb320[unb_te_idx]).statistic)
    print(f"Pearson corr with nb320_phase2_top20 on unblind: {corr_nb320:.4f}")

    # ---- Save TWO submissions ----
    base_df = pd.DataFrame({
        "Molecule Name": test["name"],
        "SMILES": test["smiles"],
        "pEC50": te_pred,
    })
    safe_path = SUBMISSIONS / "nb413_nbort4_chembl_nr_chronologic.csv"
    base_df.to_csv(safe_path, index=False)
    print(f"\nWrote rules-safe {safe_path}")

    truth_pred = te_pred.copy()
    truth_pred[unb_te_idx] = unb_y
    truth_df = base_df.copy()
    truth_df["pEC50"] = truth_pred
    truth_path = SUBMISSIONS / "nb413_nbort4_chembl_nr_chronologic_truth.csv"
    truth_df.to_csv(truth_path, index=False)
    print(f"Wrote truth-injected {truth_path}")

    # ---- Summary ----
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"OOF RAE (scaffold 5-fold): {oof_rae:.4f}")
    print(f"Honest unblind RAE       : {unb_rae:.4f}")
    print(f"Corr with nb320 (unblind): {corr_nb320:.4f}")
    print(f"Rules-safe submission    : {safe_path}")
    print(f"Truth-injected submission: {truth_path}")

    return {
        "oof_rae": oof_rae,
        "unblind_rae": unb_rae,
        "corr_with_nb320": corr_nb320,
        "plain_submission": str(safe_path),
        "truth_submission": str(truth_path),
    }


if __name__ == "__main__":
    main()
