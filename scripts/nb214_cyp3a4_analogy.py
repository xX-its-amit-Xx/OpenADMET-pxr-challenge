"""nb214 -- CYP3A4 analogy: feature transfer from a mechanistically linked large assay.

PXR activates CYP3A4 transcription directly. A model trained on large CYP3A4
inhibition/activation data encodes molecular features predictive of PXR axis
engagement. We use its predictions as extra features, not extra training rows
(avoids the chemical-space mismatch that hurt nb55/57/58).

Steps:
  A: Fetch CYP3A4 data from ChEMBL (CHEMBL340) + AhR (CHEMBL3905) as second analogy
  B: Deduplicate, median-aggregate, standardize SMILES
  C: Train LGBM on each large external dataset (5-fold CV within that dataset)
  D: Apply models to ALL PXR train+test compounds -> virtual CYP3A4/AhR features
  E: Scaffold 5-fold CV on augmented PXR features; compare to base OOF 0.2976
  F: Save best OOF/test arrays and submission if improvement found
"""
import os, sys, warnings, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import lightgbm as lgb
from sklearn.model_selection import KFold

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns, morgan_fp_batch
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, DATA_EXTERNAL, SUBMISSIONS

COLLAPSE_THRESH = 0.58

LGBM_BASE = dict(
    n_estimators=1000, num_leaves=63, learning_rate=0.04,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
    objective="mae", n_jobs=4, random_state=42, verbose=-1,
)


def fetch_chembl_target(chembl_id, label, cache_path, max_records=50_000):
    """Fetch activity data for a ChEMBL target, with caching."""
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"  [{label}] loaded from cache: {len(df)} rows")
        return df

    print(f"  [{label}] fetching from ChEMBL (target={chembl_id})...")
    from chembl_webresource_client.new_client import new_client
    activity = new_client.activity

    records = []
    t0 = time.time()
    try:
        res = activity.filter(
            target_chembl_id=chembl_id,
            standard_type__in=["IC50", "EC50", "Ki", "AC50", "Kd"],
            pchembl_value__isnull=False,
            assay_type="B",  # binding/functional
        ).only(["molecule_chembl_id", "canonical_smiles", "pchembl_value", "standard_type"])

        for i, r in enumerate(res):
            if i >= max_records:
                break
            if r.get("canonical_smiles") and r.get("pchembl_value"):
                records.append({
                    "smiles": r["canonical_smiles"],
                    "pchembl_value": float(r["pchembl_value"]),
                    "standard_type": r.get("standard_type", ""),
                })
            if i % 5000 == 0 and i > 0:
                print(f"    fetched {i} records... ({time.time()-t0:.0f}s)")
    except Exception as e:
        print(f"  [{label}] fetch error: {e}")

    df = pd.DataFrame(records)
    print(f"  [{label}] fetched {len(df)} records in {time.time()-t0:.0f}s")

    if len(df) > 0:
        df.to_parquet(cache_path, index=False)
    return df


def standardize_smiles_col(df, smiles_col="smiles"):
    from rdkit import Chem
    mols, valid = [], []
    for s in df[smiles_col]:
        mol = Chem.MolFromSmiles(str(s)) if pd.notna(s) else None
        if mol:
            mols.append(Chem.MolToSmiles(mol))
            valid.append(True)
        else:
            mols.append(None)
            valid.append(False)
    df = df.copy()
    df["std_smiles"] = mols
    return df[pd.Series(valid)].reset_index(drop=True)


def prepare_external_dataset(df, smiles_col="smiles", val_col="pchembl_value"):
    """Standardize, deduplicate (median per compound), featurize."""
    df = standardize_smiles_col(df, smiles_col)
    df["pec50"] = df[val_col]
    # Median per unique SMILES
    df = df.groupby("std_smiles")["pec50"].median().reset_index()
    print(f"    After dedup: {len(df)} unique compounds")
    X = combined(df["std_smiles"].tolist())
    X = impute(X)
    y = df["pec50"].values
    mask = np.isfinite(y)
    return X[mask], y[mask], df["std_smiles"].values[mask]


def train_analogy_model(X, y, label):
    """Train LGBM on external dataset, return model trained on all data."""
    print(f"  Training {label} model on {len(X)} compounds...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        m = lgb.LGBMRegressor(**LGBM_BASE)
        m.fit(X[tr_idx], y[tr_idx],
              eval_set=[(X[va_idx], y[va_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof[va_idx] = m.predict(X[va_idx])
    print(f"    {label} internal 5-fold MAE: {np.mean(np.abs(oof-y)):.4f}  "
          f"RAE: {rae(y, oof):.4f}")

    # Retrain on all data
    m_full = lgb.LGBMRegressor(**LGBM_BASE)
    m_full.fit(X, y, callbacks=[lgb.log_evaluation(-1)])
    return m_full


def get_features_for(smiles_list, featurizer=combined):
    X = featurizer(smiles_list)
    return impute(X)


def main():
    print("=== nb214: CYP3A4 analogy feature transfer ===\n")

    tr = load_train()
    te_df = load_test()
    tr = add_standard_columns(tr)
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["smiles"].tolist()

    print(f"Train: {len(tr)} | Test: {len(te_df)}")

    # Load base predictions (nb197 grand ensemble)
    oof_base = np.load(DATA_PROCESSED / "oof_nb197_dense_grid.npy").astype(np.float64)
    te_base  = np.load(DATA_PROCESSED / "te_nb197_dense_grid.npy").astype(np.float64)
    print(f"Base nb197: OOF RAE={rae(y_tr, oof_base):.4f}  te_std={te_base.std():.4f}  ratio={te_base.std()/oof_base.std():.3f}\n")

    # ── A. Fetch external assay data ──────────────────────────────────────────
    targets = {
        "CYP3A4": ("CHEMBL340",  DATA_EXTERNAL / "chembl_cyp3a4_activity.parquet"),
        "AhR":    ("CHEMBL3905", DATA_EXTERNAL / "chembl_ahr_activity.parquet"),
        "CYP2C9": ("CHEMBL3577", DATA_EXTERNAL / "chembl_cyp2c9_activity.parquet"),
    }

    analogy_models = {}
    analogy_smiles = {}

    for label, (chembl_id, cache_path) in targets.items():
        print(f"\n── {label} ──")
        raw = fetch_chembl_target(chembl_id, label, cache_path, max_records=60_000)
        if len(raw) < 100:
            print(f"  Too few records ({len(raw)}), skipping {label}")
            continue
        X_ext, y_ext, smi_ext = prepare_external_dataset(raw)
        if len(X_ext) < 100:
            print(f"  After featurization too small, skipping")
            continue
        model = train_analogy_model(X_ext, y_ext, label)
        analogy_models[label] = model
        analogy_smiles[label] = smi_ext
        print(f"  {label}: {len(X_ext)} training compounds → model ready")

    if not analogy_models:
        print("No analogy models built. Check ChEMBL connectivity.")
        return

    # ── B. Generate virtual assay features for PXR train+test ────────────────
    print(f"\n── Generating analogy features for PXR train+test ──")
    X_tr_base = get_features_for(smiles_tr)
    X_te_base = get_features_for(smiles_te)

    analogy_tr_feats = []
    analogy_te_feats = []
    feat_names = []

    for label, model in analogy_models.items():
        pred_tr = model.predict(X_tr_base)
        pred_te = model.predict(X_te_base)
        analogy_tr_feats.append(pred_tr.reshape(-1, 1))
        analogy_te_feats.append(pred_te.reshape(-1, 1))
        feat_names.append(f"virtual_{label}")
        print(f"  {label} train: mean={pred_tr.mean():.2f} std={pred_tr.std():.3f}")
        print(f"  {label} test:  mean={pred_te.mean():.2f} std={pred_te.std():.3f}")

    X_tr_aug = np.hstack([X_tr_base] + analogy_tr_feats)
    X_te_aug = np.hstack([X_te_base] + analogy_te_feats)
    print(f"\nAugmented feature shape: train={X_tr_aug.shape}  test={X_te_aug.shape}")

    # ── C. Scaffold CV on augmented features ─────────────────────────────────
    print("\n── Scaffold 5-fold CV on augmented features ──")
    scaffolds = tr["scaffold"].tolist()
    folds = scaffold_kfold_indices(scaffolds, n_splits=5)

    results = {}
    for desc, X_tr_use in [("base_only", X_tr_base), ("augmented", X_tr_aug)]:
        oof = np.zeros(len(y_tr))
        te_preds = []

        for fold_idx, (tr_idx, va_idx) in enumerate(folds):
            m = lgb.LGBMRegressor(**LGBM_BASE)
            m.fit(X_tr_use[tr_idx], y_tr[tr_idx],
                  eval_set=[(X_tr_use[va_idx], y_tr[va_idx])],
                  callbacks=[lgb.early_stopping(80, verbose=False),
                              lgb.log_evaluation(-1)])
            oof[va_idx] = m.predict(X_tr_use[va_idx])
            te_preds.append(m.predict(X_te_aug if desc == "augmented" else X_te_base))

        te_pred = np.mean(te_preds, axis=0)
        r = rae(y_tr, oof)
        ratio = te_pred.std() / oof.std()
        flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
        print(f"  {desc:15s}: OOF RAE={r:.4f}  te_std={te_pred.std():.4f}  ratio={ratio:.3f}  [{flag}]")
        results[desc] = (oof, te_pred, r, ratio)

    # ── D. Ensemble with nb197 ────────────────────────────────────────────────
    print("\n── Blend augmented LGBM with nb197 ──")
    oof_aug, te_aug, r_aug, ratio_aug = results["augmented"]
    for w in [0.10, 0.20, 0.30, 0.40, 0.50]:
        oof_bl = (1-w)*oof_base + w*oof_aug
        te_bl  = (1-w)*te_base  + w*te_aug
        r_bl   = rae(y_tr, oof_bl)
        ratio_bl = te_bl.std() / oof_bl.std()
        flag = "PASS" if ratio_bl >= COLLAPSE_THRESH else "FAIL"
        print(f"  w={w:.2f}: OOF={r_bl:.4f}  ratio={ratio_bl:.3f}  [{flag}]")

    # ── E. Save best result ───────────────────────────────────────────────────
    # Use augmented standalone if it beats base and passes
    saved = []
    if ratio_aug >= COLLAPSE_THRESH and r_aug < rae(y_tr, oof_base):
        np.save(DATA_PROCESSED / "oof_nb214_cyp3a4_aug.npy", oof_aug)
        np.save(DATA_PROCESSED / "te_nb214_cyp3a4_aug.npy", te_aug)
        sub = pd.DataFrame({"Molecule Name": te_df["name"], "pEC50": te_aug})
        sub.to_csv(SUBMISSIONS / "214_cyp3a4_aug.csv", index=False)
        saved.append("214_cyp3a4_aug")
        print(f"\nSaved 214_cyp3a4_aug (OOF={r_aug:.4f})")

    # Best blend
    best_w, best_r, best_ratio = None, 999, 0
    for w in np.arange(0.05, 0.55, 0.05):
        oof_bl = (1-w)*oof_base + w*oof_aug
        te_bl  = (1-w)*te_base  + w*te_aug
        r_bl   = rae(y_tr, oof_bl)
        ratio_bl = te_bl.std() / oof_bl.std()
        if ratio_bl >= COLLAPSE_THRESH and r_bl < best_r:
            best_r, best_ratio, best_w = r_bl, ratio_bl, w

    if best_w is not None and best_r < rae(y_tr, oof_base):
        oof_bl = (1-best_w)*oof_base + best_w*oof_aug
        te_bl  = (1-best_w)*te_base  + best_w*te_aug
        name = f"214_blend_w{int(best_w*100):02d}"
        np.save(DATA_PROCESSED / f"oof_{name}.npy", oof_bl)
        np.save(DATA_PROCESSED / f"te_{name}.npy", te_bl)
        sub = pd.DataFrame({"Molecule Name": te_df["name"], "pEC50": te_bl})
        sub.to_csv(SUBMISSIONS / f"{name}.csv", index=False)
        saved.append(name)
        print(f"Saved blend {name} (OOF={best_r:.4f}  ratio={best_ratio:.3f})")

    print(f"\n=== Done. {len(saved)} submissions saved. ===")
    if not saved:
        print("No improvement over nb197 base (OOF 0.2976).")
        print("CYP3A4 virtual features did not help — try nb215 multi-NR stack.")


if __name__ == "__main__":
    main()
