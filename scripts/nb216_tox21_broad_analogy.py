"""nb216 -- Tox21 + broad ADMET analogy: download large multi-assay panel,
train per-assay classifiers/regressors, use predictions as PXR features.

Tox21 NR panel (8014 compounds): binary active/inactive for
  NR-AhR, NR-AR, NR-AR-LBD, NR-ER, NR-ER-LBD, NR-PPAR-gamma

AhR is the most relevant (xenosensor with overlapping ligand profile to PXR).
PPAR-gamma overlaps in ligand pharmacophore with RXRa-heterodimerization partners.

Additionally, fetch from MoleculeNet/TDC:
  - BACE (protease, structural signal)
  - BBBP (membrane permeability, logP signal)
  - Lipophilicity (logD, strong structural correlate to PXR agonism)
  - ESOL (aqueous solubility, inverse of lipophilicity)

The insight: PXR agonists are lipophilic, membrane-permeable, bulky compounds.
ADMET endpoints encoding these physical properties provide orthogonal signal
to the fingerprint features already in the model.

Data download: MoleculeNet CSVs directly from their public S3 bucket.
"""
import os, sys, warnings, urllib.request
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, DATA_EXTERNAL, SUBMISSIONS

COLLAPSE_THRESH = 0.58

LGBM_BASE = dict(
    n_estimators=800, num_leaves=63, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
    objective="mae", n_jobs=4, random_state=42, verbose=-1,
)
LGBM_CLASS = dict(
    n_estimators=500, num_leaves=31, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
    objective="binary", n_jobs=4, random_state=42, verbose=-1,
)

DATASETS = {
    "lipophilicity": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
        "smiles_col": "smiles", "label_col": "exp", "task": "regression",
    },
    "esol": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
        "smiles_col": "smiles", "label_col": "measured log solubility in mols per litre",
        "task": "regression",
    },
    "bbbp": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
        "smiles_col": "smiles", "label_col": "p_np", "task": "classification",
    },
    "tox21_ahr": None,   # loaded from local parquet
}


def download_dataset(name, cfg, cache_dir):
    path = cache_dir / f"{name}.csv"
    if not path.exists():
        print(f"  Downloading {name}...")
        urllib.request.urlretrieve(cfg["url"], path)
    df = pd.read_csv(path)
    print(f"  {name}: {len(df)} rows, cols={list(df.columns)}")
    return df


def featurize_smiles(smiles_list):
    X = combined(smiles_list)
    return impute(X)


def standardize_col(df, smiles_col):
    from rdkit import Chem
    valid_mask = []
    canon = []
    for s in df[smiles_col]:
        try:
            mol = Chem.MolFromSmiles(str(s))
            if mol:
                canon.append(Chem.MolToSmiles(mol))
                valid_mask.append(True)
            else:
                canon.append(None)
                valid_mask.append(False)
        except Exception:
            canon.append(None)
            valid_mask.append(False)
    df = df.copy()
    df["std_smiles"] = canon
    return df[valid_mask].reset_index(drop=True)


def train_analogy_model(X, y, task, label, pxr_train_set, smiles):
    """Train model on external dataset, return fitted model."""
    mask_ext = ~pd.Series(smiles).isin(pxr_train_set)
    X_ext = X[mask_ext.values]
    y_ext = y[mask_ext.values]
    print(f"  {label}: {len(X_ext)} ext compounds (removed {mask_ext.sum()==False})")

    if len(X_ext) < 50:
        return None

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y_ext))
    cfg = LGBM_BASE if task == "regression" else LGBM_CLASS

    for tr_idx, va_idx in kf.split(X_ext):
        m = lgb.LGBMRegressor(**cfg) if task == "regression" else lgb.LGBMClassifier(**cfg)
        m.fit(X_ext[tr_idx], y_ext[tr_idx],
              eval_set=[(X_ext[va_idx], y_ext[va_idx])],
              callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)])
        if task == "regression":
            oof[va_idx] = m.predict(X_ext[va_idx])
        else:
            oof[va_idx] = m.predict_proba(X_ext[va_idx])[:, 1]

    if task == "regression":
        print(f"    Internal MAE={np.mean(np.abs(oof-y_ext)):.4f}")
    else:
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(y_ext, oof)
            print(f"    Internal AUC={auc:.4f}")
        except Exception:
            pass

    m_full = lgb.LGBMRegressor(**cfg) if task == "regression" else lgb.LGBMClassifier(**cfg)
    m_full.fit(X_ext, y_ext, callbacks=[lgb.log_evaluation(-1)])
    return m_full, task


def load_tox21_ahr():
    """Load Tox21 NR data and return AhR + PPAR-gamma columns with SMILES."""
    # Our tox21_nr_data.parquet has labels but may lack SMILES; try to load
    df = pd.read_parquet(DATA_EXTERNAL / "tox21_nr_data.parquet")
    print(f"  Tox21 parquet columns: {list(df.columns)}")

    # Try to get SMILES column
    if "smiles" in df.columns or "SMILES" in df.columns:
        smiles_col = "smiles" if "smiles" in df.columns else "SMILES"
        return df, smiles_col

    # No SMILES — download from MoleculeNet
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
    cache = DATA_EXTERNAL / "tox21_moleculenet.csv.gz"
    if not cache.exists():
        print("  Downloading Tox21 from MoleculeNet...")
        urllib.request.urlretrieve(url, cache)
    df_full = pd.read_csv(cache)
    print(f"  Tox21 MoleculeNet: {len(df_full)} rows, cols={list(df_full.columns)[:10]}")
    return df_full, "smiles" if "smiles" in df_full.columns else df_full.columns[0]


def main():
    print("=== nb216: Tox21 + broad ADMET analogy features ===\n")

    tr = load_train()
    te_df = load_test()
    tr = add_standard_columns(tr)
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["smiles"].tolist()
    pxr_train_set = set(smiles_tr)

    oof_base = np.load(DATA_PROCESSED / "oof_nb197_dense_grid.npy").astype(np.float64)
    te_base  = np.load(DATA_PROCESSED / "te_nb197_dense_grid.npy").astype(np.float64)
    base_rae = rae(y_tr, oof_base)
    print(f"Base nb197: OOF RAE={base_rae:.4f}  ratio={te_base.std()/oof_base.std():.3f}\n")

    # ── Featurize PXR compounds ───────────────────────────────────────────────
    print("Featurizing PXR train+test...")
    X_tr_base = featurize_smiles(smiles_tr)
    X_te_base = featurize_smiles(smiles_te)

    # ── Build analogy models ──────────────────────────────────────────────────
    analogy_tr = []
    analogy_te = []
    built = []

    # 1. Lipophilicity (logD) — strong correlate to PXR agonism
    print("\n── Lipophilicity (logD) ──")
    try:
        lipo_df = download_dataset("lipophilicity", DATASETS["lipophilicity"], DATA_EXTERNAL)
        lipo_df = standardize_col(lipo_df, "smiles")
        lipo_df = lipo_df.dropna(subset=["exp"])
        X_lipo = featurize_smiles(lipo_df["std_smiles"].tolist())
        result = train_analogy_model(X_lipo, lipo_df["exp"].values, "regression",
                                     "Lipophilicity", pxr_train_set, lipo_df["std_smiles"].tolist())
        if result:
            m, task = result
            pred_tr = m.predict(X_tr_base)
            pred_te = m.predict(X_te_base)
            rho, _ = spearmanr(pred_tr, y_tr)
            print(f"    virtual_logD vs PXR train: Spearman rho={rho:.3f}")
            analogy_tr.append(pred_tr); analogy_te.append(pred_te); built.append("logD")
    except Exception as e:
        print(f"  Lipophilicity failed: {e}")

    # 2. ESOL (aqueous solubility, inverse of logP)
    print("\n── ESOL (solubility) ──")
    try:
        esol_df = download_dataset("esol", DATASETS["esol"], DATA_EXTERNAL)
        label_col = "measured log solubility in mols per litre"
        esol_df = standardize_col(esol_df, "smiles")
        esol_df = esol_df.dropna(subset=[label_col])
        X_esol = featurize_smiles(esol_df["std_smiles"].tolist())
        result = train_analogy_model(X_esol, esol_df[label_col].values, "regression",
                                     "ESOL", pxr_train_set, esol_df["std_smiles"].tolist())
        if result:
            m, task = result
            pred_tr = m.predict(X_tr_base)
            pred_te = m.predict(X_te_base)
            rho, _ = spearmanr(pred_tr, y_tr)
            print(f"    virtual_ESOL vs PXR train: Spearman rho={rho:.3f}")
            analogy_tr.append(pred_tr); analogy_te.append(pred_te); built.append("ESOL")
    except Exception as e:
        print(f"  ESOL failed: {e}")

    # 3. BBBP (blood-brain barrier permeability)
    print("\n── BBBP (BBB permeability) ──")
    try:
        bbbp_df = download_dataset("bbbp", DATASETS["bbbp"], DATA_EXTERNAL)
        bbbp_df = standardize_col(bbbp_df, "smiles")
        bbbp_df = bbbp_df.dropna(subset=["p_np"])
        X_bbbp = featurize_smiles(bbbp_df["std_smiles"].tolist())
        result = train_analogy_model(X_bbbp, bbbp_df["p_np"].values.astype(float),
                                     "classification", "BBBP", pxr_train_set,
                                     bbbp_df["std_smiles"].tolist())
        if result:
            m, task = result
            pred_tr = m.predict_proba(X_tr_base)[:, 1]
            pred_te = m.predict_proba(X_te_base)[:, 1]
            rho, _ = spearmanr(pred_tr, y_tr)
            print(f"    virtual_BBBP vs PXR train: Spearman rho={rho:.3f}")
            analogy_tr.append(pred_tr); analogy_te.append(pred_te); built.append("BBBP")
    except Exception as e:
        print(f"  BBBP failed: {e}")

    # 4. Tox21 AhR + PPAR-gamma (nuclear receptor xenosensor panel)
    print("\n── Tox21 NR (AhR + PPAR-gamma) ──")
    try:
        tox_df, smiles_col = load_tox21_ahr()
        tox_df = standardize_col(tox_df, smiles_col)

        for nr_col in ["NR-AhR", "NR-PPAR-gamma"]:
            if nr_col not in tox_df.columns:
                print(f"    {nr_col} not found, skipping")
                continue
            sub = tox_df.dropna(subset=[nr_col, "std_smiles"]).copy()
            sub[nr_col] = sub[nr_col].astype(float)
            X_tox = featurize_smiles(sub["std_smiles"].tolist())
            result = train_analogy_model(X_tox, sub[nr_col].values, "classification",
                                         nr_col, pxr_train_set, sub["std_smiles"].tolist())
            if result:
                m, task = result
                pred_tr = m.predict_proba(X_tr_base)[:, 1]
                pred_te = m.predict_proba(X_te_base)[:, 1]
                rho, _ = spearmanr(pred_tr, y_tr)
                print(f"    virtual_{nr_col} vs PXR train: Spearman rho={rho:.3f}")
                analogy_tr.append(pred_tr); analogy_te.append(pred_te)
                built.append(nr_col)
    except Exception as e:
        print(f"  Tox21 failed: {e}")

    print(f"\nBuilt {len(built)} analogy features: {built}")
    if not built:
        print("No analogy features built. Check network access.")
        return

    # ── Stack and evaluate ────────────────────────────────────────────────────
    V_tr = np.column_stack(analogy_tr)
    V_te = np.column_stack(analogy_te)
    X_tr_aug = np.hstack([X_tr_base, V_tr])
    X_te_aug = np.hstack([X_te_base, V_te])

    print(f"\nAugmented features: {X_tr_aug.shape[1]} ({X_tr_base.shape[1]} base + {V_tr.shape[1]} analogy)")

    scaffolds = tr["scaffold"].tolist()
    folds = scaffold_kfold_indices(scaffolds, n_splits=5)

    print("\n── Scaffold 5-fold CV ──")
    cv_results = {}
    for name, X_tr_use, X_te_use in [
        ("base_only", X_tr_base, X_te_base),
        ("admet_aug", X_tr_aug, X_te_aug),
    ]:
        oof = np.zeros(len(y_tr))
        te_preds = []
        for tr_idx, va_idx in folds:
            m = lgb.LGBMRegressor(**LGBM_BASE)
            m.fit(X_tr_use[tr_idx], y_tr[tr_idx],
                  eval_set=[(X_tr_use[va_idx], y_tr[va_idx])],
                  callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
            oof[va_idx] = m.predict(X_tr_use[va_idx])
            te_preds.append(m.predict(X_te_use))
        te_pred = np.mean(te_preds, axis=0)
        r = rae(y_tr, oof)
        ratio = te_pred.std() / oof.std()
        flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
        print(f"  {name:12s}: OOF RAE={r:.4f}  ratio={ratio:.3f}  [{flag}]")
        cv_results[name] = (oof, te_pred, r, ratio)

    # ── Blend with nb197 ─────────────────────────────────────────────────────
    print("\n── Blend admet_aug with nb197 ──")
    oof_aug, te_aug, r_aug, ratio_aug = cv_results["admet_aug"]
    best_blend = None; best_r_bl = 999

    for w in np.arange(0.05, 0.65, 0.05):
        oof_bl = (1-w)*oof_base + w*oof_aug
        te_bl  = (1-w)*te_base  + w*te_aug
        r_bl   = rae(y_tr, oof_bl)
        ratio_bl = te_bl.std() / oof_bl.std()
        flag = "PASS" if ratio_bl >= COLLAPSE_THRESH else "FAIL"
        print(f"  w={w:.2f}: OOF={r_bl:.4f}  ratio={ratio_bl:.3f}  [{flag}]")
        if ratio_bl >= COLLAPSE_THRESH and r_bl < best_r_bl:
            best_r_bl = r_bl; best_blend = (w, oof_bl, te_bl, ratio_bl)

    # ── Save ─────────────────────────────────────────────────────────────────
    saved = []
    if ratio_aug >= COLLAPSE_THRESH and r_aug < base_rae:
        np.save(DATA_PROCESSED / "oof_nb216_admet_aug.npy", oof_aug)
        np.save(DATA_PROCESSED / "te_nb216_admet_aug.npy", te_aug)
        sub = pd.DataFrame({"Molecule Name": te_df["name"], "pEC50": te_aug})
        sub.to_csv(SUBMISSIONS / "216_admet_aug.csv", index=False)
        saved.append(f"216_admet_aug (OOF={r_aug:.4f})")

    if best_blend and best_r_bl < base_rae:
        w_b, oof_b, te_b, ratio_b = best_blend
        name = f"216_blend_w{int(w_b*100):02d}"
        np.save(DATA_PROCESSED / f"oof_{name}.npy", oof_b)
        np.save(DATA_PROCESSED / f"te_{name}.npy", te_b)
        sub = pd.DataFrame({"Molecule Name": te_df["name"], "pEC50": te_b})
        sub.to_csv(SUBMISSIONS / f"{name}.csv", index=False)
        saved.append(f"{name} (OOF={best_r_bl:.4f})")

    print(f"\n=== Done. Saved: {saved or ['none']}")
    print(f"\nFeature correlation summary with PXR train labels:")
    for name, feat_tr in zip(built, analogy_tr):
        rho, pval = spearmanr(feat_tr, y_tr)
        print(f"  {name}: rho={rho:.3f} (p={pval:.2e})")


if __name__ == "__main__":
    main()
