"""nb215 -- Multi-NR transfer stack: virtual assay features from all nuclear receptors.

For each NR in our existing external dataset (PPARg 4302, FXR 3185, RXRa 1364,
LXRa 1173, VDR 523 compounds), train an LGBM predictor, then apply it to every
PXR compound to produce a "virtual NR activity" feature.

Insight: the 5-assay NR activity profile of a compound encodes pharmacophore
information that individual descriptors don't capture cleanly. PXR is in the
same superfamily and shares ligand-recognition features with these receptors.

Variants:
  A: 5-NR virtual features appended to Morgan+RDKit (2265 -> 2270 features)
  B: Only the NR features + Morgan+RDKit passed to a dedicated meta-LGBM
  C: Blend of variant A model with nb197 grand ensemble
  D: Include BindingDB NR data for additional compounds per target
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, DATA_EXTERNAL, SUBMISSIONS

COLLAPSE_THRESH = 0.58

LGBM_BASE = dict(
    n_estimators=1000, num_leaves=63, learning_rate=0.04,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
    objective="mae", n_jobs=4, random_state=42, verbose=-1,
)

NR_TARGETS = ["PPARg", "FXR", "RXRa", "LXRa", "VDR", "PXR"]


def load_nr_data():
    """Load and merge ChEMBL NR + BindingDB NR, returning dict target->df."""
    chembl = pd.read_parquet(DATA_EXTERNAL / "chembl_nr_extended.parquet")
    bdb    = pd.read_parquet(DATA_EXTERNAL / "bindingdb_nr_data.parquet")

    # Normalize target names
    target_map_chembl = {
        "PPARg": "PPARg", "FXR": "FXR", "RXRa": "RXRa",
        "LXRa": "LXRa",  "VDR": "VDR",  "PXR": "PXR",
    }
    target_map_bdb = {
        "PPARgamma": "PPARg", "FXR": "FXR", "RXRalpha": "RXRa",
        "LXRalpha": "LXRa",  "VDR": "VDR",  "PXR": "PXR",
        "PPARg": "PPARg", "RXRa": "RXRa", "LXRa": "LXRa",
    }

    chembl["target_norm"] = chembl["target_name"].map(
        lambda x: next((v for k, v in target_map_chembl.items() if k in str(x)), None)
    )
    bdb["target_norm"] = bdb["target_name"].map(
        lambda x: next((v for k, v in target_map_bdb.items() if k in str(x)), None)
    )

    combined_df = pd.concat([
        chembl[["std_smiles", "pec50", "target_norm"]].rename(columns={"target_norm": "target"}),
        bdb[["std_smiles", "pec50", "target_norm"]].rename(columns={"target_norm": "target"}),
    ], ignore_index=True)

    combined_df = combined_df.dropna(subset=["std_smiles", "pec50", "target"])

    result = {}
    for tgt in NR_TARGETS:
        sub = combined_df[combined_df["target"] == tgt].copy()
        # Median per unique SMILES
        sub = sub.groupby("std_smiles")["pec50"].median().reset_index()
        result[tgt] = sub
        print(f"  {tgt}: {len(sub)} unique compounds")

    return result


def featurize_series(smiles_list):
    X = combined(smiles_list)
    return impute(X)


def train_nr_model(smiles, y, label, pxr_train_smiles_set):
    """Train LGBM on NR data excluding PXR training compounds."""
    mask_ext = ~pd.Series(smiles).isin(pxr_train_smiles_set)
    smiles_ext = [s for s, m in zip(smiles, mask_ext) if m]
    y_ext = y[mask_ext.values]

    print(f"  {label}: {len(smiles_ext)} compounds after removing PXR train overlap")
    if len(smiles_ext) < 30:
        print(f"    Too few compounds, skipping {label}")
        return None

    X_ext = featurize_series(smiles_ext)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y_ext))
    for tr_idx, va_idx in kf.split(X_ext):
        m = lgb.LGBMRegressor(**LGBM_BASE)
        m.fit(X_ext[tr_idx], y_ext[tr_idx],
              eval_set=[(X_ext[va_idx], y_ext[va_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof[va_idx] = m.predict(X_ext[va_idx])

    print(f"    Internal MAE={np.mean(np.abs(oof-y_ext)):.4f}  "
          f"pec50 range [{y_ext.min():.1f}, {y_ext.max():.1f}]")

    m_full = lgb.LGBMRegressor(**LGBM_BASE)
    m_full.fit(X_ext, y_ext, callbacks=[lgb.log_evaluation(-1)])
    return m_full


def main():
    print("=== nb215: Multi-NR transfer stack ===\n")

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

    # ── Load NR data ──────────────────────────────────────────────────────────
    print("Loading NR external data...")
    nr_data = load_nr_data()

    # ── Featurize PXR train+test once ────────────────────────────────────────
    print("\nFeaturizing PXR train+test...")
    X_tr_base = featurize_series(smiles_tr)
    X_te_base = featurize_series(smiles_te)

    # ── Train per-NR models and generate virtual features ────────────────────
    print("\nTraining per-NR models...")
    virtual_tr = []
    virtual_te = []
    feat_labels = []

    for tgt in ["PPARg", "FXR", "RXRa", "LXRa", "VDR"]:  # exclude PXR (avoid leakage)
        df_nr = nr_data.get(tgt, pd.DataFrame())
        if len(df_nr) < 30:
            print(f"  {tgt}: skipping (too few compounds)")
            continue

        smiles_nr = df_nr["std_smiles"].tolist()
        y_nr = df_nr["pec50"].values.astype(np.float64)

        model = train_nr_model(smiles_nr, y_nr, tgt, pxr_train_set)
        if model is None:
            continue

        pred_tr = model.predict(X_tr_base)
        pred_te = model.predict(X_te_base)

        virtual_tr.append(pred_tr)
        virtual_te.append(pred_te)
        feat_labels.append(tgt)

        # Correlation with PXR train labels
        from scipy.stats import spearmanr
        rho, pval = spearmanr(pred_tr, y_tr)
        print(f"    virtual_{tgt} vs PXR train: Spearman rho={rho:.3f} (p={pval:.3e})")

    print(f"\nBuilt {len(feat_labels)} virtual NR features: {feat_labels}")

    if not feat_labels:
        print("No NR models built.")
        return

    # Stack features
    V_tr = np.column_stack(virtual_tr)
    V_te = np.column_stack(virtual_te)
    X_tr_aug = np.hstack([X_tr_base, V_tr])
    X_te_aug = np.hstack([X_te_base, V_te])
    X_tr_nr_only = np.hstack([V_tr])   # just NR features for meta

    print(f"Feature shapes: base={X_tr_base.shape[1]}  aug={X_tr_aug.shape[1]}  nr_only={X_tr_nr_only.shape[1]}")

    # ── Scaffold 5-fold CV ────────────────────────────────────────────────────
    print("\n── Scaffold 5-fold CV ──")
    scaffolds = tr["scaffold"].tolist()
    folds = scaffold_kfold_indices(scaffolds, n_splits=5)

    configs = [
        ("base_only",    X_tr_base, X_te_base),
        ("nr_aug",       X_tr_aug,  X_te_aug),
    ]

    cv_results = {}
    for name, X_tr_use, X_te_use in configs:
        oof = np.zeros(len(y_tr))
        te_preds = []
        for tr_idx, va_idx in folds:
            m = lgb.LGBMRegressor(**LGBM_BASE)
            m.fit(X_tr_use[tr_idx], y_tr[tr_idx],
                  eval_set=[(X_tr_use[va_idx], y_tr[va_idx])],
                  callbacks=[lgb.early_stopping(80, verbose=False),
                              lgb.log_evaluation(-1)])
            oof[va_idx] = m.predict(X_tr_use[va_idx])
            te_preds.append(m.predict(X_te_use))

        te_pred = np.mean(te_preds, axis=0)
        r = rae(y_tr, oof)
        ratio = te_pred.std() / oof.std()
        flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
        print(f"  {name:15s}: OOF RAE={r:.4f}  te_std={te_pred.std():.4f}  ratio={ratio:.3f}  [{flag}]")
        cv_results[name] = (oof, te_pred, r, ratio)

    # ── Blend with nb197 ─────────────────────────────────────────────────────
    print("\n── Blend nr_aug model with nb197 ──")
    oof_aug, te_aug, r_aug, ratio_aug = cv_results["nr_aug"]
    best_blend = None
    best_r_blend = 999

    for w in np.arange(0.05, 0.65, 0.05):
        oof_bl = (1-w)*oof_base + w*oof_aug
        te_bl  = (1-w)*te_base  + w*te_aug
        r_bl   = rae(y_tr, oof_bl)
        ratio_bl = te_bl.std() / oof_bl.std()
        flag = "PASS" if ratio_bl >= COLLAPSE_THRESH else "FAIL"
        print(f"  w={w:.2f}: OOF={r_bl:.4f}  ratio={ratio_bl:.3f}  [{flag}]")
        if ratio_bl >= COLLAPSE_THRESH and r_bl < best_r_blend:
            best_r_blend = r_bl
            best_blend = (w, oof_bl, te_bl, ratio_bl)

    # ── Save ─────────────────────────────────────────────────────────────────
    saved = []

    if ratio_aug >= COLLAPSE_THRESH and r_aug < base_rae:
        np.save(DATA_PROCESSED / "oof_nb215_nr_aug.npy", oof_aug)
        np.save(DATA_PROCESSED / "te_nb215_nr_aug.npy", te_aug)
        sub = pd.DataFrame({"Molecule Name": te_df["name"], "pEC50": te_aug})
        sub.to_csv(SUBMISSIONS / "215_nr_aug.csv", index=False)
        saved.append(f"215_nr_aug (OOF={r_aug:.4f} ratio={ratio_aug:.3f})")

    if best_blend and best_r_blend < base_rae:
        w_b, oof_b, te_b, ratio_b = best_blend
        name = f"215_blend_w{int(w_b*100):02d}"
        np.save(DATA_PROCESSED / f"oof_{name}.npy", oof_b)
        np.save(DATA_PROCESSED / f"te_{name}.npy", te_b)
        sub = pd.DataFrame({"Molecule Name": te_df["name"], "pEC50": te_b})
        sub.to_csv(SUBMISSIONS / f"{name}.csv", index=False)
        saved.append(f"{name} (OOF={best_r_blend:.4f} ratio={ratio_b:.3f})")

    print(f"\n=== Done. Saved: {saved or ['none (no improvement over 0.2976)']}")


if __name__ == "__main__":
    main()
