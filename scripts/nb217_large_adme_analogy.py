"""nb217 -- Large ADME analogy: fix nb214's assay_type filter bug, get real CYP3A4 data.

nb214 fetched only 907 CYP3A4 records because assay_type="B" (binding) excludes
all ADME-type CYP assays. The correct type for CYP3A4 metabolism assays is "A" (ADME).
Without filtering, ChEMBL CYP3A4 has ~15,000-50,000+ records.

Strategy:
  1. Fetch CYP3A4 (CHEMBL340) ALL assay types → large ADME dataset
  2. Fetch CYP1A2 (CHEMBL1832) ALL types → second major P450
  3. Fetch MDR1/P-gp (CHEMBL4302530) → PXR directly induces MDR1 expression
  4. For each, compute empirical Spearman rho on overlapping compounds with PXR train
  5. Only build feature models for assays with rho > 0.15
  6. Generate virtual features, run scaffold CV, blend with nb197

Key fix: remove assay_type="B" filter, use all assay types (A=ADME, B=binding, F=functional).
"""
import os, sys, warnings, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, DATA_EXTERNAL, SUBMISSIONS

COLLAPSE_THRESH = 0.58
MIN_CORR_TO_USE = 0.10   # minimum Spearman rho with PXR to include as feature

LGBM_BASE = dict(
    n_estimators=1500, num_leaves=63, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
    objective="mae", n_jobs=4, random_state=42, verbose=-1,
)

TARGETS = {
    "CYP3A4":  "CHEMBL340",    # main CYP P450; PXR activates CYP3A4 transcription
    "CYP1A2":  "CHEMBL1832",   # xenobiotic-induced P450; AhR/PXR crosstalk
    "MDR1":    "CHEMBL4302530",# PXR directly induces MDR1/ABCB1 expression
    "PXR":     "CHEMBL3401485",# additional PXR data from ChEMBL (different assays)
}


def fetch_target_all_types(chembl_id, label, cache_path, max_records=100_000):
    """Fetch activity for a target across ALL assay types (no type filter)."""
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"  [{label}] loaded from cache: {len(df)} rows")
        return df

    print(f"  [{label}] fetching (no assay_type filter, max={max_records})...")
    from chembl_webresource_client.new_client import new_client
    activity = new_client.activity

    records = []
    t0 = time.time()
    try:
        res = activity.filter(
            target_chembl_id=chembl_id,
            standard_type__in=["IC50", "EC50", "Ki", "AC50", "Kd", "GI50", "potency"],
            pchembl_value__isnull=False,
            # NO assay_type filter — gets A (ADME), B (binding), F (functional)
        ).only(["canonical_smiles", "pchembl_value", "standard_type", "assay_type"])

        for i, r in enumerate(res):
            if i >= max_records:
                break
            if r.get("canonical_smiles") and r.get("pchembl_value"):
                records.append({
                    "smiles": r["canonical_smiles"],
                    "pchembl_value": float(r["pchembl_value"]),
                    "standard_type": r.get("standard_type", ""),
                    "assay_type": r.get("assay_type", ""),
                })
            if i % 5000 == 0 and i > 0:
                print(f"    {i} records... ({time.time()-t0:.0f}s)")
    except Exception as e:
        print(f"  [{label}] error: {e}")

    df = pd.DataFrame(records)
    print(f"  [{label}] fetched {len(df)} records in {time.time()-t0:.0f}s")
    if len(df) > 0:
        df.to_parquet(cache_path, index=False)
    return df


def standardize_and_dedup(df, smiles_col="smiles", val_col="pchembl_value"):
    """Canonical SMILES, remove unparseable, median-aggregate duplicates."""
    valid_smiles, valid_mask = [], []
    for s in df[smiles_col]:
        try:
            mol = Chem.MolFromSmiles(str(s))
            if mol:
                valid_smiles.append(Chem.MolToSmiles(mol))
                valid_mask.append(True)
            else:
                valid_smiles.append(None)
                valid_mask.append(False)
        except Exception:
            valid_smiles.append(None)
            valid_mask.append(False)
    df = df.copy()
    df["std_smiles"] = valid_smiles
    df = df[valid_mask].reset_index(drop=True)
    df["pec50"] = df[val_col]
    df = df.groupby("std_smiles")["pec50"].median().reset_index()
    return df


def compute_overlap_correlation(ext_smiles_set, ext_df, pxr_df):
    """Compute Spearman rho on compounds in both PXR train and external dataset."""
    overlap_smi = set(pxr_df["std_smiles"]) & ext_smiles_set
    if len(overlap_smi) < 10:
        print(f"    Overlap too small ({len(overlap_smi)} compounds), rho=N/A")
        return 0.0, len(overlap_smi)

    pxr_vals = pxr_df.set_index("std_smiles")["pec50"].reindex(list(overlap_smi))
    ext_vals = ext_df.set_index("std_smiles")["pec50"].reindex(list(overlap_smi))
    both = pd.DataFrame({"pxr": pxr_vals, "ext": ext_vals}).dropna()
    if len(both) < 5:
        return 0.0, len(both)
    rho, pval = spearmanr(both["pxr"], both["ext"])
    print(f"    Overlap: {len(both)} compounds, Spearman rho={rho:.3f} (p={pval:.2e})")
    return rho, len(both)


def train_model_on_external(ext_df, label):
    """Train LGBM on external dataset, return model and its training info."""
    print(f"  Training {label} model on {len(ext_df)} compounds...")
    X = combined(ext_df["std_smiles"].tolist())
    X = impute(X)
    y = ext_df["pec50"].values.astype(np.float64)
    mask = np.isfinite(y)
    X, y = X[mask], y[mask]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    for tr_idx, va_idx in kf.split(X):
        m = lgb.LGBMRegressor(**LGBM_BASE)
        m.fit(X[tr_idx], y[tr_idx],
              eval_set=[(X[va_idx], y[va_idx])],
              callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(-1)])
        oof[va_idx] = m.predict(X[va_idx])
    print(f"    Internal 5-fold MAE={np.mean(np.abs(oof-y)):.4f}  RAE={rae(y, oof):.4f}")

    m_full = lgb.LGBMRegressor(**LGBM_BASE)
    m_full.fit(X, y, callbacks=[lgb.log_evaluation(-1)])
    return m_full


def main():
    print("=== nb217: Large ADME analogy (no assay_type filter) ===\n")

    tr = load_train()
    te_df = load_test()
    tr = add_standard_columns(tr)
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["smiles"].tolist()

    pxr_df = tr[["std_smiles", "pec50"]].copy()

    oof_base = np.load(DATA_PROCESSED / "oof_nb197_dense_grid.npy").astype(np.float64)
    te_base  = np.load(DATA_PROCESSED / "te_nb197_dense_grid.npy").astype(np.float64)
    base_rae = rae(y_tr, oof_base)
    print(f"Base nb197: OOF RAE={base_rae:.4f}  ratio={te_base.std()/oof_base.std():.3f}\n")

    print("Featurizing PXR compounds...")
    X_tr_base = combined(smiles_tr); X_tr_base = impute(X_tr_base)
    X_te_base = combined(smiles_te); X_te_base = impute(X_te_base)

    # ── Fetch all targets ─────────────────────────────────────────────────────
    analogy_models = {}
    analogy_rhos   = {}

    for label, chembl_id in TARGETS.items():
        cache = DATA_EXTERNAL / f"chembl_{label.lower()}_alltype.parquet"
        print(f"\n── {label} (CHEMBL={chembl_id}) ──")
        raw = fetch_target_all_types(chembl_id, label, cache, max_records=100_000)
        if len(raw) < 50:
            print(f"  Skipping {label}: too few records")
            continue

        ext_df = standardize_and_dedup(raw)
        print(f"  After dedup: {len(ext_df)} unique compounds")

        # Exclude PXR training set compounds (prevent leakage)
        pxr_train_set = set(smiles_tr)
        n_before = len(ext_df)
        ext_df = ext_df[~ext_df["std_smiles"].isin(pxr_train_set)].reset_index(drop=True)
        print(f"  After removing PXR train overlap: {len(ext_df)} compounds (removed {n_before-len(ext_df)})")

        rho, n_overlap = compute_overlap_correlation(set(ext_df["std_smiles"]), ext_df, pxr_df)
        analogy_rhos[label] = rho

        if abs(rho) < MIN_CORR_TO_USE and n_overlap >= 10:
            print(f"  rho={rho:.3f} < threshold ({MIN_CORR_TO_USE}), skipping model")
            continue
        if len(ext_df) < 50:
            print(f"  Too few compounds after filtering")
            continue

        model = train_model_on_external(ext_df, label)
        analogy_models[label] = model
        print(f"  {label} model built on {len(ext_df)} compounds")

    print(f"\n\nCorrelation summary:")
    for label, rho in analogy_rhos.items():
        used = "USED" if label in analogy_models else "skipped"
        print(f"  {label}: rho={rho:.3f}  [{used}]")

    if not analogy_models:
        print("No analogy models built (all below correlation threshold or too few records).")
        print("Consider lowering MIN_CORR_TO_USE or fetching more data.")
        return

    # ── Generate virtual features ─────────────────────────────────────────────
    print(f"\n── Generating {len(analogy_models)} virtual assay features ──")
    V_tr, V_te = [], []
    for label, model in analogy_models.items():
        p_tr = model.predict(X_tr_base)
        p_te = model.predict(X_te_base)
        rho_applied, _ = spearmanr(p_tr, y_tr)
        print(f"  virtual_{label}: train_std={p_tr.std():.3f}  rho_with_PXR={rho_applied:.3f}")
        V_tr.append(p_tr); V_te.append(p_te)

    X_tr_aug = np.hstack([X_tr_base] + [v.reshape(-1,1) for v in V_tr])
    X_te_aug = np.hstack([X_te_base] + [v.reshape(-1,1) for v in V_te])

    # ── Scaffold 5-fold CV ────────────────────────────────────────────────────
    print(f"\n── Scaffold 5-fold CV (aug shape={X_tr_aug.shape}) ──")
    scaffolds = tr["scaffold"].tolist()
    folds = scaffold_kfold_indices(scaffolds, n_splits=5)

    cv_results = {}
    for name, X_tr_use, X_te_use in [
        ("base_only",  X_tr_base, X_te_base),
        ("adme_aug",   X_tr_aug,  X_te_aug),
    ]:
        oof = np.zeros(len(y_tr))
        te_preds = []
        for tr_idx, va_idx in folds:
            m = lgb.LGBMRegressor(**LGBM_BASE)
            m.fit(X_tr_use[tr_idx], y_tr[tr_idx],
                  eval_set=[(X_tr_use[va_idx], y_tr[va_idx])],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
            oof[va_idx] = m.predict(X_tr_use[va_idx])
            te_preds.append(m.predict(X_te_use))
        te_pred = np.mean(te_preds, axis=0)
        r = rae(y_tr, oof)
        ratio = te_pred.std() / oof.std()
        flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
        print(f"  {name:12s}: OOF RAE={r:.4f}  te_std={te_pred.std():.4f}  ratio={ratio:.3f}  [{flag}]")
        cv_results[name] = (oof, te_pred, r, ratio)

    # ── Blend with nb197 ─────────────────────────────────────────────────────
    print("\n── Blend adme_aug with nb197 ──")
    oof_aug, te_aug, r_aug, ratio_aug = cv_results["adme_aug"]
    best_blend, best_r_bl = None, 999

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
        np.save(DATA_PROCESSED / "oof_nb217_adme_aug.npy", oof_aug)
        np.save(DATA_PROCESSED / "te_nb217_adme_aug.npy", te_aug)
        sub = pd.DataFrame({"Molecule Name": te_df["name"], "pEC50": te_aug})
        sub.to_csv(SUBMISSIONS / "217_adme_aug.csv", index=False)
        saved.append(f"217_adme_aug (OOF={r_aug:.4f} ratio={ratio_aug:.3f})")

    if best_blend and best_r_bl < base_rae:
        w_b, oof_b, te_b, ratio_b = best_blend
        name = f"217_blend_w{int(w_b*100):02d}"
        np.save(DATA_PROCESSED / f"oof_{name}.npy", oof_b)
        np.save(DATA_PROCESSED / f"te_{name}.npy", te_b)
        sub = pd.DataFrame({"Molecule Name": te_df["name"], "pEC50": te_b})
        sub.to_csv(SUBMISSIONS / f"{name}.csv", index=False)
        saved.append(f"{name} (OOF={best_r_bl:.4f} ratio={ratio_b:.3f})")

    print(f"\n=== Done. Saved: {saved or ['none']}")


if __name__ == "__main__":
    main()
