"""nb218 -- Direct experimental lookup: retrieve actual ChEMBL measurements
for PXR training/test compounds, not predicted virtual features.

The problem with nb214-217: virtual features (predicted from external models)
are redundant with what LGBM already extracts from fingerprints.

The RIGHT analogy: retrieve ACTUAL EXPERIMENTAL measurements from public databases
for the exact compounds in our dataset (or their close structural analogs).
These are measurements that fingerprints CANNOT replicate.

Strategy:
  1. For each compound in PXR train+test, query ChEMBL for ANY recorded activity
     (logD, CYP3A4 IC50, PPARg EC50, hERG, etc.) using InChIKey or SMILES
  2. This gives us experimental measurements for a subset of compounds
  3. For compounds without direct measurements, impute from k=3 nearest neighbors
     that DO have measurements (Tanimoto-weighted average)
  4. Add these as per-compound experimental features

Additional: fetch Papyrus PXR data beyond what we have (papyrus has more PXR
assay data aggregated from multiple sources including patented compounds).

Note: this is NOT "training on external data" — it's enriching the FEATURES
of our existing training/test set with experimental measurements.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from scipy.stats import spearmanr

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, DATA_EXTERNAL, SUBMISSIONS

COLLAPSE_THRESH = 0.58

LGBM_BASE = dict(
    n_estimators=1500, num_leaves=63, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
    objective="mae", n_jobs=4, random_state=42, verbose=-1,
)

# Which ChEMBL activity types to retrieve per compound
QUERY_TYPES = ["logD", "logP", "pKa", "Solubility", "Permeability"]
QUERY_ASSAY_TYPES = ["A"]   # ADME assays


def get_mol_fp(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)


def tanimoto_sim(fp1, fp2):
    if fp1 is None or fp2 is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def fetch_compound_activities_by_inchikey(inchikeys, label, cache_path):
    """Fetch all recorded activities for a list of compounds by InChIKey."""
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"  [{label}] loaded from cache: {len(df)} rows covering {df['inchikey'].nunique()} compounds")
        return df

    from chembl_webresource_client.new_client import new_client
    molecule = new_client.molecule
    activity = new_client.activity

    print(f"  [{label}] querying {len(inchikeys)} compounds for experimental measurements...")
    records = []
    batch_size = 50

    for i in range(0, len(inchikeys), batch_size):
        batch = inchikeys[i:i+batch_size]
        try:
            # Get ChEMBL IDs for these InChIKeys
            mols = molecule.filter(
                molecule_structures__standard_inchi_key__in=batch
            ).only(["molecule_chembl_id", "molecule_structures"])

            chembl_ids = []
            ik_to_chembl = {}
            for mol_rec in mols:
                cid = mol_rec.get("molecule_chembl_id")
                ik = mol_rec.get("molecule_structures", {}).get("standard_inchi_key", "")
                if cid and ik:
                    chembl_ids.append(cid)
                    ik_to_chembl[ik] = cid

            if not chembl_ids:
                continue

            # Get ADME activities for these compounds
            acts = activity.filter(
                molecule_chembl_id__in=chembl_ids,
                assay_type__in=["A"],
                standard_type__in=QUERY_TYPES,
                standard_value__isnull=False,
            ).only(["molecule_chembl_id", "standard_type", "standard_value",
                    "standard_units", "pchembl_value"])

            chembl_to_ik = {v: k for k, v in ik_to_chembl.items()}
            for act in acts:
                cid = act.get("molecule_chembl_id")
                records.append({
                    "inchikey": chembl_to_ik.get(cid, ""),
                    "standard_type": act.get("standard_type"),
                    "standard_value": act.get("standard_value"),
                    "pchembl_value": act.get("pchembl_value"),
                })
        except Exception as e:
            pass

        if i % 500 == 0 and i > 0:
            print(f"    Processed {i}/{len(inchikeys)} compounds, {len(records)} activities found")

    df = pd.DataFrame(records)
    print(f"  [{label}] found {len(df)} activity records for {df['inchikey'].nunique() if len(df)>0 else 0} compounds")
    if len(df) > 0:
        df.to_parquet(cache_path, index=False)
    return df


def lookup_pparg_for_train(smiles_list, inchikeys, pxr_train_set_ik):
    """Check how many PXR training compounds have PPARg data in our external dataset."""
    pparg = pd.read_parquet(DATA_EXTERNAL / "chembl_nr_extended.parquet")
    pparg = pparg[pparg["target_name"].str.contains("PPARg", na=False)].copy()
    pparg_ik_set = set(pparg["inchikey"].dropna())
    overlap = pparg_ik_set & set(inchikeys)
    print(f"  PPARg data: {len(pparg)} compounds, {len(overlap)} overlap with PXR train")
    return pparg, overlap


def knn_impute_feature(compounds_with_val, all_compounds_smi, k=5, min_sim=0.3):
    """For compounds without experimental values, impute from k nearest neighbors that have values.

    compounds_with_val: dict {smiles: value}
    all_compounds_smi: list of all SMILES to impute for
    Returns: array of length len(all_compounds_smi), NaN where no neighbor found
    """
    # Compute FPs for all compounds
    all_fps = [get_mol_fp(s) for s in all_compounds_smi]
    ref_smiles = list(compounds_with_val.keys())
    ref_vals = np.array(list(compounds_with_val.values()))
    ref_fps = [get_mol_fp(s) for s in ref_smiles]

    result = np.full(len(all_compounds_smi), np.nan)

    for i, (smi, fp_i) in enumerate(zip(all_compounds_smi, all_fps)):
        if smi in compounds_with_val:
            result[i] = compounds_with_val[smi]
            continue
        if fp_i is None:
            continue

        # Find k nearest neighbors in reference set
        sims = []
        for j, fp_ref in enumerate(ref_fps):
            sim = tanimoto_sim(fp_i, fp_ref)
            if sim >= min_sim:
                sims.append((sim, j))

        if not sims:
            continue

        sims.sort(reverse=True)
        top_k = sims[:k]
        weights = np.array([s for s, _ in top_k])
        vals = np.array([ref_vals[j] for _, j in top_k])
        result[i] = np.dot(weights, vals) / weights.sum()

    return result


def build_experimental_feature_matrix(smiles_list, inchikeys, label="train"):
    """Build matrix of experimental features from ChEMBL ADME data + NR activity."""
    features = {}

    # 1. PPARg activity (best correlated NR, rho=0.375)
    pparg_df = pd.read_parquet(DATA_EXTERNAL / "chembl_nr_extended.parquet")
    pparg_df = pparg_df[pparg_df["target_name"].str.contains("PPARg", na=False)].copy()
    pparg_map = dict(zip(pparg_df["inchikey"].dropna(), pparg_df["pec50"]))
    pparg_smi_map = dict(zip(pparg_df["std_smiles"].dropna(), pparg_df["pec50"]))

    # Match by inchikey first, then SMILES
    pparg_vals = {}
    for smi, ik in zip(smiles_list, inchikeys):
        if ik in pparg_map:
            pparg_vals[smi] = pparg_map[ik]
        elif smi in pparg_smi_map:
            pparg_vals[smi] = pparg_smi_map[smi]

    print(f"  PPARg direct matches ({label}): {len(pparg_vals)}/{len(smiles_list)}")

    if len(pparg_vals) > 20:
        pparg_feat = knn_impute_feature(pparg_vals, smiles_list, k=5, min_sim=0.35)
        features["pparg_exp"] = pparg_feat
        n_imputed = np.sum(np.isfinite(pparg_feat)) - len(pparg_vals)
        n_nan = np.sum(np.isnan(pparg_feat))
        print(f"    After KNN imputation: {np.sum(np.isfinite(pparg_feat))} filled, {n_nan} NaN")

    # 2. FXR activity (rho=0.226)
    fxr_df = pd.read_parquet(DATA_EXTERNAL / "chembl_nr_extended.parquet")
    fxr_df = fxr_df[fxr_df["target_name"].str.contains("FXR", na=False)].copy()
    fxr_map = dict(zip(fxr_df["inchikey"].dropna(), fxr_df["pec50"]))
    fxr_smi_map = dict(zip(fxr_df["std_smiles"].dropna(), fxr_df["pec50"]))

    fxr_vals = {}
    for smi, ik in zip(smiles_list, inchikeys):
        if ik in fxr_map:
            fxr_vals[smi] = fxr_map[ik]
        elif smi in fxr_smi_map:
            fxr_vals[smi] = fxr_smi_map[smi]

    print(f"  FXR direct matches ({label}): {len(fxr_vals)}/{len(smiles_list)}")

    if len(fxr_vals) > 10:
        fxr_feat = knn_impute_feature(fxr_vals, smiles_list, k=5, min_sim=0.35)
        features["fxr_exp"] = fxr_feat
        print(f"    After KNN imputation: {np.sum(np.isfinite(fxr_feat))} filled")

    return features


def main():
    print("=== nb218: Direct experimental lookup + KNN imputation ===\n")

    tr = load_train()
    te_df = load_test()
    tr = add_standard_columns(tr)
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["smiles"].tolist()
    ik_tr = tr["inchikey"].tolist()

    # Get test inchikeys
    from rdkit.Chem.inchi import MolToInchiKey
    ik_te = []
    for s in smiles_te:
        mol = Chem.MolFromSmiles(s)
        if mol:
            ik_te.append(MolToInchiKey(mol) or "")
        else:
            ik_te.append("")

    oof_base = np.load(DATA_PROCESSED / "oof_nb197_dense_grid.npy").astype(np.float64)
    te_base  = np.load(DATA_PROCESSED / "te_nb197_dense_grid.npy").astype(np.float64)
    base_rae = rae(y_tr, oof_base)
    print(f"Base nb197: OOF RAE={base_rae:.4f}  ratio={te_base.std()/oof_base.std():.3f}\n")

    print("Featurizing PXR compounds...")
    X_tr_base = combined(smiles_tr); X_tr_base = impute(X_tr_base)
    X_te_base = combined(smiles_te); X_te_base = impute(X_te_base)

    # ── Build experimental feature matrices ───────────────────────────────────
    print("\n── Building experimental feature matrix ──")
    print("Training set:")
    tr_exp_feats = build_experimental_feature_matrix(smiles_tr, ik_tr, "train")
    print("\nTest set:")
    te_exp_feats = build_experimental_feature_matrix(smiles_te, ik_te, "test")

    if not tr_exp_feats:
        print("No experimental features built. Check ChEMBL data files.")
        return

    # Assemble augmented feature matrices, filling NaN with column median
    feat_names = list(tr_exp_feats.keys())
    print(f"\nBuilt {len(feat_names)} experimental features: {feat_names}")

    def assemble(base_X, exp_dict):
        cols = []
        for name in feat_names:
            arr = exp_dict.get(name, np.full(len(base_X), np.nan))
            med = np.nanmedian(arr) if np.any(np.isfinite(arr)) else 0.0
            arr_filled = np.where(np.isfinite(arr), arr, med)
            cols.append(arr_filled.reshape(-1, 1))
            rho, _ = spearmanr(arr_filled, y_tr if len(arr_filled)==len(y_tr) else arr_filled)
        return np.hstack([base_X] + cols)

    X_tr_aug = assemble(X_tr_base, tr_exp_feats)
    X_te_aug = assemble(X_te_base, te_exp_feats)

    # Spearman correlations
    print("\nFeature correlations with PXR train labels:")
    for name in feat_names:
        arr = tr_exp_feats[name]
        rho, pval = spearmanr(np.where(np.isfinite(arr), arr, np.nanmedian(arr)), y_tr)
        print(f"  {name}: rho={rho:.3f} (p={pval:.2e})")

    # ── Scaffold 5-fold CV ────────────────────────────────────────────────────
    print("\n── Scaffold 5-fold CV ──")
    scaffolds = tr["scaffold"].tolist()
    folds = scaffold_kfold_indices(scaffolds, n_splits=5)

    cv_results = {}
    for name, X_tr_use, X_te_use in [
        ("base_only",  X_tr_base, X_te_base),
        ("exp_aug",    X_tr_aug,  X_te_aug),
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
        print(f"  {name:12s}: OOF RAE={r:.4f}  ratio={ratio:.3f}  [{flag}]")
        cv_results[name] = (oof, te_pred, r, ratio)

    # ── Blend with nb197 ─────────────────────────────────────────────────────
    print("\n── Blend exp_aug with nb197 ──")
    oof_aug, te_aug, r_aug, ratio_aug = cv_results["exp_aug"]
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
        np.save(DATA_PROCESSED / "oof_nb218_exp_aug.npy", oof_aug)
        np.save(DATA_PROCESSED / "te_nb218_exp_aug.npy", te_aug)
        sub = pd.DataFrame({"Molecule Name": te_df["name"], "pEC50": te_aug})
        sub.to_csv(SUBMISSIONS / "218_exp_aug.csv", index=False)
        saved.append(f"218_exp_aug OOF={r_aug:.4f}")

    if best_blend and best_r_bl < base_rae:
        w_b, oof_b, te_b, ratio_b = best_blend
        name = f"218_blend_w{int(w_b*100):02d}"
        np.save(DATA_PROCESSED / f"oof_{name}.npy", oof_b)
        np.save(DATA_PROCESSED / f"te_{name}.npy", te_b)
        sub = pd.DataFrame({"Molecule Name": te_df["name"], "pEC50": te_b})
        sub.to_csv(SUBMISSIONS / f"{name}.csv", index=False)
        saved.append(f"{name} OOF={best_r_bl:.4f}")

    print(f"\n=== Done. Saved: {saved or ['none']}")


if __name__ == "__main__":
    main()
