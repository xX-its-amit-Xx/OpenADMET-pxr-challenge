"""nb171 -- SHAP interpretability + Morgan bit → substructure decoding.

Train LGBM on combined (Morgan 2048 + RDKit ~217) features. Compute SHAP values.
For top-N important features:
  - If Morgan bit (col 0-2047): decode to substructure SMARTS via RDKit bit info
  - If RDKit descriptor (col 2048+): print descriptor name
Save findings to data/processed/nb171_top_features.csv with interpretations.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Descriptors import descList

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED


def get_descriptor_names():
    """Return list of 217 RDKit descriptor names from pxr.featurize.rdkit_desc."""
    return [name for name, _ in descList]


def decode_morgan_bit(bit_idx, smiles_list, radius=2, n_bits=2048, top_k=3):
    """Return up to top_k example substructures for a given Morgan bit.

    For each SMILES, identify which atoms set that bit, then write the
    fragment around them as SMARTS.
    """
    examples = []
    for smi in smiles_list:
        if len(examples) >= top_k:
            break
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        bi = {}
        AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, bitInfo=bi)
        if bit_idx not in bi:
            continue
        atom_idx, atom_radius = bi[bit_idx][0]
        if atom_radius == 0:
            symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
            examples.append((symbol, smi))
        else:
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, atom_radius, atom_idx)
            atoms = set()
            for bond in env:
                b = mol.GetBondWithIdx(bond)
                atoms.add(b.GetBeginAtomIdx()); atoms.add(b.GetEndAtomIdx())
            atoms.add(atom_idx)
            try:
                sub = Chem.PathToSubmol(mol, env)
                smarts = Chem.MolToSmarts(sub) if sub.GetNumAtoms() > 0 else "?"
            except Exception:
                smarts = "?"
            examples.append((smarts, smi))
    return examples


def main():
    print("=== nb171: SHAP interpretability ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    print(f"Train: {len(smiles_tr)}")

    print("Featurizing...")
    X_tr = combined(smiles_tr); X_tr = impute(X_tr)
    print(f"X_tr shape: {X_tr.shape}")

    # Train a robust LGBM (use scaffold CV to be faithful)
    LGBM = dict(n_estimators=1500, num_leaves=63, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
                objective="mae", n_jobs=4, random_state=42, verbose=-1)
    scaffolds = tr["scaffold"].tolist()
    folds = scaffold_kfold_indices(scaffolds, n_splits=5)

    # Train on full data for SHAP (interpretability target)
    print("Training final LGBM on all train data...")
    m = lgb.LGBMRegressor(**LGBM)
    m.fit(X_tr, y_tr, callbacks=[lgb.log_evaluation(-1)])

    # SHAP TreeExplainer (fast on tree models)
    print("Computing SHAP values on a 500-sample subset...")
    import shap
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_tr), size=500, replace=False)
    explainer = shap.TreeExplainer(m)
    shap_values = explainer.shap_values(X_tr[sample_idx])
    print(f"SHAP shape: {shap_values.shape}")

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    n_morgan = 2048
    desc_names = get_descriptor_names()
    feat_names = [f"morgan_{i}" for i in range(n_morgan)] + desc_names

    # Top 50
    top_idx = np.argsort(mean_abs_shap)[::-1][:50]
    rows = []
    print("\nTop 50 features by mean |SHAP|:")
    for rank, idx in enumerate(top_idx, 1):
        feat = feat_names[idx]
        importance = mean_abs_shap[idx]
        # Determine semantic
        if idx < n_morgan:
            examples = decode_morgan_bit(idx, smiles_tr[:300], radius=2, n_bits=n_morgan, top_k=2)
            meaning = "; ".join(e[0] for e in examples) if examples else "no_example"
            feat_type = "morgan_bit"
        else:
            meaning = desc_names[idx - n_morgan] if idx - n_morgan < len(desc_names) else "?"
            feat_type = "rdkit_desc"
        rows.append({"rank": rank, "feat_idx": int(idx), "feat_name": feat,
                     "feat_type": feat_type, "mean_abs_shap": float(importance),
                     "interpretation": meaning})
        print(f"  {rank:3d}. {feat:20s}  shap={importance:.4f}  type={feat_type:10s}  meaning={meaning[:60]}")

    df = pd.DataFrame(rows)
    df.to_csv(DATA_PROCESSED / "nb171_top_features.csv", index=False)
    print(f"\nSaved {DATA_PROCESSED}/nb171_top_features.csv")

    # Also save all SHAP values for further analysis
    np.save(DATA_PROCESSED / "nb171_shap_values.npy", shap_values)
    np.save(DATA_PROCESSED / "nb171_shap_sample_idx.npy", sample_idx)
    np.save(DATA_PROCESSED / "nb171_feat_importance.npy", mean_abs_shap)
    print(f"Saved nb171_shap_values.npy + sample_idx + feat_importance")

    # Summary stats by feature type
    morgan_imp = mean_abs_shap[:n_morgan].sum()
    desc_imp = mean_abs_shap[n_morgan:].sum()
    total = morgan_imp + desc_imp
    print(f"\nFeature-type contribution share:")
    print(f"  Morgan FP:  {morgan_imp/total:.1%}  (sum of |SHAP|)")
    print(f"  RDKit desc: {desc_imp/total:.1%}")


if __name__ == "__main__":
    main()
