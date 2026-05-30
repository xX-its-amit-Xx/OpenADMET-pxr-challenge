"""nb274 -- Per-scaffold/functional-group performance analysis.

Group train compounds by Murcko scaffold + functional group features.
For each group, evaluate which model has lowest OOF MAE.
Find scaffolds where MolFormer/ChemBERTa beat nb239.

Build piecewise router: classifier predicts compound family from SMILES,
then routes to family-specific best model.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Lipinski, Crippen, Descriptors
from pxr.data import load_train
from pxr.eval import rae
from pxr.chem import add_standard_columns
from pxr.paths import DATA_PROCESSED


def main():
    print("=== nb274: Per-scaffold/family interpretability ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()

    # Load model OOFs
    nb239 = np.load(DATA_PROCESSED / "oof_nb239_full_slsqp.npy")
    nb224 = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
    chemberta = np.load(DATA_PROCESSED / "oof_nb94_chemberta_mtr.npy")
    molformer = np.load(DATA_PROCESSED / "oof_nb273_molformer_only.npy")
    grover = np.load(DATA_PROCESSED / "oof_grover_large.npy")
    nb179s = np.load(DATA_PROCESSED / "oof_nb179_stack.npy")

    models = {
        "nb239": nb239, "nb224": nb224, "molformer": molformer,
        "chemberta": chemberta, "grover": grover, "nb179s": nb179s
    }
    print(f"Models loaded: {list(models.keys())}")
    print(f"\nFull OOF RAE per model:")
    for n, m in models.items():
        print(f"  {n}: {rae(y_tr, m):.4f}")

    # Compute residuals per model
    residuals = {n: np.abs(m - y_tr) for n, m in models.items()}

    # Group compounds by scaffold + functional group classes
    print("\nClassifying compounds by structural features...")
    feature_groups = defaultdict(list)  # group_key -> list of compound indices
    for i, smi in enumerate(smiles_tr):
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue
        # Group key by 6-tuple of binary indicators
        n_arom_rings = Lipinski.NumAromaticRings(mol)
        n_arom_het = Lipinski.NumAromaticHeterocycles(mol)
        has_sulfonyl = any(a.GetSymbol() == 'S' and any(b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(a).GetSymbol() == 'O' for b in a.GetBonds()) for a in mol.GetAtoms())
        has_amide = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'N' and any(b.GetBondType() == Chem.BondType.SINGLE for b in a.GetBonds()) and any(nbr.GetSymbol() == 'C' and any(b2.GetBondType() == Chem.BondType.DOUBLE and b2.GetOtherAtom(nbr).GetSymbol() == 'O' for b2 in nbr.GetBonds()) for nbr in a.GetNeighbors())) > 0
        has_carboxyl = any(a.GetSymbol() == 'O' and a.GetTotalNumHs() == 1 and any(nbr.GetSymbol() == 'C' for nbr in a.GetNeighbors()) for a in mol.GetAtoms())
        n_fused_rings = sum(1 for r1 in mol.GetRingInfo().AtomRings() for r2 in mol.GetRingInfo().AtomRings() if r1 < r2 and set(r1) & set(r2))
        n_rotbonds = Lipinski.NumRotatableBonds(mol)
        logp = Crippen.MolLogP(mol)
        mw = Descriptors.MolWt(mol)

        # Coarse buckets
        size_bucket = 'small' if mw < 300 else 'med' if mw < 450 else 'large'
        flex_bucket = 'rigid' if n_rotbonds < 3 else 'flex' if n_rotbonds < 7 else 'verylex'
        ring_class = 'no_arom' if n_arom_rings == 0 else 'hetero' if n_arom_het > 0 else 'carbo'
        fn_class = 'sulfonyl' if has_sulfonyl else 'amide' if has_amide else 'acid' if has_carboxyl else 'plain'

        key = f"{size_bucket}_{flex_bucket}_{ring_class}_{fn_class}"
        feature_groups[key].append(i)

    print(f"\nTotal groups: {len(feature_groups)}")
    sized = sorted(feature_groups.items(), key=lambda x: -len(x[1]))[:30]
    print(f"\nTop 30 groups by size:")
    print(f"  {'group':40s} {'n':>5} {'mean_y':>7}  {'best_model':12s} {'best_MAE':>9} | other model MAEs")
    for grp, idx in sized:
        idx_arr = np.array(idx)
        if len(idx_arr) < 20: continue  # skip small groups
        mean_y = y_tr[idx_arr].mean()
        model_maes = {n: residuals[n][idx_arr].mean() for n in models}
        best_n = min(model_maes, key=model_maes.get)
        others = " ".join(f"{n}={v:.2f}" for n, v in sorted(model_maes.items()))
        print(f"  {grp:40s} {len(idx_arr):>5} {mean_y:>7.2f}  {best_n:12s} {model_maes[best_n]:>9.4f} | {others}")

    # Identify groups where MolFormer or ChemBERTa or Grover beats nb239
    print(f"\n=== Groups where deep models beat nb239 ===")
    beats = defaultdict(list)
    for grp, idx in feature_groups.items():
        idx_arr = np.array(idx)
        if len(idx_arr) < 30: continue
        model_maes = {n: residuals[n][idx_arr].mean() for n in models}
        for n in ['molformer', 'chemberta', 'grover']:
            if model_maes[n] < model_maes['nb239']:
                beats[n].append((grp, len(idx), model_maes['nb239'] - model_maes[n], model_maes[n], model_maes['nb239']))

    for n in ['molformer', 'chemberta', 'grover']:
        if not beats[n]:
            print(f"  {n}: never beats nb239 on any group (n>=30)")
            continue
        print(f"\n  {n} beats nb239 on {len(beats[n])} groups:")
        for grp, n_cmpd, delta, m_mae, nb_mae in sorted(beats[n], key=lambda x: -x[2])[:8]:
            print(f"    {grp:40s} n={n_cmpd:>4}  {n}={m_mae:.3f} nb239={nb_mae:.3f}  delta=+{delta:.3f}")

    # Save the group classification for use in piecewise router
    np.save(DATA_PROCESSED / "nb274_group_indices.npy", np.array([list(feature_groups.keys())], dtype=object), allow_pickle=True)
    # Map each compound to its group
    group_label = np.full(len(smiles_tr), "unknown", dtype=object)
    for grp, idx in feature_groups.items():
        for i in idx:
            group_label[i] = grp
    pd.DataFrame({"smiles": smiles_tr, "group": group_label}).to_csv(DATA_PROCESSED / "nb274_compound_groups.csv", index=False)
    print(f"\nSaved nb274_compound_groups.csv")


if __name__ == "__main__":
    main()
