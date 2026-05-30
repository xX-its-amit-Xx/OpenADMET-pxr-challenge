"""nb278 -- Fragment-residue contact database.

Take pdb64 contacts (atom-level), map ligand atoms to BRICS fragments,
aggregate to FRAGMENT-RESIDUE level.

For each known PXR binder fragment, identify which pocket residue(s) it
preferentially contacts. Build a database for prediction:
  fragment X → typically contacts residue Y at distance Z

Then for test compounds: decompose to fragments, look up which residue
each fragment WOULD contact, predict binding contribution.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import urllib.request
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import BRICS, AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity

PDB_DIR = Path("data/external/pdb64_structures")


def parse_cif_for_ligand(cif_path, ligand_ccd):
    """Extract ligand atom coords + their atom names (PDB nomenclature)."""
    atoms = []
    in_atom_loop = False
    with open(cif_path) as f:
        for line in f:
            if line.startswith("loop_"):
                in_atom_loop = False; continue
            if line.startswith("_atom_site."):
                in_atom_loop = True; continue
            if in_atom_loop and line.startswith("HETATM"):
                parts = line.split()
                try:
                    if parts[5] != ligand_ccd: continue
                    atomname = parts[3].strip('"')
                    elem = parts[2]
                    try:
                        x = float(parts[10]); y = float(parts[11]); z = float(parts[12])
                    except (ValueError, IndexError):
                        for off in [9, 11, 13]:
                            try:
                                x = float(parts[off]); y = float(parts[off+1]); z = float(parts[off+2])
                                break
                            except (ValueError, IndexError): continue
                    atoms.append({'atom_name': atomname, 'element': elem, 'x': x, 'y': y, 'z': z})
                except: pass
    return atoms


def main():
    print("=== nb278: Fragment-residue contact database ===\n")
    contacts_df = pd.read_parquet("data/processed/pxr_pdb64_contacts.parquet")
    print(f"Loaded {len(contacts_df)} atom-level contacts")
    smiles_df = pd.read_csv("data/external/dargason_cofolding/pdb64_raw_smiles.csv")

    # For each ligand, identify atom-fragment mapping via BRICS
    # This is tricky: BRICS gives fragments as SMILES; we need to map them back to atom indices
    print("\nMapping ligand atoms to BRICS fragments...")
    all_frag_contacts = []
    for _, row in smiles_df.iterrows():
        pid = row['pdb_id']; lig_ccd = row['ligand_ccd']; smi = row['smiles']
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue

        # Get BRICS bonds (where to break)
        try:
            brics_bonds = list(BRICS.FindBRICSBonds(mol))
            # Get fragment atom indices via fragmentOnBonds + getMolFrags
            if brics_bonds:
                bond_indices = [mol.GetBondBetweenAtoms(*pair[0]).GetIdx() for pair in brics_bonds]
                fragmented = Chem.FragmentOnBonds(mol, bond_indices)
                frag_atom_idx_list = Chem.GetMolFrags(fragmented, asMols=False)
            else:
                frag_atom_idx_list = [tuple(range(mol.GetNumAtoms()))]
        except Exception:
            frag_atom_idx_list = [tuple(range(mol.GetNumAtoms()))]

        # Get fragment SMILES
        try:
            if brics_bonds:
                fragmented = Chem.FragmentOnBonds(mol, bond_indices)
                frag_mols = Chem.GetMolFrags(fragmented, asMols=True)
                frag_smiles = [Chem.MolToSmiles(fm) for fm in frag_mols]
            else:
                frag_smiles = [Chem.MolToSmiles(mol)]
        except Exception:
            frag_smiles = ['_unknown']

        # Get this PDB's ligand contacts
        pdb_contacts = contacts_df[(contacts_df['pdb_id'] == pid) & (contacts_df['ligand_ccd'] == lig_ccd)]
        if len(pdb_contacts) == 0: continue

        # For each contact (lig_atom name), we need to map to fragment.
        # Get CIF atom-name → mol atom-idx via the actual PDB structure
        cif = PDB_DIR / f"{pid}.cif"
        if not cif.exists(): continue
        cif_atoms = parse_cif_for_ligand(cif, lig_ccd)
        if not cif_atoms: continue

        # Heuristic: PDB atom names like "C1", "C2", "N", "O3" - approximately follow SMILES order
        # For accurate mapping we'd need to parse 3D coords and run substructure match
        # Simpler: just associate contacts with fragments by COUNT (atom-level granularity OK for aggregate stats)
        # For each fragment, count how many of its atoms have contact, distribute equally

        # Map ligand atom-name -> mol atom-idx
        # Use simple element + position-order heuristic
        elem_idx = {}
        for atom in mol.GetAtoms():
            elem = atom.GetSymbol()
            elem_idx.setdefault(elem, 0)
            atom.SetIntProp('_seq', elem_idx[elem])
            elem_idx[elem] += 1
        name_to_idx = {}
        for cif_atom in cif_atoms:
            elem = cif_atom['element']
            # Extract trailing digit from atom name
            name = cif_atom['atom_name']
            # match: same element, sequence index
            seq_in_name = int(''.join(c for c in name if c.isdigit()) or '1') - 1
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == elem and atom.GetIntProp('_seq') == seq_in_name:
                    name_to_idx[name] = atom.GetIdx()
                    break

        # For each contact, find fragment index
        for _, contact in pdb_contacts.iterrows():
            lig_atom = contact['lig_atom']
            atom_idx = name_to_idx.get(lig_atom)
            if atom_idx is None: continue
            # Find which fragment contains this atom
            for fi, fatoms in enumerate(frag_atom_idx_list):
                if atom_idx in fatoms:
                    all_frag_contacts.append({
                        'pdb_id': pid,
                        'fragment_smiles': frag_smiles[fi] if fi < len(frag_smiles) else '_unknown',
                        'fragment_idx': fi,
                        'res_name': contact['res_name'],
                        'res_num': contact['res_num'],
                        'distance': contact['distance']
                    })
                    break

    print(f"\nFragment-residue contacts: {len(all_frag_contacts)}")
    fc_df = pd.DataFrame(all_frag_contacts)
    fc_df.to_parquet("data/processed/pxr_fragment_residue_contacts.parquet", index=False)
    print(f"Saved pxr_fragment_residue_contacts.parquet")

    # Aggregate: for each (fragment, residue) pair, count occurrences across ligands
    print("\nTop fragment-residue pairs (most frequent):")
    pair_freq = fc_df.groupby(['fragment_smiles', 'res_name', 'res_num']).agg(
        n_contacts=('pdb_id', 'count'),
        n_ligands=('pdb_id', 'nunique'),
        mean_dist=('distance', 'mean'),
    ).reset_index().sort_values('n_ligands', ascending=False)
    print(pair_freq.head(20).to_string())

    # Per-fragment summary: which residue does this fragment most prefer?
    frag_prefer = fc_df.groupby('fragment_smiles').agg(
        n_contacts=('res_name', 'count'),
        n_pdbs=('pdb_id', 'nunique'),
        most_freq_res=('res_name', lambda x: x.value_counts().idxmax() if len(x) > 0 else None),
    ).reset_index().sort_values('n_contacts', ascending=False)
    print("\nTop fragments by total contacts:")
    print(frag_prefer.head(15).to_string())


if __name__ == "__main__":
    main()
