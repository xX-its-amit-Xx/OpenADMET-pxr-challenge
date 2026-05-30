"""nb277 -- Direct PDB download + contact extraction for pdb64 PXR co-crystals.

64 unique PDB IDs with PXR-ligand co-crystals. Download CIF directly from RCSB.
For each: extract ligand atoms + protein residues + compute 4Å contacts.

Build fragment-residue contact database for the 70 known PXR binders.
"""
import os, sys, warnings, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import urllib.request
from pathlib import Path
from collections import defaultdict, Counter

PDB_DIR = Path("data/external/pdb64_structures")
PDB_DIR.mkdir(exist_ok=True, parents=True)


def download_pdb(pdb_id):
    """Download CIF from RCSB."""
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
    dest = PDB_DIR / f"{pdb_id}.cif"
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"  FAIL {pdb_id}: {e}")
        return False


def parse_cif_atoms(cif_path):
    """Lightweight CIF atom parser. Returns list of (chain, resnum, resname, atomname, x, y, z, is_hetatm)."""
    atoms = []
    in_atom_loop = False
    headers = []
    with open(cif_path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("loop_"):
                in_atom_loop = False
                headers = []
                continue
            if line.startswith("_atom_site."):
                headers.append(line.strip())
                in_atom_loop = True
                continue
            if in_atom_loop and line.startswith(("ATOM ", "HETATM")):
                parts = line.split()
                if len(parts) < len(headers): continue
                # Standard CIF atom_site fields: group_PDB, id, type_symbol, label_atom_id, label_alt_id, label_comp_id, label_asym_id, label_entity_id, label_seq_id, pdbx_PDB_ins_code, Cartn_x, Cartn_y, Cartn_z, ...
                try:
                    is_hetatm = parts[0] == "HETATM"
                    atomname = parts[3].strip('"')
                    resname = parts[5]
                    chain = parts[7] if len(parts) > 7 else parts[6]
                    resnum = parts[8] if len(parts) > 8 else parts[7]
                    # Coordinates: Cartn_x, Cartn_y, Cartn_z (typically idx 10, 11, 12)
                    # Try multiple positions in case format varies
                    try:
                        x = float(parts[10]); y = float(parts[11]); z = float(parts[12])
                    except (ValueError, IndexError):
                        # Try alternative positions
                        for off in [9, 11, 13]:
                            try:
                                x = float(parts[off]); y = float(parts[off+1]); z = float(parts[off+2])
                                break
                            except (ValueError, IndexError):
                                continue
                    atoms.append((chain, resnum, resname, atomname, x, y, z, is_hetatm))
                except (ValueError, IndexError):
                    pass
    return atoms


def compute_contacts(atoms, ligand_resname, cutoff=4.0):
    """For ligand atoms (resname=ligand_resname, hetatm), compute contacts with protein."""
    ligand_atoms = [a for a in atoms if a[7] and a[2] == ligand_resname]
    protein_atoms = [a for a in atoms if not a[7] or (a[2] not in ('HOH', ligand_resname))]
    contacts = []  # (lig_atom_name, prot_resname, prot_resnum, prot_atomname, distance)
    for la in ligand_atoms:
        lx, ly, lz = la[4], la[5], la[6]
        for pa in protein_atoms:
            d = ((lx-pa[4])**2 + (ly-pa[5])**2 + (lz-pa[6])**2) ** 0.5
            if d < cutoff:
                contacts.append((la[3], pa[2], pa[1], pa[3], d))
    return contacts


def main():
    print("=== nb277: PDB download + contact extraction ===\n")
    df = pd.read_csv("data/external/dargason_cofolding/pdb64_raw_smiles.csv")
    pdb_ids = sorted(df.pdb_id.unique())
    print(f"PDB IDs to download: {len(pdb_ids)}")

    print("\nDownloading CIFs from RCSB...")
    t0 = time.time()
    for i, pid in enumerate(pdb_ids):
        ok = download_pdb(pid)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(pdb_ids)}  elapsed={elapsed:.0f}s")
        time.sleep(0.1)

    # Parse + extract contacts
    print("\nParsing CIFs + computing contacts...")
    all_contacts = []  # (pdb_id, ligand_smiles, ligand_resname, lig_atom, res_name, res_num, prot_atom, dist)
    for i, row in df.iterrows():
        pid = row['pdb_id']
        lig_ccd = row['ligand_ccd']
        smiles = row['smiles']
        cif = PDB_DIR / f"{pid}.cif"
        if not cif.exists(): continue
        try:
            atoms = parse_cif_atoms(cif)
            contacts = compute_contacts(atoms, lig_ccd)
            for la, pr, pn, pa, d in contacts:
                all_contacts.append({
                    'pdb_id': pid, 'ligand_smiles': smiles, 'ligand_ccd': lig_ccd,
                    'lig_atom': la, 'res_name': pr, 'res_num': pn, 'prot_atom': pa, 'distance': d
                })
        except Exception as e:
            print(f"  parse fail {pid}: {e}")

    print(f"\nTotal contacts: {len(all_contacts)}")
    cdf = pd.DataFrame(all_contacts)
    cdf.to_parquet("data/processed/pxr_pdb64_contacts.parquet", index=False)
    print(f"Saved pxr_pdb64_contacts.parquet")

    # Summary: which residues are most-contacted across all 70 ligands?
    if len(cdf) > 0:
        contact_per_lig = cdf.groupby(['pdb_id', 'res_name', 'res_num']).size().reset_index(name='atom_contacts')
        residue_freq = cdf.groupby(['res_name', 'res_num']).agg(
            n_ligands=('pdb_id', 'nunique'),
            total_contacts=('lig_atom', 'count'),
            mean_dist=('distance', 'mean')
        ).reset_index().sort_values('n_ligands', ascending=False)
        print("\nTop 20 most-contacted residues (across 70 ligands):")
        print(residue_freq.head(20).to_string())

if __name__ == "__main__":
    main()
