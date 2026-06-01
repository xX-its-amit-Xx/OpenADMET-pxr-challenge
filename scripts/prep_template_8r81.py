"""Prepare 8R81 apo receptor + pocket centroid for docking.

Reads data/external/pdb64_structures/8r81.cif, keeps protein chain A only,
strips ligands/waters/HET, writes apo PDB. Computes centroid of bound Y8B
ligand atoms (from holo structure) for docking box center.
"""
from __future__ import annotations

import json
from pathlib import Path

import gemmi
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CIF = ROOT / "data" / "external" / "pdb64_structures" / "8r81.cif"
APO_PDB = ROOT / "data" / "processed" / "template_apo_receptor.pdb"
CENTROID_JSON = ROOT / "data" / "processed" / "template_pocket_centroid.json"

LIGAND_CODE = "Y8B"
TEMPLATE_ID = "8R81"
CHAIN_KEEP = "A"

STD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL",
}


def main() -> None:
    APO_PDB.parent.mkdir(parents=True, exist_ok=True)
    struct = gemmi.read_structure(str(CIF))
    struct.setup_entities()

    # Step 1: compute centroid of Y8B atoms in chain A (or first occurrence) before stripping
    ligand_coords: list[tuple[float, float, float]] = []
    for model in struct:
        for chain in model:
            for res in chain:
                if res.name == LIGAND_CODE:
                    for atom in res:
                        if atom.element.name == "H":
                            continue
                        ligand_coords.append((atom.pos.x, atom.pos.y, atom.pos.z))
            if ligand_coords:
                break
        if ligand_coords:
            break

    if not ligand_coords:
        raise RuntimeError(f"Ligand {LIGAND_CODE} not found in {CIF}")
    arr = np.array(ligand_coords, dtype=float)
    cx, cy, cz = arr.mean(axis=0).tolist()
    n_lig_atoms = len(ligand_coords)

    # Step 2: build apo - keep only chain A standard AAs
    new_struct = gemmi.Structure()
    new_struct.name = TEMPLATE_ID
    new_struct.cell = struct.cell
    new_struct.spacegroup_hm = struct.spacegroup_hm

    # use only model 1
    m0 = struct[0]
    new_model = gemmi.Model(getattr(m0, "num", 1) and "1" or "1")
    kept_residues = 0
    for chain in m0:
        if chain.name != CHAIN_KEEP:
            continue
        new_chain = gemmi.Chain(chain.name)
        for res in chain:
            if res.name not in STD_AA:
                continue
            new_chain.add_residue(res)
            kept_residues += 1
        if len(new_chain) > 0:
            new_model.add_chain(new_chain)
    new_struct.add_model(new_model)

    new_struct.write_pdb(str(APO_PDB))

    centroid = {
        "template_id": TEMPLATE_ID,
        "ligand_code": LIGAND_CODE,
        "chain": CHAIN_KEEP,
        "centroid": [round(cx, 3), round(cy, 3), round(cz, 3)],
        "n_ligand_heavy_atoms": n_lig_atoms,
        "n_protein_residues": kept_residues,
        "source_cif": str(CIF.relative_to(ROOT)).replace("\\", "/"),
        "apo_pdb": str(APO_PDB.relative_to(ROOT)).replace("\\", "/"),
    }
    CENTROID_JSON.write_text(json.dumps(centroid, indent=2))
    print(json.dumps(centroid, indent=2))


if __name__ == "__main__":
    main()
