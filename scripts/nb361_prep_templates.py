"""
Prepare apo receptors + pocket centroids for all 64 PDB templates.

For each CIF:
  - parse with gemmi (mmCIF)
  - select chain A protein residues only (drop ligands, waters, ions, hetatm)
  - identify bound ligand by HET code (non-standard residue >= 6 heavy atoms,
    not water/ion, with the most heavy atoms)
  - compute geometric mean of ligand heavy atoms = pocket centroid
  - write apo PDB + centroid JSON
"""
from __future__ import annotations

import json
from pathlib import Path

import gemmi

ROOT = Path(r"d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge")
CIF_DIR = ROOT / "data/external/pdb64_structures"
OUT_DIR = ROOT / "data/processed/templates"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Standard amino acids
STD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL", "MSE", "SEC", "PYL",
}
# Common ions / waters / cofactor noise to ignore as "the ligand"
NON_LIGANDS = {
    "HOH", "WAT", "DOD", "H2O",
    "NA", "K", "MG", "CA", "ZN", "FE", "MN", "CL", "BR", "I", "F",
    "SO4", "PO4", "CL-", "NA+", "K+",
    "EDO", "GOL", "PEG", "PG4", "PGE", "MPD", "DMS", "ACT", "FMT",
    "TRS", "BME", "IPA", "EPE", "MES", "BTB", "HEPES",
}


def prep_one(cif_path: Path) -> dict | None:
    pdb_id = cif_path.stem.lower()
    try:
        st = gemmi.read_structure(str(cif_path))
    except Exception as e:
        return {"pdb_id": pdb_id, "error": f"read_failed: {e}"}

    if len(st) == 0:
        return {"pdb_id": pdb_id, "error": "empty_structure"}

    model = st[0]
    chain_a = None
    for chain in model:
        if chain.name == "A":
            chain_a = chain
            break
    if chain_a is None:
        # take first chain as fallback
        chain_a = model[0]

    # Build a fresh apo structure with only chain A protein residues
    apo = gemmi.Structure()
    apo.spacegroup_hm = st.spacegroup_hm
    apo.cell = st.cell
    apo_model = gemmi.Model("1")
    apo_chain = gemmi.Chain("A")

    # Collect candidate ligands from the entire model (any chain)
    ligand_candidates: list[tuple[str, list[gemmi.Atom]]] = []

    for chain in model:
        for res in chain:
            name = res.name.strip().upper()
            if name in STD_AA:
                if chain.name == chain_a.name:
                    apo_chain.add_residue(res)
                continue
            if name in NON_LIGANDS:
                continue
            # treat as ligand candidate: collect heavy atoms
            heavy = [a for a in res if a.element.name not in ("H", "D")]
            if len(heavy) >= 6:
                ligand_candidates.append((name, heavy))

    if len(apo_chain) == 0:
        return {"pdb_id": pdb_id, "error": "no_protein_chainA"}

    apo_model.add_chain(apo_chain)
    apo.add_model(apo_model)

    # Pick ligand with the most heavy atoms (typical for PXR LBD)
    if not ligand_candidates:
        # write apo anyway, no centroid
        apo_path = OUT_DIR / f"{pdb_id}_apo.pdb"
        apo.write_pdb(str(apo_path))
        return {
            "pdb_id": pdb_id,
            "apo_pdb": str(apo_path),
            "het_code": None,
            "centroid": None,
            "n_ligand_atoms": 0,
            "warning": "no_ligand_found",
        }

    ligand_candidates.sort(key=lambda x: len(x[1]), reverse=True)
    het_code, heavy_atoms = ligand_candidates[0]

    n = len(heavy_atoms)
    cx = sum(a.pos.x for a in heavy_atoms) / n
    cy = sum(a.pos.y for a in heavy_atoms) / n
    cz = sum(a.pos.z for a in heavy_atoms) / n

    apo_path = OUT_DIR / f"{pdb_id}_apo.pdb"
    apo.write_pdb(str(apo_path))

    centroid_path = OUT_DIR / f"{pdb_id}_centroid.json"
    rec = {
        "pdb_id": pdb_id,
        "het_code": het_code,
        "centroid": [round(cx, 3), round(cy, 3), round(cz, 3)],
        "n_ligand_heavy_atoms": n,
    }
    centroid_path.write_text(json.dumps(rec, indent=2))
    rec["apo_pdb"] = str(apo_path)
    rec["centroid_json"] = str(centroid_path)
    return rec


def main():
    cifs = sorted(CIF_DIR.glob("*.cif"))
    print(f"Found {len(cifs)} CIF files")
    results = []
    for cif in cifs:
        rec = prep_one(cif)
        results.append(rec)
        if rec and "error" in rec:
            print(f"  [ERR] {rec['pdb_id']}: {rec['error']}")
        else:
            tag = rec.get("warning", "ok") if rec else "none"
            c = rec.get("centroid") if rec else None
            print(f"  [{tag:>16s}] {rec['pdb_id']}  het={rec.get('het_code')}  centroid={c}")
    # write summary
    summary = OUT_DIR / "templates_summary.json"
    summary.write_text(json.dumps(results, indent=2))
    ok = sum(1 for r in results if r and r.get("centroid"))
    print(f"\nPrepared {ok}/{len(results)} templates with ligand centroid")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
