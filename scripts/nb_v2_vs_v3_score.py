"""Score v2 vs v3 redocked structures for 14 low-conf compounds.

For each id:
  - Extract LIG residue from v2 and v3 PDBs.
  - Parse ligand into RDKit Mol via AssignBondOrdersFromTemplate (using TEST_BLINDED SMILES).
  - Compute MMFF94s energy.
  - Compute steric clash: count ligand-protein atom pairs within 1.5 A.
  - Pick "better" pose: lower MMFF energy AND lower clash count.

Memory-safe: extract one file at a time, delete after.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

ROOT = Path("d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge")
V2_ZIP = ROOT / "submissions" / "structure_baseline_v2.zip"
V3_ZIP = ROOT / "submissions" / "structure_baseline_v3.zip"
STRUCT_CSV = ROOT / "data" / "raw" / "pxr-challenge_structure_TEST_BLINDED.csv"
OUT_CSV = ROOT / "data" / "processed" / "v2_vs_v3_scores.csv"

IDS = [
    "x03406-1", "x02797-1", "x02865-1", "x01334-1", "x03325-1",
    "x03026-1", "x03462-1", "x03145-1", "x02872-1", "x02696-1",
    "x00463-1", "x00046-1", "x03273-1", "x01131-1",
]

CLASH_DIST = 1.5  # Angstroms
CLASH_DIST_SQ = CLASH_DIST * CLASH_DIST


def split_pdb_ligand_protein(pdb_text: str) -> tuple[str, np.ndarray]:
    """Return (ligand_pdb_block, protein_heavy_atom_coords[N,3]).

    Ligand = ATOM/HETATM lines with resname 'LIG' (also 'UNL', 'UNK' fallback).
    Protein = standard amino acid ATOM lines (heavy atoms only, no H).
    """
    lig_lines: list[str] = []
    prot_xyz: list[tuple[float, float, float]] = []
    # broader ligand resname set in case PDBs differ
    lig_resnames = {"LIG", "UNL", "UNK", "MOL"}
    for line in pdb_text.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if len(line) < 54:
            continue
        resname = line[17:20].strip()
        atom_name = line[12:16].strip()
        element = line[76:78].strip() if len(line) >= 78 else ""
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        if resname in lig_resnames:
            lig_lines.append(line)
        else:
            # protein: skip hydrogens
            is_h = (element == "H") or (atom_name.startswith("H") and not atom_name.startswith("HG") is False and atom_name[0] == "H")
            if element == "H":
                continue
            if element == "" and atom_name and atom_name[0] == "H":
                continue
            prot_xyz.append((x, y, z))
    lig_block = "\n".join(lig_lines) + "\nEND\n" if lig_lines else ""
    return lig_block, np.asarray(prot_xyz, dtype=np.float32)


def score_pose(lig_pdb_block: str, prot_xyz: np.ndarray, smiles: str) -> tuple[float, int, int]:
    """Return (mmff_energy, clash_count, lig_heavy_atoms).

    energy = NaN on failure.
    """
    if not lig_pdb_block.strip():
        return (float("nan"), -1, 0)

    # parse ligand pose from PDB (no bond inference yet)
    pose = Chem.MolFromPDBBlock(lig_pdb_block, sanitize=False, removeHs=False)
    if pose is None:
        return (float("nan"), -1, 0)

    # template from SMILES
    tmpl = Chem.MolFromSmiles(smiles)
    if tmpl is None:
        return (float("nan"), -1, 0)
    tmpl = Chem.AddHs(tmpl)
    # we need heavy-atom-count match for AssignBondOrdersFromTemplate
    # the PDB pose typically lacks Hs; strip Hs from template
    tmpl_noH = Chem.RemoveHs(tmpl)

    try:
        mol = AllChem.AssignBondOrdersFromTemplate(tmpl_noH, pose)
    except Exception:
        # try without Hs in pose
        try:
            pose_noH = Chem.RemoveHs(pose, sanitize=False)
            mol = AllChem.AssignBondOrdersFromTemplate(tmpl_noH, pose_noH)
        except Exception:
            return (float("nan"), -1, pose.GetNumHeavyAtoms())

    # ligand coords for clash counting (heavy atoms)
    conf = mol.GetConformer()
    lig_xyz = np.asarray(
        [(conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z)
         for i in range(mol.GetNumAtoms())],
        dtype=np.float32,
    )

    # clash count vs protein heavy atoms
    if prot_xyz.size and lig_xyz.size:
        diffs = lig_xyz[:, None, :] - prot_xyz[None, :, :]
        d2 = (diffs * diffs).sum(axis=-1)
        clashes = int((d2 < CLASH_DIST_SQ).sum())
    else:
        clashes = 0

    # MMFF94s energy
    try:
        mol_h = Chem.AddHs(mol, addCoords=True)
        props = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94s")
        if props is None:
            energy = float("nan")
        else:
            ff = AllChem.MMFFGetMoleculeForceField(mol_h, props)
            if ff is None:
                energy = float("nan")
            else:
                energy = float(ff.CalcEnergy())
    except Exception:
        energy = float("nan")

    return (energy, clashes, mol.GetNumHeavyAtoms())


def load_smiles_map() -> dict[str, str]:
    df = pd.read_csv(STRUCT_CSV)
    return dict(zip(df["structure"].astype(str), df["smiles"].astype(str)))


def main() -> None:
    smi_map = load_smiles_map()
    rows = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        with zipfile.ZipFile(V2_ZIP) as zv2, zipfile.ZipFile(V3_ZIP) as zv3:
            for cid in IDS:
                name = f"{cid}.pdb"
                smi = smi_map.get(cid, "")
                if not smi:
                    print(f"[WARN] no SMILES for {cid}")
                    rows.append(dict(id=cid, v2_energy=np.nan, v2_clashes=-1,
                                     v3_energy=np.nan, v3_clashes=-1,
                                     recommended_version="unknown",
                                     reason="no_smiles"))
                    continue

                try:
                    pdb_v2 = zv2.read(name).decode("utf-8", errors="ignore")
                except KeyError:
                    pdb_v2 = ""
                try:
                    pdb_v3 = zv3.read(name).decode("utf-8", errors="ignore")
                except KeyError:
                    pdb_v3 = ""

                lig_v2, prot_v2 = split_pdb_ligand_protein(pdb_v2) if pdb_v2 else ("", np.empty((0, 3)))
                lig_v3, prot_v3 = split_pdb_ligand_protein(pdb_v3) if pdb_v3 else ("", np.empty((0, 3)))

                e_v2, c_v2, n_v2 = score_pose(lig_v2, prot_v2, smi)
                e_v3, c_v3, n_v3 = score_pose(lig_v3, prot_v3, smi)

                # decision
                if np.isnan(e_v2) and np.isnan(e_v3):
                    rec, reason = "unknown", "both_failed"
                elif np.isnan(e_v2):
                    rec, reason = "v3", "v2_failed"
                elif np.isnan(e_v3):
                    rec, reason = "v2", "v3_failed"
                else:
                    e_better = "v2" if e_v2 < e_v3 else ("v3" if e_v3 < e_v2 else "tie")
                    c_better = "v2" if c_v2 < c_v3 else ("v3" if c_v3 < c_v2 else "tie")
                    if e_better == c_better and e_better != "tie":
                        rec = e_better
                        reason = "both_metrics_agree"
                    elif e_better != "tie" and c_better == "tie":
                        rec = e_better
                        reason = "energy_only"
                    elif c_better != "tie" and e_better == "tie":
                        rec = c_better
                        reason = "clash_only"
                    elif e_better == "tie" and c_better == "tie":
                        rec = "tie"
                        reason = "identical"
                    else:
                        # disagreement: prefer fewer clashes (safer geom),
                        # but record disagreement
                        rec = c_better
                        reason = "disagree_pick_lower_clash"

                row = dict(
                    id=cid,
                    v2_energy=round(e_v2, 3) if not np.isnan(e_v2) else np.nan,
                    v2_clashes=c_v2,
                    v3_energy=round(e_v3, 3) if not np.isnan(e_v3) else np.nan,
                    v3_clashes=c_v3,
                    n_heavy_atoms=max(n_v2, n_v3),
                    recommended_version=rec,
                    reason=reason,
                )
                print(f"{cid}: v2(E={row['v2_energy']}, clash={c_v2}) "
                      f"v3(E={row['v3_energy']}, clash={c_v3}) -> {rec} ({reason})")
                rows.append(row)

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}  ({len(out)} rows)")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
