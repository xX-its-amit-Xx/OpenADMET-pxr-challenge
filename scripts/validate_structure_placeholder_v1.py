"""
Standalone validator that mirrors structure_validation.validate_structure_submission
but writes each extracted PDB to a unique file and explicitly closes MDAnalysis
universes before deleting (avoids Windows file-lock race in the tutorial validator).
"""
from __future__ import annotations
import sys
import zipfile
from pathlib import Path
import warnings
import tempfile
import os

import pandas as pd
import MDAnalysis as mda
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

REPO = Path(r"d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge")
ZIP_PATH = REPO / "submissions" / "structure_placeholder_v1.zip"
TEST_CSV = REPO / "data" / "raw" / "pxr-challenge_structure_TEST_BLINDED.csv"


def main() -> int:
    df = pd.read_csv(TEST_CSV)
    expected_ids = set(df["structure"].astype(str))
    smi_map = dict(zip(df["structure"].astype(str), df["smiles"].astype(str)))

    errors: list[str] = []
    n_lig_ok = 0
    n_conn_ok = 0
    n_conn_checked = 0

    tmpdir = Path(tempfile.mkdtemp(prefix="pxr_validate_"))
    try:
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            pdb_names = [n for n in zf.namelist() if n.lower().endswith(".pdb")]
            print(f"[INFO] zip contains {len(pdb_names)} PDBs")

            submitted_ids = {Path(n).stem for n in pdb_names}
            missing = expected_ids - submitted_ids
            extra = submitted_ids - expected_ids
            if missing:
                errors.append(f"Missing {len(missing)} expected IDs: {sorted(missing)[:10]}")
            if extra:
                errors.append(f"Extra {len(extra)} IDs not in test set: {sorted(extra)[:10]}")

            for name in pdb_names:
                pid = Path(name).stem
                out_pdb = tmpdir / f"{pid}.pdb"
                with zf.open(name) as src, open(out_pdb, "wb") as dst:
                    dst.write(src.read())
                try:
                    u = mda.Universe(str(out_pdb))
                    ligands = u.select_atoms("resname LIG")
                    if len(ligands) == 0:
                        errors.append(f"{name}: missing 'LIG' residue")
                        continue
                    if len(ligands.residues) > 1:
                        errors.append(f"{name}: {len(ligands.residues)} LIG residues, expected 1")
                    if len(u.segments) > 2:
                        errors.append(f"{name}: {len(u.segments)} chains, expected <=2")
                    n_lig_ok += 1

                    expected_smi = smi_map.get(pid)
                    if expected_smi:
                        n_conn_checked += 1
                        ref_mol = Chem.MolFromSmiles(expected_smi)
                        if ref_mol is None:
                            errors.append(f"{name}: bad expected SMILES")
                            continue
                        lig_pdb = tmpdir / f"{pid}_lig.pdb"
                        ligands.write(str(lig_pdb))
                        del u  # release file handle
                        lig_mol = Chem.MolFromPDBFile(str(lig_pdb), removeHs=True, sanitize=False)
                        if lig_mol is None:
                            errors.append(f"{name}: rdkit could not parse LIG")
                        else:
                            try:
                                AllChem.AssignBondOrdersFromTemplate(ref_mol, lig_mol)
                                n_conn_ok += 1
                            except ValueError:
                                errors.append(f"{name}: ligand connectivity != SMILES")
                except Exception as e:
                    errors.append(f"{name}: parse error {type(e).__name__}: {e}")
                finally:
                    try:
                        if out_pdb.exists():
                            os.remove(out_pdb)
                    except Exception:
                        pass

        print(f"[INFO] LIG-present OK : {n_lig_ok}/{len(pdb_names)}")
        print(f"[INFO] connectivity OK: {n_conn_ok}/{n_conn_checked}")
        print(f"[INFO] errors        : {len(errors)}")
        for e in errors[:15]:
            print(f"  - {e}")
        return 0 if not errors else 2
    finally:
        # best-effort cleanup
        for f in tmpdir.glob("*"):
            try:
                f.unlink()
            except Exception:
                pass
        try:
            tmpdir.rmdir()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
