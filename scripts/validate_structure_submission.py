"""Build (if needed) and validate the structure submission zip."""
from __future__ import annotations
import sys
import zipfile
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tutorial.validation.structure_validation import validate_structure_submission as _orig_validate

# Windows tempdir-cleanup workaround: validator opens PDBs with MDAnalysis which
# can leave file handles, breaking TemporaryDirectory cleanup. Wrap in a try.
import zipfile, tempfile
import MDAnalysis as mda
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem


def validate_structure_submission(structure_predictions_file, expected_ids=None,
                                  expected_ligand_smiles=None, require_lig_resname=True):
    errors = []
    path = Path(structure_predictions_file)
    if not path.exists():
        return False, [f"File does not exist: {path}"]
    if path.suffix.lower() != ".zip":
        return False, ["Structure predictions file must be a .zip file."]

    tmpdir = Path(tempfile.mkdtemp(prefix="pxr_struct_"))
    try:
        with zipfile.ZipFile(path, "r") as zf:
            pdb_files = [n for n in zf.namelist() if n.lower().endswith(".pdb")]
            if not pdb_files:
                return False, ["Zip file contains no PDB files."]
            submitted_ids = {Path(n).stem for n in pdb_files}
            if expected_ids is not None:
                expected_ids = {str(x) for x in expected_ids}
                missing = sorted(expected_ids - submitted_ids)
                if missing:
                    errors.append(f"Missing {len(missing)} expected structure(s): {missing[:20]}")
            elif len(pdb_files) != 184:
                errors.append(f"Zip contains {len(pdb_files)} pdbs, expected 184.")

            if require_lig_resname:
                for name in pdb_files:
                    tmp_path = zf.extract(name, path=str(tmpdir))
                    try:
                        u = mda.Universe(tmp_path)
                        ligands = u.select_atoms("resname LIG")
                        if len(ligands) == 0:
                            errors.append(f"{name}: Missing residue 'LIG'")
                            del u
                            continue
                        if len(ligands.residues) > 1:
                            errors.append(f"{name}: Found {len(ligands.residues)} 'LIG' residues, expected 1")
                        if len(u.segments) > 2:
                            errors.append(f"{name}: Found {len(u.segments)} chains, expected 2 or fewer")
                        if expected_ligand_smiles is not None:
                            pdb_id = Path(name).stem
                            expected_smi = expected_ligand_smiles.get(pdb_id)
                            if expected_smi is not None:
                                ref_mol = Chem.MolFromSmiles(expected_smi)
                                if ref_mol is None:
                                    errors.append(f"{name}: Could not parse expected SMILES")
                                else:
                                    lig_pdb_path = tmpdir / f"{pdb_id}_lig.pdb"
                                    ligands.write(str(lig_pdb_path))
                                    lig_mol = Chem.MolFromPDBFile(str(lig_pdb_path), removeHs=True, sanitize=False)
                                    if lig_mol is None:
                                        errors.append(f"{name}: RDKit could not parse LIG residue")
                                    else:
                                        try:
                                            RDLogger.DisableLog("rdApp.*")
                                            AllChem.AssignBondOrdersFromTemplate(ref_mol, lig_mol)
                                            RDLogger.EnableLog("rdApp.*")
                                        except ValueError:
                                            RDLogger.EnableLog("rdApp.*")
                                            errors.append(f"{name}: Ligand connectivity mismatch vs SMILES")
                        del u, ligands
                    except Exception as e:
                        errors.append(f"{name}: MDAnalysis failed: {e}")
    except zipfile.BadZipFile:
        return False, ["File is not a valid zip archive."]
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
    return len(errors) == 0, errors

EXAMPLE_DIR = ROOT / "tutorial" / "outputs" / "example_structure_submission"
ZIP_PATH = ROOT / "tutorial" / "outputs" / "pxr_structure_submission.zip"
TEST_CSV = ROOT / "data" / "raw" / "pxr-challenge_structure_TEST_BLINDED.csv"


def build_zip(src_dir: Path, out_zip: Path) -> None:
    pdbs = sorted(src_dir.glob("*.pdb"))
    print(f"Building zip from {len(pdbs)} PDBs in {src_dir}")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in pdbs:
            zf.write(p, arcname=p.name)
    print(f"Wrote {out_zip} ({out_zip.stat().st_size/1024:.1f} KB)")


def main() -> int:
    if not ZIP_PATH.exists():
        if not EXAMPLE_DIR.exists():
            print(f"FATAL: neither zip nor example dir exists ({EXAMPLE_DIR})")
            return 2
        build_zip(EXAMPLE_DIR, ZIP_PATH)
    else:
        print(f"Using existing zip {ZIP_PATH}")

    df = pd.read_csv(TEST_CSV)
    expected_ids = set(df["structure"].astype(str))
    expected_smiles = dict(zip(df["structure"].astype(str), df["smiles"].astype(str)))
    print(f"Expected {len(expected_ids)} structures from {TEST_CSV.name}")

    ok, errors = validate_structure_submission(
        ZIP_PATH,
        expected_ids=expected_ids,
        expected_ligand_smiles=expected_smiles,
        require_lig_resname=True,
    )

    print("\n=== RESULT ===")
    print(f"VALID: {ok}")
    print(f"Errors: {len(errors)}")
    if errors:
        print("\n--- First 30 errors ---")
        for e in errors[:30]:
            print(" -", e)
        if len(errors) > 30:
            print(f"... and {len(errors)-30} more")

    # Summarize by PDB
    failed_pdbs = sorted({e.split(":")[0].split(".pdb")[0] for e in errors if ".pdb" in e})
    print(f"\nFailing PDB count: {len(failed_pdbs)}")
    if failed_pdbs:
        print("Failing IDs (first 30):", failed_pdbs[:30])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
