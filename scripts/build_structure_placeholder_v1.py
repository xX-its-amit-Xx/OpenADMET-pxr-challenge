"""
Build placeholder structure-track submission v1 from the tutorial example PDB pack.

The tutorial ships 184 PDB files in tutorial/outputs/example_structure_submission/
that match exactly the 184 test structure IDs and contain a 'LIG' residue with the
expected ligand connectivity. We zip them as-is, validate via the official
structure_validation.py, and write to submissions/structure_placeholder_v1.zip.

This is a guaranteed-valid baseline submission so we have something on the
structure-track leaderboard. Will be superseded by smina/Boltz dockings later.
"""
from __future__ import annotations
import sys
import zipfile
from pathlib import Path

import pandas as pd

REPO = Path(r"d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge")
SRC_DIR = REPO / "tutorial" / "outputs" / "example_structure_submission"
OUT_ZIP = REPO / "submissions" / "structure_placeholder_v1.zip"
TEST_CSV = REPO / "data" / "raw" / "pxr-challenge_structure_TEST_BLINDED.csv"

# Make validator importable
sys.path.insert(0, str(REPO / "tutorial"))
from validation.structure_validation import validate_structure_submission  # noqa: E402


def main() -> int:
    pdb_files = sorted(SRC_DIR.glob("*.pdb"))
    if len(pdb_files) != 184:
        print(f"[FATAL] expected 184 PDBs in {SRC_DIR}, found {len(pdb_files)}")
        return 1

    df = pd.read_csv(TEST_CSV)
    expected_ids = set(df["structure"].astype(str))
    expected_ligand_smiles = dict(zip(df["structure"].astype(str), df["smiles"].astype(str)))
    print(f"[INFO] expected_ids: {len(expected_ids)}; pdb files: {len(pdb_files)}")

    src_ids = {p.stem for p in pdb_files}
    missing = expected_ids - src_ids
    extra = src_ids - expected_ids
    if missing:
        print(f"[WARN] {len(missing)} expected IDs not in tutorial pack: {sorted(missing)[:10]}")
    if extra:
        print(f"[WARN] {len(extra)} extra IDs in tutorial pack: {sorted(extra)[:10]}")

    OUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()

    # Write zip with flat layout (just PDB basenames)
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in pdb_files:
            zf.write(p, arcname=p.name)
    size_mb = OUT_ZIP.stat().st_size / 1e6
    print(f"[INFO] wrote {OUT_ZIP} ({size_mb:.1f} MB, {len(pdb_files)} PDBs)")

    print("[INFO] running structure_validation.validate_structure_submission ...")
    ok, errors = validate_structure_submission(
        OUT_ZIP,
        expected_ids=expected_ids,
        expected_ligand_smiles=expected_ligand_smiles,
        require_lig_resname=True,
    )
    print(f"[RESULT] ok={ok}, n_errors={len(errors)}")
    for e in errors[:20]:
        print(f"  - {e}")
    if not ok:
        return 2
    print("[OK] placeholder structure submission validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
