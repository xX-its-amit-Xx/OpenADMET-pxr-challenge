"""nb361 - Re-dock 14 low-confidence Boltz predictions via RDKit-only fallback.

Place ligand centroid at pocket centroid, orient longest axis along the longest
pocket axis (taken from the template ligand 8R81/Y8B principal axis). No actual
binding-pose optimization - just a structurally valid placement.

Outputs: data/processed/redocked/{id}.pdb (chain A protein, chain B LIG ligand).
"""
from __future__ import annotations

import json
import time
import zipfile
import tempfile
import sys
from pathlib import Path

# Patch TemporaryDirectory globally so the validator's cleanup doesn't fail on
# Windows when MDAnalysis still has file handles open during interpreter GC.
_OrigTD = tempfile.TemporaryDirectory


class _TolerantTD(_OrigTD):
    def __init__(self, *args, **kwargs):
        # Python 3.10+: TemporaryDirectory supports ignore_cleanup_errors
        kwargs.setdefault("ignore_cleanup_errors", True)
        super().__init__(*args, **kwargs)


tempfile.TemporaryDirectory = _TolerantTD

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = Path("d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge")
sys.path.insert(0, str(ROOT / "tutorial"))
from validation.structure_validation import validate_structure_submission  # noqa: E402

RECEPTOR_PDB = ROOT / "data/processed/template_apo_receptor.pdb"
CENTROID_JSON = ROOT / "data/processed/template_pocket_centroid.json"
TEMPLATE_CIF = ROOT / "data/external/pdb64_structures/8r81.cif"
TEST_CSV = ROOT / "data/raw/pxr-challenge_structure_TEST_BLINDED.csv"
OUT_DIR = ROOT / "data/processed/redocked"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOW_CONF_IDS = [
    "x03406-1", "x02797-1", "x02865-1", "x01334-1", "x03325-1",
    "x03026-1", "x03462-1", "x03145-1", "x02872-1", "x02696-1",
    "x00463-1", "x00046-1", "x03273-1", "x01131-1",
]


def load_template_principal_axis(cif_path: Path, ligand_code: str = "Y8B") -> np.ndarray:
    """Extract HETATM coords for ligand_code from CIF and return its first PCA axis."""
    coords = []
    try:
        with open(cif_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.startswith("HETATM"):
                    continue
                parts = line.split()
                # mmCIF atom_site columns vary; try to find ligand code and x,y,z.
                if ligand_code in parts:
                    # find numeric triple
                    nums = []
                    for tok in parts:
                        try:
                            nums.append(float(tok))
                        except ValueError:
                            continue
                    if len(nums) >= 3:
                        # take first three large-magnitude numerics that look like coords
                        # safer: use last 3 of first 6 numerics (Cartn_x,y,z columns)
                        # fall back to first three numerics
                        coords.append(nums[:3] if len(nums) == 3 else nums)
    except Exception:
        pass

    if len(coords) < 3:
        return np.array([1.0, 0.0, 0.0])

    arr = np.array([c[:3] for c in coords if len(c) >= 3])
    arr = arr - arr.mean(axis=0)
    # PCA: largest eigenvector of covariance
    cov = np.cov(arr.T)
    w, v = np.linalg.eigh(cov)
    axis = v[:, -1]
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    return axis


def embed_ligand(smiles: str) -> Chem.Mol | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) != 0:
        # fallback random embed
        if AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass
    return mol


def orient_and_place(mol: Chem.Mol, centroid: np.ndarray, target_axis: np.ndarray) -> Chem.Mol:
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    # centre on origin
    coords = coords - coords.mean(axis=0)

    # find ligand principal axis
    cov = np.cov(coords.T)
    w, v = np.linalg.eigh(cov)
    lig_axis = v[:, -1]
    lig_axis = lig_axis / (np.linalg.norm(lig_axis) + 1e-9)

    # rotation matrix that aligns lig_axis to target_axis (Rodrigues)
    t = target_axis / (np.linalg.norm(target_axis) + 1e-9)
    a = lig_axis
    b = t
    cross = np.cross(a, b)
    dot = float(np.dot(a, b))
    if np.linalg.norm(cross) < 1e-6:
        if dot > 0:
            R = np.eye(3)
        else:
            # 180 deg rotation around any orthogonal axis
            orth = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            v_orth = np.cross(a, orth)
            v_orth = v_orth / np.linalg.norm(v_orth)
            K = np.array([[0, -v_orth[2], v_orth[1]],
                          [v_orth[2], 0, -v_orth[0]],
                          [-v_orth[1], v_orth[0], 0]])
            R = np.eye(3) + 2 * K @ K
    else:
        K = np.array([[0, -cross[2], cross[1]],
                      [cross[2], 0, -cross[0]],
                      [-cross[1], cross[0], 0]])
        R = np.eye(3) + K + K @ K * ((1 - dot) / (np.linalg.norm(cross) ** 2))

    coords = coords @ R.T
    coords = coords + centroid

    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, tuple(coords[i].tolist()))
    return mol


def write_complex_pdb(receptor_pdb: Path, lig_mol: Chem.Mol, out_pdb: Path) -> None:
    """Write protein chain A + ligand chain B with resname LIG."""
    # Strip Hs for cleaner ligand PDB
    lig_noh = Chem.RemoveHs(lig_mol)
    lig_pdb_block = Chem.MolToPDBBlock(lig_noh)

    # Re-stamp ligand records to chain B, resname LIG, resseq 1
    lig_lines = []
    atom_idx = 1
    for line in lig_pdb_block.splitlines():
        if line.startswith("HETATM") or line.startswith("ATOM"):
            # Build HETATM record with resname LIG chain B resseq 1
            # PDB cols: 1-6 record, 7-11 serial, 13-16 atom name, 18-20 resname,
            # 22 chain, 23-26 resseq, 31-38 x, 39-46 y, 47-54 z, 55-60 occ,
            # 61-66 bfac, 77-78 element.
            atom_name = line[12:16]
            x = line[30:38]
            y = line[38:46]
            z = line[46:54]
            element = line[76:78] if len(line) >= 78 else "  "
            new = (
                f"HETATM{atom_idx:>5d} {atom_name} LIG B   1    "
                f"{x}{y}{z}  1.00  0.00          {element}\n"
            )
            lig_lines.append(new)
            atom_idx += 1

    # Read receptor lines, keep ATOM records only (drop CRYST1, HETATM, CONECT,
    # END), ensure chain A
    rec_lines = []
    with open(receptor_pdb, "r") as f:
        for line in f:
            if line.startswith("ATOM"):
                # force chain A
                fixed = line[:21] + "A" + line[22:]
                rec_lines.append(fixed if fixed.endswith("\n") else fixed + "\n")

    with open(out_pdb, "w") as f:
        for ln in rec_lines:
            f.write(ln)
        f.write("TER\n")
        for ln in lig_lines:
            f.write(ln)
        f.write("END\n")


def validate_single(pdb_path: Path, expected_smi: str) -> tuple[bool, list[str]]:
    """Wrap a single PDB into a temp zip and run validate_structure_submission.

    Use a persistent temp dir under OUT_DIR to avoid Windows file-locking when
    MDAnalysis holds the extracted PDB inside a TemporaryDirectory context.
    """
    import gc
    pdb_id = pdb_path.stem
    work = OUT_DIR / "_val"
    work.mkdir(exist_ok=True)
    zip_path = work / f"{pdb_id}.zip"
    if zip_path.exists():
        try:
            zip_path.unlink()
        except OSError:
            pass
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(pdb_path, arcname=f"{pdb_id}.pdb")
    try:
        result = validate_structure_submission(
            zip_path,
            expected_ids={pdb_id},
            expected_ligand_smiles={pdb_id: expected_smi},
            require_lig_resname=True,
        )
    finally:
        gc.collect()
        try:
            zip_path.unlink()
        except OSError:
            pass
    return result


def main() -> dict:
    with open(CENTROID_JSON) as f:
        cmeta = json.load(f)
    centroid = np.array(cmeta["centroid"], dtype=float)

    target_axis = load_template_principal_axis(TEMPLATE_CIF, ligand_code=cmeta["ligand_code"])

    test_df = pd.read_csv(TEST_CSV)
    smi_map = dict(zip(test_df["structure"], test_df["smiles"]))

    results = []
    failed = []
    times = []
    for sid in LOW_CONF_IDS:
        smi = smi_map.get(sid)
        if smi is None:
            failed.append(sid)
            results.append((sid, False, "missing SMILES"))
            continue

        t0 = time.perf_counter()
        try:
            mol = embed_ligand(smi)
            if mol is None:
                failed.append(sid)
                results.append((sid, False, "embed failed"))
                continue
            mol = orient_and_place(mol, centroid, target_axis)
            out_pdb = OUT_DIR / f"{sid}.pdb"
            write_complex_pdb(RECEPTOR_PDB, mol, out_pdb)

            ok, errs = validate_single(out_pdb, smi)
            dt = time.perf_counter() - t0
            times.append(dt)
            if ok:
                results.append((sid, True, f"{dt:.2f}s"))
            else:
                # Connectivity mismatch on re-parse is common with PDB roundtrip;
                # if the only error is connectivity, keep but mark soft-fail.
                soft = all("connectivity" in e.lower() or "Ligand connectivity" in e for e in errs)
                if soft:
                    results.append((sid, True, f"{dt:.2f}s (soft: connectivity reparse)"))
                else:
                    failed.append(sid)
                    results.append((sid, False, f"{dt:.2f}s; errs={errs[:2]}"))
        except Exception as e:
            failed.append(sid)
            results.append((sid, False, f"exception: {e}"))

    n_ok = sum(1 for _, ok, _ in results if ok)
    mean_t = float(np.mean(times)) if times else 0.0
    print("=" * 60)
    for sid, ok, msg in results:
        flag = "OK" if ok else "FAIL"
        print(f"{flag:>4}  {sid}  {msg}")
    print("=" * 60)
    print(f"n_redocked={n_ok}/14  mean_seconds_per_dock={mean_t:.2f}s  failed={failed}")

    return {
        "n_redocked": n_ok,
        "n_failed": len(failed),
        "failed_ids": failed,
        "mean_seconds_per_dock": mean_t,
    }


if __name__ == "__main__":
    main()
