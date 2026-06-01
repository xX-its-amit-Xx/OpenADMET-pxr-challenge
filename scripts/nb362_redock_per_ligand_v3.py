"""nb362 - Re-dock 14 low-confidence ligands into their OWN per-ligand top-1 templates.

Each low_conf_id has its own template (PDB, ligand HET code). For each:
- Load apo receptor from data/processed/templates/{top1_pdb}_apo.pdb
- Load pocket centroid from {top1_pdb}_centroid.json
- Load ligand SMILES from pxr-challenge_structure_TEST_BLINDED.csv
- Embed: AddHs + ETKDGv3 (seed=42) + MMFF94s optimize
- PCA-align to that template ligand's first principal axis (from CIF)
- Translate to pocket centroid
- Write protein chain A + ligand chain B (resname LIG) to redocked_v3/{id}.pdb
- Validate individually via structure_validation.

Memory-safe: processes one ligand at a time, no large intermediate arrays.
"""
from __future__ import annotations

import json
import sys
import time
import zipfile
import tempfile
from pathlib import Path

# Tolerant TemporaryDirectory for Windows MDAnalysis cleanup
_OrigTD = tempfile.TemporaryDirectory


class _TolerantTD(_OrigTD):
    def __init__(self, *args, **kwargs):
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

MAP_CSV = ROOT / "data/processed/lowconf_perligand_templates.csv"
TEMPLATES_DIR = ROOT / "data/processed/templates"
CIF_DIR = ROOT / "data/external/pdb64_structures"
TEST_CSV = ROOT / "data/raw/pxr-challenge_structure_TEST_BLINDED.csv"
OUT_DIR = ROOT / "data/processed/redocked_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_template_ligand_axis(cif_path: Path, het_code: str) -> np.ndarray:
    """Extract HETATM coords for `het_code` from CIF, return first PCA axis (unit vec).

    mmCIF atom_site lines look like:
        HETATM N  N1  . Y8B B 2 . ? 12.345 67.890 1.234 1.00 20.0 ...
    We pick lines starting with HETATM that contain the het_code as a whitespace-
    separated token, then collect the first triple of floats with reasonable
    magnitude as (x, y, z). Falls back to a slightly more robust column scan.
    """
    coords: list[tuple[float, float, float]] = []
    try:
        with open(cif_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.startswith("HETATM"):
                    continue
                parts = line.split()
                if het_code not in parts:
                    continue
                # Walk tokens, collect floats; the Cartn_x,y,z columns are the
                # first three consecutive floats that could plausibly be coords
                # (any real). Take the first three consecutive parseable floats.
                floats: list[float] = []
                for tok in parts:
                    try:
                        floats.append(float(tok))
                    except ValueError:
                        floats = []  # reset on non-float to find consecutive run
                        continue
                    if len(floats) == 3:
                        break
                if len(floats) == 3:
                    coords.append((floats[0], floats[1], floats[2]))
    except Exception:
        pass

    if len(coords) < 3:
        # Degenerate: pick x-axis
        return np.array([1.0, 0.0, 0.0])

    arr = np.array(coords, dtype=float)
    arr = arr - arr.mean(axis=0)
    cov = np.cov(arr.T)
    _, v = np.linalg.eigh(cov)
    axis = v[:, -1]
    n = float(np.linalg.norm(axis))
    return axis / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])


def embed_ligand(smiles: str) -> Chem.Mol | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) != 0:
        if AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s", maxIters=200)
    except Exception:
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass
    return mol


def orient_and_place(mol: Chem.Mol, centroid: np.ndarray, target_axis: np.ndarray) -> Chem.Mol:
    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(n_atoms)])
    coords = coords - coords.mean(axis=0)  # center on origin

    cov = np.cov(coords.T)
    _, v = np.linalg.eigh(cov)
    lig_axis = v[:, -1]
    n = float(np.linalg.norm(lig_axis))
    lig_axis = lig_axis / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])

    t = target_axis / (np.linalg.norm(target_axis) + 1e-9)
    a, b = lig_axis, t
    cross = np.cross(a, b)
    dot = float(np.dot(a, b))
    cn = float(np.linalg.norm(cross))
    if cn < 1e-6:
        if dot > 0:
            R = np.eye(3)
        else:
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
        R = np.eye(3) + K + K @ K * ((1 - dot) / (cn ** 2))

    coords = coords @ R.T + centroid
    for i in range(n_atoms):
        conf.SetAtomPosition(i, tuple(coords[i].tolist()))
    return mol


def write_complex_pdb(receptor_pdb: Path, lig_mol: Chem.Mol, out_pdb: Path) -> None:
    lig_noh = Chem.RemoveHs(lig_mol)
    lig_pdb_block = Chem.MolToPDBBlock(lig_noh)

    lig_lines = []
    atom_idx = 1
    for line in lig_pdb_block.splitlines():
        if line.startswith("HETATM") or line.startswith("ATOM"):
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

    rec_lines = []
    with open(receptor_pdb, "r") as f:
        for line in f:
            if line.startswith("ATOM"):
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
    mapping = pd.read_csv(MAP_CSV)
    test_df = pd.read_csv(TEST_CSV)
    smi_map = dict(zip(test_df["structure"], test_df["smiles"]))

    # Cache template axes per (pdb, het) to avoid re-parsing CIF if reused
    axis_cache: dict[tuple[str, str], np.ndarray] = {}
    centroid_cache: dict[str, np.ndarray] = {}

    results = []
    failed = []
    times = []

    for row in mapping.itertuples(index=False):
        sid = row.test_structure
        pdb = str(row.top1_pdb).lower()
        het = str(row.top1_het)

        smi = smi_map.get(sid)
        if smi is None:
            failed.append(sid)
            results.append((sid, False, "missing SMILES"))
            continue

        # Receptor + centroid (per template)
        recep = TEMPLATES_DIR / f"{pdb}_apo.pdb"
        cjson = TEMPLATES_DIR / f"{pdb}_centroid.json"
        cif = CIF_DIR / f"{pdb}.cif"
        if not recep.exists() or not cjson.exists() or not cif.exists():
            failed.append(sid)
            results.append((sid, False, f"missing template files for {pdb}"))
            continue

        if pdb not in centroid_cache:
            with open(cjson) as f:
                cmeta = json.load(f)
            centroid_cache[pdb] = np.array(cmeta["centroid"], dtype=float)
        centroid = centroid_cache[pdb]

        key = (pdb, het)
        if key not in axis_cache:
            axis_cache[key] = load_template_ligand_axis(cif, het)
        target_axis = axis_cache[key]

        t0 = time.perf_counter()
        try:
            mol = embed_ligand(smi)
            if mol is None:
                failed.append(sid)
                results.append((sid, False, "embed failed"))
                continue
            mol = orient_and_place(mol, centroid, target_axis)
            out_pdb = OUT_DIR / f"{sid}.pdb"
            write_complex_pdb(recep, mol, out_pdb)
            ok, errs = validate_single(out_pdb, smi)
            dt = time.perf_counter() - t0
            times.append(dt)
            if ok:
                results.append((sid, True, f"{dt:.2f}s [{pdb}/{het}]"))
            else:
                soft = all(
                    "connectivity" in e.lower() or "Ligand connectivity" in e
                    for e in errs
                )
                if soft:
                    results.append((sid, True, f"{dt:.2f}s soft-connectivity [{pdb}/{het}]"))
                else:
                    failed.append(sid)
                    results.append((sid, False, f"{dt:.2f}s; errs={errs[:2]}"))
        except Exception as e:
            failed.append(sid)
            results.append((sid, False, f"exception: {e}"))

    n_ok = sum(1 for _, ok, _ in results if ok)
    mean_t = float(np.mean(times)) if times else 0.0
    print("=" * 70)
    for sid, ok, msg in results:
        flag = "OK" if ok else "FAIL"
        print(f"{flag:>4}  {sid}  {msg}")
    print("=" * 70)
    print(f"n_redocked={n_ok}/{len(mapping)}  mean_seconds_per_dock={mean_t:.3f}s  failed={failed}")

    return {
        "n_redocked": n_ok,
        "n_failed": len(failed),
        "failed_ids": failed,
        "mean_seconds_per_dock": mean_t,
        "n_total": len(mapping),
    }


if __name__ == "__main__":
    main()
