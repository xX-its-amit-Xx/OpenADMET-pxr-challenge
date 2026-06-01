"""
nb750_chai_consensus_pose.py — Build structure_baseline_v5 from Chai-1 consensus
poses, falling back to v1 Boltz cofold for any ligand without local Chai CIFs.

Per-ligand pipeline:
  1. Inventory data/external/dargason_cofolding/ for Chai-1 CIFs.
     - model_manifest.csv is Boltz-only (no Chai rows).
     - activity_challenge_manifests_chai1_*_model_manifest.csv has Chai-1
       references but cif_path entries are inside HF .tar.gz archives (not
       extracted locally). We accept extracted CIFs at any of:
         data/external/dargason_cofolding/structures/chai1/{sid}/*.cif
         data/external/dargason_cofolding/chai1_extract/{sid}/*.cif
         data/external/chai1_predictions/{sid}/*.cif
       plus per-CIF confidence JSONs (chai pattern: pred.model_idx_N.cif
       paired with scores.model_idx_N.json or a single all_scores.json).
  2. For each of the 184 structure-track IDs, load all Chai CIFs available,
     read iptm/ptm/ligand_pLDDT, pick the highest-ipTM pose, and convert to
     PDB with residue name LIG for the ligand.
  3. If no Chai pose is available, fall back to the v1 Boltz cofold PDB
     (already validated in submissions/structure_baseline_v1.zip).
  4. Zip the 184 PDBs to submissions/structure_baseline_v5.zip.
  5. Validate via tutorial/validation/structure_validation.py.

Memory-safe: processes one ligand at a time, reads CIFs lazily, frees buffers.
Disk-safe: writes only the final 184 PDBs into a tempdir, then zips and
removes the tempdir. No bulk extraction or caching.
"""
from __future__ import annotations

import json
import shutil
import statistics
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd

REPO = Path(r"d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge")
COFOLD_ROOT = REPO / "data" / "external" / "dargason_cofolding"
CHAI_MANIFEST = (
    COFOLD_ROOT
    / "activity_challenge_manifests_chai1_msa_seed11_25_50_67_model_manifest.csv"
)
TEST_CSV = REPO / "data" / "raw" / "pxr-challenge_structure_TEST_BLINDED.csv"
V1_ZIP = REPO / "submissions" / "structure_baseline_v1.zip"
OUT_ZIP = REPO / "submissions" / "structure_baseline_v5.zip"
META_JSON = REPO / "data" / "processed" / "structure_baseline_v5_metadata.json"

# Candidate roots where extracted Chai-1 CIFs may live. We search in order.
CHAI_CANDIDATE_ROOTS: list[Path] = [
    COFOLD_ROOT / "structures" / "chai1",
    COFOLD_ROOT / "chai1_extract",
    REPO / "data" / "external" / "chai1_predictions",
    Path(r"d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-structure")
    / "data"
    / "external"
    / "chai1_predictions",
]

# Make tutorial validator importable
sys.path.insert(0, str(REPO / "tutorial"))
from validation.structure_validation import validate_structure_submission  # noqa: E402


def validate_zip_windows_safe(
    zip_path: Path,
    expected_ids: set[str],
    expected_ligand_smiles: dict[str, str],
) -> tuple[bool, list[str]]:
    """Mirror tutorial validate_structure_submission but use a manually managed
    tempdir with shutil.rmtree(ignore_errors=True). The tutorial implementation
    uses tempfile.TemporaryDirectory(), whose Windows cleanup races against
    MDAnalysis Universe file handles and raises WinError 32 ("file in use")
    even when the zip is structurally fine. The error mode is a Windows
    cleanup race, not a content defect — bypassing it gives us the real
    validation result.
    """
    import gc as _gc
    import warnings as _w
    import zipfile as _zf

    _w.filterwarnings("ignore")
    import MDAnalysis as mda  # type: ignore
    from rdkit import Chem, RDLogger  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore

    RDLogger.DisableLog("rdApp.*")
    errors: list[str] = []
    tmpdir = Path(tempfile.mkdtemp(prefix="val_v5_"))
    try:
        with _zf.ZipFile(zip_path) as zf:
            pdbs = [n for n in zf.namelist() if n.lower().endswith(".pdb")]
            submitted = {Path(n).stem for n in pdbs}
            missing = sorted(set(map(str, expected_ids)) - submitted)
            if missing:
                errors.append(f"Missing {len(missing)} expected structure(s): {missing[:20]}")
            for name in pdbs:
                tp = Path(zf.extract(name, path=tmpdir))
                try:
                    u = mda.Universe(str(tp))
                    lig = u.select_atoms("resname LIG")
                    if len(lig) == 0:
                        errors.append(f"{name}: Missing residue 'LIG'")
                        continue
                    if len(lig.residues) > 1:
                        errors.append(
                            f"{name}: Found {len(lig.residues)} 'LIG' residues, expected 1"
                        )
                    if len(u.segments) > 2:
                        errors.append(
                            f"{name}: Found {len(u.segments)} chains, expected 2 or fewer"
                        )
                    smi = expected_ligand_smiles.get(Path(name).stem)
                    if smi is not None:
                        ref = Chem.MolFromSmiles(smi)
                        ligp = tmpdir / f"{Path(name).stem}_lig.pdb"
                        lig.write(str(ligp))
                        lm = Chem.MolFromPDBFile(str(ligp), removeHs=True, sanitize=False)
                        if ref is None:
                            errors.append(f"{name}: Could not parse expected SMILES")
                        elif lm is None:
                            errors.append(f"{name}: RDKit could not parse LIG residue")
                        else:
                            try:
                                AllChem.AssignBondOrdersFromTemplate(ref, lm)
                            except ValueError:
                                errors.append(
                                    f"{name}: Ligand connectivity does not match expected SMILES"
                                )
                    del u, lig
                except Exception as e:
                    errors.append(f"{name}: MDAnalysis failed to parse file: {e}")
        _gc.collect()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return len(errors) == 0, errors


# ------------------------------- CIF -> PDB ------------------------------- #

def _try_biopython_cif_to_pdb(cif_path: Path, pdb_path: Path, lig_resname: str = "LIG") -> bool:
    """Convert CIF to PDB using Biopython if available. Rename ligand chain residue to LIG.

    Returns True if a PDB with at least one ATOM and one HETATM (LIG) line was written.
    """
    try:
        from Bio import PDB as biopdb  # type: ignore
    except Exception:
        return False
    parser = biopdb.MMCIFParser(QUIET=True)
    structure = parser.get_structure("X", str(cif_path))
    # Rename non-standard residues (anything not in the 20 standard AAs) to LIG.
    standard_aa = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        "HOH", "WAT",
    }
    for model in structure:
        for chain in model:
            for res in list(chain):
                resname = res.get_resname().strip().upper()
                if resname not in standard_aa:
                    res.resname = lig_resname
    io = biopdb.PDBIO()
    io.set_structure(structure)
    io.save(str(pdb_path))
    text = pdb_path.read_text(errors="ignore")
    has_atom = "\nATOM " in ("\n" + text)
    has_lig = f" {lig_resname} " in text or f"\n{lig_resname} " in text
    return has_atom and has_lig


def cif_to_pdb(cif_path: Path, pdb_path: Path) -> bool:
    """Convert CIF -> PDB, rename ligand residue to LIG.  Returns True on success."""
    return _try_biopython_cif_to_pdb(cif_path, pdb_path, lig_resname="LIG")


# ----------------------------- Chai inventory ----------------------------- #

def inventory_chai(structure_ids: set[str]) -> dict[str, list[Path]]:
    """Return mapping {structure_id -> list of locally available CIF paths}.

    Searches CHAI_CANDIDATE_ROOTS.  Ligands keyed by the structure-track ID
    (e.g. "x00011-1").  Returns empty list for any ligand with no CIFs.
    """
    out: dict[str, list[Path]] = {sid: [] for sid in structure_ids}
    for root in CHAI_CANDIDATE_ROOTS:
        if not root.exists():
            continue
        for sid in structure_ids:
            sub = root / sid
            if sub.is_dir():
                out[sid].extend(sorted(sub.glob("*.cif")))
    return out


def _load_chai_scores(cif: Path) -> dict:
    """Find and parse a confidence JSON next to a Chai CIF.

    Chai layouts seen in the wild:
      - per-CIF JSON: scores.model_idx_<k>.json or confidence_<k>.json
      - aggregated: all_scores.json (list of dicts keyed by sample_idx / model_idx)
    Returns a dict with at least one of: iptm, ptm, aggregate_score, ligand_plddt.
    """
    parent = cif.parent
    stem = cif.stem  # e.g. pred.model_idx_3
    idx = None
    if "model_idx_" in stem:
        try:
            idx = int(stem.split("model_idx_")[-1])
        except ValueError:
            idx = None
    if "_sample_" in stem:
        try:
            idx = int(stem.split("_sample_")[-1])
        except ValueError:
            pass
    # per-CIF candidates
    candidates = [
        parent / f"scores.model_idx_{idx}.json",
        parent / f"confidence_{idx}.json",
        parent / f"scores_{idx}.json",
        parent / f"{stem}.scores.json",
    ]
    for c in candidates:
        if c.exists():
            try:
                return json.loads(c.read_text())
            except Exception:
                pass
    # aggregated all_scores.json
    agg = parent / "all_scores.json"
    if agg.exists():
        try:
            arr = json.loads(agg.read_text())
            if isinstance(arr, list):
                for s in arr:
                    if s.get("sample_idx") == idx or s.get("model_idx") == idx:
                        return s
                if arr:
                    return arr[0]
        except Exception:
            pass
    return {}


def pick_consensus_pose(cifs: list[Path]) -> tuple[Path | None, dict]:
    """Pick highest-ipTM Chai pose from a list of CIFs.  Falls back to first
    pose if no confidence JSONs are findable.
    """
    if not cifs:
        return None, {}
    best: tuple[Path, dict, float] | None = None
    for cif in cifs:
        scores = _load_chai_scores(cif)
        iptm = scores.get("iptm")
        if iptm is None:
            iptm = scores.get("aggregate_score", -1.0)
        try:
            iptm = float(iptm)
        except (TypeError, ValueError):
            iptm = -1.0
        if best is None or iptm > best[2]:
            best = (cif, scores, iptm)
    assert best is not None
    return best[0], best[1]


# ------------------------------ Build & ship ------------------------------ #

def main() -> int:
    if not TEST_CSV.exists():
        print(f"[FATAL] missing {TEST_CSV}")
        return 1
    if not V1_ZIP.exists():
        print(f"[FATAL] missing v1 fallback zip {V1_ZIP}")
        return 1

    df = pd.read_csv(TEST_CSV)
    expected_ids = set(df["structure"].astype(str))
    expected_ligand_smiles = dict(
        zip(df["structure"].astype(str), df["smiles"].astype(str))
    )
    print(f"[INFO] expected structure IDs: {len(expected_ids)}")

    # Chai-1 manifest sanity print
    if CHAI_MANIFEST.exists():
        chai_targets = set(
            pd.read_csv(CHAI_MANIFEST, usecols=["target_id"])["target_id"].astype(str)
        )
        print(
            f"[INFO] chai1 activity manifest covers {len(chai_targets)} OADMET ligand IDs "
            f"(activity track keying; structure track uses x*-1 IDs and is not in this "
            f"manifest)."
        )
    else:
        print(f"[WARN] chai1 manifest not found at {CHAI_MANIFEST}")

    # Inventory locally extracted Chai-1 CIFs (by structure-track ID)
    chai_by_id = inventory_chai(expected_ids)
    chai_with_cifs = {sid: cifs for sid, cifs in chai_by_id.items() if cifs}
    print(
        f"[INFO] local Chai-1 CIFs available for {len(chai_with_cifs)}/"
        f"{len(expected_ids)} structure-track ligands."
    )

    # Extract v1 fallback PDBs to a tempdir we can read from per-ligand.
    with tempfile.TemporaryDirectory(prefix="v1_fallback_") as v1_tmp, tempfile.TemporaryDirectory(
        prefix="v5_build_"
    ) as out_tmp:
        v1_tmp_p = Path(v1_tmp)
        out_tmp_p = Path(out_tmp)
        with zipfile.ZipFile(V1_ZIP) as zf:
            zf.extractall(v1_tmp_p)
        v1_pdbs = {p.stem: p for p in v1_tmp_p.glob("*.pdb")}
        missing_v1 = expected_ids - set(v1_pdbs)
        if missing_v1:
            print(f"[FATAL] {len(missing_v1)} expected IDs missing from v1 fallback")
            return 2

        # Build 184 PDBs
        chai_iptms: list[float] = []
        n_chai = 0
        n_fallback = 0
        per_ligand: list[dict] = []
        for sid in sorted(expected_ids):
            out_pdb = out_tmp_p / f"{sid}.pdb"
            used_chai = False
            iptm_used: float | None = None
            cifs = chai_by_id.get(sid, [])
            if cifs:
                cif, scores = pick_consensus_pose(cifs)
                if cif is not None and cif_to_pdb(cif, out_pdb):
                    used_chai = True
                    iptm_used = scores.get("iptm")
                    if iptm_used is not None:
                        try:
                            chai_iptms.append(float(iptm_used))
                        except (TypeError, ValueError):
                            pass
            if not used_chai:
                # Fall back to v1 Boltz cofold PDB
                shutil.copy(v1_pdbs[sid], out_pdb)
            if used_chai:
                n_chai += 1
            else:
                n_fallback += 1
            per_ligand.append(
                {
                    "structure": sid,
                    "src": "chai1" if used_chai else "boltz_v1",
                    "iptm": iptm_used,
                    "n_chai_cifs": len(cifs),
                }
            )

        print(f"[INFO] chai-covered: {n_chai}; v1-boltz fallback: {n_fallback}")
        mean_iptm = statistics.mean(chai_iptms) if chai_iptms else float("nan")
        print(f"[INFO] mean ipTM on Chai-selected poses: {mean_iptm}")

        # Zip 184 PDBs
        OUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
        if OUT_ZIP.exists():
            OUT_ZIP.unlink()
        pdbs = sorted(out_tmp_p.glob("*.pdb"))
        with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for p in pdbs:
                zf.write(p, arcname=p.name)
        size_mb = OUT_ZIP.stat().st_size / 1e6
        print(f"[INFO] wrote {OUT_ZIP} ({size_mb:.1f} MB, {len(pdbs)} PDBs)")

        # Validate. Use the Windows-safe wrapper because the tutorial
        # validator's tempdir cleanup races MDAnalysis file handles on
        # Windows; the wrapper performs identical structural checks.
        print("[INFO] running structure validation (Windows-safe wrapper) ...")
        ok, errors = validate_zip_windows_safe(
            OUT_ZIP,
            expected_ids=expected_ids,
            expected_ligand_smiles=expected_ligand_smiles,
        )
        print(f"[RESULT] valid={ok}, n_errors={len(errors)}")
        for e in errors[:10]:
            print(f"  - {e}")
        # Also call the unmodified tutorial validator for traceability
        # (may return WinError 32 due to known Windows tempdir race).
        tutorial_ok, tutorial_errs = validate_structure_submission(
            OUT_ZIP,
            expected_ids=expected_ids,
            expected_ligand_smiles=expected_ligand_smiles,
            require_lig_resname=True,
        )
        print(
            f"[INFO] tutorial validator (reference): valid={tutorial_ok}, "
            f"n_errors={len(tutorial_errs)}"
        )

        # Write metadata
        META_JSON.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "name": "structure_baseline_v5",
            "path": "submissions/structure_baseline_v5.zip",
            "generator": "Chai-1 consensus pose (highest ipTM) with v1 Boltz cofold fallback",
            "size_bytes": OUT_ZIP.stat().st_size,
            "n_files": len(pdbs),
            "expected_n_files": 184,
            "n_chai_covered": n_chai,
            "n_fallback_to_boltz_v1": n_fallback,
            "mean_iptm_chai_selected": mean_iptm if chai_iptms else None,
            "chai_iptm_n": len(chai_iptms),
            "validation": {
                "valid": ok,
                "n_errors": len(errors),
                "errors_first10": errors[:10],
                "validator": "validate_zip_windows_safe (mirrors tutorial validator)",
                "tutorial_validator_valid": tutorial_ok,
                "tutorial_validator_n_errors": len(tutorial_errs),
                "tutorial_validator_note": (
                    "Tutorial validator may report WinError 32 on Windows due to "
                    "MDAnalysis tempdir cleanup race; the Windows-safe wrapper "
                    "runs identical structural checks."
                ),
            },
            "chai_candidate_roots_checked": [str(r) for r in CHAI_CANDIDATE_ROOTS],
            "chai_manifest": str(CHAI_MANIFEST),
            "fallback_zip": str(V1_ZIP),
            "per_ligand_first10": per_ligand[:10],
        }
        META_JSON.write_text(json.dumps(meta, indent=2))
        print(f"[INFO] metadata -> {META_JSON}")

        return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
