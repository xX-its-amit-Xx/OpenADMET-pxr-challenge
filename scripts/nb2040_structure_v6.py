"""nb2040 -- Structure ladder v6: per-ligand best-pose selection across v1-v5 zips.

CONTEXT (per memory feedback_structure_track_pivot, lb_actual_scores):
    Structure ladder exhausted: 4 entries all submitted (last 2026-06-02).
    - v4 (170 Boltz + 14 RDKit redocks) LB 0.4583 (rank 29/48; UNDERPERFORMED
      openadmet-boltz-baseline 0.4632 by -0.0049 because RDKit redocks hurt)
    - v1 (pure Boltz-2 cofold) expected ~0.46 LB, matches baseline
    - v5 == v1 (Chai-1 inputs missing on this disk; fell back to v1)
    - v2/v3 = same 14 redock IDs as v4 (DEPRECATED, same failure mode)
    Need a NEW candidate. Local box is FULL on D: -- no GPU, can't run fresh
    Boltz-2 multi-seed locally. Kaggle P100 path documented in memory but
    requires a USER ACTION (push to Kaggle, wait ~12h GPU). This script does
    what is locally achievable RIGHT NOW.

HYPOTHESIS:
    The 5 existing zips agree on 170/184 ligands (bit-identical Boltz cofolds)
    and differ only on the 14 LOWEST-confidence ligands (lig_conf 0.616-0.726,
    well below the corpus median 0.853). For those 14, v1 has the original
    Boltz pose, v2 has a global 8R81 RDKit redock, v3 has a per-ligand
    template redock, v4 has yet another redock variant.
    LB result so far: v1 (pure Boltz) ~0.46 > v4 (with RDKit redocks) 0.4583.
    => Pure Boltz wins ON AVERAGE, but per-ligand the optimal choice may
       differ. Some of the 14 redocks may individually beat Boltz; others
       hurt. A per-ligand POSE-QUALITY-SELECTED hybrid:
         - For 170 high-conf: use v1 pose (best Boltz)
         - For 14 low-conf:   choose per-ligand the pose with lowest
                              protein-ligand atom clash count AND ligand
                              center-of-mass closest to the v1 pocket centroid
       should match-or-beat v1 because we eliminate the worst redocks
       per-ligand while keeping the good ones (if any).
    Expected LB band 0.45-0.55. Lower bound: tied with v1 if all 14 picks
    revert to v1. Upper bound: improved if ANY redock pose is geometrically
    cleaner than v1 for its ligand.

PROTOCOL:
    1. Open all 5 existing zips and extract per-ligand PDB bytes.
    2. For each of the 14 disputed IDs:
       a. Parse each candidate PDB with MDAnalysis.
       b. Compute clash count (heavy-atom pairs within 1.5 A between protein
          and ligand). Lower = better.
       c. Compute ligand CoM distance to v1 pocket centroid (protein atoms
          within 5 A of v1 ligand). Lower = better (stays in pocket).
       d. Composite score: clash_count + 0.1 * dist_to_pocket
          (clash dominates; distance is tiebreaker).
       e. Pick the candidate with min score; record source zip.
    3. For all 170 unchallenged IDs: use v1 bytes verbatim.
    4. Write submissions/structure_v6_perlig_qsel.zip with exactly 184 PDBs.
    5. Validate with validate_structure_submission (Windows-safe wrapper).
    6. Append to STRUCTURE_LADDER as PRIMARY-1 (above the now-exhausted v5/v1)
    7. Write metadata JSON.

KAGGLE GPU PATH (user action, NOT executed by this script):
    For LDDT-PLI 0.55-0.75 (per memory band) need actual fresh Boltz-2
    multi-seed (seeds 0,1,42) on Kaggle P100 (12h compute):
       1. python scripts/kaggle_push.py --target boltz_structure_multiseed
       2. Trigger Kaggle kernel knowledgegraphlover/pxr-challenge-boltz
       3. Wait ~12h; pull predictions/<id>/seed_X/<id>_model_K.pdb
       4. Ensemble seeds via mean-iptm-weighted Kabsch alignment
       5. Re-run this script with --use-kaggle-multiseed flag for v7
    THIS SCRIPT does NOT touch Kaggle (user-initiated only).
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SUBM = REPO / "submissions"
DATA_PROCESSED = REPO / "data" / "processed"
TEST_CSV = REPO / "data" / "raw" / "pxr-challenge_structure_TEST_BLINDED.csv"

OUT_ZIP = SUBM / "structure_v6_perlig_qsel.zip"
OUT_META = DATA_PROCESSED / "structure_v6_perlig_qsel_metadata.json"

CANDIDATE_ZIPS = [
    SUBM / "structure_baseline_v1.zip",
    SUBM / "structure_baseline_v2.zip",
    SUBM / "structure_baseline_v3.zip",
    SUBM / "structure_baseline_v4.zip",
    SUBM / "structure_baseline_v5.zip",
]

POCKET_RADIUS = 5.0   # A; protein atoms within this of v1 ligand define pocket
CLASH_RADIUS = 1.5    # A; heavy-atom pairs closer than this are clashes
ALPHA_DIST = 0.1      # weight on pocket-distance tiebreaker


def _load_pdb_bytes(zips: List[Path]) -> Dict[str, Dict[str, bytes]]:
    """Returns {pdb_basename: {zip_stem: bytes}} for all zips."""
    out: Dict[str, Dict[str, bytes]] = {}
    for zp in zips:
        if not zp.exists():
            print(f"  SKIP (missing): {zp.name}")
            continue
        with zipfile.ZipFile(zp) as zf:
            for n in zf.namelist():
                if not n.lower().endswith(".pdb"):
                    continue
                out.setdefault(n, {})[zp.stem] = zf.read(n)
    return out


def _parse_pdb_coords(pdb_bytes: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (prot_xyz, lig_xyz) for a single PDB. Heavy atoms only."""
    prot, lig = [], []
    for line in pdb_bytes.decode("utf-8", errors="replace").splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        elem = line[76:78].strip()
        if elem == "H":
            continue
        try:
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
        except ValueError:
            continue
        resname = line[17:20].strip()
        coord = (x, y, z)
        if resname == "LIG" or line.startswith("HETATM"):
            lig.append(coord)
        else:
            prot.append(coord)
    return (np.asarray(prot, dtype=np.float32),
            np.asarray(lig,  dtype=np.float32))


def _pocket_centroid(prot_xyz: np.ndarray, lig_xyz: np.ndarray,
                     radius: float) -> np.ndarray:
    """Centroid of protein atoms within `radius` of any ligand atom."""
    if len(prot_xyz) == 0 or len(lig_xyz) == 0:
        return np.zeros(3, dtype=np.float32)
    # broadcast distance: (n_prot, n_lig)
    d2 = np.sum((prot_xyz[:, None, :] - lig_xyz[None, :, :])**2, axis=2)
    near = (d2 < radius**2).any(axis=1)
    if not near.any():
        return prot_xyz.mean(axis=0)
    return prot_xyz[near].mean(axis=0)


def _clash_count(prot_xyz: np.ndarray, lig_xyz: np.ndarray,
                 radius: float) -> int:
    if len(prot_xyz) == 0 or len(lig_xyz) == 0:
        return 0
    d2 = np.sum((prot_xyz[:, None, :] - lig_xyz[None, :, :])**2, axis=2)
    return int((d2 < radius**2).sum())


def _score_candidate(pdb_bytes: bytes, ref_pocket: np.ndarray) -> Tuple[float, int, float]:
    """Return (composite_score, clash_count, dist_to_pocket)."""
    prot, lig = _parse_pdb_coords(pdb_bytes)
    if len(lig) == 0:
        return (1e9, 999999, 1e6)
    clash = _clash_count(prot, lig, CLASH_RADIUS)
    lig_com = lig.mean(axis=0)
    dist = float(np.linalg.norm(lig_com - ref_pocket))
    score = float(clash + ALPHA_DIST * dist)
    return (score, clash, dist)


def select_best_per_ligand(pdb_map: Dict[str, Dict[str, bytes]],
                           disputed_ids: List[str]) -> Dict[str, Tuple[str, dict]]:
    """For disputed PDBs, choose lowest-score candidate. Returns
    {pdb_name: (chosen_zip_stem, score_breakdown_dict)}."""
    out: Dict[str, Tuple[str, dict]] = {}
    for pid in disputed_ids:
        name = f"{pid}.pdb"
        sources = pdb_map.get(name, {})
        if not sources:
            continue
        # Use v1 ligand placement as the reference pocket centroid.
        v1_bytes = sources.get("structure_baseline_v1")
        if v1_bytes is None:
            v1_bytes = next(iter(sources.values()))
        v1_prot, v1_lig = _parse_pdb_coords(v1_bytes)
        ref_pocket = _pocket_centroid(v1_prot, v1_lig, POCKET_RADIUS)

        best_zip = None
        best_score = float("inf")
        breakdown: dict = {}
        for zstem, data in sources.items():
            score, clash, dist = _score_candidate(data, ref_pocket)
            breakdown[zstem] = {"score": score, "clash": clash, "dist": round(dist, 3)}
            if score < best_score:
                best_score = score
                best_zip = zstem
        out[name] = (best_zip, breakdown)
    return out


def _identify_disputed(pdb_map: Dict[str, Dict[str, bytes]]) -> List[str]:
    import hashlib
    disputed = []
    for name, sources in pdb_map.items():
        hashes = {hashlib.sha256(b).hexdigest() for b in sources.values()}
        if len(hashes) > 1:
            disputed.append(name.replace(".pdb", ""))
    return sorted(disputed)


def build_v6_zip(out_path: Path) -> Tuple[int, dict]:
    print(f"[INFO] Loading {len(CANDIDATE_ZIPS)} candidate zips...")
    pdb_map = _load_pdb_bytes(CANDIDATE_ZIPS)
    if not pdb_map:
        raise RuntimeError("No candidate zips found.")
    all_pdbs = sorted(pdb_map.keys())
    print(f"[INFO] {len(all_pdbs)} unique PDB names across all zips.")

    disputed_ids = _identify_disputed(pdb_map)
    print(f"[INFO] {len(disputed_ids)} disputed PDBs (differ across zips): "
          f"{disputed_ids[:5]}...")

    decisions = select_best_per_ligand(pdb_map, disputed_ids)

    # Build new zip
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    per_ligand_log = []
    src_counts: Dict[str, int] = {}
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as out_zf:
        for name in all_pdbs:
            sources = pdb_map[name]
            pid = name.replace(".pdb", "")
            if name in decisions:
                chosen_stem, breakdown = decisions[name]
                data = sources[chosen_stem]
                per_ligand_log.append({
                    "id": pid, "src": chosen_stem,
                    "scores": breakdown,
                })
            else:
                # consensus across all -- pick v1 by convention
                chosen_stem = "structure_baseline_v1"
                data = sources.get(chosen_stem) or next(iter(sources.values()))
            src_counts[chosen_stem] = src_counts.get(chosen_stem, 0) + 1
            out_zf.writestr(name, data)

    n_files = len(all_pdbs)
    summary = {
        "n_files": n_files,
        "n_disputed": len(disputed_ids),
        "n_consensus": n_files - len(disputed_ids),
        "src_counts": src_counts,
        "per_ligand_log": per_ligand_log,
    }
    return n_files, summary


def validate(out_path: Path) -> Tuple[bool, list]:
    sys.path.insert(0, str(REPO / "scripts"))
    from validate_structure_submission import validate_structure_submission
    df = pd.read_csv(TEST_CSV)
    expected_ids = set(df["structure"].astype(str))
    expected_smiles = dict(zip(df["structure"].astype(str),
                               df["smiles"].astype(str)))
    return validate_structure_submission(
        out_path,
        expected_ids=expected_ids,
        expected_ligand_smiles=expected_smiles,
        require_lig_resname=True,
    )


def update_ladder() -> bool:
    """Append v6 entry to STRUCTURE_LADDER in auto_submit_structure_ladder.py."""
    p = REPO / "scripts" / "auto_submit_structure_ladder.py"
    txt = p.read_text(encoding="utf-8")
    if "structure_v6_perlig_qsel.zip" in txt:
        print("[INFO] auto_submit_structure_ladder.py already has v6 entry; skipping.")
        return False
    # Insert v6 as new PRIMARY-1 ABOVE v5
    needle = '    {"file": "structure_baseline_v5.zip",'
    new_entry = (
        '    {"file": "structure_v6_perlig_qsel.zip", '
        '"note": "PRIMARY-1 (nb2040): per-ligand best-pose selection across v1-v5 by clash+pocket-distance score; 170 consensus + 14 disputed (lowest-confidence Boltz cofolds)"},\n'
    )
    new_txt = txt.replace(needle, new_entry + needle, 1)
    if new_txt == txt:
        print("[WARN] could not locate insertion point in ladder script.")
        return False
    p.write_text(new_txt, encoding="utf-8")
    print("[INFO] STRUCTURE_LADDER updated: v6 inserted as new PRIMARY-1.")
    return True


def main() -> int:
    print("=" * 60)
    print("nb2040 -- Structure v6 per-ligand pose-quality selection")
    print("=" * 60)
    n_files, summary = build_v6_zip(OUT_ZIP)

    if n_files != 184:
        print(f"[FATAL] expected 184 PDBs, wrote {n_files}")
        return 1
    size_mb = OUT_ZIP.stat().st_size / 1e6
    print(f"[INFO] wrote {OUT_ZIP.name} ({size_mb:.1f} MB, {n_files} PDBs)")
    print(f"[INFO] src_counts: {summary['src_counts']}")

    print("[INFO] validating ...")
    ok, errors = validate(OUT_ZIP)
    print(f"[RESULT] ok={ok}, n_errors={len(errors)}")
    for e in errors[:10]:
        print(f"  - {e}")

    summary["validation"] = {"ok": ok, "n_errors": len(errors),
                             "errors_first10": [str(e) for e in errors[:10]]}
    summary["path"] = str(OUT_ZIP)
    summary["size_bytes"] = OUT_ZIP.stat().st_size

    OUT_META.parent.mkdir(parents=True, exist_ok=True)
    OUT_META.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[INFO] metadata -> {OUT_META.name}")

    if not ok:
        print("[FAIL] Validation failed; NOT updating ladder.")
        return 2

    update_ladder()
    print("[DONE] v6 candidate built, validated, and ladder updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
