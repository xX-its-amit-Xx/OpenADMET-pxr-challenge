"""nb2073 -- ChEMBL PXR (NR1I2 / CHEMBL3401) delta probe vs cycle-130 snapshot.

GOAL:
    Decide whether a fresh pull of ChEMBL PXR bioactivity contains enough
    novel-scaffold pEC50 records to be worth a Phase-2 augmented retrain
    (queued as nb2078, cycle 160). Otherwise mark the axis CLOSED.

PROTOCOL:
    1. Pull current ChEMBL34 PXR (CHEMBL3401) bioactivity. Prefer the ChEMBL
       MCP tool if reachable, else hit the public REST API directly.
    2. Compare InChIKey set against the cycle-130 snapshot at
       data/external/chembl_pxr_nr_kb.parquet (PXR rows only) and
       data/external/chembl_pxr_CHEMBL3401.parquet.
    3. Report new compound count, novel-scaffold count, and the
       max-Tanimoto-to-test-513 distribution for the delta.
    4. Gate: if >= 50 new pEC50 records carrying novel scaffolds, write a
       queue marker for nb2078; else mark axis CLOSED with rationale.
    5. Persist summary JSON to data/processed/nb2073_summary.json.

NOTE:
    Read-only probe. Does NOT write any submission CSV. Does NOT touch the
    leaderboard ladder. nb2078 is gated downstream on this summary.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize  # noqa: E402

TAG = "nb2073"
TARGET_ID = "CHEMBL3401"  # NR1I2 / PXR
SNAPSHOT_KB = REPO / "data" / "external" / "chembl_pxr_nr_kb.parquet"
SNAPSHOT_RAW = REPO / "data" / "external" / "chembl_pxr_CHEMBL3401.parquet"
TEST_CSV = REPO / "data" / "raw" / "pxr-challenge_TEST_BLINDED.csv"

OUT_SUMMARY = REPO / "data" / "processed" / "nb2073_summary.json"
QUEUE_MARKER = REPO / "data" / "processed" / "nb2078_queued.flag"
CACHE_NEW_PULL = REPO / "data" / "processed" / "nb2073_chembl_current.parquet"

GATE_MIN_NEW_PEC50_NOVEL_SCAFFOLD = 50


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def std_inchikey(smi: str) -> Optional[str]:
    mol = standardize(smi)
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def std_scaffold(smi: str) -> Optional[str]:
    mol = standardize(smi)
    if mol is None:
        return None
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaf, canonical=True) if scaf else ""
    except Exception:
        return None


def morgan_arr(smiles_list):
    """Return list of explicit RDKit BitVect (radius 2, 2048 bits)."""
    out = []
    for smi in smiles_list:
        mol = standardize(smi)
        if mol is None:
            out.append(None)
            continue
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            out.append(fp)
        except Exception:
            out.append(None)
    return out


def max_tanimoto_to_test(query_fps, test_fps):
    """For each query fp, return max Tanimoto across the test_fps set."""
    out = np.full(len(query_fps), np.nan, dtype=np.float32)
    for i, qfp in enumerate(query_fps):
        if qfp is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(qfp, test_fps)
        if sims:
            out[i] = float(max(sims))
    return out


# --------------------------------------------------------------------------
# Step 1: current ChEMBL pull
# --------------------------------------------------------------------------
def fetch_current_chembl_rest(limit: int = 20000) -> pd.DataFrame:
    """Paginate the ChEMBL REST activity endpoint for CHEMBL3401."""
    import urllib.parse
    import urllib.request

    base = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    params_base = {
        "target_chembl_id": TARGET_ID,
        "limit": 1000,
    }
    rows = []
    offset = 0
    while offset < limit:
        params = dict(params_base, offset=offset)
        url = base + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "pxr-nb2073/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        acts = payload.get("activities", [])
        if not acts:
            break
        rows.extend(acts)
        page_meta = payload.get("page_meta", {})
        next_url = page_meta.get("next")
        if not next_url:
            break
        offset += len(acts)
    return pd.json_normalize(rows)


def fetch_current_chembl() -> pd.DataFrame:
    if CACHE_NEW_PULL.exists():
        try:
            df = pd.read_parquet(CACHE_NEW_PULL)
            print(f"[{TAG}] cached current pull: {len(df)} rows")
            return df
        except Exception:
            pass
    print(f"[{TAG}] pulling current ChEMBL via REST (target={TARGET_ID})")
    try:
        df = fetch_current_chembl_rest()
    except Exception as exc:
        print(f"[{TAG}] REST pull failed: {exc!r}")
        return pd.DataFrame()
    if not df.empty:
        try:
            df.to_parquet(CACHE_NEW_PULL, index=False)
        except Exception as exc:
            print(f"[{TAG}] cache write failed (non-fatal): {exc!r}")
    return df


# --------------------------------------------------------------------------
# Step 2-3: delta math
# --------------------------------------------------------------------------
def load_snapshot_inchikeys() -> set[str]:
    keys: set[str] = set()
    if SNAPSHOT_KB.exists():
        df = pd.read_parquet(SNAPSHOT_KB, columns=["inchikey", "source_target"])
        pxr_mask = df["source_target"].astype(str).str.contains(
            "PXR|NR1I2|3401", case=False, na=False
        )
        keys.update(df.loc[pxr_mask, "inchikey"].dropna().astype(str).tolist())
    if SNAPSHOT_RAW.exists():
        df = pd.read_parquet(SNAPSHOT_RAW, columns=["canonical_smiles"])
        for smi in df["canonical_smiles"].dropna().astype(str):
            k = std_inchikey(smi)
            if k:
                keys.add(k)
    return keys


def run() -> dict:
    if not TEST_CSV.exists():
        raise SystemExit(f"missing {TEST_CSV}")

    # Test 513 reference fingerprints + scaffolds
    test_df = pd.read_csv(TEST_CSV)
    print(f"[{TAG}] test rows: {len(test_df)}")
    test_fps_all = morgan_arr(test_df["SMILES"].tolist())
    test_fps = [fp for fp in test_fps_all if fp is not None]
    test_scaffolds = set(
        s for s in (std_scaffold(s) for s in test_df["SMILES"]) if s
    )

    # Snapshot (cycle 130) InChIKey set + scaffold set
    snap_keys = load_snapshot_inchikeys()
    snap_scaffolds: set[str] = set()
    if SNAPSHOT_RAW.exists():
        snap_smiles = pd.read_parquet(
            SNAPSHOT_RAW, columns=["canonical_smiles"]
        )["canonical_smiles"].dropna().astype(str)
        for smi in snap_smiles:
            s = std_scaffold(smi)
            if s:
                snap_scaffolds.add(s)
    print(f"[{TAG}] snapshot InChIKeys: {len(snap_keys)}; scaffolds: {len(snap_scaffolds)}")

    # Current pull
    cur = fetch_current_chembl()
    if cur.empty:
        summary = {
            "tag": TAG,
            "status": "REST_UNREACHABLE",
            "decision": "DEFER",
            "rationale": "ChEMBL REST pull returned no rows; cannot compute delta.",
            "snapshot_inchikeys": len(snap_keys),
        }
        OUT_SUMMARY.write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return summary

    # Normalize fields across MCP / REST shapes
    if "canonical_smiles" not in cur.columns and "molecule_structures.canonical_smiles" in cur.columns:
        cur["canonical_smiles"] = cur["molecule_structures.canonical_smiles"]
    smi_col = "canonical_smiles" if "canonical_smiles" in cur.columns else None
    if smi_col is None or "pchembl_value" not in cur.columns:
        summary = {
            "tag": TAG,
            "status": "SCHEMA_MISMATCH",
            "decision": "DEFER",
            "rationale": f"unexpected columns: {list(cur.columns)[:20]}",
        }
        OUT_SUMMARY.write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return summary

    cur = cur[cur[smi_col].notna()].copy()
    cur["pchembl_value"] = pd.to_numeric(cur["pchembl_value"], errors="coerce")
    cur["inchikey"] = cur[smi_col].astype(str).map(std_inchikey)
    cur = cur[cur["inchikey"].notna()]
    cur["scaffold"] = cur[smi_col].astype(str).map(std_scaffold)

    # Delta = current keys NOT in snapshot
    delta_mask = ~cur["inchikey"].isin(snap_keys)
    delta = cur.loc[delta_mask].copy()

    new_compounds = int(delta["inchikey"].nunique())
    delta_unique = delta.drop_duplicates("inchikey").copy()

    novel_scaffold_mask = ~delta_unique["scaffold"].isin(snap_scaffolds)
    novel_scaffold_unique = int(novel_scaffold_mask.sum())

    # New rows that ALSO carry a pchembl_value (the actionable subset)
    delta_with_pchembl = delta[delta["pchembl_value"].notna()].copy()
    new_pec50_records = int(len(delta_with_pchembl))
    novel_scaffold_with_pchembl = int(
        delta_with_pchembl.drop_duplicates("inchikey")["scaffold"]
        .apply(lambda s: s not in snap_scaffolds)
        .sum()
    )

    # Max-Tanimoto distribution for the new compounds vs test 513
    if new_compounds and test_fps:
        new_fps = morgan_arr(delta_unique[smi_col].tolist())
        max_sim = max_tanimoto_to_test(new_fps, test_fps)
        valid = max_sim[~np.isnan(max_sim)]
        if valid.size:
            sim_dist = {
                "n": int(valid.size),
                "min": float(np.min(valid)),
                "p25": float(np.percentile(valid, 25)),
                "median": float(np.median(valid)),
                "mean": float(np.mean(valid)),
                "p75": float(np.percentile(valid, 75)),
                "p90": float(np.percentile(valid, 90)),
                "max": float(np.max(valid)),
                "frac_ge_0.40": float(np.mean(valid >= 0.40)),
                "frac_ge_0.50": float(np.mean(valid >= 0.50)),
            }
        else:
            sim_dist = {"n": 0}
    else:
        sim_dist = {"n": 0}

    # Gate
    if novel_scaffold_with_pchembl >= GATE_MIN_NEW_PEC50_NOVEL_SCAFFOLD:
        decision = "QUEUE_NB2078"
        rationale = (
            f"{novel_scaffold_with_pchembl} novel-scaffold pEC50 records "
            f">= gate {GATE_MIN_NEW_PEC50_NOVEL_SCAFFOLD}; augmented retrain warranted."
        )
        try:
            QUEUE_MARKER.write_text(
                json.dumps(
                    {
                        "queued_by": TAG,
                        "queued_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "novel_scaffold_pec50_records": novel_scaffold_with_pchembl,
                        "new_compounds": new_compounds,
                        "next_script": "scripts/nb2078_chembl_augmented_retrain.py",
                    },
                    indent=2,
                )
            )
        except Exception as exc:
            print(f"[{TAG}] queue marker write failed: {exc!r}")
    else:
        decision = "AXIS_CLOSED"
        rationale = (
            f"only {novel_scaffold_with_pchembl} novel-scaffold pEC50 records "
            f"(gate {GATE_MIN_NEW_PEC50_NOVEL_SCAFFOLD}); ChEMBL PXR delta too thin "
            f"to move the LB. Re-probe next cycle."
        )

    summary = {
        "tag": TAG,
        "status": "OK",
        "target_chembl_id": TARGET_ID,
        "snapshot_inchikeys": len(snap_keys),
        "snapshot_scaffolds": len(snap_scaffolds),
        "current_rows": int(len(cur)),
        "current_unique_compounds": int(cur["inchikey"].nunique()),
        "new_compounds": new_compounds,
        "new_compounds_novel_scaffold": novel_scaffold_unique,
        "new_pec50_records_total": new_pec50_records,
        "new_pec50_records_novel_scaffold": novel_scaffold_with_pchembl,
        "max_tanimoto_to_test_513": sim_dist,
        "gate_threshold": GATE_MIN_NEW_PEC50_NOVEL_SCAFFOLD,
        "decision": decision,
        "rationale": rationale,
        "queue_marker": str(QUEUE_MARKER) if decision == "QUEUE_NB2078" else None,
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
