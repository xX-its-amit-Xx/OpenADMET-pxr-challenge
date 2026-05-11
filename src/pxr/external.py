"""Fetch bioactivity data from ChEMBL and PubChem BioAssay.

Primary sources:
  - ChEMBL REST API  (no API key required, rate-limit friendly)
  - PubChem BioAssay REST API  (for qHTS PXR assays)

Usage
-----
from pxr.external import fetch_chembl_target, NUCLEAR_RECEPTOR_TARGETS
df = fetch_chembl_target("CHEMBL3401")          # PXR
all_df = fetch_all_nr_targets()                 # all receptors
pubchem_df = fetch_pubchem_aid(720659)          # qHTS PXR activators
"""
from __future__ import annotations

import math
import time
from io import StringIO

import numpy as np
import pandas as pd
import requests

from .chem import standardize_smiles, to_inchikey

# ---------------------------------------------------------------------------
# Target registry — human nuclear receptors related to PXR
# ---------------------------------------------------------------------------

NUCLEAR_RECEPTOR_TARGETS: dict[str, str] = {
    "PXR":   "CHEMBL3401",      # NR1I2 — the challenge target
    "CAR":   "CHEMBL3509594",   # NR1I3 — closest structural relative
    "VDR":   "CHEMBL1977",      # NR1I1
    "FXR":   "CHEMBL2047",      # NR1H4
    "LXRa":  "CHEMBL2808",      # NR1H3
    "PPARg": "CHEMBL235",       # NR1C3 — large dataset
    "PPARa": "CHEMBL2111325",   # NR1C1
    "RXRa":  "CHEMBL2061",      # NR2B1 (heterodimerises with PXR)
}

# PubChem qHTS assay IDs for PXR activation
PUBCHEM_PXR_AIDS: list[int] = [720659, 1346985, 1346984]

_CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
_PUBCHEM_API = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pchembl_to_float(val) -> float | None:
    try:
        v = float(val)
        return v if 0 < v <= 12 else None
    except (TypeError, ValueError):
        return None


def _standard_to_pec50(value, units: str) -> float | None:
    """Convert IC50 / EC50 / Ki in assorted units to pEC50 (-log10 molar)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v <= 0:
        return None
    units = (units or "").lower().strip()
    if "nm" in units:
        molar = v * 1e-9
    elif "um" in units or "µm" in units:
        molar = v * 1e-6
    elif "mm" in units:
        molar = v * 1e-3
    elif units in ("m", "mol/l"):
        molar = v
    else:
        return None
    return -math.log10(molar)


def _get_json(url: str, retries: int = 3, backoff: float = 2.0) -> dict | None:
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            if r.status_code == 404:
                return None
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
        except Exception:
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    return None


# ---------------------------------------------------------------------------
# ChEMBL activity fetcher
# ---------------------------------------------------------------------------

def fetch_chembl_target(
    target_chembl_id: str,
    standard_types: tuple[str, ...] = ("IC50", "EC50", "Ki", "Kd", "AC50"),
    page_size: int = 1000,
    sleep_s: float = 0.3,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fetch all activity records with a pChEMBL value for one ChEMBL target.

    Returns a DataFrame with columns:
        smiles, pec50, standard_type, chembl_id
    One row per unique canonical SMILES (median pec50 when duplicated).
    """
    types_param = ",".join(standard_types)
    rows: list[dict] = []
    offset = 0

    while True:
        url = (
            f"{_CHEMBL_API}/activity.json"
            f"?target_chembl_id={target_chembl_id}"
            f"&standard_type__in={types_param}"
            f"&pchembl_value__isnull=false"
            f"&limit={page_size}&offset={offset}"
        )
        data = _get_json(url)
        if data is None:
            break

        activities = data.get("activities", [])
        if not activities:
            break

        for act in activities:
            smi = act.get("canonical_smiles")
            if not smi:
                continue
            p = _pchembl_to_float(act.get("pchembl_value"))
            if p is None:
                p = _standard_to_pec50(
                    act.get("standard_value"),
                    act.get("standard_units", ""),
                )
            if p is None:
                continue
            rows.append({
                "smiles":        smi,
                "pec50":         p,
                "standard_type": act.get("standard_type", ""),
                "chembl_id":     target_chembl_id,
            })

        total = data.get("page_meta", {}).get("total_count", 0)
        offset += page_size
        if offset >= total:
            break
        time.sleep(sleep_s)

    if not rows:
        if verbose:
            print(f"  {target_chembl_id}: 0 records found")
        return pd.DataFrame(columns=["smiles", "pec50", "standard_type", "chembl_id"])

    df = pd.DataFrame(rows)
    # Deduplicate: median pec50 per canonical SMILES
    df = (
        df.groupby("smiles", as_index=False)
        .agg(
            pec50=("pec50", "median"),
            standard_type=("standard_type", "first"),
            chembl_id=("chembl_id", "first"),
        )
    )
    if verbose:
        print(f"  {target_chembl_id}: {len(df):,} unique SMILES  "
              f"(pec50 {df['pec50'].min():.2f}–{df['pec50'].max():.2f})")
    return df


def fetch_all_nr_targets(
    targets: dict[str, str] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Fetch ChEMBL data for all nuclear receptor targets and merge.

    Returns DataFrame with columns:
        smiles, pec50, standard_type, chembl_id, target_name
    """
    if targets is None:
        targets = NUCLEAR_RECEPTOR_TARGETS

    dfs = []
    for name, chembl_id in targets.items():
        print(f"Fetching {name} ({chembl_id}) ...")
        df = fetch_chembl_target(chembl_id, **kwargs)
        if not df.empty:
            df["target_name"] = name
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# PubChem BioAssay fetcher (qHTS PXR assays)
# ---------------------------------------------------------------------------

def fetch_pubchem_aid(
    aid: int,
    outcome_filter: tuple[str, ...] = ("Active", "Inactive"),
    verbose: bool = True,
) -> pd.DataFrame:
    """Download compound data for one PubChem BioAssay AID.

    Fetches the full CSV, then resolves CIDs → isomeric SMILES.
    Returns DataFrame with columns:
        cid, smiles, activity_outcome, aid
    (Slow on large assays — use verbose=True to monitor progress.)
    """
    # Step 1: download assay CSV (SID-level)
    csv_url = f"{_PUBCHEM_API}/assay/aid/{aid}/CSV"
    try:
        r = requests.get(csv_url, timeout=120)
        r.raise_for_status()
        raw = pd.read_csv(StringIO(r.text), skiprows=5)
    except Exception as e:
        print(f"AID {aid}: CSV download failed — {e}")
        return pd.DataFrame()

    if verbose:
        print(f"AID {aid}: {len(raw):,} raw rows, columns: {raw.columns.tolist()[:6]}")

    # Normalise column names (PubChem varies across assays)
    raw.columns = [c.strip() for c in raw.columns]
    col_map = {}
    for c in raw.columns:
        lc = c.lower()
        if "cid" in lc and "sid" not in lc:
            col_map[c] = "cid"
        elif "outcome" in lc or "activity outcome" in lc:
            col_map[c] = "activity_outcome"
    raw = raw.rename(columns=col_map)

    if "cid" not in raw.columns:
        print(f"AID {aid}: could not find CID column in {raw.columns.tolist()}")
        return pd.DataFrame()

    raw = raw[raw["cid"].notna()].copy()
    raw["cid"] = raw["cid"].astype(int)

    if "activity_outcome" in raw.columns and outcome_filter:
        raw = raw[raw["activity_outcome"].isin(outcome_filter)]

    if verbose:
        print(f"AID {aid}: {len(raw):,} rows after outcome filter")

    # Step 2: fetch isomeric SMILES for unique CIDs in batches of 200
    unique_cids = raw["cid"].unique().tolist()
    cid_to_smi: dict[int, str] = {}
    batch = 200
    for i in range(0, len(unique_cids), batch):
        cid_str = ",".join(map(str, unique_cids[i : i + batch]))
        url = f"{_PUBCHEM_API}/compound/cid/{cid_str}/property/IsomericSMILES/JSON"
        data = _get_json(url)
        if data:
            for prop in data.get("PropertyTable", {}).get("Properties", []):
                cid_to_smi[prop["CID"]] = prop.get("IsomericSMILES", "")
        time.sleep(0.1)

    raw["smiles"] = raw["cid"].map(cid_to_smi)
    raw = raw[raw["smiles"].notna() & (raw["smiles"] != "")]
    raw["aid"] = aid
    if verbose:
        print(f"AID {aid}: {len(raw):,} rows with SMILES resolved")
    return raw[["cid", "smiles", "activity_outcome", "aid"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Post-processing: standardise and deduplicate against training data
# ---------------------------------------------------------------------------

def standardize_external(
    df: pd.DataFrame,
    train_inchikeys: set[str] | None = None,
    smiles_col: str = "smiles",
    verbose: bool = True,
) -> pd.DataFrame:
    """Standardise SMILES and remove compounds already in training data.

    Adds columns ``std_smiles`` and ``inchikey``.  Drops rows where
    standardisation fails or whose InChIKey matches the training set.
    """
    df = df.copy()
    df["std_smiles"] = df[smiles_col].map(standardize_smiles)
    df["inchikey"] = df["std_smiles"].map(lambda s: to_inchikey(s) if s else None)
    n_fail = df["std_smiles"].isna().sum()

    df = df.dropna(subset=["std_smiles", "inchikey"])
    n_after_std = len(df)

    if train_inchikeys:
        df = df[~df["inchikey"].isin(train_inchikeys)]
        n_overlap = n_after_std - len(df)
    else:
        n_overlap = 0

    if verbose:
        print(
            f"standardize_external: {len(df):,} kept  "
            f"({n_fail} std-fail, {n_overlap} train-overlap removed)"
        )
    return df.reset_index(drop=True)
