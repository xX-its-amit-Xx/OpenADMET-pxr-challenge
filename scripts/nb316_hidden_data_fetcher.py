"""nb316 - Hidden data source fetcher.

Pulls PXR / nuclear-receptor / CYP-induction-adjacent datasets that are NOT yet
integrated in data/external/. Each source is materialized as a parquet with
columns: smiles, std_smiles, inchikey, target_name, pec50, source, unit.

Sources (ranked by expected PXR-relevance):
  1. openadmet/Octant_CYP_inhibition_reactivity_blog_release  -- 1340 CYP3A4 pIC50
     (PXR's downstream target; correlated structural prior)
  2. openadmet/openadmet-expansionrx-challenge-data            -- 7610 ADMET
     (same provider, same standardization; supplies HLM/MLM/LogD priors)
  3. PubChem AID 1346985 (Tox21 hPXR activation qHTS)          -- ~10k AC50
  4. PubChem AID 1346984 (Tox21 CYP3A4 induction via PXR qHTS) -- ~10k AC50
  5. PubChem AID 720659  (NCATS hPXR activator qHTS)           -- ~3k AC50

Each fetcher is best-effort: failure of one source is logged but does not halt
the script. Outputs land under data/external/<source_name>/.
"""
from __future__ import annotations

import io
import sys
import time
import traceback
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# --- repo imports ------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.pxr.paths import DATA_EXTERNAL, DATA_RAW  # noqa: E402
from src.pxr.chem import standardize_smiles, to_inchikey  # noqa: E402

USER_AGENT = "Mozilla/5.0 (PXR-Challenge-Research nb316)"


def _http_get(url: str, timeout: int = 90) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _normalize(
    df: pd.DataFrame,
    smiles_col: str,
    value_col: str | None,
    target_name: str,
    source: str,
    unit: str,
    pec50_transform: str = "passthrough",
) -> pd.DataFrame:
    """Standardize SMILES, compute pec50 column, attach metadata.

    pec50_transform options:
      passthrough  -- value_col is already a pX (pIC50/pEC50). Just rename.
      log10_uM_inv -- value_col is potency in uM; pec50 = 6 - log10(uM)
      log10_M_inv  -- value_col is potency in M;  pec50 = -log10(M)
      ac50_uM_inv  -- PubChem-style AC50 in uM.  pec50 = 6 - log10(uM)
      none         -- no pec50 column (toxicity flag only)
    """
    out = pd.DataFrame()
    out["smiles"] = df[smiles_col].astype(str)
    out["std_smiles"] = out["smiles"].map(standardize_smiles)
    out["inchikey"] = out["std_smiles"].map(to_inchikey)
    out["target_name"] = target_name
    out["source"] = source
    out["unit"] = unit

    if value_col is None or pec50_transform == "none":
        out["pec50"] = np.nan
    else:
        v = pd.to_numeric(df[value_col], errors="coerce")
        if pec50_transform == "passthrough":
            out["pec50"] = v
        elif pec50_transform == "log10_uM_inv":
            out["pec50"] = 6.0 - np.log10(v.clip(lower=1e-6))
        elif pec50_transform == "log10_M_inv":
            out["pec50"] = -np.log10(v.clip(lower=1e-12))
        elif pec50_transform == "ac50_uM_inv":
            out["pec50"] = 6.0 - np.log10(v.clip(lower=1e-6))
        else:
            raise ValueError(f"unknown transform: {pec50_transform}")

    out = out.dropna(subset=["std_smiles"]).reset_index(drop=True)
    out = out.drop_duplicates(subset=["inchikey", "target_name"], keep="first")
    return out


def _save(df: pd.DataFrame, source_name: str, fname: str) -> Path:
    outdir = DATA_EXTERNAL / source_name
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / fname
    df.to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Source 1: OpenADMET Octant CYP inhibition (CYP3A4 pIC50)
# ---------------------------------------------------------------------------

def fetch_octant_cyp() -> dict:
    source_name = "openadmet_octant_cyp"
    out: dict = {"source": source_name, "status": "ok"}
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "openadmet/Octant_CYP_inhibition_reactivity_blog_release",
            "inhibition",
            split="train",
        )
        raw = ds.to_pandas()
        # Find SMILES + pIC50 columns flexibly
        smi_col = next(
            (c for c in raw.columns if "smiles" in c.lower()), None
        )
        ic_col = next(
            (c for c in raw.columns if "pic50" in c.lower() and "se" not in c.lower()
             and "ci_" not in c.lower()),
            None,
        )
        if smi_col is None or ic_col is None:
            out.update(status="schema_mismatch", columns=list(raw.columns))
            return out
        norm = _normalize(
            raw, smi_col, ic_col,
            target_name="CYP3A4",
            source=source_name,
            unit="pIC50",
            pec50_transform="passthrough",
        )
        path = _save(norm, source_name, "cyp3a4_inhibition.parquet")
        out.update(rows=len(norm), path=str(path), columns_used=[smi_col, ic_col])
    except Exception as e:
        out.update(status="error", error=repr(e), trace=traceback.format_exc())
    return out


# ---------------------------------------------------------------------------
# Source 2: OpenADMET ExpansionRx ADMET panel
# ---------------------------------------------------------------------------

def fetch_expansionrx() -> dict:
    source_name = "openadmet_expansionrx"
    out: dict = {"source": source_name, "status": "ok"}
    try:
        # The dataset has a fixed CSV inside it; prefer pandas to avoid hf
        # schema-cast issues (mixed-modifier rows).
        url = (
            "hf://datasets/openadmet/openadmet-expansionrx-challenge-data/"
            "expansion_data_train.csv"
        )
        try:
            raw = pd.read_csv(url)
        except Exception:
            # Fall back via huggingface_hub
            from huggingface_hub import hf_hub_download
            local = hf_hub_download(
                repo_id="openadmet/openadmet-expansionrx-challenge-data",
                filename="expansion_data_train.csv",
                repo_type="dataset",
            )
            raw = pd.read_csv(local)
        smi_col = next(c for c in raw.columns if c.upper() == "SMILES")
        # No PXR/CYP induction here; carry HLM as a clearance proxy (informative
        # for nuclear-receptor induction, often correlated).
        hlm_col = next(
            (c for c in raw.columns if "HLM" in c.upper()), None
        )
        norm = _normalize(
            raw, smi_col, hlm_col,
            target_name="HLM_CLint",
            source=source_name,
            unit="mL/min/kg",
            pec50_transform="none",  # not a pX; keep as raw clearance via value_col
        )
        # Carry raw clearance separately for downstream
        norm["raw_value"] = pd.to_numeric(raw[hlm_col], errors="coerce").values[
            : len(norm)
        ] if hlm_col else np.nan
        path = _save(norm, source_name, "admet_panel.parquet")
        out.update(rows=len(norm), path=str(path), columns_used=[smi_col, hlm_col])
    except Exception as e:
        out.update(status="error", error=repr(e), trace=traceback.format_exc())
    return out


# ---------------------------------------------------------------------------
# Helper: PubChem AID -> compounds with AC50
# ---------------------------------------------------------------------------

def _pubchem_aid_to_df(aid: int) -> pd.DataFrame:
    """Download an AID via PubChem PUG REST as concise CSV and join SMILES."""
    base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    # 1) Concise CSV: AID, SID, CID, outcome, activity value
    csv = _http_get(f"{base}/assay/aid/{aid}/concise/CSV", timeout=180)
    df = pd.read_csv(io.BytesIO(csv))
    # PubChem outcomes 2 = active. Keep only rows with a CID.
    if "CID" in df.columns:
        df = df.dropna(subset=["CID"])
        df["CID"] = df["CID"].astype(int)
    # 2) Bulk-fetch canonical SMILES in CID chunks
    cids = df["CID"].dropna().astype(int).unique().tolist() if "CID" in df.columns else []
    smiles_map: dict[int, str] = {}
    chunk = 200
    # PubChem now returns ConnectivitySMILES (post-2024 rename); fall back to
    # IsomericSMILES / CanonicalSMILES for older endpoints.
    for i in range(0, len(cids), chunk):
        sub = cids[i : i + chunk]
        cid_str = ",".join(map(str, sub))
        smap = None
        for prop in ("IsomericSMILES", "ConnectivitySMILES", "CanonicalSMILES"):
            try:
                data = _http_get(
                    f"{base}/compound/cid/{cid_str}/property/{prop}/CSV",
                    timeout=120,
                )
                cand = pd.read_csv(io.BytesIO(data))
                if cand.shape[1] >= 2:
                    smap = cand
                    smap.columns = ["CID", "smiles"] + list(cand.columns[2:])
                    break
            except Exception:
                continue
        if smap is None:
            continue
        for _, row in smap.iterrows():
            try:
                smiles_map[int(row["CID"])] = str(row["smiles"])
            except Exception:
                pass
        time.sleep(0.2)  # be polite to NCBI
    df["smiles"] = df["CID"].map(smiles_map)
    return df


def _fetch_pubchem_aid(aid: int, target_name: str, source_name: str) -> dict:
    out: dict = {"source": source_name, "status": "ok", "aid": aid}
    try:
        raw = _pubchem_aid_to_df(aid)
        # Activity-value column heuristic: prefer columns named like AC50 / EC50 / IC50
        value_col = None
        for c in raw.columns:
            cl = c.lower()
            if any(k in cl for k in ("ac50", "ec50", "ic50", "activity value")):
                value_col = c
                break
        norm = _normalize(
            raw,
            "smiles",
            value_col,
            target_name=target_name,
            source=source_name,
            unit="uM" if value_col else "binary",
            pec50_transform="ac50_uM_inv" if value_col else "none",
        )
        path = _save(norm, source_name, f"aid_{aid}.parquet")
        active_rows = int((raw.get("Activity Outcome", "") == "Active").sum()) if "Activity Outcome" in raw.columns else None
        out.update(
            rows=len(norm),
            path=str(path),
            value_col=value_col,
            active=active_rows,
        )
    except Exception as e:
        out.update(status="error", error=repr(e), trace=traceback.format_exc())
    return out


def fetch_pubchem_pxr_activation_1346985() -> dict:
    return _fetch_pubchem_aid(
        1346985, "PXR_activation", "pubchem_aid_1346985_tox21_pxr"
    )


def fetch_pubchem_cyp3a4_induction_1346984() -> dict:
    return _fetch_pubchem_aid(
        1346984, "CYP3A4_induction_via_PXR", "pubchem_aid_1346984_tox21_cyp3a4"
    )


def fetch_pubchem_pxr_ncats_720659() -> dict:
    return _fetch_pubchem_aid(
        720659, "PXR_activation", "pubchem_aid_720659_ncats_pxr"
    )


# ---------------------------------------------------------------------------
# Overlap estimator (train+test InChIKey overlap with the fetched source)
# ---------------------------------------------------------------------------

def _local_inchikeys() -> set[str]:
    try:
        tr = pd.read_csv(DATA_RAW / "pxr-challenge_TRAIN.csv")
        te = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_BLINDED.csv")
        smi_col_tr = next(
            c for c in tr.columns if "smiles" in c.lower()
        )
        smi_col_te = next(
            c for c in te.columns if "smiles" in c.lower()
        )
        keys: set[str] = set()
        for s in pd.concat([tr[smi_col_tr], te[smi_col_te]], ignore_index=True):
            k = to_inchikey(s)
            if k:
                keys.add(k)
        return keys
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

FETCHERS = [
    ("octant_cyp", fetch_octant_cyp),
    ("expansionrx", fetch_expansionrx),
    ("pubchem_pxr_1346985", fetch_pubchem_pxr_activation_1346985),
    ("pubchem_cyp3a4_1346984", fetch_pubchem_cyp3a4_induction_1346984),
    ("pubchem_pxr_720659", fetch_pubchem_pxr_ncats_720659),
]


def main() -> None:
    print(f"[nb316] DATA_EXTERNAL = {DATA_EXTERNAL}")
    local_keys = _local_inchikeys()
    print(f"[nb316] local train+test inchikeys: {len(local_keys)}")

    reports: list[dict] = []
    for label, fn in FETCHERS:
        print(f"\n[nb316] >>> {label}")
        rep = fn()
        # Overlap (best-effort)
        if rep.get("status") == "ok" and "path" in rep:
            try:
                df = pd.read_parquet(rep["path"])
                src_keys = set(df["inchikey"].dropna().unique().tolist())
                rep["overlap_with_train_test"] = len(src_keys & local_keys)
                rep["unique_compounds"] = len(src_keys)
            except Exception:
                pass
        reports.append(rep)
        print(f"    -> {rep}")

    summary = pd.DataFrame(reports)
    summary_path = DATA_EXTERNAL / "nb316_fetch_summary.parquet"
    summary.to_parquet(summary_path, index=False)
    csv_path = DATA_EXTERNAL / "nb316_fetch_summary.csv"
    # cols vary; keep just the headline ones for csv
    cols = [
        c for c in [
            "source", "status", "rows", "unique_compounds",
            "overlap_with_train_test", "path", "value_col", "active",
            "error",
        ]
        if c in summary.columns
    ]
    summary[cols].to_csv(csv_path, index=False)
    print(f"\n[nb316] summary -> {csv_path}")
    print(summary[cols].to_string())


if __name__ == "__main__":
    main()
