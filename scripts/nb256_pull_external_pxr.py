"""nb256 -- Pull external PXR datasets from public sources.

Sources:
1. Tox21: nuclear receptor screening, PXR included. From Tripod NIH.
2. PubChem BioAssay PXR AIDs via PUG REST.
3. BindingDB PXR (UniProt O75469).

Save as parquet for downstream feature engineering.
"""
import os, sys, warnings, json, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import urllib.request
import urllib.error
from pathlib import Path

OUT_DIR = Path("data/external")
OUT_DIR.mkdir(exist_ok=True, parents=True)


def download_safe(url, dest):
    try:
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"  Cached: {dest.name}")
            return True
        print(f"  Downloading {url[:80]}...")
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def pull_tox21():
    """Tox21 SDF + assay data. PXR is in NR panel."""
    print("\n=== Tox21 ===")
    # The Tox21 data is at the EPA challenge site. URLs may change.
    # Use the SDF + activity labels
    urls = {
        "tox21_10k_data_all_train": "https://tripod.nih.gov/tox21/challenge/download?id=tox21_10k_data_allsdf",
        "tox21_10k_challenge_test": "https://tripod.nih.gov/tox21/challenge/download?id=tox21_10k_challenge_testsdf",
    }
    # These URL patterns may have changed; try alternative sources
    # Alternative: chembl_downloader has Tox21 assay
    # Most reliable: bioassay download via NCBI E-utils for specific PXR AIDs

    # Use PubChem BioAssay PXR direct
    return pull_pubchem_pxr()


def pull_pubchem_pxr():
    """PubChem BioAssay PXR AID 1645842 = hPXR agonist."""
    print("\n=== PubChem BioAssay PXR (AID 1645842, hPXR agonist) ===")
    aid = "1645842"
    # JSON download
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/concise/JSON"
    dest = OUT_DIR / f"pubchem_aid_{aid}_concise.json"
    if not download_safe(url, dest):
        return None
    try:
        with open(dest) as f:
            data = json.load(f)
        # Parse
        if "Table" in data:
            columns = data["Table"].get("Columns", {}).get("Column", [])
            rows = data["Table"].get("Row", [])
            print(f"  Columns: {columns[:5]}...")
            print(f"  Rows: {len(rows)}")
            # Build DataFrame
            df_rows = []
            for r in rows:
                cells = r.get("Cell", [])
                df_rows.append(cells)
            df = pd.DataFrame(df_rows, columns=columns)
            print(f"  Saved to {dest.with_suffix('.parquet')}")
            df.to_parquet(dest.with_suffix(".parquet"), index=False)
            return df
    except Exception as e:
        print(f"  Parse error: {e}")
    return None


def pull_bindingdb_pxr():
    """BindingDB query for PXR (UniProt O75469)."""
    print("\n=== BindingDB PXR ===")
    # BindingDB has a web query interface; direct CSV download requires API key or web scraping
    # Use the BindingDB SDF subset or compound search
    # Their REST API: https://bindingdb.org/rest/getLigandsByUniprots?uniprot=O75469
    url = "https://www.bindingdb.org/axis2/services/BDBService/getLigandsByUniprots?uniprot=O75469&cutoff=10&code=0"
    dest = OUT_DIR / "bindingdb_pxr_O75469.xml"
    if download_safe(url, dest):
        print(f"  Saved {dest}")
    return None


def pull_chembl_pxr_target():
    """ChEMBL bioactivity records for CHEMBL3401 (NR1I2/PXR target)."""
    print("\n=== ChEMBL PXR (CHEMBL3401) ===")
    # ChEMBL REST API
    url = "https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id=CHEMBL3401&limit=1000"
    rows = []
    offset = 0
    for _ in range(50):  # up to 50000
        u = url + f"&offset={offset}"
        dest = OUT_DIR / f"chembl_pxr_offset_{offset}.json"
        if not download_safe(u, dest):
            break
        try:
            with open(dest) as f:
                d = json.load(f)
            acts = d.get("activities", [])
            if not acts: break
            rows.extend(acts)
            offset += 1000
            if len(rows) >= 5000: break  # cap
        except Exception as e:
            print(f"  parse err: {e}")
            break
        time.sleep(0.5)  # be nice
    print(f"  Total ChEMBL records: {len(rows)}")
    if rows:
        df = pd.DataFrame(rows)
        df.to_parquet(OUT_DIR / "chembl_pxr_CHEMBL3401.parquet", index=False)
        print(f"  Saved chembl_pxr_CHEMBL3401.parquet ({len(df)} rows)")
        print(f"  Columns: {df.columns.tolist()[:10]}")
        return df
    return None


def main():
    print("=== nb256: External PXR data pull ===\n")
    # 1. PubChem BioAssay
    pub = pull_pubchem_pxr()
    if pub is not None:
        print(f"\nPubChem rows: {len(pub)}")
        print(pub.head(3))

    # 2. ChEMBL direct PXR target
    ch = pull_chembl_pxr_target()
    if ch is not None:
        print(f"\nChEMBL PXR records: {len(ch)}")

    # 3. BindingDB
    pull_bindingdb_pxr()

    print("\nDone.")


if __name__ == "__main__":
    main()
