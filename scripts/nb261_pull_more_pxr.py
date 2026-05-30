"""nb261 -- Pull more PXR data from PubChem BioAssay (multiple AIDs).

PXR-related BioAssays:
- AID 1224870: TR-FRET hPXR full agonist
- AID 1224872: hPXR antagonist
- AID 504447: PXR-CYP3A4 cell-based
- AID 651608: PXR-mediated CYP3A4 induction
- AID 1346982: PXR antagonist for liver toxicity
- AID 1259244, 1259245, 1259246: hPXR agonist confirmation
"""
import os, sys, warnings, json, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import pandas as pd
import urllib.request
from pathlib import Path

OUT = Path("data/external")
OUT.mkdir(exist_ok=True, parents=True)


def pull_aid(aid):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/concise/JSON"
    dest = OUT / f"pubchem_aid_{aid}_concise.json"
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"  Cached AID {aid}")
    else:
        try:
            print(f"  Downloading AID {aid}...")
            urllib.request.urlretrieve(url, dest)
            time.sleep(0.5)
        except Exception as e:
            print(f"    FAIL: {e}")
            return None
    try:
        with open(dest) as f:
            data = json.load(f)
        if "Table" not in data:
            print(f"    AID {aid}: no Table key, skipping")
            return None
        columns = data["Table"].get("Columns", {}).get("Column", [])
        rows = data["Table"].get("Row", [])
        df_rows = [r.get("Cell", []) for r in rows]
        df = pd.DataFrame(df_rows, columns=columns)
        df["source_aid"] = aid
        return df
    except Exception as e:
        print(f"    Parse fail AID {aid}: {e}")
        return None


def main():
    print("=== nb261: Pull multiple PXR PubChem AIDs ===\n")
    aids = [1224870, 1224872, 504447, 651608, 1346982, 1259244, 1259245, 1259246, 1645842]
    dfs = []
    for aid in aids:
        df = pull_aid(aid)
        if df is not None:
            print(f"  AID {aid}: {len(df)} rows, cols: {df.columns.tolist()[:5]}")
            dfs.append(df)
    if not dfs:
        print("\nNo data!")
        return

    # Combine — different AIDs may have different columns; only keep common
    common_cols = set(dfs[0].columns)
    for d in dfs[1:]:
        common_cols &= set(d.columns)
    print(f"\nCommon columns across AIDs: {common_cols}")

    all_df = pd.concat([d[list(common_cols)] for d in dfs], ignore_index=True)
    print(f"Total PXR-AID records: {len(all_df)}")

    if "Activity Outcome" in all_df.columns:
        print(all_df["Activity Outcome"].value_counts())
    all_df.to_parquet(OUT / "pubchem_pxr_multi_aid.parquet", index=False)
    print(f"\nSaved pubchem_pxr_multi_aid.parquet ({len(all_df)} rows)")


if __name__ == "__main__":
    main()
