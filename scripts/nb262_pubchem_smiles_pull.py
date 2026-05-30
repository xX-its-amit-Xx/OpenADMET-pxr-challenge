"""nb262 -- Pull SMILES for 5740 active PXR CIDs via PubChem batch API.

For each chunk of 200 CIDs, query PubChem PUG REST for CanonicalSMILES.
Build a 'PXR-active reference library' and use as Tanimoto-NN feature.
"""
import os, sys, warnings, time, json
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path("data/external")


def fetch_smiles_batch(cids):
    cid_str = ",".join(str(int(c)) for c in cids)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_str}/property/ConnectivitySMILES/JSON"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
        props = data.get("PropertyTable", {}).get("Properties", [])
        return {p["CID"]: p.get("ConnectivitySMILES") or p.get("CanonicalSMILES") or p.get("SMILES") for p in props if (p.get("ConnectivitySMILES") or p.get("CanonicalSMILES") or p.get("SMILES"))}
    except Exception as e:
        print(f"    batch err: {e}")
        return {}


def main():
    print("=== nb262: PubChem CID -> SMILES batch ===\n")
    summary = pd.read_parquet("data/processed/pubchem_pxr_cid_summary.parquet")
    active = summary[summary["n_active"] > 0].copy()
    print(f"Active CIDs to pull: {len(active)}")

    # CID column may be string; strip empties; convert numeric
    active = active[active["CID"].notna()].copy()
    active["CID_int"] = pd.to_numeric(active["CID"], errors="coerce")
    active = active.dropna(subset=["CID_int"]).copy()
    active["CID_int"] = active["CID_int"].astype(int)
    cids = active["CID_int"].tolist()

    smiles_map = {}
    batch_size = 100
    t0 = time.time()
    for i in range(0, len(cids), batch_size):
        batch = cids[i:i+batch_size]
        result = fetch_smiles_batch(batch)
        smiles_map.update(result)
        if (i + batch_size) % 500 == 0 or (i + batch_size) >= len(cids):
            elapsed = time.time() - t0
            print(f"  {i+batch_size}/{len(cids)}  got {len(smiles_map)} SMILES  elapsed={elapsed:.0f}s")
        time.sleep(0.3)  # be nice to PubChem

    # Save mapping without merge complications
    rows = []
    for c, s in smiles_map.items():
        try:
            c_int = int(c)
        except: continue
        rows.append({"CID": c_int, "smiles": str(s)})
    df = pd.DataFrame(rows)
    # Lookup active info via dict
    cid_to_rate = dict(zip(active["CID_int"].tolist(), active["active_rate"].tolist()))
    cid_to_nact = dict(zip(active["CID_int"].tolist(), active["n_active"].tolist()))
    df["active_rate"] = df["CID"].map(cid_to_rate)
    df["n_active"] = df["CID"].map(cid_to_nact)
    df.to_parquet(OUT / "pubchem_pxr_active_smiles.parquet", index=False)
    print(f"\nSaved pubchem_pxr_active_smiles.parquet ({len(df)} compounds)")
    print(f"Sample SMILES: {df['smiles'].iloc[:3].tolist()}")
    print(f"active_rate stats: mean={df.active_rate.mean():.3f}, fully active (>0.5): {(df.active_rate > 0.5).sum()}")


if __name__ == "__main__":
    main()
