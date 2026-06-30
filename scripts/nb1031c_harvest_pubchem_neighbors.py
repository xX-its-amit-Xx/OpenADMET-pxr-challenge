"""nb1031c — harvest PubChem similarity NEIGHBORS of the 513 test compounds that are PXR-TESTED.

For each test compound: PubChem fastsimilarity_2d (CACTVS) -> neighbor CIDs -> intersect the 52k PXR-tested CID
universe (pubchem_pxr_cid_summary: CID -> n_active/n_inactive/active_rate). Pull neighbor SMILES (bulk) so we can
compute the REAL Morgan similarity for weighting. Saves raw matches; nb1032 builds the prior + validates.

Checkpointed: writes test_pxr_neighbor_matches.csv incrementally so a crash/rate-limit doesn't lose progress.
"""
import os, sys, json, time, urllib.request, urllib.parse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test

D = "data/processed"
THRESH = 80; MAXREC = 500


def get(url, tries=3):
    for _ in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=45) as r:
                return r.read().decode()
        except Exception as e:
            if "404" in str(e) or "PUGREST.NotFound" in str(e):
                return "404"
            time.sleep(1.2)
    return "ERR"


def main():
    summ = pd.read_parquet(f"{D}/pubchem_pxr_cid_summary.parquet")
    summ["CID"] = pd.to_numeric(summ["CID"], errors="coerce")
    summ = summ.dropna(subset=["CID"]); summ["CID"] = summ["CID"].astype(int)
    pxr_cid = dict(zip(summ["CID"], zip(summ["active_rate"], summ["n_active"], summ["n_total"])))
    pxr_set = set(pxr_cid)
    print(f"PXR-tested universe: {len(pxr_set)} CIDs")

    te = load_test().reset_index(drop=True)
    matches = []  # test_pos, neighbor_cid, active_rate, n_active, n_total
    t0 = time.time()
    for i in range(len(te)):
        enc = urllib.parse.quote(str(te["smiles"].iloc[i]))
        s = get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsimilarity_2d/smiles/{enc}/cids/JSON?Threshold={THRESH}&MaxRecords={MAXREC}")
        if s not in ("404", "ERR"):
            try:
                cids = json.loads(s)["IdentifierList"]["CID"]
            except Exception:
                cids = []
            for c in set(cids) & pxr_set:
                ar, na, nt = pxr_cid[c]
                matches.append({"test_pos": i, "cid": c, "active_rate": ar, "n_active": na, "n_total": nt})
        if (i + 1) % 50 == 0:
            pd.DataFrame(matches).to_csv(f"{D}/test_pxr_neighbor_matches.csv", index=False)
            print(f"  {i+1}/513  matches so far {len(matches)}  ({time.time()-t0:.0f}s)")
        time.sleep(0.33)
    mdf = pd.DataFrame(matches)
    mdf.to_csv(f"{D}/test_pxr_neighbor_matches.csv", index=False)
    cov = mdf["test_pos"].nunique() if len(mdf) else 0
    print(f"\nharvest done: {len(mdf)} matches across {cov}/513 test compounds")

    # bulk-fetch SMILES for the matched neighbor CIDs (for real Morgan-sim weighting in nb1032)
    ucids = sorted(mdf["cid"].unique().tolist()) if len(mdf) else []
    print(f"fetching SMILES for {len(ucids)} unique neighbor CIDs...")
    smi = {}
    for b in range(0, len(ucids), 100):
        chunk = ucids[b:b + 100]
        ids = ",".join(map(str, chunk))
        s = get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{ids}/property/CanonicalSMILES/JSON")
        if s not in ("404", "ERR"):
            try:
                for p in json.loads(s)["PropertyTable"]["Properties"]:
                    if "CanonicalSMILES" in p:
                        smi[int(p["CID"])] = p["CanonicalSMILES"]
            except Exception:
                pass
        time.sleep(0.33)
    pd.DataFrame([{"cid": c, "smiles": smi.get(c)} for c in ucids]).to_csv(f"{D}/test_pxr_neighbor_smiles.csv", index=False)
    print(f"saved {len(smi)} neighbor SMILES -> test_pxr_neighbor_smiles.csv")
    json.dump({"matches": len(mdf), "covered_test": int(cov), "unique_neighbor_cids": len(ucids),
               "smiles_fetched": len(smi)}, open(f"{D}/nb1031c_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
