"""nb1336 — pull on-target FXR (NR1H4) potency for cross-NR read-across.

Primary: ChEMBL CHEMBL2047 (human FXR, Bile acid receptor) EC50/AC50/Potency.
Augment: Tox21 FXR-bla agonist actives (PubChem AID 743239).
Normalize all to p-scale (pX = 9 - log10(value_nM)); dedup by InChIKey.
Output rows compatible with nb1332-style gate:  (id, std_smiles, inchikey, outcome, p_potency)
"""
import os, sys, time, pickle
import numpy as np, pandas as pd, requests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.chem import standardize_smiles as standardize
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi
RDLogger.DisableLog("rdApp.*")

OUT = "C:/pxr_work/fxr_tox21"; os.makedirs(OUT, exist_ok=True)
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
B = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

def to_p_nM(v):
    return 9.0 - np.log10(v) if v and v > 0 else None

# ── ChEMBL CHEMBL2047 continuous potency
def pull_chembl(stype):
    f = f"{OUT}/chembl_{stype}.parquet"
    if os.path.exists(f):
        return pd.read_parquet(f)
    recs, off = [], 0
    while True:
        url = (f"{CHEMBL}/activity.json?target_chembl_id=CHEMBL2047&standard_type={stype}"
               f"&limit=1000&offset={off}")
        j = requests.get(url, timeout=180).json()
        for a in j["activities"]:
            recs.append({
                "smiles": a.get("canonical_smiles"),
                "value": a.get("standard_value"),
                "units": a.get("standard_units"),
                "rel": a.get("standard_relation"),
                "type": stype,
            })
        nxt = j["page_meta"]["next"]
        if not nxt: break
        off += 1000
        if off > 8000: break
    df = pd.DataFrame(recs); df.to_parquet(f); return df

print("pulling ChEMBL CHEMBL2047 ...", flush=True)
ch = pd.concat([pull_chembl(s) for s in ["EC50", "AC50", "Potency"]], ignore_index=True)
ch = ch[ch["units"] == "nM"].dropna(subset=["smiles", "value"])
ch = ch[ch["rel"].isin(["=", "<", "<=", None])]
ch["p"] = ch["value"].astype(float).map(to_p_nM)
ch = ch.dropna(subset=["p"])
print(f"ChEMBL nM rows: {len(ch)}", flush=True)

# ── Tox21 actives (binary) -> append at active floor pAC50 if SMILES resolvable
def fetch_concise(aid):
    f = f"{OUT}/aid_{aid}_concise.parquet"
    if os.path.exists(f): return pd.read_parquet(f)
    j = requests.get(f"{B}/assay/aid/{aid}/concise/JSON", timeout=180).json()["Table"]
    df = pd.DataFrame([row["Cell"] for row in j["Row"]], columns=j["Columns"]["Column"])
    df.to_parquet(f); return df

def fetch_smiles(cids):
    out = {}
    for i in range(0, len(cids), 180):
        chunk = cids[i:i+180]
        for prop in ["SMILES,IsomericSMILES", "CanonicalSMILES"]:
            try:
                r = requests.get(f"{B}/compound/cid/{','.join(map(str,chunk))}/property/{prop}/JSON", timeout=180)
                if r.status_code == 200:
                    for p in r.json()["PropertyTable"]["Properties"]:
                        s = (p.get("IsomericSMILES") or p.get("SMILES")
                             or p.get("ConnectivitySMILES") or p.get("CanonicalSMILES"))
                        if s: out[p["CID"]] = s
                    break
            except Exception:
                continue
        time.sleep(0.15)
    return out

tox_smis = []
try:
    tdf = fetch_concise(743239)
    tdf = tdf[tdf.get("Activity Outcome") == "Active"]
    tcids = [int(x) for x in tdf["CID"].dropna().tolist() if str(x).strip().isdigit()]
    c2s = fetch_smiles(tcids)
    tox_smis = list(c2s.values())
    print(f"Tox21 FXR actives w/ SMILES: {len(tox_smis)}", flush=True)
except Exception as e:
    print("Tox21 pull skipped:", e, flush=True)

# ── standardize + dedup by InChIKey (keep max potency per compound)
best = {}   # ik -> (smiles, p, outcome)
def add(smi, p, outcome):
    std = standardize(smi)
    if not std: return
    m = Chem.MolFromSmiles(std)
    if m is None: return
    try: ik = inchi.InchiToInchiKey(inchi.MolToInchi(m))
    except Exception: return
    if (ik not in best) or (p is not None and best[ik][1] is not None and p > best[ik][1]):
        best[ik] = (std, p, outcome)

for _, r in ch.iterrows():
    add(r["smiles"], float(r["p"]), "Active")
# Tox21 actives: assign median active potency floor if novel
floor = float(np.nanmedian([v[1] for v in best.values()])) if best else 6.0
for s in tox_smis:
    add(s, floor, "Active")

rows = [(i, v[0], ik, v[2], v[1]) for i, (ik, v) in enumerate(best.items())]
print(f"final unique compounds: {len(rows)}  (p range "
      f"{min(r[4] for r in rows):.2f}..{max(r[4] for r in rows):.2f})", flush=True)
pickle.dump(rows, open(f"{OUT}/fxr_std.pkl", "wb"))
print("saved", f"{OUT}/fxr_std.pkl", flush=True)
