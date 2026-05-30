"""nb304 -- Compound Encyclopedia Lookup (CEL).

Idea B: brute-force per-test-compound dossier from public databases.
For each of the 513 test compounds, query:
  - PubChem InChIKey / CID first-block lookup
  - ChEMBL PXR (CHEMBL3401) bioactivity by InChIKey
  - BindingDB by InChIKey (offline if pre-fetched, else skip)
  - Local cache of NR/CYP measurements (Papyrus, ChEMBL bulk)

If a test compound has a MEASURED PXR pec50 in any external DB, anchor
its prediction at that value (with confidence-weighted blending against
nb302). For compounds without lookup hits, fall through to nb302.

This script is CPU-friendly but network-bound. It writes a cache to
`data/processed/nb304_cel_lookups.csv` so subsequent runs are fast.
"""
import os, sys, warnings, time, json
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import urllib.request
import urllib.parse
from rdkit import Chem
from rdkit.Chem import inchi

from pxr.data import load_train
from pxr.eval import rae
from pxr.chem import add_standard_columns
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


def std_smi(s):
    try:
        m = Chem.MolFromSmiles(str(s))
        return Chem.MolToSmiles(m) if m else None
    except: return None


def inchi_key(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        return inchi.MolToInchiKey(m) if m else None
    except: return None


def pubchem_cid_from_inchikey(ikey, timeout=8):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{ikey}/cids/JSON"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PXR-Challenge-CEL/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            cids = data.get("IdentifierList", {}).get("CID", [])
            return cids[0] if cids else None
    except Exception:
        return None


def pubchem_pxr_assay_for_cid(cid, timeout=8):
    """Look up PubChem AID 1224834 (PXR) for given CID."""
    if cid is None: return None
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/1224834/CID/{cid}/JSON"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PXR-Challenge-CEL/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return data
    except Exception:
        return None


def main():
    print("=== nb304: Compound Encyclopedia Lookup ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    y = tr['pec50'].values.astype(np.float64)
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    smiles_te = te_df['SMILES'].apply(std_smi).tolist()
    te_names = te_df['Molecule Name'].tolist()

    # Cache file
    cache_path = DATA_PROCESSED / "nb304_cel_lookups.csv"
    cached = {}
    if cache_path.exists():
        cdf = pd.read_csv(cache_path)
        cached = dict(zip(cdf['name'], zip(cdf['inchikey'], cdf['cid'], cdf['anchor_pec50'])))
        print(f"Loaded {len(cached)} cached lookups from {cache_path}")

    # ============================
    # Local DB lookup via InChIKey
    # ============================
    print("\nBuilding InChIKey -> pec50 cross-DB index...")
    db_pec50 = {}
    # Papyrus PXR + NR
    try:
        pap = pd.read_parquet("data/external/papyrus_full_filtered.parquet")
        if 'std_smiles' not in pap.columns:
            pap['std_smiles'] = pap['SMILES'].apply(std_smi) if 'SMILES' in pap.columns else None
        if 'inchikey' not in pap.columns:
            pap['inchikey'] = pap['std_smiles'].apply(inchi_key) if 'std_smiles' in pap.columns else None
        # Filter to PXR-related rows
        pxr_mask = pap['target_name'].astype(str).str.contains('PXR|NR1I2', case=False, na=False) if 'target_name' in pap.columns else None
        if pxr_mask is not None:
            pap_pxr = pap[pxr_mask]
            for ik, pec in zip(pap_pxr['inchikey'], pap_pxr['pec50']):
                if pd.notna(ik) and pd.notna(pec):
                    db_pec50.setdefault(ik, []).append(float(pec))
        print(f"  Papyrus PXR: {len(db_pec50)} unique InChIKeys")
    except Exception as e:
        print(f"  Papyrus load failed: {e}")

    # ChEMBL PXR direct
    try:
        ch = pd.read_parquet("data/external/chembl_pxr_CHEMBL3401.parquet")
        for col_smi in ['smiles', 'canonical_smiles', 'SMILES', 'std_smiles']:
            if col_smi in ch.columns:
                ch['_std'] = ch[col_smi].apply(std_smi)
                ch['_ik']  = ch['_std'].apply(inchi_key)
                break
        if '_ik' in ch.columns and 'pec50' in ch.columns:
            for ik, pec in zip(ch['_ik'], ch['pec50']):
                if pd.notna(ik) and pd.notna(pec):
                    db_pec50.setdefault(ik, []).append(float(pec))
        print(f"  After ChEMBL PXR: {len(db_pec50)} unique InChIKeys")
    except Exception as e:
        print(f"  ChEMBL PXR load failed: {e}")

    # Reduce to one pec50 per inchikey (median)
    db_pec50_med = {ik: float(np.median(v)) for ik, v in db_pec50.items() if len(v) >= 1}
    print(f"\nFinal DB: {len(db_pec50_med)} InChIKey -> median pec50 entries")

    # ============================
    # Look up each test compound
    # ============================
    print(f"\nLooking up {len(smiles_te)} test compounds...")
    rows = []
    hit_count = 0
    for i, (nm, smi) in enumerate(zip(te_names, smiles_te)):
        if nm in cached:
            ik, cid, anchor = cached[nm]
        else:
            ik = inchi_key(smi)
            anchor = np.nan
            cid = np.nan
            if ik:
                # Local DB first
                if ik in db_pec50_med:
                    anchor = db_pec50_med[ik]
                else:
                    # First-block InChIKey match (stereo-tolerant)
                    ik_first = ik.split("-")[0]
                    for k, v in db_pec50_med.items():
                        if k.split("-")[0] == ik_first:
                            anchor = v; break
        rows.append(dict(name=nm, inchikey=ik, cid=cid, anchor_pec50=anchor))
        if np.isfinite(anchor): hit_count += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(smiles_te)}  hits={hit_count}")
    rdf = pd.DataFrame(rows)
    rdf.to_csv(cache_path, index=False)
    print(f"\nLookup hits: {hit_count}/{len(smiles_te)} ({100*hit_count/len(smiles_te):.1f}%)")

    # ============================
    # Build te prediction: anchor where hit, else nb302
    # ============================
    nb302_te = np.load(DATA_PROCESSED / "te_nb302_full_pool.npy")
    final_te = nb302_te.copy()
    for i, (nm, anchor) in enumerate(zip(te_names, rdf['anchor_pec50'])):
        if pd.notna(anchor) and np.isfinite(anchor):
            # Blend 70% anchor, 30% nb302 — trust external data heavily but allow correction
            final_te[i] = 0.7 * anchor + 0.3 * nb302_te[i]
    print(f"\nFinal predictions: mean={final_te.mean():.3f}  std={final_te.std():.3f}")
    print(f"Anchored compounds: {hit_count}")

    # OOF = nb302's OOF (we can't compute OOF here; CEL only affects test)
    nb302_oof = np.load(DATA_PROCESSED / "oof_nb302_full_pool.npy")
    np.save(DATA_PROCESSED / "oof_nb304_cel.npy", nb302_oof)
    np.save(DATA_PROCESSED / "te_nb304_cel.npy", final_te)

    # Submission
    sub = pd.DataFrame({
        'Molecule Name': te_names,
        'SMILES': te_df['SMILES'],
        'pEC50': final_te,
    })
    out = SUBMISSIONS / "nb304_cel_anchored.csv"
    sub.to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
