"""nb360 -- Download PubChem AID 1346985 (Tox21 hPXR qHTS activation).

Steps:
1. Download bioassay concise CSV from PubChem PUG-REST.
2. Resolve CID -> CanonicalSMILES (+ InChIKey) in batches.
3. Standardize SMILES, compute Morgan/RDKit features per compound.
4. Save data/processed/hidden_compound_features.parquet
   columns: [inchikey, smiles, cid, aid_active, aid_pec50, feat_*]
"""
import os, sys, json, time, urllib.request, warnings
from pathlib import Path
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "external" / "multimodal_hidden"
RAW.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "data" / "processed" / "hidden_compound_features.parquet"

AID = 1346985


def http_get(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def download_assay_csv():
    """Concise CSV listing per-CID outcome rows. Returns either Path or BytesIO."""
    from io import BytesIO
    fp = RAW / f"aid_{AID}_concise.csv"
    if fp.exists() and fp.stat().st_size > 1000:
        print(f"  reuse {fp.name} ({fp.stat().st_size:,} bytes)")
        return fp
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{AID}/concise/CSV"
    print(f"  GET {url}")
    data = http_get(url, timeout=180)
    # try disk first, fall back to memory if disk full
    try:
        fp.write_bytes(data)
        print(f"  wrote {fp.name} ({len(data):,} bytes)")
        return fp
    except OSError as e:
        print(f"  disk write failed ({e}); using in-memory ({len(data):,} bytes)")
        return BytesIO(data)


def fetch_props_batch(cids):
    """CID -> {CID, ConnectivitySMILES, InChIKey}"""
    cid_str = ",".join(str(int(c)) for c in cids)
    url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_str}/"
           f"property/ConnectivitySMILES,InChIKey/JSON")
    try:
        data = json.loads(http_get(url, timeout=90))
        return data.get("PropertyTable", {}).get("Properties", [])
    except Exception as e:
        print(f"    batch err: {e}")
        return []


def main():
    print(f"=== nb360: Download PubChem AID {AID} (Tox21 hPXR qHTS) ===")
    # 1. download
    try:
        fp = download_assay_csv()
    except Exception as e:
        print(f"DOWNLOAD FAILED attempt1: {e}")
        time.sleep(5)
        try:
            fp = download_assay_csv()
        except Exception as e2:
            print(f"DOWNLOAD FAILED attempt2: {e2}")
            raise

    # 2. parse
    # PubChem concise CSV has metadata header rows; the real data starts after header
    df = pd.read_csv(fp, low_memory=False)
    print(f"  raw rows: {len(df)}, cols: {list(df.columns)[:12]}...")
    # drop metadata rows (PUBCHEM_RESULT_TAG is non-numeric for header rows)
    if "PUBCHEM_CID" in df.columns:
        cid_col = "PUBCHEM_CID"
    else:
        cid_col = [c for c in df.columns if "CID" in c.upper()][0]
    df[cid_col] = pd.to_numeric(df[cid_col], errors="coerce")
    df = df.dropna(subset=[cid_col]).copy()
    df[cid_col] = df[cid_col].astype(int)
    print(f"  rows w/ CID: {len(df)}")

    # outcome + activity fields
    outcome_col = next((c for c in df.columns if c.upper().endswith("ACTIVITY_OUTCOME")), None)
    score_col = next((c for c in df.columns if c.upper().endswith("ACTIVITY_SCORE")), None)
    print(f"  outcome={outcome_col}  score={score_col}")

    # Look for EC50 / potency cols
    pot_cols = [c for c in df.columns if any(k in c.upper() for k in ["EC50", "AC50", "POTENCY", "LOGAC50"])]
    print(f"  potency cols: {pot_cols}")

    # Aggregate to unique CID (Tox21 has multiple readouts per CID): max score, min EC50, modal outcome
    agg = {}
    if outcome_col:
        agg[outcome_col] = lambda s: s.mode().iloc[0] if len(s.mode()) else None
    if score_col:
        agg[score_col] = "max"
    for pc in pot_cols:
        agg[pc] = "min"  # most potent
    if not agg:
        print("  no useful agg columns -- using CID only")
        cid_summary = df[[cid_col]].drop_duplicates().rename(columns={cid_col: "cid"})
    else:
        cid_summary = df.groupby(cid_col).agg(agg).reset_index().rename(columns={cid_col: "cid"})
    print(f"  unique CIDs: {len(cid_summary)}")

    # 3. CID -> SMILES, InChIKey
    cids = cid_summary["cid"].astype(int).tolist()
    props_rows = []
    batch_size = 200
    t0 = time.time()
    for i in range(0, len(cids), batch_size):
        batch = cids[i:i+batch_size]
        props = fetch_props_batch(batch)
        props_rows.extend(props)
        if (i + batch_size) % 1000 == 0 or (i + batch_size) >= len(cids):
            print(f"    {min(i+batch_size, len(cids))}/{len(cids)} props={len(props_rows)} t={time.time()-t0:.0f}s")
        time.sleep(0.25)

    props_df = pd.DataFrame(props_rows).rename(columns={"CID": "cid",
                                                        "ConnectivitySMILES": "smiles",
                                                        "InChIKey": "inchikey"})
    print(f"  props rows: {len(props_df)}")
    merged = cid_summary.merge(props_df, on="cid", how="left")
    print(f"  merged rows: {len(merged)}  with smiles: {merged['smiles'].notna().sum()}")

    # Rename activity cols
    rename = {}
    if outcome_col: rename[outcome_col] = "aid_outcome"
    if score_col: rename[score_col] = "aid_score"
    for pc in pot_cols:
        rename[pc] = f"aid_{pc.lower().replace('pubchem_', '')}"
    merged = merged.rename(columns=rename)

    # binary active flag
    if "aid_outcome" in merged.columns:
        merged["aid_active"] = merged["aid_outcome"].astype(str).str.lower().eq("active").astype(int)

    # 4. compute per-compound features
    sys.path.insert(0, str(ROOT / "src"))
    from pxr.chem import standardize_smiles, morgan_fp_batch
    from rdkit import Chem

    keep = merged.dropna(subset=["smiles"]).copy()
    print(f"  computing features on {len(keep)} compounds...")
    std_smis = []
    ikeys = []
    for smi in keep["smiles"].tolist():
        try:
            ss = standardize_smiles(smi)
            std_smis.append(ss)
            ikeys.append(Chem.MolToInchiKey(Chem.MolFromSmiles(ss)) if ss else None)
        except Exception:
            std_smis.append(None); ikeys.append(None)
    keep["std_smiles"] = std_smis
    # prefer computed InChIKey but fall back to PubChem-provided
    keep["inchikey"] = [ik or pk for ik, pk in zip(ikeys, keep["inchikey"].tolist())]
    keep = keep.dropna(subset=["std_smiles", "inchikey"]).drop_duplicates(subset=["inchikey"])
    print(f"  after std+dedup: {len(keep)} unique InChIKeys")

    # Morgan FP 1024 -> feat_*
    fp = morgan_fp_batch(keep["std_smiles"].tolist(), radius=2, n_bits=1024)
    feat_cols = [f"feat_{i:04d}" for i in range(fp.shape[1])]
    feat_df = pd.DataFrame(fp, columns=feat_cols, index=keep.index).astype(np.uint8)
    final = pd.concat([keep.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)

    # also include assay-derived numeric features (active + score) as feat columns
    if "aid_active" in final.columns:
        final["feat_aid_active"] = final["aid_active"].astype(np.float32)
    if "aid_score" in final.columns:
        final["feat_aid_score"] = pd.to_numeric(final["aid_score"], errors="coerce").astype(np.float32)

    # Reorder
    front = [c for c in ["inchikey", "smiles", "std_smiles", "cid",
                          "aid_outcome", "aid_active", "aid_score"] if c in final.columns]
    pot_kept = [c for c in final.columns if c.startswith("aid_") and c not in front]
    feat_all = [c for c in final.columns if c.startswith("feat_")]
    final = final[front + pot_kept + feat_all]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(OUT, index=False)
    print(f"\nSAVED {OUT}")
    print(f"  rows: {len(final)}  feat cols: {len(feat_all)}  unique IK: {final['inchikey'].nunique()}")

    # Overlap with PXR train+test
    try:
        from pxr.data import load_train, load_test
        tr = load_train(); te = load_test()
        from rdkit import Chem
        tr_ik = set()
        for s in tr["smiles"]:
            try:
                tr_ik.add(Chem.MolToInchiKey(Chem.MolFromSmiles(s)))
            except: pass
        te_ik = set()
        for s in te["smiles"]:
            try:
                te_ik.add(Chem.MolToInchiKey(Chem.MolFromSmiles(s)))
            except: pass
        all_ik = tr_ik | te_ik
        hidden_ik = set(final["inchikey"].dropna().tolist())
        overlap = len(hidden_ik & all_ik)
        print(f"  overlap with PXR train+test InChIKeys: {overlap}")
    except Exception as e:
        print(f"  overlap calc failed: {e}")


if __name__ == "__main__":
    main()
