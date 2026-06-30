"""nb1095 — Tox21/NCATS PXR pseudo-labels: STAGE 1 = fetch + coverage decision.

The decisive question before any modeling: do the Tox21 hPXR-agonist actives COVER our 513 test compounds —
especially the novel-scaffold BLIND SPOTS that the catalog libraries (nb1094) could NOT reach (best cand sim to the
30 worst-covered test compounds was only 0.41, +0.003)? If Tox21's structurally-diverse environmental chemical space
reaches those blind spots, this is real coverage-expanding data (and escapes the cycle-299 'synthetic=chemistry-bounded'
saturation, because these are REAL measurements). If not, the lever is weak.

Fetch (PubChem PUG-REST concise): PXR-agonist summary (AID 1347033), CYP3A4-via-PXR (1346984, corroboration),
AhR (743122, de-confounder). Get SMILES, standardize, dedup vs our 4139 train (inchikey), compute Morgan coverage of
the 513 test + the blind-spot tail. Cache raw pulls to data/external/tox21/.
"""
import os, sys, json, time
import numpy as np, pandas as pd, requests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test, load_train
from src.pxr.chem import morgan_fp_batch
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

B = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
OUT = "data/external/tox21"; os.makedirs(OUT, exist_ok=True)
AIDS = {"pxr_agonist": 1347033, "cyp3a4_via_pxr": 1346984, "ahr": 743122}


def fetch_concise(aid):
    f = f"{OUT}/aid_{aid}_concise.parquet"
    if os.path.exists(f):
        return pd.read_parquet(f)
    r = requests.get(f"{B}/assay/aid/{aid}/concise/JSON", timeout=120); r.raise_for_status()
    t = r.json()["Table"]; cols = t["Columns"]["Column"]
    df = pd.DataFrame([row["Cell"] for row in t["Row"]], columns=cols)
    df.to_parquet(f); return df


def cids_to_smiles(cids):
    """batch fetch SMILES for CIDs (PubChem property; key is 'SMILES' or 'ConnectivitySMILES')."""
    out = {}
    cids = [int(c) for c in cids if str(c).isdigit()]
    for i in range(0, len(cids), 100):
        chunk = cids[i:i + 100]
        url = f"{B}/compound/cid/{','.join(map(str, chunk))}/property/SMILES,IsomericSMILES/JSON"
        try:
            r = requests.get(url, timeout=120)
            if r.status_code != 200:
                url2 = f"{B}/compound/cid/{','.join(map(str, chunk))}/property/CanonicalSMILES/JSON"
                r = requests.get(url2, timeout=120)
            for p in r.json()["PropertyTable"]["Properties"]:
                s = p.get("IsomericSMILES") or p.get("SMILES") or p.get("ConnectivitySMILES") or p.get("CanonicalSMILES")
                if s: out[p["CID"]] = s
        except Exception as e:
            print("  smiles batch err", i, str(e)[:60])
        time.sleep(0.25)
    return out


def ik(s):
    m = Chem.MolFromSmiles(str(s)); return Chem.MolToInchiKey(m) if m else None


def fpf(sm):
    return (morgan_fp_batch(sm).astype(np.float32) > 0).astype(np.float32)


def tani_cross(A, B_):
    inter = A @ B_.T
    return inter / np.clip(A.sum(1)[:, None] + B_.sum(1)[None, :] - inter, 1, None)


def main():
    print("fetching Tox21 concise tables...", flush=True)
    tabs = {k: fetch_concise(a) for k, a in AIDS.items()}
    for k, df in tabs.items():
        oc = df["Activity Outcome"].value_counts().to_dict()
        print(f"  {k}: {len(df)} rows, outcomes={oc}", flush=True)

    pxr = tabs["pxr_agonist"]
    pxr_act = pxr[pxr["Activity Outcome"] == "Active"]["CID"].dropna().unique().tolist()
    print(f"\nPXR agonist actives (unique CID): {len(pxr_act)}", flush=True)

    print("fetching SMILES for PXR actives...", flush=True)
    smi_map = cids_to_smiles(pxr_act)
    json.dump({str(k): v for k, v in smi_map.items()}, open(f"{OUT}/pxr_active_smiles.json", "w"))
    act = pd.DataFrame([(c, smi_map[c]) for c in smi_map], columns=["cid", "smiles"])
    act["ik"] = act["smiles"].map(ik); act = act.dropna(subset=["ik"]).drop_duplicates("ik").reset_index(drop=True)
    print(f"  standardized unique actives: {len(act)}", flush=True)

    # dedup vs our train
    tr = load_train().dropna(subset=["pec50"]); te = load_test()
    tr_ik = set(tr["smiles"].map(ik).dropna())
    act_new = act[~act["ik"].isin(tr_ik)].reset_index(drop=True)
    print(f"  Tox21 PXR actives NOT already in our train: {len(act_new)} (of {len(act)})", flush=True)

    # COVERAGE DECISION: do these cover the test blind spots?
    Fte = fpf(te["smiles"].tolist()); Ftr = fpf(tr["smiles"].tolist()); Fca = fpf(act_new["smiles"].tolist())
    cov_tr = tani_cross(Ftr, Fte).max(0)                      # current train coverage per test
    cov_tx = tani_cross(Fca, Fte).max(0)                      # best Tox21-active coverage per test
    blind = np.argsort(cov_tr)[:30]
    print("\n=== COVERAGE DECISION (vs nb1094 catalog: blind-30 best cand sim 0.41, +0.003) ===")
    print(f"  test median coverage  train-only {np.median(cov_tr):.3f} -> +Tox21 {np.median(np.maximum(cov_tr,cov_tx)):.3f}")
    print(f"  blind-30 mean max-sim train-only {cov_tr[blind].mean():.3f} -> best Tox21 {cov_tx[blind].mean():.3f} "
          f"(delta {cov_tx[blind].mean()-cov_tr[blind].mean():+.3f})")
    print(f"  test compounds where a Tox21 active beats train by >0.05: {((cov_tx-cov_tr)>0.05).sum()}/513")
    print(f"  test compounds where a Tox21 active beats train by >0.10: {((cov_tx-cov_tr)>0.10).sum()}/513")

    act_new.to_csv(f"{OUT}/pxr_actives_new.csv", index=False)
    rep = dict(n_pxr_actives=int(len(act)), n_new=int(len(act_new)),
               test_median_cov_before=float(np.median(cov_tr)),
               test_median_cov_after=float(np.median(np.maximum(cov_tr, cov_tx))),
               blind30_before=float(cov_tr[blind].mean()), blind30_tox21=float(cov_tx[blind].mean()),
               n_test_improved_0p05=int(((cov_tx - cov_tr) > 0.05).sum()),
               n_test_improved_0p10=int(((cov_tx - cov_tr) > 0.10).sum()))
    json.dump(rep, open("data/processed/nb1095_tox21_coverage.json", "w"), indent=2)
    print(f"\nwrote {OUT}/pxr_actives_new.csv + coverage report")


if __name__ == "__main__":
    main()
