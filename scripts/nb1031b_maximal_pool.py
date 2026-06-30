"""nb1031b — MAXIMAL external pool coverage test. Pool EVERY SMILES-bearing source (PXR activity + CYP3A4 +
ADMET panel + Tox21 multi-NR + PubChem PXR actives) and measure, per test compound, the best Morgan neighbor and
what data it carries. Decisive test of "is there ANY near-neighbor with ANY data, anywhere we can reach?"
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test
from src.pxr.chem import morgan_fp_batch
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

D = "data/processed"; E = "data/external"


def main():
    pool = []
    def add(path, sc, label):
        if not os.path.exists(path):
            print(f"  MISSING {path}"); return
        df = pd.read_parquet(path) if path.endswith("parquet") else pd.read_csv(path)
        if sc not in df.columns:
            sc2 = [c for c in df.columns if "smile" in c.lower()]
            if not sc2:
                print(f"  no smiles in {label}"); return
            sc = sc2[0]
        sub = pd.DataFrame({"smiles": df[sc].astype(str), "source": label})
        pool.append(sub); print(f"  + {label}: {len(sub)}")

    add(f"{E}/pubchem_aid_1346985_tox21_pxr/aid_1346985.parquet", "std_smiles", "pxr_tox21")
    add(f"{E}/pubchem_aid_720659_ncats_pxr/aid_720659.parquet", "std_smiles", "pxr_ncats")
    add(f"{E}/papyrus_pxr_nr.parquet", "std_smiles", "pxr_papyrus")
    add(f"{E}/chembl_pxr_all_types.parquet", "smiles", "pxr_chembl")
    add(f"{E}/bindingdb_pxr_direct.parquet", "std_smiles", "pxr_bindingdb")
    add(f"{E}/pubchem_pxr_active_smiles.parquet", "smiles", "pxr_pubchem_active")
    add(f"{E}/openadmet_octant_cyp/cyp3a4_inhibition.parquet", "std_smiles", "cyp3a4")
    add(f"{E}/openadmet_expansionrx/admet_panel.parquet", "std_smiles", "admet_panel")
    add(f"{E}/tox21_nr_data.parquet", "smiles", "tox21_nr")

    ref = pd.concat(pool, ignore_index=True).dropna()
    ref = ref[ref["smiles"].str.len() > 3].drop_duplicates("smiles").reset_index(drop=True)
    print(f"\nMAXIMAL pool: {len(ref)} unique SMILES")

    te = load_test().reset_index(drop=True)
    nov = pd.read_csv(f"{D}/novel_targets.csv")[["test_pos", "top1_sim", "in_unb"]]

    reffp = morgan_fp_batch(ref["smiles"].tolist()).astype(np.uint8)
    tefp = morgan_fp_batch(te["smiles"].tolist()).astype(np.uint8)
    refsum = reffp.sum(1); refsrc = ref["source"].to_numpy()

    best_sim = np.zeros(len(te)); best_src = np.empty(len(te), dtype=object)
    for i in range(len(te)):
        inter = reffp @ tefp[i]; uni = refsum + tefp[i].sum() - inter
        sims = inter / np.clip(uni, 1, None)
        j = int(np.argmax(sims)); best_sim[i] = sims[j]; best_src[i] = refsrc[j]
    out = pd.DataFrame({"test_pos": range(len(te)), "pool_best_sim": best_sim, "pool_best_src": best_src}).merge(nov, on="test_pos")
    out.to_csv(f"{D}/test_maximal_pool.csv", index=False)

    print(f"\n=== MAXIMAL-pool best-neighbor coverage (Morgan ECFP4) ===")
    for thr in [0.85, 0.7, 0.6, 0.5, 0.4]:
        n = (out.pool_best_sim >= thr).sum(); nov_n = ((out.pool_best_sim >= thr) & (out.top1_sim < 0.5)).sum()
        print(f"  best neighbor >= {thr}: {n}/513 test   ({nov_n} of the 127 novel-tail)")
    print(f"  overall median best-neighbor sim: {out.pool_best_sim.median():.3f}")
    print(f"  novel-tail median best-neighbor sim: {out[out.top1_sim<0.5].pool_best_sim.median():.3f}")
    print(f"\n  source of best neighbor (counts): {out.pool_best_src.value_counts().to_dict()}")
    json.dump({"pool_size": len(ref), "median_best_sim": float(out.pool_best_sim.median()),
               "n_ge_0.6": int((out.pool_best_sim >= 0.6).sum()),
               "novel_ge_0.6": int(((out.pool_best_sim >= 0.6) & (out.top1_sim < 0.5)).sum())},
              open(f"{D}/nb1031b_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
