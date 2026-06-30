"""nb1050 — [AF3-B1] assemble the BFD-style PXR-ligand cofold candidate pool + diversity-prioritized order + budget.

The geometric model (B2) needs a LARGE, scaffold-diverse set of PXR-ligand cofolds. Consolidate every compound we
have a PXR-relevant signal for: CRC train (pEC50), counter (null), single-conc (weak log2FC, scaffold-diverse),
external PXR actives/inactives. Dedupe by InChIKey, annotate source + label + scaffold, and emit a cofold order that
greedily MAXIMIZES scaffold diversity (so a partial cofold still covers the most binding-mode space). Budget the GPU-h.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.chem import morgan_fp_batch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

D = "data/processed"; E = "data/external"


def ik_scaf(s):
    m = Chem.MolFromSmiles(str(s))
    if not m:
        return None, None
    try:
        return Chem.MolToInchiKey(m), MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    except Exception:
        return None, None


def main():
    pool = []  # smiles, source, label, label_type
    fc = pd.read_parquet(f"{D}/found_compounds.parquet")
    crc = pd.read_parquet(f"{D}/found_crc.parquet").set_index("comp_id")
    cnt = pd.read_parquet(f"{D}/found_counter.parquet").set_index("comp_id")
    sc = pd.read_parquet(f"{D}/found_sc.parquet")
    sc_ids = set(sc["comp_id"].unique())
    cid2smi = dict(zip(fc["comp_id"], fc["smiles"]))
    for cid, smi in cid2smi.items():
        if cid in crc.index:
            pool.append((smi, "crc", float(crc.loc[cid, "pec50"]), "pec50"))
        elif cid in cnt.index:
            pool.append((smi, "counter", float(cnt.loc[cid, "pec50_null"]), "null_pec50"))
        elif cid in sc_ids:
            pool.append((smi, "single_conc", np.nan, "log2fc_weak"))
    # external PXR actives/inactives (with SMILES) — binding-mode diversity
    for path, sc_col, src in [(f"{E}/pubchem_aid_1346985_tox21_pxr/aid_1346985.parquet", "std_smiles", "ext_tox21"),
                              (f"{E}/pubchem_aid_720659_ncats_pxr/aid_720659.parquet", "std_smiles", "ext_ncats"),
                              (f"{E}/papyrus_pxr_nr.parquet", "std_smiles", "ext_papyrus")]:
        if os.path.exists(path):
            df = pd.read_parquet(path)
            col = sc_col if sc_col in df.columns else [c for c in df.columns if "smile" in c.lower()][0]
            for s in df[col].dropna().astype(str):
                pool.append((s, src, np.nan, "ext"))
    raw = pd.DataFrame(pool, columns=["smiles", "source", "label", "label_type"])
    print(f"raw candidate rows: {len(raw)}")

    # dedupe by InChIKey, prefer CRC label
    iks, scafs = [], []
    for s in raw["smiles"]:
        ik, sc_ = ik_scaf(s); iks.append(ik); scafs.append(sc_)
    raw["ik"] = iks; raw["scaffold"] = scafs
    raw = raw.dropna(subset=["ik"])
    src_pri = {"crc": 0, "counter": 1, "ext_papyrus": 2, "ext_tox21": 2, "ext_ncats": 2, "single_conc": 3}
    raw["pri"] = raw["source"].map(src_pri).fillna(4)
    dedup = raw.sort_values("pri").drop_duplicates("ik").reset_index(drop=True)
    print(f"deduped unique compounds: {len(dedup)}  | unique scaffolds: {dedup['scaffold'].nunique()}")
    print("by source:", dedup["source"].value_counts().to_dict())

    # greedy scaffold-diverse order: round-robin by scaffold so partial cofold maximizes scaffold coverage
    dedup["si"] = dedup.groupby("scaffold").cumcount()
    order = dedup.sort_values(["si", "pri"]).reset_index(drop=True)
    order["cofold_idx"] = range(len(order))
    order[["cofold_idx", "smiles", "ik", "scaffold", "source", "label", "label_type"]].to_csv(f"{D}/bfd_cofold_candidates.csv", index=False)

    n = len(order); per = 1.5  # min/ligand on A100 (5 samples)
    print(f"\nGPU-h budget @ {per} min/ligand A100:")
    for k in [2000, 5000, 10000, n]:
        kk = min(k, n)
        print(f"  first {kk:6d} compounds ({order.iloc[:kk]['scaffold'].nunique()} scaffolds): {kk*per/60:.1f} A100-h")
    json.dump({"unique": int(len(dedup)), "scaffolds": int(dedup["scaffold"].nunique()),
               "by_source": {k: int(v) for k, v in dedup["source"].value_counts().items()},
               "full_a100_h": round(n * per / 60, 1)}, open(f"{D}/nb1050_bfd_summary.json", "w"), indent=2)
    print(f"\nsaved bfd_cofold_candidates.csv (diversity-ordered, {n} compounds)")


if __name__ == "__main__":
    main()
