"""nb1030 — identify & rank the NOVEL test compounds (the F2 tail) and emit a target list for external-data hunting.

Novelty = 1 - max Tanimoto(test compound, full TRAIN corpus). The train corpus = every compound we have a label
for across ALL configs (CRC + counter + single-conc + crudes), i.e. found_compounds[in_test==0]. The most novel
test compounds are where nb3200 fails (post-mortem F2). Output a ranked CSV so we can go fetch external data for them.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test
from src.pxr.chem import morgan_fp_batch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

D = "data/processed"


def main():
    comp = pd.read_parquet(f"{D}/found_compounds.parquet")
    train = comp[comp["in_test"] == 0].reset_index(drop=True)
    te = load_test().reset_index(drop=True)
    unb = set(np.load(f"{D}/_audit_unblind_idx.npy").tolist())

    print(f"train corpus: {len(train)} compounds | test: {len(te)}")
    trfp = morgan_fp_batch(train["smiles"].tolist()).astype(bool)
    tefp = morgan_fp_batch(te["smiles"].tolist()).astype(bool)

    rows = []
    for i in range(len(te)):
        inter = (tefp[i] & trfp).sum(1); uni = (tefp[i] | trfp).sum(1)
        sims = inter / np.clip(uni, 1, None)
        order = np.argsort(-sims)[:3]
        m = Chem.MolFromSmiles(str(te["smiles"].iloc[i]))
        ik = Chem.MolToInchiKey(m) if m else None
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else None
        rows.append({"test_pos": i, "name": te["name"].iloc[i], "smiles": te["smiles"].iloc[i],
                     "inchikey": ik, "scaffold": scaf, "top1_sim": float(sims[order[0]]),
                     "top3_mean_sim": float(sims[order].mean()), "in_unb": int(i in unb)})
    df = pd.DataFrame(rows).sort_values("top1_sim").reset_index(drop=True)
    df.to_csv(f"{D}/novel_targets.csv", index=False)

    for thr in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        n = (df["top1_sim"] < thr).sum(); nu = ((df["top1_sim"] < thr) & (df["in_unb"] == 1)).sum()
        print(f"  top1_sim < {thr}: {n}/513 test  ({nu} in the 253 eval)")
    print(f"\nmost-novel 10 (top1_sim):")
    print(df[["test_pos", "name", "top1_sim", "scaffold", "in_unb"]].head(10).to_string(index=False))
    json.dump({"median_top1": float(df["top1_sim"].median()),
               "n_below_0.5": int((df["top1_sim"] < 0.5).sum()),
               "n_below_0.4_unb": int(((df["top1_sim"] < 0.4) & (df["in_unb"] == 1)).sum())},
              open(f"{D}/nb1030_novel_summary.json", "w"), indent=2)
    print(f"\nsaved {D}/novel_targets.csv")


if __name__ == "__main__":
    main()
