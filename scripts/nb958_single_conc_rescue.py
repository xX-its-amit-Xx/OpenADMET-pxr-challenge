"""nb958 — can the challenge's OWN single-concentration screen (21,003 cpds) rescue
the novel-scaffold test wall where external ChEMBL data could not?

ChEMBL PXR harvest (nb-subagent): of 334 novel-to-train test scaffolds, external data
covered 1, proximity median Tanimoto 0.23 (off-manifold). But the single-conc screen
is the SAME campaign the test analogs were expanded from — it should be local. This
probes test-scaffold coverage + proximity for the INTERNAL screen, the direct analog
of the ChEMBL rescue metric.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train, load_test
from src.pxr.chem import morgan_fp_batch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

D = "data/processed"


def murcko(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else None
    except Exception:
        return None


def load_single_conc():
    try:
        from src.pxr.data import load_single_conc as L
        return L()
    except Exception:
        import pandas as pd
        for p in ["data/raw/pxr-challenge_single_concentration_TRAIN.csv"]:
            if os.path.exists(p):
                return pd.read_csv(p)
        raise FileNotFoundError("single-conc CSV not found")


def max_tan(fp_a, fp_b, bs=200):
    """max Tanimoto of each row in fp_a to any row in fp_b (dense bits)."""
    A = fp_a.astype(np.float32); B = fp_b.astype(np.float32)
    bsum = B.sum(1)[None, :]
    out = np.empty(len(A))
    for i in range(0, len(A), bs):
        chunk = A[i:i+bs]
        inter = chunk @ B.T
        u = chunk.sum(1)[:, None] + bsum - inter
        u[u == 0] = 1.0
        out[i:i+bs] = (inter / u).max(1)
    return out


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test()
    sc = load_single_conc()
    print("single-conc columns:", list(sc.columns)[:12])
    sc_smi_col = "smiles" if "smiles" in sc.columns else [c for c in sc.columns if "smi" in c.lower()][0]
    sc_smiles = sc[sc_smi_col].dropna().unique().tolist()
    print(f"single-conc: {len(sc)} rows, {len(sc_smiles)} unique SMILES")

    tr_scaf = set(murcko(s) for s in tr["smiles"]) - {None}
    te_smiles = te["smiles"].tolist()
    te_scaf = [murcko(s) for s in te_smiles]
    novel_mask = np.array([s is None or s not in tr_scaf for s in te_scaf])
    novel_scafs = set(s for s, nv in zip(te_scaf, novel_mask) if nv and s)
    print(f"test: {len(te_smiles)} cpds; novel-to-train scaffolds present = {len(novel_scafs)} "
          f"(on {int(novel_mask.sum())} test cpds)")

    # single-conc scaffold coverage of the novel test scaffolds
    sc_scaf = set(murcko(s) for s in sc_smiles) - {None}
    covered = novel_scafs & sc_scaf
    print(f"\n=== TEST-SCAFFOLD RESCUE (internal single-conc) ===")
    print(f"novel test scaffolds covered by single-conc: {len(covered)} / {len(novel_scafs)} "
          f"({100*len(covered)/max(1,len(novel_scafs)):.1f}%)")
    print(f"  (ChEMBL external covered 1 / 334 = 0.3% for comparison)")

    # proximity: for novel-scaffold test cpds, max Tanimoto to single-conc
    fp_te = morgan_fp_batch([te_smiles[i] for i in range(len(te_smiles)) if novel_mask[i]])
    # subsample single-conc to 8000 for the Tanimoto matrix (memory)
    import random
    random.seed(42)
    sc_sample = sc_smiles if len(sc_smiles) <= 8000 else random.sample(sc_smiles, 8000)
    fp_sc = morgan_fp_batch(sc_sample)
    sims = max_tan(fp_te, fp_sc)
    print(f"\n=== PROXIMITY (novel-scaffold test cpds -> single-conc) ===")
    print(f"max-Tanimoto to single-conc: median={np.median(sims):.3f} "
          f"p25={np.percentile(sims,25):.3f} p75={np.percentile(sims,75):.3f}")
    print(f"  frac >=0.4: {np.mean(sims>=0.4):.3f}   >=0.5: {np.mean(sims>=0.5):.3f}   "
          f">=0.7: {np.mean(sims>=0.7):.3f}")
    print(f"  (ChEMBL external: median 0.23, frac>=0.4 = 0.003, ceiling 0.39)")

    # how many single-conc compounds are 'active' (usable weak positive labels)?
    act_info = {}
    for col in ["log2FC", "log2fc", "fold_change", "FDR", "fdr"]:
        if col in sc.columns:
            act_info[col] = [float(sc[col].min()), float(sc[col].median()), float(sc[col].max())]
    print(f"\nsingle-conc activity columns ranges: {act_info}")

    out = {
        "single_conc_unique": len(sc_smiles),
        "novel_test_scaffolds": len(novel_scafs),
        "novel_covered_by_single_conc": len(covered),
        "novel_covered_frac": round(len(covered)/max(1,len(novel_scafs)), 4),
        "proximity_median": round(float(np.median(sims)), 4),
        "proximity_frac_ge_0.4": round(float(np.mean(sims>=0.4)), 4),
        "proximity_frac_ge_0.5": round(float(np.mean(sims>=0.5)), 4),
        "proximity_frac_ge_0.7": round(float(np.mean(sims>=0.7)), 4),
    }
    json.dump(out, open(f"{D}/nb958_single_conc_rescue.json", "w"), indent=2)
    print(f"\n=== VERDICT ===")
    if out["novel_covered_frac"] > 0.2 and out["proximity_frac_ge_0.5"] > 0.1:
        print(">>> single-conc COVERS the novel test wall with near neighbors -> INTERNAL rescue lever LIVE")
    elif out["novel_covered_frac"] > 0.1 or out["proximity_frac_ge_0.4"] > 0.2:
        print(">>> single-conc PARTIALLY covers the wall -> worth a weak-label aug test")
    else:
        print(">>> single-conc also off-manifold -> test analogs are isolated even from the screen")
    print(f"saved -> {D}/nb958_single_conc_rescue.json")


if __name__ == "__main__":
    main()
