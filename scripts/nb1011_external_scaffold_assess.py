"""nb1011 — can external ChEMBL PXR actives break the OOD scaffold wall?
The documented TRUE activity ceiling = scaffold support on the analog-expansion test (rare-scaffold tail = F2).
Decisive check: do external ChEMBL CHEMBL3401 (PXR) compounds POPULATE the test-513 rare-scaffold regions our
train-4139 misses? If externals just re-cover train scaffolds -> won't help. If they raise the near-neighbor floor
for low-train-similarity test compounds -> external data is a real OOD lever. (No retrain yet — feasibility only.)
"""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

TR = "data/processed/unimol_train.csv"; TE = "data/processed/unimol_test513.csv"
OUT = "C:/pxr_struct/external"; os.makedirs(OUT, exist_ok=True)


def murcko(s):
    m = Chem.MolFromSmiles(str(s));
    return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else None


def fps(smis):
    out, valid = [], []
    for s in smis:
        m = Chem.MolFromSmiles(str(s))
        if m:
            out.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)); valid.append(s)
    return out, valid


def max_sim_to(query_fps, ref_fps):
    res = []
    for q in query_fps:
        s = DataStructs.BulkTanimotoSimilarity(q, ref_fps)
        res.append(max(s) if s else 0.0)
    return np.array(res)


def main():
    f = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "**", "tool-results",
               "mcp-claude_ai_ChEMBL-get_bioactivity-*.txt"), recursive=True))
    f += sorted(glob.glob("D:/Users/ashenoy00000/.claude/projects/*/*/tool-results/mcp-claude_ai_ChEMBL-get_bioactivity-*.txt"))
    f = sorted(set(f))[-1]
    d = json.load(open(f, encoding="utf-8"))
    acts = d.get("activities", [])
    print(f"ChEMBL pull: total={d.get('total')} returned={len(acts)}")
    # extract SMILES + pchembl
    skeys = [k for k in acts[0].keys()] if acts else []
    smi_key = next((k for k in ["canonical_smiles", "smiles", "molecule_smiles"] if k in skeys), None)
    print(f"activity keys (sample): {skeys[:12]}\n  smiles_key={smi_key}")
    ext = {}
    for a in acts:
        smi = a.get(smi_key); pc = a.get("pchembl_value")
        if not smi:
            continue
        ext.setdefault(smi, []).append(float(pc) if pc not in (None, "") else np.nan)
    ext_smiles = list(ext.keys())
    print(f"external unique compounds (by raw SMILES): {len(ext_smiles)}")

    tr = pd.read_csv(TR)["smiles"].astype(str).tolist()
    te = pd.read_csv(TE)["smiles"].astype(str).tolist()
    sc_tr = set(filter(None, (murcko(s) for s in tr)))
    sc_te = [murcko(s) for s in te]
    sc_ext = set(filter(None, (murcko(s) for s in ext_smiles)))
    print(f"\nscaffolds: train={len(sc_tr)} test_unique={len(set(filter(None,sc_te)))} external={len(sc_ext)}")

    # test scaffolds missing from train
    te_missing = [s for s in sc_te if s and s not in sc_tr]
    te_missing_u = set(te_missing)
    rescued = {s for s in te_missing_u if s in sc_ext}
    print(f"test-513 compounds on a scaffold MISSING from train: {len(te_missing)}/{len(te)}")
    print(f"  unique missing scaffolds: {len(te_missing_u)}; of those PRESENT in external ChEMBL: {len(rescued)} ({100*len(rescued)/max(1,len(te_missing_u)):.0f}%)")

    # near-neighbor floor: does external raise max-sim for low-train-sim test compounds?
    te_fp, te_v = fps(te); tr_fp, _ = fps(tr); ext_fp, _ = fps(ext_smiles)
    sim_tr = max_sim_to(te_fp, tr_fp); sim_ext = max_sim_to(te_fp, ext_fp)
    low = sim_tr < 0.35   # OOD test compounds (rare-scaffold tail)
    print(f"\nnear-neighbor floor (n_test_valid={len(te_v)}):")
    print(f"  median max-Tanimoto to TRAIN={np.median(sim_tr):.3f}  to EXTERNAL={np.median(sim_ext):.3f}")
    print(f"  OOD test compounds (train-sim<0.35): {int(low.sum())}")
    if low.sum():
        gain = sim_ext[low] - sim_tr[low]
        print(f"    of these, external provides a CLOSER neighbor for {int((gain>0).sum())}/{int(low.sum())} ({100*(gain>0).mean():.0f}%)")
        print(f"    mean sim lift on OOD tail: {gain.mean():+.3f} (median external-sim {np.median(sim_ext[low]):.3f} vs train {np.median(sim_tr[low]):.3f})")
    verdict = low.sum() and (sim_ext[low] - sim_tr[low] > 0).mean() > 0.5 and np.median(sim_ext[low]) > 0.4
    print("\n" + "=" * 62)
    print(">>> EXTERNAL DATA IS A REAL OOD LEVER -> worth a scaffold-diverse retrain" if verdict
          else ">>> external ChEMBL PXR does NOT cover the test rare-scaffold tail -> OOD wall holds (data-only won't break it)")
    print("=" * 62)
    pd.DataFrame({"smiles": ext_smiles, "pchembl_mean": [np.nanmean(ext[s]) for s in ext_smiles]}).to_csv(f"{OUT}/chembl_pxr_ec50.csv", index=False)
    json.dump({"n_external": len(ext_smiles), "n_ext_scaffolds": len(sc_ext), "test_missing_from_train": len(te_missing),
               "missing_scaffolds_rescued_by_external": len(rescued), "median_sim_train": float(np.median(sim_tr)),
               "median_sim_external": float(np.median(sim_ext)), "n_ood": int(low.sum()),
               "ood_closer_frac": float((sim_ext[low] - sim_tr[low] > 0).mean()) if low.sum() else 0.0,
               "verdict_real_lever": bool(verdict)}, open(f"{OUT}/nb1011_external_assess.json", "w"), indent=2)
    print(f"saved -> {OUT}/chembl_pxr_ec50.csv + nb1011_external_assess.json")


if __name__ == "__main__":
    main()
