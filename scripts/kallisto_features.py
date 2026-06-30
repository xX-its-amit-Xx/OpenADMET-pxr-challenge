"""kallisto classical descriptors (polarizability + proximity + coordination).

Cheap, no-compiler, pure-Python physics axis from the QUEUED backlog (ledger row
'kallisto', cy-380 'cheapest no-sink physics FIRST'). Distinct observable from the
deployed blocks: AIMNet2 = learned electronic charges/forces; strain = MMFF conformer
energetics. kallisto gives the STATIC MOLECULAR POLARIZABILITY (alpha) axis (cf.
MACE-POLAR memory: alpha is a distinct observable worth gating) plus its unique atomic
PROXIMITY (steric-crowding) descriptor and coordination numbers.

ETKDGv3 + MMFF single conformer -> kallisto Molecule (coords in BOHR) -> 9 scalars.
Resumable per-row append. Output: C:/pxr_work/kallisto/kallisto_features.csv
"""
import os, sys, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
from kallisto.molecule import Molecule

CORPUS = "C:/pxr_work/xtb/corpus.csv"
OUTDIR = "C:/pxr_work/kallisto"; os.makedirs(OUTDIR, exist_ok=True)
OUT = f"{OUTDIR}/kallisto_features.csv"
BOHR = 1.8897259886
COLS = ["k_alp_sum", "k_alp_mean", "k_alp_max", "k_alp_std",
        "k_prox_mean", "k_prox_max", "k_prox_std", "k_cn_mean", "k_cn_std"]


def featurize(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    m = Chem.AddHs(m)
    p = AllChem.ETKDGv3(); p.randomSeed = 42
    if AllChem.EmbedMolecule(m, p) != 0:
        if AllChem.EmbedMolecule(m, useRandomCoords=True, randomSeed=42) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(m, maxIters=500)
    except Exception:
        pass
    conf = m.GetConformer()
    nums = np.array([a.GetAtomicNum() for a in m.GetAtoms()])
    pos = np.array([list(conf.GetAtomPosition(i)) for i in range(m.GetNumAtoms())]) * BOHR
    mol = Molecule(numbers=nums, positions=pos)
    alp = np.asarray(mol.get_alp(0), float)
    prox = np.asarray(mol.get_prox((3, 5)), float)
    cn = np.asarray(mol.get_cns("cov"), float)
    return [float(alp.sum()), float(alp.mean()), float(alp.max()), float(alp.std()),
            float(prox.mean()), float(prox.max()), float(prox.std()),
            float(cn.mean()), float(cn.std())]


def main():
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for r in csv.DictReader(f):
                done.add(r["name"])
    rows = list(csv.DictReader(open(CORPUS)))
    print(f"corpus={len(rows)} already_done={len(done)}", flush=True)
    new = os.path.exists(OUT)
    fh = open(OUT, "a", newline="")
    w = csv.writer(fh)
    if not new:
        w.writerow(["name", "src", "smiles"] + COLS + ["status"])
    n = 0
    for r in rows:
        if r["name"] in done:
            continue
        try:
            feats = featurize(r["smiles"])
            if feats is None:
                w.writerow([r["name"], r["src"], r["smiles"]] + [""] * len(COLS) + ["embed_fail"])
            else:
                w.writerow([r["name"], r["src"], r["smiles"]] + feats + ["ok"])
        except Exception as e:
            w.writerow([r["name"], r["src"], r["smiles"]] + [""] * len(COLS) + [f"err:{str(e)[:40]}"])
        n += 1
        if n % 200 == 0:
            fh.flush(); print(f"  {n} done", flush=True)
    fh.close()
    print(f"DONE wrote {n} new rows -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
