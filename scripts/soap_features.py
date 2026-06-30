"""dscribe SOAP geometric descriptors for the PXR activity corpus.

The QUEUED other half of the strain row (feedback_strain_mmff_WIN): SOAP encodes
the local 3D atomic ENVIRONMENT (rotation/translation invariant power spectrum),
a DISTINCT geometric observable from AIMNet2's electronic scalars and from the
MMFF strain/flexibility scalars. Same conformer-ensemble axis that keeps winning
marginally (StrainRelief memory).

Per mol: embed 1 ETKDGv3 conformer, MMFF94 optimize, build an ASE Atoms, compute
SOAP with average="inner" (one fixed-length molecular vector regardless of size).
Cache the raw averaged SOAP vectors to an npz (name -> vector); a later reduce
step PCA-compresses to a handful of scalars for the GBM block.

Resumable: writes partial npz every CHUNK mols; on restart skips cached names.
Run:  .venv/Scripts/python.exe scripts/soap_features.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import csv
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
from ase import Atoms
from dscribe.descriptors import SOAP

CORPUS = "C:/pxr_work/xtb/corpus.csv"
OUTDIR = "C:/pxr_work/soap"
NPZ = os.path.join(OUTDIR, "soap_raw.npz")
META = os.path.join(OUTDIR, "soap_meta.csv")  # name,src,smiles,status
os.makedirs(OUTDIR, exist_ok=True)
CHUNK = 100

SPECIES = ["H", "B", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
SOAP_DESC = SOAP(species=SPECIES, r_cut=5.0, n_max=3, l_max=3,
                 periodic=False, average="inner", sparse=False)
NFEAT = SOAP_DESC.get_number_of_features()


def load_corpus():
    with open(CORPUS, newline="") as f:
        return [(r["name"], r["src"], r["smiles"]) for r in csv.DictReader(f)]


def load_cache():
    if not os.path.exists(NPZ):
        return {}
    d = np.load(NPZ, allow_pickle=True)
    return {n: d["X"][i] for i, n in enumerate(d["names"])}


def conformer_atoms(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        raise RuntimeError("parse_fail")
    for s in SPECIES:
        pass
    syms = {a.GetSymbol() for a in m.GetAtoms()}
    extra = syms - set(SPECIES)
    if extra:
        raise RuntimeError(f"unsupported:{','.join(sorted(extra))}")
    m = Chem.AddHs(m)
    p = AllChem.ETKDGv3(); p.randomSeed = 42; p.numThreads = 0
    if AllChem.EmbedMolecule(m, p) != 0:
        p.useRandomCoords = True
        if AllChem.EmbedMolecule(m, p) != 0:
            raise RuntimeError("embed_fail")
    try:
        AllChem.MMFFOptimizeMolecule(m, maxIters=1000)
    except Exception:
        pass
    conf = m.GetConformer()
    pos = conf.GetPositions()
    z = [a.GetSymbol() for a in m.GetAtoms()]
    return Atoms(symbols=z, positions=pos)


def main():
    rows = load_corpus()
    cache = load_cache()
    meta_done = set(cache.keys())
    todo = [r for r in rows if r[0] not in meta_done]
    print(f"corpus {len(rows)} | cached {len(cache)} | todo {len(todo)} | nfeat {NFEAT}", flush=True)

    meta_existed = os.path.exists(META)
    mf = open(META, "a", newline="")
    mw = csv.DictWriter(mf, fieldnames=["name", "src", "smiles", "status"])
    if not meta_existed:
        mw.writeheader()

    ok = err = 0
    for i, (name, src, smi) in enumerate(todo):
        status = "ok"
        try:
            at = conformer_atoms(smi)
            vec = SOAP_DESC.create(at)  # (NFEAT,)
            cache[name] = vec.astype(np.float32)
            ok += 1
        except Exception as e:
            status = f"err:{str(e)[:40]}"
            cache[name] = np.full(NFEAT, np.nan, np.float32)
            err += 1
        mw.writerow({"name": name, "src": src, "smiles": smi, "status": status})
        if (i + 1) % CHUNK == 0:
            mf.flush()
            names = list(cache.keys())
            X = np.vstack([cache[n] for n in names])
            np.savez(NPZ, names=np.array(names, object), X=X)
            print(f"  {i+1}/{len(todo)} ok={ok} err={err}", flush=True)
    mf.flush(); mf.close()
    names = list(cache.keys())
    X = np.vstack([cache[n] for n in names])
    np.savez(NPZ, names=np.array(names, object), X=X)
    print(f"DONE ok={ok} err={err} | cached {len(names)}", flush=True)


if __name__ == "__main__":
    main()
