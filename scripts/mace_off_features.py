"""MACE-OFF learned-NNP QM/physics scalar features for the PXR activity corpus.

Sibling of scripts/aimnet2_features.py. AIMNet2 (DFT-trained NNP) just WON the honest gate
(matched -0.0070, deployed nb1177). Memory feedback_aimnet2_qm_WIN says learned DFT-trained NNPs
ESCAPE the chempropembed sink where semi-empirical xTB did not, and to prioritize
MACE-OFF / OrbMol / UMA / g-xTB (scalars only, NOT embeddings). This is the MACE-OFF probe.

Reuses the existing standardized corpus C:/pxr_work/xtb/corpus.csv (4139 train + 513 test) and
the SAME ETKDGv3 + MMFF conformer as aimnet2_features.py so the two NNPs are compared on identical
geometries. MACE-OFF (medium, SPICE/wB97M-D3 trained) -> energy + forces per mol.

Derived QM scalars (a DIFFERENT functional from AIMNet2 -> potential orthogonal energy/strain axis):
  mace_energy       total potential energy (eV)
  mace_epa          energy per atom (eV/atom)  ~ atomization-like size-normalized stability
  mace_fmax         max atomic force norm on the MMFF geom (eV/Ang) ~ MACE-PES strain of MMFF pose
  mace_frms         rms atomic force norm (eV/Ang)
  mace_nestd        std of per-atom node energies (eV) ~ electronic-environment heterogeneity
  mace_nemin        min per-atom node energy (eV)
  mace_nemax        max per-atom node energy (eV)

Resumable: appends per row; on restart skips names already in the output CSV.
Run (isolated venv, mace-torch installed):
  TORCHDYNAMO_DISABLE=1 C:/aimnet_venv/Scripts/python.exe scripts/mace_off_features.py
"""
import os, sys, csv
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
from mace.calculators import mace_off

CORPUS = "C:/pxr_work/xtb/corpus.csv"
OUTDIR = "C:/pxr_work/mace_off"
OUT = os.path.join(OUTDIR, "mace_features.csv")
os.makedirs(OUTDIR, exist_ok=True)

COLS = ["name", "src", "smiles", "mace_energy", "mace_epa", "mace_fmax",
        "mace_frms", "mace_nestd", "mace_nemin", "mace_nemax", "status"]


def load_corpus():
    rows = []
    with open(CORPUS, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((r["name"], r["src"], r["smiles"]))
    return rows


def done_names():
    if not os.path.exists(OUT):
        return set()
    s = set()
    with open(OUT, newline="") as f:
        for r in csv.DictReader(f):
            s.add(r["name"])
    return s


def conformer(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None, None
    m = Chem.AddHs(m)
    p = AllChem.ETKDGv3()
    p.randomSeed = 42
    if AllChem.EmbedMolecule(m, p) != 0:
        if AllChem.EmbedMolecule(m, useRandomCoords=True, randomSeed=42) != 0:
            return None, None
    try:
        AllChem.MMFFOptimizeMolecule(m, maxIters=500)
    except Exception:
        pass
    conf = m.GetConformer()
    coord = np.array([list(conf.GetAtomPosition(i)) for i in range(m.GetNumAtoms())], dtype=float)
    numbers = np.array([a.GetAtomicNum() for a in m.GetAtoms()], dtype=int)
    return coord, numbers


def featurize(calc, coord, numbers):
    atoms = Atoms(numbers=numbers, positions=coord)
    atoms.calc = calc
    e = float(atoms.get_potential_energy())
    f = atoms.get_forces().reshape(-1, 3)
    fn = np.linalg.norm(f, axis=1)
    na = len(numbers)
    ne = calc.results.get("node_energy", None)
    if ne is not None:
        ne = np.asarray(ne, dtype=float).ravel()
        nestd, nemin, nemax = float(ne.std()), float(ne.min()), float(ne.max())
    else:
        nestd = nemin = nemax = ""
    return dict(
        mace_energy=e, mace_epa=e / na,
        mace_fmax=float(fn.max()), mace_frms=float(np.sqrt((fn ** 2).mean())),
        mace_nestd=nestd, mace_nemin=nemin, mace_nemax=nemax,
    )


def main():
    rows = load_corpus()
    done = done_names()
    exists = os.path.exists(OUT)
    calc = mace_off(model="medium", device="cpu", default_dtype="float64")
    todo = [r for r in rows if r[0] not in done]
    print(f"corpus {len(rows)} | done {len(done)} | todo {len(todo)}", flush=True)
    f = open(OUT, "a", newline="")
    w = csv.DictWriter(f, fieldnames=COLS)
    if not exists:
        w.writeheader()
    ok = err = 0
    for i, (name, src, smi) in enumerate(todo):
        rec = {"name": name, "src": src, "smiles": smi, "status": "ok"}
        try:
            coord, numbers = conformer(smi)
            if coord is None:
                raise RuntimeError("conformer_fail")
            rec.update(featurize(calc, coord, numbers))
            ok += 1
        except Exception as e:
            rec["status"] = f"err:{str(e)[:40]}"
            for c in COLS[3:-1]:
                rec[c] = ""
            err += 1
        w.writerow(rec)
        if (i + 1) % 25 == 0:
            f.flush()
            print(f"  {i+1}/{len(todo)} ok={ok} err={err}", flush=True)
    f.flush(); f.close()
    print(f"DONE ok={ok} err={err}", flush=True)


if __name__ == "__main__":
    main()
