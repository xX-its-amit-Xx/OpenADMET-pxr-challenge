"""MACE-POLAR-1 polarisable-electrostatic QM scalars for the PXR activity corpus.

Sibling of scripts/aimnet2_features.py + scripts/mace_off_features.py. AIMNet2 (DFT-trained NNP
ground-state energy/charge scalars) is DEPLOYED (nb1177, best 0.4297). MACE-OFF (a 2nd ground-state
MLIP) was REDUNDANT with it (nb1178, +0.0001). Memory feedback_mace_off_redundant_with_aimnet2 says
the ONLY DISTINCT observables left worth gating are alpha-polarisability / long-range electrostatics
(MACE-POLAR-1), vibrational, e-density, strain. This is the MACE-POLAR-1 probe.

MACE-POLAR-1 (arXiv 2602.19411, ACEsuit) is a POLARISABLE-ELECTROSTATIC foundation model: it carries
an explicit long-range electrostatic/charge-density head (graph_electrostatics / graph_longrange).
Its results expose energy terms AIMNet2 has NO analogue for -> a genuinely orthogonal QM axis, not
another ground-state energy. We reuse the SAME ETKDGv3+MMFF conformer (C:/pxr_work/xtb/corpus.csv)
so the comparison vs AIMNet2/MACE-OFF is on identical geometries.

Derived DISTINCT QM scalars (long-range electrostatics + electron-density distribution):
  mp_estat      electrostatic_energy (eV)   -- long-range electrostatic energy (no AIMNet2 analogue)
  mp_eelec      electron_energy (eV)        -- electronic-structure energy term
  mp_eint       interaction_energy (eV)     -- many-body interaction energy
  mp_dipole     |dipole| (e*Ang)            -- polar dipole magnitude (field-response model)
  mp_dc_std     std of atom-centered density_coefficients -- e-density distribution heterogeneity
  mp_dc_absmn   mean |density_coefficients|              -- e-density spread
  mp_spinabs    mean |spin| (spin-charge separation)
  mp_qstd       std of partial charges (model's own charge eq, distinct functional)

Resumable: appends per row; on restart skips names already in the output CSV.
Run (isolated venv, mace-torch + graph_longrange installed):
  TORCHDYNAMO_DISABLE=1 C:/aimnet_venv/Scripts/python.exe scripts/mace_polar_features.py
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
from mace.calculators import mace_polar

CORPUS = "C:/pxr_work/xtb/corpus.csv"
OUTDIR = "C:/pxr_work/mace_polar"
OUT = os.path.join(OUTDIR, "mace_polar_features.csv")
os.makedirs(OUTDIR, exist_ok=True)

COLS = ["name", "src", "smiles", "mp_estat", "mp_eelec", "mp_eint", "mp_dipole",
        "mp_dc_std", "mp_dc_absmn", "mp_spinabs", "mp_qstd", "status"]


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


def _scalar(res, key):
    v = res.get(key, None)
    if v is None:
        return ""
    return float(np.asarray(v, dtype=float).ravel()[0])


def featurize(calc, coord, numbers):
    atoms = Atoms(numbers=numbers, positions=coord)
    atoms.calc = calc
    atoms.get_potential_energy()  # triggers full forward; populates calc.results
    r = calc.results
    dip = np.asarray(r.get("dipole", [0, 0, 0]), dtype=float).ravel()
    dc = r.get("density_coefficients", None)
    if dc is not None:
        dc = np.asarray(dc, dtype=float).ravel()
        dc_std, dc_absmn = float(dc.std()), float(np.abs(dc).mean())
    else:
        dc_std = dc_absmn = ""
    sp = r.get("spins", None)
    spinabs = float(np.abs(np.asarray(sp, dtype=float)).mean()) if sp is not None else ""
    q = r.get("charges", None)
    qstd = float(np.asarray(q, dtype=float).std()) if q is not None else ""
    return dict(
        mp_estat=_scalar(r, "electrostatic_energy"),
        mp_eelec=_scalar(r, "electron_energy"),
        mp_eint=_scalar(r, "interaction_energy"),
        mp_dipole=float(np.linalg.norm(dip)),
        mp_dc_std=dc_std, mp_dc_absmn=dc_absmn,
        mp_spinabs=spinabs, mp_qstd=qstd,
    )


def main():
    rows = load_corpus()
    done = done_names()
    exists = os.path.exists(OUT)
    calc = mace_polar(model="polar-1-m", device="cpu", default_dtype="float64")
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
