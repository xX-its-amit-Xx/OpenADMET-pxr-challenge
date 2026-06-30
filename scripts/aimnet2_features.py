"""AIMNet2 QM/physics scalar features for the PXR activity corpus.

Reuses the existing standardized corpus C:/pxr_work/xtb/corpus.csv (4139 train + 513 test).
Per mol: ETKDG conformer + MMFF opt -> AIMNet2Calculator (DFT-trained NNP) ->
  energy, per-atom partial charges, forces.
Derived QM scalars (axis comparable to the xTB GFN2 9-feature block):
  aimnet_energy, aimnet_qmin, aimnet_qmax, aimnet_qabs_mean, aimnet_qstd,
  aimnet_qsum_abs, aimnet_dipole (|sum q_i r_i|, e*Ang), aimnet_fmax, aimnet_frms.

Resumable: appends per row; on restart skips names already in the output CSV.
Run (isolated venv):
  TORCHDYNAMO_DISABLE=1 C:/aimnet_venv/Scripts/python.exe scripts/aimnet2_features.py
"""
import os, sys, csv
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch
try:
    import torch._dynamo as _d; _d.config.suppress_errors = True
except Exception:
    pass
from rdkit import Chem
from rdkit.Chem import AllChem
from aimnet.calculators import AIMNet2Calculator

CORPUS = "C:/pxr_work/xtb/corpus.csv"
OUTDIR = "C:/pxr_work/aimnet2"
OUT = os.path.join(OUTDIR, "aimnet_features.csv")
os.makedirs(OUTDIR, exist_ok=True)

COLS = ["name", "src", "smiles", "aimnet_energy", "aimnet_qmin", "aimnet_qmax",
        "aimnet_qabs_mean", "aimnet_qstd", "aimnet_qsum_abs", "aimnet_dipole",
        "aimnet_fmax", "aimnet_frms", "status"]


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
        return None, None, None
    m = Chem.AddHs(m)
    p = AllChem.ETKDGv3()
    p.randomSeed = 42
    if AllChem.EmbedMolecule(m, p) != 0:
        # fallback: random coords
        if AllChem.EmbedMolecule(m, useRandomCoords=True, randomSeed=42) != 0:
            return None, None, None
    try:
        AllChem.MMFFOptimizeMolecule(m, maxIters=500)
    except Exception:
        pass
    conf = m.GetConformer()
    coord = np.array([list(conf.GetAtomPosition(i)) for i in range(m.GetNumAtoms())], dtype=float)
    numbers = np.array([a.GetAtomicNum() for a in m.GetAtoms()], dtype=int)
    charge = Chem.GetFormalCharge(m)
    return coord, numbers, charge


def featurize(calc, coord, numbers, charge):
    out = calc({"coord": coord, "numbers": numbers, "charge": int(charge)}, forces=True)
    q = out["charges"].detach().cpu().numpy().ravel().astype(float)
    e = float(out["energy"].detach().cpu().numpy().ravel()[0])
    f = out["forces"].detach().cpu().numpy().reshape(-1, 3).astype(float)
    fn = np.linalg.norm(f, axis=1)
    # dipole from partial charges (origin-independent only for neutral; report magnitude)
    dip = float(np.linalg.norm((q[:, None] * coord).sum(axis=0)))
    return dict(
        aimnet_energy=e,
        aimnet_qmin=float(q.min()), aimnet_qmax=float(q.max()),
        aimnet_qabs_mean=float(np.abs(q).mean()), aimnet_qstd=float(q.std()),
        aimnet_qsum_abs=float(np.abs(q).sum()), aimnet_dipole=dip,
        aimnet_fmax=float(fn.max()), aimnet_frms=float(np.sqrt((fn**2).mean())),
    )


def main():
    rows = load_corpus()
    done = done_names()
    new = os.path.exists(OUT)
    calc = AIMNet2Calculator("aimnet2", device="cpu")
    todo = [r for r in rows if r[0] not in done]
    print(f"corpus {len(rows)} | done {len(done)} | todo {len(todo)}", flush=True)
    f = open(OUT, "a", newline="")
    w = csv.DictWriter(f, fieldnames=COLS)
    if not new:
        w.writeheader()
    ok = err = 0
    for i, (name, src, smi) in enumerate(todo):
        rec = {"name": name, "src": src, "smiles": smi, "status": "ok"}
        try:
            coord, numbers, charge = conformer(smi)
            if coord is None:
                raise RuntimeError("conformer_fail")
            rec.update(featurize(calc, coord, numbers, charge))  # forces need autograd
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
