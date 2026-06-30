"""MMFF conformational STRAIN / flexibility scalars for the PXR activity corpus.

Reuses the standardized corpus C:/pxr_work/xtb/corpus.csv (4139 train + 513 test).
This is a DISTINCT physical observable from AIMNet2's electronic scalars
(charges/dipole/forces/energy at one geometry): here we sample a conformer
ENSEMBLE and measure strain-release + conformational spread + shape diversity.
Memory flag (feedback_mace_off_redundant_with_aimnet2): strain (StrainRelief)
is one of the few observables NOT covered by the deployed AIMNet2 QM block.

Per mol: embed K ETKDGv3 conformers, MMFF94 optimize each, collect pre/post
energies + pairwise heavy-atom RMSD. Derived 8 size-aware scalars:
  strain_relax_mean, strain_relax_max   (E_embed - E_opt, strain released)
  conf_espread, conf_erange             (std / range of optimized energies)
  conf_n                                (# conformers kept)
  rmsd_mean, rmsd_max                   (heavy-atom shape diversity)
  e_per_heavy                           (min opt energy / n_heavy, strain density)

Resumable: appends per row; on restart skips names already in the output CSV.
Run:  .venv/Scripts/python.exe scripts/strain_features.py
"""
import os, csv
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

CORPUS = "C:/pxr_work/xtb/corpus.csv"
OUTDIR = "C:/pxr_work/strain"
OUT = os.path.join(OUTDIR, "strain_features.csv")
os.makedirs(OUTDIR, exist_ok=True)
K = 12  # conformers per molecule

COLS = ["name", "src", "smiles", "strain_relax_mean", "strain_relax_max",
        "conf_espread", "conf_erange", "conf_n", "rmsd_mean", "rmsd_max",
        "e_per_heavy", "status"]


def load_corpus():
    with open(CORPUS, newline="") as f:
        return [(r["name"], r["src"], r["smiles"]) for r in csv.DictReader(f)]


def done_names():
    if not os.path.exists(OUT):
        return set()
    with open(OUT, newline="") as f:
        return {r["name"] for r in csv.DictReader(f)}


def featurize(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        raise RuntimeError("parse_fail")
    n_heavy = m.GetNumHeavyAtoms()
    m = Chem.AddHs(m)
    p = AllChem.ETKDGv3(); p.randomSeed = 42; p.numThreads = 0
    cids = list(AllChem.EmbedMultipleConfs(m, numConfs=K, params=p))
    if not cids:
        p.useRandomCoords = True
        cids = list(AllChem.EmbedMultipleConfs(m, numConfs=K, params=p))
    if not cids:
        raise RuntimeError("embed_fail")
    e_embed, e_opt = [], []
    props = AllChem.MMFFGetMoleculeProperties(m)
    for cid in cids:
        ff0 = AllChem.MMFFGetMoleculeForceField(m, props, confId=cid) if props else None
        e0 = ff0.CalcEnergy() if ff0 is not None else np.nan
        res = AllChem.MMFFOptimizeMolecule(m, confId=cid, maxIters=1000)
        ff1 = AllChem.MMFFGetMoleculeForceField(m, props, confId=cid) if props else None
        e1 = ff1.CalcEnergy() if ff1 is not None else np.nan
        if np.isfinite(e0) and np.isfinite(e1):
            e_embed.append(e0); e_opt.append(e1)
    if not e_opt:
        raise RuntimeError("mmff_fail")
    e_embed = np.array(e_embed); e_opt = np.array(e_opt)
    relax = e_embed - e_opt
    # heavy-atom pairwise RMSD across optimized conformers
    rmsds = []
    mh = Chem.RemoveHs(m)
    cids_h = list(mh.GetConformers())
    ids = [c.GetId() for c in cids_h]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            try:
                rmsds.append(AllChem.GetConformerRMS(mh, ids[i], ids[j], prealigned=False))
            except Exception:
                pass
    rmsds = np.array(rmsds) if rmsds else np.array([0.0])
    return dict(
        strain_relax_mean=float(relax.mean()), strain_relax_max=float(relax.max()),
        conf_espread=float(e_opt.std()), conf_erange=float(e_opt.max() - e_opt.min()),
        conf_n=int(len(e_opt)), rmsd_mean=float(rmsds.mean()), rmsd_max=float(rmsds.max()),
        e_per_heavy=float(e_opt.min() / max(n_heavy, 1)),
    )


def main():
    rows = load_corpus(); done = done_names(); existed = os.path.exists(OUT)
    todo = [r for r in rows if r[0] not in done]
    print(f"corpus {len(rows)} | done {len(done)} | todo {len(todo)}", flush=True)
    f = open(OUT, "a", newline=""); w = csv.DictWriter(f, fieldnames=COLS)
    if not existed:
        w.writeheader()
    ok = err = 0
    for i, (name, src, smi) in enumerate(todo):
        rec = {"name": name, "src": src, "smiles": smi, "status": "ok"}
        try:
            rec.update(featurize(smi)); ok += 1
        except Exception as e:
            rec["status"] = f"err:{str(e)[:40]}"
            for c in COLS[3:-1]:
                rec[c] = ""
            err += 1
        w.writerow(rec)
        if (i + 1) % 50 == 0:
            f.flush(); print(f"  {i+1}/{len(todo)} ok={ok} err={err}", flush=True)
    f.flush(); f.close()
    print(f"DONE ok={ok} err={err}", flush=True)


if __name__ == "__main__":
    main()
