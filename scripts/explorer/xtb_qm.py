"""Compute GFN2-xTB semi-empirical QM descriptors for the PXR corpus (4139 train + 513 test).

PRIME physics axis (ledger ★ tblite GFN2-xTB + morfeus): orthogonal to the 2D D-MPNN / chempropembed
sink. Per molecule: RDKit ETKDGv3 conformer + MMFF opt -> GFN2-xTB single point (tblite) ->
HOMO/LUMO/gap/dipole/energy + Mulliken-charge stats; + (optional) morfeus global electrophilicity /
hardness / electrofugality / nucleofugality / SASA / dispersion.

CRITICAL: tblite expects coordinates in BOHR (RDKit gives Angstrom -> * 1.8897259886).

Resumable: writes one row per molecule to out CSV; on restart skips smiles already present.
Usage: python xtb_qm.py <corpus.csv> <out.csv> <n_slices> <slice_idx>
"""
import sys, os, time, math
import numpy as np
import pandas as pd

ANG2BOHR = 1.8897259886
HARTREE2EV = 27.211386245988
AU2DEBYE = 2.541746

corpus_path, out_path, n_slices, slice_idx = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

try:
    from tblite.interface import Calculator
    HAVE_TBLITE = True
except Exception as e:
    print("FATAL: tblite import failed:", e); sys.exit(2)

try:
    from morfeus import XTB as MorfeusXTB
    HAVE_MORF = True
except Exception as e:
    print("morfeus unavailable (optional):", e); HAVE_MORF = False


def conformer(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    p = AllChem.ETKDGv3(); p.randomSeed = 0xC0FFEE
    if AllChem.EmbedMolecule(mol, p) != 0:
        if AllChem.EmbedMolecule(mol, AllChem.ETKDGv2()) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=300)
    except Exception:
        pass
    conf = mol.GetConformer()
    nums = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
    pos = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                    for i in range(mol.GetNumAtoms())])
    chg = Chem.GetFormalCharge(mol)
    return nums, pos, chg


def xtb_feats(nums, pos_ang, chg):
    calc = Calculator("GFN2-xTB", nums, pos_ang * ANG2BOHR, charge=float(chg))
    calc.set("verbosity", 0)
    res = calc.singlepoint()
    energy = float(res.get("energy"))
    occ = np.asarray(res.get("orbital-occupations"))
    eps = np.asarray(res.get("orbital-energies"))  # Hartree
    occ_idx = np.where(occ > 0.5)[0]
    homo = float(eps[occ_idx[-1]]) * HARTREE2EV
    lumo = float(eps[occ_idx[-1] + 1]) * HARTREE2EV if occ_idx[-1] + 1 < len(eps) else np.nan
    gap = lumo - homo
    dip = float(np.linalg.norm(np.asarray(res.get("dipole")))) * AU2DEBYE
    q = np.asarray(res.get("charges"))
    return dict(xtb_energy=energy, xtb_homo=homo, xtb_lumo=lumo, xtb_gap=gap, xtb_dipole=dip,
                xtb_qmin=float(q.min()), xtb_qmax=float(q.max()), xtb_qabs_mean=float(np.abs(q).mean()),
                xtb_qstd=float(q.std()))


def morf_feats(nums, pos_ang):
    out = {}
    try:
        x = MorfeusXTB(nums, pos_ang, version="2")  # morfeus takes Angstrom
        out["mf_ip"] = float(x.get_ip())
        out["mf_ea"] = float(x.get_ea())
        out["mf_electrophilicity"] = float(x.get_global_descriptor("electrophilicity", corrected=True))
        out["mf_nucleophilicity"] = float(x.get_global_descriptor("nucleophilicity", corrected=True))
        out["mf_hardness"] = float(x.get_global_descriptor("hardness", corrected=True))
        dip = x.get_dipole()
        out["mf_dipole"] = float(np.linalg.norm(dip))
    except Exception:
        pass
    return out


df = pd.read_csv(corpus_path)
df = df.iloc[slice_idx::n_slices].reset_index(drop=True)

done = set()
if os.path.exists(out_path):
    try:
        done = set(pd.read_csv(out_path)["smiles"].tolist())
    except Exception:
        done = set()

cols = ["smiles", "name", "src", "xtb_energy", "xtb_homo", "xtb_lumo", "xtb_gap", "xtb_dipole",
        "xtb_qmin", "xtb_qmax", "xtb_qabs_mean", "xtb_qstd",
        "mf_ip", "mf_ea", "mf_electrophilicity", "mf_nucleophilicity", "mf_hardness", "mf_dipole"]
new = not os.path.exists(out_path)
f = open(out_path, "a")
if new:
    f.write(",".join(cols) + "\n"); f.flush()

t0 = time.time(); ok = 0; fail = 0
for i, r in df.iterrows():
    smi = r["smiles"]
    if smi in done:
        continue
    rec = {"smiles": smi, "name": r.get("name", ""), "src": r.get("src", "")}
    try:
        c = conformer(smi)
        if c is None:
            fail += 1; continue
        nums, pos, chg = c
        rec.update(xtb_feats(nums, pos, chg))
        if HAVE_MORF:
            rec.update(morf_feats(nums, pos))
        ok += 1
    except Exception:
        fail += 1
    f.write(",".join(str(rec.get(c, "")) for c in cols) + "\n"); f.flush()
    if (ok + fail) % 25 == 0:
        print(f"[slice {slice_idx}] {ok+fail}/{len(df)} ok={ok} fail={fail} {(time.time()-t0)/max(ok,1):.2f}s/mol", flush=True)
f.close()
print(f"SLICE_{slice_idx}_DONE ok={ok} fail={fail} wall={time.time()-t0:.0f}s")
