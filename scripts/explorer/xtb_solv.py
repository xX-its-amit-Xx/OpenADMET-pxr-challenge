"""GFN2-xTB IMPLICIT-SOLVATION (ALPB water) descriptors for the PXR corpus.

DISTINCT physics axis vs the deployed stack (AIMNet2 charges / D4 dispersion-
polarizability / MMFF strain / DBSTEP steric-shape) — ALL of which are gas-phase.
Solvation / hydrophobic-desolvation is the single most physically-motivated
observable still missing for a hydrophobic ligand-binding pocket (PXR).

Per molecule: RDKit ETKDGv3 + MMFF conformer -> two GFN2-xTB singlepoints
(gas + ALPB-water) -> distinct scalars:
  dG_solv (kcal/mol)            total electrostatic+nonpolar solvation free energy
  dip_gas, dip_solv, ddip       solvent-induced dipole POLARIZATION (distinct!)
  gap_gas, gap_solv, dgap       solvation-induced HOMO-LUMO shift
  qabs_gas, qabs_solv, dqabs    charge redistribution on solvation
  dE_per_heavy                  size-normalized solvation

CRITICAL: tblite coordinates in BOHR (RDKit Angstrom * 1.8897259886).

ROBUSTNESS: this Explorer tblite build HANGS on some ALPB singlepoints. Each
molecule's solvated singlepoint is guarded by signal.alarm(SOLV_TIMEOUT); on
hang the row is written with NaN solv columns (gas columns still valid) and the
worker proceeds. The working solvation API is auto-detected once at startup
(_pick_solv_api) over a small molecule with the same alarm guard.

Resumable: one row per molecule, skip smiles already present on restart.
Usage: python xtb_solv.py <corpus.csv> <out.csv> <n_slices> <slice_idx>
"""
import sys, os, time, signal
import numpy as np
import pandas as pd

ANG2BOHR = 1.8897259886
HARTREE2EV = 27.211386245988
HARTREE2KCAL = 627.5094740631
AU2DEBYE = 2.541746
SOLV_TIMEOUT = 25   # seconds per solvated singlepoint before skip

corpus_path, out_path, n_slices, slice_idx = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
from tblite.interface import Calculator


class _TO(Exception):
    pass


def _alarm(sig, frm):
    raise _TO()


signal.signal(signal.SIGALRM, _alarm)

# candidate solvation API forms (this tblite build differs across versions)
_SOLV_VARIANTS = [
    ("add", ("alpb-solvation", "water")),
    ("add", ("alpb-solvation", "water", "gfn2")),
    ("add", ("gbsa", "water")),
    ("add", ("gbsa", "water", "gfn2")),
    ("add", ("cpcm-solvation", 80.0)),
]


def _apply_solv(calc, api):
    fn, args = api
    getattr(calc, fn)(*args)


def _pick_solv_api():
    """Find a solvation API that runs (not hang/err) on a 3-atom probe. Returns api or None."""
    nums = np.array([8, 1, 1])
    pos = np.array([[0, 0, 0], [0, 0, 1.81], [1.71, 0, -0.4]], float) * ANG2BOHR
    for api in _SOLV_VARIANTS:
        try:
            signal.alarm(SOLV_TIMEOUT)
            c = Calculator("GFN2-xTB", nums, pos)
            c.set("verbosity", 0)
            _apply_solv(c, api)
            float(c.singlepoint().get("energy"))
            signal.alarm(0)
            print(f"SOLV_API_CHOSEN {api}", flush=True)
            return api
        except _TO:
            print(f"SOLV_API_HANG {api}", flush=True)
        except Exception as e:
            print(f"SOLV_API_ERR {api} {type(e).__name__} {str(e)[:60]}", flush=True)
        finally:
            signal.alarm(0)
    print("SOLV_API_NONE -- no working solvation model on this build", flush=True)
    return None


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
    n_heavy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    return nums, pos, chg, max(n_heavy, 1)


def _singlepoint(nums, pos_ang, chg, solv_api=None):
    calc = Calculator("GFN2-xTB", nums, pos_ang * ANG2BOHR, charge=float(chg))
    calc.set("verbosity", 0)
    if solv_api is not None:
        _apply_solv(calc, solv_api)
    res = calc.singlepoint()
    energy = float(res.get("energy"))
    occ = np.asarray(res.get("orbital-occupations"))
    eps = np.asarray(res.get("orbital-energies"))
    oi = np.where(occ > 0.5)[0]
    homo = float(eps[oi[-1]]) * HARTREE2EV
    lumo = float(eps[oi[-1] + 1]) * HARTREE2EV if oi[-1] + 1 < len(eps) else np.nan
    gap = lumo - homo
    dip = float(np.linalg.norm(np.asarray(res.get("dipole")))) * AU2DEBYE
    q = np.asarray(res.get("charges"))
    return energy, gap, dip, float(np.abs(q).mean())


COLS = ["smiles", "name", "src",
        "solv_dG_kcal", "solv_dE_per_heavy",
        "dip_gas", "dip_solv", "solv_ddip",
        "gap_gas", "gap_solv", "solv_dgap",
        "qabs_gas", "qabs_solv", "solv_dqabs"]


def main():
    solv_api = _pick_solv_api()
    df = pd.read_csv(corpus_path).iloc[slice_idx::n_slices].reset_index(drop=True)
    done = set()
    if os.path.exists(out_path):
        try:
            done = set(pd.read_csv(out_path)["smiles"].tolist())
        except Exception:
            pass
    new = not os.path.exists(out_path)
    f = open(out_path, "a")
    if new:
        f.write(",".join(COLS) + "\n"); f.flush()
    t0 = time.time(); ok = 0; fail = 0
    for _, r in df.iterrows():
        smi = r["smiles"]
        if smi in done:
            continue
        rec = {"smiles": smi, "name": r.get("name", ""), "src": r.get("src", "")}
        try:
            c = conformer(smi)
            if c is None:
                fail += 1
                f.write(",".join(str(rec.get(k, "")) for k in COLS) + "\n"); f.flush(); continue
            nums, pos, chg, nh = c
            e_g, gap_g, dip_g, q_g = _singlepoint(nums, pos, chg, None)
            rec.update(dip_gas=dip_g, gap_gas=gap_g, qabs_gas=q_g)
            if solv_api is not None:
                try:
                    signal.alarm(SOLV_TIMEOUT)
                    e_s, gap_s, dip_s, q_s = _singlepoint(nums, pos, chg, solv_api)
                    signal.alarm(0)
                    rec.update(
                        solv_dG_kcal=(e_s - e_g) * HARTREE2KCAL,
                        solv_dE_per_heavy=(e_s - e_g) * HARTREE2KCAL / nh,
                        dip_solv=dip_s, solv_ddip=dip_s - dip_g,
                        gap_solv=gap_s, solv_dgap=gap_s - gap_g,
                        qabs_solv=q_s, solv_dqabs=q_s - q_g)
                except _TO:
                    pass
                finally:
                    signal.alarm(0)
            ok += 1
        except Exception:
            fail += 1
        f.write(",".join(str(rec.get(k, "")) for k in COLS) + "\n"); f.flush()
        if (ok + fail) % 25 == 0:
            print(f"[slice {slice_idx}] {ok+fail}/{len(df)} ok={ok} fail={fail} "
                  f"{(time.time()-t0)/max(ok,1):.2f}s/mol", flush=True)
    f.close()
    print(f"SOLV_SLICE_{slice_idx}_DONE ok={ok} fail={fail} wall={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
