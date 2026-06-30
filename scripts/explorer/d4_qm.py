"""Compute DFT-D4 (Grimme/Caldeweyher) dispersion + polarizability descriptors for the PXR corpus.

DISTINCT physics observable not covered by the existing axes:
  - xTB run (xtb_qm.py): HOMO/LUMO/gap/dipole/Mulliken charges  (NO polarizability)
  - AIMNet2 (WIN -0.0070): learned-NNP energy / NNP charges / dipole / forces
  - DFT-D4 here: charge/CN-dependent dynamic atomic POLARIZABILITIES + C6 dispersion
    coefficients + D4 dispersion energy + EEQ electronegativity-equilibration charges.

Gate like rich-z: marginal-over-deployed-anchor on the matched nb1177-style honest gate,
NOT standalone RAE. Scalars only (no per-atom embeddings).

dftd4 wants coordinates in BOHR (RDKit gives Angstrom -> * 1.8897259886).
Resumable: one row per molecule; on restart skips smiles already present.
Usage: python d4_qm.py <corpus.csv> <out.csv> <n_slices> <slice_idx>
"""
import sys, os, time
import numpy as np
import pandas as pd

ANG2BOHR = 1.8897259886

corpus_path, out_path, n_slices, slice_idx = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

try:
    from dftd4.interface import DampingParam, DispersionModel
except Exception as e:
    print("FATAL: dftd4 import failed:", e); sys.exit(2)


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


def d4_feats(nums, pos_ang, chg):
    model = DispersionModel(nums, pos_ang * ANG2BOHR, charge=float(chg))
    prop = model.get_properties()
    # property keys: 'partial charges', 'coordination numbers',
    #                'c6 coefficients', 'polarizabilities'
    alpha = np.asarray(prop.get("polarizabilities"), dtype=float).ravel()   # per-atom static alpha (au)
    cn = np.asarray(prop.get("coordination numbers"), dtype=float).ravel()
    qeeq = np.asarray(prop.get("partial charges"), dtype=float).ravel()
    c6 = np.asarray(prop.get("c6 coefficients"), dtype=float)               # NxN matrix
    c6diag = np.diag(c6) if c6.ndim == 2 else c6.ravel()
    # D4 dispersion energy with a fixed functional damping (PBE) -> descriptor scale only
    res = model.get_dispersion(DampingParam(method="pbe"), grad=False)
    edisp = float(res.get("energy"))
    n = max(len(alpha), 1)
    return dict(
        d4_alpha_sum=float(alpha.sum()),
        d4_alpha_mean=float(alpha.mean()),
        d4_alpha_std=float(alpha.std()),
        d4_alpha_max=float(alpha.max()),
        d4_c6diag_mean=float(c6diag.mean()),
        d4_c6diag_std=float(c6diag.std()),
        d4_c6_total=float(c6.sum()) if c6.ndim == 2 else float(c6diag.sum()),
        d4_edisp=edisp,
        d4_edisp_per_atom=edisp / n,
        d4_cn_mean=float(cn.mean()),
        d4_cn_max=float(cn.max()),
        d4_qeeq_min=float(qeeq.min()),
        d4_qeeq_max=float(qeeq.max()),
        d4_qeeq_std=float(qeeq.std()),
        d4_qeeq_absum=float(np.abs(qeeq).sum()),
    )


df = pd.read_csv(corpus_path)
df = df.iloc[slice_idx::n_slices].reset_index(drop=True)

done = set()
if os.path.exists(out_path):
    try:
        done = set(pd.read_csv(out_path)["smiles"].tolist())
    except Exception:
        done = set()

cols = ["smiles", "name", "src",
        "d4_alpha_sum", "d4_alpha_mean", "d4_alpha_std", "d4_alpha_max",
        "d4_c6diag_mean", "d4_c6diag_std", "d4_c6_total",
        "d4_edisp", "d4_edisp_per_atom", "d4_cn_mean", "d4_cn_max",
        "d4_qeeq_min", "d4_qeeq_max", "d4_qeeq_std", "d4_qeeq_absum"]
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
            fail += 1
            f.write(",".join(str(rec.get(col, "")) for col in cols) + "\n"); f.flush()
            continue
        nums, pos, chg = c
        rec.update(d4_feats(nums, pos, chg))
        ok += 1
    except Exception:
        fail += 1
    f.write(",".join(str(rec.get(col, "")) for col in cols) + "\n"); f.flush()
    if (ok + fail) % 50 == 0:
        print(f"[slice {slice_idx}] {ok+fail}/{len(df)} ok={ok} fail={fail} {(time.time()-t0)/max(ok,1):.2f}s/mol", flush=True)
f.close()
print(f"D4_SLICE_{slice_idx}_DONE ok={ok} fail={fail} wall={time.time()-t0:.0f}s")
