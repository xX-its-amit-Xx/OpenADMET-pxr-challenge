"""ANI-2x AEV featurizer (torchani 2.8.2).

ANI-2x Behler-Parrinello atomic-environment-vector (AEV, 1008-d) mean+std pooled per
molecule (2016-d) + ANI-2x total-energy scalars, for the 4139 train + 513 test.
ANI-2x supports H,C,N,O,F,S,Cl only; molecules with any other heavy atom are flagged
supported=0 with NaN features (imputed downstream like the other physics blocks).
3D: standardize -> AddHs -> ETKDGv3(seed 0xF00D) -> MMFF94 optimize.

Output npz: C:/pxr_work/ani2x/ani2x_features.npz
  names(str), src(str), supported(int8), n_heavy(int16),
  ani_e(float32), ani_e_per_heavy(float32), X(float32, N x 2016)  [aevmean|aevstd]
Checkpoints every 500 mols; resumes from checkpoint.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torchani
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")
from src.pxr.data import load_train, load_test
from src.pxr.chem import standardize

OUT = "C:/pxr_work/ani2x"; os.makedirs(OUT, exist_ok=True)
NPZ = f"{OUT}/ani2x_features.npz"
DIM = 2016
ANI_Z = {1, 6, 7, 8, 9, 16, 17}
torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
model = torchani.models.ANI2x(periodic_table_index=True)


def embed3d(mol):
    mol = Chem.AddHs(mol)
    p = AllChem.ETKDGv3(); p.randomSeed = 0xF00D
    if AllChem.EmbedMolecule(mol, p) != 0:
        if AllChem.EmbedMolecule(mol, AllChem.ETKDGv2()) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=400)
    except Exception:
        pass
    return mol


def featurize(smi):
    """returns (supported, n_heavy, ani_e, ani_e_per_heavy, feat[2016] or None)"""
    m = standardize(smi)
    if m is None:
        return 0, 0, np.nan, np.nan, None
    zs = [a.GetAtomicNum() for a in m.GetAtoms()]
    n_heavy = sum(1 for z in zs if z > 1)
    if not all(z in ANI_Z for z in zs):
        return 0, n_heavy, np.nan, np.nan, None
    m3 = embed3d(m)
    if m3 is None:
        return 0, n_heavy, np.nan, np.nan, None
    conf = m3.GetConformer()
    Z = torch.tensor([[a.GetAtomicNum() for a in m3.GetAtoms()]])
    xyz = torch.tensor([conf.GetPositions()], dtype=torch.float32)
    with torch.no_grad():
        ei, coords = model.species_converter((Z, xyz))
        aev = model.aev_computer((ei, coords)).aevs[0].numpy()
        e = float(model((Z, xyz)).energies)
    feat = np.concatenate([aev.mean(0), aev.std(0)]).astype(np.float32)
    return 1, n_heavy, np.float32(e), np.float32(e / max(n_heavy, 1)), feat


def main():
    tr = load_train().dropna(subset=["pec50"]).drop_duplicates("name")
    te = load_test().drop_duplicates("name")
    jobs = [(n, s, "train") for n, s in zip(tr["name"], tr["smiles"])] + \
           [(n, s, "test") for n, s in zip(te["name"], te["smiles"])]
    N = len(jobs)
    names = np.array([j[0] for j in jobs], dtype=object)
    srcs = np.array([j[2] for j in jobs], dtype=object)
    supported = np.zeros(N, np.int8); n_heavy = np.zeros(N, np.int16)
    ani_e = np.full(N, np.nan, np.float32); ani_eph = np.full(N, np.nan, np.float32)
    X = np.full((N, DIM), np.nan, np.float32)
    start = 0
    if os.path.exists(NPZ):
        z = np.load(NPZ, allow_pickle=True)
        done = int(z["n_done"]) if "n_done" in z else 0
        supported[:done] = z["supported"][:done]; n_heavy[:done] = z["n_heavy"][:done]
        ani_e[:done] = z["ani_e"][:done]; ani_eph[:done] = z["ani_e_per_heavy"][:done]
        X[:done] = z["X"][:done]; start = done
        print(f"resume from {start}", flush=True)

    def save(done):
        np.savez_compressed(NPZ, names=names, src=srcs, supported=supported,
                            n_heavy=n_heavy, ani_e=ani_e, ani_e_per_heavy=ani_eph,
                            X=X, n_done=done)

    n_ok = int(supported[:start].sum())
    for i in range(start, N):
        sup, nh, e, eph, feat = featurize(jobs[i][1])
        supported[i] = sup; n_heavy[i] = nh; ani_e[i] = e; ani_eph[i] = eph
        if feat is not None:
            X[i] = feat; n_ok += 1
        if (i + 1) % 500 == 0:
            save(i + 1)
            print(f"  {i+1}/{N} ok={n_ok}", flush=True)
    save(N)
    print(f"DONE {N} mols, supported/ok={n_ok}, unsupported={N-int(supported.sum())} -> {NPZ}", flush=True)


if __name__ == "__main__":
    main()
