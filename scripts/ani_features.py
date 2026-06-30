"""TorchANI / ANI-2x AEV (fixed Behler-Parrinello symmetry-function) descriptor
+ free DFT-quality NNP energy, for the PXR activity 4139 train + 513 test.

Distinct-axis physics probe (ledger cycle-339, niche-b). ANI's representation is
a FIXED symmetry-function basis (radial+angular Gaussians over local geometry),
NOT a learned message-passing embedding -> argued lowest sink-absorption risk of
the MLIP family. Reads the shared corpus C:/pxr_work/xtb/corpus.csv (name,src,smiles).

Per mol: standardize -> RDKit ETKDG conformer + MMFF -> ANI2x species_converter ->
aev_computer -> mean-pool the 1008-d per-atom AEV; also the total NNP energy +
energy-per-atom. ANI-2x supports H,C,N,O,S,F,Cl; mols with other elements are
flagged unsupported (imputed downstream). Caches incrementally (resumable) to
C:/pxr_work/ani/ani_raw.npz.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "C:/pxr_work/torchani_pkgs")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, torch, torchani
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
CORPUS = "C:/pxr_work/xtb/corpus.csv"
OUT = "C:/pxr_work/ani"; os.makedirs(OUT, exist_ok=True)
RAW = f"{OUT}/ani_raw.npz"
ALLOWED = {1, 6, 7, 8, 16, 9, 17}  # H C N O S F Cl
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))


def embed(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    m = Chem.AddHs(m)
    p = AllChem.ETKDGv3(); p.randomSeed = 1; p.maxIterations = 200
    if AllChem.EmbedMolecule(m, p) != 0:
        p2 = AllChem.ETKDGv3(); p2.randomSeed = 7; p2.useRandomCoords = True; p2.maxIterations = 400
        if AllChem.EmbedMolecule(m, p2) != 0:
            return None
    try: AllChem.MMFFOptimizeMolecule(m, maxIters=500)
    except Exception: pass
    return m


def main():
    df = pd.read_csv(CORPUS)
    cache = {}
    if os.path.exists(RAW):
        z = np.load(RAW, allow_pickle=True)
        names = z["names"].tolist(); aevs = z["aevs"]; en = z["energy"]; epa = z["epa"]; uns = z["unsupported"]
        for i, n in enumerate(names):
            cache[n] = (aevs[i], float(en[i]), float(epa[i]), bool(uns[i]))
        print(f"resumed {len(cache)} cached")

    model = torchani.models.ANI2x(periodic_table_index=True)
    todo = [r for r in df.itertuples() if r.name not in cache]
    print(f"corpus {len(df)} | cached {len(cache)} | todo {len(todo)}")
    AEVD = 1008
    for k, r in enumerate(todo):
        aev = np.zeros(AEVD, np.float32); energy = epa_v = np.nan; unsupported = True
        try:
            mol = embed(r.smiles)
            if mol is not None:
                Z = [a.GetAtomicNum() for a in mol.GetAtoms()]
                if set(Z) <= ALLOWED:
                    unsupported = False
                    nat = len(Z)
                    Zt = torch.tensor([Z])
                    xyz = torch.tensor([mol.GetConformer().GetPositions()], dtype=torch.float32)
                    sp, co = model.species_converter((Zt, xyz))
                    a = model.aev_computer((sp, co))
                    a = a.aevs if hasattr(a, "aevs") else a[1]
                    aev = a[0].mean(0).detach().numpy().astype(np.float32)
                    e = float(model((Zt, xyz)).energies)
                    energy = e; epa_v = e / max(nat, 1)
        except Exception as ex:
            if k < 5: print("  err", r.name, repr(ex)[:80])
        cache[r.name] = (aev, energy, epa_v, unsupported)
        if (k + 1) % 200 == 0 or k == len(todo) - 1:
            ns = list(cache.keys())
            np.savez_compressed(RAW,
                names=np.array(ns),
                aevs=np.stack([cache[n][0] for n in ns]),
                energy=np.array([cache[n][1] for n in ns]),
                epa=np.array([cache[n][2] for n in ns]),
                unsupported=np.array([cache[n][3] for n in ns]))
            print(f"  {k+1}/{len(todo)} saved (total {len(cache)})", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
