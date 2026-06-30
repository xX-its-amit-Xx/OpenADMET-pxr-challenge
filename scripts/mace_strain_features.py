"""
StrainRelief-style DFT-accurate strain via MACE-OFF23.
Computes: mace_strain = E(MMFF_pose) - E(MMFF_global_min)
where energies are MACE-OFF23 single-points (fast; no MACE geometry opt).
Writes: /scratch/shenoy.am/pxr_work/mace_strain/mace_strain_features.csv
"""
import os, sys, json, argparse, time, traceback
import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR = Path("/scratch/shenoy.am/pxr_work/mace_strain")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE = OUT_DIR / "mace_strain_features.csv"
ERR_LOG = OUT_DIR / "errors.txt"

N_CONFS = 20   # conformers for global-min search
SEED = 42

def smiles_to_mace_energy(smiles, calc, n_confs=N_CONFS):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from ase import Atoms
    import torch

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None

    mol = Chem.AddHs(mol)

    # Generate reference conformer (seed=42) + MMFF minimize
    params = AllChem.EmbedParameters()
    params.randomSeed = SEED
    params.useSmallRingTorsions = True
    if AllChem.EmbedMolecule(mol, params) != 0:
        # fallback random embed
        AllChem.EmbedMolecule(mol, randomSeed=SEED)
    if mol.GetNumConformers() == 0:
        return None, None, None
    AllChem.MMFFOptimizeMolecule(mol, confId=0)

    def conf_to_ase(mol, conf_id):
        conf = mol.GetConformer(conf_id)
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        positions = conf.GetPositions()
        return Atoms(symbols=symbols, positions=positions)

    def mace_energy(atoms):
        atoms.calc = calc
        try:
            return atoms.get_potential_energy()
        except Exception:
            return None

    # Energy of reference pose
    ref_atoms = conf_to_ase(mol, 0)
    e_pose = mace_energy(ref_atoms)
    if e_pose is None:
        return None, None, None

    # Generate N conformers for global-min search
    mol_c = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params2 = AllChem.ETKDGv3()
    params2.randomSeed = SEED
    params2.numThreads = 0
    n_gen = AllChem.EmbedMultipleConfs(mol_c, numConfs=n_confs, params=params2)
    if n_gen == 0:
        return e_pose, e_pose, 0.0

    # MMFF minimize all, then MACE single-point energies
    res = AllChem.MMFFOptimizeMoleculeConfs(mol_c, numThreads=0)
    e_all = []
    for cid in range(n_gen):
        atoms = conf_to_ase(mol_c, cid)
        e = mace_energy(atoms)
        if e is not None:
            e_all.append(e)

    if not e_all:
        return e_pose, e_pose, 0.0

    e_globalmin = min(e_all)
    mace_strain = e_pose - e_globalmin
    return e_pose, e_globalmin, mace_strain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/scratch/shenoy.am/pxr_data", help="dir with train/test CSVs")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--model", default="MACE-OFF23(S)", help="MACE-OFF model size")
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--chunk_id", type=int, default=0)
    args = parser.parse_args()

    # Load SMILES
    data_dir = Path(args.data_dir)
    # Try to find train/test CSVs
    train_files = list(data_dir.glob("*TRAIN*.csv")) + list(data_dir.glob("*train*.csv"))
    test_files = list(data_dir.glob("*TEST*.csv")) + list(data_dir.glob("*test*.csv"))

    if not train_files:
        # Try the blinded challenge format
        train_files = list(data_dir.glob("*.csv"))

    print(f"Found train files: {[f.name for f in train_files[:3]]}")
    print(f"Found test files: {[f.name for f in test_files[:3]]}")

    smiles_list = []

    for f in train_files:
        df = pd.read_csv(f)
        smi_col = next((c for c in df.columns if 'smiles' in c.lower() or 'SMILES' in c), None)
        if smi_col:
            smiles_list.extend(df[smi_col].tolist())

    for f in test_files:
        df = pd.read_csv(f)
        smi_col = next((c for c in df.columns if 'smiles' in c.lower() or 'SMILES' in c), None)
        if smi_col:
            smiles_list.extend(df[smi_col].tolist())

    # Deduplicate
    smiles_list = list(dict.fromkeys([s for s in smiles_list if isinstance(s, str) and s]))
    print(f"Total unique SMILES: {len(smiles_list)}")

    # Chunk for parallel jobs
    chunks = [smiles_list[i:i+args.chunk_size] for i in range(0, len(smiles_list), args.chunk_size)]
    if args.chunk_id >= len(chunks):
        print(f"Chunk {args.chunk_id} out of range ({len(chunks)} total chunks). Done.")
        return

    chunk = chunks[args.chunk_id]
    out_file = OUT_DIR / f"chunk_{args.chunk_id:04d}.csv"

    # Skip if already done
    if out_file.exists():
        df_existing = pd.read_csv(out_file)
        if len(df_existing) == len(chunk):
            print(f"Chunk {args.chunk_id} already done ({len(df_existing)} rows). Skip.")
            return

    print(f"Processing chunk {args.chunk_id}: {len(chunk)} SMILES on {args.device}")

    # Load MACE-OFF
    from mace.calculators import mace_off
    calc = mace_off(model=args.model, device=args.device)
    print(f"MACE-OFF23 loaded: {args.model}")

    results = []
    errs = 0
    for i, smi in enumerate(chunk):
        t0 = time.time()
        try:
            e_pose, e_min, strain = smiles_to_mace_energy(smi, calc)
            results.append({
                "smiles": smi,
                "mace_e_pose": e_pose,
                "mace_e_globalmin": e_min,
                "mace_strain": strain,
            })
        except Exception as e:
            errs += 1
            results.append({"smiles": smi, "mace_e_pose": None, "mace_e_globalmin": None, "mace_strain": None})
            with open(ERR_LOG, "a") as ef:
                ef.write(f"chunk{args.chunk_id} mol{i}: {e}\n")
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(chunk)}] errors={errs}")

    df_out = pd.DataFrame(results)
    df_out.to_csv(out_file, index=False)
    print(f"Saved {len(df_out)} rows to {out_file} (errors={errs})")


if __name__ == "__main__":
    main()
