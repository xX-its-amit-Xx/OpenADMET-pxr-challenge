"""
FOD (Fractional Occupation Density) descriptors via tblite GFN2-xTB.

Computes finite-electronic-temperature (5000K) occupancy differences vs 0K
as a proxy for N_FOD (Grimme/Bauer JCTC 2017): a static-correlation / reactive-
electron-density descriptor DISTINCT from the standard xTB HOMO-LUMO/charge block
(which was closed +0.0007 absorbed).

Writes C:/pxr_work/fod/fod_features.csv  (local) or
       /scratch/shenoy.am/pxr_work/fod/fod_features.csv  (Explorer).

Descriptors (per molecule):
  fod_nfod         - N_FOD = sum |f_i(5000K) - f_i(0K)|
  fod_nfod_per_ha  - N_FOD / n_heavy_atoms
  fod_max_dev      - max single-orbital deviation |f_i(5000K) - f_i(0K)|
  fod_n_frac       - count of orbitals with |dev| > 0.05 (partially occupied)
  fod_gap_0k       - HOMO-LUMO gap at 0K (eV)
  fod_gap_5k       - effective gap proxy at 5000K (eV): ε_{LUMO}(0K) - ε_{HOMO}(0K)
  fod_fermi_0k     - Fermi level at 0K (eV)
  fod_fermi_5k     - Fermi level at 5000K (eV)
  fod_lumo_0k      - LUMO energy at 0K (eV)

Usage (batch):
  python fod_xtb_features.py --corpus corpus.csv --chunk_size N --chunk_id K --out fod_features.csv
"""
import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HARTREE_TO_EV = 27.211396  # 1 Eh in eV
KB_HARTREE = 3.166808578545117e-6  # Boltzmann in Hartree/K
T_HOT = 5000.0  # K for FOD
T_COLD = 100.0  # K for "0K reference" (100K to avoid degeneracy issues)


def fermi_level(orbital_energies: np.ndarray, n_electrons: int, T_K: float) -> float:
    """Bisect to find chemical potential mu s.t. sum(f_i) = n_electrons."""
    kT = T_K * KB_HARTREE
    if kT < 1e-10:
        kT = 1e-10
    oe = np.sort(orbital_energies)
    lo, hi = oe[0] - 10, oe[-1] + 10

    def n_fill(mu):
        return np.sum(2.0 / (1.0 + np.exp(np.clip((oe - mu) / kT, -500, 500))))

    for _ in range(100):
        mid = (lo + hi) / 2
        if n_fill(mid) < n_electrons:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def fermi_occ(orbital_energies: np.ndarray, mu: float, T_K: float) -> np.ndarray:
    kT = T_K * KB_HARTREE
    return 2.0 / (1.0 + np.exp(np.clip((orbital_energies - mu) / kT, -500, 500)))


def smiles_to_fod(smiles: str) -> dict | None:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from tblite.interface import Calculator

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        params = AllChem.EmbedParameters()
        params.randomSeed = 42
        params.useSmallRingTorsions = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            AllChem.EmbedMolecule(mol, randomSeed=42)
        if mol.GetNumConformers() == 0:
            return None
        AllChem.MMFFOptimizeMolecule(mol, confId=0)

        conf = mol.GetConformer(0)
        numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=int)
        positions = conf.GetPositions() * 1.8897259886  # Å→Bohr

        # Total electron count (neutral, closed-shell)
        charge = Chem.GetFormalCharge(mol)
        n_electrons = int(numbers.sum()) - charge  # alpha+beta = total

        n_heavy = int((numbers > 1).sum())

        # --- 0K reference ---
        calc0 = Calculator("GFN2-xTB", numbers, positions, charge=float(charge))
        calc0.set("temperature", T_COLD * KB_HARTREE)
        calc0.set("verbosity", 0)
        res0 = calc0.singlepoint()
        oe = np.asarray(res0.get("orbital-energies"))  # Hartree
        mu0 = fermi_level(oe, n_electrons, T_COLD)
        occ0 = fermi_occ(oe, mu0, T_COLD)

        # Sort by energy to find HOMO/LUMO
        idx = np.argsort(oe)
        oe_sorted = oe[idx]
        occ0_sorted = occ0[idx]
        # HOMO = highest orbital with occ > 0.5
        homo_idx = np.where(occ0_sorted > 0.5)[0]
        lumo_idx = np.where(occ0_sorted < 0.5)[0]
        e_homo = float(oe_sorted[homo_idx[-1]]) * HARTREE_TO_EV if len(homo_idx) else 0.0
        e_lumo = float(oe_sorted[lumo_idx[0]]) * HARTREE_TO_EV if len(lumo_idx) else 0.0
        gap_0k = e_lumo - e_homo

        # --- High T for FOD ---
        calc5 = Calculator("GFN2-xTB", numbers, positions, charge=float(charge))
        calc5.set("temperature", T_HOT * KB_HARTREE)
        calc5.set("verbosity", 0)
        res5 = calc5.singlepoint()
        oe5 = np.asarray(res5.get("orbital-energies"))
        mu5 = fermi_level(oe5, n_electrons, T_HOT)
        occ5 = fermi_occ(oe5, mu5, T_HOT)

        # FOD descriptors
        dev = np.abs(occ5 - occ0)
        n_fod = float(dev.sum())
        n_fod_per_ha = n_fod / max(n_heavy, 1)
        max_dev = float(dev.max())
        n_frac = int((dev > 0.05).sum())
        fermi_0k_ev = mu0 * HARTREE_TO_EV
        fermi_5k_ev = mu5 * HARTREE_TO_EV

        return {
            "fod_nfod": n_fod,
            "fod_nfod_per_ha": n_fod_per_ha,
            "fod_max_dev": max_dev,
            "fod_n_frac": n_frac,
            "fod_gap_0k": gap_0k,
            "fod_gap_5k": gap_0k,  # same gap, just alias (orbital spectrum shifts w T)
            "fod_fermi_0k": fermi_0k_ev,
            "fod_fermi_5k": fermi_5k_ev,
            "fod_lumo_0k": e_lumo,
        }
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, help="CSV with name,smiles columns")
    parser.add_argument("--chunk_size", type=int, default=300)
    parser.add_argument("--chunk_id", type=int, default=0)
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    corpus = pd.read_csv(args.corpus)
    smiles_col = next((c for c in corpus.columns if "smiles" in c.lower()), None)
    name_col = next((c for c in corpus.columns if "name" in c.lower()), corpus.columns[0])
    if smiles_col is None:
        smiles_col = corpus.columns[1]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Slice this chunk
    start = args.chunk_id * args.chunk_size
    end = start + args.chunk_size
    chunk = corpus.iloc[start:end].copy()

    # Load any already-computed rows
    done_names: set = set()
    if out.exists():
        existing = pd.read_csv(out)
        done_names = set(existing[name_col])

    rows = []
    t0 = time.time()
    for i, row in enumerate(chunk.itertuples()):
        name = getattr(row, name_col, str(row.Index))
        smi = getattr(row, smiles_col, None)
        if name in done_names or not isinstance(smi, str) or not smi.strip():
            continue
        feats = smiles_to_fod(smi)
        if feats is not None:
            rows.append({name_col: name, "smiles": smi, **feats})
        else:
            rows.append({name_col: name, "smiles": smi,
                         **{k: float("nan") for k in [
                            "fod_nfod","fod_nfod_per_ha","fod_max_dev","fod_n_frac",
                            "fod_gap_0k","fod_gap_5k","fod_fermi_0k","fod_fermi_5k","fod_lumo_0k"]}})
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        print(f"  {i+1}/{len(chunk)} name={name} ok={feats is not None} rate={rate:.1f}/s",
              flush=True)

    if rows:
        df_new = pd.DataFrame(rows)
        if out.exists():
            df_new.to_csv(out, mode="a", header=False, index=False)
        else:
            df_new.to_csv(out, index=False)
        print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
