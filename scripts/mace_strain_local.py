"""
MACE-StrainRelief: DFT-accurate conformational strain via MACE-OFF23 multi-conformer.

Approach:
 - Reuse seed conformer mace_energy from C:/pxr_work/mace_off/mace_features.csv (E_seed)
 - Generate N_EXTRA=4 additional ETKDGv3+MMFF conformers; compute MACE single-point on each
 - mace_strain = E_seed - min(E_seed, E_conf1..4)   [=0 if seed IS global min]
 - mace_conf_espread = std of MACE energies across conformers / n_heavy
 - mace_conf_erange  = max-min of MACE energies / n_heavy

Distinct from existing:
  - mace_fmax/frms (force norms at MMFF pose) -> absorbed as SINGLE-POINT scalar
  - MMFF strain -> cruder force-field energies
  This gives DFT-accurate CONFORMATIONAL STRAIN (multi-conformer ensemble).

Run: TORCHDYNAMO_DISABLE=1 C:/aimnet_venv/Scripts/python.exe scripts/mace_strain_local.py
Output: C:/pxr_work/mace_strain_local/mace_strain_local.csv
"""
import os, sys, csv, time
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
from mace.calculators import mace_off

SEED_CACHE   = "C:/pxr_work/mace_off/mace_features.csv"
CORPUS       = "C:/pxr_work/xtb/corpus.csv"
OUTDIR       = "C:/pxr_work/mace_strain_local"
OUT          = os.path.join(OUTDIR, "mace_strain_local.csv")
N_EXTRA      = 3   # additional conformers beyond the cached seed (~1.7h on CPU)
SEED         = 42

os.makedirs(OUTDIR, exist_ok=True)
COLS = ["name", "src", "smiles",
        "mace_strain", "mace_conf_espread", "mace_conf_erange", "mace_conf_n",
        "status"]


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
            if r.get("status", "") == "ok":
                s.add(r["name"])
    return s


def seed_energies():
    """Load the already-computed seed conformer MACE energies."""
    df = pd.read_csv(SEED_CACHE)
    df = df[df["status"] == "ok"][["name", "mace_energy"]].set_index("name")
    return df["mace_energy"].to_dict()


def extra_conformer_energies(smi, calc, n_extra, seed_offset=100):
    """Generate n_extra ETKDG+MMFF conformers and return MACE single-point energies."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return []
    m = Chem.AddHs(m)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed_offset  # different seed from seed conformer (42)
    params.numThreads = 0
    n_gen = AllChem.EmbedMultipleConfs(m, numConfs=n_extra, params=params)
    if n_gen == 0:
        return []
    AllChem.MMFFOptimizeMoleculeConfs(m, numThreads=0)
    energies = []
    for cid in range(m.GetNumConformers()):
        conf = m.GetConformer(cid)
        positions = conf.GetPositions()
        symbols = [atom.GetSymbol() for atom in m.GetAtoms()]
        atoms = Atoms(symbols=symbols, positions=positions)
        atoms.calc = calc
        try:
            e = float(atoms.get_potential_energy())
            energies.append(e)
        except Exception:
            pass
    return energies


def main():
    corpus = load_corpus()
    done = done_names()
    e_seed = seed_energies()
    todo = [r for r in corpus if r[0] not in done]
    print(f"corpus {len(corpus)} | seed_cache {len(e_seed)} | done {len(done)} | todo {len(todo)}", flush=True)

    calc = mace_off(model="medium", device="cpu", default_dtype="float64")
    print("MACE-OFF23 medium loaded", flush=True)

    exists = os.path.exists(OUT)
    fout = open(OUT, "a", newline="")
    w = csv.DictWriter(fout, fieldnames=COLS)
    if not exists:
        w.writeheader()

    ok = err = 0
    t_start = time.time()
    for i, (name, src, smi) in enumerate(todo):
        rec = {"name": name, "src": src, "smiles": smi, "status": "ok"}
        try:
            e_ref = e_seed.get(name)
            if e_ref is None:
                # Need to compute seed conformer too (shouldn't happen for most)
                e_ref = None

            extras = extra_conformer_energies(smi, calc, n_extra=N_EXTRA, seed_offset=100)
            all_e = extras
            if e_ref is not None:
                all_e = [e_ref] + extras

            if not all_e:
                raise RuntimeError("no_conformers")

            e_min = min(all_e)
            e_max = max(all_e)
            strain = (e_ref if e_ref is not None else e_min) - e_min
            n_heavy = Chem.MolFromSmiles(smi).GetNumHeavyAtoms() if Chem.MolFromSmiles(smi) else 1
            e_pa = np.array(all_e) / n_heavy

            rec.update({
                "mace_strain": round(strain, 6),
                "mace_conf_espread": round(float(np.std(e_pa)), 6),
                "mace_conf_erange": round((e_max - e_min) / n_heavy, 6),
                "mace_conf_n": len(all_e),
            })
            ok += 1
        except Exception as ex:
            for c in COLS[3:-1]:
                rec[c] = ""
            rec["status"] = f"err:{str(ex)[:40]}"
            err += 1

        w.writerow(rec)
        if (i + 1) % 50 == 0:
            fout.flush()
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(todo) - i - 1) / rate / 3600
            print(f"  {i+1}/{len(todo)} ok={ok} err={err} rate={rate:.1f}/s ETA={eta:.1f}h", flush=True)

    fout.flush(); fout.close()
    print(f"DONE ok={ok} err={err}", flush=True)


if __name__ == "__main__":
    main()
