"""
Run CREST --quick --entropy on a slice of the corpus (conformational entropy / ensemble).
Usage: python run_crest_slice.py <start> <end>
Outputs: /scratch/shenoy.am/crest_pxr/results/slice_{start}_{end}.csv

Resumable: writes the output CSV incrementally every FLUSH_EVERY mols and, on restart,
skips molecules already present in the output CSV. An 8h-walltime timeout therefore
loses at most FLUSH_EVERY mols, and a resubmit picks up where it left off.
"""
import sys, os, re, subprocess, tempfile, shutil
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

WORK = "/scratch/shenoy.am/crest_pxr"
CORPUS = f"{WORK}/corpus.csv"
OUT_DIR = f"{WORK}/results"
os.makedirs(OUT_DIR, exist_ok=True)

FLUSH_EVERY = 20

start, end = int(sys.argv[1]), int(sys.argv[2])
df = pd.read_csv(CORPUS).iloc[start:end].reset_index(drop=True)
out_csv = f"{OUT_DIR}/slice_{start}_{end}.csv"

# resume: load already-done names
done_names = set()
records = []
if os.path.exists(out_csv):
    prev = pd.read_csv(out_csv)
    records = prev.to_dict("records")
    done_names = set(prev["name"].astype(str))
    print(f"Resuming slice {start}:{end}: {len(done_names)} already done", flush=True)

print(f"Slice {start}:{end} = {len(df)} mols ({len(done_names)} skipped)", flush=True)


def flush():
    pd.DataFrame(records).to_csv(out_csv, index=False)


for idx, row in df.iterrows():
    name = str(row['name'])
    if name in done_names:
        continue
    smi = str(row['smiles'])
    src = str(row['src'])

    rec = {'name': name, 'src': src,
           'crest_sconf': float('nan'), 'crest_gconf': float('nan'),
           'crest_nconf': float('nan'), 'crest_espread': float('nan'),
           'crest_emin': float('nan'), 'crest_ok': False}

    tmpdir = tempfile.mkdtemp(prefix='crest_')
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError("SMILES parse failed")
        mol = Chem.AddHs(mol)
        ps = AllChem.ETKDGv3(); ps.randomSeed = 42
        if AllChem.EmbedMolecule(mol, ps) < 0:
            raise ValueError("3D embed failed")
        AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
        conf = mol.GetConformer()
        atoms = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]
        pos = conf.GetPositions()

        xyz_file = os.path.join(tmpdir, "mol.xyz")
        with open(xyz_file, 'w') as f:
            f.write(f"{len(atoms)}\n{name}\n")
            for sym, (x, y, z) in zip(atoms, pos):
                f.write(f"{sym:3s} {x:12.6f} {y:12.6f} {z:12.6f}\n")

        # Run CREST --quick --entropy (conformational entropy)
        result = subprocess.run(
            ['crest', 'mol.xyz', '--T', '4', '--quick', '--gfn2', '--entropy'],
            cwd=tmpdir, capture_output=True, text=True, timeout=300
        )
        log = result.stdout + result.stderr

        # Parse Sconf (cal/mol/K) -- CREST prints "Sconf" / "S_conf" in the entropy block
        m = re.search(r'S[_ ]?conf\s*[=:]?\s*([\d.\-]+)', log, re.I)
        if m: rec['crest_sconf'] = float(m.group(1))

        # Parse G_conf / ensemble free energy
        m = re.search(r'G[_ ]?conf\s*[=:]?\s*([\d.\-]+)', log, re.I)
        if m: rec['crest_gconf'] = float(m.group(1))

        # Parse number of conformers and energy spread from crest.energies
        ene_file = os.path.join(tmpdir, "crest.energies")
        if os.path.exists(ene_file):
            energies = []
            for line in open(ene_file):
                parts = line.split()
                if len(parts) >= 2:
                    try: energies.append(float(parts[1]))
                    except: pass
            if energies:
                rec['crest_nconf'] = len(energies)
                rec['crest_emin'] = min(energies)
                rec['crest_espread'] = (max(energies) - min(energies)) * 627.509  # Hartree->kcal/mol
        elif re.search(r'CREST\s+TERMINATED\s+NORMALLY', log, re.I):
            m = re.search(r'(\d+)\s+conformers?\s+within', log, re.I)
            if m: rec['crest_nconf'] = float(m.group(1))

        rec['crest_ok'] = True

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: {name}", flush=True)
    except Exception as e:
        print(f"  ERR {name}: {e}", flush=True)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    records.append(rec)
    if (idx + 1) % FLUSH_EVERY == 0:
        flush()
        print(f"  Done {idx+1}/{len(df)} (flushed)", flush=True)

flush()
print(f"Saved {len(records)} records -> {out_csv}", flush=True)
