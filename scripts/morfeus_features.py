"""morfeus SASA + surface-Dispersion descriptor block over the shared 4652-mol
corpus.

Genuinely-distinct from every deployed physics block: AIMNet2 (charges/forces),
MMFF strain (conformer energetics), DFT-D4 (dispersion ENERGY + C6/alpha), DBSTEP
(buried-volume/Sterimol/shape). morfeus contributes two surface observables NOT
encoded elsewhere:
  - SASA: solvent-accessible surface area + volume (solvent-exposure geometry,
    untested; DBSTEP's Vbur is occluded-volume, not solvent-exposed surface).
  - Dispersion (Pollice/Chen P_int surface model): the dispersion potential
    INTEGRATED OVER THE SURFACE (p_int) + its surface extrema (p_min/p_max) +
    dispersion area/volume. D4 gives the dispersion-ENERGY scalar and C6/alpha
    atomic coefficients; morfeus P_int is the SURFACE distribution of dispersion
    -- a different read of the same physics, low but non-zero orthogonality.

morfeus buried-volume/Sterimol deliberately NOT computed (needs a metal center;
failed for drug-like organics -> DBSTEP already covers that steric axis).

Resumable: appends per-row to C:/pxr_work/morfeus/morfeus_features.csv.
CPU only, multiprocess. NO protein/MSA (ligand-only physics).
"""
import os, csv, warnings
import numpy as np
warnings.filterwarnings("ignore")
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

CORPUS = "C:/pxr_work/xtb/corpus.csv"
OUTDIR = "C:/pxr_work/morfeus"; os.makedirs(OUTDIR, exist_ok=True)
OUT = f"{OUTDIR}/morfeus_features.csv"

COLS = ["name", "src", "mf_sasa_area", "mf_sasa_vol",
        "mf_disp_area", "mf_disp_vol", "mf_disp_pint",
        "mf_disp_pmin", "mf_disp_pmax"]


def featurize(args):
    name, src, smi = args
    from morfeus import SASA, Dispersion
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        m = Chem.AddHs(m)
        p = AllChem.ETKDGv3(); p.randomSeed = 42
        if AllChem.EmbedMolecule(m, p) != 0:
            if AllChem.EmbedMolecule(m, AllChem.ETKDGv2()) != 0:
                return None
        try:
            AllChem.MMFFOptimizeMolecule(m, maxIters=400)
        except Exception:
            pass
        conf = m.GetConformer()
        elems = [a.GetSymbol() for a in m.GetAtoms()]
        coords = conf.GetPositions()
        s = SASA(elems, coords)
        d = Dispersion(elems, coords)
        return [name, src, float(s.area), float(s.volume),
                float(d.area), float(d.volume), float(d.p_int),
                float(d.p_min), float(d.p_max)]
    except Exception:
        return None


def main():
    rows = list(csv.DictReader(open(CORPUS)))
    done = set()
    if os.path.exists(OUT):
        for r in csv.DictReader(open(OUT)):
            done.add(r["name"])
    todo = [(r["name"], r["src"], r["smiles"]) for r in rows if r["name"] not in done]
    print(f"corpus={len(rows)} done={len(done)} todo={len(todo)}", flush=True)
    new = not os.path.exists(OUT)
    fh = open(OUT, "a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(COLS)
    from multiprocessing import Pool
    n = 0
    with Pool(processes=6) as pool:
        for res in pool.imap_unordered(featurize, todo, chunksize=8):
            if res is not None:
                w.writerow(res); n += 1
                if n % 200 == 0:
                    fh.flush(); print(f"  wrote {n}/{len(todo)}", flush=True)
    fh.close()
    print(f"DONE wrote {n} new rows -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
