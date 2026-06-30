"""DBSTEP steric / 3D-shape descriptor block over the shared 4652-mol corpus.

Genuinely-new STERIC/SHAPE physics axis (every prior QM row is electronic/
energetic; morfeus steric feats failed to populate). For each molecule:
  - ETKDGv3 + MMFF 3D conformer
  - Vol2Vec: %V_Bur at multiple sphere radii from the centroid-atom (shape
    PROFILE, the least-redundant DBSTEP read) + Sterimol L/Bmin/Bmax
  - RDKit 3D shape-anisotropy descriptors (NPR1/NPR2/Asphericity/Spherocity/
    Eccentricity/RadiusOfGyration/InertialShapeFactor) -- rod/disc/sphere
    geometry, least correlated with MW.

Resumable: appends per-row to C:/pxr_work/dbstep/dbstep_features.csv.
CPU only, multiprocess. NO protein/MSA (ligand-only physics).
"""
import os, sys, csv, warnings
import numpy as np
warnings.filterwarnings("ignore")
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors3D
RDLogger.DisableLog("rdApp.*")

CORPUS = "C:/pxr_work/xtb/corpus.csv"
OUTDIR = "C:/pxr_work/dbstep"; os.makedirs(OUTDIR, exist_ok=True)
OUT = f"{OUTDIR}/dbstep_features.csv"
RADII = [2.5, 3.5, 4.5, 5.5, 6.5]

COLS = ["name", "src"] + \
    [f"vbur_r{int(r*10)}" for r in RADII] + \
    ["ster_L", "ster_Bmin", "ster_Bmax", "ster_aniso"] + \
    ["npr1", "npr2", "asphericity", "spherocity", "eccentricity",
     "radgyr", "inertial_sf"]


def featurize(args):
    name, src, smi = args
    import dbstep.Dbstep as db
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
        pos = conf.GetPositions()
        cen = pos.mean(0)
        center_atom = int(np.argmin(((pos - cen) ** 2).sum(1))) + 1  # 1-indexed
        vburs = []
        L = Bmin = Bmax = np.nan
        for i, r in enumerate(RADII):
            try:
                obj = db.dbstep(m, atom1=center_atom, volume=True,
                                sterimol=(i == 0), commandline=False,
                                verbose=False, measure="grid", r=r, grid=0.5)
                vburs.append(float(obj.bur_vol))
                if i == 0:
                    L, Bmin, Bmax = float(obj.L), float(obj.Bmin), float(obj.Bmax)
            except Exception:
                vburs.append(np.nan)
        ster_aniso = (Bmax - Bmin) if (np.isfinite(Bmax) and np.isfinite(Bmin)) else np.nan
        # RDKit 3D shape
        try:
            npr1 = Descriptors3D.NPR1(m); npr2 = Descriptors3D.NPR2(m)
            asph = Descriptors3D.Asphericity(m); spher = Descriptors3D.SpherocityIndex(m)
            ecc = Descriptors3D.Eccentricity(m); rg = Descriptors3D.RadiusOfGyration(m)
            isf = Descriptors3D.InertialShapeFactor(m)
        except Exception:
            npr1 = npr2 = asph = spher = ecc = rg = isf = np.nan
        return [name, src] + vburs + [L, Bmin, Bmax, ster_aniso,
                                      npr1, npr2, asph, spher, ecc, rg, isf]
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
        for res in pool.imap_unordered(featurize, todo, chunksize=4):
            if res is not None:
                w.writerow(res); n += 1
                if n % 100 == 0:
                    fh.flush(); print(f"  wrote {n}/{len(todo)}", flush=True)
    fh.close()
    print(f"DONE wrote {n} new rows -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
