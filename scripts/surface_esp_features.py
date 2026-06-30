"""Politzer/GIPF molecular-surface electrostatic-potential (ESP) statistical
descriptors over the shared 4652-mol corpus (rows 191/192 of the ledger).

GENUINELY-NEW axis: every deployed physics block is per-atom or energetic --
AIMNet2 = atomic charges/forces/dipole, strain = conformer energetics, D4 =
dispersion/polarizability, DBSTEP = steric/shape. NONE encodes the SPATIAL
electrostatic-potential field PROJECTED ONTO THE MOLECULAR SURFACE -- the actual
electrostatic-complementarity observable a binding pocket reads. Politzer's GIPF
descriptors (Vmin/Vmax, sigma2_pos/neg/tot, balance nu, local-polarity Pi) are
classic predictors of intermolecular interaction.

Per molecule: ETKDGv3+MMFF conformer -> Gasteiger partial charges -> sample
vdW-surface points (Fibonacci sphere per atom, keep solvent-exposed/outer) ->
ESP V(r)=sum q_i/|r-R_i| at each surface point -> Politzer statistics.

Resumable: appends per-row to C:/pxr_work/surface_esp/surface_esp_features.csv.
CPU only, multiprocess. Ligand-only (no protein).
"""
import os, csv, warnings
import numpy as np
warnings.filterwarnings("ignore")
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

CORPUS = "C:/pxr_work/xtb/corpus.csv"
OUTDIR = "C:/pxr_work/surface_esp"; os.makedirs(OUTDIR, exist_ok=True)
OUT = f"{OUTDIR}/surface_esp_features.csv"

ESP_COLS = ["esp_vmax", "esp_vmin", "esp_vrange", "esp_vavg",
            "esp_vavg_pos", "esp_vavg_neg", "esp_sig2_pos", "esp_sig2_neg",
            "esp_sig2_tot", "esp_nu", "esp_pi", "esp_frac_pos"]
COLS = ["name", "src"] + ESP_COLS + ["status"]

N_SPHERE = 60          # Fibonacci points per atom on its vdW sphere
SCALE = 1.0            # sample on the vdW surface (Politzer ~0.001 au iso ~ vdW)


def fib_sphere(n):
    # unit-sphere points via Fibonacci spiral
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.column_stack([np.sin(phi) * np.cos(theta),
                            np.sin(phi) * np.sin(theta), np.cos(phi)])


UNIT = fib_sphere(N_SPHERE)


def featurize(args):
    name, src, smi = args
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return [name, src] + [np.nan] * len(ESP_COLS) + ["parse_fail"]
        m = Chem.AddHs(m)
        p = AllChem.ETKDGv3(); p.randomSeed = 42
        if AllChem.EmbedMolecule(m, p) != 0:
            if AllChem.EmbedMolecule(m, AllChem.ETKDGv2()) != 0:
                return [name, src] + [np.nan] * len(ESP_COLS) + ["embed_fail"]
        try:
            AllChem.MMFFOptimizeMolecule(m, maxIters=400)
        except Exception:
            pass
        AllChem.ComputeGasteigerCharges(m)
        conf = m.GetConformer()
        pos = conf.GetPositions()                       # (Na,3) Angstrom
        q = np.array([a.GetDoubleProp("_GasteigerCharge") for a in m.GetAtoms()])
        if not np.all(np.isfinite(q)):
            q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
        pt = Chem.GetPeriodicTable()
        rvdw = np.array([pt.GetRvdw(a.GetAtomicNum()) for a in m.GetAtoms()])
        Na = len(pos)
        # generate candidate surface points: each atom's vdW sphere
        surf = []
        for i in range(Na):
            cand = pos[i] + SCALE * rvdw[i] * UNIT       # (N_SPHERE,3)
            # keep points NOT buried inside any other atom's vdW sphere
            d = np.linalg.norm(cand[:, None, :] - pos[None, :, :], axis=2)  # (N_SPHERE,Na)
            d[:, i] = np.inf
            exposed = (d > (SCALE * rvdw[None, :])).all(axis=1)
            if exposed.any():
                surf.append(cand[exposed])
        if not surf:
            return [name, src] + [np.nan] * len(ESP_COLS) + ["no_surface"]
        surf = np.vstack(surf)                           # (Ns,3)
        # ESP at each surface point: V = sum_i q_i / |r - R_i|  (e / Angstrom)
        rij = np.linalg.norm(surf[:, None, :] - pos[None, :, :], axis=2)  # (Ns,Na)
        rij = np.clip(rij, 0.3, None)
        V = (q[None, :] / rij).sum(axis=1)               # (Ns,)
        vmax = float(V.max()); vmin = float(V.min()); vavg = float(V.mean())
        pos_m = V > 0; neg_m = ~pos_m
        vavg_pos = float(V[pos_m].mean()) if pos_m.any() else 0.0
        vavg_neg = float(V[neg_m].mean()) if neg_m.any() else 0.0
        sig2_pos = float(((V[pos_m] - vavg_pos) ** 2).mean()) if pos_m.sum() > 1 else 0.0
        sig2_neg = float(((V[neg_m] - vavg_neg) ** 2).mean()) if neg_m.sum() > 1 else 0.0
        sig2_tot = sig2_pos + sig2_neg
        nu = float(sig2_pos * sig2_neg / sig2_tot ** 2) if sig2_tot > 1e-9 else 0.0
        pi = float(np.abs(V - vavg).mean())              # local polarity index
        frac_pos = float(pos_m.mean())
        vals = [vmax, vmin, vmax - vmin, vavg, vavg_pos, vavg_neg,
                sig2_pos, sig2_neg, sig2_tot, nu, pi, frac_pos]
        return [name, src] + vals + ["ok"]
    except Exception as e:
        return [name, src] + [np.nan] * len(ESP_COLS) + [f"err:{type(e).__name__}"]


def main():
    rows = list(csv.DictReader(open(CORPUS)))
    done = set()
    if os.path.exists(OUT):
        for r in csv.DictReader(open(OUT)):
            done.add(r["name"])
    todo = [(r["name"], r["src"], r["smiles"]) for r in rows if r["name"] not in done]
    print(f"corpus={len(rows)} done={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        print("nothing to do", flush=True); return
    new = not os.path.exists(OUT)
    fh = open(OUT, "a", newline=""); w = csv.writer(fh)
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
