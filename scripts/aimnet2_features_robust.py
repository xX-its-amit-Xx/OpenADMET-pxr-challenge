"""Robust resumable AIMNet2 QM/physics scalar extractor (per-molecule timeout guard).

Same features/output as aimnet2_features.py but each molecule runs in a worker
process with a hard wall-clock timeout, so a single pathological molecule
(ETKDG/MMFF/AIMNet2 hang) can't stall the whole array. On timeout the worker is
killed, the molecule recorded as err:timeout, and a fresh worker spawned.

Resumable: appends per row to the SAME OUT; on restart skips names already present.
Run:
  TORCHDYNAMO_DISABLE=1 C:/aimnet_venv/Scripts/python.exe scripts/aimnet2_features_robust.py
"""
import os, csv, time
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FTimeout

CORPUS = "C:/pxr_work/xtb/corpus.csv"
OUTDIR = "C:/pxr_work/aimnet2"
OUT = os.path.join(OUTDIR, "aimnet_features.csv")
TIMEOUT = 90  # seconds per molecule
COLS = ["name", "src", "smiles", "aimnet_energy", "aimnet_qmin", "aimnet_qmax",
        "aimnet_qabs_mean", "aimnet_qstd", "aimnet_qsum_abs", "aimnet_dipole",
        "aimnet_fmax", "aimnet_frms", "status"]

_CALC = None


def _init_worker():
    global _CALC
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    import torch
    try:
        import torch._dynamo as _d; _d.config.suppress_errors = True
    except Exception:
        pass
    from aimnet.calculators import AIMNet2Calculator
    _CALC = AIMNet2Calculator("aimnet2", device="cpu")


def _work(smi):
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem
    global _CALC
    m = Chem.MolFromSmiles(smi)
    if m is None:
        raise RuntimeError("parse_fail")
    m = Chem.AddHs(m)
    p = AllChem.ETKDGv3(); p.randomSeed = 42
    if AllChem.EmbedMolecule(m, p) != 0:
        if AllChem.EmbedMolecule(m, useRandomCoords=True, randomSeed=42) != 0:
            raise RuntimeError("embed_fail")
    try:
        AllChem.MMFFOptimizeMolecule(m, maxIters=500)
    except Exception:
        pass
    conf = m.GetConformer()
    coord = np.array([list(conf.GetAtomPosition(i)) for i in range(m.GetNumAtoms())], dtype=float)
    numbers = np.array([a.GetAtomicNum() for a in m.GetAtoms()], dtype=int)
    charge = Chem.GetFormalCharge(m)
    out = _CALC({"coord": coord, "numbers": numbers, "charge": int(charge)}, forces=True)
    q = out["charges"].detach().cpu().numpy().ravel().astype(float)
    e = float(out["energy"].detach().cpu().numpy().ravel()[0])
    f = out["forces"].detach().cpu().numpy().reshape(-1, 3).astype(float)
    fn = np.linalg.norm(f, axis=1)
    dip = float(np.linalg.norm((q[:, None] * coord).sum(axis=0)))
    return dict(
        aimnet_energy=e, aimnet_qmin=float(q.min()), aimnet_qmax=float(q.max()),
        aimnet_qabs_mean=float(np.abs(q).mean()), aimnet_qstd=float(q.std()),
        aimnet_qsum_abs=float(np.abs(q).sum()), aimnet_dipole=dip,
        aimnet_fmax=float(fn.max()), aimnet_frms=float(np.sqrt((fn**2).mean())),
    )


def load_corpus():
    with open(CORPUS, newline="") as f:
        return [(r["name"], r["src"], r["smiles"]) for r in csv.DictReader(f)]


def done_names():
    if not os.path.exists(OUT):
        return set()
    with open(OUT, newline="") as f:
        return {r["name"] for r in csv.DictReader(f)}


def main():
    rows = load_corpus()
    done = done_names()
    new = os.path.exists(OUT)
    todo = [r for r in rows if r[0] not in done]
    print(f"corpus {len(rows)} | done {len(done)} | todo {len(todo)}", flush=True)
    f = open(OUT, "a", newline="")
    w = csv.DictWriter(f, fieldnames=COLS)
    if not new:
        w.writeheader()
    ok = err = 0
    ex = ProcessPoolExecutor(max_workers=1, initializer=_init_worker)
    try:
        for i, (name, src, smi) in enumerate(todo):
            rec = {"name": name, "src": src, "smiles": smi, "status": "ok"}
            try:
                fut = ex.submit(_work, smi)
                rec.update(fut.result(timeout=TIMEOUT))
                ok += 1
            except FTimeout:
                rec["status"] = "err:timeout"
                for c in COLS[3:-1]:
                    rec[c] = ""
                err += 1
                ex.shutdown(wait=False, cancel_futures=True)
                ex = ProcessPoolExecutor(max_workers=1, initializer=_init_worker)
            except Exception as e:
                rec["status"] = f"err:{str(e)[:40]}"
                for c in COLS[3:-1]:
                    rec[c] = ""
                err += 1
            w.writerow(rec)
            if (i + 1) % 25 == 0:
                f.flush()
                print(f"  {i+1}/{len(todo)} ok={ok} err={err}", flush=True)
    finally:
        f.flush(); f.close()
        ex.shutdown(wait=False, cancel_futures=True)
    print(f"DONE ok={ok} err={err}", flush=True)


if __name__ == "__main__":
    main()
