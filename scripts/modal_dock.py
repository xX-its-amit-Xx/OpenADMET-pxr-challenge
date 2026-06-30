"""modal_dock.py — REAL Vina docking of PXR activity ligands on Modal (Linux) -> pose-energy features.
The compound (pEC50) track USING STRUCTURES. Physics-based docking score = the one signal flagged most
likely NOT absorbed by chempropembed (dock_pxr.py). Receptor=PXR 2o9i pocket (nb980 box).
Pilot:  modal run scripts/modal_dock.py --mode pilot
Full :  modal run scripts/modal_dock.py --mode test   (513) ;  --mode train (4139)
"""
import modal

app = modal.App("pxr-dock")
image = (
    modal.Image.micromamba(python_version="3.11")
    .micromamba_install("vina", "meeko", "rdkit", "openbabel", "numpy", "pandas", "scipy",
                        channels=["conda-forge"])
    .add_local_file("data/external/dock/pxr_receptor_2o9i.pdb", "/root/receptor.pdb")
    .add_local_file("data/external/dock/dock_box.json", "/root/box.json")
    .add_local_file("data/processed/unimol_test513.csv", "/root/test.csv")
    .add_local_file("data/processed/unimol_train.csv", "/root/train.csv")
)
EXHAUST = 8


def _prep_receptor():
    """obabel PDB -> rigid receptor PDBQT (robust; adds charges+AD atom types)."""
    import subprocess, os
    if not os.path.exists("/root/receptor.pdbqt"):
        subprocess.run(["obabel", "/root/receptor.pdb", "-O", "/root/receptor.pdbqt", "-xr"],
                       capture_output=True, text=True)
    return os.path.exists("/root/receptor.pdbqt") and os.path.getsize("/root/receptor.pdbqt") > 0


def _lig_pdbqt(smiles):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from meeko import MoleculePreparation
    try:
        m = Chem.AddHs(Chem.MolFromSmiles(smiles))
        p = AllChem.ETKDGv3(); p.randomSeed = 42
        if AllChem.EmbedMolecule(m, p) != 0:
            p.useRandomCoords = True
            if AllChem.EmbedMolecule(m, p) != 0:
                return None
        AllChem.MMFFOptimizeMolecule(m, maxIters=200)
        prep = MoleculePreparation(); setups = prep.prepare(m)
        from meeko import PDBQTWriterLegacy
        pdbqt, ok, _ = PDBQTWriterLegacy.write_string(setups[0])
        return pdbqt if ok else None
    except Exception:
        return None


@app.function(image=image, cpu=4.0, timeout=7200)
def dock_chunk(args):
    import json, numpy as np
    smiles_list, cid = args
    from vina import Vina
    ok = _prep_receptor()
    box = json.load(open("/root/box.json"))
    out = np.full(len(smiles_list), np.nan, dtype=np.float64)
    if not ok:
        print(f"chunk{cid}: RECEPTOR PREP FAILED", flush=True); return out.tolist()
    v = Vina(sf_name="vina", verbosity=0)
    v.set_receptor("/root/receptor.pdbqt")
    v.compute_vina_maps(center=[box["center_x"], box["center_y"], box["center_z"]],
                        box_size=[box["size_x"], box["size_y"], box["size_z"]])
    for i, smi in enumerate(smiles_list):
        lp = _lig_pdbqt(smi)
        if lp is None:
            continue
        try:
            v.set_ligand_from_string(lp)
            v.dock(exhaustiveness=EXHAUST, n_poses=5)
            out[i] = float(v.energies(n_poses=1)[0][0])  # best-pose total energy (kcal/mol; lower=better)
        except Exception:
            pass
    print(f"chunk{cid}: docked {int(np.isfinite(out).sum())}/{len(smiles_list)}", flush=True)
    return out.tolist()


def _run(smiles, tag, n_chunks):
    import numpy as np
    chunks = [(list(smiles[i::n_chunks]), i) for i in range(n_chunks)]  # round-robin
    results = list(dock_chunk.map(chunks))
    scores = np.full(len(smiles), np.nan)
    for ci in range(n_chunks):
        idx = list(range(ci, len(smiles), n_chunks))
        for j, gi in enumerate(idx):
            scores[gi] = results[ci][j]
    return scores


@app.local_entrypoint()
def main(mode: str = "pilot"):
    import os, pandas as pd, numpy as np
    OUT = "C:/pxr_struct/dock"; os.makedirs(OUT, exist_ok=True)
    te = pd.read_csv("data/processed/unimol_test513.csv"); tr = pd.read_csv("data/processed/unimol_train.csv")
    if mode == "pilot":
        smi = te["smiles"].astype(str).tolist()[:6]
        s = _run(smi, "pilot", 2)
        print("PILOT vina best-pose energies (kcal/mol):")
        for q, v in zip(smi, s):
            print(f"  {v if np.isfinite(v) else 'FAIL':>8}  {q[:50]}")
        print(f"finite {int(np.isfinite(s).sum())}/6 -> {'PIPELINE OK, scale to --mode test' if np.isfinite(s).sum()>=4 else 'PIPELINE BROKEN, debug'}")
        return
    df = te if mode == "test" else tr
    smi = df["smiles"].astype(str).tolist()
    nch = 16 if mode == "test" else 32
    print(f"docking {len(smi)} {mode} ligands across {nch} Modal containers (Vina exh={EXHAUST})...")
    s = _run(smi, mode, nch)
    np.save(f"{OUT}/vina_scores_{mode}.npy", s.astype(np.float64))
    print(f"saved {OUT}/vina_scores_{mode}.npy  finite={int(np.isfinite(s).sum())}/{len(s)}  "
          f"mean={np.nanmean(s):.2f} std={np.nanstd(s):.2f}")
    print("NEXT: nb1012 test vina score as residual feature on combined+chempropembed (nb982 protocol)")
