"""dock_pxr.py — REAL docking of PXR ligands (Vina) -> pose-energy features.
RUN IN THE CODESPACE (Linux); NOT tested on Windows (vina/meeko are Linux-friendly).

WHY: cycle-291 found ChempropEmbed absorbs every structure-derived feature (ADMET, 3D-shape,
anchor-fit). Vina docking score is PHYSICS-based (a scoring function on a 3D pose), NOT a learned
structure embedding -> the one signal most likely NOT absorbed by chempropembed. This is the last
distinct lever for the activity ladder that needs no GPU.

PIPELINE (see DOCKING_RUNBOOK.md):
  1. receptor: run scripts/nb980_prep_receptor_box.py -> pxr_receptor_2o9i.pdb + dock_box.json
     then: mk_prepare_receptor.py -i pxr_receptor_2o9i.pdb -o receptor.pdbqt -p   (meeko)
  2. this script: rdkit 3D + meeko per ligand -> vina dock -> best score
  3. test: does vina score add to combined+chempropembed (the nb964/nb982 ladder protocol)?

Outputs to ./dock_out/ (Codespace workspace volume).
"""
import os, sys, json, time
import numpy as np

OUT = os.environ.get("DOCK_OUT", "dock_out")
os.makedirs(OUT, exist_ok=True)
BOX = os.environ.get("DOCK_BOX", "dock_box.json")          # from nb980
RECEPTOR_PDBQT = os.environ.get("RECEPTOR_PDBQT", "receptor.pdbqt")
EXHAUSTIVENESS = int(os.environ.get("VINA_EXHAUST", "8"))


def prep_ligand_pdbqt(smiles):
    """SMILES -> 3D (rdkit ETKDG+MMFF) -> meeko PDBQT string. None on failure."""
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
        prep = MoleculePreparation()
        setups = prep.prepare(m)
        # meeko >=0.5 returns setups; write_pdbqt_string
        try:
            from meeko import PDBQTWriterLegacy
            pdbqt, ok, _ = PDBQTWriterLegacy.write_string(setups[0])
            return pdbqt if ok else None
        except Exception:
            return prep.write_pdbqt_string()
    except Exception:
        return None


def dock_all(smiles_list, tag):
    from vina import Vina
    box = json.load(open(BOX))
    ckpt = f"{OUT}/scores_{tag}.npy"
    scores = np.load(ckpt) if os.path.exists(ckpt) else np.full(len(smiles_list), np.nan)
    v = Vina(sf_name="vina", verbosity=0)
    v.set_receptor(RECEPTOR_PDBQT)
    v.compute_vina_maps(center=[box["center_x"], box["center_y"], box["center_z"]],
                        box_size=[box["size_x"], box["size_y"], box["size_z"]])
    t0 = time.time()
    for i, smi in enumerate(smiles_list):
        if np.isfinite(scores[i]):
            continue
        pdbqt = prep_ligand_pdbqt(smi)
        if pdbqt is None:
            scores[i] = np.nan; continue
        try:
            v.set_ligand_from_string(pdbqt)
            v.dock(exhaustiveness=EXHAUSTIVENESS, n_poses=5)
            scores[i] = v.energies(n_poses=1)[0][0]   # best pose total energy (kcal/mol; lower=better)
        except Exception:
            scores[i] = np.nan
        if (i + 1) % 25 == 0:
            np.save(ckpt, scores)
            print(f"  {tag} {i+1}/{len(smiles_list)} ({(time.time()-t0)/60:.1f} min) "
                  f"done={int(np.isfinite(scores).sum())}", flush=True)
    np.save(ckpt, scores)
    return scores


def main():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.pxr.data import load_train, load_test
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test()
    print(f"docking {len(te)} test + {len(tr)} train ligands (Vina, exhaustiveness={EXHAUSTIVENESS})")
    print("NOTE: ~5-30 s/ligand on CPU -> test set ~1-4 h; train ~10-40 h. Checkpointed; resumable.")
    s_te = dock_all(te["smiles"].tolist(), "test")
    s_tr = dock_all(tr["smiles"].tolist(), "train")
    print(f"\ndone. test scores finite: {int(np.isfinite(s_te).sum())}/{len(s_te)}; "
          f"train: {int(np.isfinite(s_tr).sum())}/{len(s_tr)}")
    print(f"-> {OUT}/scores_test.npy + scores_train.npy")
    print("NEXT: test if vina score adds to combined+chempropembed on the 253 (reuse nb982 protocol, "
          "swap anchorfit for the vina score). If stable-negative -> first real ladder break.")


if __name__ == "__main__":
    main()
