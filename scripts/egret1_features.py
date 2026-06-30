"""egret1_features.py — Extract Egret-1 + Egret-1T energy/force scalars for 4652 mols.

Uses C:/aimnet_venv (has mace-torch + ase + matscipy already working).
Run as: C:/aimnet_venv/Scripts/python.exe scripts/egret1_features.py

Output: C:/pxr_work/egret/egret1_v2_features.csv
  cols: name, egret_energy, egret_energy_pa, egret_fmax, egret_frms,
        egret_fstd, egret_conf_mean, egret_conf_std,
        egret1t_energy, egret1t_energy_pa, egret1t_fmax, egret1t_frms,
        egret1t_conf_mean, egret1t_conf_std, src

Egret-1T is trained on transition states; running on reactant geometry gives
a "TS-energy-at-reactant" proxy that encodes reactivity/metabolic-lability.
This is distinct from ground-state energy (AIMNet2/OrbMol already deployed).
"""
import os, sys, time, traceback
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train, load_test

OUT_DIR = "C:/pxr_work/egret"
OUT_CSV = f"{OUT_DIR}/egret1_v2_features.csv"
CKPT    = f"{OUT_DIR}/egret1_v2_ckpt.csv"
MODEL1  = "C:/pxr_work/egret-public/compiled_models/EGRET_1.model"
MODEL1T = "C:/pxr_work/egret-public/compiled_models/EGRET_1T.model"

os.makedirs(OUT_DIR, exist_ok=True)


def make_ase_atoms(smi):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from ase import Atoms
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    ok = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if ok != 0:
        ok = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if ok != 0:
        return None
    AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    conf = mol.GetConformer()
    pos = conf.GetPositions()
    syms = [a.GetSymbol() for a in mol.GetAtoms()]
    return Atoms(symbols=syms, positions=pos)


def calc_egret(calc, atoms):
    atoms.calc = calc
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    n_atoms = len(atoms)
    fm = np.linalg.norm(f, axis=1)
    # Node-level confidence via model internals
    conf_vals = None
    try:
        from mace import data as mace_data
        from torch_geometric.data import Batch
        import torch
        batch = Batch.from_data_list([calc._atoms_to_batch(atoms)])
        with torch.no_grad():
            out = calc.models[0](batch)
        if "node_feat" in out:
            nf = out["node_feat"].detach().cpu().numpy()
            conf_vals = (float(np.mean(np.abs(nf))), float(np.std(nf)))
    except Exception:
        pass
    scalars = {
        "energy": float(e),
        "energy_pa": float(e / n_atoms),
        "fmax": float(fm.max()),
        "frms": float(np.sqrt(np.mean(fm**2))),
        "fstd": float(fm.std()),
        "conf_mean": conf_vals[0] if conf_vals else np.nan,
        "conf_std":  conf_vals[1] if conf_vals else np.nan,
    }
    return scalars


def main():
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test().reset_index(drop=True)
    rows_tr = [{"name": r["name"], "smiles": r["smiles"], "src": "train"} for _, r in tr.iterrows()]
    rows_te = [{"name": r["name"], "smiles": r["smiles"], "src": "test"}  for _, r in te.iterrows()]
    corpus = rows_tr + rows_te  # 4652 rows
    print(f"Corpus: {len(corpus)} molecules", flush=True)

    # Load existing checkpoint
    done = {}
    if os.path.exists(CKPT):
        ck = pd.read_csv(CKPT)
        for _, r in ck.iterrows():
            done[r["name"]] = r.to_dict()
        print(f"Loaded {len(done)} checkpointed rows", flush=True)

    # Load Egret-1 and Egret-1T calculators
    print("Loading Egret-1 and Egret-1T...", flush=True)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from mace.calculators import mace_off
        calc1  = mace_off(model=MODEL1,  default_dtype="float32", device="cpu")
        calc1t = mace_off(model=MODEL1T, default_dtype="float32", device="cpu")
    print("Models loaded", flush=True)

    COLS = ["name", "src",
            "egret_energy", "egret_energy_pa", "egret_fmax", "egret_frms", "egret_fstd",
            "egret_conf_mean", "egret_conf_std",
            "egret1t_energy", "egret1t_energy_pa", "egret1t_fmax", "egret1t_frms",
            "egret1t_conf_mean", "egret1t_conf_std"]

    t0 = time.time()
    ok_cnt, err_cnt = len(done), 0
    # Count pre-existing errors
    for d in done.values():
        if np.isnan(d.get("egret_energy", float("nan"))):
            err_cnt += 1
            ok_cnt -= 1

    new_rows = []
    for i, mol in enumerate(corpus):
        name = mol["name"]
        if name in done:
            continue  # skip already processed

        try:
            atoms = make_ase_atoms(mol["smiles"])
            if atoms is None:
                raise ValueError("Embedding failed")

            s1  = calc_egret(calc1,  atoms.copy())
            s1t = calc_egret(calc1t, atoms.copy())

            row = {
                "name": name, "src": mol["src"],
                "egret_energy":    s1["energy"],
                "egret_energy_pa": s1["energy_pa"],
                "egret_fmax":      s1["fmax"],
                "egret_frms":      s1["frms"],
                "egret_fstd":      s1["fstd"],
                "egret_conf_mean": s1["conf_mean"],
                "egret_conf_std":  s1["conf_std"],
                "egret1t_energy":    s1t["energy"],
                "egret1t_energy_pa": s1t["energy_pa"],
                "egret1t_fmax":      s1t["fmax"],
                "egret1t_frms":      s1t["frms"],
                "egret1t_conf_mean": s1t["conf_mean"],
                "egret1t_conf_std":  s1t["conf_std"],
            }
            done[name] = row
            new_rows.append(row)
            ok_cnt += 1
        except Exception as ex:
            err_row = {c: np.nan for c in COLS}
            err_row["name"] = name
            err_row["src"] = mol["src"]
            done[name] = err_row
            new_rows.append(err_row)
            err_cnt += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = max((i + 1 - len(done) + len(new_rows)), 1) / max(elapsed, 1)
            remaining = len(corpus) - (i + 1)
            eta = remaining / max(rate, 0.01)
            print(f"  [{i+1}/{len(corpus)}] ok={ok_cnt} err={err_cnt} "
                  f"{rate:.1f}/s ETA={eta/60:.1f}min", flush=True)
            # Checkpoint
            if new_rows:
                ck_df = pd.DataFrame(list(done.values()))
                ck_df.to_csv(CKPT, index=False)
                new_rows = []

    # Final save
    if new_rows:
        ck_df = pd.DataFrame(list(done.values()))
        ck_df.to_csv(CKPT, index=False)

    out_df = pd.DataFrame([done[m["name"]] for m in corpus])[COLS]
    out_df.to_csv(OUT_CSV, index=False)
    ok_final = out_df["egret_energy"].notna().sum()
    print(f"\nDone: {len(corpus)} rows, {ok_final} ok, {len(corpus)-ok_final} err")
    print(f"Saved -> {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
