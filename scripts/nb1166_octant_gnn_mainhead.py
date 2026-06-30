"""nb1166 — Octant HTChem pEC50 as EXTRA MAIN-HEAD rows inside the DEPLOYED multitask GNN.

nb1165 tested Octant as extra GBM rows (held GNN fixed) -> NEGATIVE (+0.0002). But the deployed lift
mechanism (nb1163/nb1164) is the chemprop GNN, and Octant is the SAME Octant agonism reporter assay +
SAME drug-like chemspace as our 513 test (median octant->test Tanimoto 0.419, 48 test cmpds nbr>=0.5).
So the highest-prior mechanism is to inject the 435 novel Octant pEC50s as EXTRA rows of the MAIN PXR head
of the deployed multitask GNN (PXR main + 959 ext-EC50 aux), not as a GBM row or a separate head.

  control  = deployed multitask GNN [PXR main(trn) + ext-EC50 aux]            (CACHED: treat_oof_seed*.npy)
  octant   = deployed multitask GNN [PXR main(trn + 435 octant) + ext aux]    (this script -> octant_oof_seed*.npy)

Same ~250 scaffold holdouts (MTL/ho_idx_seed*.npy), 3 seeds. Only difference vs control = the octant main-head
rows. Resumable: caches OOF per seed. Honest gate (ensemble swap) is nb1166_gate.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.pxr.data import load_train
from nb1163_external_multitask_head import train_predict
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

OUT = "C:/pxr_work/mtl"
EXT = "C:/pxr_work/cpi2m/ext_ec50_aux.csv"
OC = "C:/pxr_work/octant_htchem/octant_novel_curated.csv"
N_SEEDS = 3
EPOCHS = int(os.environ.get("MTL_EPOCHS", "30"))


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    ytr = tr["pec50"].to_numpy(); smi = tr["smiles"].tolist()
    ext = pd.read_csv(EXT); ext_smi = ext["smiles"].tolist(); ext_p = ext["p"].to_numpy()
    oc = pd.read_csv(OC); oc_smi = oc["cs"].tolist(); oc_p = oc["pec50"].to_numpy(float)
    print(f"train {len(tr)} | ext aux {len(ext)} | octant main-head rows {len(oc)}", flush=True)

    for seed in range(N_SEEDS):
        ho = np.load(f"{OUT}/ho_idx_seed{seed}.npy")
        trn = np.array([i for i in range(len(tr)) if i not in set(ho.tolist())])
        ho_smi = [smi[i] for i in ho]
        opath = f"{OUT}/octant_oof_seed{seed}.npy"
        if os.path.exists(opath):
            print(f"[seed {seed}] octant cached", flush=True); continue
        print(f"[seed {seed}] octant main-head (trn {len(trn)} +{len(oc)} octant, +{len(ext)} ext aux)...", flush=True)
        # main PXR head: trn pec50 + octant pec50 (same assay).  aux head: ext EC50.
        main_smi = [smi[i] for i in trn] + oc_smi
        main_Y = np.column_stack([np.concatenate([ytr[trn], oc_p]), np.full(len(trn) + len(oc), np.nan)])
        aux_Y = np.column_stack([np.full(len(ext), np.nan), ext_p])
        allsmi = main_smi + ext_smi
        allY = np.vstack([main_Y, aux_Y])
        po = train_predict(allsmi, allY, ho_smi, 2, seed, EPOCHS)
        np.save(opath, po); print(f"  octant done -> {opath}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
