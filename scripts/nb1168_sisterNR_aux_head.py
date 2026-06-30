"""nb1168 — pooled sister-NR (VDR/PPARg/GR) agonist EC50/AC50 as an ADDITIONAL aux multitask head
on top of the deployed octant-main-head GNN (nb1166). CAR alone (143 rows) was too small to lift
(-0.0006, cy320); this pools the larger NR1I/NR1C/NR3C agonist sets into ONE bigger activation head.

  control   = deployed octant main-head GNN [PXR main(trn+435 octant) + 959 ext-EC50 aux]  (CACHED octant_oof_seed*)
  sn        = [PXR main(trn+435 octant) + ext-EC50 aux + sisterNR aux]                     (this -> sn_oof_seed*)

Same ~250 scaffold holdouts (MTL/ho_idx_seed*.npy), 3 seeds. Only difference vs control = the sisterNR aux head.
Resumable: caches OOF per seed. Honest gate (ensemble swap, nb1166 convention) is nb1168_gate.
All labels = ACTIVATION endpoint (EC50/AC50), best label match; binding!=activation caveat does NOT apply.
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
SN = "C:/pxr_work/sisterNR/sisterNR_aux.csv"
N_SEEDS = 3
EPOCHS = int(os.environ.get("MTL_EPOCHS", "30"))


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    ytr = tr["pec50"].to_numpy(); smi = tr["smiles"].tolist()
    ext = pd.read_csv(EXT); ext_smi = ext["smiles"].tolist(); ext_p = ext["p"].to_numpy()
    oc = pd.read_csv(OC); oc_smi = oc["cs"].tolist(); oc_p = oc["pec50"].to_numpy(float)
    sn = pd.read_csv(SN); sn_smi = sn["cs"].tolist(); sn_p = sn["p"].to_numpy(float)
    print(f"train {len(tr)} | octant main {len(oc)} | ext aux {len(ext)} | sisterNR aux {len(sn)}", flush=True)

    for seed in range(N_SEEDS):
        ho = np.load(f"{OUT}/ho_idx_seed{seed}.npy")
        trn = np.array([i for i in range(len(tr)) if i not in set(ho.tolist())])
        ho_smi = [smi[i] for i in ho]
        opath = f"{OUT}/sn_oof_seed{seed}.npy"
        if os.path.exists(opath):
            print(f"[seed {seed}] sn cached", flush=True); continue
        print(f"[seed {seed}] sisterNR 3-head (trn {len(trn)}+{len(oc)}oct main, +{len(ext)}ext, +{len(sn)}sn)...", flush=True)
        main_smi = [smi[i] for i in trn] + oc_smi
        n_main = len(trn) + len(oc)
        main_Y = np.column_stack([np.concatenate([ytr[trn], oc_p]),
                                  np.full(n_main, np.nan), np.full(n_main, np.nan)])
        ext_Y = np.column_stack([np.full(len(ext), np.nan), ext_p, np.full(len(ext), np.nan)])
        sn_Y = np.column_stack([np.full(len(sn), np.nan), np.full(len(sn), np.nan), sn_p])
        allsmi = main_smi + ext_smi + sn_smi
        allY = np.vstack([main_Y, ext_Y, sn_Y])
        po = train_predict(allsmi, allY, ho_smi, 3, seed, EPOCHS)
        np.save(opath, po); print(f"  sn done -> {opath}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
