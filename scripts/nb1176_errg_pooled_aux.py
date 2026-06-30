"""nb1176 — pool 75 novel ERRγ/ESRRG functional EC50 (ChEMBL CHEMBL4245, activation endpoint)
INTO the deployed sister-NR aux head (cy318/325 diverse-activation logic). Treatment head =
sisterNR(4295) + ERRγ(75). Control = deployed sisterNR (sn_oof, already cached).

  treatment = [PXR main(trn+435 octant) + 959 ext-EC50 aux + (sisterNR+ERRγ) aux] -> sn_errg_oof_seed*
Same ~250 scaffold holdouts (MTL/ho_idx_seed*), 3 seeds. Gate = nb1176_gate (sn_errg vs sn deployed).
All labels = ACTIVATION endpoint (EC50). binding!=activation caveat does NOT apply. Resumable (caches OOF).
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
ERRG = "C:/pxr_work/sisterNR/errg_aux.csv"
N_SEEDS = 3
EPOCHS = int(os.environ.get("MTL_EPOCHS", "30"))


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    ytr = tr["pec50"].to_numpy(); smi = tr["smiles"].tolist()
    ext = pd.read_csv(EXT); ext_smi = ext["smiles"].tolist(); ext_p = ext["p"].to_numpy()
    oc = pd.read_csv(OC); oc_smi = oc["cs"].tolist(); oc_p = oc["pec50"].to_numpy(float)
    sn = pd.read_csv(SN); errg = pd.read_csv(ERRG)
    sn_smi = sn["cs"].tolist() + errg["cs"].tolist()
    sn_p = np.concatenate([sn["p"].to_numpy(float), errg["p"].to_numpy(float)])
    print(f"train {len(tr)} | octant main {len(oc)} | ext aux {len(ext)} | sisterNR+ERRG aux {len(sn_smi)} (errg {len(errg)})", flush=True)

    for seed in range(N_SEEDS):
        ho = np.load(f"{OUT}/ho_idx_seed{seed}.npy")
        trn = np.array([i for i in range(len(tr)) if i not in set(ho.tolist())])
        ho_smi = [smi[i] for i in ho]
        opath = f"{OUT}/sn_errg_oof_seed{seed}.npy"
        if os.path.exists(opath):
            print(f"[seed {seed}] sn_errg cached", flush=True); continue
        print(f"[seed {seed}] 3-head (trn {len(trn)}+{len(oc)}oct main, +{len(ext)}ext, +{len(sn_smi)}sn+errg)...", flush=True)
        main_smi = [smi[i] for i in trn] + oc_smi
        n_main = len(trn) + len(oc)
        main_Y = np.column_stack([np.concatenate([ytr[trn], oc_p]),
                                  np.full(n_main, np.nan), np.full(n_main, np.nan)])
        ext_Y = np.column_stack([np.full(len(ext), np.nan), ext_p, np.full(len(ext), np.nan)])
        sn_Y = np.column_stack([np.full(len(sn_smi), np.nan), np.full(len(sn_smi), np.nan), sn_p])
        allsmi = main_smi + ext_smi + sn_smi
        allY = np.vstack([main_Y, ext_Y, sn_Y])
        po = train_predict(allsmi, allY, ho_smi, 3, seed, EPOCHS)
        np.save(opath, po); print(f"  sn_errg done -> {opath}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
