"""nb1170 — EXTENDED sister-NR agonist EC50 (FXR/NR1H4, LXRa/NR1H3, LXRb/NR1H2, RORg/RORC,
RXRa/NR2B1, ERa/ESR1) POOLED into the sister-NR aux head, on top of the deployed sisterNR 3-head GNN
(nb1168, best 0.4367). cy330: the deployed sisterNR head only pools CAR+VDR+PPARg+GR; this extends it
with the rest of the druggable-NR agonist space (documented PXR cross-activators). cy318 mechanism:
diverse/sister activation rows lift as aux heads even off-manifold.

  control = deployed sisterNR 3-head GNN [PXR main(trn+435 octant) + ext-EC50 aux + sisterNR aux]  (CACHED sn_oof_seed*)
  sne     = [PXR main(trn+435 octant) + ext-EC50 aux + (sisterNR UNION sisterNR_ext) aux]          (this -> sne_oof_seed*)

Same ~250 scaffold holdouts (MTL/ho_idx_seed*.npy), 3 seeds. Only difference vs control = the EXTENDED
rows added to the sister-NR aux head. Resumable: caches OOF per seed. Honest gate is nb1170_gate.
All labels = ACTIVATION endpoint (EC50/AC50); binding!=activation caveat does NOT apply.
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
SNE = "C:/pxr_work/sisterNR_ext/sisterNR_ext_aux.csv"
N_SEEDS = 3
EPOCHS = int(os.environ.get("MTL_EPOCHS", "30"))


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    ytr = tr["pec50"].to_numpy(); smi = tr["smiles"].tolist()
    ext = pd.read_csv(EXT); ext_smi = ext["smiles"].tolist(); ext_p = ext["p"].to_numpy()
    oc = pd.read_csv(OC); oc_smi = oc["cs"].tolist(); oc_p = oc["pec50"].to_numpy(float)
    sn = pd.read_csv(SN); sne = pd.read_csv(SNE)
    # pool deployed sisterNR head with the extended rows (both already novel/deduped)
    pool_smi = sn["cs"].tolist() + sne["cs"].tolist()
    pool_p = np.concatenate([sn["p"].to_numpy(float), sne["p"].to_numpy(float)])
    print(f"train {len(tr)} | octant main {len(oc)} | ext aux {len(ext)} | sisterNR aux {len(sn)} "
          f"+ ext {len(sne)} = pooled {len(pool_smi)}", flush=True)

    for seed in range(N_SEEDS):
        ho = np.load(f"{OUT}/ho_idx_seed{seed}.npy")
        trn = np.array([i for i in range(len(tr)) if i not in set(ho.tolist())])
        ho_smi = [smi[i] for i in ho]
        opath = f"{OUT}/sne_oof_seed{seed}.npy"
        if os.path.exists(opath):
            print(f"[seed {seed}] sne cached", flush=True); continue
        print(f"[seed {seed}] sisterNR-EXT 3-head (trn {len(trn)}+{len(oc)}oct main, +{len(ext)}ext, +{len(pool_smi)}sn-pooled)...", flush=True)
        main_smi = [smi[i] for i in trn] + oc_smi
        n_main = len(trn) + len(oc)
        main_Y = np.column_stack([np.concatenate([ytr[trn], oc_p]),
                                  np.full(n_main, np.nan), np.full(n_main, np.nan)])
        ext_Y = np.column_stack([np.full(len(ext), np.nan), ext_p, np.full(len(ext), np.nan)])
        sn_Y = np.column_stack([np.full(len(pool_smi), np.nan), np.full(len(pool_smi), np.nan), pool_p])
        allsmi = main_smi + ext_smi + pool_smi
        allY = np.vstack([main_Y, ext_Y, sn_Y])
        po = train_predict(allsmi, allY, ho_smi, 3, seed, EPOCHS)
        np.save(opath, po); print(f"  sne done -> {opath}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
