"""nb1169 — NCATS qHTS PXR agonist AC50 (PubChem AID 1346982+1346985) as a 4th aux multitask head
on top of the deployed sisterNR 3-head GNN (nb1168). Same PXR receptor (agonist), different assay (qHTS),
activation endpoint (AC50 -> p-scale). 1711 novel rows, off-manifold coverage (med Tanimoto 0.210) but
cy318 established diverse-activation aux heads still lift; this is the closest receptor match after octant.

  control = deployed sisterNR 3-head GNN [PXR main(trn+435oct) + ext-EC50 aux + sisterNR aux]  (CACHED sn_oof_seed*)
  nc      = [PXR main(trn+435oct) + ext aux + sisterNR aux + NCATS aux]                          (this -> nc_oof_seed*)

Same ~250 scaffold holdouts (MTL/ho_idx_seed*.npy), 3 seeds. Only difference vs control = the NCATS aux head.
Resumable: caches OOF per seed. Honest gate (ensemble swap, nb1168 convention) is nb1169_gate.
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
NC = "C:/pxr_work/ncats/ncats_aux.csv"
N_SEEDS = 3
EPOCHS = int(os.environ.get("MTL_EPOCHS", "30"))


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    ytr = tr["pec50"].to_numpy(); smi = tr["smiles"].tolist()
    ext = pd.read_csv(EXT); ext_smi = ext["smiles"].tolist(); ext_p = ext["p"].to_numpy()
    oc = pd.read_csv(OC); oc_smi = oc["cs"].tolist(); oc_p = oc["pec50"].to_numpy(float)
    sn = pd.read_csv(SN); sn_smi = sn["cs"].tolist(); sn_p = sn["p"].to_numpy(float)
    nc = pd.read_csv(NC); nc_smi = nc["cs"].tolist(); nc_p = nc["p"].to_numpy(float)
    print(f"train {len(tr)} | octant main {len(oc)} | ext aux {len(ext)} | sisterNR aux {len(sn)} | ncats aux {len(nc)}", flush=True)

    for seed in range(N_SEEDS):
        ho = np.load(f"{OUT}/ho_idx_seed{seed}.npy")
        trn = np.array([i for i in range(len(tr)) if i not in set(ho.tolist())])
        ho_smi = [smi[i] for i in ho]
        opath = f"{OUT}/nc_oof_seed{seed}.npy"
        if os.path.exists(opath):
            print(f"[seed {seed}] nc cached", flush=True); continue
        print(f"[seed {seed}] ncats 4-head (trn {len(trn)}+{len(oc)}oct main, +{len(ext)}ext, +{len(sn)}sn, +{len(nc)}nc)...", flush=True)
        main_smi = [smi[i] for i in trn] + oc_smi
        n_main = len(trn) + len(oc)
        # 4 task columns: [PXR-main, ext, sisterNR, ncats]
        main_Y = np.column_stack([np.concatenate([ytr[trn], oc_p]),
                                  np.full(n_main, np.nan), np.full(n_main, np.nan), np.full(n_main, np.nan)])
        ext_Y = np.column_stack([np.full(len(ext), np.nan), ext_p, np.full(len(ext), np.nan), np.full(len(ext), np.nan)])
        sn_Y = np.column_stack([np.full(len(sn), np.nan), np.full(len(sn), np.nan), sn_p, np.full(len(sn), np.nan)])
        nc_Y = np.column_stack([np.full(len(nc), np.nan), np.full(len(nc), np.nan), np.full(len(nc), np.nan), nc_p])
        allsmi = main_smi + ext_smi + sn_smi + nc_smi
        allY = np.vstack([main_Y, ext_Y, sn_Y, nc_Y])
        po = train_predict(allsmi, allY, ho_smi, 4, seed, EPOCHS)
        np.save(opath, po); print(f"  nc done -> {opath}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
