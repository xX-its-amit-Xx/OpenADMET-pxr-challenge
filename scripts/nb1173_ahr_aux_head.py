"""nb1173 — AhR agonist qHTS AC50 (PubChem AID 743122, Tox21) as a 4th aux multitask head
on top of the deployed sisterNR 3-head GNN (nb1168). AhR is PXR's canonical co-regulated xenosensor
(shared CYP3A/2B induction; rifampicin/indole/flavone/PAH cross-agonists). Its distinct PLANAR-aromatic
chemotype is the diversity the saturating NR1I/sister-NR bundle lacks (cy340/335 NEGATIVE). Activation
endpoint (AC50 -> p-scale). 283 novel rows, off-manifold coverage (med Tanimoto 0.224) but cy318
established diverse-activation aux heads still lift.

  control = deployed sisterNR 3-head GNN [PXR main(trn+435oct) + ext-EC50 aux + sisterNR aux]  (CACHED sn_oof_seed*)
  ahr     = [PXR main(trn+435oct) + ext aux + sisterNR aux + AhR aux]                           (this -> ahr_oof_seed*)

Same ~250 scaffold holdouts (MTL/ho_idx_seed*.npy), 3 seeds. Only difference vs control = the AhR aux head.
Resumable: caches OOF per seed. Honest gate (ensemble swap, nb1168 convention) is nb1173_gate.
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
AHR = "C:/pxr_work/ahr/ahr_aux.csv"
N_SEEDS = 3
EPOCHS = int(os.environ.get("MTL_EPOCHS", "30"))


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    ytr = tr["pec50"].to_numpy(); smi = tr["smiles"].tolist()
    ext = pd.read_csv(EXT); ext_smi = ext["smiles"].tolist(); ext_p = ext["p"].to_numpy()
    oc = pd.read_csv(OC); oc_smi = oc["cs"].tolist(); oc_p = oc["pec50"].to_numpy(float)
    sn = pd.read_csv(SN); sn_smi = sn["cs"].tolist(); sn_p = sn["p"].to_numpy(float)
    ah = pd.read_csv(AHR); ah_smi = ah["cs"].tolist(); ah_p = ah["p"].to_numpy(float)
    print(f"train {len(tr)} | octant main {len(oc)} | ext aux {len(ext)} | sisterNR aux {len(sn)} | ahr aux {len(ah)}", flush=True)

    for seed in range(N_SEEDS):
        ho = np.load(f"{OUT}/ho_idx_seed{seed}.npy")
        trn = np.array([i for i in range(len(tr)) if i not in set(ho.tolist())])
        ho_smi = [smi[i] for i in ho]
        opath = f"{OUT}/ahr_oof_seed{seed}.npy"
        if os.path.exists(opath):
            print(f"[seed {seed}] ahr cached", flush=True); continue
        print(f"[seed {seed}] ahr 4-head (trn {len(trn)}+{len(oc)}oct main, +{len(ext)}ext, +{len(sn)}sn, +{len(ah)}ahr)...", flush=True)
        main_smi = [smi[i] for i in trn] + oc_smi
        n_main = len(trn) + len(oc)
        # 4 task columns: [PXR-main, ext, sisterNR, ahr]
        main_Y = np.column_stack([np.concatenate([ytr[trn], oc_p]),
                                  np.full(n_main, np.nan), np.full(n_main, np.nan), np.full(n_main, np.nan)])
        ext_Y = np.column_stack([np.full(len(ext), np.nan), ext_p, np.full(len(ext), np.nan), np.full(len(ext), np.nan)])
        sn_Y = np.column_stack([np.full(len(sn), np.nan), np.full(len(sn), np.nan), sn_p, np.full(len(sn), np.nan)])
        ah_Y = np.column_stack([np.full(len(ah), np.nan), np.full(len(ah), np.nan), np.full(len(ah), np.nan), ah_p])
        allsmi = main_smi + ext_smi + sn_smi + ah_smi
        allY = np.vstack([main_Y, ext_Y, sn_Y, ah_Y])
        po = train_predict(allsmi, allY, ho_smi, 4, seed, EPOCHS)
        np.save(opath, po); print(f"  ahr done -> {opath}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
