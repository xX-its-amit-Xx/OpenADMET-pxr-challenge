"""nb1303 — rat rPXR qHTS Potency (402 novel rows) as a 4th aux multitask head.
Tests whether rat-species PXR activation data (same endpoint, coverage-neg) lifts
on top of the deployed 3-head sisterNR GNN config.

  control   = deployed sisterNR 3-head GNN [PXR main + ext-EC50 aux + sisterNR aux]  (CACHED sn_oof_seed*)
  rat_pxr   = [PXR main + ext-EC50 aux + sisterNR aux + rat rPXR aux]                (-> rpxr_oof_seed*)

Same ~250 scaffold holdouts (MTL/ho_idx_seed*.npy), 3 seeds. Only difference = rat rPXR 4th head.
Coverage-neg (median 0.203 Tanimoto to 513 test). Prior: likely sub-threshold like CAR/AhR.
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
RAT = "C:/pxr_work/orthologs/rat_pxr_novel.csv"
N_SEEDS = 3
EPOCHS = int(os.environ.get("MTL_EPOCHS", "30"))


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    ytr = tr["pec50"].to_numpy(); smi = tr["smiles"].tolist()
    ext = pd.read_csv(EXT); ext_smi = ext["smiles"].tolist(); ext_p = ext["p"].to_numpy()
    oc = pd.read_csv(OC); oc_smi = oc["cs"].tolist(); oc_p = oc["pec50"].to_numpy(float)
    sn = pd.read_csv(SN); sn_smi = sn["cs"].tolist(); sn_p = sn["p"].to_numpy(float)
    rat = pd.read_csv(RAT); rat_smi = rat["std_smiles"].tolist(); rat_p = rat["p_activity"].to_numpy(float)
    print(f"train {len(tr)} | octant main {len(oc)} | ext aux {len(ext)} | sisterNR aux {len(sn)} | rat rPXR aux {len(rat)}", flush=True)

    for seed in range(N_SEEDS):
        opath = f"{OUT}/rpxr_oof_seed{seed}.npy"
        if os.path.exists(opath):
            print(f"[seed {seed}] rat cached", flush=True); continue
        ho = np.load(f"{OUT}/ho_idx_seed{seed}.npy")
        trn = np.array([i for i in range(len(tr)) if i not in set(ho.tolist())])
        ho_smi = [smi[i] for i in ho]
        print(f"[seed {seed}] rat 4-head (trn {len(trn)}+{len(oc)}oct, +{len(ext)}ext, +{len(sn)}sn, +{len(rat)}rat)...", flush=True)
        # head0=PXR main (trn+octant), head1=ext EC50, head2=sisterNR, head3=rat rPXR
        main_smi = [smi[i] for i in trn] + oc_smi
        n_main = len(main_smi)
        main_Y = np.column_stack([
            np.concatenate([ytr[trn], oc_p]),
            np.full(n_main, np.nan), np.full(n_main, np.nan), np.full(n_main, np.nan)
        ])
        ext_Y = np.column_stack([
            np.full(len(ext), np.nan), ext_p,
            np.full(len(ext), np.nan), np.full(len(ext), np.nan)
        ])
        sn_Y = np.column_stack([
            np.full(len(sn), np.nan), np.full(len(sn), np.nan),
            sn_p, np.full(len(sn), np.nan)
        ])
        rat_Y = np.column_stack([
            np.full(len(rat), np.nan), np.full(len(rat), np.nan),
            np.full(len(rat), np.nan), rat_p
        ])
        allsmi = main_smi + ext_smi + sn_smi + rat_smi
        allY = np.vstack([main_Y, ext_Y, sn_Y, rat_Y])
        po = train_predict(allsmi, allY, ho_smi, 4, seed, EPOCHS)
        np.save(opath, po)
        print(f"  rat done -> {opath}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
