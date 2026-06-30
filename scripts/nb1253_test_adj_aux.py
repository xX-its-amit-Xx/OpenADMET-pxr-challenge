"""nb1253 — Test-adjacent oracle aux head: pseudo-labeled test compounds as 4th multitask head.

Hypothesis (ledger L128): adding oracle-labeled test-space compounds as an AUX MULTITASK HEAD
forces the GNN to learn better representations in the test chemical space.

Oracle = Tanimoto-weighted kNN average of top-5 training neighbors (independent of primary model).
Generated set = 513 test compounds themselves (Tanimoto=1.0 to test, oracle labels from kNN).

  control  = sn_oof  (current deployed 3-head GNN: PXR-main + ext-EC50 + sisterNR)
  treatment = tadj_oof (4-head GNN: PXR-main + ext-EC50 + sisterNR + oracle-test-adj)

Same ~250 scaffold holdouts (MTL/ho_idx_seed*), 3 seeds. Gate: tadj replaces sn in ensemble.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.pxr.data import load_train
from nb1163_external_multitask_head import train_predict
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

OUT   = "C:/pxr_work/mtl"
EXT   = "C:/pxr_work/cpi2m/ext_ec50_aux.csv"
OC    = "C:/pxr_work/octant_htchem/octant_novel_curated.csv"
SN    = "C:/pxr_work/sisterNR/sisterNR_aux.csv"
TADJ  = "C:/pxr_work/gen/test_adj_gen_labeled.csv"  # oracle-labeled test compounds
N_SEEDS = 3
EPOCHS  = int(os.environ.get("MTL_EPOCHS", "30"))
START_SEED = int(os.environ.get("START_SEED", "0"))


def main():
    tr   = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    ytr  = tr["pec50"].to_numpy()
    smi  = tr["smiles"].tolist()

    ext  = pd.read_csv(EXT);  ext_smi  = ext["smiles"].tolist();  ext_p  = ext["p"].to_numpy()
    oc   = pd.read_csv(OC);   oc_smi   = oc["cs"].tolist();       oc_p   = oc["pec50"].to_numpy(float)
    sn   = pd.read_csv(SN);   sn_smi   = sn["cs"].tolist();       sn_p   = sn["p"].to_numpy(float)
    tadj = pd.read_csv(TADJ); tadj_smi = tadj["smiles"].tolist(); tadj_p = tadj["oracle_pec50"].to_numpy(float)

    print(f"train {len(tr)} | octant main {len(oc)} | ext-EC50 aux {len(ext)} "
          f"| sisterNR aux {len(sn)} | test-adj oracle aux {len(tadj)}", flush=True)

    # 4-head: [PXR, ext-EC50, sisterNR, test-adj-oracle]
    N_TASKS = 4

    for seed in range(START_SEED, N_SEEDS):
        ho   = np.load(f"{OUT}/ho_idx_seed{seed}.npy")
        trn  = np.array([i for i in range(len(tr)) if i not in set(ho.tolist())])
        ho_smi = [smi[i] for i in ho]

        opath = f"{OUT}/tadj_oof_seed{seed}.npy"
        if os.path.exists(opath):
            print(f"[seed {seed}] tadj cached", flush=True)
            continue

        print(f"[seed {seed}] 4-head MTL (trn {len(trn)}+{len(oc)}oct main, "
              f"+{len(ext)}ext, +{len(sn)}sn, +{len(tadj)}tadj)...", flush=True)

        # Build multi-task label matrix (N, 4)
        main_smi = [smi[i] for i in trn] + oc_smi
        n_main   = len(main_smi)

        # Each block only labeled on its head, NaN elsewhere
        main_Y = np.column_stack([
            np.concatenate([ytr[trn], oc_p]),
            np.full(n_main, np.nan),
            np.full(n_main, np.nan),
            np.full(n_main, np.nan),
        ])
        ext_Y = np.column_stack([
            np.full(len(ext), np.nan),
            ext_p,
            np.full(len(ext), np.nan),
            np.full(len(ext), np.nan),
        ])
        sn_Y = np.column_stack([
            np.full(len(sn), np.nan),
            np.full(len(sn), np.nan),
            sn_p,
            np.full(len(sn), np.nan),
        ])
        tadj_Y = np.column_stack([
            np.full(len(tadj), np.nan),
            np.full(len(tadj), np.nan),
            np.full(len(tadj), np.nan),
            tadj_p,
        ])

        allsmi = main_smi + ext_smi + sn_smi + tadj_smi
        allY   = np.vstack([main_Y, ext_Y, sn_Y, tadj_Y])

        preds = train_predict(allsmi, allY, ho_smi, N_TASKS, seed, EPOCHS)
        np.save(opath, preds)
        print(f"[seed {seed}] tadj OOF saved: {opath}", flush=True)
        print(f"  OOF mean={preds.mean():.3f} std={preds.std():.3f}", flush=True)

    print("All seeds done.", flush=True)


if __name__ == "__main__":
    main()
