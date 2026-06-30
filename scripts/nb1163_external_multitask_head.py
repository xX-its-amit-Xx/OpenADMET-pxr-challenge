"""nb1163 — RyeCatcher-mechanism test: external EC50 data as a chemprop AUXILIARY MULTITASK HEAD.

Cycle-309 finding: we previously tested ChEMBL/CPI2M/BindingDB external EC50 as POST-HOC RESIDUALS
(negative, "absorbed by the chempropembed sink"). RyeCatcher made the SAME data LIFT by injecting it as
auxiliary multitask heads inside Chemprop. This builds the matched experiment:

  control   = single-task ChemProp D-MPNN on PXR pEC50 (train minus holdout)
  treatment = multitask D-MPNN [PXR pEC50 main head + external-EC50 aux head], train minus holdout + 959 ext rows

Both trained on the SAME ~250 scaffold holdout (nb1127/nb1130 convention), 3 seeds. The only difference is the
aux head, so any delta is attributable to it. Produces matched OOF arrays for the PXR head on the holdout.
Resumable: caches OOF per (seed, variant) under C:/pxr_work/mtl/. Honest gate is a separate step (nb1163_gate).
EC50 = ACTIVATION endpoint (best label match; binding!=activation caveat does NOT apply here).
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train
from src.pxr.eval import scaffold_kfold_indices
from src.pxr.chem import standardize
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

OUT = "C:/pxr_work/mtl"; os.makedirs(OUT, exist_ok=True)
EXT = "C:/pxr_work/cpi2m/ext_ec50_aux.csv"
N_SEEDS = 3
EPOCHS = int(os.environ.get("MTL_EPOCHS", "30"))


def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""


def canon(s):
    m = standardize(s); return Chem.MolToSmiles(m) if m is not None else None


def train_predict(train_smiles, train_Y, pred_smiles, n_tasks, seed, epochs):
    """ChemProp 2.x multitask. train_Y shape (n, n_tasks) with NaN for missing. Returns pred for task 0."""
    import torch
    from lightning import pytorch as pl
    from chemprop import data, featurizers, models, nn
    pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)

    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    tr_dp = [data.MoleculeDatapoint.from_smi(s, np.asarray(y, dtype=float)) for s, y in zip(train_smiles, train_Y)]
    tr_ds = data.MoleculeDataset(tr_dp, feat)
    scaler = tr_ds.normalize_targets()  # masks NaN per-task
    tr_loader = data.build_dataloader(tr_ds, batch_size=64, num_workers=0, shuffle=True)

    pr_dp = [data.MoleculeDatapoint.from_smi(s, np.full(n_tasks, np.nan)) for s in pred_smiles]
    pr_ds = data.MoleculeDataset(pr_dp, feat)
    pr_loader = data.build_dataloader(pr_ds, batch_size=128, num_workers=0, shuffle=False)

    mp = nn.BondMessagePassing(depth=3, d_h=300)
    agg = nn.MeanAggregation()
    out_tf = nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn = nn.RegressionFFN(n_tasks=n_tasks, input_dim=300, hidden_dim=300, n_layers=2, dropout=0.1, output_transform=out_tf)
    model = models.MPNN(mp, agg, ffn, batch_norm=False, metrics=[nn.metrics.RMSE()])

    trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=epochs, enable_checkpointing=False,
                         enable_progress_bar=False, logger=False, enable_model_summary=False)
    trainer.fit(model, tr_loader)
    preds = trainer.predict(model, pr_loader)
    P = np.concatenate([p.numpy() for p in preds], 0)  # (n_pred, n_tasks)
    return P[:, 0]


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    ytr = tr["pec50"].to_numpy()
    smi = tr["smiles"].tolist()
    scaf = np.array([murcko(s) for s in smi])

    ext = pd.read_csv(EXT)
    ext_smi = ext["smiles"].tolist(); ext_p = ext["p"].to_numpy()
    print(f"train {len(tr)} | external EC50 aux {len(ext)} (novel-vs-train)", flush=True)

    n_splits = round(len(tr) / 250)
    for seed in range(N_SEEDS):
        folds = scaffold_kfold_indices(scaf.tolist(), n_splits=n_splits, seed=100 + seed)
        ho = min((f[1] for f in folds), key=lambda ix: abs(len(ix) - 250))
        ho = np.asarray(ho)
        trn = np.array([i for i in range(len(tr)) if i not in set(ho.tolist())])
        np.save(f"{OUT}/ho_idx_seed{seed}.npy", ho)
        ho_smi = [smi[i] for i in ho]

        # ---- control: single-task PXR ----
        cpath = f"{OUT}/control_oof_seed{seed}.npy"
        if not os.path.exists(cpath):
            print(f"[seed {seed}] control single-task (n_trn={len(trn)})...", flush=True)
            Yc = ytr[trn].reshape(-1, 1)
            pc = train_predict([smi[i] for i in trn], Yc, ho_smi, 1, seed, EPOCHS)
            np.save(cpath, pc); print(f"  control done -> {cpath}", flush=True)
        else:
            print(f"[seed {seed}] control cached", flush=True)

        # ---- treatment: multitask PXR + external EC50 ----
        tpath = f"{OUT}/treat_oof_seed{seed}.npy"
        if not os.path.exists(tpath):
            print(f"[seed {seed}] treatment multitask (+{len(ext)} ext aux)...", flush=True)
            # main rows: [pec50, NaN]; ext rows: [NaN, ext_p]
            main_smi = [smi[i] for i in trn]
            main_Y = np.column_stack([ytr[trn], np.full(len(trn), np.nan)])
            aux_Y = np.column_stack([np.full(len(ext), np.nan), ext_p])
            allsmi = main_smi + ext_smi
            allY = np.vstack([main_Y, aux_Y])
            pt = train_predict(allsmi, allY, ho_smi, 2, seed, EPOCHS)
            np.save(tpath, pt); print(f"  treatment done -> {tpath}", flush=True)
        else:
            print(f"[seed {seed}] treatment cached", flush=True)

    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
