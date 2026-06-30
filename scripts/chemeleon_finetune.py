"""chemeleon_finetune.py — fine-tune CheMeleon (MIT Mordred-descriptor D-MPNN foundation model) on PXR.

Runs on a GPU venue (molab Blackwell / Colab / Kaggle). Fine-tunes CheMeleon as a BASE ANCHOR (the worthwhile test
per research — not a frozen feature), scaffold 5-fold, predicts the 513 test. Download test_pred.csv and score the
253 unblind locally (chemeleon_score.py) on RAE + corr-with-nb3200-error (the deploy metric).

Setup on the venue:
  pip install 'chemprop>=2.2.0'
  python chemeleon_finetune.py   # expects train4139.csv + test513.csv in cwd
"""
import os, sys, subprocess
import numpy as np, pandas as pd

try:
    import chemprop  # noqa
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "chemprop>=2.2.0"], check=True)

from chemprop import data, featurizers, models, nn
from chemprop.models import MPNN
import torch
from lightning import pytorch as pl
from sklearn.model_selection import KFold
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device", DEV, flush=True)


def scaffold(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""


def make_datapoints(smis, ys=None):
    return [data.MoleculeDatapoint.from_smi(s, y=None if ys is None else np.array([ys[i]], float))
            for i, s in enumerate(smis)]


def main():
    tr = pd.read_csv("train4139.csv"); te = pd.read_csv("test513.csv")
    smis, y = tr["smiles"].tolist(), tr["pEC50"].to_numpy()
    te_smis = te["smiles"].tolist()
    scafs = np.array([scaffold(s) for s in smis])
    uniq = pd.unique(scafs); rng = np.random.default_rng(42); rng.shuffle(uniq)
    folds = np.array_split(uniq, 5); sc2f = {s: k for k, fk in enumerate(folds) for s in fk}
    foldid = np.array([sc2f[s] for s in scafs])

    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    te_dset = data.MoleculeDataset(make_datapoints(te_smis), feat)
    te_loader = data.build_dataloader(te_dset, shuffle=False, batch_size=64)
    test_preds = np.zeros((len(te_smis), 5))

    for k in range(5):
        trn = foldid != k
        tr_dp = make_datapoints([smis[i] for i in np.where(trn)[0]], y[trn])
        scaler = data.MoleculeDataset(tr_dp, feat).normalize_targets()
        tr_dset = data.MoleculeDataset(tr_dp, feat); tr_dset.normalize_targets()
        tr_loader = data.build_dataloader(tr_dset, batch_size=64)
        # load CheMeleon foundation message-passing
        try:
            mp = nn.BondMessagePassing.from_foundation("CheMeleon")
        except Exception:
            from chemprop.utils import load_foundation
            mp = load_foundation("CheMeleon")
        agg = nn.MeanAggregation()
        ffn = nn.RegressionFFN(input_dim=mp.output_dim, output_transform=nn.UnscaleTransform.from_standard_scaler(scaler))
        model = MPNN(mp, agg, ffn, batch_norm=True)
        trainer = pl.Trainer(max_epochs=40, accelerator="auto", devices=1, enable_progress_bar=False,
                             enable_checkpointing=False, logger=False)
        trainer.fit(model, tr_loader)
        with torch.no_grad():
            preds = trainer.predict(model, te_loader)
        test_preds[:, k] = np.concatenate([p.numpy().ravel() for p in preds])
        print(f"fold {k} done", flush=True)

    out = pd.DataFrame({"smiles": te_smis, "pred": test_preds.mean(1)})
    out.to_csv("chemeleon_test_pred.csv", index=False)
    print("wrote chemeleon_test_pred.csv", flush=True)


if __name__ == "__main__":
    main()
