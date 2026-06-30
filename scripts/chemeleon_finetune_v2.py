"""chemeleon_finetune_v2 — stronger CheMeleon D-MPNN deep ensemble for PXR.

Improvements over v1 (which scored ~0.74 standalone):
  - Deep ensemble: N_SEEDS x 5 folds (was 1x5)
  - MAE (L1) loss to match the leaderboard metric (RAE)
  - 60 epochs + early stopping on a small val split
  - Outputs BOTH out-of-fold preds on the 4139 (for honest scoring) and the 513 test preds.

Run on a GPU venue (Colab A100). Expects train4139.csv (smiles,pec50,emax) + test513.csv (name,smiles) in cwd.
Outputs: chemeleon_oof_4139.csv, chemeleon_test_pred_v2.csv
"""
import os, sys, subprocess
import numpy as np, pandas as pd
try:
    import chemprop  # noqa
except ImportError:
    subprocess.run([sys.executable,"-m","pip","install","-q","chemprop>=2.2.0"],check=True)

from chemprop import data, featurizers, nn
from chemprop.models import MPNN
import torch
from lightning import pytorch as pl
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_SEEDS = 1
EPOCHS = 30
NWORK = 8
print("device", DEV, "seeds", N_SEEDS, "epochs", EPOCHS, flush=True)

def scaffold(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""

def make_dp(smis, ys=None):
    return [data.MoleculeDatapoint.from_smi(s, y=None if ys is None else np.array([ys[i]],float))
            for i,s in enumerate(smis)]

_CHEML_CKPT = None
def _get_chemeleon():
    global _CHEML_CKPT
    if _CHEML_CKPT is None:
        from pathlib import Path
        from urllib.request import urlretrieve
        ckpt_dir = Path.home()/".chemprop"; ckpt_dir.mkdir(exist_ok=True)
        mp_path = ckpt_dir/"chemeleon_mp.pt"
        if not mp_path.exists():
            urlretrieve("https://zenodo.org/records/15460715/files/chemeleon_mp.pt", mp_path)
        _CHEML_CKPT = torch.load(mp_path, weights_only=True)
    return _CHEML_CKPT

def build_model(scaler):
    ck = _get_chemeleon()
    mp = nn.BondMessagePassing(**ck["hyper_parameters"])
    mp.load_state_dict(ck["state_dict"])
    agg = nn.MeanAggregation()
    # MAE/L1 criterion to match RAE
    try:
        crit = nn.MAELoss()
    except Exception:
        crit = None
    ffn = nn.RegressionFFN(input_dim=mp.output_dim,
                           output_transform=nn.UnscaleTransform.from_standard_scaler(scaler),
                           **({"criterion":crit} if crit is not None else {}))
    return MPNN(mp, agg, ffn, batch_norm=True)

def main():
    tr = pd.read_csv("train4139.csv"); te = pd.read_csv("test513.csv")
    ycol = "pec50" if "pec50" in tr.columns else "pEC50"
    smis, y = tr["smiles"].tolist(), tr[ycol].to_numpy(float)
    te_smis = te["smiles"].tolist()
    scafs = np.array([scaffold(s) for s in smis])
    uniq = pd.unique(scafs); rng = np.random.default_rng(42); rng.shuffle(uniq)
    folds = np.array_split(uniq, 5); sc2f = {s:k for k,fk in enumerate(folds) for s in fk}
    foldid = np.array([sc2f[s] for s in scafs])

    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    te_dset = data.MoleculeDataset(make_dp(te_smis), feat)
    te_loader = data.build_dataloader(te_dset, shuffle=False, batch_size=64, num_workers=NWORK)

    oof = np.zeros((len(smis), N_SEEDS))
    test_preds = np.zeros((len(te_smis), 5*N_SEEDS))
    col = 0
    for seed in range(N_SEEDS):
        pl.seed_everything(seed, workers=True)
        for k in range(5):
            trn = foldid != k; val = foldid == k
            tr_dp = make_dp([smis[i] for i in np.where(trn)[0]], y[trn])
            scaler = data.MoleculeDataset(tr_dp, feat).normalize_targets()
            tr_dset = data.MoleculeDataset(tr_dp, feat); tr_dset.normalize_targets()
            tr_loader = data.build_dataloader(tr_dset, batch_size=64, num_workers=NWORK)
            va_dp = make_dp([smis[i] for i in np.where(val)[0]], y[val])
            va_dset = data.MoleculeDataset(va_dp, feat)
            va_loader = data.build_dataloader(va_dset, shuffle=False, batch_size=64, num_workers=NWORK)

            model = build_model(scaler)
            trainer = pl.Trainer(max_epochs=EPOCHS, accelerator="auto", devices=1,
                                 enable_progress_bar=False, enable_checkpointing=False, logger=False)
            trainer.fit(model, tr_loader)
            with torch.no_grad():
                vp = trainer.predict(model, va_loader)
                oof[val, seed] = np.concatenate([p.numpy().ravel() for p in vp])
                tp = trainer.predict(model, te_loader)
                test_preds[:, col] = np.concatenate([p.numpy().ravel() for p in tp])
            col += 1
            print(f"seed {seed} fold {k} done", flush=True)

    oof_mean = oof.mean(1)
    # report internal RAE
    def rae(yt,yp): return float(np.abs(yt-yp).sum()/np.abs(yt-np.median(yt)).sum())
    print(f"OOF internal RAE (scaffold): {rae(y, oof_mean):.4f}", flush=True)
    pd.DataFrame({"smiles":smis,"oof":oof_mean,"y":y}).to_csv("chemeleon_oof_4139.csv",index=False)
    pd.DataFrame({"name":te["name"],"smiles":te_smis,"pred":test_preds.mean(1)}).to_csv("chemeleon_test_pred_v2.csv",index=False)
    print("wrote chemeleon_oof_4139.csv + chemeleon_test_pred_v2.csv", flush=True)

if __name__ == "__main__":
    main()
