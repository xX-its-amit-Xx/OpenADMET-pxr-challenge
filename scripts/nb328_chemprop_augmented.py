"""nb328 -- Chemprop retrain on augmented set (4139 train + 253 unblind + 96 + 457).

Uses Chemprop 2.x's BondMessagePassing + FFN. Single-task (pec50 regression).
Trains for limited epochs on CPU; predicts on the 513 test compounds.
nb93_chemprop_large_gpu is currently the top-1 contributor at 33.9% in nb320 blend.
A retrain on augmented data should push it further.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from lightning import pytorch as L
from chemprop import data, featurizers, nn, models

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


def std_smi(s):
    try:
        m = Chem.MolFromSmiles(str(s))
        return Chem.MolToSmiles(m) if m else None
    except: return None


def main():
    print("=== nb328: Chemprop retrain on augmented set ===\n")
    # Build augmented training set: orig + unblind + uscale + htchem
    rows = []
    tr_orig = pd.read_csv("data/raw/pxr-challenge_TRAIN.csv")
    for _, r in tr_orig.iterrows():
        rows.append((r['SMILES'], r['pEC50']))
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    for _, r in unb.iterrows():
        rows.append((r['SMILES'], r['pEC50']))
    mu = pd.read_csv("data/raw/pxr-challenge_96-compound-uscale-semi-pure_TRAIN.csv")
    mu['_pec'] = pd.to_numeric(mu['Corrected Semi-Pure pEC50 (log)'], errors='coerce')
    for _, r in mu.iterrows():
        if pd.notna(r['_pec']):
            rows.append((r['SMILES'], float(r['_pec'])))
    ht = pd.read_csv("data/raw/pxr-challenge_htchem-libraries_TRAIN.csv")
    ht['_pec'] = pd.to_numeric(ht['Corrected Crude pEC50 (log)'], errors='coerce')
    for _, r in ht.iterrows():
        if pd.notna(r['_pec']):
            rows.append((r['SMILES'], float(r['_pec'])))
    df = pd.DataFrame(rows, columns=['SMILES', 'pec50']).dropna()
    df['std_smiles'] = df['SMILES'].apply(std_smi)
    df = df.dropna(subset=['std_smiles']).drop_duplicates('std_smiles').reset_index(drop=True)
    print(f"Augmented training set: {len(df)} unique compounds")

    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    te_df['std_smiles'] = te_df['SMILES'].apply(std_smi)

    # Build Chemprop datasets
    print("Building Chemprop datasets...")
    train_data = [data.MoleculeDatapoint.from_smi(s, np.array([float(y)])) for s, y in zip(df['std_smiles'], df['pec50'])]
    test_data  = [data.MoleculeDatapoint.from_smi(s) for s in te_df['std_smiles']]
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_dset = data.MoleculeDataset(train_data, featurizer)
    test_dset  = data.MoleculeDataset(test_data, featurizer)

    # Normalize targets (chemprop 2.x: get scaler, pass via output_transform)
    scaler = train_dset.normalize_targets()
    # Build UnscaleTransform from sklearn StandardScaler attrs
    output_transform = nn.transforms.UnscaleTransform(mean=scaler.mean_, scale=scaler.scale_)

    # Random 90/10 split for early stopping
    rng = np.random.default_rng(42)
    n = len(train_dset); perm = rng.permutation(n)
    cut = int(0.9 * n)
    tr_idx = perm[:cut]; va_idx = perm[cut:]
    tr_sub = torch.utils.data.Subset(train_dset, tr_idx.tolist())
    va_sub = torch.utils.data.Subset(train_dset, va_idx.tolist())
    tr_loader = data.build_dataloader(tr_sub, batch_size=64, num_workers=0)
    va_loader = data.build_dataloader(va_sub, batch_size=64, num_workers=0, shuffle=False)
    te_loader = data.build_dataloader(test_dset, batch_size=64, num_workers=0, shuffle=False)

    # Build model: MPNN + FFN regression
    mp = nn.BondMessagePassing(depth=3, d_h=300)
    agg = nn.MeanAggregation()
    ffn = nn.RegressionFFN(input_dim=mp.output_dim, hidden_dim=300, n_layers=2, output_transform=output_transform)
    metric_list = [nn.metrics.RMSE(), nn.metrics.MAE()]
    model = models.MPNN(mp, agg, ffn, batch_norm=True, metrics=metric_list)
    print(f"Model: {sum(p.numel() for p in model.parameters())} params")

    # Train (CPU, ~30-40 epochs feasible)
    trainer = L.Trainer(
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        max_epochs=30, accelerator='cpu', devices=1,
        callbacks=[L.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')],
    )
    print("Training chemprop (30 epochs CPU)...")
    trainer.fit(model, tr_loader, va_loader)

    # Predict test
    print("Predicting test...")
    pred = trainer.predict(model, te_loader)
    pred = torch.cat(pred).cpu().numpy().squeeze()
    print(f"te_pred: shape={pred.shape}  mean={pred.mean():.3f}  std={pred.std():.3f}")
    np.save(DATA_PROCESSED / "te_nb328_chemprop_aug.npy", pred)

    # Sanity on unblind 253 (leaky)
    name_to_idx = {n: i for i, n in enumerate(te_df['Molecule Name'])}
    unb_idx_in_te = np.array([name_to_idx[n] for n in unb['Molecule Name'] if n in name_to_idx])
    unb_y_in_te = np.array([float(unb.loc[unb['Molecule Name'] == te_df['Molecule Name'].iloc[i], 'pEC50'].iloc[0]) for i in unb_idx_in_te])
    print(f"\nOn unblind 253 (leaky): RAE={rae(unb_y_in_te, pred[unb_idx_in_te]):.4f}")

    sub = pd.DataFrame({
        'Molecule Name': te_df['Molecule Name'],
        'SMILES': te_df['SMILES'],
        'pEC50': pred,
    })
    out = SUBMISSIONS / "nb328_chemprop_augmented.csv"
    sub.to_csv(out, index=False)
    print(f"Wrote {out.name}")


if __name__ == "__main__":
    main()
