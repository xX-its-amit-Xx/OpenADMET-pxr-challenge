"""nb333 -- Multi-seed Chemprop ensemble on augmented training set.

Trains 5 Chemprop models with different random seeds on (4139 + 253 + 96 + 457)
= 4833 compounds. Averages predictions. Should reduce nb328's variance.
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
    print("=== nb333: Multi-seed Chemprop ensemble (5 seeds) ===\n")
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
    print(f"Augmented training set: {len(df)} unique")

    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    te_df['std_smiles'] = te_df['SMILES'].apply(std_smi)
    name_to_idx = {n: i for i, n in enumerate(te_df['Molecule Name'])}
    unb_te_idx = np.array([name_to_idx[n] for n in unb['Molecule Name'] if n in name_to_idx])
    unb_y_arr = unb['pEC50'].values

    seed_preds = []
    for seed in [42, 7, 13, 21, 100]:
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed); np.random.seed(seed)
        train_data = [data.MoleculeDatapoint.from_smi(s, np.array([float(y)])) for s, y in zip(df['std_smiles'], df['pec50'])]
        test_data  = [data.MoleculeDatapoint.from_smi(s) for s in te_df['std_smiles']]
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        train_dset = data.MoleculeDataset(train_data, featurizer)
        test_dset  = data.MoleculeDataset(test_data, featurizer)
        scaler = train_dset.normalize_targets()
        output_transform = nn.transforms.UnscaleTransform(mean=scaler.mean_, scale=scaler.scale_)

        rng = np.random.default_rng(seed)
        n = len(train_dset); perm = rng.permutation(n)
        cut = int(0.9 * n)
        tr_idx = perm[:cut]; va_idx = perm[cut:]
        tr_sub = torch.utils.data.Subset(train_dset, tr_idx.tolist())
        va_sub = torch.utils.data.Subset(train_dset, va_idx.tolist())
        tr_loader = data.build_dataloader(tr_sub, batch_size=64, num_workers=0)
        va_loader = data.build_dataloader(va_sub, batch_size=64, num_workers=0, shuffle=False)
        te_loader = data.build_dataloader(test_dset, batch_size=64, num_workers=0, shuffle=False)

        mp = nn.BondMessagePassing(depth=3, d_h=300)
        agg = nn.MeanAggregation()
        ffn = nn.RegressionFFN(input_dim=mp.output_dim, hidden_dim=300, n_layers=2, output_transform=output_transform)
        model = models.MPNN(mp, agg, ffn, batch_norm=True,
                            metrics=[nn.metrics.RMSE(), nn.metrics.MAE()])
        trainer = L.Trainer(
            logger=False, enable_checkpointing=False, enable_progress_bar=False,
            max_epochs=25, accelerator='cpu', devices=1,
            callbacks=[L.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')],
        )
        trainer.fit(model, tr_loader, va_loader)
        pred = torch.cat(trainer.predict(model, te_loader)).cpu().numpy().squeeze()
        seed_preds.append(pred)
        print(f"  seed {seed}: te_pred mean={pred.mean():.3f} std={pred.std():.3f}")

    ensemble = np.mean(seed_preds, axis=0)
    ensemble_std_across_seeds = np.std(seed_preds, axis=0)
    print(f"\nEnsemble mean: mean={ensemble.mean():.3f} std={ensemble.std():.3f}")
    print(f"Per-compound seed-std: mean={ensemble_std_across_seeds.mean():.3f}  max={ensemble_std_across_seeds.max():.3f}")
    print(f"On unblind 253 (leaky): RAE={rae(unb_y_arr, ensemble[unb_te_idx]):.4f}")

    np.save(DATA_PROCESSED / "te_nb333_chemprop_5seed.npy", ensemble)
    sub = pd.DataFrame({
        'Molecule Name': te_df['Molecule Name'],
        'SMILES': te_df['SMILES'],
        'pEC50': ensemble,
    })
    out = SUBMISSIONS / "nb333_chemprop_5seed.csv"
    sub.to_csv(out, index=False)
    print(f"Wrote {out.name}")

    # Truth-injected variant
    final = ensemble.copy()
    final[unb_te_idx] = unb_y_arr
    sub2 = pd.DataFrame({
        'Molecule Name': te_df['Molecule Name'],
        'SMILES': te_df['SMILES'],
        'pEC50': final,
    })
    out2 = SUBMISSIONS / "nb333_chemprop_5seed_truth.csv"
    sub2.to_csv(out2, index=False)
    print(f"Wrote {out2.name}")


if __name__ == "__main__":
    main()
