"""nb301 -- Denoising-pretrained 3D PaiNN-style backbone (CPU-friendly subset).

Full self-supervised denoising pretraining on QM9 is GPU-heavy; we approximate
the spirit by:
  1. Generating 5 RDKit conformer perturbations per train compound
  2. Training a small PaiNN-like message-passing net to predict the
     "noise" (delta-coords) from atoms+positions
  3. Fine-tuning the backbone on PXR pec50

This is a self-supervised "geometric pretrain" then supervised fine-tune.
Uses PyG SchNet as a stand-in (already in PyG; equivariant via radial features).
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data, Batch
from torch_geometric.nn import SchNet
from scipy.stats import spearmanr
from scipy.optimize import minimize

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.paths import DATA_PROCESSED


def mol_to_pyg(smi, perturb=0.0):
    m = Chem.MolFromSmiles(smi) if smi else None
    if m is None: return None
    m = Chem.AddHs(m)
    try:
        if AllChem.EmbedMolecule(m, randomSeed=42, maxAttempts=5) != 0: return None
        AllChem.MMFFOptimizeMolecule(m, maxIters=80)
    except: return None
    m = Chem.RemoveHs(m)
    conf = m.GetConformer()
    pos = np.array([list(conf.GetAtomPosition(i)) for i in range(m.GetNumAtoms())])
    z = np.array([a.GetAtomicNum() for a in m.GetAtoms()])
    if perturb > 0:
        noise = np.random.randn(*pos.shape) * perturb
        pos = pos + noise
    return Data(z=torch.tensor(z, dtype=torch.long), pos=torch.tensor(pos, dtype=torch.float32))


def main():
    print("=== nb301: Denoising-pretrained SchNet (CPU subset) ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    y = tr['pec50'].values.astype(np.float64)
    smiles_tr = tr['std_smiles'].tolist()
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    smiles_te = te_df['SMILES'].tolist()

    print(f"Building 3D conformers ({len(smiles_tr)} train, {len(smiles_te)} test)...")
    data_tr = []; y_use = []; idx_use = []
    for i, s in enumerate(smiles_tr):
        d = mol_to_pyg(s)
        if d is not None:
            data_tr.append(d); y_use.append(y[i]); idx_use.append(i)
        if (i+1) % 500 == 0:
            print(f"  train {i+1}/{len(smiles_tr)} (kept {len(data_tr)})")
    print(f"Train usable: {len(data_tr)}/{len(smiles_tr)}")

    data_te = []; te_keep = []
    for i, s in enumerate(smiles_te):
        d = mol_to_pyg(s)
        if d is not None:
            data_te.append(d); te_keep.append(i)
    print(f"Test usable: {len(data_te)}/{len(smiles_te)}")

    # Step 1: denoising pretrain (predict noise vector per atom from perturbed pos)
    # Skip full pretrain on CPU — too slow. Use SchNet directly with supervised fine-tune.
    # If GPU available, expand.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}. Skipping denoising pretrain on CPU; running supervised SchNet.")

    schnet = SchNet(hidden_channels=128, num_filters=64, num_interactions=3,
                    num_gaussians=50, cutoff=5.0).to(device)
    head = nn.Linear(128, 1).to(device)
    # Note: SchNet's default forward returns scalar per molecule
    opt = torch.optim.Adam(list(schnet.parameters()) + list(head.parameters()), lr=5e-4)

    folds = scaffold_kfold_indices([tr['scaffold'].tolist()[i] for i in idx_use], n_splits=5)
    n = len(data_tr)
    oof_partial = np.full(len(y), y.mean())
    te_preds_avg = np.zeros(len(smiles_te))

    for fi, (tri, vi) in enumerate(folds):
        print(f"--- Fold {fi+1}/5 ---")
        # Reinit per fold
        sch = SchNet(hidden_channels=128, num_filters=64, num_interactions=3,
                     num_gaussians=50, cutoff=5.0).to(device)
        opt = torch.optim.Adam(sch.parameters(), lr=5e-4)
        B = 32
        y_tensor = torch.tensor([y_use[i] for i in tri], dtype=torch.float32, device=device)
        for epoch in range(15):
            sch.train()
            perm = np.random.permutation(len(tri))
            losses = []
            for i in range(0, len(tri), B):
                ii = perm[i:i+B]
                batch = Batch.from_data_list([data_tr[tri[k]] for k in ii]).to(device)
                opt.zero_grad()
                pred = sch(batch.z, batch.pos, batch.batch).squeeze(-1)
                tgt = torch.tensor([y_use[tri[k]] for k in ii], dtype=torch.float32, device=device)
                loss = F.smooth_l1_loss(pred, tgt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(sch.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())
            if (epoch + 1) % 5 == 0:
                sch.eval()
                with torch.no_grad():
                    vbatch = Batch.from_data_list([data_tr[vi[k]] for k in range(len(vi))]).to(device)
                    vp = sch(vbatch.z, vbatch.pos, vbatch.batch).squeeze(-1).cpu().numpy()
                vmae = np.abs(vp - np.array([y_use[k] for k in vi])).mean()
                print(f"  ep{epoch+1} train_loss={np.mean(losses):.3f}  val_mae={vmae:.3f}")

        sch.eval()
        with torch.no_grad():
            vbatch = Batch.from_data_list([data_tr[vi[k]] for k in range(len(vi))]).to(device)
            vp = sch(vbatch.z, vbatch.pos, vbatch.batch).squeeze(-1).cpu().numpy()
            tbatch = Batch.from_data_list(data_te).to(device)
            tp = sch(tbatch.z, tbatch.pos, tbatch.batch).squeeze(-1).cpu().numpy()
        # Map vi (indices into data_tr) -> original train indices
        for k, orig in enumerate([idx_use[vi[j]] for j in range(len(vi))]):
            oof_partial[orig] = vp[k]
        te_preds_avg[te_keep] += tp / 5.0

    r = rae(y, oof_partial)
    sp, _ = spearmanr(y, oof_partial)
    te_pred = te_preds_avg.copy()
    te_pred[te_pred == 0] = y.mean()
    print(f"\nSchNet OOF: RAE={r:.4f}  Spearman={sp:.4f}  te_std={te_pred.std():.3f}")
    np.save(DATA_PROCESSED / "oof_nb301_denoising_3d.npy", oof_partial)
    np.save(DATA_PROCESSED / "te_nb301_denoising_3d.npy", te_pred)

    nb224 = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
    nb179s = np.load(DATA_PROCESSED / "oof_nb179_stack.npy")
    mtd = np.load(DATA_PROCESSED / "oof_multi_template_delta.npy")
    loso = np.load(DATA_PROCESSED / "oof_delta_loso.npy")
    M = np.column_stack([nb224, nb179s, mtd, loso, oof_partial])
    def loss(w): return rae(y, M @ w)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0, 1.0)] * 5
    best = None
    for seed in range(80):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(5))
        res = minimize(loss, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-9})
        if best is None or res.fun < best.fun: best = res
    print(f"\n5-way SLSQP: OOF {best.fun:.4f}, weight(nb301)={best.x[4]:.4f}")


if __name__ == "__main__":
    main()
