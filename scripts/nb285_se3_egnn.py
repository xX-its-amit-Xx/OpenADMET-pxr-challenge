"""nb285_se3_egnn.py — E(3)-equivariant GNN (SchNet) for PXR pEC50.

3D-aware regression. One conformer per molecule (ETKDG + MMFF). 5-fold
scaffold CV. Ends with SLSQP blend vs the nb239 4-way base (nb224 +
nb179_stack + multi_template_delta + delta_loso).
"""
from __future__ import annotations

import os
import sys
import time
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.optimize import minimize
from scipy.stats import spearmanr
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import SchNet

from pxr.chem import add_standard_columns
from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

RDLogger.DisableLog("rdApp.*")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[nb285] device = {DEVICE}")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Common atoms; "other" bucket at index 9
ATOM_TYPES = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
ATOM_IDX = {a: i for i, a in enumerate(ATOM_TYPES)}
N_ATOM_TYPES = len(ATOM_TYPES) + 1  # +1 for "other"
# Atomic numbers used as SchNet's `z` input (it expects element numbers)
ATOM_Z = {"C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16, "Cl": 17, "Br": 35, "I": 53}


def smiles_to_data(smi: str, y: float | None = None, cutoff: float = 5.0) -> Data | None:
    """Embed a single 3D conformer and return a PyG Data object, or None on failure."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = SEED
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None
    try:
        if AllChem.MMFFOptimizeMolecule(mol, maxIters=200) != 0:
            # not converged, but still usable
            pass
    except Exception:
        return None
    mol = Chem.RemoveHs(mol)
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    if n == 0:
        return None
    z = torch.zeros(n, dtype=torch.long)
    pos = torch.zeros(n, 3, dtype=torch.float32)
    for i, atom in enumerate(mol.GetAtoms()):
        sym = atom.GetSymbol()
        z[i] = ATOM_Z.get(sym, 6)  # fallback to C
        p = conf.GetAtomPosition(i)
        pos[i] = torch.tensor([p.x, p.y, p.z], dtype=torch.float32)
    # Bond edges
    src, dst = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        src += [i, j]
        dst += [j, i]
    # Radial cutoff edges
    d = torch.cdist(pos, pos)
    mask = (d < cutoff) & (d > 1e-6)
    ridx = mask.nonzero(as_tuple=False)
    src += ridx[:, 0].tolist()
    dst += ridx[:, 1].tolist()
    if len(src) == 0:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        # de-dupe
        ek = edge_index[0] * n + edge_index[1]
        _, uniq = torch.unique(ek, return_inverse=False, return_counts=False, sorted=True), None
        # simple unique via numpy
        ek_np = ek.numpy()
        keep = np.unique(ek_np, return_index=True)[1]
        edge_index = edge_index[:, sorted(keep.tolist())]
    data = Data(z=z, pos=pos, edge_index=edge_index)
    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float32)
    return data


def build_dataset(smiles, ys=None):
    out = []
    failed = []
    t0 = time.time()
    for i, smi in enumerate(smiles):
        y = None if ys is None else float(ys[i])
        d = smiles_to_data(smi, y)
        if d is None:
            failed.append(i)
            out.append(None)
        else:
            out.append(d)
        if (i + 1) % 500 == 0:
            print(f"  embed {i+1}/{len(smiles)} ({len(failed)} failed) [{time.time()-t0:.0f}s]")
    print(f"  done {len(smiles)} ({len(failed)} failed) in {time.time()-t0:.0f}s")
    return out, failed


def train_one_fold(train_data, val_data, epochs=50, lr=5e-4, bs=64, mean=0.0, std=1.0):
    model = SchNet(
        hidden_channels=128,
        num_filters=128,
        num_interactions=4,
        num_gaussians=50,
        cutoff=5.0,
        mean=mean,
        std=std,
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=bs, shuffle=False)
    best_val_mae = float("inf")
    best_state = None
    for ep in range(epochs):
        model.train()
        tloss = 0.0
        n = 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            pred = model(batch.z, batch.pos, batch.batch).view(-1)
            loss = F.l1_loss(pred, batch.y.view(-1))
            loss.backward()
            opt.step()
            tloss += loss.item() * batch.num_graphs
            n += batch.num_graphs
        tloss /= max(n, 1)
        # quick val every 5 ep
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            model.eval()
            preds, ys = [], []
            with torch.no_grad():
                for b in val_loader:
                    b = b.to(DEVICE)
                    p = model(b.z, b.pos, b.batch).view(-1).cpu().numpy()
                    preds.append(p)
                    ys.append(b.y.view(-1).cpu().numpy())
            preds = np.concatenate(preds)
            ys = np.concatenate(ys)
            vmae = float(np.mean(np.abs(preds - ys)))
            if vmae < best_val_mae:
                best_val_mae = vmae
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"    ep {ep+1:2d}  train MAE {tloss:.3f}  val MAE {vmae:.3f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict(model, data_list, bs=64):
    if len(data_list) == 0:
        return np.zeros(0, dtype=np.float32)
    loader = DataLoader(data_list, batch_size=bs, shuffle=False)
    model.eval()
    preds = []
    with torch.no_grad():
        for b in loader:
            b = b.to(DEVICE)
            p = model(b.z, b.pos, b.batch).view(-1).cpu().numpy()
            preds.append(p)
    return np.concatenate(preds)


def main():
    t_all = time.time()
    print("[nb285] loading train + test")
    tr = load_train()
    te = pd.read_csv(DATA_PROCESSED.parent / "raw" / "pxr-challenge_TEST_BLINDED.csv")
    te.columns = [c.lower().replace(" ", "_") for c in te.columns]
    print(f"  train {tr.shape}  test {te.shape}")

    # scaffolds for CV
    tr = add_standard_columns(tr, smi_col="smiles")
    y = tr["pec50"].to_numpy(dtype=np.float32)
    median_y = float(np.median(y))
    print(f"  median pec50 = {median_y:.3f}")

    print("[nb285] embedding train conformers")
    tr_data, tr_failed = build_dataset(tr["std_smiles"].tolist(), y)
    print("[nb285] embedding test conformers")
    te_data, te_failed = build_dataset(te["smiles"].tolist())

    # Normalize y for stability (SchNet's mean/std applied internally)
    y_mean = float(y[~np.isin(np.arange(len(y)), tr_failed)].mean())
    y_std = float(y[~np.isin(np.arange(len(y)), tr_failed)].std() + 1e-6)
    print(f"  y_mean={y_mean:.3f}  y_std={y_std:.3f}")

    scaffolds = tr["scaffold"].tolist()
    folds = scaffold_kfold_indices(scaffolds, n_splits=5, shuffle=True, seed=SEED)

    oof = np.full(len(tr), median_y, dtype=np.float32)
    te_preds = []  # per fold predictions, averaged at end

    for fi, (tr_idx, va_idx) in enumerate(folds):
        print(f"[nb285] fold {fi+1}/5  train={len(tr_idx)} val={len(va_idx)}")
        tr_ds = [tr_data[i] for i in tr_idx if tr_data[i] is not None]
        va_ds = [tr_data[i] for i in va_idx if tr_data[i] is not None]
        va_keep = [i for i in va_idx if tr_data[i] is not None]
        print(f"  usable train {len(tr_ds)}  val {len(va_ds)}")
        model = train_one_fold(tr_ds, va_ds, epochs=50, lr=5e-4, bs=64,
                               mean=y_mean, std=y_std)
        vp = predict(model, va_ds)
        oof[va_keep] = vp
        # test predictions for this fold (skip failed)
        te_keep = [i for i, d in enumerate(te_data) if d is not None]
        te_ds_ok = [te_data[i] for i in te_keep]
        tp_full = np.full(len(te_data), median_y, dtype=np.float32)
        tp_ok = predict(model, te_ds_ok)
        for k, idx in enumerate(te_keep):
            tp_full[idx] = tp_ok[k]
        te_preds.append(tp_full)
        fold_rae = rae(y[va_keep], vp)
        print(f"  fold {fi+1} val RAE = {fold_rae:.4f}")

    te_pred = np.mean(np.stack(te_preds, axis=0), axis=0)

    np.save(DATA_PROCESSED / "oof_nb285_se3_egnn.npy", oof)
    np.save(DATA_PROCESSED / "te_nb285_se3_egnn.npy", te_pred)

    oof_rae = rae(y, oof)
    rho, _ = spearmanr(y, oof)
    print(f"\n[nb285] OOF RAE = {oof_rae:.4f}   Spearman = {rho:.4f}")

    # ---- SLSQP blend vs nb239 base ----
    print("\n[nb285] SLSQP blend vs nb239 4-way base")
    nb224 = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
    nb179s = np.load(DATA_PROCESSED / "oof_nb179_stack.npy")
    mtd = np.load(DATA_PROCESSED / "oof_multi_template_delta.npy")
    loso = np.load(DATA_PROCESSED / "oof_delta_loso.npy")

    X = np.stack([nb224, nb179s, mtd, loso, oof], axis=1)
    names = ["nb224", "nb179_stack", "mtd", "loso", "nb285"]

    def obj(w):
        return rae(y, X @ w)

    w0 = np.ones(5) / 5
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bnds = [(0.0, 1.0)] * 5
    res = minimize(obj, w0, method="SLSQP", constraints=cons, bounds=bnds,
                   options={"maxiter": 500, "ftol": 1e-9})
    w = res.x
    blend_rae = rae(y, X @ w)

    print("  weights:")
    for n, wi in zip(names, w):
        print(f"    {n:14s} {wi:.4f}")
    print(f"  blend OOF RAE = {blend_rae:.4f}")
    print(f"  nb285 weight  = {w[-1]:.4f}")

    # base-only (4-way) for comparison
    Xb = X[:, :4]

    def obj_b(w):
        return rae(y, Xb @ w)

    res_b = minimize(obj_b, np.ones(4) / 4, method="SLSQP",
                     constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                     bounds=[(0.0, 1.0)] * 4, options={"maxiter": 500, "ftol": 1e-9})
    base_rae = rae(y, Xb @ res_b.x)
    print(f"  base 4-way RAE (no nb285) = {base_rae:.4f}  delta = {blend_rae - base_rae:+.4f}")

    print(f"\n[nb285] total wall time = {time.time()-t_all:.0f}s")


if __name__ == "__main__":
    main()
