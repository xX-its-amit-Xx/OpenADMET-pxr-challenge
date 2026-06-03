"""nb920 -- SchNet-style equivariant 3D GNN (hand-rolled SchNet-lite).

Pipeline:
  1) Embed 3D conformer for each train+test SMILES (EmbedMolecule + MMFFOptimize,
     seed=42, single conformer). 2D-distance fallback if embed fails.
  2) Atom features: atomic-number one-hot (16 common elements) + formal charge
     + num explicit H  =>  16 + 1 + 1 = 18 dims.
  3) Pairwise distance Gaussian basis (n_rbf=16, cutoff=5 A).
  4) 3 message-passing layers: gather neighbors with d_ij < cutoff, MLP on
     (atom_i, atom_j, rbf(d_ij)) -> 128-d, sum-reduce -> residual update.
  5) Mean pool over atoms -> 64 -> 1.
  6) Huber loss, Adam lr=5e-4, 50 epochs, 4139 CRC. Score on 253 unblind names.
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

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
from rdkit import RDLogger

from pxr.data import load_train
from pxr.eval import rae
from pxr.paths import DATA_RAW, DATA_PROCESSED, SUBMISSIONS

RDLogger.DisableLog("rdApp.*")

TAG = "nb920"
SEED = 42
CUTOFF = 5.0
N_RBF = 16
D_ATOM = 18
D_HIDDEN = 128
D_FFN = 64
N_LAYERS = 3
EPOCHS = 50
LR = 5e-4
BATCH_GRAPHS = 32
HUBER_DELTA = 1.0

ART = Path("C:/pxr_artifacts")
ART.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cpu")

# 16 common organic + drug-like elements; everything else hits the last bin.
ELEMENT_LIST = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 26, 29, 30, 35, 53, 0]
EL2IDX = {z: i for i, z in enumerate(ELEMENT_LIST)}


def _featurize_mol(smi: str):
    """Return (X, coords) numpy arrays or None on failure."""
    if not isinstance(smi, str) or not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    coords_ok = False
    try:
        if AllChem.EmbedMolecule(mol, randomSeed=SEED) == 0:
            try:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
            except Exception:
                pass
            coords_ok = True
    except Exception:
        coords_ok = False

    if coords_ok:
        conf = mol.GetConformer()
        coords = np.array(
            [[conf.GetAtomPosition(i).x,
              conf.GetAtomPosition(i).y,
              conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())],
            dtype=np.float32,
        )
    else:
        # 2D distance fallback: compute graph distance via shortest path and
        # spread atoms along a line scaled by topological distance.
        mol = Chem.RemoveHs(Chem.MolFromSmiles(smi))
        if mol is None:
            return None
        n = mol.GetNumAtoms()
        if n == 0:
            return None
        dm = Chem.GetDistanceMatrix(mol).astype(np.float32)  # graph distances
        # PCA-like placement: use first row as 1-D coordinate scaled by 1.5 A.
        coords = np.zeros((n, 3), dtype=np.float32)
        coords[:, 0] = dm[0] * 1.5

    n_atoms = mol.GetNumAtoms()
    X = np.zeros((n_atoms, D_ATOM), dtype=np.float32)
    for i, atom in enumerate(mol.GetAtoms()):
        z = atom.GetAtomicNum()
        idx = EL2IDX.get(z, EL2IDX[0])
        X[i, idx] = 1.0
        X[i, 16] = float(atom.GetFormalCharge())
        X[i, 17] = float(atom.GetTotalNumHs())
    return X.astype(np.float32), coords.astype(np.float32)


def _std_smi(s):
    if not isinstance(s, str):
        return ""
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m is not None else s


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class GaussianBasis(nn.Module):
    def __init__(self, n_rbf=N_RBF, cutoff=CUTOFF):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, n_rbf)
        self.register_buffer("centers", centers)
        self.gamma = 1.0 / (centers[1] - centers[0]).item() ** 2

    def forward(self, d):  # d: (E,)
        return torch.exp(-self.gamma * (d.unsqueeze(-1) - self.centers) ** 2)


class SchNetLayer(nn.Module):
    def __init__(self, d_hidden=D_HIDDEN, n_rbf=N_RBF):
        super().__init__()
        self.mlp_e = nn.Sequential(
            nn.Linear(2 * d_hidden + n_rbf, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
        )
        self.upd = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
        )

    def forward(self, h, src, dst, rbf, n_atoms):
        m = self.mlp_e(torch.cat([h[src], h[dst], rbf], dim=-1))  # (E, d)
        agg = torch.zeros(n_atoms, h.size(-1), device=h.device, dtype=h.dtype)
        agg.index_add_(0, dst, m)
        return h + self.upd(agg)


class SchNetLite(nn.Module):
    def __init__(self, d_in=D_ATOM, d_hidden=D_HIDDEN, n_layers=N_LAYERS,
                 d_ffn=D_FFN, n_rbf=N_RBF, cutoff=CUTOFF):
        super().__init__()
        self.embed = nn.Linear(d_in, d_hidden)
        self.rbf = GaussianBasis(n_rbf=n_rbf, cutoff=cutoff)
        self.layers = nn.ModuleList(
            [SchNetLayer(d_hidden=d_hidden, n_rbf=n_rbf) for _ in range(n_layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(d_hidden, d_ffn),
            nn.SiLU(),
            nn.Linear(d_ffn, 1),
        )

    def forward(self, X, src, dst, dist, batch, n_graphs):
        h = self.embed(X)
        rbf = self.rbf(dist)
        for layer in self.layers:
            h = layer(h, src, dst, rbf, X.size(0))
        # mean pool per graph
        pooled = torch.zeros(n_graphs, h.size(-1), device=h.device, dtype=h.dtype)
        cnt = torch.zeros(n_graphs, device=h.device, dtype=h.dtype)
        pooled.index_add_(0, batch, h)
        cnt.index_add_(0, batch, torch.ones(h.size(0), device=h.device, dtype=h.dtype))
        pooled = pooled / cnt.clamp_min(1.0).unsqueeze(-1)
        return self.head(pooled).squeeze(-1)


# ----------------------------------------------------------------------------
# Batch builder
# ----------------------------------------------------------------------------
def _build_edges(coords, cutoff=CUTOFF):
    n = coords.shape[0]
    if n <= 1:
        return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.float32))
    diff = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt((diff * diff).sum(-1) + 1e-12)
    mask = (d < cutoff) & (d > 1e-6)
    src, dst = np.where(mask)
    return src.astype(np.int64), dst.astype(np.int64), d[mask].astype(np.float32)


def collate(graphs, indices):
    Xs, srcs, dsts, dists, batch = [], [], [], [], []
    off = 0
    for k, gi in enumerate(indices):
        X, src, dst, dist = graphs[gi]
        Xs.append(X)
        srcs.append(src + off)
        dsts.append(dst + off)
        dists.append(dist)
        batch.append(np.full(X.shape[0], k, dtype=np.int64))
        off += X.shape[0]
    X = torch.from_numpy(np.concatenate(Xs, axis=0))
    src = torch.from_numpy(np.concatenate(srcs, axis=0))
    dst = torch.from_numpy(np.concatenate(dsts, axis=0))
    dist = torch.from_numpy(np.concatenate(dists, axis=0))
    b = torch.from_numpy(np.concatenate(batch, axis=0))
    return X, src, dst, dist, b, len(indices)


# ----------------------------------------------------------------------------
def featurize_all(smiles_list, tag):
    cache = ART / f"{TAG}_graphs_{tag}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        graphs = list(z["graphs"])
        n_fail = int(z["n_fail"])
        print(f"  loaded {len(graphs)} cached graphs ({n_fail} fallbacks) <- {cache.name}")
        return graphs, n_fail

    graphs = []
    n_fail = 0
    t0 = time.time()
    for i, smi in enumerate(smiles_list):
        feat = _featurize_mol(smi)
        if feat is None:
            # last-resort dummy: single carbon atom
            X = np.zeros((1, D_ATOM), dtype=np.float32)
            X[0, EL2IDX[6]] = 1.0
            coords = np.zeros((1, 3), dtype=np.float32)
            n_fail += 1
        else:
            X, coords = feat
        src, dst, dist = _build_edges(coords)
        graphs.append((X, src, dst, dist))
        if (i + 1) % 500 == 0:
            print(f"    [{tag}] {i+1}/{len(smiles_list)} "
                  f"({time.time()-t0:.1f}s, {n_fail} fallbacks)")
    np.savez_compressed(cache, graphs=np.array(graphs, dtype=object),
                        n_fail=n_fail)
    print(f"  cached -> {cache.name} ({n_fail} fallbacks)")
    return graphs, n_fail


def main() -> dict:
    print("=" * 78)
    print(f"{TAG} -- SchNet-lite 3D GNN (cutoff={CUTOFF} A, "
          f"d={D_HIDDEN}, layers={N_LAYERS}, ep={EPOCHS})")
    print("=" * 78)

    needed = {
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED":    DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
    }
    miss = [k for k, p in needed.items() if not Path(p).exists()]
    if miss:
        print("MISSING raw inputs:", miss)
        return {"success": False, "missing": miss}

    tr = load_train()
    te_df = pd.read_csv(needed["TEST_BLINDED"])
    smiles_tr = [_std_smi(s) for s in tr["smiles"].tolist()]
    y_tr = tr["pec50"].astype(np.float32).values
    smiles_te = [_std_smi(s) for s in te_df["SMILES"].tolist()]
    print(f"Train n={len(smiles_tr)}  Test n={len(smiles_te)}")

    print("Embedding 3D conformers (train)...")
    g_tr, fail_tr = featurize_all(smiles_tr, "train")
    print("Embedding 3D conformers (test)...")
    g_te, fail_te = featurize_all(smiles_te, "test")
    print(f"  embed fallbacks: train={fail_tr}  test={fail_te}")
    n_atoms_tr = np.array([g[0].shape[0] for g in g_tr])
    print(f"  atoms/graph train: mean={n_atoms_tr.mean():.1f}  "
          f"min={n_atoms_tr.min()}  max={n_atoms_tr.max()}")

    model = SchNetLite().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.HuberLoss(delta=HUBER_DELTA)

    n_tr = len(g_tr)
    idx_all = np.arange(n_tr)
    y_tr_t = torch.from_numpy(y_tr.astype(np.float32))

    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        rng = np.random.default_rng(SEED + ep)
        rng.shuffle(idx_all)
        running = 0.0
        n_batches = 0
        for k in range(0, n_tr, BATCH_GRAPHS):
            batch_idx = idx_all[k:k + BATCH_GRAPHS]
            X, src, dst, dist, b, ng = collate(g_tr, batch_idx)
            X = X.to(DEVICE); src = src.to(DEVICE); dst = dst.to(DEVICE)
            dist = dist.to(DEVICE); b = b.to(DEVICE)
            yhat = model(X, src, dst, dist, b, ng)
            y = y_tr_t[batch_idx].to(DEVICE)
            loss = loss_fn(yhat, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += float(loss.item())
            n_batches += 1
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  ep {ep+1:02d}/{EPOCHS}  huber={running/max(n_batches,1):.4f}  "
                  f"t={time.time()-t0:.1f}s")

    print("Predicting on test...")
    model.eval()
    preds = np.zeros(len(g_te), dtype=np.float32)
    with torch.no_grad():
        for k in range(0, len(g_te), BATCH_GRAPHS):
            batch_idx = np.arange(k, min(k + BATCH_GRAPHS, len(g_te)))
            X, src, dst, dist, b, ng = collate(g_te, batch_idx)
            X = X.to(DEVICE); src = src.to(DEVICE); dst = dst.to(DEVICE)
            dist = dist.to(DEVICE); b = b.to(DEVICE)
            preds[batch_idx] = model(X, src, dst, dist, b, ng).cpu().numpy()
    print(f"  pred mean={preds.mean():.3f}  std={preds.std():.3f}  "
          f"min={preds.min():.3f}  max={preds.max():.3f}")

    # ---- Unblind scoring ---------------------------------------------------
    te_names = te_df["Molecule Name"].tolist()
    n2i = {n: i for i, n in enumerate(te_names)}
    unb = pd.read_csv(needed["UNBLINDED"])
    unb = unb[unb["Molecule Name"].isin(n2i)].reset_index(drop=True)
    unb_idx = np.array([n2i[n] for n in unb["Molecule Name"]], dtype=int)
    unb_y = unb["pEC50"].astype(float).values.astype(np.float64)
    rae_u = float(rae(unb_y, preds[unb_idx]))

    print("=" * 78)
    print(f"UNBLIND (n={len(unb_idx)}): RAE = {rae_u:.4f}")
    print(f"  truth mean/std = {unb_y.mean():.3f} / {unb_y.std():.3f}")
    print(f"  pred  mean/std = {preds[unb_idx].mean():.3f} / "
          f"{preds[unb_idx].std():.3f}")
    print("=" * 78)

    # ---- Save artifacts ----------------------------------------------------
    torch.save(model.state_dict(), ART / f"{TAG}_schnet.pt")
    np.save(ART / f"{TAG}_te_pred.npy", preds)
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", preds)

    sub = SUBMISSIONS / f"{TAG}_schnet_3d.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES":        te_df["SMILES"],
        "pEC50":         preds,
    }).to_csv(sub, index=False)
    print(f"Wrote {sub}")

    return {
        "success": True,
        "in_rae": rae_u,
        "n_unblind": int(len(unb_idx)),
        "fail_train": int(fail_tr),
        "fail_test": int(fail_te),
        "submission": str(sub),
    }


if __name__ == "__main__":
    out = main()
    print("\nRESULT:", out)
