"""nb283 -- RNN on SMILES + Transformer MPNN on graphs (fusion).

Two-branch model:
  - BiGRU on character-tokenized SMILES (256-dim embed)
  - Graph TransformerConv MPNN (128-dim embed)
  - Fusion MLP -> pec50

Scaffold 5-fold CV, MAE loss, Adam. Outputs OOF + test arrays for blending.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGLoader
from torch_geometric.nn import TransformerConv, global_mean_pool

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.paths import DATA_PROCESSED

torch.manual_seed(42); np.random.seed(42)
DEVICE = torch.device("cpu")

# ---------------- SMILES tokenization ----------------
SMILES_CHARS = list("#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTUVWXYZ[\\]abcdefgilmnoprstuy")
CHAR2IDX = {c: i + 2 for i, c in enumerate(SMILES_CHARS)}  # 0 = pad, 1 = unk
VOCAB_SIZE = len(SMILES_CHARS) + 2
MAX_LEN = 150


def tokenize_smiles(smi: str):
    ids = [CHAR2IDX.get(c, 1) for c in smi[:MAX_LEN]]
    ids = ids + [0] * (MAX_LEN - len(ids))
    return ids


# ---------------- Graph featurization ----------------
ATOM_NUMS = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]  # H,B,C,N,O,F,Si,P,S,Cl,Br,I
HYBRIDS = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
           Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
           Chem.rdchem.HybridizationType.SP3D2]
BOND_TYPES = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]


def one_hot(val, choices):
    v = [0] * (len(choices) + 1)
    if val in choices:
        v[choices.index(val)] = 1
    else:
        v[-1] = 1
    return v


def atom_features(atom):
    feats = []
    feats += one_hot(atom.GetAtomicNum(), ATOM_NUMS)                 # 13
    feats += one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5])           # 7
    feats += one_hot(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])      # 6
    feats += one_hot(atom.GetHybridization(), HYBRIDS)               # 6
    feats += [int(atom.GetIsAromatic())]                              # 1
    feats += one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])           # 6
    return feats


NODE_DIM = 13 + 7 + 6 + 6 + 1 + 6  # = 39
EDGE_DIM = len(BOND_TYPES) + 1     # = 5


def bond_features(bond):
    return one_hot(bond.GetBondType(), BOND_TYPES)


def mol_to_pyg(smi, y=None):
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol is None or mol.GetNumAtoms() == 0:
        # fallback single dummy atom graph
        x = torch.zeros((1, NODE_DIM), dtype=torch.float32)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, EDGE_DIM), dtype=torch.float32)
        tok = torch.tensor(tokenize_smiles(smi or ""), dtype=torch.long)
        d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, tokens=tok.unsqueeze(0),
                 valid=torch.tensor([0], dtype=torch.float32))
        if y is not None:
            d.y = torch.tensor([y], dtype=torch.float32)
        return d
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)
    src, dst, eattr = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_features(b)
        src += [i, j]; dst += [j, i]
        eattr += [bf, bf]
    if len(src) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, EDGE_DIM), dtype=torch.float32)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(eattr, dtype=torch.float32)
    tok = torch.tensor(tokenize_smiles(smi), dtype=torch.long)
    d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
             tokens=tok.unsqueeze(0), valid=torch.tensor([1], dtype=torch.float32))
    if y is not None:
        d.y = torch.tensor([y], dtype=torch.float32)
    return d


# ---------------- Model ----------------
class RNNTransformerMPNN(nn.Module):
    def __init__(self, vocab=VOCAB_SIZE, node_dim=NODE_DIM, edge_dim=EDGE_DIM,
                 rnn_hidden=128, gnn_hidden=128, heads=4, gnn_layers=3):
        super().__init__()
        # RNN branch
        self.embed = nn.Embedding(vocab, 64, padding_idx=0)
        self.gru = nn.GRU(64, rnn_hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.1)
        # Graph branch
        self.node_proj = nn.Linear(node_dim, gnn_hidden)
        self.gnn_layers = nn.ModuleList()
        for _ in range(gnn_layers):
            self.gnn_layers.append(
                TransformerConv(gnn_hidden, gnn_hidden // heads, heads=heads,
                                edge_dim=edge_dim, dropout=0.1, beta=True)
            )
        # Fusion (RNN: 2*rnn_hidden=256, GNN: gnn_hidden=128)
        self.head = nn.Sequential(
            nn.Linear(2 * rnn_hidden + gnn_hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, data):
        # tokens shape: [B, MAX_LEN]
        tok = data.tokens
        emb = self.embed(tok)
        # pack-pad would be ideal; for simplicity use mask-mean
        out, _ = self.gru(emb)  # [B, L, 2*H]
        mask = (tok != 0).float().unsqueeze(-1)
        rnn_emb = (out * mask).sum(1) / mask.sum(1).clamp(min=1)

        # Graph branch
        x = self.node_proj(data.x)
        for layer in self.gnn_layers:
            x = layer(x, data.edge_index, edge_attr=data.edge_attr)
            x = F.relu(x)
        g = global_mean_pool(x, data.batch)

        z = torch.cat([rnn_emb, g], dim=1)
        return self.head(z).squeeze(-1)


# ---------------- Train / Eval ----------------
def train_fold(train_ds, val_ds, te_ds, epochs=60, batch_size=64, lr=1e-3):
    train_loader = PyGLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = PyGLoader(val_ds, batch_size=batch_size, shuffle=False)
    te_loader = PyGLoader(te_ds, batch_size=batch_size, shuffle=False)
    model = RNNTransformerMPNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val = float("inf"); best_state = None
    for ep in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            pred = model(batch)
            loss = F.l1_loss(pred, batch.y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        # quick val
        model.eval()
        with torch.no_grad():
            vps, vys = [], []
            for batch in val_loader:
                batch = batch.to(DEVICE)
                vps.append(model(batch).cpu().numpy()); vys.append(batch.y.cpu().numpy())
        vp = np.concatenate(vps); vy = np.concatenate(vys)
        vmae = np.abs(vp - vy).mean()
        if vmae < best_val:
            best_val = vmae
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    # final predict
    model.eval()
    with torch.no_grad():
        vps = []
        for batch in val_loader:
            batch = batch.to(DEVICE); vps.append(model(batch).cpu().numpy())
        val_pred = np.concatenate(vps)
        tps = []
        for batch in te_loader:
            batch = batch.to(DEVICE); tps.append(model(batch).cpu().numpy())
        te_pred = np.concatenate(tps)
    return val_pred, te_pred, best_val


def main():
    print("=== nb283: RNN + Transformer MPNN ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    y = tr["pec50"].values.astype(np.float32)
    smi_tr = tr["std_smiles"].tolist()
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")

    def std(s):
        try:
            m = Chem.MolFromSmiles(str(s)); return Chem.MolToSmiles(m) if m else str(s)
        except Exception:
            return str(s)
    smi_te = [std(s) for s in te_df["SMILES"].tolist()]

    print(f"Train: {len(smi_tr)}, Test: {len(smi_te)}")
    print(f"Node feat dim={NODE_DIM}, Edge feat dim={EDGE_DIM}, Vocab={VOCAB_SIZE}")

    print("Building PyG graphs...")
    t0 = time.time()
    tr_data = [mol_to_pyg(s, float(y[i])) for i, s in enumerate(smi_tr)]
    te_data = [mol_to_pyg(s, 0.0) for s in smi_te]
    print(f"  built in {time.time()-t0:.1f}s")

    folds = scaffold_kfold_indices(tr["scaffold"].tolist(), n_splits=5)
    EPOCHS = 20  # CPU time budget

    oof = np.zeros(len(y), dtype=np.float32)
    te_preds = []
    for fi, (ti, vi) in enumerate(folds):
        t0 = time.time()
        train_ds = [tr_data[i] for i in ti]
        val_ds = [tr_data[i] for i in vi]
        vp, tp, best_val = train_fold(train_ds, val_ds, te_data, epochs=EPOCHS)
        oof[vi] = vp; te_preds.append(tp)
        fold_rae = rae(y[vi], vp)
        print(f"  fold {fi}: best_val_mae={best_val:.4f} fold_rae={fold_rae:.4f} time={time.time()-t0:.1f}s", flush=True)
        # checkpoint after each fold
        np.save(DATA_PROCESSED / "oof_nb283_rnn_transformer_mpnn.npy", oof)
        np.save(DATA_PROCESSED / "te_nb283_rnn_transformer_mpnn.npy", np.mean(te_preds, axis=0))
    te_pred = np.mean(te_preds, axis=0)

    oof_rae = rae(y, oof)
    print(f"\nOOF RAE: {oof_rae:.4f}")
    print(f"te_pred: mean={te_pred.mean():.3f} std={te_pred.std():.3f} min={te_pred.min():.3f} max={te_pred.max():.3f}")

    np.save(DATA_PROCESSED / "oof_nb283_rnn_transformer_mpnn.npy", oof)
    np.save(DATA_PROCESSED / "te_nb283_rnn_transformer_mpnn.npy", te_pred)

    # ---------------- SLSQP blend ----------------
    print("\n=== 5-way SLSQP blend: nb224 + nb179s + mtd + loso + nb283 ===")
    from scipy.optimize import minimize
    nb224 = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
    nb179s = np.load(DATA_PROCESSED / "oof_nb179_stack.npy")
    mtd = np.load(DATA_PROCESSED / "oof_multi_template_delta.npy")
    loso = np.load(DATA_PROCESSED / "oof_delta_loso.npy")
    M = np.column_stack([nb224, nb179s, mtd, loso, oof])
    y64 = y.astype(np.float64)

    def loss(w):
        return rae(y64, M @ w)

    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0.0, 1.0)] * 5
    best = None
    for seed in range(100):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(5))
        res = minimize(loss, w0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"ftol": 1e-9, "maxiter": 200})
        if best is None or res.fun < best.fun:
            best = res
    print(f"5-way SLSQP OOF RAE: {best.fun:.4f}")
    labels = ["nb224", "nb179s", "mtd", "loso", "nb283"]
    for lab, w in zip(labels, best.x):
        print(f"  {lab}: {w:.4f}")
    print(f"\nnb283 weight: {best.x[4]:.4f}")


if __name__ == "__main__":
    main()
