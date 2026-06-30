"""nb1132 — ADVANCED architectures (the ones not to abandon): custom torch_geometric GNN + TabNet, scaffold-CV.

Applies the 'schedule, don't abandon' rule: train a real graph neural net (GINE on RDKit molecular graphs) and TabNet
(attention deep-tabular) from scratch, scaffold-CV -> clean OOF (4139) + 513-test predictions, cached to
C:/pxr_work/search/ for the ensemble (added like the ChemProp GNN). Resumable: skips a model if its cache exists.
Run in background. Each model cached separately so partial success is useful.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test, load_train
from src.pxr.eval import rae, scaffold_kfold_indices
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

P = "data/processed"; SD = "C:/pxr_work/search"; os.makedirs(SD, exist_ok=True)


def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""


# ---------- torch_geometric GNN ----------
def mol_to_graph(smi):
    import torch
    from torch_geometric.data import Data
    m = Chem.MolFromSmiles(str(smi))
    if m is None: return None
    af = []
    for a in m.GetAtoms():
        af.append([a.GetAtomicNum(), a.GetDegree(), a.GetFormalCharge(), int(a.GetHybridization()),
                   int(a.GetIsAromatic()), a.GetTotalNumHs(), int(a.IsInRing())])
    ei, ef = [], []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bt = [int(b.GetBondTypeAsDouble()), int(b.GetIsConjugated()), int(b.IsInRing())]
        ei += [[i, j], [j, i]]; ef += [bt, bt]
    if not ei: ei = [[0, 0]]; ef = [[0, 0, 0]]
    return Data(x=torch.tensor(af, dtype=torch.float), edge_index=torch.tensor(ei, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(ef, dtype=torch.float))


def train_gnn(smis, y, smte):
    import torch, torch.nn as nn
    from torch_geometric.nn import GINEConv, global_mean_pool
    from torch_geometric.loader import DataLoader
    graphs = [mol_to_graph(s) for s in smis]; ok = [i for i, g in enumerate(graphs) if g is not None]
    for i in ok: graphs[i].y = torch.tensor([y[i]], dtype=torch.float)
    tegraphs = [mol_to_graph(s) for s in smte]

    class GNN(nn.Module):
        def __init__(self, h=128):
            super().__init__()
            self.lin = nn.Linear(7, h); self.elin = nn.Linear(3, h)
            self.c1 = GINEConv(nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h)))
            self.c2 = GINEConv(nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h)))
            self.c3 = GINEConv(nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h)))
            self.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.2), nn.Linear(h, 1))
        def forward(self, d):
            x = self.lin(d.x); e = self.elin(d.edge_attr)
            x = torch.relu(self.c1(x, d.edge_index, e)); x = torch.relu(self.c2(x, d.edge_index, e))
            x = torch.relu(self.c3(x, d.edge_index, e)); return self.head(global_mean_pool(x, d.batch)).squeeze(-1)

    def fit_predict(tr_idx, pred_graphs):
        m = GNN(); opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5); lossf = nn.SmoothL1Loss()
        dl = DataLoader([graphs[i] for i in tr_idx if graphs[i] is not None], batch_size=64, shuffle=True)
        for ep in range(40):
            m.train()
            for batch in dl:
                opt.zero_grad(); loss = lossf(m(batch), batch.y); loss.backward(); opt.step()
        m.eval(); out = np.full(len(pred_graphs), np.nan)
        pl = DataLoader([g if g is not None else graphs[ok[0]] for g in pred_graphs], batch_size=128)
        preds = []
        with torch.no_grad():
            for batch in pl: preds.append(m(batch).numpy())
        return np.concatenate(preds)

    scaf = [murcko(s) for s in smis]
    oof = np.zeros(len(smis))
    for trn, val in scaffold_kfold_indices(scaf, n_splits=5, seed=42):
        oof[val] = fit_predict(trn, [graphs[i] for i in val])
    te_pred = np.mean([fit_predict(np.arange(len(smis)), tegraphs) for _ in range(2)], 0)
    return oof, te_pred


# ---------- TabNet ----------
def train_tabnet(X, y, Xte):
    from pytorch_tabnet.tab_model import TabNetRegressor
    scaf_idx = None
    def fp(tr, va_X):
        m = TabNetRegressor(verbose=0, seed=0)
        m.fit(X[tr], y[tr].reshape(-1, 1), max_epochs=120, patience=20, batch_size=512, eval_set=[], eval_metric=["mae"])
        return m.predict(va_X).ravel()
    return fp


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test().reset_index(drop=True)
    y = tr["pec50"].to_numpy(); smis = tr["smiles"].tolist(); smte = te["smiles"].tolist()
    scaf = [murcko(s) for s in smis]

    # GNN
    if not os.path.exists(f"{SD}/gnn_oof.npy"):
        print("training torch_geometric GINE GNN (scaffold-CV)...", flush=True)
        try:
            oof, tep = train_gnn(smis, y, smte)
            np.save(f"{SD}/gnn_oof.npy", oof); np.save(f"{SD}/gnn_te.npy", tep)
            print(f"  GNN scaffold-CV RAE {rae(y, oof):.4f} -> cached gnn_oof/gnn_te", flush=True)
        except Exception as e:
            print(f"  GNN FAILED: {str(e)[:120]}", flush=True)
    else:
        print(f"GNN cached: scaffold-CV RAE {rae(y, np.load(f'{SD}/gnn_oof.npy')):.4f}")

    # TabNet (combined+chempropembed)
    if not os.path.exists(f"{SD}/tabnet_oof.npy"):
        print("training TabNet (scaffold-CV)...", flush=True)
        try:
            from src.pxr.featurize import combined, impute
            X = np.hstack([impute(combined(smis)), np.load(f"{P}/tr_chemprop_embed_300.npy")]).astype(np.float32)
            Xte = np.hstack([impute(combined(smte)), np.load(f"{P}/te_chemprop_embed_300.npy")]).astype(np.float32)
            from pytorch_tabnet.tab_model import TabNetRegressor
            oof = np.zeros(len(y))
            for trn, val in scaffold_kfold_indices(scaf, n_splits=5, seed=42):
                m = TabNetRegressor(verbose=0, seed=0)
                m.fit(X[trn], y[trn].reshape(-1, 1), max_epochs=120, patience=20, batch_size=512)
                oof[val] = m.predict(X[val]).ravel()
            mt = TabNetRegressor(verbose=0, seed=0); mt.fit(X, y.reshape(-1, 1), max_epochs=120, patience=20, batch_size=512)
            tep = mt.predict(Xte).ravel()
            np.save(f"{SD}/tabnet_oof.npy", oof); np.save(f"{SD}/tabnet_te.npy", tep)
            print(f"  TabNet scaffold-CV RAE {rae(y, oof):.4f} -> cached tabnet_oof/tabnet_te", flush=True)
        except Exception as e:
            print(f"  TabNet FAILED: {str(e)[:120]}", flush=True)
    else:
        print(f"TabNet cached: scaffold-CV RAE {rae(y, np.load(f'{SD}/tabnet_oof.npy')):.4f}")
    print("DONE. Add gnn_oof/gnn_te + tabnet_oof/tabnet_te to the nb1130 ensemble pool.")


if __name__ == "__main__":
    main()
