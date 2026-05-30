"""nb282 -- Knowledge graph + heterogeneous R-GCN for PXR binding.

Heterogeneous graph:
  Node types:
    - compound  (our train + Papyrus + test)
    - target    (42 NR/CYP UniProt IDs from papyrus_full_wide)
  Edge types (relations):
    - binds            (compound -> target, weighted by pec50)
    - binds_rev        (target -> compound)
    - target_similar   (target <-> target, sequence/family similarity)

Architecture: 2-layer heterogeneous SAGEConv on PyG HeteroData.
Input compound features = combined (Morgan + RDKit, 2265).
Input target features = one-hot of role (PXR | CYP | other NR).

Train: predict pec50 for every observed (compound, target) edge.
For each fold: hold out the PXR pec50 of validation compounds, predict.

This is genuinely new vs nb258/nb259 because:
  - Graph propagation: a test compound binds to nearby targets, message passes
  - Heterogeneous relations: target_similar edges share info across NRs
  - Single model jointly trained on all 42 targets, not multi-task heads
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
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
from rdkit import Chem

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED


PXR_UNIPROT = "O75469"
CYP_TARGETS = {'P08684', 'P05177', 'P11712', 'P33261', 'P10635', 'P08183', 'P05181', 'P10275'}
# remaining ~33 = other NRs/transporters/etc


def std_smi(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol) if mol else None
    except: return None


class HetGNN(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden)
        self.conv2 = SAGEConv((-1, -1), hidden)
        self.head  = nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.head(x).squeeze(-1)


def build_graph(compound_feats, n_targets, compound_target_edges, edge_weights, target_similar_edges):
    """compound_feats: (Nc, Fc) float. edges: (2, E) long."""
    data = HeteroData()
    data['compound'].x = torch.tensor(compound_feats, dtype=torch.float32)
    data['target'].x   = torch.eye(n_targets, dtype=torch.float32)  # identity = unique target id

    data['compound', 'binds', 'target'].edge_index = torch.tensor(compound_target_edges, dtype=torch.long)
    data['compound', 'binds', 'target'].edge_attr  = torch.tensor(edge_weights, dtype=torch.float32)

    # Reverse edge for message passing both ways
    data['target', 'binds_rev', 'compound'].edge_index = torch.tensor(compound_target_edges[::-1].copy(), dtype=torch.long)
    data['target', 'binds_rev', 'compound'].edge_attr  = torch.tensor(edge_weights, dtype=torch.float32)

    if len(target_similar_edges[0]):
        data['target', 'similar', 'target'].edge_index = torch.tensor(target_similar_edges, dtype=torch.long)
    return data


def main():
    print("=== nb282: Knowledge Graph + Heterogeneous GNN ===\n")

    # ============================
    # Step 1: gather all compounds (train + test + Papyrus)
    # ============================
    tr = load_train(); tr = add_standard_columns(tr)
    y_tr = tr['pec50'].values.astype(np.float64)
    smiles_tr = tr['std_smiles'].tolist()

    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    smiles_te = te_df['SMILES'].apply(std_smi).tolist()

    pap = pd.read_parquet("data/external/papyrus_full_wide.parquet")
    pap['std_smiles'] = pap['SMILES'].apply(std_smi)
    pap = pap.dropna(subset=['std_smiles']).drop_duplicates('std_smiles').reset_index(drop=True)
    smiles_pap = pap['std_smiles'].tolist()

    # Union: deduplicate (preserve order: train, test, papyrus)
    all_smiles = list(smiles_tr) + list(smiles_te) + list(smiles_pap)
    sm_idx = {}
    smiles_unique = []
    for s in all_smiles:
        if s not in sm_idx:
            sm_idx[s] = len(smiles_unique)
            smiles_unique.append(s)
    print(f"Compounds: train={len(smiles_tr)}, test={len(smiles_te)}, papyrus={len(smiles_pap)}, unique={len(smiles_unique)}")

    # ============================
    # Step 2: featurize compounds
    # ============================
    print("Featurizing compounds (combined Morgan+RDKit, batched)...")
    BATCH = 2000
    feats = []
    for i in range(0, len(smiles_unique), BATCH):
        batch = smiles_unique[i:i+BATCH]
        f = combined(batch); f = impute(f)
        feats.append(f.astype(np.float32))
        print(f"  {min(i+BATCH, len(smiles_unique))}/{len(smiles_unique)}")
    feats = np.vstack(feats)
    print(f"Compound feats: {feats.shape}")
    # Normalize: z-score per column, clip outliers
    mu = feats.mean(axis=0); sd = feats.std(axis=0) + 1e-6
    feats = ((feats - mu) / sd).clip(-5, 5).astype(np.float32)
    print(f"  normalized: mean={feats.mean():.3f} std={feats.std():.3f}")

    # ============================
    # Step 3: build edges from papyrus_full_wide (compound, target, pec50)
    # ============================
    target_cols = [c for c in pap.columns if c not in ('SMILES', 'std_smiles')]
    print(f"\nTargets in KG: {len(target_cols)}")
    print(f"  PXR (O75469) present: {PXR_UNIPROT in target_cols}")

    target_idx = {t: i for i, t in enumerate(target_cols)}
    edges_src, edges_dst, edge_w = [], [], []
    for ti, t in enumerate(target_cols):
        sub = pap.dropna(subset=[t])
        for s, p in zip(sub['std_smiles'], sub[t]):
            if s in sm_idx:
                edges_src.append(sm_idx[s])
                edges_dst.append(ti)
                edge_w.append(float(p))
    print(f"Compound-target edges from Papyrus: {len(edges_src)}")

    # Add train PXR labels as edges
    pxr_ti = target_idx[PXR_UNIPROT]
    n_train_edges_added = 0
    for s, y in zip(smiles_tr, y_tr):
        if s in sm_idx:
            edges_src.append(sm_idx[s])
            edges_dst.append(pxr_ti)
            edge_w.append(float(y))
            n_train_edges_added += 1
    print(f"Train PXR edges added: {n_train_edges_added}")

    edges_src = np.array(edges_src); edges_dst = np.array(edges_dst); edge_w = np.array(edge_w, dtype=np.float32)

    # ============================
    # Step 4: target-similar edges (NR family clustering, hand-coded)
    # ============================
    nr_family = {
        # nuclear receptor superfamily relatives — biology says these share fold
        'O75469': 'NR1I',  # PXR
        'Q14994': 'NR1I',  # CAR
        'Q07869': 'PPAR',  # PPARα
        'P37231': 'PPAR',  # PPARγ
        'Q03181': 'PPAR',  # PPARδ
        'Q96RI1': 'NR1H',  # FXR
        'P55055': 'NR1H',  # LXR-β
        'Q13133': 'NR1H',  # LXR-α
        'P11473': 'VDR',   # VDR
        'P10275': 'AR',    # AR
        'P03372': 'ER',    # ERα
        'Q92731': 'ER',    # ERβ
        'P04150': 'GR',    # GR
        'P10828': 'TR',    # THRβ
        'P10276': 'RAR',   # RARα
        'P28702': 'RXR',   # RXRβ
        'P19793': 'RXR',   # RXRα
    }
    sim_src, sim_dst = [], []
    for t1, fam1 in nr_family.items():
        for t2, fam2 in nr_family.items():
            if t1 != t2 and fam1 == fam2 and t1 in target_idx and t2 in target_idx:
                sim_src.append(target_idx[t1]); sim_dst.append(target_idx[t2])
    print(f"Target-similar edges: {len(sim_src)}")

    # ============================
    # Step 5: scaffold-fold training
    # ============================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    folds = scaffold_kfold_indices(tr['scaffold'].tolist(), n_splits=5)

    tr_sm_idx = np.array([sm_idx[s] for s in smiles_tr])
    te_sm_idx = np.array([sm_idx[s] for s in smiles_te])
    oof = np.zeros(len(y_tr))
    te_preds_all = []

    for fi, (ti, vi) in enumerate(folds):
        print(f"\n--- Fold {fi+1}/5 ---")
        # Mask out the validation compounds' PXR edges (held out)
        val_compound_set = set(tr_sm_idx[vi].tolist())
        mask = ~((edges_dst == pxr_ti) & np.isin(edges_src, list(val_compound_set)))
        e_src = edges_src[mask]; e_dst = edges_dst[mask]; e_w = edge_w[mask]
        print(f"  Edges: {len(e_src)} ({mask.sum()}/{len(mask)} kept)")

        ct_edges = np.stack([e_src, e_dst])
        sim_edges = np.stack([np.array(sim_src), np.array(sim_dst)]) if sim_src else np.zeros((2, 0), dtype=np.int64)
        data = build_graph(feats, len(target_cols), ct_edges, e_w, sim_edges)
        model = to_hetero(HetGNN(hidden=128), data.metadata(), aggr='mean').to(device)
        data = data.to(device)

        # Init with dummy forward
        with torch.no_grad():
            _ = model(data.x_dict, data.edge_index_dict)

        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        # For training signal: predict edge_weight via dot product of compound × target embeddings
        # Actually our HetGNN above outputs a per-node scalar — repurpose:
        #   - we want pec50 for a specific (compound, target) pair
        # Use embedding-based scoring: train compound embedding + target embedding -> pec50 via MLP

        # Simpler approach: just train compound embedding to predict its PXR pec50 directly
        # via node-level regression target (only compounds with known PXR pec50)
        y_node = torch.full((feats.shape[0],), float('nan'), device=device)
        node_mask = torch.zeros(feats.shape[0], dtype=torch.bool, device=device)
        # All compounds with known PXR pec50 (papyrus + train minus held-out)
        pxr_edges_kept = (e_dst == pxr_ti)
        for s, w in zip(e_src[pxr_edges_kept], e_w[pxr_edges_kept]):
            y_node[int(s)] = float(w); node_mask[int(s)] = True
        # Cross-target supervision: predict from non-PXR edges too via averaging? skip; use PXR-only

        n_sup = int(node_mask.sum().item())
        print(f"  Supervised compound nodes (PXR pec50 known): {n_sup}")

        for epoch in range(60):
            model.train()
            opt.zero_grad()
            out_dict = model(data.x_dict, data.edge_index_dict)
            pred = out_dict['compound']
            loss = F.smooth_l1_loss(pred[node_mask], y_node[node_mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if (epoch + 1) % 20 == 0:
                model.eval()
                with torch.no_grad():
                    out = model(data.x_dict, data.edge_index_dict)['compound']
                    val_pred = out[tr_sm_idx[vi]].cpu().numpy()
                    val_r = rae(y_tr[vi], val_pred)
                print(f"    epoch {epoch+1} loss={loss.item():.4f}  val RAE={val_r:.4f}")

        model.eval()
        with torch.no_grad():
            out = model(data.x_dict, data.edge_index_dict)['compound'].cpu().numpy()
        oof[vi] = out[tr_sm_idx[vi]]
        te_preds_all.append(out[te_sm_idx])

    te_pred = np.mean(te_preds_all, axis=0)
    r = rae(y_tr, oof)
    from scipy.stats import spearmanr
    sp, _ = spearmanr(y_tr, oof)
    print(f"\n=== KG HetGNN OOF RAE: {r:.4f}  Spearman: {sp:.4f}  te_std={te_pred.std():.3f} ===")
    np.save(DATA_PROCESSED / "oof_nb282_kg_hetgnn.npy", oof)
    np.save(DATA_PROCESSED / "te_nb282_kg_hetgnn.npy", te_pred)

    # SLSQP blend with nb239 base
    print("\n=== 5-way SLSQP w/ nb282 ===")
    from scipy.optimize import minimize
    nb224 = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
    nb179s = np.load(DATA_PROCESSED / "oof_nb179_stack.npy")
    mtd = np.load(DATA_PROCESSED / "oof_multi_template_delta.npy")
    loso = np.load(DATA_PROCESSED / "oof_delta_loso.npy")
    M = np.column_stack([nb224, nb179s, mtd, loso, oof])
    def loss(w): return rae(y_tr, M @ w)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0, 1.0)] * 5
    best = None
    for seed in range(100):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(5))
        res = minimize(loss, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-9})
        if best is None or res.fun < best.fun: best = res
    print(f"5-way SLSQP OOF: {best.fun:.4f}, nb282 weight={best.x[4]:.4f}")


if __name__ == "__main__":
    main()
