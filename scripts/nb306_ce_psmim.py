"""nb306 -- Conformer-Ensemble Pocket-Shape Multi-Instance Model (CE-PSMIM).

Idea D: each compound is a BAG of K conformers. Each conformer has shape
descriptors. An attention-MIL aggregator learns which conformer matters.

Lightweight CPU version (so it can finish in <30 min):
  - K=8 conformers per compound (not 20)
  - Per-conformer shape features: PMI ratios, asphericity, eccentricity,
    radius_of_gyration, NPR1/NPR2, surface area (from RDKit)
  - Attention-MIL: gated attention pool over (n_atoms, n_conformers, d_shape)
  - Fuses with combined (Morgan+RDKit) base features.

This is the first model that explicitly treats compounds as ensembles of
poses rather than fixed feature vectors.
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
from rdkit.Chem import AllChem, Descriptors3D
from scipy.stats import spearmanr
from scipy.optimize import minimize

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED


K_CONF = 8
D_SHAPE = 10  # per-conformer shape features


def conformer_shape_features(smi, k_conf=K_CONF, seed=42):
    """Return (K_CONF, D_SHAPE) shape features per conformer; pad with zeros if fewer."""
    feats = np.zeros((k_conf, D_SHAPE), dtype=np.float32)
    if smi is None: return feats
    m = Chem.MolFromSmiles(smi)
    if m is None: return feats
    m = Chem.AddHs(m)
    try:
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        cids = AllChem.EmbedMultipleConfs(m, numConfs=k_conf, params=params)
        if len(cids) == 0:
            # Single ETKDG attempt
            if AllChem.EmbedMolecule(m, randomSeed=seed) != 0:
                return feats
            try: AllChem.MMFFOptimizeMolecule(m, maxIters=50)
            except: pass
            cids = [0]
        else:
            try: AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=50, numThreads=1)
            except: pass
    except Exception:
        return feats

    for ci, conf_id in enumerate(cids[:k_conf]):
        try:
            f = [
                Descriptors3D.Asphericity(m, confId=conf_id),
                Descriptors3D.Eccentricity(m, confId=conf_id),
                Descriptors3D.RadiusOfGyration(m, confId=conf_id),
                Descriptors3D.SpherocityIndex(m, confId=conf_id),
                Descriptors3D.PMI1(m, confId=conf_id),
                Descriptors3D.PMI2(m, confId=conf_id),
                Descriptors3D.PMI3(m, confId=conf_id),
                Descriptors3D.NPR1(m, confId=conf_id),
                Descriptors3D.NPR2(m, confId=conf_id),
                Descriptors3D.InertialShapeFactor(m, confId=conf_id),
            ]
            feats[ci] = np.array(f, dtype=np.float32)
        except Exception:
            pass
    return feats


class AttnMIL(nn.Module):
    """Per-compound bag-of-conformers attention pool + regression head."""
    def __init__(self, d_base, d_shape=D_SHAPE, d_emb=64, dropout=0.2):
        super().__init__()
        self.conf_enc = nn.Sequential(
            nn.Linear(d_shape, d_emb), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_emb, d_emb), nn.ReLU(),
        )
        # Gated attention
        self.attn_V = nn.Linear(d_emb, d_emb)
        self.attn_U = nn.Linear(d_emb, d_emb)
        self.attn_w = nn.Linear(d_emb, 1)

        self.base_enc = nn.Sequential(
            nn.Linear(d_base, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + d_emb, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, base_x, conf_x):
        # conf_x: (B, K, D_SHAPE)
        h = self.conf_enc(conf_x)  # (B, K, d_emb)
        # Gated attention
        a = torch.tanh(self.attn_V(h)) * torch.sigmoid(self.attn_U(h))
        a = self.attn_w(a).squeeze(-1)  # (B, K)
        a = F.softmax(a, dim=1)
        bag = (h * a.unsqueeze(-1)).sum(dim=1)  # (B, d_emb)
        b = self.base_enc(base_x)
        z = torch.cat([b, bag], dim=-1)
        return self.head(z).squeeze(-1)


def main():
    print("=== nb306: CE-PSMIM (Conformer-Ensemble Pocket-Shape MIL) ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    y = tr['pec50'].values.astype(np.float64)
    smiles_tr = tr['std_smiles'].tolist()
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    smiles_te = te_df['SMILES'].tolist()

    print(f"Computing {K_CONF}-conformer shape features...")
    conf_tr = np.zeros((len(smiles_tr), K_CONF, D_SHAPE), dtype=np.float32)
    conf_te = np.zeros((len(smiles_te), K_CONF, D_SHAPE), dtype=np.float32)
    n_fail_tr = 0; n_fail_te = 0
    for i, s in enumerate(smiles_tr):
        c = conformer_shape_features(s)
        if c.sum() == 0: n_fail_tr += 1
        conf_tr[i] = c
        if (i + 1) % 500 == 0: print(f"  train {i+1}/{len(smiles_tr)} (fail={n_fail_tr})")
    for i, s in enumerate(smiles_te):
        c = conformer_shape_features(s)
        if c.sum() == 0: n_fail_te += 1
        conf_te[i] = c
    print(f"Train conf fails: {n_fail_tr}/{len(smiles_tr)}  Test: {n_fail_te}/{len(smiles_te)}")

    # Standardise shape features (per dim, across train)
    mask_valid = conf_tr.sum(axis=(1, 2)) > 0
    mu_s = conf_tr[mask_valid].reshape(-1, D_SHAPE).mean(0)
    sd_s = conf_tr[mask_valid].reshape(-1, D_SHAPE).std(0) + 1e-6
    conf_tr = ((conf_tr - mu_s) / sd_s).clip(-5, 5)
    conf_te = ((conf_te - mu_s) / sd_s).clip(-5, 5)

    print("\nFeaturising base...")
    X_tr = impute(combined(smiles_tr)).astype(np.float32)
    X_te = impute(combined(smiles_te)).astype(np.float32)
    mu = X_tr.mean(0); sd = X_tr.std(0) + 1e-6
    X_tr = ((X_tr - mu) / sd).clip(-5, 5).astype(np.float32)
    X_te = ((X_te - mu) / sd).clip(-5, 5).astype(np.float32)

    folds = scaffold_kfold_indices(tr['scaffold'].tolist(), n_splits=5)
    oof = np.zeros(len(y))
    te_preds = []
    device = 'cpu'

    for fi, (ti, vi) in enumerate(folds):
        print(f"\n--- Fold {fi+1}/5 ---")
        model = AttnMIL(d_base=X_tr.shape[1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        Xtr_t = torch.tensor(X_tr[ti]); Ctr_t = torch.tensor(conf_tr[ti])
        ytr_t = torch.tensor(y[ti], dtype=torch.float32)
        Xva_t = torch.tensor(X_tr[vi]); Cva_t = torch.tensor(conf_tr[vi])
        yva_t = torch.tensor(y[vi], dtype=torch.float32)
        Xte_t = torch.tensor(X_te); Cte_t = torch.tensor(conf_te)
        n_tr = len(ti); B = 128
        best_val = float('inf'); best_state = None
        for ep in range(40):
            model.train()
            perm = np.random.permutation(n_tr)
            losses = []
            for i in range(0, n_tr, B):
                ii = perm[i:i+B]
                opt.zero_grad()
                p = model(Xtr_t[ii], Ctr_t[ii])
                loss = F.smooth_l1_loss(p, ytr_t[ii])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())
            model.eval()
            with torch.no_grad():
                vp = model(Xva_t, Cva_t)
                vloss = F.smooth_l1_loss(vp, yva_t).item()
            if vloss < best_val:
                best_val = vloss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if (ep + 1) % 10 == 0:
                print(f"  ep{ep+1} tr_loss={np.mean(losses):.3f}  val_loss={vloss:.3f}")

        if best_state is not None: model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            vp = model(Xva_t, Cva_t).numpy()
            tp = model(Xte_t, Cte_t).numpy()
        oof[vi] = vp
        te_preds.append(tp)

    te_pred = np.mean(te_preds, axis=0)
    r = rae(y, oof)
    sp, _ = spearmanr(y, oof)
    print(f"\nCE-PSMIM OOF: RAE={r:.4f}  Spearman={sp:.4f}  te_std={te_pred.std():.3f}")
    np.save(DATA_PROCESSED / "oof_nb306_cepsmim.npy", oof)
    np.save(DATA_PROCESSED / "te_nb306_cepsmim.npy", te_pred)

    nb224 = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
    nb179s = np.load(DATA_PROCESSED / "oof_nb179_stack.npy")
    mtd = np.load(DATA_PROCESSED / "oof_multi_template_delta.npy")
    loso = np.load(DATA_PROCESSED / "oof_delta_loso.npy")
    M = np.column_stack([nb224, nb179s, mtd, loso, oof])
    def loss_fn(w): return rae(y, M @ w)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0, 1.0)] * 5
    best = None
    for seed in range(80):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(5))
        res = minimize(loss_fn, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-9})
        if best is None or res.fun < best.fun: best = res
    print(f"\n5-way SLSQP: {best.fun:.4f}, weight(nb306)={best.x[4]:.4f}")


if __name__ == "__main__":
    main()
