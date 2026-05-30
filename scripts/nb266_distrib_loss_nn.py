"""nb266 -- Distribution-matching loss neural network.

User idea: optimize for the prediction DISTRIBUTION matching the training
label distribution, with point-wise MAE as secondary.

Architecture: 3-layer MLP on combined features.
Loss = alpha * MAE(pred, y) + (1-alpha) * Wasserstein1(sorted(pred), sorted(y))

Wasserstein1 on sorted prediction vs sorted label forces the cumulative
distribution of predictions to match. Even if pointwise errors exist, the
overall distribution shape is preserved.

This is meaningful for LB because: if our test set distribution should look
like train (similar pec50 spread), the model is regularized to predict
within that distribution.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import Chem

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED


def std_smi(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol) if mol else None
    except: return None


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(1024, 512, 256), dropout=0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def wasserstein1_sorted(pred, y):
    """Wasserstein-1 distance via sorting (since 1D)."""
    pred_sorted = torch.sort(pred)[0]
    y_sorted = torch.sort(y)[0]
    return torch.abs(pred_sorted - y_sorted).mean()


def hybrid_loss(pred, y, alpha=0.7):
    """alpha * MAE + (1-alpha) * Wasserstein1."""
    mae = torch.abs(pred - y).mean()
    wass = wasserstein1_sorted(pred, y)
    return alpha * mae + (1 - alpha) * wass, mae.item(), wass.item()


def train_fold(X_tr, y_tr, X_va, y_va, device, epochs=50, alpha=0.7):
    model = MLP(X_tr.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xt = torch.tensor(X_tr, dtype=torch.float32).to(device)
    Yt = torch.tensor(y_tr, dtype=torch.float32).to(device)
    Xv = torch.tensor(X_va, dtype=torch.float32).to(device)
    Yv = torch.tensor(y_va, dtype=torch.float32).to(device)

    best_val = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    batch_size = 256
    n = X_tr.shape[0]
    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(n)
        for i in range(0, n, batch_size):
            b = idx[i:i+batch_size]
            opt.zero_grad()
            pred = model(Xt[b])
            loss, _, _ = hybrid_loss(pred, Yt[b], alpha)
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xv)
            val_mae = torch.abs(pv - Yv).mean().item()
        if not np.isnan(val_mae) and val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def main():
    print("=== nb266: Distribution-matching loss MLP ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    y_tr = tr["pec50"].values.astype(np.float32)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["SMILES"].apply(std_smi).tolist()

    X_tr = combined(smiles_tr); X_tr = impute(X_tr).astype(np.float32)
    X_te = combined(smiles_te); X_te = impute(X_te).astype(np.float32)
    print(f"X_tr: {X_tr.shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    folds = scaffold_kfold_indices(tr["scaffold"].tolist(), n_splits=5)

    # Try different alpha values (MAE-only to pure-Wasserstein)
    for alpha in [1.0, 0.7, 0.5, 0.3]:
        print(f"\n=== alpha={alpha} (MAE weight) ===")
        oof = np.zeros(len(y_tr), dtype=np.float32)
        te_preds = []
        for fold_i, (ti, vi) in enumerate(folds):
            model = train_fold(X_tr[ti], y_tr[ti], X_tr[vi], y_tr[vi], device, epochs=40, alpha=alpha)
            model.eval()
            with torch.no_grad():
                oof[vi] = model(torch.tensor(X_tr[vi]).to(device)).cpu().numpy()
                te_preds.append(model(torch.tensor(X_te).to(device)).cpu().numpy())
        te_pred = np.mean(te_preds, axis=0)
        r = rae(y_tr, oof)
        print(f"  OOF RAE: {r:.4f}  te_mean={te_pred.mean():.3f} te_std={te_pred.std():.3f}")
        np.save(DATA_PROCESSED / f"oof_nb266_alpha{int(alpha*100):03d}.npy", oof)
        np.save(DATA_PROCESSED / f"te_nb266_alpha{int(alpha*100):03d}.npy", te_pred)

    # Stack best with 239
    print("\n=== Stack alpha=0.7 with 239 ===")
    from scipy.optimize import minimize
    nb224 = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
    nb179s = np.load(DATA_PROCESSED / "oof_nb179_stack.npy")
    mtd = np.load(DATA_PROCESSED / "oof_multi_template_delta.npy")
    loso = np.load(DATA_PROCESSED / "oof_delta_loso.npy")
    nb266 = np.load(DATA_PROCESSED / "oof_nb266_alpha070.npy")

    M = np.column_stack([nb224, nb179s, mtd, loso, nb266])
    def loss(w): return rae(y_tr.astype(np.float64), M @ w)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0, 1.0)] * 5
    best = None
    for seed in range(100):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(5))
        res = minimize(loss, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"ftol": 1e-9})
        if best is None or res.fun < best.fun: best = res
    print(f"5-way OOF: {best.fun:.4f}")
    for n, w in zip(['nb224', 'nb179s', 'mtd', 'loso', 'nb266'], best.x):
        print(f"  {n}: {w:.4f}")


if __name__ == "__main__":
    main()
