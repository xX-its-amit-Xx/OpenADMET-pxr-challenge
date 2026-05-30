"""nb294 -- Heteroscedastic NLL: per-row pEC50_SE weighting in the LOSS only.

User insight: down-weight noisy labels (high pEC50_SE) during training, but
NEVER use SE as a feature (collapses test preds). Train LGBM with sample_weight =
1/SE^2 (already attempted in nb281); here we also train an MLP that PREDICTS both
mean and variance from features, with Gaussian-NLL loss:
  L = (y - mu)^2 / (2*sigma^2) + 0.5*log(sigma^2)
This lets the model learn aleatoric uncertainty per-compound and use it to
recalibrate.
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
from scipy.stats import spearmanr
from scipy.optimize import minimize

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED


class HetMLP(nn.Module):
    def __init__(self, d_in, hidden=256, dropout=0.2):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout),
        )
        self.mu_head = nn.Linear(hidden//2, 1)
        self.logvar_head = nn.Linear(hidden//2, 1)
    def forward(self, x):
        z = self.body(x)
        mu = self.mu_head(z).squeeze(-1)
        logvar = self.logvar_head(z).squeeze(-1).clamp(-3, 3)  # var in [exp(-3), exp(3)]
        return mu, logvar


def gaussian_nll(mu, logvar, y):
    var = logvar.exp()
    return ((y - mu) ** 2 / (2 * var) + 0.5 * logvar).mean()


def main():
    print("=== nb294: Heteroscedastic NLL MLP ===\n")
    tr_csv = pd.read_csv('data/raw/pxr-challenge_TRAIN.csv')
    se_col = 'pEC50_std.error (-log10(molarity))'
    tr = load_train(); tr = add_standard_columns(tr)
    y = tr['pec50'].values.astype(np.float64)
    smiles_tr = tr['std_smiles'].tolist()
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    smiles_te = te_df['SMILES'].tolist()

    # Get SE for sample weighting
    name_to_se = dict(zip(tr_csv['Molecule Name'], tr_csv[se_col]))
    tr_names = tr['name'].tolist() if 'name' in tr.columns else tr['Molecule Name'].tolist()
    se = np.array([name_to_se.get(n, 0.24) for n in tr_names])
    se = np.clip(se, 0.05, 1.0)
    print(f"SE: median={np.median(se):.3f}")

    print("Featurising...")
    X_tr = impute(combined(smiles_tr)).astype(np.float32)
    X_te = impute(combined(smiles_te)).astype(np.float32)

    # Standardize
    mu_f = X_tr.mean(axis=0); sd_f = X_tr.std(axis=0) + 1e-6
    X_tr = ((X_tr - mu_f) / sd_f).clip(-5, 5).astype(np.float32)
    X_te = ((X_te - mu_f) / sd_f).clip(-5, 5).astype(np.float32)

    folds = scaffold_kfold_indices(tr['scaffold'].tolist(), n_splits=5)
    device = 'cpu'
    oof_mu = np.zeros(len(y))
    oof_logvar = np.zeros(len(y))
    te_preds = []

    for fi, (ti, vi) in enumerate(folds):
        print(f"\n--- Fold {fi+1}/5 ---")
        model = HetMLP(X_tr.shape[1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        Xt = torch.tensor(X_tr[ti]); yt = torch.tensor(y[ti], dtype=torch.float32)
        Xv = torch.tensor(X_tr[vi]); yv = torch.tensor(y[vi], dtype=torch.float32)
        Xte = torch.tensor(X_te)
        n_tr = len(ti); B = 256
        best_val = float('inf'); best_state = None
        for epoch in range(60):
            model.train()
            idx = np.random.permutation(n_tr)
            losses = []
            for i in range(0, n_tr, B):
                b = idx[i:i+B]
                opt.zero_grad()
                mu, lv = model(Xt[b])
                loss = gaussian_nll(mu, lv, yt[b])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())
            model.eval()
            with torch.no_grad():
                vmu, vlv = model(Xv)
                vloss = gaussian_nll(vmu, vlv, yv).item()
            if vloss < best_val:
                best_val = vloss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if (epoch + 1) % 20 == 0:
                rmae = (vmu.numpy() - y[vi]).__abs__().mean()
                print(f"  ep{epoch+1} train_nll={np.mean(losses):.3f} val_nll={vloss:.3f} val_mae={rmae:.3f}")

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            vmu, vlv = model(Xv)
            tmu, _ = model(Xte)
        oof_mu[vi] = vmu.numpy()
        oof_logvar[vi] = vlv.numpy()
        te_preds.append(tmu.numpy())

    te_pred = np.mean(te_preds, axis=0)
    r = rae(y, oof_mu)
    sp, _ = spearmanr(y, oof_mu)
    print(f"\nHetNLL MLP OOF: RAE={r:.4f}  Spearman={sp:.4f}  te_std={te_pred.std():.3f}")
    print(f"  logvar stats: min={oof_logvar.min():.3f}  max={oof_logvar.max():.3f}  mean={oof_logvar.mean():.3f}")

    np.save(DATA_PROCESSED / "oof_nb294_hetnll.npy", oof_mu)
    np.save(DATA_PROCESSED / "te_nb294_hetnll.npy", te_pred)
    np.save(DATA_PROCESSED / "oof_nb294_logvar.npy", oof_logvar)

    # SLSQP blend
    nb224 = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
    nb179s = np.load(DATA_PROCESSED / "oof_nb179_stack.npy")
    mtd = np.load(DATA_PROCESSED / "oof_multi_template_delta.npy")
    loso = np.load(DATA_PROCESSED / "oof_delta_loso.npy")
    M = np.column_stack([nb224, nb179s, mtd, loso, oof_mu])
    def loss_fn(w): return rae(y, M @ w)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0, 1.0)] * 5
    best = None
    for seed in range(100):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(5))
        res = minimize(loss_fn, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-9})
        if best is None or res.fun < best.fun: best = res
    print(f"\n5-way SLSQP: OOF {best.fun:.4f}, weight(nb294)={best.x[4]:.4f}")


if __name__ == "__main__":
    main()
