"""nb303 -- Test-Set-Aware Adversarial Domain Adaptation (TS-ADA).

Idea A from the deep-thinking exercise: train an encoder + pec50 regressor
jointly with a TRAIN-vs-TEST discriminator that receives REVERSED gradients
on its way back to the encoder. The encoder must produce features that
(1) predict pec50 on labelled train data, and (2) cannot be used by the
discriminator to distinguish train compounds from the 513 unlabelled test
compounds. This is "DANN" (Ganin & Lempitsky, 2015) applied to PXR.

Mechanism: the 0.46 OOF->LB gap is biological — test compounds are OOD
analog expansions. Standard models maximally fit the train manifold;
DANN forces them to use only signal that *also* lives on the test manifold.
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
from torch.autograd import Function
from scipy.stats import spearmanr
from scipy.optimize import minimize

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED


# Gradient reversal layer (Ganin & Lempitsky 2015)
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.clone()
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lam * grad_output, None

def grad_reverse(x, lam=1.0):
    return GradReverse.apply(x, lam)


class DANN(nn.Module):
    def __init__(self, d_in, d_h=256, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, d_h), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_h, d_h // 2), nn.ReLU(), nn.Dropout(dropout),
        )
        self.regressor = nn.Sequential(
            nn.Linear(d_h // 2, d_h // 4), nn.ReLU(),
            nn.Linear(d_h // 4, 1),
        )
        # Domain discriminator: train vs test (binary)
        self.discriminator = nn.Sequential(
            nn.Linear(d_h // 2, d_h // 4), nn.ReLU(),
            nn.Linear(d_h // 4, 1),
        )

    def forward(self, x, lam=1.0):
        z = self.encoder(x)
        y = self.regressor(z).squeeze(-1)
        d = self.discriminator(grad_reverse(z, lam)).squeeze(-1)
        return y, d, z


def main():
    print("=== nb303: Test-Set Adversarial Domain Adaptation ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    y = tr['pec50'].values.astype(np.float64)
    smiles_tr = tr['std_smiles'].tolist()
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    smiles_te = te_df['SMILES'].tolist()

    print("Featurising...")
    X_tr = impute(combined(smiles_tr)).astype(np.float32)
    X_te = impute(combined(smiles_te)).astype(np.float32)
    # Standardize using TRAIN+TEST stats — both domains shape the feature space
    X_all_for_stats = np.vstack([X_tr, X_te])
    mu = X_all_for_stats.mean(0); sd = X_all_for_stats.std(0) + 1e-6
    X_tr_n = ((X_tr - mu) / sd).clip(-5, 5).astype(np.float32)
    X_te_n = ((X_te - mu) / sd).clip(-5, 5).astype(np.float32)

    device = 'cpu'
    folds = scaffold_kfold_indices(tr['scaffold'].tolist(), n_splits=5)
    oof = np.zeros(len(y))
    te_preds = []

    Xte_t = torch.tensor(X_te_n)
    yte_dom = torch.ones(len(X_te_n))  # test = 1
    yte_dom_label = torch.ones(len(X_te_n))

    for fi, (ti, vi) in enumerate(folds):
        print(f"\n--- Fold {fi+1}/5 ---")
        Xtr_t = torch.tensor(X_tr_n[ti])
        Xva_t = torch.tensor(X_tr_n[vi])
        ytr_t = torch.tensor(y[ti], dtype=torch.float32)
        yva_t = torch.tensor(y[vi], dtype=torch.float32)
        ytr_dom = torch.zeros(len(ti))  # train = 0

        model = DANN(X_tr.shape[1], d_h=256).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

        n_tr = len(ti)
        B = 256
        EPOCHS = 50
        best_val = float('inf'); best_state = None
        for ep in range(EPOCHS):
            model.train()
            # Lambda schedule: ramp adversarial weight 0 -> 0.3 over training
            p = ep / EPOCHS
            lam = (2.0 / (1.0 + np.exp(-5 * p)) - 1.0) * 0.3

            perm = np.random.permutation(n_tr)
            losses_r = []; losses_d = []
            for i in range(0, n_tr, B):
                ii = perm[i:i+B]
                # Combine batch of train + matching size from test
                te_idx = np.random.choice(len(X_te_n), len(ii), replace=False)
                X_batch = torch.cat([Xtr_t[ii], Xte_t[te_idx]], dim=0)
                y_batch = ytr_t[ii]
                d_batch = torch.cat([torch.zeros(len(ii)), torch.ones(len(te_idx))])

                opt.zero_grad()
                # Forward: regressor only sees train half
                y_pred, d_pred, _ = model(X_batch, lam=lam)
                y_pred_train = y_pred[:len(ii)]
                loss_r = F.smooth_l1_loss(y_pred_train, y_batch)
                loss_d = F.binary_cross_entropy_with_logits(d_pred, d_batch)
                loss = loss_r + loss_d
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses_r.append(loss_r.item()); losses_d.append(loss_d.item())

            model.eval()
            with torch.no_grad():
                vp, _, _ = model(Xva_t, lam=0.0)
                vloss = F.smooth_l1_loss(vp, yva_t).item()
            if vloss < best_val:
                best_val = vloss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if (ep + 1) % 10 == 0:
                vmae = (vp.numpy() - y[vi]).__abs__().mean()
                print(f"  ep{ep+1} lam={lam:.3f} lr={np.mean(losses_r):.3f} ld={np.mean(losses_d):.3f} val_mae={vmae:.3f}")

        if best_state is not None: model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            vp, _, _ = model(Xva_t, lam=0.0)
            tp, _, _ = model(Xte_t, lam=0.0)
        oof[vi] = vp.numpy()
        te_preds.append(tp.numpy())

    te_pred = np.mean(te_preds, axis=0)
    r = rae(y, oof)
    sp, _ = spearmanr(y, oof)
    print(f"\nDANN OOF: RAE={r:.4f}  Spearman={sp:.4f}  te_std={te_pred.std():.3f}")
    np.save(DATA_PROCESSED / "oof_nb303_dann.npy", oof)
    np.save(DATA_PROCESSED / "te_nb303_dann.npy", te_pred)

    # SLSQP 5-way
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
    print(f"\n5-way SLSQP OOF: {best.fun:.4f}, weight(nb303)={best.x[4]:.4f}")


if __name__ == "__main__":
    main()
