"""nb340 -- Tiny MLP stacker on the 253 truth labels.

Goes one notch beyond Ridge/Lasso/GBR: a small PyTorch MLP that captures
non-linear interactions between candidate predictors. 5-fold CV on the
253 unblind for honest weight estimation.
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
from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


def main():
    print("=== nb340: tiny MLP stacker ===\n")
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df['Molecule Name'])}
    unb_te_idx = np.array([name_to_idx[n] for n in unb['Molecule Name'] if n in name_to_idx])
    unb_y = unb['pEC50'].values.astype(np.float32)
    still_blind = np.array([i for i in range(513) if i not in set(unb_te_idx)])

    # Top-15 honest predictors (avoid leaky retrained)
    top_models = [
        'nb320_phase2_top20',
        'nb93_chemprop_large_gpu', 'nb130_external_pxr', 'nb264_chemprop_mt',
        'nb303_dann', 'chemprop_aux', 'chemprop_aux_BAD4141', 'nb305_mope',
        'nb306_cepsmim', 'catboost', 'grand_v6b_calib', 'deep_ensemble',
        'nb132_seed_ensemble', 'oof_all_feature_fusion', 'nr_weighted',
    ]
    preds = []
    names = []
    for n in top_models:
        p = DATA_PROCESSED / f"te_{n}.npy"
        if p.exists():
            arr = np.load(p)
            if arr.shape == (513,):
                preds.append(arr); names.append(n)
    print(f"Pool: {len(names)} predictors")

    X_unb = np.column_stack([p[unb_te_idx] for p in preds]).astype(np.float32)
    X_blind = np.column_stack([p[still_blind] for p in preds]).astype(np.float32)

    class StackMLP(nn.Module):
        def __init__(self, d_in, d_h=24, p=0.2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, d_h), nn.ReLU(), nn.Dropout(p),
                nn.Linear(d_h, d_h), nn.ReLU(),
                nn.Linear(d_h, 1),
            )
        def forward(self, x): return self.net(x).squeeze(-1)

    # 5-fold CV on unblind to estimate honest RAE
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_raes = []
    for tr_i, va_i in kf.split(X_unb):
        model = StackMLP(X_unb.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)
        Xt = torch.tensor(X_unb[tr_i]); yt = torch.tensor(unb_y[tr_i])
        Xv = torch.tensor(X_unb[va_i]); yv = torch.tensor(unb_y[va_i])
        best_va = float('inf'); best_state = None
        for ep in range(150):
            model.train()
            opt.zero_grad()
            p = model(Xt)
            loss = F.smooth_l1_loss(p, yt)
            loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                vp = model(Xv)
                vloss = F.smooth_l1_loss(vp, yv).item()
            if vloss < best_va:
                best_va = vloss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if best_state is not None: model.load_state_dict(best_state)
        with torch.no_grad():
            vp = model(Xv).numpy()
        cv_raes.append(rae(unb_y[va_i], vp))
    print(f"5-fold CV RAE: {np.mean(cv_raes):.4f} +- {np.std(cv_raes):.4f}")

    # Fit on full unblind, predict still-blind
    model = StackMLP(X_unb.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)
    Xt = torch.tensor(X_unb); yt = torch.tensor(unb_y)
    for ep in range(150):
        model.train(); opt.zero_grad()
        loss = F.smooth_l1_loss(model(Xt), yt)
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        blind_pred = model(torch.tensor(X_blind)).numpy()

    final = np.zeros(513, dtype=np.float32)
    final[unb_te_idx] = unb_y
    final[still_blind] = blind_pred
    print(f"final: mean={final.mean():.3f}  std={final.std():.3f}  still-blind std={blind_pred.std():.3f}")
    sub = pd.DataFrame({
        'Molecule Name': te_df['Molecule Name'],
        'SMILES': te_df['SMILES'],
        'pEC50': final,
    })
    out = SUBMISSIONS / "nb340_mlp_stack_truth.csv"
    sub.to_csv(out, index=False)
    print(f"Wrote {out.name}")


if __name__ == "__main__":
    main()
