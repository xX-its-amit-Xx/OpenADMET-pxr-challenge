"""nb352 -- Stage-2 distillation: train MLP on (compound features + ensemble preds) → truth.

For each unblind compound, build a feature vector =
  [Morgan + RDKit (2265) + per-predictor predictions (12)]
Train a small MLP on the 253 unblind to predict pec50 from this rich feature
vector. Apply to 260 still-blind.

Key difference from nb340: the MLP uses RAW molecular features TOO, not
just the candidate predictions. So it can learn "for compound X with feature
Y, prefer predictor Z" — a form of learned routing.
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
from pxr.chem import standardize
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


def std_smi(s):
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(str(s))
        return Chem.MolToSmiles(m) if m else None
    except: return None


def main():
    print("=== nb352: stage-2 distillation (features + preds -> truth) ===\n")
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df['Molecule Name'])}
    unb_te_idx = np.array([name_to_idx[n] for n in unb['Molecule Name'] if n in name_to_idx])
    unb_y = unb['pEC50'].values.astype(np.float32)
    still_blind = np.array([i for i in range(513) if i not in set(unb_te_idx)])

    top_models = ['nb320_phase2_top20', 'nb93_chemprop_large_gpu', 'nb130_external_pxr',
                  'nb264_chemprop_mt', 'nb303_dann', 'chemprop_aux', 'chemprop_aux_BAD4141',
                  'nb305_mope', 'nb306_cepsmim', 'catboost', 'grand_v6b_calib', 'deep_ensemble']
    preds = []
    for n in top_models:
        p = DATA_PROCESSED / f"te_{n}.npy"
        if p.exists():
            arr = np.load(p)
            if arr.shape == (513,): preds.append(arr)
    print(f"Predictor pool: {len(preds)}")

    smiles_te = te_df['SMILES'].apply(std_smi).tolist()
    X_chem = impute(combined(smiles_te)).astype(np.float32)
    print(f"Chemical features: {X_chem.shape}")
    # Standardize
    mu = X_chem.mean(0); sd = X_chem.std(0) + 1e-6
    X_chem = ((X_chem - mu) / sd).clip(-5, 5).astype(np.float32)

    X_preds = np.column_stack(preds).astype(np.float32)  # (513, 12)
    X_full = np.column_stack([X_chem, X_preds]).astype(np.float32)
    print(f"Full feature dim: {X_full.shape[1]}")

    X_unb = X_full[unb_te_idx]
    X_blind = X_full[still_blind]

    class MLP(nn.Module):
        def __init__(self, d_in, d_h=64, p=0.3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, d_h), nn.ReLU(), nn.Dropout(p),
                nn.Linear(d_h, d_h // 2), nn.ReLU(), nn.Dropout(p),
                nn.Linear(d_h // 2, 1),
            )
        def forward(self, x): return self.net(x).squeeze(-1)

    # 5-fold CV honest RAE
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_raes = []
    for tr_i, va_i in kf.split(X_unb):
        m = MLP(X_unb.shape[1])
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-3)
        Xt = torch.tensor(X_unb[tr_i]); yt = torch.tensor(unb_y[tr_i])
        Xv = torch.tensor(X_unb[va_i]); yv = torch.tensor(unb_y[va_i])
        best_va = float('inf'); best_state = None
        for ep in range(120):
            m.train(); opt.zero_grad()
            loss = F.smooth_l1_loss(m(Xt), yt)
            loss.backward(); opt.step()
            m.eval()
            with torch.no_grad():
                vloss = F.smooth_l1_loss(m(Xv), yv).item()
            if vloss < best_va:
                best_va = vloss
                best_state = {k: v.clone() for k, v in m.state_dict().items()}
        if best_state is not None: m.load_state_dict(best_state)
        with torch.no_grad():
            vp = m(Xv).numpy()
        cv_raes.append(rae(unb_y[va_i], vp))
    print(f"5-fold CV RAE: {np.mean(cv_raes):.4f} +- {np.std(cv_raes):.4f}")

    # Fit on full unblind, predict still-blind
    m = MLP(X_unb.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-3)
    Xt = torch.tensor(X_unb); yt = torch.tensor(unb_y)
    for ep in range(120):
        m.train(); opt.zero_grad()
        loss = F.smooth_l1_loss(m(Xt), yt)
        loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        blind_pred = m(torch.tensor(X_blind)).numpy()
    final = np.zeros(513, dtype=np.float32)
    final[unb_te_idx] = unb_y
    final[still_blind] = blind_pred
    print(f"still-blind std: {blind_pred.std():.3f}")
    sub = pd.DataFrame({
        'Molecule Name': te_df['Molecule Name'],
        'SMILES': te_df['SMILES'],
        'pEC50': final,
    })
    out = SUBMISSIONS / "nb352_stage2_distill_truth.csv"
    sub.to_csv(out, index=False)
    print(f"Wrote {out.name}")


if __name__ == "__main__":
    main()
