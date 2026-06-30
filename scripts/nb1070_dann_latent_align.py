"""nb1070 — DANN v3: LATENT-SPACE ALIGNMENT of train<->blinded-test (user's idea, thorough, on nb3200 substrate).

Encoder(combined feats) -> latent; regressor head -> pEC50 (trained on 4139 train labels); domain discriminator
-> train(0) vs ALL-513-test(1) via Gradient Reversal Layer (Ganin 2015). GRL forces the encoder to make train and
test compounds share the SAME latent region -> the model can only use signal that lives on BOTH manifolds -> better
OOD generalization to the blinded test. TRANSDUCTIVE (uses unlabeled 513). Decisive test: does alignment (lambda>0)
beat the IDENTICAL model with alignment OFF (lambda=0) on the held-out 253 (treated as blinded)?

Metrics on 253: standalone RAE, corr with nb3200 error, and as a FEATURE on nb3200 (does aligned add over non-aligned?).
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train, load_test
from src.pxr.eval import rae, scaffold_kfold_indices
from src.pxr.featurize import combined, impute
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import torch, torch.nn as nn

D = "data/processed"


def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else None


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lamb):
        ctx.lamb = lamb; return x.view_as(x)
    @staticmethod
    def backward(ctx, g):
        return -ctx.lamb * g, None


class DANN(nn.Module):
    def __init__(self, d, h=256, z=64, p=0.3):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(h, z), nn.BatchNorm1d(z), nn.ReLU())
        self.reg = nn.Sequential(nn.Linear(z, z), nn.ReLU(), nn.Dropout(p), nn.Linear(z, 1))
        self.dis = nn.Sequential(nn.Linear(z, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x, lamb=0.0):
        zf = self.enc(x)
        return self.reg(zf).squeeze(-1), self.dis(GRL.apply(zf, lamb)).squeeze(-1), zf


def train_dann(Xtr, ytr, Xtgt, lamb, epochs=150, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "cpu"
    m = DANN(Xtr.shape[1]).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.float32)
    Xg = torch.tensor(Xtgt, dtype=torch.float32)
    nS, nT = len(Xt), len(Xg); bs = 256
    for ep in range(epochs):
        lam = lamb * (2 / (1 + np.exp(-10 * ep / epochs)) - 1)   # GRL warmup schedule
        m.train(); perm = torch.randperm(nS)
        for b in range(0, nS, bs):
            si = perm[b:b + bs]; ti = torch.randint(0, nT, (len(si),))
            opt.zero_grad()
            yp, ds, _ = m(Xt[si], lam); _, dt, _ = m(Xg[ti], lam)
            lr = nn.functional.smooth_l1_loss(yp, yt[si])
            ld = nn.functional.binary_cross_entropy_with_logits(
                torch.cat([ds, dt]), torch.cat([torch.zeros(len(si)), torch.ones(len(ti))]))
            (lr + ld).backward(); opt.step()
    m.eval()
    return m


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test().reset_index(drop=True)
    print("featurizing...", flush=True)
    Xtr = impute(combined(tr["smiles"].tolist())).astype(np.float32)
    Xte = impute(combined(te["smiles"].tolist())).astype(np.float32)
    mu, sd = np.vstack([Xtr, Xte]).mean(0), np.vstack([Xtr, Xte]).std(0) + 1e-6   # target-aware standardize
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    ytr = tr["pec50"].to_numpy().astype(np.float32)

    unb = np.load(f"{D}/_audit_unblind_idx.npy"); y = np.load(f"{D}/_audit_unblind_y.npy")
    anchor = np.load(f"{D}/nb3200_pred_oof.npy"); resid = y - anchor

    res = {}
    for lamb in [0.0, 0.1, 0.3, 1.0]:
        preds = []
        for sd_ in range(4):
            m = train_dann(Xtr, ytr, Xte, lamb, seed=sd_)   # target domain = ALL 513 test
            with torch.no_grad():
                p, _, _ = m(torch.tensor(Xte[unb], dtype=torch.float32))
            preds.append(p.numpy())
        geo = np.mean(preds, 0)
        r = rae(y, geo); cerr = np.corrcoef(geo, resid)[0, 1]
        # blend with nb3200
        bw, br = 0.0, rae(y, anchor)
        for w in np.linspace(0, 0.5, 26):
            rr = rae(y, (1 - w) * anchor + w * geo)
            if rr < br:
                br, bw = rr, w
        res[lamb] = (r, float(cerr), bw, br)
        tag = "ALIGN-OFF" if lamb == 0 else f"lambda={lamb}"
        print(f"  {tag:12s}: standalone RAE {r:.4f} | corr(pred,nb3200err) {cerr:+.3f} | blend w={bw:.2f} RAE {br:.4f} (d {br-rae(y,anchor):+.4f})", flush=True)
    print(f"\nnb3200 anchor = {rae(y, anchor):.4f}")
    print(f"DANN works if a lambda>0 generalizes (lower standalone RAE / higher |corr w/ err| / better blend) than ALIGN-OFF (lambda=0)")
    json.dump({str(k): v for k, v in res.items()}, open(f"{D}/nb1070_dann.json", "w"), indent=2)


if __name__ == "__main__":
    main()
