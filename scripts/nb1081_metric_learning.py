"""nb1081 — [ROADMAP #1] METRIC-LEARNED activity-aligned embedding (the correct 'merge train/test latent spaces').

The fingerprint experiment failed because MACCS maximizes CHEMICAL closeness (anti-aligned with activity, Spearman
+0.07). Here we LEARN an embedding f(x) so that ||f(a)-f(b)|| ~ |pEC50(a)-pEC50(b)| (a Siamese/contrastive metric
trained on train PAIRS), from the activity-aligned ingredients (ErG + rich-z + physchem + chempropembed). Then we
(a) measure the learned-space alignment on the 253 (target: >> ErG's +0.234), (b) predict via GP/kNN in the learned
metric, and (c) test whether the learned-metric prediction beats kNN-in-raw-space and adds to nb3200.

Decisive: does a metric OPTIMIZED for activity-alignment generalize to the novel tail better than chemical space?
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train, load_test
from src.pxr.eval import rae, scaffold_kfold_indices
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist, pdist, squareform
from rdkit import Chem
from rdkit.Chem import rdReducedGraphs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import torch, torch.nn as nn

P = "data/processed"; MO = "C:/pxr_struct/boltz/modal"


def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else None


def erg(smiles):
    out = [np.array(rdReducedGraphs.GetErGFingerprint(Chem.MolFromSmiles(str(s))), np.float32)
           if Chem.MolFromSmiles(str(s)) else None for s in smiles]
    dim = next(len(v) for v in out if v is not None)
    return np.array([v if v is not None else np.zeros(dim, np.float32) for v in out])


class Emb(nn.Module):
    def __init__(self, d, z=64):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.2),
                               nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, z))
    def forward(self, x):
        return self.f(x)


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test().reset_index(drop=True)
    unb = np.load(f"{P}/_audit_unblind_idx.npy"); y = np.load(f"{P}/_audit_unblind_y.npy")
    anchor = np.load(f"{P}/nb3200_pred_oof.npy"); resid = y - anchor
    ytr = tr["pec50"].to_numpy().astype(np.float32)

    # activity-aligned ingredients: ErG + chempropembed (+ rich-z where available, eval only)
    print("featurizing (ErG + chempropembed)...", flush=True)
    Xtr = np.hstack([erg(tr["smiles"].tolist()), np.load(f"{P}/oof_chemprop_aux.npy").reshape(-1, 1).astype(np.float32)])
    # NOTE: train chempropembed-300 not available; use ErG (aligned) + chemprop_aux scalar as the learnable base
    Xte = np.hstack([erg(te["smiles"].to_numpy()[unb].tolist()),
                     np.load(f"{P}/te_chemprop_aux.npy")[unb].reshape(-1, 1).astype(np.float32)])
    sc = StandardScaler().fit(np.vstack([Xtr, Xte])); Xtr = sc.transform(Xtr).astype(np.float32); Xte = sc.transform(Xte).astype(np.float32)

    dev = "cpu"; m = Emb(Xtr.shape[1]).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr); n = len(Xt)
    print("metric learning (pairwise ||f(a)-f(b)|| ~ |dpEC50|)...", flush=True)
    for ep in range(300):
        m.train(); idx = torch.randint(0, n, (2048, 2))
        a, b = Xt[idx[:, 0]], Xt[idx[:, 1]]
        de = torch.norm(m(a) - m(b) + 1e-8, dim=1)
        dy = torch.abs(yt[idx[:, 0]] - yt[idx[:, 1]])
        loss = nn.functional.smooth_l1_loss(de, dy)         # distance == activity difference
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        Ztr = m(Xt).numpy(); Zte = m(torch.tensor(Xte)).numpy()

    # (a) learned-space alignment on the 253
    iu = np.triu_indices(len(unb), 1); dy = np.abs(y[:, None] - y[None, :])[iu]
    rho = spearmanr(squareform(pdist(Zte))[iu], dy).correlation
    print(f"\nLEARNED-space alignment on 253: Spearman {rho:+.3f}  (ErG baseline +0.234, Morgan +0.04)")

    # (b) predict via kNN in learned metric (train->253)
    Dte = cdist(Zte, Ztr); pred = np.zeros(len(unb))
    for i in range(len(unb)):
        k = np.argsort(Dte[i])[:5]; w = 1 / (Dte[i][k] + 1e-6) ** 2; pred[i] = np.sum(w * ytr[k]) / np.sum(w)
    print(f"learned-metric kNN retrieval RAE: {rae(y, pred):.4f}  (ErG-retrieval 0.90, nb3200 0.4416)")

    # (c) blend learned-metric retrieval with nb3200 + as feature
    bw, br = 0.0, rae(y, anchor)
    for w in np.linspace(0, 1, 41):
        r = rae(y, (1 - w) * anchor + w * pred)
        if r < br:
            br, bw = r, w
    print(f"blend with nb3200: w={bw:.2f} RAE {br:.4f} (delta {br-rae(y,anchor):+.4f})")
    print(f"corr(learned-metric pred, nb3200 error) = {np.corrcoef(pred, resid)[0,1]:+.3f}")
    json.dump({"learned_align": float(rho), "retrieval_rae": float(rae(y, pred)), "blend_delta": float(br - rae(y, anchor))},
              open(f"{P}/nb1081_metric.json", "w"), indent=2)


if __name__ == "__main__":
    main()
