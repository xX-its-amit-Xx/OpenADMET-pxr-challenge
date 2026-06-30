"""nb1020 — does the interaction-head's structural signal help the DEPLOY model (nb3200), not just chemprop_aux?
Train the head (fixed M=64 mean, 3581 train, 5-seed ensemble), take its 253 structural output, and test it as a
FEATURE on the nb3200 substrate (combined+chempropembed base, nb3200 anchor, cross-fit 30 seeds) -- apples-to-apples
with rich-z's -0.008. The head output is selection-free (fixed config). Stable-negative -> deployable on the best model.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train, load_test
from src.pxr.eval import rae, scaffold_kfold_indices
from src.pxr.featurize import combined, impute
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import torch, torch.nn as nn, lightgbm as lgb
from nb1018_interaction_graph_head import AdditiveHead

D = "data/processed"; U = "C:/pxr_struct/boltz"; QL, QH = 0.05, 0.98


def ik(s):
    m = Chem.MolFromSmiles(str(s)); return Chem.MolToInchiKey(m) if m else None


def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else None


def train_head_get_253():
    """train head on 3581 train (chemprop_aux residual), return its raw structural output on the 253 (5-seed avg)."""
    uni = pd.read_csv(f"{D}/unimol_train.csv"); nedge = np.load(f"{U}/train_nedge.npy")
    lt = load_train().dropna(subset=["pec50"]).reset_index(drop=True); anc_src = np.load(f"{D}/oof_chemprop_aux.npy")
    lt_ik = {}
    for i, s in enumerate(lt["smiles"]):
        k = ik(s)
        if k and k not in lt_ik:
            lt_ik[k] = i
    rows, y, anc = [], [], []
    for ci in range(len(uni)):
        if nedge[ci] == 0:
            continue
        k = ik(uni["smiles"].iloc[ci])
        if k and k in lt_ik:
            rows.append(ci); y.append(float(uni["pec50"].iloc[ci])); anc.append(float(anc_src[lt_ik[k]]))
    rows = np.array(rows)
    E = np.load(f"{U}/train_edges.npy")[rows]; EN = np.load(f"{U}/train_enorm.npy")[rows]; NE = np.load(f"{U}/train_nedge.npy")[rows]
    mu = E.mean((0, 1), keepdims=True); sd = E.std((0, 1), keepdims=True) + 1e-6
    Xe = torch.tensor((E - mu) / sd, dtype=torch.float32)
    Mk = torch.tensor((np.arange(128)[None] < NE[:, None]).astype(np.float32))
    smu = EN[NE[:, None] > np.arange(128)[None]].mean() if (NE > 0).any() else 0.0
    ssd = EN[NE[:, None] > np.arange(128)[None]].std() + 1e-6
    St = torch.tensor((EN - smu) / ssd, dtype=torch.float32)
    resid = torch.tensor(np.array(y) - np.array(anc), dtype=torch.float32)
    # test 253 edges
    te = load_test(); unb = np.load(f"{D}/_audit_unblind_idx.npy")
    Ete = np.load(f"{U}/test_edges.npy")[unb]; ENte = np.load(f"{U}/test_enorm.npy")[unb]; NEte = np.load(f"{U}/test_nedge.npy")[unb]
    Xte = torch.tensor((Ete - mu) / sd, dtype=torch.float32)
    Mte = torch.tensor((np.arange(128)[None] < NEte[:, None]).astype(np.float32))
    Ste = torch.tensor((ENte - smu) / ssd, dtype=torch.float32)
    n = len(rows); outs = []
    for seed in range(5):
        torch.manual_seed(seed); np.random.seed(seed)
        perm = np.random.permutation(n); vi, ti = perm[:n // 6], perm[n // 6:]
        m = AdditiveHead(d=129, h=64, p=0.5)
        def fwd(e, mk, st):
            x = torch.cat([e, st.unsqueeze(-1)], -1); c = m.f(x).squeeze(-1) * mk
            return m.scale * (c.sum(1) / mk.sum(1)) + m.bias
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-3)
        best, bs, pat = 1e9, None, 0
        for ep in range(300):
            m.train()
            for b in range(0, len(ti), 128):
                bi = ti[b:b + 128]; opt.zero_grad()
                nn.functional.smooth_l1_loss(fwd(Xe[bi], Mk[bi], St[bi]), resid[bi]).backward(); opt.step()
            m.eval()
            with torch.no_grad():
                vr = float(((fwd(Xe[vi], Mk[vi], St[vi]) - resid[vi]) ** 2).mean())
            if vr < best - 1e-5:
                best, bs, pat = vr, {k: v.clone() for k, v in m.state_dict().items()}, 0
            else:
                pat += 1
                if pat > 25:
                    break
        m.load_state_dict(bs); m.eval()
        with torch.no_grad():
            outs.append(fwd(Xte, Mte, Ste).numpy())
    return np.mean(outs, 0)   # 253 structural output (5-seed avg)


def clipped(X, resid, anchor, y, folds):
    pred = anchor.copy()
    for tri, vai in folds:
        m = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1).fit(X[tri], resid[tri])
        p = anchor[vai] + m.predict(X[vai]); lo, hi = np.quantile(y[tri], QL), np.quantile(y[tri], QH)
        pred[vai] = np.clip(p, lo, hi)
    return float(rae(y, pred))


def main():
    head253 = train_head_get_253()
    te = load_test(); unb = np.load(f"{D}/_audit_unblind_idx.npy"); y = np.load(f"{D}/_audit_unblind_y.npy")
    smiles = te["smiles"].to_numpy()[unb].tolist(); scaf = [murcko(s) for s in smiles]
    base = np.hstack([impute(combined(smiles)).astype(np.float32), np.load(f"{D}/te_chemprop_embed_300.npy")[unb].astype(np.float32)])
    anchor = np.load(f"{D}/nb3200_pred_oof.npy"); resid = y - anchor
    hz = ((head253 - head253.mean()) / (head253.std() + 1e-9)).reshape(-1, 1).astype(np.float32)
    SEEDS = list(range(1400, 1430)); ds = []
    for s in SEEDS:
        f = scaffold_kfold_indices(scaf, 5, seed=s)
        ds.append(clipped(np.hstack([base, hz]), resid, anchor, y, f) - clipped(base, resid, anchor, y, f))
    ds = np.array(ds); st = ds.mean() < 0 and abs(ds.mean()) > ds.std()
    print(f"HEAD-output as feature on nb3200 substrate (30 seeds): mean={ds.mean():+.5f} std={ds.std():.5f} wins={int((ds<0).sum())}/30 stable={st}")
    print(f"  vs rich-z on nb3200 = -0.008 honest")
    json.dump({"mean": float(ds.mean()), "std": float(ds.std()), "wins": int((ds < 0).sum()), "stable": bool(st)},
              open(f"{U}/nb1020_head_on_nb3200.json", "w"), indent=2)


if __name__ == "__main__":
    main()
