"""nb1018 — INTERACTION-GRAPH ADDITIVE HEAD (the ambitious model; user's micro-interaction decomposition).
Each ligand = a SET of protein-ligand interaction edges (top-128 z-pairs, 128-d each + strength). The head:
    contribution_e = MLP(edge_e)          # scalar per micro-interaction
    pEC50_residual = sum_e contribution_e  # ADDITIVE readout (interpretable)
OVERFIT-SAFE: train on the 1906 INDEPENDENT train cofolds, evaluate on the 253 (never trained on).
Anchor = chemprop_aux; head predicts its residual. Beat the rich-z LGBM baseline (-0.008) to justify itself.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train, load_test
from src.pxr.eval import rae
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import torch, torch.nn as nn

D = "data/processed"; U = "C:/pxr_struct/boltz"
torch.manual_seed(0); np.random.seed(0)


def ik(s):
    m = Chem.MolFromSmiles(str(s)); return Chem.MolToInchiKey(m) if m else None


class AdditiveHead(nn.Module):
    def __init__(self, d=129, h=64, p=0.3):
        super().__init__()    # d=129 = 128 edge feats + 1 normalized strength feature
        self.f = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(p),
                               nn.Linear(h, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, 1))
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, edges, mask, strength):
        x = torch.cat([edges, strength.unsqueeze(-1)], dim=-1)   # (B, M, 129)
        c = self.f(x).squeeze(-1) * mask                          # (B, M) per-edge contribution, padding zeroed
        return self.scale * c.sum(1) + self.bias                  # ADDITIVE readout


def load_split(prefix, idx_map, smiles, anchor, y):
    edges = np.load(f"{U}/{prefix}_edges.npy")[idx_map]          # (n, 128, 128)
    enorm = np.load(f"{U}/{prefix}_enorm.npy")[idx_map]          # (n, 128)
    nedge = np.load(f"{U}/{prefix}_nedge.npy")[idx_map]
    mask = (np.arange(128)[None, :] < nedge[:, None]).astype(np.float32)
    # normalize edge features + strength
    return (torch.tensor(edges, dtype=torch.float32), torch.tensor(mask),
            torch.tensor(enorm, dtype=torch.float32), torch.tensor(anchor, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32))


def main():
    # ---- TRAIN: 1906 independent cofolds (cofold/unimol order), align anchor by InChIKey ----
    uni = pd.read_csv(f"{D}/unimol_train.csv")
    nedge_tr = np.load(f"{U}/train_nedge.npy")
    lt = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    anc_tr_src = np.load(f"{D}/oof_chemprop_aux.npy")
    lt_ik = {}
    for i, s in enumerate(lt["smiles"]):
        k = ik(s)
        if k and k not in lt_ik:
            lt_ik[k] = i
    tr_rows, tr_y, tr_anc = [], [], []
    for ci in range(len(uni)):
        if nedge_tr[ci] == 0:
            continue
        k = ik(uni["smiles"].iloc[ci])
        if k is None or k not in lt_ik:
            continue
        tr_rows.append(ci); tr_y.append(float(uni["pec50"].iloc[ci])); tr_anc.append(float(anc_tr_src[lt_ik[k]]))
    tr_rows = np.array(tr_rows)
    Xe, Mk, St, Anc, Y = load_split("train", tr_rows, None, np.array(tr_anc), np.array(tr_y))
    # global feature normalization (from train)
    mu, sd = Xe.mean((0, 1), keepdim=True), Xe.std((0, 1), keepdim=True) + 1e-6
    Xe = (Xe - mu) / sd
    smu, ssd = St[Mk > 0].mean(), St[Mk > 0].std() + 1e-6      # strength stats over valid edges (train)
    St = (St - smu) / ssd
    n = len(tr_rows); perm = np.random.permutation(n); vi = perm[:n // 6]; ti = perm[n // 6:]
    print(f"train edges: {n} compounds (val {len(vi)})  anchor RAE(train)={rae(Y.numpy(), Anc.numpy()):.4f}")

    # ---- 253 eval ----
    te = load_test(); unb = np.load(f"{D}/_audit_unblind_idx.npy"); yv = np.load(f"{D}/_audit_unblind_y.npy")
    anc_te = np.load(f"{D}/te_chemprop_aux.npy")[unb]
    Xe_te = np.load(f"{U}/test_edges.npy")[unb]; en_te = np.load(f"{U}/test_enorm.npy")[unb]; ne_te = np.load(f"{U}/test_nedge.npy")[unb]
    mask_te = (np.arange(128)[None, :] < ne_te[:, None]).astype(np.float32)
    Xte = (torch.tensor(Xe_te, dtype=torch.float32) - mu) / sd
    Mte = torch.tensor(mask_te); Ste = (torch.tensor(en_te, dtype=torch.float32) - smu) / ssd

    resid = Y - Anc
    r_base = rae(yv, anc_te)
    va = rae(Y[vi].numpy(), Anc[vi].numpy())   # val anchor baseline

    def train_eval(M, h, p, wd, lr, pool, seeds=10):
        deltas, vrs = [], []
        for sd_ in range(seeds):
            torch.manual_seed(sd_)
            m = AdditiveHead(d=129, h=h, p=p); m.pool = pool
            # monkeypatch pooling
            def fwd(edges, mask, strength, _m=m):
                x = torch.cat([edges, strength.unsqueeze(-1)], -1)
                c = _m.f(x).squeeze(-1) * mask
                agg = c.sum(1) / (mask.sum(1) if pool == "mean" else 1.0)
                return _m.scale * agg + _m.bias
            opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
            best, bstate, pat = 1e9, None, 0
            for ep in range(300):
                m.train()
                for b in range(0, len(ti), 128):
                    bi = ti[b:b + 128]
                    opt.zero_grad()
                    loss = nn.functional.smooth_l1_loss(fwd(Xe[bi, :M], Mk[bi, :M], St[bi, :M]), resid[bi])
                    loss.backward(); opt.step()
                m.eval()
                with torch.no_grad():
                    vr = rae(Y[vi].numpy(), (Anc[vi] + fwd(Xe[vi, :M], Mk[vi, :M], St[vi, :M])).numpy())
                if vr < best - 1e-4:
                    best, bstate, pat = vr, {k: v.clone() for k, v in m.state_dict().items()}, 0
                else:
                    pat += 1
                    if pat > 25:
                        break
            m.load_state_dict(bstate); m.eval()
            with torch.no_grad():
                te_pred = torch.tensor(anc_te) + fwd(Xte[:, :M], Mte[:, :M], Ste[:, :M])
            deltas.append(rae(yv, te_pred.numpy()) - r_base); vrs.append(best - va)
        return np.mean(vrs), np.mean(deltas)

    print(f"253 base(chemprop_aux)={r_base:.4f}  val_anchor={va:.4f}  (target: 253 delta < -0.008 to beat rich-z)")
    # FIXED a-priori config (chosen for robustness, NOT cherry-picked by 253 score): 10 seeds -> honest, selection-free
    cfg = (64, 64, 0.5, 1e-3, 1e-3, "mean")
    vd, td = train_eval(*cfg)
    print(f"  FIXED {cfg}: val_delta={vd:+.4f}  253_delta={td:+.5f}  (10-seed avg, no config selection)")
    print(f"  -> honest selection-free estimate vs rich-z baseline -0.0080")
    best_cfg, best_d = cfg, td
    json.dump({"best_cfg": list(best_cfg), "best_253_delta": float(best_d), "base_253": float(r_base)},
              open(f"{U}/nb1018_head.json", "w"), indent=2)


if __name__ == "__main__":
    main()
