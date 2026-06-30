"""nb1019 — PRETRAIN the interaction head on the single-conc drove (weak log2FC labels), then FINE-TUNE on the
4139 PXR pEC50. The ambitious "embed a prior from droves, then fine-tune" route. Tests whether a pretrained
structural prior lets the head EXCEED what from-scratch (nb1018, -0.021 on chemprop_aux) achieves -- and crucially
whether it can then beat nb3200 (which from-scratch could not). Ready to run when sc_pool_* (merged) lands.

Pipeline: pretrain head->log2FC on ~6000 SC edges; warm-start; fine-tune head->chemprop_aux residual on 3581;
eval on 253 (both chemprop_aux substrate AND, via the output-as-feature, the nb3200 deploy substrate).
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train, load_test
from src.pxr.eval import rae
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import torch, torch.nn as nn
from nb1018_interaction_graph_head import AdditiveHead

D = "data/processed"; U = "C:/pxr_struct/boltz"


def ik(s):
    m = Chem.MolFromSmiles(str(s)); return Chem.MolToInchiKey(m) if m else None


def fwd(m, e, mk, st):
    x = torch.cat([e, st.unsqueeze(-1)], -1); c = m.f(x).squeeze(-1) * mk
    return m.scale * (c.sum(1) / mk.sum(1).clamp(min=1)) + m.bias


def norm_edges(E, EN, NE, mu, sd, smu, ssd):
    Xe = torch.tensor((E - mu) / sd, dtype=torch.float32)
    Mk = torch.tensor((np.arange(128)[None] < NE[:, None]).astype(np.float32))
    St = torch.tensor((EN - smu) / ssd, dtype=torch.float32)
    return Xe, Mk, St


def train(m, Xe, Mk, St, target, epochs=200, lr=1e-3, wd=1e-3, patience=20):
    n = len(target); perm = np.random.permutation(n); vi, ti = perm[:n // 6], perm[n // 6:]
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    best, bs, pat = 1e9, None, 0
    for ep in range(epochs):
        m.train()
        for b in range(0, len(ti), 128):
            bi = ti[b:b + 128]; opt.zero_grad()
            nn.functional.smooth_l1_loss(fwd(m, Xe[bi], Mk[bi], St[bi]), target[bi]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vr = float(((fwd(m, Xe[vi], Mk[vi], St[vi]) - target[vi]) ** 2).mean())
        if vr < best - 1e-5:
            best, bs, pat = vr, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            pat += 1
            if pat > patience:
                break
    m.load_state_dict(bs); return m


def main():
    # ---- normalization from the FINETUNE train edges (consistent across pretrain/finetune/eval) ----
    uni = pd.read_csv(f"{D}/unimol_train.csv"); NEtr_all = np.load(f"{U}/train_nedge.npy")
    lt = load_train().dropna(subset=["pec50"]).reset_index(drop=True); anc_src = np.load(f"{D}/oof_chemprop_aux.npy")
    lt_ik = {}
    for i, s in enumerate(lt["smiles"]):
        k = ik(s)
        if k and k not in lt_ik:
            lt_ik[k] = i
    rows, y, anc = [], [], []
    for ci in range(len(uni)):
        if NEtr_all[ci] == 0:
            continue
        k = ik(uni["smiles"].iloc[ci])
        if k and k in lt_ik:
            rows.append(ci); y.append(float(uni["pec50"].iloc[ci])); anc.append(float(anc_src[lt_ik[k]]))
    rows = np.array(rows)
    Etr = np.load(f"{U}/train_edges.npy")[rows]; ENtr = np.load(f"{U}/train_enorm.npy")[rows]; NEtr = NEtr_all[rows]
    mu = Etr.mean((0, 1), keepdims=True); sd = Etr.std((0, 1), keepdims=True) + 1e-6
    valid = NEtr[:, None] > np.arange(128)[None]; smu = ENtr[valid].mean(); ssd = ENtr[valid].std() + 1e-6
    resid = torch.tensor(np.array(y) - np.array(anc), dtype=torch.float32)
    Xtr, Mtr, Str = norm_edges(Etr, ENtr, NEtr, mu, sd, smu, ssd)

    # ---- single-conc pretrain set (log2FC) ----
    sc = pd.read_csv(f"{D}/sc_pretrain_subset.csv"); Esc = np.load(f"{U}/sc_pool_edges.npy")
    ENsc = np.load(f"{U}/sc_pool_enorm.npy"); NEsc = np.load(f"{U}/sc_pool_nedge.npy")
    scmask = NEsc > 0
    yfc = sc["log2_fc_estimate"].to_numpy()[:len(NEsc)]
    keep = scmask & np.isfinite(yfc)
    Xsc, Msc, Ssc = norm_edges(Esc[keep], ENsc[keep], NEsc[keep], mu, sd, smu, ssd)
    yfc_t = torch.tensor((yfc[keep] - np.nanmean(yfc[keep])) / (np.nanstd(yfc[keep]) + 1e-6), dtype=torch.float32)
    print(f"pretrain SC: {int(keep.sum())} compounds | finetune train: {len(rows)} | 253 eval")

    # ---- 253 eval edges ----
    te = load_test(); unb = np.load(f"{D}/_audit_unblind_idx.npy"); yv = np.load(f"{D}/_audit_unblind_y.npy")
    anc_te = np.load(f"{D}/te_chemprop_aux.npy")[unb]
    Xte, Mte, Ste = norm_edges(np.load(f"{U}/test_edges.npy")[unb], np.load(f"{U}/test_enorm.npy")[unb],
                               np.load(f"{U}/test_nedge.npy")[unb], mu, sd, smu, ssd)
    r_base = rae(yv, anc_te)

    res = {}
    for mode in ["scratch", "pretrain_finetune"]:
        deltas = []
        for sd_ in range(5):
            torch.manual_seed(sd_); np.random.seed(sd_)
            m = AdditiveHead(d=129, h=64, p=0.5)
            if mode == "pretrain_finetune":
                m = train(m, Xsc, Msc, Ssc, yfc_t, epochs=120, lr=1e-3, wd=1e-3)   # prior
            m = train(m, Xtr, Mtr, Str, resid, epochs=200, lr=1e-3, wd=1e-3)        # PXR fine-tune
            m.eval()
            with torch.no_grad():
                deltas.append(rae(yv, anc_te + fwd(m, Xte, Mte, Ste).numpy()) - r_base)
        res[mode] = (float(np.mean(deltas)), float(np.std(deltas)))
        print(f"  {mode:18s}: 253 delta {res[mode][0]:+.5f} +/- {res[mode][1]:.5f}")
    print(f"\nfrom-scratch ~-0.021; pretrain {'HELPS' if res['pretrain_finetune'][0] < res['scratch'][0]-0.002 else 'no clear gain'} "
          f"(delta-of-deltas {res['pretrain_finetune'][0]-res['scratch'][0]:+.5f})")
    json.dump(res, open(f"{U}/nb1019_pretrain_finetune.json", "w"), indent=2)


if __name__ == "__main__":
    main()
