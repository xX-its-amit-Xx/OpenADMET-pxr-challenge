"""nb1022 — COMPLEMENTARITY: does the standalone interaction-head (nb1021 abs_scratch, the only positive lever)
ADD on top of rich-z (the deployed -0.008 structural feature)? Both are views of the SAME Boltz z block:
rich-z = hand-pooled mean/std/max (PCA<=15); head = learned additive micro-interaction readout. If they capture
different structure, base+rich-z+head beats base+rich-z and we deploy BOTH. If the head is subsumed by rich-z,
rich-z alone stays the structural deliverable.

FRESH seeds {1415..1444} (nb1021 selected abs_scratch on 1400..1414 -> avoid reusing them). Incremental deltas:
  d_richz   = RAE(base+richz)        - RAE(base)
  d_head    = RAE(base+head)         - RAE(base)
  d_both    = RAE(base+richz+head)   - RAE(base)
  d_add     = RAE(base+richz+head)   - RAE(base+richz)   <-- the deploy question (head's marginal value over rich-z)
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test
from src.pxr.eval import rae, scaffold_kfold_indices
from src.pxr.featurize import combined, impute
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
from nb1021_mechanistic_pretrain import (build_train, norm_edges, fit_task, MultiAdditiveHead,
                                         murcko, clipped, D, U, QL, QH)


def abs_scratch_head_253():
    """Recompute the nb1021 abs_scratch head output on the 253 (5-seed avg)."""
    rows, E, EN, NE, y, anc, null = build_train()
    mu = E.mean((0, 1), keepdims=True); sd = E.std((0, 1), keepdims=True) + 1e-6
    valid = NE[:, None] > np.arange(128)[None]; smu = EN[valid].mean(); ssd = EN[valid].std() + 1e-6
    Xtr, Mtr, Str = norm_edges(E, EN, NE, mu, sd, smu, ssd)
    ym, ystd = y.mean(), y.std() + 1e-6
    y_abs = torch.tensor((y - ym) / ystd, dtype=torch.float32)
    unb = np.load(f"{D}/_audit_unblind_idx.npy")
    Xte, Mte, Ste = norm_edges(np.load(f"{U}/test_edges.npy")[unb], np.load(f"{U}/test_enorm.npy")[unb],
                               np.load(f"{U}/test_nedge.npy")[unb], mu, sd, smu, ssd)
    outs = []
    for s in range(5):
        torch.manual_seed(s); np.random.seed(s)
        m = MultiAdditiveHead(d=129, h=64, p=0.5)
        m = fit_task(m, Xtr, Mtr, Str, y_abs, "pec50", epochs=250)
        m.eval()
        with torch.no_grad():
            outs.append(m(Xte, Mte, Ste, "pec50").numpy())
    return np.mean(outs, 0)


def main():
    head253 = abs_scratch_head_253()
    unb = np.load(f"{D}/_audit_unblind_idx.npy"); yv = np.load(f"{D}/_audit_unblind_y.npy")
    te = load_test(); smiles = te["smiles"].to_numpy()[unb].tolist(); scaf = [murcko(s) for s in smiles]
    base = np.hstack([impute(combined(smiles)).astype(np.float32),
                      np.load(f"{D}/te_chemprop_embed_300.npy")[unb].astype(np.float32)])
    anchor = np.load(f"{D}/nb3200_pred_oof.npy"); resid = yv - anchor

    # rich-z PCA<=15 (fit on all 513, no label leak), take unb rows
    rz = np.load(f"{U}/boltz_z_rich_513.npy")
    rz_pca = PCA(n_components=15, random_state=0).fit_transform(StandardScaler().fit_transform(rz))[unb].astype(np.float32)
    hz = ((head253 - head253.mean()) / (head253.std() + 1e-9)).reshape(-1, 1).astype(np.float32)

    Xb = base
    Xrz = np.hstack([base, rz_pca])
    Xhd = np.hstack([base, hz])
    Xboth = np.hstack([base, rz_pca, hz])

    SEEDS = list(range(1415, 1445))
    d_richz, d_head, d_both, d_add = [], [], [], []
    for s in SEEDS:
        f = scaffold_kfold_indices(scaf, 5, seed=s)
        rb = clipped(Xb, resid, anchor, yv, f)
        rrz = clipped(Xrz, resid, anchor, yv, f)
        rhd = clipped(Xhd, resid, anchor, yv, f)
        rbo = clipped(Xboth, resid, anchor, yv, f)
        d_richz.append(rrz - rb); d_head.append(rhd - rb); d_both.append(rbo - rb); d_add.append(rbo - rrz)

    def summ(name, arr):
        a = np.array(arr); st = a.mean() < 0 and abs(a.mean()) > a.std()
        print(f"  {name:32s}: {a.mean():+.5f} +/- {a.std():.5f}  wins {int((a<0).sum())}/{len(a)}  stable={st}", flush=True)
        return {"mean": float(a.mean()), "std": float(a.std()), "wins": int((a < 0).sum()), "stable": bool(st)}

    print(f"\nincremental deltas vs base (anchor=nb3200), 30 fresh seeds {SEEDS[0]}..{SEEDS[-1]}:")
    res = {"d_richz_vs_base": summ("rich-z over base", d_richz),
           "d_head_vs_base": summ("head over base", d_head),
           "d_both_vs_base": summ("rich-z+head over base", d_both),
           "d_head_marginal_over_richz": summ("head MARGINAL over rich-z <<<", d_add)}
    print("\n  DEPLOY if 'head MARGINAL over rich-z' is stable-negative -> base+rich-z+head beats rich-z alone")
    json.dump(res, open(f"{U}/nb1022_richz_plus_head.json", "w"), indent=2)


if __name__ == "__main__":
    main()
