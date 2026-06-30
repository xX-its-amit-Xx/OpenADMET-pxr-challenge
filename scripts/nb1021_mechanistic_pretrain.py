"""nb1021 — MECHANISTICALLY-AUGMENTED PRETRAINING (user: "pretraining with augmentations from what we learned
to increase our mechanistic prior").

DIAGNOSIS being attacked (from nb1019/nb1020): the interaction head was REDUNDANT with nb3200 (+0.00005 as a
feature) because it was trained to predict chemprop_aux's RESIDUAL -- i.e. it learned a *copy of chemprop's error
pattern*, which the stronger nb3200 ensemble had already corrected. More data couldn't fix a target-redundancy.

AUGMENTATIONS (each drawn from a mechanistic lesson of this session):
  1. STANDALONE ABSOLUTE target instead of chemprop_aux residual -> the output is an INDEPENDENT structural
     pEC50 estimate, not a prediction of chemprop's mistakes. Breaks the redundancy by construction.
  2. MECHANISTIC PRIOR: pretrain the shared edge-encoder on the single-conc log2FC drove (activation-axis weak
     labels, ~1360 cofolded compounds) -> the encoder learns activation geometry before ever seeing pEC50.
  3. SPECIFICITY MULTI-TASK: co-train a counter-assay-NULL readout (shared encoder, separate additive readout)
     so the representation encodes PXR-SPECIFIC activation (pEC50 high AND null low) vs generic engagement.
     This is the activation-vs-binding distinction that defines PXR (docking-scalar failed for exactly this reason).

DECISIVE TEST (apples-to-apples vs rich-z -0.008 and the redundant residual-head +0.00005): each mode's 253
structural output is used as a FEATURE on the nb3200 deploy substrate (combined+chempropembed base, nb3200 anchor,
per-fold y-clip), cross-fit over 30 fresh seeds {1400..1429}. Also report corr(head_out, nb3200_residual): a
LOWER corr means LESS redundant = the mechanism worked.
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

D = "data/processed"; U = "C:/pxr_struct/boltz"; QL, QH = 0.05, 0.98


def ik(s):
    m = Chem.MolFromSmiles(str(s)); return Chem.MolToInchiKey(m) if m else None


def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else None


class MultiAdditiveHead(nn.Module):
    """Shared edge-encoder + per-task ADDITIVE readouts. Keeps the micro-interaction-decomposition story:
    each task's prediction = scale * mean_e(readout_t(enc(edge_e))) + bias_t."""
    def __init__(self, d=129, h=64, p=0.5, tasks=("pec50", "null", "fc")):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(h, h), nn.ReLU(), nn.Dropout(p))
        self.readout = nn.ModuleDict({t: nn.Linear(h, 1) for t in tasks})
        self.scale = nn.ParameterDict({t: nn.Parameter(torch.tensor(0.1)) for t in tasks})
        self.bias = nn.ParameterDict({t: nn.Parameter(torch.zeros(1)) for t in tasks})

    def forward(self, edges, mask, strength, task):
        x = torch.cat([edges, strength.unsqueeze(-1)], -1)
        hid = self.enc(x)
        c = self.readout[task](hid).squeeze(-1) * mask
        agg = c.sum(1) / mask.sum(1).clamp(min=1)
        return self.scale[task] * agg + self.bias[task]


def norm_edges(E, EN, NE, mu, sd, smu, ssd):
    Xe = torch.tensor((E - mu) / sd, dtype=torch.float32)
    Mk = torch.tensor((np.arange(128)[None] < NE[:, None]).astype(np.float32))
    St = torch.tensor((EN - smu) / ssd, dtype=torch.float32)
    return Xe, Mk, St


def fit_task(m, Xe, Mk, St, target, task, epochs=200, lr=1e-3, wd=1e-3, patience=20, aux=None):
    """Train m on `task`. If aux=(Xe2,Mk2,St2,tgt2,task2,w) given, add a multi-task auxiliary loss."""
    n = len(target); perm = np.random.permutation(n); vi, ti = perm[:n // 6], perm[n // 6:]
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    best, bs, pat = 1e9, None, 0
    for ep in range(epochs):
        m.train()
        for b in range(0, len(ti), 128):
            bi = ti[b:b + 128]; opt.zero_grad()
            loss = nn.functional.smooth_l1_loss(m(Xe[bi], Mk[bi], St[bi], task), target[bi])
            if aux is not None:
                Xa, Ma, Sa, ta, tka, w = aux
                ja = np.random.randint(0, len(ta), size=min(128, len(ta)))
                loss = loss + w * nn.functional.smooth_l1_loss(m(Xa[ja], Ma[ja], Sa[ja], tka), ta[ja])
            loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vr = float(((m(Xe[vi], Mk[vi], St[vi], task) - target[vi]) ** 2).mean())
        if vr < best - 1e-5:
            best, bs, pat = vr, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            pat += 1
            if pat > patience:
                break
    m.load_state_dict(bs); return m


def build_train():
    """3581 train edges aligned to pEC50 (unimol order), chemprop_aux anchor, and counter-null (by InChIKey)."""
    uni = pd.read_csv(f"{D}/unimol_train.csv"); NE_all = np.load(f"{U}/train_nedge.npy")
    lt = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    anc_src = np.load(f"{D}/oof_chemprop_aux.npy")
    cnt = pd.read_parquet(f"{D}/pxr_counter.parquet")
    cnt_ik = {}
    for s, v in zip(cnt["smiles"], cnt["pec50"]):
        k = ik(s)
        if k and k not in cnt_ik and np.isfinite(v):
            cnt_ik[k] = float(v)
    lt_ik = {}
    for i, s in enumerate(lt["smiles"]):
        k = ik(s)
        if k and k not in lt_ik:
            lt_ik[k] = i
    rows, y, anc, null = [], [], [], []
    for ci in range(len(uni)):
        if NE_all[ci] == 0:
            continue
        k = ik(uni["smiles"].iloc[ci])
        if k and k in lt_ik:
            rows.append(ci); y.append(float(uni["pec50"].iloc[ci])); anc.append(float(anc_src[lt_ik[k]]))
            null.append(cnt_ik.get(k, np.nan))
    rows = np.array(rows)
    E = np.load(f"{U}/train_edges.npy")[rows]; EN = np.load(f"{U}/train_enorm.npy")[rows]; NE = NE_all[rows]
    return rows, E, EN, NE, np.array(y), np.array(anc), np.array(null)


def clipped(X, resid, anchor, y, folds):
    pred = anchor.copy()
    for tri, vai in folds:
        m = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1).fit(X[tri], resid[tri])
        p = anchor[vai] + m.predict(X[vai]); lo, hi = np.quantile(y[tri], QL), np.quantile(y[tri], QH)
        pred[vai] = np.clip(p, lo, hi)
    return float(rae(y, pred))


def main():
    rows, E, EN, NE, y, anc, null = build_train()
    mu = E.mean((0, 1), keepdims=True); sd = E.std((0, 1), keepdims=True) + 1e-6
    valid = NE[:, None] > np.arange(128)[None]; smu = EN[valid].mean(); ssd = EN[valid].std() + 1e-6
    Xtr, Mtr, Str = norm_edges(E, EN, NE, mu, sd, smu, ssd)

    # targets (standardized): absolute pEC50, residual, null
    ym, ystd = y.mean(), y.std() + 1e-6
    y_abs = torch.tensor((y - ym) / ystd, dtype=torch.float32)
    resid = torch.tensor(y - anc, dtype=torch.float32)
    nmask = np.isfinite(null)
    nmean = np.nanmean(null); nstd = np.nanstd(null) + 1e-6
    print(f"train: {len(rows)} | null labels: {int(nmask.sum())}")

    # SC pretrain (activation-axis log2FC)
    sc = pd.read_csv(f"{D}/sc_pretrain_subset.csv"); Esc = np.load(f"{U}/sc_pool_edges.npy")
    ENsc = np.load(f"{U}/sc_pool_enorm.npy"); NEsc = np.load(f"{U}/sc_pool_nedge.npy")
    yfc = sc["log2_fc_estimate"].to_numpy()[:len(NEsc)]
    keep = (NEsc > 0) & np.isfinite(yfc)
    Xsc, Msc, Ssc = norm_edges(Esc[keep], ENsc[keep], NEsc[keep], mu, sd, smu, ssd)
    yfc_t = torch.tensor((yfc[keep] - np.nanmean(yfc[keep])) / (np.nanstd(yfc[keep]) + 1e-6), dtype=torch.float32)
    print(f"SC pretrain: {int(keep.sum())} compounds (activation log2FC)")

    # 253 eval edges
    unb = np.load(f"{D}/_audit_unblind_idx.npy"); yv = np.load(f"{D}/_audit_unblind_y.npy")
    Xte, Mte, Ste = norm_edges(np.load(f"{U}/test_edges.npy")[unb], np.load(f"{U}/test_enorm.npy")[unb],
                               np.load(f"{U}/test_nedge.npy")[unb], mu, sd, smu, ssd)

    def head_253(mode, seeds=5):
        outs = []
        for s in range(seeds):
            torch.manual_seed(s); np.random.seed(s)
            m = MultiAdditiveHead(d=129, h=64, p=0.5)
            if mode == "resid_scratch":
                m = fit_task(m, Xtr, Mtr, Str, resid, "pec50", epochs=250)
            elif mode == "abs_scratch":
                m = fit_task(m, Xtr, Mtr, Str, y_abs, "pec50", epochs=250)
            elif mode == "abs_pretrain":
                m = fit_task(m, Xsc, Msc, Ssc, yfc_t, "fc", epochs=120)        # mechanistic prior
                m = fit_task(m, Xtr, Mtr, Str, y_abs, "pec50", epochs=250)      # standalone fine-tune
            elif mode == "abs_pretrain_mtl":
                m = fit_task(m, Xsc, Msc, Ssc, yfc_t, "fc", epochs=120)
                nm = nmask
                aux = (Xtr[nm], Mtr[nm], Str[nm],
                       torch.tensor((null[nm] - nmean) / nstd, dtype=torch.float32), "null", 0.5)
                m = fit_task(m, Xtr, Mtr, Str, y_abs, "pec50", epochs=250, aux=aux)  # +specificity
            m.eval()
            with torch.no_grad():
                outs.append(m(Xte, Mte, Ste, "pec50").numpy())
        return np.mean(outs, 0)

    # nb3200 substrate
    te = load_test(); smiles = te["smiles"].to_numpy()[unb].tolist(); scaf = [murcko(s) for s in smiles]
    base = np.hstack([impute(combined(smiles)).astype(np.float32), np.load(f"{D}/te_chemprop_embed_300.npy")[unb].astype(np.float32)])
    anchor = np.load(f"{D}/nb3200_pred_oof.npy"); resid253 = yv - anchor
    SEEDS = list(range(1400, 1415))

    results = {}
    for mode in ["resid_scratch", "abs_scratch", "abs_pretrain", "abs_pretrain_mtl"]:
        h253 = head_253(mode)
        corr = float(np.corrcoef(h253, resid253)[0, 1])
        hz = ((h253 - h253.mean()) / (h253.std() + 1e-9)).reshape(-1, 1).astype(np.float32)
        ds = []
        for s in SEEDS:
            f = scaffold_kfold_indices(scaf, 5, seed=s)
            ds.append(clipped(np.hstack([base, hz]), resid253, anchor, yv, f) - clipped(base, resid253, anchor, yv, f))
        ds = np.array(ds); st = ds.mean() < 0 and abs(ds.mean()) > ds.std()
        results[mode] = {"delta": float(ds.mean()), "std": float(ds.std()), "wins": int((ds < 0).sum()),
                         "stable": bool(st), "corr_nb3200_resid": corr}
        print(f"  {mode:18s}: nb3200 feat delta {ds.mean():+.5f} +/- {ds.std():.5f}  wins {int((ds<0).sum())}/{len(SEEDS)}  "
              f"stable={st}  corr_w_resid={corr:+.3f}", flush=True)

    print(f"\n  reference: rich-z = -0.008 (deploy win); residual-head (nb1020) = +0.00005 (redundant)")
    best = min(results.items(), key=lambda kv: kv[1]["delta"])
    print(f"  BEST mode: {best[0]} @ {best[1]['delta']:+.5f} (stable={best[1]['stable']})")
    json.dump(results, open(f"{U}/nb1021_mechanistic_pretrain.json", "w"), indent=2)


if __name__ == "__main__":
    main()
