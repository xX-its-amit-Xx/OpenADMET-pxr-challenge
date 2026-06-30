"""nb1114 — APPROACH #2: pairwise DIFFERENTIAL SAR (DeepDelta-style) for the analog test.

Instead of predicting absolute pEC50, predict Δ-potency between a test compound and its labeled TRAIN neighbors
(a smaller, easier target for analogs; uses bounded/weak data losslessly), then reconstruct absolute pEC50 by
averaging neighbor_pEC50 + predicted_Δ. Cross-series robust by construction (relative SAR within local neighborhoods).
Train a Δ-model on train-train pairs (feature differences -> Δy). Validate vs nb3200 pooled AND leave-series-out.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test, load_train
from src.pxr.featurize import combined, impute
from src.pxr.chem import morgan_fp_batch
from src.pxr.eval import rae
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb

P = "data/processed"; rng = np.random.default_rng(0)


def fpf(sm): return (morgan_fp_batch(sm).astype(np.float32) > 0).astype(np.float32)


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test().reset_index(drop=True)
    unb = np.load(f"{P}/_audit_unblind_idx.npy"); y = np.load(f"{P}/_audit_unblind_y.npy")
    anchor = np.load(f"{P}/nb3200_pred_oof.npy"); err = y - anchor
    ytr = tr["pec50"].to_numpy()
    Xtr = np.hstack([impute(combined(tr["smiles"].tolist())), np.load(f"{P}/tr_chemprop_embed_300.npy")])
    Xte = np.hstack([impute(combined(te["smiles"].to_numpy()[unb].tolist())),
                     np.load(f"{P}/te_chemprop_embed_300.npy")[unb]])
    Ftr = fpf(tr["smiles"].tolist()); Fte = fpf(te["smiles"].to_numpy()[unb].tolist())

    # train-train similar pairs (Tanimoto 0.3-0.97) -> delta model
    print("sampling train-train pairs...", flush=True)
    sims = (Ftr @ Ftr.T)
    s = Ftr.sum(1); sims = sims / np.clip(s[:, None] + s[None, :] - sims, 1, None)
    ia, ib = [], []
    for i in range(len(tr)):
        cand = np.where((sims[i] > 0.3) & (sims[i] < 0.97))[0]
        if len(cand) == 0: continue
        pick = rng.choice(cand, min(60, len(cand)), replace=False)
        ia += [i] * len(pick); ib += list(pick)
    ia, ib = np.array(ia), np.array(ib)
    if len(ia) > 300000:
        sel = rng.choice(len(ia), 300000, replace=False); ia, ib = ia[sel], ib[sel]
    Xd = Xtr[ia] - Xtr[ib]; yd = ytr[ia] - ytr[ib]
    print(f"  {len(ia)} pairs", flush=True)
    m = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, n_jobs=4, verbose=-1).fit(Xd, yd)

    # test inference: each test cpd -> top-K train neighbors, pEC50 = mean(neighbor + delta)
    K = 10
    sim_te = (Fte @ Ftr.T); sim_te = sim_te / np.clip(Fte.sum(1)[:, None] + s[None, :] - sim_te, 1, None)
    pred = np.zeros(len(unb))
    for i in range(len(unb)):
        nb = np.argsort(sim_te[i])[::-1][:K]
        d = m.predict(Xte[i][None, :] - Xtr[nb])
        w = sim_te[i][nb] ** 2 + 1e-6
        pred[i] = np.sum(w * (ytr[nb] + d)) / np.sum(w)
    print(f"\nnb3200 anchor RAE {rae(y, anchor):.4f}")
    print(f"pairwise-SAR standalone RAE {rae(y, pred):.4f}")
    print(f"corr(pairwise-SAR, nb3200 error) {np.corrcoef(pred, err)[0,1]:+.3f}")

    # blend pooled + leave-series-out
    bb = rae(y, anchor)
    for w in np.linspace(0, 1, 41): bb = min(bb, rae(y, (1 - w) * anchor + w * pred))
    Z = PCA(20, random_state=0).fit_transform(StandardScaler().fit_transform(Xte))
    series = KMeans(6, n_init=5, random_state=0).fit_predict(Z)
    oof = anchor.copy()
    for kk in range(6):
        trn = series != kk; val = series == kk
        if val.sum() < 5: continue
        bw, bbk = 0, rae(y[trn], anchor[trn])
        for w in np.linspace(0, 1, 41):
            r = rae(y[trn], (1 - w) * anchor[trn] + w * pred[trn])
            if r < bbk: bbk, bw = r, w
        oof[val] = (1 - bw) * anchor[val] + bw * pred[val]
    print(f"blend pooled {bb:.4f} (delta {bb-rae(y,anchor):+.4f}) | blend X-SERIES {rae(y,oof):.4f} (delta {rae(y,oof)-rae(y,anchor):+.4f})")
    json.dump({"standalone": float(rae(y, pred)), "blend_pooled": float(bb), "blend_xseries": float(rae(y, oof)),
               "corr_err": float(np.corrcoef(pred, err)[0, 1])}, open(f"{P}/nb1114_pairwise.json", "w"), indent=2)
    print("GATE: real lever if blend_xseries < 0.4416 (held-out series).")


if __name__ == "__main__":
    main()
