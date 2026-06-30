"""nb1121 — CASCADE / low-tail SPECIALIST head (user idea): route nb3200's low-predicted compounds through a 2nd model.

For test compounds where nb3200 predicts below threshold T (the high-variance low tail), replace/blend with a
second-model prediction. Candidate 2nd heads (each may have LOWER VARIANCE or better low-range specialization):
  low_specialist : LGBM trained ONLY on train pEC50<5 (combined+chempropembed) - specialized on the low range
  knn_mean       : mean pEC50 of k=8 nearest train neighbors (low-variance robust local estimate = variance fix)
  quantile_low   : LGBM quantile alpha=0.3 (downward, for the over-predicted low tail)
Cascade: final = nb3200 for pred>=T, 2nd-head for pred<T. Tune T + head CROSS-SERIES (leave-series-out, cycle-305).
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

P = "data/processed"


def fpf(sm): return (morgan_fp_batch(sm).astype(np.float32) > 0).astype(np.float32)


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test().reset_index(drop=True)
    unb = np.load(f"{P}/_audit_unblind_idx.npy"); y = np.load(f"{P}/_audit_unblind_y.npy")
    anchor = np.load(f"{P}/nb3200_pred_oof.npy")
    ytr = tr["pec50"].to_numpy()
    Xtr = np.hstack([impute(combined(tr["smiles"].tolist())), np.load(f"{P}/tr_chemprop_embed_300.npy")])
    Xte = np.hstack([impute(combined(te["smiles"].to_numpy()[unb].tolist())),
                     np.load(f"{P}/te_chemprop_embed_300.npy")[unb]])

    def bag(Xa, ya, w=None, obj=None, alpha=None):
        ps = []
        for s in range(4):
            kw = dict(n_estimators=500, num_leaves=48, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                      n_jobs=4, verbose=-1, random_state=s)
            if obj == "quantile": kw.update(objective="quantile", alpha=alpha)
            m = lgb.LGBMRegressor(**kw); m.fit(Xa, ya); ps.append(m.predict(Xte))
        return np.mean(ps, 0)

    # 2nd-head predictions on the 253
    lowmask_tr = ytr < 5.0
    heads = {
        "low_specialist": bag(Xtr[lowmask_tr], ytr[lowmask_tr]),
        "quantile_low": bag(Xtr, ytr, obj="quantile", alpha=0.3),
    }
    # knn_mean
    Ftr = fpf(tr["smiles"].tolist()); Fte = fpf(te["smiles"].to_numpy()[unb].tolist())
    sim = (Fte @ Ftr.T); s = Ftr.sum(1); sim = sim / np.clip(Fte.sum(1)[:, None] + s[None, :] - sim, 1, None)
    knn = np.array([np.mean(ytr[np.argsort(sim[i])[::-1][:8]]) for i in range(len(unb))])
    heads["knn_mean"] = knn

    series = KMeans(6, n_init=5, random_state=0).fit_predict(
        PCA(20, random_state=0).fit_transform(StandardScaler().fit_transform(Xte)))
    base = rae(y, anchor)
    print(f"nb3200 anchor RAE {base:.4f}\n")
    print(f"{'head':16s} {'best_T':>7s} {'cascade_pooled':>15s} {'cascade_XSERIES':>16s}")
    out = {}
    for hname, hp in heads.items():
        # pooled: best (T, blend) over the low set
        def cascade(T, w, idx=None):
            p = anchor.copy(); m = anchor < T
            if idx is not None: m = m & np.isin(np.arange(len(y)), idx)
            p[m] = (1 - w) * anchor[m] + w * hp[m]; return p
        bestp, bT = base, None
        for T in np.quantile(anchor, [0.15, 0.25, 0.35, 0.5]):
            for w in np.linspace(0, 1, 21):
                r = rae(y, cascade(T, w))
                if r < bestp: bestp, bT = r, T
        # cross-series: tune (T,w) on K-1 series, apply held-out
        oof = anchor.copy()
        for kk in range(6):
            trn = np.where(series != kk)[0]; val = np.where(series == kk)[0]
            if len(val) < 5: continue
            bb, bTw = rae(y[trn], anchor[trn]), (anchor.min() - 1, 0)
            for T in np.quantile(anchor, [0.15, 0.25, 0.35, 0.5]):
                for w in np.linspace(0, 1, 21):
                    mtr = (anchor[trn] < T)
                    pr = anchor[trn].copy(); pr[mtr] = (1 - w) * anchor[trn][mtr] + w * hp[trn][mtr]
                    r = rae(y[trn], pr)
                    if r < bb: bb, bTw = r, (T, w)
            T, w = bTw; mval = anchor[val] < T
            pv = anchor[val].copy(); pv[mval] = (1 - w) * anchor[val][mval] + w * hp[val][mval]
            oof[val] = pv
        xs = rae(y, oof)
        out[hname] = dict(cascade_pooled=float(bestp), cascade_xseries=float(xs))
        print(f"{hname:16s} {bT if bT else 0:>7.2f} {bestp:>15.4f} {xs:>16.4f}")
    print(f"\n(GATE: real lever if cascade_XSERIES < {base:.4f} on held-out series. Pooled overfits.)")
    json.dump({"anchor": base, **out}, open(f"{P}/nb1121_cascade.json", "w"), indent=2)


if __name__ == "__main__":
    main()
