"""nb1113 — heteroscedastic CURVE-QUALITY weighting (research lead #2; the cleanest untried lever).

Literature: weighting each training pEC50 by its dose-response FIT QUALITY (inverse measurement variance) gave up to
22% RMSE reduction on 31/40 datasets. We have pec50_se + emax for all 4139. Untried as a LOSS WEIGHT here. It's
inherently CROSS-SERIES robust (about label noise, not unblind-specific structure) -> safe vs the cycle-305 transfer risk.

Strong base = combined + chempropembed. Weight schemes: uniform | inv-variance (1/se^2) | inv-se | capped |
emax-gated (down-weight low-efficacy/near-censored curves). For each: standalone RAE, blend-with-nb3200, AND
LEAVE-SERIES-OUT cross-series blend (does the gain hold on held-out chemical series, per cycle-305).
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test, load_train
from src.pxr.featurize import combined, impute
from src.pxr.eval import rae
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb

P = "data/processed"


def bag(Xtr, ytr, w, Xte, nseed=4):
    ps = []
    for s in range(nseed):
        m = lgb.LGBMRegressor(n_estimators=600, num_leaves=64, learning_rate=0.04, subsample=0.8,
                              colsample_bytree=0.8, n_jobs=4, verbose=-1, random_state=s)
        m.fit(Xtr, ytr, sample_weight=w); ps.append(m.predict(Xte))
    return np.mean(ps, 0)


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test().reset_index(drop=True)
    unb = np.load(f"{P}/_audit_unblind_idx.npy"); y = np.load(f"{P}/_audit_unblind_y.npy")
    anchor = np.load(f"{P}/nb3200_pred_oof.npy"); err = y - anchor
    ytr = tr["pec50"].to_numpy(); se = tr["pec50_se"].to_numpy(); emax = tr["emax"].to_numpy()
    se = np.where(np.isfinite(se) & (se > 0), se, np.nanmedian(se))

    Xtr = np.hstack([impute(combined(tr["smiles"].tolist())), np.load(f"{P}/tr_chemprop_embed_300.npy")])
    Xte = np.hstack([impute(combined(te["smiles"].to_numpy()[unb].tolist())),
                     np.load(f"{P}/te_chemprop_embed_300.npy")[unb]])
    n = len(ytr)
    schemes = {
        "uniform": np.ones(n),
        "inv_variance": 1.0 / (se ** 2),
        "inv_se": 1.0 / se,
        "inv_var_capped": np.clip(1.0 / (se ** 2), None, np.quantile(1.0 / (se ** 2), 0.9)),
        "emax_gated": (1.0 / se) * np.clip((emax - np.nanmin(emax)) / (np.nanmedian(emax) - np.nanmin(emax) + 1e-6), 0.3, 1.5),
    }
    for k in schemes: schemes[k] = schemes[k] * n / schemes[k].sum()

    # chemical series for leave-series-out (cycle-305)
    Z = PCA(20, random_state=0).fit_transform(StandardScaler().fit_transform(Xte))
    series = KMeans(6, n_init=5, random_state=0).fit_predict(Z)

    print(f"nb3200 anchor RAE {rae(y, anchor):.4f} | n_train {n}\n")
    print(f"{'weights':16s} {'standalone':>10s} {'blend_pooled':>12s} {'blend_xseries':>14s} {'corr_err':>9s}")
    out = {}
    for name, w in schemes.items():
        p = bag(Xtr, ytr, w, Xte)
        # pooled blend
        bb = rae(y, anchor)
        for ww in np.linspace(0, 1, 41): bb = min(bb, rae(y, (1 - ww) * anchor + ww * p))
        # leave-series-out blend: fit blend weight on K-1 series, apply to held-out, aggregate
        oof = anchor.copy()
        for kk in range(6):
            trn = series != kk; val = series == kk
            if val.sum() < 5: continue
            bw, bbk = 0, rae(y[trn], anchor[trn])
            for ww in np.linspace(0, 1, 41):
                r = rae(y[trn], (1 - ww) * anchor[trn] + ww * p[trn])
                if r < bbk: bbk, bw = r, ww
            oof[val] = (1 - bw) * anchor[val] + bw * p[val]
        xs = rae(y, oof)
        c = np.corrcoef(p, err)[0, 1]
        out[name] = dict(standalone=float(rae(y, p)), blend_pooled=float(bb), blend_xseries=float(xs),
                         delta_xseries=float(xs - rae(y, anchor)))
        print(f"{name:16s} {rae(y,p):>10.4f} {bb:>12.4f} {xs:>14.4f} {c:>+9.3f}")
    print(f"\n(blend_xseries is the HONEST deploy number: blend weight fit on OTHER series, applied to held-out)")
    print("GATE: real lever if a weighting's blend_xseries < 0.4416 (held-out series), not just pooled.")
    json.dump({"anchor": float(rae(y, anchor)), **out}, open(f"{P}/nb1113_curvequality.json", "w"), indent=2)


if __name__ == "__main__":
    main()
