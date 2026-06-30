"""nb1120 — MULTI-FIDELITY (user's idea): proxy-label the 8131 single-conc-only compounds from their REAL single-point
measurement -> CRC mapping (Stage-1 gate: R2 0.64), then train on 4139 real-CRC + 8131 proxy-CRC, predict the 253.

Stage 1: single-point readouts (log2fc, fdr, cohens) -> CRC pEC50, fit on the 2739 overlap (R2 0.64).
Stage 2: apply to 8131 single-conc-only -> proxy CRC labels (REAL activity info, not chemistry-pseudo-labels).
Stage 3: combined-LGBM on 4139 (w=1) + 8131 proxy (w in {0,0.3,0.5,1.0}), predict 253. Does the extra real-activity
data help? + does it ADD to nb3200 (blend + cross-series, cycle-305)? NOTE base is combined-only (no chempropembed for
the 8131), so standalone ~0.6; the deploy question is the blend/corr-with-error vs nb3200.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test, load_train, load_single_conc
from src.pxr.featurize import combined, impute
from src.pxr.eval import rae
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb

P = "data/processed"


def ik(s):
    m = Chem.MolFromSmiles(str(s)); return Chem.MolToInchiKey(m) if m else None


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test().reset_index(drop=True)
    sc = load_single_conc()
    unb = np.load(f"{P}/_audit_unblind_idx.npy"); y = np.load(f"{P}/_audit_unblind_y.npy")
    anchor = np.load(f"{P}/nb3200_pred_oof.npy"); err = y - anchor

    sc["ik"] = sc["smiles"].map(ik); tr["ik"] = tr["smiles"].map(ik)
    agg = sc.groupby("ik").agg(log2fc_max=("log2_fc_estimate", "max"), log2fc_med=("median_log2_fc", "median"),
        neglogfdr_max=("neg_log10_fdr", "max"), cohens_max=("cohens_d", "max"),
        conc_max=("concentration_M", "max"), nrows=("log2_fc_estimate", "size")).reset_index()
    agg["smiles"] = sc.drop_duplicates("ik").set_index("ik")["smiles"].reindex(agg["ik"]).values
    FEAT = ["log2fc_max", "log2fc_med", "neglogfdr_max", "cohens_max", "conc_max", "nrows"]

    # Stage 1: single-point -> CRC on overlap
    ov = tr[["ik", "pec50"]].drop_duplicates("ik").merge(agg, on="ik")
    s1 = HistGradientBoostingRegressor(max_iter=300).fit(ov[FEAT].values, ov["pec50"].values)
    # Stage 2: proxy-label single-conc-only
    only = agg[~agg["ik"].isin(set(tr["ik"]))].dropna(subset=["smiles"]).reset_index(drop=True)
    proxy = s1.predict(only[FEAT].values)
    print(f"single-conc-only proxy-labeled: {len(only)} | proxy pEC50 range {proxy.min():.2f}..{proxy.max():.2f} "
          f"(median {np.median(proxy):.2f})", flush=True)

    # Stage 3: combined features (no chempropembed for the 8131)
    print("featurizing (combined) train + proxy + test...", flush=True)
    Xtr = impute(combined(tr["smiles"].tolist())); ytr = tr["pec50"].to_numpy()
    Xpx = impute(combined(only["smiles"].tolist()))
    Xte = impute(combined(te["smiles"].to_numpy()[unb].tolist()))
    low = y <= np.quantile(y, 0.25)

    series = KMeans(6, n_init=5, random_state=0).fit_predict(
        PCA(20, random_state=0).fit_transform(StandardScaler().fit_transform(Xte)))

    def fit_pred(w):
        ps = []
        X = np.vstack([Xtr, Xpx]); yy = np.concatenate([ytr, proxy])
        sw = np.concatenate([np.ones(len(ytr)), np.full(len(proxy), w)])
        for s in range(4):
            m = lgb.LGBMRegressor(n_estimators=600, num_leaves=64, learning_rate=0.04, subsample=0.8,
                                  colsample_bytree=0.8, n_jobs=4, verbose=-1, random_state=s)
            if w == 0: m.fit(Xtr, ytr)
            else: m.fit(X, yy, sample_weight=sw)
            ps.append(m.predict(Xte))
        return np.mean(ps, 0)

    print(f"\nnb3200 anchor RAE {rae(y, anchor):.4f}")
    print(f"{'proxy_w':>8s} {'RAE':>7s} {'lowMAE':>7s} {'predmin':>8s} {'blend_pool':>11s} {'blend_xser':>11s} {'corr_err':>9s}")
    out = {}
    for w in [0.0, 0.3, 0.5, 1.0]:
        p = fit_pred(w)
        bp = rae(y, anchor)
        for ww in np.linspace(0, 1, 41): bp = min(bp, rae(y, (1 - ww) * anchor + ww * p))
        oof = anchor.copy()
        for kk in range(6):
            trn = series != kk; val = series == kk
            if val.sum() < 5: continue
            bw, bb = 0, rae(y[trn], anchor[trn])
            for ww in np.linspace(0, 1, 41):
                r = rae(y[trn], (1 - ww) * anchor[trn] + ww * p[trn])
                if r < bb: bb, bw = r, ww
            oof[val] = (1 - bw) * anchor[val] + bw * p[val]
        xs = rae(y, oof); c = np.corrcoef(p, err)[0, 1]
        out[f"w{w}"] = dict(rae=float(rae(y, p)), blend_pool=float(bp), blend_xser=float(xs))
        print(f"{w:>8.1f} {rae(y,p):>7.4f} {np.mean(np.abs(p[low]-y[low])):>7.3f} {p.min():>8.2f} "
              f"{bp:>11.4f} {xs:>11.4f} {c:>+9.3f}")
    print("\n(w=0 is the no-augmentation baseline. GATE: a real lever needs blend_xser < 0.4416 (held-out series).)")
    json.dump({"anchor": float(rae(y, anchor)), "stage1_r2_note": "0.644", **out},
              open(f"{P}/nb1120_multifidelity.json", "w"), indent=2)


if __name__ == "__main__":
    main()
