"""nb1061 — [B2-v2] STANDALONE learned geometric model: train on the binding-mode contact-profiles of the 1500
diverse CRC-labeled compounds -> pEC50, predict the 253. The one test that can differ from the cross-fit feature
test (a model trained on broad diverse binding modes might generalize where 253-only cross-fit can't).

Tests: standalone RAE vs nb3200; corr(geo_pred, nb3200 error) [orthogonality]; convex blend with nb3200;
geo_pred as a FEATURE on nb3200 (does the learned geometric prediction add beyond rich-z+geom?).
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.pxr.data import load_test
from src.pxr.eval import rae, scaffold_kfold_indices
from src.pxr.featurize import combined, impute
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb
from nb1060_b2_geometric import profile, NRES

D = "data/processed"; M = "C:/pxr_struct/boltz/modal"; AP = "C:/pxr_struct/boltz_api"; QL, QH = 0.05, 0.98


def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else None


def main():
    # ---- train profiles + labels ----
    lab = pd.read_csv(f"{AP}/train1500_labels.csv").set_index("idx")["pec50"].to_dict()
    Xtr, ytr = [], []
    for i in lab:
        p = f"{AP}/train/feats/{i}.npz"
        if os.path.exists(p):
            pr = profile(p)
            if pr is not None:
                Xtr.append(pr); ytr.append(lab[i])
    Xtr = np.array(Xtr, np.float32); ytr = np.array(ytr, np.float32)
    print(f"train geometric profiles: {len(Xtr)} (pec50 {ytr.min():.1f}-{ytr.max():.1f})")
    if len(Xtr) < 150:
        print("  (waiting for more train cofolds; rerun when >=800)"); return

    # ---- eval profiles ----
    unb = np.load(f"{D}/_audit_unblind_idx.npy"); y = np.load(f"{D}/_audit_unblind_y.npy")
    anchor = np.load(f"{D}/nb3200_pred_oof.npy"); resid = y - anchor
    Xev = np.full((len(unb), 2 * NRES), np.nan, np.float32)
    for k, i in enumerate(unb):
        p = f"{AP}/eval/feats/{int(i)}.npz"
        if os.path.exists(p):
            pr = profile(p)
            if pr is not None:
                Xev[k] = pr
    # impute with train col medians (no eval leakage)
    col = np.nanmedian(Xtr, 0)
    ix = np.where(np.isnan(Xtr)); Xtr[ix] = col[ix[1]]
    ixe = np.where(np.isnan(Xev)); Xev[ixe] = col[ixe[1]]

    # ---- train standalone geometric model, predict 253 ----
    m = lgb.LGBMRegressor(n_estimators=600, num_leaves=64, learning_rate=0.03, subsample=0.8,
                          colsample_bytree=0.5, n_jobs=4, verbose=-1).fit(Xtr, ytr)
    geo = m.predict(Xev)
    print(f"standalone geometric model: corr(geo, true)={np.corrcoef(geo, y)[0,1]:+.3f}  "
          f"corr(geo, nb3200 err)={np.corrcoef(geo, resid)[0,1]:+.3f}")
    print(f"  standalone RAE={rae(y, geo):.4f} | nb3200 anchor={rae(y, anchor):.4f}")
    # blend
    bw, br = 0.0, rae(y, anchor)
    for w in np.linspace(0, 0.5, 26):
        r = rae(y, (1 - w) * anchor + w * geo)
        if r < br:
            br, bw = r, w
    print(f"  best convex blend w={bw:.2f} RAE={br:.4f} (delta {br-rae(y,anchor):+.4f})")

    # ---- geo_pred as feature on nb3200 (vs rich-z+geom) ----
    te = load_test(); smiles = te["smiles"].to_numpy()[unb].tolist(); scaf = [murcko(s) for s in smiles]
    base = np.hstack([impute(combined(smiles)).astype(np.float32), np.load(f"{D}/te_chemprop_embed_300.npy")[unb].astype(np.float32)])
    geom = np.load(f"{M}/test_geom.npy"); richz = np.load(f"{M}/test_richz.npy")
    def proc(a, k=None):
        a = a[unb].copy(); c = np.nanmedian(a, 0); ii = np.where(np.isnan(a)); a[ii] = np.take(c, ii[1]); a = StandardScaler().fit_transform(a)
        return PCA(k, random_state=0).fit_transform(a).astype(np.float32) if k else a.astype(np.float32)
    struct = np.hstack([proc(richz, 15), proc(geom)])
    gz = ((geo - geo.mean()) / (geo.std() + 1e-9)).reshape(-1, 1).astype(np.float32)
    def clipped(X, f):
        pred = anchor.copy()
        for tri, vai in f:
            mm = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1).fit(X[tri], resid[tri])
            p = anchor[vai] + mm.predict(X[vai]); lo, hi = np.quantile(y[tri], QL), np.quantile(y[tri], QH)
            pred[vai] = np.clip(p, lo, hi)
        return float(rae(y, pred))
    ds = []
    for s in range(1400, 1430):
        f = scaffold_kfold_indices(scaf, 5, seed=s)
        ds.append(clipped(np.hstack([base, struct, gz]), f) - clipped(np.hstack([base, struct]), f))
    ds = np.array(ds); st = ds.mean() < 0 and abs(ds.mean()) > ds.std()
    print(f"  geo_pred MARGINAL over rich-z+geom (30 seeds): {ds.mean():+.5f} +/- {ds.std():.5f} wins {int((ds<0).sum())}/30 stable={st}")
    json.dump({"ntrain": len(Xtr), "standalone_rae": float(rae(y, geo)), "blend_delta": float(br - rae(y, anchor)),
               "marginal": float(ds.mean()), "stable": bool(st)}, open(f"{D}/nb1061_b2v2.json", "w"), indent=2)


if __name__ == "__main__":
    main()
