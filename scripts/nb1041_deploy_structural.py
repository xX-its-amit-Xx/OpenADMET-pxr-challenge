"""nb1041 — [A3] deploy the validated structural feature (rich-z + geom) on nb3200, and test the CHEAPEST
extraction improvement for free: the full per-residue RMSF profile (293-dim, already saved, only its helix-12
summary was used in geom). Does the RMSF profile add a STABLE marginal over rich-z+geom? If yes -> cheap win and
B2 motivated; if not -> cofold features saturated at ~-0.008.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test
from src.pxr.eval import rae, scaffold_kfold_indices
from src.pxr.featurize import combined, impute
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb

D = "data/processed"; M = "C:/pxr_struct/boltz/modal"; QL, QH = 0.05, 0.98


def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else None


def clipped(X, resid, anchor, y, folds):
    pred = anchor.copy()
    for tri, vai in folds:
        m = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1).fit(X[tri], resid[tri])
        p = anchor[vai] + m.predict(X[vai]); lo, hi = np.quantile(y[tri], QL), np.quantile(y[tri], QH)
        pred[vai] = np.clip(p, lo, hi)
    return float(rae(y, pred)), pred


def main():
    geom = np.load(f"{M}/test_geom.npy"); richz = np.load(f"{M}/test_richz.npy"); rmsf = np.load(f"{M}/test_rmsf.npy")
    unb = np.load(f"{D}/_audit_unblind_idx.npy"); y = np.load(f"{D}/_audit_unblind_y.npy")
    anchor = np.load(f"{D}/nb3200_pred_oof.npy"); resid = y - anchor
    te = load_test(); smiles = te["smiles"].to_numpy()[unb].tolist(); scaf = [murcko(s) for s in smiles]
    base = np.hstack([impute(combined(smiles)).astype(np.float32), np.load(f"{D}/te_chemprop_embed_300.npy")[unb].astype(np.float32)])

    def std(a):
        a = a[unb].copy(); col = np.nanmedian(a, 0); idx = np.where(np.isnan(a)); a[idx] = np.take(col, idx[1])
        return StandardScaler().fit_transform(a).astype(np.float32)
    def pca(a, k):
        a = a[unb].copy(); col = np.nanmedian(a, 0); idx = np.where(np.isnan(a)); a[idx] = np.take(col, idx[1])
        sc = StandardScaler().fit(a); return PCA(n_components=k, random_state=0).fit_transform(sc.transform(a)).astype(np.float32)

    gz = std(geom); rz = pca(richz, 15); rmz = pca(rmsf, 10)
    struct = np.hstack([rz, gz])                       # the validated structural block (rich-z + geom)

    SEEDS = list(range(1400, 1430))
    def block_test(extra, baseextra, label):
        ds = []
        for s in SEEDS:
            f = scaffold_kfold_indices(scaf, 5, seed=s)
            r1, _ = clipped(np.hstack([base, baseextra, extra]), resid, anchor, y, f) if baseextra is not None else clipped(np.hstack([base, extra]), resid, anchor, y, f)
            r0, _ = clipped(np.hstack([base, baseextra]), resid, anchor, y, f) if baseextra is not None else clipped(base, resid, anchor, y, f)
            ds.append(r1 - r0)
        ds = np.array(ds); st = ds.mean() < 0 and abs(ds.mean()) > ds.std()
        print(f"  {label:36s}: {ds.mean():+.5f} +/- {ds.std():.5f} wins {int((ds<0).sum())}/30 stable={st}")
        return ds.mean()

    print("structural blocks on nb3200 (30 seeds):")
    block_test(struct, None, "rich-z + geom (DEPLOY block)")
    block_test(rmz, struct, "rmsf-profile MARGINAL over (rz+geom)")
    block_test(np.hstack([struct, rmz]), None, "rich-z + geom + rmsf-profile")

    # ---- deploy: cross-fit structural residual prediction for the 253 (and save a 513 deployable where covered) ----
    r_struct, pred253 = [], None
    f0 = scaffold_kfold_indices(scaf, 5, seed=1234)
    r, pred253 = clipped(np.hstack([base, struct]), resid, anchor, y, f0)
    print(f"\nDEPLOY (rich-z+geom) cross-fit RAE on 253 = {r:.4f}  vs anchor {rae(y, anchor):.4f}")
    np.save(f"{D}/nb1041_struct_pred253.npy", pred253)
    json.dump({"anchor_253": float(rae(y, anchor)), "struct_253": float(r)}, open(f"{D}/nb1041_deploy.json", "w"), indent=2)
    print(f"saved deployable structural prediction (253) -> nb1041_struct_pred253.npy")


if __name__ == "__main__":
    main()
