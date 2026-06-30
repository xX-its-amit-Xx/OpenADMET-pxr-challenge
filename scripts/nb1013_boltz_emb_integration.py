"""nb1013 — does the Boltz-2 cofold TRUNK EMBEDDING add to combined+chempropembed on the 253?
The target-aware, ligand-conditioned structural representation (the one untested axis). 1024-d pooled
(ligand-token s 768 + protein-ligand z block 256) -> PCA<=15 (fit on 513, no label leak). cycle-291 protocol:
chemprop_aux residual + range-clip, 7 seeds. Stable-negative -> FIRST real structural signal vs the sink.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test
from src.pxr.eval import rae, scaffold_kfold_indices
from src.pxr.featurize import combined, impute
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

D = "data/processed"; U = "C:/pxr_struct/boltz"
SEEDS = [42, 101, 202, 303, 404, 505, 606]
QL, QH = 0.05, 0.98


def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else None


def clipped(X, resid, anchor, y, folds):
    pred = anchor.copy()
    for tri, vai in folds:
        m = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1).fit(X[tri], resid[tri])
        p = anchor[vai] + m.predict(X[vai]); lo, hi = np.quantile(y[tri], QL), np.quantile(y[tri], QH)
        pred[vai] = np.clip(p, lo, hi)
    return float(rae(y, pred))


def main():
    te = load_test(); unb = np.load(f"{D}/_audit_unblind_idx.npy"); y = np.load(f"{D}/_audit_unblind_y.npy")
    smiles = te["smiles"].to_numpy()[unb].tolist(); scaf = [murcko(s) for s in smiles]
    anchor = np.load(f"{D}/te_chemprop_aux.npy")[unb]
    Xc = impute(combined(smiles)).astype(np.float32)
    emb_cp = np.load(f"{D}/te_chemprop_embed_300.npy")[unb].astype(np.float32)
    base = np.hstack([Xc, emb_cp])

    bz = np.nan_to_num(np.load(f"{U}/boltz_emb_513.npy").astype(np.float32))   # (513, 1024)
    for K in (15, 30):
        bz_pca = PCA(n_components=K, random_state=0).fit_transform(StandardScaler().fit_transform(bz))
        bz_u = bz_pca[unb].astype(np.float32)
        base_bz = np.hstack([base, bz_u])
        resid = y - anchor
        ds = []
        for seed in SEEDS:
            folds = scaffold_kfold_indices(scaf, n_splits=5, seed=seed)
            rb = clipped(base, resid, anchor, y, folds); rz = clipped(base_bz, resid, anchor, y, folds)
            ds.append(rz - rb)
        ds = np.array(ds); stable = ds.mean() < 0 and abs(ds.mean()) > ds.std()
        print(f"boltz_emb PCA{K}: base->+boltz mean={ds.mean():+.5f} std={ds.std():.5f} "
              f"wins={int((ds<0).sum())}/7 stable={stable}")
        if K == 15:
            res = {"K15_mean": float(ds.mean()), "K15_std": float(ds.std()), "K15_stable": bool(stable),
                   "K15_wins": int((ds < 0).sum())}
    print("=" * 60)
    print(">>> BOLTZ COFOLD EMBEDDING BREAKS THE SINK -> target-aware structure REAL" if res["K15_stable"]
          else ">>> boltz cofold embedding absorbed too -> single-seed target-aware structure redundant on n=253")
    print("=" * 60)
    json.dump(res, open(f"{U}/nb1013_boltz_emb_integration.json", "w"), indent=2)
    print(f"anchor {rae(y,anchor):.4f}; base {base.shape}; saved -> {U}/nb1013_boltz_emb_integration.json")


if __name__ == "__main__":
    main()
