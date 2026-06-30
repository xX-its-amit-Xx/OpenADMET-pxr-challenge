"""nb1008 — does the CLOOME phenotypic (cell-painting) embedding ADD to combined+chempropembed on the 253?
cycle-291 protocol (nb1006/nb1007 twin): chemprop_aux residual + range-clip, multi-seed, PCA<=15 (fit on 513).
Phenotypic axis = most orthogonal-by-construction. Stable-negative -> phenotypic biology breaks the sink.
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

D = "data/processed"; U = "C:/pxr_struct/cloome"
SEEDS = [42, 101, 202, 303, 404, 505, 606]
QL, QH = 0.05, 0.98
N_PCA = 15


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

    cl513 = np.nan_to_num(np.load(f"{U}/cloome_emb_513.npy").astype(np.float32))
    # degeneracy check: how many compounds are distinguishable
    row_var = cl513.var(axis=1).mean(); uniq = len(np.unique(np.round(cl513, 4), axis=0))
    print(f"cloome emb: shape{cl513.shape} per-row-var={row_var:.4f} unique_rows={uniq}/{len(cl513)}")
    cl_pca = PCA(n_components=N_PCA, random_state=0).fit_transform(StandardScaler().fit_transform(cl513))
    cl_u = cl_pca[unb].astype(np.float32)
    base_cl = np.hstack([base, cl_u])
    print(f"anchor {rae(y,anchor):.4f}; base={base.shape} +cloome_pca{N_PCA}={base_cl.shape}\n")
    resid = y - anchor
    rows = []
    for seed in SEEDS:
        folds = scaffold_kfold_indices(scaf, n_splits=5, seed=seed)
        rb = clipped(base, resid, anchor, y, folds); rc = clipped(base_cl, resid, anchor, y, folds)
        rows.append({"seed": seed, "base": round(rb, 4), "base_cloome": round(rc, 4), "delta": round(rc - rb, 5)})
        print(f"  seed {seed}: base={rb:.4f} +cloome={rc:.4f} delta={rc-rb:+.5f}")
    d = np.array([r["delta"] for r in rows]); stable = d.mean() < 0 and abs(d.mean()) > d.std()
    print("\n" + "=" * 60)
    print(f"cloome-PCA{N_PCA} on chemprop substrate: mean={d.mean():+.5f} std={d.std():.5f} wins={int((d<0).sum())}/{len(SEEDS)} stable={stable}")
    print(">>> CLOOME PHENOTYPIC BREAKS THE SINK -> phenotypic axis REAL" if stable
          else ">>> cloome absorbed too -> even phenotypic (cell-painting) emb redundant on the n=253 substrate")
    print("=" * 60)
    json.dump({"rows": rows, "delta_mean": float(d.mean()), "delta_std": float(d.std()), "stable": bool(stable),
               "row_var": float(row_var), "unique_rows": int(uniq)},
              open(f"{U}/nb1008_cloome_integration.json", "w"), indent=2)
    print(f"saved -> {U}/nb1008_cloome_integration.json")


if __name__ == "__main__":
    main()
