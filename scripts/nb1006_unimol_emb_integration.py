"""nb1006 — THE decisive representation test: do Uni-Mol's 512-dim 3D EMBEDDINGS add to
combined+chempropembed (the chemprop sink) on the 253? The prediction washed out (nb1004); the
full 3D representation is a different (2D-GNN vs 3D-SE3) projection that might NOT be absorbed.
chemprop_aux residual + range-clip, multi-seed (nb982 protocol). If stable-negative -> the
representation path breaks the sink = first real ladder lever from the GPU work.
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
from sklearn.impute import SimpleImputer

D = "data/processed"; U = "C:/pxr_struct/unimol"
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
    um = np.load(f"{U}/unimol_emb_test.npy")[unb].astype(np.float32)
    um = np.clip(np.nan_to_num(SimpleImputer(strategy="median").fit_transform(um)), -1e6, 1e6).astype(np.float32)
    base_um = np.hstack([base, um])
    print(f"anchor {rae(y,anchor):.4f}; base={base.shape} +unimol_emb={base_um.shape}\n")
    resid = y - anchor
    rows = []
    for seed in SEEDS:
        folds = scaffold_kfold_indices(scaf, n_splits=5, seed=seed)
        rb = clipped(base, resid, anchor, y, folds); ru = clipped(base_um, resid, anchor, y, folds)
        rows.append({"seed": seed, "base": round(rb, 4), "base_unimol": round(ru, 4), "delta": round(ru - rb, 5)})
        print(f"  seed {seed}: base={rb:.4f} +unimol_emb={ru:.4f} delta={ru-rb:+.5f}")
    d = np.array([r["delta"] for r in rows]); stable = d.mean() < 0 and abs(d.mean()) > d.std()
    print("\n" + "=" * 60)
    print(f"unimol-emb on chemprop substrate: mean={d.mean():+.5f} std={d.std():.5f} wins={int((d<0).sum())}/{len(SEEDS)} stable={stable}")
    print(">>> UNI-MOL EMBEDDINGS BREAK THE SINK -> representation path REAL, build deploy" if stable
          else ">>> unimol emb absorbed too -> the chempropembed sink absorbs even a 3D-SE3 representation")
    print("=" * 60)
    json.dump({"rows": rows, "delta_mean": float(d.mean()), "delta_std": float(d.std()), "stable": bool(stable)},
              open(f"{U}/nb1006_emb_integration.json", "w"), indent=2)
    print(f"saved -> {U}/nb1006_emb_integration.json")


if __name__ == "__main__":
    main()
