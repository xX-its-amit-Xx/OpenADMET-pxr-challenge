"""nb1009 — does the DeepPurpose Mode B cross-target NR-panel profile ADD to combined+chempropembed on the 253?
The 12 NR-activity columns vary the TARGET (orthogonal to single-target chemprop). 12-dim = no PCA needed.
cycle-291 protocol: chemprop_aux residual + range-clip, multi-seed. Stable-negative -> cross-target axis breaks the sink.
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
from sklearn.preprocessing import StandardScaler

D = "data/processed"; U = "C:/pxr_struct/deeppurpose"
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

    dp513 = np.nan_to_num(np.load(f"{U}/dp_modeB_test.npy").astype(np.float32))   # (513, 12)
    cols = json.load(open(f"{U}/dp_modeB_cols.json"))
    dp_u = StandardScaler().fit_transform(dp513)[unb].astype(np.float32)
    # also a PXR-relative variant: each NR minus PXR column (relative engagement)
    pxr_i = cols.index("PXR") if "PXR" in cols else 0
    dp_rel = (dp513 - dp513[:, [pxr_i]])
    dp_rel_u = StandardScaler().fit_transform(np.delete(dp_rel, pxr_i, axis=1))[unb].astype(np.float32)
    base_dp = np.hstack([base, dp_u]); base_rel = np.hstack([base, dp_rel_u])
    print(f"anchor {rae(y,anchor):.4f}; cols={cols}; base={base.shape} +dp12={base_dp.shape} +dprel={base_rel.shape}\n")
    resid = y - anchor
    rows = []
    for seed in SEEDS:
        folds = scaffold_kfold_indices(scaf, n_splits=5, seed=seed)
        rb = clipped(base, resid, anchor, y, folds)
        rd = clipped(base_dp, resid, anchor, y, folds)
        rr = clipped(base_rel, resid, anchor, y, folds)
        rows.append({"seed": seed, "base": round(rb, 4), "base_dp": round(rd, 4), "base_rel": round(rr, 4),
                     "d_dp": round(rd - rb, 5), "d_rel": round(rr - rb, 5)})
        print(f"  seed {seed}: base={rb:.4f} +dp={rd:.4f}({rd-rb:+.4f}) +rel={rr:.4f}({rr-rb:+.4f})")
    dd = np.array([r["d_dp"] for r in rows]); dr = np.array([r["d_rel"] for r in rows])
    d_stable = dd.mean() < 0 and abs(dd.mean()) > dd.std(); r_stable = dr.mean() < 0 and abs(dr.mean()) > dr.std()
    print("\n" + "=" * 62)
    print(f"dp-12 on chemprop substrate:      mean={dd.mean():+.5f} std={dd.std():.5f} wins={int((dd<0).sum())}/{len(SEEDS)} stable={d_stable}")
    print(f"dp-relative-to-PXR on substrate:  mean={dr.mean():+.5f} std={dr.std():.5f} wins={int((dr<0).sum())}/{len(SEEDS)} stable={r_stable}")
    print(">>> DEEPPURPOSE CROSS-TARGET BREAKS THE SINK -> NR-panel axis REAL" if (d_stable or r_stable)
          else ">>> dp absorbed too -> cross-target NR-panel profile redundant on the n=253 substrate")
    print("=" * 62)
    json.dump({"rows": rows, "cols": cols, "d_dp_mean": float(dd.mean()), "d_dp_std": float(dd.std()), "dp_stable": bool(d_stable),
               "d_rel_mean": float(dr.mean()), "d_rel_std": float(dr.std()), "rel_stable": bool(r_stable)},
              open(f"{U}/nb1009_dp_integration.json", "w"), indent=2)
    print(f"saved -> {U}/nb1009_dp_integration.json")


if __name__ == "__main__":
    main()
