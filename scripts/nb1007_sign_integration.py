"""nb1007 — does the CC Signaturizer measured-bioactivity signature ADD to combined+chempropembed on the 253?
cycle-291 protocol (nb1006 twin): chemprop_aux residual + range-clip, multi-seed. 896-d signature (7x128) is
too wide for n=253 -> PCA to <=15 dims (unsupervised, fit on all 513 = no label leak). Also test the 7-d
applicability (alpha) block separately (abstention axis). Stable-negative -> measured-bioactivity axis breaks the sink.
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

D = "data/processed"; U = "C:/pxr_struct/signaturizer"
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

    sig513 = np.nan_to_num(np.load(f"{U}/sign_signatures_513.npy").astype(np.float32))
    alpha513 = np.nan_to_num(np.load(f"{U}/sign_applicability_513.npy").astype(np.float32))
    sig_pca = PCA(n_components=N_PCA, random_state=0).fit_transform(StandardScaler().fit_transform(sig513))
    sig_u = sig_pca[unb].astype(np.float32); alpha_u = alpha513[unb].astype(np.float32)
    base_sig = np.hstack([base, sig_u]); base_alpha = np.hstack([base, alpha_u])
    print(f"anchor {rae(y,anchor):.4f}; base={base.shape} +sign_pca{N_PCA}={base_sig.shape} +alpha={base_alpha.shape}\n")
    resid = y - anchor
    rows = []
    for seed in SEEDS:
        folds = scaffold_kfold_indices(scaf, n_splits=5, seed=seed)
        rb = clipped(base, resid, anchor, y, folds)
        rs = clipped(base_sig, resid, anchor, y, folds)
        ra = clipped(base_alpha, resid, anchor, y, folds)
        rows.append({"seed": seed, "base": round(rb, 4), "base_sign": round(rs, 4), "base_alpha": round(ra, 4),
                     "d_sign": round(rs - rb, 5), "d_alpha": round(ra - rb, 5)})
        print(f"  seed {seed}: base={rb:.4f} +sign={rs:.4f}({rs-rb:+.4f}) +alpha={ra:.4f}({ra-rb:+.4f})")
    ds = np.array([r["d_sign"] for r in rows]); da = np.array([r["d_alpha"] for r in rows])
    s_stable = ds.mean() < 0 and abs(ds.mean()) > ds.std(); a_stable = da.mean() < 0 and abs(da.mean()) > da.std()
    print("\n" + "=" * 64)
    print(f"sign-PCA{N_PCA} on chemprop substrate: mean={ds.mean():+.5f} std={ds.std():.5f} wins={int((ds<0).sum())}/{len(SEEDS)} stable={s_stable}")
    print(f"applicability(7) on substrate:       mean={da.mean():+.5f} std={da.std():.5f} wins={int((da<0).sum())}/{len(SEEDS)} stable={a_stable}")
    print(">>> CC SIGNATURIZER BREAKS THE SINK -> measured-bioactivity axis REAL" if (s_stable or a_stable)
          else ">>> sign absorbed too -> measured-bioactivity signature redundant with chemprop on the nb3200 substrate")
    print("=" * 64)
    json.dump({"rows": rows, "d_sign_mean": float(ds.mean()), "d_sign_std": float(ds.std()), "sign_stable": bool(s_stable),
               "d_alpha_mean": float(da.mean()), "d_alpha_std": float(da.std()), "alpha_stable": bool(a_stable)},
              open(f"{U}/nb1007_sign_integration.json", "w"), indent=2)
    print(f"saved -> {U}/nb1007_sign_integration.json")


if __name__ == "__main__":
    main()
