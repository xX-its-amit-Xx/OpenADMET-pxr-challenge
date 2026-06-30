"""nb981 — is the pharmacophore-anchor-fit gain REAL or single-seed noise?

nb979: combined+anchorfit beat combined by -0.0155 deep-MAE @ sim<0.3 (+ -0.0037 overall) on ONE seed.
The 3D-descriptor probe (nb954) looked positive single-seed too and was seed-noise (nb956). Same gate:
8 disjoint scaffold-fold seeds. STABLE (mean<0 AND |mean|>std on overall OR deep) -> real lever.
Reuses anchorfit features from nb979_ckpt.npz. All on C:.
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train
from src.pxr.eval import rae, scaffold_kfold_indices
from src.pxr.featurize import combined, impute
from src.pxr.chem import morgan_fp_batch
from sklearn.impute import SimpleImputer
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb

D = "data/processed"; OUT = "C:/pxr_struct"
SEEDS = [42, 101, 202, 303, 404, 505, 606, 707]
NOVEL_T = 0.3


def max_tan(fa, fb):
    A = fa.astype(np.float32); B = fb.astype(np.float32)
    inter = A @ B.T; u = A.sum(1)[:, None] + B.sum(1)[None, :] - inter; u[u == 0] = 1.0
    return (inter / u).max(1)


def oof(X, y, folds):
    o = np.full(len(y), np.nan)
    for tri, vai in folds:
        m = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1).fit(X[tri], y[tri])
        o[vai] = m.predict(X[vai])
    return o


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    y = tr["pec50"].to_numpy(float); smiles = tr["smiles"].tolist()
    scaf = [MurckoScaffold.MurckoScaffoldSmiles(s) if Chem.MolFromSmiles(s) else None for s in smiles]
    Xc = impute(combined(smiles)).astype(np.float32)
    F = np.load("C:/pxr_struct/nb979_ckpt.npz")["F_tr"]
    A = SimpleImputer(strategy="median").fit_transform(F).astype(np.float32)
    Xca = np.hstack([Xc, A]); fp = morgan_fp_batch(smiles)
    print(f"combined={Xc.shape} +anchorfit={Xca.shape}", flush=True)

    rows = []; t0 = time.time()
    for seed in SEEDS:
        folds = scaffold_kfold_indices(scaf, n_splits=5, seed=seed)
        msim = np.full(len(y), np.nan)
        for tri, vai in folds: msim[vai] = max_tan(fp[vai], fp[tri])
        nov = msim < NOVEL_T
        oc = oof(Xc, y, folds); oa = oof(Xca, y, folds)
        d_ov = rae(y, oa) - rae(y, oc)
        d_dp = np.mean(np.abs(y[nov]-oa[nov])) - np.mean(np.abs(y[nov]-oc[nov]))
        rows.append({"seed": seed, "d_overall": round(float(d_ov), 5), "d_deep": round(float(d_dp), 5)})
        print(f"  seed {seed}: dRAE={d_ov:+.5f} dMAE@sim<.3={d_dp:+.5f} [{time.time()-t0:.0f}s]", flush=True)

    d_ov = np.array([r["d_overall"] for r in rows]); d_dp = np.array([r["d_deep"] for r in rows])
    so = d_ov.mean() < 0 and abs(d_ov.mean()) > d_ov.std()
    sd = d_dp.mean() < 0 and abs(d_dp.mean()) > d_dp.std()
    print("\n" + "=" * 64)
    print(f"anchorfit verify over {len(SEEDS)} seeds (negative = helps):")
    print(f"  overall: mean={d_ov.mean():+.5f} std={d_ov.std():.5f} wins={int((d_ov<0).sum())}/{len(SEEDS)} stable={so}")
    print(f"  deep:    mean={d_dp.mean():+.5f} std={d_dp.std():.5f} wins={int((d_dp<0).sum())}/{len(SEEDS)} stable={sd}")
    print(">>> ANCHOR-FIT REAL -> test on chemprop substrate (ladder) + deploy" if (so or sd)
          else ">>> seed noise -> anchorfit absorbed too; CPU-local exhausted")
    print("=" * 64)
    json.dump({"rows": rows, "d_overall_mean": float(d_ov.mean()), "d_overall_std": float(d_ov.std()),
               "d_deep_mean": float(d_dp.mean()), "d_deep_std": float(d_dp.std()),
               "stable_overall": bool(so), "stable_deep": bool(sd)},
              open(f"{OUT}/nb981_verify_anchorfit.json", "w"), indent=2)
    print(f"saved -> {OUT}/nb981_verify_anchorfit.json")


if __name__ == "__main__":
    main()
