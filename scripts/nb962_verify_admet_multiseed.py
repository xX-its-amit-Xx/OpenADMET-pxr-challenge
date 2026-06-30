"""nb962 — is the ADMET-AI feature gain REAL or single-seed noise?

nb961: combined+ADMET beat combined by -0.019 overall RAE and -0.015 deep-MAE @ sim<0.3 on
ONE scaffold-fold seed (42). The 3D probe (nb954) looked positive single-seed too and was
seed-noise (nb956). Same gate here: re-run across 8 disjoint scaffold-fold seeds.

DECISION (no moving goalposts):
  - mean(delta) < 0 AND |mean| > std on overall OR deep  -> ADMET is REAL -> promote / build deploy features
  - straddles 0                                           -> seed noise -> C9 negative
ADMET features reused from C:/admet_out/admet_train.csv. LGBM only; ~12 min.
"""
import os, sys, json, time
import numpy as np, pandas as pd
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

D = "data/processed"
SEEDS = [42, 101, 202, 303, 404, 505, 606, 707]
NOVEL_T = 0.3


def max_tan(fa, fb):
    A = fa.astype(np.float32); B = fb.astype(np.float32)
    inter = A @ B.T; u = A.sum(1)[:, None] + B.sum(1)[None, :] - inter; u[u == 0] = 1.0
    return (inter / u).max(1)


def oof(X, y, folds):
    o = np.full(len(y), np.nan)
    for tri, vai in folds:
        m = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05,
                              n_jobs=4, verbose=-1).fit(X[tri], y[tri])
        o[vai] = m.predict(X[vai])
    return o


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    y = tr["pec50"].to_numpy(float); smiles = tr["smiles"].tolist()
    scaf = [MurckoScaffold.MurckoScaffoldSmiles(s) if Chem.MolFromSmiles(s) else None for s in smiles]

    Xc = impute(combined(smiles)).astype(np.float32)
    adf = pd.read_csv("C:/admet_out/admet_train.csv")
    assert len(adf) == len(tr)
    props = [c for c in adf.columns if c != "smiles" and pd.api.types.is_numeric_dtype(adf[c])]
    A = SimpleImputer(strategy="median").fit_transform(adf[props].to_numpy(float)).astype(np.float32)
    A = np.clip(np.nan_to_num(A, posinf=1e6, neginf=-1e6), -1e6, 1e6)
    Xca = np.hstack([Xc, A])
    fp = morgan_fp_batch(smiles)
    print(f"combined={Xc.shape} ADMET={A.shape} combined+ADMET={Xca.shape}", flush=True)

    rows = []; t0 = time.time()
    for seed in SEEDS:
        folds = scaffold_kfold_indices(scaf, n_splits=5, seed=seed)
        msim = np.full(len(y), np.nan)
        for tri, vai in folds:
            msim[vai] = max_tan(fp[vai], fp[tri])
        nov = msim < NOVEL_T
        oc = oof(Xc, y, folds); oa = oof(Xca, y, folds)
        d_ov = rae(y, oa) - rae(y, oc)
        d_dp = np.mean(np.abs(y[nov]-oa[nov])) - np.mean(np.abs(y[nov]-oc[nov]))
        rows.append({"seed": seed, "rae_c": round(float(rae(y, oc)), 4), "rae_ca": round(float(rae(y, oa)), 4),
                     "d_overall": round(float(d_ov), 5), "d_deep": round(float(d_dp), 5)})
        print(f"  seed {seed}: dRAE={d_ov:+.5f}  dMAE@sim<.3={d_dp:+.5f}  [{time.time()-t0:.0f}s]", flush=True)

    d_ov = np.array([r["d_overall"] for r in rows]); d_dp = np.array([r["d_deep"] for r in rows])
    stab_ov = d_ov.mean() < 0 and abs(d_ov.mean()) > d_ov.std()
    stab_dp = d_dp.mean() < 0 and abs(d_dp.mean()) > d_dp.std()
    print("\n" + "=" * 66)
    print(f"ADMET feature verification over {len(SEEDS)} seeds (negative = ADMET helps)")
    print(f"  overall RAE delta: mean={d_ov.mean():+.5f} std={d_ov.std():.5f} wins={int((d_ov<0).sum())}/{len(SEEDS)} stable={stab_ov}")
    print(f"  deep MAE  delta:   mean={d_dp.mean():+.5f} std={d_dp.std():.5f} wins={int((d_dp<0).sum())}/{len(SEEDS)} stable={stab_dp}")
    print("-" * 66)
    print(">>> ADMET REAL -> promote/build deploy features" if (stab_ov or stab_dp)
          else ">>> seed noise -> C9 negative")
    print("=" * 66)
    json.dump({"seeds": SEEDS, "rows": rows, "d_overall_mean": float(d_ov.mean()),
               "d_overall_std": float(d_ov.std()), "d_deep_mean": float(d_dp.mean()),
               "d_deep_std": float(d_dp.std()), "stable_overall": bool(stab_ov),
               "stable_deep": bool(stab_dp)}, open(f"{D}/nb962_verify_admet.json", "w"), indent=2)
    print(f"saved -> {D}/nb962_verify_admet.json")


if __name__ == "__main__":
    main()
