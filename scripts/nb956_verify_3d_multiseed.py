"""nb956 — is the '3D adds at the novel end' effect REAL or single-seed noise?

nb954 showed combined+3D beats combined by -0.0058 MAE @ sim<0.3 and -0.0022 RAE
overall, on ONE scaffold-fold seed (42). That margin is small enough to be a
lucky split (cf. nb1086/nb2604 lucky-seed traps). This re-runs the SAME comparison
across N disjoint scaffold-fold seeds and reports the delta distribution.

DECISION:
  - if mean(delta) < 0 AND |mean| > std  (consistently helps)        -> Uni-Mol GREEN
  - if delta straddles 0                  (noise)                     -> 3D prior WEAK too
3D descriptors reused from nb954 cache (no recompute). LGBM only; ~12 min.
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

D = "data/processed"
SEEDS = [42, 101, 202, 303, 404, 505, 606, 707]
NOVEL_T = 0.3


def max_tan(fp_val, fp_train):
    V = fp_val.astype(np.float32); T = fp_train.astype(np.float32)
    inter = V @ T.T
    a = V.sum(1)[:, None]; b = T.sum(1)[None, :]
    u = a + b - inter; u[u == 0] = 1.0
    return (inter / u).max(1)


def oof_for(X, y, folds):
    oof = np.full(len(y), np.nan)
    for tr_idx, va_idx in folds:
        m = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05,
                              n_jobs=4, verbose=-1).fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = m.predict(X[va_idx])
    return oof


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    smiles = tr["smiles"].tolist()
    y = tr["pec50"].to_numpy(float)
    scaffolds = [MurckoScaffold.MurckoScaffoldSmiles(s) if Chem.MolFromSmiles(s) else None
                 for s in smiles]

    print("loading features ...", flush=True)
    Xc = impute(combined(smiles)).astype(np.float32)
    X3 = np.load(f"{D}/nb954_desc3d_ckpt.npy")
    X3 = SimpleImputer(strategy="median").fit_transform(X3)
    X3 = np.clip(X3, -1e6, 1e6).astype(np.float32)
    Xcomb3d = np.hstack([Xc, X3])
    fp = morgan_fp_batch(smiles)

    rows = []
    t0 = time.time()
    for si, seed in enumerate(SEEDS):
        folds = scaffold_kfold_indices(scaffolds, n_splits=5, seed=seed)
        # per-seed max-sim for the novel subset
        msim = np.full(len(y), np.nan)
        for tr_idx, va_idx in folds:
            msim[va_idx] = max_tan(fp[va_idx], fp[tr_idx])
        novel = msim < NOVEL_T

        oof_c = oof_for(Xc, y, folds)
        oof_3 = oof_for(Xcomb3d, y, folds)

        d_overall = rae(y, oof_3) - rae(y, oof_c)
        mae_c = np.mean(np.abs(y[novel] - oof_c[novel]))
        mae_3 = np.mean(np.abs(y[novel] - oof_3[novel]))
        d_deep = mae_3 - mae_c
        rows.append({"seed": seed, "n_novel": int(novel.sum()),
                     "rae_combined": round(float(rae(y, oof_c)), 4),
                     "rae_combined3d": round(float(rae(y, oof_3)), 4),
                     "d_overall_rae": round(float(d_overall), 5),
                     "mae_deep_combined": round(float(mae_c), 4),
                     "mae_deep_combined3d": round(float(mae_3), 4),
                     "d_deep_mae": round(float(d_deep), 5)})
        print(f"  seed {seed}: dRAE={d_overall:+.5f}  dMAE@sim<.3={d_deep:+.5f}  "
              f"(n_novel={int(novel.sum())})  [{time.time()-t0:.0f}s]", flush=True)

    d_ov = np.array([r["d_overall_rae"] for r in rows])
    d_dp = np.array([r["d_deep_mae"] for r in rows])
    print("\n" + "=" * 66)
    print(f"3D-adds verification over {len(SEEDS)} scaffold-fold seeds (negative = 3D helps)")
    print(f"  overall RAE delta : mean={d_ov.mean():+.5f}  std={d_ov.std():.5f}  "
          f"wins={int((d_ov<0).sum())}/{len(SEEDS)}")
    print(f"  deep MAE  delta   : mean={d_dp.mean():+.5f}  std={d_dp.std():.5f}  "
          f"wins={int((d_dp<0).sum())}/{len(SEEDS)}")
    consistent_ov = d_ov.mean() < 0 and abs(d_ov.mean()) > d_ov.std()
    consistent_dp = d_dp.mean() < 0 and abs(d_dp.mean()) > d_dp.std()
    print("-" * 66)
    if consistent_ov or consistent_dp:
        print(">>> VERDICT: 3D effect is STABLE (mean exceeds std on "
              f"{'overall' if consistent_ov else ''}{' & ' if consistent_ov and consistent_dp else ''}"
              f"{'deep-novel' if consistent_dp else ''}) -> Uni-Mol GREEN LIGHT")
    else:
        print(">>> VERDICT: 3D effect straddles zero (mean < std) -> seed NOISE; "
              "3D prior is WEAK; Uni-Mol burden raised")
    print("=" * 66)

    json.dump({"seeds": SEEDS, "rows": rows,
               "d_overall_mean": float(d_ov.mean()), "d_overall_std": float(d_ov.std()),
               "d_deep_mean": float(d_dp.mean()), "d_deep_std": float(d_dp.std()),
               "stable_overall": bool(consistent_ov), "stable_deep": bool(consistent_dp)},
              open(f"{D}/nb956_verify_3d_multiseed.json", "w"), indent=2)
    print(f"saved -> {D}/nb956_verify_3d_multiseed.json")


if __name__ == "__main__":
    main()
