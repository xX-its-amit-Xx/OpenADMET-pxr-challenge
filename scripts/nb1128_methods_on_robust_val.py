"""nb1128 — re-test data-prep + architectures on the ROBUST (clean train-holdout) validation, NOT the over-tuned 253.

User skepticism: "no data prep doing anything" was measured against the 253 (whose nb3200 anchor is already tuned on
the answer key). This re-evaluates every method on 5 CLEAN analog-expansion holdouts from the train (never tuned
against), averaged -> the honest test of whether data prep / architecture / upsampling actually helps generalization.
Uses cached combined features (C:/pxr_work/search/feats.npz). If a method beats baseline HERE, it's a real lever.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train
from src.pxr.eval import rae, scaffold_kfold_indices
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostRegressor

P = "data/processed"; CACHE = "C:/pxr_work/search/feats.npz"
rng = np.random.default_rng(0)


def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""


def clip_q(p, ytr): return np.clip(p, np.quantile(ytr, 0.05), np.quantile(ytr, 0.98))


def uniform_idx(y, scaf, nbins=10, per_bin=500):
    edges = np.linspace(y.min(), y.max(), nbins + 1); b = np.clip(np.digitize(y, edges[1:-1]), 0, nbins - 1)
    out = []
    for bi in range(nbins):
        ix = np.where(b == bi)[0]
        if len(ix) == 0: continue
        sc = np.array([scaf[i] for i in ix]); _, inv, cnt = np.unique(sc, return_inverse=True, return_counts=True)
        w = 1.0 / cnt[inv]; w /= w.sum(); out.append(rng.choice(ix, per_bin, replace=True, p=w))
    return np.concatenate(out)


def main():
    d = np.load(CACHE); X = d["comb_tr"]; y = d["ytr"]; se = d["se"]
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    scaf = np.array([murcko(s) for s in tr["smiles"]])

    # 5 clean analog-expansion holdouts
    holdouts = []
    for seed in range(5):
        folds = scaffold_kfold_indices(scaf.tolist(), n_splits=round(len(y) / 250), seed=100 + seed)
        ho = min((f[1] for f in folds), key=lambda ix: abs(len(ix) - 250))
        trn = np.array([i for i in range(len(y)) if i not in set(ho.tolist())])
        holdouts.append((trn, ho))

    def lgbm(Xa, ya): return np.mean([lgb.LGBMRegressor(n_estimators=600, num_leaves=64, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8, n_jobs=4, verbose=-1, random_state=s).fit(Xa[0], Xa[1]).predict(ya) for s in range(2)], 0)

    methods = {}
    def evalm(name, fn):
        rs = []
        for trn, ho in holdouts:
            p = clip_q(fn(trn, ho), y[trn]); rs.append(rae(y[ho], p))
        methods[name] = (float(np.mean(rs)), float(np.std(rs)))

    evalm("baseline LGBM", lambda trn, ho: lgbm((X[trn], y[trn]), X[ho]))
    evalm("uniform upsample", lambda trn, ho: lgbm((X[trn][uniform_idx(y[trn], scaf[trn])], y[trn][uniform_idx(y[trn], scaf[trn])]), X[ho]))
    evalm("excl noisy 20%", lambda trn, ho: lgbm((X[trn][se[trn] <= np.quantile(se[trn], .8)], y[trn][se[trn] <= np.quantile(se[trn], .8)]), X[ho]))
    evalm("XGBoost", lambda trn, ho: np.mean([xgb.XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=s).fit(X[trn], y[trn]).predict(X[ho]) for s in range(2)], 0))
    evalm("CatBoost", lambda trn, ho: CatBoostRegressor(iterations=600, depth=6, learning_rate=0.04, verbose=0, random_seed=0).fit(X[trn], y[trn]).predict(X[ho]))
    def ens(trn, ho):
        a = lgbm((X[trn], y[trn]), X[ho])
        b = xgb.XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=0).fit(X[trn], y[trn]).predict(X[ho])
        c = CatBoostRegressor(iterations=600, depth=6, learning_rate=0.04, verbose=0, random_seed=0).fit(X[trn], y[trn]).predict(X[ho])
        return (a + b + c) / 3
    evalm("GBM ensemble", ens)

    base = methods["baseline LGBM"][0]
    print(f"=== METHODS on ROBUST validation (5 clean train holdouts, mean RAE +/- std) ===")
    print(f"{'method':18s} {'RAE':>8s} {'std':>7s} {'vs baseline':>12s}")
    for name, (m, s) in sorted(methods.items(), key=lambda kv: kv[1][0]):
        print(f"{name:18s} {m:>8.4f} {s:>7.4f} {m-base:>+12.4f}")
    print(f"\nbaseline LGBM = {base:.4f}. A method is a REAL lever if it beats baseline here (clean, untuned validation).")
    json.dump({k: v[0] for k, v in methods.items()}, open(f"{P}/nb1128_methods_robust.json", "w"), indent=2)


if __name__ == "__main__":
    main()
