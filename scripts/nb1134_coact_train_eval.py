"""nb1134 — HONEST-GATE eval of the TRAIN-side ternary (coactivator) cofold z.

The cycle-302 verdict ("ternary redundant with rich-z, -0.0003") came from a TEST-only correlation proxy on the
distrusted 253. This is the rigorous version: full TRAIN-side ternary z -> fit a real residual -> evaluate on the
never-tuned clean train-holdout gate (nb1127/nb1130 style). The sharp question:

    Does the ternary ACTIVATION z (PXR-peptide + ligand-peptide + coactivator-context lig-PXR)
    add over the binary BINDING z (rich-z), on a never-tuned holdout?

All features here are NON-LEAKING on train holdouts: combined (deterministic from SMILES) + Boltz structure z
(never saw pEC50). chempropembed is EXCLUDED (it leaks on train holdouts). Runs on whatever train compounds are
cofolded so far (merges the 4 task partials); reports N_done and the RAE delta. Idempotent — the cron calls it.
"""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train, load_test
from src.pxr.eval import rae, scaffold_kfold_indices
from src.pxr.featurize import combined, impute
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

P = "data/processed"; OUT = f"{P}/nb1134_coact_eval.json"


def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""


def merge_partials(prefix, n, blocks=("ligpxr", "pxrpep", "ligpep")):
    """Merge coact_train_t{0..3}_{block}.npy partials. Each task fills a disjoint index slice."""
    done = np.zeros(n, bool); out = {b: np.zeros((n, 512), np.float32) for b in blocks}
    tasks = sorted(glob.glob(f"{P}/{prefix}_t*_done.npy"))
    for dp in tasks:
        t = dp.replace("_done.npy", "")
        dmask = np.load(dp).astype(bool)
        for b in blocks:
            bp = f"{t}_{b}.npy"
            if os.path.exists(bp):
                arr = np.load(bp); out[b][dmask] = arr[dmask]
        done |= dmask
    return out, done


def lgbm():
    return LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1)


def clean_holdout_rae(X, y, scaf, add=None, seeds=(100, 101, 102)):
    """nb1127-style: scaffold holdout ~250, train on rest, per-fold y-range clip. Optional `add` extra feature block
    is PCA-reduced FIT ON THE TRAIN FOLD ONLY (no leak)."""
    raes = []
    for s in seeds:
        folds = scaffold_kfold_indices(scaf.tolist(), n_splits=max(2, round(len(y) / 250)), seed=s)
        ho = min((f[1] for f in folds), key=lambda ix: abs(len(ix) - 250))
        trn = np.array([i for i in range(len(y)) if i not in set(ho.tolist())])
        Xtr, Xho = X[trn], X[ho]
        if add is not None:
            sc = StandardScaler().fit(add[trn]); k = min(48, add.shape[1])
            pca = PCA(n_components=k, random_state=0).fit(sc.transform(add[trn]))
            Xtr = np.hstack([Xtr, pca.transform(sc.transform(add[trn]))])
            Xho = np.hstack([Xho, pca.transform(sc.transform(add[ho]))])
        m = lgbm(); m.fit(Xtr, y[trn]); pred = m.predict(Xho)
        lo, hi = np.quantile(y[trn], 0.05), np.quantile(y[trn], 0.98)
        raes.append(rae(y[ho], np.clip(pred, lo, hi)))
    return float(np.mean(raes)), float(np.std(raes))


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    y = tr["pec50"].to_numpy(); smis = tr["smiles"].tolist()
    scaf_all = np.array([murcko(s) for s in smis])
    richz = np.load(f"{P}/boltz_z_rich_train.npy")              # binary BINDING z (4139, 512), yaml-index aligned
    blocks, done = merge_partials("coact_train", len(y))
    nd = int(done.sum())
    print(f"train ternary cofold done: {nd}/{len(y)}", flush=True)
    if nd < 800:
        json.dump({"n_done": nd, "status": "accumulating"}, open(OUT, "w"))
        print("  <800 done; accumulating, skip eval"); return

    idx = np.where(done)[0]
    cb = impute(combined([smis[i] for i in idx])).astype(np.float32)
    base = np.hstack([cb, richz[idx]])                          # combined + binary binding z (non-leaking)
    coact = np.hstack([blocks["pxrpep"][idx], blocks["ligpep"][idx], blocks["ligpxr"][idx]])  # activation z
    yy, sc = y[idx], scaf_all[idx]

    b_m, b_s = clean_holdout_rae(base, yy, sc)
    c_m, c_s = clean_holdout_rae(base, yy, sc, add=coact)
    delta = c_m - b_m
    res = {"n_done": nd, "base_combined+richz": round(b_m, 4), "base_std": round(b_s, 4),
           "+coact_z": round(c_m, 4), "coact_std": round(c_s, 4), "delta": round(delta, 4),
           "verdict": "HELPS" if delta < -0.003 else ("noise" if abs(delta) <= 0.003 else "HURTS")}
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2), flush=True)
    print(f"\nVERDICT: ternary activation z {res['verdict']} over binary binding z "
          f"(delta {delta:+.4f} on never-tuned holdouts, n_done={nd})")


if __name__ == "__main__":
    main()
