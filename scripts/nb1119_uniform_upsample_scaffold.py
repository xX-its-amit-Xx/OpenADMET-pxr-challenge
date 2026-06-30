"""nb1119 — binned UP-SAMPLING stratified by SCAFFOLD -> uniform pEC50 training distribution (user's anti-shrinkage idea).

Distinct from nb1105 (which used inverse-density WEIGHTS): here we physically RESAMPLE the 4139 train to a uniform
pEC50 distribution (10 equal bins), sampling within each bin with prob inversely proportional to scaffold frequency
(scaffold-stratified -> don't just duplicate one scaffold). Trained ONLY on the 4139 (NO unblinded compounds), predict
the 253 novel test. The decisive diagnostic the user wants: does the PREDICTION RANGE expand toward the true range
(= less regression-to-mean), and does the weak-tail over-prediction drop? + cross-series gate. Figure of pred-vs-true.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test, load_train
from src.pxr.featurize import combined, impute
from src.pxr.eval import rae, scaffold_kfold_indices
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

P = "data/processed"; FIG = "C:/pxr_work/figures"; os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(0)


def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else "?"


def bag(Xtr, ytr, Xte, nseed=5, idx=None):
    ps = []
    for s in range(nseed):
        m = lgb.LGBMRegressor(n_estimators=600, num_leaves=64, learning_rate=0.04, subsample=0.8,
                              colsample_bytree=0.8, n_jobs=4, verbose=-1, random_state=s)
        if idx is None: m.fit(Xtr, ytr)
        else: m.fit(Xtr[idx], ytr[idx])
        ps.append(m.predict(Xte))
    return np.mean(ps, 0)


def scaffold_uniform_idx(y, scaf, nbins=10, per_bin=600):
    """resample to uniform pEC50 bins; within bin sample prob ~ 1/scaffold_count (scaffold-balanced)."""
    edges = np.linspace(y.min(), y.max(), nbins + 1); b = np.clip(np.digitize(y, edges[1:-1]), 0, nbins - 1)
    out = []
    for bi in range(nbins):
        ix = np.where(b == bi)[0]
        if len(ix) == 0: continue
        sc = np.array([scaf[i] for i in ix])
        _, inv, cnt = np.unique(sc, return_inverse=True, return_counts=True)
        w = 1.0 / cnt[inv]; w = w / w.sum()
        out.append(rng.choice(ix, per_bin, replace=True, p=w))
    return np.concatenate(out)


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test().reset_index(drop=True)
    unb = np.load(f"{P}/_audit_unblind_idx.npy"); y = np.load(f"{P}/_audit_unblind_y.npy")
    ytr = tr["pec50"].to_numpy(); scaf = [murcko(s) for s in tr["smiles"]]
    Xtr = np.hstack([impute(combined(tr["smiles"].tolist())), np.load(f"{P}/tr_chemprop_embed_300.npy")])
    Xte = np.hstack([impute(combined(te["smiles"].to_numpy()[unb].tolist())),
                     np.load(f"{P}/te_chemprop_embed_300.npy")[unb]])

    low = y <= np.quantile(y, 0.25); high = y >= np.quantile(y, 0.75)
    print(f"true range [{y.min():.2f},{y.max():.2f}] | train pec50 median {np.median(ytr):.2f}\n")

    # baseline (natural distribution) vs scaffold-uniform upsample
    p_base = bag(Xtr, ytr, Xte)
    idx_u = scaffold_uniform_idx(ytr, scaf)
    p_up = bag(Xtr, ytr, Xte, idx=idx_u)
    print(f"{'model':22s} {'RAE':>7s} {'predmin':>8s} {'predmax':>8s} {'lowbias':>8s} {'highbias':>9s}")
    for name, p in [("baseline (natural)", p_base), ("scaffold-uniform upsample", p_up)]:
        print(f"{name:22s} {rae(y,p):>7.4f} {p.min():>8.2f} {p.max():>8.2f} "
              f"{np.mean(p[low]-y[low]):>+8.2f} {np.mean(p[high]-y[high]):>+9.2f}")
    print(f"\nupsampled train: {len(idx_u)} rows, pEC50 std {ytr[idx_u].std():.2f} vs natural {ytr.std():.2f} "
          f"(higher = more uniform/spread)")

    # figure: pred vs true, baseline vs upsample
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    lims = [y.min() - 0.3, y.max() + 0.3]
    for a, (name, p) in zip(ax, [("baseline (natural dist)", p_base), ("scaffold-uniform upsample", p_up)]):
        a.scatter(y, p, s=20, alpha=0.6); a.plot(lims, lims, "k--", lw=1)
        a.set_xlim(lims); a.set_ylim(lims); a.set_xlabel("true pEC50"); a.set_ylabel("predicted")
        a.set_title(f"{name}\nRAE {rae(y,p):.4f}, pred range [{p.min():.1f},{p.max():.1f}]")
    plt.tight_layout(); plt.savefig(f"{FIG}/nb1119_uniform_upsample.png", dpi=115); plt.close()

    # cross-series gate (does the upsample's prediction beat baseline on held-out series? it's a standalone model,
    # so compare standalone RAE per held-out series)
    from sklearn.cluster import KMeans; from sklearn.decomposition import PCA; from sklearn.preprocessing import StandardScaler
    series = KMeans(6, n_init=5, random_state=0).fit_predict(PCA(20, random_state=0).fit_transform(StandardScaler().fit_transform(Xte)))
    db, du = [], []
    for kk in range(6):
        m = series == kk
        if m.sum() < 5: continue
        db.append(rae(y[m], p_base[m])); du.append(rae(y[m], p_up[m]))
    print(f"per-series RAE: baseline mean {np.mean(db):.4f} | upsample mean {np.mean(du):.4f} "
          f"(upsample better on {np.mean(np.array(du)<np.array(db))*100:.0f}% of series)")
    json.dump({"base_rae": float(rae(y, p_base)), "upsample_rae": float(rae(y, p_up)),
               "base_range": [float(p_base.min()), float(p_base.max())],
               "upsample_range": [float(p_up.min()), float(p_up.max())], "true_range": [float(y.min()), float(y.max())]},
              open(f"{P}/nb1119_uniform_upsample.json", "w"), indent=2)
    print(f"\nwrote {FIG}/nb1119_uniform_upsample.png")


if __name__ == "__main__":
    main()
