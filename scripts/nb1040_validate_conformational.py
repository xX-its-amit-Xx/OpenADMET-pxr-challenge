"""nb1040 — validate the Boltz conformational-ensemble features (Modal) on the honest 253.

Features per test ligand (from 5 diffusion samples):
  geom[18] = [d_helix12<->ligand, d_helix12<->core, helix12_Rg, pocket_contacts,   (across-sample MEAN)
              std of each of those 4,                                              (FLUCTUATION)
              helix12_RMSF, core_RMSF,                                             (backbone fluctuation)
              conf_score, ligand_iptm, complex_plddt, complex_pde,                (MEAN confidence)
              std of each of those 4]                                             (confidence fluctuation)
  richz[512] = fresh multi-sample rich-z (trunk z lig x prot pooling)
Tests: geom as feature on nb3200; fresh richz vs old rich-z; does geom ADD on top of rich-z? (30 seeds, clipped).
Plus per-feature corr with nb3200 error / true pEC50 (which conformational signal, if any, tracks activation).
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test
from src.pxr.eval import rae, scaffold_kfold_indices
from src.pxr.featurize import combined, impute
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb

D = "data/processed"; M = "C:/pxr_struct/boltz/modal"; U = "C:/pxr_struct/boltz"; QL, QH = 0.05, 0.98
GNAMES = ["d_h12_lig", "d_h12_core", "h12_Rg", "contacts", "std_d_h12_lig", "std_d_h12_core", "std_h12_Rg",
          "std_contacts", "h12_rmsf", "core_rmsf", "conf", "lig_iptm", "plddt", "pde",
          "std_conf", "std_lig_iptm", "std_plddt", "std_pde"]


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
    geom = np.load(f"{M}/test_geom.npy"); richz = np.load(f"{M}/test_richz.npy")
    cov = ~np.isnan(geom).any(1)
    print(f"conformational features: {cov.sum()}/513 cofolded")
    unb = np.load(f"{D}/_audit_unblind_idx.npy"); y = np.load(f"{D}/_audit_unblind_y.npy")
    anchor = np.load(f"{D}/nb3200_pred_oof.npy"); resid = y - anchor
    gu = geom[unb]; rzu = richz[unb]
    # impute any missing rows with column medians
    for A in (gu, rzu):
        col = np.nanmedian(A, 0)
        idx = np.where(np.isnan(A))
        A[idx] = np.take(col, idx[1])

    # per-feature diagnostics
    print("\nper-geom-feature corr (vs true pEC50 | vs nb3200 error):")
    for j, nm in enumerate(GNAMES):
        v = gu[:, j]
        if np.std(v) < 1e-9:
            continue
        print(f"  {nm:16s}: true {np.corrcoef(v, y)[0,1]:+.3f}   err {np.corrcoef(v, resid)[0,1]:+.3f}")

    te = load_test(); smiles = te["smiles"].to_numpy()[unb].tolist(); scaf = [murcko(s) for s in smiles]
    base = np.hstack([impute(combined(smiles)).astype(np.float32), np.load(f"{D}/te_chemprop_embed_300.npy")[unb].astype(np.float32)])
    gz = StandardScaler().fit_transform(gu).astype(np.float32)
    # rich-z PCA<=15 (consistent with deploy), fit on the finite eval rows only (all 253 eval cofolded)
    def pca15(arr):
        a = arr[unb]
        sc = StandardScaler().fit(a)
        return PCA(n_components=15, random_state=0).fit_transform(sc.transform(a)).astype(np.float32)
    rz_pca = pca15(richz)
    rz_old = pca15(np.load(f"{U}/boltz_z_rich_513.npy")) if os.path.exists(f"{U}/boltz_z_rich_513.npy") else None

    SEEDS = list(range(1400, 1430))
    def test_block(extra, label):
        ds = []
        for s in SEEDS:
            f = scaffold_kfold_indices(scaf, 5, seed=s)
            ds.append(clipped(np.hstack([base, extra]), resid, anchor, y, f) - clipped(base, resid, anchor, y, f))
        ds = np.array(ds); st = ds.mean() < 0 and abs(ds.mean()) > ds.std()
        print(f"  {label:34s}: {ds.mean():+.5f} +/- {ds.std():.5f} wins {int((ds<0).sum())}/30 stable={st}")
        return ds.mean()

    print("\nfeature blocks on nb3200 (vs base, 30 seeds):")
    test_block(gz, "geom(18) [activation+fluctuation]")
    test_block(rz_pca, "fresh rich-z PCA15")
    if rz_old is not None:
        test_block(rz_old, "old rich-z PCA15 (reference)")
    # does geom ADD on top of rich-z?
    def test_marginal(extra, baseextra, label):
        ds = []
        for s in SEEDS:
            f = scaffold_kfold_indices(scaf, 5, seed=s)
            ds.append(clipped(np.hstack([base, baseextra, extra]), resid, anchor, y, f)
                      - clipped(np.hstack([base, baseextra]), resid, anchor, y, f))
        ds = np.array(ds); st = ds.mean() < 0 and abs(ds.mean()) > ds.std()
        print(f"  {label:34s}: {ds.mean():+.5f} +/- {ds.std():.5f} wins {int((ds<0).sum())}/30 stable={st}")
    print("\nmarginal value:")
    test_marginal(gz, rz_pca, "geom MARGINAL over rich-z")
    json.dump({"cofolded": int(cov.sum())}, open(f"{D}/nb1040_conformational.json", "w"), indent=2)


if __name__ == "__main__":
    main()
