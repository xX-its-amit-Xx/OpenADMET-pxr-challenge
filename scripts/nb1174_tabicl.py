"""nb1174 — TabICL v2 (Soda-INRIA in-context tabular FM) as a SECOND in-context-tabular-FM ensemble member,
the last untested model in the dossier-★ leaderboard recipe (CheMeleon✓ TabPFN✓ TabICL?).

Mirrors nb1135's TabPFN recipe: scaffold 5-fold OOF + 513 test preds of TabICLRegressor on combined-PCA100
(standardized), CPU-only, ctx-capped bagging to bound O(n) cost. Cache to C:/pxr_work/search/.
Resumable (skips cached). Honest add-member gate is nb1174_gate.py.

Usage: python nb1174_tabicl.py [smoke]
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", "C:/pxr_work/hf_home")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

SD = "C:/pxr_work/search"; P = "data/processed"; os.makedirs(SD, exist_ok=True)
CTX, BAGS = 2000, 3   # TabICL scales further than TabPFN; cap ctx + bag to use all data at bounded cost


def make_reg():
    from tabicl import TabICLRegressor
    return TabICLRegressor(device="cpu", n_estimators=4, allow_auto_download=True)


def bagged(Zfit, yfit, Zpred, seed0):
    preds = []
    for b in range(BAGS):
        rs = np.random.RandomState(seed0 + b)
        sub = rs.choice(len(Zfit), min(CTX, len(Zfit)), replace=False)
        m = make_reg(); m.fit(Zfit[sub], yfit[sub]); preds.append(m.predict(Zpred))
    return np.mean(preds, 0)


def main():
    smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"
    from src.pxr.data import load_train, load_test
    from src.pxr.eval import rae, scaffold_kfold_indices
    from src.pxr.featurize import combined, impute
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test().reset_index(drop=True)
    smis_tr = tr["smiles"].tolist(); smis_te = te["smiles"].tolist()
    y = tr["pec50"].to_numpy()

    if smoke:
        X = impute(combined(smis_tr[:80])).astype(np.float32)
        sc = StandardScaler().fit(X); pca = PCA(n_components=30, random_state=0).fit(sc.transform(X))
        Z = pca.transform(sc.transform(X))
        t = time.time(); p = bagged(Z[:60], y[:60], Z[60:], 0)
        print(f"SMOKE OK pred {p.shape} finite={np.isfinite(p).all()} t={time.time()-t:.1f}s"); return

    if os.path.exists(f"{SD}/tabicl_oof.npy") and os.path.exists(f"{SD}/tabicl_te.npy"):
        oof = np.load(f"{SD}/tabicl_oof.npy")
        print(f"already cached; scaffold-CV RAE {rae(y, oof):.4f}"); return

    scaf = [MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else "" for s in smis_tr]
    Xtr = impute(combined(smis_tr)).astype(np.float32); Xte = impute(combined(smis_te)).astype(np.float32)

    oof = np.zeros(len(y))
    for fi, (trn, val) in enumerate(scaffold_kfold_indices(scaf, n_splits=5, seed=42)):
        t = time.time()
        sc = StandardScaler().fit(Xtr[trn]); pca = PCA(n_components=100, random_state=0).fit(sc.transform(Xtr[trn]))
        Ztr, Zval = pca.transform(sc.transform(Xtr[trn])), pca.transform(sc.transform(Xtr[val]))
        oof[val] = bagged(Ztr, y[trn], Zval, 100 * fi)
        np.save(f"{SD}/tabicl_oof.npy", oof)   # checkpoint after each fold (resumable-ish)
        print(f"  fold {fi} done ({len(val)} preds) t={time.time()-t:.0f}s", flush=True)
    sc = StandardScaler().fit(Xtr); pca = PCA(n_components=100, random_state=0).fit(sc.transform(Xtr))
    Ztr_all, Zte = pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xte))
    tep = bagged(Ztr_all, y, Zte, 999)
    np.save(f"{SD}/tabicl_oof.npy", oof); np.save(f"{SD}/tabicl_te.npy", tep)
    print(f"  TabICL(combined-PCA100, ctx{CTX}x{BAGS}bag) scaffold-CV RAE {rae(y, oof):.4f}", flush=True)
    print("DONE nb1174. Run nb1174_gate.py for honest add-member gate.")


if __name__ == "__main__":
    main()
