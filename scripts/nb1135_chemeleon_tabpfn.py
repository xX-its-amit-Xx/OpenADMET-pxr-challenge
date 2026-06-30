"""nb1135 — CheMeleon (foundation MP encoder) + TabPFN: the two models EVERY public PXR leaderboard leader used
and we have NEITHER (cycle-309 recon). Extract frozen CheMeleon 2048-d embeddings (Zenodo weights, the actual
chemprop foundation model) for 4139 train + 513 test; build scaffold-CV OOF + 513 preds for a CheMeleon-LGBM and a
TabPFN regressor; cache to C:/pxr_work/search/. Resumable (skips cached). Honest-gate integration done in nb1136.

Usage: python nb1135_chemeleon_tabpfn.py [smoke]   # 'smoke' = 5-mol API check only
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from urllib.request import urlretrieve
from pathlib import Path
import torch

SD = "C:/pxr_work/search"; P = "data/processed"; os.makedirs(SD, exist_ok=True)
TABPFN_CKPT = "C:/pxr_work/tabpfn_v2/tabpfn-v2-regressor.ckpt"   # ungated v2 weights (v2.5/2.6/3 are HF-license-gated)
os.environ.setdefault("TABPFN_NO_BROWSER", "1")


def get_chemeleon_mp():
    ckpt = Path.home() / ".chemprop" / "chemeleon_mp.pt"
    if not ckpt.exists():
        ckpt.parent.mkdir(exist_ok=True, parents=True)
        print("downloading CheMeleon from Zenodo...", flush=True)
        urlretrieve("https://zenodo.org/records/15460715/files/chemeleon_mp.pt", ckpt)
    cm = torch.load(ckpt, weights_only=True)
    from chemprop.nn import BondMessagePassing
    mp = BondMessagePassing(**cm["hyper_parameters"]); mp.load_state_dict(cm["state_dict"]); mp.eval()
    return mp, cm["hyper_parameters"]


def chemeleon_embed(smis, mp, bs=256):
    from chemprop import data
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer, MultiHotAtomFeaturizer
    from chemprop.nn import MeanAggregation
    feat = SimpleMoleculeMolGraphFeaturizer(atom_featurizer=MultiHotAtomFeaturizer.v2())
    agg = MeanAggregation()
    if len(smis) % bs == 1:                          # chemprop dataloader DROPS a final batch of size 1 -> lose a mol
        bs -= 1
    dps = [data.MoleculeDatapoint.from_smi(s) for s in smis]
    dset = data.MoleculeDataset(dps, featurizer=feat)
    loader = data.build_dataloader(dset, batch_size=bs, shuffle=False)
    out = []
    with torch.no_grad():
        for batch in loader:
            H = mp(batch.bmg)                       # node embeddings
            G = agg(H, batch.bmg.batch)             # graph embedding (mean agg)
            out.append(G.cpu().numpy())
    emb = np.vstack(out).astype(np.float32)
    assert len(emb) == len(smis), f"embedding count {len(emb)} != {len(smis)} (drop_last bug)"
    return emb


def main():
    smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"
    from src.pxr.data import load_train, load_test
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test().reset_index(drop=True)
    smis_tr = tr["smiles"].tolist(); smis_te = te["smiles"].tolist()

    mp, hp = get_chemeleon_mp()
    print(f"CheMeleon MP loaded; output dim ~{hp.get('d_h', hp)}", flush=True)
    if smoke:
        emb = chemeleon_embed(smis_tr[:5], mp)
        print(f"SMOKE OK: 5-mol CheMeleon embedding shape {emb.shape}, finite={np.isfinite(emb).all()}")
        return

    # 1. CheMeleon embeddings (cache)
    if not os.path.exists(f"{SD}/chemeleon_tr.npy"):
        print("embedding train (4139)...", flush=True)
        etr = chemeleon_embed(smis_tr, mp); np.save(f"{SD}/chemeleon_tr.npy", etr)
        print("embedding test (513)...", flush=True)
        ete = chemeleon_embed(smis_te, mp); np.save(f"{SD}/chemeleon_te.npy", ete)
        print(f"  cached chemeleon_tr {etr.shape} / chemeleon_te {ete.shape}", flush=True)
    etr = np.load(f"{SD}/chemeleon_tr.npy"); ete = np.load(f"{SD}/chemeleon_te.npy")

    from src.pxr.eval import rae, scaffold_kfold_indices
    from src.pxr.featurize import combined, impute
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
    y = tr["pec50"].to_numpy()
    scaf = [MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else "" for s in smis_tr]

    # 2. CheMeleon-LGBM OOF + test
    if not os.path.exists(f"{SD}/chemeleon_oof.npy"):
        from lightgbm import LGBMRegressor
        oof = np.zeros(len(y))
        for trn, val in scaffold_kfold_indices(scaf, n_splits=5, seed=42):
            m = LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1)
            m.fit(etr[trn], y[trn]); oof[val] = m.predict(etr[val])
        mf = LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1).fit(etr, y)
        np.save(f"{SD}/chemeleon_oof.npy", oof); np.save(f"{SD}/chemeleon_te.npy", ete)
        np.save(f"{SD}/chemeleon_lgbm_te.npy", mf.predict(ete))
        print(f"  CheMeleon-LGBM scaffold-CV RAE {rae(y, oof):.4f}", flush=True)

    # 3. TabPFN on combined (PCA to fit TabPFN feature budget) — OOF + test
    if not os.path.exists(f"{SD}/tabpfn_oof.npy"):
        from tabpfn import TabPFNRegressor
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        CTX, BAGS, NEST = 1200, 3, 2   # TabPFN in-context: cap ctx (CPU O(n^2)); bag subsamples to use more data
        Xtr = impute(combined(smis_tr)).astype(np.float32); Xte = impute(combined(smis_te)).astype(np.float32)

        def tabpfn_bagged(Zfit, yfit, Zpred, seed0):
            preds = []
            for b in range(BAGS):
                rs = np.random.RandomState(seed0 + b)
                sub = rs.choice(len(Zfit), min(CTX, len(Zfit)), replace=False)
                m = TabPFNRegressor(device="cpu", ignore_pretraining_limits=True, model_path=TABPFN_CKPT, n_estimators=NEST)
                m.fit(Zfit[sub], yfit[sub]); preds.append(m.predict(Zpred))
            return np.mean(preds, 0)

        oof = np.zeros(len(y))
        for fi, (trn, val) in enumerate(scaffold_kfold_indices(scaf, n_splits=5, seed=42)):
            sc = StandardScaler().fit(Xtr[trn]); pca = PCA(n_components=100, random_state=0).fit(sc.transform(Xtr[trn]))
            Ztr, Zval = pca.transform(sc.transform(Xtr[trn])), pca.transform(sc.transform(Xtr[val]))
            oof[val] = tabpfn_bagged(Ztr, y[trn], Zval, 100 * fi)
            print(f"    fold {fi} done ({len(val)} preds)", flush=True)
        sc = StandardScaler().fit(Xtr); pca = PCA(n_components=100, random_state=0).fit(sc.transform(Xtr))
        Ztr_all, Zte = pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xte))
        tep = tabpfn_bagged(Ztr_all, y, Zte, 999)
        np.save(f"{SD}/tabpfn_oof.npy", oof); np.save(f"{SD}/tabpfn_te.npy", tep)
        print(f"  TabPFN(combined-PCA100, ctx{CTX}x{BAGS}bag) scaffold-CV RAE {rae(y, oof):.4f}", flush=True)
    print("DONE nb1135. Run nb1136 for honest-gate integration.")


if __name__ == "__main__":
    main()
