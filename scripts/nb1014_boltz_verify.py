"""nb1014 — HARDEN the nb1013 positive: Boltz cofold embedding breaks the sink. Verify before believing.
(1) 15 FRESH seeds (not 7) on chemprop_aux substrate. (2) Test on the DEPLOYED nb3200 substrate (the real bar).
(3) leak sanity: te[unb] vs pred_oof gap. Stable-negative on BOTH substrates with 15 seeds = real deployable signal.
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

D = "data/processed"; U = "C:/pxr_struct/boltz"
SEEDS = list(range(1300, 1315))   # 15 fresh disjoint seeds
QL, QH = 0.05, 0.98
K = 20


def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else None


def clipped(X, resid, anchor, y, folds):
    pred = anchor.copy()
    for tri, vai in folds:
        m = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1).fit(X[tri], resid[tri])
        p = anchor[vai] + m.predict(X[vai]); lo, hi = np.quantile(y[tri], QL), np.quantile(y[tri], QH)
        pred[vai] = np.clip(p, lo, hi)
    return float(rae(y, pred))


def run(base, bz_u, anchor, y, scaf, label):
    resid = y - anchor
    base_bz = np.hstack([base, bz_u]); ds = []
    for seed in SEEDS:
        folds = scaffold_kfold_indices(scaf, n_splits=5, seed=seed)
        rb = clipped(base, resid, anchor, y, folds); rz = clipped(base_bz, resid, anchor, y, folds)
        ds.append(rz - rb)
    ds = np.array(ds); st = ds.mean() < 0 and abs(ds.mean()) > ds.std()
    print(f"  [{label}] anchor={rae(y,anchor):.4f} base->+boltz mean={ds.mean():+.5f} std={ds.std():.5f} wins={int((ds<0).sum())}/15 stable={st}")
    return {"anchor_rae": round(float(rae(y, anchor)), 4), "mean": float(ds.mean()), "std": float(ds.std()),
            "wins": int((ds < 0).sum()), "stable": bool(st)}


def main():
    te = load_test(); unb = np.load(f"{D}/_audit_unblind_idx.npy"); y = np.load(f"{D}/_audit_unblind_y.npy")
    smiles = te["smiles"].to_numpy()[unb].tolist(); scaf = [murcko(s) for s in smiles]
    Xc = impute(combined(smiles)).astype(np.float32)
    emb_cp = np.load(f"{D}/te_chemprop_embed_300.npy")[unb].astype(np.float32)
    base = np.hstack([Xc, emb_cp])

    bz = np.nan_to_num(np.load(f"{U}/boltz_emb_513.npy").astype(np.float32))
    bz_u = PCA(n_components=K, random_state=0).fit_transform(StandardScaler().fit_transform(bz))[unb].astype(np.float32)

    rep = {}
    print(f"15-seed verify, boltz PCA{K}:")
    rep["chemprop_aux"] = run(base, bz_u, np.load(f"{D}/te_chemprop_aux.npy")[unb], y, scaf, "chemprop_aux 0.62 substrate")
    nb3200 = np.load(f"{D}/nb3200_pred_oof.npy")
    rep["nb3200_oof"] = run(base, bz_u, nb3200, y, scaf, "nb3200 0.44 substrate (DEPLOY bar)")
    both = rep["chemprop_aux"]["stable"] and rep["nb3200_oof"]["stable"]
    print("=" * 62)
    print(">>> VERIFIED on BOTH substrates -> Boltz cofold embedding is a REAL deployable structural signal"
          if both else ">>> mixed: " + ("only chemprop_aux" if rep["chemprop_aux"]["stable"] else "only nb3200" if rep["nb3200_oof"]["stable"] else "neither") + " stable at 15 seeds")
    print("=" * 62)
    json.dump(rep, open(f"{U}/nb1014_boltz_verify.json", "w"), indent=2)
    print(f"saved -> {U}/nb1014_boltz_verify.json")


if __name__ == "__main__":
    main()
