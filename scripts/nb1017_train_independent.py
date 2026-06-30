"""nb1017 — INDEPENDENT validation of the Boltz-z signal: does it hold on the ~1906 cofolded TRAIN compounds,
which the feature was NEVER selected on? (The 253 was reused for all selection; train is untouched.)
Replicates the sink test: base = combined + chempropembed, anchor = chemprop_aux OOF, residual cross-fit +/- rich-z.
ALIGNMENT: unimol_train.csv (cofold order) -> load_train order (tr_chemprop_embed_300, oof_chemprop_aux) by InChIKey.
If honest train signal ~ the -0.009 nested-CV number -> robust + independent. If ~0 -> the 253 result was set-specific.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train
from src.pxr.eval import rae, scaffold_kfold_indices
from src.pxr.featurize import combined, impute
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

D = "data/processed"; U = "C:/pxr_struct/boltz"
SEEDS = list(range(1600, 1615)); QL, QH = 0.05, 0.98; K = 20


def ik(s):
    m = Chem.MolFromSmiles(str(s)); return Chem.MolToInchiKey(m) if m else None


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
    uni = pd.read_csv(f"{D}/unimol_train.csv")                  # cofold order: name, smiles, pec50
    rich = np.load(f"{U}/boltz_z_rich_train.npy")               # (4139, 512) cofold order
    lt = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    emb = np.load(f"{D}/tr_chemprop_embed_300.npy"); anc = np.load(f"{D}/oof_chemprop_aux.npy")
    assert len(lt) == emb.shape[0] == anc.shape[0], (len(lt), emb.shape, anc.shape)
    # map cofold-row -> load_train-row by InChIKey
    lt_ik = {}
    for i, s in enumerate(lt["smiles"]):
        k = ik(s)
        if k and k not in lt_ik:
            lt_ik[k] = i
    rows, smiles, y, anchor, cp = [], [], [], [], []
    for ci in range(len(uni)):
        if not np.isfinite(rich[ci]).all():
            continue                                            # not cofolded yet
        k = ik(uni["smiles"].iloc[ci])
        if k is None or k not in lt_ik:
            continue
        li = lt_ik[k]
        rows.append(ci); smiles.append(uni["smiles"].iloc[ci]); y.append(float(uni["pec50"].iloc[ci]))
        anchor.append(float(anc[li])); cp.append(emb[li])
    rows = np.array(rows); y = np.array(y); anchor = np.array(anchor); cp = np.vstack(cp)
    print(f"aligned independent-train validation set: {len(rows)} compounds (cofolded + InChIKey-mapped)")
    scaf = [murcko(s) for s in smiles]
    base = np.hstack([impute(combined(smiles)).astype(np.float32), cp.astype(np.float32)])
    # rich-z PCA fit on the SAME 4139-train pool (unsupervised), sliced to the aligned rows
    rz_fit = np.nan_to_num(rich); rz_pca_full = PCA(n_components=K, random_state=0).fit_transform(StandardScaler().fit_transform(rz_fit))
    rz = rz_pca_full[rows].astype(np.float32)
    resid = y - anchor
    ds = []
    for seed in SEEDS:
        folds = scaffold_kfold_indices(scaf, n_splits=5, seed=seed)
        rb = clipped(base, resid, anchor, y, folds)
        rz_ = clipped(np.hstack([base, rz]), resid, anchor, y, folds)
        ds.append(rz_ - rb)
    ds = np.array(ds); stable = ds.mean() < 0 and abs(ds.mean()) > ds.std()
    print(f"anchor RAE={rae(y, anchor):.4f} (chemprop_aux OOF on independent train)")
    print(f"boltz rich-z on INDEPENDENT TRAIN: mean={ds.mean():+.5f} std={ds.std():.5f} wins={int((ds<0).sum())}/{len(SEEDS)} stable={stable}")
    print("=" * 64)
    print(">>> INDEPENDENT validation CONFIRMS the signal (not 253-specific)" if stable
          else ">>> faint/absent on independent train -> 253 result may be set-specific; be skeptical")
    print(f"    compare to nested-CV honest -0.0094 on the 253")
    print("=" * 64)
    json.dump({"n_aligned": int(len(rows)), "anchor_rae": float(rae(y, anchor)), "mean": float(ds.mean()),
               "std": float(ds.std()), "wins": int((ds < 0).sum()), "stable": bool(stable)},
              open(f"{U}/nb1017_train_independent.json", "w"), indent=2)


if __name__ == "__main__":
    main()
