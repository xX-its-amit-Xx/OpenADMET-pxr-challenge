"""nb1016 — is the Boltz-z win inflated by SELECTING the best feature/K on the fixed 253?
NESTED CV: outer 5-fold; inside each outer-train, pick the best config by inner cross-fit; apply the
SELECTED config to the outer-test fold (which never touched selection). Compare:
  OPTIMISTIC = best config chosen + evaluated on the SAME full 253 (what I've been reporting).
  HONEST     = nested (selection inside CV, eval on untouched fold).
Gap = the selection/multiple-testing inflation. Anchor = nb3200 deploy substrate.
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
SEEDS = [1500, 1501, 1502, 1503, 1504]
QL, QH = 0.05, 0.98


def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else None


def lgbm():
    return lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1)


def cf_pred(base, feat, resid, anchor, y, rows, folds_local):
    """cross-fit predict on `rows` (subset) using its own internal folds; return pred on rows."""
    X = base if feat is None else np.hstack([base, feat])
    pred = anchor[rows].copy()
    for tri, vai in folds_local:
        m = lgbm().fit(X[rows][tri], resid[rows][tri])
        p = anchor[rows][vai] + m.predict(X[rows][vai]); lo, hi = np.quantile(y[rows][tri], QL), np.quantile(y[rows][tri], QH)
        pred[vai] = np.clip(p, lo, hi)
    return pred


def main():
    te = load_test(); unb = np.load(f"{D}/_audit_unblind_idx.npy"); y = np.load(f"{D}/_audit_unblind_y.npy")
    smiles = te["smiles"].to_numpy()[unb].tolist(); scaf = np.array([murcko(s) for s in smiles], dtype=object)
    anchor = np.load(f"{D}/nb3200_pred_oof.npy"); resid = y - anchor
    base = np.hstack([impute(combined(smiles)).astype(np.float32),
                      np.load(f"{D}/te_chemprop_embed_300.npy")[unb].astype(np.float32)])
    emb = np.nan_to_num(np.load(f"{U}/boltz_emb_513.npy").astype(np.float32))
    rich = np.nan_to_num(np.load(f"{U}/boltz_z_rich_513.npy").astype(np.float32))
    raw = {"s": emb[:, :768], "z": emb[:, 768:], "full": emb, "rich": rich}
    # configs = feature x K  (PCA fit on 513, unsupervised -> no label selection)
    configs = {}
    for fn, arr in raw.items():
        sc = StandardScaler().fit_transform(arr)
        for K in (12, 15, 20):
            configs[f"{fn}_K{K}"] = PCA(n_components=K, random_state=0).fit_transform(sc)[unb].astype(np.float32)

    def imp_on(rows, feat, seed):
        fl = scaffold_kfold_indices(list(scaf[rows]), n_splits=5, seed=seed)
        pb = cf_pred(base, None, resid, anchor, y, rows, fl)
        pf = cf_pred(base, feat, resid, anchor, y, rows, fl)
        return rae(y[rows], pf) - rae(y[rows], pb)

    opt, honest, picks = [], [], {}
    for seed in SEEDS:
        # OPTIMISTIC: best config on full 253, eval on full 253
        allrows = np.arange(len(y))
        scores = {cn: imp_on(allrows, cf, seed) for cn, cf in configs.items()}
        best = min(scores, key=scores.get); opt.append(scores[best]); picks[best] = picks.get(best, 0) + 1
        # HONEST nested: outer 5-fold; select config on outer-train (inner CV), eval selected on outer-test
        outer = scaffold_kfold_indices(list(scaf), n_splits=5, seed=seed)
        pred_n = anchor.copy(); pred_b = anchor.copy()
        for tri, vai in outer:
            inner_scores = {cn: imp_on(tri, cf, seed + 7) for cn, cf in configs.items()}
            sel = min(inner_scores, key=inner_scores.get)
            Xs = np.hstack([base, configs[sel]])
            mb = lgbm().fit(base[tri], resid[tri]); mf = lgbm().fit(Xs[tri], resid[tri])
            lo, hi = np.quantile(y[tri], QL), np.quantile(y[tri], QH)
            pred_b[vai] = np.clip(anchor[vai] + mb.predict(base[vai]), lo, hi)
            pred_n[vai] = np.clip(anchor[vai] + mf.predict(Xs[vai]), lo, hi)
        honest.append(rae(y, pred_n) - rae(y, pred_b))
        print(f"  seed {seed}: optimistic(best={best})={opt[-1]:+.5f}  honest_nested={honest[-1]:+.5f}")
    opt, honest = np.array(opt), np.array(honest)
    print("\n" + "=" * 64)
    print(f"OPTIMISTIC (select+eval on same 253): {opt.mean():+.5f} +/- {opt.std():.5f}")
    print(f"HONEST (nested, eval on untouched fold): {honest.mean():+.5f} +/- {honest.std():.5f}  wins={int((honest<0).sum())}/{len(SEEDS)}")
    print(f"SELECTION INFLATION = {opt.mean()-honest.mean():+.5f}  | config picks: {picks}")
    print(">>> SIGNAL SURVIVES honest selection-corrected CV" if honest.mean() < -0.005 and (honest < 0).mean() >= 0.8
          else ">>> signal LARGELY SELECTION-INFLATED -- be skeptical" if honest.mean() > -0.003
          else ">>> partial: real but smaller than reported")
    print("=" * 64)
    json.dump({"optimistic": float(opt.mean()), "honest_nested": float(honest.mean()),
               "inflation": float(opt.mean() - honest.mean()), "honest_wins": int((honest < 0).sum()),
               "picks": picks}, open(f"{U}/nb1016_selection_honest.json", "w"), indent=2)


if __name__ == "__main__":
    main()
