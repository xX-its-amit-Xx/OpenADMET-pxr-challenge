"""nb1100 — does the coactivator AF-2 signal add ON TOP of rich-z? (closes/opens the structural lever)

nb1099: coact blocks carry a faint but directionally-consistent activation signal (corr +0.07-0.08, 80-90% seeds)
but magnitude -0.002 < the -0.005 gate. rich-z (binary cofold) already encodes helix-12 activation geometry
(deploy ~-0.008). Same mechanism -> likely redundant. Decisive test: predict (y - nb3200) residual from
  (a) rich-z alone   (b) rich-z + pxr_pep(AF2)   (c) rich-z + lig_pxr(coact)
30-seed scaffold-CV residual-LGBM; if (b)/(c) beat (a) by a stable margin, the coactivator interface is a NEW
structural axis beyond rich-z. If not, it's a weaker echo -> structural lever closed at rich-z.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test
from src.pxr.eval import rae, scaffold_kfold_indices
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

P = "data/processed"; MO = "C:/pxr_struct/boltz/modal"


def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""


def resid_oof(X, resid, scaf, seed):
    oof = np.zeros(len(resid))
    for trn, val in scaffold_kfold_indices(scaf, n_splits=5, seed=seed):
        m = lgb.LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, n_jobs=4, verbose=-1, random_state=seed)
        m.fit(X[trn], resid[trn]); oof[val] = m.predict(X[val])
    return oof


def main():
    unb = np.load(f"{P}/_audit_unblind_idx.npy"); y = np.load(f"{P}/_audit_unblind_y.npy")
    anchor = np.load(f"{P}/nb3200_pred_oof.npy"); resid = y - anchor
    te = load_test(); scaf = [murcko(s) for s in te["smiles"].to_numpy()[unb]]
    richz = np.load(f"{MO}/test_richz.npy")[unb]
    pxr_pep = np.load(f"{P}/coact_pxrpep.npy")[unb]
    lig_pxr = np.load(f"{P}/coact_ligpxr.npy")[unb]

    def clean(B): return StandardScaler().fit_transform(np.where(np.isfinite(B), B, 0.0))
    blocks = {"richz_only": clean(richz),
              "richz+pxr_pep": clean(np.hstack([richz, pxr_pep])),
              "richz+lig_pxr": clean(np.hstack([richz, lig_pxr])),
              "richz+both_coact": clean(np.hstack([richz, pxr_pep, lig_pxr]))}
    print(f"nb3200 anchor RAE {rae(y, anchor):.4f}\n")
    print(f"{'block':20s} {'corr':>8s} {'blend_delta':>12s} {'frac_improved':>14s}")
    out = {}
    for name, B in blocks.items():
        deltas, corrs = [], []
        for seed in range(1200, 1230):
            oof = resid_oof(B, resid, scaf, seed)
            corrs.append(np.corrcoef(oof, resid)[0, 1] if oof.std() > 1e-9 else 0.0)
            best = rae(y, anchor)
            for w in np.linspace(0, 1.5, 31):
                best = min(best, rae(y, anchor + w * oof))
            deltas.append(best - rae(y, anchor))
        d = float(np.mean(deltas))
        out[name] = dict(corr=float(np.mean(corrs)), blend_delta=d,
                         frac=float(np.mean(np.array(deltas) < -1e-6)))
        print(f"{name:20s} {np.mean(corrs):>+8.3f} {d:>+12.4f} {np.mean(np.array(deltas)<-1e-6):>14.2f}")
    base = out["richz_only"]["blend_delta"]
    print(f"\nADD-OVER-RICHZ: pxr_pep {out['richz+pxr_pep']['blend_delta']-base:+.4f} | "
          f"lig_pxr {out['richz+lig_pxr']['blend_delta']-base:+.4f} | "
          f"both {out['richz+both_coact']['blend_delta']-base:+.4f}  (negative = coact adds beyond rich-z)")
    json.dump(out, open(f"{P}/nb1100_coact_vs_richz.json", "w"), indent=2)


if __name__ == "__main__":
    main()
