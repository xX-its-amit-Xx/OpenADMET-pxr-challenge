"""nb1004 — does Uni-Mol (stronger + decorrelated base) translate to a deployable gain?
Uni-Mol cvmean->253 = 0.6101 (beats chemprop_aux 0.6216), corr 0.88. Test RIGOROUSLY (cross-fit
blend weights + multi-seed, given the cycle-282/287 'decorrelated anchor' burn):
 (1) blend(unimol, chemprop_aux) cross-fit weight.
 (2) residual pipeline (anchor + LGBM(combined) + range-clip) for each anchor + the blend.
If a unimol-based anchor beats the chemprop_aux pipeline -> the representation path is REAL.
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

D = "data/processed"; OUT = "C:/pxr_struct/unimol"
SEEDS = [42, 101, 202, 303, 404, 505, 606]
QL, QH = 0.05, 0.98


def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else None


def clipped_resid(anchor, X, y, folds):
    pred = anchor.copy()
    for tri, vai in folds:
        m = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1).fit(X[tri], (y - anchor)[tri])
        p = anchor[vai] + m.predict(X[vai])
        lo, hi = np.quantile(y[tri], QL), np.quantile(y[tri], QH)
        pred[vai] = np.clip(p, lo, hi)
    return pred


def main():
    te = load_test(); unb = np.load(f"{D}/../_audit_unblind_idx.npy") if not os.path.exists(f"{D}/_audit_unblind_idx.npy") else np.load(f"{D}/_audit_unblind_idx.npy")
    unb = np.load("data/processed/_audit_unblind_idx.npy"); y = np.load("data/processed/_audit_unblind_y.npy")
    smiles = te["smiles"].to_numpy()[unb].tolist()
    ca = np.load("data/processed/te_chemprop_aux.npy")[unb]
    um = np.load(f"{OUT}/unimol_test_cvmean.npy")[unb]
    Xc = impute(combined(smiles)).astype(np.float32)
    scaf = [murcko(s) for s in smiles]
    print(f"anchors: chemprop_aux {rae(y,ca):.4f}, unimol {rae(y,um):.4f}, corr {np.corrcoef(ca,um)[0,1]:.3f}")

    rows = []
    for seed in SEEDS:
        folds = scaffold_kfold_indices(scaf, n_splits=5, seed=seed)
        # (1) cross-fit blend weight
        blend = ca.copy()
        for tri, vai in folds:
            best = (0.0, rae(y[tri], ca[tri]))
            for w in np.linspace(0, 1, 21):
                r = rae(y[tri], (w * um + (1 - w) * ca)[tri])
                if r < best[1]: best = (w, r)
            blend[vai] = best[0] * um[vai] + (1 - best[0]) * ca[vai]
        # (2) residual pipelines
        r_ca = rae(y, clipped_resid(ca, Xc, y, folds))
        r_um = rae(y, clipped_resid(um, Xc, y, folds))
        r_bl = rae(y, clipped_resid(blend, Xc, y, folds))
        rows.append({"seed": seed, "blend_anchor": round(rae(y, blend), 4),
                     "resid_chemprop": round(r_ca, 4), "resid_unimol": round(r_um, 4), "resid_blend": round(r_bl, 4)})
        print(f"  seed {seed}: blend-anchor={rae(y,blend):.4f} | resid: chemprop={r_ca:.4f} unimol={r_um:.4f} blend={r_bl:.4f}")

    import numpy as _np
    for key in ["resid_chemprop", "resid_unimol", "resid_blend"]:
        v = _np.array([r[key] for r in rows]); print(f"{key:16s} mean={v.mean():.4f} +/- {v.std():.4f}")
    rc = _np.array([r["resid_chemprop"] for r in rows]); rb = _np.array([r["resid_blend"] for r in rows]); ru = _np.array([r["resid_unimol"] for r in rows])
    d_bl = rb - rc; d_um = ru - rc
    print("\n" + "=" * 62)
    print(f"blend-residual vs chemprop-residual: mean delta {d_bl.mean():+.5f} +/- {d_bl.std():.5f} wins {int((d_bl<0).sum())}/{len(SEEDS)}")
    print(f"unimol-residual vs chemprop-residual: mean delta {d_um.mean():+.5f} +/- {d_um.std():.5f} wins {int((d_um<0).sum())}/{len(SEEDS)}")
    win = (d_bl.mean() < 0 and abs(d_bl.mean()) > d_bl.std()) or (d_um.mean() < 0 and abs(d_um.mean()) > d_um.std())
    print(">>> UNI-MOL anchor HELPS (representation path REAL) -> rebuild nb3200 with unimol anchor + SE-weighting" if win
          else ">>> unimol decorrelated but blend-inert (like cycle-282 fresh chemprop) -> representation alone not enough")
    print("=" * 62)
    json.dump({"ca_rae": round(rae(y, ca), 4), "um_rae": round(rae(y, um), 4), "rows": rows,
               "d_blend_mean": float(d_bl.mean()), "d_unimol_mean": float(d_um.mean())},
              open(f"{OUT}/nb1004_integration.json", "w"), indent=2)
    print(f"saved -> {OUT}/nb1004_integration.json")


if __name__ == "__main__":
    main()
