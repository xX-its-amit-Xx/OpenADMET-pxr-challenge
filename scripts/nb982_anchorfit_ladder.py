"""nb982 — does anchor-fit survive on the chemprop substrate (the ladder test)?

ADMET was REAL over fingerprints but ABSORBED by chempropembed (nb964) -> no ladder break.
Anchor-fit is 3D GEOMETRY (orthogonal to chemprop's 2D embeddings) -> may NOT be absorbed.
Test: chemprop_aux + LGBM(combined+chempropembed [+anchorfit]) residual + range-clip, 253-unblind
scaffold-CV, multi-seed. If anchorfit's delta is stably negative HERE, it's a real ladder candidate.
Reuses anchorfit test features from nb979_anchorfit_test.npy. All on C:.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test
from src.pxr.eval import rae, scaffold_kfold_indices
from src.pxr.featurize import combined, impute
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb
from sklearn.impute import SimpleImputer

D = "data/processed"; OUT = "C:/pxr_struct"
SEEDS = [42, 101, 202, 303, 404, 505, 606]
QL, QH = 0.05, 0.98


def murcko(s):
    try:
        m = Chem.MolFromSmiles(s); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else None
    except Exception: return None


def clipped_cv(X, resid, anchor, y, folds):
    pred = anchor.copy()
    for tri, vai in folds:
        m = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1).fit(X[tri], resid[tri])
        p = anchor[vai] + m.predict(X[vai])
        lo, hi = np.quantile(y[tri], QL), np.quantile(y[tri], QH)
        pred[vai] = np.clip(p, lo, hi)
    return float(rae(y, pred)), pred


def main():
    te = load_test(); unb = np.load(f"{D}/_audit_unblind_idx.npy"); y = np.load(f"{D}/_audit_unblind_y.npy")
    smiles = te["smiles"].to_numpy()[unb].tolist()
    scaf = [murcko(s) for s in smiles]
    anchor = np.load(f"{D}/te_chemprop_aux.npy")[unb]
    Xc = impute(combined(smiles)).astype(np.float32)
    emb = np.load(f"{D}/te_chemprop_embed_300.npy")[unb].astype(np.float32)
    base = np.hstack([Xc, emb])
    F = np.load(f"{OUT}/nb979_anchorfit_test.npy")[unb]
    A = np.clip(np.nan_to_num(SimpleImputer(strategy="median").fit_transform(F)), -1e6, 1e6).astype(np.float32)
    base_a = np.hstack([base, A])
    resid = y - anchor
    print(f"anchor RAE {rae(y,anchor):.4f}; base={base.shape} +anchorfit={base_a.shape}; nb3200 ceiling 0.4416\n")

    rows = []
    for seed in SEEDS:
        folds = scaffold_kfold_indices(scaf, n_splits=5, seed=seed)
        rb, _ = clipped_cv(base, resid, anchor, y, folds)
        ra, pa = clipped_cv(base_a, resid, anchor, y, folds)
        rows.append({"seed": seed, "base": round(rb, 4), "base_anchorfit": round(ra, 4), "delta": round(ra - rb, 5)})
        print(f"  seed {seed}: base={rb:.4f} +anchorfit={ra:.4f} delta={ra-rb:+.5f}")

    d = np.array([r["delta"] for r in rows])
    stable = d.mean() < 0 and abs(d.mean()) > d.std()
    bm = np.mean([r["base"] for r in rows]); am = np.mean([r["base_anchorfit"] for r in rows])
    print("\n" + "=" * 62)
    print(f"anchorfit on chemprop substrate (253): mean delta={d.mean():+.5f} std={d.std():.5f} "
          f"wins={int((d<0).sum())}/{len(SEEDS)} stable={stable}")
    print(f"  base mean RAE={bm:.4f}  +anchorfit mean RAE={am:.4f}")
    print(">>> ANCHOR-FIT SURVIVES on chemprop substrate -> REAL LADDER CANDIDATE (build deploy)" if stable
          else ">>> absorbed by chempropembed too (like ADMET) -> helps fingerprints only")
    print("=" * 62)
    json.dump({"anchor_rae": round(float(rae(y, anchor)), 4), "rows": rows, "delta_mean": float(d.mean()),
               "delta_std": float(d.std()), "stable": bool(stable), "base_mean": float(bm), "anchorfit_mean": float(am)},
              open(f"{OUT}/nb982_anchorfit_ladder.json", "w"), indent=2)
    print(f"saved -> {OUT}/nb982_anchorfit_ladder.json")


if __name__ == "__main__":
    main()
