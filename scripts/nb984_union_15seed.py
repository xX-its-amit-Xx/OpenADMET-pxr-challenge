"""nb983 — kitchen-sink final test: does the UNION of all real-over-fingerprint features
(ADMET + anchor-fit + USR-shape) collectively break the chemprop ladder?

Each is individually ABSORBED by chempropembed (nb964 ADMET, nb982 anchorfit, nb977 USR). But the
union might carry enough orthogonal signal to add. Low prior; this is the airtight close before
declaring the CPU-local feature space exhausted. 253-unblind, chemprop_aux residual + clip, multi-seed.
All features cached (513 -> [unb_idx]). All on C:.
"""
import os, sys, json
import numpy as np, pandas as pd
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
SEEDS = [1300,1301,1302,1303,1304,1305,1306,1307,1308,1309,1310,1311,1312,1313,1314]
QL, QH = 0.05, 0.98


def murcko(s):
    try:
        m = Chem.MolFromSmiles(s); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else None
    except Exception: return None


def imp(M):
    return np.clip(np.nan_to_num(SimpleImputer(strategy="median").fit_transform(M)), -1e6, 1e6).astype(np.float32)


def clipped_cv(X, resid, anchor, y, folds):
    pred = anchor.copy()
    for tri, vai in folds:
        m = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1).fit(X[tri], resid[tri])
        p = anchor[vai] + m.predict(X[vai])
        lo, hi = np.quantile(y[tri], QL), np.quantile(y[tri], QH)
        pred[vai] = np.clip(p, lo, hi)
    return float(rae(y, pred))


def main():
    te = load_test(); unb = np.load(f"{D}/_audit_unblind_idx.npy"); y = np.load(f"{D}/_audit_unblind_y.npy")
    smiles = te["smiles"].to_numpy()[unb].tolist()
    scaf = [murcko(s) for s in smiles]
    anchor = np.load(f"{D}/te_chemprop_aux.npy")[unb]
    Xc = impute(combined(smiles)).astype(np.float32)
    emb = np.load(f"{D}/te_chemprop_embed_300.npy")[unb].astype(np.float32)
    base = np.hstack([Xc, emb])

    # all real-over-fingerprint feature blocks for the 253
    adf = pd.read_csv("C:/admet_out/admet_test.csv")
    apc = [c for c in adf.columns if c != "smiles" and pd.api.types.is_numeric_dtype(adf[c])]
    ADMET = imp(adf[apc].to_numpy(float)[unb])
    ANCHOR = imp(np.load(f"{OUT}/nb979_anchorfit_test.npy")[unb])
    USR = imp(np.load(f"{OUT}/nb977_usr_test.npy")[unb])
    allf = np.hstack([base, ADMET, ANCHOR, USR])
    print(f"anchor RAE {rae(y,anchor):.4f}; base={base.shape} +ALL={allf.shape} "
          f"(ADMET {ADMET.shape[1]} + anchorfit {ANCHOR.shape[1]} + USR {USR.shape[1]})\n")

    resid = y - anchor
    rows = []
    for seed in SEEDS:
        folds = scaffold_kfold_indices(scaf, n_splits=5, seed=seed)
        rb = clipped_cv(base, resid, anchor, y, folds)
        ra = clipped_cv(allf, resid, anchor, y, folds)
        rows.append({"seed": seed, "base": round(rb, 4), "base_all": round(ra, 4), "delta": round(ra - rb, 5)})
        print(f"  seed {seed}: base={rb:.4f} +ALL={ra:.4f} delta={ra-rb:+.5f}")

    d = np.array([r["delta"] for r in rows]); stable = d.mean() < 0 and abs(d.mean()) > d.std()
    print("\n" + "=" * 62)
    print(f"UNION(ADMET+anchorfit+USR) on chemprop substrate: mean delta={d.mean():+.5f} "
          f"std={d.std():.5f} wins={int((d<0).sum())}/{len(SEEDS)} stable={stable}")
    print(">>> UNION BREAKS the ladder -> REAL deploy candidate" if stable
          else ">>> UNION absorbed too -> CPU-local feature space AIRTIGHT EXHAUSTED; only docking/better-base-rep/ext-data left")
    print("=" * 62)
    json.dump({"rows": rows, "delta_mean": float(d.mean()), "delta_std": float(d.std()), "stable": bool(stable)},
              open(f"{OUT}/nb984_union_15seed.json", "w"), indent=2)
    print(f"saved -> {OUT}/nb984_union_15seed.json")


if __name__ == "__main__":
    main()
