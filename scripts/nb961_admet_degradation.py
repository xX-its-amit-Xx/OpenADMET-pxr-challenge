"""nb961 — AXIS-C9 probe: do ADMET-AI predicted properties (CYP3A4/metabolism/solubility,
a biological axis orthogonal to substructure) flatten the novel-scaffold degradation curve?
RUN IN THE MAIN venv after nb960 produced admet_train.csv. Ref: nb952 deep-extrap MAE 0.5924.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train
from src.pxr.eval import rae, scaffold_kfold_indices
from src.pxr.featurize import combined, impute
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb
from sklearn.impute import SimpleImputer

D = "data/processed"
BINS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.01]


def murcko(s):
    try:
        m = Chem.MolFromSmiles(s); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else None
    except Exception: return None


def curve(y, p, sv):
    out = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (sv >= lo) & (sv < hi); n = int(m.sum())
        out.append((f"[{lo:.1f},{hi:.1f})", n, round(float(np.mean(np.abs(y[m]-p[m]))), 4) if n else None))
    return out


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    y = tr["pec50"].to_numpy(float)
    smiles = tr["smiles"].tolist()
    scaf = [murcko(s) for s in smiles]
    folds = scaffold_kfold_indices(scaf, n_splits=5, seed=42)
    max_sim = np.load(f"{D}/nb952_max_sim_4139.npy")

    adf = pd.read_csv("C:/admet_out/admet_train.csv")
    assert len(adf) == len(tr), f"admet rows {len(adf)} != train {len(tr)}"
    # numeric ADMET property columns only (drop smiles)
    prop_cols = [c for c in adf.columns if c != "smiles" and pd.api.types.is_numeric_dtype(adf[c])]
    A = SimpleImputer(strategy="median").fit_transform(adf[prop_cols].to_numpy(float)).astype(np.float32)
    A = np.clip(np.nan_to_num(A, posinf=1e6, neginf=-1e6), -1e6, 1e6)
    print(f"ADMET features: {A.shape} ({len(prop_cols)} properties)")
    print(f"  has CYP3A4-related: {[c for c in prop_cols if 'CYP3A4' in c or 'CYP' in c]}")

    Xc = impute(combined(smiles)).astype(np.float32)
    ref = json.load(open(f"{D}/nb952_stress_curve.json"))
    ref_deep = next(r for r in ref["lgbm_curve"] if r["bin"] == "[0.0,0.3)")["mae"]

    variants = {"combined (ref)": Xc,
                "combined+ADMET": np.hstack([Xc, A]),
                "ADMET-only": A}
    res = {}
    for tag, X in variants.items():
        oof = np.full(len(y), np.nan)
        for tri, vai in folds:
            m = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05,
                                  n_jobs=4, verbose=-1).fit(X[tri], y[tri])
            oof[vai] = m.predict(X[vai])
        rows = curve(y, oof, max_sim); deep = next(mae for b, n, mae in rows if b == "[0.0,0.3)")
        res[tag] = {"overall_rae": round(float(rae(y, oof)), 4), "deep_extrap_mae": deep, "curve": rows}
        print(f"\n{tag}: overall RAE={res[tag]['overall_rae']}  deep@sim<0.3={deep}")
        for b, n, mae in rows: print(f"   {b:14s} n={n:5d} MAE={mae}")
        np.save(f"{D}/nb961_{tag.split()[0].replace('+','_').replace('-','_')}_oof.npy", oof)

    print("\n" + "=" * 60)
    print(f"deep-extrap MAE @ sim<0.3 (ref LGBM {ref_deep}):")
    for tag in variants: print(f"  {tag:18s} {res[tag]['deep_extrap_mae']}")
    cb3d = res["combined+ADMET"]["deep_extrap_mae"]
    print("VERDICT:", "ADMET ADDS at novel end -> multi-seed verify" if cb3d < ref_deep
          else "ADMET adds nothing at novel end -> C9 negative")
    print("=" * 60)
    json.dump({"ref_deep": ref_deep, "n_props": len(prop_cols), "results": res},
              open(f"{D}/nb961_admet_degradation.json", "w"), indent=2)
    print(f"saved -> {D}/nb961_admet_degradation.json")


if __name__ == "__main__":
    main()
