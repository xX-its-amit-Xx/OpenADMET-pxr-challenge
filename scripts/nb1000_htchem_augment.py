"""nb1000 — does the NEW htchem data help as training augmentation?
441 corrected-pEC50 analog compounds (crude/semi-pure, enriched actives). Test LGBM(combined) on
4139 vs 4139+htchem, with/without SE-weighting (htchem is crude -> higher SE -> down-weighted).
Deploy 4139(+htchem) -> predict 253. Multi-seed. The empirical 'does the new data move RAE' answer.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train, load_test
from src.pxr.eval import rae
from src.pxr.featurize import combined, impute
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb

D = "data/processed"; OUT = "C:/pxr_struct/dash"
SEEDS = [0, 1, 2, 3, 4]


def load_htchem():
    h = pd.read_csv("data/external/htchem/pxr-challenge_htchem-libraries_TRAIN.csv")
    sp = pd.read_csv("data/external/htchem/pxr-challenge_96-compound-uscale-semi-pure_TRAIN.csv")
    a = pd.DataFrame({"smiles": h["SMILES"], "pec50": pd.to_numeric(h["Corrected Crude pEC50 (log)"], errors="coerce"),
                      "se": pd.to_numeric(h["Crude DRC pEC50 SE (log)"], errors="coerce")})
    b = pd.DataFrame({"smiles": sp["SMILES"], "pec50": pd.to_numeric(sp["Corrected Semi-Pure pEC50 (log)"], errors="coerce"),
                      "se": pd.to_numeric(sp["Semi-Pure DRC pEC50 SE (log)"], errors="coerce")})
    n = pd.concat([a, b], ignore_index=True).dropna(subset=["smiles", "pec50"])
    n["se"] = n["se"].fillna(n["se"].median()).clip(0.05, 2.0)
    return n.reset_index(drop=True)


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test(); unb = np.load(f"{D}/_audit_unblind_idx.npy"); y253 = np.load(f"{D}/_audit_unblind_y.npy")
    ht = load_htchem()
    print(f"train {len(tr)} + htchem {len(ht)}")

    y_tr = tr["pec50"].to_numpy(float)
    se_tr = np.clip(tr["pec50_se"].fillna(tr["pec50_se"].median()).to_numpy(float), 0.05, 2.0)
    print("featurizing ...", flush=True)
    Xtr = impute(combined(tr["smiles"].tolist())).astype(np.float32)
    Xht = impute(combined(ht["smiles"].tolist())).astype(np.float32)
    Xte = impute(combined(te["smiles"].to_numpy()[unb].tolist())).astype(np.float32)
    y_ht = ht["pec50"].to_numpy(float); se_ht = ht["se"].to_numpy(float)

    Xaug = np.vstack([Xtr, Xht]); y_aug = np.concatenate([y_tr, y_ht])
    w_tr = np.clip(1/se_tr**2, 0, 50); w_ht = np.clip(1/se_ht**2, 0, 50)
    w_aug = np.concatenate([w_tr, w_ht])
    w_base = np.clip(1/se_tr**2, 0, 50)

    def deploy(X, y, w=None):
        rs = []
        for s in SEEDS:
            m = lgb.LGBMRegressor(n_estimators=600, num_leaves=64, learning_rate=0.04, random_state=s, n_jobs=4, verbose=-1)
            m.fit(X, y, sample_weight=w); rs.append(rae(y253, m.predict(Xte)))
        return np.mean(rs), np.std(rs)

    configs = {
        "4139 only (unweighted)": (Xtr, y_tr, None),
        "4139 only (SE-weighted)": (Xtr, y_tr, w_base),
        "4139+htchem (unweighted)": (Xaug, y_aug, None),
        "4139+htchem (SE-weighted)": (Xaug, y_aug, w_aug),
    }
    print("\n=== deploy -> 253 RAE (mean over 5 LGBM seeds) ===")
    res = {}
    for nm, (X, y, w) in configs.items():
        mu, sd = deploy(X, y, w); res[nm] = round(float(mu), 4)
        print(f"  {nm:30s} 253-RAE = {mu:.4f} +/- {sd:.4f}")
    best = min(res, key=res.get)
    base = res["4139 only (unweighted)"]
    print(f"\nbest: {best} = {res[best]} (delta vs 4139-only-unweighted: {res[best]-base:+.4f})")
    print(">>> htchem augmentation HELPS" if res.get("4139+htchem (SE-weighted)", 9) < res["4139 only (SE-weighted)"] - 0.003
          else ">>> htchem augmentation does NOT clearly help on 253 (off-manifold to novel test, as scoped)")
    json.dump(res, open(f"{OUT}/nb1000_htchem_augment.json", "w"), indent=2)
    print(f"saved -> {OUT}/nb1000_htchem_augment.json")


if __name__ == "__main__":
    main()
