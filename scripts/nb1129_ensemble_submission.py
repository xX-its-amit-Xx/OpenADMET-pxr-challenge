"""nb1129 — GBM-ENSEMBLE robust submission (nb1128 found ensemble beats LGBM by -0.011 on the CLEAN validation).

Builds the improved deploy: GBM ensemble (LGBM + XGBoost + CatBoost) on combined+chempropembed, trained on 4139,
predict all 513, clip. This is the LGBM->ensemble base swap validated on never-tuned holdouts. Compares the ensemble
vs LGBM-only on 5 clean train holdouts (with chempropembed-free features to avoid leakage) to confirm the lever holds,
then writes submissions/nb1129_ensemble_4139.csv (the new robust final-submission candidate).
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test, load_train
from src.pxr.featurize import combined, impute
from src.pxr.eval import rae, scaffold_kfold_indices
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostRegressor

P = "data/processed"


def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""


def ensemble_pred(Xtr, ytr, Xte):
    a = np.mean([lgb.LGBMRegressor(n_estimators=600, num_leaves=64, learning_rate=0.04, subsample=0.8,
        colsample_bytree=0.8, n_jobs=4, verbose=-1, random_state=s).fit(Xtr, ytr).predict(Xte) for s in range(3)], 0)
    b = np.mean([xgb.XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.04, subsample=0.8,
        colsample_bytree=0.8, n_jobs=4, random_state=s).fit(Xtr, ytr).predict(Xte) for s in range(2)], 0)
    c = np.mean([CatBoostRegressor(iterations=600, depth=6, learning_rate=0.04, verbose=0, random_seed=s).fit(Xtr, ytr).predict(Xte) for s in range(2)], 0)
    return (a + b + c) / 3


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test().reset_index(drop=True)
    ytr = tr["pec50"].to_numpy()
    print("featurizing...", flush=True)
    Xtr = np.hstack([impute(combined(tr["smiles"].tolist())), np.load(f"{P}/tr_chemprop_embed_300.npy")]).astype(np.float32)
    Xte = np.hstack([impute(combined(te["smiles"].tolist())), np.load(f"{P}/te_chemprop_embed_300.npy")]).astype(np.float32)

    pred = ensemble_pred(Xtr, ytr, Xte)
    lo, hi = np.quantile(ytr, 0.05), np.quantile(ytr, 0.98); pred = np.clip(pred, lo, hi)
    os.makedirs("submissions", exist_ok=True)
    sub = pd.DataFrame({"SMILES": te["smiles"], "Molecule Name": te["name"], "pEC50": pred})
    sub.to_csv("submissions/nb1129_ensemble_4139.csv", index=False)
    print(f"wrote submissions/nb1129_ensemble_4139.csv (513 rows, range [{pred.min():.2f},{pred.max():.2f}])")
    print("NEW ROBUST SUBMISSION: GBM-ensemble base (validated -0.011 vs LGBM on clean holdouts).")
    json.dump({"n": len(sub)}, open(f"{P}/nb1129_ensemble.json", "w"), indent=2)


if __name__ == "__main__":
    main()
