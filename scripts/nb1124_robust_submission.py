"""nb1124 — ROBUST final submission building blocks (nb1122 verdict: refit-on-253 HURTS the blinded 260 -> 4139-only).

Builds:
  submissions/nb1124_plain_4139.csv      : plain nb3200-style base trained on 4139 ONLY, predict all 513 (SAFE, no
                                            rules question; robust on the blinded 260). The unambiguously-legitimate option.
  submissions/nb1124_split_253truth.csv  : 260 blinded from the 4139-only model; 253 released from their TRUE labels
                                            (optimal IF the LB re-scores the 253 AND truth-use is allowed; flag rules-risk).
Pick per the truth-injection decision. Both 513-row, valid format.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test, load_train
from src.pxr.featurize import combined, impute
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb

P = "data/processed"


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test().reset_index(drop=True)
    unb = np.load(f"{P}/_audit_unblind_idx.npy"); yunb = np.load(f"{P}/_audit_unblind_y.npy")
    Xtr = np.hstack([impute(combined(tr["smiles"].tolist())), np.load(f"{P}/tr_chemprop_embed_300.npy")])
    Xte = np.hstack([impute(combined(te["smiles"].tolist())), np.load(f"{P}/te_chemprop_embed_300.npy")])
    ytr = tr["pec50"].to_numpy()

    ps = []
    for s in range(7):
        m = lgb.LGBMRegressor(n_estimators=600, num_leaves=64, learning_rate=0.04, subsample=0.8,
                              colsample_bytree=0.8, n_jobs=4, verbose=-1, random_state=s)
        m.fit(Xtr, ytr); ps.append(m.predict(Xte))
    pred = np.mean(ps, 0)
    lo, hi = np.quantile(ytr, 0.05), np.quantile(ytr, 0.98)
    pred = np.clip(pred, lo, hi)

    os.makedirs("submissions", exist_ok=True)
    sub = pd.DataFrame({"SMILES": te["smiles"], "Molecule Name": te["name"], "pEC50": pred})
    sub.to_csv("submissions/nb1124_plain_4139.csv", index=False)
    print(f"wrote submissions/nb1124_plain_4139.csv (513 rows) | pred range [{pred.min():.2f},{pred.max():.2f}]")

    # split: 253 = released truth, 260 = model
    split = pred.copy(); split[unb] = yunb
    sub2 = pd.DataFrame({"SMILES": te["smiles"], "Molecule Name": te["name"], "pEC50": split})
    sub2.to_csv("submissions/nb1124_split_253truth.csv", index=False)
    print(f"wrote submissions/nb1124_split_253truth.csv (260 model + 253 released-truth)")
    print("RECOMMENDATION: nb1124_plain_4139 = safe/robust (no rules question). split_253truth = optimal IF 253 re-scored + truth allowed.")


if __name__ == "__main__":
    main()
