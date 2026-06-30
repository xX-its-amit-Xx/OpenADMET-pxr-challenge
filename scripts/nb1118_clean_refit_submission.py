"""nb1118 — (a) CLEAN rules-safe submission: nb3200-style base REFIT on 4139 train + 253 released Analog-Set-1 labels.

Legitimate Phase-2 use of released labels (model trained on them), NOT label-pasting/truth-injection. Predicts all 513,
applies per-fold y-range clip. Reports how much including the 253 changes the BLINDED-260 predictions (the part that
actually matters if the LB scores blinded-only). Output: submissions/nb1118_clean_refit_4139p253.csv (SMILES, Molecule Name, pEC50).
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test, load_train
from src.pxr.featurize import combined, impute
from src.pxr.eval import rae
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb

P = "data/processed"


def bag(Xtr, ytr, Xte, nseed=5):
    ps = []
    for s in range(nseed):
        m = lgb.LGBMRegressor(n_estimators=600, num_leaves=64, learning_rate=0.04, subsample=0.8,
                              colsample_bytree=0.8, n_jobs=4, verbose=-1, random_state=s)
        m.fit(Xtr, ytr); ps.append(m.predict(Xte))
    return np.mean(ps, 0)


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test().reset_index(drop=True)
    unb = np.load(f"{P}/_audit_unblind_idx.npy"); yunb = np.load(f"{P}/_audit_unblind_y.npy")
    blind = np.array([i for i in range(len(te)) if i not in set(unb.tolist())])

    Xtr = np.hstack([impute(combined(tr["smiles"].tolist())), np.load(f"{P}/tr_chemprop_embed_300.npy")])
    Xte = np.hstack([impute(combined(te["smiles"].tolist())), np.load(f"{P}/te_chemprop_embed_300.npy")])
    ytr = tr["pec50"].to_numpy()

    # baseline: train on 4139 only
    pred_4139 = bag(Xtr, ytr, Xte)
    # refit: train on 4139 + 253 released labels
    Xref = np.vstack([Xtr, Xte[unb]]); yref = np.concatenate([ytr, yunb])
    pred_4392 = bag(Xref, yref, Xte)

    # per-fold-style y-range clip (q05/q98 of training labels)
    lo, hi = np.quantile(yref, 0.05), np.quantile(yref, 0.98)
    pred = np.clip(pred_4392.copy(), lo, hi)
    # use released truth for the 253 (legit — they ARE the labels) OR keep model? -> keep MODEL prediction (clean, no pasting)
    # report
    print(f"refit changes BLINDED-260 predictions vs 4139-only: mean|delta| {np.mean(np.abs(pred_4392[blind]-pred_4139[blind])):.3f}")
    print(f"  253 in-sample RAE (refit, sanity) {rae(yunb, pred_4392[unb]):.4f}  (low = model learned the released labels)")
    print(f"  clip bounds [{lo:.2f},{hi:.2f}] | pred range [{pred.min():.2f},{pred.max():.2f}]")

    sub = pd.DataFrame({"SMILES": te["smiles"], "Molecule Name": te["name"], "pEC50": pred})
    os.makedirs("submissions", exist_ok=True)
    out = "submissions/nb1118_clean_refit_4139p253.csv"
    sub.to_csv(out, index=False)
    print(f"\nwrote {out} ({len(sub)} rows, cols {list(sub.columns)})")
    print("CLEAN: model trained on released data, no label-pasting. Validate: python tutorial/validation/activity_validation.py " + out)


if __name__ == "__main__":
    main()
