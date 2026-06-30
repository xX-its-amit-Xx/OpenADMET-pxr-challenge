"""nb1092 — CROSS-TARGET TRANSFER (the user's 'borrow discrimination from rich external targets' idea, on ground truth).

cycle-299 found the 7-NR cross-target affinity profile REDUNDANT with nb3200 — but PXR has decent data, so chemistry
is already saturated. The user's hypothesis is sharper: a model trained on a DATA-RICH neighbor (FXR/PPARg, 2500 cpds)
carries discrimination in chemical regions a DATA-POOR target (RXRa/LXRa/papyrus-PXR, ~440) never measured. Does
borrowing it help WHERE the target is data-poor? The mirror lets us test transfer where ground truth exists, and find
whether there is a poverty regime where transfer helps (which would re-open the lever for PXR Phase-2).

For each POOR target T (held-out scaffold-disjoint test):
  - BASELINE   : LGBM(combined) on T's poor-train only
  - +TRANSFER  : append cross-target features = predicted activity on each RICH donor (donor model = LGBM on donor FULL
                 data), retrain LGBM on T's poor-train. (borrow donor discrimination as features)
  - POOLED-MTL : one LGBM on (T-train + all donor data) with target-id one-hots; predict T-test. (joint manifold)
Compare RAE on T's blinded test. Donor models never see T's test labels -> no leakage.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.eval import rae, scaffold_kfold_indices
from src.pxr.featurize import combined, impute
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb

P = "data/processed"
PARQ = "data/external/papyrus_pxr_nr.parquet"
POOR = ["RXRa", "LXRa", "PXR", "VDR"]
DONORS = ["FXR", "PPARg", "RXRa", "LXRa", "PXR", "VDR"]   # all NRs; a target never donates to itself
N_TEST = 200
SEED = 42


def murcko(s):
    m = Chem.MolFromSmiles(str(s))
    if not m: return None
    try: return MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    except Exception: return None


def mk(Xtr, ytr, Xte):
    m = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, n_jobs=4, verbose=-1)
    m.fit(Xtr, ytr); return m


def main():
    df = pd.read_parquet(PARQ)
    df = df[df["standard_type"].isin(["EC50", "AC50"])].drop_duplicates(["target_name", "inchikey"]).reset_index(drop=True)
    df["scaf"] = df["std_smiles"].map(murcko); df = df.dropna(subset=["scaf"]).reset_index(drop=True)

    # featurize the whole table once (cache per std_smiles)
    uniq = df["std_smiles"].drop_duplicates().tolist()
    print(f"featurizing {len(uniq)} unique smiles...", flush=True)
    Xu = impute(combined(uniq)); idx = {s: i for i, s in enumerate(uniq)}
    df["fid"] = df["std_smiles"].map(idx)

    # donor models (trained on each donor's FULL data) -> reusable predictors
    donor_models = {}
    for d in DONORS:
        sub = df[df["target_name"] == d]
        if len(sub) < 120: continue
        donor_models[d] = mk(Xu[sub["fid"].to_numpy()], sub["pec50"].to_numpy(), Xu[:1])  # full-data donor predictor
    print("donor models:", list(donor_models), flush=True)

    rows = []
    for T in POOR:
        sub = df[df["target_name"] == T].reset_index(drop=True)
        if len(sub) < N_TEST + 120:
            print(f"[{T}] only {len(sub)} — skip"); continue
        folds = scaffold_kfold_indices(sub["scaf"].tolist(), n_splits=max(2, round(len(sub) / N_TEST)), seed=SEED)
        te = min((f[1] for f in folds), key=lambda ix: abs(len(ix) - N_TEST))[:N_TEST]
        te_set = set(te.tolist()); tr = np.array([i for i in range(len(sub)) if i not in te_set])
        fid = sub["fid"].to_numpy(); y = sub["pec50"].to_numpy()
        Xtr, Xte, ytr, yte = Xu[fid[tr]], Xu[fid[te]], y[tr], y[te]

        rae_base = rae(yte, mk(Xtr, ytr, Xte).predict(Xte))

        # +transfer features: donor predictions (donors != T)
        dons = [d for d in donor_models if d != T]
        Ctr = np.column_stack([donor_models[d].predict(Xtr) for d in dons])
        Cte = np.column_stack([donor_models[d].predict(Xte) for d in dons])
        rae_tr = rae(yte, mk(np.hstack([Xtr, Ctr]), ytr, np.hstack([Xte, Cte])).predict(np.hstack([Xte, Cte])))

        # pooled MTL: T-train + all donor data, one-hot target id
        pool_parts, py = [], []
        tid = {name: k for k, name in enumerate(DONORS)}
        for d in DONORS:
            s2 = df[df["target_name"] == d]
            f2 = s2["fid"].to_numpy()
            if d == T: f2 = fid[tr]; yy = ytr            # only T's TRAIN side
            else: yy = s2["pec50"].to_numpy()
            oh = np.zeros((len(f2), len(DONORS)), np.float32); oh[:, tid[d]] = 1
            pool_parts.append(np.hstack([Xu[f2], oh])); py.append(yy)
        Xpool = np.vstack(pool_parts); ypool = np.concatenate(py)
        ohte = np.zeros((len(te), len(DONORS)), np.float32); ohte[:, tid[T]] = 1
        rae_pool = rae(yte, mk(Xpool, ypool, np.hstack([Xte, ohte])).predict(np.hstack([Xte, ohte])))

        # is the donor-prediction feature even correlated with T truth? (upper bound on usefulness)
        best_donor_corr = max(abs(np.corrcoef(Cte[:, j], yte)[0, 1]) for j in range(Cte.shape[1]))
        rows.append(dict(target=T, n=int(len(sub)), n_test=int(len(te)),
                         rae_base=float(rae_base), rae_transfer=float(rae_tr), rae_pooled=float(rae_pool),
                         transfer_delta=float(rae_tr - rae_base), pooled_delta=float(rae_pool - rae_base),
                         best_donor_corr=float(best_donor_corr)))
        print(f"[{T}] base={rae_base:.4f}  +transfer={rae_tr:.4f} ({rae_tr-rae_base:+.4f})  "
              f"pooled={rae_pool:.4f} ({rae_pool-rae_base:+.4f})  best-donor-corr={best_donor_corr:.2f}", flush=True)

    json.dump(rows, open(f"{P}/nb1092_transfer.json", "w"), indent=2)
    print("\n=== CROSS-TARGET TRANSFER (does borrowing rich-neighbor discrimination help poor targets?) ===")
    print(pd.DataFrame(rows)[["target", "n", "rae_base", "rae_transfer", "transfer_delta",
                              "rae_pooled", "pooled_delta", "best_donor_corr"]].to_string(index=False))
    print(f"\nwrote {P}/nb1092_transfer.json")


if __name__ == "__main__":
    main()
