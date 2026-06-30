"""nb1177 deploy — DEPLOY AIMNet2 9 QM scalars into the 4-GBM members (honest matched gate WIN).

nb1177_aimnet2_deployed_gate (deployed 4-GBM + CheMeleon + TabPFN + sn_oof config):
  control   0.4361  treatment(+AIMNet2 QM) 0.4291  matched delta -0.0070  (3/3 seeds, all negative)
  treatment < deployed best 0.4367 -> DEPLOY.

Unlike the data-aux-head deploys, the GNN/CheMeleon/TabPFN members are UNCHANGED (deployed sisterNR te_sn
average + chemeleon_lgbm_te + tabpfn_te). The ONLY change: the 4 GBMs (lgbm/histgb/xgb/cat) train on
combined + 9 AIMNet2 QM scalars (per-train StandardScaler) and predict the 513 with the same scalars.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from nb1126_combinatorial_search import feature_matrix, CACHE, make_model
from src.pxr.data import load_train, load_test
from sklearn.preprocessing import StandardScaler
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

P = "data/processed"; SD = "C:/pxr_work/search"; MTL = "C:/pxr_work/mtl"; LOG = f"{SD}/results.jsonl"
AIM = "C:/pxr_work/aimnet2/aimnet_features.csv"
ACOLS = ["aimnet_energy", "aimnet_qmin", "aimnet_qmax", "aimnet_qabs_mean", "aimnet_qstd",
         "aimnet_qsum_abs", "aimnet_dipole", "aimnet_fmax", "aimnet_frms"]
N_SEEDS = 3


def topK_configs():
    recs = [json.loads(l) for l in open(LOG) if l.strip()]
    valid = [r for r in recs if "ps_rae" in r and "error" not in r and r["arch"] in ("lgbm", "xgb", "cat", "histgb")]
    valid.sort(key=lambda r: r["ps_rae"]); bp = {}
    for r in valid: bp.setdefault(r["arch"], r)
    return list(bp.values())


def qm_matrix(df, names):
    sub = df.set_index("name").reindex(names)
    X = sub[ACOLS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    med = np.nanmedian(X, axis=0); inds = np.where(np.isnan(X)); X[inds] = np.take(med, inds[1])
    return X


def fit_pred_qm(c, Xtr, ytr, Xte, Xqm_tr, Xqm_te, se):
    A = np.hstack([Xtr, Xqm_tr]); B = np.hstack([Xte, Xqm_te])
    mask = np.ones(len(ytr), bool)
    if c["prep"] == "noisy20": mask = se <= np.quantile(se, 0.8)
    elif c["prep"] == "noisy30": mask = se <= np.quantile(se, 0.7)
    use = np.where(mask)[0]
    m = make_model(c["arch"], c["hp"])
    if c["arch"] in ("ridge", "enet"):
        sc = StandardScaler().fit(A[use]); m.fit(sc.transform(A[use]), ytr[use]); return m.predict(sc.transform(B))
    m.fit(A[use], ytr[use]); return m.predict(B)


def main():
    d = np.load(CACHE); ytr = d["ytr"]; se = d["se"]
    Xtr, _ = feature_matrix(d, "combined")
    ff = np.load(f"{SD}/feats_full513.npz"); Xte = ff["combined"]
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test().reset_index(drop=True)

    adf = pd.read_csv(AIM)
    Xqm_tr = qm_matrix(adf[adf.src == "train"], tr["name"])
    Xqm_te = qm_matrix(adf[adf.src == "test"], te["name"])
    scq = StandardScaler().fit(Xqm_tr)
    Xqm_tr = scq.transform(Xqm_tr); Xqm_te = scq.transform(Xqm_te)
    print(f"train {len(tr)} | test {len(te)} | QM cols {len(ACOLS)}", flush=True)

    topK = topK_configs()
    print(f"deployed 4-GBM: {[c['arch'] for c in topK]} + sisterNR_gnn + CheMeleon + TabPFN", flush=True)
    ps = [fit_pred_qm(c, Xtr, ytr, Xte, Xqm_tr, Xqm_te, se) for c in topK]
    te_sn = np.mean([np.load(f"{MTL}/te_sn_seed{s}.npy") for s in range(N_SEEDS)], 0)
    ps.append(te_sn)
    ps.append(np.load(f"{SD}/chemeleon_lgbm_te.npy"))
    ps.append(np.load(f"{SD}/tabpfn_te.npy"))
    full = np.clip(np.mean(ps, 0), np.quantile(ytr, 0.05), np.quantile(ytr, 0.98))

    out = "submissions/nb1177_aimnet2_ensemble.csv"
    pd.DataFrame({"SMILES": te["smiles"], "Molecule Name": te["name"], "pEC50": full}).to_csv(out, index=False)
    print(f"saved {out}  (n={len(full)}, mean {full.mean():.3f}, range {full.min():.2f}-{full.max():.2f})", flush=True)
    json.dump({"rae": 0.4297,
               "via": "+aimnet2_QM_scalars in 4-GBM (matched -0.0070 vs deployed, nb1177_gate)",
               "submission": out,
               "members": [c["arch"] for c in topK] + ["sisterNR_gnn", "chemeleon", "tabpfn"],
               "qm_feats": ACOLS},
              open(f"{SD}/best_ensemble.json", "w"), indent=2)
    print("updated best_ensemble.json -> 0.4297 (+aimnet2_QM_scalars)", flush=True)


if __name__ == "__main__":
    main()
