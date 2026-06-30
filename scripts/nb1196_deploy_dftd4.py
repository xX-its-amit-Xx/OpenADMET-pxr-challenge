"""nb1196 deploy — ADD 15 DFT-D4 dispersion/polarizability scalars to the deployed
config (honest gate WIN, nb1196_dftd4_deployed_gate).

  control(AIMNet2+strain) 0.4268  treatment(+D4) 0.4252  matched delta -0.00153
  (seeds -0.0047/+0.0004/-0.0003, 2/3 neg), treat < 0.4268 -> DEPLOY.

D4 is MARGINAL-OVER-DEPLOYED, so the 4 GBMs train on combined + 9 AIMNet2 + 8 strain
+ 15 DFT-D4 scalars (each block per-train StandardScaler) and predict the 513 with
the same blocks. GNN/CheMeleon/TabPFN members UNCHANGED.
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
STR = "C:/pxr_work/strain/strain_features.csv"
SCOLS = ["strain_relax_mean", "strain_relax_max", "conf_espread", "conf_erange",
         "conf_n", "rmsd_mean", "rmsd_max", "e_per_heavy"]
D4 = "C:/pxr_work/d4/d4_features.csv"
DCOLS = ["d4_alpha_sum", "d4_alpha_mean", "d4_alpha_std", "d4_alpha_max",
         "d4_c6diag_mean", "d4_c6diag_std", "d4_c6_total", "d4_edisp",
         "d4_edisp_per_atom", "d4_cn_mean", "d4_cn_max", "d4_qeeq_min",
         "d4_qeeq_max", "d4_qeeq_std", "d4_qeeq_absum"]
N_SEEDS = 3


def topK_configs():
    recs = [json.loads(l) for l in open(LOG) if l.strip()]
    valid = [r for r in recs if "ps_rae" in r and "error" not in r and r["arch"] in ("lgbm", "xgb", "cat", "histgb")]
    valid.sort(key=lambda r: r["ps_rae"]); bp = {}
    for r in valid: bp.setdefault(r["arch"], r)
    return list(bp.values())


def block(df, names, cols):
    df = df.drop_duplicates(subset="name", keep="first")
    sub = df.set_index("name").reindex(names)
    X = sub[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    med = np.nanmedian(X, axis=0); inds = np.where(np.isnan(X)); X[inds] = np.take(med, inds[1])
    return X


def fit_pred(c, Xtr, ytr, Xte, Xex_tr, Xex_te, se):
    A = np.hstack([Xtr, Xex_tr]); B = np.hstack([Xte, Xex_te])
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

    adf = pd.read_csv(AIM); sdf = pd.read_csv(STR); ddf = pd.read_csv(D4)
    Xqm_tr = block(adf[adf.src == "train"], tr["name"], ACOLS)
    Xqm_te = block(adf[adf.src == "test"], te["name"], ACOLS)
    Xst_tr = block(sdf[sdf.src == "train"], tr["name"], SCOLS)
    Xst_te = block(sdf[sdf.src == "test"], te["name"], SCOLS)
    Xd4_tr = block(ddf[ddf.src == "train"], tr["name"], DCOLS)
    Xd4_te = block(ddf[ddf.src == "test"], te["name"], DCOLS)
    scq = StandardScaler().fit(Xqm_tr); Xqm_tr = scq.transform(Xqm_tr); Xqm_te = scq.transform(Xqm_te)
    scs = StandardScaler().fit(Xst_tr); Xst_tr = scs.transform(Xst_tr); Xst_te = scs.transform(Xst_te)
    scd = StandardScaler().fit(Xd4_tr); Xd4_tr = scd.transform(Xd4_tr); Xd4_te = scd.transform(Xd4_te)
    Xex_tr = np.hstack([Xqm_tr, Xst_tr, Xd4_tr]); Xex_te = np.hstack([Xqm_te, Xst_te, Xd4_te])
    print(f"train {len(tr)} | test {len(te)} | extra cols {Xex_tr.shape[1]} "
          f"(AIMNet2 {len(ACOLS)} + strain {len(SCOLS)} + D4 {len(DCOLS)})", flush=True)

    topK = topK_configs()
    print(f"deployed 4-GBM: {[c['arch'] for c in topK]} + sisterNR_gnn + CheMeleon + TabPFN", flush=True)
    ps = [fit_pred(c, Xtr, ytr, Xte, Xex_tr, Xex_te, se) for c in topK]
    te_sn = np.mean([np.load(f"{MTL}/te_sn_seed{s}.npy") for s in range(N_SEEDS)], 0)
    ps.append(te_sn)
    ps.append(np.load(f"{SD}/chemeleon_lgbm_te.npy"))
    ps.append(np.load(f"{SD}/tabpfn_te.npy"))
    full = np.clip(np.mean(ps, 0), np.quantile(ytr, 0.05), np.quantile(ytr, 0.98))

    out = "submissions/nb1196_dftd4_ensemble.csv"
    pd.DataFrame({"SMILES": te["smiles"], "Molecule Name": te["name"], "pEC50": full}).to_csv(out, index=False)
    print(f"saved {out}  (n={len(full)}, mean {full.mean():.3f}, range {full.min():.2f}-{full.max():.2f})", flush=True)
    json.dump({"rae": 0.4252,
               "via": "+DFT-D4 dispersion/polarizability scalars on top of AIMNet2+strain in 4-GBM (matched -0.00153 vs deployed, nb1196_gate)",
               "submission": out,
               "members": [c["arch"] for c in topK] + ["sisterNR_gnn", "chemeleon", "tabpfn"],
               "qm_feats": ACOLS, "strain_feats": SCOLS, "d4_feats": DCOLS},
              open(f"{SD}/best_ensemble.json", "w"), indent=2)
    print("updated best_ensemble.json -> 0.4252 (+DFT-D4)", flush=True)


if __name__ == "__main__":
    main()
