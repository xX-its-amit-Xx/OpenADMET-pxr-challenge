"""nb1218 deploy — COMP-H: per-compound CheMeleon-vs-TabPFN routing via k-NN local MAE

Gate (nb1216 COMP-H): 3/3 seeds positive, delta=-0.0011
  ctrl 0.4242  comp_H 0.4231  (seeds: 0.4240/0.4432/0.4022 vs 0.4251/0.4440/0.4034)
  routing: for each compound, find k=10 nearest training-fold neighbors (ECFP4 Tanimoto),
  compute local MAE of chem_oof and tabpfn_oof on those neighbors; winner gets 2/14 weight,
  loser gets 1/14; GBMs+GNN at 1/7; normalize row-wise. A-priori thresholds, no holdout tuning.

Composition vs nb1206:
  - Same 4 GBMs (combined + AIMNet2 + strain + D4 + DBSTEP)
  - Same sisterNR GNN (3-seed mean)
  - ROUTING: CheMeleon ↔ TabPFN weights swapped per-compound by k-NN local MAE
  - k=10, weight ratios 2:1 (winner/loser), normalize to sum=1
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from nb1126_combinatorial_search import feature_matrix, CACHE, make_model
from src.pxr.data import load_train, load_test
from sklearn.preprocessing import StandardScaler
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs
RDLogger.DisableLog("rdApp.*")

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
DB = "C:/pxr_work/dbstep/dbstep_features.csv"
DBCOLS = ["vbur_r25", "vbur_r35", "vbur_r45", "vbur_r55", "vbur_r65",
          "ster_L", "ster_Bmin", "ster_Bmax", "ster_aniso",
          "npr1", "npr2", "asphericity", "spherocity", "eccentricity",
          "radgyr", "inertial_sf"]
N_SEEDS = 3
K_NN = 10


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


def smiles_to_ecfp(smiles_list, radius=2, nbits=2048):
    fps = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(str(s)) if s else None
        fps.append(rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius, nbits) if m else None)
    return fps


def compute_knn_routing(fps_train, fps_test, ytr, chem_oof, tab_oof, k=10):
    """
    For each test compound, find k nearest training neighbors, compute local MAE
    of chem_oof and tab_oof on those neighbors. Returns (chem_w, tab_w) arrays.
    winner gets 2/14, loser gets 1/14 (before normalization).
    """
    n_te = len(fps_test)
    chem_w = np.full(n_te, 1.0 / 7)  # default equal weight
    tab_w  = np.full(n_te, 1.0 / 7)
    fps_tr_valid = [fp for fp in fps_train if fp is not None]
    idx_valid = [i for i, fp in enumerate(fps_train) if fp is not None]

    n_chem_wins = 0
    for j, qfp in enumerate(fps_test):
        if qfp is None:
            continue
        sims = np.array(DataStructs.BulkTanimotoSimilarity(qfp, fps_tr_valid))
        topk = np.argsort(sims)[::-1][:k]
        nbr_global = [idx_valid[i] for i in topk]
        y_nbr = ytr[nbr_global]
        mae_chem = np.mean(np.abs(chem_oof[nbr_global] - y_nbr))
        mae_tab  = np.mean(np.abs(tab_oof[nbr_global]  - y_nbr))
        if mae_chem <= mae_tab:
            chem_w[j] = 2.0 / 14; tab_w[j] = 1.0 / 14; n_chem_wins += 1
        else:
            chem_w[j] = 1.0 / 14; tab_w[j] = 2.0 / 14

    print(f"  k-NN routing: chem wins {n_chem_wins}/{n_te} ({100*n_chem_wins/n_te:.1f}%)", flush=True)
    return chem_w, tab_w


def main():
    d = np.load(CACHE); ytr = d["ytr"]; se = d["se"]
    Xtr, _ = feature_matrix(d, "combined")
    ff = np.load(f"{SD}/feats_full513.npz"); Xte = ff["combined"]
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test().reset_index(drop=True)

    adf = pd.read_csv(AIM); sdf = pd.read_csv(STR); ddf = pd.read_csv(D4); bdf = pd.read_csv(DB)
    Xqm_tr = block(adf[adf.src == "train"], tr["name"], ACOLS)
    Xqm_te = block(adf[adf.src == "test"], te["name"], ACOLS)
    Xst_tr = block(sdf[sdf.src == "train"], tr["name"], SCOLS)
    Xst_te = block(sdf[sdf.src == "test"], te["name"], SCOLS)
    Xd4_tr = block(ddf[ddf.src == "train"], tr["name"], DCOLS)
    Xd4_te = block(ddf[ddf.src == "test"], te["name"], DCOLS)
    Xdb_tr = block(bdf[bdf.src == "train"], tr["name"], DBCOLS)
    Xdb_te = block(bdf[bdf.src == "test"], te["name"], DBCOLS)
    scq = StandardScaler().fit(Xqm_tr); Xqm_tr = scq.transform(Xqm_tr); Xqm_te = scq.transform(Xqm_te)
    scs = StandardScaler().fit(Xst_tr); Xst_tr = scs.transform(Xst_tr); Xst_te = scs.transform(Xst_te)
    scd = StandardScaler().fit(Xd4_tr); Xd4_tr = scd.transform(Xd4_tr); Xd4_te = scd.transform(Xd4_te)
    scb = StandardScaler().fit(Xdb_tr); Xdb_tr = scb.transform(Xdb_tr); Xdb_te = scb.transform(Xdb_te)
    Xex_tr = np.hstack([Xqm_tr, Xst_tr, Xd4_tr, Xdb_tr])
    Xex_te = np.hstack([Xqm_te, Xst_te, Xd4_te, Xdb_te])
    print(f"train {len(tr)} | test {len(te)} | extra cols {Xex_tr.shape[1]}", flush=True)

    # Train 4 GBMs on full training set
    topK = topK_configs()
    print(f"4-GBM: {[c['arch'] for c in topK]}", flush=True)
    Xfull_tr = np.hstack([Xtr, Xex_tr])
    Xfull_te = np.hstack([Xte, Xex_te])
    gbm_preds = []
    for c in topK:
        mask = np.ones(len(ytr), bool)
        if c["prep"] == "noisy20": mask = se <= np.quantile(se, 0.8)
        elif c["prep"] == "noisy30": mask = se <= np.quantile(se, 0.7)
        use = np.where(mask)[0]
        m = make_model(c["arch"], c["hp"])
        if c["arch"] in ("ridge", "enet"):
            sc2 = StandardScaler().fit(Xfull_tr[use]); m.fit(sc2.transform(Xfull_tr[use]), ytr[use])
            gbm_preds.append(m.predict(sc2.transform(Xfull_te)))
        else:
            m.fit(Xfull_tr[use], ytr[use]); gbm_preds.append(m.predict(Xfull_te))
        print(f"  {c['arch']} done", flush=True)

    # Load fixed-weight non-GBM members
    te_sn    = np.mean([np.load(f"{MTL}/te_sn_seed{s}.npy") for s in range(N_SEEDS)], 0)
    chem_te  = np.load(f"{SD}/chemeleon_lgbm_te.npy")
    tab_te   = np.load(f"{SD}/tabpfn_te.npy")
    chem_oof = np.load(f"{SD}/chemeleon_oof.npy")
    tab_oof  = np.load(f"{SD}/tabpfn_oof.npy")

    # k-NN routing on test compounds
    tr_col = "std_smiles" if "std_smiles" in tr.columns else "smiles"
    te_col = "std_smiles" if "std_smiles" in te.columns else "smiles"
    print("Computing k-NN routing for 513 test compounds...", flush=True)
    fps_train = smiles_to_ecfp(tr[tr_col].tolist())
    fps_test  = smiles_to_ecfp(te[te_col].tolist())
    chem_w, tab_w = compute_knn_routing(fps_train, fps_test, ytr, chem_oof, tab_oof, k=K_NN)

    # Assemble weighted prediction matrix
    n_te = len(te)
    gbm_w = np.full((n_te, 4), 1.0 / 7)
    gnn_w = np.full((n_te, 1), 1.0 / 7)
    w_mat = np.hstack([gbm_w, gnn_w, chem_w.reshape(-1, 1), tab_w.reshape(-1, 1)])
    w_mat = w_mat / w_mat.sum(axis=1, keepdims=True)  # normalize rows

    preds_mat = np.column_stack(gbm_preds + [te_sn, chem_te, tab_te])
    full = np.clip(
        (w_mat * preds_mat).sum(1),
        np.quantile(ytr, 0.05),
        np.quantile(ytr, 0.98)
    )

    out = "submissions/nb1218_comph_knn_routing.csv"
    pd.DataFrame({"SMILES": te["smiles"], "Molecule Name": te["name"], "pEC50": full}).to_csv(out, index=False)
    print(f"saved {out}  (n={len(full)}, mean {full.mean():.3f}, range {full.min():.2f}-{full.max():.2f})", flush=True)

    json.dump({
        "rae": 0.4231,
        "via": "COMP-H k-NN local routing CheMeleon vs TabPFN (nb1216_gate, 3/3 seeds neg, delta=-0.0011)",
        "submission": out,
        "members": [c["arch"] for c in topK] + ["sisterNR_gnn", "chemeleon_knn_routed", "tabpfn_knn_routed"],
        "routing": f"k={K_NN} ECFP4 Tanimoto NN, local-MAE winner 2/14 weight, loser 1/14, normalize",
        "qm_feats": ACOLS, "strain_feats": SCOLS, "d4_feats": DCOLS, "dbstep_feats": DBCOLS
    }, open(f"{SD}/best_ensemble.json", "w"), indent=2)
    print("updated best_ensemble.json -> 0.4231 (COMP-H k-NN routing)", flush=True)


if __name__ == "__main__":
    main()
