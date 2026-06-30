"""nb2520_xtb_gate — honest gate for GFN2-xTB QM descriptors as an ADDED GBM feature block.

Physics validates like rich-z: marginal-over-anchor, NOT standalone RAE. Mirrors nb1168_gate
(same ho_idx_seed{0,1,2} train holdouts, same topK GBM configs, same fixed CheMeleon/TabPFN/GNN
members) but the ONLY thing that changes between arms is the GBM feature matrix:
  control   = combined  (deployed)
  xtb       = combined + 9 GFN2-xTB QM descriptors (standardized)
GNN member held FIXED at the deployed sisterNR head (sn_oof). Deploy iff xtb < control - 0.001
AND xtb < deployed best (0.4367). Also reports corr(xtb-only-residual, nb3200 error).
"""
import os, sys, json, glob
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb1126_combinatorial_search import feature_matrix, CACHE, make_model, StandardScaler
from src.pxr.data import load_train
from src.pxr.eval import rae
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

P = "data/processed"; SD = "C:/pxr_work/search"; MTL = "C:/pxr_work/mtl"; XTB = "C:/pxr_work/xtb"
BEST = f"{SD}/best_ensemble.json"; LOG = f"{SD}/results.jsonl"; N_SEEDS = 3
XCOLS = ["xtb_energy", "xtb_homo", "xtb_lumo", "xtb_gap", "xtb_dipole",
         "xtb_qmin", "xtb_qmax", "xtb_qabs_mean", "xtb_qstd"]


def load_xtb_train():
    fs = glob.glob(f"{XTB}/raw/out_*.csv")
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).drop_duplicates("name")
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    m = tr[["name"]].merge(df[["name"] + XCOLS], on="name", how="left")
    X = m[XCOLS].values.astype(np.float32)
    # column median impute (no NaN expected for the 9 xtb cols, but be safe)
    med = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(med, inds[1])
    return X


def topK_configs(archs):
    recs = [json.loads(l) for l in open(LOG) if l.strip()]
    valid = [r for r in recs if "ps_rae" in r and "error" not in r and r["arch"] in archs]
    valid.sort(key=lambda r: r["ps_rae"]); bp = {}
    for r in valid: bp.setdefault(r["arch"], r)
    return list(bp.values())


def fit_one(c, Xtr, ytr, use_idx, Xho):
    m = make_model(c["arch"], c["hp"])
    if c["arch"] in ("ridge", "enet"):
        sc = StandardScaler().fit(Xtr[use_idx]); m.fit(sc.transform(Xtr[use_idx]), ytr[use_idx]); return m.predict(sc.transform(Xho))
    m.fit(Xtr[use_idx], ytr[use_idx]); return m.predict(Xho)


def run_config(archs, label, feats, ytr, se, n_tr, chem, tab):
    topK = topK_configs(archs)
    res = {"control": [], "xtb": []}
    for seed in range(N_SEEDS):
        ho = np.load(f"{MTL}/ho_idx_seed{seed}.npy")
        trn = np.array([i for i in range(n_tr) if i not in set(ho.tolist())])
        noisy20 = se <= np.quantile(se, 0.8); noisy30 = se <= np.quantile(se, 0.7)
        lo, hi = np.quantile(ytr[trn], 0.05), np.quantile(ytr[trn], 0.98)
        gnn = np.load(f"{MTL}/sn_oof_seed{seed}.npy").ravel()
        for arm, Xtr in feats.items():
            Xho = Xtr[ho]; gbm = []
            for c in topK:
                use = trn
                if c["prep"] == "noisy20": use = trn[noisy20[trn]]
                elif c["prep"] == "noisy30": use = trn[noisy30[trn]]
                gbm.append(fit_one(c, Xtr, ytr, use, Xho))
            ens = np.clip(np.mean(gbm + [chem[ho], tab[ho], gnn], 0), lo, hi)
            res[arm].append(rae(ytr[ho], ens))
    means = {k: round(float(np.mean(v)), 4) for k, v in res.items()}
    stds = {k: round(float(np.std(v)), 4) for k, v in res.items()}
    delta = means["xtb"] - means["control"]
    print(f"[{label}] {[r['arch'] for r in topK]} + CheMeleon + TabPFN + GNN")
    for k in res: print(f"    {k:8s} {means[k]:.4f} +/- {stds[k]:.4f}")
    print(f"    xtb - control = {delta:+.4f}  (- is better)")
    return means, stds, delta


def corr_with_anchor_error(comb, xtb_X, ytr, se, n_tr):
    """xTB-only GBM residual contribution vs nb3200 error, on the union of holdouts."""
    import lightgbm as lgb
    d = np.load(CACHE)
    # anchor (nb3200) is over the 253 unblind set, not the 4139 train -> use train-internal proxy:
    # corr( xtb-only holdout prediction error , combined-only holdout prediction error ) low = orthogonal info.
    errs_x, errs_c = [], []
    for seed in range(N_SEEDS):
        ho = np.load(f"{MTL}/ho_idx_seed{seed}.npy")
        trn = np.array([i for i in range(n_tr) if i not in set(ho.tolist())])
        mx = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, n_jobs=4, verbose=-1)
        mx.fit(xtb_X[trn], ytr[trn]); ex = ytr[ho] - mx.predict(xtb_X[ho])
        mc = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, n_jobs=4, verbose=-1)
        mc.fit(comb[trn], ytr[trn]); ec = ytr[ho] - mc.predict(comb[ho])
        errs_x.append(ex); errs_c.append(ec)
    ex = np.concatenate(errs_x); ec = np.concatenate(errs_c)
    return float(np.corrcoef(ex, ec)[0, 1])


def main():
    d = np.load(CACHE); ytr = d["ytr"]; se = d["se"]; n_tr = len(ytr)
    comb, _ = feature_matrix(d, "combined")
    chem = np.load(f"{SD}/chemeleon_oof.npy"); tab = np.load(f"{SD}/tabpfn_oof.npy")

    xtb_raw = load_xtb_train()
    np.save(f"{XTB}/xtb_train_9feat.npy", xtb_raw)
    sc = StandardScaler().fit(xtb_raw); xtb_z = sc.transform(xtb_raw).astype(np.float32)
    comb_xtb = np.hstack([comb, xtb_z]).astype(np.float32)
    feats = {"control": comb, "xtb": comb_xtb}
    print(f"combined dims={comb.shape[1]}  +xtb -> {comb_xtb.shape[1]} (9 GFN2-xTB QM descriptors)")

    m5, s5, d5 = run_config(("lgbm", "xgb", "cat", "histgb", "mlp"), "5-arch", feats, ytr, se, n_tr, chem, tab)
    m4, s4, d4 = run_config(("lgbm", "xgb", "cat", "histgb"), "4-GBM(deployed)", feats, ytr, se, n_tr, chem, tab)
    corr = corr_with_anchor_error(comb, xtb_z, ytr, se, n_tr)
    print(f"corr(xtb-only err, combined-only err) over holdouts = {corr:+.3f}  (lower = more orthogonal physics info)")

    out = {"5arch": {"means": m5, "std": s5, "xtb_minus_control": round(d5, 4)},
           "4gbm_deployed": {"means": m4, "std": s4, "xtb_minus_control": round(d4, 4)},
           "corr_xtb_vs_combined_err": round(corr, 4), "n_feat": 9, "morfeus": "ALL-NaN (failed)"}
    json.dump(out, open(f"{P}/nb2522_xtb_gate_summary.json", "w"), indent=2)

    prev = json.load(open(BEST))["rae"] if os.path.exists(BEST) else 0.4367
    if d4 < -0.001 and m4["xtb"] < prev:
        print(f"\nDEPLOY-WORTHY: 4-GBM matched xtb-control {d4:+.4f} (<-0.001) AND xtb {m4['xtb']:.4f} < best {prev}.")
    else:
        print(f"\nno deploy: 4-GBM matched delta {d4:+.4f} (need <-0.001); best stays {prev}")


if __name__ == "__main__":
    main()
