"""nb1177 — AIMNet2 QM scalars on the DEPLOYED ensemble config (real deploy bar).

nb1174 gated AIMNet2 on a WEAK anchor (5-arch GBM + chemprop_aux GNN, ctrl 0.4596) and got a
robust matched -0.0081. But the deployed best (0.4367) has CheMeleon + TabPFN + the deployed
sisterNR GNN (sn_oof) — much stronger, likely to absorb the signal (cy290/cy292 lesson). This
re-gates AIMNet2 on the EXACT deployed config, matched on the same MTL ho_idx holdouts as
nb1168 so the control reproduces the deployed ~0.4367.

  control   = 4-GBM(combined) + CheMeleon + TabPFN + sn_oof
  treatment = 4-GBM(combined + 9 AIMNet2 QM scalars) + CheMeleon + TabPFN + sn_oof
Deploy only if matched delta < -0.001 AND treatment < 0.4367 (deployed best).
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb1126_combinatorial_search import feature_matrix, CACHE, make_model
from src.pxr.data import load_train
from src.pxr.eval import rae
from sklearn.preprocessing import StandardScaler
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

P = "data/processed"; SD = "C:/pxr_work/search"; MTL = "C:/pxr_work/mtl"
BEST = f"{SD}/best_ensemble.json"; LOG = f"{SD}/results.jsonl"; N_SEEDS = 3
AIM = "C:/pxr_work/aimnet2/aimnet_features.csv"
ACOLS = ["aimnet_energy", "aimnet_qmin", "aimnet_qmax", "aimnet_qabs_mean", "aimnet_qstd",
         "aimnet_qsum_abs", "aimnet_dipole", "aimnet_fmax", "aimnet_frms"]


def topK_configs(archs):
    recs = [json.loads(l) for l in open(LOG) if l.strip()]
    valid = [r for r in recs if "ps_rae" in r and "error" not in r and r["arch"] in archs]
    valid.sort(key=lambda r: r["ps_rae"]); bp = {}
    for r in valid: bp.setdefault(r["arch"], r)
    return list(bp.values())


def fit_pred(c, Xfull, ytr, use_idx, te_rows, Xqm=None):
    A, B = Xfull, Xfull[te_rows]
    if Xqm is not None:
        A = np.hstack([Xfull, Xqm]); B = np.hstack([Xfull[te_rows], Xqm[te_rows]])
    m = make_model(c["arch"], c["hp"])
    if c["arch"] in ("ridge", "enet"):
        sc = StandardScaler().fit(A[use_idx]); m.fit(sc.transform(A[use_idx]), ytr[use_idx]); return m.predict(sc.transform(B))
    m.fit(A[use_idx], ytr[use_idx]); return m.predict(B)


def main():
    d = np.load(CACHE); ytr = d["ytr"]; se = d["se"]; n_tr = len(ytr)
    Xtr, _ = feature_matrix(d, "combined")
    chem = np.load(f"{SD}/chemeleon_oof.npy"); tab = np.load(f"{SD}/tabpfn_oof.npy")

    # AIMNet2 aligned to 4139 train order, median-impute errored rows
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    adf = pd.read_csv(AIM)
    atr = adf[adf.src == "train"].set_index("name").reindex(tr["name"])
    Xqm = atr[ACOLS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    nan_rows = int(np.isnan(Xqm).any(axis=1).sum())
    med = np.nanmedian(Xqm, axis=0); inds = np.where(np.isnan(Xqm)); Xqm[inds] = np.take(med, inds[1])
    print(f"AIMNet2 train rows={len(atr)} errored/imputed={nan_rows}")

    topK = topK_configs(("lgbm", "xgb", "cat", "histgb"))
    print(f"deployed 4-GBM: {[r['arch'] for r in topK]} + CheMeleon + TabPFN + sn_oof")
    c_raes, t_raes = [], []
    for seed in range(N_SEEDS):
        ho = np.load(f"{MTL}/ho_idx_seed{seed}.npy")
        trn = np.array([i for i in range(n_tr) if i not in set(ho.tolist())])
        noisy20 = se <= np.quantile(se, 0.8); noisy30 = se <= np.quantile(se, 0.7)
        lo, hi = np.quantile(ytr[trn], 0.05), np.quantile(ytr[trn], 0.98)
        # per-fold standardize QM on train rows
        scq = StandardScaler().fit(Xqm[trn]); Xqm_std = scq.transform(Xqm)
        gnn = np.load(f"{MTL}/sn_oof_seed{seed}.npy").ravel()
        gbm_c, gbm_t = [], []
        for c in topK:
            use = trn
            if c["prep"] == "noisy20": use = trn[noisy20[trn]]
            elif c["prep"] == "noisy30": use = trn[noisy30[trn]]
            gbm_c.append(fit_pred(c, Xtr, ytr, use, ho))
            gbm_t.append(fit_pred(c, Xtr, ytr, use, ho, Xqm=Xqm_std))
        ctrl = np.clip(np.mean(gbm_c + [chem[ho], tab[ho], gnn], 0), lo, hi)
        treat = np.clip(np.mean(gbm_t + [chem[ho], tab[ho], gnn], 0), lo, hi)
        c_raes.append(rae(ytr[ho], ctrl)); t_raes.append(rae(ytr[ho], treat))
        print(f"  seed{seed} ctrl {c_raes[-1]:.4f}  treat {t_raes[-1]:.4f}  d {t_raes[-1]-c_raes[-1]:+.4f}")

    cm, tm = float(np.mean(c_raes)), float(np.mean(t_raes)); delta = tm - cm
    prev = json.load(open(BEST))["rae"] if os.path.exists(BEST) else 0.4367
    print(f"\ncontrol  (deployed)            RAE = {cm:.4f} +/- {np.std(c_raes):.4f}")
    print(f"treatment(deployed+AIMNet2)    RAE = {tm:.4f} +/- {np.std(t_raes):.4f}")
    print(f"MATCHED delta = {delta:+.4f}   deployed best = {prev}")
    deploy = bool(delta < -0.001 and tm < prev)
    out = {"control_rae": cm, "treatment_rae": tm, "matched_delta": delta,
           "c_raes": c_raes, "t_raes": t_raes, "errored_imputed": nan_rows,
           "deployed_best": prev, "deploy": deploy}
    json.dump(out, open(f"{P}/nb1177_summary.json", "w"), indent=2)
    print(f"\nsaved {P}/nb1177_summary.json -> DEPLOY={deploy}")


if __name__ == "__main__":
    main()
