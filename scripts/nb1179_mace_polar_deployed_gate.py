"""nb1179 - MACE-POLAR-1 polarisable-electrostatic QM scalars on the DEPLOYED ensemble config.

AIMNet2 QM scalars are DEPLOYED (nb1177, best 0.4297). MACE-OFF (a 2nd GROUND-STATE NNP) was
REDUNDANT (nb1178, +0.0001). MACE-POLAR-1 carries a DISTINCT observable axis -- explicit long-range
electrostatics + electron-density coefficients (not ground-state energy). This gates those scalars
ON TOP of the deployed config that already contains the AIMNet2 block.

  control   = 4-GBM(combined + 9 AIMNet2 scalars) + CheMeleon + TabPFN + sn_oof   (== deployed 0.4297)
  treatment = 4-GBM(combined + 9 AIMNet2 + 8 MACE-POLAR scalars) + CheMeleon + TabPFN + sn_oof
Matched on the same MTL ho_idx holdouts (nb1177/nb1178 convention). Deploy only if matched
delta < -0.001 AND treatment < deployed best (best_ensemble.json rae).
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
POLAR = "C:/pxr_work/mace_polar/mace_polar_features.csv"
ACOLS = ["aimnet_energy", "aimnet_qmin", "aimnet_qmax", "aimnet_qabs_mean", "aimnet_qstd",
         "aimnet_qsum_abs", "aimnet_dipole", "aimnet_fmax", "aimnet_frms"]
PCOLS = ["mp_estat", "mp_eelec", "mp_eint", "mp_dipole", "mp_dc_std", "mp_dc_absmn",
         "mp_spinabs", "mp_qstd"]


def topK_configs(archs):
    recs = [json.loads(l) for l in open(LOG) if l.strip()]
    valid = [r for r in recs if "ps_rae" in r and "error" not in r and r["arch"] in archs]
    valid.sort(key=lambda r: r["ps_rae"]); bp = {}
    for r in valid: bp.setdefault(r["arch"], r)
    return list(bp.values())


def aligned(path, cols, names):
    df = pd.read_csv(path)
    d = df[df.src == "train"].set_index("name").reindex(names)
    X = d[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    nan_rows = int(np.isnan(X).any(axis=1).sum())
    med = np.nanmedian(X, axis=0); inds = np.where(np.isnan(X)); X[inds] = np.take(med, inds[1])
    return X, nan_rows


def fit_pred(c, Xfull, ytr, use_idx, te_rows, Xadd=None):
    A, B = Xfull, Xfull[te_rows]
    if Xadd is not None:
        A = np.hstack([Xfull, Xadd]); B = np.hstack([Xfull[te_rows], Xadd[te_rows]])
    m = make_model(c["arch"], c["hp"])
    if c["arch"] in ("ridge", "enet"):
        sc = StandardScaler().fit(A[use_idx]); m.fit(sc.transform(A[use_idx]), ytr[use_idx]); return m.predict(sc.transform(B))
    m.fit(A[use_idx], ytr[use_idx]); return m.predict(B)


def main():
    d = np.load(CACHE); ytr = d["ytr"]; se = d["se"]; n_tr = len(ytr)
    Xtr, _ = feature_matrix(d, "combined")
    chem = np.load(f"{SD}/chemeleon_oof.npy"); tab = np.load(f"{SD}/tabpfn_oof.npy")
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    Xaim, na_aim = aligned(AIM, ACOLS, tr["name"])
    Xpol, na_pol = aligned(POLAR, PCOLS, tr["name"])
    print(f"aimnet imputed={na_aim}  polar imputed={na_pol}  (n_tr={n_tr})")

    topK = topK_configs(("lgbm", "xgb", "cat", "histgb"))
    print(f"deployed 4-GBM: {[r['arch'] for r in topK]} + CheMeleon + TabPFN + sn_oof + AIMNet2")
    c_raes, t_raes = [], []
    for seed in range(N_SEEDS):
        ho = np.load(f"{MTL}/ho_idx_seed{seed}.npy")
        trn = np.array([i for i in range(n_tr) if i not in set(ho.tolist())])
        noisy20 = se <= np.quantile(se, 0.8); noisy30 = se <= np.quantile(se, 0.7)
        lo, hi = np.quantile(ytr[trn], 0.05), np.quantile(ytr[trn], 0.98)
        sca = StandardScaler().fit(Xaim[trn]); Xa = sca.transform(Xaim)
        scp = StandardScaler().fit(Xpol[trn]); Xp = scp.transform(Xpol)
        Xctrl = Xa; Xtreat = np.hstack([Xa, Xp])
        gnn = np.load(f"{MTL}/sn_oof_seed{seed}.npy").ravel()
        gbm_c, gbm_t = [], []
        for c in topK:
            use = trn
            if c["prep"] == "noisy20": use = trn[noisy20[trn]]
            elif c["prep"] == "noisy30": use = trn[noisy30[trn]]
            gbm_c.append(fit_pred(c, Xtr, ytr, use, ho, Xadd=Xctrl))
            gbm_t.append(fit_pred(c, Xtr, ytr, use, ho, Xadd=Xtreat))
        ctrl = np.clip(np.mean(gbm_c + [chem[ho], tab[ho], gnn], 0), lo, hi)
        treat = np.clip(np.mean(gbm_t + [chem[ho], tab[ho], gnn], 0), lo, hi)
        c_raes.append(rae(ytr[ho], ctrl)); t_raes.append(rae(ytr[ho], treat))
        print(f"  seed{seed} ctrl {c_raes[-1]:.4f}  treat {t_raes[-1]:.4f}  d {t_raes[-1]-c_raes[-1]:+.4f}")

    cm, tm = float(np.mean(c_raes)), float(np.mean(t_raes)); delta = tm - cm
    prev = json.load(open(BEST))["rae"] if os.path.exists(BEST) else 0.4297
    print(f"\ncontrol  (deployed+AIMNet2)            RAE = {cm:.4f} +/- {np.std(c_raes):.4f}")
    print(f"treatment(deployed+AIMNet2+MACE-POLAR) RAE = {tm:.4f} +/- {np.std(t_raes):.4f}")
    print(f"MATCHED delta = {delta:+.4f}   deployed best = {prev}")
    deploy = bool(delta < -0.001 and tm < prev)
    out = {"control_rae": cm, "treatment_rae": tm, "matched_delta": delta,
           "c_raes": c_raes, "t_raes": t_raes, "aimnet_imputed": na_aim, "polar_imputed": na_pol,
           "deployed_best": prev, "deploy": deploy}
    json.dump(out, open(f"{P}/nb1179_summary.json", "w"), indent=2)
    print(f"\nsaved {P}/nb1179_summary.json -> DEPLOY={deploy}")


if __name__ == "__main__":
    main()
