"""nb1295 - qmdesc (ML-DFT Fukui/NMR/bond-order reactivity) on the DEPLOYED
ensemble config (AIMNet2 + MMFF strain + DFT-D4, best 0.4252).

15 qmdesc scalars (Guan et al. ML-predicted DFT atomic descriptors from SMILES):
condensed Fukui indices f+ /f- (qmd_fe_*, qmd_fn_*) = FRONTIER-ORBITAL REACTIVITY
(electrophilic/nucleophilic susceptibility), per-atom NMR shielding moments
(qmd_nmr_*) = electronic-environment spread, bond-order moments (qmd_bo_*), and
DFT-fit partial-charge moments (qmd_q_*). DISTINCT axis vs the deployed QM blocks:
AIMNet2 = charges/energy/dipole/forces, D4 = polarizability/dispersion, strain =
conformational. Fukui/frontier-orbital reactivity + NMR shielding are NOT in any
deployed block -> genuine sink-escape candidate (reactivity/metabolic-lability is
mechanistically linked to PXR ligand turnover). SMILES-only, CPU, 0 NaN/4652.

  control   = 4-GBM + CheMeleon + TabPFN + sn_oof + AIMNet2 + strain + D4
  treatment = control + 15 qmdesc scalars
Deploy only if matched delta < -0.001 AND treatment < deployed best.
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
ACOLS = ["aimnet_energy","aimnet_qmin","aimnet_qmax","aimnet_qabs_mean","aimnet_qstd",
         "aimnet_qsum_abs","aimnet_dipole","aimnet_fmax","aimnet_frms"]
STR = "C:/pxr_work/strain/strain_features.csv"
SCOLS = ["strain_relax_mean","strain_relax_max","conf_espread","conf_erange",
         "conf_n","rmsd_mean","rmsd_max","e_per_heavy"]
D4 = "C:/pxr_work/d4/d4_features.csv"
DCOLS = ["d4_alpha_sum","d4_alpha_mean","d4_alpha_std","d4_alpha_max","d4_c6diag_mean",
         "d4_c6diag_std","d4_c6_total","d4_edisp","d4_edisp_per_atom","d4_cn_mean",
         "d4_cn_max","d4_qeeq_min","d4_qeeq_max","d4_qeeq_std","d4_qeeq_absum"]
QMD = "C:/pxr_work/qmdesc/qmdesc_features.csv"
QCOLS = ["qmd_q_min","qmd_q_max","qmd_q_absmean","qmd_fe_max","qmd_fe_min","qmd_fe_mean",
         "qmd_fe_std","qmd_fn_max","qmd_fn_std","qmd_nmr_min","qmd_nmr_max","qmd_nmr_std",
         "qmd_bo_min","qmd_bo_max","qmd_bo_mean"]


def topK_configs(archs):
    recs = [json.loads(l) for l in open(LOG) if l.strip()]
    valid = [r for r in recs if "ps_rae" in r and "error" not in r and r["arch"] in archs]
    valid.sort(key=lambda r: r["ps_rae"]); bp = {}
    for r in valid: bp.setdefault(r["arch"], r)
    return list(bp.values())


def fit_pred(c, Xfull, ytr, use_idx, te_rows, Xextra=None):
    A, B = Xfull, Xfull[te_rows]
    if Xextra is not None:
        A = np.hstack([Xfull, Xextra]); B = np.hstack([Xfull[te_rows], Xextra[te_rows]])
    m = make_model(c["arch"], c["hp"])
    if c["arch"] in ("ridge", "enet"):
        sc = StandardScaler().fit(A[use_idx]); m.fit(sc.transform(A[use_idx]), ytr[use_idx]); return m.predict(sc.transform(B))
    m.fit(A[use_idx], ytr[use_idx]); return m.predict(B)


def aligned_block(csv_path, cols, tr_names):
    df = pd.read_csv(csv_path)
    df = df[df.src == "train"].drop_duplicates(subset="name", keep="first")
    sub = df.set_index("name").reindex(tr_names)
    X = sub[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    nan_rows = int(np.isnan(X).any(axis=1).sum())
    med = np.nanmedian(X, axis=0); inds = np.where(np.isnan(X)); X[inds] = np.take(med, inds[1])
    return X, nan_rows


def main():
    d = np.load(CACHE); ytr = d["ytr"]; se = d["se"]; n_tr = len(ytr)
    Xtr, _ = feature_matrix(d, "combined")
    chem = np.load(f"{SD}/chemeleon_oof.npy"); tab = np.load(f"{SD}/tabpfn_oof.npy")

    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    Xqm, qn = aligned_block(AIM, ACOLS, tr["name"])
    Xst, sn = aligned_block(STR, SCOLS, tr["name"])
    Xd4, dn = aligned_block(D4, DCOLS, tr["name"])
    Xqd, qdn = aligned_block(QMD, QCOLS, tr["name"])
    print(f"imputed AIMNet2={qn} strain={sn} d4={dn} qmdesc={qdn}")

    topK = topK_configs(("lgbm", "xgb", "cat", "histgb"))
    print(f"deployed: {[r['arch'] for r in topK]} + CheMeleon + TabPFN + sn_oof + AIMNet2 + strain + D4")
    c_raes, t_raes, resid_corrs = [], [], []
    for seed in range(N_SEEDS):
        ho = np.load(f"{MTL}/ho_idx_seed{seed}.npy")
        trn = np.array([i for i in range(n_tr) if i not in set(ho.tolist())])
        noisy20 = se <= np.quantile(se, 0.8); noisy30 = se <= np.quantile(se, 0.7)
        lo, hi = np.quantile(ytr[trn], 0.05), np.quantile(ytr[trn], 0.98)
        def std(X):
            sc = StandardScaler().fit(X[trn]); return sc.transform(X)
        Xqm_s, Xst_s, Xd4_s, Xqd_s = std(Xqm), std(Xst), std(Xd4), std(Xqd)
        Xctrl = np.hstack([Xqm_s, Xst_s, Xd4_s])
        Xtreat = np.hstack([Xqm_s, Xst_s, Xd4_s, Xqd_s])
        gnn = np.load(f"{MTL}/sn_oof_seed{seed}.npy").ravel()
        gbm_c, gbm_t = [], []
        for c in topK:
            use = trn
            if c["prep"] == "noisy20": use = trn[noisy20[trn]]
            elif c["prep"] == "noisy30": use = trn[noisy30[trn]]
            gbm_c.append(fit_pred(c, Xtr, ytr, use, ho, Xextra=Xctrl))
            gbm_t.append(fit_pred(c, Xtr, ytr, use, ho, Xextra=Xtreat))
        ctrl = np.clip(np.mean(gbm_c + [chem[ho], tab[ho], gnn], 0), lo, hi)
        treat = np.clip(np.mean(gbm_t + [chem[ho], tab[ho], gnn], 0), lo, hi)
        c_raes.append(rae(ytr[ho], ctrl)); t_raes.append(rae(ytr[ho], treat))
        err = ytr[ho] - ctrl
        resid_corrs.append(float(np.corrcoef(np.abs(err), np.abs(Xqd_s[ho].mean(1)))[0, 1]))
        print(f"  seed{seed} ctrl {c_raes[-1]:.4f}  treat {t_raes[-1]:.4f}  d {t_raes[-1]-c_raes[-1]:+.4f}")

    cm, tm = float(np.mean(c_raes)), float(np.mean(t_raes)); delta = tm - cm
    prev = json.load(open(BEST))["rae"] if os.path.exists(BEST) else 0.4252
    print(f"\ncontrol  (deployed AIMNet2+strain+D4)   RAE = {cm:.4f} +/- {np.std(c_raes):.4f}")
    print(f"treatment(deployed + qmdesc reactivity) RAE = {tm:.4f} +/- {np.std(t_raes):.4f}")
    print(f"MATCHED delta = {delta:+.4f}   deployed best = {prev}")
    deploy = bool(delta < -0.001 and tm < prev)
    out = {"control_rae": cm, "treatment_rae": tm, "matched_delta": delta,
           "c_raes": c_raes, "t_raes": t_raes, "resid_corr_mean": float(np.mean(resid_corrs)),
           "qmdesc_imputed": qdn, "deployed_best": prev, "deploy": deploy}
    json.dump(out, open(f"{P}/nb1295_summary.json", "w"), indent=2)
    print(f"\nsaved {P}/nb1295_summary.json -> DEPLOY={deploy}")


if __name__ == "__main__":
    main()
