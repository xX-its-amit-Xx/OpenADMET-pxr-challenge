"""nb1320 — CYP3A4 triple-class classifier probabilities on DEPLOYED ensemble
config (AIMNet2 + MMFF strain + DFT-D4 + DBSTEP, best 0.4242).

3 CYP3A4 proba scalars: P(non-active), P(CYP3A4-inhibitor), P(CYP3A4-inducer)
from a LightGBM trained on 13817 CYP3A4-labeled compounds (JCIM 2025 acs.jcim.5c01192).
CYP3A4 induction is the DIRECT downstream readout of PXR activation (XREM enhancer
is PXR-driven). The inducer probability is mechanistically the most aligned cross-assay
feature we have. Coverage is low (med Tanimoto 0.253 for inducers) but corr-with-error +0.179.

  control   = 4-GBM(combined) + CheMeleon + TabPFN + sn_oof + AIMNet2 + strain + D4 + DBSTEP
  treatment = control + 3 CYP3A4 proba scalars
Deploy only if matched delta < -0.001 AND treatment < deployed best (0.4242).
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
CYP_PROBA = "C:/pxr_work/cyp3a4/cyp3a4_tr_proba.npy"


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
    med = np.nanmedian(X, axis=0); inds = np.where(np.isnan(X)); X[inds] = np.take(med, inds[1])
    return X


def main():
    d = np.load(CACHE); ytr = d["ytr"]; se = d["se"]; n_tr = len(ytr)
    Xtr, _ = feature_matrix(d, "combined")
    chem = np.load(f"{SD}/chemeleon_oof.npy"); tab = np.load(f"{SD}/tabpfn_oof.npy")

    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    Xqm = aligned_block(AIM, ACOLS, tr["name"])
    Xst = aligned_block(STR, SCOLS, tr["name"])
    Xd4 = aligned_block(D4, DCOLS, tr["name"])
    Xdb = aligned_block(DB, DBCOLS, tr["name"])
    # CYP3A4 proba: (4139, 3) 
    Xcyp = np.load(CYP_PROBA).astype(np.float64)
    print(f"Feature blocks: AIMNet2({Xqm.shape[1]}) strain({Xst.shape[1]}) D4({Xd4.shape[1]}) DBSTEP({Xdb.shape[1]}) CYP3A4({Xcyp.shape[1]})")
    print(f"CYP3A4 inducer proba range: {Xcyp[:,2].min():.3f}-{Xcyp[:,2].max():.3f} mean={Xcyp[:,2].mean():.3f}")

    topK = topK_configs(("lgbm", "xgb", "cat", "histgb"))
    print(f"deployed: {[r['arch'] for r in topK]} + CheMeleon + TabPFN + sn_oof + AIMNet2 + strain + D4 + DBSTEP")
    c_raes, t_raes, resid_corrs = [], [], []
    for seed in range(N_SEEDS):
        ho = np.load(f"{MTL}/ho_idx_seed{seed}.npy")
        trn = np.array([i for i in range(n_tr) if i not in set(ho.tolist())])
        noisy20 = se <= np.quantile(se, 0.8); noisy30 = se <= np.quantile(se, 0.7)
        lo, hi = np.quantile(ytr[trn], 0.05), np.quantile(ytr[trn], 0.98)
        scq = StandardScaler().fit(Xqm[trn]); Xqm_std = scq.transform(Xqm)
        scs = StandardScaler().fit(Xst[trn]); Xst_std = scs.transform(Xst)
        scd = StandardScaler().fit(Xd4[trn]); Xd4_std = scd.transform(Xd4)
        scb = StandardScaler().fit(Xdb[trn]); Xdb_std = scb.transform(Xdb)
        scc = StandardScaler().fit(Xcyp[trn]); Xcyp_std = scc.transform(Xcyp)
        # control: deployed (AIMNet2+strain+D4+DBSTEP)
        Xctrl = np.hstack([Xqm_std, Xst_std, Xd4_std, Xdb_std])
        # treatment: + CYP3A4 proba
        Xtreat = np.hstack([Xqm_std, Xst_std, Xd4_std, Xdb_std, Xcyp_std])
        gnn = np.load(f"{MTL}/sn_oof_seed{seed}.npy").ravel()
        cbm_c, cbm_t = [], []
        for c in topK:
            use = trn
            if c["prep"] == "noisy20": use = trn[noisy20[trn]]
            elif c["prep"] == "noisy30": use = trn[noisy30[trn]]
            cbm_c.append(fit_pred(c, np.hstack([Xtr, Xctrl]), ytr, use, ho))
            cbm_t.append(fit_pred(c, np.hstack([Xtr, Xtreat]), ytr, use, ho))
        ens_c = np.clip(np.mean(cbm_c + [chem[ho], tab[ho], gnn], 0), lo, hi)
        ens_t = np.clip(np.mean(cbm_t + [chem[ho], tab[ho], gnn], 0), lo, hi)
        rc, rt = rae(ytr[ho], ens_c), rae(ytr[ho], ens_t)
        resid = ytr[ho] - ens_c
        cyp_ho = Xcyp[ho, 2]
        corr = float(np.corrcoef(cyp_ho, np.abs(resid))[0, 1])
        c_raes.append(rc); t_raes.append(rt); resid_corrs.append(corr)
        print(f"  seed {seed}: control={rc:.4f} treatment={rt:.4f} delta={rt-rc:+.4f} corr-w-err={corr:.4f}")

    mc, mt = float(np.mean(c_raes)), float(np.mean(t_raes))
    delta = round(mt - mc, 4)
    print(f"\n  MATCHED: control={mc:.4f} treatment={mt:.4f} delta={delta:+.4f} (- is better)")
    print(f"  Mean corr-with-error: {np.mean(resid_corrs):.4f}")

    out = {"control_rae": mc, "treatment_rae": mt, "matched_delta": delta,
           "seed_raes": {"control": c_raes, "treatment": t_raes},
           "corr_with_error": float(np.mean(resid_corrs))}
    json.dump(out, open(f"{P}/nb1320_summary.json", "w"), indent=2)

    prev = json.load(open(BEST))["rae"] if os.path.exists(BEST) else 0.4242
    if delta < -0.001 and mt < min(mc, prev):
        print(f"\nDEPLOY-WORTHY: matched {delta:+.4f} < -0.001. Build submission nb1320.")
    else:
        print(f"\nno deploy: delta {delta:+.4f} (need <-0.001); best stays {prev}")


if __name__ == "__main__":
    main()
