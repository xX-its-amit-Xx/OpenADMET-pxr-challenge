"""COMBINATOR TICK nb1410 -- single-conc MEASURED-ASSAY residual corrector (2 forms).

Deployed best = 0.4149: 7-member flat mean {4 GBM(combined+AIMNet2/strain/D4/DBSTEP/OrbMol),
CheMeleon-OOF, TabPFN-OOF, sisterNR-GNN-OOF} -> [SOAP24|PMapper24] Ridge(alpha=100) residual
corrector, blend 0.5, clipped.

EVERY corrector axis tried so far has been a STRUCTURAL descriptor (SOAP/PMapper geometry,
MACE-OFF/CheMeleon embeddings, boltz-z, ErG).  The single-concentration PXR screen (log2FC at
one dose + FDR) is a genuinely DIFFERENT KIND of axis -- an independent MEASUREMENT, never sees
pEC50, leak-free by construction (exactly what we have for the 513 test compounds too).  It won
on the LB-253 path (feedback_singleconc_pactive_WIN, -0.0178) but was NEVER tested on the honest
train-holdout gate.  Diagnostic (vs 6-member proxy base): corr(-log10fdr, resid)=+0.44 and the
156 low-log2fc TRUE-INACTIVE compounds are over-predicted by -0.935 log -- precisely the cy303
inactive-tail weak spot, MEASURED directly.  Test whether the FULL deployed ensemble+SOAP|PMapper
still leaves single-conc residual signal.

COMP-AD  AUGMENTED GLOBAL CORRECTOR (direct analog of how PMapper was added to SOAP, COMP-M):
  corrector features = [SOAP24 | PMapper24 | sc3], sc3 = [log2fc, -log10(fdr), covered_ind]
  imputed (trn-covered-median) + standardized.  ONE leak-free Ridge(alpha=100), blend 0.5.
  Parameter-free: Ridge learns the single-conc weight; if absorbed -> ~0 (neutral).

COMP-AE  COVERED-ONLY PARALLEL CORRECTOR (isolate single-conc to measured compounds):
  total_corr = blend*(ridge_SOAP|PMAP(C_ho) + sc_corr_ho).  sc_corr = Ridge(alpha=100) fit on
  COVERED trn rows mapping [log2fc, -log10(fdr)] -> base-mean residual; applied to ho ONLY where
  the compound has a single-conc measurement (0 elsewhere).  2nd corrector axis = the assay
  measurement instead of PMapper geometry.

Control = deployed best faithfully reproduced w/ leak-free INNER-OOF GBM trn preds (matched).
Gate: matched delta vs control per seed on ho_idx_seed{0,1,2}.
Deploy iff delta < -0.001 AND treat_mean < 0.4149 AND >=2/3 neg.
"""
import os, sys, json, time
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb1126_combinatorial_search import feature_matrix, CACHE, make_model
from src.pxr.data import load_train
from src.pxr.eval import rae

P = "data/processed"; SD = "C:/pxr_work/search"; MTL = "C:/pxr_work/mtl"
LOG = f"{SD}/results.jsonl"
N_SEEDS = 3; DEPLOYED_BEST = 0.4149; GATE = 0.001; ALPHA = 100.0; BLEND = 0.5
INNER_K = 3
SC_AGG = "C:/pxr_work/fm_build/singleconc_agg.csv"

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
DB = "C:/pxr_work/dbstep/dbstep_features.csv"
DBCOLS = ["vbur_r25","vbur_r35","vbur_r45","vbur_r55","vbur_r65","ster_L","ster_Bmin",
          "ster_Bmax","ster_aniso","npr1","npr2","asphericity","spherocity",
          "eccentricity","radgyr","inertial_sf"]
ORB = "C:/pxr_work/orbmol/orbmol_features.csv"
OCOLS = ["orb_energy","orb_energy_per_ha","orb_fmax","orb_frms","orb_fstd",
         "orb_conf_mean","orb_conf_std","orb_conf_node_mean","orb_conf_node_std",
         "orb_conf_node_min","orb_node_emb_mean","orb_node_emb_std","orb_node_emb_norm"]


def aligned_block(csv_path, cols, tr_names):
    df = pd.read_csv(csv_path)
    df = df[df.src == "train"].drop_duplicates(subset="name", keep="first")
    sub = df.set_index("name").reindex(tr_names)
    X = sub[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    med = np.nanmedian(X, axis=0); ii = np.where(np.isnan(X)); X[ii] = np.take(med, ii[1])
    return X


def singleconc_raw(tr_smiles):
    """Raw single-conc measurement per train compound (matched by raw SMILES). LEAK-FREE:
    independent assay, never sees pEC50. Returns log2fc, neglog10fdr, covered mask (NaN where
    not measured)."""
    sc = pd.read_csv(SC_AGG).drop_duplicates("smiles").set_index("smiles")
    log2fc = np.full(len(tr_smiles), np.nan); fdr = np.full(len(tr_smiles), np.nan)
    for i, s in enumerate(tr_smiles):
        if s in sc.index:
            log2fc[i] = sc.loc[s, "max_log2fc"]; fdr[i] = sc.loc[s, "min_fdr"]
    cov = ~np.isnan(log2fc)
    neglogfdr = -np.log10(np.clip(fdr, 1e-12, None))
    return log2fc, neglogfdr, cov


def topK_configs():
    recs = [json.loads(l) for l in open(LOG) if l.strip()]
    valid = [r for r in recs if "ps_rae" in r and "error" not in r
             and r["arch"] in ("lgbm","xgb","cat","histgb")]
    valid.sort(key=lambda r: r["ps_rae"]); bp = {}
    for r in valid: bp.setdefault(r["arch"], r)
    return list(bp.values())


def prep_filter(c, idx, se):
    if c["prep"] == "noisy20": return idx[se[idx] <= np.quantile(se, 0.8)]
    if c["prep"] == "noisy30": return idx[se[idx] <= np.quantile(se, 0.7)]
    return idx


def fit_pred(c, Xfull, y, fit_idx, pred_idx, se):
    m = make_model(c["arch"], c["hp"])
    fi = prep_filter(c, fit_idx, se)
    m.fit(Xfull[fi], y[fi])
    return m.predict(Xfull[pred_idx])


def gbm_label_oof(topK, Xfull, y, trn, ho, se, seed):
    kf = KFold(INNER_K, shuffle=True, random_state=1000 + seed)
    oof_trn = [np.zeros(len(trn)) for _ in topK]; ho_pred = []
    for ci, c in enumerate(topK):
        for itr, iva in kf.split(trn):
            oof_trn[ci][iva] = fit_pred(c, Xfull, y, trn[itr], trn[iva], se)
        ho_pred.append(fit_pred(c, Xfull, y, trn, ho, se))
    return oof_trn, ho_pred


def ridge_fit(Xtrn, resid_trn, alpha=ALPHA):
    sc = StandardScaler().fit(Xtrn)
    r = Ridge(alpha=alpha).fit(sc.transform(Xtrn), resid_trn)
    return sc, r


def ridge_pred(sc, r, X):
    return r.predict(sc.transform(X))


def main():
    from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
    t0 = time.time()
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    names_tr = tr["name"].tolist(); smiles_tr = tr["smiles"].tolist()
    d = np.load(CACHE); ytr = d["ytr"]; se = d["se"]; n_tr = len(ytr); assert n_tr == 4139, n_tr
    Xbase, _ = feature_matrix(d, "combined")

    Xqm_raw = aligned_block(AIM, ACOLS, names_tr); Xst_raw = aligned_block(STR, SCOLS, names_tr)
    Xd4_raw = aligned_block(D4, DCOLS, names_tr);  Xdb_raw = aligned_block(DB, DBCOLS, names_tr)
    Xorb_raw = aligned_block(ORB, OCOLS, names_tr)

    Xsoap = np.load("C:/pxr_work/soap/soap_train_matrix.npy"); assert Xsoap.shape == (n_tr, 24)
    PMAP = np.load("C:/pxr_work/pmapper_feats.npz", allow_pickle=True)["train"].astype(float)
    assert PMAP.shape == (n_tr, 2048), PMAP.shape

    log2fc, neglogfdr, cov = singleconc_raw(smiles_tr)
    print(f"single-conc coverage: {cov.sum()}/{n_tr} = {cov.mean():.3f}", flush=True)

    chem = np.load(f"{SD}/chemeleon_oof.npy"); tab = np.load(f"{SD}/tabpfn_oof.npy")
    gnn_full = np.load(f"{SD}/gnn_oof.npy")
    topK = topK_configs(); print(f"topK GBMs: {[c['arch'] for c in topK]}", flush=True)

    arms = ["CTRL", "COMP-AD_aug_global", "COMP-AE_covered_parallel"]
    raes = {a: [] for a in arms}

    for seed in range(N_SEEDS):
        ho = np.load(f"{MTL}/ho_idx_seed{seed}.npy"); ho_set = set(ho.tolist())
        trn = np.array([i for i in range(n_tr) if i not in ho_set])
        lo, hi = np.quantile(ytr[trn], 0.05), np.quantile(ytr[trn], 0.98)
        sn_ho = np.load(f"{MTL}/sn_oof_seed{seed}.npy").ravel()

        sc_aim = StandardScaler().fit(Xqm_raw[trn]); sc_str = StandardScaler().fit(Xst_raw[trn])
        sc_d4 = StandardScaler().fit(Xd4_raw[trn]);  sc_db = StandardScaler().fit(Xdb_raw[trn])
        sc_orb = StandardScaler().fit(Xorb_raw[trn])
        Xqm_full = np.hstack([sc_aim.transform(Xqm_raw), sc_str.transform(Xst_raw),
                              sc_d4.transform(Xd4_raw), sc_db.transform(Xdb_raw),
                              sc_orb.transform(Xorb_raw)])
        Xfull = np.hstack([Xbase, Xqm_full])

        glab_trn, glab_ho = gbm_label_oof(topK, Xfull, ytr, trn, ho, se, seed)

        # ===== deployed base: flat mean 4 GBM + 3 foundation, then base residual =====
        sa_trn = np.mean(glab_trn + [chem[trn], tab[trn], gnn_full[trn]], 0)
        sa_ho = np.mean(glab_ho + [chem[ho], tab[ho], sn_ho], 0)
        r_trn = ytr[trn] - np.clip(sa_trn, lo, hi)
        sa_ho_c = np.clip(sa_ho, lo, hi)

        # corrector features (deployed): SOAP24 + PMapper PCA-24 (leak-free)
        pmap_pca = PCA(24, random_state=0).fit(PMAP[trn]); Zp = pmap_pca.transform(PMAP)
        C_trn = np.hstack([Xsoap[trn], Zp[trn]]); C_ho = np.hstack([Xsoap[ho], Zp[ho]])

        # ----- CONTROL -----
        scC, rC = ridge_fit(C_trn, r_trn)
        ctrl_pred = np.clip(sa_ho_c + BLEND * ridge_pred(scC, rC, C_ho), lo, hi)
        raes["CTRL"].append(rae(ytr[ho], ctrl_pred))

        # single-conc feature block (impute on trn-covered median, standardize, + covered ind)
        cov_trn = cov[trn]
        l_med = np.median(log2fc[trn][cov_trn]); f_med = np.median(neglogfdr[trn][cov_trn])
        l_imp = np.where(cov, log2fc, np.nan); f_imp = np.where(cov, neglogfdr, np.nan)
        l_imp = np.where(np.isnan(l_imp), l_med, l_imp); f_imp = np.where(np.isnan(f_imp), f_med, f_imp)
        SCf = np.column_stack([l_imp, f_imp, cov.astype(float)])

        # ----- COMP-AD: augmented global corrector [SOAP|PMAP|sc3] -----
        Cad_trn = np.hstack([C_trn, SCf[trn]]); Cad_ho = np.hstack([C_ho, SCf[ho]])
        scAD, rAD = ridge_fit(Cad_trn, r_trn)
        ad_pred = np.clip(sa_ho_c + BLEND * ridge_pred(scAD, rAD, Cad_ho), lo, hi)
        raes["COMP-AD_aug_global"].append(rae(ytr[ho], ad_pred))

        # ----- COMP-AE: covered-only parallel single-conc corrector -----
        SCm_trn = np.column_stack([log2fc[trn], neglogfdr[trn]])  # raw (NaN where uncovered)
        ct = cov_trn
        scSC, rSC = ridge_fit(SCm_trn[ct], r_trn[ct])
        sc_corr_ho = np.zeros(len(ho))
        cov_ho = cov[ho]
        if cov_ho.any():
            SCm_ho = np.column_stack([log2fc[ho], neglogfdr[ho]])
            sc_corr_ho[cov_ho] = ridge_pred(scSC, rSC, SCm_ho[cov_ho])
        ae_pred = np.clip(sa_ho_c + BLEND * (ridge_pred(scC, rC, C_ho) + sc_corr_ho), lo, hi)
        raes["COMP-AE_covered_parallel"].append(rae(ytr[ho], ae_pred))

        print(f"seed{seed} ctrl={raes['CTRL'][-1]:.4f} "
              f"AD={raes['COMP-AD_aug_global'][-1]:.4f} "
              f"AE={raes['COMP-AE_covered_parallel'][-1]:.4f} "
              f"(cov_ho={int(cov_ho.sum())}/{len(ho)})", flush=True)

    cm = float(np.mean(raes["CTRL"]))
    print(f"\ncontrol mean RAE = {cm:.4f}  (deployed best = {DEPLOYED_BEST})", flush=True)
    out = {"control_mean_rae": cm, "ctrl_raes": raes["CTRL"], "deployed_best": DEPLOYED_BEST,
           "sc_coverage": float(cov.mean()), "compositions": {}}
    deploy = []
    for a in ["COMP-AD_aug_global", "COMP-AE_covered_parallel"]:
        mn = float(np.mean(raes[a])); delta = mn - cm
        neg = sum(1 for i, r in enumerate(raes[a]) if r < raes["CTRL"][i])
        ok = (delta < -GATE) and (mn < DEPLOYED_BEST) and (neg >= 2)
        out["compositions"][a] = {"mean_rae": mn, "delta": delta, "raes": raes[a],
                                  "neg_seeds": neg, "deploy": ok}
        print(f"{a}: RAE={mn:.4f} delta={delta:+.4f} {neg}/3 neg  deploy={ok}", flush=True)
        if ok: deploy.append(a)
    out["deploy_candidates"] = deploy
    out["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(out, open(f"{P}/nb1410_combinator_summary.json", "w"), indent=2)
    print(f"\nsaved {P}/nb1410_combinator_summary.json  ({out['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
