"""COMBINATOR TICK nb1380 -- MACE-OFF learned-MLIP EMBEDDING as a residual corrector.

NON-STANDARD, untried (combinator_tried.md): the deployed SOAP chain proved that a
representation axis ORTHOGONAL-to + ABSENT-from all members works as a residual
corrector even when its SCALAR summary is "absorbed". We have MACE-OFF 256-d learned
equivariant node embeddings cached (maceoff_tr/te.npy) -- only ever tested as
(a) 7 absorbed scalar summaries and (b) a dilutive OOF flat member (COMP-S). The
256-d LEARNED embedding has NEVER been tested as a residual-corrector axis.

Diagnostics (seed0): maceoff-emb-PCA24 standalone Ridge RAE=0.70 (real signal);
mean|corr| vs SOAP-PCA = 0.07, max 0.47 -> MOSTLY ORTHOGONAL to handcrafted SOAP
(PMapper failed COMP-M precisely because it was the SAME local-geometry info as SOAP;
a learned MLIP embedding encodes QM-force-trained features SOAP's power-spectrum lacks).

COMP-X: corrector = MACEOFF-emb-PCA24 ALONE (replace SOAP).  Does the learned-MLIP
        axis carry the residual signal at all / vs handcrafted SOAP?
COMP-Y: corrector = [SOAP24 | MACEOFF-emb-PCA24] 48-d Ridge.  DEPLOY CANDIDATE --
        does MACE-OFF emb add ORTHOGONAL residual on TOP of SOAP? (analogous to
        COMP-M [SOAP|PMapper] but on a genuinely orthogonal learned axis).

Control = deployed best faithfully reproduced (nb1334 convention): 4 GBM(combined +
AIMNet2/strain/D4/DBSTEP/OrbMol) + CheMeleon-OOF + TabPFN-OOF + sisterNR-GNN, flat
mean, then SOAP PCA-24 Ridge(alpha=100) blend=0.5. SOAP corrector REFIT on the
stage-A member-mean residual per seed (matched comparison; corrector swapped per arm).
Gate: matched delta vs control per seed on ho_idx_seed{0,1,2}. a-priori COMP-Y is the
deploy candidate. Deploy iff delta < -0.001 AND treat_mean < 0.4149 AND >=2/3 neg.
"""
import os, sys, json, time
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb1126_combinatorial_search import feature_matrix, CACHE, make_model
from src.pxr.data import load_train
from src.pxr.eval import rae

P = "data/processed"; SD = "C:/pxr_work/search"; MTL = "C:/pxr_work/mtl"
BEST = f"{SD}/best_ensemble.json"; LOG = f"{SD}/results.jsonl"
N_SEEDS = 3; DEPLOYED_BEST = 0.4149; GATE = 0.001

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


def topK_configs():
    recs = [json.loads(l) for l in open(LOG) if l.strip()]
    valid = [r for r in recs if "ps_rae" in r and "error" not in r
             and r["arch"] in ("lgbm","xgb","cat","histgb")]
    valid.sort(key=lambda r: r["ps_rae"]); bp = {}
    for r in valid: bp.setdefault(r["arch"], r)
    return list(bp.values())


def fit_pred_qm(c, Xbase, Xqm, ytr, use_idx, te_rows):
    Xfull = np.hstack([Xbase, Xqm]); m = make_model(c["arch"], c["hp"])
    d = np.load(CACHE); se = d["se"]
    if c["prep"] == "noisy20": use_idx = use_idx[se[use_idx] <= np.quantile(se, 0.8)]
    elif c["prep"] == "noisy30": use_idx = use_idx[se[use_idx] <= np.quantile(se, 0.7)]
    m.fit(Xfull[use_idx], ytr[use_idx])
    return m.predict(Xfull[te_rows])


def ridge_correct(Xtrn, Xho, resid_trn, alpha=100.0):
    sc = StandardScaler().fit(Xtrn)
    r = Ridge(alpha=alpha).fit(sc.transform(Xtrn), resid_trn)
    return r.predict(sc.transform(Xho))


def main():
    from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
    t0 = time.time()
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    names_tr = tr["name"].tolist()
    d = np.load(CACHE); ytr = d["ytr"]; n_tr = len(ytr); assert n_tr == 4139, n_tr
    Xbase, _ = feature_matrix(d, "combined")

    Xqm_raw = aligned_block(AIM, ACOLS, names_tr); Xst_raw = aligned_block(STR, SCOLS, names_tr)
    Xd4_raw = aligned_block(D4, DCOLS, names_tr);  Xdb_raw = aligned_block(DB, DBCOLS, names_tr)
    Xorb_raw = aligned_block(ORB, OCOLS, names_tr)

    Xsoap = np.load("C:/pxr_work/soap/soap_train_matrix.npy"); assert Xsoap.shape == (n_tr, 24)

    # MACE-OFF 256-d learned embedding (median-impute the 2 NaN rows)
    M = np.load(f"{SD}/maceoff_tr.npy"); assert M.shape == (n_tr, 256), M.shape
    med = np.nanmedian(M, axis=0); ii = np.where(np.isnan(M)); M[ii] = np.take(med, ii[1])

    chem = np.load(f"{SD}/chemeleon_oof.npy"); tab = np.load(f"{SD}/tabpfn_oof.npy")
    gnn_full = np.load(f"{SD}/gnn_oof.npy")
    topK = topK_configs(); print(f"topK GBMs: {[c['arch'] for c in topK]}", flush=True)

    arms = ["CTRL_soap", "COMP-X_maceoff_only", "COMP-Y_soap+maceoff"]
    raes = {a: [] for a in arms}; ortho = []

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

        gbm_ho, gbm_trn = [], []
        for c in topK:
            gbm_ho.append(fit_pred_qm(c, Xbase, Xqm_full, ytr, trn, ho))
            gbm_trn.append(fit_pred_qm(c, Xbase, Xqm_full, ytr, trn, trn))

        # stage-A flat mean (deployed members)
        base_ho = gbm_ho + [chem[ho], tab[ho], sn_ho]
        base_trn = gbm_trn + [chem[trn], tab[trn], gnn_full[trn]]
        sa_ho = np.mean(base_ho, 0); sa_trn = np.mean(base_trn, 0)
        r_trn = ytr[trn] - np.clip(sa_trn, lo, hi)

        # MACE-OFF emb PCA-24 (leak-free: fit on trn)
        pca = PCA(24, random_state=0).fit(M[trn]); Zm = pca.transform(M)

        # corrector design matrices per arm
        soap_trn, soap_ho = Xsoap[trn], Xsoap[ho]
        mace_trn, mace_ho = Zm[trn], Zm[ho]
        comb_trn = np.hstack([soap_trn, mace_trn]); comb_ho = np.hstack([soap_ho, mace_ho])

        corr_soap = ridge_correct(soap_trn, soap_ho, r_trn)
        corr_mace = ridge_correct(mace_trn, mace_ho, r_trn)
        corr_comb = ridge_correct(comb_trn, comb_ho, r_trn)

        sa_ho_c = np.clip(sa_ho, lo, hi)
        raes["CTRL_soap"].append(rae(ytr[ho], np.clip(sa_ho_c + 0.5 * corr_soap, lo, hi)))
        raes["COMP-X_maceoff_only"].append(rae(ytr[ho], np.clip(sa_ho_c + 0.5 * corr_mace, lo, hi)))
        raes["COMP-Y_soap+maceoff"].append(rae(ytr[ho], np.clip(sa_ho_c + 0.5 * corr_comb, lo, hi)))

        # orthogonality diagnostic (mean |corr| maceoff-PCA vs soap)
        ss = StandardScaler().fit(np.hstack([soap_trn, mace_trn]))
        Zall = ss.transform(np.hstack([Xsoap, Zm]))
        cc = np.abs(np.corrcoef(Zall.T)[:24, 24:48]); ortho.append(float(cc.mean()))
        print(f"seed{seed} ctrl={raes['CTRL_soap'][-1]:.4f} X={raes['COMP-X_maceoff_only'][-1]:.4f} "
              f"Y={raes['COMP-Y_soap+maceoff'][-1]:.4f} | ortho_mean|corr|={ortho[-1]:.3f}", flush=True)

    cm = float(np.mean(raes["CTRL_soap"]))
    print(f"\ncontrol (SOAP-chain) mean RAE = {cm:.4f}  (deployed best = {DEPLOYED_BEST})", flush=True)
    out = {"control_mean_rae": cm, "ctrl_raes": raes["CTRL_soap"], "deployed_best": DEPLOYED_BEST,
           "ortho_mean_abscorr": float(np.mean(ortho)), "compositions": {}}
    deploy = []
    for a in ["COMP-X_maceoff_only", "COMP-Y_soap+maceoff"]:
        mn = float(np.mean(raes[a])); delta = mn - cm
        neg = sum(1 for i, r in enumerate(raes[a]) if r < raes["CTRL_soap"][i])
        ok = (delta < -GATE) and (mn < DEPLOYED_BEST) and (neg >= 2)
        out["compositions"][a] = {"mean_rae": mn, "delta": delta, "raes": raes[a],
                                  "neg_seeds": neg, "deploy": ok}
        print(f"{a}: RAE={mn:.4f} delta={delta:+.4f} {neg}/3 neg  deploy={ok}", flush=True)
        if ok: deploy.append(a)
    out["deploy_candidates"] = deploy
    out["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(out, open(f"{P}/nb1380_combinator_summary.json", "w"), indent=2)
    print(f"\nsaved {P}/nb1380_combinator_summary.json  ({out['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
