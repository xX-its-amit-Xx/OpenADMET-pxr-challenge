"""COMBINATOR TICK nb1430 -- DECOUPLE richz's shrinkage from the weak geometry axes.

DEPLOYED best = 0.4133 (best_ensemble.json, nb1400 COMP-Z2): 7-member flat mean
{4 GBM(combined+AIMNet2/strain/D4/DBSTEP/OrbMol), CheMeleon-OOF, TabPFN-OOF, sisterNR-GNN-OOF}
-> ONE Ridge(alpha=100) on the 72-d corrector [SOAP24 | PMapper-PCA24 | richz-PCA24], blend 0.5.

UNTESTED FLAW: in the deployed JOINT 72-d Ridge with a SHARED alpha=100, the richz block carries a
strong activity-aligned PC (max|corr-w-y|=0.576) while SOAP/PMapper top out at ~0.275.  A single L2
penalty shrinks the strong-signal richz coefficients the SAME as the 2x-weaker geometry dims -> richz
is OVER-shrunk.  combinator_tried never decoupled the per-block regularization (COMP-N varied SOAP
dims; every chain was single-axis or one shared-alpha joint).  Two mechanisms, same hypothesis:

COMP-AH  BLOCK-REWEIGHTED JOINT Ridge (my primary, ONE robust joint fit):
  standardize the 72-d corrector, then multiply ONLY the richz-24 columns by g before a single
  Ridge(alpha=100).  Scaling a feature block UP by g reduces its effective L2 penalty by g^2
  (effective alpha_richz = 100/g^2) WITHOUT a separate fit -> richz less shrunk, geometry unchanged.
  g in {1.5, 2.5, 4.0}.  Cleaner/more robust than a boosted 2-stage chain (no residual double-fit).

COMP-AI  SEQUENTIAL BOOSTED per-axis-alpha corrector (mechanism check):
  Stage1 Ridge(alpha=100) on [SOAP24|PMapper24] -> correction c1, residual r1.  Stage2 Ridge(alpha_z)
  on richz24 -> fit r1, correction c2.  total = c1 + c2 @ blend 0.5.  alpha_z in {30, 100}: the
  strong richz axis gets its OWN (smaller) shrinkage.  Confirms whether per-axis decoupling helps at
  all, independent of the block-reweight mechanism.

richz is 100% test-covered (leak-free, deployable -- unlike nb1410 single-conc which is 0/513 on test).
Control = deployed [SOAP24|PMapper24|richz24] joint Ridge faithfully reproduced w/ leak-free INNER-OOF
GBM trn preds (matched).  richz PCA fit per-seed on trn-real (nonzero) rows only; non-real -> zero.
Gate: matched delta vs control per seed on ho_idx_seed{0,1,2}.
Deploy iff delta < -0.001 AND treat_mean < 0.4133 AND >=2/3 neg.
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
N_SEEDS = 3; DEPLOYED_BEST = 0.4133; GATE = 0.001; ALPHA = 100.0; BLEND = 0.5
INNER_K = 3
RICHZ_TR = "C:/pxr_struct/boltz/boltz_z_rich_train.npy"

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


def richz_pca(richz_raw, trn, d):
    real = np.abs(richz_raw).sum(1) > 0
    fit_idx = trn[real[trn]]
    scz = StandardScaler().fit(richz_raw[fit_idx])
    pz = PCA(d, random_state=0).fit(scz.transform(richz_raw[fit_idx]))
    Z = np.zeros((len(richz_raw), d))
    Z[real] = pz.transform(scz.transform(richz_raw[real]))
    return Z


def reweighted_joint(G_trn, Z_trn, G_ho, Z_ho, r_trn, g):
    """Standardize [G|Z], scale ONLY the richz columns by g, joint Ridge(alpha=100).
    Effective alpha on richz = ALPHA/g^2 (less shrinkage); geometry unchanged."""
    Xtrn = np.hstack([G_trn, Z_trn]); Xho = np.hstack([G_ho, Z_ho])
    sc = StandardScaler().fit(Xtrn)
    Ttrn = sc.transform(Xtrn); Tho = sc.transform(Xho)
    nz = Z_trn.shape[1]
    Ttrn[:, -nz:] *= g; Tho[:, -nz:] *= g
    r = Ridge(alpha=ALPHA).fit(Ttrn, r_trn)
    return r.predict(Tho)


def main():
    from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
    t0 = time.time()
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    names_tr = tr["name"].tolist()
    d = np.load(CACHE); ytr = d["ytr"]; se = d["se"]; n_tr = len(ytr); assert n_tr == 4139, n_tr
    Xbase, _ = feature_matrix(d, "combined")

    Xqm_raw = aligned_block(AIM, ACOLS, names_tr); Xst_raw = aligned_block(STR, SCOLS, names_tr)
    Xd4_raw = aligned_block(D4, DCOLS, names_tr);  Xdb_raw = aligned_block(DB, DBCOLS, names_tr)
    Xorb_raw = aligned_block(ORB, OCOLS, names_tr)

    Xsoap = np.load("C:/pxr_work/soap/soap_train_matrix.npy"); assert Xsoap.shape == (n_tr, 24)
    PMAP = np.load("C:/pxr_work/pmapper_feats.npz", allow_pickle=True)["train"].astype(float)
    assert PMAP.shape == (n_tr, 2048), PMAP.shape
    RICHZ = np.load(RICHZ_TR); assert RICHZ.shape == (n_tr, 512), RICHZ.shape
    print(f"richz real coverage: {(np.abs(RICHZ).sum(1)>0).sum()}/{n_tr}", flush=True)

    chem = np.load(f"{SD}/chemeleon_oof.npy"); tab = np.load(f"{SD}/tabpfn_oof.npy")
    gnn_full = np.load(f"{SD}/gnn_oof.npy")
    topK = topK_configs(); print(f"topK GBMs: {[c['arch'] for c in topK]}", flush=True)

    arms = ["CTRL_joint_richz24",
            "AH_rw_g1.5", "AH_rw_g2.5", "AH_rw_g4.0",
            "AI_seq_az30", "AI_seq_az100"]
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

        sa_trn = np.mean(glab_trn + [chem[trn], tab[trn], gnn_full[trn]], 0)
        sa_ho = np.mean(glab_ho + [chem[ho], tab[ho], sn_ho], 0)
        r_trn = ytr[trn] - np.clip(sa_trn, lo, hi)
        sa_ho_c = np.clip(sa_ho, lo, hi)

        pmap_pca = PCA(24, random_state=0).fit(PMAP[trn]); Zp = pmap_pca.transform(PMAP)
        G_trn = np.hstack([Xsoap[trn], Zp[trn]]); G_ho = np.hstack([Xsoap[ho], Zp[ho]])  # 48-d geom
        Z24 = richz_pca(RICHZ, trn, 24)

        # ===== CONTROL: deployed joint Ridge [SOAP|PMap|richz24], shared alpha=100 =====
        C_trn = np.hstack([G_trn, Z24[trn]]); C_ho = np.hstack([G_ho, Z24[ho]])
        scC, rC = ridge_fit(C_trn, r_trn)
        ctrl_pred = np.clip(sa_ho_c + BLEND * ridge_pred(scC, rC, C_ho), lo, hi)
        raes["CTRL_joint_richz24"].append(rae(ytr[ho], ctrl_pred))

        # ===== COMP-AH: block-reweighted joint Ridge (richz less shrunk) =====
        for g, arm in [(1.5, "AH_rw_g1.5"), (2.5, "AH_rw_g2.5"), (4.0, "AH_rw_g4.0")]:
            corr_ho = reweighted_joint(G_trn, Z24[trn], G_ho, Z24[ho], r_trn, g)
            pred = np.clip(sa_ho_c + BLEND * corr_ho, lo, hi)
            raes[arm].append(rae(ytr[ho], pred))

        # ===== COMP-AI: sequential boosted per-axis alpha (geom -> richz on residual) =====
        scG, rG = ridge_fit(G_trn, r_trn)
        c1_trn = ridge_pred(scG, rG, G_trn); c1_ho = ridge_pred(scG, rG, G_ho)
        r1_trn = r_trn - c1_trn
        for az, arm in [(30.0, "AI_seq_az30"), (100.0, "AI_seq_az100")]:
            scZ, rZ = ridge_fit(Z24[trn], r1_trn, alpha=az)
            c2_ho = ridge_pred(scZ, rZ, Z24[ho])
            seq_pred = np.clip(sa_ho_c + BLEND * (c1_ho + c2_ho), lo, hi)
            raes[arm].append(rae(ytr[ho], seq_pred))

        print(f"seed{seed} ctrl={raes['CTRL_joint_richz24'][-1]:.4f} "
              f"AHg1.5={raes['AH_rw_g1.5'][-1]:.4f} AHg2.5={raes['AH_rw_g2.5'][-1]:.4f} "
              f"AHg4.0={raes['AH_rw_g4.0'][-1]:.4f} "
              f"AIaz30={raes['AI_seq_az30'][-1]:.4f} AIaz100={raes['AI_seq_az100'][-1]:.4f}", flush=True)

    cm = float(np.mean(raes["CTRL_joint_richz24"]))
    print(f"\ncontrol (joint [SOAP|PMap|richz24]) mean RAE = {cm:.4f}  (deployed best = {DEPLOYED_BEST})",
          flush=True)
    out = {"control_mean_rae": cm, "ctrl_raes": raes["CTRL_joint_richz24"],
           "deployed_best": DEPLOYED_BEST, "compositions": {}}
    deploy = []
    for a in arms:
        if a == "CTRL_joint_richz24": continue
        mn = float(np.mean(raes[a])); delta = mn - cm
        neg = sum(1 for i, r in enumerate(raes[a]) if r < raes["CTRL_joint_richz24"][i])
        ok = (delta < -GATE) and (mn < DEPLOYED_BEST) and (neg >= 2)
        out["compositions"][a] = {"mean_rae": mn, "delta": delta, "raes": raes[a],
                                  "neg_seeds": neg, "deploy": ok}
        print(f"{a}: RAE={mn:.4f} delta={delta:+.4f} {neg}/3 neg  deploy={ok}", flush=True)
        if ok: deploy.append(a)
    out["deploy_candidates"] = deploy
    out["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(out, open(f"{P}/nb1430_combinator_summary.json", "w"), indent=2)
    print(f"\nsaved {P}/nb1430_combinator_summary.json  ({out['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
