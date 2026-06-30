"""COMBINATOR TICK nb1420 -- two NON-STANDARD corrector-architecture compositions.

DEPLOYED best = 0.4133 (best_ensemble.json, nb1400 COMP-Z2): 7-member flat mean
{4 GBM(combined+AIMNet2/strain/D4/DBSTEP/OrbMol), CheMeleon-OOF, TabPFN-OOF, sisterNR-GNN-OOF}
-> ONE Ridge(alpha=100) on the 72-d corrector [SOAP24 | PMapper-PCA24 | richz-PCA24], blend 0.5.

rich-z (Boltz-2 cofold protein x ligand INTERACTION embedding, the PXR ACTIVATION axis) was the
last WIN: it is orthogonal to all members AND activity-aligned (PCA-24 max|corr-w-y|=0.50 vs
SOAP/PMapper ~0.27).  Both compositions below exploit ONE untested structural fact: in the deployed
JOINT 72-d Ridge, the 24 activity-aligned richz dims SHARE a single alpha=100 shrinkage budget with
48 local-geometry dims (SOAP+PMapper) that are 2x lower corr-w-y.  A joint L2 with shared alpha
SHRINKS the strong-signal richz block the same as the weak-signal geometry blocks, and caps richz at
the same 24-PC budget the geometry axes saturate at.  Neither "decouple richz's regularization" nor
"give richz more PCA budget" has been tried (combinator_tried COMP-N varied SOAP dims only; every
corrector chain was single-axis stage-B or a single joint concat -- never a per-axis-alpha boosted
chain).

COMP-AF  SEQUENTIAL / BOOSTED corrector with per-axis alpha (vs deployed JOINT concat):
  Stage1: Ridge(alpha=100) on [SOAP24|PMapper24] -> fit base residual r0, correction c1, r1=r0-c1.
  Stage2: Ridge(alpha_z) on richz24 -> fit the running residual r1, correction c2.
  total correction = c1 + c2, applied at blend 0.5.  alpha_z in {30,100}: lower alpha lets the
  strong activity-aligned richz axis be LESS shrunk than the shared-budget joint Ridge allows.
  REV arm: richz-first (alpha_z) then [SOAP|PMapper](alpha=100) -- order sensitivity.

COMP-AG  ASYMMETRIC richz PCA dimensionality (vs deployed equal-24 budget):
  Joint Ridge(alpha=100) on [SOAP24 | PMapper24 | richz-D] with D in {48,96}.  Local-geometry
  (SOAP power-spectrum, PMapper pharmacophore) saturates at 24 PCs, but richz's 512-d activation
  source may carry activity signal in PCs 24-96.  Same blend/alpha as deployed; only richz budget grows.

Control = deployed [SOAP24|PMapper24|richz24] joint Ridge faithfully reproduced w/ leak-free
INNER-OOF GBM trn preds (matched).  richz PCA fit per-seed on trn-real (nonzero) rows only (leak-free);
non-real rows -> zero (centroid, no correction).  Gate: matched delta vs control per seed on
ho_idx_seed{0,1,2}.  Deploy iff delta < -0.001 AND treat_mean < 0.4133 AND >=2/3 neg.
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
RICHZ_TR = "C:/pxr_struct/boltz/boltz_z_rich_train.npy"   # (4139, 512) raw train richz

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
    """Leak-free richz PCA-d: fit StandardScaler+PCA on trn-real (nonzero) rows only;
    transform all rows; non-real rows -> zero (centroid, no correction)."""
    real = np.abs(richz_raw).sum(1) > 0
    fit_idx = trn[real[trn]]
    scz = StandardScaler().fit(richz_raw[fit_idx])
    pz = PCA(d, random_state=0).fit(scz.transform(richz_raw[fit_idx]))
    Z = np.zeros((len(richz_raw), d))
    Z[real] = pz.transform(scz.transform(richz_raw[real]))
    return Z


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
            "AF_seq_az30", "AF_seq_az100", "AF_seqREV_az30",
            "AG_richz48", "AG_richz96"]
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

        # deployed base: flat mean 4 GBM + 3 foundation
        sa_trn = np.mean(glab_trn + [chem[trn], tab[trn], gnn_full[trn]], 0)
        sa_ho = np.mean(glab_ho + [chem[ho], tab[ho], sn_ho], 0)
        r_trn = ytr[trn] - np.clip(sa_trn, lo, hi)
        sa_ho_c = np.clip(sa_ho, lo, hi)

        # corrector geometry blocks (leak-free): SOAP24 (cached) + PMapper PCA-24
        pmap_pca = PCA(24, random_state=0).fit(PMAP[trn]); Zp = pmap_pca.transform(PMAP)
        G_trn = np.hstack([Xsoap[trn], Zp[trn]]); G_ho = np.hstack([Xsoap[ho], Zp[ho]])  # 48-d geom

        # richz PCA blocks (leak-free, per seed) at 24/48/96
        Z24 = richz_pca(RICHZ, trn, 24); Z48 = richz_pca(RICHZ, trn, 48); Z96 = richz_pca(RICHZ, trn, 96)

        # ===== CONTROL: deployed joint Ridge [SOAP|PMap|richz24] =====
        C_trn = np.hstack([G_trn, Z24[trn]]); C_ho = np.hstack([G_ho, Z24[ho]])
        scC, rC = ridge_fit(C_trn, r_trn)
        ctrl_pred = np.clip(sa_ho_c + BLEND * ridge_pred(scC, rC, C_ho), lo, hi)
        raes["CTRL_joint_richz24"].append(rae(ytr[ho], ctrl_pred))

        # ===== COMP-AF: SEQUENTIAL boosted corrector, geom-first then richz(alpha_z) =====
        scG, rG = ridge_fit(G_trn, r_trn)                  # stage1: geometry
        c1_trn = ridge_pred(scG, rG, G_trn); c1_ho = ridge_pred(scG, rG, G_ho)
        r1_trn = r_trn - c1_trn                            # running residual after geometry
        for az, arm in [(30.0, "AF_seq_az30"), (100.0, "AF_seq_az100")]:
            scZ, rZ = ridge_fit(Z24[trn], r1_trn, alpha=az)  # stage2: richz on residual
            c2_ho = ridge_pred(scZ, rZ, Z24[ho])
            seq_pred = np.clip(sa_ho_c + BLEND * (c1_ho + c2_ho), lo, hi)
            raes[arm].append(rae(ytr[ho], seq_pred))

        # AF reverse order: richz(alpha_z=30) first, then geometry on the residual
        scZ0, rZ0 = ridge_fit(Z24[trn], r_trn, alpha=30.0)
        cz0_trn = ridge_pred(scZ0, rZ0, Z24[trn]); cz0_ho = ridge_pred(scZ0, rZ0, Z24[ho])
        scG2, rG2 = ridge_fit(G_trn, r_trn - cz0_trn)
        cg2_ho = ridge_pred(scG2, rG2, G_ho)
        rev_pred = np.clip(sa_ho_c + BLEND * (cz0_ho + cg2_ho), lo, hi)
        raes["AF_seqREV_az30"].append(rae(ytr[ho], rev_pred))

        # ===== COMP-AG: asymmetric richz dimensionality (joint Ridge, richz D=48/96) =====
        for Z, arm in [(Z48, "AG_richz48"), (Z96, "AG_richz96")]:
            Ca_trn = np.hstack([G_trn, Z[trn]]); Ca_ho = np.hstack([G_ho, Z[ho]])
            scA, rA = ridge_fit(Ca_trn, r_trn)
            ag_pred = np.clip(sa_ho_c + BLEND * ridge_pred(scA, rA, Ca_ho), lo, hi)
            raes[arm].append(rae(ytr[ho], ag_pred))

        print(f"seed{seed} ctrl={raes['CTRL_joint_richz24'][-1]:.4f} "
              f"AF30={raes['AF_seq_az30'][-1]:.4f} AF100={raes['AF_seq_az100'][-1]:.4f} "
              f"AFrev={raes['AF_seqREV_az30'][-1]:.4f} "
              f"AG48={raes['AG_richz48'][-1]:.4f} AG96={raes['AG_richz96'][-1]:.4f}", flush=True)

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
    json.dump(out, open(f"{P}/nb1420_combinator_summary.json", "w"), indent=2)
    print(f"\nsaved {P}/nb1420_combinator_summary.json  ({out['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
