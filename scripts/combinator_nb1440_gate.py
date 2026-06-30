"""COMBINATOR TICK nb1440 -- COVERAGE-RESTRICTED richz corrector fit (zero-row dilution fix).

DEPLOYED best = 0.4133 (best_ensemble.json, nb1400 COMP-Z2): 7-member flat mean
{4 GBM(combined+AIMNet2/strain/D4/DBSTEP/OrbMol), CheMeleon-OOF, TabPFN-OOF, sisterNR-GNN-OOF}
-> ONE Ridge(alpha=100) on the 72-d corrector [SOAP24 | PMapper-PCA24 | richz-PCA24], blend 0.5.

UNTESTED FLAW (distinct from nb1420/1430's "shared-alpha over-shrinks richz"): richz is only
86.5% train-covered (3581/4139); the ~558 ZERO-richz train rows have Z24 == 0 (a single point in
richz-space) but carry NONZERO, varying base residuals -> at that one point the corrector sees pure
NOISE, which biases the richz intercept/coefficients toward zero (dilution).  EVERY prior richz arm
fit the richz corrector on ALL trn rows:
  - deployed JOINT 72-d Ridge (nb1400): all rows, shared alpha.
  - nb1420/1430 SEQUENTIAL per-axis-alpha (AI_seq_az30/az100): all rows -> only -0.00013/-0.00019,
    1/3 neg, SUB-GATE.  The low alpha (30) that should help the strong activity-aligned richz axis
    actually AMPLIFIED the zero-row noise -> no robust gain.
NO tick has fit the richz part of the corrector on COVERED ROWS ONLY.  Mechanism: if the near-miss
was caused by zero-row dilution, restricting the richz fit to the 3581 real rows yields a cleaner
richz coefficient estimate -> a larger, more robust correction; and a LOWER richz alpha (less
shrinkage of the strong axis) should now help instead of amplifying noise.  Test is 100% richz-
covered, and we zero the richz correction wherever a holdout/test row is uncovered (matches deploy).

COMP-AJ  COVERED-ONLY SEQUENTIAL richz corrector (alpha_z = 100, deployed shrinkage):
  Stage1: Ridge(alpha=100) on [SOAP24|PMapper24] over ALL trn -> c1, residual r1.
  Stage2: Ridge(alpha=100) on richz24, fit on r1 RESTRICTED to richz-COVERED trn rows only -> c2.
  total = c1 + c2 (c2 zeroed where ho uncovered), blend 0.5.  Isolates the covered-only effect at
  the deployed alpha (clean A/B vs nb1430 AI_seq_az100 = all-rows, -0.00013).

COMP-AK  COVERED-ONLY SEQUENTIAL richz corrector with LOWER richz alpha (alpha_z = 30):
  Same as AJ but stage2 alpha_z=30.  nb1430 showed alpha_z=30 on ALL rows = -0.00019 (sub-gate,
  noise-amplified); covered-only removes the noise, so the less-shrunk strong richz axis may now
  clear the gate.  This is the untested COMBINATION (covered-only x low-alpha), not retried.

REF  AJ_allrows  -- replicate nb1430 AI_seq_az100 (sequential, richz on ALL rows, alpha=100) so the
  covered-only restriction (AJ) is a clean controlled A/B against the all-rows baseline this tick.

richz is 100% test-covered (leak-free, deployable).  Control = deployed [SOAP24|PMapper24|richz24]
joint Ridge faithfully reproduced w/ leak-free INNER-OOF GBM trn preds (matched).  richz PCA fit
per-seed on trn-real (nonzero) rows only; non-real -> zero.  Gate: matched delta vs control per seed
on ho_idx_seed{0,1,2}.  Deploy iff delta < -0.001 AND treat_mean < 0.4133 AND >=2/3 neg.
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
    real_richz = np.abs(RICHZ).sum(1) > 0
    print(f"richz real coverage: {real_richz.sum()}/{n_tr} = {real_richz.mean()*100:.1f}%", flush=True)

    chem = np.load(f"{SD}/chemeleon_oof.npy"); tab = np.load(f"{SD}/tabpfn_oof.npy")
    gnn_full = np.load(f"{SD}/gnn_oof.npy")
    topK = topK_configs(); print(f"topK GBMs: {[c['arch'] for c in topK]}", flush=True)

    arms = ["CTRL_joint_richz24",
            "AJ_cov_az100", "AK_cov_az30", "REF_allrows_az100"]
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

        cov_trn = real_richz[trn]      # bool over trn positions
        cov_ho = real_richz[ho]        # bool over ho positions

        # ===== CONTROL: deployed joint Ridge [SOAP|PMap|richz24], shared alpha=100 =====
        C_trn = np.hstack([G_trn, Z24[trn]]); C_ho = np.hstack([G_ho, Z24[ho]])
        scC, rC = ridge_fit(C_trn, r_trn)
        ctrl_pred = np.clip(sa_ho_c + BLEND * ridge_pred(scC, rC, C_ho), lo, hi)
        raes["CTRL_joint_richz24"].append(rae(ytr[ho], ctrl_pred))

        # ===== Stage1: geometry [SOAP|PMap] on ALL trn (shared by AJ/AK/REF) =====
        scG, rG = ridge_fit(G_trn, r_trn)
        c1_trn = ridge_pred(scG, rG, G_trn); c1_ho = ridge_pred(scG, rG, G_ho)
        r1_trn = r_trn - c1_trn

        # ===== COMP-AJ / COMP-AK: Stage2 richz on COVERED trn rows only =====
        for az, arm in [(100.0, "AJ_cov_az100"), (30.0, "AK_cov_az30")]:
            scZ, rZ = ridge_fit(Z24[trn][cov_trn], r1_trn[cov_trn], alpha=az)
            c2_ho = ridge_pred(scZ, rZ, Z24[ho]); c2_ho[~cov_ho] = 0.0  # no correction if uncovered
            pred = np.clip(sa_ho_c + BLEND * (c1_ho + c2_ho), lo, hi)
            raes[arm].append(rae(ytr[ho], pred))

        # ===== REF: Stage2 richz on ALL trn rows, alpha=100 (== nb1430 AI_seq_az100) =====
        scZr, rZr = ridge_fit(Z24[trn], r1_trn, alpha=100.0)
        c2r_ho = ridge_pred(scZr, rZr, Z24[ho])  # all-rows fit (no covered mask), as nb1430
        ref_pred = np.clip(sa_ho_c + BLEND * (c1_ho + c2r_ho), lo, hi)
        raes["REF_allrows_az100"].append(rae(ytr[ho], ref_pred))

        print(f"seed{seed} ctrl={raes['CTRL_joint_richz24'][-1]:.4f} "
              f"AJ_cov_az100={raes['AJ_cov_az100'][-1]:.4f} "
              f"AK_cov_az30={raes['AK_cov_az30'][-1]:.4f} "
              f"REF_allrows={raes['REF_allrows_az100'][-1]:.4f}  "
              f"(cov_trn={cov_trn.sum()}/{len(trn)} cov_ho={cov_ho.sum()}/{len(ho)})", flush=True)

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
        print(f"{a}: RAE={mn:.4f} delta={delta:+.5f} {neg}/3 neg  deploy={ok}", flush=True)
        if ok: deploy.append(a)
    out["deploy_candidates"] = deploy
    out["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(out, open(f"{P}/nb1440_combinator_summary.json", "w"), indent=2)
    print(f"\nsaved {P}/nb1440_combinator_summary.json  ({out['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
