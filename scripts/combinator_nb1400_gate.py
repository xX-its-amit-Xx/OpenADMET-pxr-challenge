"""COMBINATOR TICK nb1400 -- two NON-STANDARD compositions over EXISTING components.

Deployed best = 0.4149: 7-member flat mean {4 GBM(combined+AIMNet2/strain/D4/DBSTEP/OrbMol),
CheMeleon-OOF, TabPFN-OOF, sisterNR-GNN-OOF} -> [SOAP24|PMapper24] Ridge(alpha=100) residual
corrector, blend 0.5, clipped.  combinator_tried.md is heavily converged: weight-learning
overfits, robust-agg loses, corrector loss/feature/blend/weighting axes all closed, diverse
members dilute.  These two attack UNTRIED structural axes:

COMP-AB  FAMILY RESIDUAL CASCADE (re-architect stage-A; NOT flat mean):
  Instead of flat-averaging the 4 GBMs and 3 foundation models as EQUAL members, use the
  foundation models (CheMeleon+TabPFN+sisterNR, smooth pretrained priors) as the BASE and let
  the 4 GBMs (combined+QM, fine-grained tree learners) fit the foundation RESIDUAL.
  final = found_mean + mean(GBM-on-found-residual).  Mechanism: heterogeneous model families
  with a coarse(prior)->fine(residual) relationship can beat flat averaging.  Same components,
  different composition.  ALL prior comps kept stage-A as flat mean -> genuinely untried.
  Leak-free: foundation OOF + GBM residual-correctors via INNER 3-fold OOF on trn.

COMP-AC  REGIME-ROUTED CORRECTOR (specialist routing of the corrector):
  Replace the single global [SOAP|PMapper] Ridge with TWO regime-specialist Ridges --
  one fit on trn rows with base-pred < median (inactive regime), one >= median (active
  regime) -- and route each holdout compound to the matching corrector by its base pred.
  A-priori median split (computed on trn, NOT tuned), alpha=100, blend 0.5.  Mechanism:
  cy303 found regime-dependent error (low-activity tail = regression-to-mean variance); the
  geometry->residual map may differ by activity regime.  Distinct from COMP-T (blend-gating
  by AD density) and COMP-U (sample-weighting by disagreement) -- this ROUTES separate
  correctors, it does not gate the blend or reweight one fit.

Control = deployed best faithfully reproduced but with leak-free INNER-OOF GBM trn preds
(so the corrector residual is real OOF, matched for both treatments).  Gate: matched delta
vs control per seed on ho_idx_seed{0,1,2}.  Deploy iff delta < -0.001 AND treat_mean < 0.4149
AND >=2/3 neg.
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
BEST = f"{SD}/best_ensemble.json"; LOG = f"{SD}/results.jsonl"
N_SEEDS = 3; DEPLOYED_BEST = 0.4149; GATE = 0.001; ALPHA = 100.0; BLEND = 0.5
INNER_K = 3

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
    """Apply the config's noisy-quantile prep to a fit-index set."""
    if c["prep"] == "noisy20": return idx[se[idx] <= np.quantile(se, 0.8)]
    if c["prep"] == "noisy30": return idx[se[idx] <= np.quantile(se, 0.7)]
    return idx


def fit_pred(c, Xfull, y, fit_idx, pred_idx, se):
    m = make_model(c["arch"], c["hp"])
    fi = prep_filter(c, fit_idx, se)
    m.fit(Xfull[fi], y[fi])
    return m.predict(Xfull[pred_idx])


def gbm_label_oof(topK, Xfull, y, trn, ho, se, seed):
    """Per-arch: OOF preds on trn (inner KFold) + full-trn-fit preds on ho. Returns
    (oof_trn [K archs], ho_pred [K archs])."""
    kf = KFold(INNER_K, shuffle=True, random_state=1000 + seed)
    oof_trn = [np.zeros(len(trn)) for _ in topK]
    ho_pred = []
    for ci, c in enumerate(topK):
        for itr, iva in kf.split(trn):
            oof_trn[ci][iva] = fit_pred(c, Xfull, y, trn[itr], trn[iva], se)
        ho_pred.append(fit_pred(c, Xfull, y, trn, ho, se))
    return oof_trn, ho_pred


def gbm_resid_oof(topK, Xfull, resid_full, trn, ho, se, seed):
    """GBMs fit the (foundation) residual target. OOF on trn + full-trn preds on ho."""
    kf = KFold(INNER_K, shuffle=True, random_state=2000 + seed)
    oof_trn = [np.zeros(len(trn)) for _ in topK]
    ho_pred = []
    for ci, c in enumerate(topK):
        for itr, iva in kf.split(trn):
            oof_trn[ci][iva] = fit_pred(c, Xfull, resid_full, trn[itr], trn[iva], se)
        ho_pred.append(fit_pred(c, Xfull, resid_full, trn, ho, se))
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
    names_tr = tr["name"].tolist()
    d = np.load(CACHE); ytr = d["ytr"]; se = d["se"]; n_tr = len(ytr); assert n_tr == 4139, n_tr
    Xbase, _ = feature_matrix(d, "combined")

    Xqm_raw = aligned_block(AIM, ACOLS, names_tr); Xst_raw = aligned_block(STR, SCOLS, names_tr)
    Xd4_raw = aligned_block(D4, DCOLS, names_tr);  Xdb_raw = aligned_block(DB, DBCOLS, names_tr)
    Xorb_raw = aligned_block(ORB, OCOLS, names_tr)

    Xsoap = np.load("C:/pxr_work/soap/soap_train_matrix.npy"); assert Xsoap.shape == (n_tr, 24)
    PMAP = np.load("C:/pxr_work/pmapper_feats.npz", allow_pickle=True)["train"].astype(float)
    assert PMAP.shape == (n_tr, 2048), PMAP.shape

    chem = np.load(f"{SD}/chemeleon_oof.npy"); tab = np.load(f"{SD}/tabpfn_oof.npy")
    gnn_full = np.load(f"{SD}/gnn_oof.npy")
    topK = topK_configs(); print(f"topK GBMs: {[c['arch'] for c in topK]}", flush=True)

    arms = ["CTRL_flatmean", "COMP-AB_cascade", "COMP-AC_routed_corr"]
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

        # ---- foundation members (OOF) ----
        found_trn = np.mean([chem[trn], tab[trn], gnn_full[trn]], 0)   # OOF on trn
        found_ho = np.mean([chem[ho], tab[ho], sn_ho], 0)             # OOF on ho

        # ---- GBMs on label: OOF trn + ho ----
        glab_trn, glab_ho = gbm_label_oof(topK, Xfull, ytr, trn, ho, se, seed)

        # ===== CONTROL: flat mean of 4 GBM + 3 foundation, then [SOAP|PMapper] corrector =====
        sa_trn = np.mean(glab_trn + [chem[trn], tab[trn], gnn_full[trn]], 0)
        sa_ho = np.mean(glab_ho + [chem[ho], tab[ho], sn_ho], 0)
        r_trn = ytr[trn] - np.clip(sa_trn, lo, hi)

        # PMapper PCA-24 (leak-free fit on trn) + cached SOAP-24
        pmap_pca = PCA(24, random_state=0).fit(PMAP[trn]); Zp = pmap_pca.transform(PMAP)
        C_trn = np.hstack([Xsoap[trn], Zp[trn]]); C_ho = np.hstack([Xsoap[ho], Zp[ho]])

        scC, rC = ridge_fit(C_trn, r_trn)
        sa_ho_c = np.clip(sa_ho, lo, hi)
        ctrl_pred = np.clip(sa_ho_c + BLEND * ridge_pred(scC, rC, C_ho), lo, hi)
        raes["CTRL_flatmean"].append(rae(ytr[ho], ctrl_pred))

        # ===== COMP-AB: FAMILY RESIDUAL CASCADE =====
        resid_found_full = ytr - np.clip(  # foundation resid as GBM target; OOF on trn rows only used
            np.mean([chem, tab, gnn_full], 0), lo, hi)
        gres_trn, gres_ho = gbm_resid_oof(topK, Xfull, resid_found_full, trn, ho, se, seed)
        casc_trn = np.clip(found_trn, lo, hi) + np.mean(gres_trn, 0)
        casc_ho = np.clip(found_ho, lo, hi) + np.mean(gres_ho, 0)
        rc_trn = ytr[trn] - np.clip(casc_trn, lo, hi)
        scC2, rC2 = ridge_fit(C_trn, rc_trn)   # same corrector features, refit on cascade resid
        casc_ho_c = np.clip(casc_ho, lo, hi)
        casc_pred = np.clip(casc_ho_c + BLEND * ridge_pred(scC2, rC2, C_ho), lo, hi)
        raes["COMP-AB_cascade"].append(rae(ytr[ho], casc_pred))

        # ===== COMP-AC: REGIME-ROUTED CORRECTOR (2-bin by base-pred median) =====
        split = np.median(sa_trn)            # a-priori split, computed on trn base-pred
        lo_mask = sa_trn < split
        # specialist Ridges on the SAME [SOAP|PMapper] features
        scL, rL = ridge_fit(C_trn[lo_mask], r_trn[lo_mask])
        scH, rH = ridge_fit(C_trn[~lo_mask], r_trn[~lo_mask])
        corr_lo = ridge_pred(scL, rL, C_ho)
        corr_hi = ridge_pred(scH, rH, C_ho)
        ho_lo = sa_ho < split
        routed_corr = np.where(ho_lo, corr_lo, corr_hi)
        routed_pred = np.clip(sa_ho_c + BLEND * routed_corr, lo, hi)
        raes["COMP-AC_routed_corr"].append(rae(ytr[ho], routed_pred))

        print(f"seed{seed} ctrl={raes['CTRL_flatmean'][-1]:.4f} "
              f"AB_casc={raes['COMP-AB_cascade'][-1]:.4f} "
              f"AC_routed={raes['COMP-AC_routed_corr'][-1]:.4f} "
              f"(n_lo_split={int(lo_mask.sum())}/{len(trn)})", flush=True)

    cm = float(np.mean(raes["CTRL_flatmean"]))
    print(f"\ncontrol (flat-mean OOF) mean RAE = {cm:.4f}  (deployed best = {DEPLOYED_BEST})", flush=True)
    out = {"control_mean_rae": cm, "ctrl_raes": raes["CTRL_flatmean"],
           "deployed_best": DEPLOYED_BEST, "compositions": {}}
    deploy = []
    for a in ["COMP-AB_cascade", "COMP-AC_routed_corr"]:
        mn = float(np.mean(raes[a])); delta = mn - cm
        neg = sum(1 for i, r in enumerate(raes[a]) if r < raes["CTRL_flatmean"][i])
        ok = (delta < -GATE) and (mn < DEPLOYED_BEST) and (neg >= 2)
        out["compositions"][a] = {"mean_rae": mn, "delta": delta, "raes": raes[a],
                                  "neg_seeds": neg, "deploy": ok}
        print(f"{a}: RAE={mn:.4f} delta={delta:+.4f} {neg}/3 neg  deploy={ok}", flush=True)
        if ok: deploy.append(a)
    out["deploy_candidates"] = deploy
    out["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(out, open(f"{P}/nb1400_combinator_summary.json", "w"), indent=2)
    print(f"\nsaved {P}/nb1400_combinator_summary.json  ({out['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
