"""nb1334 -- COMBINATOR TICK: diverse-member injection on deployed best (SOAP chain, 0.4161).

Two NON-STANDARD compositions, both untried (combinator_tried.md flags "TabNet NOT
tried yet as any composition component"; maceoff-as-member also never tried):

COMP-R: TabNet OOF as a DOWN-WEIGHTED FLAT MEMBER (pre-SOAP).
  TabNet is the 2nd-most-diverse OOF available (corr 0.896 vs deployed ensemble;
  every deployed member is 0.93-0.98). Flat-mean ensembles reward DIVERSITY, not
  individual accuracy. Inject as a weighted member into stage-A, w in {0.3,0.5,1.0}.
COMP-S: MACE-OFF model OOF as a DOWN-WEIGHTED FLAT MEMBER (pre-SOAP).
  maceoff is the MOST diverse OOF available (corr 0.844 vs ensemble). As a model
  PREDICTION (not the absorbed MACE-OFF feature scalars) it may carry orthogonal
  signal. Same weight grid.

KEY DISTINCTION from prior ticks: COMP-A/B (routing / NNLS weights) LEARNED weights
and overfit; COMP-C robust-agg discards signal; COMP-G/H QM/kNN residuals; COMP-O
(running) tests TabNet as a *residual corrector after SOAP*. This tick tests the
simplest untried thing: TabNet/maceoff as a *flat member BEFORE SOAP*, with the SOAP
chain REFIT on the new stage-A residual each treatment (fair matched comparison).

Control = deployed: 4 GBM(combined + AIMNet2/strain/D4/DBSTEP/OrbMol) + CheMeleon +
TabPFN + sisterNR-GNN, flat mean, then SOAP PCA-24 Ridge(alpha=100) blend=0.5.
Gate: matched delta vs control per seed on MTL/ho_idx_seed{0,1,2}. a-priori = w=0.3
(most conservative). Deploy iff a-priori delta < -0.001 AND treatment < 0.4161.
"""
import os, sys, json
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb1126_combinatorial_search import feature_matrix, CACHE, make_model
from src.pxr.data import load_train
from src.pxr.eval import rae
import lightgbm as lgb

P = "data/processed"; SD = "C:/pxr_work/search"; MTL = "C:/pxr_work/mtl"
BEST = f"{SD}/best_ensemble.json"; LOG = f"{SD}/results.jsonl"; N_SEEDS = 3
WEIGHTS = [0.3, 0.5, 1.0]

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
    med = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X)); X[inds] = np.take(med, inds[1])
    return X


def topK_configs():
    recs = [json.loads(l) for l in open(LOG) if l.strip()]
    valid = [r for r in recs if "ps_rae" in r and "error" not in r
             and r["arch"] in ("lgbm","xgb","cat","histgb")]
    valid.sort(key=lambda r: r["ps_rae"]); bp = {}
    for r in valid: bp.setdefault(r["arch"], r)
    return list(bp.values())


def fit_pred_qm(c, Xbase, Xqm, ytr, use_idx, te_rows):
    Xfull = np.hstack([Xbase, Xqm])
    m = make_model(c["arch"], c["hp"])
    d = np.load(CACHE); se = d["se"]
    if c["prep"] == "noisy20": use_idx = use_idx[se[use_idx] <= np.quantile(se, 0.8)]
    elif c["prep"] == "noisy30": use_idx = use_idx[se[use_idx] <= np.quantile(se, 0.7)]
    m.fit(Xfull[use_idx], ytr[use_idx])
    return m.predict(Xfull[te_rows])


def soap_residual_correct(Xsoap_tr_trn, Xsoap_tr_ho, resid_trn):
    """Fit Ridge(alpha=100) on SOAP(train-fold) -> stageA residual; predict holdout corr."""
    sc = StandardScaler().fit(Xsoap_tr_trn)
    r = Ridge(alpha=100.0).fit(sc.transform(Xsoap_tr_trn), resid_trn)
    return r.predict(sc.transform(Xsoap_tr_ho))


def main():
    from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    names_tr = tr["name"].tolist()
    d = np.load(CACHE); ytr = d["ytr"]; n_tr = len(ytr)
    assert n_tr == 4139, n_tr
    Xbase, _ = feature_matrix(d, "combined")

    # physics blocks (5) - raw, scaled per train-fold inside loop
    Xqm_raw = aligned_block(AIM, ACOLS, names_tr)
    Xst_raw = aligned_block(STR, SCOLS, names_tr)
    Xd4_raw = aligned_block(D4, DCOLS, names_tr)
    Xdb_raw = aligned_block(DB, DBCOLS, names_tr)
    Xorb_raw = aligned_block(ORB, OCOLS, names_tr)

    # SOAP PCA-24 train matrix (aligned to load_train order)
    Xsoap = np.load("C:/pxr_work/soap/soap_train_matrix.npy")
    assert Xsoap.shape == (n_tr, 24), Xsoap.shape

    chem = np.load(f"{SD}/chemeleon_oof.npy")
    tab = np.load(f"{SD}/tabpfn_oof.npy")
    gnn_full = np.load(f"{SD}/gnn_oof.npy")
    tabnet = np.load(f"{SD}/tabnet_oof.npy")
    maceoff = np.load(f"{SD}/maceoff_oof.npy")
    topK = topK_configs()
    print(f"topK GBMs: {[c['arch'] for c in topK]}")

    EXTRAS = {"COMP-R_tabnet": tabnet, "COMP-S_maceoff": maceoff}
    # results[name][w] = list of treatment RAEs across seeds
    results = {nm: {w: [] for w in WEIGHTS} for nm in EXTRAS}
    ctrl_raes = []

    for seed in range(N_SEEDS):
        ho = np.load(f"{MTL}/ho_idx_seed{seed}.npy")
        ho_set = set(ho.tolist())
        trn = np.array([i for i in range(n_tr) if i not in ho_set])
        lo, hi = np.quantile(ytr[trn], 0.05), np.quantile(ytr[trn], 0.98)
        sn_ho = np.load(f"{MTL}/sn_oof_seed{seed}.npy").ravel()

        sc_aim = StandardScaler().fit(Xqm_raw[trn]); sc_str = StandardScaler().fit(Xst_raw[trn])
        sc_d4 = StandardScaler().fit(Xd4_raw[trn]); sc_db = StandardScaler().fit(Xdb_raw[trn])
        sc_orb = StandardScaler().fit(Xorb_raw[trn])
        Xqm_full = np.hstack([sc_aim.transform(Xqm_raw), sc_str.transform(Xst_raw),
                              sc_d4.transform(Xd4_raw), sc_db.transform(Xdb_raw),
                              sc_orb.transform(Xorb_raw)])  # (4139, 61)

        gbm_ho, gbm_trn = [], []
        for c in topK:
            gbm_ho.append(fit_pred_qm(c, Xbase, Xqm_full, ytr, trn, ho))
            gbm_trn.append(fit_pred_qm(c, Xbase, Xqm_full, ytr, trn, trn))

        # ---- CONTROL: deployed flat mean + SOAP chain ----
        base_ho = gbm_ho + [chem[ho], tab[ho], sn_ho]
        base_trn = gbm_trn + [chem[trn], tab[trn], gnn_full[trn]]
        sa_ho_ctrl = np.mean(base_ho, 0)
        sa_trn_ctrl = np.mean(base_trn, 0)
        r_trn_ctrl = ytr[trn] - np.clip(sa_trn_ctrl, lo, hi)
        soap_corr_ctrl = soap_residual_correct(Xsoap[trn], Xsoap[ho], r_trn_ctrl)
        ctrl_pred = np.clip(np.clip(sa_ho_ctrl, lo, hi) + 0.5 * soap_corr_ctrl, lo, hi)
        ctrl_raes.append(rae(ytr[ho], ctrl_pred))

        # ---- TREATMENTS: weighted-member flat mean + SOAP chain (refit) ----
        nb = len(base_ho)  # number of base members
        line = [f"seed{seed} ctrl={ctrl_raes[-1]:.4f}"]
        for nm, extra in EXTRAS.items():
            for w in WEIGHTS:
                # weighted mean: (sum base + w*extra) / (nb + w)
                sa_ho = (np.sum(base_ho, 0) + w * extra[ho]) / (nb + w)
                sa_trn = (np.sum(base_trn, 0) + w * extra[trn]) / (nb + w)
                r_trn = ytr[trn] - np.clip(sa_trn, lo, hi)
                scorr = soap_residual_correct(Xsoap[trn], Xsoap[ho], r_trn)
                pred = np.clip(np.clip(sa_ho, lo, hi) + 0.5 * scorr, lo, hi)
                results[nm][w].append(rae(ytr[ho], pred))
            line.append(f"{nm}@0.5={results[nm][0.5][-1]:.4f}")
        print("  ".join(line))

    cm = float(np.mean(ctrl_raes))
    prev = json.load(open(BEST))["rae"] if os.path.exists(BEST) else 0.4161
    print(f"\ncontrol (deployed SOAP-chain) mean RAE = {cm:.4f}  (best_ensemble.json={prev})")

    out = {"control_mean_rae": cm, "ctrl_raes": ctrl_raes, "deployed_best": prev,
           "weights": WEIGHTS, "compositions": {}}
    deploy_candidates = []
    for nm in EXTRAS:
        out["compositions"][nm] = {}
        print(f"\n=== {nm} ===")
        for w in WEIGHTS:
            raes = results[nm][w]; mn = float(np.mean(raes)); delta = mn - cm
            neg = sum(1 for i, r in enumerate(raes) if r < ctrl_raes[i])
            out["compositions"][nm][f"w{w}"] = {"mean_rae": mn, "delta": delta,
                                                "raes": raes, "neg_seeds": neg}
            print(f"  w={w}: {mn:.4f} (delta {delta:+.4f}, {neg}/3 neg)")
        # a-priori = smallest weight (most conservative)
        ap = WEIGHTS[0]; apd = out["compositions"][nm][f"w{ap}"]
        print(f"  a-priori (w={ap}): RAE={apd['mean_rae']:.4f} delta={apd['delta']:+.4f} {apd['neg_seeds']}/3 neg")
        if apd["delta"] < -0.001 and apd["mean_rae"] < prev and apd["neg_seeds"] >= 2:
            print(f"  -> BEATS gate: CANDIDATE")
            deploy_candidates.append((nm, ap, apd))
        else:
            print(f"  -> no deploy")

    out["deploy_candidates"] = [(nm, w) for nm, w, _ in deploy_candidates]
    json.dump(out, open(f"{P}/nb1334_summary.json", "w"), indent=2)
    print(f"\nsaved {P}/nb1334_summary.json")


if __name__ == "__main__":
    main()
