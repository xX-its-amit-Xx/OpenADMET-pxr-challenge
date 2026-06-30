"""nb1325 — ErG pharmacophore fingerprint gate (SOAP-chain control).

RDKit ErG (Extended Reduced Graph): 315-float pharmacophore topology vector.
Zero-install, SMILES-only. Prior: corr(ErG, pEC50)=+0.234 cycle-299 (strongest
2D signal ever seen, NEVER gated on honest holdout).

Gate pattern mirrors COMP-I from combinator_nb1315.py:
  ctrl:  4-GBM(combined+QM) + SOAP residual chain  → ~0.4161
  treat: 4-GBM(combined+QM+ErG) + SOAP residual chain

Inner loop per seed:
  1. 4-GBM(+ErG) trained on trn → preds on ho (treat_ho, ctrl_ho)
  2. Single LGBM inner-3-fold on trn → pseudo-OOF residuals → Ridge(SOAP, alpha=100, blend=0.5)
  3. pred_soap = clip(pred_ctrl + 0.5 * ridge.predict(soap_ho))
  4. RAE on ho

Deploy if matched delta < -0.001 AND treat_mean < 0.4161 (deployed best).
"""
import os, sys, json, time
import numpy as np, pandas as pd
import lightgbm as lgb
from rdkit import Chem, RDLogger
from rdkit.Chem.rdReducedGraphs import GetErGFingerprint
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.pxr.data import load_train, load_test
from src.pxr.eval import rae
from nb1126_combinatorial_search import feature_matrix, CACHE, make_model

SD   = "C:/pxr_work/search"
MTL  = "C:/pxr_work/mtl"
BEST = f"{SD}/best_ensemble.json"
LOG  = f"{SD}/results.jsonl"
N_SEEDS  = 3
BEST_RAE = 0.4161
GATE     = 0.001

ACOLS  = ["aimnet_energy","aimnet_qmin","aimnet_qmax","aimnet_qabs_mean",
          "aimnet_qstd","aimnet_qsum_abs","aimnet_dipole","aimnet_fmax","aimnet_frms"]
SCOLS  = ["strain_relax_mean","strain_relax_max","conf_espread","conf_erange",
          "conf_n","rmsd_mean","rmsd_max","e_per_heavy"]
DCOLS  = ["d4_alpha_sum","d4_alpha_mean","d4_alpha_std","d4_alpha_max",
          "d4_c6diag_mean","d4_c6diag_std","d4_c6_total","d4_edisp",
          "d4_edisp_per_atom","d4_cn_mean","d4_cn_max","d4_qeeq_min",
          "d4_qeeq_max","d4_qeeq_std","d4_qeeq_absum"]
DBCOLS = ["vbur_r25","vbur_r35","vbur_r45","vbur_r55","vbur_r65",
          "ster_L","ster_Bmin","ster_Bmax","ster_aniso",
          "npr1","npr2","asphericity","spherocity","eccentricity","radgyr","inertial_sf"]
OCOLS  = ["orb_energy","orb_energy_per_ha","orb_fmax","orb_frms","orb_fstd",
          "orb_conf_mean","orb_conf_std","orb_conf_node_mean","orb_conf_node_std",
          "orb_conf_node_min","orb_node_emb_mean","orb_node_emb_std","orb_node_emb_norm"]


def ablk(csv, src_val, cols, names):
    df = pd.read_csv(csv)
    if "src" in df.columns:
        df = df[df.src == src_val]
    df = df.drop_duplicates(subset="name", keep="first")
    sub = df.set_index("name").reindex(names)
    X = sub[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    med = np.nanmedian(X, axis=0); ii = np.where(np.isnan(X))
    X[ii] = np.take(med, ii[1])
    return X.astype(np.float32)


def compute_erg(smiles_list):
    n = len(smiles_list)
    out = np.zeros((n, 315), dtype=np.float32)
    errs = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            errs.append(i); continue
        try:
            out[i] = np.asarray(GetErGFingerprint(mol), dtype=np.float32)
        except Exception:
            errs.append(i)
    if errs:
        # fill with column medians
        valid_rows = np.array([j for j in range(n) if j not in errs])
        if len(valid_rows):
            col_med = np.median(out[valid_rows], axis=0)
            for i in errs:
                out[i] = col_med
        print(f"  ErG: {len(errs)}/{n} errors filled with median")
    return out


def topK_configs(archs):
    recs = [json.loads(l) for l in open(LOG) if l.strip()]
    valid = [r for r in recs if "ps_rae" in r and "error" not in r and r["arch"] in archs]
    valid.sort(key=lambda r: r["ps_rae"])
    bp = {}
    for r in valid:
        bp.setdefault(r["arch"], r)
    return list(bp.values())


def fp_ho(c, A_full, ytr, trn, ho):
    """Fit GBM on trn, predict ho."""
    mc = make_model(c["arch"], c["hp"])
    mc.fit(A_full[trn], ytr[trn])
    return mc.predict(A_full[ho])


def main():
    t0 = time.time()
    print("=== nb1325 ErG pharmacophore gate ===", flush=True)

    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test().reset_index(drop=True)
    ytr = tr["pec50"].to_numpy(float)
    n   = len(ytr)
    tr_names = tr["name"].tolist()
    se = tr["pec50_se"].to_numpy(float)

    # === Compute ErG features ===
    print("Computing ErG fingerprints...", flush=True)
    Xerg = compute_erg(tr["smiles"].tolist())
    print(f"  ErG: {Xerg.shape}  non-zero rows: {(Xerg.sum(1) > 0).sum()}", flush=True)

    # Corr with nb3200 error
    nb3200_path = "data/processed/oof_chemprop_aux.npy"
    if os.path.exists(nb3200_path):
        err = ytr - np.load(nb3200_path)
        all_corrs = np.array([float(np.corrcoef(err, Xerg[:, j])[0, 1]) for j in range(315)])
        print(f"  Max |corr(nb3200_err, ErG[j])|  = {np.max(np.abs(all_corrs)):.4f}", flush=True)
        print(f"  Mean |corr(nb3200_err, ErG[j])| = {np.mean(np.abs(all_corrs)):.4f}", flush=True)
        top5 = np.argsort(-np.abs(all_corrs))[:5]
        for j in top5:
            print(f"    ErG[{j}]: corr={all_corrs[j]:+.4f}", flush=True)

    # === Load combined features + QM blocks ===
    d = np.load(CACHE)
    Xtr_comb, _ = feature_matrix(d, "combined")  # (n, 2265)

    Xqm  = ablk("C:/pxr_work/aimnet2/aimnet_features.csv",  "train", ACOLS,  tr_names)
    Xst  = ablk("C:/pxr_work/strain/strain_features.csv",   "train", SCOLS,  tr_names)
    Xd4  = ablk("C:/pxr_work/d4/d4_features.csv",           "train", DCOLS,  tr_names)
    Xdb  = ablk("C:/pxr_work/dbstep/dbstep_features.csv",   "train", DBCOLS, tr_names)
    Xorb = ablk("C:/pxr_work/orbmol/orbmol_features.csv",   "train", OCOLS,  tr_names)

    # SOAP PCA features (24-d)
    soap_df   = pd.read_csv("C:/pxr_work/soap/soap_pca.csv")
    soap_cols = [c for c in soap_df.columns if c.startswith("soap_")]
    soap_tr_df = soap_df[soap_df["src"] == "train"].copy().set_index("name").reindex(tr_names)
    Xsoap      = soap_tr_df[soap_cols].values.astype(float)
    col_med    = np.nanmedian(Xsoap, axis=0)
    ii = np.where(np.isnan(Xsoap)); Xsoap[ii] = np.take(col_med, ii[1])
    print(f"  SOAP: {Xsoap.shape}", flush=True)

    chem_oof = np.load(f"{SD}/chemeleon_oof.npy")
    tab_oof  = np.load(f"{SD}/tabpfn_oof.npy")
    topK = topK_configs(("lgbm", "xgb", "cat", "histgb"))
    print(f"  4-GBM: {[c['arch'] for c in topK]}", flush=True)
    print(f"  Deployed best = {BEST_RAE:.4f}", flush=True)

    ctrl_raes, treat_raes = [], []

    for seed in range(N_SEEDS):
        ho  = np.load(f"{MTL}/ho_idx_seed{seed}.npy")
        hs  = set(ho.tolist())
        trn = np.array([i for i in range(n) if i not in hs])
        lo, hi = np.quantile(ytr[trn], 0.05), np.quantile(ytr[trn], 0.98)

        # QM scalar scalers
        sc_qm  = StandardScaler().fit(Xqm[trn]);  Xqm_s  = sc_qm.transform(Xqm)
        sc_st  = StandardScaler().fit(Xst[trn]);  Xst_s  = sc_st.transform(Xst)
        sc_d4  = StandardScaler().fit(Xd4[trn]);  Xd4_s  = sc_d4.transform(Xd4)
        sc_db  = StandardScaler().fit(Xdb[trn]);  Xdb_s  = sc_db.transform(Xdb)
        sc_orb = StandardScaler().fit(Xorb[trn]); Xorb_s = sc_orb.transform(Xorb)
        Xqm_block = np.hstack([Xqm_s, Xst_s, Xd4_s, Xdb_s, Xorb_s])

        # ErG scaler
        sc_erg = StandardScaler().fit(Xerg[trn])
        Xerg_s = sc_erg.transform(Xerg)

        # GNN members — these files are (244,) holdout-only predictions (no [ho] indexing needed)
        sn_oof   = np.load(f"{MTL}/sn_oof_seed{seed}.npy").ravel()   # (244,)
        rpxr_p   = f"{MTL}/rpxr_oof_seed{seed}.npy"
        rpxr_oof = np.load(rpxr_p).ravel() if os.path.exists(rpxr_p) else np.zeros(len(ho))

        # SOAP scaler
        soap_sc      = StandardScaler().fit(Xsoap[trn])
        soap_trn_s   = soap_sc.transform(Xsoap[trn])
        soap_ho_s    = soap_sc.transform(Xsoap[ho])

        # ==== CTRL ====
        # 4-GBM on combined + QM
        A_ctrl = np.hstack([Xtr_comb, Xqm_block])
        gbm_ctrl_ho = [fp_ho(c, A_ctrl, ytr, trn, ho) for c in topK]
        pred_ctrl_ho = np.clip(
            np.mean(gbm_ctrl_ho + [chem_oof[ho], tab_oof[ho], sn_oof, rpxr_oof], 0),
            lo, hi)

        # Inner 3-fold for SOAP Ridge (single LGBM proxy)
        rng = np.random.RandomState(seed * 7 + 42)
        inner_perm = rng.permutation(len(trn))
        fold_size  = len(trn) // 3
        trn_resid_ctrl = np.zeros(len(trn))
        for k in range(3):
            vm = inner_perm[k*fold_size:(k+1)*fold_size]
            tm = np.concatenate([inner_perm[:k*fold_size], inner_perm[(k+1)*fold_size:]])
            m_i = lgb.LGBMRegressor(n_estimators=300, num_leaves=64,
                                     learning_rate=0.05, random_state=seed, verbose=-1)
            m_i.fit(A_ctrl[trn[tm]], ytr[trn[tm]])
            p = m_i.predict(A_ctrl[trn[vm]])
            trn_resid_ctrl[vm] = ytr[trn[vm]] - np.clip(p, lo, hi)
        ridge_ctrl = Ridge(alpha=100.0).fit(soap_trn_s, trn_resid_ctrl)
        pred_soap_ctrl = np.clip(pred_ctrl_ho + 0.5 * ridge_ctrl.predict(soap_ho_s), lo, hi)
        c_rae = rae(ytr[ho], pred_soap_ctrl)
        ctrl_raes.append(c_rae)

        # ==== TREAT (+ ErG) ====
        A_treat = np.hstack([Xtr_comb, Xqm_block, Xerg_s])
        gbm_treat_ho = [fp_ho(c, A_treat, ytr, trn, ho) for c in topK]
        pred_treat_ho = np.clip(
            np.mean(gbm_treat_ho + [chem_oof[ho], tab_oof[ho], sn_oof, rpxr_oof], 0),
            lo, hi)

        rng2 = np.random.RandomState(seed * 7 + 42)
        inner_perm2 = rng2.permutation(len(trn))
        trn_resid_treat = np.zeros(len(trn))
        for k in range(3):
            vm = inner_perm2[k*fold_size:(k+1)*fold_size]
            tm = np.concatenate([inner_perm2[:k*fold_size], inner_perm2[(k+1)*fold_size:]])
            m_i = lgb.LGBMRegressor(n_estimators=300, num_leaves=64,
                                     learning_rate=0.05, random_state=seed, verbose=-1)
            m_i.fit(A_treat[trn[tm]], ytr[trn[tm]])
            p = m_i.predict(A_treat[trn[vm]])
            trn_resid_treat[vm] = ytr[trn[vm]] - np.clip(p, lo, hi)
        ridge_treat = Ridge(alpha=100.0).fit(soap_trn_s, trn_resid_treat)
        pred_soap_treat = np.clip(pred_treat_ho + 0.5 * ridge_treat.predict(soap_ho_s), lo, hi)
        t_rae = rae(ytr[ho], pred_soap_treat)
        treat_raes.append(t_rae)

        print(f"  seed={seed} | ctrl={c_rae:.4f} | treat={t_rae:.4f} | delta={t_rae-c_rae:+.4f}", flush=True)

    ctrl_mean  = float(np.mean(ctrl_raes))
    treat_mean = float(np.mean(treat_raes))
    delta      = treat_mean - ctrl_mean
    neg_seeds  = sum(t < c for t, c in zip(treat_raes, ctrl_raes))
    deploy     = (delta < -GATE) and (treat_mean < BEST_RAE)

    print(f"\nErG RESULT: ctrl={ctrl_mean:.4f} treat={treat_mean:.4f} "
          f"delta={delta:+.4f} neg_seeds={neg_seeds}/3 deploy={deploy}", flush=True)

    summary = {
        "feature": "erg_315_pharmacophore",
        "control_rae": ctrl_mean,
        "treatment_rae": treat_mean,
        "matched_delta": delta,
        "c_raes": [float(x) for x in ctrl_raes],
        "t_raes": [float(x) for x in treat_raes],
        "neg_seeds": neg_seeds,
        "deployed_best": BEST_RAE,
        "deploy": deploy,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path = "data/processed/nb1325_erg_summary.json"
    json.dump(summary, open(out_path, "w"), indent=2)
    print(f"Saved: {out_path}", flush=True)
    return summary


if __name__ == "__main__":
    main()
