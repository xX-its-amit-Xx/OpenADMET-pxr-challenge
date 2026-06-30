"""nb1329 — EspalomaCharge AM1-BCC partial-charge gate (SOAP-chain control).

EspalomaCharge (choderalab/espaloma_charge, MIT): GNN surrogate for AM1-BCC
partial charges. Per-atom charge array -> pool to mol-level electrostatics
descriptors (10 scalars).

Gate pattern mirrors nb1325_erg_gate.py:
  ctrl:  4-GBM(combined+QM) + SOAP residual chain  -> ~0.4161
  treat: 4-GBM(combined+QM+espaloma10) + SOAP residual chain

Deploy if matched delta < -0.001 AND treat_mean < 0.4161 (deployed best).
"""
import os, sys, json, time, warnings
import numpy as np, pandas as pd
import lightgbm as lgb
from rdkit import Chem, RDLogger
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")
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

FEAT_DIR = "C:/pxr_work/espaloma"
os.makedirs(FEAT_DIR, exist_ok=True)

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

ESP_COLS = ["esp_q_abssum","esp_q_max","esp_q_min","esp_q_var","esp_q_range",
            "esp_q_absmean","esp_q_std","esp_n_pos","esp_n_neg","esp_n_charged"]


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


def compute_espaloma_features(smiles_list):
    """Pool AM1-BCC partial charges to 10 mol-level electrostatic descriptors."""
    from espaloma_charge import charge as esp_charge
    from rdkit.Chem import AllChem

    n = len(smiles_list)
    out = np.zeros((n, len(ESP_COLS)), dtype=np.float32)
    errs = []

    for i, smi in enumerate(smiles_list):
        if i % 500 == 0:
            print(f"  Espaloma: {i}/{n}...", flush=True)
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                errs.append(i); continue
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            mol = Chem.RemoveHs(mol)
            # Get charges — works on heavy-atom mol
            q = esp_charge(mol)  # numpy array, shape (n_heavy_atoms,)
            q = np.array(q, dtype=float)
            if q is None or len(q) == 0:
                errs.append(i); continue
            # Pool to molecule-level descriptors
            out[i, 0] = float(np.sum(np.abs(q)))       # esp_q_abssum
            out[i, 1] = float(np.max(q))                # esp_q_max (most positive)
            out[i, 2] = float(np.min(q))                # esp_q_min (most negative)
            out[i, 3] = float(np.var(q))                # esp_q_var
            out[i, 4] = float(np.max(q) - np.min(q))   # esp_q_range
            out[i, 5] = float(np.mean(np.abs(q)))       # esp_q_absmean
            out[i, 6] = float(np.std(q))                # esp_q_std
            out[i, 7] = float(np.sum(q > 0.1))          # esp_n_pos (formal-positive-like)
            out[i, 8] = float(np.sum(q < -0.1))         # esp_n_neg
            out[i, 9] = float(np.sum(np.abs(q) > 0.15)) # esp_n_charged
        except Exception as e:
            errs.append(i)

    if errs:
        valid_rows = np.array([j for j in range(n) if j not in errs])
        if len(valid_rows):
            col_med = np.median(out[valid_rows], axis=0)
            for i in errs:
                out[i] = col_med
        print(f"  Espaloma: {len(errs)}/{n} errors filled with median", flush=True)
    else:
        print(f"  Espaloma: all {n} OK", flush=True)
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
    mc = make_model(c["arch"], c["hp"])
    mc.fit(A_full[trn], ytr[trn])
    return mc.predict(A_full[ho])


def main():
    t0 = time.time()
    print("=== nb1329 EspalomaCharge AM1-BCC partial-charge gate ===", flush=True)

    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test().reset_index(drop=True)
    ytr = tr["pec50"].to_numpy(float)
    n   = len(ytr)
    tr_names = tr["name"].tolist()
    te_names = te["name"].tolist()

    # === Compute EspalomaCharge features ===
    feat_cache = f"{FEAT_DIR}/espaloma_features.csv"
    if os.path.exists(feat_cache):
        print(f"Loading cached features from {feat_cache}", flush=True)
        df_feats = pd.read_csv(feat_cache)
    else:
        print("Computing EspalomaCharge features for train+test...", flush=True)
        all_smiles = tr["smiles"].tolist() + te["smiles"].tolist()
        all_names  = tr_names + te_names
        all_src    = ["train"] * len(tr_names) + ["test"] * len(te_names)
        Xall = compute_espaloma_features(all_smiles)
        df_feats = pd.DataFrame(Xall, columns=ESP_COLS)
        df_feats["name"] = all_names
        df_feats["src"]  = all_src
        df_feats.to_csv(feat_cache, index=False)
        print(f"Saved: {feat_cache}", flush=True)

    Xesp_tr = ablk(feat_cache, "train", ESP_COLS, tr_names)
    print(f"  Espaloma train: {Xesp_tr.shape}", flush=True)

    # Corr with nb3200 error
    nb3200_path = "data/processed/oof_chemprop_aux.npy"
    if os.path.exists(nb3200_path):
        err = ytr - np.load(nb3200_path)
        corrs = np.array([float(np.corrcoef(err, Xesp_tr[:, j])[0, 1]) for j in range(len(ESP_COLS))])
        for j, col in enumerate(ESP_COLS):
            print(f"    corr(err, {col}) = {corrs[j]:+.4f}", flush=True)
        print(f"  Max |corr(err, espaloma)|  = {np.max(np.abs(corrs)):.4f}", flush=True)

    # === Load combined features + QM blocks ===
    d = np.load(CACHE)
    Xtr_comb, _ = feature_matrix(d, "combined")

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
    print(f"  4-GBM archs: {[c['arch'] for c in topK]}", flush=True)
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

        # EspalomaCharge scaler
        sc_esp = StandardScaler().fit(Xesp_tr[trn])
        Xesp_s = sc_esp.transform(Xesp_tr)

        # GNN members
        sn_oof   = np.load(f"{MTL}/sn_oof_seed{seed}.npy").ravel()
        rpxr_p   = f"{MTL}/rpxr_oof_seed{seed}.npy"
        rpxr_oof = np.load(rpxr_p).ravel() if os.path.exists(rpxr_p) else np.zeros(len(ho))

        # SOAP scaler
        soap_sc    = StandardScaler().fit(Xsoap[trn])
        soap_trn_s = soap_sc.transform(Xsoap[trn])
        soap_ho_s  = soap_sc.transform(Xsoap[ho])

        # ==== CTRL ====
        A_ctrl = np.hstack([Xtr_comb, Xqm_block])
        gbm_ctrl_ho = [fp_ho(c, A_ctrl, ytr, trn, ho) for c in topK]
        pred_ctrl_ho = np.clip(
            np.mean(gbm_ctrl_ho + [chem_oof[ho], tab_oof[ho], sn_oof, rpxr_oof], 0),
            lo, hi)

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

        # ==== TREAT (+ EspalomaCharge) ====
        A_treat = np.hstack([Xtr_comb, Xqm_block, Xesp_s])
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

        print(f"  seed={seed} | ctrl={c_rae:.4f} | treat={t_rae:.4f} | delta={t_rae-c_rae:+.4f}",
              flush=True)

    ctrl_mean  = float(np.mean(ctrl_raes))
    treat_mean = float(np.mean(treat_raes))
    delta      = treat_mean - ctrl_mean
    neg_seeds  = sum(t < c for t, c in zip(treat_raes, ctrl_raes))
    deploy     = (delta < -GATE) and (treat_mean < BEST_RAE)

    print(f"\nEspalomaCharge RESULT: ctrl={ctrl_mean:.4f} treat={treat_mean:.4f} "
          f"delta={delta:+.4f} neg_seeds={neg_seeds}/3 deploy={deploy}", flush=True)

    summary = {
        "feature": "espaloma_charge_am1bcc_10scalars",
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
    out_path = "data/processed/nb1329_espaloma_summary.json"
    json.dump(summary, open(out_path, "w"), indent=2)
    print(f"Saved: {out_path}", flush=True)
    return summary


if __name__ == "__main__":
    main()
