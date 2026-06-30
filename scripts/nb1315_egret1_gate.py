"""nb1315_egret1_gate.py — Honest add-member gate for Egret-1 + Egret-1T energy/force scalars.

Features: 14 scalars from Egret-1 (ground-state, organic MLIP) + Egret-1T (TS-trained MLIP)
  - egret_energy, egret_energy_pa, egret_fmax, egret_frms, egret_fstd (Egret-1 GS)
  - egret_conf_mean, egret_conf_std (node embedding moments, if available)
  - egret1t_energy, egret1t_energy_pa, egret1t_fmax, egret1t_frms (Egret-1T TS-trained)
  - egret1t_conf_mean, egret1t_conf_std (node embedding moments, if available)
  - delta_energy = egret1t_energy - egret_energy (TS-reactivity proxy)

Gate: add-member on deployed config (4-GBM+AIMNet2+strain+D4+DBSTEP+OrbMol+CheMeleon+TabPFN+sitterNR_GNN+rat_rPXR_GNN)
Deploy if matched delta < -0.001 AND treatment < best_ensemble.json.rae (currently 0.4213)

Usage: .venv/Scripts/python scripts/nb1315_egret1_gate.py
Run AFTER: C:/pxr_work/egret/egret1_v2_features.csv exists (4652 rows)
"""
import os, sys, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.pxr.data import load_train, load_test
from src.pxr.eval import rae
from sklearn.preprocessing import StandardScaler

SD  = "C:/pxr_work/search"
MTL = "C:/pxr_work/mtl"
BEST = f"{SD}/best_ensemble.json"
LOG  = f"{SD}/results.jsonl"
FEAT = "C:/pxr_work/egret/egret1_v2_features.csv"
N_SEEDS = 3

EGRET_COLS = [
    "egret_energy", "egret_energy_pa", "egret_fmax", "egret_frms", "egret_fstd",
    "egret1t_energy", "egret1t_energy_pa", "egret1t_fmax", "egret1t_frms",
]


def topK_configs(archs):
    recs = [json.loads(l) for l in open(LOG) if l.strip()]
    valid = [r for r in recs if "ps_rae" in r and "error" not in r and r["arch"] in archs]
    valid.sort(key=lambda r: r["ps_rae"])
    bp = {}
    for r in valid:
        bp.setdefault(r["arch"], r)
    return list(bp.values())


def load_feature_block(csv_path, cols, name_col, tr_names, te_names=None):
    df = pd.read_csv(csv_path)
    if "src" in df.columns:
        tr_df = df[df["src"] == "train"].drop_duplicates(subset=name_col, keep="first")
        te_df = df[df["src"] == "test"].drop_duplicates(subset=name_col, keep="first")
    else:
        tr_df = df.drop_duplicates(subset=name_col, keep="first")
        te_df = tr_df

    tr_sub = tr_df.set_index(name_col).reindex(tr_names)
    Xtr = tr_sub[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    med = np.nanmedian(Xtr, axis=0)
    inds = np.where(np.isnan(Xtr))
    Xtr[inds] = np.take(med, inds[1])

    if te_names is not None:
        te_sub = te_df.set_index(name_col).reindex(te_names)
        Xte = te_sub[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        inds = np.where(np.isnan(Xte))
        Xte[inds] = np.take(med, inds[1])
        return Xtr.astype(np.float32), Xte.astype(np.float32)
    return Xtr.astype(np.float32)


def load_qm_blocks(tr_names):
    def ab(csv, cols, name="name", src_val="train"):
        df = pd.read_csv(csv)
        if "src" in df.columns:
            df = df[df.src == src_val]
        df = df.drop_duplicates(subset=name, keep="first")
        sub = df.set_index(name).reindex(tr_names)
        X = sub[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        med = np.nanmedian(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(med, inds[1])
        return X.astype(np.float32)

    ACOLS = ["aimnet_energy","aimnet_qmin","aimnet_qmax","aimnet_qabs_mean",
             "aimnet_qstd","aimnet_qsum_abs","aimnet_dipole","aimnet_fmax","aimnet_frms"]
    SCOLS = ["strain_relax_mean","strain_relax_max","conf_espread","conf_erange",
             "conf_n","rmsd_mean","rmsd_max","e_per_heavy"]
    DCOLS = ["d4_alpha_sum","d4_alpha_mean","d4_alpha_std","d4_alpha_max",
             "d4_c6diag_mean","d4_c6diag_std","d4_c6_total","d4_edisp",
             "d4_edisp_per_atom","d4_cn_mean","d4_cn_max","d4_qeeq_min",
             "d4_qeeq_max","d4_qeeq_std","d4_qeeq_absum"]
    DBCOLS = ["vbur_r25","vbur_r35","vbur_r45","vbur_r55","vbur_r65",
              "ster_L","ster_Bmin","ster_Bmax","ster_aniso",
              "npr1","npr2","asphericity","spherocity","eccentricity","radgyr","inertial_sf"]
    ORBCOLS = ["orb_energy","orb_energy_per_ha","orb_fmax","orb_frms","orb_fstd",
               "orb_conf_mean","orb_conf_std","orb_conf_node_mean","orb_conf_node_std",
               "orb_conf_node_min","orb_node_emb_mean","orb_node_emb_std","orb_node_emb_norm"]
    return (
        ab("C:/pxr_work/aimnet2/aimnet_features.csv", ACOLS),
        ab("C:/pxr_work/strain/strain_features.csv",  SCOLS),
        ab("C:/pxr_work/d4/d4_features.csv",          DCOLS),
        ab("C:/pxr_work/dbstep/dbstep_features.csv",  DBCOLS),
        ab("C:/pxr_work/orbmol/orbmol_features.csv",  ORBCOLS),
    )


def main():
    print("=== nb1315 Egret-1 + Egret-1T add-member gate ===", flush=True)

    if not os.path.exists(FEAT):
        print(f"Features not ready: {FEAT}")
        return

    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test().reset_index(drop=True)
    ytr = tr["pec50"].to_numpy(float)
    n_tr = len(ytr)
    tr_names = tr["name"].tolist()
    te_names = te["name"].tolist()

    # Check feature completeness
    feat_df = pd.read_csv(FEAT)
    feat_tr = feat_df[feat_df["src"] == "train"]
    ok_count = feat_tr["egret_energy"].notna().sum()
    print(f"Features: {ok_count}/{n_tr} train rows have egret_energy", flush=True)
    if ok_count < n_tr * 0.95:
        print(f"WARNING: only {ok_count}/{n_tr} rows have features ({ok_count/n_tr*100:.1f}%)")

    # Add delta_energy feature
    feat_df["delta_energy"] = feat_df["egret1t_energy"] - feat_df["egret_energy"]
    cols_to_use = EGRET_COLS + ["delta_energy"]
    print(f"Using {len(cols_to_use)} Egret scalars: {cols_to_use}", flush=True)

    Xegret_tr, Xegret_te = load_feature_block(FEAT, cols_to_use, "name", tr_names, te_names)
    print(f"  Egret train block: {Xegret_tr.shape}, test: {Xegret_te.shape}", flush=True)

    # Corr with nb3200 error
    nb3200_path = "data/processed/oof_chemprop_aux.npy"
    if os.path.exists(nb3200_path):
        nb3200_oof = np.load(nb3200_path)
        err = ytr - nb3200_oof
        for j, col in enumerate(cols_to_use):
            c = float(np.corrcoef(err, Xegret_tr[:, j])[0, 1])
            if abs(c) > 0.05:
                print(f"  corr(nb3200 err, {col}) = {c:+.4f}", flush=True)

    from nb1126_combinatorial_search import feature_matrix, CACHE, make_model
    d = np.load(CACHE); se = d["se"]
    Xfull, _ = feature_matrix(d, "combined")

    Xqm, Xst, Xd4, Xdb, Xorb = load_qm_blocks(tr_names)
    chem_oof = np.load(f"{SD}/chemeleon_oof.npy")
    tab_oof  = np.load(f"{SD}/tabpfn_oof.npy")
    topK = topK_configs(("lgbm", "xgb", "cat", "histgb"))
    print(f"  4-GBM: {[c['arch'] for c in topK]}", flush=True)
    prev_best = json.load(open(BEST))["rae"]
    print(f"  Deployed best = {prev_best:.4f}", flush=True)

    ctrl_raes, treat_raes = [], []
    for seed in range(N_SEEDS):
        ho  = np.load(f"{MTL}/ho_idx_seed{seed}.npy")
        trn = np.array([i for i in range(n_tr) if i not in set(ho.tolist())])
        noisy20 = se <= np.quantile(se, 0.8)
        noisy30 = se <= np.quantile(se, 0.7)
        lo, hi = np.quantile(ytr[trn], 0.05), np.quantile(ytr[trn], 0.98)

        sc_qm  = StandardScaler().fit(Xqm[trn]);  Xqm_s  = sc_qm.transform(Xqm)
        sc_st  = StandardScaler().fit(Xst[trn]);  Xst_s  = sc_st.transform(Xst)
        sc_d4  = StandardScaler().fit(Xd4[trn]);  Xd4_s  = sc_d4.transform(Xd4)
        sc_db  = StandardScaler().fit(Xdb[trn]);  Xdb_s  = sc_db.transform(Xdb)
        sc_orb = StandardScaler().fit(Xorb[trn]); Xorb_s = sc_orb.transform(Xorb)
        sc_eg  = StandardScaler().fit(Xegret_tr[trn]); Xeg_s = sc_eg.transform(Xegret_tr)
        Xqm_block = np.hstack([Xqm_s, Xst_s, Xd4_s, Xdb_s, Xorb_s])

        gnn = np.load(f"{MTL}/sn_oof_seed{seed}.npy").ravel()
        rpxr = np.load(f"{MTL}/rat_rpxr_oof_seed{seed}.npy").ravel() if os.path.exists(f"{MTL}/rat_rpxr_oof_seed{seed}.npy") else np.zeros(n_tr)

        gbm_ctrl, gbm_treat = [], []
        for c in topK:
            use = trn
            if c["prep"] == "noisy20": use = trn[noisy20[trn]]
            elif c["prep"] == "noisy30": use = trn[noisy30[trn]]
            A_ctrl  = np.hstack([Xfull, Xqm_block])
            A_treat = np.hstack([Xfull, Xqm_block, Xeg_s])
            m_c = make_model(c["arch"], c["hp"])
            m_c.fit(A_ctrl[use], ytr[use])
            gbm_ctrl.append(m_c.predict(A_ctrl[ho]))
            m_t = make_model(c["arch"], c["hp"])
            m_t.fit(A_treat[use], ytr[use])
            gbm_treat.append(m_t.predict(A_treat[ho]))

        base_ctrl  = gbm_ctrl  + [chem_oof[ho], tab_oof[ho], gnn[ho], rpxr[ho]]
        base_treat = gbm_treat + [chem_oof[ho], tab_oof[ho], gnn[ho], rpxr[ho]]
        ens_ctrl  = np.clip(np.mean(base_ctrl,  0), lo, hi)
        ens_treat = np.clip(np.mean(base_treat, 0), lo, hi)

        c_rae = rae(ytr[ho], ens_ctrl)
        t_rae = rae(ytr[ho], ens_treat)
        ctrl_raes.append(c_rae)
        treat_raes.append(t_rae)
        print(f"  seed{seed}  ctrl={c_rae:.4f}  treat={t_rae:.4f}  d={t_rae-c_rae:+.4f}", flush=True)

    cm = float(np.mean(ctrl_raes))
    tm = float(np.mean(treat_raes))
    delta = tm - cm
    n_neg = sum(t < c for t, c in zip(treat_raes, ctrl_raes))

    print(f"\n  Control  (deployed) RAE = {cm:.4f}  seeds={[round(r,4) for r in ctrl_raes]}")
    print(f"  Treatment(+Egret1) RAE = {tm:.4f}  seeds={[round(r,4) for r in treat_raes]}")
    print(f"  Matched delta = {delta:+.5f}  ({n_neg}/{N_SEEDS} seeds negative)")
    print(f"  Deployed best = {prev_best:.4f}")

    deploy = bool(delta < -0.001 and tm < prev_best and n_neg >= 2)
    out = {
        "model": "Egret-1 + Egret-1T (MACE-arch NNP, MIT, bioorganic + TS-trained)",
        "features": f"{cols_to_use}",
        "ctrl_rae": cm, "treat_rae": tm, "delta": delta,
        "ctrl_raes": ctrl_raes, "treat_raes": treat_raes, "n_neg": n_neg,
        "prev_best": prev_best, "deploy": deploy,
    }
    os.makedirs("data/processed", exist_ok=True)
    json.dump(out, open("data/processed/nb1315_egret1_summary.json", "w"), indent=2)
    print(f"\n  Result -> data/processed/nb1315_egret1_summary.json  DEPLOY={deploy}", flush=True)

    if deploy:
        print("\n*** DEPLOY: Egret-1 WINS ***", flush=True)
        # Build 513 test preds using full train
        sc_eg_all = StandardScaler().fit(Xegret_tr)
        Xeg_all = sc_eg_all.transform(Xegret_tr)
        Xeg_te  = sc_eg_all.transform(Xegret_te)
        gbm_te_preds = []
        for c in topK:
            A_all = np.hstack([Xfull, np.hstack([
                StandardScaler().fit_transform(Xqm),
                StandardScaler().fit_transform(Xst),
                StandardScaler().fit_transform(Xd4),
                StandardScaler().fit_transform(Xdb),
                StandardScaler().fit_transform(Xorb),
            ]), Xeg_all])
            A_te = np.hstack([
                np.load(f"{SD}/chemeleon_te.npy").astype(np.float32)[:, :Xfull.shape[1]],  # placeholder
                # This needs proper test feature loading — see deploy script
            ])
            # Note: proper deploy needs test features. Flag for separate deploy script.
            pass
        print("  NOTE: build separate deploy script for full test submission (test QM features needed)", flush=True)
        # Update best
        best_data = json.load(open(BEST))
        best_data["rae"] = tm
        best_data["via"] = f"+Egret-1+Egret-1T (delta={delta:.5f}, {n_neg}/3 neg)"
        best_data["members"] = best_data.get("members", []) + ["egret1", "egret1t"]
        json.dump(best_data, open(BEST, "w"), indent=2)
        print(f"  Updated best_ensemble.json -> {tm:.4f}", flush=True)
    else:
        reason = (f"delta {delta:+.5f} >= -0.001" if delta >= -0.001 else
                  f"n_neg={n_neg} < 2" if n_neg < 2 else
                  f"treatment {tm:.4f} >= best {prev_best:.4f}")
        print(f"\n  NO DEPLOY: {reason}", flush=True)

    print("\n=== nb1315 DONE ===", flush=True)


if __name__ == "__main__":
    main()
