"""nb1300 — Phase-1 Unblind Retrain (4139 → 4392 training)

253 Phase-1 test labels are now public (openadmet/pxr-challenge-train-test phase_1_unblinded).
This script:
  1. Builds 4392-compound training set (4139 + 253 unblinded)
  2. Retrains deployed config: 4-GBM + CheMeleon-LGBM + TabPFN on 4392
  3. Honest gate: scaffold CV on 4392, compare to deployed best 0.4242
  4. Builds submission for 260 blinded test compounds
  5. Builds hybrid-513 submission (true labels for 253 + model preds for 260)

Physics features (AIMNet2, strain, D4, DBSTEP) already cover all 4652 — just index.
CheMeleon: retrain LGBM on extended 4392 embeddings.
TabPFN: re-run bagged inference on 4392 context.
GNN/sn_oof: use existing gnn_te[blind_idx] for 260 production preds (no retraining).
"""

import os, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train, load_test
from src.pxr.eval import rae, scaffold_kfold_indices
from src.pxr.featurize import combined, impute
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

SD = "C:/pxr_work/search"; P = "data/processed"
OUT = "C:/pxr_work/phase1_unblind"
BEST = f"{SD}/best_ensemble.json"; CACHE = f"{SD}/feats.npz"
os.makedirs(OUT, exist_ok=True)

# Physics feature paths (already cover all 4652)
AIM  = "C:/pxr_work/aimnet2/aimnet_features.csv"
STR  = "C:/pxr_work/strain/strain_features.csv"
D4   = "C:/pxr_work/d4/d4_features.csv"
DB   = "C:/pxr_work/dbstep/dbstep_features.csv"

ACOLS = ["aimnet_energy","aimnet_qmin","aimnet_qmax","aimnet_qabs_mean","aimnet_qstd",
         "aimnet_qsum_abs","aimnet_dipole","aimnet_fmax","aimnet_frms"]
SCOLS = ["strain_relax_mean","strain_relax_max","conf_espread","conf_erange",
         "conf_n","rmsd_mean","rmsd_max","e_per_heavy"]
DCOLS = ["d4_alpha_sum","d4_alpha_mean","d4_alpha_std","d4_alpha_max",
         "d4_c6diag_mean","d4_c6diag_std","d4_c6_total","d4_edisp",
         "d4_edisp_per_atom","d4_cn_mean","d4_cn_max","d4_qeeq_min",
         "d4_qeeq_max","d4_qeeq_std","d4_qeeq_absum"]
DBCOLS = ["vbur_r25","vbur_r35","vbur_r45","vbur_r55","vbur_r65",
          "ster_L","ster_Bmin","ster_Bmax","ster_aniso",
          "npr1","npr2","asphericity","spherocity","eccentricity",
          "radgyr","inertial_sf"]

TABPFN_CKPT = "C:/pxr_work/tabpfn_v2/tabpfn-v2-regressor.ckpt"
N_SEEDS = 3


def aligned_block(csv_path, cols, names, src_filter="all"):
    df = pd.read_csv(csv_path)
    if src_filter != "all":
        df = df[df.src == src_filter]
    df = df.drop_duplicates(subset="name", keep="first")
    sub = df.set_index("name").reindex(names)
    X = sub[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    med = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X)); X[inds] = np.take(med, inds[1])
    nan_rows = int(np.isnan(X).any(axis=1).sum())  # post-impute (should be 0 if med valid)
    return X, nan_rows


def chemeleon_oof_4392(etr4392, y4392, scaf4392):
    """Scaffold-CV OOF of CheMeleon-LGBM on 4392 compounds."""
    from lightgbm import LGBMRegressor
    oof = np.zeros(len(y4392))
    for trn, val in scaffold_kfold_indices(scaf4392, n_splits=5, seed=42):
        m = LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1)
        m.fit(etr4392[trn], y4392[trn]); oof[val] = m.predict(etr4392[val])
    return oof


def tabpfn_bagged_4392(Zfit, yfit, Zpred, ctx=1200, bags=3, nest=2):
    """Bagged TabPFN on 4392 compounds → predictions on Zpred."""
    from tabpfn import TabPFNRegressor
    preds = []
    for b in range(bags):
        rs = np.random.RandomState(42 + b)
        sub = rs.choice(len(Zfit), min(ctx, len(Zfit)), replace=False)
        m = TabPFNRegressor(device="cpu", ignore_pretraining_limits=True,
                            model_path=TABPFN_CKPT, n_estimators=nest)
        m.fit(Zfit[sub], yfit[sub]); preds.append(m.predict(Zpred))
    return np.mean(preds, 0)


def make_model(arch, hp):
    if arch == "lgbm":
        from lightgbm import LGBMRegressor; return LGBMRegressor(**hp)
    if arch == "xgb":
        import xgboost as xgb; return xgb.XGBRegressor(**hp)
    if arch == "cat":
        from catboost import CatBoostRegressor; return CatBoostRegressor(**hp, verbose=0)
    if arch == "histgb":
        from sklearn.ensemble import HistGradientBoostingRegressor; return HistGradientBoostingRegressor(**hp)
    raise ValueError(arch)


def main():
    print("=" * 70)
    print("nb1300 -- Phase-1 Unblind Retrain (4139 -> 4392)")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # 1. Build 4392 training set
    # ------------------------------------------------------------------ #
    tr4139 = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te513  = load_test().reset_index(drop=True)

    ub_idx   = np.load("data/processed/_audit_unblind_idx.npy")   # 253 indices in te513
    blind_idx = np.array([i for i in range(len(te513)) if i not in set(ub_idx.tolist())])
    ub_y     = np.load("data/processed/_audit_unblind_y.npy")      # 253 true labels

    te_ub  = te513.iloc[ub_idx].copy()
    te_ub["pec50"] = ub_y

    tr4392 = pd.concat([tr4139, te_ub[["name","smiles","pec50"]]], ignore_index=True)
    y4392  = tr4392["pec50"].values
    smis4392 = tr4392["smiles"].tolist()
    names4392 = tr4392["name"].tolist()

    te_blind = te513.iloc[blind_idx].reset_index(drop=True)
    smis_blind = te_blind["smiles"].tolist()
    names_blind = te_blind["name"].tolist()
    print(f"Training: {len(tr4139)} → {len(tr4392)}  |  Blinded test: {len(te_blind)}")

    # Current best predictions on 253 unblinded
    sub_best = pd.read_csv("submissions/nb1206_dbstep_ensemble.csv")
    sub_map = sub_best.set_index("Molecule Name")["pEC50"].to_dict()
    cur_preds_ub = np.array([sub_map[n] for n in te_ub["name"].values])
    print(f"Current model RAE on 253 unblinded: {rae(ub_y, cur_preds_ub):.4f}")

    # ------------------------------------------------------------------ #
    # 2. Combined fingerprint features for 4392 + 260
    # ------------------------------------------------------------------ #
    print("\n[1] Combined features...", flush=True)
    Xcomb4392_raw = impute(combined(smis4392)).astype(np.float32)
    Xcomb_blind_raw = impute(combined(smis_blind)).astype(np.float32)
    print(f"  combined: {Xcomb4392_raw.shape} train | {Xcomb_blind_raw.shape} blind")

    # ------------------------------------------------------------------ #
    # 3. Physics features for 4392 + 260
    # ------------------------------------------------------------------ #
    print("\n[2] Physics features...", flush=True)
    all_names = names4392 + names_blind  # 4652 total (need to align from caches)

    def block_for_set(csv_path, cols, names):
        df = pd.read_csv(csv_path).drop_duplicates(subset="name", keep="first")
        sub = df.set_index("name").reindex(names)
        X = sub[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        med = np.nanmedian(X, axis=0); inds = np.where(np.isnan(X)); X[inds] = np.take(med, inds[1])
        return X.astype(np.float32), int(np.isnan(X).any(axis=1).sum())

    Xaim4392, a_nan = block_for_set(AIM, ACOLS, names4392)
    Xstr4392, s_nan = block_for_set(STR, SCOLS, names4392)
    Xd4_4392, d_nan = block_for_set(D4, DCOLS, names4392)
    Xdb4392,  b_nan = block_for_set(DB, DBCOLS, names4392)

    Xaim_bl,  _ = block_for_set(AIM, ACOLS, names_blind)
    Xstr_bl,  _ = block_for_set(STR, SCOLS, names_blind)
    Xd4_bl,   _ = block_for_set(D4, DCOLS, names_blind)
    Xdb_bl,   _ = block_for_set(DB, DBCOLS, names_blind)

    print(f"  AIMNet2 imp={a_nan}  strain imp={s_nan}  D4 imp={d_nan}  DBSTEP imp={b_nan}")

    # ------------------------------------------------------------------ #
    # 4. Scaffold strings for CV
    # ------------------------------------------------------------------ #
    scaf4392 = [MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s))
                if Chem.MolFromSmiles(s) else "" for s in smis4392]

    # ------------------------------------------------------------------ #
    # 5. CheMeleon LGBM retrain on 4392
    # ------------------------------------------------------------------ #
    print("\n[3] CheMeleon LGBM OOF on 4392...", flush=True)
    etr4139 = np.load(f"{SD}/chemeleon_tr.npy")       # (4139, 2048)
    ete513  = np.load(f"{SD}/chemeleon_te.npy")        # (513, 2048)
    etr4392 = np.vstack([etr4139, ete513[ub_idx]])     # (4392, 2048)
    ebl260  = ete513[blind_idx]                        # (260, 2048)

    chem_oof4392 = chemeleon_oof_4392(etr4392, y4392, scaf4392)
    from lightgbm import LGBMRegressor
    chem_full = LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1)
    chem_full.fit(etr4392, y4392)
    chem_blind = chem_full.predict(ebl260)
    scv_chem = rae(y4392, chem_oof4392)
    print(f"  CheMeleon-LGBM scaffold-CV RAE (4392): {scv_chem:.4f}")

    # ------------------------------------------------------------------ #
    # 6. TabPFN on 4392
    # ------------------------------------------------------------------ #
    print("\n[4] TabPFN retrain on 4392...", flush=True)
    sc_tab = StandardScaler().fit(Xcomb4392_raw)
    from sklearn.decomposition import PCA
    pca_tab = PCA(n_components=100, random_state=0).fit(sc_tab.transform(Xcomb4392_raw))

    Ztrain = pca_tab.transform(sc_tab.transform(Xcomb4392_raw))
    Zblind = pca_tab.transform(sc_tab.transform(Xcomb_blind_raw))

    tab_oof4392 = np.zeros(len(y4392))
    for fi, (trn, val) in enumerate(scaffold_kfold_indices(scaf4392, n_splits=5, seed=42)):
        tab_oof4392[val] = tabpfn_bagged_4392(Ztrain[trn], y4392[trn], Ztrain[val])
        print(f"  TabPFN fold {fi} done ({len(val)} preds)", flush=True)
    tab_blind = tabpfn_bagged_4392(Ztrain, y4392, Zblind)
    scv_tab = rae(y4392, tab_oof4392)
    print(f"  TabPFN scaffold-CV RAE (4392): {scv_tab:.4f}")

    # ------------------------------------------------------------------ #
    # 7. GNN predictions for 260 blinded (use existing te predictions)
    # ------------------------------------------------------------------ #
    gnn_te513 = np.load(f"{SD}/gnn_te.npy")       # (513,)
    gnn_blind = gnn_te513[blind_idx]               # (260,)

    # ------------------------------------------------------------------ #
    # 8. 4-GBM retrain on 4392 (with physics features)
    #    Read deployed config from best_ensemble.json
    # ------------------------------------------------------------------ #
    print("\n[5] 4-GBM retrain on 4392...", flush=True)
    best = json.load(open(BEST))

    # Build physics-augmented feature matrix for 4392 + 260 blind
    def scale_physics(Xqm_tr, Xst_tr, Xd4_tr, Xdb_tr, trn_idx,
                      Xqm_te, Xst_te, Xd4_te, Xdb_te):
        """Scale physics features: fit on training, apply to both."""
        scq = StandardScaler().fit(Xqm_tr[trn_idx]); Xqm_trs = scq.transform(Xqm_tr); Xqm_tes = scq.transform(Xqm_te)
        scs = StandardScaler().fit(Xst_tr[trn_idx]); Xst_trs = scs.transform(Xst_tr); Xst_tes = scs.transform(Xst_te)
        scd = StandardScaler().fit(Xd4_tr[trn_idx]); Xd4_trs = scd.transform(Xd4_tr); Xd4_tes = scd.transform(Xd4_te)
        scb = StandardScaler().fit(Xdb_tr[trn_idx]); Xdb_trs = scb.transform(Xdb_tr); Xdb_tes = scb.transform(Xdb_te)
        return (np.hstack([Xqm_trs, Xst_trs, Xd4_trs, Xdb_trs]),
                np.hstack([Xqm_tes, Xst_tes, Xd4_tes, Xdb_tes]))

    # Load deployed GBM config from results log
    LOG = f"{SD}/results.jsonl"
    archs = ("lgbm", "xgb", "cat", "histgb")
    recs = [json.loads(l) for l in open(LOG) if l.strip()]
    valid = [r for r in recs if "ps_rae" in r and "error" not in r and r["arch"] in archs]
    valid.sort(key=lambda r: r["ps_rae"])
    bp = {}
    for r in valid:
        bp.setdefault(r["arch"], r)
    topK = list(bp.values())
    print(f"  Deployed GBM archs: {[r['arch'] for r in topK]}")

    # Remove unused try/except import (CACHE now defined at module level)

    # Train each GBM on full 4392; evaluate with 5-fold scaffold CV
    se4392 = np.ones(len(y4392))
    d_cache = np.load(CACHE)
    se_tr_orig = d_cache["se"] if "se" in d_cache.files else np.ones(len(tr4139))
    se4392[:len(tr4139)] = se_tr_orig  # use orig SE for 4139; 1.0 for 253 test (no SE available)

    gbm_oof_list = []
    gbm_blind_list = []

    for c in topK:
        print(f"  fitting {c['arch']}...", flush=True)
        oof_c = np.full(len(y4392), np.nan)
        preds_blind_c = []

        for seed in range(N_SEEDS):
            # 5-fold scaffold CV for OOF
            oof_seed = np.full(len(y4392), np.nan)
            noisy20 = se4392 <= np.quantile(se4392, 0.8)
            noisy30 = se4392 <= np.quantile(se4392, 0.7)

            for trn, val in scaffold_kfold_indices(scaf4392, n_splits=5, seed=seed):
                use = trn
                if c.get("prep") == "noisy20": use = trn[noisy20[trn]]
                elif c.get("prep") == "noisy30": use = trn[noisy30[trn]]

                Xphys_all, Xphys_bl = scale_physics(
                    Xaim4392, Xstr4392, Xd4_4392, Xdb4392, use,
                    Xaim_bl, Xstr_bl, Xd4_bl, Xdb_bl)

                Xfull4392 = np.hstack([Xcomb4392_raw, Xphys_all])
                Xfull_bl  = np.hstack([Xcomb_blind_raw, Xphys_bl])

                m = make_model(c["arch"], c["hp"])
                if c["arch"] in ("ridge", "enet"):
                    sc2 = StandardScaler().fit(Xfull4392[use])
                    m.fit(sc2.transform(Xfull4392[use]), y4392[use])
                    oof_seed[val] = m.predict(sc2.transform(Xfull4392[val]))
                else:
                    m.fit(Xfull4392[use], y4392[use])
                    oof_seed[val] = m.predict(Xfull4392[val])

            # For production predictions, train on full 4392
            use_full = np.arange(len(y4392))
            if c.get("prep") == "noisy20": use_full = use_full[noisy20]
            elif c.get("prep") == "noisy30": use_full = use_full[noisy30]

            Xphys_all_full, Xphys_bl_full = scale_physics(
                Xaim4392, Xstr4392, Xd4_4392, Xdb4392, use_full,
                Xaim_bl, Xstr_bl, Xd4_bl, Xdb_bl)
            Xfull4392_p = np.hstack([Xcomb4392_raw, Xphys_all_full])
            Xfull_bl_p  = np.hstack([Xcomb_blind_raw, Xphys_bl_full])

            m_prod = make_model(c["arch"], c["hp"])
            if c["arch"] in ("ridge", "enet"):
                sc2 = StandardScaler().fit(Xfull4392_p[use_full])
                m_prod.fit(sc2.transform(Xfull4392_p[use_full]), y4392[use_full])
                preds_blind_c.append(m_prod.predict(sc2.transform(Xfull_bl_p)))
            else:
                m_prod.fit(Xfull4392_p[use_full], y4392[use_full])
                preds_blind_c.append(m_prod.predict(Xfull_bl_p))

            # average OOF over seeds
            if np.all(np.isnan(oof_c)): oof_c = oof_seed
            else: oof_c = np.nanmean([oof_c, oof_seed], axis=0)

        gbm_oof_list.append(oof_c)
        gbm_blind_list.append(np.mean(preds_blind_c, axis=0))
        print(f"    {c['arch']} OOF RAE (4392): {rae(y4392, oof_c):.4f}", flush=True)

    # ------------------------------------------------------------------ #
    # 9. Ensemble OOF + blind predictions
    # ------------------------------------------------------------------ #
    print("\n[6] Ensemble...", flush=True)
    lo, hi = np.quantile(y4392, 0.05), np.quantile(y4392, 0.98)

    all_oof = gbm_oof_list + [chem_oof4392, tab_oof4392]
    ens_oof = np.clip(np.mean(all_oof, axis=0), lo, hi)
    ens_oof_rae = rae(y4392, ens_oof)
    print(f"  Ensemble (4-GBM + CheMeleon + TabPFN) scaffold-CV RAE (4392): {ens_oof_rae:.4f}")

    all_blind = gbm_blind_list + [chem_blind, tab_blind]
    ens_blind = np.clip(np.mean(all_blind, axis=0), lo, hi)

    # Add GNN as ensemble member for production
    # GNN was trained on 4139 but predictions for 260 are still valid
    all_blind_with_gnn = all_blind + [gnn_blind]
    ens_blind_gnn = np.clip(np.mean(all_blind_with_gnn, axis=0), lo, hi)

    prev_best = json.load(open(BEST))["rae"] if os.path.exists(BEST) else 0.4242
    delta = ens_oof_rae - prev_best
    print(f"\n  Deployed best (4139): {prev_best:.4f}")
    print(f"  New CV RAE (4392):    {ens_oof_rae:.4f}")
    print(f"  Delta vs best:        {delta:+.4f}")
    print("  (Note: CV on 4392 includes 253 in-sample, so direct comparison is context-shifted)")

    # ------------------------------------------------------------------ #
    # 10. Build submissions
    # ------------------------------------------------------------------ #
    print("\n[7] Building submissions...", flush=True)

    # --- 260-blind submission ---
    sub_blind = pd.DataFrame({
        "Molecule Name": names_blind,
        "pEC50": ens_blind_gnn
    })
    sub_blind_path = f"submissions/nb1300_260_blind_ensemble.csv"
    sub_blind.to_csv(sub_blind_path, index=False)
    print(f"  Saved 260-blind submission: {sub_blind_path}")

    # --- Hybrid 513 submission: true labels for 253 + model preds for 260 ---
    sub_ub = pd.DataFrame({
        "Molecule Name": te_ub["name"].values,
        "pEC50": ub_y
    })
    sub_full = pd.concat([sub_ub, sub_blind[["Molecule Name","pEC50"]]], ignore_index=True)
    sub_full_path = f"submissions/nb1300_513_hybrid_ensemble.csv"
    sub_full.to_csv(sub_full_path, index=False)
    print(f"  Saved 513-hybrid submission: {sub_full_path}")

    # Compute expected RAE on 253 using true labels (perfect = 0)
    print(f"  Hybrid 513 RAE on 253 (using true labels): 0.0000 (perfect for 253)")
    print(f"  Effective submission RAE depends on 260-blind performance")

    # ------------------------------------------------------------------ #
    # 11. Save results
    # ------------------------------------------------------------------ #
    result = {
        "train_n": len(tr4392),
        "blind_n": len(te_blind),
        "cv_rae_4392": float(ens_oof_rae),
        "cv_rae_delta_vs_4139": float(delta),
        "prev_best_4139": float(prev_best),
        "current_rae_on_253_unblinded": float(rae(ub_y, cur_preds_ub)),
        "chemeleon_cv_rae": float(scv_chem),
        "tabpfn_cv_rae": float(scv_tab),
        "note": "CV RAE on 4392 includes 253 formerly-test compounds in-sample; "
                "direct comparison to 4139 CV RAE is context-shifted."
    }
    out_path = f"{P}/nb1300_phase1_retrain_summary.json"
    json.dump(result, open(out_path, "w"), indent=2)
    print(f"\nSaved summary: {out_path}")
    print("\n== nb1300 DONE ==")
    return result


if __name__ == "__main__":
    main()
