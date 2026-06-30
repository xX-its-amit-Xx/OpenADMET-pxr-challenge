"""nb1310 — COMBINATOR tick: two non-standard compositions on the OrbMol deployed config.

COMP-G: QM-only physics GBM as 8th ensemble member.
  Train LightGBM on ONLY the 61 physics features (AIMNet2+strain+D4+DBSTEP+OrbMol),
  no combined fingerprints. The 4 deployed GBMs are dominated by combined(2265 cols)
  and may under-utilize the QM signal. A physics-only GBM forces the model to learn
  purely from QM/physics space — potentially orthogonal to the fingerprint-dominated GBMs.
  Add as 8th member at equal weight. Expected diversity: QM-only GBM should correlate
  ~0.75-0.85 with combined GBMs (better than TabICL's 0.97 with TabPFN).

COMP-H: MAE-loss LightGBM as 5th GBM member (L1 objective aligned with RAE metric).
  Current 4 GBMs all train with MSE (L2) loss. The target metric (RAE) is MAE-based.
  Training with objective='mae' directly aligns the loss surface with the evaluation metric.
  Add the best-hp LGBM but with loss='mae', trained on combined+QM (same as deployed GBMs),
  as an extra 5th GBM member → 8-member ensemble {lgbm_l2, lgbm_mae, xgb, cat, histgb,
  sn_gnn, chemeleon, tabpfn}.

Gate = matched delta on 3-seed MTL holdouts. Deploy if delta < -0.001 AND RAE < 0.4231.
"""
import os, sys, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb1126_combinatorial_search import feature_matrix, CACHE, make_model
from src.pxr.data import load_train, load_test
from src.pxr.eval import rae
from sklearn.preprocessing import StandardScaler
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

P   = "data/processed"
SD  = "C:/pxr_work/search"
MTL = "C:/pxr_work/mtl"
BEST = f"{SD}/best_ensemble.json"
LOG  = f"{SD}/results.jsonl"
N_SEEDS = 3

AIM    = "C:/pxr_work/aimnet2/aimnet_features.csv"
STR    = "C:/pxr_work/strain/strain_features.csv"
D4     = "C:/pxr_work/d4/d4_features.csv"
DB     = "C:/pxr_work/dbstep/dbstep_features.csv"
ORB    = "C:/pxr_work/orbmol/orbmol_features.csv"

ACOLS  = ["aimnet_energy","aimnet_qmin","aimnet_qmax","aimnet_qabs_mean","aimnet_qstd",
          "aimnet_qsum_abs","aimnet_dipole","aimnet_fmax","aimnet_frms"]
SCOLS  = ["strain_relax_mean","strain_relax_max","conf_espread","conf_erange",
          "conf_n","rmsd_mean","rmsd_max","e_per_heavy"]
DCOLS  = ["d4_alpha_sum","d4_alpha_mean","d4_alpha_std","d4_alpha_max","d4_c6diag_mean",
          "d4_c6diag_std","d4_c6_total","d4_edisp","d4_edisp_per_atom","d4_cn_mean",
          "d4_cn_max","d4_qeeq_min","d4_qeeq_max","d4_qeeq_std","d4_qeeq_absum"]
DBCOLS = ["vbur_r25","vbur_r35","vbur_r45","vbur_r55","vbur_r65",
          "ster_L","ster_Bmin","ster_Bmax","ster_aniso",
          "npr1","npr2","asphericity","spherocity","eccentricity","radgyr","inertial_sf"]
ORBCOLS= ["orb_energy","orb_energy_per_ha","orb_fmax","orb_frms","orb_fstd",
          "orb_conf_mean","orb_conf_std","orb_conf_node_mean","orb_conf_node_std",
          "orb_conf_node_min","orb_node_emb_mean","orb_node_emb_std","orb_node_emb_norm"]

QM_ALL = ACOLS + SCOLS + DCOLS + DBCOLS + ORBCOLS  # 61 physics features


def topK_configs():
    recs = [json.loads(l) for l in open(LOG) if l.strip()]
    valid = [r for r in recs if "ps_rae" in r and "error" not in r
             and r["arch"] in ("lgbm","xgb","cat","histgb")]
    valid.sort(key=lambda r: r["ps_rae"]); bp = {}
    for r in valid: bp.setdefault(r["arch"], r)
    return list(bp.values())


def best_lgbm_hp():
    recs = [json.loads(l) for l in open(LOG) if l.strip()]
    valid = [r for r in recs if "ps_rae" in r and "error" not in r and r["arch"] == "lgbm"]
    valid.sort(key=lambda r: r["ps_rae"])
    return valid[0]["hp"]


def block(df, names, cols):
    df = df.drop_duplicates(subset="name", keep="first")
    sub = df.set_index("name").reindex(names)
    X = sub[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    med = np.nanmedian(X, axis=0); inds = np.where(np.isnan(X)); X[inds] = np.take(med, inds[1])
    return X


def fit_pred(arch, hp, Xtrain, ytrain, use_idx, Xtest):
    m = make_model(arch, hp)
    m.fit(Xtrain[use_idx], ytrain[use_idx])
    return m.predict(Xtest)


def main():
    t0 = time.time()
    print("nb1310 — COMBINATOR: COMP-G (QM-only GBM) + COMP-H (MAE-loss GBM)")
    print("=" * 70)

    d   = np.load(CACHE); ytr = d["ytr"]; se = d["se"]; n_tr = len(ytr)
    Xtr, _ = feature_matrix(d, "combined")  # (4139, 2265)
    chem   = np.load(f"{SD}/chemeleon_oof.npy")
    tab    = np.load(f"{SD}/tabpfn_oof.npy")

    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    assert len(tr) == n_tr, f"Train size mismatch: {len(tr)} vs {n_tr}"

    # Load all QM blocks
    adf = pd.read_csv(AIM); sdf = pd.read_csv(STR)
    ddf = pd.read_csv(D4);  bdf = pd.read_csv(DB); odf = pd.read_csv(ORB)
    Xqm  = block(adf[adf.src=="train"], tr["name"], ACOLS)
    Xst  = block(sdf[sdf.src=="train"], tr["name"], SCOLS)
    Xd4  = block(ddf[ddf.src=="train"], tr["name"], DCOLS)
    Xdb  = block(bdf[bdf.src=="train"], tr["name"], DBCOLS)
    Xorb = block(odf[odf.src=="train"], tr["name"], ORBCOLS)
    Xphys_raw = np.hstack([Xqm, Xst, Xd4, Xdb, Xorb])   # (4139, 61)
    print(f"Physics block: {Xphys_raw.shape[1]} cols "
          f"(AIM={len(ACOLS)} STR={len(SCOLS)} D4={len(DCOLS)} DB={len(DBCOLS)} ORB={len(ORBCOLS)})")

    topK  = topK_configs()
    best_lgbm  = best_lgbm_hp()
    mae_hp     = {**best_lgbm, "loss": "mae"}   # COMP-H: same hp, L1 loss
    print(f"Deployed 4-GBM archs: {[c['arch'] for c in topK]}")
    print(f"Best LGBM hp (COMP-G/H base): {best_lgbm}")
    print(f"MAE-loss hp (COMP-H):         {mae_hp}")

    ctrl_raes = []; g_raes = []; h_raes = []
    for seed in range(N_SEEDS):
        ho  = np.load(f"{MTL}/ho_idx_seed{seed}.npy")
        trn = np.array([i for i in range(n_tr) if i not in set(ho.tolist())])
        sn  = np.load(f"{MTL}/sn_oof_seed{seed}.npy").ravel()

        noisy20 = se <= np.quantile(se, 0.8)
        noisy30 = se <= np.quantile(se, 0.7)
        lo = np.quantile(ytr[trn], 0.05); hi = np.quantile(ytr[trn], 0.98)

        # Standardize physics block (fit on trn only)
        sc_phys = StandardScaler().fit(Xphys_raw[trn])
        Xphys   = sc_phys.transform(Xphys_raw)  # (4139, 61) standardized

        # Full feature matrix = combined + standardized physics
        Xfull = np.hstack([Xtr, Xphys])   # (4139, 2265+61)

        # Deployed 4-GBM predictions (on holdout ho)
        gbm_preds = []
        for c in topK:
            use = trn
            if c["prep"] == "noisy20": use = trn[noisy20[trn]]
            elif c["prep"] == "noisy30": use = trn[noisy30[trn]]
            gbm_preds.append(fit_pred(c["arch"], c["hp"], Xfull, ytr, use, Xfull[ho]))

        # Foundation model OOF on holdout
        found_preds = [chem[ho], tab[ho], sn]

        # CONTROL: 7-member flat mean
        ctrl = np.clip(np.mean(gbm_preds + found_preds, 0), lo, hi)

        # COMP-G: add QM-only LGBM as 8th member
        # Train on ONLY physics features (no combined)
        qm_only_pred = fit_pred("lgbm", best_lgbm, Xphys, ytr, trn, Xphys[ho])
        treat_g = np.clip(np.mean(gbm_preds + found_preds + [qm_only_pred], 0), lo, hi)

        # COMP-H: add MAE-loss LGBM on combined+physics as 8th member
        mae_pred = fit_pred("lgbm", mae_hp, Xfull, ytr, trn, Xfull[ho])
        treat_h = np.clip(np.mean(gbm_preds + found_preds + [mae_pred], 0), lo, hi)

        rc = rae(ytr[ho], ctrl)
        rg = rae(ytr[ho], treat_g)
        rh = rae(ytr[ho], treat_h)
        ctrl_raes.append(rc); g_raes.append(rg); h_raes.append(rh)
        print(f"  seed{seed}: ctrl={rc:.4f}  COMP-G={rg:.4f} (d={rg-rc:+.4f})"
              f"  COMP-H={rh:.4f} (d={rh-rc:+.4f})")

    cm = float(np.mean(ctrl_raes)); gm = float(np.mean(g_raes)); hm = float(np.mean(h_raes))
    dg = gm - cm; dh = hm - cm
    prev = json.load(open(BEST))["rae"] if os.path.exists(BEST) else 0.4231

    print(f"\nCONTROL    RAE = {cm:.4f} ± {np.std(ctrl_raes):.4f}  (seeds: {ctrl_raes})")
    print(f"COMP-G     RAE = {gm:.4f} ± {np.std(g_raes):.4f}  delta = {dg:+.4f}  (QM-only 8th member)")
    print(f"COMP-H     RAE = {hm:.4f} ± {np.std(h_raes):.4f}  delta = {dh:+.4f}  (MAE-loss 8th member)")
    print(f"deployed best = {prev:.4f}")
    print(f"Gate: delta < -0.001 AND treatment < {prev:.4f}")
    deploy_g = bool(dg < -0.001 and gm < prev)
    deploy_h = bool(dh < -0.001 and hm < prev)
    print(f"COMP-G deploy = {deploy_g}   COMP-H deploy = {deploy_h}")

    out = {
        "tag": "nb1310",
        "control_rae": cm, "control_raes": ctrl_raes,
        "compG_rae": gm, "compG_raes": g_raes, "compG_delta": dg, "compG_deploy": deploy_g,
        "compH_rae": hm, "compH_raes": h_raes, "compH_delta": dh, "compH_deploy": deploy_h,
        "best_lgbm_hp": best_lgbm, "mae_hp": mae_hp,
        "n_phys_cols": Xphys_raw.shape[1], "wall_sec": round(time.time() - t0, 1),
    }
    json.dump(out, open(f"{P}/nb1310_summary.json", "w"), indent=2)
    print(f"\nsaved {P}/nb1310_summary.json  ({out['wall_sec']:.0f}s)")


if __name__ == "__main__":
    main()
