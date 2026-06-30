"""nb1301 — MACE-StrainRelief stacked gate: test MACE_strain on top of DEPLOYED OrbMol config.
Control  = 4-GBM(combined+AIMNet2+strain+D4+DBSTEP+OrbMol) + CheMeleon + TabPFN + sn_oof  [best 0.4231]
Treatment = same 4-GBMs but also +3 MACE_strain scalars (mace_strain, mace_conf_espread, mace_conf_erange)
Deploy if matched delta < -0.001 AND treatment < best_ensemble.json.rae.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb1126_combinatorial_search import feature_matrix, CACHE, make_model
from src.pxr.data import load_train
from src.pxr.eval import rae
from sklearn.preprocessing import StandardScaler
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

P = "data/processed"; SD = "C:/pxr_work/search"; MTL = "C:/pxr_work/mtl"
BEST = f"{SD}/best_ensemble.json"; LOG = f"{SD}/results.jsonl"; N_SEEDS = 3

AIM = "C:/pxr_work/aimnet2/aimnet_features.csv"
ACOLS = ["aimnet_energy", "aimnet_qmin", "aimnet_qmax", "aimnet_qabs_mean", "aimnet_qstd",
         "aimnet_qsum_abs", "aimnet_dipole", "aimnet_fmax", "aimnet_frms"]

STR = "C:/pxr_work/strain/strain_features.csv"
SCOLS = ["strain_relax_mean", "strain_relax_max", "conf_espread", "conf_erange",
         "conf_n", "rmsd_mean", "rmsd_max", "e_per_heavy"]

D4 = "C:/pxr_work/d4/d4_features.csv"
DCOLS = ["d4_alpha_sum", "d4_alpha_mean", "d4_alpha_std", "d4_alpha_max",
         "d4_c6diag_mean", "d4_c6diag_std", "d4_c6_total", "d4_edisp",
         "d4_edisp_per_atom", "d4_cn_mean", "d4_cn_max", "d4_qeeq_min",
         "d4_qeeq_max", "d4_qeeq_std", "d4_qeeq_absum"]

DB = "C:/pxr_work/dbstep/dbstep_features.csv"
DBCOLS = ["vbur_r25", "vbur_r35", "vbur_r45", "vbur_r55", "vbur_r65",
          "ster_L", "ster_Bmin", "ster_Bmax", "ster_aniso",
          "npr1", "npr2", "asphericity", "spherocity", "eccentricity",
          "radgyr", "inertial_sf"]

ORB = "C:/pxr_work/orbmol/orbmol_features.csv"
ORBCOLS = ["orb_energy", "orb_energy_per_ha", "orb_fmax", "orb_frms", "orb_fstd",
           "orb_conf_mean", "orb_conf_std", "orb_conf_node_mean", "orb_conf_node_std",
           "orb_conf_node_min", "orb_node_emb_mean", "orb_node_emb_std", "orb_node_emb_norm"]

MS = "C:/pxr_work/mace_strain_local/mace_strain_local.csv"
MSCOLS = ["mace_strain", "mace_conf_espread", "mace_conf_erange"]


def topK_configs(archs):
    recs = [json.loads(l) for l in open(LOG) if l.strip()]
    valid = [r for r in recs if "ps_rae" in r and "error" not in r and r["arch"] in archs]
    valid.sort(key=lambda r: r["ps_rae"]); bp = {}
    for r in valid: bp.setdefault(r["arch"], r)
    return list(bp.values())


def fit_pred(c, Xfull, ytr, use_idx, te_rows, Xextra=None):
    A, B = Xfull, Xfull[te_rows]
    if Xextra is not None:
        A = np.hstack([Xfull, Xextra]); B = np.hstack([Xfull[te_rows], Xextra[te_rows]])
    m = make_model(c["arch"], c["hp"])
    if c["arch"] in ("ridge", "enet"):
        sc = StandardScaler().fit(A[use_idx]); m.fit(sc.transform(A[use_idx]), ytr[use_idx])
        return m.predict(sc.transform(B))
    m.fit(A[use_idx], ytr[use_idx]); return m.predict(B)


def aligned_block(csv_path, cols, tr_names):
    df = pd.read_csv(csv_path)
    df = df[df.src == "train"].drop_duplicates(subset="name", keep="first")
    sub = df.set_index("name").reindex(tr_names)
    X = sub[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    nan_rows = int(np.isnan(X).any(axis=1).sum())
    med = np.nanmedian(X, axis=0); inds = np.where(np.isnan(X)); X[inds] = np.take(med, inds[1])
    return X, nan_rows


def main():
    print("nb1301 -- MACE-StrainRelief stacked gate (on OrbMol deployed config)")
    print("=" * 70)

    for path, name in [(MS, "MACE strain"), (ORB, "OrbMol")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} cache not found: {path}"); sys.exit(1)

    df_ms = pd.read_csv(MS)
    n_done = (df_ms["status"] == "ok").sum()
    print(f"MACE strain cache: {len(df_ms)} rows, {n_done} ok")

    d = np.load(CACHE); ytr = d["ytr"]; se = d["se"]; n_tr = len(ytr)
    Xtr, _ = feature_matrix(d, "combined")
    chem = np.load(f"{SD}/chemeleon_oof.npy"); tab = np.load(f"{SD}/tabpfn_oof.npy")

    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    Xqm,  qm_nan  = aligned_block(AIM, ACOLS,   tr["name"])
    Xst,  st_nan  = aligned_block(STR, SCOLS,   tr["name"])
    Xd4,  d4_nan  = aligned_block(D4,  DCOLS,   tr["name"])
    Xdb,  db_nan  = aligned_block(DB,  DBCOLS,  tr["name"])
    Xorb, orb_nan = aligned_block(ORB, ORBCOLS, tr["name"])
    Xms,  ms_nan  = aligned_block(MS,  MSCOLS,  tr["name"])
    print(f"AIMNet2 imp={qm_nan} mmff_strain imp={st_nan} d4 imp={d4_nan} "
          f"dbstep imp={db_nan} orbmol imp={orb_nan} mace_strain imp={ms_nan}")

    # Correlation of MACE strain scalars with target and existing features
    y = tr["pec50"].values
    for j, col in enumerate(MSCOLS):
        x = Xms[:, j]
        if np.std(x) > 0:
            corr_y = float(np.corrcoef(x, y)[0, 1])
            corr_mmff = float(np.corrcoef(x, Xst[:, 0])[0, 1])
            corr_orb  = float(np.corrcoef(x, Xorb[:, 0])[0, 1])
            print(f"  {col}: corr_y={corr_y:.3f}  corr_MMFF_strain={corr_mmff:.3f}  corr_OrbMol_E={corr_orb:.3f}")

    topK = topK_configs(("lgbm", "xgb", "cat", "histgb"))
    print(f"Control config: {[r['arch'] for r in topK]} + CheMeleon + TabPFN + sn_oof + AIMNet2 + MMFF_strain + D4 + DBSTEP + OrbMol")
    print(f"Treatment config: Control + 3 MACE_strain scalars")

    c_raes, t_raes = [], []
    for seed in range(N_SEEDS):
        ho  = np.load(f"{MTL}/ho_idx_seed{seed}.npy")
        trn = np.array([i for i in range(n_tr) if i not in set(ho.tolist())])
        noisy20 = se <= np.quantile(se, 0.8); noisy30 = se <= np.quantile(se, 0.7)
        lo, hi = np.quantile(ytr[trn], 0.05), np.quantile(ytr[trn], 0.98)

        scq  = StandardScaler().fit(Xqm[trn]);  Xqm_std  = scq.transform(Xqm)
        scs  = StandardScaler().fit(Xst[trn]);  Xst_std  = scs.transform(Xst)
        scd  = StandardScaler().fit(Xd4[trn]);  Xd4_std  = scd.transform(Xd4)
        scb  = StandardScaler().fit(Xdb[trn]);  Xdb_std  = scb.transform(Xdb)
        sco  = StandardScaler().fit(Xorb[trn]); Xorb_std = sco.transform(Xorb)
        scm  = StandardScaler().fit(Xms[trn]);  Xms_std  = scm.transform(Xms)

        Xctrl  = np.hstack([Xqm_std, Xst_std, Xd4_std, Xdb_std, Xorb_std])
        Xtreat = np.hstack([Xqm_std, Xst_std, Xd4_std, Xdb_std, Xorb_std, Xms_std])

        gnn = np.load(f"{MTL}/sn_oof_seed{seed}.npy").ravel()
        gbm_c, gbm_t = [], []
        for c in topK:
            use = trn
            if c["prep"] == "noisy20": use = trn[noisy20[trn]]
            elif c["prep"] == "noisy30": use = trn[noisy30[trn]]
            gbm_c.append(fit_pred(c, Xtr, ytr, use, ho, Xextra=Xctrl))
            gbm_t.append(fit_pred(c, Xtr, ytr, use, ho, Xextra=Xtreat))

        ctrl  = np.clip(np.mean(gbm_c + [chem[ho], tab[ho], gnn], 0), lo, hi)
        treat = np.clip(np.mean(gbm_t + [chem[ho], tab[ho], gnn], 0), lo, hi)
        c_raes.append(rae(ytr[ho], ctrl)); t_raes.append(rae(ytr[ho], treat))
        print(f"  seed{seed} ctrl {c_raes[-1]:.4f}  treat {t_raes[-1]:.4f}  d {t_raes[-1]-c_raes[-1]:+.4f}")

    cm, tm = float(np.mean(c_raes)), float(np.mean(t_raes)); delta = tm - cm
    prev = json.load(open(BEST))["rae"] if os.path.exists(BEST) else 0.4231
    n_neg = sum(1 for t, c in zip(t_raes, c_raes) if t < c)
    print(f"\nControl  (OrbMol deployed) RAE = {cm:.4f} +/- {np.std(c_raes):.4f}")
    print(f"Treatment(+3 MACE_strain)  RAE = {tm:.4f} +/- {np.std(t_raes):.4f}")
    print(f"MATCHED delta = {delta:+.4f}   deployed best = {prev}")
    print(f"Seeds negative: {n_neg}/{N_SEEDS}")

    deploy = bool(delta < -0.001 and tm < prev)
    out = {"control_rae": cm, "treatment_rae": tm, "matched_delta": delta,
           "c_raes": c_raes, "t_raes": t_raes, "n_seeds_neg": n_neg,
           "ms_imputed": ms_nan, "orb_imputed": orb_nan,
           "deployed_best": prev, "deploy": deploy}
    json.dump(out, open(f"{P}/nb1301_mace_strain_orbmol_summary.json", "w"), indent=2)
    print(f"\nsaved {P}/nb1301_mace_strain_orbmol_summary.json -> DEPLOY={deploy}")
    if deploy:
        print("GATE PASSED: update best_ensemble.json and build 513 submission with OrbMol+MACE_strain")
    else:
        print(f"GATE FAILED: delta={delta:+.4f} (need <-0.001 AND treatment<{prev:.4f})")


if __name__ == "__main__":
    main()
