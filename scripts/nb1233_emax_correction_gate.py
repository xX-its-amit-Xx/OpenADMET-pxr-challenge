"""nb1233 — Emax efficacy-correction probe (memo P1) on the DEPLOYED ensemble.

Tests whether the activation/efficacy axis (emax_rel) the GAL4-LBD transactivation
assay measures adds signal our ligand-only model misses.

METHOD (honest, label-decomposition):
  1. Cross-fit an emax_rel predictor (LGBM on combined feats) on the training fold
     (trn) -> predict emax_rel_pred for the held-out rows (ho). emax_rel is train-only,
     so the 513 test would need a predicted Emax too (a 2D problem -> absorbed).
  2. Correction:  pEC50_corr = pEC50_pred + gamma * log10(emax_rel_pred).
  3. gamma is tuned ONLY on seed-0's holdout, then FROZEN and evaluated OUT-OF-SAMPLE
     on seed-1 & seed-2 holdouts (avoids the in-sample overfitting trap the memo warns
     about: tuning gamma on the same set you score is the Ekins-2009 pattern).

  control   = deployed 4-GBM(combined) + CheMeleon + TabPFN + sn_oof + AIMNet2+strain+D4+DBSTEP
  treatment = control prediction, then + gamma*log10(emax_rel_pred), re-clipped
Deploy only if OUT-OF-SAMPLE matched delta < -0.001 AND treatment < deployed best (0.4242).
Expected NULL: memo fact corr(|ligand-error|, emax_rel) = -0.018 ~ 0.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb1126_combinatorial_search import feature_matrix, CACHE, make_model
from src.pxr.data import load_train
from src.pxr.eval import rae
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
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
GAMMA_GRID = np.linspace(-1.5, 1.5, 61)


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
        sc = StandardScaler().fit(A[use_idx]); m.fit(sc.transform(A[use_idx]), ytr[use_idx]); return m.predict(sc.transform(B))
    m.fit(A[use_idx], ytr[use_idx]); return m.predict(B)


def aligned_block(csv_path, cols, tr_names):
    df = pd.read_csv(csv_path)
    df = df[df.src == "train"].drop_duplicates(subset="name", keep="first")
    sub = df.set_index("name").reindex(tr_names)
    X = sub[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    med = np.nanmedian(X, axis=0); inds = np.where(np.isnan(X)); X[inds] = np.take(med, inds[1])
    return X


def main():
    d = np.load(CACHE); ytr = d["ytr"]; se = d["se"]; n_tr = len(ytr)
    Xtr, _ = feature_matrix(d, "combined")
    chem = np.load(f"{SD}/chemeleon_oof.npy"); tab = np.load(f"{SD}/tabpfn_oof.npy")

    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    emax_rel = tr["emax_rel"].to_numpy(float)
    log_emax = np.log10(np.clip(emax_rel, 1e-3, None))
    Xqm = aligned_block(AIM, ACOLS, tr["name"])
    Xst = aligned_block(STR, SCOLS, tr["name"])
    Xd4 = aligned_block(D4, DCOLS, tr["name"])
    Xdb = aligned_block(DB, DBCOLS, tr["name"])

    topK = topK_configs(("lgbm", "xgb", "cat", "histgb"))
    print(f"deployed: {[r['arch'] for r in topK]} + CheMeleon + TabPFN + sn_oof + AIMNet2+strain+D4+DBSTEP")

    # per-seed: build control ho preds, emax_pred ho, and store for gamma tuning
    seed_data = []
    emax_corrs = []  # (corr(|err|,emax_rel), signed corr(err, log_emax), emax-pred corr)
    for seed in range(N_SEEDS):
        ho = np.load(f"{MTL}/ho_idx_seed{seed}.npy")
        trn = np.array([i for i in range(n_tr) if i not in set(ho.tolist())])
        noisy20 = se <= np.quantile(se, 0.8); noisy30 = se <= np.quantile(se, 0.7)
        lo, hi = np.quantile(ytr[trn], 0.05), np.quantile(ytr[trn], 0.98)
        scq = StandardScaler().fit(Xqm[trn]); scs = StandardScaler().fit(Xst[trn])
        scd = StandardScaler().fit(Xd4[trn]); scb = StandardScaler().fit(Xdb[trn])
        Xphys = np.hstack([scq.transform(Xqm), scs.transform(Xst),
                           scd.transform(Xd4), scb.transform(Xdb)])
        gnn = np.load(f"{MTL}/sn_oof_seed{seed}.npy").ravel()
        gbm = []
        for c in topK:
            use = trn
            if c["prep"] == "noisy20": use = trn[noisy20[trn]]
            elif c["prep"] == "noisy30": use = trn[noisy30[trn]]
            gbm.append(fit_pred(c, Xtr, ytr, use, ho, Xextra=Xphys))
        ctrl = np.clip(np.mean(gbm + [chem[ho], tab[ho], gnn], 0), lo, hi)

        # emax_rel predictor: LGBM on combined feats, trn -> ho (never sees ho labels)
        em = lgb.LGBMRegressor(n_estimators=400, num_leaves=63, learning_rate=0.03,
                               subsample=0.8, colsample_bytree=0.8, verbose=-1)
        em.fit(Xtr[trn], log_emax[trn])
        log_emax_pred = em.predict(Xtr[ho])

        err = ytr[ho] - ctrl
        emax_corrs.append((
            float(np.corrcoef(np.abs(err), emax_rel[ho])[0, 1]),
            float(np.corrcoef(err, log_emax[ho])[0, 1]),
            float(np.corrcoef(log_emax_pred, log_emax[ho])[0, 1]),
        ))
        seed_data.append(dict(yho=ytr[ho], ctrl=ctrl, lep=log_emax_pred, lo=lo, hi=hi))
        print(f"  seed{seed} ctrl {rae(ytr[ho], ctrl):.4f}  emax-pred corr {emax_corrs[-1][2]:+.3f}")

    # --- tune gamma on seed0, freeze, evaluate OOS on seed1,2 ---
    s0 = seed_data[0]
    best_g, best_r = 0.0, 1e9
    for g in GAMMA_GRID:
        r = rae(s0["yho"], np.clip(s0["ctrl"] + g * s0["lep"], s0["lo"], s0["hi"]))
        if r < best_r: best_r, best_g = r, g
    ctrl0 = rae(s0["yho"], s0["ctrl"])
    print(f"\ngamma* (tuned on seed0) = {best_g:+.3f}  insample seed0 {ctrl0:.4f}->{best_r:.4f} ({best_r-ctrl0:+.4f})")

    # out-of-sample deltas (frozen gamma on seed1,2)
    oos_c, oos_t = [], []
    for sd in seed_data[1:]:
        c = rae(sd["yho"], sd["ctrl"])
        t = rae(sd["yho"], np.clip(sd["ctrl"] + best_g * sd["lep"], sd["lo"], sd["hi"]))
        oos_c.append(c); oos_t.append(t)
        print(f"  OOS seed ctrl {c:.4f}  treat {t:.4f}  d {t-c:+.4f}")
    oos_delta = float(np.mean([t - c for c, t in zip(oos_c, oos_t)]))

    # also report all-3-seed in-sample-tuned delta (optimistic upper bound)
    insamp = []
    for sd in seed_data:
        bg, br = 0.0, 1e9
        for g in GAMMA_GRID:
            r = rae(sd["yho"], np.clip(sd["ctrl"] + g * sd["lep"], sd["lo"], sd["hi"]))
            if r < br: br, bg = r, g
        insamp.append(br - rae(sd["yho"], sd["ctrl"]))
    insamp_delta = float(np.mean(insamp))

    prev = json.load(open(BEST))["rae"] if os.path.exists(BEST) else 0.4242
    cm = float(np.mean(oos_c)); tm = float(np.mean(oos_t))
    deploy = bool(oos_delta < -0.001 and tm < prev)
    print(f"\nOUT-OF-SAMPLE matched delta = {oos_delta:+.4f}  (deployed best {prev})")
    print(f"in-sample-tuned delta (optimistic) = {insamp_delta:+.4f}")
    print(f"mean corr(|err|, emax_rel)   = {np.mean([c[0] for c in emax_corrs]):+.3f}")
    print(f"mean signed corr(err, log_emax) = {np.mean([c[1] for c in emax_corrs]):+.3f}")
    print(f"mean emax-pred OOF corr      = {np.mean([c[2] for c in emax_corrs]):+.3f}")

    out = {"oos_control_rae": cm, "oos_treatment_rae": tm, "oos_matched_delta": oos_delta,
           "gamma_star": float(best_g), "insample_seed0_delta": float(best_r - ctrl0),
           "insample_tuned_delta_mean": insamp_delta,
           "corr_abserr_emaxrel_mean": float(np.mean([c[0] for c in emax_corrs])),
           "corr_signed_err_logemax_mean": float(np.mean([c[1] for c in emax_corrs])),
           "emax_pred_oof_corr_mean": float(np.mean([c[2] for c in emax_corrs])),
           "deployed_best": prev, "deploy": deploy}
    json.dump(out, open(f"{P}/nb1233_emax_summary.json", "w"), indent=2)
    # save OOF for viz: stack all seeds' ctrl preds and y
    allerr_oof = np.concatenate([sd["yho"] for sd in seed_data])
    np.save(f"{SD}/emax_corr_oof.npy", np.concatenate([sd["ctrl"] for sd in seed_data]))
    np.save(f"{SD}/emax_corr_y.npy", allerr_oof)
    print(f"\nsaved {P}/nb1233_emax_summary.json -> DEPLOY={deploy}")


if __name__ == "__main__":
    main()
