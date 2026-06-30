"""nb1404 — NCATS qHTS PXR (NR1I2) on-target read-across feature, honest gate on the
FULL DEPLOYED config (best ~0.4133/0.4149). NCATS = PubChem qHTS PXR-transactivation
actives (AID 1346982/1346985), 1711 compounds with p-scale potency (different ASSAY than
our CRC pEC50 but SAME target). Leakage-safe read-across: train an NCATS-qHTS p-potency
predictor (combined feats) that NEVER sees our PXR pEC50, predict NCATS-activity for our
4139 train + 513 test, append that single prediction as ONE extra feature to the full
deployed ensemble.

  control   = 4-GBM(combined) + CheMeleon + TabPFN + sn_oof + rpxr_oof
              + AIMNet2 + strain + D4 + DBSTEP + OrbMol   (full deployed feature config)
  treatment = control + NCATS-read-across scalar
Deploy only if matched delta < -0.001 AND n_neg>=2/3 AND treatment < deployed best.

Cycle-310 found on-target external EC50 lifts as an AUX MULTITASK HEAD where post-hoc
read-across/residual failed; this read-across probe is the CHEAP directional test on the
full stack — if it shows corr-with-error signal, escalate to a chemprop aux head.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb1126_combinatorial_search import feature_matrix, CACHE, make_model
from src.pxr.data import load_train, load_test
from src.pxr.featurize import combined, impute
from src.pxr.chem import morgan_fp_batch
from src.pxr.eval import rae
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi
RDLogger.DisableLog("rdApp.*")

P = "data/processed"; SD = "C:/pxr_work/search"; MTL = "C:/pxr_work/mtl"; OUT = "C:/pxr_work/ncats"
BEST = f"{SD}/best_ensemble.json"; LOG = f"{SD}/results.jsonl"; N_SEEDS = 3
NCATS_CSV = f"{OUT}/ncats_aux.csv"

BLOCKS = {
    "AIMNet2": ("C:/pxr_work/aimnet2/aimnet_features.csv",
        ["aimnet_energy","aimnet_qmin","aimnet_qmax","aimnet_qabs_mean","aimnet_qstd",
         "aimnet_qsum_abs","aimnet_dipole","aimnet_fmax","aimnet_frms"]),
    "strain": ("C:/pxr_work/strain/strain_features.csv",
        ["strain_relax_mean","strain_relax_max","conf_espread","conf_erange","conf_n",
         "rmsd_mean","rmsd_max","e_per_heavy"]),
    "D4": ("C:/pxr_work/d4/d4_features.csv",
        ["d4_alpha_sum","d4_alpha_mean","d4_alpha_std","d4_alpha_max","d4_c6diag_mean",
         "d4_c6diag_std","d4_c6_total","d4_edisp","d4_edisp_per_atom","d4_cn_mean","d4_cn_max",
         "d4_qeeq_min","d4_qeeq_max","d4_qeeq_std","d4_qeeq_absum"]),
    "DBSTEP": ("C:/pxr_work/dbstep/dbstep_features.csv",
        ["vbur_r25","vbur_r35","vbur_r45","vbur_r55","vbur_r65","ster_L","ster_Bmin","ster_Bmax",
         "ster_aniso","npr1","npr2","asphericity","spherocity","eccentricity","radgyr","inertial_sf"]),
    "OrbMol": ("C:/pxr_work/orbmol/orbmol_features.csv",
        ["orb_energy","orb_energy_per_ha","orb_fmax","orb_frms","orb_fstd","orb_conf_mean",
         "orb_conf_std","orb_conf_node_mean","orb_conf_node_std","orb_conf_node_min",
         "orb_node_emb_mean","orb_node_emb_std","orb_node_emb_norm"]),
}


def ikey(s):
    m = Chem.MolFromSmiles(str(s))
    try: return inchi.InchiToInchiKey(inchi.MolToInchi(m)) if m else None
    except Exception: return None


def topK_configs(archs):
    recs = [json.loads(l) for l in open(LOG) if l.strip()]
    valid = [r for r in recs if "ps_rae" in r and "error" not in r and r["arch"] in archs]
    valid.sort(key=lambda r: r["ps_rae"]); bp = {}
    for r in valid: bp.setdefault(r["arch"], r)
    return list(bp.values())


def fit_pred(c, Xfull, ytr, use_idx, te_rows, Xextra):
    A = np.hstack([Xfull, Xextra]); B = np.hstack([Xfull[te_rows], Xextra[te_rows]])
    m = make_model(c["arch"], c["hp"])
    if c["arch"] in ("ridge", "enet"):
        sc = StandardScaler().fit(A[use_idx]); m.fit(sc.transform(A[use_idx]), ytr[use_idx])
        return m.predict(sc.transform(B))
    m.fit(A[use_idx], ytr[use_idx]); return m.predict(B)


def aligned_block(csv_path, cols, tr_names):
    df = pd.read_csv(csv_path)
    if "src" in df.columns:
        df = df[df.src == "train"]
    df = df.drop_duplicates(subset="name", keep="first")
    sub = df.set_index("name").reindex(tr_names)
    X = sub[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    med = np.nanmedian(X, axis=0); med = np.where(np.isnan(med), 0.0, med)
    inds = np.where(np.isnan(X)); X[inds] = np.take(med, inds[1])
    return X


def build_readacross(tr_smi, te_smi, tr_ik):
    """LGBM trained on NCATS qHTS potency (combined feats) -> predict our train/test."""
    nd = pd.read_csv(NCATS_CSV)
    nc_smi = nd["cs"].astype(str).tolist(); nc_y = nd["p"].to_numpy(float)
    nc_ik = set(filter(None, (ikey(s) for s in nc_smi)))
    overlap = len(nc_ik & tr_ik)
    print(f"NCATS compounds: {len(nc_smi)}  p-range {nc_y.min():.2f}..{nc_y.max():.2f}  "
          f"InChIKey overlap NCATS-vs-PXR-train: {overlap} (NCATS model never sees PXR pEC50 -> no leakage)")
    # coverage: NCATS actives (p>=6) -> our 513 test
    act = [s for s, p in zip(nc_smi, nc_y) if p >= 6.0]
    med_tani, frac_cov = 0.0, 0.0
    if act:
        fp_a = morgan_fp_batch(act).astype(np.float32); fp_t = morgan_fp_batch(te_smi).astype(np.float32)
        inter = fp_a @ fp_t.T; union = fp_a.sum(1)[:, None] + fp_t.sum(1)[None, :] - inter
        sim = (inter / np.maximum(union, 1)).max(0)
        med_tani = float(np.median(sim)); frac_cov = float((sim >= 0.4).mean())
    print(f"Coverage (test->nearest NCATS active p>=6): median Tanimoto={med_tani:.3f}  frac>=0.4={frac_cov:.3f}")
    Xnc = impute(combined(nc_smi)).astype(np.float32)
    Xtr = impute(combined(tr_smi)).astype(np.float32)
    Xte = impute(combined(te_smi)).astype(np.float32)
    mdl = LGBMRegressor(n_estimators=600, num_leaves=64, learning_rate=0.04, n_jobs=4, verbose=-1)
    mdl.fit(Xnc, nc_y)
    ra_tr = mdl.predict(Xtr).astype(np.float32); ra_te = mdl.predict(Xte).astype(np.float32)
    np.save(f"{OUT}/ra_tr.npy", ra_tr); np.save(f"{OUT}/ra_te.npy", ra_te)
    return ra_tr, ra_te, dict(n_ncats=len(nc_smi), ik_overlap=overlap,
                              med_tani=round(med_tani, 3), frac_cov=round(frac_cov, 3))


def main():
    d = np.load(CACHE); ytr = d["ytr"]; se = d["se"]; n_tr = len(ytr)
    Xtr, _ = feature_matrix(d, "combined")
    chem = np.load(f"{SD}/chemeleon_oof.npy"); tab = np.load(f"{SD}/tabpfn_oof.npy")
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    tr_names = tr["name"].astype(str).tolist(); tr_smi = tr["smiles"].tolist()
    te = load_test(); te_smi = te["smiles"].tolist()
    tr_ik = set(filter(None, (ikey(s) for s in tr_smi)))

    ra_tr, ra_te, meta = build_readacross(tr_smi, te_smi, tr_ik)
    print(f"read-across feat: train range [{ra_tr.min():.2f},{ra_tr.max():.2f}]  "
          f"corr(read-across, PXR pEC50)={np.corrcoef(ra_tr, ytr)[0,1]:+.3f}")

    ctrl_blocks = {n: aligned_block(p, c, tr_names) for n, (p, c) in BLOCKS.items()}

    nb32 = f"{P}/oof_chemprop_aux.npy"
    if os.path.exists(nb32):
        err = ytr - np.load(nb32)
        print(f"corr(nb3200_err, NCATS-read-across) = {np.corrcoef(err, ra_tr)[0,1]:+.4f}")

    topK = topK_configs(("lgbm", "xgb", "cat", "histgb"))
    print(f"deployed GBMs: {[r['arch'] for r in topK]} + CheMeleon+TabPFN+sn+rpxr + {'+'.join(BLOCKS)}")
    c_raes, t_raes = [], []
    for seed in range(N_SEEDS):
        ho = np.load(f"{MTL}/ho_idx_seed{seed}.npy")
        trn = np.array([i for i in range(n_tr) if i not in set(ho.tolist())])
        noisy20 = se <= np.quantile(se, 0.8); noisy30 = se <= np.quantile(se, 0.7)
        lo, hi = np.quantile(ytr[trn], 0.05), np.quantile(ytr[trn], 0.98)

        def scaled(X): return StandardScaler().fit(X[trn]).transform(X)
        ctrl_feat = np.hstack([scaled(ctrl_blocks[n]) for n in BLOCKS])
        ra_col = scaled(ra_tr[:, None])
        treat_feat = np.hstack([ctrl_feat, ra_col])

        sn = np.load(f"{MTL}/sn_oof_seed{seed}.npy").ravel()
        rp = np.load(f"{MTL}/rpxr_oof_seed{seed}.npy").ravel()
        gbm_c, gbm_t = [], []
        for c in topK:
            use = trn
            if c["prep"] == "noisy20": use = trn[noisy20[trn]]
            elif c["prep"] == "noisy30": use = trn[noisy30[trn]]
            gbm_c.append(fit_pred(c, Xtr, ytr, use, ho, ctrl_feat))
            gbm_t.append(fit_pred(c, Xtr, ytr, use, ho, treat_feat))
        ctrl = np.clip(np.mean(gbm_c + [chem[ho], tab[ho], sn, rp], 0), lo, hi)
        treat = np.clip(np.mean(gbm_t + [chem[ho], tab[ho], sn, rp], 0), lo, hi)
        c_raes.append(rae(ytr[ho], ctrl)); t_raes.append(rae(ytr[ho], treat))
        print(f"  seed{seed} ctrl {c_raes[-1]:.4f}  treat {t_raes[-1]:.4f}  d {t_raes[-1]-c_raes[-1]:+.4f}", flush=True)

    cm, tm = float(np.mean(c_raes)), float(np.mean(t_raes)); delta = tm - cm
    n_neg = int(sum(1 for a, b in zip(t_raes, c_raes) if b - a < 0))
    prev = json.load(open(BEST))["rae"] if os.path.exists(BEST) else 0.4149
    print(f"\ncontrol  (full deployed config)        RAE = {cm:.4f} +/- {np.std(c_raes):.4f}")
    print(f"treatment(deployed + NCATS-read-across) RAE = {tm:.4f} +/- {np.std(t_raes):.4f}")
    print(f"MATCHED delta = {delta:+.4f}  ({n_neg}/{N_SEEDS} seeds neg)  deployed best = {prev}")
    deploy = bool(delta < -0.001 and n_neg >= 2 and tm < prev)
    out = {"approach": "ncats_qhts_pxr_readacross_scalar_on_deployed_config",
           "control_rae": cm, "treatment_rae": tm, "matched_delta": delta, "n_seeds_neg": n_neg,
           "c_raes": c_raes, "t_raes": t_raes, "corr_ra_vs_pec50": round(float(np.corrcoef(ra_tr, ytr)[0,1]),4),
           "deployed_best": prev, "deploy": deploy, **meta}
    json.dump(out, open(f"{P}/nb1404_ncats_summary.json", "w"), indent=2)
    print(f"\nsaved {P}/nb1404_ncats_summary.json -> DEPLOY={deploy}")


if __name__ == "__main__":
    main()
