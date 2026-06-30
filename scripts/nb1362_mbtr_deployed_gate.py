"""nb1362 — dscribe MBTR (global many-body 3D) honest gate on the FULL DEPLOYED config
(best 0.4149, nb1328 COMP-M). The untested OTHER dscribe half of L319: SOAP (LOCAL power
spectrum) won as a residual chain; MBTR is the GLOBAL counterpart (smooth k=2 inverse-distance
+ k=3 angle distributions, element-pair/triple resolved). Genuinely distinct math on the same
proven-winning 3D-geometry axis.

MBTR raw (C:/pxr_work/mbtr/mbtr_raw.npz: names, X[39600]) -> per-fold PCA-N (fit on TRAIN
rows only, no leakage), like the ANI-2x AEV gate (nb1351).

  control   = 4-GBM(combined) + CheMeleon + TabPFN + sn_oof + rpxr_oof
              + AIMNet2 + strain + D4 + DBSTEP + OrbMol   (full deployed feature config)
  treatment = control + MBTR-PCA-N block
Deploy only if matched delta < -0.001 AND n_neg>=2/3 AND treatment < deployed best (0.4149).
If block is promising, a SOAP-style residual chain (Ridge a=100, blend 0.5) is the deploy path.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb1126_combinatorial_search import feature_matrix, CACHE, make_model
from src.pxr.data import load_train
from src.pxr.eval import rae
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

P = "data/processed"; SD = "C:/pxr_work/search"; MTL = "C:/pxr_work/mtl"
BEST = f"{SD}/best_ensemble.json"; LOG = f"{SD}/results.jsonl"; N_SEEDS = 3
N_PCA = 24
MBTR_NPZ = "C:/pxr_work/mbtr/mbtr_raw.npz"

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


def aligned_mbtr(tr_names):
    d = np.load(MBTR_NPZ, allow_pickle=True)
    nm = d["names"].astype(str); X = d["X"].astype(np.float32)
    idx = {n: i for i, n in enumerate(nm)}
    feat = X.shape[1]
    out = np.full((len(tr_names), feat), np.nan, np.float32)
    miss = 0
    for r, n in enumerate(tr_names):
        if n in idx: out[r] = X[idx[n]]
        else: miss += 1
    med = np.nanmedian(out, axis=0); med = np.where(np.isnan(med), 0.0, med)
    ii = np.where(np.isnan(out)); out[ii] = np.take(med, ii[1])
    # drop all-zero / zero-variance columns (most of the 39600 are unused element triples)
    var = out.var(axis=0); keep = var > 1e-12
    return out[:, keep], miss, int(keep.sum())


def main():
    d = np.load(CACHE); ytr = d["ytr"]; se = d["se"]; n_tr = len(ytr)
    Xtr, _ = feature_matrix(d, "combined")
    chem = np.load(f"{SD}/chemeleon_oof.npy"); tab = np.load(f"{SD}/tabpfn_oof.npy")
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    tr_names = tr["name"].astype(str).tolist()

    ctrl_blocks = {n: aligned_block(p, c, tr_names) for n, (p, c) in BLOCKS.items()}
    Xmbtr, miss, nkeep = aligned_mbtr(tr_names)
    print(f"MBTR aligned: missing={miss}/{n_tr}  nonzero-var dims kept={nkeep}/39600")

    # corr of leading MBTR-PCA comps (global PCA, report only) with nb3200 error
    nb32 = f"{P}/oof_chemprop_aux.npy"
    if os.path.exists(nb32):
        err = ytr - np.load(nb32)
        Z = StandardScaler().fit_transform(Xmbtr)
        pcs = PCA(n_components=8, random_state=0).fit_transform(Z)
        cors = [float(np.corrcoef(err, pcs[:, j])[0, 1]) for j in range(8)]
        print("corr(nb3200_err, MBTR-PCA[0:8]):", " ".join(f"{c:+.3f}" for c in cors))

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
        # MBTR PCA-N fit on TRAIN rows only
        scm = StandardScaler().fit(Xmbtr[trn]); Zm = scm.transform(Xmbtr)
        pca = PCA(n_components=N_PCA, random_state=0).fit(Zm[trn])
        mbtr_pca = pca.transform(Zm)
        treat_feat = np.hstack([ctrl_feat, mbtr_pca])

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
    print(f"treatment(deployed + MBTR-PCA{N_PCA})       RAE = {tm:.4f} +/- {np.std(t_raes):.4f}")
    print(f"MATCHED delta = {delta:+.4f}  ({n_neg}/{N_SEEDS} seeds neg)  deployed best = {prev}")
    deploy = bool(delta < -0.001 and n_neg >= 2 and tm < prev)
    out = {"approach": "dscribe_MBTR_global_manybody_PCA_block_on_deployed_config",
           "control_rae": cm, "treatment_rae": tm, "matched_delta": delta, "n_seeds_neg": n_neg,
           "c_raes": c_raes, "t_raes": t_raes, "mbtr_missing": miss, "n_pca": N_PCA,
           "deployed_best": prev, "deploy": deploy}
    json.dump(out, open(f"{P}/nb1362_mbtr_summary.json", "w"), indent=2)
    print(f"\nsaved {P}/nb1362_mbtr_summary.json -> DEPLOY={deploy}")


if __name__ == "__main__":
    main()
