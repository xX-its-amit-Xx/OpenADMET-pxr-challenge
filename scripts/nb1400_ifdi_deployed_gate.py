"""nb1400 — IFDI/FLI interfragment delocalization (Wiberg bond-order DI) honest gate
on the FULL DEPLOYED feature config (best 0.4149, nb1328 COMP-M ensemble pre-residual-chain).

10 fragment-level quantum-delocalization scalars from GFN2-xTB Wiberg bond orders
(C:/pxr_work/ifdi/ifdi_features.csv): n_ringsys, total_ifdi, max_pair_di, mean_cross_di,
std_cross_di, fli_mean, fli_min, fli_max, fli_std, frac_deloc. Cross-ring-system DI =
inter-fragment electron delocalization (conjugation / charge-transfer between aromatic ring
systems in PXR's aromatic pocket) — a FRAGMENT-LEVEL observable distinct from the deployed
per-atom AIMNet2 charges, D4 dispersion, DBSTEP steric and CDFT global reactivity (all absorbed).

  control   = 4-GBM(combined) + CheMeleon + TabPFN + sn_oof + rpxr_oof
              + AIMNet2 + strain + D4 + DBSTEP + OrbMol   (full deployed feature config)
  treatment = control + 10 IFDI/FLI scalars
Deploy only if matched delta < -0.001 AND treatment < deployed best (0.4149).
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
IFDI = ("C:/pxr_work/ifdi/ifdi_features.csv",
        ["n_ringsys","total_ifdi","max_pair_di","mean_cross_di","std_cross_di",
         "fli_mean","fli_min","fli_max","fli_std","frac_deloc"])


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
    nan_rows = int(np.isnan(X).any(axis=1).sum())
    med = np.nanmedian(X, axis=0); med = np.where(np.isnan(med), 0.0, med)
    inds = np.where(np.isnan(X)); X[inds] = np.take(med, inds[1])
    return X, nan_rows


def main():
    d = np.load(CACHE); ytr = d["ytr"]; se = d["se"]; n_tr = len(ytr)
    Xtr, _ = feature_matrix(d, "combined")
    chem = np.load(f"{SD}/chemeleon_oof.npy"); tab = np.load(f"{SD}/tabpfn_oof.npy")
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)

    ctrl_blocks = {}
    for name, (path, cols) in BLOCKS.items():
        X, nn = aligned_block(path, cols, tr["name"]); ctrl_blocks[name] = X
        print(f"{name:9s} imputed={nn}")
    Xifdi, ifdi_nan = aligned_block(*IFDI, tr["name"]); print(f"IFDI      imputed={ifdi_nan}")

    topK = topK_configs(("lgbm", "xgb", "cat", "histgb"))
    print(f"deployed GBMs: {[r['arch'] for r in topK]} + CheMeleon + TabPFN + sn_oof + rpxr_oof"
          f" + {'+'.join(BLOCKS)}")
    c_raes, t_raes = [], []
    for seed in range(N_SEEDS):
        ho = np.load(f"{MTL}/ho_idx_seed{seed}.npy")
        trn = np.array([i for i in range(n_tr) if i not in set(ho.tolist())])
        noisy20 = se <= np.quantile(se, 0.8); noisy30 = se <= np.quantile(se, 0.7)
        lo, hi = np.quantile(ytr[trn], 0.05), np.quantile(ytr[trn], 0.98)

        def scaled(X):  # fit scaler on train rows only
            return StandardScaler().fit(X[trn]).transform(X)
        ctrl_feat = np.hstack([scaled(ctrl_blocks[n]) for n in BLOCKS])
        treat_feat = np.hstack([ctrl_feat, scaled(Xifdi)])

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
        print(f"  seed{seed} ctrl {c_raes[-1]:.4f}  treat {t_raes[-1]:.4f}  d {t_raes[-1]-c_raes[-1]:+.4f}")

    cm, tm = float(np.mean(c_raes)), float(np.mean(t_raes)); delta = tm - cm
    n_neg = int(sum(1 for a, b in zip(t_raes, c_raes) if b - a < 0))
    prev = json.load(open(BEST))["rae"] if os.path.exists(BEST) else 0.4149
    print(f"\ncontrol  (full deployed feature config)   RAE = {cm:.4f} +/- {np.std(c_raes):.4f}")
    print(f"treatment(deployed + IFDI/FLI)            RAE = {tm:.4f} +/- {np.std(t_raes):.4f}")
    print(f"MATCHED delta = {delta:+.4f}  ({n_neg}/{N_SEEDS} seeds neg)  deployed best = {prev}")
    deploy = bool(delta < -0.001 and n_neg >= 2 and tm < prev)
    out = {"control_rae": cm, "treatment_rae": tm, "matched_delta": delta, "n_seeds_neg": n_neg,
           "c_raes": c_raes, "t_raes": t_raes, "ifdi_imputed": ifdi_nan,
           "deployed_best": prev, "deploy": deploy}
    json.dump(out, open(f"{P}/nb1400_summary.json", "w"), indent=2)
    print(f"\nsaved {P}/nb1400_summary.json -> DEPLOY={deploy}")


if __name__ == "__main__":
    main()
