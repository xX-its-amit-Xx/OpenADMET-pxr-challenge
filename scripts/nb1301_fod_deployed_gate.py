"""
nb1301 — FOD (Fractional Occupation Density) GFN2-xTB feature gate on DEPLOYED config.
Control  = 4-GBM(combined+AIMNet2+strain+D4+DBSTEP) + CheMeleon + TabPFN + sn_oof [best=0.4242]
Treatment = same but with 9 FOD scalars appended to the GBM feature vector.
NOTE: FOD is a finite-temperature (5000K) static-correlation observable DISTINCT from the closed
      xTB HOMO-LUMO/charge block (+0.0007 absorbed). Prior is LOW (semi-empirical xTB again) but
      this is a distinct observable axis (fractional orbital occupation / reactive-electron density).
Gate: 3 seeds, matched CTRL, report delta; deploy if delta < -0.001 AND treat < best_ensemble.
"""
import os, sys, json
import numpy as np
PROJ = "D:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge"
sys.path.insert(0, PROJ)
sys.path.insert(0, f"{PROJ}/scripts")
from nb1126_combinatorial_search import feature_matrix, CACHE, make_model
from src.pxr.data import load_train
from src.pxr.eval import rae
from sklearn.preprocessing import StandardScaler
import pandas as pd
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

P    = f"{PROJ}/data/processed"
SD   = "C:/pxr_work/search"
FOD  = "C:/pxr_work/fod/fod_features_merged.csv"
BEST = f"{SD}/best_ensemble.json"
LOG  = f"{SD}/results.jsonl"
N_SEEDS = 3

AIM   = "C:/pxr_work/aimnet2/aimnet_features.csv"
ACOLS = ["aimnet_energy","aimnet_qmin","aimnet_qmax","aimnet_qabs_mean","aimnet_qstd",
         "aimnet_qsum_abs","aimnet_dipole","aimnet_fmax","aimnet_frms"]
STR   = "C:/pxr_work/strain/strain_features.csv"
SCOLS = ["strain_relax_mean","strain_relax_max","conf_espread","conf_erange","conf_n",
         "rmsd_mean","rmsd_max","e_per_heavy"]
D4    = "C:/pxr_work/d4/d4_features.csv"
DCOLS = ["d4_alpha_sum","d4_alpha_mean","d4_alpha_std","d4_alpha_max","d4_c6diag_mean",
         "d4_c6diag_std","d4_c6_total","d4_edisp","d4_edisp_per_atom","d4_cn_mean",
         "d4_cn_max","d4_qeeq_min","d4_qeeq_max","d4_qeeq_std","d4_qeeq_absum"]
DB    = "C:/pxr_work/dbstep/dbstep_features.csv"
DBCOLS= ["vbur_r25","vbur_r35","vbur_r45","vbur_r55","vbur_r65","ster_L","ster_Bmin",
         "ster_Bmax","ster_aniso","npr1","npr2","asphericity","spherocity","eccentricity",
         "radgyr","inertial_sf"]

FODCOLS = ["fod_nfod","fod_nfod_per_ha","fod_max_dev","fod_n_frac",
           "fod_gap_0k","fod_gap_5k","fod_fermi_0k","fod_fermi_5k","fod_lumo_0k"]


def topK(archs):
    recs  = [json.loads(l) for l in open(LOG) if l.strip()]
    valid = [r for r in recs if "ps_rae" in r and "error" not in r and r["arch"] in archs]
    valid.sort(key=lambda r: r["ps_rae"]); bp = {}
    for r in valid: bp.setdefault(r["arch"], r)
    return list(bp.values())


def aligned(csv, cols, names):
    df  = pd.read_csv(csv)
    name_col = next((c for c in df.columns if "name" in c.lower()), df.columns[0])
    if "src" in df.columns:
        df = df[df.src == "train"].drop_duplicates(subset=name_col, keep="first")
    else:
        df = df.drop_duplicates(subset=name_col, keep="first")
    sub = df.set_index(name_col).reindex(names)
    X   = sub[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    med = np.nanmedian(X, axis=0)
    ix  = np.where(np.isnan(X)); X[ix] = np.take(med, ix[1])
    return X


def aligned_test(csv, cols, test_names):
    df  = pd.read_csv(csv)
    name_col = next((c for c in df.columns if "name" in c.lower()), df.columns[0])
    if "src" in df.columns:
        df = df[df.src == "test"].drop_duplicates(subset=name_col, keep="first")
    else:
        df = df.drop_duplicates(subset=name_col, keep="first")
    sub = df.set_index(name_col).reindex(test_names)
    X   = sub[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    med = np.nanmedian(X, axis=0)
    ix  = np.where(np.isnan(X)); X[ix] = np.take(med, ix[1])
    return X


def fit_pred(c, Xfeat, ytr, use, ho):
    m = make_model(c["arch"], c["hp"])
    if c["arch"] in ("ridge", "enet"):
        sc = StandardScaler().fit(Xfeat[use])
        m.fit(sc.transform(Xfeat[use]), ytr[use])
        return m.predict(sc.transform(Xfeat[ho]))
    m.fit(Xfeat[use], ytr[use])
    return m.predict(Xfeat[ho])


def main():
    d    = np.load(CACHE); ytr = d["ytr"]; se = d["se"]; n_tr = len(ytr)
    Xtr, _ = feature_matrix(d, "combined")
    chem = np.load(f"{SD}/chemeleon_oof.npy")
    tab  = np.load(f"{SD}/tabpfn_oof.npy")

    tr   = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    Xqm  = aligned(AIM,   ACOLS,  tr["name"])
    Xstr = aligned(STR,   SCOLS,  tr["name"])
    Xd4  = aligned(D4,    DCOLS,  tr["name"])
    Xdb  = aligned(DB,    DBCOLS, tr["name"])
    Xfod = aligned(FOD,   FODCOLS, tr["name"])

    Xctrl = np.hstack([Xtr, Xqm, Xstr, Xd4, Xdb])
    Xtreat= np.hstack([Xtr, Xqm, Xstr, Xd4, Xdb, Xfod])

    configs = topK(["lgbm","xgb","histgb","cat"])
    ho_idx  = np.load(f"{SD}/ho_idx.npy")
    sn_oof  = np.load(f"{SD}/sn_oof.npy")

    best_meta = json.load(open(BEST))
    best_rae  = best_meta["rae"]

    c_raes, t_raes = [], []
    for seed in range(N_SEEDS):
        rng  = np.random.default_rng(seed * 1000 + 42)
        perm = rng.permutation(n_tr)
        ho   = perm[:ho_idx]; use = perm[ho_idx:]

        ctrl_preds = []
        for c in configs:
            pred = fit_pred(c, Xctrl, ytr, use, ho)
            ctrl_preds.append(pred)
        ctrl_avg = np.mean(ctrl_preds, axis=0)

        treat_preds = []
        for c in configs:
            pred = fit_pred(c, Xtreat, ytr, use, ho)
            treat_preds.append(pred)
        treat_avg = np.mean(treat_preds, axis=0)

        # Add CheMeleon + TabPFN + sn_oof (fixed components, not retrained)
        chem_ho  = chem[perm[:ho_idx]]
        tab_ho   = tab[perm[:ho_idx]]
        sn_ho    = sn_oof[perm[:ho_idx]]

        ctrl_ens  = (ctrl_avg  + chem_ho + tab_ho + sn_ho) / 4.0
        treat_ens = (treat_avg + chem_ho + tab_ho + sn_ho) / 4.0

        c_rae = rae(ytr[ho], ctrl_ens)
        t_rae = rae(ytr[ho], treat_ens)
        c_raes.append(c_rae); t_raes.append(t_rae)
        print(f"  seed={seed} ctrl={c_rae:.4f} treat={t_rae:.4f} delta={t_rae-c_rae:+.4f}")

    ctrl_mean  = float(np.mean(c_raes))
    treat_mean = float(np.mean(t_raes))
    delta      = treat_mean - ctrl_mean
    neg_seeds  = sum(t < c for t, c in zip(t_raes, c_raes))
    deploy     = (delta < -0.001) and (treat_mean < best_rae) and (neg_seeds >= 2)

    # Residual correlation as secondary signal
    try:
        nb3200 = np.load(f"{P}/te_nb3200.npy")[:n_tr]
        err3200 = ytr - nb3200
        from scipy.stats import spearmanr
        fod_signal = Xfod[:, 0]  # N_FOD scalar
        corr, pval = spearmanr(fod_signal, np.abs(err3200))
        resid_corr = float(corr)
    except Exception:
        resid_corr = float("nan")

    result = {
        "control_rae":  ctrl_mean,
        "treatment_rae": treat_mean,
        "matched_delta": delta,
        "c_raes": c_raes,
        "t_raes": t_raes,
        "neg_seeds": neg_seeds,
        "deployed_best": best_rae,
        "deploy": deploy,
        "resid_corr": resid_corr,
        "feature": "fod_9scalars",
    }
    print("\n=== FOD GATE RESULT ===")
    print(json.dumps(result, indent=2))
    import pathlib
    pathlib.Path(f"{P}/nb1301_fod_summary.json").write_text(json.dumps(result, indent=2))
    print(f"\nDeploy={deploy} | ctrl={ctrl_mean:.4f} | treat={treat_mean:.4f} | delta={delta:+.4f} | neg_seeds={neg_seeds}/3")
    return result


if __name__ == "__main__":
    main()
