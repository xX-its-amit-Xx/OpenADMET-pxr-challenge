"""
Open-shell RADICAL spin/charge-distribution add-member gate on the deployed COMP-M config.

DISTINCT sub-axis (ledger c336 / row 401): every prior QM row is closed-shell. cdft already
captured the GLOBAL redox SCALARS (mu/chi/eta/omega) and was ABSORBED (nb1330 +0.0014). The
genuinely-new observable tested here is the SPIN/CHARGE-DISTRIBUTION GEOMETRY of the radical
cation/anion (where the hole / added electron localises, how delocalised, on what atoms).

Control   = deployed COMP-M ensemble: 4-GBM(Xtr+AIM+STR+D4+DB+ORB) + CheMeleon + TabPFN + sisterNR.
Treatment = same + 12 radical scalars (rad_*).
Gate: deploy if treat < ctrl AND delta < -0.001 AND n_neg >= 2/3. Mirrors nb1397/nb1330 exactly.
"""
import sys, json, time, warnings, os
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
os.chdir("D:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge")

from nb1126_combinatorial_search import feature_matrix, CACHE, make_model
from src.pxr.data import load_train
from src.pxr.eval import rae
from sklearn.preprocessing import StandardScaler

SD  = "C:/pxr_work/search"
MTL = "C:/pxr_work/mtl"
BEST_RAE = 0.4149
GATE_THRESHOLD = 0.001
N_SEEDS = 3

t0 = time.time()
print("=== open-shell radical spin-distribution add-member gate on COMP-M control ===")

d   = np.load(CACHE)
ytr = d["ytr"]; n = len(ytr)
Xtr, _ = feature_matrix(d, "combined")

tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
names_tr = tr["name"].tolist()

chem = np.load(f"{SD}/chemeleon_oof.npy")
tab  = np.load(f"{SD}/tabpfn_oof.npy")

def ablk(csv, cols, names):
    df = pd.read_csv(csv)
    df = df[df.src == "train"].drop_duplicates(subset="name", keep="first")
    sub = df.set_index("name").reindex(names)
    X = sub[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    med = np.nanmedian(X, axis=0); ii = np.where(np.isnan(X)); X[ii] = np.take(med, ii[1])
    return X.astype(np.float32)

ACOLS = ["aimnet_energy","aimnet_qmin","aimnet_qmax","aimnet_qabs_mean","aimnet_qstd",
         "aimnet_qsum_abs","aimnet_dipole","aimnet_fmax","aimnet_frms"]
SCOLS = ["strain_relax_mean","strain_relax_max","conf_espread","conf_erange",
         "conf_n","rmsd_mean","rmsd_max","e_per_heavy"]
DCOLS = ["d4_alpha_sum","d4_alpha_mean","d4_alpha_std","d4_alpha_max","d4_c6diag_mean",
         "d4_c6diag_std","d4_c6_total","d4_edisp","d4_edisp_per_atom","d4_cn_mean",
         "d4_cn_max","d4_qeeq_min","d4_qeeq_max","d4_qeeq_std","d4_qeeq_absum"]
DBCOLS = ["vbur_r25","vbur_r35","vbur_r45","vbur_r55","vbur_r65","ster_L",
          "ster_Bmin","ster_Bmax","ster_aniso","npr1","npr2","asphericity",
          "spherocity","eccentricity","radgyr","inertial_sf"]
OCOLS = ["orb_energy","orb_energy_per_ha","orb_fmax","orb_frms","orb_fstd",
         "orb_conf_mean","orb_conf_std","orb_conf_node_mean","orb_conf_node_std",
         "orb_conf_node_min","orb_node_emb_mean","orb_node_emb_std","orb_node_emb_norm"]
RADCOLS = ["rad_ip_ev","rad_ea_ev","rad_hole_max","rad_hole_pr","rad_hole_fhet","rad_hole_farom",
           "rad_el_max","rad_el_pr","rad_el_fhet","rad_el_farom","rad_somo_cat","rad_dq_asym"]

Xqm  = ablk("C:/pxr_work/aimnet2/aimnet_features.csv", ACOLS, names_tr)
Xst  = ablk("C:/pxr_work/strain/strain_features.csv",  SCOLS, names_tr)
Xd4  = ablk("C:/pxr_work/d4/d4_features.csv",          DCOLS, names_tr)
Xdb  = ablk("C:/pxr_work/dbstep/dbstep_features.csv",  DBCOLS, names_tr)
Xorb = ablk("C:/pxr_work/orbmol/orbmol_features.csv",  OCOLS, names_tr)
Xrad = ablk("C:/pxr_work/radical/radical_features.csv", RADCOLS, names_tr)
print(f"Blocks: AIM={Xqm.shape[1]} STR={Xst.shape[1]} D4={Xd4.shape[1]} DB={Xdb.shape[1]} "
      f"ORB={Xorb.shape[1]} RAD={Xrad.shape[1]}  t={time.time()-t0:.0f}s")

n_ok = int((~np.isnan(Xrad).any(axis=1)).sum())
print(f"RAD coverage before imputation: {n_ok}/{n} ({100*n_ok/n:.1f}%)")

nb3200_path = "data/processed/oof_chemprop_aux.npy"
if os.path.exists(nb3200_path):
    nb3200_oof = np.load(nb3200_path)
    err = ytr - nb3200_oof
    print("corr(nb3200_err, rad_*):")
    for j, col in enumerate(RADCOLS):
        c = float(np.corrcoef(err, Xrad[:, j])[0, 1])
        print(f"   {col:14s} {c:+.4f}")

LOG = f"{SD}/results.jsonl"
def topK(archs):
    recs = [json.loads(l) for l in open(LOG) if l.strip()]
    v = [r for r in recs if "ps_rae" in r and "error" not in r and r["arch"] in archs]
    v.sort(key=lambda r: r["ps_rae"]); bp = {}
    for r in v: bp.setdefault(r["arch"], r)
    return list(bp.values())

tk = topK(("lgbm","xgb","cat","histgb"))
print(f"GBM archs: {[c['arch'] for c in tk]}")

ctrl_raes, treat_raes = [], []
for seed in range(N_SEEDS):
    ho  = np.load(f"{MTL}/ho_idx_seed{seed}.npy")
    hs  = set(ho.tolist())
    trn = np.array([i for i in range(n) if i not in hs])
    lo, hi = np.quantile(ytr[trn], 0.05), np.quantile(ytr[trn], 0.98)

    sc_list = [StandardScaler().fit(X[trn]) for X in (Xqm, Xst, Xd4, Xdb, Xorb)]
    Xex_ctrl  = np.hstack([s.transform(X) for s, X in zip(sc_list, (Xqm, Xst, Xd4, Xdb, Xorb))])
    sc_rad = StandardScaler().fit(Xrad[trn])
    Xex_treat = np.hstack([Xex_ctrl, sc_rad.transform(Xrad)])

    gbm_c, gbm_t = [], []
    for c in tk:
        Ac = np.hstack([Xtr, Xex_ctrl]);  m = make_model(c["arch"], c["hp"]); m.fit(Ac[trn], ytr[trn]); gbm_c.append(m.predict(Ac[ho]))
        At = np.hstack([Xtr, Xex_treat]); m = make_model(c["arch"], c["hp"]); m.fit(At[trn], ytr[trn]); gbm_t.append(m.predict(At[ho]))

    sn = np.load(f"{MTL}/sn_oof_seed{seed}.npy").ravel()
    ctrl  = np.clip(np.mean(gbm_c + [chem[ho], tab[ho], sn], 0), lo, hi)
    treat = np.clip(np.mean(gbm_t + [chem[ho], tab[ho], sn], 0), lo, hi)

    rc, rt = rae(ytr[ho], ctrl), rae(ytr[ho], treat)
    ctrl_raes.append(rc); treat_raes.append(rt)
    d_ = rt - rc
    print(f"  seed={seed}  ctrl={rc:.4f}  treat={rt:.4f}  delta={d_:+.4f}  {'NEG' if d_<0 else 'pos'}", flush=True)

cm = float(np.mean(ctrl_raes)); tm = float(np.mean(treat_raes))
delta = tm - cm
n_neg = sum(1 for t, c in zip(treat_raes, ctrl_raes) if t < c)

print(f"\n=== radical spin-distribution SUMMARY ===")
print(f"Control  (deployed, no RAD): {cm:.4f}")
print(f"Treatment (+RAD open-shell): {tm:.4f}")
print(f"Matched delta:               {delta:+.4f}")
print(f"Neg seeds:                   {n_neg}/{N_SEEDS}")
print(f"Deployed best (nb1328):      {BEST_RAE}")
print(f"Wall time: {(time.time()-t0)/60:.1f} min")

GATE_PASS = bool(delta < -GATE_THRESHOLD and n_neg >= 2)
print(f"GATE PASS: {GATE_PASS}")

out = {
    "approach": "open_shell_radical_spin_distribution_12scalars_on_deployed_COMP-M_config",
    "axis": "open-shell radical cation/anion spin & charge-distribution geometry (UHF GFN2-xTB)",
    "ctrl_rae": cm, "treat_rae": tm, "delta": delta,
    "ctrl_raes": ctrl_raes, "treat_raes": treat_raes,
    "n_neg": n_neg, "deployed_best": BEST_RAE, "gate_pass": GATE_PASS,
}
with open("data/processed/nb1398_radical_summary.json", "w") as f:
    json.dump(out, f, indent=2)
print("Saved: data/processed/nb1398_radical_summary.json")
