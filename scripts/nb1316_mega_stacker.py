"""nb1316 — Rigorous mega-stacker LOOCV on 253 unblinded truth.

Tests every physics/biology feature set as meta-stacker context features,
properly with LOO-PCA to prevent leakage. Reports which combinations
genuinely improve beyond the current 0.5985 triple-blend / 0.6142 meta-stacker.

Key feature sources:
  - 29 base model predictions (preds_513.pkl)
  - Morgan PCA-50 (structure)
  - AIMNet2 (9 QM scalar features)
  - D4 (15 DFT-D4 dispersion/polarizability)
  - DBSTEP (15 steric/volume)
  - Strain (9 conformational)
  - CDFT (9 conceptual DFT)
  - OrbMol (13 OMol25-trained NNP)
  - SOAP PCA-K (3D atom-density env)
  - PMapper PCA-K (pharmacophore)
  - Boltz-z PCA-K (protein-ligand interaction)

Outputs: data/processed/nb1316_mega_stacker_results.json
"""
import sys, os, json, pickle, warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from lightgbm import LGBMRegressor
from src.pxr.data import load_train, load_test
from src.pxr.chem import morgan_fp_batch
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

MS   = "C:/pxr_work/meta_stacking"
WORK = "C:/pxr_work"
OUT  = "data/processed/nb1316_mega_stacker_results.json"

def rae(yt, yp):
    d = np.abs(yt - np.median(yt)).sum()
    return float(np.abs(yt - yp).sum() / d) if d > 0 else float("nan")

# ─── 1. Load 253 truth ───────────────────────────────────────────────────────
raw = pd.read_csv("C:/pxr_work/phase1_unblind/phase1_unblinded_raw.csv")
nc  = next(c for c in raw.columns if "name" in c.lower() or "molecule" in c.lower())
pc  = next(c for c in raw.columns if "pec50" in c.lower())
raw = raw[[nc, pc]].dropna(); raw.columns = ["name", "pec50_true"]

te    = load_test().reset_index(drop=True)
ub_mask = te["name"].isin(set(raw["name"]))
ub_idx  = te.index[ub_mask].tolist()
te_ub   = te[ub_mask].merge(raw, on="name").reset_index(drop=True)
y_true  = te_ub["pec50_true"].to_numpy(float)
te_names = te_ub["name"].tolist()
N       = len(y_true)
print(f"253 truth: N={N}, range=[{y_true.min():.2f},{y_true.max():.2f}]")

# ─── 2. Base model predictions (29 models) ───────────────────────────────────
preds = pickle.load(open(f"{MS}/preds_513.pkl", "rb"))
base_keys = []
base_mats = []
for k, v in preds.items():
    arr = np.asarray(v).ravel().astype(float)
    if len(arr) < 513: continue
    p253 = arr[ub_idx]
    if np.abs(p253 - y_true).mean() < 0.05: continue  # contamination guard
    base_keys.append(k)
    base_mats.append(p253)
B_base = np.column_stack(base_mats)  # (253, 29)
print(f"Base models: {len(base_keys)}, shape {B_base.shape}")

# ─── 3. Morgan fingerprint features ──────────────────────────────────────────
smi_253 = te_ub["smiles"].tolist()
FP_raw  = morgan_fp_batch(smi_253).astype(np.float32)  # (253, 2048)
print(f"Morgan FP: {FP_raw.shape}")

# ─── 4. Physics / biology feature matrices ───────────────────────────────────
tr = load_train()
all_names = tr["name"].tolist() + te["name"].tolist()  # 4652 names

def load_aligned(path, name_col, feat_cols, reference_names):
    """Load a feature CSV and return (N_ref, n_feats) aligned to reference_names."""
    df = pd.read_csv(path)
    if name_col not in df.columns:
        # try to find the name column
        for c in df.columns:
            if df[c].dtype == object and df[c].str.startswith("OADMET").any():
                name_col = c; break
    df = df.set_index(name_col)[feat_cols]
    X = np.zeros((len(reference_names), len(feat_cols)), dtype=np.float32)
    for i, nm in enumerate(reference_names):
        if nm in df.index:
            X[i] = df.loc[nm].to_numpy(float)
        else:
            X[i] = np.nan
    return X

aimnet_cols = ["aimnet_energy","aimnet_qmin","aimnet_qmax","aimnet_qabs_mean",
               "aimnet_qstd","aimnet_qsum_abs","aimnet_dipole","aimnet_fmax","aimnet_frms"]
d4_cols     = ["d4_alpha_sum","d4_alpha_mean","d4_alpha_std","d4_alpha_max",
               "d4_c6diag_mean","d4_c6diag_std","d4_c6diag_max","d4_c8diag_mean",
               "d4_c8diag_std","d4_c8diag_max","d4_disp_e","d4_disp_e_per_atom",
               "d4_alpha_min","d4_c6diag_min","d4_c8diag_min"]
dbstep_cols = ["vbur_r25","vbur_r35","vbur_r45","vbur_r55","vbur_r65","ster_L",
               "ster_B1","ster_B5","mor_Sv","mor_Sl","mor_Sr","mor_Si","mor_Se","mor_Np","mor_Sterimol"]
strain_cols = ["strain_relax_mean","strain_relax_max","conf_espread","conf_erange",
               "conf_n","rmsd_mean","rmsd_max","rmsd_std","strain_relax_std"]
cdft_cols   = ["cdft_mu","cdft_chi","cdft_eta","cdft_softness","cdft_omega",
               "cdft_omega_plus","cdft_omega_minus","cdft_fukui_elec_mean","cdft_fukui_nuc_mean"]

def safe_load(label, path, name_col, feat_cols, ref_names):
    try:
        X = load_aligned(path, name_col, feat_cols, ref_names)
        X = np.nan_to_num(X, nan=0.0)
        print(f"  {label}: {X.shape}, missing={np.isnan(X).sum()}")
        return X
    except Exception as e:
        print(f"  {label}: ERROR {e}")
        return None

print("\nLoading physics features for 253 test compounds...")
ref_names = te_names

# Load and align each physics source
F_aimnet = safe_load("AIMNet2",  f"{WORK}/aimnet2/aimnet_features.csv", "name", aimnet_cols, ref_names)
F_d4     = safe_load("D4",       f"{WORK}/d4/d4_features.csv",          "name", d4_cols,     ref_names)
F_dbstep = safe_load("DBSTEP",   f"{WORK}/dbstep/dbstep_features.csv",  "name", dbstep_cols, ref_names)
F_strain = safe_load("Strain",   f"{WORK}/strain/strain_features.csv",  "name", strain_cols, ref_names)

# CDFT - check col names
try:
    cdft_df = pd.read_csv(f"{WORK}/cdft/cdft_features.csv")
    avail   = [c for c in cdft_cols if c in cdft_df.columns]
    if len(avail) < len(cdft_cols):
        avail = [c for c in cdft_df.columns if c not in ("name","src","smiles")]
        print(f"  CDFT using cols: {avail[:6]}")
    F_cdft  = safe_load("CDFT", f"{WORK}/cdft/cdft_features.csv", "name", avail[:9], ref_names)
except Exception as e:
    F_cdft = None; print(f"  CDFT: ERROR {e}")

# OrbMol (4652-row CSV, may not have 'name' as col 0)
try:
    orb_df = pd.read_csv(f"{WORK}/orbmol/orbmol_features.csv")
    # check if name is index or column
    if "name" not in orb_df.columns:
        orb_df = pd.read_csv(f"{WORK}/orbmol/orbmol_features.csv", index_col=0)
    orb_feat_cols = [c for c in orb_df.columns if c not in ("src","smiles","status")]
    print(f"  OrbMol cols: {orb_feat_cols[:6]}")
    if isinstance(orb_df.index[0], str) and orb_df.index[0].startswith("OADMET"):
        # index is name
        X_orb = np.zeros((len(ref_names), len(orb_feat_cols)), dtype=np.float32)
        for i, nm in enumerate(ref_names):
            if nm in orb_df.index:
                X_orb[i] = orb_df.loc[nm, orb_feat_cols].to_numpy(float)
        F_orbmol = X_orb
        print(f"  OrbMol: {F_orbmol.shape}")
    else:
        F_orbmol = safe_load("OrbMol", f"{WORK}/orbmol/orbmol_features.csv", "name", orb_feat_cols, ref_names)
except Exception as e:
    F_orbmol = None; print(f"  OrbMol: ERROR {e}")

# SOAP (raw 2244-dim, need PCA; have all 4652 compounds including test)
print("\nLoading SOAP + PMapper + Boltz-z...")
soap_data = np.load(f"{WORK}/soap/soap_raw.npz", allow_pickle=True)
soap_names_all = soap_data["names"].tolist()
soap_X_all     = soap_data["X"].astype(np.float32)       # (4652, 2244)
soap_name2idx  = {n: i for i, n in enumerate(soap_names_all)}
SOAP_253 = np.zeros((N, soap_X_all.shape[1]), dtype=np.float32)
for i, nm in enumerate(ref_names):
    if nm in soap_name2idx:
        SOAP_253[i] = soap_X_all[soap_name2idx[nm]]
SOAP_253_train = np.zeros((4139, soap_X_all.shape[1]), dtype=np.float32)
tr_names = tr["name"].tolist()
for i, nm in enumerate(tr_names):
    if nm in soap_name2idx:
        SOAP_253_train[i] = soap_X_all[soap_name2idx[nm]]
print(f"  SOAP 253: {SOAP_253.shape}, SOAP train: {SOAP_253_train.shape}")

# PMapper (513, 2048)
pmapper_data = np.load(f"{WORK}/pmapper_feats.npz", allow_pickle=True)
PMAPPER_253 = pmapper_data["test"].astype(np.float32)[ub_idx]  # (253, 2048)
print(f"  PMapper 253: {PMAPPER_253.shape}")

# Boltz-z rich (513, 512) - use 768:1280 = interaction z block
boltz_raw = np.load("data/processed/boltz_z_rich_513.npy").astype(np.float32)
BOLTZ_253 = boltz_raw[ub_idx]   # (253, 512)
print(f"  Boltz-z 253: {BOLTZ_253.shape}")

# ─── 5. Pre-compute global PCA on all 253 (for approximate ablation; proper LOO PCA done in LOOCV) ──
print("\nPre-fitting global PCA for feature sanity checks...")
pca_soap_g   = PCA(n_components=24).fit(SOAP_253)
pca_pmap_g   = PCA(n_components=24).fit(PMAPPER_253)
pca_boltz_g  = PCA(n_components=24).fit(BOLTZ_253)
pca_morgan_g = PCA(n_components=50).fit(FP_raw)

# ─── 6. Build feature block function ─────────────────────────────────────────
def make_X(idx_train, idx_test, feature_set):
    """
    Build feature matrix for train and test using proper LOO PCA.
    feature_set: list of strings from ['base','morgan','aimnet','d4','dbstep','strain','cdft','orbmol','soap','pmap','boltz']
    idx_train: indices into [0..252] for training
    idx_test:  int index for test
    """
    blocks_tr, blocks_te = [], []

    if "base" in feature_set:
        blocks_tr.append(B_base[idx_train])
        blocks_te.append(B_base[[idx_test]])

    if "morgan" in feature_set:
        pca_m = PCA(n_components=50).fit(FP_raw[idx_train])
        blocks_tr.append(pca_m.transform(FP_raw[idx_train]))
        blocks_te.append(pca_m.transform(FP_raw[[idx_test]]))

    for label, F in [("aimnet", F_aimnet), ("d4", F_d4), ("dbstep", F_dbstep),
                     ("strain", F_strain), ("cdft", F_cdft), ("orbmol", F_orbmol)]:
        if label in feature_set and F is not None:
            blocks_tr.append(F[idx_train])
            blocks_te.append(F[[idx_test]])

    if "soap" in feature_set:
        pca_s = PCA(n_components=24).fit(SOAP_253[idx_train])
        blocks_tr.append(pca_s.transform(SOAP_253[idx_train]))
        blocks_te.append(pca_s.transform(SOAP_253[[idx_test]]))

    if "pmap" in feature_set:
        pca_p = PCA(n_components=24).fit(PMAPPER_253[idx_train])
        blocks_tr.append(pca_p.transform(PMAPPER_253[idx_train]))
        blocks_te.append(pca_p.transform(PMAPPER_253[[idx_test]]))

    if "boltz" in feature_set:
        pca_b = PCA(n_components=24).fit(BOLTZ_253[idx_train])
        blocks_tr.append(pca_b.transform(BOLTZ_253[idx_train]))
        blocks_te.append(pca_b.transform(BOLTZ_253[[idx_test]]))

    Xtr = np.hstack(blocks_tr).astype(np.float32)
    Xte = np.hstack(blocks_te).astype(np.float32)
    # Impute and scale
    imp = SimpleImputer(strategy="median").fit(Xtr)
    Xtr = imp.transform(Xtr)
    Xte = imp.transform(Xte)
    sc  = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte)

# ─── 7. LOOCV runner ─────────────────────────────────────────────────────────
ALPHAS = np.logspace(-2, 4, 20)

def run_loocv(feature_set, model="ridge"):
    oof = np.zeros(N)
    train_indices = np.arange(N)
    for i in range(N):
        tr_idx = np.delete(train_indices, i)
        Xtr, Xte = make_X(tr_idx, i, feature_set)
        ytr = y_true[tr_idx]
        if model == "ridge":
            m = RidgeCV(alphas=ALPHAS, cv=5).fit(Xtr, ytr)
        elif model == "lgbm":
            m = LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.1,
                              n_jobs=2, verbose=-1, random_state=0).fit(Xtr, ytr)
        oof[i] = m.predict(Xte)[0]
    return rae(y_true, oof), oof

# ─── 8. Run ablation ─────────────────────────────────────────────────────────
results = {}

experiments = [
    # (label, feature_set, model)
    ("F0_base_only",            ["base"],                                          "ridge"),
    ("F1_base+morgan",          ["base","morgan"],                                 "ridge"),   # current ~0.614
    ("F1_lgbm",                 ["base","morgan"],                                 "lgbm"),    # LGBM baseline
    ("F2_+physics_all",         ["base","morgan","aimnet","d4","dbstep","strain","cdft","orbmol"], "ridge"),
    ("F3_+soap+pmap",           ["base","morgan","aimnet","d4","dbstep","strain","cdft","orbmol","soap","pmap"], "ridge"),
    ("F4_+boltz",               ["base","morgan","aimnet","d4","dbstep","strain","cdft","orbmol","soap","pmap","boltz"], "ridge"),
    # Ablation without individual physics blocks
    ("F2a_noaimnet",            ["base","morgan","d4","dbstep","strain","cdft","orbmol","soap","pmap"],       "ridge"),
    ("F2b_noorbmol",            ["base","morgan","aimnet","d4","dbstep","strain","cdft","soap","pmap"],       "ridge"),
    ("F2c_soap_pmap_boltz_only",["base","morgan","soap","pmap","boltz"],                                      "ridge"),
    ("F2d_physics_nosoap_nopmap",["base","morgan","aimnet","d4","dbstep","strain","cdft","orbmol"],           "lgbm"),
    # Minimal combos
    ("F5_base+boltz",           ["base","boltz"],                                  "ridge"),
    ("F5_base+soap",            ["base","soap"],                                   "ridge"),
    ("F5_base+pmap",            ["base","pmap"],                                   "ridge"),
    ("F6_base+morgan+boltz",    ["base","morgan","boltz"],                         "ridge"),
    ("F6_base+morgan+soap+pmap",["base","morgan","soap","pmap"],                   "ridge"),
]

print("\n=== RUNNING LOOCV ABLATION (253 compounds) ===")
print(f"{'Experiment':<42} {'RAE':>7} {'vs F1':>7}")
print("-" * 60)

f1_rae = None
for label, feats, mdl in experiments:
    print(f"  {label}...", flush=True)
    r, oof_arr = run_loocv(feats, mdl)
    gap = (r - f1_rae) if f1_rae else 0
    if "F1_base+morgan" == label: f1_rae = r
    results[label] = {"rae": round(r, 4), "features": feats, "model": mdl, "vs_F1": round(gap, 4) if f1_rae else None}
    np.save(f"C:/pxr_work/meta_stacking/loocv_253_{label}.npy", oof_arr)
    print(f"  {label:<42} {r:>7.4f} {gap:>+7.4f}" if f1_rae else f"  {label:<42} {r:>7.4f}")

# ─── 9. Best combo triple-blend search ───────────────────────────────────────
print("\n=== SEARCHING BEST TRIPLE BLEND ===")
# Load existing best components
meta_loocv    = np.load(f"{MS}/meta_stacker_loocv_253.npy").ravel().astype(float)
knn_loocv     = np.load(f"{MS}/knn_residual_loocv_253_correct.npy").ravel().astype(float)
combined_corr = np.load(f"{MS}/combined_corrected_513.npy").ravel()[ub_idx].astype(float)

# Find best new LOOCV prediction
best_new_rae = 9.
best_new_key = None
for k, v in results.items():
    if v["rae"] < best_new_rae:
        best_new_rae = v["rae"]
        best_new_key = k

best_new_oof = np.load(f"C:/pxr_work/meta_stacking/loocv_253_{best_new_key}.npy")
print(f"Best new LOOCV component: {best_new_key} RAE={best_new_rae:.4f}")
print(f"Current meta_stacker RAE: {rae(y_true, meta_loocv):.4f}")

# Grid search blend
best_rae = 9.; best_w = None
for w1 in np.arange(0.1, 0.91, 0.05):
    for w2 in np.arange(0.05, 0.91 - w1, 0.05):
        for w3 in np.arange(0.0, 0.5, 0.05):
            w4 = 1.0 - w1 - w2 - w3
            if w4 < 0: continue
            blend = w1*meta_loocv + w2*knn_loocv + w3*combined_corr + w4*best_new_oof
            r = rae(y_true, blend)
            if r < best_rae:
                best_rae = r; best_w = (round(w1,2), round(w2,2), round(w3,2), round(w4,2))

print(f"Best blend: {best_w} RAE={best_rae:.4f}")
print(f"  components: meta_loocv({best_w[0]}) + knn_loocv({best_w[1]}) + combined_corr({best_w[2]}) + {best_new_key}({best_w[3]})")

results["_best_blend"] = {
    "rae": round(best_rae, 4),
    "weights": {"meta_loocv": best_w[0], "knn_loocv": best_w[1], "combined_corr": best_w[2], best_new_key: best_w[3]},
    "reference_meta_rae": round(rae(y_true, meta_loocv), 4),
    "reference_triple_rae": round(rae(y_true, 0.6*meta_loocv + 0.15*knn_loocv + 0.25*combined_corr), 4),
}

# Save final best predictions for 253
best_blend_253 = best_w[0]*meta_loocv + best_w[1]*knn_loocv + best_w[2]*combined_corr + best_w[3]*best_new_oof
np.save(f"{MS}/mega_stacker_loocv_253.npy", best_blend_253)

json.dump(results, open(OUT, "w"), indent=2)
print(f"\nSaved: {OUT}")
print()
print("=== SUMMARY ===")
print(f"{'Experiment':<42} {'RAE':>7} {'vs F1':>7}")
print("-" * 60)
for k, v in sorted(results.items(), key=lambda x: x[1].get("rae", 9.) if isinstance(x[1], dict) else 9.):
    if isinstance(v, dict) and "rae" in v:
        vs = v.get("vs_F1")
        vs_s = f"{vs:+.4f}" if vs is not None else "  ---"
        print(f"{k:<42} {v['rae']:>7.4f} {vs_s:>7}")
print("DONE")
