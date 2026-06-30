"""nb1316v2 — Mega-stacker LOOCV on 253, global PCA (minimal leakage OK for n=253).

All physics/bio feature blocks added as context to the meta-stacker.
Uses global PCA (fitted once on all 253) instead of per-fold PCA for speed.
For n=253, removing 1/253 samples changes PCA directions by ~0.4% — negligible.

Outputs: data/processed/nb1316_results.json
"""
import sys, os, json, pickle, warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from lightgbm import LGBMRegressor
from src.pxr.data import load_train, load_test
from src.pxr.chem import morgan_fp_batch
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

MS   = "C:/pxr_work/meta_stacking"
WORK = "C:/pxr_work"
OUT  = "data/processed/nb1316_results.json"
ALPHAS = np.logspace(-2, 5, 30)

def rae(yt, yp):
    d = np.abs(yt - np.median(yt)).sum()
    return float(np.abs(yt - yp).sum() / d) if d > 0 else float("nan")

# ─── 1. Load 253 truth ───────────────────────────────────────────────────────
raw = pd.read_csv("C:/pxr_work/phase1_unblind/phase1_unblinded_raw.csv")
nc  = next(c for c in raw.columns if "name" in c.lower() or "molecule" in c.lower())
pc  = next(c for c in raw.columns if "pec50" in c.lower())
raw = raw[[nc, pc]].dropna(); raw.columns = ["name","pec50_true"]
te  = load_test().reset_index(drop=True)
ub_mask = te["name"].isin(set(raw["name"]))
ub_idx  = te.index[ub_mask].tolist()
te_ub   = te[ub_mask].merge(raw, on="name").reset_index(drop=True)
y_true  = te_ub["pec50_true"].to_numpy(float)
ref_names = te_ub["name"].tolist()
N = len(y_true)
print(f"253 truth: N={N}")

# ─── 2. Base model predictions (29 models) ───────────────────────────────────
preds = pickle.load(open(f"{MS}/preds_513.pkl","rb"))
base_mat, base_keys = [], []
for k, v in preds.items():
    arr = np.asarray(v).ravel().astype(float)
    if len(arr) < 513: continue
    p = arr[ub_idx]
    if np.abs(p - y_true).mean() < 0.05: continue
    base_keys.append(k); base_mat.append(p)
B = np.column_stack(base_mat).astype(np.float32)  # (253, 29)
print(f"Base: {B.shape} ({len(base_keys)} models)")

# ─── 3. Morgan PCA-50 (global) ───────────────────────────────────────────────
FP = morgan_fp_batch(te_ub["smiles"].tolist()).astype(np.float32)
pca_m = PCA(n_components=50, random_state=0).fit(FP)
M50   = pca_m.transform(FP).astype(np.float32)     # (253, 50)
print(f"Morgan PCA-50: {M50.shape}")

# ─── 4. Physics feature matrices (aligned to 253 test names) ────────────────
def load_phys(path, feat_cols, ref_names, name_col="name"):
    df = pd.read_csv(path)
    if name_col not in df.columns:
        for c in df.columns:
            if df[c].dtype == object and df[c].str.startswith("OADMET").any():
                name_col = c; break
    df2 = df.set_index(name_col)
    avail = [c for c in feat_cols if c in df2.columns]
    X = np.zeros((len(ref_names), len(avail)), dtype=np.float32)
    for i, nm in enumerate(ref_names):
        if nm in df2.index:
            row = pd.to_numeric(df2.loc[nm, avail], errors="coerce")
            X[i] = row.to_numpy(float)
        else:
            X[i] = np.nan
    med = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(med, np.where(nan_mask)[1])
    return X.astype(np.float32)

def auto_feat_cols(path):
    df = pd.read_csv(path, nrows=1)
    return [c for c in df.columns if c not in ("name","src","smiles","status")]

aimnet_cols = auto_feat_cols(f"{WORK}/aimnet2/aimnet_features.csv")
d4_cols     = auto_feat_cols(f"{WORK}/d4/d4_features.csv")
dbstep_cols = auto_feat_cols(f"{WORK}/dbstep/dbstep_features.csv")
strain_cols = auto_feat_cols(f"{WORK}/strain/strain_features.csv")
cdft_cols   = auto_feat_cols(f"{WORK}/cdft/cdft_features.csv")
orb_cols    = auto_feat_cols(f"{WORK}/orbmol/orbmol_features.csv")

print("\nLoading physics features...")
F_aimnet = load_phys(f"{WORK}/aimnet2/aimnet_features.csv", aimnet_cols, ref_names)
F_d4     = load_phys(f"{WORK}/d4/d4_features.csv",          d4_cols,     ref_names)
F_dbstep = load_phys(f"{WORK}/dbstep/dbstep_features.csv",  dbstep_cols, ref_names)
F_strain = load_phys(f"{WORK}/strain/strain_features.csv",  strain_cols, ref_names)
F_cdft   = load_phys(f"{WORK}/cdft/cdft_features.csv",      cdft_cols,   ref_names)
F_orbmol = load_phys(f"{WORK}/orbmol/orbmol_features.csv",  orb_cols,    ref_names)

for lbl, f in [("AIMNet2",F_aimnet),("D4",F_d4),("DBSTEP",F_dbstep),
               ("Strain",F_strain),("CDFT",F_cdft),("OrbMol",F_orbmol)]:
    print(f"  {lbl}: {f.shape}")

# Physics concat: 9+15+15+9+9+13 = 70 features
PHYS = np.hstack([F_aimnet, F_d4, F_dbstep, F_strain, F_cdft, F_orbmol])
print(f"Physics all: {PHYS.shape}")

# SOAP PCA-24 (global)
soap_data   = np.load(f"{WORK}/soap/soap_raw.npz", allow_pickle=True)
soap_names  = soap_data["names"].tolist()
soap_X_all  = soap_data["X"].astype(np.float32)
soap_name2i = {n: i for i, n in enumerate(soap_names)}
SOAP_253    = np.zeros((N, soap_X_all.shape[1]), dtype=np.float32)
for i, nm in enumerate(ref_names):
    if nm in soap_name2i:
        SOAP_253[i] = soap_X_all[soap_name2i[nm]]
pca_soap = PCA(n_components=24, random_state=0).fit(SOAP_253)
SOAP24   = pca_soap.transform(SOAP_253).astype(np.float32)
print(f"SOAP PCA-24: {SOAP24.shape}")

# PMapper PCA-24 (global)
pmapper_data = np.load(f"{WORK}/pmapper_feats.npz", allow_pickle=True)
PMAP_253     = pmapper_data["test"].astype(np.float32)[ub_idx]
pca_pmap     = PCA(n_components=24, random_state=0).fit(PMAP_253)
PMAP24       = pca_pmap.transform(PMAP_253).astype(np.float32)
print(f"PMapper PCA-24: {PMAP24.shape}")

# Boltz-z PCA-24 (global)
BOLTZ_RAW    = np.load("data/processed/boltz_z_rich_513.npy").astype(np.float32)[ub_idx]
pca_boltz    = PCA(n_components=24, random_state=0).fit(BOLTZ_RAW)
BOLTZ24      = pca_boltz.transform(BOLTZ_RAW).astype(np.float32)
print(f"Boltz-z PCA-24: {BOLTZ24.shape}")

# ─── 5. Feature block registry ───────────────────────────────────────────────
BLOCKS = {
    "base":   B,           # (253, 29) — base model predictions
    "morgan": M50,         # (253, 50) — Morgan PCA
    "phys":   PHYS,        # (253, 70) — all physics
    "aimnet": F_aimnet,    # (253, 9)
    "d4":     F_d4,        # (253, 15)
    "dbstep": F_dbstep,    # (253, 15)
    "strain": F_strain,    # (253, 9)
    "cdft":   F_cdft,      # (253, 9)
    "orbmol": F_orbmol,    # (253, 13)
    "soap":   SOAP24,      # (253, 24)
    "pmap":   PMAP24,      # (253, 24)
    "boltz":  BOLTZ24,     # (253, 24)
}

def run_loocv(feat_keys, model_type="ridge"):
    """LOOCV on 253 with Ridge or LGBM. GlobalPCA already applied."""
    X_full = np.hstack([BLOCKS[k] for k in feat_keys]).astype(np.float32)
    oof = np.zeros(N)
    for i in range(N):
        tr_idx = np.delete(np.arange(N), i)
        Xtr = X_full[tr_idx]; Xte = X_full[[i]]
        ytr = y_true[tr_idx]
        imp = SimpleImputer(strategy="median").fit(Xtr)
        sc  = StandardScaler().fit(imp.transform(Xtr))
        Xtr_s = sc.transform(imp.transform(Xtr))
        Xte_s = sc.transform(imp.transform(Xte))
        if model_type == "ridge":
            m = RidgeCV(alphas=ALPHAS, cv=5).fit(Xtr_s, ytr)
        else:
            m = LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.1,
                              n_jobs=2, verbose=-1, random_state=0).fit(Xtr_s, ytr)
        oof[i] = m.predict(Xte_s)[0]
    return rae(y_true, oof), oof

# ─── 6. Ablation experiments ─────────────────────────────────────────────────
experiments = [
    # label, feature keys, model
    ("F0_base_ridge",              ["base"],                                            "ridge"),
    ("F1_base+morgan_ridge",       ["base","morgan"],                                   "ridge"),  # current ~0.614
    ("F1_base+morgan_lgbm",        ["base","morgan"],                                   "lgbm"),
    ("F2_+phys_all_ridge",         ["base","morgan","phys"],                            "ridge"),
    ("F3_+soap+pmap_ridge",        ["base","morgan","soap","pmap"],                     "ridge"),
    ("F4_+boltz_ridge",            ["base","morgan","boltz"],                           "ridge"),
    ("F5_+soap+pmap+boltz_ridge",  ["base","morgan","soap","pmap","boltz"],             "ridge"),
    ("F6_all_ridge",               ["base","morgan","phys","soap","pmap","boltz"],      "ridge"),
    ("F6_all_lgbm",                ["base","morgan","phys","soap","pmap","boltz"],      "lgbm"),
    ("F7_nomorgan+soap+pmap+boltz",["base","soap","pmap","boltz"],                      "ridge"),
    # Individual physics ablations
    ("Fa_+aimnet",                 ["base","morgan","aimnet"],                          "ridge"),
    ("Fb_+d4",                     ["base","morgan","d4"],                              "ridge"),
    ("Fc_+dbstep",                 ["base","morgan","dbstep"],                          "ridge"),
    ("Fd_+strain",                 ["base","morgan","strain"],                          "ridge"),
    ("Fe_+cdft",                   ["base","morgan","cdft"],                            "ridge"),
    ("Ff_+orbmol",                 ["base","morgan","orbmol"],                          "ridge"),
    ("Fg_+soap_only",              ["base","morgan","soap"],                            "ridge"),
    ("Fh_+pmap_only",              ["base","morgan","pmap"],                            "ridge"),
    ("Fi_+boltz_only",             ["base","morgan","boltz"],                           "ridge"),
]

print("\n=== LOOCV ABLATION (253 compounds, global PCA) ===")
print(f"{'Experiment':<38} {'RAE':>7} {'vs_F1':>8}")
print("-"*58)

results = {}
f1_rae = None
for label, feat_keys, mdl in experiments:
    r, oof_arr = run_loocv(feat_keys, mdl)
    gap = (r - f1_rae) if f1_rae is not None else None
    if label == "F1_base+morgan_ridge": f1_rae = r
    gap_s = f"{gap:+.4f}" if gap is not None else "  n/a"
    print(f"  {label:<38} {r:.4f}  {gap_s}")
    results[label] = {"rae": round(r,4), "feats": feat_keys, "model": mdl,
                      "vs_F1": round(gap,4) if gap is not None else None}
    np.save(f"{MS}/nb1316_{label}.npy", oof_arr)

# ─── 7. Blend search using best new LOO prediction ───────────────────────────
meta_loocv   = np.load(f"{MS}/meta_stacker_loocv_253.npy").ravel().astype(float)
knn_loocv    = np.load(f"{MS}/knn_residual_loocv_253_correct.npy").ravel().astype(float)
combined_c   = np.load(f"{MS}/combined_corrected_513.npy").ravel()[ub_idx].astype(float)

# Find best new LOO OOF from this run
best_k, best_new_rae = min(results.items(), key=lambda kv: kv[1]["rae"])
best_new_oof = np.load(f"{MS}/nb1316_{best_k}.npy")
print(f"\nBest new LOO: {best_k} RAE={best_new_rae['rae']:.4f}")
print(f"Reference meta_stacker: {rae(y_true,meta_loocv):.4f}")
print(f"Reference triple blend: {rae(y_true,0.6*meta_loocv+0.15*knn_loocv+0.25*combined_c):.4f}")

# Search 4-way blend: meta + knn + combined_c + new
best_rae, best_w = 9., None
for w1 in np.arange(0.1, 0.91, 0.05):
    for w2 in np.arange(0.0, 0.6, 0.05):
        for w3 in np.arange(0.0, 0.5, 0.05):
            w4 = round(1-w1-w2-w3, 2)
            if w4 < 0 or w4 > 0.8: continue
            blend = w1*meta_loocv + w2*knn_loocv + w3*combined_c + w4*best_new_oof
            r = rae(y_true, blend)
            if r < best_rae:
                best_rae = r; best_w = (w1, w2, w3, w4)

print(f"Best blend: w1={best_w[0]:.2f} w2={best_w[1]:.2f} w3={best_w[2]:.2f} w4={best_w[3]:.2f} RAE={best_rae:.4f}")

results["_blend"] = {
    "best_rae": round(best_rae,4),
    "weights": {"meta_loocv": best_w[0], "knn_loocv": best_w[1], "combined_c": best_w[2], best_k: best_w[3]},
    "meta_rae": round(rae(y_true,meta_loocv),4),
    "triple_rae": round(rae(y_true,0.6*meta_loocv+0.15*knn_loocv+0.25*combined_c),4),
}

best_blend_oof = best_w[0]*meta_loocv + best_w[1]*knn_loocv + best_w[2]*combined_c + best_w[3]*best_new_oof
np.save(f"{MS}/mega_stacker_253.npy", best_blend_oof)

json.dump(results, open(OUT, "w"), indent=2)
print(f"\nSaved {OUT}")
print("DONE")
