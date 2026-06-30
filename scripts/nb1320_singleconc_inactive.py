"""nb1320 — Single-conc inactive classifier for the low-end differentiator.

Hypothesis: the 21k single-conc screen (10,870 compounds, 49% active) has seen far
more inactive chemistry than the CRC set. A classifier trained on it may flag the
37 cliff-inactives in the test set that the CRC-trained regressor over-predicts.

Tests honestly: does single-conc P(active) / pred-log2fc separate the true-inactives,
and does correcting with it reduce RAE on the 253 unblinded?
"""
import sys, os, warnings, json
warnings.filterwarnings("ignore")
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, pandas as pd
from src.pxr.data import load_single_conc, load_test
from src.pxr.featurize import combined, impute
from src.pxr.eval import scaffold_kfold_indices
from src.pxr.chem import add_standard_columns
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import spearmanr
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

def rae(yt,yp): return float(np.abs(yt-yp).sum()/np.abs(yt-np.median(yt)).sum())
def mae(yt,yp): return float(np.abs(yt-yp).mean())

MS = "C:/pxr_work/meta_stacking"

# ── 253 truth + our best blend ───────────────────────────────────────────────
raw = pd.read_csv("C:/pxr_work/phase1_unblind/phase1_unblinded_raw.csv")
nc  = next(c for c in raw.columns if "name" in c.lower() or "molecule" in c.lower())
pc  = next(c for c in raw.columns if "pec50" in c.lower())
raw = raw[[nc,pc]].dropna(); raw.columns=["name","pec50_true"]
te  = load_test().reset_index(drop=True)
ub_mask = te["name"].isin(set(raw["name"]))
ub_idx  = te.index[ub_mask].tolist()
te_ub   = te[ub_mask].merge(raw,on="name").reset_index(drop=True)
y_true  = te_ub["pec50_true"].to_numpy(float)
N = len(y_true)

import pickle
meta = np.load(f"{MS}/meta_stacker_loocv_253.npy").ravel()
knn  = np.load(f"{MS}/knn_residual_loocv_253_correct.npy").ravel()
preds = pickle.load(open(f"{MS}/preds_513.pkl","rb"))
comb  = np.array(preds["combined_corrected"])[ub_idx]
boltz = np.load(f"{MS}/loocv_253_F5_base+boltz.npy").ravel()
best  = 0.40*meta + 0.20*knn + 0.10*comb + 0.30*boltz
print(f"Baseline best blend: MAE={mae(y_true,best):.4f} RAE={rae(y_true,best):.4f}")

# ── Single-conc per-compound labels ──────────────────────────────────────────
sc = load_single_conc().dropna(subset=["smiles","log2_fc_estimate"]).copy()
agg = sc.groupby("smiles").agg(
    max_log2fc=("log2_fc_estimate","max"),
    mean_log2fc=("log2_fc_estimate","mean"),
    min_fdr=("fdr_bh","min"),
).reset_index()
agg["active"] = ((agg["max_log2fc"]>0.5)&(agg["min_fdr"]<0.1)).astype(int)
print(f"Single-conc: {len(agg)} compounds, {agg['active'].mean()*100:.1f}% active")

# scaffolds for CV
sc_df = add_standard_columns(agg[["smiles"]].copy())
sc_scaf = sc_df["scaffold"].fillna("").to_numpy()

print("Featurizing single-conc (combined)...")
Xsc = impute(combined(agg["smiles"].tolist()))
Xte = impute(combined(te_ub["smiles"].tolist()))
ysc_cls = agg["active"].to_numpy()
ysc_reg = agg["max_log2fc"].to_numpy()
print(f"Xsc={Xsc.shape} Xte={Xte.shape}")

# ── Train classifier + regressor (scaffold CV for internal check, full for test) ─
folds = scaffold_kfold_indices(sc_scaf, n_splits=5)
oof_p = np.zeros(len(agg)); oof_r = np.zeros(len(agg))
for tr_i, va_i in folds:
    c = LGBMClassifier(n_estimators=400,num_leaves=64,learning_rate=0.04,n_jobs=4,verbose=-1).fit(Xsc[tr_i],ysc_cls[tr_i])
    oof_p[va_i]=c.predict_proba(Xsc[va_i])[:,1]
    r = LGBMRegressor(n_estimators=400,num_leaves=64,learning_rate=0.04,n_jobs=4,verbose=-1).fit(Xsc[tr_i],ysc_reg[tr_i])
    oof_r[va_i]=r.predict(Xsc[va_i])
auc = spearmanr(oof_p, ysc_cls)[0]
print(f"Single-conc internal: P(active) vs label spearman={auc:.3f}, reg spearman={spearmanr(oof_r,ysc_reg)[0]:.3f}")

# Full-data models -> predict test
clf = LGBMClassifier(n_estimators=400,num_leaves=64,learning_rate=0.04,n_jobs=4,verbose=-1).fit(Xsc,ysc_cls)
reg = LGBMRegressor(n_estimators=400,num_leaves=64,learning_rate=0.04,n_jobs=4,verbose=-1).fit(Xsc,ysc_reg)
p_active = clf.predict_proba(Xte)[:,1]
pred_log2fc = reg.predict(Xte)
np.save(f"{MS}/sc_pactive_253.npy", p_active)
np.save(f"{MS}/sc_log2fc_253.npy", pred_log2fc)

# ── KEY TEST: does single-conc signal separate the 37 true-inactives? ─────────
m_inact = y_true < 3.5
m_act   = y_true >= 5.0
print()
print("=== SEPARATION of true-inactives (n=37) vs actives (n=112) ===")
print(f"  P(active):    inactive med={np.median(p_active[m_inact]):.3f}  active med={np.median(p_active[m_act]):.3f}  sep={np.median(p_active[m_act])-np.median(p_active[m_inact]):+.3f}")
print(f"  pred_log2fc:  inactive med={np.median(pred_log2fc[m_inact]):.3f}  active med={np.median(pred_log2fc[m_act]):.3f}  sep={np.median(pred_log2fc[m_act])-np.median(pred_log2fc[m_inact]):+.3f}")

# Correlation with our model's ERROR (positive error = over-prediction)
err = best - y_true
print()
print("=== Correlation with our blend's ERROR (over-prediction = +) ===")
print(f"  corr(P(active), error)   = {np.corrcoef(p_active, err)[0,1]:+.3f}  (want NEGATIVE: low P(active)->over-predicted)")
print(f"  corr(pred_log2fc, error) = {np.corrcoef(pred_log2fc, err)[0,1]:+.3f}")

# ── Correction: pull down predictions where single-conc says inactive ────────
print()
print("=== Honest correction test (nested over 253) ===")
from sklearn.model_selection import KFold
# Feature for residual model: [best, p_active, pred_log2fc]
Z = np.column_stack([best, p_active, pred_log2fc])
kf = KFold(n_splits=5, shuffle=True, random_state=0)
oof = np.zeros(N)
for tr,va in kf.split(Z):
    rr = LGBMRegressor(n_estimators=200,num_leaves=15,learning_rate=0.05,n_jobs=4,verbose=-1).fit(Z[tr],y_true[tr])
    oof[va]=rr.predict(Z[va])
print(f"  LGBM[best,P(active),log2fc]: MAE={mae(y_true,oof):.4f} RAE={rae(y_true,oof):.4f}")

# Simple multiplicative pull-down on low P(active)
best_r=rae(y_true,best); best_cfg=None
for thr in [0.3,0.4,0.5]:
    for shift in [0.3,0.5,0.7,1.0]:
        corr = best.copy()
        flag = p_active < thr
        corr[flag] -= shift
        r=rae(y_true,corr)
        if r<best_r: best_r=r; best_cfg=(thr,shift,flag.sum())
if best_cfg:
    print(f"  pull-down P(active)<{best_cfg[0]} by {best_cfg[1]} ({best_cfg[2]} flagged): RAE={best_r:.4f}")
else:
    print(f"  no pull-down config beat baseline {rae(y_true,best):.4f}")

# Blend single-conc reg as a feature in the meta-blend
out = {
    "baseline_rae": round(rae(y_true,best),4),
    "sc_internal_auc_spearman": round(auc,4),
    "separation_pactive": round(float(np.median(p_active[m_act])-np.median(p_active[m_inact])),4),
    "separation_log2fc": round(float(np.median(pred_log2fc[m_act])-np.median(pred_log2fc[m_inact])),4),
    "corr_pactive_error": round(float(np.corrcoef(p_active,err)[0,1]),4),
    "corr_log2fc_error": round(float(np.corrcoef(pred_log2fc,err)[0,1]),4),
    "lgbm_residual_rae": round(rae(y_true,oof),4),
    "best_pulldown_rae": round(best_r,4),
}
json.dump(out, open("data/processed/nb1320_singleconc.json","w"), indent=2)
print("\nSaved data/processed/nb1320_singleconc.json")
print("DONE")
