"""nb1336 — AR (NR3C4) cross-NR READ-ACROSS feature, honest gate.

AR is a Type-I activation NR co-expressed + cross-talking with PXR in hepatocyte
xenobiotic/bile-acid handling; shares ligand space (bile-acid mimetics, GW4064-class).
Leakage-safe read-across: train an AR p-potency predictor on 3855 ChEMBL+Tox21 AR
compounds (NEVER sees PXR pEC50), predict AR-activity for our 4139 train + 513 test,
append that single prediction as ONE extra feature to the deployed 4-GBM combined
ensemble. Gate: matched scaffold-CV control vs treat, 3 seeds.
Deploy iff delta < -0.001 AND treat < best (0.4149).
"""
import os, sys, json, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train, load_test
from src.pxr.featurize import combined, impute
from src.pxr.eval import scaffold_kfold_indices
from src.pxr.chem import morgan_fp_batch
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from catboost import CatBoostRegressor
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import inchi
RDLogger.DisableLog("rdApp.*")

P = "data/processed"; OUT = "C:/pxr_work/ar_tox21"
BEST_RAE = 0.4149
N_SEEDS = 3

def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""
def rae_(yt, yp):
    d = np.abs(yt - np.median(yt)).sum()
    return float(np.abs(yt - yp).sum() / d) if d else float("nan")
def ikey(s):
    m = Chem.MolFromSmiles(str(s))
    try: return inchi.InchiToInchiKey(inchi.MolToInchi(m)) if m else None
    except Exception: return None

# ── AR data
ded = pickle.load(open(f"{OUT}/ar_std.pkl", "rb"))
ar_smi = [r[1] for r in ded]
ar_y   = np.array([r[4] for r in ded], float)
ar_ik  = set(r[2] for r in ded)
print(f"AR compounds: {len(ar_smi)}  p-range {ar_y.min():.2f}..{ar_y.max():.2f}")

# ── our data
tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
y  = tr["pec50"].to_numpy()
tr_smi = tr["smiles"].tolist()
scaf = np.array([murcko(s) for s in tr_smi])
te = load_test(); te_smi = te["smiles"].tolist()

# ── dedup / overlap report (InChIKey AR vs our train)
tr_ik = set(filter(None, (ikey(s) for s in tr_smi)))
overlap = len(ar_ik & tr_ik)
print(f"InChIKey overlap AR-cap-train: {overlap} (AR predictor never sees PXR labels -> no leakage)")

# ── coverage: AR actives (p>=6) -> our 513 test
act_smi = [r[1] for r in ded if r[4] >= 6.0]
if act_smi:
    fp_a = morgan_fp_batch(act_smi).astype(np.float32)
    fp_t = morgan_fp_batch(te_smi).astype(np.float32)
    inter = fp_a @ fp_t.T
    union = fp_a.sum(1)[:, None] + fp_t.sum(1)[None, :] - inter
    sim = inter / np.maximum(union, 1)
    maxsim = sim.max(0)
    med_tani = float(np.median(maxsim)); frac_cov = float((maxsim >= 0.4).mean())
else:
    med_tani, frac_cov = 0.0, 0.0
print(f"Coverage (test->nearest AR active): median Tanimoto={med_tani:.3f}  frac>=0.4={frac_cov:.3f}")

# ── read-across: AR predictor (combined feats), leakage-safe re: PXR labels
print("Featurizing AR + our train/test ...")
Xar = impute(combined(ar_smi)).astype(np.float32)
Xtr  = impute(combined(tr_smi)).astype(np.float32)
Xte  = impute(combined(te_smi)).astype(np.float32)
ar_model = LGBMRegressor(n_estimators=600, num_leaves=64, learning_rate=0.04, n_jobs=4, verbose=-1)
ar_model.fit(Xar, ar_y)
ra_tr = ar_model.predict(Xtr).astype(np.float32)
ra_te = ar_model.predict(Xte).astype(np.float32)
print(f"read-across feat: train range [{ra_tr.min():.2f},{ra_tr.max():.2f}]  "
      f"corr(read-across, PXR pEC50)={np.corrcoef(ra_tr, y)[0,1]:+.3f}")
np.save(f"{OUT}/ra_tr.npy", ra_tr); np.save(f"{OUT}/ra_te.npy", ra_te)

Xtr_aug = np.hstack([Xtr, ra_tr[:, None]])

MODELS = lambda: [
    LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1),
    XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, n_jobs=4, verbosity=0, tree_method="hist"),
    HistGradientBoostingRegressor(max_iter=500, max_leaf_nodes=64, learning_rate=0.05),
    CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05, verbose=0),
]
def oof(X, y, scaf, seed):
    folds = scaffold_kfold_indices(scaf.tolist(), n_splits=5, seed=seed)
    o = np.zeros(len(y))
    for trn, val in folds:
        preds = []
        for m in MODELS():
            m.fit(X[trn], y[trn]); preds.append(m.predict(X[val]))
        o[val] = np.mean(preds, axis=0)
    return o

c_raes, t_raes = [], []
for seed in range(N_SEEDS):
    ctrl  = oof(Xtr, y, scaf, seed)
    treat = oof(Xtr_aug, y, scaf, seed)
    rc, rt = rae_(y, ctrl), rae_(y, treat)
    c_raes.append(rc); t_raes.append(rt)
    print(f"seed {seed}: ctrl={rc:.4f} treat={rt:.4f} d={rt-rc:+.4f}")
    if seed == 0:
        np.save(f"{OUT}/oof_treat.npy", treat); np.save(f"{OUT}/oof_ctrl.npy", ctrl)
        np.save(f"{OUT}/y.npy", y)

ctrl_mean = float(np.mean(c_raes)); treat_mean = float(np.mean(t_raes))
delta = treat_mean - ctrl_mean
n_neg = int(np.sum(np.array(t_raes) < np.array(c_raes)))
corr_err = float(np.corrcoef(ra_tr, np.abs(y - np.load(f"{OUT}/oof_ctrl.npy")))[0,1])
deploy = delta < -0.001 and treat_mean < BEST_RAE

res = {
    "feature": "ar_nr3c4_readacross",
    "n_ar": len(ar_smi), "ik_overlap_train": overlap,
    "med_tani_to_test": round(med_tani,3), "frac_covered": round(frac_cov,3),
    "corr_ra_vs_pec50": round(float(np.corrcoef(ra_tr,y)[0,1]),3),
    "corr_ra_vs_ctrl_error": round(corr_err,3),
    "control_rae": round(ctrl_mean,4), "treatment_rae": round(treat_mean,4),
    "delta": round(delta,4), "neg_seeds": n_neg, "best_rae": BEST_RAE, "deploy": deploy,
}
json.dump(res, open(f"{P}/nb1338_ar_readacross_gate.json","w"), indent=2)
print("\n=== AR READ-ACROSS GATE ===")
print(json.dumps(res, indent=2))
