"""nb1302 — Emax efficacy as input feature (honest gate).

Emax = maximal activation level in the CRC assay. If Emax correlates with PXR
activation potency or partial-agonist behaviour, adding it as a training feature
might reduce error on the high-Emax compounds. Expected: null (corr(|err|, Emax) ~0).
For test compounds (no Emax measured), impute with median of training Emax.

Gate: matched never-tuned holdouts, 3 seeds. Deploy iff delta < -0.001 vs 0.4231.
"""
import os, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train, load_test
from src.pxr.featurize import combined, impute
from src.pxr.eval import rae, scaffold_kfold_indices
from src.pxr.chem import morgan_fp_batch
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from catboost import CatBoostRegressor
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

SD = "C:/pxr_work/search"; P = "data/processed"
BEST_RAE = 0.4231
N_SEEDS  = 3

def murcko(s):
    m = Chem.MolFromSmiles(str(s))
    return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""

def rae_(yt, yp):
    d = np.abs(yt - np.median(yt)).sum()
    return float(np.abs(yt - yp).sum() / d) if d else float("nan")

# ── Load training data
tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
y  = tr["pec50"].to_numpy()
smis = tr["smiles"].tolist()
scaf = np.array([murcko(s) for s in smis])

print(f"Train: {len(tr)} compounds")
print(f"Emax columns: {[c for c in tr.columns if 'emax' in c.lower() or 'efficacy' in c.lower()]}")

# Find the Emax column
emax_col = next((c for c in tr.columns if c.lower() in ("emax", "emax_rel", "efficacy", "e_max")), None)
if emax_col is None:
    # Try a fuzzy match
    for c in tr.columns:
        if "emax" in c.lower():
            emax_col = c; break
if emax_col is None:
    print("ERROR: no Emax column found. Columns:", list(tr.columns))
    sys.exit(1)

emax_raw = tr[emax_col].to_numpy(float)
n_with   = int((~np.isnan(emax_raw)).sum())
print(f"Emax column '{emax_col}': {n_with}/{len(tr)} non-null, "
      f"range [{np.nanmin(emax_raw):.2f}, {np.nanmax(emax_raw):.2f}]")

emax_med = float(np.nanmedian(emax_raw))
emax_imp = np.where(np.isnan(emax_raw), emax_med, emax_raw).reshape(-1, 1)

# Correlation of |error| with Emax (diagnostic)
oof_path = f"{SD}/gnn_oof.npy"
if os.path.exists(oof_path):
    gnn_oof = np.load(oof_path)
    err = np.abs(y - gnn_oof)
    valid = ~np.isnan(emax_raw)
    corr_emax_err = float(np.corrcoef(emax_raw[valid], err[valid])[0, 1])
    print(f"corr(|GNN error|, Emax) = {corr_emax_err:+.4f}  (expected ~0)")

# ── Feature matrices
print("Building combined features...")
Xbase = impute(combined(smis)).astype(np.float32)      # (4139, 2265)
Xemax = np.hstack([Xbase, emax_imp]).astype(np.float32) # (4139, 2266)

# Load physics features
def load_phys():
    import warnings
    feats = []
    for path, cols, src_filter in [
        ("C:/pxr_work/aimnet2/aimnet_features.csv",
         ["aimnet_energy","aimnet_qmin","aimnet_qmax","aimnet_qabs_mean","aimnet_qstd",
          "aimnet_qsum_abs","aimnet_dipole","aimnet_fmax","aimnet_frms"], "all"),
        ("C:/pxr_work/strain/strain_features.csv",
         ["strain_relax_mean","strain_relax_max","conf_espread","conf_erange",
          "conf_n","rmsd_mean","rmsd_max","e_per_heavy"], "all"),
        ("C:/pxr_work/d4/d4_features.csv",
         ["d4_alpha_sum","d4_alpha_mean","d4_alpha_std","d4_alpha_max",
          "d4_c6diag_mean","d4_c6diag_std","d4_c6_total","d4_edisp",
          "d4_edisp_per_atom","d4_cn_mean","d4_cn_max","d4_qeeq_min",
          "d4_qeeq_max","d4_qeeq_std","d4_qeeq_absum"], "all"),
        ("C:/pxr_work/dbstep/dbstep_features.csv",
         ["vbur_r25","vbur_r35","vbur_r45","vbur_r55","vbur_r65",
          "ster_L","ster_Bmin","ster_Bmax","ster_aniso",
          "npr1","npr2","asphericity","spherocity","eccentricity",
          "radgyr","inertial_sf"], "all"),
        ("C:/pxr_work/orbmol/orbmol_features.csv",
         ["orb_energy","orb_energy_per_ha","orb_fmax","orb_frms","orb_fstd",
          "orb_conf_mean","orb_conf_std","orb_conf_node_mean","orb_conf_node_std",
          "orb_conf_node_min","orb_node_emb_mean","orb_node_emb_std",
          "orb_node_emb_norm"], "all"),
    ]:
        if not os.path.exists(path):
            print(f"  Missing: {path}"); continue
        df = pd.read_csv(path)
        if src_filter != "all" and "src" in df.columns:
            df = df[df.src == src_filter]
        df = df.drop_duplicates(subset="name", keep="first")
        names = tr["name"].tolist() if "name" in tr.columns else [f"OADMET-{i}" for i in tr.index]
        sub = df.set_index("name").reindex(names)[cols]
        X = sub.apply(pd.to_numeric, errors="coerce").to_numpy(float)
        med = np.nanmedian(X, axis=0); idx = np.where(np.isnan(X))
        X[idx] = np.take(med, idx[1])
        feats.append(X)
        print(f"  {os.path.basename(path)}: {X.shape[1]} cols, {int(np.isnan(X).any(1).sum())} rows with NaN post-impute")
    return np.hstack(feats).astype(np.float32) if feats else np.zeros((len(tr), 0), np.float32)

Xphys = load_phys()
print(f"Physics feats: {Xphys.shape}")

Xfull_base = np.hstack([Xbase, Xphys])
Xfull_emax = np.hstack([Xbase, Xphys, emax_imp])
print(f"Full feat shape: base={Xfull_base.shape}, +emax={Xfull_emax.shape}")

# ── Honest gate: matched clean holdouts, 3 seeds
def gbm_ensemble_oof(X, y, scaf, seed):
    folds = scaffold_kfold_indices(scaf.tolist(), n_splits=5, seed=seed)
    oof = np.zeros(len(y))
    for trn, val in folds:
        preds = []
        for mdl in [
            LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1),
            XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, n_jobs=4, verbosity=0,
                         tree_method="hist"),
            HistGradientBoostingRegressor(max_iter=500, max_leaf_nodes=64, learning_rate=0.05),
            CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05, verbose=0),
        ]:
            mdl.fit(X[trn], y[trn]); preds.append(mdl.predict(X[val]))
        oof[val] = np.mean(preds, axis=0)
    return oof

c_raes, t_raes = [], []
for seed in range(N_SEEDS):
    print(f"seed {seed}:", end=" ", flush=True)
    ctrl = gbm_ensemble_oof(Xfull_base, y, scaf, seed=seed)
    treat = gbm_ensemble_oof(Xfull_emax, y, scaf, seed=seed)
    rc = rae_(y, ctrl); rt = rae_(y, treat)
    c_raes.append(rc); t_raes.append(rt)
    print(f"ctrl={rc:.4f}  treat={rt:.4f}  d={rt-rc:+.4f}")

ctrl_mean  = float(np.mean(c_raes))
treat_mean = float(np.mean(t_raes))
delta      = treat_mean - ctrl_mean
n_neg      = int(np.sum(np.array(t_raes) < np.array(c_raes)))
deploy     = delta < -0.001 and treat_mean < BEST_RAE

print(f"\n=== EMAX PROBE SUMMARY ===")
print(f"control  RAE = {ctrl_mean:.4f}")
print(f"treat    RAE = {treat_mean:.4f}  delta={delta:+.4f}  ({n_neg}/{N_SEEDS} seeds neg)")
print(f"deployed best = {BEST_RAE}")
print(f"DEPLOY = {deploy}")

result = {
    "feature": "emax_as_input",
    "emax_column": emax_col,
    "n_with_emax": n_with,
    "corr_emax_err": round(corr_emax_err, 4) if "corr_emax_err" in dir() else None,
    "control_rae": round(ctrl_mean, 4),
    "treatment_rae": round(treat_mean, 4),
    "delta": round(delta, 4),
    "neg_seeds": n_neg,
    "deploy": deploy,
    "deployed_best": BEST_RAE,
}
json.dump(result, open("data/processed/nb1302_emax_probe.json", "w"), indent=2)
print(json.dumps(result, indent=2))
