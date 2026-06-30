"""nb1303 — BindingDB IC50 data as additional chemprop aux-head (incremental over nb1164 EC50).

nb1164 already added 959 PXR EC50 rows as an aux head (→ WIN -0.003).
The full bdb_pxr_curated.csv has 1235 rows including IC50 data not yet in the model.
Strategy: Re-train chemprop with BOTH the original 959 EC50 rows AND the ~276 additional
IC50-only rows from BindingDB (converted to p-scale Ki/IC50 binding ≠ EC50 activation,
so we set a lower task weight for IC50 head). Also include coverage test.

If this HURTS (IC50 binding ≠ activation), we close the IC50 head axis.
Gate: matched clean holdouts, 3 seeds. Threshold: delta < -0.001 vs 0.4231.

NOTE: This probes whether BINDING data (IC50/Ki) adds over ACTIVATION data (EC50) already
incorporated. Expected: null or marginal (different endpoint).
"""
import os, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train, load_test
from src.pxr.featurize import combined, impute
from src.pxr.eval import rae, scaffold_kfold_indices
from src.pxr.chem import morgan_fp_batch, standardize
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from catboost import CatBoostRegressor
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

SD   = "C:/pxr_work/search"
BDB  = "C:/pxr_work/bindingdb/bdb_pxr_curated.csv"
BEST_RAE = 0.4231; N_SEEDS = 3
P = "data/processed"

def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""

def rae_(yt, yp):
    d = np.abs(yt - np.median(yt)).sum()
    return float(np.abs(yt - yp).sum() / d) if d else float("nan")

# ── Load and filter BindingDB PXR data
bdb = pd.read_csv(BDB)
print(f"BindingDB PXR: {len(bdb)} rows, cols={list(bdb.columns)}")
print(bdb["types"].value_counts().head(10))
print(f"in_train: {bdb['in_train'].sum()} rows already in training set")

# Drop rows in training set
bdb_novel = bdb[~bdb["in_train"]].copy()
print(f"Novel (not in train): {len(bdb_novel)} rows")

# Coverage test: similarity to 513 test set
te = load_test()
te_smi = te["smiles"].tolist()
bdb_smi = bdb_novel.dropna(subset=["std"])
# std column has Mol objects — use 'ik' (InChIKey) to get SMILES via test/train lookup
# Actually use the smiles from the std column if it's valid SMILES (may be Mol repr)
# Fall back to using the raw data: try to use bdb's original smiles
# Check which column has proper SMILES
for col in ["smiles", "std", "canonical_smiles"]:
    if col in bdb.columns:
        sample = str(bdb_novel[col].iloc[0])
        if not sample.startswith("<rdkit") and len(sample) > 3:
            smi_col = col; break
        else:
            smi_col = None

if smi_col is None:
    # Extract SMILES from the raw BindingDB JSON (has 'smile' field per affinity entry)
    import json as _json
    bdb_json = _json.load(open("C:/pxr_work/bindingdb/bdb_O75469.json"))
    aff = bdb_json["getLindsByUniprotsResponse"]["affinities"]
    # Build monomerid -> canonical SMILES map
    from rdkit import Chem as _Chem
    mono2smi = {}
    for entry in aff:
        mid = str(entry.get("monomerid", ""))
        smi = entry.get("smile", "")
        if mid and smi and mid not in mono2smi:
            m = _Chem.MolFromSmiles(smi)
            if m: mono2smi[mid] = _Chem.MolToSmiles(m)
    # bdb_pxr_curated may have monomerid or ik; try ik-based lookup via train/test
    tr_all = load_train()
    # build ik->smiles from train (smiles column is clean) + test
    ik_smi = {}
    from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
    for _, r in tr_all.dropna(subset=["smiles"]).iterrows():
        smi = str(r["smiles"])
        m = _Chem.MolFromSmiles(smi)
        if m:
            try:
                inchi = MolToInchi(m)
                if inchi:
                    ik = InchiToInchiKey(inchi)
                    if ik: ik_smi[ik] = _Chem.MolToSmiles(m)
            except Exception: pass
    for _, r in te.dropna(subset=["smiles"]).iterrows():
        smi = str(r["smiles"])
        m = _Chem.MolFromSmiles(smi)
        if m:
            try:
                inchi = MolToInchi(m)
                if inchi:
                    ik = InchiToInchiKey(inchi)
                    if ik: ik_smi[ik] = _Chem.MolToSmiles(m)
            except Exception: pass
    bdb_novel["smi_resolved"] = bdb_novel["ik"].map(ik_smi)
    # For remaining: try extracting from JSON smile field via monomerid if available
    if "monomerid" in bdb_novel.columns:
        mask = bdb_novel["smi_resolved"].isna()
        bdb_novel.loc[mask, "smi_resolved"] = bdb_novel.loc[mask, "monomerid"].astype(str).map(mono2smi)
    smi_col = "smi_resolved"
    n_resolved = bdb_novel[smi_col].notna().sum()
    print(f"Resolved SMILES: {n_resolved} / {len(bdb_novel)} (via InChIKey+JSON)")

bdb_novel = bdb_novel.dropna(subset=[smi_col, "p"]).copy()
bdb_novel["smiles_use"] = bdb_novel[smi_col].astype(str)
bdb_novel = bdb_novel[~bdb_novel["smiles_use"].str.startswith("<rdkit")]
print(f"After SMILES filter: {len(bdb_novel)} rows")

if len(bdb_novel) == 0:
    print("ERROR: no valid SMILES found in BindingDB novel set. Closing axis.")
    json.dump({"verdict": "SMILES_FAIL", "n_novel": 0},
              open(f"{P}/nb1303_ext_ic50_gate.json", "w"), indent=2)
    sys.exit(0)

# Coverage test
fp_bdb = morgan_fp_batch(bdb_novel["smiles_use"].tolist()).astype(np.float32)
fp_te  = morgan_fp_batch(te_smi[:300]).astype(np.float32)
inter  = fp_bdb @ fp_te.T
union  = fp_bdb.sum(1)[:, None] + fp_te.sum(1)[None, :] - inter
sim    = inter / np.maximum(union, 1)
max_sim = sim.max(1)
med_tani = float(np.median(max_sim))
frac_covered = float((max_sim >= 0.4).mean())
print(f"Coverage: median Tanimoto to test = {med_tani:.3f}, {frac_covered*100:.1f}% >= 0.4")

if med_tani < 0.25 and frac_covered < 0.05:
    print("COVERAGE FAIL — BindingDB IC50 too far from test set. Closing IC50-binding axis.")
    json.dump({"verdict": "coverage_NEG", "med_tani": round(med_tani,3),
               "frac_covered": round(frac_covered,3)},
              open(f"{P}/nb1303_ext_ic50_gate.json","w"), indent=2)
    sys.exit(0)

# ── Build augmented feature matrix using IC50/Ki as aux training signal
# Strategy: treat IC50 rows as a weak secondary signal (down-weighted in LGBM)
# This is a GBM-only approach (not full chemprop retrain); a proper aux head
# would retrain the chemprop GNN with these as a 2nd head — flag for swarm
tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
y  = tr["pec50"].to_numpy()
tr_smi = tr["smiles"].tolist()
scaf   = np.array([murcko(s) for s in tr_smi])

Xtr = impute(combined(tr_smi)).astype(np.float32)

# Add IC50 as pseudo-labels: augment training with bdb_novel
# Down-weight (0.3) since binding ≠ activation endpoint
bdb_smi_list = bdb_novel["smiles_use"].tolist()
Xaug_raw = combined(bdb_smi_list)
Xaug     = impute(Xaug_raw).astype(np.float32)
y_aug    = bdb_novel["p"].to_numpy(float)
w_aug    = np.full(len(y_aug), 0.3)

Xfull = np.vstack([Xtr, Xaug])
yfull = np.concatenate([y, y_aug])
wfull = np.concatenate([np.ones(len(y)), w_aug])

print(f"\nAugmented train: {len(yfull)} rows ({len(y)} real + {len(y_aug)} IC50 pseudo)")

def ensemble_oof_aug(X_tr, y_tr, X_aug, y_aug, w_aug, scaf, seed):
    """Scaffold CV on original train only; each fold fits on augmented data."""
    folds = scaffold_kfold_indices(scaf.tolist(), n_splits=5, seed=seed)
    oof = np.zeros(len(y_tr))
    for trn, val in folds:
        Xfit = np.vstack([X_tr[trn], X_aug])
        yfit = np.concatenate([y_tr[trn], y_aug])
        wfit = np.concatenate([np.ones(len(trn)), w_aug])
        preds = []
        for mdl in [
            LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1),
            XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, n_jobs=4, verbosity=0,
                         tree_method="hist"),
            HistGradientBoostingRegressor(max_iter=500, max_leaf_nodes=64, learning_rate=0.05),
            CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05, verbose=0),
        ]:
            mdl.fit(Xfit, yfit, **({} if isinstance(mdl, HistGradientBoostingRegressor) else
                                   {"sample_weight": wfit}))
            preds.append(mdl.predict(X_tr[val]))
        oof[val] = np.mean(preds, axis=0)
    return oof

def ensemble_oof_base(X, y, scaf, seed):
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
    ctrl  = ensemble_oof_base(Xtr, y, scaf, seed)
    treat = ensemble_oof_aug(Xtr, y, Xaug, y_aug, w_aug, scaf, seed)
    rc = rae_(y, ctrl); rt = rae_(y, treat)
    c_raes.append(rc); t_raes.append(rt)
    print(f"ctrl={rc:.4f}  treat={rt:.4f}  d={rt-rc:+.4f}")

ctrl_mean  = float(np.mean(c_raes)); treat_mean = float(np.mean(t_raes))
delta      = treat_mean - ctrl_mean
n_neg      = int(np.sum(np.array(t_raes) < np.array(c_raes)))
deploy     = delta < -0.001 and treat_mean < BEST_RAE

print(f"\n=== EXT IC50 GATE ===")
print(f"control  RAE = {ctrl_mean:.4f}")
print(f"treat    RAE = {treat_mean:.4f}  delta={delta:+.4f}  ({n_neg}/{N_SEEDS} seeds neg)")
print(f"DEPLOY = {deploy}")

result = {
    "feature": "bindingdb_ic50_augmentation",
    "n_ic50_novel": len(bdb_novel), "med_tani_to_test": round(med_tani,3),
    "frac_covered": round(frac_covered,3),
    "control_rae": round(ctrl_mean,4), "treatment_rae": round(treat_mean,4),
    "delta": round(delta,4), "neg_seeds": n_neg, "deploy": deploy,
    "note": "GBM-only aug (not chemprop aux retrain); flag for chemprop aux if helpful",
}
json.dump(result, open(f"{P}/nb1303_ext_ic50_gate.json","w"), indent=2)
print(json.dumps(result, indent=2))
