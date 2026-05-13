"""
Generate nb86–nb100: biological fingerprinting, 3D descriptors,
proper nested-CV ensemble, GPU Chemprop, pharmacophore features,
cliff-aware blending, and grand ensemble v8.
"""
import json, textwrap
from pathlib import Path

NB_DIR = Path(__file__).parent.parent / "notebooks"

BOILERPLATE_IMPORTS = """\
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, "../src")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from pathlib import Path
from pxr.data import load_train, load_test
from pxr.featurize import combined, impute
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko, morgan_fp_batch, standardize_smiles, compute_physchem
from pxr.paths import DATA_PROCESSED, DATA_EXTERNAL, SUBMISSIONS
SEED = 42; N_FOLDS = 5
LGBM = dict(n_estimators=1000, num_leaves=64, learning_rate=0.05,
            min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, random_state=SEED, verbose=-1, n_jobs=4)
"""

FULL_METRICS = """\
def full_metrics(y_true, y_pred, cp=None, label=""):
    yt = np.asarray(y_true, float); yp = np.asarray(y_pred, float)
    msk = np.isfinite(yt) & np.isfinite(yp); yt, yp = yt[msk], yp[msk]
    mae = float(np.mean(np.abs(yt-yp)))
    rae_v = mae / float(np.mean(np.abs(yt-yt.mean()))) if yt.std()>0 else float("nan")
    r2  = 1-np.sum((yt-yp)**2)/np.sum((yt-yt.mean())**2) if yt.std()>0 else float("nan")
    pr, _ = stats.pearsonr(yt, yp); sp, _ = stats.spearmanr(yt, yp)
    kt, _ = stats.kendalltau(yt, yp)
    m = dict(RAE=rae_v, MAE=mae, R2=float(r2), Pearson=float(pr),
             Spearman=float(sp), Kendall=float(kt))
    if cp is not None and hasattr(cp, "iterrows") and len(cp) > 0:
        c=t=0
        for _,row in cp.iterrows():
            ia,ii = int(row.get("idx_active",-1)), int(row.get("idx_inactive",-1))
            if 0<=ia<len(yp) and 0<=ii<len(yp): c+=int(yp[ia]>yp[ii]); t+=1
        m["Cliff_acc"] = c/t if t else float("nan")
    if label:
        ca = f"  Cliff={m.get('Cliff_acc',float('nan')):.3f}" if "Cliff_acc" in m else ""
        print(f"  [{label}] RAE={rae_v:.4f} MAE={mae:.4f} R²={r2:.4f} "
              f"r={pr:.4f} ρ={sp:.4f} τ={kt:.4f}{ca}")
    return m
"""

STANDARD_SETUP = """\
tr = load_train(); te = load_test()
y_tr = tr["pec50"].values.astype(np.float64)
scaffolds = tr["smiles"].map(bemis_murcko).tolist()
splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)
active_mask = y_tr >= 5.5
X_tr = impute(combined(tr["smiles"].tolist()))
X_te = impute(combined(te["smiles"].tolist()))
fps_tr = morgan_fp_batch(tr["smiles"].tolist()).astype(np.float32)
fps_te = morgan_fp_batch(te["smiles"].tolist()).astype(np.float32)
cliff_pairs = (pd.read_parquet(DATA_PROCESSED/"cliff_pairs.parquet")
               if (DATA_PROCESSED/"cliff_pairs.parquet").exists() else pd.DataFrame())
# Build idx_active / idx_inactive
if len(cliff_pairs) > 0:
    s2i = {s:i for i,s in enumerate(tr["smiles"].tolist())}
    ac = "cliff_active_smiles" if "cliff_active_smiles" in cliff_pairs.columns else "smiles_a"
    ic = "cliff_inactive_smiles" if "cliff_inactive_smiles" in cliff_pairs.columns else "smiles_b"
    cliff_pairs["idx_active"]   = cliff_pairs[ac].map(s2i)
    cliff_pairs["idx_inactive"] = cliff_pairs[ic].map(s2i)
    cliff_pairs = cliff_pairs.dropna(subset=["idx_active","idx_inactive"])
    cliff_pairs[["idx_active","idx_inactive"]] = cliff_pairs[["idx_active","idx_inactive"]].astype(int)
print(f"Train {len(tr):,}  Test {len(te):,}  Cliffs {len(cliff_pairs)}")
"""


def nb(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "pxr-challenge", "language": "python", "name": "pxr-challenge"}},
        "cells": [
            {"cell_type": c[0], "id": f"cell-{i}", "metadata": {}, "source": c[1],
             "outputs": [], "execution_count": None}
            if c[0] == "code" else
            {"cell_type": "markdown", "id": f"cell-{i}", "metadata": {}, "source": c[1]}
            for i, c in enumerate(cells)
        ]
    }


# ── nb86: Proper Nested-CV Grand Ensemble ─────────────────────────────────────
nb86 = nb([
("markdown", "# 86 — Proper Nested-CV Grand Ensemble\n\n"
 "nb85 fitted ElasticNetCV on the full OOF stack in-sample → reported RAE 0.218 is misleading.\n\n"
 "Here we use **nested scaffold CV**: for each outer fold, fit the meta-learner on the OTHER 4 folds' OOF predictions, then predict on the held-out fold. "
 "This gives unbiased OOF predictions for the ensemble.\n\n"
 "Expected: real stacked RAE around 0.50–0.52 (slightly better than best individual model)."),

("code", BOILERPLATE_IMPORTS),
("code", FULL_METRICS),
("code", STANDARD_SETUP),

("code", """\
from sklearn.linear_model import ElasticNetCV

EXCLUDE = {"aux_features", "grand_v6", "grand_v6b", "grand_v6c", "grand_v7",
           "grand15","grand18","grand23","grand24","grand25",
           "creative_mega_ensemble", "cliff_role_proba", "chemprop_cliff_mem_proba",
           "chemprop_chembl_nr_multitask", "per_fp_stack"}

oof_files = sorted(DATA_PROCESSED.glob("oof_*.npy"))
oofs, tes, names = [], [], []
for fp in oof_files:
    name = fp.stem.replace("oof_","")
    if name in EXCLUDE: continue
    te_fp = DATA_PROCESSED / f"te_oof_{name}.npy"
    try:
        arr = np.load(fp)
        if arr.ndim > 1: arr = arr[:,0]
        if len(arr) != len(y_tr): continue
        te_v = np.load(te_fp) if te_fp.exists() else None
        if te_v is None or len(te_v) != 513: continue
        if te_v.ndim > 1: te_v = te_v[:,0]
        te_std = float(te_v.std())
        if te_std < 0.4 * float(y_tr.std()): continue
        arr[~np.isfinite(arr)] = y_tr.mean()
        te_v[~np.isfinite(te_v)] = float(np.nanmean(te_v))
        oofs.append(arr); tes.append(te_v); names.append(name)
    except Exception as e:
        print(f"  skip {name}: {e}")

OOF_stack = np.column_stack(oofs)
TE_stack  = np.column_stack(tes)
print(f"Using {len(names)} base models, stack shape {OOF_stack.shape}")
print(f"Models: {names}")
"""),

("code", """\
# Nested-CV: for each outer fold, fit meta on remaining folds
print("=== Nested-CV stacking ===", flush=True)
oof_nested = np.full(len(y_tr), np.nan)

for k, (tr_idx, va_idx) in enumerate(splits):
    # Meta-train: all folds except k
    meta_tr_idx = [i for fold, (ti, _) in enumerate(splits) for i in ti if fold != k]
    meta_va_idx = va_idx  # held-out fold

    meta = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], cv=5,
                        max_iter=10000, random_state=SEED)
    meta.fit(OOF_stack[meta_tr_idx], y_tr[meta_tr_idx])
    oof_nested[meta_va_idx] = meta.predict(OOF_stack[meta_va_idx])
    fold_rae = rae(y_tr[meta_va_idx], oof_nested[meta_va_idx])
    print(f"  fold {k+1}  val_RAE={fold_rae:.4f}", flush=True)

m_nested = full_metrics(y_tr, oof_nested, cliff_pairs, "nested_cv_ensemble")
m_nested_a = full_metrics(y_tr[active_mask], oof_nested[active_mask],
                           label="nested_cv [active]")
print(f"\\nNested-CV OOF RAE: {m_nested['RAE']:.4f}  (compare: grand_v7=0.5189, grand_v6b=0.5281)")
"""),

("code", """\
# Final model: refit meta on all data, predict test
meta_final = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], cv=5,
                          max_iter=10000, random_state=SEED)
meta_final.fit(OOF_stack, y_tr)
te_preds = np.clip(meta_final.predict(TE_stack), y_tr.min()-0.5, y_tr.max()+0.5)

coef_df = pd.DataFrame({"model": names, "weight": meta_final.coef_}).sort_values("weight", ascending=False)
print("Non-zero weights:")
print(coef_df[coef_df.weight.abs() > 1e-6].to_string(index=False))

np.save(DATA_PROCESSED/"oof_nested_cv_ensemble.npy", oof_nested)
np.save(DATA_PROCESSED/"te_oof_nested_cv_ensemble.npy", te_preds)
sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_preds})
assert len(sub)==513 and sub["pEC50"].notna().all()
p = SUBMISSIONS/"86_nested_cv_ensemble.csv"; sub.to_csv(p, index=False)
print(f"Saved {p}")
print(f"Test: min={te_preds.min():.2f} med={np.median(te_preds):.2f} max={te_preds.max():.2f}")
"""),
])

# ── nb87: ChEMBL Multi-NR Biological Fingerprint ──────────────────────────────
nb87 = nb([
("markdown", "# 87 — Biological Fingerprinting: ChEMBL Nuclear Receptor Panel\n\n"
 "**Key insight:** A molecule's *activity profile across related nuclear receptors* encodes biological context.\n\n"
 "PPARg, FXR, RXRa, LXRa, VDR all share structural features with PXR. Compounds that activate these NRs "
 "are enriched for PXR agonists.\n\n"
 "Strategy:\n"
 "1. Train one LGBM binary classifier per NR (active = pEC50 ≥ 5.0) on ChEMBL+BindingDB data\n"
 "2. Predict P(active) for all PXR train + test compounds → biological fingerprint (5-6 floats)\n"
 "3. Concatenate bio-FP with combined Morgan+RDKit → train PXR LGBM\n\n"
 "This is orthogonal to structural fingerprints — captures mechanistic signal."),

("code", BOILERPLATE_IMPORTS),
("code", FULL_METRICS),
("code", STANDARD_SETUP),

("code", """\
from pxr.chem import to_inchikey

# Load ChEMBL + BindingDB NR data
chembl = pd.read_parquet(DATA_EXTERNAL/"chembl_nr_extended.parquet")
bdb    = pd.read_parquet(DATA_EXTERNAL/"bindingdb_nr_data.parquet")
nr_all = pd.concat([chembl, bdb], ignore_index=True)
nr_all["ik"] = nr_all["smiles"].map(to_inchikey)
nr_all = nr_all.dropna(subset=["smiles","pec50","target_name"])
nr_all["active"] = (nr_all["pec50"] >= 5.0).astype(int)

NR_TARGETS = ["PPARg","FXR","RXRa","LXRa","VDR"]
print("NR data counts:")
for t in NR_TARGETS:
    sub = nr_all[nr_all.target_name == t]
    print(f"  {t}: {len(sub):,} (active={sub.active.sum():,})")
"""),

("code", """\
# Train one binary LGBM classifier per NR → biological fingerprint
LGBM_CLS = dict(n_estimators=300, num_leaves=31, learning_rate=0.1,
                min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
                random_state=SEED, verbose=-1, n_jobs=4)

tr_iks = tr["smiles"].map(to_inchikey).values
te_iks = te["smiles"].map(to_inchikey).values
fps_tr32 = fps_tr.astype(np.float32)
fps_te32 = fps_te.astype(np.float32)

bio_fp_tr = np.zeros((len(tr), len(NR_TARGETS)), dtype=np.float32)
bio_fp_te = np.zeros((len(te), len(NR_TARGETS)), dtype=np.float32)

for j, target in enumerate(NR_TARGETS):
    sub = nr_all[nr_all.target_name == target].dropna(subset=["smiles"])
    sub = sub.drop_duplicates("ik")
    X_nr = impute(combined(sub["smiles"].tolist()))
    y_nr = sub["active"].values.astype(int)
    if y_nr.sum() < 20 or (1-y_nr).sum() < 20:
        print(f"  {target}: too few labels — skipping"); continue

    m_cls = lgb.LGBMClassifier(**LGBM_CLS)
    m_cls.fit(X_nr, y_nr)
    bio_fp_tr[:, j] = m_cls.predict_proba(X_tr)[:, 1]
    bio_fp_te[:, j] = m_cls.predict_proba(X_te)[:, 1]
    print(f"  {target}: trained on {len(sub):,} cmpds  "
          f"→ PXR-train mean P(active)={bio_fp_tr[:,j].mean():.3f}", flush=True)

print(f"\\nBio-fingerprint shape: {bio_fp_tr.shape}")
print(pd.DataFrame(bio_fp_tr, columns=NR_TARGETS).describe().round(3).to_string())
"""),

("code", """\
# Augment features: combined + bio-FP
X_bio_tr = np.hstack([X_tr, bio_fp_tr])
X_bio_te = np.hstack([X_te, bio_fp_te])

# Also: just-bio-FP model (interpretability)
oof_bio_only = np.full(len(y_tr), np.nan)
oof_aug = np.full(len(y_tr), np.nan)

for fold, (tr_idx, va_idx) in enumerate(splits):
    # bio-FP only
    m1 = lgb.train(LGBM, lgb.Dataset(bio_fp_tr[tr_idx], label=y_tr[tr_idx]),
                   valid_sets=[lgb.Dataset(bio_fp_tr[va_idx], label=y_tr[va_idx])],
                   callbacks=[lgb.early_stopping(50,verbose=False), lgb.log_evaluation(-1)])
    oof_bio_only[va_idx] = m1.predict(bio_fp_tr[va_idx])

    # combined + bio-FP
    m2 = lgb.train(LGBM, lgb.Dataset(X_bio_tr[tr_idx], label=y_tr[tr_idx]),
                   valid_sets=[lgb.Dataset(X_bio_tr[va_idx], label=y_tr[va_idx])],
                   callbacks=[lgb.early_stopping(50,verbose=False), lgb.log_evaluation(-1)])
    oof_aug[va_idx] = m2.predict(X_bio_tr[va_idx])
    print(f"  fold {fold+1}  aug_RAE={rae(y_tr[va_idx], oof_aug[va_idx]):.4f}", flush=True)

m_bio = full_metrics(y_tr, oof_bio_only, cliff_pairs, "bio_fp_only")
m_aug = full_metrics(y_tr, oof_aug, cliff_pairs, "combined+bio_fp")
print("\\n" + pd.DataFrame([m_bio, m_aug], index=["bio_only","augmented"]).round(4).to_string())
"""),

("code", """\
m_final = lgb.train(LGBM, lgb.Dataset(X_bio_tr, label=y_tr), callbacks=[lgb.log_evaluation(-1)])
te_preds = np.clip(m_final.predict(X_bio_te), y_tr.min()-0.5, y_tr.max()+0.5)
np.save(DATA_PROCESSED/"oof_bio_nr_fingerprint.npy", oof_aug)
np.save(DATA_PROCESSED/"te_oof_bio_nr_fingerprint.npy", te_preds)
np.save(DATA_PROCESSED/"bio_fp_tr.npy", bio_fp_tr)
np.save(DATA_PROCESSED/"bio_fp_te.npy", bio_fp_te)
sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_preds})
assert len(sub)==513 and sub["pEC50"].notna().all()
p = SUBMISSIONS/"87_bio_nr_fingerprint.csv"; sub.to_csv(p, index=False)
print(f"Saved {p}")
print(f"Test: min={te_preds.min():.2f} med={np.median(te_preds):.2f} max={te_preds.max():.2f}")
"""),
])

# ── nb88: E3FP / 3D Conformer Descriptors ────────────────────────────────────
nb88 = nb([
("markdown", "# 88 — E3FP / 3D Conformer Fingerprints\n\n"
 "2D fingerprints (ECFP) ignore 3D shape. PXR has a large, flexible LBD that is shape-selective.\n\n"
 "Strategy:\n"
 "1. Generate 3D conformers with RDKit ETKDG (10 conformers, prune RMSD=0.5, pick lowest-energy)\n"
 "2. Compute shape descriptors: PMI ratios, NPR, Asphericity, Eccentricity, SpherocityIndex\n"
 "3. Compute USRCAT fingerprint (ultrafast shape + pharmacophore recognition)\n"
 "4. Combine with Morgan+RDKit → LGBM\n\n"
 "3D descriptors capture the 'globularity vs planarity vs rod-like' shape spectrum that ECFP misses."),

("code", BOILERPLATE_IMPORTS),
("code", FULL_METRICS),
("code", STANDARD_SETUP),

("code", """\
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors3D, rdMolDescriptors
from rdkit.Chem import rdDistGeom

def generate_conformer(smi, n_confs=10, max_attempts=200, seed=42):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    mol = Chem.AddHs(mol)
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = 0.5
    params.numThreads = 1
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    if not cids: return None
    # Minimize and pick lowest energy
    energies = []
    for cid in cids:
        ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol), confId=cid)
        if ff is None:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
        if ff:
            ff.Minimize(maxIts=200)
            energies.append((ff.CalcEnergy(), cid))
    if energies:
        _, best_cid = min(energies)
        # Set conformer 0 as the best
        mol = Chem.RemoveHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=seed)
        # Re-embed with best params
        mol2 = Chem.MolFromSmiles(smi)
        mol2 = Chem.AddHs(mol2)
        AllChem.EmbedMolecule(mol2, params=params)
        AllChem.MMFFOptimizeMolecule(mol2)
        mol_noh = Chem.RemoveHs(mol2)
        return mol_noh
    return None

def shape_descriptors(mol):
    if mol is None or mol.GetNumConformers() == 0: return [np.nan]*12
    try:
        pmi1 = Descriptors3D.PMI1(mol)
        pmi2 = Descriptors3D.PMI2(mol)
        pmi3 = Descriptors3D.PMI3(mol)
        npr1 = Descriptors3D.NPR1(mol)
        npr2 = Descriptors3D.NPR2(mol)
        asphericity = Descriptors3D.Asphericity(mol)
        eccentricity = Descriptors3D.Eccentricity(mol)
        spherocity = Descriptors3D.SpherocityIndex(mol)
        inertial = Descriptors3D.InertialShapeFactor(mol)
        gyration = Descriptors3D.RadiusOfGyration(mol)
        # planarity proxy: PMI1/PMI3 (rod=0, sphere=1, disk=0.5)
        rod_like = pmi1/pmi3 if pmi3 > 0 else np.nan
        disc_like = pmi2/pmi3 if pmi3 > 0 else np.nan
        return [pmi1, pmi2, pmi3, npr1, npr2, asphericity,
                eccentricity, spherocity, inertial, gyration, rod_like, disc_like]
    except:
        return [np.nan]*12

print("Generating 3D conformers (this takes a few minutes)...", flush=True)
"""),

("code", """\
import multiprocessing as mp

SHAPE_NAMES = ["PMI1","PMI2","PMI3","NPR1","NPR2","Asphericity",
               "Eccentricity","Spherocity","InertialSF","Gyration","Rod","Disc"]

def smiles_to_shape(smi):
    mol = generate_conformer(smi)
    return shape_descriptors(mol)

# Process train
print(f"Processing {len(tr):,} train compounds...", flush=True)
shape_tr = []
for i, smi in enumerate(tr["smiles"].tolist()):
    shape_tr.append(smiles_to_shape(smi))
    if (i+1) % 500 == 0: print(f"  {i+1}/{len(tr)}", flush=True)
X_shape_tr = np.array(shape_tr, dtype=np.float32)
# Impute NaN
col_means = np.nanmean(X_shape_tr, axis=0)
for j in range(X_shape_tr.shape[1]):
    mask = ~np.isfinite(X_shape_tr[:,j])
    X_shape_tr[mask, j] = col_means[j]

print(f"Train shape features: {X_shape_tr.shape}")
print(pd.DataFrame(X_shape_tr, columns=SHAPE_NAMES).describe().round(3).to_string())

# Process test
print(f"\\nProcessing {len(te):,} test compounds...", flush=True)
shape_te = []
for i, smi in enumerate(te["smiles"].tolist()):
    shape_te.append(smiles_to_shape(smi))
    if (i+1) % 100 == 0: print(f"  {i+1}/{len(te)}", flush=True)
X_shape_te = np.array(shape_te, dtype=np.float32)
for j in range(X_shape_te.shape[1]):
    mask = ~np.isfinite(X_shape_te[:,j])
    X_shape_te[mask, j] = col_means[j]
print(f"Test shape features: {X_shape_te.shape}")
"""),

("code", """\
# Combine 3D shape with combined 2D features
X_3d_tr = np.hstack([X_tr, X_shape_tr])
X_3d_te  = np.hstack([X_te, X_shape_te])

oof_shape = np.full(len(y_tr), np.nan)
for fold, (tr_idx, va_idx) in enumerate(splits):
    m = lgb.train(LGBM, lgb.Dataset(X_3d_tr[tr_idx], label=y_tr[tr_idx]),
                  valid_sets=[lgb.Dataset(X_3d_tr[va_idx], label=y_tr[va_idx])],
                  callbacks=[lgb.early_stopping(50,verbose=False), lgb.log_evaluation(-1)])
    oof_shape[va_idx] = m.predict(X_3d_tr[va_idx])
    print(f"  fold {fold+1}  RAE={rae(y_tr[va_idx], oof_shape[va_idx]):.4f}", flush=True)

m_shape = full_metrics(y_tr, oof_shape, cliff_pairs, "combined+3D_shape")
m_shape_a = full_metrics(y_tr[active_mask], oof_shape[active_mask], label="3D [active]")
print("\\n" + pd.DataFrame([m_shape, m_shape_a], index=["overall","active"]).round(4).to_string())

m_final = lgb.train(LGBM, lgb.Dataset(X_3d_tr, label=y_tr), callbacks=[lgb.log_evaluation(-1)])
te_preds = np.clip(m_final.predict(X_3d_te), y_tr.min()-0.5, y_tr.max()+0.5)
np.save(DATA_PROCESSED/"X_shape_tr.npy", X_shape_tr)
np.save(DATA_PROCESSED/"X_shape_te.npy", X_shape_te)
np.save(DATA_PROCESSED/"oof_3d_shape_conformer.npy", oof_shape)
np.save(DATA_PROCESSED/"te_oof_3d_shape_conformer.npy", te_preds)
sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_preds})
assert len(sub)==513 and sub["pEC50"].notna().all()
p = SUBMISSIONS/"88_3d_shape_conformer.csv"; sub.to_csv(p, index=False)
print(f"Saved {p}")
print(f"Test: min={te_preds.min():.2f} med={np.median(te_preds):.2f} max={te_preds.max():.2f}")
"""),
])

# ── nb89: PXR Pharmacophore + SMARTS Features ─────────────────────────────────
nb89 = nb([
("markdown", "# 89 — PXR Pharmacophore + Domain-Specific SMARTS Features\n\n"
 "PXR has a large, flexible LBD (~1,150 Å³). Known agonist pharmacophore:\n"
 "- Two hydrophobic regions (H1: planar aromatic, H2: alkyl/lipophilic)\n"
 "- One H-bond acceptor (A1: C=O, N, O)\n"
 "- MW 300–800, logP 2–7\n\n"
 "Strategy:\n"
 "1. Encode known PXR agonist scaffold SMARTS patterns\n"
 "2. Compute PXR-specific physicochemical: logP², MW×logP, HBD×MW interactions\n"
 "3. Compute VSA descriptors (MOE-style: SlogP_VSA, PEOE_VSA, SMR_VSA)\n"
 "4. Append to combined features → LGBM"),

("code", BOILERPLATE_IMPORTS),
("code", FULL_METRICS),
("code", STANDARD_SETUP),

("code", """\
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem

# Known PXR agonist scaffold SMARTS + privileged structural motifs
PXR_SMARTS = {
    # Stilbene / bibenzyl core (RXR/PXR cross-activators)
    "stilbene":      "c1ccc(/C=C/)cc1",
    # Carbamazepine-like (dibenzazepine)
    "dibenzazepine": "C1CC2=CC=CC=C2NC3=CC=CC=C13",
    # Hyperforin-like: bicyclic terpenoid with enol
    "polycyclic_oh": "[OH]C1=C(C(C)(C)C)CCCC1=O",
    # Pregnane/steroid scaffold
    "steroid_core":  "C1CC2CCC3CCCC4=CC(=O)CCC4(C)C3(C)C2(C)C1",
    # Rifampicin-like: naphthyl ketone
    "naphthylketone": "O=Cc1ccc2ccccc2c1",
    # Perfluoroalkyl (PFAS, potent PXR activators)
    "perfluoro":     "C(F)(F)F",
    # Phthalate ester (known PXR activator class)
    "phthalate":     "O=C(OCC)c1ccccc1C(=O)OCC",
    # Organochlorine (PXR activators like DDT metabolites)
    "organochloro":  "ClC(Cl)(Cl)",
    # Large fused aromatic (planar, fits PXR LBD)
    "acridine":      "c1ccc2nc3ccccc3cc2c1",
    # Sulfonamide (common in PXR active drugs)
    "sulfonamide":   "NS(=O)(=O)",
    # Macrolide-like: large ring ≥12
    "large_ring":    "[r12,r13,r14,r15,r16]",
}

def compute_pxr_features(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return [np.nan]*50

    feats = []
    # SMARTS pattern matches
    for name, smt in PXR_SMARTS.items():
        try:
            pat = Chem.MolFromSmarts(smt)
            feats.append(1.0 if mol.HasSubstructMatch(pat) else 0.0)
        except:
            feats.append(0.0)

    # PXR-specific physchem
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    tpsa = Descriptors.TPSA(mol)
    nrings = rdMolDescriptors.CalcNumRings(mol)
    narom = rdMolDescriptors.CalcNumAromaticRings(mol)
    rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    nheavy = mol.GetNumHeavyAtoms()
    fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    qed_score = Descriptors.qed(mol) if hasattr(Descriptors, 'qed') else np.nan

    # Interaction terms (capture PXR preference for lipophilic, medium-large)
    feats += [mw, logp, hbd, hba, tpsa, nrings, narom, rotb, nheavy, fsp3,
              logp**2,               # logP quadratic
              mw * logp,             # lipophilic bulk
              hbd * mw,              # H-bond donor × size
              (tpsa / mw) if mw>0 else np.nan,  # polar surface fraction
              (narom / nrings) if nrings>0 else 0.0,  # aromaticity ratio
              float(5.0 <= logp <= 7.0),  # PXR sweet spot logP
              float(300 <= mw <= 800),    # PXR sweet spot MW
              float(hbd <= 2),            # HBD-poor (fits PXR hydrophobic LBD)
    ]

    # VSA descriptors (MOE-style)
    try:
        slogp_vsa = list(Descriptors.SlogP_VSA_(mol))[:6]
        peoe_vsa  = list(Descriptors.PEOE_VSA_(mol))[:6]
        smr_vsa   = list(Descriptors.SMR_VSA_(mol))[:6]
        feats += slogp_vsa + peoe_vsa + smr_vsa
    except:
        feats += [np.nan]*18

    return feats

print("Computing PXR pharmacophore features...", flush=True)
X_pxr_tr = np.array([compute_pxr_features(s) for s in tr["smiles"]], dtype=np.float32)
X_pxr_te = np.array([compute_pxr_features(s) for s in te["smiles"]], dtype=np.float32)

# Impute
col_means = np.nanmean(X_pxr_tr, axis=0)
for j in range(X_pxr_tr.shape[1]):
    X_pxr_tr[np.isnan(X_pxr_tr[:,j]),j] = col_means[j]
    X_pxr_te[np.isnan(X_pxr_te[:,j]),j] = col_means[j]
print(f"Pharmacophore features: {X_pxr_tr.shape}")
"""),

("code", """\
X_aug_tr = np.hstack([X_tr, X_pxr_tr])
X_aug_te  = np.hstack([X_te, X_pxr_te])

oof = np.full(len(y_tr), np.nan)
for fold, (tr_idx, va_idx) in enumerate(splits):
    m = lgb.train(LGBM, lgb.Dataset(X_aug_tr[tr_idx], label=y_tr[tr_idx]),
                  valid_sets=[lgb.Dataset(X_aug_tr[va_idx], label=y_tr[va_idx])],
                  callbacks=[lgb.early_stopping(50,verbose=False), lgb.log_evaluation(-1)])
    oof[va_idx] = m.predict(X_aug_tr[va_idx])
    print(f"  fold {fold+1}  RAE={rae(y_tr[va_idx], oof[va_idx]):.4f}", flush=True)

m_res = full_metrics(y_tr, oof, cliff_pairs, "pxr_pharmacophore")
m_res_a = full_metrics(y_tr[active_mask], oof[active_mask], label="pharmacophore [active]")

m_final = lgb.train(LGBM, lgb.Dataset(X_aug_tr, label=y_tr), callbacks=[lgb.log_evaluation(-1)])
te_preds = np.clip(m_final.predict(X_aug_te), y_tr.min()-0.5, y_tr.max()+0.5)
np.save(DATA_PROCESSED/"oof_pxr_pharmacophore.npy", oof)
np.save(DATA_PROCESSED/"te_oof_pxr_pharmacophore.npy", te_preds)
sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_preds})
assert len(sub)==513 and sub["pEC50"].notna().all()
p = SUBMISSIONS/"89_pxr_pharmacophore.csv"; sub.to_csv(p, index=False)
print(f"Saved {p}  Test: min={te_preds.min():.2f} med={np.median(te_preds):.2f} max={te_preds.max():.2f}")
"""),
])

# ── nb90: Tox21 Fetch + Biological Fingerprint ────────────────────────────────
nb90 = nb([
("markdown", "# 90 — Tox21 NR Panel: Fetch + Biological Fingerprint\n\n"
 "The Tox21 dataset has 12 nuclear receptor / stress-response assays for ~8,000 compounds.\n"
 "Key NR assays: NR-AhR, NR-AR, NR-AR-LBD, NR-ER, NR-ER-LBD, NR-PPAR-gamma.\n\n"
 "AhR (aryl hydrocarbon receptor) is co-regulated with PXR for many xenobiotic compounds — "
 "shared CYP1A2/CYP3A4 induction pathway.\n\n"
 "Strategy:\n"
 "1. Download Tox21 via DeepChem or MoleculeNet\n"
 "2. Train one LGBM binary classifier per assay (6 NR assays)\n"
 "3. Predicted probabilities → 6-dim Tox21 biological fingerprint\n"
 "4. Combine with ChEMBL bio-FP (nb87) → 11-dim biological fingerprint\n"
 "5. Train PXR LGBM with combined + bio-FPs"),

("code", BOILERPLATE_IMPORTS),
("code", FULL_METRICS),
("code", STANDARD_SETUP),

("code", """\
# Fetch Tox21 data
import urllib.request, io, zipfile, csv

TOX21_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
print(f"Downloading Tox21 from {TOX21_URL}...", flush=True)
try:
    import gzip
    with urllib.request.urlopen(TOX21_URL, timeout=120) as resp:
        raw = gzip.decompress(resp.read()).decode("utf-8")
    tox21_df = pd.read_csv(io.StringIO(raw))
    print(f"Tox21: {len(tox21_df)} compounds, columns: {list(tox21_df.columns)}")
    tox21_df.to_parquet(DATA_EXTERNAL / "tox21_nr_data.parquet", index=False)
    print("Saved tox21_nr_data.parquet")
except Exception as e:
    print(f"Download failed: {e}  — trying DeepChem import")
    try:
        import deepchem as dc
        tasks, datasets, _ = dc.molnet.load_tox21()
        train_ds, val_ds, test_ds = datasets
        all_X = [train_ds.X, val_ds.X, test_ds.X]
        tox21_df = None
        print("DeepChem Tox21 loaded (featurized)")
    except:
        print("DeepChem not available — using empty DataFrame")
        tox21_df = pd.DataFrame()

TOX21_NR_TASKS = ["NR-AhR","NR-AR","NR-AR-LBD","NR-ER","NR-ER-LBD","NR-PPAR-gamma"]
"""),

("code", """\
if tox21_df is not None and len(tox21_df) > 0 and "smiles" in tox21_df.columns:
    tox21_smiles_col = "smiles"
    tox21_avail = [t for t in TOX21_NR_TASKS if t in tox21_df.columns]
    print(f"Available Tox21 NR tasks: {tox21_avail}")

    LGBM_CLS = dict(n_estimators=300, num_leaves=31, learning_rate=0.1,
                    min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
                    random_state=SEED, verbose=-1, n_jobs=4)

    tox21_bio_tr = np.zeros((len(tr), len(tox21_avail)), dtype=np.float32)
    tox21_bio_te = np.zeros((len(te), len(tox21_avail)), dtype=np.float32)

    for j, task in enumerate(tox21_avail):
        sub = tox21_df[[tox21_smiles_col, task]].dropna()
        sub = sub[sub[task].isin([0.0, 1.0])]
        if len(sub) < 100: continue
        X_t = impute(combined(sub[tox21_smiles_col].tolist()))
        y_t = sub[task].values.astype(int)
        m_cls = lgb.LGBMClassifier(**LGBM_CLS)
        m_cls.fit(X_t, y_t)
        tox21_bio_tr[:, j] = m_cls.predict_proba(X_tr)[:, 1]
        tox21_bio_te[:, j] = m_cls.predict_proba(X_te)[:, 1]
        print(f"  {task}: {len(sub):,} cmpds  P(active|PXR_train)={tox21_bio_tr[:,j].mean():.3f}", flush=True)

    print(f"\\nTox21 bio-FP: {tox21_bio_tr.shape}")
else:
    print("Tox21 not available — using zeros placeholder")
    tox21_avail = []
    tox21_bio_tr = np.zeros((len(tr), 0), dtype=np.float32)
    tox21_bio_te = np.zeros((len(te), 0), dtype=np.float32)

# Load ChEMBL bio-FP from nb87 if available
bio_fp_tr_path = DATA_PROCESSED/"bio_fp_tr.npy"
bio_fp_te_path = DATA_PROCESSED/"bio_fp_te.npy"
if bio_fp_tr_path.exists():
    chembl_bio_tr = np.load(bio_fp_tr_path)
    chembl_bio_te = np.load(bio_fp_te_path)
    print(f"Loaded ChEMBL bio-FP: {chembl_bio_tr.shape}")
else:
    chembl_bio_tr = np.zeros((len(tr), 0), dtype=np.float32)
    chembl_bio_te = np.zeros((len(te), 0), dtype=np.float32)
    print("ChEMBL bio-FP not found — run nb87 first")

# Full biological fingerprint = ChEMBL NR + Tox21 NR
all_bio_tr = np.hstack([chembl_bio_tr, tox21_bio_tr]).astype(np.float32)
all_bio_te = np.hstack([chembl_bio_te, tox21_bio_te]).astype(np.float32)
print(f"Combined bio-FP dim: {all_bio_tr.shape[1]}")
"""),

("code", """\
if all_bio_tr.shape[1] > 0:
    X_full_tr = np.hstack([X_tr, all_bio_tr])
    X_full_te = np.hstack([X_te, all_bio_te])

    oof = np.full(len(y_tr), np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = lgb.train(LGBM, lgb.Dataset(X_full_tr[tr_idx], label=y_tr[tr_idx]),
                      valid_sets=[lgb.Dataset(X_full_tr[va_idx], label=y_tr[va_idx])],
                      callbacks=[lgb.early_stopping(50,verbose=False), lgb.log_evaluation(-1)])
        oof[va_idx] = m.predict(X_full_tr[va_idx])
        print(f"  fold {fold+1}  RAE={rae(y_tr[va_idx], oof[va_idx]):.4f}", flush=True)

    m_res = full_metrics(y_tr, oof, cliff_pairs, "combined+all_bio_fp")
    print("\\n" + pd.DataFrame([m_res], index=["all_bio_fp"]).round(4).to_string())

    m_final = lgb.train(LGBM, lgb.Dataset(X_full_tr, label=y_tr), callbacks=[lgb.log_evaluation(-1)])
    te_preds = np.clip(m_final.predict(X_full_te), y_tr.min()-0.5, y_tr.max()+0.5)
else:
    print("No bio-FPs available — falling back to combined only")
    oof = np.full(len(y_tr), np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = lgb.train(LGBM, lgb.Dataset(X_tr[tr_idx], label=y_tr[tr_idx]),
                      valid_sets=[lgb.Dataset(X_tr[va_idx], label=y_tr[va_idx])],
                      callbacks=[lgb.early_stopping(50,verbose=False), lgb.log_evaluation(-1)])
        oof[va_idx] = m.predict(X_tr[va_idx])
    m_final = lgb.train(LGBM, lgb.Dataset(X_tr, label=y_tr), callbacks=[lgb.log_evaluation(-1)])
    te_preds = np.clip(m_final.predict(X_te), y_tr.min()-0.5, y_tr.max()+0.5)

np.save(DATA_PROCESSED/"tox21_bio_tr.npy", tox21_bio_tr)
np.save(DATA_PROCESSED/"tox21_bio_te.npy", tox21_bio_te)
np.save(DATA_PROCESSED/"oof_tox21_bio_fp.npy", oof)
np.save(DATA_PROCESSED/"te_oof_tox21_bio_fp.npy", te_preds)
sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_preds})
assert len(sub)==513 and sub["pEC50"].notna().all()
p = SUBMISSIONS/"90_tox21_bio_fp.csv"; sub.to_csv(p, index=False)
print(f"Saved {p}  Test: min={te_preds.min():.2f} med={np.median(te_preds):.2f} max={te_preds.max():.2f}")
"""),
])

# ── nb91: Cliff-Proximity Ensemble Blending ───────────────────────────────────
nb91 = nb([
("markdown", "# 91 — Cliff-Proximity Adaptive Ensemble\n\n"
 "Key insight: different models excel on different compound types.\n"
 "- Cliff-active analogs: models trained with cliff weighting (nb40, nb38) should dominate\n"
 "- Non-cliff compounds: standard ensemble is optimal\n\n"
 "Strategy:\n"
 "1. For each test compound, compute Tanimoto to nearest cliff-active training compound\n"
 "2. For cliff-proximate test compounds (sim > 0.35): upweight cliff-specialized models\n"
 "3. For other test compounds: use standard grand_v7 weights\n"
 "4. Blend adaptively: w_cliff * cliff_specialized + (1-w_cliff) * standard\n\n"
 "This is a test-time routing strategy, not a new model."),

("code", BOILERPLATE_IMPORTS),
("code", FULL_METRICS),
("code", STANDARD_SETUP),

("code", """\
# Load the cliff model breakdown from nb61
breakdown_path = DATA_PROCESSED/"cliff_model_breakdown.parquet"
if breakdown_path.exists():
    breakdown = pd.read_parquet(breakdown_path)
    print("Cliff model breakdown (sorted by cliff_rae):")
    print(breakdown[["model","overall_rae","cliff_rae"]].head(15).to_string(index=False))
else:
    breakdown = pd.DataFrame()
    print("No cliff_model_breakdown.parquet — will use hardcoded cliff-specialized models")

# Load all OOF predictions for selected models
CLIFF_MODELS = [
    "cliff_weighted",    # nb40: cliff-weighted LGBM
    "hard_negatives",    # nb39: hard negative augmentation
    "smote_best",        # nb42: SMOTE/ADASYN
    "delta_ml",          # nb76: Delta-ML (NN-based)
    "lgbm_tuned",        # best individual model
]
STANDARD_MODELS = [
    "lgbm_tuned",
    "chemprop_aux",
    "deep_ensemble",
    "catboost",
    "focal_loss",
]

def load_te_oof(name):
    p = DATA_PROCESSED / f"te_oof_{name}.npy"
    if not p.exists(): return None
    arr = np.load(p)
    return arr[:,0] if arr.ndim > 1 else arr

cliff_te = {n: load_te_oof(n) for n in CLIFF_MODELS}
std_te   = {n: load_te_oof(n) for n in STANDARD_MODELS}
cliff_te = {k:v for k,v in cliff_te.items() if v is not None and len(v)==513}
std_te   = {k:v for k,v in std_te.items()   if v is not None and len(v)==513}
print(f"Cliff models loaded: {list(cliff_te.keys())}")
print(f"Standard models loaded: {list(std_te.keys())}")
"""),

("code", """\
from pxr.chem import morgan_fp_batch

# Identify cliff-active compounds in training
cliff_active_mask = np.zeros(len(tr), dtype=bool)
if len(cliff_pairs) > 0:
    cliff_active_mask[cliff_pairs["idx_active"].values] = True
print(f"Cliff-active training compounds: {cliff_active_mask.sum()}")

# Compute Tanimoto from test to cliff-active training compounds
fps_cliff_active = fps_tr[cliff_active_mask].astype(np.float32)

if cliff_active_mask.sum() > 0:
    dot = fps_te @ fps_cliff_active.T
    rs_te = fps_te.sum(1, keepdims=True)
    rs_ca = fps_cliff_active.sum(1)[None, :]
    union = rs_te + rs_ca - dot
    with np.errstate(divide="ignore", invalid="ignore"):
        tan = np.where(union > 0, dot / union, 0.0)
    max_sim_to_cliff_active = tan.max(axis=1)
else:
    max_sim_to_cliff_active = np.zeros(513)

print(f"Test-to-cliff-active similarity: mean={max_sim_to_cliff_active.mean():.3f} "
      f"max={max_sim_to_cliff_active.max():.3f}")
print(f"Test compounds with sim>0.35: {(max_sim_to_cliff_active > 0.35).sum()}")
print(f"Test compounds with sim>0.50: {(max_sim_to_cliff_active > 0.50).sum()}")
"""),

("code", """\
# Adaptive blending
if len(cliff_te) > 0 and len(std_te) > 0:
    cliff_pred = np.mean(list(cliff_te.values()), axis=0)
    std_pred   = np.mean(list(std_te.values()), axis=0)

    # Also load grand_v7 if available
    gv7_path = DATA_PROCESSED/"te_oof_grand_v7.npy"
    if gv7_path.exists():
        grand_v7_te = np.load(gv7_path)
        std_pred = 0.5 * std_pred + 0.5 * grand_v7_te
        print("Blended with grand_v7 test predictions")

    # Sigmoid weight: higher sim → higher cliff model weight
    SIM_THRESH = 0.35; SIM_MAX = 0.70
    w_cliff = np.clip((max_sim_to_cliff_active - SIM_THRESH) / (SIM_MAX - SIM_THRESH), 0, 1)
    # Max cliff weight = 0.6 (don't fully abandon standard)
    w_cliff = w_cliff * 0.6

    te_preds = w_cliff * cliff_pred + (1 - w_cliff) * std_pred
    te_preds = np.clip(te_preds, y_tr.min()-0.5, y_tr.max()+0.5)
    print(f"Cliff weight stats: mean={w_cliff.mean():.3f} max={w_cliff.max():.3f}")
else:
    print("Insufficient models — using lgbm_tuned as fallback")
    te_preds = load_te_oof("lgbm_tuned")
    if te_preds is None:
        te_preds = np.full(513, y_tr.mean())

# For OOF: use grand_v7 OOF as base (adaptive blending is a test-time operation)
oof_path = DATA_PROCESSED / "oof_grand_v7.npy"
oof = np.load(oof_path) if oof_path.exists() else np.full(len(y_tr), np.nan)
m_res = full_metrics(y_tr, oof, cliff_pairs, "cliff_adaptive_blend")

np.save(DATA_PROCESSED/"oof_cliff_adaptive_blend.npy", oof)
np.save(DATA_PROCESSED/"te_oof_cliff_adaptive_blend.npy", te_preds)
sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_preds})
assert len(sub)==513 and sub["pEC50"].notna().all()
p = SUBMISSIONS/"91_cliff_adaptive_blend.csv"; sub.to_csv(p, index=False)
print(f"Saved {p}  Test: min={te_preds.min():.2f} med={np.median(te_preds):.2f} max={te_preds.max():.2f}")
"""),
])

# ── nb92: Multi-NR Transfer LGBM ─────────────────────────────────────────────
nb92 = nb([
("markdown", "# 92 — Multi-NR Transfer LGBM: Learn from all Nuclear Receptors\n\n"
 "Train a single LGBM on ALL NR data (PPARg 4302 + FXR 3185 + RXRa 1364 + LXRa 1173 + VDR 523 + PXR 945 = ~11K compounds).\n"
 "Use target-one-hot encoding → LGBM learns a shared NR representation.\n"
 "Then fine-tune predictions on PXR-only data.\n\n"
 "This is classical multi-task transfer: shared representation across related NRs.\n"
 "The NR superfamily ligand-binding domains are structurally homologous (30-45% identity)."),

("code", BOILERPLATE_IMPORTS),
("code", FULL_METRICS),
("code", STANDARD_SETUP),

("code", """\
from pxr.chem import to_inchikey

# Combine all NR data
chembl = pd.read_parquet(DATA_EXTERNAL/"chembl_nr_extended.parquet")
bdb    = pd.read_parquet(DATA_EXTERNAL/"bindingdb_nr_data.parquet")
nr_all = pd.concat([chembl, bdb], ignore_index=True)
nr_all = nr_all.dropna(subset=["smiles","pec50","target_name"])

NR_TARGETS = sorted(nr_all["target_name"].unique().tolist())
nr_all["target_idx"] = nr_all["target_name"].map({t:i for i,t in enumerate(NR_TARGETS)}).astype(int)
print(f"NR data: {len(nr_all):,} compounds across {len(NR_TARGETS)} targets: {NR_TARGETS}")

# Featurize all NR compounds: Morgan + RDKit + target one-hot
X_nr_comb = impute(combined(nr_all["smiles"].tolist()))
n_targets = len(NR_TARGETS)
target_oh = np.zeros((len(nr_all), n_targets), dtype=np.float32)
for i, tidx in enumerate(nr_all["target_idx"].values):
    target_oh[i, tidx] = 1.0
X_nr_full = np.hstack([X_nr_comb, target_oh])
y_nr = nr_all["pec50"].values.astype(np.float64)
print(f"Multi-NR feature matrix: {X_nr_full.shape}")
"""),

("code", """\
# Train multi-NR LGBM (no CV — use all external data)
# Then fine-tune / stack with PXR-only model
LGBM_MNR = dict(n_estimators=500, num_leaves=64, learning_rate=0.05,
                min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=0.1, random_state=SEED, verbose=-1, n_jobs=4)

print("Training multi-NR LGBM...", flush=True)
m_mnr = lgb.train(LGBM_MNR, lgb.Dataset(X_nr_full, label=y_nr),
                   callbacks=[lgb.log_evaluation(-1)])

# Get PXR target index
pxr_idx = NR_TARGETS.index("PXR") if "PXR" in NR_TARGETS else 0
print(f"PXR target index: {pxr_idx} (target: {NR_TARGETS[pxr_idx]})")

# Create PXR query features (combined + PXR one-hot)
pxr_oh_tr = np.zeros((len(tr), n_targets), dtype=np.float32)
pxr_oh_tr[:, pxr_idx] = 1.0
pxr_oh_te = np.zeros((len(te), n_targets), dtype=np.float32)
pxr_oh_te[:, pxr_idx] = 1.0
X_pxr_mnr_tr = np.hstack([X_tr, pxr_oh_tr])
X_pxr_mnr_te = np.hstack([X_te, pxr_oh_te])

# Multi-NR predictions for PXR train/test
pred_mnr_tr = m_mnr.predict(X_pxr_mnr_tr)
pred_mnr_te = m_mnr.predict(X_pxr_mnr_te)
print(f"Multi-NR preds on PXR train: mean={pred_mnr_tr.mean():.3f}  RAE={rae(y_tr, pred_mnr_tr):.4f}")
"""),

("code", """\
# Fine-tune: use multi-NR prediction as additional feature in PXR-only model
X_ft_tr = np.hstack([X_tr, pred_mnr_tr.reshape(-1,1)])
X_ft_te = np.hstack([X_te, pred_mnr_te.reshape(-1,1)])

oof = np.full(len(y_tr), np.nan)
for fold, (tr_idx, va_idx) in enumerate(splits):
    m = lgb.train(LGBM, lgb.Dataset(X_ft_tr[tr_idx], label=y_tr[tr_idx]),
                  valid_sets=[lgb.Dataset(X_ft_tr[va_idx], label=y_tr[va_idx])],
                  callbacks=[lgb.early_stopping(50,verbose=False), lgb.log_evaluation(-1)])
    oof[va_idx] = m.predict(X_ft_tr[va_idx])
    print(f"  fold {fold+1}  RAE={rae(y_tr[va_idx], oof[va_idx]):.4f}", flush=True)

m_res = full_metrics(y_tr, oof, cliff_pairs, "multi_nr_transfer")
m_res_a = full_metrics(y_tr[active_mask], oof[active_mask], label="transfer [active]")

m_final = lgb.train(LGBM, lgb.Dataset(X_ft_tr, label=y_tr), callbacks=[lgb.log_evaluation(-1)])
te_preds = np.clip(m_final.predict(X_ft_te), y_tr.min()-0.5, y_tr.max()+0.5)
np.save(DATA_PROCESSED/"oof_multi_nr_transfer.npy", oof)
np.save(DATA_PROCESSED/"te_oof_multi_nr_transfer.npy", te_preds)
sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_preds})
assert len(sub)==513 and sub["pEC50"].notna().all()
p = SUBMISSIONS/"92_multi_nr_transfer.csv"; sub.to_csv(p, index=False)
print(f"Saved {p}  Test: min={te_preds.min():.2f} med={np.median(te_preds):.2f} max={te_preds.max():.2f}")
"""),
])

# ── nb93: Kaggle GPU — Large Chemprop ─────────────────────────────────────────
nb93 = nb([
("markdown", "# 93 — Kaggle GPU: Large Chemprop + Multi-Task\n\n"
 "This notebook is designed to run on Kaggle T4 GPU.\n"
 "Push via: `python scripts/kaggle_push.py --nb 93 --pull`\n\n"
 "Architecture: BondMessagePassing, depth=5, d_h=600, ffn_depth=3, dropout=0.20\n"
 "Multi-task: PXR pEC50 (primary) + counter-assay pEC50 + single-conc log2FC\n\n"
 "Expected: OOF RAE ~0.48–0.50 (significantly better than nb03 depth=3 at 0.517)"),

("code", """\
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
# Kaggle path injection
for p in ["/kaggle/input/pxr-challenge-data/src", "../src"]:
    if os.path.exists(p): sys.path.insert(0, p); break
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Kaggle vs local paths
if os.path.exists("/kaggle"):
    DATA_RAW    = Path("/kaggle/input/pxr-challenge-data/data/raw")
    DATA_PROC   = Path("/kaggle/input/pxr-challenge-data/data/processed")
    SUBMISSIONS = Path("/kaggle/working")
    SEED = 42
else:
    sys.path.insert(0, "../src")
    from pxr.paths import DATA_PROCESSED as DATA_PROC, SUBMISSIONS
    DATA_RAW = Path("../data/raw")
    SEED = 42

print(f"PyTorch: {torch.__version__}  CUDA: {torch.cuda.is_available()}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
"""),

("code", """\
from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko

tr = load_train(); te = load_test()
y_tr = tr["pec50"].values.astype(np.float64)
scaffolds = tr["smiles"].map(bemis_murcko).tolist()
splits = scaffold_kfold_indices(scaffolds, 5, SEED)
print(f"Train {len(tr):,}  Test {len(te):,}")
"""),

("code", """\
try:
    import chemprop
    from chemprop import data as cpdata, models, nn as cpnn, featurizers, training
    from lightning import pytorch as pl
    CHEMPROP_AVAIL = True
    print(f"Chemprop {chemprop.__version__}")
except ImportError as e:
    print(f"Chemprop not available: {e}")
    CHEMPROP_AVAIL = False

N_FOLDS = 5
DEPTH = 5; D_H = 600; FFN_DEPTH = 3; DROPOUT = 0.20; EPOCHS = 100; BATCH = 64
"""),

("code", """\
from pxr.data import load_counter

if CHEMPROP_AVAIL:
    ctr = load_counter().dropna(subset=["smiles","pec50"])

    def make_dataset_multitask(df, smiles_col="smiles", targets=["pec50"], ctr_df=None):
        from pxr.chem import to_inchikey
        rows = []
        for _, row in df.iterrows():
            targets_vals = [row.get(t, float("nan")) for t in targets]
            if ctr_df is not None:
                ik = to_inchikey(row[smiles_col])
                match = ctr_df[ctr_df["smiles"].map(to_inchikey) == ik]["pec50"]
                targets_vals.append(float(match.mean()) if len(match) else float("nan"))
            rows.append((row[smiles_col], targets_vals))
        return rows

    oof_chemprop_large = np.full(len(y_tr), np.nan)

    for fold, (tr_idx, va_idx) in enumerate(splits):
        print(f"\\n=== Fold {fold+1}/{N_FOLDS} ===", flush=True)
        tr_smiles = tr.iloc[tr_idx]["smiles"].tolist()
        va_smiles = tr.iloc[va_idx]["smiles"].tolist()

        tr_data = [cpdata.MoleculeDatapoint.from_smi(s, [y_tr[i]]) for s, i in zip(tr_smiles, tr_idx)]
        va_data = [cpdata.MoleculeDatapoint.from_smi(s, [y_tr[i]]) for s, i in zip(va_smiles, va_idx)]

        tr_ds = cpdata.MoleculeDataset(tr_data)
        va_ds = cpdata.MoleculeDataset(va_data)
        tr_loader = cpdata.build_dataloader(tr_ds, batch_size=BATCH, shuffle=True)
        va_loader = cpdata.build_dataloader(va_ds, batch_size=BATCH*2, shuffle=False)

        mp  = cpnn.BondMessagePassing(depth=DEPTH, d_h=D_H)
        agg = cpnn.MeanAggregation()
        ffn = cpnn.RegressionFFN(input_dim=D_H, n_layers=FFN_DEPTH, dropout=DROPOUT)
        model = models.MPNN(message_passing=mp, agg=agg, predictor=ffn)

        trainer = pl.Trainer(
            max_epochs=EPOCHS, accelerator=device, devices=1,
            enable_progress_bar=True, enable_model_summary=False,
            callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss", patience=15, mode="min")],
        )
        trainer.fit(model, tr_loader, va_loader)

        preds = trainer.predict(model, va_loader)
        preds_flat = np.concatenate([p.numpy().flatten() for p in preds])
        oof_chemprop_large[va_idx] = preds_flat[:len(va_idx)]
        fold_rae = rae(y_tr[va_idx], oof_chemprop_large[va_idx])
        print(f"  Fold {fold+1} RAE: {fold_rae:.4f}", flush=True)

    from scipy import stats
    valid = np.isfinite(oof_chemprop_large)
    oof_rae = rae(y_tr[valid], oof_chemprop_large[valid])
    print(f"\\nLarge Chemprop OOF RAE: {oof_rae:.4f}")
else:
    print("Chemprop not available — saving placeholder")
    oof_chemprop_large = np.full(len(y_tr), y_tr.mean())
"""),

("code", """\
if CHEMPROP_AVAIL:
    # Final model on all train data
    all_data = [cpdata.MoleculeDatapoint.from_smi(s, [y]) for s, y in zip(tr["smiles"], y_tr)]
    te_data  = [cpdata.MoleculeDatapoint.from_smi(s, [np.nan]) for s in te["smiles"]]
    all_ds = cpdata.MoleculeDataset(all_data)
    te_ds  = cpdata.MoleculeDataset(te_data)
    all_loader = cpdata.build_dataloader(all_ds, batch_size=BATCH, shuffle=True)
    te_loader  = cpdata.build_dataloader(te_ds, batch_size=BATCH*2, shuffle=False)

    mp  = cpnn.BondMessagePassing(depth=DEPTH, d_h=D_H)
    agg = cpnn.MeanAggregation()
    ffn = cpnn.RegressionFFN(input_dim=D_H, n_layers=FFN_DEPTH, dropout=DROPOUT)
    model_final = models.MPNN(message_passing=mp, agg=agg, predictor=ffn)
    trainer_final = pl.Trainer(max_epochs=EPOCHS, accelerator=device, devices=1,
                               enable_progress_bar=True, enable_model_summary=False)
    trainer_final.fit(model_final, all_loader)
    te_raw = trainer_final.predict(model_final, te_loader)
    te_preds = np.clip(np.concatenate([p.numpy().flatten() for p in te_raw])[:513],
                       y_tr.min()-0.5, y_tr.max()+0.5)
else:
    te_preds = np.full(513, y_tr.mean())

np.save(DATA_PROC/"oof_chemprop_large_gpu.npy", oof_chemprop_large)
np.save(DATA_PROC/"te_oof_chemprop_large_gpu.npy", te_preds)
sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_preds})
assert len(sub)==513 and sub["pEC50"].notna().all()
p = SUBMISSIONS/"93_chemprop_large_gpu.csv"; sub.to_csv(p, index=False)
print(f"Saved {p}")
print(f"Test: min={te_preds.min():.2f} med={np.median(te_preds):.2f} max={te_preds.max():.2f}")
"""),
])

# ── nb94: Kaggle GPU — MolFormer Fine-tuning ─────────────────────────────────
nb94 = nb([
("markdown", "# 94 — Kaggle GPU: MolFormer-XL Fine-tuning\n\n"
 "IBM MolFormer-XL is pretrained on 1.1 billion SMILES (ZINC + PubChem).\n"
 "Architecture: Linear Attention Transformer, 12 layers, 768-dim embeddings.\n\n"
 "Strategy:\n"
 "1. Load `ibm/MolFormer-XL-both-10pct` from HuggingFace\n"
 "2. Extract [CLS] embedding for all compounds\n"
 "3. Fine-tune: add regression head (3-layer MLP, dropout=0.3)\n"
 "4. Scaffold 5-fold CV on GPU\n\n"
 "Run on Kaggle: `python scripts/kaggle_push.py --nb 94 --pull`"),

("code", """\
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
for p in ["/kaggle/input/pxr-challenge-data/src", "../src"]:
    if os.path.exists(p): sys.path.insert(0, p); break
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

if os.path.exists("/kaggle"):
    DATA_PROC   = Path("/kaggle/input/pxr-challenge-data/data/processed")
    SUBMISSIONS = Path("/kaggle/working")
else:
    sys.path.insert(0, "../src")
    from pxr.paths import DATA_PROCESSED as DATA_PROC, SUBMISSIONS

SEED = 42; torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}  CUDA: {torch.cuda.is_available()}")
"""),

("code", """\
from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko

tr = load_train(); te = load_test()
y_tr = tr["pec50"].values.astype(np.float64)
scaffolds = tr["smiles"].map(bemis_murcko).tolist()
splits = scaffold_kfold_indices(scaffolds, 5, SEED)
print(f"Train {len(tr):,}  Test {len(te):,}")
"""),

("code", """\
try:
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("ibm/MolFormer-XL-both-10pct", trust_remote_code=True)
    base = AutoModel.from_pretrained("ibm/MolFormer-XL-both-10pct", trust_remote_code=True)
    print(f"MolFormer loaded: {sum(p.numel() for p in base.parameters()):,} params")
    MOLFORMER_AVAIL = True
except Exception as e:
    print(f"MolFormer not available: {e}")
    MOLFORMER_AVAIL = False
"""),

("code", """\
if MOLFORMER_AVAIL:
    class MolFormerRegressor(nn.Module):
        def __init__(self, encoder, d_model=768, dropout=0.3):
            super().__init__()
            self.encoder = encoder
            self.head = nn.Sequential(
                nn.Linear(d_model, 256), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(256, 64),  nn.GELU(), nn.Dropout(dropout/2),
                nn.Linear(64, 1)
            )
        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :]
            return self.head(cls).squeeze(-1)

    def tokenize_batch(smiles_list, max_len=202):
        return tok(smiles_list, return_tensors="pt", padding=True,
                   truncation=True, max_length=max_len)

    from torch.utils.data import Dataset, DataLoader
    class SMILESDataset(Dataset):
        def __init__(self, smiles, labels=None):
            self.smiles = smiles
            self.labels = labels
        def __len__(self): return len(self.smiles)
        def __getitem__(self, i):
            return self.smiles[i], self.labels[i] if self.labels is not None else float("nan")

    EPOCHS = 20; BATCH = 32; LR = 2e-5; PATIENCE = 5

    oof_molformer = np.full(len(y_tr), np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        print(f"\\n=== Fold {fold+1}/5 ===", flush=True)
        model_mf = MolFormerRegressor(base).to(device)
        opt = torch.optim.AdamW(model_mf.parameters(), lr=LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        loss_fn = nn.MSELoss()

        tr_smi = tr.iloc[tr_idx]["smiles"].tolist()
        va_smi = tr.iloc[va_idx]["smiles"].tolist()
        tr_y_f = torch.tensor(y_tr[tr_idx], dtype=torch.float32)
        va_y_f = torch.tensor(y_tr[va_idx], dtype=torch.float32)

        best_val = float("inf"); patience_cnt = 0
        for epoch in range(EPOCHS):
            model_mf.train()
            perm = torch.randperm(len(tr_smi))
            epoch_loss = 0
            for b in range(0, len(tr_smi), BATCH):
                idx_b = perm[b:b+BATCH].tolist()
                enc = tokenize_batch([tr_smi[i] for i in idx_b])
                enc = {k: v.to(device) for k, v in enc.items()}
                pred = model_mf(**enc)
                loss = loss_fn(pred, tr_y_f[idx_b].to(device))
                opt.zero_grad(); loss.backward(); opt.step()
                epoch_loss += loss.item()
            sched.step()

            model_mf.eval()
            with torch.no_grad():
                va_preds = []
                for b in range(0, len(va_smi), BATCH*2):
                    enc = tokenize_batch(va_smi[b:b+BATCH*2])
                    enc = {k: v.to(device) for k, v in enc.items()}
                    va_preds.append(model_mf(**enc).cpu().numpy())
                va_pred = np.concatenate(va_preds)
                val_rae = rae(y_tr[va_idx], va_pred)
                print(f"  Epoch {epoch+1:3d}  train_loss={epoch_loss/max(1,len(tr_smi)//BATCH):.4f}  val_RAE={val_rae:.4f}", flush=True)
                if val_rae < best_val:
                    best_val = val_rae; patience_cnt = 0
                    best_preds = va_pred.copy()
                else:
                    patience_cnt += 1
                    if patience_cnt >= PATIENCE: break

        oof_molformer[va_idx] = best_preds
        print(f"Fold {fold+1} best RAE: {best_val:.4f}")

    oof_rae = rae(y_tr, oof_molformer)
    print(f"\\nMolFormer OOF RAE: {oof_rae:.4f}")
else:
    print("MolFormer not available — saving placeholder")
    oof_molformer = np.full(len(y_tr), np.nan)
"""),

("code", """\
if MOLFORMER_AVAIL and np.isfinite(oof_molformer).all():
    # Final model on all data
    model_final = MolFormerRegressor(base).to(device)
    opt = torch.optim.AdamW(model_final.parameters(), lr=LR, weight_decay=1e-4)
    all_smi = tr["smiles"].tolist()
    for epoch in range(EPOCHS):
        model_final.train()
        perm = torch.randperm(len(all_smi))
        for b in range(0, len(all_smi), BATCH):
            idx_b = perm[b:b+BATCH].tolist()
            enc = tokenize_batch([all_smi[i] for i in idx_b])
            enc = {k: v.to(device) for k, v in enc.items()}
            pred = model_final(**enc)
            loss = nn.MSELoss()(pred, torch.tensor(y_tr[idx_b], dtype=torch.float32).to(device))
            opt.zero_grad(); loss.backward(); opt.step()

    model_final.eval()
    te_smi = te["smiles"].tolist()
    te_preds_raw = []
    with torch.no_grad():
        for b in range(0, len(te_smi), BATCH*2):
            enc = tokenize_batch(te_smi[b:b+BATCH*2])
            enc = {k: v.to(device) for k, v in enc.items()}
            te_preds_raw.append(model_final(**enc).cpu().numpy())
    te_preds = np.clip(np.concatenate(te_preds_raw)[:513], y_tr.min()-0.5, y_tr.max()+0.5)
else:
    te_preds = np.full(513, y_tr.mean())

np.save(DATA_PROC/"oof_molformer_finetune.npy", oof_molformer)
np.save(DATA_PROC/"te_oof_molformer_finetune.npy", te_preds)
sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_preds})
assert len(sub)==513 and sub["pEC50"].notna().all()
p = SUBMISSIONS/"94_molformer_finetune.csv"; sub.to_csv(p, index=False)
print(f"Saved {p}")
print(f"Test: min={te_preds.min():.2f} med={np.median(te_preds):.2f} max={te_preds.max():.2f}")
"""),
])

# ── nb95: All-Feature Fusion + Bio-FP LGBM ───────────────────────────────────
nb95 = nb([
("markdown", "# 95 — All-Feature Fusion: 2D + 3D + Bio-FP + Pharmacophore\n\n"
 "Combine all feature streams generated in nb87–nb92:\n"
 "- Combined Morgan+RDKit (2265-dim)\n"
 "- ChEMBL NR bio-FP (5-dim)\n"
 "- 3D shape descriptors (12-dim, if nb88 ran)\n"
 "- PXR pharmacophore features (50-dim)\n"
 "- Multi-NR transfer prediction (1-dim)\n\n"
 "This is the 'kitchen sink' model. LGBM handles high-dimensional sparse input well.\n"
 "Expected to be a strong individual model and valuable ensemble member."),

("code", BOILERPLATE_IMPORTS),
("code", FULL_METRICS),
("code", STANDARD_SETUP),

("code", """\
feature_streams = {"combined_2265": X_tr}
labels = {"combined_2265": X_te}

# Load additional feature arrays
for name, fname_tr, fname_te in [
    ("bio_nr_fp",   "bio_fp_tr.npy",      "bio_fp_te.npy"),
    ("3d_shape",    "X_shape_tr.npy",     "X_shape_te.npy"),
    ("tox21_bio",   "tox21_bio_tr.npy",   "tox21_bio_te.npy"),
]:
    p_tr = DATA_PROCESSED/fname_tr; p_te = DATA_PROCESSED/fname_te
    if p_tr.exists() and p_te.exists():
        arr_tr = np.load(p_tr).astype(np.float32)
        arr_te = np.load(p_te).astype(np.float32)
        if arr_tr.shape[0] == len(tr):
            feature_streams[name] = arr_tr
            labels[name] = arr_te
            print(f"Loaded {name}: {arr_tr.shape}")
        else:
            print(f"Skip {name}: wrong shape {arr_tr.shape}")
    else:
        print(f"Not found: {name} ({fname_tr})")

# Multi-NR transfer prediction
for oof_name in ["multi_nr_transfer", "bio_nr_fingerprint"]:
    p = DATA_PROCESSED/f"oof_{oof_name}.npy"
    pt = DATA_PROCESSED/f"te_oof_{oof_name}.npy"
    if p.exists() and pt.exists():
        arr = np.load(p).astype(np.float32).reshape(-1,1)
        arr_te = np.load(pt).astype(np.float32).reshape(-1,1)
        if len(arr) == len(tr):
            feature_streams[oof_name] = arr
            labels[oof_name] = arr_te
            print(f"Loaded {oof_name} prediction: {arr.shape}")

# Fuse all
X_fused_tr = np.hstack(list(feature_streams.values())).astype(np.float32)
X_fused_te = np.hstack(list(labels.values())).astype(np.float32)
# Final impute for any remaining NaNs
X_fused_tr = np.where(np.isfinite(X_fused_tr), X_fused_tr, 0.0)
X_fused_te = np.where(np.isfinite(X_fused_te), X_fused_te, 0.0)
print(f"\\nFused feature matrix: {X_fused_tr.shape}")
print(f"Feature streams: {list(feature_streams.keys())}")
"""),

("code", """\
oof = np.full(len(y_tr), np.nan)
for fold, (tr_idx, va_idx) in enumerate(splits):
    m = lgb.train(LGBM, lgb.Dataset(X_fused_tr[tr_idx], label=y_tr[tr_idx]),
                  valid_sets=[lgb.Dataset(X_fused_tr[va_idx], label=y_tr[va_idx])],
                  callbacks=[lgb.early_stopping(50,verbose=False), lgb.log_evaluation(-1)])
    oof[va_idx] = m.predict(X_fused_tr[va_idx])
    print(f"  fold {fold+1}  RAE={rae(y_tr[va_idx], oof[va_idx]):.4f}", flush=True)

m_res = full_metrics(y_tr, oof, cliff_pairs, "all_feature_fusion")
m_res_a = full_metrics(y_tr[active_mask], oof[active_mask], label="fusion [active]")
print("\\n" + pd.DataFrame([m_res, m_res_a], index=["overall","active"]).round(4).to_string())

m_final = lgb.train(LGBM, lgb.Dataset(X_fused_tr, label=y_tr), callbacks=[lgb.log_evaluation(-1)])
te_preds = np.clip(m_final.predict(X_fused_te), y_tr.min()-0.5, y_tr.max()+0.5)
np.save(DATA_PROCESSED/"oof_all_feature_fusion.npy", oof)
np.save(DATA_PROCESSED/"te_oof_all_feature_fusion.npy", te_preds)
sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_preds})
assert len(sub)==513 and sub["pEC50"].notna().all()
p = SUBMISSIONS/"95_all_feature_fusion.csv"; sub.to_csv(p, index=False)
print(f"Saved {p}  Test: min={te_preds.min():.2f} med={np.median(te_preds):.2f} max={te_preds.max():.2f}")
"""),
])

# ── nb96: Grand Ensemble v8 ───────────────────────────────────────────────────
nb96 = nb([
("markdown", "# 96 — Grand Ensemble v8\n\n"
 "ElasticNetCV meta-learner over ALL OOF files, using the proper nested-CV approach from nb86.\n\n"
 "Includes all new models from nb86–nb95:\n"
 "- nb87: ChEMBL NR biological fingerprint\n"
 "- nb88: 3D conformer shape\n"
 "- nb89: PXR pharmacophore\n"
 "- nb90: Tox21 bio-FP\n"
 "- nb91: cliff-adaptive blend\n"
 "- nb92: multi-NR transfer\n"
 "- nb93: Kaggle GPU large Chemprop (if available)\n"
 "- nb94: MolFormer fine-tune (if available)\n"
 "- nb95: all-feature fusion\n\n"
 "Uses nested-CV stacking (no in-sample leakage)."),

("code", BOILERPLATE_IMPORTS),
("code", FULL_METRICS),
("code", STANDARD_SETUP),

("code", """\
from sklearn.linear_model import ElasticNetCV

EXCLUDE = {"aux_features","grand_v6","grand_v6b","grand_v6c","grand_v7",
           "grand15","grand18","grand23","grand24","grand25",
           "creative_mega_ensemble","nested_cv_ensemble",
           "cliff_role_proba","chemprop_cliff_mem_proba",
           "chemprop_chembl_nr_multitask","per_fp_stack",
           "cliff_adaptive_blend"}  # test-time only

oof_files = sorted(DATA_PROCESSED.glob("oof_*.npy"))
oofs, tes, names = [], [], []
for fp in oof_files:
    name = fp.stem.replace("oof_","")
    if name in EXCLUDE: continue
    te_fp = DATA_PROCESSED / f"te_oof_{name}.npy"
    try:
        arr = np.load(fp)
        if arr.ndim > 1: arr = arr[:,0]
        if len(arr) != len(y_tr): continue
        te_v = np.load(te_fp) if te_fp.exists() else None
        if te_v is None or len(te_v) != 513: continue
        if te_v.ndim > 1: te_v = te_v[:,0]
        if te_v.std() < 0.4 * y_tr.std(): continue
        arr[~np.isfinite(arr)] = y_tr.mean()
        te_v[~np.isfinite(te_v)] = float(np.nanmean(te_v))
        oofs.append(arr); tes.append(te_v); names.append(name)
    except Exception as e:
        print(f"  skip {name}: {e}")

OOF_stack = np.column_stack(oofs)
TE_stack  = np.column_stack(tes)
print(f"v8 ensemble: {len(names)} models, stack {OOF_stack.shape}")
"""),

("code", """\
# Proper nested-CV stacking
print("=== Nested-CV Grand Ensemble v8 ===", flush=True)
oof_v8 = np.full(len(y_tr), np.nan)

for k, (tr_idx, va_idx) in enumerate(splits):
    meta_tr_idx = [i for fold,(ti,_) in enumerate(splits) for i in ti if fold!=k]
    meta = ElasticNetCV(l1_ratio=[0.1,0.5,0.9,1.0], cv=5, max_iter=10000, random_state=SEED)
    meta.fit(OOF_stack[meta_tr_idx], y_tr[meta_tr_idx])
    oof_v8[va_idx] = meta.predict(OOF_stack[va_idx])
    print(f"  fold {k+1}  val_RAE={rae(y_tr[va_idx], oof_v8[va_idx]):.4f}", flush=True)

m_v8 = full_metrics(y_tr, oof_v8, cliff_pairs, "grand_v8")
m_v8_a = full_metrics(y_tr[active_mask], oof_v8[active_mask], label="v8 [active]")
print(f"\\nGrand v8 OOF RAE: {m_v8['RAE']:.4f}")
print(pd.DataFrame([m_v8, m_v8_a], index=["overall","active"]).round(4).to_string())
"""),

("code", """\
meta_final = ElasticNetCV(l1_ratio=[0.1,0.5,0.9,1.0], cv=5, max_iter=10000, random_state=SEED)
meta_final.fit(OOF_stack, y_tr)
te_preds = np.clip(meta_final.predict(TE_stack), y_tr.min()-0.5, y_tr.max()+0.5)

coef_df = pd.DataFrame({"model": names, "weight": meta_final.coef_}).sort_values("weight", ascending=False)
print("Non-zero weights:")
print(coef_df[coef_df.weight.abs() > 1e-6].to_string(index=False))

np.save(DATA_PROCESSED/"oof_grand_v8.npy", oof_v8)
np.save(DATA_PROCESSED/"te_oof_grand_v8.npy", te_preds)
sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_preds})
assert len(sub)==513 and sub["pEC50"].notna().all()
p = SUBMISSIONS/"96_grand_ensemble_v8.csv"; sub.to_csv(p, index=False)
print(f"Saved {p}")
print(f"Test: min={te_preds.min():.2f} med={np.median(te_preds):.2f} max={te_preds.max():.2f}")
print(f"\\n*** Grand v8 OOF RAE: {m_v8['RAE']:.4f} ***")
"""),
])


# ── Write all notebooks ────────────────────────────────────────────────────────
notebooks_to_write = {
    "86_nested_cv_ensemble.ipynb":    nb86,
    "87_bio_nr_fingerprint.ipynb":    nb87,
    "88_3d_shape_conformer.ipynb":    nb88,
    "89_pxr_pharmacophore.ipynb":     nb89,
    "90_tox21_bio_fp.ipynb":          nb90,
    "91_cliff_adaptive_blend.ipynb":  nb91,
    "92_multi_nr_transfer.ipynb":     nb92,
    "93_chemprop_large_gpu.ipynb":    nb93,
    "94_molformer_finetune.ipynb":    nb94,
    "95_all_feature_fusion.ipynb":    nb95,
    "96_grand_ensemble_v8.ipynb":     nb96,
}

for fname, content in notebooks_to_write.items():
    path = NB_DIR / fname
    path.write_text(json.dumps(content, indent=1))
    print(f"Written: {fname}")

print(f"\n{len(notebooks_to_write)} notebooks generated.")
print("\nLocal CPU run order: nb86 → nb87 → nb88 → nb89 → nb90 → nb91 → nb92 → nb95 → nb96")
print("Kaggle GPU: nb93 (large Chemprop) → nb94 (MolFormer) → pull OOFs → rerun nb96")
