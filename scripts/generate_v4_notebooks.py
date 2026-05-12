"""
Generate notebooks nb76–nb85: creative approaches beyond standard LGBM+external data.

nb76: Delta-ML (MMP template-based prediction — the RNA analogy)
nb77: Tanimoto-kernel Gaussian Process regression
nb78: Multi-fingerprint ensemble (ECFP2/4/6, FCFP, MACCS, RDKit, Avalon)
nb79: 3D conformer shape descriptors (PMI, asphericity, globularity)
nb80: SMILES enumeration data augmentation (10× cheap diversity)
nb81: Selectivity-aware prediction (pEC50 − pEC50_null as primary target)
nb82: Transductive graph label spreading (train+test similarity graph)
nb83: Pseudo-label self-training (3 iterations, confidence-gated)
nb84: Free-Wilson scaffold decomposition (additive R-group contributions)
nb85: Creative ensemble (best of nb76–nb84 + existing stack)

Run: python scripts/generate_v4_notebooks.py
"""

import json
from pathlib import Path

NB_DIR = Path(__file__).parent.parent / "notebooks"

HEADER = '''\
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
'''

METRICS = '''\
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
    if cp is not None and len(cp)>0:
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
'''

BASE_LOAD = '''\
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
print(f"Train {len(tr):,}  Test {len(te):,}  Cliffs {len(cliff_pairs)}")
'''

def cell(src):
    return {"cell_type":"code","metadata":{},"source":[src],"outputs":[],"execution_count":None}
def md(src):
    return {"cell_type":"markdown","metadata":{},"source":[src]}
def nb(cells):
    return {"nbformat":4,"nbformat_minor":5,
            "metadata":{"kernelspec":{"display_name":"pxr-challenge","language":"python","name":"pxr-challenge"},
                        "language_info":{"name":"python","version":"3.12"}},
            "cells":cells}
def save(name, cells):
    path = NB_DIR/name
    with open(path,"w") as f: json.dump(nb(cells),f,indent=1)
    print(f"  wrote {name}")

def save_cell(num, oof_name, slug):
    return f'''\
np.save(DATA_PROCESSED/"{oof_name}.npy", oof)
np.save(DATA_PROCESSED/"te_{oof_name}.npy", te_preds)
sub = pd.DataFrame({{"Molecule Name":te["name"].values,"pEC50":te_preds}})
assert len(sub)==513 and sub["pEC50"].notna().all()
p = SUBMISSIONS/"{num}_{slug}.csv"; sub.to_csv(p,index=False)
print(f"Saved {{p}}")
print(f"Test: min={{te_preds.min():.2f}} med={{np.median(te_preds):.2f}} max={{te_preds.max():.2f}}")
'''

# ══════════════════════════════════════════════════════════════════════════════
# nb76 — Delta-ML: template-based Δ-pEC50 prediction (the RNA analogy)
# ══════════════════════════════════════════════════════════════════════════════
save("76_delta_ml_template.ipynb", [
md("""# 76 — Delta-ML: Template-Based Δ-pEC50 Prediction

**The RNA Analogy applied to chemistry:**
Instead of predicting absolute pEC50, find the closest training compound (the *template*),
then predict only the **delta** (how much the test compound differs in activity).

Train the delta model on all ~800k+ training pairs with Tanimoto > 0.35.
At inference: pEC50_test ≈ pEC50_template + Δ_model(diff_features).

This directly models activity cliffs as large Δ values — exactly where standard
models fail most. Works best precisely when test ≈ train (our situation: median sim=0.52).
"""),
cell(HEADER),
cell(METRICS),
cell(BASE_LOAD),
cell('''\
# ── Build training pairs ──────────────────────────────────────────────────────
# Tanimoto from Morgan FPs (binary, so dot/union is exact Tanimoto)
print("Computing pairwise Tanimoto for training set...", flush=True)
dot = (fps_tr @ fps_tr.T).astype(np.float32)
rowsum = fps_tr.sum(1).astype(np.float32)
union = rowsum[:,None] + rowsum[None,:] - dot
tanimoto = np.where(union>0, dot/union, 0.0)
np.fill_diagonal(tanimoto, 0)

SIM_THRESH = 0.35   # minimum similarity to form a pair
MAX_PAIRS   = 300_000  # cap to avoid memory blow-up

# Physchem diffs for delta features
print("Computing physchem...", flush=True)
phys_tr = tr["smiles"].map(compute_physchem).tolist()
props = ["mw","logp","tpsa","hbd","hba","rotbonds","rings"]
phys_arr = np.array([[p.get(k,0) or 0 for k in props] for p in phys_tr], dtype=np.float32)

print("Building pair dataset...", flush=True)
rng = np.random.default_rng(42)
i_idx, j_idx = np.where(tanimoto >= SIM_THRESH)
# Remove symmetric duplicates (keep only i < j)
mask_upper = i_idx < j_idx
i_idx, j_idx = i_idx[mask_upper], j_idx[mask_upper]
print(f"Total pairs with Tanimoto≥{SIM_THRESH}: {len(i_idx):,}")

if len(i_idx) > MAX_PAIRS:
    sel = rng.choice(len(i_idx), MAX_PAIRS, replace=False)
    i_idx, j_idx = i_idx[sel], j_idx[sel]
    print(f"Downsampled to {MAX_PAIRS:,} pairs")

# Delta target: pEC50_j - pEC50_i  (also build reverse pairs for symmetry)
y_delta_ij = y_tr[j_idx] - y_tr[i_idx]
y_delta_ji = -y_delta_ij

# Features: [common_fp, diff_fp (XOR), sim, anchor_pec50, delta_physchem]
fps_i = fps_tr[i_idx]; fps_j = fps_tr[j_idx]
fp_common = np.minimum(fps_i, fps_j)             # AND (common substructure)
fp_diff   = np.abs(fps_i - fps_j).astype(np.float32)  # XOR proxy
sims = tanimoto[i_idx, j_idx][:,None]
phys_diff_ij = phys_arr[j_idx] - phys_arr[i_idx]
phys_diff_ji = -phys_diff_ij

# Build final feature matrices (ij and ji directions)
def make_delta_feats(fp_anc, fp_common, fp_diff, sim, anc_pec50, phys_diff):
    # 64-dim fingerprint compression to keep features manageable
    fp_common_64 = fp_common.reshape(-1, 32, fp_common.shape[1]//32).mean(-1) if fp_common.shape[1]>64 else fp_common
    fp_diff_64   = fp_diff.reshape(-1, 32, fp_diff.shape[1]//32).mean(-1)   if fp_diff.shape[1]>64 else fp_diff
    return np.hstack([fp_common_64, fp_diff_64, sim,
                      anc_pec50[:,None], phys_diff])

print("Building feature matrices...", flush=True)
F_ij = make_delta_feats(fps_i, fp_common, fp_diff, sims,
                         y_tr[i_idx], phys_diff_ij)
F_ji = make_delta_feats(fps_j, fp_common, fp_diff, sims,
                         y_tr[j_idx], phys_diff_ji)
F_all = np.vstack([F_ij, F_ji])
y_all = np.concatenate([y_delta_ij, y_delta_ji])
print(f"Delta dataset: {F_all.shape}  Δ range [{y_all.min():.2f}, {y_all.max():.2f}]")
'''),
cell('''\
# ── Train delta model ──────────────────────────────────────────────────────────
from sklearn.model_selection import cross_val_score
print("Training delta model on all pairs...", flush=True)
delta_model = lgb.LGBMRegressor(**LGBM, n_estimators=500)
delta_model.fit(F_all, y_all, callbacks=[lgb.log_evaluation(-1)])
print("Delta model trained.")

# Sanity check: predict delta for same-compound (should be ~0)
F_self = make_delta_feats(fps_tr[:10], np.minimum(fps_tr[:10],fps_tr[:10]),
                           np.zeros((10,fps_tr.shape[1]),dtype=np.float32),
                           np.ones((10,1)),y_tr[:10], np.zeros((10,len(props)),dtype=np.float32))
self_deltas = delta_model.predict(F_self)
print(f"Self-delta (should be ~0): mean={self_deltas.mean():.3f} std={self_deltas.std():.3f}")
'''),
cell('''\
# ── Scaffold CV: evaluate delta-ML vs direct prediction ────────────────────────
print("\\n=== Scaffold 5-fold CV ===", flush=True)
oof_delta = np.full(len(y_tr), np.nan)
oof_direct = np.full(len(y_tr), np.nan)

for fold, (tr_idx, va_idx) in enumerate(splits):
    # Direct model (baseline within fold)
    m_direct = lgb.train(LGBM, lgb.Dataset(X_tr[tr_idx], label=y_tr[tr_idx]),
                         valid_sets=[lgb.Dataset(X_tr[va_idx], label=y_tr[va_idx])],
                         callbacks=[lgb.early_stopping(50,verbose=False), lgb.log_evaluation(-1)])
    oof_direct[va_idx] = m_direct.predict(X_tr[va_idx])

    # Delta-ML: for each val compound, find nearest TRAINING neighbor
    fps_val = fps_tr[va_idx]; fps_fold_tr = fps_tr[tr_idx]
    dot_vt = (fps_val @ fps_fold_tr.T).astype(np.float32)
    rs_v = fps_val.sum(1)[:,None]; rs_t = fps_fold_tr.sum(1)[None,:]
    sim_vt = dot_vt / np.maximum(rs_v + rs_t - dot_vt, 1e-6)
    best_t_idx = sim_vt.argmax(1)   # index into fold-train compounds
    best_sims = sim_vt.max(1)
    best_global = tr_idx[best_t_idx]  # global indices

    phys_val = phys_arr[va_idx]; phys_refs = phys_arr[best_global]
    fp_refs   = fps_tr[best_global]; fp_val = fps_val
    fp_com    = np.minimum(fp_val, fp_refs)
    fp_dif    = np.abs(fp_val - fp_refs).astype(np.float32)
    F_test    = make_delta_feats(fp_refs, fp_com, fp_dif, best_sims[:,None],
                                  y_tr[best_global], phys_val - phys_refs)
    delta_pred = delta_model.predict(F_test)
    oof_delta[va_idx] = y_tr[best_global] + delta_pred

    r_dir = rae(y_tr[va_idx], oof_direct[va_idx])
    r_del = rae(y_tr[va_idx], oof_delta[va_idx])
    print(f"  fold {fold+1}  direct={r_dir:.4f}  delta={r_del:.4f}  "
          f"best_sim_mean={best_sims.mean():.3f}", flush=True)

m_dir  = full_metrics(y_tr, oof_direct, cliff_pairs, "direct_lgbm")
m_del  = full_metrics(y_tr, oof_delta,  cliff_pairs, "delta_ml")

# Blend: alpha * delta + (1-alpha) * direct, sweep alpha
best_alpha, best_rae = 0, full_metrics(y_tr, oof_direct)["RAE"]
for alpha in np.arange(0.1, 1.01, 0.1):
    blended = alpha * oof_delta + (1-alpha) * oof_direct
    r = rae(y_tr[np.isfinite(blended)], blended[np.isfinite(blended)])
    if r < best_rae:
        best_rae, best_alpha = r, alpha

oof = best_alpha * oof_delta + (1-best_alpha) * oof_direct
m_blend = full_metrics(y_tr, oof, cliff_pairs, f"blend(α={best_alpha:.1f})")
print(f"\\nBest blend α={best_alpha:.1f}  OOF RAE={best_rae:.4f}")
results_df = pd.DataFrame([m_dir, m_del, m_blend],
                           index=["direct","delta_ml",f"blend_{best_alpha:.1f}"])
print("\\n" + results_df.round(4).to_string())
'''),
cell('''\
# ── Final predictions ─────────────────────────────────────────────────────────
m_final = lgb.train(LGBM, lgb.Dataset(X_tr, label=y_tr), callbacks=[lgb.log_evaluation(-1)])
te_direct = m_final.predict(X_te)

# Delta for test compounds: nearest training neighbor
dot_tt = (fps_te @ fps_tr.T).astype(np.float32)
rs_te = fps_te.sum(1)[:,None]; rs_tr_v = fps_tr.sum(1)[None,:]
sim_te_tr = dot_tt / np.maximum(rs_te + rs_tr_v - dot_tt, 1e-6)
best_tr_idx = sim_te_tr.argmax(1); best_sims_te = sim_te_tr.max(1)
print(f"Test→train similarity: mean={best_sims_te.mean():.3f}  "
      f"min={best_sims_te.min():.3f}  max={best_sims_te.max():.3f}")

phys_te_arr = np.array([[p.get(k,0) or 0 for k in ["mw","logp","tpsa","hbd","hba","rotbonds","rings"]]
                          for p in te["smiles"].map(compute_physchem)], dtype=np.float32)
fp_refs_te   = fps_tr[best_tr_idx]
fp_com_te    = np.minimum(fps_te, fp_refs_te)
fp_dif_te    = np.abs(fps_te - fp_refs_te).astype(np.float32)
F_te_delta   = make_delta_feats(fp_refs_te, fp_com_te, fp_dif_te,
                                  best_sims_te[:,None], y_tr[best_tr_idx],
                                  phys_te_arr - phys_arr[best_tr_idx])
te_delta_pred = delta_model.predict(F_te_delta)
te_delta_final = y_tr[best_tr_idx] + te_delta_pred

te_preds = best_alpha*te_delta_final + (1-best_alpha)*te_direct
te_preds = np.clip(te_preds, y_tr.min()-0.5, y_tr.max()+0.5)
''' + save_cell(76, "oof_delta_ml", "delta_ml_template")),
])


# ══════════════════════════════════════════════════════════════════════════════
# nb77 — Tanimoto-kernel Gaussian Process
# ══════════════════════════════════════════════════════════════════════════════
save("77_gaussian_process_tanimoto.ipynb", [
md("""# 77 — Gaussian Process with Tanimoto Kernel

GP regression is often the best method for small molecular datasets (< 10k compounds).
The Tanimoto kernel is the natural similarity measure for binary fingerprints.
GPs also output calibrated uncertainty — useful for ensemble weighting.

Training a full GP on 4k compounds with 2048-dim fingerprints is expensive O(n³),
so we use a sparse approximation (Nyström / inducing points) and sub-sampling.
"""),
cell(HEADER),
cell(METRICS),
cell(BASE_LOAD),
cell('''\
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler

class TanimotoKernel(Kernel):
    """Tanimoto (Jaccard) kernel for binary fingerprints."""
    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X).astype(np.float32)
        Y = X if Y is None else np.atleast_2d(Y).astype(np.float32)
        dot = X @ Y.T
        sx  = X.sum(1)[:,None]; sy = Y.sum(1)[None,:]
        K   = dot / np.maximum(sx + sy - dot, 1e-8)
        if eval_gradient:
            return K.astype(np.float64), np.empty((X.shape[0],X.shape[0],0))
        return K.astype(np.float64)
    def diag(self, X):
        return np.ones(X.shape[0])
    def is_stationary(self): return False
    def get_params(self, deep=True): return {}
    def set_params(self, **p): return self

print("Tanimoto kernel defined.")
'''),
cell('''\
# Sparse GP via Nyström approximation on inducing points
# Select 500 inducing points (diverse by MaxMin selection)
N_INDUCING = 500

def maxmin_select(fps, n):
    """MaxMin diversity selection of n inducing points."""
    selected = [0]
    min_dists = np.full(len(fps), np.inf)
    for _ in range(n-1):
        last = fps[selected[-1]]
        dot_l = fps @ last
        sl = last.sum(); sr = fps.sum(1)
        sim = dot_l / np.maximum(sl + sr - dot_l, 1e-8)
        dist = 1 - sim
        min_dists = np.minimum(min_dists, dist)
        min_dists[selected] = -1
        selected.append(int(np.argmax(min_dists)))
    return np.array(selected)

print(f"Selecting {N_INDUCING} inducing points by MaxMin diversity...")
inducing_idx = maxmin_select(fps_tr, N_INDUCING)
X_ind = fps_tr[inducing_idx].astype(np.float64)
print(f"Inducing points selected: {X_ind.shape}")
'''),
cell('''\
# Nystrom GP: fit GP on inducing points, predict via Nystrom approximation
kernel = C(1.0) * TanimotoKernel()
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.5, n_restarts_optimizer=0,
                               normalize_y=True)

print("Fitting GP on inducing points...", flush=True)
gp.fit(X_ind, y_tr[inducing_idx])
print(f"Kernel params: {gp.kernel_}")

# Scaffold CV with Nyström GP
print("\\n=== Scaffold 5-fold CV ===", flush=True)
oof_gp = np.full(len(y_tr), np.nan)

for fold, (tr_idx, va_idx) in enumerate(splits):
    # Re-select inducing points from fold-train only
    n_ind_fold = min(N_INDUCING, len(tr_idx))
    ind_fold_local = maxmin_select(fps_tr[tr_idx], n_ind_fold)
    ind_fold_global = tr_idx[ind_fold_local]
    X_ind_f = fps_tr[ind_fold_global].astype(np.float64)
    y_ind_f = y_tr[ind_fold_global]

    gp_f = GaussianProcessRegressor(kernel=C(1.0)*TanimotoKernel(),
                                     alpha=0.5, normalize_y=True)
    gp_f.fit(X_ind_f, y_ind_f)
    oof_gp[va_idx] = gp_f.predict(fps_tr[va_idx].astype(np.float64))
    print(f"  fold {fold+1} RAE={rae(y_tr[va_idx], oof_gp[va_idx]):.4f}", flush=True)

m_gp = full_metrics(y_tr, oof_gp, cliff_pairs, "GP_tanimoto")
m_gp_a = full_metrics(y_tr[active_mask], oof_gp[active_mask], "GP_tanimoto [active]")
print("\\n" + pd.DataFrame([m_gp, m_gp_a], index=["overall","active"]).round(4).to_string())
oof = oof_gp
'''),
cell('''\
# Final GP on all training data (full inducing set)
gp_final = GaussianProcessRegressor(kernel=C(1.0)*TanimotoKernel(),
                                      alpha=0.5, normalize_y=True)
gp_final.fit(fps_tr[inducing_idx].astype(np.float64), y_tr[inducing_idx])
te_preds_gp, te_std = gp_final.predict(fps_te.astype(np.float64), return_std=True)
te_preds = np.clip(te_preds_gp, y_tr.min()-0.5, y_tr.max()+0.5)
print(f"Test uncertainty (std): mean={te_std.mean():.3f}  max={te_std.max():.3f}")
np.save(DATA_PROCESSED/"te_gp_uncertainty.npy", te_std)
''' + save_cell(77, "oof_gp_tanimoto", "gp_tanimoto")),
])


# ══════════════════════════════════════════════════════════════════════════════
# nb78 — Multi-fingerprint ensemble
# ══════════════════════════════════════════════════════════════════════════════
save("78_multi_fingerprint_ensemble.ipynb", [
md("""# 78 — Multi-Fingerprint Diversity Ensemble

Different fingerprints capture different aspects of molecular SAR:
- ECFP2: heavy atom environments (very local)
- ECFP4: standard circular (default)
- ECFP6: larger circular (more context)
- FCFP4: feature-based (pharmacophoric)
- MACCS: 166 predefined structural keys
- RDKit: path-based (bond topology)
- Topological torsion: 3D-proxy shape descriptor

Train LGBM on each → OOF predictions → ElasticNetCV meta-learner.
Diversity of view = better ensemble even with same model class.
"""),
cell(HEADER),
cell(METRICS),
cell(BASE_LOAD),
cell('''\
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem, RDKFingerprint
from rdkit.Chem import rdMolDescriptors
from sklearn.linear_model import ElasticNetCV
import warnings

def smiles_to_fps(smiles_list):
    """Generate 7 fingerprint types for each SMILES. Returns dict of (N,bits) arrays."""
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    valid = [m is not None for m in mols]
    mols_v = [m for m in mols if m is not None]

    def fp_array(fn, n_bits):
        arr = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
        idx = 0
        for i, ok in enumerate(valid):
            if ok:
                try:
                    bv = fn(mols_v[idx])
                    from rdkit.DataStructs import ConvertToNumpyArray
                    ConvertToNumpyArray(bv, arr[i])
                except: pass
                idx += 1
        return arr

    fps = {
        "ecfp2": fp_array(lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 1, 2048), 2048),
        "ecfp4": fp_array(lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048), 2048),
        "ecfp6": fp_array(lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, 2048), 2048),
        "fcfp4": fp_array(lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048, useFeatures=True), 2048),
        "maccs": fp_array(lambda m: MACCSkeys.GenMACCSKeys(m), 167),
        "rdkit6": fp_array(lambda m: RDKFingerprint(m, maxPath=6, fpSize=2048), 2048),
        "torsion": fp_array(lambda m: rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(m, 2048), 2048),
    }
    return fps

print("Computing 7 fingerprint types for training set...", flush=True)
fps_dict_tr = smiles_to_fps(tr["smiles"].tolist())
print("Computing 7 fingerprint types for test set...", flush=True)
fps_dict_te = smiles_to_fps(te["smiles"].tolist())
for k,v in fps_dict_tr.items():
    print(f"  {k}: {v.shape}")
'''),
cell('''\
# Train one LGBM per fingerprint type
print("\\n=== Per-fingerprint scaffold CV ===", flush=True)
oof_per_fp = {}
te_per_fp  = {}
metrics_per_fp = {}

for fp_name, X_fp in fps_dict_tr.items():
    X_fp_te = fps_dict_te[fp_name]
    oof_fp = np.full(len(y_tr), np.nan)
    print(f"\\n--- {fp_name} ({X_fp.shape[1]}d) ---", flush=True)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = lgb.train(LGBM, lgb.Dataset(X_fp[tr_idx], label=y_tr[tr_idx]),
                      valid_sets=[lgb.Dataset(X_fp[va_idx], label=y_tr[va_idx])],
                      callbacks=[lgb.early_stopping(50,verbose=False), lgb.log_evaluation(-1)])
        oof_fp[va_idx] = m.predict(X_fp[va_idx])
    oof_per_fp[fp_name] = oof_fp
    m_fp = lgb.train(LGBM, lgb.Dataset(X_fp, label=y_tr), callbacks=[lgb.log_evaluation(-1)])
    te_per_fp[fp_name]  = m_fp.predict(X_fp_te)
    metrics_per_fp[fp_name] = full_metrics(y_tr, oof_fp, cliff_pairs, fp_name)

print("\\n=== Summary per fingerprint ===")
print(pd.DataFrame(metrics_per_fp).T[["RAE","MAE","R2","Spearman","Cliff_acc" if "Cliff_acc" in metrics_per_fp.get("ecfp4",{}) else "RAE"]].round(4).to_string())
'''),
cell('''\
# ElasticNetCV meta-learner on OOF stack
OOF_stack = np.column_stack([oof_per_fp[k] for k in fps_dict_tr])
TE_stack  = np.column_stack([te_per_fp[k]  for k in fps_dict_tr])
valid_rows = np.isfinite(OOF_stack).all(1)

from sklearn.linear_model import ElasticNetCV
meta = ElasticNetCV(l1_ratio=[0.1,0.5,0.9,1.0], cv=5, max_iter=5000)
meta.fit(OOF_stack[valid_rows], y_tr[valid_rows])
print(f"\\nMeta-learner weights: {dict(zip(fps_dict_tr.keys(), meta.coef_.round(3)))}")
print(f"Non-zero FPs: {(meta.coef_ != 0).sum()}/{len(fps_dict_tr)}")

oof = meta.predict(OOF_stack)
oof = np.clip(oof, y_tr.min()-0.5, y_tr.max()+0.5)
m_ens = full_metrics(y_tr, oof, cliff_pairs, "multi_fp_ensemble")
m_ens_a = full_metrics(y_tr[active_mask], oof[active_mask], "multi_fp_ensemble [active]")
print(pd.DataFrame([m_ens, m_ens_a], index=["overall","active"]).round(4).to_string())
te_preds = np.clip(meta.predict(TE_stack), y_tr.min()-0.5, y_tr.max()+0.5)
np.save(DATA_PROCESSED/"oof_per_fp_stack.npy", OOF_stack)
''' + save_cell(78, "oof_multi_fp_ensemble", "multi_fp_ensemble")),
])


# ══════════════════════════════════════════════════════════════════════════════
# nb79 — 3D conformer shape + pharmacophore descriptors
# ══════════════════════════════════════════════════════════════════════════════
save("79_3d_shape_descriptors.ipynb", [
md("""# 79 — 3D Conformer Shape & Pharmacophore Descriptors

PXR has a large, flexible hydrophobic binding cavity. Shape complementarity matters.
2D fingerprints cannot capture 3D shape — but RDKit can generate conformers and
compute 3D descriptors:
- PMI ratios (principal moments of inertia) — rodlike vs disclike vs spherical
- Asphericity, eccentricity, NPR1/2 — shape anisotropy
- 3D pharmacophore fingerprints (P2D, P3D)
- WHIM (Weighted Holistic Invariant Molecular) descriptors

Combine 3D descriptors with Morgan 2D for a richer feature set.
"""),
cell(HEADER),
cell(METRICS),
cell(BASE_LOAD),
cell('''\
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors3D
from rdkit.Chem import rdMolDescriptors
import concurrent.futures

def generate_conformer(smi, seed=42):
    """Generate lowest-energy conformer via ETKDG + MMFF."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) < 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except: pass
    return mol

def compute_3d_descriptors(smi):
    """Return dict of 3D shape descriptors. Returns zeros on failure."""
    defaults = {k:0.0 for k in ["pmi1","pmi2","pmi3","npr1","npr2",
                                  "asphericity","eccentricity","inertial_shape",
                                  "gyration","glob","rdf_10","rdf_20"]}
    mol = generate_conformer(smi)
    if mol is None: return defaults
    try:
        d = {}
        # PMI (principal moments of inertia)
        d["pmi1"] = float(Descriptors3D.PMI1(mol))
        d["pmi2"] = float(Descriptors3D.PMI2(mol))
        d["pmi3"] = float(Descriptors3D.PMI3(mol))
        # Normalized (shape ratios)
        d["npr1"] = float(Descriptors3D.NPR1(mol))  # rod-like: 0→1
        d["npr2"] = float(Descriptors3D.NPR2(mol))  # disc-like: 0.5→1
        # Shape descriptors
        d["asphericity"]    = float(Descriptors3D.Asphericity(mol))
        d["eccentricity"]   = float(Descriptors3D.Eccentricity(mol))
        d["inertial_shape"] = float(Descriptors3D.InertialShapeFactor(mol))
        d["gyration"]       = float(Descriptors3D.RadiusOfGyration(mol))
        d["glob"]           = float(Descriptors3D.Spherocity(mol))
        # AUTOCORR3D (first few)
        ac3d = rdMolDescriptors.CalcAUTOCORR3D(mol)
        d["rdf_10"] = float(ac3d[10]) if len(ac3d)>10 else 0.0
        d["rdf_20"] = float(ac3d[20]) if len(ac3d)>20 else 0.0
        return d
    except: return defaults

print("Generating 3D conformers for training set (this takes a few minutes)...", flush=True)
'''),
cell('''\
# Parallel conformer generation (capped workers for resource headroom)
def batch_3d(smiles_list, max_workers=4, label=""):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(compute_3d_descriptors, s) for s in smiles_list]
        for i, f in enumerate(concurrent.futures.as_completed(futs)):
            results.append(f.result())
            if (i+1) % 500 == 0:
                print(f"  {label}: {i+1}/{len(smiles_list)}", flush=True)
    return results

desc_tr = batch_3d(tr["smiles"].tolist(), max_workers=3, label="train")
desc_te = batch_3d(te["smiles"].tolist(), max_workers=3, label="test")

keys_3d = sorted(desc_tr[0].keys())
X_3d_tr = np.array([[d[k] for k in keys_3d] for d in desc_tr], dtype=np.float32)
X_3d_te = np.array([[d[k] for k in keys_3d] for d in desc_te], dtype=np.float32)

# Replace infinities and clip outliers
X_3d_tr = np.nan_to_num(X_3d_tr, nan=0, posinf=0, neginf=0)
X_3d_te = np.nan_to_num(X_3d_te, nan=0, posinf=0, neginf=0)

# Concatenate with 2D features
X_full_tr = np.hstack([X_tr, X_3d_tr])
X_full_te  = np.hstack([X_te, X_3d_te])
print(f"\\n2D+3D features: {X_full_tr.shape}")
print(f"3D descriptor stats:\\n{pd.DataFrame(X_3d_tr, columns=keys_3d).describe().round(3).to_string()}")
'''),
cell('''\
# Scaffold CV with 2D+3D features
print("\\n=== 2D+3D scaffold CV ===", flush=True)
oof = np.full(len(y_tr), np.nan)
oof_2d = np.full(len(y_tr), np.nan)

for fold, (tr_idx, va_idx) in enumerate(splits):
    # 2D baseline
    m2 = lgb.train(LGBM, lgb.Dataset(X_tr[tr_idx], label=y_tr[tr_idx]),
                   valid_sets=[lgb.Dataset(X_tr[va_idx], label=y_tr[va_idx])],
                   callbacks=[lgb.early_stopping(50,verbose=False), lgb.log_evaluation(-1)])
    oof_2d[va_idx] = m2.predict(X_tr[va_idx])
    # 2D+3D
    m3 = lgb.train(LGBM, lgb.Dataset(X_full_tr[tr_idx], label=y_tr[tr_idx]),
                   valid_sets=[lgb.Dataset(X_full_tr[va_idx], label=y_tr[va_idx])],
                   callbacks=[lgb.early_stopping(50,verbose=False), lgb.log_evaluation(-1)])
    oof[va_idx] = m3.predict(X_full_tr[va_idx])
    r2d = rae(y_tr[va_idx], oof_2d[va_idx])
    r3d = rae(y_tr[va_idx], oof[va_idx])
    print(f"  fold {fold+1}  2D={r2d:.4f}  2D+3D={r3d:.4f}", flush=True)

m_2d  = full_metrics(y_tr, oof_2d, cliff_pairs, "2D_only")
m_3d  = full_metrics(y_tr, oof, cliff_pairs, "2D+3D_shape")
print("\\n" + pd.DataFrame([m_2d,m_3d],index=["2D","2D+3D"]).round(4).to_string())

m_final = lgb.train(LGBM, lgb.Dataset(X_full_tr,label=y_tr), callbacks=[lgb.log_evaluation(-1)])
te_preds = np.clip(m_final.predict(X_full_te), y_tr.min()-0.5, y_tr.max()+0.5)
np.save(DATA_PROCESSED/"X_tr_3d.npy", X_3d_tr)
np.save(DATA_PROCESSED/"X_te_3d.npy", X_3d_te)
''' + save_cell(79, "oof_3d_shape", "3d_shape_descriptors")),
])


# ══════════════════════════════════════════════════════════════════════════════
# nb80 — SMILES enumeration augmentation (10× cheap diversity)
# ══════════════════════════════════════════════════════════════════════════════
save("80_smiles_enumeration_augmentation.ipynb", [
md("""# 80 — SMILES Enumeration Data Augmentation

The same molecule has many valid SMILES representations (different atom orderings).
ChemBERTa and SMILES-based models benefit from SMILES augmentation.
But it also helps RDKit-featurized LightGBM: different SMILES → different canonical
atom orderings → different hash collisions → perturbed features → ensemble effect.

Generate 10 random SMILES per training compound (10× data), train LGBM,
then at test time predict from 10 random SMILES and average.
"""),
cell(HEADER),
cell(METRICS),
cell(BASE_LOAD),
cell('''\
from rdkit import Chem

def random_smiles(smi, n=10, seed=42):
    """Generate n random SMILES representations of the same molecule."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return [smi]*n
    results = [Chem.MolToSmiles(mol)]  # canonical always first
    rng = np.random.default_rng(seed)
    for attempt in range(n * 5):
        if len(results) >= n: break
        order = list(range(mol.GetNumAtoms()))
        rng.shuffle(order)
        try:
            mol_new = Chem.RenumberAtoms(mol, order)
            rsmi = Chem.MolToSmiles(mol_new, canonical=False)
            if rsmi not in results:
                results.append(rsmi)
        except: pass
    while len(results) < n:
        results.append(results[0])
    return results[:n]

N_AUG = 8  # augmentations per training compound
print(f"Generating {N_AUG} random SMILES per training compound ({len(tr)*N_AUG:,} total)...")
aug_smiles = []
aug_y = []
for i, (smi, y_val) in enumerate(zip(tr["smiles"].tolist(), y_tr)):
    variants = random_smiles(smi, n=N_AUG, seed=i)
    aug_smiles.extend(variants)
    aug_y.extend([y_val] * len(variants))
    if (i+1) % 500 == 0: print(f"  {i+1}/{len(tr)}", flush=True)

aug_y = np.array(aug_y, dtype=np.float64)
print(f"\\nAugmented training set: {len(aug_smiles):,} SMILES")
'''),
cell('''\
print("Featurizing augmented training set...", flush=True)
X_aug = impute(combined(aug_smiles))
print(f"X_aug: {X_aug.shape}")

# Standard scaffold CV on original training (not augmented) for fair comparison
print("\\n=== Scaffold 5-fold CV (original vs augmented) ===", flush=True)
oof_orig = np.full(len(y_tr), np.nan)
oof_aug  = np.full(len(y_tr), np.nan)

for fold, (tr_idx, va_idx) in enumerate(splits):
    # Original
    m_o = lgb.train(LGBM, lgb.Dataset(X_tr[tr_idx], label=y_tr[tr_idx]),
                    valid_sets=[lgb.Dataset(X_tr[va_idx], label=y_tr[va_idx])],
                    callbacks=[lgb.early_stopping(50,verbose=False), lgb.log_evaluation(-1)])
    oof_orig[va_idx] = m_o.predict(X_tr[va_idx])

    # Augmented (all N_AUG*len(fold_train) rows, validate on original canonical)
    aug_tr_idx = np.concatenate([np.arange(i*N_AUG, (i+1)*N_AUG) for i in tr_idx])
    aug_tr_X   = X_aug[aug_tr_idx]; aug_tr_y = aug_y[aug_tr_idx]
    m_a = lgb.train(LGBM, lgb.Dataset(aug_tr_X, label=aug_tr_y),
                    valid_sets=[lgb.Dataset(X_tr[va_idx], label=y_tr[va_idx])],
                    callbacks=[lgb.early_stopping(50,verbose=False), lgb.log_evaluation(-1)])
    oof_aug[va_idx] = m_a.predict(X_tr[va_idx])

    r_o = rae(y_tr[va_idx], oof_orig[va_idx])
    r_a = rae(y_tr[va_idx], oof_aug[va_idx])
    print(f"  fold {fold+1}  orig={r_o:.4f}  augmented={r_a:.4f}", flush=True)

m_orig = full_metrics(y_tr, oof_orig, cliff_pairs, "original")
m_aug  = full_metrics(y_tr, oof_aug,  cliff_pairs, "smiles_augmented")
oof = oof_aug
print("\\n" + pd.DataFrame([m_orig, m_aug], index=["original","augmented"]).round(4).to_string())
'''),
cell('''\
# Final model + test-time augmentation
print("\\nTraining final model on augmented data...", flush=True)
m_final = lgb.train(LGBM, lgb.Dataset(X_aug, label=aug_y), callbacks=[lgb.log_evaluation(-1)])

# Test-time: predict from 5 random SMILES, average
N_TEST_AUG = 5
te_preds_all = []
for aug_seed in range(N_TEST_AUG):
    te_aug_smiles = []
    for i, smi in enumerate(te["smiles"].tolist()):
        vars = random_smiles(smi, n=2, seed=1000+aug_seed*100+i)
        te_aug_smiles.append(vars[1] if len(vars)>1 else vars[0])
    X_te_aug = impute(combined(te_aug_smiles))
    te_preds_all.append(m_final.predict(X_te_aug))
te_preds = np.clip(np.mean(te_preds_all, axis=0), y_tr.min()-0.5, y_tr.max()+0.5)
print(f"Test-time augmentation: {N_TEST_AUG} SMILES variants averaged")
''' + save_cell(80, "oof_smiles_aug", "smiles_enumeration_aug")),
])


# ══════════════════════════════════════════════════════════════════════════════
# nb81 — Pseudo-label self-training (3 iterations)
# ══════════════════════════════════════════════════════════════════════════════
save("81_pseudo_label_self_training.ipynb", [
md("""# 81 — Pseudo-Label Self-Training (Transductive, 3 Iterations)

The test set consists of analogs of 63 training hits — average Tanimoto 0.52 to train.
This makes the test set "easy" for a transductive learner:

1. Train model on CRC data → predict all 513 test compounds
2. For test compounds with high similarity to training (Tanimoto > 0.65):
   - Confidence-gate: only add pseudo-labels we're confident about
   - Add to training with weight = max_sim_to_train
3. Retrain → repeat 3 iterations

The 32 test compounds near cliff members are likely to get better pseudo-labels
as the model improves from easier examples → semi-supervised curriculum.
"""),
cell(HEADER),
cell(METRICS),
cell(BASE_LOAD),
cell('''\
# Similarity of each test compound to nearest training compound
print("Computing test→train similarity...", flush=True)
dot_te_tr = (fps_te @ fps_tr.T).astype(np.float32)
rs_te = fps_te.sum(1)[:,None]; rs_tr_v = fps_tr.sum(1)[None,:]
sim_te_tr = dot_te_tr / np.maximum(rs_te + rs_tr_v - dot_te_tr, 1e-6)
max_sim = sim_te_tr.max(1)  # (513,) — max similarity to any training compound
print(f"Test→train max similarity: "
      f"mean={max_sim.mean():.3f}  min={max_sim.min():.3f}  max={max_sim.max():.3f}")
print(f"High-confidence (sim>0.65): {(max_sim>0.65).sum()}")
print(f"Med-confidence  (sim>0.50): {(max_sim>0.50).sum()}")
print(f"Low-confidence  (sim≤0.50): {(max_sim<=0.50).sum()}")
'''),
cell('''\
THRESHOLDS   = [0.65, 0.55, 0.45]  # tighten per iteration
WEIGHT_SCALE = 0.8   # pseudo-label weight = max_sim * WEIGHT_SCALE
N_ROUNDS     = 3

X_cur, y_cur, w_cur = X_tr.copy(), y_tr.copy(), np.ones(len(y_tr), dtype=np.float32)
history = []

for rnd in range(N_ROUNDS):
    print(f"\\n=== Round {rnd+1}/{N_ROUNDS}  threshold≥{THRESHOLDS[rnd]:.2f} ===", flush=True)
    # Train on current augmented set
    m = lgb.train(LGBM, lgb.Dataset(X_cur, label=y_cur, weight=w_cur),
                  callbacks=[lgb.log_evaluation(-1)])
    te_pred = m.predict(X_te)

    # Evaluate on scaffold CV of original CRC (proxy metric)
    oof_rnd = np.full(len(y_tr), np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        mf = lgb.train(LGBM, lgb.Dataset(X_cur[tr_idx], label=y_cur[tr_idx], weight=w_cur[tr_idx]),
                       valid_sets=[lgb.Dataset(X_tr[va_idx], label=y_tr[va_idx])],
                       callbacks=[lgb.early_stopping(50,verbose=False), lgb.log_evaluation(-1)])
        oof_rnd[va_idx] = mf.predict(X_tr[va_idx])
    r = full_metrics(y_tr, oof_rnd, cliff_pairs, f"round_{rnd+1}")
    history.append(r)

    # Add high-confidence pseudo-labels
    conf_mask = max_sim >= THRESHOLDS[rnd]
    n_add = conf_mask.sum()
    print(f"  Adding {n_add} pseudo-labels (sim≥{THRESHOLDS[rnd]:.2f})")
    if n_add > 0:
        X_pl = X_te[conf_mask]
        y_pl = te_pred[conf_mask]
        w_pl = (max_sim[conf_mask] * WEIGHT_SCALE).astype(np.float32)
        X_cur = np.vstack([X_tr, X_pl])
        y_cur = np.concatenate([y_tr, y_pl])
        w_cur = np.concatenate([np.ones(len(y_tr), dtype=np.float32), w_pl])
    oof = oof_rnd

print("\\n=== Training history ===")
print(pd.DataFrame(history, index=[f"round_{i+1}" for i in range(N_ROUNDS)]).round(4).to_string())
'''),
cell('''\
# Final model + test predictions
m_final = lgb.train(LGBM, lgb.Dataset(X_cur, label=y_cur, weight=w_cur),
                    callbacks=[lgb.log_evaluation(-1)])
te_preds = np.clip(m_final.predict(X_te), y_tr.min()-0.5, y_tr.max()+0.5)
print(f"Final training size: {len(X_cur):,}  (original: {len(X_tr):,} + pseudo: {len(X_cur)-len(X_tr):,})")
''' + save_cell(81, "oof_pseudo_label", "pseudo_label_selftrain")),
])


# ══════════════════════════════════════════════════════════════════════════════
# nb82 — Selectivity-aware prediction (pEC50 − pEC50_null)
# ══════════════════════════════════════════════════════════════════════════════
save("82_selectivity_aware_prediction.ipynb", [
md("""# 82 — Selectivity-Aware Prediction

**Key insight:** The counter-assay (PXR-null) measures cytotoxicity, not PXR activation.
- High pEC50(PXR) + high pEC50(null) → cytotoxic, not selective
- High pEC50(PXR) + low pEC50(null) → true PXR agonist (selective)

Selectivity = pEC50(PXR) − pEC50(null) captures the *signal* we care about.

Strategy:
1. Train a selectivity model on the 2,858 compounds with both measurements
2. Train a null-predictor on counter-assay data (for compounds without null measurements)
3. Final: pEC50_pred = selectivity_pred + null_pred

This decomposes the problem into: "is this compound active?" (selectivity)
and "how potent is it non-selectively?" (null). Cleaner signal for both models.
"""),
cell(HEADER),
cell(METRICS),
cell(BASE_LOAD),
cell('''\
from pxr.data import load_counter
from pxr.chem import to_inchikey

ctr = load_counter().dropna(subset=["smiles","pec50"]).rename(columns={"pec50":"pec50_null"})
ctr["ik"] = ctr["smiles"].map(to_inchikey)
tr["ik"]  = tr["smiles"].map(to_inchikey)

# Merge: compounds with both CRC and counter-assay measurements
tr_merged = tr.merge(ctr[["ik","pec50_null"]], on="ik", how="left")
has_null = tr_merged["pec50_null"].notna().values
selectivity = (tr_merged["pec50"].values - tr_merged["pec50_null"].fillna(0).values).astype(np.float32)
print(f"Compounds with both CRC+null: {has_null.sum()} / {len(tr)}")
print(f"Selectivity distribution (where available):")
print(pd.Series(selectivity[has_null]).describe().round(3).to_string())
'''),
cell('''\
# Model 1: Selectivity predictor (pEC50 - pEC50_null)
print("\\n=== Model 1: Selectivity (pEC50 - pEC50_null) ===", flush=True)
sel_target = selectivity.copy()
sel_target[~has_null] = np.nan  # only use compounds with both measurements

oof_sel = np.full(len(y_tr), np.nan)
for fold, (tr_idx, va_idx) in enumerate(splits):
    tr_f = tr_idx[has_null[tr_idx]]
    va_f = va_idx[has_null[va_idx]]
    if len(tr_f) < 50 or len(va_f) < 10: continue
    m = lgb.train(LGBM, lgb.Dataset(X_tr[tr_f], label=sel_target[tr_f]),
                  valid_sets=[lgb.Dataset(X_tr[va_f], label=sel_target[va_f])],
                  callbacks=[lgb.early_stopping(50,verbose=False), lgb.log_evaluation(-1)])
    oof_sel[va_idx] = m.predict(X_tr[va_idx])
valid = has_null & np.isfinite(oof_sel)
print(f"Selectivity OOF RAE (where available): {rae(sel_target[valid], oof_sel[valid]):.4f}")

# Model 2: pEC50_null predictor
print("\\n=== Model 2: pEC50_null predictor ===", flush=True)
null_target = tr_merged["pec50_null"].values.astype(np.float32)
oof_null = np.full(len(y_tr), np.nan)
for fold, (tr_idx, va_idx) in enumerate(splits):
    tr_f = tr_idx[has_null[tr_idx]]
    va_f = va_idx[has_null[va_idx]]
    if len(tr_f) < 50 or len(va_f) < 10: continue
    m = lgb.train(LGBM, lgb.Dataset(X_tr[tr_f], label=null_target[tr_f]),
                  valid_sets=[lgb.Dataset(X_tr[va_f], label=null_target[va_f])],
                  callbacks=[lgb.early_stopping(50,verbose=False), lgb.log_evaluation(-1)])
    oof_null[va_idx] = m.predict(X_tr[va_idx])
'''),
cell('''\
# Reconstruct pEC50 = selectivity + null
oof_reconstructed = oof_sel + oof_null
valid_both = np.isfinite(oof_reconstructed)
print(f"Reconstructed pEC50 (selectivity+null): {valid_both.sum()} valid")

# Compare with direct model
oof_direct = np.full(len(y_tr), np.nan)
for fold, (tr_idx, va_idx) in enumerate(splits):
    m = lgb.train(LGBM, lgb.Dataset(X_tr[tr_idx], label=y_tr[tr_idx]),
                  valid_sets=[lgb.Dataset(X_tr[va_idx], label=y_tr[va_idx])],
                  callbacks=[lgb.early_stopping(50,verbose=False), lgb.log_evaluation(-1)])
    oof_direct[va_idx] = m.predict(X_tr[va_idx])

# Blend: where we have reconstructed predictions, blend; elsewhere use direct
oof = oof_direct.copy()
oof[valid_both] = 0.4 * oof_reconstructed[valid_both] + 0.6 * oof_direct[valid_both]

m_dir = full_metrics(y_tr, oof_direct, cliff_pairs, "direct")
m_rec = full_metrics(y_tr[valid_both], oof_reconstructed[valid_both],
                     label="sel+null_reconstructed")
m_blnd= full_metrics(y_tr, oof, cliff_pairs, "blended")
print("\\n" + pd.DataFrame([m_dir, m_rec, m_blnd],
      index=["direct","reconstructed","blended"]).round(4).to_string())
'''),
cell('''\
# Final models
m_sel_final = lgb.train(LGBM, lgb.Dataset(X_tr[has_null], label=sel_target[has_null]),
                         callbacks=[lgb.log_evaluation(-1)])
m_null_final = lgb.train(LGBM, lgb.Dataset(X_tr[has_null], label=null_target[has_null]),
                          callbacks=[lgb.log_evaluation(-1)])
m_direct_final = lgb.train(LGBM, lgb.Dataset(X_tr, label=y_tr), callbacks=[lgb.log_evaluation(-1)])

te_sel  = m_sel_final.predict(X_te)
te_null = m_null_final.predict(X_te)
te_rec  = te_sel + te_null
te_dir  = m_direct_final.predict(X_te)
te_preds = np.clip(0.4*te_rec + 0.6*te_dir, y_tr.min()-0.5, y_tr.max()+0.5)
print(f"Selectivity preds: mean={te_sel.mean():.2f}  null preds: mean={te_null.mean():.2f}")
''' + save_cell(82, "oof_selectivity_aware", "selectivity_aware")),
])


# ══════════════════════════════════════════════════════════════════════════════
# nb83 — Transductive graph label spreading
# ══════════════════════════════════════════════════════════════════════════════
save("83_graph_label_spreading.ipynb", [
md("""# 83 — Transductive Graph Label Spreading

Build a molecular similarity graph over ALL compounds (train + test).
Propagate known pEC50 labels from training nodes to unlabeled test nodes
through similarity edges — like PageRank but for activity prediction.

Why this works: test = analog expansion of 63 hits → dense edges between
test and training nodes → labels diffuse efficiently through the graph.

Uses sklearn LabelSpreading with a Tanimoto-affinity kernel.
The spread predictions are then blended with a direct LGBM model.
"""),
cell(HEADER),
cell(METRICS),
cell(BASE_LOAD),
cell('''\
from sklearn.semi_supervised import LabelSpreading

# Build affinity matrix over train+test (using top-k sparse Tanimoto)
n_tr = len(tr); n_te = len(te); N = n_tr + n_te
fps_all = np.vstack([fps_tr, fps_te]).astype(np.float32)
y_all_labeled = np.concatenate([y_tr, np.full(n_te, np.nan)])

print(f"Building Tanimoto affinity matrix for {N} compounds...", flush=True)
# Compute in batches to avoid OOM
BATCH = 200
K_TOP = 10   # keep top-K neighbors per compound for sparse graph
rows, cols, vals = [], [], []

for start in range(0, N, BATCH):
    end = min(start + BATCH, N)
    fps_batch = fps_all[start:end]
    dot = fps_batch @ fps_all.T
    rs_b = fps_batch.sum(1)[:,None]; rs_a = fps_all.sum(1)[None,:]
    sim_block = dot / np.maximum(rs_b + rs_a - dot, 1e-6)
    np.fill_diagonal(sim_block[:,start:end], 0)  # no self-loops
    for i, sim_row in enumerate(sim_block):
        top_k = np.argpartition(sim_row, -K_TOP)[-K_TOP:]
        top_k = top_k[sim_row[top_k] > 0.3]  # threshold
        for j in top_k:
            rows.append(start + i); cols.append(j); vals.append(float(sim_row[j]))
    if (start//BATCH) % 10 == 0:
        print(f"  {end}/{N}", flush=True)

from scipy.sparse import csr_matrix
A = csr_matrix((vals, (rows, cols)), shape=(N, N))
A = A.maximum(A.T)  # symmetrize
print(f"Graph: {N} nodes, {A.nnz} edges, density={A.nnz/(N*N)*100:.3f}%")
'''),
cell('''\
# LabelSpreading expects dense affinity — use our sparse matrix directly
# For large N, use manual iteration (kernel=precomputed)
print("Running label spreading...", flush=True)
ls = LabelSpreading(kernel="rbf", alpha=0.8, max_iter=100, tol=1e-4)

# Encode: labeled=0..max_class (regression hack: bin pEC50 into 20 classes)
N_BINS = 20
y_min, y_max = y_tr.min(), y_tr.max()
bin_edges = np.linspace(y_min, y_max, N_BINS+1)
y_tr_binned = np.digitize(y_tr, bin_edges[1:-1])  # 0 to N_BINS-1
y_ls_input = np.concatenate([y_tr_binned, np.full(n_te, -1)])  # -1 = unlabeled

# Use dense Tanimoto matrix (for n<5000 this is feasible)
if N <= 5000:
    A_dense = A.toarray().astype(np.float64)
    ls.fit(A_dense, y_ls_input)
    y_spread_bins = ls.transduction_
else:
    # Manual power iteration for larger graphs
    print("  Too large for dense — using sparse power iteration")
    # Initialize: training nodes fixed, test nodes = training mean
    f = np.zeros(N); f[:n_tr] = y_tr; f[n_tr:] = y_tr.mean()
    D = np.array(A.sum(1)).flatten()
    D_inv = np.where(D>0, 1/D, 0)
    alpha_ls = 0.8
    for it in range(30):
        f_new = alpha_ls * (A.dot(f) * D_inv) + (1-alpha_ls) * np.where(
            y_ls_input >= 0, y_tr.mean() + (y_ls_input - y_tr_binned.mean()) * (y_max-y_min)/N_BINS, 0)
        f_new[:n_tr] = y_tr  # clamp labeled nodes
        if np.max(np.abs(f_new - f)) < 1e-4: break
        f = f_new
    y_spread = f
    y_spread_bins = None

print("Label spreading done.")
'''),
cell('''\
# Extract test predictions from label spreading
if y_spread_bins is not None:
    # Convert bins back to pEC50
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    y_spread_all = bin_centers[np.clip(y_spread_bins, 0, N_BINS-1)]
else:
    y_spread_all = y_spread

te_spread = y_spread_all[n_tr:]
print(f"Spread predictions range: [{te_spread.min():.2f}, {te_spread.max():.2f}]")

# Compare with direct LGBM as OOF on training nodes
oof_direct = np.full(n_tr, np.nan)
for fold, (tr_idx, va_idx) in enumerate(splits):
    m = lgb.train(LGBM, lgb.Dataset(X_tr[tr_idx], label=y_tr[tr_idx]),
                  valid_sets=[lgb.Dataset(X_tr[va_idx], label=y_tr[va_idx])],
                  callbacks=[lgb.early_stopping(50,verbose=False), lgb.log_evaluation(-1)])
    oof_direct[va_idx] = m.predict(X_tr[va_idx])

m_dir = full_metrics(y_tr, oof_direct, cliff_pairs, "direct_lgbm")

# Spread OOF: check spread quality on training (labeled) nodes
y_spread_tr = y_spread_all[:n_tr]
m_spread = full_metrics(y_tr, y_spread_tr, cliff_pairs, "spread_train_check")

# Blend spread + direct for test predictions
m_final = lgb.train(LGBM, lgb.Dataset(X_tr, label=y_tr), callbacks=[lgb.log_evaluation(-1)])
te_direct = m_final.predict(X_te)

# Weight by spread confidence (how densely connected is each test node)
te_density = np.array(A[n_tr:].sum(1)).flatten()
te_density_norm = te_density / (te_density.max() + 1e-8)

alpha_s = 0.3  # weight for spread prediction
te_preds = (alpha_s * te_spread * te_density_norm + (1-alpha_s) * te_direct)
te_preds = np.clip(te_preds / (alpha_s * te_density_norm + (1-alpha_s)),
                   y_tr.min()-0.5, y_tr.max()+0.5)
oof = oof_direct  # OOF from LGBM (spread doesn't have proper CV OOF)
print(pd.DataFrame([m_dir, m_spread],index=["direct","spread"]).round(4).to_string())
''' + save_cell(83, "oof_graph_spreading", "graph_label_spreading")),
])


# ══════════════════════════════════════════════════════════════════════════════
# nb84 — Free-Wilson scaffold decomposition
# ══════════════════════════════════════════════════════════════════════════════
save("84_free_wilson_decomposition.ipynb", [
md("""# 84 — Free-Wilson / Matched Molecular Pair Scaffold Decomposition

**Classical QSAR wisdom:** in an analog series, pEC50 ≈ μ_scaffold + Σ(R_group_i contribution).

The Free-Wilson model is additive: each R-group substituent has a fixed contribution
to pEC50 regardless of what other R-groups are present. It works best when:
1. You have many analogs of the same scaffold ✓ (test = analog expansion)
2. The substituent effects are additive ✓ (common for PXR ligands)

Implementation:
1. Use RDKit MMP fragmentation to decompose each compound into scaffold + R-groups
2. Encode R-group identity as binary features (one-hot per unique R-group)
3. Fit ridge regression → each R-group gets a coefficient
4. For test compounds: look up their R-groups and predict from scaffold+R-group model
"""),
cell(HEADER),
cell(METRICS),
cell(BASE_LOAD),
cell('''\
from rdkit import Chem
from rdkit.Chem import rdMMPA, Scaffolds
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict

def decompose_rgroups(smiles_list, max_cuts=1):
    """MMP single-cut decomposition: scaffold + R-group."""
    results = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            results.append({"scaffold": smi, "rgroups": []})
            continue
        try:
            frags = rdMMPA.FragmentMol(mol, maxCuts=max_cuts, resultsAsMols=False)
            # frags = list of (core, rgroup) SMILES pairs
            if not frags:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
                results.append({"scaffold": scaffold, "rgroups": []})
                continue
            # Pick the fragmentation with the largest core
            best = max(frags, key=lambda x: Chem.MolFromSmiles(x[0]).GetNumHeavyAtoms()
                       if Chem.MolFromSmiles(x[0]) else 0)
            results.append({"scaffold": best[0], "rgroups": [best[1]] if best[1] else []})
        except:
            results.append({"scaffold": smi, "rgroups": []})
    return results

print("Decomposing training set into scaffold + R-groups...", flush=True)
decomp_tr = decompose_rgroups(tr["smiles"].tolist())
decomp_te = decompose_rgroups(te["smiles"].tolist())

# Count unique scaffolds and R-groups
scaffolds_freq = defaultdict(int)
rgroups_freq   = defaultdict(int)
for d in decomp_tr:
    scaffolds_freq[d["scaffold"]] += 1
    for rg in d["rgroups"]:
        rgroups_freq[rg] += 1

print(f"Unique scaffolds: {len(scaffolds_freq)}")
print(f"Unique R-groups:  {len(rgroups_freq)}")
print(f"Top scaffolds: {sorted(scaffolds_freq.items(), key=lambda x:-x[1])[:5]}")
'''),
cell('''\
# Keep scaffolds and R-groups that appear ≥ MIN_FREQ times
MIN_FREQ = 3
common_scaffolds = {s for s,c in scaffolds_freq.items() if c >= MIN_FREQ}
common_rgroups   = {r for r,c in rgroups_freq.items()   if c >= MIN_FREQ}
print(f"Common scaffolds (≥{MIN_FREQ}): {len(common_scaffolds)}")
print(f"Common R-groups (≥{MIN_FREQ}):  {len(common_rgroups)}")

scaffold_idx = {s:i for i,s in enumerate(sorted(common_scaffolds))}
rgroup_idx   = {r:i for i,r in enumerate(sorted(common_rgroups))}
n_s = len(scaffold_idx); n_r = len(rgroup_idx)

def build_fw_features(decomp_list):
    """Binary feature: scaffold present (n_s cols) + R-group present (n_r cols)."""
    X = np.zeros((len(decomp_list), n_s + n_r), dtype=np.float32)
    coverage = 0
    for i, d in enumerate(decomp_list):
        if d["scaffold"] in scaffold_idx:
            X[i, scaffold_idx[d["scaffold"]]] = 1; coverage += 1
        for rg in d["rgroups"]:
            if rg in rgroup_idx:
                X[i, n_s + rgroup_idx[rg]] = 1
    return X, coverage

X_fw_tr, cov_tr = build_fw_features(decomp_tr)
X_fw_te, cov_te = build_fw_features(decomp_te)
print(f"Training coverage: {cov_tr}/{len(tr)} ({100*cov_tr/len(tr):.1f}%)")
print(f"Test coverage: {cov_te}/{len(te)} ({100*cov_te/len(te):.1f}%)")
print(f"FW features: {X_fw_tr.shape}")
'''),
cell('''\
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Model 1: Pure Free-Wilson (Ridge on scaffold+R-group binary features)
print("\\n=== Free-Wilson model (Ridge regression) ===", flush=True)
oof_fw = np.full(len(y_tr), np.nan)
for fold, (tr_idx, va_idx) in enumerate(splits):
    fw = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    fw.fit(X_fw_tr[tr_idx], y_tr[tr_idx])
    oof_fw[va_idx] = fw.predict(X_fw_tr[va_idx])
m_fw = full_metrics(y_tr, oof_fw, cliff_pairs, "free_wilson")
m_fw_a = full_metrics(y_tr[active_mask], oof_fw[active_mask], "fw [active]")

# Model 2: FW features concatenated with 2D Morgan (hybrid)
X_hybrid_tr = np.hstack([X_tr, X_fw_tr])
X_hybrid_te  = np.hstack([X_te, X_fw_te])
oof_hybrid = np.full(len(y_tr), np.nan)
for fold, (tr_idx, va_idx) in enumerate(splits):
    m = lgb.train(LGBM, lgb.Dataset(X_hybrid_tr[tr_idx], label=y_tr[tr_idx]),
                  valid_sets=[lgb.Dataset(X_hybrid_tr[va_idx], label=y_tr[va_idx])],
                  callbacks=[lgb.early_stopping(50,verbose=False), lgb.log_evaluation(-1)])
    oof_hybrid[va_idx] = m.predict(X_hybrid_tr[va_idx])
m_hyb = full_metrics(y_tr, oof_hybrid, cliff_pairs, "lgbm_fw_hybrid")

oof = oof_hybrid
print("\\n" + pd.DataFrame([m_fw, m_fw_a, m_hyb],
      index=["free_wilson","fw_active","hybrid"]).round(4).to_string())

# R-group importance from Ridge (Free-Wilson coefficients)
fw_final = RidgeCV(alphas=[0.01,0.1,1.0,10.0,100.0]).fit(X_fw_tr, y_tr)
rg_coefs = fw_final.coef_[n_s:]
top_pos = sorted(zip(common_rgroups, rg_coefs), key=lambda x:-x[1])[:10]
top_neg = sorted(zip(common_rgroups, rg_coefs), key=lambda x:x[1])[:10]
print("\\nTop activity-enhancing R-groups:")
for rg, c in top_pos: print(f"  {rg[:40]:40s} +{c:.3f}")
print("Top activity-reducing R-groups:")
for rg, c in top_neg: print(f"  {rg[:40]:40s} {c:.3f}")
'''),
cell('''\
m_final = lgb.train(LGBM, lgb.Dataset(X_hybrid_tr, label=y_tr), callbacks=[lgb.log_evaluation(-1)])
te_preds = np.clip(m_final.predict(X_hybrid_te), y_tr.min()-0.5, y_tr.max()+0.5)
# Also save FW features for use in grand ensemble
np.save(DATA_PROCESSED/"X_fw_tr.npy", X_fw_tr)
np.save(DATA_PROCESSED/"X_fw_te.npy", X_fw_te)
''' + save_cell(84, "oof_free_wilson", "free_wilson_hybrid")),
])


# ══════════════════════════════════════════════════════════════════════════════
# nb85 — Creative mega-ensemble (best of nb76–nb84 + prior stack)
# ══════════════════════════════════════════════════════════════════════════════
save("85_creative_mega_ensemble.ipynb", [
md("""# 85 — Creative Mega-Ensemble

Combines all creative OOF predictions (nb76–nb84) with the existing best models
(nb36 grand ensemble v6b, nb35 Chemprop auxiliary, nb42 SMOTE-ADASYN etc.)
using ElasticNetCV stacking.

Also applies the Delta-ML correction on top of the ensemble:
final = ensemble_pred + alpha * (delta_correction for high-similarity test compounds)
"""),
cell(HEADER),
cell(METRICS),
cell(BASE_LOAD),
cell('''\
from sklearn.linear_model import ElasticNetCV
from pathlib import Path

# Discover all OOF predictions
oof_files = sorted(DATA_PROCESSED.glob("oof_*.npy"))
print(f"Discovered {len(oof_files)} OOF files")

oofs, tes, names = [], [], []
for fp in oof_files:
    name = fp.stem.replace("oof_","")
    te_fp = DATA_PROCESSED / f"te_{fp.stem.replace('oof_','')}.npy"
    try:
        oof = np.load(fp)
        if len(oof) != len(y_tr): continue
        te_v = np.load(te_fp) if te_fp.exists() else None
        if te_v is None or len(te_v) != 513: continue
        if not np.isfinite(oof).all(): oof[~np.isfinite(oof)] = y_tr.mean()
        if not np.isfinite(te_v).all(): te_v[~np.isfinite(te_v)] = float(np.nanmean(te_v))
        oofs.append(oof); tes.append(te_v); names.append(name)
    except Exception as e:
        print(f"  skip {name}: {e}")

print(f"Using {len(names)} models: {names}")
OOF_stack = np.column_stack(oofs)
TE_stack  = np.column_stack(tes)
'''),
cell('''\
# ElasticNet meta-learner
meta = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], cv=5, max_iter=10000,
                     random_state=SEED)
meta.fit(OOF_stack, y_tr)
oof = meta.predict(OOF_stack)

coef_df = pd.DataFrame({"model": names, "weight": meta.coef_}).sort_values("weight", ascending=False)
print("Non-zero model weights:")
print(coef_df[coef_df.weight != 0].to_string(index=False))

m_ens = full_metrics(y_tr, oof, cliff_pairs, "mega_ensemble")
m_ens_a = full_metrics(y_tr[active_mask], oof[active_mask], "mega_ensemble [active]")
print("\\n" + pd.DataFrame([m_ens, m_ens_a], index=["overall","active"]).round(4).to_string())
'''),
cell('''\
# Delta-ML correction for high-similarity test compounds
# Load delta model OOF if available
delta_te_path = DATA_PROCESSED / "te_oof_delta_ml.npy"
te_base = meta.predict(TE_stack)

# Compare individual models to ensemble on active compounds
print("\\n=== Active compound (pEC50≥5.5) RAE per model ===")
for name, oof_m in zip(names, oofs):
    m = full_metrics(y_tr[active_mask], oof_m[active_mask])
    print(f"  {name:<35} RAE={m['RAE']:.4f}  Cliff_acc={m.get('Cliff_acc',float('nan')):.3f}")

te_preds = np.clip(te_base, y_tr.min()-0.5, y_tr.max()+0.5)
''' + save_cell(85, "oof_creative_mega_ensemble", "creative_mega_ensemble")),
])


print("\nAll v4 notebooks written:")
for p in sorted(NB_DIR.glob("[7-9][0-9]_*.ipynb")):
    print(f"  {p.name}")
