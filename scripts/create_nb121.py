import json

cells = []

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": [src]}

cells.append(md("# nb121 - Adaptive-K Delta (Proper Nested CV)\n\nFixes the data leakage in nb118: delta models are retrained per fold using only train-fold pairs.\nThis gives a true (unbiased) OOF RAE estimate.\n\nSame architecture as nb118:\n- 4 tiers: HIGH/MED/LOW/VERY_LOW with adaptive K (5/10/20/30)\n- 128-dim FP compression, ECFP6 Tanimoto as extra feature\n- VERY_LOW tier extends to sim=0.25\n\nFinal test predictions use all training data (no leakage for test set)."))

cells.append(code("""import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "../src")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from pxr.data import load_train, load_test
from pxr.featurize import combined, impute
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko, morgan_fp_batch, compute_physchem
from pxr.paths import DATA_PROCESSED, SUBMISSIONS
SEED = 42; N_FOLDS = 5
LGBM_BASE = dict(n_estimators=1200, num_leaves=64, learning_rate=0.04,
                 min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
                 reg_alpha=0.1, reg_lambda=0.1, random_state=SEED, verbose=-1, n_jobs=4)
print("imports OK")
"""))

cells.append(code("""def full_metrics(y_true, y_pred, label=""):
    yt = np.asarray(y_true, float); yp = np.asarray(y_pred, float)
    msk = np.isfinite(yt) & np.isfinite(yp); yt, yp = yt[msk], yp[msk]
    mae = float(np.mean(np.abs(yt-yp)))
    rae_v = mae / float(np.mean(np.abs(yt-yt.mean()))) if yt.std()>0 else float("nan")
    r2  = 1-np.sum((yt-yp)**2)/np.sum((yt-yt.mean())**2) if yt.std()>0 else float("nan")
    pr, _ = stats.pearsonr(yt, yp); sp, _ = stats.spearmanr(yt, yp)
    m = dict(RAE=rae_v, MAE=mae, R2=float(r2), Pearson=float(pr), Spearman=float(sp))
    if label:
        print(f"  [{label}] RAE={rae_v:.4f} MAE={mae:.4f} R2={r2:.4f} r={pr:.4f} rho={sp:.4f}")
    return m
"""))

cells.append(code("""def ecfp4_batch(smiles_list, n_bits=2048):
    fps = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(str(s))
        if mol: fps.append(list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)))
        else: fps.append([0]*n_bits)
    return np.array(fps, dtype=np.float32)

def ecfp6_batch(smiles_list, n_bits=2048):
    fps = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(str(s))
        if mol: fps.append(list(AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=n_bits)))
        else: fps.append([0]*n_bits)
    return np.array(fps, dtype=np.float32)

def maccs_batch(smiles_list):
    fps = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(str(s))
        if mol: fps.append(list(MACCSkeys.GenMACCSKeys(mol)))
        else: fps.append([0]*167)
    return np.array(fps, dtype=np.float32)[:, 1:]

PHYS_PROPS = ["mw","logp","tpsa","hbd","hba","rotbonds","fsp3",
               "n_rings","n_aromatic_rings","heavy_atoms","formal_charge"]

def physchem_batch(smiles_list):
    rows = []
    for s in smiles_list:
        p = compute_physchem(str(s))
        rows.append([p.get(k, 0) or 0 for k in PHYS_PROPS])
    return np.array(rows, dtype=np.float32)

print("FP functions ready.")
"""))

cells.append(code("""tr = load_train(); te = load_test()
y_tr = tr["pec50"].values.astype(np.float64)
scaffolds = tr["smiles"].map(bemis_murcko).tolist()
splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

X_tr = impute(combined(tr["smiles"].tolist()))
X_te = impute(combined(te["smiles"].tolist()))

print("Computing fingerprints...", flush=True)
fps4_tr = ecfp4_batch(tr["smiles"].tolist())
fps4_te = ecfp4_batch(te["smiles"].tolist())
fps6_tr = ecfp6_batch(tr["smiles"].tolist())
fps6_te = ecfp6_batch(te["smiles"].tolist())
maccs_tr = maccs_batch(tr["smiles"].tolist())
maccs_te = maccs_batch(te["smiles"].tolist())
phys_tr = physchem_batch(tr["smiles"].tolist())
phys_te = physchem_batch(te["smiles"].tolist())

# Full train-train Tanimoto
print("Computing train-train Tanimoto...", flush=True)
dot_tt = (fps4_tr @ fps4_tr.T).astype(np.float32)
rowsum = fps4_tr.sum(1).astype(np.float32)
union_tt = rowsum[:,None] + rowsum[None,:] - dot_tt
tanimoto_tr = np.where(union_tt>0, dot_tt/union_tt, 0.0)
np.fill_diagonal(tanimoto_tr, 0.0)

# Test-train Tanimoto (for final predictions)
dot_te = (fps4_te @ fps4_tr.T).astype(np.float32)
rs_te = fps4_te.sum(1)[:,None]; rs_tr_v = fps4_tr.sum(1)[None,:]
sim_te_tr = dot_te / np.maximum(rs_te + rs_tr_v - dot_te, 1e-6)

print(f"Train {len(tr):,}  Test {len(te):,}")
"""))

cells.append(code("""COMP_DIM = 128
MACCS_DIM = 64

def compress(fp, out_dim):
    N, D = fp.shape; block = D // out_dim
    return fp[:, :block*out_dim].reshape(N, out_dim, block).mean(-1).astype(np.float32)

def make_delta_feats(fp4_a, fp4_q, fp6_a, fp6_q, maccs_a, maccs_q,
                     sim_col, anchor_pec50, phys_diff):
    fp4_common = np.minimum(fp4_a, fp4_q).astype(np.float32)
    fp4_diff   = np.abs(fp4_a - fp4_q).astype(np.float32)
    c4 = compress(fp4_common, COMP_DIM)
    d4 = compress(fp4_diff,   COMP_DIM)
    fp6_common = np.minimum(fp6_a, fp6_q).astype(np.float32)
    fp6_diff   = np.abs(fp6_a - fp6_q).astype(np.float32)
    c6 = compress(fp6_common, COMP_DIM)
    d6 = compress(fp6_diff,   COMP_DIM)
    maccs_diff = np.abs(maccs_a - maccs_q).astype(np.float32)
    cm = compress(maccs_diff, MACCS_DIM)
    # ECFP6 Tanimoto
    dot6 = np.sum(fp6_a * fp6_q, axis=1, keepdims=True)
    rs6a = fp6_a.sum(1, keepdims=True); rs6q = fp6_q.sum(1, keepdims=True)
    sim6 = dot6 / np.maximum(rs6a + rs6q - dot6, 1e-6)
    return np.hstack([c4, d4, c6, d6, cm, sim_col, sim6, anchor_pec50[:,None], phys_diff])

TIERS = {
    "HIGH":     (0.60, 0.90, 5,  3.0),
    "MED":      (0.45, 0.60, 10, 2.0),
    "LOW":      (0.35, 0.45, 20, 1.5),
    "VERY_LOW": (0.25, 0.35, 30, 1.0),
}

DELTA_LGBM = dict(n_estimators=1500, num_leaves=63, learning_rate=0.04,
                  min_child_samples=15, subsample=0.8, colsample_bytree=0.7,
                  reg_alpha=0.05, reg_lambda=0.1, random_state=SEED, verbose=-1, n_jobs=4)

def train_tier_models_on_subset(idx_subset, tanimoto_sub):
    # Build pairs ONLY within idx_subset
    i_f, j_f = np.where(np.triu(tanimoto_sub > 0.20, k=1))
    sim_f = tanimoto_sub[i_f, j_f]

    models = {}
    for tier, (lo, hi, K, pw) in TIERS.items():
        if tier == "HIGH" or tier == "VERY_LOW":
            mask = (sim_f >= lo) & (sim_f <= hi)
        else:
            mask = (sim_f >= lo) & (sim_f < hi)
        ii_l, jj_l = i_f[mask], j_f[mask]
        # Convert local indices to global
        ii_g = idx_subset[ii_l]
        jj_g = idx_subset[jj_l]
        sim_ij = sim_f[mask][:,None]

        pd_ij = phys_tr[jj_g] - phys_tr[ii_g]
        F_ij = make_delta_feats(fps4_tr[ii_g], fps4_tr[jj_g],
                                fps6_tr[ii_g], fps6_tr[jj_g],
                                maccs_tr[ii_g], maccs_tr[jj_g],
                                sim_ij, y_tr[ii_g], pd_ij)
        F_ji = make_delta_feats(fps4_tr[jj_g], fps4_tr[ii_g],
                                fps6_tr[jj_g], fps6_tr[ii_g],
                                maccs_tr[jj_g], maccs_tr[ii_g],
                                sim_ij, y_tr[jj_g], -pd_ij)
        F = np.vstack([F_ij, F_ji])
        y = np.concatenate([y_tr[jj_g]-y_tr[ii_g], y_tr[ii_g]-y_tr[jj_g]])

        if len(F) < 4:
            models[tier] = None
            print(f"    {tier}: only {len(F)} pairs, skipping", flush=True)
            continue
        m = lgb.LGBMRegressor(**DELTA_LGBM)
        m.fit(F, y, callbacks=[lgb.log_evaluation(-1)])
        models[tier] = m
        print(f"    {tier}: trained on {len(F):,} pairs", flush=True)
    return models

def predict_with_models(fold_models, fps4_q, fps4_r, fps6_q, fps6_r,
                        maccs_q, maccs_r, y_ref, phys_q, phys_r, sim_mat, fallback):
    N = len(fps4_q)
    preds = np.full(N, np.nan)
    tc = {t: 0 for t in TIERS}; tc["fallback"] = 0

    for qi in range(N):
        sim_row = sim_mat[qi]
        assigned = False
        for tier, (lo, hi, K, pw) in TIERS.items():
            cand_mask = (sim_row >= lo) & (sim_row <= hi)
            cand_idx = np.where(cand_mask)[0]
            if len(cand_idx) == 0: continue
            mdl = fold_models.get(tier)
            if mdl is None: continue

            top_k = np.argsort(-sim_row[cand_idx])[:K]
            sel_idx = cand_idx[top_k]
            cand_sims = sim_row[sel_idx]

            fp4q = np.tile(fps4_q[qi:qi+1], (len(sel_idx),1))
            fp6q = np.tile(fps6_q[qi:qi+1], (len(sel_idx),1))
            macq = np.tile(maccs_q[qi:qi+1], (len(sel_idx),1))
            pd   = phys_q[qi:qi+1] - phys_r[sel_idx]

            F_k = make_delta_feats(
                fps4_r[sel_idx], fp4q, fps6_r[sel_idx], fp6q,
                maccs_r[sel_idx], macq,
                cand_sims[:,None], y_ref[sel_idx], pd)
            delta_k = mdl.predict(F_k)
            template_preds = y_ref[sel_idx] + delta_k
            weights = cand_sims ** pw
            preds[qi] = np.average(template_preds, weights=weights)
            tc[tier] += 1; assigned = True; break

        if not assigned:
            preds[qi] = fallback[qi]; tc["fallback"] += 1

    return preds, tc

print("Functions ready.")
"""))

cells.append(code("""print("\\n=== Scaffold 5-fold CV (PROPER NESTED) ===", flush=True)
oof_delta = np.full(len(y_tr), np.nan)
oof_direct = np.full(len(y_tr), np.nan)

for fold, (tr_idx, va_idx) in enumerate(splits):
    print(f"\\nFold {fold+1}/5 — training delta models on {len(tr_idx):,} train compounds...", flush=True)

    # Direct LGBM on fold
    m_dir = lgb.train(LGBM_BASE, lgb.Dataset(X_tr[tr_idx], label=y_tr[tr_idx]),
                      valid_sets=[lgb.Dataset(X_tr[va_idx], label=y_tr[va_idx])],
                      callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(-1)])
    oof_direct[va_idx] = m_dir.predict(X_tr[va_idx])

    # Train-fold-only Tanimoto submatrix
    tanimoto_fold = tanimoto_tr[np.ix_(tr_idx, tr_idx)]

    # Train delta models ONLY on train-fold pairs (no leakage!)
    fold_models = train_tier_models_on_subset(np.array(tr_idx), tanimoto_fold)

    # Compute val-train Tanimoto for inference
    fps4_va = fps4_tr[va_idx]; fps4_ft = fps4_tr[tr_idx]
    dot_vf = (fps4_va @ fps4_ft.T).astype(np.float32)
    rs_v = fps4_va.sum(1)[:,None]; rs_f = fps4_ft.sum(1)[None,:]
    sim_vf = dot_vf / np.maximum(rs_v + rs_f - dot_vf, 1e-6)

    preds_d, tc = predict_with_models(
        fold_models,
        fps4_tr[va_idx], fps4_tr[tr_idx],
        fps6_tr[va_idx], fps6_tr[tr_idx],
        maccs_tr[va_idx], maccs_tr[tr_idx],
        y_tr[tr_idx], phys_tr[va_idx], phys_tr[tr_idx],
        sim_vf, oof_direct[va_idx])
    oof_delta[va_idx] = preds_d

    r_dir = rae(y_tr[va_idx], oof_direct[va_idx])
    r_dlt = rae(y_tr[va_idx], oof_delta[va_idx])
    print(f"  fold {fold+1}  direct={r_dir:.4f}  nested_delta={r_dlt:.4f}  tiers={tc}", flush=True)

m_dir = full_metrics(y_tr, oof_direct, "direct_lgbm")
m_dlt = full_metrics(y_tr, oof_delta,  "nested_adaptive_delta")

for nb_path, name in [("oof_allfp_delta_3tier.npy","nb117"), ("oof_adaptive_delta_4tier.npy","nb118 (leaky)")]:
    p = DATA_PROCESSED / nb_path
    if p.exists():
        r = rae(y_tr, np.load(p))
        print(f"{name}: {r:.4f}")
print(f"nb121 (nested CV, unbiased): {m_dlt['RAE']:.4f}")
print(f"\\n*** nb121 OOF RAE = {m_dlt['RAE']:.4f} ***")
"""))

cells.append(code("""# Final test predictions (all training data, same as nb118)
print("\\nTraining global delta models for test prediction...", flush=True)
global_models = train_tier_models_on_subset(np.arange(len(y_tr)), tanimoto_tr)

print("Fitting final direct LGBM...", flush=True)
m_final = lgb.train(LGBM_BASE, lgb.Dataset(X_tr, label=y_tr), callbacks=[lgb.log_evaluation(-1)])
te_direct = m_final.predict(X_te)

print("Running nested-adaptive delta on test...", flush=True)
te_delta, te_tc = predict_with_models(
    global_models,
    fps4_te, fps4_tr, fps6_te, fps6_tr,
    maccs_te, maccs_tr,
    y_tr, phys_te, phys_tr,
    sim_te_tr, te_direct)
print(f"Test tier usage: {te_tc}")

te_preds = np.clip(te_delta, y_tr.min()-0.5, y_tr.max()+0.5)

np.save(DATA_PROCESSED/"oof_nested_adaptive_delta.npy", oof_delta)
np.save(DATA_PROCESSED/"te_oof_nested_adaptive_delta.npy", te_preds)
sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_preds})
assert len(sub)==513 and sub["pEC50"].notna().all()
p = SUBMISSIONS/"121_nested_adaptive_delta.csv"; sub.to_csv(p, index=False)
print(f"Saved {p}")
print(f"Test: min={te_preds.min():.2f} med={np.median(te_preds):.2f} max={te_preds.max():.2f}")
print(f"\\n*** nb121 OOF RAE = {m_dlt['RAE']:.4f} ***")
"""))

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "pxr-challenge", "language": "python", "name": "pxr-challenge"},
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "cells": cells
}

with open('notebooks/121_nested_adaptive_delta.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Written notebooks/121_nested_adaptive_delta.ipynb")
