import json

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": [src]}

C0 = md("# nb122 - Adaptive-K + All FPs + Full RDKit Desc (Proper Nested CV)\n\nKitchen-sink model combining:\n- nb117: 5 FP types (ECFP4+ECFP6+AtomPair+TopoTorsion+RDKitFP) compressed to 128 dims\n- nb118: 4 tiers adaptive-K (5/10/20/30), VERY_LOW extends to sim=0.25\n- nb120: Full 217 rdkit_desc delta (normalized) instead of 11-prop physchem\n- nb121: Proper nested CV (fold-specific delta models, no leakage)\n\nFeature dim: 5*256 + 64 (MACCS) + 2 (sims) + 1 (anchor) + 217 (rdkit) = 1564 features.")

C1 = code("""import os, sys, warnings
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
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors
from pxr.data import load_train, load_test
from pxr.featurize import combined, rdkit_desc, impute
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS
SEED = 42; N_FOLDS = 5
LGBM_BASE = dict(n_estimators=1200, num_leaves=64, learning_rate=0.04,
                 min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
                 reg_alpha=0.1, reg_lambda=0.1, random_state=SEED, verbose=-1, n_jobs=4)
DELTA_PARAMS = {
    "HIGH":     dict(n_estimators=1500, num_leaves=63, learning_rate=0.04,
                     min_child_samples=10, subsample=0.8, colsample_bytree=0.7,
                     reg_alpha=0.05, reg_lambda=0.1, random_state=SEED, verbose=-1, n_jobs=4),
    "MED":      dict(n_estimators=1500, num_leaves=63, learning_rate=0.04,
                     min_child_samples=15, subsample=0.8, colsample_bytree=0.7,
                     reg_alpha=0.05, reg_lambda=0.1, random_state=SEED, verbose=-1, n_jobs=4),
    "LOW":      dict(n_estimators=1200, num_leaves=63, learning_rate=0.04,
                     min_child_samples=15, subsample=0.8, colsample_bytree=0.7,
                     reg_alpha=0.05, reg_lambda=0.1, random_state=SEED, verbose=-1, n_jobs=4),
    "VERY_LOW": dict(n_estimators=800, num_leaves=63, learning_rate=0.05,
                     min_child_samples=20, subsample=0.8, colsample_bytree=0.7,
                     reg_alpha=0.05, reg_lambda=0.1, random_state=SEED, verbose=-1, n_jobs=4),
}
print("imports OK")
""")

C2 = code("""def full_metrics(y_true, y_pred, label=""):
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

def ecfp4_batch(smiles_list, n_bits=2048):
    fps = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(str(s))
        fps.append(list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)) if mol else [0]*n_bits)
    return np.array(fps, dtype=np.float32)

def ecfp6_batch(smiles_list, n_bits=2048):
    fps = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(str(s))
        fps.append(list(AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=n_bits)) if mol else [0]*n_bits)
    return np.array(fps, dtype=np.float32)

def maccs_batch(smiles_list):
    fps = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(str(s))
        fps.append(list(MACCSkeys.GenMACCSKeys(mol)) if mol else [0]*167)
    return np.array(fps, dtype=np.float32)[:, 1:]

def atom_pair_batch(smiles_list, n_bits=2048):
    fps = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(str(s))
        if mol:
            fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
            fps.append(list(fp))
        else: fps.append([0]*n_bits)
    return np.array(fps, dtype=np.float32)

def topo_torsion_batch(smiles_list, n_bits=2048):
    fps = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(str(s))
        if mol:
            fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=n_bits)
            fps.append(list(fp))
        else: fps.append([0]*n_bits)
    return np.array(fps, dtype=np.float32)

def rdkit_fp_batch(smiles_list, n_bits=2048):
    fps = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(str(s))
        if mol:
            fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
            fps.append(list(fp))
        else: fps.append([0]*n_bits)
    return np.array(fps, dtype=np.float32)

print("FP functions ready.")
""")

C3 = code("""tr = load_train(); te = load_test()
y_tr = tr["pec50"].values.astype(np.float64)
scaffolds = tr["smiles"].map(bemis_murcko).tolist()
splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

X_tr = impute(combined(tr["smiles"].tolist()))
X_te = impute(combined(te["smiles"].tolist()))

smiles_tr = tr["smiles"].tolist()
smiles_te = te["smiles"].tolist()

print("Computing RDKit descriptors (217 props)...", flush=True)
rdkit_raw_tr = rdkit_desc(smiles_tr)
rdkit_raw_te = rdkit_desc(smiles_te)
tr_med = np.nanmedian(rdkit_raw_tr, axis=0)
rdkit_imp_tr = np.where(np.isfinite(rdkit_raw_tr), rdkit_raw_tr, tr_med).astype(np.float32)
rdkit_imp_te = np.where(np.isfinite(rdkit_raw_te), rdkit_raw_te, tr_med).astype(np.float32)
tr_std = rdkit_imp_tr.std(0) + 1e-8
rdkit_n_tr = rdkit_imp_tr / tr_std
rdkit_n_te = rdkit_imp_te / tr_std
print(f"  RDKit desc: {rdkit_n_tr.shape}")

print("Computing all fingerprints...", flush=True)
fps4_tr = ecfp4_batch(smiles_tr); fps4_te = ecfp4_batch(smiles_te)
fps6_tr = ecfp6_batch(smiles_tr); fps6_te = ecfp6_batch(smiles_te)
maccs_tr = maccs_batch(smiles_tr); maccs_te = maccs_batch(smiles_te)
ap_tr = atom_pair_batch(smiles_tr); ap_te = atom_pair_batch(smiles_te)
tt_tr = topo_torsion_batch(smiles_tr); tt_te = topo_torsion_batch(smiles_te)
rdfp_tr = rdkit_fp_batch(smiles_tr); rdfp_te = rdkit_fp_batch(smiles_te)

print("Computing pairwise Tanimoto...", flush=True)
dot_tt = (fps4_tr @ fps4_tr.T).astype(np.float32)
rowsum = fps4_tr.sum(1).astype(np.float32)
union_tt = rowsum[:,None] + rowsum[None,:] - dot_tt
tanimoto_tr = np.where(union_tt>0, dot_tt/union_tt, 0.0)
np.fill_diagonal(tanimoto_tr, 0.0)

dot_te = (fps4_te @ fps4_tr.T).astype(np.float32)
rs_te = fps4_te.sum(1)[:,None]; rs_tr_v = fps4_tr.sum(1)[None,:]
sim_te_tr = dot_te / np.maximum(rs_te + rs_tr_v - dot_te, 1e-6)

print(f"Train {len(tr):,}  Test {len(te):,}  All FPs computed.")
""")

C4 = code("""COMP_DIM = 128
MACCS_DIM = 64

def compress(fp, out_dim):
    N, D = fp.shape; block = D // out_dim
    return fp[:, :block*out_dim].reshape(N, out_dim, block).mean(-1).astype(np.float32)

def tanimoto_col(fa, fb):
    dot = np.sum(fa * fb, axis=1, keepdims=True)
    rsa = fa.sum(1, keepdims=True); rsb = fb.sum(1, keepdims=True)
    return dot / np.maximum(rsa + rsb - dot, 1e-6)

def make_features(fp4_a, fp4_q, fp6_a, fp6_q, ap_a, ap_q, tt_a, tt_q, rdfp_a, rdfp_q,
                  maccs_a, maccs_q, sim_col, anchor_pec50, rdkit_diff):
    def cd(fa, fb):
        c = np.minimum(fa, fb).astype(np.float32)
        d = np.abs(fa - fb).astype(np.float32)
        return compress(c, COMP_DIM), compress(d, COMP_DIM)

    c4,  d4  = cd(fp4_a,  fp4_q)
    c6,  d6  = cd(fp6_a,  fp6_q)
    cap, dap = cd(ap_a,   ap_q)
    ctt, dtt = cd(tt_a,   tt_q)
    crd, drd = cd(rdfp_a, rdfp_q)

    maccs_d = np.abs(maccs_a - maccs_q).astype(np.float32)
    cm = compress(maccs_d, MACCS_DIM)
    sim6 = tanimoto_col(fp6_a, fp6_q)

    # 5*256 + 64 + 1 + 1 + 1 + 217 = 1564 features
    return np.hstack([c4, d4, c6, d6, cap, dap, ctt, dtt, crd, drd, cm,
                      sim_col, sim6, anchor_pec50[:,None], rdkit_diff])

# Verify feature dim
_t = make_features(
    fps4_tr[:2], fps4_tr[2:4], fps6_tr[:2], fps6_tr[2:4],
    ap_tr[:2], ap_tr[2:4], tt_tr[:2], tt_tr[2:4],
    rdfp_tr[:2], rdfp_tr[2:4], maccs_tr[:2], maccs_tr[2:4],
    np.ones((2,1)), y_tr[:2], rdkit_n_tr[:2]-rdkit_n_tr[2:4]
)
print(f"Feature dim: {_t.shape[1]}  (expect ~1564)")

TIERS = {
    "HIGH":     (0.60, 0.90, 5,  3.0),
    "MED":      (0.45, 0.60, 10, 2.0),
    "LOW":      (0.35, 0.45, 20, 1.5),
    "VERY_LOW": (0.25, 0.35, 30, 1.0),
}
""")

C5 = code("""def train_tier_models(idx_subset, tanimoto_sub):
    i_f, j_f = np.where(np.triu(tanimoto_sub > 0.20, k=1))
    sim_f = tanimoto_sub[i_f, j_f]
    models = {}

    for tier, (lo, hi, K, pw) in TIERS.items():
        if tier in ("HIGH", "VERY_LOW"):
            mask = (sim_f >= lo) & (sim_f <= hi)
        else:
            mask = (sim_f >= lo) & (sim_f < hi)
        ii_l, jj_l = i_f[mask], j_f[mask]
        ii_g = idx_subset[ii_l]; jj_g = idx_subset[jj_l]
        sim_ij = sim_f[mask][:,None]

        rd_diff_ij = rdkit_n_tr[jj_g] - rdkit_n_tr[ii_g]
        F_ij = make_features(fps4_tr[ii_g], fps4_tr[jj_g],
                              fps6_tr[ii_g], fps6_tr[jj_g],
                              ap_tr[ii_g],  ap_tr[jj_g],
                              tt_tr[ii_g],  tt_tr[jj_g],
                              rdfp_tr[ii_g],rdfp_tr[jj_g],
                              maccs_tr[ii_g],maccs_tr[jj_g],
                              sim_ij, y_tr[ii_g], rd_diff_ij)
        F_ji = make_features(fps4_tr[jj_g], fps4_tr[ii_g],
                              fps6_tr[jj_g], fps6_tr[ii_g],
                              ap_tr[jj_g],  ap_tr[ii_g],
                              tt_tr[jj_g],  tt_tr[ii_g],
                              rdfp_tr[jj_g],rdfp_tr[ii_g],
                              maccs_tr[jj_g],maccs_tr[ii_g],
                              sim_ij, y_tr[jj_g], -rd_diff_ij)
        F = np.vstack([F_ij, F_ji])
        y = np.concatenate([y_tr[jj_g]-y_tr[ii_g], y_tr[ii_g]-y_tr[jj_g]])

        if len(F) < 4:
            models[tier] = None; continue

        params = DELTA_PARAMS[tier]
        m = lgb.LGBMRegressor(**params)
        m.fit(F, y, callbacks=[lgb.log_evaluation(-1)])
        models[tier] = m
        print(f"    {tier}: {len(F):,} pairs, done.", flush=True)
    return models

def predict(models, fps4_q, fps4_r, fps6_q, fps6_r, ap_q, ap_r, tt_q, tt_r,
            rdfp_q, rdfp_r, maccs_q, maccs_r, rdkit_q, rdkit_r, y_ref, sim_mat, fallback):
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
            mdl = models.get(tier)
            if mdl is None: continue

            top_k = np.argsort(-sim_row[cand_idx])[:K]
            sel_idx = cand_idx[top_k]
            cand_sims = sim_row[sel_idx]
            n = len(sel_idx)

            def tile(arr): return np.tile(arr[qi:qi+1], (n, 1))

            rd_diff = rdkit_q[qi:qi+1] - rdkit_r[sel_idx]
            F_k = make_features(
                fps4_r[sel_idx], tile(fps4_q),
                fps6_r[sel_idx], tile(fps6_q),
                ap_r[sel_idx],   tile(ap_q),
                tt_r[sel_idx],   tile(tt_q),
                rdfp_r[sel_idx], tile(rdfp_q),
                maccs_r[sel_idx],tile(maccs_q),
                cand_sims[:,None], y_ref[sel_idx], rd_diff)
            delta_k = mdl.predict(F_k)
            weights = cand_sims ** pw
            preds[qi] = np.average(y_ref[sel_idx] + delta_k, weights=weights)
            tc[tier] += 1; assigned = True; break

        if not assigned:
            preds[qi] = fallback[qi]; tc["fallback"] += 1

    return preds, tc

print("Functions ready.")
""")

C6 = code("""print("\\n=== Scaffold 5-fold CV (NESTED) ===", flush=True)
oof_delta = np.full(len(y_tr), np.nan)
oof_direct = np.full(len(y_tr), np.nan)

for fold, (tr_idx, va_idx) in enumerate(splits):
    print(f"\\nFold {fold+1}/5 ({len(tr_idx):,} train, {len(va_idx):,} val)...", flush=True)

    m_dir = lgb.train(LGBM_BASE, lgb.Dataset(X_tr[tr_idx], label=y_tr[tr_idx]),
                      valid_sets=[lgb.Dataset(X_tr[va_idx], label=y_tr[va_idx])],
                      callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(-1)])
    oof_direct[va_idx] = m_dir.predict(X_tr[va_idx])

    tanimoto_fold = tanimoto_tr[np.ix_(tr_idx, tr_idx)]
    fold_models = train_tier_models(np.array(tr_idx), tanimoto_fold)

    fps4_va = fps4_tr[va_idx]; fps4_ft = fps4_tr[tr_idx]
    dot_vf = (fps4_va @ fps4_ft.T).astype(np.float32)
    rs_v = fps4_va.sum(1)[:,None]; rs_f = fps4_ft.sum(1)[None,:]
    sim_vf = dot_vf / np.maximum(rs_v + rs_f - dot_vf, 1e-6)

    preds_d, tc = predict(
        fold_models,
        fps4_tr[va_idx], fps4_tr[tr_idx],
        fps6_tr[va_idx], fps6_tr[tr_idx],
        ap_tr[va_idx],   ap_tr[tr_idx],
        tt_tr[va_idx],   tt_tr[tr_idx],
        rdfp_tr[va_idx], rdfp_tr[tr_idx],
        maccs_tr[va_idx],maccs_tr[tr_idx],
        rdkit_n_tr[va_idx], rdkit_n_tr[tr_idx],
        y_tr[tr_idx], sim_vf, oof_direct[va_idx])
    oof_delta[va_idx] = preds_d

    r_dir = rae(y_tr[va_idx], oof_direct[va_idx])
    r_dlt = rae(y_tr[va_idx], oof_delta[va_idx])
    print(f"  fold {fold+1}  direct={r_dir:.4f}  allfp_adaptive={r_dlt:.4f}  tiers={tc}", flush=True)

m_dir = full_metrics(y_tr, oof_direct, "direct_lgbm")
m_dlt = full_metrics(y_tr, oof_delta,  "allfp_adaptive_rdkit_nested")

for nb_path, name in [("oof_allfp_delta_3tier.npy","nb117 0.2333"),
                       ("oof_full_desc_delta_3tier.npy","nb120 0.2266"),
                       ("oof_adaptive_delta_4tier.npy","nb118 0.1626(leaky)"),
                       ("oof_nested_adaptive_delta.npy","nb121 (nested adaptive)")]:
    p = DATA_PROCESSED / nb_path
    if p.exists():
        print(f"  {name}: {rae(y_tr, np.load(p)):.4f}")
print(f"  nb122 (nested, all FPs + rdkit): {m_dlt['RAE']:.4f}")
print(f"\\n*** nb122 OOF RAE = {m_dlt['RAE']:.4f} ***")
""")

C7 = code("""print("\\nFitting global models for test prediction...", flush=True)
global_models = train_tier_models(np.arange(len(y_tr)), tanimoto_tr)

print("Fitting final direct LGBM...", flush=True)
m_final = lgb.train(LGBM_BASE, lgb.Dataset(X_tr, label=y_tr), callbacks=[lgb.log_evaluation(-1)])
te_direct = m_final.predict(X_te)

print("Running full adaptive delta on test...", flush=True)
te_delta, te_tc = predict(
    global_models,
    fps4_te, fps4_tr, fps6_te, fps6_tr,
    ap_te, ap_tr, tt_te, tt_tr,
    rdfp_te, rdfp_tr, maccs_te, maccs_tr,
    rdkit_n_te, rdkit_n_tr,
    y_tr, sim_te_tr, te_direct)
print(f"Test tier usage: {te_tc}")

te_preds = np.clip(te_delta, y_tr.min()-0.5, y_tr.max()+0.5)

np.save(DATA_PROCESSED/"oof_allfp_adaptive_rdkit.npy", oof_delta)
np.save(DATA_PROCESSED/"te_oof_allfp_adaptive_rdkit.npy", te_preds)
sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_preds})
assert len(sub)==513 and sub["pEC50"].notna().all()
p = SUBMISSIONS/"122_allfp_adaptive_rdkit.csv"; sub.to_csv(p, index=False)
print(f"Saved {p}")
print(f"Test: min={te_preds.min():.2f} med={np.median(te_preds):.2f} max={te_preds.max():.2f}")
print(f"\\n*** nb122 OOF RAE = {m_dlt['RAE']:.4f} ***")
""")

cells = [C0, C1, C2, C3, C4, C5, C6, C7]

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "pxr-challenge", "language": "python", "name": "pxr-challenge"},
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "cells": cells
}

with open('notebooks/122_allfp_adaptive_rdkit.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Written notebooks/122_allfp_adaptive_rdkit.ipynb")
