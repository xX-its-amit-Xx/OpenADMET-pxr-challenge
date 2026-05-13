import json

cells = []

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": [src]}

cells.append(md("# nb120 - Delta-ML: Full RDKit Descriptors + RDKit FP\n\nExtends nb117 (OOF 0.2333) with:\n1. Full RDKit descriptor delta (217 props) replacing 11-prop physchem\n2. RDKit topological FPs (2048 bits -> 64-dim compressed) added\n\nTotal feature dim: 5*128 + 32 + 1 + 1 + 217 = 891 features (vs 557 in nb117)."))

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
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors
from pxr.data import load_train, load_test
from pxr.featurize import combined, rdkit_desc, impute
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

print("FP batch functions ready.")
"""))

cells.append(code("""tr = load_train(); te = load_test()
y_tr = tr["pec50"].values.astype(np.float64)
scaffolds = tr["smiles"].map(bemis_murcko).tolist()
splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

X_tr = impute(combined(tr["smiles"].tolist()))
X_te = impute(combined(te["smiles"].tolist()))

# Full RDKit descriptors (217 props) used as physchem delta
print("Computing RDKit descriptors (217 props)...", flush=True)
rdkit_raw_tr = rdkit_desc(tr["smiles"].tolist())
rdkit_raw_te = rdkit_desc(te["smiles"].tolist())
# Impute using training median
tr_medians = np.nanmedian(rdkit_raw_tr, axis=0)
rdkit_imp_tr = np.where(np.isfinite(rdkit_raw_tr), rdkit_raw_tr, tr_medians[None, :]).astype(np.float32)
rdkit_imp_te = np.where(np.isfinite(rdkit_raw_te), rdkit_raw_te, tr_medians[None, :]).astype(np.float32)
# Normalize by training std to make delta comparable across descriptors
tr_std = rdkit_imp_tr.std(0) + 1e-8
rdkit_norm_tr = rdkit_imp_tr / tr_std
rdkit_norm_te = rdkit_imp_te / tr_std
print(f"  RDKit desc shape: {rdkit_imp_tr.shape}")

print("Computing ECFP4...", flush=True)
fps4_tr = ecfp4_batch(tr["smiles"].tolist())
fps4_te = ecfp4_batch(te["smiles"].tolist())
print("Computing ECFP6...", flush=True)
fps6_tr = ecfp6_batch(tr["smiles"].tolist())
fps6_te = ecfp6_batch(te["smiles"].tolist())
print("Computing MACCS...", flush=True)
maccs_tr = maccs_batch(tr["smiles"].tolist())
maccs_te = maccs_batch(te["smiles"].tolist())
print("Computing Atom-pair FPs...", flush=True)
ap_tr = atom_pair_batch(tr["smiles"].tolist())
ap_te = atom_pair_batch(te["smiles"].tolist())
print("Computing Topological-Torsion FPs...", flush=True)
tt_tr = topo_torsion_batch(tr["smiles"].tolist())
tt_te = topo_torsion_batch(te["smiles"].tolist())
print("Computing RDKit topological FPs...", flush=True)
rdfp_tr = rdkit_fp_batch(tr["smiles"].tolist())
rdfp_te = rdkit_fp_batch(te["smiles"].tolist())

cliff_pairs = (pd.read_parquet(DATA_PROCESSED/"cliff_pairs.parquet")
               if (DATA_PROCESSED/"cliff_pairs.parquet").exists() else pd.DataFrame())
print(f"Train {len(tr):,}  Test {len(te):,}  Cliffs {len(cliff_pairs)}")
"""))

cells.append(code("""print("Computing pairwise Tanimoto...", flush=True)
dot_tt = (fps4_tr @ fps4_tr.T).astype(np.float32)
rowsum = fps4_tr.sum(1).astype(np.float32)
union_tt = rowsum[:,None] + rowsum[None,:] - dot_tt
tanimoto_tr = np.where(union_tt>0, dot_tt/union_tt, 0.0)
np.fill_diagonal(tanimoto_tr, 0.0)

dot_te = (fps4_te @ fps4_tr.T).astype(np.float32)
rs_te = fps4_te.sum(1)[:,None]; rs_tr_v = fps4_tr.sum(1)[None,:]
sim_te_tr = dot_te / np.maximum(rs_te + rs_tr_v - dot_te, 1e-6)
print("Tanimoto done.")
"""))

cells.append(code("""def compress_fp(fp, out_dim=64):
    N, D = fp.shape; block = D // out_dim
    return fp[:, :block*out_dim].reshape(N, out_dim, block).mean(-1).astype(np.float32)

def compress_maccs(fp, out_dim=32):
    N, D = fp.shape; block = D // out_dim
    return fp[:, :block*out_dim].reshape(N, out_dim, block).mean(-1).astype(np.float32)

def make_full_delta_feats(fp4_a, fp4_q, fp6_a, fp6_q, maccs_a, maccs_q,
                           ap_a, ap_q, tt_a, tt_q, rdfp_a, rdfp_q,
                           sim_col, anchor_pec50, rdkit_diff):
    def common_diff(fa, fb, outdim=64):
        c = np.minimum(fa, fb).astype(np.float32)
        d = np.abs(fa - fb).astype(np.float32)
        return compress_fp(c, outdim), compress_fp(d, outdim)

    c4,  d4  = common_diff(fp4_a,  fp4_q)
    c6,  d6  = common_diff(fp6_a,  fp6_q)
    cap, dap = common_diff(ap_a,   ap_q)
    ctt, dtt = common_diff(tt_a,   tt_q)
    crd, drd = common_diff(rdfp_a, rdfp_q)

    maccs_d = np.abs(maccs_a - maccs_q).astype(np.float32)
    cm = compress_maccs(maccs_d, 32)

    # 5*128 + 32 + 1 + 1 + 217 = 891 features
    return np.hstack([c4, d4, c6, d6, cap, dap, ctt, dtt, crd, drd, cm,
                      sim_col, anchor_pec50[:,None], rdkit_diff])

_test = make_full_delta_feats(
    fps4_tr[:2], fps4_tr[2:4], fps6_tr[:2], fps6_tr[2:4],
    maccs_tr[:2], maccs_tr[2:4], ap_tr[:2], ap_tr[2:4],
    tt_tr[:2], tt_tr[2:4], rdfp_tr[:2], rdfp_tr[2:4],
    np.ones((2,1)), y_tr[:2],
    rdkit_norm_tr[:2] - rdkit_norm_tr[2:4]
)
print(f"Full-desc delta feature dim: {_test.shape[1]}  (expect ~891)")
"""))

cells.append(code("""TIERS = {"HIGH": (0.60, 0.90), "MED": (0.45, 0.60), "LOW": (0.35, 0.45)}
i_idx_gl, j_idx_gl = np.where(np.triu(tanimoto_tr > 0.30, k=1))
sim_gl = tanimoto_tr[i_idx_gl, j_idx_gl]

def build_tier(tier_name, sim_lo, sim_hi):
    if tier_name == "HIGH": mask = (sim_gl >= sim_lo) & (sim_gl <= sim_hi)
    else: mask = (sim_gl >= sim_lo) & (sim_gl < sim_hi)
    ii, jj = i_idx_gl[mask], j_idx_gl[mask]
    sim_ij = sim_gl[mask][:,None]
    rd_diff_ij = rdkit_norm_tr[jj] - rdkit_norm_tr[ii]
    F_ij = make_full_delta_feats(
        fps4_tr[ii], fps4_tr[jj], fps6_tr[ii], fps6_tr[jj],
        maccs_tr[ii], maccs_tr[jj], ap_tr[ii], ap_tr[jj],
        tt_tr[ii], tt_tr[jj], rdfp_tr[ii], rdfp_tr[jj],
        sim_ij, y_tr[ii], rd_diff_ij)
    F_ji = make_full_delta_feats(
        fps4_tr[jj], fps4_tr[ii], fps6_tr[jj], fps6_tr[ii],
        maccs_tr[jj], maccs_tr[ii], ap_tr[jj], ap_tr[ii],
        tt_tr[jj], tt_tr[ii], rdfp_tr[jj], rdfp_tr[ii],
        sim_ij, y_tr[jj], -rd_diff_ij)
    F = np.vstack([F_ij, F_ji])
    y = np.concatenate([y_tr[jj]-y_tr[ii], y_tr[ii]-y_tr[jj]])
    return F, y, len(ii)

for tier, (lo, hi) in TIERS.items():
    _, _, n = build_tier(tier, lo, hi)
    print(f"  {tier} [{lo},{hi}]: {n:,} pairs")
"""))

cells.append(code("""DELTA_LGBM = dict(n_estimators=1200, num_leaves=63, learning_rate=0.04,
                  min_child_samples=15, subsample=0.8, colsample_bytree=0.7,
                  reg_alpha=0.05, reg_lambda=0.1, random_state=SEED, verbose=-1, n_jobs=4)

tier_models = {}
for tier, (lo, hi) in TIERS.items():
    F, y, n = build_tier(tier, lo, hi)
    print(f"Training {tier} on {len(F):,} pairs...", flush=True)
    m = lgb.LGBMRegressor(**DELTA_LGBM)
    m.fit(F, y, callbacks=[lgb.log_evaluation(-1)])
    tier_models[tier] = m
    print(f"  {tier} done.", flush=True)
"""))

cells.append(code("""K_NEIGHBORS = 10

def predict_full(fps4_q, fps4_r, fps6_q, fps6_r, maccs_q, maccs_r,
                 ap_q, ap_r, tt_q, tt_r, rdfp_q, rdfp_r,
                 rdkit_q, rdkit_r, y_ref, sim_matrix, fallback_preds):
    N = len(fps4_q)
    preds = np.full(N, np.nan)
    tier_counts = {t: 0 for t in TIERS}; tier_counts["fallback"] = 0

    for qi in range(N):
        sim_row = sim_matrix[qi]
        assigned = False
        for tier, (lo, hi) in TIERS.items():
            if tier == "HIGH": cand_mask = (sim_row >= lo) & (sim_row <= hi)
            else: cand_mask = (sim_row >= lo) & (sim_row < hi)
            cand_idx = np.where(cand_mask)[0]
            if len(cand_idx) == 0: continue

            top_k = np.argsort(-sim_row[cand_idx])[:K_NEIGHBORS]
            sel_idx = cand_idx[top_k]
            cand_sims = sim_row[sel_idx]

            fp4q = np.tile(fps4_q[qi:qi+1], (len(sel_idx),1))
            fp6q = np.tile(fps6_q[qi:qi+1], (len(sel_idx),1))
            macq = np.tile(maccs_q[qi:qi+1], (len(sel_idx),1))
            apq  = np.tile(ap_q[qi:qi+1],   (len(sel_idx),1))
            ttq  = np.tile(tt_q[qi:qi+1],   (len(sel_idx),1))
            rdq  = np.tile(rdfp_q[qi:qi+1], (len(sel_idx),1))
            rd_diff = rdkit_q[qi:qi+1] - rdkit_r[sel_idx]

            F_k = make_full_delta_feats(
                fps4_r[sel_idx], fp4q, fps6_r[sel_idx], fp6q,
                maccs_r[sel_idx], macq, ap_r[sel_idx], apq,
                tt_r[sel_idx], ttq, rdfp_r[sel_idx], rdq,
                cand_sims[:,None], y_ref[sel_idx], rd_diff)
            delta_k = tier_models[tier].predict(F_k)
            template_preds = y_ref[sel_idx] + delta_k
            weights = cand_sims ** 2
            preds[qi] = np.average(template_preds, weights=weights)
            tier_counts[tier] += 1
            assigned = True
            break

        if not assigned:
            preds[qi] = fallback_preds[qi]
            tier_counts["fallback"] += 1

    return preds, tier_counts

print("Predict function ready.")
"""))

cells.append(code("""print("\\n=== Scaffold 5-fold CV ===", flush=True)
oof_delta = np.full(len(y_tr), np.nan)
oof_direct = np.full(len(y_tr), np.nan)

for fold, (tr_idx, va_idx) in enumerate(splits):
    m_dir = lgb.train(LGBM_BASE, lgb.Dataset(X_tr[tr_idx], label=y_tr[tr_idx]),
                      valid_sets=[lgb.Dataset(X_tr[va_idx], label=y_tr[va_idx])],
                      callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(-1)])
    oof_direct[va_idx] = m_dir.predict(X_tr[va_idx])

    fps4_va = fps4_tr[va_idx]; fps4_ft = fps4_tr[tr_idx]
    dot_vf = (fps4_va @ fps4_ft.T).astype(np.float32)
    rs_v = fps4_va.sum(1)[:,None]; rs_f = fps4_ft.sum(1)[None,:]
    sim_vf = dot_vf / np.maximum(rs_v + rs_f - dot_vf, 1e-6)

    preds_d, tc = predict_full(
        fps4_tr[va_idx], fps4_tr[tr_idx],
        fps6_tr[va_idx], fps6_tr[tr_idx],
        maccs_tr[va_idx], maccs_tr[tr_idx],
        ap_tr[va_idx],   ap_tr[tr_idx],
        tt_tr[va_idx],   tt_tr[tr_idx],
        rdfp_tr[va_idx], rdfp_tr[tr_idx],
        rdkit_norm_tr[va_idx], rdkit_norm_tr[tr_idx],
        y_tr[tr_idx], sim_vf, oof_direct[va_idx])
    oof_delta[va_idx] = preds_d

    r_dir = rae(y_tr[va_idx], oof_direct[va_idx])
    r_dlt = rae(y_tr[va_idx], oof_delta[va_idx])
    print(f"  fold {fold+1}  direct={r_dir:.4f}  full_desc_delta={r_dlt:.4f}  tiers={tc}", flush=True)

m_dir = full_metrics(y_tr, oof_direct, "direct_lgbm")
m_dlt = full_metrics(y_tr, oof_delta,  "full_desc_delta_3tier")

nb117_path = DATA_PROCESSED / "oof_allfp_delta_3tier.npy"
if nb117_path.exists():
    r117 = rae(y_tr, np.load(nb117_path))
    print(f"\\nnb117 (5 FPs + 11 physchem): {r117:.4f}")
    print(f"nb120 (6 FPs + 217 rdkit):   {m_dlt['RAE']:.4f}")
    print(f"Delta: {m_dlt['RAE'] - r117:+.4f}")
print(f"\\n*** nb120 OOF RAE = {m_dlt['RAE']:.4f} ***")
"""))

cells.append(code("""print("\\nFitting final direct LGBM...", flush=True)
m_final = lgb.train(LGBM_BASE, lgb.Dataset(X_tr, label=y_tr), callbacks=[lgb.log_evaluation(-1)])
te_direct = m_final.predict(X_te)

print("Running full-desc delta on test...", flush=True)
te_delta, te_tc = predict_full(
    fps4_te, fps4_tr, fps6_te, fps6_tr,
    maccs_te, maccs_tr, ap_te, ap_tr, tt_te, tt_tr,
    rdfp_te, rdfp_tr, rdkit_norm_te, rdkit_norm_tr,
    y_tr, sim_te_tr, te_direct)
print(f"Test tier usage: {te_tc}")

te_preds = np.clip(te_delta, y_tr.min()-0.5, y_tr.max()+0.5)

np.save(DATA_PROCESSED/"oof_full_desc_delta_3tier.npy", oof_delta)
np.save(DATA_PROCESSED/"te_oof_full_desc_delta_3tier.npy", te_preds)
sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_preds})
assert len(sub)==513 and sub["pEC50"].notna().all()
p = SUBMISSIONS/"120_delta_full_rdkit_desc.csv"; sub.to_csv(p, index=False)
print(f"Saved {p}")
print(f"Test: min={te_preds.min():.2f} med={np.median(te_preds):.2f} max={te_preds.max():.2f}")
print(f"\\n*** nb120 OOF RAE = {m_dlt['RAE']:.4f} ***")
"""))

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "pxr-challenge", "language": "python", "name": "pxr-challenge"},
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "cells": cells
}

with open('notebooks/120_delta_full_rdkit_desc.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Written notebooks/120_delta_full_rdkit_desc.ipynb")
