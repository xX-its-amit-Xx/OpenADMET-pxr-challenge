"""nb1315 - Comprehensive evaluation across PXR-253 unblinded, mirror NRs,
and holdout. Tests how our best pipeline (GBM combined + meta-correction)
generalises across nuclear receptors and data regimes.

Outputs: data/processed/nb1315_mirror_eval.json
"""
import sys, os, json
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train, load_test
from src.pxr.featurize import combined, impute
from src.pxr.eval import scaffold_kfold_indices, rae as _rae
from src.pxr.chem import morgan_fp_batch
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from catboost import CatBoostRegressor
RDLogger.DisableLog("rdApp.*")

OUT = "data/processed/nb1315_mirror_eval.json"
MS  = "C:/pxr_work/meta_stacking"

def rae(yt, yp):
    d = np.abs(yt - np.median(yt)).sum()
    return float(np.abs(yt - yp).sum() / d) if d > 0 else float("nan")

def murcko(s):
    m = Chem.MolFromSmiles(str(s))
    return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""

def gbm_scaffold_cv(smiles, y, n_seeds=3, n_splits=5):
    """4-GBM ensemble scaffold-CV, averaged over seeds. Returns OOF RAE."""
    scaffolds = [murcko(s) for s in smiles]
    X = impute(combined(smiles)).astype(np.float32)
    seed_raes = []
    for seed in range(n_seeds):
        folds = scaffold_kfold_indices(scaffolds, n_splits=n_splits, seed=seed)
        oof = np.zeros(len(y))
        for trn, val in folds:
            mdls = [
                LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05,
                              n_jobs=4, verbose=-1, random_state=seed),
                XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                             n_jobs=4, verbosity=0, tree_method="hist",
                             random_state=seed),
                HistGradientBoostingRegressor(max_iter=500, max_leaf_nodes=64,
                                              learning_rate=0.05, random_state=seed),
                CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05,
                                  verbose=0, random_state=seed),
            ]
            preds = []
            for m in mdls:
                m.fit(X[trn], y[trn]); preds.append(m.predict(X[val]))
            oof[val] = np.mean(preds, axis=0)
        seed_raes.append(rae(y, oof))
    return float(np.mean(seed_raes)), float(np.std(seed_raes)), seed_raes

def knn_loocv_correction(smiles, y, base_oof, k=5, alpha=0.4):
    """Apply kNN residual LOOCV correction on top of base OOF predictions."""
    fp = morgan_fp_batch(smiles).astype(np.float32)
    corrected = np.zeros(len(y))
    for i in range(len(y)):
        fp_i = fp[i]
        others = list(range(len(y))); others.remove(i)
        fp_o = fp[others]
        inter = fp_o @ fp_i
        union = fp_i.sum() + fp_o.sum(1) - inter
        sim = inter / np.maximum(union, 1)
        top_k = np.argsort(sim)[::-1][:k]
        if sim[top_k[0]] < 0.1:
            corrected[i] = base_oof[i]; continue
        sims = sim[top_k]
        nb_resid = y[np.array(others)[top_k]] - base_oof[np.array(others)[top_k]]
        w = sims / sims.sum()
        corrected[i] = base_oof[i] + alpha * (w * nb_resid).sum()
    return corrected

def noise_floor_estimate(smiles, y, frac_near_dup=0.3):
    """Estimate noise floor via near-duplicate pairs (top-10% sim, |Deltay| std)."""
    fp = morgan_fp_batch(smiles[:500]).astype(np.float32)  # subsample for speed
    sim_mat = (fp @ fp.T)
    denom = fp.sum(1)[:, None] + fp.sum(1)[None, :] - sim_mat
    sim_mat = sim_mat / np.maximum(denom, 1)
    np.fill_diagonal(sim_mat, 0)
    high_sim_pairs = np.argwhere(sim_mat > 0.7)
    if len(high_sim_pairs) == 0:
        return float("nan")
    delta_y = [abs(y[i] - y[j]) for i, j in high_sim_pairs if i < j]
    return float(np.mean(delta_y)) if delta_y else float("nan")

results = {}

# ─── 1. PXR 253 unblinded (our honest test) ──────────────────────────────────
print("=" * 60)
print("1. PXR 253 unblinded (ground truth)")
raw = pd.read_csv("C:/pxr_work/phase1_unblind/phase1_unblinded_raw.csv")
nc = next(c for c in raw.columns if "name" in c.lower() or "molecule" in c.lower())
pc = next(c for c in raw.columns if "pec50" in c.lower())
raw = raw[[nc, pc]].dropna(); raw.columns = ["name", "pec50_true"]
te = load_test().reset_index(drop=True)
ub_mask = te["name"].isin(set(raw["name"]))
ub_idx = te.index[ub_mask].tolist()
te_ub = te[ub_mask].merge(raw, on="name").reset_index(drop=True)
y_pxr = te_ub["pec50_true"].to_numpy()
smi_pxr = te_ub["smiles"].tolist()

meta_loocv = np.load(f"{MS}/meta_stacker_loocv_253.npy").ravel().astype(float)
knn_loocv  = np.load(f"{MS}/knn_residual_loocv_253_correct.npy").ravel().astype(float)
combined_corr = np.load(f"{MS}/combined_corrected_513.npy").ravel()[ub_idx].astype(float)

rae_meta  = rae(y_pxr, meta_loocv)
rae_knn   = rae(y_pxr, knn_loocv)
rae_triple = rae(y_pxr, 0.6*meta_loocv + 0.15*knn_loocv + 0.25*combined_corr)
rae_gbm_direct, _, _ = gbm_scaffold_cv(smi_pxr, y_pxr, n_seeds=1, n_splits=5)

print(f"  n=253, pEC50 [{y_pxr.min():.2f},{y_pxr.max():.2f}], median={np.median(y_pxr):.2f}")
print(f"  RAE - meta_stacker_loocv: {rae_meta:.4f}")
print(f"  RAE - knn_residual_loocv: {rae_knn:.4f}")
print(f"  RAE - triple_blend (0.6m+0.15k+0.25c): {rae_triple:.4f}  <- BEST")
print(f"  RAE - GBM 5-fold scaffold (n=253 itself): {rae_gbm_direct:.4f}")

results["PXR_253_unblinded"] = {
    "n": 253, "target": "PXR", "source": "OpenADMET Phase2 truth",
    "pec50_range": [float(y_pxr.min()), float(y_pxr.max())],
    "pec50_median": float(np.median(y_pxr)),
    "rae_meta_stacker_loocv": round(rae_meta, 4),
    "rae_knn_residual_loocv": round(rae_knn, 4),
    "rae_triple_blend": round(rae_triple, 4),
    "rae_gbm_scaffold5cv_self": round(rae_gbm_direct, 4),
    "note": "Honest evaluation on 253 newly-unblinded test compounds (LOOCV methods)"
}

# ─── 2. PXR 4139 training scaffold CV (reference) ────────────────────────────
print("\n2. PXR 4139 training set (scaffold CV reference)")
tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
y_tr = tr["pec50"].to_numpy(); smi_tr = tr["smiles"].tolist()
rae_tr, std_tr, seed_raes_tr = gbm_scaffold_cv(smi_tr, y_tr, n_seeds=3)
print(f"  n=4139, pEC50 [{y_tr.min():.2f},{y_tr.max():.2f}], median={np.median(y_tr):.2f}")
print(f"  RAE - GBM scaffold 5-fold CV: {rae_tr:.4f} ± {std_tr:.4f}")
results["PXR_4139_train_scaffoldCV"] = {
    "n": 4139, "target": "PXR", "source": "OpenADMET train set",
    "pec50_range": [float(y_tr.min()), float(y_tr.max())],
    "pec50_median": float(np.median(y_tr)),
    "rae_gbm_scaffold5cv": round(rae_tr, 4),
    "rae_std": round(std_tr, 4),
    "seed_raes": [round(r, 4) for r in seed_raes_tr],
    "note": "Internal scaffold CV, 3 seeds"
}

# ─── 3. Mirror NR datasets from ChEMBL NR KB ─────────────────────────────────
kb = pd.read_parquet("data/external/chembl_pxr_nr_kb.parquet")
print("\n=== ChEMBL NR Knowledge Base targets ===")
print(kb["source_target"].value_counts())

for target_key, tname, min_n in [
    ("NR_FXR", "FXR", 500),
    ("NR_PPARg", "PPARg", 500),
    ("NR_RXRa", "RXRa", 200),
    ("NR_LXRa", "LXRa", 200),
    ("NR_VDR", "VDR", 100),
    ("PXR_CHEMBL3401", "PXR_CHEMBL", 100),
]:
    sub = kb[kb["source_target"] == target_key].dropna(subset=["smiles", "pec50_chembl"])
    sub = sub.drop_duplicates(subset=["smiles"]).reset_index(drop=True)
    n = len(sub)
    if n < min_n:
        print(f"\n{tname}: only {n} rows, skip")
        continue

    y_sub = sub["pec50_chembl"].to_numpy(float)
    smi_sub = sub["smiles"].tolist()

    # subsample large datasets to speed up
    if n > 2000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, 2000, replace=False)
        y_sub = y_sub[idx]; smi_sub = [smi_sub[i] for i in idx]
        print(f"\n{tname}: subsampled to 2000 from {n}")
        n = 2000
    else:
        print(f"\n{tname}: n={n}")

    print(f"  pEC50 [{y_sub.min():.2f},{y_sub.max():.2f}], median={np.median(y_sub):.2f}")

    rae_base, std_base, seed_raes_base = gbm_scaffold_cv(smi_sub, y_sub, n_seeds=3)
    print(f"  RAE - GBM scaffold 5-fold CV: {rae_base:.4f} ± {std_base:.4f}")

    # kNN LOOCV correction (subsample for speed on large sets)
    n_knn = min(n, 400)
    rng2 = np.random.default_rng(99)
    knn_idx = rng2.choice(n, n_knn, replace=False) if n > n_knn else np.arange(n)
    y_knn = y_sub[knn_idx]; smi_knn = [smi_sub[i] for i in knn_idx]
    # get base OOF for knn_idx subset
    X_knn = impute(combined(smi_knn)).astype(np.float32)
    scaffolds_knn = [murcko(s) for s in smi_knn]
    folds_knn = scaffold_kfold_indices(scaffolds_knn, n_splits=5, seed=0)
    oof_knn = np.zeros(n_knn)
    for trn, val in folds_knn:
        m = LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05,
                          n_jobs=4, verbose=-1).fit(X_knn[trn], y_knn[trn])
        oof_knn[val] = m.predict(X_knn[val])

    corrected_knn = knn_loocv_correction(smi_knn, y_knn, oof_knn)
    rae_knn_corr = rae(y_knn, corrected_knn)
    print(f"  RAE - kNN residual LOOCV (n={n_knn} subset): {rae_knn_corr:.4f}")
    print(f"  Gain from kNN correction: {rae_knn_corr - rae(y_knn, oof_knn):+.4f}")

    nf_est = noise_floor_estimate(smi_sub, y_sub)
    print(f"  Noise floor estimate (high-sim pair Deltay mean): {nf_est:.4f}")

    results[f"{target_key}_mirror"] = {
        "n": n, "target": tname, "source": f"ChEMBL NR KB ({target_key})",
        "pec50_range": [float(y_sub.min()), float(y_sub.max())],
        "pec50_median": float(np.median(y_sub)),
        "rae_gbm_scaffold5cv": round(rae_base, 4),
        "rae_std": round(std_base, 4),
        "seed_raes": [round(r, 4) for r in seed_raes_base],
        "rae_knn_residual_loocv": round(rae_knn_corr, 4),
        "knn_base_rae": round(rae(y_knn, oof_knn), 4),
        "knn_gain": round(rae_knn_corr - rae(y_knn, oof_knn), 4),
        "noise_floor_hisim_delta": round(nf_est, 4) if not np.isnan(nf_est) else None,
    }

# ─── 4. Holdout from nb318 ──────────────────────────────────────────────────
print("\n4. PXR clean holdout from nb318")
hdf = pd.read_csv("data/processed/nb318_holdout_metrics.csv")
print(f"  nb318_holdout: {len(hdf)} rows, cols={list(hdf.columns[:8])}")

# ─── 5. Summary table ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print(f"{'Dataset':35s} {'N':>5} {'GBM ScCV':>9} {'kNN corr':>9} {'Improve':>8}")
print("-" * 70)
for k, v in results.items():
    gbm_r = v.get("rae_gbm_scaffold5cv", v.get("rae_gbm_scaffold5cv_self", None))
    knn_r = v.get("rae_knn_residual_loocv", None)
    impr  = (knn_r - gbm_r) if (knn_r and gbm_r) else None
    label = f"{k[:35]}"
    print(f"{label:35s} {v['n']:>5}  {gbm_r or 'n/a':>8}  {knn_r or 'n/a':>8}  {impr or 'n/a':>7}")

print("\nNote: kNN residual correction on FXR/PPARg uses LOO across the subset,")
print("so is a fair estimate of real-world gain from this calibration method.")

json.dump(results, open(OUT, "w"), indent=2)
print(f"\nSaved: {OUT}")
print("DONE")
