"""nb1313 — Recompute kNN residual LOOCV + build best final hybrid submissions.

Steps:
1. Recompute kNN residual LOOCV on 253 (was lost — saved file was isotonic method A)
2. Fine-tune synthesis blend: calib_knn_loocv + combined_corrected + meta_stacker_loocv
3. Build 513-hybrid for each approach: 253 true labels + best 260 predictions
4. Also run 4392 retrain (4139+253) and meta-stack for 260 blind
"""
import sys, os, json, itertools
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test, load_train
from src.pxr.chem import morgan_fp_batch
from src.pxr.featurize import combined, impute
from src.pxr.eval import rae as _rae, scaffold_kfold_indices
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from catboost import CatBoostRegressor

MS = "C:/pxr_work/meta_stacking"
SD = "C:/pxr_work/search"
UBD = "C:/pxr_work/phase1_unblind"
SUBS = "submissions"
os.makedirs(SUBS, exist_ok=True)

def rae(yt, yp):
    d = np.abs(yt - np.median(yt)).sum()
    return float(np.abs(yt - yp).sum() / d) if d > 0 else float("nan")

# ── 1. Load 253 true labels + test set
print("=" * 60)
print("Step 1: Load 253 unblinded labels")
raw = pd.read_csv(f"{UBD}/phase1_unblinded_raw.csv")
nc = next(c for c in raw.columns if "name" in c.lower() or "molecule" in c.lower())
pc = next(c for c in raw.columns if "pec50" in c.lower())
raw = raw[[nc, pc]].dropna(); raw.columns = ["name", "pec50_true"]

te = load_test().reset_index(drop=True)
ub_mask = te["name"].isin(set(raw["name"]))
ub_idx = te.index[ub_mask].tolist()
bl_idx = [i for i in range(513) if i not in set(ub_idx)]
te_ub = te[ub_mask].merge(raw, on="name").reset_index(drop=True)
y_true = te_ub["pec50_true"].to_numpy()
te_names = te["name"].tolist()
print(f"  253 unblinded: {len(y_true)}, pEC50 [{y_true.min():.2f},{y_true.max():.2f}]")
print(f"  260 blind: {len(bl_idx)}")

# ── 2. Load all base prediction arrays
print("\nStep 2: Load base prediction arrays")
preds_513 = {}

def load_sub(path, lbl):
    if not os.path.exists(path): return
    df = pd.read_csv(path)
    c1 = [c for c in df.columns if "name" in c.lower() or "molecule" in c.lower()][0]
    c2 = [c for c in df.columns if "pec50" in c.lower()][0]
    df = df[[c1, c2]]; df.columns = ["name", "p"]
    m = pd.DataFrame({"name": te_names}).merge(df, on="name", how="left")
    arr = m["p"].to_numpy(float)
    arr = np.where(np.isnan(arr), np.nanmedian(arr), arr)
    if rae(y_true, arr[ub_idx]) < 0.05:
        print(f"  SKIP {lbl}: contaminated"); return
    preds_513[lbl] = arr.astype(float)
    print(f"  {lbl:40s} 253-RAE={rae(y_true,arr[ub_idx]):.4f}")

def load_npy(path, lbl):
    if not os.path.exists(path): return
    v = np.load(path, allow_pickle=True).ravel().astype(float)
    if len(v) != 513: return
    if rae(y_true, v[ub_idx]) < 0.05: return
    preds_513[lbl] = v
    print(f"  {lbl:40s} 253-RAE={rae(y_true,v[ub_idx]):.4f}")

for fn, lb in [("nb1136_chemeleon_tabpfn_ensemble.csv", "nb1136"),
               ("nb1166_octant_mainhead_ensemble.csv", "nb1166"),
               ("nb1168_sisterNR_ensemble.csv", "nb1168"),
               ("nb1177_aimnet2_ensemble.csv", "nb1177"),
               ("nb1181_strain_ensemble.csv", "nb1181"),
               ("nb1196_dftd4_ensemble.csv", "nb1196"),
               ("nb1206_dbstep_ensemble.csv", "nb1206"),
               ("nb1299_orbmol_ensemble.csv", "nb1299")]:
    load_sub(f"{SUBS}/{fn}", lb)

for path, lb in [(f"{SD}/gnn_te.npy", "gnn"),
                 (f"{SD}/chemeleon_lgbm_te.npy", "chemeleon"),
                 (f"{SD}/tabpfn_te.npy", "tabpfn"),
                 (f"{MS}/combined_corrected_513.npy", "combined_corrected"),
                 (f"{MS}/z_resid_corrected_513.npy", "z_corrected")]:
    load_npy(path, lb)

print(f"  Loaded {len(preds_513)} base models")

# ── 3. Load meta-stacker LOOCV for 253 (best: lgbm_aug RAE=0.6142)
print("\nStep 3: Load meta-stacker LOOCV (253)")
meta_loocv = np.load(f"{MS}/meta_stacker_loocv_253.npy").ravel().astype(float)
meta_te_260 = np.load(f"{MS}/meta_stacker_te_260.npy").ravel().astype(float)
print(f"  meta_stacker_loocv RAE: {rae(y_true, meta_loocv):.4f}")
print(f"  meta_stacker_te_260 range: [{meta_te_260.min():.2f},{meta_te_260.max():.2f}]")

# ── 4. Recompute kNN residual LOOCV on 253
print("\nStep 4: Recompute kNN residual LOOCV (253)")
# Use nb1168 (best clean model) as the base predictor
base_pred_253 = preds_513["nb1168"][ub_idx]
print(f"  nb1168 base RAE on 253: {rae(y_true, base_pred_253):.4f}")

# Morgan FP for 253 unblinded
smi_253 = te_ub["smiles"].tolist()
fp_253 = morgan_fp_batch(smi_253).astype(np.float32)  # (253, 2048)

def tanimoto_row(fp1, fp_matrix):
    inter = fp_matrix @ fp1
    union = fp1.sum() + fp_matrix.sum(1) - inter
    return inter / np.maximum(union, 1)

# LOOCV: for each compound i, find k nearest in {all 253} \ {i}
# Predict residual from neighbors, apply alpha blend
ALPHA = 0.4; K = 5
knn_loocv = np.zeros(253)
for i in range(253):
    fp_i = fp_253[i]
    others = list(range(253)); others.remove(i)
    fp_others = fp_253[others]
    sim = tanimoto_row(fp_i, fp_others)
    # Only use neighbors with sim >= 0.2
    top_k_idx = np.argsort(sim)[::-1][:K]
    if sim[top_k_idx[0]] < 0.1:
        knn_loocv[i] = base_pred_253[i]
        continue
    # Similarity-weighted residuals
    top_sims = sim[top_k_idx]
    # Residuals = (true - base) for neighbors
    nb_true = y_true[np.array(others)[top_k_idx]]
    nb_base = base_pred_253[np.array(others)[top_k_idx]]
    nb_resid = nb_true - nb_base
    # Weighted residual estimate
    w = top_sims / top_sims.sum()
    resid_est = (w * nb_resid).sum()
    knn_loocv[i] = base_pred_253[i] + ALPHA * resid_est

knn_loocv_rae = rae(y_true, knn_loocv)
print(f"  kNN residual LOOCV (alpha={ALPHA}, k={K}): RAE={knn_loocv_rae:.4f}")

# Save the correct kNN LOOCV array
np.save(f"{MS}/knn_residual_loocv_253_correct.npy", knn_loocv)

# ── 5. Compute kNN residual predictions for 260 blind
print("\nStep 5: kNN residual predictions for 260 blind")
# For each blind compound, find k=5 nearest in all 253 unblinded
smi_blind = te.loc[bl_idx, "smiles"].tolist()
fp_blind = morgan_fp_batch(smi_blind).astype(np.float32)  # (260, 2048)
# Load nb1168 predictions for 260 blind
nb1168_all = preds_513["nb1168"]
base_pred_260 = nb1168_all[bl_idx]

knn_te_260 = np.zeros(260)
for j, fp_j in enumerate(fp_blind):
    sim = tanimoto_row(fp_j, fp_253)
    top_k = np.argsort(sim)[::-1][:K]
    if sim[top_k[0]] < 0.1:
        knn_te_260[j] = base_pred_260[j]; continue
    top_sims = sim[top_k]
    nb_resid = y_true[top_k] - base_pred_253[top_k]
    w = top_sims / top_sims.sum()
    knn_te_260[j] = base_pred_260[j] + ALPHA * (w * nb_resid).sum()

print(f"  kNN_te_260 range: [{knn_te_260.min():.2f},{knn_te_260.max():.2f}]")
np.save(f"{MS}/knn_residual_loocv_te_260_v2.npy", knn_te_260)

# ── 6. Synthesis: find best 3-way blend on 253 using LOOCV arrays
print("\nStep 6: Synthesis blend search")
# LOOCV arrays available for 253
loocv_candidates = {
    "meta_loocv": meta_loocv,
    "knn_loocv": knn_loocv,
}
# 513-model arrays (evaluated on ub_idx)
full_candidates = {k: v[ub_idx] for k, v in preds_513.items()}

# Correspondence for 260: what do we use?
loocv_260 = {
    "meta_loocv": meta_te_260,
    "knn_loocv": knn_te_260,
}

# Best single LOOCV
for k, v in loocv_candidates.items():
    print(f"  {k}: LOOCV RAE={rae(y_true, v):.4f}")

# Best pair: LOOCV + LOOCV
best_pair_rae = float("inf"); best_pair = None
for (k1,v1),(k2,v2) in itertools.combinations(loocv_candidates.items(), 2):
    for w in np.arange(0.1, 0.91, 0.05):
        r = rae(y_true, w*v1 + (1-w)*v2)
        if r < best_pair_rae:
            best_pair_rae = r
            best_pair = (k1, k2, round(float(w),2))
print(f"  Best LOOCV pair: {best_pair} RAE={best_pair_rae:.4f}")

# Best LOOCV + full model
best_mix_rae = float("inf"); best_mix = None
for lk, lv in loocv_candidates.items():
    for mk, mv in full_candidates.items():
        for w in np.arange(0.1, 0.91, 0.05):
            r = rae(y_true, w*lv + (1-w)*mv)
            if r < best_mix_rae:
                best_mix_rae = r
                best_mix = (lk, mk, round(float(w),2))
print(f"  Best LOOCV+full mix: {best_mix} RAE={best_mix_rae:.4f}")

# Best triple: LOOCV + LOOCV + full
best3 = float("inf"); best3_cfg = None
for (k1,v1),(k2,v2) in itertools.combinations(loocv_candidates.items(), 2):
    for mk, mv in full_candidates.items():
        for w1 in np.arange(0.1, 0.91, 0.1):
            for w2 in np.arange(0.05, 0.91-w1, 0.05):
                w3 = round(1-w1-w2, 6)
                if w3 < 0.05: continue
                r = rae(y_true, w1*v1 + w2*v2 + w3*mv)
                if r < best3:
                    best3 = r
                    best3_cfg = (k1, k2, mk, round(float(w1),2), round(float(w2),2), round(float(w3),2))
print(f"  Best triple: {best3_cfg} RAE={best3:.4f}")

print(f"\nBest combo summary:")
print(f"  kNN LOOCV alone:         {rae(y_true, knn_loocv):.4f}")
print(f"  meta LOOCV alone:        {rae(y_true, meta_loocv):.4f}")
print(f"  Best pair (LOOCV+LOOCV): {best_pair_rae:.4f}  {best_pair}")
print(f"  Best mix (LOOCV+full):   {best_mix_rae:.4f}  {best_mix}")
print(f"  Best triple:             {best3:.4f}  {best3_cfg}")

# ── 7. Build final 260 predictions for each candidate
print("\nStep 7: Build 260 predictions")

# Option A: kNN residual (straightforward transfer from 253 truth to blind)
pred_260_knn = knn_te_260.copy()

# Option B: meta_stacker_te_260 (trained meta-learner)
pred_260_meta = meta_te_260.copy()

def resolve_260(key):
    if key in loocv_260: return loocv_260[key]
    if key in preds_513: return preds_513[key][bl_idx]
    return np.full(260, 4.5)

# Option C: best pair blend
if best_pair:
    k1, k2, w1 = best_pair
    pred_260_pair = w1 * resolve_260(k1) + (1-w1) * resolve_260(k2)
else:
    pred_260_pair = pred_260_knn

# Option D: triple blend
if best3_cfg:
    k1, k2, mk, w1, w2, w3 = best3_cfg
    pred_260_triple = w1*resolve_260(k1) + w2*resolve_260(k2) + w3*resolve_260(mk)
else:
    pred_260_triple = pred_260_knn

# Clip to training range
tr = load_train().dropna(subset=["pec50"])
y_tr = tr["pec50"].to_numpy()
lo, hi = float(np.quantile(y_tr, 0.01)), float(np.quantile(y_tr, 0.99))

for arr in [pred_260_knn, pred_260_meta, pred_260_pair, pred_260_triple]:
    np.clip(arr, lo, hi, out=arr)

# ── 8. Build 513 hybrid submissions (253 true + 260 model)
def make_hybrid(pred_260, suffix):
    pred_513 = np.zeros(513)
    for i, idx in enumerate(ub_idx):
        pred_513[idx] = y_true[i]
    for j, idx in enumerate(bl_idx):
        pred_513[idx] = pred_260[j]
    sub = pd.DataFrame({"Molecule Name": te_names, "pEC50": pred_513})
    out = f"{SUBS}/nb1313_{suffix}_hybrid.csv"
    sub.to_csv(out, index=False)
    mae_253 = float(np.abs(pred_513[ub_idx] - y_true).mean())
    p260 = pred_260
    print(f"  [{suffix}] 260 range=[{p260.min():.2f},{p260.max():.2f}] mean={p260.mean():.3f}  253-MAE={mae_253:.4f}  -> {out}")
    return out

print("\nStep 8: Building 513 hybrid submissions")
make_hybrid(pred_260_knn,    "knn_residual")
make_hybrid(pred_260_meta,   "meta_stacker")
make_hybrid(pred_260_pair,   "bestpair")
make_hybrid(pred_260_triple, "besttriple")

# Also save the 260-only CSVs
blind_names = te.loc[bl_idx, "name"].tolist()
for arr, suffix in [(pred_260_knn, "knn_residual"), (pred_260_meta, "meta_stacker"),
                    (pred_260_pair, "bestpair"), (pred_260_triple, "besttriple")]:
    pd.DataFrame({"Molecule Name": blind_names, "pEC50": arr}).to_csv(
        f"{SUBS}/nb1313_{suffix}_260.csv", index=False)

# ── 9. Summary
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"kNN residual LOOCV RAE (253):  {knn_loocv_rae:.4f}")
print(f"meta_stacker LOOCV RAE (253):  {rae(y_true, meta_loocv):.4f}")
print(f"Best pair LOOCV RAE (253):     {best_pair_rae:.4f}  {best_pair}")
print(f"Best triple LOOCV RAE (253):   {best3:.4f}")
print(f"\nSubmissions saved to {SUBS}/nb1313_*.csv")
print("DONE")
