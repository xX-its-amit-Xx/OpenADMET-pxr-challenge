"""
nb1307_calibration.py
Calibration / residual-correction methods on the 253 unblinded compounds.
Methods: A=Isotonic, B=Quantile Mapping, C=Bias-per-Bin, D=kNN Residual Transfer, E=Local Linear
"""

import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, 'D:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge')
from src.pxr.data import load_test, load_train
from src.pxr.chem import morgan_fp_batch

# ─── paths ────────────────────────────────────────────────────────────────────
PROJ       = 'D:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge'
SUB_DIR    = f'{PROJ}/submissions'
WORK       = 'C:/pxr_work'
OUT_DIR    = f'{WORK}/meta_stacking'
UNBLIND    = f'{WORK}/phase1_unblind'
os.makedirs(OUT_DIR, exist_ok=True)

# ─── 1. Load data ─────────────────────────────────────────────────────────────
print("=== Loading data ===")

# true labels for 253 unblinded
raw = pd.read_csv(f'{UNBLIND}/phase1_unblinded_raw.csv')
print(f"raw unblinded: {raw.shape}")

# Test set (513 rows, order matters for indexing)
test_df = load_test()
print(f"test_df: {test_df.shape}, cols: {test_df.columns.tolist()[:6]}")

# Indices
unblind_idx = np.load(f'{UNBLIND}/unblind_te_idx.npy')   # 253 indices into 513
blind_idx   = np.load(f'{UNBLIND}/blind_te_idx.npy')      # 260 indices into 513
print(f"unblind_idx: {unblind_idx.shape}, blind_idx: {blind_idx.shape}")

# Base predictions — nb1168 (full 513)
nb1168_full = pd.read_csv(f'{SUB_DIR}/nb1168_sisterNR_ensemble.csv')
print(f"nb1168_full: {nb1168_full.shape}")

# nb1306 (260 blind only)
nb1306_blind = pd.read_csv(f'{SUB_DIR}/nb1306_260_blind_nophy_default.csv')
print(f"nb1306_260 blind: {nb1306_blind.shape}")

# Align by molecule name
# nb1168 has cols: SMILES, Molecule Name, pEC50
mol_col = 'Molecule Name' if 'Molecule Name' in nb1168_full.columns else nb1168_full.columns[1]
pred_col = 'pEC50'

nb1168_full = nb1168_full.set_index(mol_col)[pred_col]

# test_df molecule names (positional order)
# detect molecule name column in test_df
test_name_col = [c for c in test_df.columns if 'name' in c.lower() or 'molecule' in c.lower()][0]
print(f"test name col: {test_name_col}")
test_mol_names = test_df[test_name_col].values  # 513 names in order

# Get predictions aligned to test order
pred_513 = nb1168_full.reindex(test_mol_names).values  # 513 preds
print(f"pred_513 NaN count: {np.isnan(pred_513).sum()}")

# True labels for unblinded 253
# raw has 'Molecule Name' and 'pEC50'
raw_indexed = raw.set_index('Molecule Name')['pEC50']
unblind_mol_names = test_mol_names[unblind_idx]
y_true_253 = raw_indexed.reindex(unblind_mol_names).values
y_pred_nb1168_253 = pred_513[unblind_idx]

print(f"y_true_253 NaN: {np.isnan(y_true_253).sum()}, y_pred_253 NaN: {np.isnan(y_pred_nb1168_253).sum()}")
print(f"y_true range: [{y_true_253.min():.3f}, {y_true_253.max():.3f}]")
print(f"y_pred range: [{y_pred_nb1168_253.min():.3f}, {y_pred_nb1168_253.max():.3f}]")

# 260 blind predictions from nb1306
blind_mol_names = test_mol_names[blind_idx]
nb1306_indexed = nb1306_blind.set_index('Molecule Name')['pEC50']
y_pred_blind_260 = nb1306_indexed.reindex(blind_mol_names).values
print(f"y_pred_blind_260 NaN: {np.isnan(y_pred_blind_260).sum()}")

# ─── RAE helper ───────────────────────────────────────────────────────────────
def rae(y_true, y_pred):
    baseline = np.abs(y_true - y_true.mean()).sum()
    if baseline == 0:
        return np.nan
    return np.abs(y_true - y_pred).sum() / baseline

# Baseline RAE on 253
base_rae = rae(y_true_253, y_pred_nb1168_253)
print(f"\nBaseline RAE (nb1168 on 253): {base_rae:.4f}")

# ─── Fingerprints for similarity methods ──────────────────────────────────────
print("\n=== Computing Morgan fingerprints ===")
smiles_col_test = [c for c in test_df.columns if 'smi' in c.lower() or 'smiles' in c.lower()][0]
smiles_253 = test_df.iloc[unblind_idx][smiles_col_test].values
smiles_260 = test_df.iloc[blind_idx][smiles_col_test].values

fp_253 = morgan_fp_batch(smiles_253).astype(np.float32)  # (253, 2048)
fp_260 = morgan_fp_batch(smiles_260).astype(np.float32)  # (260, 2048)
print(f"fp_253: {fp_253.shape}, fp_260: {fp_260.shape}")

def tanimoto_matrix(A, B):
    """Compute Tanimoto similarity between rows of A and B. A: (n,d), B: (m,d) -> (n,m)"""
    # Tanimoto = intersection / union for binary fps
    # intersection = dot product (for 0/1 fps)
    # union = |a| + |b| - intersection
    inter = A @ B.T
    sum_A = A.sum(axis=1, keepdims=True)
    sum_B = B.sum(axis=1, keepdims=True)
    union = sum_A + sum_B.T - inter
    # avoid div by zero
    sim = np.where(union > 0, inter / union, 0.0)
    return sim.astype(np.float32)

print("Computing Tanimoto 253x253...")
sim_253_253 = tanimoto_matrix(fp_253, fp_253)  # (253, 253)
print("Computing Tanimoto 260x253...")
sim_260_253 = tanimoto_matrix(fp_260, fp_253)  # (260, 253)
print("Tanimoto done.")

results = {}

# ─── Method A: Isotonic Regression Calibration (LOOCV) ───────────────────────
print("\n=== Method A: Isotonic Regression (LOOCV) ===")
from sklearn.isotonic import IsotonicRegression

loocv_A = np.zeros(253)
for i in range(253):
    mask = np.ones(253, bool); mask[i] = False
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(y_pred_nb1168_253[mask], y_true_253[mask])
    loocv_A[i] = iso.predict([y_pred_nb1168_253[i]])[0]

rae_A = rae(y_true_253, loocv_A)
print(f"Method A (Isotonic) LOOCV RAE: {rae_A:.4f}")
results['isotonic_loocv_rae'] = float(rae_A)

# Fit on all 253 for applying to 260
iso_full = IsotonicRegression(out_of_bounds='clip')
iso_full.fit(y_pred_nb1168_253, y_true_253)
corr_blind_A = iso_full.predict(y_pred_blind_260)

# ─── Method B: Quantile Mapping (LOOCV) ───────────────────────────────────────
print("\n=== Method B: Quantile Mapping ===")

def quantile_map(y_src, y_ref, y_query):
    """Map y_query from y_src distribution to y_ref distribution via empirical CDF matching."""
    # compute quantile of each query in src distribution
    src_sorted = np.sort(y_src)
    ref_sorted = np.sort(y_ref)
    n_src = len(src_sorted)
    n_ref = len(ref_sorted)
    # for each query, find its quantile in src
    quantiles = np.searchsorted(src_sorted, y_query, side='left') / n_src
    quantiles = np.clip(quantiles, 0, 1)
    # map to ref distribution
    ref_idx = quantiles * (n_ref - 1)
    lo = np.floor(ref_idx).astype(int)
    hi = np.ceil(ref_idx).astype(int)
    lo = np.clip(lo, 0, n_ref - 1)
    hi = np.clip(hi, 0, n_ref - 1)
    frac = ref_idx - lo
    return ref_sorted[lo] * (1 - frac) + ref_sorted[hi] * frac

loocv_B = np.zeros(253)
for i in range(253):
    mask = np.ones(253, bool); mask[i] = False
    mapped = quantile_map(y_pred_nb1168_253[mask], y_true_253[mask], np.array([y_pred_nb1168_253[i]]))
    loocv_B[i] = mapped[0]

rae_B = rae(y_true_253, loocv_B)
print(f"Method B (Quantile Mapping) LOOCV RAE: {rae_B:.4f}")
results['quantile_map_loocv_rae'] = float(rae_B)

# Full fit for 260
corr_blind_B = quantile_map(y_pred_nb1168_253, y_true_253, y_pred_blind_260)

# ─── Method C: Bias Correction per Activity Bin ────────────────────────────────
print("\n=== Method C: Bias Correction per Bin ===")

N_BINS = 5

def bias_correct_loocv_one(i, preds, truths, n_bins=5):
    mask = np.ones(len(preds), bool); mask[i] = False
    p_train = preds[mask]
    t_train = truths[mask]
    # bin edges from pred distribution of training
    bins = np.quantile(p_train, np.linspace(0, 1, n_bins + 1))
    bins[0] -= 1e-6; bins[-1] += 1e-6
    # assign each training point to bin
    bin_labels = np.digitize(p_train, bins) - 1
    bin_labels = np.clip(bin_labels, 0, n_bins - 1)
    # compute bias per bin
    bias_per_bin = np.zeros(n_bins)
    for b in range(n_bins):
        bmask = bin_labels == b
        if bmask.sum() > 0:
            bias_per_bin[b] = (p_train[bmask] - t_train[bmask]).mean()
        # if no points, bias stays 0
    # assign query to bin
    q_bin = int(np.clip(np.digitize([preds[i]], bins)[0] - 1, 0, n_bins - 1))
    return preds[i] - bias_per_bin[q_bin]

loocv_C = np.array([bias_correct_loocv_one(i, y_pred_nb1168_253, y_true_253, N_BINS)
                    for i in range(253)])
rae_C = rae(y_true_253, loocv_C)
print(f"Method C (Bias-per-Bin) LOOCV RAE: {rae_C:.4f}")
results['bias_per_bin_loocv_rae'] = float(rae_C)

# Apply to 260: use all 253 to compute bins and biases
bins_full = np.quantile(y_pred_nb1168_253, np.linspace(0, 1, N_BINS + 1))
bins_full[0] -= 1e-6; bins_full[-1] += 1e-6
bin_labels_full = np.clip(np.digitize(y_pred_nb1168_253, bins_full) - 1, 0, N_BINS - 1)
bias_per_bin_full = np.array([
    (y_pred_nb1168_253[bin_labels_full == b] - y_true_253[bin_labels_full == b]).mean()
    if (bin_labels_full == b).sum() > 0 else 0.0
    for b in range(N_BINS)
])
blind_bin_labels = np.clip(np.digitize(y_pred_blind_260, bins_full) - 1, 0, N_BINS - 1)
corr_blind_C = y_pred_blind_260 - bias_per_bin_full[blind_bin_labels]

# ─── Method D: kNN Residual Transfer ─────────────────────────────────────────
print("\n=== Method D: kNN Residual Transfer ===")

K = 5
# Tune alpha via LOOCV
best_rae_D = np.inf
best_alpha = 0.0

residuals_253 = y_true_253 - y_pred_nb1168_253  # shape (253,)

def knn_residual_loocv(alpha, k=5):
    loocv = np.zeros(253)
    for i in range(253):
        # similarity to other 252 (exclude self which has sim=1.0 on diagonal)
        sims = sim_253_253[i].copy()
        sims[i] = -1.0  # exclude self
        # top-k
        top_k_idx = np.argpartition(sims, -k)[-k:]
        top_k_sim = sims[top_k_idx]
        # handle negative sims
        top_k_sim = np.maximum(top_k_sim, 0)
        w = top_k_sim
        if w.sum() < 1e-9:
            w = np.ones(k) / k
        else:
            w = w / w.sum()
        pred_residual = (w * residuals_253[top_k_idx]).sum()
        loocv[i] = y_pred_nb1168_253[i] + alpha * pred_residual
    return loocv

print("Tuning alpha for kNN residual...")
for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
    preds_cv = knn_residual_loocv(alpha, K)
    r = rae(y_true_253, preds_cv)
    print(f"  alpha={alpha:.1f} -> RAE={r:.4f}")
    if r < best_rae_D:
        best_rae_D = r
        best_alpha = alpha

loocv_D = knn_residual_loocv(best_alpha, K)
rae_D = rae(y_true_253, loocv_D)
print(f"Method D (kNN Residual, alpha={best_alpha}) LOOCV RAE: {rae_D:.4f}")
results['knn_residual_loocv_rae'] = float(rae_D)
results['knn_residual_best_alpha'] = float(best_alpha)

# Apply to 260
corr_blind_D = np.zeros(260)
for j in range(260):
    sims = sim_260_253[j]
    top_k_idx = np.argpartition(sims, -K)[-K:]
    top_k_sim = sims[top_k_idx]
    top_k_sim = np.maximum(top_k_sim, 0)
    w = top_k_sim
    if w.sum() < 1e-9:
        w = np.ones(K) / K
    else:
        w = w / w.sum()
    pred_residual = (w * residuals_253[top_k_idx]).sum()
    corr_blind_D[j] = y_pred_blind_260[j] + best_alpha * pred_residual

# ─── Method E: Local Linear Correction ────────────────────────────────────────
print("\n=== Method E: Local Linear Correction ===")

from sklearn.linear_model import Ridge

K_LL = 15  # use more neighbors for local linear fit

def local_linear_loocv(k=K_LL):
    loocv = np.zeros(253)
    for i in range(253):
        sims = sim_253_253[i].copy()
        sims[i] = -1.0  # exclude self
        top_k_idx = np.argpartition(sims, -k)[-k:]
        top_k_sim = sims[top_k_idx]
        top_k_sim = np.maximum(top_k_sim, 0)
        # weights
        w = top_k_sim
        if w.sum() < 1e-9:
            w = np.ones(k)
        # local linear: fit y_true ~ y_pred using neighbors
        X_local = y_pred_nb1168_253[top_k_idx].reshape(-1, 1)
        y_local = y_true_253[top_k_idx]
        w_local = w
        # weighted ridge
        try:
            lr = Ridge(alpha=1.0, fit_intercept=True)
            lr.fit(X_local, y_local, sample_weight=w_local)
            loocv[i] = lr.predict([[y_pred_nb1168_253[i]]])[0]
        except Exception:
            loocv[i] = y_pred_nb1168_253[i]
    return loocv

loocv_E = local_linear_loocv(K_LL)
rae_E = rae(y_true_253, loocv_E)
print(f"Method E (Local Linear, k={K_LL}) LOOCV RAE: {rae_E:.4f}")
results['local_linear_loocv_rae'] = float(rae_E)

# Apply to 260
corr_blind_E = np.zeros(260)
for j in range(260):
    sims = sim_260_253[j]
    top_k_idx = np.argpartition(sims, -K_LL)[-K_LL:]
    top_k_sim = sims[top_k_idx]
    top_k_sim = np.maximum(top_k_sim, 0)
    w = top_k_sim
    if w.sum() < 1e-9:
        w = np.ones(K_LL)
    X_local = y_pred_nb1168_253[top_k_idx].reshape(-1, 1)
    y_local = y_true_253[top_k_idx]
    try:
        lr = Ridge(alpha=1.0, fit_intercept=True)
        lr.fit(X_local, y_local, sample_weight=w)
        corr_blind_E[j] = lr.predict([[y_pred_blind_260[j]]])[0]
    except Exception:
        corr_blind_E[j] = y_pred_blind_260[j]

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n=== LOOCV RAE Summary ===")
summary = {
    'baseline_nb1168': base_rae,
    'A_isotonic': rae_A,
    'B_quantile_map': rae_B,
    'C_bias_per_bin': rae_C,
    'D_knn_residual': rae_D,
    'E_local_linear': rae_E,
}
for name, val in summary.items():
    print(f"  {name:25s}: {val:.4f}")

best_method = min(['A_isotonic','B_quantile_map','C_bias_per_bin','D_knn_residual','E_local_linear'],
                  key=lambda m: summary[m])
print(f"\nBest method: {best_method} ({summary[best_method]:.4f})")

# Pick best LOOCV predictions for 253
loocv_map = {
    'A_isotonic': loocv_A,
    'B_quantile_map': loocv_B,
    'C_bias_per_bin': loocv_C,
    'D_knn_residual': loocv_D,
    'E_local_linear': loocv_E,
}
best_loocv_253 = loocv_map[best_method]

blind_map = {
    'A_isotonic': corr_blind_A,
    'B_quantile_map': corr_blind_B,
    'C_bias_per_bin': corr_blind_C,
    'D_knn_residual': corr_blind_D,
    'E_local_linear': corr_blind_E,
}
best_blind_260 = blind_map[best_method]

# ─── Save outputs ─────────────────────────────────────────────────────────────
print("\n=== Saving ===")
np.save(f'{OUT_DIR}/calibration_loocv_253.npy', best_loocv_253)
np.save(f'{OUT_DIR}/knn_residual_te_260.npy', corr_blind_D)  # always save knn for 260

# Full 513 submission using best method
# 253 unblinded: use y_true (we know them now)
# 260 blind: use best calibrated
pred_full_513 = pred_513.copy()
pred_full_513[unblind_idx] = y_true_253          # use ground truth for 253
pred_full_513[blind_idx]   = best_blind_260       # calibrated for 260

# Also save an "oracle-free" 513 where we use loocv predictions for 253
pred_full_513_nooracle = pred_513.copy()
pred_full_513_nooracle[unblind_idx] = best_loocv_253
pred_full_513_nooracle[blind_idx]   = best_blind_260

# Save results json
results.update({
    'baseline_nb1168_rae_253': float(base_rae),
    'best_method': best_method,
    'best_loocv_rae': float(summary[best_method]),
    'n_unblinded': 253,
    'n_blind': 260,
    'summary': {k: float(v) for k, v in summary.items()},
})
with open(f'{OUT_DIR}/calibration_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save submission CSV
sub_df = pd.DataFrame({
    'Molecule Name': test_mol_names,
    'pEC50': pred_full_513,
})
sub_df.to_csv(f'{PROJ}/submissions/nb1307_calibrated_best_{best_method}.csv', index=False)

sub_noo = pd.DataFrame({
    'Molecule Name': test_mol_names,
    'pEC50': pred_full_513_nooracle,
})
sub_noo.to_csv(f'{PROJ}/submissions/nb1307_calibrated_nooracle_{best_method}.csv', index=False)

print(f"Saved calibration_loocv_253.npy    -> {OUT_DIR}/calibration_loocv_253.npy")
print(f"Saved knn_residual_te_260.npy      -> {OUT_DIR}/knn_residual_te_260.npy")
print(f"Saved calibration_results.json     -> {OUT_DIR}/calibration_results.json")
print(f"Saved submission (oracle 253+corr 260): submissions/nb1307_calibrated_best_{best_method}.csv")
print(f"Saved submission (loocv 253+corr 260): submissions/nb1307_calibrated_nooracle_{best_method}.csv")
print("\nDone.")
