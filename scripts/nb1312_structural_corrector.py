"""
nb1312_structural_corrector.py

Boltz-2 z-embedding residual corrector + ensemble diversity analysis.

Steps:
  1. Load Boltz-2 z-embeddings (interaction block, cols 768:1024 pooled => rich-z)
  2. Train residual corrector (LGBM on z-train → residuals w.r.t. GNN OOF)
     Apply to test, tune alpha on 253 unblinded
  3. Uncertainty-weighted abstention on 260 blinded
  4. Ensemble diversity analysis: pairwise corr, max-diverse subsets of size 3-5
  5. Save artefacts to C:/pxr_work/meta_stacking/
"""

import sys, os, json, glob, warnings
warnings.filterwarnings('ignore')

os.makedirs('C:/pxr_work/meta_stacking', exist_ok=True)

sys.path.insert(0, 'D:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge')
import numpy as np
import pandas as pd
from itertools import combinations

from src.pxr.data import load_train, load_test
from src.pxr.eval import rae

# ─── 0. Load core data ────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 0: Loading core data")
print("=" * 70)

tr = load_train()
te = load_test()
y_train = tr['pec50'].values
train_names = tr['name'].values
test_names  = te['name'].values

# 253 unblinded labels
unblind = pd.read_csv(
    'D:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge/'
    'data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv'
)
unblind_names = unblind['Molecule Name'].values
y_unblind     = unblind['pEC50'].values

# Align unblinded to test order
te_name_to_idx = {n: i for i, n in enumerate(test_names)}
unblind_mask   = np.array([n in te_name_to_idx for n in unblind_names])
unblind_idx    = np.array([te_name_to_idx[n] for n in unblind_names[unblind_mask]])
y_unb          = y_unblind[unblind_mask]          # 253 true labels aligned to test order
unb_in_test    = unblind_idx                      # indices into test_names

# 260 blinded indices
all_idx    = np.arange(len(test_names))
blind_idx  = np.setdiff1d(all_idx, unb_in_test)  # 260 blinded
print(f"Train: {len(y_train)}, Test: {len(test_names)}, "
      f"Unblinded: {len(y_unb)}, Blinded: {len(blind_idx)}")

# ─── 1. Load Boltz z-embeddings ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 1: Loading Boltz-2 z-embeddings")
print("=" * 70)

boltz_dir = 'C:/pxr_struct/boltz'
npy_files = glob.glob(boltz_dir + '/**/*.npy', recursive=True)
print(f"Found {len(npy_files)} npy files in boltz dir")

# Load rich-z = interaction block (pre-pooled)
z_rich_te_path = os.path.join(boltz_dir, 'boltz_z_rich_513.npy')
z_rich_tr_path = os.path.join(boltz_dir, 'boltz_z_rich_train.npy')

z_te = np.load(z_rich_te_path).astype(np.float32)   # (513, 512) or (513, 256)
z_tr = np.load(z_rich_tr_path).astype(np.float32)   # (4139, d)
print(f"z_rich_te shape: {z_te.shape}")
print(f"z_rich_tr shape: {z_tr.shape}")

# Handle NaNs / Infs
z_tr = np.nan_to_num(z_tr, nan=0.0, posinf=0.0, neginf=0.0)
z_te = np.nan_to_num(z_te, nan=0.0, posinf=0.0, neginf=0.0)

# Also check if full boltz_emb_513 (1024-d) is there
boltz_full_path = os.path.join(boltz_dir, 'boltz_emb_513.npy')
boltz_full = np.load(boltz_full_path).astype(np.float32)
print(f"boltz_emb_513 full shape: {boltz_full.shape}")

# ─── 2. Residual corrector on z-embeddings ────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2: Residual corrector (LGBM on z-train -> GNN residuals)")
print("=" * 70)

from lightgbm import LGBMRegressor
from src.pxr.eval import scaffold_kfold_indices

# Load GNN OOF and test predictions
gnn_oof = np.load('C:/pxr_work/search/gnn_oof.npy')    # (4139,)
gnn_te  = np.load('C:/pxr_work/search/gnn_te.npy')     # (513,)
print(f"gnn_oof shape: {gnn_oof.shape}, gnn_te shape: {gnn_te.shape}")

# Residuals (train-level)
resid_train = y_train - gnn_oof
print(f"Residual stats: mean={resid_train.mean():.4f}, "
      f"std={resid_train.std():.4f}, "
      f"abs_mean={np.abs(resid_train).mean():.4f}")

# Best submission predictions (nb1299 OrbMol ensemble)
best_sub = pd.read_csv(
    'D:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge/'
    'submissions/nb1299_orbmol_ensemble.csv'
)
best_sub = best_sub.set_index('Molecule Name')['pEC50']
pred_te_best = np.array([best_sub[n] for n in test_names])   # (513,)
print(f"Best submission (nb1299) preds shape: {pred_te_best.shape}")

# Residuals of best submission on 253 unblinded
resid_best_unb = y_unb - pred_te_best[unb_in_test]
print(f"Best sub residual on 253: mean={resid_best_unb.mean():.4f}, "
      f"abs_mean={np.abs(resid_best_unb).mean():.4f}")
print(f"Best sub RAE on 253: {rae(y_unb, pred_te_best[unb_in_test]):.4f}")

# Scaffold CV to train z-residual corrector
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
z_tr_sc = scaler.fit_transform(z_tr)
z_te_sc = scaler.transform(z_te)

from src.pxr.chem import add_standard_columns
scaffolds_tr = tr['smiles'].apply(
    lambda s: __import__('src.pxr.chem', fromlist=['standardize']).standardize(s)
)
# Use scaffold col if it exists, otherwise use smiles for scaffold splitting
if 'scaffold' in tr.columns:
    scaffolds_tr = tr['scaffold'].fillna('').values
else:
    scaffolds_tr = tr['smiles'].values

fold_indices = scaffold_kfold_indices(scaffolds_tr, n_splits=5)

# Train residual model on z-train via scaffold CV
print("\nTraining LGBM residual corrector on z-train (5-fold scaffold CV)...")
oof_resid_pred = np.zeros(len(y_train))
test_resid_preds = []

lgbm_params = dict(
    n_estimators=300,
    num_leaves=31,
    learning_rate=0.05,
    min_child_samples=20,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=42,
    verbose=-1,
)

for fold, (tr_idx, val_idx) in enumerate(fold_indices):
    Xtr, Xval = z_tr_sc[tr_idx], z_tr_sc[val_idx]
    ytr_r, yval_r = resid_train[tr_idx], resid_train[val_idx]

    m = LGBMRegressor(**lgbm_params)
    m.fit(Xtr, ytr_r)
    oof_resid_pred[val_idx] = m.predict(Xval)
    test_resid_preds.append(m.predict(z_te_sc))
    val_rae = rae(y_train[val_idx], gnn_oof[val_idx] + oof_resid_pred[val_idx])
    print(f"  Fold {fold}: val RAE (gnn+resid) = {val_rae:.4f}")

# Average test residual predictions across folds
test_resid_mean = np.mean(test_resid_preds, axis=0)
print(f"\nTest residual pred stats: mean={test_resid_mean.mean():.4f}, "
      f"std={test_resid_mean.std():.4f}")

# Tune alpha on 253 unblinded
print("\nTuning alpha (correction weight) on 253 unblinded...")
alphas = np.linspace(0.0, 1.0, 41)
alpha_raes = []
for a in alphas:
    corrected = pred_te_best[unb_in_test] + a * test_resid_mean[unb_in_test]
    alpha_raes.append(rae(y_unb, corrected))

best_alpha_idx = np.argmin(alpha_raes)
best_alpha = alphas[best_alpha_idx]
best_rae_corrected = alpha_raes[best_alpha_idx]
baseline_rae_253   = rae(y_unb, pred_te_best[unb_in_test])

print(f"Baseline RAE (no correction): {baseline_rae_253:.4f}")
print(f"Best alpha: {best_alpha:.3f}  → RAE: {best_rae_corrected:.4f}  "
      f"(delta: {best_rae_corrected - baseline_rae_253:+.4f})")

# Apply best alpha to full 513
corrected_pred_513 = pred_te_best + best_alpha * test_resid_mean
print(f"\nCorrected predictions (alpha={best_alpha:.3f}):")
print(f"  253 RAE: {rae(y_unb, corrected_pred_513[unb_in_test]):.4f}")

# ─── 3. Uncertainty-based abstention / shrinkage ──────────────────────────────
print("\n" + "=" * 70)
print("STEP 3: Uncertainty-based abstention (variance shrinkage)")
print("=" * 70)

# Load all available 513-row submission CSVs
sub_paths = sorted(glob.glob(
    'D:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge/submissions/nb1*.csv'
))
pred_matrix = []  # will be (n_models, 513)
sub_names   = []

for sp in sub_paths:
    try:
        df = pd.read_csv(sp)
        if len(df) != 513:
            continue
        # Align to test_names order
        name_col = 'Molecule Name' if 'Molecule Name' in df.columns else df.columns[0]
        pred_col = 'pEC50' if 'pEC50' in df.columns else df.columns[-1]
        df = df.set_index(name_col)
        if not all(n in df.index for n in test_names[:10]):
            continue
        preds = np.array([df.loc[n, pred_col] for n in test_names])
        pred_matrix.append(preds)
        sub_names.append(os.path.basename(sp))
    except Exception:
        continue

pred_matrix = np.array(pred_matrix)  # (n_models, 513)
print(f"Loaded {pred_matrix.shape[0]} valid submission arrays × {pred_matrix.shape[1]} compounds")

# Per-compound prediction variance
pred_var_513  = pred_matrix.var(axis=0)   # (513,)
pred_std_513  = np.sqrt(pred_var_513)
pred_mean_513 = pred_matrix.mean(axis=0)
pred_med_513  = np.median(pred_matrix, axis=0)

print(f"Prediction std (all 513): mean={pred_std_513.mean():.4f}, "
      f"max={pred_std_513.max():.4f}, min={pred_std_513.min():.4f}")
print(f"253 unblinded: mean std={pred_std_513[unb_in_test].mean():.4f}")
print(f"260 blinded:   mean std={pred_std_513[blind_idx].mean():.4f}")

# Tune shrinkage on 253 LOOCV style: w(compound) = f(std)
# final_i = (1 - w_i) * pred_i + w_i * train_mean
train_median = np.median(y_train)
print(f"\nTraining median pEC50: {train_median:.4f}")

# Normalize std to [0,1] for weight function
std_min = pred_std_513.min()
std_max = pred_std_513.max()
std_norm = (pred_std_513 - std_min) / (std_max - std_min + 1e-9)

# Test different gamma (sensitivity) and max_w values
best_var_rae = 1e9
best_gamma = 0.0
best_maxw  = 0.0

for gamma in np.linspace(0, 3, 16):
    for max_w in np.linspace(0.0, 0.5, 11):
        w_i = max_w * (std_norm ** gamma)
        final_unb = (1 - w_i[unb_in_test]) * pred_te_best[unb_in_test] + \
                    w_i[unb_in_test] * train_median
        r = rae(y_unb, final_unb)
        if r < best_var_rae:
            best_var_rae = r
            best_gamma   = gamma
            best_maxw    = max_w

w_opt = best_maxw * (std_norm ** best_gamma)
shrunk_pred_513 = (1 - w_opt) * pred_te_best + w_opt * train_median
rae_shrunk_253  = rae(y_unb, shrunk_pred_513[unb_in_test])

print(f"Variance shrinkage: gamma={best_gamma:.2f}, max_w={best_maxw:.2f}")
print(f"Baseline RAE (253): {baseline_rae_253:.4f}")
print(f"Shrunk RAE (253):   {rae_shrunk_253:.4f}  "
      f"(delta: {rae_shrunk_253 - baseline_rae_253:+.4f})")

# ─── 4. Structural correction + uncertainty combined ─────────────────────────
print("\n" + "=" * 70)
print("STEP 4: Combined z-residual + uncertainty shrinkage")
print("=" * 70)

# Grid search over (alpha, gamma, max_w)
best_comb_rae = 1e9
best_a2 = best_g2 = best_mw2 = 0.0

for a in np.linspace(0.0, 0.8, 17):
    for gamma in [0.0, 0.5, 1.0, 2.0]:
        for max_w in [0.0, 0.05, 0.10, 0.15, 0.20]:
            w_i = max_w * (std_norm ** gamma)
            # Step 1: z-corrected
            z_corr = pred_te_best + a * test_resid_mean
            # Step 2: variance shrinkage
            final_  = (1 - w_i) * z_corr + w_i * train_median
            r = rae(y_unb, final_[unb_in_test])
            if r < best_comb_rae:
                best_comb_rae = r
                best_a2  = a
                best_g2  = gamma
                best_mw2 = max_w

# Build final combined prediction
w_comb = best_mw2 * (std_norm ** best_g2)
combined_pred = pred_te_best + best_a2 * test_resid_mean
combined_pred = (1 - w_comb) * combined_pred + w_comb * train_median
rae_combined_253 = rae(y_unb, combined_pred[unb_in_test])

print(f"Combined (alpha={best_a2:.3f}, gamma={best_g2:.2f}, max_w={best_mw2:.3f}):")
print(f"  RAE (253): {rae_combined_253:.4f}  "
      f"(vs baseline {baseline_rae_253:.4f}, delta {rae_combined_253 - baseline_rae_253:+.4f})")

# ─── 5. Ensemble diversity analysis ───────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 5: Ensemble diversity analysis (pairwise correlations + subsets)")
print("=" * 70)

# Only use unblinded slice for correlation analysis
P = pred_matrix[:, unb_in_test]  # (n_models, 253)
n_models = P.shape[0]
print(f"Correlation analysis on {n_models} models × {P.shape[1]} unblinded compounds")

# Pairwise correlation matrix
corr_mat = np.corrcoef(P)  # (n_models, n_models)

# Per-model RAE on 253
model_raes = np.array([rae(y_unb, P[i]) for i in range(n_models)])
print(f"\nPer-model RAE stats on 253:")
print(f"  min={model_raes.min():.4f}, max={model_raes.max():.4f}, "
      f"mean={model_raes.mean():.4f}, median={np.median(model_raes):.4f}")

# Find most diverse pair (lowest correlation among top-50 models by RAE)
top_k = min(50, n_models)
top_idx = np.argsort(model_raes)[:top_k]  # lowest RAE = best

best_pair_corr = 1.0
best_pair = (0, 1)
for i, j in combinations(top_idx, 2):
    c = corr_mat[i, j]
    if c < best_pair_corr:
        best_pair_corr = c
        best_pair = (i, j)

pi, pj = best_pair
pair_blend = 0.5 * P[pi] + 0.5 * P[pj]
rae_pair = rae(y_unb, pair_blend)
print(f"\nMost diverse pair (top-50 models):")
print(f"  {sub_names[pi]} (RAE={model_raes[pi]:.4f})")
print(f"  {sub_names[pj]} (RAE={model_raes[pj]:.4f})")
print(f"  Correlation: {best_pair_corr:.4f}")
print(f"  50/50 blend RAE: {rae_pair:.4f}  "
      f"(vs best single {min(model_raes[pi], model_raes[pj]):.4f})")

# Find max-diverse subsets of size 3, 4, 5
def avg_pairwise_corr(subset, corr):
    pairs = list(combinations(subset, 2))
    if not pairs:
        return 1.0
    return np.mean([corr[i, j] for i, j in pairs])

def mean_rae_blend(subset, P, y):
    blend = P[list(subset)].mean(axis=0)
    return rae(y, blend)

results_by_size = {}
for size in [3, 4, 5]:
    # Restrict search to top-K models for speed
    search_k = min(30, n_models)
    search_idx = np.argsort(model_raes)[:search_k]

    best_rae_s = 1e9
    best_subset_s = None
    best_corr_s = 1.0

    # For each subset: evaluate both diversity-first and RAE-first
    candidate_subsets = list(combinations(search_idx, size))
    print(f"\nSize-{size}: evaluating {len(candidate_subsets)} subsets "
          f"(from top-{search_k} models)...")

    rae_vals   = []
    corr_vals  = []
    for sub in candidate_subsets:
        r  = mean_rae_blend(sub, P, y_unb)
        ac = avg_pairwise_corr(sub, corr_mat)
        rae_vals.append(r)
        corr_vals.append(ac)

    rae_vals  = np.array(rae_vals)
    corr_vals = np.array(corr_vals)

    # Best by RAE
    best_rae_idx     = np.argmin(rae_vals)
    best_rae_sub     = candidate_subsets[best_rae_idx]
    best_rae_sub_rae = rae_vals[best_rae_idx]

    # Most diverse subset (min avg corr)
    best_div_idx     = np.argmin(corr_vals)
    best_div_sub     = candidate_subsets[best_div_idx]
    best_div_sub_rae = rae_vals[best_div_idx]
    best_div_corr    = corr_vals[best_div_idx]

    results_by_size[size] = {
        'best_rae_subset_names': [sub_names[i] for i in best_rae_sub],
        'best_rae_subset_rae':   float(best_rae_sub_rae),
        'best_div_subset_names': [sub_names[i] for i in best_div_sub],
        'best_div_subset_rae':   float(best_div_sub_rae),
        'best_div_avg_corr':     float(best_div_corr),
    }
    print(f"  Best by RAE ({best_rae_sub_rae:.4f}): "
          f"{[sub_names[i] for i in best_rae_sub]}")
    print(f"  Most diverse (avg_corr={best_div_corr:.4f}, RAE={best_div_sub_rae:.4f}): "
          f"{[sub_names[i] for i in best_div_sub]}")

# ─── 6. Save outputs ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 6: Saving outputs")
print("=" * 70)

out_dir = 'C:/pxr_work/meta_stacking'
os.makedirs(out_dir, exist_ok=True)

# Save corrected predictions (full 513)
np.save(os.path.join(out_dir, 'structural_corrector_te_513.npy'), corrected_pred_513)
np.save(os.path.join(out_dir, 'z_resid_corrected_513.npy'),       corrected_pred_513)
np.save(os.path.join(out_dir, 'combined_corrected_513.npy'),       combined_pred)

print(f"Saved corrected preds to {out_dir}/structural_corrector_te_513.npy")

# Summary JSON
results = {
    'z_embedding_shape_te':   list(z_te.shape),
    'z_embedding_shape_tr':   list(z_tr.shape),
    'n_submissions_loaded':   int(pred_matrix.shape[0]),
    'n_unblinded':            int(len(y_unb)),
    'n_blinded':              int(len(blind_idx)),
    'baseline_rae_253':       float(baseline_rae_253),
    'z_resid_corrector': {
        'best_alpha':         float(best_alpha),
        'rae_253':            float(best_rae_corrected),
        'delta_vs_baseline':  float(best_rae_corrected - baseline_rae_253),
    },
    'variance_shrinkage': {
        'best_gamma':         float(best_gamma),
        'best_max_w':         float(best_maxw),
        'rae_253':            float(rae_shrunk_253),
        'delta_vs_baseline':  float(rae_shrunk_253 - baseline_rae_253),
    },
    'combined': {
        'alpha':              float(best_a2),
        'gamma':              float(best_g2),
        'max_w':              float(best_mw2),
        'rae_253':            float(rae_combined_253),
        'delta_vs_baseline':  float(rae_combined_253 - baseline_rae_253),
    },
    'diversity_analysis': {
        'most_diverse_pair': {
            'models':       [sub_names[pi], sub_names[pj]],
            'correlation':  float(best_pair_corr),
            'blend_rae':    float(rae_pair),
            'individual_raes': [float(model_raes[pi]), float(model_raes[pj])],
        },
        'subsets': results_by_size,
    },
    'pred_variance_stats': {
        'mean_std_all_513':     float(pred_std_513.mean()),
        'mean_std_253_unb':     float(pred_std_513[unb_in_test].mean()),
        'mean_std_260_blind':   float(pred_std_513[blind_idx].mean()),
    },
    'train_median_pec50': float(train_median),
}

out_json = os.path.join(out_dir, 'structural_corrector_results.json')
with open(out_json, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved results JSON to {out_json}")

# Print summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Baseline RAE on 253:              {baseline_rae_253:.4f}")
print(f"Z-residual corrector (alpha={best_alpha:.2f}): {best_rae_corrected:.4f}  "
      f"({best_rae_corrected - baseline_rae_253:+.4f})")
print(f"Variance shrinkage:               {rae_shrunk_253:.4f}  "
      f"({rae_shrunk_253 - baseline_rae_253:+.4f})")
print(f"Combined (z+shrink):              {rae_combined_253:.4f}  "
      f"({rae_combined_253 - baseline_rae_253:+.4f})")
print()
print("Diversity subsets (best RAE per size):")
for sz, r in results_by_size.items():
    print(f"  Size {sz}: {r['best_rae_subset_rae']:.4f}")
