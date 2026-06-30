"""
nb1308_novel_models.py
Novel model families — SVM/GP/RF/ET/kNN/MAE-LGBM
Trained on 4139, evaluated on 253 unblinded.
"""
import sys, os, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, 'D:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge')

from src.pxr.data import load_train, load_test
from src.pxr.featurize import combined, impute
from src.pxr.chem import morgan_fp_batch
from src.pxr.eval import scaffold_kfold_indices, rae

# ─── Output dir ───────────────────────────────────────────────────────────────
OUT = "C:/pxr_work/meta_stacking"
os.makedirs(OUT, exist_ok=True)

# ─── Load data ────────────────────────────────────────────────────────────────
print("Loading train / test …")
train = load_train()
test  = load_test()

y_tr = train["pec50"].values.astype(np.float64)
smiles_tr = train["smiles"].tolist()
smiles_te = test["smiles"].tolist()
names_te  = test["name"].tolist()

# ─── Load 253 unblinded labels ────────────────────────────────────────────────
print("Loading unblinded labels …")
unblind_raw = pd.read_csv("C:/pxr_work/phase1_unblind/phase1_unblinded_raw.csv")
unblind_raw.columns = unblind_raw.columns.str.strip()

# Align with the 513-test ordering
name_col = "Molecule Name"
unblind_map = dict(zip(unblind_raw[name_col], unblind_raw["pEC50"]))

# Build index into the 513 test set
unblind_idx = []
unblind_y   = []
for i, nm in enumerate(names_te):
    if nm in unblind_map:
        unblind_idx.append(i)
        unblind_y.append(unblind_map[nm])

unblind_idx = np.array(unblind_idx)
unblind_y   = np.array(unblind_y, dtype=np.float64)
print(f"  Unblinded: {len(unblind_idx)} / 513 compounds")

# ─── Featurise ────────────────────────────────────────────────────────────────
print("Computing Morgan fingerprints …")
fp_tr = morgan_fp_batch(smiles_tr).astype(np.uint8)   # (4139, 2048)
fp_te = morgan_fp_batch(smiles_te).astype(np.uint8)   # (513,  2048)
print(f"  fp_tr {fp_tr.shape}, fp_te {fp_te.shape}")

print("Computing combined features …")
X_tr_raw = combined(smiles_tr)
X_te_raw = combined(smiles_te)
X_tr = impute(X_tr_raw).astype(np.float32)
X_te = impute(X_te_raw).astype(np.float32)
print(f"  X_tr {X_tr.shape}, X_te {X_te.shape}")

# ─── Tanimoto kernel helper ───────────────────────────────────────────────────
def tanimoto_kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Tanimoto (Jaccard) kernel for binary fingerprints. A:(n,d), B:(m,d) → (n,m)"""
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    dot   = A @ B.T
    sum_A = A.sum(1, keepdims=True)
    sum_B = B.sum(1, keepdims=True)
    denom = sum_A + sum_B.T - dot + 1e-8
    return (dot / denom).astype(np.float64)

# ─── Scaffold CV helper ───────────────────────────────────────────────────────
scaffolds = train["scaffold"].tolist() if "scaffold" in train.columns else [None]*len(train)

def scaffold_cv_rae(predict_fn, n_splits=5):
    """5-fold scaffold CV, returns mean RAE."""
    folds = scaffold_kfold_indices(scaffolds, n_splits=n_splits)
    raes = []
    for tr_idx, va_idx in folds:
        preds_va = predict_fn(tr_idx, va_idx)
        raes.append(rae(y_tr[va_idx], preds_va))
    return float(np.mean(raes))

# ─── Results collector ───────────────────────────────────────────────────────
results = {}

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — SVM with precomputed Tanimoto kernel
# ══════════════════════════════════════════════════════════════════════════════
print("\n[MODEL 1] SVM (precomputed Tanimoto) …")
from sklearn.svm import SVR

print("  Computing K_trtr …")
K_trtr = tanimoto_kernel(fp_tr, fp_tr)   # (4139, 4139)
print("  Computing K_trte …")
K_trte = tanimoto_kernel(fp_tr, fp_te)   # (4139, 513)

# Scaffold CV
def svr_predict(tr_idx, va_idx):
    K_sub = K_trtr[np.ix_(tr_idx, tr_idx)]
    svr_cv = SVR(kernel='precomputed', C=10, epsilon=0.1)
    svr_cv.fit(K_sub, y_tr[tr_idx])
    K_va = K_trtr[np.ix_(va_idx, tr_idx)]
    return svr_cv.predict(K_va)

t0 = time.time()
cv_rae_svr = scaffold_cv_rae(svr_predict)
print(f"  CV RAE: {cv_rae_svr:.4f}  ({time.time()-t0:.1f}s)")

# Fit on all 4139
svr_full = SVR(kernel='precomputed', C=10, epsilon=0.1)
svr_full.fit(K_trtr, y_tr)
preds_svr = svr_full.predict(K_trte.T)   # (513,)
rae_svr_253 = rae(unblind_y, preds_svr[unblind_idx])
print(f"  253-unblind RAE: {rae_svr_253:.4f}")

np.save(f"{OUT}/novel_model_svr_te_513.npy", preds_svr)
results["svr"] = {"cv_rae": cv_rae_svr, "rae_253": rae_svr_253}

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — Gaussian Process with Tanimoto kernel (subset for speed)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[MODEL 2] Gaussian Process (Tanimoto, n=1000 subset) …")
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import PairwiseKernel, WhiteKernel, ConstantKernel

# Subsample 1000 training points for speed
rng = np.random.default_rng(42)
gp_idx = rng.choice(len(y_tr), size=1000, replace=False)
gp_idx.sort()

fp_gp = fp_tr[gp_idx]
y_gp  = y_tr[gp_idx]

K_gpgp = tanimoto_kernel(fp_gp, fp_gp)
K_gpte = tanimoto_kernel(fp_gp, fp_te)   # (1000, 513)

# GP with precomputed kernel trick via PairwiseKernel
# We build it manually: use DotProduct as proxy, pass precomputed Gram
# Simpler: wrap via a custom kernel class
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter

class PrecomputedTanimotoKernel(Kernel):
    """Fixed precomputed kernel — no hyperparameters."""
    def __init__(self, K_train, K_cross=None):
        self.K_train  = K_train
        self.K_cross  = K_cross

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.asarray(X, dtype=int).ravel()
        if Y is None:
            K = self.K_train[np.ix_(X, X)]
            if eval_gradient:
                return K, np.empty((len(X), len(X), 0))
            return K
        Y = np.asarray(Y, dtype=int).ravel()
        # X are train indices, Y are test indices (into K_cross rows)
        K = self.K_cross[np.ix_(X, Y)]
        return K

    def diag(self, X):
        X = np.asarray(X, dtype=int).ravel()
        return np.ones(len(X))

    def is_stationary(self):
        return False

    @property
    def hyperparameters(self):
        return []

    @property
    def theta(self):
        return np.array([])

    @theta.setter
    def theta(self, val):
        pass

    @property
    def bounds(self):
        return np.empty((0, 2))

    def get_params(self, deep=True):
        return {"K_train": self.K_train, "K_cross": self.K_cross}

    def __repr__(self):
        return "PrecomputedTanimotoKernel()"

# Simpler approach: use dot-product on fp vectors directly via sklearn
# (PairwiseKernel wraps metric='additive_chi2' etc. — we use 'linear' on fp floats)
# For a true Tanimoto GP, build manually:

from scipy.linalg import cho_solve, cho_factor

alpha_noise = 0.5   # noise regularisation

t0 = time.time()
# Cholesky solve for GP posterior mean
K_noisy = K_gpgp + alpha_noise * np.eye(len(gp_idx))
try:
    L, lower = cho_factor(K_noisy, lower=True)
    alpha_gp = cho_solve((L, lower), y_gp)
    # Predict mean
    preds_gp_mean = K_gpte.T @ alpha_gp      # (513,)
    # Predict variance
    v = cho_solve((L, lower), K_gpte)         # (1000, 513)
    K_te_diag = np.ones(fp_te.shape[0])       # diag(K(te,te)) = 1 for Tanimoto
    gp_var = K_te_diag - (K_gpte * v).sum(0)  # (513,)
    gp_std = np.sqrt(np.maximum(gp_var, 1e-8))
    print(f"  GP fit done ({time.time()-t0:.1f}s)")
    gp_ok = True
except Exception as e:
    print(f"  GP failed: {e} — falling back to zeros")
    preds_gp_mean = np.full(513, y_tr.mean())
    gp_std = np.ones(513)
    gp_ok  = False

rae_gp_253 = rae(unblind_y, preds_gp_mean[unblind_idx])
print(f"  253-unblind RAE: {rae_gp_253:.4f}  (std mean={gp_std.mean():.3f})")

np.save(f"{OUT}/novel_model_gp_te_513.npy",           preds_gp_mean)
np.save(f"{OUT}/gp_uncertainty_513.npy",               gp_std)

# GP CV on 4139 not practical (1000-subset); report NA
results["gp_subset1000"] = {"cv_rae": None, "rae_253": rae_gp_253,
                             "note": "subset=1000, no scaffold CV"}

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 3 — Random Forest (combined features)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[MODEL 3] Random Forest (combined, 2265 feats) …")
from sklearn.ensemble import RandomForestRegressor

def rf_cv_predict(tr_idx, va_idx):
    rf = RandomForestRegressor(n_estimators=300, max_depth=None,
                               min_samples_leaf=2, n_jobs=-1, random_state=42)
    rf.fit(X_tr[tr_idx], y_tr[tr_idx])
    return rf.predict(X_tr[va_idx])

t0 = time.time()
cv_rae_rf = scaffold_cv_rae(rf_cv_predict)
print(f"  CV RAE: {cv_rae_rf:.4f}  ({time.time()-t0:.1f}s)")

rf_full = RandomForestRegressor(n_estimators=500, max_depth=None,
                                min_samples_leaf=2, n_jobs=-1, random_state=42)
rf_full.fit(X_tr, y_tr)
preds_rf = rf_full.predict(X_te)
rae_rf_253 = rae(unblind_y, preds_rf[unblind_idx])
print(f"  253-unblind RAE: {rae_rf_253:.4f}")

np.save(f"{OUT}/novel_model_rf_te_513.npy", preds_rf)
results["random_forest"] = {"cv_rae": cv_rae_rf, "rae_253": rae_rf_253}

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 4 — Extra Trees
# ══════════════════════════════════════════════════════════════════════════════
print("\n[MODEL 4] Extra Trees (combined, 2265 feats) …")
from sklearn.ensemble import ExtraTreesRegressor

def et_cv_predict(tr_idx, va_idx):
    et = ExtraTreesRegressor(n_estimators=300, max_depth=None,
                             min_samples_leaf=2, n_jobs=-1, random_state=42)
    et.fit(X_tr[tr_idx], y_tr[tr_idx])
    return et.predict(X_tr[va_idx])

t0 = time.time()
cv_rae_et = scaffold_cv_rae(et_cv_predict)
print(f"  CV RAE: {cv_rae_et:.4f}  ({time.time()-t0:.1f}s)")

et_full = ExtraTreesRegressor(n_estimators=500, max_depth=None,
                               min_samples_leaf=2, n_jobs=-1, random_state=42)
et_full.fit(X_tr, y_tr)
preds_et = et_full.predict(X_te)
rae_et_253 = rae(unblind_y, preds_et[unblind_idx])
print(f"  253-unblind RAE: {rae_et_253:.4f}")

np.save(f"{OUT}/novel_model_et_te_513.npy", preds_et)
results["extra_trees"] = {"cv_rae": cv_rae_et, "rae_253": rae_et_253}

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 5 — Tanimoto k-NN regression
# ══════════════════════════════════════════════════════════════════════════════
print("\n[MODEL 5] k-NN Tanimoto regression …")

def knn_predict_full(K_tr2q, y_train, k):
    """K_tr2q: (n_train, n_query) Tanimoto similarities"""
    # Top-k for each query
    top_k_idx = np.argsort(K_tr2q, axis=0)[-k:]   # (k, n_query)
    top_k_sim = K_tr2q[top_k_idx, np.arange(K_tr2q.shape[1])[None,:]]
    top_k_y   = y_train[top_k_idx]
    w = top_k_sim + 1e-8
    w_sum = w.sum(0, keepdims=True)
    return (top_k_y * w / w_sum).sum(0)

# CV to pick best k
best_k, best_k_rae = None, 1e9
for k in [3, 5, 10, 15]:
    def knn_cv(tr_idx, va_idx, _k=k):
        K_sub = K_trtr[np.ix_(tr_idx, va_idx)]   # (n_tr, n_va)
        return knn_predict_full(K_sub, y_tr[tr_idx], _k)
    cv_r = scaffold_cv_rae(knn_cv)
    print(f"  k={k:2d}  CV RAE={cv_r:.4f}")
    if cv_r < best_k_rae:
        best_k, best_k_rae = k, cv_r

print(f"  Best k={best_k} CV RAE={best_k_rae:.4f}")

preds_knn = knn_predict_full(K_trte, y_tr, best_k)   # K_trte: (4139,513)
rae_knn_253 = rae(unblind_y, preds_knn[unblind_idx])
print(f"  253-unblind RAE: {rae_knn_253:.4f}")

np.save(f"{OUT}/novel_model_knn_te_513.npy", preds_knn)
results["knn_tanimoto"] = {"cv_rae": best_k_rae, "rae_253": rae_knn_253,
                           "best_k": best_k}

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 6 — LGBM with MAE loss
# ══════════════════════════════════════════════════════════════════════════════
print("\n[MODEL 6] LGBM MAE loss …")
from lightgbm import LGBMRegressor

def lgbm_mae_cv(tr_idx, va_idx):
    m = LGBMRegressor(objective='mae', n_estimators=500, num_leaves=64,
                      learning_rate=0.05, n_jobs=-1, random_state=42, verbose=-1)
    m.fit(X_tr[tr_idx], y_tr[tr_idx])
    return m.predict(X_tr[va_idx])

t0 = time.time()
cv_rae_lgbm_mae = scaffold_cv_rae(lgbm_mae_cv)
print(f"  CV RAE: {cv_rae_lgbm_mae:.4f}  ({time.time()-t0:.1f}s)")

lgbm_mae_full = LGBMRegressor(objective='mae', n_estimators=500, num_leaves=64,
                               learning_rate=0.05, n_jobs=-1, random_state=42, verbose=-1)
lgbm_mae_full.fit(X_tr, y_tr)
preds_lgbm_mae = lgbm_mae_full.predict(X_te)
rae_lgbm_mae_253 = rae(unblind_y, preds_lgbm_mae[unblind_idx])
print(f"  253-unblind RAE: {rae_lgbm_mae_253:.4f}")

np.save(f"{OUT}/novel_model_lgbm_mae_te_513.npy", preds_lgbm_mae)
results["lgbm_mae"] = {"cv_rae": cv_rae_lgbm_mae, "rae_253": rae_lgbm_mae_253}

# ─── Also save the 253 unblind index + y_true ─────────────────────────────────
np.save(f"{OUT}/unblind_idx_253.npy",  unblind_idx)
np.save(f"{OUT}/unblind_y_true_253.npy", unblind_y)

# ─── Summary ──────────────────────────────────────────────────────────────────
with open(f"{OUT}/novel_models_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print(f"{'Model':<22} {'4139-CV-RAE':>12} {'253-unblind-RAE':>16}")
print("-"*52)
for name, d in results.items():
    cv  = f"{d['cv_rae']:.4f}" if d['cv_rae'] is not None else "   N/A"
    r253 = f"{d['rae_253']:.4f}"
    print(f"{name:<22} {cv:>12} {r253:>16}")
print("="*60)
print(f"\nSaved to {OUT}/")
