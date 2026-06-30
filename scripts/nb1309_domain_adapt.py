"""nb1309 — Domain Adaptation via Importance Weighting.

Train/test distributions differ (train=4139 diverse, test=513 analog expansion).
Importance weighting re-focuses training on compounds similar to test, potentially
reducing distribution shift and improving predictions on test-like compounds.

Three approaches:
  A) Tanimoto importance weights   — max sim to test, normalize, clip [0.1, 5.0]
  B) Kernel Mean Matching (KMM)    — match train/test Morgan FP distributions via QP
  C) Adversarial domain adaptation — classifier P(test|x) / P(train|x) as weights

Evaluate each approach:
  - Scaffold 5-fold CV RAE on 4139 training compounds
  - RAE on 253 unblinded test compounds (honest blinded gate)
  - Compare to unweighted baseline (same 4-GBM architecture)
"""

import os, sys, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, 'D:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge')

from src.pxr.data import load_train, load_test
from src.pxr.featurize import combined, impute
from src.pxr.chem import morgan_fp_batch
from src.pxr.eval import rae, scaffold_kfold_indices

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from catboost import CatBoostRegressor

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# ── Paths ──────────────────────────────────────────────────────────────────────
UBD  = "C:/pxr_work/phase1_unblind"
OUT  = "C:/pxr_work/meta_stacking"
os.makedirs(OUT, exist_ok=True)

SEED = 42
N_SEEDS = 3

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...", flush=True)
tr_df = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
te_df = load_test().reset_index(drop=True)

print(f"  train: {len(tr_df)} | test (513): {len(te_df)}", flush=True)

# Load 253 unblinded labels
raw = pd.read_csv(f"{UBD}/phase1_unblinded_raw.csv")
name_col = [c for c in raw.columns if "name" in c.lower() or "molecule" in c.lower()][0]
pec_col  = [c for c in raw.columns if "pec50" in c.lower() or "activity" in c.lower()][0]
raw = raw[[name_col, pec_col]].dropna()
raw.columns = ["name", "pec50_true"]
print(f"  unblinded labels: {len(raw)} compounds", flush=True)

# Identify the 253 unblinded in te_df
unblind_mask = te_df["name"].isin(set(raw["name"]))
unblind_idx  = te_df.index[unblind_mask].tolist()
te_ub        = te_df[unblind_mask].merge(raw, on="name", how="left")
y_true_253   = te_ub["pec50_true"].to_numpy()
print(f"  matched {len(y_true_253)} unblinded in test | {(~unblind_mask).sum()} remain blinded", flush=True)

# ── Featurize ─────────────────────────────────────────────────────────────────
print("\nFeaturizing...", flush=True)
tr_smiles = tr_df["smiles"].tolist()
te_smiles  = te_df["smiles"].tolist()

X_tr_raw = combined(tr_smiles)
X_te_raw = combined(te_smiles)
X_tr = impute(X_tr_raw).astype(np.float32)
X_te = impute(X_te_raw).astype(np.float32)

y_tr = tr_df["pec50"].to_numpy()
se_tr = tr_df["pec50_se"].to_numpy() if "pec50_se" in tr_df.columns else np.ones(len(y_tr)) * 0.24

print(f"  X_tr: {X_tr.shape}  X_te: {X_te.shape}", flush=True)

# Morgan FPs for Tanimoto / KMM
print("Computing Morgan FPs...", flush=True)
fp_tr = morgan_fp_batch(tr_smiles).astype(np.float32)   # (4139, 2048)
fp_te = morgan_fp_batch(te_smiles).astype(np.float32)   # (513,  2048)

# Extract 253 test compounds (unblind) + all 513 for weights
fp_te_unb = fp_te[unblind_idx]   # (253, 2048) — unblinded only
# Use all 513 for weighting (we want to predict well on ALL test)
fp_te_all = fp_te                  # (513, 2048)

print(f"  fp_tr: {fp_tr.shape}  fp_te_all: {fp_te_all.shape}", flush=True)

# ── 4-GBM ensemble helpers ────────────────────────────────────────────────────

def make_lgbm(seed=42):
    return LGBMRegressor(
        n_estimators=600, num_leaves=63, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        n_jobs=4, verbose=-1, random_state=seed
    )

def make_xgb(seed=42):
    return XGBRegressor(
        n_estimators=600, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        n_jobs=4, verbosity=0, random_state=seed, tree_method="hist"
    )

def make_cat(seed=42):
    return CatBoostRegressor(
        iterations=600, depth=6, learning_rate=0.05,
        l2_leaf_reg=3.0, verbose=0, random_state=seed
    )

def make_histgb(seed=42):
    return HistGradientBoostingRegressor(
        max_iter=600, max_leaf_nodes=63, learning_rate=0.05,
        l2_regularization=1.0, random_state=seed
    )

MODELS = [make_lgbm, make_xgb, make_cat, make_histgb]


def fit_ensemble(X_tr_, y_tr_, X_te_, w=None, seeds=(0, 1, 2)):
    """Fit 4-GBM ensemble (each model, each seed) and return mean predictions."""
    preds = []
    for seed in seeds:
        for make_fn in MODELS:
            m = make_fn(seed=seed)
            name = type(m).__name__
            # Not all models support sample_weight in the same way
            try:
                if w is not None:
                    m.fit(X_tr_, y_tr_, sample_weight=w)
                else:
                    m.fit(X_tr_, y_tr_)
            except TypeError:
                # HistGradientBoostingRegressor sample_weight param name varies
                m.fit(X_tr_, y_tr_)
            preds.append(m.predict(X_te_))
    return np.mean(preds, axis=0)


def cv_rae_weighted(X, y, weights, scaffolds, n_splits=5, seeds=(0, 1, 2)):
    """Scaffold CV RAE with per-sample weights."""
    folds = scaffold_kfold_indices(scaffolds, n_splits=n_splits)
    oof = np.zeros(len(y))
    for tr_idx, va_idx in folds:
        Xf, yf, wf = X[tr_idx], y[tr_idx], weights[tr_idx] if weights is not None else None
        Xv = X[va_idx]
        oof[va_idx] = fit_ensemble(Xf, yf, Xv, w=wf, seeds=seeds)
    return rae(y, oof)


def eval_on_253(X_tr_, y_tr_, X_te_all_, unblind_idx_, y_true_, weights=None, seeds=(0, 1, 2)):
    """Train on full training set, predict all 513, extract 253."""
    preds_all = fit_ensemble(X_tr_, y_tr_, X_te_all_, w=weights, seeds=seeds)
    preds_253 = preds_all[unblind_idx_]
    return rae(y_true_, preds_253), preds_all


# ── Baseline (no weighting) ───────────────────────────────────────────────────
print("\n" + "="*70, flush=True)
print("BASELINE — no domain adaptation", flush=True)

t0 = time.time()
scaffolds_tr = tr_df["scaffold"].tolist() if "scaffold" in tr_df.columns else tr_df["smiles"].tolist()

print("  Scaffold CV (5-fold)...", flush=True)
baseline_cv_rae = cv_rae_weighted(X_tr, y_tr, None, scaffolds_tr, seeds=(0,))
print(f"  CV RAE: {baseline_cv_rae:.4f}", flush=True)

print("  Eval on 253 unblinded...", flush=True)
baseline_253_rae, baseline_preds_all = eval_on_253(
    X_tr, y_tr, X_te, unblind_idx, y_true_253, weights=None, seeds=(0, 1, 2)
)
print(f"  253 RAE: {baseline_253_rae:.4f}  ({time.time()-t0:.0f}s)", flush=True)
np.save(f"{OUT}/domain_adapt_baseline_te_513.npy", baseline_preds_all)


# ── Tanimoto helpers ───────────────────────────────────────────────────────────

def tanimoto_max_to_test(fp_train, fp_test_all):
    """For each training compound, compute max Tanimoto to ANY test compound.
    fp_train: (n_tr, 2048) float32  (binary Morgan)
    fp_test_all: (n_te, 2048) float32
    Returns (n_tr,) float — max Tanimoto per training compound.
    """
    # Tanimoto = |A & B| / |A | B| = dot(a,b) / (sum_a + sum_b - dot(a,b))
    # Vectorized in chunks to avoid OOM
    n_tr = fp_train.shape[0]
    n_te = fp_test_all.shape[0]
    sum_tr = fp_train.sum(axis=1, keepdims=True)  # (n_tr, 1)
    sum_te = fp_test_all.sum(axis=1, keepdims=True)  # (n_te, 1)

    max_tan = np.zeros(n_tr, dtype=np.float32)
    chunk = 64  # process 64 test compounds at a time
    for start in range(0, n_te, chunk):
        fp_chunk = fp_test_all[start:start+chunk]  # (c, 2048)
        dot = fp_train @ fp_chunk.T                 # (n_tr, c)
        s_te = sum_te[start:start+chunk].T          # (1, c)
        denom = sum_tr + s_te - dot                  # (n_tr, c)
        denom = np.where(denom == 0, 1e-6, denom)
        tan = dot / denom                             # (n_tr, c)
        max_tan = np.maximum(max_tan, tan.max(axis=1))
    return max_tan


# ── APPROACH A — Tanimoto importance weights ──────────────────────────────────
print("\n" + "="*70, flush=True)
print("APPROACH A — Tanimoto importance weights", flush=True)

t0 = time.time()
print("  Computing max Tanimoto (train vs all 513 test)...", flush=True)
max_tan = tanimoto_max_to_test(fp_tr, fp_te_all)
print(f"  max_tan: min={max_tan.min():.3f}  median={np.median(max_tan):.3f}  max={max_tan.max():.3f}", flush=True)

# Weight = max Tanimoto; normalize to sum to n_train; clip [0.1, 5.0]
w_a_raw = max_tan.copy()
w_a_raw = w_a_raw / w_a_raw.mean()          # normalize to mean=1 (sum=n_tr)
w_a = np.clip(w_a_raw, 0.1, 5.0)
w_a = w_a / w_a.mean()                      # re-normalize after clip
print(f"  weights_A: min={w_a.min():.3f}  median={np.median(w_a):.3f}  max={w_a.max():.3f}", flush=True)

print("  Scaffold CV (5-fold)...", flush=True)
a_cv_rae = cv_rae_weighted(X_tr, y_tr, w_a, scaffolds_tr, seeds=(0,))
print(f"  CV RAE: {a_cv_rae:.4f}", flush=True)

print("  Eval on 253 unblinded...", flush=True)
a_253_rae, a_preds_all = eval_on_253(
    X_tr, y_tr, X_te, unblind_idx, y_true_253, weights=w_a, seeds=(0, 1, 2)
)
print(f"  253 RAE: {a_253_rae:.4f}  delta vs baseline: {a_253_rae - baseline_253_rae:+.4f}  ({time.time()-t0:.0f}s)", flush=True)
np.save(f"{OUT}/domain_adapt_A_te_513.npy", a_preds_all)


# ── APPROACH B — Kernel Mean Matching (KMM) ───────────────────────────────────
print("\n" + "="*70, flush=True)
print("APPROACH B — Kernel Mean Matching (KMM)", flush=True)

t0 = time.time()

def kmm_weights(fp_train, fp_test, B=5.0, eps=None, sigma=None, n_iter=200):
    """Approximate KMM using Gaussian RBF kernel on Morgan FP.
    Minimize  0.5 * w^T K_tr w - k_tr_te^T w
    s.t.  0 <= w <= B,  |sum(w) - n_tr| <= eps*n_tr

    We use a simple projected gradient approach (closed-form per-entry update).
    sigma: RBF bandwidth (None = median heuristic on 500-sample subset)
    """
    from scipy.spatial.distance import cdist

    n_tr = fp_train.shape[0]
    n_te = fp_test.shape[0]
    if eps is None:
        eps = (np.sqrt(n_tr) - 1) / np.sqrt(n_tr)

    # Estimate sigma via median heuristic on a subsample to save memory
    idx_s = np.random.RandomState(42).choice(n_tr, min(500, n_tr), replace=False)
    sub = fp_train[idx_s].astype(np.float64)
    d2_sub = cdist(sub, sub, "sqeuclidean")
    if sigma is None:
        sigma = np.sqrt(np.median(d2_sub[d2_sub > 0]) / 2.0)
    print(f"    KMM sigma={sigma:.3f}", flush=True)

    # Kernel matrix K_tr_tr (chunked to avoid OOM — n_tr=4139 x 4139 ~68MB float32)
    K = np.zeros((n_tr, n_tr), dtype=np.float32)
    chunk = 256
    fp64_tr = fp_train.astype(np.float64)
    fp64_te = fp_test.astype(np.float64)
    for i in range(0, n_tr, chunk):
        d2 = cdist(fp64_tr[i:i+chunk], fp64_tr, "sqeuclidean")
        K[i:i+chunk] = np.exp(-d2 / (2 * sigma**2)).astype(np.float32)

    # k_tr_te: mean kernel from each train compound to all test compounds
    k_te = np.zeros(n_tr, dtype=np.float32)
    for i in range(0, n_tr, chunk):
        d2 = cdist(fp64_tr[i:i+chunk], fp64_te, "sqeuclidean")
        k_te[i:i+chunk] = np.exp(-d2 / (2 * sigma**2)).mean(axis=1).astype(np.float32)
    k_te_scaled = k_te * (n_tr / n_te)

    # Projected gradient ascent (maximize w^T k_te_scaled - 0.5 w^T K w)
    w = np.ones(n_tr, dtype=np.float64)
    lr = 1.0 / (np.diag(K).max() + 1e-6)
    for it in range(n_iter):
        grad = k_te_scaled.astype(np.float64) - K.astype(np.float64) @ w
        w = w + lr * grad
        # Project onto [0, B]
        w = np.clip(w, 0.0, B)
        # Satisfy sum constraint: |sum(w) - n_tr| <= eps*n_tr
        s = w.sum()
        if abs(s - n_tr) > eps * n_tr:
            w = w * n_tr / s
        if it % 50 == 0:
            obj = w @ k_te_scaled - 0.5 * w @ (K @ w)
            print(f"    iter {it:3d}  obj={obj:.4f}  sum_w={w.sum():.1f}  B-clip={np.sum(w>=B-1e-3)}", flush=True)

    return w.astype(np.float32)


print("  Running KMM projected gradient (n_iter=200)...", flush=True)
w_b_raw = kmm_weights(fp_tr, fp_te_all, B=5.0, n_iter=200)
w_b = np.clip(w_b_raw, 0.1, 5.0)
w_b = w_b / w_b.mean()
print(f"  weights_B: min={w_b.min():.3f}  median={np.median(w_b):.3f}  max={w_b.max():.3f}", flush=True)

print("  Scaffold CV (5-fold)...", flush=True)
b_cv_rae = cv_rae_weighted(X_tr, y_tr, w_b, scaffolds_tr, seeds=(0,))
print(f"  CV RAE: {b_cv_rae:.4f}", flush=True)

print("  Eval on 253 unblinded...", flush=True)
b_253_rae, b_preds_all = eval_on_253(
    X_tr, y_tr, X_te, unblind_idx, y_true_253, weights=w_b, seeds=(0, 1, 2)
)
print(f"  253 RAE: {b_253_rae:.4f}  delta vs baseline: {b_253_rae - baseline_253_rae:+.4f}  ({time.time()-t0:.0f}s)", flush=True)
np.save(f"{OUT}/domain_adapt_B_te_513.npy", b_preds_all)


# ── APPROACH C — Adversarial domain adaptation ────────────────────────────────
print("\n" + "="*70, flush=True)
print("APPROACH C — Adversarial domain adaptation (train/test classifier)", flush=True)

t0 = time.time()

from sklearn.ensemble import GradientBoostingClassifier

# Build domain classification dataset
# train compounds → label 0, test compounds → label 1
n_tr = len(tr_df)
n_te = len(te_df)

X_dom = np.vstack([fp_tr.astype(np.float32), fp_te_all.astype(np.float32)])
y_dom = np.array([0] * n_tr + [1] * n_te)

print(f"  Domain classifier: {n_tr} train (0) + {n_te} test (1)", flush=True)
clf = GradientBoostingClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.1,
    subsample=0.8, random_state=42, verbose=0
)
clf.fit(X_dom, y_dom)

# Predict domain proba for training compounds
p_test_given_tr = clf.predict_proba(fp_tr.astype(np.float32))[:, 1]   # P(test | x)
p_train_given_tr = 1.0 - p_test_given_tr                               # P(train | x)

# Importance ratio: w = P(test|x) / P(train|x), clipped
p_train_clipped = np.clip(p_train_given_tr, 0.05, 1.0)
w_c_raw = p_test_given_tr / p_train_clipped
w_c = np.clip(w_c_raw, 0.1, 5.0)
w_c = w_c / w_c.mean()
print(f"  Domain AUC check: {clf.score(X_dom[:n_tr], y_dom[:n_tr]):.3f} on train", flush=True)
print(f"  weights_C: min={w_c.min():.3f}  median={np.median(w_c):.3f}  max={w_c.max():.3f}", flush=True)
print(f"  P(test|train): min={p_test_given_tr.min():.3f}  mean={p_test_given_tr.mean():.3f}  max={p_test_given_tr.max():.3f}", flush=True)

# Domain classifier accuracy on test set (sanity check)
acc_test = clf.score(fp_te_all.astype(np.float32), np.ones(n_te))
print(f"  Classifier accuracy on test (predict=1): {acc_test:.3f}", flush=True)

print("  Scaffold CV (5-fold)...", flush=True)
c_cv_rae = cv_rae_weighted(X_tr, y_tr, w_c, scaffolds_tr, seeds=(0,))
print(f"  CV RAE: {c_cv_rae:.4f}", flush=True)

print("  Eval on 253 unblinded...", flush=True)
c_253_rae, c_preds_all = eval_on_253(
    X_tr, y_tr, X_te, unblind_idx, y_true_253, weights=w_c, seeds=(0, 1, 2)
)
print(f"  253 RAE: {c_253_rae:.4f}  delta vs baseline: {c_253_rae - baseline_253_rae:+.4f}  ({time.time()-t0:.0f}s)", flush=True)
np.save(f"{OUT}/domain_adapt_C_te_513.npy", c_preds_all)


# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*70, flush=True)
print("SUMMARY", flush=True)
print(f"{'Method':<35} {'CV RAE':>8} {'253 RAE':>9} {'delta_253':>10}", flush=True)
print("-"*65, flush=True)
results = {
    "baseline": {
        "cv_rae": float(baseline_cv_rae),
        "rae_253": float(baseline_253_rae),
        "delta_vs_baseline": 0.0
    },
    "approach_A_tanimoto": {
        "cv_rae": float(a_cv_rae),
        "rae_253": float(a_253_rae),
        "delta_vs_baseline": float(a_253_rae - baseline_253_rae),
        "weight_stats": {
            "min": float(w_a.min()), "median": float(np.median(w_a)), "max": float(w_a.max())
        }
    },
    "approach_B_kmm": {
        "cv_rae": float(b_cv_rae),
        "rae_253": float(b_253_rae),
        "delta_vs_baseline": float(b_253_rae - baseline_253_rae),
        "weight_stats": {
            "min": float(w_b.min()), "median": float(np.median(w_b)), "max": float(w_b.max())
        }
    },
    "approach_C_adversarial": {
        "cv_rae": float(c_cv_rae),
        "rae_253": float(c_253_rae),
        "delta_vs_baseline": float(c_253_rae - baseline_253_rae),
        "weight_stats": {
            "min": float(w_c.min()), "median": float(np.median(w_c)), "max": float(w_c.max())
        },
        "domain_clf_acc_on_test": float(acc_test)
    }
}

for method, res in results.items():
    delta = res["delta_vs_baseline"]
    flag = " <<" if delta < -0.003 else ""
    print(f"  {method:<33} {res['cv_rae']:>8.4f} {res['rae_253']:>9.4f} {delta:>+10.4f}{flag}", flush=True)

json.dump(results, open(f"{OUT}/domain_adapt_results.json", "w"), indent=2)
print(f"\nResults saved to {OUT}/domain_adapt_results.json", flush=True)
print("Predictions saved:", flush=True)
print(f"  {OUT}/domain_adapt_baseline_te_513.npy", flush=True)
print(f"  {OUT}/domain_adapt_A_te_513.npy", flush=True)
print(f"  {OUT}/domain_adapt_B_te_513.npy", flush=True)
print(f"  {OUT}/domain_adapt_C_te_513.npy", flush=True)
