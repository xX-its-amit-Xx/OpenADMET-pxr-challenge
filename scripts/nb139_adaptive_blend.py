"""nb139 — Compound-Adaptive Ensemble Blend.

The nb129 best k=3 ensemble uses FIXED weights:
  nb109_calib(0.498) + nb107_calib(0.251) + counter_delta(0.251)

But the optimal weights may vary by compound. For example:
  - Compounds with high counter-assay similarity → counter_delta more reliable
  - Compounds with many training analogs → nb109 (meta-stack) more reliable
  - Compounds near training distribution center → nb107 (assay decomp) better

Strategy: train a "gating" LGBM that learns per-compound ensemble weights.
  1. For each training compound (OOF), compute features describing its position
     in chemical space, counter-assay similarity, neighbor statistics.
  2. Train a gating model that predicts WHICH base model will be most accurate.
  3. Apply gating model to test compounds.

Implementation:
  - For each fold, compute per-fold model errors: e_i = (y - m_i)² for each model i
  - Train gating LGBM to predict e_1 - e_2 (signed error difference)
  - Use sigmoid gating: w_1 = sigma(gate_pred), w_2 = 1 - sigma(gate_pred)
  - For the k=3 ensemble, learn 2D weights via softmax gating
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import special, stats

from pxr.data import load_train, load_test
from pxr.featurize import combined, impute
from pxr.chem import morgan_fp_batch, bemis_murcko
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
N_FOLDS = 5
LGBM_GATE = dict(
    n_estimators=200, num_leaves=16, learning_rate=0.05,
    min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, verbose=-1, n_jobs=4
)
LGBM_AUX = dict(
    n_estimators=400, num_leaves=32, learning_rate=0.05,
    min_child_samples=8, subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, verbose=-1, n_jobs=4
)
K_COUNTER = 10
K_TRAIN = 10


def tanimoto_topk(fp_q, fp_ref, k=10, batch=300):
    """Top-k Tanimoto similarity for query against reference."""
    nq, nr = len(fp_q), len(fp_ref)
    k = min(k, nr)
    top_sims = np.zeros((nq, k), dtype=np.float32)
    top_idx  = np.zeros((nq, k), dtype=np.int32)
    sum_r = fp_ref.sum(axis=1)
    for start in range(0, nq, batch):
        end = min(start + batch, nq)
        fp_b = fp_q[start:end]
        dot = fp_b @ fp_ref.T
        sum_b = fp_b.sum(axis=1, keepdims=True)
        denom = sum_b + sum_r.reshape(1, -1) - dot
        denom[denom == 0] = 1e-6
        sim = dot / denom
        idx = np.argpartition(-sim, k, axis=1)[:, :k]
        sims_k = np.take_along_axis(sim, idx, axis=1)
        order = np.argsort(-sims_k, axis=1)
        top_sims[start:end] = np.take_along_axis(sims_k, order, axis=1)
        top_idx[start:end]  = np.take_along_axis(idx, order, axis=1)
    return top_sims, top_idx


def compute_gating_features(fp_q, fp_counter, fp_train, y_train,
                              counter_pec50, pxr_pred_109, pxr_pred_107):
    """Compute per-compound gating features."""
    # Counter-assay neighborhood
    csims, cidx = tanimoto_topk(fp_q, fp_counter, K_COUNTER)
    cpec50 = counter_pec50[cidx]
    w_c = csims / (csims.sum(axis=1, keepdims=True) + 1e-9)
    counter_knn = (w_c * cpec50).sum(axis=1)
    counter_top1 = csims[:, 0]
    counter_mean = csims.mean(axis=1)

    # Training neighborhood
    tsims, tidx = tanimoto_topk(fp_q, fp_train, K_TRAIN)
    ty = y_train[tidx]
    w_t = tsims / (tsims.sum(axis=1, keepdims=True) + 1e-9)
    train_knn = (w_t * ty).sum(axis=1)
    train_top1 = tsims[:, 0]
    train_std  = ty.std(axis=1)

    # Model disagreement
    model_diff = pxr_pred_109 - pxr_pred_107

    return np.column_stack([
        counter_knn, counter_top1, counter_mean,
        train_knn, train_top1, train_std,
        model_diff,
        np.abs(model_diff),
        pxr_pred_109, pxr_pred_107,
    ])


def full_metrics(y_true, y_pred, label=""):
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    msk = np.isfinite(yt) & np.isfinite(yp); yt, yp = yt[msk], yp[msk]
    mae_v = float(np.mean(np.abs(yt - yp)))
    rae_v = mae_v / float(np.mean(np.abs(yt - yt.mean()))) if yt.std() > 0 else np.nan
    pr, _ = stats.pearsonr(yt, yp)
    if label:
        print(f"  [{label:55s}] RAE={rae_v:.4f}  MAE={mae_v:.4f}  r={pr:.4f}")
    return rae_v


def main():
    print("=== nb139: Compound-Adaptive Ensemble Blend ===\n")

    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    raw_counter = pd.read_csv("data/raw/pxr-challenge_counter-assay_TRAIN.csv")
    counter_smiles = raw_counter["SMILES"].fillna("").tolist()
    counter_pec50  = raw_counter["pEC50"].fillna(raw_counter["pEC50"].median()).values.astype(np.float64)

    # Load core model OOF predictions
    def load_oof(stem, n_tr):
        for op in ("oof_", ""):
            p = DATA_PROCESSED / f"{op}{stem}.npy"
            if p.exists():
                arr = np.load(p).astype(np.float64)
                if arr.ndim == 2: arr = arr[:, 0]
                if len(arr) == n_tr: return arr
        return None

    def load_te(stem):
        for tp in ("te_", "te_oof_"):
            for op in ("", "nb"):
                p = DATA_PROCESSED / f"{tp}{stem}.npy"
                if p.exists():
                    arr = np.load(p).astype(np.float64)
                    if arr.ndim == 2: arr = arr[:, 0]
                    return arr
        return None

    oof_109 = load_oof("nb109_deep_meta_stack_calib", n_tr)
    oof_107 = load_oof("nb107_assay_decomp_calib", n_tr)
    oof_cd  = load_oof("counter_delta", n_tr)
    te_109  = load_te("nb109_deep_meta_stack_calib")
    te_107  = load_te("nb107_assay_decomp_calib")
    te_cd   = load_te("counter_delta")

    if any(x is None for x in [oof_109, oof_107, oof_cd, te_109, te_107, te_cd]):
        print("ERROR: Missing core model predictions")
        return

    print("Core model OOF RAEs:")
    print(f"  nb109_calib: {rae(y_tr, oof_109):.4f}")
    print(f"  nb107_calib: {rae(y_tr, oof_107):.4f}")
    print(f"  counter_delta: {rae(y_tr, oof_cd):.4f}")
    core_fixed = 0.498 * oof_109 + 0.251 * oof_107 + 0.251 * oof_cd
    print(f"  Fixed-weight k=3: {rae(y_tr, core_fixed):.4f}")

    print("\nComputing Morgan fingerprints...")
    fp_tr = morgan_fp_batch(tr["smiles"].tolist()).astype(np.float32)
    fp_te = morgan_fp_batch(te["smiles"].tolist()).astype(np.float32)
    fp_counter = morgan_fp_batch(counter_smiles).astype(np.float32)

    # Full structural features
    X_str = impute(combined(tr["smiles"].tolist()))
    X_str_te = impute(combined(te["smiles"].tolist()))

    # Test gating features (uses full training set as reference)
    print("Computing gating features for test set...")
    gate_te = compute_gating_features(fp_te, fp_counter, fp_tr, y_tr,
                                       counter_pec50, te_109, te_107)
    gate_te_full = np.hstack([gate_te, X_str_te])

    print("\n=== CV: Adaptive Blend ===")
    oof_adaptive = np.full(n_tr, np.nan)
    gate_target  = np.zeros(n_tr)  # will store per-fold gating targets

    # Compute per-fold gating target: for each sample in val,
    # the optimal weight for nb109 vs. 0.5*(nb107+counter_delta)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        oof_109_va = oof_109[va_idx]
        oof_107_va = oof_107[va_idx]
        oof_cd_va  = oof_cd[va_idx]
        y_va = y_tr[va_idx]

        # Error of each model
        err_109 = (y_va - oof_109_va)**2
        err_other = (y_va - 0.5*(oof_107_va + oof_cd_va))**2
        # Gating target: positive = prefer nb109, negative = prefer other
        gate_target[va_idx] = err_other - err_109

        # Gating features using only training fold as neighbor reference
        gate_tr_fold = compute_gating_features(
            fp_tr[tr_idx], fp_counter, fp_tr[tr_idx], y_tr[tr_idx],
            counter_pec50, oof_109[tr_idx], oof_107[tr_idx]
        )
        gate_va_fold = compute_gating_features(
            fp_tr[va_idx], fp_counter, fp_tr[tr_idx], y_tr[tr_idx],
            counter_pec50, oof_109_va, oof_107_va
        )
        X_gate_tr = np.hstack([gate_tr_fold, X_str[tr_idx]])
        X_gate_va = np.hstack([gate_va_fold, X_str[va_idx]])

        m_gate = lgb.train(LGBM_GATE, lgb.Dataset(X_gate_tr, label=gate_target[tr_idx]),
                           callbacks=[lgb.log_evaluation(-1)])
        gate_score_va = m_gate.predict(X_gate_va)
        # Convert to weight: sigmoid of gate score
        w_109_va = special.expit(gate_score_va * 2)  # scale for sigmoid
        oof_adaptive[va_idx] = w_109_va * oof_109_va + (1 - w_109_va) * (0.5*oof_107_va + 0.5*oof_cd_va)
        r = rae(y_va, oof_adaptive[va_idx])
        r_fixed = rae(y_va, 0.498*oof_109_va + 0.251*oof_107_va + 0.251*oof_cd_va)
        print(f"  fold {fold+1}  RAE={r:.4f} (fixed={r_fixed:.4f})", flush=True)

    full_metrics(y_tr, oof_adaptive, "Adaptive blend (gating)")
    full_metrics(y_tr, core_fixed,   "Fixed k=3 blend (reference)")

    # Full gating model
    gate_tr_full = compute_gating_features(fp_tr, fp_counter, fp_tr, y_tr,
                                            counter_pec50, oof_109, oof_107)
    X_gate_full = np.hstack([gate_tr_full, X_str])
    m_gate_full = lgb.train(LGBM_GATE, lgb.Dataset(X_gate_full, label=gate_target),
                            callbacks=[lgb.log_evaluation(-1)])
    gate_score_te = m_gate_full.predict(gate_te_full)
    w_109_te = special.expit(gate_score_te * 2)
    te_adaptive = w_109_te * te_109 + (1 - w_109_te) * (0.5*te_107 + 0.5*te_cd)
    te_adaptive = np.clip(te_adaptive, y_tr.min() - 0.5, y_tr.max() + 0.5)
    ratio = te_adaptive.std() / oof_adaptive.std()
    print(f"  Test: med={np.median(te_adaptive):.2f}  std={te_adaptive.std():.3f}  ratio={ratio:.2f}")
    print(f"  Gate weight stats: mean w_109={w_109_te.mean():.3f}  "
          f"std={w_109_te.std():.3f}  min={w_109_te.min():.3f}  max={w_109_te.max():.3f}")

    np.save(DATA_PROCESSED / "oof_nb139_adaptive_blend.npy", oof_adaptive)
    np.save(DATA_PROCESSED / "te_nb139_adaptive_blend.npy",  te_adaptive)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_adaptive})
    sub.to_csv(SUBMISSIONS / "139_adaptive_blend.csv", index=False)
    print(f"\nSaved: submissions/139_adaptive_blend.csv")
    print(f"OOF RAE: {rae(y_tr, oof_adaptive):.4f}")


if __name__ == "__main__":
    main()
