"""nb1132 -- DEEPER LGBM Huber on the SIGNED residual of nb1070.

Same protocol as nb1123 but with INCREASED LGBM capacity:
  max_depth=5, num_leaves=15, n_estimators=200, learning_rate=0.03,
  min_child_samples=15.

Hypothesis: nb1123's shallow probe (depth=3, n_est=80) at pooled
RAE 0.5704 either (a) under-fits residual structure that more capacity
can extract, or (b) is already saturated and deeper trees will simply
memorize n=253 train-fold noise. With ~200 samples per training fold,
2nd-order feature interactions on a 2265-dim X are firmly in the
overfit-risk regime. The Huber objective + reg_lambda=1.0 and the
honest 5-fold cross-fit are the only buffers.

Decision rule (vs nb1123 shallow 0.5704):
  - delta <= -0.003  -> DEEPER_HELPS (real signal hiding in interactions)
  - |delta| <  0.003 -> DEEPER_NEUTRAL (capacity-irrelevant)
  - delta >= +0.003  -> DEEPER_OVERFITS (noise memorization, shallow was right)

Procedure: identical to nb1123 except the LGBM_PARAMS block.
  1. Regenerate / load nb1070 cross-fit OOF on 253 unblind.
  2. residual = y_unb - nb1070_oof
  3. X = combined(Morgan + RDKit) on 253 unblind SMILES.
  4. Honest 5-fold cross-fit DEEPER LGBM Huber.
  5. pred_corrected = nb1070_oof + residual_oof; pooled RAE.

Outputs:
  data/processed/nb1132_residual_oof.npy
  data/processed/nb1132_pred_oof.npy
  data/processed/nb1132_summary.json
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor

from pxr.data import load_test
from pxr.eval import rae
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED

TAG = "nb1132"
ANCHOR = "nb1070"          # median-bag anchor whose residual we correct
STRETCH_ANCHOR = "nb1014"  # nb1070 stretches this; needed to regen OOF

# nb1070 protocol constants (re-derive its 253 OOF in-script to avoid
# stale file dependency).
N_BINS = 5
STRETCH_GRID = np.round(np.arange(0.80, 2.001, 0.05), 2).tolist()
N_FOLDS = 5
NB1070_SEEDS = [0, 1, 7, 42, 137]

# Residual model: DEEPER LGBM Huber.
RESID_SEED = 42
RESID_FOLDS = 5
LGBM_PARAMS = dict(
    objective="huber",
    alpha=1.0,
    learning_rate=0.03,
    n_estimators=200,
    max_depth=5,
    num_leaves=15,           # 2^4 - 1 (leaves cap one below 2^max_depth)
    min_child_samples=15,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    verbosity=-1,
    random_state=RESID_SEED,
    n_jobs=2,
)

NB1070_REF_POOLED = 0.5790  # nb1070 median-bag honest cross-fit pooled RAE
NB1123_REF_POOLED = 0.5704  # nb1123 shallow residual cross-fit pooled RAE


# ---------- nb1070 cross-fit OOF regeneration (verbatim from nb1123) ----------

def _bin_assign(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(values, edges, right=False), 0, N_BINS - 1)


def _fit_per_bin_stretch(p_train: np.ndarray, y_train: np.ndarray,
                         edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bins_tr = _bin_assign(p_train, edges)
    mus = np.zeros(N_BINS, dtype=np.float64)
    ss = np.ones(N_BINS, dtype=np.float64)
    for b in range(N_BINS):
        mask = bins_tr == b
        if int(mask.sum()) < 2:
            mus[b] = float(p_train.mean())
            ss[b] = 1.0
            continue
        mu_b = float(p_train[mask].mean())
        mus[b] = mu_b
        y_b, p_b = y_train[mask], p_train[mask]
        best_s, best_r = 1.0, float("inf")
        for s in STRETCH_GRID:
            stretched = mu_b + s * (p_b - mu_b)
            r = float(rae(y_b, stretched))
            if r < best_r:
                best_r, best_s = r, float(s)
        ss[b] = best_s
    return mus, ss


def _apply_per_bin_stretch(p: np.ndarray, edges: np.ndarray,
                           mus: np.ndarray, ss: np.ndarray) -> np.ndarray:
    bins = _bin_assign(p, edges)
    out = np.empty_like(p, dtype=np.float64)
    for b in range(N_BINS):
        mask = bins == b
        if not mask.any():
            continue
        out[mask] = mus[b] + ss[b] * (p[mask] - mus[b])
    return out


def _nb1070_seed_oof(seed: int, p_unb: np.ndarray,
                     y_unb: np.ndarray) -> np.ndarray:
    n = len(y_unb)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        p_tr, y_tr = p_unb[tr_loc], y_unb[tr_loc]
        qs = np.linspace(0.0, 1.0, N_BINS + 1)[1:-1]
        edges = np.quantile(p_tr, qs)
        mus, ss = _fit_per_bin_stretch(p_tr, y_tr, edges)
        oof[va_loc] = _apply_per_bin_stretch(p_unb[va_loc], edges, mus, ss)
    return oof


def regenerate_nb1070_oof(p_unb: np.ndarray, y_unb: np.ndarray) -> np.ndarray:
    """Median bag of nb1070 OOF across the 5 nb1070 KFold seeds."""
    stack = np.zeros((len(NB1070_SEEDS), len(y_unb)), dtype=np.float64)
    for i, s in enumerate(NB1070_SEEDS):
        stack[i] = _nb1070_seed_oof(s, p_unb, y_unb)
    return np.median(stack, axis=0)


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- DEEPER LGBM Huber on (truth - nb1070_oof) residual")
    print(f"          features = combined Morgan+RDKit (2265)")
    print(f"          LGBM: max_depth=5, num_leaves=15, n_est=200, lr=0.03, "
          f"min_child_samples=15, obj=huber(alpha=1.0)")
    print(f"          baselines: nb1070={NB1070_REF_POOLED:.4f}  "
          f"nb1123_shallow={NB1123_REF_POOLED:.4f}")
    print("=" * 78)

    # ---- Load 513 test, unblind index + truth ----
    te = load_test()
    te_smiles = te["smiles"].values
    preds_513_anchor = np.load(
        DATA_PROCESSED / f"te_{STRETCH_ANCHOR}.npy").astype(np.float64)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    p_unb = preds_513_anchor[unb_idx]
    n_unb = len(y_unb)
    print(f"[load] te_{STRETCH_ANCHOR}.npy = {preds_513_anchor.shape}, "
          f"y_unb = {y_unb.shape}")

    # ---- Regenerate nb1070 honest cross-fit OOF (median bag of 5 seeds) ----
    nb1070_oof_path = DATA_PROCESSED / f"{ANCHOR}_pred_oof.npy"
    if nb1070_oof_path.exists():
        nb1070_oof = np.load(nb1070_oof_path).astype(np.float64)
        if nb1070_oof.shape[0] != n_unb:
            print(f"[regen] stale {ANCHOR}_pred_oof.npy "
                  f"(shape {nb1070_oof.shape}), regenerating")
            nb1070_oof = regenerate_nb1070_oof(p_unb, y_unb)
            np.save(nb1070_oof_path, nb1070_oof)
        else:
            print(f"[load] {ANCHOR}_pred_oof.npy shape={nb1070_oof.shape}")
    else:
        print(f"[regen] generating {ANCHOR}_pred_oof.npy "
              f"(median bag, {len(NB1070_SEEDS)} seeds)")
        nb1070_oof = regenerate_nb1070_oof(p_unb, y_unb)
        np.save(nb1070_oof_path, nb1070_oof)
        print(f"[save] {nb1070_oof_path}")

    rae_anchor_oof = float(rae(y_unb, nb1070_oof))
    print(f"[base] pooled RAE(nb1070 median-bag OOF, 253) = "
          f"{rae_anchor_oof:.4f}  (ref ~{NB1070_REF_POOLED:.4f})")

    # ---- Signed residual ----
    residual = y_unb - nb1070_oof
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}  "
          f"min={residual.min():+.4f}  max={residual.max():+.4f}")

    # ---- Features: combined Morgan+RDKit on unblind SMILES ----
    smi_unb = te_smiles[unb_idx].tolist()
    print(f"[feat] computing combined(Morgan+RDKit) on n={len(smi_unb)} "
          f"unblind SMILES")
    X_unb = impute(combined(smi_unb))
    print(f"[feat] X_unb shape = {X_unb.shape}")

    # ---- Honest 5-fold cross-fit DEEPER LGBM on residual ----
    print("\n" + "-" * 78)
    print(f"RESIDUAL CROSS-FIT  (KFold n={RESID_FOLDS}, seed={RESID_SEED})")
    print("-" * 78)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=RESID_SEED)
    residual_oof = np.full(n_unb, np.nan, dtype=np.float64)
    fold_records = []
    for k, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n_unb))):
        mdl = LGBMRegressor(**LGBM_PARAMS)
        mdl.fit(X_unb[tr_loc], residual[tr_loc])
        r_pred = mdl.predict(X_unb[va_loc])
        residual_oof[va_loc] = r_pred
        # Fold diagnostics
        resid_va_true = residual[va_loc]
        r2 = 1.0 - (((resid_va_true - r_pred) ** 2).sum()
                    / max(((resid_va_true - resid_va_true.mean()) ** 2).sum(),
                          1e-12))
        sp_fold = float(spearmanr(resid_va_true, r_pred).correlation)
        pred_corr_va = nb1070_oof[va_loc] + r_pred
        rae_base = float(rae(y_unb[va_loc], nb1070_oof[va_loc]))
        rae_corr = float(rae(y_unb[va_loc], pred_corr_va))
        fold_records.append({
            "fold": k,
            "n_va": int(len(va_loc)),
            "rae_base": rae_base,
            "rae_corr": rae_corr,
            "delta": rae_corr - rae_base,
            "resid_r2_on_va": float(r2),
            "spearman_resid_pred_vs_true_va": sp_fold,
            "pred_resid_mean": float(r_pred.mean()),
            "pred_resid_std": float(r_pred.std()),
        })
        print(f"   fold {k}: n_va={len(va_loc):3d}  "
              f"base_RAE={rae_base:.4f}  corr_RAE={rae_corr:.4f}  "
              f"(d={rae_corr - rae_base:+.4f})  "
              f"resid_R2_va={r2:+.3f}  rho={sp_fold:+.3f}  "
              f"|pred_resid|_std={r_pred.std():.3f}")

    # ---- Apply correction and report pooled ----
    pred_corrected_oof = nb1070_oof + residual_oof
    pooled_rae_base = float(rae(y_unb, nb1070_oof))
    pooled_rae_corr = float(rae(y_unb, pred_corrected_oof))
    delta_vs_nb1070 = pooled_rae_corr - pooled_rae_base
    delta_vs_nb1123 = pooled_rae_corr - NB1123_REF_POOLED
    rho_resid_truth = float(spearmanr(residual_oof, residual).correlation)
    rho_pred_truth_corr = float(spearmanr(pred_corrected_oof, y_unb).correlation)
    rho_pred_truth_base = float(spearmanr(nb1070_oof, y_unb).correlation)

    print("\n" + "-" * 78)
    print("POOLED CROSS-FIT")
    print("-" * 78)
    print(f"   pooled RAE base (nb1070_oof)         = {pooled_rae_base:.4f}")
    print(f"   pooled RAE corrected (oof + resid)   = {pooled_rae_corr:.4f}")
    print(f"   delta vs nb1070 base                 = "
          f"{delta_vs_nb1070:+.4f}")
    print(f"   delta vs nb1123 shallow (0.5704)     = "
          f"{delta_vs_nb1123:+.4f}")
    print(f"   |residual_oof|.std                    = "
          f"{residual_oof.std():.4f}")
    print(f"   rho(residual_oof, residual_truth)    = {rho_resid_truth:+.4f}")
    print(f"   rho(pred_corrected, truth)           = "
          f"{rho_pred_truth_corr:+.4f}  "
          f"(base {rho_pred_truth_base:+.4f})")

    if delta_vs_nb1123 <= -0.003:
        verdict = "DEEPER_HELPS"
    elif abs(delta_vs_nb1123) < 0.003:
        verdict = "DEEPER_NEUTRAL"
    else:
        verdict = "DEEPER_OVERFITS"
    print(f"   verdict vs nb1123                    = {verdict}")

    # ---- Save artefacts ----
    np.save(DATA_PROCESSED / f"{TAG}_residual_oof.npy",
            residual_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy",
            pred_corrected_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_residual_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_pred_oof.npy'}")

    beats_nb1070 = pooled_rae_corr < pooled_rae_base - 0.003
    beats_nb1123 = pooled_rae_corr < NB1123_REF_POOLED - 0.003

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "stretch_anchor": STRETCH_ANCHOR,
        "n_unb": n_unb,
        "lgbm_params": LGBM_PARAMS,
        "resid_folds": RESID_FOLDS,
        "resid_seed": RESID_SEED,
        "feature_dim": int(X_unb.shape[1]),
        "pooled_rae_base": pooled_rae_base,
        "pooled_rae_corrected": pooled_rae_corr,
        "delta_vs_nb1070": delta_vs_nb1070,
        "delta_vs_nb1123": delta_vs_nb1123,
        "beats_nb1070": bool(beats_nb1070),
        "beats_nb1123": bool(beats_nb1123),
        "verdict": verdict,
        "residual_oof_std": float(residual_oof.std()),
        "residual_oof_mean": float(residual_oof.mean()),
        "residual_truth_std": float(residual.std()),
        "spearman_resid_oof_vs_resid_truth": rho_resid_truth,
        "spearman_pred_corrected_vs_truth": rho_pred_truth_corr,
        "spearman_nb1070_vs_truth": rho_pred_truth_base,
        "rae_anchor_on_253": rae_anchor_oof,
        "nb1070_ref_pooled": NB1070_REF_POOLED,
        "nb1123_ref_pooled": NB1123_REF_POOLED,
        "fold_records": fold_records,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("pooled_rae_base", "pooled_rae_corrected",
              "delta_vs_nb1070", "delta_vs_nb1123",
              "beats_nb1070", "beats_nb1123", "verdict",
              "spearman_resid_oof_vs_resid_truth",
              "spearman_pred_corrected_vs_truth",
              "residual_oof_std", "residual_truth_std"):
        print(f"  {k}: {res.get(k)}")
