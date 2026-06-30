"""nb2982 -- Per-fold SLSQP simplex on JUST {K18, K20} (the auto-zeroed subset).

NEW PARADIGM:
    nb2973 (4-anchor pool {K18, K20, K24, K28}) deploy SLSQP collapsed
    on FULL 253 to K18=0.7015, K20=0.2985, K24=K28=0. The simplex
    optimizer auto-zeroed K24 and K28. This script asks: does the
    explicit 2-anchor pool {K18, K20} reproduce the same per-fold and
    deploy result?

    Predictions:
        - Per-fold weights may shift slightly because in 4-pool folds
          K24/K28 occasionally took non-zero mass (fold 2: K24=0.33,
          fold 4: K24=0.01). Removing them forces those mass shares
          onto K18+K20.
        - Deploy weights should land near {K18=0.70, K20=0.30}.
        - Pooled outer-val RAE should be within ~0.005 of nb2973
          (0.4539). If significantly better, the auto-zeroed pool was
          adding fold-train noise. If worse, K24/K28 carried per-fold
          signal even though they zeroed at deploy.

PROTOCOL:
    Anchors: nb2960 deep-30 fresh-seed OOFs + te arrays for K=18, 20
    Outer CV: 5-fold scaffold split on 253 unblind, kf_seed=1001
    Per fold:
        - SLSQP minimize fold-train RAE on simplex (w >= 0, sum w = 1)
        - 8 multi-starts (uniform + 7 Dirichlet draws)
        - Apply per-fold weights to held-out fold-val slice
    Pooled RAE on 5 outer-val folds = mean_rae (gate metric).

    Deploy:
        - Refit SLSQP on FULL 253 -> single weight vector
        - Apply to (513, 2) stacked te arrays -> te_nb2982

GATE:
    mean_rae < 0.4539  -> "BETTER_THAN_NB2973"
    mean_rae < 0.4570  -> "PROMOTE"
    else               -> "FAIL"

References:
    nb2960 K18 deep-30 OOF        = 0.4536  (best single, dominates deploy)
    nb2960 K20 deep-30 OOF        = 0.4625
    nb2973 4-K per-fold simplex   = 0.4539  (collapsed to K18,K20 at deploy)
    nb2960 equal_K(K18,K24,K28)   = 0.4567

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb2960_K{18,20}_30seed_oof.npy
    data/processed/nb2960_K{18,20}_30seed_te.npy

Outputs:
    data/processed/nb2982_summary.json
    data/processed/nb2982_pred_oof.npy   (253,) float32 -- per-fold simplex OOF
    data/processed/te_nb2982.npy         (513,) float32 -- deploy te
    submissions/nb2982_per_fold_simplex_K18_K20.csv  (only if verdict != "FAIL")
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import RDLogger
from scipy.optimize import minimize

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2982"
PARENT_TAG = "nb2960"

# -- Inputs --------------------------------------------------------------------
K_LABELS = ["K18", "K20"]
OOF_PATHS = {k: DATA_PROCESSED / f"nb2960_{k}_30seed_oof.npy" for k in K_LABELS}
TE_PATHS = {k: DATA_PROCESSED / f"nb2960_{k}_30seed_te.npy" for k in K_LABELS}

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEED = 1001
N_STARTS_FOLD = 8
N_STARTS_FULL = 12
DEGEN_MAX_W = 0.85

# -- Gates ---------------------------------------------------------------------
GATE_BETTER_THAN_NB2973 = 0.4539
GATE_PROMOTE = 0.4570

# -- References ----------------------------------------------------------------
REF_K18 = 0.4536
REF_K20 = 0.4625
REF_NB2973 = 0.4539
REF_EQUAL_K_18_24_28 = 0.4567
REF_NB2171 = 0.4682


def _simplex_slsqp(P: np.ndarray, y: np.ndarray, n_starts: int = 8,
                   seed: int = 0) -> tuple[np.ndarray, float]:
    """Minimize RAE(y, P @ w) over the simplex (w>=0, sum w=1) with multi-start."""
    K = P.shape[1]
    rng = np.random.default_rng(seed)

    def loss(w: np.ndarray) -> float:
        return float(rae(y, P @ w))

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bnds = [(0.0, 1.0)] * K

    starts = [np.full(K, 1.0 / K)]
    for _ in range(max(0, n_starts - 1)):
        starts.append(rng.dirichlet(np.ones(K)))

    best_w, best_r = None, np.inf
    for x0 in starts:
        try:
            res = minimize(loss, x0, method="SLSQP", bounds=bnds, constraints=cons,
                           options={"maxiter": 300, "ftol": 1e-9})
            w = np.clip(res.x, 0.0, 1.0)
            s = float(w.sum())
            if s <= 0.0:
                continue
            w = w / s
            r = float(rae(y, P @ w))
            if r < best_r:
                best_r, best_w = r, w
        except Exception:
            continue
    if best_w is None:
        best_w = np.full(K, 1.0 / K)
        best_r = float(rae(y, P @ best_w))
    return best_w, best_r


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- per-fold SLSQP simplex on 2 K-anchors {K_LABELS}")
    print(f"          parent={PARENT_TAG} (deep-30 fresh-seed K-anchors)")
    print(f"          motivation: nb2973 deploy collapsed to "
          "{K18=0.7015, K20=0.2985}")
    print(f"          outer CV: {N_FOLDS}-fold scaffold, kf_seed={KF_SEED}")
    print(f"          per fold: SLSQP simplex w (sum=1, w>=0), {N_STARTS_FOLD} starts")
    print(f"          gate: <{GATE_BETTER_THAN_NB2973} BETTER_THAN_NB2973 / "
          f"<{GATE_PROMOTE} PROMOTE")
    print("=" * 78)

    # -- Load test, truth ----------------------------------------------------
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values
                if "name" in te.columns else te["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    assert n_unb == 253, f"expected 253 unblind, got {n_unb}"
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    # -- Load deep-30 K-anchor OOFs + te arrays -------------------------------
    print("\n" + "-" * 78)
    print("STEP 1: load nb2960 deep-30 K-anchor OOFs and te arrays")
    print("-" * 78)
    oof_cols, te_cols = [], []
    per_K_full_rae = {}
    for k in K_LABELS:
        oof = np.load(OOF_PATHS[k]).astype(np.float64)
        te_arr = np.load(TE_PATHS[k]).astype(np.float64)
        if oof.shape != (n_unb,):
            raise ValueError(f"{k} OOF shape {oof.shape} != ({n_unb},)")
        if te_arr.shape != (n_test,):
            raise ValueError(f"{k} te shape {te_arr.shape} != ({n_test},)")
        oof_cols.append(oof)
        te_cols.append(te_arr)
        r = float(rae(y_unb, oof))
        per_K_full_rae[k] = round(r, 4)
        print(f"   {k}: oof_RAE = {r:.4f}  te_mean={te_arr.mean():.3f}  "
              f"te_std={te_arr.std():.3f}")

    K = len(K_LABELS)
    P_unb = np.column_stack(oof_cols)  # (253, 2)
    P_te = np.column_stack(te_cols)    # (513, 2)

    # Leak sanity
    leak_flags = {}
    for i, k in enumerate(K_LABELS):
        frac = float(np.mean(np.isclose(P_unb[:, i], y_unb, atol=1e-6)))
        leak_flags[k] = round(frac, 4)
        if frac > 0.05:
            print(f"   WARN {k}: {frac:.1%} rows == truth -- possible leak")

    # Pair-wise correlations
    corr_mat = np.corrcoef(P_unb.T)
    print(f"\n  OOF correlation matrix:")
    print(f"        {'  '.join([f'{k:>6s}' for k in K_LABELS])}")
    for i, ki in enumerate(K_LABELS):
        row = "  ".join([f"{corr_mat[i, j]:6.3f}" for j in range(K)])
        print(f"   {ki:>6s}  {row}")

    # -- Build scaffolds ------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 2: scaffolds for outer CV")
    print("-" * 78)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"   unique scaffolds = {n_unique_scaf}")

    # -- Per-fold SLSQP simplex ----------------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 3: outer CV with per-fold SLSQP simplex")
    print("-" * 78)
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=KF_SEED,
    )
    oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
    fold_records = []
    fold_w_list = []
    any_fold_degenerate = False
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        w, r_train = _simplex_slsqp(
            P_unb[tr_loc], y_unb[tr_loc],
            n_starts=N_STARTS_FOLD,
            seed=KF_SEED * 11 + fold_i,
        )
        val_pred = P_unb[va_loc] @ w
        oof_blend[va_loc] = val_pred
        r_val = float(rae(y_unb[va_loc], val_pred))
        fold_w_list.append(w)
        degen = bool(w.max() > DEGEN_MAX_W)
        any_fold_degenerate = any_fold_degenerate or degen
        fold_records.append({
            "fold": int(fold_i),
            "n_train": int(len(tr_loc)),
            "n_val": int(len(va_loc)),
            "weights": {K_LABELS[k]: round(float(w[k]), 4) for k in range(K)},
            "train_rae": round(float(r_train), 4),
            "val_rae": round(r_val, 4),
            "max_w": round(float(w.max()), 4),
            "degenerate": degen,
        })
        wstr = "  ".join(f"{K_LABELS[k]}={w[k]:.3f}" for k in range(K))
        print(f"   fold {fold_i}: tr={len(tr_loc):3d} va={len(va_loc):3d}  "
              f"[{wstr}]  train_RAE={r_train:.4f}  val_RAE={r_val:.4f}  "
              f"max_w={w.max():.3f}  degen={degen}")

    if np.isnan(oof_blend).any():
        raise RuntimeError("scaffold splits did not cover all 253 rows")
    pooled_rae = float(rae(y_unb, oof_blend))
    per_fold_val = [r["val_rae"] for r in fold_records]
    mean_fold = float(np.mean(per_fold_val))
    std_fold = float(np.std(per_fold_val, ddof=1))
    print(f"\n   pooled outer-val RAE = {pooled_rae:.4f}")
    print(f"   per-fold val RAE mean = {mean_fold:.4f}  std = {std_fold:.4f}")

    # Mean weights across folds (LB-honester alternative)
    w_stack = np.stack(fold_w_list, axis=0)  # (5, 2)
    w_mean = w_stack.mean(axis=0)
    w_mean = w_mean / w_mean.sum()
    print(f"   mean-of-fold weights: "
          + ", ".join(f"{K_LABELS[k]}={w_mean[k]:.4f}" for k in range(K)))

    # -- Deploy: single-pool SLSQP on FULL 253 -> 1 global weight vector ------
    print("\n" + "-" * 78)
    print("STEP 4: deploy SLSQP on FULL 253")
    print("-" * 78)
    w_full, r_full = _simplex_slsqp(P_unb, y_unb, n_starts=N_STARTS_FULL, seed=0)
    full_pool_weights = {K_LABELS[k]: round(float(w_full[k]), 4) for k in range(K)}
    full_pool_degen = bool(w_full.max() > DEGEN_MAX_W)
    print(f"   in-sample RAE = {r_full:.4f}  max_w={w_full.max():.4f}  "
          f"degen={full_pool_degen}")
    for k in range(K):
        flag = " (zeroed)" if w_full[k] < 1e-6 else ""
        print(f"     w[{K_LABELS[k]:6s}] = {w_full[k]:+.4f}{flag}")

    te_pred = (P_te @ w_full).astype(np.float32)
    te_pred = np.clip(te_pred, 3.0, 9.0)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))

    te_pred_mean_w = (P_te @ w_mean).astype(np.float32)
    te_pred_mean_w = np.clip(te_pred_mean_w, 3.0, 9.0)
    te_unb_in_mean_w = float(rae(y_unb, te_pred_mean_w[unb_idx]))

    print(f"   te(full-pool) mean={te_pred.mean():.3f} std={te_pred.std():.3f} "
          f"in-sample unb RAE={te_unb_in_rae:.4f}")
    print(f"   te(mean-fold)  mean={te_pred_mean_w.mean():.3f} "
          f"std={te_pred_mean_w.std():.3f} "
          f"in-sample unb RAE={te_unb_in_mean_w:.4f}")

    # -- Gate -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 5: GATE")
    print("-" * 78)
    if pooled_rae < GATE_BETTER_THAN_NB2973:
        verdict = "BETTER_THAN_NB2973"
    elif pooled_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    else:
        verdict = "FAIL"
    delta_vs_nb2973 = pooled_rae - REF_NB2973
    delta_vs_K18 = pooled_rae - REF_K18
    delta_vs_K20 = pooled_rae - REF_K20
    delta_vs_equal_K = pooled_rae - REF_EQUAL_K_18_24_28
    delta_vs_nb2171 = pooled_rae - REF_NB2171
    print(f"   pooled_rae               = {pooled_rae:.4f}")
    print(f"   delta vs K18 (0.4536)    = {delta_vs_K18:+.4f}")
    print(f"   delta vs K20 (0.4625)    = {delta_vs_K20:+.4f}")
    print(f"   delta vs nb2973 (0.4539) = {delta_vs_nb2973:+.4f}")
    print(f"   delta vs equal_K (0.4567)= {delta_vs_equal_K:+.4f}")
    print(f"   delta vs nb2171 (0.4682) = {delta_vs_nb2171:+.4f}")
    print(f"   verdict                  = {verdict}")

    # -- Save artifacts -------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 6: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_blend.astype(np.float32))
    np.save(te_path, te_pred)
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_per_fold_simplex_K18_K20.csv"
    if verdict != "FAIL":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": te_pred,
        }).to_csv(sub_csv, index=False)
        print(f"   [save] {sub_csv}")
    else:
        print(f"   [skip] verdict={verdict}; no submission CSV written")

    summary = {
        "tag": TAG,
        "parent_tag": PARENT_TAG,
        "method": "per_fold_slsqp_simplex_K18_K20_only",
        "paradigm": "2_anchor_explicit_pool_vs_nb2973_4_anchor_auto_zeroed",
        "anchor_pool": K_LABELS,
        "anchor_pre_unblind": True,
        "per_K_full_oof_rae": per_K_full_rae,
        "anchor_leak_eq_truth_frac": leak_flags,
        "oof_corr_matrix": corr_mat.tolist(),
        "oof_corr_labels": K_LABELS,
        "n_folds": N_FOLDS,
        "kf_seed": KF_SEED,
        "n_starts_fold": N_STARTS_FOLD,
        "n_starts_full": N_STARTS_FULL,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "K_anchors": K,
        "fold_records": fold_records,
        "pooled_outer_val_rae": pooled_rae,
        "per_fold_val_rae_mean": mean_fold,
        "per_fold_val_rae_std": std_fold,
        "mean_w_across_folds": {K_LABELS[k]: round(float(w_mean[k]), 4)
                                for k in range(K)},
        "any_fold_degenerate": any_fold_degenerate,
        "full_pool_slsqp": {
            "weights": full_pool_weights,
            "rae_in_sample": round(float(r_full), 4),
            "max_w": round(float(w_full.max()), 4),
            "degenerate": full_pool_degen,
        },
        "te_unb_in_sample_rae_full_pool": round(te_unb_in_rae, 4),
        "te_unb_in_sample_rae_mean_fold": round(te_unb_in_mean_w, 4),
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict != "FAIL" else None,
        "mean_rae": pooled_rae,
        "ref_K18_deep30": REF_K18,
        "ref_K20_deep30": REF_K20,
        "ref_nb2973_4K": REF_NB2973,
        "ref_equal_K_18_24_28": REF_EQUAL_K_18_24_28,
        "ref_nb2171": REF_NB2171,
        "delta_vs_K18": delta_vs_K18,
        "delta_vs_K20": delta_vs_K20,
        "delta_vs_nb2973": delta_vs_nb2973,
        "delta_vs_equal_K": delta_vs_equal_K,
        "delta_vs_nb2171": delta_vs_nb2171,
        "gate_better_than_nb2973": GATE_BETTER_THAN_NB2973,
        "gate_promote": GATE_PROMOTE,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   per-K full-OOF RAE       = "
          + ", ".join([f"{k}={v:.4f}" for k, v in per_K_full_rae.items()]))
    print(f"   pooled outer-val RAE     = {pooled_rae:.4f}")
    print(f"   mean-of-fold weights     = "
          + ", ".join([f"{K_LABELS[k]}={w_mean[k]:.3f}" for k in range(K)]))
    print(f"   full-pool weights        = {full_pool_weights}")
    print(f"   te[unb_idx] in-sample    = {te_unb_in_rae:.4f}")
    print(f"   verdict                  = {verdict}")
    print(f"   wall                     = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "pooled_outer_val_rae",
        "per_fold_val_rae_mean",
        "per_fold_val_rae_std",
        "mean_w_across_folds",
        "full_pool_slsqp",
        "te_unb_in_sample_rae_full_pool",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
    print(f"  per_K_full_oof_rae: {res.get('per_K_full_oof_rae')}")
