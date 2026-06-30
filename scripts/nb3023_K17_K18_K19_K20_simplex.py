"""nb3023 -- Per-fold SLSQP simplex on 4-anchor pool {K17, K18, K19, K20}
              (all deep-30).

NEW PARADIGM: combine every deep-30 K anchor available at and below K=20.

K=17 was the lowest single-K deep-30 RAE seen so far (nb3010 = 0.4680).
nb3002 (K18 + K19, deep-30) reached pooled outer-val RAE 0.4511.
nb3011 added K17 to that pool ({K17, K18, K19}); nb2992 had {K18, K19(5sd),
K20} and zeroed K20 in deploy.

This script tests whether the FULL deep-30 K-pyramid {K17, K18, K19, K20}
unlocks additional cross-fold orthogonality. K=17 is the off-RFE-plateau
floor and K=20 has been a deploy-zero-weight in 3-anchor pools -- the
4-anchor test gives SLSQP the option to either revive K=20 or confirm it
should stay zeroed.

PROTOCOL:
    Anchors (all deep-30):
        K17: nb3010 deep-30 fresh-seed OOF + te (bag RAE 0.4680)
        K18: nb2960 deep-30 fresh-seed OOF + te (full-OOF RAE 0.4536)
        K19: nb3000 deep-30 fresh-seed OOF + te (full-OOF RAE 0.4607)
        K20: nb2960 deep-30 fresh-seed OOF + te (full-OOF RAE 0.4625)
    Outer CV: 5-fold scaffold split, 5 fresh kf_seeds {1036..1040}
    Per fold:
        - SLSQP minimize fold-train RAE on simplex (w >= 0, sum w = 1)
        - 8 multi-starts (uniform + 7 Dirichlet draws)
        - Apply per-fold weights to held-out fold-val slice
    Per seed: pooled RAE on the 5 outer-val folds (covering all 253)
    Reported gate metric = MEAN pooled RAE across the 5 seeds.

    Deploy:
        - Refit SLSQP on FULL 253 -> single global weight vector
        - Apply to (513, 4) stacked te arrays -> te_nb3023

GATE:
    mean_pooled_rae < 0.4511 -> "BETTER_THAN_NB3001"
    else                     -> "FAIL"

References:
    nb3010 K17 deep-30 bag-mean RAE  = 0.4680
    nb2960 K18 deep-30 OOF RAE       = 0.4536
    nb3000 K19 deep-30 OOF RAE       = 0.4607
    nb2960 K20 deep-30 OOF RAE       = 0.4625
    nb3001 K18+K19(deep30)+K20 simplex                      = 0.4515
    nb3002 K18+K19 deep-30 per-fold simplex                  = 0.4511
    nb3011 K17+K18+K19 deep-30 (15 seeds)                    = (cycle 247+ verify)
    nb2992 K18+K19(5sd)+K20 deploy                           = 0.4479 (deploy)
    nb2171 prior post-hoc-blend ceiling                      = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb3010_K17_30seed_oof.npy
    data/processed/te_nb3010_K17.npy
    data/processed/nb2960_K18_30seed_oof.npy
    data/processed/nb2960_K18_30seed_te.npy
    data/processed/nb3000_K19_30seed_oof.npy
    data/processed/te_nb3000_K19.npy
    data/processed/nb2960_K20_30seed_oof.npy
    data/processed/nb2960_K20_30seed_te.npy

Outputs:
    data/processed/nb3023_summary.json
    data/processed/nb3023_pred_oof.npy  (253,) float32 -- first-seed per-fold OOF blend
    data/processed/te_nb3023.npy        (513,) float32 -- deploy te
    submissions/nb3023_per_fold_simplex_K17_K18_K19_K20_deep30.csv  (only if verdict != "FAIL")
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

TAG = "nb3023"
PARENT_TAG = "nb3010+nb2960+nb3000"

# -- Inputs --------------------------------------------------------------------
K_LABELS = ["K17", "K18", "K19", "K20"]
OOF_PATHS = {
    "K17": DATA_PROCESSED / "nb3010_K17_30seed_oof.npy",
    "K18": DATA_PROCESSED / "nb2960_K18_30seed_oof.npy",
    "K19": DATA_PROCESSED / "nb3000_K19_30seed_oof.npy",
    "K20": DATA_PROCESSED / "nb2960_K20_30seed_oof.npy",
}
TE_PATHS = {
    "K17": DATA_PROCESSED / "te_nb3010_K17.npy",
    "K18": DATA_PROCESSED / "nb2960_K18_30seed_te.npy",
    "K19": DATA_PROCESSED / "te_nb3000_K19.npy",
    "K20": DATA_PROCESSED / "nb2960_K20_30seed_te.npy",
}
K_DEPTH = {"K17": "deep30", "K18": "deep30", "K19": "deep30", "K20": "deep30"}

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1036, 1041))  # 5 fresh seeds {1036..1040}
N_STARTS_FOLD = 8
N_STARTS_FULL = 12
DEGEN_MAX_W = 0.85

# -- Gates ---------------------------------------------------------------------
GATE_BETTER_THAN_NB3001 = 0.4511

# -- References ----------------------------------------------------------------
REF_K17 = 0.4680
REF_K18 = 0.4536
REF_K19 = 0.4607
REF_K20 = 0.4625
REF_NB3001 = 0.4515
REF_NB3002 = 0.4511
REF_NB2992 = 0.4479
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


def _run_one_seed(kf_seed: int, P_unb: np.ndarray, y_unb: np.ndarray,
                  unb_scaffolds: list[str]) -> tuple[float, list[dict], np.ndarray, list[np.ndarray]]:
    """Per-fold SLSQP simplex with one kf_seed.

    Returns (pooled_rae, fold_records, oof_blend, fold_weights).
    """
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
    fold_records = []
    fold_w_list = []
    K = P_unb.shape[1]
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        w, r_train = _simplex_slsqp(
            P_unb[tr_loc], y_unb[tr_loc],
            n_starts=N_STARTS_FOLD,
            seed=kf_seed * 11 + fold_i,
        )
        val_pred = P_unb[va_loc] @ w
        oof_blend[va_loc] = val_pred
        r_val = float(rae(y_unb[va_loc], val_pred))
        fold_w_list.append(w)
        fold_records.append({
            "fold": int(fold_i),
            "n_train": int(len(tr_loc)),
            "n_val": int(len(va_loc)),
            "weights": {K_LABELS[k]: round(float(w[k]), 4) for k in range(K)},
            "train_rae": round(float(r_train), 4),
            "val_rae": round(r_val, 4),
        })
    if np.isnan(oof_blend).any():
        raise RuntimeError(f"scaffold splits did not cover all 253 rows (kf_seed={kf_seed})")
    pooled_rae = float(rae(y_unb, oof_blend))
    return pooled_rae, fold_records, oof_blend, fold_w_list


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- per-fold SLSQP simplex on 4 deep-30 K-anchors {K_LABELS}")
    print(f"          parents: nb3010 (K17) + nb2960 (K18,K20) + nb3000 (K19), all deep-30")
    print(f"          motivation: full K-pyramid {{K17..K20}} -- new paradigm vs nb3002/nb3011")
    print(f"          outer CV: {N_FOLDS}-fold scaffold, {len(KF_SEEDS)} fresh seeds "
          f"{KF_SEEDS[0]}..{KF_SEEDS[-1]}")
    print(f"          per fold: SLSQP simplex w (sum=1, w>=0), {N_STARTS_FOLD} starts")
    print(f"          gate: <{GATE_BETTER_THAN_NB3001} BETTER_THAN_NB3001 / else FAIL")
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

    # -- Load K-anchor OOFs + te arrays --------------------------------------
    print("\n" + "-" * 78)
    print("STEP 1: load K-anchor OOFs and te arrays (all deep-30)")
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
        print(f"   {k} ({K_DEPTH[k]:>6s}): oof_RAE = {r:.4f}  "
              f"te_mean={te_arr.mean():.3f}  te_std={te_arr.std():.3f}")

    K = len(K_LABELS)
    P_unb = np.column_stack(oof_cols)  # (253, 4)
    P_te = np.column_stack(te_cols)    # (513, 4)

    # Leak sanity
    leak_flags = {}
    for i, k in enumerate(K_LABELS):
        frac = float(np.mean(np.isclose(P_unb[:, i], y_unb, atol=1e-6)))
        leak_flags[k] = round(frac, 4)
        if frac > 0.05:
            print(f"   WARN {k}: {frac:.1%} rows == truth -- possible leak")

    # Pair-wise correlation
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

    # -- Per-seed per-fold SLSQP simplex (5 seeds) ---------------------------
    print("\n" + "-" * 78)
    print(f"STEP 3: outer CV with per-fold SLSQP, {len(KF_SEEDS)} fresh kf_seeds")
    print("-" * 78)
    per_seed_pooled = []
    per_seed_fold_records = {}
    per_seed_fold_weights = {}
    first_seed_oof_blend = None
    for seed in KF_SEEDS:
        p_rae, fold_recs, oof_blend, fold_ws = _run_one_seed(
            seed, P_unb, y_unb, unb_scaffolds,
        )
        per_seed_pooled.append(p_rae)
        per_seed_fold_records[str(seed)] = fold_recs
        per_seed_fold_weights[str(seed)] = [
            [round(float(w[k]), 4) for k in range(K)] for w in fold_ws
        ]
        if first_seed_oof_blend is None:
            first_seed_oof_blend = oof_blend
        mean_val = float(np.mean([r["val_rae"] for r in fold_recs]))
        print(f"   seed={seed}  pooled={p_rae:.4f}  per-fold mean={mean_val:.4f}")

    arr_pooled = np.asarray(per_seed_pooled)
    mean_pooled = float(arr_pooled.mean())
    std_pooled = float(arr_pooled.std(ddof=1))
    min_pooled = float(arr_pooled.min())
    max_pooled = float(arr_pooled.max())
    print(f"\n   POOLED-OUTER-VAL RAE over {len(KF_SEEDS)} seeds:")
    print(f"     mean = {mean_pooled:.4f}")
    print(f"     std  = {std_pooled:.4f}")
    print(f"     min  = {min_pooled:.4f}")
    print(f"     max  = {max_pooled:.4f}")

    # -- Aggregate per-seed weights across folds (mean across folds and seeds)
    all_weights = []
    for seed_key, wlist in per_seed_fold_weights.items():
        for fw in wlist:
            all_weights.append(fw)
    all_w_arr = np.asarray(all_weights, dtype=np.float64)  # (25, 4)
    mean_w_across_all = all_w_arr.mean(axis=0)
    s_sum = mean_w_across_all.sum()
    if s_sum > 0:
        mean_w_across_all = mean_w_across_all / s_sum
    print(f"\n   mean weights across {len(all_weights)} (seed,fold) cells:")
    for k in range(K):
        print(f"     w[{K_LABELS[k]:>4s}] = {mean_w_across_all[k]:+.4f}")

    # -- Deploy: single-pool SLSQP on FULL 253 -> 1 global weight vector -----
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
        print(f"     w[{K_LABELS[k]:>4s}] = {w_full[k]:+.4f}{flag}")

    te_pred = (P_te @ w_full).astype(np.float32)
    te_pred = np.clip(te_pred, 3.0, 9.0)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"   te(full-pool) mean={te_pred.mean():.3f} std={te_pred.std():.3f} "
          f"in-sample unb RAE={te_unb_in_rae:.4f}")

    # -- Gate ----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 5: GATE on mean pooled outer-val RAE across 5 seeds")
    print("-" * 78)
    if mean_pooled < GATE_BETTER_THAN_NB3001:
        verdict = "BETTER_THAN_NB3001"
    else:
        verdict = "FAIL"
    delta_vs_K17 = mean_pooled - REF_K17
    delta_vs_K18 = mean_pooled - REF_K18
    delta_vs_K19 = mean_pooled - REF_K19
    delta_vs_K20 = mean_pooled - REF_K20
    delta_vs_nb3001 = mean_pooled - REF_NB3001
    delta_vs_nb3002 = mean_pooled - REF_NB3002
    delta_vs_nb2992 = mean_pooled - REF_NB2992
    delta_vs_nb2171 = mean_pooled - REF_NB2171
    print(f"   mean_pooled_rae          = {mean_pooled:.4f} (std {std_pooled:.4f})")
    print(f"   delta vs K17  (0.4680)   = {delta_vs_K17:+.4f}")
    print(f"   delta vs K18  (0.4536)   = {delta_vs_K18:+.4f}")
    print(f"   delta vs K19  (0.4607)   = {delta_vs_K19:+.4f}")
    print(f"   delta vs K20  (0.4625)   = {delta_vs_K20:+.4f}")
    print(f"   delta vs nb3001 (0.4515) = {delta_vs_nb3001:+.4f}")
    print(f"   delta vs nb3002 (0.4511) = {delta_vs_nb3002:+.4f}")
    print(f"   delta vs nb2992 (0.4479) = {delta_vs_nb2992:+.4f}")
    print(f"   delta vs nb2171 (0.4682) = {delta_vs_nb2171:+.4f}")
    print(f"   verdict                  = {verdict}")

    # -- Save artifacts ------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 6: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    # pred_oof = first-seed (1036) per-fold OOF blend (single-seed slice).
    np.save(oof_path, first_seed_oof_blend.astype(np.float32))
    np.save(te_path, te_pred)
    print(f"   [save] {oof_path}  (single-seed OOF, kf_seed={KF_SEEDS[0]})")
    print(f"   [save] {te_path}   (deploy from FULL-253 SLSQP weights)")

    sub_csv = SUBMISSIONS / f"{TAG}_per_fold_simplex_K17_K18_K19_K20_deep30.csv"
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
        "method": "per_fold_slsqp_simplex_K17_K18_K19_K20_deep30_5seed",
        "paradigm": "4_anchor_full_K_pyramid_below_and_at_K20_deep30",
        "anchor_pool": K_LABELS,
        "anchor_depth": K_DEPTH,
        "anchor_pre_unblind": True,
        "per_K_full_oof_rae": per_K_full_rae,
        "anchor_leak_eq_truth_frac": leak_flags,
        "oof_corr_matrix": corr_mat.tolist(),
        "oof_corr_labels": K_LABELS,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": len(KF_SEEDS),
        "n_starts_fold": N_STARTS_FOLD,
        "n_starts_full": N_STARTS_FULL,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "K_anchors": K,
        "per_seed_pooled_rae": [round(r, 5) for r in per_seed_pooled],
        "per_seed_fold_records": per_seed_fold_records,
        "per_seed_fold_weights": per_seed_fold_weights,
        "pooled_rae_mean": round(mean_pooled, 5),
        "pooled_rae_std": round(std_pooled, 5),
        "pooled_rae_min": round(min_pooled, 5),
        "pooled_rae_max": round(max_pooled, 5),
        "mean_w_across_all_cells": {K_LABELS[k]: round(float(mean_w_across_all[k]), 4)
                                    for k in range(K)},
        "full_pool_slsqp": {
            "weights": full_pool_weights,
            "rae_in_sample": round(float(r_full), 4),
            "max_w": round(float(w_full.max()), 4),
            "degenerate": full_pool_degen,
        },
        "te_unb_in_sample_rae_full_pool": round(te_unb_in_rae, 4),
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict != "FAIL" else None,
        "mean_rae": mean_pooled,
        "ref_K17_deep30": REF_K17,
        "ref_K18_deep30": REF_K18,
        "ref_K19_deep30": REF_K19,
        "ref_K20_deep30": REF_K20,
        "ref_nb3001": REF_NB3001,
        "ref_nb3002": REF_NB3002,
        "ref_nb2992": REF_NB2992,
        "ref_nb2171": REF_NB2171,
        "delta_vs_K17": delta_vs_K17,
        "delta_vs_K18": delta_vs_K18,
        "delta_vs_K19": delta_vs_K19,
        "delta_vs_K20": delta_vs_K20,
        "delta_vs_nb3001": delta_vs_nb3001,
        "delta_vs_nb3002": delta_vs_nb3002,
        "delta_vs_nb2992": delta_vs_nb2992,
        "delta_vs_nb2171": delta_vs_nb2171,
        "gate_better_than_nb3001": GATE_BETTER_THAN_NB3001,
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
    print(f"   pooled outer-val RAE     = {mean_pooled:.4f} +/- {std_pooled:.4f} "
          f"({len(KF_SEEDS)} seeds)")
    print(f"   min/max pooled RAE       = {min_pooled:.4f} / {max_pooled:.4f}")
    print(f"   mean-cell weights        = "
          + ", ".join([f"{K_LABELS[k]}={mean_w_across_all[k]:.3f}" for k in range(K)]))
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
        "pooled_rae_mean",
        "pooled_rae_std",
        "pooled_rae_min",
        "pooled_rae_max",
        "mean_w_across_all_cells",
        "full_pool_slsqp",
        "te_unb_in_sample_rae_full_pool",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
    print(f"  per_K_full_oof_rae: {res.get('per_K_full_oof_rae')}")
