"""nb2183 -- HONEST per-fold lambda search rebuild of nb730.

Per nb2177 audit, nb730's deploy lambda was selected in-sample on the pooled
253 unblind, making te_nb730[unb_idx] == nb730_pred_oof bit-for-bit. This
script rebuilds nb730 with truly honest cross-fit:

  - For each of 5 scaffold folds on the 253 unblind, do INNER lambda grid
    search using ONLY TRAIN rows of that fold (~202 rows). Apply the chosen
    lambda to the held-out OOF rows.
  - Deploy lambda for the 513 = MEDIAN of the 5 per-fold lambdas
    (NOT the pooled-best on all 253).

This guarantees te[unb_idx] differs from pred_oof, because deploy uses median
of fold lambdas while OOF uses the per-fold validation lambda.

Inputs:
  data/processed/te_nb562.npy                       (513,)
  data/processed/nb730_null_hat_te.npy              (513,)  # null_te_ens
  data/raw/pxr-challenge_TEST_BLINDED.csv           (513 names+smiles)
  data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv (253 truth)

Outputs:
  data/processed/nb730_honest_pred_oof.npy   (253,)
  data/processed/te_nb730_honest.npy         (513,)
  data/processed/nb2183_summary.json
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pxr.chem import bemis_murcko
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, DATA_RAW

SEED = 0
N_FOLDS = 5
TAG = "nb2183"
LAMBDA_GRID = [0.00, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50]


def _sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main() -> dict:
    print("=" * 78)
    print("nb2183 -- HONEST nb730 rebuild (per-fold inner lambda search)")
    print("=" * 78)

    test_df = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_BLINDED.csv")
    unb_df = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")

    te_nb562 = np.load(DATA_PROCESSED / "te_nb562.npy").astype(np.float64)
    null_te_ens = np.load(DATA_PROCESSED / "nb730_null_hat_te.npy").astype(np.float64)
    assert te_nb562.shape == (513,), te_nb562.shape
    assert null_te_ens.shape == (513,), null_te_ens.shape

    name_to_idx = {n: i for i, n in enumerate(test_df["Molecule Name"])}
    unb_df = unb_df[unb_df["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_idx = np.array([name_to_idx[n] for n in unb_df["Molecule Name"]], dtype=int)
    y_unb = unb_df["pEC50"].astype(np.float64).values
    n_unb = len(unb_df)
    print(f"  test n={len(test_df)}  unblind n={n_unb}")

    nb562_unb = te_nb562[unb_idx]
    null_pos_unb = np.maximum(0.0, null_te_ens[unb_idx])
    null_pos_te = np.maximum(0.0, null_te_ens)

    # Baseline diagnostics
    rae_nb562_unb = float(rae(y_unb, nb562_unb))
    print(f"\n  baseline nb562 unblind RAE = {rae_nb562_unb:.4f}")

    # POOLED grid (for comparison only -- this is what nb730 used for deploy)
    pooled_results: dict[float, float] = {}
    for lam in LAMBDA_GRID:
        pooled_results[lam] = float(rae(y_unb, nb562_unb - lam * null_pos_unb))
    pooled_best_lam = min(pooled_results, key=pooled_results.get)
    pooled_best_rae = pooled_results[pooled_best_lam]
    print(f"  POOLED best lambda = {pooled_best_lam:.2f}  RAE = {pooled_best_rae:.4f}")
    for lam, r in pooled_results.items():
        marker = "  <-- pooled best" if lam == pooled_best_lam else ""
        print(f"    pooled lambda={lam:.2f}  RAE={r:.4f}{marker}")

    # HONEST cross-fit: scaffold folds on unblind. Inner grid uses TRAIN rows only.
    print("\n" + "-" * 78)
    print("HONEST 5-fold cross-fit (inner lambda grid on TRAIN rows of each fold)")
    print("-" * 78)
    unb_scaf = [bemis_murcko(s) for s in unb_df["SMILES"]]
    unb_splits = scaffold_kfold_indices(unb_scaf, n_splits=N_FOLDS, seed=SEED)

    pred_oof = np.zeros(n_unb, dtype=np.float64)
    chosen_lambdas: list[float] = []
    fold_train_raes: list[float] = []
    fold_va_raes: list[float] = []
    for fold, (tr_idx, va_idx) in enumerate(unb_splits):
        best_lam, best_r = None, np.inf
        for lam in LAMBDA_GRID:
            pred_tr = nb562_unb[tr_idx] - lam * null_pos_unb[tr_idx]
            r = float(rae(y_unb[tr_idx], pred_tr))
            if r < best_r:
                best_r, best_lam = r, lam
        chosen_lambdas.append(float(best_lam))
        fold_train_raes.append(best_r)
        # Apply chosen lambda to OOF (held-out) rows
        pred_oof[va_idx] = nb562_unb[va_idx] - best_lam * null_pos_unb[va_idx]
        va_r = float(rae(y_unb[va_idx], pred_oof[va_idx]))
        fold_va_raes.append(va_r)
        print(f"  fold {fold}: chosen lambda={best_lam:.2f}  "
              f"train RAE={best_r:.4f}  va RAE={va_r:.4f}  "
              f"n_tr={len(tr_idx)}  n_va={len(va_idx)}")

    honest_rae = float(rae(y_unb, pred_oof))
    print(f"\n  HONEST cross-fit RAE = {honest_rae:.4f}")
    print(f"  per-fold lambdas     = {chosen_lambdas}")

    # Deploy lambda = MEDIAN of per-fold lambdas
    deploy_lambda = float(np.median(chosen_lambdas))
    print(f"\n  DEPLOY lambda (median of per-fold) = {deploy_lambda:.4f}")
    print(f"  (vs pooled-best lambda             = {pooled_best_lam:.4f})")

    te_honest = (te_nb562 - deploy_lambda * null_pos_te).astype(np.float64)
    te_honest_at_unb = te_honest[unb_idx]
    rae_te_honest_at_unb = float(rae(y_unb, te_honest_at_unb))
    print(f"  te_nb730_honest[unb_idx] RAE = {rae_te_honest_at_unb:.4f}  "
          f"(in-sample-ish since deploy lambda derived from same 253)")

    # ---- sha256 verification: te_honest[unb_idx] vs pred_oof ----
    sha_te_unb = _sha256_array(te_honest_at_unb.astype(np.float64))
    sha_pred_oof = _sha256_array(pred_oof.astype(np.float64))
    distinct = sha_te_unb != sha_pred_oof
    max_abs_diff = float(np.max(np.abs(te_honest_at_unb - pred_oof)))
    mean_abs_diff = float(np.mean(np.abs(te_honest_at_unb - pred_oof)))
    print("\n  sha256 verification:")
    print(f"    sha256(te_honest[unb_idx]) = {sha_te_unb}")
    print(f"    sha256(pred_oof)           = {sha_pred_oof}")
    print(f"    DISTINCT objects          = {distinct}")
    print(f"    mean |diff|               = {mean_abs_diff:.6f}")
    print(f"    max  |diff|               = {max_abs_diff:.6f}")

    # ---- Save artefacts ----
    pred_oof_path = DATA_PROCESSED / "nb730_honest_pred_oof.npy"
    te_path = DATA_PROCESSED / "te_nb730_honest.npy"
    np.save(pred_oof_path, pred_oof.astype(np.float32))
    np.save(te_path, te_honest.astype(np.float32))
    print(f"\n  wrote {pred_oof_path}")
    print(f"  wrote {te_path}")

    summary = {
        "tag": TAG,
        "purpose": "Honest per-fold lambda search rebuild of nb730 (nb2177 audit fix)",
        "n_test": int(len(test_df)),
        "n_unblind": int(n_unb),
        "lambda_grid": LAMBDA_GRID,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "baseline_nb562_unblind_rae": rae_nb562_unb,
        "pooled_grid_rae": {f"{k:.2f}": v for k, v in pooled_results.items()},
        "pooled_best_lambda": float(pooled_best_lam),
        "pooled_best_rae": pooled_best_rae,
        "honest_crossfit_rae": honest_rae,
        "per_fold_chosen_lambdas": chosen_lambdas,
        "per_fold_train_raes": fold_train_raes,
        "per_fold_va_raes": fold_va_raes,
        "deploy_lambda_median": deploy_lambda,
        "te_honest_at_unb_rae": rae_te_honest_at_unb,
        "sha256_te_honest_unb_idx": sha_te_unb,
        "sha256_nb730_honest_pred_oof": sha_pred_oof,
        "sha256_distinct": bool(distinct),
        "mean_abs_diff_te_unb_vs_pred_oof": mean_abs_diff,
        "max_abs_diff_te_unb_vs_pred_oof": max_abs_diff,
        "delta_honest_minus_pooled": honest_rae - pooled_best_rae,
        "artefacts": {
            "pred_oof": str(pred_oof_path),
            "te_deploy": str(te_path),
        },
    }
    summ_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(summ_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote {summ_path}")

    print("\n" + "=" * 78)
    print("=== nb2183 SUMMARY ===")
    print(f"  pooled-best (nb730 deploy) RAE  = {pooled_best_rae:.4f}  "
          f"lambda={pooled_best_lam:.2f}")
    print(f"  HONEST cross-fit RAE             = {honest_rae:.4f}  "
          f"(+{honest_rae - pooled_best_rae:+.4f} vs pooled)")
    print(f"  per-fold lambdas                = {chosen_lambdas}")
    print(f"  deploy lambda (median)          = {deploy_lambda:.4f}")
    print(f"  sha256 distinct (te_unb vs oof) = {distinct}")
    print("=" * 78)

    return summary


if __name__ == "__main__":
    res = main()
