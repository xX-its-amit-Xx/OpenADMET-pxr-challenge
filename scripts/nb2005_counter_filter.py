"""nb2005 -- COUNTER-ASSAY PROXY UNCERTAINTY FILTER on nb1191.

Following nb702 promiscuity-discount logic: rows with high predicted
counter-assay activity are likely promiscuous binders; subtract a fraction
of the counter signal from the prediction. Unlike nb702 (applies a global
discount to all rows), nb2005 restricts the shrinkage to the TOP-DECILE
(10%) most counter-active rows, leaving the other 90% untouched. This is
a targeted, low-capacity post-hoc transform that risks little degradation.

Pipeline
--------
1. Load nb1191 deploy preds on 513 (te_nb1191.npy) and the cached
   counter-assay predictions on 513 (nb702_null_hat_te.npy).
2. Top-decile gate (10%): mask = counter_pred >= quantile(counter_pred, 0.90)
3. For gated rows only:
       shrunk = nb1191_pred - lambda * (counter_pred - corpus_mean)
   where corpus_mean is the mean of counter_pred over all 513 rows.
4. Lambda sweep {0.05, 0.10, 0.20}.
5. Honest scaffold 5-fold CV on the 253 unblind subset (kf_seed=1001):
   each fold picks lambda on training-fold RAE, applies to held-out fold.
6. Gate: pooled cross-fit RAE <= 0.4680  (nb1191 0.4703 - 0.003).
7. If beats: write submissions/nb2005_counter_filter.csv.

Save:
    data/processed/nb2005_summary.json
    data/processed/te_nb2005.npy
    submissions/nb2005_counter_filter.csv  (only on gate pass)
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

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2005"
LAMBDA_GRID = [0.05, 0.10, 0.20]
TOP_DECILE = 0.90          # rows >= this quantile of counter_pred are gated
N_FOLDS = 5
KF_SEED = 1001
NB1191_OOF = 0.4703        # mean-of-seeds cross-fit RAE on 253
GATE_RAE = NB1191_OOF - 0.003     # 0.4673 (memo: <= 0.4680)
GATE_TARGET = 0.4680


def apply_filter(nb1191: np.ndarray, counter: np.ndarray, lam: float,
                 gate_mask: np.ndarray, corpus_mean: float) -> np.ndarray:
    out = nb1191.copy()
    out[gate_mask] = (
        nb1191[gate_mask] - lam * (counter[gate_mask] - corpus_mean)
    )
    return out


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- COUNTER-ASSAY PROXY UNCERTAINTY FILTER on nb1191")
    print("=" * 78)

    # ------------------------------------------------------------------
    # Step 1: load nb1191 deploy and counter-assay predictions on 513
    # ------------------------------------------------------------------
    te_nb1191_p = DATA_PROCESSED / "te_nb1191.npy"
    counter_p = DATA_PROCESSED / "nb702_null_hat_te.npy"
    unb_idx_p = DATA_PROCESSED / "_audit_unblind_idx.npy"
    unb_y_p = DATA_PROCESSED / "_audit_unblind_y.npy"

    for p in (te_nb1191_p, counter_p, unb_idx_p, unb_y_p):
        assert p.exists(), f"missing {p}"

    nb1191 = np.load(te_nb1191_p).astype(np.float64)
    counter = np.load(counter_p).astype(np.float64)
    unb_idx = np.load(unb_idx_p).astype(int)
    y_unb = np.load(unb_y_p).astype(np.float64)

    te = load_test()
    te_names = te["name"].values
    te_smiles = te["smiles"].values
    n_te = len(te_names)
    n_unb = len(y_unb)
    assert nb1191.shape == (n_te,) and counter.shape == (n_te,), \
        f"shape mismatch: nb1191 {nb1191.shape} counter {counter.shape}"

    print(f"[load] n_te={n_te}  n_unb={n_unb}")
    print(f"[load] nb1191  mean={nb1191.mean():.3f}  std={nb1191.std():.3f}")
    print(f"[load] counter mean={counter.mean():.3f}  std={counter.std():.3f}")

    # ------------------------------------------------------------------
    # Step 2: top-decile gate (10% most counter-active)
    # ------------------------------------------------------------------
    q90 = float(np.quantile(counter, TOP_DECILE))
    gate_mask_te = counter >= q90
    n_gate_te = int(gate_mask_te.sum())
    print(f"\n[gate] counter q{TOP_DECILE*100:.0f} = {q90:.3f}")
    print(f"[gate] gated rows (513) = {n_gate_te}  "
          f"({100.0*n_gate_te/n_te:.1f}%)")

    corpus_mean = float(counter.mean())
    print(f"[gate] corpus_mean (counter) = {corpus_mean:.3f}")

    # Same gate applied to the 253 unblind subset
    counter_unb = counter[unb_idx]
    nb1191_unb = nb1191[unb_idx]
    gate_mask_unb = gate_mask_te[unb_idx]
    n_gate_unb = int(gate_mask_unb.sum())
    print(f"[gate] gated rows in 253 unblind = {n_gate_unb}")

    base_unb_rae = float(rae(y_unb, nb1191_unb))
    print(f"[ref ] nb1191 unblind in-sample RAE = {base_unb_rae:.4f}")

    # ------------------------------------------------------------------
    # Step 3: pooled lambda sweep (diagnostic)
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("POOLED LAMBDA SWEEP  (apply to gated rows only)")
    print("-" * 78)
    pooled_results = {}
    for lam in LAMBDA_GRID:
        pred = apply_filter(
            nb1191_unb, counter_unb, lam, gate_mask_unb, corpus_mean
        )
        r = float(rae(y_unb, pred))
        pooled_results[lam] = r
        delta_str = f"{r - base_unb_rae:+.4f}"
        print(f"  lambda={lam:.2f}  unblind RAE = {r:.4f}  "
              f"(delta vs nb1191 {delta_str})")

    best_pool_lam = min(pooled_results, key=pooled_results.get)
    best_pool_rae = pooled_results[best_pool_lam]
    print(f"\n  best pooled lambda = {best_pool_lam:.2f}  "
          f"pooled RAE = {best_pool_rae:.4f}")

    # ------------------------------------------------------------------
    # Step 4: honest scaffold 5-fold CV on 253 unblind
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print(f"SCAFFOLD {N_FOLDS}-FOLD CROSS-FIT  kf_seed={KF_SEED}")
    print("-" * 78)
    unb_smiles = te_smiles[unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=KF_SEED,
    )

    cross_pred = np.full(n_unb, np.nan, dtype=np.float64)
    chosen_lambdas = []
    for fold, (tr_loc, va_loc) in enumerate(splits):
        # Pick lambda by minimizing training-fold RAE
        best_lam, best_r = None, np.inf
        for lam in LAMBDA_GRID:
            pred_tr = apply_filter(
                nb1191_unb[tr_loc], counter_unb[tr_loc], lam,
                gate_mask_unb[tr_loc], corpus_mean,
            )
            r_tr = float(rae(y_unb[tr_loc], pred_tr))
            if r_tr < best_r:
                best_r = r_tr
                best_lam = lam
        chosen_lambdas.append(best_lam)
        cross_pred[va_loc] = apply_filter(
            nb1191_unb[va_loc], counter_unb[va_loc], best_lam,
            gate_mask_unb[va_loc], corpus_mean,
        )
        print(f"  fold {fold}: chose lambda={best_lam:.2f}  "
              f"tr_RAE={best_r:.4f}  n_va={len(va_loc)}")

    cross_rae = float(rae(y_unb, cross_pred))
    delta_vs_nb1191 = cross_rae - NB1191_OOF
    print(f"\n[cv] HONEST cross-fit RAE = {cross_rae:.4f}  "
          f"(vs nb1191 {NB1191_OOF:.4f}, delta {delta_vs_nb1191:+.4f})")

    # ------------------------------------------------------------------
    # Step 5: gate evaluation
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE EVALUATION")
    print("-" * 78)
    gate_pass = bool(cross_rae <= GATE_TARGET)
    print(f"   cross-fit RAE {cross_rae:.4f} <= {GATE_TARGET:.4f}  "
          f"-> {'PASS' if gate_pass else 'FAIL'}")

    # ------------------------------------------------------------------
    # Deploy: apply best pooled lambda to the 513
    # ------------------------------------------------------------------
    deploy_lam = float(best_pool_lam)
    te_deploy = apply_filter(
        nb1191, counter, deploy_lam, gate_mask_te, corpus_mean
    ).astype(np.float32)
    print(f"\n[deploy] lambda={deploy_lam:.2f}  "
          f"mean={te_deploy.mean():.3f}  std={te_deploy.std():.3f}")
    print(f"[deploy] delta vs nb1191 mean = "
          f"{(te_deploy - nb1191).mean():+.4f}  "
          f"max suppression = {(nb1191 - te_deploy).max():.4f}")

    # Save the te artefact regardless
    te_npy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_npy_path, te_deploy)
    print(f"\n[save] {te_npy_path}")

    sub_csv_path = SUBMISSIONS / f"{TAG}_counter_filter.csv"
    if gate_pass:
        pd.DataFrame({
            "SMILES":        te_smiles,
            "Molecule Name": te_names,
            "pEC50":         te_deploy,
        }).to_csv(sub_csv_path, index=False)
        print(f"[save] {sub_csv_path}  (gate PASSED)")
    else:
        print(f"[skip] gate FAILED -- no submission CSV written")

    summary = {
        "tag": TAG,
        "method": "top_decile_counter_assay_shrink_on_nb1191",
        "anchor": "nb1191",
        "anchor_te_path": str(te_nb1191_p),
        "counter_te_path": str(counter_p),
        "n_te": n_te,
        "n_unb": n_unb,
        "lambda_grid": LAMBDA_GRID,
        "top_decile_quantile": TOP_DECILE,
        "counter_q90": q90,
        "corpus_mean_counter": corpus_mean,
        "n_gated_513": n_gate_te,
        "n_gated_253": n_gate_unb,
        "kf_seed": KF_SEED,
        "n_folds": N_FOLDS,
        "nb1191_unblind_insample_rae": base_unb_rae,
        "nb1191_oof_reference": NB1191_OOF,
        "pooled_lambda_sweep": {f"{k:.2f}": v for k, v in pooled_results.items()},
        "best_pooled_lambda": float(best_pool_lam),
        "best_pooled_rae": float(best_pool_rae),
        "cross_fit_chosen_lambdas": [float(x) for x in chosen_lambdas],
        "honest_cross_fit_rae": cross_rae,
        "delta_vs_nb1191": delta_vs_nb1191,
        "gate_target_rae": GATE_TARGET,
        "gate_pass": gate_pass,
        "deploy_lambda": deploy_lam,
        "deploy_te_mean": float(te_deploy.mean()),
        "deploy_te_std": float(te_deploy.std()),
        "deploy_delta_mean_vs_nb1191": float((te_deploy - nb1191).mean()),
        "deploy_max_suppression": float((nb1191 - te_deploy).max()),
        "te_npy_path": str(te_npy_path),
        "submission_csv": str(sub_csv_path) if gate_pass else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    json_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {json_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   pooled best lambda     = {best_pool_lam:.2f}  "
          f"(pooled RAE {best_pool_rae:.4f})")
    print(f"   honest cross-fit RAE   = {cross_rae:.4f}")
    print(f"   nb1191 reference OOF   = {NB1191_OOF:.4f}")
    print(f"   gate target            = {GATE_TARGET:.4f}")
    print(f"   gate pass              = {gate_pass}")
    print(f"   wall                   = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "best_pooled_lambda",
        "best_pooled_rae",
        "honest_cross_fit_rae",
        "delta_vs_nb1191",
        "gate_pass",
        "n_gated_253",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
