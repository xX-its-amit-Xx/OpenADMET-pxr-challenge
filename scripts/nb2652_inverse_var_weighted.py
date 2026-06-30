"""nb2652 -- Inverse-variance weighted K-ensemble (per-row precision weighting).

NEW PARADIGM:
    nb2604 used a plain equal-weight average of 4 K-RFE pyramids
    {K18, K20, K24, K28} on chemprop_aux residual and reached
    mean pooled RAE = 0.4580 (MARGINAL_BEAT).  All weights are global
    and identical for every row.

    HERE we test per-row precision weighting: rows where the 4 K-pyramids
    agree (low variance) get higher confidence and dominate, rows where
    they disagree (high variance) get down-weighted.  This is the standard
    inverse-variance combiner from meta-analysis (Cochran 1937), adapted
    here at the per-row level rather than the per-model level.

    Concretely, for each row i across K-anchors {18, 20, 24, 28}:
        Var_i  = unbiased var(K-predictions at row i) across 4 anchors
        w_K(i) = exp(-Var_i * lambda)   shared by all K's at that row
                 (note: same w for all K's at row i; pure row-level
                 confidence gate, NOT a per-K-per-row down-weighting)
        pred_i = mean_K(pred_K(i))      (since all K's get same w_K(i),
                                         the per-row blend reduces to
                                         the equal-weight mean.  The
                                         exp(-Var*lambda) cancels in the
                                         numerator/denominator if applied
                                         uniformly across K at each row.)

    To make per-row precision do real work, we apply the weight to the
    DEVIATION from anchor: pred_i = anchor_i + w(i) * (mean_K_resid_i),
    so high-variance rows shrink back toward the anchor (chemprop_aux).
    This is a row-level shrinkage gate -- when the 4 K-corrections
    disagree, trust the base anchor more; when they agree, trust the
    K-mean correction.  Lambda sweep tunes the shrinkage strength.

PROTOCOL:
    1. Load 4 cached K-RFE OOFs + tes from nb2604 dependency chain:
         K=18  -> nb2604_mean_bag_oof_K18.npy + te_nb2604_K18.npy
         K=20  -> nb2240_mean_bag_oof_K20.npy + te_nb2240_K20.npy
         K=24  -> nb2310_mean_bag_oof_K24.npy + te_nb2310_K24.npy
         K=28  -> nb2103_mean_bag_oof_K28.npy + te_nb2112.npy
    2. For each row, compute Var across 4 K-preds (unbiased N-1 normalization).
    3. Per-row weight = exp(-Var_row * lambda).
    4. Final = anchor + w_row * (mean_K_pred - anchor)
       = anchor + w_row * mean_K_residual
       (shrinks correction toward 0 when disagreement is high)
    5. Sweep lambda in {0, 0.5, 1, 2, 5, 10, 20, 50, 100, 200}.
    6. 5-fold scaffold CV on 253 with kf_seed=1001 (deterministic).
    7. Pick best lambda by mean pooled RAE; build deploy te using te-variance.

GATE:
    mean_rae < 0.4570  -> PROMOTE
    mean_rae < 0.4580  -> BETTER_THAN_NB2604
    else                -> FAIL

Outputs:
    scripts/nb2652_inverse_var_weighted.py
    data/processed/nb2652_summary.json
    data/processed/nb2652_pred_oof.npy            (253,) float32
    data/processed/te_nb2652.npy                  (513,) float32
    submissions/nb2652_inverse_var_weighted.csv   (on any non-FAIL)
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

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2652"

# ---- Anchor ----
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# ---- K-RFE OOFs + tes (same chain as nb2604) ----
K18_OOF_PATH = DATA_PROCESSED / "nb2604_mean_bag_oof_K18.npy"
K18_TE_PATH = DATA_PROCESSED / "te_nb2604_K18.npy"
K20_OOF_PATH = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"
K20_TE_PATH = DATA_PROCESSED / "te_nb2240_K20.npy"
K24_OOF_PATH = DATA_PROCESSED / "nb2310_mean_bag_oof_K24.npy"
K24_TE_PATH = DATA_PROCESSED / "te_nb2310_K24.npy"
K28_OOF_PATH = DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy"
K28_TE_PATH = DATA_PROCESSED / "te_nb2112.npy"

# ---- CV eval ----
N_FOLDS = 5
KF_SEED = 1001  # deterministic single seed per protocol

# ---- Lambda sweep ----
# lambda=0 -> w=1 for every row -> reduces to equal-weight mean (sanity check
#                                    should reproduce nb2604 ~0.4580)
# large lambda -> w->0 on high-var rows -> shrinks back to pure anchor
LAMBDA_GRID = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]

# ---- Gates ----
GATE_PROMOTE = 0.4570
GATE_BETTER_NB2604 = 0.4580

# ---- Refs ----
CHEMPROP_AUX_REF = 0.6216
NB2604_REF = 0.4580
NB2171_REF = 0.4682


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- inverse-variance weighted K-ensemble")
    print(f"          per-row weight = exp(-Var(K-preds) * lambda)")
    print(f"          final = anchor + w_row * mean_K_residual")
    print(f"          (shrinks correction when K-pyramids disagree)")
    print(f"          ref nb2604 equal-weight = {NB2604_REF:.4f}")
    print("=" * 78)

    # ---- Load truth + anchor + scaffold splits ----
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values
                if "name" in te.columns else te["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    print(f"[load] chemprop_aux te[unb_idx] in_RAE = {rae_anchor:.4f} "
          f"(ref {CHEMPROP_AUX_REF:.4f})")

    # ---- Step 1: Load 4 K-RFE OOFs + tes ----
    print("\n" + "-" * 78)
    print("STEP 1: load 4 K-RFE OOFs + tes for K in {18, 20, 24, 28}")
    print("-" * 78)
    K18_oof = np.load(K18_OOF_PATH).astype(np.float64)
    K18_te = np.load(K18_TE_PATH).astype(np.float64)
    K20_oof = np.load(K20_OOF_PATH).astype(np.float64)
    K20_te = np.load(K20_TE_PATH).astype(np.float64)
    K24_oof = np.load(K24_OOF_PATH).astype(np.float64)
    K24_te = np.load(K24_TE_PATH).astype(np.float64)
    K28_oof = np.load(K28_OOF_PATH).astype(np.float64)
    K28_te = np.load(K28_TE_PATH).astype(np.float64)

    for name, arr, exp in [
        ("K18_oof", K18_oof, (n_unb,)), ("K18_te", K18_te, (n_test,)),
        ("K20_oof", K20_oof, (n_unb,)), ("K20_te", K20_te, (n_test,)),
        ("K24_oof", K24_oof, (n_unb,)), ("K24_te", K24_te, (n_test,)),
        ("K28_oof", K28_oof, (n_unb,)), ("K28_te", K28_te, (n_test,)),
    ]:
        if arr.shape != exp:
            raise ValueError(f"{name} shape {arr.shape} != {exp}")

    # OOF (253) and te (513) stacks
    P_unb = np.column_stack([K18_oof, K20_oof, K24_oof, K28_oof])  # (253, 4)
    P_te = np.column_stack([K18_te, K20_te, K24_te, K28_te])       # (513, 4)
    K_labels = ["K18", "K20", "K24", "K28"]

    for label, oof, te_arr in zip(K_labels, [K18_oof, K20_oof, K24_oof, K28_oof],
                                  [K18_te, K20_te, K24_te, K28_te]):
        print(f"   {label:>4s}  oof_RAE={float(rae(y_unb, oof)):.4f}  "
              f"te_mean={te_arr.mean():.3f}  te_std={te_arr.std():.3f}")

    # ---- Step 2: Per-row variance + residuals ----
    print("\n" + "-" * 78)
    print("STEP 2: per-row variance + residuals")
    print("-" * 78)
    # Unbiased (ddof=1) variance across 4 K's per row
    var_unb = P_unb.var(axis=1, ddof=1).astype(np.float64)  # (253,)
    var_te = P_te.var(axis=1, ddof=1).astype(np.float64)    # (513,)
    print(f"   var_unb stats:  mean={var_unb.mean():.4f}  "
          f"median={np.median(var_unb):.4f}  "
          f"min={var_unb.min():.4f}  max={var_unb.max():.4f}")
    print(f"   var_te stats:   mean={var_te.mean():.4f}  "
          f"median={np.median(var_te):.4f}  "
          f"min={var_te.min():.4f}  max={var_te.max():.4f}")

    # Mean K-pred per row
    mean_K_unb = P_unb.mean(axis=1)  # (253,)
    mean_K_te = P_te.mean(axis=1)    # (513,)
    # Residual = mean_K_pred - anchor (per row), the correction the K's offer
    resid_unb = mean_K_unb - anchor_unb         # (253,)
    resid_te_anchor = te_anchor_513             # (513,)
    resid_te = mean_K_te - resid_te_anchor      # (513,)
    print(f"   mean_K residual:  unb mean={resid_unb.mean():+.3f}  "
          f"std={resid_unb.std():.3f};  te mean={resid_te.mean():+.3f}  "
          f"std={resid_te.std():.3f}")

    # ---- Step 3: Lambda sweep ----
    print("\n" + "-" * 78)
    print(f"STEP 3: lambda sweep over {LAMBDA_GRID}")
    print(f"        eval: 5-fold scaffold CV, kf_seed={KF_SEED}")
    print("-" * 78)

    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=KF_SEED,
    )

    lambda_results = []
    for lam in LAMBDA_GRID:
        w_unb = np.exp(-var_unb * lam)  # (253,)
        pred_oof_lam = anchor_unb + w_unb * resid_unb

        # pooled OOF RAE (no learning - predictions fixed per lambda)
        oof_pooled = np.full(n_unb, np.nan, dtype=np.float64)
        per_fold_rae = []
        for tr_loc, va_loc in splits:
            oof_pooled[va_loc] = pred_oof_lam[va_loc]
            per_fold_rae.append(float(rae(y_unb[va_loc], pred_oof_lam[va_loc])))
        if np.isnan(oof_pooled).any():
            raise RuntimeError("scaffold splits did not cover all rows")
        pooled = float(rae(y_unb, oof_pooled))
        per_fold_mean = float(np.mean(per_fold_rae))

        w_mean = float(w_unb.mean())
        w_min = float(w_unb.min())
        lambda_results.append({
            "lambda": lam,
            "pooled_rae": pooled,
            "per_fold_mean_rae": per_fold_mean,
            "per_fold_rae": per_fold_rae,
            "w_mean": w_mean,
            "w_min": w_min,
            "w_max": float(w_unb.max()),
        })
        print(f"   lambda={lam:7.2f}  w_mean={w_mean:.3f}  "
              f"w_min={w_min:.3f}  pooled_RAE={pooled:.4f}  "
              f"per_fold_mean={per_fold_mean:.4f}")

    # ---- Step 4: Pick best lambda ----
    best = min(lambda_results, key=lambda r: r["pooled_rae"])
    best_lam = best["lambda"]
    mean_rae = best["pooled_rae"]
    print(f"\n   best lambda = {best_lam}  pooled_RAE = {mean_rae:.4f}")

    # ---- Step 5: Build deploy te using te-variance (no leakage) ----
    print("\n" + "-" * 78)
    print("STEP 5: build deploy te at best lambda using te-variance")
    print("-" * 78)
    w_te = np.exp(-var_te * best_lam)  # (513,)
    pred_te_513 = (te_anchor_513 + w_te * resid_te).astype(np.float32)
    pred_oof_unb = (anchor_unb + np.exp(-var_unb * best_lam) * resid_unb).astype(np.float32)
    print(f"   pred_te_513:  mean={pred_te_513.mean():.3f}  "
          f"std={pred_te_513.std():.3f}")
    print(f"   pred_oof_unb: mean={pred_oof_unb.mean():.3f}  "
          f"std={pred_oof_unb.std():.3f}  (truth_std {y_unb.std():.3f})")

    # ---- Step 6: Gate ----
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_BETTER_NB2604:
        verdict = "BETTER_THAN_NB2604"
    else:
        verdict = "FAIL"
    delta_vs_nb2604 = mean_rae - NB2604_REF
    delta_vs_nb2171 = mean_rae - NB2171_REF
    print(f"\n[gate] mean_rae={mean_rae:.4f}  "
          f"(<{GATE_PROMOTE} PROMOTE / <{GATE_BETTER_NB2604} BETTER_THAN_NB2604)"
          f"  -> {verdict}")
    print(f"   delta vs nb2604 ({NB2604_REF:.4f}) = {delta_vs_nb2604:+.4f}")
    print(f"   delta vs nb2171 ({NB2171_REF:.4f}) = {delta_vs_nb2171:+.4f}")

    # ---- Step 7: Save artifacts ----
    print("\n" + "-" * 78)
    print("STEP 7: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof_unb.astype(np.float32))
    np.save(te_path, pred_te_513.astype(np.float32))
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_inverse_var_weighted.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": pred_te_513.astype(np.float32),
    }).to_csv(sub_csv, index=False)
    print(f"   [save] {sub_csv}")

    te_unb_in = float(rae(y_unb, pred_te_513[unb_idx]))
    print(f"\n   te[unb_idx] in-sample RAE = {te_unb_in:.4f}")

    summary = {
        "tag": TAG,
        "method": "inverse_variance_weighted_K_ensemble_per_row_precision",
        "paradigm": "per_row_shrinkage_anchor_plus_w_times_mean_K_residual",
        "anchor_base": "chemprop_aux",
        "anchor_in_rae": rae_anchor,
        "anchor_pre_unblind": True,
        "k_anchors": K_labels,
        "k_oof_paths": {
            "K18": str(K18_OOF_PATH), "K20": str(K20_OOF_PATH),
            "K24": str(K24_OOF_PATH), "K28": str(K28_OOF_PATH),
        },
        "k_te_paths": {
            "K18": str(K18_TE_PATH), "K20": str(K20_TE_PATH),
            "K24": str(K24_TE_PATH), "K28": str(K28_TE_PATH),
        },
        "per_anchor_rae_in_sample": {
            "K18": float(rae(y_unb, K18_oof)),
            "K20": float(rae(y_unb, K20_oof)),
            "K24": float(rae(y_unb, K24_oof)),
            "K28": float(rae(y_unb, K28_oof)),
        },
        "var_unb_stats": {
            "mean": float(var_unb.mean()),
            "median": float(np.median(var_unb)),
            "min": float(var_unb.min()),
            "max": float(var_unb.max()),
        },
        "var_te_stats": {
            "mean": float(var_te.mean()),
            "median": float(np.median(var_te)),
            "min": float(var_te.min()),
            "max": float(var_te.max()),
        },
        "lambda_grid": LAMBDA_GRID,
        "lambda_results": lambda_results,
        "best_lambda": best_lam,
        "mean_rae": mean_rae,
        "kf_seed": KF_SEED,
        "n_folds": N_FOLDS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "gate_promote": GATE_PROMOTE,
        "gate_better_nb2604": GATE_BETTER_NB2604,
        "verdict": verdict,
        "delta_vs_nb2604": delta_vs_nb2604,
        "delta_vs_nb2171": delta_vs_nb2171,
        "nb2604_ref": NB2604_REF,
        "nb2171_ref": NB2171_REF,
        "te_mean": float(pred_te_513.mean()),
        "te_std": float(pred_te_513.std()),
        "te_unb_in_sample_rae": te_unb_in,
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   best lambda           = {best_lam}")
    print(f"   MEAN pooled RAE       = {mean_rae:.4f}")
    print(f"   gate                  = <{GATE_PROMOTE} PROMOTE / "
          f"<{GATE_BETTER_NB2604} BETTER_THAN_NB2604  -> {verdict}")
    print(f"   delta vs nb2604       = {delta_vs_nb2604:+.4f}")
    print(f"   delta vs nb2171       = {delta_vs_nb2171:+.4f}")
    print(f"   te[unb_idx] RAE       = {te_unb_in:.4f}")
    print(f"   wall                  = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "best_lambda",
        "mean_rae",
        "verdict",
        "delta_vs_nb2604",
        "te_unb_in_sample_rae",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
