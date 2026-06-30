"""nb2491 -- Per-fold lambda search for K=20 residual via golden-section.

CONTEXT:
    Prior fixed-lambda work (nb2023/nb2031/nb2121) sweeps a single global
    lambda over the chemprop_aux + lambda * K_residual_correction blend.
    nb2491 instead searches lambda INSIDE EACH SCAFFOLD FOLD independently
    via golden-section search (0.01 to 5.0, tolerance 1e-3), then applies
    the best fold-lambda to the fold-val.  Distinct from the fixed-lambda
    family because each fold gets its own optimal lambda tuned on
    fold-train RAE -- no single global lambda choice.

ANCHOR:
    chemprop_aux te[unb_idx] -- the only PRE-unblind clean anchor.

CORRECTION:
    K=20 residual correction = nb2240_mean_bag_oof_K20 - anchor_unb.
    nb2240 OOF is anchor + lambda=1.0 residual already, so subtracting
    anchor recovers the pure K=20 LGBM residual cross-fit on the 253.

PROTOCOL:
    For each scaffold-CV fold:
        Define f(lambda) = RAE_fold_train( anchor + lambda * resid_corr )
        on fold-train rows.  Golden-section search on [0.01, 5.0] with
        tolerance 1e-3 to find lambda*.  Apply lambda* to fold-val:
            pred_va = anchor_va + lambda* * resid_corr_va.
    Pool va preds across 5 scaffold folds -> per-seed RAE.
    5 kf_seeds in {1001..1005}, mean -> headline RAE.

GATE:
    mean_rae < 0.4570  -> "PROMOTE"
    mean_rae < 0.4601  -> "MARGINAL_BEAT"   (nb2240 deep-30 ref ~0.4601)
    else               -> "FAIL"

Outputs:
    scripts/nb2491_perfold_lambda_kbma.py
    data/processed/nb2491_summary.json
    data/processed/nb2491_pred_oof.npy   (253,) float32
    data/processed/te_nb2491.npy         (513,) float32
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

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2491"

# ---------- inputs ----------
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
K20_OOF_PATH   = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"
K20_TE_PATH    = DATA_PROCESSED / "te_nb2240_K20.npy"
UNB_IDX_PATH   = DATA_PROCESSED / "_audit_unblind_idx.npy"
UNB_Y_PATH     = DATA_PROCESSED / "_audit_unblind_y.npy"

# ---------- outer CV ----------
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# ---------- golden-section search ----------
LAMBDA_LO = 0.01
LAMBDA_HI = 5.0
LAMBDA_TOL = 1e-3
PHI = (1.0 + np.sqrt(5.0)) / 2.0  # ~1.618
INV_PHI = 1.0 / PHI

# ---------- gates ----------
GATE_PROMOTE = 0.4570
GATE_MARGINAL_REF = 0.4601


# ============================================================================
# golden-section search on lambda within a fold
# ============================================================================

def golden_section_lambda(anchor_tr, resid_corr_tr, y_tr,
                          lo=LAMBDA_LO, hi=LAMBDA_HI, tol=LAMBDA_TOL,
                          max_iter=200):
    """Find lambda in [lo, hi] minimizing RAE(y_tr, anchor + lambda*corr).

    Standard golden-section search.  Assumes unimodality (RAE is a smooth
    convex-ish function of lambda for fixed anchor + correction direction).
    """
    a, b = float(lo), float(hi)

    def f(x):
        return float(rae(y_tr, anchor_tr + x * resid_corr_tr))

    c = b - (b - a) * INV_PHI
    d = a + (b - a) * INV_PHI
    fc = f(c)
    fd = f(d)
    n_iter = 0
    while abs(b - a) > tol and n_iter < max_iter:
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - (b - a) * INV_PHI
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + (b - a) * INV_PHI
            fd = f(d)
        n_iter += 1

    lam_star = 0.5 * (a + b)
    rae_star = f(lam_star)
    return lam_star, rae_star, n_iter


def cv_run_for_seed(anchor_unb, resid_corr_unb, y_unb, unb_scaffolds, kf_seed):
    """Per-fold golden-section lambda search; apply best fold-lambda to fold-val."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
    fold_records = []
    for fold_idx, (tr_loc, va_loc) in enumerate(splits):
        lam_star, rae_tr, n_iter = golden_section_lambda(
            anchor_unb[tr_loc], resid_corr_unb[tr_loc], y_unb[tr_loc],
        )
        pred_va = anchor_unb[va_loc] + lam_star * resid_corr_unb[va_loc]
        oof_blend[va_loc] = pred_va
        rae_va = float(rae(y_unb[va_loc], pred_va))
        fold_records.append({
            "fold": int(fold_idx),
            "lambda_star": float(lam_star),
            "rae_train_at_lam_star": float(rae_tr),
            "rae_val_at_lam_star": rae_va,
            "n_iter": int(n_iter),
            "n_train": int(len(tr_loc)),
            "n_val": int(len(va_loc)),
        })
    pooled = float(rae(y_unb, oof_blend))
    return pooled, oof_blend, fold_records


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- per-fold lambda search (golden-section) for K=20 residual")
    print("=" * 78)

    # ---- truth + anchor + correction ----
    te_df = load_test()
    n_test = len(te_df)
    te_smiles = (te_df["smiles"].astype(str).tolist()
                 if "smiles" in te_df.columns
                 else te_df["SMILES"].astype(str).tolist())
    te_names = (te_df["name"].values
                if "name" in te_df.columns
                else te_df["Molecule Name"].values)

    unb_idx = np.load(UNB_IDX_PATH)
    y_unb = np.load(UNB_Y_PATH).astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    # Anchor on 513 then slice to 253
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    print(f"[anchor] chemprop_aux in_RAE = {rae_anchor:.4f}")

    # K=20 corrected OOF (anchor + 1.0*K20_resid_oof)
    k20_oof_corrected = np.load(K20_OOF_PATH).astype(np.float64)
    assert k20_oof_corrected.shape == (n_unb,), f"K20 oof shape {k20_oof_corrected.shape}"
    rae_K20 = float(rae(y_unb, k20_oof_corrected))
    print(f"[K20] corrected in_RAE       = {rae_K20:.4f}  (lambda=1.0)")

    # Recover pure K=20 residual correction
    resid_corr_unb = k20_oof_corrected - anchor_unb
    print(f"[resid] K20 residual_corr   mean={resid_corr_unb.mean():+.4f} "
          f"std={resid_corr_unb.std():.4f}  "
          f"range=[{resid_corr_unb.min():+.4f}, {resid_corr_unb.max():+.4f}]")

    # te side: nb2240_K20 te = chemprop_aux te + K20_residual_te
    te_k20_513 = np.load(K20_TE_PATH).astype(np.float64)
    assert te_k20_513.shape == (n_test,), f"K20 te shape {te_k20_513.shape}"
    resid_corr_te = te_k20_513 - te_anchor_513
    print(f"[te] K20 residual_corr_te   mean={resid_corr_te.mean():+.4f} "
          f"std={resid_corr_te.std():.4f}")

    # ---- per-seed scaffold CV with per-fold golden-section lambda ----
    print("\n" + "-" * 78)
    print(f"OUTER SCAFFOLD 5-FOLD CV  kf_seeds={KF_SEEDS}  "
          f"golden-section [{LAMBDA_LO}, {LAMBDA_HI}]  tol={LAMBDA_TOL}")
    print("-" * 78)
    per_seed = []
    all_oofs = []
    all_fold_lams = []
    for kf_seed in KF_SEEDS:
        pooled, oof_blend, fold_recs = cv_run_for_seed(
            anchor_unb, resid_corr_unb, y_unb, unb_scaffolds, kf_seed,
        )
        lams = [r["lambda_star"] for r in fold_recs]
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": float(pooled),
            "fold_lambdas": [float(x) for x in lams],
            "fold_records": fold_recs,
            "mean_lambda": float(np.mean(lams)),
            "median_lambda": float(np.median(lams)),
            "std_lambda": float(np.std(lams)),
        })
        all_oofs.append(oof_blend)
        all_fold_lams.extend(lams)
        print(f"   seed={kf_seed}  pooled_RAE={pooled:.4f}  "
              f"lams={np.round(lams, 3).tolist()}  "
              f"mean_lam={np.mean(lams):.3f}")

    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    final_oof_rae = float(rae(y_unb, mean_oof))
    pooled_rae_mean_seeds = float(np.mean([r["pooled_rae"] for r in per_seed]))
    pooled_rae_std_seeds = float(np.std([r["pooled_rae"] for r in per_seed]))
    pooled_rae_min_seeds = float(np.min([r["pooled_rae"] for r in per_seed]))
    pooled_rae_max_seeds = float(np.max([r["pooled_rae"] for r in per_seed]))

    print(f"\n[cv] mean across {len(KF_SEEDS)} seeds  pooled_RAE = "
          f"{pooled_rae_mean_seeds:.4f}  (+/- {pooled_rae_std_seeds:.4f})  "
          f"range=[{pooled_rae_min_seeds:.4f}, {pooled_rae_max_seeds:.4f}]")
    print(f"[cv] RAE of mean-of-seed OOFs           = {final_oof_rae:.4f}")
    print(f"[cv] global fold-lambda  mean={np.mean(all_fold_lams):.3f}  "
          f"median={np.median(all_fold_lams):.3f}  std={np.std(all_fold_lams):.3f}  "
          f"range=[{np.min(all_fold_lams):.3f}, {np.max(all_fold_lams):.3f}]")

    # ---- Deploy ----
    print("\n" + "-" * 78)
    print("DEPLOY (golden-section on full 253; predict te(513))")
    print("-" * 78)
    lam_deploy, rae_full, n_iter_deploy = golden_section_lambda(
        anchor_unb, resid_corr_unb, y_unb,
    )
    print(f"   deploy lambda*    = {lam_deploy:.4f}   ({n_iter_deploy} iters)")
    print(f"   in-sample RAE     = {rae_full:.4f}")
    deploy_te = (te_anchor_513 + lam_deploy * resid_corr_te).astype(np.float32)
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))
    print(f"   te[unb_idx] RAE   = {te_unb_rae:.4f}")
    print(f"   te(513) mean/std  = {deploy_te.mean():.3f}/{deploy_te.std():.3f}")

    # ---- gate ----
    print("\n" + "-" * 78)
    print(f"GATE  PROMOTE<{GATE_PROMOTE:.4f}  MARGINAL_BEAT<{GATE_MARGINAL_REF:.4f}")
    print("-" * 78)
    if pooled_rae_mean_seeds < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif pooled_rae_mean_seeds < GATE_MARGINAL_REF:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"   pooled_RAE mean = {pooled_rae_mean_seeds:.4f}")
    print(f"   verdict         = {verdict}")

    # ---- save artefacts ----
    pred_oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(pred_oof_path, mean_oof.astype(np.float32))
    np.save(te_path, deploy_te)
    print(f"\n[save] {pred_oof_path}")
    print(f"[save] {te_path}")

    summary = {
        "tag": TAG,
        "method": "perfold_lambda_search_goldensec_K20_residual_chempropaux_anchor",
        "anchor": "chemprop_aux",
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_in_rae": rae_anchor,
        "correction": "nb2240_mean_bag_oof_K20 - anchor",
        "k20_oof_path": str(K20_OOF_PATH),
        "k20_te_path": str(K20_TE_PATH),
        "k20_corrected_in_rae_lambda_1": rae_K20,
        "golden_section": {
            "lambda_lo": LAMBDA_LO,
            "lambda_hi": LAMBDA_HI,
            "tol": LAMBDA_TOL,
            "phi": PHI,
        },
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "n_unb": n_unb,
        "n_te": n_test,
        "n_unique_scaffolds_unb": n_unique_scaf,
        "per_seed_results": per_seed,
        "pooled_rae_mean_seeds": pooled_rae_mean_seeds,
        "pooled_rae_std_seeds": pooled_rae_std_seeds,
        "pooled_rae_min_seeds": pooled_rae_min_seeds,
        "pooled_rae_max_seeds": pooled_rae_max_seeds,
        "rae_of_mean_of_seed_oofs": final_oof_rae,
        "global_fold_lambda_stats": {
            "mean": float(np.mean(all_fold_lams)),
            "median": float(np.median(all_fold_lams)),
            "std": float(np.std(all_fold_lams)),
            "min": float(np.min(all_fold_lams)),
            "max": float(np.max(all_fold_lams)),
            "n": int(len(all_fold_lams)),
        },
        "deploy_lambda": float(lam_deploy),
        "deploy_in_sample_rae": float(rae_full),
        "te_unb_rae_in_sample": te_unb_rae,
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "gate_promote": GATE_PROMOTE,
        "gate_marginal_ref": GATE_MARGINAL_REF,
        "verdict": verdict,
        "pre_unblind_clean": True,
        "pred_oof_path": str(pred_oof_path),
        "te_npy_path": str(te_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   pooled_RAE mean (5 seeds)   = {pooled_rae_mean_seeds:.4f}  "
          f"(+/- {pooled_rae_std_seeds:.4f})")
    print(f"   global fold-lambda mean     = {np.mean(all_fold_lams):.3f}")
    print(f"   deploy lambda*              = {lam_deploy:.4f}")
    print(f"   verdict                     = {verdict}")
    print(f"   wall                        = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "pooled_rae_mean_seeds",
        "pooled_rae_std_seeds",
        "global_fold_lambda_stats",
        "deploy_lambda",
        "verdict",
        "pred_oof_path",
        "te_npy_path",
    ):
        print(f"  {k}: {res.get(k)}")
