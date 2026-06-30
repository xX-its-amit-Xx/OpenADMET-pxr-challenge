"""nb2514 -- James-Stein shrinkage of K=20 OOF toward chemprop_aux anchor.

NEW PARADIGM (closed-form):
    Prior post-hoc work on this anchor stack used either rank-stretch
    (scalar variance decompression, nb562 / nb2200) or per-decile isotonic
    routing (nb2504).  Both shape the prediction toward truth-variance.

    James-Stein flips the polarity: it SHRINKS the K=20 prediction back
    toward chemprop_aux when the residual (nb2240 - chemprop_aux) is
    dominated by noise relative to its own ||delta||^2 norm.  This is the
    EXACT closed-form Bayes-admissible estimator when k>=3, and unlike
    rank-stretch it does NOT touch the variance scale, only the lever-arm
    between K=20 and the chemprop_aux baseline.

    pred_js = chemprop_aux + max(0, 1 - (k - 2) * sigma^2 / ||delta||^2) *
              (nb2240 - chemprop_aux)

    where:
        k       = 253 (number of unblind rows)
        sigma^2 = MAD-based robust variance of fold-train residuals
                  (= (1.4826 * median(|r - median(r)|))^2)
        delta   = nb2240 - chemprop_aux on the fold-train

    The shrinkage factor 'lambda' is estimated once per fold-train and
    applied to fold-val (and to all 513 for deploy).  When the K=20 lift
    over chemprop_aux is small (||delta||^2 -> sigma^2 * (k-2)), lambda
    collapses to 0 and we deploy chemprop_aux verbatim.  When ||delta||^2
    is large, lambda -> 1 and we deploy nb2240 verbatim.

PROTOCOL (exact):
    1. nb2240_oof = nb2240_mean_bag_oof_K20.npy on 253 (K=20 anchor).
    2. chemprop_aux_oof = nb1133_chemprop_aux_pred_oof.npy on 253.
    3. 5-fold scaffold CV outer, 5 kf_seeds {1001..1005}.
    4. For each fold (tr, va):
         a. delta_tr = nb2240_oof[tr] - chemprop_aux_oof[tr]
         b. sigma_tr = 1.4826 * median(|delta_tr - median(delta_tr)|)
            sigma2_tr = sigma_tr**2
         c. norm2_tr = sum(delta_tr**2)        # ||delta||^2 on fold-train
         d. k_tr = len(tr)
         e. lambda_tr = max(0.0, 1.0 - (k_tr - 2) * sigma2_tr / norm2_tr)
         f. pred_js_va = chemprop_aux_oof[va] +
                        lambda_tr * (nb2240_oof[va] - chemprop_aux_oof[va])
    5. Pool val predictions per kf_seed -> per-seed RAE.
    6. mean_rae = mean over 5 kf_seeds.
    7. Deploy:
         a. delta_full = nb2240_oof - chemprop_aux_oof on 253
         b. lambda_full from same MAD/norm2 formula on full 253
         c. Apply same lambda_full on the 513 te:
            te = te_chemprop_aux + lambda_full * (te_nb2240_K20 - te_chemprop_aux)

GATE:
    mean_rae < 0.4570  -> "PROMOTE"
    mean_rae < 0.4601  -> "MARGINAL_BEAT"
    else               -> "FAIL"

Outputs:
    data/processed/nb2514_summary.json
    data/processed/nb2514_pred_oof.npy   (253,) float32
    data/processed/te_nb2514.npy         (513,) float32
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

TAG = "nb2514"

# ------------------------------ paths --------------------------------
NB2240_OOF_PATH = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"
CHEMPROP_AUX_OOF_PATH = DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy"
NB2240_TE_PATH = DATA_PROCESSED / "te_nb2240_K20.npy"
NB2240_TE_FALLBACK = DATA_PROCESSED / "te_nb2240.npy"
CHEMPROP_AUX_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# ------------------------------ knobs --------------------------------
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
N_FOLDS = 5
MAD_K = 1.4826  # MAD -> sigma scaling factor under normality

GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4601


# ============================================================================
# James-Stein helpers
# ============================================================================

def _mad_sigma(residuals: np.ndarray) -> float:
    """Return MAD-based robust sigma estimate (normal-consistent)."""
    if residuals.size == 0:
        return 0.0
    med = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - med)))
    return MAD_K * mad


def _js_lambda(delta_tr: np.ndarray) -> tuple[float, float, float, float]:
    """Closed-form positive-part James-Stein shrinkage factor.

    Returns: (lambda, sigma2, norm2, raw_shrink_arg)
        lambda          = max(0, 1 - (k-2) * sigma^2 / ||delta||^2)
        sigma2          = (1.4826 * median(|delta - median(delta)|))^2
        norm2           = sum(delta**2)
        raw_shrink_arg  = (k-2) * sigma^2 / ||delta||^2     (pre-clip)
    """
    k = delta_tr.size
    if k < 3:
        # JS undefined; degenerate to lambda = 1 (no shrinkage applied)
        return 1.0, 0.0, float((delta_tr ** 2).sum()), 0.0
    sigma = _mad_sigma(delta_tr)
    sigma2 = sigma * sigma
    norm2 = float((delta_tr ** 2).sum())
    if norm2 <= 0.0:
        # nothing to shrink -- delta is identically 0; lambda has no effect
        return 0.0, sigma2, 0.0, float("inf")
    raw = (k - 2) * sigma2 / norm2
    lam = max(0.0, 1.0 - raw)
    return float(lam), float(sigma2), float(norm2), float(raw)


def _apply_js(nb2240_va: np.ndarray, anchor_va: np.ndarray, lam: float) -> np.ndarray:
    """pred_js = anchor + lambda * (nb2240 - anchor)."""
    return anchor_va + lam * (nb2240_va - anchor_va)


# ============================================================================
# scaffold-CV James-Stein
# ============================================================================

def scaffold_cv_james_stein(
    nb2240_oof: np.ndarray,
    anchor_oof: np.ndarray,
    y: np.ndarray,
    scaffolds,
    kf_seeds,
    n_folds: int,
):
    """Per-fold MAD-based JS shrinkage; pool val preds for each kf_seed."""
    n = len(y)
    per_seed_oofs = []
    per_seed_rae = []
    per_seed_diag = []
    for kf_seed in kf_seeds:
        splits = scaffold_kfold_indices(
            scaffolds, n_splits=n_folds, shuffle=True, seed=kf_seed,
        )
        oof = np.full(n, np.nan, dtype=np.float64)
        fold_records = []
        for fold_i, (tr_loc, va_loc) in enumerate(splits):
            delta_tr = nb2240_oof[tr_loc] - anchor_oof[tr_loc]
            lam, sigma2, norm2, raw = _js_lambda(delta_tr)
            pred_va = _apply_js(nb2240_oof[va_loc], anchor_oof[va_loc], lam)
            oof[va_loc] = pred_va
            fold_records.append({
                "fold": int(fold_i),
                "n_train": int(tr_loc.size),
                "n_val": int(va_loc.size),
                "lambda": lam,
                "sigma2": sigma2,
                "norm2_delta": norm2,
                "raw_shrink_arg": raw,
                "delta_tr_mean": float(delta_tr.mean()),
                "delta_tr_std": float(delta_tr.std()),
            })
        if np.isnan(oof).any():
            raise RuntimeError(
                "OOF has NaN -- scaffold splits did not cover all rows"
            )
        seed_rae = float(rae(y, oof))
        per_seed_oofs.append(oof)
        per_seed_rae.append(seed_rae)
        per_seed_diag.append({
            "kf_seed": int(kf_seed),
            "rae": seed_rae,
            "mean_lambda": float(np.mean([r["lambda"] for r in fold_records])),
            "lambda_range": [
                float(min(r["lambda"] for r in fold_records)),
                float(max(r["lambda"] for r in fold_records)),
            ],
            "per_fold": fold_records,
        })
    stack = np.column_stack(per_seed_oofs)
    mean_oof = stack.mean(axis=1)
    return per_seed_rae, mean_oof, per_seed_diag


def deploy_te_james_stein(
    nb2240_oof: np.ndarray,
    anchor_oof: np.ndarray,
    nb2240_te: np.ndarray,
    anchor_te: np.ndarray,
) -> tuple[np.ndarray, float, float, float, float]:
    """Estimate lambda once on the full 253 then apply to 513 deploy preds.

    Returns: (te_pred, lambda_full, sigma2_full, norm2_full, raw_full)
    """
    delta_full = nb2240_oof - anchor_oof
    lam_full, sigma2_full, norm2_full, raw_full = _js_lambda(delta_full)
    te_pred = anchor_te + lam_full * (nb2240_te - anchor_te)
    return (
        te_pred.astype(np.float32),
        float(lam_full),
        float(sigma2_full),
        float(norm2_full),
        float(raw_full),
    )


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- James-Stein shrinkage of K=20 toward chemprop_aux")
    print("=" * 78)

    # ---- truth ----
    te = load_test()
    n_test = len(te)
    te_smiles = (
        te["smiles"].astype(str).tolist()
        if "smiles" in te.columns
        else te["SMILES"].astype(str).tolist()
    )
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    # ---- nb2240 K=20 anchor on 253 ----
    if not NB2240_OOF_PATH.exists():
        raise FileNotFoundError(NB2240_OOF_PATH)
    nb2240_oof = np.load(NB2240_OOF_PATH).astype(np.float64)
    if nb2240_oof.shape != (n_unb,):
        raise ValueError(
            f"nb2240 OOF shape mismatch: {nb2240_oof.shape}  expected ({n_unb},)"
        )
    rae_nb2240 = float(rae(y_unb, nb2240_oof))

    # ---- chemprop_aux anchor on 253 ----
    if not CHEMPROP_AUX_OOF_PATH.exists():
        raise FileNotFoundError(CHEMPROP_AUX_OOF_PATH)
    anchor_oof = np.load(CHEMPROP_AUX_OOF_PATH).astype(np.float64)
    if anchor_oof.shape != (n_unb,):
        raise ValueError(
            f"chemprop_aux OOF shape mismatch: {anchor_oof.shape}  "
            f"expected ({n_unb},)"
        )
    rae_anchor = float(rae(y_unb, anchor_oof))
    print(f"[load] nb2240 K=20    OOF in_RAE = {rae_nb2240:.4f}  "
          f"std={nb2240_oof.std():.3f}")
    print(f"[load] chemprop_aux   OOF in_RAE = {rae_anchor:.4f}  "
          f"std={anchor_oof.std():.3f}")
    delta_diag = nb2240_oof - anchor_oof
    print(f"[diag] delta_full(253)  mean={delta_diag.mean():+.4f}  "
          f"std={delta_diag.std():.4f}  "
          f"||delta||^2={float((delta_diag**2).sum()):.3f}")

    # ---- te deploy anchors on 513 ----
    if NB2240_TE_PATH.exists():
        nb2240_te = np.load(NB2240_TE_PATH).astype(np.float64)
        nb2240_te_src = str(NB2240_TE_PATH)
    elif NB2240_TE_FALLBACK.exists():
        nb2240_te = np.load(NB2240_TE_FALLBACK).astype(np.float64)
        nb2240_te_src = str(NB2240_TE_FALLBACK)
        print(f"[warn] te_nb2240_K20.npy missing, using fallback "
              f"{NB2240_TE_FALLBACK.name}")
    else:
        raise FileNotFoundError(
            f"No K=20 deploy te found: tried {NB2240_TE_PATH} and "
            f"{NB2240_TE_FALLBACK}"
        )
    if nb2240_te.shape != (n_test,):
        raise ValueError(
            f"nb2240 te shape mismatch: {nb2240_te.shape}  "
            f"expected ({n_test},)"
        )

    if not CHEMPROP_AUX_TE_PATH.exists():
        raise FileNotFoundError(CHEMPROP_AUX_TE_PATH)
    anchor_te = np.load(CHEMPROP_AUX_TE_PATH).astype(np.float64)
    if anchor_te.shape != (n_test,):
        raise ValueError(
            f"chemprop_aux te shape mismatch: {anchor_te.shape}  "
            f"expected ({n_test},)"
        )
    print(f"[load] nb2240 te       = {nb2240_te_src}  "
          f"mean={nb2240_te.mean():.3f}  std={nb2240_te.std():.3f}")
    print(f"[load] chemprop_aux te = {CHEMPROP_AUX_TE_PATH}  "
          f"mean={anchor_te.mean():.3f}  std={anchor_te.std():.3f}")

    # ---- scaffold CV James-Stein ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD CV  kf_seeds={KF_SEEDS}  n_folds={N_FOLDS}  "
          f"MAD_K={MAD_K}")
    print("-" * 78)
    per_seed_rae, mean_oof, per_seed_diag = scaffold_cv_james_stein(
        nb2240_oof, anchor_oof, y_unb, unb_scaffolds, KF_SEEDS, N_FOLDS,
    )
    for d in per_seed_diag:
        print(f"   kf_seed={d['kf_seed']}  pooled_RAE={d['rae']:.4f}  "
              f"mean_lambda={d['mean_lambda']:.3f}  "
              f"lambda_range={d['lambda_range']}")
    mean_rae = float(np.mean(per_seed_rae))
    std_rae = float(np.std(per_seed_rae))
    rae_mean_oof = float(rae(y_unb, mean_oof))
    print(f"\n[bag] mean_rae across {len(KF_SEEDS)} kf_seeds = "
          f"{mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"[bag] rae(mean_oof)                            = "
          f"{rae_mean_oof:.4f}")
    print(f"[diag] mean_oof std = {mean_oof.std():.3f}  "
          f"(nb2240 {nb2240_oof.std():.3f}, "
          f"anchor {anchor_oof.std():.3f}, "
          f"truth {y_unb.std():.3f})")

    # ---- gate ----
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"\n[gate] mean_rae={mean_rae:.4f}  "
          f"thresholds(<{GATE_PROMOTE}/<{GATE_MARGINAL})  verdict={verdict}")

    # ---- deploy 513 ----
    print("\n[deploy] estimating lambda on full 253; applying to 513...")
    te_pred, lam_full, sigma2_full, norm2_full, raw_full = deploy_te_james_stein(
        nb2240_oof, anchor_oof, nb2240_te, anchor_te,
    )
    te_unb_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"[deploy] lambda_full={lam_full:.4f}  "
          f"sigma2_full={sigma2_full:.4f}  "
          f"||delta||^2_full={norm2_full:.3f}  "
          f"raw_arg=(k-2)*sigma2/norm2={raw_full:.4f}")
    print(f"[deploy] te_unb_rae(in-sample)={te_unb_rae:.4f}  "
          f"te(513) mean={te_pred.mean():.3f}  std={te_pred.std():.3f}")

    pred_oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(pred_oof_path, mean_oof.astype(np.float32))
    np.save(te_path, te_pred)
    print(f"[save] {pred_oof_path}")
    print(f"[save] {te_path}")

    summary = {
        "tag": TAG,
        "method": "james_stein_shrinkage_K20_toward_chemprop_aux",
        "nb2240_oof_path": str(NB2240_OOF_PATH),
        "chemprop_aux_oof_path": str(CHEMPROP_AUX_OOF_PATH),
        "nb2240_te_path": nb2240_te_src,
        "chemprop_aux_te_path": str(CHEMPROP_AUX_TE_PATH),
        "anchor_pre_unblind": False,
        "n_unb": n_unb,
        "n_te": n_test,
        "n_unique_scaffolds": n_unique_scaf,
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "mad_k": MAD_K,
        "rae_nb2240_K20_oof": rae_nb2240,
        "rae_chemprop_aux_oof": rae_anchor,
        "nb2240_oof_std": float(nb2240_oof.std()),
        "anchor_oof_std": float(anchor_oof.std()),
        "delta_oof_mean": float(delta_diag.mean()),
        "delta_oof_std": float(delta_diag.std()),
        "delta_oof_norm2": float((delta_diag ** 2).sum()),
        "truth_mean": float(y_unb.mean()),
        "truth_std": float(y_unb.std()),
        "per_kf_seed_rae": [float(x) for x in per_seed_rae],
        "mean_rae": mean_rae,
        "std_rae": std_rae,
        "rae_of_mean_oof": rae_mean_oof,
        "mean_oof_std": float(mean_oof.std()),
        "per_seed_diag": per_seed_diag,
        "deploy_lambda_full": lam_full,
        "deploy_sigma2_full": sigma2_full,
        "deploy_norm2_delta_full": norm2_full,
        "deploy_raw_shrink_arg_full": raw_full,
        "deploy_te_mean": float(te_pred.mean()),
        "deploy_te_std": float(te_pred.std()),
        "te_unb_rae_in_sample": te_unb_rae,
        "gate_promote_below": GATE_PROMOTE,
        "gate_marginal_below": GATE_MARGINAL,
        "verdict": verdict,
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
    print(f"   nb2240 K=20 in_RAE             = {rae_nb2240:.4f}")
    print(f"   chemprop_aux in_RAE            = {rae_anchor:.4f}")
    print(f"   per-kf-seed RAE                = "
          f"{[float('%.4f' % r) for r in per_seed_rae]}")
    print(f"   MEAN RAE (5 kf_seeds)          = "
          f"{mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   RAE of mean-OOF                = {rae_mean_oof:.4f}")
    print(f"   deploy lambda_full             = {lam_full:.4f}")
    print(f"   gate thresholds                = <{GATE_PROMOTE} PROMOTE | "
          f"<{GATE_MARGINAL} MARGINAL")
    print(f"   verdict                        = {verdict}")
    print(f"   wall                           = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "rae_nb2240_K20_oof",
        "rae_chemprop_aux_oof",
        "mean_rae",
        "std_rae",
        "rae_of_mean_oof",
        "deploy_lambda_full",
        "verdict",
        "te_unb_rae_in_sample",
    ):
        print(f"  {k}: {res.get(k)}")
