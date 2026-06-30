"""nb2840 -- Bayesian credible interval shrinkage of K=20 anchor.

NEW PARADIGM (empirical-Bayes credible-interval midpoint):
    Prior post-hoc work on this anchor stack used either rank-stretch
    (scalar variance decompression, nb562 / nb2200), James-Stein global
    shrinkage (nb2514), or per-decile isotonic routing (nb2504).  All of
    these apply a SHARED shape transform across all 253 rows.

    nb2840 flips the polarity: the per-row prediction is the MIDPOINT of
    an empirical-Bayes credible interval built from two ingredients
    estimated separately for every row:
        1. sigma^2   = fold-train PER-ROW residual variance (noise)
        2. tau^2     = PER-ROW prediction variance (cross-anchor signal
                       dispersion at this row, a.k.a. ensemble disagreement
                       around the K=20 anchor)

    The closed-form posterior mean (under a Gaussian-Gaussian conjugate
    model with prior centred at the fold-train marginal mean mu0) is

        pred(row) = shrinkage(row) * mu0
                  + (1 - shrinkage(row)) * anchor(row)

        shrinkage(row) = sigma^2(row) / (sigma^2(row) + tau^2(row))

    Intuition: where the fold-train residual noise dominates the local
    cross-anchor signal (sigma^2 >> tau^2), the row is shrunk hard toward
    mu0 (the fold-train mean).  Where the anchors agree strongly and the
    K=20 anchor's local noise is small (tau^2 >> sigma^2), the prediction
    sticks at the K=20 anchor.

    This is the EXACT Stein-admissible posterior-mean estimator under
    the conjugate model, and unlike rank-stretch it provides PER-ROW
    capacity (each row gets its own shrinkage factor), which is exactly
    what the cycle-167 substrate-change requirement targets.

PROTOCOL (exact):
    1. Anchor              = nb2240_mean_bag_oof_K20.npy on 253.
       Deploy anchor       = te_nb2240_K20.npy on 513.
    2. Companion anchors for cross-anchor tau^2(row):
         a. chemprop_aux   = nb1133_chemprop_aux_pred_oof.npy / te_chemprop_aux.npy
         b. nb503          = nb503_pred_oof.npy           / te_nb503.npy
         c. nb562          = nb562_pred_oof.npy           / te_nb562.npy
         d. nb1191         = (reconstructed via nb1191 deploy weights, same
                             pattern as nb2171) / te_nb1191.npy
       tau^2(row) = variance across {K=20, chemprop_aux, nb503, nb562, nb1191}
                    at that row (5-anchor cross-section).
    3. 5-fold scaffold CV outer, kf_seed 1001 (per spec, single seed).
    4. For each outer fold (tr, va):
         a. mu0_tr = mean(y[tr])              # fold-train marginal mean
         b. Inner 5-fold CV on tr:
              - For each inner fold, fit a 1-D linear-regression calibrator
                anchor -> y on inner-train, predict inner-val.
              - Collect residuals over tr.
            Local sigma^2(row) is then the within-bin residual variance,
            where bins are 10 quantiles of the K=20 anchor on tr.
            For va rows, sigma^2(row) = bin variance of the matching bin
            (chosen from tr-anchor bin edges).  Bin variance is a stable
            local-noise estimate at n_tr~200; bins with <5 rows fall back
            to the global tr residual variance.
         c. tau^2(row) for va = sample variance across the 5 anchors at
            that row (population variance, ddof=0).
         d. shrinkage_va = sigma^2_va / (sigma^2_va + tau^2_va)
         e. pred_va      = shrinkage_va * mu0_tr
                         + (1 - shrinkage_va) * anchor_va
    5. Pool val preds -> outer-RAE for kf_seed 1001.
    6. mean_rae = outer-RAE (single seed per spec; report std=0).
    7. Deploy on 513:
         a. Full-pool mu0 = mean(y_unb).
         b. Full-pool inner-CV residuals on 253 -> per-bin sigma^2.
         c. Match each of 513 to bin via K=20 anchor; lookup sigma^2.
         d. tau^2 for 513 = cross-anchor variance on the te side.
         e. shrinkage_te, then deploy pred.

GATE:
    mean_rae < 0.4570  -> "PROMOTE"
    mean_rae < 0.4598  -> "MARGINAL_BEAT"
    else               -> "FAIL"

Outputs:
    data/processed/nb2840_summary.json
    data/processed/nb2840_pred_oof.npy   (253,) float32
    data/processed/te_nb2840.npy         (513,) float32
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

TAG = "nb2840"

# ------------------------------ paths --------------------------------
# Primary anchor: K=20 mean bag (highest cycle-160+ honest ceiling)
ANCHOR_OOF_PATH = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_nb2240_K20.npy"
ANCHOR_TE_FALLBACK = DATA_PROCESSED / "te_nb2240.npy"

# Companion anchors for cross-anchor tau^2(row)
CHEMPROP_OOF = DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy"
CHEMPROP_TE  = DATA_PROCESSED / "te_chemprop_aux.npy"
NB503_OOF    = DATA_PROCESSED / "nb503_pred_oof.npy"
NB503_TE     = DATA_PROCESSED / "te_nb503.npy"
NB562_OOF    = DATA_PROCESSED / "nb562_pred_oof.npy"
NB562_TE     = DATA_PROCESSED / "te_nb562.npy"
NB1191_TE    = DATA_PROCESSED / "te_nb1191.npy"

# nb1191 OOF reconstruction (cycle-167 pattern from nb2171)
NB1191_DEPLOY_WEIGHTS = {
    "chemprop_aux": 0.0,
    "nb1150":       0.641721304028517,
    "nb1158_K32":   0.23970131778546713,
    "nb2112_K28":   0.11857737818601592,
}
NB1191_DEPLOY_S = 1.031
NB1150_SLSQP4_OOFS = [
    "nb1133_chemprop_aux_pred_oof.npy",
    "nb503_pred_oof.npy",
    "nb1133_nb1014_pred_oof.npy",
    "nb2103_mean_bag_oof_K28.npy",
]
NB1150_SLSQP4_WEIGHTS = [0.0, 0.2942, 0.0, 0.7058]

# ------------------------------ knobs --------------------------------
KF_SEED = 1001            # per spec: single outer seed
N_FOLDS_OUTER = 5
N_FOLDS_INNER = 5
N_BINS = 10               # quantile bins for local sigma^2
MIN_BIN_N = 5             # fall back to global if bin smaller than this
EPS = 1e-9                # divide-by-zero guard

GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598


# ============================================================================
# nb1191 OOF reconstruction (matches nb2171 byte-for-byte)
# ============================================================================

def reconstruct_nb1150_oof(n_unb: int) -> np.ndarray:
    cols = []
    for rel in NB1150_SLSQP4_OOFS:
        p = DATA_PROCESSED / rel
        if not p.exists():
            raise FileNotFoundError(f"missing nb1150 sub-anchor OOF: {p}")
        v = np.load(p).astype(np.float64)
        if v.shape != (n_unb,):
            raise ValueError(f"{p.name} shape {v.shape}, expected ({n_unb},)")
        cols.append(v)
    P = np.column_stack(cols)
    w = np.asarray(NB1150_SLSQP4_WEIGHTS, dtype=np.float64)
    return P @ w


def reconstruct_nb1191_oof(n_unb: int) -> np.ndarray:
    chemprop_oof = np.load(CHEMPROP_OOF).astype(np.float64)
    nb1150_oof = reconstruct_nb1150_oof(n_unb)
    nb1158_oof = np.load(
        DATA_PROCESSED / "nb1158_mean_bag_oof_K32.npy"
    ).astype(np.float64)
    nb2112_oof = np.load(
        DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy"
    ).astype(np.float64)
    blend = (
        NB1191_DEPLOY_WEIGHTS["chemprop_aux"] * chemprop_oof
        + NB1191_DEPLOY_WEIGHTS["nb1150"]       * nb1150_oof
        + NB1191_DEPLOY_WEIGHTS["nb1158_K32"]   * nb1158_oof
        + NB1191_DEPLOY_WEIGHTS["nb2112_K28"]   * nb2112_oof
    )
    mu = float(blend.mean())
    return mu + NB1191_DEPLOY_S * (blend - mu)


# ============================================================================
# Local sigma^2 estimator (per-row residual variance via inner-CV + binning)
# ============================================================================

def _fit_linear_calibrator(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Closed-form OLS slope/intercept for 1-D y = a + b*x."""
    if x.size < 2:
        return float(y.mean() if y.size else 0.0), 0.0
    x_m = x.mean()
    y_m = y.mean()
    num = float(np.sum((x - x_m) * (y - y_m)))
    den = float(np.sum((x - x_m) ** 2))
    b = num / den if den > EPS else 0.0
    a = y_m - b * x_m
    return float(a), float(b)


def _inner_cv_residuals(
    anchor_tr: np.ndarray,
    y_tr: np.ndarray,
    scaffolds_tr: list[str],
    n_folds_inner: int,
    seed: int,
) -> np.ndarray:
    """Return per-row residuals on tr via inner scaffold CV.

    Predictor is a 1-D linear calibrator of the anchor on inner-train,
    applied to inner-val. This is the cheapest unbiased estimator of
    fold-train residual structure that respects scaffold leakage.
    """
    n = len(y_tr)
    inner_pred = np.full(n, np.nan, dtype=np.float64)
    splits = scaffold_kfold_indices(
        scaffolds_tr, n_splits=n_folds_inner, shuffle=True, seed=seed,
    )
    for itr_loc, iva_loc in splits:
        a, b = _fit_linear_calibrator(anchor_tr[itr_loc], y_tr[itr_loc])
        inner_pred[iva_loc] = a + b * anchor_tr[iva_loc]
    # Singletons / empty splits: fall back to global anchor calibrator
    if np.isnan(inner_pred).any():
        a, b = _fit_linear_calibrator(anchor_tr, y_tr)
        mask = np.isnan(inner_pred)
        inner_pred[mask] = a + b * anchor_tr[mask]
    return y_tr - inner_pred


def _bin_sigma2(
    anchor_tr: np.ndarray,
    residuals_tr: np.ndarray,
    n_bins: int,
    min_bin_n: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Quantile-bin the anchor on tr; compute per-bin residual variance.

    Returns (bin_edges, bin_sigma2, global_sigma2).
    bin_edges has length n_bins+1 (clipped to anchor range).
    bin_sigma2 has length n_bins; bins with fewer than min_bin_n rows
    inherit the global variance.
    """
    global_var = float(np.var(residuals_tr, ddof=0))
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(anchor_tr, quantiles)
    # Make edges strictly monotone (handle ties)
    edges = np.maximum.accumulate(edges)
    edges[0] = -np.inf
    edges[-1] = np.inf
    bin_idx_tr = np.clip(
        np.searchsorted(edges[1:-1], anchor_tr, side="right"),
        0, n_bins - 1,
    )
    sigma2 = np.full(n_bins, global_var, dtype=np.float64)
    for b in range(n_bins):
        mask = bin_idx_tr == b
        if mask.sum() >= min_bin_n:
            sigma2[b] = float(np.var(residuals_tr[mask], ddof=0))
    return edges, sigma2, global_var


def _lookup_sigma2(
    anchor_va: np.ndarray,
    edges: np.ndarray,
    sigma2: np.ndarray,
) -> np.ndarray:
    """Map each va row to its tr-bin sigma^2."""
    n_bins = sigma2.shape[0]
    bin_idx_va = np.clip(
        np.searchsorted(edges[1:-1], anchor_va, side="right"),
        0, n_bins - 1,
    )
    return sigma2[bin_idx_va]


def _row_tau2(stack: np.ndarray) -> np.ndarray:
    """Per-row variance across anchors (population variance, ddof=0)."""
    return np.var(stack, axis=1, ddof=0)


# ============================================================================
# Outer CV pass
# ============================================================================

def cv_credible_interval(
    anchor_oof: np.ndarray,
    companion_stack: np.ndarray,
    y: np.ndarray,
    scaffolds: list[str],
    kf_seed: int,
    n_folds_outer: int,
    n_folds_inner: int,
    n_bins: int,
    min_bin_n: int,
):
    """Run one outer scaffold-CV pass; return OOF + per-fold diagnostics."""
    n = len(y)
    oof = np.full(n, np.nan, dtype=np.float64)
    splits = scaffold_kfold_indices(
        scaffolds, n_splits=n_folds_outer, shuffle=True, seed=kf_seed,
    )
    fold_records = []
    # Pre-compute global tau^2 across all rows (companion variance) for
    # diagnostic only -- the deploy uses te-side cross-anchor variance.
    full_anchor_stack = np.column_stack([anchor_oof, companion_stack])
    tau2_full = _row_tau2(full_anchor_stack)

    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        mu0_tr = float(y[tr_loc].mean())
        scaf_tr = [scaffolds[i] for i in tr_loc]
        residuals_tr = _inner_cv_residuals(
            anchor_oof[tr_loc], y[tr_loc], scaf_tr, n_folds_inner, kf_seed,
        )
        edges, bin_sigma2, global_var = _bin_sigma2(
            anchor_oof[tr_loc], residuals_tr, n_bins, min_bin_n,
        )
        sigma2_va = _lookup_sigma2(anchor_oof[va_loc], edges, bin_sigma2)
        tau2_va = tau2_full[va_loc]
        denom = sigma2_va + tau2_va
        shrink_va = np.where(denom > EPS, sigma2_va / denom, 0.0)
        pred_va = shrink_va * mu0_tr + (1.0 - shrink_va) * anchor_oof[va_loc]
        oof[va_loc] = pred_va
        fold_records.append({
            "fold": int(fold_i),
            "n_train": int(tr_loc.size),
            "n_val": int(va_loc.size),
            "mu0_tr": mu0_tr,
            "global_sigma2_tr": global_var,
            "bin_sigma2_min": float(bin_sigma2.min()),
            "bin_sigma2_max": float(bin_sigma2.max()),
            "tau2_va_mean": float(tau2_va.mean()),
            "shrink_va_mean": float(shrink_va.mean()),
            "shrink_va_min": float(shrink_va.min()),
            "shrink_va_max": float(shrink_va.max()),
        })
    if np.isnan(oof).any():
        raise RuntimeError("OOF has NaN -- scaffold splits did not cover all rows")
    pooled = float(rae(y, oof))
    return pooled, oof, fold_records


# ============================================================================
# Deploy on 513
# ============================================================================

def deploy_credible_interval(
    anchor_oof: np.ndarray,
    companion_oof_stack: np.ndarray,
    y: np.ndarray,
    scaffolds: list[str],
    anchor_te: np.ndarray,
    companion_te_stack: np.ndarray,
    kf_seed: int,
    n_folds_inner: int,
    n_bins: int,
    min_bin_n: int,
):
    """Estimate full-pool sigma^2 bins on 253 then apply to 513."""
    mu0_full = float(y.mean())
    residuals_full = _inner_cv_residuals(
        anchor_oof, y, scaffolds, n_folds_inner, kf_seed,
    )
    edges, bin_sigma2, global_var = _bin_sigma2(
        anchor_oof, residuals_full, n_bins, min_bin_n,
    )
    sigma2_te = _lookup_sigma2(anchor_te, edges, bin_sigma2)
    te_stack = np.column_stack([anchor_te, companion_te_stack])
    tau2_te = _row_tau2(te_stack)
    denom = sigma2_te + tau2_te
    shrink_te = np.where(denom > EPS, sigma2_te / denom, 0.0)
    te_pred = shrink_te * mu0_full + (1.0 - shrink_te) * anchor_te
    diag = {
        "mu0_full": mu0_full,
        "global_sigma2_full": global_var,
        "bin_sigma2_min": float(bin_sigma2.min()),
        "bin_sigma2_max": float(bin_sigma2.max()),
        "tau2_te_mean": float(tau2_te.mean()),
        "shrink_te_mean": float(shrink_te.mean()),
        "shrink_te_min": float(shrink_te.min()),
        "shrink_te_max": float(shrink_te.max()),
    }
    return te_pred.astype(np.float32), diag


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Bayesian credible interval shrinkage of K=20 anchor")
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

    # ---- K=20 anchor (OOF on 253, te on 513) ----
    if not ANCHOR_OOF_PATH.exists():
        raise FileNotFoundError(ANCHOR_OOF_PATH)
    anchor_oof = np.load(ANCHOR_OOF_PATH).astype(np.float64)
    if anchor_oof.shape != (n_unb,):
        raise ValueError(
            f"anchor OOF shape {anchor_oof.shape}, expected ({n_unb},)"
        )
    rae_anchor = float(rae(y_unb, anchor_oof))

    if ANCHOR_TE_PATH.exists():
        anchor_te = np.load(ANCHOR_TE_PATH).astype(np.float64)
        anchor_te_src = str(ANCHOR_TE_PATH)
    elif ANCHOR_TE_FALLBACK.exists():
        anchor_te = np.load(ANCHOR_TE_FALLBACK).astype(np.float64)
        anchor_te_src = str(ANCHOR_TE_FALLBACK)
        print(f"[warn] te_nb2240_K20.npy missing, fallback "
              f"{ANCHOR_TE_FALLBACK.name}")
    else:
        raise FileNotFoundError(
            f"No K=20 te found: {ANCHOR_TE_PATH} / {ANCHOR_TE_FALLBACK}"
        )
    if anchor_te.shape != (n_test,):
        raise ValueError(
            f"anchor te shape {anchor_te.shape}, expected ({n_test},)"
        )
    print(f"[anchor] K=20  oof_RAE={rae_anchor:.4f}  "
          f"oof_std={anchor_oof.std():.3f}  "
          f"te_mean={anchor_te.mean():.3f}  te_std={anchor_te.std():.3f}")

    # ---- companion anchors for cross-anchor tau^2(row) ----
    print("\n[companions]")
    companion_oofs = []
    companion_tes = []
    companion_names = []

    def _add_companion(name: str, oof_path: Path, te_path: Path,
                       oof_arr: np.ndarray | None = None,
                       te_arr: np.ndarray | None = None):
        if oof_arr is None:
            if not oof_path.exists():
                raise FileNotFoundError(oof_path)
            oof_arr = np.load(oof_path).astype(np.float64)
        if te_arr is None:
            if not te_path.exists():
                raise FileNotFoundError(te_path)
            te_arr = np.load(te_path).astype(np.float64)
        if oof_arr.shape != (n_unb,):
            raise ValueError(f"{name} oof shape {oof_arr.shape}")
        if te_arr.shape != (n_test,):
            raise ValueError(f"{name} te shape {te_arr.shape}")
        r = float(rae(y_unb, oof_arr))
        companion_oofs.append(oof_arr)
        companion_tes.append(te_arr)
        companion_names.append(name)
        print(f"   {name:14s} oof_RAE={r:.4f}  oof_std={oof_arr.std():.3f}  "
              f"te_std={te_arr.std():.3f}")

    _add_companion("chemprop_aux", CHEMPROP_OOF, CHEMPROP_TE)
    _add_companion("nb503", NB503_OOF, NB503_TE)
    _add_companion("nb562", NB562_OOF, NB562_TE)

    # nb1191 reconstructed OOF + cached te
    nb1191_oof = reconstruct_nb1191_oof(n_unb)
    nb1191_te = np.load(NB1191_TE).astype(np.float64) if NB1191_TE.exists() else None
    if nb1191_te is None:
        raise FileNotFoundError(NB1191_TE)
    if nb1191_te.shape != (n_test,):
        raise ValueError(f"nb1191 te shape {nb1191_te.shape}")
    _add_companion("nb1191", NB1191_TE, NB1191_TE,
                   oof_arr=nb1191_oof, te_arr=nb1191_te)

    companion_oof_stack = np.column_stack(companion_oofs)
    companion_te_stack = np.column_stack(companion_tes)
    print(f"[companions] stack shape oof={companion_oof_stack.shape}  "
          f"te={companion_te_stack.shape}")

    # ---- scaffold CV credible-interval shrinkage ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD CV  kf_seed={KF_SEED}  n_folds_outer={N_FOLDS_OUTER}  "
          f"n_folds_inner={N_FOLDS_INNER}  n_bins={N_BINS}")
    print("-" * 78)
    pooled_rae, mean_oof, fold_records = cv_credible_interval(
        anchor_oof, companion_oof_stack, y_unb, unb_scaffolds,
        KF_SEED, N_FOLDS_OUTER, N_FOLDS_INNER, N_BINS, MIN_BIN_N,
    )
    for r in fold_records:
        print(f"   fold={r['fold']}  n_va={r['n_val']}  "
              f"mu0_tr={r['mu0_tr']:.3f}  "
              f"global_sigma2={r['global_sigma2_tr']:.3f}  "
              f"bin_sigma2=[{r['bin_sigma2_min']:.3f},{r['bin_sigma2_max']:.3f}]  "
              f"tau2_va_mean={r['tau2_va_mean']:.3f}  "
              f"shrink_va=[{r['shrink_va_min']:.3f},{r['shrink_va_max']:.3f}] "
              f"mean={r['shrink_va_mean']:.3f}")
    mean_rae = pooled_rae
    std_rae = 0.0  # single kf_seed per spec
    print(f"\n[cv] pooled_RAE (kf_seed {KF_SEED}) = {mean_rae:.4f}")
    print(f"[cv] mean_oof std = {mean_oof.std():.3f}  "
          f"(anchor {anchor_oof.std():.3f}, truth {y_unb.std():.3f})")

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
    print("\n[deploy] estimating full-pool sigma^2 bins on 253; "
          "applying to 513...")
    te_pred, deploy_diag = deploy_credible_interval(
        anchor_oof, companion_oof_stack, y_unb, unb_scaffolds,
        anchor_te, companion_te_stack,
        KF_SEED, N_FOLDS_INNER, N_BINS, MIN_BIN_N,
    )
    te_unb_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"[deploy] mu0_full={deploy_diag['mu0_full']:.3f}  "
          f"global_sigma2={deploy_diag['global_sigma2_full']:.3f}  "
          f"bin_sigma2=[{deploy_diag['bin_sigma2_min']:.3f},"
          f"{deploy_diag['bin_sigma2_max']:.3f}]")
    print(f"[deploy] tau2_te_mean={deploy_diag['tau2_te_mean']:.3f}  "
          f"shrink_te=[{deploy_diag['shrink_te_min']:.3f},"
          f"{deploy_diag['shrink_te_max']:.3f}] "
          f"mean={deploy_diag['shrink_te_mean']:.3f}")
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
        "method": "empirical_bayes_credible_interval_shrinkage_K20_per_row",
        "anchor_oof_path": str(ANCHOR_OOF_PATH),
        "anchor_te_path": anchor_te_src,
        "companion_names": companion_names,
        "companion_oof_paths": [
            str(CHEMPROP_OOF), str(NB503_OOF), str(NB562_OOF),
            "reconstructed_nb1191_from_nb2171_weights",
        ],
        "companion_te_paths": [
            str(CHEMPROP_TE), str(NB503_TE), str(NB562_TE), str(NB1191_TE),
        ],
        "anchor_pre_unblind": False,   # K=20 anchor is POST-unblind regime
        "n_unb": n_unb,
        "n_te": n_test,
        "n_unique_scaffolds": n_unique_scaf,
        "kf_seed": KF_SEED,
        "n_folds_outer": N_FOLDS_OUTER,
        "n_folds_inner": N_FOLDS_INNER,
        "n_bins": N_BINS,
        "min_bin_n": MIN_BIN_N,
        "rae_anchor_K20_oof": rae_anchor,
        "rae_companion_oof": {
            n: float(rae(y_unb, c))
            for n, c in zip(companion_names, companion_oofs)
        },
        "anchor_oof_std": float(anchor_oof.std()),
        "anchor_te_std": float(anchor_te.std()),
        "truth_mean": float(y_unb.mean()),
        "truth_std": float(y_unb.std()),
        "pooled_rae_outer": mean_rae,
        "mean_rae": mean_rae,
        "std_rae": std_rae,
        "mean_oof_std": float(mean_oof.std()),
        "fold_records": fold_records,
        "deploy_diag": deploy_diag,
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
    print(f"   K=20 anchor in_RAE             = {rae_anchor:.4f}")
    print(f"   companions in_RAE              = "
          f"{ {n: float(f'%.4f' % float(rae(y_unb, c))) for n, c in zip(companion_names, companion_oofs)} }")
    print(f"   MEAN RAE (kf_seed {KF_SEED})       = "
          f"{mean_rae:.4f}  (single seed; std reported 0)")
    print(f"   deploy shrink_te mean          = "
          f"{deploy_diag['shrink_te_mean']:.3f}")
    print(f"   te_unb_rae(in-sample)          = {te_unb_rae:.4f}")
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
        "rae_anchor_K20_oof",
        "mean_rae",
        "std_rae",
        "verdict",
        "te_unb_rae_in_sample",
    ):
        print(f"  {k}: {res.get(k)}")
