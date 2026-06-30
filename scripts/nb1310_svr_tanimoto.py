"""nb1310 -- SVR with precomputed Tanimoto kernel on Morgan-2048 for residual learning.

HYPOTHESIS:
    Support Vector Regression with explicit Tanimoto kernel (chemistry-native
    similarity) over Morgan-2048 directly. Classical QSAR move not yet tested.
    Tanimoto kernel is positive-definite under conditions typically satisfied
    by ECFP fingerprints, and explicitly encodes chemical similarity (vs
    descriptor regression / tree splits). May extract different signal.

PROTOCOL:
    1. Compute Morgan-2048 on standardized SMILES for test (513), slice to
       unblind 253.
    2. Build (253, 253) Tanimoto kernel via bit-AND / bit-OR counts.
       Verify positive-semi-definiteness (smallest eigenvalue >= -tol).
    3. Anchor = nb1070_pred_oof; residual target = y_unb - anchor.
    4. 5-seed bag: KFold(n=5, shuffle, seed) on residual.
       Per seed: grid CV C in {0.01, 0.1, 1.0, 10.0} via inner 3-fold mean RAE
       on the residual-corrected predictions; pick best C; refit on the
       outer training fold; predict on the outer val fold.
       SVR(kernel='precomputed', C=best, epsilon=0.1).
    5. pred_corrected_s = anchor + residual_oof_s; pooled RAE.
    6. mean_bag_oof, median_bag_oof, pooled RAE.
    7. Verdict at 0.003 margin vs nb1183 (0.5513) and nb1242 (0.5431).

NO deploy refit -- honest 253 cross-fit diagnostic.

Outputs:
    data/processed/nb1310_per_seed_corrected_oof.npy (5, 253) float32
    data/processed/nb1310_mean_bag_oof.npy           (253,)   float32
    data/processed/nb1310_median_bag_oof.npy         (253,)   float32
    data/processed/nb1310_summary.json
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from collections import Counter

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVR

from pxr.chem import morgan_fp_batch, standardize_smiles
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1310"
ANCHOR = "nb1070"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]
C_GRID = [0.01, 0.1, 1.0, 10.0]
SVR_EPSILON = 0.1
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048

# Reference numbers (pooled RAE on 253 unblind).
NB1070_REF_POOLED = 0.5771
NB1183_REF = 0.5513   # MACCS-LGBM residual mean-bag
NB1242_REF = 0.5431   # ChEMBL-LGBM residual mean-bag
DECISION_MARGIN = 0.003


def _tanimoto_kernel(fp: np.ndarray) -> np.ndarray:
    """Tanimoto kernel for binary bit vectors.

    K[i, j] = |x_i AND x_j| / |x_i OR x_j|.
    Equivalent to: inter / (|x_i| + |x_j| - inter).
    """
    X = fp.astype(np.float64)
    pop = X.sum(axis=1)                 # popcount per row
    inter = X @ X.T                     # (n, n) intersection counts
    denom = pop[:, None] + pop[None, :] - inter
    # avoid 0/0 -- if both vectors are all-zero, define similarity = 1
    out = np.where(denom > 0, inter / np.maximum(denom, 1e-12), 1.0)
    # guarantee symmetry against float drift
    out = 0.5 * (out + out.T)
    np.fill_diagonal(out, 1.0)
    return out


def _check_psd(K: np.ndarray, tol: float = 1e-6) -> dict:
    """Diagnose PSD-ness via min eigenvalue of the symmetric matrix."""
    try:
        w = np.linalg.eigvalsh(K)
        return {
            "min_eig": float(w.min()),
            "max_eig": float(w.max()),
            "is_psd": bool(w.min() > -tol),
            "cond": float(w.max() / max(abs(w.min()), 1e-12)),
        }
    except Exception as e:
        return {"min_eig": None, "max_eig": None, "is_psd": None,
                "psd_error": repr(e)}


def _resid_oof_for_seed_with_inner_cv(
    K: np.ndarray, residual: np.ndarray, anchor: np.ndarray,
    y: np.ndarray, seed: int,
) -> tuple[np.ndarray, list[float], list[dict]]:
    """5-fold residual cross-fit with inner 3-fold CV per outer fold to pick C.

    Returns the residual OOF (length n), the per-outer-fold chosen C, and
    per-outer-fold inner-CV diagnostics.
    """
    n = len(residual)
    kf_outer = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    best_Cs: list[float] = []
    outer_records: list[dict] = []
    for fold_i, (tr_loc, va_loc) in enumerate(kf_outer.split(np.arange(n))):
        # Inner 3-fold CV on tr_loc to pick best C by RAE of CORRECTED preds
        kf_inner = KFold(n_splits=3, shuffle=True, random_state=seed + 991)
        c_to_inner_rae: dict[float, list[float]] = {C: [] for C in C_GRID}
        for in_tr, in_va in kf_inner.split(tr_loc):
            tr_idx = tr_loc[in_tr]
            va_idx = tr_loc[in_va]
            K_train = K[np.ix_(tr_idx, tr_idx)]
            K_val = K[np.ix_(va_idx, tr_idx)]
            for C in C_GRID:
                mdl = SVR(kernel="precomputed", C=C, epsilon=SVR_EPSILON)
                mdl.fit(K_train, residual[tr_idx])
                resid_hat_val = mdl.predict(K_val)
                pred_corr = anchor[va_idx] + resid_hat_val
                c_to_inner_rae[C].append(float(rae(y[va_idx], pred_corr)))
        c_mean = {C: float(np.mean(v)) for C, v in c_to_inner_rae.items()}
        # tie-breaking: pick smallest mean RAE; on ties pick smaller C (more reg)
        best_C = min(C_GRID, key=lambda C: (c_mean[C], C))
        best_Cs.append(best_C)

        # Refit on full outer train fold with best_C
        K_tr_outer = K[np.ix_(tr_loc, tr_loc)]
        K_va_outer = K[np.ix_(va_loc, tr_loc)]
        mdl = SVR(kernel="precomputed", C=best_C, epsilon=SVR_EPSILON)
        mdl.fit(K_tr_outer, residual[tr_loc])
        oof[va_loc] = mdl.predict(K_va_outer)

        outer_records.append({
            "fold": fold_i,
            "best_C": best_C,
            "inner_rae_per_C": c_mean,
        })
    return oof, best_Cs, outer_records


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- SVR Tanimoto kernel residual on nb1070, Morgan-{MORGAN_NBITS} "
          f"radius={MORGAN_RADIUS}, {len(RESID_SEEDS)} KFold seeds")
    print(f"          seeds = {RESID_SEEDS}")
    print(f"          residual target = y_unb - nb1070_pred_oof")
    print(f"          C grid = {C_GRID}  epsilon = {SVR_EPSILON}")
    print("=" * 78)

    te = load_test()
    n_test = len(te)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    anchor_path = DATA_PROCESSED / f"{ANCHOR}_pred_oof.npy"
    if not anchor_path.exists():
        raise FileNotFoundError(
            f"{anchor_path} not found; required anchor OOF (run nb1070 first)."
        )
    anchor_oof = np.load(anchor_path).astype(np.float64)
    if anchor_oof.shape[0] != n_unb:
        raise ValueError(
            f"{anchor_path} shape mismatch: {anchor_oof.shape} vs n_unb={n_unb}"
        )
    rae_anchor = float(rae(y_unb, anchor_oof))
    print(f"[load] {ANCHOR}_pred_oof.npy shape={anchor_oof.shape}  "
          f"pooled RAE = {rae_anchor:.4f}  (ref ~{NB1070_REF_POOLED:.4f})")

    residual = y_unb - anchor_oof
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}  "
          f"min={residual.min():+.4f}  max={residual.max():+.4f}")

    # ---- Build Morgan-2048 on test, slice to unblind 253 ----
    print(f"\n[fp] computing Morgan-{MORGAN_NBITS} r={MORGAN_RADIUS} "
          f"on standardized test SMILES ...")
    smis_test = te["smiles"].tolist()
    std_smis_test = [standardize_smiles(s) or s for s in smis_test]
    fp_test = morgan_fp_batch(std_smis_test,
                              radius=MORGAN_RADIUS, n_bits=MORGAN_NBITS)
    fp_unb = fp_test[unb_idx]
    print(f"[fp] fp_test shape = {fp_test.shape}  fp_unb shape = {fp_unb.shape}")
    pop_unb = fp_unb.sum(axis=1)
    print(f"[fp] popcount unb: min={int(pop_unb.min())}  "
          f"median={float(np.median(pop_unb)):.1f}  max={int(pop_unb.max())}")

    # ---- Tanimoto kernel on unblind 253 ----
    print("\n[kernel] computing Tanimoto kernel (253, 253) ...")
    K = _tanimoto_kernel(fp_unb)
    print(f"[kernel] K shape = {K.shape}  "
          f"diag mean = {np.diag(K).mean():.4f}  "
          f"off-diag mean = {(K.sum() - np.trace(K)) / (K.size - K.shape[0]):.4f}  "
          f"off-diag max = {(K - np.eye(K.shape[0])).max():.4f}")
    psd = _check_psd(K)
    print(f"[kernel] PSD check: min_eig = {psd.get('min_eig')}  "
          f"max_eig = {psd.get('max_eig')}  is_psd = {psd.get('is_psd')}")

    # ---- Per-seed residual cross-fit with inner CV on C ----
    print("\n" + "-" * 78)
    print(f"PER-SEED RESIDUAL SVR CROSS-FIT (precomputed Tanimoto kernel)")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records: list[dict] = []
    per_seed_best_Cs: list[list[float]] = []
    for i, s in enumerate(RESID_SEEDS):
        resid_oof_s, best_Cs, outer_records = _resid_oof_for_seed_with_inner_cv(
            K=K, residual=residual, anchor=anchor_oof, y=y_unb, seed=s,
        )
        pred_corr_s = anchor_oof + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        per_seed_best_Cs.append(best_Cs)
        c_count = Counter(best_Cs)
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_nb1070": rae_s - rae_anchor,
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
            "best_Cs_per_outer_fold": best_Cs,
            "best_C_majority": float(c_count.most_common(1)[0][0]),
            "outer_records": outer_records,
        })
        print(f"   seed {s:3d}:  rae_corr = {rae_s:.4f}  "
              f"(d_vs_nb1070 = {rae_s - rae_anchor:+.4f})  "
              f"best_Cs = {best_Cs}  "
              f"|resid_oof|.std = {resid_oof_s.std():.3f}")

    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))

    per_seed_rae_arr = np.array(per_seed_rae)
    rae_per_seed_mean = float(per_seed_rae_arr.mean())
    rae_per_seed_median = float(np.median(per_seed_rae_arr))
    rae_per_seed_std = float(per_seed_rae_arr.std())
    rae_per_seed_min = float(per_seed_rae_arr.min())
    rae_per_seed_max = float(per_seed_rae_arr.max())

    # Global majority best-C across all seeds x outer-folds
    all_Cs = [C for L in per_seed_best_Cs for C in L]
    global_C_count = Counter(all_Cs)
    best_C_global = float(global_C_count.most_common(1)[0][0])

    print("\n" + "-" * 78)
    print("BAG AGGREGATIONS")
    print("-" * 78)
    print(f"   per-seed RAE list      = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   per-seed mean          = {rae_per_seed_mean:.4f}")
    print(f"   per-seed median        = {rae_per_seed_median:.4f}")
    print(f"   per-seed std           = {rae_per_seed_std:.4f}")
    print(f"   per-seed min/max       = "
          f"{rae_per_seed_min:.4f} / {rae_per_seed_max:.4f}")
    print(f"   pooled RAE(mean_bag)   = {rae_mean_bag:.4f}  "
          f"(d_vs_nb1070 = {rae_mean_bag - rae_anchor:+.4f})")
    print(f"   pooled RAE(median_bag) = {rae_median_bag:.4f}  "
          f"(d_vs_nb1070 = {rae_median_bag - rae_anchor:+.4f})")
    print(f"   global best-C majority = {best_C_global}  "
          f"(counts {dict(global_C_count)})")
    print(f"   nb1183 ref             = {NB1183_REF:.4f}  (MACCS-LGBM)")
    print(f"   nb1242 ref             = {NB1242_REF:.4f}  (ChEMBL-LGBM)")

    beats_nb1070 = rae_mean_bag < rae_anchor - DECISION_MARGIN
    beats_nb1183 = rae_mean_bag < NB1183_REF - DECISION_MARGIN
    beats_nb1242 = rae_mean_bag < NB1242_REF - DECISION_MARGIN

    if beats_nb1242:
        verdict = "SVR_TANIMOTO_BEATS_NB1242_NEW_BEST_RESIDUAL"
    elif beats_nb1183:
        verdict = "SVR_TANIMOTO_BEATS_NB1183_BUT_NOT_NB1242"
    elif beats_nb1070:
        verdict = "SVR_TANIMOTO_HELPS_NB1070_BUT_NOT_NB1183"
    elif abs(rae_mean_bag - rae_anchor) < DECISION_MARGIN:
        verdict = "SVR_TANIMOTO_FLAT_NO_NEW_SIGNAL"
    else:
        verdict = "SVR_TANIMOTO_HURTS_NB1070"
    print(f"   verdict                = {verdict}")

    # ---- Save outputs ----
    np.save(DATA_PROCESSED / f"{TAG}_per_seed_corrected_oof.npy",
            per_seed_corrected.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy",
            mean_bag_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_median_bag_oof.npy",
            median_bag_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_per_seed_corrected_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_mean_bag_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_median_bag_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "feature_source": f"morgan-{MORGAN_NBITS}_r{MORGAN_RADIUS}",
        "kernel": "tanimoto_precomputed",
        "svr_C_grid": C_GRID,
        "svr_epsilon": SVR_EPSILON,
        "n_unb": n_unb,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "rae_anchor_nb1070": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "kernel_psd_check": psd,
        "kernel_diag_mean": float(np.diag(K).mean()),
        "kernel_offdiag_mean": float(
            (K.sum() - np.trace(K)) / (K.size - K.shape[0])
        ),
        "per_seed_rae": per_seed_rae,
        "per_seed_records": per_seed_records,
        "per_seed_best_Cs": per_seed_best_Cs,
        "best_C_global_majority": best_C_global,
        "best_C_global_counts": {str(k): int(v)
                                 for k, v in global_C_count.items()},
        "rae_per_seed_mean": rae_per_seed_mean,
        "rae_per_seed_median": rae_per_seed_median,
        "rae_per_seed_std": rae_per_seed_std,
        "rae_per_seed_min": rae_per_seed_min,
        "rae_per_seed_max": rae_per_seed_max,
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_nb1070": rae_mean_bag - rae_anchor,
        "delta_mean_bag_vs_nb1183": rae_mean_bag - NB1183_REF,
        "delta_mean_bag_vs_nb1242": rae_mean_bag - NB1242_REF,
        "beats_nb1070": bool(beats_nb1070),
        "beats_nb1183": bool(beats_nb1183),
        "beats_nb1242": bool(beats_nb1242),
        "verdict": verdict,
        "nb1070_ref_pooled": NB1070_REF_POOLED,
        "nb1183_ref": NB1183_REF,
        "nb1242_ref": NB1242_REF,
        "decision_margin": DECISION_MARGIN,
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
    for k in (
        "rae_anchor_nb1070", "per_seed_rae",
        "rae_per_seed_mean", "rae_per_seed_median", "rae_per_seed_std",
        "rae_mean_bag", "rae_median_bag",
        "best_C_global_majority", "best_C_global_counts",
        "delta_mean_bag_vs_nb1070",
        "delta_mean_bag_vs_nb1183",
        "delta_mean_bag_vs_nb1242",
        "beats_nb1070", "beats_nb1183", "beats_nb1242",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
