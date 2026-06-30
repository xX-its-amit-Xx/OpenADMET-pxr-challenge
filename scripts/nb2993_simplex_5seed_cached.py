"""nb2993 -- Per-fold SLSQP simplex on {K18, K20} using ORIGINAL 5-seed cached OOFs.

NEW PARADIGM (sensitivity test of nb2982 / nb2973):
    nb2982 reached pooled outer-val RAE 0.4505 using nb2960 deep-30
    fresh-seed K-anchor OOFs in a per-fold SLSQP simplex over
    {K18, K20}.  This script asks the orthogonal question:
        is the 0.4505 result a property of the *deep-30 averaging*,
        or does it survive on the ORIGINAL 5-seed mean-bag cached
        OOFs that nb2604 (K18) and nb2240 (K20) produced?

    Predictions:
        - If 5-seed mean-bag K-anchors recover the same band
          (< 0.4505), the per-fold simplex transfer is robust to
          OOF noise level -- the win is structural, not from
          deep-30 averaging.
        - If 5-seed under-performs by > +0.003 RAE, the deep-30
          averaging itself was load-bearing and any future K-anchor
          must be re-bagged to 30 seeds before per-fold blending.

PROTOCOL:
    Anchors (BOTH from cached 5-seed mean-bag):
        K18 -> nb2604_mean_bag_oof_K18.npy + te_nb2604_K18.npy
        K20 -> nb2240_mean_bag_oof_K20.npy + te_nb2240_K20.npy
    Outer CV: 5-fold scaffold split, 15 fresh kf_seeds {1021..1035}.
    Per fold:
        - SLSQP minimize fold-train RAE on simplex (w >= 0, sum w = 1)
        - 8 multi-starts (uniform + 7 Dirichlet draws)
        - Apply per-fold weights to held-out fold-val slice
    Pooled outer-val RAE per kf_seed, then mean across 15 kf_seeds =
    sensitivity-test gate metric.

    Deploy:
        - Refit SLSQP on FULL 253 -> single weight vector
        - Apply to (513, 2) stacked te arrays -> te_nb2993

GATE:
    mean_rae < 0.4505 -> "MATCHES_DEEP30"
    mean_rae < 0.4535 -> "BETTER_THAN_NB2973"
    else              -> "FAIL_SENSITIVITY"

References:
    nb2982 deep-30 K18+K20 per-fold simplex pooled    = 0.4505
    nb2973 deep-30 4-K per-fold simplex pooled        = 0.4535 (close to)
    nb2604 5-seed equal-weight 4-K reference          ~0.46x
    nb2240 5-seed K20 standalone                      ~0.50x
    nb2171 5-anchor pyramid deep-30 ceiling           = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb2604_mean_bag_oof_K18.npy
    data/processed/te_nb2604_K18.npy
    data/processed/nb2240_mean_bag_oof_K20.npy
    data/processed/te_nb2240_K20.npy

Outputs:
    data/processed/nb2993_summary.json
    data/processed/nb2993_pred_oof.npy   (253,) float32 -- per-fold simplex OOF
                                                            from best kf_seed
    data/processed/te_nb2993.npy         (513,) float32 -- deploy te
    submissions/nb2993_simplex_5seed_cached.csv  (only if verdict != FAIL_*)
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

TAG = "nb2993"
PARENT_TAG = "nb2982"

# -- Inputs (5-seed cached) ----------------------------------------------------
K_LABELS = ["K18", "K20"]
OOF_PATHS = {
    "K18": DATA_PROCESSED / "nb2604_mean_bag_oof_K18.npy",
    "K20": DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy",
}
TE_PATHS = {
    "K18": DATA_PROCESSED / "te_nb2604_K18.npy",
    "K20": DATA_PROCESSED / "te_nb2240_K20.npy",
}

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1021, 1036))  # 15 fresh seeds {1021..1035}
N_STARTS_FOLD = 8
N_STARTS_FULL = 12
DEGEN_MAX_W = 0.85

# -- Gates ---------------------------------------------------------------------
GATE_MATCHES_DEEP30 = 0.4505
GATE_BETTER_THAN_NB2973 = 0.4535

# -- References ----------------------------------------------------------------
REF_NB2982_DEEP30 = 0.4505
REF_NB2973_DEEP30 = 0.4535
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
    print(f"{TAG} -- per-fold SLSQP simplex on {K_LABELS} (5-seed cached OOFs)")
    print(f"          sensitivity test of {PARENT_TAG} deep-30 result 0.4505")
    print(f"          outer CV: {N_FOLDS}-fold scaffold, "
          f"{len(KF_SEEDS)} kf_seeds {KF_SEEDS[0]}..{KF_SEEDS[-1]}")
    print(f"          per fold: SLSQP simplex w (sum=1, w>=0), {N_STARTS_FOLD} starts")
    print(f"          gate: <{GATE_MATCHES_DEEP30} MATCHES_DEEP30 / "
          f"<{GATE_BETTER_THAN_NB2973} BETTER_THAN_NB2973")
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

    # -- Load 5-seed cached K-anchor OOFs + te arrays -------------------------
    print("\n" + "-" * 78)
    print("STEP 1: load 5-seed cached K-anchor OOFs and te arrays")
    print("-" * 78)
    oof_cols, te_cols = [], []
    per_K_full_rae = {}
    src_tags = {"K18": "nb2604", "K20": "nb2240"}
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
        print(f"   {k} ({src_tags[k]} 5-seed): oof_RAE = {r:.4f}  "
              f"te_mean={te_arr.mean():.3f}  te_std={te_arr.std():.3f}")

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

    # -- Per-kf_seed: per-fold SLSQP simplex ----------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 3: outer CV with per-fold SLSQP simplex over {len(KF_SEEDS)} kf_seeds")
    print("-" * 78)

    seed_records = []
    pooled_per_seed = []
    best_seed_oof = None
    best_seed_pooled = np.inf
    best_seed = None
    any_seed_degenerate = False

    for kf_seed in KF_SEEDS:
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
        fold_records = []
        fold_w_list = []
        seed_has_degen = False
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
            degen = bool(w.max() > DEGEN_MAX_W)
            seed_has_degen = seed_has_degen or degen
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

        if np.isnan(oof_blend).any():
            raise RuntimeError(
                f"kf_seed={kf_seed}: scaffold splits did not cover all 253 rows"
            )
        pooled_rae = float(rae(y_unb, oof_blend))
        pooled_per_seed.append(pooled_rae)
        per_fold_val = [r["val_rae"] for r in fold_records]
        w_stack = np.stack(fold_w_list, axis=0)  # (5, 2)
        w_mean = w_stack.mean(axis=0)
        w_mean = w_mean / w_mean.sum()

        seed_records.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": round(pooled_rae, 4),
            "per_fold_mean": round(float(np.mean(per_fold_val)), 4),
            "per_fold_std": round(float(np.std(per_fold_val, ddof=1)), 4),
            "mean_w": {K_LABELS[k]: round(float(w_mean[k]), 4) for k in range(K)},
            "any_degenerate": seed_has_degen,
            "fold_records": fold_records,
        })
        any_seed_degenerate = any_seed_degenerate or seed_has_degen

        if pooled_rae < best_seed_pooled:
            best_seed_pooled = pooled_rae
            best_seed_oof = oof_blend.copy()
            best_seed = int(kf_seed)

        print(f"   kf_seed={kf_seed}: pooled={pooled_rae:.4f}  "
              f"mean_w=[{', '.join(f'{K_LABELS[k]}={w_mean[k]:.3f}' for k in range(K))}]"
              f"  degen={seed_has_degen}")

    pooled_per_seed = np.asarray(pooled_per_seed, dtype=np.float64)
    mean_rae = float(pooled_per_seed.mean())
    std_rae = float(pooled_per_seed.std(ddof=1))
    min_rae = float(pooled_per_seed.min())
    max_rae = float(pooled_per_seed.max())
    print(f"\n   mean over {len(KF_SEEDS)} kf_seeds: pooled_rae = "
          f"{mean_rae:.4f} +/- {std_rae:.4f}  (min={min_rae:.4f}, max={max_rae:.4f})")

    # -- Deploy: SLSQP on FULL 253 -> single weight vector --------------------
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
    print(f"   te(full-pool) mean={te_pred.mean():.3f} std={te_pred.std():.3f} "
          f"in-sample unb RAE={te_unb_in_rae:.4f}")

    # -- Gate -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 5: GATE (sensitivity)")
    print("-" * 78)
    if mean_rae < GATE_MATCHES_DEEP30:
        verdict = "MATCHES_DEEP30"
    elif mean_rae < GATE_BETTER_THAN_NB2973:
        verdict = "BETTER_THAN_NB2973"
    else:
        verdict = "FAIL_SENSITIVITY"
    delta_vs_nb2982 = mean_rae - REF_NB2982_DEEP30
    delta_vs_nb2973 = mean_rae - REF_NB2973_DEEP30
    delta_vs_nb2171 = mean_rae - REF_NB2171
    print(f"   mean_rae                 = {mean_rae:.4f}")
    print(f"   delta vs nb2982 (0.4505) = {delta_vs_nb2982:+.4f}")
    print(f"   delta vs nb2973 (0.4535) = {delta_vs_nb2973:+.4f}")
    print(f"   delta vs nb2171 (0.4682) = {delta_vs_nb2171:+.4f}")
    print(f"   verdict                  = {verdict}")

    # -- Save artifacts -------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 6: save artifacts")
    print("-" * 78)
    # pred_oof = best single-kf_seed OOF (for downstream tools that expect
    # a single 253-vector); also store full per-seed pooled in summary
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, best_seed_oof.astype(np.float32))
    np.save(te_path, te_pred)
    print(f"   [save] {oof_path}  (best kf_seed={best_seed}, pooled={best_seed_pooled:.4f})")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_simplex_5seed_cached.csv"
    if verdict != "FAIL_SENSITIVITY":
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
        "method": "per_fold_slsqp_simplex_K18_K20_5seed_cached_sensitivity",
        "paradigm": "5seed_cached_anchors_vs_nb2982_deep30",
        "anchor_pool": K_LABELS,
        "anchor_sources": src_tags,
        "anchor_pre_unblind": True,
        "per_K_full_oof_rae": per_K_full_rae,
        "anchor_leak_eq_truth_frac": leak_flags,
        "oof_corr_matrix": corr_mat.tolist(),
        "oof_corr_labels": K_LABELS,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_kf_seeds": len(KF_SEEDS),
        "n_starts_fold": N_STARTS_FOLD,
        "n_starts_full": N_STARTS_FULL,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "K_anchors": K,
        "seed_records": seed_records,
        "pooled_per_seed": [round(float(x), 4) for x in pooled_per_seed.tolist()],
        "mean_rae": mean_rae,
        "std_rae": std_rae,
        "min_rae": min_rae,
        "max_rae": max_rae,
        "best_kf_seed": best_seed,
        "best_kf_seed_pooled": round(float(best_seed_pooled), 4),
        "any_seed_degenerate": any_seed_degenerate,
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
        "submission_csv": str(sub_csv) if verdict != "FAIL_SENSITIVITY" else None,
        "ref_nb2982_deep30": REF_NB2982_DEEP30,
        "ref_nb2973_deep30": REF_NB2973_DEEP30,
        "ref_nb2171": REF_NB2171,
        "delta_vs_nb2982": delta_vs_nb2982,
        "delta_vs_nb2973": delta_vs_nb2973,
        "delta_vs_nb2171": delta_vs_nb2171,
        "gate_matches_deep30": GATE_MATCHES_DEEP30,
        "gate_better_than_nb2973": GATE_BETTER_THAN_NB2973,
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
    print(f"   mean pooled RAE (15 sd)  = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   range                    = [{min_rae:.4f}, {max_rae:.4f}]")
    print(f"   best kf_seed             = {best_seed} ({best_seed_pooled:.4f})")
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
        "mean_rae",
        "std_rae",
        "min_rae",
        "max_rae",
        "best_kf_seed",
        "best_kf_seed_pooled",
        "full_pool_slsqp",
        "te_unb_in_sample_rae_full_pool",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
    print(f"  per_K_full_oof_rae: {res.get('per_K_full_oof_rae')}")
