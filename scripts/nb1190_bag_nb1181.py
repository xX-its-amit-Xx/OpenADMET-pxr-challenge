"""nb1190 -- Outer-bag VALIDATION of nb1181 (triple naive 1/3 mean).

PRECEDENT
---------
nb1143 (0.5649) looked great but outer-bag nb1151 pulled the answer to 0.5705.
nb1181's triple naive-mean cross-fit RAE = 0.5566 (nb1181_summary.json).  This
script stress-tests it by REBUILDING each of the three components
(nb1130, nb1153, nb1172) under five outer KFold-seed regimes and measuring the
distribution of the triple mean RAE.

PROTOCOL
--------
For each OUTER seed o in {0, 1, 7, 42, 137}:
  inner_seeds = [o * 1000 + s for s in {0, 1, 7, 42, 137}]
    => outer 0    -> inner [   0,    1,    7,   42,  137]   (reproduces nb1181 exactly)
       outer 1    -> inner [1000, 1001, 1007, 1042, 1137]
       outer 7    -> inner [7000, 7001, 7007, 7042, 7137]
       outer 42   -> inner [42000,42001,42007,42042,42137]
       outer 137  -> inner [137000,137001,137007,137042,137137]
  Rebuild three components on the 253 unblind with these inner seeds:
    * nb1130_o : shallow LGBM Huber on Morgan+RDKit (combined, 2265) residual over nb1070.
    * nb1153_o : shallow LGBM Huber on cached Mordred (1533)         residual over nb1070.
    * nb1172_o : shallow LGBM Huber on cached AtomPair (2048)        residual over nb1070.
  For each inner seed in the bag, run 5-fold KFold cross-fit residual OOF,
  then mean over the 5 inner seeds -> the o-th mean_bag for each component.
  Triple naive 1/3 mean = (nb1130_o + nb1153_o + nb1172_o) / 3.
  Pooled RAE(triple_mean) is the outer-seed measurement.

SANITY CHECK
------------
Outer seed 0 should reproduce nb1181's 0.5566 within rounding (uses the same
inner seeds nb1181 used).

REPORT
------
Per-outer RAE list, mean, std, min, max.  Row-level bag-of-bags mean and median
across the 5 outer-seed predictions.  Verdict NB1181_REPRODUCES if the per-outer
mean is within 0.003 of 0.5566.

Outputs:
  data/processed/nb1190_per_outer_oof.npy    (5, 253) float32  triple-mean per outer seed
  data/processed/nb1190_bob_mean_oof.npy     (253,)   float32  bag-of-bags mean
  data/processed/nb1190_bob_median_oof.npy   (253,)   float32  bag-of-bags median
  data/processed/nb1190_summary.json
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
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor

from pxr.data import load_test
from pxr.eval import rae
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED

TAG = "nb1190"
ANCHOR = "nb1070"

# Outer / inner seed grid.
OUTER_SEEDS = [0, 1, 7, 42, 137]
INNER_BASES = [0, 1, 7, 42, 137]   # inner_seed = outer * 1000 + base
RESID_FOLDS = 5

# Cached feature caches (verified to exist by prelaunch check).
ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"      # (513, 2048) uint8
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")              # X_mordred_test.npy (513, 1533)

# Reference (nb1181 naive 1/3 mean RAE on 253 unblind).
NB1181_NAIVE_MEAN_REF = 0.5566
REPRO_MARGIN = 0.003


# -----------------------------------------------------------------------------
# Residual cross-fit primitive (identical capacity to nb1130 / nb1153 / nb1172).
# -----------------------------------------------------------------------------
def _lgbm_params(seed: int) -> dict:
    return dict(
        objective="huber",
        alpha=1.0,
        learning_rate=0.05,
        n_estimators=80,
        max_depth=3,
        num_leaves=7,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        verbosity=-1,
        random_state=seed,
        n_jobs=2,
    )


def _residual_cross_fit_one_seed(
    X: np.ndarray, residual: np.ndarray, seed: int
) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _mean_bag_for_outer(
    X: np.ndarray, residual: np.ndarray, anchor_oof: np.ndarray,
    inner_seeds: list[int], label: str
) -> tuple[np.ndarray, list[float]]:
    """Rebuild a 5-seed mean bag of shallow LGBM Huber on residual.

    Returns (mean_bag_corrected_oof_253, per_inner_seed_rae_list).
    """
    n = len(residual)
    bag = np.zeros((len(inner_seeds), n), dtype=np.float64)
    rae_list: list[float] = []
    # truth is implicit via residual = y - anchor; compute proxy RAE on
    # the corrected prediction (anchor + resid).  We don't need y here
    # because we'll get the final per-outer triple RAE downstream against
    # y_unb.  The per-inner-seed RAE is reported only for diagnostics --
    # so accept y as a global rebound below.
    return bag, rae_list   # placeholder, replaced inside main()


# -----------------------------------------------------------------------------
# Feature loaders for the three orthogonal channels.
# -----------------------------------------------------------------------------
def _load_morgan_rdkit_unb(unb_idx: np.ndarray, te_smiles: np.ndarray) -> np.ndarray:
    smi_unb = te_smiles[unb_idx].tolist()
    X = impute(combined(smi_unb))
    return X


def _load_mordred_unb(unb_idx: np.ndarray, n_test_expected: int) -> np.ndarray:
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(f"Mordred cache missing -- run nb1030 first ({mte_p})")
    X_te = np.load(mte_p).astype(np.float32)
    if X_te.shape[0] != n_test_expected:
        raise ValueError(
            f"Mordred test shape mismatch: {X_te.shape} vs n_test={n_test_expected}"
        )
    X_te = np.where(np.isfinite(X_te), X_te, np.nan).astype(np.float32)
    col_med = np.nanmedian(X_te, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X_te)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X_te[idx_r, idx_c] = col_med[idx_c]
    return X_te[unb_idx]


def _load_atompair_unb(unb_idx: np.ndarray, n_test_expected: int) -> np.ndarray:
    if not ATOMPAIR_TE_PATH.exists():
        raise FileNotFoundError(
            f"AtomPair test cache missing: {ATOMPAIR_TE_PATH}"
        )
    X_te = np.load(ATOMPAIR_TE_PATH)
    if X_te.shape[0] != n_test_expected or X_te.shape[1] != 2048:
        raise ValueError(
            f"AtomPair test cache shape mismatch: {X_te.shape}"
        )
    return X_te[unb_idx].astype(np.float32)


# -----------------------------------------------------------------------------
# Main.
# -----------------------------------------------------------------------------
def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- OUTER-BAG validation of nb1181 (triple 1/3 mean) "
          f"under {len(OUTER_SEEDS)} outer seeds")
    print(f"          outer seeds = {OUTER_SEEDS}")
    print(f"          inner bases = {INNER_BASES}  (inner = outer*1000 + base)")
    print(f"          residual target = y_unb - nb1070_pred_oof")
    print(f"          components per outer = "
          f"nb1130(Morgan+RDKit 2265) + nb1153(Mordred 1533) + nb1172(AtomPair 2048)")
    print(f"          LGBM: depth=3, num_leaves=7, n_est=80, lr=0.05, "
          f"min_child_samples=20, obj=huber(alpha=1.0)")
    print(f"          reference (nb1181 naive 1/3 mean) = "
          f"{NB1181_NAIVE_MEAN_REF:.4f}   margin = {REPRO_MARGIN:.3f}")
    print("=" * 78)

    # ---- Load truth + anchor + features (once) ----
    te = load_test()
    n_test = len(te)
    te_smiles = te["smiles"].values
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    anchor_path = DATA_PROCESSED / f"{ANCHOR}_pred_oof.npy"
    if not anchor_path.exists():
        raise FileNotFoundError(f"{anchor_path} missing")
    anchor_oof = np.load(anchor_path).astype(np.float64)
    if anchor_oof.shape[0] != n_unb:
        raise ValueError(
            f"{anchor_path} shape mismatch: {anchor_oof.shape} vs n_unb={n_unb}"
        )
    rae_anchor = float(rae(y_unb, anchor_oof))
    residual = y_unb - anchor_oof
    print(f"[load] {ANCHOR}_pred_oof.npy  pooled RAE = {rae_anchor:.4f}")
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # Features (compute / load once).
    print(f"[feat] computing Morgan+RDKit (combined 2265) on {n_unb} unblind ...")
    X_mr_unb = _load_morgan_rdkit_unb(unb_idx, te_smiles)
    print(f"[feat] X_mr_unb shape = {X_mr_unb.shape}")
    print(f"[feat] loading cached Mordred (1533) for {n_unb} unblind ...")
    X_mor_unb = _load_mordred_unb(unb_idx, n_test)
    print(f"[feat] X_mor_unb shape = {X_mor_unb.shape}")
    print(f"[feat] loading cached AtomPair (2048) for {n_unb} unblind ...")
    X_ap_unb = _load_atompair_unb(unb_idx, n_test)
    print(f"[feat] X_ap_unb shape = {X_ap_unb.shape}")

    # ---- Per-outer-seed rebuild ----
    print("\n" + "-" * 78)
    print("PER-OUTER REBUILD  (each: 3 components x 5 inner seeds x 5 folds = "
          f"{3 * len(INNER_BASES) * RESID_FOLDS} LGBM fits)")
    print("-" * 78)

    per_outer_triple = np.zeros((len(OUTER_SEEDS), n_unb), dtype=np.float64)
    per_outer_rae: list[float] = []
    per_outer_records: list[dict] = []

    for oi, o in enumerate(OUTER_SEEDS):
        t_outer = time.time()
        inner_seeds = [int(o) * 1000 + int(b) for b in INNER_BASES]
        # For outer seed 0, this should give [0, 1, 7, 42, 137] -- nb1181 reproduction.

        # ---- Component A: nb1130 (Morgan+RDKit) ----
        bagA = np.zeros((len(inner_seeds), n_unb), dtype=np.float64)
        for j, s in enumerate(inner_seeds):
            resid_oof = _residual_cross_fit_one_seed(X_mr_unb, residual, s)
            bagA[j] = anchor_oof + resid_oof
        meanA = bagA.mean(axis=0)
        rae_A = float(rae(y_unb, meanA))

        # ---- Component B: nb1153 (Mordred) ----
        bagB = np.zeros((len(inner_seeds), n_unb), dtype=np.float64)
        for j, s in enumerate(inner_seeds):
            resid_oof = _residual_cross_fit_one_seed(X_mor_unb, residual, s)
            bagB[j] = anchor_oof + resid_oof
        meanB = bagB.mean(axis=0)
        rae_B = float(rae(y_unb, meanB))

        # ---- Component C: nb1172 (AtomPair) ----
        bagC = np.zeros((len(inner_seeds), n_unb), dtype=np.float64)
        for j, s in enumerate(inner_seeds):
            resid_oof = _residual_cross_fit_one_seed(X_ap_unb, residual, s)
            bagC[j] = anchor_oof + resid_oof
        meanC = bagC.mean(axis=0)
        rae_C = float(rae(y_unb, meanC))

        # ---- Triple naive 1/3 mean ----
        triple = (meanA + meanB + meanC) / 3.0
        rae_triple = float(rae(y_unb, triple))
        per_outer_triple[oi] = triple
        per_outer_rae.append(rae_triple)

        # Did the outer-seed=0 case reproduce nb1181?
        nb1181_match_note = ""
        if o == 0:
            d = rae_triple - NB1181_NAIVE_MEAN_REF
            nb1181_match_note = (
                f"  [REPRO_CHECK outer=0 vs nb1181 0.5566: d = {d:+.4f}, "
                f"|d|<0.003? {abs(d) < 0.003}]"
            )

        per_outer_records.append({
            "outer_seed": int(o),
            "inner_seeds": inner_seeds,
            "rae_nb1130_mean_bag": rae_A,
            "rae_nb1153_mean_bag": rae_B,
            "rae_nb1172_mean_bag": rae_C,
            "rae_triple_mean": rae_triple,
            "delta_vs_nb1181_ref": rae_triple - NB1181_NAIVE_MEAN_REF,
            "elapsed_sec": round(time.time() - t_outer, 1),
        })
        print(f"   outer {o:5d}  inner={inner_seeds}")
        print(f"     nb1130_o = {rae_A:.4f}  nb1153_o = {rae_B:.4f}  "
              f"nb1172_o = {rae_C:.4f}  triple_mean = {rae_triple:.4f}  "
              f"(elapsed {time.time() - t_outer:.1f}s){nb1181_match_note}")

    # ---- Aggregate across outer seeds ----
    per_outer_rae_arr = np.array(per_outer_rae)
    outer_mean = float(per_outer_rae_arr.mean())
    outer_std = float(per_outer_rae_arr.std())
    outer_min = float(per_outer_rae_arr.min())
    outer_max = float(per_outer_rae_arr.max())

    # Row-level bag-of-bags: mean and median across outer-seed predictions.
    bob_mean_oof = per_outer_triple.mean(axis=0)
    bob_median_oof = np.median(per_outer_triple, axis=0)
    rae_bob_mean = float(rae(y_unb, bob_mean_oof))
    rae_bob_median = float(rae(y_unb, bob_median_oof))

    # Verdict.
    reproduces = abs(outer_mean - NB1181_NAIVE_MEAN_REF) <= REPRO_MARGIN
    outer0 = per_outer_rae_arr[0]
    outer0_reproduces = abs(outer0 - NB1181_NAIVE_MEAN_REF) <= REPRO_MARGIN

    if reproduces:
        verdict = "NB1181_REPRODUCES"
    elif outer_mean < NB1181_NAIVE_MEAN_REF - REPRO_MARGIN:
        verdict = "NB1181_PESSIMISTIC_OUTER_BAG_BETTER"
    else:
        verdict = "NB1181_OPTIMISTIC_OUTER_BAG_PULLS_UP"

    print("\n" + "=" * 78)
    print("OUTER-BAG AGGREGATIONS")
    print("=" * 78)
    print(f"   per-outer triple-mean RAE list = "
          f"[{', '.join(f'{r:.4f}' for r in per_outer_rae)}]")
    print(f"   per-outer mean   = {outer_mean:.4f}")
    print(f"   per-outer std    = {outer_std:.4f}")
    print(f"   per-outer min    = {outer_min:.4f}")
    print(f"   per-outer max    = {outer_max:.4f}")
    print(f"   bag-of-bags MEAN   row-level RAE = {rae_bob_mean:.4f}")
    print(f"   bag-of-bags MEDIAN row-level RAE = {rae_bob_median:.4f}")
    print(f"   nb1181 naive-mean reference     = {NB1181_NAIVE_MEAN_REF:.4f}")
    print(f"   delta(per-outer mean vs nb1181) = "
          f"{outer_mean - NB1181_NAIVE_MEAN_REF:+.4f}  (margin {REPRO_MARGIN:.3f})")
    print(f"   outer-seed=0 reproduces?  {outer0_reproduces}  "
          f"(outer0 RAE = {outer0:.4f}, d = {outer0 - NB1181_NAIVE_MEAN_REF:+.4f})")
    print(f"   VERDICT = {verdict}")

    # ---- Save artefacts ----
    np.save(DATA_PROCESSED / f"{TAG}_per_outer_oof.npy",
            per_outer_triple.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_bob_mean_oof.npy",
            bob_mean_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_bob_median_oof.npy",
            bob_median_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_per_outer_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_bob_mean_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_bob_median_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "n_unb": n_unb,
        "outer_seeds": OUTER_SEEDS,
        "inner_bases": INNER_BASES,
        "resid_folds": RESID_FOLDS,
        "rae_anchor_nb1070": rae_anchor,
        "per_outer_records": per_outer_records,
        "per_outer_rae": [float(x) for x in per_outer_rae],
        "outer_mean": outer_mean,
        "outer_std": outer_std,
        "outer_min": outer_min,
        "outer_max": outer_max,
        "rae_bag_of_bags_mean": rae_bob_mean,
        "rae_bag_of_bags_median": rae_bob_median,
        "nb1181_naive_mean_ref": NB1181_NAIVE_MEAN_REF,
        "delta_outer_mean_vs_nb1181": outer_mean - NB1181_NAIVE_MEAN_REF,
        "delta_outer0_vs_nb1181": float(outer0 - NB1181_NAIVE_MEAN_REF),
        "outer0_reproduces": bool(outer0_reproduces),
        "repro_margin": REPRO_MARGIN,
        "reproduces": bool(reproduces),
        "verdict": verdict,
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
    for k in ("rae_anchor_nb1070", "per_outer_rae",
              "outer_mean", "outer_std", "outer_min", "outer_max",
              "rae_bag_of_bags_mean", "rae_bag_of_bags_median",
              "delta_outer_mean_vs_nb1181", "delta_outer0_vs_nb1181",
              "outer0_reproduces", "reproduces", "verdict"):
        print(f"  {k}: {res.get(k)}")
