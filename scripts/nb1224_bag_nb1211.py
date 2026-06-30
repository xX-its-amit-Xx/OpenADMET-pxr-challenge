"""nb1224 -- Outer-bag VALIDATION of nb1211 (the 0.5/0.5 mean blend of nb1190
triple BoB + nb1200 MACCS BoB at pooled RAE 0.5451).

PRECEDENT
---------
nb1190 stress-tested nb1181 (triple naive 1/3 mean of nb1130 / nb1153 / nb1172)
under outer seeds {0, 1, 7, 42, 137} with inner family inner = outer*1000+base,
and confirmed the answer at outer-bag mean ~0.5499 (nb1181 0.5566 was actually
pessimistic).  nb1200 stress-tested nb1183 (MACCS-167 residual mean bag)
under the SAME outer-seed grid and confirmed mean-bag pooled RAE ~0.5495.
nb1211 then took those two BoB rasters and computed the naive 0.5/0.5 mean,
landing at pooled RAE 0.5451.

The question this script answers
--------------------------------
Is the 0.5451 ALSO an artifact of the {0, 1, 7, 42, 137} x outer*1000+base
seed family that BOTH nb1190 and nb1200 used?  To stress-test it, we need
to BUILD a disjoint outer-seed family AND a disjoint inner-seed family, then
re-run the entire nb1190 triple BoB pipeline AND the entire nb1200 MACCS BoB
pipeline, and take the same 0.5/0.5 mean per outer seed.  Distribution of
those per-outer RAEs vs 0.5451 = the reproducibility verdict.

PROTOCOL
--------
NEW OUTER SEEDS  : {2, 13, 99, 271, 500}    (disjoint from nb1190/nb1200's
                                              {0, 1, 7, 42, 137})
NEW INNER FAMILY : inner_seeds(o) = [o*7 + s for s in 0..4]
                                              (disjoint from nb1190/nb1200's
                                              inner = outer*1000 + base, even
                                              if you set base in {0..4})
    => outer 2    -> inner [ 14,  15,  16,  17,  18]
       outer 13   -> inner [ 91,  92,  93,  94,  95]
       outer 99   -> inner [693, 694, 695, 696, 697]
       outer 271  -> inner [1897,1898,1899,1900,1901]
       outer 500  -> inner [3500,3501,3502,3503,3504]

For each NEW outer seed o:
  inner_seeds = [o*7 + s for s in range(5)]
  ---- nb1190 component (triple BoB) ----
    For each of nb1130 (Morgan+RDKit 2265), nb1153 (Mordred 1533),
    nb1172 (AtomPair 2048): build 5-seed mean bag of shallow LGBM Huber
    on residual = y_unb - nb1070_pred_oof, 5-fold KFold cross-fit per
    inner seed.  Mean over 5 inner seeds -> component mean bag.
    Triple naive 1/3 mean (nb1130_o + nb1153_o + nb1172_o) / 3
        -> nb1190_o triple mean OOF (253,).
  ---- nb1200 component (MACCS BoB) ----
    Build 5-seed mean bag of shallow LGBM Huber on residual using
    cached MACCS-167 features.  Mean over 5 inner seeds
        -> nb1200_o mean bag OOF (253,).
  ---- 0.5/0.5 mean (nb1211 blend) ----
    nb1211_o = 0.5 * nb1190_o + 0.5 * nb1200_o
    per-outer-RAE = pooled RAE(y_unb, nb1211_o).

REPORT
------
Per-outer 0.5/0.5 RAE list (5 NEW outer seeds), mean, std, min, max.
Also per-outer nb1190 component RAE and nb1200 component RAE, for
diagnostic.  Bag-of-bags row-level mean and median across the 5 outer-seed
0.5/0.5 predictions.

VERDICT
-------
NB1211_REPRODUCES if abs(per_outer_mean - 0.5451) <= 0.003.
Otherwise: NB1211_PESSIMISTIC (better) or NB1211_LUCKY (pulls up).

Outputs:
  data/processed/nb1224_per_outer_nb1190.npy   (5, 253) float32  triple BoB
                                                                  per NEW outer
  data/processed/nb1224_per_outer_nb1200.npy   (5, 253) float32  MACCS BoB
                                                                  per NEW outer
  data/processed/nb1224_per_outer_blend.npy    (5, 253) float32  0.5/0.5 blend
                                                                  per NEW outer
  data/processed/nb1224_bob_mean_oof.npy       (253,)   float32  row-level
                                                                  mean across 5
  data/processed/nb1224_bob_median_oof.npy     (253,)   float32  row-level
                                                                  median across 5
  data/processed/nb1224_summary.json
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

TAG = "nb1224"
ANCHOR = "nb1070"

# NEW outer / inner seed grid -- disjoint from nb1190/nb1200.
NEW_OUTER_SEEDS = [2, 13, 99, 271, 500]
INNER_OFFSETS = [0, 1, 2, 3, 4]  # inner_seed = outer*7 + offset
RESID_FOLDS = 5

# Cached feature paths.
ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"      # (513, 2048)
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")              # X_mordred_test.npy (513, 1533)
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"            # (513, 167)

# Reference: nb1211's 0.5/0.5 mean OOF (block_mean naive mean RAE) at 0.5451.
NB1211_HALF_HALF_REF = 0.5451
REPRO_MARGIN = 0.003


# -----------------------------------------------------------------------------
# Residual cross-fit primitive (identical capacity to nb1130/53/72/83).
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


# -----------------------------------------------------------------------------
# Feature loaders.
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


def _load_maccs_unblind(unb_idx: np.ndarray, n_test_expected: int) -> np.ndarray:
    if not MACCS_TE_PATH.exists():
        raise FileNotFoundError(f"MACCS test cache missing: {MACCS_TE_PATH}")
    X_te = np.load(MACCS_TE_PATH)
    if X_te.shape[0] != n_test_expected:
        raise ValueError(
            f"MACCS test cache shape mismatch: {X_te.shape} vs n_test={n_test_expected}"
        )
    if X_te.shape[1] not in (166, 167):
        raise ValueError(
            f"MACCS test cache unexpected width: {X_te.shape[1]} (expected 166 or 167)"
        )
    return X_te[unb_idx].astype(np.float32)


# -----------------------------------------------------------------------------
# Main.
# -----------------------------------------------------------------------------
def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- OUTER-BAG validation of nb1211 (0.5/0.5 mean of nb1190 triple BoB +")
    print(f"          nb1200 MACCS BoB)  reference = {NB1211_HALF_HALF_REF:.4f}")
    print(f"          NEW outer seeds  = {NEW_OUTER_SEEDS}   (disjoint from nb1190/nb1200)")
    print(f"          inner family     = inner_seeds(o) = [o*7 + k for k in {INNER_OFFSETS}]")
    print(f"          residual target  = y_unb - {ANCHOR}_pred_oof")
    print(f"          components       = nb1130 (Mor+RDKit 2265) + nb1153 (Mordred 1533)")
    print(f"                              + nb1172 (AtomPair 2048) + nb1183 (MACCS 167)")
    print(f"          LGBM             = depth=3, leaves=7, n_est=80, lr=0.05,")
    print(f"                              min_child_samples=20, obj=huber(alpha=1.0)")
    print(f"          repro margin     = {REPRO_MARGIN:.3f}")
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

    print(f"[feat] computing Morgan+RDKit (combined 2265) on {n_unb} unblind ...")
    X_mr_unb = _load_morgan_rdkit_unb(unb_idx, te_smiles)
    print(f"[feat] X_mr_unb  shape = {X_mr_unb.shape}")
    print(f"[feat] loading cached Mordred (1533) for {n_unb} unblind ...")
    X_mor_unb = _load_mordred_unb(unb_idx, n_test)
    print(f"[feat] X_mor_unb shape = {X_mor_unb.shape}")
    print(f"[feat] loading cached AtomPair (2048) for {n_unb} unblind ...")
    X_ap_unb = _load_atompair_unb(unb_idx, n_test)
    print(f"[feat] X_ap_unb  shape = {X_ap_unb.shape}")
    print(f"[feat] loading cached MACCS-167 for {n_unb} unblind ...")
    X_mc_unb = _load_maccs_unblind(unb_idx, n_test)
    print(f"[feat] X_mc_unb  shape = {X_mc_unb.shape}")

    # ---- Per-outer-seed rebuild ----
    print("\n" + "-" * 78)
    print("PER-OUTER REBUILD  (each: 4 components x 5 inner seeds x 5 folds = "
          f"{4 * len(INNER_OFFSETS) * RESID_FOLDS} LGBM fits)")
    print("-" * 78)

    per_outer_nb1190 = np.zeros((len(NEW_OUTER_SEEDS), n_unb), dtype=np.float64)
    per_outer_nb1200 = np.zeros((len(NEW_OUTER_SEEDS), n_unb), dtype=np.float64)
    per_outer_blend  = np.zeros((len(NEW_OUTER_SEEDS), n_unb), dtype=np.float64)

    per_outer_rae_nb1190: list[float] = []
    per_outer_rae_nb1200: list[float] = []
    per_outer_rae_blend:  list[float] = []
    per_outer_records: list[dict] = []

    for oi, o in enumerate(NEW_OUTER_SEEDS):
        t_outer = time.time()
        inner_seeds = [int(o) * 7 + int(k) for k in INNER_OFFSETS]

        # ---- nb1190 component A: Morgan+RDKit ----
        bagA = np.zeros((len(inner_seeds), n_unb), dtype=np.float64)
        for j, s in enumerate(inner_seeds):
            resid_oof = _residual_cross_fit_one_seed(X_mr_unb, residual, s)
            bagA[j] = anchor_oof + resid_oof
        meanA = bagA.mean(axis=0)
        rae_A = float(rae(y_unb, meanA))

        # ---- nb1190 component B: Mordred ----
        bagB = np.zeros((len(inner_seeds), n_unb), dtype=np.float64)
        for j, s in enumerate(inner_seeds):
            resid_oof = _residual_cross_fit_one_seed(X_mor_unb, residual, s)
            bagB[j] = anchor_oof + resid_oof
        meanB = bagB.mean(axis=0)
        rae_B = float(rae(y_unb, meanB))

        # ---- nb1190 component C: AtomPair ----
        bagC = np.zeros((len(inner_seeds), n_unb), dtype=np.float64)
        for j, s in enumerate(inner_seeds):
            resid_oof = _residual_cross_fit_one_seed(X_ap_unb, residual, s)
            bagC[j] = anchor_oof + resid_oof
        meanC = bagC.mean(axis=0)
        rae_C = float(rae(y_unb, meanC))

        # ---- nb1190 triple naive 1/3 mean ----
        triple_o = (meanA + meanB + meanC) / 3.0
        rae_triple = float(rae(y_unb, triple_o))

        # ---- nb1200 component: MACCS ----
        bagM = np.zeros((len(inner_seeds), n_unb), dtype=np.float64)
        for j, s in enumerate(inner_seeds):
            resid_oof = _residual_cross_fit_one_seed(X_mc_unb, residual, s)
            bagM[j] = anchor_oof + resid_oof
        maccs_o = bagM.mean(axis=0)
        rae_maccs = float(rae(y_unb, maccs_o))

        # ---- nb1211 0.5/0.5 blend ----
        blend_o = 0.5 * triple_o + 0.5 * maccs_o
        rae_blend = float(rae(y_unb, blend_o))

        per_outer_nb1190[oi] = triple_o
        per_outer_nb1200[oi] = maccs_o
        per_outer_blend[oi]  = blend_o
        per_outer_rae_nb1190.append(rae_triple)
        per_outer_rae_nb1200.append(rae_maccs)
        per_outer_rae_blend.append(rae_blend)

        per_outer_records.append({
            "outer_seed": int(o),
            "inner_seeds": inner_seeds,
            "rae_nb1130_mean_bag": rae_A,
            "rae_nb1153_mean_bag": rae_B,
            "rae_nb1172_mean_bag": rae_C,
            "rae_nb1190_triple_mean": rae_triple,
            "rae_nb1200_maccs_mean_bag": rae_maccs,
            "rae_nb1211_half_half": rae_blend,
            "delta_blend_vs_nb1211_ref": rae_blend - NB1211_HALF_HALF_REF,
            "elapsed_sec": round(time.time() - t_outer, 1),
        })
        print(f"   outer {o:5d}  inner={inner_seeds}")
        print(f"     nb1130_o={rae_A:.4f}  nb1153_o={rae_B:.4f}  "
              f"nb1172_o={rae_C:.4f}  triple={rae_triple:.4f}")
        print(f"     maccs_o={rae_maccs:.4f}   "
              f"0.5/0.5 blend = {rae_blend:.4f}   "
              f"(elapsed {time.time() - t_outer:.1f}s)")

    # ---- Aggregate per-outer RAEs ----
    blend_arr = np.array(per_outer_rae_blend)
    nb1190_arr = np.array(per_outer_rae_nb1190)
    nb1200_arr = np.array(per_outer_rae_nb1200)

    outer_mean_blend = float(blend_arr.mean())
    outer_std_blend  = float(blend_arr.std())
    outer_min_blend  = float(blend_arr.min())
    outer_max_blend  = float(blend_arr.max())

    outer_mean_nb1190 = float(nb1190_arr.mean())
    outer_std_nb1190  = float(nb1190_arr.std())
    outer_mean_nb1200 = float(nb1200_arr.mean())
    outer_std_nb1200  = float(nb1200_arr.std())

    # Row-level bag-of-bags across the 5 NEW outer seeds.
    bob_mean_oof = per_outer_blend.mean(axis=0)
    bob_median_oof = np.median(per_outer_blend, axis=0)
    rae_bob_mean = float(rae(y_unb, bob_mean_oof))
    rae_bob_median = float(rae(y_unb, bob_median_oof))

    # Verdict.
    delta = outer_mean_blend - NB1211_HALF_HALF_REF
    reproduces = abs(delta) <= REPRO_MARGIN

    if reproduces:
        verdict = "NB1211_REPRODUCES"
    elif delta < -REPRO_MARGIN:
        verdict = "NB1211_PESSIMISTIC_OUTER_BAG_BETTER"
    else:
        verdict = "NB1211_LUCKY_OUTER_BAG_PULLS_UP"

    print("\n" + "=" * 78)
    print("OUTER-BAG AGGREGATIONS  (NEW outer family, NEW inner family)")
    print("=" * 78)
    print(f"   per-outer 0.5/0.5 blend RAE = "
          f"[{', '.join(f'{r:.4f}' for r in per_outer_rae_blend)}]")
    print(f"   per-outer nb1190 triple RAE = "
          f"[{', '.join(f'{r:.4f}' for r in per_outer_rae_nb1190)}]")
    print(f"   per-outer nb1200 MACCS  RAE = "
          f"[{', '.join(f'{r:.4f}' for r in per_outer_rae_nb1200)}]")
    print(f"")
    print(f"   blend  per-outer mean   = {outer_mean_blend:.4f}")
    print(f"   blend  per-outer std    = {outer_std_blend:.4f}")
    print(f"   blend  per-outer min    = {outer_min_blend:.4f}")
    print(f"   blend  per-outer max    = {outer_max_blend:.4f}")
    print(f"   nb1190 per-outer mean   = {outer_mean_nb1190:.4f}  "
          f"std = {outer_std_nb1190:.4f}")
    print(f"   nb1200 per-outer mean   = {outer_mean_nb1200:.4f}  "
          f"std = {outer_std_nb1200:.4f}")
    print(f"   bag-of-bags MEAN   row-level RAE = {rae_bob_mean:.4f}")
    print(f"   bag-of-bags MEDIAN row-level RAE = {rae_bob_median:.4f}")
    print(f"   nb1211 0.5/0.5 reference        = {NB1211_HALF_HALF_REF:.4f}")
    print(f"   delta(per-outer mean vs nb1211) = {delta:+.4f}  "
          f"(margin {REPRO_MARGIN:.3f})")
    print(f"   VERDICT = {verdict}")

    # ---- Save artefacts ----
    np.save(DATA_PROCESSED / f"{TAG}_per_outer_nb1190.npy",
            per_outer_nb1190.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_per_outer_nb1200.npy",
            per_outer_nb1200.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_per_outer_blend.npy",
            per_outer_blend.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_bob_mean_oof.npy",
            bob_mean_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_bob_median_oof.npy",
            bob_median_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_per_outer_nb1190.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_per_outer_nb1200.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_per_outer_blend.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_bob_mean_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_bob_median_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "n_unb": n_unb,
        "new_outer_seeds": NEW_OUTER_SEEDS,
        "inner_offsets": INNER_OFFSETS,
        "inner_family_rule": "inner = outer * 7 + offset",
        "resid_folds": RESID_FOLDS,
        "rae_anchor_nb1070": rae_anchor,
        "per_outer_records": per_outer_records,
        "per_outer_rae_nb1190_triple": [float(x) for x in per_outer_rae_nb1190],
        "per_outer_rae_nb1200_maccs":  [float(x) for x in per_outer_rae_nb1200],
        "per_outer_rae_blend_half_half": [float(x) for x in per_outer_rae_blend],
        "outer_mean_blend": outer_mean_blend,
        "outer_std_blend":  outer_std_blend,
        "outer_min_blend":  outer_min_blend,
        "outer_max_blend":  outer_max_blend,
        "outer_mean_nb1190": outer_mean_nb1190,
        "outer_std_nb1190":  outer_std_nb1190,
        "outer_mean_nb1200": outer_mean_nb1200,
        "outer_std_nb1200":  outer_std_nb1200,
        "rae_bag_of_bags_mean":   rae_bob_mean,
        "rae_bag_of_bags_median": rae_bob_median,
        "nb1211_half_half_ref": NB1211_HALF_HALF_REF,
        "delta_outer_mean_vs_nb1211": delta,
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
    for k in ("rae_anchor_nb1070",
              "per_outer_rae_blend_half_half",
              "per_outer_rae_nb1190_triple",
              "per_outer_rae_nb1200_maccs",
              "outer_mean_blend", "outer_std_blend",
              "outer_min_blend", "outer_max_blend",
              "outer_mean_nb1190", "outer_mean_nb1200",
              "rae_bag_of_bags_mean", "rae_bag_of_bags_median",
              "delta_outer_mean_vs_nb1211",
              "reproduces", "verdict"):
        print(f"  {k}: {res.get(k)}")
