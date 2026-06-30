"""nb1104 -- Big bag K=28 with 50 seeds (variance reduction only).

HYPOTHESIS:
    nb2103 K=28 5-seed bag (seeds {0, 1, 7, 42, 137}) hit mean-bag RAE 0.4737 /
    median-bag RAE 0.4698 -- the best result in the entire fine SHAP K-grid
    sweep, narrowly beating nb2081 K=30 mean-bag 0.4788 by -0.0051.

    With only 5 seeds the pooled estimate's standard error is non-trivial:
    per-seed std was 0.00960, so the standard error of a 5-seed mean is
    ~0.0043 -- about the same magnitude as the apparent edge over K=30.

    This notebook isolates the VARIANCE-REDUCTION effect by re-running the
    EXACT same LGBM(MSE) cross-fit on the EXACT same X_unb_28 feature matrix,
    using 50 random seeds (0..49) instead of 5.  No feature, no hyperparam, no
    K change.  The only difference is bag size.

PROTOCOL:
    1. Load cached X_unb_28 from nb2103 (data/processed/X_unb_28_nb2103.npy).
    2. Load chemprop_aux te[unb_idx] anchor and y_unb truth.
    3. For seed in [0..49]: LGBM(MSE) 5-fold KFold cross-fit, predict the
       residual y_unb - anchor, anchor + resid_oof = corrected prediction.
    4. Pool 50 per-seed corrected predictions: mean-bag and median-bag.
    5. Compare vs nb2103 K=28 5-seed bag (mean 0.4737, median 0.4698) at
       decision margin 0.003.
    6. Report per-seed RAE distribution: mean, median, std, min, max,
       histogram bins; and bootstrap SE of the 50-seed pooled estimates.
    7. If passes, run FRESH SEEDS [100..149] to verify (no peeking, no
       gaming the seed pool).

OUTPUTS:
    scripts/nb1104_big_bag.py
    data/processed/nb1104_summary.json
    data/processed/nb1104_mean_bag_oof.npy        (253,)  -- 50-seed mean
    data/processed/nb1104_median_bag_oof.npy      (253,)  -- 50-seed median
    data/processed/nb1104_per_seed_oof.npy        (50, 253) -- all per-seed
    data/processed/nb1104_fresh_mean_bag_oof.npy  (if FRESH check fires)
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
import lightgbm as lgb

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1104"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
UNB_IDX_PATH = DATA_PROCESSED / "_audit_unblind_idx.npy"
UNB_Y_PATH = DATA_PROCESSED / "_audit_unblind_y.npy"
X_UNB_28_PATH = DATA_PROCESSED / "X_unb_28_nb2103.npy"
NB2103_SUMMARY = DATA_PROCESSED / "nb2103_summary.json"

RESID_FOLDS = 5
SEEDS_MAIN = list(range(50))           # 0..49 -- main 50-seed bag
SEEDS_FRESH = list(range(100, 150))    # 100..149 -- fresh verification
DECISION_MARGIN = 0.003

# References
CHEMPROP_AUX_REF = 0.6216
NB2103_K28_MEAN_BAG_REF = 0.4737
NB2103_K28_MEDIAN_BAG_REF = 0.4698


def _lgbm_params(seed: int) -> dict:
    """LGBM(MSE) -- identical to nb2063/nb2081/nb2091/nb2103."""
    return dict(
        objective="regression",
        max_depth=4,
        num_leaves=15,
        n_estimators=300,
        learning_rate=0.03,
        min_child_samples=5,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _residual_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray,
                                 seed: int) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _run_bag(X: np.ndarray, residual: np.ndarray, anchor: np.ndarray,
             y_unb: np.ndarray, seeds: list[int], tag: str) -> dict:
    n_unb = len(y_unb)
    n_seeds = len(seeds)
    per_seed_corrected = np.zeros((n_seeds, n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []
    t_bag = time.time()
    for i, s in enumerate(seeds):
        ts = time.time()
        resid_oof_s = _residual_cross_fit_one_seed(X, residual, s)
        pred_corr_s = anchor + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
            "wall_sec": round(time.time() - ts, 3),
        })
        if (i + 1) % 10 == 0 or i == 0:
            print(f"   [{tag}] {i + 1:3d}/{n_seeds}  seed={s:3d}  "
                  f"rae={rae_s:.4f}  wall={time.time() - ts:.2f}s")

    # Pooled metrics
    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))

    per_seed_arr = np.array(per_seed_rae)

    # Bootstrap SE on pooled RAE (resample seeds, recompute pooled)
    rng = np.random.default_rng(20260608)
    n_boot = 1000
    boot_mean_rae = np.empty(n_boot, dtype=np.float64)
    boot_median_rae = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sel = rng.integers(0, n_seeds, size=n_seeds)
        bag_mean_b = per_seed_corrected[sel].mean(axis=0)
        bag_median_b = np.median(per_seed_corrected[sel], axis=0)
        boot_mean_rae[b] = rae(y_unb, bag_mean_b)
        boot_median_rae[b] = rae(y_unb, bag_median_b)

    return {
        "tag": tag,
        "seeds": [int(s) for s in seeds],
        "n_seeds": int(n_seeds),
        "per_seed_records": per_seed_records,
        "per_seed_rae": per_seed_rae,
        "rae_per_seed_mean": float(per_seed_arr.mean()),
        "rae_per_seed_median": float(np.median(per_seed_arr)),
        "rae_per_seed_std": float(per_seed_arr.std()),
        "rae_per_seed_min": float(per_seed_arr.min()),
        "rae_per_seed_max": float(per_seed_arr.max()),
        "rae_per_seed_p25": float(np.percentile(per_seed_arr, 25)),
        "rae_per_seed_p75": float(np.percentile(per_seed_arr, 75)),
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "bootstrap_n": int(n_boot),
        "rae_mean_bag_boot_se": float(boot_mean_rae.std()),
        "rae_mean_bag_boot_p025": float(np.percentile(boot_mean_rae, 2.5)),
        "rae_mean_bag_boot_p975": float(np.percentile(boot_mean_rae, 97.5)),
        "rae_median_bag_boot_se": float(boot_median_rae.std()),
        "rae_median_bag_boot_p025": float(np.percentile(boot_median_rae, 2.5)),
        "rae_median_bag_boot_p975": float(np.percentile(boot_median_rae, 97.5)),
        "wall_sec": round(time.time() - t_bag, 2),
        "per_seed_corrected": per_seed_corrected,   # not JSON-serialized
        "mean_bag_oof": mean_bag_oof,
        "median_bag_oof": median_bag_oof,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- BIG BAG K=28 with 50 seeds (variance reduction only)")
    print(f"          anchor={ANCHOR}  seeds=0..49  folds={RESID_FOLDS}")
    print(f"          ref: nb2103 K=28 5-seed mean-bag = "
          f"{NB2103_K28_MEAN_BAG_REF:.4f}  median-bag = "
          f"{NB2103_K28_MEDIAN_BAG_REF:.4f}")
    print(f"          margin = {DECISION_MARGIN}")
    print("=" * 78)

    # ---- Pre-flight: cached inputs ----
    for p in (X_UNB_28_PATH, UNB_IDX_PATH, UNB_Y_PATH, ANCHOR_TE_PATH,
              NB2103_SUMMARY):
        if not p.exists():
            raise FileNotFoundError(f"missing cache: {p}")

    X_unb_28 = np.load(X_UNB_28_PATH).astype(np.float32)
    unb_idx = np.load(UNB_IDX_PATH)
    y_unb = np.load(UNB_Y_PATH).astype(np.float64)
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    residual = y_unb - anchor
    n_unb = len(y_unb)

    print(f"[load] X_unb_28 = {X_unb_28.shape}  y_unb = {y_unb.shape}")
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # nb2103 K=28 reference (from cached summary)
    with open(NB2103_SUMMARY) as f:
        nb2103_sum = json.load(f)
    nb2103_k28 = None
    for r in nb2103_sum.get("per_K_records", []):
        if int(r.get("K", -1)) == 28:
            nb2103_k28 = r
            break
    if nb2103_k28 is None:
        raise KeyError("nb2103 K=28 record not found")
    nb2103_k28_mean_bag = float(nb2103_k28["rae_mean_bag"])
    nb2103_k28_median_bag = float(nb2103_k28["rae_median_bag"])
    nb2103_k28_per_seed = nb2103_k28["per_seed_rae"]
    nb2103_k28_per_seed_std = float(nb2103_k28["rae_per_seed_std"])
    print(f"[ref] nb2103 K=28 mean_bag   = {nb2103_k28_mean_bag:.4f}")
    print(f"[ref] nb2103 K=28 median_bag = {nb2103_k28_median_bag:.4f}")
    print(f"[ref] nb2103 K=28 per_seed   = "
          f"[{', '.join(f'{r:.4f}' for r in nb2103_k28_per_seed)}]")
    print(f"[ref] nb2103 K=28 per_seed_std = {nb2103_k28_per_seed_std:.4f}  "
          f"(SE of 5-seed mean ~= {nb2103_k28_per_seed_std / np.sqrt(5):.4f})")

    # ---- MAIN 50-seed bag (seeds 0..49) ----
    print("\n" + "-" * 78)
    print(f"MAIN BAG: 50 seeds [0..49]  K=28  KFold={RESID_FOLDS}")
    print("-" * 78)
    main_res = _run_bag(X_unb_28, residual, anchor, y_unb,
                        SEEDS_MAIN, tag="MAIN")
    print(f"\n   [MAIN] per-seed RAE: mean={main_res['rae_per_seed_mean']:.4f}  "
          f"median={main_res['rae_per_seed_median']:.4f}  "
          f"std={main_res['rae_per_seed_std']:.4f}  "
          f"min={main_res['rae_per_seed_min']:.4f}  "
          f"max={main_res['rae_per_seed_max']:.4f}")
    print(f"   [MAIN] pooled mean-bag   RAE = {main_res['rae_mean_bag']:.4f}  "
          f"(SE_boot = {main_res['rae_mean_bag_boot_se']:.4f}  "
          f"95% CI [{main_res['rae_mean_bag_boot_p025']:.4f}, "
          f"{main_res['rae_mean_bag_boot_p975']:.4f}])")
    print(f"   [MAIN] pooled median-bag RAE = "
          f"{main_res['rae_median_bag']:.4f}  "
          f"(SE_boot = {main_res['rae_median_bag_boot_se']:.4f}  "
          f"95% CI [{main_res['rae_median_bag_boot_p025']:.4f}, "
          f"{main_res['rae_median_bag_boot_p975']:.4f}])")
    print(f"   [MAIN] wall = {main_res['wall_sec']:.1f}s")

    delta_main_mean_vs_nb2103 = main_res["rae_mean_bag"] - nb2103_k28_mean_bag
    delta_main_median_vs_nb2103 = (
        main_res["rae_median_bag"] - nb2103_k28_median_bag
    )
    delta_main_mean_vs_anchor = main_res["rae_mean_bag"] - rae_anchor
    delta_main_median_vs_anchor = main_res["rae_median_bag"] - rae_anchor

    print(f"   [MAIN] d_mean_vs_nb2103_5seed   = {delta_main_mean_vs_nb2103:+.4f}")
    print(f"   [MAIN] d_median_vs_nb2103_5seed = {delta_main_median_vs_nb2103:+.4f}")
    print(f"   [MAIN] d_mean_vs_anchor          = {delta_main_mean_vs_anchor:+.4f}")
    print(f"   [MAIN] d_median_vs_anchor        = {delta_main_median_vs_anchor:+.4f}")

    # Verdict on mean-bag
    if main_res["rae_mean_bag"] < nb2103_k28_mean_bag - DECISION_MARGIN:
        verdict_mean = "BIG_BAG_MEAN_BEATS_NB2103_5SEED_MEAN"
    elif abs(delta_main_mean_vs_nb2103) < DECISION_MARGIN:
        verdict_mean = "BIG_BAG_MEAN_FLAT_VS_NB2103_5SEED_MEAN"
    else:
        verdict_mean = "BIG_BAG_MEAN_WORSE_THAN_NB2103_5SEED_MEAN"
    print(f"   [MAIN] verdict_mean   = {verdict_mean}")

    if main_res["rae_median_bag"] < nb2103_k28_median_bag - DECISION_MARGIN:
        verdict_median = "BIG_BAG_MEDIAN_BEATS_NB2103_5SEED_MEDIAN"
    elif abs(delta_main_median_vs_nb2103) < DECISION_MARGIN:
        verdict_median = "BIG_BAG_MEDIAN_FLAT_VS_NB2103_5SEED_MEDIAN"
    else:
        verdict_median = "BIG_BAG_MEDIAN_WORSE_THAN_NB2103_5SEED_MEDIAN"
    print(f"   [MAIN] verdict_median = {verdict_median}")

    # ---- Histogram of per-seed RAE (text bars) ----
    per_seed_arr = np.array(main_res["per_seed_rae"])
    hist, edges = np.histogram(per_seed_arr, bins=10)
    print(f"\n   [MAIN] per-seed RAE histogram (n=50):")
    for i in range(len(hist)):
        bar = "#" * int(hist[i])
        print(f"      {edges[i]:.4f}-{edges[i+1]:.4f}  {hist[i]:2d}  {bar}")

    # ---- Save MAIN outputs ----
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy",
            main_res["mean_bag_oof"].astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_median_bag_oof.npy",
            main_res["median_bag_oof"].astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_per_seed_oof.npy",
            main_res["per_seed_corrected"].astype(np.float32))
    print(f"\n   [save] {DATA_PROCESSED / (TAG + '_mean_bag_oof.npy')}")
    print(f"   [save] {DATA_PROCESSED / (TAG + '_median_bag_oof.npy')}")
    print(f"   [save] {DATA_PROCESSED / (TAG + '_per_seed_oof.npy')}")

    # ---- FRESH 50-seed bag (seeds 100..149) -- only if MAIN passes ----
    fresh_fired = False
    fresh_res_serial = None
    fresh_pass = None
    if (verdict_mean == "BIG_BAG_MEAN_BEATS_NB2103_5SEED_MEAN"
            or verdict_median == "BIG_BAG_MEDIAN_BEATS_NB2103_5SEED_MEDIAN"):
        fresh_fired = True
        print("\n" + "-" * 78)
        print(f"FRESH BAG: 50 seeds [100..149]  (verification, no peeking)")
        print("-" * 78)
        fresh_res = _run_bag(X_unb_28, residual, anchor, y_unb,
                             SEEDS_FRESH, tag="FRESH")
        print(f"\n   [FRESH] per-seed RAE: "
              f"mean={fresh_res['rae_per_seed_mean']:.4f}  "
              f"median={fresh_res['rae_per_seed_median']:.4f}  "
              f"std={fresh_res['rae_per_seed_std']:.4f}  "
              f"min={fresh_res['rae_per_seed_min']:.4f}  "
              f"max={fresh_res['rae_per_seed_max']:.4f}")
        print(f"   [FRESH] pooled mean-bag   RAE = "
              f"{fresh_res['rae_mean_bag']:.4f}")
        print(f"   [FRESH] pooled median-bag RAE = "
              f"{fresh_res['rae_median_bag']:.4f}")
        np.save(DATA_PROCESSED / f"{TAG}_fresh_mean_bag_oof.npy",
                fresh_res["mean_bag_oof"].astype(np.float32))
        np.save(DATA_PROCESSED / f"{TAG}_fresh_median_bag_oof.npy",
                fresh_res["median_bag_oof"].astype(np.float32))
        print(f"   [save] {DATA_PROCESSED / (TAG + '_fresh_mean_bag_oof.npy')}")

        # Cross-check: do MAIN and FRESH agree within bootstrap CI?
        delta_fresh_mean = fresh_res["rae_mean_bag"] - main_res["rae_mean_bag"]
        delta_fresh_median = (
            fresh_res["rae_median_bag"] - main_res["rae_median_bag"]
        )
        within_se = (
            abs(delta_fresh_mean) < 2.0 * main_res["rae_mean_bag_boot_se"]
        )
        print(f"   [FRESH] d_mean_vs_MAIN   = {delta_fresh_mean:+.4f}  "
              f"(MAIN SE_boot = {main_res['rae_mean_bag_boot_se']:.4f})  "
              f"within_2_SE = {within_se}")
        print(f"   [FRESH] d_median_vs_MAIN = {delta_fresh_median:+.4f}")
        fresh_pass = bool(within_se) and (
            fresh_res["rae_mean_bag"] < nb2103_k28_mean_bag - DECISION_MARGIN
            or fresh_res["rae_median_bag"]
            < nb2103_k28_median_bag - DECISION_MARGIN
        )
        print(f"   [FRESH] fresh_pass = {fresh_pass}")

        # Strip arrays for JSON
        fresh_res_serial = {k: v for k, v in fresh_res.items()
                            if k not in ("per_seed_corrected",
                                         "mean_bag_oof",
                                         "median_bag_oof")}
        fresh_res_serial["delta_mean_vs_MAIN"] = delta_fresh_mean
        fresh_res_serial["delta_median_vs_MAIN"] = delta_fresh_median
        fresh_res_serial["within_2_SE_of_MAIN"] = bool(within_se)
    else:
        print("\n   [SKIP] MAIN did not pass margin -- skipping FRESH "
              "verification")

    # ---- Global verdict ----
    if (verdict_mean == "BIG_BAG_MEAN_BEATS_NB2103_5SEED_MEAN"
            and fresh_fired and fresh_pass):
        global_verdict = "BIG_BAG_50SEED_BEATS_5SEED_AND_FRESH_CONFIRMS"
    elif verdict_mean == "BIG_BAG_MEAN_BEATS_NB2103_5SEED_MEAN":
        global_verdict = ("BIG_BAG_50SEED_BEATS_5SEED_MAIN_ONLY"
                          if not fresh_fired else
                          "BIG_BAG_50SEED_BEATS_5SEED_MAIN_FRESH_FAILS")
    elif verdict_median == "BIG_BAG_MEDIAN_BEATS_NB2103_5SEED_MEDIAN":
        global_verdict = "BIG_BAG_50SEED_BEATS_5SEED_ON_MEDIAN_ONLY"
    elif (abs(delta_main_mean_vs_nb2103) < DECISION_MARGIN
          and abs(delta_main_median_vs_nb2103) < DECISION_MARGIN):
        global_verdict = "BIG_BAG_50SEED_FLAT_VS_5SEED_VARIANCE_NOT_ANSWER"
    else:
        global_verdict = "BIG_BAG_50SEED_WORSE_OR_INCONCLUSIVE"

    print("\n" + "=" * 78)
    print(f"GLOBAL VERDICT: {global_verdict}")
    print("=" * 78)

    # ---- Save summary ----
    main_serial = {k: v for k, v in main_res.items()
                   if k not in ("per_seed_corrected",
                                "mean_bag_oof",
                                "median_bag_oof")}
    summary = {
        "tag": TAG,
        "method": ("lgbm_mse_K28_big_bag_50_seeds_variance_reduction_"
                   "on_nb2103_X_unb_28_cache"),
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "X_unb_28_path": str(X_UNB_28_PATH),
        "K": 28,
        "feat_dim": int(X_unb_28.shape[1]),
        "n_unb": int(n_unb),
        "n_seeds_main": int(len(SEEDS_MAIN)),
        "n_seeds_fresh": int(len(SEEDS_FRESH)) if fresh_fired else 0,
        "seeds_main_first": SEEDS_MAIN[0],
        "seeds_main_last": SEEDS_MAIN[-1],
        "seeds_fresh_first": SEEDS_FRESH[0],
        "seeds_fresh_last": SEEDS_FRESH[-1],
        "resid_folds": RESID_FOLDS,
        "lgbm_objective": "regression",
        "lgbm_max_depth": 4,
        "lgbm_num_leaves": 15,
        "lgbm_n_estimators": 300,
        "lgbm_learning_rate": 0.03,
        "lgbm_min_child_samples": 5,
        "lgbm_reg_lambda": 2.0,
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "nb2103_K28_mean_bag_ref": nb2103_k28_mean_bag,
        "nb2103_K28_median_bag_ref": nb2103_k28_median_bag,
        "nb2103_K28_per_seed_ref": nb2103_k28_per_seed,
        "nb2103_K28_per_seed_std_ref": nb2103_k28_per_seed_std,
        "nb2103_K28_5seed_mean_SE_approx": float(
            nb2103_k28_per_seed_std / np.sqrt(5)
        ),
        "main": main_serial,
        "fresh": fresh_res_serial,
        "delta_main_mean_vs_nb2103": delta_main_mean_vs_nb2103,
        "delta_main_median_vs_nb2103": delta_main_median_vs_nb2103,
        "delta_main_mean_vs_anchor": delta_main_mean_vs_anchor,
        "delta_main_median_vs_anchor": delta_main_median_vs_anchor,
        "verdict_mean": verdict_mean,
        "verdict_median": verdict_median,
        "fresh_fired": bool(fresh_fired),
        "fresh_pass": fresh_pass,
        "global_verdict": global_verdict,
        "pre_unblind_clean": True,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "nb2103_K28_5seed_ref_mean_bag": NB2103_K28_MEAN_BAG_REF,
        "nb2103_K28_5seed_ref_median_bag": NB2103_K28_MEDIAN_BAG_REF,
        "decision_margin": DECISION_MARGIN,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "K", "feat_dim", "n_seeds_main", "n_seeds_fresh",
        "rae_anchor_chemprop_aux",
        "nb2103_K28_mean_bag_ref",
        "nb2103_K28_median_bag_ref",
        "nb2103_K28_per_seed_std_ref",
        "nb2103_K28_5seed_mean_SE_approx",
    ):
        print(f"  {k}: {res.get(k)}")
    print("\n==== MAIN BAG (50 seeds) ====")
    m = res["main"]
    print(f"  per_seed_mean   = {m['rae_per_seed_mean']:.4f}")
    print(f"  per_seed_median = {m['rae_per_seed_median']:.4f}")
    print(f"  per_seed_std    = {m['rae_per_seed_std']:.4f}")
    print(f"  per_seed_min    = {m['rae_per_seed_min']:.4f}")
    print(f"  per_seed_max    = {m['rae_per_seed_max']:.4f}")
    print(f"  per_seed_p25    = {m['rae_per_seed_p25']:.4f}")
    print(f"  per_seed_p75    = {m['rae_per_seed_p75']:.4f}")
    print(f"  pooled mean_bag   = {m['rae_mean_bag']:.4f}  "
          f"SE_boot = {m['rae_mean_bag_boot_se']:.4f}  "
          f"95% CI [{m['rae_mean_bag_boot_p025']:.4f}, "
          f"{m['rae_mean_bag_boot_p975']:.4f}]")
    print(f"  pooled median_bag = {m['rae_median_bag']:.4f}  "
          f"SE_boot = {m['rae_median_bag_boot_se']:.4f}  "
          f"95% CI [{m['rae_median_bag_boot_p025']:.4f}, "
          f"{m['rae_median_bag_boot_p975']:.4f}]")
    print(f"  d_mean_vs_nb2103_5seed   = {res['delta_main_mean_vs_nb2103']:+.4f}")
    print(f"  d_median_vs_nb2103_5seed = {res['delta_main_median_vs_nb2103']:+.4f}")
    print(f"  verdict_mean   = {res['verdict_mean']}")
    print(f"  verdict_median = {res['verdict_median']}")
    if res.get("fresh") is not None:
        f50 = res["fresh"]
        print("\n==== FRESH BAG (50 seeds, 100..149) ====")
        print(f"  per_seed_mean   = {f50['rae_per_seed_mean']:.4f}")
        print(f"  per_seed_std    = {f50['rae_per_seed_std']:.4f}")
        print(f"  pooled mean_bag   = {f50['rae_mean_bag']:.4f}  "
              f"SE_boot = {f50['rae_mean_bag_boot_se']:.4f}")
        print(f"  pooled median_bag = {f50['rae_median_bag']:.4f}")
        print(f"  d_mean_vs_MAIN   = {f50['delta_mean_vs_MAIN']:+.4f}")
        print(f"  d_median_vs_MAIN = {f50['delta_median_vs_MAIN']:+.4f}")
        print(f"  within_2_SE_of_MAIN = {f50['within_2_SE_of_MAIN']}")
        print(f"  fresh_pass = {res.get('fresh_pass')}")
    print(f"\nGLOBAL VERDICT: {res['global_verdict']}")
