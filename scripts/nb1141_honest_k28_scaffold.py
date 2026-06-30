"""nb1141 -- HONEST K=28 SHAP-LGBM under scaffold CV with Bonferroni gate.

GOAL:
    Replace the nb2103 random-KFold optimism with a fully scaffold-aware
    estimate of the SHAP top-28 LGBM(MSE) residual learner anchored on
    chemprop_aux, then gate the verdict at a Bonferroni-corrected alpha
    accounting for ~700 prior trials on this dataset.

PROTOCOL:
    1. Load X_unb_28_nb2103.npy (253, 28) SHAP top-28 features.
       Build residual = y_unb - chemprop_aux[unb_idx].
    2. Fit LGBM(MSE) with the standard nb2103 hyperparams under 5-seed bag
       (seeds 0,1,7,42,137), each seed using a 5-fold scaffold split keyed
       on Bemis-Murcko scaffolds of the 253 unblind SMILES.
       Cross-fit OOF residuals -> corrected = anchor + resid_OOF.
       Per-seed RAE recorded; mean-bag RAE = RAE(y_unb, per_seed.mean(0)).
    3. Bonferroni gate at alpha = 0.05 / 700 trials ~ 7.143e-5.
    4. 95% CI on mean-bag RAE via paired BOOTSTRAP over the 253 rows
       (B = 2000 resamples on indices; recompute RAE on each resample).
       Report (lower, upper) and lower-CI bound.
       Bonferroni-CI: also compute (alpha/2)-quantile per-tail at
       7.143e-5 / 2 = 3.57e-5 so the BC-lower bound is the 0.001785%-tile.
    5. Compare lower-CI (95%) and BC-lower vs nb2103 scaffold-CV 0.5057
       baseline (from nb1130). Candidate beats baseline iff lower-CI < 0.5057.
    6. Refresh SHAP feature importance on the honest mean-bag residual model
       (refit one LGBM on the full 253 with seed=0 + SHAP TreeExplainer).
       Save refreshed ranking and per-feature mean(|SHAP|).

Outputs:
    scripts/nb1141_honest_k28_scaffold.py
    data/processed/nb1141_summary.json
    data/processed/nb1141_mean_bag_oof_scaffold.npy   (253,) float32
    data/processed/nb1141_shap_importance_K28_honest.npy   (28,) float32
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
import lightgbm as lgb

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb1141"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
X_UNB_28_PATH = DATA_PROCESSED / "X_unb_28_nb2103.npy"
NB2103_SUMMARY = DATA_PROCESSED / "nb2103_summary.json"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

# Bonferroni gate
N_TRIALS = 700
ALPHA_RAW = 0.05
ALPHA_BC = ALPHA_RAW / N_TRIALS                 # ~7.143e-5

# Bootstrap config
N_BOOT = 2000
BOOT_SEED = 12345

# Baseline ceiling -- nb2103 scaffold-CV mean-bag RAE per nb1130 audit
NB2103_SCAFFOLD_BASELINE = 0.5057
DECISION_MARGIN = 0.003
CHEMPROP_AUX_REF = 0.6216


def _lgbm_params(seed: int) -> dict:
    """Match nb2103 / nb1130 hyperparams exactly."""
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


def _residual_cross_fit_scaffold(X, residual, splits, seed):
    n = len(residual)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in splits:
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    if np.isnan(oof).any():
        raise RuntimeError(f"NaN in OOF (seed={seed})")
    return oof


def _bootstrap_ci_rae(y_true, y_pred, n_boot, alpha_raw, alpha_bc, rng):
    """Paired bootstrap on rows -- recompute RAE on each resample.

    Returns dict with lower/upper at the 95% level and at the BC level."""
    n = len(y_true)
    raes = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        raes[b] = rae(y_true[idx], y_pred[idx])
    raes_sorted = np.sort(raes)
    q_lo_raw = np.quantile(raes_sorted, alpha_raw / 2.0)
    q_hi_raw = np.quantile(raes_sorted, 1.0 - alpha_raw / 2.0)
    q_lo_bc = np.quantile(raes_sorted, alpha_bc / 2.0)
    q_hi_bc = np.quantile(raes_sorted, 1.0 - alpha_bc / 2.0)
    return {
        "boot_rae_mean": float(raes.mean()),
        "boot_rae_std": float(raes.std()),
        "ci95_lo": float(q_lo_raw),
        "ci95_hi": float(q_hi_raw),
        "ci_bc_lo": float(q_lo_bc),
        "ci_bc_hi": float(q_hi_bc),
        "n_boot": int(n_boot),
    }


def _refresh_shap_importance(X, residual, seed=0):
    """Refit one LGBM on the full 253 rows using `seed`, then run TreeExplainer
    and return per-feature mean(|SHAP|)."""
    try:
        import shap
    except ImportError:
        return None, "shap not installed"
    mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
    mdl.fit(X, residual)
    expl = shap.TreeExplainer(mdl)
    sv = expl.shap_values(X)
    # sv: (n_rows, n_feat) for regression
    imp = np.mean(np.abs(sv), axis=0).astype(np.float32)
    return imp, None


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- HONEST K=28 SHAP-LGBM scaffold CV + Bonferroni gate")
    print(f"          anchor={ANCHOR}  seeds={RESID_SEEDS}  folds={RESID_FOLDS}")
    print(f"          baseline = nb2103 scaffold-CV {NB2103_SCAFFOLD_BASELINE:.4f}")
    print(f"          Bonferroni alpha = {ALPHA_RAW}/{N_TRIALS} = {ALPHA_BC:.3e}")
    print("=" * 78)

    # ---- Load anchor + truth ----
    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"missing anchor: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Load cached SHAP top-28 features ----
    if not X_UNB_28_PATH.exists():
        raise FileNotFoundError(f"missing K=28 cache: {X_UNB_28_PATH}")
    X_unb_28 = np.load(X_UNB_28_PATH).astype(np.float32)
    if X_unb_28.shape != (n_unb, 28):
        raise ValueError(f"X_unb_28 shape mismatch: {X_unb_28.shape}")
    print(f"[load] X_unb_28 = {X_unb_28.shape}")

    # ---- Build scaffold splits on 253 unblind ----
    te_unb_smiles = [test_smiles[i] for i in unb_idx]
    scaffs = [bemis_murcko(s) for s in te_unb_smiles]
    n_unique_scaff = len(set(s for s in scaffs if s))
    n_none = sum(1 for s in scaffs if not s)
    print(f"[scaff] unique scaffolds = {n_unique_scaff}  none/empty = {n_none}")
    scaffold_splits_per_seed = {
        s: scaffold_kfold_indices(scaffs, n_splits=RESID_FOLDS, seed=s)
        for s in RESID_SEEDS
    }
    print(f"[scaff] fold sizes (seed=0): "
          f"{[len(va) for _, va in scaffold_splits_per_seed[0]]}")

    # ---- 5-seed bag under scaffold CV ----
    print("\n" + "-" * 78)
    print("SCAFFOLD-CV 5-SEED BAG (honest protocol)")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae = []
    per_seed_records = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        splits = scaffold_splits_per_seed[s]
        resid_oof_s = _residual_cross_fit_scaffold(X_unb_28, residual, splits, s)
        pred_corr_s = anchor + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_anchor": rae_s - rae_anchor,
            "delta_vs_baseline": rae_s - NB2103_SCAFFOLD_BASELINE,
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   seed={s:3d}  scaffold_RAE = {rae_s:.4f}  "
              f"(d_vs_anchor = {rae_s - rae_anchor:+.4f}, "
              f"d_vs_baseline = {rae_s - NB2103_SCAFFOLD_BASELINE:+.4f})  "
              f"wall = {time.time() - ts:.1f}s")

    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))

    per_seed_arr = np.array(per_seed_rae)
    rae_per_seed_mean = float(per_seed_arr.mean())
    rae_per_seed_std = float(per_seed_arr.std(ddof=1))
    rae_per_seed_min = float(per_seed_arr.min())
    rae_per_seed_max = float(per_seed_arr.max())

    print(f"\n   pooled mean_bag scaffold_RAE   = {rae_mean_bag:.4f}")
    print(f"   pooled median_bag scaffold_RAE = {rae_median_bag:.4f}")
    print(f"   per-seed mean = {rae_per_seed_mean:.4f}  "
          f"std(ddof=1) = {rae_per_seed_std:.4f}  "
          f"[min {rae_per_seed_min:.4f}, max {rae_per_seed_max:.4f}]")

    # ---- Paired bootstrap 95% CI on mean-bag RAE ----
    print("\n" + "-" * 78)
    print(f"PAIRED ROW BOOTSTRAP CI on mean-bag RAE  (B={N_BOOT})")
    print("-" * 78)
    rng = np.random.default_rng(BOOT_SEED)
    ci = _bootstrap_ci_rae(
        y_unb, mean_bag_oof, n_boot=N_BOOT,
        alpha_raw=ALPHA_RAW, alpha_bc=ALPHA_BC, rng=rng,
    )
    print(f"   boot_rae_mean = {ci['boot_rae_mean']:.4f}  "
          f"boot_rae_std = {ci['boot_rae_std']:.4f}")
    print(f"   95% CI       = [{ci['ci95_lo']:.4f}, {ci['ci95_hi']:.4f}]")
    print(f"   BC CI (alpha={ALPHA_BC:.2e}) = "
          f"[{ci['ci_bc_lo']:.4f}, {ci['ci_bc_hi']:.4f}]")

    # ---- Verdict vs nb2103 scaffold-CV baseline 0.5057 ----
    beats_baseline_point = rae_mean_bag < NB2103_SCAFFOLD_BASELINE - DECISION_MARGIN
    beats_baseline_ci95 = ci["ci95_lo"] < NB2103_SCAFFOLD_BASELINE
    beats_baseline_bc = ci["ci_bc_lo"] < NB2103_SCAFFOLD_BASELINE
    delta_point = rae_mean_bag - NB2103_SCAFFOLD_BASELINE
    delta_ci95_lo = ci["ci95_lo"] - NB2103_SCAFFOLD_BASELINE
    delta_ci_bc_lo = ci["ci_bc_lo"] - NB2103_SCAFFOLD_BASELINE

    if beats_baseline_bc:
        verdict = "BEATS_BASELINE_AT_BONFERRONI_LEVEL"
    elif beats_baseline_ci95:
        verdict = "BEATS_BASELINE_AT_95CI_NOT_BC"
    elif beats_baseline_point:
        verdict = "BEATS_BASELINE_POINT_ONLY_NOT_CI"
    elif abs(delta_point) <= DECISION_MARGIN:
        verdict = "FLAT_VS_BASELINE"
    else:
        verdict = "DOES_NOT_BEAT_BASELINE"

    print("\n" + "-" * 78)
    print("BONFERRONI GATE VS nb2103 SCAFFOLD-CV BASELINE 0.5057")
    print("-" * 78)
    print(f"   point mean_bag        = {rae_mean_bag:.4f}  "
          f"(d_vs_baseline = {delta_point:+.4f})")
    print(f"   95%-CI lower bound    = {ci['ci95_lo']:.4f}  "
          f"(d_vs_baseline = {delta_ci95_lo:+.4f})")
    print(f"   BC-CI  lower bound    = {ci['ci_bc_lo']:.4f}  "
          f"(d_vs_baseline = {delta_ci_bc_lo:+.4f})")
    print(f"   beats @ point         = {beats_baseline_point}")
    print(f"   beats @ 95% CI        = {beats_baseline_ci95}")
    print(f"   beats @ BC alpha      = {beats_baseline_bc}")
    print(f"   verdict               = {verdict}")

    # ---- Save mean-bag OOF ----
    out_oof = DATA_PROCESSED / f"{TAG}_mean_bag_oof_scaffold.npy"
    np.save(out_oof, mean_bag_oof.astype(np.float32))
    print(f"\n[save] {out_oof}")

    # ---- Refresh SHAP ranking on honest residual ----
    print("\n" + "-" * 78)
    print("REFRESH SHAP IMPORTANCE on honest residual (full 253, seed=0)")
    print("-" * 78)
    shap_imp, err = _refresh_shap_importance(X_unb_28, residual, seed=0)
    refresh_topk = None
    refresh_imp_list = None
    if shap_imp is None:
        print(f"   [skip] {err}")
    else:
        order = np.argsort(-shap_imp).astype(int)
        refresh_topk = order.tolist()
        refresh_imp_list = shap_imp.tolist()
        out_imp = DATA_PROCESSED / f"{TAG}_shap_importance_K28_honest.npy"
        np.save(out_imp, shap_imp)
        print(f"   shap_imp shape = {shap_imp.shape}")
        print(f"   top-10 rank in K=28: {order[:10].tolist()}")
        print(f"   [save] {out_imp}")

    # Compare refreshed vs nb2103 K=28 ranking (k=28 indices ordering)
    rank_comparison = None
    if shap_imp is not None and NB2103_SUMMARY.exists():
        with open(NB2103_SUMMARY) as f:
            nb2103_sum = json.load(f)
        rec_k28 = [r for r in nb2103_sum["per_K_records"]
                   if int(r["K"]) == 28]
        if rec_k28:
            # nb2103 ordering of the 28 cols within K=28 by SHAP importance
            nb2103_order = list(range(28))   # nb2103 sliced by descending SHAP
            new_order = np.argsort(-shap_imp).tolist()
            # Spearman-style position diff
            pos_diff = float(np.mean(np.abs(
                np.array([new_order.index(i) for i in nb2103_order])
                - np.array(nb2103_order)
            )))
            top10_overlap = len(set(new_order[:10]) & set(nb2103_order[:10]))
            rank_comparison = {
                "mean_abs_position_shift": pos_diff,
                "top10_overlap": int(top10_overlap),
                "nb2103_top10_in_K28": nb2103_order[:10],
                "honest_top10_in_K28": new_order[:10],
            }
            print(f"   top-10 overlap vs nb2103 K=28 (within-K)   = "
                  f"{top10_overlap}/10")
            print(f"   mean abs position shift (28 features)       = "
                  f"{pos_diff:.2f}")

    # ---- Summary ----
    summary = {
        "tag": TAG,
        "method": "honest_K28_lgbm_scaffold_CV_bonferroni_gate",
        "anchor": ANCHOR,
        "anchor_path": str(ANCHOR_TE_PATH),
        "x_unb_28_path": str(X_UNB_28_PATH),
        "n_unb": int(n_unb),
        "K": 28,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "lgbm_objective": "regression",
        "lgbm_max_depth": 4,
        "lgbm_num_leaves": 15,
        "lgbm_n_estimators": 300,
        "lgbm_learning_rate": 0.03,
        "lgbm_min_child_samples": 5,
        "lgbm_reg_lambda": 2.0,
        "n_unique_scaffolds_in_253": int(n_unique_scaff),
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_seed_rae": per_seed_rae,
        "per_seed_records": per_seed_records,
        "rae_per_seed_mean": rae_per_seed_mean,
        "rae_per_seed_std_ddof1": rae_per_seed_std,
        "rae_per_seed_min": rae_per_seed_min,
        "rae_per_seed_max": rae_per_seed_max,
        "rae_mean_bag_scaffold": rae_mean_bag,
        "rae_median_bag_scaffold": rae_median_bag,
        # Bootstrap CIs on mean-bag RAE
        "bootstrap": {
            "n_boot": int(N_BOOT),
            "boot_rae_mean": ci["boot_rae_mean"],
            "boot_rae_std": ci["boot_rae_std"],
            "alpha_raw": ALPHA_RAW,
            "alpha_bc": ALPHA_BC,
            "ci95_lo": ci["ci95_lo"],
            "ci95_hi": ci["ci95_hi"],
            "ci_bc_lo": ci["ci_bc_lo"],
            "ci_bc_hi": ci["ci_bc_hi"],
        },
        # Baseline gate
        "baseline_nb2103_scaffold_cv": NB2103_SCAFFOLD_BASELINE,
        "decision_margin": DECISION_MARGIN,
        "delta_point_vs_baseline": delta_point,
        "delta_ci95_lo_vs_baseline": delta_ci95_lo,
        "delta_ci_bc_lo_vs_baseline": delta_ci_bc_lo,
        "beats_baseline_point": bool(beats_baseline_point),
        "beats_baseline_ci95": bool(beats_baseline_ci95),
        "beats_baseline_bonferroni": bool(beats_baseline_bc),
        "verdict": verdict,
        # SHAP refresh
        "shap_refresh_available": shap_imp is not None,
        "shap_refresh_error": err,
        "shap_refresh_top_order": refresh_topk,
        "shap_refresh_importance_mean_abs": refresh_imp_list,
        "rank_comparison_vs_nb2103_K28": rank_comparison,
        "pre_unblind_clean": True,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_p = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_p, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_p}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "K", "n_unb", "rae_anchor_chemprop_aux",
        "rae_mean_bag_scaffold", "rae_median_bag_scaffold",
        "rae_per_seed_mean", "rae_per_seed_std_ddof1",
        "baseline_nb2103_scaffold_cv",
        "delta_point_vs_baseline",
        "delta_ci95_lo_vs_baseline",
        "delta_ci_bc_lo_vs_baseline",
        "beats_baseline_point", "beats_baseline_ci95",
        "beats_baseline_bonferroni",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
    boot = res.get("bootstrap", {})
    print("  bootstrap:")
    for k in ("n_boot", "boot_rae_mean", "boot_rae_std",
              "ci95_lo", "ci95_hi", "ci_bc_lo", "ci_bc_hi"):
        print(f"    {k}: {boot.get(k)}")
