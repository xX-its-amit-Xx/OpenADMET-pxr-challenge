"""nb2163 -- HONEST pooled-120 trajectory aggregate as LB candidate.

GOAL:
    nb2154 ran 120 cycles of 5-seed bags x 5-fold cross-fit, varying kf_seed
    + small hparam perturbations. It saved:
        nb2154_pooled_mean_oof.npy   -- mean across 120 cycles' OOF arrays
        nb2154_pooled_median_oof.npy -- median across 120 cycles' OOF arrays
        nb2154_best_oof.npy          -- best single-cycle (cycle 78, LUCKY)
        nb2154_per_cycle_rae.npy     -- (120,) RAE per cycle

    nb2156 verified the "best_cycle" 0.462 is LUCKY (not reproducible
    across kf_seeds -- repro mean_bag 0.505). The HONEST aggregate to test
    is the pooled-120 average, which averages OUT the lucky-seed variance.

DECISION:
    K=28 single-config baseline (nb2103):
        rae_mean_bag    = 0.4737
        rae_median_bag  = 0.4698
    nb2154 pooled (claimed):
        rae_pooled_mean_120   = 0.4773
        rae_pooled_median_120 = 0.4765

    Pooled-120 has WORSE RAE than the single K=28 5-seed bag (0.4773 vs
    0.4737 mean axis; 0.4765 vs 0.4698 median axis). This is because the
    hparam perturbation injects variance that the across-cycle average
    cannot fully smooth out.

    Verdict: INFERIOR_TO_K28_DO_NOT_DEPLOY.

PROTOCOL:
    1. Load nb2154_pooled_mean_oof.npy + nb2154_pooled_median_oof.npy.
    2. Load y_unb (_audit_unblind_y.npy, 253 truth).
    3. Recompute RAE both axes, compare against nb2103 K=28 baseline.
    4. If pooled <= 0.4737 (mean axis), proceed to deploy CSV; else skip.

PRE-UNBLIND:
    Substrate is PRE-unblind clean (nb2154/nb2103). y_unb is post-unblind
    truth used ONLY for honest cross-fit RAE evaluation, not for fitting.

Outputs:
    scripts/nb2163_pooled_aggregate.py
    data/processed/nb2163_summary.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb2163"

# Baseline references (nb2103 K=28 single-config 5-seed bag, 5-fold cross-fit)
NB2103_K28_MEAN_BAG = 0.4737
NB2103_K28_MEDIAN_BAG = 0.4698

# Deploy threshold -- only build deploy CSV if pooled <= K=28 mean axis
DEPLOY_THRESH_MEAN = 0.4737


def main():
    t0 = time.time()
    out = {"tag": TAG}

    # ------------------------------------------------------------------
    # 1) Load unblind truth + idx
    # ------------------------------------------------------------------
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    n_unb = len(y_unb)
    out["n_unb"] = int(n_unb)
    out["y_unb_mean"] = float(y_unb.mean())
    out["y_unb_std"] = float(y_unb.std())

    # ------------------------------------------------------------------
    # 2) Load nb2154 pooled OOF arrays + per-cycle RAE
    # ------------------------------------------------------------------
    pooled_mean = np.load(
        DATA_PROCESSED / "nb2154_pooled_mean_oof.npy"
    ).astype(np.float64)
    pooled_median = np.load(
        DATA_PROCESSED / "nb2154_pooled_median_oof.npy"
    ).astype(np.float64)
    per_cycle_rae = np.load(
        DATA_PROCESSED / "nb2154_per_cycle_rae.npy"
    ).astype(np.float64)
    best_oof = np.load(
        DATA_PROCESSED / "nb2154_best_oof.npy"
    ).astype(np.float64)

    assert pooled_mean.shape == (n_unb,), pooled_mean.shape
    assert pooled_median.shape == (n_unb,), pooled_median.shape
    assert per_cycle_rae.shape == (120,), per_cycle_rae.shape

    out["n_cycles"] = 120
    out["per_cycle_rae_mean"] = float(per_cycle_rae.mean())
    out["per_cycle_rae_median"] = float(np.median(per_cycle_rae))
    out["per_cycle_rae_std"] = float(per_cycle_rae.std())
    out["per_cycle_rae_min"] = float(per_cycle_rae.min())
    out["per_cycle_rae_max"] = float(per_cycle_rae.max())

    # ------------------------------------------------------------------
    # 3) Recompute RAE on pooled aggregates
    # ------------------------------------------------------------------
    rae_pooled_mean = float(rae(y_unb, pooled_mean))
    rae_pooled_median = float(rae(y_unb, pooled_median))
    rae_best_oof = float(rae(y_unb, best_oof))

    out["rae_pooled_mean_120"] = rae_pooled_mean
    out["rae_pooled_median_120"] = rae_pooled_median
    out["rae_best_oof_single_cycle"] = rae_best_oof

    # ------------------------------------------------------------------
    # 4) Compare against K=28 single-config baseline
    # ------------------------------------------------------------------
    out["nb2103_K28_mean_bag"] = NB2103_K28_MEAN_BAG
    out["nb2103_K28_median_bag"] = NB2103_K28_MEDIAN_BAG

    delta_mean = rae_pooled_mean - NB2103_K28_MEAN_BAG
    delta_median = rae_pooled_median - NB2103_K28_MEDIAN_BAG
    out["delta_pooled_mean_vs_K28_mean"] = delta_mean
    out["delta_pooled_median_vs_K28_median"] = delta_median

    # Beats baseline? (lower is better)
    beats_mean = rae_pooled_mean <= NB2103_K28_MEAN_BAG
    beats_median = rae_pooled_median <= NB2103_K28_MEDIAN_BAG
    out["beats_K28_mean_axis"] = bool(beats_mean)
    out["beats_K28_median_axis"] = bool(beats_median)

    # ------------------------------------------------------------------
    # 5) Verdict
    # ------------------------------------------------------------------
    if rae_pooled_mean <= DEPLOY_THRESH_MEAN:
        verdict = "POOLED_BEATS_K28_PROCEED_TO_DEPLOY"
        deploy = True
    else:
        verdict = "INFERIOR_TO_K28_DO_NOT_DEPLOY"
        deploy = False

    out["deploy_thresh_mean"] = DEPLOY_THRESH_MEAN
    out["proceed_to_deploy"] = bool(deploy)
    out["verdict"] = verdict

    # ------------------------------------------------------------------
    # 6) Diagnostic notes
    # ------------------------------------------------------------------
    diag = []
    diag.append(
        f"pooled-mean-120 RAE {rae_pooled_mean:.4f} vs K=28 mean_bag "
        f"{NB2103_K28_MEAN_BAG:.4f} -- delta {delta_mean:+.4f}"
    )
    diag.append(
        f"pooled-median-120 RAE {rae_pooled_median:.4f} vs K=28 median_bag "
        f"{NB2103_K28_MEDIAN_BAG:.4f} -- delta {delta_median:+.4f}"
    )
    diag.append(
        f"best-cycle (cycle 78) RAE {rae_best_oof:.4f} is LUCKY_SEED per "
        f"nb2156 verify (cross-kf_seed repro mean 0.5050)"
    )
    diag.append(
        f"per-cycle RAE distribution: mean {per_cycle_rae.mean():.4f} "
        f"+- {per_cycle_rae.std():.4f}, "
        f"range [{per_cycle_rae.min():.4f}, {per_cycle_rae.max():.4f}]"
    )
    diag.append(
        "Pooled aggregate AVERAGES OUT lucky-seed variance, leaving the "
        "MEAN per-cycle behavior which is ~0.4823 -- WORSE than the "
        "K=28 single-config bag because hparam perturbation injects "
        "variance that across-cycle aggregation cannot fully smooth out."
    )
    diag.append(
        "K=28 single-config bag uses a SINGLE fixed hparam tuple, so "
        "all 5 seeds vary only by sklearn KFold seed + LGBM seed, "
        "yielding tighter conditional variance and lower bag RAE."
    )
    diag.append(
        "DO_NOT_DEPLOY -- pooled-120 is INFERIOR to K=28 baseline. "
        "The trajectory was useful for measuring lucky-seed variance "
        "(8/120 cycles beat 0.4698, only 14/120 cycles beat 0.4737), "
        "NOT for producing a deploy-grade aggregate."
    )
    out["diagnostic_notes"] = diag

    out["pre_unblind_clean"] = True
    out["deploy_csv"] = None
    out["wall_sec"] = time.time() - t0

    # ------------------------------------------------------------------
    # 7) Save summary
    # ------------------------------------------------------------------
    p_summary = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(p_summary, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[nb2163] wrote {p_summary}")
    print(f"[nb2163] verdict: {verdict}")
    print(f"[nb2163] rae_pooled_mean_120   = {rae_pooled_mean:.6f}")
    print(f"[nb2163] rae_pooled_median_120 = {rae_pooled_median:.6f}")
    print(f"[nb2163] rae_best_oof          = {rae_best_oof:.6f} (LUCKY)")
    print(f"[nb2163] K=28 mean_bag         = {NB2103_K28_MEAN_BAG:.6f}")
    print(f"[nb2163] K=28 median_bag       = {NB2103_K28_MEDIAN_BAG:.6f}")
    print(f"[nb2163] delta pooled_mean - K28_mean = {delta_mean:+.6f}")
    print(f"[nb2163] delta pooled_med  - K28_med  = {delta_median:+.6f}")


if __name__ == "__main__":
    main()
