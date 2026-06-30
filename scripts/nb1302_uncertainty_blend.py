"""nb1302 -- Uncertainty-aware stacking: per-row inverse-std weighting.

Hypothesis:
    Both nb1242 (per_seed_corrected_oof, 5 seeds, 0.5431 pooled) and nb1190
    (per_outer_oof, 5 outer seeds, 0.5499 pooled) carry per-row std as a
    free uncertainty estimate. Rows where the seeds agree (low std) should
    be trusted more. Inverse-std (or inverse-variance) per-row weighting
    may extract gain over the flat 0.35/0.65 fixed-w blend baseline
    (nb1290 = 0.5390).

Protocol:
  1. Load nb1242_per_seed_corrected_oof.npy (5, 253) -> mean / std per row.
  2. Load nb1190_per_outer_oof.npy (5, 253) -> mean / std per row.
  3. Variant A: w_nb1190 = (1/s1190) / (1/s1190 + 1/s1242), per row.
                pred = w*p1190 + (1-w)*p1242.
  4. Variant B: w_nb1190 = (1/v1190) / (1/v1190 + 1/v1242)  (inverse variance).
  5. Variant C: floored inverse-std (eps = max(std, 0.05)).
  6. Variant D: rank-by-std with abstention -- when MAX(s1190, s1242) is in
               top-10% (most uncertain), fall back to nb1190 mean prediction
               (most conservative).
  7. Variant E: rank-by-std with abstention -> fall back to nb1242 mean.
  8. Variant F: rank-by-std with abstention -> fall back to nb1290 best fixed-w
               (0.35*nb1190 + 0.65*nb1242).
  9. Pool RAE per variant.
 10. Verdict at 0.003 margin vs nb1290 (0.5390 fixed-w best).

NO deploy (513) refit -- 253-only honest cross-fit diagnostic.

Outputs:
  data/processed/nb1302_invstd_oof.npy
  data/processed/nb1302_invvar_oof.npy
  data/processed/nb1302_invstd_floored_oof.npy
  data/processed/nb1302_abst_nb1190_oof.npy
  data/processed/nb1302_abst_nb1242_oof.npy
  data/processed/nb1302_abst_nb1290_oof.npy
  data/processed/nb1302_summary.json
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

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1302"

# Reference numbers (pooled RAE on 253 unblind).
NB1190_REF = 0.5499
NB1242_REF = 0.5431
NB1290_REF = 0.5390
MARGIN = 0.003

EPS = 1e-12


def _percentile_stats(arr: np.ndarray) -> dict:
    return {
        "min":  float(arr.min()),
        "p10":  float(np.percentile(arr, 10)),
        "p50":  float(np.percentile(arr, 50)),
        "mean": float(arr.mean()),
        "p90":  float(np.percentile(arr, 90)),
        "max":  float(arr.max()),
        "std":  float(arr.std()),
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Uncertainty-aware stacking: per-row inverse-std weighting")
    print(f"          nb1242 5-seed std vs nb1190 5-outer std")
    print(f"          verdict margin {MARGIN} vs nb1290 (0.5390 best fixed-w)")
    print("=" * 78)

    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"\n[load] y_unb shape = {y_unb.shape}")

    # ---- nb1242 per-seed (5, 253) ----
    ps1242_path = DATA_PROCESSED / "nb1242_per_seed_corrected_oof.npy"
    ps1242 = np.load(ps1242_path).astype(np.float64)  # (5, 253)
    if ps1242.shape != (5, n_unb):
        raise ValueError(f"unexpected nb1242 per-seed shape: {ps1242.shape}")
    mean_1242 = ps1242.mean(axis=0)
    std_1242  = ps1242.std(axis=0, ddof=0)  # population std across 5 seeds
    rae_1242  = float(rae(y_unb, mean_1242))
    print(f"[load] nb1242 per-seed: shape={ps1242.shape}  "
          f"mean-bag RAE={rae_1242:.4f}  (ref {NB1242_REF:.4f})")

    # ---- nb1190 per-outer (5, 253) ----
    ps1190_path = DATA_PROCESSED / "nb1190_per_outer_oof.npy"
    ps1190 = np.load(ps1190_path).astype(np.float64)  # (5, 253)
    if ps1190.shape != (5, n_unb):
        raise ValueError(f"unexpected nb1190 per-outer shape: {ps1190.shape}")
    mean_1190 = ps1190.mean(axis=0)
    std_1190  = ps1190.std(axis=0, ddof=0)
    rae_1190  = float(rae(y_unb, mean_1190))
    print(f"[load] nb1190 per-outer: shape={ps1190.shape}  "
          f"mean-bag RAE={rae_1190:.4f}  (ref {NB1190_REF:.4f})")

    # ---- nb1290 best fixed-w baseline (0.35*nb1190 + 0.65*nb1242) ----
    p1290 = 0.35 * mean_1190 + 0.65 * mean_1242
    rae_1290 = float(rae(y_unb, p1290))
    print(f"[load] nb1290 0.35/0.65 fixed-w: RAE={rae_1290:.4f}  "
          f"(ref {NB1290_REF:.4f})")

    # ---- Diagnostics on per-row std ----
    print("\n" + "-" * 78)
    print("  BLOCK: per-row std distributions")
    print("-" * 78)
    s1190_stats = _percentile_stats(std_1190)
    s1242_stats = _percentile_stats(std_1242)
    print(f"   std_nb1190 across 5 outers:")
    print(f"     min={s1190_stats['min']:.4f}  p10={s1190_stats['p10']:.4f}  "
          f"p50={s1190_stats['p50']:.4f}  mean={s1190_stats['mean']:.4f}  "
          f"p90={s1190_stats['p90']:.4f}  max={s1190_stats['max']:.4f}")
    print(f"   std_nb1242 across 5 seeds:")
    print(f"     min={s1242_stats['min']:.4f}  p10={s1242_stats['p10']:.4f}  "
          f"p50={s1242_stats['p50']:.4f}  mean={s1242_stats['mean']:.4f}  "
          f"p90={s1242_stats['p90']:.4f}  max={s1242_stats['max']:.4f}")
    # Std correlation
    std_corr = float(np.corrcoef(std_1190, std_1242)[0, 1])
    print(f"   Pearson(std_1190, std_1242) = {std_corr:.4f}")

    # Does std correlate with abs error?
    abs_err_1190 = np.abs(mean_1190 - y_unb)
    abs_err_1242 = np.abs(mean_1242 - y_unb)
    corr_s1190_err1190 = float(np.corrcoef(std_1190, abs_err_1190)[0, 1])
    corr_s1242_err1242 = float(np.corrcoef(std_1242, abs_err_1242)[0, 1])
    print(f"\n   uncertainty-calibration diagnostic:")
    print(f"     Pearson(std_1190, |err_1190|) = {corr_s1190_err1190:.4f}")
    print(f"     Pearson(std_1242, |err_1242|) = {corr_s1242_err1242:.4f}")
    print(f"     (positive -> high std rows tend to be high error)")

    # ---- Variant A: inverse-std weight per row ----
    inv_s1190 = 1.0 / np.maximum(std_1190, EPS)
    inv_s1242 = 1.0 / np.maximum(std_1242, EPS)
    w_invstd = inv_s1190 / (inv_s1190 + inv_s1242)
    p_invstd = w_invstd * mean_1190 + (1.0 - w_invstd) * mean_1242
    rae_invstd = float(rae(y_unb, p_invstd))

    # ---- Variant B: inverse-variance weight per row ----
    inv_v1190 = 1.0 / np.maximum(std_1190 * std_1190, EPS)
    inv_v1242 = 1.0 / np.maximum(std_1242 * std_1242, EPS)
    w_invvar = inv_v1190 / (inv_v1190 + inv_v1242)
    p_invvar = w_invvar * mean_1190 + (1.0 - w_invvar) * mean_1242
    rae_invvar = float(rae(y_unb, p_invvar))

    # ---- Variant C: floored inverse-std (eps = 0.05) ----
    FLOOR = 0.05
    inv_s1190f = 1.0 / np.maximum(std_1190, FLOOR)
    inv_s1242f = 1.0 / np.maximum(std_1242, FLOOR)
    w_floored = inv_s1190f / (inv_s1190f + inv_s1242f)
    p_floored = w_floored * mean_1190 + (1.0 - w_floored) * mean_1242
    rae_floored = float(rae(y_unb, p_floored))

    # ---- Variants D/E/F: abstention via max-std percentile fallback ----
    max_std = np.maximum(std_1190, std_1242)
    abst_threshold_p90 = float(np.percentile(max_std, 90))
    abst_mask = max_std >= abst_threshold_p90
    n_abst = int(abst_mask.sum())

    # baseline: use the inverse-std weighted blend on confident rows
    p_abst_base = p_invstd.copy()

    p_abst_1190 = p_abst_base.copy()
    p_abst_1190[abst_mask] = mean_1190[abst_mask]
    rae_abst_1190 = float(rae(y_unb, p_abst_1190))

    p_abst_1242 = p_abst_base.copy()
    p_abst_1242[abst_mask] = mean_1242[abst_mask]
    rae_abst_1242 = float(rae(y_unb, p_abst_1242))

    p_abst_1290 = p_abst_base.copy()
    p_abst_1290[abst_mask] = p1290[abst_mask]
    rae_abst_1290 = float(rae(y_unb, p_abst_1290))

    # ---- per-variant report ----
    print("\n" + "-" * 78)
    print("  BLOCK: variant RAE (pooled 253)")
    print("-" * 78)
    variants = {
        "A_invstd":         rae_invstd,
        "B_invvar":         rae_invvar,
        "C_invstd_floored": rae_floored,
        "D_abst_to_nb1190": rae_abst_1190,
        "E_abst_to_nb1242": rae_abst_1242,
        "F_abst_to_nb1290": rae_abst_1290,
    }
    for tag, val in sorted(variants.items(), key=lambda kv: kv[1]):
        print(f"   {tag:22s} RAE = {val:.4f}   delta vs nb1290 = "
              f"{val - NB1290_REF:+.4f}")
    print(f"\n   (abstention threshold p90 max_std = {abst_threshold_p90:.4f}, "
          f"n_abstained = {n_abst}/{n_unb})")

    best_variant = min(variants, key=variants.get)
    best_rae = variants[best_variant]
    beats_nb1290 = best_rae < NB1290_REF - MARGIN
    flat_nb1290 = abs(best_rae - NB1290_REF) < MARGIN

    if beats_nb1290:
        verdict = f"UNCERTAINTY_BLEND_BEATS_NB1290 ({best_variant} @ {best_rae:.4f})"
    elif flat_nb1290:
        verdict = f"UNCERTAINTY_BLEND_FLAT_VS_NB1290 ({best_variant} @ {best_rae:.4f})"
    else:
        verdict = f"UNCERTAINTY_BLEND_HURTS_VS_NB1290 ({best_variant} @ {best_rae:.4f})"

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"   nb1190 standalone        : {rae_1190:.4f}  (ref {NB1190_REF:.4f})")
    print(f"   nb1242 standalone        : {rae_1242:.4f}  (ref {NB1242_REF:.4f})")
    print(f"   nb1290 0.35/0.65 fixed-w : {rae_1290:.4f}  (ref {NB1290_REF:.4f})")
    print(f"")
    print(f"   best variant             : {best_variant}")
    print(f"   best RAE                 : {best_rae:.4f}")
    print(f"   delta vs nb1290 (0.5390) : {best_rae - NB1290_REF:+.4f}")
    print(f"   beats_nb1290 (>= {MARGIN})  : {beats_nb1290}")
    print(f"   verdict                  : {verdict}")

    # Persist canonical artifacts.
    np.save(DATA_PROCESSED / f"{TAG}_invstd_oof.npy",        p_invstd.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_invvar_oof.npy",        p_invvar.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_invstd_floored_oof.npy", p_floored.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_abst_nb1190_oof.npy",   p_abst_1190.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_abst_nb1242_oof.npy",   p_abst_1242.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_abst_nb1290_oof.npy",   p_abst_1290.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_invstd_oof.npy'} (and 5 sibling variants)")

    summary = {
        "tag": TAG,
        "n_unb": n_unb,
        "components": ["nb1190_per_outer", "nb1242_per_seed_corrected"],
        "standalone_rae": {
            "nb1190_mean_bag":      rae_1190,
            "nb1242_mean_bag":      rae_1242,
            "nb1290_best_fixed_w":  rae_1290,
        },
        "std_1190_stats": s1190_stats,
        "std_1242_stats": s1242_stats,
        "std_pearson_1190_1242": std_corr,
        "uncertainty_calibration": {
            "pearson_std1190_err1190": corr_s1190_err1190,
            "pearson_std1242_err1242": corr_s1242_err1242,
        },
        "abstention": {
            "max_std_p90_threshold": abst_threshold_p90,
            "n_abstained":           n_abst,
            "frac_abstained":        float(n_abst / n_unb),
        },
        "variants": variants,
        "best_variant": best_variant,
        "best_rae":     best_rae,
        "nb1190_ref": NB1190_REF,
        "nb1242_ref": NB1242_REF,
        "nb1290_ref": NB1290_REF,
        "delta_best_vs_nb1290": best_rae - NB1290_REF,
        "beats_nb1290":  bool(beats_nb1290),
        "flat_vs_nb1290": bool(flat_nb1290),
        "margin": MARGIN,
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
    for k in ("standalone_rae",
              "std_1190_stats", "std_1242_stats",
              "std_pearson_1190_1242",
              "uncertainty_calibration",
              "abstention",
              "variants",
              "best_variant", "best_rae",
              "delta_best_vs_nb1290",
              "beats_nb1290", "verdict"):
        print(f"  {k}: {res.get(k)}")
