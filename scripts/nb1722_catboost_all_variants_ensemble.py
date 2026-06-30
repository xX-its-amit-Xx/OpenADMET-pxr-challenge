"""nb1722 -- Ensemble of all CatBoost variants on the 5-way K-tuned 117-col stack.

HYPOTHESIS:
    Six CatBoost variants were tested on the identical 5-way K-tuned 117-col
    feature stack (anchor = chemprop_aux):
        nb1554  CatBoost MAE d4 n200 lr0.05 l2=5     -> 0.5163 mean_bag (PRE)
        nb1573  CatBoost MAE d3 n200 lr0.05 l2=5     -> per nb1573_summary.json
        nb1602  CatBoost MAE d5 n200 lr0.05 l2=5     -> per nb1602_summary.json
        nb1703  CatBoost MAE d4 alt-params           -> per nb1703_summary.json
        nb1712  CatBoost MAE d4 n300 lr0.03 l2=10    -> 0.5262 mean_bag
        nb1721  CatBoost MAE d2 (very shallow)       -> per nb1721_summary.json
    Each variant has a slightly different hyper-param recipe but the same
    residual surface.  Their per-row mean_bag_oof preds are highly correlated
    (Pearson 0.99+ vs nb1554 in prior reports), but a NAIVE MEAN over variants
    should still shave a small amount of seed/hp noise via independent-error
    averaging.  Reference: nb1561 BoB mean_bag = 0.5155 (the verified ceiling
    of the nb1554 family under outer-seed bagging).

PROTOCOL:
    1. Load mean_bag_oof for each available variant in the list.
       Missing variants are skipped with a warning (e.g., nb1721 not yet built).
    2. Naive arithmetic mean across all loaded variants -> ens_oof (253,).
    3. Pool RAE = rae(y_unb, ens_oof).
    4. Verdict vs nb1561 BoB mean_bag = 0.5155 at the standard 0.003 margin.

Outputs:
    scripts/nb1722_catboost_all_variants_ensemble.py
    data/processed/nb1722_summary.json
    data/processed/nb1722_ens_oof.npy             (253,) float32
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

TAG = "nb1722"
ANCHOR = "chemprop_aux"

VARIANTS = [
    ("nb1554", "CatBoost MAE d4 n200 lr0.05 l2=5"),
    ("nb1573", "CatBoost MAE d3 n200 lr0.05 l2=5"),
    ("nb1602", "CatBoost MAE d5 n200 lr0.05 l2=5"),
    ("nb1703", "CatBoost MAE d4 alt-params"),
    ("nb1712", "CatBoost MAE d4 n300 lr0.03 l2=10"),
    ("nb1721", "CatBoost MAE d2 shallow"),
]

NB1561_BOB_MEAN_REF = 0.5155
DECISION_MARGIN = 0.003


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"[{TAG}] ENSEMBLE OF ALL CATBOOST VARIANTS  (naive mean)")
    print(f"          variants: {[v[0] for v in VARIANTS]}")
    print(f"          ref: nb1561 BoB mean ({NB1561_BOB_MEAN_REF:.4f})"
          f"  margin = {DECISION_MARGIN}")
    print("=" * 78)

    # ---- Load truth ----
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_unb = {n_unb}")

    # ---- Load every available variant ----
    loaded = []
    skipped = []
    per_variant_rae = {}
    for tag, desc in VARIANTS:
        p = DATA_PROCESSED / f"{tag}_mean_bag_oof.npy"
        if not p.exists():
            print(f"[skip] {tag}  missing: {p}")
            skipped.append({"tag": tag, "desc": desc,
                            "reason": f"file not found: {p.name}"})
            continue
        arr = np.load(p).astype(np.float64)
        if arr.shape[0] != n_unb:
            print(f"[skip] {tag}  shape mismatch: {arr.shape} vs ({n_unb},)")
            skipped.append({"tag": tag, "desc": desc,
                            "reason": f"shape {arr.shape} != ({n_unb},)"})
            continue
        r = float(rae(y_unb, arr))
        per_variant_rae[tag] = r
        loaded.append((tag, desc, arr))
        print(f"[load] {tag:>7s}  RAE = {r:.4f}  ({desc})")

    if len(loaded) < 2:
        raise RuntimeError(
            f"need at least 2 variants for ensemble; loaded {len(loaded)}"
        )

    # ---- Naive arithmetic mean ----
    stack = np.stack([arr for _, _, arr in loaded], axis=0)
    ens_oof = stack.mean(axis=0)
    rae_ens = float(rae(y_unb, ens_oof))

    # ---- Pairwise pearson against best single ----
    best_single_tag = min(per_variant_rae, key=per_variant_rae.get)
    best_single_rae = per_variant_rae[best_single_tag]
    best_arr = dict([(t, a) for t, _, a in loaded])[best_single_tag]
    pearson_vs_best = float(np.corrcoef(ens_oof, best_arr)[0, 1])

    delta_vs_best_single = rae_ens - best_single_rae
    delta_vs_nb1561 = rae_ens - NB1561_BOB_MEAN_REF
    beats_best_single = delta_vs_best_single < -DECISION_MARGIN
    beats_nb1561 = delta_vs_nb1561 < -DECISION_MARGIN
    flat_vs_nb1561 = abs(delta_vs_nb1561) <= DECISION_MARGIN

    print()
    print("=" * 78)
    print(f"[{TAG}] RESULTS")
    print(f"   n_variants_loaded = {len(loaded)}  (skipped: {len(skipped)})")
    for tag, r in sorted(per_variant_rae.items(), key=lambda kv: kv[1]):
        print(f"     {tag:>7s}  RAE = {r:.4f}")
    print(f"   ENSEMBLE (naive mean) RAE = {rae_ens:.4f}")
    print(f"     vs best single ({best_single_tag} {best_single_rae:.4f}): "
          f"delta = {delta_vs_best_single:+.4f}  "
          f"beats = {beats_best_single}")
    print(f"     vs nb1561 BoB ({NB1561_BOB_MEAN_REF:.4f}):     "
          f"delta = {delta_vs_nb1561:+.4f}  "
          f"beats = {beats_nb1561}  flat = {flat_vs_nb1561}")
    print(f"     pearson(ens, best_single={best_single_tag}) = "
          f"{pearson_vs_best:.4f}")
    print("=" * 78)

    if beats_nb1561:
        verdict = "CATBOOST_ALL_VARIANTS_NAIVE_MEAN_BEATS_NB1561_BOB"
    elif flat_vs_nb1561:
        verdict = "CATBOOST_ALL_VARIANTS_NAIVE_MEAN_FLAT_VS_NB1561_BOB"
    else:
        verdict = "CATBOOST_ALL_VARIANTS_NAIVE_MEAN_WORSE_THAN_NB1561_BOB"
    print(f"[verdict] {verdict}")

    # ---- Save ----
    out_oof = DATA_PROCESSED / f"{TAG}_ens_oof.npy"
    np.save(out_oof, ens_oof.astype(np.float32))
    print(f"[save] {out_oof}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx_via_loaded_variants",
        "method": "naive_arithmetic_mean_over_catboost_variant_mean_bag_oofs",
        "variants_requested": [
            {"tag": t, "desc": d} for t, d in VARIANTS
        ],
        "variants_loaded": [t for t, _, _ in loaded],
        "variants_skipped": skipped,
        "n_variants_loaded": len(loaded),
        "n_unb": n_unb,
        "per_variant_rae": per_variant_rae,
        "best_single_tag": best_single_tag,
        "best_single_rae": best_single_rae,
        "rae_ensemble": rae_ens,
        "pearson_ens_vs_best_single": pearson_vs_best,
        "nb1561_bob_mean_ref": NB1561_BOB_MEAN_REF,
        "decision_margin": DECISION_MARGIN,
        "delta_vs_best_single": delta_vs_best_single,
        "delta_vs_nb1561_bob_mean": delta_vs_nb1561,
        "beats_best_single": beats_best_single,
        "beats_nb1561": beats_nb1561,
        "flat_vs_nb1561": flat_vs_nb1561,
        "verdict": verdict,
        "pre_unblind_clean": True,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_json = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_json}")
    print(f"[done] wall = {summary['wall_sec']:.2f}s")
    return summary


if __name__ == "__main__":
    main()
