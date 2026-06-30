"""nb1265 -- OUTER-BAG VALIDATION of nb1251 best-w blend (w=0.55 nb1242 + w=0.45
nb1211 = pooled RAE 0.5394 on 253 unblind).

PRECEDENT
---------
nb1252 outer-bagged nb1242 (ChEMBL-kNN feat residual on nb1070 anchor): its
per-outer mean over outer seeds {0, 1, 7, 42, 137} with inner family
[o*1000 + s for s in {0, 1, 7, 42, 137}] landed at 0.5462 -- +0.0031 vs
the standalone nb1242 0.5431 reference, confirming nb1242 was a lucky outer
seed.  Same risk for nb1251's best-w blend: maybe the 0.5394 is partly
riding nb1242's lucky tail; if we rebuild nb1242 under fresh outer seeds and
rebuild nb1211's components (nb1190 triple BoB + nb1200 MACCS BoB) under
the SAME outer seeds, the best-w 0.55/0.45 blend should land somewhere
between 0.539 and ~0.550 -- the gap quantifies the seed-luck premium baked
into the 0.5394 number.

PROTOCOL
--------
We reuse the three already-cached per-outer arrays (all built on the SAME
outer family {0, 1, 7, 42, 137} with inner = [o*1000 + s for s in
{0, 1, 7, 42, 137}]):
    data/processed/nb1252_per_seed_corrected_oof.npy   (5, 253)  nb1242_o
    data/processed/nb1190_per_outer_oof.npy            (5, 253)  nb1190_o
    data/processed/nb1200_per_outer_oof.npy            (5, 253)  nb1200_o

For each outer seed o in {0, 1, 7, 42, 137}:
    1. nb1242_o = nb1252_per_seed_corrected_oof[oi]
    2. nb1211_o = 0.5 * nb1190_per_outer_oof[oi] + 0.5 * nb1200_per_outer_oof[oi]
       (the nb1211 "best blend" was the naive 0.5/0.5 MEAN-block;
        per-outer-rebuild uses the same 0.5/0.5 recipe so the operator
        applied to nb1211 components matches the static nb1211 reference)
    3. nb1251_o = W_NB1242 * nb1242_o + W_NB1211 * nb1211_o
       with W_NB1242 = 0.55, W_NB1211 = 0.45 (from nb1251 grid).
    4. per-outer-RAE = pooled RAE(y_unb, nb1251_o)

Also report:
    - per-outer nb1242_o RAE (sanity check vs nb1252)
    - per-outer nb1211_o RAE (sanity check vs nb1224 / nb1211 mean)
    - bag-of-bags row-level MEAN across the 5 outer-blend predictions
    - bag-of-bags row-level MEDIAN across the 5 outer-blend predictions

VERDICT
-------
NB1251_REPRODUCES if abs(per_outer_mean - 0.5394) <= 0.003.
Otherwise: NB1251_PESSIMISTIC (better) or NB1251_LUCKY (pulls up).

Outputs:
  data/processed/nb1265_per_outer_nb1242.npy  (5, 253) float32  nb1242_o (copy)
  data/processed/nb1265_per_outer_nb1211.npy  (5, 253) float32  nb1211_o
  data/processed/nb1265_per_outer_blend.npy   (5, 253) float32  nb1251_o
  data/processed/nb1265_bob_mean_oof.npy      (253,)   float32  row-mean
  data/processed/nb1265_bob_median_oof.npy    (253,)   float32  row-median
  data/processed/nb1265_summary.json
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

TAG = "nb1265"

# Outer-seed family + inner family (sanity record only; arrays already built).
OUTER_SEEDS = [0, 1, 7, 42, 137]
INNER_BASE = [0, 1, 7, 42, 137]
INNER_FAMILY_RULE = "inner = outer * 1000 + base  (matches nb1242 / nb1252 / nb1190 / nb1200)"

# nb1251 best-w grid output: w[nb1242]=0.55, w[nb1211]=0.45 -> pooled RAE 0.5394.
W_NB1242 = 0.55
W_NB1211 = 0.45

# References (pooled RAE on 253 unblind).
NB1251_BESTW_REF = 0.5394
NB1242_STANDALONE_REF = 0.5431
NB1211_STANDALONE_REF = 0.5451
NB1252_BOB_MEAN_REF = 0.5462   # nb1242 outer-bag mean from nb1252
NB1224_BOB_MEAN_REF = 0.5451   # nb1211 outer-bag mean from nb1224 (different outer family)

REPRO_MARGIN = 0.003


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- OUTER-BAG validation of nb1251 best-w blend")
    print(f"          w[nb1242]={W_NB1242:.2f}  w[nb1211]={W_NB1211:.2f}")
    print(f"          reference = {NB1251_BESTW_REF:.4f}")
    print(f"          OUTER seeds  = {OUTER_SEEDS}")
    print(f"          INNER family = {INNER_FAMILY_RULE}")
    print(f"          repro margin = {REPRO_MARGIN:.3f}")
    print("=" * 78)

    # ---- Load truth ----
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_unb = {n_unb}")

    # ---- Load cached per-outer arrays ----
    print("\n" + "-" * 78)
    print("LOAD CACHED PER-OUTER ARRAYS")
    print("-" * 78)

    nb1242_path = DATA_PROCESSED / "nb1252_per_seed_corrected_oof.npy"
    nb1190_path = DATA_PROCESSED / "nb1190_per_outer_oof.npy"
    nb1200_path = DATA_PROCESSED / "nb1200_per_outer_oof.npy"

    for p in (nb1242_path, nb1190_path, nb1200_path):
        if not p.exists():
            raise FileNotFoundError(f"required cache missing: {p}")

    nb1242_per_outer = np.load(nb1242_path).astype(np.float64)  # (5, 253) nb1242_o
    nb1190_per_outer = np.load(nb1190_path).astype(np.float64)  # (5, 253) nb1190_o
    nb1200_per_outer = np.load(nb1200_path).astype(np.float64)  # (5, 253) nb1200_o

    for name, arr in [
        ("nb1242 (from nb1252)", nb1242_per_outer),
        ("nb1190",               nb1190_per_outer),
        ("nb1200",               nb1200_per_outer),
    ]:
        if arr.shape != (len(OUTER_SEEDS), n_unb):
            raise ValueError(
                f"{name} shape mismatch: {arr.shape} vs ({len(OUTER_SEEDS)}, {n_unb})"
            )
        print(f"   {name:30s} : {arr.shape}")

    # ---- Cross-check sanity vs static refs ----
    # nb1252 BoB mean (across 5 outer seeds) should match its reference 0.5462.
    nb1242_bob_mean = nb1242_per_outer.mean(axis=0)
    rae_nb1242_bob_mean = float(rae(y_unb, nb1242_bob_mean))
    print(f"\n[sanity] nb1242 outer-bag MEAN (this run): {rae_nb1242_bob_mean:.4f}  "
          f"(nb1252 ref {NB1252_BOB_MEAN_REF:.4f})")

    # ---- Per-outer compose ----
    print("\n" + "-" * 78)
    print(f"PER-OUTER COMPOSE  ({len(OUTER_SEEDS)} outer seeds)")
    print("-" * 78)

    per_outer_nb1242 = np.zeros((len(OUTER_SEEDS), n_unb), dtype=np.float64)
    per_outer_nb1211 = np.zeros((len(OUTER_SEEDS), n_unb), dtype=np.float64)
    per_outer_blend  = np.zeros((len(OUTER_SEEDS), n_unb), dtype=np.float64)

    per_outer_rae_nb1242: list[float] = []
    per_outer_rae_nb1211: list[float] = []
    per_outer_rae_blend:  list[float] = []
    per_outer_records: list[dict] = []

    for oi, o in enumerate(OUTER_SEEDS):
        inner_seeds = [int(o) * 1000 + int(s) for s in INNER_BASE]

        nb1242_o = nb1242_per_outer[oi]
        nb1211_o = 0.5 * nb1190_per_outer[oi] + 0.5 * nb1200_per_outer[oi]
        blend_o  = W_NB1242 * nb1242_o + W_NB1211 * nb1211_o

        rae_nb1242_o = float(rae(y_unb, nb1242_o))
        rae_nb1211_o = float(rae(y_unb, nb1211_o))
        rae_blend_o  = float(rae(y_unb, blend_o))

        per_outer_nb1242[oi] = nb1242_o
        per_outer_nb1211[oi] = nb1211_o
        per_outer_blend[oi]  = blend_o

        per_outer_rae_nb1242.append(rae_nb1242_o)
        per_outer_rae_nb1211.append(rae_nb1211_o)
        per_outer_rae_blend.append(rae_blend_o)

        per_outer_records.append({
            "outer_seed": int(o),
            "inner_seeds": inner_seeds,
            "rae_nb1242_o": rae_nb1242_o,
            "rae_nb1211_o": rae_nb1211_o,
            "rae_blend_o":  rae_blend_o,
            "delta_blend_vs_nb1251_ref": rae_blend_o - NB1251_BESTW_REF,
        })
        print(f"   outer {o:5d}  inner={inner_seeds}")
        print(f"     nb1242_o = {rae_nb1242_o:.4f}   "
              f"nb1211_o = {rae_nb1211_o:.4f}   "
              f"blend_o  = {rae_blend_o:.4f}   "
              f"(d vs nb1251 = {rae_blend_o - NB1251_BESTW_REF:+.4f})")

    # ---- Aggregate ----
    arr_nb1242 = np.array(per_outer_rae_nb1242)
    arr_nb1211 = np.array(per_outer_rae_nb1211)
    arr_blend  = np.array(per_outer_rae_blend)

    outer_mean_nb1242 = float(arr_nb1242.mean())
    outer_std_nb1242  = float(arr_nb1242.std())
    outer_mean_nb1211 = float(arr_nb1211.mean())
    outer_std_nb1211  = float(arr_nb1211.std())
    outer_mean_blend  = float(arr_blend.mean())
    outer_std_blend   = float(arr_blend.std())
    outer_min_blend   = float(arr_blend.min())
    outer_max_blend   = float(arr_blend.max())

    # Bag-of-bags row-level aggregations.
    bob_mean_oof   = per_outer_blend.mean(axis=0)
    bob_median_oof = np.median(per_outer_blend, axis=0)
    rae_bob_mean   = float(rae(y_unb, bob_mean_oof))
    rae_bob_median = float(rae(y_unb, bob_median_oof))

    # ---- Verdict ----
    delta = outer_mean_blend - NB1251_BESTW_REF
    reproduces = abs(delta) <= REPRO_MARGIN
    if reproduces:
        verdict = "NB1251_REPRODUCES"
    elif delta < -REPRO_MARGIN:
        verdict = "NB1251_PESSIMISTIC_OUTER_BAG_BETTER"
    else:
        verdict = "NB1251_LUCKY_OUTER_BAG_PULLS_UP"

    print("\n" + "=" * 78)
    print("OUTER-BAG AGGREGATIONS")
    print("=" * 78)
    print(f"   per-outer nb1242_o RAE = "
          f"[{', '.join(f'{r:.4f}' for r in per_outer_rae_nb1242)}]")
    print(f"   per-outer nb1211_o RAE = "
          f"[{', '.join(f'{r:.4f}' for r in per_outer_rae_nb1211)}]")
    print(f"   per-outer blend    RAE = "
          f"[{', '.join(f'{r:.4f}' for r in per_outer_rae_blend)}]")
    print(f"")
    print(f"   nb1242_o per-outer mean = {outer_mean_nb1242:.4f}  "
          f"std = {outer_std_nb1242:.4f}")
    print(f"   nb1211_o per-outer mean = {outer_mean_nb1211:.4f}  "
          f"std = {outer_std_nb1211:.4f}")
    print(f"   blend    per-outer mean = {outer_mean_blend:.4f}  "
          f"std = {outer_std_blend:.4f}")
    print(f"   blend    per-outer min  = {outer_min_blend:.4f}")
    print(f"   blend    per-outer max  = {outer_max_blend:.4f}")
    print(f"")
    print(f"   bag-of-bags MEAN   row-level RAE = {rae_bob_mean:.4f}")
    print(f"   bag-of-bags MEDIAN row-level RAE = {rae_bob_median:.4f}")
    print(f"")
    print(f"   nb1251 best-w reference          = {NB1251_BESTW_REF:.4f}")
    print(f"   delta(per-outer blend mean - ref) = {delta:+.4f}  "
          f"(margin {REPRO_MARGIN:.3f})")
    print(f"   VERDICT = {verdict}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_per_outer_nb1242.npy",
            per_outer_nb1242.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_per_outer_nb1211.npy",
            per_outer_nb1211.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_per_outer_blend.npy",
            per_outer_blend.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_bob_mean_oof.npy",
            bob_mean_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_bob_median_oof.npy",
            bob_median_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_per_outer_nb1242.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_per_outer_nb1211.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_per_outer_blend.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_bob_mean_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_bob_median_oof.npy'}")

    summary = {
        "tag": TAG,
        "n_unb": n_unb,
        "outer_seeds": OUTER_SEEDS,
        "inner_base": INNER_BASE,
        "inner_family_rule": INNER_FAMILY_RULE,
        "w_nb1242": W_NB1242,
        "w_nb1211": W_NB1211,
        "nb1251_bestw_ref": NB1251_BESTW_REF,
        "nb1242_standalone_ref": NB1242_STANDALONE_REF,
        "nb1211_standalone_ref": NB1211_STANDALONE_REF,
        "nb1252_bob_mean_ref": NB1252_BOB_MEAN_REF,
        "nb1224_bob_mean_ref": NB1224_BOB_MEAN_REF,
        "rae_nb1242_outer_bag_mean_sanity": rae_nb1242_bob_mean,
        "per_outer_records": per_outer_records,
        "per_outer_rae_nb1242": [float(x) for x in per_outer_rae_nb1242],
        "per_outer_rae_nb1211": [float(x) for x in per_outer_rae_nb1211],
        "per_outer_rae_blend":  [float(x) for x in per_outer_rae_blend],
        "outer_mean_nb1242": outer_mean_nb1242,
        "outer_std_nb1242":  outer_std_nb1242,
        "outer_mean_nb1211": outer_mean_nb1211,
        "outer_std_nb1211":  outer_std_nb1211,
        "outer_mean_blend":  outer_mean_blend,
        "outer_std_blend":   outer_std_blend,
        "outer_min_blend":   outer_min_blend,
        "outer_max_blend":   outer_max_blend,
        "rae_bob_mean":      rae_bob_mean,
        "rae_bob_median":    rae_bob_median,
        "delta_outer_mean_vs_nb1251": delta,
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
    for k in (
        "outer_seeds",
        "w_nb1242", "w_nb1211",
        "per_outer_rae_nb1242",
        "per_outer_rae_nb1211",
        "per_outer_rae_blend",
        "outer_mean_nb1242", "outer_mean_nb1211",
        "outer_mean_blend", "outer_std_blend",
        "outer_min_blend", "outer_max_blend",
        "rae_bob_mean", "rae_bob_median",
        "delta_outer_mean_vs_nb1251",
        "reproduces", "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
