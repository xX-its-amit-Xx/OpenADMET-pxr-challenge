"""nb1403 -- Outer-bag VALIDATION of nb1391 (best-w blend nb1373 + nb1352).

PROTOCOL
    For each of 5 outer seeds o in {0, 1, 7, 42, 137}:
        nb1373_o = top-30 AtomPair + ChEMBL residual learner with
                   inner seeds = [o*1000 + s for s in {0, 1, 7, 42, 137}].
                   Bag-mean across 5 inner seeds  --> (253,)
        nb1352_o = top-20 MACCS + ChEMBL residual learner, same inner-seed
                   family.  Bag-mean across 5 inner seeds  --> (253,)
        blend_o  = 0.85 * nb1373_o + 0.15 * nb1352_o    (nb1391 best-w)

    Per-outer pooled RAE = rae(y_unb, blend_o).
    BoB MEAN   = mean   of 5 blend_o vectors (row-level avg)  --> pooled RAE
    BoB MEDIAN = median of 5 blend_o vectors (row-level med)  --> pooled RAE

    Verdict NB1391_REPRODUCES iff |per_outer_mean - 0.5076| < 0.003.

IMPLEMENTATION NOTE
    The two component outer-bag families are already cached on disk:
        data/processed/nb1381_per_outer_mean_oof.npy   (5, 253)  <- nb1373 outer-bag-means
        data/processed/nb1361_outer_mean_bag.npy       (5, 253)  <- nb1352 outer-bag-means
    Both were built with the EXACT same outer seed list [0, 1, 7, 42, 137] and
    the EXACT inner-seed mapping inner_seeds(o) = [o*1000 + s for s in base].
    So the validated rebuild is a deterministic linear combination of those
    two arrays -- no LGBM refits required.  This is the canonical way to
    audit a blend's outer-bag stability (Sec.3 of phase1_postmortem -- "audit
    blends as fixed weights over already-bag-validated components").

OUTPUTS
    scripts/nb1403_bag_nb1391.py
    data/processed/nb1403_summary.json
    data/processed/nb1403_bob_mean_oof.npy     (253,) float32
    data/processed/nb1403_bob_median_oof.npy   (253,) float32
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1403"

OUTER_SEEDS = [0, 1, 7, 42, 137]
INNER_BASE_SEEDS = [0, 1, 7, 42, 137]

# Blend weights from nb1391 best variant (grid_mean_mean top-1).
W_NB1373 = 0.85
W_NB1352 = 0.15

# nb1391 reference: grid_mean_mean best = 0.5076454604858259 at w=0.85.
NB1391_REF = 0.5076
REPRODUCE_MARGIN = 0.003

# Component outer-bag caches (already validated by nb1381 / nb1361).
NB1373_OUTER_PATH = DATA_PROCESSED / "nb1381_per_outer_mean_oof.npy"
NB1352_OUTER_PATH = DATA_PROCESSED / "nb1361_outer_mean_bag.npy"


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- OUTER-BAG VALIDATION of nb1391 blend "
          f"({W_NB1373:.2f} * nb1373 + {W_NB1352:.2f} * nb1352)")
    print(f"         outer seeds      = {OUTER_SEEDS}")
    print(f"         inner base seeds = {INNER_BASE_SEEDS}")
    print(f"         inner_seeds(o)   = [o*1000 + s for s in base]")
    print(f"         nb1391 ref pooled RAE = {NB1391_REF:.4f}  "
          f"margin = {REPRODUCE_MARGIN}")
    print("=" * 78)

    # ---- Load truth ----
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] y_unb shape = ({n_unb},)")

    # ---- Load component outer-bag matrices (5, 253) ----
    if not NB1373_OUTER_PATH.exists():
        raise FileNotFoundError(
            f"{NB1373_OUTER_PATH} missing -- run nb1381 first."
        )
    if not NB1352_OUTER_PATH.exists():
        raise FileNotFoundError(
            f"{NB1352_OUTER_PATH} missing -- run nb1361 first."
        )

    nb1373_outer = np.load(NB1373_OUTER_PATH).astype(np.float64)
    nb1352_outer = np.load(NB1352_OUTER_PATH).astype(np.float64)
    print(f"[load] nb1373 outer-bag means  shape = {nb1373_outer.shape}  "
          f"(from {NB1373_OUTER_PATH.name})")
    print(f"[load] nb1352 outer-bag means  shape = {nb1352_outer.shape}  "
          f"(from {NB1352_OUTER_PATH.name})")

    n_outer = len(OUTER_SEEDS)
    if nb1373_outer.shape != (n_outer, n_unb):
        raise ValueError(
            f"nb1373 outer-bag shape mismatch: {nb1373_outer.shape} != "
            f"({n_outer}, {n_unb})"
        )
    if nb1352_outer.shape != (n_outer, n_unb):
        raise ValueError(
            f"nb1352 outer-bag shape mismatch: {nb1352_outer.shape} != "
            f"({n_outer}, {n_unb})"
        )

    # ---- Per-outer component sanity ----
    print("\n" + "-" * 78)
    print("PER-OUTER COMPONENT RAE (sanity vs nb1381 / nb1361 summaries)")
    print("-" * 78)
    per_outer_rae_nb1373: list[float] = []
    per_outer_rae_nb1352: list[float] = []
    for oi, o in enumerate(OUTER_SEEDS):
        r1 = float(rae(y_unb, nb1373_outer[oi]))
        r2 = float(rae(y_unb, nb1352_outer[oi]))
        per_outer_rae_nb1373.append(r1)
        per_outer_rae_nb1352.append(r2)
        print(f"   outer {o:3d}:  nb1373_o RAE = {r1:.4f}   "
              f"nb1352_o RAE = {r2:.4f}")

    # ---- Per-outer blend ----
    print("\n" + "-" * 78)
    print(f"PER-OUTER BLEND  (blend_o = {W_NB1373} * nb1373_o + "
          f"{W_NB1352} * nb1352_o)")
    print("-" * 78)
    per_outer_blend = (
        W_NB1373 * nb1373_outer + W_NB1352 * nb1352_outer
    )  # (5, 253)
    per_outer_rae_blend: list[float] = []
    per_outer_inner_seeds: list[list[int]] = []
    for oi, o in enumerate(OUTER_SEEDS):
        r = float(rae(y_unb, per_outer_blend[oi]))
        per_outer_rae_blend.append(r)
        inner_seeds = [int(o * 1000 + s) for s in INNER_BASE_SEEDS]
        per_outer_inner_seeds.append(inner_seeds)
        print(f"   outer {o:3d}:  blend RAE = {r:.4f}   "
              f"inner seeds = {inner_seeds}")

    per_outer_arr = np.array(per_outer_rae_blend)
    per_outer_mean = float(per_outer_arr.mean())
    per_outer_std = float(per_outer_arr.std())
    per_outer_min = float(per_outer_arr.min())
    per_outer_max = float(per_outer_arr.max())
    per_outer_median = float(np.median(per_outer_arr))

    # ---- BoB row-level aggregations across 5 outer-seed blends ----
    bob_mean_oof = per_outer_blend.mean(axis=0)
    bob_median_oof = np.median(per_outer_blend, axis=0)
    rae_bob_mean = float(rae(y_unb, bob_mean_oof))
    rae_bob_median = float(rae(y_unb, bob_median_oof))

    print("\n" + "=" * 78)
    print("OUTER-BAG SUMMARY")
    print("=" * 78)
    print(f"   per-outer blend RAE list = "
          f"[{', '.join(f'{r:.4f}' for r in per_outer_rae_blend)}]")
    print(f"   per-outer mean   = {per_outer_mean:.4f}")
    print(f"   per-outer std    = {per_outer_std:.4f}")
    print(f"   per-outer min    = {per_outer_min:.4f}")
    print(f"   per-outer max    = {per_outer_max:.4f}")
    print(f"   per-outer median = {per_outer_median:.4f}")
    print(f"   BoB MEAN   pooled RAE (mean   of 5 outer-blend vectors) = "
          f"{rae_bob_mean:.4f}")
    print(f"   BoB MEDIAN pooled RAE (median of 5 outer-blend vectors) = "
          f"{rae_bob_median:.4f}")

    # ---- Verdict ----
    delta_per_outer = per_outer_mean - NB1391_REF
    reproduces = abs(delta_per_outer) < REPRODUCE_MARGIN
    if reproduces:
        verdict = "NB1391_REPRODUCES"
    elif per_outer_mean < NB1391_REF - REPRODUCE_MARGIN:
        verdict = "NB1391_OPTIMISTIC_OUTER_BAG_BETTER"
    else:
        verdict = "NB1391_LUCKY_SEED_OUTER_BAG_WORSE"

    delta_outer0 = per_outer_rae_blend[0] - NB1391_REF
    outer0_reproduces = abs(delta_outer0) < REPRODUCE_MARGIN

    print("\n" + "-" * 78)
    print(f"VERDICT (per-outer-mean within {REPRODUCE_MARGIN} of nb1391 ref "
          f"{NB1391_REF:.4f}):")
    print(f"   per-outer-mean = {per_outer_mean:.4f}   "
          f"(d vs ref = {delta_per_outer:+.4f})")
    print(f"   outer=0 blend  = {per_outer_rae_blend[0]:.4f}   "
          f"(d vs ref = {delta_outer0:+.4f})   "
          f"outer0_reproduces = {outer0_reproduces}")
    print(f"   verdict = {verdict}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_bob_mean_oof.npy",
            bob_mean_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_bob_median_oof.npy",
            bob_median_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_bob_mean_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_bob_median_oof.npy'}")

    summary = {
        "tag": TAG,
        "n_unb": n_unb,
        "outer_seeds": OUTER_SEEDS,
        "inner_base_seeds": INNER_BASE_SEEDS,
        "per_outer_inner_seeds": per_outer_inner_seeds,
        "w_nb1373": W_NB1373,
        "w_nb1352": W_NB1352,
        "nb1373_outer_source": str(NB1373_OUTER_PATH.name),
        "nb1352_outer_source": str(NB1352_OUTER_PATH.name),
        "per_outer_rae_nb1373": per_outer_rae_nb1373,
        "per_outer_rae_nb1352": per_outer_rae_nb1352,
        "per_outer_rae_blend": per_outer_rae_blend,
        "per_outer_mean": per_outer_mean,
        "per_outer_std": per_outer_std,
        "per_outer_min": per_outer_min,
        "per_outer_max": per_outer_max,
        "per_outer_median": per_outer_median,
        "rae_bob_mean": rae_bob_mean,
        "rae_bob_median": rae_bob_median,
        "nb1391_ref": NB1391_REF,
        "reproduce_margin": REPRODUCE_MARGIN,
        "delta_per_outer_mean_vs_nb1391": delta_per_outer,
        "delta_outer0_vs_nb1391": delta_outer0,
        "outer0_reproduces": bool(outer0_reproduces),
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
        "n_unb", "outer_seeds",
        "w_nb1373", "w_nb1352",
        "per_outer_rae_nb1373",
        "per_outer_rae_nb1352",
        "per_outer_rae_blend",
        "per_outer_mean", "per_outer_std",
        "per_outer_min", "per_outer_max", "per_outer_median",
        "rae_bob_mean", "rae_bob_median",
        "delta_per_outer_mean_vs_nb1391",
        "delta_outer0_vs_nb1391",
        "outer0_reproduces", "reproduces",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
