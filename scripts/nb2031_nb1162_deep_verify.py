"""nb2031 -- DEEP 30-seed verification of nb1162 PRIMARY-1 AGGRESSIVE.

Per cycle 147 nb1190: 5-seed mean was 0.4215 (sub-noise gate miss); this
script extends to 30 FRESH kf_seeds {1011..1040}, disjoint from any prior
audit, to bound the cross-fit RAE with a 95% CI tight enough to decide
KEEP vs DEMOTE.

Pipeline (identical to nb1162_stack_pyramid.py):
    Stage 1 anchors (5 cached OOFs on the 253 unblind):
        nb2103_K28, chemprop_aux, nb730_honest, nb503, nb562
    Stage 2: SLSQP convex blend (w>=0, sum=1) per TRAIN fold
    Stage 3: rank-stretch grid {1.000, 1.025, ..., 1.150} per TRAIN fold
    OOF blend = (mu_tr + s_tr * (P_val @ w_tr - mu_tr)) gathered across folds
    Pooled scaffold-CV RAE = rae(y_unb, oof_blend)

For each fresh kf_seed in {1011..1040}:
    - Pooled scaffold-CV RAE under FULL 5-anchor pipeline
    - Pooled scaffold-CV RAE under DROP-nb730 pipeline (LB-faithful floor)

Reporting:
    - per-seed RAE
    - mean +/- std (population, ddof=0)
    - 95% CI on the mean via t-distribution (n=30, df=29)
    - Check: is 0.4206 (nb1162's claimed pooled CV RAE) inside the CI?
    - drop-nb730 mean +/- std (the LB-faithful floor)

Promotion / demotion logic:
    PROMOTE  (KEEP nb1162 PRIMARY-1) if:   mean <= 0.4250  AND  std <= 0.012
    DEMOTE   (revert to nb1191 PRIMARY-1) if:   mean > 0.4250  OR  std > 0.015
    HOLD     (no ladder change, manual review) otherwise

Outputs:
    scripts/nb2031_nb1162_deep_verify.py  (this file)
    data/processed/nb2031_summary.json
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
from scipy.optimize import minimize
from scipy import stats as scistats

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2031"
FRESH_SEEDS = list(range(1011, 1041))   # 30 seeds, disjoint from 1006..1010
N_FOLDS = 5
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]

NB1162_CLAIMED_RAE = 0.4206326222520904

# Promotion gates (from prompt)
PROMOTE_MEAN = 0.4250
PROMOTE_STD = 0.012
DEMOTE_MEAN = 0.4250
DEMOTE_STD = 0.015

# Full 5-anchor set (matches nb1162_stack_pyramid.py)
ANCHORS_FULL = [
    ("nb2103_K28",   "nb2103_mean_bag_oof_K28.npy"),
    ("chemprop_aux", "nb1133_chemprop_aux_pred_oof.npy"),
    ("nb730_honest", "nb730_honest_pred_oof.npy"),
    ("nb503",        "nb503_pred_oof.npy"),
    ("nb562",        "nb562_pred_oof.npy"),
]
# Drop-nb730 -- the LB-faithful floor (PRE-unblind clean only).
ANCHORS_DROP = [a for a in ANCHORS_FULL if a[0] != "nb730_honest"]


def slsqp_simplex(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    K = P.shape[1]
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * K
    res = minimize(
        lambda w: float(np.sum((P @ w - y) ** 2)),
        np.full(K, 1.0 / K),
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
        options={"ftol": 1e-10, "maxiter": 1000},
    )
    w = np.clip(res.x, 0.0, 1.0)
    s = w.sum()
    return w / s if s > 0 else np.full(K, 1.0 / K)


def best_stretch_on(blend_tr: np.ndarray, y_tr: np.ndarray,
                    mu: float, grid) -> tuple[float, float]:
    best_s, best_r = 1.0, float("inf")
    for s in grid:
        pred = mu + s * (blend_tr - mu)
        r = float(rae(y_tr, pred))
        if r < best_r:
            best_r = r
            best_s = float(s)
    return best_s, best_r


def stack_one_seed(P: np.ndarray, y: np.ndarray,
                   scaffolds: list, seed: int) -> dict:
    """Run nb1162's full SLSQP+stretch scaffold-CV at one kf_seed."""
    splits = scaffold_kfold_indices(
        scaffolds, n_splits=N_FOLDS, shuffle=True, seed=seed,
    )
    oof = np.full(len(y), np.nan)
    fold_s = []
    fold_w = []
    for tr, va in splits:
        w_f = slsqp_simplex(P[tr], y[tr])
        blend_tr = P[tr] @ w_f
        mu_tr = float(blend_tr.mean())
        s_f, _ = best_stretch_on(blend_tr, y[tr], mu_tr, STRETCH_GRID)
        blend_va = P[va] @ w_f
        oof[va] = mu_tr + s_f * (blend_va - mu_tr)
        fold_s.append(float(s_f))
        fold_w.append([float(x) for x in w_f])
    return {
        "seed": int(seed),
        "rae": float(rae(y, oof)),
        "mean_s": float(np.mean(fold_s)),
        "fold_s": fold_s,
        "fold_w_max": float(np.max([np.max(w) for w in fold_w])),
    }


def load_anchor_matrix(anchor_list, n_unb):
    cols = []
    indiv = {}
    for disp, oof_rel in anchor_list:
        p = DATA_PROCESSED / oof_rel
        assert p.exists(), f"missing OOF: {p}"
        a = np.load(p).astype(np.float64)
        assert a.shape == (n_unb,), f"{disp} shape {a.shape}"
        cols.append(a)
        indiv[disp] = float("nan")
    return np.column_stack(cols), indiv


def t_ci_95(x: np.ndarray) -> tuple[float, float, float]:
    """Two-sided 95% CI on the mean via Student's t (df = n-1)."""
    n = len(x)
    m = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(n))   # sample-SD SE
    if se == 0.0 or n < 2:
        return m, m, m
    tcrit = float(scistats.t.ppf(0.975, df=n - 1))
    half = tcrit * se
    return m - half, m, m + half


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- DEEP 30-seed verification of nb1162 PRIMARY-1")
    print(f"         fresh kf_seeds {FRESH_SEEDS[0]}..{FRESH_SEEDS[-1]}  "
          f"(n={len(FRESH_SEEDS)})")
    print("=" * 78)

    # ---- Load 513 + 253 substrate ----
    te = load_test()
    te_smiles = te["smiles"].values
    n_te = len(te_smiles)

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    unb_smiles = te_smiles[unb_idx]
    scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in scaffolds if s})
    n_singleton = sum(1 for s in scaffolds if not s)
    print(f"[load] n_te={n_te}  n_unb={n_unb}  "
          f"unique_scaffolds={n_unique_scaf}  singleton={n_singleton}")

    # ---- Load anchor matrices ----
    P_full, indiv_full = load_anchor_matrix(ANCHORS_FULL, n_unb)
    for (disp, _), col in zip(ANCHORS_FULL, P_full.T):
        indiv_full[disp] = float(rae(y_unb, col))
    print("\n[anchors-FULL]")
    for disp, _ in ANCHORS_FULL:
        print(f"   {disp:14s}  oof_RAE={indiv_full[disp]:.4f}")

    P_drop, indiv_drop = load_anchor_matrix(ANCHORS_DROP, n_unb)
    for (disp, _), col in zip(ANCHORS_DROP, P_drop.T):
        indiv_drop[disp] = float(rae(y_unb, col))
    print("\n[anchors-DROP-nb730]  (LB-faithful floor)")
    for disp, _ in ANCHORS_DROP:
        print(f"   {disp:14s}  oof_RAE={indiv_drop[disp]:.4f}")

    # ============================================================
    # 30-seed FULL pipeline
    # ============================================================
    print("\n" + "-" * 78)
    print(f"FULL 5-anchor stack across {len(FRESH_SEEDS)} fresh kf_seeds")
    print("-" * 78)
    per_seed_full = []
    for sd in FRESH_SEEDS:
        rec = stack_one_seed(P_full, y_unb, scaffolds, sd)
        per_seed_full.append(rec)
        print(f"   seed={sd}: RAE={rec['rae']:.4f}  "
              f"mean_s={rec['mean_s']:.3f}  max_w={rec['fold_w_max']:.3f}")

    rae_full = np.array([d["rae"] for d in per_seed_full], dtype=np.float64)
    mean_full = float(rae_full.mean())
    std_full = float(rae_full.std(ddof=0))   # population SD per prompt
    std_full_sample = float(rae_full.std(ddof=1))
    min_full = float(rae_full.min())
    max_full = float(rae_full.max())
    ci_lo, ci_mid, ci_hi = t_ci_95(rae_full)
    claim_in_ci = bool(ci_lo <= NB1162_CLAIMED_RAE <= ci_hi)

    print(f"\n[FULL n={len(rae_full)}]"
          f"  mean={mean_full:.4f}  std(pop)={std_full:.4f}  "
          f"std(sample)={std_full_sample:.4f}")
    print(f"        range=[{min_full:.4f}, {max_full:.4f}]")
    print(f"        95% CI on mean = [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"        nb1162 claimed RAE = {NB1162_CLAIMED_RAE:.4f}  "
          f"-> inside CI? {claim_in_ci}")

    # ============================================================
    # 30-seed DROP-nb730 pipeline (LB-faithful floor)
    # ============================================================
    print("\n" + "-" * 78)
    print(f"DROP-nb730 stack across {len(FRESH_SEEDS)} fresh kf_seeds "
          "(LB-faithful floor)")
    print("-" * 78)
    per_seed_drop = []
    for sd in FRESH_SEEDS:
        rec = stack_one_seed(P_drop, y_unb, scaffolds, sd)
        per_seed_drop.append(rec)
        print(f"   seed={sd}: RAE={rec['rae']:.4f}  "
              f"mean_s={rec['mean_s']:.3f}  max_w={rec['fold_w_max']:.3f}")

    rae_drop = np.array([d["rae"] for d in per_seed_drop], dtype=np.float64)
    mean_drop = float(rae_drop.mean())
    std_drop = float(rae_drop.std(ddof=0))
    min_drop = float(rae_drop.min())
    max_drop = float(rae_drop.max())
    ci_lo_d, _, ci_hi_d = t_ci_95(rae_drop)
    print(f"\n[DROP n={len(rae_drop)}]"
          f"  mean={mean_drop:.4f}  std(pop)={std_drop:.4f}")
    print(f"        range=[{min_drop:.4f}, {max_drop:.4f}]")
    print(f"        95% CI on mean = [{ci_lo_d:.4f}, {ci_hi_d:.4f}]")
    print(f"        delta(drop - full) = +{mean_drop - mean_full:.4f}")

    # ============================================================
    # PROMOTE / DEMOTE / HOLD decision
    # ============================================================
    print("\n" + "-" * 78)
    print("DECISION  (gates apply to FULL pipeline mean/std)")
    print("-" * 78)
    promote = (mean_full <= PROMOTE_MEAN) and (std_full <= PROMOTE_STD)
    demote = (mean_full > DEMOTE_MEAN) or (std_full > DEMOTE_STD)

    if promote:
        ladder_action = "PROMOTE_KEEP_nb1162_PRIMARY1_CONFIRMED"
    elif demote:
        ladder_action = "DEMOTE_REVERT_TO_nb1191_PRIMARY1"
    else:
        ladder_action = "HOLD_NO_LADDER_CHANGE"

    print(f"   promote rule:  mean<={PROMOTE_MEAN} AND std<={PROMOTE_STD}  "
          f"-> {promote}  (mean={mean_full:.4f}, std={std_full:.4f})")
    print(f"   demote rule:   mean>{DEMOTE_MEAN}  OR  std>{DEMOTE_STD}    "
          f"-> {demote}")
    print(f"   95% CI contains claimed 0.4206:  {claim_in_ci}")
    print(f"   LADDER ACTION = {ladder_action}")

    summary = {
        "tag": TAG,
        "method": "nb1162_deep_verification_30fresh_seeds_with_drop_nb730_floor",
        "audit_target": "nb1162_stack_pyramid_PRIMARY1_AGGRESSIVE_0p4206",
        "rerun_of": "nb1190_5seed_audit_mean_0p4215",
        "fresh_seeds": FRESH_SEEDS,
        "n_seeds": len(FRESH_SEEDS),
        "n_folds": N_FOLDS,
        "stretch_grid": STRETCH_GRID,
        "n_unb": n_unb,
        "n_te": n_te,
        "n_unique_scaffolds": n_unique_scaf,
        "n_singleton_scaffolds": n_singleton,
        "nb1162_claimed_pooled_cv_rae": NB1162_CLAIMED_RAE,
        "anchors_full": [a[0] for a in ANCHORS_FULL],
        "anchors_drop_nb730": [a[0] for a in ANCHORS_DROP],
        "indiv_anchor_oof_rae_full": indiv_full,
        "indiv_anchor_oof_rae_drop": indiv_drop,

        # FULL pipeline stats
        "full_per_seed": per_seed_full,
        "full_mean_rae": mean_full,
        "full_std_rae_population": std_full,
        "full_std_rae_sample": std_full_sample,
        "full_min_rae": min_full,
        "full_max_rae": max_full,
        "full_ci95_lo": ci_lo,
        "full_ci95_hi": ci_hi,
        "full_ci95_width": ci_hi - ci_lo,
        "full_claim_inside_ci95": claim_in_ci,
        "full_delta_vs_claimed": mean_full - NB1162_CLAIMED_RAE,

        # DROP-nb730 stats (LB-faithful floor)
        "drop_nb730_per_seed": per_seed_drop,
        "drop_nb730_mean_rae": mean_drop,
        "drop_nb730_std_rae_population": std_drop,
        "drop_nb730_min_rae": min_drop,
        "drop_nb730_max_rae": max_drop,
        "drop_nb730_ci95_lo": ci_lo_d,
        "drop_nb730_ci95_hi": ci_hi_d,
        "drop_nb730_delta_vs_full": mean_drop - mean_full,

        # Decision
        "promote_mean_target": PROMOTE_MEAN,
        "promote_std_target": PROMOTE_STD,
        "demote_mean_threshold": DEMOTE_MEAN,
        "demote_std_threshold": DEMOTE_STD,
        "promote": bool(promote),
        "demote": bool(demote),
        "ladder_action": ladder_action,
        "wall_sec": round(time.time() - t0, 2),
    }
    out = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   FULL  n={len(rae_full)}  mean={mean_full:.4f} +/- "
          f"{std_full:.4f}  95% CI=[{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"   DROP  n={len(rae_drop)}  mean={mean_drop:.4f} +/- "
          f"{std_drop:.4f}")
    print(f"   claim 0.4206 inside CI: {claim_in_ci}")
    print(f"   action = {ladder_action}")
    print(f"   wall   = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    main()
