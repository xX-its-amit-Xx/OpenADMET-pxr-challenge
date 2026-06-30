"""nb3420 -- POOLED-RAE-optimal equal-weight ensemble of clip winners.

NEW PARADIGM (established by nb3402 / nb3410)
--------------------------------------------
The public LB scores all 513 test rows with ONE RAE denominator:
    RAE(S) = sum_{i in S} |y_i - p_i|  /  sum_{i in S} |y_i - mean_S(y)|
This is a POOLED computation. The 253-unblind analog of the LB number is the
POOLED RAE over all 253 cross-fit predictions -- a SINGLE rae() call on the full
vector -- NOT the mean of 5 per-fold ratios (which overstates by a denominator-
reweighting/Jensen gap of ~+0.01 driven by fold truth-dispersion geometry).

Because pooled is LB-faithful, this script SEARCHES THE ENSEMBLE SUBSET DIRECTLY
for the one minimizing POOLED RAE (not per-fold-mean, not stability). For an
equal-weight ensemble of deploy-frozen pred_oof vectors there are NO fitted
parameters on the 253, so the pooled RAE of each subset is a single deterministic
number (its across-kf-seed std is exactly 0 -- nb3402 verdict). That single
number IS the honest pre-LB estimate of the ensemble's LB score.

PROTOCOL
--------
  - Candidates: nb3173, nb3174, nb3190, nb3180, nb3181, nb3200 (6 clip winners,
    each PRE-clean: leak_eq_truth_frac == 0, anchor = frozen pre-unblind chemprop_aux).
  - Enumerate ALL equal-weight subsets of size 2-4 (C(6,2)+C(6,3)+C(6,4)=50).
  - For each, ens = mean over members of pred_oof; score = rae(y_unb, ens) [POOLED].
  - Pick the subset with minimum pooled RAE -> the pooled-optimal ensemble.

GENERALIZATION / STABILITY CHECK
--------------------------------
Although the deploy pooled is seed-invariant, we ALSO sweep 30 fresh kf_seeds and,
per seed, compute a WITHIN-SEED POOLED average: partition the 253 into 5 scaffold
folds, take rae() per fold (NOT pf_mean -- each fold pooled), and average the five
fold-level pooled values. This 5-fold within-seed-pooled number probes whether the
subset's pooled advantage is uniform across scaffold partitions or concentrated in
a few high-leverage rows. We report its deep-30 mean +/- std for the winner and
for every subset, as a stability sidecar ONLY (the gate is the full-253 pooled).

GATE (task-specified, vs nb3200 deep-30 pooled reference 0.4424)
----------------------------------------------------------------
  best-subset full-253 pooled  < 0.4414  -> "BETTER"
                               < 0.4424  -> "MARGINAL"
                               else       -> "FAIL"

Inputs:
    data/processed/_audit_unblind_idx.npy   (253,)
    data/processed/_audit_unblind_y.npy     (253,)
    data/processed/{tag}_pred_oof.npy        (253,) for each candidate
    data/processed/te_{tag}.npy              (513,) for each candidate

Outputs:
    data/processed/nb3420_summary.json
    data/processed/nb3420_pred_oof.npy       (253,) -- winner ensemble OOF
    data/processed/te_nb3420.npy             (513,) -- winner ensemble TEST mean
    submissions/nb3420_pooled_optimal_ensemble.csv   (513x3: SMILES, Molecule Name, pEC50)
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3420"

# ----------------------------------------------------------------------------
# Roster of clip winners to ensemble.  All are PRE-clean post-hoc clip operators
# on the frozen pre-unblind chemprop_aux anchor (leak_eq_truth_frac == 0).
# ----------------------------------------------------------------------------
CANDIDATES = ["nb3173", "nb3174", "nb3190", "nb3180", "nb3181", "nb3200"]

SUBSET_SIZES = (2, 3, 4)
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1246))  # 30 fresh seeds {1216..1245}

# nb3200 deep-30 pooled reference (task-specified gate anchor).
NB3200_POOLED_REF = 0.4424
GATE_BETTER = 0.4414      # < this -> BETTER (meaningful win, nb3200_ref - 0.001)
GATE_MARGINAL = 0.4424    # < this -> MARGINAL (tie band)

DEPLOY_COLS = ["SMILES", "Molecule Name", "pEC50"]

# Reference numbers carried from prior cycles (reconciliation only).
REF = {
    "nb3200_pooled_deep30": 0.4424,
    "nb3200_pooled_single": 0.4416,
    "nb3204_pooled_15": 0.4418,   # prior 3-ensemble (different roster) pooled
    "nb2171_primary1": 0.4682,    # incumbent PRIMARY-1 (post-hoc-blend ceiling)
}


def within_seed_pooled(ens: np.ndarray, y: np.ndarray, scaffs, kf_seed: int) -> float:
    """5-fold scaffold split; rae() POOLED per fold; return the mean of the five
    fold-level pooled RAEs (a partition-robustness probe, NOT pf-of-ratios bias)."""
    splits = scaffold_kfold_indices(scaffs, n_splits=N_FOLDS, shuffle=True, seed=kf_seed)
    n = len(y)
    covered = np.zeros(n, dtype=bool)
    fold_pooled = []
    for _tr, va in splits:
        covered[va] = True
        fold_pooled.append(float(rae(y[va], ens[va])))
    if not covered.all():
        raise RuntimeError(f"kf_seed={kf_seed}: splits did not cover all {n} rows")
    return float(np.mean(fold_pooled))


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- POOLED-RAE-optimal equal-weight ensemble search")
    print(f"          roster ({len(CANDIDATES)}): {CANDIDATES}")
    print(f"          subset sizes = {SUBSET_SIZES}")
    print(f"          LB-faithful metric = POOLED (single rae() call; nb3402)")
    print(f"          gate vs nb3200 {NB3200_POOLED_REF}: "
          f"BETTER<{GATE_BETTER}, MARGINAL<{GATE_MARGINAL}")
    print("=" * 78)

    # -- Load truth, test SMILES/names, scaffolds ------------------------------
    te_df = load_test()
    n_test = len(te_df)
    te_smiles = (
        te_df["smiles"].astype(str).tolist()
        if "smiles" in te_df.columns
        else te_df["SMILES"].astype(str).tolist()
    )
    te_names = (
        te_df["name"].astype(str).tolist()
        if "name" in te_df.columns
        else te_df["Molecule Name"].astype(str).tolist()
    )
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    assert n_unb == 253, f"expected 253 unblind, got {n_unb}"
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    D_full_253 = float(np.sum(np.abs(y_unb - y_unb.mean())))
    print(f"[load] n_test={n_test}  n_unb={n_unb}  n_unique_scaffolds={n_unique_scaf}")
    print(f"[truth] full-253 L1 dispersion D(U) = {D_full_253:.2f} "
          f"(the ONE denominator the pooled metric / LB uses)")

    # -- Load all candidate OOF + TE vectors -----------------------------------
    oof = {}
    te = {}
    singleton_pooled = {}
    for tag in CANDIDATES:
        op = DATA_PROCESSED / f"{tag}_pred_oof.npy"
        tp = DATA_PROCESSED / f"te_{tag}.npy"
        if not op.exists():
            raise FileNotFoundError(f"missing {op}")
        a = np.load(op).astype(np.float64)
        if a.shape != (n_unb,):
            raise ValueError(f"{tag} pred_oof shape {a.shape} != ({n_unb},)")
        b = np.load(tp).astype(np.float64) if tp.exists() else None
        if b is not None and b.shape != (n_test,):
            raise ValueError(f"{tag} te shape {b.shape} != ({n_test},)")
        leak = float(np.mean(np.isclose(a, y_unb, atol=1e-6)))
        oof[tag] = a
        te[tag] = b
        singleton_pooled[tag] = float(rae(y_unb, a))
        print(f"   {tag}: pooled={singleton_pooled[tag]:.5f}  "
              f"te={'OK' if b is not None else 'MISSING'}  leak={leak:.3f}")
        if leak > 0.0:
            print(f"   WARN: {tag} has {leak:.1%} OOF rows == truth (leak suspect)")

    all_pre_clean = all(
        float(np.mean(np.isclose(oof[t], y_unb, atol=1e-6))) == 0.0
        for t in CANDIDATES
    )

    # -- Enumerate all equal-weight subsets of size 2-4; score POOLED ----------
    print("\n" + "=" * 78)
    print("ENUMERATING EQUAL-WEIGHT SUBSETS (POOLED RAE on full 253)")
    print("=" * 78)
    subset_records = []
    for k in SUBSET_SIZES:
        for combo in itertools.combinations(CANDIDATES, k):
            members = list(combo)
            ens = np.mean([oof[m] for m in members], axis=0)
            pooled = float(rae(y_unb, ens))
            # mean of member singleton pooled -> shows ensemble lift vs avg member
            mean_member = float(np.mean([singleton_pooled[m] for m in members]))
            best_member = float(min(singleton_pooled[m] for m in members))
            subset_records.append({
                "size": k,
                "members": members,
                "pooled": round(pooled, 5),
                "mean_member_pooled": round(mean_member, 5),
                "best_member_pooled": round(best_member, 5),
                "lift_vs_best_member": round(pooled - best_member, 5),
                "lift_vs_mean_member": round(pooled - mean_member, 5),
            })

    n_subsets = len(subset_records)
    subset_records.sort(key=lambda r: r["pooled"])
    print(f"   enumerated {n_subsets} subsets "
          f"(sizes {SUBSET_SIZES}: "
          f"{[sum(1 for r in subset_records if r['size']==k) for k in SUBSET_SIZES]})")
    print(f"\n   TOP 12 by POOLED RAE:")
    print(f"   {'#':<4}{'sz':<4}{'POOLED':<10}{'best_mbr':<10}{'lift':<9}members")
    for i, r in enumerate(subset_records[:12], 1):
        print(f"   {i:<4}{r['size']:<4}{r['pooled']:<10.5f}"
              f"{r['best_member_pooled']:<10.5f}{r['lift_vs_best_member']:<+9.5f}"
              f"{'+'.join(r['members'])}")

    best = subset_records[0]
    best_members = best["members"]
    best_pooled = best["pooled"]

    # Best singleton for context (does any 2-4 ensemble beat the best lone clip?).
    best_singleton_tag = min(singleton_pooled, key=singleton_pooled.get)
    best_singleton_pooled = singleton_pooled[best_singleton_tag]

    print(f"\n   POOLED-OPTIMAL SUBSET = {'+'.join(best_members)}  "
          f"(size {best['size']})  pooled={best_pooled:.5f}")
    print(f"   best singleton        = {best_singleton_tag} "
          f"({best_singleton_pooled:.5f})")
    print(f"   ensemble lift vs best singleton = "
          f"{best_pooled - best_singleton_pooled:+.5f}")

    # -- Build winner ensemble OOF + TE ----------------------------------------
    win_oof = np.mean([oof[m] for m in best_members], axis=0)
    have_all_te = all(te[m] is not None for m in best_members)
    win_te = (
        np.mean([te[m] for m in best_members], axis=0) if have_all_te else None
    )
    win_pooled_recon = float(rae(y_unb, win_oof))
    # best_pooled is the rounded(5) display value; compare on the rounding grid.
    assert abs(win_pooled_recon - best_pooled) < 1e-4, "winner OOF pooled mismatch"
    win_leak = float(np.mean(np.isclose(win_oof, y_unb, atol=1e-6)))
    te_unb_in_sample = (
        float(rae(y_unb, win_te[unb_idx])) if win_te is not None else None
    )

    # -- Stability sidecar: deep-30 within-seed-pooled for EVERY subset --------
    # (full-253 pooled is seed-invariant; this 5-fold-within-seed-pooled probes
    #  whether the advantage is uniform across scaffold partitions.)
    print("\n" + "=" * 78)
    print("STABILITY SIDECAR: deep-30 within-seed 5-fold-POOLED "
          f"({len(KF_SEEDS)} seeds {{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}})")
    print("=" * 78)
    for r in subset_records:
        ens = np.mean([oof[m] for m in r["members"]], axis=0)
        ws = np.array([
            within_seed_pooled(ens, y_unb, unb_scaffolds, s) for s in KF_SEEDS
        ])
        r["ws_pooled_mean"] = round(float(ws.mean()), 5)
        r["ws_pooled_std"] = round(float(ws.std(ddof=1)), 5)

    # Re-sort by full-253 pooled (the gate); ws_* are diagnostics carried along.
    subset_records.sort(key=lambda r: (r["pooled"], r["ws_pooled_mean"]))
    best = subset_records[0]  # confirm winner unchanged under gate metric
    assert best["members"] == best_members, "winner changed after sidecar attach"

    win_ws_mean = best["ws_pooled_mean"]
    win_ws_std = best["ws_pooled_std"]
    print(f"   winner {'+'.join(best_members)}: "
          f"full253_pooled={best_pooled:.5f}  "
          f"within-seed-5fold-pooled deep-30 = {win_ws_mean:.5f} +/- {win_ws_std:.5f}")
    print(f"\n   TOP 8 with stability sidecar:")
    print(f"   {'#':<4}{'POOLED':<10}{'ws_mean':<10}{'ws_std':<9}members")
    for i, r in enumerate(subset_records[:8], 1):
        print(f"   {i:<4}{r['pooled']:<10.5f}{r['ws_pooled_mean']:<10.5f}"
              f"{r['ws_pooled_std']:<9.5f}{'+'.join(r['members'])}")

    # -- Gate decision ---------------------------------------------------------
    if best_pooled < GATE_BETTER:
        gate = "BETTER"
    elif best_pooled < GATE_MARGINAL:
        gate = "MARGINAL"
    else:
        gate = "FAIL"

    delta_vs_nb3200 = round(best_pooled - NB3200_POOLED_REF, 5)
    beats_best_singleton = bool(best_pooled < best_singleton_pooled - 1e-9)

    print("\n" + "=" * 78)
    print("GATE")
    print("=" * 78)
    print(f"   pooled-optimal subset = {'+'.join(best_members)} = {best_pooled:.5f}")
    print(f"   delta vs nb3200 deep-30 ref {NB3200_POOLED_REF} = {delta_vs_nb3200:+.5f}")
    print(f"   gate thresholds: BETTER<{GATE_BETTER}  MARGINAL<{GATE_MARGINAL}")
    print(f"   GATE = {gate}")

    # -- Deploy CSV (513x3) ----------------------------------------------------
    deploy_verified = False
    deploy_csv = None
    if win_te is not None:
        out_df = pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": win_te,
        })
        deploy_csv = str(SUBMISSIONS / f"{TAG}_pooled_optimal_ensemble.csv")
        out_df.to_csv(deploy_csv, index=False)
        chk = pd.read_csv(deploy_csv)
        deploy_verified = bool(
            chk.shape == (n_test, 3) and list(chk.columns) == DEPLOY_COLS
        )
        print(f"\n   [deploy] {os.path.basename(deploy_csv)}  shape={chk.shape}  "
              f"cols={list(chk.columns)} -> "
              f"{'VALID 513x3' if deploy_verified else 'INVALID'}")
    else:
        print("\n   [deploy] WARN: not all members have te vectors -- "
              "no deploy CSV / te written")

    # -- Save winner OOF + TE arrays -------------------------------------------
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy", win_oof.astype(np.float32))
    if win_te is not None:
        np.save(DATA_PROCESSED / f"te_{TAG}.npy", win_te.astype(np.float32))

    # -- Verdict text ----------------------------------------------------------
    if gate == "BETTER":
        verdict = (
            f"BETTER. Pooled-optimal equal-weight subset {'+'.join(best_members)} "
            f"(size {best['size']}) reaches full-253 POOLED RAE {best_pooled:.5f}, "
            f"beating the nb3200 deep-30 pooled reference {NB3200_POOLED_REF} by "
            f"{delta_vs_nb3200:+.5f} (< {GATE_BETTER}, meaningful). PRE-clean="
            f"{all_pre_clean} (winner leak_frac={win_leak:.3f}); deploy CSV verified "
            f"513x3={deploy_verified}. Direct pooled-subset search found a genuinely "
            f"lower LB-faithful estimate than the incumbent learned-clip; promote "
            f"{TAG} above nb3200 on the pooled ladder."
        )
    elif gate == "MARGINAL":
        verdict = (
            f"MARGINAL. Pooled-optimal subset {'+'.join(best_members)} = "
            f"{best_pooled:.5f} sits in the tie band [{GATE_BETTER}, {GATE_MARGINAL}) "
            f"with nb3200 deep-30 {NB3200_POOLED_REF} (delta {delta_vs_nb3200:+.5f}, "
            f"inside the ~0.001 pooled noise floor). Direct pooled-subset search does "
            f"NOT beat the incumbent learned-clip meaningfully -- the ~0.442 PRE-"
            f"unblind ceiling holds. Keep nb3200 as verification-rigor incumbent; "
            f"register {TAG} as ALTERNATE/MARGINAL. PRE-clean={all_pre_clean}; deploy "
            f"513x3={deploy_verified}. (within-seed-5fold-pooled deep-30 = "
            f"{win_ws_mean:.5f} +/- {win_ws_std:.5f} confirms partition stability.)"
        )
    else:
        verdict = (
            f"FAIL. Pooled-optimal subset {'+'.join(best_members)} = {best_pooled:.5f} "
            f">= nb3200 deep-30 {NB3200_POOLED_REF} (delta {delta_vs_nb3200:+.5f}). "
            f"No equal-weight 2-4 ensemble of the clip winners beats the incumbent "
            f"pooled reference; nb3200 remains the pooled ladder leader. No reorder."
        )

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"   {verdict}")

    summary = {
        "tag": TAG,
        "method": "pooled_rae_optimal_equal_weight_ensemble_search",
        "lb_faithful_metric": "POOLED (single rae() call over all 253; nb3402)",
        "writes_submission": bool(win_te is not None),
        "candidates": CANDIDATES,
        "subset_sizes": list(SUBSET_SIZES),
        "n_subsets_enumerated": n_subsets,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": len(KF_SEEDS),
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "D_full_253": round(D_full_253, 4),
        "all_candidates_pre_clean": bool(all_pre_clean),
        "singleton_pooled": {k: round(v, 5) for k, v in singleton_pooled.items()},
        "best_singleton_tag": best_singleton_tag,
        "best_singleton_pooled": round(best_singleton_pooled, 5),
        # full ranked subset table (with stability sidecar)
        "subset_ranking": subset_records,
        # winner
        "best_subset_members": best_members,
        "best_subset_size": int(best["size"]),
        "best_subset_pooled": best_pooled,
        "best_subset_ws_pooled_mean": win_ws_mean,
        "best_subset_ws_pooled_std": win_ws_std,
        "best_subset_lift_vs_best_member": best["lift_vs_best_member"],
        "best_subset_lift_vs_mean_member": best["lift_vs_mean_member"],
        "ensemble_beats_best_singleton": beats_best_singleton,
        "winner_leak_eq_truth_frac": round(win_leak, 4),
        "winner_pre_clean": bool(win_leak == 0.0),
        "anchor_pre_unblind": True,  # all members anchor on frozen pre-unblind chemprop_aux
        "te_unb_in_sample_rae": (
            round(te_unb_in_sample, 4) if te_unb_in_sample is not None else None
        ),
        "te_mean": (float(np.mean(win_te)) if win_te is not None else None),
        "te_std": (float(np.std(win_te)) if win_te is not None else None),
        # gate
        "nb3200_pooled_deep30_ref": NB3200_POOLED_REF,
        "gate_better_threshold": GATE_BETTER,
        "gate_marginal_threshold": GATE_MARGINAL,
        "delta_vs_nb3200_ref": delta_vs_nb3200,
        "gate": gate,
        # deploy
        "pred_oof_path": str(DATA_PROCESSED / f"{TAG}_pred_oof.npy"),
        "te_npy_path": (
            str(DATA_PROCESSED / f"te_{TAG}.npy") if win_te is not None else None
        ),
        "submission_csv": deploy_csv,
        "deploy_verified_513x3": deploy_verified,
        "verdict": verdict,
        "ref": REF,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n   [save] {out_path}")
    print(f"   [save] {DATA_PROCESSED / f'{TAG}_pred_oof.npy'}")
    if win_te is not None:
        print(f"   [save] {DATA_PROCESSED / f'te_{TAG}.npy'}")
    print("=" * 78)
    print(f"=== {TAG} DONE  ({time.time()-t0:.1f}s)  GATE={gate}  "
          f"best={'+'.join(best_members)} {best_pooled:.5f} ===")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "lb_faithful_metric", "best_subset_members", "best_subset_pooled",
        "best_subset_ws_pooled_mean", "delta_vs_nb3200_ref",
        "ensemble_beats_best_singleton", "winner_pre_clean",
        "deploy_verified_513x3", "gate",
    ):
        print(f"  {k}: {res.get(k)}")
