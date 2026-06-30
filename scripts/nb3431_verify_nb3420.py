"""nb3431 -- SELECTION-VALIDATED cross-fit verification of nb3420.

CONTEXT
-------
nb3420 enumerated ALL 50 equal-weight subsets (sizes 2-4) of the 6 clip winners
{nb3173, nb3174, nb3190, nb3180, nb3181, nb3200} and picked the subset with the
minimum POOLED RAE *on the full 253 unblind*:
    winner = {nb3174, nb3190, nb3180, nb3200}  ->  pooled 0.44130  (gate BETTER).

That 0.44130 is an IN-SAMPLE subset-selection number: the same 253 rows used to
SCORE the ensemble were also used to CHOOSE which of 50 subsets to deploy. With
50 candidate subsets clustered in a ~0.001 RAE band, the global-min pick is biased
DOWN by selection (the search greedily grabs whichever subset's pooled happens to
ride the favorable side of the 253's particular noise realization). This script
asks: does the pooled-optimal-subset *procedure* survive honest cross-validation,
or is 0.44130 selection-bias?

WHY THIS IS THE RIGHT TEST
--------------------------
Each candidate {tag}_pred_oof.npy is a DEPLOY-FROZEN vector (the clip bounds were
already fit per-fold upstream; at this stage equal-weight averaging adds NO fitted
parameter on the 253). The ONLY thing fit on the 253 in nb3420 is the discrete
SUBSET CHOICE. So the honest analog is NESTED subset selection:

  Outer 5-fold scaffold CV. Within each outer fold:
    1. SEARCH the best subset on outer-TRAIN pooled RAE (enumerate the same 50).
    2. APPLY that train-selected subset (equal-weight mean) to outer-VAL rows.
  Stitch the 5 outer-VAL prediction blocks -> a full-253 OOF vector whose subset
  for every row was chosen WITHOUT seeing that row's truth.
  Score: ONE pooled rae() over the stitched 253 (LB-faithful aggregation, nb3402).

If the subset advantage is real structure, the per-fold-selected subset transfers
and the stitched pooled stays ~0.4413. If 0.44130 was selection-bias, the nested
pooled degrades toward the singleton/best-member band (~0.4416-0.4426), because
the train-optimal subset on a different partition is a near-random draw from the
0.001-wide cluster and does not lower VAL pooled.

15 FRESH kf_seeds {1431..1445} (disjoint from nb3420's {1216..1245}) -> deep-15
mean +/- std of the selection-validated pooled.

DIAGNOSTIC SIDECARS (not gated)
-------------------------------
  * fixed_winner_pooled: apply nb3420's full-253 winner subset
    {nb3174,nb3190,nb3180,nb3200} directly to every outer-VAL (NO per-fold search).
    This isolates the bias attributable purely to per-fold SELECTION vs the fixed
    deploy ensemble. (Its full-253 single-shot value is exactly nb3420's 0.44130.)
  * winner_select_freq: fraction of the 15*5 = 75 outer folds whose train-optimal
    subset equals the nb3420 winner subset (procedure stability).
  * best_singleton_pooled (nb3200 = 0.44157) carried for reference.

GATE (task-specified, vs nb3420 in-sample 0.44130)
--------------------------------------------------
  selection-validated pooled (deep-15 mean)  < 0.4414  -> "VERIFIED"
                                              < 0.4424  -> "MARGINAL"
                                              else       -> "SELECTION_BIAS"

Inputs:
    data/processed/_audit_unblind_idx.npy   (253,)
    data/processed/_audit_unblind_y.npy     (253,)
    data/processed/{tag}_pred_oof.npy        (253,) for each candidate
    data/processed/te_{tag}.npy              (513,) for each candidate

Outputs:
    data/processed/nb3431_summary.json
    data/processed/nb3431_pred_oof.npy   (253,) -- selection-validated OOF at the
                                          MEDIAN-pooled seed (representative cross-fit)
    data/processed/te_nb3431.npy         (513,) -- deploy TEST mean of whichever
                                          subset the FULL-253 search selects (= the
                                          nb3420 winner subset; carried for ladder)
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
from pxr.paths import DATA_PROCESSED

TAG = "nb3431"

# Same 6-candidate roster nb3420 searched.
CANDIDATES = ["nb3173", "nb3174", "nb3190", "nb3180", "nb3181", "nb3200"]
SUBSET_SIZES = (2, 3, 4)
N_FOLDS = 5
KF_SEEDS = list(range(1431, 1446))  # 15 FRESH seeds {1431..1445}

# nb3420 in-sample (full-253 global-min) result -- the number under test.
NB3420_INSAMPLE_POOLED = 0.44130
NB3420_WINNER_SUBSET = ("nb3174", "nb3190", "nb3180", "nb3200")
NB3200_SINGLETON_POOLED = 0.44157  # best lone clip (reference floor)

# Task gate thresholds.
GATE_VERIFIED = 0.4414
GATE_MARGINAL = 0.4424

# All 50 subsets, precomputed once (frozen tuple order).
ALL_SUBSETS = [
    combo
    for k in SUBSET_SIZES
    for combo in itertools.combinations(CANDIDATES, k)
]


def best_subset_on(idx: np.ndarray, oof: dict, y: np.ndarray) -> tuple[tuple, float]:
    """Enumerate the 50 equal-weight subsets; return (members, pooled) minimizing
    POOLED rae() over the given index set `idx`."""
    best_members = None
    best_pooled = np.inf
    for combo in ALL_SUBSETS:
        ens = np.mean([oof[m][idx] for m in combo], axis=0)
        p = rae(y[idx], ens[...])  # pooled rae on this index block
        if p < best_pooled:
            best_pooled = p
            best_members = combo
    return best_members, float(best_pooled)


def nested_selection_validated(
    oof: dict, y: np.ndarray, scaffs, kf_seed: int
):
    """One outer 5-fold scaffold CV pass.

    Within each outer fold: choose the pooled-optimal subset on outer-TRAIN, apply
    it (equal-weight mean) to outer-VAL. Stitch VAL blocks -> full-253 OOF; score
    ONE pooled rae() over the stitched vector (LB-faithful).

    Returns (stitched_pooled, stitched_oof_vector, per_fold_selected_members).
    """
    splits = scaffold_kfold_indices(scaffs, n_splits=N_FOLDS, shuffle=True, seed=kf_seed)
    n = len(y)
    oof_pred = np.full(n, np.nan, dtype=np.float64)
    covered = np.zeros(n, dtype=bool)
    fold_members = []
    for tr, va in splits:
        sel_members, _train_pooled = best_subset_on(tr, oof, y)
        val_ens = np.mean([oof[m][va] for m in sel_members], axis=0)
        oof_pred[va] = val_ens
        covered[va] = True
        fold_members.append(sel_members)
    if not covered.all():
        raise RuntimeError(f"kf_seed={kf_seed}: splits did not cover all {n} rows")
    stitched_pooled = float(rae(y, oof_pred))
    return stitched_pooled, oof_pred, fold_members


def fixed_subset_crossfit(
    oof: dict, y: np.ndarray, scaffs, kf_seed: int, members: tuple
) -> float:
    """Sidecar: apply a FIXED subset (no per-fold search) to outer-VAL, stitch,
    pooled. Isolates the selection effect: this should ~equal the deploy single-shot
    pooled of `members` (modulo partition; here it is just the full-253 pooled since
    equal-weight averaging is partition-invariant -- computed per-fold only to mirror
    the nested harness)."""
    splits = scaffold_kfold_indices(scaffs, n_splits=N_FOLDS, shuffle=True, seed=kf_seed)
    n = len(y)
    oof_pred = np.full(n, np.nan, dtype=np.float64)
    for _tr, va in splits:
        oof_pred[va] = np.mean([oof[m][va] for m in members], axis=0)
    return float(rae(y, oof_pred))


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- SELECTION-VALIDATED cross-fit verification of nb3420")
    print(f"          roster ({len(CANDIDATES)}): {CANDIDATES}")
    print(f"          {len(ALL_SUBSETS)} subsets searched WITHIN each outer-TRAIN fold")
    print(f"          {len(KF_SEEDS)} fresh kf_seeds {{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}")
    print(f"          nb3420 in-sample (global-min on 253) = {NB3420_INSAMPLE_POOLED}")
    print(f"          gate: VERIFIED<{GATE_VERIFIED}  MARGINAL<{GATE_MARGINAL}  "
          f"else SELECTION_BIAS")
    print("=" * 78)

    # -- Load truth, test SMILES/names, scaffolds ------------------------------
    te_df = load_test()
    n_test = len(te_df)
    te_smiles = (
        te_df["smiles"].astype(str).tolist()
        if "smiles" in te_df.columns
        else te_df["SMILES"].astype(str).tolist()
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
    print(f"[truth] full-253 L1 dispersion D(U) = {D_full_253:.2f}")

    # -- Load candidate OOF + TE; verify PRE-clean -----------------------------
    oof = {}
    te = {}
    singleton_pooled = {}
    leak_frac = {}
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
        oof[tag] = a
        te[tag] = b
        singleton_pooled[tag] = float(rae(y_unb, a))
        leak_frac[tag] = float(np.mean(np.isclose(a, y_unb, atol=1e-6)))
        print(f"   {tag}: pooled={singleton_pooled[tag]:.5f}  "
              f"te={'OK' if b is not None else 'MISSING'}  leak={leak_frac[tag]:.3f}")

    all_pre_clean = all(v == 0.0 for v in leak_frac.values())
    if not all_pre_clean:
        print("   WARN: at least one candidate has OOF==truth rows (leak suspect)")

    # -- Confirm the full-253 global-min reproduces nb3420's winner ------------
    full_members, full_pooled = best_subset_on(np.arange(n_unb), oof, y_unb)
    print("\n" + "=" * 78)
    print("REPRODUCE nb3420 IN-SAMPLE (global-min subset on full 253)")
    print("=" * 78)
    print(f"   global-min subset = {'+'.join(full_members)}  pooled={full_pooled:.5f}")
    print(f"   nb3420 reported    = {'+'.join(NB3420_WINNER_SUBSET)}  "
          f"pooled={NB3420_INSAMPLE_POOLED}")
    reproduces_nb3420 = (
        set(full_members) == set(NB3420_WINNER_SUBSET)
        and abs(full_pooled - NB3420_INSAMPLE_POOLED) < 1e-4
    )
    print(f"   reproduces nb3420 winner+value: {reproduces_nb3420}")

    # -- NESTED selection-validated cross-fit over 15 fresh seeds --------------
    print("\n" + "=" * 78)
    print("NESTED SELECTION-VALIDATED CROSS-FIT (search-on-train, apply-on-val)")
    print("=" * 78)
    sv_pooled = []          # selection-validated stitched pooled per seed
    fixed_pooled = []       # fixed-winner-subset stitched pooled per seed (sidecar)
    seed_oof_vectors = []   # stitched OOF per seed (to pick representative median)
    winner_selected_folds = 0
    total_folds = 0
    subset_select_counter: dict[tuple, int] = {}
    print(f"   {'seed':<7}{'sel_valid':<12}{'fixed_win':<12}"
          f"{'sel-fixed':<11}folds_picking_nb3420_winner")
    for s in KF_SEEDS:
        p_sv, oof_vec, fold_members = nested_selection_validated(
            oof, y_unb, unb_scaffolds, s
        )
        p_fx = fixed_subset_crossfit(
            oof, y_unb, unb_scaffolds, s, NB3420_WINNER_SUBSET
        )
        sv_pooled.append(p_sv)
        fixed_pooled.append(p_fx)
        seed_oof_vectors.append(oof_vec)
        nwin = 0
        for fm in fold_members:
            total_folds += 1
            subset_select_counter[fm] = subset_select_counter.get(fm, 0) + 1
            if set(fm) == set(NB3420_WINNER_SUBSET):
                winner_selected_folds += 1
                nwin += 1
        print(f"   {s:<7}{p_sv:<12.5f}{p_fx:<12.5f}{p_sv - p_fx:<+11.5f}{nwin}/5")

    sv_pooled = np.array(sv_pooled)
    fixed_pooled = np.array(fixed_pooled)
    sv_mean = float(sv_pooled.mean())
    sv_std = float(sv_pooled.std(ddof=1))
    sv_min = float(sv_pooled.min())
    sv_max = float(sv_pooled.max())
    fx_mean = float(fixed_pooled.mean())
    fx_std = float(fixed_pooled.std(ddof=1))
    winner_select_freq = winner_selected_folds / total_folds

    # Selection penalty = how much per-fold SEARCH costs vs the fixed deploy subset.
    selection_penalty = sv_mean - fx_mean

    # Representative OOF: the seed whose sv_pooled is closest to the deep-15 median.
    sv_median = float(np.median(sv_pooled))
    rep_i = int(np.argmin(np.abs(sv_pooled - sv_median)))
    rep_oof = seed_oof_vectors[rep_i]
    rep_seed = KF_SEEDS[rep_i]

    # Most-frequently-selected subset across all 75 folds (procedure consensus).
    top_subset = max(subset_select_counter, key=subset_select_counter.get)
    top_subset_count = subset_select_counter[top_subset]

    print("\n   " + "-" * 70)
    print(f"   selection-validated pooled deep-15 = {sv_mean:.5f} +/- {sv_std:.5f} "
          f"[min {sv_min:.5f}, max {sv_max:.5f}]")
    print(f"   fixed-winner-subset  pooled deep-15 = {fx_mean:.5f} +/- {fx_std:.5f}")
    print(f"   SELECTION PENALTY (sel - fixed)     = {selection_penalty:+.5f}")
    print(f"   nb3200 singleton reference          = {NB3200_SINGLETON_POOLED:.5f}")
    print(f"   nb3420 in-sample (global-min)       = {NB3420_INSAMPLE_POOLED:.5f}")
    print(f"   winner-subset selection frequency   = {winner_selected_folds}/"
          f"{total_folds} = {winner_select_freq:.3f}")
    print(f"   most-selected subset across folds   = {'+'.join(top_subset)} "
          f"({top_subset_count}/{total_folds})")

    # -- Gate decision (on selection-validated deep-15 MEAN) -------------------
    if sv_mean < GATE_VERIFIED:
        gate = "VERIFIED"
    elif sv_mean < GATE_MARGINAL:
        gate = "MARGINAL"
    else:
        gate = "SELECTION_BIAS"

    delta_vs_insample = round(sv_mean - NB3420_INSAMPLE_POOLED, 5)
    delta_vs_singleton = round(sv_mean - NB3200_SINGLETON_POOLED, 5)

    print("\n" + "=" * 78)
    print("GATE")
    print("=" * 78)
    print(f"   selection-validated deep-15 mean = {sv_mean:.5f}")
    print(f"   delta vs nb3420 in-sample {NB3420_INSAMPLE_POOLED} = {delta_vs_insample:+.5f}")
    print(f"   delta vs nb3200 singleton {NB3200_SINGLETON_POOLED} = {delta_vs_singleton:+.5f}")
    print(f"   thresholds: VERIFIED<{GATE_VERIFIED}  MARGINAL<{GATE_MARGINAL}")
    print(f"   GATE = {gate}")

    # -- Deploy TE: mean of the full-253-selected (= nb3420 winner) subset -----
    have_all_te = all(te[m] is not None for m in full_members)
    win_te = (
        np.mean([te[m] for m in full_members], axis=0) if have_all_te else None
    )
    te_unb_in_sample = (
        float(rae(y_unb, win_te[unb_idx])) if win_te is not None else None
    )

    # -- Save representative OOF + deploy TE -----------------------------------
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy", rep_oof.astype(np.float32))
    if win_te is not None:
        np.save(DATA_PROCESSED / f"te_{TAG}.npy", win_te.astype(np.float32))

    # -- Verdict ---------------------------------------------------------------
    if gate == "VERIFIED":
        verdict = (
            f"VERIFIED. The pooled-optimal-subset PROCEDURE survives nested "
            f"selection-validated cross-fit: deep-15 stitched pooled = {sv_mean:.5f} "
            f"+/- {sv_std:.5f} (< {GATE_VERIFIED}), within {delta_vs_insample:+.5f} of "
            f"nb3420's in-sample {NB3420_INSAMPLE_POOLED}. Per-fold train-selected "
            f"subsets transfer to held-out VAL (selection penalty {selection_penalty:+.5f} "
            f"vs the fixed winner subset; winner picked in {winner_select_freq:.0%} of "
            f"folds). nb3420's {('+'.join(NB3420_WINNER_SUBSET))} is NOT subset-selection "
            f"bias; the ~0.4413 pooled is honest. PRE-clean={all_pre_clean}."
        )
    elif gate == "MARGINAL":
        verdict = (
            f"MARGINAL. Nested selection-validated pooled = {sv_mean:.5f} +/- {sv_std:.5f} "
            f"sits in [{GATE_VERIFIED}, {GATE_MARGINAL}): the subset-search procedure "
            f"holds up to roughly the nb3200 singleton floor ({NB3200_SINGLETON_POOLED}, "
            f"delta {delta_vs_singleton:+.5f}) but the {delta_vs_insample:+.5f} gap to "
            f"nb3420's in-sample {NB3420_INSAMPLE_POOLED} is the selection optimism "
            f"(penalty {selection_penalty:+.5f} vs fixed winner subset). The 0.44130 "
            f"is mildly selection-flattered; the honest ensemble pooled is ~{sv_mean:.4f}, "
            f"a near-tie with the lone nb3200 clip. Treat nb3420 as ALTERNATE, not a "
            f"genuine sub-0.4414 improvement. PRE-clean={all_pre_clean}."
        )
    else:
        verdict = (
            f"SELECTION_BIAS. Nested selection-validated pooled = {sv_mean:.5f} +/- "
            f"{sv_std:.5f} >= {GATE_MARGINAL}, i.e. {delta_vs_insample:+.5f} above "
            f"nb3420's in-sample {NB3420_INSAMPLE_POOLED}. When the subset is chosen on "
            f"outer-TRAIN and scored on held-out VAL, the ~0.4413 advantage evaporates "
            f"(selection penalty {selection_penalty:+.5f} vs the fixed winner subset; "
            f"winner picked in only {winner_select_freq:.0%} of folds). The global-min "
            f"pick over 50 near-tied subsets rode the 253's noise realization. DO NOT "
            f"promote nb3420 below 0.4414; the honest ensemble pooled is ~{sv_mean:.4f}, "
            f"no better than the lone nb3200 clip ({NB3200_SINGLETON_POOLED})."
        )

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"   {verdict}")

    summary = {
        "tag": TAG,
        "method": "nested_selection_validated_pooled_subset_search",
        "purpose": "verify nb3420 pooled-optimal subset is not subset-selection bias",
        "lb_faithful_metric": "POOLED rae() on stitched 253 cross-fit OOF (nb3402)",
        "protocol": (
            "outer 5-fold scaffold CV; per fold search 50 subsets on outer-TRAIN "
            "pooled, apply train-selected subset to outer-VAL; stitch -> one pooled "
            "rae(); deep-15 over fresh kf_seeds {1431..1445}"
        ),
        "candidates": CANDIDATES,
        "subset_sizes": list(SUBSET_SIZES),
        "n_subsets_searched_per_fold": len(ALL_SUBSETS),
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": len(KF_SEEDS),
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "D_full_253": round(D_full_253, 4),
        "all_candidates_pre_clean": bool(all_pre_clean),
        "leak_eq_truth_frac": {k: round(v, 4) for k, v in leak_frac.items()},
        "singleton_pooled": {k: round(v, 5) for k, v in singleton_pooled.items()},
        "nb3200_singleton_pooled": NB3200_SINGLETON_POOLED,
        # nb3420 reproduction
        "nb3420_insample_pooled": NB3420_INSAMPLE_POOLED,
        "nb3420_winner_subset": list(NB3420_WINNER_SUBSET),
        "global_min_subset_this_run": list(full_members),
        "global_min_pooled_this_run": round(full_pooled, 5),
        "reproduces_nb3420_winner": bool(reproduces_nb3420),
        # selection-validated results (THE GATE NUMBER)
        "selection_validated_pooled_mean": round(sv_mean, 5),
        "selection_validated_pooled_std": round(sv_std, 5),
        "selection_validated_pooled_min": round(sv_min, 5),
        "selection_validated_pooled_max": round(sv_max, 5),
        "selection_validated_pooled_per_seed": [round(float(v), 5) for v in sv_pooled],
        # fixed-winner-subset sidecar + selection penalty
        "fixed_winner_subset_pooled_mean": round(fx_mean, 5),
        "fixed_winner_subset_pooled_std": round(fx_std, 5),
        "selection_penalty_sel_minus_fixed": round(selection_penalty, 5),
        # procedure stability
        "winner_subset_select_freq": round(winner_select_freq, 4),
        "winner_subset_selected_folds": int(winner_selected_folds),
        "total_outer_folds": int(total_folds),
        "most_selected_subset": list(top_subset),
        "most_selected_subset_count": int(top_subset_count),
        "n_distinct_subsets_selected": len(subset_select_counter),
        # deltas
        "delta_vs_nb3420_insample": delta_vs_insample,
        "delta_vs_nb3200_singleton": delta_vs_singleton,
        # gate
        "gate_verified_threshold": GATE_VERIFIED,
        "gate_marginal_threshold": GATE_MARGINAL,
        "gate": gate,
        # deploy carry-overs
        "representative_seed": int(rep_seed),
        "representative_seed_pooled": round(float(sv_pooled[rep_i]), 5),
        "anchor_pre_unblind": True,
        "te_unb_in_sample_rae": (
            round(te_unb_in_sample, 4) if te_unb_in_sample is not None else None
        ),
        "te_mean": (float(np.mean(win_te)) if win_te is not None else None),
        "te_std": (float(np.std(win_te)) if win_te is not None else None),
        "pred_oof_path": str(DATA_PROCESSED / f"{TAG}_pred_oof.npy"),
        "te_npy_path": (
            str(DATA_PROCESSED / f"te_{TAG}.npy") if win_te is not None else None
        ),
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n   [save] {out_path}")
    print(f"   [save] {DATA_PROCESSED / f'{TAG}_pred_oof.npy'} "
          f"(representative seed {rep_seed})")
    if win_te is not None:
        print(f"   [save] {DATA_PROCESSED / f'te_{TAG}.npy'}")
    print("=" * 78)
    print(f"=== {TAG} DONE  ({time.time()-t0:.1f}s)  GATE={gate}  "
          f"sel_valid={sv_mean:.5f} +/- {sv_std:.5f} ===")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "selection_validated_pooled_mean", "selection_validated_pooled_std",
        "fixed_winner_subset_pooled_mean", "selection_penalty_sel_minus_fixed",
        "winner_subset_select_freq", "delta_vs_nb3420_insample",
        "reproduces_nb3420_winner", "gate",
    ):
        print(f"  {k}: {res.get(k)}")
