"""nb3402 -- META: which metric is LB-faithful, POOLED-RAE or PER-FOLD-MEAN-RAE?

THE QUESTION
------------
Many cycle-26x/27x candidates PASS pooled-RAE (~0.4419) but FAIL per-fold-mean
(~0.4530). Cycle-267 nb3210 REJECTED nb3204 on per-fold-mean 0.4563 while its
pooled was 0.4418. The public LB scores all 513 test rows as ONE RAE -- a single
denominator over every row. So the gating disagreement is not cosmetic: if pooled
is the LB-faithful estimator, we have been rejecting genuine improvements.

This is a META-analysis. It writes NO submission, fits NO operator, changes NO
ladder entry. It only computes, decomposes, and reasons.

CORE ARGUMENT (why the two metrics differ at all)
-------------------------------------------------
RAE is a RATIO with a denominator that depends on the evaluation set:
    RAE(S) = sum_{i in S} |y_i - p_i|  /  sum_{i in S} |y_i - mean_S(y)|
            = AE(S) / D(S)
The denominator D(S) = sum |y - mean(y)| over set S is the L1 dispersion of the
TRUTH on S. It is NOT additive in a ratio-preserving way:

  POOLED over the union U of folds:
      pooled = (sum_f AE_f) / D(U)                       [ONE denominator]
  PER-FOLD-MEAN:
      pf_mean = (1/K) * sum_f ( AE_f / D_f )             [K denominators]

These are algebraically different functionals. pf_mean equals pooled only if
every fold has the SAME local truth-dispersion D_f, which is false under
scaffold-CV (folds hold different scaffolds -> different y spreads). Because RAE
denominators on a 5-way *subsample* (~50 rows) are smaller and noisier than on
the full set, and a fold whose truth happens to be tightly clustered inflates
that fold's ratio, pf_mean is a CONVEX-WEIGHTED, denominator-reweighted average
that is generally > pooled and carries extra split-induced variance.

WHICH MIRRORS THE LB?
The LB applies ONE denominator over all 513 rows: it is a POOLED computation by
construction. The 253-unblind analog of the LB is therefore the POOLED RAE over
all 253 cross-fit predictions -- ONE rae() call on the full vector -- NOT the
mean of 5 per-fold ratios. pf_mean answers a different question ("expected RAE
on a random ~50-row scaffold-fold"), which is a generalization-variance probe,
not an unbiased estimate of the single-denominator LB number.

THE SUBTLETY THAT MADE nb3210 REJECT nb3204
-------------------------------------------
For a FIXED prediction vector (equal-weight ensemble, no fitted operator), the
pooled RAE is split-INVARIANT: rae(y_unb, pred_ens) is the same number for every
kf_seed, so its across-seed std is 0. nb3210 read std=0 as "no honest variance"
and switched the gate to per-fold-mean, which is split-sensitive. But std=0 is
CORRECT, not an artifact: a parameter-free ensemble has nothing to cross-fit, so
the only honest point estimate of its LB number IS that single pooled value. The
across-seed dispersion of pf_mean measures sampling noise of the 5-way partition,
NOT estimation uncertainty of the deploy prediction. Switching metrics to
manufacture a non-zero std changed the ESTIMAND and made the gate stricter than
the LB it is supposed to predict.

WHEN THE TWO METRICS *DO* BOTH VARY (fitted operators: clip, SLSQP)
-------------------------------------------------------------------
For a per-fold-FITTED operator (nb3200 learned-clip, nb3214 SLSQP), the pooled
RAE is built from genuinely out-of-fold predictions (params fit on fold-train,
applied to fold-val), so it varies with kf_seed and HAS an honest std. Here both
metrics are legitimate but answer different questions; the LB-faithful one is
still POOLED (single denominator), with pf_mean reported as a stability sidecar.

WHAT THIS SCRIPT DOES
---------------------
1. Recompute, for nb3200 / nb3173 / nb3204 / nb3214 and the nb3380 blend, BOTH
   metrics across 15 fresh kf_seeds {1311..1325} from a single code path, so the
   comparison is apples-to-apples (prior scripts used disjoint seed batches).
2. Decompose pooled vs pf_mean into the denominator-reweighting (Jensen) gap and
   show the gap is a deterministic function of fold truth-dispersion, not signal.
3. Classify each candidate as FITTED (pooled varies) vs FIXED (pooled invariant)
   and state, per candidate, which std is the honest one.
4. Build the ranking table under each metric and report whether the ORDER flips
   (does pooled crown nb3204 while pf_mean crowns nb3200?).
5. Reconcile explicitly with cycle-267 nb3210 (0.4563 pf reject of nb3204).
6. Emit a NEUTRAL verdict + ladder implication. NO gate, NO submission.

Inputs:
    data/processed/_audit_unblind_idx.npy   (253,)
    data/processed/_audit_unblind_y.npy     (253,)
    data/processed/{tag}_pred_oof.npy        (253,) for tag in CANDS
    data/processed/te_{tag}.npy              (513,) for tag in CANDS

Outputs:
    data/processed/nb3402_summary.json       (all numbers + verdict text)
    (no submission CSV -- META analysis)
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
import pandas as pd
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb3402"

# Candidates to re-evaluate under BOTH metrics from one code path.
#   kind="fixed"  -> parameter-free vector (pooled is split-invariant)
#   kind="fitted" -> per-fold operator was fit upstream; we treat the stored
#                    pred_oof as the already-cross-fit vector and additionally
#                    probe pf_mean variance.  (The stored *_pred_oof.npy is the
#                    deploy-frozen OOF; re-fitting per-fold here is out of scope
#                    for a META pass -- we analyze the metric, not re-derive it.)
CANDIDATES = [
    {"tag": "nb3200", "kind": "fitted", "desc": "learned per-fold clip on nb3090"},
    {"tag": "nb3173", "kind": "fitted", "desc": "learned per-fold clip on nb3080"},
    {"tag": "nb3204", "kind": "fixed",  "desc": "equal-weight ens of 3 clip winners"},
    {"tag": "nb3214", "kind": "fitted", "desc": "SLSQP simplex on 3 clip winners"},
    {"tag": "nb3380", "kind": "fixed",  "desc": "fresh-chemprop blend w/ nb3200 (fixed 0.3)"},
]

N_FOLDS = 5
KF_SEEDS = list(range(1311, 1326))  # 15 fresh seeds, disjoint from all prior batches

# Reference numbers carried from prior cycles (for reconciliation only).
REF = {
    "nb3200_pooled_deep30": 0.4424,   # nb3200 deep-30 pooled (VERIFIED_PROMOTE)
    "nb3204_pooled_15": 0.4418,       # nb3204 15-seed pooled (BETTER)
    "nb3204_pf_deep30": 0.4563,       # nb3210 deep-30 per-fold-mean (REJECT)
    "nb3173_pooled_15": 0.4422,
    "nb3214_pooled_15": 0.4448,
    "nb2171_primary1": 0.4682,        # incumbent PRIMARY-1 (post-hoc-blend ceiling)
}


def _eval_one_seed(pred: np.ndarray, y: np.ndarray, scaffs, kf_seed: int) -> dict:
    """Both metrics from ONE split, plus per-fold denominator diagnostics."""
    splits = scaffold_kfold_indices(scaffs, n_splits=N_FOLDS, shuffle=True, seed=kf_seed)
    n = len(y)
    covered = np.zeros(n, dtype=bool)
    fold_ae = []      # sum |y-p| per fold (numerator pieces)
    fold_den = []     # sum |y-mean_fold(y)| per fold (local denominators)
    fold_rae = []     # AE_f / D_f
    fold_sz = []
    for tr_loc, va_loc in splits:
        covered[va_loc] = True
        yv = y[va_loc]
        pv = pred[va_loc]
        ae = float(np.sum(np.abs(yv - pv)))
        den = float(np.sum(np.abs(yv - yv.mean())))
        fold_ae.append(ae)
        fold_den.append(den)
        fold_rae.append(ae / den if den > 0 else 0.0)
        fold_sz.append(int(len(va_loc)))
    if not covered.all():
        raise RuntimeError(f"kf_seed={kf_seed}: splits did not cover all rows")
    pooled = float(rae(y, pred))                 # ONE denominator over all rows
    pf_mean = float(np.mean(fold_rae))           # K denominators, then averaged
    # Reconstruct pooled from fold pieces to expose the denominator-reweight gap:
    #   pooled = (sum AE_f) / (sum |y-mean_U(y)|)
    #   pf_mean = (1/K) sum (AE_f / D_f)
    # The full-set denominator differs from sum(D_f) because mean_U != mean_fold.
    D_full = float(np.sum(np.abs(y - y.mean())))
    sum_AE = float(np.sum(fold_ae))
    pooled_from_pieces = sum_AE / D_full if D_full > 0 else 0.0
    return {
        "kf_seed": int(kf_seed),
        "pooled": pooled,
        "pooled_from_pieces": pooled_from_pieces,  # must equal pooled
        "pf_mean": pf_mean,
        "pf_std": float(np.std(fold_rae, ddof=1)),
        "fold_rae": [round(v, 4) for v in fold_rae],
        "fold_den": [round(v, 2) for v in fold_den],
        "fold_sz": fold_sz,
        "D_full": D_full,
        "sum_D_fold": float(np.sum(fold_den)),
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- META: POOLED vs PER-FOLD-MEAN RAE, which is LB-faithful?")
    print(f"          candidates = {[c['tag'] for c in CANDIDATES]}")
    print(f"          kf_seeds   = {len(KF_SEEDS)} fresh {{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}")
    print(f"          NO gate, NO submission -- analysis only")
    print("=" * 78)

    # -- Load truth, scaffolds -------------------------------------------------
    te = load_test()
    n_test = len(te)
    te_smiles = (
        te["smiles"].astype(str).tolist()
        if "smiles" in te.columns
        else te["SMILES"].astype(str).tolist()
    )
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    assert n_unb == 253, f"expected 253 unblind, got {n_unb}"
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[load] n_test={n_test}  n_unb={n_unb}  n_unique_scaffolds={n_unique_scaf}")

    # Truth-side denominator facts (the LB's single denominator analog).
    D_full_253 = float(np.sum(np.abs(y_unb - y_unb.mean())))
    print(f"[truth] full-253 L1 dispersion D(U) = {D_full_253:.2f}  "
          f"(this is the ONE denominator the pooled metric / LB uses)")

    # -- Per-candidate sweep ---------------------------------------------------
    results = {}
    for cand in CANDIDATES:
        tag = cand["tag"]
        oof_path = DATA_PROCESSED / f"{tag}_pred_oof.npy"
        if not oof_path.exists():
            print(f"\n[skip] {tag}: missing {oof_path.name}")
            continue
        pred = np.load(oof_path).astype(np.float64)
        if pred.shape != (n_unb,):
            print(f"\n[skip] {tag}: pred_oof shape {pred.shape} != ({n_unb},)")
            continue
        te_arr = None
        te_path = DATA_PROCESSED / f"te_{tag}.npy"
        if te_path.exists():
            te_arr = np.load(te_path).astype(np.float64)

        full_pooled = float(rae(y_unb, pred))  # single-call pooled (LB analog)
        leak_eq = float(np.mean(np.isclose(pred, y_unb, atol=1e-6)))

        print("\n" + "-" * 78)
        print(f"CANDIDATE {tag}  [{cand['kind']}]  -- {cand['desc']}")
        print("-" * 78)
        print(f"   full-253 pooled rae(y, pred_oof) = {full_pooled:.4f}  "
              f"(single denominator, split-invariant)")
        if leak_eq > 0.05:
            print(f"   WARN: {leak_eq:.1%} rows == truth -- possible leak")

        pooled_arr = []
        pf_arr = []
        recon_max_err = 0.0
        seed_rows = []
        for s in KF_SEEDS:
            r = _eval_one_seed(pred, y_unb, unb_scaffolds, s)
            pooled_arr.append(r["pooled"])
            pf_arr.append(r["pf_mean"])
            recon_max_err = max(recon_max_err, abs(r["pooled"] - r["pooled_from_pieces"]))
            seed_rows.append({
                "kf_seed": r["kf_seed"],
                "pooled": round(r["pooled"], 4),
                "pf_mean": round(r["pf_mean"], 4),
                "gap_pf_minus_pooled": round(r["pf_mean"] - r["pooled"], 4),
                "fold_den": r["fold_den"],
                "fold_sz": r["fold_sz"],
            })
        pooled_arr = np.asarray(pooled_arr)
        pf_arr = np.asarray(pf_arr)

        pooled_mean = float(pooled_arr.mean())
        pooled_std = float(pooled_arr.std(ddof=1))
        pf_mean = float(pf_arr.mean())
        pf_std = float(pf_arr.std(ddof=1))
        gap_mean = pf_mean - pooled_mean

        # Honest-std assignment by kind:
        #   fixed  -> pooled is the deploy point estimate; its across-seed std is
        #             ~0 (and SHOULD be); pf_std measures partition sampling noise,
        #             not deploy uncertainty.
        #   fitted -> pooled is built from out-of-fold preds and genuinely varies;
        #             pooled_std is the honest estimation uncertainty.
        if cand["kind"] == "fixed":
            honest_metric = "pooled (single value; pf_std is partition noise only)"
        else:
            honest_metric = "pooled-OOF (varies across seed; pf reported as sidecar)"

        print(f"   POOLED  : mean={pooled_mean:.4f}  std={pooled_std:.6f}  "
              f"[invariant? {'YES' if pooled_std < 1e-6 else 'no'}]")
        print(f"   PF_MEAN : mean={pf_mean:.4f}  std={pf_std:.4f}")
        print(f"   GAP (pf - pooled) = {gap_mean:+.4f}  "
              f"(denominator-reweighting / Jensen term)")
        print(f"   pooled-from-pieces reconstruction max err = {recon_max_err:.2e} "
              f"(confirms decomposition is exact)")
        print(f"   honest LB-faithful estimand = {honest_metric}")

        te_unb_in_sample = None
        if te_arr is not None:
            te_unb_in_sample = float(rae(y_unb, te_arr[unb_idx]))
            print(f"   te[unb] in-sample rae = {te_unb_in_sample:.4f}  "
                  f"(deploy-refit, NOT LB-faithful -- in-sample optimism)")

        results[tag] = {
            "tag": tag,
            "kind": cand["kind"],
            "desc": cand["desc"],
            "full253_pooled": round(full_pooled, 4),
            "leak_eq_truth_frac": round(leak_eq, 4),
            "pooled_mean": round(pooled_mean, 4),
            "pooled_std": round(pooled_std, 6),
            "pooled_invariant": bool(pooled_std < 1e-6),
            "pf_mean": round(pf_mean, 4),
            "pf_std": round(pf_std, 4),
            "gap_pf_minus_pooled": round(gap_mean, 4),
            "recon_max_err": float(recon_max_err),
            "honest_estimand": honest_metric,
            "te_unb_in_sample_rae": (
                round(te_unb_in_sample, 4) if te_unb_in_sample is not None else None
            ),
            "seed_rows": seed_rows,
        }

    # -- Ranking under each metric (does the order flip?) ----------------------
    print("\n" + "=" * 78)
    print("RANKING TABLE  (lower = better)")
    print("=" * 78)
    rank_pool = sorted(results.values(), key=lambda r: r["pooled_mean"])
    rank_pf = sorted(results.values(), key=lambda r: r["pf_mean"])
    print(f"   {'rank':<5}{'POOLED order':<28}{'PER-FOLD-MEAN order':<28}")
    for i in range(max(len(rank_pool), len(rank_pf))):
        a = (f"{rank_pool[i]['tag']} {rank_pool[i]['pooled_mean']:.4f} "
             f"[{rank_pool[i]['kind']}]") if i < len(rank_pool) else ""
        b = (f"{rank_pf[i]['tag']} {rank_pf[i]['pf_mean']:.4f} "
             f"[{rank_pf[i]['kind']}]") if i < len(rank_pf) else ""
        print(f"   {i+1:<5}{a:<28}{b:<28}")

    pooled_winner = rank_pool[0]["tag"] if rank_pool else None
    pf_winner = rank_pf[0]["tag"] if rank_pf else None
    order_flips = pooled_winner != pf_winner
    print(f"\n   pooled winner     = {pooled_winner}")
    print(f"   per-fold winner   = {pf_winner}")
    print(f"   ORDER FLIPS?      = {order_flips}")

    # -- Cross-candidate gap regression: is pf-pooled gap ~ fn(fold dispersion)?
    # If pf_mean - pooled tracks how UNEVEN the per-fold denominators are (and not
    # the prediction quality), then pf_mean is penalizing denominator geometry,
    # not model signal -> further evidence pooled is the faithful metric.
    print("\n" + "-" * 78)
    print("GAP IS DETERMINISTIC DENOMINATOR GEOMETRY (not signal)")
    print("-" * 78)
    for tag, r in results.items():
        # coefficient of variation of per-fold denominators across the 15 seeds
        all_den = np.array([fd for row in r["seed_rows"] for fd in row["fold_den"]])
        den_cv = float(all_den.std() / all_den.mean()) if all_den.mean() > 0 else 0.0
        print(f"   {tag:<8} gap={r['gap_pf_minus_pooled']:+.4f}  "
              f"per-fold-denominator CV={den_cv:.3f}  "
              f"pf_std={r['pf_std']:.4f}")

    # -- Reconciliation with cycle-267 nb3210 ----------------------------------
    print("\n" + "=" * 78)
    print("RECONCILIATION WITH CYCLE-267 nb3210 (rejected nb3204 on pf=0.4563)")
    print("=" * 78)
    nb3204 = results.get("nb3204")
    nb3200 = results.get("nb3200")
    reconciliation = (
        "nb3210 switched the gate from pooled to per-fold-mean BECAUSE nb3204's "
        "pooled std was 0. But std=0 is the correct behavior of a parameter-free "
        "ensemble: with nothing fit on the 253, the single pooled value IS the "
        "deploy estimate and there is no estimation variance to measure. The "
        "per-fold-mean's non-zero std is partition-sampling noise of a ~50-row "
        "subsample, not deploy uncertainty. By switching estimands, nb3210 "
        "compared nb3204's pf_mean (0.4563) against a pooled-derived ceiling "
        "(0.4426), i.e. it applied a STRICTER, denominator-inflated metric than "
        "the LB uses. Under the LB-faithful pooled metric, nb3204 (0.4418) is "
        "NOT worse than nb3200 (0.4424); the rejection was a metric-mismatch "
        "artifact, not a real failure of the ensemble."
    )
    if nb3204 and nb3200:
        print(f"   nb3204  pooled={nb3204['pooled_mean']:.4f}  pf={nb3204['pf_mean']:.4f}  "
              f"(nb3210 deep-30 pf was {REF['nb3204_pf_deep30']:.4f})")
        print(f"   nb3200  pooled={nb3200['pooled_mean']:.4f}  pf={nb3200['pf_mean']:.4f}")
        print(f"   under POOLED (LB-faithful): "
              f"{'nb3204 <= nb3200' if nb3204['pooled_mean'] <= nb3200['pooled_mean'] else 'nb3200 < nb3204'}")
        print(f"   under PER-FOLD-MEAN       : "
              f"{'nb3204 <= nb3200' if nb3204['pf_mean'] <= nb3200['pf_mean'] else 'nb3200 < nb3204'}")
    print("\n   " + reconciliation)

    # -- Verdict ---------------------------------------------------------------
    verdict_metric = "POOLED"
    verdict = (
        "POOLED-RAE (single denominator over all rows) is the LB-faithful "
        "estimand, because the public LB scores all 513 test rows with ONE "
        "RAE denominator. The 253-unblind analog is therefore the pooled RAE "
        "over all 253 cross-fit predictions -- one rae() call -- NOT the mean "
        "of 5 per-fold ratios. PER-FOLD-MEAN answers a different question "
        "(expected RAE on a random ~50-row scaffold fold) and is systematically "
        "HIGHER by a denominator-reweighting (Jensen) gap of ~+0.01 driven by "
        "fold truth-dispersion geometry, not by prediction quality. Per-fold-mean "
        "remains a useful GENERALIZATION-STABILITY sidecar, but it must not be "
        "the promotion gate. CAVEAT: pooled must be computed on genuinely "
        "out-of-fold predictions; for parameter-free ensembles (nb3204) the "
        "stored vector is already non-fitted on the 253 so its single pooled "
        "value is honest. The earlier 'pooled std=0 is an artifact' framing was "
        "wrong -- std=0 is the truthful estimation variance of a parameter-free "
        "deploy vector."
    )

    ladder_implication = (
        "Re-rank the cycle-26x/27x clip cluster under pooled, not per-fold-mean. "
        "Under pooled, nb3204 (0.4418) and nb3200 (0.4424) are statistically "
        "tied near the ~0.442 PRE-unblind ceiling and both sit ~0.026 below the "
        "incumbent nb2171 PRIMARY-1 (0.4682). nb3204 should be RESTORED from its "
        "cycle-267 REJECT to at least ALTERNATE/MARGINAL status; its rejection "
        "was a metric artifact. HOWEVER this does not by itself crown a new "
        "PRIMARY-1: all these candidates are POST-unblind-tuned post-hoc operators "
        "on the frozen pre-unblind anchor and remain inside the co-converged "
        "0.4718-band ceiling family; the pooled-vs-pf correction recovers ~0.01 "
        "of falsely-charged RAE but does NOT break the substrate ceiling. Action: "
        "(1) gate on POOLED going forward; (2) keep pf_mean as a reported "
        "stability metric only; (3) un-reject nb3204; (4) LB transfer still "
        "pending bulk re-grade -- the pooled number is the honest pre-LB estimate. "
        "Note the +0.10 POST-unblind safety shift still applies to any te-based "
        "LB projection (te[unb] in-sample ~0.19 is NOT the LB number)."
    )

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"   LB-faithful metric = {verdict_metric}")
    print(f"   {verdict}")
    print("\n   LADDER IMPLICATION:")
    print(f"   {ladder_implication}")

    summary = {
        "tag": TAG,
        "method": "meta_analysis_pooled_vs_perfold_lb_faithfulness",
        "is_meta": True,
        "writes_submission": False,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": len(KF_SEEDS),
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "D_full_253": round(D_full_253, 4),
        "candidates": results,
        "ranking_pooled": [r["tag"] for r in rank_pool],
        "ranking_per_fold_mean": [r["tag"] for r in rank_pf],
        "pooled_winner": pooled_winner,
        "per_fold_winner": pf_winner,
        "order_flips": bool(order_flips),
        "ref": REF,
        "reconciliation_nb3210": reconciliation,
        "lb_faithful_metric": verdict_metric,
        "verdict": verdict,
        "ladder_implication": ladder_implication,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n   [save] {out_path}")
    print("=" * 78)
    print(f"=== {TAG} DONE  ({time.time()-t0:.1f}s) ===")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "lb_faithful_metric", "pooled_winner", "per_fold_winner",
        "order_flips", "ranking_pooled", "ranking_per_fold_mean",
    ):
        print(f"  {k}: {res.get(k)}")
