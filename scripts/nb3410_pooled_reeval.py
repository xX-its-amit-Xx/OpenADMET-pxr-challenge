"""nb3410 -- POOLED-metric re-evaluation of the strongest activity candidates.

THE PREMISE (established by nb3402)
-----------------------------------
The public LB scores all 513 test rows with ONE RAE denominator:
    RAE(S) = sum_{i in S} |y_i - p_i|  /  sum_{i in S} |y_i - mean_S(y)|
This is a POOLED computation. The 253-unblind analog of the LB number is
therefore the POOLED RAE over all 253 cross-fit predictions -- a single rae()
call on the full vector -- NOT the mean of 5 per-fold ratios. nb3402 proved
PER-FOLD-MEAN overstates by a denominator-reweighting (Jensen) gap of ~+0.01
that is driven by fold truth-dispersion GEOMETRY, not by prediction quality.

CONSEQUENCE: several candidates I dismissed on per-fold-mean (e.g. nb3204 was
REJECTED by cycle-267 nb3210 on pf_mean 0.4563) may genuinely tie or beat nb3200
once they are scored under the LB-faithful pooled metric. This script re-ranks
the full strong-candidate roster under pooled and identifies the TRUE pooled-best.

A CRITICAL ESTIMAND NOTE (why pooled is seed-invariant here)
------------------------------------------------------------
Every candidate below has a STORED, DEPLOY-FROZEN `*_pred_oof.npy` (253,) vector.
For a FIXED vector, rae(y_unb, pred_oof) does not depend on the kf_seed at all:
the single denominator sum|y - mean(y)| over the full 253 is a property of the
TRUTH alone, and the numerator sum|y - p| is a property of the fixed (y, p) pair.
So the honest pooled point estimate of each candidate is ONE number, and its
across-seed std is exactly 0. This is NOT a defect -- it is the correct
estimation variance of a parameter-free deploy vector (nb3402 verdict).

We still sweep 15 fresh kf_seeds {1216..1230} to:
  (a) EMPIRICALLY CONFIRM pooled seed-invariance (assert std < 1e-9), and
  (b) report the PER-FOLD-MEAN sidecar + its across-seed dispersion as a
      generalization-stability diagnostic only (never as the gate).

Note on nb3200's two pooled numbers: nb3402 reports nb3200 single-call pooled =
0.4416, while nb3200's own deep-30 *re-fit* mean is 0.4424 (deep-30 re-derives
the per-fold clip per seed, producing a DIFFERENT OOF vector each seed). To keep
an apples-to-apples ranking, ALL candidates here are scored on their STORED
deploy-frozen pred_oof via a single rae() call (the deploy estimand). The 0.4424
deep-30 mean is carried as a reference only.

GATE (vs nb3200 pooled reference 0.4424)
----------------------------------------
  pooled-best < 0.4414  (nb3200 - 0.001, meaningful)  -> "POOLED_PROMOTE"
  pooled-best in [0.4414, 0.4424]                      -> "POOLED_MARGINAL"
  else                                                  -> "NB3200_BEST"

PRE-clean / deploy verification for the pooled-best
---------------------------------------------------
  - anchor_leak_eq_truth_frac == 0  (no row equals truth -> no anchor leak)
  - deploy CSV is 513x3 with columns {SMILES, Molecule Name, pEC50}
    (regenerated from te vector + load_test() if the FAIL-verdict candidate
     never wrote a CSV)

Inputs:
    data/processed/_audit_unblind_idx.npy   (253,)
    data/processed/_audit_unblind_y.npy     (253,)
    data/processed/{tag}_pred_oof.npy        (253,) for each candidate
    data/processed/te_{tag}.npy              (513,) for each candidate

Outputs:
    data/processed/nb3410_summary.json
    submissions/nb3410_pooled_best_{tag}.csv   (only if a CSV had to be regenerated)
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
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3410"

# ----------------------------------------------------------------------------
# Candidate roster: the strongest activity candidates to re-evaluate under
# pooled.  `desc` is carried for the report; `kind` records whether the upstream
# operator was parameter-free (ensemble/mix -> truly fixed on the 253) or a
# per-fold-fitted operator whose stored pred_oof is the deploy-frozen OOF.
# (For pooled scoring the distinction does not change the math -- a stored vector
#  is a stored vector -- but it is reported for honesty about estimand origin.)
# ----------------------------------------------------------------------------
CANDIDATES = [
    {"tag": "nb3200", "kind": "fitted", "desc": "deep-30 learned per-fold clip on nb3090 (incumbent ref)"},
    {"tag": "nb3173", "kind": "fitted", "desc": "learned per-fold clip on nb3080"},
    {"tag": "nb3204", "kind": "fixed",  "desc": "3-ensemble: equal-weight mean of 3 clip winners"},
    {"tag": "nb3211", "kind": "fixed",  "desc": "wRAE-ensemble: inverse-RAE-weighted mean of 3 clip winners"},
    {"tag": "nb3212", "kind": "fitted", "desc": "clip-on-ensemble: per-fold clip on nb3204 anchor"},
    {"tag": "nb3214", "kind": "fitted", "desc": "SLSQP: per-fold simplex on 3 clip winners"},
    {"tag": "nb3190", "kind": "fitted", "desc": "learned per-fold clip on nb3090 (15-seed)"},
    {"tag": "nb3180", "kind": "fitted", "desc": "wide-seed verify clip on nb3080"},
    {"tag": "nb3181", "kind": "fitted", "desc": "wide-seed verify nb3174 clip on nb3090"},
    {"tag": "nb3170", "kind": "fitted", "desc": "wide-seed verify y-range clip nb3080 q05/q95"},
    {"tag": "nb3231", "kind": "fitted", "desc": "2-mix: per-fold SLSQP simplex on 2 clip winners"},
    {"tag": "nb3343", "kind": "fitted", "desc": "2-clip: per-fold SLSQP on 2 differently-built clip winners"},
]

N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))  # 15 fresh seeds {1216..1230}

# nb3200 pooled reference for the gate (task-specified deep-30 mean).
NB3200_POOLED_REF = 0.4424
GATE_PROMOTE = round(NB3200_POOLED_REF - 0.001, 4)  # 0.4414  (meaningful win)
GATE_MARGINAL_HI = NB3200_POOLED_REF                # 0.4424  (tie band upper)

# Reference numbers carried from prior cycles (reconciliation only).
REF = {
    "nb3200_pooled_deep30": 0.4424,   # task-specified incumbent reference
    "nb3204_pf_deep30_reject": 0.4563,  # cycle-267 nb3210 per-fold-mean reject of nb3204
    "nb2171_primary1": 0.4682,        # prior post-hoc-blend ceiling PRIMARY-1
}

DEPLOY_COLS = ["SMILES", "Molecule Name", "pEC50"]


def _per_fold_sidecar(pred: np.ndarray, y: np.ndarray, scaffs, kf_seed: int) -> dict:
    """One split: pooled (must be seed-invariant) + per-fold-mean sidecar."""
    splits = scaffold_kfold_indices(scaffs, n_splits=N_FOLDS, shuffle=True, seed=kf_seed)
    n = len(y)
    covered = np.zeros(n, dtype=bool)
    fold_rae = []
    fold_den = []
    fold_sz = []
    for _tr, va in splits:
        covered[va] = True
        yv = y[va]
        pv = pred[va]
        ae = float(np.sum(np.abs(yv - pv)))
        den = float(np.sum(np.abs(yv - yv.mean())))
        fold_rae.append(ae / den if den > 0 else 0.0)
        fold_den.append(round(den, 2))
        fold_sz.append(int(len(va)))
    if not covered.all():
        raise RuntimeError(f"kf_seed={kf_seed}: splits did not cover all 253 rows")
    return {
        "kf_seed": int(kf_seed),
        "pooled": float(rae(y, pred)),       # single denominator -> seed-invariant
        "pf_mean": float(np.mean(fold_rae)),
        "fold_den": fold_den,
        "fold_sz": fold_sz,
    }


def _load_summary(tag: str) -> dict:
    f = DATA_PROCESSED / f"{tag}_summary.json"
    if not f.exists():
        return {}
    try:
        return json.load(open(f, encoding="utf-8"))
    except Exception:
        return {}


def _anchor_pre_unblind_flag(tag: str, summ: dict) -> bool | None:
    """Resolve PRE-clean anchor status. None -> unknown (not False)."""
    if "anchor_pre_unblind" in summ and summ["anchor_pre_unblind"] is not None:
        return bool(summ["anchor_pre_unblind"])
    return None


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- POOLED-metric re-evaluation of strongest activity candidates")
    print(f"          roster ({len(CANDIDATES)}): {[c['tag'] for c in CANDIDATES]}")
    print(f"          kf_seeds = {len(KF_SEEDS)} fresh {{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}")
    print(f"          LB-faithful metric = POOLED (single rae() call, nb3402)")
    print(f"          gate vs nb3200 pooled {NB3200_POOLED_REF}: "
          f"PROMOTE<{GATE_PROMOTE}, MARGINAL[{GATE_PROMOTE},{GATE_MARGINAL_HI}]")
    print("=" * 78)

    # -- Load truth, test SMILES/names, scaffolds ------------------------------
    te_df = load_test()
    n_test = len(te_df)
    te_smiles = te_df["smiles"].astype(str).tolist()
    te_names = te_df["name"].astype(str).tolist()
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

    # -- Per-candidate pooled evaluation ---------------------------------------
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

        summ = _load_summary(tag)

        # Single-call pooled (the deploy estimand, LB-faithful, seed-invariant).
        pooled_single = float(rae(y_unb, pred))
        # Anchor-leak fraction: rows whose OOF prediction == truth (-> leak).
        leak_eq = float(np.mean(np.isclose(pred, y_unb, atol=1e-6)))

        # 15-seed sweep -> confirm seed-invariance + per-fold-mean sidecar.
        pooled_arr, pf_arr, seed_rows = [], [], []
        for s in KF_SEEDS:
            r = _per_fold_sidecar(pred, y_unb, unb_scaffolds, s)
            pooled_arr.append(r["pooled"])
            pf_arr.append(r["pf_mean"])
            seed_rows.append({
                "kf_seed": r["kf_seed"],
                "pooled": round(r["pooled"], 5),
                "pf_mean": round(r["pf_mean"], 4),
                "gap_pf_minus_pooled": round(r["pf_mean"] - r["pooled"], 4),
            })
        pooled_arr = np.asarray(pooled_arr)
        pf_arr = np.asarray(pf_arr)
        pooled_std = float(pooled_arr.std(ddof=1))
        pf_mean = float(pf_arr.mean())
        pf_std = float(pf_arr.std(ddof=1))
        seed_invariant = bool(pooled_std < 1e-9)
        # Sanity: the swept pooled must equal the single-call pooled.
        recon_err = float(abs(pooled_single - float(pooled_arr.mean())))

        anchor_pre = _anchor_pre_unblind_flag(tag, summ)
        te_unb_in_sample = (
            float(rae(y_unb, te_arr[unb_idx])) if te_arr is not None else None
        )

        print("\n" + "-" * 78)
        print(f"CANDIDATE {tag}  [{cand['kind']}]  -- {cand['desc']}")
        print("-" * 78)
        print(f"   POOLED (single rae call) = {pooled_single:.5f}  "
              f"[seed-invariant? {'YES' if seed_invariant else 'NO'} "
              f"std={pooled_std:.2e}]")
        print(f"   PF_MEAN sidecar          = {pf_mean:.4f}  std={pf_std:.4f}  "
              f"(gap +{pf_mean - pooled_single:.4f}, NOT the gate)")
        print(f"   anchor_leak_eq_truth_frac= {leak_eq:.4f}  "
              f"anchor_pre_unblind={anchor_pre}")
        if te_unb_in_sample is not None:
            print(f"   te[unb] in-sample rae    = {te_unb_in_sample:.4f}  "
                  f"(deploy-refit optimism, NOT LB-faithful)")
        if recon_err > 1e-6:
            print(f"   WARN: single-call vs swept pooled mismatch {recon_err:.2e}")
        if leak_eq > 0.02:
            print(f"   WARN: {leak_eq:.1%} OOF rows == truth -- possible anchor leak")

        results[tag] = {
            "tag": tag,
            "kind": cand["kind"],
            "desc": cand["desc"],
            "pooled": round(pooled_single, 5),
            "pooled_std_across_seed": round(pooled_std, 10),
            "pooled_seed_invariant": seed_invariant,
            "pf_mean": round(pf_mean, 4),
            "pf_std": round(pf_std, 4),
            "gap_pf_minus_pooled": round(pf_mean - pooled_single, 4),
            "anchor_leak_eq_truth_frac": round(leak_eq, 4),
            "anchor_pre_unblind": anchor_pre,
            "te_unb_in_sample_rae": (
                round(te_unb_in_sample, 4) if te_unb_in_sample is not None else None
            ),
            "stored_mean_rae_ref": summ.get("mean_rae"),
            "upstream_verdict": summ.get("verdict"),
            "recon_err_single_vs_swept": recon_err,
            "seed_rows": seed_rows,
        }

    if not results:
        raise RuntimeError("no candidates evaluated")

    # -- Ranking under POOLED (the gate metric) --------------------------------
    rank = sorted(results.values(), key=lambda r: r["pooled"])
    print("\n" + "=" * 78)
    print("POOLED RANKING  (lower = better; this IS the LB-faithful gate metric)")
    print("=" * 78)
    print(f"   {'#':<4}{'tag':<10}{'POOLED':<10}{'pf_mean':<10}{'leak':<7}{'desc'}")
    for i, r in enumerate(rank, 1):
        print(f"   {i:<4}{r['tag']:<10}{r['pooled']:<10.5f}{r['pf_mean']:<10.4f}"
              f"{r['anchor_leak_eq_truth_frac']:<7.3f}{r['desc'][:44]}")

    pooled_best = rank[0]
    best_tag = pooled_best["tag"]
    best_pooled = pooled_best["pooled"]
    nb3200_pooled_single = results.get("nb3200", {}).get("pooled")

    print(f"\n   POOLED-BEST          = {best_tag}  ({best_pooled:.5f})")
    print(f"   nb3200 single-call   = {nb3200_pooled_single}")
    print(f"   nb3200 deep-30 ref   = {NB3200_POOLED_REF}  (gate reference)")

    # -- Gate decision (vs the task-specified nb3200 deep-30 reference 0.4424) --
    if best_pooled < GATE_PROMOTE:
        gate = "POOLED_PROMOTE"
    elif best_pooled <= GATE_MARGINAL_HI:
        gate = "POOLED_MARGINAL"
    else:
        gate = "NB3200_BEST"

    delta_vs_nb3200_ref = round(best_pooled - NB3200_POOLED_REF, 5)
    meaningful = bool(delta_vs_nb3200_ref < -0.001)

    print("\n" + "=" * 78)
    print("GATE")
    print("=" * 78)
    print(f"   pooled-best {best_tag} = {best_pooled:.5f}")
    print(f"   delta vs nb3200 ref 0.4424 = {delta_vs_nb3200_ref:+.5f}  "
          f"({'MEANINGFUL >0.001' if meaningful else 'NOT meaningful (<=0.001)'})")
    print(f"   GATE = {gate}")

    # -- PRE-clean + deploy verification for the pooled-best -------------------
    print("\n" + "-" * 78)
    print(f"PRE-CLEAN + DEPLOY VERIFICATION for pooled-best = {best_tag}")
    print("-" * 78)
    best_leak = pooled_best["anchor_leak_eq_truth_frac"]
    pre_clean = bool(best_leak == 0.0)
    print(f"   anchor_leak_eq_truth_frac = {best_leak}  -> "
          f"{'PRE-CLEAN (0 rows == truth)' if pre_clean else 'LEAK SUSPECT'}")
    print(f"   anchor_pre_unblind flag   = {pooled_best['anchor_pre_unblind']}")

    # Deploy CSV: prefer the upstream-written submission CSV; if the candidate
    # was a FAIL-verdict that never wrote one, regenerate from te + load_test().
    best_summ = _load_summary(best_tag)
    deploy_csv_path = best_summ.get("submission_csv")
    deploy_ok = False
    deploy_shape = None
    deploy_cols = None
    deploy_csv_used = None
    regenerated = False

    def _validate_csv(path: str) -> tuple[bool, tuple | None, list | None]:
        try:
            df = pd.read_csv(path)
        except Exception:
            return False, None, None
        ok = (df.shape == (n_test, 3)) and (list(df.columns) == DEPLOY_COLS)
        return ok, df.shape, list(df.columns)

    if isinstance(deploy_csv_path, str) and os.path.exists(deploy_csv_path):
        deploy_ok, deploy_shape, deploy_cols = _validate_csv(deploy_csv_path)
        deploy_csv_used = deploy_csv_path
        print(f"   upstream deploy CSV: {os.path.basename(deploy_csv_path)}  "
              f"shape={deploy_shape} cols={deploy_cols} -> "
              f"{'VALID 513x3' if deploy_ok else 'INVALID'}")
    else:
        print(f"   upstream deploy CSV: {deploy_csv_path} (missing/None) "
              f"-> regenerating from te_{best_tag}.npy + load_test()")

    if not deploy_ok:
        te_best_path = DATA_PROCESSED / f"te_{best_tag}.npy"
        if te_best_path.exists():
            te_best = np.load(te_best_path).astype(np.float64)
            if te_best.shape == (n_test,):
                out_df = pd.DataFrame({
                    "SMILES": te_smiles,
                    "Molecule Name": te_names,
                    "pEC50": te_best,
                })
                regen_path = SUBMISSIONS / f"{TAG}_pooled_best_{best_tag}.csv"
                out_df.to_csv(regen_path, index=False)
                regenerated = True
                deploy_ok, deploy_shape, deploy_cols = _validate_csv(str(regen_path))
                deploy_csv_used = str(regen_path)
                print(f"   regenerated deploy CSV: {regen_path.name}  "
                      f"shape={deploy_shape} cols={deploy_cols} -> "
                      f"{'VALID 513x3' if deploy_ok else 'INVALID'}")
            else:
                print(f"   ERROR: te_{best_tag} shape {te_best.shape} != ({n_test},)")
        else:
            print(f"   ERROR: te_{best_tag}.npy missing -- cannot build deploy CSV")

    deploy_verified = bool(deploy_ok and deploy_shape == (n_test, 3)
                           and deploy_cols == DEPLOY_COLS)

    # -- Build pooled-best pred_oof + te for the summary -----------------------
    best_pred_oof = np.load(DATA_PROCESSED / f"{best_tag}_pred_oof.npy").astype(np.float64)
    best_te_path = DATA_PROCESSED / f"te_{best_tag}.npy"
    best_te = (
        np.load(best_te_path).astype(np.float64).tolist()
        if best_te_path.exists() else None
    )

    # -- Verdict / ladder text -------------------------------------------------
    if gate == "POOLED_PROMOTE":
        verdict = (
            f"POOLED_PROMOTE. Under the LB-faithful pooled metric (single rae() "
            f"call, nb3402), {best_tag} = {best_pooled:.5f} beats the nb3200 deep-30 "
            f"reference 0.4424 by {delta_vs_nb3200_ref:+.5f} (> 0.001, MEANINGFUL). "
            f"PRE-clean={pre_clean} (leak_frac={best_leak}); deploy CSV verified "
            f"513x3={deploy_verified}. Promote {best_tag} above nb3200 on the pooled "
            f"ladder."
        )
    elif gate == "POOLED_MARGINAL":
        verdict = (
            f"POOLED_MARGINAL. {best_tag} = {best_pooled:.5f} sits in the tie band "
            f"[{GATE_PROMOTE}, {GATE_MARGINAL_HI}] with the nb3200 deep-30 reference "
            f"0.4424 (delta {delta_vs_nb3200_ref:+.5f}, within the 0.001 noise floor). "
            f"The pooled re-evaluation confirms {best_tag} is NOT worse than nb3200 "
            f"(it was likely a per-fold-mean rejection artifact), but the gain is "
            f"inside noise -- KEEP nb3200 as the verification-rigor incumbent and "
            f"restore {best_tag} to ALTERNATE/MARGINAL. PRE-clean={pre_clean}; deploy "
            f"513x3={deploy_verified}."
        )
    else:
        verdict = (
            f"NB3200_BEST. The pooled-best {best_tag} = {best_pooled:.5f} does not "
            f"reach the tie band; nb3200 (deep-30 0.4424) remains the pooled ladder "
            f"leader. No reorder."
        )

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"   {verdict}")

    # Honest cross-candidate context: how many beat / tie nb3200 single-call?
    nb3200_single = nb3200_pooled_single if nb3200_pooled_single is not None else NB3200_POOLED_REF
    n_beat_single = sum(1 for r in results.values()
                        if r["pooled"] < nb3200_single - 1e-9)
    n_tie_single = sum(1 for r in results.values()
                       if abs(r["pooled"] - nb3200_single) <= 0.001)
    print(f"\n   vs nb3200 single-call pooled {nb3200_single}: "
          f"{n_beat_single} candidates strictly beat, {n_tie_single} within 0.001 tie band")

    summary = {
        "tag": TAG,
        "method": "pooled_metric_reeval_of_strongest_activity_candidates",
        "lb_faithful_metric": "POOLED (single rae() call over all 253; nb3402)",
        "writes_submission": bool(regenerated),
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": len(KF_SEEDS),
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "D_full_253": round(D_full_253, 4),
        "nb3200_pooled_deep30_ref": NB3200_POOLED_REF,
        "nb3200_pooled_single_call": nb3200_pooled_single,
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_hi": GATE_MARGINAL_HI,
        "candidates": results,
        "ranking_pooled": [r["tag"] for r in rank],
        "pooled_best_tag": best_tag,
        "pooled_best_rae": best_pooled,
        "pooled_best_kind": pooled_best["kind"],
        "pooled_best_desc": pooled_best["desc"],
        "delta_vs_nb3200_ref": delta_vs_nb3200_ref,
        "delta_meaningful_gt_0p001": meaningful,
        "pooled_best_pre_clean": pre_clean,
        "pooled_best_anchor_leak_eq_truth_frac": best_leak,
        "pooled_best_anchor_pre_unblind": pooled_best["anchor_pre_unblind"],
        "pooled_best_deploy_csv": deploy_csv_used,
        "pooled_best_deploy_shape": list(deploy_shape) if deploy_shape else None,
        "pooled_best_deploy_cols": deploy_cols,
        "pooled_best_deploy_verified_513x3": deploy_verified,
        "pooled_best_deploy_csv_regenerated": regenerated,
        "pooled_best_pred_oof": best_pred_oof.tolist(),
        "pooled_best_te": best_te,
        "n_candidates_beat_nb3200_single": int(n_beat_single),
        "n_candidates_tie_nb3200_single": int(n_tie_single),
        "gate": gate,
        "verdict": verdict,
        "ref": REF,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n   [save] {out_path}")
    print("=" * 78)
    print(f"=== {TAG} DONE  ({time.time()-t0:.1f}s)  GATE={gate}  "
          f"best={best_tag} {best_pooled:.5f} ===")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "lb_faithful_metric", "ranking_pooled", "pooled_best_tag",
        "pooled_best_rae", "delta_vs_nb3200_ref", "delta_meaningful_gt_0p001",
        "pooled_best_pre_clean", "pooled_best_deploy_verified_513x3", "gate",
    ):
        print(f"  {k}: {res.get(k)}")
