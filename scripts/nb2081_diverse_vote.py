"""nb2081 -- Diverse tri-vote: {nb1158, nb503, nb562}.

CONTEXT (per nb2076):
    nb2076 tri-vote on {nb1191, nb2060, nb1158} echoed nb1191 because both
    nb1191 and nb2060 are built from large overlapping anchor sets (chemprop_aux
    /nb1150/nb1158_K32/nb2112_K28). Spearman(vote, nb1191) was effectively 1.0
    -- the "vote" was just nb1191 in disguise.

HYPOTHESIS:
    Replace the two echo anchors with two genuinely independent OOFs:
        nb503 -- post-hoc rank-stretch survivor (honest cross-fit 0.5116)
        nb562 -- 1-param scalar rank-stretch baseline (honest cross-fit 0.5065)
    Keep nb1158 (honest cross-fit 0.4902) as the strong anchor and CDF source.
    Pairwise Spearman is expected to be < 0.95 because nb503 and nb562 are
    rank-stretches over different base predictors (nb503 is mid-stack, nb562
    is a scalar-stretched corpus-mean LGBM). If the vote mechanism is real
    (rather than an echo of nb1158), we expect the tri-vote OOF RAE to land
    distinctly different from 0.4902 -- ideally < 0.4902.

MECHANISM (identical to nb2076):
    1. Per-row rank for each of the 3 vectors using rankdata(method="average").
    2. Element-wise MEDIAN of the 3 rank vectors -> rank_vote in [1..N].
    3. Map rank_vote back to pec50 via the empirical CDF of nb1158
       (anchor CDF). nb1158 is the strongest individual; using its CDF
       produces a rank-stable, strong-anchor-calibrated pec50.

GATES:
    - Sanity: max pairwise Spearman among the 3 candidates must be < 0.95
      (else they are not "diverse").
    - Distinctness: rho(vote, anchor=nb1158) must be <= 0.98 (else the vote
      is just nb1158 in disguise).
    - Beat: pooled_RAE (mean across 5 KF seeds) must be <= 0.4697 - 0.003
      = 0.4667 to promote (matching nb2076's nb2060-baseline gate).

EVALUATION:
    Scaffold 5-fold CV across 5 seeds (1001..1005), applying tri_vote per
    fold to the validation rows only. CDF anchor inside each fold is the
    fold-specific nb1158 validation slice (no leakage -- the CDF is built
    from nb1158 PREDICTIONS, not labels). Report pooled RAE per seed and
    mean across seeds.

DEPLOY:
    tri_vote applied to the full 513-row te vectors with anchor = nb1158 te.
    Saved to data/processed/te_nb2081.npy and (only if promote==True)
    submissions/nb2081_diverse_vote.csv.

Inputs (all verified to exist):
    OOFs on 253 (unblind):
        data/processed/nb1158_mean_bag_oof_K32.npy
        data/processed/nb503_pred_oof.npy
        data/processed/nb562_pred_oof.npy
    te on 513:
        data/processed/te_nb1158.npy
        data/processed/te_nb503.npy
        data/processed/te_nb562.npy
    Audit:
        data/processed/_audit_unblind_idx.npy
        data/processed/_audit_unblind_y.npy

Outputs:
    scripts/nb2081_diverse_vote.py  (this file)
    data/processed/nb2081_summary.json
    data/processed/te_nb2081.npy
    submissions/nb2081_diverse_vote.csv   (only if promote=True)
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
from scipy.stats import rankdata, spearmanr

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2081"
GATE_MARGIN = 0.003
NB2060_REF_RAE = 0.4697
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
SPEARMAN_MAX_ANCHOR = 0.98          # vote must not echo anchor
SPEARMAN_MAX_PAIRWISE = 0.95        # candidates must be diverse

# Individual baselines (memo'd honest cross-fit numbers)
INDIV_REF = {
    "nb1158": 0.4902,
    "nb503":  0.5116,
    "nb562":  0.5065,
}


def empirical_cdf_map(anchor_values: np.ndarray, target_ranks: np.ndarray,
                      n_anchor: int) -> np.ndarray:
    """Map rank vector (1..n_anchor) back to pec50 via sorted anchor values.

    Linear interpolation between anchor-rank positions; rank interpreted as a
    continuous quantile position in the anchor sorted array.
    """
    anchor_sorted = np.sort(anchor_values)
    pos = np.clip(target_ranks - 1.0, 0.0, n_anchor - 1.0)
    lo = np.floor(pos).astype(int)
    hi = np.clip(lo + 1, 0, n_anchor - 1)
    frac = pos - lo
    return (1.0 - frac) * anchor_sorted[lo] + frac * anchor_sorted[hi]


def tri_vote(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
             anchor_values: np.ndarray):
    """Per-row median rank of 3 vectors, mapped back via anchor CDF.

    Returns (vote_pec50, rank_vote).
    """
    assert p1.shape == p2.shape == p3.shape == anchor_values.shape, \
        f"shape mismatch: {p1.shape} {p2.shape} {p3.shape} {anchor_values.shape}"
    n = len(p1)
    r1 = rankdata(p1, method="average")
    r2 = rankdata(p2, method="average")
    r3 = rankdata(p3, method="average")
    rank_vote = np.median(np.column_stack([r1, r2, r3]), axis=1)
    return empirical_cdf_map(anchor_values, rank_vote, n), rank_vote


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- diverse tri-vote {{nb1158, nb503, nb562}}")
    print("=" * 78)

    te = load_test()
    te_names = te["name"].values
    te_smiles = te["smiles"].values
    n_te = len(te_names)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_te={n_te}  n_unb={n_unb}")

    unb_smiles = te_smiles[unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]

    # --- Load the 3 candidates (all cached -- no reconstruction) ---
    print("\n[load] candidates")
    nb1158_oof = np.load(DATA_PROCESSED / "nb1158_mean_bag_oof_K32.npy").astype(np.float64)
    nb503_oof  = np.load(DATA_PROCESSED / "nb503_pred_oof.npy").astype(np.float64)
    nb562_oof  = np.load(DATA_PROCESSED / "nb562_pred_oof.npy").astype(np.float64)
    nb1158_te  = np.load(DATA_PROCESSED / "te_nb1158.npy").astype(np.float64)
    nb503_te   = np.load(DATA_PROCESSED / "te_nb503.npy").astype(np.float64)
    nb562_te   = np.load(DATA_PROCESSED / "te_nb562.npy").astype(np.float64)
    assert nb1158_oof.shape == nb503_oof.shape == nb562_oof.shape == (n_unb,)
    assert nb1158_te.shape  == nb503_te.shape  == nb562_te.shape  == (n_te,)

    indiv = {
        "nb1158": float(rae(y_unb, nb1158_oof)),
        "nb503":  float(rae(y_unb, nb503_oof)),
        "nb562":  float(rae(y_unb, nb562_oof)),
    }
    for name, val in indiv.items():
        ref = INDIV_REF[name]
        gap = val - ref
        print(f"   {name}  oof_RAE={val:.4f}  (memo={ref:.4f}, gap={gap:+.4f})  "
              f"te(513) mean={(nb1158_te if name=='nb1158' else nb503_te if name=='nb503' else nb562_te).mean():.3f} "
              f"std={(nb1158_te if name=='nb1158' else nb503_te if name=='nb503' else nb562_te).std():.3f}")

    # --- Pairwise Spearman among candidates (diversity check) ---
    print("\n[diversity] pairwise Spearman among candidate OOFs")
    rho_15_503 = float(spearmanr(nb1158_oof, nb503_oof).correlation)
    rho_15_562 = float(spearmanr(nb1158_oof, nb562_oof).correlation)
    rho_503_562 = float(spearmanr(nb503_oof,  nb562_oof).correlation)
    pairwise = {
        "nb1158_vs_nb503":  rho_15_503,
        "nb1158_vs_nb562":  rho_15_562,
        "nb503_vs_nb562":   rho_503_562,
    }
    for k, v in pairwise.items():
        print(f"   rho({k:<18}) = {v:.4f}")
    max_pairwise = max(pairwise.values())
    diversity_pass = max_pairwise < SPEARMAN_MAX_PAIRWISE
    print(f"[diversity] max pairwise = {max_pairwise:.4f}  "
          f"(<{SPEARMAN_MAX_PAIRWISE} ? {'PASS' if diversity_pass else 'FAIL'})")

    # --- Tri-vote on full 253 (deploy diagnostic; anchor CDF = nb1158 unb) ---
    vote_oof_global, rank_vote_unb = tri_vote(
        nb1158_oof, nb503_oof, nb562_oof, nb1158_oof,
    )
    rae_global = float(rae(y_unb, vote_oof_global))
    rho_nb1158 = float(spearmanr(rank_vote_unb, rankdata(nb1158_oof)).correlation)
    rho_nb503  = float(spearmanr(rank_vote_unb, rankdata(nb503_oof)).correlation)
    rho_nb562  = float(spearmanr(rank_vote_unb, rankdata(nb562_oof)).correlation)
    print(f"\n[global] tri-vote OOF RAE (anchor=nb1158)   = {rae_global:.4f}")
    print(f"[global] Spearman(rank_vote, nb1158)         = {rho_nb1158:.4f}")
    print(f"[global] Spearman(rank_vote, nb503)          = {rho_nb503:.4f}")
    print(f"[global] Spearman(rank_vote, nb562)          = {rho_nb562:.4f}")
    distinct_pass = rho_nb1158 <= SPEARMAN_MAX_ANCHOR
    print(f"[distinct] rho(vote, nb1158) {rho_nb1158:.4f} <= "
          f"{SPEARMAN_MAX_ANCHOR} -> {'PASS' if distinct_pass else 'FAIL'}")

    # --- Scaffold 5-fold CV across 5 seeds (apply tri_vote per fold) ---
    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV  kf_seeds={KF_SEEDS}")
    print("-" * 78)
    per_seed = []
    all_oofs = []
    for kf_seed in KF_SEEDS:
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        oof_blend = np.full(n_unb, np.nan)
        for tr_loc, va_loc in splits:
            vote_va, _ = tri_vote(
                nb1158_oof[va_loc],
                nb503_oof[va_loc],
                nb562_oof[va_loc],
                nb1158_oof[va_loc],     # CDF anchor = nb1158 va slice
            )
            oof_blend[va_loc] = vote_va
        pooled = float(rae(y_unb, oof_blend))
        per_seed.append({"kf_seed": int(kf_seed), "pooled_rae": pooled})
        all_oofs.append(oof_blend)
        print(f"   seed={kf_seed}  pooled_RAE={pooled:.4f}")
    pooled_mean = float(np.mean([r["pooled_rae"] for r in per_seed]))
    pooled_std = float(np.std([r["pooled_rae"] for r in per_seed]))
    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    rae_of_mean = float(rae(y_unb, mean_oof))
    print(f"\n[cv] pooled_RAE (mean of seeds) = {pooled_mean:.4f} "
          f"(+/- {pooled_std:.4f})")
    print(f"[cv] RAE of mean-of-seed OOFs   = {rae_of_mean:.4f}")

    # --- Deploy: tri-vote on 513, anchor = nb1158 te(513) ---
    vote_te, rank_vote_te = tri_vote(nb1158_te, nb503_te, nb562_te, nb1158_te)
    deploy_te = vote_te.astype(np.float32)
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))
    print(f"\n[deploy] te(513) tri-vote: mean={deploy_te.mean():.3f} "
          f"std={deploy_te.std():.3f}")
    print(f"[deploy] te[unb_idx] in-sample RAE = {te_unb_rae:.4f}")

    # --- Decision gate ---
    print("\n" + "-" * 78)
    print("DECISION GATE")
    print("-" * 78)
    delta_vs_nb2060 = pooled_mean - NB2060_REF_RAE
    delta_vs_nb1158 = pooled_mean - INDIV_REF["nb1158"]
    delta_vs_nb503  = pooled_mean - INDIV_REF["nb503"]
    delta_vs_nb562  = pooled_mean - INDIV_REF["nb562"]
    print(f"   tri-vote OOF (pooled mean of seeds)  = {pooled_mean:.4f}")
    print(f"   nb2060 ref OOF                       = {NB2060_REF_RAE:.4f}")
    print(f"   delta vs nb2060 (negative = better)  = {delta_vs_nb2060:+.4f}")
    print(f"   delta vs nb1158                      = {delta_vs_nb1158:+.4f}")
    print(f"   delta vs nb503                       = {delta_vs_nb503:+.4f}")
    print(f"   delta vs nb562                       = {delta_vs_nb562:+.4f}")
    print(f"   gate margin                          = -{GATE_MARGIN:.4f}")
    beats_nb2060 = delta_vs_nb2060 <= -GATE_MARGIN
    beats_nb1158 = delta_vs_nb1158 <= -GATE_MARGIN
    promote = bool(diversity_pass and distinct_pass and beats_nb2060)
    print(f"   beats nb2060 by margin               = "
          f"{'YES' if beats_nb2060 else 'NO'}")
    print(f"   beats nb1158 (best indiv) by margin  = "
          f"{'YES' if beats_nb1158 else 'NO'}")
    print(f"   diversity gate                       = "
          f"{'PASS' if diversity_pass else 'FAIL'}")
    print(f"   distinctness gate                    = "
          f"{'PASS' if distinct_pass else 'FAIL'}")
    print(f"   promote                              = "
          f"{'YES' if promote else 'NO'}")

    # --- Save artefacts ---
    te_npy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_npy_path, deploy_te)
    print(f"\n[save] {te_npy_path}")

    sub_csv_path = SUBMISSIONS / f"{TAG}_diverse_vote.csv"
    if promote:
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": deploy_te,
        }).to_csv(sub_csv_path, index=False)
        print(f"[save] {sub_csv_path}  (promote PASSED)")
    else:
        print(f"[skip] promote=NO -- no submission CSV written")

    summary = {
        "tag": TAG,
        "method": "tri_vote_median_rank_anchor_nb1158_cdf_DIVERSE",
        "candidates": ["nb1158", "nb503", "nb562"],
        "n_unb": int(n_unb),
        "n_te": int(n_te),
        "indiv_oof_rae_unb": indiv,
        "indiv_ref_memo": INDIV_REF,
        "pairwise_spearman": pairwise,
        "max_pairwise_spearman": max_pairwise,
        "spearman_max_pairwise_gate": SPEARMAN_MAX_PAIRWISE,
        "diversity_pass": bool(diversity_pass),
        "global_oof_rae_anchor_nb1158": rae_global,
        "spearman_vote_vs_nb1158": rho_nb1158,
        "spearman_vote_vs_nb503":  rho_nb503,
        "spearman_vote_vs_nb562":  rho_nb562,
        "spearman_max_anchor_gate": SPEARMAN_MAX_ANCHOR,
        "distinct_pass": bool(distinct_pass),
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "per_seed_results": per_seed,
        "pooled_rae_mean_seeds": pooled_mean,
        "pooled_rae_std_seeds": pooled_std,
        "rae_of_mean_of_seed_oofs": rae_of_mean,
        "te_unb_rae_in_sample": te_unb_rae,
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "nb2060_ref_oof_rae": NB2060_REF_RAE,
        "delta_vs_nb2060": delta_vs_nb2060,
        "delta_vs_nb1158": delta_vs_nb1158,
        "delta_vs_nb503":  delta_vs_nb503,
        "delta_vs_nb562":  delta_vs_nb562,
        "gate_margin": GATE_MARGIN,
        "beats_nb2060_by_margin": bool(beats_nb2060),
        "beats_nb1158_by_margin": bool(beats_nb1158),
        "promote": promote,
        "te_npy_path": str(te_npy_path),
        "submission_csv": str(sub_csv_path) if promote else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    json_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {json_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   pooled OOF RAE (cross-fit)        = {pooled_mean:.4f}")
    print(f"   max pairwise Spearman             = {max_pairwise:.4f}")
    print(f"   rho(vote, nb1158)                 = {rho_nb1158:.4f}")
    print(f"   delta vs nb2060                   = {delta_vs_nb2060:+.4f}")
    print(f"   delta vs nb1158                   = {delta_vs_nb1158:+.4f}")
    print(f"   diversity / distinct / beat       = "
          f"{diversity_pass} / {distinct_pass} / {beats_nb2060}")
    print(f"   promote                           = {promote}")
    print(f"   wall                              = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "pooled_rae_mean_seeds",
        "max_pairwise_spearman",
        "spearman_vote_vs_nb1158",
        "delta_vs_nb2060",
        "delta_vs_nb1158",
        "diversity_pass",
        "distinct_pass",
        "beats_nb2060_by_margin",
        "promote",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
