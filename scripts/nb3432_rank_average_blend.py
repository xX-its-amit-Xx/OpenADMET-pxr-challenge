"""nb3432 -- Rank-average blend of nb3200 + decorrelated anchors.

NEW PARADIGM (rank-blend, NOT value-blend):
    Average the RANKS (not the values) of three decorrelated predictors
    {nb3200, nb2171, nb1191}, then map the averaged rank back onto nb3200's
    value distribution by quantile mapping. Rank-blending is robust to scale
    differences between decorrelated predictors -- each predictor contributes
    only its ordering, and the final values are drawn from nb3200's marginal
    (the strongest anchor, honest cross-fit RAE ~0.4424).

Anchors (all chemprop_aux-lineage, the only verified-clean PRE-unblind family):
    nb3200  deep-30 learned-clip on nb3090 (honest OOF RAE ~0.4424)  -- PRIMARY
    nb2171  nb1162 anchor-swap pyramid (honest OOF RAE ~0.4676)
    nb1191  PRE-unblind pyramid         (honest OOF RAE ~0.4703)

DATA:
    Honest 5-fold cross-fit OOF on the 253 unblind (LB-faithful) is available
    for ALL THREE anchors:
        data/processed/nb3200_pred_oof.npy   (253,)
        data/processed/nb2171_pred_oof.npy   (253,)
        data/processed/nb1191_pred_oof.npy   (253,)
    -> the honest metric below uses these (no te[unb] caveat needed).

    Deploy te (513) refit-on-all-labels vectors:
        data/processed/te_nb3200.npy   (513,)
        data/processed/te_nb2171.npy   (513,)
        data/processed/te_nb1191.npy   (513,)
    te[unb_idx] is IN-SAMPLE (model saw the 253 truth) -> reported separately
    as an optimistic lower bound only, NOT the gate metric.

RANK-AVERAGE OPERATOR (rank_blend):
    given predictor matrix P (n, K):
        R[:,k] = rankdata(P[:,k]) / n              # uniform [~0,1] per predictor
        mean_rank = mean_k R[:,k]                   # averaged normalized rank
        out = quantile_map(mean_rank, ref_sorted)   # map back to ref distribution
    where ref_sorted = sorted nb3200 values. quantile_map places each row at the
    (mean_rank)-th quantile of ref_sorted via linear interpolation. This yields a
    blend whose ORDER is the rank-consensus and whose VALUES are nb3200-marginal.

HONEST PROTOCOL (gate metric):
    For each of 15 fresh kf_seeds:
      scaffold-5-fold split the 253 unblind. For each fold:
        - fit the quantile-map REFERENCE (sorted nb3200 OOF values) on TRAIN only
        - compute averaged normalized ranks of validation rows by inserting them
          into the TRAIN rank distribution per predictor (rank-honest: a val row's
          normalized rank = fraction of TRAIN values it exceeds), then map through
          the TRAIN reference distribution.
      pool the validation predictions across the 5 folds -> pooled OOF RAE.
    Mean / std of pooled RAE over the 15 seeds is the honest number.

    A full-pool (in-sample) rank-blend RAE is also reported as the optimistic
    bound (reference distribution + ranks computed on all 253 at once).

GATE:
    honest pooled mean RAE < 0.4414  -> verdict "BETTER"
    else                              -> verdict "FAIL"

OUTPUTS:
    scripts/nb3432_rank_average_blend.py   (this file)
    data/processed/nb3432_summary.json
    data/processed/nb3432_pred_oof.npy     (253,) honest pooled OOF (median seed)
    data/processed/te_nb3432.npy           (513,) deploy rank-blend
    submissions/nb3432_rank_average_blend.csv  (always written; deploy vector)
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
from scipy.stats import rankdata

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3432"
GATE_BETTER = 0.4414      # honest pooled mean RAE must be strictly below this
N_FOLDS = 5
KF_SEEDS = list(range(1300, 1315))   # 15 fresh seeds (distinct from nb3200's 1186-1215)
LB_W_OOF = 0.51
LB_W_TE = 0.49

# anchor order: nb3200 is the REFERENCE (index 0) whose value distribution we map onto.
ANCHORS = [
    ("nb3200", "nb3200_pred_oof.npy", "te_nb3200.npy"),
    ("nb2171", "nb2171_pred_oof.npy", "te_nb2171.npy"),
    ("nb1191", "nb1191_pred_oof.npy", "te_nb1191.npy"),
]
REF_IDX = 0   # nb3200 marginal is the quantile-map target

# baselines for the comparison block
NB3200_HONEST_DEEP30 = 0.4424
NB2171_HONEST = 0.4676
NB1191_HONEST = 0.4703


# ---------------------------------------------------------------------------
# Rank-average operator
# ---------------------------------------------------------------------------

def _normed_rank(x: np.ndarray) -> np.ndarray:
    """rankdata -> (0,1] normalized rank (average ties)."""
    n = len(x)
    return rankdata(x, method="average") / n


def _normed_rank_vs_ref(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Normalized rank of each x against a fixed reference vector `ref`.

    Honest version for held-out rows: a value's normalized rank is the fraction
    of `ref` values it is >= (searchsorted 'right' / n_ref), squashed into (0,1).
    """
    ref_sorted = np.sort(ref)
    n_ref = len(ref_sorted)
    # number of ref values <= x  -> in [0, n_ref]; map to (0,1] center-of-bin
    pos = np.searchsorted(ref_sorted, x, side="right").astype(np.float64)
    return (pos - 0.5) / n_ref


def _quantile_map(u: np.ndarray, ref_sorted: np.ndarray) -> np.ndarray:
    """Map normalized positions u in (0,1) onto the sorted reference values via
    linear interpolation of the reference empirical quantile function."""
    n_ref = len(ref_sorted)
    # reference quantile positions for its sorted order-statistics
    ref_q = (np.arange(n_ref) + 0.5) / n_ref
    u_c = np.clip(u, ref_q[0], ref_q[-1])
    return np.interp(u_c, ref_q, ref_sorted)


def rank_blend_full(P: np.ndarray, ref_idx: int = REF_IDX) -> np.ndarray:
    """In-sample (full-pool) rank-average blend on matrix P (n, K)."""
    R = np.column_stack([_normed_rank(P[:, k]) for k in range(P.shape[1])])
    mean_rank = R.mean(axis=1)
    ref_sorted = np.sort(P[:, ref_idx])
    return _quantile_map(mean_rank, ref_sorted)


def rank_blend_traineval(
    P_tr: np.ndarray, P_va: np.ndarray, ref_idx: int = REF_IDX
) -> np.ndarray:
    """Honest train->val rank-average blend.

    Reference value distribution AND per-predictor rank scales are fit on TRAIN
    only; validation rows are scored against those fixed train statistics.
    """
    K = P_tr.shape[1]
    R_va = np.column_stack(
        [_normed_rank_vs_ref(P_va[:, k], P_tr[:, k]) for k in range(K)]
    )
    mean_rank_va = R_va.mean(axis=1)
    ref_sorted = np.sort(P_tr[:, ref_idx])
    return _quantile_map(mean_rank_va, ref_sorted)


# ---------------------------------------------------------------------------
# Honest CV
# ---------------------------------------------------------------------------

def cv_pooled_for_seed(P_unb, y_unb, scaffolds, kf_seed):
    splits = scaffold_kfold_indices(
        scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed
    )
    oof = np.full(P_unb.shape[0], np.nan)
    for tr_loc, va_loc in splits:
        oof[va_loc] = rank_blend_traineval(P_unb[tr_loc], P_unb[va_loc])
    assert not np.isnan(oof).any(), "OOF has NaN -- fold coverage gap"
    return float(rae(y_unb, oof)), oof


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- rank-average blend (nb3200 + nb2171 + nb1191)")
    print("=" * 78)

    te = load_test()
    te_names = te["name"].values
    te_smiles = te["smiles"].values
    n_te = len(te_names)

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_te={n_te}  n_unb={n_unb}  truth_std={y_unb.std():.4f}")

    unb_smiles = te_smiles[unb_idx]
    scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    # ---- load anchors (honest OOF + te) ----
    print("\n[anchors]  (honest 5-fold cross-fit OOF on 253)")
    oof_cols, te_cols, indiv = [], [], {}
    has_honest = True
    for disp, oof_rel, te_rel in ANCHORS:
        oof_p = DATA_PROCESSED / oof_rel
        te_p = DATA_PROCESSED / te_rel
        assert oof_p.exists(), f"missing OOF: {oof_p}"
        assert te_p.exists(), f"missing te: {te_p}"
        oof = np.load(oof_p).astype(np.float64)
        te_arr = np.load(te_p).astype(np.float64)
        assert oof.shape == (n_unb,), f"{disp} oof {oof.shape}"
        assert te_arr.shape == (n_te,), f"{disp} te {te_arr.shape}"
        r = float(rae(y_unb, oof))
        indiv[disp] = r
        oof_cols.append(oof)
        te_cols.append(te_arr)
        print(
            f"   {disp:8s} honest_OOF_RAE={r:.4f}  "
            f"oof[std={oof.std():.3f}]  te[mean={te_arr.mean():.3f} "
            f"std={te_arr.std():.3f}]"
        )

    P_unb = np.column_stack(oof_cols)
    P_te = np.column_stack(te_cols)

    # cross-anchor rank correlations (decorrelation check)
    print("\n[rank-corr] Spearman between anchor OOFs")
    Rn = np.column_stack([_normed_rank(P_unb[:, k]) for k in range(P_unb.shape[1])])
    corr = np.corrcoef(Rn.T)
    names = [a[0] for a in ANCHORS]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            print(f"   {names[i]} vs {names[j]}: rho={corr[i, j]:.4f}")

    # =================================================================
    # Honest pooled RAE across 15 fresh seeds
    # =================================================================
    print("\n" + "-" * 78)
    print(f"HONEST scaffold-5-fold CV  kf_seeds={KF_SEEDS[0]}..{KF_SEEDS[-1]} (15)")
    print("-" * 78)
    pooled_list, oof_list = [], []
    for s in KF_SEEDS:
        pooled, oof = cv_pooled_for_seed(P_unb, y_unb, scaffolds, s)
        pooled_list.append(pooled)
        oof_list.append(oof)
        print(f"   seed={s}  pooled_honest_RAE={pooled:.4f}")
    pooled_arr = np.asarray(pooled_list)
    honest_mean = float(pooled_arr.mean())
    honest_std = float(pooled_arr.std())
    honest_median = float(np.median(pooled_arr))
    honest_min = float(pooled_arr.min())
    honest_max = float(pooled_arr.max())
    sem = honest_std / np.sqrt(len(pooled_arr))
    print(
        f"\n[honest] mean={honest_mean:.4f}  std={honest_std:.4f}  "
        f"sem={sem:.4f}  median={honest_median:.4f}  "
        f"range=[{honest_min:.4f}, {honest_max:.4f}]"
    )

    # median-seed OOF for the saved artefact
    median_seed_pos = int(np.argsort(pooled_arr)[len(pooled_arr) // 2])
    pred_oof = oof_list[median_seed_pos].astype(np.float32)
    pred_oof_rae = float(rae(y_unb, pred_oof))

    # in-sample (full-pool) optimistic bound
    insample_oof = rank_blend_full(P_unb)
    insample_rae = float(rae(y_unb, insample_oof))
    print(f"[in-sample] full-pool rank-blend RAE (optimistic) = {insample_rae:.4f}")

    # =================================================================
    # Deploy: rank-blend the 513 te vectors onto te_nb3200 marginal
    # =================================================================
    print("\n" + "-" * 78)
    print("DEPLOY  (rank-blend on 513 te, mapped to te_nb3200 marginal)")
    print("-" * 78)
    te_blend = rank_blend_full(P_te).astype(np.float32)
    te_unb_rae_in_sample = float(rae(y_unb, te_blend[unb_idx]))
    print(
        f"   te(513) mean/std = {te_blend.mean():.3f}/{te_blend.std():.3f}  "
        f"min/max = {te_blend.min():.3f}/{te_blend.max():.3f}"
    )
    print(
        f"   te[unb_idx] RAE  = {te_unb_rae_in_sample:.4f}  "
        f"(IN-SAMPLE optimistic, deploy refit -- NOT the gate)"
    )

    lb_band_est = LB_W_OOF * honest_mean + LB_W_TE * te_unb_rae_in_sample
    print(
        f"\n[LB-band] {LB_W_OOF:.2f}*honest({honest_mean:.4f}) + "
        f"{LB_W_TE:.2f}*te_unb({te_unb_rae_in_sample:.4f}) = {lb_band_est:.4f}  "
        f"[{lb_band_est - 0.05:.4f}, {lb_band_est + 0.05:.4f}]"
    )

    # =================================================================
    # Gate
    # =================================================================
    is_better = honest_mean < GATE_BETTER
    verdict = "BETTER" if is_better else "FAIL"
    delta_vs_nb3200 = honest_mean - NB3200_HONEST_DEEP30
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    print(
        f"   honest pooled mean {honest_mean:.4f} < {GATE_BETTER:.4f}  "
        f"-> {verdict}"
    )
    print(
        f"   delta vs nb3200 deep-30 ({NB3200_HONEST_DEEP30:.4f}) = "
        f"{delta_vs_nb3200:+.4f}"
    )

    # =================================================================
    # Save artefacts
    # =================================================================
    pred_oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(pred_oof_path, pred_oof)
    np.save(te_path, te_blend)
    print(f"\n[save] {pred_oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_rank_average_blend.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": te_blend,
    }).to_csv(sub_csv, index=False)
    print(f"[save] {sub_csv}")

    summary = {
        "tag": TAG,
        "method": "rank_average_blend_quantile_map_to_nb3200_marginal",
        "paradigm": (
            "average normalized RANKS of {nb3200, nb2171, nb1191}, then "
            "quantile-map mean_rank onto nb3200's value distribution"
        ),
        "anchors": [a[0] for a in ANCHORS],
        "anchor_oof_paths": [a[1] for a in ANCHORS],
        "anchor_te_paths": [a[2] for a in ANCHORS],
        "ref_anchor": ANCHORS[REF_IDX][0],
        "honest_oof_available": bool(has_honest),
        "honest_oof_caveat": (
            "all three anchors have honest 5-fold cross-fit pred_oof on the 253; "
            "the gate uses these (no te[unb] proxy needed)"
        ),
        "indiv_honest_oof_rae_unb": indiv,
        "anchor_rank_corr": {
            f"{names[i]}__{names[j]}": float(corr[i, j])
            for i in range(len(names)) for j in range(i + 1, len(names))
        },
        "kf_seeds": KF_SEEDS,
        "n_seeds": len(KF_SEEDS),
        "n_folds": N_FOLDS,
        "n_unb": n_unb,
        "n_te": n_te,
        "n_unique_scaffolds": n_unique_scaf,
        "truth_std": float(y_unb.std()),
        "pooled_rae_per_seed": [float(x) for x in pooled_arr],
        "honest_pooled_mean_rae": honest_mean,
        "honest_pooled_std_rae": honest_std,
        "honest_pooled_sem_rae": float(sem),
        "honest_pooled_median_rae": honest_median,
        "honest_pooled_min_rae": honest_min,
        "honest_pooled_max_rae": honest_max,
        "median_seed": int(KF_SEEDS[median_seed_pos]),
        "pred_oof_rae": pred_oof_rae,
        "in_sample_full_pool_rae": insample_rae,
        "te_unb_rae_in_sample": te_unb_rae_in_sample,
        "in_sample_optimism_gap": honest_mean - te_unb_rae_in_sample,
        "lb_band_estimate": lb_band_est,
        "lb_band_low": lb_band_est - 0.05,
        "lb_band_high": lb_band_est + 0.05,
        "lb_band_w_oof": LB_W_OOF,
        "lb_band_w_te": LB_W_TE,
        "gate_better_target": GATE_BETTER,
        "delta_vs_nb3200_deep30": delta_vs_nb3200,
        "compare_nb3200_honest_deep30": NB3200_HONEST_DEEP30,
        "compare_nb2171_honest": NB2171_HONEST,
        "compare_nb1191_honest": NB1191_HONEST,
        "is_better": bool(is_better),
        "verdict": verdict,
        "te_mean": float(te_blend.mean()),
        "te_std": float(te_blend.std()),
        "te_min": float(te_blend.min()),
        "te_max": float(te_blend.max()),
        "pred_oof_path": str(pred_oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv),
        "wall_sec": round(time.time() - t0, 2),
    }
    json_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {json_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   anchors (honest OOF) : nb3200={indiv['nb3200']:.4f}  "
          f"nb2171={indiv['nb2171']:.4f}  nb1191={indiv['nb1191']:.4f}")
    print(f"   honest pooled mean   : {honest_mean:.4f} +/- {honest_std:.4f}")
    print(f"   in-sample (full-pool): {insample_rae:.4f}")
    print(f"   te[unb] in-sample    : {te_unb_rae_in_sample:.4f}")
    print(f"   gate (< {GATE_BETTER})       : {verdict}")
    print(f"   wall                 : {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== RESULT ====")
    for k in (
        "honest_pooled_mean_rae",
        "honest_pooled_std_rae",
        "in_sample_full_pool_rae",
        "te_unb_rae_in_sample",
        "delta_vs_nb3200_deep30",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
