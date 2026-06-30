"""nb3280 -- Conformal prediction interval MIDPOINT calibration on nb3090.

NEW PARADIGM (cycle 250+):
    Split-conformal calibration normally produces a prediction *interval*
    [center - q, center + q] where q is a quantile of the calibration-set
    conformity scores |y_cal - pred_cal|. The interval MIDPOINT is the point
    prediction. Crucially, that midpoint need not equal the raw anchor: if we
    re-center each region of the calibration set on its LOCAL mean residual,
    the midpoint becomes a per-region bias-corrected estimate. This may correct
    systematic bias DIFFERENTLY than the quantile-clip primitive (nb3170 etc.),
    because clip only touches the tails while a conformal local-mean shift
    moves the body of every similarity region.

    Biological hook (Phase-1 post-mortem F2 lever): novel-scaffold inactives
    are over-predicted by +1.23 RAE. Those rows live in the LOW train-similarity
    region. A calibration-set local-mean shift in the low-sim bin should be
    strongly negative and pull exactly those rows down -- a correction the
    symmetric quantile clip cannot make region-aware.

PROTOCOL (per kf_seed, 5-fold scaffold split on the 253 unblind):
    anchor = nb3090_pred_oof (quantile-conditional K18/K19 blend), RAE 0.4470.
    Binning axis = top-1 Tanimoto similarity of each unblind row to the 4139
    TRAIN compounds (computed once, leak-free: train carries no unblind labels).
    Per OUTER fold:
        a) Split fold-train rows -> proper-train (70%) + calibration (30%)
           (random split, seeded by kf_seed+fold; proper-train is unused by the
           anchor since the anchor is precomputed -- it only defines which rows
           may donate bin-edge / lambda info, mirroring split-conformal's
           proper/calib separation so no val leakage occurs).
        b) Inner grid on fold-train ONLY (proper-train edges + calib deltas):
             for n_bins in {3, 4, 5}:
               for lam (shrink strength) in {0, 2, 5, 10}:
                 - bin edges from sim quantiles of proper-train
                 - per-bin signed delta = mean(y_cal - anchor_cal) in that bin
                 - shrink: delta_b *= n_b / (n_b + lam)  (James-Stein toward 0)
                 - global fallback delta for empty calib bins
                 - corrected_tr = anchor_tr + delta[bin(tr)]
                 - score on the held-in proper-train+calib (= fold-train) RAE
           Pick (n_bins*, lam*) minimizing fold-train RAE.
        c) Refit chosen (n_bins*, lam*) deltas on the FULL fold-train calib mass
           and apply to fold-VAL: val_pred = anchor_val + delta[bin(val)].
           Conformity band q = quantile(|y_cal - anchor_cal|, 0.9) recorded as
           the interval half-width (diagnostic; midpoint is the deliverable).
        d) Stitch into oof; pooled + per-fold-val RAE.
    Repeat for 15 FRESH kf_seeds {1216..1230}; primary metric = per-fold-mean.

GATE (on 15-seed per-fold-mean):
    mean < 0.4423 -> "BETTER"
    else          -> "FAIL"

References:
    nb3090 anchor (quantile-conditional blend) full OOF       = 0.4470
    nb3170 fixed q05/q95 clip on nb3080                        = 0.4437
    nb3001 wide-15-seed 3K mean                               = 0.4511
    nb2992 per-fold simplex 3K                                = 0.4479
    nb2171 prior post-hoc top                                 = 0.4682

Inputs (all PRE-unblind; anchor trained on <320-era models):
    data/processed/nb3090_pred_oof.npy   (253,) float32 -- anchor OOF
    data/processed/te_nb3090.npy         (513,) float32 -- anchor deploy te
    data/processed/_audit_unblind_idx.npy (253,) int64
    data/processed/_audit_unblind_y.npy   (253,) float64
    (train SMILES via pxr.data.load_train for similarity axis)

Outputs:
    data/processed/nb3280_summary.json
    data/processed/nb3280_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3280.npy         (513,) float32 -- deploy te
    submissions/nb3280_conformal_calibration.csv  (only on BETTER verdict)
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from collections import Counter

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko, morgan_fp_batch, standardize_smiles
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3280"
ANCHOR_TAG = "nb3090"

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))  # 15 FRESH seeds {1216..1230}
CALIB_FRAC = 0.30                   # split-conformal calibration fraction

# -- Inner grid (chosen per outer fold on fold-train only) ---------------------
N_BINS_GRID = [3, 4, 5]
LAMBDA_GRID = [0.0, 2.0, 5.0, 10.0]  # James-Stein shrink strength on bin deltas
CONFORMAL_ALPHA = 0.90               # band quantile (half-width, diagnostic)

# -- Gate ----------------------------------------------------------------------
GATE_BETTER = 0.4423

# -- References ----------------------------------------------------------------
REF_NB3090_ANCHOR = 0.4470
REF_NB3170_FIXED = 0.4437
REF_NB3001 = 0.4511
REF_NB2992 = 0.4479
REF_NB2171 = 0.4682


def tanimoto_max_to_ref(fp_query: np.ndarray, fp_ref: np.ndarray,
                        block: int = 512) -> np.ndarray:
    """Top-1 Tanimoto of each query row to the reference set (blocked matmul).

    Returns (n_query,) float32 = max similarity to any reference compound.
    """
    Q = fp_query.astype(np.float32)
    R = fp_ref.astype(np.float32)
    r_sum = R.sum(axis=1)  # (n_ref,)
    out = np.zeros(Q.shape[0], dtype=np.float32)
    for s in range(0, Q.shape[0], block):
        e = min(s + block, Q.shape[0])
        Qb = Q[s:e]
        inter = Qb @ R.T                       # (b, n_ref)
        q_sum = Qb.sum(axis=1, keepdims=True)  # (b, 1)
        denom = q_sum + r_sum[None, :] - inter
        sim = np.where(denom > 0, inter / denom, 0.0)
        out[s:e] = sim.max(axis=1)
    return out


def _fit_bin_deltas(
    sim_edges: np.ndarray,
    sim_cal: np.ndarray,
    resid_cal: np.ndarray,
    lam: float,
    n_bins: int,
) -> tuple[np.ndarray, float]:
    """Per-bin shrunk signed delta = shrink * mean(y_cal - anchor_cal) in bin.

    resid_cal = y_cal - anchor_cal (signed). Returns (deltas[n_bins],
    global_delta). Bins with no calib support fall back to global_delta.
    Shrinkage pulls sparse-bin deltas toward 0 (James-Stein), NOT toward the
    global mean, so a well-supported low-sim bin keeps its full bias correction.
    """
    global_delta = float(np.mean(resid_cal)) if resid_cal.size else 0.0
    bin_idx = np.clip(np.digitize(sim_cal, sim_edges[1:-1]), 0, n_bins - 1)
    deltas = np.full(n_bins, global_delta, dtype=np.float64)
    for b in range(n_bins):
        m = bin_idx == b
        n_b = int(m.sum())
        if n_b == 0:
            deltas[b] = global_delta
        else:
            raw = float(np.mean(resid_cal[m]))
            shrink = n_b / (n_b + lam) if (n_b + lam) > 0 else 1.0
            deltas[b] = raw * shrink
    return deltas, global_delta


def _apply_bin_deltas(
    sim_q: np.ndarray,
    sim_edges: np.ndarray,
    deltas: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """Map query similarities to bins and return the per-row delta to add."""
    bin_idx = np.clip(np.digitize(sim_q, sim_edges[1:-1]), 0, n_bins - 1)
    return deltas[bin_idx]


def _edges_from_quantiles(sim_ref: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-spaced bin edges (balanced support) with safe degenerate guard."""
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(sim_ref, qs)
    # ensure strictly increasing interior edges
    edges = np.maximum.accumulate(edges)
    eps = 1e-6
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + eps
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _pick_best_config(
    sim_tr: np.ndarray,
    y_tr: np.ndarray,
    anchor_tr: np.ndarray,
    cal_mask: np.ndarray,
) -> tuple[int, float, np.ndarray, np.ndarray]:
    """Inner grid: pick (n_bins*, lam*) minimizing fold-train RAE.

    Edges come from PROPER-train similarities; deltas from CALIB residuals.
    Scoring is on the full fold-train (proper + calib) corrected prediction.
    """
    proper_mask = ~cal_mask
    sim_proper = sim_tr[proper_mask]
    sim_cal = sim_tr[cal_mask]
    resid_cal = y_tr[cal_mask] - anchor_tr[cal_mask]

    best_rae = np.inf
    best_nb = N_BINS_GRID[0]
    best_lam = LAMBDA_GRID[0]
    best_edges = _edges_from_quantiles(sim_proper, best_nb)
    best_deltas = np.zeros(best_nb)

    for nb in N_BINS_GRID:
        edges = _edges_from_quantiles(sim_proper, nb)
        for lam in LAMBDA_GRID:
            deltas, _ = _fit_bin_deltas(edges, sim_cal, resid_cal, lam, nb)
            add = _apply_bin_deltas(sim_tr, edges, deltas, nb)
            corrected = anchor_tr + add
            r = float(rae(y_tr, corrected))
            if r < best_rae:
                best_rae = r
                best_nb = nb
                best_lam = lam
                best_edges = edges
                best_deltas = deltas
    return best_nb, best_lam, best_edges, best_deltas


def _run_one_seed(
    anchor_unb: np.ndarray,
    sim_unb: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Run conformal-midpoint calibration at a single kf_seed."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_nb = []
    fold_lam = []
    fold_band = []
    fold_low_bin_delta = []
    fold_n_shifted = []
    rng = np.random.default_rng(kf_seed)

    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        # split-conformal: carve calibration subset out of fold-train
        n_tr = len(tr_loc)
        perm = rng.permutation(n_tr)
        n_cal = max(2, int(round(CALIB_FRAC * n_tr)))
        cal_mask = np.zeros(n_tr, dtype=bool)
        cal_mask[perm[:n_cal]] = True

        sim_tr = sim_unb[tr_loc]
        y_tr = y_unb[tr_loc]
        anchor_tr = anchor_unb[tr_loc]

        nb, lam, _, _ = _pick_best_config(sim_tr, y_tr, anchor_tr, cal_mask)

        # Refit chosen config on FULL fold-train calib mass; apply to val.
        sim_cal = sim_tr[cal_mask]
        resid_cal = y_tr[cal_mask] - anchor_tr[cal_mask]
        edges = _edges_from_quantiles(sim_tr[~cal_mask], nb)
        deltas, _ = _fit_bin_deltas(edges, sim_cal, resid_cal, lam, nb)

        # conformity band (half-width) from calib abs residuals -- diagnostic
        band = (
            float(np.quantile(np.abs(resid_cal), CONFORMAL_ALPHA))
            if resid_cal.size else 0.0
        )
        fold_band.append(band)

        sim_va = sim_unb[va_loc]
        add_va = _apply_bin_deltas(sim_va, edges, deltas, nb)
        val_pred = anchor_unb[va_loc] + add_va  # interval midpoint
        oof[va_loc] = val_pred
        fold_val_raes.append(float(rae(y_unb[va_loc], val_pred)))
        fold_nb.append(int(nb))
        fold_lam.append(float(lam))
        fold_low_bin_delta.append(float(deltas[0]))   # lowest-sim region shift
        fold_n_shifted.append(int(np.sum(np.abs(add_va) > 1e-9)))

    if np.isnan(oof).any():
        raise RuntimeError(
            f"kf_seed={kf_seed}: scaffold splits did not cover all rows"
        )
    pooled = float(rae(y_unb, oof))
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(fold_val_raes)),
        "per_fold_val_rae_std": float(np.std(fold_val_raes, ddof=1)),
        "fold_nb": fold_nb,
        "fold_lam": fold_lam,
        "fold_band_mean": float(np.mean(fold_band)),
        "fold_low_bin_delta_mean": float(np.mean(fold_low_bin_delta)),
        "n_shifted": int(np.sum(fold_n_shifted)),
        "oof": oof,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- CONFORMAL prediction-interval MIDPOINT calibration on "
          f"{ANCHOR_TAG}")
    print(f"          calib_frac  = {CALIB_FRAC}")
    print(f"          n_bins_grid = {N_BINS_GRID}")
    print(f"          lambda_grid = {LAMBDA_GRID}  (James-Stein shrink)")
    print(f"          band alpha  = {CONFORMAL_ALPHA} (half-width diagnostic)")
    print(f"          kf_seeds    = {len(KF_SEEDS)} fresh "
          f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}")
    print(f"          gate: per-fold-mean < {GATE_BETTER:.4f} -> BETTER, "
          f"else FAIL")
    print("=" * 78)

    # -- Load test, truth ----------------------------------------------------
    te = load_test()
    n_test = len(te)
    te_smiles_raw = (
        te["smiles"].astype(str).tolist()
        if "smiles" in te.columns
        else te["SMILES"].astype(str).tolist()
    )
    te_names = (
        te["name"].values
        if "name" in te.columns
        else te["Molecule Name"].values
    )
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    assert n_unb == 253, f"expected 253 unblind, got {n_unb}"
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    # -- Load anchor OOF + te ------------------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 1: load anchor {ANCHOR_TAG} OOF (253) + te (513)")
    print("-" * 78)
    anchor_unb = np.load(
        DATA_PROCESSED / f"{ANCHOR_TAG}_pred_oof.npy"
    ).astype(np.float64)
    anchor_te = np.load(
        DATA_PROCESSED / f"te_{ANCHOR_TAG}.npy"
    ).astype(np.float64)
    if anchor_unb.shape != (n_unb,):
        raise ValueError(f"anchor oof shape {anchor_unb.shape} != ({n_unb},)")
    if anchor_te.shape != (n_test,):
        raise ValueError(f"anchor te shape {anchor_te.shape} != ({n_test},)")
    anchor_oof_rae = float(rae(y_unb, anchor_unb))
    leak_eq = float(np.mean(np.isclose(anchor_unb, y_unb, atol=1e-6)))
    print(f"   anchor oof_RAE = {anchor_oof_rae:.4f}  "
          f"mean={anchor_unb.mean():.3f} std={anchor_unb.std():.3f}")
    print(f"   anchor te:     mean={anchor_te.mean():.3f} "
          f"std={anchor_te.std():.3f} min={anchor_te.min():.3f} "
          f"max={anchor_te.max():.3f}")
    print(f"   anchor leak_eq_truth_frac = {leak_eq:.4f}")
    print(f"   y_unb stats: mean={y_unb.mean():.3f} std={y_unb.std():.3f} "
          f"min={y_unb.min():.3f} max={y_unb.max():.3f}")

    # -- Build train-similarity binning axis (leak-free) ----------------------
    print("\n" + "-" * 78)
    print("STEP 2: top-1 Tanimoto similarity to 4139 TRAIN (binning axis)")
    print("-" * 78)
    tr = load_train()
    tr_smiles = tr["smiles"].astype(str).tolist()
    tr_std = [standardize_smiles(s) or s for s in tr_smiles]
    te_std = [standardize_smiles(s) or s for s in te_smiles_raw]
    fp_tr = morgan_fp_batch(tr_std)        # (4139, 2048)
    fp_te = morgan_fp_batch(te_std)        # (513, 2048)
    sim_te = tanimoto_max_to_ref(fp_te, fp_tr)   # (513,)
    sim_unb = sim_te[unb_idx]                    # (253,)
    print(f"   sim_unb: mean={sim_unb.mean():.3f} std={sim_unb.std():.3f} "
          f"min={sim_unb.min():.3f} max={sim_unb.max():.3f} "
          f"median={np.median(sim_unb):.3f}")
    print(f"   sim_te : mean={sim_te.mean():.3f} std={sim_te.std():.3f} "
          f"min={sim_te.min():.3f} max={sim_te.max():.3f}")
    # sanity: low-sim region should carry positive over-prediction bias
    lo_mask = sim_unb <= np.quantile(sim_unb, 0.33)
    bias_lo = float(np.mean(y_unb[lo_mask] - anchor_unb[lo_mask]))
    bias_hi = float(np.mean(y_unb[~lo_mask] - anchor_unb[~lo_mask]))
    print(f"   signed resid (y-anchor): low-sim third={bias_lo:+.3f}  "
          f"rest={bias_hi:+.3f}  (neg low => anchor over-predicts low-sim)")

    # -- Scaffolds for outer CV ----------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 3: scaffolds for outer CV")
    print("-" * 78)
    unb_smiles = [te_smiles_raw[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"   n_unique_scaffolds = {n_unique_scaf}")

    # -- Multi-seed sweep -----------------------------------------------------
    print("\n" + "-" * 78)
    print(f"SWEEP: {len(KF_SEEDS)} FRESH kf_seeds "
          f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}")
    print("-" * 78)
    seed_records = []
    pooled_raes = []
    per_fold_means = []
    oof_stack = []
    all_fold_nb = []
    all_fold_lam = []
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(anchor_unb, sim_unb, y_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        per_fold_means.append(res["per_fold_val_rae_mean"])
        oof_stack.append(res["oof"])
        all_fold_nb.extend(res["fold_nb"])
        all_fold_lam.extend(res["fold_lam"])
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
            "per_fold_val_rae_std": round(res["per_fold_val_rae_std"], 4),
            "fold_nb": res["fold_nb"],
            "fold_lam": res["fold_lam"],
            "fold_band_mean": round(res["fold_band_mean"], 4),
            "fold_low_bin_delta_mean": round(res["fold_low_bin_delta_mean"], 4),
            "n_shifted": res["n_shifted"],
        })
        print(f"   kf={s}: pooled_RAE={res['pooled_rae']:.4f}  "
              f"pf_mean={res['per_fold_val_rae_mean']:.4f}  "
              f"nb={res['fold_nb']}  lam={res['fold_lam']}  "
              f"low_delta={res['fold_low_bin_delta_mean']:+.3f}  "
              f"band={res['fold_band_mean']:.3f}  "
              f"wall={time.time()-ts:.2f}s")

    pooled_arr = np.asarray(pooled_raes, dtype=np.float64)
    pf_arr = np.asarray(per_fold_means, dtype=np.float64)
    n_s = len(pooled_arr)
    pooled_mean = float(pooled_arr.mean())
    pooled_std = float(pooled_arr.std(ddof=1)) if n_s > 1 else 0.0
    pf_mean = float(pf_arr.mean())
    pf_std = float(pf_arr.std(ddof=1)) if n_s > 1 else 0.0
    sem = pf_std / np.sqrt(n_s) if n_s > 1 else 0.0
    t_mult = 2.145  # df=14, two-sided 95%
    ci_low = pf_mean - t_mult * sem
    ci_high = pf_mean + t_mult * sem
    median_pf = float(np.median(pf_arr))

    nb_counter = Counter(all_fold_nb)
    lam_counter = Counter(all_fold_lam)
    nb_mode = nb_counter.most_common(1)[0][0]
    lam_mode = lam_counter.most_common(1)[0][0]

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds)")
    print("-" * 78)
    print(f"   per-fold-mean RAE")
    print(f"     mean    = {pf_mean:.4f}")
    print(f"     std     = {pf_std:.4f}")
    print(f"     sem     = {sem:.4f}")
    print(f"     95% CI  = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"     median  = {median_pf:.4f}")
    print(f"     min/max = [{pf_arr.min():.4f}, {pf_arr.max():.4f}]")
    print(f"   pooled RAE")
    print(f"     mean    = {pooled_mean:.4f}")
    print(f"     std     = {pooled_std:.4f}")
    print(f"\n   ref nb3090 anchor OOF         = {anchor_oof_rae:.4f}  "
          f"<- anchor (no calibration)")
    print(f"   delta vs nb3090 anchor (pf)   = {pf_mean - anchor_oof_rae:+.4f}")
    print(f"   ref nb3170 fixed q05/q95      = {REF_NB3170_FIXED:.4f}")
    print(f"   delta vs nb3170 fixed (pf)    = {pf_mean - REF_NB3170_FIXED:+.4f}")
    print(f"   ref nb3001 wide-seed 3K mean  = {REF_NB3001:.4f}")
    print(f"   ref nb2992 per-fold simplex   = {REF_NB2992:.4f}")
    print(f"\n   n_bins_distribution (75 folds) = {dict(nb_counter)}")
    print(f"   lambda_distribution (75 folds) = {dict(lam_counter)}")
    print(f"   nb_mode = {nb_mode}  lam_mode = {lam_mode}")

    # -- Deploy: fit config on FULL 253 (calib carved from all 253) -----------
    print("\n" + "-" * 78)
    print("STEP 4: deploy config on FULL 253 -> apply to 513 te")
    print("-" * 78)
    rng_dep = np.random.default_rng(KF_SEEDS[0])
    perm = rng_dep.permutation(n_unb)
    n_cal = max(2, int(round(CALIB_FRAC * n_unb)))
    cal_mask_full = np.zeros(n_unb, dtype=bool)
    cal_mask_full[perm[:n_cal]] = True
    dep_nb, dep_lam, dep_edges, _ = _pick_best_config(
        sim_unb, y_unb, anchor_unb, cal_mask_full
    )
    # refit deltas on the full-253 calib mass with chosen config
    sim_cal_full = sim_unb[cal_mask_full]
    resid_cal_full = y_unb[cal_mask_full] - anchor_unb[cal_mask_full]
    dep_edges = _edges_from_quantiles(sim_unb[~cal_mask_full], dep_nb)
    dep_deltas, dep_global = _fit_bin_deltas(
        dep_edges, sim_cal_full, resid_cal_full, dep_lam, dep_nb
    )
    add_te = _apply_bin_deltas(sim_te, dep_edges, dep_deltas, dep_nb)
    te_pred = (anchor_te + add_te).astype(np.float32)
    n_te_shifted = int(np.sum(np.abs(add_te) > 1e-9))
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"   deploy config: n_bins={dep_nb}  lambda={dep_lam}")
    print(f"   deploy bin deltas = "
          f"{[round(float(d),3) for d in dep_deltas]}  "
          f"(global={dep_global:+.3f})")
    print(f"   bin edges (interior) = "
          f"{[round(float(e),3) for e in dep_edges[1:-1]]}")
    print(f"   te shifted rows = {n_te_shifted}/513")
    print(f"   te(513) mean={te_pred.mean():.3f} std={te_pred.std():.3f} "
          f"min={te_pred.min():.3f} max={te_pred.max():.3f}")
    print(f"   te[unb] in-sample RAE = {te_unb_in_rae:.4f}")

    # Median-seed OOF for storage (by per-fold mean ranking)
    med_seed_idx = int(np.argsort(pf_arr)[n_s // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(f"   median seed = {median_seed} "
          f"(pf_rae={pf_arr[med_seed_idx]:.4f})")

    # -- Gate -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    if pf_mean < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE-CANDIDATE. nb3280 15-seed per-fold-mean {pf_mean:.4f} "
            f"clears BETTER gate {GATE_BETTER:.4f}. Split-conformal MIDPOINT "
            f"calibration (per-similarity-bin local-mean shift) on the "
            f"{ANCHOR_TAG} anchor extracts gain over the raw anchor "
            f"({pf_mean - anchor_oof_rae:+.4f}). Modal config = "
            f"(n_bins={nb_mode}, lambda={lam_mode}). The conformal local-mean "
            f"re-centering corrects region-specific bias the symmetric "
            f"quantile-clip primitive cannot. Re-verify with deep-30 seed bag "
            f"before any PRIMARY-1 swap."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3280 15-seed per-fold-mean {pf_mean:.4f} fails BETTER "
            f"gate {GATE_BETTER:.4f}. Split-conformal midpoint calibration "
            f"(per-similarity-bin local-mean shift) on {ANCHOR_TAG} does NOT "
            f"beat the gate ({pf_mean - anchor_oof_rae:+.4f} vs raw anchor). "
            f"The conformal local-mean correction is paradigm-matched to the "
            f"post-hoc shift family already explored; binning the residual on "
            f"train-similarity does not unlock new signal at n=253. Hold "
            f"current ladder."
        )
    print(f"   verdict       = {verdict}")
    print(f"   ladder action = {ladder_action}")

    # -- Save artifacts -------------------------------------------------------
    print("\n" + "-" * 78)
    print("SAVE")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_for_save)
    np.save(te_path, te_pred)
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_conformal_calibration.csv"
    if verdict == "BETTER":
        pd.DataFrame({
            "SMILES": te_smiles_raw,
            "Molecule Name": te_names,
            "pEC50": te_pred,
        }).to_csv(sub_csv, index=False)
        print(f"   [save] {sub_csv}")
    else:
        print(f"   [skip] verdict={verdict}; no submission CSV written")

    summary = {
        "tag": TAG,
        "anchor_tag": ANCHOR_TAG,
        "method": (
            "split_conformal_interval_midpoint_calibration_"
            "per_train_similarity_bin_local_mean_shift_on_nb3090"
        ),
        "anchor_pre_unblind": True,
        "anchor_oof_rae": round(anchor_oof_rae, 4),
        "anchor_leak_eq_truth_frac": round(leak_eq, 4),
        "binning_axis": "top1_tanimoto_to_4139_train",
        "calib_frac": CALIB_FRAC,
        "n_bins_grid": N_BINS_GRID,
        "lambda_grid": LAMBDA_GRID,
        "conformal_alpha": CONFORMAL_ALPHA,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": n_s,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "sim_unb_mean": float(sim_unb.mean()),
        "sim_unb_median": float(np.median(sim_unb)),
        "sim_te_mean": float(sim_te.mean()),
        "bias_low_sim_third": round(bias_lo, 4),
        "bias_rest": round(bias_hi, 4),
        "seed_records": seed_records,
        "pooled_rae_array": [round(float(v), 4) for v in pooled_raes],
        "per_fold_mean_array": [round(float(v), 4) for v in per_fold_means],
        # primary gate metric: per-fold-mean across 15 seeds
        "mean_rae": round(pf_mean, 4),
        "std_rae": round(pf_std, 4),
        "sem_rae": round(sem, 4),
        "ci95_low": round(ci_low, 4),
        "ci95_high": round(ci_high, 4),
        "median_rae": round(median_pf, 4),
        "min_rae": round(float(pf_arr.min()), 4),
        "max_rae": round(float(pf_arr.max()), 4),
        # pooled also recorded
        "pooled_mean": round(pooled_mean, 4),
        "pooled_std": round(pooled_std, 4),
        # config picks
        "n_bins_distribution": {str(k): int(v) for k, v in nb_counter.items()},
        "lambda_distribution": {str(k): int(v) for k, v in lam_counter.items()},
        "nb_mode": int(nb_mode),
        "lam_mode": float(lam_mode),
        # references
        "ref_nb3090_anchor": REF_NB3090_ANCHOR,
        "delta_vs_nb3090_anchor": round(pf_mean - anchor_oof_rae, 4),
        "ref_nb3170_fixed": REF_NB3170_FIXED,
        "delta_vs_nb3170_fixed": round(pf_mean - REF_NB3170_FIXED, 4),
        "ref_nb3001": REF_NB3001,
        "ref_nb2992": REF_NB2992,
        "ref_nb2171": REF_NB2171,
        # deploy
        "deploy_n_bins": int(dep_nb),
        "deploy_lambda": float(dep_lam),
        "deploy_bin_deltas": [round(float(d), 4) for d in dep_deltas],
        "deploy_global_delta": round(float(dep_global), 4),
        "deploy_bin_edges_interior": [
            round(float(e), 4) for e in dep_edges[1:-1]
        ],
        "n_te_shifted": n_te_shifted,
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "te_min": float(te_pred.min()),
        "te_max": float(te_pred.max()),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "median_seed": int(median_seed),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": (str(sub_csv) if verdict == "BETTER" else None),
        "gate_better": GATE_BETTER,
        "verdict": verdict,
        "ladder_action": ladder_action,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   pf_mean ({n_s} seeds)         = {pf_mean:.4f} +/- {pf_std:.4f}")
    print(f"   95% CI                       = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   delta vs nb3090 anchor (pf)  = {pf_mean - anchor_oof_rae:+.4f}")
    print(f"   delta vs nb3170 fixed (pf)   = {pf_mean - REF_NB3170_FIXED:+.4f}")
    print(f"   verdict                      = {verdict}")
    print(f"   wall                         = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae", "std_rae", "ci95_low", "ci95_high",
        "pooled_mean", "pooled_std",
        "delta_vs_nb3090_anchor", "delta_vs_nb3170_fixed",
        "nb_mode", "lam_mode",
        "deploy_n_bins", "deploy_lambda", "deploy_bin_deltas",
        "n_te_shifted",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
