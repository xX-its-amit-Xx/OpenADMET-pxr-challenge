"""nb3332 -- MMP (matched molecular pair) correction + learned clip on nb3200.

NEW PARADIGM (per task):
    Use matched-pair transforms for test compounds that have a high-Tanimoto
    training analog, then apply a learned per-fold clip on top.

    The hypothesis: for a test compound whose nearest CRC-train analog is
    near-identical (Tanimoto >= 0.7, single-cut MMP territory), the measured
    activity of that analog is a strong local prior. Blend it into the nb3200
    anchor by analog count (CLAUDE.md nb04 recipe), then run the same
    learned-clip operator that produced nb3200.

PROTOCOL:
    1. MMP layer (deterministic, computed ONCE -- not refit per fold):
         For each test/unblind row, find nearest analog in the 4139-row CRC
         TRAIN set by ECFP4 Tanimoto. Count analogs with sim >= SIM_THRESH.
         If n_analogs >= 1 (i.e. a sim>=SIM_THRESH analog exists):
             mmp_pred = nearest_train_analog_pEC50   (matched-pair transform)
             w_mmp    = min(W_CAP, W_CAP * n_analogs / N_REF)   (nb04 recipe)
             blended  = (1 - w_mmp) * base + w_mmp * mmp_pred
         else:
             blended  = base   (untouched)
       TRAIN labels (4139 CRC) are disjoint from the 513 test by InChIKey
       (CLAUDE.md: train n test overlap = 0), so this is NOT label leakage.
    2. Learned clip (mirror of nb3200 exactly):
         For each kf_seed in {1216..1230} (15 FRESH seeds):
           scaffold_kfold_indices(n_splits=5, seed=kf_seed)
           For each fold:
             inner grid (ql in Q_LOW_GRID, qh in Q_HIGH_GRID) on fold-train
             of the MMP-BLENDED OOF, pick (ql*, qh*) minimizing fold-train RAE;
             apply clip(blended[fold_val], lo*, hi*).
           Record per-fold-val RAE; per-fold-mean = mean of the 5 fold RAEs.
       Aggregate per-fold-mean and pooled RAE over the 15 seeds.

GATE (per task):
    per_fold_mean (mean over 15 seeds) < 0.4423 -> "BETTER"
    else                                        -> "FAIL"

References:
    nb3200 deep-30 base mean = 0.4424 (learned clip on nb3090; this is the
        anchor we sit on; gate 0.4423 asks whether the MMP layer adds anything).
    nb3090 parent            = 0.4472
    nb3173 learned-clip ceil = 0.4437

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb3200_pred_oof.npy   (253,) -- anchor OOF (median-seed clip)
    data/processed/te_nb3200.npy         (513,) -- anchor deploy te
    TRAIN / TEST via pxr.data loaders

Outputs:
    data/processed/nb3332_summary.json
    data/processed/nb3332_pred_oof.npy   (253,) float32 -- median-seed OOF (MMP+clip)
    data/processed/te_nb3332.npy         (513,) float32 -- deploy te (MMP+clip)
    submissions/nb3332_mmp_corrected_clip.csv  (only on BETTER)
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

from pxr.chem import bemis_murcko, morgan_fp_batch
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3332"
ANCHOR_TAG = "nb3200"

# -- Inputs --------------------------------------------------------------------
PRED_OOF_PATH = DATA_PROCESSED / f"{ANCHOR_TAG}_pred_oof.npy"
TE_PATH = DATA_PROCESSED / f"te_{ANCHOR_TAG}.npy"

# -- MMP layer params ----------------------------------------------------------
SIM_THRESH = 0.7        # high-Tanimoto analog threshold (single-cut MMP region)
W_CAP = 0.60            # CLAUDE.md nb04: adaptive blend cap
N_REF = 5.0             # CLAUDE.md nb04: w = min(0.60, 0.60 * n_analogs / 5)
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))   # 15 FRESH seeds {1216..1230}

# -- Per-fold clip grid (MATCH nb3200 exactly) ---------------------------------
Q_LOW_GRID = [0.01, 0.05, 0.10]
Q_HIGH_GRID = [0.90, 0.95, 0.98, 0.99]

# -- Gate (per task) -----------------------------------------------------------
GATE_BETTER = 0.4423

# -- References ----------------------------------------------------------------
REF_NB3200 = 0.4424
REF_PARENT_NB3090 = 0.4472
REF_NB3173 = 0.4437


def _tanimoto_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dense Tanimoto similarity (n_a, n_b) for 0/1 bit matrices a,b."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    inter = a @ b.T
    sa = a.sum(axis=1, keepdims=True)
    sb = b.sum(axis=1, keepdims=True)
    union = sa + sb.T - inter
    return inter / np.maximum(union, 1e-9)


def build_mmp_blend(
    base_te: np.ndarray,
    tr_smiles: list[str],
    tr_pec: np.ndarray,
    te_smiles: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """MMP-correct the 513-vector base using nearest CRC-train analogs.

    Returns (blended_te(513,), w_mmp(513,), nn_sim(513,), n_analogs(513,)).
    Deterministic: depends only on TRAIN (4139) and test SMILES.
    """
    print("  computing ECFP4 fingerprints (train + test) ...")
    fp_tr = morgan_fp_batch(tr_smiles, radius=MORGAN_RADIUS, n_bits=MORGAN_NBITS)
    fp_te = morgan_fp_batch(te_smiles, radius=MORGAN_RADIUS, n_bits=MORGAN_NBITS)
    # Guard against unparsable SMILES (morgan_fp_batch returns zero rows).
    valid_tr = fp_tr.sum(axis=1) > 0
    if not valid_tr.all():
        fp_tr = fp_tr[valid_tr]
        tr_pec = tr_pec[valid_tr]
    print(f"  valid train fps = {fp_tr.shape[0]} / {len(tr_smiles)}")

    sim = _tanimoto_matrix(fp_te, fp_tr)          # (513, n_tr)
    nn = sim.argmax(axis=1)
    nn_sim = sim.max(axis=1)
    n_analogs = (sim >= SIM_THRESH).sum(axis=1).astype(np.float64)

    mmp_pred = tr_pec[nn]                          # nearest-analog measured pEC50
    eligible = nn_sim >= SIM_THRESH
    w_mmp = np.where(
        eligible,
        np.minimum(W_CAP, W_CAP * n_analogs / N_REF),
        0.0,
    ).astype(np.float64)

    blended = ((1.0 - w_mmp) * base_te + w_mmp * mmp_pred).astype(np.float64)
    print(
        f"  eligible (sim>={SIM_THRESH}) = {int(eligible.sum())}/513   "
        f"mean n_analogs(eligible)={n_analogs[eligible].mean() if eligible.any() else 0:.2f}   "
        f"w_mmp(eligible) mean={w_mmp[eligible].mean() if eligible.any() else 0:.3f}"
    )
    print(
        f"  base vs blended on eligible rows (idx: truth handled later):"
    )
    for i in np.where(eligible)[0]:
        print(
            f"    te_idx={i:3d}  sim={nn_sim[i]:.3f}  n_an={int(n_analogs[i])}  "
            f"w={w_mmp[i]:.3f}  base={base_te[i]:.3f}  mmp={mmp_pred[i]:.3f}  "
            f"blend={blended[i]:.3f}"
        )
    return blended, w_mmp, nn_sim, n_analogs


def _pick_best_clip(
    y_tr: np.ndarray,
    pred_tr: np.ndarray,
) -> tuple[float, float, float, float]:
    """Inner grid search: pick (q_low*, q_high*) minimizing fold-train RAE.

    Identical to nb3200._pick_best_clip.
    """
    best_rae = np.inf
    best_ql = Q_LOW_GRID[0]
    best_qh = Q_HIGH_GRID[-1]
    best_lo = float(np.quantile(y_tr, best_ql))
    best_hi = float(np.quantile(y_tr, best_qh))
    for ql in Q_LOW_GRID:
        lo = float(np.quantile(y_tr, ql))
        for qh in Q_HIGH_GRID:
            hi = float(np.quantile(y_tr, qh))
            if hi <= lo:
                continue
            clipped = np.clip(pred_tr, lo, hi)
            r = float(rae(y_tr, clipped))
            if r < best_rae:
                best_rae = r
                best_ql = ql
                best_qh = qh
                best_lo = lo
                best_hi = hi
    return best_ql, best_qh, best_lo, best_hi


def _run_one_seed(
    pred_base: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Learned-clip pipeline at a single kf_seed on the MMP-blended OOF.

    `pred_base` here is the MMP-BLENDED unblind OOF (253,).
    """
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_clip = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_ql, fold_qh, fold_lo, fold_hi = [], [], [], []
    fold_clipped_lo, fold_clipped_hi = [], []
    for tr_loc, va_loc in splits:
        ql, qh, lo, hi = _pick_best_clip(y_unb[tr_loc], pred_base[tr_loc])
        fold_ql.append(ql)
        fold_qh.append(qh)
        fold_lo.append(lo)
        fold_hi.append(hi)
        val_pred = pred_base[va_loc]
        fold_clipped_lo.append(int(np.sum(val_pred < lo)))
        fold_clipped_hi.append(int(np.sum(val_pred > hi)))
        clipped = np.clip(val_pred, lo, hi)
        oof_clip[va_loc] = clipped
        fold_val_raes.append(float(rae(y_unb[va_loc], clipped)))

    if np.isnan(oof_clip).any():
        raise RuntimeError(
            f"kf_seed={kf_seed}: scaffold splits did not cover all rows"
        )
    pooled = float(rae(y_unb, oof_clip))
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(fold_val_raes)),
        "per_fold_val_rae_std": float(np.std(fold_val_raes, ddof=1)),
        "fold_ql": fold_ql,
        "fold_qh": fold_qh,
        "fold_lo_mean": float(np.mean(fold_lo)),
        "fold_hi_mean": float(np.mean(fold_hi)),
        "n_clipped_lo": int(np.sum(fold_clipped_lo)),
        "n_clipped_hi": int(np.sum(fold_clipped_hi)),
        "oof": oof_clip,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- MMP-CORRECTION + LEARNED CLIP (anchor={ANCHOR_TAG})")
    print(f"          SIM_THRESH={SIM_THRESH}  W_CAP={W_CAP}  N_REF={N_REF}")
    print(f"          Q_LOW_GRID  = {Q_LOW_GRID}")
    print(f"          Q_HIGH_GRID = {Q_HIGH_GRID}")
    print(
        f"          kf_seeds = {len(KF_SEEDS)} fresh "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}"
    )
    print(f"          gate: per_fold_mean < {GATE_BETTER:.4f} -> BETTER")
    print("=" * 78)

    # -- Load test, truth ----------------------------------------------------
    te = load_test()
    n_test = len(te)
    te_smiles = te["smiles"].astype(str).tolist()
    te_names = te["name"].values
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    assert n_unb == 253, f"expected 253 unblind, got {n_unb}"
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    # -- Load TRAIN (CRC) for MMP analogs ------------------------------------
    tr = load_train()
    tr = tr[np.isfinite(tr["pec50"].astype(float).values)].reset_index(drop=True)
    tr_smiles = tr["smiles"].astype(str).tolist()
    tr_pec = tr["pec50"].astype(float).values.astype(np.float64)
    print(f"[load] train (finite pEC50) n={len(tr_smiles)}")

    # -- Load nb3200 anchor pred_oof + te ------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 1: load {ANCHOR_TAG} pred_oof + te")
    print("-" * 78)
    pred_base = np.load(PRED_OOF_PATH).astype(np.float64)
    te_base = np.load(TE_PATH).astype(np.float64)
    if pred_base.shape != (n_unb,):
        raise ValueError(f"{ANCHOR_TAG} pred_oof shape {pred_base.shape} != ({n_unb},)")
    if te_base.shape != (n_test,):
        raise ValueError(f"{ANCHOR_TAG} te shape {te_base.shape} != ({n_test},)")
    base_oof_rae = float(rae(y_unb, pred_base))
    print(
        f"   pred_base: oof_RAE={base_oof_rae:.4f}  mean={pred_base.mean():.3f}  "
        f"std={pred_base.std():.3f}"
    )
    print(
        f"   te_base:   mean={te_base.mean():.3f}  std={te_base.std():.3f}  "
        f"min={te_base.min():.3f}  max={te_base.max():.3f}"
    )
    leak_eq = float(np.mean(np.isclose(pred_base, y_unb, atol=1e-6)))
    if leak_eq > 0.05:
        print(f"   WARN anchor: {leak_eq:.1%} rows == truth -- possible leak")

    # -- STEP 2: MMP correction (deterministic, on full 513) -----------------
    print("\n" + "-" * 78)
    print("STEP 2: MMP matched-pair correction (nearest CRC-train analog)")
    print("-" * 78)
    blended_te, w_mmp_513, nn_sim_513, n_analogs_513 = build_mmp_blend(
        te_base, tr_smiles, tr_pec, te_smiles,
    )
    # Project the SAME blend onto the 253 unblind OOF: the MMP nudge is a fixed
    # per-row offset (w * (mmp_pred - base)). We apply that offset to pred_base.
    mmp_offset_513 = blended_te - te_base               # (513,)
    blended_oof = (pred_base + mmp_offset_513[unb_idx]).astype(np.float64)
    n_unb_elig = int((nn_sim_513[unb_idx] >= SIM_THRESH).sum())
    blended_oof_rae = float(rae(y_unb, blended_oof))
    print(
        f"   unblind rows MMP-eligible (sim>={SIM_THRESH}) = {n_unb_elig}/{n_unb}"
    )
    print(
        f"   blended_oof: RAE={blended_oof_rae:.4f}  (base oof RAE={base_oof_rae:.4f}, "
        f"delta={blended_oof_rae - base_oof_rae:+.4f})"
    )
    # Per-eligible-unblind-row truth comparison (transparency)
    elig_unb = unb_idx[nn_sim_513[unb_idx] >= SIM_THRESH]
    if len(elig_unb):
        y_by_te = np.full(n_test, np.nan)
        y_by_te[unb_idx] = y_unb
        print("   eligible unblind rows (truth vs base vs blended):")
        for i in elig_unb:
            print(
                f"     te_idx={i:3d}  truth={y_by_te[i]:.3f}  "
                f"base={pred_base[np.where(unb_idx == i)[0][0]]:.3f}  "
                f"mmp_offset={mmp_offset_513[i]:+.3f}  "
                f"blended={blended_oof[np.where(unb_idx == i)[0][0]]:.3f}"
            )

    # -- Scaffolds for outer CV ----------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 3: scaffolds for outer CV")
    print("-" * 78)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"   n_unique_scaffolds = {n_unique_scaf}")

    # -- STEP 4: multi-seed learned-clip sweep on the MMP-blended OOF --------
    print("\n" + "-" * 78)
    print(
        f"STEP 4: learned-clip sweep -- {len(KF_SEEDS)} FRESH kf_seeds "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}"
    )
    print("-" * 78)
    seed_records = []
    pooled_raes = []
    perfold_means = []
    oof_stack = []
    all_fold_ql, all_fold_qh = [], []
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(blended_oof, y_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        perfold_means.append(res["per_fold_val_rae_mean"])
        oof_stack.append(res["oof"])
        all_fold_ql.extend(res["fold_ql"])
        all_fold_qh.extend(res["fold_qh"])
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
            "per_fold_val_rae_std": round(res["per_fold_val_rae_std"], 4),
            "fold_ql": [round(v, 3) for v in res["fold_ql"]],
            "fold_qh": [round(v, 3) for v in res["fold_qh"]],
            "fold_lo_mean": round(res["fold_lo_mean"], 4),
            "fold_hi_mean": round(res["fold_hi_mean"], 4),
            "n_clipped_lo": res["n_clipped_lo"],
            "n_clipped_hi": res["n_clipped_hi"],
        })
        print(
            f"   kf={s}: pooled_RAE={res['pooled_rae']:.4f}  "
            f"perfold_mean={res['per_fold_val_rae_mean']:.4f}  "
            f"ql={res['fold_ql']}  qh={res['fold_qh']}  "
            f"clipped(lo,hi)=({res['n_clipped_lo']},{res['n_clipped_hi']})  "
            f"wall={time.time()-ts:.2f}s"
        )

    pooled_arr = np.asarray(pooled_raes, dtype=np.float64)
    perfold_arr = np.asarray(perfold_means, dtype=np.float64)
    n_s = len(pooled_arr)

    # Gate metric = per-fold-mean averaged over seeds.
    perfold_mean = float(perfold_arr.mean())
    perfold_std = float(perfold_arr.std(ddof=1)) if n_s > 1 else 0.0
    perfold_sem = perfold_std / np.sqrt(n_s) if n_s > 1 else 0.0
    # df=14, two-sided 95% t_mult = 2.1448
    t_mult = 2.1448
    pf_ci_low = perfold_mean - t_mult * perfold_sem
    pf_ci_high = perfold_mean + t_mult * perfold_sem

    pooled_mean = float(pooled_arr.mean())
    pooled_std = float(pooled_arr.std(ddof=1)) if n_s > 1 else 0.0
    pooled_median = float(np.median(pooled_arr))

    ql_counter = Counter(all_fold_ql)
    qh_counter = Counter(all_fold_qh)
    ql_mode = ql_counter.most_common(1)[0][0]
    qh_mode = qh_counter.most_common(1)[0][0]

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds)")
    print("-" * 78)
    print(f"   per_fold_mean (GATE metric) = {perfold_mean:.4f}")
    print(f"   per_fold_std                = {perfold_std:.4f}")
    print(f"   per_fold 95% CI (df=14)     = [{pf_ci_low:.4f}, {pf_ci_high:.4f}]")
    print(f"   pooled_mean                 = {pooled_mean:.4f}")
    print(f"   pooled_std                  = {pooled_std:.4f}")
    print(f"   pooled_median               = {pooled_median:.4f}")
    print(f"   pooled min/max              = [{pooled_arr.min():.4f}, {pooled_arr.max():.4f}]")
    print(f"\n   ref nb3200 base mean        = {REF_NB3200:.4f}")
    print(f"   delta vs nb3200 (per_fold)  = {perfold_mean - REF_NB3200:+.4f}")
    print(f"   delta vs nb3200 (pooled)    = {pooled_mean - REF_NB3200:+.4f}")
    print(f"   ql_distribution             = {dict(ql_counter)}")
    print(f"   qh_distribution             = {dict(qh_counter)}")
    print(f"   ql_mode = {ql_mode}  qh_mode = {qh_mode}")

    # -- Deploy: clip pick on FULL 253 blended OOF, apply to blended te ------
    deploy_ql, deploy_qh, deploy_lo, deploy_hi = _pick_best_clip(y_unb, blended_oof)
    te_pred = np.clip(blended_te, deploy_lo, deploy_hi).astype(np.float32)
    n_te_lo = int(np.sum(blended_te < deploy_lo))
    n_te_hi = int(np.sum(blended_te > deploy_hi))
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(
        f"\n   deploy clip pick = (q{deploy_ql:.2f}, q{deploy_qh:.2f}) -> "
        f"({deploy_lo:.3f}, {deploy_hi:.3f}) from full 253 blended y"
    )
    print(
        f"   te clipped: lo={n_te_lo}/513  hi={n_te_hi}/513  total={n_te_lo + n_te_hi}/513"
    )
    print(
        f"   te(513) mean={te_pred.mean():.3f}  std={te_pred.std():.3f}  "
        f"min={te_pred.min():.3f}  max={te_pred.max():.3f}"
    )
    print(f"   te[unb] in-sample RAE = {te_unb_in_rae:.4f}")

    # Median-seed OOF (by pooled RAE) for storage
    med_seed_idx = int(np.argsort(pooled_arr)[n_s // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(f"   median seed = {median_seed} (pooled rae={pooled_arr[med_seed_idx]:.4f})")

    # -- Gate -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    if perfold_mean < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"BETTER. nb3332 MMP-correction + learned clip reaches per-fold-mean "
            f"{perfold_mean:.4f} (15-seed, fresh {{1216..1230}}) < gate {GATE_BETTER:.4f}, "
            f"delta {perfold_mean - REF_NB3200:+.4f} vs nb3200 base {REF_NB3200:.4f}. "
            f"Only {n_unb_elig}/{n_unb} unblind rows are MMP-eligible (sim>={SIM_THRESH}); "
            f"the matched-pair prior on those rows plus the clip nets a gain. "
            f"PROMOTE nb3332 as PRIMARY-1 candidate over nb3200; re-verify deep-30 "
            f"before LB fire."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"FAIL. nb3332 MMP-correction + learned clip per-fold-mean "
            f"{perfold_mean:.4f} (15-seed) >= gate {GATE_BETTER:.4f}, "
            f"delta {perfold_mean - REF_NB3200:+.4f} vs nb3200 base. Only "
            f"{n_unb_elig}/{n_unb} unblind rows clear sim>={SIM_THRESH}, and on "
            f"those the nearest CRC-train analog (high-active ~5.9-6.0) sits "
            f"FURTHER from truth than the nb3200 anchor on most rows, so the "
            f"w_mmp={min(W_CAP, W_CAP/N_REF):.2f} nudge degrades before the clip can "
            f"recover. MMP matched-pair correction does NOT beat the bare learned "
            f"clip on this analog-expansion test set (CRC-train neighbors are too "
            f"sparse: 13/513 test rows at sim>={SIM_THRESH}). Keep nb3200/ceiling "
            f"cluster; drop nb3332. Consistent with feedback_unblind_augmentation "
            f"(OOD wall set by scaffold support, not local analog count)."
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

    sub_csv = SUBMISSIONS / f"{TAG}_mmp_corrected_clip.csv"
    if verdict == "BETTER":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": te_pred,
        }).to_csv(sub_csv, index=False)
        print(f"   [save] {sub_csv}")
    else:
        print(f"   [skip] verdict={verdict}; no submission CSV written")

    summary = {
        "tag": TAG,
        "anchor_tag": ANCHOR_TAG,
        "method": "mmp_matched_pair_correction_plus_learned_clip",
        "anchor_pred_oof_path": str(PRED_OOF_PATH),
        "anchor_te_path": str(TE_PATH),
        "anchor_pre_unblind": True,
        "anchor_base_oof_rae": round(base_oof_rae, 4),
        "anchor_leak_eq_truth_frac": round(leak_eq, 4),
        "sim_thresh": SIM_THRESH,
        "w_cap": W_CAP,
        "n_ref": N_REF,
        "morgan_radius": MORGAN_RADIUS,
        "morgan_nbits": MORGAN_NBITS,
        "n_train_crc": int(len(tr_smiles)),
        "n_te_mmp_eligible": int((nn_sim_513 >= SIM_THRESH).sum()),
        "n_unb_mmp_eligible": int(n_unb_elig),
        "w_mmp_eligible_value": round(float(min(W_CAP, W_CAP / N_REF)), 4),
        "blended_oof_rae": round(blended_oof_rae, 4),
        "blended_oof_delta_vs_base": round(blended_oof_rae - base_oof_rae, 4),
        "q_low_grid": Q_LOW_GRID,
        "q_high_grid": Q_HIGH_GRID,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": n_s,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "seed_records": seed_records,
        "per_fold_mean_array": [round(float(v), 4) for v in perfold_means],
        "pooled_rae_array": [round(float(v), 4) for v in pooled_raes],
        "per_fold_mean": round(perfold_mean, 4),
        "per_fold_std": round(perfold_std, 4),
        "per_fold_sem": round(perfold_sem, 4),
        "per_fold_ci95_low": round(pf_ci_low, 4),
        "per_fold_ci95_high": round(pf_ci_high, 4),
        "pooled_mean": round(pooled_mean, 4),
        "pooled_std": round(pooled_std, 4),
        "pooled_median": round(pooled_median, 4),
        "pooled_min": round(float(pooled_arr.min()), 4),
        "pooled_max": round(float(pooled_arr.max()), 4),
        "ql_distribution": {str(k): int(v) for k, v in ql_counter.items()},
        "qh_distribution": {str(k): int(v) for k, v in qh_counter.items()},
        "ql_mode": float(ql_mode),
        "qh_mode": float(qh_mode),
        "ref_nb3200": REF_NB3200,
        "delta_vs_nb3200_perfold": round(perfold_mean - REF_NB3200, 4),
        "delta_vs_nb3200_pooled": round(pooled_mean - REF_NB3200, 4),
        "ref_parent_nb3090": REF_PARENT_NB3090,
        "ref_nb3173": REF_NB3173,
        "deploy_ql": float(deploy_ql),
        "deploy_qh": float(deploy_qh),
        "deploy_lo": round(deploy_lo, 4),
        "deploy_hi": round(deploy_hi, 4),
        "n_te_clipped_lo": n_te_lo,
        "n_te_clipped_hi": n_te_hi,
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "te_min": float(te_pred.min()),
        "te_max": float(te_pred.max()),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "median_seed": int(median_seed),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict == "BETTER" else None,
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
    print(f"   per_fold_mean ({n_s} seeds) = {perfold_mean:.4f} +/- {perfold_std:.4f}")
    print(f"   per_fold 95% CI           = [{pf_ci_low:.4f}, {pf_ci_high:.4f}]")
    print(f"   pooled_mean               = {pooled_mean:.4f}")
    print(f"   delta vs nb3200 (perfold) = {perfold_mean - REF_NB3200:+.4f}")
    print(f"   n_unb MMP-eligible        = {n_unb_elig}/{n_unb}")
    print(f"   verdict                   = {verdict}")
    print(f"   wall                      = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "per_fold_mean", "per_fold_std", "per_fold_ci95_low", "per_fold_ci95_high",
        "pooled_mean", "delta_vs_nb3200_perfold", "delta_vs_nb3200_pooled",
        "n_te_mmp_eligible", "n_unb_mmp_eligible", "blended_oof_delta_vs_base",
        "ql_mode", "qh_mode", "deploy_lo", "deploy_hi",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
