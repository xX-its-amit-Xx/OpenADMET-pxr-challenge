"""nb3252 -- Clip with chemprop_aux fallback for clipped rows on nb3090 anchor.

NEW PARADIGM:
    nb3200-style learned-clip on nb3090 forces extreme predictions to a hard
    quantile bound (lo, hi), which acts as variance compression but DISCARDS
    the model's signal on those exact rows. If chemprop_aux (a verified PRE-
    clean, biologically-orthogonal anchor) has a meaningful prediction on
    those same rows, it may carry more information than a flat clip value.

    nb3252 replaces clipped values with chemprop_aux_oof prediction on the
    253 unblind (and chemprop_aux_te on the 513 deploy).

    Hypothesis: chemprop_aux is calibrated differently (mean 4.684 vs y_unb
    mean ~4.7, std 0.811 vs nb3090's compressed tails). For rows that nb3090
    over-extends in either direction, chemprop_aux's softer extremes provide
    a per-row signal-preserving alternative to flat-clip variance compression.

PROTOCOL (mirror nb3200, swap clip-action for fallback):
    For each kf_seed in {1216..1230} (15 FRESH seeds, disjoint from nb3190
    {1171..1185} and nb3200 {1186..1215}):
      scaffold_kfold_indices(n_splits=5, seed=kf_seed)
      For each fold:
        a) Inner grid search on fold-train ONLY (same grid as nb3200):
            ql in {0.01, 0.05, 0.10}, qh in {0.90, 0.95, 0.98, 0.99}
            For each (ql, qh):
              lo = quantile(y[fold_train], ql)
              hi = quantile(y[fold_train], qh)
              # FALLBACK action: where pred_base would be clipped,
              # substitute chemprop_aux_oof[fold_train] instead.
              mask_lo = pred_base[fold_train] < lo
              mask_hi = pred_base[fold_train] > hi
              out_tr = pred_base[fold_train].copy()
              out_tr[mask_lo] = ca_oof[fold_train][mask_lo]
              out_tr[mask_hi] = ca_oof[fold_train][mask_hi]
              rae_tr = rae(y[fold_train], out_tr)
            Pick (ql*, qh*) minimizing fold-train RAE.
        b) Apply fallback on fold-val using lo*, hi*:
            mask_lo = pred_base[fold_val] < lo*
            mask_hi = pred_base[fold_val] > hi*
            val_pred = pred_base[fold_val].copy()
            val_pred[mask_lo] = ca_oof[fold_val][mask_lo]
            val_pred[mask_hi] = ca_oof[fold_val][mask_hi]
        c) Stitch into oof_fb; pooled RAE across 5 folds.
      pooled_rae per seed.
    Aggregate mean +/- std over 15 seeds.

GATE (per task):
    per-fold-mean < 0.4423 -> "BETTER"  (beats nb3200 deep-30 0.4423)
    else                    -> "FAIL"

References:
    nb3090 best combo 15-seed mean = 0.4472    (parent anchor)
    nb3190 15-seed clip            = 0.4426
    nb3200 deep-30 clip            = 0.4423   <- gate
    nb3173 learned-clip on nb3080  = 0.4437
    chemprop_aux RAE on 253        = 0.5879  (orthogonal axis, PRE-clean)
    nb2171 prior post-hoc top      = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb3090_pred_oof.npy
    data/processed/te_nb3090.npy
    data/processed/nb1133_chemprop_aux_pred_oof.npy   (253 unblind aligned)
    data/processed/te_chemprop_aux.npy                (513 deploy aligned)

Outputs:
    data/processed/nb3252_summary.json
    data/processed/nb3252_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3252.npy         (513,) float32 -- deploy te
    submissions/nb3252_clip_chemprop_fallback.csv  (only on BETTER)
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

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3252"
PARENT_TAG = "nb3090"
FALLBACK_TAG = "chemprop_aux"

# -- Inputs --------------------------------------------------------------------
PRED_OOF_PATH = DATA_PROCESSED / "nb3090_pred_oof.npy"
TE_PATH = DATA_PROCESSED / "te_nb3090.npy"
CA_OOF_PATH = DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy"
CA_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))  # 15 FRESH seeds {1216..1230}

# -- Per-fold grid (mirror nb3200 exactly) -------------------------------------
Q_LOW_GRID = [0.01, 0.05, 0.10]
Q_HIGH_GRID = [0.90, 0.95, 0.98, 0.99]

# -- Gates (per task) ----------------------------------------------------------
GATE_BETTER = 0.4423   # per-fold-mean < this -> BETTER (beats nb3200 deep-30)

# -- References ----------------------------------------------------------------
REF_PARENT_NB3090 = 0.4472
REF_NB3190 = 0.4426
REF_NB3200 = 0.4423
REF_NB3173 = 0.4437
REF_CHEMPROP_AUX = 0.5879
REF_NB2171 = 0.4682


def _apply_fallback(
    pred: np.ndarray,
    ca: np.ndarray,
    lo: float,
    hi: float,
) -> np.ndarray:
    """Replace clipped rows (pred<lo or pred>hi) with chemprop_aux prediction."""
    out = pred.copy()
    mask_lo = pred < lo
    mask_hi = pred > hi
    out[mask_lo] = ca[mask_lo]
    out[mask_hi] = ca[mask_hi]
    return out


def _pick_best_clip_fallback(
    y_tr: np.ndarray,
    pred_tr: np.ndarray,
    ca_tr: np.ndarray,
) -> tuple[float, float, float, float]:
    """Inner grid search: pick (q_low*, q_high*) minimizing fold-train RAE
    using chemprop_aux-fallback action on clipped rows."""
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
            out = _apply_fallback(pred_tr, ca_tr, lo, hi)
            r = float(rae(y_tr, out))
            if r < best_rae:
                best_rae = r
                best_ql = ql
                best_qh = qh
                best_lo = lo
                best_hi = hi
    return best_ql, best_qh, best_lo, best_hi


def _run_one_seed(
    pred_base: np.ndarray,
    ca_oof: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Run chemprop_aux-fallback pipeline at a single kf_seed."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_fb = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_ql = []
    fold_qh = []
    fold_lo = []
    fold_hi = []
    fold_n_fallback_lo = []
    fold_n_fallback_hi = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        ql, qh, lo, hi = _pick_best_clip_fallback(
            y_unb[tr_loc], pred_base[tr_loc], ca_oof[tr_loc],
        )
        fold_ql.append(ql)
        fold_qh.append(qh)
        fold_lo.append(lo)
        fold_hi.append(hi)
        val_pred = pred_base[va_loc]
        n_lo = int(np.sum(val_pred < lo))
        n_hi = int(np.sum(val_pred > hi))
        fold_n_fallback_lo.append(n_lo)
        fold_n_fallback_hi.append(n_hi)
        out = _apply_fallback(val_pred, ca_oof[va_loc], lo, hi)
        oof_fb[va_loc] = out
        fold_val_raes.append(float(rae(y_unb[va_loc], out)))

    if np.isnan(oof_fb).any():
        raise RuntimeError(
            f"kf_seed={kf_seed}: scaffold splits did not cover all rows"
        )
    pooled = float(rae(y_unb, oof_fb))
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(fold_val_raes)),
        "per_fold_val_rae_std": float(np.std(fold_val_raes, ddof=1)),
        "fold_ql": fold_ql,
        "fold_qh": fold_qh,
        "fold_lo_mean": float(np.mean(fold_lo)),
        "fold_hi_mean": float(np.mean(fold_hi)),
        "n_fallback_lo": int(np.sum(fold_n_fallback_lo)),
        "n_fallback_hi": int(np.sum(fold_n_fallback_hi)),
        "oof": oof_fb,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(
        f"{TAG} -- CHEMPROP_AUX FALLBACK on clipped rows ({PARENT_TAG} anchor)"
    )
    print(f"          Q_LOW_GRID  = {Q_LOW_GRID}")
    print(f"          Q_HIGH_GRID = {Q_HIGH_GRID}")
    print(
        f"          kf_seeds = {len(KF_SEEDS)} fresh "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}"
    )
    print(
        f"          gate: per-fold-mean < {GATE_BETTER:.4f} -> BETTER, "
        f"else FAIL"
    )
    print("=" * 78)

    # -- Load test, truth ----------------------------------------------------
    te = load_test()
    n_test = len(te)
    te_smiles = (
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

    # -- Load anchors --------------------------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 1: load {PARENT_TAG} + {FALLBACK_TAG} pred_oof + te")
    print("-" * 78)
    pred_base = np.load(PRED_OOF_PATH).astype(np.float64)
    te_base = np.load(TE_PATH).astype(np.float64)
    ca_oof = np.load(CA_OOF_PATH).astype(np.float64)
    ca_te = np.load(CA_TE_PATH).astype(np.float64)
    if pred_base.shape != (n_unb,):
        raise ValueError(
            f"{PARENT_TAG} pred_oof shape {pred_base.shape} != ({n_unb},)"
        )
    if te_base.shape != (n_test,):
        raise ValueError(
            f"{PARENT_TAG} te shape {te_base.shape} != ({n_test},)"
        )
    if ca_oof.shape != (n_unb,):
        raise ValueError(
            f"{FALLBACK_TAG} pred_oof shape {ca_oof.shape} != ({n_unb},)"
        )
    if ca_te.shape != (n_test,):
        raise ValueError(
            f"{FALLBACK_TAG} te shape {ca_te.shape} != ({n_test},)"
        )
    full_oof_rae = float(rae(y_unb, pred_base))
    ca_oof_rae = float(rae(y_unb, ca_oof))
    print(
        f"   pred_base ({PARENT_TAG}): oof_RAE={full_oof_rae:.4f}  "
        f"mean={pred_base.mean():.3f}  std={pred_base.std():.3f}  "
        f"min={pred_base.min():.3f}  max={pred_base.max():.3f}"
    )
    print(
        f"   ca_oof   ({FALLBACK_TAG}): oof_RAE={ca_oof_rae:.4f}  "
        f"mean={ca_oof.mean():.3f}  std={ca_oof.std():.3f}  "
        f"min={ca_oof.min():.3f}  max={ca_oof.max():.3f}"
    )
    corr = float(np.corrcoef(pred_base, ca_oof)[0, 1])
    print(f"   corr(pred_base, ca_oof) = {corr:.4f}")
    print(
        f"   te_base:  mean={te_base.mean():.3f}  std={te_base.std():.3f}  "
        f"min={te_base.min():.3f}  max={te_base.max():.3f}"
    )
    print(
        f"   ca_te:    mean={ca_te.mean():.3f}  std={ca_te.std():.3f}  "
        f"min={ca_te.min():.3f}  max={ca_te.max():.3f}"
    )

    # Leak sanity on parent
    leak_eq = float(np.mean(np.isclose(pred_base, y_unb, atol=1e-6)))
    if leak_eq > 0.05:
        print(f"   WARN parent: {leak_eq:.1%} rows == truth -- possible leak")

    # -- Scaffolds ------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 2: scaffolds for outer CV")
    print("-" * 78)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"   n_unique_scaffolds = {n_unique_scaf}")

    # -- Multi-seed sweep -----------------------------------------------------
    print("\n" + "-" * 78)
    print(
        f"SWEEP: {len(KF_SEEDS)} FRESH kf_seeds "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}"
    )
    print("-" * 78)
    seed_records = []
    pooled_raes = []
    per_fold_means = []
    oof_stack = []
    all_fold_ql = []
    all_fold_qh = []
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(pred_base, ca_oof, y_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        per_fold_means.append(res["per_fold_val_rae_mean"])
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
            "n_fallback_lo": res["n_fallback_lo"],
            "n_fallback_hi": res["n_fallback_hi"],
        })
        print(
            f"   kf={s}: pooled_RAE={res['pooled_rae']:.4f}  "
            f"fold_mean={res['per_fold_val_rae_mean']:.4f}  "
            f"ql={res['fold_ql']}  qh={res['fold_qh']}  "
            f"fb(lo,hi)=({res['n_fallback_lo']},{res['n_fallback_hi']})  "
            f"wall={time.time()-ts:.2f}s"
        )

    arr = np.asarray(pooled_raes, dtype=np.float64)
    fm_arr = np.asarray(per_fold_means, dtype=np.float64)
    n_s = len(arr)
    mean_rae = float(arr.mean())
    std_rae = float(arr.std(ddof=1)) if n_s > 1 else 0.0
    sem = std_rae / np.sqrt(n_s) if n_s > 1 else 0.0
    t_mult = 2.145  # df=14, two-sided 95%
    ci_low = mean_rae - t_mult * sem
    ci_high = mean_rae + t_mult * sem
    median_rae = float(np.median(arr))

    # Per-fold-mean aggregate (gate metric per task)
    pfm_mean = float(fm_arr.mean())
    pfm_std = float(fm_arr.std(ddof=1)) if n_s > 1 else 0.0

    shift_vs_nb3200 = mean_rae - REF_NB3200
    shift_pfm_vs_nb3200 = pfm_mean - REF_NB3200

    # Most-picked q values
    ql_counter = Counter(all_fold_ql)
    qh_counter = Counter(all_fold_qh)
    ql_mode = ql_counter.most_common(1)[0][0]
    qh_mode = qh_counter.most_common(1)[0][0]

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds)")
    print("-" * 78)
    print(f"   pooled-RAE mean   = {mean_rae:.4f}")
    print(f"   pooled-RAE std    = {std_rae:.4f}")
    print(f"   pooled-RAE sem    = {sem:.4f}")
    print(f"   pooled-RAE 95% CI = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   pooled-RAE median = {median_rae:.4f}")
    print(f"   pooled-RAE min/max= [{arr.min():.4f}, {arr.max():.4f}]")
    print(f"\n   per-fold-mean ({n_s} seeds) = {pfm_mean:.4f} +/- {pfm_std:.4f}")
    print(f"   per-fold-mean min/max       = [{fm_arr.min():.4f}, {fm_arr.max():.4f}]")
    print(f"\n   ref nb3200 deep-30 gate   = {REF_NB3200:.4f}")
    print(f"   shift pooled vs nb3200    = {shift_vs_nb3200:+.4f}")
    print(f"   shift per-fold vs nb3200  = {shift_pfm_vs_nb3200:+.4f}")
    print(f"\n   ref {PARENT_TAG} parent          = {REF_PARENT_NB3090:.4f}")
    print(f"   delta pooled vs {PARENT_TAG}     = {mean_rae - REF_PARENT_NB3090:+.4f}")
    print(f"   ref nb3173 ceiling        = {REF_NB3173:.4f}")
    print(f"   ref chemprop_aux on 253   = {REF_CHEMPROP_AUX:.4f}")
    print(f"\n   ql_distribution (75 folds) = {dict(ql_counter)}")
    print(f"   qh_distribution (75 folds) = {dict(qh_counter)}")
    print(f"   ql_mode = {ql_mode}  qh_mode = {qh_mode}")

    # -- Deploy: pick (q_low, q_high) on FULL 253 by same inner search --------
    deploy_ql, deploy_qh, deploy_lo, deploy_hi = _pick_best_clip_fallback(
        y_unb, pred_base, ca_oof,
    )
    te_pred = _apply_fallback(te_base, ca_te, deploy_lo, deploy_hi).astype(np.float32)
    n_te_lo = int(np.sum(te_base < deploy_lo))
    n_te_hi = int(np.sum(te_base > deploy_hi))
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(
        f"\n   deploy clip pick = (q{deploy_ql:.2f}, q{deploy_qh:.2f}) -> "
        f"({deploy_lo:.3f}, {deploy_hi:.3f}) from full 253 y"
    )
    print(
        f"   te fallback: lo={n_te_lo}/513  hi={n_te_hi}/513  "
        f"total={n_te_lo + n_te_hi}/513"
    )
    print(
        f"   te(513) mean={te_pred.mean():.3f}  std={te_pred.std():.3f}  "
        f"min={te_pred.min():.3f}  max={te_pred.max():.3f}"
    )
    print(f"   te[unb] in-sample RAE  = {te_unb_in_rae:.4f}")

    # Median-seed OOF for storage
    med_seed_idx = int(np.argsort(arr)[n_s // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(f"   median seed = {median_seed} (rae={arr[med_seed_idx]:.4f})")

    # -- Gate -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE (per-fold-mean)")
    print("-" * 78)
    if pfm_mean < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE-CANDIDATE. nb3252 per-fold-mean {pfm_mean:.4f} beats "
            f"gate {GATE_BETTER:.4f} ({pfm_mean - GATE_BETTER:+.4f}). "
            f"chemprop_aux fallback on clipped rows of {PARENT_TAG} anchor "
            f"carries new orthogonal-axis information vs flat clip "
            f"(nb3200 0.4423). Pooled-RAE mean {mean_rae:.4f} +/- {std_rae:.4f}. "
            f"Modal pick (q{ql_mode}, q{qh_mode}). Deploy fallback applied to "
            f"{n_te_lo + n_te_hi}/{n_test} test rows. Re-verify with deep-30 "
            f"before PRIMARY-1 swap."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3252 per-fold-mean {pfm_mean:.4f} fails gate "
            f"{GATE_BETTER:.4f} ({pfm_mean - GATE_BETTER:+.4f}). "
            f"chemprop_aux fallback does NOT beat flat-clip baseline "
            f"(nb3200 0.4423) on the {PARENT_TAG} anchor. Pooled mean "
            f"{mean_rae:.4f}, shift vs nb3200 = {shift_vs_nb3200:+.4f}. "
            f"Either chemprop_aux on those rows is no more informative than "
            f"the quantile bound, or the 0.91 correlation between anchors "
            f"means fallback brings little new signal at the tails. "
            f"Keep nb3200 / nb3190 on the ladder."
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

    sub_csv = SUBMISSIONS / f"{TAG}_clip_chemprop_fallback.csv"
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
        "parent_tag": PARENT_TAG,
        "fallback_tag": FALLBACK_TAG,
        "method": (
            "per_fold_clip_with_chemprop_aux_fallback_on_clipped_rows"
        ),
        "anchor_pred_oof_path": str(PRED_OOF_PATH),
        "anchor_te_path": str(TE_PATH),
        "fallback_pred_oof_path": str(CA_OOF_PATH),
        "fallback_te_path": str(CA_TE_PATH),
        "anchor_pre_unblind": True,
        "fallback_pre_unblind": True,
        "parent_full_oof_rae": round(full_oof_rae, 4),
        "fallback_full_oof_rae": round(ca_oof_rae, 4),
        "corr_parent_fallback": round(corr, 4),
        "parent_leak_eq_truth_frac": round(leak_eq, 4),
        "q_low_grid": Q_LOW_GRID,
        "q_high_grid": Q_HIGH_GRID,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": n_s,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "seed_records": seed_records,
        "pooled_rae_array": [round(float(v), 4) for v in pooled_raes],
        "per_fold_mean_array": [round(float(v), 4) for v in per_fold_means],
        "mean_rae": round(mean_rae, 4),
        "std_rae": round(std_rae, 4),
        "sem_rae": round(sem, 4),
        "ci95_low": round(ci_low, 4),
        "ci95_high": round(ci_high, 4),
        "median_rae": round(median_rae, 4),
        "min_rae": round(float(arr.min()), 4),
        "max_rae": round(float(arr.max()), 4),
        "per_fold_mean_mean": round(pfm_mean, 4),
        "per_fold_mean_std": round(pfm_std, 4),
        "per_fold_mean_min": round(float(fm_arr.min()), 4),
        "per_fold_mean_max": round(float(fm_arr.max()), 4),
        "ql_distribution": {str(k): int(v) for k, v in ql_counter.items()},
        "qh_distribution": {str(k): int(v) for k, v in qh_counter.items()},
        "ql_mode": float(ql_mode),
        "qh_mode": float(qh_mode),
        "ref_parent_nb3090": REF_PARENT_NB3090,
        "delta_vs_parent": round(mean_rae - REF_PARENT_NB3090, 4),
        "ref_nb3190": REF_NB3190,
        "ref_nb3200": REF_NB3200,
        "shift_pooled_vs_nb3200": round(shift_vs_nb3200, 4),
        "shift_per_fold_vs_nb3200": round(shift_pfm_vs_nb3200, 4),
        "ref_nb3173": REF_NB3173,
        "ref_chemprop_aux_on_253": REF_CHEMPROP_AUX,
        "ref_nb2171": REF_NB2171,
        "deploy_ql": float(deploy_ql),
        "deploy_qh": float(deploy_qh),
        "deploy_lo": round(deploy_lo, 4),
        "deploy_hi": round(deploy_hi, 4),
        "n_te_fallback_lo": n_te_lo,
        "n_te_fallback_hi": n_te_hi,
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "te_min": float(te_pred.min()),
        "te_max": float(te_pred.max()),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "median_seed": int(median_seed),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": (
            str(sub_csv) if verdict == "BETTER" else None
        ),
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
    print(f"   pooled mean_rae ({n_s} seeds) = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   per-fold-mean ({n_s} seeds)   = {pfm_mean:.4f} +/- {pfm_std:.4f}")
    print(f"   95% CI                       = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   shift pooled vs nb3200       = {shift_vs_nb3200:+.4f}")
    print(f"   shift per-fold vs nb3200     = {shift_pfm_vs_nb3200:+.4f}")
    print(f"   verdict                      = {verdict}")
    print(f"   wall                         = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae", "std_rae", "ci95_low", "ci95_high",
        "per_fold_mean_mean", "per_fold_mean_std",
        "shift_pooled_vs_nb3200", "shift_per_fold_vs_nb3200",
        "delta_vs_parent",
        "ql_mode", "qh_mode",
        "deploy_ql", "deploy_qh", "deploy_lo", "deploy_hi",
        "n_te_fallback_lo", "n_te_fallback_hi",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
