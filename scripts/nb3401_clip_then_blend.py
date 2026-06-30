"""nb3401 -- Clip-then-blend: clip K18 and K19 SEPARATELY first, then q35 blend.

NEW PARADIGM (REVERSE of nb3200):
    nb3200 does q35 quantile-conditional blend FIRST, then per-fold learned clip
    on the blended output (blend -> clip). nb3401 reverses the operator order:
    each anchor is clipped to its OWN fold-train y-range BEFORE blending, then the
    q35 quantile-conditional blend is applied to the two clipped anchors
    (clip -> blend).

    Hypothesis: clipping each anchor independently before the blend bounds the
    K18 / K19 tails on the same y support, so the per-row q35 hard-split sees
    already-compressed inputs. The K18 threshold (which gates the blend weights)
    is recomputed on the CLIPPED K18, so the low/high routing is consistent with
    the clipped support. If clipping the anchors separately removes rank-order
    that the blend needs (or shifts the K18 routing threshold unfavourably), RAE
    inflates above the blend-then-clip ceiling and the gate fails.

PROTOCOL (per kf_seed, 5-fold scaffold split):
    p_k18 = nb2960_K18_30seed_oof   (253,)  deep-30 K18 residual on chemprop_aux
    p_k19 = nb3000_K19_30seed_oof   (253,)  deep-30 K19 residual on chemprop_aux
    Per outer fold:
        a) y_tr = y[fold_train]
           lo   = quantile(y_tr, 0.05)
           hi   = quantile(y_tr, 0.98)          # SAME band for both anchors
        b) k18_tr_c = clip(p_k18[tr], lo, hi)   # clip EACH anchor separately
           k18_va_c = clip(p_k18[va], lo, hi)
           k19_tr_c = clip(p_k19[tr], lo, hi)
           k19_va_c = clip(p_k19[va], lo, hi)
        c) q_thr = quantile(k18_tr_c, 0.35)     # q35 threshold on CLIPPED K18 train
           rows k18_va_c <= q_thr -> w_low *k18 + (1-w_low )*k19   (clipped inputs)
           rows k18_va_c >  q_thr -> w_high*k18 + (1-w_high)*k19   (clipped inputs)
        d) stitch blended fold-val into oof (253,); record fold-val RAE.
    Repeat for 15 FRESH kf_seeds {1216..1230}; per-fold-mean = mean of 5 fold-val
    RAEs, averaged over the 15 seeds (distinct from nb3200's pooled metric,
    matching this task's per-fold-mean prescription).

GATE (on per-fold-mean over 15 seeds):
    per-fold-mean < 0.4423 -> "BETTER"
    else                   -> "FAIL"

References:
    nb3200 (blend -> learned clip)  = 0.4424 (15-seed pooled)  <- order being reversed
    nb3090 (q35 blend, no clip)     = best combo (q=0.35,0.95,0.40)
    nb2960 K18 deep-30 OOF          = 0.4536
    nb3000 K19 deep-30 OOF          = 0.4607
    nb2171 prior post-hoc ceiling   = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb2960_K18_30seed_oof.npy
    data/processed/nb2960_K18_30seed_te.npy
    data/processed/nb3000_K19_30seed_oof.npy
    data/processed/te_nb3000_K19.npy

Outputs:
    data/processed/nb3401_summary.json
    data/processed/nb3401_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3401.npy         (513,) float32 -- deploy te
    submissions/nb3401_clip_then_blend.csv  (only on BETTER verdict)
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

TAG = "nb3401"

# -- Inputs (canonical K18 / K19 deep-30 anchors, same as nb3090) --------------
K18_OOF_PATH = DATA_PROCESSED / "nb2960_K18_30seed_oof.npy"
K18_TE_PATH = DATA_PROCESSED / "nb2960_K18_30seed_te.npy"
K19_OOF_PATH = DATA_PROCESSED / "nb3000_K19_30seed_oof.npy"
K19_TE_PATH = DATA_PROCESSED / "te_nb3000_K19.npy"

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))  # 15 FRESH seeds {1216..1230}

# -- Clip band (per task: each anchor clipped to (q05, q98) of fold-train y) ---
Q_CLIP_LOW = 0.05
Q_CLIP_HIGH = 0.98

# -- q35 quantile-conditional blend (nb3090 winning combo) ---------------------
Q_CUT = 0.35
W_LOW = 0.95
W_HIGH = 0.40

# -- Gate ----------------------------------------------------------------------
GATE_BETTER = 0.4423

# -- References ----------------------------------------------------------------
REF_NB3200 = 0.4424   # blend -> clip (order being reversed)
REF_K18 = 0.4536
REF_K19 = 0.4607
REF_NB2171 = 0.4682


def _clip_band(y_tr: np.ndarray) -> tuple[float, float]:
    """(lo, hi) = (q05, q98) of fold-train truth -- shared by both anchors."""
    lo = float(np.quantile(y_tr, Q_CLIP_LOW))
    hi = float(np.quantile(y_tr, Q_CLIP_HIGH))
    return lo, hi


def _blend_quantile_conditional(
    p_k18: np.ndarray,
    p_k19: np.ndarray,
    q_thr: float,
    w_low: float,
    w_high: float,
) -> np.ndarray:
    """nb3090 per-row hard-split blend on the K18 threshold.

    rows with p_k18 <= q_thr -> w_low  * p_k18 + (1-w_low ) * p_k19
    rows with p_k18 >  q_thr -> w_high * p_k18 + (1-w_high) * p_k19
    Here p_k18 / p_k19 are the CLIPPED anchors and q_thr is computed on the
    clipped K18 train values.
    """
    low_mask = p_k18 <= q_thr
    out = np.empty_like(p_k18, dtype=np.float64)
    out[low_mask] = w_low * p_k18[low_mask] + (1.0 - w_low) * p_k19[low_mask]
    out[~low_mask] = w_high * p_k18[~low_mask] + (1.0 - w_high) * p_k19[~low_mask]
    return out


def _run_one_seed(
    p_k18: np.ndarray,
    p_k19: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Run clip-then-blend pipeline at a single kf_seed.

    Returns oof (253,), the 5 per-fold-val RAEs, and per-fold diagnostics.
    """
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_lo = []
    fold_hi = []
    fold_q_thr = []
    fold_n_clip_k18 = []
    fold_n_clip_k19 = []
    fold_low_share = []
    for tr_loc, va_loc in splits:
        # --- (a) shared clip band from fold-train truth ---
        lo, hi = _clip_band(y_unb[tr_loc])
        fold_lo.append(lo)
        fold_hi.append(hi)

        # --- (b) clip EACH anchor separately, train and val ---
        k18_tr_c = np.clip(p_k18[tr_loc], lo, hi)
        k18_va_c = np.clip(p_k18[va_loc], lo, hi)
        k19_tr_c = np.clip(p_k19[tr_loc], lo, hi)
        k19_va_c = np.clip(p_k19[va_loc], lo, hi)

        n_clip_k18 = int(
            np.sum(p_k18[va_loc] < lo) + np.sum(p_k18[va_loc] > hi)
        )
        n_clip_k19 = int(
            np.sum(p_k19[va_loc] < lo) + np.sum(p_k19[va_loc] > hi)
        )
        fold_n_clip_k18.append(n_clip_k18)
        fold_n_clip_k19.append(n_clip_k19)

        # --- (c) q35 threshold on CLIPPED K18 train, blend clipped anchors ---
        q_thr = float(np.quantile(k18_tr_c, Q_CUT))
        fold_q_thr.append(q_thr)
        fold_low_share.append(float(np.mean(k18_va_c <= q_thr)))

        val_pred = _blend_quantile_conditional(
            k18_va_c, k19_va_c, q_thr, W_LOW, W_HIGH,
        )
        oof[va_loc] = val_pred
        fold_val_raes.append(float(rae(y_unb[va_loc], val_pred)))

    if np.isnan(oof).any():
        raise RuntimeError(
            f"kf_seed={kf_seed}: scaffold splits did not cover all rows"
        )
    pooled = float(rae(y_unb, oof))
    return {
        "kf_seed": int(kf_seed),
        "per_fold_mean_rae": float(np.mean(fold_val_raes)),
        "per_fold_val_rae_std": float(np.std(fold_val_raes, ddof=1)),
        "pooled_rae": pooled,
        "fold_lo_mean": float(np.mean(fold_lo)),
        "fold_hi_mean": float(np.mean(fold_hi)),
        "fold_q_thr_mean": float(np.mean(fold_q_thr)),
        "n_clipped_k18": int(np.sum(fold_n_clip_k18)),
        "n_clipped_k19": int(np.sum(fold_n_clip_k19)),
        "low_share_mean": float(np.mean(fold_low_share)),
        "oof": oof,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- CLIP-then-BLEND (reverse of nb3200): clip K18 & K19 "
          f"SEPARATELY to (q{Q_CLIP_LOW},q{Q_CLIP_HIGH}) of fold-train y, "
          f"then q{Q_CUT} blend")
    print(f"          clip band = (q{Q_CLIP_LOW}, q{Q_CLIP_HIGH}) of fold-train y "
          f"(shared by both anchors)")
    print(f"          q-blend   = q_cut={Q_CUT}  w_low={W_LOW}  w_high={W_HIGH}")
    print(f"          kf_seeds  = {len(KF_SEEDS)} fresh "
          f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}")
    print(f"          metric    = per-fold-mean (mean of 5 fold-val RAEs, "
          f"avg over seeds)")
    print(f"          gate: per-fold-mean < {GATE_BETTER:.4f} -> BETTER, "
          f"else FAIL")
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
    print(f"   y_unb: mean={y_unb.mean():.3f}  std={y_unb.std():.3f}  "
          f"min={y_unb.min():.3f}  max={y_unb.max():.3f}")

    # -- Load K18 / K19 deep-30 anchor OOFs + te arrays ----------------------
    print("\n" + "-" * 78)
    print("STEP 1: load K18 (nb2960) + K19 (nb3000) deep-30 OOFs + te arrays")
    print("-" * 78)
    p_k18 = np.load(K18_OOF_PATH).astype(np.float64)
    te_k18 = np.load(K18_TE_PATH).astype(np.float64)
    p_k19 = np.load(K19_OOF_PATH).astype(np.float64)
    te_k19 = np.load(K19_TE_PATH).astype(np.float64)
    for name, arr, exp in (
        ("K18 oof", p_k18, (n_unb,)),
        ("K18 te", te_k18, (n_test,)),
        ("K19 oof", p_k19, (n_unb,)),
        ("K19 te", te_k19, (n_test,)),
    ):
        if arr.shape != exp:
            raise ValueError(f"{name} shape {arr.shape} != {exp}")
    rae_k18 = float(rae(y_unb, p_k18))
    rae_k19 = float(rae(y_unb, p_k19))
    print(f"   K18 oof_RAE = {rae_k18:.4f} (ref {REF_K18:.4f})  "
          f"mean={p_k18.mean():.3f} std={p_k18.std():.3f}")
    print(f"   K19 oof_RAE = {rae_k19:.4f} (ref {REF_K19:.4f})  "
          f"mean={p_k19.mean():.3f} std={p_k19.std():.3f}")
    corr = float(np.corrcoef(p_k18, p_k19)[0, 1])
    print(f"   corr(K18, K19) = {corr:.4f}")

    # Leak sanity
    leak_k18 = float(np.mean(np.isclose(p_k18, y_unb, atol=1e-6)))
    leak_k19 = float(np.mean(np.isclose(p_k19, y_unb, atol=1e-6)))
    if leak_k18 > 0.05:
        print(f"   WARN K18: {leak_k18:.1%} rows == truth -- possible leak")
    if leak_k19 > 0.05:
        print(f"   WARN K19: {leak_k19:.1%} rows == truth -- possible leak")

    # -- Scaffolds -----------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 2: scaffolds for outer CV")
    print("-" * 78)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"   n_unique_scaffolds = {n_unique_scaf}")

    # -- Multi-seed sweep ----------------------------------------------------
    print("\n" + "-" * 78)
    print(f"SWEEP: {len(KF_SEEDS)} FRESH kf_seeds "
          f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}  (per-fold-mean metric)")
    print("-" * 78)
    seed_records = []
    per_fold_means = []
    pooled_raes = []
    oof_stack = []
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(p_k18, p_k19, y_unb, unb_scaffolds, s)
        per_fold_means.append(res["per_fold_mean_rae"])
        pooled_raes.append(res["pooled_rae"])
        oof_stack.append(res["oof"])
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "per_fold_mean_rae": round(res["per_fold_mean_rae"], 4),
            "per_fold_val_rae_std": round(res["per_fold_val_rae_std"], 4),
            "pooled_rae": round(res["pooled_rae"], 4),
            "fold_lo_mean": round(res["fold_lo_mean"], 4),
            "fold_hi_mean": round(res["fold_hi_mean"], 4),
            "fold_q_thr_mean": round(res["fold_q_thr_mean"], 4),
            "n_clipped_k18": res["n_clipped_k18"],
            "n_clipped_k19": res["n_clipped_k19"],
            "low_share_mean": round(res["low_share_mean"], 4),
        })
        print(f"   kf={s}: per_fold_mean={res['per_fold_mean_rae']:.4f}  "
              f"pooled={res['pooled_rae']:.4f}  "
              f"band=({res['fold_lo_mean']:.3f},{res['fold_hi_mean']:.3f})  "
              f"q_thr={res['fold_q_thr_mean']:.3f}  "
              f"clipped(K18,K19)=({res['n_clipped_k18']},{res['n_clipped_k19']})  "
              f"low_share={res['low_share_mean']:.3f}  "
              f"wall={time.time()-ts:.2f}s")

    arr = np.asarray(per_fold_means, dtype=np.float64)
    pooled_arr = np.asarray(pooled_raes, dtype=np.float64)
    n_s = len(arr)
    mean_rae = float(arr.mean())
    std_rae = float(arr.std(ddof=1)) if n_s > 1 else 0.0
    sem = std_rae / np.sqrt(n_s) if n_s > 1 else 0.0
    t_mult = 2.145  # df=14, two-sided 95%
    ci_low = mean_rae - t_mult * sem
    ci_high = mean_rae + t_mult * sem
    median_rae = float(np.median(arr))

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds, per-fold-mean metric)")
    print("-" * 78)
    print(f"   per-fold-mean     = {mean_rae:.4f}")
    print(f"   std               = {std_rae:.4f}")
    print(f"   sem               = {sem:.4f}")
    print(f"   95% CI (df=14)    = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   median            = {median_rae:.4f}")
    print(f"   min/max           = [{arr.min():.4f}, {arr.max():.4f}]")
    print(f"   pooled mean (ref) = {pooled_arr.mean():.4f}")
    print(f"\n   ref nb3200 (blend->clip) = {REF_NB3200:.4f}")
    print(f"   delta vs nb3200          = {mean_rae - REF_NB3200:+.4f}")
    print(f"   ref K18 / K19 deep-30    = {REF_K18:.4f} / {REF_K19:.4f}")
    print(f"   ref nb2171 post-hoc      = {REF_NB2171:.4f}")

    # -- Deploy: clip each anchor to (q05,q98) of FULL 253 y, then blend -----
    print("\n" + "-" * 78)
    print("STEP 3: deploy te (clip each anchor to full-253 (q05,q98), q35 blend)")
    print("-" * 78)
    dlo, dhi = _clip_band(y_unb)
    te_k18_c = np.clip(te_k18, dlo, dhi)
    te_k19_c = np.clip(te_k19, dlo, dhi)
    # q35 threshold on the full-253 CLIPPED K18 OOF (deploy convention)
    k18_oof_c = np.clip(p_k18, dlo, dhi)
    deploy_q_thr = float(np.quantile(k18_oof_c, Q_CUT))
    te_pred = _blend_quantile_conditional(
        te_k18_c, te_k19_c, deploy_q_thr, W_LOW, W_HIGH,
    ).astype(np.float32)
    n_te_clip_k18 = int(np.sum(te_k18 < dlo) + np.sum(te_k18 > dhi))
    n_te_clip_k19 = int(np.sum(te_k19 < dlo) + np.sum(te_k19 > dhi))
    te_low_share = float(np.mean(te_k18_c <= deploy_q_thr))
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"   deploy clip band = (q{Q_CLIP_LOW},q{Q_CLIP_HIGH}) -> "
          f"({dlo:.3f}, {dhi:.3f})  on full 253 y")
    print(f"   deploy q_thr (full clipped-K18 q{Q_CUT}) = {deploy_q_thr:.4f}")
    print(f"   te clipped: K18={n_te_clip_k18}/513  K19={n_te_clip_k19}/513")
    print(f"   te(513) low-half share = {te_low_share:.3f}")
    print(f"   te(513) mean={te_pred.mean():.3f}  std={te_pred.std():.3f}  "
          f"min={te_pred.min():.3f}  max={te_pred.max():.3f}")
    print(f"   te[unb] in-sample RAE  = {te_unb_in_rae:.4f}")

    # Median-seed OOF for storage
    med_seed_idx = int(np.argsort(arr)[n_s // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(f"   median seed = {median_seed} "
          f"(per_fold_mean={arr[med_seed_idx]:.4f})")

    # -- Gate ----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    if mean_rae < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE-CANDIDATE. nb3401 clip-then-blend per-fold-mean "
            f"{mean_rae:.4f} BEATS the nb3200 blend-then-clip order "
            f"{REF_NB3200:.4f} ({mean_rae - REF_NB3200:+.4f}). Clipping each "
            f"anchor to its own fold-train (q{Q_CLIP_LOW},q{Q_CLIP_HIGH}) BEFORE "
            f"the q{Q_CUT} blend compresses the K18/K19 tails on shared support "
            f"and recomputes the routing threshold on clipped K18, which the "
            f"blend exploits. Re-verify with deep-30 before any PRIMARY-1 swap "
            f"(cycle-160 deep-30 rule: 15-seed std under-dispersed ~4x)."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3401 clip-then-blend per-fold-mean {mean_rae:.4f} does "
            f"NOT beat the nb3200 blend-then-clip order {REF_NB3200:.4f} "
            f"({mean_rae - REF_NB3200:+.4f}). Reversing the operator order "
            f"(clip each anchor first, then q{Q_CUT} blend) does not help: "
            f"clipping K18/K19 independently before the blend either removes "
            f"rank-order the q-split needs or shifts the K18 routing threshold "
            f"unfavourably vs clipping the blended output once. Keep nb3200 "
            f"(blend->clip) order; operator-order axis closed for this anchor "
            f"pair."
        )
    print(f"   verdict       = {verdict}")
    print(f"   ladder action = {ladder_action}")

    # -- Save artifacts ------------------------------------------------------
    print("\n" + "-" * 78)
    print("SAVE")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_for_save)
    np.save(te_path, te_pred)
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_clip_then_blend.csv"
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
        "method": (
            "clip_each_anchor_separately_to_q05q98_fold_train_y_then_q35_blend"
        ),
        "reverse_of": "nb3200 (q35 blend -> learned clip)",
        "k18_oof_path": str(K18_OOF_PATH),
        "k18_te_path": str(K18_TE_PATH),
        "k19_oof_path": str(K19_OOF_PATH),
        "k19_te_path": str(K19_TE_PATH),
        "anchor_pre_unblind": True,
        "k18_full_oof_rae": round(rae_k18, 4),
        "k19_full_oof_rae": round(rae_k19, 4),
        "k18_k19_corr": round(corr, 4),
        "k18_leak_eq_truth_frac": round(leak_k18, 4),
        "k19_leak_eq_truth_frac": round(leak_k19, 4),
        "q_clip_low": Q_CLIP_LOW,
        "q_clip_high": Q_CLIP_HIGH,
        "q_cut": Q_CUT,
        "w_low": W_LOW,
        "w_high": W_HIGH,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": n_s,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "metric": (
            "per_fold_mean (mean of 5 fold-val RAEs, averaged over seeds)"
        ),
        "seed_records": seed_records,
        "per_fold_mean_array": [round(float(v), 4) for v in per_fold_means],
        "pooled_rae_array": [round(float(v), 4) for v in pooled_raes],
        "per_fold_mean_rae": round(mean_rae, 4),
        "std_rae": round(std_rae, 4),
        "sem_rae": round(sem, 4),
        "ci95_low": round(ci_low, 4),
        "ci95_high": round(ci_high, 4),
        "median_rae": round(median_rae, 4),
        "min_rae": round(float(arr.min()), 4),
        "max_rae": round(float(arr.max()), 4),
        "pooled_mean_rae": round(float(pooled_arr.mean()), 4),
        "ref_nb3200": REF_NB3200,
        "delta_vs_nb3200": round(mean_rae - REF_NB3200, 4),
        "ref_K18_deep30": REF_K18,
        "ref_K19_deep30": REF_K19,
        "ref_nb2171": REF_NB2171,
        "deploy_clip_lo": round(dlo, 4),
        "deploy_clip_hi": round(dhi, 4),
        "deploy_q_thr": round(deploy_q_thr, 4),
        "n_te_clipped_k18": n_te_clip_k18,
        "n_te_clipped_k19": n_te_clip_k19,
        "te_low_share": round(te_low_share, 4),
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
    print(f"   per-fold-mean ({n_s} seeds) = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   95% CI                  = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   delta vs nb3200         = {mean_rae - REF_NB3200:+.4f}")
    print(f"   verdict                 = {verdict}")
    print(f"   wall                    = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "per_fold_mean_rae", "std_rae", "ci95_low", "ci95_high",
        "pooled_mean_rae", "delta_vs_nb3200",
        "deploy_clip_lo", "deploy_clip_hi", "deploy_q_thr",
        "n_te_clipped_k18", "n_te_clipped_k19", "te_low_share",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
