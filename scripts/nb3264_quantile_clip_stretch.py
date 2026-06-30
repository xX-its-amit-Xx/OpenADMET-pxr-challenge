"""nb3264 -- Quantile blend + clip + per-fold scalar stretch (3-operator stack).

NEW PARADIGM: stack three post-hoc operators sequentially on nb3070.

    Most prior post-hoc work applies ONE operator at a time. nb3070 is the
    wide-seed verified q50-quantile-conditional hard-split blend on {K18, K19}
    deep-30 (pf_mean ~ 0.4477 wide pooled). nb3220 ran learned-clip on nb3070
    and FAILED at pf_mean 0.4536 (the learned grid over-fit), so this script
    pins the clip to fixed quantiles (q05, q98) -- a generic variance-reining
    operator with no inner search -- and then chains a per-fold golden-section
    scalar rank-stretch in [0.95, 1.20] on top. The hypothesis: a non-tuned
    clip plus a 1-parameter rank-stretch are mutually orthogonal post-hoc
    operators on a quantile-conditional blend anchor. If true, the stacked
    pipeline should improve on each individual operator's pf_mean.

PROTOCOL (per kf_seed, 5-fold scaffold split):
    Step 1 (quantile blend, MATCHES nb3070 exactly):
        per fold: q50 = median(K18_pred[tr_loc])
                  rows with p_k18 <= q50  -> 0.8*K18 + 0.2*K19
                  rows with p_k18 >  q50  -> 0.5*K18 + 0.5*K19
        -> blended pred (in-sample on train, out-of-sample on val)
    Step 2 (FIXED clip q05/q98 on blended pred):
        per fold: lo = quantile(blend[tr_loc], 0.05)
                  hi = quantile(blend[tr_loc], 0.98)
                  clipped[va_loc] = np.clip(blend[va_loc], lo, hi)
    Step 3 (per-fold golden-section scalar stretch on clipped pred):
        per fold: mu_tr = mean(clipped[tr_loc])
                  s_star = argmin_{s in [0.95, 1.20]} RAE(y_tr,
                              mu_tr + s*(clipped[tr_loc] - mu_tr))
                  oof[va_loc] = mu_tr + s_star * (clipped[va_loc] - mu_tr)

    Repeat for 15 FRESH kf_seeds {1216..1230}; report mean +/- std + 95% CI.

GATE (on 15-seed PER-FOLD-MEAN):
    pf_mean < 0.4423 -> "BETTER"
    else             -> "FAIL"

References:
    nb3070 wide-seed verify (q50, parent) = 0.4477
    nb3190 learned-clip on nb3090 (q35)   = 0.4422  <- compounding target
    nb3220 learned-clip on nb3070 (q50)   = 0.4536  <- FAIL on pf_mean
    nb3173 clip-on-nb3080 ceiling          = 0.4437
    nb3022 per-fold stretch on nb3002      (scalar stretch precedent)
    nb2171 prior PRIMARY-1                 = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb2960_K18_30seed_oof.npy
    data/processed/nb2960_K18_30seed_te.npy
    data/processed/nb3000_K19_30seed_oof.npy
    data/processed/te_nb3000_K19.npy

Outputs:
    data/processed/nb3264_summary.json
    data/processed/nb3264_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3264.npy         (513,) float32 -- deploy te
    submissions/nb3264_quantile_clip_stretch.csv  (only on BETTER verdict)
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

TAG = "nb3264"
PARENT_TAG = "nb3070"

# -- Inputs --------------------------------------------------------------------
K_LABELS = ["K18", "K19"]
OOF_PATHS = {
    "K18": DATA_PROCESSED / "nb2960_K18_30seed_oof.npy",
    "K19": DATA_PROCESSED / "nb3000_K19_30seed_oof.npy",
}
TE_PATHS = {
    "K18": DATA_PROCESSED / "nb2960_K18_30seed_te.npy",
    "K19": DATA_PROCESSED / "te_nb3000_K19.npy",
}
K_DEPTH = {"K18": "deep30", "K19": "deep30"}

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))  # 15 FRESH seeds {1216..1230}

# -- Quantile blend (Step 1; MATCHES nb3070 / nb3063 exactly) ------------------
W_LOW_K18, W_LOW_K19 = 0.8, 0.2
W_HIGH_K18, W_HIGH_K19 = 0.5, 0.5
QUANTILE_CUT = 0.5  # median split for K18

# -- Fixed clip (Step 2; per-task spec q05 / q98) ------------------------------
Q_LOW = 0.05
Q_HIGH = 0.98

# -- Per-fold golden-section scalar stretch (Step 3; per-task spec) ------------
S_LO = 0.95
S_HI = 1.20
GS_TOL = 1e-4
GS_MAX_ITER = 60

# -- Gates ---------------------------------------------------------------------
GATE_BETTER = 0.4423  # pf_mean < this -> BETTER

# -- References ----------------------------------------------------------------
REF_PARENT_NB3070 = 0.4477
REF_NB3190 = 0.4422
REF_NB3220 = 0.4536  # learned clip on nb3070 (pf_mean) -- FAILED
REF_NB3173 = 0.4437
REF_NB2171 = 0.4682


def _blend_quantile_conditional(
    p_k18: np.ndarray,
    p_k19: np.ndarray,
    q50: float,
) -> np.ndarray:
    """Per-row hard-split blend (matches nb3070 exactly)."""
    low_mask = p_k18 <= q50
    out = np.empty_like(p_k18, dtype=np.float64)
    out[low_mask] = W_LOW_K18 * p_k18[low_mask] + W_LOW_K19 * p_k19[low_mask]
    out[~low_mask] = W_HIGH_K18 * p_k18[~low_mask] + W_HIGH_K19 * p_k19[~low_mask]
    return out


def _stretch(pred: np.ndarray, mu: float, s: float) -> np.ndarray:
    return mu + s * (pred - mu)


def golden_section_min(f, lo, hi, tol=GS_TOL, max_iter=GS_MAX_ITER):
    """Minimize unimodal f on [lo, hi] via golden-section search."""
    phi = (np.sqrt(5.0) - 1.0) / 2.0  # 0.618...
    a, b = float(lo), float(hi)
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = f(c)
    fd = f(d)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = f(d)
    if fc < fd:
        return c, fc
    return d, fd


def _run_one_seed(
    P_unb: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Run the 3-operator stack at a single kf_seed."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_final = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_q50s = []
    fold_los = []
    fold_his = []
    fold_n_clipped_lo = []
    fold_n_clipped_hi = []
    fold_mu_trs = []
    fold_s_stars = []
    fold_train_rae_s_star = []

    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        # --- Step 1: quantile-conditional blend (q50 hard split) ---
        q50 = float(np.median(P_unb[tr_loc, 0]))
        tr_blend = _blend_quantile_conditional(
            P_unb[tr_loc, 0], P_unb[tr_loc, 1], q50,
        )
        va_blend = _blend_quantile_conditional(
            P_unb[va_loc, 0], P_unb[va_loc, 1], q50,
        )
        fold_q50s.append(q50)

        # --- Step 2: fixed clip (q05, q98) learned on fold-train blend ---
        lo = float(np.quantile(tr_blend, Q_LOW))
        hi = float(np.quantile(tr_blend, Q_HIGH))
        fold_los.append(lo)
        fold_his.append(hi)
        n_lo = int(np.sum(va_blend < lo))
        n_hi = int(np.sum(va_blend > hi))
        fold_n_clipped_lo.append(n_lo)
        fold_n_clipped_hi.append(n_hi)
        tr_clipped = np.clip(tr_blend, lo, hi)
        va_clipped = np.clip(va_blend, lo, hi)

        # --- Step 3: per-fold golden-section stretch on clipped pred ---
        y_tr = y_unb[tr_loc]
        mu_tr = float(np.mean(tr_clipped))
        fold_mu_trs.append(mu_tr)

        def f(s, p_tr=tr_clipped, y_tr=y_tr, mu_tr=mu_tr):
            return float(rae(y_tr, _stretch(p_tr, mu_tr, s)))

        s_star, fold_tr_rae = golden_section_min(f, S_LO, S_HI)
        fold_s_stars.append(s_star)
        fold_train_rae_s_star.append(fold_tr_rae)

        val_pred = _stretch(va_clipped, mu_tr, s_star)
        oof_final[va_loc] = val_pred
        fold_val_raes.append(float(rae(y_unb[va_loc], val_pred)))

    if np.isnan(oof_final).any():
        raise RuntimeError(
            f"kf_seed={kf_seed}: scaffold splits did not cover all rows"
        )
    pooled = float(rae(y_unb, oof_final))
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(fold_val_raes)),
        "per_fold_val_rae_std": float(np.std(fold_val_raes, ddof=1)),
        "fold_q50_mean": float(np.mean(fold_q50s)),
        "fold_lo_mean": float(np.mean(fold_los)),
        "fold_hi_mean": float(np.mean(fold_his)),
        "n_clipped_lo": int(np.sum(fold_n_clipped_lo)),
        "n_clipped_hi": int(np.sum(fold_n_clipped_hi)),
        "fold_s_star_mean": float(np.mean(fold_s_stars)),
        "fold_s_star_std": float(np.std(fold_s_stars, ddof=1)),
        "fold_s_stars": fold_s_stars,
        "fold_train_rae_s_star_mean": float(np.mean(fold_train_rae_s_star)),
        "oof": oof_final,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(
        f"{TAG} -- 3-OPERATOR STACK on {PARENT_TAG} anchors "
        f"({K_LABELS[0]} + {K_LABELS[1]} deep-30)"
    )
    print(f"   Step 1: quantile blend  q_cut={QUANTILE_CUT}  "
          f"low=({W_LOW_K18}, {W_LOW_K19})  high=({W_HIGH_K18}, {W_HIGH_K19})")
    print(f"   Step 2: fixed clip      (q{Q_LOW:.2f}, q{Q_HIGH:.2f}) on "
          "fold-train blend distribution")
    print(f"   Step 3: per-fold golden-section stretch  s in [{S_LO}, {S_HI}]")
    print(
        f"   kf_seeds = {len(KF_SEEDS)} fresh "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}"
    )
    print(f"   gate    : pf_mean < {GATE_BETTER:.4f} -> BETTER, else FAIL")
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

    # -- Load K18, K19 deep-30 anchor OOFs + te arrays ------------------------
    print("\n" + "-" * 78)
    print("STEP 1: load K18, K19 deep-30 OOFs and te arrays")
    print("-" * 78)
    oof_cols, te_cols = [], []
    per_K_full_rae = {}
    leak_flags = {}
    for k in K_LABELS:
        oof = np.load(OOF_PATHS[k]).astype(np.float64)
        te_arr = np.load(TE_PATHS[k]).astype(np.float64)
        if oof.shape != (n_unb,):
            raise ValueError(f"{k} OOF shape {oof.shape} != ({n_unb},)")
        if te_arr.shape != (n_test,):
            raise ValueError(f"{k} te shape {te_arr.shape} != ({n_test},)")
        oof_cols.append(oof)
        te_cols.append(te_arr)
        r = float(rae(y_unb, oof))
        per_K_full_rae[k] = round(r, 4)
        frac = float(np.mean(np.isclose(oof, y_unb, atol=1e-6)))
        leak_flags[k] = round(frac, 4)
        print(
            f"   {k} ({K_DEPTH[k]:>6s}): oof_RAE = {r:.4f}  "
            f"te_mean={te_arr.mean():.3f}  te_std={te_arr.std():.3f}  "
            f"leak_eq={frac:.3f}"
        )

    P_unb = np.column_stack(oof_cols)  # (253, 2)
    P_te = np.column_stack(te_cols)    # (513, 2)
    corr = float(np.corrcoef(P_unb.T)[0, 1])
    print(f"   pairwise corr({K_LABELS[0]}, {K_LABELS[1]}) = {corr:.4f}")
    print(
        f"   y_unb stats: mean={y_unb.mean():.3f}  std={y_unb.std():.3f}  "
        f"min={y_unb.min():.3f}  max={y_unb.max():.3f}"
    )

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
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}  (3-op stack per fold)"
    )
    print("-" * 78)
    seed_records = []
    pooled_raes = []
    per_fold_means = []
    oof_stack = []
    all_s_stars = []
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(P_unb, y_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        per_fold_means.append(res["per_fold_val_rae_mean"])
        oof_stack.append(res["oof"])
        all_s_stars.extend(res["fold_s_stars"])
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
            "per_fold_val_rae_std": round(res["per_fold_val_rae_std"], 4),
            "fold_q50_mean": round(res["fold_q50_mean"], 4),
            "fold_lo_mean": round(res["fold_lo_mean"], 4),
            "fold_hi_mean": round(res["fold_hi_mean"], 4),
            "n_clipped_lo": res["n_clipped_lo"],
            "n_clipped_hi": res["n_clipped_hi"],
            "fold_s_star_mean": round(res["fold_s_star_mean"], 4),
            "fold_s_star_std": round(res["fold_s_star_std"], 4),
            "fold_s_stars": [round(v, 4) for v in res["fold_s_stars"]],
            "fold_train_rae_s_star_mean": round(
                res["fold_train_rae_s_star_mean"], 4
            ),
        })
        print(
            f"   kf={s}: pooled={res['pooled_rae']:.4f}  "
            f"pf_mean={res['per_fold_val_rae_mean']:.4f}  "
            f"q50={res['fold_q50_mean']:.3f}  "
            f"clip(lo,hi)=({res['fold_lo_mean']:.3f},{res['fold_hi_mean']:.3f})  "
            f"n_clip=({res['n_clipped_lo']},{res['n_clipped_hi']})  "
            f"s_mean={res['fold_s_star_mean']:.3f}  "
            f"wall={time.time()-ts:.2f}s"
        )

    arr = np.asarray(pooled_raes, dtype=np.float64)
    n_s = len(arr)
    mean_rae = float(arr.mean())
    std_rae = float(arr.std(ddof=1)) if n_s > 1 else 0.0
    sem = std_rae / np.sqrt(n_s) if n_s > 1 else 0.0
    t_mult = 2.145  # df=14, two-sided 95%
    ci_low = mean_rae - t_mult * sem
    ci_high = mean_rae + t_mult * sem
    median_rae = float(np.median(arr))

    arr_pf = np.asarray(per_fold_means, dtype=np.float64)
    pf_mean = float(arr_pf.mean())
    pf_std = float(arr_pf.std(ddof=1)) if n_s > 1 else 0.0
    pf_sem = pf_std / np.sqrt(n_s) if n_s > 1 else 0.0
    pf_ci_low = pf_mean - t_mult * pf_sem
    pf_ci_high = pf_mean + t_mult * pf_sem
    pf_median = float(np.median(arr_pf))

    all_s_arr = np.asarray(all_s_stars, dtype=np.float64)
    s_mean_all = float(all_s_arr.mean())
    s_std_all = float(all_s_arr.std(ddof=1)) if len(all_s_arr) > 1 else 0.0
    s_at_lo_frac = float(np.mean(np.isclose(all_s_arr, S_LO, atol=1e-3)))
    s_at_hi_frac = float(np.mean(np.isclose(all_s_arr, S_HI, atol=1e-3)))

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds, 5 folds each = {n_s*N_FOLDS} folds total)")
    print("-" * 78)
    print(f"   POOLED RAE:")
    print(f"     mean   = {mean_rae:.4f}")
    print(f"     std    = {std_rae:.4f}")
    print(f"     sem    = {sem:.4f}")
    print(f"     95% CI = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"     median = {median_rae:.4f}")
    print(f"     min/max = [{arr.min():.4f}, {arr.max():.4f}]")
    print(f"   PER-FOLD-MEAN RAE (HONEST GATE METRIC):")
    print(f"     mean   = {pf_mean:.4f}")
    print(f"     std    = {pf_std:.4f}")
    print(f"     sem    = {pf_sem:.4f}")
    print(f"     95% CI = [{pf_ci_low:.4f}, {pf_ci_high:.4f}]")
    print(f"     median = {pf_median:.4f}")
    print(f"     min/max = [{arr_pf.min():.4f}, {arr_pf.max():.4f}]")
    print(f"   stretch s_star aggregate (n={len(all_s_arr)} folds):")
    print(
        f"     mean   = {s_mean_all:.4f}  std = {s_std_all:.4f}  "
        f"frac_at_lo({S_LO})={s_at_lo_frac:.2f}  "
        f"frac_at_hi({S_HI})={s_at_hi_frac:.2f}"
    )
    print(
        f"\n   ref {PARENT_TAG} (parent, q50-blend)   = {REF_PARENT_NB3070:.4f}"
    )
    print(
        f"   delta vs {PARENT_TAG} (pf_mean)        = "
        f"{pf_mean - REF_PARENT_NB3070:+.4f}"
    )
    print(
        f"   ref nb3190 (clip on q35-blend)   = {REF_NB3190:.4f}  "
        f"(compounding target)"
    )
    print(
        f"   delta vs nb3190 (pf_mean)        = "
        f"{pf_mean - REF_NB3190:+.4f}"
    )
    print(
        f"   ref nb3220 (clip on q50-blend)   = {REF_NB3220:.4f}  (FAILED)"
    )
    print(
        f"   delta vs nb3220 (pf_mean)        = "
        f"{pf_mean - REF_NB3220:+.4f}"
    )
    print(f"   ref nb2171 prior PRIMARY-1       = {REF_NB2171:.4f}")
    print(f"   gain vs nb2171 (pf_mean)         = {REF_NB2171 - pf_mean:+.4f}")

    # -- Deploy: fit all 3 ops on FULL 253 then apply to te[513] --------------
    deploy_q50 = float(np.median(P_unb[:, 0]))
    full_blend_unb = _blend_quantile_conditional(
        P_unb[:, 0], P_unb[:, 1], deploy_q50,
    )
    full_blend_te = _blend_quantile_conditional(
        P_te[:, 0], P_te[:, 1], deploy_q50,
    )
    deploy_lo = float(np.quantile(full_blend_unb, Q_LOW))
    deploy_hi = float(np.quantile(full_blend_unb, Q_HIGH))
    full_clipped_unb = np.clip(full_blend_unb, deploy_lo, deploy_hi)
    full_clipped_te = np.clip(full_blend_te, deploy_lo, deploy_hi)

    deploy_mu_tr = float(np.mean(full_clipped_unb))

    def f_deploy(s, p_tr=full_clipped_unb, y_tr=y_unb, mu_tr=deploy_mu_tr):
        return float(rae(y_tr, _stretch(p_tr, mu_tr, s)))

    deploy_s_full, _ = golden_section_min(f_deploy, S_LO, S_HI)
    # Deploy stretch uses MEAN of per-fold s_stars (robust) rather than the
    # full-fit s (which over-fits on the 253). See nb3022 precedent.
    deploy_s_robust = s_mean_all
    te_pred = _stretch(
        full_clipped_te, deploy_mu_tr, deploy_s_robust,
    ).astype(np.float32)
    te_pred = np.clip(te_pred, 3.0, 9.0)
    n_te_lo = int(np.sum(full_blend_te < deploy_lo))
    n_te_hi = int(np.sum(full_blend_te > deploy_hi))
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"\n   deploy q50 (full K18 OOF median) = {deploy_q50:.4f}")
    print(
        f"   deploy clip = (q{Q_LOW:.2f}, q{Q_HIGH:.2f}) -> "
        f"({deploy_lo:.3f}, {deploy_hi:.3f})"
    )
    print(
        f"   te clipped: lo={n_te_lo}/513  hi={n_te_hi}/513  "
        f"total={n_te_lo + n_te_hi}/513"
    )
    print(f"   deploy mu_tr_clipped = {deploy_mu_tr:.4f}")
    print(
        f"   deploy_s_full_fit  = {deploy_s_full:.4f}  "
        f"(in-sample on 253, NOT used)"
    )
    print(
        f"   deploy_s_robust    = {deploy_s_robust:.4f}  "
        f"(mean of {len(all_s_arr)} fold s_stars, USED)"
    )
    print(
        f"   te(513) mean={te_pred.mean():.3f}  std={te_pred.std():.3f}  "
        f"min={te_pred.min():.3f}  max={te_pred.max():.3f}"
    )
    print(f"   te[unb] in-sample RAE  = {te_unb_in_rae:.4f}")

    # Median-seed OOF for storage (median over per-fold-mean -- honest metric)
    med_seed_idx = int(np.argsort(arr_pf)[n_s // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(
        f"   median seed = {median_seed} "
        f"(pf_mean={arr_pf[med_seed_idx]:.4f}, pooled={arr[med_seed_idx]:.4f})"
    )

    # -- Gate (on PER-FOLD-MEAN per task) ------------------------------------
    print("\n" + "-" * 78)
    print("GATE (honest metric = PER-FOLD-MEAN)")
    print("-" * 78)
    if pf_mean < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE-CANDIDATE. nb3264 15-seed PER-FOLD-MEAN {pf_mean:.4f} "
            f"clears BETTER gate {GATE_BETTER:.4f} "
            f"({pf_mean - GATE_BETTER:+.4f}). The 3-operator stack "
            f"(quantile-q50 + fixed-clip q{Q_LOW}/q{Q_HIGH} + per-fold "
            f"golden-section stretch) compounds three orthogonal post-hoc "
            f"axes on the {PARENT_TAG} parent ({REF_PARENT_NB3070:.4f}) for "
            f"a {REF_PARENT_NB3070 - pf_mean:+.4f} RAE reduction, AND beats "
            f"the nb3190 compounding target ({REF_NB3190:.4f}) by "
            f"{REF_NB3190 - pf_mean:+.4f}. The fixed-clip choice (no inner "
            f"search) sidesteps the over-fit that hurt nb3220 "
            f"({REF_NB3220:.4f}). Deploy stretch uses mean of "
            f"{len(all_s_arr)} per-fold s_stars = {deploy_s_robust:.4f}. "
            f"Re-verify with deep-30 before PRIMARY-1 swap. "
            f"anchor_pre_unblind=True (K18/K19 deep-30 PRE-clean)."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3264 15-seed PER-FOLD-MEAN {pf_mean:.4f} fails "
            f"BETTER gate {GATE_BETTER:.4f} "
            f"({pf_mean - GATE_BETTER:+.4f}). Delta vs parent nb3070 "
            f"(pf_mean) = {pf_mean - REF_PARENT_NB3070:+.4f}, delta vs "
            f"nb3190 = {pf_mean - REF_NB3190:+.4f}. The 3-operator stack "
            f"on the q50 anchor does not compound either (a) the clip "
            f"injures the rank order the stretch then mis-decompresses, "
            f"(b) the q50 anchor already absorbs the clip mass that q35 "
            f"left open in nb3190, or (c) the per-fold stretch over-fits "
            f"on clipped fold-train predictions. Stretch picks: "
            f"mean={s_mean_all:.4f} (frac_at_lo={s_at_lo_frac:.2f}, "
            f"frac_at_hi={s_at_hi_frac:.2f}) -- watch for boundary pin. "
            f"Closes the q50+clip+stretch stacking axis at the current "
            f"target gate."
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

    sub_csv = SUBMISSIONS / f"{TAG}_quantile_clip_stretch.csv"
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
        "method": (
            "three_operator_stack_quantile_blend_then_fixed_clip_q05_q98_"
            "then_per_fold_golden_section_stretch_on_K18_K19_deep30"
        ),
        "anchor_pool": K_LABELS,
        "anchor_depth": K_DEPTH,
        "anchor_pre_unblind": True,
        "per_K_full_oof_rae": per_K_full_rae,
        "anchor_leak_eq_truth_frac": leak_flags,
        "oof_pairwise_corr": round(corr, 4),
        "step1_w_low": {"K18": W_LOW_K18, "K19": W_LOW_K19},
        "step1_w_high": {"K18": W_HIGH_K18, "K19": W_HIGH_K19},
        "step1_quantile_cut": QUANTILE_CUT,
        "step2_q_low": Q_LOW,
        "step2_q_high": Q_HIGH,
        "step3_s_lo": S_LO,
        "step3_s_hi": S_HI,
        "step3_gs_tol": GS_TOL,
        "step3_gs_max_iter": GS_MAX_ITER,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": n_s,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "seed_records": seed_records,
        "pooled_rae_array": [round(float(v), 4) for v in pooled_raes],
        "per_fold_val_rae_means_array": [
            round(float(v), 4) for v in per_fold_means
        ],
        "mean_rae": round(mean_rae, 4),
        "std_rae": round(std_rae, 4),
        "sem_rae": round(sem, 4),
        "ci95_low": round(ci_low, 4),
        "ci95_high": round(ci_high, 4),
        "median_rae": round(median_rae, 4),
        "min_rae": round(float(arr.min()), 4),
        "max_rae": round(float(arr.max()), 4),
        "per_fold_mean_rae_mean": round(pf_mean, 4),
        "per_fold_mean_rae_std": round(pf_std, 4),
        "per_fold_mean_rae_sem": round(pf_sem, 4),
        "per_fold_mean_rae_ci95_low": round(pf_ci_low, 4),
        "per_fold_mean_rae_ci95_high": round(pf_ci_high, 4),
        "per_fold_mean_rae_median": round(pf_median, 4),
        "per_fold_mean_rae_min": round(float(arr_pf.min()), 4),
        "per_fold_mean_rae_max": round(float(arr_pf.max()), 4),
        "honest_metric": "per_fold_mean",
        "stretch_s_star_mean_all_folds": round(s_mean_all, 4),
        "stretch_s_star_std_all_folds": round(s_std_all, 4),
        "stretch_frac_at_lo": round(s_at_lo_frac, 4),
        "stretch_frac_at_hi": round(s_at_hi_frac, 4),
        "ref_parent_nb3070": REF_PARENT_NB3070,
        "delta_vs_parent_pooled": round(mean_rae - REF_PARENT_NB3070, 4),
        "delta_vs_parent_pf_mean": round(pf_mean - REF_PARENT_NB3070, 4),
        "ref_nb3190": REF_NB3190,
        "delta_vs_nb3190_pf_mean": round(pf_mean - REF_NB3190, 4),
        "ref_nb3220": REF_NB3220,
        "delta_vs_nb3220_pf_mean": round(pf_mean - REF_NB3220, 4),
        "ref_nb3173": REF_NB3173,
        "ref_nb2171": REF_NB2171,
        "gain_vs_nb2171_pf_mean": round(REF_NB2171 - pf_mean, 4),
        "deploy_q50": round(deploy_q50, 4),
        "deploy_lo": round(deploy_lo, 4),
        "deploy_hi": round(deploy_hi, 4),
        "deploy_mu_tr_clipped": round(deploy_mu_tr, 4),
        "deploy_s_full_fit": round(deploy_s_full, 4),
        "deploy_s_robust": round(deploy_s_robust, 4),
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
        "gate_metric": "per_fold_mean",
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
    print(f"   pf_mean ({n_s} seeds) = {pf_mean:.4f} +/- {pf_std:.4f}")
    print(f"   pf_mean 95% CI       = [{pf_ci_low:.4f}, {pf_ci_high:.4f}]")
    print(f"   pooled mean          = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   delta vs nb3070 (pf) = {pf_mean - REF_PARENT_NB3070:+.4f}")
    print(f"   delta vs nb3190 (pf) = {pf_mean - REF_NB3190:+.4f}")
    print(f"   delta vs nb3220 (pf) = {pf_mean - REF_NB3220:+.4f}")
    print(f"   gain vs nb2171  (pf) = {REF_NB2171 - pf_mean:+.4f}")
    print(f"   verdict              = {verdict}")
    print(f"   wall                 = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "per_fold_mean_rae_mean", "per_fold_mean_rae_std",
        "per_fold_mean_rae_ci95_low", "per_fold_mean_rae_ci95_high",
        "mean_rae", "std_rae",
        "delta_vs_parent_pf_mean", "delta_vs_nb3190_pf_mean",
        "delta_vs_nb3220_pf_mean",
        "stretch_s_star_mean_all_folds",
        "deploy_q50", "deploy_lo", "deploy_hi", "deploy_s_robust",
        "n_te_clipped_lo", "n_te_clipped_hi",
        "te_unb_in_sample_rae",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
