"""nb3162 -- K18-ONLY scalar stretch then nb3080 quantile blend.

REFINED PARADIGM (from nb3094 ablation):
    nb3094 applied per-fold golden-section stretch to BOTH K18 and K19
    independently before the nb3080 best-combo quantile-conditional blend.
    Here we test the K18-ONLY variant: stretch K18 only (s_K18 found per fold
    by golden-section on fold-train RAE), pass K19 through unchanged, then
    apply the nb3080 hard-split blend.

    Rationale: K18 is the GATING anchor in the nb3080 blend (the per-row
    quantile decision uses K18 only). Stretching K18 before the gating
    quantile is computed reshapes WHICH rows fall into the low vs high
    region, while leaving K19 raw. K19 receives modest weights (0.1 in low
    region, 0.5 in high region) -- its stretch in nb3094 may have been
    over-fitting the per-fold train RAE without conferring rank gain on val.

PROTOCOL (per kf_seed, 5-fold scaffold split):
    For each outer fold (tr_loc, va_loc):
        mu_K18 = mean(K18_pred[tr_loc])
        s_K18 = argmin_{s in [0.95, 1.20]} RAE(y_tr,
                    _stretch(K18_pred[tr_loc], mu_K18, s))
        K18_s_val = _stretch(K18_pred[va_loc], mu_K18, s_K18)
        K19_val   = K19_pred[va_loc]                   # UNCHANGED
        q_thr = quantile(_stretch(K18_pred[tr_loc], mu_K18, s_K18), q_cut=0.4)
        blend per-row:
            K18_s_val <= q_thr -> 0.9 * K18_s_val + 0.1 * K19_val
            K18_s_val >  q_thr -> 0.5 * K18_s_val + 0.5 * K19_val
        oof[va_loc] = blend
    pooled_rae = rae(y_unb, oof)
    Repeat for 15 FRESH kf_seeds {1141..1155}; report mean +/- std + 95% CI.

GATE (on 15-seed mean):
    mean < 0.4475 -> "BETTER"  (new PRIMARY-1 candidate)
    else          -> "FAIL"

References:
    nb3080 15-seed wide-mean (CEILING)        = 0.4475 +/- 0.0006
    nb3094 K18+K19 stretch (paired ablation)  = anchor for this experiment
    nb2960 K18 deep-30 OOF                    = 0.4536
    nb3000 K19 deep-30 OOF                    = 0.4607
    nb2171 prior post-hoc top                 = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb2960_K18_30seed_oof.npy
    data/processed/nb2960_K18_30seed_te.npy
    data/processed/nb3000_K19_30seed_oof.npy
    data/processed/te_nb3000_K19.npy

Outputs:
    data/processed/nb3162_summary.json
    data/processed/nb3162_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3162.npy         (513,) float32 -- deploy te
    submissions/nb3162_K18_stretch_in_nb3080.csv  (only on PROMOTE verdict)
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

TAG = "nb3162"
PARENT_TAG = "nb3080"

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
KF_SEEDS = list(range(1141, 1156))  # 15 FRESH seeds {1141..1155}

# -- Quantile-conditional blend (nb3080 best combo, FIXED) ---------------------
Q_CUT = 0.4
W_K18_LOW = 0.9   # w_K19_low = 0.1
W_K18_HIGH = 0.5  # w_K19_high = 0.5
W_K19_LOW = 1.0 - W_K18_LOW
W_K19_HIGH = 1.0 - W_K18_HIGH

# -- K18-only stretch search ---------------------------------------------------
S_LO = 0.95
S_HI = 1.20
GS_TOL = 1e-4
GS_MAX_ITER = 60

# -- Gates ---------------------------------------------------------------------
GATE_BETTER = 0.4475  # mean < this -> BETTER

# -- References ----------------------------------------------------------------
REF_NB3080 = 0.4475
REF_NB3080_STD = 0.0006
REF_NB3094 = None  # Compare against nb3094 in summary if available
REF_NB3030 = 0.4509
REF_K18 = 0.4536
REF_K19 = 0.4607
REF_NB2171 = 0.4682


def _stretch(pred: np.ndarray, mu: float, s: float) -> np.ndarray:
    """Scalar rank-stretch about a fixed center: mu + s * (pred - mu)."""
    return mu + s * (pred - mu)


def golden_section_min(f, lo: float, hi: float,
                       tol: float = GS_TOL,
                       max_iter: int = GS_MAX_ITER) -> tuple[float, float]:
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


def _blend_quantile_conditional(
    p_k18_s: np.ndarray,
    p_k19: np.ndarray,
    q_thr: float,
) -> np.ndarray:
    """nb3080 best-combo per-row hard-split blend.

    K18 is STRETCHED (s_K18 from fold-train golden-section).
    K19 is RAW (no stretch applied).
    Gating uses stretched K18 vs q_thr (also derived from stretched K18).

    rows with p_k18_s <= q_thr -> (W_K18_LOW=0.9, W_K19_LOW=0.1)
    rows with p_k18_s >  q_thr -> (W_K18_HIGH=0.5, W_K19_HIGH=0.5)
    """
    low_mask = p_k18_s <= q_thr
    out = np.empty_like(p_k18_s, dtype=np.float64)
    out[low_mask] = (
        W_K18_LOW * p_k18_s[low_mask] + W_K19_LOW * p_k19[low_mask]
    )
    out[~low_mask] = (
        W_K18_HIGH * p_k18_s[~low_mask] + W_K19_HIGH * p_k19[~low_mask]
    )
    return out


def _run_one_seed(
    P_unb: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """One scaffold-CV pass: per-fold K18-only stretch then quantile blend."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_s_K18 = []
    fold_mu_K18 = []
    fold_q_thrs = []
    fold_high_share = []
    fold_train_raes_s = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        p_k18_tr = P_unb[tr_loc, 0]
        p_k19_tr = P_unb[tr_loc, 1]
        y_tr = y_unb[tr_loc]

        mu_K18 = float(np.mean(p_k18_tr))

        # K18-only golden-section stretch on fold-train RAE
        def f_K18(s, p=p_k18_tr, y=y_tr, mu=mu_K18):
            return float(rae(y, _stretch(p, mu, s)))

        s_K18, fold_train_rae_K18 = golden_section_min(f_K18, S_LO, S_HI)

        # Apply stretch to K18 only; K19 stays raw
        p_k18_tr_s = _stretch(p_k18_tr, mu_K18, s_K18)
        p_k18_va_s = _stretch(P_unb[va_loc, 0], mu_K18, s_K18)
        p_k19_va = P_unb[va_loc, 1]  # UNCHANGED

        # nb3080 quantile threshold on STRETCHED fold-train K18
        q_thr = float(np.quantile(p_k18_tr_s, Q_CUT))

        # Per-row hard-split blend: stretched K18 + raw K19
        val_pred = _blend_quantile_conditional(p_k18_va_s, p_k19_va, q_thr)
        oof_blend[va_loc] = val_pred

        fold_val_raes.append(float(rae(y_unb[va_loc], val_pred)))
        fold_s_K18.append(s_K18)
        fold_mu_K18.append(mu_K18)
        fold_q_thrs.append(q_thr)
        fold_high_share.append(float(np.mean(p_k18_va_s > q_thr)))
        fold_train_raes_s.append({
            "K18": float(fold_train_rae_K18),
        })

    if np.isnan(oof_blend).any():
        raise RuntimeError(
            f"kf_seed={kf_seed}: scaffold splits did not cover all rows"
        )
    pooled = float(rae(y_unb, oof_blend))
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(fold_val_raes)),
        "per_fold_val_rae_std": float(np.std(fold_val_raes, ddof=1)),
        "fold_s_K18": [round(s, 4) for s in fold_s_K18],
        "fold_s_K18_mean": float(np.mean(fold_s_K18)),
        "fold_mu_K18_mean": float(np.mean(fold_mu_K18)),
        "fold_q_thr_mean": float(np.mean(fold_q_thrs)),
        "fold_q_thr_std": float(np.std(fold_q_thrs, ddof=1)),
        "fold_high_share_mean": float(np.mean(fold_high_share)),
        "fold_train_raes_s": fold_train_raes_s,
        "oof": oof_blend,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(
        f"{TAG} -- K18-ONLY scalar STRETCH then nb3080 quantile blend on "
        f"{K_LABELS} deep-30"
    )
    print(
        f"          stretch range : s_K18 in [{S_LO}, {S_HI}] golden-section "
        f"per fold (K19 RAW)"
    )
    print(
        f"          blend (FIXED) : q_cut={Q_CUT}, "
        f"low (K18_s<=q): (K18={W_K18_LOW}, K19={W_K19_LOW}), "
        f"high (K18_s>q): (K18={W_K18_HIGH}, K19={W_K19_HIGH})"
    )
    print(
        f"          kf_seeds      : {len(KF_SEEDS)} fresh "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}"
    )
    print(
        f"          parent ref    : {PARENT_TAG} 15-seed mean = "
        f"{REF_NB3080:.4f} +/- {REF_NB3080_STD:.4f}"
    )
    print(
        f"          gate          : mean < {GATE_BETTER:.4f} -> BETTER"
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

    # -- Load K18, K19 deep-30 anchor OOFs + te arrays ------------------------
    print("\n" + "-" * 78)
    print("STEP 1: load K18, K19 deep-30 OOFs and te arrays")
    print("-" * 78)
    oof_cols, te_cols = [], []
    per_K_full_rae = {}
    per_K_compression = {}
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
        compression = float(oof.std() / y_unb.std())
        per_K_compression[k] = round(compression, 4)
        print(
            f"   {k} ({K_DEPTH[k]:>6s}): oof_RAE = {r:.4f}  "
            f"oof_mean={oof.mean():.3f}  oof_std={oof.std():.3f}  "
            f"compression(pred/truth)={compression:.3f}  "
            f"te_mean={te_arr.mean():.3f}  te_std={te_arr.std():.3f}"
        )

    P_unb = np.column_stack(oof_cols)  # (253, 2)
    P_te = np.column_stack(te_cols)    # (513, 2)
    truth_std = float(y_unb.std())
    print(f"   truth_std = {truth_std:.3f}")

    # Leak sanity
    leak_flags = {}
    for i, k in enumerate(K_LABELS):
        frac = float(np.mean(np.isclose(P_unb[:, i], y_unb, atol=1e-6)))
        leak_flags[k] = round(frac, 4)
        if frac > 0.05:
            print(f"   WARN {k}: {frac:.1%} rows == truth -- possible leak")

    corr = float(np.corrcoef(P_unb.T)[0, 1])
    print(f"   pairwise corr({K_LABELS[0]}, {K_LABELS[1]}) = {corr:.4f}")

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
    oof_stack = []
    all_s_K18 = []
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(P_unb, y_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        oof_stack.append(res["oof"])
        all_s_K18.extend(res["fold_s_K18"])
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
            "per_fold_val_rae_std": round(res["per_fold_val_rae_std"], 4),
            "fold_s_K18": res["fold_s_K18"],
            "fold_s_K18_mean": round(res["fold_s_K18_mean"], 4),
            "fold_q_thr_mean": round(res["fold_q_thr_mean"], 4),
            "fold_q_thr_std": round(res["fold_q_thr_std"], 4),
            "fold_high_share_mean": round(res["fold_high_share_mean"], 4),
        })
        print(
            f"   kf={s}: pooled_RAE={res['pooled_rae']:.4f}  "
            f"s_K18={res['fold_s_K18_mean']:.3f}  "
            f"q_thr={res['fold_q_thr_mean']:.3f}  "
            f"high_share={res['fold_high_share_mean']:.2f}  "
            f"wall={time.time()-ts:.2f}s"
        )

    arr = np.asarray(pooled_raes, dtype=np.float64)
    n_s = len(arr)
    mean_rae = float(arr.mean())
    std_rae = float(arr.std(ddof=1)) if n_s > 1 else 0.0
    sem = std_rae / np.sqrt(n_s) if n_s > 1 else 0.0
    # 95% CI via t-multiplier (n=15, df=14, t~2.145)
    t_mult = 2.145
    ci_low = mean_rae - t_mult * sem
    ci_high = mean_rae + t_mult * sem
    median_rae = float(np.median(arr))

    s_K18_arr = np.asarray(all_s_K18, dtype=np.float64)
    s_K18_mean = float(s_K18_arr.mean())
    s_K18_std = float(s_K18_arr.std(ddof=1))

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds)")
    print("-" * 78)
    print(f"   mean    = {mean_rae:.4f}")
    print(f"   std     = {std_rae:.4f}")
    print(f"   sem     = {sem:.4f}")
    print(f"   95% CI  = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   median  = {median_rae:.4f}")
    print(f"   min/max = [{arr.min():.4f}, {arr.max():.4f}]")
    print(
        f"   s_K18 mean={s_K18_mean:.4f} +/- {s_K18_std:.4f}  "
        f"(n={len(s_K18_arr)})"
    )
    print(
        f"\n   ref {PARENT_TAG} 15-seed mean      = "
        f"{REF_NB3080:.4f} +/- {REF_NB3080_STD:.4f}"
    )
    print(f"   delta vs {PARENT_TAG}              = {mean_rae - REF_NB3080:+.4f}")
    print(f"   delta vs nb3030 wide ceiling     = {mean_rae - REF_NB3030:+.4f}")

    # -- Deploy: full-data K18-only stretch + blend on 513 -------------------
    mu_K18_full = float(P_unb[:, 0].mean())
    s_K18_deploy = s_K18_mean

    te_K18_s = _stretch(P_te[:, 0], mu_K18_full, s_K18_deploy)
    te_K19_raw = P_te[:, 1]  # K19 unchanged
    # Deploy quantile threshold from full 253 stretched K18 OOF
    p_k18_full_s = _stretch(P_unb[:, 0], mu_K18_full, s_K18_deploy)
    deploy_q_thr = float(np.quantile(p_k18_full_s, Q_CUT))
    te_pred = _blend_quantile_conditional(
        te_K18_s, te_K19_raw, deploy_q_thr,
    ).astype(np.float32)
    te_pred = np.clip(te_pred, 3.0, 9.0)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    te_low_share = float(np.mean(te_K18_s <= deploy_q_thr))
    print(f"\n   deploy mu_K18    = {mu_K18_full:.4f}")
    print(f"   deploy s_K18     = {s_K18_deploy:.4f}  (K19 raw)")
    print(f"   deploy q_thr     = {deploy_q_thr:.4f}")
    print(f"   te(513) low-half share = {te_low_share:.3f}")
    print(f"   te(513) mean={te_pred.mean():.3f}  std={te_pred.std():.3f}")
    print(f"   te[unb] in-sample RAE  = {te_unb_in_rae:.4f}")

    # Median-seed OOF for storage
    med_seed_idx = int(np.argsort(arr)[n_s // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(f"   median seed = {median_seed} (rae={arr[med_seed_idx]:.4f})")

    # -- Gate -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    if mean_rae < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE. nb3162 wide-seed 15-mean {mean_rae:.4f} beats "
            f"{PARENT_TAG} {REF_NB3080:.4f} by {mean_rae - REF_NB3080:+.4f}. "
            f"K18-only stretch (s={s_K18_mean:.3f}) with K19 raw before "
            f"the nb3080 quantile-conditional blend extracts gain over the "
            f"nb3094 paired-stretch and nb3080 raw-anchor baselines. "
            "New PRIMARY-1 candidate."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3162 wide-seed 15-mean {mean_rae:.4f} does NOT beat "
            f"{PARENT_TAG} ceiling {REF_NB3080:.4f} "
            f"(delta {mean_rae - REF_NB3080:+.4f}). K18-only pre-stretch "
            f"before the quantile-conditional blend offers no gain over "
            f"raw-anchor blending; the quantile threshold and blend weights "
            f"already adapt to per-region scale. Keep nb3080 PRIMARY-1."
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

    sub_csv = SUBMISSIONS / f"{TAG}_K18_stretch_in_nb3080.csv"
    promote_verdicts = {"BETTER"}
    if verdict in promote_verdicts:
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
            "K18_only_golden_section_stretch_then_quantile_conditional_blend_"
            "K18_K19_deep30"
        ),
        "paradigm": (
            "per_fold_K18_only_scalar_stretch_K19_raw_BEFORE_"
            "nb3080_best_combo_quantile_blend"
        ),
        "anchor_pool": K_LABELS,
        "anchor_depth": K_DEPTH,
        "anchor_pre_unblind": True,
        "per_K_full_oof_rae": per_K_full_rae,
        "per_K_compression_ratio": per_K_compression,
        "anchor_leak_eq_truth_frac": leak_flags,
        "oof_pairwise_corr": round(corr, 4),
        "truth_std": round(truth_std, 4),
        "blend_combo": {
            "q_cut": Q_CUT,
            "w_K18_low": W_K18_LOW,
            "w_K19_low": W_K19_LOW,
            "w_K18_high": W_K18_HIGH,
            "w_K19_high": W_K19_HIGH,
        },
        "stretch_search": {
            "s_lo": S_LO,
            "s_hi": S_HI,
            "gs_tol": GS_TOL,
            "gs_max_iter": GS_MAX_ITER,
            "applied_to": "K18_only_K19_raw",
        },
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": n_s,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "seed_records": seed_records,
        "pooled_rae_array": [round(float(v), 4) for v in pooled_raes],
        "mean_rae": round(mean_rae, 4),
        "std_rae": round(std_rae, 4),
        "sem_rae": round(sem, 4),
        "ci95_low": round(ci_low, 4),
        "ci95_high": round(ci_high, 4),
        "median_rae": round(median_rae, 4),
        "min_rae": round(float(arr.min()), 4),
        "max_rae": round(float(arr.max()), 4),
        "s_K18_mean": round(s_K18_mean, 4),
        "s_K18_std": round(s_K18_std, 4),
        "s_n_samples": int(len(s_K18_arr)),
        "ref_nb3080": REF_NB3080,
        "ref_nb3080_std": REF_NB3080_STD,
        "ref_nb3030": REF_NB3030,
        "ref_K18_deep30": REF_K18,
        "ref_K19_deep30": REF_K19,
        "ref_nb2171": REF_NB2171,
        "delta_vs_nb3080": round(mean_rae - REF_NB3080, 4),
        "delta_vs_nb3030": round(mean_rae - REF_NB3030, 4),
        "deploy_mu_K18": round(mu_K18_full, 4),
        "deploy_s_K18": round(s_K18_deploy, 4),
        "deploy_q_thr": round(deploy_q_thr, 4),
        "te_low_share": round(te_low_share, 4),
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "median_seed": int(median_seed),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": (
            str(sub_csv) if verdict in promote_verdicts else None
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
    print(f"   mean_rae ({n_s} seeds)   = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   95% CI                = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   delta vs nb3080       = {mean_rae - REF_NB3080:+.4f}")
    print(f"   delta vs nb3030       = {mean_rae - REF_NB3030:+.4f}")
    print(f"   s_K18 mean            = {s_K18_mean:.4f} +/- {s_K18_std:.4f}")
    print(f"   verdict               = {verdict}")
    print(f"   wall                  = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae", "std_rae", "ci95_low", "ci95_high",
        "delta_vs_nb3080", "delta_vs_nb3030",
        "s_K18_mean",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
    print(f"  per_K_full_oof_rae: {res.get('per_K_full_oof_rae')}")
    print(f"  per_K_compression_ratio: {res.get('per_K_compression_ratio')}")
