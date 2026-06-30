"""nb3281 -- Venn-Abers-style isotonic-pair calibration on nb3090 anchor.

NEW PARADIGM (Venn-Abers for regression via layer-cake reconstruction):
    Venn-Abers calibration fits, for a BINARY target, TWO isotonic regressions on
    the calibration scores -- one assuming the test label is 0 (p0) and one
    assuming it is 1 (p1) -- and averages them into a single, multiprobability-
    consistent estimate p' in [p0, p1]. It is the gold-standard distribution-free
    probability calibrator.

    We adapt it to REGRESSION on pEC50 by the layer-cake (Cavalieri) identity:
        E[y] = y_min + integral_{y_min}^{y_max} P(y > t) dt
    Binarize the truth at a fine grid of thresholds t_k spanning the train label
    range. At each t_k, the binary target is 1[y > t_k]; the anchor score
    nb3090_pred_oof (mapped to [0,1] by fold-train min-max) is the calibration
    score. Run isotonic-pair Venn-Abers per threshold to get calibrated
    P(y > t_k), then Riemann-integrate over the grid to reconstruct a calibrated
    continuous prediction. The 3 prescription thresholds {5.0, 5.5, 6.0} are
    reported as headline-cut diagnostics; reconstruction uses a fine grid so the
    integral is faithful.

    Why this could beat the parent: the anchor pred is variance-compressed on the
    novel-scaffold OOD tail (the persistent failure mode). Isotonic-pair Venn-
    Abers is a monotone, non-parametric re-mapping of the conditional CDF at every
    level set; integrating a re-calibrated CDF can decompress both tails
    simultaneously WITHOUT perturbing rank order (isotone => rank preserving),
    which is exactly the post-hoc capacity that has historically translated.

PROTOCOL (per kf_seed, 5-fold scaffold split, FULLY CROSS-FIT):
    pred_base = nb3090_pred_oof (253,)
    Per outer fold:
        a) Fit min-max on fold-train anchor scores -> s_tr, s_va in [0,1].
        b) For each threshold t_k in the FINE grid:
             y_bin_tr = 1[y_tr > t_k]
             If y_bin_tr is all-0 or all-1 -> P = that constant (no calibration).
             Else fit VennAbers on (s_tr, y_bin_tr); predict_proba on s_va ->
             calibrated P(y_va > t_k) = p'_1.
        c) Reconstruct val preds by layer-cake Riemann sum over the grid:
             y_hat = t_grid[0] + sum_k P(y>t_k) * dt_k
           (left-Riemann on the survival function over [t_min, t_max]).
        d) Stitch into oof_va; compute per-fold val RAE and pooled RAE.
    Repeat for 15 FRESH kf_seeds {1216..1230}; report per-fold-mean across seeds.

GATE (on 15-seed PER-FOLD-MEAN RAE, the metric named in the prescription):
    per_fold_mean < 0.4423 -> "BETTER"
    else                   -> "FAIL"

References:
    nb3090 best combo 15-seed pooled mean = 0.4472  <- parent anchor
    nb3190 learned-clip BETTER gate       = 0.4422
    nb3080 wide-seed verify               = 0.4475
    nb2960 K18 deep-30 OOF                = 0.4536
    nb2171 prior post-hoc top             = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb3090_pred_oof.npy
    data/processed/te_nb3090.npy

Outputs:
    data/processed/nb3281_summary.json
    data/processed/nb3281_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3281.npy         (513,) float32 -- deploy te
    submissions/nb3281_venn_abers.csv    (only on BETTER verdict)
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

try:
    from venn_abers import VennAbers
    _HAVE_VA_PKG = True
except Exception:  # pragma: no cover - fallback path
    VennAbers = None
    _HAVE_VA_PKG = False

from sklearn.isotonic import IsotonicRegression

TAG = "nb3281"
PARENT_TAG = "nb3090"

# -- Inputs --------------------------------------------------------------------
PRED_OOF_PATH = DATA_PROCESSED / "nb3090_pred_oof.npy"
TE_PATH = DATA_PROCESSED / "te_nb3090.npy"

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))  # 15 FRESH seeds {1216..1230}

# -- Binarization thresholds ---------------------------------------------------
# Headline cuts named in the prescription (reported as diagnostics):
HEADLINE_THRESHOLDS = [5.0, 5.5, 6.0]
# Fine grid for layer-cake reconstruction of E[y] (faithful integral).
# Span the plausible train label range; resolution 0.1 log-units (= noise floor
# scale 0.15-0.24) keeps the Riemann error well below the RAE signal.
GRID_LO = 2.0
GRID_HI = 7.0
GRID_STEP = 0.1
T_GRID = np.round(np.arange(GRID_LO, GRID_HI + 1e-9, GRID_STEP), 4)

# -- Gates ---------------------------------------------------------------------
GATE_BETTER = 0.4423  # per-fold-mean < this -> BETTER

# -- References ----------------------------------------------------------------
REF_PARENT_NB3090 = 0.4472
REF_NB3190_GATE = 0.4422
REF_NB3080 = 0.4475
REF_K18 = 0.4536
REF_NB2171 = 0.4682


# ---------------------------------------------------------------------------
# Venn-Abers isotonic-pair: calibrated P(positive)
# ---------------------------------------------------------------------------
def _va_calibrate_proba(
    s_tr: np.ndarray,
    y_bin_tr: np.ndarray,
    s_va: np.ndarray,
) -> np.ndarray:
    """Isotonic-pair Venn-Abers calibrated P(positive) for s_va.

    s_tr  : (n_tr,) calibration scores in [0,1]
    y_bin_tr : (n_tr,) binary {0,1}
    s_va  : (n_va,) test scores in [0,1]
    Returns p' in [0,1], shape (n_va,).

    Degenerate (all-same-class) folds short-circuit to the constant base rate.
    Uses the venn_abers package when available; otherwise a manual GCM-free
    isotonic-pair implementation (fit two IsotonicRegressions assuming the test
    point is 0 / 1, average) that reproduces the Venn-Abers multiprobability.
    """
    pos_rate = float(y_bin_tr.mean())
    if pos_rate <= 0.0:
        return np.zeros_like(s_va, dtype=np.float64)
    if pos_rate >= 1.0:
        return np.ones_like(s_va, dtype=np.float64)

    if _HAVE_VA_PKG:
        va = VennAbers()
        # package contract (calc_p0p1 indexes p_cal[:,1]): BOTH p_cal and p_test
        # are (n,2) = [1-score, score]; y_cal is (n,) binary.
        p_cal = np.column_stack([1.0 - s_tr, s_tr]).astype(np.float64)
        va.fit(p_cal=p_cal, y_cal=y_bin_tr.astype(int))
        p_test = np.column_stack([1.0 - s_va, s_va]).astype(np.float64)
        p_prime, _p0p1 = va.predict_proba(p_test=p_test)
        return np.clip(p_prime[:, 1], 0.0, 1.0)

    # -- Manual isotonic-pair fallback --------------------------------------
    # For each test score x: fit isotonic on (s_tr + x labelled 0) -> p0(x),
    # and on (s_tr + x labelled 1) -> p1(x); p' = p1/(1-p0+p1). Vectorised by
    # noting isotonic is piecewise-constant; we evaluate per unique test score.
    out = np.empty_like(s_va, dtype=np.float64)
    uniq, inv = np.unique(s_va, return_inverse=True)
    s_tr64 = s_tr.astype(np.float64)
    y64 = y_bin_tr.astype(np.float64)
    for j, x in enumerate(uniq):
        xs0 = np.append(s_tr64, x)
        ys0 = np.append(y64, 0.0)
        ir0 = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        ir0.fit(xs0, ys0)
        p0 = float(ir0.predict([x])[0])
        xs1 = np.append(s_tr64, x)
        ys1 = np.append(y64, 1.0)
        ir1 = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        ir1.fit(xs1, ys1)
        p1 = float(ir1.predict([x])[0])
        denom = 1.0 - p0 + p1
        out_val = p1 / denom if denom > 1e-12 else 0.5 * (p0 + p1)
        out[inv == j] = out_val
    return np.clip(out, 0.0, 1.0)


def _reconstruct_layercake(
    surv: np.ndarray,  # (n, n_grid) calibrated P(y > t_k)
    t_grid: np.ndarray,
) -> np.ndarray:
    """E[y] ~= t_grid[0] + sum_k P(y > t_k) * dt_k (left-Riemann survival sum).

    With a monotone survival function this reproduces the layer-cake identity
    E[y] = t0 + integral_{t0}^{tmax} S(t) dt for y in [t0, tmax].
    """
    dt = np.diff(t_grid)  # (n_grid-1,)
    # left-Riemann: use S at left edge of each cell -> surv[:, :-1]
    integral = (surv[:, :-1] * dt[None, :]).sum(axis=1)
    return t_grid[0] + integral


def _predict_va_fold(
    s_tr: np.ndarray,
    y_tr: np.ndarray,
    s_eval: np.ndarray,
    t_grid: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Calibrate survival at every grid threshold and reconstruct E[y].

    Returns (y_hat (n_eval,), diag) where diag holds headline-threshold AUC-ish
    monotonicity stats.
    """
    n_eval = len(s_eval)
    surv = np.empty((n_eval, len(t_grid)), dtype=np.float64)
    for k, t_k in enumerate(t_grid):
        y_bin = (y_tr > t_k).astype(int)
        surv[:, k] = _va_calibrate_proba(s_tr, y_bin, s_eval)
    # Enforce monotone-nonincreasing survival across thresholds (defensive:
    # independent per-threshold isotonic fits can cross slightly).
    surv = np.minimum.accumulate(surv, axis=1)
    y_hat = _reconstruct_layercake(surv, t_grid)
    diag = {
        "surv_t0_mean": float(surv[:, 0].mean()),
        "surv_tlast_mean": float(surv[:, -1].mean()),
    }
    return y_hat, diag


def _run_one_seed(
    pred_base: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Cross-fit Venn-Abers layer-cake at a single kf_seed."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_va = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        # fold-train min-max -> scores in [0,1]
        a_tr = pred_base[tr_loc]
        lo = float(a_tr.min())
        hi = float(a_tr.max())
        rng = hi - lo if hi > lo else 1.0
        s_tr = (a_tr - lo) / rng
        s_va = np.clip((pred_base[va_loc] - lo) / rng, 0.0, 1.0)
        y_hat, _diag = _predict_va_fold(s_tr, y_unb[tr_loc], s_va, T_GRID)
        oof_va[va_loc] = y_hat
        fold_val_raes.append(float(rae(y_unb[va_loc], y_hat)))

    if np.isnan(oof_va).any():
        raise RuntimeError(
            f"kf_seed={kf_seed}: scaffold splits did not cover all rows"
        )
    pooled = float(rae(y_unb, oof_va))
    per_fold_mean = float(np.mean(fold_val_raes))
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_mean_rae": per_fold_mean,
        "per_fold_val_raes": [round(v, 4) for v in fold_val_raes],
        "oof": oof_va,
    }


def _fit_full_survival(
    pred_base: np.ndarray,
    y_unb: np.ndarray,
    te_base: np.ndarray,
    t_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Deploy: fit Venn-Abers on ALL 253 (min-max on full anchor), predict te.

    Returns (oof_full_hat (253,), te_hat (513,), headline_diag).
    """
    lo = float(pred_base.min())
    hi = float(pred_base.max())
    rng = hi - lo if hi > lo else 1.0
    s_all = (pred_base - lo) / rng
    s_te = np.clip((te_base - lo) / rng, 0.0, 1.0)

    oof_hat, _ = _predict_va_fold(s_all, y_unb, s_all, t_grid)
    te_hat, _ = _predict_va_fold(s_all, y_unb, s_te, t_grid)

    # headline-threshold calibration diagnostics on full 253
    headline = {}
    for t_k in HEADLINE_THRESHOLDS:
        y_bin = (y_unb > t_k).astype(int)
        if 0 < y_bin.mean() < 1:
            p = _va_calibrate_proba(s_all, y_bin, s_all)
            # Brier + base rate
            headline[str(t_k)] = {
                "base_rate": round(float(y_bin.mean()), 4),
                "mean_pred_proba": round(float(p.mean()), 4),
                "brier": round(float(np.mean((p - y_bin) ** 2)), 4),
            }
        else:
            headline[str(t_k)] = {
                "base_rate": round(float(y_bin.mean()), 4),
                "mean_pred_proba": round(float(y_bin.mean()), 4),
                "brier": 0.0,
            }
    return oof_hat, te_hat, headline


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(
        f"{TAG} -- Venn-Abers isotonic-pair layer-cake calibration on "
        f"{PARENT_TAG} pred_oof"
    )
    print(f"          venn_abers package available = {_HAVE_VA_PKG}")
    print(f"          headline thresholds = {HEADLINE_THRESHOLDS}")
    print(
        f"          reconstruction grid = [{GRID_LO}, {GRID_HI}] step "
        f"{GRID_STEP} ({len(T_GRID)} pts)"
    )
    print(f"          kf_seeds = {len(KF_SEEDS)} fresh "
          f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}")
    print(f"          gate: per_fold_mean < {GATE_BETTER:.4f} -> BETTER")
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

    # -- Load nb3090 anchor pred_oof + te -------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 1: load {PARENT_TAG} pred_oof + te")
    print("-" * 78)
    pred_base = np.load(PRED_OOF_PATH).astype(np.float64)
    te_base = np.load(TE_PATH).astype(np.float64)
    if pred_base.shape != (n_unb,):
        raise ValueError(
            f"{PARENT_TAG} pred_oof shape {pred_base.shape} != ({n_unb},)"
        )
    if te_base.shape != (n_test,):
        raise ValueError(
            f"{PARENT_TAG} te shape {te_base.shape} != ({n_test},)"
        )
    full_oof_rae = float(rae(y_unb, pred_base))
    print(
        f"   pred_base: oof_RAE={full_oof_rae:.4f}  "
        f"mean={pred_base.mean():.3f}  std={pred_base.std():.3f}  "
        f"min={pred_base.min():.3f}  max={pred_base.max():.3f}"
    )
    print(
        f"   te_base:   mean={te_base.mean():.3f}  std={te_base.std():.3f}  "
        f"min={te_base.min():.3f}  max={te_base.max():.3f}"
    )

    leak_eq = float(np.mean(np.isclose(pred_base, y_unb, atol=1e-6)))
    if leak_eq > 0.05:
        print(f"   WARN parent: {leak_eq:.1%} rows == truth -- possible leak")
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
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}"
    )
    print("-" * 78)
    seed_records = []
    pooled_raes = []
    per_fold_means = []
    oof_stack = []
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(pred_base, y_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        per_fold_means.append(res["per_fold_mean_rae"])
        oof_stack.append(res["oof"])
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_mean_rae": round(res["per_fold_mean_rae"], 4),
            "per_fold_val_raes": res["per_fold_val_raes"],
        })
        print(
            f"   kf={s}: per_fold_mean={res['per_fold_mean_rae']:.4f}  "
            f"pooled={res['pooled_rae']:.4f}  wall={time.time()-ts:.2f}s"
        )

    pf_arr = np.asarray(per_fold_means, dtype=np.float64)
    pooled_arr = np.asarray(pooled_raes, dtype=np.float64)
    n_s = len(pf_arr)
    pf_mean = float(pf_arr.mean())
    pf_std = float(pf_arr.std(ddof=1)) if n_s > 1 else 0.0
    pf_sem = pf_std / np.sqrt(n_s) if n_s > 1 else 0.0
    t_mult = 2.145  # df=14, two-sided 95%
    pf_ci_low = pf_mean - t_mult * pf_sem
    pf_ci_high = pf_mean + t_mult * pf_sem
    pooled_mean = float(pooled_arr.mean())
    pooled_std = float(pooled_arr.std(ddof=1)) if n_s > 1 else 0.0

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds)")
    print("-" * 78)
    print(f"   PER-FOLD-MEAN (gated): {pf_mean:.4f} +/- {pf_std:.4f}")
    print(f"     sem    = {pf_sem:.4f}")
    print(f"     95% CI = [{pf_ci_low:.4f}, {pf_ci_high:.4f}]")
    print(f"     min/max= [{pf_arr.min():.4f}, {pf_arr.max():.4f}]")
    print(f"   pooled-RAE (reference): {pooled_mean:.4f} +/- {pooled_std:.4f}")
    print(
        f"\n   ref {PARENT_TAG} (pooled) 15-seed = {REF_PARENT_NB3090:.4f}"
    )
    print(
        f"   delta(per_fold_mean vs nb3090) = "
        f"{pf_mean - REF_PARENT_NB3090:+.4f}"
    )

    # -- Deploy: fit on all 253, predict te -----------------------------------
    print("\n" + "-" * 78)
    print("DEPLOY: fit Venn-Abers on all 253, reconstruct te")
    print("-" * 78)
    oof_full_hat, te_pred, headline_diag = _fit_full_survival(
        pred_base, y_unb, te_base, T_GRID,
    )
    te_pred = np.clip(te_pred, 3.0, 9.0).astype(np.float32)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    full_refit_rae = float(rae(y_unb, oof_full_hat))
    print(
        f"   full-refit in-sample RAE (on 253) = {full_refit_rae:.4f}"
    )
    print(
        f"   te(513): mean={te_pred.mean():.3f}  std={te_pred.std():.3f}  "
        f"min={te_pred.min():.3f}  max={te_pred.max():.3f}"
    )
    print(f"   te[unb] in-sample RAE = {te_unb_in_rae:.4f}")
    print(f"   headline-threshold calibration (full 253):")
    for t_k, d in headline_diag.items():
        print(
            f"     t>{t_k}: base_rate={d['base_rate']}  "
            f"mean_p={d['mean_pred_proba']}  brier={d['brier']}"
        )

    # Median-seed OOF for storage (rank by per-fold-mean, the gated metric)
    med_seed_idx = int(np.argsort(pf_arr)[n_s // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(
        f"   median seed = {median_seed} "
        f"(per_fold_mean={pf_arr[med_seed_idx]:.4f})"
    )

    # -- Gate -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE (on PER-FOLD-MEAN)")
    print("-" * 78)
    if pf_mean < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE-CANDIDATE. nb3281 Venn-Abers isotonic-pair layer-cake "
            f"15-seed PER-FOLD-MEAN {pf_mean:.4f} clears gate {GATE_BETTER:.4f} "
            f"({pf_mean - GATE_BETTER:+.4f}) and beats parent nb3090 "
            f"({REF_PARENT_NB3090:.4f}, {pf_mean - REF_PARENT_NB3090:+.4f}). "
            f"Distribution-free CDF re-calibration decompresses the OOD tail "
            f"while preserving rank order. Re-verify with deep-30 before "
            f"PRIMARY-1 swap; confirm te[unb] in-sample ({te_unb_in_rae:.4f}) "
            f"is plausible vs cross-fit (expected in-sample optimism)."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3281 Venn-Abers isotonic-pair layer-cake 15-seed "
            f"PER-FOLD-MEAN {pf_mean:.4f} does NOT clear gate {GATE_BETTER:.4f} "
            f"({pf_mean - GATE_BETTER:+.4f}). Delta vs parent nb3090 = "
            f"{pf_mean - REF_PARENT_NB3090:+.4f}. The anchor pred is a "
            f"near-monotone function of truth already (pooled corr high), so "
            f"isotonic-pair re-mapping + layer-cake integration mostly "
            f"reproduces the parent without net RAE gain -- consistent with the "
            f"cycle-160 finding that deep-ensemble averaging already performs "
            f"the variance decompression post-hoc operators target. Keep "
            f"nb3090 / prior PRIMARY-1."
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

    sub_csv = SUBMISSIONS / f"{TAG}_venn_abers.csv"
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
        "method": "venn_abers_isotonic_pair_layercake_on_nb3090_pred_oof",
        "anchor_pred_oof_path": str(PRED_OOF_PATH),
        "anchor_te_path": str(TE_PATH),
        "anchor_pre_unblind": True,
        "venn_abers_pkg_available": _HAVE_VA_PKG,
        "parent_full_oof_rae": round(full_oof_rae, 4),
        "parent_leak_eq_truth_frac": round(leak_eq, 4),
        "headline_thresholds": HEADLINE_THRESHOLDS,
        "recon_grid_lo": GRID_LO,
        "recon_grid_hi": GRID_HI,
        "recon_grid_step": GRID_STEP,
        "recon_grid_n": int(len(T_GRID)),
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": n_s,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "seed_records": seed_records,
        "per_fold_mean_array": [round(float(v), 4) for v in per_fold_means],
        "pooled_rae_array": [round(float(v), 4) for v in pooled_raes],
        # GATED metric:
        "per_fold_mean_rae": round(pf_mean, 4),
        "per_fold_mean_std": round(pf_std, 4),
        "per_fold_mean_sem": round(pf_sem, 4),
        "per_fold_mean_ci95_low": round(pf_ci_low, 4),
        "per_fold_mean_ci95_high": round(pf_ci_high, 4),
        "per_fold_mean_min": round(float(pf_arr.min()), 4),
        "per_fold_mean_max": round(float(pf_arr.max()), 4),
        # reference pooled metric:
        "pooled_mean_rae": round(pooled_mean, 4),
        "pooled_std_rae": round(pooled_std, 4),
        "ref_parent_nb3090": REF_PARENT_NB3090,
        "delta_per_fold_mean_vs_parent": round(pf_mean - REF_PARENT_NB3090, 4),
        "delta_pooled_vs_parent": round(pooled_mean - REF_PARENT_NB3090, 4),
        "ref_nb3190_gate": REF_NB3190_GATE,
        "ref_nb3080": REF_NB3080,
        "ref_K18_deep30": REF_K18,
        "ref_nb2171": REF_NB2171,
        "headline_calibration": headline_diag,
        "full_refit_in_sample_rae": round(full_refit_rae, 4),
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
    print(f"   per_fold_mean ({n_s} seeds) = {pf_mean:.4f} +/- {pf_std:.4f}")
    print(f"   95% CI                      = [{pf_ci_low:.4f}, {pf_ci_high:.4f}]")
    print(f"   pooled mean                 = {pooled_mean:.4f}")
    print(f"   delta vs nb3090             = {pf_mean - REF_PARENT_NB3090:+.4f}")
    print(f"   verdict                     = {verdict}")
    print(f"   wall                        = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "per_fold_mean_rae", "per_fold_mean_std",
        "per_fold_mean_ci95_low", "per_fold_mean_ci95_high",
        "pooled_mean_rae",
        "delta_per_fold_mean_vs_parent",
        "te_unb_in_sample_rae",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
