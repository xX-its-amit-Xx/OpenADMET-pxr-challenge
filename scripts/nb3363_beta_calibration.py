"""nb3363 -- 3-parameter Beta calibration curve on nb3090.

NEW PARADIGM: a smooth parametric Beta calibration map (a, b, c) fit by
minimizing fold-train RAE.

    Prior post-hoc families on the nb3090 q35-quantile-conditional anchor:
      - hard clip      (nb3173/nb3190/nb3220) -- discontinuous, breaks tail rank
      - sigmoid soft-clip (nb3234)            -- symmetric tanh, 2 fixed knots
      - learned clip   (nb3190)               -- 2 learned thresholds
    nb3363 substitutes the classical Kull-et-al. Beta calibration map, a
    3-parameter (a, b, c) monotone family that subsumes the logistic/identity
    and adds independent low-tail (a) and high-tail (b) curvature plus a global
    shift (c). It is the smooth parametric correction with the richest shape
    among 3-DOF monotone maps, fit DIRECTLY to fold-train RAE (not log-loss).

    Beta map (operates on a [0,1]-normalized prediction p):
        g(p) = 1 / (1 + 1 / (exp(c) * p^a / (1-p)^b))
             = sigmoid( c + a*ln(p) - b*ln(1-p) )
      a, b >= 0 enforce monotonicity (so the map is a proper calibration curve).
      a = b = 1, c = 0  ->  g(p) = p  (identity).
      a controls low-tail steepness, b controls high-tail steepness, c the shift.

    Normalization (task-specified, pEC50 domain assumed ~[3, 8]):
        p   = (pred - 3) / (8 - 3)          # -> [0,1] (clipped to (eps, 1-eps))
        g   = beta_map(p; a, b, c)          # -> (0,1)
        out = g * (8 - 3) + 3               # -> back to pEC50

PROTOCOL (per kf_seed, 5-fold scaffold split):
    pred_base = nb3090_pred_oof  (253,)  -- q35-quantile-conditional blend
    Per outer fold:
        a) Normalize fold-train preds to p_tr in (eps, 1-eps).
        b) Fit (a, b, c) by scipy.minimize (Nelder-Mead) of
           rae(y[fold_train], beta_map(p_tr; a,b,c) -> pEC50)
           starting from identity (a=1, b=1, c=0); a,b clamped >= 0 inside obj.
        c) Apply the fitted map to fold-val preds; stitch into oof_beta.
           Record per-fold val RAE.
    Repeat for 15 FRESH kf_seeds {1216..1230}; report PER-FOLD-MEAN.

GATE (on 15-seed PER-FOLD-MEAN):
    pf_mean < 0.4423 -> "BETTER"
    else             -> "FAIL"

References:
    nb3090 q35 fine-cut blend            = 0.4472  <- parent anchor
    nb3190 learned hard-clip on nb3090   = 0.4422  (clip-operator target)
    nb3234 sigmoid soft-clip on nb3090   = (smooth-clip sibling)
    nb2171 prior post-hoc PRIMARY-1      = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb3090_pred_oof.npy
    data/processed/te_nb3090.npy

Outputs:
    data/processed/nb3363_summary.json
    data/processed/nb3363_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3363.npy         (513,) float32 -- deploy te
    submissions/nb3363_beta_calibration.csv  (only on BETTER)
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
from scipy.optimize import minimize

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3363"
PARENT_TAG = "nb3090"

# -- Inputs --------------------------------------------------------------------
PRED_OOF_PATH = DATA_PROCESSED / "nb3090_pred_oof.npy"
TE_PATH = DATA_PROCESSED / "te_nb3090.npy"

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))  # 15 FRESH seeds {1216..1230}

# -- Beta calibration normalization domain (pEC50 -> [0,1]) --------------------
P_LO = 3.0
P_HI = 8.0
EPS = 1e-4  # clip p into (EPS, 1-EPS) to keep ln finite

# -- Gates ---------------------------------------------------------------------
GATE_BETTER = 0.4423     # per-fold-mean < this -> BETTER

# -- References ----------------------------------------------------------------
REF_PARENT_NB3090 = 0.4472
REF_NB3190 = 0.4422
REF_NB2171 = 0.4682


def _normalize(pred: np.ndarray) -> np.ndarray:
    """pEC50 -> p in (EPS, 1-EPS) via (pred - P_LO)/(P_HI - P_LO)."""
    p = (pred - P_LO) / (P_HI - P_LO)
    return np.clip(p, EPS, 1.0 - EPS)


def _beta_map(p: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Kull-et-al. Beta calibration map on p in (0,1).

    g(p) = sigmoid(c + a*ln(p) - b*ln(1-p))
         = 1 / (1 + 1 / (exp(c) * p^a / (1-p)^b))

    Computed in logit space for numerical stability. a, b are clamped to be
    >= 0 by the caller's objective so the map stays monotone.
    """
    z = c + a * np.log(p) - b * np.log(1.0 - p)
    # numerically stable sigmoid
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _denormalize(g: np.ndarray) -> np.ndarray:
    """[0,1] beta output -> pEC50."""
    return g * (P_HI - P_LO) + P_LO


def _apply_beta(pred: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Full pipeline: pEC50 -> normalize -> beta_map -> pEC50."""
    p = _normalize(pred)
    g = _beta_map(p, a, b, c)
    return _denormalize(g)


def _fit_beta(
    pred_tr: np.ndarray,
    y_tr: np.ndarray,
) -> tuple[float, float, float, float, bool]:
    """Fit (a, b, c) minimizing fold-train RAE. Returns (a, b, c, train_rae, ok).

    Starts from identity (a=1, b=1, c=0). a, b clamped >= 0 inside the
    objective (softplus-free hard clamp) so the calibration map is monotone.
    Nelder-Mead is used because the RAE objective is non-smooth (abs value).
    """
    p_tr = _normalize(pred_tr)

    def obj(theta: np.ndarray) -> float:
        a = max(theta[0], 0.0)
        b = max(theta[1], 0.0)
        c = theta[2]
        g = _beta_map(p_tr, a, b, c)
        pec = _denormalize(g)
        return rae(y_tr, pec)

    x0 = np.array([1.0, 1.0, 0.0], dtype=np.float64)
    res = minimize(
        obj, x0, method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-7},
    )
    a = max(float(res.x[0]), 0.0)
    b = max(float(res.x[1]), 0.0)
    c = float(res.x[2])
    train_rae = float(res.fun)
    # Guard: never let the fit be worse than identity on fold-train.
    ident_rae = rae(y_tr, _denormalize(_beta_map(p_tr, 1.0, 1.0, 0.0)))
    if not (np.isfinite(train_rae)) or train_rae > ident_rae + 1e-9:
        a, b, c, train_rae = 1.0, 1.0, 0.0, float(ident_rae)
        ok = False
    else:
        ok = bool(res.success)
    return a, b, c, train_rae, ok


def _run_one_seed(
    pred_base: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Run beta-calibration pipeline at a single kf_seed."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_beta = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_train_raes = []
    fold_params = []
    fold_ok = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        a, b, c, tr_rae, ok = _fit_beta(pred_base[tr_loc], y_unb[tr_loc])
        val_pred = _apply_beta(pred_base[va_loc], a, b, c)
        oof_beta[va_loc] = val_pred
        fold_val_raes.append(float(rae(y_unb[va_loc], val_pred)))
        fold_train_raes.append(tr_rae)
        fold_params.append((a, b, c))
        fold_ok.append(ok)

    if np.isnan(oof_beta).any():
        raise RuntimeError(
            f"kf_seed={kf_seed}: scaffold splits did not cover all rows"
        )
    pooled = float(rae(y_unb, oof_beta))
    a_arr = [p[0] for p in fold_params]
    b_arr = [p[1] for p in fold_params]
    c_arr = [p[2] for p in fold_params]
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(fold_val_raes)),
        "per_fold_val_rae_std": float(np.std(fold_val_raes, ddof=1)),
        "per_fold_train_rae_mean": float(np.mean(fold_train_raes)),
        "fold_a": a_arr,
        "fold_b": b_arr,
        "fold_c": c_arr,
        "n_fit_ok": int(sum(fold_ok)),
        "oof": oof_beta,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(
        f"{TAG} -- 3-param BETA calibration on {PARENT_TAG} "
        f"(q35-quantile-conditional blend)"
    )
    print(
        f"          g(p)=sigmoid(c + a*ln(p) - b*ln(1-p)); fit (a,b,c) "
        f"on fold-train RAE"
    )
    print(f"          normalize: p=(pred-{P_LO})/({P_HI}-{P_LO}), clip eps={EPS}")
    print(
        f"          kf_seeds = {len(KF_SEEDS)} fresh "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}"
    )
    print(f"          honest gate metric = PER-FOLD-MEAN")
    print(f"          gate: pf_mean < {GATE_BETTER:.4f} -> BETTER, else FAIL")
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
    print(f"STEP 1: load {PARENT_TAG} pred_oof + te (q35 hard-split blend)")
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

    # Leak sanity on parent
    leak_eq = float(np.mean(np.isclose(pred_base, y_unb, atol=1e-6)))
    if leak_eq > 0.05:
        print(f"   WARN parent: {leak_eq:.1%} rows == truth -- possible leak")

    # Range sanity: how much mass falls outside [P_LO, P_HI] (gets clipped)
    n_below = int(np.sum(pred_base < P_LO))
    n_above = int(np.sum(pred_base > P_HI))
    print(
        f"   pred_base outside [{P_LO},{P_HI}]: below={n_below}  above={n_above}"
        f" (clipped to eps band)"
    )
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
    all_a = []
    all_b = []
    all_c = []
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(pred_base, y_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        per_fold_means.append(res["per_fold_val_rae_mean"])
        oof_stack.append(res["oof"])
        all_a.extend(res["fold_a"])
        all_b.extend(res["fold_b"])
        all_c.extend(res["fold_c"])
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
            "per_fold_val_rae_std": round(res["per_fold_val_rae_std"], 4),
            "per_fold_train_rae_mean": round(res["per_fold_train_rae_mean"], 4),
            "fold_a": [round(v, 4) for v in res["fold_a"]],
            "fold_b": [round(v, 4) for v in res["fold_b"]],
            "fold_c": [round(v, 4) for v in res["fold_c"]],
            "n_fit_ok": res["n_fit_ok"],
        })
        print(
            f"   kf={s}: pooled_RAE={res['pooled_rae']:.4f}  "
            f"pf_mean={res['per_fold_val_rae_mean']:.4f}  "
            f"a~{np.mean(res['fold_a']):.2f} b~{np.mean(res['fold_b']):.2f} "
            f"c~{np.mean(res['fold_c']):.2f}  "
            f"ok={res['n_fit_ok']}/{N_FOLDS}  wall={time.time()-ts:.2f}s"
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

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds)")
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
    print(f"\n   ref {PARENT_TAG} pooled baseline = {REF_PARENT_NB3090:.4f}")
    print(
        f"   delta vs {PARENT_TAG} (pooled)   = "
        f"{mean_rae - REF_PARENT_NB3090:+.4f}"
    )
    print(
        f"   delta vs {PARENT_TAG} (pf_mean)  = "
        f"{pf_mean - REF_PARENT_NB3090:+.4f}"
    )
    print(
        f"   ref nb3190 hard-clip on nb3090 = {REF_NB3190:.4f}"
    )
    print(f"   delta vs nb3190 (pf_mean)      = {pf_mean - REF_NB3190:+.4f}")
    print(f"   ref nb2171 prior PRIMARY-1 = {REF_NB2171:.4f}")
    print(f"   gain vs nb2171 (pf_mean)        = {REF_NB2171 - pf_mean:+.4f}")

    a_all = np.asarray(all_a, dtype=np.float64)
    b_all = np.asarray(all_b, dtype=np.float64)
    c_all = np.asarray(all_c, dtype=np.float64)
    print(
        f"\n   75-fold a stats: mean={a_all.mean():.4f}  "
        f"std={a_all.std(ddof=1):.4f}  min={a_all.min():.4f}  "
        f"max={a_all.max():.4f}"
    )
    print(
        f"   75-fold b stats: mean={b_all.mean():.4f}  "
        f"std={b_all.std(ddof=1):.4f}  min={b_all.min():.4f}  "
        f"max={b_all.max():.4f}"
    )
    print(
        f"   75-fold c stats: mean={c_all.mean():.4f}  "
        f"std={c_all.std(ddof=1):.4f}  min={c_all.min():.4f}  "
        f"max={c_all.max():.4f}"
    )

    # -- Deploy: fit (a, b, c) on FULL 253 y ---------------------------------
    da, db, dc, deploy_tr_rae, deploy_ok = _fit_beta(pred_base, y_unb)
    te_pred = _apply_beta(te_base, da, db, dc).astype(np.float32)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(
        f"\n   deploy (a, b, c) = ({da:.4f}, {db:.4f}, {dc:.4f})  "
        f"train_RAE={deploy_tr_rae:.4f}  ok={deploy_ok}"
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

    # -- Gate (on PER-FOLD-MEAN) ---------------------------------------------
    print("\n" + "-" * 78)
    print("GATE (honest metric = PER-FOLD-MEAN)")
    print("-" * 78)
    if pf_mean < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE-CANDIDATE. nb3363 15-seed PER-FOLD-MEAN {pf_mean:.4f} "
            f"clears BETTER gate {GATE_BETTER:.4f} ({pf_mean - GATE_BETTER:+.4f}). "
            f"3-param Beta calibration (a,b,c fit to fold-train RAE) on q35 "
            f"anchor nb3090 ({REF_PARENT_NB3090:.4f}) -> {pf_mean:.4f} = "
            f"{REF_PARENT_NB3090 - pf_mean:.4f} RAE reduction; delta vs hard-clip "
            f"sibling nb3190 ({REF_NB3190:.4f}) = {pf_mean - REF_NB3190:+.4f}. "
            f"The richer 3-DOF monotone map (independent low-/high-tail "
            f"curvature + shift) found smooth correction headroom while "
            f"preserving rank order. Re-verify with deep-30 before PRIMARY-1 "
            f"swap; watch (a,b,c) fold-dispersion for selection variance. "
            f"anchor_pre_unblind=True (parent nb3090 on PRE-clean K18/K19 "
            f"deep-30 OOFs)."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3363 15-seed PER-FOLD-MEAN {pf_mean:.4f} fails BETTER "
            f"gate {GATE_BETTER:.4f} ({pf_mean - GATE_BETTER:+.4f}). Delta vs "
            f"parent nb3090 (pf_mean) = {pf_mean - REF_PARENT_NB3090:+.4f}, "
            f"delta vs hard-clip nb3190 = {pf_mean - REF_NB3190:+.4f}. The "
            f"3-param Beta map fit to fold-train RAE either overfits the "
            f"77-row fold-train calibration set (3 DOF x 5 folds selection "
            f"variance, mirroring deep-30 under-dispersion findings) or the "
            f"q35 anchor is already near rank-optimal so monotone reshaping "
            f"cannot extract headroom (cf. nb3234 sigmoid, nb2200 stretch "
            f"both REJECTED, nb2095 deep-30 post-hoc ceiling 0.4720). If "
            f"pf_mean <= parent ({REF_PARENT_NB3090:.4f}) the Beta map is at "
            f"least rank-preserving non-degrading. Closes the parametric-Beta "
            f"calibration axis on this anchor; post-hoc operator change is "
            f"exhausted -- pivot to substrate (new anchor / off-manifold "
            f"features / abstention)."
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

    sub_csv = SUBMISSIONS / f"{TAG}_beta_calibration.csv"
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
            "3param_beta_calibration_a_b_c_fit_fold_train_rae_neldermead_"
            "on_nb3090_q35_quantile_conditional_blend"
        ),
        "anchor_pred_oof_path": str(PRED_OOF_PATH),
        "anchor_te_path": str(TE_PATH),
        "anchor_pre_unblind": True,  # nb3090 built on PRE-clean K18/K19 deep-30
        "parent_full_oof_rae": round(full_oof_rae, 4),
        "parent_leak_eq_truth_frac": round(leak_eq, 4),
        "norm_p_lo": P_LO,
        "norm_p_hi": P_HI,
        "norm_eps": EPS,
        "beta_map": "sigmoid(c + a*ln(p) - b*ln(1-p))",
        "fit_objective": "fold_train_rae",
        "fit_method": "Nelder-Mead",
        "n_pred_below_plo": n_below,
        "n_pred_above_phi": n_above,
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
        "fold_a_mean": round(float(a_all.mean()), 4),
        "fold_a_std": round(float(a_all.std(ddof=1)), 4),
        "fold_b_mean": round(float(b_all.mean()), 4),
        "fold_b_std": round(float(b_all.std(ddof=1)), 4),
        "fold_c_mean": round(float(c_all.mean()), 4),
        "fold_c_std": round(float(c_all.std(ddof=1)), 4),
        "ref_parent_nb3090": REF_PARENT_NB3090,
        "delta_vs_parent_pooled": round(mean_rae - REF_PARENT_NB3090, 4),
        "delta_vs_parent_pf_mean": round(pf_mean - REF_PARENT_NB3090, 4),
        "ref_nb3190_hard_clip": REF_NB3190,
        "delta_vs_nb3190_pf_mean": round(pf_mean - REF_NB3190, 4),
        "ref_nb2171": REF_NB2171,
        "gain_vs_nb2171_pf_mean": round(REF_NB2171 - pf_mean, 4),
        "deploy_a": round(da, 4),
        "deploy_b": round(db, 4),
        "deploy_c": round(dc, 4),
        "deploy_train_rae": round(deploy_tr_rae, 4),
        "deploy_fit_ok": bool(deploy_ok),
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
    print(f"   delta vs nb3090 (pf) = {pf_mean - REF_PARENT_NB3090:+.4f}")
    print(f"   delta vs nb3190 (pf) = {pf_mean - REF_NB3190:+.4f}")
    print(f"   gain vs nb2171  (pf) = {REF_NB2171 - pf_mean:+.4f}")
    print(f"   deploy (a,b,c)       = ({da:.3f}, {db:.3f}, {dc:.3f})")
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
        "deploy_a", "deploy_b", "deploy_c",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
