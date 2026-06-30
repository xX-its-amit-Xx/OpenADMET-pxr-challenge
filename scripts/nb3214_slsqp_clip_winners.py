"""nb3214 -- SLSQP simplex blend on 3 clip-winner anchors (vs equal weight).

NEW PARADIGM: LEARN SLSQP weights for the clip-winner blend.

ANCHORS (all clip operators, all PRE-unblind anchor chain, all in the
~0.4420-0.4427 ceiling band):
    1. nb3190  -- learned per-fold clip on nb3090 anchor (15-seed mean 0.4426)
    2. nb3173  -- learned per-fold clip on nb3080 anchor (15-seed mean 0.4422)
    3. nb3174  -- fixed (q05, q95) clip on nb3090 anchor  (5-seed  mean 0.4427)

PROTOCOL (per kf_seed, 5-fold scaffold split on 253 unblind):
    P = stack([nb3190_oof, nb3173_oof, nb3174_oof], axis=1)  # (253, 3)
    Per outer fold:
        a) SLSQP minimize RAE on fold-TRAIN portion; w on simplex
           (w >= 0, sum(w) = 1).
        b) Apply fold-train weights to fold-VAL rows.
    Concat all fold-val predictions -> blended_oof (no leak).
    Repeat for 15 FRESH kf_seeds {1216..1230}.

GATE (on 15-seed mean):
    mean < 0.4418 -> "BETTER"  (beats best single clip-winner nb3173 0.4422)
    else          -> "FAIL"

Deploy te uses mean-of-fold-weights (averaged across all 75 folds).

References (PRE-unblind ceiling cluster):
    nb3173  learned clip on nb3080 = 0.4422  <- best single clip-winner
    nb3190  learned clip on nb3090 = 0.4426
    nb3174  fixed     clip on nb3090 = 0.4427
    nb3161  fixed     clip on nb3080 = 0.4437
    nb3080  parent (no clip)         = 0.4475
    nb3090  parent (no clip)         = 0.4472

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb3190_pred_oof.npy   data/processed/te_nb3190.npy
    data/processed/nb3173_pred_oof.npy   data/processed/te_nb3173.npy
    data/processed/nb3174_pred_oof.npy   data/processed/te_nb3174.npy

Outputs:
    data/processed/nb3214_summary.json
    data/processed/nb3214_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3214.npy         (513,) float32 -- deploy te
    submissions/nb3214_slsqp_clip_winners.csv  (only on BETTER)
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

TAG = "nb3214"

# -- Anchors -------------------------------------------------------------------
ANCHOR_NAMES = ["nb3190", "nb3173", "nb3174"]
ANCHOR_OOF_PATHS = {
    "nb3190": DATA_PROCESSED / "nb3190_pred_oof.npy",
    "nb3173": DATA_PROCESSED / "nb3173_pred_oof.npy",
    "nb3174": DATA_PROCESSED / "nb3174_pred_oof.npy",
}
ANCHOR_TE_PATHS = {
    "nb3190": DATA_PROCESSED / "te_nb3190.npy",
    "nb3173": DATA_PROCESSED / "te_nb3173.npy",
    "nb3174": DATA_PROCESSED / "te_nb3174.npy",
}

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))  # 15 FRESH seeds {1216..1230}

# -- SLSQP options -------------------------------------------------------------
N_SLSQP_STARTS = 8
DEGEN_MAX_W = 0.85

# -- Gates ---------------------------------------------------------------------
GATE_BETTER = 0.4418  # mean < this -> BETTER (beats nb3173 best single 0.4422)

# -- References ----------------------------------------------------------------
REF_NB3173 = 0.4422
REF_NB3190 = 0.4426
REF_NB3174 = 0.4427
REF_NB3161 = 0.4437
REF_NB3080 = 0.4475
REF_NB3090 = 0.4472
REF_NB2171 = 0.4682


def _simplex_slsqp(
    P: np.ndarray,
    y: np.ndarray,
    n_starts: int = N_SLSQP_STARTS,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Minimize RAE(y, P @ w) on the simplex (w>=0, sum(w)=1)."""
    K = P.shape[1]
    rng = np.random.default_rng(seed)

    def loss(w: np.ndarray) -> float:
        return float(rae(y, P @ w))

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bnds = [(0.0, 1.0)] * K

    starts = [np.full(K, 1.0 / K)]
    for _ in range(n_starts - 1):
        starts.append(rng.dirichlet(np.ones(K)))

    best_w: np.ndarray | None = None
    best_r = np.inf
    for x0 in starts:
        try:
            res = minimize(
                loss, x0, method="SLSQP",
                bounds=bnds, constraints=cons,
                options={"maxiter": 300, "ftol": 1e-9},
            )
            w = np.clip(res.x, 0.0, 1.0)
            s = float(w.sum())
            if s <= 0.0:
                continue
            w = w / s
            r = float(rae(y, P @ w))
            if r < best_r:
                best_r = r
                best_w = w
        except Exception:
            continue
    if best_w is None:
        best_w = np.full(K, 1.0 / K)
        best_r = float(rae(y, P @ best_w))
    return best_w, best_r


def _run_one_seed(
    P: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Per-fold SLSQP simplex at a single kf_seed."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    K = P.shape[1]
    oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes: list[float] = []
    fold_weights: list[np.ndarray] = []
    fold_train_raes: list[float] = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        w, r_tr = _simplex_slsqp(
            P[tr_loc], y_unb[tr_loc],
            n_starts=N_SLSQP_STARTS, seed=kf_seed * 100 + fold_i,
        )
        fold_weights.append(w)
        fold_train_raes.append(float(r_tr))
        val_pred = P[va_loc] @ w
        oof_blend[va_loc] = val_pred
        fold_val_raes.append(float(rae(y_unb[va_loc], val_pred)))
    if np.isnan(oof_blend).any():
        raise RuntimeError(
            f"kf_seed={kf_seed}: scaffold splits did not cover all rows"
        )
    pooled = float(rae(y_unb, oof_blend))
    W = np.stack(fold_weights, axis=0)  # (n_folds, K)
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(fold_val_raes)),
        "per_fold_val_rae_std": float(np.std(fold_val_raes, ddof=1)),
        "per_fold_train_rae_mean": float(np.mean(fold_train_raes)),
        "fold_weights": W,
        "any_fold_degenerate": bool(W.max(axis=1).max() > DEGEN_MAX_W),
        "oof": oof_blend,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- per-fold SLSQP simplex on 3 clip winners")
    print(f"          anchors  = {ANCHOR_NAMES}")
    print(
        f"          kf_seeds = {len(KF_SEEDS)} fresh "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}"
    )
    print(f"          gate: mean < {GATE_BETTER:.4f} -> BETTER, else FAIL")
    print("=" * 78)

    # -- Load test, truth, unblind idx ---------------------------------------
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

    # -- Load anchors ---------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 1: load 3 clip-winner anchors (pred_oof + te)")
    print("-" * 78)
    P_list = []
    te_list = []
    indiv_rae = {}
    leak_eq = {}
    for name in ANCHOR_NAMES:
        oof = np.load(ANCHOR_OOF_PATHS[name]).astype(np.float64)
        te_a = np.load(ANCHOR_TE_PATHS[name]).astype(np.float64)
        if oof.shape != (n_unb,):
            raise ValueError(
                f"{name} pred_oof shape {oof.shape} != ({n_unb},)"
            )
        if te_a.shape != (n_test,):
            raise ValueError(
                f"{name} te shape {te_a.shape} != ({n_test},)"
            )
        r = float(rae(y_unb, oof))
        leak = float(np.mean(np.isclose(oof, y_unb, atol=1e-6)))
        indiv_rae[name] = round(r, 4)
        leak_eq[name] = round(leak, 4)
        P_list.append(oof)
        te_list.append(te_a)
        print(
            f"   {name}: oof_RAE={r:.4f}  "
            f"oof mean={oof.mean():.3f} std={oof.std():.3f}  "
            f"leak_eq={leak:.2%}  "
            f"te mean={te_a.mean():.3f} std={te_a.std():.3f}"
        )
        if leak > 0.05:
            print(f"   WARN {name}: {leak:.1%} rows == truth -- possible leak")
    P = np.stack(P_list, axis=1)  # (253, 3)
    TE = np.stack(te_list, axis=1)  # (513, 3)

    # Cross-anchor correlation
    corr_mat = np.corrcoef(P.T)
    print("\n   cross-anchor Pearson corr (OOF):")
    for i, ni in enumerate(ANCHOR_NAMES):
        cells = "  ".join(f"{corr_mat[i, j]:.3f}" for j in range(len(ANCHOR_NAMES)))
        print(f"     {ni:>8}: {cells}")

    # Equal-weight baseline
    w_eq = np.full(P.shape[1], 1.0 / P.shape[1])
    eq_oof = P @ w_eq
    eq_oof_rae = float(rae(y_unb, eq_oof))
    print(
        f"\n   equal-weight baseline OOF RAE = {eq_oof_rae:.4f} "
        f"(reference for SLSQP delta)"
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
    pooled_raes: list[float] = []
    oof_stack: list[np.ndarray] = []
    all_fold_weights: list[np.ndarray] = []
    any_degen = False
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(P, y_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        oof_stack.append(res["oof"])
        all_fold_weights.append(res["fold_weights"])  # (n_folds, K)
        any_degen = any_degen or res["any_fold_degenerate"]
        per_fold_w_round = [
            {ANCHOR_NAMES[k]: round(float(res["fold_weights"][f, k]), 3)
             for k in range(len(ANCHOR_NAMES))}
            for f in range(res["fold_weights"].shape[0])
        ]
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
            "per_fold_val_rae_std": round(res["per_fold_val_rae_std"], 4),
            "per_fold_train_rae_mean": round(
                res["per_fold_train_rae_mean"], 4
            ),
            "per_fold_weights": per_fold_w_round,
            "any_fold_degenerate": res["any_fold_degenerate"],
        })
        seed_mean_w = res["fold_weights"].mean(axis=0)
        seed_mean_w = seed_mean_w / max(seed_mean_w.sum(), 1e-12)
        wmean_str = "  ".join(
            f"{ANCHOR_NAMES[k]}={seed_mean_w[k]:.2f}"
            for k in range(len(ANCHOR_NAMES))
        )
        print(
            f"   kf={s}: pooled_RAE={res['pooled_rae']:.4f}  "
            f"w_mean({wmean_str})  "
            f"degen={res['any_fold_degenerate']}  "
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

    # Mean-of-fold weights (across all n_seeds * n_folds = 75 folds)
    all_W = np.concatenate(all_fold_weights, axis=0)  # (75, 3)
    w_mean_of_folds = all_W.mean(axis=0)
    w_mean_of_folds = w_mean_of_folds / max(w_mean_of_folds.sum(), 1e-12)

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
        f"\n   equal-weight baseline OOF       = {eq_oof_rae:.4f}"
    )
    print(
        f"   delta vs equal-weight (mean-of-folds in-sample) = "
        f"{rae(y_unb, P @ w_mean_of_folds) - eq_oof_rae:+.4f}"
    )
    print(
        f"\n   ref nb3173 (best clip single)   = {REF_NB3173:.4f}"
    )
    print(
        f"   delta vs nb3173                  = "
        f"{mean_rae - REF_NB3173:+.4f}"
    )
    print(
        f"   ref nb3190                       = {REF_NB3190:.4f}"
    )
    print(
        f"   ref nb3174                       = {REF_NB3174:.4f}"
    )

    print("\n   mean-of-folds weight (across 75 folds):")
    for k, n in enumerate(ANCHOR_NAMES):
        print(f"     {n:>8}: {w_mean_of_folds[k]:.4f}")

    # -- Deploy te ------------------------------------------------------------
    print("\n" + "-" * 78)
    print("DEPLOY te (mean-of-folds weights applied to anchor te(513,3))")
    print("-" * 78)
    te_pred = (TE @ w_mean_of_folds).astype(np.float32)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
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
    print("GATE")
    print("-" * 78)
    if mean_rae < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE-CANDIDATE. nb3214 15-seed mean {mean_rae:.4f} beats "
            f"BETTER gate {GATE_BETTER:.4f} "
            f"({mean_rae - GATE_BETTER:+.4f}) and the best single clip-winner "
            f"nb3173 {REF_NB3173:.4f} ({mean_rae - REF_NB3173:+.4f}). SLSQP "
            f"simplex extracts a learnable blend across the 3 clip operators "
            f"even though they are highly correlated. Mean-of-fold weights = "
            f"{ {n: round(float(w_mean_of_folds[k]),3) for k,n in enumerate(ANCHOR_NAMES)} }. "
            f"Re-verify with deep-30 before any PRIMARY-1 swap."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3214 15-seed mean {mean_rae:.4f} fails BETTER gate "
            f"{GATE_BETTER:.4f} ({mean_rae - GATE_BETTER:+.4f}). The 3 clip "
            f"winners are too correlated for SLSQP to extract residual "
            f"orthogonality; learned blend collapses to (or below) the best "
            f"single member. Delta vs nb3173 {REF_NB3173:.4f} = "
            f"{mean_rae - REF_NB3173:+.4f}, vs equal-weight {eq_oof_rae:.4f} "
            f"= {mean_rae - eq_oof_rae:+.4f}. Keep nb3173 on the ladder; "
            f"close the SLSQP-on-clip-winners axis. Substrate change "
            f"(orthogonal anchor) remains the open lever."
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

    sub_csv = SUBMISSIONS / f"{TAG}_slsqp_clip_winners.csv"
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
        "method": "per_fold_slsqp_simplex_on_3_clip_winners",
        "anchors": ANCHOR_NAMES,
        "anchor_oof_paths": {k: str(v) for k, v in ANCHOR_OOF_PATHS.items()},
        "anchor_te_paths": {k: str(v) for k, v in ANCHOR_TE_PATHS.items()},
        "anchor_pre_unblind": True,
        "indiv_oof_rae": indiv_rae,
        "leak_eq_truth_frac": leak_eq,
        "cross_anchor_corr_oof": [
            [round(float(corr_mat[i, j]), 4) for j in range(len(ANCHOR_NAMES))]
            for i in range(len(ANCHOR_NAMES))
        ],
        "equal_weight_oof_rae": round(eq_oof_rae, 4),
        "n_folds": N_FOLDS,
        "n_slsqp_starts": N_SLSQP_STARTS,
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
        "any_fold_degenerate_across_seeds": bool(any_degen),
        "mean_of_fold_weights": {
            ANCHOR_NAMES[k]: round(float(w_mean_of_folds[k]), 4)
            for k in range(len(ANCHOR_NAMES))
        },
        "ref_nb3173_best_single_clip": REF_NB3173,
        "delta_vs_nb3173": round(mean_rae - REF_NB3173, 4),
        "delta_vs_equal_weight": round(mean_rae - eq_oof_rae, 4),
        "ref_nb3190": REF_NB3190,
        "ref_nb3174": REF_NB3174,
        "ref_nb3161": REF_NB3161,
        "ref_nb3080": REF_NB3080,
        "ref_nb3090": REF_NB3090,
        "ref_nb2171": REF_NB2171,
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
    print(f"   mean_rae ({n_s} seeds)   = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   95% CI                = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   delta vs nb3173       = {mean_rae - REF_NB3173:+.4f}")
    print(f"   delta vs equal-weight = {mean_rae - eq_oof_rae:+.4f}")
    print(f"   mean-of-fold weights  = "
          f"{ {n: round(float(w_mean_of_folds[k]),3) for k,n in enumerate(ANCHOR_NAMES)} }")
    print(f"   verdict               = {verdict}")
    print(f"   wall                  = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae", "std_rae", "ci95_low", "ci95_high",
        "delta_vs_nb3173", "delta_vs_equal_weight",
        "equal_weight_oof_rae", "indiv_oof_rae",
        "mean_of_fold_weights", "any_fold_degenerate_across_seeds",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
