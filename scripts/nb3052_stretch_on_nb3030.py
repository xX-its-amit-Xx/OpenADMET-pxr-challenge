"""nb3052 -- Per-fold golden-section rank-stretch on nb3030 deploy.

NEW PARADIGM:
    Cycle-249 nb3022 tried per-fold stretch on nb3002 (2-anchor K18+K19 deep-30
    per-fold simplex base, mean pooled RAE 0.45105 over 5 kf_seeds).  nb3030 is
    the wide-verified PRIMARY-1: 15-seed per-fold-SLSQP on K18+K19+K23 (3-anchor
    deep-30 simplex), wide-seed mean RAE = 0.4509 +/- 0.0017 over kf_seeds
    {1051-1065}, gate-promoted VERIFIED_NEW_PRIMARY1.

    This script applies the same per-fold scalar rank-stretch paradigm to the
    nb3030 wide-verified base, with 15 FRESH kf_seeds {1081-1095} (disjoint from
    nb3030's verification set) and a wider search range s in [0.95, 1.20].  If
    the 3-anchor simplex blend is still variance-compressed (pred_std <
    truth_std), a single scalar s per fold should extract a small further gain.

PROTOCOL:
    pred_base = nb3030_pred_oof                       (253-vector, float32)
    te_base   = te_nb3030                             (513-vector, float32)
    For each kf_seed in {1081, ..., 1095} (15 fresh seeds):
        scaffold-CV 5-fold over unb_scaffolds (shuffle, seed=kf_seed)
        per fold:
            mu_tr  = mean(pred_base[tr_loc])
            s_star = argmin_{s in [0.95, 1.20]} RAE(y_tr, mu_tr + s*(p_tr - mu_tr))
            pred_val[va_loc] = mu_tr + s_star * (pred_base[va_loc] - mu_tr)
        pooled_rae[seed] = rae(y_unb, pred_val)
    mean_rae = mean(pooled_rae across 15 seeds)

GATES:
    mean_rae < 0.4509  -> "BETTER_THAN_NB3030"  (nb3030 wide-verified mean)
    else               -> "FAIL"

References:
    nb3030 wide-seed (15 kf_seeds {1051-1065}) mean = 0.4509 +/- 0.0017
    nb3022 per-fold stretch on nb3002 (cycle 249) -- mirror experiment
    nb2983 per-fold stretch on nb2973 = +0.0014 (FAILED)
    nb2171 ceiling                    = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb3030_pred_oof.npy   (253,) float32
    data/processed/te_nb3030.npy         (513,) float32

Outputs:
    data/processed/nb3052_summary.json
    data/processed/nb3052_pred_oof.npy   (253,) float32 -- best-seed (min RAE) OOF
    data/processed/te_nb3052.npy         (513,) float32 -- deploy te (mean s)
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

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb3052"
PARENT_TAG = "nb3030"

# -- Base predictor ------------------------------------------------------------
BASE_OOF_PATH = DATA_PROCESSED / "nb3030_pred_oof.npy"
BASE_TE_PATH = DATA_PROCESSED / "te_nb3030.npy"

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1081, 1096))  # 15 fresh seeds {1081..1095}

# -- Stretch search ------------------------------------------------------------
S_LO = 0.95
S_HI = 1.20
GS_TOL = 1e-4
GS_MAX_ITER = 60

# -- Gates ---------------------------------------------------------------------
GATE_BETTER_NB3030 = 0.4509

# -- References ----------------------------------------------------------------
REF_NB3030 = 0.4509  # wide-seed (15 kf_seeds {1051..1065}) mean
REF_NB3030_STD = 0.0017
REF_NB3022 = None  # per-fold stretch on nb3002 (cycle 249)
REF_NB2983 = 0.4553  # nb2973 (0.4539) + stretch hurt by +0.0014
REF_NB2171 = 0.4682


def _stretch(pred, mu, s):
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


def cv_run_for_seed(pred_base, y_unb, unb_scaffolds, kf_seed):
    """One scaffold-CV pass: per-fold golden-section stretch on TRAIN, apply to VAL."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(pred_base)
    oof_pred = np.full(n_unb, np.nan)
    per_fold = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        p_tr = pred_base[tr_loc]
        y_tr = y_unb[tr_loc]
        mu_tr = float(np.mean(p_tr))

        def f(s, p_tr=p_tr, y_tr=y_tr, mu_tr=mu_tr):
            return float(rae(y_tr, _stretch(p_tr, mu_tr, s)))

        s_star, fold_train_rae = golden_section_min(f, S_LO, S_HI)
        f_lo = f(S_LO)
        f_hi = f(S_HI)
        oof_pred[va_loc] = _stretch(pred_base[va_loc], mu_tr, s_star)
        per_fold.append({
            "fold": int(fold_i),
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "mu_tr_pred": float(mu_tr),
            "s_star": float(s_star),
            "fold_train_rae_at_s_star": float(fold_train_rae),
            "fold_train_rae_at_s_lo": float(f_lo),
            "fold_train_rae_at_s_hi": float(f_hi),
        })
    assert not np.isnan(oof_pred).any()
    return float(rae(y_unb, oof_pred)), oof_pred, per_fold


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- per-fold golden-section rank-stretch on {PARENT_TAG}")
    print(f"   base       : nb3030 pred_oof + te (K18+K19+K23 wide-verified PRIMARY-1)")
    print(f"   stretch    : s in [{S_LO}, {S_HI}] golden-section per fold")
    print(f"   kf_seeds   : {KF_SEEDS[0]}..{KF_SEEDS[-1]} (n={len(KF_SEEDS)} fresh)")
    print(f"   n_folds    : {N_FOLDS}")
    print(f"   gate       : BETTER_THAN_NB3030 < {GATE_BETTER_NB3030}")
    print("=" * 78)

    # ---- Load test set + unblind labels ----
    te = load_test()
    te_names = (
        te["name"].values if "name" in te.columns else te["Molecule Name"].values
    )
    te_smiles = (
        te["smiles"].values if "smiles" in te.columns else te["SMILES"].values
    )
    n_te = len(te_names)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"\n[load] n_te={n_te}  n_unb={n_unb}  unique_scaffolds={n_unique_scaf}")
    print(f"[load] y_unb mean={y_unb.mean():.4f}  std={y_unb.std():.4f}")

    # ---- Load base predictions ----
    assert BASE_OOF_PATH.exists(), f"missing base: {BASE_OOF_PATH}"
    assert BASE_TE_PATH.exists(), f"missing base: {BASE_TE_PATH}"
    pred_base_oof = np.load(BASE_OOF_PATH).astype(np.float64)
    pred_base_te = np.load(BASE_TE_PATH).astype(np.float64)
    assert pred_base_oof.shape == (n_unb,), \
        f"base OOF shape {pred_base_oof.shape} != ({n_unb},)"
    assert pred_base_te.shape == (n_te,), \
        f"base te shape {pred_base_te.shape} != ({n_te},)"

    rae_base = float(rae(y_unb, pred_base_oof))
    print(f"\n[base] nb3030 pred_oof RAE = {rae_base:.4f}")
    print(f"[base] pred mean={pred_base_oof.mean():.4f}  "
          f"std={pred_base_oof.std():.4f}")
    print(f"[base] variance-compression ratio (pred_std/truth_std) = "
          f"{pred_base_oof.std() / y_unb.std():.4f} "
          f"(<1.0 => compressed; stretch motivated)")

    # Leak sanity
    eq_truth_frac = float(np.mean(np.isclose(pred_base_oof, y_unb, atol=1e-6)))
    if eq_truth_frac > 0.05:
        print(f"   WARN: {eq_truth_frac:.1%} rows == truth -- possible leak")

    # ---- Sweep kf_seeds ----
    print("\n" + "-" * 78)
    print(f"PER-FOLD GOLDEN-SECTION s in [{S_LO}, {S_HI}]  x  {len(KF_SEEDS)} kf_seeds")
    print("-" * 78)
    seed_results = []
    oof_by_seed = {}
    for kf_seed in KF_SEEDS:
        r, oof_pred, per_fold = cv_run_for_seed(
            pred_base_oof, y_unb, unb_scaffolds, kf_seed,
        )
        s_vals = [pf["s_star"] for pf in per_fold]
        seed_results.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": float(r),
            "s_stars": [float(s) for s in s_vals],
            "s_mean": float(np.mean(s_vals)),
            "s_std": float(np.std(s_vals)),
            "pred_va_std": float(oof_pred.std()),
            "pred_va_mean": float(oof_pred.mean()),
            "per_fold": per_fold,
        })
        oof_by_seed[int(kf_seed)] = oof_pred
        print(f"   kf_seed={kf_seed}   pooled_RAE={r:.4f}   "
              f"s_stars={np.round(s_vals, 3).tolist()}   "
              f"s_mean={np.mean(s_vals):.3f}   "
              f"pred_std={oof_pred.std():.4f}")

    raes = np.array([r["pooled_rae"] for r in seed_results])
    mean_rae = float(raes.mean())
    std_rae = float(raes.std())
    sem_rae = float(std_rae / np.sqrt(len(raes)))
    min_rae = float(raes.min())
    max_rae = float(raes.max())
    median_rae = float(np.median(raes))
    p5_rae = float(np.percentile(raes, 5))
    p95_rae = float(np.percentile(raes, 95))
    ci95_low = mean_rae - 1.96 * sem_rae
    ci95_high = mean_rae + 1.96 * sem_rae
    best_seed = int(KF_SEEDS[int(np.argmin(raes))])

    print("\n" + "-" * 78)
    print("AGGREGATE")
    print("-" * 78)
    print(f"   mean RAE          = {mean_rae:.4f}")
    print(f"   std  RAE          = {std_rae:.4f}")
    print(f"   sem  RAE          = {sem_rae:.4f}")
    print(f"   median RAE        = {median_rae:.4f}")
    print(f"   p5/p95            = {p5_rae:.4f} / {p95_rae:.4f}")
    print(f"   min  RAE          = {min_rae:.4f}  (kf_seed={best_seed})")
    print(f"   max  RAE          = {max_rae:.4f}")
    print(f"   95% CI            = [{ci95_low:.4f}, {ci95_high:.4f}]")
    print(f"   delta_mean_vs_base = {mean_rae - rae_base:+.4f}")
    print(f"   delta_mean_vs_nb3030_ref ({REF_NB3030:.4f}) = "
          f"{mean_rae - REF_NB3030:+.4f}")

    # ---- Gate (on MEAN across seeds) ----
    if mean_rae < GATE_BETTER_NB3030:
        verdict = "BETTER_THAN_NB3030"
    else:
        verdict = "FAIL"
    print("\n" + "-" * 78)
    print(f"GATE  (on mean across {len(KF_SEEDS)} kf_seeds)")
    print("-" * 78)
    print(f"   mean_rae           = {mean_rae:.4f}")
    print(f"   BETTER_THAN_NB3030 < {GATE_BETTER_NB3030}  ->  "
          f"{mean_rae < GATE_BETTER_NB3030}")
    print(f"   verdict            = {verdict}")

    # ---- Save canonical pred_oof at best seed ----
    pred_oof_canon = oof_by_seed[best_seed].astype(np.float32)
    pred_oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    np.save(pred_oof_path, pred_oof_canon)
    print(f"\n[save] pred_oof @ best kf_seed={best_seed} -> {pred_oof_path}")
    print(f"       RAE_on_y_unb = {rae(y_unb, pred_oof_canon):.4f}")

    # ---- Deploy on 513 ----
    # Mean across all per-fold per-seed s_stars as deploy stretch factor;
    # mu_te = mean(pred_base_te).
    all_s = np.array(
        [s for r in seed_results for s in r["s_stars"]], dtype=np.float64,
    )
    s_deploy = float(all_s.mean())
    s_deploy_std = float(all_s.std())
    mu_te = float(pred_base_te.mean())
    te_deploy = (mu_te + s_deploy * (pred_base_te - mu_te)).astype(np.float32)
    te_deploy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_deploy_path, te_deploy)
    te_unb_rae = float(rae(y_unb, te_deploy[unb_idx]))
    print(f"\n[deploy] s_deploy mean = {s_deploy:.4f} +/- {s_deploy_std:.4f}  "
          f"(n={len(all_s)})")
    print(f"[deploy] te_base       : mean={pred_base_te.mean():.4f}  "
          f"std={pred_base_te.std():.4f}")
    print(f"[deploy] te_stretched  : mean={te_deploy.mean():.4f}  "
          f"std={te_deploy.std():.4f}")
    print(f"[deploy] te[unb] in-sample RAE = {te_unb_rae:.4f}")

    # ---- Summary ----
    summary = {
        "tag": TAG,
        "parent_tag": PARENT_TAG,
        "method": "nb3030_per_fold_golden_section_rank_stretch",
        "paradigm": "scalar_rank_stretch_on_K18_K19_K23_wide_verified_PRIMARY1_per_fold_simplex",
        "anchor_pre_unblind": True,
        "base_oof_path": str(BASE_OOF_PATH),
        "base_te_path": str(BASE_TE_PATH),
        "base_pooled_rae_on_unb": rae_base,
        "base_pred_mean": float(pred_base_oof.mean()),
        "base_pred_std": float(pred_base_oof.std()),
        "truth_std": float(y_unb.std()),
        "variance_compression_ratio": float(pred_base_oof.std() / y_unb.std()),
        "leak_eq_truth_frac": eq_truth_frac,
        "n_unb": int(n_unb),
        "n_te": int(n_te),
        "n_unique_scaffolds": int(n_unique_scaf),
        "kf_seeds": KF_SEEDS,
        "n_seeds": len(KF_SEEDS),
        "n_folds": N_FOLDS,
        "s_lo": S_LO,
        "s_hi": S_HI,
        "gs_tol": GS_TOL,
        "gs_max_iter": GS_MAX_ITER,
        "seed_results": seed_results,
        "mean_rae": mean_rae,
        "std_rae": std_rae,
        "sem_rae": sem_rae,
        "median_rae": median_rae,
        "p5_rae": p5_rae,
        "p95_rae": p95_rae,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "min_rae": min_rae,
        "max_rae": max_rae,
        "best_seed": best_seed,
        "delta_mean_vs_base": mean_rae - rae_base,
        "delta_min_vs_base": min_rae - rae_base,
        "delta_mean_vs_nb3030_ref": mean_rae - REF_NB3030,
        "ref_nb3030_wide_mean": REF_NB3030,
        "ref_nb3030_wide_std": REF_NB3030_STD,
        "ref_nb2983": REF_NB2983,
        "ref_nb2171": REF_NB2171,
        "gate_better_nb3030": GATE_BETTER_NB3030,
        "verdict": verdict,
        "s_deploy": s_deploy,
        "s_deploy_std": s_deploy_std,
        "s_deploy_n_samples": int(len(all_s)),
        "te_unb_in_sample_rae": te_unb_rae,
        "deploy_te_mean": float(te_deploy.mean()),
        "deploy_te_std": float(te_deploy.std()),
        "pred_oof_path": str(pred_oof_path),
        "te_npy_path": str(te_deploy_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    json_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {json_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   base                 = {PARENT_TAG} (K18+K19+K23 wide-verified PRIMARY-1)")
    print(f"   base RAE             = {rae_base:.4f}")
    print(f"   stretch range        = [{S_LO}, {S_HI}]")
    print(f"   mean RAE ({len(KF_SEEDS)} seeds)  = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   min  RAE             = {min_rae:.4f}  (kf_seed={best_seed})")
    print(f"   delta_mean_vs_base   = {mean_rae - rae_base:+.4f}")
    print(f"   delta_mean_vs_nb3030 = {mean_rae - REF_NB3030:+.4f}")
    print(f"   s_deploy             = {s_deploy:.4f} +/- {s_deploy_std:.4f}")
    print(f"   verdict              = {verdict}")
    print(f"   te[unb] in-sample    = {te_unb_rae:.4f}")
    print(f"   wall                 = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== KEY ====")
    for k in (
        "base_pooled_rae_on_unb",
        "mean_rae",
        "std_rae",
        "min_rae",
        "best_seed",
        "delta_mean_vs_base",
        "delta_mean_vs_nb3030_ref",
        "s_deploy",
        "verdict",
        "te_unb_in_sample_rae",
    ):
        print(f"  {k}: {res.get(k)}")
