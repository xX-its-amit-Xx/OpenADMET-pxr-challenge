"""nb2922 -- nb2902 0.5/0.5 blend + scalar rank-stretch.

NEW PARADIGM:
    Try rank-stretch on top of the cheap 2-anchor blend rather than on the
    raw nb2240 anchor (nb2850 explored stretch-on-nb2240).  The nb2902
    equal-weight {nb2240_K20, nb1191} mean pooled at 0.4599 on the 253;
    if the blend itself is variance-compressed (pred_std < truth_std),
    per-fold golden-section stretch should extract a small further gain.

PROTOCOL:
    pred_base = 0.5 * nb2240_K20 + 0.5 * nb1191              (253 OOF, 513 te)
    For each kf_seed in {1001, 1002, 1003, 1004, 1005}:
        scaffold-CV 5-fold over unb_scaffolds (shuffle, seed=kf_seed)
        per fold:
            mu_tr  = mean(pred_base[tr_loc])
            s_star = argmin_{s in [0.95, 1.20]} RAE(y_tr, mu_tr + s*(p_tr - mu_tr))
            pred_val[va_loc] = mu_tr + s_star * (pred_base[va_loc] - mu_tr)
        pooled_rae[seed] = rae(y_unb, pred_val)
    mean_rae = mean(pooled_rae across 5 seeds)

GATES:
    best (= MIN across seeds) and mean both reported; gate on MEAN
    mean_rae < 0.4570 -> PROMOTE
    mean_rae < 0.4598 -> MARGINAL_BEAT
    otherwise         -> FAIL

Outputs:
    data/processed/nb2922_summary.json
    data/processed/nb2922_pred_oof.npy   (253-vector, kf_seed=1001 OOF)
    data/processed/te_nb2922.npy         (513-vector, deploy mean-s stretch)
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

TAG = "nb2922"

# ---- Anchors (mirror nb2902) ----
ANCHOR_A_NAME = "nb2240_K20"
ANCHOR_A_OOF = "nb2240_mean_bag_oof_K20.npy"
ANCHOR_A_TE = "te_nb2240_K20.npy"

ANCHOR_B_NAME = "nb1191"
ANCHOR_B_OOF = "nb1191_pred_oof.npy"
ANCHOR_B_TE = "te_nb1191.npy"

W_A = 0.5
W_B = 0.5

# ---- CV protocol ----
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# ---- Stretch search ----
S_LO = 0.95
S_HI = 1.20
GS_TOL = 1e-4
GS_MAX_ITER = 60

# ---- Gates ----
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598


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
    print(f"{TAG} -- 2-anchor blend + per-fold scalar rank-stretch")
    print(f"   anchors    : {ANCHOR_A_NAME} ({W_A}) + {ANCHOR_B_NAME} ({W_B})")
    print(f"   stretch    : s in [{S_LO}, {S_HI}] golden-section per fold")
    print(f"   kf_seeds   : {KF_SEEDS}")
    print(f"   n_folds    : {N_FOLDS}")
    print(f"   gates      : PROMOTE<{GATE_PROMOTE}  MARGINAL<{GATE_MARGINAL}")
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

    # ---- Load anchors ----
    p_a_oof = DATA_PROCESSED / ANCHOR_A_OOF
    p_b_oof = DATA_PROCESSED / ANCHOR_B_OOF
    p_a_te = DATA_PROCESSED / ANCHOR_A_TE
    p_b_te = DATA_PROCESSED / ANCHOR_B_TE
    for p in (p_a_oof, p_b_oof, p_a_te, p_b_te):
        assert p.exists(), f"missing anchor artefact: {p}"

    oof_a = np.load(p_a_oof).astype(np.float64)
    oof_b = np.load(p_b_oof).astype(np.float64)
    te_a = np.load(p_a_te).astype(np.float64)
    te_b = np.load(p_b_te).astype(np.float64)
    assert oof_a.shape == (n_unb,), f"{ANCHOR_A_NAME} oof {oof_a.shape}"
    assert oof_b.shape == (n_unb,), f"{ANCHOR_B_NAME} oof {oof_b.shape}"
    assert te_a.shape == (n_te,), f"{ANCHOR_A_NAME} te {te_a.shape}"
    assert te_b.shape == (n_te,), f"{ANCHOR_B_NAME} te {te_b.shape}"

    indiv_oof_rae = {
        ANCHOR_A_NAME: float(rae(y_unb, oof_a)),
        ANCHOR_B_NAME: float(rae(y_unb, oof_b)),
    }
    print(f"\n[anchor] {ANCHOR_A_NAME:12s} oof_RAE={indiv_oof_rae[ANCHOR_A_NAME]:.4f}")
    print(f"[anchor] {ANCHOR_B_NAME:12s} oof_RAE={indiv_oof_rae[ANCHOR_B_NAME]:.4f}")

    # ---- Build base equal-weight blend ----
    pred_base_oof = W_A * oof_a + W_B * oof_b   # (253,)
    pred_base_te = W_A * te_a + W_B * te_b      # (513,)

    rae_base = float(rae(y_unb, pred_base_oof))
    print(f"\n[base] equal-weight blend pooled RAE = {rae_base:.4f}")
    print(f"[base] blend mean={pred_base_oof.mean():.4f}  "
          f"std={pred_base_oof.std():.4f}")
    print(f"[base] variance-compression ratio (pred_std/truth_std) = "
          f"{pred_base_oof.std() / y_unb.std():.4f} "
          f"(<1.0 => compressed; stretch motivated)")

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
    min_rae = float(raes.min())
    max_rae = float(raes.max())
    best_seed = int(KF_SEEDS[int(np.argmin(raes))])

    print("\n" + "-" * 78)
    print("AGGREGATE")
    print("-" * 78)
    print(f"   mean RAE   = {mean_rae:.4f}")
    print(f"   std  RAE   = {std_rae:.4f}")
    print(f"   min  RAE   = {min_rae:.4f}  (kf_seed={best_seed})")
    print(f"   max  RAE   = {max_rae:.4f}")
    print(f"   delta_mean_vs_base = {mean_rae - rae_base:+.4f}")
    print(f"   delta_min_vs_base  = {min_rae - rae_base:+.4f}")

    # ---- Gate (on MEAN across seeds) ----
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "-" * 78)
    print("GATE  (on mean across 5 kf_seeds)")
    print("-" * 78)
    print(f"   mean_rae       = {mean_rae:.4f}")
    print(f"   PROMOTE  < {GATE_PROMOTE}  ->  {mean_rae < GATE_PROMOTE}")
    print(f"   MARGINAL < {GATE_MARGINAL}  ->  {mean_rae < GATE_MARGINAL}")
    print(f"   verdict  = {verdict}")

    # ---- Save canonical pred_oof at kf_seed=1001 ----
    pred_oof_canon = oof_by_seed[1001].astype(np.float32)
    pred_oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    np.save(pred_oof_path, pred_oof_canon)
    print(f"\n[save] pred_oof @ kf_seed=1001 -> {pred_oof_path}")
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
    print(f"[deploy] te_base blend : mean={pred_base_te.mean():.4f}  "
          f"std={pred_base_te.std():.4f}")
    print(f"[deploy] te_stretched  : mean={te_deploy.mean():.4f}  "
          f"std={te_deploy.std():.4f}")
    print(f"[deploy] te[unb] in-sample RAE = {te_unb_rae:.4f}")

    # ---- Summary ----
    summary = {
        "tag": TAG,
        "method": "nb2902_equal_weight_blend_plus_per_fold_golden_section_rank_stretch",
        "anchors": [
            {"name": ANCHOR_A_NAME, "oof_path": ANCHOR_A_OOF,
             "te_path": ANCHOR_A_TE, "weight": W_A,
             "oof_rae_unb": indiv_oof_rae[ANCHOR_A_NAME]},
            {"name": ANCHOR_B_NAME, "oof_path": ANCHOR_B_OOF,
             "te_path": ANCHOR_B_TE, "weight": W_B,
             "oof_rae_unb": indiv_oof_rae[ANCHOR_B_NAME]},
        ],
        "n_unb": n_unb,
        "n_te": n_te,
        "n_unique_scaffolds": n_unique_scaf,
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "s_lo": S_LO,
        "s_hi": S_HI,
        "gs_tol": GS_TOL,
        "gs_max_iter": GS_MAX_ITER,
        "indiv_oof_rae_unb": indiv_oof_rae,
        "base_blend_pooled_rae": rae_base,
        "base_blend_pred_mean": float(pred_base_oof.mean()),
        "base_blend_pred_std": float(pred_base_oof.std()),
        "truth_std": float(y_unb.std()),
        "variance_compression_ratio": float(pred_base_oof.std() / y_unb.std()),
        "seed_results": seed_results,
        "mean_rae": mean_rae,
        "std_rae": std_rae,
        "min_rae": min_rae,
        "max_rae": max_rae,
        "best_seed": best_seed,
        "delta_mean_vs_base": mean_rae - rae_base,
        "delta_min_vs_base": min_rae - rae_base,
        "promote_thr": GATE_PROMOTE,
        "marginal_thr": GATE_MARGINAL,
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
    print(f"   anchors             = {ANCHOR_A_NAME} + {ANCHOR_B_NAME}")
    print(f"   base blend RAE      = {rae_base:.4f}")
    print(f"   stretch range       = [{S_LO}, {S_HI}]")
    print(f"   mean RAE (5 seeds)  = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   min  RAE            = {min_rae:.4f}  (kf_seed={best_seed})")
    print(f"   delta_mean_vs_base  = {mean_rae - rae_base:+.4f}")
    print(f"   s_deploy            = {s_deploy:.4f} +/- {s_deploy_std:.4f}")
    print(f"   verdict             = {verdict}")
    print(f"   te[unb] in-sample   = {te_unb_rae:.4f}")
    print(f"   wall                = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== KEY ====")
    for k in (
        "base_blend_pooled_rae",
        "mean_rae",
        "std_rae",
        "min_rae",
        "best_seed",
        "delta_mean_vs_base",
        "s_deploy",
        "verdict",
        "te_unb_in_sample_rae",
    ):
        print(f"  {k}: {res.get(k)}")
