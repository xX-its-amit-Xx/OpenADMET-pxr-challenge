"""nb2503 -- Residual-of-Residual ResNet style stacking on chemprop_aux residual.

CONTEXT:
    Cycle 167+ post-hoc-blend ceiling on chemprop_aux anchor is 0.4682 (nb2171
    deep-30) and the K=20-anchored anchor nb2240 sits at OOF RAE 0.4630 on the
    253 unblind.  Most post-hoc moves saturate; per-anchor residual mean-bag
    has already taken signal out at K=20. This script tests whether a
    *recursive* residual model (ResNet style) can squeeze additional signal by
    stacking THREE K=20 LGBM mean-bags, each fit on the residual of the previous
    level.

PROTOCOL:
    Level 0:  nb2240 anchor (chemprop_aux + K=20 LGBM mean-bag)
    Level 1:  K=20 LGBM mean-bag on residual r1 = y - level0_oof,
              features X_117 (the 117-col canonical pyramid matrix, NOT the
              20-RFE slice; the residual still has signal in the dropped 97
              columns).
    Level 2:  K=20 LGBM mean-bag on residual r2 = y - level0 - level1_oof
    Level 3:  K=20 LGBM mean-bag on residual r3 = y - level0 - level1 - level2_oof
    Final  =  alpha_0 * level0 + alpha_1 * level1 + alpha_2 * level2 + alpha_3 * level3
              with alpha_i >= 0, sum = 1, and monotonic shrinkage
              alpha_0 >= alpha_1 >= alpha_2 >= alpha_3  (each subsequent
              residual should contribute LESS, otherwise the stack is
              overfitting the noise tail). Fit by SLSQP with explicit
              monotonicity inequality constraints.
    5-fold scaffold CV across 5 kf_seeds {1001..1005}; deep K-bag mean uses
    K=20 inner seeds for the LGBM bag at each level.

GATE: mean_rae < 0.4570 -> "PROMOTE"; <0.4601 -> "MARGINAL_BEAT"; else "FAIL".

Outputs:
    scripts/nb2503_resnet_residual_stack.py
    data/processed/nb2503_summary.json
    data/processed/nb2503_pred_oof.npy   (253,) float32
    data/processed/te_nb2503.npy         (513,) float32
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import lightgbm as lgb
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2503"

# -----------------------------
# Config
# -----------------------------
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# K=20 inner bag seeds for each level's LGBM.
# (Task spec says K=20; the inner bag plus per-level deploy refit dominates
# runtime: outer 5 seeds * 5 folds * 3 levels * K * (inner_folds + 1) LGBM
# fits = 5*5*3*20*6 = 9000 fits ~ 30-50 min on this box. K=20 fixed per spec.)
BAG_SEEDS = [0, 1, 7, 13, 17, 21, 29, 33, 37, 41,
             42, 47, 53, 59, 61, 67, 71, 79, 83, 137]
assert len(BAG_SEEDS) == 20, f"expected K=20 bag, got {len(BAG_SEEDS)}"

# Inner KFold for the per-level residual cross-fit (independent of the
# outer scaffold-CV; each level uses random KFold inside the outer-train,
# same idiom as nb2240).
INNER_FOLDS = 5
INNER_SEED_OFFSET = 10000   # used to derive inner seeds deterministically

N_LEVELS = 3                # plus level 0 = 4 components total

GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4601

ANCHOR_OOF_PATH = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_nb2240.npy"
X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"


def _lgbm_params(seed):
    return dict(
        objective="regression",
        max_depth=4,
        num_leaves=15,
        n_estimators=300,
        learning_rate=0.03,
        min_child_samples=5,
        reg_lambda=2.0,
        random_state=int(seed),
        n_jobs=2,
        verbosity=-1,
    )


# ============================================================================
# Per-level mean-bag residual cross-fit
# ============================================================================

def level_cross_fit_mean_bag(X_unb, residual, X_te, outer_kf_seed):
    """Fit K=20 LGBM bag inside an inner KFold; return (oof_resid (n_unb,),
    te_resid (n_te,)). Each bag seed gets its own KFold partition derived
    from outer_kf_seed + BAG_SEEDS[i] + INNER_SEED_OFFSET so cross-fits are
    not coincident across seeds. te is the mean across bag of a *deploy*
    model refit on the full local-train (n_unb rows)."""
    n_unb = len(residual)
    n_te = X_te.shape[0]
    bag_oof = np.zeros((len(BAG_SEEDS), n_unb), dtype=np.float64)
    bag_te = np.zeros((len(BAG_SEEDS), n_te), dtype=np.float64)
    for bi, s in enumerate(BAG_SEEDS):
        inner_seed = int(outer_kf_seed) + int(s) + INNER_SEED_OFFSET
        kf = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=inner_seed)
        oof_s = np.full(n_unb, np.nan, dtype=np.float64)
        for tr_loc, va_loc in kf.split(np.arange(n_unb)):
            mdl = lgb.LGBMRegressor(**_lgbm_params(inner_seed))
            mdl.fit(X_unb[tr_loc], residual[tr_loc])
            oof_s[va_loc] = mdl.predict(X_unb[va_loc])
        bag_oof[bi] = oof_s
        # deploy refit on full n_unb -> predict on te
        mdl = lgb.LGBMRegressor(**_lgbm_params(inner_seed))
        mdl.fit(X_unb, residual)
        bag_te[bi] = mdl.predict(X_te).astype(np.float64)
    return bag_oof.mean(axis=0), bag_te.mean(axis=0)


# ============================================================================
# Monotonic-simplex SLSQP   alpha_0 >= alpha_1 >= alpha_2 >= alpha_3
# ============================================================================

def slsqp_monotonic_simplex(P, y, init=None):
    K = P.shape[1]
    cons = [
        {"type": "eq", "fun": lambda w: w.sum() - 1.0},
    ]
    # monotonic shrinkage: w_i - w_{i+1} >= 0
    for i in range(K - 1):
        cons.append({
            "type": "ineq",
            "fun": (lambda w, i=i: w[i] - w[i + 1]),
        })
    bnds = [(0.0, 1.0)] * K
    if init is None:
        # decaying init (50, 25, 12.5, 12.5) so monotonicity already holds
        base = np.array([2.0 ** -i for i in range(K)], dtype=np.float64)
        init = base / base.sum()
    res = minimize(
        lambda w: float(np.sum((P @ w - y) ** 2)),
        init,
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
        options={"ftol": 1e-10, "maxiter": 1000},
    )
    w = np.clip(res.x, 0.0, 1.0)
    s = w.sum()
    if s <= 0:
        w = init.copy()
    else:
        w = w / s
    # Re-enforce monotonicity defensively (clip-cascade)
    for i in range(K - 1):
        if w[i + 1] > w[i]:
            w[i + 1] = w[i]
    s = w.sum()
    return w / s if s > 0 else init


# ============================================================================
# Outer scaffold-CV: build the 4-column level stack on the OOF rows,
# fit monotonic-simplex weights inside each train fold, predict on the va fold.
# ============================================================================

def cv_run_for_seed(X_unb, X_te, y_unb, anchor_oof, anchor_te,
                    unb_scaffolds, kf_seed):
    n_unb = len(y_unb)
    n_te = X_te.shape[0]
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )

    oof_levels = np.zeros((n_unb, N_LEVELS + 1), dtype=np.float64)
    te_levels = np.zeros((n_te, N_LEVELS + 1), dtype=np.float64)
    # Level 0 is fixed across folds (it's nb2240 which is itself cross-fit).
    oof_levels[:, 0] = anchor_oof
    te_levels[:, 0] = anchor_te

    oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
    te_blend_per_fold = np.zeros((N_FOLDS, n_te), dtype=np.float64)
    fold_w = []

    for fi, (tr_loc, va_loc) in enumerate(splits):
        # For each later level, do a per-fold inner cross-fit on tr only.
        # Each call returns: oof_tr_lv (residual cross-fit on tr rows),
        # pred_va_lv (deploy on va), pred_te_lv (deploy on te).
        # We stack tr+va+te in a single te-extension to share one
        # level_cross_fit_mean_bag call per level.
        cum_pred_tr = anchor_oof[tr_loc].copy()
        n_tr = len(tr_loc)
        n_va = len(va_loc)

        cum_preds_tr_levels = np.zeros((n_tr, N_LEVELS + 1), dtype=np.float64)
        cum_preds_tr_levels[:, 0] = anchor_oof[tr_loc]
        cum_preds_va_levels = np.zeros((n_va, N_LEVELS + 1), dtype=np.float64)
        cum_preds_va_levels[:, 0] = anchor_oof[va_loc]
        cum_preds_te_levels = np.zeros((n_te, N_LEVELS + 1), dtype=np.float64)
        cum_preds_te_levels[:, 0] = anchor_te
        cum_pred_va = anchor_oof[va_loc].copy()
        cum_pred_te = anchor_te.copy()

        for lv in range(1, N_LEVELS + 1):
            resid_tr = y_unb[tr_loc] - cum_pred_tr
            # te-extension: concatenate va + te so a single bag-deploy run
            # gives predictions on both.
            X_va_plus_te = np.concatenate([X_unb[va_loc], X_te], axis=0)
            oof_tr_lv, va_plus_te_lv = level_cross_fit_mean_bag(
                X_unb[tr_loc],
                resid_tr,
                X_va_plus_te,
                outer_kf_seed=kf_seed * 100 + fi * 10 + lv,
            )
            pred_va_lv = va_plus_te_lv[:n_va]
            pred_te_lv = va_plus_te_lv[n_va:]
            cum_pred_tr = cum_pred_tr + oof_tr_lv
            cum_pred_va = cum_pred_va + pred_va_lv
            cum_pred_te = cum_pred_te + pred_te_lv
            cum_preds_tr_levels[:, lv] = cum_pred_tr
            cum_preds_va_levels[:, lv] = cum_pred_va
            cum_preds_te_levels[:, lv] = cum_pred_te

        # Monotonic SLSQP fit on cum-preds (each column = sum_{0..lv})
        # Final = sum alpha_lv * cum_pred_lv, alpha simplex with monotonic
        # shrinkage (alpha_0 >= alpha_1 >= alpha_2 >= alpha_3).
        w = slsqp_monotonic_simplex(cum_preds_tr_levels, y_unb[tr_loc])
        oof_blend[va_loc] = cum_preds_va_levels @ w
        te_blend_per_fold[fi] = cum_preds_te_levels @ w
        fold_w.append(w)

    pooled_rae = float(rae(y_unb, oof_blend))
    te_blend = te_blend_per_fold.mean(axis=0)
    return pooled_rae, oof_blend, te_blend, fold_w


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Residual-of-Residual ResNet stack (3 levels) on X_117")
    print("=" * 78)

    # --- Load test set, scaffolds, truth, anchor ---
    te = load_test()
    n_test = len(te)
    te_smiles = (
        te["smiles"].astype(str).tolist()
        if "smiles" in te.columns
        else te["SMILES"].astype(str).tolist()
    )
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_uniq_scaf = len({s for s in unb_scaffolds if s})
    print(f"[load] n_test={n_test}  n_unb={n_unb}  unique_scaf={n_uniq_scaf}")

    X_unb = np.load(X117_UNB_PATH).astype(np.float32)
    X_te = np.load(X117_TE_PATH).astype(np.float32)
    print(f"[feat] X_unb={X_unb.shape}  X_te={X_te.shape}")
    assert X_unb.shape == (n_unb, 117) and X_te.shape == (n_test, 117)

    anchor_oof = np.load(ANCHOR_OOF_PATH).astype(np.float64)
    anchor_te = np.load(ANCHOR_TE_PATH).astype(np.float64)
    assert anchor_oof.shape == (n_unb,) and anchor_te.shape == (n_test,)
    rae_anchor = float(rae(y_unb, anchor_oof))
    print(f"[anchor] nb2240 oof RAE = {rae_anchor:.4f}")
    print(f"[bag] K={len(BAG_SEEDS)} per level, levels={N_LEVELS}, "
          f"outer kf_seeds={KF_SEEDS}")

    # ---- Scaffold 5-fold CV across 5 kf_seeds ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV (seeds={KF_SEEDS})")
    print("-" * 78)
    per_seed = []
    all_oofs = []
    all_tes = []
    for kf_seed in KF_SEEDS:
        ts = time.time()
        pooled, oof_blend, te_blend, fw = cv_run_for_seed(
            X_unb, X_te, y_unb, anchor_oof, anchor_te, unb_scaffolds, kf_seed,
        )
        mean_w = np.mean(np.array(fw), axis=0).tolist()
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": float(pooled),
            "fold_w_mean": [float(x) for x in mean_w],
            "wall_sec": round(time.time() - ts, 2),
        })
        all_oofs.append(oof_blend)
        all_tes.append(te_blend)
        print(
            f"   seed={kf_seed}  pooled_RAE={pooled:.4f}  "
            f"w_mean={np.round(mean_w, 3).tolist()}  "
            f"wall={time.time()-ts:.1f}s"
        )

    pooled_mean = float(np.mean([r["pooled_rae"] for r in per_seed]))
    pooled_std = float(np.std([r["pooled_rae"] for r in per_seed]))
    final_oof = np.mean(np.column_stack(all_oofs), axis=1)
    final_te = np.mean(np.column_stack(all_tes), axis=1).astype(np.float32)
    final_oof_rae = float(rae(y_unb, final_oof))
    te_unb_rae = float(rae(y_unb, final_te[unb_idx]))

    print(f"\n[oof] pooled RAE mean across seeds = {pooled_mean:.4f}  "
          f"(+/- {pooled_std:.4f})")
    print(f"[oof] RAE of mean-of-seed OOFs     = {final_oof_rae:.4f}")
    print(f"[oof] delta vs anchor              = {pooled_mean - rae_anchor:+.4f}")
    print(f"[te] te[unb_idx] RAE                = {te_unb_rae:.4f}")
    print(f"[te] te(513) mean/std               = {final_te.mean():.3f}/"
          f"{final_te.std():.3f}")

    # Gate
    if pooled_mean < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif pooled_mean < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "-" * 78)
    print("GATE EVALUATION")
    print("-" * 78)
    print(f"   mean_rae       = {pooled_mean:.4f}")
    print(f"   gate PROMOTE   = < {GATE_PROMOTE}")
    print(f"   gate MARGINAL  = < {GATE_MARGINAL}")
    print(f"   verdict        = {verdict}")

    # Save artefacts
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, final_oof.astype(np.float32))
    np.save(te_path, final_te.astype(np.float32))
    print(f"[save] {oof_path}")
    print(f"[save] {te_path}")

    summary = {
        "tag": TAG,
        "method": (
            "ResNet-style 3-level residual stacking on X_117 with "
            "K=20 LGBM mean-bag per level + SLSQP monotonic-simplex blend"
        ),
        "anchor_name": "nb2240_mean_bag_oof_K20",
        "rae_anchor": rae_anchor,
        "n_levels_after_anchor": N_LEVELS,
        "n_components_total": N_LEVELS + 1,
        "bag_seeds": BAG_SEEDS,
        "K_bag": len(BAG_SEEDS),
        "inner_folds": INNER_FOLDS,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_unb": n_unb,
        "n_te": n_test,
        "n_unique_scaffolds": n_uniq_scaf,
        "feat_dim": int(X_unb.shape[1]),
        "per_seed": per_seed,
        "mean_rae": pooled_mean,
        "pooled_rae_std_seeds": pooled_std,
        "rae_of_mean_of_seed_oofs": final_oof_rae,
        "delta_vs_anchor": pooled_mean - rae_anchor,
        "te_unb_rae_in_sample": te_unb_rae,
        "te_deploy_mean": float(final_te.mean()),
        "te_deploy_std": float(final_te.std()),
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "verdict": verdict,
        "oof_path": str(oof_path),
        "te_path": str(te_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   mean_rae (5 seeds)            = {pooled_mean:.4f} "
          f"(+/- {pooled_std:.4f})")
    print(f"   delta vs anchor (nb2240)      = {pooled_mean - rae_anchor:+.4f}")
    print(f"   verdict                       = {verdict}")
    print(f"   wall                          = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae",
        "pooled_rae_std_seeds",
        "rae_of_mean_of_seed_oofs",
        "delta_vs_anchor",
        "verdict",
        "te_unb_rae_in_sample",
    ):
        print(f"  {k}: {res.get(k)}")
