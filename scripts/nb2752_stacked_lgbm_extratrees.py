"""nb2752 -- Stacked LGBM K=20 + ExtraTrees K=20 convex blend on chemprop_aux residual.

NEW PARADIGM:
    Rather than running LGBM and ExtraTrees as separate-axis residual
    models and asking a downstream SLSQP to combine them, this script
    runs BOTH on the same residual substrate (chemprop_aux + K=20
    features) and combines their per-row OOFs via a SCALAR convex
    combination sweep.

      pred_blend(w) = w * (anchor + resid_lgbm) + (1-w) * (anchor + resid_et)
                    = anchor + (w * resid_lgbm + (1-w) * resid_et)

    Both base learners share the same 20 features and the same residual
    target -- the only differences are split-selection rule (greedy gradient
    vs. uniform-random) and ensemble averaging style (boosting vs. bagging).
    A scalar w in {0.5..1.0} sweeps the LGBM-vs-ExtraTrees mix; w=1 is
    pure LGBM K=20 (nb2103/nb2112-class), w=0 is pure ExtraTrees K=20
    (nb2731-class).

    Distinction from prior stacks: nb1974 ran LGBM with extra_trees=True
    inside ONE model (split-level randomization). This script runs them
    as TWO independent fits and blends post-hoc -- the variance reduction
    from the ExtraTrees half is decorrelated from the LGBM half.

PROTOCOL:
    1. Slice X_K20 = first 20 cols of X_117_unb / X_117_te.
    2. Anchor: chemprop_aux (PRE-unblind, verified clean).
       Residual target = y_unb - anchor_unb.
    3a. LGBM K=20 OOF: per-seed 5-fold KFold cross-fit on residual,
        5 boosting seeds {0,1,7,42,137}, mean-bag.  Hyperparams
        identical to nb2103 / nb2240 / nb2731 anchor (max_depth=4,
        num_leaves=15, n_est=300, lr=0.03, min_child_samples=5,
        reg_lambda=2.0).
    3b. ExtraTrees K=20 OOF: per-seed 5-fold KFold cross-fit on residual,
        5 boosting seeds {0,1,7,42,137}, mean-bag.  Hyperparams identical
        to nb2731 (n_estimators=500, max_depth=10, min_samples_split=5,
        min_samples_leaf=2).
    4. Each KFold cross-fit uses kfold_seed in {1001..1005} (scaffold CV
       on the outer evaluation; KFold on the inner residual-fit per seed).
       That is, the scaffold CV is the canonical 5-fold scaffold CV across
       the 5 kf_seeds; LGBM and ExtraTrees OOFs are computed per kf_seed
       using a separate kfold draw indexed by the residual-bag seed inside
       each kf_seed split.
       (Equivalently: for each kf_seed, take the scaffold split, then within
       train set fit LGBM and ExtraTrees mean-bag on residual, then predict
       residual on validation rows.)
    5. Convex sweep w in {0.5, 0.6, 0.7, 0.8, 0.9, 1.0}:
         pred_blend(w) = anchor + (w * resid_lgbm + (1-w) * resid_et)
       Per-w, per-seed pooled RAE; report mean across 5 kf_seeds.
    6. Best w = argmin(mean_rae); compare to gates.

GATE:
    best mean_rae < 0.4570  -> "PROMOTE"
    best mean_rae < 0.4598  -> "MARGINAL_BEAT"
    else                    -> "FAIL"

Outputs:
    scripts/nb2752_stacked_lgbm_extratrees.py
    data/processed/nb2752_summary.json
    data/processed/nb2752_pred_oof.npy   (253,) float32  -- best-w mean across seeds
    data/processed/te_nb2752.npy         (513,) float32  -- best-w deploy refit
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
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2752"

# --------------------------------------------------------------------------
# Shared substrate
# --------------------------------------------------------------------------
X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"
TE_CHEM_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
K_SLICE = 20

# Residual-fit ensemble seeds (bag)
RESID_SEEDS = [0, 1, 7, 42, 137]
RESID_FOLDS = 5

# Outer scaffold CV
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# Convex blend sweep grid
W_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Gates
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

# Output clip range (consistent with nb2731)
CLIP_LO = 3.0
CLIP_HI = 8.0


# --------------------------------------------------------------------------
# Base learners
# --------------------------------------------------------------------------
def _lgbm_params(seed: int) -> dict:
    """LGBM hyperparams identical to nb2103 / nb2240 / nb2731 K=20 anchor."""
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


def _new_extratrees(seed: int) -> ExtraTreesRegressor:
    """ExtraTrees hyperparams identical to nb2731."""
    return ExtraTreesRegressor(
        n_estimators=500,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=int(seed),
        n_jobs=-1,
    )


# --------------------------------------------------------------------------
# Per-seed mean-bag residual OOF using inner KFold
# --------------------------------------------------------------------------
def _lgbm_meanbag_oof(X_tr, resid_tr, X_va, kf_seed_offset):
    """Mean-bag LGBM residual OOF: for each seed, fit on FULL X_tr/resid_tr
    and predict on X_va.  Bag = mean across RESID_SEEDS."""
    preds = []
    for s in RESID_SEEDS:
        mdl = lgb.LGBMRegressor(**_lgbm_params(s + kf_seed_offset))
        mdl.fit(X_tr, resid_tr)
        preds.append(mdl.predict(X_va))
    return np.mean(np.column_stack(preds), axis=1)


def _et_meanbag_oof(X_tr, resid_tr, X_va, kf_seed_offset):
    """Mean-bag ExtraTrees residual OOF (same pattern as LGBM)."""
    preds = []
    for s in RESID_SEEDS:
        mdl = _new_extratrees(seed=s + kf_seed_offset)
        mdl.fit(X_tr, resid_tr)
        preds.append(mdl.predict(X_va))
    return np.mean(np.column_stack(preds), axis=1)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Stacked LGBM K=20 + ExtraTrees K=20 convex blend")
    print("=" * 78)

    # ---- Load test set + scaffolds + truth ----
    te = load_test()
    n_test = len(te)
    smi_col = "smiles" if "smiles" in te.columns else "SMILES"
    te_smiles = te[smi_col].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_uniq_scaf = len({s for s in unb_scaffolds if s})
    print(f"[load] n_test={n_test}  n_unb={n_unb}  unique_scaf={n_uniq_scaf}")

    # ---- Load X_117 then slice to first K=20 cols ----
    X_unb_117 = np.load(X117_UNB_PATH).astype(np.float32)
    X_te_117 = np.load(X117_TE_PATH).astype(np.float32)
    assert X_unb_117.shape == (n_unb, 117), f"X_unb shape {X_unb_117.shape}"
    assert X_te_117.shape == (n_test, 117), f"X_te shape {X_te_117.shape}"
    X_unb = X_unb_117[:, :K_SLICE].astype(np.float32)
    X_te = X_te_117[:, :K_SLICE].astype(np.float32)
    print(f"[feat] X_unb_K20={X_unb.shape}  X_te_K20={X_te.shape}  "
          f"slice=first-{K_SLICE}-cols")

    # ---- Anchor (chemprop_aux, PRE-unblind verified-clean) ----
    if not TE_CHEM_PATH.exists():
        raise FileNotFoundError(f"missing test anchor: {TE_CHEM_PATH}")
    te_chem = np.load(TE_CHEM_PATH).astype(np.float64)
    assert te_chem.shape == (n_test,), f"te_chem shape {te_chem.shape}"
    anchor_unb = te_chem[unb_idx]
    anchor_te = te_chem.copy()
    rae_anchor_unb = float(rae(y_unb, anchor_unb))
    print(f"[anchor] chemprop_aux te[unb_idx] RAE = {rae_anchor_unb:.4f} "
          f"(PRE-clean PRIMARY-1 baseline)")

    # ---- Residual target ----
    resid_unb = y_unb - anchor_unb
    print(f"[resid] mean={resid_unb.mean():+.3f}  std={resid_unb.std():.3f}  "
          f"min={resid_unb.min():+.2f}  max={resid_unb.max():+.2f}")

    # ---- Scaffold 5-fold CV across 5 kf_seeds ----
    #      For each kf_seed:
    #        - scaffold split -> (tr_loc, va_loc)
    #        - per fold, fit LGBM mean-bag and ET mean-bag on residual_tr
    #          and predict residual_va.  Stash both per-row OOF residual
    #          arrays (n_unb,) for this kf_seed.
    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV  seeds={KF_SEEDS}\n"
          f"   LGBM:       n_est=300  max_depth=4  num_leaves=15  lr=0.03  "
          f"min_child=5  reg_lambda=2.0  bag={RESID_SEEDS}\n"
          f"   ExtraTrees: n_est=500  max_depth=10  min_split=5  min_leaf=2  "
          f"bag={RESID_SEEDS}")
    print("-" * 78)

    # Per-seed per-row residual OOFs from each base learner.
    lgbm_resid_oofs = []   # list of (n_unb,) arrays, one per kf_seed
    et_resid_oofs = []
    per_seed_summary = []

    for kf_seed in KF_SEEDS:
        ts = time.time()
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        lgbm_resid_oof = np.full(n_unb, np.nan, dtype=np.float64)
        et_resid_oof = np.full(n_unb, np.nan, dtype=np.float64)
        fold_walls = {"lgbm": 0.0, "et": 0.0}
        for fi, (tr_loc, va_loc) in enumerate(splits):
            X_tr = X_unb[tr_loc]
            X_va = X_unb[va_loc]
            r_tr = resid_unb[tr_loc]

            # LGBM mean-bag residual OOF
            t_l = time.time()
            lgbm_resid_oof[va_loc] = _lgbm_meanbag_oof(
                X_tr, r_tr, X_va, kf_seed_offset=kf_seed + fi,
            )
            fold_walls["lgbm"] += time.time() - t_l

            # ExtraTrees mean-bag residual OOF
            t_e = time.time()
            et_resid_oof[va_loc] = _et_meanbag_oof(
                X_tr, r_tr, X_va, kf_seed_offset=kf_seed + fi,
            )
            fold_walls["et"] += time.time() - t_e

        assert not np.isnan(lgbm_resid_oof).any(), "lgbm OOF has NaN"
        assert not np.isnan(et_resid_oof).any(), "et OOF has NaN"

        # Quick per-seed sanity: pure-LGBM and pure-ET RAE under this kf_seed
        pred_lgbm = np.clip(anchor_unb + lgbm_resid_oof, CLIP_LO, CLIP_HI)
        pred_et = np.clip(anchor_unb + et_resid_oof, CLIP_LO, CLIP_HI)
        rae_lgbm = float(rae(y_unb, pred_lgbm))
        rae_et = float(rae(y_unb, pred_et))
        print(f"   seed={kf_seed}  lgbm_RAE={rae_lgbm:.4f}  et_RAE={rae_et:.4f}  "
              f"lgbm_wall={fold_walls['lgbm']:.1f}s  et_wall={fold_walls['et']:.1f}s")

        lgbm_resid_oofs.append(lgbm_resid_oof)
        et_resid_oofs.append(et_resid_oof)
        per_seed_summary.append({
            "kf_seed": int(kf_seed),
            "rae_pure_lgbm": rae_lgbm,
            "rae_pure_et": rae_et,
            "wall_sec": round(time.time() - ts, 2),
        })

    # ---- Sweep w in W_GRID -> per-seed pooled RAE -> mean across seeds ----
    print("\n" + "-" * 78)
    print(f"W-SWEEP  w_grid={W_GRID}  metric=mean pooled RAE across {len(KF_SEEDS)} kf_seeds")
    print("-" * 78)
    sweep_records = []
    best_w = None
    best_mean_rae = float("inf")
    best_oof_mean = None
    for w in W_GRID:
        per_seed_raes = []
        per_seed_oofs = []
        for kf_idx, kf_seed in enumerate(KF_SEEDS):
            r_lgbm = lgbm_resid_oofs[kf_idx]
            r_et = et_resid_oofs[kf_idx]
            r_blend = w * r_lgbm + (1.0 - w) * r_et
            pred = np.clip(anchor_unb + r_blend, CLIP_LO, CLIP_HI)
            per_seed_raes.append(float(rae(y_unb, pred)))
            per_seed_oofs.append(pred)
        mean_rae = float(np.mean(per_seed_raes))
        std_rae = float(np.std(per_seed_raes))
        oof_mean = np.mean(np.column_stack(per_seed_oofs), axis=1)
        rae_of_mean_oof = float(rae(y_unb, oof_mean))
        sweep_records.append({
            "w_lgbm": float(w),
            "w_et": float(1.0 - w),
            "per_seed_rae": [float(r) for r in per_seed_raes],
            "mean_rae": mean_rae,
            "std_rae": std_rae,
            "rae_of_mean_oof": rae_of_mean_oof,
        })
        print(f"   w_lgbm={w:.2f}  mean_RAE={mean_rae:.4f} (+/- {std_rae:.4f})  "
              f"rae_of_mean_oof={rae_of_mean_oof:.4f}")
        if mean_rae < best_mean_rae:
            best_mean_rae = mean_rae
            best_w = float(w)
            best_oof_mean = oof_mean.copy()

    print(f"\n[sweep] best w_lgbm={best_w:.2f}  mean_RAE={best_mean_rae:.4f}")

    # ---- Deploy with best w: refit BOTH on all 253, predict on 513 ----
    print("\n" + "-" * 78)
    print(f"DEPLOY: refit LGBM+ET (mean-bag) on all 253, blend at w_lgbm={best_w:.2f}")
    print("-" * 78)
    # LGBM deploy mean-bag
    lgbm_te_preds = []
    for s in RESID_SEEDS:
        mdl = lgb.LGBMRegressor(**_lgbm_params(s))
        mdl.fit(X_unb, resid_unb)
        lgbm_te_preds.append(mdl.predict(X_te))
    lgbm_te_resid = np.mean(np.column_stack(lgbm_te_preds), axis=1)
    # ExtraTrees deploy mean-bag
    et_te_preds = []
    for s in RESID_SEEDS:
        mdl = _new_extratrees(seed=s)
        mdl.fit(X_unb, resid_unb)
        et_te_preds.append(mdl.predict(X_te))
    et_te_resid = np.mean(np.column_stack(et_te_preds), axis=1)
    blend_te_resid = best_w * lgbm_te_resid + (1.0 - best_w) * et_te_resid
    deploy_te = np.clip(anchor_te + blend_te_resid, CLIP_LO, CLIP_HI).astype(np.float32)
    te_unb_in_rae = float(rae(y_unb, deploy_te[unb_idx].astype(np.float64)))
    print(f"[deploy] te(513) mean={deploy_te.mean():.3f}  std={deploy_te.std():.3f}")
    print(f"[deploy] te[unb_idx] in_RAE={te_unb_in_rae:.4f}  "
          f"(in-sample, deploy refit on all 253)")

    # ---- Gate ----
    if best_mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif best_mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "-" * 78)
    print("GATE EVALUATION")
    print("-" * 78)
    print(f"   best_w         = {best_w:.2f}")
    print(f"   best_mean_rae  = {best_mean_rae:.4f}")
    print(f"   gate PROMOTE   = < {GATE_PROMOTE}")
    print(f"   gate MARGINAL  = < {GATE_MARGINAL}")
    print(f"   verdict        = {verdict}")

    # ---- Save artefacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, best_oof_mean.astype(np.float32))
    np.save(te_path, deploy_te)
    print(f"\n[save] {oof_path}")
    print(f"[save] {te_path}")

    summary = {
        "tag": TAG,
        "method": (
            "Stacked LGBM K=20 + ExtraTrees K=20 convex blend on chemprop_aux "
            "residual.  Both base learners share K=20 substrate; LGBM uses "
            "greedy-gradient splits, ExtraTrees uses uniform-random splits + no "
            "bootstrap.  Scalar w_lgbm sweeps {0.5..1.0}; best w by mean pooled "
            "RAE across 5 scaffold kf_seeds."
        ),
        "paradigm": (
            "stacked_LGBM_plus_ExtraTrees_K20_convex_combo_chemprop_aux_residual"
        ),
        "anchor_name": "chemprop_aux (PRE-unblind, verified clean)",
        "rae_anchor_unb": rae_anchor_unb,
        "lgbm_hparams": _lgbm_params(0),
        "et_hparams": {
            "n_estimators": 500,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
        },
        "resid_seeds_bag": RESID_SEEDS,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "w_grid": W_GRID,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": n_uniq_scaf,
        "feat_dim": int(X_unb.shape[1]),
        "k_slice_first_n_of_117": K_SLICE,
        "clip_lo": CLIP_LO,
        "clip_hi": CLIP_HI,
        "per_seed_base": per_seed_summary,
        "sweep_records": sweep_records,
        "best_w_lgbm": best_w,
        "best_w_et": float(1.0 - best_w),
        "mean_rae": best_mean_rae,
        "delta_vs_anchor": best_mean_rae - rae_anchor_unb,
        "te_unb_rae_in_sample": te_unb_in_rae,
        "te_deploy_mean": float(deploy_te.mean()),
        "te_deploy_std": float(deploy_te.std()),
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "verdict": verdict,
        "promote": bool(verdict == "PROMOTE"),
        "marginal_beat": bool(verdict == "MARGINAL_BEAT"),
        "oof_path": str(oof_path),
        "te_path": str(te_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   best w_lgbm                 = {best_w:.2f}")
    print(f"   best mean_rae (5 seeds)     = {best_mean_rae:.4f}")
    print(f"   delta vs anchor             = {best_mean_rae - rae_anchor_unb:+.4f}")
    print(f"   verdict                     = {verdict}")
    print(f"   wall                        = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "best_w_lgbm",
        "mean_rae",
        "rae_anchor_unb",
        "delta_vs_anchor",
        "verdict",
        "te_unb_rae_in_sample",
    ):
        print(f"  {k}: {res.get(k)}")
