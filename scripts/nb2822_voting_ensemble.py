"""nb2822 -- Voting ensemble of 5 model classes on chemprop_aux residual (K=20).

NEW PARADIGM (vs Stacking):
    Instead of learning a meta-learner over base predictions, simply
    AVERAGE the residual predictions from 5 fundamentally different
    model classes:
        1. LGBM (gradient boosted trees, greedy split)
        2. RF   (random forest, bootstrap + greedy split)
        3. ET   (extra trees, bootstrap=False, uniform-random split)
        4. Ridge (L2 linear regression)
        5. SVR-RBF (kernel regression)

    Each model fits the residual target (y - chemprop_aux) on the
    same K=20 feature substrate. The final per-row residual is the
    unweighted mean of the 5 base predictions. The blended prediction
    is anchor + mean_residual.

    Rationale:
    - 5 diverse inductive biases (boosting/bagging/random/linear/kernel)
      decorrelate errors without the variance-amplifying free parameters
      of a meta-learner. The hope is that averaging cancels each class's
      idiosyncratic bias on novel-scaffold rows.
    - No SLSQP/no meta-NN -> 0 fitted blend parameters, no overfit risk
      beyond the residual fits themselves.
    - Distinct from stacking: stacking learns weights (1 per anchor),
      voting uses 1/K weights forever.

PROTOCOL:
    - Anchor: chemprop_aux (PRE-unblind, verified clean).
    - Features: first K=20 cols of X_117 substrate.
    - Residual target = y_unb - anchor_unb.
    - 5-fold scaffold CV across 5 kf_seeds {1001..1005}.
    - For each (kf_seed, fold): fit each of 5 base classes on
      (X_tr_K20, resid_tr); predict residual on X_va; vote-average.
    - Final pred = anchor + vote_mean_residual, clipped [3.0, 8.0].
    - Per-seed pooled RAE; report mean across 5 kf_seeds.
    - Deploy: refit all 5 on all 253; predict 513; vote-average; save.

GATE:
    mean_rae < 0.4570  -> "PROMOTE"
    mean_rae < 0.4598  -> "MARGINAL_BEAT"
    else                -> "FAIL"

Outputs:
    data/processed/nb2822_summary.json
    data/processed/nb2822_pred_oof.npy   (253,) float32
    data/processed/te_nb2822.npy         (513,) float32
    submissions/nb2822_voting_ensemble.csv
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
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2822"

# --------------------------------------------------------------------------
# Substrate
# --------------------------------------------------------------------------
X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"
TE_CHEM_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
K_SLICE = 20

# Outer scaffold CV
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# Output clip range (consistent with K=20 family)
CLIP_LO = 3.0
CLIP_HI = 8.0

# Gates
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

# Model classes (5)
MODEL_NAMES = ["lgbm", "rf", "et", "ridge", "svr_rbf"]


# --------------------------------------------------------------------------
# Base learner factories
# --------------------------------------------------------------------------
def _new_lgbm(seed: int) -> lgb.LGBMRegressor:
    """LGBM hyperparams identical to nb2103 / nb2240 K=20 anchor."""
    return lgb.LGBMRegressor(
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


def _new_rf(seed: int) -> RandomForestRegressor:
    """RandomForest with sensible defaults for n=253 K=20."""
    return RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        bootstrap=True,
        random_state=int(seed),
        n_jobs=-1,
    )


def _new_et(seed: int) -> ExtraTreesRegressor:
    """ExtraTrees hyperparams identical to nb2731 / nb2752."""
    return ExtraTreesRegressor(
        n_estimators=500,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        bootstrap=False,
        random_state=int(seed),
        n_jobs=-1,
    )


def _new_ridge(seed: int) -> Ridge:
    """Ridge regression, alpha=1.0; seed unused (deterministic)."""
    return Ridge(alpha=1.0, random_state=int(seed))


def _new_svr_rbf() -> SVR:
    """SVR with RBF kernel. C=1.0, gamma='scale', epsilon=0.1."""
    return SVR(kernel="rbf", C=1.0, gamma="scale", epsilon=0.1)


# --------------------------------------------------------------------------
# Fit + predict all 5 base learners on a given (X_tr, resid_tr) -> X_va
# --------------------------------------------------------------------------
def _fit_predict_all(X_tr, resid_tr, X_va, seed: int):
    """Fit all 5 base learners on (X_tr, resid_tr) and predict residuals
    on X_va. Returns dict {model_name: (n_va,) array}."""
    out = {}

    # LGBM (raw features, no scaling)
    m = _new_lgbm(seed)
    m.fit(X_tr, resid_tr)
    out["lgbm"] = m.predict(X_va)

    # RF (raw features, no scaling)
    m = _new_rf(seed)
    m.fit(X_tr, resid_tr)
    out["rf"] = m.predict(X_va)

    # ET (raw features, no scaling)
    m = _new_et(seed)
    m.fit(X_tr, resid_tr)
    out["et"] = m.predict(X_va)

    # Ridge + SVR-RBF need standardization on K=20 (mixed scales)
    scaler = StandardScaler()
    X_tr_z = scaler.fit_transform(X_tr)
    X_va_z = scaler.transform(X_va)

    m = _new_ridge(seed)
    m.fit(X_tr_z, resid_tr)
    out["ridge"] = m.predict(X_va_z)

    m = _new_svr_rbf()
    m.fit(X_tr_z, resid_tr)
    out["svr_rbf"] = m.predict(X_va_z)

    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Voting ensemble of 5 model classes (vs Stacking)")
    print("=" * 78)

    # ---- Load test set + scaffolds + truth ----
    te = load_test()
    n_test = len(te)
    smi_col = "smiles" if "smiles" in te.columns else "SMILES"
    name_col = "name" if "name" in te.columns else "Molecule Name"
    te_smiles = te[smi_col].astype(str).tolist()
    te_names = te[name_col].values
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
    print(f"[anchor] chemprop_aux te[unb_idx] RAE = {rae_anchor_unb:.4f}")

    # ---- Residual target ----
    resid_unb = y_unb - anchor_unb
    print(f"[resid] mean={resid_unb.mean():+.3f}  std={resid_unb.std():.3f}  "
          f"min={resid_unb.min():+.2f}  max={resid_unb.max():+.2f}")

    # ---- 5-fold scaffold CV across 5 kf_seeds ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV  seeds={KF_SEEDS}")
    print(f"   models = {MODEL_NAMES}")
    print(f"   voting = unweighted mean (1/{len(MODEL_NAMES)} each)")
    print("-" * 78)

    per_seed_pooled = []
    per_seed_mean_fold = []
    per_seed_per_model = {m: [] for m in MODEL_NAMES}
    oof_seed_stack = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)
    per_seed_summary = []

    for s_idx, kf_seed in enumerate(KF_SEEDS):
        ts = time.time()
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        # Per-model OOF residuals for this kf_seed (n_unb,)
        per_model_oof = {m: np.full(n_unb, np.nan, dtype=np.float64)
                         for m in MODEL_NAMES}
        vote_oof_pred = np.full(n_unb, np.nan, dtype=np.float64)
        fold_rae = []
        for fi, (tr_loc, va_loc) in enumerate(splits):
            X_tr = X_unb[tr_loc]
            X_va = X_unb[va_loc]
            r_tr = resid_unb[tr_loc]
            seed_fit = int(kf_seed + fi)
            preds = _fit_predict_all(X_tr, r_tr, X_va, seed=seed_fit)
            # Stash per-model OOFs
            for m in MODEL_NAMES:
                per_model_oof[m][va_loc] = preds[m]
            # Vote-average residuals -> blended pred
            vote_resid = np.mean(np.column_stack(
                [preds[m] for m in MODEL_NAMES]), axis=1)
            blended = np.clip(anchor_unb[va_loc] + vote_resid,
                              CLIP_LO, CLIP_HI)
            vote_oof_pred[va_loc] = blended
            fold_rae.append(float(rae(y_unb[va_loc], blended)))

        assert not np.isnan(vote_oof_pred).any(), "vote OOF has NaN"
        pooled = float(rae(y_unb, vote_oof_pred))
        mean_fold = float(np.mean(fold_rae))
        per_seed_pooled.append(pooled)
        per_seed_mean_fold.append(mean_fold)
        oof_seed_stack[s_idx] = vote_oof_pred

        # Per-model standalone RAE under this seed
        per_model_rae = {}
        for m in MODEL_NAMES:
            pred_m = np.clip(anchor_unb + per_model_oof[m],
                             CLIP_LO, CLIP_HI)
            per_model_rae[m] = float(rae(y_unb, pred_m))
            per_seed_per_model[m].append(per_model_rae[m])
        model_rae_str = "  ".join(f"{m}={per_model_rae[m]:.4f}"
                                  for m in MODEL_NAMES)
        per_seed_summary.append({
            "kf_seed": int(kf_seed),
            "pooled_vote_rae": pooled,
            "mean_fold_vote_rae": mean_fold,
            "per_model_pooled_rae": per_model_rae,
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   seed={kf_seed}  vote_pooled={pooled:.4f}  "
              f"mean_fold={mean_fold:.4f}")
        print(f"      per_model: {model_rae_str}")

    mean_pooled = float(np.mean(per_seed_pooled))
    std_pooled = float(np.std(per_seed_pooled))
    oof_final = oof_seed_stack.mean(axis=0)
    rae_on_seed_mean = float(rae(y_unb, oof_final))

    per_model_means = {m: float(np.mean(per_seed_per_model[m]))
                       for m in MODEL_NAMES}

    print(f"\n[wide-seed] vote mean pooled = {mean_pooled:.4f} +/- "
          f"{std_pooled:.4f}  (n_seeds={len(KF_SEEDS)})")
    print(f"[wide-seed] pooled on seed-mean vote OOF = {rae_on_seed_mean:.4f}")
    print(f"[wide-seed] per-model mean pooled:")
    for m in MODEL_NAMES:
        print(f"   {m:8s} = {per_model_means[m]:.4f}")

    # ---- Gate ----
    if mean_pooled < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_pooled < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "-" * 78)
    print("GATE EVALUATION")
    print("-" * 78)
    print(f"   mean_pooled_rae = {mean_pooled:.4f}")
    print(f"   gate PROMOTE    = < {GATE_PROMOTE}")
    print(f"   gate MARGINAL   = < {GATE_MARGINAL}")
    print(f"   verdict         = {verdict}")

    # ---- Deploy: refit all 5 on all 253, predict 513, vote-average ----
    print("\n" + "-" * 78)
    print("DEPLOY: refit 5 base learners on all 253, vote-mean on 513")
    print("-" * 78)
    # Tree models on raw X
    deploy_seed = 42
    te_resids = {}

    m = _new_lgbm(deploy_seed)
    m.fit(X_unb, resid_unb)
    te_resids["lgbm"] = m.predict(X_te)

    m = _new_rf(deploy_seed)
    m.fit(X_unb, resid_unb)
    te_resids["rf"] = m.predict(X_te)

    m = _new_et(deploy_seed)
    m.fit(X_unb, resid_unb)
    te_resids["et"] = m.predict(X_te)

    # Ridge + SVR on standardized X (fit scaler on full 253)
    scaler = StandardScaler()
    X_unb_z = scaler.fit_transform(X_unb)
    X_te_z = scaler.transform(X_te)

    m = _new_ridge(deploy_seed)
    m.fit(X_unb_z, resid_unb)
    te_resids["ridge"] = m.predict(X_te_z)

    m = _new_svr_rbf()
    m.fit(X_unb_z, resid_unb)
    te_resids["svr_rbf"] = m.predict(X_te_z)

    vote_te_resid = np.mean(np.column_stack(
        [te_resids[mn] for mn in MODEL_NAMES]), axis=1)
    deploy_te = np.clip(anchor_te + vote_te_resid,
                        CLIP_LO, CLIP_HI).astype(np.float32)
    te_unb_in_rae = float(rae(y_unb, deploy_te[unb_idx].astype(np.float64)))
    print(f"[deploy] te(513) mean={deploy_te.mean():.3f}  "
          f"std={deploy_te.std():.3f}")
    print(f"[deploy] te[unb_idx] in_RAE={te_unb_in_rae:.4f} (in-sample)")

    # ---- Save artefacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_final.astype(np.float32))
    np.save(te_path, deploy_te)
    print(f"\n[save] {oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_voting_ensemble.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": deploy_te,
    }).to_csv(sub_csv, index=False)
    print(f"[save] {sub_csv}")

    summary = {
        "tag": TAG,
        "method": (
            "Voting ensemble: unweighted mean of 5 model classes "
            "(LGBM, RF, ET, Ridge, SVR-RBF) on chemprop_aux residual, "
            "K=20 features."
        ),
        "paradigm": "voting_5model_residual_K20_chemprop_aux",
        "anchor_name": "chemprop_aux (PRE-unblind, verified clean)",
        "rae_anchor_unb": rae_anchor_unb,
        "model_classes": MODEL_NAMES,
        "voting_weights": [1.0 / len(MODEL_NAMES)] * len(MODEL_NAMES),
        "model_hparams": {
            "lgbm": dict(max_depth=4, num_leaves=15, n_estimators=300,
                         learning_rate=0.03, min_child_samples=5,
                         reg_lambda=2.0),
            "rf": dict(n_estimators=500, max_depth=10,
                       min_samples_split=5, min_samples_leaf=2,
                       bootstrap=True),
            "et": dict(n_estimators=500, max_depth=10,
                       min_samples_split=5, min_samples_leaf=2,
                       bootstrap=False),
            "ridge": dict(alpha=1.0, scaled=True),
            "svr_rbf": dict(C=1.0, gamma="scale", epsilon=0.1,
                            kernel="rbf", scaled=True),
        },
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": n_uniq_scaf,
        "feat_dim": int(X_unb.shape[1]),
        "k_slice_first_n_of_117": K_SLICE,
        "clip_lo": CLIP_LO,
        "clip_hi": CLIP_HI,
        "per_seed_summary": per_seed_summary,
        "per_seed_pooled_vote_rae": per_seed_pooled,
        "per_seed_mean_fold_vote_rae": per_seed_mean_fold,
        "per_model_mean_pooled_rae": per_model_means,
        "mean_pooled_rae": mean_pooled,
        "std_pooled_rae": std_pooled,
        "rae_on_seed_mean_oof": rae_on_seed_mean,
        "mean_rae": mean_pooled,
        "delta_vs_anchor": mean_pooled - rae_anchor_unb,
        "te_unb_in_sample_rae": te_unb_in_rae,
        "te_mean": float(deploy_te.mean()),
        "te_std": float(deploy_te.std()),
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "promote": bool(verdict == "PROMOTE"),
        "marginal_beat": bool(verdict == "MARGINAL_BEAT"),
        "oof_path": str(oof_path),
        "te_path": str(te_path),
        "submission_csv": str(sub_csv),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   vote mean pooled RAE      = {mean_pooled:.4f} +/- "
          f"{std_pooled:.4f}  ({verdict})")
    print(f"   delta vs anchor           = {mean_pooled - rae_anchor_unb:+.4f}")
    print(f"   per-model mean pooled RAE:")
    for m in MODEL_NAMES:
        print(f"      {m:8s} = {per_model_means[m]:.4f}")
    print(f"   wall                      = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_pooled_rae",
        "std_pooled_rae",
        "delta_vs_anchor",
        "verdict",
        "te_unb_in_sample_rae",
        "per_model_mean_pooled_rae",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
