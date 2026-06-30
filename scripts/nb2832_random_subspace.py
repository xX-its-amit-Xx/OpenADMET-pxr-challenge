"""nb2832 -- Random Subspace ensemble on chemprop_aux residual (K=20).

NEW PARADIGM (vs Voting / Stacking / Bagging):
    The Random Subspace Method (Ho 1998): train each member of the
    ensemble on a RANDOM SUBSET of features (NOT a random subset of
    rows like Bagging). Each bag sees a different "slice" of the
    feature space, so the bags are decorrelated in feature-coverage,
    not row-coverage.

    PROTOCOL:
        - 20 LGBM bags
        - Per bag: pick 10 random of K=20 features (uniform without
          replacement) using bag-specific RNG seeded from bag index
        - LGBM(max_depth=4, num_leaves=15, n_est=300, lr=0.03) on
          chemprop_aux RESIDUAL (y_unb - anchor_unb)
        - Per-row residual = MEAN of 20 bag predictions
        - Final pred = anchor + mean_residual, clipped [3.0, 8.0]

    Rationale:
        - Each bag's tree splits live in a different 10-of-20 subspace,
          so each tree's variance manifests on different features.
          Averaging cancels feature-localised variance.
        - Distinct from Bagging (row subsets) and Voting (different
          model classes on same features). Single model class, single
          row set, 20 distinct feature subsets.
        - Zero meta-learner free parameters (uniform 1/20 weights).

    CV: 5-fold scaffold CV on 253 rows, single kf_seed=1001 (per spec).
    Deploy: refit 20 bags on all 253, predict 513, mean-average.

GATE:
    mean_rae < 0.4570  -> "PROMOTE"
    mean_rae < 0.4598  -> "MARGINAL_BEAT"
    else                -> "FAIL"

Outputs:
    data/processed/nb2832_summary.json
    data/processed/nb2832_pred_oof.npy   (253,) float32
    data/processed/te_nb2832.npy         (513,) float32
    submissions/nb2832_random_subspace.csv
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

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2832"

# --------------------------------------------------------------------------
# Substrate
# --------------------------------------------------------------------------
X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"
TE_CHEM_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
K_SLICE = 20             # use first K=20 cols of 117-col substrate
N_BAGS = 20              # 20 random-subspace bags
SUBSPACE_SIZE = 10       # each bag sees 10 of K_SLICE=20 features
SUBSPACE_SEED_BASE = 7777  # base seed for subspace RNG

# Outer scaffold CV
N_FOLDS = 5
KF_SEEDS = [1001]        # single seed per spec

# Output clip range (consistent with K=20 family)
CLIP_LO = 3.0
CLIP_HI = 8.0

# Gates
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598


# --------------------------------------------------------------------------
# Base learner factory
# --------------------------------------------------------------------------
def _new_lgbm(seed: int) -> lgb.LGBMRegressor:
    """LGBM hyperparams per spec: max_depth=4, num_leaves=15, n_est=300, lr=0.03."""
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


# --------------------------------------------------------------------------
# Per-bag random subspace selector
# --------------------------------------------------------------------------
def _bag_subspaces(n_bags: int, k_total: int, subspace_size: int,
                   base_seed: int) -> list[np.ndarray]:
    """For each bag b in [0, n_bags), pick `subspace_size` distinct columns
    from {0..k_total-1} using a deterministic per-bag RNG."""
    out = []
    for b in range(n_bags):
        rng = np.random.default_rng(base_seed + b)
        idx = rng.choice(k_total, size=subspace_size, replace=False)
        idx = np.sort(idx).astype(np.int32)
        out.append(idx)
    return out


# --------------------------------------------------------------------------
# Fit + predict one random-subspace bag on a given (X_tr, resid_tr) -> X_va
# --------------------------------------------------------------------------
def _fit_predict_bag(X_tr, resid_tr, X_va, cols: np.ndarray, seed: int):
    """Fit one LGBM bag on the random subspace (cols) of X_tr; predict X_va."""
    X_tr_sub = X_tr[:, cols]
    X_va_sub = X_va[:, cols]
    m = _new_lgbm(seed)
    m.fit(X_tr_sub, resid_tr)
    return m.predict(X_va_sub)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Random Subspace ensemble (20 LGBM bags, 10-of-20 features)")
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

    # ---- Pre-compute the 20 random subspaces (one per bag) ----
    subspaces = _bag_subspaces(
        n_bags=N_BAGS, k_total=K_SLICE,
        subspace_size=SUBSPACE_SIZE, base_seed=SUBSPACE_SEED_BASE,
    )
    # Coverage stats
    cov = np.zeros(K_SLICE, dtype=np.int32)
    for s in subspaces:
        cov[s] += 1
    print(f"[subspace] n_bags={N_BAGS}  subspace_size={SUBSPACE_SIZE}  "
          f"base_seed={SUBSPACE_SEED_BASE}")
    print(f"[subspace] per-column coverage (count across {N_BAGS} bags):")
    print(f"   min={cov.min()}  max={cov.max()}  mean={cov.mean():.2f}  "
          f"std={cov.std():.2f}")

    # ---- 5-fold scaffold CV across kf_seeds ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV  seeds={KF_SEEDS}")
    print(f"   bags={N_BAGS}  subspace_size={SUBSPACE_SIZE}-of-{K_SLICE}")
    print(f"   per-row aggregation = unweighted mean (1/{N_BAGS} each)")
    print("-" * 78)

    per_seed_pooled = []
    per_seed_mean_fold = []
    per_seed_per_bag = {b: [] for b in range(N_BAGS)}
    oof_seed_stack = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)
    per_seed_summary = []

    for s_idx, kf_seed in enumerate(KF_SEEDS):
        ts = time.time()
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        # Per-bag OOF residuals for this kf_seed (n_unb,)
        per_bag_oof = np.full((N_BAGS, n_unb), np.nan, dtype=np.float64)
        mean_oof_pred = np.full(n_unb, np.nan, dtype=np.float64)
        fold_rae = []
        for fi, (tr_loc, va_loc) in enumerate(splits):
            X_tr = X_unb[tr_loc]
            X_va = X_unb[va_loc]
            r_tr = resid_unb[tr_loc]
            # Fit each of N_BAGS bags on the same fold but different feature subset
            bag_resids_va = np.zeros((N_BAGS, len(va_loc)), dtype=np.float64)
            for b in range(N_BAGS):
                seed_fit = int(kf_seed * 1000 + fi * N_BAGS + b)
                cols_b = subspaces[b]
                bag_resids_va[b] = _fit_predict_bag(
                    X_tr, r_tr, X_va, cols=cols_b, seed=seed_fit,
                )
                per_bag_oof[b, va_loc] = bag_resids_va[b]
            mean_resid_va = bag_resids_va.mean(axis=0)
            blended = np.clip(anchor_unb[va_loc] + mean_resid_va,
                              CLIP_LO, CLIP_HI)
            mean_oof_pred[va_loc] = blended
            fold_rae.append(float(rae(y_unb[va_loc], blended)))

        assert not np.isnan(mean_oof_pred).any(), "mean OOF has NaN"
        pooled = float(rae(y_unb, mean_oof_pred))
        mean_fold = float(np.mean(fold_rae))
        per_seed_pooled.append(pooled)
        per_seed_mean_fold.append(mean_fold)
        oof_seed_stack[s_idx] = mean_oof_pred

        # Per-bag standalone RAE under this seed
        per_bag_rae = {}
        for b in range(N_BAGS):
            pred_b = np.clip(anchor_unb + per_bag_oof[b], CLIP_LO, CLIP_HI)
            per_bag_rae[b] = float(rae(y_unb, pred_b))
            per_seed_per_bag[b].append(per_bag_rae[b])
        per_seed_summary.append({
            "kf_seed": int(kf_seed),
            "pooled_mean_rae": pooled,
            "mean_fold_rae": mean_fold,
            "per_bag_pooled_rae": {str(b): per_bag_rae[b] for b in range(N_BAGS)},
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   seed={kf_seed}  mean_pooled={pooled:.4f}  "
              f"mean_fold={mean_fold:.4f}  wall={time.time() - ts:.1f}s")
        bag_rae_vals = np.array([per_bag_rae[b] for b in range(N_BAGS)])
        print(f"      per-bag standalone RAE: mean={bag_rae_vals.mean():.4f}  "
              f"std={bag_rae_vals.std():.4f}  "
              f"min={bag_rae_vals.min():.4f}  max={bag_rae_vals.max():.4f}")

    mean_pooled = float(np.mean(per_seed_pooled))
    std_pooled = float(np.std(per_seed_pooled))
    oof_final = oof_seed_stack.mean(axis=0)
    rae_on_seed_mean = float(rae(y_unb, oof_final))

    per_bag_means = {b: float(np.mean(per_seed_per_bag[b]))
                     for b in range(N_BAGS)}
    pbm_vals = np.array([per_bag_means[b] for b in range(N_BAGS)])
    print(f"\n[wide-seed] random-subspace mean pooled = {mean_pooled:.4f} +/- "
          f"{std_pooled:.4f}  (n_seeds={len(KF_SEEDS)})")
    print(f"[wide-seed] pooled on seed-mean OOF      = {rae_on_seed_mean:.4f}")
    print(f"[wide-seed] per-bag mean pooled across seeds: "
          f"mean={pbm_vals.mean():.4f}  std={pbm_vals.std():.4f}  "
          f"min={pbm_vals.min():.4f}  max={pbm_vals.max():.4f}")

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

    # ---- Deploy: refit 20 bags on all 253, mean-average on 513 ----
    print("\n" + "-" * 78)
    print(f"DEPLOY: refit {N_BAGS} random-subspace bags on all 253, "
          f"mean-average on 513")
    print("-" * 78)
    deploy_seed_base = 424242
    te_bag_resids = np.zeros((N_BAGS, n_test), dtype=np.float64)
    for b in range(N_BAGS):
        cols_b = subspaces[b]
        seed_fit = int(deploy_seed_base + b)
        X_unb_sub = X_unb[:, cols_b]
        X_te_sub = X_te[:, cols_b]
        m = _new_lgbm(seed_fit)
        m.fit(X_unb_sub, resid_unb)
        te_bag_resids[b] = m.predict(X_te_sub)
    mean_te_resid = te_bag_resids.mean(axis=0)
    deploy_te = np.clip(anchor_te + mean_te_resid,
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

    sub_csv = SUBMISSIONS / f"{TAG}_random_subspace.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": deploy_te,
    }).to_csv(sub_csv, index=False)
    print(f"[save] {sub_csv}")

    summary = {
        "tag": TAG,
        "method": (
            "Random Subspace ensemble: 20 LGBM bags, each on a random "
            f"{SUBSPACE_SIZE}-of-{K_SLICE} feature subset of the K=20 "
            "substrate, per-row mean of 20 bag predictions on the "
            "chemprop_aux residual."
        ),
        "paradigm": ("random_subspace_20bag_lgbm_residual_K20_"
                     "chemprop_aux"),
        "anchor_name": "chemprop_aux (PRE-unblind, verified clean)",
        "rae_anchor_unb": rae_anchor_unb,
        "n_bags": N_BAGS,
        "subspace_size": SUBSPACE_SIZE,
        "subspace_seed_base": SUBSPACE_SEED_BASE,
        "subspace_indices_per_bag": [s.tolist() for s in subspaces],
        "subspace_column_coverage": cov.tolist(),
        "voting_weights": [1.0 / N_BAGS] * N_BAGS,
        "lgbm_hparams": dict(
            objective="regression",
            max_depth=4,
            num_leaves=15,
            n_estimators=300,
            learning_rate=0.03,
            min_child_samples=5,
            reg_lambda=2.0,
        ),
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": n_uniq_scaf,
        "feat_dim_substrate": int(X_unb.shape[1]),
        "feat_dim_per_bag": SUBSPACE_SIZE,
        "k_slice_first_n_of_117": K_SLICE,
        "clip_lo": CLIP_LO,
        "clip_hi": CLIP_HI,
        "per_seed_summary": per_seed_summary,
        "per_seed_pooled_rae": per_seed_pooled,
        "per_seed_mean_fold_rae": per_seed_mean_fold,
        "per_bag_mean_pooled_rae": {str(b): per_bag_means[b]
                                     for b in range(N_BAGS)},
        "per_bag_mean_pooled_stats": {
            "mean": float(pbm_vals.mean()),
            "std": float(pbm_vals.std()),
            "min": float(pbm_vals.min()),
            "max": float(pbm_vals.max()),
        },
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
    print(f"   random-subspace mean pooled RAE = {mean_pooled:.4f} +/- "
          f"{std_pooled:.4f}  ({verdict})")
    print(f"   delta vs anchor                 = "
          f"{mean_pooled - rae_anchor_unb:+.4f}")
    print(f"   per-bag mean pooled RAE         = "
          f"{pbm_vals.mean():.4f} +/- {pbm_vals.std():.4f}")
    print(f"   te[unb_idx] in_sample RAE       = {te_unb_in_rae:.4f}")
    print(f"   wall                            = {time.time() - t0:.1f}s")
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
        "per_bag_mean_pooled_stats",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
