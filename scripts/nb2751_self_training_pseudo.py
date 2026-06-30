"""nb2751 -- Self-training with confident pseudo-labels on still-blind test.

NEW PARADIGM:
    Standard supervised LGBM K=20 on the 4139 train set lives entirely on
    the train manifold. The 513 test compounds are analog-expansion: 253
    have been unblinded, 260 remain blind. Self-training proposes a
    novel substrate enrichment: predict pEC50 on the still-blind 260,
    pick the most CONFIDENT predictions (defined via quantile-head
    spread), promote them to pseudo-labels with weight w_pseudo<1, retrain
    LGBM K=20 on (4139 + top-50 pseudo-labeled), then evaluate on the 253
    unblind via 5-fold scaffold CV.

    The hypothesis is that high-confidence pseudo-labels carry signal
    even when the labels are model-generated, because low-spread cases are
    well-constrained by the existing manifold and provide additional
    near-test-distribution training rows. This sidesteps the
    counter-paradigm cycles 167/169 closed (post-hoc blending exhausted)
    by changing the SUBSTRATE (more training rows in the analog-expansion
    chemistry), not the operator.

    The confidence metric is 1 / (P84 - P16) from three LGBM quantile
    regression heads (alpha=0.16, 0.50, 0.84) -- a one-sigma prediction
    interval. The top-50 most-confident still-blind rows get pseudo-label
    = P50.

PROTOCOL:
    Stage 1: LGBM K=20 quantile heads (alpha=0.16, 0.50, 0.84) on the
        4139 train, 5 seeds {0, 1, 7, 42, 137}, mean-bag per quantile;
        predict on full 513 test.
    Stage 2: Compute confidence = 1 / max(P84 - P16, 1e-3) for the
        still-blind 260 rows (test rows NOT in unb_idx).
        Pick top-50 by confidence; assign pseudo-label = P50 (mean-bag).
    Stage 3: Retrain LGBM K=20 (MSE) on (4139 train + 50 pseudo-labeled),
        5 kf_seeds {1001..1005} x 5-fold scaffold CV on the 253;
        emit pred_oof on 253 and deploy te(513).

GATE (mean_rae = mean pooled RAE across 5 kf_seeds on the 253):
    mean_rae < 0.4570 -> "PROMOTE"
    mean_rae < 0.4598 -> "MARGINAL_BEAT"
    else              -> "FAIL"

OUTPUTS:
    scripts/nb2751_self_training_pseudo.py
    data/processed/nb2751_summary.json
    data/processed/nb2751_pred_oof.npy   (253,) float32 cross-fit OOF
    data/processed/te_nb2751.npy         (513,) float32 deploy
    submissions/nb2751_self_training_pseudo.csv
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
import pandas as pd
import lightgbm as lgb
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, bemis_murcko
from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2751"

# ---- substrate paths ----
X117_TRAIN_PATH = DATA_PROCESSED / "pyramid" / "X_117_train.npy"
X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"
NB2240_SUMMARY = DATA_PROCESSED / "nb2240_summary.json"

UNBLIND_IDX = DATA_PROCESSED / "_audit_unblind_idx.npy"
UNBLIND_Y = DATA_PROCESSED / "_audit_unblind_y.npy"

# ---- self-training params ----
QUANTILE_ALPHAS = [0.16, 0.50, 0.84]
RESID_SEEDS = [0, 1, 7, 42, 137]
N_PSEUDO_LABELS = 50

# ---- scaffold-CV gate eval ----
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# ---- gate thresholds ----
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

CHEMPROP_AUX_REF = 0.6216
NB2240_REF = 0.4676  # nb2171 pyramid ref for context


def _lgbm_params_mse(seed: int) -> dict:
    """LGBM hyperparams used by nb2103/nb2240 residual heads."""
    return dict(
        objective="regression",
        max_depth=4,
        num_leaves=15,
        n_estimators=300,
        learning_rate=0.03,
        min_child_samples=5,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _lgbm_params_quantile(alpha: float, seed: int) -> dict:
    """LGBM quantile head hyperparams.  alpha={0.16, 0.50, 0.84}."""
    return dict(
        objective="quantile",
        alpha=alpha,
        max_depth=4,
        num_leaves=15,
        n_estimators=300,
        learning_rate=0.03,
        min_child_samples=5,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _load_x117_substrate(n_train: int, n_unb: int, n_test: int):
    if not (X117_TRAIN_PATH.exists()
            and X117_UNB_PATH.exists()
            and X117_TE_PATH.exists()):
        raise FileNotFoundError("X_117 cache missing in data/processed/pyramid/")
    X117_train = np.load(X117_TRAIN_PATH).astype(np.float32)
    X117_unb = np.load(X117_UNB_PATH).astype(np.float32)
    X117_te = np.load(X117_TE_PATH).astype(np.float32)
    if X117_train.shape != (n_train, 117):
        raise ValueError(
            f"X117_train shape {X117_train.shape} expected ({n_train},117)"
        )
    if X117_unb.shape != (n_unb, 117):
        raise ValueError(f"X117_unb shape {X117_unb.shape}")
    if X117_te.shape != (n_test, 117):
        raise ValueError(f"X117_te shape {X117_te.shape}")
    X117_train = np.where(np.isfinite(X117_train), X117_train, 0.0).astype(np.float32)
    X117_unb = np.where(np.isfinite(X117_unb), X117_unb, 0.0).astype(np.float32)
    X117_te = np.where(np.isfinite(X117_te), X117_te, 0.0).astype(np.float32)
    return X117_train, X117_unb, X117_te


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- self-training pseudo-labels (top-{N_PSEUDO_LABELS}) on")
    print(f"          still-blind test rows; LGBM K=20 quantile confidence")
    print(f"          alphas={QUANTILE_ALPHAS}  resid_seeds={RESID_SEEDS}")
    print(f"          {N_FOLDS}-fold scaffold-CV across {len(KF_SEEDS)} kf_seeds")
    print(f"          GATE: <{GATE_PROMOTE} PROMOTE / "
          f"<{GATE_MARGINAL} MARGINAL_BEAT / else FAIL")
    print("=" * 78)

    # ---- 1. Load truth + indices ----
    tr = load_train()
    te = load_test()
    tr = tr[tr["pec50"].notna() & tr["smiles"].notna()].reset_index(drop=True)
    n_train = len(tr)
    n_test = len(te)
    test_smiles = (
        te["smiles"].astype(str).tolist()
        if "smiles" in te.columns
        else te["SMILES"].astype(str).tolist()
    )
    te_names = (
        te["name"].values
        if "name" in te.columns
        else te["Molecule Name"].values
    )

    unb_idx = np.load(UNBLIND_IDX).astype(int)
    y_unb = np.load(UNBLIND_Y).astype(np.float64)
    n_unb = len(y_unb)
    still_blind_mask = np.ones(n_test, dtype=bool)
    still_blind_mask[unb_idx] = False
    still_blind_idx = np.where(still_blind_mask)[0]
    n_still_blind = len(still_blind_idx)
    print(f"[load] n_train={n_train}  n_unb={n_unb}  "
          f"n_test={n_test}  n_still_blind={n_still_blind}")

    # ---- 2. Load X_117 substrate + K=20 slice ----
    X117_train, X117_unb, X117_te = _load_x117_substrate(
        n_train=n_train, n_unb=n_unb, n_test=n_test,
    )
    if not NB2240_SUMMARY.exists():
        raise FileNotFoundError(f"missing {NB2240_SUMMARY}")
    with open(NB2240_SUMMARY) as f:
        nb2240 = json.load(f)
    k20_idx = list(nb2240["k20_surviving_idx_in_117"])
    k20_names = list(nb2240["k20_surviving_names"])
    assert len(k20_idx) == 20, f"K=20 expected, got {len(k20_idx)}"
    X_tr = X117_train[:, k20_idx]
    X_unb = X117_unb[:, k20_idx]
    X_te = X117_te[:, k20_idx]
    tr_y = tr["pec50"].to_numpy(dtype=np.float32)
    print(f"[feat] K=20 sliced: X_tr={X_tr.shape}  "
          f"X_unb={X_unb.shape}  X_te={X_te.shape}")

    # ============================================================
    # STAGE 1: quantile heads on 4139 train -> predict 513 test
    # ============================================================
    print("\n" + "-" * 78)
    print(f"STAGE 1: LGBM K=20 quantile heads (alphas={QUANTILE_ALPHAS})")
    print(f"          mean-bag over {len(RESID_SEEDS)} seeds {RESID_SEEDS}")
    print("-" * 78)
    q_preds = {}  # alpha -> mean-bag (n_test,)
    for alpha in QUANTILE_ALPHAS:
        per_seed_te = np.zeros((len(RESID_SEEDS), n_test), dtype=np.float64)
        for i, s in enumerate(RESID_SEEDS):
            ts = time.time()
            mdl = lgb.LGBMRegressor(**_lgbm_params_quantile(alpha, s))
            mdl.fit(X_tr, tr_y)
            per_seed_te[i] = mdl.predict(X_te)
            print(f"   alpha={alpha:.2f}  seed={s:3d}  "
                  f"wall={time.time()-ts:.1f}s")
        q_preds[alpha] = per_seed_te.mean(axis=0)
        print(f"   alpha={alpha:.2f}  mean-bag te(513) "
              f"mean={q_preds[alpha].mean():.3f}  "
              f"std={q_preds[alpha].std():.3f}")

    p16 = q_preds[0.16]
    p50 = q_preds[0.50]
    p84 = q_preds[0.84]
    width = np.maximum(p84 - p16, 1e-3)
    confidence = 1.0 / width

    # ============================================================
    # STAGE 2: pick top-50 most-confident STILL-BLIND pseudo-labels
    # ============================================================
    print("\n" + "-" * 78)
    print(f"STAGE 2: confidence = 1 / (P84-P16); pick top-{N_PSEUDO_LABELS}")
    print(f"          from {n_still_blind} still-blind test rows")
    print("-" * 78)
    # confidence restricted to still-blind subset
    conf_still_blind = confidence[still_blind_idx]
    # top-N indices within still-blind
    if N_PSEUDO_LABELS >= n_still_blind:
        top_within_sb = np.argsort(-conf_still_blind)
    else:
        top_within_sb = np.argpartition(
            -conf_still_blind, kth=N_PSEUDO_LABELS - 1,
        )[:N_PSEUDO_LABELS]
        # sort descending for determinism
        top_within_sb = top_within_sb[np.argsort(-conf_still_blind[top_within_sb])]
    pseudo_test_idx = still_blind_idx[top_within_sb]  # (50,) indices in 0..512
    pseudo_labels = p50[pseudo_test_idx].astype(np.float32)
    pseudo_widths = width[pseudo_test_idx].astype(np.float32)
    pseudo_confidences = confidence[pseudo_test_idx].astype(np.float32)
    print(f"   selected n={len(pseudo_test_idx)} pseudo labels")
    print(f"   width:      med={np.median(pseudo_widths):.3f}  "
          f"min={pseudo_widths.min():.3f}  max={pseudo_widths.max():.3f}")
    print(f"   confidence: med={np.median(pseudo_confidences):.3f}  "
          f"min={pseudo_confidences.min():.3f}  "
          f"max={pseudo_confidences.max():.3f}")
    print(f"   pseudo_lab: med={np.median(pseudo_labels):.3f}  "
          f"mean={pseudo_labels.mean():.3f}  "
          f"std={pseudo_labels.std():.3f}")
    print(f"   p50 vs full-test: full mean={p50.mean():.3f}, "
          f"pseudo mean={pseudo_labels.mean():.3f}")
    # corpus width comparison
    width_still_blind = width[still_blind_idx]
    print(f"   width still-blind corpus med={np.median(width_still_blind):.3f}  "
          f"mean={width_still_blind.mean():.3f}")

    # ============================================================
    # STAGE 3: retrain LGBM K=20 on (4139 train + 50 pseudo); eval 253
    # ============================================================
    print("\n" + "-" * 78)
    print(f"STAGE 3: retrain LGBM K=20 on (4139 + {N_PSEUDO_LABELS} pseudo)")
    print(f"          {N_FOLDS}-fold scaffold-CV on 253 across "
          f"{len(KF_SEEDS)} kf_seeds")
    print("-" * 78)
    X_pseudo = X_te[pseudo_test_idx]
    # augmented train pool (shared across all folds; scaffold CV is on 253 only)
    X_aug = np.concatenate([X_tr, X_pseudo], axis=0)
    y_aug = np.concatenate([tr_y, pseudo_labels.astype(np.float32)], axis=0)
    print(f"   X_aug={X_aug.shape}  y_aug={y_aug.shape}  "
          f"(train={len(X_tr)} + pseudo={len(X_pseudo)})")

    unb_smiles = [test_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique unb scaffolds = {n_unique_scaf}")

    per_seed_results = []
    all_oofs = []  # list of (n_unb,) arrays, one per kf_seed
    for kf_seed in KF_SEEDS:
        ts_seed = time.time()
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        pred_oof_seed = np.full(n_unb, np.nan, dtype=np.float64)
        for fold_i, (tr_loc, va_loc) in enumerate(splits):
            # pool: augmented 4139+50 + fold-train portion of 253
            X_pool = np.concatenate([X_aug, X_unb[tr_loc]], axis=0)
            y_pool = np.concatenate(
                [y_aug, y_unb[tr_loc].astype(np.float32)], axis=0,
            )
            # mean-bag over 5 model seeds for this fold
            preds_va = np.zeros(len(va_loc), dtype=np.float64)
            for s in RESID_SEEDS:
                mdl = lgb.LGBMRegressor(**_lgbm_params_mse(s))
                mdl.fit(X_pool, y_pool)
                preds_va += mdl.predict(X_unb[va_loc])
            preds_va /= len(RESID_SEEDS)
            pred_oof_seed[va_loc] = preds_va
        if np.isnan(pred_oof_seed).any():
            raise RuntimeError(f"kf_seed={kf_seed}: pred_oof has NaN")
        rae_seed = float(rae(y_unb, pred_oof_seed))
        per_seed_results.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": float(rae_seed),
        })
        all_oofs.append(pred_oof_seed)
        print(f"   kf_seed={kf_seed}  pooled_RAE={rae_seed:.4f}  "
              f"wall={time.time()-ts_seed:.1f}s")

    raes = np.asarray([r["pooled_rae"] for r in per_seed_results])
    mean_rae = float(raes.mean())
    std_rae = float(raes.std())
    min_rae = float(raes.min())
    max_rae = float(raes.max())
    mean_of_seed_oofs = np.mean(np.column_stack(all_oofs), axis=1)
    rae_of_mean_seed_oofs = float(rae(y_unb, mean_of_seed_oofs))
    print(f"\n[cv] mean across {len(KF_SEEDS)} kf_seeds = "
          f"{mean_rae:.4f} (+/- {std_rae:.4f})  "
          f"range=[{min_rae:.4f}, {max_rae:.4f}]")
    print(f"[cv] RAE of mean-of-seed OOFs = {rae_of_mean_seed_oofs:.4f}")
    print(f"     vs chemprop_aux ref     = {CHEMPROP_AUX_REF:.4f}  "
          f"(d {mean_rae - CHEMPROP_AUX_REF:+.4f})")
    print(f"     vs nb2240/nb2171 ref    = {NB2240_REF:.4f}  "
          f"(d {mean_rae - NB2240_REF:+.4f})")

    # ---- Deploy: train on aug + ALL 253 unblind -> predict 513 ----
    print("\n[deploy] train on (4139 + 50 pseudo + 253 unblind) -> predict 513")
    X_pool_full = np.concatenate([X_aug, X_unb], axis=0)
    y_pool_full = np.concatenate(
        [y_aug, y_unb.astype(np.float32)], axis=0,
    )
    te_deploy = np.zeros(n_test, dtype=np.float64)
    for s in RESID_SEEDS:
        mdl = lgb.LGBMRegressor(**_lgbm_params_mse(s))
        mdl.fit(X_pool_full, y_pool_full)
        te_deploy += mdl.predict(X_te)
    te_deploy /= len(RESID_SEEDS)
    te_deploy = te_deploy.astype(np.float32)
    te_unb_rae = float(rae(y_unb, te_deploy[unb_idx]))
    print(f"   te(513) mean/std = {te_deploy.mean():.3f}/{te_deploy.std():.3f}")
    print(f"   te[unb_idx] in-sample RAE = {te_unb_rae:.4f}  "
          f"(expected: in-sample optimism vs {mean_rae:.4f})")

    # ---- Save artefacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    # save mean-of-seed-oofs as the canonical cross-fit OOF
    np.save(oof_path, mean_of_seed_oofs.astype(np.float32))
    np.save(te_path, te_deploy)
    print(f"\n[save] {oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_self_training_pseudo.csv"
    pd.DataFrame({
        "SMILES": test_smiles,
        "Molecule Name": te_names,
        "pEC50": te_deploy,
    }).to_csv(sub_csv, index=False)
    print(f"[save] {sub_csv}")

    # ---- Gate ----
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "=" * 78)
    print("GATE EVALUATION")
    print("=" * 78)
    print(f"   mean_rae          = {mean_rae:.4f}")
    print(f"   < {GATE_PROMOTE:.4f}  (PROMOTE)        = "
          f"{mean_rae < GATE_PROMOTE}")
    print(f"   < {GATE_MARGINAL:.4f}  (MARGINAL_BEAT)  = "
          f"{mean_rae < GATE_MARGINAL}")
    print(f"   VERDICT           = {verdict}")

    summary = {
        "tag": TAG,
        "method": "self_training_confident_pseudo_labels_top50",
        "rationale": (
            "Stage 1 LGBM K=20 quantile heads (alpha 0.16/0.50/0.84) on 4139 "
            "train; stage 2 pick top-50 most-confident still-blind test rows "
            "(confidence = 1/(P84-P16)) and assign P50 as pseudo-label; "
            "stage 3 retrain LGBM K=20 MSE on (4139 + 50 pseudo) and "
            "evaluate via 5-fold scaffold-CV on 253 across 5 kf_seeds."
        ),
        # stage 1 quantile heads
        "quantile_alphas": QUANTILE_ALPHAS,
        "resid_seeds": RESID_SEEDS,
        "n_train": int(n_train),
        "n_unb": int(n_unb),
        "n_test": int(n_test),
        "n_still_blind": int(n_still_blind),
        "n_unique_unb_scaffolds": int(n_unique_scaf),
        "k20_idx_source": str(NB2240_SUMMARY),
        "k20_surviving_idx_in_117": [int(j) for j in k20_idx],
        "k20_surviving_names": k20_names,
        # stage 2 pseudo-labels
        "n_pseudo_labels": int(N_PSEUDO_LABELS),
        "pseudo_test_idx": [int(j) for j in pseudo_test_idx],
        "pseudo_labels": [float(v) for v in pseudo_labels],
        "pseudo_widths": [float(v) for v in pseudo_widths],
        "pseudo_confidences": [float(v) for v in pseudo_confidences],
        "pseudo_labels_mean": float(pseudo_labels.mean()),
        "pseudo_labels_std": float(pseudo_labels.std()),
        "pseudo_widths_median": float(np.median(pseudo_widths)),
        "pseudo_widths_mean": float(pseudo_widths.mean()),
        "pseudo_widths_min": float(pseudo_widths.min()),
        "pseudo_widths_max": float(pseudo_widths.max()),
        "still_blind_widths_median": float(np.median(width_still_blind)),
        "still_blind_widths_mean": float(width_still_blind.mean()),
        # stage 3 scaffold-CV
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "per_seed_results": per_seed_results,
        "mean_rae": mean_rae,
        "std_rae": std_rae,
        "min_rae": min_rae,
        "max_rae": max_rae,
        "rae_of_mean_of_seed_oofs": rae_of_mean_seed_oofs,
        "delta_vs_chemprop_aux": mean_rae - CHEMPROP_AUX_REF,
        "delta_vs_nb2240_ref": mean_rae - NB2240_REF,
        # deploy
        "te_unb_in_sample_rae": te_unb_rae,
        "deploy_te_mean": float(te_deploy.mean()),
        "deploy_te_std": float(te_deploy.std()),
        # gate
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "gate_promote": bool(mean_rae < GATE_PROMOTE),
        "gate_marginal_beat": bool(mean_rae < GATE_MARGINAL),
        "verdict": verdict,
        # artefacts
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   mean_rae (5 kf_seeds, 5-fold scaffold-CV) = {mean_rae:.4f}")
    print(f"   std_rae across seeds                      = {std_rae:.4f}")
    print(f"   range                                     = "
          f"[{min_rae:.4f}, {max_rae:.4f}]")
    print(f"   delta vs chemprop_aux ({CHEMPROP_AUX_REF:.4f})       = "
          f"{mean_rae - CHEMPROP_AUX_REF:+.4f}")
    print(f"   verdict                                   = {verdict}")
    print(f"   wall                                      = "
          f"{time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae",
        "std_rae",
        "min_rae",
        "max_rae",
        "rae_of_mean_of_seed_oofs",
        "pseudo_labels_mean",
        "pseudo_labels_std",
        "pseudo_widths_median",
        "still_blind_widths_median",
        "te_unb_in_sample_rae",
        "verdict",
        "gate_promote",
        "gate_marginal_beat",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
