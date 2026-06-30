"""nb2553 -- LGBMRanker (lambdarank) + post-hoc value calibration.

NEW PARADIGM:
    Predict ORDERING, not values.  LightGBM lambdarank optimizes a pairwise
    ranking surrogate (NDCG), which is a different inductive bias than the
    L2-regression objective used by every other LGBM on this anchor:
      - decision function is rank-preserving (monotone wrt latent score)
      - loss is invariant to monotone transforms of y
      - the SHAP/feature interactions are not tied to absolute pEC50 scale
    After fitting on integer ranks {0..N-1}, post-hoc calibration maps the
    predicted ranks back to pEC50 by looking up the fold-train empirical CDF
    of y at the val-row's rank-quantile.  This decouples ordinal learning
    from value calibration entirely.

PROTOCOL:
    - Features  : X_117 from data/processed/pyramid/X_117_unb.npy (+ X_117_te.npy)
    - Anchor    : chemprop_aux NOT used as a residual base (the ranker is a
                  STANDALONE candidate -- different paradigm from anchor+resid).
                  We still report rae_anchor_unb for reference.
    - Model     : LGBMRanker(objective='lambdarank', metric='ndcg',
                  max_depth=4, num_leaves=15, n_estimators=300,
                  learning_rate=0.05, random_state=...)
    - Single-group per fold (all rows in one group; pairwise pairs are
      formed across the entire fold-train).
    - Relevance : integer bin of y's rank quantile, 0..N_BINS-1
                  (LGBMRanker label_gain table is bounded; binned relevance
                  avoids the "Label X not less than label mappings" overflow
                  while preserving the ordinal signal at fine granularity).
    - 5-fold scaffold CV on 253 unblind, kf_seeds {1001..1005}
    - Post-hoc value calibration:
          score_val  = ranker.predict(X_val)
          rank_quant_val = rank(score_val) / (n_val - 1)          # in [0,1]
          oof_pred  = quantile(y_train, q=rank_quant_val)         # train ECDF lookup
    - Deploy: refit on ALL 253 -> predict on 513, calibrate against 253 ECDF.

GATE: mean_rae < 0.4570 -> "PROMOTE"
      mean_rae < 0.4601 -> "MARGINAL_BEAT"
      else            -> "FAIL"

Outputs:
    scripts/nb2553_lgbm_rank_objective.py
    data/processed/nb2553_summary.json
    data/processed/nb2553_pred_oof.npy   (253,) float32
    data/processed/te_nb2553.npy         (513,) float32
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
from lightgbm import LGBMRanker

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2553"

# -----------------------------
# LGBMRanker hyperparameters (spec)
# -----------------------------
RNK_MAX_DEPTH = 4
RNK_NUM_LEAVES = 15
RNK_N_EST = 300
RNK_LR = 0.05
RNK_BASE_SEED = 42

# LGBMRanker default label_gain table covers 0..30 (31 entries).  Bin the
# fold-train rank into N_BINS levels so labels stay <= max_label_gain.
N_BINS = 31

# CV protocol
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# Gates
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4601

# Paths
X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"
TE_CHEM_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"     # (513,) PRE-clean anchor


def _new_ranker(seed: int) -> "LGBMRanker":
    return LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        max_depth=RNK_MAX_DEPTH,
        num_leaves=RNK_NUM_LEAVES,
        n_estimators=RNK_N_EST,
        learning_rate=RNK_LR,
        random_state=int(seed),
        n_jobs=2,
        verbose=-1,
        # Provide an explicit label_gain table sized for N_BINS so
        # binned relevance labels (0..N_BINS-1) are valid.
        label_gain=list(range(N_BINS)),
    )


def _rank_to_bin(y: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Map raw y values to integer rank-bins 0..n_bins-1 (ties stable)."""
    n = len(y)
    order = np.argsort(y, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    # Quantile in (0, 1) for binning
    q = ranks / max(n - 1, 1)
    bins = np.minimum(
        (q * n_bins).astype(np.int32), np.int32(n_bins - 1)
    )
    return bins


def _rank_to_quantile(scores: np.ndarray) -> np.ndarray:
    """Convert raw ranker scores to rank-quantiles in [0, 1]."""
    n = len(scores)
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    return ranks / (n - 1)


def _calibrate_to_train_ecdf(y_train: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Map rank-quantiles `q` in [0,1] to values via train empirical CDF."""
    return np.quantile(y_train, q, method="linear").astype(np.float64)


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- LGBMRanker (lambdarank) + post-hoc value calibration (X_117)")
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

    # ---- Load features ----
    X_unb = np.load(X117_UNB_PATH).astype(np.float32)
    X_te = np.load(X117_TE_PATH).astype(np.float32)
    print(f"[feat] X_unb={X_unb.shape}  X_te={X_te.shape}")
    assert X_unb.shape == (n_unb, 117), f"X_unb shape {X_unb.shape}"
    assert X_te.shape == (n_test, 117), f"X_te shape {X_te.shape}"

    # ---- Reference anchor RAE (for delta reporting only) ----
    te_chem = np.load(TE_CHEM_PATH).astype(np.float64)
    assert te_chem.shape == (n_test,), f"te_chem shape {te_chem.shape}"
    anchor_unb = te_chem[unb_idx]
    rae_anchor_unb = float(rae(y_unb, anchor_unb))
    print(
        f"[anchor-ref] chemprop_aux te[unb_idx] RAE = {rae_anchor_unb:.4f} "
        f"(reference baseline; ranker is STANDALONE not residual)"
    )

    # ---- Scaffold 5-fold CV across 5 kf_seeds ----
    print("\n" + "-" * 78)
    print(
        f"SCAFFOLD 5-FOLD CV  seeds={KF_SEEDS}\n"
        f"LGBMRanker: lambdarank/ndcg  max_depth={RNK_MAX_DEPTH}  "
        f"num_leaves={RNK_NUM_LEAVES}  n_est={RNK_N_EST}  lr={RNK_LR}"
    )
    print("-" * 78)

    per_seed = []
    all_oofs = []
    for kf_seed in KF_SEEDS:
        ts = time.time()
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        oof_pred = np.full(n_unb, np.nan, dtype=np.float64)
        fold_info = []
        for fi, (tr_loc, va_loc) in enumerate(splits):
            y_tr = y_unb[tr_loc]
            n_tr = len(tr_loc)
            n_va = len(va_loc)

            # Relevance = integer rank-bin of y in the fold-train set
            # (0..N_BINS-1).  Fine enough to preserve ordinal signal,
            # small enough to fit in LGBMRanker's label_gain table.
            rel_tr = _rank_to_bin(y_tr, n_bins=N_BINS)

            # Single-group per fold: all rows in one group.
            group_tr = [n_tr]

            mdl = _new_ranker(seed=RNK_BASE_SEED + kf_seed + fi)
            mdl.fit(X_unb[tr_loc], rel_tr, group=group_tr)

            # Predict raw scores on val.
            score_va = mdl.predict(X_unb[va_loc])
            # Map to rank-quantile in [0,1] across the val fold.
            q_va = _rank_to_quantile(score_va)
            # Lookup pEC50 from the fold-train empirical CDF.
            pred_va = _calibrate_to_train_ecdf(y_tr, q_va)
            oof_pred[va_loc] = pred_va

            fold_info.append({
                "fold": fi,
                "n_tr": int(n_tr),
                "n_va": int(n_va),
                "score_va_mean": float(score_va.mean()),
                "score_va_std": float(score_va.std()),
                "pred_va_mean": float(pred_va.mean()),
                "pred_va_std": float(pred_va.std()),
            })

        # Gentle clip to a sane pEC50 range
        oof_pred = np.clip(oof_pred, 3.0, 8.0)
        pooled = float(rae(y_unb, oof_pred))
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": pooled,
            "folds": fold_info,
            "wall_sec": round(time.time() - ts, 2),
        })
        all_oofs.append(oof_pred)
        print(
            f"   seed={kf_seed}  pooled_RAE={pooled:.4f}  "
            f"wall={time.time()-ts:.1f}s"
        )

    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    final_oof_rae = float(rae(y_unb, mean_oof))
    pooled_rae_mean = float(np.mean([r["pooled_rae"] for r in per_seed]))
    pooled_rae_std = float(np.std([r["pooled_rae"] for r in per_seed]))
    print(
        f"\n[cv] pooled RAE mean across seeds = {pooled_rae_mean:.4f} "
        f"(+/- {pooled_rae_std:.4f})"
    )
    print(f"[cv] RAE of mean-of-seed OOFs      = {final_oof_rae:.4f}")
    print(
        f"[cv] delta vs anchor (chemprop_aux)= "
        f"{pooled_rae_mean - rae_anchor_unb:+.4f}"
    )

    # ---- Deploy: refit on ALL 253 -> predict on 513 ----
    print("\n" + "-" * 78)
    print("DEPLOY: refit LGBMRanker on all 253 unblind -> apply to 513")
    print("-" * 78)
    n_all = n_unb
    rel_all = _rank_to_bin(y_unb, n_bins=N_BINS)
    group_all = [n_all]

    deploy_mdl = _new_ranker(seed=RNK_BASE_SEED)
    deploy_mdl.fit(X_unb, rel_all, group=group_all)
    score_te = deploy_mdl.predict(X_te)
    q_te = _rank_to_quantile(score_te)
    deploy_te = _calibrate_to_train_ecdf(y_unb, q_te)
    deploy_te = np.clip(deploy_te, 3.0, 8.0).astype(np.float32)

    te_unb_in_rae = float(rae(y_unb, deploy_te[unb_idx].astype(np.float64)))
    print(
        f"[deploy] te(513) mean={deploy_te.mean():.3f}  "
        f"std={deploy_te.std():.3f}"
    )
    print(
        f"[deploy] te[unb_idx] in_RAE={te_unb_in_rae:.4f}  "
        f"(in-sample, deploy refit on all 253)"
    )

    # ---- Gate ----
    if pooled_rae_mean < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif pooled_rae_mean < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "-" * 78)
    print("GATE EVALUATION")
    print("-" * 78)
    print(f"   mean_rae       = {pooled_rae_mean:.4f}")
    print(f"   gate PROMOTE   = < {GATE_PROMOTE}")
    print(f"   gate MARGINAL  = < {GATE_MARGINAL}")
    print(f"   verdict        = {verdict}")

    # ---- Save artefacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, mean_oof.astype(np.float32))
    np.save(te_path, deploy_te)
    print(f"\n[save] {oof_path}")
    print(f"[save] {te_path}")

    summary = {
        "tag": TAG,
        "method": (
            "LGBMRanker (lambdarank/ndcg) standalone on X_117 + "
            "post-hoc rank->pEC50 calibration via fold-train ECDF"
        ),
        "anchor_name": "chemprop_aux (PRE-unblind, reference only -- ranker is STANDALONE)",
        "rae_anchor_unb": rae_anchor_unb,
        "rnk_objective": "lambdarank",
        "rnk_metric": "ndcg",
        "rnk_max_depth": RNK_MAX_DEPTH,
        "rnk_num_leaves": RNK_NUM_LEAVES,
        "rnk_n_estimators": RNK_N_EST,
        "rnk_learning_rate": RNK_LR,
        "rnk_base_seed": RNK_BASE_SEED,
        "n_relevance_bins": N_BINS,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_unb": n_unb,
        "n_te": n_test,
        "n_unique_scaffolds": n_uniq_scaf,
        "feat_dim": int(X_unb.shape[1]),
        "per_seed": per_seed,
        "mean_rae": pooled_rae_mean,
        "pooled_rae_std_seeds": pooled_rae_std,
        "rae_of_mean_of_seed_oofs": final_oof_rae,
        "delta_vs_anchor": pooled_rae_mean - rae_anchor_unb,
        "te_unb_rae_in_sample": te_unb_in_rae,
        "te_deploy_mean": float(deploy_te.mean()),
        "te_deploy_std": float(deploy_te.std()),
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
    print(
        f"   mean_rae (5 seeds)            = {pooled_rae_mean:.4f} "
        f"(+/- {pooled_rae_std:.4f})"
    )
    print(
        f"   delta vs anchor (chemprop_aux)= "
        f"{pooled_rae_mean - rae_anchor_unb:+.4f}"
    )
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
        "rae_anchor_unb",
        "delta_vs_anchor",
        "verdict",
        "te_unb_rae_in_sample",
    ):
        print(f"  {k}: {res.get(k)}")
