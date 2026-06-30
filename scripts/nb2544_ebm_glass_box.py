"""nb2544 -- Explainable Boosting Machine (EBM) on chemprop_aux residual.

NEW PARADIGM:
    EBM = generalized additive model with pairwise interactions; cyclic
    boosting + bagging.  Different inductive bias than LGBM trees:
    - additive shape per feature (smooth, monotone-friendly)
    - explicit pairwise interaction terms (fixed budget = `interactions`)
    - cyclic round-robin updates per feature/interaction
    - internal bagging (`outer_bags`) for variance reduction
    Residual = y - chemprop_aux (PRE-unblind verified-clean anchor).

PROTOCOL:
    - Features  : X_117 from data/processed/pyramid/X_117_unb.npy (+ X_117_te.npy)
    - Anchor    : oof_chemprop_aux.npy (sliced to unb_idx) on the 253 unblind;
                  te_chemprop_aux.npy on the 513 test set.
    - Model     : ExplainableBoostingRegressor(
                      interactions=10, max_bins=64, max_rounds=300,
                      learning_rate=0.05, random_state=42
                  )
    - 5-fold scaffold CV on 253 unblind, kf_seeds {1001..1005}
    - Fit on residual y - anchor; final pred = anchor + ebm_resid
    - Deploy: refit on ALL 253 -> predict on 513

GATE: mean_rae < 0.4570 -> "PROMOTE"
      mean_rae < 0.4601 -> "MARGINAL_BEAT"
      else            -> "FAIL"

Outputs:
    scripts/nb2544_ebm_glass_box.py
    data/processed/nb2544_summary.json
    data/processed/nb2544_pred_oof.npy   (253,) float32
    data/processed/te_nb2544.npy         (513,) float32
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

# Attempt EBM import - exit clean if install failed.
try:
    from interpret.glassbox import ExplainableBoostingRegressor
except Exception as e:
    print(f"[fatal] interpret import failed: {e}")
    print("INSTALL_FAILED")
    sys.exit(0)

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2544"

# -----------------------------
# EBM hyperparameters (spec)
# -----------------------------
EBM_INTERACTIONS = 10
EBM_MAX_BINS = 64
EBM_MAX_ROUNDS = 300
EBM_LR = 0.05
EBM_SEED = 42

# CV protocol
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# Gates
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4601

# Paths
X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"
OOF_CHEM_PATH = DATA_PROCESSED / "oof_chemprop_aux.npy"   # (4139,)
TE_CHEM_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"     # (513,)


def _new_ebm(seed: int = EBM_SEED) -> "ExplainableBoostingRegressor":
    return ExplainableBoostingRegressor(
        interactions=EBM_INTERACTIONS,
        max_bins=EBM_MAX_BINS,
        max_rounds=EBM_MAX_ROUNDS,
        learning_rate=EBM_LR,
        random_state=int(seed),
        n_jobs=2,
    )


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Explainable Boosting Machine on chemprop_aux residual (X_117)")
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

    # ---- Load anchor (chemprop_aux) ----
    # oof_chemprop_aux is on the FULL train (4139,) - we don't need it here;
    # for the 253 unblind we use te_chemprop_aux[unb_idx] which is the deploy
    # PRE-unblind prediction (model trained on the full 4139 train, evaluated
    # on the 513 test set). This is the verified-clean PRE-unblind anchor.
    if not TE_CHEM_PATH.exists():
        raise FileNotFoundError(f"missing test anchor: {TE_CHEM_PATH}")
    te_chem = np.load(TE_CHEM_PATH).astype(np.float64)
    assert te_chem.shape == (n_test,), f"te_chem shape {te_chem.shape}"
    anchor_unb = te_chem[unb_idx]
    anchor_te = te_chem.copy()
    rae_anchor_unb = float(rae(y_unb, anchor_unb))
    print(
        f"[anchor] chemprop_aux te[unb_idx] RAE = {rae_anchor_unb:.4f} "
        f"(PRE-clean PRIMARY-1 baseline)"
    )

    # ---- Residual target ----
    resid_unb = y_unb - anchor_unb
    print(
        f"[resid] mean={resid_unb.mean():+.3f}  "
        f"std={resid_unb.std():.3f}  "
        f"min={resid_unb.min():+.2f}  max={resid_unb.max():+.2f}"
    )

    # ---- Scaffold 5-fold CV across 5 kf_seeds ----
    print("\n" + "-" * 78)
    print(
        f"SCAFFOLD 5-FOLD CV  seeds={KF_SEEDS}\n"
        f"EBM: interactions={EBM_INTERACTIONS}  max_bins={EBM_MAX_BINS}  "
        f"max_rounds={EBM_MAX_ROUNDS}  lr={EBM_LR}  seed={EBM_SEED}"
    )
    print("-" * 78)

    per_seed = []
    all_oofs = []
    for kf_seed in KF_SEEDS:
        ts = time.time()
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        oof_resid = np.full(n_unb, np.nan, dtype=np.float64)
        fold_info = []
        for fi, (tr_loc, va_loc) in enumerate(splits):
            mdl = _new_ebm(seed=EBM_SEED + kf_seed + fi)
            mdl.fit(X_unb[tr_loc], resid_unb[tr_loc])
            oof_resid[va_loc] = mdl.predict(X_unb[va_loc]).astype(np.float64)
            fold_info.append({
                "fold": fi,
                "n_tr": int(len(tr_loc)),
                "n_va": int(len(va_loc)),
                "resid_pred_mean": float(oof_resid[va_loc].mean()),
                "resid_pred_std": float(oof_resid[va_loc].std()),
            })
        oof_pred = anchor_unb + oof_resid
        # gentle clip to a sane pEC50 range
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
    print(f"[cv] delta vs anchor (chemprop_aux)= {pooled_rae_mean - rae_anchor_unb:+.4f}")

    # ---- Deploy: refit on ALL 253 -> predict on 513 ----
    print("\n" + "-" * 78)
    print("DEPLOY: refit on all 253 unblind -> apply to 513")
    print("-" * 78)
    deploy_mdl = _new_ebm(seed=EBM_SEED)
    deploy_mdl.fit(X_unb, resid_unb)
    deploy_resid_te = deploy_mdl.predict(X_te).astype(np.float64)
    deploy_te = anchor_te + deploy_resid_te
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
            "Explainable Boosting Machine (GAM + pairwise interactions) "
            "on X_117 fit to chemprop_aux residual"
        ),
        "anchor_name": "chemprop_aux (PRE-unblind, verified clean)",
        "rae_anchor_unb": rae_anchor_unb,
        "ebm_interactions": EBM_INTERACTIONS,
        "ebm_max_bins": EBM_MAX_BINS,
        "ebm_max_rounds": EBM_MAX_ROUNDS,
        "ebm_lr": EBM_LR,
        "ebm_seed": EBM_SEED,
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
    print(f"   mean_rae (5 seeds)            = {pooled_rae_mean:.4f} "
          f"(+/- {pooled_rae_std:.4f})")
    print(f"   delta vs anchor (chemprop_aux)= {pooled_rae_mean - rae_anchor_unb:+.4f}")
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
