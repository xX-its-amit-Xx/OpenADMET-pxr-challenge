"""nb2761 -- sklearn MLPRegressor on K=20 chemprop_aux residual.

NEW PARADIGM:
    sklearn.neural_network.MLPRegressor with Adam solver, two-hidden-layer
    feed-forward (64, 32) ReLU + L2 alpha=0.01 + early stopping on a 10%
    internal validation hold-out.

    Why sklearn over torch on the K=20 substrate:
      1. SIMPLER API -- no manual optimizer loop, no dataloader scaffolding,
         no manual early-stopping bookkeeping; reduces implementation risk
         when scanning many small variants on the chemprop_aux residual.
      2. Built-in early stopping uses an INTERNAL stratified hold-out
         (validation_fraction=0.1) and quits once val score stops improving
         (n_iter_no_change=10 by default); no external CV-fold-level early
         stop needed -> deterministic per (seed, fold).
      3. StandardScaler upstream is the canonical sklearn pairing (MLPs are
         not scale-invariant); makes the (anchor=chemprop_aux, K=20, n=253)
         tuple comparable to prior nn-style scripts that used z-score input.

    Distinction from prior NN-style scripts on this substrate:
      - torch BN+Dropout+AdamW heads (cycle-134-era) used manual epoch loop,
        bag-of-seeds averaging, hand-tuned dropout. NEUTRAL on chemprop_aux.
      - This script: sklearn MLP, Adam default, alpha=0.01 L2, single seed
        per fold, early stopping. Tests whether the EARLY-STOPPING + WEIGHT-
        DECAY axis (rather than dropout / BN) is the missing regularizer.

PROTOCOL:
    1. Slice X_K20 = first 20 cols of X_117_unb / X_117_te.
    2. Anchor: chemprop_aux (PRE-unblind, verified clean).
       Residual target = y_unb - anchor_unb.
    3. Per fold:
         StandardScaler fit on tr fold -> apply to va fold + 513 test.
         MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu',
                      solver='adam', alpha=0.01, learning_rate_init=0.001,
                      max_iter=500, random_state=42, early_stopping=True,
                      validation_fraction=0.1).
       Fit on residual; predict per row -> resid_hat.
       Final per-row pred = anchor + resid_hat.
    4. 5-fold scaffold CV on 253 unblind, kf_seeds {1001..1005}.
    5. Deploy: refit scaler+MLP on all 253 -> predict on 513.

GATE:
    mean_rae < 0.4570 -> "PROMOTE"
    mean_rae < 0.4598 -> "MARGINAL_BEAT"
    else            -> "FAIL"

Outputs:
    scripts/nb2761_sklearn_mlp.py
    data/processed/nb2761_summary.json
    data/processed/nb2761_pred_oof.npy   (253,) float32
    data/processed/te_nb2761.npy         (513,) float32
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
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2761"

# --------------------------------------------------------------------------
# sklearn MLPRegressor hyperparameters (per spec)
# --------------------------------------------------------------------------
MLP_HIDDEN_LAYER_SIZES = (64, 32)
MLP_ACTIVATION = "relu"
MLP_SOLVER = "adam"
MLP_ALPHA = 0.01
MLP_LEARNING_RATE_INIT = 0.001
MLP_MAX_ITER = 500
MLP_RANDOM_STATE = 42
MLP_EARLY_STOPPING = True
MLP_VALIDATION_FRACTION = 0.1

# CV protocol
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# Gates
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

# Paths
X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"
TE_CHEM_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# Number of K=20 cols sliced from the 117-col block (first-20 contract).
K_SLICE = 20


def _new_mlp(seed: int = MLP_RANDOM_STATE) -> MLPRegressor:
    return MLPRegressor(
        hidden_layer_sizes=MLP_HIDDEN_LAYER_SIZES,
        activation=MLP_ACTIVATION,
        solver=MLP_SOLVER,
        alpha=MLP_ALPHA,
        learning_rate_init=MLP_LEARNING_RATE_INIT,
        max_iter=MLP_MAX_ITER,
        random_state=int(seed),
        early_stopping=MLP_EARLY_STOPPING,
        validation_fraction=MLP_VALIDATION_FRACTION,
    )


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- sklearn MLPRegressor on K=20 chemprop_aux residual")
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
    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV  seeds={KF_SEEDS}\n"
          f"MLPRegressor: hidden={MLP_HIDDEN_LAYER_SIZES}  act={MLP_ACTIVATION}  "
          f"solver={MLP_SOLVER}  alpha={MLP_ALPHA}\n"
          f"  lr_init={MLP_LEARNING_RATE_INIT}  max_iter={MLP_MAX_ITER}  "
          f"early_stop={MLP_EARLY_STOPPING}  val_frac={MLP_VALIDATION_FRACTION}")
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
            # StandardScaler fit on tr fold only -> apply to va fold
            scaler = StandardScaler()
            X_tr_z = scaler.fit_transform(X_unb[tr_loc])
            X_va_z = scaler.transform(X_unb[va_loc])
            mdl = _new_mlp(seed=MLP_RANDOM_STATE + kf_seed + fi)
            mdl.fit(X_tr_z, resid_unb[tr_loc])
            oof_resid[va_loc] = mdl.predict(X_va_z)
            fold_info.append({
                "fold": fi,
                "n_tr": int(len(tr_loc)),
                "n_va": int(len(va_loc)),
                "n_iter": int(getattr(mdl, "n_iter_", -1)),
                "resid_pred_mean": float(oof_resid[va_loc].mean()),
                "resid_pred_std": float(oof_resid[va_loc].std()),
            })
        assert not np.isnan(oof_resid).any(), "oof_resid has NaN -- fold cover incomplete"
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
        print(f"   seed={kf_seed}  pooled_RAE={pooled:.4f}  wall={time.time()-ts:.1f}s")

    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    final_oof_rae = float(rae(y_unb, mean_oof))
    pooled_rae_mean = float(np.mean([r["pooled_rae"] for r in per_seed]))
    pooled_rae_std = float(np.std([r["pooled_rae"] for r in per_seed]))
    print(f"\n[cv] pooled RAE mean across seeds = {pooled_rae_mean:.4f} "
          f"(+/- {pooled_rae_std:.4f})")
    print(f"[cv] RAE of mean-of-seed OOFs      = {final_oof_rae:.4f}")
    print(f"[cv] delta vs anchor (chemprop_aux)= {pooled_rae_mean - rae_anchor_unb:+.4f}")

    # ---- Deploy: refit on ALL 253 -> predict on 513 ----
    print("\n" + "-" * 78)
    print("DEPLOY: refit scaler+MLP on all 253 unblind -> apply to 513")
    print("-" * 78)
    deploy_scaler = StandardScaler()
    X_unb_z = deploy_scaler.fit_transform(X_unb)
    X_te_z = deploy_scaler.transform(X_te)
    deploy_mdl = _new_mlp(seed=MLP_RANDOM_STATE)
    deploy_mdl.fit(X_unb_z, resid_unb)
    deploy_resid_te = deploy_mdl.predict(X_te_z)
    deploy_te = anchor_te + deploy_resid_te
    deploy_te = np.clip(deploy_te, 3.0, 8.0).astype(np.float32)
    te_unb_in_rae = float(rae(y_unb, deploy_te[unb_idx].astype(np.float64)))
    print(f"[deploy] te(513) mean={deploy_te.mean():.3f}  std={deploy_te.std():.3f}  "
          f"n_iter={int(getattr(deploy_mdl, 'n_iter_', -1))}")
    print(f"[deploy] te[unb_idx] in_RAE={te_unb_in_rae:.4f}  "
          f"(in-sample, deploy refit on all 253)")

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
            "sklearn.neural_network.MLPRegressor with Adam solver on "
            "z-scored (StandardScaler per fold) first-K=20 cols of X_117; "
            "hidden_layer_sizes=(64,32) ReLU, alpha=0.01 L2, "
            "early_stopping=True (val_frac=0.1), max_iter=500; "
            "fit on chemprop_aux residual."
        ),
        "paradigm": (
            "sklearn_mlp_regressor_adam_early_stopping_weight_decay_"
            "axis_simpler_api_vs_torch_nn_bn_dropout_adamw"
        ),
        "anchor_name": "chemprop_aux (PRE-unblind, verified clean)",
        "rae_anchor_unb": rae_anchor_unb,
        "mlp_hidden_layer_sizes": list(MLP_HIDDEN_LAYER_SIZES),
        "mlp_activation": MLP_ACTIVATION,
        "mlp_solver": MLP_SOLVER,
        "mlp_alpha": MLP_ALPHA,
        "mlp_learning_rate_init": MLP_LEARNING_RATE_INIT,
        "mlp_max_iter": MLP_MAX_ITER,
        "mlp_random_state": MLP_RANDOM_STATE,
        "mlp_early_stopping": MLP_EARLY_STOPPING,
        "mlp_validation_fraction": MLP_VALIDATION_FRACTION,
        "scaler": "sklearn.preprocessing.StandardScaler (per fold; deploy refit on 253)",
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": n_uniq_scaf,
        "feat_dim": int(X_unb.shape[1]),
        "k_slice_first_n_of_117": K_SLICE,
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
        "promote": bool(verdict == "PROMOTE"),
        "marginal_beat": bool(verdict == "MARGINAL_BEAT"),
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
