"""nb2811 -- TabNet (Arik & Pfister 2019) on K=20 chemprop_aux residual.

NEW PARADIGM:
    TabNet is an attention-based tabular DL architecture (Arik & Pfister 2019,
    "TabNet: Attentive Interpretable Tabular Learning"). At each decision step
    it uses a learnable sparse attention mask (Sparsemax) to select a SUBSET
    of input features to attend over, then passes the masked input through a
    feature transformer block. Sparsity is encouraged by a `lambda_sparse`
    entropy penalty on the attention masks.

    Distinction from prior DL/tabular attacks on this anchor:
      - MLP/NN heads (cycle-134 paradigm exhaustion): every neuron sees every
        feature every forward pass (dense attention, no instance-level
        feature gating).
      - LGBM (nb2103/nb2112): hard axis-aligned splits, no learned attention.
      - ExtraTrees (nb2731): random split thresholds, no attention.
      - TabNet here: per-instance soft+sparse feature attention with multi-step
        sequential reasoning, the only sparse-attention DL in our zoo on this
        anchor.

PROTOCOL:
    1. Slice X_K20 = first 20 cols of X_117_unb / X_117_te.
    2. Anchor: chemprop_aux (PRE-unblind, verified clean).
       Residual target = y_unb - anchor_unb.
    3. Model: pytorch_tabnet.tab_model.TabNetRegressor(
                n_d=32, n_a=32, n_steps=3, gamma=1.3, lambda_sparse=1e-3,
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=1e-2)).
       fit max_epochs=100, batch_size=32, patience=10 (early stop on RMSE).
       Predict per row -> resid_hat.
       Final per-row pred = anchor + resid_hat.
    4. 5-fold scaffold CV on 253 unblind, kf_seeds {1001..1005}.
    5. Deploy: refit on all 253 -> predict on 513.

GATE:
    mean_rae < 0.4570 -> "PROMOTE"
    mean_rae < 0.4598 -> "MARGINAL_BEAT"
    else            -> "FAIL"

Outputs:
    scripts/nb2811_tabnet.py
    data/processed/nb2811_summary.json
    data/processed/nb2811_pred_oof.npy   (253,) float32
    data/processed/te_nb2811.npy         (513,) float32

If pytorch_tabnet not importable -> print "INSTALL_FAILED" and exit clean (rc 0).
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

# ---- TabNet install guard (exit clean on missing dep) ----
try:
    import torch
    from pytorch_tabnet.tab_model import TabNetRegressor
except Exception as e:  # noqa: BLE001
    print("INSTALL_FAILED")
    print(f"reason: {type(e).__name__}: {e}")
    sys.exit(0)

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2811"

# --------------------------------------------------------------------------
# TabNet hyperparameters (per spec)
# --------------------------------------------------------------------------
TN_N_D = 32
TN_N_A = 32
TN_N_STEPS = 3
TN_GAMMA = 1.3
TN_LAMBDA_SPARSE = 1e-3
TN_LR = 1e-2
TN_MAX_EPOCHS = 100
TN_BATCH_SIZE = 32
TN_PATIENCE = 10
TN_SEED = 42

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

# Number of K=20 cols sliced from the 117-col block.
K_SLICE = 20


def _new_tabnet(seed: int = TN_SEED) -> "TabNetRegressor":
    return TabNetRegressor(
        n_d=TN_N_D,
        n_a=TN_N_A,
        n_steps=TN_N_STEPS,
        gamma=TN_GAMMA,
        lambda_sparse=TN_LAMBDA_SPARSE,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=TN_LR),
        seed=int(seed),
        verbose=0,
        device_name="cpu",
    )


def _fit_predict(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Fit TabNet on (X_tr, y_tr), early-stop on (X_va, y_va_placeholder)
    using RMSE on a held-out slice of TRAIN (we don't pass va labels to the
    fit -- we just want predictions on X_va). For early stopping we carve
    a small 15% slice off TRAIN. Returns predictions on X_va (1-D)."""
    mdl = _new_tabnet(seed=seed)

    # Convert to TabNet's expected dtype/shape.
    Xtr = X_tr.astype(np.float32)
    ytr = y_tr.astype(np.float32).reshape(-1, 1)

    # Carve a small early-stopping slice from train (15%, min 16).
    rng = np.random.default_rng(seed)
    n = Xtr.shape[0]
    n_es = max(16, int(round(0.15 * n)))
    perm = rng.permutation(n)
    es_idx = perm[:n_es]
    fit_idx = perm[n_es:]

    Xfit, yfit = Xtr[fit_idx], ytr[fit_idx]
    Xes, yes = Xtr[es_idx], ytr[es_idx]

    mdl.fit(
        Xfit, yfit,
        eval_set=[(Xes, yes)],
        eval_name=["es"],
        eval_metric=["rmse"],
        max_epochs=TN_MAX_EPOCHS,
        patience=TN_PATIENCE,
        batch_size=TN_BATCH_SIZE,
        virtual_batch_size=min(TN_BATCH_SIZE, 16),
        num_workers=0,
        drop_last=False,
    )
    pred = mdl.predict(X_va.astype(np.float32)).reshape(-1)
    return pred.astype(np.float64)


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- TabNet attention DL on K=20 chemprop_aux residual")
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
          f"TabNet: n_d={TN_N_D}  n_a={TN_N_A}  n_steps={TN_N_STEPS}  "
          f"gamma={TN_GAMMA}  lambda_sparse={TN_LAMBDA_SPARSE}  lr={TN_LR}  "
          f"max_epochs={TN_MAX_EPOCHS}  batch={TN_BATCH_SIZE}  "
          f"patience={TN_PATIENCE}")
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
            torch.manual_seed(TN_SEED + kf_seed + fi)
            np.random.seed(TN_SEED + kf_seed + fi)
            pred = _fit_predict(
                X_unb[tr_loc],
                resid_unb[tr_loc],
                X_unb[va_loc],
                seed=TN_SEED + kf_seed + fi,
            )
            oof_resid[va_loc] = pred
            fold_info.append({
                "fold": fi,
                "n_tr": int(len(tr_loc)),
                "n_va": int(len(va_loc)),
                "resid_pred_mean": float(pred.mean()),
                "resid_pred_std": float(pred.std()),
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
    print("DEPLOY: refit on all 253 unblind -> apply to 513")
    print("-" * 78)
    torch.manual_seed(TN_SEED)
    np.random.seed(TN_SEED)
    deploy_resid_te = _fit_predict(X_unb, resid_unb, X_te, seed=TN_SEED)
    deploy_te = anchor_te + deploy_resid_te
    deploy_te = np.clip(deploy_te, 3.0, 8.0).astype(np.float32)
    te_unb_in_rae = float(rae(y_unb, deploy_te[unb_idx].astype(np.float64)))
    print(f"[deploy] te(513) mean={deploy_te.mean():.3f}  std={deploy_te.std():.3f}")
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
            "TabNet (Arik & Pfister 2019) via pytorch_tabnet.tab_model."
            "TabNetRegressor; attention-based tabular DL with Sparsemax "
            "feature-mask selection at each of n_steps decision steps, "
            "lambda_sparse entropy penalty, fit on chemprop_aux residual "
            "over first-K=20 cols of X_117."
        ),
        "paradigm": (
            "sparse_attention_tabular_dl_tabnet_arik_pfister_2019_"
            "sequential_feature_mask_selection_via_sparsemax"
        ),
        "anchor_name": "chemprop_aux (PRE-unblind, verified clean)",
        "rae_anchor_unb": rae_anchor_unb,
        "tn_n_d": TN_N_D,
        "tn_n_a": TN_N_A,
        "tn_n_steps": TN_N_STEPS,
        "tn_gamma": TN_GAMMA,
        "tn_lambda_sparse": TN_LAMBDA_SPARSE,
        "tn_lr": TN_LR,
        "tn_max_epochs": TN_MAX_EPOCHS,
        "tn_batch_size": TN_BATCH_SIZE,
        "tn_patience": TN_PATIENCE,
        "tn_seed": TN_SEED,
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
