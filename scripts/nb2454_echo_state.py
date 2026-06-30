"""nb2454 -- Echo-State Reservoir on MACCS-167 + Morgan-1024 substructure bag.

CONTEXT:
    Genuinely-different architecture vs the LGBM/MLP/chemprop family that
    dominates the existing pyramid. Reservoir computing freezes a random
    recurrent net (W_in, W_rec with spectral radius 0.9) and trains only a
    ridge readout on the final state. K=5 unrolled time-steps feed the SAME
    1191-dim feature vector each tick to settle the reservoir into a non-
    linear fixed-point response.

PROTOCOL:
    1. Feature: MACCS-167 (te_maccs.npy, uint8) + Morgan-1024 (radius=2,
       n_bits=1024) -> 1191-dim float32 on 513.
    2. Reservoir: W_in (500, 1191) ~ N(0, 1/sqrt(1191)),
                  W_rec (500, 500)  rescaled to spectral_radius=0.9,
                  state h_t = tanh(W_rec h_{t-1} + W_in u),  h_0 = 0.
       Unroll K=5 ticks with constant input u; collect h_5.
    3. Ridge readout (alpha=1.0) on residual r = y_unb - chemprop_aux.
    4. 5-fold scaffold CV on the 253 unblind compounds:
         per fold: fit ridge on train h_5 -> predict valid residual.
       Standalone RAE = rae(y_unb, anchor + oof_resid).
    5. If standalone <0.55, evaluate as additive correction to nb2240 pyramid:
         blended = 0.5 * (anchor + esn_resid) + 0.5 * nb2240
       Report blended RAE; compare to nb2240 0.4630.
    6. Save submissions/nb2454_esn.csv (deploy: refit on all 253).

Outputs:
    scripts/nb2454_echo_state.py
    data/processed/nb2454_summary.json
    data/processed/nb2454_pred_oof.npy   (253,) standalone OOF prediction
    data/processed/te_nb2454.npy         (513,) deploy prediction
    submissions/nb2454_esn.csv           (always written)
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
from rdkit import RDLogger
from sklearn.linear_model import Ridge

RDLogger.DisableLog("rdApp.*")

from pxr.chem import morgan_fp_batch, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2454"

# ---- reservoir hyperparams ----
RESERVOIR_DIM = 500
SPECTRAL_RADIUS = 0.9
N_TIMESTEPS = 5
RIDGE_ALPHA = 1.0
MORGAN_NBITS = 1024
MORGAN_RADIUS = 2
MACCS_DIM = 167
INPUT_DIM = MACCS_DIM + MORGAN_NBITS  # 1191

# ---- CV setup ----
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
RES_SEEDS = [0, 1, 7, 42, 137]  # multi-seed reservoir bag

NB2240_REF = 0.4630
GATE_STANDALONE = 0.55
GATE_BLEND_VS_NB2240 = 0.003

ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
NB2240_TE_PATH = DATA_PROCESSED / "te_nb2240.npy"
NB2240_OOF_PATH = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"


def build_reservoir(seed: int):
    """Construct W_in and W_rec with target spectral radius."""
    rng = np.random.default_rng(seed)
    W_in = (rng.standard_normal((RESERVOIR_DIM, INPUT_DIM)) /
            np.sqrt(INPUT_DIM)).astype(np.float32)
    W_rec_raw = rng.standard_normal((RESERVOIR_DIM, RESERVOIR_DIM)).astype(np.float32)
    # rescale to spectral radius 0.9
    eigvals = np.linalg.eigvals(W_rec_raw)
    cur_radius = float(np.max(np.abs(eigvals)))
    W_rec = (W_rec_raw * (SPECTRAL_RADIUS / cur_radius)).astype(np.float32)
    return W_in, W_rec


def reservoir_states(X: np.ndarray, W_in: np.ndarray, W_rec: np.ndarray) -> np.ndarray:
    """Unroll K time-steps with constant input vector per compound -> final state."""
    # u_proj: (N, R)
    u_proj = X @ W_in.T
    n = X.shape[0]
    h = np.zeros((n, RESERVOIR_DIM), dtype=np.float32)
    for _ in range(N_TIMESTEPS):
        h = np.tanh(h @ W_rec.T + u_proj)
    return h


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Echo-State Reservoir on MACCS+Morgan substructure bag")
    print("=" * 78)

    # ---- Load data ----
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values
                if "name" in te.columns else te["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test} n_unb={n_unb}")

    # ---- Features: MACCS-167 + Morgan-1024 ----
    X_maccs = np.load(MACCS_TE_PATH).astype(np.float32)
    assert X_maccs.shape == (n_test, MACCS_DIM)
    X_morgan = morgan_fp_batch(te_smiles, radius=MORGAN_RADIUS,
                               n_bits=MORGAN_NBITS).astype(np.float32)
    assert X_morgan.shape == (n_test, MORGAN_NBITS)
    X_te = np.concatenate([X_maccs, X_morgan], axis=1)
    assert X_te.shape == (n_test, INPUT_DIM), f"got {X_te.shape}"
    X_unb = X_te[unb_idx]
    print(f"[feat] X_te {X_te.shape}  X_unb {X_unb.shape}  density={X_te.mean():.4f}")

    # ---- Anchor (chemprop_aux) ----
    anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor_unb = anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    print(f"[anchor] chemprop_aux in_RAE = {rae_anchor:.4f}")
    residual = y_unb - anchor_unb

    # ---- Scaffolds for CV ----
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    # ---- Build reservoir states once per seed (reservoir is data-agnostic) ----
    print("\n" + "-" * 78)
    print(f"RESERVOIR build  R={RESERVOIR_DIM}  K={N_TIMESTEPS}  "
          f"rho={SPECTRAL_RADIUS}  seeds={RES_SEEDS}")
    print("-" * 78)
    H_unb_per_seed = []
    H_te_per_seed = []
    for s in RES_SEEDS:
        ts = time.time()
        W_in, W_rec = build_reservoir(s)
        H_te = reservoir_states(X_te, W_in, W_rec)
        H_unb_per_seed.append(H_te[unb_idx])
        H_te_per_seed.append(H_te)
        print(f"   seed={s:3d}  H_unb {H_unb_per_seed[-1].shape}  "
              f"|h|_mean={float(np.abs(H_te).mean()):.3f}  "
              f"wall={time.time()-ts:.1f}s")

    # ---- Per-seed 5-fold scaffold CV on residual; mean-bag final ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD CV  folds={N_FOLDS}  kf_seeds={KF_SEEDS}  ridge_alpha={RIDGE_ALPHA}")
    print("-" * 78)
    per_seed_summary = []
    bag_oof = np.zeros(n_unb, dtype=np.float64)  # mean-bag standalone prediction
    n_combos = 0
    for kf_seed in KF_SEEDS:
        splits = scaffold_kfold_indices(unb_scaffolds, n_splits=N_FOLDS,
                                        shuffle=True, seed=kf_seed)
        per_res_oof = []
        for r_idx, res_seed in enumerate(RES_SEEDS):
            H_unb = H_unb_per_seed[r_idx]
            oof_resid = np.full(n_unb, np.nan, dtype=np.float64)
            for tr_loc, va_loc in splits:
                mdl = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
                mdl.fit(H_unb[tr_loc], residual[tr_loc])
                oof_resid[va_loc] = mdl.predict(H_unb[va_loc])
            per_res_oof.append(anchor_unb + oof_resid)
        seed_mean = np.mean(np.column_stack(per_res_oof), axis=1)
        seed_rae = float(rae(y_unb, seed_mean))
        per_seed_summary.append({"kf_seed": int(kf_seed), "pooled_rae": seed_rae})
        bag_oof += seed_mean
        n_combos += 1
        print(f"   kf_seed={kf_seed}  pooled_RAE={seed_rae:.4f}")
    bag_oof /= n_combos
    rae_standalone = float(rae(y_unb, bag_oof))
    pooled_mean = float(np.mean([r["pooled_rae"] for r in per_seed_summary]))
    pooled_std = float(np.std([r["pooled_rae"] for r in per_seed_summary]))
    print(f"\n[standalone] mean(kf_seed pooled_RAE) = {pooled_mean:.4f} +/- {pooled_std:.4f}")
    print(f"[standalone] RAE of bag_oof           = {rae_standalone:.4f}")
    print(f"[standalone] delta vs anchor           = {rae_standalone - rae_anchor:+.4f}")
    print(f"[standalone] delta vs nb2240 ({NB2240_REF:.4f}) = "
          f"{rae_standalone - NB2240_REF:+.4f}")

    # ---- Blend with nb2240 pyramid if standalone <0.55 ----
    blend_block = None
    if rae_standalone < GATE_STANDALONE:
        print("\n" + "-" * 78)
        print(f"BLEND with nb2240  (standalone {rae_standalone:.4f} < {GATE_STANDALONE})")
        print("-" * 78)
        nb2240_oof = np.load(NB2240_OOF_PATH).astype(np.float64)
        rae_nb2240_oof = float(rae(y_unb, nb2240_oof))
        # equal-weight 0.5/0.5 blend on OOF
        blend_oof = 0.5 * bag_oof + 0.5 * nb2240_oof
        rae_blend = float(rae(y_unb, blend_oof))
        print(f"   nb2240 OOF RAE (in-sample mean-bag) = {rae_nb2240_oof:.4f}")
        print(f"   0.5*ESN + 0.5*nb2240 OOF RAE        = {rae_blend:.4f}")
        print(f"   blend delta vs nb2240 ref {NB2240_REF:.4f} = "
              f"{rae_blend - NB2240_REF:+.4f}")
        blend_block = {
            "rae_nb2240_oof_in_sample": rae_nb2240_oof,
            "rae_blend_50_50_oof": rae_blend,
            "delta_vs_nb2240_ref": rae_blend - NB2240_REF,
            "blend_helps": bool(rae_blend < NB2240_REF - GATE_BLEND_VS_NB2240),
        }
    else:
        print(f"\n[skip-blend] standalone {rae_standalone:.4f} >= {GATE_STANDALONE}, "
              f"no blend evaluation")

    # ---- Deploy: refit ridge on all 253; mean over reservoir seeds ----
    print("\n" + "-" * 78)
    print("DEPLOY (refit ridge on all 253; mean over reservoir seeds)")
    print("-" * 78)
    te_preds = []
    for r_idx, res_seed in enumerate(RES_SEEDS):
        H_unb = H_unb_per_seed[r_idx]
        H_te = H_te_per_seed[r_idx]
        mdl = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
        mdl.fit(H_unb, residual)
        te_resid = mdl.predict(H_te)
        te_preds.append(anchor_513 + te_resid)
    te_deploy = np.mean(np.column_stack(te_preds), axis=1).astype(np.float32)
    # clip to a sane pEC50 band
    te_deploy = np.clip(te_deploy, 3.0, 9.0).astype(np.float32)
    te_unb_in = float(rae(y_unb, te_deploy[unb_idx]))
    print(f"   te_deploy mean={te_deploy.mean():.3f}  std={te_deploy.std():.3f}")
    print(f"   te[unb_idx] in-sample RAE = {te_unb_in:.4f}")

    # ---- Save artifacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, bag_oof.astype(np.float32))
    np.save(te_path, te_deploy)
    print(f"[save] {oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_esn.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": te_deploy,
    }).to_csv(sub_csv, index=False)
    print(f"[save] {sub_csv}")

    summary = {
        "tag": TAG,
        "method": "echo_state_reservoir_on_chemprop_aux_residual",
        "architecture": {
            "input_dim": INPUT_DIM,
            "maccs_dim": MACCS_DIM,
            "morgan_nbits": MORGAN_NBITS,
            "morgan_radius": MORGAN_RADIUS,
            "reservoir_dim": RESERVOIR_DIM,
            "spectral_radius": SPECTRAL_RADIUS,
            "n_timesteps": N_TIMESTEPS,
            "ridge_alpha": RIDGE_ALPHA,
            "activation": "tanh",
        },
        "anchor": "chemprop_aux",
        "anchor_in_rae": rae_anchor,
        "n_unb": n_unb,
        "n_te": n_test,
        "n_unique_scaffolds": n_unique_scaf,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "reservoir_seeds": RES_SEEDS,
        "per_seed_pooled": per_seed_summary,
        "pooled_rae_mean_seeds": pooled_mean,
        "pooled_rae_std_seeds": pooled_std,
        "rae_standalone_bag": rae_standalone,
        "delta_standalone_vs_anchor": rae_standalone - rae_anchor,
        "delta_standalone_vs_nb2240": rae_standalone - NB2240_REF,
        "nb2240_ref": NB2240_REF,
        "gate_standalone_threshold": GATE_STANDALONE,
        "passed_standalone_gate": bool(rae_standalone < GATE_STANDALONE),
        "blend_block": blend_block,
        "te_unb_in_sample_rae": te_unb_in,
        "deploy_te_mean": float(te_deploy.mean()),
        "deploy_te_std": float(te_deploy.std()),
        "oof_npy_path": str(oof_path),
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
    print(f"   standalone RAE              = {rae_standalone:.4f}")
    print(f"   delta vs anchor (chemprop)  = {rae_standalone - rae_anchor:+.4f}")
    print(f"   delta vs nb2240 (0.4630)    = {rae_standalone - NB2240_REF:+.4f}")
    if blend_block:
        print(f"   0.5/0.5 blend with nb2240   = {blend_block['rae_blend_50_50_oof']:.4f}")
        print(f"   blend_helps                 = {blend_block['blend_helps']}")
    print(f"   wall                        = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "rae_standalone_bag",
        "delta_standalone_vs_anchor",
        "delta_standalone_vs_nb2240",
        "passed_standalone_gate",
        "blend_block",
        "te_unb_in_sample_rae",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
