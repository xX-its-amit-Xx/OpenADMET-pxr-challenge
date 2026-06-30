"""nb2450 -- Tanimoto Kernel SVR on chemprop_aux residual.

Different model class than trees/GBM. Kernel method: SVR with
precomputed Tanimoto kernel over Morgan FP-2048 (radius 2).

PROTOCOL
    1. Compute Morgan FP-2048 (radius 2) on 4139 train + 253 unblind + 513 test
       (we only need 253 + 513 for residual modeling; 4139 computed for any
       optional pool/inspection downstream).
    2. Custom Tanimoto kernel: K(x,y) = |x AND y| / |x OR y|  (uint8 bit FPs).
       K_train_train (253x253), K_test_train (513x253).
    3. SVR(epsilon=0.1, kernel='precomputed') with C grid {0.1, 1, 10}.
       Trained on chemprop_aux RESIDUAL (y_unb - te_chemprop_aux[unb_idx]).
    4. 5-fold scaffold CV on the 253; per fold we re-slice the precomputed
       kernel to (n_tr,n_tr) for fit and (n_va,n_tr) for predict. Pick C
       with best pooled OOF RAE on (anchor + residual_pred).
    5. Standalone RAE = rae(y_unb, anchor + oof_residual). Compare to nb2240
       0.4630 anchor.
    6. If standalone < 0.55: SLSQP convex blend with the nb2240 pyramid
       (5 anchors: nb2240_K20, chemprop_aux, nb1191, nb503, nb562) plus
       the new nb2450 anchor (6 columns), 5-fold scaffold CV, rank-stretch.
    7. Always save nb2450_summary.json + nb2450_pred_oof.npy + te_nb2450.npy.

Outputs
    scripts/nb2450_tanimoto_svr.py
    data/processed/nb2450_summary.json
    data/processed/nb2450_pred_oof.npy   (253,) float32
    data/processed/te_nb2450.npy         (513,) float32
    submissions/nb2450_tanimoto_svr.csv  (only on blend-gate pass)
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
from scipy.optimize import minimize
from sklearn.svm import SVR

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko, morgan_fp_batch
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2450"
FP_RADIUS = 2
FP_BITS = 2048

C_GRID = [0.1, 1.0, 10.0]
EPSILON = 0.1
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]
LB_W_OOF = 0.51
LB_W_TE = 0.49

NB2240_REF_OOF = 0.4630          # nb2240_K20 mean-bag oof RAE on 253
STANDALONE_BLEND_THRESH = 0.55   # standalone < 0.55 -> trigger pyramid blend


# ---------------------------------------------------------------------------
# Tanimoto kernel on uint8 bit FPs
# ---------------------------------------------------------------------------

def tanimoto_kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """K[i,j] = |A_i AND B_j| / |A_i OR B_j| for binary bit-vectors.

    A: (nA, n_bits) uint8, B: (nB, n_bits) uint8. Returns (nA, nB) float32.
    """
    A = A.astype(np.float32, copy=False)
    B = B.astype(np.float32, copy=False)
    inter = A @ B.T                          # (nA, nB)
    a_sum = A.sum(axis=1)                    # (nA,)
    b_sum = B.sum(axis=1)                    # (nB,)
    union = a_sum[:, None] + b_sum[None, :] - inter
    union = np.maximum(union, 1.0)
    return (inter / union).astype(np.float32)


# ---------------------------------------------------------------------------
# SLSQP simplex + rank-stretch (matches nb2240 / nb2171 lineage)
# ---------------------------------------------------------------------------

def slsqp_simplex(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    K = P.shape[1]
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * K
    res = minimize(
        lambda w: float(np.sum((P @ w - y) ** 2)),
        np.full(K, 1.0 / K),
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
        options={"ftol": 1e-10, "maxiter": 1000},
    )
    w = np.clip(res.x, 0.0, 1.0)
    s = w.sum()
    return w / s if s > 0 else np.full(K, 1.0 / K)


def best_stretch_on(blend_tr, y_tr, mu, grid):
    best_s, best_r = 1.0, float("inf")
    for s in grid:
        pred = mu + s * (blend_tr - mu)
        r = float(rae(y_tr, pred))
        if r < best_r:
            best_r, best_s = r, float(s)
    return best_s, best_r


def cv_run_for_seed(P_unb, y_unb, unb_scaffolds, kf_seed):
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = P_unb.shape[0]
    oof_blend = np.full(n_unb, np.nan)
    fold_w, fold_s = [], []
    for tr_loc, va_loc in splits:
        w_f = slsqp_simplex(P_unb[tr_loc], y_unb[tr_loc])
        blend_tr = P_unb[tr_loc] @ w_f
        mu_tr = float(blend_tr.mean())
        s_f, _ = best_stretch_on(blend_tr, y_unb[tr_loc], mu_tr, STRETCH_GRID)
        blend_va = P_unb[va_loc] @ w_f
        oof_blend[va_loc] = mu_tr + s_f * (blend_va - mu_tr)
        fold_w.append(w_f)
        fold_s.append(s_f)
    return float(rae(y_unb, oof_blend)), oof_blend, fold_w, fold_s


# ---------------------------------------------------------------------------
# SVR with precomputed Tanimoto kernel
# ---------------------------------------------------------------------------

def svr_cv_oof_for_C(K_unb_unb: np.ndarray,
                     residual: np.ndarray,
                     anchor_unb: np.ndarray,
                     y_unb: np.ndarray,
                     unb_scaffolds,
                     C: float,
                     kf_seed: int = 1001):
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(residual)
    oof_resid = np.full(n_unb, np.nan, dtype=np.float64)
    for tr_loc, va_loc in splits:
        K_tr = K_unb_unb[np.ix_(tr_loc, tr_loc)]
        K_va = K_unb_unb[np.ix_(va_loc, tr_loc)]
        mdl = SVR(kernel="precomputed", C=C, epsilon=EPSILON)
        mdl.fit(K_tr, residual[tr_loc])
        oof_resid[va_loc] = mdl.predict(K_va)
    oof_corrected = anchor_unb + oof_resid
    return float(rae(y_unb, oof_corrected)), oof_resid, oof_corrected


def svr_deploy_te(K_unb_unb, K_te_unb, residual, C):
    mdl = SVR(kernel="precomputed", C=C, epsilon=EPSILON)
    mdl.fit(K_unb_unb, residual)
    return mdl.predict(K_te_unb).astype(np.float32)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Tanimoto Kernel SVR on chemprop_aux residual")
    print("=" * 78)

    # ---- Load truth, anchor, test ----
    te = load_test()
    n_test = len(te)
    te_smiles = te["smiles"].astype(str).tolist() if "smiles" in te.columns \
        else te["SMILES"].astype(str).tolist()
    te_names = te["name"].values if "name" in te.columns else te["Molecule Name"].values

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    te_anchor_513 = np.load(DATA_PROCESSED / "te_chemprop_aux.npy").astype(np.float64)
    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    print(f"[load] chemprop_aux in_RAE = {rae_anchor:.4f}")
    residual = y_unb - anchor_unb

    # ---- Morgan FP-2048 r=2 on 4139 + 253 + 513 (we materialise all three) ----
    print(f"\n[fp] computing Morgan FP-{FP_BITS} radius={FP_RADIUS} on 4139 + 253 + 513 ...")
    tr_df = load_train()
    tr_smiles = tr_df["smiles"].astype(str).tolist()
    t_fp = time.time()
    fp_train = morgan_fp_batch(tr_smiles, radius=FP_RADIUS, n_bits=FP_BITS)
    fp_test = morgan_fp_batch(te_smiles,  radius=FP_RADIUS, n_bits=FP_BITS)
    fp_unb = fp_test[unb_idx]
    print(f"   shapes train={fp_train.shape}  test={fp_test.shape}  unb={fp_unb.shape}  "
          f"({time.time() - t_fp:.1f}s)")

    # ---- Tanimoto kernel matrices ----
    print("\n[kernel] computing Tanimoto kernel matrices ...")
    t_k = time.time()
    K_unb_unb = tanimoto_kernel(fp_unb, fp_unb)     # (253, 253)
    K_te_unb = tanimoto_kernel(fp_test, fp_unb)     # (513, 253)
    diag_med = float(np.median(np.diag(K_unb_unb)))
    off_med = float(np.median(K_unb_unb[~np.eye(n_unb, dtype=bool)]))
    print(f"   K_unb_unb {K_unb_unb.shape}  diag_med={diag_med:.3f}  off_med={off_med:.3f}")
    print(f"   K_te_unb {K_te_unb.shape}  median={float(np.median(K_te_unb)):.3f}  "
          f"({time.time() - t_k:.1f}s)")

    # ---- SVR C-grid with 5-fold scaffold CV on residual ----
    print("\n" + "-" * 78)
    print(f"SVR(epsilon={EPSILON}, kernel='precomputed') over C-grid {C_GRID}")
    print("-" * 78)
    c_results = []
    best_C, best_rae, best_oof_resid, best_oof_corrected = None, float("inf"), None, None
    for C in C_GRID:
        tC = time.time()
        rae_C, oof_resid_C, oof_corrected_C = svr_cv_oof_for_C(
            K_unb_unb, residual, anchor_unb, y_unb, unb_scaffolds, C=C, kf_seed=KF_SEEDS[0],
        )
        c_results.append({
            "C": float(C),
            "rae_corrected_oof": float(rae_C),
            "residual_oof_mean": float(np.mean(oof_resid_C)),
            "residual_oof_std": float(np.std(oof_resid_C)),
            "wall_sec": round(time.time() - tC, 2),
        })
        print(f"   C={C:>5g}  rae_oof_corrected={rae_C:.4f}  "
              f"resid_mean={float(np.mean(oof_resid_C)):+.3f}  "
              f"resid_std={float(np.std(oof_resid_C)):.3f}  "
              f"({time.time() - tC:.1f}s)")
        if rae_C < best_rae:
            best_rae = rae_C
            best_C = float(C)
            best_oof_resid = oof_resid_C
            best_oof_corrected = oof_corrected_C
    print(f"\n[selected] best_C={best_C}  standalone_oof_RAE={best_rae:.4f}")

    # ---- Deploy: refit SVR on all 253 with best C; predict residual on 513 ----
    print(f"\n[deploy] SVR(C={best_C}) refit on all 253; predict te(513) residual")
    te_resid = svr_deploy_te(K_unb_unb, K_te_unb, residual, C=best_C).astype(np.float32)
    te_nb2450_513 = (te_anchor_513.astype(np.float32) + te_resid).astype(np.float32)
    te_unb_insample_rae = float(rae(y_unb, te_nb2450_513[unb_idx]))
    print(f"   te_nb2450 mean/std = {te_nb2450_513.mean():.3f}/{te_nb2450_513.std():.3f}")
    print(f"   te[unb_idx] in-sample RAE = {te_unb_insample_rae:.4f}")

    # Save standalone artefacts
    pred_oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(pred_oof_path, best_oof_corrected.astype(np.float32))
    np.save(te_path, te_nb2450_513)
    print(f"[save] {pred_oof_path}")
    print(f"[save] {te_path}")

    # ---- Compare standalone vs nb2240 anchor ----
    delta_vs_nb2240 = best_rae - NB2240_REF_OOF
    print("\n" + "-" * 78)
    print(f"STANDALONE COMPARE vs nb2240 K20 ref {NB2240_REF_OOF:.4f}")
    print("-" * 78)
    print(f"   nb2450 standalone OOF = {best_rae:.4f}")
    print(f"   nb2240 reference      = {NB2240_REF_OOF:.4f}")
    print(f"   delta                 = {delta_vs_nb2240:+.4f}")

    # ---- If standalone < 0.55: SLSQP blend with nb2240 pyramid ----
    blend_summary = None
    sub_csv_path = SUBMISSIONS / f"{TAG}_tanimoto_svr.csv"
    if best_rae < STANDALONE_BLEND_THRESH:
        print("\n" + "=" * 78)
        print("STAGE 2: SLSQP BLEND WITH NB2240 PYRAMID (6 columns)")
        print("=" * 78)

        nb2240_oof = np.load(DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy").astype(np.float64)
        chemprop_oof = np.load(DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy").astype(np.float64)
        nb503_oof = np.load(DATA_PROCESSED / "nb503_pred_oof.npy").astype(np.float64)
        nb562_oof = np.load(DATA_PROCESSED / "nb562_pred_oof.npy").astype(np.float64)

        te_nb2240 = np.load(DATA_PROCESSED / "te_nb2240_K20.npy").astype(np.float64)
        te_nb503 = np.load(DATA_PROCESSED / "te_nb503.npy").astype(np.float64)
        te_nb562 = np.load(DATA_PROCESSED / "te_nb562.npy").astype(np.float64)

        # nb1191 oof: deploy weights stamp from nb2171 lineage. Use cached nb1191
        # reconstruction from nb2240_summary if present; otherwise skip 1191.
        nb1191_oof = None
        te_nb1191 = None
        p1191_oof = DATA_PROCESSED / "nb1191_pred_oof.npy"
        p1191_te = DATA_PROCESSED / "te_nb1191.npy"
        if p1191_oof.exists() and p1191_te.exists():
            nb1191_oof = np.load(p1191_oof).astype(np.float64)
            te_nb1191 = np.load(p1191_te).astype(np.float64)

        anchors = [
            ("nb2450_svr",   best_oof_corrected.astype(np.float64),
             te_nb2450_513.astype(np.float64)),
            ("nb2240_K20",   nb2240_oof, te_nb2240),
            ("chemprop_aux", chemprop_oof, te_anchor_513),
            ("nb503",        nb503_oof, te_nb503),
            ("nb562",        nb562_oof, te_nb562),
        ]
        if nb1191_oof is not None:
            anchors.insert(2, ("nb1191", nb1191_oof, te_nb1191))

        indiv_rae = {}
        oof_cols, te_cols = [], []
        print("\n[anchors]")
        for disp, oof, te_arr in anchors:
            assert oof.shape == (n_unb,), f"{disp} oof {oof.shape}"
            assert te_arr.shape == (n_test,), f"{disp} te {te_arr.shape}"
            r = float(rae(y_unb, oof))
            indiv_rae[disp] = r
            oof_cols.append(oof)
            te_cols.append(te_arr)
            print(f"   {disp:14s} oof_RAE={r:.4f}  te_mean={te_arr.mean():.3f}  "
                  f"te_std={te_arr.std():.3f}")
        P_unb = np.column_stack(oof_cols)
        P_te = np.column_stack(te_cols)
        K_anch = P_unb.shape[1]
        print(f"[stack] P_unb {P_unb.shape}  P_te {P_te.shape}  K={K_anch}")

        per_seed = []
        all_oofs = []
        for kf_seed in KF_SEEDS:
            pooled, oof_blend, fw, fs = cv_run_for_seed(P_unb, y_unb, unb_scaffolds, kf_seed)
            per_seed.append({
                "kf_seed": int(kf_seed),
                "pooled_rae": float(pooled),
                "fold_s": [float(x) for x in fs],
                "fold_w_mean": [float(x) for x in np.mean(fw, axis=0)],
            })
            all_oofs.append(oof_blend)
            print(f"   seed={kf_seed}  pooled_RAE={pooled:.4f}  mean_s={np.mean(fs):.3f}  "
                  f"w_mean={np.round(np.mean(fw, axis=0), 3).tolist()}")
        mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
        final_oof_rae = float(rae(y_unb, mean_oof))
        pooled_rae_mean_seeds = float(np.mean([r["pooled_rae"] for r in per_seed]))
        pooled_rae_std_seeds = float(np.std([r["pooled_rae"] for r in per_seed]))
        print(f"\n[blend cv] mean across {len(KF_SEEDS)} seeds  pooled_RAE = "
              f"{pooled_rae_mean_seeds:.4f} (+/- {pooled_rae_std_seeds:.4f})")
        print(f"[blend cv] RAE of mean-of-seed OOFs = {final_oof_rae:.4f}")

        # Deploy
        w_deploy = slsqp_simplex(P_unb, y_unb)
        blend_unb = P_unb @ w_deploy
        mu_deploy = float(blend_unb.mean())
        s_deploy = float(np.mean([s for r in per_seed for s in r["fold_s"]]))
        in_rae_final = float(rae(y_unb, mu_deploy + s_deploy * (blend_unb - mu_deploy)))
        blend_te = P_te @ w_deploy
        deploy_te = (mu_deploy + s_deploy * (blend_te - mu_deploy)).astype(np.float32)
        te_unb_blend_insample = float(rae(y_unb, deploy_te[unb_idx]))
        lb_band = LB_W_OOF * pooled_rae_mean_seeds + LB_W_TE * te_unb_blend_insample
        nb2450_w = float(w_deploy[0])

        print("\n[blend deploy]")
        for (disp, _, _), w in zip(anchors, w_deploy):
            print(f"   {disp:14s} w={float(w):.4f}")
        print(f"   mu={mu_deploy:.4f}  s={s_deploy:.4f}  "
              f"in_rae={in_rae_final:.4f}  te[unb]={te_unb_blend_insample:.4f}")
        print(f"   LB band estimate = {lb_band:.4f}")

        # Gate: nb2450 weight > 1% in deploy AND pooled improves vs nb2240 by >= 0.003
        blend_gate_beat = (
            nb2450_w > 0.01 and pooled_rae_mean_seeds < NB2240_REF_OOF - 0.003
        )
        if blend_gate_beat:
            pd.DataFrame({
                "SMILES": te_smiles,
                "Molecule Name": te_names,
                "pEC50": deploy_te,
            }).to_csv(sub_csv_path, index=False)
            print(f"[save] {sub_csv_path}  (blend GATE BEAT)")
        else:
            print(f"[skip] blend gate not beat -- no submission CSV written "
                  f"(nb2450_w={nb2450_w:.4f}, pooled={pooled_rae_mean_seeds:.4f})")

        blend_summary = {
            "anchors": [a[0] for a in anchors],
            "anchor_oof_rae_unb": indiv_rae,
            "per_seed_results": per_seed,
            "pooled_rae_mean_seeds": pooled_rae_mean_seeds,
            "pooled_rae_std_seeds": pooled_rae_std_seeds,
            "rae_of_mean_of_seed_oofs": final_oof_rae,
            "deploy_weights": [
                {"name": disp, "w": float(w)}
                for (disp, _, _), w in zip(anchors, w_deploy)
            ],
            "nb2450_deploy_weight": nb2450_w,
            "deploy_mu_blend": mu_deploy,
            "deploy_s": s_deploy,
            "in_sample_rae_overfit_bound": in_rae_final,
            "te_unb_rae_in_sample": te_unb_blend_insample,
            "lb_band_estimate": lb_band,
            "lb_band_w_oof": LB_W_OOF,
            "lb_band_w_te": LB_W_TE,
            "blend_gate_beat_nb2240": bool(blend_gate_beat),
            "submission_csv": str(sub_csv_path) if blend_gate_beat else None,
        }
    else:
        print(f"\n[skip blend] standalone {best_rae:.4f} >= threshold "
              f"{STANDALONE_BLEND_THRESH} -- no SLSQP blend run")

    summary = {
        "tag": TAG,
        "method": "tanimoto_kernel_SVR_on_chemprop_aux_residual",
        "fp_radius": FP_RADIUS,
        "fp_bits": FP_BITS,
        "epsilon": EPSILON,
        "C_grid": C_GRID,
        "n_unb": n_unb,
        "n_te": n_test,
        "n_train": int(fp_train.shape[0]),
        "n_unique_scaffolds": n_unique_scaf,
        "rae_anchor_chemprop_aux": rae_anchor,
        "c_grid_results": c_results,
        "best_C": best_C,
        "standalone_oof_rae": float(best_rae),
        "nb2240_ref_oof": NB2240_REF_OOF,
        "delta_vs_nb2240": float(delta_vs_nb2240),
        "te_unb_in_sample_rae": te_unb_insample_rae,
        "kernel_diag_median": diag_med,
        "kernel_off_median": off_med,
        "pred_oof_path": str(pred_oof_path),
        "te_path": str(te_path),
        "standalone_blend_threshold": STANDALONE_BLEND_THRESH,
        "blend_triggered": bool(best_rae < STANDALONE_BLEND_THRESH),
        "blend_results": blend_summary,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   anchor chemprop_aux RAE      = {rae_anchor:.4f}")
    print(f"   best C                       = {best_C}")
    print(f"   standalone OOF RAE           = {best_rae:.4f}")
    print(f"   delta vs nb2240 (0.4630)     = {delta_vs_nb2240:+.4f}")
    print(f"   blend triggered (<0.55)      = {best_rae < STANDALONE_BLEND_THRESH}")
    if blend_summary is not None:
        print(f"   blend pooled OOF (5 seeds)   = {blend_summary['pooled_rae_mean_seeds']:.4f}")
        print(f"   nb2450 deploy weight         = {blend_summary['nb2450_deploy_weight']:.4f}")
        print(f"   blend gate beat nb2240       = {blend_summary['blend_gate_beat_nb2240']}")
    print(f"   wall                          = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "best_C",
        "standalone_oof_rae",
        "delta_vs_nb2240",
        "blend_triggered",
    ):
        print(f"  {k}: {res.get(k)}")
    if res.get("blend_results") is not None:
        br = res["blend_results"]
        for k in (
            "pooled_rae_mean_seeds",
            "nb2450_deploy_weight",
            "blend_gate_beat_nb2240",
            "lb_band_estimate",
        ):
            print(f"  {k}: {br.get(k)}")
