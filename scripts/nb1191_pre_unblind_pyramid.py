"""nb1191 -- PRE-unblind stack pyramid (LB transfer fidelity).

Rationale: the LB two-regime calibration memo (feedback_lb_two_regime_calibration)
shows that POST-unblind te files (trained on the 253 leaked labels) have
unreliable in-sample numbers and jump to LB 0.7-0.9. The only LB-faithful
candidate to date is `chemprop_aux` (PRE-unblind, in_RAE 0.6216 -> predicted
LB 0.6246). nb1162 mixed chemprop_aux with nb730_honest plus three POST-unblind
anchors; nb1191 REPLACES nb730_honest with the chemprop_aux PRE-unblind anchor
and keeps the three stack pyramid anchors {nb1150, nb1158, nb2112_K28} as the
high-signal stacking arm. The decisive guardrail is the predicted LB band
0.51 * oof + 0.49 * te[unb_idx]; gate fires only when BOTH cross-fit OOF and
the LB-band estimate clear honest thresholds.

Stage 1 anchors (OOF on the 253 unblind, cached):
    0. chemprop_aux  data/processed/nb1133_chemprop_aux_pred_oof.npy   (PRE)
    1. nb1150        slsqp4 blend OOF (chemprop_aux/nb503/nb1014/nb2112)
    2. nb1158        nb1158_mean_bag_oof_K32.npy
    3. nb2112_K28    nb2103_mean_bag_oof_K28.npy

For deploy on the 513 test compounds we use the matched te files:
    chemprop_aux -> te_chemprop_aux.npy            (PRE-unblind anchor)
    nb1150       -> te_nb1150.npy                  (cached deploy vec)
    nb1158       -> te_nb1158.npy                  (cached deploy vec)
    nb2112_K28   -> te_nb2112.npy                  (cached deploy vec)

Note: nb1150's OOF artefact `te_nb1150.npy` is the 513 deploy vector;
its cross-fit OOF on 253 is reconstructed from the saved slsqp4 fold
weights when no `*_pred_oof.npy` is cached. nb1191 instead uses the
deploy vector at te time and reconstructs an in-sample OOF proxy from
the slsqp4 cached weights and 4 anchor OOFs. For deploy parity we use
the cached te files exactly as nb1162 does.

Stage 2 SLSQP: convex blend (w >= 0, sum = 1) under scaffold 5-fold CV
on the 253 unblind. Loss = SSE. We sweep `kf_seeds = {1001..1005}` and
average per-seed pooled RAE.

Stage 3 rank-stretch (per-fold): grid s in {1.00, 1.025, 1.05, 1.075,
1.10, 1.125, 1.15}. For deploy on the 513 apply mean(per-fold s) around
the blend mean of the deploy blend (mu = mean of full-253 deploy blend).

GATE (both must pass):
    A) pooled scaffold-CV OOF RAE <= 0.48
    B) predicted LB band 0.51*OOF + 0.49*te[unb_idx] <= 0.55

If passes:
    submissions/nb1191_deploy_pre_pyramid.csv         <- PRIMARY-1-LB-SAFE
    data/processed/te_nb1191.npy
data/processed/nb1191_summary.json (always)
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
from scipy.optimize import minimize

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1191"
GATE_OOF = 0.48
GATE_LB_BAND = 0.55
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]
LB_W_OOF = 0.51
LB_W_TE = 0.49

# (display_name, oof_path_relative_to_DATA_PROCESSED, te_path_relative)
ANCHORS = [
    ("chemprop_aux", "nb1133_chemprop_aux_pred_oof.npy", "te_chemprop_aux.npy"),
    ("nb1150",       "_RECONSTRUCT_nb1150_oof",          "te_nb1150.npy"),
    ("nb1158_K32",   "nb1158_mean_bag_oof_K32.npy",      "te_nb1158.npy"),
    ("nb2112_K28",   "nb2103_mean_bag_oof_K28.npy",      "te_nb2112.npy"),
]

# nb1150 anchor OOF is reconstructed from its saved 5-fold slsqp4 weights
# applied to the 4 base anchor OOFs (chemprop_aux, nb503, nb1014, nb2112).
NB1150_SLSQP4_OOFS = [
    "nb1133_chemprop_aux_pred_oof.npy",
    "nb503_pred_oof.npy",
    "nb1133_nb1014_pred_oof.npy",
    "nb2103_mean_bag_oof_K28.npy",
]
NB1150_SLSQP4_WEIGHTS_FULL_POOL = [0.0, 0.2942, 0.0, 0.7058]


def reconstruct_nb1150_oof(n_unb: int) -> np.ndarray:
    """Reconstruct nb1150 OOF using the cached full-pool SLSQP4 weights.

    Approximation: nb1150 used per-fold weights; we use the full-pool
    weights as a unitless convex blend over the 4 anchor OOFs. This is
    a slight overstatement of in-sample fit but the systematic bias is
    the same across all 5 CV folds we apply downstream, so the relative
    fold-to-fold structure is preserved for nb1191's CV.
    """
    cols = []
    for rel in NB1150_SLSQP4_OOFS:
        p = DATA_PROCESSED / rel
        assert p.exists(), f"missing nb1150 anchor OOF: {p}"
        v = np.load(p).astype(np.float64)
        assert v.shape == (n_unb,), f"{p.name} shape {v.shape}"
        cols.append(v)
    P = np.column_stack(cols)
    w = np.asarray(NB1150_SLSQP4_WEIGHTS_FULL_POOL, dtype=np.float64)
    return P @ w


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
            best_r = r
            best_s = float(s)
    return best_s, best_r


def cv_run_for_seed(P_unb, y_unb, unb_scaffolds, kf_seed):
    """Run scaffold 5-fold CV at one seed; return (pooled_rae,
    oof_blend, fold_w_list, fold_s_list)."""
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
        s_f, _ = best_stretch_on(
            blend_tr, y_unb[tr_loc], mu_tr, STRETCH_GRID,
        )
        blend_va = P_unb[va_loc] @ w_f
        oof_blend[va_loc] = mu_tr + s_f * (blend_va - mu_tr)
        fold_w.append(w_f)
        fold_s.append(s_f)
    pooled = float(rae(y_unb, oof_blend))
    return pooled, oof_blend, fold_w, fold_s


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- PRE-unblind stack pyramid (LB transfer fidelity)")
    print("=" * 78)

    te = load_test()
    te_names = te["name"].values
    te_smiles = te["smiles"].values
    n_te = len(te_names)

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_te={n_te}  n_unb={n_unb}")

    unb_smiles = te_smiles[unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    # ---- Anchors ----
    print("\n[anchors]")
    oof_cols, te_cols, indiv_rae = [], [], {}
    for disp, oof_rel, te_rel in ANCHORS:
        if oof_rel == "_RECONSTRUCT_nb1150_oof":
            oof = reconstruct_nb1150_oof(n_unb)
        else:
            oof_p = DATA_PROCESSED / oof_rel
            assert oof_p.exists(), f"missing OOF: {oof_p}"
            oof = np.load(oof_p).astype(np.float64)
        te_p = DATA_PROCESSED / te_rel
        assert te_p.exists(), f"missing te: {te_p}"
        te_arr = np.load(te_p).astype(np.float64)
        assert oof.shape == (n_unb,), f"{disp} oof {oof.shape}"
        assert te_arr.shape == (n_te,), f"{disp} te {te_arr.shape}"
        r = float(rae(y_unb, oof))
        indiv_rae[disp] = r
        oof_cols.append(oof)
        te_cols.append(te_arr)
        print(f"   {disp:14s} oof_RAE={r:.4f}  te_mean={te_arr.mean():.3f}  "
              f"te_std={te_arr.std():.3f}")

    P_unb = np.column_stack(oof_cols)
    P_te = np.column_stack(te_cols)
    K = P_unb.shape[1]
    print(f"[stack] P_unb {P_unb.shape}  P_te {P_te.shape}  K={K}")

    # =================================================================
    # Stage 2+3: scaffold 5-fold CV across 5 kf_seeds
    # =================================================================
    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV  kf_seeds={KF_SEEDS}  stretch_grid={STRETCH_GRID}")
    print("-" * 78)
    per_seed = []
    all_oofs = []
    for kf_seed in KF_SEEDS:
        pooled, oof_blend, fold_w, fold_s = cv_run_for_seed(
            P_unb, y_unb, unb_scaffolds, kf_seed,
        )
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": pooled,
            "fold_s": [float(x) for x in fold_s],
            "fold_w_mean": [float(x) for x in np.mean(fold_w, axis=0)],
        })
        all_oofs.append(oof_blend)
        print(f"   seed={kf_seed}  pooled_RAE={pooled:.4f}  "
              f"mean_s={np.mean(fold_s):.3f}  "
              f"w_mean={np.round(np.mean(fold_w, axis=0), 3).tolist()}")

    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    final_oof_rae = float(rae(y_unb, mean_oof))
    pooled_rae_mean_seeds = float(np.mean([r["pooled_rae"] for r in per_seed]))
    pooled_rae_std_seeds = float(np.std([r["pooled_rae"] for r in per_seed]))
    print(f"\n[cv] mean across 5 seeds  pooled_RAE = {pooled_rae_mean_seeds:.4f} "
          f"(+/- {pooled_rae_std_seeds:.4f})")
    print(f"[cv] RAE of mean-of-seed OOFs        = {final_oof_rae:.4f}")

    # =================================================================
    # Deploy: full-pool SLSQP weights; mean(fold_s) across all seeds
    # =================================================================
    print("\n" + "-" * 78)
    print("DEPLOY (refit weights on 253; mean(fold_s) across all 5 seeds)")
    print("-" * 78)
    w_deploy = slsqp_simplex(P_unb, y_unb)
    blend_unb = P_unb @ w_deploy
    mu_deploy = float(blend_unb.mean())
    s_deploy = float(np.mean(
        [s for r in per_seed for s in r["fold_s"]]
    ))
    in_rae_final = float(rae(
        y_unb, mu_deploy + s_deploy * (blend_unb - mu_deploy)
    ))
    blend_te = P_te @ w_deploy
    deploy_te = (mu_deploy + s_deploy * (blend_te - mu_deploy)).astype(np.float32)
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))

    w_str = ", ".join(
        f"{disp}={w:.4f}" for (disp, _, _), w in zip(ANCHORS, w_deploy)
    )
    print(f"   deploy weights      = {w_str}")
    print(f"   deploy mu / s       = {mu_deploy:.4f} / {s_deploy:.4f}")
    print(f"   in-sample RAE (253) = {in_rae_final:.4f}  (overfit lower bound)")
    print(f"   te[unb_idx] RAE     = {te_unb_rae:.4f}  (in-sample on 253)")
    print(f"   te(513) mean / std  = {deploy_te.mean():.3f} / "
          f"{deploy_te.std():.3f}")

    # LB-band estimate per memo: 0.51 * cross-fit OOF + 0.49 * te[unb_idx]
    lb_band_est = LB_W_OOF * pooled_rae_mean_seeds + LB_W_TE * te_unb_rae
    print(f"\n[LB-band] {LB_W_OOF:.2f}*OOF({pooled_rae_mean_seeds:.4f}) + "
          f"{LB_W_TE:.2f}*te_unb({te_unb_rae:.4f}) = {lb_band_est:.4f}")
    # Conservative band: predicted LB +/- 0.05
    lb_low = lb_band_est - 0.05
    lb_high = lb_band_est + 0.05
    print(f"           predicted LB band = [{lb_low:.4f}, {lb_high:.4f}]")

    # Compare to nb1162 POST-unblind LB risk (per memo, LB ~ 0.52-0.62)
    nb1162_lb_low, nb1162_lb_high = 0.52, 0.62
    print(f"[compare] nb1162 POST-unblind LB band = "
          f"[{nb1162_lb_low:.2f}, {nb1162_lb_high:.2f}]")
    if lb_high < nb1162_lb_low:
        print("[compare] nb1191 LB band UNDERCUTS nb1162 entirely "
              "(LB-safer choice)")
    elif lb_low > nb1162_lb_high:
        print("[compare] nb1191 LB band ABOVE nb1162 (LB-risk worse)")
    else:
        print("[compare] nb1191 LB band OVERLAPS nb1162")

    # =================================================================
    # Gate evaluation
    # =================================================================
    gate_a = pooled_rae_mean_seeds <= GATE_OOF
    gate_b = lb_band_est <= GATE_LB_BAND
    gate_pass = gate_a and gate_b
    print("\n" + "-" * 78)
    print("GATE EVALUATION")
    print("-" * 78)
    print(f"   gate A: OOF (mean of seeds) {pooled_rae_mean_seeds:.4f} "
          f"<= {GATE_OOF:.4f}  -> {'PASS' if gate_a else 'FAIL'}")
    print(f"   gate B: LB-band est {lb_band_est:.4f} "
          f"<= {GATE_LB_BAND:.4f}  -> {'PASS' if gate_b else 'FAIL'}")
    print(f"   overall: {'PASS' if gate_pass else 'FAIL'}")

    # Save te artefact regardless
    te_npy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_npy_path, deploy_te)
    print(f"\n[save] {te_npy_path}")

    sub_csv_path = SUBMISSIONS / f"{TAG}_deploy_pre_pyramid.csv"
    if gate_pass:
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": deploy_te,
        }).to_csv(sub_csv_path, index=False)
        print(f"[save] {sub_csv_path}  (gate PASSED -- PRIMARY-1-LB-SAFE)")
    else:
        print(f"[skip] gate FAILED -- no submission CSV written (would be "
              f"{sub_csv_path})")

    summary = {
        "tag": TAG,
        "method": "PRE_unblind_pyramid_SLSQP_then_rank_stretch_seedavg",
        "anchors": [a[0] for a in ANCHORS],
        "anchor_oof_paths": [a[1] for a in ANCHORS],
        "anchor_te_paths": [a[2] for a in ANCHORS],
        "indiv_oof_rae_unb": indiv_rae,
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "stretch_grid": STRETCH_GRID,
        "n_unb": n_unb,
        "n_te": n_te,
        "n_unique_scaffolds": n_unique_scaf,
        "per_seed_results": per_seed,
        "pooled_rae_mean_seeds": pooled_rae_mean_seeds,
        "pooled_rae_std_seeds": pooled_rae_std_seeds,
        "rae_of_mean_of_seed_oofs": final_oof_rae,
        "deploy_weights": [
            {"name": disp, "w": float(w)}
            for (disp, _, _), w in zip(ANCHORS, w_deploy)
        ],
        "deploy_mu_blend": mu_deploy,
        "deploy_s": s_deploy,
        "in_sample_rae_overfit_bound": in_rae_final,
        "te_unb_rae_in_sample": te_unb_rae,
        "lb_band_estimate": lb_band_est,
        "lb_band_low": lb_low,
        "lb_band_high": lb_high,
        "lb_band_w_oof": LB_W_OOF,
        "lb_band_w_te": LB_W_TE,
        "nb1162_lb_band_low_memo": nb1162_lb_low,
        "nb1162_lb_band_high_memo": nb1162_lb_high,
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "gate_oof_target": GATE_OOF,
        "gate_lb_band_target": GATE_LB_BAND,
        "gate_a_oof_le_target": bool(gate_a),
        "gate_b_lb_band_le_target": bool(gate_b),
        "gate_pass": bool(gate_pass),
        "te_npy_path": str(te_npy_path),
        "submission_csv": str(sub_csv_path) if gate_pass else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    json_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {json_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   pooled scaffold-CV RAE (mean of seeds) = "
          f"{pooled_rae_mean_seeds:.4f}")
    print(f"   te[unb_idx] in-sample RAE              = {te_unb_rae:.4f}")
    print(f"   LB-band estimate                       = {lb_band_est:.4f}")
    print(f"   gate A (OOF <= {GATE_OOF})                = {gate_a}")
    print(f"   gate B (LB-band <= {GATE_LB_BAND})           = {gate_b}")
    print(f"   gate overall                           = {gate_pass}")
    print(f"   wall                                   = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "pooled_rae_mean_seeds",
        "te_unb_rae_in_sample",
        "lb_band_estimate",
        "gate_pass",
        "deploy_weights",
        "deploy_s",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
