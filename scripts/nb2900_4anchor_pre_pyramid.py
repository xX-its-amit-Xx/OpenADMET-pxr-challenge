"""nb2900 -- 4-anchor PRE-clean SLSQP pyramid {nb2240_K20, chemprop_aux, counter_clean, nb1191}.

NEW PARADIGM: now that nb1191 OOF exists (PRE-clean), build full 4-anchor pyramid
(vs prior 3-anchor variants). counter_clean is the counter-axis signal lifted onto
the 253 unblind via te_nb2490_counter[unb_idx] and onto the 513 test via the same
te-array (refit on 2858, no nb730 contamination).

ANCHORS (all PRE-clean, no nb730/POST-unblind contamination):
    nb2240_K20      RAE 0.4630  (K=20 RFE residual on chemprop_aux)
    chemprop_aux    RAE 0.5879  (frozen PRE-unblind chemprop_v2 multitask)
    counter_clean   RAE 2.138   (counter-axis -- different y-axis, biological orth)
    nb1191          RAE 0.4697  (5-seed PRE-pyramid post-hoc-blend)

PROTOCOL:
    - SLSQP simplex blend (w >= 0, sum = 1) per scaffold-fold
    - Rank-stretch (grid 1.000..1.150) per fold
    - 5-fold scaffold-CV on 253, 5 fresh seeds {1001..1005}
    - Pooled RAE mean over seeds = verdict
    - Save te_<tag>.npy + pred_oof (mean-of-seed OOFs)

GATE: mean_rae < 0.4570 -> "PROMOTE"
      mean_rae < 0.4598 -> "MARGINAL_BEAT"
      else              -> "FAIL"

Reference: nb2240 ref 0.4601 (5-anchor pyramid pooled OOF), nb2171 ref 0.4682
(K=28 swap), nb1191/2095/2060 cluster 0.4718-0.4720.

Outputs:
    scripts/nb2900_4anchor_pre_pyramid.py
    data/processed/nb2900_summary.json
    data/processed/nb2900_pred_oof.npy            (253,) mean-of-seed OOFs
    data/processed/te_nb2900.npy                  (513,) deploy
    submissions/nb2900_4anchor_pre_pyramid.csv    (only on PROMOTE)
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

TAG = "nb2900"

# ---- Gates ----
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

# ---- CV protocol ----
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]
LB_W_OOF = 0.51
LB_W_TE = 0.49

# ---- 4 PRE-clean anchors ----
# Each entry: (display_name, oof_path_relative_or_None, te_path_relative,
#              use_te_unb_for_oof) -- if oof path is None we lift te[unb_idx]
ANCHORS = [
    # (name, oof_rel, te_rel, lift_te_to_oof)
    ("nb2240_K20",    "nb2240_mean_bag_oof_K20.npy",       "te_nb2240_K20.npy",     False),
    ("chemprop_aux",  "nb1133_chemprop_aux_pred_oof.npy",  "te_chemprop_aux.npy",   False),
    ("counter_clean", None,                                "te_nb2490_counter.npy", True),
    ("nb1191",        "nb1191_pred_oof.npy",               "te_nb1191.npy",         False),
]


# ============================================================================
# helpers
# ============================================================================

def slsqp_simplex(P, y):
    """SLSQP simplex (w >= 0, sum w = 1) minimizing |P @ w - y|^2."""
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
    """Pick best rank-stretch scalar on training fold."""
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


def load_anchor_columns(anchor_list, unb_idx, n_unb, n_te):
    """Load (n_unb, K) and (n_te, K) anchor stacks."""
    oof_cols, te_cols, names = [], [], []
    for disp, oof_rel, te_rel, lift in anchor_list:
        te_p = DATA_PROCESSED / te_rel
        assert te_p.exists(), f"missing te: {te_p}"
        te_arr = np.load(te_p).astype(np.float64)
        assert te_arr.shape == (n_te,), f"{disp} te {te_arr.shape}"

        if lift:
            # counter_clean lives on (513,) only; lift te[unb_idx] -> oof
            oof = te_arr[unb_idx].copy()
        else:
            oof_p = DATA_PROCESSED / oof_rel
            assert oof_p.exists(), f"missing OOF: {oof_p}"
            oof = np.load(oof_p).astype(np.float64)
            assert oof.shape == (n_unb,), f"{disp} oof {oof.shape}"

        oof_cols.append(oof)
        te_cols.append(te_arr)
        names.append(disp)
    return np.column_stack(oof_cols), np.column_stack(te_cols), names


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- 4-anchor PRE-clean SLSQP pyramid")
    print("=" * 78)
    print(f"   anchors  : {[a[0] for a in ANCHORS]}")
    print(f"   kf_seeds : {KF_SEEDS} ({len(KF_SEEDS)} seeds)")
    print(f"   gates    : PROMOTE < {GATE_PROMOTE}  MARGINAL < {GATE_MARGINAL}")

    te = load_test()
    te_names = te["name"].values if "name" in te.columns else te["Molecule Name"].values
    te_smiles = (
        te["smiles"].values if "smiles" in te.columns else te["SMILES"].values
    )
    n_te = len(te_names)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"\n[load] n_te={n_te}  n_unb={n_unb}  unique_scaffolds={n_unique_scaf}")

    # Load 4-anchor stack
    P_unb, P_te, names = load_anchor_columns(ANCHORS, unb_idx, n_unb, n_te)
    print(f"[stack] P_unb {P_unb.shape}  P_te {P_te.shape}")
    indiv_rae = {}
    for j, nm in enumerate(names):
        r = float(rae(y_unb, P_unb[:, j]))
        indiv_rae[nm] = r
        print(
            f"   anchor {j} {nm:14s} oof_RAE={r:.4f}  "
            f"mean={P_unb[:, j].mean():.4f}  std={P_unb[:, j].std():.4f}"
        )

    # 5-seed sweep
    print(f"\n[CV] kf_seeds={KF_SEEDS[0]}..{KF_SEEDS[-1]}  n_seeds={len(KF_SEEDS)}")
    per_seed = []
    all_oofs = []
    t_seed_start = time.time()
    for k, kf_seed in enumerate(KF_SEEDS):
        pooled, oof_blend, fw, fs = cv_run_for_seed(
            P_unb, y_unb, unb_scaffolds, kf_seed
        )
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": float(pooled),
            "fold_s_mean": float(np.mean(fs)),
            "fold_w_mean": [float(x) for x in np.mean(fw, axis=0)],
        })
        all_oofs.append(oof_blend)
        print(
            f"   seed[{k+1:2d}/{len(KF_SEEDS)}]={kf_seed}  "
            f"pooled={pooled:.4f}  mean_s={np.mean(fs):.3f}  "
            f"w_mean=[" + ", ".join(f"{x:.3f}" for x in np.mean(fw, axis=0)) + "]  "
            f"wall={time.time()-t_seed_start:.1f}s"
        )

    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    pooled_arr = np.asarray([r["pooled_rae"] for r in per_seed])
    pooled_mean = float(pooled_arr.mean())
    pooled_std = float(pooled_arr.std())
    pooled_min = float(pooled_arr.min())
    pooled_max = float(pooled_arr.max())
    rae_of_mean_oof = float(rae(y_unb, mean_oof))
    print(f"\n[CV] pooled_RAE mean = {pooled_mean:.4f} +/- {pooled_std:.4f}  "
          f"[{pooled_min:.4f}, {pooled_max:.4f}]")
    print(f"[CV] RAE(mean-of-seed OOFs)          = {rae_of_mean_oof:.4f}")

    # Deploy: full-pool SLSQP weights + mean of per-fold s
    w_deploy = slsqp_simplex(P_unb, y_unb)
    blend_unb = P_unb @ w_deploy
    mu_deploy = float(blend_unb.mean())
    s_deploy = float(np.mean(
        [s for r in per_seed for s in [r["fold_s_mean"]]]
    ))
    in_rae = float(rae(y_unb, mu_deploy + s_deploy * (blend_unb - mu_deploy)))
    blend_te = P_te @ w_deploy
    deploy_te = (mu_deploy + s_deploy * (blend_te - mu_deploy)).astype(np.float32)
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))
    lb_band_est = LB_W_OOF * pooled_mean + LB_W_TE * te_unb_rae
    print(
        f"\n[deploy] weights = "
        + ", ".join(f"{nm}={w:.4f}" for nm, w in zip(names, w_deploy))
    )
    print(f"[deploy] mu={mu_deploy:.4f}  s={s_deploy:.4f}")
    print(f"[deploy] in_sample_RAE={in_rae:.4f}  te[unb_idx]_RAE={te_unb_rae:.4f}  "
          f"LB_band={lb_band_est:.4f}")

    # Gate
    if pooled_mean < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif pooled_mean < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"\n[gate] pooled_mean={pooled_mean:.4f}  verdict={verdict}")

    # Save artefacts
    pred_oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    np.save(pred_oof_path, mean_oof.astype(np.float32))
    print(f"[save] {pred_oof_path}")

    te_npy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_npy_path, deploy_te)
    print(f"[save] {te_npy_path}")

    sub_csv_path = SUBMISSIONS / f"{TAG}_4anchor_pre_pyramid.csv"
    if verdict == "PROMOTE":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": deploy_te,
        }).to_csv(sub_csv_path, index=False)
        print(f"[save] {sub_csv_path}  (gate PROMOTE)")
        wrote_csv = True
    else:
        print(f"[skip] verdict={verdict} -- no submission CSV written")
        wrote_csv = False

    summary = {
        "tag": TAG,
        "method": "4anchor_PRE_clean_SLSQP_pyramid",
        "anchors": names,
        "indiv_oof_rae_unb": indiv_rae,
        "kf_seeds": KF_SEEDS,
        "n_seeds": len(KF_SEEDS),
        "n_folds": N_FOLDS,
        "stretch_grid": STRETCH_GRID,
        "n_unb": n_unb,
        "n_te": n_te,
        "n_unique_scaffolds": n_unique_scaf,
        "per_seed_results": per_seed,
        "pooled_rae_mean_seeds": pooled_mean,
        "pooled_rae_std_seeds": pooled_std,
        "pooled_rae_min_seeds": pooled_min,
        "pooled_rae_max_seeds": pooled_max,
        "rae_of_mean_of_seed_oofs": rae_of_mean_oof,
        "deploy_weights": [
            {"name": nm, "w": float(w)} for nm, w in zip(names, w_deploy)
        ],
        "deploy_mu_blend": mu_deploy,
        "deploy_s": s_deploy,
        "in_sample_rae_overfit_bound": in_rae,
        "te_unb_rae_in_sample": te_unb_rae,
        "lb_band_estimate": lb_band_est,
        "lb_band_w_oof": LB_W_OOF,
        "lb_band_w_te": LB_W_TE,
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "pred_oof_path": str(pred_oof_path),
        "te_npy_path": str(te_npy_path),
        "submission_csv": str(sub_csv_path) if wrote_csv else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   anchors           = {names}")
    print(f"   pooled RAE mean   = {pooled_mean:.4f} +/- {pooled_std:.4f}")
    print(f"   verdict           = {verdict}")
    print(f"   deploy weights    = "
          + ", ".join(f"{nm}={w:.3f}" for nm, w in zip(names, w_deploy)))
    print(f"   deploy s          = {s_deploy:.4f}")
    print(f"   LB band estimate  = {lb_band_est:.4f}")
    print(f"   wall              = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "pooled_rae_mean_seeds",
        "pooled_rae_std_seeds",
        "verdict",
        "deploy_weights",
        "lb_band_estimate",
    ):
        print(f"  {k}: {res.get(k)}")
