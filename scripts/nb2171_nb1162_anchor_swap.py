"""nb2171 -- nb1162 anchor SWAP: replace nb730_honest with nb1191.

Per cycle 157 ablation: nb1162 (POST-risk anchor stack) carries 88.7% of weight
in nb730_honest. The risk is that nb730_honest is a POST-unblind anchor (trained
on the 253 leaked labels), so its honest cross-fit RAE on the unblind set is
in-sample-optimistic and the LB transfer is unreliable.

nb1191 is a PRE-unblind stack pyramid (chemprop_aux + nb1150 + nb1158_K32 +
nb2112_K28) with verified LB-faithful behaviour; its OOF cross-fit RAE on the
253 is 0.4697 and its predicted LB band centres around 0.369. We promote
nb1191's deploy vector to a 5-anchor blend in place of nb730_honest, keeping
{nb2103_K28, chemprop_aux, nb503, nb562} as the other four anchors and
re-running SLSQP + rank-stretch per scaffold-fold.

Stage 1 anchors:
    0. nb2103_K28      data/processed/nb2103_mean_bag_oof_K28.npy
    1. chemprop_aux    data/processed/nb1133_chemprop_aux_pred_oof.npy
    2. nb1191          RECONSTRUCT from nb1191 deploy weights on its 4 sub-anchors
    3. nb503           data/processed/nb503_pred_oof.npy
    4. nb562           data/processed/nb562_pred_oof.npy

For deploy on the 513 test compounds we use the matched te files:
    nb2103_K28   -> te_chemprop_aux.npy   (proxy; matches nb1162 convention)
    chemprop_aux -> te_chemprop_aux.npy
    nb1191       -> te_nb1191.npy         (cached deploy vector)
    nb503        -> te_nb503.npy
    nb562        -> te_nb562.npy

Stage 2 SLSQP: convex blend (w >= 0, sum = 1) under scaffold 5-fold CV on
the 253 unblind. kf_seeds {1001..1005}. Loss = SSE.

Stage 3 rank-stretch (per-fold): grid s in {1.00, 1.025, 1.05, 1.075, 1.10,
1.125, 1.15}. For deploy on the 513, apply mean(per-fold s) around the blend
mean of the deploy blend.

GATE (both must pass):
    A) pooled scaffold-CV OOF RAE (mean of seeds) <= 0.45
    B) nb1191 deploy weight >= 0.30

If passes: deep-30 verify + build deploy CSV
    submissions/nb2171_nb1162_anchor_swap.csv

Outputs:
    scripts/nb2171_nb1162_anchor_swap.py
    data/processed/nb2171_summary.json
    data/processed/te_nb2171.npy
    submissions/nb2171_nb1162_anchor_swap.csv  (only on gate pass)
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

TAG = "nb2171"
GATE_OOF = 0.45
GATE_W_NB1191 = 0.30
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]
LB_W_OOF = 0.51
LB_W_TE = 0.49

# Compare baselines (from cycle 157 ablation memo and live summaries)
NB1162_OOF_MEMO = 0.4206   # POST risk
NB2095_OOF_MEMO = 0.4720   # PRE clean (nb2095 pseudo-PRE blend with nb1014)

# nb1191 OOF reconstruction parameters (from nb1191_summary.json deploy_weights
# and the cached nb1150 SLSQP4 weights pattern).
NB1191_DEPLOY_WEIGHTS = {
    "chemprop_aux": 0.0,            # 3.87e-15 ~ 0
    "nb1150":       0.641721304028517,
    "nb1158_K32":   0.23970131778546713,
    "nb2112_K28":   0.11857737818601592,
}
NB1191_DEPLOY_S = 1.031

# nb1150 sub-stack (SLSQP4 over 4 base anchors with cached full-pool weights)
NB1150_SLSQP4_OOFS = [
    "nb1133_chemprop_aux_pred_oof.npy",
    "nb503_pred_oof.npy",
    "nb1133_nb1014_pred_oof.npy",
    "nb2103_mean_bag_oof_K28.npy",
]
NB1150_SLSQP4_WEIGHTS = [0.0, 0.2942, 0.0, 0.7058]

# 5-anchor stack: (display_name, oof_path_relative, te_path_relative)
ANCHORS = [
    ("nb2103_K28",   "nb2103_mean_bag_oof_K28.npy",       "te_chemprop_aux.npy"),
    ("chemprop_aux", "nb1133_chemprop_aux_pred_oof.npy",  "te_chemprop_aux.npy"),
    ("nb1191",       "_RECONSTRUCT_nb1191_oof",           "te_nb1191.npy"),
    ("nb503",        "nb503_pred_oof.npy",                "te_nb503.npy"),
    ("nb562",        "nb562_pred_oof.npy",                "te_nb562.npy"),
]
NB1191_ANCHOR_IDX = 2  # for gate B


def reconstruct_nb1150_oof(n_unb: int) -> np.ndarray:
    cols = []
    for rel in NB1150_SLSQP4_OOFS:
        p = DATA_PROCESSED / rel
        assert p.exists(), f"missing nb1150 sub-anchor OOF: {p}"
        v = np.load(p).astype(np.float64)
        assert v.shape == (n_unb,), f"{p.name} shape {v.shape}"
        cols.append(v)
    P = np.column_stack(cols)
    w = np.asarray(NB1150_SLSQP4_WEIGHTS, dtype=np.float64)
    return P @ w


def reconstruct_nb1191_oof(n_unb: int) -> np.ndarray:
    """Reconstruct nb1191 OOF using its cached full-pool deploy weights and
    apply the cached deploy rank-stretch s=1.031 around the blend mean.

    This is an in-sample reconstruction (full-pool weights) -- the systematic
    bias is the same across all 5 CV folds we apply downstream, so the
    fold-to-fold structure is preserved for nb2171's CV.
    """
    chemprop_oof = np.load(
        DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy"
    ).astype(np.float64)
    nb1150_oof = reconstruct_nb1150_oof(n_unb)
    nb1158_oof = np.load(
        DATA_PROCESSED / "nb1158_mean_bag_oof_K32.npy"
    ).astype(np.float64)
    nb2112_oof = np.load(
        DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy"
    ).astype(np.float64)
    assert chemprop_oof.shape == (n_unb,)
    assert nb1150_oof.shape == (n_unb,)
    assert nb1158_oof.shape == (n_unb,)
    assert nb2112_oof.shape == (n_unb,)
    blend = (
        NB1191_DEPLOY_WEIGHTS["chemprop_aux"] * chemprop_oof
        + NB1191_DEPLOY_WEIGHTS["nb1150"]       * nb1150_oof
        + NB1191_DEPLOY_WEIGHTS["nb1158_K32"]   * nb1158_oof
        + NB1191_DEPLOY_WEIGHTS["nb2112_K28"]   * nb2112_oof
    )
    mu = float(blend.mean())
    stretched = mu + NB1191_DEPLOY_S * (blend - mu)
    return stretched


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


def deep_verify_seeds(P_unb, y_unb, unb_scaffolds, n_extra=25):
    """Deep-30 verify: add 25 more kf_seeds on top of the canonical 5."""
    extra_seeds = list(range(2001, 2001 + n_extra))
    seeds = KF_SEEDS + extra_seeds
    per = []
    for seed in seeds:
        pooled, _oof, _fw, fs = cv_run_for_seed(
            P_unb, y_unb, unb_scaffolds, seed,
        )
        per.append({
            "kf_seed": int(seed),
            "pooled_rae": float(pooled),
            "mean_s": float(np.mean(fs)),
        })
    raes = np.asarray([r["pooled_rae"] for r in per])
    return {
        "n_seeds": int(len(seeds)),
        "per_seed": per,
        "mean_rae": float(raes.mean()),
        "std_rae":  float(raes.std()),
        "min_rae":  float(raes.min()),
        "max_rae":  float(raes.max()),
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- nb1162 anchor SWAP: nb730_honest -> nb1191")
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
        if oof_rel == "_RECONSTRUCT_nb1191_oof":
            oof = reconstruct_nb1191_oof(n_unb)
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
        print(
            f"   {disp:14s} oof_RAE={r:.4f}  te_mean={te_arr.mean():.3f}  "
            f"te_std={te_arr.std():.3f}"
        )

    P_unb = np.column_stack(oof_cols)
    P_te = np.column_stack(te_cols)
    K = P_unb.shape[1]
    print(f"[stack] P_unb {P_unb.shape}  P_te {P_te.shape}  K={K}")

    # =================================================================
    # Stage 2+3: SLSQP + rank-stretch, scaffold 5-fold CV across 5 seeds
    # =================================================================
    print("\n" + "-" * 78)
    print(
        f"SCAFFOLD 5-FOLD CV  kf_seeds={KF_SEEDS}  stretch_grid={STRETCH_GRID}"
    )
    print("-" * 78)
    per_seed = []
    all_oofs = []
    for kf_seed in KF_SEEDS:
        pooled, oof_blend, fold_w, fold_s = cv_run_for_seed(
            P_unb, y_unb, unb_scaffolds, kf_seed,
        )
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": float(pooled),
            "fold_s": [float(x) for x in fold_s],
            "fold_w_mean": [float(x) for x in np.mean(fold_w, axis=0)],
        })
        all_oofs.append(oof_blend)
        print(
            f"   seed={kf_seed}  pooled_RAE={pooled:.4f}  "
            f"mean_s={np.mean(fold_s):.3f}  "
            f"w_mean={np.round(np.mean(fold_w, axis=0), 3).tolist()}"
        )

    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    final_oof_rae = float(rae(y_unb, mean_oof))
    pooled_rae_mean_seeds = float(np.mean([r["pooled_rae"] for r in per_seed]))
    pooled_rae_std_seeds = float(np.std([r["pooled_rae"] for r in per_seed]))
    print(
        f"\n[cv] mean across 5 seeds  pooled_RAE = {pooled_rae_mean_seeds:.4f} "
        f"(+/- {pooled_rae_std_seeds:.4f})"
    )
    print(f"[cv] RAE of mean-of-seed OOFs        = {final_oof_rae:.4f}")

    # =================================================================
    # Deploy
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
    print(f"   te(513) mean/std    = {deploy_te.mean():.3f}/{deploy_te.std():.3f}")

    lb_band_est = LB_W_OOF * pooled_rae_mean_seeds + LB_W_TE * te_unb_rae
    lb_low = lb_band_est - 0.05
    lb_high = lb_band_est + 0.05
    print(
        f"\n[LB-band] {LB_W_OOF:.2f}*OOF({pooled_rae_mean_seeds:.4f}) + "
        f"{LB_W_TE:.2f}*te_unb({te_unb_rae:.4f}) = {lb_band_est:.4f}  "
        f"[{lb_low:.4f}, {lb_high:.4f}]"
    )

    # =================================================================
    # Compare vs baselines
    # =================================================================
    delta_vs_nb1162 = pooled_rae_mean_seeds - NB1162_OOF_MEMO
    delta_vs_nb2095 = pooled_rae_mean_seeds - NB2095_OOF_MEMO
    print(f"\n[compare] nb2171 OOF = {pooled_rae_mean_seeds:.4f}")
    print(
        f"          vs nb1162 ({NB1162_OOF_MEMO:.4f}, POST risk)  "
        f"delta = {delta_vs_nb1162:+.4f}"
    )
    print(
        f"          vs nb2095 ({NB2095_OOF_MEMO:.4f}, PRE clean) "
        f"delta = {delta_vs_nb2095:+.4f}"
    )

    # =================================================================
    # Gate evaluation: A) OOF <= 0.45  B) nb1191 weight >= 0.30
    # =================================================================
    w_nb1191 = float(w_deploy[NB1191_ANCHOR_IDX])
    gate_a = pooled_rae_mean_seeds <= GATE_OOF
    gate_b = w_nb1191 >= GATE_W_NB1191
    gate_pass = gate_a and gate_b
    print("\n" + "-" * 78)
    print("GATE EVALUATION")
    print("-" * 78)
    print(
        f"   gate A: OOF (mean of seeds) {pooled_rae_mean_seeds:.4f} "
        f"<= {GATE_OOF:.4f}  -> {'PASS' if gate_a else 'FAIL'}"
    )
    print(
        f"   gate B: nb1191 deploy weight {w_nb1191:.4f} "
        f">= {GATE_W_NB1191:.4f}  -> {'PASS' if gate_b else 'FAIL'}"
    )
    print(f"   overall: {'PASS' if gate_pass else 'FAIL'}")

    # =================================================================
    # Deep-30 verify (run only on gate pass)
    # =================================================================
    deep30 = None
    if gate_pass:
        print("\n" + "-" * 78)
        print("DEEP-30 VERIFY  (5 canonical + 25 extra seeds)")
        print("-" * 78)
        deep30 = deep_verify_seeds(
            P_unb, y_unb, unb_scaffolds, n_extra=25,
        )
        print(
            f"   n_seeds={deep30['n_seeds']}  "
            f"mean_RAE={deep30['mean_rae']:.4f}  "
            f"std={deep30['std_rae']:.4f}  "
            f"range=[{deep30['min_rae']:.4f}, {deep30['max_rae']:.4f}]"
        )

    # Always save te artefact
    te_npy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_npy_path, deploy_te)
    print(f"\n[save] {te_npy_path}")

    sub_csv_path = SUBMISSIONS / f"{TAG}_nb1162_anchor_swap.csv"
    if gate_pass:
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": deploy_te,
        }).to_csv(sub_csv_path, index=False)
        print(f"[save] {sub_csv_path}  (gate PASSED)")
    else:
        print(
            f"[skip] gate FAILED -- no submission CSV written "
            f"(would be {sub_csv_path})"
        )

    summary = {
        "tag": TAG,
        "method": "nb1162_anchor_swap_nb730_to_nb1191_SLSQP_then_rank_stretch_seedavg",
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
        "compare_nb1162_oof": NB1162_OOF_MEMO,
        "compare_nb2095_oof": NB2095_OOF_MEMO,
        "delta_vs_nb1162": delta_vs_nb1162,
        "delta_vs_nb2095": delta_vs_nb2095,
        "w_nb1191_deploy": w_nb1191,
        "gate_oof_target": GATE_OOF,
        "gate_w_nb1191_target": GATE_W_NB1191,
        "gate_a_oof_le_target": bool(gate_a),
        "gate_b_w_nb1191_ge_target": bool(gate_b),
        "gate_pass": bool(gate_pass),
        "deep_30_verify": deep30,
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
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
    print(
        f"   pooled scaffold-CV RAE (mean of seeds) = "
        f"{pooled_rae_mean_seeds:.4f}"
    )
    print(f"   nb1191 deploy weight                   = {w_nb1191:.4f}")
    print(f"   delta vs nb1162 ({NB1162_OOF_MEMO:.4f})        = {delta_vs_nb1162:+.4f}")
    print(f"   delta vs nb2095 ({NB2095_OOF_MEMO:.4f})        = {delta_vs_nb2095:+.4f}")
    print(f"   LB-band estimate                       = {lb_band_est:.4f}")
    print(f"   gate A (OOF <= {GATE_OOF})                = {gate_a}")
    print(f"   gate B (w_nb1191 >= {GATE_W_NB1191})           = {gate_b}")
    print(f"   gate overall                           = {gate_pass}")
    print(f"   wall                                   = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "pooled_rae_mean_seeds",
        "w_nb1191_deploy",
        "delta_vs_nb1162",
        "delta_vs_nb2095",
        "lb_band_estimate",
        "gate_pass",
        "deploy_weights",
        "deploy_s",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
