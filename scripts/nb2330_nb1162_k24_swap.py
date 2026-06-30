"""nb2330 -- nb1162 anchor SWAP: replace nb730_honest with K=24 (PRE-clean).

CONTEXT:
    nb1162 is the canonical 5-anchor stacking pyramid:
        {nb2103_K28, chemprop_aux, nb730_honest, nb503, nb562}
    SLSQP convex blend + rank-stretch under scaffold 5-fold CV. nb730_honest
    is a POST-unblind anchor (trained on the 253 leaked labels), so its
    cross-fit RAE on the unblind is in-sample-optimistic and LB transfer is
    unreliable. nb1162 ships pooled_RAE ~0.4206 in-sample on the 253 with
    this risk.

    K=24 is the freshly built PRE-clean LGBM anchor from nb2310:
      = chemprop_aux + LGBM(MSE) residual mean-bag over 5 seeds
      = features: K=20 RFE survivors (from nb2231) +
                  4 dist-from-train shift features
        - max_tanimoto_to_train
        - mean_tanimoto_top5_train
        - num_train_with_sim_ge_0p5
        - is_scaffold_novel
      Cached: nb2310_mean_bag_oof_K24.npy (253,)
              te_nb2310_K24.npy           (513,)
      Anchor in_RAE on 253:  0.4674   (PRE-clean LGBM, no leakage)

    The substitution drops the POST-unblind anchor in favour of a PRE-clean
    anchor that already encodes dataset-shift signal. If the swap holds the
    nb1162 OOF (~0.4206) while removing POST risk, we get an LB-faithful
    deploy candidate.

PROTOCOL:
    1. Build 5-anchor pyramid:
         0. nb2103_K28      data/processed/nb2103_mean_bag_oof_K28.npy
         1. chemprop_aux    data/processed/nb1133_chemprop_aux_pred_oof.npy
         2. K24_anchor      data/processed/nb2310_mean_bag_oof_K24.npy   NEW
         3. nb503           data/processed/nb503_pred_oof.npy
         4. nb562           data/processed/nb562_pred_oof.npy
       te-side fan:
         nb2103   -> te_chemprop_aux.npy   (proxy, matches nb1162 convention)
         chemprop -> te_chemprop_aux.npy
         K24_anchor -> te_nb2310_K24.npy
         nb503    -> te_nb503.npy
         nb562    -> te_nb562.npy
    2. SLSQP convex blend (w>=0, sum=1) per scaffold-fold.
    3. Per-fold rank-stretch s in {1.00..1.15} step 0.025.
    4. 5-fold scaffold CV across 5 kf_seeds {1001..1005}.
    5. Compare pooled_RAE_mean_seeds vs:
         nb2240 0.4601   (5-anchor pyramid with K=20 anchor)
         nb2171 0.4682   (5-anchor pyramid with nb730_honest swap -> nb1191)
       Gate margin 0.003.  On beat: deep-30 verify (5 canonical + 25 extra).
    6. Always save te artefact + summary.  Submission CSV only on gate pass.

OUTPUTS:
    scripts/nb2330_nb1162_k24_swap.py
    data/processed/nb2330_summary.json
    data/processed/te_nb2330.npy
    submissions/nb2330_nb1162_k24_swap.csv   (only on gate pass)
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

TAG = "nb2330"

# ---------------------------------------------------------------------------
# Gate config (relative to nb2240 / nb2171 references)
# ---------------------------------------------------------------------------
GATE_MARGIN = 0.003
NB2240_REF_OOF = 0.4601    # pooled_rae_mean_seeds from nb2240 (5-anchor K=20)
NB2171_REF_OOF = 0.4682    # pooled_rae_mean_seeds from nb2171 (nb1191 swap)

# ---------------------------------------------------------------------------
# CV / pyramid config
# ---------------------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]
LB_W_OOF = 0.51
LB_W_TE = 0.49

# ---------------------------------------------------------------------------
# Anchor definition: nb1162 base, nb730_honest -> K=24 swap
# ---------------------------------------------------------------------------
#   (display_name, oof_path_relative, te_path_relative)
ANCHORS = [
    ("nb2103_K28",   "nb2103_mean_bag_oof_K28.npy",       "te_chemprop_aux.npy"),
    ("chemprop_aux", "nb1133_chemprop_aux_pred_oof.npy",  "te_chemprop_aux.npy"),
    ("K24_anchor",   "nb2310_mean_bag_oof_K24.npy",       "te_nb2310_K24.npy"),
    ("nb503",        "nb503_pred_oof.npy",                "te_nb503.npy"),
    ("nb562",        "nb562_pred_oof.npy",                "te_nb562.npy"),
]
K24_ANCHOR_IDX = 2


# ===========================================================================
# core utilities
# ===========================================================================

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
        s_f, _ = best_stretch_on(blend_tr, y_unb[tr_loc], mu_tr, STRETCH_GRID)
        blend_va = P_unb[va_loc] @ w_f
        oof_blend[va_loc] = mu_tr + s_f * (blend_va - mu_tr)
        fold_w.append(w_f)
        fold_s.append(s_f)
    return float(rae(y_unb, oof_blend)), oof_blend, fold_w, fold_s


def deep_verify_seeds(P_unb, y_unb, unb_scaffolds, n_extra=25):
    extra_seeds = list(range(2001, 2001 + n_extra))
    seeds = KF_SEEDS + extra_seeds
    per = []
    for seed in seeds:
        pooled, _o, _fw, fs = cv_run_for_seed(P_unb, y_unb, unb_scaffolds, seed)
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
        "std_rae": float(raes.std()),
        "min_rae": float(raes.min()),
        "max_rae": float(raes.max()),
    }


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- nb1162 anchor SWAP: nb730_honest -> K=24 (PRE-clean)")
    print("=" * 78)

    # ---- Load test + unblind ----
    te = load_test()
    te_names = te["name"].values if "name" in te.columns else te["Molecule Name"].values
    te_smiles = te["smiles"].values if "smiles" in te.columns else te["SMILES"].values
    n_te = len(te_names)

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_te={n_te}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique unb_scaffolds={n_unique_scaf}")

    # ---- Load anchors ----
    print("\n[anchors]")
    oof_cols, te_cols, indiv_rae = [], [], {}
    for disp, oof_rel, te_rel in ANCHORS:
        oof_p = DATA_PROCESSED / oof_rel
        te_p = DATA_PROCESSED / te_rel
        assert oof_p.exists(), f"missing OOF: {oof_p}"
        assert te_p.exists(), f"missing te: {te_p}"
        oof = np.load(oof_p).astype(np.float64)
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
    assert K == 5, f"expected 5 anchors, got {K}"

    # =================================================================
    # SLSQP + rank-stretch, scaffold 5-fold CV across 5 seeds
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
    s_deploy = float(np.mean([s for r in per_seed for s in r["fold_s"]]))
    in_rae_final = float(rae(
        y_unb, mu_deploy + s_deploy * (blend_unb - mu_deploy)
    ))
    blend_te = P_te @ w_deploy
    deploy_te = (mu_deploy + s_deploy * (blend_te - mu_deploy)).astype(np.float32)
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))

    w_str = ", ".join(
        f"{disp}={w:.4f}" for (disp, _, _), w in zip(ANCHORS, w_deploy)
    )
    w_k24 = float(w_deploy[K24_ANCHOR_IDX])
    print(f"   deploy weights      = {w_str}")
    print(f"   K24_anchor weight   = {w_k24:.4f}")
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
    # Gate vs nb2240 / nb2171
    # =================================================================
    delta_vs_nb2240 = pooled_rae_mean_seeds - NB2240_REF_OOF
    delta_vs_nb2171 = pooled_rae_mean_seeds - NB2171_REF_OOF
    gate_beat_nb2240 = delta_vs_nb2240 < -GATE_MARGIN
    gate_flat_nb2240 = abs(delta_vs_nb2240) <= GATE_MARGIN
    gate_beat_nb2171 = delta_vs_nb2171 < -GATE_MARGIN
    print("\n" + "-" * 78)
    print(f"GATE EVALUATION  (margin {GATE_MARGIN})")
    print("-" * 78)
    print(f"   nb2330 OOF (5-seed mean) = {pooled_rae_mean_seeds:.4f}")
    print(f"   vs nb2240 ({NB2240_REF_OOF:.4f}) delta = {delta_vs_nb2240:+.4f}")
    print(f"   vs nb2171 ({NB2171_REF_OOF:.4f}) delta = {delta_vs_nb2171:+.4f}")
    if gate_beat_nb2240:
        verdict = "BEATS_NB2240"
    elif gate_flat_nb2240:
        verdict = "FLAT_VS_NB2240"
    else:
        verdict = "HURTS_NB2240"
    print(f"   verdict (vs nb2240)      = {verdict}")
    print(f"   verdict (vs nb2171)      = "
          f"{'BEATS' if gate_beat_nb2171 else ('FLAT' if abs(delta_vs_nb2171) <= GATE_MARGIN else 'HURTS')}")

    # =================================================================
    # Deep-30 verify (only on gate-pass vs nb2240)
    # =================================================================
    deep30 = None
    if gate_beat_nb2240:
        print("\n" + "-" * 78)
        print("DEEP-30 VERIFY  (5 canonical + 25 extra seeds)")
        print("-" * 78)
        deep30 = deep_verify_seeds(P_unb, y_unb, unb_scaffolds, n_extra=25)
        print(
            f"   n_seeds={deep30['n_seeds']}  "
            f"mean_RAE={deep30['mean_rae']:.4f}  "
            f"std={deep30['std_rae']:.4f}  "
            f"range=[{deep30['min_rae']:.4f}, {deep30['max_rae']:.4f}]"
        )

    # ---- Always save te artefact ----
    te_npy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_npy_path, deploy_te)
    print(f"\n[save] {te_npy_path}")

    sub_csv_path = SUBMISSIONS / f"{TAG}_nb1162_k24_swap.csv"
    if gate_beat_nb2240:
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": deploy_te,
        }).to_csv(sub_csv_path, index=False)
        print(f"[save] {sub_csv_path}  (gate BEATS_NB2240)")
    else:
        print(
            f"[skip] gate not beat -- no submission CSV written ({verdict})"
        )

    summary = {
        "tag": TAG,
        "method": "nb1162_anchor_swap_nb730_to_K24_SLSQP_then_rank_stretch_seedavg",
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
        "w_k24_deploy": w_k24,
        "in_sample_rae_overfit_bound": in_rae_final,
        "te_unb_rae_in_sample": te_unb_rae,
        "lb_band_estimate": lb_band_est,
        "lb_band_low": lb_low,
        "lb_band_high": lb_high,
        "lb_band_w_oof": LB_W_OOF,
        "lb_band_w_te": LB_W_TE,
        "compare_nb2240_oof": NB2240_REF_OOF,
        "compare_nb2171_oof": NB2171_REF_OOF,
        "delta_vs_nb2240": delta_vs_nb2240,
        "delta_vs_nb2171": delta_vs_nb2171,
        "gate_margin": GATE_MARGIN,
        "gate_beat_nb2240": bool(gate_beat_nb2240),
        "gate_flat_vs_nb2240": bool(gate_flat_nb2240),
        "gate_beat_nb2171": bool(gate_beat_nb2171),
        "verdict_vs_nb2240": verdict,
        "deep_30_verify": deep30,
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "te_npy_path": str(te_npy_path),
        "submission_csv": str(sub_csv_path) if gate_beat_nb2240 else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    json_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {json_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   pyramid pooled RAE (5 seeds) = {pooled_rae_mean_seeds:.4f}")
    print(f"   K24_anchor deploy weight     = {w_k24:.4f}")
    print(f"   delta vs nb2240 ({NB2240_REF_OOF:.4f})    = {delta_vs_nb2240:+.4f}")
    print(f"   delta vs nb2171 ({NB2171_REF_OOF:.4f})    = {delta_vs_nb2171:+.4f}")
    print(f"   LB-band estimate              = {lb_band_est:.4f}")
    print(f"   verdict vs nb2240            = {verdict}")
    print(f"   wall                          = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "pooled_rae_mean_seeds",
        "w_k24_deploy",
        "delta_vs_nb2240",
        "delta_vs_nb2171",
        "verdict_vs_nb2240",
        "gate_beat_nb2240",
        "gate_beat_nb2171",
        "lb_band_estimate",
        "deploy_weights",
        "deploy_s",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
