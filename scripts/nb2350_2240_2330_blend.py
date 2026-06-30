"""nb2350 -- SLSQP simplex blend of nb2240 + nb2330 (PRE-clean PRIMARY candidates).

CONTEXT:
    Two PRE-clean candidates currently sit close on the unblind 253:
      - nb2240  deep-30 OOF 0.4601  (5-anchor pyramid, K=20 RFE features)
      - nb2330  5-seed  OOF 0.4662  (5-anchor pyramid, K=24 + dist-from-train)
    Both use SLSQP + rank-stretch over 5 kf_seeds.

    Question: does a 2-anchor convex blend {nb2240, nb2330} beat nb2240 alone?
    Both come from related-but-not-identical pyramids -- nb2330 swaps in a
    K=24 anchor with explicit dist-from-train features that nb2240 lacks --
    so the residual correlation should be high but not 1.0.

PROTOCOL:
    1. Load nb2240 OOF (cached at nb2240_mean_bag_oof_K20.npy is the K=20
       residual anchor only, NOT the post-stretch blend OOF). Reconstruct
       nb2240's post-stretch pyramid OOF on the 253 by rerunning its
       5-anchor SLSQP+stretch CV with the same kf_seeds and stretch grid.
       Reconstruct nb2330 the same way.
    2. Compute Pearson(nb2240_oof, nb2330_oof) on 253.
    3. SLSQP simplex {nb2240, nb2330} per scaffold-fold across 5 kf_seeds
       {1001..1005}.  No additional rank-stretch (anchors already stretched).
    4. Pooled RAE = mean of 5 per-seed pooled RAEs.
    5. Compare vs nb2240 standalone 0.4601.  Gate margin 0.003.
    6. If beats: build deploy CSV using deploy-side blend on 513.

OUTPUTS:
    scripts/nb2350_2240_2330_blend.py
    data/processed/nb2350_summary.json
    data/processed/te_nb2350.npy
    submissions/nb2350_2240_2330_blend.csv   (only on gate pass)
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
from scipy.optimize import minimize

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2350"

GATE_MARGIN = 0.003
NB2240_REF_OOF = 0.4601
NB2330_REF_OOF = 0.4662

N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]
LB_W_OOF = 0.51
LB_W_TE = 0.49

# ---------------------------------------------------------------------------
# nb1191 reconstruction parameters (copied from nb2240_nb2171_k20.py)
# nb1191 OOF is not saved to disk -- it's a deploy-side blend computed
# from 4 sub-anchors.
# ---------------------------------------------------------------------------
NB1191_DEPLOY_WEIGHTS = {
    "chemprop_aux": 0.0,
    "nb1150":       0.641721304028517,
    "nb1158_K32":   0.23970131778546713,
    "nb2112_K28":   0.11857737818601592,
}
NB1191_DEPLOY_S = 1.031
NB1150_SLSQP4_OOFS = [
    "nb1133_chemprop_aux_pred_oof.npy",
    "nb503_pred_oof.npy",
    "nb1133_nb1014_pred_oof.npy",
    "nb2103_mean_bag_oof_K28.npy",
]
NB1150_SLSQP4_WEIGHTS = [0.0, 0.2942, 0.0, 0.7058]


def reconstruct_nb1150_oof(n_unb):
    cols = []
    for rel in NB1150_SLSQP4_OOFS:
        p = DATA_PROCESSED / rel
        assert p.exists(), f"missing nb1150 sub-anchor OOF: {p}"
        v = np.load(p).astype(np.float64)
        assert v.shape == (n_unb,), f"{p.name} shape {v.shape}"
        cols.append(v)
    return np.column_stack(cols) @ np.asarray(NB1150_SLSQP4_WEIGHTS, dtype=np.float64)


def reconstruct_nb1191_oof(n_unb):
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
    blend = (
        NB1191_DEPLOY_WEIGHTS["chemprop_aux"] * chemprop_oof
        + NB1191_DEPLOY_WEIGHTS["nb1150"]       * nb1150_oof
        + NB1191_DEPLOY_WEIGHTS["nb1158_K32"]   * nb1158_oof
        + NB1191_DEPLOY_WEIGHTS["nb2112_K28"]   * nb2112_oof
    )
    mu = float(blend.mean())
    return mu + NB1191_DEPLOY_S * (blend - mu)


# ---------------------------------------------------------------------------
# Anchor definitions for nb2240 and nb2330 pyramids (per their summaries)
# nb1191 is provided via reconstruct_nb1191_oof() (no on-disk OOF)
# ---------------------------------------------------------------------------
NB2240_ANCHORS = [
    ("nb2240_K20",   "nb2240_mean_bag_oof_K20.npy",       "te_nb2240_K20.npy"),
    ("chemprop_aux", "nb1133_chemprop_aux_pred_oof.npy",  "te_chemprop_aux.npy"),
    ("nb1191",       None,                                "te_nb1191.npy"),  # reconstructed
    ("nb503",        "nb503_pred_oof.npy",                "te_nb503.npy"),
    ("nb562",        "nb562_pred_oof.npy",                "te_nb562.npy"),
]
NB2330_ANCHORS = [
    ("nb2103_K28",   "nb2103_mean_bag_oof_K28.npy",       "te_chemprop_aux.npy"),
    ("chemprop_aux", "nb1133_chemprop_aux_pred_oof.npy",  "te_chemprop_aux.npy"),
    ("K24_anchor",   "nb2310_mean_bag_oof_K24.npy",       "te_nb2310_K24.npy"),
    ("nb503",        "nb503_pred_oof.npy",                "te_nb503.npy"),
    ("nb562",        "nb562_pred_oof.npy",                "te_nb562.npy"),
]


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


def load_anchor_stack(anchors, n_unb, n_te):
    """Returns (P_unb, P_te) shape (n_unb, K) / (n_te, K).

    If oof_rel is None, the disp is reconstructed (currently only nb1191).
    """
    oof_cols, te_cols = [], []
    for disp, oof_rel, te_rel in anchors:
        if oof_rel is None:
            if disp == "nb1191":
                oof = reconstruct_nb1191_oof(n_unb).astype(np.float64)
            else:
                raise ValueError(f"no oof_rel and no reconstructor for {disp}")
        else:
            oof_p = DATA_PROCESSED / oof_rel
            assert oof_p.exists(), f"missing OOF: {oof_p}"
            oof = np.load(oof_p).astype(np.float64)
        te_p = DATA_PROCESSED / te_rel
        assert te_p.exists(), f"missing te: {te_p}"
        te_arr = np.load(te_p).astype(np.float64)
        assert oof.shape == (n_unb,), f"{disp} oof {oof.shape}"
        assert te_arr.shape == (n_te,), f"{disp} te {te_arr.shape}"
        oof_cols.append(oof)
        te_cols.append(te_arr)
    return np.column_stack(oof_cols), np.column_stack(te_cols)


def reconstruct_pyramid_oof(P_unb, y_unb, unb_scaffolds, kf_seeds, stretch_grid):
    """Reconstruct the pyramid's per-seed OOFs and return mean-of-seeds OOF."""
    all_oofs = []
    per_seed_rae = []
    for kf_seed in kf_seeds:
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        n_unb = P_unb.shape[0]
        oof_blend = np.full(n_unb, np.nan)
        for tr_loc, va_loc in splits:
            w_f = slsqp_simplex(P_unb[tr_loc], y_unb[tr_loc])
            blend_tr = P_unb[tr_loc] @ w_f
            mu_tr = float(blend_tr.mean())
            s_f, _ = best_stretch_on(
                blend_tr, y_unb[tr_loc], mu_tr, stretch_grid
            )
            blend_va = P_unb[va_loc] @ w_f
            oof_blend[va_loc] = mu_tr + s_f * (blend_va - mu_tr)
        all_oofs.append(oof_blend)
        per_seed_rae.append(float(rae(y_unb, oof_blend)))
    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    return mean_oof, float(np.mean(per_seed_rae)), per_seed_rae


def deploy_pyramid_te(P_unb, y_unb, P_te, kf_seeds, stretch_grid):
    """Replicate the per-pyramid deploy: refit on all 253, mean(fold_s)."""
    fold_s_all = []
    for kf_seed in kf_seeds:
        splits = scaffold_kfold_indices(
            unb_scaffolds_global, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        for tr_loc, _va in splits:
            w_f = slsqp_simplex(P_unb[tr_loc], y_unb[tr_loc])
            blend_tr = P_unb[tr_loc] @ w_f
            mu_tr = float(blend_tr.mean())
            s_f, _ = best_stretch_on(
                blend_tr, y_unb[tr_loc], mu_tr, stretch_grid
            )
            fold_s_all.append(s_f)
    w_deploy = slsqp_simplex(P_unb, y_unb)
    blend_unb = P_unb @ w_deploy
    mu_deploy = float(blend_unb.mean())
    s_deploy = float(np.mean(fold_s_all))
    blend_te = P_te @ w_deploy
    deploy_te = (mu_deploy + s_deploy * (blend_te - mu_deploy)).astype(np.float64)
    deploy_unb = (mu_deploy + s_deploy * (blend_unb - mu_deploy)).astype(np.float64)
    return deploy_unb, deploy_te, w_deploy, mu_deploy, s_deploy


def cv_run_for_seed_2way(P_unb, y_unb, unb_scaffolds, kf_seed):
    """SLSQP simplex on 2-col P, NO additional stretch (anchors already stretched)."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = P_unb.shape[0]
    oof_blend = np.full(n_unb, np.nan)
    fold_w = []
    for tr_loc, va_loc in splits:
        w_f = slsqp_simplex(P_unb[tr_loc], y_unb[tr_loc])
        oof_blend[va_loc] = P_unb[va_loc] @ w_f
        fold_w.append(w_f)
    return float(rae(y_unb, oof_blend)), oof_blend, fold_w


# Module-level scaffolds for deploy_pyramid_te closure
unb_scaffolds_global = None


def main() -> dict:
    global unb_scaffolds_global
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- SLSQP simplex blend of nb2240 + nb2330")
    print("=" * 78)

    te = load_test()
    te_names = te["name"].values if "name" in te.columns else te["Molecule Name"].values
    te_smiles = te["smiles"].values if "smiles" in te.columns else te["SMILES"].values
    n_te = len(te_names)

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    unb_scaffolds_global = unb_scaffolds
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[load] n_te={n_te}  n_unb={n_unb}  n_unique_scaf={n_unique_scaf}")

    # ----- nb2240 reconstruct -----
    print("\n[nb2240] loading 5-anchor stack")
    P2240_unb, P2240_te = load_anchor_stack(NB2240_ANCHORS, n_unb, n_te)
    oof_2240, rae_2240_cv, per_seed_2240 = reconstruct_pyramid_oof(
        P2240_unb, y_unb, unb_scaffolds, KF_SEEDS, STRETCH_GRID,
    )
    deploy_unb_2240, deploy_te_2240, w_dep_2240, mu_2240, s_2240 = deploy_pyramid_te(
        P2240_unb, y_unb, P2240_te, KF_SEEDS, STRETCH_GRID,
    )
    print(
        f"   nb2240 reconstruct: OOF RAE (mean seeds) = {rae_2240_cv:.4f}  "
        f"(ref {NB2240_REF_OOF})"
    )

    # ----- nb2330 reconstruct -----
    print("\n[nb2330] loading 5-anchor stack")
    P2330_unb, P2330_te = load_anchor_stack(NB2330_ANCHORS, n_unb, n_te)
    oof_2330, rae_2330_cv, per_seed_2330 = reconstruct_pyramid_oof(
        P2330_unb, y_unb, unb_scaffolds, KF_SEEDS, STRETCH_GRID,
    )
    deploy_unb_2330, deploy_te_2330, w_dep_2330, mu_2330, s_2330 = deploy_pyramid_te(
        P2330_unb, y_unb, P2330_te, KF_SEEDS, STRETCH_GRID,
    )
    print(
        f"   nb2330 reconstruct: OOF RAE (mean seeds) = {rae_2330_cv:.4f}  "
        f"(ref {NB2330_REF_OOF})"
    )

    # ----- Pearson on 253 -----
    pearson_oof = float(np.corrcoef(oof_2240, oof_2330)[0, 1])
    pearson_te = float(np.corrcoef(deploy_te_2240, deploy_te_2330)[0, 1])
    print(f"\n[pearson] OOF(nb2240, nb2330) on 253 = {pearson_oof:.4f}")
    print(f"[pearson] te(nb2240, nb2330) on 513   = {pearson_te:.4f}")

    # ----- 2-anchor SLSQP blend, scaffold 5-fold CV across 5 kf_seeds -----
    P_unb = np.column_stack([oof_2240, oof_2330])
    P_te = np.column_stack([deploy_te_2240, deploy_te_2330])
    print("\n" + "-" * 78)
    print(f"2-anchor SLSQP simplex  kf_seeds={KF_SEEDS}  (no extra stretch)")
    print("-" * 78)
    per_seed = []
    all_oofs = []
    for kf_seed in KF_SEEDS:
        pooled, oof_blend, fold_w = cv_run_for_seed_2way(
            P_unb, y_unb, unb_scaffolds, kf_seed,
        )
        w_mean = np.mean(fold_w, axis=0).tolist()
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": float(pooled),
            "fold_w_mean": [float(x) for x in w_mean],
        })
        all_oofs.append(oof_blend)
        print(
            f"   seed={kf_seed}  pooled_RAE={pooled:.4f}  "
            f"w_mean={np.round(w_mean, 3).tolist()}"
        )
    pooled_rae_mean_seeds = float(np.mean([r["pooled_rae"] for r in per_seed]))
    pooled_rae_std_seeds = float(np.std([r["pooled_rae"] for r in per_seed]))
    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    rae_of_mean_oof = float(rae(y_unb, mean_oof))
    print(
        f"\n[cv] mean across 5 seeds  pooled_RAE = {pooled_rae_mean_seeds:.4f} "
        f"(+/- {pooled_rae_std_seeds:.4f})"
    )
    print(f"[cv] RAE of mean-of-seed OOFs        = {rae_of_mean_oof:.4f}")

    # ----- Deploy: refit weights on all 253 -----
    w_deploy = slsqp_simplex(P_unb, y_unb)
    blend_te = P_te @ w_deploy
    blend_unb = P_unb @ w_deploy
    in_rae = float(rae(y_unb, blend_unb))
    deploy_te = blend_te.astype(np.float32)
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))
    print("\n[deploy]")
    print(
        f"   w_deploy = nb2240={w_deploy[0]:.4f}  nb2330={w_deploy[1]:.4f}"
    )
    print(f"   in-sample RAE (253)  = {in_rae:.4f}")
    print(f"   te[unb_idx] RAE      = {te_unb_rae:.4f}")
    print(f"   te(513) mean/std     = {deploy_te.mean():.3f}/{deploy_te.std():.3f}")

    lb_band_est = LB_W_OOF * pooled_rae_mean_seeds + LB_W_TE * te_unb_rae
    print(
        f"\n[LB-band] {LB_W_OOF:.2f}*OOF({pooled_rae_mean_seeds:.4f}) + "
        f"{LB_W_TE:.2f}*te_unb({te_unb_rae:.4f}) = {lb_band_est:.4f}"
    )

    # ----- Gate vs nb2240 -----
    delta_vs_nb2240 = pooled_rae_mean_seeds - NB2240_REF_OOF
    delta_vs_nb2330 = pooled_rae_mean_seeds - NB2330_REF_OOF
    gate_beat_nb2240 = delta_vs_nb2240 < -GATE_MARGIN
    gate_flat_nb2240 = abs(delta_vs_nb2240) <= GATE_MARGIN
    if gate_beat_nb2240:
        verdict = "BEATS_NB2240"
    elif gate_flat_nb2240:
        verdict = "FLAT_VS_NB2240"
    else:
        verdict = "HURTS_NB2240"
    print("\n" + "-" * 78)
    print(f"GATE EVALUATION  (margin {GATE_MARGIN})")
    print("-" * 78)
    print(f"   nb2350 OOF (5-seed mean) = {pooled_rae_mean_seeds:.4f}")
    print(f"   vs nb2240 ({NB2240_REF_OOF}) delta = {delta_vs_nb2240:+.4f}")
    print(f"   vs nb2330 ({NB2330_REF_OOF}) delta = {delta_vs_nb2330:+.4f}")
    print(f"   verdict (vs nb2240)      = {verdict}")

    # ----- Save artefacts -----
    te_npy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_npy_path, deploy_te)
    print(f"\n[save] {te_npy_path}")

    sub_csv_path = SUBMISSIONS / f"{TAG}_2240_2330_blend.csv"
    if gate_beat_nb2240:
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": deploy_te,
        }).to_csv(sub_csv_path, index=False)
        print(f"[save] {sub_csv_path}  (gate BEATS_NB2240)")
    else:
        print(f"[skip] gate not beat -- no submission CSV written ({verdict})")

    summary = {
        "tag": TAG,
        "method": "slsqp_simplex_blend_nb2240_plus_nb2330",
        "anchors": ["nb2240_pyramid", "nb2330_pyramid"],
        "indiv_oof_rae_unb": {
            "nb2240_reconstruct": rae_2240_cv,
            "nb2330_reconstruct": rae_2330_cv,
            "nb2240_ref": NB2240_REF_OOF,
            "nb2330_ref": NB2330_REF_OOF,
        },
        "pearson_oof_2240_2330_unb": pearson_oof,
        "pearson_te_2240_2330_513": pearson_te,
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "n_unb": n_unb,
        "n_te": n_te,
        "n_unique_scaffolds": n_unique_scaf,
        "per_seed_results": per_seed,
        "pooled_rae_mean_seeds": pooled_rae_mean_seeds,
        "pooled_rae_std_seeds": pooled_rae_std_seeds,
        "rae_of_mean_of_seed_oofs": rae_of_mean_oof,
        "deploy_weights": {
            "nb2240": float(w_deploy[0]),
            "nb2330": float(w_deploy[1]),
        },
        "in_sample_rae_overfit_bound": in_rae,
        "te_unb_rae_in_sample": te_unb_rae,
        "lb_band_estimate": lb_band_est,
        "lb_band_low": lb_band_est - 0.05,
        "lb_band_high": lb_band_est + 0.05,
        "lb_band_w_oof": LB_W_OOF,
        "lb_band_w_te": LB_W_TE,
        "compare_nb2240_oof": NB2240_REF_OOF,
        "compare_nb2330_oof": NB2330_REF_OOF,
        "delta_vs_nb2240": delta_vs_nb2240,
        "delta_vs_nb2330": delta_vs_nb2330,
        "gate_margin": GATE_MARGIN,
        "gate_beat_nb2240": bool(gate_beat_nb2240),
        "gate_flat_vs_nb2240": bool(gate_flat_nb2240),
        "verdict_vs_nb2240": verdict,
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
    print(f"   blend pooled RAE (5 seeds) = {pooled_rae_mean_seeds:.4f}")
    print(f"   pearson(nb2240, nb2330)    = {pearson_oof:.4f}")
    print(f"   deploy w_nb2240/w_nb2330   = {w_deploy[0]:.4f} / {w_deploy[1]:.4f}")
    print(f"   delta vs nb2240 ({NB2240_REF_OOF})   = {delta_vs_nb2240:+.4f}")
    print(f"   verdict                    = {verdict}")
    print(f"   wall                       = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
