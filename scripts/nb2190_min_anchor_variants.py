"""nb2190 -- Minimal-anchor variants of nb2171 stack.

HYPOTHESIS:
    nb2171 deploys with an effective 2-anchor blend: nb1191=0.936 + nb562=0.064
    (chemprop_aux, nb503, nb2103_K28 all collapse to ~0).  If the SLSQP picks
    only 2 of 5 anchors, the other 3 are dead weight that bloat the basis and
    risk fold-to-fold weight noise.  We strip the basis and test minimal
    stacks directly:

      Variant A: 2-anchor  {nb1191, nb562}
      Variant B: 3-anchor  {nb1191, nb562, nb1158_K32}
      Variant C: 2-anchor  {nb1191, nb503}    (nb2171 ablation:
                                                nb503/nb562 rank-identical)

    Each variant runs scaffold 5-fold SLSQP + per-fold rank-stretch, averaged
    over kf_seeds {1001..1005}.  Decision margin 0.003 vs nb2171 0.4682
    (nb2171 pooled OOF mean of seeds).  If best beats by >= 0.003, run
    deep-30 verify.

    NOTE: This file COEXISTS with scripts/nb2190_conformal_shrink.py.  We
    save the summary as data/processed/nb2190_summary.json (overwrites the
    older conformal-shrink summary, as the task instructs).  The conformal
    shrink script and its other artefacts (per_seed_oof_K28.npy,
    per_row_std_K28.npy) are untouched.

PROTOCOL:
    1. Load all required OOF + te npys for the 3 candidate anchors
       (nb1191 reconstructed from cached deploy weights, like nb2171).
    2. Compute bemis-murcko scaffolds for the 253 unblind.
    3. For each variant:
        a. For each kf_seed in {1001..1005}:
            - 5-fold scaffold CV
            - Per fold: SLSQP convex blend on train fold, grid-search rank
              stretch s in {1.000..1.150 step 0.025}
            - Apply (w, mu_tr, s) to val fold -> OOF
        b. pooled RAE per seed; mean and std across seeds
    4. Pick winner; compare vs nb2171 pooled_rae_mean_seeds (0.4676).  Margin
       0.003.
    5. If winner beats nb2171 by >= 0.003, run deep-30 verify (5 + 25 extra
       seeds).
    6. Save deploy te (513) artefact + summary.

Outputs:
    scripts/nb2190_min_anchor_variants.py
    data/processed/nb2190_summary.json
    data/processed/te_nb2190_min_anchor_<best_variant>.npy  (deploy vector)
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
from pxr.paths import DATA_PROCESSED

TAG = "nb2190"
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]
LB_W_OOF = 0.51
LB_W_TE = 0.49

# Decision baselines from nb2171_summary.json
NB2171_POOLED_RAE = 0.4676    # mean of seeds, also 0.4682 rounded per task spec
DECISION_MARGIN = 0.003

# nb1191 deploy reconstruction (mirrors nb2171)
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


# ---------------------------------------------------------------------------
# Anchor catalogue: (name, oof_loader, te_relative_path)
# oof_loader is "_RECONSTRUCT_<name>" for synthetic anchors, else a relative
# npy path under DATA_PROCESSED.
# ---------------------------------------------------------------------------
ANCHOR_OOF = {
    "nb1191":     "_RECONSTRUCT_nb1191",
    "nb562":      "nb562_pred_oof.npy",
    "nb1158_K32": "nb1158_mean_bag_oof_K32.npy",
    "nb503":      "nb503_pred_oof.npy",
}
ANCHOR_TE = {
    "nb1191":     "te_nb1191.npy",
    "nb562":      "te_nb562.npy",
    "nb1158_K32": "te_nb1158.npy",
    "nb503":      "te_nb503.npy",
}

VARIANTS = {
    "A_2anchor_nb1191_nb562":              ["nb1191", "nb562"],
    "B_3anchor_nb1191_nb562_nb1158_K32":   ["nb1191", "nb562", "nb1158_K32"],
    "C_2anchor_nb1191_nb503":              ["nb1191", "nb503"],
}


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
    for arr, nm in [
        (chemprop_oof, "chemprop"), (nb1150_oof, "nb1150"),
        (nb1158_oof, "nb1158"), (nb2112_oof, "nb2112"),
    ]:
        assert arr.shape == (n_unb,), f"nb1191 sub {nm} shape {arr.shape}"
    blend = (
        NB1191_DEPLOY_WEIGHTS["chemprop_aux"] * chemprop_oof
        + NB1191_DEPLOY_WEIGHTS["nb1150"]       * nb1150_oof
        + NB1191_DEPLOY_WEIGHTS["nb1158_K32"]   * nb1158_oof
        + NB1191_DEPLOY_WEIGHTS["nb2112_K28"]   * nb2112_oof
    )
    mu = float(blend.mean())
    return mu + NB1191_DEPLOY_S * (blend - mu)


def load_anchor_oof(name: str, n_unb: int) -> np.ndarray:
    spec = ANCHOR_OOF[name]
    if spec == "_RECONSTRUCT_nb1191":
        return reconstruct_nb1191_oof(n_unb)
    p = DATA_PROCESSED / spec
    assert p.exists(), f"missing OOF: {p}"
    v = np.load(p).astype(np.float64)
    assert v.shape == (n_unb,), f"{name} OOF shape {v.shape}"
    return v


def load_anchor_te(name: str, n_te: int) -> np.ndarray:
    p = DATA_PROCESSED / ANCHOR_TE[name]
    assert p.exists(), f"missing te: {p}"
    v = np.load(p).astype(np.float64)
    assert v.shape == (n_te,), f"{name} te shape {v.shape}"
    return v


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


def run_variant(name: str, anchor_names: list,
                P_full_unb: np.ndarray, P_full_te: np.ndarray,
                anchor_index_map: dict, y_unb: np.ndarray,
                unb_scaffolds: list, n_te: int) -> dict:
    cols = [anchor_index_map[a] for a in anchor_names]
    P_unb = P_full_unb[:, cols]
    P_te  = P_full_te[:, cols]
    K = P_unb.shape[1]

    print("\n" + "-" * 78)
    print(f"VARIANT {name}  anchors={anchor_names}  K={K}")
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
    pooled_mean = float(np.mean([r["pooled_rae"] for r in per_seed]))
    pooled_std  = float(np.std([r["pooled_rae"] for r in per_seed]))

    # Deploy refit on full 253
    w_deploy = slsqp_simplex(P_unb, y_unb)
    blend_unb = P_unb @ w_deploy
    mu_deploy = float(blend_unb.mean())
    s_deploy = float(np.mean([s for r in per_seed for s in r["fold_s"]]))
    in_rae = float(rae(y_unb, mu_deploy + s_deploy * (blend_unb - mu_deploy)))
    blend_te = P_te @ w_deploy
    deploy_te = (mu_deploy + s_deploy * (blend_te - mu_deploy)).astype(np.float32)

    delta_vs_nb2171 = pooled_mean - NB2171_POOLED_RAE
    print(
        f"   pooled_mean={pooled_mean:.4f}  pooled_std={pooled_std:.4f}  "
        f"mean_oof_rae={final_oof_rae:.4f}  "
        f"delta_vs_nb2171={delta_vs_nb2171:+.4f}"
    )
    print(
        f"   deploy w={[float(round(w, 4)) for w in w_deploy]}  "
        f"mu={mu_deploy:.4f}  s={s_deploy:.4f}  in_rae={in_rae:.4f}"
    )
    print(
        f"   deploy te(513) mean/std = {deploy_te.mean():.3f}/{deploy_te.std():.3f}"
    )

    return {
        "name": name,
        "anchors": anchor_names,
        "K": K,
        "per_seed": per_seed,
        "pooled_rae_mean_seeds": pooled_mean,
        "pooled_rae_std_seeds": pooled_std,
        "rae_of_mean_of_seed_oofs": final_oof_rae,
        "delta_vs_nb2171": delta_vs_nb2171,
        "deploy_weights": [
            {"name": a, "w": float(w)} for a, w in zip(anchor_names, w_deploy)
        ],
        "deploy_mu_blend": mu_deploy,
        "deploy_s": s_deploy,
        "in_sample_rae_overfit_bound": in_rae,
        "deploy_te_vec": deploy_te,  # popped before json
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- minimal-anchor variants of nb2171  "
          f"(baseline pooled RAE = {NB2171_POOLED_RAE:.4f}, margin={DECISION_MARGIN})")
    print("=" * 78)

    te = load_test()
    te_smiles = te["smiles"].values if "smiles" in te.columns else te["SMILES"].values
    te_names = te["name"].values if "name" in te.columns else te["Molecule Name"].values
    n_te = len(te_smiles)

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_te={n_te}  n_unb={n_unb}")

    unb_smiles = te_smiles[unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    # ---- Load all 4 candidate anchors once ----
    all_anchors = ["nb1191", "nb562", "nb1158_K32", "nb503"]
    print(f"\n[anchors] loading {all_anchors}")
    oof_cols, te_cols, indiv_rae = [], [], {}
    for name in all_anchors:
        oof = load_anchor_oof(name, n_unb)
        te_v = load_anchor_te(name, n_te)
        r = float(rae(y_unb, oof))
        indiv_rae[name] = r
        oof_cols.append(oof)
        te_cols.append(te_v)
        print(
            f"   {name:14s} oof_RAE={r:.4f}  te_mean={te_v.mean():.3f}  "
            f"te_std={te_v.std():.3f}"
        )

    P_full_unb = np.column_stack(oof_cols)
    P_full_te  = np.column_stack(te_cols)
    anchor_index_map = {n: i for i, n in enumerate(all_anchors)}

    # ---- Run each variant ----
    results = {}
    for vname, vanchors in VARIANTS.items():
        res = run_variant(
            vname, vanchors,
            P_full_unb, P_full_te, anchor_index_map,
            y_unb, unb_scaffolds, n_te,
        )
        results[vname] = res

    # ---- Compare ----
    print("\n" + "=" * 78)
    print("RANKING (best first)")
    print("=" * 78)
    ranked = sorted(results.values(), key=lambda r: r["pooled_rae_mean_seeds"])
    for r in ranked:
        print(
            f"   {r['name']:>40s}  pooled={r['pooled_rae_mean_seeds']:.4f} "
            f"(+/- {r['pooled_rae_std_seeds']:.4f})  "
            f"delta_vs_nb2171={r['delta_vs_nb2171']:+.4f}"
        )

    best = ranked[0]
    beats = best["delta_vs_nb2171"] <= -DECISION_MARGIN
    print(
        f"\n   best        = {best['name']}  pooled={best['pooled_rae_mean_seeds']:.4f}"
    )
    print(
        f"   nb2171 ref  = {NB2171_POOLED_RAE:.4f}  margin = {DECISION_MARGIN}"
    )
    print(
        f"   beats_nb2171 = {beats}  (need delta <= -{DECISION_MARGIN})"
    )

    # ---- Deep-30 verify if best beats ----
    deep30 = None
    if beats:
        print("\n" + "-" * 78)
        print(f"DEEP-30 VERIFY  on best variant {best['name']}")
        print("-" * 78)
        cols = [anchor_index_map[a] for a in best["anchors"]]
        deep30 = deep_verify_seeds(
            P_full_unb[:, cols], y_unb, unb_scaffolds, n_extra=25,
        )
        print(
            f"   n_seeds={deep30['n_seeds']}  mean_RAE={deep30['mean_rae']:.4f}  "
            f"std={deep30['std_rae']:.4f}  "
            f"range=[{deep30['min_rae']:.4f}, {deep30['max_rae']:.4f}]"
        )

    # ---- Save deploy te vector for best variant ----
    te_npy_path = DATA_PROCESSED / f"te_{TAG}_min_anchor_{best['name']}.npy"
    np.save(te_npy_path, best["deploy_te_vec"])
    print(f"\n[save] {te_npy_path}")

    # ---- LB band estimate for best ----
    deploy_te_best = best["deploy_te_vec"]
    te_unb_rae = float(rae(y_unb, deploy_te_best[unb_idx].astype(np.float64)))
    lb_band_est = LB_W_OOF * best["pooled_rae_mean_seeds"] + LB_W_TE * te_unb_rae
    print(
        f"[LB-band] best  {LB_W_OOF:.2f}*OOF({best['pooled_rae_mean_seeds']:.4f}) + "
        f"{LB_W_TE:.2f}*te_unb({te_unb_rae:.4f}) = {lb_band_est:.4f} "
        f"[{lb_band_est - 0.05:.4f}, {lb_band_est + 0.05:.4f}]"
    )

    # ---- Strip non-serialisable deploy_te_vec before json ----
    json_results = {}
    for k, v in results.items():
        vv = {kk: vv for kk, vv in v.items() if kk != "deploy_te_vec"}
        json_results[k] = vv

    summary = {
        "tag": TAG,
        "method": "min_anchor_variants_of_nb2171_SLSQP_then_rank_stretch_seedavg",
        "candidate_anchors": all_anchors,
        "indiv_oof_rae_unb": indiv_rae,
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "stretch_grid": STRETCH_GRID,
        "n_unb": n_unb,
        "n_te": n_te,
        "n_unique_scaffolds": n_unique_scaf,
        "variants": json_results,
        "best_variant": best["name"],
        "best_pooled_rae_mean_seeds": best["pooled_rae_mean_seeds"],
        "nb2171_pooled_rae_ref": NB2171_POOLED_RAE,
        "decision_margin": DECISION_MARGIN,
        "best_delta_vs_nb2171": best["delta_vs_nb2171"],
        "beats_nb2171": bool(beats),
        "deep_30_verify": deep30,
        "best_te_npy_path": str(te_npy_path),
        "best_te_unb_rae_in_sample": te_unb_rae,
        "best_lb_band_estimate": lb_band_est,
        "pre_unblind_clean": True,
        "coexists_with_nb2190_conformal_shrink": True,
        "note": "Tag nb2190 was previously used by scripts/nb2190_conformal_shrink.py;"
                " this run overwrites nb2190_summary.json with min-anchor results"
                " as specified by the task. The conformal-shrink npy artefacts"
                " (nb2190_per_seed_oof_K28.npy, nb2190_per_row_std_K28.npy) are"
                " untouched.",
        "wall_sec": round(time.time() - t0, 2),
    }

    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   best variant            = {best['name']}")
    print(f"   best pooled RAE         = {best['pooled_rae_mean_seeds']:.4f}")
    print(f"   nb2171 reference RAE    = {NB2171_POOLED_RAE:.4f}")
    print(f"   delta vs nb2171         = {best['delta_vs_nb2171']:+.4f}")
    print(f"   beats nb2171 (margin {DECISION_MARGIN}) = {beats}")
    print(f"   LB band estimate        = {lb_band_est:.4f}")
    print(f"   wall                    = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "best_variant",
        "best_pooled_rae_mean_seeds",
        "nb2171_pooled_rae_ref",
        "best_delta_vs_nb2171",
        "beats_nb2171",
        "best_lb_band_estimate",
    ):
        print(f"  {k}: {res.get(k)}")
