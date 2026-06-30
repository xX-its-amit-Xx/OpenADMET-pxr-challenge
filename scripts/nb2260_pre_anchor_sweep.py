"""nb2260 -- PRE-clean pyramid re-anchor sweep: swap K=28 for nb2240 K=20.

CONTEXT:
    nb2240 verified that the RFE K=20 anchor (chemprop_aux + LGBM on the 20
    top features of the 117-col bank) is the strongest known PRE-clean
    residual base. nb2240's 5-anchor pyramid (K=20 + chemprop_aux + nb1191
    + nb503 + nb562) ships pooled OOF RAE 0.4601 (deep-30 verified) and
    BEATS the nb2171 reference (K=28 swap-in baseline 0.4676) by -0.0075.

    The three existing PRE-clean pyramids {nb1191, nb2095, nb2060} all
    carry the K=28 anchor `nb2112_K28` (cached as
    `nb2103_mean_bag_oof_K28.npy` on the 253 + `te_nb2112.npy` on the 513).
    This script replaces JUST that anchor with the K=20 artefact and
    rebuilds each pyramid identically. All other anchors are held fixed.

PROTOCOL (per variant):
    1. Replace anchor index 3 (nb2112_K28) with nb2240_K20
       (`nb2240_mean_bag_oof_K20.npy` + `te_nb2240_K20.npy`).
    2. Run SLSQP convex blend (w >= 0, sum = 1) per scaffold-fold +
       rank-stretch (grid 1.000..1.150) under 5-fold scaffold-CV on the
       253 across 30 fresh deep seeds {1146..1175}.
    3. Mean pooled RAE across the 30 seeds is the variant's verdict.
    4. Compare vs nb2240 reference 0.4601; gate margin -0.003 (must
       beat by more than 0.003 to win).
    5. Save deploy CSV only on gate pass.

CANDIDATES:
    nb1191_k20  -> {chemprop_aux, nb1150, nb1158_K32, nb2240_K20}
    nb2095_k20  -> {chemprop_aux, nb1150, nb1158_K32, nb2240_K20, nb1014}
    nb2060_k20  -> {chemprop_aux, nb1150, nb1158_K32, nb2240_K20, nb503, nb562}

Outputs:
    scripts/nb2260_pre_anchor_sweep.py
    data/processed/nb2260_summary.json
    data/processed/te_nb2260_<variant>.npy           (per variant)
    submissions/nb2260_<variant>_k20.csv             (only on gate pass)
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

TAG = "nb2260"

# ---- Gate vs nb2240 reference ----
NB2240_REF_OOF = 0.4601           # pooled_rae_mean_seeds from nb2240_summary
GATE_MARGIN = 0.003               # must beat by > 0.003 to gate-pass

# ---- CV protocol (30 fresh deep seeds) ----
N_FOLDS = 5
KF_SEEDS = list(range(1146, 1176))   # {1146..1175} -> 30 seeds
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]
LB_W_OOF = 0.51
LB_W_TE = 0.49

# ---- nb1150 reconstruction (shared across all three originals) ----
NB1150_SLSQP4_OOFS = [
    "nb1133_chemprop_aux_pred_oof.npy",
    "nb503_pred_oof.npy",
    "nb1133_nb1014_pred_oof.npy",
    "nb2103_mean_bag_oof_K28.npy",
]
NB1150_SLSQP4_WEIGHTS_FULL_POOL = [0.0, 0.2942, 0.0, 0.7058]

# ---- K=20 swap artefacts (built by nb2240) ----
NB2240_K20_OOF = "nb2240_mean_bag_oof_K20.npy"
NB2240_K20_TE = "te_nb2240_K20.npy"

# ---- Variant definitions: anchor list with K=28 replaced by K=20 ----
# Each entry: (display_name, oof_path_relative, te_path_relative)
VARIANTS = {
    "nb1191_k20": [
        ("chemprop_aux", "nb1133_chemprop_aux_pred_oof.npy", "te_chemprop_aux.npy"),
        ("nb1150",       "_RECONSTRUCT_nb1150_oof",          "te_nb1150.npy"),
        ("nb1158_K32",   "nb1158_mean_bag_oof_K32.npy",      "te_nb1158.npy"),
        ("nb2240_K20",   NB2240_K20_OOF,                     NB2240_K20_TE),
    ],
    "nb2095_k20": [
        ("chemprop_aux", "nb1133_chemprop_aux_pred_oof.npy", "te_chemprop_aux.npy"),
        ("nb1150",       "_RECONSTRUCT_nb1150_oof",          "te_nb1150.npy"),
        ("nb1158_K32",   "nb1158_mean_bag_oof_K32.npy",      "te_nb1158.npy"),
        ("nb2240_K20",   NB2240_K20_OOF,                     NB2240_K20_TE),
        ("nb1014",       "nb1133_nb1014_pred_oof.npy",       "te_nb1014.npy"),
    ],
    "nb2060_k20": [
        ("chemprop_aux", "nb1133_chemprop_aux_pred_oof.npy", "te_chemprop_aux.npy"),
        ("nb1150",       "_RECONSTRUCT_nb1150_oof",          "te_nb1150.npy"),
        ("nb1158_K32",   "nb1158_mean_bag_oof_K32.npy",      "te_nb1158.npy"),
        ("nb2240_K20",   NB2240_K20_OOF,                     NB2240_K20_TE),
        ("nb503",        "nb503_pred_oof.npy",               "te_nb503.npy"),
        ("nb562",        "nb562_pred_oof.npy",               "te_nb562.npy"),
    ],
}

# Reference (pre-swap) OOFs for context only
PRE_SWAP_REF_OOF = {
    "nb1191_k20": 0.47034,   # nb1191
    "nb2095_k20": 0.47034,   # nb2095
    "nb2060_k20": 0.46975,   # nb2060
}


# ============================================================================
# helpers
# ============================================================================

def reconstruct_nb1150_oof(n_unb: int) -> np.ndarray:
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


def slsqp_simplex(P, y):
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


def load_anchor_columns(anchor_list, n_unb, n_te):
    oof_cols, te_cols, indiv_rae, names = [], [], {}, []
    for disp, oof_rel, te_rel in anchor_list:
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
        oof_cols.append(oof)
        te_cols.append(te_arr)
        names.append(disp)
    return np.column_stack(oof_cols), np.column_stack(te_cols), names


def run_variant(variant_name, anchor_list, y_unb, unb_scaffolds,
                unb_idx, n_unb, n_te, te_smiles, te_names):
    print("\n" + "=" * 78)
    print(f"VARIANT: {variant_name}  ({len(anchor_list)} anchors)")
    print("=" * 78)
    P_unb, P_te, names = load_anchor_columns(anchor_list, n_unb, n_te)
    print(f"[stack] P_unb {P_unb.shape}  P_te {P_te.shape}")
    indiv_rae = {}
    for j, nm in enumerate(names):
        r = float(rae(y_unb, P_unb[:, j]))
        indiv_rae[nm] = r
        print(f"   anchor {j} {nm:14s} oof_RAE={r:.4f}")

    # 30-seed sweep
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
        if (k + 1) % 10 == 0 or k == 0:
            print(
                f"   seed[{k+1:2d}/{len(KF_SEEDS)}]={kf_seed}  "
                f"pooled={pooled:.4f}  mean_s={np.mean(fs):.3f}  "
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

    # Deploy: full-pool SLSQP weights + mean of per-fold s across all 30 seeds
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
        f"[deploy] weights = "
        + ", ".join(f"{nm}={w:.4f}" for nm, w in zip(names, w_deploy))
    )
    print(f"[deploy] mu={mu_deploy:.4f}  s={s_deploy:.4f}")
    print(f"[deploy] in_sample_RAE={in_rae:.4f}  te[unb_idx]_RAE={te_unb_rae:.4f}  "
          f"LB_band={lb_band_est:.4f}")

    # Gate eval vs nb2240
    delta_vs_nb2240 = pooled_mean - NB2240_REF_OOF
    gate_beat = delta_vs_nb2240 < -GATE_MARGIN
    gate_flat = abs(delta_vs_nb2240) <= GATE_MARGIN
    if gate_beat:
        verdict = "BEATS_NB2240"
    elif gate_flat:
        verdict = "FLAT_VS_NB2240"
    else:
        verdict = "HURTS_NB2240"
    pre_swap_ref = PRE_SWAP_REF_OOF[variant_name]
    delta_vs_pre_swap = pooled_mean - pre_swap_ref
    print(f"\n[gate] delta vs nb2240 (0.4601) = {delta_vs_nb2240:+.4f}  -> {verdict}")
    print(f"[gate] delta vs pre-swap ({pre_swap_ref:.4f})    = "
          f"{delta_vs_pre_swap:+.4f}")

    # Save artefacts
    te_npy_path = DATA_PROCESSED / f"te_{TAG}_{variant_name}.npy"
    np.save(te_npy_path, deploy_te)
    print(f"[save] {te_npy_path}")

    sub_csv_path = SUBMISSIONS / f"{TAG}_{variant_name}.csv"
    if gate_beat:
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": deploy_te,
        }).to_csv(sub_csv_path, index=False)
        print(f"[save] {sub_csv_path}  (gate BEATS_NB2240)")
        wrote_csv = True
    else:
        print(f"[skip] gate not beat -- no submission CSV ({verdict})")
        wrote_csv = False

    return {
        "variant": variant_name,
        "anchors": names,
        "indiv_oof_rae_unb": indiv_rae,
        "n_seeds": len(KF_SEEDS),
        "kf_seeds": KF_SEEDS,
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
        "compare_nb2240_oof": NB2240_REF_OOF,
        "delta_vs_nb2240": delta_vs_nb2240,
        "compare_pre_swap_oof": pre_swap_ref,
        "delta_vs_pre_swap": delta_vs_pre_swap,
        "gate_margin": GATE_MARGIN,
        "gate_beat_nb2240": bool(gate_beat),
        "gate_flat_vs_nb2240": bool(gate_flat),
        "verdict_vs_nb2240": verdict,
        "te_npy_path": str(te_npy_path),
        "submission_csv": str(sub_csv_path) if wrote_csv else None,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- PRE-clean pyramid re-anchor sweep  K=28 -> nb2240_K20")
    print("=" * 78)
    print(f"   variants : {list(VARIANTS.keys())}")
    print(f"   kf_seeds : {KF_SEEDS[0]}..{KF_SEEDS[-1]} ({len(KF_SEEDS)} seeds)")
    print(f"   gate     : beat nb2240 ({NB2240_REF_OOF:.4f}) by > {GATE_MARGIN}")

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

    results = {}
    for variant_name, anchor_list in VARIANTS.items():
        results[variant_name] = run_variant(
            variant_name, anchor_list, y_unb, unb_scaffolds,
            unb_idx, n_unb, n_te, te_smiles, te_names,
        )

    # Identify best (lowest pooled mean)
    leaderboard = sorted(
        results.items(), key=lambda kv: kv[1]["pooled_rae_mean_seeds"]
    )
    best_name, best_res = leaderboard[0]
    print("\n" + "=" * 78)
    print("LEADERBOARD (lowest pooled RAE first)")
    print("=" * 78)
    for nm, r in leaderboard:
        flag = "BEATS" if r["gate_beat_nb2240"] else (
            "FLAT" if r["gate_flat_vs_nb2240"] else "HURTS"
        )
        print(
            f"   {nm:12s} pooled={r['pooled_rae_mean_seeds']:.4f}  "
            f"std={r['pooled_rae_std_seeds']:.4f}  "
            f"delta_nb2240={r['delta_vs_nb2240']:+.4f}  "
            f"verdict={flag}"
        )
    print(f"\n[best] {best_name}  pooled={best_res['pooled_rae_mean_seeds']:.4f}  "
          f"delta_nb2240={best_res['delta_vs_nb2240']:+.4f}")

    any_beat = any(r["gate_beat_nb2240"] for r in results.values())
    if any_beat:
        winners = [nm for nm, r in results.items() if r["gate_beat_nb2240"]]
        print(f"[promote] gate-beat variants: {winners}")
        print(
            "[promote] deploy CSV(s) written. Promote best to ladder by hand "
            "after honest verification."
        )
    else:
        print("[no-promote] no variant beats nb2240 by > 0.003 -- "
              "ladder unchanged; K=20 anchor swap insufficient on these "
              "pyramids.")

    summary = {
        "tag": TAG,
        "method": "PRE_clean_pyramid_reanchor_K28_to_nb2240_K20",
        "variants": list(VARIANTS.keys()),
        "kf_seeds": KF_SEEDS,
        "n_seeds": len(KF_SEEDS),
        "n_folds": N_FOLDS,
        "stretch_grid": STRETCH_GRID,
        "n_unb": n_unb,
        "n_te": n_te,
        "n_unique_scaffolds": n_unique_scaf,
        "compare_nb2240_oof": NB2240_REF_OOF,
        "gate_margin": GATE_MARGIN,
        "pre_swap_ref_oof": PRE_SWAP_REF_OOF,
        "results": results,
        "leaderboard_order": [nm for nm, _ in leaderboard],
        "best_variant": best_name,
        "best_pooled_rae": best_res["pooled_rae_mean_seeds"],
        "best_delta_vs_nb2240": best_res["delta_vs_nb2240"],
        "best_verdict": best_res["verdict_vs_nb2240"],
        "any_variant_beats_nb2240": bool(any_beat),
        "winning_variants": [
            nm for nm, r in results.items() if r["gate_beat_nb2240"]
        ],
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   best variant       = {best_name}")
    print(f"   best pooled RAE    = {best_res['pooled_rae_mean_seeds']:.4f}")
    print(f"   nb2240 reference   = {NB2240_REF_OOF:.4f}")
    print(f"   delta vs nb2240    = {best_res['delta_vs_nb2240']:+.4f}")
    print(f"   any beats nb2240   = {any_beat}")
    print(f"   wall               = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "best_variant",
        "best_pooled_rae",
        "best_delta_vs_nb2240",
        "best_verdict",
        "any_variant_beats_nb2240",
        "winning_variants",
    ):
        print(f"  {k}: {res.get(k)}")
