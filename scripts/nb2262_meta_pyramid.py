"""nb2262 -- Meta-SLSQP over already-pyramid'd predictors (pyramid-of-pyramids).

CONTEXT:
    nb2170 / prior cycles demonstrated that SLSQP on RAW anchors collapses to
    one or two non-zero weights and reproduces the strongest single anchor.
    This script tests a meta-level question: does SLSQP behave any differently
    when the inputs are themselves already-pyramid'd predictors?  Per the
    nb2170 prior, this is expected to collapse to the strongest pyramid
    (nb2240, OOF 0.4598) at weight=1, with the other three pyramids zeroed
    -- but we want the empirical confirmation and the diagnostic numbers.

INPUTS (4 already-pyramid'd predictors on the 253 unblind):
    0. nb2240  ->  K=20 RFE swap; OOF 0.4598 (already best, deploy primary).
    1. nb1191  ->  classic 4-anchor PRE pyramid; OOF 0.4703.
    2. nb2095  ->  nb1191 + nb1014 5-anchor pseudo-PRE; OOF 0.4703.
    3. nb2060  ->  6-anchor (nb1191 anchors + nb503 + nb562); OOF 0.4697.

OOF RECONSTRUCTION:
    Only nb2240 ships a cached pyramid OOF artefact
    (nb2240_mean_bag_oof_K20.npy is the K=20 RESIDUAL, NOT the post-pyramid
    blend!).  For all four we reconstruct the post-pyramid OOF from cached
    deploy_weights / deploy_mu_blend / deploy_s in each summary JSON:
        blend_oof = sum_i w_deploy_i * anchor_oof_i
        post_blend = mu_deploy + s_deploy * (blend_oof - mu_deploy)
    This is exactly the deploy formula applied to OOF columns rather than
    te columns, so it represents what the deploy blend produces on the
    253 unblind (a single mean-of-seeds expression, not the per-seed run).

PROTOCOL:
    1. Reconstruct 4 OOF columns and a 4-col P_unb (253, 4).
    2. Compute per-column individual OOF RAE + 4x4 Pearson correlation
       matrix (expect 0.99+).
    3. Sweep kf_seeds {1146..1175} (30 seeds), scaffold 5-fold CV per
       seed.  Per fold: SLSQP simplex (w>=0, sum=1) on train, predict on
       val.  No rank-stretch in the meta layer (inputs are already
       stretched; a second stretch would double-shrink).
    4. Pool pooled RAE across seeds (mean, std, median, ci95).
    5. Compute deploy weights via SLSQP on all 253; report mean weight,
       max weight, n_zero (w < 1e-6).
    6. Gate vs nb2240 OOF 0.4601 with margin 0.003: BEATS if
       pooled_rae_mean_seeds < 0.4571.

OUTPUTS:
    scripts/nb2262_meta_pyramid.py                 (this file)
    data/processed/nb2262_summary.json
    (NO submission CSV -- meta layer is diagnostic only)

EXPECTED VERDICT: COLLAPSE_TO_NB2240 (w_nb2240 -> 1, others -> 0; pooled_rae
close to 0.4601). If the meta-blend actually improves beyond gate, that
would itself be evidence of orthogonality structure missed by the lower
pyramids.
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
from scipy.optimize import minimize
from scipy.stats import pearsonr

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2262"
GATE_MARGIN = 0.003
NB2240_REF_OOF = 0.4601                  # nb2240 pooled_rae_mean_seeds (5 seed)
KF_SEEDS = list(range(1146, 1176))       # 30 seeds, per spec
N_FOLDS = 5
LB_W_OOF = 0.51
LB_W_TE = 0.49

# nb1150 anchor OOF reconstructed from cached SLSQP4 weights -- same recipe
# used by nb1191/nb2060/nb2095/nb2240.
NB1150_SLSQP4_OOFS = [
    "nb1133_chemprop_aux_pred_oof.npy",
    "nb503_pred_oof.npy",
    "nb1133_nb1014_pred_oof.npy",
    "nb2103_mean_bag_oof_K28.npy",
]
NB1150_SLSQP4_WEIGHTS = [0.0, 0.2942, 0.0, 0.7058]


def _load_oof(rel: str, n_unb: int) -> np.ndarray:
    p = DATA_PROCESSED / rel
    assert p.exists(), f"missing OOF cache: {p}"
    v = np.load(p).astype(np.float64)
    assert v.shape == (n_unb,), f"{rel} shape {v.shape} != ({n_unb},)"
    return v


def _reconstruct_nb1150_oof(n_unb: int) -> np.ndarray:
    cols = [_load_oof(rel, n_unb) for rel in NB1150_SLSQP4_OOFS]
    P = np.column_stack(cols)
    w = np.asarray(NB1150_SLSQP4_WEIGHTS, dtype=np.float64)
    return P @ w


def _post_pyramid_oof(anchor_oofs, deploy_weights, deploy_mu, deploy_s):
    """Apply deploy formula to OOF columns to reconstruct post-pyramid OOF."""
    P = np.column_stack(anchor_oofs)
    w = np.asarray(deploy_weights, dtype=np.float64)
    blend = P @ w
    return deploy_mu + deploy_s * (blend - deploy_mu)


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


def cv_run_for_seed(P_unb, y_unb, unb_scaffolds, kf_seed):
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


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- meta-SLSQP over 4 already-pyramid'd predictors")
    print("=" * 78)

    # ---- Load truth ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)

    te = load_test()
    te_smiles = te["smiles"].astype(str).tolist() if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[load] n_unb={n_unb}  n_unique_scaffolds={n_unique_scaf}")

    # =====================================================================
    # Reconstruct 4 post-pyramid OOF columns
    # =====================================================================
    pyramid_oofs = {}
    pyramid_meta = {}

    # ---------- nb2240 ----------
    sum_2240 = json.loads((DATA_PROCESSED / "nb2240_summary.json").read_text())
    # anchors order in nb2240: nb2240_K20, chemprop_aux, nb1191, nb503, nb562
    nb2240_K20_oof = _load_oof("nb2240_mean_bag_oof_K20.npy", n_unb)
    chemprop_oof = _load_oof("nb1133_chemprop_aux_pred_oof.npy", n_unb)
    nb503_oof = _load_oof("nb503_pred_oof.npy", n_unb)
    nb562_oof = _load_oof("nb562_pred_oof.npy", n_unb)
    nb1150_oof = _reconstruct_nb1150_oof(n_unb)
    nb1158_oof = _load_oof("nb1158_mean_bag_oof_K32.npy", n_unb)
    nb2112_oof = _load_oof("nb2103_mean_bag_oof_K28.npy", n_unb)
    nb1014_oof = _load_oof("nb1133_nb1014_pred_oof.npy", n_unb)

    # nb1191 inner stretch s=1.031 must be applied to nb1191 input first,
    # so reconstruct it directly
    w_2240 = [d["w"] for d in sum_2240["deploy_weights"]]
    mu_2240 = float(sum_2240["deploy_mu_blend"])
    s_2240 = float(sum_2240["deploy_s"])
    # nb2240 anchor #2 ("nb1191") needs its own post-pyramid OOF computed first
    sum_1191 = json.loads((DATA_PROCESSED / "nb1191_summary.json").read_text())
    w_1191 = [d["w"] for d in sum_1191["deploy_weights"]]
    mu_1191 = float(sum_1191["deploy_mu_blend"])
    s_1191 = float(sum_1191["deploy_s"])
    # nb1191 anchors: chemprop_aux, nb1150, nb1158_K32, nb2112_K28
    nb1191_oof = _post_pyramid_oof(
        [chemprop_oof, nb1150_oof, nb1158_oof, nb2112_oof],
        w_1191, mu_1191, s_1191,
    )

    # Now nb2240 = post(nb2240_K20, chemprop_aux, nb1191, nb503, nb562)
    nb2240_oof = _post_pyramid_oof(
        [nb2240_K20_oof, chemprop_oof, nb1191_oof, nb503_oof, nb562_oof],
        w_2240, mu_2240, s_2240,
    )
    pyramid_oofs["nb2240"] = nb2240_oof
    pyramid_meta["nb2240"] = {
        "anchors": [a["name"] for a in sum_2240["deploy_weights"]],
        "deploy_weights": w_2240,
        "deploy_mu": mu_2240,
        "deploy_s": s_2240,
        "claim_oof": float(sum_2240.get("pooled_rae_mean_seeds", float("nan"))),
    }

    # ---------- nb1191 ---------- (already built above)
    pyramid_oofs["nb1191"] = nb1191_oof
    pyramid_meta["nb1191"] = {
        "anchors": [a["name"] for a in sum_1191["deploy_weights"]],
        "deploy_weights": w_1191,
        "deploy_mu": mu_1191,
        "deploy_s": s_1191,
        "claim_oof": float(sum_1191.get("pooled_rae_mean_seeds", float("nan"))),
    }

    # ---------- nb2095 ---------- 5 anchors: chemprop_aux, nb1150, nb1158_K32, nb2112_K28, nb1014
    sum_2095 = json.loads((DATA_PROCESSED / "nb2095_summary.json").read_text())
    w_2095 = [d["w"] for d in sum_2095["deploy_weights"]]
    mu_2095 = float(sum_2095["deploy_mu_blend"])
    s_2095 = float(sum_2095["deploy_s"])
    nb2095_oof = _post_pyramid_oof(
        [chemprop_oof, nb1150_oof, nb1158_oof, nb2112_oof, nb1014_oof],
        w_2095, mu_2095, s_2095,
    )
    pyramid_oofs["nb2095"] = nb2095_oof
    pyramid_meta["nb2095"] = {
        "anchors": [a["name"] for a in sum_2095["deploy_weights"]],
        "deploy_weights": w_2095,
        "deploy_mu": mu_2095,
        "deploy_s": s_2095,
        "claim_oof": float(sum_2095.get("pooled_rae_mean_seeds", float("nan"))),
    }

    # ---------- nb2060 ---------- 6 anchors: chemprop_aux, nb1150, nb1158_K32, nb2112_K28, nb503, nb562
    sum_2060 = json.loads((DATA_PROCESSED / "nb2060_summary.json").read_text())
    w_2060 = [d["w"] for d in sum_2060["deploy_weights"]]
    mu_2060 = float(sum_2060["deploy_mu_blend"])
    s_2060 = float(sum_2060["deploy_s"])
    nb2060_oof = _post_pyramid_oof(
        [chemprop_oof, nb1150_oof, nb1158_oof, nb2112_oof, nb503_oof, nb562_oof],
        w_2060, mu_2060, s_2060,
    )
    pyramid_oofs["nb2060"] = nb2060_oof
    pyramid_meta["nb2060"] = {
        "anchors": [a["name"] for a in sum_2060["deploy_weights"]],
        "deploy_weights": w_2060,
        "deploy_mu": mu_2060,
        "deploy_s": s_2060,
        "claim_oof": float(sum_2060.get("pooled_rae_mean_seeds", float("nan"))),
    }

    pyramid_names = ["nb2240", "nb1191", "nb2095", "nb2060"]
    P_unb = np.column_stack([pyramid_oofs[n] for n in pyramid_names])
    K = P_unb.shape[1]

    # ---- Individual OOF RAE ----
    indiv_rae = {}
    print("\n[indiv RAE]")
    for n in pyramid_names:
        r = float(rae(y_unb, pyramid_oofs[n]))
        indiv_rae[n] = r
        claim = pyramid_meta[n].get("claim_oof", float("nan"))
        print(f"   {n:8s} reconstructed_OOF={r:.4f}  claim_oof_5seed={claim:.4f}  "
              f"delta={r - claim:+.4f}")

    # ---- Pearson correlation matrix ----
    pearson = np.eye(K, dtype=np.float64)
    for i in range(K):
        for j in range(i + 1, K):
            r_ij, _ = pearsonr(P_unb[:, i], P_unb[:, j])
            pearson[i, j] = pearson[j, i] = float(r_ij)
    print("\n[pearson matrix]")
    print("           " + " ".join(f"{n:>8s}" for n in pyramid_names))
    for i, ni in enumerate(pyramid_names):
        row = "  ".join(f"{pearson[i, j]:.4f}  " for j in range(K))
        print(f"   {ni:8s}  {row}")
    pearson_min = float(np.min(pearson[np.triu_indices(K, k=1)]))
    pearson_max = float(np.max(pearson[np.triu_indices(K, k=1)]))
    pearson_mean = float(np.mean(pearson[np.triu_indices(K, k=1)]))
    print(f"   off-diag min={pearson_min:.4f}  mean={pearson_mean:.4f}  max={pearson_max:.4f}")

    # ---- Scaffold 5-fold meta-SLSQP across 30 seeds ----
    print("\n" + "-" * 78)
    print(f"META-SLSQP CV  kf_seeds={KF_SEEDS[0]}..{KF_SEEDS[-1]}  ({len(KF_SEEDS)} seeds)")
    print("-" * 78)
    per_seed = []
    all_oofs = []
    fold_w_all = []
    for kf_seed in KF_SEEDS:
        pooled, oof_blend, fw = cv_run_for_seed(P_unb, y_unb, unb_scaffolds, kf_seed)
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": float(pooled),
            "fold_w_mean": [float(x) for x in np.mean(fw, axis=0)],
        })
        all_oofs.append(oof_blend)
        fold_w_all.extend(fw)
        if (kf_seed - KF_SEEDS[0]) % 5 == 0 or kf_seed == KF_SEEDS[-1]:
            wm = np.round(np.mean(fw, axis=0), 4).tolist()
            print(f"   seed={kf_seed}  pooled_RAE={pooled:.4f}  w_mean={wm}")
    raes = np.array([r["pooled_rae"] for r in per_seed], dtype=np.float64)
    pooled_rae_mean_seeds = float(raes.mean())
    pooled_rae_std_seeds = float(raes.std())
    pooled_rae_median = float(np.median(raes))
    pooled_rae_min = float(raes.min())
    pooled_rae_max = float(raes.max())
    sem = pooled_rae_std_seeds / np.sqrt(len(raes))
    ci95_low = pooled_rae_mean_seeds - 1.96 * sem
    ci95_high = pooled_rae_mean_seeds + 1.96 * sem
    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    rae_of_mean = float(rae(y_unb, mean_oof))

    # Average fold weights (over 30*5 = 150 fits)
    fold_w_arr = np.array(fold_w_all, dtype=np.float64)
    fold_w_mean = fold_w_arr.mean(axis=0)
    fold_w_std = fold_w_arr.std(axis=0)
    fold_w_max = fold_w_arr.max(axis=0)
    fold_w_zero_frac = (fold_w_arr < 1e-6).mean(axis=0)

    print(f"\n[cv] mean across {len(KF_SEEDS)} seeds: pooled_RAE = {pooled_rae_mean_seeds:.4f} "
          f"(+/- {pooled_rae_std_seeds:.4f})")
    print(f"[cv] median = {pooled_rae_median:.4f}  range = [{pooled_rae_min:.4f}, {pooled_rae_max:.4f}]")
    print(f"[cv] CI95 = [{ci95_low:.4f}, {ci95_high:.4f}]  rae_of_mean_oofs = {rae_of_mean:.4f}")
    print(f"\n[fold-w stats over {len(fold_w_all)} fits]")
    for j, n in enumerate(pyramid_names):
        print(f"   {n:8s}  mean={fold_w_mean[j]:.4f}  std={fold_w_std[j]:.4f}  "
              f"max={fold_w_max[j]:.4f}  zero_frac={fold_w_zero_frac[j]:.3f}")

    # ---- Deploy SLSQP on all 253 ----
    w_deploy = slsqp_simplex(P_unb, y_unb)
    deploy_blend = P_unb @ w_deploy
    in_rae = float(rae(y_unb, deploy_blend))
    n_zero_deploy = int(np.sum(w_deploy < 1e-6))
    print("\n[deploy SLSQP on all 253]")
    for j, n in enumerate(pyramid_names):
        print(f"   {n:8s}  w_deploy={w_deploy[j]:.4f}")
    print(f"   in_sample RAE = {in_rae:.4f}  n_zero_deploy={n_zero_deploy}")

    # ---- Gate vs nb2240 ----
    delta_vs_nb2240 = pooled_rae_mean_seeds - NB2240_REF_OOF
    gate_beat = delta_vs_nb2240 < -GATE_MARGIN
    gate_flat = abs(delta_vs_nb2240) <= GATE_MARGIN
    if gate_beat:
        verdict = "BEATS_NB2240"
    elif gate_flat:
        verdict = "FLAT_VS_NB2240"
    else:
        # Likely collapse-to-nb2240 (pooled_rae ~ 0.4601)
        # Specifically check: does deploy weight concentrate on nb2240?
        if w_deploy[0] > 0.95:
            verdict = "COLLAPSE_TO_NB2240"
        else:
            verdict = "HURTS_NB2240"
    print("\n" + "-" * 78)
    print(f"GATE EVAL: pooled_rae_mean = {pooled_rae_mean_seeds:.4f}  vs nb2240 {NB2240_REF_OOF:.4f}  "
          f"(margin {GATE_MARGIN})")
    print(f"   delta = {delta_vs_nb2240:+.4f}  -> {verdict}")
    print("-" * 78)

    # ---- Summary ----
    summary = {
        "tag": TAG,
        "method": "meta_SLSQP_over_4_pyramids_no_meta_stretch",
        "pyramid_names": pyramid_names,
        "pyramid_meta": pyramid_meta,
        "reconstructed_indiv_oof_rae": indiv_rae,
        "pearson_matrix": pearson.tolist(),
        "pearson_offdiag_min": pearson_min,
        "pearson_offdiag_mean": pearson_mean,
        "pearson_offdiag_max": pearson_max,
        "kf_seeds": KF_SEEDS,
        "n_kf_seeds": len(KF_SEEDS),
        "n_folds": N_FOLDS,
        "n_unb": n_unb,
        "n_unique_scaffolds": n_unique_scaf,
        "per_seed_results": per_seed,
        "pooled_rae_mean_seeds": pooled_rae_mean_seeds,
        "pooled_rae_std_seeds": pooled_rae_std_seeds,
        "pooled_rae_median": pooled_rae_median,
        "pooled_rae_min": pooled_rae_min,
        "pooled_rae_max": pooled_rae_max,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "rae_of_mean_of_seed_oofs": rae_of_mean,
        "fold_w_mean": [float(x) for x in fold_w_mean],
        "fold_w_std": [float(x) for x in fold_w_std],
        "fold_w_max": [float(x) for x in fold_w_max],
        "fold_w_zero_frac": [float(x) for x in fold_w_zero_frac],
        "deploy_weights": [
            {"name": pyramid_names[j], "w": float(w_deploy[j])} for j in range(K)
        ],
        "deploy_in_sample_rae": in_rae,
        "deploy_n_zero": n_zero_deploy,
        "deploy_w_nb2240": float(w_deploy[0]),
        "compare_nb2240_oof": NB2240_REF_OOF,
        "delta_vs_nb2240": delta_vs_nb2240,
        "gate_margin": GATE_MARGIN,
        "gate_beat_nb2240": bool(gate_beat),
        "gate_flat_vs_nb2240": bool(gate_flat),
        "verdict_vs_nb2240": verdict,
        "no_submission_csv": True,
        "lb_band_w_oof": LB_W_OOF,
        "lb_band_w_te": LB_W_TE,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   indiv OOF (reconstructed):")
    for n in pyramid_names:
        print(f"      {n:8s}  {indiv_rae[n]:.4f}")
    print(f"   pearson off-diag: min={pearson_min:.4f} mean={pearson_mean:.4f} max={pearson_max:.4f}")
    print(f"   meta-SLSQP pooled RAE (30 seeds) = {pooled_rae_mean_seeds:.4f} +/- {pooled_rae_std_seeds:.4f}")
    print(f"   delta vs nb2240 (0.4601)         = {delta_vs_nb2240:+.4f}")
    print(f"   deploy w_nb2240 / n_zero         = {w_deploy[0]:.4f} / {n_zero_deploy}")
    print(f"   verdict                          = {verdict}")
    print(f"   wall                             = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "reconstructed_indiv_oof_rae",
        "pearson_offdiag_min", "pearson_offdiag_mean", "pearson_offdiag_max",
        "pooled_rae_mean_seeds", "pooled_rae_std_seeds",
        "delta_vs_nb2240", "verdict_vs_nb2240",
        "deploy_weights", "deploy_in_sample_rae", "deploy_n_zero",
        "wall_sec",
    ):
        print(f"  {k}: {res.get(k)}")
