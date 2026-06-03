"""nb1051 -- Quad-orthogonal multi-seed SLSQP+stretch bag.

Pool (K=4):
    0. chemprop_aux       (anchor; PRIMARY-1 candidate)
    1. nb972_long_train   (long-train Chemprop)
    2. nb1030             (Mordred LGBM)
    3. nb1042             (Avalon FP LGBM)

Hypothesis: 4-way SLSQP simplex can extract more orthogonal value than
the nb1014 3-way bag at n=253 unblind. Mordred (Avalon) features are
descriptor-axis orthogonal to chemprop's learned MPNN representation,
and to each other (RDKit-style descriptors vs path-fingerprints).

Procedure (5 KFold seeds x 5 folds = 25 honest fits):
    For seed in {0, 1, 7, 42, 137}:
        KFold(shuffle=True, random_state=seed) on the 253 unblind.
        Per fold:
            a. SLSQP w0..w3 in [0,1], sum=1 on the 4 train folds (SSE).
            b. Grid-scan s in {1.00, 1.05, ..., 2.00} on train-fold
               blend around train-fold mu.
            c. Apply (w_f, s_f, mu_tr) to held-out fold; collect OOF.
        Record pooled cross-fit RAE for this seed.
    Bag = mean over seeds (RAE + per-fold (w, s)).

Deploy: apply (mean_w, mean_s) with global blend mean as anchor to 513.

Reference:
    nb1014 3-way bag       -- mean cross-fit RAE for tie/beat checks.

Outputs:
    data/processed/te_nb1051.npy
    data/processed/nb1051_summary.json
    submissions/nb1051_quad_ortho_bag.csv
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

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1051"
# (display_name, te_npy_stem, submission_csv_stem)
CANDIDATES = [
    ("chemprop_aux",      "chemprop_aux",      "chemprop_aux"),
    ("nb972_long_train",  "nb972_long_train",  "nb972_long_train_optim"),
    ("nb1030_mordred",    "nb1030",            "nb1030_mordred_lgbm"),
    ("nb1042_avalon",     "nb1042",            "nb1042_avalon_fp_lgbm"),
]
STRETCH_GRID = np.round(np.arange(1.00, 2.001, 0.05), 2).tolist()
N_FOLDS = 5
SEEDS = [0, 1, 7, 42, 137]
NB1014_BAG_REF = 0.5994  # nb1001 seed=42 reference (nb1014 bag mean)


def load_te(stem: str, csv_stem: str, te_names: np.ndarray) -> np.ndarray:
    npy = DATA_PROCESSED / f"te_{stem}.npy"
    if npy.exists():
        arr = np.load(npy).astype(np.float64)
        if arr.shape[0] == len(te_names):
            return arr
        print(f"   [warn] te_{stem}.npy shape {arr.shape}, expected "
              f"{len(te_names)}; falling back to csv")
    sub = pd.read_csv(SUBMISSIONS / f"{csv_stem}.csv")
    assert (sub["Molecule Name"].values == te_names).all(), (
        f"{csv_stem}: submission row order does not match test order")
    return sub["pEC50"].values.astype(np.float64)


def slsqp_simplex(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit w on K-simplex (>=0, sum=1) minimizing SSE of P@w vs y."""
    K = P.shape[1]
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * K
    res = minimize(
        lambda w: float(np.sum((P @ w - y) ** 2)),
        np.full(K, 1.0 / K),
        method="SLSQP", bounds=bnds, constraints=cons,
        options={"ftol": 1e-10, "maxiter": 1000},
    )
    w = np.clip(res.x, 0.0, 1.0)
    s = w.sum()
    return w / s if s > 0 else np.full(K, 1.0 / K)


def best_stretch_on(blend_train: np.ndarray, y_train: np.ndarray,
                    mu: float) -> tuple[float, float]:
    """Grid-scan s on the train-fold; return (best_s, best_train_rae)."""
    best_s, best_r = 1.0, float("inf")
    for s in STRETCH_GRID:
        stretched = mu + s * (blend_train - mu)
        r = float(rae(y_train, stretched))
        if r < best_r:
            best_r = r
            best_s = float(s)
    return best_s, best_r


def run_one_seed(P_unb: np.ndarray, y_unb: np.ndarray, seed: int) -> dict:
    """Run the 5-fold SLSQP+stretch protocol for one KFold seed."""
    n_unb = len(y_unb)
    K = P_unb.shape[1]
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n_unb, np.nan)
    folds = []
    for k, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n_unb))):
        w_f = slsqp_simplex(P_unb[tr_loc], y_unb[tr_loc])
        blend_tr = P_unb[tr_loc] @ w_f
        mu_tr = float(blend_tr.mean())
        s_f, rae_tr = best_stretch_on(blend_tr, y_unb[tr_loc], mu_tr)
        blend_va = P_unb[va_loc] @ w_f
        oof[va_loc] = mu_tr + s_f * (blend_va - mu_tr)
        rae_va = float(rae(y_unb[va_loc], oof[va_loc]))
        folds.append({
            "fold": k,
            "w": w_f.tolist(),
            "s": s_f,
            "mu_tr": mu_tr,
            "train_rae": rae_tr,
            "val_rae": rae_va,
            "n_va": int(len(va_loc)),
        })
    pooled = float(rae(y_unb, oof))
    return {"seed": seed, "folds": folds, "pooled_rae": pooled, "oof": oof}


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- quad-ortho 4-way SLSQP+stretch multi-seed bag "
          f"(seeds={SEEDS})")
    print("=" * 78)

    # ---- Load 513 test ----
    te = load_test()
    te_names = te["name"].values
    n_te = len(te_names)
    preds_513 = np.column_stack([
        load_te(stem, csv_stem, te_names)
        for _, stem, csv_stem in CANDIDATES
    ])
    print(f"[load] preds_513 shape = {preds_513.shape}")

    # ---- Load 253 unblind ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    P_unb = preds_513[unb_idx]
    n_unb = len(y_unb)
    K = P_unb.shape[1]
    print(f"[load] P_unb shape = {P_unb.shape}, y shape = {y_unb.shape}")

    # ---- Individual in_RAE + pairwise corr ----
    print("\n[indiv] in_RAE on 253 unblind:")
    indiv_rae = {}
    for j, (name, _, _) in enumerate(CANDIDATES):
        r = float(rae(y_unb, P_unb[:, j]))
        indiv_rae[name] = r
        print(f"   {name:24s}: {r:.4f}")

    print("\n[corr] pairwise Pearson on 253:")
    corr = np.corrcoef(P_unb.T)
    names = [c[0] for c in CANDIDATES]
    for i in range(K):
        for j in range(i + 1, K):
            print(f"   {names[i]:24s} vs {names[j]:24s}: "
                  f"r={corr[i, j]:+.3f}")

    # =================================================================
    # Run multi-seed cross-fit
    # =================================================================
    print("\n" + "-" * 78)
    print(f"MULTI-SEED 4-WAY CROSS-FIT  (N_FOLDS={N_FOLDS}, "
          f"stretch grid {STRETCH_GRID[0]}..{STRETCH_GRID[-1]})")
    print("-" * 78)

    seed_results = []
    per_seed_rae = []
    all_w = []  # 25 x K
    all_s = []  # 25
    for seed in SEEDS:
        res = run_one_seed(P_unb, y_unb, seed)
        seed_results.append({
            "seed": seed,
            "pooled_rae": res["pooled_rae"],
            "folds": res["folds"],
        })
        per_seed_rae.append(res["pooled_rae"])
        for f in res["folds"]:
            all_w.append(f["w"])
            all_s.append(f["s"])
        fold_vals = [round(f["val_rae"], 3) for f in res["folds"]]
        print(f"   seed {seed:>3d}: pooled_RAE = {res['pooled_rae']:.4f}  "
              f"(folds: {fold_vals})")

    mean_rae = float(np.mean(per_seed_rae))
    std_rae = float(np.std(per_seed_rae))
    print(f"\n[bag] mean pooled CV RAE = {mean_rae:.4f}  "
          f"(std across seeds {std_rae:.4f})")
    print(f"[bag] vs nb1014 (nb1001) ref = {NB1014_BAG_REF:.4f}")

    W = np.array(all_w)  # (25, K)
    s_arr = np.array(all_s)  # (25,)
    mean_w = W.mean(axis=0)
    # Re-normalize the simplex projection for numerical safety
    mean_w = np.clip(mean_w, 0.0, 1.0)
    if mean_w.sum() > 0:
        mean_w = mean_w / mean_w.sum()
    else:
        mean_w = np.full(K, 1.0 / K)
    mean_s = float(s_arr.mean())
    print(f"\n[bag] mean w across 25 folds:")
    for j, (name, _, _) in enumerate(CANDIDATES):
        print(f"   {name:24s}: {mean_w[j]:.4f}  "
              f"(std {W[:, j].std():.4f})")
    print(f"[bag] mean s  across 25 folds = {mean_s:.4f}  "
          f"(std {s_arr.std():.4f})")

    # =================================================================
    # Deploy: apply (mean_w, mean_s) with global blend mean as anchor
    # =================================================================
    print("\n" + "-" * 78)
    print("DEPLOY  (mean_w, mean_s applied to all 513)")
    print("-" * 78)
    blend_unb_all = P_unb @ mean_w
    mu_deploy = float(blend_unb_all.mean())
    blend_unb_stretched = mu_deploy + mean_s * (blend_unb_all - mu_deploy)
    in_rae_final = float(rae(y_unb, blend_unb_stretched))
    print(f"   deploy w           = {[round(float(x), 4) for x in mean_w]}")
    print(f"   deploy mu (blend)  = {mu_deploy:.4f}")
    print(f"   deploy s           = {mean_s:.4f}")
    print(f"   in-sample RAE (253) = {in_rae_final:.4f}  "
          "(overfit lower bound)")

    blend_513 = preds_513 @ mean_w
    deploy_513 = (mu_deploy + mean_s * (blend_513 - mu_deploy)).astype(
        np.float32)
    print(f"   te(513) mean/std    = "
          f"{deploy_513.mean():.3f} / {deploy_513.std():.3f}")

    # =================================================================
    # Save
    # =================================================================
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy_513)
    plain = SUBMISSIONS / f"{TAG}_quad_ortho_bag.csv"
    pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te_names,
        "pEC50": deploy_513,
    }).to_csv(plain, index=False)
    print(f"\n[save] te_{TAG}.npy")
    print(f"[save] {plain}")

    delta_vs_ref = mean_rae - NB1014_BAG_REF
    if delta_vs_ref < -0.005:
        verdict = "BEATS_NB1014"
    elif abs(delta_vs_ref) <= 0.005:
        verdict = "TIES_NB1014"
    else:
        verdict = "WORSE_THAN_NB1014"
    print(f"\n[verdict] 4-way bag vs nb1014 ({NB1014_BAG_REF:.4f}): "
          f"delta={delta_vs_ref:+.4f}  -> {verdict}")

    summary = {
        "tag": TAG,
        "candidates": [c[0] for c in CANDIDATES],
        "indiv_in_rae": indiv_rae,
        "pairwise_corr": corr.tolist(),
        "stretch_grid": STRETCH_GRID,
        "n_folds": N_FOLDS,
        "seeds": SEEDS,
        "per_seed_pooled_rae": per_seed_rae,
        "mean_pooled_rae": mean_rae,
        "std_pooled_rae": std_rae,
        "nb1014_bag_reference": NB1014_BAG_REF,
        "delta_vs_nb1014": delta_vs_ref,
        "verdict": verdict,
        "mean_w": mean_w.tolist(),
        "std_w_across_folds": W.std(axis=0).tolist(),
        "mean_s": mean_s,
        "std_s_across_folds": float(s_arr.std()),
        "deploy_mu_blend": mu_deploy,
        "deploy_w": mean_w.tolist(),
        "deploy_s": mean_s,
        "in_sample_rae_overfit_bound": in_rae_final,
        "deploy_te_mean": float(deploy_513.mean()),
        "deploy_te_std": float(deploy_513.std()),
        "seed_results": seed_results,
        "plain_submission": str(plain),
        "wall_sec": round(time.time() - t0, 2),
    }
    with open(DATA_PROCESSED / f"{TAG}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {DATA_PROCESSED / f'{TAG}_summary.json'}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   pool                       = {[c[0] for c in CANDIDATES]}")
    print(f"   per-seed pooled RAE        = "
          f"{[f'{r:.4f}' for r in per_seed_rae]}")
    print(f"   mean (bagged honest CV)    = {mean_rae:.4f}  "
          f"(std {std_rae:.4f})")
    print(f"   nb1014 reference           = {NB1014_BAG_REF:.4f}")
    print(f"   delta                      = {delta_vs_ref:+.4f}")
    print(f"   verdict                    = {verdict}")
    print(f"   deploy mean_w              = "
          f"{[round(float(x), 3) for x in mean_w]}")
    print(f"   deploy mean_s              = {mean_s:.3f}")
    print(f"   wall                       = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("per_seed_pooled_rae", "mean_pooled_rae", "std_pooled_rae",
              "delta_vs_nb1014", "verdict",
              "mean_w", "mean_s",
              "in_sample_rae_overfit_bound", "plain_submission"):
        print(f"  {k}: {res.get(k)}")
