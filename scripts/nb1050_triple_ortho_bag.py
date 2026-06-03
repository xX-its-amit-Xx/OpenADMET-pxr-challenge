"""nb1050 -- Triple-orthogonal multi-seed bag.

Extends the nb1014 (chemprop_aux + nb972_long_train) 2-way bag to a 3-way
mix by adding the cycle-9 Avalon-FP LGBM predictor (nb1042) as a third
anchor.

Hypothesis: Avalon FP gives the lowest Pearson correlation (0.57) to the
two dominant anchors of any candidate we have built; even though its
standalone in_RAE on the 253 is ~0.80 (worse than either chemprop_aux
or nb972), an SLSQP-constrained 3-way blend can pick a small positive
weight on Avalon and use that orthogonal signal to lower the bagged
honest cross-fit RAE relative to the 2-way nb1014.

Procedure (mirrors nb1014, generalised to K=3):
  Pool = [chemprop_aux, nb972_long_train, nb1042_avalon_fp].
  For each seed in SEEDS = {0, 1, 7, 42, 137}:
    5-fold KFold(shuffle=True, random_state=seed) on the 253 unblind:
      For each fold f:
        a. SLSQP for (w0, w1, w2) >= 0, sum = 1 minimising SSE
           of the 3-way blend on the 4 train folds.
        b. Grid-scan s_f in {1.00, 1.05, ..., 2.00} on the train folds
           using train-fold blend mean as mu.
        c. Apply (w_f, s_f, mu_tr) to held-out fold; collect OOF.
    Record pooled cross-fit RAE for this seed.
  Mean of 5 pooled RAEs = bagged honest cross-fit RAE.
  Mean of (w0, w1, w2, s) across 25 folds = deploy point.

Deploy: apply (mean_w, mean_s) to the 513 te files.

Outputs:
  data/processed/te_nb1050.npy
  data/processed/nb1050_summary.json
  submissions/nb1050_triple_ortho_bag.csv
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

TAG = "nb1050"
CANDIDATES = ["chemprop_aux", "nb972_long_train", "nb1042"]
STRETCH_GRID = np.round(np.arange(1.00, 2.001, 0.05), 2).tolist()
N_FOLDS = 5
SEEDS = [0, 1, 7, 42, 137]
NB1014_BAG_REF = 0.5994  # nb1014 / nb1001 seed=42 reference; nb1014 bag close


def load_te(name: str, te_names: np.ndarray) -> np.ndarray:
    npy = DATA_PROCESSED / f"te_{name}.npy"
    if npy.exists():
        return np.load(npy).astype(np.float64)
    sub_path = SUBMISSIONS / f"{name}.csv"
    if not sub_path.exists():
        raise FileNotFoundError(
            f"Neither {npy} nor {sub_path} exists for candidate {name}")
    sub = pd.read_csv(sub_path)
    assert (sub["Molecule Name"].values == te_names).all(), (
        f"{name}: submission row order does not match test order")
    return sub["pEC50"].values.astype(np.float64)


def slsqp_weights(P: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Fit w in simplex (w_i >= 0, sum=1) minimising SSE of P @ w vs y.

    Returns length-k weight vector.
    """
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * k
    w0 = np.full(k, 1.0 / k)
    res = minimize(
        lambda w: float(np.sum((P @ w - y) ** 2)),
        w0,
        method="SLSQP", bounds=bnds, constraints=cons,
        options={"ftol": 1e-10, "maxiter": 1000},
    )
    w = np.clip(res.x, 0.0, 1.0)
    s = w.sum()
    if s <= 0:
        return np.full(k, 1.0 / k)
    return w / s


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


def run_one_seed(P_unb: np.ndarray, y_unb: np.ndarray,
                 seed: int) -> dict:
    """Run the 5-fold protocol for one KFold seed (K=3)."""
    n_unb = len(y_unb)
    k_dim = P_unb.shape[1]
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n_unb, np.nan)
    folds = []
    for fold_idx, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n_unb))):
        w_f = slsqp_weights(P_unb[tr_loc], y_unb[tr_loc], k_dim)
        blend_tr = P_unb[tr_loc] @ w_f
        mu_tr = float(blend_tr.mean())
        s_f, rae_tr = best_stretch_on(blend_tr, y_unb[tr_loc], mu_tr)
        blend_va = P_unb[va_loc] @ w_f
        oof[va_loc] = mu_tr + s_f * (blend_va - mu_tr)
        rae_va = float(rae(y_unb[va_loc], oof[va_loc]))
        folds.append({"fold": fold_idx,
                      "w": w_f.tolist(),
                      "s": s_f, "mu_tr": mu_tr,
                      "train_rae": rae_tr, "val_rae": rae_va,
                      "n_va": int(len(va_loc))})
    pooled = float(rae(y_unb, oof))
    return {"seed": seed, "folds": folds, "pooled_rae": pooled, "oof": oof}


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- triple-orthogonal multi-seed bag "
          f"(seeds={SEEDS}, K={len(CANDIDATES)})")
    print("=" * 78)

    # ---- Load 513 test ----
    te = load_test()
    te_names = te["name"].values
    n_te = len(te_names)
    preds_513 = np.column_stack([load_te(c, te_names) for c in CANDIDATES])
    print(f"[load] preds_513 shape = {preds_513.shape}")

    # ---- Load 253 unblind ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    P_unb = preds_513[unb_idx]
    n_unb = len(y_unb)
    print(f"[load] P_unb shape = {P_unb.shape}, y shape = {y_unb.shape}")

    # ---- Individual in_RAE sanity ----
    print("\n[indiv] in_RAE on 253 unblind:")
    indiv_rae = {}
    for j, c in enumerate(CANDIDATES):
        r = float(rae(y_unb, P_unb[:, j]))
        indiv_rae[c] = r
        print(f"   {c:30s}: {r:.4f}")

    # ---- Pairwise Pearson on 253 ----
    print("\n[indiv] pairwise Pearson (253 unblind):")
    corrs = {}
    for i in range(len(CANDIDATES)):
        for j in range(i + 1, len(CANDIDATES)):
            r = float(np.corrcoef(P_unb[:, i], P_unb[:, j])[0, 1])
            corrs[f"{CANDIDATES[i]}__{CANDIDATES[j]}"] = r
            print(f"   {CANDIDATES[i]:24s} x {CANDIDATES[j]:24s}: {r:+.4f}")

    # =================================================================
    # Run protocol for each seed
    # =================================================================
    print("\n" + "-" * 78)
    print(f"MULTI-SEED CROSS-FIT  (N_FOLDS={N_FOLDS}, K={len(CANDIDATES)}, "
          f"stretch grid {STRETCH_GRID[0]}..{STRETCH_GRID[-1]})")
    print("-" * 78)

    seed_results = []
    per_seed_rae = []
    all_w = []
    all_s = []
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
    print(f"[bag] vs nb1014/nb1001 ref = {NB1014_BAG_REF:.4f}")

    all_w_arr = np.array(all_w)  # (25, K)
    mean_w = all_w_arr.mean(axis=0)
    std_w = all_w_arr.std(axis=0)
    mean_s = float(np.mean(all_s))
    print(f"\n[bag] mean weights across 25 folds:")
    for j, c in enumerate(CANDIDATES):
        print(f"   w[{j}] {c:30s} = {mean_w[j]:.4f}  (std {std_w[j]:.4f})")
    print(f"[bag] mean s across 25 folds = {mean_s:.4f}  "
          f"(std {np.std(all_s):.4f})")

    # =================================================================
    # Deploy: apply (mean_w, mean_s) with global blend mean as anchor
    # =================================================================
    print("\n" + "-" * 78)
    print("DEPLOY  (mean weights, mean s applied to all 513)")
    print("-" * 78)
    blend_unb_all = P_unb @ mean_w
    mu_deploy = float(blend_unb_all.mean())
    blend_unb_stretched = mu_deploy + mean_s * (blend_unb_all - mu_deploy)
    in_rae_final = float(rae(y_unb, blend_unb_stretched))
    print(f"   deploy w            = {mean_w.tolist()}")
    print(f"   deploy mu (blend)   = {mu_deploy:.4f}")
    print(f"   deploy s            = {mean_s:.4f}")
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
    plain = SUBMISSIONS / f"{TAG}_triple_ortho_bag.csv"
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
    print(f"\n[verdict] bag vs nb1014/nb1001 (0.5994): "
          f"delta={delta_vs_ref:+.4f}  -> {verdict}")

    summary = {
        "tag": TAG,
        "candidates": CANDIDATES,
        "indiv_in_rae": indiv_rae,
        "pairwise_pearson": corrs,
        "stretch_grid": STRETCH_GRID,
        "n_folds": N_FOLDS,
        "seeds": SEEDS,
        "per_seed_pooled_rae": per_seed_rae,
        "mean_pooled_rae": mean_rae,
        "std_pooled_rae": std_rae,
        "nb1014_bag_ref": NB1014_BAG_REF,
        "delta_vs_nb1014": delta_vs_ref,
        "verdict": verdict,
        "mean_weights": mean_w.tolist(),
        "std_weights_across_folds": std_w.tolist(),
        "mean_s": mean_s,
        "std_s_across_folds": float(np.std(all_s)),
        "deploy_mu_blend": mu_deploy,
        "deploy_weights": mean_w.tolist(),
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
    print(f"   pool                       = {CANDIDATES}")
    print(f"   per-seed pooled RAE        = "
          f"{[f'{r:.4f}' for r in per_seed_rae]}")
    print(f"   mean (bagged honest CV)    = {mean_rae:.4f}  "
          f"(std {std_rae:.4f})")
    print(f"   nb1014/nb1001 reference    = {NB1014_BAG_REF:.4f}")
    print(f"   delta                      = {delta_vs_ref:+.4f}")
    print(f"   verdict                    = {verdict}")
    print(f"   deploy weights             = "
          f"{[round(float(w), 3) for w in mean_w]}")
    print(f"   deploy s                   = {mean_s:.3f}")
    print(f"   wall                       = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("per_seed_pooled_rae", "mean_pooled_rae", "std_pooled_rae",
              "delta_vs_nb1014", "verdict",
              "mean_weights", "mean_s",
              "in_sample_rae_overfit_bound", "plain_submission"):
        print(f"  {k}: {res.get(k)}")
