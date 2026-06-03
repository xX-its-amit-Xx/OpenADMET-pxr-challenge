"""nb1110 -- Weak-orthogonal aggregation + 3-way bag with nb1014 backbone.

Hypothesis: averaging 5 weak-but-orthogonal predictors creates a moderately-
strong yet still-orthogonal compound predictor that earns meaningful SLSQP
weight in the chemprop_aux + nb972_long_train pool. Individually, each weak
predictor's correlation to nb972 is already lower than the strong members,
but their in_RAE is too high (0.74-0.90) for SLSQP to grant weight. By
z-score-averaging them and de-standardizing onto the nb972 scale, the
combined predictor should sit at moderate in_RAE while preserving the low-
correlation property.

Pool (5 weak orthogonal candidates):
  nb1030  -- Mordred standalone LGBM
  nb1042  -- Avalon fingerprint LGBM
  nb1101  -- Linear Ridge
  nb1103  -- XGB extreme hyperparams
  nb1104  -- KNN k=15

Procedure:
  1. Load 513-row te files for the 5 weak candidates and nb972.
  2. z-score each (per-vector mean/std), average, de-standardize using
     (mean, std) of te_nb972_long_train -> te_weak_ens.
  3. Compute Pearson(te_weak_ens, te_nb972). Should be lower than any
     individual weak vs nb972, validating "averaging preserves orthogonality".
  4. 3-way pool: [chemprop_aux, nb972_long_train, te_weak_ens].
     Apply nb1014 backbone: 5 KFold seeds x 5 folds = 25 SLSQP fits,
     per-fold mu/s grid-stretch, OOF on the 253 unblind.
  5. Pooled honest cross-fit RAE; deploy (mean_w, mean_s).

Reference: nb1014 seed=42 pooled CV RAE = 0.5994 (chemprop_aux + nb972 only).
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
from sklearn.model_selection import KFold

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1110_weak_ortho_aggregation"
WEAK_CANDIDATES = ["nb1030", "nb1042", "nb1101", "nb1103", "nb1104"]
STRONG_REF = "nb972_long_train"
CHEMPROP = "chemprop_aux"
STRETCH_GRID = np.round(np.arange(1.00, 2.001, 0.05), 2).tolist()
N_FOLDS = 5
SEEDS = [0, 1, 7, 42, 137]
NB1014_REFERENCE = 0.5994


def load_te(name: str, te_names: np.ndarray) -> np.ndarray:
    npy = DATA_PROCESSED / f"te_{name}.npy"
    if npy.exists():
        return np.load(npy).astype(np.float64)
    sub = pd.read_csv(SUBMISSIONS / f"{name}.csv")
    return sub["pEC50"].values.astype(np.float64)


def slsqp_simplex(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    K = P.shape[1]
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * K
    res = minimize(
        lambda w: float(np.sum((P @ w - y) ** 2)),
        np.full(K, 1.0 / K),
        method="SLSQP", bounds=bnds, constraints=cons,
        options={"ftol": 1e-10, "maxiter": 500},
    )
    return res.x


def best_stretch_on(blend_train, y_train, mu):
    best_s, best_r = 1.0, float("inf")
    for s in STRETCH_GRID:
        stretched = mu + s * (blend_train - mu)
        r = float(rae(y_train, stretched))
        if r < best_r:
            best_r, best_s = r, float(s)
    return best_s, best_r


def run_one_seed(P_unb, y_unb, seed):
    n_unb = len(y_unb)
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
        folds.append({"fold": k, "w": w_f.tolist(), "s": s_f, "mu_tr": mu_tr,
                      "train_rae": rae_tr,
                      "val_rae": float(rae(y_unb[va_loc], oof[va_loc])),
                      "n_va": int(len(va_loc))})
    pooled = float(rae(y_unb, oof))
    return {"seed": seed, "folds": folds, "pooled_rae": pooled, "oof": oof}


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- weak-orthogonal aggregation -> 3-way bag")
    print("=" * 78)

    te = load_test()
    te_names = te["name"].values
    n_te = len(te_names)

    # ---- Load weak candidates and strong reference ----
    weak_513 = np.column_stack([load_te(c, te_names) for c in WEAK_CANDIDATES])
    te_nb972 = load_te(STRONG_REF, te_names)
    te_chemprop = load_te(CHEMPROP, te_names)
    print(f"[load] weak_513 shape = {weak_513.shape}")
    print(f"[load] te_nb972 shape = {te_nb972.shape}")

    # ---- Load 253 unblind ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    print(f"[load] unb_idx n = {len(unb_idx)}, y_unb n = {len(y_unb)}")

    # ---- Individual diagnostics on the 253 ----
    print("\n[indiv] in_RAE on 253 unblind:")
    indiv_rae = {}
    indiv_pearson_vs_nb972 = {}
    for j, c in enumerate(WEAK_CANDIDATES):
        r = float(rae(y_unb, weak_513[unb_idx, j]))
        p = float(np.corrcoef(weak_513[:, j], te_nb972)[0, 1])
        indiv_rae[c] = r
        indiv_pearson_vs_nb972[c] = p
        print(f"   {c:25s}: in_RAE = {r:.4f}   Pearson(nb972) = {p:.4f}")
    chemprop_in_rae = float(rae(y_unb, te_chemprop[unb_idx]))
    nb972_in_rae = float(rae(y_unb, te_nb972[unb_idx]))
    print(f"   {CHEMPROP:25s}: in_RAE = {chemprop_in_rae:.4f}")
    print(f"   {STRONG_REF:25s}: in_RAE = {nb972_in_rae:.4f}")

    # =================================================================
    # Step 2: z-score average -> de-standardize onto nb972 scale
    # =================================================================
    print("\n" + "-" * 78)
    print("WEAK-ORTHO AGGREGATION (z-score average, de-standardize to nb972)")
    print("-" * 78)
    zs = np.zeros_like(weak_513)
    for j in range(weak_513.shape[1]):
        v = weak_513[:, j]
        zs[:, j] = (v - v.mean()) / (v.std() + 1e-12)
    z_mean = zs.mean(axis=1)
    nb972_mu = float(te_nb972.mean())
    nb972_sd = float(te_nb972.std())
    te_weak_ens = z_mean * nb972_sd + nb972_mu
    print(f"   z_mean stats        : mean={z_mean.mean():.3f}, "
          f"std={z_mean.std():.3f}")
    print(f"   nb972 (mu,sd)       : ({nb972_mu:.3f}, {nb972_sd:.3f})")
    print(f"   te_weak_ens stats   : mean={te_weak_ens.mean():.3f}, "
          f"std={te_weak_ens.std():.3f}")

    weak_ens_in_rae = float(rae(y_unb, te_weak_ens[unb_idx]))
    weak_ens_pearson_nb972 = float(np.corrcoef(te_weak_ens, te_nb972)[0, 1])
    print(f"\n   te_weak_ens in_RAE          = {weak_ens_in_rae:.4f}")
    print(f"   Pearson(te_weak_ens, nb972) = {weak_ens_pearson_nb972:.4f}")
    min_indiv_pearson = min(indiv_pearson_vs_nb972.values())
    print(f"   min individual Pearson      = {min_indiv_pearson:.4f}  "
          f"({'PASS' if weak_ens_pearson_nb972 < min_indiv_pearson else 'FAIL'} "
          f"vs aggregation hypothesis)")

    np.save(DATA_PROCESSED / "te_weak_ens_nb1110.npy",
            te_weak_ens.astype(np.float32))
    print("   [save] te_weak_ens_nb1110.npy")

    # =================================================================
    # Step 4: 3-way bag with nb1014 backbone
    # =================================================================
    print("\n" + "-" * 78)
    print(f"3-WAY MULTI-SEED BAG  N_FOLDS={N_FOLDS}  seeds={SEEDS}")
    print(f"   pool = [{CHEMPROP}, {STRONG_REF}, te_weak_ens]")
    print("-" * 78)

    preds_513 = np.column_stack([te_chemprop, te_nb972, te_weak_ens])
    P_unb = preds_513[unb_idx]
    pool_names = [CHEMPROP, STRONG_REF, "te_weak_ens"]

    # 3-way pool Pearson
    print("\n[pool-pearson]")
    for i in range(3):
        for j in range(i + 1, 3):
            p = np.corrcoef(preds_513[:, i], preds_513[:, j])[0, 1]
            print(f"   {pool_names[i]:20s} vs {pool_names[j]:20s}: {p:.4f}")

    per_seed_rae = []
    all_w = []
    all_s = []
    seed_results = []
    for seed in SEEDS:
        res = run_one_seed(P_unb, y_unb, seed)
        per_seed_rae.append(res["pooled_rae"])
        for f in res["folds"]:
            all_w.append(f["w"])
            all_s.append(f["s"])
        fold_vals = [round(f["val_rae"], 3) for f in res["folds"]]
        print(f"   seed {seed:>3d}: pooled_RAE = {res['pooled_rae']:.4f}  "
              f"(folds: {fold_vals})")
        seed_results.append({"seed": seed,
                             "pooled_rae": res["pooled_rae"],
                             "folds": res["folds"]})

    mean_rae = float(np.mean(per_seed_rae))
    std_rae = float(np.std(per_seed_rae))
    mean_w = np.mean(np.array(all_w), axis=0)
    mean_s = float(np.mean(all_s))
    print(f"\n[bag] mean pooled CV RAE = {mean_rae:.4f}  (std {std_rae:.4f})")
    print(f"[bag] mean w  = {mean_w.tolist()}")
    print(f"[bag] mean s  = {mean_s:.4f}")
    weak_weight = float(mean_w[2])
    print(f"[bag] te_weak_ens weight = {weak_weight:.4f}  "
          f"({'MEANINGFUL' if weak_weight >= 0.05 else 'NEGLIGIBLE'} >= 0.05?)")

    # =================================================================
    # Deploy
    # =================================================================
    blend_unb = P_unb @ mean_w
    mu_deploy = float(blend_unb.mean())
    in_sample = float(rae(y_unb,
                          mu_deploy + mean_s * (blend_unb - mu_deploy)))
    print(f"\n   in-sample RAE (253) = {in_sample:.4f}  "
          "(overfit lower bound)")

    blend_513 = preds_513 @ mean_w
    deploy_513 = (mu_deploy + mean_s * (blend_513 - mu_deploy)).astype(
        np.float32)
    print(f"   te(513) mean/std = "
          f"{deploy_513.mean():.3f}/{deploy_513.std():.3f}")

    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy_513)
    plain = SUBMISSIONS / f"{TAG}.csv"
    pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te_names,
        "pEC50": deploy_513,
    }).to_csv(plain, index=False)
    print(f"\n[save] te_{TAG}.npy")
    print(f"[save] {plain}")

    delta = mean_rae - NB1014_REFERENCE
    if delta < -0.005:
        verdict = "BEATS_NB1014"
    elif abs(delta) <= 0.005:
        verdict = "TIES_NB1014"
    else:
        verdict = "WORSE_THAN_NB1014"
    print(f"\n[verdict] vs nb1014 (0.5994): delta={delta:+.4f}  -> {verdict}")

    summary = {
        "tag": TAG,
        "weak_candidates": WEAK_CANDIDATES,
        "pool": pool_names,
        "indiv_in_rae_weak": indiv_rae,
        "indiv_pearson_vs_nb972": indiv_pearson_vs_nb972,
        "chemprop_aux_in_rae": chemprop_in_rae,
        "nb972_in_rae": nb972_in_rae,
        "weak_ens_in_rae": weak_ens_in_rae,
        "weak_ens_pearson_nb972": weak_ens_pearson_nb972,
        "min_indiv_pearson_nb972": min_indiv_pearson,
        "aggregation_lowers_pearson": bool(
            weak_ens_pearson_nb972 < min_indiv_pearson),
        "per_seed_pooled_rae": per_seed_rae,
        "mean_pooled_rae": mean_rae,
        "std_pooled_rae": std_rae,
        "nb1014_reference": NB1014_REFERENCE,
        "delta_vs_nb1014": delta,
        "verdict": verdict,
        "mean_w": mean_w.tolist(),
        "mean_w_chemprop_aux": float(mean_w[0]),
        "mean_w_nb972": float(mean_w[1]),
        "mean_w_weak_ens": weak_weight,
        "mean_s": mean_s,
        "deploy_mu_blend": mu_deploy,
        "in_sample_rae": in_sample,
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
    print(f"   weak ensemble in_RAE       = {weak_ens_in_rae:.4f}")
    print(f"   weak ensemble P(nb972)     = {weak_ens_pearson_nb972:.4f}  "
          f"(min indiv = {min_indiv_pearson:.4f})")
    print(f"   3-way pool weights         = {mean_w.tolist()}")
    print(f"   per-seed pooled RAE        = "
          f"{[f'{r:.4f}' for r in per_seed_rae]}")
    print(f"   mean pooled CV RAE         = {mean_rae:.4f}  "
          f"(std {std_rae:.4f})")
    print(f"   nb1014 reference           = {NB1014_REFERENCE:.4f}")
    print(f"   delta                      = {delta:+.4f}")
    print(f"   verdict                    = {verdict}")
    print(f"   in-sample (overfit bound)  = {in_sample:.4f}")
    print(f"   wall                       = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("weak_ens_in_rae", "weak_ens_pearson_nb972",
              "min_indiv_pearson_nb972", "aggregation_lowers_pearson",
              "mean_w", "per_seed_pooled_rae", "mean_pooled_rae",
              "delta_vs_nb1014", "verdict", "in_sample_rae",
              "plain_submission"):
        print(f"  {k}: {res.get(k)}")
