"""nb1014 variant -- multi-seed bag with chemprop_aux + nb972 + nb932 (CatBoost).

Extends nb1014 protocol to K=3 candidates:
  chemprop_aux (in_RAE 0.6216)
  nb972_long_train (in_RAE 0.6898)
  nb932 (CatBoost, cycle 4; in_RAE 0.6985, Pearson(nb972)=0.926 - orthogonal)
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

TAG = "nb1014_cycle4_nb932"
CANDIDATES = ["chemprop_aux", "nb972_long_train", "nb932"]
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
    print(f"{TAG} -- 3-way bag: {CANDIDATES}")
    print("=" * 78)

    te = load_test()
    te_names = te["name"].values
    preds_513 = np.column_stack([load_te(c, te_names) for c in CANDIDATES])
    print(f"[load] preds_513 shape = {preds_513.shape}")

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    P_unb = preds_513[unb_idx]
    print(f"[load] P_unb shape = {P_unb.shape}, y shape = {y_unb.shape}")

    print("\n[indiv] in_RAE on 253 unblind:")
    indiv_rae = {}
    for j, c in enumerate(CANDIDATES):
        r = float(rae(y_unb, P_unb[:, j]))
        indiv_rae[c] = r
        print(f"   {c:30s}: {r:.4f}")

    # Pearson among pool
    print("\n[pearson] pool correlations:")
    for i in range(len(CANDIDATES)):
        for j in range(i + 1, len(CANDIDATES)):
            p = np.corrcoef(preds_513[:, i], preds_513[:, j])[0, 1]
            print(f"   {CANDIDATES[i]:25s} vs {CANDIDATES[j]:25s}: {p:.4f}")

    print("\n" + "-" * 78)
    print(f"MULTI-SEED CROSS-FIT  N_FOLDS={N_FOLDS}")
    print("-" * 78)

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
        seed_results.append({"seed": seed, "pooled_rae": res["pooled_rae"],
                             "folds": res["folds"]})

    mean_rae = float(np.mean(per_seed_rae))
    std_rae = float(np.std(per_seed_rae))
    mean_w = np.mean(np.array(all_w), axis=0)
    mean_s = float(np.mean(all_s))
    print(f"\n[bag] mean pooled CV RAE = {mean_rae:.4f}  (std {std_rae:.4f})")
    print(f"[bag] mean w  = {mean_w.tolist()}")
    print(f"[bag] mean s  = {mean_s:.4f}")

    blend_unb = P_unb @ mean_w
    mu_deploy = float(blend_unb.mean())
    in_sample = float(rae(y_unb, mu_deploy + mean_s * (blend_unb - mu_deploy)))
    print(f"   in-sample RAE (253) = {in_sample:.4f}")

    blend_513 = preds_513 @ mean_w
    deploy_513 = (mu_deploy + mean_s * (blend_513 - mu_deploy)).astype(np.float32)
    print(f"   te(513) mean/std = {deploy_513.mean():.3f}/{deploy_513.std():.3f}")

    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy_513)
    plain = SUBMISSIONS / f"{TAG}.csv"
    pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te_names,
        "pEC50": deploy_513,
    }).to_csv(plain, index=False)
    print(f"[save] te_{TAG}.npy")
    print(f"[save] {plain}")

    delta = mean_rae - NB1014_REFERENCE
    verdict = ("BEATS_NB1014" if delta < -0.005
               else "TIES_NB1014" if abs(delta) <= 0.005
               else "WORSE_THAN_NB1014")
    print(f"\n[verdict] vs nb1014 (0.5994): delta={delta:+.4f}  -> {verdict}")

    summary = {
        "tag": TAG,
        "candidates": CANDIDATES,
        "indiv_in_rae": indiv_rae,
        "per_seed_pooled_rae": per_seed_rae,
        "mean_pooled_rae": mean_rae,
        "std_pooled_rae": std_rae,
        "nb1014_reference": NB1014_REFERENCE,
        "delta_vs_nb1014": delta,
        "verdict": verdict,
        "mean_w": mean_w.tolist(),
        "mean_s": mean_s,
        "deploy_mu_blend": mu_deploy,
        "in_sample_rae": in_sample,
        "deploy_te_mean": float(deploy_513.mean()),
        "deploy_te_std": float(deploy_513.std()),
        "plain_submission": str(plain),
        "wall_sec": round(time.time() - t0, 2),
    }
    with open(DATA_PROCESSED / f"{TAG}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {DATA_PROCESSED / f'{TAG}_summary.json'}")
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("per_seed_pooled_rae", "mean_pooled_rae", "delta_vs_nb1014",
              "verdict", "mean_w", "mean_s", "in_sample_rae",
              "plain_submission"):
        print(f"  {k}: {res.get(k)}")
