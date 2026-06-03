"""nb1033 -- StratifiedKFold variant of nb1014 (chemprop_aux + nb972 + stretch).

Same protocol as nb1014 (pool=[chemprop_aux, nb972_long_train], SLSQP w0 on
train folds, stretch grid s in {1.00..2.00} step 0.05, deploy = mean across
folds) BUT replace random KFold with StratifiedKFold over 5 quantile bins
of pec50 on the 253 unblind. 5 seeds.

Hypothesis: stratified folds may capture failure modes more evenly
(every fold sees the full pec50 range, especially the high-truth tail
that is currently variance-compressed) reducing per-fold variance of
the pooled estimate.

Outputs:
  data/processed/te_nb1033.npy
  data/processed/nb1033_summary.json
  submissions/nb1033_stratified_kf_bag.csv
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
from sklearn.model_selection import StratifiedKFold

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1033"
CANDIDATES = ["chemprop_aux", "nb972_long_train"]
STRETCH_GRID = np.round(np.arange(1.00, 2.001, 0.05), 2).tolist()
N_FOLDS = 5
N_BINS = 5
SEEDS = [0, 1, 7, 42, 137]
NB1014_REFERENCE = 0.5930


def load_te(name: str, te_names: np.ndarray) -> np.ndarray:
    npy = DATA_PROCESSED / f"te_{name}.npy"
    if npy.exists():
        return np.load(npy).astype(np.float64)
    sub = pd.read_csv(SUBMISSIONS / f"{name}.csv")
    assert (sub["Molecule Name"].values == te_names).all(), (
        f"{name}: submission row order does not match test order")
    return sub["pEC50"].values.astype(np.float64)


def slsqp_w0(p0: np.ndarray, p1: np.ndarray, y: np.ndarray) -> float:
    P = np.column_stack([p0, p1])
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0), (0.0, 1.0)]
    res = minimize(
        lambda w: float(np.sum((P @ w - y) ** 2)),
        np.array([0.5, 0.5]),
        method="SLSQP", bounds=bnds, constraints=cons,
        options={"ftol": 1e-10, "maxiter": 500},
    )
    return float(res.x[0])


def best_stretch_on(blend_train: np.ndarray, y_train: np.ndarray,
                    mu: float) -> tuple[float, float]:
    best_s, best_r = 1.0, float("inf")
    for s in STRETCH_GRID:
        stretched = mu + s * (blend_train - mu)
        r = float(rae(y_train, stretched))
        if r < best_r:
            best_r = r
            best_s = float(s)
    return best_s, best_r


def make_bins(y: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Quantile bins of y for stratification."""
    qs = np.quantile(y, np.linspace(0.0, 1.0, n_bins + 1))
    qs[0] -= 1e-9
    qs[-1] += 1e-9
    bins = np.digitize(y, qs[1:-1], right=False)
    return bins.astype(int)


def run_one_seed(P_unb: np.ndarray, y_unb: np.ndarray, bins: np.ndarray,
                 seed: int) -> dict:
    n_unb = len(y_unb)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n_unb, np.nan)
    folds = []
    for k, (tr_loc, va_loc) in enumerate(skf.split(np.arange(n_unb), bins)):
        w0_f = slsqp_w0(P_unb[tr_loc, 0], P_unb[tr_loc, 1], y_unb[tr_loc])
        blend_tr = w0_f * P_unb[tr_loc, 0] + (1.0 - w0_f) * P_unb[tr_loc, 1]
        mu_tr = float(blend_tr.mean())
        s_f, rae_tr = best_stretch_on(blend_tr, y_unb[tr_loc], mu_tr)
        blend_va = w0_f * P_unb[va_loc, 0] + (1.0 - w0_f) * P_unb[va_loc, 1]
        oof[va_loc] = mu_tr + s_f * (blend_va - mu_tr)
        rae_va = float(rae(y_unb[va_loc], oof[va_loc]))
        folds.append({"fold": k, "w0": w0_f, "s": s_f, "mu_tr": mu_tr,
                      "train_rae": rae_tr, "val_rae": rae_va,
                      "n_va": int(len(va_loc))})
    pooled = float(rae(y_unb, oof))
    return {"seed": seed, "folds": folds, "pooled_rae": pooled, "oof": oof}


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- StratifiedKFold (pec50 quantile bins, {N_BINS} bins) "
          f"variant of nb1014 (seeds={SEEDS})")
    print("=" * 78)

    te = load_test()
    te_names = te["name"].values
    preds_513 = np.column_stack([load_te(c, te_names) for c in CANDIDATES])
    print(f"[load] preds_513 shape = {preds_513.shape}")

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    P_unb = preds_513[unb_idx]
    n_unb = len(y_unb)
    print(f"[load] P_unb shape = {P_unb.shape}, y shape = {y_unb.shape}")

    bins = make_bins(y_unb, N_BINS)
    uniq, cnts = np.unique(bins, return_counts=True)
    print(f"[strat] {N_BINS} pec50 quantile bins -> counts {dict(zip(uniq.tolist(), cnts.tolist()))}")

    print("\n[indiv] in_RAE on 253 unblind:")
    indiv_rae = {}
    for j, c in enumerate(CANDIDATES):
        r = float(rae(y_unb, P_unb[:, j]))
        indiv_rae[c] = r
        print(f"   {c:30s}: {r:.4f}")

    print("\n" + "-" * 78)
    print(f"STRATIFIED CROSS-FIT  (N_FOLDS={N_FOLDS}, N_BINS={N_BINS}, "
          f"stretch grid {STRETCH_GRID[0]}..{STRETCH_GRID[-1]})")
    print("-" * 78)

    seed_results = []
    per_seed_rae = []
    all_w0, all_s = [], []
    for seed in SEEDS:
        res = run_one_seed(P_unb, y_unb, bins, seed)
        seed_results.append({"seed": seed, "pooled_rae": res["pooled_rae"],
                             "folds": res["folds"]})
        per_seed_rae.append(res["pooled_rae"])
        for f in res["folds"]:
            all_w0.append(f["w0"])
            all_s.append(f["s"])
        fold_vals = [round(f["val_rae"], 3) for f in res["folds"]]
        print(f"   seed {seed:>3d}: pooled_RAE = {res['pooled_rae']:.4f}  "
              f"(folds: {fold_vals})")

    mean_rae = float(np.mean(per_seed_rae))
    std_rae = float(np.std(per_seed_rae))
    print(f"\n[bag] mean pooled CV RAE = {mean_rae:.4f}  "
          f"(std across seeds {std_rae:.4f})")
    print(f"[bag] vs nb1014 (random KFold mean) = {NB1014_REFERENCE:.4f}")

    mean_w0 = float(np.mean(all_w0))
    mean_s = float(np.mean(all_s))
    print(f"\n[bag] mean w0 across 25 folds = {mean_w0:.4f}  "
          f"(std {np.std(all_w0):.4f})")
    print(f"[bag] mean s  across 25 folds = {mean_s:.4f}  "
          f"(std {np.std(all_s):.4f})")

    print("\n" + "-" * 78)
    print("DEPLOY  (mean_w0, mean_s applied to all 513)")
    print("-" * 78)
    blend_unb_all = mean_w0 * P_unb[:, 0] + (1.0 - mean_w0) * P_unb[:, 1]
    mu_deploy = float(blend_unb_all.mean())
    blend_unb_stretched = mu_deploy + mean_s * (blend_unb_all - mu_deploy)
    in_rae_final = float(rae(y_unb, blend_unb_stretched))
    print(f"   deploy w0           = {mean_w0:.4f}  "
          f"(chemprop_aux weight; w1={1.0 - mean_w0:.4f} for nb972)")
    print(f"   deploy mu (blend)   = {mu_deploy:.4f}")
    print(f"   deploy s            = {mean_s:.4f}")
    print(f"   in-sample RAE (253) = {in_rae_final:.4f}  "
          "(overfit lower bound)")

    blend_513 = mean_w0 * preds_513[:, 0] + (1.0 - mean_w0) * preds_513[:, 1]
    deploy_513 = (mu_deploy + mean_s * (blend_513 - mu_deploy)).astype(np.float32)
    print(f"   te(513) mean/std    = {deploy_513.mean():.3f} / "
          f"{deploy_513.std():.3f}")

    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy_513)
    plain = SUBMISSIONS / f"{TAG}_stratified_kf_bag.csv"
    pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te_names,
        "pEC50": deploy_513,
    }).to_csv(plain, index=False)
    print(f"\n[save] te_{TAG}.npy")
    print(f"[save] {plain}")

    delta_vs_nb1014 = mean_rae - NB1014_REFERENCE
    if delta_vs_nb1014 < -0.005:
        verdict = "BEATS_NB1014"
    elif abs(delta_vs_nb1014) <= 0.005:
        verdict = "TIES_NB1014"
    else:
        verdict = "WORSE_THAN_NB1014"
    print(f"\n[verdict] stratified vs nb1014 random ({NB1014_REFERENCE:.4f}): "
          f"delta={delta_vs_nb1014:+.4f}  -> {verdict}")

    summary = {
        "tag": TAG,
        "candidates": CANDIDATES,
        "indiv_in_rae": indiv_rae,
        "stretch_grid": STRETCH_GRID,
        "n_folds": N_FOLDS,
        "n_bins": N_BINS,
        "seeds": SEEDS,
        "bin_counts": {int(k): int(v) for k, v in zip(uniq.tolist(), cnts.tolist())},
        "per_seed_pooled_rae": per_seed_rae,
        "mean_pooled_rae": mean_rae,
        "std_pooled_rae": std_rae,
        "nb1014_reference": NB1014_REFERENCE,
        "delta_vs_nb1014": delta_vs_nb1014,
        "verdict": verdict,
        "mean_w0_chemprop_aux": mean_w0,
        "mean_w1_nb972": float(1.0 - mean_w0),
        "mean_s": mean_s,
        "std_w0_across_folds": float(np.std(all_w0)),
        "std_s_across_folds": float(np.std(all_s)),
        "deploy_mu_blend": mu_deploy,
        "deploy_w0": mean_w0,
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
    print(f"   strategy                   = StratifiedKFold on pec50 quantile bins")
    print(f"   per-seed pooled RAE        = "
          f"{[f'{r:.4f}' for r in per_seed_rae]}")
    print(f"   mean (bagged honest CV)    = {mean_rae:.4f}  "
          f"(std {std_rae:.4f})")
    print(f"   nb1014 reference           = {NB1014_REFERENCE:.4f}")
    print(f"   delta                      = {delta_vs_nb1014:+.4f}")
    print(f"   verdict                    = {verdict}")
    print(f"   deploy (mean_w0, mean_s)   = "
          f"({mean_w0:.3f}, {mean_s:.3f})")
    print(f"   wall                       = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("per_seed_pooled_rae", "mean_pooled_rae", "std_pooled_rae",
              "delta_vs_nb1014", "verdict",
              "mean_w0_chemprop_aux", "mean_s",
              "in_sample_rae_overfit_bound", "plain_submission"):
        print(f"  {k}: {res.get(k)}")
