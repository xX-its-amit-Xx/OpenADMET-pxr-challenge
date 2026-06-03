"""nb1021 -- Multi-seed bag of (chemprop_aux + stretch ONLY), no nb972 blend.

Companion to nb1014 (which bags the chemprop_aux + nb972 + stretch protocol).
This script bags the nb1011 protocol: stretch-only on chemprop_aux, no blend.

nb1011 single-seed = 0.6109 (stretch-only on chemprop_aux).
nb1014 5-seed bag of (chemprop+nb972+stretch) = 0.5930.

Hypothesis: bagging alone may NOT save chemprop-only -- the blend is what
makes the gain large. If nb1021 mean ~= 0.6109, the blend is the lever and
bagging is a small noise-reduction add-on. If nb1021 mean << 0.6109, bagging
is genuinely worth doing even on a single anchor.

Procedure (5-seed bag of the nb1011 protocol):
  Pool = [chemprop_aux] on the 253 unblind (SINGLE candidate, no SLSQP).
  For each seed in SEEDS = {0, 1, 7, 42, 137}:
    5-fold KFold(shuffle=True, random_state=seed) on the 253:
      For each fold f:
        a. mu_tr = mean of chemprop_aux predictions on the 4 train folds.
        b. Grid-scan s_f in {1.00, 1.05, ..., 2.00} on the train folds.
        c. Apply (s_f, mu_tr) to held-out fold; collect OOF.
    Record pooled cross-fit RAE on the 253 for this seed.
  Mean of the 5 pooled RAEs = bagged honest cross-fit estimate.
  Mean of s_f across all 5*5 = 25 folds = deploy s.

Deploy: apply (mean_s) with the global anchor mean as mu to the 513 te file.

Outputs:
  data/processed/te_nb1021.npy
  data/processed/nb1021_summary.json
  submissions/nb1021_stretch_only_bag.csv
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
from sklearn.model_selection import KFold

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1021"
ANCHOR = "chemprop_aux"
STRETCH_GRID = np.round(np.arange(1.00, 2.001, 0.05), 2).tolist()
N_FOLDS = 5
SEEDS = [0, 1, 7, 42, 137]
NB1011_SINGLE_SEED = 0.6109   # nb1011 stretch-only, seed=42
NB1014_BLEND_BAG = 0.5930      # nb1014 5-seed blend+stretch
CHEMPROP_AUX_RAW = 0.6216


def load_te(name: str, te_names: np.ndarray) -> np.ndarray:
    npy = DATA_PROCESSED / f"te_{name}.npy"
    if npy.exists():
        return np.load(npy).astype(np.float64)
    sub = pd.read_csv(SUBMISSIONS / f"{name}.csv")
    assert (sub["Molecule Name"].values == te_names).all(), (
        f"{name}: submission row order does not match test order")
    return sub["pEC50"].values.astype(np.float64)


def best_stretch_on(p_train: np.ndarray, y_train: np.ndarray,
                    mu: float) -> tuple[float, float]:
    """Grid-scan s on the train-fold; return (best_s, best_train_rae)."""
    best_s, best_r = 1.0, float("inf")
    for s in STRETCH_GRID:
        stretched = mu + s * (p_train - mu)
        r = float(rae(y_train, stretched))
        if r < best_r:
            best_r = r
            best_s = float(s)
    return best_s, best_r


def run_one_seed(p_unb: np.ndarray, y_unb: np.ndarray, seed: int) -> dict:
    """Run the nb1011 stretch-only 5-fold protocol for one KFold seed."""
    n_unb = len(y_unb)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n_unb, np.nan)
    folds = []
    for k, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n_unb))):
        mu_tr = float(p_unb[tr_loc].mean())
        s_f, rae_tr = best_stretch_on(p_unb[tr_loc], y_unb[tr_loc], mu_tr)
        oof[va_loc] = mu_tr + s_f * (p_unb[va_loc] - mu_tr)
        rae_va = float(rae(y_unb[va_loc], oof[va_loc]))
        folds.append({"fold": k, "s": s_f, "mu_tr": mu_tr,
                      "train_rae": rae_tr, "val_rae": rae_va,
                      "n_va": int(len(va_loc))})
    pooled = float(rae(y_unb, oof))
    return {"seed": seed, "folds": folds, "pooled_rae": pooled, "oof": oof}


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- multi-seed bag of nb1011 stretch-only "
          f"(seeds={SEEDS})")
    print("=" * 78)

    # ---- Load 513 test ----
    te = load_test()
    te_names = te["name"].values
    p_513 = load_te(ANCHOR, te_names)
    print(f"[load] p_513({ANCHOR}) shape = {p_513.shape}")

    # ---- Load 253 unblind ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    p_unb = p_513[unb_idx]
    n_unb = len(y_unb)
    print(f"[load] p_unb shape = {p_unb.shape}, y shape = {y_unb.shape}")

    # ---- Anchor in_RAE sanity ----
    raw_in_rae = float(rae(y_unb, p_unb))
    print(f"\n[indiv] in_RAE on 253 unblind:")
    print(f"   {ANCHOR:30s}: {raw_in_rae:.4f}  "
          f"(expected ~{CHEMPROP_AUX_RAW})")

    # =================================================================
    # Run nb1011 protocol for each seed
    # =================================================================
    print("\n" + "-" * 78)
    print(f"MULTI-SEED CROSS-FIT  (N_FOLDS={N_FOLDS}, "
          f"stretch grid {STRETCH_GRID[0]}..{STRETCH_GRID[-1]}, no blending)")
    print("-" * 78)

    seed_results = []
    per_seed_rae = []
    all_s = []
    for seed in SEEDS:
        res = run_one_seed(p_unb, y_unb, seed)
        seed_results.append({
            "seed": seed,
            "pooled_rae": res["pooled_rae"],
            "folds": res["folds"],
        })
        per_seed_rae.append(res["pooled_rae"])
        for f in res["folds"]:
            all_s.append(f["s"])
        fold_vals = [round(f["val_rae"], 3) for f in res["folds"]]
        print(f"   seed {seed:>3d}: pooled_RAE = {res['pooled_rae']:.4f}  "
              f"(folds: {fold_vals})")

    mean_rae = float(np.mean(per_seed_rae))
    std_rae = float(np.std(per_seed_rae))
    print(f"\n[bag] mean pooled CV RAE = {mean_rae:.4f}  "
          f"(std across seeds {std_rae:.4f})")
    print(f"[bag] vs nb1011 single-seed = {NB1011_SINGLE_SEED:.4f}")
    print(f"[bag] vs nb1014 blend bag    = {NB1014_BLEND_BAG:.4f}")

    mean_s = float(np.mean(all_s))
    std_s = float(np.std(all_s))
    print(f"\n[bag] mean s across 25 folds = {mean_s:.4f}  "
          f"(std {std_s:.4f})")

    # =================================================================
    # Deploy: apply mean_s with the global anchor mean as mu
    # =================================================================
    print("\n" + "-" * 78)
    print("DEPLOY  (mean_s applied with global anchor mean as mu)")
    print("-" * 78)
    mu_deploy = float(p_unb.mean())
    deployed_unb = mu_deploy + mean_s * (p_unb - mu_deploy)
    in_rae_final = float(rae(y_unb, deployed_unb))
    print(f"   deploy mu (anchor)  = {mu_deploy:.4f}")
    print(f"   deploy s            = {mean_s:.4f}")
    print(f"   in-sample RAE (253) = {in_rae_final:.4f}  "
          "(overfit lower bound)")

    deploy_513 = (mu_deploy + mean_s * (p_513 - mu_deploy)).astype(np.float32)
    print(f"   te(513) mean/std    = "
          f"{deploy_513.mean():.3f} / {deploy_513.std():.3f}")

    # =================================================================
    # Save
    # =================================================================
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy_513)
    plain = SUBMISSIONS / f"{TAG}_stretch_only_bag.csv"
    pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te_names,
        "pEC50": deploy_513,
    }).to_csv(plain, index=False)
    print(f"\n[save] te_{TAG}.npy")
    print(f"[save] {plain}")

    delta_vs_nb1011 = mean_rae - NB1011_SINGLE_SEED
    delta_vs_nb1014 = mean_rae - NB1014_BLEND_BAG
    if delta_vs_nb1011 < -0.005:
        verdict_vs_nb1011 = "BAG_HELPS_SOLO"
    elif abs(delta_vs_nb1011) <= 0.005:
        verdict_vs_nb1011 = "BAG_INERT_SOLO"
    else:
        verdict_vs_nb1011 = "BAG_HURTS_SOLO"

    if delta_vs_nb1014 < -0.005:
        verdict_vs_nb1014 = "SOLO_BEATS_BLEND"
    elif abs(delta_vs_nb1014) <= 0.005:
        verdict_vs_nb1014 = "SOLO_TIES_BLEND"
    else:
        verdict_vs_nb1014 = "BLEND_IS_THE_LEVER"

    print(f"\n[verdict] bag vs nb1011 single-seed "
          f"({NB1011_SINGLE_SEED:.4f}): "
          f"delta={delta_vs_nb1011:+.4f}  -> {verdict_vs_nb1011}")
    print(f"[verdict] bag vs nb1014 blend-bag "
          f"({NB1014_BLEND_BAG:.4f}): "
          f"delta={delta_vs_nb1014:+.4f}  -> {verdict_vs_nb1014}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "anchor_raw_in_rae_253": raw_in_rae,
        "stretch_grid": STRETCH_GRID,
        "n_folds": N_FOLDS,
        "seeds": SEEDS,
        "per_seed_pooled_rae": per_seed_rae,
        "mean_pooled_rae": mean_rae,
        "std_pooled_rae": std_rae,
        "nb1011_single_seed_baseline": NB1011_SINGLE_SEED,
        "nb1014_blend_bag_baseline": NB1014_BLEND_BAG,
        "chemprop_aux_raw_baseline": CHEMPROP_AUX_RAW,
        "delta_vs_nb1011": delta_vs_nb1011,
        "delta_vs_nb1014": delta_vs_nb1014,
        "verdict_vs_nb1011": verdict_vs_nb1011,
        "verdict_vs_nb1014": verdict_vs_nb1014,
        "mean_s": mean_s,
        "std_s_across_folds": std_s,
        "deploy_mu_anchor": mu_deploy,
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
    print(f"   anchor                     = {ANCHOR}")
    print(f"   raw chemprop_aux           = {CHEMPROP_AUX_RAW:.4f}")
    print(f"   per-seed pooled RAE        = "
          f"{[f'{r:.4f}' for r in per_seed_rae]}")
    print(f"   mean (bagged honest CV)    = {mean_rae:.4f}  "
          f"(std {std_rae:.4f})")
    print(f"   nb1011 single-seed         = {NB1011_SINGLE_SEED:.4f}")
    print(f"   nb1014 blend bag           = {NB1014_BLEND_BAG:.4f}")
    print(f"   delta vs nb1011            = {delta_vs_nb1011:+.4f}  "
          f"-> {verdict_vs_nb1011}")
    print(f"   delta vs nb1014            = {delta_vs_nb1014:+.4f}  "
          f"-> {verdict_vs_nb1014}")
    print(f"   deploy s                   = {mean_s:.3f}")
    print(f"   wall                       = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("per_seed_pooled_rae", "mean_pooled_rae", "std_pooled_rae",
              "delta_vs_nb1011", "delta_vs_nb1014",
              "verdict_vs_nb1011", "verdict_vs_nb1014",
              "mean_s", "in_sample_rae_overfit_bound", "plain_submission"):
        print(f"  {k}: {res.get(k)}")
