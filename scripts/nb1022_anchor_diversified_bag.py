"""nb1022 -- Anchor-diversified bag of 5 (2-model + stretch) recipes.

The nb1001 protocol (chemprop_aux + nb972 + stretch) gave honest CV 0.5994.
When more candidates are pooled into a single SLSQP (nb1020), some get
zero-weight and disappear. This script asks the orthogonal question:

  What if we DON'T let SLSQP collapse the pool? Instead, we run 5
  separate (anchor + partner + stretch) recipes and bag the predictions.

Pairs:
  P1 = (chemprop_aux, nb972_long_train)               # nb1001 baseline
  P2 = (chemprop_aux, nb914)                          # persistence homology
  P3 = (chemprop_aux, nb960)                          # pseudo self-train
  P4 = (nb972_long_train, nb914)                      # non-anchor pair
  P5 = (chemprop_aux, nb923)                          # WL graph kernel

For each pair, run the nb1001 protocol on the 253 unblind:
  5-fold KFold(seed=42, shuffle=True):
    a. Fit SLSQP w0 on the train folds (K=2, sum-to-1, [0,1] bounds).
    b. Grid-scan s in {1.00, 1.05, ..., 2.00} on train folds with
       mu = train-fold blend mean.
    c. Apply (w0, s, mu_tr) to held-out fold; collect OOF.
  Pool pair_oof on the 253; record per-pair pooled cross-fit RAE.

Bag = mean of the 5 pair OOFs on the 253; report bag pooled RAE.

Deploy: for each pair, refit (w0, s, mu) on all 253 and apply to the
513 te files; bag the 5 deploy vectors by mean.

Hypothesis: diverse anchor pairs may capture orthogonal slices of the
513 test set, even though SLSQP zero-weighted some of these candidates
when pooled directly. A bag of pair-level stretch recipes is a softer
regularizer than a global SLSQP.

Outputs:
  data/processed/te_nb1022.npy
  data/processed/nb1022_summary.json
  submissions/nb1022_anchor_diversified_bag.csv
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

TAG = "nb1022"
PAIRS = [
    ("chemprop_aux", "nb972_long_train"),
    ("chemprop_aux", "nb914"),
    ("chemprop_aux", "nb960"),
    ("nb972_long_train", "nb914"),
    ("chemprop_aux", "nb923"),
]
STRETCH_GRID = np.round(np.arange(1.00, 2.001, 0.05), 2).tolist()
N_FOLDS = 5
SEED = 42
NB1014_BAG_REFERENCE = 0.5994  # nb1014 multi-seed bag mean (best 2-model recipe)


def load_te(name: str, te_names: np.ndarray) -> np.ndarray:
    npy = DATA_PROCESSED / f"te_{name}.npy"
    if npy.exists():
        return np.load(npy).astype(np.float64)
    sub = pd.read_csv(SUBMISSIONS / f"{name}.csv")
    assert (sub["Molecule Name"].values == te_names).all(), (
        f"{name}: submission row order does not match test order")
    return sub["pEC50"].values.astype(np.float64)


def slsqp_w0(p0: np.ndarray, p1: np.ndarray, y: np.ndarray) -> float:
    """Fit w0 in [0,1] minimizing SSE of w0*p0 + (1-w0)*p1 vs y. Returns w0."""
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


def run_pair_crossfit(P_unb: np.ndarray, y_unb: np.ndarray,
                      seed: int) -> dict:
    """Run nb1001 5-fold (SLSQP w0 + stretch grid) for one pair."""
    n_unb = len(y_unb)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n_unb, np.nan)
    folds = []
    for k, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n_unb))):
        w0_f = slsqp_w0(P_unb[tr_loc, 0], P_unb[tr_loc, 1], y_unb[tr_loc])
        blend_tr = w0_f * P_unb[tr_loc, 0] + (1.0 - w0_f) * P_unb[tr_loc, 1]
        mu_tr = float(blend_tr.mean())
        s_f, rae_tr = best_stretch_on(blend_tr, y_unb[tr_loc], mu_tr)
        blend_va = w0_f * P_unb[va_loc, 0] + (1.0 - w0_f) * P_unb[va_loc, 1]
        oof[va_loc] = mu_tr + s_f * (blend_va - mu_tr)
        folds.append({
            "fold": k, "w0": w0_f, "s": s_f, "mu_tr": mu_tr,
            "train_rae": rae_tr,
            "val_rae": float(rae(y_unb[va_loc], oof[va_loc])),
            "n_va": int(len(va_loc)),
        })
    return {"oof": oof, "pooled_rae": float(rae(y_unb, oof)), "folds": folds}


def deploy_pair(P_unb: np.ndarray, y_unb: np.ndarray,
                preds_513: np.ndarray) -> tuple[np.ndarray, dict]:
    """Refit (w0, s, mu) on all 253, apply to 513."""
    w0 = slsqp_w0(P_unb[:, 0], P_unb[:, 1], y_unb)
    blend_unb = w0 * P_unb[:, 0] + (1.0 - w0) * P_unb[:, 1]
    mu = float(blend_unb.mean())
    s, _ = best_stretch_on(blend_unb, y_unb, mu)
    in_rae = float(rae(y_unb, mu + s * (blend_unb - mu)))
    blend_513 = w0 * preds_513[:, 0] + (1.0 - w0) * preds_513[:, 1]
    deploy_513 = (mu + s * (blend_513 - mu)).astype(np.float64)
    info = {
        "deploy_w0": float(w0),
        "deploy_w1": float(1.0 - w0),
        "deploy_mu": mu,
        "deploy_s": float(s),
        "in_sample_rae_overfit_bound": in_rae,
    }
    return deploy_513, info


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- anchor-diversified bag of 5 (anchor + partner + stretch)")
    print("=" * 78)
    for i, (a, b) in enumerate(PAIRS, 1):
        print(f"   P{i}: ({a}, {b})")

    # ---- Load 513 test ----
    te = load_test()
    te_names = te["name"].values

    # ---- Load 253 unblind ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"\n[load] 513 test = {len(te_names)} rows; 253 unblind = {n_unb}")

    # ---- Run each pair ----
    pair_results = []
    pair_oofs_253 = []
    pair_deploys_513 = []
    print("\n" + "-" * 78)
    print(f"PER-PAIR HONEST CROSS-FIT  (KFold seed={SEED}, "
          f"stretch grid {STRETCH_GRID[0]}..{STRETCH_GRID[-1]})")
    print("-" * 78)
    for i, (a, b) in enumerate(PAIRS, 1):
        # Load this pair's 513 vectors (and slice 253)
        p_a = load_te(a, te_names)
        p_b = load_te(b, te_names)
        preds_513 = np.column_stack([p_a, p_b])
        P_unb = preds_513[unb_idx]

        indiv_a = float(rae(y_unb, P_unb[:, 0]))
        indiv_b = float(rae(y_unb, P_unb[:, 1]))

        cf = run_pair_crossfit(P_unb, y_unb, seed=SEED)
        pair_oofs_253.append(cf["oof"])

        deploy_513, dinfo = deploy_pair(P_unb, y_unb, preds_513)
        pair_deploys_513.append(deploy_513)

        rec = {
            "pair": [a, b],
            "indiv_in_rae": {a: indiv_a, b: indiv_b},
            "pooled_cv_rae_253": cf["pooled_rae"],
            "folds": cf["folds"],
            **dinfo,
        }
        pair_results.append(rec)
        print(f"   P{i} ({a:>20s}, {b:>20s})  "
              f"indiv=({indiv_a:.3f},{indiv_b:.3f})  "
              f"cv_RAE={cf['pooled_rae']:.4f}  "
              f"deploy(w0={dinfo['deploy_w0']:.2f},"
              f"s={dinfo['deploy_s']:.2f})  "
              f"in={dinfo['in_sample_rae_overfit_bound']:.3f}")

    # ---- Bag the 5 pair OOFs on 253 ----
    bag_oof_253 = np.mean(np.column_stack(pair_oofs_253), axis=1)
    bag_pooled_rae = float(rae(y_unb, bag_oof_253))
    per_pair_rae = [r["pooled_cv_rae_253"] for r in pair_results]
    mean_pair_rae = float(np.mean(per_pair_rae))

    print("\n" + "-" * 78)
    print("BAG (mean across 5 pair OOFs)")
    print("-" * 78)
    print(f"   per-pair pooled CV RAE     = "
          f"{[f'{r:.4f}' for r in per_pair_rae]}")
    print(f"   mean of per-pair RAE       = {mean_pair_rae:.4f}")
    print(f"   bag pooled CV RAE (253)    = {bag_pooled_rae:.4f}")
    print(f"   nb1014 multi-seed bag ref  = {NB1014_BAG_REFERENCE:.4f}")

    delta = bag_pooled_rae - NB1014_BAG_REFERENCE
    beats = delta < -0.001
    if beats:
        verdict = "BEATS_NB1014"
    elif delta <= 0.005:
        verdict = "TIES_NB1014"
    else:
        verdict = "WORSE_THAN_NB1014"
    print(f"   delta vs nb1014            = {delta:+.4f}  -> {verdict}")

    # ---- Deploy = mean of the 5 per-pair deploy vectors ----
    deploy_513 = np.mean(np.column_stack(pair_deploys_513), axis=1).astype(
        np.float32)
    bag_in_rae = float(rae(y_unb, deploy_513[unb_idx]))
    print("\n" + "-" * 78)
    print("DEPLOY  (mean of 5 per-pair refit-on-253 vectors)")
    print("-" * 78)
    print(f"   te(513) mean/std           = "
          f"{deploy_513.mean():.3f} / {deploy_513.std():.3f}")
    print(f"   bag in-sample RAE on 253   = {bag_in_rae:.4f}  "
          "(overfit lower bound)")

    # ---- Save ----
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy_513)
    plain = SUBMISSIONS / f"{TAG}_anchor_diversified_bag.csv"
    pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te_names,
        "pEC50": deploy_513,
    }).to_csv(plain, index=False)
    print(f"\n[save] te_{TAG}.npy")
    print(f"[save] {plain}")

    summary = {
        "tag": TAG,
        "pairs": [list(p) for p in PAIRS],
        "stretch_grid": STRETCH_GRID,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "pair_results": pair_results,
        "per_pair_pooled_rae": per_pair_rae,
        "mean_pair_pooled_rae": mean_pair_rae,
        "bag_pooled_cv_rae_253": bag_pooled_rae,
        "nb1014_bag_reference": NB1014_BAG_REFERENCE,
        "delta_vs_nb1014": delta,
        "beats_nb1014": bool(beats),
        "verdict": verdict,
        "bag_in_sample_rae": bag_in_rae,
        "deploy_te_mean": float(deploy_513.mean()),
        "deploy_te_std": float(deploy_513.std()),
        "plain_submission": str(plain),
        "wall_sec": round(time.time() - t0, 2),
    }
    with open(DATA_PROCESSED / f"{TAG}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {DATA_PROCESSED / f'{TAG}_summary.json'}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   pairs                      = {len(PAIRS)}")
    print(f"   per-pair pooled CV RAE     = "
          f"{[f'{r:.4f}' for r in per_pair_rae]}")
    print(f"   bag pooled CV RAE (253)    = {bag_pooled_rae:.4f}")
    print(f"   nb1014 reference           = {NB1014_BAG_REFERENCE:.4f}")
    print(f"   delta                      = {delta:+.4f}")
    print(f"   verdict                    = {verdict}")
    print(f"   wall                       = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("per_pair_pooled_rae", "mean_pair_pooled_rae",
              "bag_pooled_cv_rae_253", "delta_vs_nb1014",
              "beats_nb1014", "verdict", "bag_in_sample_rae",
              "plain_submission"):
        print(f"  {k}: {res.get(k)}")
