"""nb1104 -- Pure non-parametric k-NN regression on Tanimoto / Morgan FPs.

Hypothesis: pure k-NN is a classic distinct method with a fundamentally
different inductive bias from tree/linear/GNN predictors that dominate the
ladder. Even if standalone in_RAE is modest, the residual axis should be
orthogonal enough to nb972 (LGBM-Mordred long-train) to earn a bag slot.

Recipe:
  1. Compute 513x4139 Tanimoto similarity matrix on Morgan ECFP4 (2048-bit).
  2. For each of the 513 test rows, take the k=15 highest-similarity train
     neighbors. Weighted mean of their training pEC50 with weights = sim^2.
  3. Evaluate in_RAE on the 253 Phase-1 unblind audit indices.
  4. Pearson correlation vs te_nb972_long_train.
  5. Conditional bag (nb1014-style SLSQP + stretch with chemprop_aux) when
     Pearson(te_nb1104, te_nb972) < 0.95 OR in_RAE < 0.65.

Outputs:
  data/processed/te_nb1104.npy
  data/processed/nb1104_summary.json
  submissions/nb1104_knn_regressor.csv
  (conditional) data/processed/te_nb1104_nb1014bag.npy
                submissions/nb1104_nb1014bag_chemprop_aux.csv
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
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from pxr.chem import morgan_fp_batch
from pxr.data import load_test, load_train
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1104"
K = 15

# nb1014 protocol constants
STRETCH_GRID = np.round(np.arange(1.00, 2.001, 0.05), 2).tolist()
SEEDS = [0, 1, 7, 42, 137]
N_FOLDS_BAG = 5
NB1014_REF_RAE = 0.5994

PEARSON_THRESH = 0.95
IN_RAE_THRESH = 0.65


# ---------- nb1014 bag helpers ----------

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


def run_one_seed(P_unb: np.ndarray, y_unb: np.ndarray, seed: int) -> dict:
    n_unb = len(y_unb)
    kf = KFold(n_splits=N_FOLDS_BAG, shuffle=True, random_state=seed)
    oof = np.full(n_unb, np.nan)
    folds = []
    for k, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n_unb))):
        w0_f = slsqp_w0(P_unb[tr_loc, 0], P_unb[tr_loc, 1], y_unb[tr_loc])
        blend_tr = w0_f * P_unb[tr_loc, 0] + (1.0 - w0_f) * P_unb[tr_loc, 1]
        mu_tr = float(blend_tr.mean())
        s_f, _ = best_stretch_on(blend_tr, y_unb[tr_loc], mu_tr)
        blend_va = w0_f * P_unb[va_loc, 0] + (1.0 - w0_f) * P_unb[va_loc, 1]
        oof[va_loc] = mu_tr + s_f * (blend_va - mu_tr)
        folds.append({"fold": k, "w0": w0_f, "s": s_f, "mu_tr": mu_tr,
                      "n_va": int(len(va_loc))})
    return {"seed": seed, "folds": folds,
            "pooled_rae": float(rae(y_unb, oof)), "oof": oof}


def run_nb1014_bag(te_nb1104: np.ndarray, te_names: np.ndarray,
                   te_smiles: np.ndarray, unb_idx: np.ndarray,
                   y_unb: np.ndarray) -> dict:
    print("\n" + "-" * 78)
    print("NB1014-STYLE BAG  pool = (chemprop_aux, nb1104)")
    print("-" * 78)
    te_chemprop = np.load(DATA_PROCESSED / "te_chemprop_aux.npy").astype(np.float64)
    preds_513 = np.column_stack([te_chemprop, te_nb1104.astype(np.float64)])
    P_unb = preds_513[unb_idx]
    per_seed_rae, all_w0, all_s = [], [], []
    for seed in SEEDS:
        res = run_one_seed(P_unb, y_unb, seed)
        per_seed_rae.append(res["pooled_rae"])
        for f in res["folds"]:
            all_w0.append(f["w0"]); all_s.append(f["s"])
        print(f"   seed {seed:>3d}: pooled_RAE = {res['pooled_rae']:.4f}")
    mean_rae = float(np.mean(per_seed_rae))
    std_rae = float(np.std(per_seed_rae))
    mean_w0 = float(np.mean(all_w0))
    mean_s = float(np.mean(all_s))
    blend_unb_all = mean_w0 * P_unb[:, 0] + (1.0 - mean_w0) * P_unb[:, 1]
    mu_deploy = float(blend_unb_all.mean())
    in_rae_bag = float(rae(y_unb,
                            mu_deploy + mean_s * (blend_unb_all - mu_deploy)))
    blend_513 = mean_w0 * preds_513[:, 0] + (1.0 - mean_w0) * preds_513[:, 1]
    deploy_513 = (mu_deploy + mean_s * (blend_513 - mu_deploy)).astype(np.float32)
    print(f"\n[bag] mean pooled CV RAE = {mean_rae:.4f}  (std {std_rae:.4f})")
    print(f"[bag] deploy w0={mean_w0:.4f}  s={mean_s:.4f}  mu={mu_deploy:.4f}")
    print(f"[bag] in-sample 253 RAE  = {in_rae_bag:.4f}")
    np.save(DATA_PROCESSED / f"te_{TAG}_nb1014bag.npy", deploy_513)
    sub_path = SUBMISSIONS / f"{TAG}_nb1014bag_chemprop_aux.csv"
    pd.DataFrame({"SMILES": te_smiles, "Molecule Name": te_names,
                  "pEC50": deploy_513}).to_csv(sub_path, index=False)
    print(f"[save] {sub_path}")
    return {
        "ran": True,
        "per_seed_pooled_rae": per_seed_rae,
        "mean_pooled_rae": mean_rae,
        "std_pooled_rae": std_rae,
        "mean_w0_chemprop_aux": mean_w0,
        "mean_w1_nb1104": float(1.0 - mean_w0),
        "mean_s": mean_s,
        "deploy_mu_blend": mu_deploy,
        "in_sample_rae_overfit_bound": in_rae_bag,
        "submission": str(sub_path),
        "delta_vs_nb1014_ref": mean_rae - NB1014_REF_RAE,
        "beats_nb1014": mean_rae < NB1014_REF_RAE - 0.005,
    }


# ---------- Tanimoto k-NN ----------

def tanimoto_513x4139(fp_te: np.ndarray, fp_tr: np.ndarray) -> np.ndarray:
    """Tanimoto = |A ∩ B| / (|A| + |B| - |A ∩ B|) for binary fingerprints.

    fp_te: (513, 2048) uint8
    fp_tr: (4139, 2048) uint8
    Returns (513, 4139) float32.
    """
    A = fp_te.astype(np.float32)
    B = fp_tr.astype(np.float32)
    inter = A @ B.T              # (513, 4139)
    a = A.sum(axis=1, keepdims=True)   # (513, 1)
    b = B.sum(axis=1, keepdims=True).T # (1, 4139)
    denom = a + b - inter
    # avoid div-by-zero: if denom==0 (both all-zero), tanimoto undefined -> 0
    out = np.where(denom > 0, inter / denom, 0.0).astype(np.float32)
    return out


def knn_weighted_mean(sim_te_tr: np.ndarray, y_tr: np.ndarray,
                      k: int) -> np.ndarray:
    """For each test row, weighted mean of top-k train pEC50, weights = sim^2."""
    n_te = sim_te_tr.shape[0]
    out = np.empty(n_te, dtype=np.float64)
    # argpartition for top-k then sort within
    topk_idx = np.argpartition(-sim_te_tr, kth=k - 1, axis=1)[:, :k]
    for i in range(n_te):
        idx = topk_idx[i]
        sims = sim_te_tr[i, idx]
        w = sims ** 2
        s = w.sum()
        if s <= 0:
            out[i] = float(y_tr.mean())
        else:
            out[i] = float((w * y_tr[idx]).sum() / s)
    return out


# ---------- main ----------

def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Tanimoto k-NN regressor (k={K}, weights = sim^2)")
    print("=" * 78)

    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr); n_te = len(te)
    print(f"[load] n_train={n_tr}  n_test={n_te}")

    # ---- Morgan FP (2048-bit ECFP4) ----
    print("\n[feat] Morgan FP batch on train+test...")
    t_f = time.time()
    fp_tr = morgan_fp_batch(tr["smiles"].tolist())
    fp_te = morgan_fp_batch(te["smiles"].tolist())
    print(f"[feat] fp_tr={fp_tr.shape}  fp_te={fp_te.shape}  "
          f"elapsed={time.time()-t_f:.1f}s")

    # ---- 513 x 4139 Tanimoto matrix ----
    print("\n[sim] computing 513x4139 Tanimoto matrix...")
    t_s = time.time()
    sim = tanimoto_513x4139(fp_te, fp_tr)
    print(f"[sim] shape={sim.shape}  mean={sim.mean():.4f}  "
          f"max={sim.max():.4f}  elapsed={time.time()-t_s:.1f}s")

    # ---- k-NN weighted mean ----
    print(f"\n[knn] k={K}, weights = sim^2, weighted mean over top-{K} train pEC50...")
    te_preds_raw = knn_weighted_mean(sim, y_tr, K)
    te_preds = np.clip(te_preds_raw, y_tr.min() - 0.5,
                       y_tr.max() + 0.5).astype(np.float32)
    print(f"[knn] te mean/std = {te_preds.mean():.3f}/{te_preds.std():.3f}")

    # ---- in_RAE on 253 ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    in_r = float(rae(y_unb, te_preds[unb_idx].astype(np.float64)))
    print(f"[deploy] in_RAE(253) = {in_r:.4f}")

    # ---- Pearson vs nb972 ----
    te_nb972 = np.load(DATA_PROCESSED / "te_nb972_long_train.npy").astype(np.float64)
    pearson_972 = float(np.corrcoef(te_preds.astype(np.float64), te_nb972)[0, 1])
    print(f"[corr] Pearson(te_nb1104, te_nb972) = {pearson_972:.4f}")
    try:
        te_chemprop = np.load(DATA_PROCESSED / "te_chemprop_aux.npy").astype(np.float64)
        pearson_cp = float(np.corrcoef(te_preds.astype(np.float64), te_chemprop)[0, 1])
        print(f"[corr] Pearson(te_nb1104, te_chemprop_aux) = {pearson_cp:.4f}")
    except FileNotFoundError:
        pearson_cp = None

    # ---- Comparison vs nb1070 (current PRIMARY) ----
    try:
        te_nb1070 = np.load(DATA_PROCESSED / "te_nb1070.npy").astype(np.float64)
        nb1070_in_rae = float(rae(y_unb, te_nb1070[unb_idx]))
        beats_nb1070 = bool(in_r < nb1070_in_rae)
        pearson_1070 = float(np.corrcoef(te_preds.astype(np.float64), te_nb1070)[0, 1])
        print(f"[ref]  nb1070 in_RAE(253) = {nb1070_in_rae:.4f}  -> beats? {beats_nb1070}")
        print(f"[corr] Pearson(te_nb1104, te_nb1070) = {pearson_1070:.4f}")
    except FileNotFoundError:
        nb1070_in_rae = None
        beats_nb1070 = False
        pearson_1070 = None

    # ---- Save base outputs ----
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", te_preds)
    base_sub = SUBMISSIONS / f"{TAG}_knn_regressor.csv"
    pd.DataFrame({"SMILES": te["smiles"].values,
                  "Molecule Name": te["name"].values,
                  "pEC50": te_preds}).to_csv(base_sub, index=False)
    print(f"[save] te_{TAG}.npy, {base_sub}")

    # ---- Neighbor sim diagnostics ----
    top1_sim = sim.max(axis=1)
    topk_sims = np.sort(sim, axis=1)[:, -K:]
    print(f"[diag] top-1 sim mean/median = {top1_sim.mean():.3f}/{np.median(top1_sim):.3f}")
    print(f"[diag] top-{K} sim mean = {topk_sims.mean():.3f}")

    # ---- Conditional bag ----
    trigger_pearson = pearson_972 < PEARSON_THRESH
    trigger_in_rae = in_r < IN_RAE_THRESH
    fire_bag = trigger_pearson or trigger_in_rae
    bag_summary = {
        "ran": False,
        "reason": (f"pearson_{pearson_972:.4f}_>=_{PEARSON_THRESH}"
                   f"_AND_in_rae_{in_r:.4f}_>=_{IN_RAE_THRESH}"),
    }
    print(f"\n[gate] pearson < {PEARSON_THRESH}? {trigger_pearson}  "
          f"in_RAE < {IN_RAE_THRESH}? {trigger_in_rae}  "
          f"-> fire_bag={fire_bag}")
    if fire_bag:
        bag_summary = run_nb1014_bag(te_preds, te["name"].values,
                                      te["smiles"].values, unb_idx, y_unb)

    summary = {
        "tag": TAG,
        "method": "tanimoto_knn_weighted_mean",
        "k": K,
        "weight": "sim_squared",
        "fp": "morgan_ecfp4_2048bit",
        "n_train": n_tr,
        "n_test": n_te,
        "sim_matrix_shape": list(sim.shape),
        "sim_mean": float(sim.mean()),
        "sim_max": float(sim.max()),
        "top1_sim_mean": float(top1_sim.mean()),
        "top1_sim_median": float(np.median(top1_sim)),
        f"top{K}_sim_mean": float(topk_sims.mean()),
        "in_rae_253": in_r,
        "test_mean": float(te_preds.mean()),
        "test_std": float(te_preds.std()),
        "pearson_nb972": pearson_972,
        "pearson_chemprop_aux": pearson_cp,
        "pearson_nb1070": pearson_1070,
        "nb1070_in_rae_253": nb1070_in_rae,
        "beats_nb1070": beats_nb1070,
        "base_submission": str(base_sub),
        "bag_triggered_pearson_lt_thresh": bool(trigger_pearson),
        "bag_triggered_in_rae_lt_thresh": bool(trigger_in_rae),
        "nb1014_bag": bag_summary,
        "wall_sec": round(time.time() - t0, 2),
    }
    with open(DATA_PROCESSED / f"{TAG}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {DATA_PROCESSED / f'{TAG}_summary.json'}")
    print(f"\n=== {TAG} done in {time.time()-t0:.1f}s ===")
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("k", "fp", "in_rae_253", "test_mean", "test_std",
              "pearson_nb972", "pearson_chemprop_aux", "pearson_nb1070",
              "nb1070_in_rae_253", "beats_nb1070", "base_submission"):
        print(f"  {k}: {res.get(k)}")
    bag = res.get("nb1014_bag", {})
    if bag.get("ran"):
        print("  bag.mean_pooled_rae:", bag.get("mean_pooled_rae"))
        print("  bag.mean_w0_chemprop_aux:", bag.get("mean_w0_chemprop_aux"))
        print("  bag.mean_w1_nb1104:", bag.get("mean_w1_nb1104"))
        print("  bag.mean_s:", bag.get("mean_s"))
        print("  bag.beats_nb1014:", bag.get("beats_nb1014"))
        print("  bag.submission:", bag.get("submission"))
    else:
        print("  bag.ran: False  (reason:", bag.get("reason"), ")")
