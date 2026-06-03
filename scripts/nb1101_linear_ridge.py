"""nb1101 -- RidgeCV on Morgan + RDKit + Mordred (3798 cols).

Hypothesis: linear inductive bias is genuinely different from the tree-based
predictors that dominate the ladder (chemprop_aux GNN, nb972 LGBM, nb932
CatBoost, nb1030 LGBM-Mordred). A pure L2-regularized linear model on a
high-dimensional concatenation (Morgan FP + RDKit + Mordred) should give an
orthogonal residual axis even though its standalone in_RAE may be modest.

Recipe:
  1. Features: combined (Morgan 2048 + RDKit ~217 = 2265) hstack with cached
     Mordred (1533) from C:/pxr_artifacts/nb1030/ -> 3798-col panel on
     4139 train + 513 test (computed jointly so the imputer/scaler are
     panel-fit, no train/test leak via mean).
  2. sklearn.linear_model.RidgeCV(alphas=[0.1, 1, 10, 100, 1000], cv=5) on
     4139 CRC pEC50. StandardScaler before Ridge (linear model needs scaling).
  3. Predict 513. in_RAE on the 253 Phase-1 unblind audit indices. Pearson
     vs te_nb972_long_train.
  4. If Pearson(te_nb1101, te_nb972) < 0.95 OR in_RAE < 0.65, fire the
     nb1014-style multi-seed SLSQP+stretch bag on the 2-way pool
     (chemprop_aux, nb1101).

Outputs:
  data/processed/oof_nb1101.npy
  data/processed/te_nb1101.npy
  data/processed/nb1101_summary.json
  submissions/nb1101_linear_ridge.csv
  (conditional) data/processed/te_nb1101_nb1014bag.npy
                submissions/nb1101_nb1014bag_chemprop_aux.csv
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
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from pxr.data import load_test, load_train
from pxr.eval import rae
from pxr.featurize import combined as combined_feats
from pxr.featurize import impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1101"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]

# nb1014 protocol constants
STRETCH_GRID = np.round(np.arange(1.00, 2.001, 0.05), 2).tolist()
SEEDS = [0, 1, 7, 42, 137]
N_FOLDS_BAG = 5
NB1014_REF_RAE = 0.5994

PEARSON_THRESH = 0.95
IN_RAE_THRESH = 0.65


# ---------- nb1014 bag helpers (mirrored from nb1030) ----------

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


def run_nb1014_bag(te_nb1101: np.ndarray, te_names: np.ndarray,
                   te_smiles: np.ndarray, unb_idx: np.ndarray,
                   y_unb: np.ndarray) -> dict:
    print("\n" + "-" * 78)
    print("NB1014-STYLE BAG  pool = (chemprop_aux, nb1101)")
    print("-" * 78)
    te_chemprop = np.load(DATA_PROCESSED / "te_chemprop_aux.npy").astype(np.float64)
    preds_513 = np.column_stack([te_chemprop, te_nb1101.astype(np.float64)])
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
        "mean_w1_nb1101": float(1.0 - mean_w0),
        "mean_s": mean_s,
        "deploy_mu_blend": mu_deploy,
        "in_sample_rae_overfit_bound": in_rae_bag,
        "submission": str(sub_path),
        "delta_vs_nb1014_ref": mean_rae - NB1014_REF_RAE,
        "beats_nb1014": mean_rae < NB1014_REF_RAE - 0.005,
    }


# ---------- main ----------

def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- RidgeCV on Morgan+RDKit+Mordred (3798 cols)")
    print("=" * 78)

    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr); n_te = len(te)
    print(f"[load] n_train={n_tr}  n_test={n_te}")

    # ---- Combined Morgan + RDKit (2265 dim) ----
    print("\n[feat] computing combined (Morgan + RDKit) on train+test...")
    t_c = time.time()
    smis_all = tr["smiles"].tolist() + te["smiles"].tolist()
    X_combined_all = combined_feats(smis_all)
    X_combined_all = impute(X_combined_all)
    print(f"[feat] combined shape = {X_combined_all.shape}  "
          f"elapsed={time.time()-t_c:.1f}s")

    # ---- Mordred (1533 dim) from nb1030 cache ----
    X_md_tr = np.load(MORDRED_DIR / "X_mordred_train.npy")
    X_md_te = np.load(MORDRED_DIR / "X_mordred_test.npy")
    X_mordred_all = np.vstack([X_md_tr, X_md_te])
    print(f"[feat] mordred (cached) shape = {X_mordred_all.shape}")

    # ---- Concat -> 3798 ----
    X_all = np.hstack([X_combined_all.astype(np.float32),
                       X_mordred_all.astype(np.float32)])
    print(f"[feat] concat shape = {X_all.shape}")

    X_tr = X_all[:n_tr]
    X_te = X_all[n_tr:]

    # ---- StandardScaler fit on train, apply both ----
    print("\n[scale] StandardScaler fit on 4139 train, apply to test...")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    # post-scale, replace any residual non-finite with 0
    X_tr_s = np.nan_to_num(X_tr_s, nan=0.0, posinf=0.0, neginf=0.0)
    X_te_s = np.nan_to_num(X_te_s, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- RidgeCV(alphas=[0.1, 1, 10, 100, 1000], cv=5) ----
    print(f"\n[ridge] RidgeCV alphas={ALPHAS} cv=5 on {X_tr_s.shape}...")
    t_r = time.time()
    ridge = RidgeCV(alphas=ALPHAS, cv=5,
                    scoring="neg_mean_squared_error")
    ridge.fit(X_tr_s, y_tr)
    best_alpha = float(ridge.alpha_)
    print(f"[ridge] best alpha = {best_alpha}  elapsed={time.time()-t_r:.1f}s")

    # ---- Predict 513 ----
    te_preds_raw = ridge.predict(X_te_s)
    te_preds = np.clip(te_preds_raw, y_tr.min() - 0.5,
                       y_tr.max() + 0.5).astype(np.float32)

    # ---- OOF on train (5-fold KFold, for diagnostic) ----
    print("\n[oof] 5-fold KFold OOF Ridge at best_alpha (diagnostic)...")
    from sklearn.linear_model import Ridge
    kf_diag = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.full(n_tr, np.nan)
    for tr_idx, va_idx in kf_diag.split(np.arange(n_tr)):
        m = Ridge(alpha=best_alpha)
        m.fit(X_tr_s[tr_idx], y_tr[tr_idx])
        oof[va_idx] = m.predict(X_tr_s[va_idx])
    oof_rae = float(rae(y_tr, oof))
    print(f"[oof] random-5fold OOF RAE = {oof_rae:.4f}")

    # ---- in_RAE on 253 ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    in_r = float(rae(y_unb, te_preds[unb_idx].astype(np.float64)))
    print(f"[deploy] te mean/std = {te_preds.mean():.3f}/{te_preds.std():.3f}  "
          f"in_RAE(253) = {in_r:.4f}")

    # ---- Pearson vs nb972 ----
    te_nb972 = np.load(DATA_PROCESSED / "te_nb972_long_train.npy").astype(np.float64)
    pearson_972 = float(np.corrcoef(te_preds.astype(np.float64), te_nb972)[0, 1])
    print(f"[corr] Pearson(te_nb1101, te_nb972) = {pearson_972:.4f}")
    try:
        te_chemprop = np.load(DATA_PROCESSED / "te_chemprop_aux.npy").astype(np.float64)
        pearson_cp = float(np.corrcoef(te_preds.astype(np.float64), te_chemprop)[0, 1])
        print(f"[corr] Pearson(te_nb1101, te_chemprop_aux) = {pearson_cp:.4f}")
    except FileNotFoundError:
        pearson_cp = None

    # ---- Save base outputs ----
    np.save(DATA_PROCESSED / f"oof_{TAG}.npy", oof)
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", te_preds)
    base_sub = SUBMISSIONS / f"{TAG}_linear_ridge.csv"
    pd.DataFrame({"SMILES": te["smiles"].values,
                  "Molecule Name": te["name"].values,
                  "pEC50": te_preds}).to_csv(base_sub, index=False)
    print(f"[save] te_{TAG}.npy, oof_{TAG}.npy, {base_sub}")

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
        "alphas": ALPHAS,
        "best_alpha": best_alpha,
        "n_features": int(X_tr.shape[1]),
        "feature_split": {"combined_morgan_rdkit": int(X_combined_all.shape[1]),
                          "mordred": int(X_mordred_all.shape[1])},
        "oof_rae_random5f": oof_rae,
        "in_rae_253": in_r,
        "test_mean": float(te_preds.mean()),
        "test_std": float(te_preds.std()),
        "pearson_nb972": pearson_972,
        "pearson_chemprop_aux": pearson_cp,
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
    for k in ("n_features", "best_alpha", "oof_rae_random5f", "in_rae_253",
              "pearson_nb972", "pearson_chemprop_aux", "base_submission"):
        print(f"  {k}: {res.get(k)}")
    bag = res.get("nb1014_bag", {})
    if bag.get("ran"):
        print("  bag.mean_pooled_rae:", bag.get("mean_pooled_rae"))
        print("  bag.mean_w0_chemprop_aux:", bag.get("mean_w0_chemprop_aux"))
        print("  bag.mean_s:", bag.get("mean_s"))
        print("  bag.beats_nb1014:", bag.get("beats_nb1014"))
        print("  bag.submission:", bag.get("submission"))
    else:
        print("  bag.ran: False  (reason:", bag.get("reason"), ")")
