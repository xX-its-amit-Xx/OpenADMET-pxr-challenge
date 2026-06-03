"""nb1103 -- XGBoost with EXTREME hyperparams as orthogonal base.

Hypothesis: The ladder is dominated by booster recipes with conservative
hyperparams (nb972 LGBM eta=0.005 depth=8, nb1023 XGB Huber eta=0.005
depth=8, nb932 CatBoost). Pushing XGB to the *opposite* extreme --
shallow tree count (50), very deep trees (max_depth=20), very high LR
(0.5), heavy L2 (10.0), aggressive row/col subsampling (0.5/0.5) --
should over-fit on idiosyncratic feature interactions that the slow
recipes smooth out, yielding a residual axis that is *not* correlated
with nb972.

Recipe (intentionally pathological):
  - XGBoost reg:pseudohubererror, huber_slope=2.0
  - max_depth=20  (very deep)
  - num_round=50  (very short)
  - eta=0.5       (very high LR)
  - reg_lambda=10.0  (heavy L2)
  - subsample=0.5, colsample_bytree=0.5
  - Features: combined (Morgan 2048 + RDKit 217 = 2265)
  - Scaffold 5-fold CV on 4139 CRC; predict 513 with final-fit booster
    trained on the full 4139 at the same 50-round budget (no early stop).
  - in_RAE on 253 Phase-1 unblind.
  - Pearson( te_nb1103 , te_nb972 ) on the 513.

Decision rule:
  Pearson < 0.95  OR  in_RAE < 0.65  -> fire nb1014-style multi-seed
                                        SLSQP+stretch bag on
                                        (chemprop_aux, nb1103).
  Otherwise -> redundant, no bag.

Outputs:
  data/processed/oof_nb1103.npy
  data/processed/te_nb1103.npy
  data/processed/nb1103_summary.json
  submissions/nb1103_xgb_extreme.csv
  (conditional) data/processed/te_nb1103_nb1014bag.npy
                submissions/nb1103_nb1014bag_chemprop_aux.csv
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
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from pxr.chem import bemis_murcko
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.featurize import combined as combined_feats
from pxr.featurize import impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1103"
SEED = 42
N_FOLDS = 5

XGB_PARAMS = dict(
    objective="reg:pseudohubererror",
    huber_slope=2.0,
    max_depth=20,
    eta=0.5,
    subsample=0.5,
    colsample_bytree=0.5,
    reg_lambda=10.0,
    tree_method="hist",
    nthread=4,
    seed=SEED,
    verbosity=0,
)
N_ROUNDS = 50

PEARSON_THRESH = 0.95
IN_RAE_THRESH = 0.65

# nb1014 bag protocol constants (mirrors nb1101 / nb1030)
STRETCH_GRID = np.round(np.arange(1.00, 2.001, 0.05), 2).tolist()
SEEDS = [0, 1, 7, 42, 137]
N_FOLDS_BAG = 5
NB1014_REF_RAE = 0.5994


# ---------- nb1014-style bag helpers ----------

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


def run_nb1014_bag(te_nb1103: np.ndarray, te_names: np.ndarray,
                   te_smiles: np.ndarray, unb_idx: np.ndarray,
                   y_unb: np.ndarray) -> dict:
    print("\n" + "-" * 78)
    print(f"NB1014-STYLE BAG  pool = (chemprop_aux, {TAG})")
    print("-" * 78)
    te_chemprop = np.load(DATA_PROCESSED / "te_chemprop_aux.npy").astype(np.float64)
    preds_513 = np.column_stack([te_chemprop, te_nb1103.astype(np.float64)])
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
        "mean_w1_nb1103": float(1.0 - mean_w0),
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
    print(f"{TAG} -- XGBoost EXTREME hyperparams (deep+short+highLR+heavyL2)")
    print("=" * 78)

    # Truth / unblind
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    assert len(unb_idx) == len(y_unb) == 253

    # Data
    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr); n_te = len(te)
    te_names = te["name"].values
    te_smiles = te["smiles"].values
    print(f"[load] n_train={n_tr}  n_test={n_te}")

    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    print("\n[feat] computing combined (Morgan + RDKit)...")
    t_c = time.time()
    X_tr = impute(combined_feats(tr["smiles"].tolist()))
    X_te = impute(combined_feats(te["smiles"].tolist()))
    print(f"[feat] X_tr={X_tr.shape}  X_te={X_te.shape}  "
          f"elapsed={time.time()-t_c:.1f}s")

    # Scaffold 5-fold CV with fixed num_rounds (no early stop -- extreme recipe)
    oof = np.full(n_tr, np.nan)
    fold_raes = []
    print(f"\n[cv] {N_FOLDS} scaffold folds  num_rounds={N_ROUNDS}  "
          f"max_depth={XGB_PARAMS['max_depth']}  eta={XGB_PARAMS['eta']}")
    for fold, (tr_idx, va_idx) in enumerate(splits):
        base = float(np.mean(y_tr[tr_idx]))
        dtr = xgb.DMatrix(X_tr[tr_idx], label=y_tr[tr_idx])
        dva = xgb.DMatrix(X_tr[va_idx], label=y_tr[va_idx])
        params_fold = dict(XGB_PARAMS, base_score=base)
        booster = xgb.train(
            params_fold,
            dtr,
            num_boost_round=N_ROUNDS,
            verbose_eval=False,
        )
        oof[va_idx] = booster.predict(dva)
        fr = float(rae(y_tr[va_idx], oof[va_idx]))
        fold_raes.append(fr)
        print(f"  fold {fold+1}  RAE={fr:.4f}  elapsed={time.time()-t0:6.1f}s",
              flush=True)

    oof_rae = float(rae(y_tr, oof))
    print(f"\n[cv] scaffold OOF RAE = {oof_rae:.4f}")

    # Final fit on full 4139 at the same fixed budget
    print(f"\n[final] fit on all {n_tr} train, num_rounds={N_ROUNDS}...")
    base_full = float(np.mean(y_tr))
    dall = xgb.DMatrix(X_tr, label=y_tr)
    dte = xgb.DMatrix(X_te)
    final_booster = xgb.train(
        dict(XGB_PARAMS, base_score=base_full),
        dall,
        num_boost_round=N_ROUNDS,
        verbose_eval=False,
    )
    te_preds_raw = final_booster.predict(dte)
    te_preds = np.clip(te_preds_raw,
                       y_tr.min() - 0.5, y_tr.max() + 0.5).astype(np.float32)
    ratio = float(te_preds.std() / oof.std()) if oof.std() > 0 else 0.0
    in_r = float(rae(y_unb, te_preds[unb_idx].astype(np.float64)))
    print(f"[deploy] TE med={np.median(te_preds):.2f}  std={te_preds.std():.3f}"
          f"  ratio(te/oof)={ratio:.2f}")
    print(f"[deploy] in_RAE(253) = {in_r:.4f}")

    # Pearson vs nb972
    te_nb972 = np.load(DATA_PROCESSED / "te_nb972_long_train.npy").astype(np.float64)
    pearson_972 = float(np.corrcoef(te_preds.astype(np.float64), te_nb972)[0, 1])
    print(f"[corr] Pearson(te_{TAG}, te_nb972) = {pearson_972:.4f}")
    try:
        te_chemprop = np.load(DATA_PROCESSED / "te_chemprop_aux.npy").astype(np.float64)
        pearson_cp = float(np.corrcoef(te_preds.astype(np.float64), te_chemprop)[0, 1])
        print(f"[corr] Pearson(te_{TAG}, te_chemprop_aux) = {pearson_cp:.4f}")
    except FileNotFoundError:
        pearson_cp = None

    # Save base
    np.save(DATA_PROCESSED / f"oof_{TAG}.npy", oof)
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", te_preds)
    base_sub = SUBMISSIONS / f"{TAG}_xgb_extreme.csv"
    pd.DataFrame({"SMILES": te_smiles, "Molecule Name": te_names,
                  "pEC50": te_preds}).to_csv(base_sub, index=False)
    print(f"[save] te_{TAG}.npy, oof_{TAG}.npy, {base_sub}")

    # Conditional bag
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
        bag_summary = run_nb1014_bag(te_preds, te_names, te_smiles,
                                     unb_idx, y_unb)

    summary = {
        "tag": TAG,
        "params": {k: v for k, v in XGB_PARAMS.items() if k != "verbosity"},
        "num_rounds": N_ROUNDS,
        "n_features": int(X_tr.shape[1]),
        "fold_scaffold_raes": fold_raes,
        "oof_rae_scaffold": oof_rae,
        "in_rae_253": in_r,
        "test_mean": float(te_preds.mean()),
        "test_std": float(te_preds.std()),
        "te_oof_std_ratio": ratio,
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
    for k in ("n_features", "oof_rae_scaffold", "in_rae_253",
              "test_std", "te_oof_std_ratio",
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
