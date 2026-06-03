"""nb1031 -- Multi-objective LightGBM orthogonality scan.

Hypothesis: training 5 LightGBMs on the same combined (Morgan + RDKit)
2265-D feature matrix but varying ONLY the loss objective produces
predictors whose residual structure differs. If any one of these
predictors has Pearson(te, te_nb972) < 0.95 on 513, it carries an
orthogonal signal we can fold into a (chemprop_aux + ortho) bag using
the nb1014 multi-seed protocol.

Objectives swept (identical features, n_est=1500, num_leaves=64,
lr=0.03 everywhere):
  1. huber          alpha=2.0   (smoothed L2 / L1 hybrid)
  2. regression_l1              (pure L1 / MAE)
  3. quantile       alpha=0.5   (median regression)
  4. poisson                    (count-style log link)
  5. regression                 (pure L2 / MSE -- baseline reference)

For each objective:
  - Scaffold 5-fold CV on 4139 CRC -> OOF preds + fold RAEs
  - Refit on full train -> te(513)
  - in_RAE on 253 unblind
  - Pearson(te, te_nb972) on full 513

If any objective has Pearson < 0.95 -> orthogonality candidate. Run
nb1014-style multi-seed (5 seeds, 5 folds) bag with pool
(chemprop_aux, that candidate) and report cross-fit RAE.

Wall-time budget: < 12 min. With combined features cached and n_est=1500
each fit is ~30-45s; 5 obj * 6 fits (5 folds + 1 full) ~= 7-10 min total.

Outputs:
  data/processed/oof_nb1031_<obj>.npy
  data/processed/te_nb1031_<obj>.npy
  data/processed/nb1031_summary.json
  submissions/nb1031_<obj>.csv
  (conditional) submissions/nb1031_bag_<obj>_chemprop_aux.csv
                data/processed/te_nb1031_bag_<obj>.npy
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

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from pxr.chem import bemis_murcko
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1031"
SEED = 42
N_FOLDS = 5

# Shared LGBM knobs across all 5 objectives
BASE_PARAMS = dict(
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=64,
    min_child_samples=20,
    reg_lambda=0.2,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    verbose=-1,
    n_jobs=4,
)

# Objective-specific keys merged onto BASE_PARAMS for each variant
OBJ_VARIANTS: list[tuple[str, dict]] = [
    ("huber",          {"objective": "huber",          "alpha": 2.0}),
    ("regression_l1",  {"objective": "regression_l1"}),
    ("quantile",       {"objective": "quantile",       "alpha": 0.5}),
    ("poisson",        {"objective": "poisson"}),
    ("regression",     {"objective": "regression"}),
]

# nb1014-style bag config
STRETCH_GRID = np.round(np.arange(1.00, 2.001, 0.05), 2).tolist()
BAG_SEEDS = [0, 1, 7, 42, 137]
NB1014_REF_RAE = 0.5994
PEARSON_THRESHOLD = 0.95


# ---------- nb1014 bag protocol (compact mirror) ----------

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
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
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


def run_bag(obj_name: str, te_cand: np.ndarray, te_names: np.ndarray,
            te_smiles: np.ndarray, unb_idx: np.ndarray,
            y_unb: np.ndarray) -> dict:
    print("\n" + "-" * 78)
    print(f"NB1014-STYLE BAG  pool = (chemprop_aux, nb1031_{obj_name})")
    print("-" * 78)
    te_chemprop = np.load(DATA_PROCESSED / "te_chemprop_aux.npy").astype(np.float64)
    preds_513 = np.column_stack([te_chemprop, te_cand.astype(np.float64)])
    P_unb = preds_513[unb_idx]
    per_seed_rae, all_w0, all_s = [], [], []
    for seed in BAG_SEEDS:
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
    np.save(DATA_PROCESSED / f"te_{TAG}_bag_{obj_name}.npy", deploy_513)
    sub_path = SUBMISSIONS / f"{TAG}_bag_{obj_name}_chemprop_aux.csv"
    pd.DataFrame({"SMILES": te_smiles, "Molecule Name": te_names,
                  "pEC50": deploy_513}).to_csv(sub_path, index=False)
    print(f"[save] {sub_path}")
    return {
        "ran": True,
        "obj": obj_name,
        "per_seed_pooled_rae": per_seed_rae,
        "mean_pooled_rae": mean_rae,
        "std_pooled_rae": std_rae,
        "mean_w0_chemprop_aux": mean_w0,
        "mean_w1_cand": float(1.0 - mean_w0),
        "mean_s": mean_s,
        "deploy_mu_blend": mu_deploy,
        "in_sample_rae_overfit_bound": in_rae_bag,
        "submission": str(sub_path),
        "delta_vs_nb1014_ref": mean_rae - NB1014_REF_RAE,
        "beats_nb1014": mean_rae < NB1014_REF_RAE - 0.005,
    }


# ---------- per-objective LGBM driver ----------

def train_one_objective(name: str, extra_params: dict,
                        X_tr: np.ndarray, y_tr: np.ndarray,
                        X_te: np.ndarray, splits) -> dict:
    """Scaffold 5-fold CV + full-train refit. Return preds + diagnostics."""
    params = {**BASE_PARAMS, **extra_params}
    n_tr = len(y_tr)
    oof = np.full(n_tr, np.nan)
    fold_raes = []
    print(f"\n[obj={name}] params={extra_params}")
    t_obj = time.time()
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = lgb.train(
            params,
            lgb.Dataset(X_tr[tr_idx], label=y_tr[tr_idx]),
            valid_sets=[lgb.Dataset(X_tr[va_idx], label=y_tr[va_idx])],
            callbacks=[lgb.log_evaluation(-1)],
        )
        oof[va_idx] = m.predict(X_tr[va_idx])
        fr = float(rae(y_tr[va_idx], oof[va_idx]))
        fold_raes.append(fr)
        print(f"   fold {fold+1}: RAE={fr:.4f}  "
              f"({time.time()-t_obj:.1f}s)", flush=True)
    oof_rae = float(rae(y_tr, oof))

    # Full-train refit
    m_full = lgb.train(params, lgb.Dataset(X_tr, label=y_tr),
                       callbacks=[lgb.log_evaluation(-1)])
    te_preds = np.clip(m_full.predict(X_te),
                       y_tr.min() - 0.5,
                       y_tr.max() + 0.5).astype(np.float32)
    print(f"   OOF RAE={oof_rae:.4f}  te mean/std="
          f"{te_preds.mean():.3f}/{te_preds.std():.3f}  "
          f"obj_wall={time.time()-t_obj:.1f}s")
    return {
        "name": name,
        "params_used": extra_params,
        "fold_raes": fold_raes,
        "oof_rae": oof_rae,
        "oof": oof.astype(np.float32),
        "te": te_preds,
        "wall_sec": round(time.time() - t_obj, 2),
    }


# ---------- main ----------

def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- multi-objective LGBM orthogonality scan vs nb972")
    print(f"   objectives: {[v[0] for v in OBJ_VARIANTS]}")
    print(f"   shared params: n_est=1500 num_leaves=64 lr=0.03")
    print("=" * 78)

    # ---- Data ----
    tr = load_train()
    te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr); n_te = len(te)
    print(f"[load] n_train={n_tr}  n_test={n_te}")
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # ---- Features (combined: Morgan 2048 + RDKit 217 -> 2265) ----
    print("[feat] computing combined (Morgan + RDKit) features...")
    t_f = time.time()
    X_tr = impute(combined(tr["smiles"].tolist()))
    X_te = impute(combined(te["smiles"].tolist()))
    print(f"   X_tr={X_tr.shape}  X_te={X_te.shape}  "
          f"feat_wall={time.time()-t_f:.1f}s")

    # ---- Unblind index for in_RAE ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    print(f"[unb] n_unb={len(unb_idx)}")

    # ---- nb972 reference ----
    te_nb972 = np.load(DATA_PROCESSED / "te_nb972_long_train.npy").astype(np.float64)

    # ---- Train each objective ----
    per_obj_results: list[dict] = []
    per_obj_in_rae: dict[str, float] = {}
    per_obj_pearson: dict[str, float] = {}
    for name, extra in OBJ_VARIANTS:
        res = train_one_objective(name, extra, X_tr, y_tr, X_te, splits)
        # Save artifacts
        np.save(DATA_PROCESSED / f"oof_{TAG}_{name}.npy", res["oof"])
        np.save(DATA_PROCESSED / f"te_{TAG}_{name}.npy", res["te"])
        sub = SUBMISSIONS / f"{TAG}_{name}.csv"
        pd.DataFrame({
            "SMILES": te["smiles"].values,
            "Molecule Name": te["name"].values,
            "pEC50": res["te"],
        }).to_csv(sub, index=False)
        # in_RAE + Pearson
        in_r = float(rae(y_unb, res["te"][unb_idx].astype(np.float64)))
        pear = float(np.corrcoef(res["te"].astype(np.float64), te_nb972)[0, 1])
        per_obj_in_rae[name] = in_r
        per_obj_pearson[name] = pear
        res["in_rae_253"] = in_r
        res["pearson_nb972"] = pear
        res["submission"] = str(sub)
        print(f"   -> in_RAE(253)={in_r:.4f}  "
              f"Pearson(te,nb972)={pear:.4f}  "
              f"(orth: {'YES' if pear < PEARSON_THRESHOLD else 'no'})")
        per_obj_results.append({
            "name": name,
            "params_used": res["params_used"],
            "fold_raes": res["fold_raes"],
            "oof_rae": res["oof_rae"],
            "in_rae_253": in_r,
            "pearson_nb972": pear,
            "te_mean": float(res["te"].mean()),
            "te_std": float(res["te"].std()),
            "submission": str(sub),
            "wall_sec": res["wall_sec"],
        })

    # ---- Identify orthogonal candidates ----
    print("\n" + "=" * 78)
    print("ORTHOGONALITY SUMMARY  (Pearson vs te_nb972, threshold "
          f"{PEARSON_THRESHOLD})")
    print("=" * 78)
    sorted_objs = sorted(per_obj_pearson.items(), key=lambda kv: kv[1])
    for name, p in sorted_objs:
        flag = "  <-- ORTH" if p < PEARSON_THRESHOLD else ""
        print(f"   {name:<16s}  Pearson={p:.4f}   "
              f"in_RAE={per_obj_in_rae[name]:.4f}{flag}")

    ortho_candidates = [n for n, p in per_obj_pearson.items()
                        if p < PEARSON_THRESHOLD]
    print(f"\n   orthogonal candidates: {ortho_candidates or 'NONE'}")

    # ---- Conditional bag(s): pick the LOWEST cross-fit RAE ----
    bag_results: list[dict] = []
    best_bag: dict | None = None
    if ortho_candidates:
        for cand_name in ortho_candidates:
            cand_te = np.load(DATA_PROCESSED / f"te_{TAG}_{cand_name}.npy")
            br = run_bag(cand_name, cand_te.astype(np.float64),
                         te["name"].values, te["smiles"].values,
                         unb_idx, y_unb)
            bag_results.append(br)
            if best_bag is None or br["mean_pooled_rae"] < best_bag["mean_pooled_rae"]:
                best_bag = br
        assert best_bag is not None
        print("\n" + "=" * 78)
        print(f"BEST BAG  obj={best_bag['obj']}  "
              f"mean_pooled_rae={best_bag['mean_pooled_rae']:.4f}  "
              f"(vs nb1014 ref {NB1014_REF_RAE:.4f})")
        print("=" * 78)
    else:
        print("\n[skip] No objective has Pearson < 0.95 -- no bag run")

    summary = {
        "tag": TAG,
        "shared_params": BASE_PARAMS,
        "objectives": [v[0] for v in OBJ_VARIANTS],
        "pearson_threshold": PEARSON_THRESHOLD,
        "per_objective": per_obj_results,
        "per_obj_in_rae_253": per_obj_in_rae,
        "per_obj_pearson_nb972": per_obj_pearson,
        "orthogonal_candidates": ortho_candidates,
        "bag_results": bag_results,
        "best_bag": best_bag,
        "nb1014_ref_rae": NB1014_REF_RAE,
        "wall_sec": round(time.time() - t0, 2),
    }
    with open(DATA_PROCESSED / f"{TAG}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: float(o))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_summary.json'}")
    print(f"=== {TAG} done in {time.time()-t0:.1f}s ===")
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    print(f"  objectives_run: {res['objectives']}")
    print(f"  per_obj_in_rae_253: {res['per_obj_in_rae_253']}")
    print(f"  per_obj_pearson_nb972: {res['per_obj_pearson_nb972']}")
    print(f"  orthogonal_candidates: {res['orthogonal_candidates']}")
    if res.get("best_bag"):
        bb = res["best_bag"]
        print(f"  best_bag.obj: {bb['obj']}")
        print(f"  best_bag.mean_pooled_rae: {bb['mean_pooled_rae']}")
        print(f"  best_bag.beats_nb1014: {bb['beats_nb1014']}")
        print(f"  best_bag.submission: {bb['submission']}")
    print(f"  wall_sec: {res['wall_sec']}")
