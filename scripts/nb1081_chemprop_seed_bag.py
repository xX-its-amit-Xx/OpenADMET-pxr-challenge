"""nb1081 -- Chemprop-distillation seed-bag (5-LGBM ensemble of chemprop_aux).

Hypothesis: the true `chemprop_aux` PRIMARY-1 is a single fixed Chemprop GNN
checkpoint -- a single sample from the model-randomness distribution. We can
distill it into 5 LGBMs that approximate its behaviour at different random
seeds. Bagging those 5 should be ORTHOGONAL to both the original chemprop_aux
(it is a distinct model class, LGBM Huber, not a GNN) and to nb972
(a different LGBM with a different feature stack & objective).

If Pearson(bag, nb972) < 0.95 we then replace nb972 in the nb1014 protocol
(2-element pool, SLSQP w0 + stretch grid, 5-seed bag) with this new
"chemprop_ensemble" and re-run the SLSQP+stretch cross-fit.

Procedure:
  Features: combined Morgan(2048) + RDKit(~217) + chemprop_aux_oof as a single
            knowledge-distillation feature (4139 train, 513 test).
  For each LGBM seed in {0, 1, 7, 42, 137}:
    - Scaffold 5-fold CV on the 4139 train, accumulate oof predictions on 4139.
    - Refit on full 4139, predict 513 test.
  Bag = mean across 5 seeds for both oof_4139 and te_513.
  Cross-fit comparison on the 253 unblind (subset of 513): Pearson(bag, nb972).

  IF Pearson < 0.95:
    Re-run nb1014's SLSQP w0 + stretch grid bag protocol with
      pool = [chemprop_aux, chemprop_ensemble]   (instead of nb972).

Outputs:
  data/processed/oof_nb1081_chemprop_ensemble.npy
  data/processed/te_nb1081_chemprop_ensemble.npy
  data/processed/nb1081_summary.json
  submissions/nb1081_chemprop_ensemble.csv
  (if Pearson<0.95) submissions/nb1081_new_blend.csv
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

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from pxr.chem import bemis_murcko, standardize
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1081"
SEEDS = [0, 1, 7, 42, 137]
N_FOLDS = 5

# nb1014 SLSQP+stretch protocol constants
STRETCH_GRID = np.round(np.arange(1.00, 2.001, 0.05), 2).tolist()
NB1070_REF_RAE = 0.5780  # nb1070 median bag (recent strong cross-fit anchor)


def make_lgbm(seed: int) -> lgb.LGBMRegressor:
    """Huber LGBM that approximates the chemprop_aux residual surface."""
    return lgb.LGBMRegressor(
        objective="huber",
        alpha=0.9,
        n_estimators=500,
        num_leaves=64,
        learning_rate=0.05,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_data_in_leaf=20,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


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


def run_nb1014_protocol(P_unb: np.ndarray, y_unb: np.ndarray,
                        seeds: list[int]) -> dict:
    """nb1014 multi-seed SLSQP+stretch bag for a 2-element pool."""
    n_unb = len(y_unb)
    per_seed_rae, all_w0, all_s = [], [], []
    for seed in seeds:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        oof = np.full(n_unb, np.nan)
        for tr_loc, va_loc in kf.split(np.arange(n_unb)):
            w0_f = slsqp_w0(P_unb[tr_loc, 0], P_unb[tr_loc, 1], y_unb[tr_loc])
            blend_tr = w0_f * P_unb[tr_loc, 0] + (1.0 - w0_f) * P_unb[tr_loc, 1]
            mu_tr = float(blend_tr.mean())
            s_f, _ = best_stretch_on(blend_tr, y_unb[tr_loc], mu_tr)
            blend_va = w0_f * P_unb[va_loc, 0] + (1.0 - w0_f) * P_unb[va_loc, 1]
            oof[va_loc] = mu_tr + s_f * (blend_va - mu_tr)
            all_w0.append(w0_f)
            all_s.append(s_f)
        per_seed_rae.append(float(rae(y_unb, oof)))
    return {
        "per_seed_rae": per_seed_rae,
        "mean_rae": float(np.mean(per_seed_rae)),
        "std_rae": float(np.std(per_seed_rae)),
        "mean_w0": float(np.mean(all_w0)),
        "mean_s": float(np.mean(all_s)),
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- chemprop_aux distillation seed-bag "
          f"(5 LGBMs, seeds={SEEDS})")
    print("=" * 78)

    # ---- Load data ----
    tr = load_train()
    te = load_test()
    n_tr, n_te = len(tr), len(te)
    print(f"[load] train rows = {n_tr}  test rows = {n_te}")

    # ---- Anchors ----
    oof_ca = np.load(DATA_PROCESSED / "oof_chemprop_aux.npy").astype(np.float64)
    te_ca = np.load(DATA_PROCESSED / "te_chemprop_aux.npy").astype(np.float64)
    te_972 = np.load(DATA_PROCESSED / "te_nb972_long_train.npy").astype(np.float64)
    assert oof_ca.shape == (n_tr,)
    assert te_ca.shape == (n_te,)
    assert te_972.shape == (n_te,)
    print(f"[load] oof_chemprop_aux  mean={oof_ca.mean():.3f} "
          f"std={oof_ca.std():.3f}")

    # ---- Features (Morgan+RDKit + chemprop_aux distillation feature) ----
    print("\n[feat] computing combined Morgan+RDKit features ...")
    smi_tr = tr["smiles"].apply(standardize).tolist()
    smi_te = te["smiles"].apply(standardize).tolist()
    X_tr_base = impute(combined(smi_tr))
    X_te_base = impute(combined(smi_te))
    print(f"[feat] X_tr_base shape = {X_tr_base.shape}")
    print(f"[feat] X_te_base shape = {X_te_base.shape}")
    # Knowledge distillation: add chemprop_aux pred as a feature.
    X_tr = np.column_stack([X_tr_base, oof_ca.astype(np.float32)])
    X_te = np.column_stack([X_te_base, te_ca.astype(np.float32)])
    print(f"[feat] X_tr full shape   = {X_tr.shape}  (+chemprop_aux feature)")

    y_tr = tr["pec50"].values.astype(np.float64)

    # ---- Scaffold folds ----
    scaffolds = [bemis_murcko(s) or "" for s in smi_tr]
    folds = scaffold_kfold_indices(scaffolds, n_splits=N_FOLDS,
                                   shuffle=True, seed=42)
    print(f"[cv] scaffold {N_FOLDS}-fold sizes: "
          f"{[len(va) for _, va in folds]}")

    # ---- Per-seed OOF + test predictions ----
    print("\n" + "-" * 78)
    print(f"TRAINING {len(SEEDS)} LGBM Huber distillation models")
    print("-" * 78)
    oof_stack = np.zeros((len(SEEDS), n_tr), dtype=np.float64)
    te_stack = np.zeros((len(SEEDS), n_te), dtype=np.float64)
    per_seed_oof_rae = []
    for i, seed in enumerate(SEEDS):
        t_seed = time.time()
        oof = np.zeros(n_tr, dtype=np.float64)
        for k, (tr_idx, va_idx) in enumerate(folds):
            model = make_lgbm(seed)
            model.fit(X_tr[tr_idx], y_tr[tr_idx])
            oof[va_idx] = model.predict(X_tr[va_idx])
        oof_stack[i] = oof
        # Refit on full train, predict 513.
        full = make_lgbm(seed)
        full.fit(X_tr, y_tr)
        te_stack[i] = full.predict(X_te)
        r = float(rae(y_tr, oof))
        per_seed_oof_rae.append(r)
        print(f"   seed {seed:>3d}: scaffold-CV OOF RAE = {r:.4f}  "
              f"te(513) mean={te_stack[i].mean():.3f} "
              f"std={te_stack[i].std():.3f}  "
              f"[{time.time() - t_seed:.1f}s]")

    # ---- Bag (mean) ----
    oof_bag = oof_stack.mean(axis=0)
    te_bag = te_stack.mean(axis=0)
    bag_oof_rae = float(rae(y_tr, oof_bag))
    print(f"\n[bag] mean across 5 seeds:")
    print(f"   scaffold-CV OOF RAE = {bag_oof_rae:.4f}  "
          f"(per-seed mean {np.mean(per_seed_oof_rae):.4f})")
    print(f"   te(513) mean={te_bag.mean():.3f}  std={te_bag.std():.3f}")

    # ---- Save bag artifacts ----
    np.save(DATA_PROCESSED / f"oof_{TAG}_chemprop_ensemble.npy",
            oof_bag.astype(np.float32))
    np.save(DATA_PROCESSED / f"te_{TAG}_chemprop_ensemble.npy",
            te_bag.astype(np.float32))
    plain = SUBMISSIONS / f"{TAG}_chemprop_ensemble.csv"
    pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te["name"].values,
        "pEC50": te_bag.astype(np.float32),
    }).to_csv(plain, index=False)
    print(f"[save] oof_{TAG}_chemprop_ensemble.npy")
    print(f"[save] te_{TAG}_chemprop_ensemble.npy")
    print(f"[save] {plain}")

    # ---- Orthogonality check: Pearson(bag, nb972) on 513 ----
    pearson_full = float(np.corrcoef(te_bag, te_972)[0, 1])
    pearson_ca = float(np.corrcoef(te_bag, te_ca)[0, 1])
    print(f"\n[ortho] Pearson(bag, nb972) on 513    = {pearson_full:.4f}")
    print(f"[ortho] Pearson(bag, chemprop_aux) 513 = {pearson_ca:.4f}")

    # ---- Unblind comparison ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    bag_unb = te_bag[unb_idx]
    ca_unb = te_ca[unb_idx]
    n972_unb = te_972[unb_idx]
    in_rae_bag = float(rae(y_unb, bag_unb))
    in_rae_ca = float(rae(y_unb, ca_unb))
    in_rae_972 = float(rae(y_unb, n972_unb))
    pearson_unb_972 = float(np.corrcoef(bag_unb, n972_unb)[0, 1])
    print(f"[unb] in_RAE(bag)         on 253 = {in_rae_bag:.4f}")
    print(f"[unb] in_RAE(chemprop_aux) 253   = {in_rae_ca:.4f}")
    print(f"[unb] in_RAE(nb972)        253   = {in_rae_972:.4f}")
    print(f"[unb] Pearson(bag, nb972)  253   = {pearson_unb_972:.4f}")

    # ---- IF orthogonal: run nb1014 SLSQP+stretch with the new pool ----
    orthogonal = pearson_full < 0.95
    print("\n" + "-" * 78)
    print(f"ORTHOGONALITY TEST: Pearson(bag, nb972) = {pearson_full:.4f}  "
          f"-> {'ORTHOGONAL (run new blend)' if orthogonal else 'COLLINEAR (skip)'}")
    print("-" * 78)

    new_blend_summary = None
    if orthogonal:
        # nb1014 protocol with pool = [chemprop_aux, chemprop_ensemble]
        P_unb_new = np.column_stack([ca_unb, bag_unb])
        new_blend_summary = run_nb1014_protocol(P_unb_new, y_unb, SEEDS)
        print(f"\n[new_blend] pool = [chemprop_aux, chemprop_ensemble]")
        print(f"   per-seed pooled RAE = "
              f"{[f'{r:.4f}' for r in new_blend_summary['per_seed_rae']]}")
        print(f"   mean bagged honest CV RAE = "
              f"{new_blend_summary['mean_rae']:.4f}  "
              f"(std {new_blend_summary['std_rae']:.4f})")
        print(f"   mean w0 (chemprop_aux) = "
              f"{new_blend_summary['mean_w0']:.4f}  "
              f"(w1 nb1081_bag = "
              f"{1.0 - new_blend_summary['mean_w0']:.4f})")
        print(f"   mean stretch s         = "
              f"{new_blend_summary['mean_s']:.4f}")

        # Deploy new blend on 513.
        w0, s_dep = new_blend_summary["mean_w0"], new_blend_summary["mean_s"]
        blend_513 = w0 * te_ca + (1.0 - w0) * te_bag
        blend_unb_anchor = w0 * ca_unb + (1.0 - w0) * bag_unb
        mu_dep = float(blend_unb_anchor.mean())
        deploy_513 = (mu_dep + s_dep * (blend_513 - mu_dep)).astype(np.float32)
        in_rae_dep = float(rae(y_unb, deploy_513[unb_idx].astype(np.float64)))
        print(f"   in_RAE deploy on 253   = {in_rae_dep:.4f}")
        new_blend_summary["deploy_in_rae_253"] = in_rae_dep
        new_blend_summary["deploy_w0_chemprop_aux"] = w0
        new_blend_summary["deploy_s"] = s_dep

        np.save(DATA_PROCESSED / f"te_{TAG}_new_blend.npy", deploy_513)
        new_plain = SUBMISSIONS / f"{TAG}_new_blend.csv"
        pd.DataFrame({
            "SMILES": te["smiles"].values,
            "Molecule Name": te["name"].values,
            "pEC50": deploy_513,
        }).to_csv(new_plain, index=False)
        print(f"[save] te_{TAG}_new_blend.npy")
        print(f"[save] {new_plain}")
        new_blend_summary["plain_submission"] = str(new_plain)

    # ---- Verdict ----
    beats_nb1070 = (new_blend_summary is not None
                    and new_blend_summary["mean_rae"] < NB1070_REF_RAE)
    new_blend_rae = (new_blend_summary["mean_rae"]
                     if new_blend_summary else None)

    summary = {
        "tag": TAG,
        "seeds": SEEDS,
        "n_folds": N_FOLDS,
        "feature_dim": int(X_tr.shape[1]),
        "per_seed_oof_rae": per_seed_oof_rae,
        "bag_oof_rae_on_4139": bag_oof_rae,
        "in_rae_bag_on_253": in_rae_bag,
        "in_rae_chemprop_aux_on_253": in_rae_ca,
        "in_rae_nb972_on_253": in_rae_972,
        "pearson_bag_nb972_513": pearson_full,
        "pearson_bag_chemprop_aux_513": pearson_ca,
        "pearson_bag_nb972_253": pearson_unb_972,
        "orthogonal_lt_0p95": bool(orthogonal),
        "te_bag_mean": float(te_bag.mean()),
        "te_bag_std": float(te_bag.std()),
        "new_blend_summary": new_blend_summary,
        "new_blend_mean_rae": new_blend_rae,
        "nb1070_ref_rae": NB1070_REF_RAE,
        "beats_nb1070": bool(beats_nb1070),
        "plain_submission_bag": str(plain),
        "wall_sec": round(time.time() - t0, 2),
    }
    with open(DATA_PROCESSED / f"{TAG}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_summary.json'}")
    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   per-seed OOF RAE (4139) = "
          f"{[f'{r:.4f}' for r in per_seed_oof_rae]}")
    print(f"   bag OOF RAE (4139)      = {bag_oof_rae:.4f}")
    print(f"   Pearson(bag, nb972) 513 = {pearson_full:.4f}  "
          f"({'orthogonal' if orthogonal else 'collinear'})")
    if new_blend_summary:
        print(f"   new blend mean RAE      = {new_blend_rae:.4f}  "
              f"vs nb1070 {NB1070_REF_RAE:.4f}  "
              f"({'BEATS' if beats_nb1070 else 'WORSE'})")
    print(f"   wall                    = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("per_seed_oof_rae", "bag_oof_rae_on_4139",
              "pearson_bag_nb972_513", "orthogonal_lt_0p95",
              "in_rae_bag_on_253", "new_blend_mean_rae",
              "beats_nb1070", "wall_sec"):
        print(f"  {k}: {res.get(k)}")
