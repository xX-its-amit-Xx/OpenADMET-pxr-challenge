"""nb1124 - Mid-LR very-long-train Huber LGBM variant.

Sits between chemprop_aux (NN architecture) and nb972 (n_est=10000,
lr=0.005). The nb1124 recipe:
  - LightGBM Huber (alpha=2.0)
  - n_estimators=5000   (very long but half of nb972)
  - learning_rate=0.002 (very slow, sub-nb972)
  - num_leaves=64
  - min_child_samples=30 (stricter than nb972's 20 -- less overfit per leaf)
  - early_stopping_rounds=1000 (huge patience)
  - Features: Morgan (2048) + RDKit desc (217) -> combined (2265)
  - Scaffold 5-fold CV on 4139 CRC; held-out fold = early-stop validation
  - Predict 513 test; in_RAE on 253 Phase-1 unblind.

Reports Pearson correlation against nb972 and chemprop_aux on 513.
If Pearson < 0.95 to BOTH (i.e. nb1124 is genuinely orthogonal to the
existing pool), trigger the nb1014-style bag (chemprop_aux + nb1124,
replacing nb972) plus per-quantile median across SEEDS, and write a
deploy submission.

Outputs:
  data/processed/oof_nb1124_chemprop_variant.npy
  data/processed/te_nb1124_chemprop_variant.npy
  data/processed/nb1124_summary.json
  submissions/nb1124_chemprop_variant.csv
  (conditional) submissions/nb1124_bag_perq_median.csv
  (conditional) data/processed/te_nb1124_bag_perq_median.npy
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
import lightgbm as lgb
from scipy.stats import pearsonr
from sklearn.model_selection import KFold

from pxr.data import load_train, load_test
from pxr.featurize import combined, impute
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1124"
SEED = 42
N_FOLDS = 5

PARAMS = dict(
    objective="huber",
    alpha=2.0,
    n_estimators=5000,
    learning_rate=0.002,
    num_leaves=64,
    min_child_samples=30,
    reg_lambda=0.2,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    verbose=-1,
    n_jobs=4,
)
EARLY_STOP = 1000

# Bag protocol parameters (mirrors nb1014 / nb1070).
N_BINS = 5
STRETCH_GRID = np.round(np.arange(0.80, 2.001, 0.05), 2).tolist()
BAG_SEEDS = [0, 1, 7, 42, 137]
NB1070_BAG_MEDIAN_RAE = 0.5771  # reference: per-quantile median bag on nb1014


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def in_rae(y_true, y_pred):
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    return float(np.mean(np.abs(yt - yp)) / np.mean(np.abs(yt - yt.mean())))


def load_te_or_csv(name: str, te_names: np.ndarray) -> np.ndarray:
    npy = DATA_PROCESSED / f"te_{name}.npy"
    if npy.exists():
        return np.load(npy).astype(np.float64)
    sub = pd.read_csv(SUBMISSIONS / f"{name}.csv")
    assert (sub["Molecule Name"].values == te_names).all(), (
        f"{name}: submission row order does not match test order")
    return sub["pEC50"].values.astype(np.float64)


def bin_assign(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(values, edges, right=False), 0, N_BINS - 1)


def fit_per_bin_stretch(p_train: np.ndarray, y_train: np.ndarray,
                        edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bins_tr = bin_assign(p_train, edges)
    mus = np.zeros(N_BINS, dtype=np.float64)
    ss = np.ones(N_BINS, dtype=np.float64)
    for b in range(N_BINS):
        mask = bins_tr == b
        n_b = int(mask.sum())
        if n_b < 2:
            mus[b] = float(p_train.mean())
            ss[b] = 1.0
            continue
        mu_b = float(p_train[mask].mean())
        mus[b] = mu_b
        y_b = y_train[mask]
        p_b = p_train[mask]
        best_s, best_r = 1.0, float("inf")
        for s in STRETCH_GRID:
            stretched = mu_b + s * (p_b - mu_b)
            r = float(rae(y_b, stretched))
            if r < best_r:
                best_r = r
                best_s = float(s)
        ss[b] = best_s
    return mus, ss


def apply_per_bin_stretch(p: np.ndarray, edges: np.ndarray,
                          mus: np.ndarray, ss: np.ndarray) -> np.ndarray:
    bins = bin_assign(p, edges)
    out = np.empty_like(p, dtype=np.float64)
    for b in range(N_BINS):
        mask = bins == b
        if not mask.any():
            continue
        out[mask] = mus[b] + ss[b] * (p[mask] - mus[b])
    return out


def run_one_seed_perq(seed: int, p_unb: np.ndarray, y_unb: np.ndarray
                      ) -> tuple[float, np.ndarray, list[list[float]]]:
    n = len(y_unb)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    per_fold_ss: list[list[float]] = []
    for tr_loc, va_loc in kf.split(np.arange(n)):
        p_tr, y_tr = p_unb[tr_loc], y_unb[tr_loc]
        p_va = p_unb[va_loc]
        qs = np.linspace(0.0, 1.0, N_BINS + 1)[1:-1]
        edges = np.quantile(p_tr, qs)
        mus, ss = fit_per_bin_stretch(p_tr, y_tr, edges)
        oof[va_loc] = apply_per_bin_stretch(p_va, edges, mus, ss)
        per_fold_ss.append(ss.tolist())
    pooled = float(rae(y_unb, oof))
    return pooled, oof, per_fold_ss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"=== {TAG}: mid-LR very-long-train Huber LGBM variant ===")
    print(f"  PARAMS: n_est={PARAMS['n_estimators']}, "
          f"lr={PARAMS['learning_rate']}, leaves={PARAMS['num_leaves']}, "
          f"min_child={PARAMS['min_child_samples']}, "
          f"early_stop={EARLY_STOP}")
    print("=" * 78)

    # ---- Truth / unblind ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    assert len(unb_idx) == len(y_unb) == 253

    # ---- Data ----
    tr = load_train()
    te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    print("\n[feat] computing combined features (Morgan + RDKit)...")
    X_tr = impute(combined(tr["smiles"].tolist()))
    X_te = impute(combined(te["smiles"].tolist()))
    print(f"  X_tr={X_tr.shape}  X_te={X_te.shape}")

    # ---- Scaffold CV with very-long-train + huge-patience early stop ----
    oof = np.full(n_tr, np.nan)
    best_iters: list[int] = []
    fold_raes: list[float] = []
    print(f"\n[cv] {N_FOLDS} folds, LR={PARAMS['learning_rate']}, "
          f"max_iter={PARAMS['n_estimators']}, patience={EARLY_STOP}...")
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = lgb.train(
            PARAMS,
            lgb.Dataset(X_tr[tr_idx], label=y_tr[tr_idx]),
            valid_sets=[lgb.Dataset(X_tr[va_idx], label=y_tr[va_idx])],
            callbacks=[
                lgb.early_stopping(EARLY_STOP, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        oof[va_idx] = m.predict(X_tr[va_idx], num_iteration=m.best_iteration)
        fr = rae(y_tr[va_idx], oof[va_idx])
        fold_raes.append(fr)
        best_iters.append(int(m.best_iteration or PARAMS["n_estimators"]))
        elapsed = time.time() - t0
        print(f"  fold {fold+1}  best_iter={best_iters[-1]:5d}  "
              f"RAE={fr:.4f}  elapsed={elapsed:6.1f}s", flush=True)

    oof_rae = float(rae(y_tr, oof))
    mean_best = int(np.mean(best_iters))
    print(f"\n[cv] OOF RAE = {oof_rae:.4f}")
    print(f"[cv] mean best_iteration across folds = {mean_best}")

    # ---- Final fit on full 4139 train ----
    print(f"\n[fit] final fit on full 4139, n_estimators={mean_best}...")
    final_params = dict(PARAMS, n_estimators=mean_best)
    m_final = lgb.train(
        final_params,
        lgb.Dataset(X_tr, label=y_tr),
        callbacks=[lgb.log_evaluation(-1)],
    )
    te_preds = np.clip(
        m_final.predict(X_te),
        y_tr.min() - 0.5, y_tr.max() + 0.5,
    ).astype(np.float64)

    # ---- In-RAE on 253 ----
    in_r = in_rae(y_unb, te_preds[unb_idx])
    ratio = te_preds.std() / oof.std() if oof.std() > 0 else 0.0
    print(f"[te] mean={te_preds.mean():.3f}  std={te_preds.std():.3f}  "
          f"ratio(te/oof)={ratio:.2f}")
    print(f"[te] in_RAE(253) = {in_r:.4f}")

    # ---- Pearson vs nb972 and chemprop_aux on 513 ----
    te_names = te["name"].values
    te_nb972 = load_te_or_csv("nb972_long_train", te_names)
    te_chem = load_te_or_csv("chemprop_aux", te_names)
    pear_nb972, _ = pearsonr(te_preds, te_nb972)
    pear_chem, _ = pearsonr(te_preds, te_chem)
    print(f"\n[orth] Pearson(nb1124, nb972_long_train) = {pear_nb972:.4f}")
    print(f"[orth] Pearson(nb1124, chemprop_aux)      = {pear_chem:.4f}")

    # ---- Persist standalone artefacts ----
    np.save(DATA_PROCESSED / f"oof_{TAG}_chemprop_variant.npy", oof)
    np.save(DATA_PROCESSED / f"te_{TAG}_chemprop_variant.npy", te_preds)
    plain = SUBMISSIONS / f"{TAG}_chemprop_variant.csv"
    pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te_names,
        "pEC50": te_preds,
    }).to_csv(plain, index=False)
    print(f"[save] {plain}")

    # ---- Trigger bag if orthogonal enough ----
    trigger = (pear_nb972 < 0.95) and (pear_chem < 0.95)
    bag_summary: dict | None = None
    bag_path: str | None = None
    bag_rae: float | None = None
    bag_te_path: str | None = None
    if trigger:
        print("\n" + "-" * 78)
        print("[bag] BOTH Pearson < 0.95 -> trigger nb1014-style bag "
              "(chemprop_aux + nb1124) + per-quantile median across seeds")
        print("-" * 78)
        # Step 1: SLSQP w0 + scalar stretch bag on the 253 (nb1014 protocol)
        # is conceptually identical; we go straight to a 50/50 anchor blend
        # (chemprop_aux replaces 0.5 weight of nb972) and apply per-quantile
        # bin stretch from nb1070 across 5 seeds, then median-bag.
        from scipy.optimize import minimize

        P_unb = np.column_stack([te_chem[unb_idx], te_preds[unb_idx]])

        # SLSQP w0 over the WHOLE 253 just to get a single anchor blend.
        def loss(w):
            return float(np.sum((w[0] * P_unb[:, 0]
                                 + (1.0 - w[0]) * P_unb[:, 1] - y_unb) ** 2))
        res = minimize(loss, np.array([0.5]),
                       method="SLSQP", bounds=[(0.0, 1.0)],
                       options={"ftol": 1e-10, "maxiter": 500})
        w0 = float(res.x[0])
        print(f"[bag] SLSQP w0(chemprop_aux) = {w0:.4f}  "
              f"w1(nb1124) = {1.0 - w0:.4f}")
        blend_unb = w0 * P_unb[:, 0] + (1.0 - w0) * P_unb[:, 1]
        blend_513 = w0 * te_chem + (1.0 - w0) * te_preds
        anchor_in_rae = float(rae(y_unb, blend_unb))
        print(f"[bag] anchor blend in_RAE(253) = {anchor_in_rae:.4f}")

        # Per-quantile stretch across BAG_SEEDS, median over seed-OOFs.
        oof_stack = np.zeros((len(BAG_SEEDS), len(y_unb)), dtype=np.float64)
        per_seed_rae: list[float] = []
        per_seed_records: list[dict] = []
        for i, seed in enumerate(BAG_SEEDS):
            pooled, oof_seed, per_fold_ss = run_one_seed_perq(
                seed, blend_unb, y_unb)
            oof_stack[i] = oof_seed
            per_seed_rae.append(pooled)
            per_seed_records.append({
                "seed": seed,
                "pooled_rae": pooled,
                "per_fold_ss": per_fold_ss,
            })
            ss_mean = np.array(per_fold_ss).mean(axis=0).tolist()
            ss_str = ",".join(f"{x:.2f}" for x in ss_mean)
            print(f"   seed {seed:>3d}: pooled_RAE={pooled:.4f}  "
                  f"ss_mean=[{ss_str}]")

        bagged_median_oof = np.median(oof_stack, axis=0)
        bag_rae = float(rae(y_unb, bagged_median_oof))
        print(f"[bag] MEDIAN-bag OOF RAE on 253 = {bag_rae:.4f}  "
              f"(nb1070 median-bag reference = {NB1070_BAG_MEDIAN_RAE:.4f})")

        # Deploy: per seed, fit per-bin (mu, s_b) on all 253; mean ss per seed;
        # apply to 513 anchor blend; median across seeds.
        deploy_stack = np.zeros((len(BAG_SEEDS), len(blend_513)),
                                dtype=np.float64)
        per_seed_deploy_ss: list[list[float]] = []
        qs_all = np.linspace(0.0, 1.0, N_BINS + 1)[1:-1]
        edges_all = np.quantile(blend_unb, qs_all)
        mus_all, _ = fit_per_bin_stretch(blend_unb, y_unb, edges_all)
        for i, seed in enumerate(BAG_SEEDS):
            rec = per_seed_records[i]
            ss_seed = np.array(rec["per_fold_ss"]).mean(axis=0)
            per_seed_deploy_ss.append(ss_seed.tolist())
            deploy_stack[i] = apply_per_bin_stretch(
                blend_513, edges_all, mus_all, ss_seed)

        deploy_513 = np.median(deploy_stack, axis=0).astype(np.float32)
        bag_in_rae_deploy = float(
            rae(y_unb, deploy_513[unb_idx].astype(np.float64)))
        print(f"[bag] deploy_513 MEDIAN-bag mean={deploy_513.mean():.3f}  "
              f"std={deploy_513.std():.3f}  "
              f"in-sample 253 = {bag_in_rae_deploy:.4f}")

        bag_te_path = str(DATA_PROCESSED / f"te_{TAG}_bag_perq_median.npy")
        np.save(bag_te_path, deploy_513)
        bag_path_p = SUBMISSIONS / f"{TAG}_bag_perq_median.csv"
        pd.DataFrame({
            "SMILES": te["smiles"].values,
            "Molecule Name": te_names,
            "pEC50": deploy_513,
        }).to_csv(bag_path_p, index=False)
        bag_path = str(bag_path_p)
        print(f"[save] {bag_path}")

        bag_summary = {
            "w0_chemprop_aux": w0,
            "w1_nb1124": float(1.0 - w0),
            "anchor_blend_in_rae_253": anchor_in_rae,
            "per_seed_rae": per_seed_rae,
            "per_seed_mean_rae": float(np.mean(per_seed_rae)),
            "per_seed_std_rae": float(np.std(per_seed_rae)),
            "bag_median_oof_rae_253": bag_rae,
            "in_rae_deploy_median_on_253": bag_in_rae_deploy,
            "per_seed_deploy_ss": per_seed_deploy_ss,
            "deploy_te_path": bag_te_path,
            "deploy_submission": bag_path,
            "nb1070_reference": NB1070_BAG_MEDIAN_RAE,
            "beats_nb1070": bool(bag_rae < NB1070_BAG_MEDIAN_RAE),
        }
    else:
        print("\n[bag] SKIP (Pearson >= 0.95 to at least one of "
              "nb972 / chemprop_aux). nb1124 not sufficiently orthogonal.")

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "params": {k: v for k, v in PARAMS.items() if k != "verbose"},
        "early_stopping_rounds": EARLY_STOP,
        "fold_best_iters": best_iters,
        "fold_raes": [float(x) for x in fold_raes],
        "mean_best_iter": mean_best,
        "oof_rae_4139": oof_rae,
        "in_rae_253": in_r,
        "test_mean": float(te_preds.mean()),
        "test_std": float(te_preds.std()),
        "te_oof_std_ratio": float(ratio),
        "pearson_nb972_long_train": float(pear_nb972),
        "pearson_chemprop_aux": float(pear_chem),
        "orthogonality_trigger": bool(trigger),
        "bag": bag_summary,
        "plain_submission": str(plain),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("oof_rae_4139", "in_rae_253", "test_std",
              "pearson_nb972_long_train", "pearson_chemprop_aux",
              "orthogonality_trigger", "plain_submission"):
        print(f"  {k}: {res.get(k)}")
    if res.get("bag"):
        b = res["bag"]
        print("  bag.bag_median_oof_rae_253:", b.get("bag_median_oof_rae_253"))
        print("  bag.beats_nb1070:          ", b.get("beats_nb1070"))
        print("  bag.deploy_submission:     ", b.get("deploy_submission"))
