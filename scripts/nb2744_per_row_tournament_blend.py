"""nb2744 -- Per-row tournament: pick best anchor per row by predicted error.

NEW PARADIGM:
    Multi-anchor routing where each row of the 513 test set is assigned to
    the anchor whose predicted absolute error is the smallest. For each
    anchor we train a separate LGBM "error predictor" on the 253 unblind
    substrate (the only label-bearing substrate where all anchor OOFs
    coexist; chemprop_aux is the lone anchor with a 4139-row train OOF,
    so a 4139-substrate joint regression is not feasible across all 4
    anchors). Within each outer scaffold fold we re-fit every error
    predictor on that fold's train portion, then route the held-out
    rows by argmin of predicted error.

    Distinction from prior post-hoc-blend cycles:
      - SLSQP pyramid (nb2095/nb2171): GLOBAL convex weights, same blend
        for every row.
      - Routing scripts on chemprop_aux residual (nb472/nb481/...):  one
        anchor + a residual learner over features.
      - This script: 4 anchors + 4 error predictors, hard per-row argmin
        (no convex blend across anchors).

PROTOCOL:
    1. Anchors:
         A = chemprop_aux (PRE-clean, oof_unb=nb1133_chemprop_aux_pred_oof,
             te_513=te_chemprop_aux).
         B = nb2240_K20 (oof_unb=nb2240_mean_bag_oof_K20,
             te_513=te_nb2240_K20).
         C = nb2490 counter_clean (oof_unb=nb2490_pred_oof,
             te_513=te_nb2490).
         D = nb1191 (te_513=te_nb1191; oof_unb reconstructed via
             nb1191 deploy weights on per-seed pyramid OOFs; if any
             component is missing we omit D and report 3 anchors).
    2. For each kf_seed in {1001..1005}:
         scaffold-K-fold on 253 unblind:
           in each outer fold (train_loc, val_loc):
             for each anchor a:
               y_err_a = |y_unb[train_loc] - anchor_oof_a[train_loc]|
               LGBM error predictor fit on X_117_unb[train_loc]
                 -> predict on X_117_unb[val_loc] => err_hat_a[val]
             pick_a[val] = argmin_a err_hat_a[val]
             oof_pred[val] = anchor_oof_pick_a[val]
       pooled RAE = rae(y_unb, oof_pred).
    3. Mean across 5 kf_seeds -> mean_rae.
    4. Deploy: refit all error predictors on ALL 253; predict err_hat_a
       on X_117_te (513,); pick_a[i] = argmin; te_pred[i] = te_a[i].

GATE:
    mean_rae < 0.4570 -> "PROMOTE"
    mean_rae < 0.4598 -> "MARGINAL_BEAT"
    else            -> "FAIL"

Outputs:
    scripts/nb2744_per_row_tournament_blend.py
    data/processed/nb2744_summary.json
    data/processed/nb2744_pred_oof.npy    (253,) float32
    data/processed/te_nb2744.npy          (513,) float32
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
import lightgbm as lgb

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2744"

# ---- CV protocol ----
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# ---- Gates ----
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

# ---- Paths ----
X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"

# ---- Anchors registry ----
# Each entry: (name, oof_unb_path, te_path)
ANCHOR_DEFS = [
    (
        "chemprop_aux",
        DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy",
        DATA_PROCESSED / "te_chemprop_aux.npy",
    ),
    (
        "nb2240_K20",
        DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy",
        DATA_PROCESSED / "te_nb2240_K20.npy",
    ),
    (
        "nb2490_counter_clean",
        DATA_PROCESSED / "nb2490_pred_oof.npy",
        DATA_PROCESSED / "te_nb2490.npy",
    ),
    (
        "nb1191",
        DATA_PROCESSED / "nb1191_pred_oof.npy",  # may not exist; handled below
        DATA_PROCESSED / "te_nb1191.npy",
    ),
]

# ---- LGBM error-predictor hyperparameters ----
LGBM_PARAMS = dict(
    objective="regression",
    metric="mae",
    learning_rate=0.05,
    num_leaves=15,
    max_depth=-1,
    min_data_in_leaf=10,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=5,
    n_estimators=300,
    verbosity=-1,
    n_jobs=1,
)


def _fit_err_predictor(X_tr: np.ndarray, err_tr: np.ndarray, seed: int) -> lgb.LGBMRegressor:
    params = dict(LGBM_PARAMS)
    params["random_state"] = int(seed)
    mdl = lgb.LGBMRegressor(**params)
    mdl.fit(X_tr, err_tr)
    return mdl


def _load_anchor(name: str, oof_path: Path, te_path: Path, n_unb: int, n_te: int):
    """Return (oof_unb, te) or (None, None) if anchor missing/unusable.

    nb1191 has no pred_oof.npy in the repo (the deploy notebook never saved
    one).  We attempt a fallback: reconstruct via nb1191 deploy weights
    over component pyramid OOFs on the unblind set.  If any component is
    missing we degrade gracefully -- skip this anchor.
    """
    te_arr = None
    if te_path.exists():
        try:
            te_arr = np.load(te_path).astype(np.float64)
            assert te_arr.shape == (n_te,), f"{name} te shape {te_arr.shape}"
        except Exception as e:
            print(f"   [skip] {name}: te load failed: {e}")
            return None, None
    else:
        print(f"   [skip] {name}: te missing at {te_path}")
        return None, None

    oof_arr = None
    if oof_path.exists():
        try:
            oof_arr = np.load(oof_path).astype(np.float64)
            assert oof_arr.shape == (n_unb,), f"{name} oof shape {oof_arr.shape}"
            return oof_arr, te_arr
        except Exception as e:
            print(f"   [skip] {name}: oof load failed: {e}")

    # nb1191 fallback: reconstruct from pyramid components if missing.
    if name == "nb1191":
        sum_path = DATA_PROCESSED / "nb1191_summary.json"
        if not sum_path.exists():
            print(f"   [skip] {name}: no oof and no summary for fallback")
            return None, None
        with open(sum_path) as f:
            sm = json.load(f)
        weights = {w["name"]: float(w["w"]) for w in sm.get("deploy_weights", [])}
        s = float(sm.get("deploy_s", 1.0))
        comp_paths = {
            "chemprop_aux": DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy",
            "nb1150": DATA_PROCESSED / "nb1150_pred_oof.npy",
            "nb1158_K32": DATA_PROCESSED / "nb1158_mean_bag_oof_K32.npy",
            "nb2112_K28": DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy",
        }
        comps = {}
        for cn, cp in comp_paths.items():
            if cp.exists():
                a = np.load(cp).astype(np.float64)
                if a.shape == (n_unb,):
                    comps[cn] = a
        if set(weights.keys()) - set(comps.keys()):
            print(
                f"   [skip] {name}: fallback components missing: "
                f"{sorted(set(weights.keys()) - set(comps.keys()))}"
            )
            return None, None
        wsum = sum(weights.get(k, 0.0) for k in comps.keys())
        if wsum <= 0:
            print(f"   [skip] {name}: fallback weights sum to 0")
            return None, None
        blend = np.zeros(n_unb, dtype=np.float64)
        for cn, w in weights.items():
            blend += (w / wsum) * comps[cn]
        mu = float(blend.mean())
        oof_arr = mu + s * (blend - mu)
        print(f"   [reconstructed] nb1191 oof from pyramid components (s={s})")
        return oof_arr, te_arr

    print(f"   [skip] {name}: no usable oof")
    return None, None


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Per-row tournament blend (error-predictor routing)")
    print("=" * 78)

    # ---- Load test set + scaffolds + truth ----
    te = load_test()
    n_test = len(te)
    smi_col = "smiles" if "smiles" in te.columns else "SMILES"
    te_smiles = te[smi_col].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_uniq_scaf = len({s for s in unb_scaffolds if s})
    print(f"[load] n_test={n_test}  n_unb={n_unb}  unique_scaf={n_uniq_scaf}")

    # ---- Load X_117 features ----
    X_unb = np.load(X117_UNB_PATH).astype(np.float32)
    X_te = np.load(X117_TE_PATH).astype(np.float32)
    assert X_unb.shape == (n_unb, 117), f"X_unb shape {X_unb.shape}"
    assert X_te.shape == (n_test, 117), f"X_te shape {X_te.shape}"
    print(f"[feat] X_unb={X_unb.shape}  X_te={X_te.shape}")

    # ---- Load anchors ----
    print("\n[anchors] loading + validating ----")
    anchor_names = []
    anchor_oof = []   # list of (n_unb,) arrays
    anchor_te = []    # list of (n_te,) arrays
    anchor_meta = []
    for name, oof_p, te_p in ANCHOR_DEFS:
        oof_arr, te_arr = _load_anchor(name, oof_p, te_p, n_unb, n_test)
        if oof_arr is None or te_arr is None:
            continue
        oof_rae = float(rae(y_unb, oof_arr))
        anchor_names.append(name)
        anchor_oof.append(oof_arr)
        anchor_te.append(te_arr)
        anchor_meta.append({
            "name": name,
            "oof_path": str(oof_p),
            "te_path": str(te_p),
            "oof_rae_unb": oof_rae,
            "te_mean": float(te_arr.mean()),
            "te_std": float(te_arr.std()),
        })
        print(f"   [keep] {name:25s}  oof_RAE={oof_rae:.4f}")
    n_anc = len(anchor_names)
    if n_anc < 2:
        raise RuntimeError(f"Not enough anchors: got {n_anc}, need >= 2")
    anchor_oof_M = np.column_stack(anchor_oof)  # (n_unb, n_anc)
    anchor_te_M = np.column_stack(anchor_te)    # (n_te, n_anc)
    print(f"[anchors] n_anchors_used = {n_anc}")

    # ---- Scaffold 5-fold CV across kf_seeds with per-row tournament ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV  seeds={KF_SEEDS}\n"
          f"  per outer fold: fit {n_anc} LGBM error predictors on train portion,\n"
          f"  per val row pick anchor with smallest predicted abs error.")
    print("-" * 78)

    per_seed = []
    all_oofs = []
    pick_counts_total = np.zeros(n_anc, dtype=np.int64)

    for kf_seed in KF_SEEDS:
        ts = time.time()
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        oof_pred = np.full(n_unb, np.nan, dtype=np.float64)
        pick_seed = np.full(n_unb, -1, dtype=np.int64)
        for fi, (tr_loc, va_loc) in enumerate(splits):
            err_hat_va = np.zeros((len(va_loc), n_anc), dtype=np.float64)
            for ai in range(n_anc):
                err_tr = np.abs(y_unb[tr_loc] - anchor_oof_M[tr_loc, ai])
                mdl = _fit_err_predictor(
                    X_unb[tr_loc], err_tr,
                    seed=1000 + kf_seed + 10 * fi + ai,
                )
                err_hat_va[:, ai] = mdl.predict(X_unb[va_loc])
            # per-row tournament: pick anchor with lowest predicted error
            pick = np.argmin(err_hat_va, axis=1)
            pick_seed[va_loc] = pick
            # Place chosen anchor's OOF as the prediction
            oof_pred[va_loc] = anchor_oof_M[va_loc, pick]
        assert not np.isnan(oof_pred).any(), "oof_pred has NaN"
        # gentle clip
        oof_pred = np.clip(oof_pred, 3.0, 8.0)
        pooled = float(rae(y_unb, oof_pred))
        seed_pick_counts = np.bincount(pick_seed, minlength=n_anc)
        pick_counts_total += seed_pick_counts
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": pooled,
            "pick_counts": {anchor_names[i]: int(seed_pick_counts[i]) for i in range(n_anc)},
            "wall_sec": round(time.time() - ts, 2),
        })
        all_oofs.append(oof_pred)
        pick_str = " ".join(
            f"{anchor_names[i]}={int(seed_pick_counts[i])}" for i in range(n_anc)
        )
        print(f"   seed={kf_seed}  pooled_RAE={pooled:.4f}  picks=[{pick_str}]  "
              f"wall={time.time()-ts:.1f}s")

    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    final_oof_rae = float(rae(y_unb, mean_oof))
    pooled_rae_mean = float(np.mean([r["pooled_rae"] for r in per_seed]))
    pooled_rae_std = float(np.std([r["pooled_rae"] for r in per_seed]))
    print(f"\n[cv] pooled RAE mean across seeds = {pooled_rae_mean:.4f} "
          f"(+/- {pooled_rae_std:.4f})")
    print(f"[cv] RAE of mean-of-seed OOFs      = {final_oof_rae:.4f}")
    rae_best_anchor = min(m["oof_rae_unb"] for m in anchor_meta)
    print(f"[cv] best anchor stand-alone        = {rae_best_anchor:.4f}")
    print(f"[cv] delta vs best anchor           = {pooled_rae_mean - rae_best_anchor:+.4f}")

    # ---- Deploy ----
    print("\n" + "-" * 78)
    print("DEPLOY: refit all error predictors on ALL 253 -> route 513 test rows")
    print("-" * 78)
    err_hat_te = np.zeros((n_test, n_anc), dtype=np.float64)
    for ai in range(n_anc):
        err_all = np.abs(y_unb - anchor_oof_M[:, ai])
        mdl = _fit_err_predictor(X_unb, err_all, seed=4242 + ai)
        err_hat_te[:, ai] = mdl.predict(X_te)
    pick_te = np.argmin(err_hat_te, axis=1)
    deploy_te = anchor_te_M[np.arange(n_test), pick_te]
    deploy_te = np.clip(deploy_te, 3.0, 8.0).astype(np.float32)
    te_unb_in_rae = float(rae(y_unb, deploy_te[unb_idx].astype(np.float64)))
    pick_te_counts = np.bincount(pick_te, minlength=n_anc)
    print(f"[deploy] te(513) mean={deploy_te.mean():.3f}  std={deploy_te.std():.3f}")
    print(f"[deploy] te[unb_idx] in_RAE={te_unb_in_rae:.4f}  (in-sample refit)")
    print("[deploy] anchor pick counts on 513:")
    for i in range(n_anc):
        print(f"   {anchor_names[i]:25s}  {int(pick_te_counts[i]):4d}  "
              f"({100.0 * pick_te_counts[i] / n_test:.1f}%)")

    # ---- Gate ----
    if pooled_rae_mean < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif pooled_rae_mean < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "-" * 78)
    print("GATE EVALUATION")
    print("-" * 78)
    print(f"   mean_rae       = {pooled_rae_mean:.4f}")
    print(f"   gate PROMOTE   = < {GATE_PROMOTE}")
    print(f"   gate MARGINAL  = < {GATE_MARGINAL}")
    print(f"   verdict        = {verdict}")

    # ---- Save artefacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, mean_oof.astype(np.float32))
    np.save(te_path, deploy_te)
    print(f"\n[save] {oof_path}")
    print(f"[save] {te_path}")

    summary = {
        "tag": TAG,
        "method": (
            "Per-row tournament blend: train LGBM error predictor for each "
            "anchor on (X_117_unb, |y_unb - anchor_oof_unb|) under nested "
            "scaffold-CV; per row pick anchor with lowest predicted abs error."
        ),
        "paradigm": "per_row_argmin_routing_via_per_anchor_error_predictors",
        "anchors_used": anchor_names,
        "anchor_meta": anchor_meta,
        "n_anchors": n_anc,
        "lgbm_err_pred_params": LGBM_PARAMS,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": n_uniq_scaf,
        "feat_dim": int(X_unb.shape[1]),
        "per_seed": per_seed,
        "pick_counts_total_oof": {
            anchor_names[i]: int(pick_counts_total[i]) for i in range(n_anc)
        },
        "pick_counts_deploy_te": {
            anchor_names[i]: int(pick_te_counts[i]) for i in range(n_anc)
        },
        "mean_rae": pooled_rae_mean,
        "pooled_rae_std_seeds": pooled_rae_std,
        "rae_of_mean_of_seed_oofs": final_oof_rae,
        "rae_best_anchor_standalone": rae_best_anchor,
        "delta_vs_best_anchor": pooled_rae_mean - rae_best_anchor,
        "te_unb_rae_in_sample": te_unb_in_rae,
        "te_deploy_mean": float(deploy_te.mean()),
        "te_deploy_std": float(deploy_te.std()),
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "verdict": verdict,
        "promote": bool(verdict == "PROMOTE"),
        "marginal_beat": bool(verdict == "MARGINAL_BEAT"),
        "oof_path": str(oof_path),
        "te_path": str(te_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   anchors_used               = {anchor_names}")
    print(f"   mean_rae (5 seeds)         = {pooled_rae_mean:.4f} "
          f"(+/- {pooled_rae_std:.4f})")
    print(f"   delta vs best anchor       = {pooled_rae_mean - rae_best_anchor:+.4f}")
    print(f"   verdict                    = {verdict}")
    print(f"   wall                       = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae",
        "pooled_rae_std_seeds",
        "rae_of_mean_of_seed_oofs",
        "rae_best_anchor_standalone",
        "delta_vs_best_anchor",
        "verdict",
        "te_unb_rae_in_sample",
        "anchors_used",
    ):
        print(f"  {k}: {res.get(k)}")
