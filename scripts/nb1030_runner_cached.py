"""nb1030 cached runner -- skip Mordred recompute, use cached features.

Loads cached Mordred features from C:/pxr_artifacts/nb1030/ and runs only
the LGBM Huber + nb1014 bag stages from nb1030_mordred_lgbm.py.
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

if not hasattr(np, "product"):
    np.product = np.prod  # type: ignore[attr-defined]

import lightgbm as lgb

from pxr.chem import bemis_murcko
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

# import shared protocol bits from the original script
import importlib.util
spec = importlib.util.spec_from_file_location(
    "nb1030_orig", os.path.join(os.path.dirname(__file__), "nb1030_mordred_lgbm.py")
)
nb1030_orig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nb1030_orig)

TAG = nb1030_orig.TAG
SEED = nb1030_orig.SEED
N_FOLDS = nb1030_orig.N_FOLDS
LGB_PARAMS = nb1030_orig.LGB_PARAMS
EARLY_STOP = nb1030_orig.EARLY_STOP
ARTIFACT_DIR = nb1030_orig.ARTIFACT_DIR
run_nb1014_bag = nb1030_orig.run_nb1014_bag


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} CACHED -- skip Mordred, run LGBM + nb1014 bag")
    print("=" * 78)

    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr); n_te = len(te)
    print(f"[load] n_train={n_tr}  n_test={n_te}")

    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # ---- LOAD CACHED Mordred ----
    print(f"\n[cache] loading cached Mordred features from {ARTIFACT_DIR}...")
    X_tr = np.load(ARTIFACT_DIR / "X_mordred_train.npy")
    X_te = np.load(ARTIFACT_DIR / "X_mordred_test.npy")
    with open(ARTIFACT_DIR / "feature_names.json") as f:
        fmeta = json.load(f)
    mord_notes = fmeta.get("notes", {"backend": fmeta.get("backend", "cached")})
    print(f"[cache] X_tr={X_tr.shape}  X_te={X_te.shape}  backend={mord_notes.get('backend')}")
    assert X_tr.shape[0] == n_tr and X_te.shape[0] == n_te, "shape mismatch"

    # ---- Scaffold CV ----
    oof = np.full(n_tr, np.nan)
    best_iters, fold_raes = [], []
    print(f"\n[cv] LGBM Huber a=2.0 n_est=2000 nl=128 lr=0.025  "
          f"({N_FOLDS}-fold scaffold)")
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = lgb.train(
            LGB_PARAMS,
            lgb.Dataset(X_tr[tr_idx], label=y_tr[tr_idx]),
            valid_sets=[lgb.Dataset(X_tr[va_idx], label=y_tr[va_idx])],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                       lgb.log_evaluation(-1)],
        )
        oof[va_idx] = m.predict(X_tr[va_idx], num_iteration=m.best_iteration)
        fr = rae(y_tr[va_idx], oof[va_idx])
        fold_raes.append(fr)
        best_iters.append(int(m.best_iteration or LGB_PARAMS["n_estimators"]))
        print(f"   fold {fold+1}  best_iter={best_iters[-1]:5d}  RAE={fr:.4f}  "
              f"elapsed={time.time()-t0:6.1f}s", flush=True)
    oof_rae = float(rae(y_tr, oof))
    mean_best = int(np.mean(best_iters))
    print(f"[cv] OOF RAE = {oof_rae:.4f}  mean_best_iter = {mean_best}")

    # ---- Final fit on all train ----
    print(f"\n[final] full-train fit, n_est={mean_best}...")
    final_params = dict(LGB_PARAMS, n_estimators=mean_best)
    m_final = lgb.train(final_params, lgb.Dataset(X_tr, label=y_tr),
                        callbacks=[lgb.log_evaluation(-1)])
    te_preds = np.clip(m_final.predict(X_te),
                       y_tr.min() - 0.5, y_tr.max() + 0.5).astype(np.float32)

    # ---- in_RAE on 253 ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    in_r = float(rae(y_unb, te_preds[unb_idx].astype(np.float64)))
    print(f"[deploy] te mean/std = {te_preds.mean():.3f}/{te_preds.std():.3f}  "
          f"in_RAE(253) = {in_r:.4f}")

    # ---- Pearson vs nb972 ----
    te_nb972 = np.load(DATA_PROCESSED / "te_nb972_long_train.npy").astype(np.float64)
    pearson_972 = float(np.corrcoef(te_preds.astype(np.float64), te_nb972)[0, 1])
    print(f"[corr] Pearson(te_nb1030, te_nb972) = {pearson_972:.4f}")
    try:
        te_chemprop = np.load(DATA_PROCESSED / "te_chemprop_aux.npy").astype(np.float64)
        pearson_cp = float(np.corrcoef(te_preds.astype(np.float64), te_chemprop)[0, 1])
        print(f"[corr] Pearson(te_nb1030, te_chemprop_aux) = {pearson_cp:.4f}")
    except FileNotFoundError:
        pearson_cp = None

    # ---- Save base submission ----
    np.save(DATA_PROCESSED / f"oof_{TAG}.npy", oof)
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", te_preds)
    base_sub = SUBMISSIONS / f"{TAG}_mordred_lgbm.csv"
    pd.DataFrame({"SMILES": te["smiles"].values,
                  "Molecule Name": te["name"].values,
                  "pEC50": te_preds}).to_csv(base_sub, index=False)
    print(f"[save] te_{TAG}.npy, oof_{TAG}.npy, {base_sub}")

    # ---- Conditional bag ----
    bag_summary = {"ran": False, "reason": f"pearson_{pearson_972:.4f}_>=_0.95"}
    if pearson_972 < 0.95:
        bag_summary = run_nb1014_bag(te_preds, te["name"].values,
                                     te["smiles"].values, unb_idx, y_unb)
    else:
        print(f"\n[skip] Pearson {pearson_972:.4f} >= 0.95 -- skip nb1014 bag")

    summary = {
        "tag": TAG,
        "cached_run": True,
        "mordred_notes": mord_notes,
        "n_features_kept": int(X_tr.shape[1]),
        "lgb_params": {k: v for k, v in LGB_PARAMS.items() if k != "verbose"},
        "fold_best_iters": best_iters,
        "fold_raes": [float(x) for x in fold_raes],
        "mean_best_iter": mean_best,
        "oof_rae": oof_rae,
        "in_rae_253": in_r,
        "test_mean": float(te_preds.mean()),
        "test_std": float(te_preds.std()),
        "pearson_nb972": pearson_972,
        "pearson_chemprop_aux": pearson_cp,
        "base_submission": str(base_sub),
        "nb1014_bag": bag_summary,
        "wall_sec": round(time.time() - t0, 2),
    }
    # Save the JSON summary in BOTH locations: DATA_PROCESSED for ladder pipeline
    # AND ARTIFACT_DIR/nb1030_summary.json as requested.
    with open(DATA_PROCESSED / f"{TAG}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(ARTIFACT_DIR / f"{TAG}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {DATA_PROCESSED / f'{TAG}_summary.json'}")
    print(f"[save] {ARTIFACT_DIR / f'{TAG}_summary.json'}")
    print(f"\n=== {TAG} done in {time.time()-t0:.1f}s ===")
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("n_features_kept", "oof_rae", "in_rae_253",
              "pearson_nb972", "pearson_chemprop_aux",
              "base_submission"):
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
