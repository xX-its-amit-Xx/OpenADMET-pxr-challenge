"""nb1093 -- Mordred AS A FEATURE (not a prediction) inside nb972 long-train recipe.

Hypothesis: post-hoc blending of Mordred-as-prediction (nb1030 / nb1014-bag)
plateaued because the model never *saw* the Mordred axes during fitting.
Concatenating Morgan(2048) + RDKit-desc(217) + Mordred(1533) = 4798-D
should let the LGBM tree splits exploit Mordred interactions in the hidden
representation.  If the resulting predictor is also <0.95-correlated with
nb972 it becomes a fresh orthogonal anchor for the nb1014-style bag protocol.

Recipe (mirrors nb972 fast-variant):
  - LightGBM Huber alpha=2.0, n_estimators=2000, num_leaves=128, lr=0.005,
    early_stopping_rounds=500.
  - Features: combined(2265) + cached Mordred(1533) -> 4798.
  - Scaffold 5-fold CV; mean best_iter rolled into full-data final fit.
  - Predict 513; report in_RAE(253) and Pearson vs chemprop_aux & nb972.
  - Promote-to-anchor if Pearson(nb972) < 0.95 OR in_RAE < 0.65.

Wall-time target < 15 min (Mordred matrix re-used from C:/pxr_artifacts/nb1030).
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
import lightgbm as lgb

from pxr.chem import bemis_murcko
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1093"
SEED = 42
N_FOLDS = 5

MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")
ARTIFACT_DIR = Path("C:/pxr_artifacts/nb1093")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# nb972 fast slow-LR Huber recipe -- max 2000 trees (not 10000) to fit budget.
PARAMS = dict(
    objective="huber",
    alpha=2.0,
    n_estimators=2000,
    learning_rate=0.005,
    num_leaves=128,
    min_child_samples=20,
    reg_lambda=0.2,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    verbose=-1,
    n_jobs=4,
)
EARLY_STOP = 500

# Reference benchmarks for verdict
NB972_IN_RAE = 0.6898       # nb972 in_RAE(253) -- two-regime calibration upper bound
NB1070_HONEST = 0.5771      # nb1070 median-bag honest cross-fit (best current anchor blend)


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Mordred AS A FEATURE inside nb972 long-train recipe")
    print("=" * 78)

    # ---- truth / unblind ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    assert len(unb_idx) == len(y_unb) == 253, "unblind shape mismatch"

    # ---- data ----
    tr = load_train()
    te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    print(f"[load] n_train={n_tr}  n_test={len(te)}")

    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # ---- features ----
    print("[feat] computing combined Morgan(2048)+RDKit(217)=2265 ...")
    t_c = time.time()
    X_tr_c = impute(combined(tr["smiles"].tolist())).astype(np.float32)
    X_te_c = impute(combined(te["smiles"].tolist())).astype(np.float32)
    print(f"[feat] combined: X_tr={X_tr_c.shape}  X_te={X_te_c.shape}  "
          f"({time.time()-t_c:.1f}s)")

    # ---- cached Mordred (nb1030 artifacts) ----
    mtr_p = MORDRED_DIR / "X_mordred_train.npy"
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not (mtr_p.exists() and mte_p.exists()):
        raise FileNotFoundError(
            f"Mordred cache missing -- run nb1030 first ({MORDRED_DIR})"
        )
    X_tr_m = np.load(mtr_p).astype(np.float32)
    X_te_m = np.load(mte_p).astype(np.float32)
    assert X_tr_m.shape[0] == n_tr and X_te_m.shape[0] == len(te), \
        f"Mordred row mismatch: {X_tr_m.shape} {X_te_m.shape}"
    print(f"[feat] Mordred (cached): X_tr_m={X_tr_m.shape}  X_te_m={X_te_m.shape}")

    # ---- concat: 2265 + 1533 = 3798 expected; tolerate cached width ----
    X_tr = np.concatenate([X_tr_c, X_tr_m], axis=1)
    X_te = np.concatenate([X_te_c, X_te_m], axis=1)
    n_feat = X_tr.shape[1]
    print(f"[feat] concat: X_tr={X_tr.shape}  X_te={X_te.shape}  total_feat={n_feat}")

    # ---- scaffold CV with slow-LR long-train + early stop ----
    oof = np.full(n_tr, np.nan)
    best_iters: list[int] = []
    fold_raes: list[float] = []
    print(f"\n[cv] LGBM Huber a=2.0 n_est={PARAMS['n_estimators']} "
          f"nl={PARAMS['num_leaves']} lr={PARAMS['learning_rate']} "
          f"patience={EARLY_STOP} ({N_FOLDS}-fold scaffold)")
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
        fr = float(rae(y_tr[va_idx], oof[va_idx]))
        fold_raes.append(fr)
        best_iters.append(int(m.best_iteration or PARAMS["n_estimators"]))
        print(f"   fold {fold+1}  best_iter={best_iters[-1]:5d}  RAE={fr:.4f}  "
              f"elapsed={time.time()-t0:6.1f}s", flush=True)

    oof_rae = float(rae(y_tr, oof))
    mean_best = int(np.mean(best_iters))
    print(f"[cv] OOF RAE = {oof_rae:.4f}  mean_best_iter = {mean_best}")

    # ---- final fit on all train ----
    print(f"\n[final] full-train fit, n_est={mean_best}...")
    final_params = dict(PARAMS, n_estimators=mean_best)
    m_final = lgb.train(
        final_params,
        lgb.Dataset(X_tr, label=y_tr),
        callbacks=[lgb.log_evaluation(-1)],
    )
    te_preds = np.clip(
        m_final.predict(X_te), y_tr.min() - 0.5, y_tr.max() + 0.5
    ).astype(np.float32)

    in_r = float(rae(y_unb, te_preds[unb_idx].astype(np.float64)))
    ratio = float(te_preds.std() / (oof.std() + 1e-12))
    print(f"[deploy] te mean/std = {te_preds.mean():.3f}/{te_preds.std():.3f}  "
          f"ratio(te/oof) = {ratio:.2f}")
    print(f"[deploy] in_RAE(253) = {in_r:.4f}")

    # ---- Pearson vs anchors ----
    te_nb972 = np.load(DATA_PROCESSED / "te_nb972_long_train.npy").astype(np.float64)
    pearson_972 = float(np.corrcoef(te_preds.astype(np.float64), te_nb972)[0, 1])
    print(f"[corr] Pearson(nb1093, nb972)         = {pearson_972:.4f}")
    pearson_cp: float | None = None
    try:
        te_chemprop = np.load(
            DATA_PROCESSED / "te_chemprop_aux.npy"
        ).astype(np.float64)
        pearson_cp = float(np.corrcoef(te_preds.astype(np.float64), te_chemprop)[0, 1])
        print(f"[corr] Pearson(nb1093, chemprop_aux)  = {pearson_cp:.4f}")
    except FileNotFoundError:
        pass

    # ---- promotion verdict ----
    promote = (pearson_972 < 0.95) or (in_r < 0.65)
    verdict = "PROMOTE_TO_ANCHOR" if promote else "NOT_PROMOTED"
    beats_nb1070_proxy = in_r < NB1070_HONEST  # weak proxy (in-sample vs cross-fit)

    # ---- persist ----
    np.save(DATA_PROCESSED / f"oof_{TAG}.npy", oof)
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", te_preds)
    sub_path = SUBMISSIONS / f"{TAG}_mordred_feature_long_train.csv"
    pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te["name"].values,
        "pEC50": te_preds,
    }).to_csv(sub_path, index=False)
    print(f"[save] te_{TAG}.npy, oof_{TAG}.npy, {sub_path}")

    summary = {
        "tag": TAG,
        "params": {k: v for k, v in PARAMS.items() if k != "verbose"},
        "early_stopping_rounds": EARLY_STOP,
        "n_features_total": int(n_feat),
        "n_features_combined": int(X_tr_c.shape[1]),
        "n_features_mordred": int(X_tr_m.shape[1]),
        "fold_best_iters": best_iters,
        "fold_raes": fold_raes,
        "mean_best_iter": mean_best,
        "oof_rae": oof_rae,
        "in_rae_253": in_r,
        "test_mean": float(te_preds.mean()),
        "test_std": float(te_preds.std()),
        "te_oof_std_ratio": ratio,
        "pearson_nb972": pearson_972,
        "pearson_chemprop_aux": pearson_cp,
        "promotion_threshold_pearson": 0.95,
        "promotion_threshold_in_rae": 0.65,
        "promote_as_anchor": bool(promote),
        "verdict": verdict,
        "beats_nb1070_proxy": bool(beats_nb1070_proxy),
        "nb972_in_rae_ref": NB972_IN_RAE,
        "nb1070_honest_ref": NB1070_HONEST,
        "delta_in_rae_vs_nb972": in_r - NB972_IN_RAE,
        "submission": str(sub_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    with open(DATA_PROCESSED / f"{TAG}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {DATA_PROCESSED / f'{TAG}_summary.json'}")
    print(f"\n=== {TAG} done in {time.time()-t0:.1f}s ===")
    print(f"VERDICT: {verdict}  "
          f"(pearson_nb972={pearson_972:.4f}, in_rae={in_r:.4f})")
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("n_features_total", "oof_rae", "in_rae_253",
              "pearson_nb972", "pearson_chemprop_aux",
              "promote_as_anchor", "verdict", "submission"):
        print(f"  {k}: {res.get(k)}")
