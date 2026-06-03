"""nb922 - Triple-head multitask LightGBM (PXR / null / single-conc).

Single LGBM-Huber model trained on a row-stacked corpus of three label sources:
  * 4139 CRC pEC50 (PXR head)
  * 2858 counter-assay pEC50 (null head)
  * ~21,003 single-concentration log2FC (SP head)

Task identity is encoded as a 3-d one-hot appended to (Morgan + RDKit) = 2268
features. Per-row sample weights upweight the PXR signal (PXR=4.0, null=1.5,
SP=0.5). Scaffold 5-fold CV is done on the PXR rows; non-PXR rows always go in
the training side of each fold. Test prediction uses PXR one-hot=1.

Saves OOF + test arrays + submission + in_RAE(253). Memory ~250 MB,
wall-time < 10 min on CPU.
"""
import os, sys, warnings, json, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

from pxr.data import load_train, load_test, load_counter, load_single_conc
from pxr.featurize import combined, impute
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
N_FOLDS = 5

W_PXR, W_NULL, W_SP = 4.0, 1.5, 0.5

LGBM_PARAMS = dict(
    objective="huber", alpha=2.0,
    n_estimators=2500, num_leaves=128, learning_rate=0.025,
    min_child_samples=10, subsample=0.85, colsample_bytree=0.85,
    reg_alpha=0.05, reg_lambda=0.1,
    random_state=SEED, verbose=-1, n_jobs=4,
)

ART = Path("C:/pxr_artifacts/nb922")
ART.mkdir(parents=True, exist_ok=True)


def in_rae(y_true, y_pred):
    yt = np.asarray(y_true, float); yp = np.asarray(y_pred, float)
    return float(np.mean(np.abs(yt - yp)) / np.mean(np.abs(yt - yt.mean())))


def main():
    t0 = time.time()
    print("=== nb922: triple-head multitask LGBM-Huber ===\n")

    # ---- Truth + unblind index ----
    unblind_idx = np.load(DATA_PROCESSED / "nb472_unblind_idx.npy")
    y_unblind   = np.load(DATA_PROCESSED / "_audit_unblind_y.npy")
    assert len(unblind_idx) == len(y_unblind) == 253

    # ---- Load three sources ----
    tr      = load_train()
    te      = load_test()
    counter = load_counter()
    sp      = load_single_conc()

    y_pxr  = tr["pec50"].values.astype(np.float64)
    n_pxr  = len(y_pxr)

    # null head: counter-assay pEC50
    counter_valid = counter.dropna(subset=["pec50", "smiles"]).reset_index(drop=True)
    y_null   = counter_valid["pec50"].values.astype(np.float64)
    sm_null  = counter_valid["smiles"].tolist()
    n_null   = len(y_null)

    # SP head: aggregate to one row per compound (mean log2_fc across plates/conc)
    sp_valid = sp.dropna(subset=["smiles", "log2_fc_estimate"])
    sp_agg = (sp_valid.groupby("smiles", as_index=False)
                      .agg(log2fc=("log2_fc_estimate", "mean")))
    y_sp   = sp_agg["log2fc"].values.astype(np.float64)
    sm_sp  = sp_agg["smiles"].tolist()
    n_sp   = len(y_sp)

    print(f"Rows: PXR={n_pxr}  null={n_null}  SP={n_sp}  total={n_pxr+n_null+n_sp}")

    # ---- Features ----
    print("Featurizing (Morgan + RDKit) for all sources...")
    X_pxr  = impute(combined(tr["smiles"].tolist())).astype(np.float32)
    X_null = impute(combined(sm_null)).astype(np.float32)
    X_sp   = impute(combined(sm_sp)).astype(np.float32)
    X_te   = impute(combined(te["smiles"].tolist())).astype(np.float32)
    d = X_pxr.shape[1]
    assert d == 2265, f"expected 2265 feats, got {d}"

    # ---- Task one-hot (3-d) ----
    def task_onehot(n, idx):
        z = np.zeros((n, 3), dtype=np.float32); z[:, idx] = 1.0; return z

    X_pxr_full  = np.hstack([X_pxr,  task_onehot(n_pxr,  0)])
    X_null_full = np.hstack([X_null, task_onehot(n_null, 1)])
    X_sp_full   = np.hstack([X_sp,   task_onehot(n_sp,   2)])
    X_te_full   = np.hstack([X_te,   task_onehot(len(X_te), 0)])   # PXR head
    print(f"Feature shape: {X_pxr_full.shape[1]} (=2265 + 3)")

    # ---- Sample weights ----
    w_pxr  = np.full(n_pxr,  W_PXR,  dtype=np.float32)
    w_null = np.full(n_null, W_NULL, dtype=np.float32)
    w_sp   = np.full(n_sp,   W_SP,   dtype=np.float32)

    # ---- Scaffold 5-fold over PXR rows only ----
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    oof = np.full(n_pxr, np.nan, dtype=np.float64)

    for fold, (tr_idx, va_idx) in enumerate(splits):
        # PXR train rows for this fold + ALL null + ALL SP rows
        X_fold = np.vstack([X_pxr_full[tr_idx], X_null_full, X_sp_full])
        y_fold = np.concatenate([y_pxr[tr_idx], y_null, y_sp])
        w_fold = np.concatenate([w_pxr[tr_idx], w_null, w_sp])

        m = lgb.train(
            LGBM_PARAMS,
            lgb.Dataset(X_fold, label=y_fold, weight=w_fold),
            callbacks=[lgb.log_evaluation(-1)],
        )
        oof[va_idx] = m.predict(X_pxr_full[va_idx])
        print(f"  fold {fold+1}/{N_FOLDS}  va_RAE={rae(y_pxr[va_idx], oof[va_idx]):.4f}"
              f"  elapsed={time.time()-t0:.1f}s", flush=True)
        del m, X_fold, y_fold, w_fold

    oof_rae = rae(y_pxr, oof)
    print(f"\nPXR scaffold OOF RAE = {oof_rae:.4f}")

    # ---- Full refit -> test predict (PXR head one-hot) ----
    print("Refitting on full corpus (PXR + null + SP)...")
    X_all = np.vstack([X_pxr_full, X_null_full, X_sp_full])
    y_all = np.concatenate([y_pxr, y_null, y_sp])
    w_all = np.concatenate([w_pxr, w_null, w_sp])

    m_final = lgb.train(
        dict(LGBM_PARAMS, n_estimators=1800),
        lgb.Dataset(X_all, label=y_all, weight=w_all),
        callbacks=[lgb.log_evaluation(-1)],
    )
    te_preds = np.clip(m_final.predict(X_te_full),
                       y_pxr.min() - 0.5, y_pxr.max() + 0.5)

    in_r = in_rae(y_unblind, te_preds[unblind_idx])
    ratio = te_preds.std() / oof.std() if oof.std() > 0 else 0.0
    print(f"TEST  med={np.median(te_preds):.2f}  std={te_preds.std():.3f}"
          f"  ratio={ratio:.2f}  in_RAE(253)={in_r:.4f}")

    # ---- Save artifacts ----
    np.save(ART / "oof_nb922_triplehead.npy", oof)
    np.save(ART / "te_nb922_triplehead.npy",  te_preds)
    np.save(DATA_PROCESSED / "oof_nb922_triplehead.npy", oof)
    np.save(DATA_PROCESSED / "te_nb922_triplehead.npy",  te_preds)

    sub = pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te["name"].values,
        "pEC50": te_preds,
    })
    sub_path = SUBMISSIONS / "nb922_triple_head_multitask.csv"
    sub.to_csv(sub_path, index=False)
    print(f"\nWrote submission: {sub_path}")

    summary = {
        "n_rows": {"pxr": int(n_pxr), "null": int(n_null), "sp": int(n_sp),
                   "total": int(n_pxr + n_null + n_sp)},
        "feat_dim": int(X_pxr_full.shape[1]),
        "sample_weights": {"pxr": W_PXR, "null": W_NULL, "sp": W_SP},
        "oof_rae": float(oof_rae),
        "in_rae_253": float(in_r),
        "test_std": float(te_preds.std()),
        "std_ratio": float(ratio),
        "wall_seconds": float(time.time() - t0),
    }
    with open(ART / "nb922_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(DATA_PROCESSED / "nb922_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
