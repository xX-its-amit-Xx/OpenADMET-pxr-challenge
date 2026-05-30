"""nb314 -- TDC ADMET multi-task fingerprint feature.

Therapeutic Data Commons (TDC) has dozens of ADMET prediction tasks. We
fetch up to 10 of them (CYP3A4 inhibition + substrate, other CYPs,
Lipophilicity, Solubility, HIA, Pgp, BBB, microsomal clearance), train a
fast LGBM per task, and use the per-model PREDICTION on PXR train+test
compounds as a 10-d "ADMET fingerprint".

We then train a final PXR LGBM with [combined features | ADMET-fp] and
scaffold 5-fold CV. Outputs:
    oof_nb314_tdc.npy
    te_nb314_tdc.npy
We end with a 5-way SLSQP blend using nb239 as base + other top stacks.

If `pytdc` is missing we attempt `pip install pytdc --quiet`.  If install
or any TDC fetch fails, we silently fall back to no-ADMET-fp (just the
combined RDKit+Morgan baseline LGBM).
"""
from __future__ import annotations

import os, sys, warnings, subprocess
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko, standardize
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
N_FOLDS = 5

TDC_TASKS = [
    "CYP3A4_Veith",
    "CYP3A4_Substrate_CarbonMangels",
    "CYP2D6_Veith",
    "CYP1A2_Veith",
    "Lipophilicity_AstraZeneca",
    "Solubility_AqSolDB",
    "HIA_Hou",
    "Pgp_Broccatelli",
    "BBB_Martins",
    "Clearance_Hepatocyte_AZ",          # Human/AZ liver microsomal clearance proxy
]

FAST_LGBM = dict(n_estimators=300, num_leaves=31, learning_rate=0.05,
                 subsample=0.85, colsample_bytree=0.85, min_child_samples=10,
                 reg_alpha=0.1, reg_lambda=0.1, n_jobs=4, random_state=SEED,
                 verbose=-1)

FINAL_LGBM = dict(n_estimators=1500, num_leaves=64, learning_rate=0.03,
                  subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
                  reg_alpha=0.1, reg_lambda=0.1, objective='mae',
                  n_jobs=4, random_state=SEED, verbose=-1)


def ensure_tdc():
    try:
        import tdc  # noqa
        return True
    except ImportError:
        print("pytdc not installed, attempting pip install...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "pytdc", "--quiet"],
            check=True, timeout=180,
        )
        import importlib
        importlib.invalidate_caches()
        import tdc  # noqa
        return True
    except Exception as e:
        print(f"pytdc install failed: {e}")
        return False


def safe_standardize_list(smiles_list):
    out = []
    for s in smiles_list:
        try:
            out.append(standardize(s) or s)
        except Exception:
            out.append(s)
    return out


def fetch_and_train_task(task_name, pxr_train_smiles, pxr_test_smiles):
    """Return (oof_on_pxr_train, pred_on_pxr_test) for one TDC ADMET task.

    Strategy: train fast LGBM on full TDC training set, predict on PXR
    train + PXR test compounds.  We do NOT cross-fit against PXR splits
    because the TDC labels are independent of PXR (no leakage).
    """
    from tdc.single_pred import ADME, Tox
    print(f"  [{task_name}] fetching...")
    try:
        data = ADME(name=task_name)
    except Exception:
        data = Tox(name=task_name)

    split = data.get_split()
    df = pd.concat([split["train"], split["valid"]], ignore_index=True)
    smi = df["Drug"].astype(str).tolist()
    y   = df["Y"].astype(float).values
    print(f"    rows={len(smi):5d}  y_mean={y.mean():.3f}  y_std={y.std():.3f}")

    # Standardise & featurise
    smi_std = safe_standardize_list(smi)
    X = impute(combined(smi_std)).astype(np.float32)

    model = lgb.LGBMRegressor(**FAST_LGBM)
    model.fit(X, y)

    Xp_tr = impute(combined(pxr_train_smiles)).astype(np.float32)
    Xp_te = impute(combined(pxr_test_smiles)).astype(np.float32)
    pred_tr = model.predict(Xp_tr).astype(np.float64)
    pred_te = model.predict(Xp_te).astype(np.float64)
    return pred_tr, pred_te


def build_admet_fp(pxr_train_smiles, pxr_test_smiles):
    """Try to fetch all TDC tasks; return (N_tr, K), (N_te, K) plus task names."""
    if not ensure_tdc():
        print("No TDC -> skipping ADMET fingerprint.")
        return None, None, []

    cols_tr, cols_te, names = [], [], []
    for task in TDC_TASKS:
        try:
            ptr, pte = fetch_and_train_task(task, pxr_train_smiles, pxr_test_smiles)
            cols_tr.append(ptr); cols_te.append(pte); names.append(task)
        except Exception as e:
            print(f"  [{task}] FAILED: {e}")
            continue
    if not cols_tr:
        return None, None, []
    return np.column_stack(cols_tr), np.column_stack(cols_te), names


def slsqp_blend(oof_mat, y, n_seeds=80):
    k = oof_mat.shape[1]
    def loss(w): return rae(y, oof_mat @ w)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0.0, 1.0)] * k
    best = None
    for s in range(n_seeds):
        rng = np.random.default_rng(s)
        w0 = rng.dirichlet(np.ones(k))
        try:
            res = minimize(loss, w0, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"ftol": 1e-9, "maxiter": 600})
            if best is None or res.fun < best.fun: best = res
        except Exception:
            continue
    return best


def main():
    print("=== nb314: TDC ADMET multi-task fingerprint ===\n")

    tr = load_train(); te = load_test()
    y = tr["pec50"].values.astype(np.float64)
    n_tr = len(y); n_te = len(te)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    smi_tr = tr["smiles"].tolist()
    smi_te = te["smiles"].tolist()

    # 1) ADMET fingerprint
    print("--- Building ADMET fingerprint from TDC ---")
    admet_tr, admet_te, names = build_admet_fp(smi_tr, smi_te)
    if admet_tr is not None:
        print(f"\nADMET fingerprint: shape={admet_tr.shape}  tasks={names}")
    else:
        print("\nADMET fingerprint UNAVAILABLE -> falling back to combined-only.")

    # 2) Base PXR features
    print("\n--- Featurising PXR base (combined) ---")
    X_tr_base = impute(combined(smi_tr)).astype(np.float32)
    X_te_base = impute(combined(smi_te)).astype(np.float32)

    if admet_tr is not None:
        X_tr_full = np.hstack([X_tr_base, admet_tr.astype(np.float32)])
        X_te_full = np.hstack([X_te_base, admet_te.astype(np.float32)])
    else:
        X_tr_full = X_tr_base; X_te_full = X_te_base

    print(f"final feature dims: train {X_tr_full.shape}  test {X_te_full.shape}")

    # 3) Scaffold 5-fold CV LGBM
    print("\n--- Training final PXR LGBM with ADMET-fp + scaffold 5-fold CV ---")
    oof = np.full(n_tr, np.nan)
    te_preds = []
    for fold, (tri, vai) in enumerate(splits):
        m = lgb.LGBMRegressor(**FINAL_LGBM)
        m.fit(X_tr_full[tri], y[tri],
              eval_set=[(X_tr_full[vai], y[vai])],
              callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        oof[vai] = m.predict(X_tr_full[vai])
        te_preds.append(m.predict(X_te_full))
        print(f"  fold {fold+1} RAE={rae(y[vai], oof[vai]):.4f}")
    te_final = np.mean(te_preds, axis=0)

    r = rae(y, oof); sp, _ = stats.spearmanr(y, oof); pr, _ = stats.pearsonr(y, oof)
    ratio = te_final.std() / oof.std() if oof.std() > 0 else 0.0
    print(f"\nnb314 OOF: RAE={r:.4f}  Spearman={sp:.4f}  Pearson={pr:.4f}  "
          f"te_std/oof_std={ratio:.2f}")

    te_final = np.clip(te_final, y.min() - 0.5, y.max() + 0.5)
    np.save(DATA_PROCESSED / "oof_nb314_tdc.npy", oof)
    np.save(DATA_PROCESSED / "te_nb314_tdc.npy",  te_final)

    sub = pd.DataFrame({"Molecule Name": te["name"].values,
                        "SMILES":         te["smiles"].values,
                        "pEC50":          te_final})
    sub.to_csv(SUBMISSIONS / "314_tdc_multitask.csv", index=False)
    print(f"saved submissions/314_tdc_multitask.csv")

    # 4) 5-way SLSQP blend with nb239 base
    print("\n--- 5-way SLSQP blend (nb239 base + 4 others + nb314) ---")
    candidates = [
        ("nb239_full_slsqp",   "oof_nb239_full_slsqp.npy",   "te_nb239_full_slsqp.npy"),
        ("nb107_assay_decomp", "oof_nb107_assay_decomp.npy", "te_nb107_assay_decomp.npy"),
        ("nb145_xgb_stack",    "oof_nb145_xgb_stack.npy",    "te_nb145_xgb_stack.npy"),
        ("nb117_knn_residual", "oof_nb117_knn_residual.npy", "te_nb117_knn_residual.npy"),
        ("nb314_tdc",          None, None),  # handled inline
    ]
    oof_cols, te_cols, stems = [], [], []
    for name, op, tp in candidates:
        if name == "nb314_tdc":
            oof_cols.append(oof); te_cols.append(te_final); stems.append(name); continue
        op_p = DATA_PROCESSED / op; tp_p = DATA_PROCESSED / tp
        if not op_p.exists() or not tp_p.exists():
            print(f"  SKIP {name} (missing)"); continue
        o = np.load(op_p).astype(np.float64); t = np.load(tp_p).astype(np.float64)
        if o.shape != (n_tr,) or t.shape != (n_te,):
            print(f"  SKIP {name} (wrong size)"); continue
        o = np.where(np.isfinite(o), o, np.nanmean(o))
        t = np.where(np.isfinite(t), t, np.nanmean(t))
        oof_cols.append(o); te_cols.append(t); stems.append(name)

    M_oof = np.column_stack(oof_cols)
    M_te  = np.column_stack(te_cols)
    print(f"  pool: {len(stems)} stems -> {stems}")
    best = slsqp_blend(M_oof, y, n_seeds=120)
    if best is not None:
        w = best.x
        print(f"\n5-way SLSQP OOF RAE: {best.fun:.4f}")
        for stem, wi in zip(stems, w):
            print(f"    {stem:25s}  w={wi:.4f}")
        blend_oof = M_oof @ w
        blend_te  = np.clip(M_te @ w, y.min() - 0.5, y.max() + 0.5)
        np.save(DATA_PROCESSED / "oof_nb314_slsqp5.npy", blend_oof)
        np.save(DATA_PROCESSED / "te_nb314_slsqp5.npy",  blend_te)
        sub2 = pd.DataFrame({"Molecule Name": te["name"].values,
                             "SMILES":         te["smiles"].values,
                             "pEC50":          blend_te})
        sub2.to_csv(SUBMISSIONS / "314_tdc_slsqp5.csv", index=False)
        print(f"saved submissions/314_tdc_slsqp5.csv")


if __name__ == "__main__":
    main()
