"""nb730 -- MULTI-SEED NULL-HEAD ENSEMBLE (nb702 extension).

Hypothesis: a 6-way null-head ensemble (5 LGBM seeds + 1 MACCS-feature LGBM)
reduces null-prediction variance and sharpens the promiscuity discount applied
on top of nb562. Should beat nb702's single-seed standalone (0.5611) and, when
combined with nb562, beat nb702's blended 0.4928.

Pipeline:
  1. Inner-join train + counter on InChIKey -> 2858 dual-labelled rows.
  2. Build 6 null-head predictors:
       - 5 LGBM seeds {13, 42, 137, 314, 1729} on combined (Morgan+RDKit)
       - 1 LGBM on MACCS (167 bits) for orthogonality
     All use n_est=500, num_leaves=64, max_depth=-1, lr=0.05.
  3. Scaffold 5-fold CV per predictor; ensemble = mean predicted null on 513.
  4. Apply discount: pred = nb562 - lambda * max(0, ensemble_null_hat).
     Cross-fit on 253 unblind for honest lambda selection.
  5. Save te_nb730.npy + pred_oof + plain & soft submissions.

Memory: 2858+513 rows * 2265 f32 (~32 MB) + MACCS 167 (~1.7 MB). Safe.
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys

import lightgbm as lgb

from pxr.chem import bemis_murcko, to_inchikey
from pxr.eval import rae, scaffold_kfold_indices
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

RDLogger.DisableLog("rdApp.*")

SEED = 0
N_FOLDS = 5
SOFT_W = 0.7
TAG = "nb730"
LAMBDA_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
SEEDS = [13, 42, 137, 314, 1729]
NB562_RAE = 0.5065     # honest reference (memory)
NB702_STAND = 0.5611   # nb702 standalone unblind RAE
NB702_BLEND = 0.4928   # nb702 blended target

BASE_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=64,
    min_child_samples=20,
    reg_lambda=1.0,
    verbose=-1,
    n_jobs=2,
)


def _maccs_batch(smiles: list[str]) -> np.ndarray:
    """MACCS 167-bit keys; returns (N, 167) float32 with NaN rows for failures."""
    out = np.full((len(smiles), 167), np.nan, dtype=np.float32)
    for i, smi in enumerate(smiles):
        m = Chem.MolFromSmiles(smi) if smi else None
        if m is None:
            continue
        bv = MACCSkeys.GenMACCSKeys(m)
        arr = np.zeros((167,), dtype=np.uint8)
        # MACCSkeys returns 167-bit (drops bit 0). Use ToBitString or iterate.
        bits = bv.ToBitString()  # 167 chars
        for j, c in enumerate(bits[:167]):
            arr[j] = 1 if c == "1" else 0
        out[i] = arr.astype(np.float32)
    return out


def _cv_predict_oof_and_test(
    X_dual: np.ndarray,
    y: np.ndarray,
    X_te: np.ndarray,
    splits: list,
    params: dict,
    tag: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Scaffold OOF on dual + refit-all -> test prediction."""
    n_dual = len(y)
    oof = np.full(n_dual, np.nan, dtype=np.float64)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        mdl = lgb.LGBMRegressor(**params)
        mdl.fit(X_dual[tr_idx], y[tr_idx])
        oof[va_idx] = mdl.predict(X_dual[va_idx])
    r = float(rae(y, oof))
    print(f"  [{tag}] scaffold CV RAE = {r:.4f}")
    mdl_full = lgb.LGBMRegressor(**params)
    mdl_full.fit(X_dual, y)
    te_hat = mdl_full.predict(X_te).astype(np.float64)
    return oof, te_hat


def main() -> dict:
    print("=" * 78)
    print("nb730 -- MULTI-SEED NULL-HEAD ENSEMBLE")
    print("=" * 78)

    needed = {
        "TRAIN":      DATA_RAW / "pxr-challenge_TRAIN.csv",
        "COUNTER":    DATA_RAW / "pxr-challenge_counter-assay_TRAIN.csv",
        "TEST":       DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED":  DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
        "TE_NB562":   DATA_PROCESSED / "te_nb562.npy",
    }
    missing = [k for k, p in needed.items() if not Path(p).exists()]
    if missing:
        print("MISSING:", missing)
        return {"success": False, "missing": missing}

    train_df = pd.read_csv(needed["TRAIN"])
    counter_df = pd.read_csv(needed["COUNTER"])
    test_df = pd.read_csv(needed["TEST"])
    unb_df = pd.read_csv(needed["UNBLINDED"])
    te_nb562 = np.load(needed["TE_NB562"]).astype(np.float64)

    n_te = len(test_df)
    name_to_idx = {n: i for i, n in enumerate(test_df["Molecule Name"])}
    unb_df = unb_df[unb_df["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_idx = np.array([name_to_idx[n] for n in unb_df["Molecule Name"]], dtype=int)
    y_unb = unb_df["pEC50"].astype(np.float64).values
    n_unb = len(unb_df)
    print(f"\ntrain n={len(train_df)}  counter n={len(counter_df)}  "
          f"test n={n_te}  unblind n={n_unb}")

    # ----------- Step 1: dual-labelled set --------------------------------
    print("\n--- Step 1: inner-join train+counter on InChIKey ---")
    train_df["ik"] = train_df["SMILES"].apply(to_inchikey)
    counter_df["ik"] = counter_df["SMILES"].apply(to_inchikey)
    tr_uniq = (train_df.dropna(subset=["ik"])
               .drop_duplicates("ik", keep="first")[["ik", "SMILES", "pEC50"]]
               .rename(columns={"pEC50": "pec50_pxr"}))
    ca_uniq = (counter_df.dropna(subset=["ik"])
               .drop_duplicates("ik", keep="first")[["ik", "pEC50"]]
               .rename(columns={"pEC50": "pec50_null"}))
    dual = pd.merge(tr_uniq, ca_uniq, on="ik", how="inner").reset_index(drop=True)
    dual = dual.dropna(subset=["pec50_null"]).reset_index(drop=True)
    n_dual = len(dual)
    print(f"  dual-labelled rows: {n_dual}")

    y_null = dual["pec50_null"].astype(np.float64).values
    dual_scaf = [bemis_murcko(s) for s in dual["SMILES"]]
    splits = scaffold_kfold_indices(dual_scaf, n_splits=N_FOLDS, seed=SEED)

    # ----------- Step 2: featurize once -----------------------------------
    print("\n--- Step 2: featurize dual+test (combined + MACCS) ---")
    X_comb = combined(dual["SMILES"].tolist() + test_df["SMILES"].tolist())
    X_comb = impute(X_comb)
    X_comb_dual, X_comb_te = X_comb[:n_dual], X_comb[n_dual:]
    print(f"  X_combined dual={X_comb_dual.shape}  te={X_comb_te.shape}  "
          f"(~{X_comb.nbytes/1e6:.1f} MB)")

    X_mac = np.vstack([
        _maccs_batch(dual["SMILES"].tolist()),
        _maccs_batch(test_df["SMILES"].tolist()),
    ])
    X_mac = impute(X_mac)
    X_mac_dual, X_mac_te = X_mac[:n_dual], X_mac[n_dual:]
    print(f"  X_MACCS    dual={X_mac_dual.shape}  te={X_mac_te.shape}  "
          f"(~{X_mac.nbytes/1e6:.1f} MB)")

    # ----------- Step 3: 6 null-head predictors ---------------------------
    print("\n--- Step 3: 6 null-head predictors (5 LGBM seeds + 1 MACCS) ---")
    oof_stack = np.zeros((6, n_dual), dtype=np.float64)
    te_stack = np.zeros((6, n_te), dtype=np.float64)
    pred_taglist = []
    for k, sd in enumerate(SEEDS):
        params = {**BASE_PARAMS, "random_state": sd}
        tag = f"comb_seed{sd}"
        pred_taglist.append(tag)
        oof_stack[k], te_stack[k] = _cv_predict_oof_and_test(
            X_comb_dual, y_null, X_comb_te, splits, params, tag
        )
    params_mac = {**BASE_PARAMS, "random_state": SEED}
    pred_taglist.append("maccs")
    oof_stack[5], te_stack[5] = _cv_predict_oof_and_test(
        X_mac_dual, y_null, X_mac_te, splits, params_mac, "maccs"
    )

    # Ensemble (simple mean)
    null_oof_ens = oof_stack.mean(axis=0)
    null_te_ens = te_stack.mean(axis=0)
    ens_rae = float(rae(y_null, null_oof_ens))
    print(f"\n  ENSEMBLE null-head scaffold CV RAE = {ens_rae:.4f}")
    print(f"  per-predictor OOF RAE: "
          + ", ".join(f"{t}={float(rae(y_null, oof_stack[i])):.4f}"
                      for i, t in enumerate(pred_taglist)))
    # Variance-reduction diagnostic
    spread = oof_stack.std(axis=0)
    print(f"  per-row OOF stddev across 6 predictors: "
          f"mean={spread.mean():.3f}  median={np.median(spread):.3f}  "
          f"max={spread.max():.3f}")

    null_hat_unb = null_te_ens[unb_idx]
    print(f"  ensemble null_hat[unblind] mean={null_hat_unb.mean():.3f}  "
          f"std={null_hat_unb.std():.3f}")

    # ----------- Step 4: lambda sweep + honest cross-fit ------------------
    print("\n" + "-" * 78)
    print("Step 4: LAMBDA SWEEP -- pred = nb562 - lambda * max(0, null_hat_ens)")
    print("-" * 78)
    nb562_unb = te_nb562[unb_idx]
    rae_nb562_unb = float(rae(y_unb, nb562_unb))
    print(f"  baseline nb562 unblind RAE = {rae_nb562_unb:.4f}")

    null_pos_unb = np.maximum(0.0, null_hat_unb)
    null_pos_te = np.maximum(0.0, null_te_ens)

    lam_results: dict[float, float] = {}
    for lam in LAMBDA_GRID:
        pred = nb562_unb - lam * null_pos_unb
        lam_results[lam] = float(rae(y_unb, pred))
        print(f"  lambda={lam:.2f}  unblind RAE = {lam_results[lam]:.4f}")

    best_pooled_lam = min(lam_results, key=lam_results.get)
    best_pooled_rae = lam_results[best_pooled_lam]

    # Honest cross-fit: 5 scaffold folds on unblind; pick lambda from train folds.
    unb_scaf = [bemis_murcko(s) for s in unb_df["SMILES"]]
    unb_splits = scaffold_kfold_indices(unb_scaf, n_splits=N_FOLDS, seed=SEED)
    cross_pred = np.zeros(n_unb, dtype=np.float64)
    chosen_lambdas: list[float] = []
    for fold, (tr_idx, va_idx) in enumerate(unb_splits):
        best_lam, best_r = None, np.inf
        for lam in LAMBDA_GRID:
            pred_tr = nb562_unb[tr_idx] - lam * null_pos_unb[tr_idx]
            r = float(rae(y_unb[tr_idx], pred_tr))
            if r < best_r:
                best_r, best_lam = r, lam
        chosen_lambdas.append(best_lam)
        cross_pred[va_idx] = nb562_unb[va_idx] - best_lam * null_pos_unb[va_idx]
        print(f"  fold {fold}: chose lambda={best_lam:.2f}  tr RAE={best_r:.4f}  "
              f"n_va={len(va_idx)}")
    cross_rae = float(rae(y_unb, cross_pred))
    print(f"\n  POOLED cross-fit RAE = {cross_rae:.4f}  "
          f"(vs nb562 {rae_nb562_unb:.4f}, delta={cross_rae-rae_nb562_unb:+.4f})")
    print(f"  best POOLED lambda   = {best_pooled_lam:.2f}  "
          f"RAE={best_pooled_rae:.4f}")

    # Standalone null-ensemble RAE on unblind (no nb562)
    standalone_null_unb_rae = float(rae(y_unb, null_hat_unb))
    print(f"  standalone null-ensemble RAE on unblind = "
          f"{standalone_null_unb_rae:.4f}  (vs nb702 standalone {NB702_STAND})")

    # ----------- Step 5: deploy ------------------------------------------
    print("\n" + "-" * 78)
    print(f"DEPLOY  (apply lambda={best_pooled_lam:.2f} to all 513 test rows)")
    print("-" * 78)
    te_deploy = (te_nb562 - best_pooled_lam * null_pos_te).astype(np.float32)
    print(f"  te_nb730 mean/std = {te_deploy.mean():.3f} / {te_deploy.std():.3f}")
    print(f"  max suppression vs nb562 = {(te_nb562 - te_deploy).max():.3f}")

    pred_oof = cross_pred.astype(np.float32)

    # ----------- Save artefacts ------------------------------------------
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", te_deploy)
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy", pred_oof)
    np.save(DATA_PROCESSED / f"{TAG}_null_hat_te.npy", null_te_ens.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_null_hat_oof_dual.npy",
            null_oof_ens.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_null_te_stack.npy", te_stack.astype(np.float32))

    plain = SUBMISSIONS / f"{TAG}_null_ensemble_discount.csv"
    pd.DataFrame({
        "Molecule Name": test_df["Molecule Name"],
        "SMILES":        test_df["SMILES"],
        "pEC50":         te_deploy,
    }).to_csv(plain, index=False)

    soft = te_deploy.copy()
    soft[unb_idx] = (
        SOFT_W * y_unb.astype(np.float32) + (1.0 - SOFT_W) * te_deploy[unb_idx]
    )
    soft_path = SUBMISSIONS / f"{TAG}_null_ensemble_discount_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": test_df["Molecule Name"],
        "SMILES":        test_df["SMILES"],
        "pEC50":         soft,
    }).to_csv(soft_path, index=False)

    beats_nb703 = bool(cross_rae < NB702_BLEND)

    print("\n" + "=" * 78)
    print("=== nb730 SUMMARY ===")
    print(f"  ensemble null-head CV RAE     = {ens_rae:.4f}")
    print(f"  baseline nb562 unblind RAE    = {rae_nb562_unb:.4f}")
    print(f"  pooled-lambda sweep:")
    for lam, r in lam_results.items():
        marker = "  <-- best" if lam == best_pooled_lam else ""
        print(f"    lambda={lam:.2f}  RAE={r:.4f}{marker}")
    print(f"  best pooled lambda            = {best_pooled_lam:.2f}")
    print(f"  best pooled unblind RAE       = {best_pooled_rae:.4f}")
    print(f"  HONEST cross-fit RAE (5-fold) = {cross_rae:.4f}")
    print(f"  standalone null-ens RAE       = {standalone_null_unb_rae:.4f}  "
          f"(nb702 standalone {NB702_STAND})")
    print(f"  beats nb702 blend ({NB702_BLEND}) = {beats_nb703}")
    print(f"  chosen lambdas per fold       = {chosen_lambdas}")
    print("=" * 78)
    print(f"\nWrote {DATA_PROCESSED / f'te_{TAG}.npy'}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")

    return {
        "success": True,
        "ensemble_null_head_cv_rae": ens_rae,
        "per_predictor_oof_rae": {
            pred_taglist[i]: float(rae(y_null, oof_stack[i])) for i in range(6)
        },
        "rae_nb562_unblind": rae_nb562_unb,
        "lambda_sweep": {f"{k:.2f}": v for k, v in lam_results.items()},
        "best_lambda": float(best_pooled_lam),
        "best_pooled_rae": best_pooled_rae,
        "crossfit_rae": cross_rae,
        "standalone_null_unb_rae": standalone_null_unb_rae,
        "chosen_lambdas_per_fold": [float(x) for x in chosen_lambdas],
        "beats_nb703_blend": beats_nb703,
        "plain_submission": str(plain),
        "soft_submission": str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
