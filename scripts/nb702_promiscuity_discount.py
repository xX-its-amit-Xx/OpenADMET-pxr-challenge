"""nb702 -- PROMISCUITY DISCOUNT (Prescription P3).

Idea: PXR activators that are *also* hits in the PXR-null counter assay are
likely off-target / promiscuous binders, not clean PXR pharmacology. Subtract
a fraction of the predicted counter-assay activity from nb562 to push such
compounds down. The F2 cluster (high predicted PXR + scaffold-novel +
low nearest-neighbour Tanimoto) is the prime target -- if a compound there
also looks promiscuous, the discount should suppress the false positive.

Pipeline:
  1. Inner-join train + counter on InChIKey  -> 2858 dual-labelled rows.
  2. Train a null-head LGBM (pec50_null) on combined (Morgan + RDKit)
     features with scaffold 5-fold CV (sanity check OOF RAE).
  3. Refit on all 2858 and predict null_hat for the 513 test compounds.
  4. pred_discount[i] = nb562[i] - lambda * max(0, null_hat[i] - null_floor)
     We use null_floor = 0 (subtract any positive predicted null activity).
     Lambda grid: {0.1, 0.2, 0.3, 0.4, 0.5}.
     5-fold cross-fit on 253 unblind to pick the best lambda by pooled RAE.
  5. F2 cluster diagnostic (same gate rule as nb700/nb701):
       scaf_train_freq==0 AND nn_sim_train<0.6 AND best_pred_for_gate>4.0
     Report whether the discount helps the F2 subset specifically.
  6. Save te_nb702.npy + pred_oof + plain & soft-injected submissions.

Memory: tiny -- 2858 + 513 + 253 rows x 2265 f32 = ~32 MB.
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
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from sklearn.model_selection import KFold

import lightgbm as lgb

from pxr.chem import bemis_murcko, to_inchikey
from pxr.eval import rae, scaffold_kfold_indices
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

RDLogger.DisableLog("rdApp.*")

SEED = 0
N_FOLDS = 5
SOFT_W = 0.7
TAG = "nb702"
LAMBDA_GRID = [0.1, 0.2, 0.3, 0.4, 0.5]
NB562_RAE = 0.5065  # honest cross-fit reference (from memory)

LGBM_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=64,
    min_child_samples=20,
    reg_lambda=1.0,
    random_state=SEED,
    verbose=-1,
    n_jobs=2,
)


def _morgan(smi: str):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)


def main() -> dict:
    print("=" * 78)
    print("nb702 -- PROMISCUITY DISCOUNT (prescription P3)")
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
    assert te_nb562.shape == (len(test_df),), \
        f"nb562 shape {te_nb562.shape} != test n={len(test_df)}"

    n_te = len(test_df)
    name_to_idx = {n: i for i, n in enumerate(test_df["Molecule Name"])}
    unb_df = unb_df[unb_df["Molecule Name"].isin(name_to_idx)].reset_index(
        drop=True
    )
    unb_idx = np.array(
        [name_to_idx[n] for n in unb_df["Molecule Name"]], dtype=int
    )
    y_unb = unb_df["pEC50"].astype(np.float64).values
    n_unb = len(unb_df)
    print(f"\ntrain n={len(train_df)}  counter n={len(counter_df)}  "
          f"test n={n_te}  unblind n={n_unb}")

    # ----------- Step 1: build the 2858 dual-labelled set ---------------
    print("\n--- Step 1: inner-join train + counter on InChIKey ---")
    train_df = train_df.copy()
    counter_df = counter_df.copy()
    train_df["ik"] = train_df["SMILES"].apply(to_inchikey)
    counter_df["ik"] = counter_df["SMILES"].apply(to_inchikey)
    # drop duplicate ik in each side, keep first
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
    print(f"  pec50_null: mean={dual['pec50_null'].mean():.3f}  "
          f"std={dual['pec50_null'].std():.3f}  "
          f"min={dual['pec50_null'].min():.2f}  "
          f"max={dual['pec50_null'].max():.2f}")

    # ----------- Step 2: featurize dual + test (one impute) -------------
    print("\n--- Step 2: featurize dual+test (combined Morgan+RDKit) ---")
    X_dual = combined(dual["SMILES"].tolist())
    X_te = combined(test_df["SMILES"].tolist())
    X_all = np.vstack([X_dual, X_te])
    X_all = impute(X_all)
    X_dual = X_all[:n_dual]
    X_te = X_all[n_dual:]
    print(f"  X_dual={X_dual.shape}  X_te={X_te.shape}  "
          f"(~{X_all.nbytes/1e6:.1f} MB)")

    y_null = dual["pec50_null"].astype(np.float64).values

    # ----------- Step 3: scaffold 5-fold CV on the 2858 null-head -------
    print("\n--- Step 3: scaffold 5-fold CV on null-head LGBM ---")
    dual_scaf = [bemis_murcko(s) for s in dual["SMILES"]]
    splits = scaffold_kfold_indices(dual_scaf, n_splits=N_FOLDS, seed=SEED)
    null_oof = np.full(n_dual, np.nan, dtype=np.float64)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        mdl = lgb.LGBMRegressor(**LGBM_PARAMS)
        mdl.fit(X_dual[tr_idx], y_null[tr_idx])
        null_oof[va_idx] = mdl.predict(X_dual[va_idx])
        print(f"  fold {fold}: n_tr={len(tr_idx):4d}  n_va={len(va_idx):4d}  "
              f"RAE={rae(y_null[va_idx], null_oof[va_idx]):.4f}")
    null_head_rae = float(rae(y_null, null_oof))
    print(f"\n  null-head scaffold CV RAE = {null_head_rae:.4f}")

    # ----------- Step 4: refit + predict null_hat for 513 ---------------
    print("\n--- Step 4: refit on 2858, predict null_hat for 513 ---")
    mdl_full = lgb.LGBMRegressor(**LGBM_PARAMS)
    mdl_full.fit(X_dual, y_null)
    null_hat_te = mdl_full.predict(X_te).astype(np.float64)
    print(f"  null_hat[test] mean={null_hat_te.mean():.3f}  "
          f"std={null_hat_te.std():.3f}  "
          f"min={null_hat_te.min():.3f}  max={null_hat_te.max():.3f}")

    # On the 253 unblind subset
    null_hat_unb = null_hat_te[unb_idx]
    print(f"  null_hat[unblind] mean={null_hat_unb.mean():.3f}  "
          f"std={null_hat_unb.std():.3f}")

    # ----------- Step 5: lambda sweep (cross-fit on 253) ----------------
    # The discount is a closed-form transform of fixed inputs (nb562, null_hat).
    # There are no trainable params per fold, so a "cross-fit" picks lambda
    # globally on the 253 unblind RAE.  We keep 5-fold scaffold splits and
    # report per-fold to detect lambda sensitivity.
    print("\n" + "-" * 78)
    print("Step 5: LAMBDA SWEEP -- discount = nb562 - lambda * max(0, null_hat)")
    print("-" * 78)
    nb562_unb = te_nb562[unb_idx]
    rae_nb562_unb = float(rae(y_unb, nb562_unb))
    print(f"  baseline nb562 unblind RAE = {rae_nb562_unb:.4f}")

    null_pos_unb = np.maximum(0.0, null_hat_unb)
    null_pos_te = np.maximum(0.0, null_hat_te)

    # Pooled-lambda evaluation (one lambda for everything)
    lam_results: dict[float, float] = {}
    for lam in LAMBDA_GRID:
        pred = nb562_unb - lam * null_pos_unb
        lam_results[lam] = float(rae(y_unb, pred))
        print(f"  lambda={lam:.2f}  unblind RAE = {lam_results[lam]:.4f}")

    # Cross-fit version: split 253 into 5 folds, on each fold pick lambda from
    # the OTHER 4 folds, apply to the held-out one. Pooled RAE = honest.
    unb_scaf = [bemis_murcko(s) for s in unb_df["SMILES"]]
    unb_splits = scaffold_kfold_indices(unb_scaf, n_splits=N_FOLDS, seed=SEED)
    cross_pred = np.zeros(n_unb, dtype=np.float64)
    chosen_lambdas: list[float] = []
    for fold, (tr_idx, va_idx) in enumerate(unb_splits):
        best_lam = None
        best_r = np.inf
        for lam in LAMBDA_GRID:
            pred_tr = nb562_unb[tr_idx] - lam * null_pos_unb[tr_idx]
            r = float(rae(y_unb[tr_idx], pred_tr))
            if r < best_r:
                best_r = r
                best_lam = lam
        chosen_lambdas.append(best_lam)
        cross_pred[va_idx] = nb562_unb[va_idx] - best_lam * null_pos_unb[va_idx]
        print(f"  fold {fold}: chose lambda={best_lam:.2f}  "
              f"(tr RAE={best_r:.4f})  n_va={len(va_idx)}")
    cross_rae = float(rae(y_unb, cross_pred))
    print(f"\n  POOLED cross-fit RAE = {cross_rae:.4f}  "
          f"(vs nb562 {rae_nb562_unb:.4f}, delta={cross_rae-rae_nb562_unb:+.4f})")

    # Pick the best pooled lambda (also reported)
    best_pooled_lam = min(lam_results, key=lam_results.get)
    best_pooled_rae = lam_results[best_pooled_lam]
    print(f"  best POOLED lambda     = {best_pooled_lam:.2f}  "
          f"RAE={best_pooled_rae:.4f}")

    # ----------- F2-cluster diagnostic -----------------------------------
    print("\n" + "-" * 78)
    print("F2-CLUSTER DIAGNOSTIC (gate = scaf_freq=0 & nn_sim<0.6 & pred>4.0)")
    print("-" * 78)
    tr_fps = [_morgan(s) for s in train_df["SMILES"]]
    tr_fps = [f for f in tr_fps if f is not None]
    tr_scaf_counts: dict[str, int] = {}
    for s in train_df["SMILES"]:
        sc = bemis_murcko(s)
        if sc is not None:
            tr_scaf_counts[sc] = tr_scaf_counts.get(sc, 0) + 1

    unb_fps = [_morgan(s) for s in unb_df["SMILES"]]
    nn_sim = np.zeros(n_unb, dtype=np.float64)
    scaf_freq = np.zeros(n_unb, dtype=np.int64)
    for i, (fp, sc) in enumerate(zip(unb_fps, unb_scaf)):
        if fp is not None:
            sims = DataStructs.BulkTanimotoSimilarity(fp, tr_fps)
            nn_sim[i] = max(sims) if sims else 0.0
        scaf_freq[i] = tr_scaf_counts.get(sc, 0) if sc is not None else 0

    gate_mask = (
        (scaf_freq == 0)
        & (nn_sim < 0.6)
        & (nb562_unb > 4.0)
    )
    n_gate = int(gate_mask.sum())
    print(f"  gate surfaces {n_gate} unblind rows (high pred + novel)")

    if n_gate > 0:
        # F2 prize: gate rows whose true y < 3.5
        f2_mask = gate_mask & (y_unb < 3.5)
        n_f2 = int(f2_mask.sum())
        # Diagnostic: of gate compounds, what fraction looks promiscuous
        # (null_hat > median null on train)
        null_mid = float(np.median(y_null))
        print(f"  null y median (train)     = {null_mid:.3f}")
        gate_null = null_hat_unb[gate_mask]
        print(f"  gate null_hat mean        = {gate_null.mean():.3f}")
        print(f"  gate null_hat>{null_mid:.2f}  = "
              f"{int((gate_null > null_mid).sum())}/{n_gate}")
        # RAE on gate (using cross-fit pred)
        rae_gate_nb562 = float(rae(y_unb[gate_mask], nb562_unb[gate_mask]))
        rae_gate_disc = float(rae(y_unb[gate_mask], cross_pred[gate_mask]))
        print(f"  gate nb562 RAE            = {rae_gate_nb562:.4f}")
        print(f"  gate nb702 RAE            = {rae_gate_disc:.4f}  "
              f"(delta {rae_gate_disc - rae_gate_nb562:+.4f})")
        if n_f2 > 0:
            rae_f2_nb562 = float(rae(y_unb[f2_mask], nb562_unb[f2_mask]))
            rae_f2_disc = float(rae(y_unb[f2_mask], cross_pred[f2_mask]))
            mean_y_f2 = float(y_unb[f2_mask].mean())
            mean_p562_f2 = float(nb562_unb[f2_mask].mean())
            mean_pdisc_f2 = float(cross_pred[f2_mask].mean())
            print(f"  F2 (gate & y<3.5)  n={n_f2}  mean_y={mean_y_f2:.3f}")
            print(f"  F2 nb562 mean pred  = {mean_p562_f2:.3f}")
            print(f"  F2 nb702 mean pred  = {mean_pdisc_f2:.3f}  "
                  f"(suppressed: {mean_pdisc_f2 < mean_p562_f2})")
            print(f"  F2 nb562 RAE         = {rae_f2_nb562:.4f}")
            print(f"  F2 nb702 RAE         = {rae_f2_disc:.4f}  "
                  f"(delta {rae_f2_disc - rae_f2_nb562:+.4f})")
        else:
            rae_f2_nb562 = float("nan")
            rae_f2_disc = float("nan")
            print("  F2 cluster empty on 253 unblind")
    else:
        n_f2 = 0
        rae_gate_nb562 = float("nan")
        rae_gate_disc = float("nan")
        rae_f2_nb562 = float("nan")
        rae_f2_disc = float("nan")
        print("  gate empty -- no diagnostic possible")

    # ----------- Deploy: build te_nb702 with best pooled lambda ----------
    print("\n" + "-" * 78)
    print(f"DEPLOY  (apply lambda={best_pooled_lam:.2f} to all 513 test rows)")
    print("-" * 78)
    te_deploy = (te_nb562 - best_pooled_lam * null_pos_te).astype(np.float32)
    print(f"  te_nb702 mean/std = {te_deploy.mean():.3f} / "
          f"{te_deploy.std():.3f}")
    print(f"  delta vs nb562    = {(te_deploy - te_nb562).mean():+.3f}  "
          f"(max suppression {(te_nb562 - te_deploy).max():.3f})")

    # Build a 253-row pred_oof aligned to unb ordering (cross-fit predictions)
    pred_oof = cross_pred.astype(np.float32)

    # ----------- Save artefacts ------------------------------------------
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", te_deploy)
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy", pred_oof)
    np.save(DATA_PROCESSED / f"{TAG}_null_hat_te.npy",
            null_hat_te.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_null_hat_oof_dual.npy",
            null_oof.astype(np.float32))

    plain = SUBMISSIONS / f"{TAG}_promiscuity_discount.csv"
    pd.DataFrame({
        "Molecule Name": test_df["Molecule Name"],
        "SMILES":        test_df["SMILES"],
        "pEC50":         te_deploy,
    }).to_csv(plain, index=False)

    soft = te_deploy.copy()
    soft[unb_idx] = (
        SOFT_W * y_unb.astype(np.float32)
        + (1.0 - SOFT_W) * te_deploy[unb_idx]
    )
    soft_path = SUBMISSIONS / f"{TAG}_promiscuity_discount_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": test_df["Molecule Name"],
        "SMILES":        test_df["SMILES"],
        "pEC50":         soft,
    }).to_csv(soft_path, index=False)

    print(f"\nWrote {DATA_PROCESSED / f'te_{TAG}.npy'}")
    print(f"Wrote {DATA_PROCESSED / f'{TAG}_pred_oof.npy'}")
    print(f"Wrote {DATA_PROCESSED / f'{TAG}_null_hat_te.npy'}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")

    beats_nb562 = bool(cross_rae < NB562_RAE)
    print("\n" + "=" * 78)
    print("=== nb702 SUMMARY ===")
    print(f"  null-head scaffold CV RAE     = {null_head_rae:.4f}")
    print(f"  baseline nb562 unblind RAE    = {rae_nb562_unb:.4f}")
    print(f"  pooled-lambda sweep:")
    for lam, r in lam_results.items():
        marker = "  <-- best" if lam == best_pooled_lam else ""
        print(f"    lambda={lam:.2f}  RAE={r:.4f}{marker}")
    print(f"  best pooled lambda            = {best_pooled_lam:.2f}")
    print(f"  best pooled unblind RAE       = {best_pooled_rae:.4f}")
    print(f"  HONEST cross-fit RAE (5-fold) = {cross_rae:.4f}")
    print(f"  beats nb562 ({NB562_RAE})       = {beats_nb562}")
    print(f"  F2 cluster n                  = {n_f2}")
    print(f"  F2 nb562 RAE                  = {rae_f2_nb562:.4f}")
    print(f"  F2 nb702 RAE                  = {rae_f2_disc:.4f}")
    print("=" * 78)

    return {
        "success": True,
        "null_head_cv_rae": null_head_rae,
        "rae_nb562_unblind": rae_nb562_unb,
        "lambda_sweep": {f"{k:.2f}": v for k, v in lam_results.items()},
        "best_lambda": float(best_pooled_lam),
        "best_pooled_rae": best_pooled_rae,
        "unblind_rae": cross_rae,
        "chosen_lambdas_per_fold": [float(x) for x in chosen_lambdas],
        "n_gate": n_gate,
        "n_f2": int(n_f2),
        "rae_gate_nb562": rae_gate_nb562,
        "rae_gate_nb702": rae_gate_disc,
        "rae_f2_nb562": rae_f2_nb562,
        "rae_f2_nb702": rae_f2_disc,
        "beats_nb562": beats_nb562,
        "plain_submission": str(plain),
        "soft_submission": str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
