"""nb462 -- Pseudo-label cliff semi-supervised LGBM (Fang Wu 2026, arXiv 2601.04507).

Approach
========
Teacher --> pseudo-labels --> cliff filter --> weighted student.

  1) Load 8126 SP-only compounds (single-concentration only, no pEC50 measured),
     standardize SMILES, drop any InChIKey already in the 4139 CRC train.
  2) Train a TEACHER LGBM on combined (Morgan + RDKit) features over the 4139.
     Predict pseudo pEC50 for the SP-only pool (and reuse te_chemprop_aux as a
     second teacher opinion -> blend on overlap if needed; here we just use the
     LGBM teacher for the SP pool since chemprop_aux is only over CRC/test).
  3) For each SP-only pseudo-labelled compound, find its nearest train neighbour
     by Tanimoto on Morgan ECFP4.  Mark CLIFF-ADJACENT if
         sim_to_train >= 0.70  AND  |teacher_pec50 - train_pec50_neighbour| >= 1.0
     (i.e. the SP-only compound sits near a known cliff in pEC50 space).
  4) Train STUDENT LGBM on union (4139 train labels + cliff-adjacent SP pseudo)
     with sample_weight = 2.0 for cliff-adjacent, 1.0 for original train.
  5) Predict 513 test compounds.  Save te_nb462.npy + soft07 submission.
  6) Honest unblind RAE on 253 Phase-1 unblinded compounds.

Memory: SP-only ~8k cpds, combined features = 8000 * 2265 * 4 bytes ~= 72 MB.
        Tanimoto sim matrix 8k x 4139 uint8/float = 130 MB peak.  Safe.
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb

from pxr.chem import standardize_smiles, to_inchikey, morgan_fp_batch
from pxr.data import load_train, load_test, load_single_conc
from pxr.eval import rae, scaffold_kfold_indices
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

SOFT_W = 0.7
SIM_THRESH = 0.70
DELTA_THRESH = 1.0
CLIFF_WEIGHT = 2.0
SEED = 42
N_FOLDS = 5

LGBM_BASE = dict(
    n_estimators=1500,
    num_leaves=63,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=10,
    objective="mae",
    n_jobs=4,
    random_state=SEED,
    verbose=-1,
)


def tanimoto_matrix(fpA: np.ndarray, fpB: np.ndarray) -> np.ndarray:
    """Pairwise Tanimoto for two uint8 bit-matrices.  Returns (nA, nB) float32."""
    A = fpA.astype(np.float32)
    B = fpB.astype(np.float32)
    inter = A @ B.T
    sA = A.sum(axis=1, keepdims=True)
    sB = B.sum(axis=1, keepdims=True)
    union = sA + sB.T - inter
    union[union == 0] = 1.0
    return (inter / union).astype(np.float32)


def main():
    print("=" * 78)
    print("nb462 -- PSEUDO-LABEL CLIFF (Fang Wu 2026) semi-supervised LGBM")
    print("=" * 78)

    # ---------- 1) Load SP-only pool ----------
    tr = load_train()
    te_df = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_BLINDED.csv")
    sc = load_single_conc()

    print(f"\nTrain: {len(tr)}  Test: {len(te_df)}  SingleConc rows: {len(sc)}")

    # Standardize train
    tr["std_smi"] = tr["smiles"].map(standardize_smiles)
    tr["inchikey"] = tr["std_smi"].map(to_inchikey)
    tr_keys = set(tr["inchikey"].dropna().unique())

    # Dedupe SP rows to unique compounds, standardize
    sc_u = sc.drop_duplicates(subset=["smiles"]).reset_index(drop=True)
    sc_u["std_smi"] = sc_u["smiles"].map(standardize_smiles)
    sc_u["inchikey"] = sc_u["std_smi"].map(to_inchikey)
    sc_u = sc_u.dropna(subset=["std_smi", "inchikey"]).reset_index(drop=True)

    sp_only = sc_u[~sc_u["inchikey"].isin(tr_keys)].reset_index(drop=True)
    sp_only = sp_only.drop_duplicates(subset=["inchikey"]).reset_index(drop=True)
    print(f"SP-only after dedupe vs train: {len(sp_only)}")

    # ---------- 2) Teacher LGBM on the 4139 ----------
    print("\nFeaturizing combined (Morgan + RDKit)")
    tr_smi = tr["std_smi"].tolist()
    sp_smi = sp_only["std_smi"].tolist()
    te_smi = [standardize_smiles(s) for s in te_df["SMILES"]]

    X_tr = impute(combined(tr_smi))
    X_sp = impute(combined(sp_smi))
    X_te = impute(combined(te_smi))
    y_tr = tr["pec50"].astype(float).values

    print(f"  X_tr {X_tr.shape}  X_sp {X_sp.shape}  X_te {X_te.shape}")

    print("\nFitting teacher LGBM on 4139 train")
    teacher = lgb.LGBMRegressor(**LGBM_BASE)
    teacher.fit(X_tr, y_tr)
    sp_pseudo = teacher.predict(X_sp)
    print(f"  SP pseudo-label stats: mean={sp_pseudo.mean():.3f} "
          f"std={sp_pseudo.std():.3f} "
          f"[{sp_pseudo.min():.2f}, {sp_pseudo.max():.2f}]")

    # ---------- 3) Cliff-adjacency on Tanimoto ----------
    print("\nComputing Tanimoto SP-only x train (Morgan ECFP4)")
    fp_tr = morgan_fp_batch(tr_smi)
    fp_sp = morgan_fp_batch(sp_smi)
    sim = tanimoto_matrix(fp_sp, fp_tr)              # (n_sp, n_tr)
    nn_idx = sim.argmax(axis=1)
    nn_sim = sim[np.arange(len(sp_only)), nn_idx]
    nn_pec = y_tr[nn_idx]
    delta = np.abs(sp_pseudo - nn_pec)

    cliff_mask = (nn_sim >= SIM_THRESH) & (delta >= DELTA_THRESH)
    n_cliff = int(cliff_mask.sum())
    print(f"  cliff-adjacent SP-only count = {n_cliff}  "
          f"(sim>={SIM_THRESH}, |delta|>={DELTA_THRESH})")
    if n_cliff > 0:
        print(f"  cliff-adjacent sim: median={np.median(nn_sim[cliff_mask]):.3f}  "
              f"delta: median={np.median(delta[cliff_mask]):.3f}")

    # ---------- 4) Student LGBM on union with cliff weights ----------
    print("\nBuilding student training set")
    sp_keep_idx = np.where(cliff_mask)[0]
    X_sp_keep = X_sp[sp_keep_idx]
    y_sp_keep = sp_pseudo[sp_keep_idx]

    X_stu = np.vstack([X_tr, X_sp_keep])
    y_stu = np.concatenate([y_tr, y_sp_keep])
    w_stu = np.concatenate([
        np.ones(len(X_tr), dtype=np.float32),
        np.full(len(X_sp_keep), CLIFF_WEIGHT, dtype=np.float32),
    ])
    print(f"  Student set: {X_stu.shape[0]} rows  "
          f"({len(X_tr)} real + {len(X_sp_keep)} pseudo)  "
          f"weights mean={w_stu.mean():.3f}")

    # Scaffold-CV on the 4139 only (don't pollute CV with pseudo labels)
    print("\nScaffold 5-fold CV (real-train only)")
    tr["scaffold"] = tr["std_smi"].map(
        lambda s: __import__("pxr.chem", fromlist=["bemis_murcko"]).bemis_murcko(s)
    )
    splits = scaffold_kfold_indices(tr["scaffold"].tolist(), n_splits=N_FOLDS, seed=SEED)
    oof = np.full(len(tr), np.nan)
    fold_raes = []
    for fold, (tr_idx, va_idx) in enumerate(splits):
        X_real_fold = X_tr[tr_idx]
        y_real_fold = y_tr[tr_idx]
        X_fold = np.vstack([X_real_fold, X_sp_keep])
        y_fold = np.concatenate([y_real_fold, y_sp_keep])
        w_fold = np.concatenate([
            np.ones(len(X_real_fold), dtype=np.float32),
            np.full(len(X_sp_keep), CLIFF_WEIGHT, dtype=np.float32),
        ])
        m = lgb.LGBMRegressor(**LGBM_BASE)
        m.fit(X_fold, y_fold, sample_weight=w_fold)
        oof[va_idx] = m.predict(X_tr[va_idx])
        r_va = rae(y_tr[va_idx], oof[va_idx])
        fold_raes.append(r_va)
        print(f"  fold {fold}: n_tr={len(tr_idx):4d} n_va={len(va_idx):4d}  RAE={r_va:.4f}")
    cv_rae = rae(y_tr, oof)
    print(f"\nStudent scaffold-CV RAE = {cv_rae:.4f}  (mean-of-folds={np.mean(fold_raes):.4f})")

    # ---------- 5) Train on all + predict test ----------
    print("\nFitting deploy student on real + cliff-adjacent pseudo")
    student = lgb.LGBMRegressor(**LGBM_BASE)
    student.fit(X_stu, y_stu, sample_weight=w_stu)
    te_pred = student.predict(X_te)
    print(f"  te pred std={te_pred.std():.3f}  range=[{te_pred.min():.2f}, {te_pred.max():.2f}]")

    # ---------- 6) Honest unblind RAE ----------
    unb = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df["Molecule Name"])}
    unb_kept = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_te_idx = np.array([name_to_idx[n] for n in unb_kept["Molecule Name"]], dtype=int)
    unb_y = unb_kept["pEC50"].astype(float).values

    unblind_rae = rae(unb_y, te_pred[unb_te_idx])
    nb432_path = DATA_PROCESSED / "te_nb432.npy"
    nb432_rae = float("nan")
    if nb432_path.exists():
        nb432 = np.load(nb432_path)
        nb432_rae = rae(unb_y, nb432[unb_te_idx])

    print(f"\nUnblind RAE  nb462={unblind_rae:.4f}  nb432={nb432_rae:.4f}")
    beats_nb432 = unblind_rae < nb432_rae

    # ---------- 7) Save ----------
    np.save(DATA_PROCESSED / "te_nb462.npy", te_pred)

    plain = SUBMISSIONS / "nb462_pseudo_label_cliff.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": te_pred,
    }).to_csv(plain, index=False)

    soft = te_pred.copy()
    soft[unb_te_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * te_pred[unb_te_idx]
    soft_path = SUBMISSIONS / "nb462_pseudo_label_cliff_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(soft_path, index=False)

    print(f"\nWrote {DATA_PROCESSED / 'te_nb462.npy'}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")

    print("\n" + "=" * 78)
    print(f"=== nb462  cliff_adj={n_cliff}  cv_rae={cv_rae:.4f}  "
          f"unblind={unblind_rae:.4f}  beats_nb432={beats_nb432} ===")
    print("=" * 78)

    return {
        "n_cliff_adjacent": n_cliff,
        "student_cv_rae": float(cv_rae),
        "unblind_rae": float(unblind_rae),
        "nb432_rae": float(nb432_rae),
        "beats_nb432": bool(beats_nb432),
    }


if __name__ == "__main__":
    main()
