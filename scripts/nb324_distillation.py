"""nb324 -- Distill nb320 ensemble into a single LGBM trained on (train + unblind).

Targets: nb320 top-50 SLSQP predictions on the 4139 train + 253 unblind.
Hypothesis: a single LGBM learning to MIMIC the ensemble on real OOD compounds
could generalise to the 260 still-blind compounds better than either nb320 alone
(needs all 50 component preds) or nb321_aug (LGBM-on-truth-labels).
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from rdkit import Chem
from scipy.stats import spearmanr

from pxr.data import load_train
from pxr.eval import rae
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


def std_smi(s):
    try:
        m = Chem.MolFromSmiles(str(s))
        return Chem.MolToSmiles(m) if m else None
    except: return None


def main():
    print("=== nb324: Distill nb320 ensemble into LGBM ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    y_tr = tr['pec50'].values.astype(np.float64)
    smiles_tr = tr['std_smiles'].tolist()
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    smiles_te = te_df['SMILES'].apply(std_smi).tolist()
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    unb_smis = unb['SMILES'].apply(std_smi).tolist()
    unb_y_true = unb['pEC50'].values
    name_to_idx = {n: i for i, n in enumerate(te_df['Molecule Name'])}
    unb_te_idx = np.array([name_to_idx[n] for n in unb['Molecule Name'] if n in name_to_idx])

    # Load nb320 OOF (for train compounds) AND TE (for test/unblind)
    # nb320 doesn't have an OOF (it's a test-only blend fit to truth); use nb302 OOF as proxy for train teacher
    nb320_te = np.load(DATA_PROCESSED / "te_nb320_phase2_top20.npy")
    nb302_oof = np.load(DATA_PROCESSED / "oof_nb302_full_pool.npy")
    print(f"nb320 test pred: shape={nb320_te.shape}  mean={nb320_te.mean():.3f}")
    print(f"nb302 train OOF: shape={nb302_oof.shape}  mean={nb302_oof.mean():.3f}")

    # Build student training set: 4139 train + 253 unblind
    # Train targets = TRUE labels (we want student to predict TRUE pec50, not nb302's OOF)
    student_smis = list(smiles_tr) + list(unb_smis)
    student_y = np.concatenate([y_tr, unb_y_true])
    print(f"\nStudent train: {len(student_smis)} compounds (4139 train + 253 unblind)")

    print("Featurising...")
    X_tr = impute(combined(student_smis)).astype(np.float32)
    X_te = impute(combined(smiles_te)).astype(np.float32)
    print(f"X_tr {X_tr.shape}  X_te {X_te.shape}")

    LGBM = dict(n_estimators=2000, num_leaves=63, learning_rate=0.025, subsample=0.8,
                colsample_bytree=0.8, min_child_samples=10, objective='mae',
                n_jobs=4, random_state=42, verbose=-1)
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(student_y))
    cut = int(0.9 * len(student_y))
    tr_idx = perm[:cut]; va_idx = perm[cut:]
    md = lgb.LGBMRegressor(**LGBM)
    md.fit(X_tr[tr_idx], student_y[tr_idx],
           eval_set=[(X_tr[va_idx], student_y[va_idx])],
           callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(-1)])
    vp = md.predict(X_tr[va_idx])
    print(f"\nVal RAE (10% holdout, random): {rae(student_y[va_idx], vp):.4f}")
    student_te = md.predict(X_te)
    print(f"Student te: mean={student_te.mean():.3f}  std={student_te.std():.3f}")

    # Sanity: RAE on unblind 253 (LEAKY since they're in train)
    print(f"\nOn unblind 253 (leaky): RAE={rae(unb_y_true, student_te[unb_te_idx]):.4f}  "
          f"Spearman={spearmanr(unb_y_true, student_te[unb_te_idx])[0]:.4f}")

    # Save submission
    sub = pd.DataFrame({
        'Molecule Name': te_df['Molecule Name'],
        'SMILES': te_df['SMILES'],
        'pEC50': student_te,
    })
    out = SUBMISSIONS / "nb324_distilled.csv"
    sub.to_csv(out, index=False)
    print(f"Wrote {out.name}")
    np.save(DATA_PROCESSED / "te_nb324_distilled.npy", student_te)

    # Blend with nb320 to see if distillation adds value
    print("\n=== Blend student + nb320 ===")
    for w in [0.0, 0.2, 0.3, 0.5, 0.7, 1.0]:
        blend = (1 - w) * nb320_te + w * student_te
        r = rae(unb_y_true, blend[unb_te_idx])
        print(f"  w(student)={w}: unblind RAE={r:.4f}  te_std={blend.std():.3f}")


if __name__ == "__main__":
    main()
