"""nb321 -- Retrain top-5 Phase 2 contributors with augmented training data.

After Phase 1 unblind we have:
  - Original train: 4139 CRC compounds
  - Phase 1 unblind: 253 newly-labeled test compounds
  - 96-compound micro-scale semi-pure batch (Phase 2 release)
  - 457 high-throughput chemistry library compounds (Phase 2 release)

Total augmented train: 4139 + 253 + 96 + 457 = 4945 compounds.

Retrain a STREAMLINED LGBM (proxy for nb93/nb264 Chemprop-style trained on
the augmented set), predict on the 260 still-blinded test compounds. Compare
to the blind 250 from the augmented dataset.

For the Chemprop-style models (nb93, nb264, chemprop_aux), we just produce
augmented-train predictions via LGBM proxy — the true Chemprop retraining
needs GPU; here we just produce a tractable approximation.

The dominant goal: produce 2 new submission CSVs:
  1. nb321_augmented_lgbm.csv — single LGBM on full augmented set
  2. nb321_augmented_blend.csv — re-blend top-50 from nb320 with augmented LGBM
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
from scipy.optimize import minimize

from pxr.eval import rae
from pxr.chem import standardize
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


def std_smi(s):
    try:
        m = Chem.MolFromSmiles(str(s))
        return Chem.MolToSmiles(m) if m else None
    except: return None


def main():
    print("=== nb321: augmented retrain ===\n")
    # Original train
    tr_orig = pd.read_csv("data/raw/pxr-challenge_TRAIN.csv")
    print(f"Original train: {len(tr_orig)}")
    # Phase 1 unblind
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    print(f"Phase 1 unblind: {len(unb)}")
    # 96 micro-scale
    mu = pd.read_csv("data/raw/pxr-challenge_96-compound-uscale-semi-pure_TRAIN.csv")
    print(f"96-compound micro-scale: {len(mu)}")
    # HT chem libraries
    ht = pd.read_csv("data/raw/pxr-challenge_htchem-libraries_TRAIN.csv")
    print(f"HT chem libraries: {len(ht)}")

    # Build unified train set
    rows = []
    for _, r in tr_orig.iterrows():
        rows.append((r['SMILES'], r['pEC50'], 'orig'))
    for _, r in unb.iterrows():
        rows.append((r['SMILES'], r['pEC50'], 'unblind'))
    # 96 micro-scale: use corrected semi-pure pEC50 (string -> float coerce)
    mu['_pec'] = pd.to_numeric(mu['Corrected Semi-Pure pEC50 (log)'], errors='coerce')
    for _, r in mu.iterrows():
        if pd.notna(r['_pec']):
            rows.append((r['SMILES'], float(r['_pec']), 'uscale'))
    # HT chem: use corrected crude pEC50 (string -> float coerce)
    ht['_pec'] = pd.to_numeric(ht['Corrected Crude pEC50 (log)'], errors='coerce')
    for _, r in ht.iterrows():
        if pd.notna(r['_pec']):
            rows.append((r['SMILES'], float(r['_pec']), 'htchem'))
    df = pd.DataFrame(rows, columns=['SMILES', 'pec50', 'source']).dropna(subset=['SMILES', 'pec50'])
    df['std_smiles'] = df['SMILES'].apply(std_smi)
    df = df.dropna(subset=['std_smiles']).reset_index(drop=True)
    print(f"\nUnified train: {len(df)} rows")
    print(df['source'].value_counts().to_string())
    print(f"pec50: mean={df['pec50'].mean():.3f} std={df['pec50'].std():.3f}")

    # Test set (blinded portion remaining)
    te_blind = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    te_blind['std_smiles'] = te_blind['SMILES'].apply(std_smi)
    print(f"\nFull test: {len(te_blind)}, of which unblinded: {len(unb)}")

    # ============================
    # Featurize + LGBM on augmented set
    # ============================
    print("\nFeaturizing...")
    X_tr = impute(combined(df['std_smiles'].tolist())).astype(np.float32)
    X_te = impute(combined(te_blind['std_smiles'].tolist())).astype(np.float32)
    y = df['pec50'].values.astype(np.float64)
    print(f"X_tr {X_tr.shape}  X_te {X_te.shape}")

    LGBM = dict(n_estimators=2000, num_leaves=63, learning_rate=0.03, subsample=0.8,
                colsample_bytree=0.8, min_child_samples=10, objective='mae',
                n_jobs=4, random_state=42, verbose=-1)
    # Hold out 10% random as val
    rng = np.random.default_rng(42)
    n = len(y); perm = rng.permutation(n)
    cut = int(0.9 * n)
    tr_idx = perm[:cut]; va_idx = perm[cut:]
    md = lgb.LGBMRegressor(**LGBM)
    md.fit(X_tr[tr_idx], y[tr_idx], eval_set=[(X_tr[va_idx], y[va_idx])],
           callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
    vp = md.predict(X_tr[va_idx])
    vrae = rae(y[va_idx], vp)
    print(f"\nValidation RAE (random 10% holdout): {vrae:.4f}  Spearman: {spearmanr(y[va_idx], vp)[0]:.4f}")
    te_pred = md.predict(X_te)
    print(f"Test pred: mean={te_pred.mean():.3f} std={te_pred.std():.3f}")

    # Save submission
    sub = pd.DataFrame({
        'Molecule Name': te_blind['Molecule Name'],
        'SMILES': te_blind['SMILES'],
        'pEC50': te_pred,
    })
    out = SUBMISSIONS / "nb321_augmented_lgbm.csv"
    sub.to_csv(out, index=False)
    print(f"Wrote {out.name}")
    np.save(DATA_PROCESSED / "te_nb321_augmented.npy", te_pred)

    # Also evaluate on the unblinded 253 (now that they're in train, this is leak —
    # but we can compare LGBM-aug's prediction on the unblinded 253 BEFORE re-fitting,
    # since LGBM is hard to remove fold-level — we already saw via te_nb320_phase2_top50.npy)
    # For honesty, compare against the already-saved blind predictions.
    print("\n=== Sanity check vs known Phase 1 unblind ===")
    name_to_idx = {n: i for i, n in enumerate(te_blind['Molecule Name'])}
    unb_idx = np.array([name_to_idx[n] for n in unb['Molecule Name'] if n in name_to_idx])
    unb_y = np.array([r['pEC50'] for _, r in unb.iterrows() if r['Molecule Name'] in name_to_idx])
    print(f"On {len(unb_idx)} unblinded test compounds:")
    print(f"  nb321_aug LGBM RAE: {rae(unb_y, te_pred[unb_idx]):.4f}  Spearman: {spearmanr(unb_y, te_pred[unb_idx])[0]:.4f}")
    print(f"    (NOTE: leaky — these compounds are in the augmented train)")

    # ============================
    # Blend nb321_augmented with nb320 top-50 (the actual best truth-fitted blend)
    # ============================
    print("\n=== Blend nb321_aug with nb320 top-50 ===")
    nb320 = np.load(DATA_PROCESSED / "te_nb320_phase2_top20.npy")
    for w_new in [0.2, 0.3, 0.4, 0.5]:
        blend = (1 - w_new) * nb320 + w_new * te_pred
        r = rae(unb_y, blend[unb_idx])
        print(f"  w(nb321_aug)={w_new}: RAE on unblind 253 = {r:.4f}  te_std={blend.std():.3f}")
    # Pick conservative 0.3 blend as candidate
    blend03 = 0.7 * nb320 + 0.3 * te_pred
    sub2 = pd.DataFrame({
        'Molecule Name': te_blind['Molecule Name'],
        'SMILES': te_blind['SMILES'],
        'pEC50': blend03,
    })
    out2 = SUBMISSIONS / "nb321_aug_blend03.csv"
    sub2.to_csv(out2, index=False)
    print(f"Wrote {out2.name}")


if __name__ == "__main__":
    main()
