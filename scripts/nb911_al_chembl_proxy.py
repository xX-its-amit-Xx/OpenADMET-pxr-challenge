"""nb911 -- Active-learning ChEMBL-PXR proxy.

Simulates the Phase-2 unblind drop pattern using ChEMBL PXR (CHEMBL3401).

Steps:
  1. Load ChEMBL PXR pec50 (target_name=='PXR' in chembl_nr_extended.parquet).
  2. Split: 1000 random "labeled pool", rest "unlabeled".
  3. Compute Tanimoto-to-labeled acquisition scores for the 513 test compounds
     (mean and max Tanimoto to the 1000 labeled).
  4. Train a final LGBM on the 4139 CRC train + 1000 ChEMBL pool jointly, with
     the acquisition score as an extra feature (so the model knows which
     compounds are well-supported vs OOD and can adjust).
  5. Score on the 253 unblind compounds; write submission.

Hypothesis: joint training + an "is-this-well-supported?" signal lets the
model temper extrapolation for OOD test compounds.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns, morgan_fp_batch
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


SEED = 42
N_LABELED = 1000

LGBM = dict(n_estimators=1500, num_leaves=63, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
            objective='mae', n_jobs=4, random_state=SEED, verbose=-1)


def mean_max_tanimoto(fp_query: np.ndarray, fp_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised Tanimoto: mean and max similarity of each query row vs reference set."""
    q = fp_query.astype(np.float32)
    r = fp_ref.astype(np.float32)
    inter = q @ r.T
    q_sum = q.sum(axis=1, keepdims=True)
    r_sum = r.sum(axis=1, keepdims=True).T
    union = q_sum + r_sum - inter
    sim = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
    return sim.mean(axis=1), sim.max(axis=1)


def main():
    print("=== nb911: ChEMBL-PXR active-learning proxy ===\n")

    # ---------- ChEMBL PXR pool ----------
    ch = pd.read_parquet("data/external/chembl_nr_extended.parquet")
    ch = ch[(ch['target_name'] == 'PXR') & ch['pec50'].notna() & ch['std_smiles'].notna()]
    ch = ch.drop_duplicates(subset='std_smiles').reset_index(drop=True)
    print(f"ChEMBL PXR pool: {len(ch)} unique compounds, pec50 mean={ch['pec50'].mean():.2f}")

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(ch))
    n_lab = min(N_LABELED, len(ch))
    lab_idx = perm[:n_lab]
    unl_idx = perm[n_lab:]
    ch_lab = ch.iloc[lab_idx].reset_index(drop=True)
    ch_unl = ch.iloc[unl_idx].reset_index(drop=True)
    print(f"Labeled: {len(ch_lab)}  Unlabeled: {len(ch_unl)}")

    # ---------- CRC train + test ----------
    tr = load_train(); tr = add_standard_columns(tr)
    smiles_tr = tr['std_smiles'].tolist()
    y_tr = tr['pec50'].values.astype(np.float64)
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    smiles_te = te_df['SMILES'].tolist()
    te_names = te_df['Molecule Name'].tolist()

    # ---------- Fingerprints + acquisition score ----------
    print("\nMorgan fingerprints...")
    fp_lab = morgan_fp_batch(ch_lab['std_smiles'].tolist())
    fp_tr = morgan_fp_batch(smiles_tr)
    fp_te = morgan_fp_batch(smiles_te)
    fp_chunl = morgan_fp_batch(ch_unl['std_smiles'].tolist())

    print("Acquisition: Tanimoto vs labeled pool...")
    mean_tr, max_tr = mean_max_tanimoto(fp_tr, fp_lab)
    mean_te, max_te = mean_max_tanimoto(fp_te, fp_lab)
    mean_chl, max_chl = mean_max_tanimoto(fp_lab, fp_lab)  # self
    mean_chu, max_chu = mean_max_tanimoto(fp_chunl, fp_lab)
    print(f"  test mean_tan: median={np.median(mean_te):.3f}  max_tan median={np.median(max_te):.3f}")
    print(f"  train mean_tan: median={np.median(mean_tr):.3f}")

    # Lower mean_tan == more OOD. Use as feature; rank is monotone.
    def acq_rank(x):
        # higher rank = more representative
        return np.argsort(np.argsort(x)).astype(np.float32) / max(len(x) - 1, 1)

    # ---------- Initial LGBM on labeled pool (sanity check) ----------
    X_lab = impute(combined(ch_lab['std_smiles'].tolist())).astype(np.float32)
    y_lab = ch_lab['pec50'].values.astype(np.float64)
    print("\nInitial LGBM on 1000 ChEMBL labeled (5-fold random CV)...")
    kf_idx = np.array_split(rng.permutation(len(y_lab)), 5)
    oof_lab = np.zeros(len(y_lab))
    for k in range(5):
        vi = kf_idx[k]; ti = np.concatenate([kf_idx[j] for j in range(5) if j != k])
        m = lgb.LGBMRegressor(**LGBM)
        m.fit(X_lab[ti], y_lab[ti], eval_set=[(X_lab[vi], y_lab[vi])],
              callbacks=[lgb.early_stopping(60, verbose=False)])
        oof_lab[vi] = m.predict(X_lab[vi])
    print(f"  ChEMBL-labeled OOF RAE={rae(y_lab, oof_lab):.4f}  Pearson={np.corrcoef(y_lab, oof_lab)[0,1]:.3f}")

    # ---------- Joint train: 4139 CRC + 1000 ChEMBL ----------
    print("\nJoint LGBM (CRC + ChEMBL pool) + acquisition feature...")
    smiles_join = smiles_tr + ch_lab['std_smiles'].tolist()
    y_join = np.concatenate([y_tr, y_lab])
    X_join_base = impute(combined(smiles_join)).astype(np.float32)
    X_te_base = impute(combined(smiles_te)).astype(np.float32)

    # Acquisition features: mean_tan, max_tan, acq_rank, and a binary "is_crc" source flag
    src_flag_join = np.concatenate([np.ones(len(smiles_tr)), np.zeros(len(ch_lab))])
    mean_tan_join = np.concatenate([mean_tr, mean_chl])
    max_tan_join = np.concatenate([max_tr, max_chl])
    rank_join = acq_rank(mean_tan_join)

    src_flag_te = np.ones(len(smiles_te))  # treat test as CRC-domain
    rank_te = np.interp(mean_te, np.sort(mean_tan_join), np.sort(rank_join))

    extra_join = np.column_stack([mean_tan_join, max_tan_join, rank_join, src_flag_join])
    extra_te = np.column_stack([mean_te, max_te, rank_te, src_flag_te])
    X_join = np.column_stack([X_join_base, extra_join])
    X_te = np.column_stack([X_te_base, extra_te])

    # Sample weight: down-weight ChEMBL so it informs but does not dominate
    sw = np.where(src_flag_join == 1, 1.0, 0.5)

    # Scaffold CV on CRC portion only (folds use tr['scaffold'])
    folds = scaffold_kfold_indices(tr['scaffold'].tolist(), n_splits=5)
    oof = np.zeros(len(y_tr))
    te_preds = []
    for k, (ti_crc, vi_crc) in enumerate(folds):
        # training rows: CRC train fold + all ChEMBL labeled
        ti = np.concatenate([ti_crc, np.arange(len(y_tr), len(y_join))])
        m = lgb.LGBMRegressor(**LGBM)
        m.fit(X_join[ti], y_join[ti], sample_weight=sw[ti],
              eval_set=[(X_join[vi_crc], y_join[vi_crc])],
              callbacks=[lgb.early_stopping(80, verbose=False)])
        oof[vi_crc] = m.predict(X_join[vi_crc])
        te_preds.append(m.predict(X_te))
    te_pred = np.mean(te_preds, axis=0)
    oof_rae = rae(y_tr, oof)
    print(f"  Scaffold-CV OOF RAE on 4139 CRC: {oof_rae:.4f}  te_std={te_pred.std():.3f}")

    # ---------- in-RAE on 253 unblind ----------
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    unb_y = np.load(DATA_PROCESSED / "_audit_unblind_y.npy")
    in_rae = rae(unb_y, te_pred[unb_idx])
    print(f"\nUnblind in_RAE (253): {in_rae:.4f}")
    print(f"  preds on unblind: mean={te_pred[unb_idx].mean():.3f}  std={te_pred[unb_idx].std():.3f}")
    print(f"  truth on unblind: mean={unb_y.mean():.3f}  std={unb_y.std():.3f}")

    # ---------- Save artefacts ----------
    np.save(DATA_PROCESSED / "oof_nb911_al_chembl.npy", oof)
    np.save(DATA_PROCESSED / "te_nb911_al_chembl.npy", te_pred)
    np.save(DATA_PROCESSED / "te_nb911_acq_mean_tan.npy", mean_te)
    np.save(DATA_PROCESSED / "te_nb911_acq_max_tan.npy", max_te)

    sub = pd.DataFrame({
        'Molecule Name': te_names,
        'SMILES': te_df['SMILES'],
        'pEC50': te_pred,
    })
    out = SUBMISSIONS / "nb911_al_chembl_proxy.csv"
    sub.to_csv(out, index=False)
    print(f"\nWrote {out}")
    print(f"Summary: CRC-scaffold-OOF={oof_rae:.4f}  unblind_in_RAE={in_rae:.4f}  "
          f"n_chembl_pool={len(ch_lab)}  acq_features=4")


if __name__ == "__main__":
    main()
