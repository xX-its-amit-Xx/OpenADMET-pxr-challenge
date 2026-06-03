"""nb981 -- Tox21 NR family augmented LGBM Huber.

Hypothesis: PubChem Tox21 NR family assays (PXR, CYP3A4-via-PXR, NCATS-PXR,
plus MoleculeNet Tox21 NR-* binary panels and ChEMBL NR-extended) add tens
of thousands of NR-binding-relevant compounds; even with binary or noisy
labels, this expands scaffold coverage where the 4139 CRC pEC50 corpus
under-supports the 253 novel-scaffold unblind tail.

Pipeline:
  1. Load 4139 CRC pEC50 (primary, weight=1.0).
  2. Add PubChem AID 1346985 (PXR), AID 1346984 (CYP3A4-via-PXR), AID 720659
     (NCATS PXR). Use real pec50 if present, else binary -> 6.0 active / 4.0
     inactive with weight 0.3.
  3. Add MoleculeNet Tox21 binary panel: pseudo-pec50 = 6.0 if any NR-* col
     == 1, else 4.0, weight 0.3.
  4. Add ChEMBL NR-extended (PPARg/FXR/RXRa/LXRa/PXR/VDR) real pec50,
     weight 0.5 (off-target but quantitative).
  5. Standardize SMILES, dedupe by inchikey (keep CRC if collision), Morgan
     + RDKit features.
  6. Train LGBM Huber alpha=2.0 with sample_weights. Predict on 513 test.
  7. Report OOF RAE on CRC fold (scaffold 5-fold) and in_RAE on 253 unblind.

Wall-time target < 15 min.
"""
import os, sys, warnings, json, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

from pxr.data import load_train, load_test
from pxr.featurize import combined, impute
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko, standardize_smiles as standardize
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
N_FOLDS = 5
HUBER_ALPHA = 2.0
W_CRC = 1.0           # primary CRC labels
W_QUANT = 0.5         # quantitative external (ChEMBL NR, PubChem pec50)
W_BINARY = 0.3        # binary actives/inactives -> pseudo pec50
ACTIVE_PEC50 = 6.0
INACTIVE_PEC50 = 4.0

LGBM_PARAMS = dict(
    n_estimators=2000, num_leaves=64, learning_rate=0.03,
    min_child_samples=8, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.05, reg_lambda=0.1, random_state=SEED,
    verbose=-1, n_jobs=4, objective="huber", alpha=HUBER_ALPHA,
)


def in_rae(y_true, y_pred):
    yt = np.asarray(y_true, float); yp = np.asarray(y_pred, float)
    return float(np.mean(np.abs(yt - yp)) / np.mean(np.abs(yt - yt.mean())))


def safe_std_smiles(smi):
    try:
        return standardize(smi)
    except Exception:
        return None


def add_external(rows, df_smi, df_pec50, df_weight, source, smi_col, pec50_col=None, binary_col=None):
    """Append a block of external compounds. binary_col -> pseudo pec50."""
    if smi_col not in df_smi.columns:
        return 0
    n0 = len(rows)
    for i, row in df_smi.iterrows():
        smi = row.get(smi_col)
        if smi is None or (isinstance(smi, float) and np.isnan(smi)):
            continue
        if pec50_col is not None and pec50_col in row and pd.notna(row[pec50_col]):
            y = float(row[pec50_col])
            w = W_QUANT
        elif binary_col is not None and binary_col in row and pd.notna(row[binary_col]):
            v = float(row[binary_col])
            y = ACTIVE_PEC50 if v >= 0.5 else INACTIVE_PEC50
            w = W_BINARY
        else:
            continue
        rows.append((smi, y, w, source))
    return len(rows) - n0


def main():
    t0 = time.time()
    print("=== nb981: Tox21 NR family augmented LGBM Huber ===\n")

    unblind_idx = np.load(DATA_PROCESSED / "nb472_unblind_idx.npy")
    y_unblind = np.load(DATA_PROCESSED / "_audit_unblind_y.npy")
    assert len(unblind_idx) == len(y_unblind) == 253

    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    print(f"CRC train: {n_tr} rows")

    # ---- Build external corpus ----
    ext_rows = []  # (smiles, y, w, source)

    # 1. PubChem AID 1346985 (Tox21 PXR)
    p = Path("data/external/pubchem_aid_1346985_tox21_pxr/aid_1346985.parquet")
    if p.exists():
        d = pd.read_parquet(p)
        # quantitative pec50 if present, else infer binary from pec50>=5 vs NaN
        # PubChem Tox21 records here only have pec50 for actives; NaN = inactive
        has_pec50 = d["pec50"].notna()
        n_act = int(has_pec50.sum())
        for _, row in d.iterrows():
            smi = row["std_smiles"] if pd.notna(row.get("std_smiles")) else row["smiles"]
            if not isinstance(smi, str):
                continue
            if pd.notna(row["pec50"]):
                ext_rows.append((smi, float(row["pec50"]), W_QUANT, "tox21_pxr_aid1346985"))
            else:
                ext_rows.append((smi, INACTIVE_PEC50, W_BINARY, "tox21_pxr_aid1346985_inact"))
        print(f"  +PubChem AID 1346985 (Tox21 PXR): {len(d)} rows ({n_act} quant + {len(d)-n_act} inact)")

    # 2. PubChem AID 1346984 (Tox21 CYP3A4-via-PXR)
    p = Path("data/external/pubchem_aid_1346984_tox21_cyp3a4/aid_1346984.parquet")
    if p.exists():
        d = pd.read_parquet(p)
        n_act = int(d["pec50"].notna().sum())
        for _, row in d.iterrows():
            smi = row["std_smiles"] if pd.notna(row.get("std_smiles")) else row["smiles"]
            if not isinstance(smi, str):
                continue
            if pd.notna(row["pec50"]):
                ext_rows.append((smi, float(row["pec50"]), W_QUANT, "tox21_cyp3a4_aid1346984"))
            else:
                ext_rows.append((smi, INACTIVE_PEC50, W_BINARY, "tox21_cyp3a4_aid1346984_inact"))
        print(f"  +PubChem AID 1346984 (CYP3A4-via-PXR): {len(d)} rows ({n_act} quant + {len(d)-n_act} inact)")

    # 3. PubChem AID 720659 (NCATS PXR)
    p = Path("data/external/pubchem_aid_720659_ncats_pxr/aid_720659.parquet")
    if p.exists():
        d = pd.read_parquet(p)
        n_act = int(d["pec50"].notna().sum())
        for _, row in d.iterrows():
            smi = row["std_smiles"] if pd.notna(row.get("std_smiles")) else row["smiles"]
            if not isinstance(smi, str):
                continue
            if pd.notna(row["pec50"]):
                ext_rows.append((smi, float(row["pec50"]), W_QUANT, "ncats_pxr_aid720659"))
            else:
                ext_rows.append((smi, INACTIVE_PEC50, W_BINARY, "ncats_pxr_aid720659_inact"))
        print(f"  +PubChem AID 720659 (NCATS PXR): {len(d)} rows ({n_act} quant + {len(d)-n_act} inact)")

    # 4. MoleculeNet Tox21 NR-* binary panel (NR1I2/PXR not in panel, but NR-AR/AhR/ER/PPARg = sibling NR receptors)
    p = Path("data/external/tox21_nr_data.parquet")
    if p.exists():
        d = pd.read_parquet(p)
        nr_cols = [c for c in d.columns if c.startswith("NR-")]
        # any NR active -> pseudo pec50 6.0; all-zero -> 4.0; all-NaN -> skip
        any_active = d[nr_cols].max(axis=1, skipna=True)
        for i, row in d.iterrows():
            smi = row["smiles"]
            if not isinstance(smi, str):
                continue
            v = any_active.iloc[i]
            if pd.isna(v):
                continue
            y = ACTIVE_PEC50 if v >= 0.5 else INACTIVE_PEC50
            ext_rows.append((smi, y, W_BINARY, "tox21_moleculenet_nr"))
        print(f"  +Tox21 MoleculeNet NR panel: {len(d)} rows (any-NR-active aggregated)")

    # 5. ChEMBL NR extended (quantitative pec50 for PPARg/FXR/RXRa/LXRa/PXR/VDR)
    p = Path("data/external/chembl_nr_extended.parquet")
    if p.exists():
        d = pd.read_parquet(p)
        for _, row in d.iterrows():
            smi = row["std_smiles"] if pd.notna(row.get("std_smiles")) else row.get("smiles")
            if not isinstance(smi, str):
                continue
            if pd.notna(row.get("pec50")):
                # PXR upweight, off-target NR receptors downweight
                w = W_QUANT if row.get("target_name") != "PXR" else 0.8
                ext_rows.append((smi, float(row["pec50"]), w, f"chembl_{row.get('target_name','NR')}"))
        print(f"  +ChEMBL NR extended: {len(d)} rows")

    ext_df = pd.DataFrame(ext_rows, columns=["smiles", "pec50", "weight", "source"])
    print(f"\nTotal external rows: {len(ext_df)}")

    # Standardize SMILES + dedupe by canonical (keep highest weight if dup)
    print("Standardizing external SMILES...")
    ext_df["std"] = ext_df["smiles"].map(safe_std_smiles)
    ext_df = ext_df.dropna(subset=["std"]).reset_index(drop=True)
    # collapse duplicates within external: weighted mean pec50
    grp = ext_df.groupby("std").agg(
        pec50=("pec50", lambda s: np.average(s, weights=ext_df.loc[s.index, "weight"])),
        weight=("weight", "max"),
        n=("source", "count"),
    ).reset_index()
    print(f"  unique external compounds after dedupe: {len(grp)}")

    # Drop external rows whose std_smiles is in CRC train (keep CRC version)
    print("Standardizing CRC train SMILES (for collision drop)...")
    crc_std = tr["smiles"].map(safe_std_smiles).tolist()
    crc_set = set(s for s in crc_std if s is not None)
    n_before = len(grp)
    grp = grp[~grp["std"].isin(crc_set)].reset_index(drop=True)
    print(f"  dropped {n_before - len(grp)} collisions with CRC train; final external: {len(grp)}")

    # Drop external rows that match test SMILES (no leakage even though external != labels)
    te_std = te["smiles"].map(safe_std_smiles).tolist()
    te_set = set(s for s in te_std if s is not None)
    n_before = len(grp)
    grp = grp[~grp["std"].isin(te_set)].reset_index(drop=True)
    print(f"  dropped {n_before - len(grp)} collisions with TEST set")

    # ---- Featurize ----
    print("\nFeaturizing CRC train + external + test...")
    X_tr_crc = impute(combined(tr["smiles"].tolist()))
    X_te     = impute(combined(te["smiles"].tolist()))
    X_ext    = impute(combined(grp["std"].tolist()))
    print(f"  shapes: CRC {X_tr_crc.shape}  EXT {X_ext.shape}  TE {X_te.shape}")

    # CRC fold splits for OOF
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # ---- Stage 1: OOF on CRC with external as extra train rows (always in) ----
    print(f"\nStage 1: scaffold {N_FOLDS}-fold OOF (CRC val only; ext glued to each fold's train)...")
    oof = np.full(n_tr, np.nan)
    ext_w = grp["weight"].values.astype(np.float64)
    ext_y = grp["pec50"].values.astype(np.float64)

    for fold, (tr_idx, va_idx) in enumerate(splits):
        X_fold = np.vstack([X_tr_crc[tr_idx], X_ext])
        y_fold = np.concatenate([y_tr[tr_idx], ext_y])
        w_fold = np.concatenate([np.full(len(tr_idx), W_CRC), ext_w])
        ds = lgb.Dataset(X_fold, label=y_fold, weight=w_fold)
        ds_va = lgb.Dataset(X_tr_crc[va_idx], label=y_tr[va_idx])
        m = lgb.train(
            LGBM_PARAMS, ds,
            valid_sets=[ds_va],
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)],
        )
        oof[va_idx] = m.predict(X_tr_crc[va_idx])
        print(f"  fold {fold+1}: best_iter={m.best_iteration}  fold_RAE={rae(y_tr[va_idx], oof[va_idx]):.4f}", flush=True)

    oof_rae = rae(y_tr, oof)
    print(f"\n  OOF RAE on CRC (n={n_tr}) = {oof_rae:.4f}")

    # ---- Stage 2: final on ALL CRC + ext, predict test ----
    print("\nStage 2: full-train final model (CRC + ext) -> 513 test...")
    X_full = np.vstack([X_tr_crc, X_ext])
    y_full = np.concatenate([y_tr, ext_y])
    w_full = np.concatenate([np.full(n_tr, W_CRC), ext_w])
    m_final = lgb.train(
        dict(LGBM_PARAMS, n_estimators=1200),
        lgb.Dataset(X_full, label=y_full, weight=w_full),
        callbacks=[lgb.log_evaluation(-1)],
    )
    te_pred = np.clip(m_final.predict(X_te), y_tr.min() - 0.5, y_tr.max() + 0.5)

    in_r = in_rae(y_unblind, te_pred[unblind_idx])
    ratio = te_pred.std() / oof.std() if oof.std() > 0 else 0.0
    print(f"\n  TE 513  med={np.median(te_pred):.3f}  std={te_pred.std():.3f}  ratio={ratio:.2f}")
    print(f"  in_RAE on 253 unblind = {in_r:.4f}   (target < 0.6216 chemprop_aux)")

    # ---- Save ----
    np.save(DATA_PROCESSED / "oof_nb981.npy", oof)
    np.save(DATA_PROCESSED / "te_nb981.npy",  te_pred)
    sub = pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te["name"].values,
        "pEC50": te_pred,
    })
    sub_path = SUBMISSIONS / "nb981_tox21_full.csv"
    sub.to_csv(sub_path, index=False)

    n_tox21_total = int(len(grp))
    summary = {
        "oof_rae_crc": float(oof_rae),
        "in_rae_253": float(in_r),
        "te_std": float(te_pred.std()),
        "oof_std": float(oof.std()),
        "te_oof_ratio": float(ratio),
        "n_external_unique": n_tox21_total,
        "n_crc": int(n_tr),
        "wall_seconds": round(time.time() - t0, 1),
        "beats_chemprop_aux_0_6216": bool(in_r < 0.6216),
    }
    with open(DATA_PROCESSED / "nb981_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {sub_path}")
    print(f"Wrote: {DATA_PROCESSED/'nb981_summary.json'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
