"""nb1044 - PubChem PXR (Tox21 AID 1346985 + NCATS AID 720659) join.

Hypothesis: external PXR activation data (~4.7k rows, but ~1063 with pec50)
adds scaffold-diverse, weakly-supervised signal. We merge the two AIDs by
inchikey, average pec50 across them where both are non-null, and weight
each external row at 0.6 (vs 1.0 for CRC). Train a Huber LGBM identical
to nb972 on the union, evaluate honest scaffold-CV OOF on the 4139 CRC
block, and report in_RAE on the 253 unblind. Pearson check vs nb972 te
predictions; if < 0.95, bag 50/50 with nb972.

Outputs
  data/processed/te_nb1044_pubchem_join.npy
  data/processed/nb1044_summary.json
  submissions/nb1044_pubchem_pxr_join.csv
"""
import os, sys, json, time, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb

from pxr.data import load_train, load_test
from pxr.featurize import combined, impute
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko, standardize_smiles
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
N_FOLDS = 5
EXT_W = 0.6      # external-row weight in train
BLEND_PEARSON_THRESH = 0.95

PARAMS = dict(
    objective="huber",
    alpha=2.0,
    n_estimators=10000,
    learning_rate=0.005,
    num_leaves=64,
    min_child_samples=20,
    reg_lambda=0.2,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    verbose=-1,
    n_jobs=4,
)
EARLY_STOP = 500

TOX21_PATH = "data/external/pubchem_aid_1346985_tox21_pxr/aid_1346985.parquet"
NCATS_PATH = "data/external/pubchem_aid_720659_ncats_pxr/aid_720659.parquet"


def in_rae(y_true, y_pred):
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    return float(np.mean(np.abs(yt - yp)) / np.mean(np.abs(yt - yt.mean())))


def load_pubchem_pxr():
    """Load both AIDs, keep non-null pec50, average by inchikey."""
    repo = os.path.join(os.path.dirname(__file__), "..")
    tox = pd.read_parquet(os.path.join(repo, TOX21_PATH))
    nca = pd.read_parquet(os.path.join(repo, NCATS_PATH))
    print(f"  TOX21 raw={len(tox)}  with pec50={tox['pec50'].notna().sum()}")
    print(f"  NCATS raw={len(nca)}  with pec50={nca['pec50'].notna().sum()}")

    # Keep only rows with valid pec50 + std_smiles + inchikey
    tox = tox.dropna(subset=["pec50", "std_smiles", "inchikey"])[
        ["std_smiles", "inchikey", "pec50"]].rename(columns={"pec50": "pec50_tox"})
    nca = nca.dropna(subset=["pec50", "std_smiles", "inchikey"])[
        ["std_smiles", "inchikey", "pec50"]].rename(columns={"pec50": "pec50_nca"})

    # Group within source by inchikey (median, take first smiles)
    tox = tox.groupby("inchikey", as_index=False).agg(
        std_smiles=("std_smiles", "first"), pec50_tox=("pec50_tox", "median"))
    nca = nca.groupby("inchikey", as_index=False).agg(
        std_smiles=("std_smiles", "first"), pec50_nca=("pec50_nca", "median"))

    merged = pd.merge(tox, nca, on="inchikey", how="outer",
                      suffixes=("_t", "_n"))
    merged["std_smiles"] = merged["std_smiles_t"].fillna(merged["std_smiles_n"])
    merged["pec50_ext"] = merged[["pec50_tox", "pec50_nca"]].mean(axis=1)
    merged = merged[["std_smiles", "inchikey", "pec50_ext"]].dropna()
    print(f"  union by inchikey={len(merged)}  "
          f"pec50 mean={merged['pec50_ext'].mean():.3f}  "
          f"std={merged['pec50_ext'].std():.3f}")
    return merged


def main():
    t0 = time.time()
    print("=== nb1044: PubChem Tox21+NCATS PXR weighted join ===\n")

    # ---- Unblind truth ----
    unb_idx = np.load(DATA_PROCESSED / "nb472_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy")
    assert len(unb_idx) == len(y_unb) == 253

    # ---- CRC ----
    tr = load_train()
    te = load_test()
    y_crc = tr["pec50"].values.astype(np.float64)
    n_crc = len(y_crc)
    scaf_crc = tr["smiles"].map(bemis_murcko).tolist()

    # ---- PubChem external ----
    print("Loading PubChem PXR (Tox21 + NCATS)...")
    ext = load_pubchem_pxr()

    # Drop external rows whose std_smiles is in CRC (avoid label leakage)
    crc_std = set(s for s in tr["smiles"].apply(standardize_smiles).tolist()
                  if s is not None)
    mask_novel = ~ext["std_smiles"].isin(crc_std)
    ext_novel = ext[mask_novel].reset_index(drop=True)
    print(f"  CRC-overlap dropped: {len(ext) - len(ext_novel)}  "
          f"novel external retained: {len(ext_novel)}")

    smi_all = list(tr["smiles"].values) + ext_novel["std_smiles"].tolist()
    y_all = np.concatenate([y_crc, ext_novel["pec50_ext"].values.astype(np.float64)])
    w_all = np.concatenate([np.ones(n_crc), np.full(len(ext_novel), EXT_W)])
    n_all = len(y_all)
    print(f"  total train rows: {n_all}  (CRC={n_crc} + ext={n_all - n_crc})")

    # ---- Featurize ALL once ----
    print("Featurizing combined (Morgan + RDKit)...")
    X_all = impute(combined(smi_all))
    X_te = impute(combined(te["smiles"].tolist()))
    X_crc = X_all[:n_crc]
    print(f"  X_all={X_all.shape}  X_te={X_te.shape}")

    # ---- Scaffold CV: split on CRC only, augment train fold with ext ----
    splits = scaffold_kfold_indices(scaf_crc, N_FOLDS, SEED)
    oof_crc = np.full(n_crc, np.nan)
    best_iters, fold_raes = [], []
    print(f"\nScaffold {N_FOLDS}-fold CV (ext rows always in train)...")
    ext_rows = np.arange(n_crc, n_all)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        full_tr = np.concatenate([tr_idx, ext_rows])
        m = lgb.train(
            PARAMS,
            lgb.Dataset(X_all[full_tr], label=y_all[full_tr],
                        weight=w_all[full_tr]),
            valid_sets=[lgb.Dataset(X_crc[va_idx], label=y_crc[va_idx])],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                       lgb.log_evaluation(-1)],
        )
        oof_crc[va_idx] = m.predict(X_crc[va_idx], num_iteration=m.best_iteration)
        fr = rae(y_crc[va_idx], oof_crc[va_idx])
        fold_raes.append(fr)
        best_iters.append(int(m.best_iteration or PARAMS["n_estimators"]))
        print(f"  fold {fold+1}  best_iter={best_iters[-1]:5d}  "
              f"RAE={fr:.4f}  elapsed={time.time()-t0:6.1f}s", flush=True)

    oof_rae = rae(y_crc, oof_crc)
    mean_best = int(np.mean(best_iters))
    print(f"\nOOF RAE (CRC) = {oof_rae:.4f}")

    # ---- Final fit on full data ----
    print(f"\nFinal fit on full {n_all} rows, n_estimators={mean_best}...")
    final = lgb.train(
        dict(PARAMS, n_estimators=mean_best),
        lgb.Dataset(X_all, label=y_all, weight=w_all),
        callbacks=[lgb.log_evaluation(-1)],
    )
    te_pred = np.clip(final.predict(X_te), y_crc.min() - 0.5, y_crc.max() + 0.5)
    in_r = in_rae(y_unb, te_pred[unb_idx])
    print(f"TEST  med={np.median(te_pred):.2f}  std={te_pred.std():.3f}")
    print(f"in_RAE(253) = {in_r:.4f}")

    # ---- Pearson check vs nb972 ----
    te_972 = np.load(DATA_PROCESSED / "te_nb972_long_train.npy")
    pearson = float(np.corrcoef(te_pred, te_972)[0, 1])
    print(f"\nPearson(te_nb1044, te_nb972) = {pearson:.4f}")

    if pearson < BLEND_PEARSON_THRESH:
        te_blend = 0.5 * te_pred + 0.5 * te_972
        in_blend = in_rae(y_unb, te_blend[unb_idx])
        action = "BAG_50_50"
        print(f"  Pearson<{BLEND_PEARSON_THRESH} -> bag 50/50 with nb972  "
              f"in_RAE={in_blend:.4f}")
    else:
        te_blend = te_pred.copy()
        in_blend = in_r
        action = "STANDALONE"
        print(f"  Pearson>={BLEND_PEARSON_THRESH} -> keep standalone")

    # ---- Persist ----
    np.save(DATA_PROCESSED / "te_nb1044_pubchem_join.npy", te_blend)
    pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te["name"].values,
        "pEC50": te_blend,
    }).to_csv(SUBMISSIONS / "nb1044_pubchem_pxr_join.csv", index=False)

    nb972_in_rae = 0.6897558359185051
    nb1014_pooled = 0.5929900142961457
    summary = {
        "tag": "nb1044",
        "ext_n_total": int(len(ext)),
        "ext_n_novel": int(len(ext_novel)),
        "ext_weight": EXT_W,
        "params": {k: v for k, v in PARAMS.items() if k != "verbose"},
        "fold_best_iters": best_iters,
        "fold_raes": [float(x) for x in fold_raes],
        "oof_rae_crc": float(oof_rae),
        "in_rae_253_standalone": float(in_r),
        "pearson_with_nb972": pearson,
        "blend_action": action,
        "in_rae_253_final": float(in_blend),
        "test_std": float(te_blend.std()),
        "ref_nb972_in_rae": nb972_in_rae,
        "ref_nb1014_pooled_rae": nb1014_pooled,
        "beats_nb1014": bool(in_blend < nb1014_pooled),
        "wall_time_sec": float(time.time() - t0),
        "plain_submission": str(SUBMISSIONS / "nb1044_pubchem_pxr_join.csv"),
    }
    with open(DATA_PROCESSED / "nb1044_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {DATA_PROCESSED/'nb1044_summary.json'}  "
          f"wall={summary['wall_time_sec']:.1f}s")
    return summary


if __name__ == "__main__":
    main()
