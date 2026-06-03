"""nb980 -- Aggressive ChEMBL NR + xenobiotic-sensing multi-task LGBM.

Cycle-3 attempt to beat nb901 (in_RAE 0.6765) by enlarging the auxiliary task
panel and re-weighting toward PXR. Adds CAR, LXRa/b, FXR, RXRa/b/g, AR,
ERRalpha/beta/gamma drawn from chembl_bulk_activities + chembl_nr_extended +
bindingdb_nr_data. Single LGBM-Huber jointly regresses pEC50 with target
one-hot. Predictions on the 513 test set use the PXR_CRC bit.

Artifacts to C:/pxr_artifacts/nb980/.
Submission: submissions/nb980_aggressive_chembl.csv.
"""
from __future__ import annotations
import os, sys, time, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb

from rdkit import Chem
from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import standardize, to_inchikey, bemis_murcko
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, DATA_EXTERNAL, SUBMISSIONS

ART = "C:/pxr_artifacts/nb980"
os.makedirs(ART, exist_ok=True)

# Target panel per spec
TARGETS = [
    "PXR_CRC", "PXR", "CAR",
    "LXRa", "LXRb", "FXR",
    "RXRa", "RXRb", "RXRg",
    "AR",
    "ERRa", "ERRb", "ERRg",
]
T2I = {t: i for i, t in enumerate(TARGETS)}

# UniProt -> task name
ACC2T = {
    "O75469": "PXR",  "Q14994": "CAR",
    "Q13133": "LXRa", "P55055": "LXRb",
    "Q96RI1": "FXR",
    "P19793": "RXRa", "P28702": "RXRb", "P48443": "RXRg",
    "P10275": "AR",
    "P11474": "ERRa", "O95718": "ERRb", "P62508": "ERRg",
}

# target_name strings in cached parquets
NAME2T = {
    "PXR": "PXR", "CAR": "CAR",
    "LXRa": "LXRa", "LXRalpha": "LXRa",
    "LXRb": "LXRb", "LXRbeta": "LXRb",
    "FXR": "FXR",
    "RXRa": "RXRa", "RXRalpha": "RXRa",
    "RXRb": "RXRb", "RXRbeta": "RXRb",
    "RXRg": "RXRg", "RXRgamma": "RXRg",
    "AR": "AR",
    "ERRa": "ERRa", "ERRalpha": "ERRa",
    "ERRb": "ERRb", "ERRbeta": "ERRb",
    "ERRg": "ERRg", "ERRgamma": "ERRg",
}


def _std(s):
    try:
        m = standardize(s)
        return Chem.MolToSmiles(m) if m is not None else None
    except Exception:
        return None


def _inchi(s):
    try:
        return to_inchikey(s)
    except Exception:
        return None


def load_aux() -> pd.DataFrame:
    frames = []

    # chembl_nr_extended (already standardized)
    ch = pd.read_parquet(DATA_EXTERNAL / "chembl_nr_extended.parquet")
    ch["target"] = ch["target_name"].map(NAME2T)
    ch = ch.dropna(subset=["target", "pec50", "std_smiles"])
    frames.append(ch[["std_smiles", "pec50", "target"]])
    print(f"[aux] chembl_nr_extended: {len(ch)}")

    # bindingdb_nr_data
    bdb = pd.read_parquet(DATA_EXTERNAL / "bindingdb_nr_data.parquet")
    bdb["target"] = bdb["target_name"].map(NAME2T)
    bdb = bdb.dropna(subset=["target", "pec50", "std_smiles"])
    frames.append(bdb[["std_smiles", "pec50", "target"]])
    print(f"[aux] bindingdb_nr_data: {len(bdb)}")

    # chembl_bulk_activities -> map by accession
    bulk = pd.read_parquet(
        DATA_EXTERNAL / "chembl_bulk_activities.parquet",
        columns=["canonical_smiles", "accession", "pchembl_value"],
    )
    bulk = bulk[bulk["accession"].isin(ACC2T)]
    bulk = bulk[bulk["pchembl_value"].notna() & bulk["canonical_smiles"].notna()]
    bulk["target"] = bulk["accession"].map(ACC2T)
    bulk = bulk.rename(columns={"canonical_smiles": "smi",
                                "pchembl_value": "pec50"})
    # standardize only unique SMILES to save time
    uniq = bulk["smi"].drop_duplicates().tolist()
    print(f"[aux] standardizing {len(uniq)} unique bulk SMILES ...")
    t0 = time.time()
    s2std = {s: _std(s) for s in uniq}
    print(f"[aux] std done in {time.time()-t0:.1f}s")
    bulk["std_smiles"] = bulk["smi"].map(s2std)
    bulk = bulk.dropna(subset=["std_smiles"])
    frames.append(bulk[["std_smiles", "pec50", "target"]])
    print(f"[aux] chembl_bulk (NR-filtered): {len(bulk)}")

    nr = pd.concat(frames, ignore_index=True)
    nr["pec50"] = pd.to_numeric(nr["pec50"], errors="coerce")
    nr = nr.dropna(subset=["pec50"])
    nr = nr[(nr["pec50"] > 2) & (nr["pec50"] < 12)]

    # dedupe within (target, inchikey)
    print("[aux] computing InChIKeys for dedup ...")
    t0 = time.time()
    uniq2 = nr["std_smiles"].drop_duplicates().tolist()
    s2ik = {s: _inchi(s) for s in uniq2}
    nr["inchikey"] = nr["std_smiles"].map(s2ik)
    nr = nr.dropna(subset=["inchikey"])
    print(f"[aux] inchikey done in {time.time()-t0:.1f}s")
    nr = (nr.groupby(["target", "inchikey"], as_index=False)
            .agg(std_smiles=("std_smiles", "first"),
                 pec50=("pec50", "median")))
    return nr[["std_smiles", "pec50", "target"]]


def build_xy(nr, tr, te):
    tr_std = tr["smiles"].map(_std).fillna(tr["smiles"])
    te_std = te["smiles"].map(_std).fillna(te["smiles"])

    crc = pd.DataFrame({"std_smiles": tr_std, "pec50": tr["pec50"],
                        "target": "PXR_CRC"})
    joined = pd.concat([crc, nr], ignore_index=True)
    joined = joined.dropna(subset=["std_smiles", "pec50"])
    joined = joined[joined["target"].isin(TARGETS)].reset_index(drop=True)

    uniq = joined["std_smiles"].drop_duplicates().tolist()
    print(f"[feat] combined features for {len(uniq)} unique SMILES")
    t0 = time.time()
    Xu = impute(combined(uniq))
    print(f"[feat] done in {time.time()-t0:.1f}s, X={Xu.shape}")
    s2i = {s: i for i, s in enumerate(uniq)}

    X = Xu[joined["std_smiles"].map(s2i).values]
    H = np.zeros((len(joined), len(TARGETS)), dtype=np.float32)
    H[np.arange(len(joined)), joined["target"].map(T2I).values] = 1.0
    X_full = np.hstack([X, H]).astype(np.float32)
    y = joined["pec50"].astype(np.float32).values
    tgt = joined["target"].values

    Xte = impute(combined(te_std.tolist()))
    Hte = np.zeros((len(te_std), len(TARGETS)), dtype=np.float32)
    Hte[:, T2I["PXR_CRC"]] = 1.0
    Xte_full = np.hstack([Xte, Hte]).astype(np.float32)

    # scaffolds (for CRC rows only — CV is over PXR_CRC subset)
    crc_idx = np.where(tgt == "PXR_CRC")[0]
    print(f"[feat] computing scaffolds for {len(crc_idx)} CRC rows")
    scaffolds = [bemis_murcko(s) or "" for s in joined.loc[crc_idx, "std_smiles"]]

    return X_full, y, tgt, Xte_full, crc_idx, scaffolds


def make_weights(tgt):
    w = np.ones(len(tgt), dtype=np.float32)
    w[tgt == "PXR_CRC"] = 5.0
    w[tgt == "PXR"] = 2.5
    return w


def fit_lgbm(Xtr, ytr, wtr):
    model = lgb.LGBMRegressor(
        objective="huber", alpha=2.0,
        n_estimators=3000, num_leaves=128, learning_rate=0.025,
        subsample=0.85, colsample_bytree=0.7, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=0.1,
        n_jobs=4, random_state=42, verbose=-1,
    )
    model.fit(Xtr, ytr, sample_weight=wtr)
    return model


def main():
    t_start = time.time()
    print("=== nb980: aggressive ChEMBL NR multi-task LGBM ===")
    tr = load_train()
    te = load_test()

    nr = load_aux()
    print("\n[nr] rows per target:")
    print(nr["target"].value_counts())
    nr.to_parquet(f"{ART}/nr_joined.parquet", index=False)

    X, y, tgt, Xte, crc_idx, scaffolds = build_xy(nr, tr, te)
    print(f"\n[data] X={X.shape} Xte={Xte.shape}")
    for t in TARGETS:
        print(f"   {t:8s}: {(tgt == t).sum()}")

    w = make_weights(tgt)

    # ---- scaffold CV on CRC rows only (auxiliary rows are always in train) ----
    print("\n[cv] scaffold 5-fold over PXR_CRC rows (aux rows always in train)")
    folds = scaffold_kfold_indices(scaffolds, n_splits=5, shuffle=True, seed=42)
    oof = np.full(len(crc_idx), np.nan, dtype=np.float32)
    aux_mask = (tgt != "PXR_CRC")

    for k, (tr_loc, va_loc) in enumerate(folds):
        t0 = time.time()
        tr_global = crc_idx[tr_loc]
        va_global = crc_idx[va_loc]
        tr_rows = np.concatenate([tr_global, np.where(aux_mask)[0]])
        model = fit_lgbm(X[tr_rows], y[tr_rows], w[tr_rows])
        oof[va_loc] = model.predict(X[va_global])
        print(f"   fold {k}: train={len(tr_rows)} val={len(va_loc)} "
              f"rae_fold={rae(y[va_global], oof[va_loc]):.4f} "
              f"({time.time()-t0:.1f}s)")

    cv_rae = float(rae(y[crc_idx], oof))
    print(f"[cv] scaffold 5f CV RAE on 4139 CRC = {cv_rae:.4f}")

    # ---- final fit on ALL data for 513 prediction ----
    print("\n[deploy] fitting final model on all rows")
    t0 = time.time()
    model_full = fit_lgbm(X, y, w)
    print(f"[deploy] done in {time.time()-t0:.1f}s")
    pred_te = model_full.predict(Xte).astype(np.float32)

    # ---- in-sample RAE on 253 ----
    idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy")
    in_rae = float(rae(y_unb, pred_te[idx]))
    print(f"[eval] in-sample RAE on 253 unblind = {in_rae:.4f}")

    # ---- submission ----
    sub = pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te["name"].values,
        "pEC50": pred_te,
    })
    out = SUBMISSIONS / "nb980_aggressive_chembl.csv"
    sub.to_csv(out, index=False)
    print(f"[sub] wrote {out} ({len(sub)} rows)")

    # save artifacts
    np.save(f"{ART}/te_pred.npy", pred_te)
    np.save(f"{ART}/oof_crc.npy", oof)
    with open(f"{ART}/summary.txt", "w") as f:
        f.write(f"n_total_rows={len(y)}\n")
        f.write(f"n_aux={len(nr)}\n")
        f.write(f"cv_rae_crc_5f={cv_rae:.4f}\n")
        f.write(f"in_rae_unblind_253={in_rae:.4f}\n")
        for t in TARGETS:
            f.write(f"  {t}: {(tgt == t).sum()}\n")

    print(f"\nDONE  in_RAE={in_rae:.4f}  cv_rae={cv_rae:.4f}  "
          f"n_aux={len(nr)}  wall={time.time()-t_start:.1f}s")
    return in_rae, cv_rae, len(nr)


if __name__ == "__main__":
    in_rae, cv_rae, n_aux = main()
