"""nb901 -- NR-family multi-task LGBM with PXR head.

Joins PXR CRC training (4139) with NR-family bioactivity from ChEMBL/BindingDB
(PXR, CAR, LXRa, LXRb, FXR, AR, ERa, PPARg). Target identity is one-hot
embedded (8 cols) and concatenated to combined Morgan+RDKit features (2265).
A single LGBM regressor with Huber loss is trained jointly; PXR predictions
are obtained at inference by setting the PXR one-hot bit. In-sample RAE on the
253-compound Phase-2 unblind set is reported.

Artifacts (large) go to C:/pxr_artifacts/nb901/.
Submission: submissions/nb901_nr_multitask.csv.
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
from pxr.eval import rae
from pxr.chem import standardize
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, DATA_EXTERNAL, SUBMISSIONS

ART = "C:/pxr_artifacts/nb901"
os.makedirs(ART, exist_ok=True)

# 8 NR targets + PXR_CRC as a 9th (challenge-specific) task head
TARGETS = ["PXR_CRC", "PXR", "CAR", "LXRa", "LXRb", "FXR", "AR", "ERa", "PPARg"]
T2I = {t: i for i, t in enumerate(TARGETS)}

# ---------------------------------------------------------------------------
# 1. Load NR data
# ---------------------------------------------------------------------------
def _norm_smiles(s):
    try:
        m = standardize(s)
        if m is None:
            return None
        return Chem.MolToSmiles(m)
    except Exception:
        return None


def load_nr_frames() -> pd.DataFrame:
    """Return long-format df: [std_smiles, pec50, target]."""
    frames = []

    # ChEMBL NR-extended (cached, ~11.5k)
    ch = pd.read_parquet(DATA_EXTERNAL / "chembl_nr_extended.parquet")
    ch_map = {"PXR": "PXR", "PPARg": "PPARg", "FXR": "FXR",
              "LXRa": "LXRa", "VDR": None, "RXRa": None, "PPARa": None}
    ch["target"] = ch["target_name"].map(ch_map)
    ch = ch.dropna(subset=["target", "pec50", "std_smiles"])
    frames.append(ch[["std_smiles", "pec50", "target"]])

    # BindingDB NR (cached, ~5.7k)
    bdb = pd.read_parquet(DATA_EXTERNAL / "bindingdb_nr_data.parquet")
    bdb_map = {"PXR": "PXR", "PPARg": "PPARg", "PPARgamma": "PPARg",
               "FXR": "FXR", "LXRa": "LXRa", "LXRalpha": "LXRa",
               "VDR": None, "RXRa": None, "RXRalpha": None}
    bdb["target"] = bdb["target_name"].map(bdb_map)
    bdb = bdb.dropna(subset=["target", "pec50", "std_smiles"])
    frames.append(bdb[["std_smiles", "pec50", "target"]])

    # ChEMBL bulk activities -> CAR, AR, ERa, LXRb (missing from cached NR file)
    bulk = pd.read_parquet(
        DATA_EXTERNAL / "chembl_bulk_activities.parquet",
        columns=["canonical_smiles", "accession", "pchembl_value", "standard_type"],
    )
    acc2t = {
        "O75469": "PXR", "Q14994": "CAR",
        "P10275": "AR",  "P03372": "ERa",
        "P37231": "PPARg",
        "Q96RI1": "FXR",
        "Q13133": "LXRa", "P55055": "LXRb",
    }
    bulk = bulk[bulk["accession"].isin(acc2t)]
    bulk = bulk[bulk["pchembl_value"].notna() & bulk["canonical_smiles"].notna()]
    bulk["target"] = bulk["accession"].map(acc2t)
    bulk = bulk.rename(columns={"canonical_smiles": "smiles",
                                "pchembl_value": "pec50"})
    # standardize
    bulk["std_smiles"] = bulk["smiles"].map(_norm_smiles)
    bulk = bulk.dropna(subset=["std_smiles"])
    frames.append(bulk[["std_smiles", "pec50", "target"]])

    nr = pd.concat(frames, ignore_index=True)
    nr["pec50"] = pd.to_numeric(nr["pec50"], errors="coerce")
    nr = nr.dropna(subset=["pec50"])
    nr = nr[(nr["pec50"] > 2) & (nr["pec50"] < 12)]
    # median-aggregate per (target, std_smiles) to dedupe
    nr = (nr.groupby(["target", "std_smiles"], as_index=False)["pec50"]
            .median())
    return nr


# ---------------------------------------------------------------------------
# 2. Build joined training matrix
# ---------------------------------------------------------------------------
def build_xy(nr: pd.DataFrame, tr: pd.DataFrame, te: pd.DataFrame):
    tr_std = tr["smiles"].map(_norm_smiles).fillna(tr["smiles"])
    te_std = te["smiles"].map(_norm_smiles).fillna(te["smiles"])

    rows = []
    rows.append(pd.DataFrame({"std_smiles": tr_std, "pec50": tr["pec50"],
                              "target": "PXR_CRC"}))
    rows.append(nr.assign(target=nr["target"]))
    joined = pd.concat(rows, ignore_index=True).dropna(subset=["std_smiles", "pec50"])

    # Featurize unique SMILES once
    uniq = joined["std_smiles"].drop_duplicates().tolist()
    print(f"[feat] computing combined features for {len(uniq)} unique SMILES")
    t0 = time.time()
    Xu = impute(combined(uniq))
    print(f"[feat] done in {time.time()-t0:.1f}s, X={Xu.shape}")
    s2i = {s: i for i, s in enumerate(uniq)}

    X = Xu[joined["std_smiles"].map(s2i).values]
    # 9-dim one-hot for target
    H = np.zeros((len(joined), len(TARGETS)), dtype=np.float32)
    H[np.arange(len(joined)), joined["target"].map(T2I).values] = 1.0
    X_full = np.hstack([X, H]).astype(np.float32)
    y = joined["pec50"].astype(np.float32).values

    # Test features (no target-onehot yet; we add the PXR_CRC bit at predict)
    Xte = impute(combined(te_std.tolist()))
    Hte = np.zeros((len(te_std), len(TARGETS)), dtype=np.float32)
    Hte[:, T2I["PXR_CRC"]] = 1.0
    Xte_full = np.hstack([Xte, Hte]).astype(np.float32)

    return X_full, y, joined["target"].values, Xte_full, Xu, s2i, tr_std.values


# ---------------------------------------------------------------------------
# 3. Train LGBM
# ---------------------------------------------------------------------------
def main():
    print("=== nb901: NR-family multi-task LGBM ===")
    tr = load_train()
    te = load_test()

    nr = load_nr_frames()
    print("[nr] per-target counts:")
    print(nr["target"].value_counts())
    nr.to_parquet(f"{ART}/nr_joined.parquet", index=False)

    X, y, tgt, Xte, Xu, s2i, tr_std = build_xy(nr, tr, te)
    print(f"[data] X={X.shape}  y={y.shape}  Xte={Xte.shape}")
    print(f"[data] task counts:")
    for t in TARGETS:
        n = (tgt == t).sum()
        print(f"        {t:8s}: {n:6d}")

    # weight PXR-related rows higher; CRC rows highest
    w = np.ones(len(y), dtype=np.float32)
    w[tgt == "PXR"] = 2.0
    w[tgt == "PXR_CRC"] = 4.0

    model = lgb.LGBMRegressor(
        objective="huber", alpha=0.9,
        n_estimators=2000, num_leaves=127, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.7, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=0.1,
        n_jobs=4, random_state=42, verbose=-1,
    )
    print("[lgbm] fitting...")
    t0 = time.time()
    model.fit(X, y, sample_weight=w)
    print(f"[lgbm] done in {time.time()-t0:.1f}s")

    # ----- In-sample RAE on 253 unblind -----
    idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unblind = np.load(DATA_PROCESSED / "_audit_unblind_y.npy")
    pred_te = model.predict(Xte)
    pred_unblind = pred_te[idx]
    in_rae = float(rae(y_unblind, pred_unblind))
    print(f"[eval] in-sample RAE on 253 unblind: {in_rae:.4f}")

    # ----- Submission -----
    sub = pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te["name"].values,
        "pEC50": pred_te,
    })
    out = SUBMISSIONS / "nb901_nr_multitask.csv"
    sub.to_csv(out, index=False)
    print(f"[sub] wrote {out}  ({len(sub)} rows)")

    # Save model + summary
    import joblib
    joblib.dump(model, f"{ART}/lgbm_nr_mt.joblib")
    with open(f"{ART}/summary.txt", "w") as f:
        f.write(f"n_total_rows={len(y)}\n")
        f.write(f"in_rae_unblind={in_rae:.4f}\n")
        for t in TARGETS:
            f.write(f"  {t}: {(tgt==t).sum()}\n")

    return in_rae, len(nr)


if __name__ == "__main__":
    in_rae, n_nr = main()
    print(f"\nDONE  in_RAE={in_rae:.4f}  n_nr_compounds={n_nr}")
