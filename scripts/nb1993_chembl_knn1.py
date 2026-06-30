"""nb1993 — ChEMBL kNN top-1 smoke-test LB submission.

Predicts pEC50 for 513 test compounds as similarity-weighted-mean of top-1 ChEMBL PXR analog.
This is a PROBE submission, NOT for PRIMARY promotion.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

REPO = Path("d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge")
RAW = REPO / "data" / "raw"
PROC = REPO / "data" / "processed"
EXT = REPO / "data" / "external"
SUB = REPO / "submissions"


def fp(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)


def main():
    # Test
    test = pd.read_csv(RAW / "pxr-challenge_TEST_BLINDED.csv")
    print(f"test: {test.shape}")

    # ChEMBL PXR (CHEMBL3401)
    cmbl = pd.read_parquet(EXT / "chembl_pxr_CHEMBL3401.parquet")
    cmbl = cmbl[["canonical_smiles", "pchembl_value", "standard_type"]].dropna()
    cmbl["pchembl_value"] = pd.to_numeric(cmbl["pchembl_value"], errors="coerce")
    cmbl = cmbl.dropna()
    # prefer EC50, fall back to all-type mean
    agg = cmbl.groupby("canonical_smiles")["pchembl_value"].mean().reset_index()
    print(f"chembl agg: {agg.shape}")

    # FPs
    test_fps = []
    test_ok = []
    for i, smi in enumerate(test["SMILES"].values):
        f = fp(smi)
        test_fps.append(f)
        test_ok.append(f is not None)
    print(f"test fp ok: {sum(test_ok)}/{len(test_ok)}")

    cmbl_fps = []
    cmbl_y = []
    cmbl_ok_smi = []
    for smi, y in zip(agg["canonical_smiles"].values, agg["pchembl_value"].values):
        f = fp(smi)
        if f is not None:
            cmbl_fps.append(f)
            cmbl_y.append(y)
            cmbl_ok_smi.append(smi)
    cmbl_y = np.asarray(cmbl_y, dtype=float)
    print(f"chembl fp ok: {len(cmbl_fps)}")

    # For each test, find top-1 Tanimoto match and use its pchembl as pred
    preds = np.full(len(test), np.nan)
    sims = np.full(len(test), np.nan)
    for i, tf in enumerate(test_fps):
        if tf is None:
            continue
        s = np.asarray(DataStructs.BulkTanimotoSimilarity(tf, cmbl_fps))
        j = int(np.argmax(s))
        sims[i] = s[j]
        preds[i] = cmbl_y[j]

    # impute missing with training pEC50 median (~4.5 typically)
    train = pd.read_csv(RAW / "pxr-challenge_TRAIN.csv")
    med = float(np.nanmedian(train["pEC50"].values))
    print(f"train pec50 median (impute fill): {med:.3f}")
    preds = np.where(np.isnan(preds), med, preds)

    print(f"pred stats — mean {preds.mean():.3f} std {preds.std():.3f} min {preds.min():.3f} max {preds.max():.3f}")
    print(f"top-1 sim stats — mean {np.nanmean(sims):.3f} median {np.nanmedian(sims):.3f} min {np.nanmin(sims):.3f} max {np.nanmax(sims):.3f}")

    # Save CSV (SMILES + Molecule Name + pEC50)
    out = pd.DataFrame({
        "SMILES": test["SMILES"].values,
        "Molecule Name": test["Molecule Name"].values,
        "pEC50": preds,
    })
    csv_path = SUB / "nb1993_chembl_knn1.csv"
    out.to_csv(csv_path, index=False)
    print(f"wrote: {csv_path}  rows={len(out)}")

    # in_RAE on 253 unblind
    unb_idx = np.load(PROC / "nb472_unblind_idx.npy")
    unb_y = np.load(PROC / "postmortem" / "pm_unblind_y.npy")
    assert len(unb_idx) == 253 and len(unb_y) == 253
    p_unb = preds[unb_idx]
    err = np.abs(p_unb - unb_y).sum()
    denom = np.abs(unb_y - unb_y.mean()).sum()
    in_rae = err / denom
    print(f"in_RAE on 253 unblind: {in_rae:.4f}")

    # Save summary
    summary = {
        "method": "nb1993_chembl_knn1",
        "csv": str(csv_path),
        "in_rae_unblind": float(in_rae),
        "pred_mean": float(preds.mean()),
        "pred_std": float(preds.std()),
        "top1_sim_median": float(np.nanmedian(sims)),
        "n_chembl_anchors": int(len(cmbl_fps)),
        "n_test": int(len(test)),
    }
    import json
    (PROC / "nb1993_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
