"""nb1360 -- shared prep for the post-hoc 'creative methods on the real 260' study.

Builds & caches everything the method scripts (nb1361 3-head, nb1362 curriculum,
nb1363 bio-readacross) need, all aligned to the now-unblinded 260:

  X_tr (4139, D_combined)   combined = morgan2048 + rdkit_desc, imputed
  y_tr (4139,)              PXR pEC50
  se_tr (4139,)            pEC50 std-error  -> real target for the NOISE head
  cliff_tr (4139,)         1 if compound has a train neighbor Tan>=0.7 & |dpEC50|>=1.0
  scaf_tr (4139,)          Bemis-Murcko scaffold (for honest scaffold-CV folds)
  X_260 (260, D)           combined features for the 260 blind compounds
  y_260 (260,)             ground truth (Phase-2 unblinded)
  base_260 (260,)          combined_corrected -- the robust 0.6318 base to beat

Everything cached under C:/pxr_work/posthoc_creative/.
"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
from src.pxr.data import load_train, load_test
from src.pxr.featurize import combined, impute
from src.pxr.chem import bemis_murcko, standardize

OUT = "C:/pxr_work/posthoc_creative"; os.makedirs(OUT, exist_ok=True)
MS  = "C:/pxr_work/meta_stacking"

def rae(yt, yp): return float(np.abs(yt-yp).sum()/np.abs(yt-np.median(yt)).sum())
def mae(yt, yp): return float(np.abs(yt-yp).mean())

def main():
    tr = load_train().reset_index(drop=True)
    # column names from loader are short snake_case
    smi_tr = tr["smiles"].tolist()
    y_tr = tr["pec50"].to_numpy(np.float32)
    # std-error column (raw csv has the long name; loader may rename) -> pull from raw
    raw = pd.read_csv("data/raw/pxr-challenge_TRAIN.csv")
    se_map = dict(zip(raw["SMILES"], raw["pEC50_std.error (-log10(molarity))"]))
    se_tr = np.array([se_map.get(s, np.nan) for s in raw["SMILES"]], np.float32)
    # align se to the loader's row order via SMILES (loader may dedup/standardize)
    # simplest: recompute se_tr on the loader frame by matching original SMILES if present
    if "smiles" in tr and len(tr) == len(raw):
        se_tr = raw["pEC50_std.error (-log10(molarity))"].to_numpy(np.float32)
    med_se = np.nanmedian(se_tr); se_tr = np.where(np.isfinite(se_tr), se_tr, med_se)

    print(f"[prep] train n={len(tr)}  y range [{y_tr.min():.2f},{y_tr.max():.2f}]  median se={med_se:.3f}")

    # ---- cliff labels: neighbor within Tanimoto>=0.7 and |dpEC50|>=1.0 ----
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048)
           if Chem.MolFromSmiles(s) else None for s in smi_tr]
    cliff = np.zeros(len(tr), np.float32)
    for i in range(len(tr)):
        if fps[i] is None: continue
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fps[i], [f for f in fps]))
        sims[i] = 0.0
        cand = np.where(sims >= 0.7)[0]
        if len(cand) and np.max(np.abs(y_tr[cand]-y_tr[i])) >= 1.0:
            cliff[i] = 1.0
    print(f"[prep] cliff-positive train compounds: {int(cliff.sum())} ({cliff.mean()*100:.1f}%)")

    scaf_tr = np.array([bemis_murcko(standardize(s)) or "" for s in smi_tr], dtype=object)

    # ---- features ----
    print("[prep] featurizing train (combined)...")
    X_tr = impute(combined(smi_tr)).astype(np.float32)

    te = load_test().reset_index(drop=True)
    sm_col = [c for c in te.columns if "smile" in c.lower()][0]
    bl = np.load(f"{MS}/_blinded_idx.npy")
    smi_260 = te.loc[bl, sm_col].tolist()
    names_260 = te.loc[bl, "name"].values
    print("[prep] featurizing 260 (combined)...")
    X_260 = impute(combined(smi_260)).astype(np.float32)

    t2 = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_2_UNBLINDED.csv")
    truth = dict(zip(t2["Molecule Name"], t2["pEC50"]))
    y_260 = np.array([truth[n] for n in names_260], np.float32)
    base_260 = np.load(f"{MS}/combined_corrected_513.npy")[bl].astype(np.float32)

    print(f"[prep] base_260 (combined_corrected) RAE={rae(y_260,base_260):.4f} MAE={mae(y_260,base_260):.4f}")

    np.savez_compressed(f"{OUT}/prep.npz",
        X_tr=X_tr, y_tr=y_tr, se_tr=se_tr, cliff_tr=cliff, scaf_tr=scaf_tr,
        X_260=X_260, y_260=y_260, base_260=base_260,
        smi_260=np.array(smi_260, dtype=object), names_260=names_260)
    # smiles for train (bio-readacross needs them)
    np.save(f"{OUT}/smi_tr.npy", np.array(smi_tr, dtype=object))
    print(f"[prep] cached -> {OUT}/prep.npz  (X_tr {X_tr.shape}, X_260 {X_260.shape})")

if __name__ == "__main__":
    main()
