"""nb1097 — does Tox21 augmentation ADD to the DEPLOYED nb3200? (the honest deploy gate)

nb1096: Tox21 actives improve PLAIN combined-LGBM by -0.037 (0.69->0.65). But nb3200 is 0.4416 (far stronger,
chempropembed base). Per cycle-290/291, the chempropembed sink absorbs signals that help weak fingerprint models.
Test directly, no new embeddings needed: does the Tox21-augmented model add ORTHOGONAL signal to nb3200?

  train_only_pred, tox21_aug_pred  : combined-LGBM on 253 (full 4139 vs 4139+Tox21-actives)
  increment = tox21_aug_pred - train_only_pred   (the change Tox21 induces)
  GATE 1: corr(increment, nb3200_error)  -> does Tox21 push predictions toward nb3200's residual?
  GATE 2: best blend nb3200 + w*tox21_aug_pred and nb3200 + w*increment -> RAE delta
If both ~0, Tox21 SAR is already in nb3200 (absorbed) -> external-data lever closed on deploy.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test, load_train
from src.pxr.featurize import combined, impute
from src.pxr.eval import rae
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb

P = "data/processed"; OUT = "data/external/tox21"


def ik(s):
    m = Chem.MolFromSmiles(str(s)); return Chem.MolToInchiKey(m) if m else None


def lgbm(Xtr, ytr, Xte, seed=0):
    m = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, n_jobs=4, verbose=-1, random_state=seed)
    m.fit(Xtr, ytr); return m.predict(Xte)


def build_tox():
    prim = pd.read_parquet(f"{OUT}/aid_720659_concise.parquet")
    prim["CID"] = pd.to_numeric(prim["CID"], errors="coerce"); prim = prim.dropna(subset=["CID"])
    prim["CID"] = prim["CID"].astype(int); prim["av"] = pd.to_numeric(prim["Activity Value [uM]"], errors="coerce")
    act = prim[(prim["Activity Outcome"] == "Active") & prim["av"].notna()].copy()
    act["pec50"] = 6.0 - np.log10(act["av"].clip(lower=1e-3))
    smi_map = {int(k): v for k, v in json.load(open(f"{OUT}/pxr_active_smiles.json")).items()}
    act["smiles"] = act["CID"].map(lambda c: smi_map.get(int(c)))
    act = act.dropna(subset=["smiles"]); act["ik"] = act["smiles"].map(ik)
    return act.dropna(subset=["ik"]).drop_duplicates("ik")[["smiles", "pec50"]].reset_index(drop=True)


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test().reset_index(drop=True)
    unb = np.load(f"{P}/_audit_unblind_idx.npy"); y = np.load(f"{P}/_audit_unblind_y.npy")
    anchor = np.load(f"{P}/nb3200_pred_oof.npy"); err = y - anchor
    tox = build_tox(); tox["ik"] = tox["smiles"].map(ik)
    tr_ik = set(tr["smiles"].map(ik).dropna()); tox = tox[~tox["ik"].isin(tr_ik)].reset_index(drop=True)
    print(f"nb3200 anchor RAE {rae(y, anchor):.4f} | Tox21 actives {len(tox)}", flush=True)

    smte = te["smiles"].to_numpy()[unb].tolist()
    Xtr = impute(combined(tr["smiles"].tolist())); ytr = tr["pec50"].to_numpy()
    Xte = impute(combined(smte)); Xtx = impute(combined(tox["smiles"].tolist())); ytx = tox["pec50"].to_numpy()

    # average over 5 seeds to de-noise the increment
    p_base = np.mean([lgbm(Xtr, ytr, Xte, s) for s in range(5)], 0)
    p_aug = np.mean([lgbm(np.vstack([Xtr, Xtx]), np.concatenate([ytr, ytx]), Xte, s) for s in range(5)], 0)
    incr = p_aug - p_base
    print(f"plain-LGBM base RAE {rae(y, p_base):.4f} | +Tox21 RAE {rae(y, p_aug):.4f} (delta {rae(y,p_aug)-rae(y,p_base):+.4f})")

    # GATE 1: does the Tox21 increment align with nb3200 error?
    c_incr = np.corrcoef(incr, err)[0, 1]
    c_aug = np.corrcoef(p_aug, err)[0, 1]
    print(f"\nGATE 1 corr(Tox21 increment, nb3200 error) = {c_incr:+.3f}  (|increment| mean {np.abs(incr).mean():.3f})")
    print(f"        corr(tox21-aug pred, nb3200 error)   = {c_aug:+.3f}")

    # GATE 2: blends
    def best_blend(p):
        b = rae(y, anchor)
        for w in np.linspace(0, 1, 41):
            b = min(b, rae(y, (1 - w) * anchor + w * p))
        return b
    bl_aug = best_blend(p_aug)
    # increment as additive correction
    b_incr = rae(y, anchor); bw = 0
    for w in np.linspace(0, 2, 41):
        r = rae(y, anchor + w * incr)
        if r < b_incr: b_incr, bw = r, w
    print(f"\nGATE 2 best blend nb3200+tox21aug : {bl_aug:.4f} (delta {bl_aug-rae(y,anchor):+.4f})")
    print(f"        best nb3200 + w*increment : {b_incr:.4f} at w={bw:.2f} (delta {b_incr-rae(y,anchor):+.4f})")
    json.dump(dict(anchor=float(rae(y, anchor)), base=float(rae(y, p_base)), aug=float(rae(y, p_aug)),
                   corr_incr_err=float(c_incr), corr_aug_err=float(c_aug),
                   blend_aug=float(bl_aug), incr_correction=float(b_incr)),
              open(f"{P}/nb1097_tox21_deploy.json", "w"), indent=2)


if __name__ == "__main__":
    main()
