"""nb1096 — Tox21 pseudo-label AUGMENTATION test (close the external-data lever honestly).

nb1095 showed Tox21 PXR actives give ZERO test-manifold coverage (0/513 improved; blind-30 farther than train).
But the mirror oracle showed even scaffold-disjoint SAME-TARGET data can lift a model. So test the actual modeling
effect: augment our 4139 drug-like train with the AC50-quantified Tox21 PXR compounds (pseudo-pEC50 = -log10(AC50)),
train combined-LGBM, score the 253 unblind. Compare honest RAE train-only vs +Tox21. Also test ACTIVES-only and a
similarity-gated subset (keep only Tox21 compounds with >=0.4 sim to any test compound) since off-manifold weak
actives likely add label noise. If no variant helps, the external-data lever is definitively closed.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test, load_train
from src.pxr.featurize import combined, impute
from src.pxr.chem import morgan_fp_batch
from src.pxr.eval import rae
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb

P = "data/processed"; OUT = "data/external/tox21"


def ik(s):
    m = Chem.MolFromSmiles(str(s)); return Chem.MolToInchiKey(m) if m else None


def fpf(sm): return (morgan_fp_batch(sm).astype(np.float32) > 0).astype(np.float32)


def lgbm(Xtr, ytr, Xte):
    m = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, n_jobs=4, verbose=-1)
    m.fit(Xtr, ytr); return m.predict(Xte)


def main():
    prim = pd.read_parquet(f"{OUT}/aid_720659_concise.parquet")
    prim["av"] = pd.to_numeric(prim["Activity Value [uM]"], errors="coerce")
    # build pseudo-pEC50: actives w/ AC50 -> -log10(AC50_M); inactives -> floor 4.0
    prim["CID"] = pd.to_numeric(prim["CID"], errors="coerce")
    prim = prim.dropna(subset=["CID"]); prim["CID"] = prim["CID"].astype(int)
    act = prim[(prim["Activity Outcome"] == "Active") & prim["av"].notna()].copy()
    act["pec50"] = 6.0 - np.log10(act["av"].clip(lower=1e-3))
    inact = prim[prim["Activity Outcome"] == "Inactive"].copy(); inact["pec50"] = 4.0
    tox = pd.concat([act, inact[["CID", "pec50"]].assign(av=np.nan)], ignore_index=True)
    # need smiles
    smi_map = {int(k): v for k, v in json.load(open(f"{OUT}/pxr_active_smiles.json")).items()}
    # also fetch inactive smiles? use only those we have smiles for (actives have them). For a fair test, fetch needed.
    import requests, time
    need = [int(c) for c in tox["CID"].dropna().unique() if int(c) not in smi_map]
    need = [c for c in need if c > 0]
    Bp = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    for i in range(0, len(need), 100):
        ch = need[i:i+100]
        try:
            r = requests.get(f"{Bp}/compound/cid/{','.join(map(str,ch))}/property/IsomericSMILES,SMILES/JSON", timeout=120)
            for p in r.json()["PropertyTable"]["Properties"]:
                s = p.get("IsomericSMILES") or p.get("SMILES") or p.get("CanonicalSMILES")
                if s: smi_map[p["CID"]] = s
        except Exception as e:
            print("smiles err", str(e)[:50])
        time.sleep(0.25)
    tox["smiles"] = tox["CID"].map(lambda c: smi_map.get(int(c)) if pd.notna(c) else None)
    tox = tox.dropna(subset=["smiles"]).copy(); tox["ik"] = tox["smiles"].map(ik)
    tox = tox.dropna(subset=["ik"]).drop_duplicates("ik")
    print(f"Tox21 pseudo set: {len(tox)} (actives w/ AC50 {len(act)}, pec50 range {tox['pec50'].min():.2f}-{tox['pec50'].max():.2f})", flush=True)

    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test().reset_index(drop=True)
    unb = np.load(f"{P}/_audit_unblind_idx.npy"); y = np.load(f"{P}/_audit_unblind_y.npy")
    tr_ik = set(tr["smiles"].map(ik).dropna())
    tox = tox[~tox["ik"].isin(tr_ik)].reset_index(drop=True)
    print(f"  not in train: {len(tox)}", flush=True)

    smte = te["smiles"].to_numpy()[unb].tolist()
    # featurize
    Xtr = impute(combined(tr["smiles"].tolist())); ytr = tr["pec50"].to_numpy()
    Xte = impute(combined(smte))
    Xtx = impute(combined(tox["smiles"].tolist())); ytx = tox["pec50"].to_numpy()

    # similarity gate: Tox21 compounds with >=0.4 sim to any test compound
    Fte = fpf(smte); Ftx = fpf(tox["smiles"].tolist())
    sim = (Ftx @ Fte.T); sim = sim / np.clip(Ftx.sum(1)[:, None] + Fte.sum(1)[None, :] - sim, 1, None)
    gate = sim.max(1) >= 0.4
    print(f"  Tox21 compounds with >=0.4 sim to test: {gate.sum()}", flush=True)

    base = rae(y, lgbm(Xtr, ytr, Xte))
    variants = {
        "train_only": base,
        "+tox21_all": rae(y, lgbm(np.vstack([Xtr, Xtx]), np.concatenate([ytr, ytx]), Xte)),
        "+tox21_actives_only": rae(y, lgbm(np.vstack([Xtr, Xtx[ytx > 4.0]]),
                                            np.concatenate([ytr, ytx[ytx > 4.0]]), Xte)),
        "+tox21_simgated": rae(y, lgbm(np.vstack([Xtr, Xtx[gate]]),
                                       np.concatenate([ytr, ytx[gate]]), Xte)) if gate.sum() else None,
    }
    print("\n=== Tox21 AUGMENTATION (honest RAE on 253; train-only is the gate) ===")
    for k, v in variants.items():
        if v is None: continue
        print(f"  {k:24s} RAE {v:.4f}  (delta {v-base:+.4f})")
    json.dump({k: (float(v) if v is not None else None) for k, v in variants.items()},
              open(f"{P}/nb1096_tox21_augment.json", "w"), indent=2)


if __name__ == "__main__":
    main()
