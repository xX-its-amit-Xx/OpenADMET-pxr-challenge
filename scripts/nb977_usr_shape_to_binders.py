"""nb977 — 3D-INTERACTION probe (the one untested distinct axis per nb964): does a compound's
3D-shape complementarity to KNOWN PXR bound poses predict pEC50?

Unlike nb954 (generic 3D descriptors = seed-noise), this is TARGETED: USR shape-similarity of each
compound to the canonical crystallographic bound poses (does its shape match a known binder?). Pure
geometry (RDKit USR, coords only -> robust, Windows-friendly). All outputs to C: (D: pressure).

Test: combined vs combined+USR-features on the nb952 degradation curve. If it adds at the novel end,
escalate to real docking (Codespace/Linux). If absorbed, the 3D-shape axis is closed too.
"""
import os, sys, json, glob, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train, load_test
from src.pxr.eval import rae, scaffold_kfold_indices
from src.pxr.featurize import combined, impute
import gemmi
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb

D = "data/processed"
OUT = "C:/pxr_struct"
CKPT = "C:/pxr_struct/nb977_usr_ckpt.npz"
STRUCT_DIR = "data/external/pdb64_structures"
BINS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.01]
# canonical bound-pose references (pdb, ligcode, mode)
REFS = [("1m13", "HYF", "tripod"), ("7axi", "EST", "tripod"), ("8f5y", "JQ1", "blade"),
        ("2o9i", "444", "blade"), ("8svp", "WSX", "skewer"), ("5x0r", "4WH", "skewer"),
        ("1skx", "RFP", "blob"), ("6nx1", "L7D", "blob"), ("6bns", "XGH", "reach")]
EXCL = set("HOH NA CL ZN MG K CA SO4 PO4 GOL EDO PEG PG4 MPD DMS ACT FMT EOH IPA NAG".split())


def is_aa(r):
    info = gemmi.find_tabulated_residue(r.name); return info is not None and info.is_amino_acid()


def usr_from_coords(coords):
    """USR shape descriptor from raw heavy-atom coords (no bonds/elements needed)."""
    rw = Chem.RWMol()
    for _ in range(len(coords)):
        rw.AddAtom(Chem.Atom(6))
    m = rw.GetMol()
    conf = Chem.Conformer(len(coords))
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))
    m.AddConformer(conf, assignId=True)
    return np.array(rdMolDescriptors.GetUSR(m))


def usr_from_smiles(smi):
    try:
        m = Chem.AddHs(Chem.MolFromSmiles(smi)); p = AllChem.ETKDGv3(); p.randomSeed = 42
        if AllChem.EmbedMolecule(m, p) != 0:
            p.useRandomCoords = True
            if AllChem.EmbedMolecule(m, p) != 0: return None
        AllChem.MMFFOptimizeMolecule(m, maxIters=100)
        m = Chem.RemoveHs(m)
        xyz = m.GetConformer().GetPositions()
        return usr_from_coords(xyz)
    except Exception:
        return None


def usr_score(u1, u2):
    return 1.0 / (1.0 + np.mean(np.abs(np.asarray(u1) - np.asarray(u2))))


def get_ref_usr():
    refs = []
    for pdb, lig, mode in REFS:
        st = gemmi.read_structure(f"{STRUCT_DIR}/{pdb}.cif"); st.setup_entities()
        for ch in st[0]:
            r = next((r for r in ch if not is_aa(r) and r.name == lig), None)
            if r:
                co = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in r if a.element.name != "H"])
                refs.append((f"{pdb}_{lig}_{mode}", usr_from_coords(co))); break
    return refs


def compute_usr_features(smiles, refs, tag):
    n = len(smiles); nf = len(refs)
    feats = np.full((n, nf), np.nan)
    todo = list(range(n))
    print(f"  {tag}: USR for {n} compounds vs {nf} refs ...", flush=True)
    t0 = time.time()
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(usr_from_smiles, smiles[i]): i for i in todo}
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]; u = fut.result()
            if u is not None:
                feats[i] = [usr_score(u, ru) for _, ru in refs]
            done += 1
            if done % 1000 == 0: print(f"    {done}/{n} ({time.time()-t0:.0f}s)", flush=True)
    return feats


def curve(y, p, sv):
    out = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (sv >= lo) & (sv < hi); nn = int(m.sum())
        out.append((f"[{lo:.1f},{hi:.1f})", nn, round(float(np.mean(np.abs(y[m]-p[m]))), 4) if nn else None))
    return out


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test()
    y = tr["pec50"].to_numpy(float)
    refs = get_ref_usr()
    print(f"built {len(refs)} canonical bound-pose USR references")

    if os.path.exists(CKPT):
        z = np.load(CKPT); F_tr, F_te = z["F_tr"], z["F_te"]
        print("resumed USR features from ckpt")
    else:
        F_tr = compute_usr_features(tr["smiles"].tolist(), refs, "train")
        F_te = compute_usr_features(te["smiles"].tolist(), refs, "test")
        np.savez(CKPT, F_tr=F_tr, F_te=F_te)
    np.save(f"{OUT}/nb977_usr_test.npy", F_te)

    from sklearn.impute import SimpleImputer
    Ftr = SimpleImputer(strategy="median").fit_transform(F_tr).astype(np.float32)
    scaf = [MurckoScaffold.MurckoScaffoldSmiles(s) if Chem.MolFromSmiles(s) else None for s in tr["smiles"]]
    folds = scaffold_kfold_indices(scaf, n_splits=5, seed=42)
    max_sim = np.load(f"{D}/nb952_max_sim_4139.npy")
    Xc = impute(combined(tr["smiles"].tolist())).astype(np.float32)
    ref_deep = next(r for r in json.load(open(f"{D}/nb952_stress_curve.json"))["lgbm_curve"]
                    if r["bin"] == "[0.0,0.3)")["mae"]

    res = {}
    for tagn, X in [("combined", Xc), ("combined+USR", np.hstack([Xc, Ftr]))]:
        oof = np.full(len(y), np.nan)
        for tri, vai in folds:
            mdl = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05,
                                    n_jobs=4, verbose=-1).fit(X[tri], y[tri])
            oof[vai] = mdl.predict(X[vai])
        rows = curve(y, oof, max_sim); deep = next(m for b, n, m in rows if b == "[0.0,0.3)")
        res[tagn] = {"overall_rae": round(float(rae(y, oof)), 4), "deep": deep}
        print(f"{tagn}: overall {res[tagn]['overall_rae']}  deep@sim<0.3 {deep}")
    d = res["combined+USR"]["deep"] - ref_deep
    print(f"\nref LGBM deep {ref_deep} ; combined+USR deep {res['combined+USR']['deep']} (delta {d:+.4f})")
    print(">>> USR shape-to-binders ADDS -> multi-seed verify + escalate to docking" if res["combined+USR"]["deep"] < res["combined"]["deep"] - 0.003
          else ">>> USR shape-to-binders absorbed/no help -> 3D-shape axis closed (consistent w/ nb954)")
    json.dump({"refs": [r[0] for r in refs], "ref_deep": ref_deep, "results": res},
              open(f"{OUT}/nb977_usr_summary.json", "w"), indent=2)
    print(f"saved -> {OUT}/nb977_usr_summary.json")


if __name__ == "__main__":
    main()
