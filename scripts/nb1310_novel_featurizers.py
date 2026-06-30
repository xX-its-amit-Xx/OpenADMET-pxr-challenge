"""
nb1310_novel_featurizers.py — Explore completely different molecular representations.

Featurizers tested:
  1. Avalon Fingerprints (1024-bit, pharmacophore-aware)
  2. ECFP6 (radius=3, 4096-bit vs deployed ECFP4 radius=2 2048-bit)
  3. Topological Torsion (hashed, 2048-bit)
  4. MACCS Keys (166 structural keys)
  5. 3D Shape Descriptors (PMI, NPR, Asphericity, etc. from ETKDG conformers)
  6. Concatenated: ECFP6(4096) + MACCS(166) + rdkit_desc(217)

For each: scaffold 5-fold CV RAE on 4139 train, RAE on 253 unblinded test.
Saves predictions to C:/pxr_work/meta_stacking/feat_<name>_te_513.npy
Saves summary to C:/pxr_work/meta_stacking/featurizer_results.json
"""
import os, sys, json, time, warnings
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

PROJ = "D:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge"
sys.path.insert(0, PROJ)

warnings.filterwarnings("ignore")

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, MACCSkeys, AllChem
RDLogger.DisableLog("rdApp.*")

from src.pxr.data import load_train, load_test
from src.pxr.chem import bemis_murcko
from src.pxr.eval import scaffold_kfold_indices, rae
from src.pxr.featurize import rdkit_desc
from lightgbm import LGBMRegressor

# ── Paths ──────────────────────────────────────────────────────────────────
OUTDIR   = "C:/pxr_work/meta_stacking"
UNB_IDX  = "C:/pxr_work/phase1_unblind/unblind_te_idx.npy"
UNB_Y    = "C:/pxr_work/phase1_unblind/unblind_y_true.npy"
RESULTS_JSON = f"{OUTDIR}/featurizer_results.json"

os.makedirs(OUTDIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading train/test data...")
tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
te = load_test().reset_index(drop=True)
ytr = tr["pec50"].values
smiles_tr = tr["smiles"].tolist()
smiles_te = te["smiles"].tolist()
scaffolds  = tr["smiles"].apply(bemis_murcko).fillna("").tolist()

n_tr = len(tr)
n_te = len(te)
print(f"  Train: {n_tr}, Test: {n_te}")

# ── Load unblinded labels ──────────────────────────────────────────────────
unb_idx = np.load(UNB_IDX)
unb_y   = np.load(UNB_Y)
print(f"  Unblinded: {len(unb_idx)} compounds")

# ── LGBM config ────────────────────────────────────────────────────────────
LGBM_PARAMS = dict(
    n_estimators=300, num_leaves=32, learning_rate=0.05,
    n_jobs=4, random_state=42, verbose=-1
)

# ── Scaffold CV ─────────────────────────────────────────────────────────────
def get_cv_folds(n_splits=5):
    return scaffold_kfold_indices(scaffolds, n_splits=n_splits)

# ── Helpers ────────────────────────────────────────────────────────────────
def impute(X):
    imp = SimpleImputer(strategy="median")
    return imp.fit_transform(X)

def train_predict(X_tr, y_tr, X_te, cv_folds):
    """Scaffold CV RAE + full-train predict on 513."""
    oof_preds = np.zeros(n_tr)
    for tr_idx, va_idx in cv_folds:
        m = LGBMRegressor(**LGBM_PARAMS)
        m.fit(X_tr[tr_idx], y_tr[tr_idx])
        oof_preds[va_idx] = m.predict(X_tr[va_idx])
    cv_rae = rae(y_tr, oof_preds)

    # Full-train refit → predict 513
    m_final = LGBMRegressor(**LGBM_PARAMS)
    m_final.fit(X_tr, y_tr)
    te_preds = m_final.predict(X_te)

    # RAE on 253 unblinded
    rae_253 = rae(unb_y, te_preds[unb_idx])
    return cv_rae, rae_253, te_preds

def mols_from_smiles(smiles_list):
    mols = []
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        mols.append(m)
    return mols

def fp_matrix(mols, fp_fn):
    """Compute fingerprint for each mol; return (N, bits) float32 array."""
    rows = []
    for m in mols:
        try:
            fp = fp_fn(m)
            arr = np.zeros(fp.GetNumBits(), dtype=np.float32)
            from rdkit.DataStructs import ConvertToNumpyArray
            ConvertToNumpyArray(fp, arr)
            rows.append(arr)
        except Exception:
            rows.append(None)
    # fill failed with median (will impute later)
    lengths = [r.shape[0] for r in rows if r is not None]
    L = lengths[0] if lengths else 1
    out = np.full((len(mols), L), np.nan, dtype=np.float32)
    for i, r in enumerate(rows):
        if r is not None:
            out[i] = r
    return out

# ── Mol objects ─────────────────────────────────────────────────────────────
print("Parsing SMILES to RDKit mol objects...")
mols_tr = mols_from_smiles(smiles_tr)
mols_te = mols_from_smiles(smiles_te)
cv_folds = get_cv_folds()
print(f"  CV folds: {len(cv_folds)}")

results = {}

# ═══════════════════════════════════════════════════════════════════════════
# FEATURIZER 1 — Avalon Fingerprints (1024-bit)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("FEATURIZER 1: Avalon Fingerprints (1024-bit)")
try:
    from rdkit.Avalon import pyAvalonTools
    t0 = time.time()

    def avalon_fn(mol):
        return pyAvalonTools.GetAvalonFP(mol, 1024)

    X_tr_av = fp_matrix(mols_tr, avalon_fn)
    X_te_av = fp_matrix(mols_te, avalon_fn)
    X_tr_av = impute(X_tr_av)
    X_te_av = impute(X_te_av)
    print(f"  Shape: {X_tr_av.shape}")

    cv_rae, rae_253, te_preds = train_predict(X_tr_av, ytr, X_te_av, cv_folds)
    np.save(f"{OUTDIR}/feat_avalon_te_513.npy", te_preds)
    results["avalon"] = {"cv_rae": float(cv_rae), "rae_253": float(rae_253), "shape": X_tr_av.shape[1]}
    print(f"  Scaffold CV RAE: {cv_rae:.4f}")
    print(f"  RAE on 253 unblinded: {rae_253:.4f}")
    print(f"  Time: {time.time()-t0:.1f}s")
except ImportError:
    print("  Avalon not available, skipping.")
    results["avalon"] = {"error": "not available"}

# ═══════════════════════════════════════════════════════════════════════════
# FEATURIZER 2 — ECFP6 (radius=3, 4096-bit)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("FEATURIZER 2: ECFP6 (radius=3, 4096-bit)")
t0 = time.time()

def ecfp6_fn(mol):
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=4096)

X_tr_e6 = fp_matrix(mols_tr, ecfp6_fn)
X_te_e6 = fp_matrix(mols_te, ecfp6_fn)
X_tr_e6 = impute(X_tr_e6)
X_te_e6 = impute(X_te_e6)
print(f"  Shape: {X_tr_e6.shape}")

cv_rae, rae_253, te_preds = train_predict(X_tr_e6, ytr, X_te_e6, cv_folds)
np.save(f"{OUTDIR}/feat_ecfp6_te_513.npy", te_preds)
results["ecfp6"] = {"cv_rae": float(cv_rae), "rae_253": float(rae_253), "shape": X_tr_e6.shape[1]}
print(f"  Scaffold CV RAE: {cv_rae:.4f}")
print(f"  RAE on 253 unblinded: {rae_253:.4f}")
print(f"  Time: {time.time()-t0:.1f}s")

# ═══════════════════════════════════════════════════════════════════════════
# FEATURIZER 3 — Topological Torsion (2048-bit)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("FEATURIZER 3: Topological Torsion (hashed, 2048-bit)")
t0 = time.time()

def torsion_fn(mol):
    return rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=2048)

X_tr_tt = fp_matrix(mols_tr, torsion_fn)
X_te_tt = fp_matrix(mols_te, torsion_fn)
X_tr_tt = impute(X_tr_tt)
X_te_tt = impute(X_te_tt)
print(f"  Shape: {X_tr_tt.shape}")

cv_rae, rae_253, te_preds = train_predict(X_tr_tt, ytr, X_te_tt, cv_folds)
np.save(f"{OUTDIR}/feat_torsion_te_513.npy", te_preds)
results["torsion"] = {"cv_rae": float(cv_rae), "rae_253": float(rae_253), "shape": X_tr_tt.shape[1]}
print(f"  Scaffold CV RAE: {cv_rae:.4f}")
print(f"  RAE on 253 unblinded: {rae_253:.4f}")
print(f"  Time: {time.time()-t0:.1f}s")

# ═══════════════════════════════════════════════════════════════════════════
# FEATURIZER 4 — MACCS Keys (166 structural keys)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("FEATURIZER 4: MACCS Keys (166 bits)")
t0 = time.time()

def maccs_fn(mol):
    return MACCSkeys.GenMACCSKeys(mol)

X_tr_mc = fp_matrix(mols_tr, maccs_fn)
X_te_mc = fp_matrix(mols_te, maccs_fn)
X_tr_mc = impute(X_tr_mc)
X_te_mc = impute(X_te_mc)
print(f"  Shape: {X_tr_mc.shape}")

cv_rae, rae_253, te_preds = train_predict(X_tr_mc, ytr, X_te_mc, cv_folds)
np.save(f"{OUTDIR}/feat_maccs_te_513.npy", te_preds)
results["maccs"] = {"cv_rae": float(cv_rae), "rae_253": float(rae_253), "shape": X_tr_mc.shape[1]}
print(f"  Scaffold CV RAE: {cv_rae:.4f}")
print(f"  RAE on 253 unblinded: {rae_253:.4f}")
print(f"  Time: {time.time()-t0:.1f}s")

# ═══════════════════════════════════════════════════════════════════════════
# FEATURIZER 5 — 3D Shape Descriptors (ETKDG conformers)
# PMI1/2/3, NPR1/2, Asphericity, Eccentricity, InertialShapeFactor,
# SpherocityIndex, RadiusOfGyration
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("FEATURIZER 5: 3D Shape Descriptors (ETKDG + MMFF)")
t0 = time.time()

def compute_3d_shape(mol):
    """Generate 3D conformer and compute shape descriptors. Returns 10 floats."""
    try:
        m3 = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        result = AllChem.EmbedMolecule(m3, params)
        if result == -1:
            # Try without seed
            result = AllChem.EmbedMolecule(m3, AllChem.ETKDGv3())
        if result == -1:
            return [np.nan] * 10
        AllChem.MMFFOptimizeMolecule(m3, maxIters=500)
        pmi1 = rdMolDescriptors.CalcPMI1(m3)
        pmi2 = rdMolDescriptors.CalcPMI2(m3)
        pmi3 = rdMolDescriptors.CalcPMI3(m3)
        npr1 = rdMolDescriptors.CalcNPR1(m3)
        npr2 = rdMolDescriptors.CalcNPR2(m3)
        asph = rdMolDescriptors.CalcAsphericity(m3)
        eccn = rdMolDescriptors.CalcEccentricity(m3)
        inrt = rdMolDescriptors.CalcInertialShapeFactor(m3)
        sphr = rdMolDescriptors.CalcSpherocityIndex(m3)
        rgyration = rdMolDescriptors.CalcRadiusOfGyration(m3)
        return [pmi1, pmi2, pmi3, npr1, npr2, asph, eccn, inrt, sphr, rgyration]
    except Exception:
        return [np.nan] * 10

SHAPE_COLS = ["pmi1","pmi2","pmi3","npr1","npr2","asphericity",
              "eccentricity","inertial_sf","spherocity","radius_gyration"]

print("  Computing 3D conformers for train (4139 mols)...")
shape_tr = []
for i, mol in enumerate(mols_tr):
    if i % 500 == 0:
        print(f"    {i}/{n_tr}...")
    if mol is None:
        shape_tr.append([np.nan] * 10)
    else:
        shape_tr.append(compute_3d_shape(mol))

print("  Computing 3D conformers for test (513 mols)...")
shape_te = []
for i, mol in enumerate(mols_te):
    if i % 100 == 0:
        print(f"    {i}/{n_te}...")
    if mol is None:
        shape_te.append([np.nan] * 10)
    else:
        shape_te.append(compute_3d_shape(mol))

X_tr_3d = np.array(shape_tr, dtype=np.float32)
X_te_3d = np.array(shape_te, dtype=np.float32)
nan_frac = np.isnan(X_tr_3d).mean()
print(f"  Shape: {X_tr_3d.shape}, NaN fraction: {nan_frac:.3f}")
X_tr_3d = impute(X_tr_3d)
X_te_3d = impute(X_te_3d)

cv_rae, rae_253, te_preds = train_predict(X_tr_3d, ytr, X_te_3d, cv_folds)
np.save(f"{OUTDIR}/feat_shape3d_te_513.npy", te_preds)
results["shape3d"] = {"cv_rae": float(cv_rae), "rae_253": float(rae_253),
                      "shape": X_tr_3d.shape[1], "nan_frac": float(nan_frac)}
print(f"  Scaffold CV RAE: {cv_rae:.4f}")
print(f"  RAE on 253 unblinded: {rae_253:.4f}")
print(f"  Time: {time.time()-t0:.1f}s")

# ═══════════════════════════════════════════════════════════════════════════
# FEATURIZER 6 — Concatenated: ECFP6(4096) + MACCS(166) + rdkit_desc(217)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("FEATURIZER 6: Concatenated ECFP6(4096) + MACCS(166) + rdkit_desc(217)")
t0 = time.time()

print("  Computing rdkit_desc for train...")
from src.pxr.featurize import rdkit_desc as rdkit_featurize
Xrd_tr = rdkit_featurize(smiles_tr)
Xrd_te = rdkit_featurize(smiles_te)
print(f"  rdkit_desc shape: {Xrd_tr.shape}")

X_tr_cat = np.hstack([X_tr_e6, X_tr_mc, Xrd_tr])
X_te_cat = np.hstack([X_te_e6, X_te_mc, Xrd_te])
print(f"  Concatenated shape: {X_tr_cat.shape}")
X_tr_cat = impute(X_tr_cat)
X_te_cat = impute(X_te_cat)

cv_rae, rae_253, te_preds = train_predict(X_tr_cat, ytr, X_te_cat, cv_folds)
np.save(f"{OUTDIR}/feat_concat_te_513.npy", te_preds)
results["concat_ecfp6_maccs_rdkit"] = {"cv_rae": float(cv_rae), "rae_253": float(rae_253),
                                        "shape": X_tr_cat.shape[1]}
print(f"  Scaffold CV RAE: {cv_rae:.4f}")
print(f"  RAE on 253 unblinded: {rae_253:.4f}")
print(f"  Time: {time.time()-t0:.1f}s")

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SUMMARY")
print(f"{'Featurizer':<35} {'ScaffoldCV RAE':>15} {'RAE@253':>10} {'Dim':>6}")
print("-" * 70)
for name, r in results.items():
    if "error" in r:
        print(f"{name:<35} {'N/A':>15} {'N/A':>10} {'N/A':>6}")
    else:
        print(f"{name:<35} {r['cv_rae']:>15.4f} {r['rae_253']:>10.4f} {r.get('shape', '?'):>6}")

# Save results
with open(RESULTS_JSON, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {RESULTS_JSON}")
print("Predictions saved to: C:/pxr_work/meta_stacking/feat_*_te_513.npy")
