"""nb286 -- Aggressive SMILES standardization variants on PXR training data.

User hypothesis: many PXR training compounds have weird PTMs / stereochemistry /
salts / isotopes / non-canonical tautomers that confuse models. Stripping
them (even though it slightly misrepresents the compound) may reduce
noise and improve generalisation. We test 4 progressively-aggressive
cleaning variants and pick the best by OOF RAE on a single LGBM.

Pipeline:
  1. Load 4139 train + 513 test compounds.
  2. For each SMILES, produce 4 cleaned variants:
       clean_v1: largest fragment + neutralize (pxr.chem.standardize)
       clean_v2: v1 + Chem.RemoveStereochemistry
       clean_v3: v1 + isotopes stripped (atom.SetIsotope(0))
       clean_v4: v3 + tautomer canonicalisation
  3. Count compounds CHANGED per variant; describe WHAT changed.
  4. Train a single LGBM (combined Morgan+RDKit) with scaffold 5-fold CV
     per variant; report OOF RAE, Spearman, Kendall, R^2.
  5. Save best variant's OOF + test preds.
  6. Save a CSV report + run SLSQP blend vs nb239 base.
"""
import os, sys, warnings, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, kendalltau
from scipy.optimize import minimize
from sklearn.metrics import r2_score

from rdkit import Chem
from rdkit.Chem import AllChem  # noqa: F401  (imported for parity with prompt)
from rdkit.Chem.MolStandardize import rdMolStandardize

import lightgbm as lgb

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns, standardize
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


# ---------------------------------------------------------------------------
# Cleaning variants
# ---------------------------------------------------------------------------

_TAUT_ENUM = rdMolStandardize.TautomerEnumerator()


def _mol_to_smiles(mol):
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def clean_v1(smi):
    """Largest fragment + neutralize (= pxr.chem.standardize)."""
    return _mol_to_smiles(standardize(smi))


def clean_v2(smi):
    """v1 + remove stereochemistry."""
    mol = standardize(smi)
    if mol is None:
        return None
    try:
        Chem.RemoveStereochemistry(mol)
        return _mol_to_smiles(mol)
    except Exception:
        return None


def clean_v3(smi):
    """v1 + strip isotopes (set isotope=0 on all atoms)."""
    mol = standardize(smi)
    if mol is None:
        return None
    try:
        for a in mol.GetAtoms():
            a.SetIsotope(0)
        return _mol_to_smiles(mol)
    except Exception:
        return None


def clean_v4(smi):
    """v3 + canonical tautomer."""
    mol = standardize(smi)
    if mol is None:
        return None
    try:
        for a in mol.GetAtoms():
            a.SetIsotope(0)
        mol = _TAUT_ENUM.Canonicalize(mol)
        return _mol_to_smiles(mol)
    except Exception:
        # tautomer enumeration can fail; fall back to v3
        return _mol_to_smiles(mol) if mol is not None else None


# ---------------------------------------------------------------------------
# Diff descriptions
# ---------------------------------------------------------------------------

_STEREO_CHARS = set("@/\\")


def describe_change(orig, new):
    """Return a short label for what changed between orig and new SMILES."""
    if orig is None or new is None:
        return "unparseable"
    if orig == new:
        return "unchanged"
    tags = []
    if any(c in orig for c in _STEREO_CHARS) and not any(c in new for c in _STEREO_CHARS):
        tags.append("stereo_removed")
    # Isotope detection: digits inside [..] block prefixed by mass
    if any(ch.isdigit() for ch in _isotope_tokens(orig)) and not any(ch.isdigit() for ch in _isotope_tokens(new)):
        tags.append("isotope_removed")
    if "." in orig and "." not in new:
        tags.append("fragment_removed")
    if "+" in orig or "-" in orig:
        if not ("+" in new or "-" in new):
            tags.append("charge_neutralized")
    if not tags:
        tags.append("tautomer_or_canonical")
    return ",".join(tags)


def _isotope_tokens(smi):
    """Return characters that look like isotope mass markers inside [..]."""
    out = []
    in_br = False
    for c in smi:
        if c == "[":
            in_br = True
            continue
        if c == "]":
            in_br = False
            continue
        if in_br and c.isdigit():
            out.append(c)
    return out


# ---------------------------------------------------------------------------
# Train + score
# ---------------------------------------------------------------------------

def score_variant(label, tr_smiles, te_smiles, y, scaffolds, n_tr, n_te, seed=42):
    """Featurize + scaffold 5-fold LGBM; return OOF + test preds + metrics."""
    print(f"\n  [{label}] featurizing {n_tr + n_te} compounds...")
    t0 = time.time()
    X_all = combined(list(tr_smiles) + list(te_smiles))
    X_all = impute(X_all)
    X_tr = X_all[:n_tr]
    X_te = X_all[n_tr:]
    print(f"  [{label}] featurization done in {time.time()-t0:.1f}s -> X shape {X_all.shape}")

    splits = scaffold_kfold_indices(scaffolds, n_splits=5, seed=seed)
    oof = np.zeros(n_tr, dtype=np.float64)
    te_preds = np.zeros((5, n_te), dtype=np.float64)
    for fold, (tr_idx, val_idx) in enumerate(splits):
        params = dict(
            objective="regression", metric="mae",
            n_estimators=500, num_leaves=64, learning_rate=0.05,
            feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
            min_data_in_leaf=20, verbosity=-1, n_jobs=4, seed=seed,
        )
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr[tr_idx], y[tr_idx])
        oof[val_idx] = m.predict(X_tr[val_idx])
        te_preds[fold] = m.predict(X_te)
    te_mean = te_preds.mean(axis=0)
    metrics = dict(
        rae=rae(y, oof),
        spearman=float(spearmanr(y, oof).statistic),
        kendall=float(kendalltau(y, oof).statistic),
        r2=float(r2_score(y, oof)),
        mae=float(np.abs(y - oof).mean()),
        te_std=float(te_mean.std()),
        te_mean=float(te_mean.mean()),
    )
    print(f"  [{label}] OOF RAE={metrics['rae']:.4f}  Spearman={metrics['spearman']:.4f}  "
          f"Kendall={metrics['kendall']:.4f}  R2={metrics['r2']:.4f}  te_std={metrics['te_std']:.3f}")
    return oof, te_mean, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== nb286: aggressive SMILES standardization variants ===\n")
    tr = load_train()
    te = load_test()
    n_tr = len(tr)
    n_te = len(te)
    print(f"train n={n_tr}  test n={n_te}")
    y = tr['pec50'].values.astype(np.float64)

    # We need scaffolds for CV — compute once on the original SMILES via standardize.
    # (Scaffold definition is robust across cleaning variants.)
    tr_std = add_standard_columns(tr.copy())
    scaffolds = tr_std['scaffold'].fillna('').tolist()

    # Original SMILES (used as comparison baseline)
    tr_orig = tr['smiles'].tolist()
    te_orig = te['smiles'].tolist()

    variants = {
        'clean_v1': clean_v1,
        'clean_v2': clean_v2,
        'clean_v3': clean_v3,
        'clean_v4': clean_v4,
    }

    # ============================
    # Step 1: produce variants + describe changes
    # ============================
    print("\n--- Step 1: produce variants + diff ---")
    smiles_by_variant = {}
    diff_summary = {}
    for label, fn in variants.items():
        print(f"\n[{label}] generating...")
        t0 = time.time()
        tr_new = [fn(s) for s in tr_orig]
        te_new = [fn(s) for s in te_orig]
        print(f"  done in {time.time()-t0:.1f}s")
        # Count changes vs original
        labels = [describe_change(o, n) for o, n in zip(tr_orig + te_orig, tr_new + te_new)]
        cnt = pd.Series(labels).value_counts().to_dict()
        n_changed = sum(v for k, v in cnt.items() if k not in ('unchanged',))
        print(f"  [{label}] {n_changed}/{n_tr + n_te} compounds CHANGED")
        for k, v in cnt.items():
            print(f"    {k:>30}: {v}")
        smiles_by_variant[label] = (tr_new, te_new)
        diff_summary[label] = dict(n_changed=n_changed, breakdown=cnt)

    # ============================
    # Step 2: train LGBM per variant
    # ============================
    print("\n--- Step 2: train LGBM per variant (scaffold 5-fold CV) ---")
    results = {}
    for label, (tr_new, te_new) in smiles_by_variant.items():
        # Replace None with empty -> combined() returns NaN row -> impute handles it
        tr_safe = [s if s is not None else 'C' for s in tr_new]
        te_safe = [s if s is not None else 'C' for s in te_new]
        oof, tepred, metrics = score_variant(label, tr_safe, te_safe, y, scaffolds, n_tr, n_te)
        results[label] = dict(oof=oof, te=tepred, metrics=metrics)

    # ============================
    # Step 3: assemble report
    # ============================
    print("\n--- Step 3: report ---")
    rows = []
    for label, r in results.items():
        rows.append({
            'variant': label,
            'n_changed': diff_summary[label]['n_changed'],
            'oof_rae': r['metrics']['rae'],
            'oof_spearman': r['metrics']['spearman'],
            'oof_kendall': r['metrics']['kendall'],
            'oof_r2': r['metrics']['r2'],
            'te_std': r['metrics']['te_std'],
            'te_mean': r['metrics']['te_mean'],
        })
    rpt = pd.DataFrame(rows).sort_values('oof_rae').reset_index(drop=True)
    rpt_path = DATA_PROCESSED / "nb286_smiles_quality_report.csv"
    rpt.to_csv(rpt_path, index=False)
    print(f"\nReport (sorted by OOF RAE):\n{rpt.to_string(index=False)}")
    print(f"\nWrote {rpt_path}")

    # ============================
    # Step 4: save best variant's OOF + TE
    # ============================
    best_label = rpt.iloc[0]['variant']
    best_n = best_label.replace('clean_v', '')
    best_r = results[best_label]
    oof_path = DATA_PROCESSED / f"oof_nb286_clean_v{best_n}.npy"
    te_path = DATA_PROCESSED / f"te_nb286_clean_v{best_n}.npy"
    np.save(oof_path, best_r['oof'])
    np.save(te_path, best_r['te'])
    print(f"\nBest variant: {best_label}  RAE={best_r['metrics']['rae']:.4f}")
    print(f"Wrote {oof_path}\nWrote {te_path}")

    # ============================
    # Step 5: SLSQP blend vs nb239 base
    # ============================
    print("\n--- Step 5: SLSQP blend vs nb239 base ---")
    base_oof_p = DATA_PROCESSED / "oof_nb239_full_slsqp.npy"
    base_te_p = DATA_PROCESSED / "te_nb239_full_slsqp.npy"
    if not (base_oof_p.exists() and base_te_p.exists()):
        print(f"  nb239 base not found at {base_oof_p}, skipping blend")
        return
    base_oof = np.load(base_oof_p)
    base_te = np.load(base_te_p)
    print(f"  nb239 base: OOF RAE={rae(y, base_oof):.4f}  te_std={base_te.std():.3f}")
    nb286_oof = best_r['oof']
    nb286_te = best_r['te']
    print(f"  nb286 ({best_label}): OOF RAE={rae(y, nb286_oof):.4f}  te_std={nb286_te.std():.3f}")

    M_oof = np.column_stack([base_oof, nb286_oof])
    M_te = np.column_stack([base_te, nb286_te])

    def loss(w):
        return rae(y, M_oof @ w)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0, 1.0)] * 2
    best = None
    for seed in range(200):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(2))
        res = minimize(loss, w0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'ftol': 1e-9})
        if best is None or res.fun < best.fun:
            best = res
    w_base, w_nb286 = best.x
    blend_oof = M_oof @ best.x
    blend_te = M_te @ best.x
    print(f"\nSLSQP blend (nb239 + nb286):")
    print(f"  weights: nb239={w_base:.4f}  nb286={w_nb286:.4f}")
    print(f"  OOF RAE={rae(y, blend_oof):.4f}  Spearman={spearmanr(y, blend_oof).statistic:.4f}  "
          f"te_std={blend_te.std():.3f}")

    # Save blend as a submission candidate
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    sub = pd.DataFrame({
        'Molecule Name': te_df['Molecule Name'],
        'SMILES': te_df['SMILES'],
        'pEC50': blend_te,
    })
    out_csv = SUBMISSIONS / f"nb286_clean_smiles_blend.csv"
    sub.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
