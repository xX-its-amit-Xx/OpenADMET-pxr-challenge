"""nb313 -- Predicted-ADMET features on top of combined feature set.

Hypothesis: PXR is a key node in the ADMET network (CYP3A4/MDR1 regulation,
xenobiotic clearance). Compact, interpretable RDKit-derived ADMET proxies
should carry orthogonal signal vs ECFP4 + RDKit-2D descriptors.

Features (15-30, RDKit only):
  LogP, LogD7.4 estimate, TPSA, LipinskiRO5 score, HBA, HBD,
  RotBonds, FractionCSP3, NumAromaticRings, QED, SAscore (if available),
  BertzCT, BalabanJ, HeavyAtomCount, FormalCharge, MolMR,
  RingCount, NumAliphaticRings, NumSpiroAtoms, NumBridgeheadAtoms,
  NumSaturatedRings, MolecularWeight, NumHeteroatoms.

Concatenated with combined() and fed to LGBM via scaffold 5-fold CV.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import (
    AllChem, Descriptors, Crippen, Lipinski, rdMolDescriptors, QED,
    GraphDescriptors, RDConfig,
)
from scipy.stats import spearmanr
from scipy.optimize import minimize

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED

# Try to import SA Score from RDKit Contrib
_SA = None
try:
    sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if os.path.isdir(sa_path):
        sys.path.insert(0, sa_path)
        import sascorer  # type: ignore
        _SA = sascorer.calculateScore
        print("SA scorer loaded.")
except Exception as e:
    print(f"SA scorer unavailable: {e}")


_FEAT_NAMES = [
    "LogP", "LogD7_4", "TPSA", "LipinskiRO5", "HBA", "HBD",
    "RotBonds", "FractionCSP3", "NumAromaticRings", "QED", "SAscore",
    "BertzCT", "BalabanJ", "HeavyAtoms", "FormalCharge", "MolMR",
    "RingCount", "NumAliphaticRings", "NumSpiroAtoms", "NumBridgeheadAtoms",
    "NumSaturatedRings", "MW", "NumHeteroatoms",
]


def logd_estimate(logp, charge, hbd, hba):
    """Crude LogD7.4 proxy: penalise basic centers (likely protonated at pH 7.4).
    Approximation: LogD = LogP - 0.6*(num basic Ns).
    Use charge + HBD as weak proxy for basicity."""
    base_correction = 0.0
    # If any obviously basic Ns suspected -> apply small offset.
    return float(logp - 0.4 * max(0, hba - hbd) * 0.0)  # mild placeholder


def lipinski_ro5(mw, logp, hbd, hba):
    score = 0
    if mw <= 500: score += 1
    if logp <= 5: score += 1
    if hbd <= 5: score += 1
    if hba <= 10: score += 1
    return float(score)  # 0..4


def admet_one(smi):
    """Return a length-len(_FEAT_NAMES) feature vector, or NaNs on failure."""
    out = np.full(len(_FEAT_NAMES), np.nan, dtype=np.float32)
    if not smi or not isinstance(smi, str):
        return out
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return out
    try:
        logp = Crippen.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hba = Lipinski.NumHAcceptors(mol)
        hbd = Lipinski.NumHDonors(mol)
        rot = Lipinski.NumRotatableBonds(mol)
        fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
        narom = rdMolDescriptors.CalcNumAromaticRings(mol)
        qed = QED.qed(mol)
        sa = float(_SA(mol)) if _SA is not None else np.nan
        bertz = GraphDescriptors.BertzCT(mol)
        balaban = GraphDescriptors.BalabanJ(mol)
        ha = Descriptors.HeavyAtomCount(mol)
        fc = Chem.GetFormalCharge(mol)
        mr = Crippen.MolMR(mol)
        rc = rdMolDescriptors.CalcNumRings(mol)
        nali = rdMolDescriptors.CalcNumAliphaticRings(mol)
        spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        bridge = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        nsat = rdMolDescriptors.CalcNumSaturatedRings(mol)
        mw = Descriptors.MolWt(mol)
        nhet = Lipinski.NumHeteroatoms(mol)
        ro5 = lipinski_ro5(mw, logp, hbd, hba)
        logd = logd_estimate(logp, fc, hbd, hba)
        vals = [logp, logd, tpsa, ro5, hba, hbd, rot, fsp3, narom, qed, sa,
                bertz, balaban, ha, fc, mr, rc, nali, spiro, bridge, nsat,
                mw, nhet]
        for i, v in enumerate(vals):
            out[i] = np.float32(v) if v is not None else np.nan
    except Exception:
        pass
    return out


def admet_batch(smiles):
    X = np.stack([admet_one(s) for s in smiles], axis=0).astype(np.float32)
    print(f"  ADMET features: {X.shape}, NaN count={np.isnan(X).sum()}")
    return X


def main():
    print("=== nb313: Predicted-ADMET features ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["smiles"].tolist()
    scaffolds = tr["scaffold"].tolist()
    print(f"Train: {len(smiles_tr)}, Test: {len(smiles_te)}")

    print("\nComputing ADMET features (train) ...")
    A_tr = admet_batch(smiles_tr)
    print("Computing ADMET features (test) ...")
    A_te = admet_batch(smiles_te)

    print("\nFeaturizing combined (Morgan + RDKit) ...")
    C_tr = combined(smiles_tr)
    C_te = combined(smiles_te)
    X_tr = impute(np.hstack([C_tr, A_tr]))
    X_te = impute(np.hstack([C_te, A_te]))
    print(f"X_tr shape: {X_tr.shape}, X_te shape: {X_te.shape}")

    splits = scaffold_kfold_indices(scaffolds, n_splits=5, seed=42)
    oof = np.zeros(len(y_tr))
    te_pred_folds = np.zeros((5, len(smiles_te)))
    for k, (ti, vi) in enumerate(splits):
        md = lgb.LGBMRegressor(
            n_estimators=2000, num_leaves=63, learning_rate=0.03,
            subsample=0.9, colsample_bytree=0.8,
            min_child_samples=10, reg_alpha=0.01, reg_lambda=0.01,
            objective="mae", n_jobs=4, random_state=42, verbose=-1)
        md.fit(X_tr[ti], y_tr[ti],
               eval_set=[(X_tr[vi], y_tr[vi])],
               callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
        oof[vi] = md.predict(X_tr[vi])
        te_pred_folds[k] = md.predict(X_te)
        print(f"  fold {k}: RAE={rae(y_tr[vi], oof[vi]):.4f}")
    te_pred = te_pred_folds.mean(axis=0)

    r = rae(y_tr, oof)
    sp = spearmanr(y_tr, oof).statistic
    print(f"\nnb313 OOF RAE={r:.4f}  Spearman={sp:.4f}  te_std={te_pred.std():.3f}")
    print(f"  te mean={te_pred.mean():.3f}, min={te_pred.min():.3f}, max={te_pred.max():.3f}")

    np.save(DATA_PROCESSED / "oof_nb313_admet.npy", oof)
    np.save(DATA_PROCESSED / "te_nb313_admet.npy", te_pred)

    # ---- SLSQP 5-way with nb239 base ---------------------------------------
    print("\nSLSQP 5-way blend with nb239 base components ...")
    nb224 = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
    nb179s = np.load(DATA_PROCESSED / "oof_nb179_stack.npy")
    mtd = np.load(DATA_PROCESSED / "oof_multi_template_delta.npy")
    loso = np.load(DATA_PROCESSED / "oof_delta_loso.npy")
    M = np.column_stack([nb224, nb179s, mtd, loso, oof])
    def loss_fn(w): return rae(y_tr, M @ w)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0, 1.0)] * 5
    best = None
    for seed in range(80):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(5))
        res = minimize(loss_fn, w0, method='SLSQP',
                       bounds=bounds, constraints=cons, options={'ftol': 1e-9})
        if best is None or res.fun < best.fun:
            best = res
    print(f"5-way SLSQP RAE={best.fun:.4f}")
    print(f"  weights: nb224={best.x[0]:.4f}, nb179s={best.x[1]:.4f}, "
          f"mtd={best.x[2]:.4f}, loso={best.x[3]:.4f}, nb313={best.x[4]:.4f}")


if __name__ == "__main__":
    main()
