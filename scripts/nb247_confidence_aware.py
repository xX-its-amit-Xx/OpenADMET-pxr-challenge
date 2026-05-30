"""nb247 -- Confidence-aware test predictions via OOF disagreement.

Wild idea: across our 30+ OOF arrays, compute per-train-compound DISAGREEMENT.
Compounds with high disagreement are "hard" — train ML models fail on them.
For HARD test compounds (those with hard nearest-train-neighbors), shrink the
prediction toward the train median.

Then submit a confidence-shrunk version of nb239.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity
from pathlib import Path

from pxr.data import load_train, load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


def morgan_fps(smiles, radius=2, n_bits=2048):
    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits) if mol else None)
    return fps


def main():
    print("=== nb247: Confidence-aware test predictions ===\n")
    tr = load_train()
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["smiles"].tolist()
    smiles_te = te_df["SMILES"].tolist()
    te_names = te_df["Molecule Name"].tolist()
    sm = dict(zip(te_df["Molecule Name"], te_df["SMILES"]))

    # === Step 1: per-train-compound disagreement ===
    P = Path("data/processed")
    LEAKAGE = {'oof_grand_v11.npy', 'oof_grand_v6.npy', 'oof_grand_v8.npy',
               'oof_grand_v9.npy', 'oof_grand_v10.npy', 'oof_delta_5tiers.npy',
               'oof_adaptive_delta_4tier.npy', 'oof_delta_ensemble_blend.npy',
               'oof_aux_features.npy', 'oof_creative_mega_ensemble.npy',
               'oof_full_desc_delta_3tier.npy', 'oof_allfp_delta_3tier.npy',
               'oof_blend_optimizer.npy', 'oof_enhanced_delta_3tier.npy',
               'oof_delta_similarity_tiers.npy', 'oof_nb116_quantile_width.npy',
               'oof_nb239_full_slsqp.npy', 'oof_nb240_safe_slsqp.npy',
               'oof_nb242_huber.npy', 'oof_nb241_7way.npy', 'oof_nb244_deep_greedy.npy'}

    oofs = []
    for f in P.glob("oof_*.npy"):
        if f.name in LEAKAGE: continue
        try:
            a = np.load(f)
            if len(a) == len(y_tr) and np.isfinite(a).all() and a.std() > 0.3:
                r = rae(y_tr, a)
                if r < 0.85:
                    oofs.append(a)
        except: pass
    print(f"Collected {len(oofs)} non-leakage OOFs")
    M = np.column_stack(oofs)  # (n_train, n_models)
    print(f"M shape: {M.shape}")

    # Per-train-compound disagreement = std across models
    train_disagreement = M.std(axis=1)
    # Per-train-compound residual (avg |y_true - y_pred|)
    train_residual = np.abs(M - y_tr[:, None]).mean(axis=1)
    # Hardness score: high disagreement + high residual
    hardness_tr = train_disagreement * train_residual / (np.median(train_disagreement) * np.median(train_residual))
    print(f"Train hardness: min={hardness_tr.min():.3f}, max={hardness_tr.max():.3f}, median={np.median(hardness_tr):.3f}")

    # === Step 2: compute test compound hardness via nearest-neighbor train hardness ===
    print("\nComputing test hardness via nearest train hardness...")
    fps_tr = morgan_fps(smiles_tr)
    fps_te = morgan_fps(smiles_te)
    hardness_te = np.zeros(len(smiles_te))
    for i, fp in enumerate(fps_te):
        if fp is None:
            hardness_te[i] = 1.0
            continue
        sims = np.array(BulkTanimotoSimilarity(fp, fps_tr))
        top_idx = np.argsort(sims)[::-1][:5]
        weights = sims[top_idx] + 0.01
        weights = weights / weights.sum()
        hardness_te[i] = (weights * hardness_tr[top_idx]).sum()
    print(f"Test hardness: min={hardness_te.min():.3f}, max={hardness_te.max():.3f}, median={np.median(hardness_te):.3f}")

    # === Step 3: Apply confidence-aware shrinkage to nb239 ===
    nb239_te = np.load(P / "te_nb239_full_slsqp.npy")
    train_median = np.median(y_tr)
    print(f"\nnb239 te: mean={nb239_te.mean():.3f}, std={nb239_te.std():.3f}")
    print(f"Train median pec50: {train_median:.3f}")

    # Normalize hardness to 0-1 (top 10% → max shrink)
    h_norm = (hardness_te - hardness_te.min()) / max(hardness_te.max() - hardness_te.min(), 1e-6)
    # Cap at a top percentile to avoid over-shrinking
    h_clipped = np.clip(h_norm, 0, 0.7)

    print("\nShrinkage variants:")
    for max_shrink in [0.1, 0.2, 0.3, 0.5]:
        shrink_w = h_clipped * max_shrink  # per-compound shrinkage 0 to max_shrink
        te_shrunk = (1 - shrink_w) * nb239_te + shrink_w * train_median
        print(f"  max_shrink={max_shrink}: mean={te_shrunk.mean():.3f}, std={te_shrunk.std():.3f}")
        # Save submission
        sub = pd.DataFrame({"SMILES": [sm[n] for n in te_names], "Molecule Name": te_names, "pEC50": te_shrunk})
        sub.to_csv(SUBMISSIONS / f"247_conf_shrink_max{int(max_shrink*100):03d}.csv", index=False)
    print("Saved 4 confidence-shrunk variants")

    # === Step 4: also do confidence-aware variance EXPANSION for confident compounds ===
    # Confident = low hardness = trust the prediction fully. Hard = shrink.
    # Actually, the inverse: for high-confidence (low hardness), maybe BOOST?
    # Let's compute the OOF mean and see if shrinking high-hardness train compounds toward median helps OOF
    # If yes → suggests test approach works
    print("\n=== Sanity check on OOF (train) ===")
    # Apply the shrinkage to OOF predictions of nb239 (built from train)
    nb239_oof = np.load(P / "oof_nb239_full_slsqp.npy")
    h_tr_norm = (hardness_tr - hardness_tr.min()) / max(hardness_tr.max() - hardness_tr.min(), 1e-6)
    h_tr_clipped = np.clip(h_tr_norm, 0, 0.7)
    print(f"nb239 OOF baseline RAE: {rae(y_tr, nb239_oof):.4f}")
    for max_shrink in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        shrink_w = h_tr_clipped * max_shrink
        oof_shrunk = (1 - shrink_w) * nb239_oof + shrink_w * train_median
        r = rae(y_tr, oof_shrunk)
        sign = "improves" if r < rae(y_tr, nb239_oof) else "worse"
        print(f"  shrink={max_shrink}: OOF RAE={r:.4f} ({sign})")

    np.save(P / "te_nb247_hardness.npy", hardness_te)
    np.save(P / "oof_nb247_hardness.npy", hardness_tr)
    print("\nSaved hardness arrays")


if __name__ == "__main__":
    main()
