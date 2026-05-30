"""nb248 -- Per-test-compound LOCAL personalized ensemble.

Idea: for each test compound, its OPTIMAL blend weights depend on its local
chemistry. A test compound similar to rifampicin-like binders may best be
predicted by delta-ML, while a sulfonamide-like compound may best be predicted
by nb224.

For each test compound:
1. Find 50 nearest train compounds via Tanimoto.
2. For these 50, fit a LOCAL SLSQP blend over our top-7 OOF candidates.
3. Apply those LOCAL weights to the test compound's predictions.

This produces a per-compound personalized prediction.
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
from scipy.optimize import minimize
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
    print("=== nb248: per-compound LOCAL personalized ensemble ===\n")
    tr = load_train()
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["smiles"].tolist()
    smiles_te = te_df["SMILES"].tolist()
    te_names = te_df["Molecule Name"].tolist()
    sm = dict(zip(te_df["Molecule Name"], te_df["SMILES"]))

    P = Path("data/processed")
    # 7 candidate OOFs (curated)
    names = [
        "nb224_pool_plus_2", "nb179_stack", "multi_template_delta", "delta_loso",
        "nb145_xgb_stack", "counter_delta", "nb179_fresh",
    ]
    oofs = []
    tes = []
    for n in names:
        oofs.append(np.load(P / f"oof_{n}.npy"))
        for p in [P / f"te_{n}.npy", P / f"te_oof_{n}.npy"]:
            if p.exists():
                tes.append(np.load(p)); break
    M = np.column_stack(oofs)  # (4139, 7)
    T = np.column_stack(tes)  # (513, 7)
    print(f"M={M.shape}, T={T.shape}")

    # Baseline: equal weights
    print(f"Equal-weight blend OOF: {rae(y_tr, M.mean(axis=1)):.4f}")

    # Compute Tanimoto similarity test-vs-train
    print("\nComputing similarity matrix...")
    fps_tr = morgan_fps(smiles_tr)
    fps_te = morgan_fps(smiles_te)
    sim_matrix = np.zeros((len(smiles_te), len(smiles_tr)))
    for i, fp in enumerate(fps_te):
        if fp is None: continue
        sim_matrix[i] = BulkTanimotoSimilarity(fp, fps_tr)

    # For each test compound: top-K nearest neighbors, fit local SLSQP
    K = 50
    print(f"\nPersonalized fit with K={K} per test compound...")
    n_te = len(smiles_te)
    te_personalized = np.zeros(n_te)

    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0, 1.0)] * len(names)

    for i in range(n_te):
        sims = sim_matrix[i]
        top_idx = np.argsort(sims)[::-1][:K]
        M_local = M[top_idx]
        y_local = y_tr[top_idx]
        weights_sim = sims[top_idx]  # similarity weights for the regression

        def loss(w):
            pred = M_local @ w
            # Weighted MAE
            return float(np.mean(weights_sim * np.abs(y_local - pred)) / np.mean(weights_sim * np.abs(y_local - y_local.mean())))

        # Quick local SLSQP
        rng = np.random.default_rng(42 + i)
        best = None
        for seed in range(10):
            w0 = rng.dirichlet(np.ones(len(names)))
            try:
                res = minimize(loss, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"ftol": 1e-7, "maxiter": 100})
                if best is None or res.fun < best.fun:
                    best = res
            except Exception: pass
        if best is None:
            te_personalized[i] = T[i].mean()
        else:
            te_personalized[i] = T[i] @ best.x

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_te}")

    # Compute OOF estimate via LOOCV
    print("\nOOF check via LOOCV (each train compound personalized from its 50-NN excluding self)...")
    # Compute train-vs-train similarities (LOOCV)
    sim_tt = np.zeros((len(smiles_tr), len(smiles_tr)))
    for i in range(len(smiles_tr)):
        if fps_tr[i] is None: continue
        sim_tt[i] = BulkTanimotoSimilarity(fps_tr[i], fps_tr)
    sim_tt[np.arange(len(smiles_tr)), np.arange(len(smiles_tr))] = -1  # exclude self

    oof_personalized = np.zeros(len(smiles_tr))
    for i in range(len(smiles_tr)):
        sims = sim_tt[i]
        top_idx = np.argsort(sims)[::-1][:K]
        M_local = M[top_idx]
        y_local = y_tr[top_idx]
        weights_sim = sims[top_idx] + 1e-6

        def loss(w):
            pred = M_local @ w
            return float(np.mean(weights_sim * np.abs(y_local - pred)) / max(np.mean(weights_sim * np.abs(y_local - y_local.mean())), 1e-6))

        rng = np.random.default_rng(42 + i)
        best = None
        for seed in range(5):
            w0 = rng.dirichlet(np.ones(len(names)))
            try:
                res = minimize(loss, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"ftol": 1e-6, "maxiter": 50})
                if best is None or res.fun < best.fun:
                    best = res
            except: pass
        if best is None:
            oof_personalized[i] = M[i].mean()
        else:
            oof_personalized[i] = M[i] @ best.x
        if (i + 1) % 500 == 0:
            print(f"  OOF {i+1}/{len(smiles_tr)}")

    print(f"\nPersonalized OOF RAE: {rae(y_tr, oof_personalized):.4f}")
    print(f"te personalized: mean={te_personalized.mean():.3f}, std={te_personalized.std():.3f}")

    np.save(P / "oof_nb248_personalized.npy", oof_personalized)
    np.save(P / "te_nb248_personalized.npy", te_personalized)

    # Save submission
    sub = pd.DataFrame({"SMILES": [sm[n] for n in te_names], "Molecule Name": te_names, "pEC50": te_personalized})
    sub.to_csv(SUBMISSIONS / "248_personalized_local.csv", index=False)
    print("Saved 248_personalized_local.csv")


if __name__ == "__main__":
    main()
