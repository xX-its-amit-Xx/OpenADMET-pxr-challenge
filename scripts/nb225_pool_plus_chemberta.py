"""nb225 -- Extend SLSQP pool: add ChemBERTa-derived models on top of nb224.

nb224 best: 0.289087 (pool + nb219 + nb228)

ChemBERTa is a 600M-compound pretrained transformer; its embeddings encode
chemical patterns that 2D fingerprints don't directly express. We have
multiple pretrain variants from earlier work.

Add to pool: nb213_chemberta_mtr, nb213_chemberta_zinc, nb213_zinc_lgbm,
nb213_zinc_mlp, nb215_chemist_features, nb217_pxr_modes, nb220_mmp_features.
Each is a saved OOF/test array; let SLSQP pick the best weight combination.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

sys.path.insert(0, str(Path(__file__).parent))
from nb197_dense_grid import (
    constrained_slsqp, build_candidates, load_pool, NB188_POOL, COLLAPSE_THRESH, SEED
)

# Extended pool — nb224's pool + ChemBERTa + 3D modes + MMP
EXTRA_CANDIDATES = [
    "nb219_aug_30pct",
    "nb228_medchem",
    "nb213_chemberta_mtr",
    "nb213_chemberta_zinc",
    "nb213_zinc_lgbm",
    "nb213_zinc_mlp",
    "nb215_chemist_features",
    "nb217_pxr_modes",
    "nb220_mmp_features",
]


def main():
    print("=== nb225: Extended SLSQP pool (ChemBERTa + 3D + MMP) ===\n")
    print("nb224 best: 0.289087 (target to beat)\n")

    tr = load_train(); te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, 5, SEED)

    base_oofs, base_tes, base_stems = load_pool(n_tr)

    oof183_p = DATA_PROCESSED / "oof_nb183_qreg_poly10.npy"
    te183_p  = DATA_PROCESSED / "te_nb183_qreg_poly10.npy"
    if oof183_p.exists():
        oof183 = np.load(oof183_p).astype(np.float64).flatten()
        te183  = np.load(te183_p).astype(np.float64).flatten()
        oof183 = np.where(np.isfinite(oof183), oof183, np.nanmean(oof183))
        te183  = np.where(np.isfinite(te183),  te183,  np.nanmean(te183))
        base_oofs.insert(0, oof183); base_tes.insert(0, te183)
        base_stems.insert(0, "nb183_qreg_poly10")

    POOL_PLUS = list(NB188_POOL) + EXTRA_CANDIDATES
    print(f"Anchor pool ({len(POOL_PLUS)} models):")

    pool_oofs_tr, pool_oofs_te, pool_names = [], [], []
    for stem in POOL_PLUS:
        oof_p = DATA_PROCESSED / f"oof_{stem}.npy"
        te_p  = DATA_PROCESSED / f"te_{stem}.npy"
        if not oof_p.exists() or not te_p.exists():
            print(f"  MISSING: {stem}")
            continue
        oof_m = np.load(oof_p).astype(np.float64).flatten()
        te_m  = np.load(te_p).astype(np.float64).flatten()
        # Length check — some may be different
        if len(oof_m) != n_tr or len(te_m) != len(te_df):
            print(f"  WRONG SIZE: {stem}  oof_len={len(oof_m)} expected={n_tr}, te_len={len(te_m)} expected={len(te_df)}")
            continue
        oof_m = np.where(np.isfinite(oof_m), oof_m, np.nanmean(oof_m))
        te_m  = np.where(np.isfinite(te_m),  te_m,  np.nanmean(te_m))
        pool_oofs_tr.append(oof_m); pool_oofs_te.append(te_m); pool_names.append(stem)
        r = rae(y_tr, oof_m); ratio = te_m.std() / oof_m.std()
        print(f"  {stem:30s}: OOF={r:.4f}  ratio={ratio:.3f}")

    poly2 = PolynomialFeatures(degree=2, include_bias=False)

    print("\n--- Dense grid candidates ---")
    alpha_dense_15 = sorted(set([round(a, 5) for a in
        list(np.linspace(0.0005, 0.004, 30)) + [0.005, 0.006, 0.007, 0.008, 0.009]]))
    alpha_high = [0.01, 0.015, 0.02, 0.03]

    cand_a_oofs, cand_a_tes, _ = build_candidates(
        base_oofs, base_tes, base_stems, y_tr, splits, poly2,
        div_k_list=[15, 20], alpha_list=alpha_dense_15 + alpha_high)
    cand_a_25_oofs, cand_a_25_tes, _ = build_candidates(
        base_oofs, base_tes, base_stems, y_tr, splits, poly2,
        div_k_list=[25], alpha_list=[0.008, 0.01, 0.015, 0.02, 0.03])
    all_cands_oofs = cand_a_oofs + cand_a_25_oofs
    all_cands_tes  = cand_a_tes  + cand_a_25_tes
    print(f"\nTotal candidates: {len(all_cands_oofs)}")

    print(f"\n--- SLSQP (pool={len(pool_oofs_tr)}, candidates={len(all_cands_oofs)}) ---")
    X_tr = np.column_stack(pool_oofs_tr + all_cands_oofs)
    X_te = np.column_stack(pool_oofs_te + all_cands_tes)
    r_best, oof_best, te_best, ratio_best = constrained_slsqp(
        X_tr, X_te, y_tr, n_starts=2000, prev_best=0.289087)

    if oof_best is not None:
        print(f"\nnb225 final: OOF={r_best:.6f}  ratio={ratio_best:.4f}")
        if r_best < 0.289087:
            np.save(DATA_PROCESSED / "oof_nb225_extended_pool.npy", oof_best)
            np.save(DATA_PROCESSED / "te_nb225_extended_pool.npy", te_best)
            te_raw = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
            smap = dict(zip(te_raw["Molecule Name"], te_raw["SMILES"]))
            sub = pd.DataFrame({"Molecule Name": te_df["name"], "pEC50": te_best})
            sub["SMILES"] = sub["Molecule Name"].map(smap)
            sub = sub[["SMILES", "Molecule Name", "pEC50"]]
            sub.to_csv(SUBMISSIONS / "225_extended_pool.csv", index=False)
            print(f"\nSaved 225_extended_pool.csv (NEW BEST: {r_best:.6f}, beats nb224 0.2891)")
        else:
            print(f"\nNo gain vs nb224 (0.2891).")


if __name__ == "__main__":
    main()
