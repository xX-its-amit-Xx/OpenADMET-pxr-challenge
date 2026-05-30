"""nb224 -- Add BOTH nb219 (single-conc aug) AND nb228 (medchem rules) to the
nb197 SLSQP pool. nb223 already showed nb219 alone lowers OOF to 0.292422;
adding nb228 may unlock further gain since the medchem rule features have
~0.660 ratio (higher than nb197's 0.580) and could shift the optimum.
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


def main():
    print("=== nb224: Add nb219 + nb228 to nb197 ensemble ===\n")
    print("nb223 best: 0.292422 (nb219 only)\n")

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

    POOL_PLUS = list(NB188_POOL) + ["nb219_aug_30pct", "nb228_medchem"]
    print(f"Pool anchors: {POOL_PLUS}")

    pool_oofs_tr, pool_oofs_te, pool_names = [], [], []
    for stem in POOL_PLUS:
        oof_p = DATA_PROCESSED / f"oof_{stem}.npy"
        te_p  = DATA_PROCESSED / f"te_{stem}.npy"
        if not oof_p.exists():
            print(f"  MISSING: {stem}  (skipping)")
            continue
        oof_m = np.load(oof_p).astype(np.float64).flatten()
        te_m  = np.load(te_p).astype(np.float64).flatten()
        oof_m = np.where(np.isfinite(oof_m), oof_m, np.nanmean(oof_m))
        te_m  = np.where(np.isfinite(te_m),  te_m,  np.nanmean(te_m))
        pool_oofs_tr.append(oof_m); pool_oofs_te.append(te_m); pool_names.append(stem)
        r = rae(y_tr, oof_m); ratio = te_m.std() / oof_m.std()
        print(f"  {stem}: OOF={r:.4f}  ratio={ratio:.3f}")

    poly2 = PolynomialFeatures(degree=2, include_bias=False)

    print("\n--- Dense grid candidates (matches nb197) ---")
    alpha_dense_15 = sorted(set([round(a, 5) for a in
        list(np.linspace(0.0005, 0.004, 30)) + [0.005, 0.006, 0.007, 0.008, 0.009]]))
    alpha_high = [0.01, 0.015, 0.02, 0.03]

    cand_a_oofs, cand_a_tes, cand_a_names = build_candidates(
        base_oofs, base_tes, base_stems, y_tr, splits, poly2,
        div_k_list=[15, 20], alpha_list=alpha_dense_15 + alpha_high)
    cand_a_25_oofs, cand_a_25_tes, _ = build_candidates(
        base_oofs, base_tes, base_stems, y_tr, splits, poly2,
        div_k_list=[25], alpha_list=[0.008, 0.01, 0.015, 0.02, 0.03])

    all_cands_oofs = cand_a_oofs + cand_a_25_oofs
    all_cands_tes  = cand_a_tes  + cand_a_25_tes
    print(f"\nTotal candidates: {len(all_cands_oofs)}")

    print(f"\n--- SLSQP with pool ({len(pool_oofs_tr)} anchors + {len(all_cands_oofs)} candidates) ---")
    X_tr = np.column_stack(pool_oofs_tr + all_cands_oofs)
    X_te = np.column_stack(pool_oofs_te + all_cands_tes)
    r_best, oof_best, te_best, ratio_best = constrained_slsqp(
        X_tr, X_te, y_tr, n_starts=1500, prev_best=0.292422)  # beat nb223

    if oof_best is not None:
        print(f"\nnb224 final: OOF={r_best:.6f}  ratio={ratio_best:.4f}")
        if r_best < 0.292422:
            np.save(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy", oof_best)
            np.save(DATA_PROCESSED / "te_nb224_pool_plus_2.npy", te_best)
            sub = pd.DataFrame({"Molecule Name": te_df["name"], "pEC50": te_best})
            # Add SMILES
            te_raw = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
            smap = dict(zip(te_raw["Molecule Name"], te_raw["SMILES"]))
            sub["SMILES"] = sub["Molecule Name"].map(smap)
            sub = sub[["SMILES", "Molecule Name", "pEC50"]]
            sub.to_csv(SUBMISSIONS / "224_pool_plus_2.csv", index=False)
            print(f"\nSaved 224_pool_plus_2.csv (NEW BEST: {r_best:.6f}, beats nb223 0.2924)")
        else:
            print(f"\nNo gain vs nb223 (0.2924). nb228 didn't unlock more.")


if __name__ == "__main__":
    main()
