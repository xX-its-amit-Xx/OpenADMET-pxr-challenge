"""nb269 -- Binary router: nb239 vs nb243.

Per-compound, find which is better (nb239 vs nb243) — train classifier to
predict from molecular features. Then route at inference.

nb243 alone overfits LB (0.7638 vs 239 0.7487). But per-compound, nb243
is best for 153 compounds.

If we can perfectly route nb239→nb243 for those compounds:
- Use nb239 for ~3986 compounds (the 96%)
- Use nb243 for ~153 high-confidence-nb243 compounds

This is a conservative router that should be safe on LB.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from rdkit import Chem

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


def std_smi(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol) if mol else None
    except: return None


def main():
    print("=== nb269: Targeted nb239-vs-nb243 router ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["SMILES"].apply(std_smi).tolist()
    te_names = te_df["Molecule Name"].tolist()
    sm = dict(zip(te_df["Molecule Name"], te_df["SMILES"]))

    nb239_oof = np.load(DATA_PROCESSED / "oof_nb239_full_slsqp.npy")
    nb239_te = np.load(DATA_PROCESSED / "te_nb239_full_slsqp.npy")
    nb243_oof = np.load(DATA_PROCESSED / "oof_nb243_greedy.npy")
    nb243_te = np.load(DATA_PROCESSED / "te_nb243_greedy.npy")

    res_239 = np.abs(nb239_oof - y_tr)
    res_243 = np.abs(nb243_oof - y_tr)
    nb243_better = (res_243 < res_239).astype(int)
    print(f"nb243 better for {nb243_better.sum()}/{len(y_tr)} compounds")
    # Strong signal: nb243 better by >=0.3
    strong_diff = (res_239 - res_243) >= 0.3
    print(f"nb243 STRONGLY better (delta>=0.3) for {strong_diff.sum()} compounds")

    # Train binary classifier
    X_tr = combined(smiles_tr); X_tr = impute(X_tr)
    X_te = combined(smiles_te); X_te = impute(X_te)

    LGBM = dict(n_estimators=500, num_leaves=31, learning_rate=0.05, min_child_samples=20,
                objective="binary", n_jobs=4, random_state=42, verbose=-1)
    folds = scaffold_kfold_indices(tr["scaffold"].tolist(), n_splits=5)

    # Target: nb243 strongly better
    target = strong_diff.astype(int)
    print(f"Class balance: positive={target.sum()}, neg={len(target)-target.sum()}")

    oof_prob = np.zeros(len(y_tr))
    te_probs = []
    for ti, vi in folds:
        md = lgb.LGBMClassifier(**LGBM)
        # Class weight
        pos_w = (target == 0).sum() / max(target.sum(), 1)
        md.fit(X_tr[ti], target[ti], sample_weight=np.where(target[ti] == 1, pos_w, 1.0))
        oof_prob[vi] = md.predict_proba(X_tr[vi])[:, 1]
        te_probs.append(md.predict_proba(X_te)[:, 1])
    te_prob = np.mean(te_probs, axis=0)
    print(f"\nRouter OOF AUC-ish stats: high-prob top 10%: mean prob {np.percentile(oof_prob, 90):.3f}")

    # Hard route: if prob > threshold, use nb243; else nb239
    print("\n=== Router OOF performance ===")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        route_to_nb243 = oof_prob > thresh
        oof_routed = np.where(route_to_nb243, nb243_oof, nb239_oof)
        r = rae(y_tr, oof_routed)
        sign = " ***" if r < rae(y_tr, nb239_oof) else ""
        print(f"  thresh={thresh}: routed_to_243={route_to_nb243.sum()}, OOF={r:.4f}{sign}")

    # Soft route: blend by router probability
    print("\n=== Soft route (blend by router prob) ===")
    for max_w_243 in [0.1, 0.2, 0.3, 0.5]:
        w_243 = max_w_243 * oof_prob
        oof_soft = (1 - w_243) * nb239_oof + w_243 * nb243_oof
        r = rae(y_tr, oof_soft)
        sign = " ***" if r < rae(y_tr, nb239_oof) else ""
        print(f"  max_w_243={max_w_243}: OOF={r:.4f}{sign}")

    # Build best submission: use threshold 0.5 hard route on test
    print("\n=== Generate test submission ===")
    # Find best parameter on OOF
    best_r = rae(y_tr, nb239_oof)
    best_params = ("baseline", "none", 0)
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        route = oof_prob > thresh
        oof_t = np.where(route, nb243_oof, nb239_oof)
        r = rae(y_tr, oof_t)
        if r < best_r:
            best_r = r; best_params = ("hard", thresh, route.sum())
    for max_w in [0.1, 0.2, 0.3, 0.5]:
        w = max_w * oof_prob
        oof_s = (1 - w) * nb239_oof + w * nb243_oof
        r = rae(y_tr, oof_s)
        if r < best_r:
            best_r = r; best_params = ("soft", max_w, w.mean())
    print(f"Best: {best_params}, OOF={best_r:.4f}")

    # Apply to test
    if best_params[0] == "hard":
        thresh = best_params[1]
        te_route = te_prob > thresh
        te_routed = np.where(te_route, nb243_te, nb239_te)
        print(f"  Test routed_to_243: {te_route.sum()}/{len(te_route)}")
    elif best_params[0] == "soft":
        max_w = best_params[1]
        w = max_w * te_prob
        te_routed = (1 - w) * nb239_te + w * nb243_te
    else:
        te_routed = nb239_te

    print(f"  te_routed: mean={te_routed.mean():.3f}, std={te_routed.std():.3f}")
    np.save(DATA_PROCESSED / "oof_nb269_routed.npy", oof_t if best_params[0] == "hard" else oof_s)
    np.save(DATA_PROCESSED / "te_nb269_routed.npy", te_routed)

    sub = pd.DataFrame({"SMILES": [sm[n] for n in te_names], "Molecule Name": te_names, "pEC50": te_routed})
    sub.to_csv(SUBMISSIONS / "269_targeted_router.csv", index=False)
    print(f"Saved 269_targeted_router.csv  delta OOF: {best_r - rae(y_tr, nb239_oof):+.4f}")


if __name__ == "__main__":
    main()
