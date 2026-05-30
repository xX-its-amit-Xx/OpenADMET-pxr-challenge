"""nb268 -- Per-compound conditional ensemble.

User insight: aggregate OOF unfairly discounts models that excel on SPECIFIC
compound types. Different chemistries may favor different models.

Plan:
1. Compute per-compound residuals across our model pool (~30+ trusted OOFs)
2. For each compound, identify which model gives lowest residual
3. Build a CLASSIFIER that predicts the BEST model from compound features
4. At inference, route each test compound to its predicted-best model
5. Compare with uniform SLSQP

Bonus: cluster compounds by which model is best, examine common chemistry.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from rdkit import Chem

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED


def std_smi(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol) if mol else None
    except: return None


def main():
    print("=== nb268: Conditional/gated ensemble ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["SMILES"].apply(std_smi).tolist()

    # Collect trusted OOFs (skip leakage)
    P = Path("data/processed")
    LEAKAGE = {'oof_adaptive_delta_4tier.npy', 'oof_grand_v11.npy', 'oof_grand_v6.npy',
               'oof_grand_v8.npy', 'oof_grand_v9.npy', 'oof_grand_v10.npy',
               'oof_delta_ensemble_blend.npy', 'oof_delta_5tiers.npy',
               'oof_aux_features.npy', 'oof_creative_mega_ensemble.npy',
               'oof_full_desc_delta_3tier.npy', 'oof_allfp_delta_3tier.npy',
               'oof_blend_optimizer.npy', 'oof_enhanced_delta_3tier.npy',
               'oof_delta_similarity_tiers.npy', 'oof_nb116_quantile_width.npy',
               'oof_nb239_full_slsqp.npy', 'oof_nb240_safe_slsqp.npy',
               'oof_nb242_huber.npy', 'oof_nb241_7way.npy', 'oof_nb244_deep_greedy.npy',
               'oof_nb265_binary.npy', 'oof_nb265_5cls.npy', 'oof_nb265_10cls.npy',
               'oof_nb266_alpha100.npy', 'oof_nb266_alpha070.npy', 'oof_nb266_alpha050.npy', 'oof_nb266_alpha030.npy'}

    candidates = []
    for f in sorted(P.glob("oof_*.npy")):
        if f.name in LEAKAGE: continue
        base = f.stem[4:]
        for te_p in [P / f"te_{base}.npy", P / f"te_oof_{base}.npy"]:
            if te_p.exists():
                try:
                    oof = np.load(f); te = np.load(te_p)
                    if len(oof) == len(y_tr) and len(te) == 513 and np.isfinite(oof).all() and np.isfinite(te).all() and oof.std() > 0.3 and te.std() > 0.3 and te.std()/oof.std() > 0.40:
                        r = rae(y_tr, oof)
                        if r < 0.7:
                            candidates.append((base, oof, te, r))
                    break
                except: pass

    print(f"Loaded {len(candidates)} trusted candidates")
    names = [c[0] for c in candidates]
    M = np.column_stack([c[1] for c in candidates])  # (n_tr, n_models)
    T = np.column_stack([c[2] for c in candidates])  # (n_te, n_models)

    # Per-compound best model
    residuals = np.abs(M - y_tr[:, None])  # (n_tr, n_models)
    best_model_per_compound = np.argmin(residuals, axis=1)  # (n_tr,)
    print(f"\nBest-model distribution per compound (top 10):")
    counts = np.bincount(best_model_per_compound, minlength=len(names))
    top_idx = np.argsort(counts)[::-1]
    for i in top_idx[:10]:
        print(f"  {names[i]:35s}: {counts[i]} compounds, OOF RAE {candidates[i][3]:.4f}")

    # ORACLE: per-compound use the best model's prediction (upper bound)
    oracle = M[np.arange(len(y_tr)), best_model_per_compound]
    print(f"\nORACLE OOF RAE (per-compound best): {rae(y_tr, oracle):.4f}")
    print(f"nb239 OOF (current best 4-way SLSQP): 0.2838")
    print(f"\nOracle is OUR THEORETICAL CEILING if we could perfectly route compounds.")

    # Now: train a MODEL SELECTOR
    # For each compound, target = best_model_per_compound (multiclass)
    # Features = combined molecular features
    X_tr_feat = combined(smiles_tr); X_tr_feat = impute(X_tr_feat)
    X_te_feat = combined(smiles_te); X_te_feat = impute(X_te_feat)

    # But ~30+ classes is too many to learn from 4139 compounds.
    # Simplify: top 8 most-common best models become classes; rest mapped to "default"
    top_models = top_idx[:8]
    name_to_class = {i: c for c, i in enumerate(top_models)}
    # Map every compound: if its best is in top 8 → that class; else default class 8
    cls_target = np.array([name_to_class.get(b, 8) for b in best_model_per_compound])
    n_classes = 9
    print(f"\nClass distribution (top 8 + default): {np.bincount(cls_target)}")

    folds = scaffold_kfold_indices(tr["scaffold"].tolist(), n_splits=5)

    print("\nTraining model-selector classifier...")
    LGBM_CLS = dict(n_estimators=500, num_leaves=31, learning_rate=0.05, min_child_samples=20,
                    objective="multiclass", num_class=n_classes, n_jobs=4, random_state=42, verbose=-1)
    selector_oof = np.zeros((len(y_tr), n_classes))
    selector_te = []
    for ti, vi in folds:
        md = lgb.LGBMClassifier(**LGBM_CLS)
        md.fit(X_tr_feat[ti], cls_target[ti])
        selector_oof[vi] = md.predict_proba(X_tr_feat[vi])
        selector_te.append(md.predict_proba(X_te_feat))
    selector_te = np.mean(selector_te, axis=0)

    # Apply selector: weighted blend per compound
    # For each compound, blend = sum over top 8 models of (P[class] * model_pred)
    # Default class 8 = nb239 (best 4-way SLSQP)
    nb239_oof = np.load(DATA_PROCESSED / "oof_nb239_full_slsqp.npy")
    nb239_te = np.load(DATA_PROCESSED / "te_nb239_full_slsqp.npy")

    oof_gated = np.zeros(len(y_tr))
    te_gated = np.zeros(len(smiles_te))
    for c, model_idx in enumerate(top_models):
        oof_gated += selector_oof[:, c] * M[:, model_idx]
        te_gated += selector_te[:, c] * T[:, model_idx]
    # Default class 8 → nb239
    oof_gated += selector_oof[:, 8] * nb239_oof
    te_gated += selector_te[:, 8] * nb239_te
    print(f"\nGated ensemble OOF RAE: {rae(y_tr, oof_gated):.4f}")
    print(f"te: mean={te_gated.mean():.3f}, std={te_gated.std():.3f}")

    # Hybrid: blend gated with nb239
    print("\n--- Blend gated + nb239 ---")
    for w in [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        b = (1-w) * nb239_oof + w * oof_gated
        r = rae(y_tr, b)
        sign = " ***" if r < rae(y_tr, nb239_oof) else ""
        print(f"  w_gated={w}: OOF={r:.4f}{sign}")

    np.save(DATA_PROCESSED / "oof_nb268_gated.npy", oof_gated)
    np.save(DATA_PROCESSED / "te_nb268_gated.npy", te_gated)


if __name__ == "__main__":
    main()
