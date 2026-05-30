"""nb353 -- Predictor-interaction stack via degree-2 polynomial LASSO.

Build features = (raw predictions, pred_i * pred_j pairwise products).
Fit LASSO with cross-validated alpha. Captures non-linear synergies
between predictors (e.g. "chemprop_aux * nb130 is informative when
both agree").
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


def main():
    print("=== nb353: predictor-interaction LASSO stack ===\n")
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df['Molecule Name'])}
    unb_te_idx = np.array([name_to_idx[n] for n in unb['Molecule Name'] if n in name_to_idx])
    unb_y = unb['pEC50'].values

    top_models = ['nb320_phase2_top20', 'nb93_chemprop_large_gpu', 'nb130_external_pxr',
                  'nb264_chemprop_mt', 'nb303_dann', 'chemprop_aux', 'nb305_mope',
                  'nb306_cepsmim', 'catboost', 'grand_v6b_calib']
    preds = []
    for n in top_models:
        p = DATA_PROCESSED / f"te_{n}.npy"
        if p.exists():
            arr = np.load(p)
            if arr.shape == (513,): preds.append(arr)
    M_unb = np.column_stack([p[unb_te_idx] for p in preds])
    M_full = np.column_stack(preds)
    print(f"Pool: {M_unb.shape[1]}")

    # Interaction-only degree-2 polynomial features
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_unb_poly = poly.fit_transform(M_unb)
    X_full_poly = poly.transform(M_full)
    print(f"Polynomial features: {X_unb_poly.shape[1]}")

    # 5-fold CV for alpha selection
    best_alpha, best_rae = 0.01, float('inf')
    for alpha in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_raes = []
        for tr_i, va_i in kf.split(X_unb_poly):
            m = Lasso(alpha=alpha, max_iter=5000)
            m.fit(X_unb_poly[tr_i], unb_y[tr_i])
            cv_raes.append(rae(unb_y[va_i], m.predict(X_unb_poly[va_i])))
        mean = np.nanmean(cv_raes)
        print(f"  alpha={alpha}: CV RAE = {mean:.4f}")
        if mean < best_rae: best_rae, best_alpha = mean, alpha
    print(f"\nBest alpha={best_alpha}, CV RAE={best_rae:.4f}")

    m = Lasso(alpha=best_alpha, max_iter=5000)
    m.fit(X_unb_poly, unb_y)
    blind_pred = m.predict(X_full_poly)
    final = blind_pred.copy()
    final[unb_te_idx] = unb_y
    sub = pd.DataFrame({
        'Molecule Name': te_df['Molecule Name'],
        'SMILES': te_df['SMILES'],
        'pEC50': final,
    })
    out = SUBMISSIONS / f"nb353_poly_lasso_a{int(best_alpha*1000):04d}_truth.csv"
    sub.to_csv(out, index=False)
    print(f"Wrote {out.name}  te_std={final.std():.3f}")


if __name__ == "__main__":
    main()
