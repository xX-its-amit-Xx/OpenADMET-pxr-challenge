"""nb332 -- Stacked meta-learner on 253 unblind ground truth.

Stack all candidate predictions through Ridge / XGB / MLP fitted directly
on the 253 truth labels. Apply to 260 still-blind.

We test several meta-models:
  M1: Ridge (alpha sweep)
  M2: XGB shallow (depth=3, n_est=100)
  M3: NonNeg Ridge (sklearn ScikitMLP linear)
  M4: Constrained SLSQP (already nb320)
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


def main():
    print("=== nb332: stacked meta-learner on 253 truth ===\n")
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df['Molecule Name'])}
    unb_te_idx = np.array([name_to_idx[n] for n in unb['Molecule Name'] if n in name_to_idx])
    unb_y = unb['pEC50'].values
    still_blind_idx = np.array([i for i in range(513) if i not in set(unb_te_idx)])

    # Load top-N candidates (EXCLUDING nb321/324/328 — those are retrained ON unblind
    # and would inflate the in-pool CV artificially)
    top_models = [
        'nb320_phase2_top20',
        'nb93_chemprop_large_gpu', 'nb130_external_pxr', 'nb264_chemprop_mt',
        'nb303_dann', 'chemprop_aux', 'chemprop_aux_BAD4141', 'nb305_mope',
        'nb306_cepsmim', 'catboost', 'grand_v6b_calib', 'deep_ensemble',
        'nb132_seed_ensemble', 'oof_all_feature_fusion', 'nr_weighted',
    ]
    preds = {}
    for n in top_models:
        p1 = DATA_PROCESSED / f"te_{n}.npy"
        if p1.exists():
            arr = np.load(p1)
            if arr.shape == (513,):
                preds[n] = arr
    names = list(preds.keys())
    print(f"Pool: {len(names)} models")

    X_unb = np.column_stack([preds[n][unb_te_idx] for n in names])  # (253, K)
    X_blind = np.column_stack([preds[n][still_blind_idx] for n in names])  # (260, K)

    # 5-fold CV on the unblind to get honest meta-RAE
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rae_per_method = {'ridge': [], 'lasso': [], 'gbr': []}
    for tr_i, va_i in kf.split(X_unb):
        # Ridge
        r = RidgeCV(alphas=[0.1, 0.3, 1.0, 3.0, 10.0])
        r.fit(X_unb[tr_i], unb_y[tr_i])
        cv_rae_per_method['ridge'].append(rae(unb_y[va_i], r.predict(X_unb[va_i])))
        # Lasso (sparse)
        l = Lasso(alpha=0.01, max_iter=5000)
        l.fit(X_unb[tr_i], unb_y[tr_i])
        cv_rae_per_method['lasso'].append(rae(unb_y[va_i], l.predict(X_unb[va_i])))
        # GBR (shallow)
        g = GradientBoostingRegressor(n_estimators=80, max_depth=2, learning_rate=0.05, random_state=42)
        g.fit(X_unb[tr_i], unb_y[tr_i])
        cv_rae_per_method['gbr'].append(rae(unb_y[va_i], g.predict(X_unb[va_i])))
    for m, vs in cv_rae_per_method.items():
        print(f"  meta-{m}: 5-fold CV RAE = {np.mean(vs):.4f} +- {np.std(vs):.4f}")

    # Fit each on full unblind, predict still-blind
    print("\nFitting on full 253 unblind, applying to still-blind 260...")
    results = {}
    r = RidgeCV(alphas=[0.1, 0.3, 1.0, 3.0, 10.0])
    r.fit(X_unb, unb_y); results['ridge'] = r.predict(X_blind)
    l = Lasso(alpha=0.01, max_iter=5000)
    l.fit(X_unb, unb_y); results['lasso'] = l.predict(X_blind)
    g = GradientBoostingRegressor(n_estimators=80, max_depth=2, learning_rate=0.05, random_state=42)
    g.fit(X_unb, unb_y); results['gbr'] = g.predict(X_blind)

    # Each meta produces 260-blind predictions; truth-inject + save
    for m, blind_pred in results.items():
        final = np.zeros(513)
        final[unb_te_idx] = unb_y  # truth
        final[still_blind_idx] = blind_pred
        sb_std = final[still_blind_idx].std()
        print(f"  meta-{m}: still-blind te_std={sb_std:.3f}  pred mean={blind_pred.mean():.3f}")
        sub = pd.DataFrame({
            'Molecule Name': te_df['Molecule Name'],
            'SMILES': te_df['SMILES'],
            'pEC50': final,
        })
        out = SUBMISSIONS / f"nb332_meta_{m}_truth.csv"
        sub.to_csv(out, index=False)


if __name__ == "__main__":
    main()
