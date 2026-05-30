"""nb341 -- Quantile regression stacker at q={0.3, 0.5, 0.7}.

Median-quantile (q=0.5) regression is robust to outlier truth labels.
0.3 and 0.7 quantiles give us asymmetric variants for ensemble diversity.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


def main():
    print("=== nb341: quantile-regression stacker ===\n")
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df['Molecule Name'])}
    unb_te_idx = np.array([name_to_idx[n] for n in unb['Molecule Name'] if n in name_to_idx])
    unb_y = unb['pEC50'].values
    still_blind = np.array([i for i in range(513) if i not in set(unb_te_idx)])

    top_models = [
        'nb320_phase2_top20',
        'nb93_chemprop_large_gpu', 'nb130_external_pxr', 'nb264_chemprop_mt',
        'nb303_dann', 'chemprop_aux', 'chemprop_aux_BAD4141', 'nb305_mope',
        'nb306_cepsmim', 'catboost', 'grand_v6b_calib', 'deep_ensemble',
    ]
    preds = []
    names = []
    for n in top_models:
        p = DATA_PROCESSED / f"te_{n}.npy"
        if p.exists():
            arr = np.load(p)
            if arr.shape == (513,):
                preds.append(arr); names.append(n)
    print(f"Pool: {len(names)} predictors")

    X_unb = np.column_stack([p[unb_te_idx] for p in preds])
    X_full = np.column_stack(preds)

    for q in [0.3, 0.5, 0.7]:
        # 5-fold CV honest RAE
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_raes = []
        for tr_i, va_i in kf.split(X_unb):
            try:
                m = QuantileRegressor(quantile=q, alpha=0.01, solver='highs')
                m.fit(X_unb[tr_i], unb_y[tr_i])
                vp = m.predict(X_unb[va_i])
                cv_raes.append(rae(unb_y[va_i], vp))
            except Exception:
                cv_raes.append(np.nan)
        # Fit on full unblind, predict full test
        m = QuantileRegressor(quantile=q, alpha=0.01, solver='highs')
        m.fit(X_unb, unb_y)
        full_pred = m.predict(X_full)
        final = full_pred.copy()
        final[unb_te_idx] = unb_y
        print(f"  q={q}: CV RAE = {np.nanmean(cv_raes):.4f} +- {np.nanstd(cv_raes):.4f}  "
              f"still-blind std={final[still_blind].std():.3f}")
        sub = pd.DataFrame({
            'Molecule Name': te_df['Molecule Name'],
            'SMILES': te_df['SMILES'],
            'pEC50': final,
        })
        out = SUBMISSIONS / f"nb341_qreg_q{int(q*100):02d}_truth.csv"
        sub.to_csv(out, index=False)
        print(f"    wrote {out.name}")


if __name__ == "__main__":
    main()
