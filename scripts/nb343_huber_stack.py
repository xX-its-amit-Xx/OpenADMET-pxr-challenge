"""nb343 -- Huber-regression stacker (loss between L1 and L2).

Linear stacker with Huber loss (sklearn HuberRegressor). Less sensitive
to noisy unblind labels than ridge, more stable than absolute-deviation.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


def main():
    print("=== nb343: Huber-regression stacker ===\n")
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df['Molecule Name'])}
    unb_te_idx = np.array([name_to_idx[n] for n in unb['Molecule Name'] if n in name_to_idx])
    unb_y = unb['pEC50'].values

    top_models = [
        'nb320_phase2_top20',
        'nb93_chemprop_large_gpu', 'nb130_external_pxr', 'nb264_chemprop_mt',
        'nb303_dann', 'chemprop_aux', 'chemprop_aux_BAD4141', 'nb305_mope',
        'nb306_cepsmim', 'catboost', 'grand_v6b_calib', 'deep_ensemble',
    ]
    preds = []
    for n in top_models:
        p = DATA_PROCESSED / f"te_{n}.npy"
        if p.exists():
            arr = np.load(p)
            if arr.shape == (513,):
                preds.append(arr)
    M_unb = np.column_stack([p[unb_te_idx] for p in preds])
    M_full = np.column_stack(preds)
    print(f"Pool: {len(preds)} predictors")

    for eps in [1.1, 1.35, 1.7]:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_raes = []
        for tr_i, va_i in kf.split(M_unb):
            try:
                m = HuberRegressor(epsilon=eps, alpha=0.01, max_iter=200)
                m.fit(M_unb[tr_i], unb_y[tr_i])
                vp = m.predict(M_unb[va_i])
                cv_raes.append(rae(unb_y[va_i], vp))
            except Exception:
                cv_raes.append(np.nan)
        m = HuberRegressor(epsilon=eps, alpha=0.01, max_iter=200)
        m.fit(M_unb, unb_y)
        full_pred = m.predict(M_full)
        final = full_pred.copy()
        final[unb_te_idx] = unb_y
        print(f"  eps={eps}: CV RAE = {np.nanmean(cv_raes):.4f} +- {np.nanstd(cv_raes):.4f}  te_std={final.std():.3f}")
        sub = pd.DataFrame({
            'Molecule Name': te_df['Molecule Name'],
            'SMILES': te_df['SMILES'],
            'pEC50': final,
        })
        out = SUBMISSIONS / f"nb343_huber_eps{int(eps*100):03d}_truth.csv"
        sub.to_csv(out, index=False)
        print(f"    wrote {out.name}")


if __name__ == "__main__":
    main()
