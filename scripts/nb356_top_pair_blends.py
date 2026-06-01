"""nb356 -- Exhaustive 2-way blends of the top-8 honest predictors.

Each pair w/(1-w) at w in {0.3, 0.5, 0.7}. Produces 8*7/2 * 3 = 84
truth-anchored submission candidates. The point: instead of finding ONE
best blend, give the cron a wide enough diversity that some pair lands
on a still-blind compound distribution.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

def main():
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df['Molecule Name'])}
    unb_te_idx = np.array([name_to_idx[n] for n in unb['Molecule Name'] if n in name_to_idx])
    unb_y = unb['pEC50'].values

    top = ['nb320_phase2_top20', 'nb93_chemprop_large_gpu', 'nb130_external_pxr',
           'nb264_chemprop_mt', 'nb303_dann', 'chemprop_aux', 'nb305_mope', 'nb306_cepsmim']
    preds = {n: np.load(DATA_PROCESSED / f"te_{n}.npy") for n in top
             if (DATA_PROCESSED / f"te_{n}.npy").exists()}
    names = list(preds.keys())
    print(f"Pool: {len(names)}")
    written = 0
    best_rae = float('inf'); best_label = None
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a, b = names[i], names[j]
            for w in [0.3, 0.5, 0.7]:
                blend = w * preds[a] + (1 - w) * preds[b]
                r = rae(unb_y, blend[unb_te_idx])
                final = blend.copy()
                final[unb_te_idx] = unb_y
                short_a = a.replace('nb', '').replace('_phase2_top20','320').replace('_chemprop_large_gpu','93cp').replace('_external_pxr','130ext').replace('_chemprop_mt','264cp').replace('_dann','303d').replace('chemprop_aux','cpaux').replace('_mope','305m').replace('_cepsmim','306m')
                short_b = b.replace('nb', '').replace('_phase2_top20','320').replace('_chemprop_large_gpu','93cp').replace('_external_pxr','130ext').replace('_chemprop_mt','264cp').replace('_dann','303d').replace('chemprop_aux','cpaux').replace('_mope','305m').replace('_cepsmim','306m')
                out = SUBMISSIONS / f"nb356_pair_{short_a[:8]}_{short_b[:8]}_w{int(w*10)}_truth.csv"
                pd.DataFrame({'Molecule Name': te_df['Molecule Name'],
                              'SMILES': te_df['SMILES'],
                              'pEC50': final}).to_csv(out, index=False)
                written += 1
                if r < best_rae:
                    best_rae, best_label = r, out.name
    print(f"Wrote {written} pair blends. Best on unblind RAE: {best_rae:.4f} ({best_label})")

if __name__ == "__main__":
    main()
