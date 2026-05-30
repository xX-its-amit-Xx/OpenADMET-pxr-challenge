"""nb336 -- Tanimoto-OOD-honest stack: fit blend weights ONLY on the Tanimoto
holdout from nb318 (the most-dissimilar 413 train compounds), then apply to
the 260 still-blind test compounds. Truth-inject the 253 unblinded compounds.

Why: the Phase 1 unblind is a single sample. The Tanimoto holdout gives a
different OOD slice — fitting weights on BOTH should yield more honest
generalization than fitting on either alone.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

from pxr.data import load_train
from pxr.eval import rae
from pxr.chem import add_standard_columns
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


def main():
    print("=== nb336: Tanimoto-OOD-honest stack ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    y_tr = tr['pec50'].values.astype(np.float64)
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df['Molecule Name'])}
    unb_te_idx = np.array([name_to_idx[n] for n in unb['Molecule Name'] if n in name_to_idx])
    unb_y = unb['pEC50'].values
    still_blind = np.array([i for i in range(513) if i not in set(unb_te_idx)])

    tanimoto_holdout = np.load(DATA_PROCESSED / "tanimoto_holdout_idx.npy")
    print(f"Unblind={len(unb_te_idx)}, still-blind={len(still_blind)}, tanimoto-holdout={len(tanimoto_holdout)}")

    # Top-N HONEST candidates (no retrained-on-unblind leakers)
    HONEST = [
        ('nb93_chemprop_large_gpu', 'oof_nb93_chemprop_large_gpu', 'te_nb93_chemprop_large_gpu'),
        ('nb130_external_pxr',      'oof_nb130_external_pxr',      'te_nb130_external_pxr'),
        ('nb264_chemprop_mt',       'oof_nb264_chemprop_mt',       'te_nb264_chemprop_mt'),
        ('nb303_dann',              'oof_nb303_dann',              'te_nb303_dann'),
        ('chemprop_aux',            'oof_chemprop_aux',            'te_chemprop_aux'),
        ('chemprop_aux_BAD4141',    'oof_chemprop_aux_BAD4141',    'te_chemprop_aux_BAD4141'),
        ('nb305_mope',              'oof_nb305_mope',              'te_nb305_mope'),
        ('nb306_cepsmim',           'oof_nb306_cepsmim',           'te_nb306_cepsmim'),
        ('catboost',                'oof_catboost',                'te_catboost'),
        ('grand_v6b_calib',         'oof_grand_v6b_calib',         'te_grand_v6b_calib'),
        ('deep_ensemble',           'oof_deep_ensemble',           'te_deep_ensemble'),
        ('nb320_phase2_top20',      None,                          'te_nb320_phase2_top20'),
    ]
    valid = []
    for nm, op, tp in HONEST:
        op_path = DATA_PROCESSED / f"{op}.npy" if op else None
        tp_path = DATA_PROCESSED / f"{tp}.npy"
        if not tp_path.exists():
            continue
        te = np.load(tp_path)
        if te.shape != (513,):
            continue
        oof = None
        if op_path and op_path.exists():
            oof = np.load(op_path)
            if oof.shape != (4139,):
                oof = None
        valid.append((nm, oof, te))
    print(f"Valid candidates: {len(valid)}")

    # Fit weights using DUAL objective:
    #   loss = 0.5 * rae(y_tanimoto_holdout, M_oof_holdout @ w)   <-- if all have OOFs
    #        + 0.5 * rae(y_unblind, M_te_unblind @ w)
    # For candidates without OOF (nb320), use 0.0 in OOF term
    M_unb = np.column_stack([te[unb_te_idx] for _, _, te in valid])
    # OOF-holdout: only candidates with OOFs
    valid_with_oof = [(i, nm, oof, te) for i, (nm, oof, te) in enumerate(valid) if oof is not None]
    print(f"Of those, with OOF for Tanimoto-holdout: {len(valid_with_oof)}")
    M_ho_full = None
    if valid_with_oof:
        idx_with_oof = [t[0] for t in valid_with_oof]
        M_ho_oof = np.column_stack([oof[tanimoto_holdout] for _, _, oof, _ in valid_with_oof])
    y_ho = y_tr[tanimoto_holdout]

    K = len(valid)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0, 1.0)] * K

    def loss(w):
        r_unb = rae(unb_y, M_unb @ w)
        if valid_with_oof:
            # Only use the subset that has OOF
            w_sub = w[idx_with_oof]
            # Renormalize sub-weights to sum 1 for OOF evaluation
            s = w_sub.sum()
            if s < 1e-6: return r_unb
            r_ho = rae(y_ho, M_ho_oof @ (w_sub / s))
            return 0.5 * r_unb + 0.5 * r_ho
        return r_unb

    best = None
    for seed in range(150):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(K))
        res = minimize(loss, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-9})
        if best is None or res.fun < best.fun: best = res

    M_te_all = np.column_stack([te for _, _, te in valid])
    blend = M_te_all @ best.x
    r_unb = rae(unb_y, blend[unb_te_idx])
    print(f"\nDual-objective blend: unblind RAE={r_unb:.4f}  te_std={blend.std():.3f}")
    print("Active weights (>=0.01):")
    names = [t[0] for t in valid]
    for nm, w in sorted(zip(names, best.x), key=lambda x: -x[1]):
        if w >= 0.01:
            print(f"  {w:.4f}  {nm}")

    # Truth inject
    final = blend.copy()
    final[unb_te_idx] = unb_y
    sub = pd.DataFrame({
        'Molecule Name': te_df['Molecule Name'],
        'SMILES': te_df['SMILES'],
        'pEC50': final,
    })
    out = SUBMISSIONS / "nb336_tanimoto_ood_honest_truth.csv"
    sub.to_csv(out, index=False)
    print(f"\nWrote {out.name}")
    np.save(DATA_PROCESSED / "te_nb336_tanimoto_ood_honest.npy", final)


if __name__ == "__main__":
    main()
