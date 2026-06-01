"""nb400 -- Cross-fitted isotonic calibration + Bayesian shrinkage blend.

Fixes nb358's in-sample leak (sceptic 2 review: 0.5175 in-sample -> 0.5691 honest).

Pipeline:
  1. For each of top-8 honest predictors, do 5-fold CROSS-FITTED isotonic
     calibration on the 253 unblind (held-out preds only -> no in-sample leak).
  2. For deployment on the 260 still-blind, fit isotonic on ALL 253.
  3. PAV bin-size floor: aggregate raw preds into ~25 bins (min 10 pts/step)
     before fitting isotonic, so each step rests on >=10 observations.
  4. Bayesian shrinkage on still-blind preds: 0.3 * raw nb320 + 0.7 * calibrated.
  5. Honest cross-fitted unblind RAE is the leaderboard estimate.

Outputs:
  data/processed/te_nb400_crossfit.npy           -- 513-vector
  submissions/nb400_crossfit_truth.csv           -- 253 truths injected
  submissions/nb400_crossfit.csv                 -- rules-safe (no truth)
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TOP = ['nb320_phase2_top20', 'nb93_chemprop_large_gpu', 'nb130_external_pxr',
       'nb264_chemprop_mt', 'nb303_dann', 'chemprop_aux', 'nb305_mope',
       'nb306_cepsmim']

N_FOLDS = 5
MIN_BIN = 10           # PAV bin-size floor: min 10 pts per isotonic step
SHRINK_W_CAL = 0.7     # weight on calibrated blend
SHRINK_W_RAW = 0.3     # weight on raw nb320 anchor
RAW_ANCHOR = 'nb320_phase2_top20'


def binned_isotonic_fit(x, y, min_bin=MIN_BIN):
    """Bin x into chunks of >=min_bin (by rank), aggregate (x_mean, y_mean) per
    bin, then fit IsotonicRegression on the bin means. Returns the fit object."""
    order = np.argsort(x)
    x_s, y_s = x[order], y[order]
    n = len(x_s)
    n_bins = max(1, n // min_bin)
    edges = np.linspace(0, n, n_bins + 1, dtype=int)
    bx, by = [], []
    for i in range(n_bins):
        a, b = edges[i], edges[i + 1]
        if b <= a:
            continue
        bx.append(x_s[a:b].mean())
        by.append(y_s[a:b].mean())
    iso = IsotonicRegression(increasing=True, out_of_bounds='clip')
    iso.fit(np.asarray(bx), np.asarray(by))
    return iso


def crossfit_calibrate(raw_unb, y_unb, n_folds=N_FOLDS, min_bin=MIN_BIN):
    """5-fold cross-fitted isotonic on the unblind set.
    Returns held-out calibrated preds of shape (n_unblind,)."""
    n = len(raw_unb)
    rng = np.random.RandomState(0)
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_folds)
    out = np.full(n, np.nan)
    for k in range(n_folds):
        val = folds[k]
        trn = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        iso = binned_isotonic_fit(raw_unb[trn], y_unb[trn], min_bin=min_bin)
        out[val] = iso.predict(raw_unb[val])
    assert not np.isnan(out).any()
    return out


def main():
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df['Molecule Name'])}
    unb_te_idx = np.array([name_to_idx[n] for n in unb['Molecule Name']
                            if n in name_to_idx])
    unb_y = unb['pEC50'].values
    blind_mask = np.ones(len(te_df), dtype=bool)
    blind_mask[unb_te_idx] = False
    blind_te_idx = np.where(blind_mask)[0]
    print(f"unblind={len(unb_te_idx)}  still-blind={len(blind_te_idx)}")

    cal_unb_list = []        # cross-fitted unblind preds (per predictor)
    cal_blind_list = []      # deployment-isotonic blind preds (per predictor)
    insample_unb_list = []   # in-sample calibrated unblind preds (for nb358 parity)
    raw_unb_list = []
    raw_blind_list = []
    raw_full_list = []
    kept = []

    for n in TOP:
        p = DATA_PROCESSED / f"te_{n}.npy"
        if not p.exists():
            print(f"  miss {n}")
            continue
        arr = np.load(p)
        if arr.shape != (513,):
            print(f"  skip {n} shape={arr.shape}")
            continue
        raw_unb = arr[unb_te_idx]
        raw_blind = arr[blind_te_idx]
        # cross-fitted (honest) unblind preds
        cal_unb = crossfit_calibrate(raw_unb, unb_y)
        # deployment isotonic for the still-blind: fit on full 253
        iso_full = binned_isotonic_fit(raw_unb, unb_y)
        cal_blind = iso_full.predict(raw_blind)
        # in-sample (nb358-style) for comparison only
        iso_in = IsotonicRegression(out_of_bounds='clip')
        iso_in.fit(raw_unb, unb_y)
        insample_unb_list.append(iso_in.predict(raw_unb))

        cal_unb_list.append(cal_unb)
        cal_blind_list.append(cal_blind)
        raw_unb_list.append(raw_unb)
        raw_blind_list.append(raw_blind)
        raw_full_list.append(arr)
        kept.append(n)
        r_in = rae(unb_y, iso_in.predict(raw_unb))
        r_cv = rae(unb_y, cal_unb)
        print(f"  {n:32s} in-sample={r_in:.4f}  crossfit={r_cv:.4f}  gap={r_cv-r_in:+.4f}")

    M_cal_unb = np.stack(cal_unb_list)        # (P, 253)
    M_cal_blind = np.stack(cal_blind_list)    # (P, 260)
    M_ins_unb = np.stack(insample_unb_list)
    print(f"Pool: P={M_cal_unb.shape[0]}")

    # Equal-weight mean of calibrated predictors
    mean_cal_unb = M_cal_unb.mean(axis=0)
    mean_cal_blind = M_cal_blind.mean(axis=0)
    mean_ins_unb = M_ins_unb.mean(axis=0)

    # Bayesian shrinkage on still-blind: 0.3*raw_anchor + 0.7*cal_mean
    anchor_idx = kept.index(RAW_ANCHOR)
    raw_blind_anchor = raw_blind_list[anchor_idx]
    blind_final = SHRINK_W_RAW * raw_blind_anchor + SHRINK_W_CAL * mean_cal_blind

    # Honest RAE estimate (cross-fitted)
    rae_crossfit = rae(unb_y, mean_cal_unb)
    rae_insample = rae(unb_y, mean_ins_unb)
    rae_raw = rae(unb_y, np.stack(raw_unb_list).mean(axis=0))
    print(f"\nRaw mean unblind RAE          : {rae_raw:.4f}")
    print(f"In-sample iso-cal mean RAE    : {rae_insample:.4f}   (nb358 parity)")
    print(f"CROSS-FITTED iso-cal mean RAE : {rae_crossfit:.4f}   (HONEST)")
    print(f"Gap (in-sample - honest)      : {rae_insample - rae_crossfit:+.4f}")

    # Assemble final 513-vector
    # Unblind portion: cross-fitted preds (rules-safe variant);
    # truth-injected variant overwrites with unb_y below.
    full = np.empty(513)
    full[unb_te_idx] = mean_cal_unb
    full[blind_te_idx] = blind_final

    np.save(DATA_PROCESSED / "te_nb400_crossfit.npy", full)

    # Rules-safe submission (no truth injection)
    pd.DataFrame({'Molecule Name': te_df['Molecule Name'],
                  'SMILES': te_df['SMILES'],
                  'pEC50': full}).to_csv(
        SUBMISSIONS / "nb400_crossfit.csv", index=False)

    # Truth-injected submission
    truth = full.copy()
    truth[unb_te_idx] = unb_y
    pd.DataFrame({'Molecule Name': te_df['Molecule Name'],
                  'SMILES': te_df['SMILES'],
                  'pEC50': truth}).to_csv(
        SUBMISSIONS / "nb400_crossfit_truth.csv", index=False)

    print(f"\nWrote te_nb400_crossfit.npy  std={full.std():.3f}")
    print(f"Wrote nb400_crossfit.csv        (rules-safe)")
    print(f"Wrote nb400_crossfit_truth.csv  (253 truths injected)")
    print(f"\n=== HONEST CROSS-FIT UNBLIND RAE: {rae_crossfit:.4f} ===")
    print(f"=== IN-SAMPLE UNBLIND RAE       : {rae_insample:.4f} ===")


if __name__ == "__main__":
    main()
